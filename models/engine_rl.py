import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from models.model import Net_timesnet_sample_onetimesnet
from models.model import PolicyNet
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLatentTensorGate2d
from utils import mdtp
from torch.distributions import Categorical
import time

class TrainerMAB:
    def __init__(self, batch_size, candidate_lengths, node_num, in_features, out_features,
                 lstm_features, device, learning_rate, weight_decay, gradient_clip,
                 smoe_start_epoch, pred_size):
        self.device = device
        self.batch_size = batch_size
        self.candidate_lengths = candidate_lengths
        self.gradient_clip = gradient_clip
        self.pred_size = pred_size

        self.smoe_config_map = {
            l: SpatialMoEConfig(
                in_planes=2,
                out_planes=3,
                num_experts=6,
                gate_block=functools.partial(SpatialLatentTensorGate2d, node_num=node_num),
                save_error_signal=True,
                dampen_expert_error=True,
                unweighted=True,
                block_gate_grad=True,
                routing_error_quantile=0.7,
                pred_size=pred_size,
                windows_size=l
            ) for l in candidate_lengths
        }

        # Build a multi-scale model (freeze parameters)
        self.models = {}
        for l in candidate_lengths:
            model = Net_timesnet_sample_onetimesnet(
                1, l, node_num, in_features, out_features,
                lstm_features, smoe_config=self.smoe_config_map[l], pred_size=pred_size
            ).to(device)

            model_path = f'./checkpoints/exp_nyc_{l}/best_model.pth'
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()  
            for param in model.parameters():
                param.requires_grad = False
            self.models[l] = model

        # Policy network (responsible for selecting window size）
        self.h = None
        self.policy_net = PolicyNet(
            input_dim=node_num * in_features * 2,  # bike + taxi
            hidden_dim=32,
            num_arms=len(candidate_lengths),
            pred_size=pred_size,
            num_layers=2
        ).to(device)

        self.cache = torch.zeros(
            (pred_size, pred_size, node_num, 4), dtype=torch.float32, device=device
        )
        self.count = 0

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = mdtp.rmse

        self.actions = []
        self.time = 0.0

    def update_cache(self, prediction):
        """
        prediction: Tensor of shape (pred_t, node, 4)
        """
        # Scroll one grid to the lower right (i.e. previous round → next round)
        self.cache = torch.roll(self.cache, shifts=(-1, -1), dims=(0, 1))

        # The last line writes the new prediction
        self.cache[-1] = prediction

    def fuse_t1_prediction(self, weight):
        """
        Fusion of all rounds of prediction results for t+1 (self.cache[:, 0]) weighted output

        Args:
            weight: Tensor of shape (pred_t,)

        Returns:
            fused_pred: Tensor of shape (node, 4)
        """
        assert self.cache.dim() == 4  # (pred_t, pred_t, node, 4)
        assert weight.shape[0] == self.cache.shape[0], "weight length must match pred_t"

        pred_t1_all = self.cache[:, 0, :, :]  # (pred_t, node, 4)

        # fusion
        fused_pred = (weight[:, None, None] * pred_t1_all).sum(dim=0)  # → (node, 4)

        return fused_pred
    

    def train(self, train_x, train_y):
        self.policy_net.train()
        bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in train_x]
        bike_y, taxi_y = [y.to(self.device) for y in train_y]

        # === poilcy net input =====================================================
        bike_feat = bike_in[:, -1].reshape(self.batch_size, -1)
        taxi_feat = taxi_in[:, -1].reshape(self.batch_size, -1)
        state     = torch.cat([bike_feat, taxi_feat], dim=1)    # (B, D)

        if self.h is not None:
            self.h = tuple([t.detach() for t in self.h])

        log_probs_tot, all_rewards, weight_losses, entropies, actions = [], [], [], [], []

        for i in range(self.batch_size):
            x = state[i].unsqueeze(0).unsqueeze(0)              # (1,1,D)
            action_probs, weight_table, self.h = self.policy_net(x, self.h)

            # --- sample action ---
            m            = Categorical(action_probs)
            action       = m.sample()       # Select windowsize
            logp_action  = m.log_prob(action)

            chosen_len   = self.candidate_lengths[action.item()]
            ref_len      = max(self.candidate_lengths)

            # --- sample weight ---
            weights = weight_table[action]

            # --- predict ---
            bike_in_i = bike_in[i:i+1, -chosen_len:]
            bike_adj_i = bike_adj[i:i+1, -chosen_len:]
            taxi_in_i = taxi_in[i:i+1, -chosen_len:]
            taxi_adj_i = taxi_adj[i:i+1, -chosen_len:]

            with torch.no_grad():
                pred      = self.models[chosen_len]((bike_in_i, bike_adj_i,
                                                     taxi_in_i, taxi_adj_i))# 4 * (1,P,N)
                pred_ref  = self.models[ref_len]((
                              bike_in[i:i+1, -ref_len:], bike_adj[i:i+1, -ref_len:],
                              taxi_in[i:i+1, -ref_len:], taxi_adj[i:i+1, -ref_len:])) # 4 * (1,P,N)

            
            pred = torch.stack([p.squeeze(0) for p in pred], dim=-1)  # (P, N, 4)          
            self.update_cache(pred)  # Update cache
            if self.count < 5:
                weights = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
                self.count += 1
            fused_pred = self.fuse_t1_prediction(weights.squeeze(0))  # Fusion t+1 predictions
            pred_ref_t1 = torch.stack([p.squeeze(0)[0] for p in pred_ref], dim=-1)  # shape: (4, N)

            # === Get the corresponding ground truth: (N,)
            gt_bike_start = bike_y[0, i, 0]
            gt_bike_end   = bike_y[1, i, 0]
            gt_taxi_start = taxi_y[0, i, 0]
            gt_taxi_end   = taxi_y[1, i, 0]

            # === loss ===
            loss_act =  (self.loss_fn(pred[0,:,0], gt_bike_start) + \
                        self.loss_fn(pred[0,:,1], gt_bike_end)) * 2  + \
                        self.loss_fn(pred[0,:,2], gt_taxi_start) + \
                        self.loss_fn(pred[0,:,3], gt_taxi_end) 

            loss_fused = self.loss_fn(fused_pred[:,0], gt_bike_start) + \
                        self.loss_fn(fused_pred[:,1], gt_bike_end)   + \
                        self.loss_fn(fused_pred[:,2], gt_taxi_start) + \
                        self.loss_fn(fused_pred[:,3], gt_taxi_end)

            loss_ref = (self.loss_fn(pred_ref_t1[:,0], gt_bike_start) + \
                    self.loss_fn(pred_ref_t1[:,1], gt_bike_end)) * 2  + \
                    self.loss_fn(pred_ref_t1[:,2], gt_taxi_start) + \
                    self.loss_fn(pred_ref_t1[:,3], gt_taxi_end)

            reward = (loss_ref - loss_act).detach()          
            reward = reward * 1e3
            all_rewards.append(reward)
            log_probs_tot.append(logp_action)
            weight_losses.append(loss_fused)
            actions.append(action.item())

            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1)  # (B,)
            entropies.append(entropy)   

        # === REINFORCE loss ================================================
        log_probs_tot = torch.stack(log_probs_tot).view(-1)
        rewards       = torch.stack(all_rewards).view(-1)
        weight_losses = torch.stack(weight_losses).mean() 
        entropy_bonus = torch.stack(entropies).view(-1).mean()  


        # baseline = rewards.median().detach()  # 
        # advantages = rewards - baseline
        # loss_rl = -(log_probs_tot * advantages).mean()·
        # print(actions)

        # rewards = torch.sign(rewards)
        loss_rl = -(log_probs_tot * rewards).mean() 
        loss_rl -= entropy_bonus * 0.1   
        total_loss = loss_rl + weight_losses
        # total_loss = loss_rl

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        return total_loss.item(), rewards.mean().item()

    # def test_online(self, test_x, test_y):
    #     self.policy_net.eval()
    #     bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in test_x]
    #     bike_y, taxi_y = [y.to(self.device) for y in test_y]

    #     bike_feat = bike_in[:, -1].reshape(1, -1)
    #     taxi_feat = taxi_in[:, -1].reshape(1, -1)
    #     state = torch.cat([bike_feat, taxi_feat], dim=1).unsqueeze(0)  # shape: (1, 1, D)

    #     t1 = time.time()
    #     probs, weight_table, self.h = self.policy_net(state, self.h)
    #     t2 = time.time()
    #     self.count += 1
    #     if self.count > 1:
    #         self.time += (t2 - t1)

    #     print(probs)
    #     action = torch.argmax(probs, dim=-1).item()
    #     print(action)
    #     self.actions.append(action)
    #     chosen_len = self.candidate_lengths[action]
    #     weights = weight_table[action]
    #     print(weights)
    #     # chosen_len = 24
    #     bike_in_i = bike_in[:, -chosen_len:, :, :]
    #     bike_adj_i = bike_adj[:, -chosen_len:, :, :]
    #     taxi_in_i = taxi_in[:, -chosen_len:, :, :]
    #     taxi_adj_i = taxi_adj[:, -chosen_len:, :, :]

    #     t1 = time.time()
    #     pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i))
    #     t2 = time.time()
    #     if self.count > 1:
    #         self.time += (t2 - t1)
    #     pred = torch.stack([p.squeeze(0) for p in pred], dim=-1)  # (P, N, 4)    
    #     self.update_cache(pred)  # update cache
    #     if self.count < 5:
    #         weights = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
    #         # self.count += 1
    #     pred= self.fuse_t1_prediction(weights.squeeze(0))  # fusion t+1 prediction

    #     return pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3] 

    def test_online(self, test_x, test_y, temperature=0.8):
        self.policy_net.eval()
        bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in test_x]
        bike_y, taxi_y = [y.to(self.device) for y in test_y]

        bike_feat = bike_in[:, -1].reshape(1, -1)
        taxi_feat = taxi_in[:, -1].reshape(1, -1)
        state = torch.cat([bike_feat, taxi_feat], dim=1).unsqueeze(0)  # shape: (1, 1, D)

        t1 = time.time()
        probs, weight_table, self.h = self.policy_net(state, self.h)
        t2 = time.time()

        if self.count > 1:
            self.time += (t2 - t1)

        # tempareture sample
        if temperature != 1.0:
            logits = torch.log(probs + 1e-8)
            probs = F.softmax(logits / temperature, dim=-1)

        m = Categorical(probs)
        action = m.sample().item()
        self.actions.append(action)

        chosen_len = self.candidate_lengths[action]
        weights = weight_table[action]
        print(f"Sampled action: {action}, Weights: {weights}")
        # chosen_len = 24
        bike_in_i = bike_in[:, -chosen_len:, :, :]
        bike_adj_i = bike_adj[:, -chosen_len:, :, :]
        taxi_in_i = taxi_in[:, -chosen_len:, :, :]
        taxi_adj_i = taxi_adj[:, -chosen_len:, :, :]

        t1 = time.time()
        pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i))
        t2 = time.time()
        if self.count > 1:
            self.time += (t2 - t1)

        pred = torch.stack([p.squeeze(0) for p in pred], dim=-1)  # (P, N, 4)
        self.update_cache(pred)  # update cache

        # fusion weight
        if self.count < 5:
            # weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=self.device)
            weights = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            self.count += 1
        # else:
        #     weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=self.device)
        pred = self.fuse_t1_prediction(weights.squeeze(0))  # fusion t+1 prediction

        return pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
