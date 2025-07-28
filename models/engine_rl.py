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

class TrainerMAB:
    def __init__(self, batch_size, candidate_lengths, node_num, in_features, out_features,
                 lstm_features, device, learning_rate, weight_decay, gradient_clip,
                 smoe_start_epoch, pred_size):
        self.device = device
        self.batch_size = batch_size
        self.candidate_lengths = candidate_lengths
        self.gradient_clip = gradient_clip
        self.pred_size = pred_size

        # 每个 window_size 都需要一个专属的 smoe_config
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

        # 构建多尺度模型（冻结参数）
        self.models = {}
        for l in candidate_lengths:
            model = Net_timesnet_sample_onetimesnet(
                1, l, node_num, in_features, out_features,
                lstm_features, smoe_config=self.smoe_config_map[l], pred_size=pred_size
            ).to(device)

            model_path = f'./checkpoints/exp_nyc_{l}/best_model.pth'
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()  # 不启用 BN/dropout
            for param in model.parameters():
                param.requires_grad = False
            self.models[l] = model

        # 策略网络（负责选择 window size）
        self.h = None
        self.policy_net = PolicyNet(
            input_dim=node_num * in_features * 2,  # bike + taxi
            hidden_dim=32,
            num_arms=len(candidate_lengths),
            pred_size=pred_size
        ).to(device)

        self.cache = torch.zeros(
            (pred_size, pred_size, node_num, 4), dtype=torch.float32, device=device
        )
        self.count = 0

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = mdtp.rmse

        self.actions = []

    # def train(self, train_x, train_y):
    #     """
    #     train_x: (bike_in, bike_adj, taxi_in, taxi_adj)
    #     train_y: (bike_y, taxi_y)
    #     """
    #     bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in train_x]
    #     bike_y, taxi_y = [y.to(self.device) for y in train_y]

    #     # 提取 bike 和 taxi 的最后一帧 (B, N, F) → 展平为 (B, N*F)
    #     bike_feat = bike_in[:, -1].reshape(self.batch_size, -1)   # shape: (B, N*F)
    #     taxi_feat = taxi_in[:, -1].reshape(self.batch_size, -1)   # shape: (B, N*F)

    #     # 拼接成策略网络输入 (B, 2*N*F)
    #     state = torch.cat([bike_feat, taxi_feat], dim=1)          # shape: (B, 2*N*F)

    #     all_losses = []
    #     ref_losses = []
    #     log_probs = []
    #     if self.h is not None:
    #         self.h = self.h.detach()
    #     for i in range(self.batch_size):

    #         # 输入策略网络，输出长度选择概率分布
    #         x = state[i, :].unsqueeze(0).unsqueeze(0)
    #         probs, h = self.policy(x, self.h)
    #         m = torch.distributions.Categorical(probs)
    #         action = m.sample()            # shape: (1,)
    #         log_prob = m.log_prob(action) # shape: (1,)
    #         log_probs.append(log_prob)

    #         chosen_len = self.candidate_lengths[action.item()]
    #         ref_len = max(self.candidate_lengths)

    #         # 截取对应长度的输入
    #         bike_in_i = bike_in[i:i+1, -chosen_len:, :, :]
    #         bike_adj_i = bike_adj[i:i+1, -chosen_len:, :, :]
    #         taxi_in_i = taxi_in[i:i+1, -chosen_len:, :, :]
    #         taxi_adj_i = taxi_adj[i:i+1, -chosen_len:, :, :]

    #         with torch.no_grad():
    #             pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i))
    #             pred_ref = self.models[ref_len]((bike_in[i:i+1, -ref_len:, :, :],
    #                                              bike_adj[i:i+1, -ref_len:, :, :],
    #                                              taxi_in[i:i+1, -ref_len:, :, :],
    #                                              taxi_adj[i:i+1, -ref_len:, :, :]))

    #         loss = self.loss_fn(pred[0], bike_y[0][i:i+1]) + self.loss_fn(pred[1], bike_y[1][i:i+1]) + \
    #                self.loss_fn(pred[2], taxi_y[0][i:i+1]) + self.loss_fn(pred[3], taxi_y[1][i:i+1])
    #         loss_ref = self.loss_fn(pred_ref[0], bike_y[0][i:i+1]) + self.loss_fn(pred_ref[1], bike_y[1][i:i+1]) + \
    #                    self.loss_fn(pred_ref[2], taxi_y[0][i:i+1]) + self.loss_fn(pred_ref[3], taxi_y[1][i:i+1])

    #         all_losses.append(loss)
    #         ref_losses.append(loss_ref)

    #     # 奖励为负的 delta loss
    #     rewards = torch.stack([ref_losses[i] - all_losses[i] for i in range(self.batch_size)]).detach()
    #     log_probs = torch.stack(log_probs).squeeze()
    #     rewards = (rewards) * 1e4
    #     loss_rl = -(log_probs * rewards).mean()

    #     self.optimizer.zero_grad()
    #     loss_rl.backward()
    #     torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
    #     self.optimizer.step()
    #     # print(rewards.sum().item())
    #     return loss_rl.item(), rewards.mean().item()

    def update_cache(self, prediction):
        """
        prediction: Tensor of shape (pred_t, node, 4)
        """
        # 右下滚动一格（即上一轮 → 下一轮）
        self.cache = torch.roll(self.cache, shifts=(-1, -1), dims=(0, 1))

        # 最后一行写入新的 prediction
        self.cache[-1] = prediction

    def fuse_t1_prediction(self, weight):
        """
        融合所有轮次对 t+1 的预测结果（self.cache[:, 0]）加权输出

        Args:
            weight: Tensor of shape (pred_t,)

        Returns:
            fused_pred: Tensor of shape (node, 4)
        """
        assert self.cache.dim() == 4  # (pred_t, pred_t, node, 4)
        assert weight.shape[0] == self.cache.shape[0], "weight length must match pred_t"

        # Normalize weights
        # weight = weight / weight.sum()

        # cache[:, 0]: 所有轮次对 t+1 的预测 → shape: (pred_t, node, 4)
        pred_t1_all = self.cache[:, 0, :, :]  # (pred_t, node, 4)

        # 加权融合
        fused_pred = (weight[:, None, None] * pred_t1_all).sum(dim=0)  # → (node, 4)

        return fused_pred
    

    def train(self, train_x, train_y):
        bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in train_x]
        bike_y, taxi_y = [y.to(self.device) for y in train_y]

        # === 构造策略输入 =====================================================
        bike_feat = bike_in[:, -1].reshape(self.batch_size, -1)
        taxi_feat = taxi_in[:, -1].reshape(self.batch_size, -1)
        state     = torch.cat([bike_feat, taxi_feat], dim=1)    # (B, D)

        if self.h is not None:
            self.h = tuple([t.detach() for t in self.h])

        log_probs_tot, all_rewards, weight_losses, entropies = [], [], [], []

        for i in range(self.batch_size):
            x = state[i].unsqueeze(0).unsqueeze(0)              # (1,1,D)
            action_probs, weight_table, self.h = self.policy_net(x, self.h)

            # --- sample 离散 action ---
            m            = Categorical(action_probs)
            action       = m.sample()       # 选窗口
            logp_action  = m.log_prob(action)

            chosen_len   = self.candidate_lengths[action.item()]
            ref_len      = max(self.candidate_lengths)

            # # --- sample Dirichlet weight (连续动作) ---
            # weights      = dirichlet_dist.rsample()             # (1, pred_size)
            # logp_weight  = dirichlet_dist.log_prob(weights)     # 标量
            weights = weight_table[action]

            # --- 预测 ---
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

            
            pred = torch.stack([p.squeeze(0) for p in pred], dim=-1)  # (P, N, 4)            self.update_cache(pred[0])  # 更新缓存
            self.update_cache(pred)  # 更新缓存
            if self.count < 5:
                weights = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
                self.count += 1
            fused_pred = self.fuse_t1_prediction(weights.squeeze(0))  # 融合 t+1 的预测
            pred_ref_t1 = torch.stack([p.squeeze(0)[0] for p in pred_ref], dim=-1)  # shape: (4, N)

            # === 取出对应 ground truth: (N,)
            gt_bike_start = bike_y[0, i, 0]
            gt_bike_end   = bike_y[1, i, 0]
            gt_taxi_start = taxi_y[0, i, 0]
            gt_taxi_end   = taxi_y[1, i, 0]

            # === loss ===
            loss_act =  self.loss_fn(pred[0,:,0], gt_bike_start) + \
                        self.loss_fn(pred[0,:,1], gt_bike_end)   + \
                        self.loss_fn(pred[0,:,2], gt_taxi_start) + \
                        self.loss_fn(pred[0,:,3], gt_taxi_end)

            loss_fused = self.loss_fn(fused_pred[:,0], gt_bike_start) + \
                        self.loss_fn(fused_pred[:,1], gt_bike_end)   + \
                        self.loss_fn(fused_pred[:,2], gt_taxi_start) + \
                        self.loss_fn(fused_pred[:,3], gt_taxi_end)

            loss_ref = self.loss_fn(pred_ref_t1[:,0], gt_bike_start) + \
                    self.loss_fn(pred_ref_t1[:,1], gt_bike_end)   + \
                    self.loss_fn(pred_ref_t1[:,2], gt_taxi_start) + \
                    self.loss_fn(pred_ref_t1[:,3], gt_taxi_end)

            reward = (loss_ref - loss_act).detach()           # 越小越好 → 正奖励
            # reward = reward * 1e2  # 放大奖励
            all_rewards.append(reward)

            # 总 log_prob = 离散 + 连续
            # log_probs_tot.append(logp_action + logp_weight)
            log_probs_tot.append(logp_action)
            weight_losses.append(loss_fused)

            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1)  # (B,)
            entropies.append(entropy)   

        # === REINFORCE loss ================================================
        log_probs_tot = torch.stack(log_probs_tot).view(-1)
        rewards       = torch.stack(all_rewards).view(-1)
        weight_losses = torch.stack(weight_losses).mean()  # 平均融合损失
        entropy_bonus = torch.stack(entropies).view(-1).mean()  # 平均熵


        # baseline = rewards.mean().detach()  # 不参与梯度
        # advantages = rewards - baseline
        # loss_rl = -(log_probs_tot * advantages*1e3).mean()

        # rewards = torch.sign(rewards)
        loss_rl = -(log_probs_tot * rewards * 1e3).mean() 
        loss_rl -= entropy_bonus * 0.01  # 熵正则化  
        total_loss = loss_rl + weight_losses
        # total_loss = loss_rl

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        return total_loss.item(), rewards.mean().item()

    def test_online(self, test_x, test_y):
        self.policy_net.eval()
        bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in test_x]
        bike_y, taxi_y = [y.to(self.device) for y in test_y]

        bike_feat = bike_in[:, -1].reshape(1, -1)
        taxi_feat = taxi_in[:, -1].reshape(1, -1)
        state = torch.cat([bike_feat, taxi_feat], dim=1).unsqueeze(0)  # shape: (1, 1, D)

        # 使用保存的 self.h 作为连续 LSTM 隐状态
        probs, weight_table, self.h = self.policy_net(state, self.h)

        print(probs)
        action = torch.argmax(probs, dim=-1).item()
        print(action)
        self.actions.append(action)
        chosen_len = self.candidate_lengths[action]
        weights = weight_table[action]
        print(weights)
        # chosen_len = 24
        # 构造子序列
        bike_in_i = bike_in[:, -chosen_len:, :, :]
        bike_adj_i = bike_adj[:, -chosen_len:, :, :]
        taxi_in_i = taxi_in[:, -chosen_len:, :, :]
        taxi_adj_i = taxi_adj[:, -chosen_len:, :, :]

        pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i))
        pred = torch.stack([p.squeeze(0) for p in pred], dim=-1)  # (P, N, 4)            self.update_cache(pred[0])  # 更新缓存
        self.update_cache(pred)  # 更新缓存
        if self.count < 5:
            weights = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            # self.count += 1
        pred= self.fuse_t1_prediction(weights.squeeze(0))  # 融合 t+1 的预测

        return pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]  # 返回 bike_start, bike_end, taxi_start, taxi_end

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import functools
# from torch.distributions import Categorical
# from models.model import Net_timesnet_sample_onetimesnet, PolicyNet
# from models.smoe_config import SpatialMoEConfig
# from models.gate import SpatialLatentTensorGate2d
# from utils import mdtp
# from models.ppo_utils import PPOAgent, RolloutBuffer, ValueNet  # 假设你已保存为 ppo_utils.py

# class TrainerMAB:
#     def __init__(self, batch_size, candidate_lengths, node_num, in_features, out_features,
#                  lstm_features, device, learning_rate, weight_decay, gradient_clip,
#                  smoe_start_epoch, pred_size):
#         self.device = device
#         self.batch_size = batch_size
#         self.candidate_lengths = candidate_lengths
#         self.gradient_clip = gradient_clip
#         self.pred_size = pred_size

#         self.smoe_config_map = {
#             l: SpatialMoEConfig(
#                 in_planes=2,
#                 out_planes=3,
#                 num_experts=6,
#                 gate_block=functools.partial(SpatialLatentTensorGate2d, node_num=node_num),
#                 save_error_signal=True,
#                 dampen_expert_error=True,
#                 unweighted=True,
#                 block_gate_grad=True,
#                 routing_error_quantile=0.7,
#                 pred_size=pred_size,
#                 windows_size=l
#             ) for l in candidate_lengths
#         }

#         self.models = {}
#         for l in candidate_lengths:
#             model = Net_timesnet_sample_onetimesnet(
#                 1, l, node_num, in_features, out_features,
#                 lstm_features, smoe_config=self.smoe_config_map[l], pred_size=pred_size
#             ).to(device)
#             model_path = f'./checkpoints/exp_nyc_{l}/best_model.pth'
#             state_dict = torch.load(model_path, map_location=device)
#             model.load_state_dict(state_dict)
#             model.eval()
#             for param in model.parameters():
#                 param.requires_grad = False
#             self.models[l] = model

#         self.policy_h = None
#         self.policy_net = PolicyNet(
#             input_dim=node_num * in_features * 2,
#             hidden_dim=32,
#             num_arms=len(candidate_lengths),
#             pred_size=pred_size
#         ).to(device)

#         self.value_h = None
#         self.value_net = ValueNet(
#             input_dim=node_num * in_features * 2,
#             hidden_dim=32,
#         ).to(device)

#         self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
#         self.value_optimizer  = torch.optim.Adam(self.value_net.parameters(),  lr=learning_rate, weight_decay=weight_decay)

#         self.agent = PPOAgent(
#             policy_net=self.policy_net,
#             value_net=self.value_net,
#             policy_optimizer=self.policy_optimizer,
#             value_optimizer=self.value_optimizer,
#             clip_param=0.2,
#             value_coef=0.5,
#             entropy_coef=0.01,
#             max_grad_norm=self.gradient_clip
#         )

#         self.buffer = RolloutBuffer()

#         self.cache = torch.zeros(
#             (pred_size, pred_size, node_num, 4), dtype=torch.float32, device=device
#         )
#         self.count = 0
#         self.loss_fn = mdtp.rmse

#     def update_cache(self, prediction):
#         """
#         prediction: Tensor of shape (pred_t, node, 4)
#         """
#         # 右下滚动一格（即上一轮 → 下一轮）
#         self.cache = torch.roll(self.cache, shifts=(-1, -1), dims=(0, 1))

#         # 最后一行写入新的 prediction
#         self.cache[-1] = prediction

#     def fuse_t1_prediction(self, weight):
#         """
#         融合所有轮次对 t+1 的预测结果（self.cache[:, 0]）加权输出

#         Args:
#             weight: Tensor of shape (pred_t,)

#         Returns:
#             fused_pred: Tensor of shape (node, 4)
#         """
#         assert self.cache.dim() == 4  # (pred_t, pred_t, node, 4)
#         assert weight.shape[0] == self.cache.shape[0], "weight length must match pred_t"

#         # Normalize weights
#         weight = weight / weight.sum()

#         # cache[:, 0]: 所有轮次对 t+1 的预测 → shape: (pred_t, node, 4)
#         pred_t1_all = self.cache[:, 0, :, :]  # (pred_t, node, 4)

#         # 加权融合
#         fused_pred = (weight[:, None, None] * pred_t1_all).sum(dim=0)  # → (node, 4)

#         return fused_pred

#     def compute_returns_and_advantages(self, rewards, values, gamma=0.99):
#         returns = []
#         advs = []
#         G = 0
#         for r, v in zip(reversed(rewards), reversed(values)):
#             G = r + gamma * G
#             returns.insert(0, G)
#         returns = torch.tensor(returns).to(values[0].device)
#         # values = torch.stack(values)
#         advantages = returns.detach() - values.detach()
#         return returns, advantages

#     def train(self, train_x, train_y):
#         self.policy_net.train()
#         self.value_net.train()

#         bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in train_x]
#         bike_y, taxi_y = [y.to(self.device) for y in train_y]

#         bike_feat = bike_in[:, -1].reshape(self.batch_size, -1)
#         taxi_feat = taxi_in[:, -1].reshape(self.batch_size, -1)
#         state = torch.cat([bike_feat, taxi_feat], dim=1)

#         if self.policy_h is not None:
#             self.policy_h = tuple([t.detach() for t in self.policy_h])
#         if self.value_h is not None:
#             self.value_h = tuple([t.detach() for t in self.value_h])

#         for i in range(self.batch_size):
#             x = state[i].unsqueeze(0).unsqueeze(0)
#             action_probs, weight_table, self.policy_h = self.policy_net(x, self.policy_h)
#             m = Categorical(action_probs)
#             action = m.sample()
#             log_prob = m.log_prob(action)
#             entropy = m.entropy()

#             chosen_len = self.candidate_lengths[action.item()]
#             ref_len = max(self.candidate_lengths)
#             weights = weight_table[action]

#             bike_in_i = bike_in[i:i+1, -chosen_len:]
#             bike_adj_i = bike_adj[i:i+1, -chosen_len:]
#             taxi_in_i = taxi_in[i:i+1, -chosen_len:]
#             taxi_adj_i = taxi_adj[i:i+1, -chosen_len:]

#             with torch.no_grad():
#                 pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i))
#                 pred_ref = self.models[ref_len](
#                     (bike_in[i:i+1, -ref_len:], bike_adj[i:i+1, -ref_len:],
#                      taxi_in[i:i+1, -ref_len:], taxi_adj[i:i+1, -ref_len:]))

#             pred = torch.stack([p.squeeze(0) for p in pred], dim=-1)
#             self.update_cache(pred)
#             # if self.count < 5:
#             #     weights = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
#             #     self.count += 1
#             # fused_pred = self.fuse_t1_prediction(weights.squeeze(0))
#             pred_ref_t1 = torch.stack([p.squeeze(0)[0] for p in pred_ref], dim=-1)

#             gt_bike_start = bike_y[0, i, 0]
#             gt_bike_end = bike_y[1, i, 0]
#             gt_taxi_start = taxi_y[0, i, 0]
#             gt_taxi_end = taxi_y[1, i, 0]

#             # loss_fused = self.loss_fn(fused_pred[:,0], gt_bike_start) + \
#             #             self.loss_fn(fused_pred[:,1], gt_bike_end)   + \
#             #             self.loss_fn(fused_pred[:,2], gt_taxi_start) + \
#             #             self.loss_fn(fused_pred[:,3], gt_taxi_end)

#             loss_act = self.loss_fn(pred[0,:,0], gt_bike_start) + \
#                        self.loss_fn(pred[0,:,1], gt_bike_end)   + \
#                        self.loss_fn(pred[0,:,2], gt_taxi_start) + \
#                        self.loss_fn(pred[0,:,3], gt_taxi_end)

#             loss_ref = self.loss_fn(pred_ref_t1[:,0], gt_bike_start) + \
#                        self.loss_fn(pred_ref_t1[:,1], gt_bike_end) + \
#                        self.loss_fn(pred_ref_t1[:,2], gt_taxi_start) + \
#                        self.loss_fn(pred_ref_t1[:,3], gt_taxi_end)

#             reward = (loss_ref - loss_act) * 1e1

#             value, self.value_h = self.value_net(state[i].unsqueeze(0).unsqueeze(0), self.value_h)
#             self.buffer.add(
#                 state=state[i].detach(),
#                 action=action.detach(),
#                 log_prob=log_prob,
#                 reward=reward,
#                 value=value,
#                 entropy=entropy
#             )

#         # loss, reward_mean = self.agent.update(self.buffer)
#         log_probs = torch.stack(self.buffer.log_probs).view(-1)
#         entropies = torch.stack(self.buffer.entropies).view(-1)
#         values = torch.stack(self.buffer.values).view(-1)        # ← 加上 stack
#         rewards = torch.stack(self.buffer.rewards).view(-1)  

#         returns, advantages = self.compute_returns_and_advantages(rewards, values)

#         # Policy update
#         # entropies = torch.stack(entropies)

#         policy_loss =  -(log_probs * advantages).mean()
#         entropy_bonus = entropies.mean()

#         self.policy_optimizer.zero_grad()
#         # loss = (policy_loss - 0.1 * entropy_bonus)
#         loss = (policy_loss)
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
#         self.policy_optimizer.step()

#         # Value update
#         # value_preds = torch.stack(values)
#         value_loss = F.mse_loss(values, returns)

#         self.value_optimizer.zero_grad()
#         (value_loss).backward()
#         nn.utils.clip_grad_norm_(self.value_net.parameters(), self.gradient_clip)
#         self.value_optimizer.step() 

#         self.buffer.clear()
#         return loss.item(), advantages.mean().item() # 返回损失和平均优势值

#     def test_online(self, test_x, test_y):
#         self.policy_net.eval()
#         bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in test_x]
#         bike_y, taxi_y = [y.to(self.device) for y in test_y]

#         bike_feat = bike_in[:, -1].reshape(1, -1)
#         taxi_feat = taxi_in[:, -1].reshape(1, -1)
#         state = torch.cat([bike_feat, taxi_feat], dim=1).unsqueeze(0)  # shape: (1, 1, D)

#         # 使用保存的 self.h 作为连续 LSTM 隐状态
#         probs, weight_table, self.policy_h = self.policy_net(state, self.policy_h)

#         print(probs)
#         action = torch.argmax(probs, dim=-1).item()
#         print(action)
#         chosen_len = self.candidate_lengths[action]
#         weights = weight_table[action]
#         print(weights)
#         # chosen_len = 16
#         # 构造子序列
#         bike_in_i = bike_in[:, -chosen_len:, :, :]
#         bike_adj_i = bike_adj[:, -chosen_len:, :, :]
#         taxi_in_i = taxi_in[:, -chosen_len:, :, :]
#         taxi_adj_i = taxi_adj[:, -chosen_len:, :, :]

#         pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i))
#         pred = torch.stack([p.squeeze(0) for p in pred], dim=-1)  # (P, N, 4)            self.update_cache(pred[0])  # 更新缓存
#         self.update_cache(pred)  # 更新缓存
#         if self.count < 5:
#             weights = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
#             self.count += 1
#         pred= self.fuse_t1_prediction(weights.squeeze(0))  # 融合 t+1 的预测

#         return pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]  # 返回 bike_start, bike_end, taxi_start, taxi_end
