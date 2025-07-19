import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from models.model import Net_timesnet_sample_onetimesnet
from models.model import PolicyNet
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLatentTensorGate2d
from utils import mdtp

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
        self.policy = PolicyNet(
            input_dim=node_num * in_features * 2,  # bike + taxi
            hidden_dim=128,
            num_arms=len(candidate_lengths)
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = mdtp.mae_weight

    def train(self, train_x, train_y):
        """
        train_x: (bike_in, bike_adj, taxi_in, taxi_adj)
        train_y: (bike_y, taxi_y)
        """
        bike_in, bike_adj, taxi_in, taxi_adj = [x.to(self.device) for x in train_x]
        bike_y, taxi_y = [y.to(self.device) for y in train_y]

        # 提取 bike 和 taxi 的最后一帧 (B, N, F) → 展平为 (B, N*F)
        bike_feat = bike_in[:, -1].reshape(self.batch_size, -1)   # shape: (B, N*F)
        taxi_feat = taxi_in[:, -1].reshape(self.batch_size, -1)   # shape: (B, N*F)

        # 拼接成策略网络输入 (B, 2*N*F)
        state = torch.cat([bike_feat, taxi_feat], dim=1)          # shape: (B, 2*N*F)

        all_losses = []
        ref_losses = []
        log_probs = []
        h = None  # LSTM 隐藏状态初始化
        for i in range(self.batch_size):

            # 输入策略网络，输出长度选择概率分布
            x = state[i, :].unsqueeze(0).unsqueeze(0)
            probs, h = self.policy(x, h)
            m = torch.distributions.Categorical(probs)
            action = m.sample()            # shape: (1,)
            log_prob = m.log_prob(action) # shape: (1,)
            log_probs.append(log_prob)

            chosen_len = self.candidate_lengths[action.item()]
            ref_len = max(self.candidate_lengths)

            # 截取对应长度的输入
            bike_in_i = bike_in[i:i+1, -chosen_len:, :, :]
            bike_adj_i = bike_adj[i:i+1, -chosen_len:, :, :]
            taxi_in_i = taxi_in[i:i+1, -chosen_len:, :, :]
            taxi_adj_i = taxi_adj[i:i+1, -chosen_len:, :, :]

            with torch.no_grad():
                pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i), window_size=chosen_len)
                pred_ref = self.models[ref_len]((bike_in[i:i+1, -ref_len:, :, :],
                                                 bike_adj[i:i+1, -ref_len:, :, :],
                                                 taxi_in[i:i+1, -ref_len:, :, :],
                                                 taxi_adj[i:i+1, -ref_len:, :, :]), window_size=ref_len)

            loss = self.loss_fn(pred[0], bike_y[0][i:i+1]) + self.loss_fn(pred[1], bike_y[1][i:i+1]) + \
                   self.loss_fn(pred[2], taxi_y[0][i:i+1]) + self.loss_fn(pred[3], taxi_y[1][i:i+1])
            loss_ref = self.loss_fn(pred_ref[0], bike_y[0][i:i+1]) + self.loss_fn(pred_ref[1], bike_y[1][i:i+1]) + \
                       self.loss_fn(pred_ref[2], taxi_y[0][i:i+1]) + self.loss_fn(pred_ref[3], taxi_y[1][i:i+1])

            all_losses.append(loss)
            ref_losses.append(loss_ref)

        # 奖励为负的 delta loss
        rewards = torch.stack([ref_losses[i] - all_losses[i] for i in range(self.batch_size)]).detach()
        log_probs = torch.stack(log_probs).squeeze()
        rewards = (rewards) * 1e4
        loss_rl = -(log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss_rl.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optimizer.step()
        # print(rewards.sum().item())
        return loss_rl.item(), rewards.mean().item()

    @torch.no_grad()
    def test(self, test_x):
        """
        使用策略网络决策输入长度，并用对应模型进行预测。

        Args:
            test_x: (bike_in, bike_adj, taxi_in, taxi_adj)
                bike_in: [B, T, N, F]
                bike_adj: [B, T, N, 2]
        Returns:
            pred: 模型预测输出（bike_start, bike_end, taxi_start, taxi_end）
        """
        self.policy.eval()

        bike_in, bike_adj, taxi_in, taxi_adj = test_x
        state = bike_in[:, -1].reshape(self.batch_size, -1)  # [B, N*F]
        probs = self.policy(state)  # [B, L]
        actions = torch.argmax(probs, dim=-1)  # greedy 选择

        preds = []
        for i in range(self.batch_size):
            chosen_len = self.candidate_lengths[actions[i]]

            bike_in_i = bike_in[i:i+1, -chosen_len:, :, :]
            bike_adj_i = bike_adj[i:i+1, -chosen_len:, :, :]
            taxi_in_i = taxi_in[i:i+1, -chosen_len:, :, :]
            taxi_adj_i = taxi_adj[i:i+1, -chosen_len:, :, :]

            pred = self.models[chosen_len]((bike_in_i, bike_adj_i, taxi_in_i, taxi_adj_i), window_size=chosen_len)
            preds.append(pred)

        # 将每个 batch 的预测拼接起来
        bike_start = torch.cat([p[0] for p in preds], dim=0)
        bike_end = torch.cat([p[1] for p in preds], dim=0)
        taxi_start = torch.cat([p[2] for p in preds], dim=0)
        taxi_end = torch.cat([p[3] for p in preds], dim=0)

        return bike_start, bike_end, taxi_start, taxi_end