import torch
from torch import nn
from models.expert_s import Generator
from models.TimesNet import Model_onetimenet
from models.smoe import GatedSpatialMoE2d

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_arms):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_arms)

    def forward(self, x):
        """
        输入: x: [B, input_dim]，例如节点特征展平后的输入状态
        输出: 每个 arm 的 softmax 概率 [B, num_arms]
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return probs


class Net_timesnet_sample_onetimesnet(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
        super(Net_timesnet_sample_onetimesnet, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size
        self.pred_size = pred_size
        self.smoe = GatedSpatialMoE2d(smoe_config)
        # 3个GCN expert for bike, 3个GCN expert for taxi
        self.bike_gcn = nn.ModuleList([
            Generator(window_size, node_num, in_features, out_features) for _ in range(3)
        ])
        self.taxi_gcn = nn.ModuleList([
            Generator(window_size, node_num, in_features, out_features) for _ in range(3)
        ])
        timesnetconfig = type('Config', (), {
            'seq_len': window_size,
            'pred_len': self.pred_size,
            'top_k': 1,
            'd_model': node_num * out_features,
            'd_ff': node_num * 2,
            'num_kernels': 2,
            'e_layers': 1,
            'c_out': node_num * 4,
            'batch_size': batch_size
        })()
        self.timesnet = Model_onetimenet(timesnetconfig)
        self.fc1 = nn.Sequential(nn.Linear(node_num, node_num), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(node_num, node_num), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(node_num, node_num), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(node_num, node_num), nn.ReLU())

    def forward(self, x, window_size):
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        # 3个bike expert
        bike_experts = [g(bike_node_ori, bike_adj_ori) for g in self.bike_gcn]
        # 3个taxi expert
        taxi_experts = [g(taxi_node_ori, taxi_adj_ori) for g in self.taxi_gcn]

        # [B, T, D, F] -> [B, T, D, 1, F]
        bike_experts = [b.unsqueeze(3) for b in bike_experts]
        taxi_experts = [t.unsqueeze(3) for t in taxi_experts]
        # 合并为 [B, T, D, E, F]
        experts = torch.cat(bike_experts + taxi_experts, dim=3)
        # 送入 SMoE 层，输出 (B, T, D, K, F)
        smoe_out = self.smoe(experts)

        # 后续处理

        # 对 K 个专家的 F 特征求和 -> (B, T, D, F)
        smoe_out = smoe_out.sum(dim=3)
        out = smoe_out.view(self.batch_size, self.window_size, -1)
        # out = out[:, -window_size:, :]
        timesnetout, _ = self.timesnet(bike_node_ori, out)
        timesnetout = timesnetout.view(self.batch_size, self.pred_size, self.node_num, -1)
        bike_start = self.fc1(timesnetout[:, :, :, 0])
        bike_end = self.fc2(timesnetout[:, :, :, 1])
        taxi_start = self.fc3(timesnetout[:, :, :, 2])
        taxi_end = self.fc4(timesnetout[:, :, :, 3])
        return bike_start, bike_end, taxi_start, taxi_end
