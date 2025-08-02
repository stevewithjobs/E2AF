import torch
from torch import nn
from models.expert_s import GeneratorMultiExpert
from models.TimesNet import Model_onetimenet
from models.smoe import GatedSpatialMoE2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.distributions import Categorical, Dirichlet


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_arms, pred_size, num_layers):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.action_head = nn.Linear(hidden_dim, num_arms)     # Discrete action logits

        init_weight = torch.tensor([0, 0, 0, 1.0])  # t+4 heaviest
        self.weight_table = nn.Parameter(init_weight.repeat(num_arms, 1))  # shape: (num_arms, 4)

    def forward(self, x, h=None):
        # x: (B, 1, D)
        out, h = self.encoder(x, h)         # LSTM
        feat = out[:, -1]                   # (B, H)

        # 动作分布
        action_logits = self.action_head(feat)
        action_probs  = F.softmax(action_logits, dim=-1)

        normalized_weights = F.softmax(self.weight_table, dim=-1)  # (num_arms, pred_size)


        return action_probs, normalized_weights, h


class Net_timesnet_sample_onetimesnet(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
        super(Net_timesnet_sample_onetimesnet, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size
        self.pred_size = pred_size
        self.smoe = GatedSpatialMoE2d(smoe_config)

        self.bike_gcn = GeneratorMultiExpert(
            window_size, node_num,in_features,out_features,num_experts=3
        )
        self.taxi_gcn = GeneratorMultiExpert(
            window_size, node_num,in_features,out_features,num_experts=3
        )

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

        self.gcntime = 0

    def forward(self, x):
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        t1 = time.time()
        bike_experts = self.bike_gcn(bike_node_ori, bike_adj_ori)
        taxi_experts = self.taxi_gcn(taxi_node_ori, taxi_adj_ori)
        t2 = time.time()
        self.gcntime += (t2 - t1)
        # print(self.gcntime)

        experts = torch.cat([bike_experts, taxi_experts], dim=3)
        # Send to SMoE layer, output (B, T, D, K, F)
        smoe_out = self.smoe(experts)
        # smoe_out = experts

        # Sum the F features of K experts -> (B, T, D, F)
        smoe_out = smoe_out.sum(dim=3)
        out = smoe_out.view(self.batch_size, self.window_size, -1)

        timesnetout, _ = self.timesnet(bike_node_ori, out)
        timesnetout = timesnetout.view(self.batch_size, self.pred_size, self.node_num, -1)
        bike_start = self.fc1(timesnetout[:, :, :, 0])
        bike_end = self.fc2(timesnetout[:, :, :, 1])
        taxi_start = self.fc3(timesnetout[:, :, :, 2])
        taxi_end = self.fc4(timesnetout[:, :, :, 3])
        return bike_start, bike_end, taxi_start, taxi_end
