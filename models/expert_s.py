# import torch

# from torch import nn
# from torch.nn import init
# import torch.nn.functional as F
# import time

# def get_crossadj(A, B):
#     """
#     对两个形状为 (batch_size, Window_size, node_num, 2) 的张量，
#     如果在第 i 个节点上两个张量的进出流量都不为 0，
#     则构建一个 (node_num, node_num) 的对角矩阵，其中 (i, i) = 1, 其他位置为 0。
    
#     参数:
#     A, B: 形状为 (batch_size, Window_size, node_num, 2) 的张量
    
#     返回:
#     对角矩阵 (batch_size, Window_size, node_num, node_num)
#     """
#     batch_size, window_size, node_num, _ = A.shape
    
#     # 检查两个张量在最后一个维度上是否进出流量都不为 0
#     non_zero_A = (A != 0).all(dim=-1)  # 在最后一个维度上进出流量都不为0的布尔矩阵
#     non_zero_B = (B != 0).all(dim=-1)  # 在最后一个维度上进出流量都不为0的布尔矩阵
    
#     # 只有当 A 和 B 在同一个节点上流量都不为 0 时，才返回 True
#     condition = non_zero_A & non_zero_B
#     tmp = torch.sum(condition, dim=-1)
    
#     # 创建一个形状为 (batch_size, Window_size, node_num, node_num) 的对角矩阵
#     diag_matrix = torch.zeros((batch_size, window_size, node_num, node_num), dtype=torch.float32, device=A.device)
    
#     # 将符合条件的 (i, i) 对角线置为 1
#     diag_matrix[:, :, torch.arange(node_num), torch.arange(node_num)] = condition.float()
    
#     return diag_matrix

# def merge_alladj(A, B, crossA, crossB):
#     """
#     对两个四维张量的最后两个维度进行对角矩阵组合，返回一个新的四维张量。
#     A: shape (batch_size, window_size, h1, w1)
#     B: shape (batch_size, window_size, h2, w2)
#     crossA: (window_size, h2, w2)
#     crossB: (window_size, h2, w2)
#     返回: 对角组合后的张量，shape (batch_size, window_size, h1+h2, w1+w2)
#     """
#     batch_size, num_channels, h1, w1 = A.shape
#     _, _, h2, w2 = B.shape
    
#     # 创建一个足够大的零张量来容纳对角矩阵
#     # h_total = h1 + h2
#     # w_total = w1 + w2
#     # C = torch.zeros((batch_size, num_channels, h_total, w_total), dtype=A.dtype, device=A.device)

    
#     # # 将 A 放在左上角
#     # C[:, :, :h1, :w1] = A
    
#     # # 将 B 放在右下角
#     # C[:, :, h1:, w1:] = B

#     # # 将对角矩阵填充到反对角线
#     # C[:, :, h1:, :w1] = crossA  # 左下角
#     # C[:, :, :h1, w1:] = crossB  # 右上角


#     top = torch.cat((A, crossB), dim=-1)  # Shape: (batch_size, window_size, h1, w1 + w2)
#     bottom = torch.cat((crossA, B), dim=-1)  # Shape: (batch_size, window_size, h2, w1 + w2)
#     C = torch.cat((top, bottom), dim=-2)  # Shape: (batch_size, window_size, h1 + h2, w1 + w2)
    
#     return C

# class GraphConvolution(nn.Module):
#     def __init__(self, window_size, in_features, out_features):
#         super(GraphConvolution, self).__init__()
#         self.window_size = window_size
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weights = nn.Parameter(
#             torch.Tensor(window_size, in_features, out_features)
#         )
#         # self.weights = nn.Parameter(
#         #     torch.Tensor(in_features, out_features)
#         # )
#         self.t = 0
#         self._reset_parameters()

#     def _reset_parameters(self):
#         init.xavier_uniform_(self.weights)

#     def forward(self, adjacency, nodes):
#         """
#         :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
#         :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
#         :return output: FloatTensor (batch_size, window_size, node_num, out_features)
#         """
#         t1 = time.time()
#         batch_size = adjacency.size(0)
#         window_size = adjacency.size(1)
#         # weights = torch.roll(self.weights, shifts=-self.t, dims=0)
#         # weights = weights.unsqueeze(0).expand(batch_size, self.window_size, self.in_features, self.out_features)
#         # weights = self.weights.unsqueeze(0).unsqueeze(0).expand(batch_size, self.window_size, self.in_features, self.out_features)
#         weights = self.weights.unsqueeze(0).expand(batch_size, self.window_size, self.in_features, self.out_features)
#         weights = weights[:, -window_size:, :, :]
#         # tmp1 = adjacency[:, 0, :, :]
#         # tmp2 = nodes[:, 0, :, :]
#         # tmp3 = weights[:, 0, :, :]
#         # t1 = time.time()
#         # output = tmp1.matmul(tmp2).matmul(tmp3)
#         # t2 = time.time()
#         # self.t += t2 - t1
#         # print("gcn time:", self.t)
        
#         output = adjacency.matmul(nodes).matmul(weights)
#         t2 = time.time()
#         self.t += t2 -t1
#         # print("gcn_time", self.t)
#         # self.t += 1
#         return output

#     # def forward(self, adjacency, nodes):
#     #     """
#     #     :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
#     #     :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
#     #     :return output: FloatTensor (batch_size, window_size, node_num, out_features)
#     #     """
#     #     batch_size = adjacency.size(0)
#     #     weights = self.weights.unsqueeze(0).expand(batch_size, self.window_size, self.in_features, self.out_features)
#     #     output = adjacency.matmul(nodes).matmul(weights)
#     #     return output
    
# # class GraphConvolution(nn.Module):
# #     def __init__(self, window_size, in_features, out_features):
# #         super(GraphConvolution, self).__init__()
# #         self.window_size = window_size
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.conv2d = nn.Conv2d(in_channels=4, out_channels=out_features, kernel_size=(1, 1))
# #         # self.linear = nn.Linear(in_features=4, out_features=self.out_features)


# #     def forward(self, adjacency, nodes):
# #         """
# #         :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
# #         :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
# #         :return output: FloatTensor (batch_size, window_size, node_num, out_features)
# #         """
# #         batch_size = adjacency.size(0)
# #         nodes = nodes.permute(0,1,3,2).reshape(batch_size * self.window_size,-1,21,11)
# #         features = self.conv2d(nodes).reshape(batch_size, self.window_size, -1, 231).permute(0, 1, 3, 2)
# #         # features = self.linear(nodes)
# #         output = adjacency.matmul(features)

# #         return output

# import torch
# import torch.nn as nn

# class NodeSampler(nn.Module):
#     def __init__(self, window_size, node_num, sample_node_num):
#         """
#         初始化 NodeSampler 类
        
#         参数:
#         - sampling_ratio (float): 采样比例，默认是 2/3。
#         """
#         super(NodeSampler, self).__init__()
#         # self.mask = torch.rand(node_num)
#         self.window_size = window_size
#         self.node_num = node_num
#         self.gate = nn.Parameter(torch.rand(node_num))
#         self.gate1 = nn.Parameter(torch.zeros(node_num))
#         self.gate2 = nn.Parameter(torch.zeros(node_num))
#         self.sample_node_num = sample_node_num
#         self.sigmoid = nn.Sigmoid()

#     def forward(self):
        
#         self.conf = self.sigmoid(self.gate)
#         self.conf1 = self.sigmoid(self.gate1)
#         self.conf2 = self.sigmoid(self.gate2)

#         connect = self.conf >= 0.7
#         mask1 = self.conf1 < 0.5
#         mask2 = self.conf2 < 0.5

#         return connect, mask1, mask2

# # class Generator(nn.Module):
# #     def __init__(self, window_size, node_num, in_features, out_features):
# #         super(Generator, self).__init__()
# #         self.batch_size = 32
# #         self.window_size = window_size
# #         self.node_num = node_num
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.gcn1 = GraphConvolution(window_size, in_features, out_features)  
# #         self.gcn2 = GraphConvolution(window_size, out_features, out_features)
# #         self.bike_cache = torch.empty((self.batch_size, window_size, node_num, out_features), device='cuda')
# #         self.taxi_cache = torch.empty((self.batch_size, window_size, node_num, out_features), device='cuda')


# #     def forward(self, bike_in_shots, bike_adj, taxi_in_shots, taxi_adj, connect):
# #         """
# #         :param bike_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
# #         :param bike_adj: FloatTensor (batch_size, window_size, node_num, node_num)
# #         :param taxi_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
# #         :param taxi_adj: FloatTensor (batch_size, window_size, node_num, node_num)
# #         :return bike_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
# #         :return taxi_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
# #         """
# #         batch_size, window_size, node_num = bike_in_shots.size()[0: 3]
# #         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #         eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
# #         bike_adj = bike_adj + eye
# #         bike_diag = bike_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(bike_adj.size()) * eye
# #         bike_adjacency = bike_diag.matmul(bike_adj).matmul(bike_diag)
# #         taxi_adj = taxi_adj + eye
# #         taxi_diag = taxi_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(taxi_adj.size()) * eye
# #         taxi_adjacency = taxi_diag.matmul(taxi_adj).matmul(taxi_diag)


# #         in_shots = torch.cat((bike_in_shots, taxi_in_shots), dim=-2)

# #         adj = merge_alladj(bike_adjacency, taxi_adjacency, 0, 0)

# #         gcn_output1 = self.gcn1(adj, in_shots)

# #         bike_gcn_output = gcn_output1[:, :, :self.node_num, :]
# #         taxi_gcn_output = gcn_output1[:, :, self.node_num:, :]
        
# #         return bike_gcn_output, taxi_gcn_output

# class Generator(nn.Module):
#     def __init__(self, batch_size, window_size, node_num, in_features, out_features):
#         super(Generator, self).__init__()
#         self.batch_size = batch_size
#         self.window_size = window_size
#         self.node_num = node_num
#         self.in_features = in_features
#         self.out_features = out_features
#         # batch_size, window_size, node_num = bike_in_shots.size()[0: 3]
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
#         self.gcn1 = GraphConvolution(window_size, in_features, out_features)  
#         self.gcn2 = GraphConvolution(window_size, out_features, out_features)
#         # self.bike_cache = torch.empty((self.batch_size, window_size, node_num, out_features), device='cuda')
#         # self.taxi_cache = torch.empty((self.batch_size, window_size, node_num, out_features), device='cuda')
#         self.adjtime = 0
#         self.gcntime = 0
#         self.t = 0
        
#         # Create zero tensors for the cross connections
#         self.crossA = torch.zeros((self.batch_size, window_size, node_num, node_num),  device=device)
#         self.crossB = torch.zeros((self.batch_size, window_size, node_num, node_num),  device=device)

#     def forward(self, bike_in_shots, bike_adj, taxi_in_shots, taxi_adj, connect):
#         """
#         :param bike_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
#         :param bike_adj: FloatTensor (batch_size, window_size, node_num, node_num)
#         :param taxi_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
#         :param taxi_adj: FloatTensor (batch_size, window_size, node_num, node_num)
#         :return bike_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
#         :return taxi_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
#         """
        
#         # batch_size, window_size, node_num = bike_in_shots.size()[0: 3]
#         # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         # eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
#         t1 = time.time()
#         bike_adj = bike_adj + self.eye
#         bike_diag = bike_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(bike_adj.size()) * self.eye
#         bike_adjacency = bike_diag.matmul(bike_adj).matmul(bike_diag)
#         taxi_adj = taxi_adj + self.eye
#         taxi_diag = taxi_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(taxi_adj.size()) * self.eye
#         taxi_adjacency = taxi_diag.matmul(taxi_adj).matmul(taxi_diag)

#         t2 = time.time()
#         if self.t > 2:
#             self.adjtime += t2 - t1
#         # print("adjtime", self.adjtime)
        
        

#         t1 = time.time()
#         in_shots = torch.cat((bike_in_shots, taxi_in_shots), dim=-2)

#         adj = merge_alladj(bike_adjacency, taxi_adjacency, self.crossA, self.crossB)

#         t2 = time.time()
#         if self.t > 2 :
#             self.gcntime += t2 - t1
#         # print("gcntime", self.gcntime)
        
#         gcn_output1 = self.gcn1(adj, in_shots)

#         bike_gcn_output = gcn_output1[:, :, :self.node_num, :]
#         # self.bike_cache = bike_gcn_output
#         taxi_gcn_output = gcn_output1[:, :, self.node_num:, :]
#         # self.taxi_cache = taxi_gcn_output

#         self.t += 1
        
#         return bike_gcn_output, taxi_gcn_output

        
# class Generator_rl(nn.Module):
#     def __init__(self, window_size, node_num, in_features, out_features):
#         super(Generator_rl, self).__init__()
#         self.window_size = window_size
#         self.node_num = node_num
#         self.in_features = in_features
#         self.out_features = out_features
#         self.gcn = GraphConvolution(window_size, in_features, out_features)
#         # self.fc = nn.Linear(node_num * 2, node_num)
#         self.crossA = torch.nn.Parameter(torch.zeros(node_num))
#         self.crossB = torch.nn.Parameter(torch.zeros(node_num))


#     def forward(self, bike_in_shots, bike_adj, taxi_in_shots, taxi_adj):
#         """
#         :param bike_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
#         :param bike_adj: FloatTensor (batch_size, window_size, node_num, node_num)
#         :param taxi_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
#         :param taxi_adj: FloatTensor (batch_size, window_size, node_num, node_num)
#         :return bike_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
#         :return taxi_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
#         """
#         batch_size, window_size, node_num = bike_in_shots.size()[0: 3]
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
#         bike_adj = bike_adj + eye
#         bike_diag = bike_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(bike_adj.size()) * eye
#         bike_adjacency = bike_diag.matmul(bike_adj).matmul(bike_diag)
#         taxi_adj = taxi_adj + eye
#         taxi_diag = taxi_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(taxi_adj.size()) * eye
#         taxi_adjacency = taxi_diag.matmul(taxi_adj).matmul(taxi_diag)

#         in_shots = torch.cat((bike_in_shots, taxi_in_shots), dim=-2)

#         adj = merge_alladj(bike_adjacency, taxi_adjacency, 0, 0)
#         gcn_output = self.gcn(adj, in_shots)

#         return gcn_output
        

# class ReasoningNet(nn.Module):
#     def __init__(self, out_features, window_size, node_num):
#         super(ReasoningNet, self).__init__()
#         self.window_size = window_size
#         self.out_features = out_features
#         self.node_num = node_num
#         # 初始化权重w
#         self.w1 = nn.Parameter(torch.randn(2 * out_features * node_num, 1))
#         self.sigmoid1 = nn.Sigmoid()
#         self.w2 = nn.Parameter(torch.randn(2 * out_features * node_num, 1))
#         self.sigmoid2 = nn.Sigmoid()

#     def forward(self,bike_feature,taxi_feature):
#         # 将ha取负并与hb在最后一个维度上连接
#         concatenated1 = torch.cat((-bike_feature, taxi_feature), dim=-1)  # 形状为 (batch_size, window_size, 2 * out_features)
#         # 重塑concatenated以适合矩阵乘法
#         concatenated1 = concatenated1.view(-1, 2 * self.out_features * self.node_num)
#         # 计算权重
#         weight1 = self.sigmoid1(torch.matmul(concatenated1, self.w1))  # 形状为 (batch_size * window_size, 1)
#         # 重塑weight以匹配ha和hb的形状
#         weight1 = weight1.view(-1, self.window_size, 1)
#         # 扩展权重以进行元素级乘法
#         weight_expanded1 = weight1.expand(-1, -1, self.out_features * self.node_num)


#         # 将ha取负并与hb在最后一个维度上连接
#         concatenated2 = torch.cat((-taxi_feature, bike_feature), dim=-1)  # 形状为 (batch_size, window_size, 2 * out_features)
#         # 重塑concatenated以适合矩阵乘法
#         concatenated2 = concatenated2.view(-1, 2 * self.out_features * self.node_num)
#         # 计算权重
#         weight2 = self.sigmoid2(torch.matmul(concatenated2, self.w2))  # 形状为 (batch_size * window_size, 1)
#         # 重塑weight以匹配ha和hb的形状
#         weight2 = weight2.view(-1, self.window_size, 1)
#         # 扩展权重以进行元素级乘法
#         weight_expanded2 = weight2.expand(-1, -1, self.out_features * self.node_num)
#         # 计算结果
#         share_feature = bike_feature * weight_expanded1 + taxi_feature * weight_expanded2

#         return share_feature,weight1,weight2


# class ReasoningNetImp(nn.Module):
#     def __init__(self, out_features, window_size,node_num):
#         super(ReasoningNetImp, self).__init__()
#         self.window_size = window_size
#         self.out_features = out_features
#         self.node_num = node_num
#         # 初始化权重w
#         self.w1 = nn.Parameter(torch.randn(2 * out_features, 1))
#         self.sigmoid1 = nn.Sigmoid()
#         self.w2 = nn.Parameter(torch.randn(2 * out_features, 1))
#         self.sigmoid2 = nn.Sigmoid()

#     def forward(self,bike_feature,taxi_feature):

#         bike_feature = bike_feature.view(-1, self.window_size, self.node_num, self.out_features)  # 形状为 (batch_size, window_size, node_num, out_features)
#         taxi_feature = taxi_feature.view(-1, self.window_size, self.node_num, self.out_features)  # 形状为 (batch_size, window_size, node_num, out_features)

#         # 将bike取负并与taxi在最后一个维度上连接
#         concatenated1 = torch.cat((-bike_feature, taxi_feature), dim=-1)  # 形状为 (batch_size, window_size, node_num，2 * out_features)
#         # 计算权重
#         weight1 = self.sigmoid1(torch.matmul(concatenated1, self.w1))  # 形状为 (batch_size, window_size, node_num，1)
#         weight_expanded1 = weight1.expand(-1, self.window_size, self.node_num, self.out_features)

#         # 将bike取负并与taxi在最后一个维度上连接
#         concatenated2 = torch.cat((-taxi_feature, bike_feature),dim=-1)  # 形状为 (batch_size, window_size, node_num，2 * out_features)
#         # 计算权重
#         weight2 = self.sigmoid2(torch.matmul(concatenated2, self.w2))  # 形状为 (batch_size, window_size, node_num，1)
#         weight_expanded2 = weight2.expand(-1, self.window_size, self.node_num, self.out_features)  # 形状为 (batch_size, window_size, 1, node_num)

#         # 计算结果
#         share_feature = bike_feature * weight_expanded1 + taxi_feature * weight_expanded2  # 形状为 (batch_size, window_size, node_num, out_features)
#         share_feature = share_feature.view(-1, self.window_size, self.node_num * self.out_features)

#         return share_feature,weight1,weight2

import torch

from torch import nn
from torch.nn import init


class GraphConvolution(nn.Module):
    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Parameter(
            torch.Tensor(window_size, in_features, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        """
        :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
        :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
        :return output: FloatTensor (batch_size, window_size, node_num, out_features)
        """
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size, in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        return output

class Generator(nn.Module):
    def __init__(self, window_size, node_num, in_features, out_features):
        super(Generator, self).__init__()
        self.gcn = GraphConvolution(window_size, in_features, out_features)

    def forward(self, x_in, adj):
        """
        x_in:  (B, T, N, in_features)
        adj:   (B, T, N, N)
        out:   (B, T, out_features, N)
        """
        B, T, N = x_in.shape[:3]
        device = x_in.device

        eye = torch.eye(N, device=device).unsqueeze(0).unsqueeze(0).expand(B, T, N, N)
        adj = adj + eye
        deg_inv_sqrt = adj.sum(dim=-1, keepdim=True).pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)
        feat = self.gcn(norm_adj, x_in)  # -> (B, T, N, F)
        return feat 
    
# class Generator(nn.Module):
#     def __init__(self, window_size, node_num, in_features, out_features):
#         super(Generator, self).__init__()
#         self.window_size = window_size
#         self.node_num = node_num
#         self.in_features = in_features
#         self.out_features = out_features
#         self.gcn = GraphConvolution(window_size, in_features, out_features)
#         self.fc = nn.Linear(node_num * 2, node_num)

#     def forward(self, bike_in_shots, bike_adj, taxi_in_shots, taxi_adj):
#         """
#         :param bike_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
#         :param bike_adj: FloatTensor (batch_size, window_size, node_num, node_num)
#         :param taxi_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
#         :param taxi_adj: FloatTensor (batch_size, window_size, node_num, node_num)
#         :return bike_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
#         :return taxi_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
#         """
#         batch_size, window_size, node_num = bike_in_shots.size()[0: 3]
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
#         bike_adj = bike_adj + eye
#         bike_diag = bike_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(bike_adj.size()) * eye
#         bike_adjacency = bike_diag.matmul(bike_adj).matmul(bike_diag)
#         taxi_adj = taxi_adj + eye
#         taxi_diag = taxi_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(taxi_adj.size()) * eye
#         taxi_adjacency = taxi_diag.matmul(taxi_adj).matmul(taxi_diag)
#         bike_gcn_output = self.gcn(bike_adjacency, bike_in_shots)
#         taxi_gcn_output = self.gcn(taxi_adjacency, taxi_in_shots)
#         bike_gcn_output = bike_gcn_output.view(batch_size, window_size, -1)
#         taxi_gcn_output = taxi_gcn_output.view(batch_size, window_size, -1)
#         return bike_gcn_output, taxi_gcn_output





