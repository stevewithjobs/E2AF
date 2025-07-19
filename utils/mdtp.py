import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
import scipy.sparse as sp
from scipy.sparse import linalg
import torch.nn as nn


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MyDataset(Dataset):
    def __init__(self, path, window_size, batch_size, pad_with_last_sample):
        super(MyDataset, self).__init__()
        original_data = np.load(path)
        self.batch_size = batch_size
        if pad_with_last_sample:
            num_padding = (batch_size - (len(original_data) % batch_size)) % batch_size + window_size
            x_padding = np.repeat(original_data[-1:], num_padding, axis=0)
            self.data = torch.from_numpy(np.concatenate([original_data, x_padding], axis=0)).float()
        else:
            self.data = torch.from_numpy(original_data).float()
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size]
    
class MyDataset_nstp(Dataset):
    def __init__(self, path, window_size, batch_size, pad_with_last_sample, pred_size):
        super(MyDataset_nstp, self).__init__()
        original_data = np.load(path)
        self.batch_size = batch_size
        if pad_with_last_sample:
            num_padding = (batch_size - (len(original_data) % batch_size)) % batch_size + window_size   
            x_padding = np.repeat(original_data[-1:], num_padding, axis=0)
            self.data = torch.from_numpy(np.concatenate([original_data, x_padding], axis=0)).float()
        else:
            self.data = torch.from_numpy(original_data).float()
        self.window_size = window_size
        self.num = self.data.size(0) - window_size 
        self.pred_size = pred_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        in_start = item
        in_end = in_start + self.window_size
        out_start = in_end
        out_end = out_start + self.pred_size
        return self.data[in_start:in_end], self.data[out_start:out_end]
    

class MyDataset_nstponline(Dataset):
    def __init__(self, path, window_size, batch_size, pred_size, data_fraction=1.0):
        super(MyDataset_nstponline, self).__init__()
        # self.data = data  # 数据 shape: (total_length, node_num, features)
        self.data = np.load(path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.total_length = self.data.shape[0]

        total_data_length = self.data.shape[0]  # 原始数据总长度
        self.total_length = int(total_data_length * data_fraction)  # 选取部分数据

        self.pred_size = pred_size

        # 如果不能整除，丢弃不足部分
        self.subset_size = (self.total_length - self.window_size - self.pred_size) // self.batch_size
        self.total_length = self.subset_size * self.batch_size  # 更新 total_length 保证可以整除

    def __len__(self):
        return self.subset_size  # 滑动窗口减少实际长度

    def __getitem__(self, index):
        batch_input = []
        batch_target = []

        for i in range(self.batch_size):
            # 获取滑动窗口的输入数据
            start_idx = i * self.subset_size + index     
            end_idx = start_idx + self.window_size
            batch_input.append(torch.tensor(self.data[start_idx:end_idx], dtype=torch.float))  # 转换为 Tensor

            # 获取下一时刻的真值
            pred_start_idx = end_idx  # 下一时刻的索引
            pred_end_idx = pred_start_idx + self.pred_size
            batch_target.append(torch.tensor(self.data[pred_start_idx:pred_end_idx], dtype=torch.float))  # 转换为 Tensor

        # 返回滑动窗口的数据和下一时刻的真值
        return torch.stack(batch_input), torch.stack(batch_target)
    
class MyDataset_rl(Dataset):
    def __init__(self, path, window_size, batch_size):
        super(MyDataset_rl, self).__init__()
        # self.data = data  # 数据 shape: (total_length, node_num, features)
        self.data = np.load(path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.total_length = self.data.shape[0]

        # 如果不能整除，丢弃不足部分
        self.subset_size = (self.total_length - self.window_size) // self.batch_size
        self.total_length = self.subset_size * self.batch_size  # 更新 total_length 保证可以整除

    def __len__(self):
        return 1 # 滑动窗口减少实际长度

    def __getitem__(self, index):
        batch_input = []
        for i in range(self.batch_size):
            # 获取滑动窗口的输入数据
            start_idx = i * self.subset_size + index     
            end_idx = start_idx + self.window_size + self.subset_size
            batch_input.append(torch.tensor(self.data[start_idx:end_idx], dtype=torch.float))  # 转换为 Tensor

        # 返回滑动窗口的数据和下一时刻的真值
        return torch.stack(batch_input)
    

class MyDataset_mmsm(Dataset):
    def __init__(self, path, window_size, batch_size):
        super(MyDataset_mmsm, self).__init__()
        # self.data = data  # 数据 shape: (total_length, node_num, features)
        self.data = np.load(path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.total_length = self.data.shape[0]

        # 如果不能整除，丢弃不足部分
        self.subset_size = (self.total_length - self.window_size) // self.batch_size
        self.total_length = self.subset_size * self.batch_size  # 更新 total_length 保证可以整除

    def __len__(self):
        return self.subset_size # 滑动窗口减少实际长度

    def __getitem__(self, index):
        batch_input = []
        batch_output = []
        for i in range(self.batch_size):
            # 获取滑动窗口的输入数据
            start_idx = i * self.subset_size + index     
            end_idx = start_idx + self.window_size
            batch_input.append(torch.tensor(self.data[start_idx:end_idx], dtype=torch.float))  # 转换为 Tensor
            batch_output.append(torch.tensor(self.data[end_idx], dtype=torch.float))  # 转换为 Tensor

        # 返回滑动窗口的数据和下一时刻的真值
        return torch.stack(batch_input), torch.stack(batch_output)




# def masked_mse(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = (preds - labels) ** 2
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def masked_rmse(preds, labels, null_val=np.nan):
#     return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


# def masked_mae(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels)
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def masked_mape(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels) / labels
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def mae(preds, labels, threshold):
#     mask = labels > threshold
#     if torch.sum(mask) != 0:
#         avg_mae = torch.mean(torch.abs(labels[mask] - preds[mask]))
#         return avg_mae
#     else:
#         return torch.tensor(0)

# def rmse(preds, labels, threshold):
#     mask = labels > threshold
#     if torch.sum(mask) != 0:
#         avg_rmse = torch.sqrt(torch.mean((labels[mask] - preds[mask]) ** 2))
#         return avg_rmse
#     else:
#         return torch.tensor(0)

# def mape(preds, labels, threshold):
#     mask = labels > threshold
#     if torch.sum(mask) != 0:
#         avg_mape = torch.mean(torch.abs(labels[mask] - preds[mask]) / labels[mask])
#         return avg_mape
#     else:
#         return torch.tensor(0)

def mae(preds, labels, mask):
    if torch.sum(mask) != 0:
        avg_mae = torch.mean(torch.abs(labels[mask] - preds[mask]))
        return avg_mae
    else:
        return torch.tensor(0)

def rmse(preds, labels, mask):
    if torch.sum(mask) != 0:
        avg_rmse = torch.sqrt(torch.mean((labels[mask] - preds[mask]) ** 2))
        return avg_rmse
    else:
        return torch.tensor(0)

def mape(preds, labels, mask):
    if torch.sum(mask) != 0:
        zero_positions = (labels == 0)
        mask = torch.logical_and(mask, ~zero_positions)
        mask = ~zero_positions
        avg_mape = torch.mean(torch.abs(labels[mask] - preds[mask]) / labels[mask])
        return avg_mape
    else:
        return torch.tensor(0)
    
# def mae_weight(preds, labels, mask, a=1, r=1):
#     """
#     Args:
#         preds: 预测值 (batch_size, num_steps, ...)
#         labels: 真实值 (batch_size, num_steps, ...)
#         mask: 掩码 (batch_size, num_steps, ...)
#         a: 等比数列首项，预测第一步的权重
#         r: 等比数列公比，控制权重的递减速率
#     """
#     assert preds.shape == labels.shape == mask.shape
#     num_steps = preds.size(1)
#     weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)
#     # weights = weights.view(1, num_steps, 1)
    
#     if torch.sum(mask) != 0:
#         abs_diff = torch.abs(labels - preds)
#         # weighted_diff = abs_diff * weights * mask
#         # avg_mae = torch.sum(weighted_diff) / torch.sum(mask)
#         mae_per_step = torch.sum(abs_diff * mask, dim=(0, 2)) / torch.sum(mask, dim=(0, 2))  # 每步的 MAE
#         weighted_mae = mae_per_step * weights  # 加权后的 MAE
#         avg_mae = torch.sum(weighted_mae) / torch.sum(weights)
#         return avg_mae
#     else:
#         return torch.tensor(0.0, device=preds.device)

def mae_weight(preds, labels, mask=None, a=1, r=1):
    """
    加权 MAE 计算，可选 mask。

    Args:
        preds: 预测值 (B, T, N)
        labels: 真实值 (B, T, N)
        mask: 掩码 (B, T, N)，可为 None（表示全参与）
        a: 权重的首项
        r: 等比公比，控制时间步的衰减权重

    Returns:
        加权 MAE（标量）
    """
    assert preds.shape == labels.shape
    if mask is None:
        mask = torch.ones_like(preds)

    num_steps = preds.size(1)
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)  # (T,)

    if torch.sum(mask) != 0:
        abs_diff = torch.abs(labels - preds)  # (B, T, N)
        mae_per_step = torch.sum(abs_diff * mask, dim=(0, 2)) / torch.sum(mask, dim=(0, 2))  # (T,)
        weighted_mae = mae_per_step * weights
        avg_mae = torch.sum(weighted_mae) / torch.sum(weights)
        return avg_mae
    else:
        return torch.tensor(0.0, device=preds.device)


def rmse_weight(preds, labels, mask, a=1, r=1):
    """
    Args:
        preds: 预测值 (batch_size, num_steps, ...)
        labels: 真实值 (batch_size, num_steps, ...)
        mask: 掩码 (batch_size, num_steps, ...)
        a: 等比数列首项，预测第一步的权重
        r: 等比数列公比，控制权重的递减速率
    """
    assert preds.shape == labels.shape == mask.shape
    num_steps = preds.size(1)
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)
    # weights = weights.view(1, num_steps, 1)
    
    if torch.sum(mask) != 0:
        squared_diff = (labels - preds) ** 2
        # weighted_squared_diff = squared_diff * weights * mask
        # avg_rmse = torch.sqrt(torch.sum(weighted_squared_diff) / torch.sum(mask))
        rmse_per_step = torch.sqrt(torch.sum(squared_diff * mask, dim=(0, 2)) / torch.sum(mask, dim=(0, 2)))  # 每步的 RMSE
        weighted_rmse = rmse_per_step * weights  # 加权后的 RMSE
        avg_rmse = torch.sum(weighted_rmse) / torch.sum(weights)
        return avg_rmse
    else:
        return torch.tensor(0.0, device=preds.device)


def mape_weight(preds, labels, mask, a=1, r=1):
    """
    Args:
        preds: 预测值 (batch_size, num_steps, ...)
        labels: 真实值 (batch_size, num_steps, ...)
        mask: 掩码 (batch_size, num_steps, ...)
        a: 等比数列首项，预测第一步的权重
        r: 等比数列公比，控制权重的递减速率
    """
    assert preds.shape == labels.shape == mask.shape
    num_steps = preds.size(1)
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)
    # weights = weights.view(1, num_steps, 1)
    
    if torch.sum(mask) != 0:
        zero_positions = (labels == 0)
        mask = torch.logical_and(mask, ~zero_positions)  # 过滤掉 labels 为 0 的位置
        abs_diff = torch.abs(labels - preds)
        relative_diff = abs_diff / torch.clamp(labels, min=1e-8)  # 防止除以 0
        mape_per_step = torch.sum(relative_diff * mask, dim=(0, 2)) / torch.sum(mask, dim=(0, 2))  # 每步的 MAPE
        weighted_mape = mape_per_step * weights  # 加权后的 MAPE
        avg_mape = torch.sum(weighted_mape) / torch.sum(weights)
        return avg_mape
    else:
        return torch.tensor(0.0, device=preds.device)

def mae_rlerror(preds, labels):# 维度是（batchsize， nodenum， 4）
    
    assert preds.shape == labels.shape
    
    abs_diff = torch.abs(labels - preds)
    
    mae = torch.mean(abs_diff, dim = 2)
    
    # mae = torch.sum(mae, dim = 2)
    
    return mae

def rmse_rlerror(preds, labels):
    """
    Args:
        preds: 预测值 (batch_size, ...)
        labels: 真实值 (batch_size, ...)
    Returns:
        每个样本的 RMSE (batch_size,)
    """
    assert preds.shape == labels.shape
    
    # 计算平方误差
    squared_diff = (labels - preds) ** 2
    
    # 按样本计算 MSE，忽略 batch_size 之外的所有维度
    mse = torch.mean(squared_diff, dim=1)
    mse = torch.sum(mse, dim=1)
    
    # 计算 RMSE
    rmse = torch.sqrt(mse)
    
    return rmse

def rmse_rlerror(preds, labels):
    """
    Args:
        preds: 预测值 (batch_size, ...)
        labels: 真实值 (batch_size, ...)
    Returns:
        每个样本的 RMSE (batch_size,)
    """
    assert preds.shape == labels.shape
    
    # 计算平方误差
    squared_diff = (labels - preds) ** 2
    
    # 按样本计算 MSE，忽略 batch_size 之外的所有维度
    mse_per_sample = torch.mean(squared_diff, dim=list(range(1, squared_diff.dim())))
    
    # 对 MSE 取平方根，得到 RMSE
    rmse_per_sample = torch.sqrt(mse_per_sample)
    
    return rmse_per_sample

def metric1(pred, real, threshold):
    m = mae(pred, real, threshold).item()
    r = rmse(pred, real, threshold).item()
    p = mape(pred, real, threshold).item()
    return m, r, p

def getCrossloss(tensor):
    # 获取 tensor 的梯度
    grad = tensor.grad

    # 选择前 30% 梯度最大的索引
    num_top_regions = int(0.3 * len(grad))  # 计算前 30% 的数量
    top_indices = torch.topk(grad.abs(), num_top_regions).indices

    # 创建一个 mask，只保留前 30% 梯度最大的位置
    mask = torch.zeros_like(tensor)
    mask[top_indices] = 1

    # 计算新损失，只在前 30% 梯度最大的区域优化
    loss = ((tensor * mask) ** 2).sum()

    return loss
def rmse_per_node(preds, labels, mask, a=1, r=0.5):
    """
    Args:
        preds: 预测值 (batch_size, num_steps, num_nodes)
        labels: 真实值 (batch_size, num_steps, num_nodes)
        mask: 掩码 (batch_size, num_steps, num_nodes)
        a: 等比数列首项，预测第一步的权重
        r: 等比数列公比，控制权重的递减速率
    Returns:
        每个节点的 RMSE (num_nodes,)
    """
    assert preds.shape == labels.shape == mask.shape
    num_steps = preds.size(1)
    num_nodes = preds.size(2)
    
    # 计算权重
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)  # (num_steps,)
    weights = weights.view(num_steps, 1)  # (1, num_steps, 1) 以便进行广播

    if torch.sum(mask) != 0:
        # 计算平方差
        squared_diff = (labels - preds) ** 2  # (batch_size, num_steps, num_nodes)
        
        # 计算每个节点在每个时间步的加权 RMSE
        rmse_per_step = torch.sqrt(torch.sum(squared_diff * mask, dim=0) / torch.sum(mask, dim=0))  # (num_steps, num_nodes)
        weighted_rmse = rmse_per_step * weights  # (num_steps, num_nodes)
        
        # 对时间步加权求和，得到每个节点的最终 RMSE
        avg_rmse_per_node = torch.sum(weighted_rmse, dim=0) / torch.sum(weights)
        return avg_rmse_per_node  # 返回每个节点的 RMSE (num_nodes,)
    else:
        return torch.zeros(num_nodes, device=preds.device)
    
def getMaskloss(conf, grad):
    """
    Args:
        samplemask: 输入的样本掩码 Tensor (大小为 231)
        loss: 损失值 Tensor (大小为 231)
        topnum: 从 loss 中选择的前 topnum 个索引来计算损失
    Returns:
        根据 top_indices 计算的 mask loss
    """
    # 通过 loss 的前 topnum 个最大的索引
    # grad = grad.view(32,24,154,-1)
    # 对 0, 1, 3 这三个维度进行求和
    grad = grad.sum(dim=(0, 1))
    q = grad.quantile(1/2, dim=-1, keepdim=True)
    label = (grad >= q).float() 
    # label_true_indices = mask_indices[grad_indices]
    # true_count = torch.sum(grad_indices)
    # otherweight = (154 - true_count)/(231-true_count)
    # otherweight = 0
    
    # label = torch.zeros(231, dtype=torch.float32, device='cuda')
    # label[true_indices==1] = 1.0
    # label[true_indices==0] = 0.0

    # 创建一个布尔掩码，将所有值初始化为 True
    # label_wrong_indices = torch.ones(231, dtype=torch.bool)

    # 将需要排除的索引位置设置为 False
    # label_wrong_indices[label_true_indices] = False
    # label[label_wrong_indices] = otherweight

    # criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    loss = criterion(conf, label)
    
    return loss

# def metric(pred, real):
#     mae = masked_mae(pred, real, 0.0).item()
#     mape = masked_mape(pred, real, 0.0).item()
#     rmse = masked_rmse(pred, real, 0.0).item()
#     return mae, mape, rmse


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def metric1_per_node(pred, real, threshold):
    # Initialize lists to store metrics for each node
    mae_per_node = []
    rmse_per_node = []
    mape_per_node = []

    # Iterate over each node in the tensor (assuming the second dimension represents nodes)
    for node_index in range(pred.shape[1]):
        # Extract the predictions and real values for the current node
        pred_node = pred[:, node_index]
        real_node = real[:, node_index]

        # Compute metrics for the current node
        mae_node = mae(pred_node, real_node, threshold)
        rmse_node = rmse(pred_node, real_node, threshold)
        mape_node = mape(pred_node, real_node, threshold)

        # Store the metrics
        mae_per_node.append(mae_node.item())
        rmse_per_node.append(rmse_node.item())
        mape_per_node.append(mape_node.item())

    # Convert lists to tensors for easier handling
    mae_per_node = torch.tensor(mae_per_node)
    rmse_per_node = torch.tensor(rmse_per_node)
    mape_per_node = torch.tensor(mape_per_node)

    return mae_per_node, rmse_per_node, mape_per_node