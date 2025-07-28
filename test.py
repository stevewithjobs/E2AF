import json
import time
import os
import yaml
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils.mdtp import MyDataset, metric1, MyDataset_nstp, MyDataset_nstponline
from models.model import Net_timesnet_sample_onetimesnet
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
import torch
import functools
import queue



# these parameter settings should be consistent with train.py
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='GPU setting')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--window_size', type=int, default=24, help='window size')
parser.add_argument('--input_size', type=int, default=16, help='input size')
parser.add_argument('--pred_size', type=int, default=4, help='pred size')
parser.add_argument('--node_num', type=int, default=231, help='number of node to predict')
parser.add_argument('--in_features', type=int, default=2, help='GCN input dimension')
parser.add_argument('--out_features', type=int, default=16, help='GCN output dimension')
parser.add_argument('--lstm_features', type=int, default=256, help='LSTM hidden feature size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=20, help='epoch')
parser.add_argument('--gradient_clip', type=int, default=5, help='gradient clip')
parser.add_argument('--pad', type=bool, default=False, help='whether padding with last batch sample')
parser.add_argument('--bike_base_path', type=str, default='./data/nyc/bike', help='bike data path')
parser.add_argument('--taxi_base_path', type=str, default='./data/nyc/taxi', help='taxi data path')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./checkpoints/exp_nyc_16/best_model.pth', help='save path')

args = parser.parse_args()
config = yaml.safe_load(open('config.yml'))

def custom_collate_fn(batch):
    # 因为 batch 是一个长度为 1 的列表，直接取出这个列表的第一个元素
    return batch

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

def visualize_expert_outputs(model, bike_node, bike_adj, taxi_node, taxi_adj):
    experts_out = []

    # 获取6个 expert 的输出
    with torch.no_grad():
        for g in model.bike_gcn:
            out = g(bike_node, bike_adj)  # (B, T, N, F)
            experts_out.append(out)
        for g in model.taxi_gcn:
            out = g(taxi_node, taxi_adj)
            experts_out.append(out)

    # batch size 必须为1
    assert bike_node.shape[0] == 1, "可视化建议设置 batch_size=1"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(experts_out):
        feat_2d = feat[0].reshape(-1, feat.shape[-1]).cpu().numpy()  # (T*N, F)
        reduced = PCA(n_components=2).fit_transform(feat_2d)
        axes[i].scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        axes[i].set_title(f'Expert {i} ({"Bike" if i < 3 else "Taxi"})')
        axes[i].axis('off')

    plt.suptitle("GCN Expert Node Embedding PCA", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"expert_outputs_iter{iter}.png")  # 保存结果
    plt.show()

def plot_all_expert_response_heatmaps(expert_outputs, grid_shape=(11, 21), save_path='heatmaps/all_expert_heatmap.png'):
    """
    expert_outputs: List of Tensor, each with shape (B, T, D, F)
    grid_shape: 2D grid shape (e.g., 11x12)
    save_path: where to save the combined figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_experts = len(expert_outputs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, expert_out in enumerate(expert_outputs):
        assert expert_out.shape[0] == 1, "仅支持 batch size=1"
        feat = expert_out[0, -1]  # shape: (D, F)
        feat_mean = feat.mean(dim=-1).cpu().numpy()
        feat_2d = feat_mean.reshape(grid_shape)

        ax = axes[i]
        im = ax.imshow(feat_2d, cmap='viridis')
        ax.set_title(f'Expert {i} Response')
        ax.axis('off')

        # 单独加 colorbar（不每张图加）
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("GCN Experts Node Response Heatmaps", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all_expert_routing_maps(routing_weights, grid_shape=(11, 21), save_dir='heatmaps'):
    """
    routing_weights: Tensor (B, T, D, E) — softmax 权重（每个 expert 的概率）
    每张图是一个 expert 在空间区域 (D=11×21) 上的 routing 权重图。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    B, T, D, E = routing_weights.shape
    assert B == 1, "只支持 batch_size=1"
    
    # 获取最后一个时间步
    weights = routing_weights[0, -1]  # shape: (D, E)

    for e in range(E):
        expert_map = weights[:, e].view(grid_shape).cpu().numpy()  # shape: (11, 21)

        plt.figure(figsize=(6, 5))
        plt.imshow(expert_map, cmap='viridis')
        plt.colorbar()
        plt.title(f"Expert {e} Routing Weight")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/routing_weight_expert_{e}.png", dpi=200)
        plt.close()

def plot_routing_indices_map(routing_indices, grid_shape=(11, 21), save_path='heatmaps/routing_indices_map.png'):
    """
    routing_indices: Tensor of shape (B, T, D) — 每个位置选中的 expert 编号（int）
    grid_shape: 2D reshape target
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    assert routing_indices.shape[0] == 1, "建议 batch_size = 1"
    indices = routing_indices[0, -1]  # shape: (D,)
    indices_2d = indices.view(grid_shape).cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(indices_2d, cmap='tab10')  # 每个 expert 编号一个颜色
    plt.colorbar()
    plt.title("Routing Indices (Selected Expert per Node)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    device = torch.device(args.device)

    bikevolume_test_save_path = os.path.join(args.bike_base_path, 'BV_test.npy')
    bikeflow_test_save_path = os.path.join(args.bike_base_path, 'BF_test.npy')
    taxivolume_test_save_path = os.path.join(args.taxi_base_path, 'TV_test.npy')
    taxiflow_test_save_path = os.path.join(args.taxi_base_path, 'TF_test.npy')

    bike_test_data = MyDataset_nstponline(bikevolume_test_save_path, args.window_size, args.batch_size, args.pred_size)
    taxi_test_data = MyDataset_nstponline(taxivolume_test_save_path, args.window_size, args.batch_size, args.pred_size)
    bike_adj_data = MyDataset_nstponline(bikeflow_test_save_path, args.window_size, args.batch_size, args.pred_size)
    taxi_adj_data = MyDataset_nstponline(taxiflow_test_save_path, args.window_size, args.batch_size, args.pred_size)

    bike_test_loader = DataLoader(
        dataset=bike_test_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )
    taxi_test_loader = DataLoader(
        dataset=taxi_test_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn ,
        num_workers=0,
    )
    bike_adj_loader = DataLoader(
        dataset=bike_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )
    taxi_adj_loader = DataLoader(
        dataset=taxi_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )

    base_smoe_config = SpatialMoEConfig(
            in_planes=2,
            out_planes=3,
            num_experts=6,
            gate_block=functools.partial(SpatialLatentTensorGate2d,
                                node_num = args.node_num),
            save_error_signal=True,
            dampen_expert_error=True,
            unweighted=True,
            block_gate_grad=True,
        )
    # model = Net_timesnet_onetimesnet(args.batch_size, args.window_size, args.node_num, args.in_features, args.out_features, args.lstm_features, base_smoe_config, args.pred_size)
    model = Net_timesnet_sample_onetimesnet(args.batch_size, args.input_size, args.node_num, args.in_features, args.out_features, args.lstm_features, base_smoe_config, args.pred_size)

    model.to(device)
    model.load_state_dict(torch.load(args.save),strict=False)
    model.eval()
    print('model load successfully!')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')

    bike_start_loss = []
    bike_end_loss = []
    bike_start_rmse = []
    bike_end_rmse = []
    bike_start_mape = []
    bike_end_mape = []
    taxi_start_loss = []
    taxi_end_loss = []
    taxi_start_rmse = []
    taxi_end_rmse = []
    taxi_start_mape = []
    taxi_end_mape = []
    

    

    bike_start_que = queue.Queue()
    bike_end_que = queue.Queue()
    taxi_start_que = queue.Queue()
    taxi_end_que = queue.Queue()
    pass_flag = False
    count = 0 
    tmp = 0
    total_time = 0
    # torch.set_num_threads(1)
    for iter, (bike_adj, bike_node, taxi_adj, taxi_node) in enumerate(zip(bike_adj_loader, bike_test_loader,
                                                                              taxi_adj_loader, taxi_test_loader)):
         
        # if iter % 5 != 0:
        #     continue
        # if bike_start_que.empty() and pass_flag:
        #     print(1)
        bike_in_shots, bike_out_shots = bike_node
        bike_adj = bike_adj[0]
        bike_out_shots = bike_out_shots.permute(3, 0, 1, 2)
        taxi_in_shots, taxi_out_shots = taxi_node
        taxi_adj = taxi_adj[0]
        taxi_out_shots = taxi_out_shots.permute(3, 0, 1, 2)
        bike_in_shots, bike_out_shots, bike_adj = bike_in_shots.to(device), bike_out_shots.to(device), bike_adj.to(
            device)
        taxi_in_shots, taxi_out_shots, taxi_adj = taxi_in_shots.to(device), taxi_out_shots.to(device), taxi_adj.to(
            device)
        test_x = (bike_in_shots[:,-args.input_size:,:,:], bike_adj[:,-args.input_size:,:,:], taxi_in_shots[:,-args.input_size:,:,:], taxi_adj[:,-args.input_size:,:,:])
        
        # if iter == 0:  # 或者你想可视化前几轮都可以
        #     visualize_expert_outputs(model, bike_in_shots[:1], bike_adj[:1], taxi_in_shots[:1], taxi_adj[:1])
            
            # 输出专家响应图
        # if not bike_start_que.empty():
        #     bike_start = bike_start_que.get()
        #     bike_end = bike_end_que.get()
        #     taxi_start =taxi_start_que.get()
        #     taxi_end = taxi_end_que.get()

        #     # if torch.equal(tmp, bike_in_shots[:, -1, :, 0]):
        #     #     print(1)
            
        #     bk_start_mask = bike_in_shots[:, -1, :, 0] != bike_start
        #     bk_end_mask = bike_in_shots[:, -1, :, 1] != bike_end
        #     tx_start_mask = taxi_in_shots[:, -1, :, 0] != taxi_start
        #     tx_end_mask = taxi_in_shots[:, -1, :, 1]!= taxi_end

        #     # mask = torch.ones_like(bike_start, dtype=torch.bool)

        #     bike_start_metrics = metric1(bike_start, bike_in_shots[:, -1, :, 0], bk_start_mask)
        #     bike_end_metrics = metric1(bike_end, bike_in_shots[:, -1, :, 1], bk_end_mask)
        #     taxi_start_metrics = metric1(taxi_start, taxi_in_shots[:, -1, :, 0], tx_start_mask)
        #     taxi_end_metrics = metric1(taxi_end, taxi_in_shots[:, -1, :, 1], tx_end_mask)

        #     # if (bike_start_metrics[1]*config['bike_volume_max'] >6 and\
        #     #     bike_end_metrics[1]*config['bike_volume_max'] > 8) or \
        #     #     (taxi_start_metrics[1]*config['taxi_volume_max'] > 22 and\
        #     #     taxi_end_metrics[1]*config['taxi_volume_max'] > 27):
        #     #     pass_flag = False
        #     # else:
        #     #     pass_flag = True
        #     #     count += 1

        #     # if bike_start_que.qsize() < 3:
        #     #     pass_flag = False
        #     # else:
        #     #     pass_flag = True
        #     #     count += 1
        #     pass_flag = False  
        #     count += 1
            
        # else:
        #     pass_flag = False
            

        # if not pass_flag:
        #     # count += 1
        #     with torch.no_grad():
        #         bike_start, bike_end, taxi_start, taxi_end = model(test_x)

        #     while not bike_start_que.empty():
        #         bike_start_que.get()
        #         bike_end_que.get()
        #         taxi_start_que.get()
        #         taxi_end_que.get()

        #     for i in range(args.pred_size):  # 遍历dim=1的所有索引
        #         bike_start_que.put(bike_start[:, i, :]) 
        #         bike_end_que.put(bike_end[:, i, :])
        #         taxi_start_que.put(taxi_start[:, i, :])
        #         taxi_end_que.put(taxi_end[:, i, :])

        #     bike_start = bike_start_que.get()
        #     bike_end = bike_end_que.get()
        #     taxi_start = taxi_start_que.get()
        #     taxi_end = taxi_end_que.get()
        #     pass_flag = False

        # # # tmp = bike_out_shots[0, :, 0, :]
        # bk_start_mask = bike_out_shots[0, :, 0, :] != bike_start
        # bk_end_mask = bike_out_shots[1, :, 0, :] != bike_end
        # tx_start_mask = taxi_out_shots[0, :, 0, :] != taxi_start
        # tx_end_mask = taxi_out_shots[1, :, 0, :] != taxi_end

        # # mask = torch.ones_like(bike_start, dtype=torch.bool)

        # bike_start_metrics = metric1(bike_start, bike_out_shots[0, :, 0, :], bk_start_mask)
        # bike_end_metrics = metric1(bike_end, bike_out_shots[1, :, 0, :], bk_end_mask)
        # taxi_start_metrics = metric1(taxi_start, taxi_out_shots[0, :, 0, :], tx_start_mask)
        # taxi_end_metrics = metric1(taxi_end, taxi_out_shots[1, :, 0, :], tx_end_mask)

        with torch.no_grad():
            t1 = time.time()
            bike_start, bike_end, taxi_start, taxi_end = model(test_x)

            # # 提取 6 个 expert 输出（你已有这部分）
            # bike_experts = [g(bike_in_shots, bike_adj) for g in model.bike_gcn]
            # taxi_experts = [g(taxi_in_shots, taxi_adj) for g in model.taxi_gcn]
            # all_experts = bike_experts + taxi_experts

            # plot_all_expert_response_heatmaps(all_experts, grid_shape=(11, 21))

            # # 如果你用了 MoE 路由器
            # if hasattr(model.smoe.smoe, "routing_weights") and model.smoe.smoe.routing_weights is not None:
            #     plot_all_expert_routing_maps(model.smoe.smoe.routing_weights, grid_shape=(11, 21))
            # if hasattr(model.smoe.smoe, "routing_indices") and model.smoe.smoe.routing_indices is not None:
            #     top1_indices = model.smoe.smoe.routing_indices[..., 0]  # shape: (B, T, D)
            #     plot_routing_indices_map(top1_indices, grid_shape=(11, 21))
            t2 = time.time()
            if iter > 1:
                total_time += t2 - t1
        

        # if iter != 0:
        #     bk_start_mask = h_bike_start != bike_start
        #     bk_end_mask = h_bike_end != bike_end
        #     tx_start_mask = h_taxi_start != taxi_start
        #     tx_end_mask = h_taxi_end != taxi_end
            
        #     bike_start_metrics = metric1(bike_start, h_bike_start, bk_start_mask)
        #     bike_end_metrics = metric1(bike_end, h_bike_end, bk_end_mask)
        #     taxi_start_metrics = metric1(taxi_start, h_taxi_start, tx_start_mask)
        #     taxi_end_metrics = metric1(taxi_end, h_taxi_end, tx_end_mask)
        #     print(h_bike_real)
        #     print(bike_out_shots[0])
        #     h_bike_start, h_bike_end, h_taxi_start, h_taxi_end = bike_start, bike_end, taxi_start, taxi_end
        # else:
        #     h_bike_real =  bike_out_shots[0]
        #     h_bike_start, h_bike_end, h_taxi_start, h_taxi_end = bike_start, bike_end, taxi_start, taxi_end
        #     continue
        # if iter > 100:
        #     break
        
        want = 0  
        bk_start_mask = bike_out_shots[0][:, want, :] != bike_start[:, want, :]
        bk_end_mask = bike_out_shots[1][:, want, :] != bike_end[:, want, :]
        tx_start_mask = taxi_out_shots[0][:, want, :] != taxi_start[:, want, :]
        tx_end_mask = taxi_out_shots[1][:, want, :] != taxi_end[:, want, :]

        bike_start_metrics = metric1(bike_start[:, want, :], bike_out_shots[0][:, want, :], bk_start_mask)
        bike_end_metrics = metric1(bike_end[:, want, :], bike_out_shots[1][:, want, :], bk_end_mask)
        taxi_start_metrics = metric1(taxi_start[:, want, :], taxi_out_shots[0][:, want, :], tx_start_mask)
        taxi_end_metrics = metric1(taxi_end[:, want, :], taxi_out_shots[1][:, want, :], tx_end_mask)

        # bike_threshold = config['threshold'] / config['bike_volume_max']
        # taxi_threshold = config['threshold'] / config['taxi_volume_max']

        # bike_start_metrics = metric1(bike_start[:, want, :], bike_out_shots[0][:, want, :], bike_threshold)
        # bike_end_metrics = metric1(bike_end[:, want, :], bike_out_shots[1][:, want, :], bike_threshold)
        # taxi_start_metrics = metric1(taxi_start[:, want, :], taxi_out_shots[0][:, want, :], taxi_threshold)
        # taxi_end_metrics = metric1(taxi_end[:, want, :], taxi_out_shots[1][:, want, :], taxi_threshold)

        # mask = torch.ones_like(bike_out_shots[0][:, want, :], dtype=torch.bool)

        # bike_start_metrics = metric1(bike_start[:, want, :], bike_out_shots[0][:, want, :], mask)
        # bike_end_metrics = metric1(bike_end[:, want, :], bike_out_shots[1][:, want, :], mask)
        # taxi_start_metrics = metric1(taxi_start[:, want, :], taxi_out_shots[0][:, want, :], mask)
        # taxi_end_metrics = metric1(taxi_end[:, want, :],  bike_out_shots[1][:, want, :], mask)


        # bk_start_mask = bike_out_shots[0] != bike_start
        # bk_end_mask = bike_out_shots[1] != bike_end
        # tx_start_mask = taxi_out_shots[0] != taxi_start
        # tx_end_mask = taxi_out_shots[1] != taxi_end

        # bike_start_metrics = metric1(bike_start, bike_out_shots[0], bk_start_mask)
        # bike_end_metrics = metric1(bike_end, bike_out_shots[1], bk_end_mask)
        # taxi_start_metrics = metric1(taxi_start, taxi_out_shots[0], tx_start_mask)
        # taxi_end_metrics = metric1(taxi_end, taxi_out_shots[1], tx_end_mask)


        bike_start_loss.append(bike_start_metrics[0])
        bike_end_loss.append(bike_end_metrics[0])
        bike_start_rmse.append(bike_start_metrics[1])
        bike_end_rmse.append(bike_end_metrics[1])
        bike_start_mape.append(bike_start_metrics[2])
        bike_end_mape.append(bike_end_metrics[2])

        taxi_start_loss.append(taxi_start_metrics[0])
        taxi_end_loss.append(taxi_end_metrics[0])
        taxi_start_rmse.append(taxi_start_metrics[1])
        taxi_end_rmse.append(taxi_end_metrics[1])
        taxi_start_mape.append(taxi_start_metrics[2])
        taxi_end_mape.append(taxi_end_metrics[2])
        log = 'Iter: {:02d}\nTest Bike Start MAE: {:.4f}, Test Bike End MAE: {:.4f}, ' \
              'Test Taxi Start MAE: {:.4f}, Test Taxi End MAE: {:.4f}, \n' \
              'Test Bike Start RMSE: {:.4f}, Test Bike End RMSE: {:.4f}, ' \
              'Test Taxi Start RMSE: {:.4f}, Test Taxi End RMSE: {:.4f}, \n' \
              'Test Bike Start MAPE: {:.4f}, Test Bike End MAPE: {:.4f}, ' \
              'Test Taxi Start MAPE: {:.4f}, Test Taxi End MAPE: {:.4f}'
        print(
            log.format(iter, bike_start_metrics[0]*config['bike_volume_max'], bike_end_metrics[0]*config['bike_volume_max'], taxi_start_metrics[0]*config['taxi_volume_max'], taxi_end_metrics[0]*config['taxi_volume_max'],
                       bike_start_metrics[1]*config['bike_volume_max'], bike_end_metrics[1]*config['bike_volume_max'], taxi_start_metrics[1]*config['taxi_volume_max'], taxi_end_metrics[1]*config['taxi_volume_max'],
                       bike_start_metrics[2]*100, bike_end_metrics[2]*100, taxi_start_metrics[2]*100, taxi_end_metrics[2]*100, ))
    print("-----------------------------------------------------------------")
    # log1 = 'Average test bike start MAE: {:.4f}, Average test bike end MAE: {:.4f}, ' \
    #        'Average test taxi start MAE: {:.4f}, Average test taxi end MAE: {:.4f}, \n' \
    #        'Average test bike start RMSE: {:.4f}, Average test bike end RMSE: {:.4f}, ' \
    #        'Average test taxi start RMSE: {:.4f}, Average test taxi end RMSE: {:.4f}, \n' \
    #        'Average test bike start MAPE: {:.4f}, Average test bike end MAPE: {:.4f},' \
    #        'Average test taxi start MAPE: {:.4f}, Average test taxi end MAPE: {:.4f}'
    log1 = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'

    # print(log1.format((np.mean(bike_start_loss))*config['bike_volume_max'], (np.mean(bike_end_loss))*config['bike_volume_max'],
    #                   (np.mean(taxi_start_loss))*config['taxi_volume_max'], (np.mean(taxi_end_loss))*config['taxi_volume_max'],
    #                   (np.mean(bike_start_rmse))*config['bike_volume_max'], (np.mean(bike_end_rmse))*config['bike_volume_max'],
    #                   (np.mean(taxi_start_rmse))*config['taxi_volume_max'], (np.mean(taxi_end_rmse))*config['taxi_volume_max'],
    #                   (np.mean(bike_start_mape))*100, (np.mean(bike_end_mape))*100,
    #                   (np.mean(taxi_start_mape))*100, (np.mean(taxi_end_mape))*100))
    print(log1.format(
        (np.mean(taxi_start_loss)) * config['taxi_volume_max'], 
        (np.mean(taxi_start_rmse)) * config['taxi_volume_max'], 
        (np.mean(taxi_start_mape)) * 100,
        
        (np.mean(taxi_end_loss)) * config['taxi_volume_max'], 
        (np.mean(taxi_end_rmse)) * config['taxi_volume_max'], 
        (np.mean(taxi_end_mape)) * 100,

        (np.mean(bike_start_loss)) * config['bike_volume_max'], 
        (np.mean(bike_start_rmse)) * config['bike_volume_max'], 
        (np.mean(bike_start_mape)) * 100,

        (np.mean(bike_end_loss)) * config['bike_volume_max'], 
        (np.mean(bike_end_rmse)) * config['bike_volume_max'], 
        (np.mean(bike_end_mape)) * 100
    ))
    
    print("inference time: ", total_time)
    print(count)


if __name__ == "__main__":
    main()
