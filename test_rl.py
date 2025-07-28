# Modified testing script to evaluate MAB-based policy network
import os
import torch
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models.engine_rl import TrainerMAB
from utils.mdtp import MyDataset_nstponline, set_seed
from utils.mdtp import metric1
import yaml
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--candidate_lengths', type=str, default='4,8,12,16,24')
parser.add_argument('--pred_size', type=int, default=4)
parser.add_argument('--node_num', type=int, default=231)
parser.add_argument('--in_features', type=int, default=2)
parser.add_argument('--out_features', type=int, default=16)
parser.add_argument('--lstm_features', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--gradient_clip', type=int, default=5)
parser.add_argument('--bike_base_path', type=str, default='./data/nyc/bike')
parser.add_argument('--taxi_base_path', type=str, default='./data/nyc/taxi')
parser.add_argument('--seed', type=int, default=99)
parser.add_argument('--smoe_start_epoch', type=int, default=99)
parser.add_argument('--save', type=str, default='./checkpoints/exp_mab/')
parser.add_argument('--gpus', type=str, default='0')
args = parser.parse_args()

args.candidate_lengths = list(map(int, args.candidate_lengths.split(',')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = yaml.safe_load(open('config.yml'))

def custom_collate_fn(batch):
    return batch

@torch.no_grad()
def main():
    set_seed(args.seed)

    # load test data
    bike_volume_path = os.path.join(args.bike_base_path, 'BV_test.npy')
    taxi_volume_path = os.path.join(args.taxi_base_path, 'TV_test.npy')
    bike_adj_path = os.path.join(args.bike_base_path, 'BF_test.npy')
    taxi_adj_path = os.path.join(args.taxi_base_path, 'TF_test.npy')

    bike_data = MyDataset_nstponline(bike_volume_path, max(args.candidate_lengths), args.batch_size, args.pred_size)
    taxi_data = MyDataset_nstponline(taxi_volume_path, max(args.candidate_lengths), args.batch_size, args.pred_size)
    bike_adj = MyDataset_nstponline(bike_adj_path, max(args.candidate_lengths), args.batch_size, args.pred_size)
    taxi_adj = MyDataset_nstponline(taxi_adj_path, max(args.candidate_lengths), args.batch_size, args.pred_size)

    bike_loader = DataLoader(dataset=bike_data, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    taxi_loader = DataLoader(dataset=taxi_data, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    bike_adj_loader = DataLoader(dataset=bike_adj, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    taxi_adj_loader = DataLoader(dataset=taxi_adj, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    # 构造模型引擎并加载策略网络权重
    engine = TrainerMAB(args.batch_size, args.candidate_lengths, args.node_num, args.in_features, args.out_features,
                        args.lstm_features, device, args.learning_rate, args.weight_decay, args.gradient_clip,
                        args.smoe_start_epoch, args.pred_size)

    policy_path = os.path.join(args.save, 'best_policy.pth')
    if not os.path.exists(policy_path):
        print(f"Policy file not found: {policy_path}")
        return
    engine.policy_net.load_state_dict(torch.load(policy_path, map_location=device))
    engine.policy_net.eval()

    # 遍历测试集，评估测试 loss
    total_loss = 0.0
    total_batches = 0
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
    engine.count = 0  # Reset count for testing
    total_time = 0.0    
    for iter, (bike_adj, bike_node, taxi_adj, taxi_node) in enumerate(zip(bike_adj_loader, bike_loader, taxi_adj_loader, taxi_loader)):
        bike_in, bike_out = bike_node
        taxi_in, taxi_out = taxi_node
        test_x = (bike_in.to(device), bike_adj[0].to(device), taxi_in.to(device), taxi_adj[0].to(device))
        bike_out, taxi_out = (bike_out.permute(3, 0, 1, 2).to(device), taxi_out.permute(3, 0, 1, 2).to(device))
        test_y = (bike_out, taxi_out)

        t1 = time.time()
        bike_start, bike_end, taxi_start, taxi_end = engine.test_online(test_x, test_y)  # 使用你之前定义的 test() 方法
        t2 = time.time()
        total_time += (t2 - t1)
        bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(0).unsqueeze(0), bike_end.unsqueeze(0).unsqueeze(0), taxi_start.unsqueeze(0).unsqueeze(0), taxi_end.unsqueeze(0).unsqueeze(0)
        want = 0  
        bk_start_mask = bike_out[0][:, want, :] != bike_start[:, want, :]
        bk_end_mask = bike_out[1][:, want, :] != bike_end[:, want, :]
        tx_start_mask = taxi_out[0][:, want, :] != taxi_start[:, want, :]
        tx_end_mask = taxi_out[1][:, want, :] != taxi_end[:, want, :]

        bike_start_metrics = metric1(bike_start[:, want, :], bike_out[0][:, want, :], bk_start_mask)
        bike_end_metrics = metric1(bike_end[:, want, :], bike_out[1][:, want, :], bk_end_mask)
        taxi_start_metrics = metric1(taxi_start[:, want, :], taxi_out[0][:, want, :], tx_start_mask)
        taxi_end_metrics = metric1(taxi_end[:, want, :], taxi_out[1][:, want, :], tx_end_mask)


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
    log1 = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
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

    from collections import Counter
    counter = Counter(engine.actions)
    print(counter)
    print(f"Total time taken for testing: {total_time:.2f} seconds")  
    print(engine.gcntime)  

if __name__ == '__main__':
    main()