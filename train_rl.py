# Modified training script to train MAB-based policy network
import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models.engine_rl import TrainerMAB  # new engine with policy network support
from utils.mdtp import MyDataset_nstponline, set_seed
from models.EarlyStopping import EarlyStopping
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--candidate_lengths', type=str, default='4,8,16,24')
parser.add_argument('--pred_size', type=int, default=4)
parser.add_argument('--node_num', type=int, default=231)
parser.add_argument('--in_features', type=int, default=2)
parser.add_argument('--out_features', type=int, default=16)
parser.add_argument('--lstm_features', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--gradient_clip', type=int, default=5)
parser.add_argument('--bike_base_path', type=str, default='./data/nyc/bike')
parser.add_argument('--taxi_base_path', type=str, default='./data/nyc/taxi')
parser.add_argument('--seed', type=int, default=99)
parser.add_argument('--save', type=str, default='./checkpoints/exp_mab/')
parser.add_argument('--smoe_start_epoch', type=int, default=99)
parser.add_argument('--gpus', type=str, default='0')
args = parser.parse_args()

args.candidate_lengths = list(map(int, args.candidate_lengths.split(',')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_collate_fn(batch):
    return batch

def main():
    set_seed(args.seed)
    # load data
    bike_volume_path = os.path.join(args.bike_base_path, 'BV_train.npy')
    taxi_volume_path = os.path.join(args.taxi_base_path, 'TV_train.npy')
    bike_adj_path = os.path.join(args.bike_base_path, 'BF_train.npy')
    taxi_adj_path = os.path.join(args.taxi_base_path, 'TF_train.npy')

    bike_data = MyDataset_nstponline(bike_volume_path, max(args.candidate_lengths), args.batch_size, args.pred_size)
    taxi_data = MyDataset_nstponline(taxi_volume_path, max(args.candidate_lengths), args.batch_size, args.pred_size)
    bike_adj = MyDataset_nstponline(bike_adj_path, max(args.candidate_lengths), args.batch_size, args.pred_size)
    taxi_adj = MyDataset_nstponline(taxi_adj_path, max(args.candidate_lengths), args.batch_size, args.pred_size)

    bike_loader = DataLoader(dataset=bike_data, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    taxi_loader = DataLoader(dataset=taxi_data, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    bike_adj_loader = DataLoader(dataset=bike_adj, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    taxi_adj_loader = DataLoader(dataset=taxi_adj, batch_size=None, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    engine = TrainerMAB(args.batch_size, args.candidate_lengths, args.node_num, args.in_features, args.out_features,
                        args.lstm_features, device, args.learning_rate, args.weight_decay, args.gradient_clip,
                        args.smoe_start_epoch, args.pred_size)

    best_reward = float('-inf')
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_rewards = []
        for iter, (bike_adj, bike_node, taxi_adj, taxi_node) in enumerate(zip(bike_adj_loader, bike_loader, taxi_adj_loader, taxi_loader)):
            bike_in, bike_out = bike_node
            taxi_in, taxi_out = taxi_node
            train_x = (bike_in, bike_adj[0], taxi_in, taxi_adj[0])
            train_y = (bike_out.permute(3, 0, 1, 2), taxi_out.permute(3, 0, 1, 2))

            loss, reward = engine.train(train_x, train_y)
            epoch_losses.append(loss)
            epoch_rewards.append(reward)
            # print(f"Epoch {epoch:03d} Iter {iter:03d} Loss: {loss:.10f}")
            # print(f"Epoch {epoch:03d} Iter {iter:03d} Reward: {reward:.10f}")

        mean_loss = np.mean(epoch_losses)
        mean_reward = np.mean(epoch_rewards)

        # print(f"Epoch {epoch:03d} Mean Loss: {mean_loss:.10f}")
        print(f"Epoch {epoch:03d} Mean Reward: {mean_reward:.10f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(engine.policy.state_dict(), os.path.join(args.save, 'best_policy.pth'))

if __name__ == '__main__':
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    main()
