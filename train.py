import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import torch
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models.engine import Trainer
from utils.mdtp import MyDataset_nstp, set_seed, MyDataset_nstponline
from models.EarlyStopping import EarlyStopping
import sys
import shutil

parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:4', help='GPU setting')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--window_size', type=int, default=24, help='window size')
parser.add_argument('--pred_size', type=int, default=4, help='pred size')
parser.add_argument('--node_num', type=int, default=231, help='number of node to predict')
parser.add_argument('--in_features', type=int, default=2, help='GCN input dimension')
parser.add_argument('--out_features', type=int, default=16, help='GCN output dimension')
parser.add_argument('--lstm_features', type=int, default=256, help='LSTM hidden feature size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1000, help='epoch')
parser.add_argument('--gradient_clip', type=int, default=5, help='gradient clip')
parser.add_argument('--pad', type=bool, default=False, help='whether padding with last batch sample')
parser.add_argument('--bike_base_path', type=str, default='./data/bike', help='bike data path')
parser.add_argument('--taxi_base_path', type=str, default='./data/taxi', help='taxi data path')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./checkpoints/exp_nyc_24/', help='save path')
parser.add_argument('--smoe_start_epoch', type=int, default=99, help='smoe start epoch')
parser.add_argument('--gpus', type=str, default='4', help='gpu')
parser.add_argument('--log', type=str, default='0.log', help='log name')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_collate_fn(batch):
    # 因为 batch 是一个长度为 1 的列表，直接取出这个列表的第一个元素
    return batch

def main():
    set_seed(args.seed)
    # load train data
    bikevolume_train_save_path = os.path.join(args.bike_base_path, 'BV_train.npy')
    bikeflow_train_save_path = os.path.join(args.bike_base_path, 'BF_train.npy')
    taxivolume_train_save_path = os.path.join(args.taxi_base_path, 'TV_train.npy')
    taxiflow_train_save_path = os.path.join(args.taxi_base_path, 'TF_train.npy')
    
    data_fraction = 1.0
    bike_train_data = MyDataset_nstponline(bikevolume_train_save_path, args.window_size, args.batch_size, args.pred_size, data_fraction)
    taxi_train_data = MyDataset_nstponline(taxivolume_train_save_path, args.window_size, args.batch_size,  args.pred_size, data_fraction)
    bike_adj_data = MyDataset_nstponline(bikeflow_train_save_path, args.window_size, args.batch_size,  args.pred_size, data_fraction)
    taxi_adj_data = MyDataset_nstponline(taxiflow_train_save_path, args.window_size, args.batch_size,  args.pred_size, data_fraction)

    bike_train_loader = DataLoader(
        dataset=bike_train_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    taxi_train_loader = DataLoader(
        dataset=taxi_train_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    bike_adj_loader = DataLoader(
        dataset=bike_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    taxi_adj_loader = DataLoader(
        dataset=taxi_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )

    # load validation data
    bikevolume_valid_save_path = os.path.join(args.bike_base_path, 'BV_val.npy')
    bikeflow_valid_save_path = os.path.join(args.bike_base_path, 'BF_val.npy')
    taxivolume_valid_save_path = os.path.join(args.taxi_base_path, 'TV_val.npy')
    taxiflow_valid_save_path = os.path.join(args.taxi_base_path, 'TF_val.npy')

    bike_valid_data = MyDataset_nstponline(bikevolume_valid_save_path, args.window_size, args.batch_size, args.pred_size)
    taxi_valid_data = MyDataset_nstponline(taxivolume_valid_save_path, args.window_size, args.batch_size, args.pred_size)
    bike_valid_adj_data = MyDataset_nstponline(bikeflow_valid_save_path, args.window_size, args.batch_size, args.pred_size)
    taxi_valid_adj_data = MyDataset_nstponline(taxiflow_valid_save_path, args.window_size, args.batch_size, args.pred_size)

    bike_valid_loader = DataLoader(
        dataset=bike_valid_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    taxi_valid_loader = DataLoader(
        dataset=taxi_valid_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    bike_valid_adj_loader = DataLoader(
        dataset=bike_valid_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    taxi_valid_adj_loader = DataLoader(
        dataset=taxi_valid_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )

    # train the model
    engine = Trainer(args.batch_size, args.window_size, args.node_num, args.in_features, args.out_features,
                     args.lstm_features, device, args.learning_rate, args.weight_decay, args.gradient_clip, args.smoe_start_epoch, args.pred_size)
    print("start training...", flush=True)
    his_loss = []
    train_time = []
    val_time = []
    early_stopping = EarlyStopping(patience=25, verbose=True)
    lr = 0.001
    wd = lr / 10
    for epoch in range(1, args.epochs + 1):
        train_bike_start_loss = []
        train_bike_end_loss = []
        train_taxi_start_loss = []
        train_taxi_end_loss = []
        train_bike_start_rmse = []
        train_bike_end_rmse = []
        train_taxi_start_rmse = []
        train_taxi_end_rmse = []
        train_bike_start_mape = []
        train_bike_end_mape = []
        train_taxi_start_mape = []
        train_taxi_end_mape = []
        t1 = time.time()
        for iter, (bike_adj, bike_node, taxi_adj, taxi_node) in enumerate(zip(bike_adj_loader, bike_train_loader,
                                                                              taxi_adj_loader, taxi_train_loader)):

            bike_in_shots, bike_out_shots = bike_node
            bike_adj = bike_adj[0]
            # np.save("tmp", bike_adj[0, 0, :, :])
            bike_out_shots = bike_out_shots.permute(3, 0, 1, 2)
            taxi_in_shots, taxi_out_shots = taxi_node
            taxi_adj = taxi_adj[0]
            taxi_out_shots = taxi_out_shots.permute(3, 0, 1, 2)
            bike_in_shots, bike_out_shots, bike_adj = bike_in_shots.to(device), bike_out_shots.to(device), bike_adj.to(
                device)
            taxi_in_shots, taxi_out_shots, taxi_adj = taxi_in_shots.to(device), taxi_out_shots.to(device), taxi_adj.to(
                device)
            train_x = (bike_in_shots, bike_adj, taxi_in_shots, taxi_adj)
            train_y = (bike_out_shots, taxi_out_shots)

            train_x = (bike_in_shots[:, -args.window_size:, :, :].contiguous(), bike_adj[:, -args.window_size:, :, :].contiguous(), 
                       taxi_in_shots[:, -args.window_size:, :, :].contiguous(), taxi_adj[:, -args.window_size:, :, :].contiguous())
            # if you want set you own parameters, delete following default parameter setting code
            lr = 0.001
            if epoch > 60:
                lr = 0.0001
            if epoch > 120:
                lr = 0.00001
            wd = lr / 10
            # if early_stopping.counter > 10:
            #     early_stopping.counter = 0
            #     lr /= 10
            #     wd = lr / 10
            metrics = engine.train(train_x, train_y, lr, wd, iter)
            train_bike_start_loss.append(metrics[0][0])
            train_bike_end_loss.append(metrics[0][1])
            train_taxi_start_loss.append(metrics[0][2])
            train_taxi_end_loss.append(metrics[0][3])
            train_bike_start_rmse.append(metrics[1][0])
            train_bike_end_rmse.append(metrics[1][1])
            train_taxi_start_rmse.append(metrics[1][2])
            train_taxi_end_rmse.append(metrics[1][3])
            train_bike_start_mape.append(metrics[2][0])
            train_bike_end_mape.append(metrics[2][1])
            train_taxi_start_mape.append(metrics[2][2])
            train_taxi_end_mape.append(metrics[2][3])
            log = 'Iter: {:03d}, Train Bike Start Loss: {:.4f}, Train Bike End Loss: {:.4f}, ' \
                  'Train Taxi Start Loss: {:.4f}, Train Taxi End Loss: {:.4f}'
            print(log.format(iter, train_bike_start_loss[-1], train_bike_end_loss[-1],
                             train_taxi_start_loss[-1], train_taxi_end_loss[-1], flush=True))
        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_bike_start_loss = []
        valid_bike_end_loss = []
        valid_taxi_start_loss = []
        valid_taxi_end_loss = []
        valid_bike_start_rmse = []
        valid_bike_end_rmse = []
        valid_taxi_start_rmse = []
        valid_taxi_end_rmse = []
        valid_bike_start_mape = []
        valid_bike_end_mape = []
        valid_taxi_start_mape = []
        valid_taxi_end_mape = []
        s1 = time.time()
        infer_time = 0
        for iter, (bike_valid_adj, bike_valid_node, taxi_valid_adj, taxi_valid_node) in enumerate(
                zip(bike_valid_adj_loader, bike_valid_loader, taxi_valid_adj_loader, taxi_valid_loader)):
            bike_in_shots, bike_out_shots = bike_valid_node
            bike_adj = bike_valid_adj[0]
            bike_out_shots = bike_out_shots.permute(3, 0, 1, 2)
            taxi_in_shots, taxi_out_shots = taxi_valid_node
            taxi_adj = taxi_valid_adj[0]
            taxi_out_shots = taxi_out_shots.permute(3, 0, 1, 2)
            bike_in_shots, bike_out_shots, bike_adj = bike_in_shots.to(device), bike_out_shots.to(device), bike_adj.to(
                device)
            taxi_in_shots, taxi_out_shots, taxi_adj = taxi_in_shots.to(device), taxi_out_shots.to(device), taxi_adj.to(
                device)
            valid_x = (bike_in_shots, bike_adj, taxi_in_shots, taxi_adj)
            valid_y = (bike_out_shots, taxi_out_shots)
            metrics, t= engine.val(valid_x, valid_y, epoch)
            infer_time += t
            valid_bike_start_loss.append(metrics[0][0])
            valid_bike_end_loss.append(metrics[0][1])
            valid_taxi_start_loss.append(metrics[0][2])
            valid_taxi_end_loss.append(metrics[0][3])
            valid_bike_start_rmse.append(metrics[1][0])
            valid_bike_end_rmse.append(metrics[1][1])
            valid_taxi_start_rmse.append(metrics[1][2])
            valid_taxi_end_rmse.append(metrics[1][3])
            valid_bike_start_mape.append(metrics[2][0])
            valid_bike_end_mape.append(metrics[2][1])
            valid_taxi_start_mape.append(metrics[2][2])
            valid_taxi_end_mape.append(metrics[2][3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch, infer_time))
        val_time.append(infer_time)
        mtrain_bike_start_loss = np.mean(train_bike_start_loss)
        mtrain_bike_end_loss = np.mean(train_bike_end_loss)
        mtrain_taxi_start_loss = np.mean(train_taxi_start_loss)
        mtrain_taxi_end_loss = np.mean(train_taxi_end_loss)
        mtrain_bike_start_rmse = np.mean(train_bike_start_rmse)
        mtrain_bike_end_rmse = np.mean(train_bike_end_rmse)
        mtrain_taxi_start_rmse = np.mean(train_taxi_start_rmse)
        mtrain_taxi_end_rmse = np.mean(train_taxi_end_rmse)
        mtrain_bike_start_mape = np.mean(train_bike_start_mape)
        mtrain_bike_end_mape = np.mean(train_bike_end_mape)
        mtrain_taxi_start_mape = np.mean(train_taxi_start_mape)
        mtrain_taxi_end_mape = np.mean(train_taxi_end_mape)

        mvalid_bike_start_loss = np.mean(valid_bike_start_loss)
        mvalid_bike_end_loss = np.mean(valid_bike_end_loss)
        mvalid_taxi_start_loss = np.mean(valid_taxi_start_loss)
        mvalid_taxi_end_loss = np.mean(valid_taxi_end_loss)
        mvalid_bike_start_rmse = np.mean(valid_bike_start_rmse)
        mvalid_bike_end_rmse = np.mean(valid_bike_end_rmse)
        mvalid_taxi_start_rmse = np.mean(valid_taxi_start_rmse)
        mvalid_taxi_end_rmse = np.mean(valid_taxi_end_rmse)
        mvalid_bike_start_mape = np.mean(valid_bike_start_mape)
        mvalid_bike_end_mape = np.mean(valid_bike_end_mape)
        mvalid_taxi_start_mape = np.mean(valid_taxi_start_mape)
        mvalid_taxi_end_mape = np.mean(valid_taxi_end_mape)

        mvalid_loss = mvalid_bike_start_loss + mvalid_bike_end_loss + mvalid_taxi_start_loss + mvalid_taxi_end_loss
        # mvalid_loss = mvalid_taxi_start_loss + mvalid_taxi_end_loss
        # mvalid_loss = mvalid_bike_start_loss + mvalid_bike_end_loss
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}\n Train Bike Start Loss: {:.4f}, Train Bike End Loss: {:.4f}, ' \
              'Train Taxi Start Loss: {:.4f}, Train Taxi End Loss: {:.4f}, \n' \
              'Train Bike Start RMSE: {:.4f}, Train Bike End RMSE: {:.4f}, ' \
              'Train Taxi Start RMSE: {:.4f}, Train Taxi End RMSE: {:.4f}, \n' \
              'Train Bike Start MAPE: {:.4f}, Train Bike End MAPE: {:.4f}, ' \
              'Train Taxi Start MAPE: {:.4f}, Train Taxi End MAPE: {:.4f}, \n' \
              'Valid Bike Start Loss: {:.4f}, Valid Bike End Loss: {:.4f}, ' \
              'Valid Taxi Start Loss: {:.4f}, Valid Taxi End Loss: {:.4f}, \n' \
              'Valid Bike Start RMSE: {:.4f}, Valid Bike End RMSE: {:.4f}, ' \
              'Valid Taxi Start RMSE: {:.4f}, Valid Taxi End RMSE: {:.4f}, \n' \
              'Valid Bike Start MAPE: {:.4f}, Valid Bike End MAPE: {:.4f}, ' \
              'Valid Taxi Start MAPE: {:.4f}, Valid Taxi End MAPE: {:.4f}'
        print(log.format(epoch, mtrain_bike_start_loss, mtrain_bike_end_loss,
                         mtrain_taxi_start_loss, mtrain_taxi_end_loss,
                         mtrain_bike_start_rmse, mtrain_bike_end_rmse,
                         mtrain_taxi_start_rmse, mtrain_taxi_end_rmse,
                         mtrain_bike_start_mape, mtrain_bike_end_mape,
                         mtrain_taxi_start_mape, mtrain_taxi_end_mape,
                         mvalid_bike_start_loss, mvalid_bike_end_loss,
                         mvalid_taxi_start_loss, mvalid_taxi_end_loss,
                         mvalid_bike_start_rmse, mvalid_bike_end_rmse,
                         mvalid_taxi_start_rmse, mvalid_taxi_end_rmse,
                         mvalid_bike_start_mape, mvalid_bike_end_mape,
                         mvalid_taxi_start_mape, mvalid_taxi_end_mape), flush=True)
        

        early_stopping(mvalid_loss, engine.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # torch.save(engine.model.state_dict(),
        #         args.save + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
            
        if mvalid_loss <= min(his_loss):
            best_model_path = os.path.join(args.save, "best_model.pth")
            torch.save(engine.model.state_dict(), best_model_path)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
 
    bestid = np.argmin(his_loss)
    # engine.model.load_state_dict(
    #     torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid], 4))+".pth"))
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    # torch.save(engine.model.state_dict(), args.save + "best_model.pth")



if __name__ == '__main__':
    from datetime import datetime
    current_time = datetime.now()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    log_file_name = os.path.join(args.save, 'log')
    sys.stdout = open(log_file_name, 'w')

    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
