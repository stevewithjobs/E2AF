import torch
import torch.nn as nn
import functools

from torch import optim
from models.model import Net_timesnet_sample_onetimesnet
import yaml
from utils import mdtp
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
from models.loss import routing_classification_loss_by_error
from utils.mdtp import metric1, getCrossloss
from collections import deque
import time


config = yaml.safe_load(open('config.yml'))


def adjust_learning_rate(optimizer, lr, wd):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd


class Trainer:
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, device, lrate,
                 wdecay, clip, smoe_start_epoch, pred_size):
        # self.model = Net(batch_size, window_size, node_num, in_features, out_features, lstm_features)
        self.base_smoe_config = SpatialMoEConfig(
            in_planes=2,
            out_planes=3,
            num_experts=6,
            gate_block=functools.partial(SpatialLatentTensorGate2d,
                                node_num = node_num),
            save_error_signal=True,
            # dampen_expert_error=False,
            dampen_expert_error=True,
            unweighted=True,
            block_gate_grad=True,
            routing_error_quantile=0.7,
            pred_size=pred_size,
            windows_size=window_size
        )
        # self.model = NetMoe(batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config=self.base_smoe_config)
        # self.model = Net_timesnet_moe(batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config=self.base_smoe_config)
        self.model = Net_timesnet_sample_onetimesnet(batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config=self.base_smoe_config, pred_size=pred_size)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params}')
        
        # self.model = Net_timesnet_moe_nstp(batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config=self.base_smoe_config, pred_size=pred_size)


        self.model.to(device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay, betas=(0.9, 0.99))
        # 提取 self.model.nodesample 模块中的所有参数
        nodesample_params = [param for name, param in self.model.named_parameters() if 'nodesample' in name]

        # 提取模型中其他的参数（不包括 nodesample）
        other_params = [param for name, param in self.model.named_parameters() if 'nodesample' not in name]

        # 定义优化器，设置不同的学习率
        self.optimizer = optim.Adam([
            {'params': nodesample_params, 'lr': 0.01},  # 为整个 nodesample 模块设置较大的学习率
            {'params': other_params, 'lr': lrate, 'betas': (0.9, 0.99), 'weight_decay': wdecay}
        ], lr=lrate, weight_decay=wdecay, betas=(0.9, 0.99))
        self.loss = mdtp.mae_weight

        self.maeloss = mdtp.mae_weight
        self.rmseloss = mdtp.rmse_weight
        self.mapeloss = mdtp.mape_weight
        self.rmse_per_node = mdtp.rmse_per_node

        self.batch_size = batch_size
        self.window_size = window_size
        self.node_num = node_num
        self.clip = clip
        self.smoe_start_epoch = smoe_start_epoch
        self.smoe_start = False

        self.gcn1_grad = 0
        self.gcn2_grad = 0

    # def train(self, input, real, lr, wd, iter):
    #     self.model.train()
    #     self.optimizer.zero_grad()

    #     bike_start_predict, bike_end_predict, taxi_start_predict, taxi_end_predict = self.model(input, 24)

    #     bike_start_predict.retain_grad()
    #     bike_end_predict.retain_grad()
    #     taxi_start_predict.retain_grad()
    #     taxi_end_predict.retain_grad()


    #     bike_real, taxi_real = real[0], real[1]
        
    #     # bike_threshold = config['threshold'] / config['bike_volume_max']
    #     # taxi_threshold = config['threshold'] / config['taxi_volume_max']
        
    #     # bike_start_loss = self.loss(bike_start_predict, bike_real[0], bike_threshold)
    #     # bike_end_loss = self.loss(bike_end_predict, bike_real[1], bike_threshold)
    #     # bike_start_loss_rmse = self.rmseloss(bike_start_predict, bike_real[0], bike_threshold)
    #     # bike_end_loss_rmse = self.rmseloss(bike_end_predict, bike_real[1], bike_threshold)
    #     # bike_start_loss_mape = self.mapeloss(bike_start_predict, bike_real[0], bike_threshold)
    #     # bike_end_loss_mape = self.mapeloss(bike_end_predict, bike_real[1], bike_threshold)

    #     # taxi_start_loss = self.loss(taxi_start_predict, taxi_real[0], taxi_threshold)
    #     # taxi_end_loss = self.loss(taxi_end_predict, taxi_real[1], taxi_threshold)
    #     # taxi_start_loss_rmse = self.rmseloss(taxi_start_predict, taxi_real[0], taxi_threshold)
    #     # taxi_end_loss_rmse = self.rmseloss(taxi_end_predict, taxi_real[1], taxi_threshold)
    #     # taxi_start_loss_mape = self.mapeloss(taxi_start_predict, taxi_real[0], taxi_threshold)
    #     # taxi_end_loss_mape = self.mapeloss(taxi_end_predict, taxi_real[1], taxi_threshold)
    #     mask = torch.ones_like(bike_real[0], dtype=torch.bool)
    #     start_mask = mask
    #     end_mask = mask

    #     bike_start_loss = self.loss(bike_start_predict, bike_real[0], start_mask)
    #     bike_end_loss = self.loss(bike_end_predict, bike_real[1], end_mask)
    #     bike_start_loss_rmse = self.rmseloss(bike_start_predict, bike_real[0], start_mask)
    #     bike_end_loss_rmse = self.rmseloss(bike_end_predict, bike_real[1], end_mask)
    #     bike_start_loss_mape = self.mapeloss(bike_start_predict, bike_real[0], start_mask)
    #     bike_end_loss_mape = self.mapeloss(bike_end_predict, bike_real[1], end_mask)

    #     taxi_start_loss = self.loss(taxi_start_predict, taxi_real[0], start_mask)
    #     taxi_end_loss = self.loss(taxi_end_predict, taxi_real[1], end_mask)
    #     taxi_start_loss_rmse = self.rmseloss(taxi_start_predict, taxi_real[0], start_mask)
    #     taxi_end_loss_rmse = self.rmseloss(taxi_end_predict, taxi_real[1], end_mask)
    #     taxi_start_loss_mape = self.mapeloss(taxi_start_predict, taxi_real[0], start_mask)
    #     taxi_end_loss_mape = self.mapeloss(taxi_end_predict, taxi_real[1], end_mask)

    #     '''rn_loss'''
    #     # weight_loss = 1 - weight1 + 1 - weight2
    #     # mean_weight_loss = torch.mean(weight_loss)  
    #     '''total_loss'''
    #     # loss = bike_start_loss + bike_end_loss + taxi_start_loss + taxi_end_loss
    #     # loss = bike_start_loss + bike_end_loss
    #     # print(weight_loss)
    #     # loss = taxi_start_loss_rmse + taxi_end_loss_rmse
    #     # loss = bike_start_loss_rmse + bike_end_loss_rmse
    #     # if epoch < 30:
    #     #     loss = taxi_start_loss_rmse + taxi_end_loss_rmse + bike_start_loss_rmse + bike_end_loss_rmse
    #     # else: 
    #     # loss = taxi_start_loss_rmse + taxi_end_loss_rmse + bike_start_loss_rmse + bike_end_loss_rmse
        


    #     loss = taxi_start_loss_rmse + taxi_end_loss_rmse + (bike_start_loss_rmse + bike_end_loss_rmse) * 4   
    #     loss.backward(retain_graph=True) 
        
    #     '''maskloss'''
    #     # rmse_per_node = 0
    #     # rmse_per_node += self.rmse_per_node(bike_start_predict, bike_real[0], start_mask)
    #     # rmse_per_node += self.rmse_per_node(bike_end_predict, bike_real[1], end_mask)
    #     # rmse_per_node += self.rmse_per_node(taxi_start_predict, taxi_real[0], start_mask)
    #     # rmse_per_node += self.rmse_per_node(taxi_end_predict, taxi_real[1], end_mask)
    #     # topnum = int(231*2/3)




    #     # self.gcn1_grad += self.model.gcn_output1.grad
    #     # gcn1_grad = self.gcn1_grad.view(32,24,231,-1)
    #     # gcn1_grad = gcn1_grad.sum(dim=(0, 1, 3))
    #     # _, top1_indices = torch.topk(gcn1_grad, 150)
    #     # print(top1_indices)

    #     # self.gcn2_grad += self.model.gcn_output2.grad
    #     # gcn2_grad = self.gcn2_grad.view(32,24,231,-1)
    #     # gcn2_grad = gcn2_grad.sum(dim=(0, 1, 3))
    #     # _, top2_indices = torch.topk(gcn2_grad, 150)
    #     # print(top2_indices)
        
    #     # maskloss = mdtp.getMaskloss(self.model.nodesample.mask_aft, self.model.nodesample.top_indices, self.model.gcn_output1.grad + self.model.gcn_output2.grad)
    #     # if iter % 3 == 2 :

        
    #     grad1 =  torch.abs(bike_start_predict.grad) + torch.abs(bike_end_predict.grad)
    #     grad2 = torch.abs(taxi_start_predict.grad) + torch.abs(taxi_end_predict.grad)
    #     # grad1 = torch.abs(self.model.gcn_output1.grad)
    #     # grad2 = torch.abs(self.model.gcn_output2.grad)

    #     maskloss1 = mdtp.getMaskloss(self.model.nodesample.conf1, grad1)
    #     maskloss2 = mdtp.getMaskloss(self.model.nodesample.conf2, grad2)
    #     maskloss = maskloss1 + maskloss2
    #     maskloss.backward()
        
    #     # cross_loss = getCrossloss(self.model.crossA) + getCrossloss(self.model.crossB)
    #     # cross_loss.backward()
    #     '''rc_loss'''
    #     if self.smoe_start:
    #         scaler = torch.cuda.amp.GradScaler(enabled=True)
    #         rc_loss = routing_classification_loss_by_error(self.model, scaler, self.base_smoe_config.routing_error_quantile)
    #         if rc_loss: 
    #             rc_loss_avg = sum(rc_loss)
    #             rc_loss_avg = rc_loss_avg
    #             rc_loss_avg.backward()
        
    #     if self.clip is not None:
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
    #     self.optimizer.step()
    #     adjust_learning_rate(self.optimizer, lr, wd)
    #     mae = (bike_start_loss.item(), bike_end_loss.item(), taxi_start_loss.item(), taxi_end_loss.item())
    #     rmse = (bike_start_loss_rmse.item(), bike_end_loss_rmse.item(), taxi_start_loss_rmse.item(), taxi_end_loss_rmse.item())
    #     mape = (bike_start_loss_mape.item(), bike_end_loss_mape.item(), taxi_start_loss_mape.item(), taxi_end_loss_mape.item())


    #     # rmse = (mdtp.rmse(bike_start_predict, bike_real[0], bike_start_mask).item(),
    #     #         mdtp.rmse(bike_end_predict, bike_real[1], bike_end_mask).item(),
    #     #         mdtp.rmse(taxi_start_predict, taxi_real[0], taxi_start_mask).item(),
    #     #         mdtp.rmse(taxi_end_predict, taxi_real[1], taxi_end_mask).item())
    #     # mape = (mdtp.mape(bike_start_predict, bike_real[0], bike_start_mask).item(),
    #     #         mdtp.mape(bike_end_predict, bike_real[1], bike_end_mask).item(),
    #     #         mdtp.mape(taxi_start_predict, taxi_real[0], taxi_start_mask).item(),
    #     #         mdtp.mape(taxi_end_predict, taxi_real[1], taxi_end_mask).item())
    
    #     return mae, rmse, mape
    def train(self, input, real, lr, wd, iter):
        print("Training iteration:", iter)
        self.model.train()
        self.optimizer.zero_grad()

        # === Step 1: Forward ===
        # retain gradient at SMOE output level
        # self.model.gcn_output1.retain_grad()
        # self.model.gcn_output2.retain_grad()

        # model forward
        bike_start_predict, bike_end_predict, taxi_start_predict, taxi_end_predict = self.model(input)

        # === Step 2: Compute main task loss ===
        bike_real, taxi_real = real[0], real[1]
        mask = torch.ones_like(bike_real[0], dtype=torch.bool)

        # Compute RMSE losses
        bike_start_loss_rmse = self.rmseloss(bike_start_predict, bike_real[0], mask)
        bike_end_loss_rmse = self.rmseloss(bike_end_predict, bike_real[1], mask)
        taxi_start_loss_rmse = self.rmseloss(taxi_start_predict, taxi_real[0], mask)
        taxi_end_loss_rmse = self.rmseloss(taxi_end_predict, taxi_real[1], mask)

        # Weighted sum of losses
        main_loss = (
            taxi_start_loss_rmse + taxi_end_loss_rmse +
            (bike_start_loss_rmse + bike_end_loss_rmse) * 2
        )

        # Backprop main task loss (retain graph for additional gradients)
        main_loss.backward(retain_graph=True)

        # === Step 4: Optionally add other regularization (e.g., RC loss) ===
        # if self.smoe_start:
        rc_loss = routing_classification_loss_by_error(
            self.model, torch.cuda.amp.GradScaler(enabled=True),
            self.base_smoe_config.routing_error_quantile
        )
        if rc_loss:
            rc_loss_avg = sum(rc_loss)
            rc_loss_avg.backward()

        # === Step 5: Gradient step ===
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # Adjust LR
        adjust_learning_rate(self.optimizer, lr, wd)

        # === Step 6: Logging ===
        mae = (
            self.loss(bike_start_predict, bike_real[0], mask).item(),
            self.loss(bike_end_predict, bike_real[1], mask).item(),
            self.loss(taxi_start_predict, taxi_real[0], mask).item(),
            self.loss(taxi_end_predict, taxi_real[1], mask).item()
        )
        rmse = (
            bike_start_loss_rmse.item(), bike_end_loss_rmse.item(),
            taxi_start_loss_rmse.item(), taxi_end_loss_rmse.item()
        )
        mape = (
            self.mapeloss(bike_start_predict, bike_real[0], mask).item(),
            self.mapeloss(bike_end_predict, bike_real[1], mask).item(),
            self.mapeloss(taxi_start_predict, taxi_real[0], mask).item(),
            self.mapeloss(taxi_end_predict, taxi_real[1], mask).item()
        )

        return mae, rmse, mape

    def val(self, input, real, epoch):
        self.model.eval()
        t1 = time.time()
        bike_start_predict, bike_end_predict, taxi_start_predict, taxi_end_predict = self.model(input)
        t2 = time.time()
        bike_real, taxi_real = real[0], real[1]

        mask = torch.ones_like(bike_real[0], dtype=torch.bool)
        start_mask = mask
        end_mask = mask

        bike_start_loss = self.loss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss = self.loss(bike_end_predict, bike_real[1], end_mask)
        bike_start_loss_rmse = self.rmseloss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss_rmse = self.rmseloss(bike_end_predict, bike_real[1], end_mask)
        bike_start_loss_mape = self.mapeloss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss_mape = self.mapeloss(bike_end_predict, bike_real[1], end_mask)

        taxi_start_loss = self.loss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss = self.loss(taxi_end_predict, taxi_real[1], end_mask)
        taxi_start_loss_rmse = self.rmseloss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss_rmse = self.rmseloss(taxi_end_predict, taxi_real[1], end_mask)
        taxi_start_loss_mape = self.mapeloss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss_mape = self.mapeloss(taxi_end_predict, taxi_real[1], end_mask)

        # bike_threshold = config['threshold'] / config['bike_volume_max']
        # taxi_threshold = config['threshold'] / config['taxi_volume_max']
        
        # bike_start_loss = self.loss(bike_start_predict, bike_real[0], bike_threshold)
        # bike_end_loss = self.loss(bike_end_predict, bike_real[1], bike_threshold)
        # bike_start_loss_rmse = self.rmseloss(bike_start_predict, bike_real[0], bike_threshold)
        # bike_end_loss_rmse = self.rmseloss(bike_end_predict, bike_real[1], bike_threshold)
        # bike_start_loss_mape = self.mapeloss(bike_start_predict, bike_real[0], bike_threshold)
        # bike_end_loss_mape = self.mapeloss(bike_end_predict, bike_real[1], bike_threshold)

        # taxi_start_loss = self.loss(taxi_start_predict, taxi_real[0], taxi_threshold)
        # taxi_end_loss = self.loss(taxi_end_predict, taxi_real[1], taxi_threshold)
        # taxi_start_loss_rmse = self.rmseloss(taxi_start_predict, taxi_real[0], taxi_threshold)
        # taxi_end_loss_rmse = self.rmseloss(taxi_end_predict, taxi_real[1], taxi_threshold)
        # taxi_start_loss_mape = self.mapeloss(taxi_start_predict, taxi_real[0], taxi_threshold)
        # taxi_end_loss_mape = self.mapeloss(taxi_end_predict, taxi_real[1], taxi_threshold)

        mae = (bike_start_loss.item(), bike_end_loss.item(), taxi_start_loss.item(), taxi_end_loss.item())
        rmse = (bike_start_loss_rmse.item(), bike_end_loss_rmse.item(), taxi_start_loss_rmse.item(), taxi_end_loss_rmse.item())
        mape = (bike_start_loss_mape.item(), bike_end_loss_mape.item(), taxi_start_loss_mape.item(), taxi_end_loss_mape.item())
        # rmse = (mdtp.rmse(bike_start_predict, bike_real[0], bike_start_mask).item(),
        #         mdtp.rmse(bike_end_predict, bike_real[1], bike_end_mask).item(),
        #         mdtp.rmse(taxi_start_predict, taxi_real[0], taxi_start_mask).item(),
        #         mdtp.rmse(taxi_end_predict, taxi_real[1], taxi_end_mask).item())
        # mape = (mdtp.mape(bike_start_predict, bike_real[0], bike_start_mask).item(),
        #         mdtp.mape(bike_end_predict, bike_real[1], bike_end_mask).item(),
        #         mdtp.mape(taxi_start_predict, taxi_real[0], taxi_start_mask).item(),
        #         mdtp.mape(taxi_end_predict, taxi_real[1], taxi_end_mask).item())
        
        return (mae, rmse, mape), (t2-t1)

