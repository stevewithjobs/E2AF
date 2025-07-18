import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from models.smoe import GatedSpatialMoE2d
import math
import time


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1)
        return res

class Inception_Block_V3(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            # kernel_size=(2*i+1, 2*i+1, 2*i+1), padding=i
            # padding = (5*i, i, i)
            kernels.append(nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(2 * i + 1, 2 * i + 1, 2 * i + 1),
                padding=i
            ))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.reshape(x.size(0), -1, 231, 7, 4)
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        # self.seq_len = 17
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
    def forward(self, x):
        B, T, N = x.size()
        # period_list, period_weight = FFT_for_Period(x, self.k)
        period_list = [4]
        # period_list = [3, 6, 12]
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (x.size(1)) % period != 0:
                length = (((x.size(1)) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (x.size(1))), x.shape[2]]).to(x.device)
                out_x = torch.cat([x, padding], dim=1)
                # out_y = torch.cat([y, padding], dim=1)
            else:
                length = (x.size(1))
                out_x = x
                # out_y = y
            # reshape
            out_x = out_x.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # out_y = out_y.reshape(B, length // period, period,
            #                   N).permute(0, 3, 1, 2).contiguous()
            # inter_concact = []
            # dim = 2
            # for i in range(out_x.size(2)):
            #     inter_concact.append(out_x.select(dim, i).unsqueeze(2))
            #     inter_concact.append(out_y.select(dim, i).unsqueeze(2))
            # out = torch.cat(inter_concact, dim)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out_x)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(x.size(1)), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # res = torch.sum(res * period_weight, -1)

        res = torch.sum(res , -1)
            
        # res = torch.sum(res, -1)
        
        # residual connection
        # res = res + x
        return res
    # def forward(self, x):
    #     B, T, N = x.size()
    #     # period_list, period_weight = FFT_for_Period(x, self.k)
    #     period_list = [4]
    #     # period_list = [3, 6, 12]
    #     res = []
    #     for i in range(self.k):
    #         period = period_list[i]
    #         # padding
    #         if (self.seq_len + self.pred_len) % period != 0:
    #             length = (((self.seq_len + self.pred_len) // period) + 1) * period
    #             padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
    #             out_x = torch.cat([x, padding], dim=1)
    #             # out_y = torch.cat([y, padding], dim=1)
    #         else:
    #             length = (self.seq_len + self.pred_len)
    #             out_x = x
    #             # out_y = y
    #         # reshape
    #         out_x = out_x.reshape(B, length // period, period,
    #                           N).permute(0, 3, 1, 2).contiguous()
    #         # out_y = out_y.reshape(B, length // period, period,
    #         #                   N).permute(0, 3, 1, 2).contiguous()
    #         # inter_concact = []
    #         # dim = 2
    #         # for i in range(out_x.size(2)):
    #         #     inter_concact.append(out_x.select(dim, i).unsqueeze(2))
    #         #     inter_concact.append(out_y.select(dim, i).unsqueeze(2))
    #         # out = torch.cat(inter_concact, dim)
    #         # 2D conv: from 1d Variation to 2d Variation
    #         out = self.conv(out_x)
    #         # reshape back
    #         out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
    #         res.append(out[:, :(self.seq_len + self.pred_len), :])
    #     res = torch.stack(res, dim=-1)
    #     # adaptive aggregation
    #     # period_weight = F.softmax(period_weight, dim=1)
    #     # period_weight = period_weight.unsqueeze(
    #     #     1).unsqueeze(1).repeat(1, T, N, 1)
    #     # res = torch.sum(res * period_weight, -1)

    #     res = torch.sum(res , -1)
            
    #     # res = torch.sum(res, -1)
        
    #     # residual connection
    #     # res = res + x
    #     return res

class TimesBlock_3D(nn.Module):
    def __init__(self, configs):
        super(TimesBlock_3D, self).__init__()
        self.seq_len = configs.seq_len
        # self.seq_len = 17
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V3(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V3(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
    def forward(self, x):
        B, T, N = x.size()
        # period_list, period_weight = FFT_for_Period(x, self.k)
        period_list = [4]
        # period_list = [3, 6, 12]
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (x.size(1)) % period != 0:
                length = (((x.size(1)) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (x.size(1))), x.shape[2]]).to(x.device)
                out_x = torch.cat([x, padding], dim=1)
                # out_y = torch.cat([y, padding], dim=1)
            else:
                length = (x.size(1))
                out_x = x
                # out_y = y
            # reshape
            out_x = out_x.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # out_y = out_y.reshape(B, length // period, period,
            #                   N).permute(0, 3, 1, 2).contiguous()
            # inter_concact = []
            # dim = 2
            # for i in range(out_x.size(2)):
            #     inter_concact.append(out_x.select(dim, i).unsqueeze(2))
            #     inter_concact.append(out_y.select(dim, i).unsqueeze(2))
            # out = torch.cat(inter_concact, dim)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out_x)
            # reshape back
            out = out.reshape(B, N, -1).permute(0, 2, 1)
            # out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(x.size(1)), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # res = torch.sum(res * period_weight, -1)

        res = torch.sum(res , -1)
            
        # res = torch.sum(res, -1)
        
        # residual connection
        # res = res + x
        return res

class TimesBlock_moe(nn.Module):
    def __init__(self, configs, smoe_config):
        super(TimesBlock_moe, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.num_kernels = configs.num_kernels
        # parameter-efficient design
        self.conv_x = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V2(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        self.conv_y = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V2(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        self.moe = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )

    def forward(self, x, y):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out_x = torch.cat([x, padding], dim=1)
                out_y = torch.cat([y, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out_x = x
                out_y = y
            # reshape
            out_x = out_x.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            out_y = out_y.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out_x = self.conv_x(out_x)
            out_y = self.conv_y(out_y)

            out = torch.cat((out_x, out_y), dim=-1)

            out = out.reshape(B , N, -1, self.num_kernels * 2)[:, :, :(self.seq_len + self.pred_len), :]

            out = out.permute(0, 2, 3, 1).reshape(32, self.seq_len + self.pred_len, self.num_kernels * 2, 231, -1)

            out = self.moe(x, out)

            # reshape back
            out = out.reshape(B, -1, N)
            res.append(out)
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        # res = res + x
        return res

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        enc_out = x_enc/stdev
        # embedding
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out[:, -self.pred_len:, :]

class Model_moe(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs, smoe_config):
        super(Model_moe, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.batch_size = configs.batch_size

        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.model_x = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.model_y = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        self.smoe_y = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
        
        self.smoe_x = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
        
        self.layer_norm_x = nn.LayerNorm(configs.d_model)
        self.layer_norm_y = nn.LayerNorm(configs.d_model)

        self.predict_linear_x = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear_y = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        
        features = int(configs.d_model)
        out = int(configs.c_out)
        self.projection_x = nn.Linear(
                features, out, bias=True)
        self.projection_y = nn.Linear(
                features, out, bias=True)
        

    def forward(self, x, y, x_enc, y_enc):
        # Normalization from Non-stationary Transformer
        x_means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_out = x_enc/x_stdev

        y_means = y_enc.mean(1, keepdim=True).detach()
        y_enc = y_enc - y_means
        y_stdev = torch.sqrt(
            torch.var(y_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        y_out = y_enc/y_stdev

        # embedding
        x_out = self.dropout1(x_out + self.position_embedding(x_out))
        y_out = self.dropout2(y_out + self.position_embedding(y_out))

        x_out = self.predict_linear_x(x_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        y_out = self.predict_linear_y(y_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):

            # 先算个gate，可以跳过没选中的gate
            experts_x = self.model_x[i](x_out) # (32, 25, f, 3)
            experts_y = self.model_y[i](y_out)

            # x_out = experts_x
            # y_out = experts_x
            experts_x = experts_x.permute(0, 1, 3, 2).reshape(self.batch_size, self.seq_len + self.pred_len, experts_x.size(-1), 231, -1)
            experts_y = experts_y.permute(0, 1, 3, 2).reshape(self.batch_size, self.seq_len + self.pred_len, experts_y.size(-1), 231, -1)
            experts = torch.cat((experts_x, experts_y), dim=2)

            x_smoeout = self.smoe_x(x, experts)
            x_out = x_smoeout.reshape(self.batch_size, self.seq_len + self.pred_len, -1)
            y_smoeout = self.smoe_y(y, experts)
            y_out = y_smoeout.reshape(self.batch_size, self.seq_len + self.pred_len, -1)

            x_out = self.layer_norm_x(x_out)
            y_out = self.layer_norm_y(y_out)

        # De-Normalization from Non-stationary Transformer
        x_out = x_out * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        x_out = x_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        y_out = y_out * \
                  (y_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        y_out = y_out + \
                  (y_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        # porject back
        # x_out = x_out.reshape(32, 25, 231, -1)
        # y_out = y_out.reshape(32, 25, 231, -1)
        x_out = self.projection_x(x_out)
        y_out = self.projection_y(y_out)

        return x_out[:, -self.pred_len:, :], y_out[:, -self.pred_len:, :]

class Model_moeconv(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs, smoe_config):
        super(Model_moeconv, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers

        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.model_x = nn.ModuleList([TimesBlock_moe(configs, smoe_config)
                                    for _ in range(configs.e_layers)])
        self.model_y = nn.ModuleList([TimesBlock_moe(configs, smoe_config)
                                    for _ in range(configs.e_layers)])
        
        self.layer_norm_x = nn.LayerNorm(configs.d_model)
        self.layer_norm_y = nn.LayerNorm(configs.d_model)

        self.predict_linear_x = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear_y = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        
        features = int(configs.d_model)
        out = int(configs.c_out)
        self.projection_x = nn.Linear(
                features, out, bias=True)
        self.projection_y = nn.Linear(
                features, out, bias=True)

    def forward(self, x, y, x_enc, y_enc):
        # Normalization from Non-stationary Transformer
        x_means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_out = x_enc/x_stdev

        y_means = y_enc.mean(1, keepdim=True).detach()
        y_enc = y_enc - y_means
        y_stdev = torch.sqrt(
            torch.var(y_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        y_out = y_enc/y_stdev


        # embedding
        x_out = self.dropout1(x_out + self.position_embedding(x_out))
        y_out = self.dropout2(y_out + self.position_embedding(y_out))

        x_out = self.predict_linear_x(x_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        y_out = self.predict_linear_y(y_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            experts_x = self.model_x[i](x_out, y_out) # (32, 25, f, 3)
            experts_y = self.model_y[i](y_out, x_out)

            x_out = self.layer_norm_x(experts_x)
            y_out = self.layer_norm_y(experts_y)

        # De-Normalization from Non-stationary Transformer
        x_out = x_out * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        x_out = x_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        y_out = y_out * \
                  (y_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        y_out = y_out + \
                  (y_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        # porject back
        # x_out = x_out.reshape(32, 25, 231, -1)
        # y_out = y_out.reshape(32, 25, 231, -1)
        x_out = self.projection_x(x_out)
        y_out = self.projection_y(y_out)

        return x_out[:, -self.pred_len:, :], y_out[:, -self.pred_len:, :]
    
class Model_withoutmoe(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model_withoutmoe, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers

        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.model_x = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.model_y = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        # self.smoe_x = nn.ModuleList([GatedSpatialMoE2d(
        #     smoe_config=smoe_config
        # )for _ in range(configs.e_layers)])
        # self.smoe_y = nn.ModuleList([GatedSpatialMoE2d(
        #     smoe_config=smoe_config
        # )for _ in range(configs.e_layers)])
        
        self.layer_norm_x = nn.LayerNorm(configs.d_model)
        self.layer_norm_y = nn.LayerNorm(configs.d_model)

        self.predict_linear_x = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear_y = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        
        features = int(configs.d_model)
        out = int(configs.c_out)
        self.projection_x = nn.Linear(
                features, out, bias=True)
        self.projection_y = nn.Linear(
                features, out, bias=True)

    def forward(self, x, y, x_enc, y_enc):
        # Normalization from Non-stationary Transformer
        x_means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_out = x_enc/x_stdev

        y_means = y_enc.mean(1, keepdim=True).detach()
        y_enc = y_enc - y_means
        y_stdev = torch.sqrt(
            torch.var(y_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        y_out = y_enc/y_stdev


        # embedding
        x_out = self.dropout1(x_out + self.position_embedding(x_out))
        y_out = self.dropout2(y_out + self.position_embedding(y_out))

        x_out = self.predict_linear_x(x_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        y_out = self.predict_linear_y(y_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet


        for i in range(self.layer):
            # 定义 CUDA 流
            start_time = time.time()

            # 在不同流中并行计算 experts_x 和 experts_y
            experts_x = self.model_x[i](x_out)
            experts_y = self.model_y[i](y_out)

            # 同步确保两个流的计算都完成
            # torch.cuda.synchronize()

            # 继续进行 layer_norm 和加法操作
            x_out = self.layer_norm_x(experts_x)
            y_out = self.layer_norm_y(experts_y)

            # 打印执行时间
            total_time = time.time() - start_time
            print(f"Layer {i} execution time: {total_time:.6f} sec")

            
        # De-Normalization from Non-stationary Transformer
        x_out = x_out * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        x_out = x_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        y_out = y_out * \
                  (y_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        y_out = y_out + \
                  (y_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        # porject back
        # x_out = x_out.reshape(32, 25, 231, -1)
        # y_out = y_out.reshape(32, 25, 231, -1)
        x_out = self.projection_x(x_out)
        y_out = self.projection_y(y_out)

        return x_out[:, -self.pred_len:, :], y_out[:, -self.pred_len:, :]

class TimesBlock_moenew(nn.Module):
    def __init__(self, configs):
        super(TimesBlock_moenew, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # period_list = [3, 6, 12]
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out_x = torch.cat([x, padding], dim=1)
                # out_y = torch.cat([y, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out_x = x
                # out_y = y
            # reshape
            out_x = out_x.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # out_y = out_y.reshape(B, length // period, period,
            #                   N).permute(0, 3, 1, 2).contiguous()
            # inter_concact = []
            # dim = 2
            # for i in range(out_x.size(2)):
            #     inter_concact.append(out_x.select(dim, i).unsqueeze(2))
            #     inter_concact.append(out_y.select(dim, i).unsqueeze(2))
            # out = torch.cat(inter_concact, dim)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out_x)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # res = torch.sum(res * period_weight, -1)
        # # residual connection
        # res = res + x
        return res

class Model_moenew(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs, smoe_config):
        super(Model_moenew, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.batch_size = configs.batch_size


        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.model_x = nn.ModuleList([TimesBlock_moenew(configs)
                                    for _ in range(configs.e_layers)])
        self.model_y = nn.ModuleList([TimesBlock_moenew(configs)
                                    for _ in range(configs.e_layers)])

        self.smoe_y = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
        self.smoe_x = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
        
        self.layer_norm_x = nn.LayerNorm(configs.d_model)
        self.layer_norm_y = nn.LayerNorm(configs.d_model)

        self.predict_linear_x = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear_y = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        
        features = int(configs.d_model)
        out = int(configs.c_out)
        self.projection_x = nn.Linear(
                features, out, bias=True)
        self.projection_y = nn.Linear(
                features, out, bias=True)

    def forward(self, x, y, x_enc, y_enc):
        # Normalization from Non-stationary Transformer
        x_means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_out = x_enc/x_stdev

        y_means = y_enc.mean(1, keepdim=True).detach()
        y_enc = y_enc - y_means
        y_stdev = torch.sqrt(
            torch.var(y_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        y_out = y_enc/y_stdev


        # embedding
        # x_out = x_enc
        # y_out = y_enc
        x_out = self.dropout1(x_out + self.position_embedding(x_out))
        y_out = self.dropout2(y_out + self.position_embedding(y_out))

        x_out = self.predict_linear_x(x_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        y_out = self.predict_linear_y(y_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            experts_x = self.model_x[i](x_out)
            experts_y = self.model_y[i](y_out)

            experts_x = experts_x.permute(0, 1, 3, 2).reshape(self.batch_size, self.seq_len + self.pred_len, experts_x.size(-1), 231, -1)
            experts_y = experts_y.permute(0, 1, 3, 2).reshape(self.batch_size, self.seq_len + self.pred_len, experts_y.size(-1), 231, -1)
            experts = torch.cat((experts_x, experts_y), dim=2)

            x_smoeout = self.smoe_x(x, experts)
            x_out = x_smoeout.reshape(self.batch_size, self.seq_len + self.pred_len, -1)
            y_smoeout = self.smoe_y(y, experts)
            y_out = y_smoeout.reshape(self.batch_size, self.seq_len + self.pred_len, -1)

            x_out = self.layer_norm_x(x_out)
            y_out = self.layer_norm_y(y_out)

        # De-Normalization from Non-stationary Transformer
        x_out = x_out * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        x_out = x_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        y_out = y_out * \
                  (y_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        y_out = y_out + \
                  (y_means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        # porject back
        # x_out = x_out.reshape(32, 25, 231, -1)
        # y_out = y_out.reshape(32, 25, 231, -1)
        x_out = self.projection_x(x_out)
        y_out = self.projection_y(y_out)

        return x_out[:, -self.pred_len:, :], y_out[:, -self.pred_len:, :]  

class Model_onetimenet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model_onetimenet, self).__init__()

        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers

        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.model_x = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.model_y = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        # self.smoe_x = nn.ModuleList([GatedSpatialMoE2d(
        #     smoe_config=smoe_config
        # )for _ in range(configs.e_layers)])
        # self.smoe_y = nn.ModuleList([GatedSpatialMoE2d(
        #     smoe_config=smoe_config
        # )for _ in range(configs.e_layers)])
        
        self.layer_norm_x = nn.LayerNorm(configs.d_model)
        self.layer_norm_y = nn.LayerNorm(configs.d_model)

        self.predict_linear_x = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear_y = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        
        features = int(configs.d_model)
        out = int(configs.c_out)
        self.projection = nn.Linear(
                features, out, bias=True)

        self.tstime = 0
        self.t = 0
    def forward(self, x, x_enc):
        
        # Normalization from Non-stationary Transformer
        x_means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_out = x_enc/x_stdev

        # embedding
        x_out = self.dropout1(x_out + self.position_embedding(x_out))

        # x_out = self.predict_linear_x(x_out.permute(0, 2, 1)).permute(
        #     0, 2, 1)  # align temporal dimensionzzz
        
        t1 = time.time()
        ##################### 线性层切片
        window_size = x_enc.size(1)
        weight_submatrix = self.predict_linear_x.weight[-(window_size + self.pred_len):, -window_size:]
        bias_subvector = self.predict_linear_x.bias[-(window_size + self.pred_len):]
        x_out = torch.matmul(x_out.permute(0, 2, 1), weight_submatrix.T) + bias_subvector
        x_out = x_out.permute(0, 2, 1)
        #####################
        t2 = time.time()
        if self.t > 2:
            self.tstime += t2 - t1
        # print("tstime for", self.tstime)

        # TimesNet
        for i in range(self.layer):
            experts_x = self.model_x[i](x_out) 

            x_out = self.layer_norm_x(experts_x)
        

        # De-Normalization from Non-stationary Transformer
        # x_out = x_out * \
        #           (x_stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # x_out = x_out + \
        #           (x_means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        x_out = x_out * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, window_size + self.pred_len, 1))
        x_out = x_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(
                      1, window_size + self.pred_len, 1))
        
        timenetout = self.projection(x_out)
        
        self.t += 1
        return timenetout[:, -self.pred_len:, :], (t2-t1)

class Model_3D(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model_3D, self).__init__()

        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers

        self.position_embedding = PositionalEmbedding(d_model=231 * configs.d_model)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.model_x = nn.ModuleList([TimesBlock_3D(configs)
                                    for _ in range(configs.e_layers)])
        self.model_y = nn.ModuleList([TimesBlock_3D(configs)
                                    for _ in range(configs.e_layers)])

        # self.smoe_x = nn.ModuleList([GatedSpatialMoE2d(
        #     smoe_config=smoe_config
        # )for _ in range(configs.e_layers)])
        # self.smoe_y = nn.ModuleList([GatedSpatialMoE2d(
        #     smoe_config=smoe_config
        # )for _ in range(configs.e_layers)])
        
        self.layer_norm_x = nn.LayerNorm(231 * configs.d_model)
        self.layer_norm_y = nn.LayerNorm(231 * configs.d_model)

        self.predict_linear_x = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear_y = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        
        features = int(231 * configs.d_model)
        out = int(231 * configs.c_out)
        self.projection = nn.Linear(
                features, out, bias=True)

        self.tstime = 0
        self.t = 0
    def forward(self, x, x_enc):
        t1 = time.time()
        # Normalization from Non-stationary Transformer
        x_means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_out = x_enc/x_stdev

        # embedding
        x_out = self.dropout1(x_out + self.position_embedding(x_out))

        # x_out = self.predict_linear_x(x_out.permute(0, 2, 1)).permute(
        #     0, 2, 1)  # align temporal dimensionzzz
        
        
        ##################### 线性层切片
        window_size = x_enc.size(1)
        weight_submatrix = self.predict_linear_x.weight[-(window_size + self.pred_len):, -window_size:]
        bias_subvector = self.predict_linear_x.bias[-(window_size + self.pred_len):]
        x_out = torch.matmul(x_out.permute(0, 2, 1), weight_submatrix.T) + bias_subvector
        x_out = x_out.permute(0, 2, 1)
        #####################

        # TimesNet
        for i in range(self.layer):
            experts_x = self.model_x[i](x_out) 
            # x_out = x_out.reshape(x_out.size(0), -1, (window_size + self.pred_len)).permute(0,2,1)

            x_out = self.layer_norm_x(experts_x)
        

        # De-Normalization from Non-stationary Transformer
        # x_out = x_out * \
        #           (x_stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # x_out = x_out + \
        #           (x_means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        x_out = x_out * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(
                      1, window_size + self.pred_len, 1))
        x_out = x_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(
                      1, window_size + self.pred_len, 1))
        
        timenetout = self.projection(x_out)
        t2 = time.time()
        if self.t > 2:
            self.tstime += t2 - t1
        # print("tstime for", self.tstime)
        self.t += 1
        return timenetout[:, -self.pred_len:, :], (t2-t1)

    
