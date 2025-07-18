import torch
import math
from torch import nn
from torch.nn import functional as F


class LstmAttention(nn.Module):
    def __init__(self, node_size, input_size, hidden_dim, n_layers):
        super(LstmAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim, node_size)
        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, x):
        _, (output, _) = self.lstm(x)
        output = output.permute(1, 0, 2)
        query = self.dropout(output)
        attn_output, alpha_n = self.attention_net(output, query)
        logit = self.fc(attn_output)
        return logit

