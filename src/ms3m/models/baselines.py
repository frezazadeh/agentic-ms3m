
from typing import Optional
import torch
from torch import nn

class LSTMBaseline(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        y, _ = self.lstm(x)  # (B,T,D)
        return self.proj(y)

class TransformerBaseline(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 4*256, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        y = self.encoder(x)  # (B,T,D)
        return self.proj(y)

class TCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.down = nn.Conv1d(d_model, d_model, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.conv2.weight, a=5**0.5)

    def forward(self, x):  # x: (B,T,D)
        y = x.transpose(1, 2)  # (B,D,T)
        out = self.conv1(y)
        out = out[:, :, :-self.conv1.padding[0]]  # causal trim
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        res = self.down(y[..., :out.size(-1)])
        out = self.relu(out + res)
        return out.transpose(1, 2)  # back to (B,T,D)

class TCNBaseline(nn.Module):
    def __init__(self, d_model: int = 256, num_layers: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TCNBlock(d_model, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        y = self.net(x)
        return self.proj(y)
