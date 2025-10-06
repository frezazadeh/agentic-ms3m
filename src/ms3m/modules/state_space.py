
from typing import Optional
import torch
from torch import nn

class SimpleSSM(nn.Module):
    """
    Minimal structured state-space layer (toy).
    Acts like a gated 1D convolution with residual mixing to emulate long-range memory.
    Not intended to be a faithful S4—replace with your preferred SSM kernel.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=1)
        self.gate = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        y = self.norm(x)
        y = y.transpose(1, 2)  # (B, D, T)
        y = self.conv(y).transpose(1, 2)  # (B, T, D)
        g = torch.sigmoid(self.gate(x))
        out = x + self.drop(g * y)
        return out, state
