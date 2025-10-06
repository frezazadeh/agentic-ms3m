
from typing import List
import torch
from torch import nn
from einops import rearrange
from ..modules.state_space import SimpleSSM

class Expert(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.ssm = SimpleSSM(d_model=d_model, kernel_size=kernel_size)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        y, _ = self.ssm(x)
        return self.proj(y)

class Router(nn.Module):
    """
    Content-based routing across experts and scales.
    Produces a convex combination over experts per token.
    """
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.score = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor):
        # x: (B,T,D)
        logits = self.score(x)  # (B,T,E)
        weights = torch.softmax(logits, dim=-1)
        return weights

class MS3M(nn.Module):
    """
    Multi-Scale Structured State-Space Mixtures (toy reference).
    - SSM experts with different kernel sizes = different temporal scales.
    - Router computes mixture weights per time step.
    """
    def __init__(self, d_model: int = 256, n_scales: int = 3, n_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        self.n_experts = n_experts

        # Define experts with increasing receptive fields (scales)
        kernel_banks = [5, 9, 17][:n_scales]
        self.experts = nn.ModuleList([Expert(d_model, k) for k in kernel_banks for _ in range(n_experts)])
        self.router = Router(d_model, n_experts * n_scales)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x: (B,T,D)
        B, T, D = x.shape
        expert_outputs: List[torch.Tensor] = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # each (B,T,D)
        E = len(expert_outputs)
        stack = torch.stack(expert_outputs, dim=-2)  # (B,T,E,D)

        weights = self.router(x)  # (B,T,E)
        weights = weights.unsqueeze(-1)  # (B,T,E,1)
        mixed = (stack * weights).sum(dim=-2)  # (B,T,D)
        return self.out(self.norm(mixed))
