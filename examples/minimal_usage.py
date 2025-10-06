
from ms3m.models.ms3m import MS3M
import torch

model = MS3M(d_model=128, n_scales=3, n_experts=2)
x = torch.randn(2, 64, 128)
y = model(x)
print("y shape:", y.shape)
