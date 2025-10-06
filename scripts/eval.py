
import torch
from ms3m.models.ms3m import MS3M

if __name__ == "__main__":
    model = MS3M(d_model=128)
    x = torch.randn(2, 64, 128)
    y = model(x)
    print("Output shape:", y.shape)
