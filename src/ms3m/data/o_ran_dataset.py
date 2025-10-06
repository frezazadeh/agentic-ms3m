
import torch
from torch.utils.data import Dataset

class ORanKPIDataset(Dataset):
    """
    Minimal dataset wrapper for O-RAN KPIs.
    Each sample is a sequence of KPI vectors.
    Expects a preprocessed torch tensor of shape (N, T, D).
    """
    def __init__(self, tensor: torch.Tensor):
        assert tensor.ndim == 3, "tensor must be (N,T,D)"
        self.tensor = tensor

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        x = self.tensor[idx]
        y = x  # toy self-supervised/identity target
        return x, y
