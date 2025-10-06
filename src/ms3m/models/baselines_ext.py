
"""
External baselines adapters for MS3M training loop.

This file declares lightweight wrappers for the following methods:
- FEDformer
- Informer
- TFT (Temporal Fusion Transformer)
- ETSformer
- Crossformer
- PatchTST
- iTransformer

Each wrapper exposes a PyTorch nn.Module with signature:
    forward(x: Tensor) -> Tensor of shape (B, T, D)

By default, these classes raise a clear ImportError / NotImplementedError until
you plug in actual implementations. You can either:
  A) vendor upstream code under `src/vendor/<model_name>/...` and edit the import paths here, or
  B) install a pip package that provides the implementation and import it here.

All wrappers are intentionally minimal and only define the contract expected by
`scripts/train.py`.
"""
from typing import Optional
import torch
from torch import nn

def _missing_impl(name: str):
    raise NotImplementedError(
        f"{name} is declared but not implemented yet.\n"
        f"→ Option 1: Vendor the official implementation into src/vendor/{name.lower()}/ and update imports in baselines_ext.py.\n"
        f"→ Option 2: Install a pip package for {name} and update imports.\n"
        f"Make sure the model returns shape (B,T,D) given input (B,T,D)."
    )

class FEDformerBaseline(nn.Module):
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        _missing_impl("FEDformer")

    def forward(self, x: torch.Tensor):
        return x

class InformerBaseline(nn.Module):
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        _missing_impl("Informer")

    def forward(self, x: torch.Tensor):
        return x

class TFTBaseline(nn.Module):
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        _missing_impl("TFT")

    def forward(self, x: torch.Tensor):
        return x

class ETSformerBaseline(nn.Module):
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        _missing_impl("ETSformer")

    def forward(self, x: torch.Tensor):
        return x

class CrossformerBaseline(nn.Module):
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        _missing_impl("Crossformer")

    def forward(self, x: torch.Tensor):
        return x

class PatchTSTBaseline(nn.Module):
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        _missing_impl("PatchTST")

    def forward(self, x: torch.Tensor):
        return x

class ITransformerBaseline(nn.Module):
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        _missing_impl("iTransformer")

    def forward(self, x: torch.Tensor):
        return x
