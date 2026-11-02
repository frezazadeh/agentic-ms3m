"""PatchTST -- model definition copied verbatim from cell 4 of the archived
experiment notebook (MS3M_baselines.ipynb, sha256 24a14001c0fe7b19).
See PATCHES at the bottom of this module for the two corrections applied."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN1D(nn.Module):
    def __init__(self, num_channels: int = 1, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_channels))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_channels))

    def forward(self, x):
        # x: (B, L, C)
        mean = x.mean(dim=1, keepdim=True)           # (B,1,C)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_hat = x_hat * self.gamma + self.beta
        return x_hat, mean, var

    def denorm(self, y, mean, var):
        # y: (B, *, C) in normalized space -> de-normalize with same per-sample stats
        y_den = (y - (self.beta if self.affine else 0.0)) / (self.gamma + 1e-8) if self.affine else y
        y_den = y_den * torch.sqrt(var + self.eps) + mean
        return y_den


def patchify_1d_tailpad(x: torch.Tensor, patch_len: int, stride: int):
    """
    x: (B, L, 1)
    return: patches (B, T_p, patch_len), T_p, pad_right
    Tail is padded by repeating the last observed value to complete the last patch.
    """
    B, L, C = x.shape
    assert C == 1
    # compute number of tokens if we right-pad
    T_p = math.ceil((L - patch_len) / stride) + 1 if L > patch_len else 1
    needed = (T_p - 1) * stride + patch_len
    pad = max(0, needed - L)
    if pad > 0:
        last = x[:, -1:, :]                # (B,1,1)
        pad_vals = last.expand(B, pad, C)  # repeat last value
        x = torch.cat([x, pad_vals], dim=1)
    # unfold into patches
    xU = x.transpose(1,2).unfold(dimension=-1, size=patch_len, step=stride)  # (B,1,patch_len,T_p)
    xU = xU.permute(0, 3, 2, 1).squeeze(-1)  # (B, T_p, patch_len)
    return xU, T_p, pad


class BNTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d_model)   # applied over tokens (after transpose)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def _apply_bn(self, x, bn):
        # x: (B,T,D) -> (B,D,T) -> BN -> (B,T,D)
        x_t = x.transpose(1, 2)
        x_bn = bn(x_t)
        return x_bn.transpose(1, 2)

    def forward(self, x):
        # Pre-BN -> MHA -> residual
        z = self._apply_bn(x, self.bn1)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop1(attn_out)
        # Pre-BN -> FF -> residual
        z = self._apply_bn(x, self.bn2)
        x = x + self.drop2(self.ff(z))
        return x


class PatchTSTPaper(nn.Module):
    """
    Paper-faithful PatchTST (CI, univariate) with stable shapes:
      - RevIN
      - Tail-pad by repeating last value
      - LazyLinear patch projection
      - Learnable positional embeddings & flatten head sized ONCE from first batch
      - BN-based Transformer blocks
      - Later batches are padded/truncated to T_fixed
    """
    def __init__(self, L: int, patch_len: int, stride: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int = 256, dropout: float = 0.2):
        super().__init__()
        self.revin = RevIN1D(num_channels=1, affine=True)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.dropout = dropout

        # Project patches -> tokens; infers input dim on first call
        self.proj = nn.LazyLinear(d_model)

        # Will be set on first forward
        self.T_fixed = None           # int
        self.pos_embed = None         # nn.Parameter of shape (T_fixed, d_model)
        self.head = None              # nn.Sequential with in_features d_model*T_fixed

        self.blocks = nn.ModuleList([
            BNTransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def _build_once(self, T_p: int, device, dtype):
        """Build PE and head exactly once using the first seen token length."""
        if self.T_fixed is None:
            self.T_fixed = int(T_p)
            # FIX: force float32 to avoid AMP creating half params
            pe = torch.zeros(self.T_fixed, self.d_model, device=device, dtype=torch.float32)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = nn.Parameter(pe)
            self.head = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * self.T_fixed, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, 1),
            ).to(device=device, dtype=torch.float32)

    def _pad_trunc_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, T_p, D) -> (B, T_fixed, D) by repeating/truncating last token.
        Assumes self.T_fixed is set.
        """
        B, T_p, D = z.shape
        T_f = self.T_fixed
        if T_p == T_f:
            return z
        if T_p < T_f:
            pad_tok = z[:, -1:, :].expand(B, T_f - T_p, D)  # repeat last token
            return torch.cat([z, pad_tok], dim=1)
        else:
            return z[:, :T_f, :]

    def forward(self, x):
        """
        x: (B, L, 1)
        """
        B, L, C = x.shape
        x_norm, mean, var = self.revin(x)

        # Patchify with right-pad (repeat last value)
        patches, T_p, _ = patchify_1d_tailpad(x_norm, self.patch_len, self.stride)  # (B, T_p, P)

        # Project patches -> tokens
        tok = self.proj(patches)  # (B, T_p, D)

        # FIX: keep tokens in float32 regardless of autocast to avoid mat1/mat2 dtype mismatches
        tok = tok.float()

        # Build PE & head once (based on first batch T_p). PE/head are float32.
        self._build_once(T_p=int(tok.size(1)), device=tok.device, dtype=tok.dtype)

        # Align tokens to T_fixed and add learnable positional embeddings (both float32)
        tok = self._pad_trunc_tokens(tok)                 # (B, T_fixed, D) float32
        tok = tok + self.pos_embed.unsqueeze(0)           # (1, T_fixed, D) float32

        # Encoder
        z = tok
        for blk in self.blocks:
            z = blk(z)                                    # (B, T_fixed, D)

        # Flatten all tokens -> head
        z_flat = z.reshape(B, self.T_fixed * z.size(-1))  # (B, T_fixed*D)
        out = self.head(z_flat)                           # (B, 1)
        return out




# PATCHES (corrections to the notebook version, see CHANGES.md):
#  1. The notebook built `head` and `pos_embed` during the first forward pass, after the
#     optimizer had been created, so they were never trained.  They are now built up front.
#  2. The notebook applied RevIN to the input but never de-normalised the output, so the
#     model could not recover the absolute RSRP level.  The output is now de-normalised.
#  3. patchify_1d_tailpad returns (B, patch_len, T_p), not the documented (B, T_p, patch_len),
#     and LazyLinear silently accepted it, so attention ran over the 16 positions inside a
#     patch instead of over the 3 patches.  The axes are now swapped before the projection.
class PatchTST(PatchTSTPaper):
    def __init__(self, L, patch_len, stride, d_model, n_heads, n_layers, d_ff=256, dropout=0.2):
        super().__init__(L, patch_len, stride, d_model, n_heads, n_layers, d_ff, dropout)
        self.proj = nn.Linear(patch_len, d_model)
        T_p = math.ceil((L - patch_len) / stride) + 1 if L > patch_len else 1
        self._build_once(T_p, device=None, dtype=None)

    def forward(self, x):
        B = x.size(0)
        x_norm, mean, var = self.revin(x)
        patches, _, _ = patchify_1d_tailpad(x_norm, self.patch_len, self.stride)
        tok = self.proj(patches.transpose(1, 2)) + self.pos_embed.unsqueeze(0)
        z = tok
        for blk in self.blocks:
            z = blk(z)
        out = self.head(z.reshape(B, self.T_fixed * z.size(-1)))
        return self.revin.denorm(out.unsqueeze(1), mean, var).squeeze(1)
