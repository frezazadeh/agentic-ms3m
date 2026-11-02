"""iTransformer -- model definition copied verbatim from cell 11 of the archived
experiment notebook (MS3M_baselines.ipynb, sha256 24a14001c0fe7b19).
See PATCHES at the bottom of this module for the corrections applied."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalLayerNorm(nn.Module):
    """
    Apply LayerNorm over the last dimension (time length L) for each variate, per sample.
    Lazily builds nn.LayerNorm with normalized_shape=L on first call (handles any L).
    """
    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.ln = None  # nn.LayerNorm(L)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, F, L)
        L = z.size(-1)
        if (self.ln is None) or (self.ln.normalized_shape != (L,)):
            self.ln = nn.LayerNorm(L, eps=self.eps, elementwise_affine=self.affine).to(z.device).to(z.dtype)
        return self.ln(z)


class ITransformer(nn.Module):
    """
    Channel-as-token encoder for next-step forecasting (paper-aligned).
    - Input x: (B, L, F). Treat each variable (F tokens) as a token.
    - Temporal LayerNorm per variate over time: (B,F,L) LN over L.
    - Series-wise projection: Linear(L -> D) shared across tokens.
    - (Optional) learned variable embeddings (F,D) — disabled by default to match paper minimal recipe.
    - Transformer encoder over F tokens.
    - Read the RSRP token (index=7) and map to scalar.
    """
    def __init__(self, L: int, F: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff_mult: int = 4, dropout: float = 0.1, rsrp_index: int = 7, use_token_embed: bool = False):
        super().__init__()
        self.F = F
        self.rsrp_index = rsrp_index
        self.temporal_ln = TemporalLayerNorm(affine=True)
        self.series_proj = nn.LazyLinear(d_model)  # projects length-L -> d_model (applies to last dim)
        self.use_token_embed = use_token_embed
        if use_token_embed:
            self.token_embed = nn.Parameter(torch.zeros(F, d_model))
            nn.init.trunc_normal_(self.token_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * d_ff_mult,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        x: (B, L, F) standardized multivariate history
        """
        B, L, F = x.shape
        # (B,L,F) -> (B,F,L), temporal LN over time per variate
        z = x.permute(0, 2, 1).contiguous()     # (B,F,L)
        z = self.temporal_ln(z)                  # (B,F,L)
        # Series-wise projection for each variable: (B,F,L) -> (B,F,D)
        z = self.series_proj(z)
        # Optional variable embeddings
        if self.use_token_embed:
            z = z + self.token_embed.unsqueeze(0)  # (B,F,D)
        # Transformer encoder over tokens (F = sequence length)
        z = self.encoder(z)                      # (B,F,D)
        # Read RSRP token and predict scalar
        h_rsrp = z[:, self.rsrp_index, :]        # (B,D)
        yhat = self.head(h_rsrp)                 # (B,1)
        return yhat




# PATCHES (corrections to the notebook version, see CHANGES.md):
#  1. The notebook's per-variate LayerNorm over time removed the level of every window and
#     the output was never de-normalised, so the absolute RSRP level was unrecoverable.
#     As in the iTransformer paper, the series are instance-normalised and the output is
#     de-normalised with the statistics of the target variate.
#  2. The lazily created LayerNorm was invisible to the optimizer; it is removed.
class ITransformerFixed(ITransformer):
    def __init__(self, L, F, d_model, n_heads, n_layers, d_ff_mult=4, dropout=0.1, rsrp_index=7, use_token_embed=False):
        super().__init__(L, F, d_model, n_heads, n_layers, d_ff_mult, dropout, rsrp_index, use_token_embed)
        self.series_proj = nn.Linear(L, d_model)
        del self.temporal_ln

    def forward(self, x):
        z = x.permute(0, 2, 1)
        mean = z.mean(dim=-1, keepdim=True)
        std = torch.sqrt(z.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        z = self.series_proj((z - mean) / std)
        if self.use_token_embed:
            z = z + self.token_embed.unsqueeze(0)
        z = self.encoder(z)
        out = self.head(z[:, self.rsrp_index, :])
        return out * std[:, self.rsrp_index] + mean[:, self.rsrp_index]
