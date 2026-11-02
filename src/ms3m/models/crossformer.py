"""Crossformer -- model definition copied verbatim from cell 57 of the archived
experiment notebook (MS3M_baselines.ipynb, sha256 24a14001c0fe7b19).
No edits."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_Tp(L, P, S):
    # number of patches with right-tail padding
    return 1 + max(0, (L - P + S - 1) // S)


def rightpad_repeat(x, L_out):
    # x: (B,L,...) -> repeat last frame until length L_out
    if x.size(1) >= L_out:
        return x[:, :L_out]
    pad_len = L_out - x.size(1)
    last = x[:, -1:, :].expand(-1, pad_len, -1)
    return torch.cat([x, last], dim=1)


def patchify_tailpad(x, P, S):
    # x: (B, L, F) -> (B, F, Tp, P)
    B, L, F = x.shape
    Tp = calc_Tp(L, P, S)
    L_needed = (Tp - 1) * S + P
    x_pad = rightpad_repeat(x, L_needed)  # (B, L_needed, F)
    patches = []
    for s in range(0, L_needed - P + 1, S):
        patches.append(x_pad[:, s:s+P, :])  # (B,P,F)
    X = torch.stack(patches, dim=2)         # (B,P,Tp,F)
    X = X.permute(0, 3, 2, 1).contiguous()  # (B,F,Tp,P)
    return X, Tp


def merge_segments(x, m=2):
    # x: (B, F, Tp, D) -> average merge along Tp by factor m (pad-tail if needed)
    B, F, Tp, D = x.shape
    if Tp % m != 0:
        pad = m - (Tp % m)
        x_pad = torch.cat([x, x[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)  # (B,F,Tp+pad,D)
    else:
        x_pad = x
    Tp2 = x_pad.size(2) // m
    xg = x_pad.view(B, F, Tp2, m, D).mean(dim=3)  # (B,F,Tp2,D)
    return xg


class FFN(nn.Module):
    def __init__(self, d_model, d_ff_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * d_ff_mult),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * d_ff_mult, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class RouterCrossDimAttention(nn.Module):
    """
    Cross-dimension attention with routers (complexity O(F*c + c^2)).
    For each time index t (patch index), we:
      - compute soft assignments a_{f->c} from feature tokens to c routers
      - aggregate router tokens r_c = sum_f a_{f,c} * x_f
      - self-attend across routers
      - scatter back: x'_f = sum_c a_{f,c} * r'_c
    Shapes:
      in:  x (B, F, Tp, D)
      out: x (B, F, Tp, D)
    """
    def __init__(self, d_model, n_routers, n_heads, dropout_att=0.1, dropout=0.1):
        super().__init__()
        self.c = n_routers
        self.D = d_model
        self.score = nn.Linear(d_model, n_routers)          # router assignment per feature
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_att, batch_first=True)  # over routers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn  = FFN(d_model, dropout=dropout)

    def forward(self, x):  # (B,F,Tp,D)
        B, F, Tp, D = x.shape
        # process per time index in a batchy way:
        # reshape to group T as batch: (B*Tp, F, D)
        y = x.permute(0, 2, 1, 3).contiguous().view(B*Tp, F, D)
        y_ln = self.ln1(y)
        # soft assignments to routers
        a_logits = self.score(y_ln)                 # (B*Tp, F, c)
        a = torch.softmax(a_logits, dim=2)          # (B*Tp, F, c)
        # aggregate router tokens
        # r = a^T @ y  -> (B*Tp, c, D)
        r = torch.einsum('bfc,bfd->bcd', a, y_ln)
        # router self-attention (small c)
        r2, _ = self.mha(r, r, r, need_weights=False)  # (B*Tp, c, D)
        r2 = r + self.drop(r2)                         # residual at router level
        # scatter back: x' = a @ r2 (using same assignments)
        y2 = torch.einsum('bfc,bcd->bfd', a, r2)       # (B*Tp, F, D)
        y = y + self.drop(y2)
        y = y + self.ffn(self.ln2(y))
        # reshape back to (B,F,Tp,D)
        out = y.view(B, Tp, F, D).permute(0, 2, 1, 3).contiguous()
        return out


class IntraTemporalAttention(nn.Module):
    """
    Temporal attention within each feature independently.
    Input:  (B, F, Tp, D)
    Output: (B, F, Tp, D)
    """
    def __init__(self, d_model, n_heads, dropout_att=0.1, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_att, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn  = FFN(d_model, dropout=dropout)

    def forward(self, x):
        B, F, Tp, D = x.shape
        y = x.view(B*F, Tp, D)
        y_ln = self.ln1(y)
        z, _ = self.mha(y_ln, y_ln, y_ln, need_weights=False)
        y = y + self.drop(z)
        y = y + self.ffn(self.ln2(y))
        y = y.view(B, F, Tp, D)
        return y


class CrossformerBlock(nn.Module):
    """
    Two-Stage Attention block:
      1) Cross-dimension (with routers)
      2) Intra-temporal
    """
    def __init__(self, d_model, n_heads, n_routers, dropout_att=0.1, dropout=0.1):
        super().__init__()
        self.cross = RouterCrossDimAttention(d_model, n_routers, n_heads, dropout_att, dropout)
        self.temporal = IntraTemporalAttention(d_model, n_heads, dropout_att, dropout)

    def forward(self, x):
        x = self.cross(x)
        x = self.temporal(x)
        return x


class CrossformerPaper(nn.Module):
    def __init__(self, L: int, F: int, d_model: int, n_heads: int, d_ff_mult: int,
                 patch_len: int, stride: int, n_routers: int,
                 n_layers_scale1: int, n_layers_scale2: int, merge_factor: int,
                 dropout: float, dropout_att: float):
        super().__init__()
        self.L=L; self.F=F; self.D=d_model; self.P=patch_len; self.S=stride
        self.c = n_routers
        self.merge = merge_factor

        # DSW patch projection: (P)->D
        self.patch_proj = nn.Linear(self.P, d_model)

        # 2D positional embeddings: time (Tp) and feature (F)
        self.max_Tp = 2048
        self.time_pos = nn.Parameter(torch.randn(self.max_Tp, d_model) * 0.02)  # (Tp,D)
        self.feat_pos = nn.Parameter(torch.randn(F, d_model) * 0.02)            # (F,D)

        # Scale 1 blocks
        self.blocks1 = nn.ModuleList([
            CrossformerBlock(d_model, n_heads, n_routers, dropout_att, dropout)
            for _ in range(n_layers_scale1)
        ])
        # Scale 2 blocks (after segment merging by 'merge')
        self.blocks2 = nn.ModuleList([
            CrossformerBlock(d_model, n_heads, n_routers, dropout_att, dropout)
            for _ in range(n_layers_scale2)
        ])

        # Heads (multi-scale fusion): pool last time token across features at each scale, fuse
        self.ln_head1 = nn.LayerNorm(d_model)
        self.ln_head2 = nn.LayerNorm(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(2*d_model, d_model * d_ff_mult),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * d_ff_mult, 1)
        )

    def forward(self, x):  # x: (B,L,F)
        B, L, F = x.shape
        # DSW patchify
        patches, Tp = patchify_tailpad(x, self.P, self.S)  # (B,F,Tp,P)
        tok = self.patch_proj(patches)                     # (B,F,Tp,D)
        # add 2D pos-emb
        tp_pe = self.time_pos[:Tp].unsqueeze(0).unsqueeze(0)  # (1,1,Tp,D)
        fp_pe = self.feat_pos.unsqueeze(0).unsqueeze(2)       # (1,F,1,D)
        h1 = tok + tp_pe + fp_pe                              # (B,F,Tp,D)

        # Scale 1
        for blk in self.blocks1:
            h1 = blk(h1)                                      # (B,F,Tp,D)

        # Segment merging for Scale 2
        h2 = merge_segments(h1, m=self.merge)                 # (B,F,Tp2,D)

        # Scale 2
        for blk in self.blocks2:
            h2 = blk(h2)                                      # (B,F,Tp2,D)

        # Multi-scale fusion using last time token across features (mean)
        last1 = h1[:, :, -1, :].mean(dim=1)                   # (B,D)
        last2 = h2[:, :, -1, :].mean(dim=1)                   # (B,D)
        z = torch.cat([self.ln_head1(last1), self.ln_head2(last2)], dim=-1)  # (B,2D)
        y_hat = self.fuse(z)                                  # (B,1)
        return y_hat
