"""FEDformer -- model definition copied verbatim from cell 19 of the archived
experiment notebook (MS3M_baselines.ipynb, sha256 24a14001c0fe7b19).
No edits."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MOEDecomp(nn.Module):
    """
    Mixture of moving-average filters for trend; seasonal = x - trend.
    - K fixed-kernel depthwise convs (MA with window k_i)
    - Learnable per-feature mixture logits -> softmax -> weights over K
    """
    def __init__(self, F: int, kernels=(3,5,7,11,25)):
        super().__init__()
        self.F = F
        self.kernels = list(kernels)
        self.K = len(kernels)
        # Depthwise convs (per feature)
        self.filters = nn.ModuleList()
        for k in self.kernels:
            conv = nn.Conv1d(F, F, kernel_size=k, padding=k//2, groups=F, bias=False)
            with torch.no_grad():
                w = torch.zeros(F, 1, k)
                w[:, 0, :] = 1.0 / k
                conv.weight.copy_(w)
            conv.weight.requires_grad = False  # fixed average kernels
            self.filters.append(conv)
        # Per-feature mixture logits (learnable, K for each feature)
        self.mix_logits = nn.Parameter(torch.zeros(F, self.K))

    def forward(self, x):
        """
        x: (B, L, F)
        returns seasonal, trend (both (B,L,F))
        """
        B, L, F = x.shape
        x1 = x.transpose(1, 2)           # (B,F,L) for conv
        comps = []
        for conv in self.filters:
            comps.append(conv(x1))        # each (B,F,L)
        comps = torch.stack(comps, dim=1) # (B,K,F,L)

        # softmax over K (per feature) -> (F,K)
        weights = torch.softmax(self.mix_logits, dim=-1)   # (F,K)
        w = weights.transpose(0,1).unsqueeze(0).unsqueeze(-1)  # (1,K,F,1)
        trend = (comps * w).sum(dim=1)                     # (B,F,L)
        trend = trend.transpose(1, 2)                      # (B,L,F)
        seasonal = x - trend
        return seasonal, trend


def random_modes(T: int, n_keep: int, device):
    """
    Return up to n_keep UNIQUE mode indices in [0, n_freq-1], with a low-frequency bias,
    always including DC (0) when possible.
    """
    n_freq = T // 2 + 1
    k = min(n_keep, n_freq)
    if k <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    low_pool = min(2 * k, n_freq)
    k_low = min(k // 2, low_pool)
    k_rem = k - k_low

    low_choices = torch.randperm(low_pool, device=device)[:k_low]

    mask = torch.ones(n_freq, dtype=torch.bool, device=device)
    mask[low_choices] = False
    rest_pool = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if rest_pool.numel() > 0 and k_rem > 0:
        rest_perm = rest_pool[torch.randperm(rest_pool.numel(), device=device)]
        rest_choices = rest_perm[:k_rem]
        idx = torch.cat([low_choices, rest_choices], dim=0)
    else:
        idx = low_choices

    # Ensure DC (0) present
    if (idx == 0).sum() == 0:
        if idx.numel() < k:
            idx = torch.cat([torch.tensor([0], device=device), idx], dim=0)
        else:
            idx[0] = 0

    idx = idx[:k]
    return idx.sort()[0]


class FEB(nn.Module):
    """
    Encoder block: layernorm -> Fourier mixing on random modes -> residual
                  -> MLP -> residual
    Uses fixed-size learnable complex weights of shape (modes, D).
    """
    def __init__(self, d_model: int, d_ff_mult: int = 4, dropout: float = 0.1, modes: int = 32):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.modes = modes
        self.d_model = d_model
        # Fixed-size learnable complex weights (real/imag)
        self.wr = nn.Parameter(torch.zeros(modes, d_model))
        self.wi = nn.Parameter(torch.zeros(modes, d_model))
        nn.init.normal_(self.wr, std=0.02)
        nn.init.normal_(self.wi, std=0.02)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * d_ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        z = self.ln1(x)
        X = torch.fft.rfft(z, dim=1)                      # (B, n_freq, D) complex
        n_freq = X.size(1)
        if n_freq == 0:
            y = torch.zeros_like(x)
        else:
            idx = random_modes(T, self.modes, device=x.device)  # (k,)
            k = idx.numel()
            if k > 0:
                Wr, Wi = self.wr[:k, :], self.wi[:k, :]         # (k,D)
                Xk = X.index_select(1, idx)                      # (B,k,D)
                Xk_mod = torch.complex(
                    Xk.real * (1.0 + Wr) - Xk.imag * Wi,
                    Xk.real * Wi + Xk.imag * (1.0 + Wr)
                )
                X = X.scatter(1, idx.view(1, -1, 1).expand(B, k, D), Xk_mod)
            y = torch.fft.irfft(X, n=T, dim=1)                  # (B,T,D)
        x = x + self.dropout(y)
        x = x + self.ff(self.ln2(x))
        return x


class FEA(nn.Module):
    """
    Decoder cross 'attention' in frequency domain with fixed-size weights.
    """
    def __init__(self, d_model: int, d_ff_mult: int = 4, dropout: float = 0.1, modes: int = 32):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.modes = modes
        self.d_model = d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * d_ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_ff_mult, d_model),
            nn.Dropout(dropout),
        )
        # Fixed-size learnable complex affines per mode for Q/K/V
        self.wq_r = nn.Parameter(torch.zeros(modes, d_model)); nn.init.normal_(self.wq_r, std=0.02)
        self.wq_i = nn.Parameter(torch.zeros(modes, d_model)); nn.init.normal_(self.wq_i, std=0.02)
        self.wk_r = nn.Parameter(torch.zeros(modes, d_model)); nn.init.normal_(self.wk_r, std=0.02)
        self.wk_i = nn.Parameter(torch.zeros(modes, d_model)); nn.init.normal_(self.wk_i, std=0.02)
        self.wv_r = nn.Parameter(torch.zeros(modes, d_model)); nn.init.normal_(self.wv_r, std=0.02)
        self.wv_i = nn.Parameter(torch.zeros(modes, d_model)); nn.init.normal_(self.wv_i, std=0.02)

    def forward(self, dec, enc):
        """
        dec: (B, T_d, D) seasonal hidden from decoder input
        enc: (B, T_e, D) seasonal hidden from encoder output (FFT length aligned to T_d)
        """
        B, Td, D = dec.shape
        q = self.ln_q(dec)
        k = self.ln_kv(enc); v = k

        Qf = torch.fft.rfft(q, n=Td, dim=1)
        Kf = torch.fft.rfft(k, n=Td, dim=1)
        Vf = torch.fft.rfft(v, n=Td, dim=1)

        n_freq = Qf.size(1)
        if n_freq == 0:
            Yf = Qf
        else:
            idx = random_modes(Td, self.modes, device=dec.device)  # (k,)
            k_modes = idx.numel()
            if k_modes > 0:
                def mod(X, wr, wi):
                    wr, wi = wr[:k_modes, :], wi[:k_modes, :]
                    Xk = X.index_select(1, idx)                    # (B,k,D)
                    Xk_mod = torch.complex(
                        Xk.real * (1.0 + wr) - Xk.imag * wi,
                        Xk.real * wi + Xk.imag * (1.0 + wr)
                    )
                    return X.scatter(1, idx.view(1, -1, 1).expand(B, k_modes, D), Xk_mod)

                Qf = mod(Qf, self.wq_r, self.wq_i)
                Kf = mod(Kf, self.wk_r, self.wk_i)
                Vf = mod(Vf, self.wv_r, self.wv_i)

                Qk = Qf.index_select(1, idx)
                Kk = Kf.index_select(1, idx)
                Vk = Vf.index_select(1, idx)
                att = Qk * torch.conj(Kk)
                Yk = att * Vk
                Yf = Qf.clone()
                Yf = Yf.scatter(1, idx.view(1, -1, 1).expand(B, k_modes, D), Yk)
            else:
                Yf = Qf

        y = torch.fft.irfft(Yf, n=Td, dim=1)
        dec = dec + self.dropout(y)
        dec = dec + self.ff(self.ln2(dec))
        return dec


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_ff_mult: int, dropout: float, modes: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            FEB(d_model, d_ff_mult=d_ff_mult, dropout=dropout, modes=modes)
            for _ in range(n_layers)
        ])
    def forward(self, x):  # x: (B,T,D)
        for blk in self.blocks:
            x = blk(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_ff_mult: int, dropout: float, modes: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            FEA(d_model, d_ff_mult=d_ff_mult, dropout=dropout, modes=modes)
            for _ in range(n_layers)
        ])
    def forward(self, x_dec, x_enc):  # (B,Td,D), (B,Te,D)
        h = x_dec
        for blk in self.blocks:
            h = blk(h, x_enc)
        return h


class FEDFormerPaper(nn.Module):
    """
    Time-as-sequence encoder-decoder with MOE decomposition and frequency-enhanced blocks.
    """
    def __init__(self, L: int, F: int, d_model: int, n_layers_enc: int, n_layers_dec: int,
                 d_ff_mult: int, dropout: float, modes: int, rsrp_index: int, ma_kernels):
        super().__init__()
        self.L = L
        self.F = F
        self.D = d_model
        self.rsrp_index = rsrp_index
        self.label_len = L            # use all L as label length
        self.pred_len = 1

        # Decomposition
        self.decomp = MOEDecomp(F=F, kernels=ma_kernels)

        # Embeddings (per time step)
        self.enc_embed = nn.Linear(F, d_model)
        self.dec_embed = nn.Linear(F, d_model)

        # Stacks
        self.encoder = Encoder(d_model, n_layers_enc, d_ff_mult, dropout, modes)
        self.decoder = Decoder(d_model, n_layers_dec, d_ff_mult, dropout, modes)

        # Head: map last decoder token to 1-step seasonal output (RSRP)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        x: (B,L,F), standardized
        """
        B, L, F = x.shape
        assert L == self.L and F == self.F

        # 1) Decompose encoder input
        s_enc, t_enc = self.decomp(x)              # (B,L,F) each

        # 2) Build decoder inputs: last label_len seasonal + one zero; trend extended by repeat
        s_label = s_enc[:, -self.label_len:, :]    # (B,L,F)
        t_label = t_enc[:, -self.label_len:, :]    # (B,L,F)
        s_dec_in = torch.cat([s_label, torch.zeros(B, self.pred_len, F, device=x.device, dtype=x.dtype)], dim=1)  # (B,L+1,F)
        t_last = t_label[:, -1:, :]                # (B,1,F)
        t_dec_in = torch.cat([t_label, t_last.repeat(1, self.pred_len, 1)], dim=1)                                 # (B,L+1,F)

        # 3) Encoder
        h_enc = self.enc_embed(s_enc)              # (B,L,D)
        h_enc = self.encoder(h_enc)                # (B,L,D)

        # 4) Decoder with FEA (cross frequency mixing)
        h_dec = self.dec_embed(s_dec_in)           # (B,L+1,D)
        h_dec = self.decoder(h_dec, h_enc)         # (B,L+1,D)

        # 5) Seasonal prediction for next step = last decoder position
        h_last = h_dec[:, -1, :]                   # (B,D)
        s_pred_rsrp = self.head(h_last)            # (B,1)

        # 6) Trend prediction for next step = last trend value for RSRP channel
        t_pred_rsrp = t_dec_in[:, -1, self.rsrp_index:self.rsrp_index+1]  # (B,1)

        # 7) Sum (seasonal + trend) for final next-step RSRP (standardized units)
        y_hat = s_pred_rsrp + t_pred_rsrp
        return y_hat
