"""ETSformer -- model definition copied verbatim from cell 48 of the archived
experiment notebook (MS3M_baselines.ipynb, sha256 24a14001c0fe7b19).
No edits."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESLevelTrend(nn.Module):
    def __init__(self, d_model: int, alpha_init=0.4, beta_init=0.2):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(alpha_init)))
        self.logit_beta  = nn.Parameter(torch.logit(torch.tensor(beta_init)))
        self.d_model = d_model
    def forward(self, x):  # x: (B,T,D)
        B, T, D = x.shape
        a = torch.sigmoid(self.logit_alpha)
        b = torch.sigmoid(self.logit_beta)
        l_prev = x[:, 0, :]   # initialize level with first obs
        tr_prev = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        levels = []; trends = []; seas = []
        for t in range(T):
            y_t = x[:, t, :]
            s_t = y_t - (l_prev + tr_prev)
            l_t = a * y_t + (1 - a) * (l_prev + tr_prev)
            b_t = b * (l_t - l_prev) + (1 - b) * tr_prev
            levels.append(l_t); trends.append(b_t); seas.append(s_t)
            l_prev, tr_prev = l_t, b_t
        Ls = torch.stack(levels, dim=1); Bs = torch.stack(trends, dim=1); Ss = torch.stack(seas, dim=1)
        return Ls, Bs, Ss  # (B,T,D)


class FrequencyAttention(nn.Module):
    """
    Select Top-K frequency bins by average magnitude across channels.
    Reconstruct seasonal with only Top-K bins; also compute next-step by one-step
    phase shift in frequency domain: multiply by exp(+i*omega) then irfft; take t=0 sample.
    """
    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk

    @staticmethod
    def _omega_vec(n_fft, device, dtype):
        # frequencies for rfft bins: k=0..n_fft//2, omega = 2π k / n_fft
        k = torch.arange(0, n_fft//2 + 1, device=device, dtype=dtype)
        return 2.0 * math.pi * k / float(n_fft)

    def forward(self, x):  # x: (B,T,D) float
        B, T, D = x.shape
        X = torch.fft.rfft(x.float(), dim=1)                 # (B, T_r, D) complex
        mag = X.abs().mean(dim=-1)                           # (B, T_r)
        K = min(self.topk, mag.size(1))
        topv, topi = torch.topk(mag, k=K, dim=1)             # (B,K)

        # Build mask
        mask = torch.zeros_like(X, dtype=X.dtype)
        # Scatter selected bins from X to mask
        idx = topi.unsqueeze(-1).expand(B, K, D)             # (B,K,D)
        mask.scatter_(1, idx, X.gather(1, idx))

        # Seasonal reconstruction with Top-K only
        s_rec = torch.fft.irfft(mask, n=T, dim=1).to(x.dtype)  # (B,T,D)

        # Next-step via one-step phase shift (circular) on selected spectrum
        omega = self._omega_vec(T, device=X.device, dtype=X.real.dtype)  # (T_r,)
        shift = torch.exp(1j * omega).view(1, -1, 1)                    # (1,T_r,1)
        mask_shift = mask * shift                                       # (B,T_r,D)
        seq_shift = torch.fft.irfft(mask_shift, n=T, dim=1).to(x.dtype) # (B,T,D)
        s_next = seq_shift[:, 0, :]                                     # (B,D) next-step (circular shift)

        return s_rec, s_next


class GrowthLagFFT(nn.Module):
    """
    Learn per-channel lag kernels (length Lk) and apply via FFT convolution
    to model growth/low-frequency trend. Includes learnable damping gamma for next-step.
    """
    def __init__(self, d_model: int, lag_len: int, gamma_init=0.85, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.lag_len = lag_len
        self.kernel = nn.Parameter(torch.randn(d_model, lag_len) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.logit_gamma = nn.Parameter(torch.logit(torch.tensor(float(gamma_init))))

    def forward(self, x):  # x: (B,T,D)
        B, T, D = x.shape
        k = self.kernel  # (D,Lk)
        Lk = k.size(1)

        # pad kernel to T and FFT
        pad = T - Lk
        k_pad = F.pad(k, (0, pad))               # (D,T)
        Kf = torch.fft.rfft(k_pad.float(), dim=1)  # (D, Tr)

        # FFT of sequence per channel
        Xf = torch.fft.rfft(x.float().transpose(1,2), dim=2)  # (B,D,Tr)
        Yf = Xf * Kf.unsqueeze(0)                             # (B,D,Tr)
        y = torch.fft.irfft(Yf, n=T, dim=2).transpose(1,2).to(x.dtype)  # (B,T,D)
        y = self.drop(y)

        # next-step growth via damping of last token
        gamma = torch.sigmoid(self.logit_gamma)
        g_next = gamma * y[:, -1, :]   # (B,D)
        return y, g_next


class ETSDecompLayer(nn.Module):
    """
    One layer:
      - Seasonal via FrequencyAttention (Top-K)
      - Growth via Lag FFT conv (ESA-like)
      - Residual peel: x <- x - s - g
    """
    def __init__(self, d_model, topk, lag_len, gamma_init, dropout):
        super().__init__()
        self.fa = FrequencyAttention(topk=topk)
        self.growth = GrowthLagFFT(d_model, lag_len=lag_len, gamma_init=gamma_init, dropout=dropout)
        self.ln_s = nn.LayerNorm(d_model)
        self.ln_g = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B,T,D)
        s_rec, s_next = self.fa(x)            # (B,T,D), (B,D)
        g_rec, g_next = self.growth(x)        # (B,T,D), (B,D)
        s_rec = self.ln_s(s_rec + 0.0)        # stabilize
        g_rec = self.ln_g(g_rec + 0.0)
        res = x - s_rec - g_rec               # residual for next layers
        return res, s_rec, g_rec, s_next, g_next


class ETSformerPaper(nn.Module):
    def __init__(self, L: int, F: int, d_model: int, n_layers: int,
                 d_ff_mult: int, dropout: float, dropout_att: float,
                 topk_freq: int, max_lag: int, rsrp_index: int,
                 use_es_baseline: bool, alpha_init: float, beta_init: float):
        super().__init__()
        self.L=L; self.F=F; self.D=d_model; self.tgt=rsrp_index
        self.proj = nn.Linear(F, d_model)

        self.layers = nn.ModuleList([
            ETSDecompLayer(d_model, topk=topk_freq, lag_len=max_lag, gamma_init=0.85, dropout=dropout_att)
            for _ in range(n_layers)
        ])

        self.use_es = use_es_baseline
        if use_es_baseline:
            self.es = ESLevelTrend(d_model, alpha_init=alpha_init, beta_init=beta_init)
            self.readout_es = nn.Linear(d_model, 1)

        # Heads for seasonal/growth contributions to scalar RSRP
        self.readout_season = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.readout_growth  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        # x: (B,L,F)
        B, L, F = x.shape
        h = self.proj(x)  # (B,L,D)

        # Optional ES baseline term
        es_next_scalar = 0.0
        if self.use_es:
            Ls, Bs, _ = self.es(h)                  # (B,L,D)
            lt_next = Ls[:, -1, :] + Bs[:, -1, :]   # (B,D)
            es_next_scalar = self.readout_es(lt_next)  # (B,1)

        residual = h
        s_next_sum = torch.zeros(B, self.D, device=x.device, dtype=x.dtype)
        g_next_sum = torch.zeros(B, self.D, device=x.device, dtype=x.dtype)

        # peel seasonal & growth per layer
        for layer in self.layers:
            residual, s_rec, g_rec, s_next, g_next = layer(residual)
            s_next_sum = s_next_sum + s_next
            g_next_sum = g_next_sum + g_next

        # Map contributions to scalar RSRP
        y_season = self.readout_season(s_next_sum)  # (B,1)
        y_growth = self.readout_growth(g_next_sum)  # (B,1)

        y_hat = y_season + y_growth
        if self.use_es:
            y_hat = y_hat + es_next_scalar
        return y_hat  # (B,1)
