"""Informer -- model definition copied verbatim from cell 34 of the archived
experiment notebook (MS3M_baselines.ipynb, sha256 24a14001c0fe7b19).
No edits."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbeddingConv1D(nn.Module):
    """ Token embedding via 1D Conv across time: Conv1d(F->D, k=3, pad=1) """
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=d_model, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
    def forward(self, x):  # x: (B,T,F)
        x = x.transpose(1,2)                     # (B,F,T)
        y = self.conv(x)                         # (B,D,T)
        return y.transpose(1,2)                  # (B,T,D)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, pe_type="learned"):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.mode = pe_type
        if pe_type == "learned":
            self.pe = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.normal_(self.pe, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe_buf", pe, persistent=False)
    def forward(self, x):  # (B,T,D)
        T = x.size(1)
        if self.mode == "learned":
            return x + self.pe[:T, :].unsqueeze(0)
        else:
            return x + self.pe_buf[:T, :].unsqueeze(0)


def causal_mask(T: int, device):
    # (T,T) bool; True=keep, False=mask
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))


class ProbSparseMHA(nn.Module):
    """
    For each head:
      - select Top-u queries by sparsity score s(q_i) = max_j (q_i k_j^T) - mean_j (q_i k_j^T)
      - compute full softmax on selected queries only
      - unselected queries get global context = mean(V)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float, prob_c: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.prob_c = prob_c

    def _u(self, Tq: int) -> int:
        return max(1, int(self.prob_c * math.log(max(Tq, 2))))

    def forward(self, q, k, v, attn_mask=None):
        # q: (B,Tq,D), k: (B,Tk,D), v: (B,Tk,D)
        B, Tq, D = q.shape
        Tk = k.shape[1]
        assert Tk == v.shape[1]
        h, d = self.h, self.d

        Q = self.q_proj(q).view(B, Tq, h, d).transpose(1, 2)  # (B,h,Tq,d)
        K = self.k_proj(k).view(B, Tk, h, d).transpose(1, 2)  # (B,h,Tk,d)
        V = self.v_proj(v).view(B, Tk, h, d).transpose(1, 2)  # (B,h,Tk,d)

        Bh = B * h
        Qh = Q.reshape(Bh, Tq, d)
        Kh = K.reshape(Bh, Tk, d)
        Vh = V.reshape(Bh, Tk, d)

        # AMP-safe: do attention math in float32
        Q32 = Qh.float(); K32 = Kh.float(); V32 = Vh.float()

        # Full scores for sparsity selection
        S = torch.matmul(Q32, K32.transpose(1, 2)) / math.sqrt(d)  # (Bh,Tq,Tk)
        if attn_mask is not None:
            add_mask_full = (~attn_mask).to(dtype=S.dtype) * (-1e9)  # (Tq,Tk)
            S = S + add_mask_full.unsqueeze(0)                       # (Bh,Tq,Tk)

        # ProbSparse: select Top-u queries by max-mean
        s_max, _ = S.max(dim=-1)   # (Bh,Tq)
        s_mean   = S.mean(dim=-1)  # (Bh,Tq)
        sparsity = s_max - s_mean  # (Bh,Tq)
        u = min(self._u(Tq), Tq)
        _, topi = torch.topk(sparsity, k=u, dim=1)   # (Bh,u)

        # Default context = mean(V)
        ctx = V32.mean(dim=1, keepdim=True).repeat(1, Tq, 1)  # (Bh,Tq,d)

        # Compute attention only for selected queries
        Q_sel = torch.gather(Q32, 1, topi.unsqueeze(-1).expand(Bh, u, d))   # (Bh,u,d)
        Sc    = torch.matmul(Q_sel, K32.transpose(1, 2)) / math.sqrt(d)     # (Bh,u,Tk)

        if attn_mask is not None:
            # Gather corresponding mask rows to match (Bh,u,Tk)
            mask_full = (~attn_mask).to(dtype=Sc.dtype) * (-1e9)            # (Tq,Tk)
            mask_batched = mask_full.unsqueeze(0).expand(Bh, -1, -1)        # (Bh,Tq,Tk)
            mask_sel = torch.gather(mask_batched, 1, topi.unsqueeze(-1).expand(Bh, u, Tk))  # (Bh,u,Tk)
            Sc = Sc + mask_sel

        A     = torch.softmax(Sc, dim=-1)       # (Bh,u,Tk)
        C_sel = torch.matmul(A, V32)            # (Bh,u,d)

        # dtype-safe scatter (avoid AMP mismatch)
        C_sel = C_sel.to(ctx.dtype)
        ctx.scatter_(1, topi.unsqueeze(-1).expand(Bh, u, d), C_sel)

        out = ctx.view(B, h, Tq, d).transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.o_proj(out.to(q.dtype))
        return self.drop(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_mult, dropout, prob_c):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = ProbSparseMHA(d_model, n_heads, dropout, prob_c)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_model * d_ff_mult),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model * d_ff_mult, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x, mask=None):
        # Encoder uses no causal mask
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x), None)
        x = x + self.ff(self.ln2(x))
        return x


class DistillBlock(nn.Module):
    """ Conv1d -> ELU -> MaxPool1d(stride=2) -> (B, T//2, D) """
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1)
        self.act  = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ln   = nn.LayerNorm(d_model)
    def forward(self, x):  # x: (B,T,D)
        y = self.conv(x.transpose(1,2))       # (B,D,T)
        y = self.pool(self.act(y))            # (B,D,T//2)
        y = y.transpose(1,2)
        return self.ln(y)


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_mult, dropout, prob_c, n_layers, distill=True):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff_mult, dropout, prob_c) for _ in range(n_layers)])
        self.distill = distill
        if distill and n_layers > 1:
            self.dists = nn.ModuleList([DistillBlock(d_model) for _ in range(n_layers-1)])
        else:
            self.dists = None
    def forward(self, x, self_mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=None)  # explicit: no mask in encoder
            if self.distill and (i < len(self.layers)-1):
                x = self.dists[i](x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_mult, dropout, prob_c):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_mha  = ProbSparseMHA(d_model, n_heads, dropout, prob_c)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_mha = ProbSparseMHA(d_model, n_heads, dropout, prob_c)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_model * d_ff_mult),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model * d_ff_mult, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, y, mem, self_mask=None):
        # pass mask positionally to avoid kwarg issues
        y = y + self.self_mha(self.ln1(y), self.ln1(y), self.ln1(y), self_mask)
        y = y + self.cross_mha(self.ln2(y), mem, mem, None)
        y = y + self.ff(self.ln3(y))
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_mult, dropout, prob_c, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff_mult, dropout, prob_c) for _ in range(n_layers)])
    def forward(self, y, mem, self_mask=None):
        for layer in self.layers:
            y = layer(y, mem, self_mask=self_mask)
        return y


class InformerPaper(nn.Module):
    def __init__(self, L: int, F: int, d_model: int, n_heads: int, n_layers_enc: int, n_layers_dec: int,
                 d_ff_mult: int, dropout: float, prob_c: float, rsrp_index: int, pe_type: str,
                 distill: bool, pred_len: int, label_frac: float, use_causal_mask: bool):
        super().__init__()
        self.L = L; self.F = F; self.D = d_model
        self.pred_len = pred_len
        self.label_len = max(1, int(L * label_frac))
        self.rsrp_index = rsrp_index
        self.use_causal_mask = use_causal_mask

        # Embeddings
        self.enc_tok = TokenEmbeddingConv1D(F, d_model)
        self.dec_tok = TokenEmbeddingConv1D(F, d_model)
        self.pe_enc  = PositionalEncoding(L, d_model, pe_type=pe_type)
        self.pe_dec  = PositionalEncoding(self.label_len + pred_len, d_model, pe_type=pe_type)

        # Stacks
        self.encoder = Encoder(d_model, n_heads, d_ff_mult, dropout, prob_c, n_layers_enc, distill=distill)
        self.decoder = Decoder(d_model, n_heads, d_ff_mult, dropout, prob_c, n_layers_dec)

        # Head for RSRP scalar(s)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)  # emits H steps; we supervise next-step (last) here
        )

    def forward(self, x):
        """
        x: (B,L,F) standardized
        Decoder input: last label_len steps + pred_len zeros (non-autoregressive)
        """
        B, L, F = x.shape
        assert L == self.L and F == self.F

        # Build decoder input
        y_label = x[:, -self.label_len:, :]                                        # (B,label_len,F)
        y_zeros = torch.zeros(B, self.pred_len, F, device=x.device, dtype=x.dtype) # (B,H,F)
        y_in = torch.cat([y_label, y_zeros], dim=1)                                 # (B,label_len+H,F)

        # Embeddings
        enc = self.pe_enc(self.enc_tok(x))               # (B,L,D)
        dec = self.pe_dec(self.dec_tok(y_in))            # (B,label_len+H,D)

        # Masks
        self_mask = None
        if self.use_causal_mask:
            T = dec.size(1)
            self_mask = causal_mask(T, dec.device)       # (T,T) bool

        # Encoder/Decoder
        mem = self.encoder(enc, self_mask=None)          # encoder uses no causal mask
        out = self.decoder(dec, mem, self_mask=self_mask)

        # Map to RSRP(s): take last decoder token
        last = out[:, -1, :]                             # (B,D)
        y_hat_seq = self.head(last)                      # (B,H)
        y_hat_rsrp = y_hat_seq[:, -1:]                  # (B,1) next-step
        return y_hat_rsrp
