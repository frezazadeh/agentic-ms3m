"""Temporal Fusion Transformer -- model definition copied verbatim from cell 41 of the archived
experiment notebook (MS3M_baselines.ipynb, sha256 24a14001c0fe7b19).
No edits."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin_a = nn.Linear(d, d)
        self.lin_b = nn.Linear(d, d)
    def forward(self, x):
        return self.lin_a(x) * torch.sigmoid(self.lin_b(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN(x, c) = LayerNorm( skip(x) + Dropout( GLU( ELU( W1 * [x; c] ) -> W2 ) ) )
    context c is optional (dim d_context); implemented as additive via linear.
    """
    def __init__(self, d_in, d_hidden, d_out, d_context=0, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in + d_context, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.glu = GLU(d_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.d_context = d_context
    def forward(self, x, c=None):
        if self.d_context and c is not None:
            if c.dim() == x.dim() - 1:  # (B, d_context) expand over time if needed
                c = c.unsqueeze(1).expand(x.size(0), x.size(1), -1)
            x_in = torch.cat([x, c], dim=-1)
        else:
            x_in = x
        y = F.elu(self.fc1(x_in))
        y = self.fc2(y)
        y = self.glu(y)
        y = self.dropout(y)
        return self.norm(self.skip(x) + y)


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        B, T = x.shape[:2]
        x_ = x.reshape(B*T, *x.shape[2:])
        y_ = self.module(x_)
        return y_.reshape(B, T, *y_.shape[1:])


class StaticEncoders(nn.Module):
    """
    Build static contexts (c_s, c_e, c_c, c_h) from static inputs s (B,S).
    If S==0 or s is None, returns learned zeros (implemented as zeros tensors).
    """
    def __init__(self, n_static: int, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.n_static = n_static
        if n_static > 0:
            self.embed = nn.Linear(n_static, d_model)
            self.grn_cs = GatedResidualNetwork(d_model, d_hidden, d_model, dropout=dropout)
            self.grn_ce = GatedResidualNetwork(d_model, d_hidden, d_model, dropout=dropout)
            self.grn_cc = GatedResidualNetwork(d_model, d_hidden, d_model, dropout=dropout)
            self.grn_ch = GatedResidualNetwork(d_model, d_hidden, d_model, dropout=dropout)
        else:
            # placeholders; we will create zeros in forward
            self.embed = None
    def forward(self, s, B: int, device, dtype):
        if self.n_static > 0 and s is not None:
            z = self.embed(s)  # (B,D)
            c_s = self.grn_cs(z.unsqueeze(1)).squeeze(1)
            c_e = self.grn_ce(z.unsqueeze(1)).squeeze(1)
            c_c = self.grn_cc(z.unsqueeze(1)).squeeze(1)
            c_h = self.grn_ch(z.unsqueeze(1)).squeeze(1)
        else:
            c_s = torch.zeros(B, 128, device=device, dtype=dtype)
            c_e = torch.zeros(B, 128, device=device, dtype=dtype)
            c_c = torch.zeros(B, 128, device=device, dtype=dtype)
            c_h = torch.zeros(B, 128, device=device, dtype=dtype)
        return c_s, c_e, c_c, c_h


class PerVariableGRN(nn.Module):
    """Applies a GRN separately to each variable's temporal series."""
    def __init__(self, d_in, d_hidden, d_out, d_context=0, dropout=0.1):
        super().__init__()
        self.grn = GatedResidualNetwork(d_in, d_hidden, d_out, d_context=d_context, dropout=dropout)
    def forward(self, x, c=None):  # x: (B,T,1) -> returns (B,T,D)
        return self.grn(x, c=c)


class VariableSelectionNetwork(nn.Module):
    """
    Variable selection per paper:
      - For each feature f, pass x_f through a GRN to produce representation v_f(t) in R^D.
      - Compute selection weights w_f(t) via scalar GRNs + softmax conditioned on c_s.
      - Aggregate z(t) = sum_f w_f(t) * v_f(t).
    """
    def __init__(self, n_feat: int, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.n_feat = n_feat
        self.value_grns = nn.ModuleList([
            PerVariableGRN(1, d_hidden, d_model, d_context=d_model, dropout=dropout) for _ in range(n_feat)
        ])
        self.weight_grns = nn.ModuleList([
            PerVariableGRN(1, d_hidden, 1, d_context=d_model, dropout=dropout) for _ in range(n_feat)
        ])
        self.post_grn = GatedResidualNetwork(d_model, d_hidden, d_model, d_context=d_model, dropout=dropout)

    def forward(self, x, c_s):  # x: (B,T,F), c_s: (B,D)
        B, T, F = x.shape
        V = []
        a = []
        x_split = torch.unbind(x, dim=2)  # list of (B,T)
        for f in range(self.n_feat):
            xf = x_split[f].unsqueeze(-1)        # (B,T,1)
            vf = self.value_grns[f](xf, c=c_s)   # (B,T,D)
            af = self.weight_grns[f](xf, c=c_s)  # (B,T,1)
            V.append(vf)
            a.append(af)
        V = torch.stack(V, dim=2)                # (B,T,F,D)
        A = torch.softmax(torch.stack(a, dim=2), dim=2)  # (B,T,F,1)
        z = (A * V).sum(dim=2)                   # (B,T,D)
        z = self.post_grn(z, c=c_s)              # (B,T,D)
        return z, A.squeeze(-1)                  # A: (B,T,F)


class InterpretableMHAttention(nn.Module):
    """
    Interpretable multi-head attention as per TFT:
      - Q and K are per-head; V is SHARED across heads.
      - Outputs from heads are ADDITIVELY aggregated.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv_shared = nn.Linear(d_model, self.dh)  # shared across heads
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):  # q: (B,Tq,D), k,v: (B,Tk,D)
        B, Tq, D = q.shape
        Tk = k.size(1)
        Q = self.Wq(q).view(B, Tq, self.h, self.dh).transpose(1, 2)  # (B,h,Tq,dh)
        K = self.Wk(k).view(B, Tk, self.h, self.dh).transpose(1, 2)  # (B,h,Tk,dh)
        V = self.Wv_shared(v)                                        # (B,Tk,dh) shared V

        # scaled dot-product attention per head (with shared V)
        scores = torch.matmul(Q.float(), K.float().transpose(2, 3)) / math.sqrt(self.dh)  # (B,h,Tq,Tk)
        A = torch.softmax(scores, dim=-1)                                                 # (B,h,Tq,Tk)
        # per head context: (B,h,Tq,dh)
        ctx_h = torch.matmul(A, V.float().unsqueeze(1))                                   # (B,h,Tq,dh)
        # additive aggregation over heads
        ctx = ctx_h.sum(dim=1)                                                            # (B,Tq,dh)
        # tile to D by concatenating heads' shared-dim sums (repeat h times then linear)
        # To keep dims consistent with paper's additive aggregation, map back to D
        ctx_cat = torch.cat([ctx for _ in range(self.h)], dim=-1)                         # (B,Tq,h*dh)= (B,Tq,D)
        out = self.out(self.dropout(ctx_cat))                                             # (B,Tq,D)
        return out


class TFTPaper(nn.Module):
    """
    Full paper-style TFT for next-step RSRP (quantile outputs).
    Streams:
      - Observed inputs only (no known-future here); static encoders included (zeros if absent).
    """
    def __init__(self, L: int, F: int, d_model: int, d_hidden: int, n_heads: int,
                 dropout: float, dropout_att: float, rsrp_index: int,
                 n_static: int, quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.L = L; self.F = F; self.D = d_model; self.rsrp_index = rsrp_index
        self.quantiles = tuple(quantiles)

        # Static encoders -> contexts (c_s, c_e, c_c, c_h)
        self.static_enc = StaticEncoders(n_static, d_model, d_hidden, dropout)

        # Variable selection for past observed inputs
        self.vsn_obs = VariableSelectionNetwork(F, d_model, d_hidden, dropout)

        # Locality LSTM encoder/decoder with static init
        self.enc_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1,
                                batch_first=True, bidirectional=False)
        self.dec_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1,
                                batch_first=True, bidirectional=False)

        # GRNs to init LSTM states from (c_c, c_h)
        self.init_enc_h = nn.Linear(d_model, d_model)
        self.init_enc_c = nn.Linear(d_model, d_model)
        self.init_dec_h = nn.Linear(d_model, d_model)
        self.init_dec_c = nn.Linear(d_model, d_model)

        # Gated residual skip from encoder input to output (post-enc GRN)
        self.post_enc_grn = GatedResidualNetwork(d_model, d_hidden, d_model, d_context=d_model, dropout=dropout)

        # Static enrichment before attention
        self.static_enrich = GatedResidualNetwork(d_model, d_hidden, d_model, d_context=d_model, dropout=dropout)

        # Interpretable multi-head attention over encoder outputs
        self.attn = InterpretableMHAttention(d_model, n_heads, dropout_att)

        # Output heads: quantile projections from fused representation
        if len(self.quantiles) > 0:
            self.head = nn.Linear(d_model, len(self.quantiles))
        else:
            self.head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, s=None):
        """
        x: (B,L,F) standardized
        s: (B,S) optional static; if None or S==0, zeros are used for contexts.
        """
        B, L, F = x.shape
        device, dtype = x.device, x.dtype

        # Static contexts
        c_s, c_e, c_c, c_h = self.static_enc(s, B, device, dtype)  # (B,D) each

        # Variable selection (observed)
        z_obs, w_obs = self.vsn_obs(x, c_s)      # (B,L,D), (B,L,F)

        # Encoder LSTM with static init
        h0_e = self.init_enc_h(c_h).unsqueeze(0) # (1,B,D)
        c0_e = self.init_enc_c(c_c).unsqueeze(0)
        enc_out, (hn_e, cn_e) = self.enc_lstm(z_obs, (h0_e, c0_e))  # (B,L,D)

        # Gated residual fusion with inputs (post-enc GRN)
        enc_fused = self.post_enc_grn(enc_out, c=c_e)               # (B,L,D)

        # One-step decoder input: use last encoder step representation
        dec_in_step = enc_fused[:, -1:, :]                          # (B,1,D)
        h0_d = self.init_dec_h(c_h).unsqueeze(0)
        c0_d = self.init_dec_c(c_c).unsqueeze(0)
        dec_out, (hn_d, cn_d) = self.dec_lstm(dec_in_step, (h0_d, c0_d))  # (B,1,D)

        # Static enrichment before attention
        dec_enriched = self.static_enrich(dec_out, c=c_e)           # (B,1,D)

        # Interpretable attention: query=decoder step, key/value=encoder fused timeline
        ctx = self.attn(dec_enriched, enc_fused, enc_fused)         # (B,1,D)
        fused = self.norm(dec_enriched + self.drop(ctx))            # (B,1,D)

        # Head
        yq = self.head(fused.squeeze(1))                            # (B,Q) or (B,1)
        return yq  # if quantiles: (B,Q); else: (B,1)


def pinball_loss(pred, target, quantiles):
    # pred: (B,Q), target: (B,1)
    # returns scalar mean pinball loss across all quantiles
    diff = target - pred  # broadcast along Q
    losses = []
    for i, q in enumerate(quantiles):
        e = torch.maximum(q * diff[:, i:i+1], (q - 1) * diff[:, i:i+1])
        losses.append(e)
    L = torch.mean(torch.cat(losses, dim=1))
    return L
