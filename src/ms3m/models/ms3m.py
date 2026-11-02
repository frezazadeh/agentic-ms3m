"""MS3M forecaster.

Default arguments reproduce the architecture of cell 0 of the original notebook
MS3M(3)(2).ipynb: width 128, four blocks, state size 64, four mixture components,
698,449 trainable parameters.

Tap order.  F.conv1d is a cross-correlation: with left padding, weight index j multiplies the
input L-1-j steps before the current one.  The taps k[l] (lag l) are therefore flipped before the
call, so that every position computes y[n] = sum_l k[l] u[n-l], the response of the recursion
s[n] = A_bar s[n-1] + B_bar u[n], y[n] = C s[n] + D u[n] with s[-1] = 0.  The original notebook
passed the taps unflipped; with its newest-first windows this gave the same response at the
read-out position.  tap_order="as_conv1d" keeps that behaviour for the ablation study.

The keyword arguments that did not exist in the notebook (kernel_len, se_mode, a_mode,
learn_dt, tap_order) only exist for the ablation study.
Internal class names keep the notebook's "S6" prefix, which was the working name of MS3M.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def hippo_legs_matrix(N: int, device=None, dtype=torch.float32):
    n = torch.arange(N, device=device, dtype=dtype)
    two_n1 = 2 * n + 1
    sq = torch.sqrt(two_n1[:, None] * two_n1[None, :])
    A = torch.tril(sq, diagonal=-1) * (-1.0)
    A = A + torch.diag(-(n + 1.0))
    B = torch.sqrt(two_n1)
    return A, B


class HiPPOLegSKernel(nn.Module):
    """One mixture component: Tustin-discretised LegS SSM rendered as FIR taps.

    Convention: s[k] = A_bar s[k-1] + B_bar u[k],  y[k] = C s[k] + D u[k], so that
    k[0] = C B_bar + D and k[l] = C A_bar^l B_bar for l >= 1.
    """

    def __init__(self, d_model, state_dim, L, dt_init=0.1, a_mode="legs", learn_dt=True):
        super().__init__()
        self.d_model, self.N, self.L = d_model, state_dim, L
        A, B = hippo_legs_matrix(state_dim)
        if a_mode == "diag":  # ablation: keep only the stable diagonal of the LegS operator
            A = torch.diag(torch.diagonal(A))
        elif a_mode != "legs":
            raise ValueError(a_mode)
        self.register_buffer("A_ct", A)
        self.register_buffer("B_ref", B)
        B0 = B[None, :].repeat(d_model, 1)
        self.B = nn.Parameter(B0 + 0.01 * torch.randn_like(B0))
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(dt_init, dtype=torch.float32)), requires_grad=learn_dt)

    def discretize(self):
        dt = F.softplus(self.log_dt) + 1e-6
        A = self.A_ct
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        lhs = I - 0.5 * dt * A
        rhs = I + 0.5 * dt * A
        A_bar = torch.linalg.solve(lhs, rhs)
        B_bar = torch.linalg.solve(lhs, (dt * self.B).transpose(0, 1)).transpose(0, 1)
        return A_bar, B_bar

    def forward(self):
        A_bar, B_bar = self.discretize()
        A_bar_T = A_bar.transpose(0, 1)
        x = B_bar
        out = []
        for _ in range(self.L):
            out.append((self.C * x).sum(dim=-1))
            x = x @ A_bar_T
        K = torch.stack(out, dim=1)
        K = K + F.pad(self.D[:, None], (0, self.L - 1))  # D only enters tap 0
        return K.unsqueeze(1)


class S6MixtureKernel(nn.Module):
    def __init__(self, d_model, state_dim, L, n_components=4, dt_init=0.1, a_mode="legs", learn_dt=True):
        super().__init__()
        self.components = nn.ModuleList([
            HiPPOLegSKernel(d_model, state_dim, L, dt_init=dt_init * (1.5 ** i), a_mode=a_mode, learn_dt=learn_dt)
            for i in range(n_components)
        ])

    def forward(self):
        """Sum of the component taps, computed for all components at once (same result as summing comp())."""
        disc = [c.discretize() for c in self.components]
        A_T = torch.stack([d[0] for d in disc]).transpose(1, 2)        # (m, N, N)
        x = torch.stack([d[1] for d in disc])                          # (m, d, N)
        C = torch.stack([c.C for c in self.components])                # (m, d, N)
        L = self.components[0].L
        out = []
        for _ in range(L):
            out.append((C * x).sum(dim=(0, 2)))
            x = torch.bmm(x, A_T)
        K = torch.stack(out, dim=1)
        D = torch.stack([c.D for c in self.components]).sum(0)
        K = K + F.pad(D[:, None], (0, L - 1))  # D only enters tap 0
        return K.unsqueeze(1)


class ChannelMix(nn.Module):
    def __init__(self, d_model, dropout=0.1, mult=2, use_glu=True):
        super().__init__()
        hidden = mult * d_model
        self.use_glu = use_glu
        if use_glu:
            self.up = nn.Linear(d_model, 2 * hidden)
            self.act = nn.GELU()
            self.down = nn.Linear(hidden, d_model)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model)
            )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.use_glu:
            a, g = torch.chunk(self.up(x), 2, dim=-1)
            y = self.down(self.act(a) * torch.sigmoid(g))
        else:
            y = self.net(x)
        return self.norm(x + self.dropout(y))


class SEGate(nn.Module):
    """Squeeze-and-excitation over time.

    mode="window": the notebook's gate; one gate per window from the mean over the whole
    window, so position t depends on later positions of the same window.
    mode="causal": the gate at position t uses only positions <= t.
    mode="none": identity (ablation).
    """

    def __init__(self, d_model, r=8, mode="window"):
        super().__init__()
        self.mode = mode
        if mode == "none":
            return
        self.fc = nn.Sequential(
            nn.Linear(d_model, max(1, d_model // r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, d_model // r), d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.mode == "none":
            return x
        if self.mode == "window":
            return x * self.fc(x.mean(dim=1)).unsqueeze(1)
        t = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype)[None, :, None]
        return x * self.fc(x.cumsum(dim=1) / t)


class S6Block(nn.Module):
    def __init__(self, d_model, state_dim, L, dropout=0.1, dt_init=0.1, n_components=4,
                 use_glu=True, se_mode="window", a_mode="legs", learn_dt=True, tap_order="ssm"):
        super().__init__()
        if tap_order not in ("ssm", "as_conv1d"):
            raise ValueError(tap_order)
        self.tap_order = tap_order
        self.kernel = S6MixtureKernel(d_model, state_dim, L, n_components, dt_init, a_mode, learn_dt)
        self.dropout = nn.Dropout(dropout)
        self.se = SEGate(d_model, mode=se_mode)
        self.norm1 = nn.LayerNorm(d_model)
        self.mix = ChannelMix(d_model, dropout=dropout, use_glu=use_glu)
        self.norm2 = nn.LayerNorm(d_model)
        self._cached = None

    def freeze_kernel(self):
        """Pre-compute the taps once for deployment; they are constant after training."""
        with torch.no_grad():
            self._cached = self.kernel().detach()

    def ssm_conv(self, x):
        """Causal depthwise convolution of x (B, T, C) with the state-space taps."""
        k = self._cached if self._cached is not None else self.kernel()
        if self.tap_order == "ssm":
            k = k.flip(-1)  # conv1d correlates; flipping makes tap l act on lag l
        Lk = k.shape[-1]
        return F.conv1d(F.pad(x.transpose(1, 2), (Lk - 1, 0)), k, groups=x.shape[-1]).transpose(1, 2)

    def forward(self, x):
        y = self.ssm_conv(x)
        y = self.dropout(self.se(y))
        y = self.norm1(x + y)
        z = self.mix(y)
        return self.norm2(y + z)


class MS3M(nn.Module):
    def __init__(self, in_dim, d_model=128, out_dim=1, L=32, n_layers=4, state_dim=64, dropout=0.1,
                 dt_init=0.1, n_components=4, kernel_len=None, use_glu=True, se_mode="window",
                 a_mode="legs", learn_dt=True, tap_order="ssm"):
        super().__init__()
        Lk = kernel_len or L
        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([
            S6Block(d_model, state_dim, Lk, dropout, dt_init, n_components, use_glu, se_mode, a_mode, learn_dt, tap_order)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def freeze_kernels(self):
        for b in self.blocks:
            b.freeze_kernel()

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1, :])
