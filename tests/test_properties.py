"""Checks of properties the paper states.  Run with: python -m pytest tests"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pytest  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ms3m import data as D, registry as Reg  # noqa: E402
from ms3m.models.ms3m import HiPPOLegSKernel, hippo_legs_matrix  # noqa: E402

PARAMS = dict(ms3m=698449, patchtst=449411, itransformer=814209, fedformer=755906, informer=1256449, tft=2510702,
              etsformer=10376, crossformer=1591321)


@pytest.mark.parametrize("name,count", PARAMS.items())
def test_parameter_counts(name, count):
    assert Reg.n_params(Reg.build(name, 32, 1 if name == "patchtst" else 13)) == count


def test_every_parameter_receives_a_gradient():
    for name in ("patchtst", "itransformer"):
        F = 1 if name == "patchtst" else 13
        m = Reg.build(name, 32, F)
        m(torch.randn(8, 32, F)).sum().backward()
        assert all(p.grad is not None for p in m.parameters()), name


def test_tustin_is_schur_stable():
    A, _ = hippo_legs_matrix(64, dtype=torch.float64)
    assert torch.linalg.eigvals(A).real.max() < 0
    for dt in (1e-3, 0.1, 1.0, 10.0):
        I = torch.eye(64, dtype=torch.float64)
        Ad = torch.linalg.solve(I - dt / 2 * A, I + dt / 2 * A)
        assert torch.linalg.eigvals(Ad).abs().max() < 1


def test_taps_match_the_state_space_recursion():
    torch.manual_seed(0)
    k = HiPPOLegSKernel(3, 8, 6)
    taps = k()[:, 0, :].detach()
    Ad, Bd = k.discretize()
    s = torch.zeros(3, 8)
    for ell in range(6):
        u = 1.0 if ell == 0 else 0.0
        s = s @ Ad.T + Bd * u
        y = (k.C * s).sum(-1) + k.D * u
        assert torch.allclose(y.detach(), taps[:, ell], atol=1e-5)


def _positions(model, x):
    h = model.in_proj(x)
    for b in model.blocks:
        h = b(h)
    return h


def test_position_causality():
    x = torch.randn(2, 32, 13)
    z = x.clone()
    z[:, 20:] += 5.0
    causal = Reg.build("ms3m", 32, 13, se_mode="causal").eval()
    assert torch.allclose(_positions(causal, x)[:, :20], _positions(causal, z)[:, :20])
    window = Reg.build("ms3m", 32, 13).eval()
    assert not torch.allclose(_positions(window, x)[:, :20], _positions(window, z)[:, :20])


def test_frozen_kernels_do_not_change_output():
    m = Reg.build("ms3m", 32, 13).eval()
    x = torch.randn(2, 32, 13)
    a = m(x)
    m.freeze_kernels()
    assert torch.allclose(a, m(x))


def test_unseen_runs_never_reach_training():
    L = 4
    runs = [np.arange(40.0), 100 + np.arange(30.0), 200 + np.arange(50.0)]   # three recordings
    x = np.concatenate([np.stack([r[i:i + L] for i in range(len(r) - L)]) for r in runs])[:, :, None].repeat(13, 2).astype(np.float32)
    y = x[:, -1, :] + 1
    bounds = D.run_bounds(x)
    assert [e - s for s, e in bounds] == [36, 26, 46]
    g = D.unseen_groups(bounds, dict(test=(1,), val=(2,)))
    sp = D.make_group_split(x, y, g)
    train_raw = sp.x_scaler.inverse(sp.train.tensors[0].numpy())
    assert train_raw.max() < 100 and sp.x_last_raw.min() >= 100 and sp.x_last_raw.max() < 200


def test_block_convolution_is_the_state_space_response():
    """Every position of the depthwise convolution equals y[n] = C s[n] + D u[n] with s[n] = A_bar s[n-1] + B_bar u[n]."""
    from ms3m.models.ms3m import S6Block
    torch.manual_seed(0)
    blk = S6Block(3, 8, 12, n_components=2).double().eval()
    with torch.no_grad():
        for c in blk.kernel.components:
            c.D.normal_()
        k = blk.kernel()
        assert torch.allclose(k, sum(c() for c in blk.kernel.components))   # batched mixture == sum of components
        u = torch.randn(1, 12, 3, dtype=torch.float64)
        conv = blk.ssm_conv(u)[0].T                     # the convolution S6Block.forward applies
        rec = torch.zeros(3, 12, dtype=torch.float64)
        for c in blk.kernel.components:
            Ad, Bd = c.discretize()
            s = torch.zeros(3, 8, dtype=torch.float64)
            for n in range(12):
                s = s @ Ad.T + Bd * u[0, n, :, None]
                rec[:, n] += (c.C * s).sum(-1) + c.D * u[0, n]
        assert torch.allclose(conv, rec, atol=1e-10)
