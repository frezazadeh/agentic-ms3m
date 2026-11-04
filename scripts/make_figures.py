#!/usr/bin/env python3
"""Regenerate the two result figures of the paper from the stored runs."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from ms3m import data as D, engine as E, registry as Reg  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 9.5, "axes.titlelocation": "left", "legend.frameon": False, "pdf.fonttype": 42})

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--results", default="results")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
x, y = D.load_arrays(a.data_dir)
split = D.make_split(x, y)
true = split.y_raw.ravel().astype(np.float64)
pers = split.x_last_raw.ravel().astype(np.float64)
pred = np.load(os.path.join(a.results, "benchmark", f"ms3m_seed{a.seed}_pred.npy")).ravel().astype(np.float64)
e = pred - true

fig, ax = plt.subplots(3, 3, figsize=(10.5, 8.2), constrained_layout=True)
ax = ax.ravel()
ax[0].plot(true[-1000:], color=BLUE, lw=1.4, label="Measured")
ax[0].plot(pred[-1000:], color=ORANGE, lw=1.0, label="MS$^3$M")
ax[0].set(title="(a) Last 1000 test samples", xlabel="Sample (20 ms step)", ylabel="RSRP (dBm)")
ax[0].legend(loc="upper right", ncol=1, fontsize=8, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
ax[1].scatter(true, pred, s=3, color=BLUE, alpha=0.25, linewidths=0)
lim = [true.min(), true.max()]
ax[1].plot(lim, lim, color=INK, lw=1)
ax[1].set(title="(b) Parity", xlabel="Measured RSRP (dBm)", ylabel="Forecast RSRP (dBm)")
ax[1].text(0.04, 0.96, f"RMSE {np.sqrt(np.mean(e**2)):.3f} dB\nMAE {np.mean(np.abs(e)):.3f} dB", transform=ax[1].transAxes, va="top", color=INK)
ax[2].hist(np.clip(e, -2, 2), bins=80, color=BLUE)
ax[2].set(title="(c) Residuals (clipped to $\\pm$2 dB)", xlabel="Forecast $-$ measured (dB)", ylabel="Count")
(osm, osr), (sl, ic, _) = stats.probplot(e, dist="norm")
ax[3].scatter(osm[::20], osr[::20], s=4, color=BLUE)
ax[3].plot(osm, sl * osm + ic, color=INK, lw=1)
ax[3].set(title="(d) Residual Q–Q", xlabel="Normal quantile", ylabel="Residual quantile (dB)")
ax[4].scatter(pred, e, s=3, color=BLUE, alpha=0.25, linewidths=0)
ax[4].axhline(0, color=INK, lw=1)
ax[4].set(title="(e) Residual vs. forecast", xlabel="Forecast RSRP (dBm)", ylabel="Residual (dB)")
for v, c, lab in ((np.abs(e), BLUE, "MS$^3$M"), (np.abs(pers - true), ORANGE, "Persistence")):
    s = np.sort(v)
    ax[5].plot(s, np.arange(1, len(s) + 1) / len(s), color=c, lw=1.8, label=lab)
ax[5].set(title="(f) Absolute-error CDF", xlabel="|Error| (dB), log scale", ylabel="Fraction of samples", xscale="log", xlim=(1e-3, 10))
ax[5].legend(loc="upper left")
ax[6].boxplot([np.abs(e), np.abs(pers - true)], vert=False, tick_labels=["MS$^3$M", "Persistence"], showfliers=False, widths=0.5,
              medianprops=dict(color=INK), boxprops=dict(color=BLUE), whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
ax[6].set(title="(g) Absolute error, outliers hidden", xlabel="|Error| (dB)")
ez = e - e.mean()
lags = np.arange(1, 51)
acf = np.array([np.corrcoef(ez[:-k], ez[k:])[0, 1] for k in lags])
ax[7].vlines(lags, 0, acf, color=BLUE, lw=1.5)
band = 1.96 / np.sqrt(len(e))
ax[7].axhline(band, color=MUTED, lw=0.8, ls="--")
ax[7].axhline(-band, color=MUTED, lw=0.8, ls="--")
ax[7].axhline(0, color=INK, lw=0.8)
ax[7].set(title="(h) Residual autocorrelation", xlabel="Lag (20 ms steps)", ylabel="ACF")

torch.set_num_threads(1)
model = Reg.build("ms3m", split.L, split.F)
model.load_state_dict(torch.load(os.path.join(a.results, "benchmark", f"ms3m_seed{a.seed}.pt"), map_location="cpu"))
model.eval()
model.freeze_kernels()
X = split.test.tensors[0]
rng = np.random.default_rng(0)
base = np.sqrt(np.mean(e ** 2))


def predict(Z):
    with torch.no_grad():
        return np.vstack([split.y_scaler.inverse(model(Z[i:i + 256]).numpy()) for i in range(0, len(Z), 256)]).ravel()


imp = {}
for j, n in enumerate(D.FEATURES):
    Z = X.clone()
    Z[:, :, j] = Z[rng.permutation(len(Z)), :, j]
    imp[n] = float(np.sqrt(np.mean((predict(Z) - true) ** 2)) - base)
order = sorted(imp, key=imp.get)
ax[8].barh(order, [imp[k] for k in order], color=BLUE, height=0.7)
ax[8].set(title="(i) Permutation importance", xlabel="RMSE increase (dB), log scale", xscale="symlog")
ax[8].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[8].grid(axis="y", visible=False)
fig.savefig(os.path.join(a.out, "Method-per.pdf"))
fig.savefig(os.path.join(a.out, "Method-per.png"), dpi=130)
json.dump(imp, open(os.path.join(a.results, "permutation_importance.json"), "w"), indent=1)
print("importance", {k: round(imp[k], 4) for k in reversed(order)})
print("residual acf", np.round(acf[:6], 3), "bias", e.mean(), "skew", stats.skew(e), "kurt", stats.kurtosis(e))
print("share |e|<0.1,0.25,0.5,1:", [round(float(np.mean(np.abs(e) < t)), 3) for t in (0.1, 0.25, 0.5, 1)])

f = os.path.join(a.results, "decision_sensitivity.json")
if os.path.exists(f):
    d = json.load(open(f))
    rows = d["rows"]
    labs = [("raw", "Forecast as is", BLUE, "o"), ("fallback", "Fallback to persistence", ORANGE, "s"), ("debias", "Running-bias correction", AQUA, "^"), ("margin", "Guard band", VIOLET, "D")]
    biases = sorted({r["bias"] for r in rows})
    sig = (0, 1)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True, constrained_layout=True)
    for axx, s in zip(axs, sig):
        for k, lab, c, mk in labs:
            v = [np.mean([r[k]["cost"] for r in rows if r["bias"] == b and r["sigma"] == s]) for b in biases]
            axx.plot(biases, v, color=c, lw=2, marker=mk, ms=5, label=lab)
        axx.axhline(d["persist_clean"]["cost"], color=MUTED, ls="--", lw=1, label="Persistence only")
        axx.set(title=f"Injected noise SD = {s} dB", xlabel="Injected forecast bias (dB)")
    axs[0].set_ylabel("Decision cost per step")
    axs[1].legend(loc="upper center", fontsize=8)
    fig.savefig(os.path.join(a.out, "decision_sensitivity.pdf"))
    fig.savefig(os.path.join(a.out, "decision_sensitivity.png"), dpi=130)
