#!/usr/bin/env python3
"""Collect all result files into results/summary.json, the only source of the numbers in the paper."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402

from ms3m import data as D, experiments as X, registry as Reg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--results", default="results")
ap.add_argument("--block", type=int, default=500, help="moving-block length in steps (500 steps = 10 s)")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
R = a.results


def load(kind):
    out = {}
    for f in sorted(glob.glob(os.path.join(R, kind, "*.json"))):
        r = json.load(open(f))
        out.setdefault(r["model"], []).append(r)
    return out


def agg(recs, keys=("rmse", "mae", "mse", "r2", "bias", "skill_rmse", "skill_mae")):
    d = dict(n_seeds=len(recs), seeds=[r["seed"] for r in recs], params=recs[0]["n_params"], epochs=[r["epochs"] for r in recs])
    for k in keys:
        v = [r["test"][k] for r in recs]
        d[k] = float(np.mean(v))
        d[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        d[k + "_runs"] = v
    return d


def block_boot_ci(fn, n, rng):
    """Moving-block bootstrap: i.i.d. resampling would understate the error of an autocorrelated series."""
    k = int(np.ceil(n / a.block))
    vals = []
    for _ in range(a.boot):
        s = rng.integers(0, n - a.block + 1, k)
        idx = (s[:, None] + np.arange(a.block)).ravel()[:n]
        vals.append(fn(idx))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5], axis=0)]


rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
mae = lambda e: float(np.mean(np.abs(e)))  # noqa: E731
x, y = D.load_arrays(a.data_dir)
N = len(y)
b, c = int(N * .7), int(N * .7) + int(N * .15)
yt = y[c:, D.RSRP].astype(np.float64)
pers = x[c:, -1, D.RSRP].astype(np.float64)
S = dict(data=dict(N=N, L=x.shape[1], F=x.shape[2], train=b, val=c - b, test=N - c))

breaks = np.where(~np.all(np.isclose(x[1:, :-1], x[:-1, 1:]), axis=(1, 2)))[0] + 1
S["data"]["run_starts"] = [0] + [int(i) for i in breaks]
S["data"]["runs"] = D.run_bounds(x)
frac = np.abs(y[:, D.RSRP] * 3 - np.round(y[:, D.RSRP] * 3))
S["data"]["rsrp_multiple_of_third"] = float(np.mean(frac < 1e-3))
S["data"]["rsrp_unchanged_step"] = float(np.mean(y[1:, D.RSRP] == y[:-1, D.RSRP]))
A = x.reshape(-1, x.shape[2])
S["data"]["table1"] = {n: dict(min=float(A[:, i].min()), max=float(A[:, i].max()), mean=float(A[:, i].mean()), std=float(A[:, i].std()))
                       for i, n in enumerate(D.FEATURES)}
S["data"]["delay_sentinel_share"] = float(np.mean(A[:, 12] == -1))
S["persistence"] = dict(rmse=rmse(pers - yt), mae=mae(pers - yt), mse=float(np.mean((pers - yt) ** 2)))
S["benchmark"] = {k: agg(v) for k, v in load("benchmark").items()}
for k in ("naive_baselines.json",):
    if os.path.exists(os.path.join(R, k)):
        S["naive"] = json.load(open(os.path.join(R, k)))
rng = np.random.default_rng(0)
n = len(yt)
REF_SEED = 42
preds = {}
for m in S["benchmark"]:
    preds[m] = {s: np.load(os.path.join(R, "benchmark", f"{m}_seed{s}_pred.npy")).ravel().astype(np.float64) for s in S["benchmark"][m]["seeds"]}
    p = preds[m][REF_SEED]
    S["benchmark"][m]["ci_ref_seed"] = dict(rmse=block_boot_ci(lambda i: rmse(p[i] - yt[i]), n, rng), mae=block_boot_ci(lambda i: mae(p[i] - yt[i]), n, rng),
                                            seed=REF_SEED)
S["persistence"]["ci"] = dict(rmse=block_boot_ci(lambda i: rmse(pers[i] - yt[i]), n, rng), mae=block_boot_ci(lambda i: mae(pers[i] - yt[i]), n, rng))
ref = {}
for lab in ("ridge_ar32_target", "ridge_all13_w32"):
    f = os.path.join(R, f"naive_RSRP_{lab}_pred.npy")
    if os.path.exists(f):
        ref[lab] = np.load(f).astype(np.float64)
if "ms3m" in preds:
    S["paired"] = {}
    s0 = REF_SEED
    pm = preds["ms3m"][s0]
    others = {**{m: preds[m][s0] for m in preds if m != "ms3m" and s0 in preds[m]}, **ref, "persistence": pers}
    for m, q in others.items():
        S["paired"][m] = dict(seed=s0, rmse_diff=rmse(pm - yt) - rmse(q - yt), mae_diff=mae(pm - yt) - mae(q - yt),
                              rmse_diff_ci=block_boot_ci(lambda i: rmse(pm[i] - yt[i]) - rmse(q[i] - yt[i]), n, rng),
                              mae_diff_ci=block_boot_ci(lambda i: mae(pm[i] - yt[i]) - mae(q[i] - yt[i]), n, rng))
    starts = [s for s in S["data"]["run_starts"] if s > c] + [N]
    edges = [c] + starts
    S["per_run"] = []
    for i, j in zip(edges[:-1], edges[1:]):
        if j - i < 100:
            continue
        ii, jj = i - c, j - c
        S["per_run"].append(dict(first=i, n=j - i, ms3m_rmse=rmse(pm[ii:jj] - yt[ii:jj]), ms3m_mae=mae(pm[ii:jj] - yt[ii:jj]),
                                 pers_rmse=rmse(pers[ii:jj] - yt[ii:jj]), pers_mae=mae(pers[ii:jj] - yt[ii:jj]), y_std=float(yt[ii:jj].std())))
for kind in ("ablation", "rolling", "target", "tune", "unseen"):
    recs = load(kind)
    if recs:
        S[kind] = {k: {**agg(v), "persistence": v[0]["persistence"], "grid": v[0].get("grid"), "model_name": v[0].get("model_name"), "fold": v[0].get("fold"),
                       "best_val": float(np.mean([r["best_val"] for r in v]))} for k, v in recs.items()}
if os.path.exists(os.path.join(R, "latency.json")):
    S["latency"] = json.load(open(os.path.join(R, "latency.json")))
if os.path.exists(os.path.join(R, "decision_sensitivity.json")):
    S["decision"] = json.load(open(os.path.join(R, "decision_sensitivity.json")))
json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)
print("persistence", S["persistence"])
for m, v in sorted(S["benchmark"].items(), key=lambda z: z[1]["rmse"]):
    print(f"{m:13s} n={v['n_seeds']} params={v['params']:>9,d} RMSE {v['rmse']:.4f} MAE {v['mae']:.4f} skill {v['skill_rmse']:.1f}/{v['skill_mae']:.1f} epochs {v['epochs']}")
