#!/usr/bin/env python3
"""Closed-form reference forecasters: persistence and ridge autoregressions."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402

from ms3m import data as D, experiments as XP  # noqa: E402


def score(p, t):
    e = p - t
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), mse=float(np.mean(e ** 2)), bias=float(np.mean(e)))


def ridge(Xtr, ytr, Xva, yva, Xte, grid=(1e-3, 1e-1, 1e1, 1e3)):
    """Standardised ridge with intercept; the penalty is chosen on the validation split."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, V, T = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    G, b = A.T @ A, A.T @ (ytr - ym)
    best = None
    for lam in grid:
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), b)
        v = np.mean((V @ w + ym - yva) ** 2)
        if best is None or v < best[0]:
            best = (v, lam, w)
    return T @ best[2] + ym, best[1]


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--out", default="results/naive_baselines.json")
a = ap.parse_args()
x, y = D.load_arrays(a.data_dir)
N = len(x)
b, c = int(N * .7), int(N * .7) + int(N * .15)
out = {}
for tname in ("RSRP", "SINR", "RSRQ"):
    t = D.FEATURES.index(tname)
    yy = y[:, t].astype(np.float64)
    res = {"persistence": score(x[c:, -1, t].astype(np.float64), yy[c:]),
           "stored_oldest_sample": score(x[c:, 0, t].astype(np.float64), yy[c:])}
    for lab, X in (("ridge_ar32_target", x[:, :, t].astype(np.float64)), ("ridge_all13_w8", x[:, -8:, :].reshape(N, -1).astype(np.float64)),
                   ("ridge_all13_w32", x.reshape(N, -1).astype(np.float64))):
        p, lam = ridge(X[:b], yy[:b], X[b:c], yy[b:c], X[c:])
        res[lab] = {**score(p, yy[c:]), "lambda": lam}
        np.save(os.path.join(os.path.dirname(a.out), f"naive_{tname}_{lab}_pred.npy"), p.astype(np.float32))
    out[tname] = res
    print(tname, json.dumps(res, indent=1), flush=True)

# Unseen-run folds (RSRP): the same references, fitted without the held-out runs.
runs = D.run_bounds(x)
out["runs"] = runs
out["unseen"] = {}
t = D.RSRP
for fold, spec in XP.UNSEEN.items():
    g = D.unseen_groups(runs, spec)
    tr, va, te = (np.concatenate([np.arange(s, e) for s, e in g[k]]) for k in ("train", "val", "test"))
    yy = y[:, t].astype(np.float64)
    res = {"persistence": score(x[te, -1, t].astype(np.float64), yy[te]), "n_test": int(len(te)), "groups": g}
    for lab, X_ in (("ridge_ar32_target", x[:, :, t]), ("ridge_all13_w32", x.reshape(N, -1))):
        p, lam = ridge(X_[tr].astype(np.float64), yy[tr], X_[va].astype(np.float64), yy[va], X_[te].astype(np.float64))
        res[lab] = {**score(p, yy[te]), "lambda": lam}
    out["unseen"][fold] = res
    print("unseen", fold, {k: round(v["rmse"], 4) for k, v in res.items() if isinstance(v, dict) and "rmse" in v}, flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
