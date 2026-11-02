#!/usr/bin/env python3
"""Trace-driven sensitivity of a threshold decision to controlled forecast error.

The decision is an A2-style trigger: act when forecast next-step RSRP falls below a threshold,
with hysteresis.  The oracle applies the same rule to the measured RSRP.  Actions do not change
the channel in this replay, so this is an offline study of decision quality, not closed-loop control.

Controlled error added to the forecast: bias b (dB) and AR(1) noise with standard deviation s (dB).
Decision logic compared (all use only errors already observable when the decision is taken):
  raw       the forecast as it is
  fallback  use persistence while the running mean error of the forecast exceeds a limit
  debias    subtract the running mean error from the forecast
  margin    raise the threshold by a guard band set from validation errors
  persist   persistence only (no forecaster)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402

from ms3m import data as D  # noqa: E402

COST_MISS, COST_FALSE = 5.0, 1.0   # a missed low-RSRP step costs five times a needless action
HYST = 0.5                         # dB
EWMA = 0.02                        # time constant of 50 steps = 1 s at the 20 ms step
LIMIT = 0.5                        # dB, fallback entry threshold on |running mean error|
BIASES = (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)
SIGMAS = (0, 0.5, 1, 2)
RHO = 0.9


def trigger(f, theta):
    """State machine with hysteresis: enter below theta, leave above theta + HYST."""
    s, on = np.zeros(len(f), bool), False
    for i, v in enumerate(f):
        on = v < theta + HYST if on else v < theta
        s[i] = on
    return s


def score(dec, truth):
    tp = int(np.sum(dec & truth))
    fp = int(np.sum(dec & ~truth))
    fn = int(np.sum(~dec & truth))
    return dict(miss_rate=fn / max(1, tp + fn), false_rate=fp / max(1, tp + fp), cost=(COST_MISS * fn + COST_FALSE * fp) / len(truth),
                f1=2 * tp / max(1, 2 * tp + fp + fn), switches_per_min=float(np.sum(dec[1:] != dec[:-1])) / (len(dec) * 0.02 / 60))


def running_mean(e_observed):
    m, out = 0.0, np.zeros(len(e_observed))
    for i, v in enumerate(e_observed):
        out[i] = m              # uses errors up to the previous step only
        m = (1 - EWMA) * m + EWMA * v
    return out


def ar1(n, sigma, rng):
    if sigma == 0:
        return np.zeros(n)
    w, out, s = rng.normal(0, sigma * np.sqrt(1 - RHO ** 2), n), np.zeros(n), rng.normal(0, sigma)
    for i in range(n):
        s = RHO * s + w[i]
        out[i] = s
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--bench-dir", default="results/benchmark")
ap.add_argument("--model", default="ms3m")
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 73])
ap.add_argument("--percentile", type=float, default=10.0)
ap.add_argument("--out", default="results/decision_sensitivity.json")
a = ap.parse_args()
x, y = D.load_arrays(a.data_dir)
N = len(y)
n_tr, n_va = int(N * .7), int(N * .15)
c = n_tr + n_va
truth_y = y[c:, D.RSRP].astype(np.float64)
pers = x[c:, -1, D.RSRP].astype(np.float64)
theta = float(np.percentile(y[:n_tr, D.RSRP], a.percentile))
truth = trigger(truth_y, theta)
out = dict(theta_dBm=theta, percentile=a.percentile, event_rate=float(truth.mean()), n=len(truth_y), rho=RHO,
           cost_miss=COST_MISS, cost_false=COST_FALSE, ewma=EWMA, limit_dB=LIMIT, hysteresis_dB=HYST)
out["persist_clean"] = score(trigger(pers, theta), truth)
rows = []
for seed in a.seeds:
    pred = np.load(os.path.join(a.bench_dir, f"{a.model}_seed{seed}_pred.npy")).ravel().astype(np.float64)
    # Guard band: the 95th percentile of the persistence error on the validation split.
    margin = float(np.percentile(np.abs(y[n_tr:c, D.RSRP] - x[n_tr:c, -1, D.RSRP]), 95))
    out.setdefault("margin_dB", margin)
    for sigma in SIGMAS:
        rng = np.random.default_rng(seed + int(10 * sigma))
        noise = ar1(len(pred), sigma, rng)
        for b in BIASES:
            f = pred + b + noise
            err = f - truth_y                # observed one step after each decision
            rm = running_mean(err)
            fb = np.where(np.abs(rm) > LIMIT, pers, f)
            row = dict(seed=seed, sigma=sigma, bias=b, fallback_share=float(np.mean(np.abs(rm) > LIMIT)))
            for lab, ff, th in (("raw", f, theta), ("fallback", fb, theta), ("debias", f - rm, theta), ("margin", f, theta + margin)):
                row[lab] = score(trigger(ff, th), truth)
            rows.append(row)
out["rows"] = rows
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print("theta", theta, "event rate", truth.mean(), "persist", out["persist_clean"])
for r in rows:
    if r["seed"] == a.seeds[0] and r["sigma"] in (0, 1):
        print(f"sigma={r['sigma']} bias={r['bias']:+.1f} " + " ".join(f"{k}:{r[k]['cost']:.3f}/{r[k]['miss_rate']:.2f}" for k in ("raw", "fallback", "debias", "margin")) + f" fb={r['fallback_share']:.2f}")
