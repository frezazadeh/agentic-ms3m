#!/usr/bin/env python3
"""Write the result tables of the revised paper as LaTeX, in blue (revision colour), from results/summary.json."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402

from ms3m import experiments as X, registry as Reg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default=None)
ap.add_argument("--data-dir", help="unused; accepted for a uniform command line")
a = ap.parse_args()
S = json.load(open(os.path.join(a.results, "summary.json")))
out = []
w = out.append
LAB = {**Reg.LABEL, "ms3m": "MS$^3$M"}


def pm(v, sd, d=3):
    return f"{v:.{d}f}" if sd is None else f"{v:.{d}f}$\\pm${sd:.{d}f}"


def table(label, caption, cols, rows):
    w("\\begin{table}[t]\\centering\\color{blue}\\footnotesize")
    w(f"\\caption{{{caption}}}\\label{{{label}}}")
    w("\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}\\toprule")
    w(" & ".join(cols) + " \\\\\\midrule")
    for r in rows:
        w("\\midrule" if r is None else " & ".join(str(c) for c in r) + " \\\\")
    w("\\bottomrule\\end{tabular}\\end{table}\n")


# Main benchmark: accuracy over three seeds and per-request latency.
B, P, L = S["benchmark"], S["persistence"], S.get("latency", {}).get("models", {})
nv = S.get("naive", {}).get("RSRP", {})
rows = [["Persistence (last value)", "0", f"{P['rmse']:.3f}", f"{P['mae']:.3f}", "0.0", "0.0", "--", "--", "--"]]
for k, lab in (("ridge_ar32_target", "Ridge, RSRP lags ($W{=}32$)"), ("ridge_all13_w32", "Ridge, 13 KPIs ($W{=}32$)")):
    if k in nv:
        r = nv[k]
        rows.append([lab, f"{33 if 'ar32' in k else 32 * 13 + 1:,}", f"{r['rmse']:.3f}", f"{r['mae']:.3f}",
                     f"{100 * (1 - r['rmse'] / P['rmse']):.1f}", f"{100 * (1 - r['mae'] / P['mae']):.1f}", "--", "--", "--"])
rows.append(None)
for m, v in sorted(B.items(), key=lambda z: z[1]["rmse"]):
    lt = L.get(m, {})
    lat = [f"{lt['p50_ms']:.2f}", f"{lt['p95_ms']:.2f}", f"{lt['windows_per_second']:,.0f}"] if lt else ["--"] * 3
    rows.append([LAB[m], f"{v['params']:,}", pm(v["rmse"], v["rmse_sd"]), pm(v["mae"], v["mae_sd"]), f"{v['skill_rmse']:.1f}",
                 f"{v['skill_mae']:.1f}", *lat])
    if m == "ms3m" and "ms3m_frozen" in L:
        lt = L["ms3m_frozen"]
        rows.append(["\\quad MS$^3$M, cached taps", "", "", "", "", "", f"{lt['p50_ms']:.2f}", f"{lt['p95_ms']:.2f}", f"{lt['windows_per_second']:,.0f}"])
table("tab:bench_rev", "Next-step RSRP forecasting on the chronological test split (mean$\\pm$SD over seeds 42, 17, 73). "
      "Skill is relative to persistence (\\%). Latency: one request (one $32{\\times}13$ window, batch 1, one CPU thread), "
      "median and 95th percentile; throughput: test windows per second in batches of 256.",
      ["Model", "\\#Params", "RMSE (dB)", "MAE (dB)", "Skill$_R$", "Skill$_M$", "p50 (ms)", "p95 (ms)", "Windows/s"], rows)

# Ablation (seeds 42 and 17, 30 epochs).
if "ablation" in S:
    A = S["ablation"]
    desc = dict(anchor="MS$^3$M (default)", state16="State size $N{=}16$", state32="State size $N{=}32$", mix1="One mixture component",
                mix2="Two mixture components", kernel8="Kernel length 8", kernel16="Kernel length 16", window8="Window $W{=}8$",
                window16="Window $W{=}16$", se_causal="Causal SE gate", se_none="No SE gate", mixer_mlp="MLP instead of GLU mixer",
                a_diag="Diagonal $A$ (no LegS coupling)", dt_fixed="Fixed time scales $\\Delta t$", taps_reversed="Taps applied time-reversed (first runs)",
                stored_order="Original notebook (newest-first windows)")
    base = A["anchor"]["rmse"]
    rows = [[desc[k], pm(A[k]["rmse"], A[k]["rmse_sd"]), pm(A[k]["mae"], A[k]["mae_sd"]), f"{100 * (A[k]['rmse'] / base - 1):+.1f}"]
            for k in X.ABLATION if k in A]
    table("tab:ablation_rev", "Ablation of MS$^3$M, one factor at a time (mean$\\pm$SD over seeds 42 and 17, 30 epochs).",
          ["Variant", "RMSE (dB)", "MAE (dB)", "$\\Delta$RMSE (\\%)"], rows)

# Validation-only search for PatchTST and iTransformer.
if "tune" in S:
    T = S["tune"]
    rows = []
    for m in X.TUNE_MODELS:
        recs = sorted(((k, v) for k, v in T.items() if v["model_name"] == m), key=lambda z: z[0])
        best = min(recs, key=lambda z: z[1]["best_val"])[0]
        for k, v in recs:
            rows.append([LAB[m] if k == recs[0][0] else "", f"{v['grid']['lr']:g}", v["grid"]["d_model"], f"{v['best_val']:.5f}",
                         f"{v['rmse']:.3f}" + (" $\\ast$" if k == best else "")])
    table("tab:tune_rev", "Validation search for PatchTST and iTransformer (seed 42). $\\ast$: lowest validation loss; "
          "its test RMSE is shown for transparency only.", ["Model", "Learning rate", "$d_{\\mathrm{model}}$", "Val. loss", "Test RMSE (dB)"], rows)

# Generalization: expanding-window folds and unseen recording runs.
if "rolling" in S:
    R = S["rolling"]
    rows = [[f"Train to {int(100 * X.ROLLING[k][0])}\\%, test {int(100 * X.ROLLING[k][1])}--{int(100 * X.ROLLING[k][2])}\\%",
             f"{R[k]['persistence']['rmse']:.3f}", f"{R[k]['rmse']:.3f}", f"{R[k]['persistence']['mae']:.3f}", f"{R[k]['mae']:.3f}"]
            for k in X.ROLLING if k in R]
    table("tab:rolling_rev", "Rolling-origin evaluation of MS$^3$M (seed 42, 30 epochs).",
          ["Fold", "Persist. RMSE", "MS$^3$M RMSE", "Persist. MAE", "MS$^3$M MAE"], rows)
if "unseen" in S:
    U, NU = S["unseen"], S.get("naive", {}).get("unseen", {})
    cols = ["Held-out runs", "Persistence", "Ridge (13 KPIs)"] + [LAB[m] for m in X.UNSEEN_MODELS]
    rows = []
    for f, spec in X.UNSEEN.items():
        r = [", ".join(str(i + 1) for i in spec["test"])]
        n = NU.get(f, {})
        r += [f"{n['persistence']['rmse']:.3f}" if n else "--", f"{n['ridge_all13_w32']['rmse']:.3f}" if n else "--"]
        r += [f"{U[f'{m}_{f}']['rmse']:.3f}" if f"{m}_{f}" in U else "--" for m in X.UNSEEN_MODELS]
        rows.append(r)
    table("tab:unseen_rev", "RMSE (dB) on recording runs never seen in training (seed 42, 30 epochs). Runs are numbered in recording order.",
          cols, rows)

# Other KPIs.
if "target" in S:
    rows = []
    for t in X.TARGETS:
        v, n = S["target"].get(t), S.get("naive", {}).get(t, {})
        if v:
            rows.append([t, f"{v['persistence']['rmse']:.3f}", f"{n['ridge_ar32_target']['rmse']:.3f}" if n else "--",
                         f"{n['ridge_all13_w32']['rmse']:.3f}" if n else "--", f"{v['rmse']:.3f}"])
    table("tab:targets_rev", "Next-step forecasting of other KPIs (RMSE in dB, seed 42, 30 epochs).",
          ["Target", "Persistence", "Ridge (own lags)", "Ridge (13 KPIs)", "MS$^3$M"], rows)

# Decision sensitivity.
if "decision" in S:
    Dd = S["decision"]
    rows = []
    for s in (0, 1):
        for b in (-2, -1, 0, 1, 2):
            rr = [r for r in Dd["rows"] if r["sigma"] == s and r["bias"] == b]
            if rr:
                rows.append([f"{s}", f"{b:+d}"] + [f"{np.mean([r[k]['cost'] for r in rr]):.3f}" for k in ("raw", "fallback", "debias", "margin")]
                            + [f"{100 * np.mean([r['raw']['miss_rate'] for r in rr]):.1f}", f"{100 * np.mean([r['fallback']['miss_rate'] for r in rr]):.1f}"])
        rows.append(None)
    rows.pop()
    table("tab:decision_rev", f"Cost per step of an RSRP threshold trigger ($\\theta={Dd['theta_dBm']:.1f}$\\,dBm) when bias and AR(1) noise "
          f"are added to the MS$^3$M forecast (mean over three seeds). Persistence only: cost {Dd['persist_clean']['cost']:.3f}.",
          ["Noise SD (dB)", "Bias (dB)", "As is", "Fallback", "Bias corr.", "Guard band", "Miss \\% (as is)", "Miss \\% (fallback)"], rows)

path = a.out or os.path.join(a.results, "paper_tables.tex")
open(path, "w").write("% Generated by scripts/make_tables.py from summary.json. Requires \\usepackage{booktabs,xcolor}.\n" + "\n".join(out))
print("wrote", path)
