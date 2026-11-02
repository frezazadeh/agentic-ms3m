#!/usr/bin/env python3
"""Per-request inference latency, measured the way a Near-RT xApp would see it.

A request is one window of 32 x 13 samples, handled by one CPU thread: standardise, forward
pass with batch size 1, inverse-standardise.  Run this on an otherwise idle machine.
Throughput over the test tail (batches of 256, data already in memory) is reported separately.
"""
import argparse
import glob
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ms3m import data as D, registry as Reg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--ckpt-dir", default="results/benchmark")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--requests", type=int, default=2000)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", default="results/latency.json")
a = ap.parse_args()
torch.set_num_threads(1)
x, y = D.load_arrays(a.data_dir)
out = dict(platform=platform.platform(), machine=platform.machine(), torch=torch.__version__, threads=1,
           requests=a.requests, warmup=a.warmup, repeats=a.repeats, models={})


def measure(fn, windows, ysc):
    lat = []
    for i in range(a.warmup + a.requests):
        w = windows[i % len(windows)]
        t = time.perf_counter()
        with torch.no_grad():
            p = fn(w.unsqueeze(0))
        ysc.inverse(p.numpy())
        if i >= a.warmup:
            lat.append(time.perf_counter() - t)
    return 1e3 * np.array(lat)


for name in list(Reg.ALL_MODELS) + ["ms3m_frozen"]:
    base = name.replace("_frozen", "")
    split = D.make_split(x, y, only_target_input=base in Reg.ONLY_TARGET_INPUT)
    F = split.test.tensors[0].shape[-1]
    model = Reg.build(base, split.L, F)
    # Timing depends on the architecture, not on the weights, so another seed stands in for a missing checkpoint.
    ckpts = [os.path.join(a.ckpt_dir, f"{base}_seed{a.seed}.pt")] + sorted(glob.glob(os.path.join(a.ckpt_dir, f"{base}_seed*.pt")))
    ckpt = next((c for c in ckpts if os.path.exists(c)), None)
    if ckpt is None:
        print(f"{base}: no checkpoint in {a.ckpt_dir}, timing the untrained model of the same architecture", flush=True)
    else:
        if ckpt != ckpts[0]:
            print(f"{base}: seed {a.seed} checkpoint is missing, using {os.path.basename(ckpt)}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    if name.endswith("_frozen"):
        model.freeze_kernels()
    fn = lambda w: Reg.point_prediction(base, model(w))  # noqa: E731
    windows = split.test.tensors[0][:512]
    runs = [measure(fn, windows, split.y_scaler) for _ in range(a.repeats)]
    lat = np.concatenate(runs)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(split.test), 256):
            fn(split.test.tensors[0][i:i + 256])
    tail = time.perf_counter() - t0
    out["models"][name] = dict(ckpt=os.path.basename(ckpt) if ckpt else "none (untrained weights)", params=Reg.n_params(model), mean_ms=float(lat.mean()), p50_ms=float(np.percentile(lat, 50)),
                               p95_ms=float(np.percentile(lat, 95)), p99_ms=float(np.percentile(lat, 99)),
                               p50_per_repeat=[float(np.percentile(r, 50)) for r in runs],
                               tail_seconds=tail, tail_windows=len(split.test), windows_per_second=len(split.test) / tail)
    print(name, json.dumps(out["models"][name]), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
