#!/usr/bin/env python3
"""Run one experiment; the result file is written last, so an existing file means "done"."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402

from ms3m import data as D, engine as E, experiments as X, registry as Reg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True, choices=["benchmark", "ablation", "rolling", "target", "tune", "unseen"])
ap.add_argument("--name", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--out", default="results")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--smoke", action="store_true", help="one epoch on a small subset, to test the pipeline quickly")
a = ap.parse_args()
torch.set_num_threads(a.threads)
log = lambda s: print(s, flush=True)  # noqa: E731
out = os.path.join(a.out, a.kind)

model, override, train_cfg, extra = "ms3m", {}, None, {"kind": a.kind, "name": a.name}
split_kw = dict(only_target_input=False)
order = "chronological"

if a.kind == "benchmark":
    model = a.name
    split_kw["only_target_input"] = model in Reg.ONLY_TARGET_INPUT
elif a.kind == "ablation":
    arm = dict(X.ABLATION[a.name])
    if "_window" in arm:
        split_kw["window"] = arm.pop("_window")
    if "_order" in arm:
        order = arm.pop("_order")
        split_kw["recent"] = 0
    override, train_cfg = arm, X.SCREEN_TRAIN
elif a.kind == "target":
    split_kw["target"] = D.FEATURES.index(a.name)
    train_cfg = X.SCREEN_TRAIN
elif a.kind == "tune":
    model, idx = a.name.split(":")
    g = dict(X.TUNE_GRID[int(idx)])
    train_cfg = {"lr": g.pop("lr")}
    override = g
    extra["grid"] = X.TUNE_GRID[int(idx)]
    split_kw["only_target_input"] = model in Reg.ONLY_TARGET_INPUT
elif a.kind == "unseen":
    model, fold = a.name.split(":")
    train_cfg = X.SCREEN_TRAIN
    split_kw["only_target_input"] = model in Reg.ONLY_TARGET_INPUT
    extra.update(fold=fold, model_runs=X.UNSEEN[fold])

x, y = D.load_arrays(a.data_dir, order)
if a.kind == "rolling":
    fr = X.ROLLING[a.name]
    N = len(x)
    n_tr, n_va, n_te = int(fr[0] * N), int((fr[1] - fr[0]) * N), int((fr[2] - fr[1]) * N)
    split_kw["train_slice"] = (0, n_tr, n_va, n_te)
    train_cfg = X.SCREEN_TRAIN

if a.kind == "unseen":
    groups = D.unseen_groups(D.run_bounds(x), X.UNSEEN[fold])
    extra["groups"] = groups
    split = D.make_group_split(x, y, groups, only_target_input=split_kw["only_target_input"])
else:
    split = D.make_split(x, y, **split_kw)
del x, y
if a.smoke:
    split.train = TensorDataset(*[t[:1024] for t in split.train.tensors])
    split.val = TensorDataset(*[t[:512] for t in split.val.tensors])
    train_cfg = {**(train_cfg or {}), "max_epochs": 1, "patience": 1}
persist = E.metrics(split.x_last_raw, split.y_raw)
res = E.fit(model, split, a.seed, train_cfg, a.device, log=log, **override)
tag = a.name.replace(":", "_")
rec = E.save_run(res, tag, a.seed, out, persist, extra={**extra, "model_name": model, "device": a.device}, save_ckpt=a.kind == "benchmark")
log(f"DONE {a.kind}/{a.name} seed={a.seed} epochs={rec['epochs']} rmse={rec['test']['rmse']:.4f} mae={rec['test']['mae']:.4f} "
    f"persist={persist['rmse']:.4f}/{persist['mae']:.4f} skill={rec['test']['skill_rmse']:.1f}/{rec['test']['skill_mae']:.1f} t={rec['train_seconds']:.0f}s")
