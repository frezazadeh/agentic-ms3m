"""The training loop and evaluation shared by every model (same as the notebook's)."""
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch

from . import data as D
from . import registry as Reg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(name, model, loader, y_scaler, device="cpu"):
    model.eval()
    out = [Reg.point_prediction(name, model(xb.to(device))).cpu().numpy() for xb, _ in loader]
    return y_scaler.inverse(np.vstack(out))


def metrics(pred, true):
    e = (pred - true).ravel()
    t = true.ravel()
    mse = float(np.mean(e ** 2))
    return dict(mse=mse, rmse=math.sqrt(mse), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                r2=float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)))


def fit(name, split, seed, train_cfg=None, device="cpu", log=None, **model_override):
    tc = {**Reg.TRAIN, **(train_cfg or {})}
    set_seed(seed)
    F = split.train.tensors[0].shape[-1]
    model = Reg.build(name, split.L, F, **model_override).to(device)
    tr, va, te = D.loaders(split, tc["batch_size"], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=tc["plateau_factor"], patience=tc["plateau_patience"])
    loss_fn = Reg.loss_fn(name)
    best, best_state, wait, hist = float("inf"), None, tc["patience"], []
    t_start = time.time()
    for ep in range(1, tc["max_epochs"] + 1):
        model.train()
        s, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
            opt.step()
            s += loss.item() * xb.size(0)
            n += xb.size(0)
        model.eval()
        with torch.no_grad():
            v, m = 0.0, 0
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                v += loss_fn(model(xb), yb).item() * xb.size(0)
                m += xb.size(0)
        v /= m
        sched.step(v)
        hist.append((ep, s / n, v, opt.param_groups[0]["lr"]))
        if log:
            log(f"{name} seed={seed} ep={ep:03d} train={s / n:.5f} val={v:.5f} lr={opt.param_groups[0]['lr']:.1e} t={time.time() - t_start:.0f}s")
        if v < best - 1e-6:
            best, wait = v, tc["patience"]
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait -= 1
            if wait == 0:
                break
    model.load_state_dict(best_state)
    pred = predict(name, model, te, split.y_scaler, device)
    return dict(model=model, pred=pred, true=split.y_raw, history=hist, best_val=best, epochs=len(hist),
                train_seconds=time.time() - t_start, n_params=Reg.n_params(model))


def save_run(res, name, seed, out_dir, persistence, extra=None, save_ckpt=True):
    os.makedirs(out_dir, exist_ok=True)
    m = metrics(res["pred"], res["true"])
    m["skill_rmse"] = 100 * (1 - m["rmse"] / persistence["rmse"])
    m["skill_mae"] = 100 * (1 - m["mae"] / persistence["mae"])
    rec = dict(model=name, seed=seed, n_params=res["n_params"], epochs=res["epochs"], best_val=res["best_val"],
               train_seconds=res["train_seconds"], test=m, persistence=persistence, history=res["history"], **(extra or {}))
    tag = f"{name}_seed{seed}"
    np.save(os.path.join(out_dir, tag + "_pred.npy"), res["pred"].astype(np.float32))
    if save_ckpt:
        torch.save(res["model"].state_dict(), os.path.join(out_dir, tag + ".pt"))
    # The metrics file is written last and atomically: its presence means the run is complete.
    tmp = os.path.join(out_dir, tag + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, tag + ".json"))
    return rec
