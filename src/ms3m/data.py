"""Data loading, window ordering, chronological splits and train-only scaling."""
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ("MCS", "CQI", "RI", "PMI", "Buffer", "PRBs", "RSRQ", "RSRP", "RSSI", "SINR", "SE", "BLER", "Delay")
RSRP = FEATURES.index("RSRP")


class Scaler:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.clip(np.asarray(std, dtype=np.float32), eps, None)

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse(self, a):
        return a * self.std + self.mean


@dataclass
class Split:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    y_scaler: Scaler
    x_scaler: Scaler
    y_raw: np.ndarray            # physical-unit target, all splits
    x_last_raw: np.ndarray       # physical-unit last observed target-KPI value per window
    bounds: tuple                # (n_train, n_val, n_test)
    L: int
    F: int


def load_arrays(data_dir, order="chronological"):
    """Return x (N, L, F) and y (N, F) in float32.

    The released x.npy stores every window newest-first: x[t, 0] is the most recent sample
    and x[t, -1] the oldest (x[t+1, j+1] == x[t, j]), and y[t] == x[t+1, 0] is the next step.
    `order="chronological"` flips the windows so that x[t, -1] is the most recent sample,
    which is what the causal convolution, the last-step head and persistence all assume.
    `order="as_stored"` keeps the stored order and reproduces the earlier experiments.
    """
    x = np.load(os.path.join(data_dir, "x.npy"), allow_pickle=False).astype(np.float32)
    y = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=False).astype(np.float32)
    if order == "chronological":
        x = x[:, ::-1, :].copy()
    elif order != "as_stored":
        raise ValueError(order)
    return x, y


def make_split(x, y, target=RSRP, train_ratio=0.70, val_ratio=0.15, window=None, only_target_input=False,
               train_slice=None, recent=-1):
    """Chronological split with scalers fitted on the training windows only.

    window: keep only the `window` most recent steps (requires chronological order).
    train_slice: optional (start, n_train, n_val, n_test) used by rolling-origin evaluation.
    recent: index of the most recent step in a window (-1 chronological, 0 as stored).
    """
    if train_slice is None:
        N = len(x)
        start, n_tr, n_va = 0, int(N * train_ratio), int(N * val_ratio)
        n_te = N - n_tr - n_va
    else:
        start, n_tr, n_va, n_te = train_slice
    a, b, c, d = start, start + n_tr, start + n_tr + n_va, start + n_tr + n_va + n_te
    x_recent = x[c:d, recent, target:target + 1].copy()
    if window is not None:
        x = x[:, -window:, :]
    L, F = x.shape[1:]
    x_sc = Scaler(x[a:b].reshape(-1, F).mean(0), x[a:b].reshape(-1, F).std(0))
    y_t = y[:, target:target + 1]
    y_sc = Scaler(y_t[a:b].mean(0), y_t[a:b].std(0))

    def ds(i, j):
        X = x_sc.transform(x[i:j])
        if only_target_input:
            X = X[:, :, target:target + 1]
        return TensorDataset(torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y_sc.transform(y_t[i:j])))

    return Split(ds(a, b), ds(b, c), ds(c, d), y_sc, x_sc, y_t[c:d].copy(), x_recent,
                 (n_tr, n_va, n_te), L, F)


def run_bounds(x, min_len=None):
    """(start, end) of every recording run in chronologically ordered windows.

    Within a run, window t+1 is window t shifted by one step (x[t+1, :-1] == x[t, 1:]); a run ends
    where that fails.  Fragments shorter than one window are merged into the preceding run.
    """
    n, L = x.shape[:2]
    min_len = min_len or L
    ok = np.ones(n - 1, bool)
    for s in range(0, n - 1, 5000):
        e = min(n - 1, s + 5000)
        ok[s:e] = np.all(np.isclose(x[s + 1:e + 1, :-1], x[s:e, 1:]), axis=(1, 2))
    starts = [0] + [int(i) + 1 for i in np.where(~ok)[0]]
    runs = []
    for s, e in zip(starts, starts[1:] + [n]):
        if runs and e - s < min_len:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    return runs


def make_group_split(x, y, groups, target=RSRP, only_target_input=False):
    """Split by recording run, so the test runs are never seen in training.

    groups: dict(train=[(start, end), ...], val=[...], test=[...]); scalers are fitted on the training windows only.
    """
    idx = {k: np.concatenate([np.arange(a, b) for a, b in v]) for k, v in groups.items()}
    L, F = x.shape[1:]
    x_tr = x[idx["train"]].reshape(-1, F)
    x_sc = Scaler(x_tr.mean(0), x_tr.std(0))
    del x_tr
    y_t = y[:, target:target + 1]
    y_sc = Scaler(y_t[idx["train"]].mean(0), y_t[idx["train"]].std(0))

    def ds(i):
        X = x_sc.transform(x[i])
        if only_target_input:
            X = X[:, :, target:target + 1]
        return TensorDataset(torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y_sc.transform(y_t[i])))

    te = idx["test"]
    return Split(ds(idx["train"]), ds(idx["val"]), ds(te), y_sc, x_sc, y_t[te].copy(), x[te, -1, target:target + 1].copy(),
                 tuple(len(idx[k]) for k in ("train", "val", "test")), L, F)


def unseen_groups(runs, fold):
    """Train/val/test runs of one unseen-run fold (fold = dict(test=(run indices), val=(run indices)))."""
    held = set(fold["test"]) | set(fold["val"])
    return dict(train=[r for i, r in enumerate(runs) if i not in held], val=[runs[i] for i in fold["val"]],
                test=[runs[i] for i in fold["test"]])


def loaders(split, batch_size=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(split.train, batch_size=batch_size, shuffle=True, drop_last=True, generator=g),
            DataLoader(split.val, batch_size=batch_size, shuffle=False),
            DataLoader(split.test, batch_size=batch_size, shuffle=False))
