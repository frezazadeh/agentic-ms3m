#!/usr/bin/env python3
"""Run every experiment in priority order; re-running skips whatever is already finished.

Default workers suit an 8 GB laptop.  On a GPU machine use e.g. --cpu-workers 0 --gpu-workers 4 --gpu-device cuda.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ms3m import experiments as X  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--cpu-workers", type=int, default=2)
ap.add_argument("--gpu-workers", type=int, default=1)
ap.add_argument("--gpu-device", default="mps")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--gpu-models", nargs="*", default=["crossformer", "fedformer"], help="trained on the GPU when one is available")
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--smoke", action="store_true", help="one epoch on a small subset of every job")
ap.add_argument("--stages", nargs="*", default=["bench42", "tune", "bench", "rolling", "target", "ablation", "unseen"])
ap.add_argument("--retries", type=int, default=3, help="attempts per job before it counts as failed")
ap.add_argument("--retry-wait", type=float, default=60, help="seconds between attempts")
ap.add_argument("--max-failed", type=int, default=3, help="stop starting jobs after this many fail in a row")
a = ap.parse_args()

BENCH = ["ms3m", "crossformer", "informer", "etsformer", "tft", "fedformer", "patchtst", "itransformer"]
jobs = []
for st in a.stages:
    if st == "bench42":
        jobs += [("benchmark", m, 42) for m in BENCH]
    elif st == "tune":
        jobs += [("tune", f"{m}:{i}", 42) for m in X.TUNE_MODELS for i in range(len(X.TUNE_GRID))]
    elif st == "bench":
        jobs += [("benchmark", m, s) for s in X.SEEDS[1:] for m in BENCH]
    elif st == "ablation":
        jobs += [("ablation", arm, s) for s in X.SCREEN_SEEDS for arm in X.ABLATION]
    elif st == "rolling":
        jobs += [("rolling", f, 42) for f in X.ROLLING]
    elif st == "target":
        jobs += [("target", t, 42) for t in X.TARGETS]
    elif st == "unseen":
        jobs += [("unseen", f"{m}:{f}", 42) for f in X.UNSEEN for m in X.UNSEEN_MODELS]

lock = threading.Lock()
failed = []
streak = 0


def done(j):
    tag = os.path.join(a.out, j[0], f"{j[1].replace(':', '_')}_seed{j[2]}")
    return os.path.exists(tag + ".json") and os.path.exists(tag + "_pred.npy")


def next_job(gpu):
    """Assigned GPU models go to the GPU worker; it also helps with screening jobs once it has nothing else."""
    with lock:
        for j in jobs:
            if not a.gpu_workers or not a.cpu_workers or (j[1] in a.gpu_models) == gpu:
                jobs.remove(j)
                return j
        if gpu:
            for j in jobs:
                if j[0] in ("ablation", "rolling", "target", "unseen"):
                    jobs.remove(j)
                    return j
    return None


def finish(j, rc, log):
    """Record the outcome.  A run of failed jobs means the machine is at fault, so the rest of the queue is kept for a rerun."""
    global streak
    with lock:
        streak = 0 if rc == 0 else streak + 1
        if rc != 0:
            failed.append(j)
            with open(log) as f:
                print(f"--- end of {os.path.basename(log)} ---\n" + "".join(f.readlines()[-15:]), flush=True)
        if streak >= a.max_failed and jobs:
            print(f"{streak} jobs failed in a row: not starting the other {len(jobs)}", flush=True)
            jobs.clear()


def worker(gpu, wid):
    while True:
        j = next_job(gpu)
        if j is None:
            return
        if done(j):
            continue
        log = os.path.join(a.out, "logs", f"{j[0]}_{j[1].replace(':', '_')}_seed{j[2]}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_job.py"), "--kind", j[0], "--name", j[1], "--seed", str(j[2]),
               "--data-dir", a.data_dir, "--out", a.out, "--threads", str(a.threads), "--device", a.gpu_device if gpu else "cpu"]
        if a.smoke:
            cmd.append("--smoke")
        for attempt in range(1, a.retries + 1):
            t = time.time()
            with open(log, "a") as f:
                f.write(f"=== attempt {attempt} at {time.strftime('%H:%M:%S')}\n")
                f.flush()
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            print(f"[{time.strftime('%H:%M')}] {'gpu' if gpu else 'cpu'}{wid} {j} rc={rc} {(time.time() - t) / 60:.1f} min", flush=True)
            if rc == 0:
                break
            if attempt < a.retries:
                time.sleep(a.retry_wait)
        finish(j, rc, log)


ths = [threading.Thread(target=worker, args=(False, i)) for i in range(a.cpu_workers)]
ths += [threading.Thread(target=worker, args=(True, i)) for i in range(a.gpu_workers)]
[t.start() for t in ths]
[t.join() for t in ths]
if failed:
    print("FAILED:", failed, flush=True)
    sys.exit(1)
print("ALL DONE", flush=True)
