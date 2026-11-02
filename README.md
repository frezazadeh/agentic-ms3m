# MS3M: Multi-Scale Structured State-Space Mixtures

Code, baseline configurations and evaluation scripts for the paper
*Rivaling Transformers: Multi-Scale Structured State-Space Mixtures for Agentic 6G O-RAN*.

This repository replaces the earlier simplified release.  It contains the implementation used for
the reported experiments and all seven baselines; `CHANGES.md` lists every difference from the
original experiment notebook.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The KPI windows are published at <https://ieee-dataport.org/documents/video-streaming-network-kpis-o-ran-testing>.
Place `x.npy` (59,441 x 32 x 13) and `y.npy` (59,441 x 13) in a folder, referred to below as `$DATA`.

## Reproduce the paper

```bash
python scripts/run_all.py --data-dir $DATA --threads 1          # all training runs; resumable
python scripts/naive_baselines.py --data-dir $DATA              # persistence and ridge references
python scripts/latency.py --data-dir $DATA                      # run on an idle machine
python scripts/decision_sensitivity.py --data-dir $DATA
python scripts/make_figures.py --data-dir $DATA
python scripts/make_report.py --data-dir $DATA                  # writes results/summary.json
python scripts/make_tables.py                                   # LaTeX tables from summary.json
```

`results/summary.json` holds every number that appears in the paper.  CPU training is slow; on a CUDA GPU
use `run_all.py --cpu-workers 0 --gpu-workers 3 --gpu-device cuda`.  Add `--smoke` to any `run_all.py` call to
test the pipeline quickly with one epoch on a small subset.

| Paper content | Produced by | Stored in |
|---|---|---|
| Accuracy and configuration tables | `run_all.py` stage `bench42`, `bench`; `src/ms3m/registry.py` | `results/benchmark/` |
| Reference forecasters | `naive_baselines.py` | `results/naive_baselines.json` |
| Latency table | `latency.py` | `results/latency.json` |
| Ablation table | `run_all.py` stage `ablation`; `src/ms3m/experiments.py` | `results/ablation/` |
| PatchTST and iTransformer search | `run_all.py` stage `tune` | `results/tune/` |
| Temporal generalization | `run_all.py` stage `rolling` | `results/rolling/` |
| Other KPIs | `run_all.py` stage `target` | `results/target/` |
| Unseen recording runs | `run_all.py` stage `unseen`; `naive_baselines.py` | `results/unseen/`, `results/naive_baselines.json` |
| Decision sensitivity | `decision_sensitivity.py` | `results/decision_sensitivity.json` |
| Diagnostic figures | `make_figures.py` | `figures/` |
| LaTeX tables | `make_tables.py` | `results/paper_tables.tex` |

## Layout

```
src/ms3m/data.py           window ordering, chronological splits, train-only scaling
src/ms3m/engine.py         the training loop and evaluation shared by all models
src/ms3m/registry.py       exact configuration of every model
src/ms3m/experiments.py    every experiment, as data
src/ms3m/models/           MS3M and the seven baselines
tests/                     checks of the properties claimed in the paper
```

## Scope

Results come from one single-UE testbed and an offline replay.  The unseen-run evaluation tests
recordings that were never seen in training, but all recordings come from the same UE, so the results
do not establish generalization to unseen UEs or closed-loop performance in a RIC.
