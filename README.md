
# Rivaling Transformers: Multi‑Scale Structured State‑Space Mixtures for Agentic 6G O‑RAN (MS³M)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#-installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research--prototype-informational.svg)](#-project-status)

> **MS³M** is an agentic, **multi‑scale structured state‑space** architecture designed to **rival Transformers** on long sequences while improving **efficiency, interpretability, and controllability** for **6G O‑RAN** workloads (traffic prediction, scheduling, anomaly detection, RIC xApps/rApps policy learning).

---

## Table of Contents
- [Motivation](#-motivation)
- [Key Contributions](#-key-contributions)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [CLI Usage](#-cli-usage)
- [Models & Baselines](#-models--baselines)
- [O‑RAN Data Hooks](#-o-ran-data-hooks)
- [Reproducibility](#-reproducibility)
- [Benchmark Harness](#-benchmark-harness)
- [Results Placeholder](#-results-placeholder)
- [Add/Activate External Baselines](#-addactivate-external-baselines)
- [Roadmap](#-roadmap)
- [Cite](#-cite)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔭 Motivation
6G O‑RAN demands models that **generalize across temporal scales** (µs→ms→s), remain **stable on long horizons**, and can be **steered by policies and KPIs**. Transformers excel at global context but are compute‑hungry and less interpretable for control. **MS³M** blends **structured SSM cores** with **mixture‑of‑experts** routing **across scales**, under **agentic control** that can adapt routing decisions to high‑level objectives and network KPIs.

## ✨ Key Contributions
- **Multi‑Scale SSM Mixtures:** experts with distinct receptive fields; routing fuses symbol/slot/subframe timescales.
- **Agentic Routing:** lightweight controller selects experts/scales conditioned on KPIs & objectives (RIC‑friendly).
- **O‑RAN Alignment:** clear I/O for near‑RT **xApps** and non‑RT **rApps**; dataset adapters for RAN KPIs.
- **Unified Training Harness:** single CLI to compare **MS³M** against **Transformers/LSTM/TCN** and **research baselines**.

---

## 📦 Repository Structure
```
ms3m/
├─ src/ms3m/
│  ├─ models/
│  │  ├─ ms3m.py                 # MS³M reference (toy) implementation
│  │  ├─ baselines.py            # Transformer / LSTM / TCN
│  │  └─ baselines_ext.py        # Adapters for FEDformer / Informer / TFT / ETSformer / Crossformer / PatchTST / iTransformer
│  ├─ modules/
│  │  └─ state_space.py          # Simple SSM-like building block (placeholder)
│  └─ data/
│     └─ o_ran_dataset.py        # KPI sequence dataset wrapper
├─ scripts/
│  ├─ train.py                   # Training entrypoint (shared across all models)
│  ├─ eval.py                    # Smoke evaluation
│  └─ benchmark.py               # Throughput benchmark (tokens/s)
├─ examples/
│  ├─ minimal_usage.py
│  └─ baselines_usage.md
├─ notebooks/
│  └─ MS3M(3)(1).ipynb
├─ docs/
│  ├─ index.md
│  └─ baselines.md               # How to activate external baselines
├─ tests/                        # (add unit tests here)
├─ pyproject.toml
├─ requirements.txt
├─ CITATION.cff
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ SECURITY.md
├─ CHANGELOG.md
├─ LICENSE
└─ README.md
```

---

## 🧩 Installation
**Requirements:** Python 3.10+, PyTorch 2.2+

```bash
git clone https://github.com/BrainOmega/ms3m.git
cd ms3m
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

> If using CUDA, install the appropriate PyTorch build from the official website.

---

## 🚀 Quickstart
Minimal run to verify everything is wired:
```bash
python examples/minimal_usage.py
```

Train a toy run (identity target on random data):
```bash
python scripts/train.py --model ms3m --epochs 1 --seq-len 256 --d-model 128
```

---

## 🛠 CLI Usage
All models share the same interface.
```bash
python scripts/train.py   --model {ms3m,transformer,lstm,tcn,fedformer,informer,tft,etsformer,crossformer,patchtst,itransformer}   --epochs 3 --batch-size 16 --seq-len 256 --d-model 128 --seed 42
```

---

## 🧠 Models & Baselines
**Proposed**
- **MS³M** — Multi‑Scale Structured State‑Space Mixtures with agentic routing.

**Built‑in baselines**
- **Transformer** (PyTorch `TransformerEncoder`)
- **LSTM**
- **TCN** (causal dilated 1D convolutions)

**Research baselines (adapters included; plug implementations to run)**
- **FEDformer**
- **Informer**
- **TFT** (Temporal Fusion Transformer)
- **ETSformer**
- **Crossformer**
- **PatchTST**
- **iTransformer**

See [`docs/baselines.md`](docs/baselines.md) for activation instructions.

---

## 📡 O‑RAN Data Hooks
This repository targets **near‑RT RIC xApps** and **non‑RT rApps** use cases via simple dataset adapters:
- `ORanKPIDataset`: accepts preprocessed tensors shaped `(N, T, D)` with KPIs such as PRB utilization, BLER, throughput, latency, HO rate, anomaly scores.
- Extend with custom loaders or gRPC/REST bridges to RIC data sources.

---

## 🔁 Reproducibility
- Deterministic seeds (`--seed`), single‑file configs (CLI flags), pinned core deps in `requirements.txt`.
- Keep experiments scripted for repeatability; add your runs and metrics to the **Results** table below.

---

## ⚡ Benchmark Harness
Quick throughput measurement (forward pass on synthetic data):
```bash
python scripts/benchmark.py --model ms3m --batches 50 --batch-size 8 --seq-len 256 --d-model 128
# Swap --model with any supported name
```

Output:
```
Model: ms3m
Elapsed: 1.234s
Tokens processed: 262,144
Throughput: 212,500 tokens/s
```

---

## 🧪 Results Placeholder
> Replace with your real datasets, tasks, and metrics (e.g., MSE/MAE for forecasting, accuracy/F1 for classification, regret for control).

| Method        | Metric‑1 | Metric‑2 | Train Time | Params | Notes |
|---------------|---------:|---------:|-----------:|-------:|-------|
| **MS³M (ours)** |         |          |            |        |       |
| Transformer   |         |          |            |        |       |
| LSTM          |         |          |            |        |       |
| TCN           |         |          |            |        |       |
| FEDformer     |         |          |            |        |       |
| Informer      |         |          |            |        |       |
| TFT           |         |          |            |        |       |
| ETSformer     |         |          |            |        |       |
| Crossformer   |         |          |            |        |       |
| PatchTST      |         |          |            |        |       |
| iTransformer  |         |          |            |        |       |

---

## 🧷 Add/Activate External Baselines
Adapters live in `src/ms3m/models/baselines_ext.py`. Two ways to enable a model:

1. **Vendor** the official repo under `src/vendor/<model>/...` and import it in the adapter, or  
2. **Install** a pip package and import within the adapter.

**Contract:** Your model must accept `(B, T, D)` and return `(B, T, D)`. Keep parameter counts/computation comparable across methods where possible.

---

## 🗺 Roadmap
- [ ] Replace toy SSM with S4/S5‑style kernels and principled initialization.
- [ ] Implement agentic planner/critic loop for routing under KPI objectives.
- [ ] Real O‑RAN data connectors; streaming inference path for near‑RT RIC.
- [ ] Evaluation suite for multiscale forecasting and control.
- [ ] CI + unit tests + documentation site (MkDocs Material).

---

## 📝 Cite
If you use this repository, please cite the paper:
```bibtex
@misc{{ms3m},
  title        = {{Rivaling Transformers: Multi-Scale Structured State-Space Mixtures for Agentic 6G O-RAN}},
  author       = {{Farhad Rezazadeh, ...}},
  year         = {{2025}},
  url          = {{}}
}
```

---

## 🤝 Contributing
Contributions are welcome! Please see **CONTRIBUTING.md** and open a draft PR early. Follow conventional commits and include tests where applicable.

## 📜 License
MIT © BrainOmega

## 🙏 Acknowledgments
This project draws inspiration from recent progress in state‑space models (S4/S5 family), efficient sequence modeling, and O‑RAN xApp/rApp architectures.
