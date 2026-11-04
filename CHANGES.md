# What changed relative to the experiment notebook

The model code in `src/ms3m/models/` was copied by script from the archived notebook
`MS3M_baselines.ipynb`, so it can be compared line by line. The corrections below are the only
differences.  `order="as_stored"` in `ms3m.data.load_arrays` restores the stored window order.

## Data handling (affects every model)

The released `x.npy` stores each window newest-first: `x[t, 0]` is the most recent sample and
`x[t, -1]` the oldest (`x[t+1, j+1] == x[t, j]`), while `y[t] == x[t+1, 0]` is the next step.
The notebook treated `x[t, -1]` as the most recent sample.  As a result:

- the causal convolution ran backward in physical time;
- the last-step read-out sat on the oldest sample;
- "persistence" compared the target with a sample 32 steps old (test RMSE 3.77 dB).  True persistence,
  the most recent observed value, has a test RMSE of 0.434 dB and MAE of 0.175 dB, so every skill
  score the notebook printed was too high.

There is no target leakage: `y[t]` is strictly after every sample of `x[t]`.
`load_arrays` now flips the windows into chronological order by default.

## PatchTST (`models/patchtst.py`)

1. `head` and `pos_embed` were created during the first forward pass, after the optimizer, so they
   were never trained.
2. RevIN normalised the input but the output was never de-normalised.
3. `patchify_1d_tailpad` returns `(B, patch_len, T_p)`, and `LazyLinear` silently accepted it, so
   attention ran over positions inside a patch rather than over patches.

Parameter count changes from 662,403 to 449,411 because of (3).

## iTransformer (`models/itransformer.py`)

1. A per-variate LayerNorm over time removed each window's level and nothing restored it.  It is
   replaced by instance normalisation with de-normalisation of the target token, as in the
   iTransformer paper.
2. The LayerNorm was created lazily and was invisible to the optimizer.

Parameter count changes from 814,273 to 814,209.

## MS3M (`models/ms3m.py`)

Architecture and parameter count are unchanged (698,449).

1. **Tap order.** `F.conv1d` is a cross-correlation: with left padding, weight index `j` multiplies the
   input `L-1-j` steps before the current one.  The notebook passed the taps `k[0..L-1]` unflipped.  With
   its newest-first windows this happened to give the state-space response at the read-out position.
   Once the windows are put in chronological order, unflipped taps apply `k[0] = C B_bar + D` to the
   oldest sample and `k[L-1]` to the newest.  The taps are now flipped before the convolution, so every
   position computes `y[n] = sum_l k[l] u[n-l]`, which equals the recursion
   `s[n] = A_bar s[n-1] + B_bar u[n]`, `y[n] = C s[n] + D u[n]` (`tests/test_properties.py`).
   The first Colab runs (before this fix) are kept as the ablation variant `taps_reversed`.
2. The taps of the mixture components are computed together (same result as summing them one by one).

Added for the ablation study: `kernel_len`, `se_mode`, `a_mode`, `learn_dt`, `tap_order`; added for
deployment: `freeze_kernels()`.

## Unchanged

FEDformer, Informer, TFT, ETSformer and Crossformer are unmodified and reproduce the parameter counts
of the notebook exactly.

## How the baselines differ from the cited architectures

All baselines are our own re-implementations, adapted to one-step forecasting of a scalar target from
a 32-step window; they are not the official code of the cited papers.  Three differences matter for
the comparison (see the paper, Section 4, "Implementation of the baselines"):

- **Crossformer** (`models/crossformer.py`): encoder only, two layers at the segment scale and one after
  merging, then an MLP head on the last segment token of each scale (averaged over the KPIs).  The
  hierarchical decoder of the original is not implemented.  The cross-dimension stage runs before the
  cross-time stage, segments overlap (length 16, stride 8) and are merged by averaging.
- **Informer** (`models/informer.py`): the query sparsity measure is computed from the full score matrix
  `Q K^T` instead of a sample of the keys, so attention costs O(L^2 d) per layer.  ProbSparse attention is
  also used for the decoder cross-attention.
- **FEDformer** (`models/fedformer.py`): Fourier enhanced blocks only; the wavelet variant is not implemented.

## Added for the revision

- `data.run_bounds`, `data.make_group_split`, stage `unseen`: evaluation on recording runs held out from training.
- `scripts/make_tables.py`: LaTeX tables generated from `results/summary.json`.
- Every model is trained by the same loop (`engine.fit`) with the same data pipeline, instead of one copy per notebook cell.
