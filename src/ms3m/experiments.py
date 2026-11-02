"""Every experiment reported in the paper, as data."""
SEEDS = (42, 17, 73)                   # main benchmark; seed 42 is the notebook's seed
SCREEN_SEEDS = (42, 17)                # ablation screening
SCREEN_TRAIN = dict(max_epochs=30, patience=10)

# One-factor-at-a-time changes relative to the default MS3M.
ABLATION = {
    "anchor": {},
    "state16": dict(state_dim=16), "state32": dict(state_dim=32),
    "mix1": dict(n_components=1), "mix2": dict(n_components=2),
    "kernel8": dict(kernel_len=8), "kernel16": dict(kernel_len=16),
    "window8": dict(_window=8), "window16": dict(_window=16),
    "se_causal": dict(se_mode="causal"), "se_none": dict(se_mode="none"),
    "mixer_mlp": dict(use_glu=False),
    "a_diag": dict(a_mode="diag"), "dt_fixed": dict(learn_dt=False),
    # The two variants below reproduce earlier code: taps applied time-reversed (the first Colab runs), and the
    # original notebook (windows newest-first, taps unflipped).
    "taps_reversed": dict(tap_order="as_conv1d"),
    "stored_order": dict(_order="as_stored", tap_order="as_conv1d"),
}

# Expanding-window folds, as fractions of the series: (train_end, val_end, test_end).
ROLLING = {"f50": (0.40, 0.50, 0.60), "f70": (0.60, 0.70, 0.80), "f90": (0.80, 0.90, 1.00)}

# Unseen-run evaluation: the test runs (separate 120 s recordings) are excluded from training.
# Indices refer to data.run_bounds(x); the data contain 13 runs.
UNSEEN = {"u1": dict(test=(0, 1), val=(2,)), "u2": dict(test=(4, 5), val=(3,)), "u3": dict(test=(8, 9), val=(10,))}
UNSEEN_MODELS = ("ms3m", "patchtst", "itransformer", "etsformer")

# Other KPIs used to test the claim that the forecaster is not specific to RSRP.
TARGETS = ("SINR", "RSRQ")

# Validation-only search for the two baselines that performed poorly before their bugs were fixed.
TUNE_MODELS = ("patchtst", "itransformer")
TUNE_GRID = [dict(lr=lr, d_model=d) for lr in (5e-4, 1e-3, 2e-3) for d in (64, 128)]
