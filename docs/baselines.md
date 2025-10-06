
# External Baselines

This repo declares adapters for:

- FEDformer
- Informer
- TFT (Temporal Fusion Transformer)
- ETSformer
- Crossformer
- PatchTST
- iTransformer

By default these wrappers are **stubs**. To activate each baseline, either vendor the model code under `src/vendor/<name>/...` and edit `ms3m/models/baselines_ext.py`, or install a pip package and update the imports.

**Contract:** Each model must accept `(B, T, D)` float tensors and return the same shape.

## Quick check

```bash
# Will raise a clear NotImplementedError until the model is wired:
python scripts/train.py --model fedformer
```
