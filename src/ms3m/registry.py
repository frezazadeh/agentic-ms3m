"""Exact configuration of every model in the benchmark (the values of Table 3)."""
from .models.crossformer import CrossformerPaper
from .models.etsformer import ETSformerPaper
from .models.fedformer import FEDFormerPaper
from .models.informer import InformerPaper
from .models.itransformer import ITransformerFixed
from .models.ms3m import MS3M
from .models.patchtst import PatchTST
from .models.tft import TFTPaper, pinball_loss

CONFIGS = {
    "ms3m": dict(d_model=128, n_layers=4, state_dim=64, n_components=4, dropout=0.1, dt_init=0.1),
    "patchtst": dict(d_model=128, n_heads=16, n_layers=3, d_ff=256, patch_len=16, stride=8, dropout=0.2),
    "itransformer": dict(d_model=128, n_heads=8, n_layers=4, d_ff_mult=4, dropout=0.1, use_token_embed=False),
    "fedformer": dict(d_model=128, n_layers_enc=3, n_layers_dec=2, d_ff_mult=4, modes=32, ma_kernels=(3, 5, 7, 11, 25), dropout=0.1),
    "informer": dict(d_model=128, n_heads=8, n_layers_enc=3, n_layers_dec=2, d_ff_mult=4, prob_c=5.0, pe_type="learned",
                     distill=True, pred_len=1, label_frac=0.5, use_causal_mask=True, dropout=0.1),
    "tft": dict(d_model=128, d_hidden=256, n_heads=8, dropout=0.1, dropout_att=0.1, n_static=0, quantiles=(0.1, 0.5, 0.9)),
    "etsformer": dict(d_model=128, n_layers=3, d_ff_mult=4, dropout=0.1, dropout_att=0.1, topk_freq=8, max_lag=16,
                      use_es_baseline=True, alpha_init=0.4, beta_init=0.2),
    "crossformer": dict(d_model=128, n_heads=8, d_ff_mult=4, patch_len=16, stride=8, n_routers=8, n_layers_scale1=2,
                        n_layers_scale2=1, merge_factor=2, dropout=0.1, dropout_att=0.1),
}

TRAIN = dict(batch_size=256, max_epochs=60, lr=2e-3, weight_decay=1e-4, patience=20, clip=1.0, plateau_factor=0.5, plateau_patience=2)
ONLY_TARGET_INPUT = {"patchtst"}   # channel-isolated (RSRP history only)
ALL_MODELS = tuple(CONFIGS)
LABEL = {"ms3m": "MS3M", "patchtst": "PatchTST", "itransformer": "iTransformer", "fedformer": "FEDformer", "informer": "Informer",
         "tft": "TFT", "etsformer": "ETSformer", "crossformer": "Crossformer"}


def build(name, L, F, target=7, **override):
    c = {**CONFIGS[name], **override}
    if name == "ms3m":
        return MS3M(in_dim=F, out_dim=1, L=L, **c)
    if name == "patchtst":
        return PatchTST(L=L, **c)
    if name == "itransformer":
        return ITransformerFixed(L=L, F=F, rsrp_index=target, **c)
    if name == "fedformer":
        return FEDFormerPaper(L=L, F=F, rsrp_index=target, **c)
    if name == "informer":
        return InformerPaper(L=L, F=F, rsrp_index=target, **c)
    if name == "tft":
        return TFTPaper(L=L, F=F, rsrp_index=target, **c)
    if name == "etsformer":
        return ETSformerPaper(L=L, F=F, rsrp_index=target, **c)
    if name == "crossformer":
        return CrossformerPaper(L=L, F=F, **c)
    raise KeyError(name)


def point_prediction(name, output):
    """TFT emits (0.1, 0.5, 0.9) quantiles; its point forecast is the median, as in the notebook."""
    return output[:, 1:2] if name == "tft" and output.size(-1) == 3 else output


def loss_fn(name):
    import torch.nn.functional as F_
    if name == "tft":
        return lambda p, y: pinball_loss(p, y, CONFIGS["tft"]["quantiles"])
    return F_.mse_loss


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
