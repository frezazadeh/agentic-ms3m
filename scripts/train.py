
import argparse, torch
from torch import nn
from torch.utils.data import DataLoader
from ms3m.models.ms3m import MS3M
from ms3m.models.baselines import TransformerBaseline, LSTMBaseline, TCNBaseline
from ms3m.models.baselines_ext import FEDformerBaseline, InformerBaseline, TFTBaseline, ETSformerBaseline, CrossformerBaseline, PatchTSTBaseline, ITransformerBaseline
from ms3m.data.o_ran_dataset import ORanKPIDataset
from tqdm import tqdm
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(name: str, d_model: int):
    name = name.lower()
    if name == "ms3m":
        return MS3M(d_model=d_model)
    if name == "transformer":
        return TransformerBaseline(d_model=d_model, nhead=8, num_layers=4)
    if name == "lstm":
        return LSTMBaseline(d_model=d_model, n_layers=2)
    if name == "tcn":
        return TCNBaseline(d_model=d_model, num_layers=4, kernel_size=3)
    if name == "fedformer":
        return FEDformerBaseline(d_model=d_model)
    if name == "informer":
        return InformerBaseline(d_model=d_model)
    if name == "tft":
        return TFTBaseline(d_model=d_model)
    if name == "etsformer":
        return ETSformerBaseline(d_model=d_model)
    if name == "crossformer":
        return CrossformerBaseline(d_model=d_model)
    if name == "patchtst":
        return PatchTSTBaseline(d_model=d_model)
    if name == "itransformer":
        return ITransformerBaseline(d_model=d_model)
    raise ValueError(f"Unknown model: {name}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="ms3m", choices=["ms3m", "transformer", "lstm", "tcn", "fedformer", "informer", "tft", "etsformer", "crossformer", "patchtst", "itransformer"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    # Toy data: (N,T,D)
    N = 32
    data = torch.randn(N, args.seq_len, args.d_model)
    ds = ORanKPIDataset(data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.model, args.d_model).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"{args.model} | Epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(args.device)
            y = y.to(args.device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
        print("epoch done")

if __name__ == "__main__":
    main()
