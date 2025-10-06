
import argparse, time, torch
from torch.utils.data import DataLoader
from ms3m.data.o_ran_dataset import ORanKPIDataset
from train import build_model, set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    N = args.batches * args.batch_size
    data = torch.randn(N, args.seq_len, args.d_model)  # noqa: E225 (dash is valid var name in argparse, not here)

ds = ORanKPIDataset(data)
dl = DataLoader(ds, batch_size=args.batch_size)

model = build_model(args.model, args.d_model).to(args.device)
model.eval()

start = time.time()
n_tokens = 0
with torch.no_grad():
    for x, _ in dl:
        x = x.to(args.device)
        y = model(x)
        n_tokens += x.numel()
elapsed = time.time() - start
toks_per_s = n_tokens / elapsed if elapsed > 0 else float("inf")

print(f"Model: {args.model}")
print(f"Elapsed: {elapsed:.3f}s")
print(f"Tokens processed: {n_tokens:,}")
print(f"Throughput: {toks_per_s:,.0f} tokens/s")
