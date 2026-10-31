"""Train RNHF or RCNF on a model space.

Outputs go to results/<config name>/, so that two configs for the same space
(configs/r2.yaml and configs/r2-2.yaml) keep separate checkpoints. A `run_name`
entry in the config overrides that directory.

Usage:
    python train.py --config configs/s2.yaml --model rnhf
    python train.py --config configs/s2.yaml --model rcnf
"""

import argparse
import os
import torch
import yaml

# Hyperbolic ambient coordinates grow like cosh(r): float64 avoids the
# catastrophic cancellation of Minkowski quantities at large radii.
torch.set_default_dtype(torch.float64)

from geometry import make_manifold
from targets import make_base, make_target
from models import build_model
from runs import run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, choices=["rnhf", "rcnf"])
    ap.add_argument("--iters", type=int, default=None, help="override config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])

    manifold = make_manifold(cfg["space"])
    base = make_base(cfg, manifold)
    target = make_target(cfg, manifold)
    data = target.sample(cfg["train"]["n_samples"])   # fixed training set
    n_val = cfg["train"].get("n_val", 2000)
    val_data = target.sample(n_val)                   # held-out validation set

    model = build_model(args.model, cfg, manifold, base)

    iters = args.iters or cfg["train"]["iters"]
    batch = cfg["train"]["batch"]

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    model.train()
    perm, ptr = torch.randperm(len(data)), 0
    for it in range(1, iters + 1):
        if ptr + batch > len(data):                   # new epoch: reshuffle
            perm, ptr = torch.randperm(len(data)), 0
        q = data[perm[ptr:ptr + batch]]
        ptr += batch
        loss = model.loss(q)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == 1:
            model.eval()
            with torch.no_grad():
                val_loss = model.loss(val_data).item()
            model.train()
            extra = f"  sigma_p {model.sigma_p.item():.3f}" if args.model == "rnhf" else ""
            gap = val_loss - loss.item()
            print(f"[{args.model} | {cfg['space']}] iter {it:5d}/{iters}  "
                  f"train {loss.item():.4f}  val {val_loss:.4f}  gap {gap:+.4f}{extra}")

    out_dir = run_dir(args.config, cfg, create=True)
    path = os.path.join(out_dir, f"{args.model}.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
