#!/usr/bin/env python
"""Train one House Edge model (dense or sparse) end to end.

Usage: .venv/bin/python scripts/train_model.py --arch dense --out runs/dense
"""
import os
import sys

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6700 XT (gfx1031) via ROCm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import argparse

from house_edge.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("dense", "sparse"), required=True)
    parser.add_argument("--out", default=None, help="output dir (default runs/<arch>)")
    parser.add_argument("--kit", default=os.path.join(REPO, "kit"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--scheduler", default="none", choices=("none", "cosine"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--pilot", action="store_true", help="short smoke run")
    parser.add_argument("--max-minutes", type=float, default=30.0,
                        help="stop training new epochs beyond this budget (best checkpoint kept)")
    args = parser.parse_args()
    out = args.out or os.path.join(REPO, "runs", args.arch if not args.pilot else f"pilot_{args.arch}")
    metrics = train_model(args.arch, out, kit_dir=args.kit, epochs=args.epochs, seed=args.seed,
                          window=args.window, batch_size=args.batch_size, lr=args.lr,
                          dim_ff=args.dim_ff, scheduler=args.scheduler,
                          pilot=args.pilot, device=args.device, log_every=50)
    print({k: metrics[k] for k in ("arch", "best_epoch", "tune_R", "confirm_R", "full_val_R")})


if __name__ == "__main__":
    main()
