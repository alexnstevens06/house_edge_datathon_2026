"""Seed-0 entry. Refuses every other seed before training."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frozen_heads as F  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    F.require_seed0(args.seed)
    F.train_seed0(args.out)


if __name__ == "__main__":
    main()
