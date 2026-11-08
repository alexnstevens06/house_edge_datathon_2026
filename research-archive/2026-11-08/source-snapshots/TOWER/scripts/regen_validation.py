#!/usr/bin/env python
"""Regenerate validation predictions from a checkpoint with CURRENT code and
compare against the saved validation_predictions.csv (provenance check).

Usage: .venv/bin/python scripts/regen_validation.py --ckpt runs/v2_dense/best.pt \
         --ref runs/v2_dense/validation_predictions.csv --out runs/verify/x_regen.csv
"""
import csv
import os
import sys

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import argparse

import torch

from house_edge.data import load_split
from house_edge.models import NextSpinTransformer
from house_edge.predict import predict_split
from house_edge.scoring import write_submission_csv


def read_probs_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]
                for r in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = state["config"]
    model = NextSpinTransformer(state["arch"], d_model=cfg["d_model"], nhead=cfg["nhead"],
                                num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"],
                                dropout=cfg["dropout"])
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    data = load_split(os.path.join(REPO, "kit", "public", args.split))
    window = cfg["window"]
    predictions = predict_split(model, data, device, window=window)
    write_submission_csv(args.out, predictions)

    reference = read_probs_csv(args.ref)
    assert set(reference) == set(predictions), "query_id sets differ"
    max_diff = max(abs(a - b) for q in reference for a, b in zip(reference[q], predictions[q]))
    print(f"REGEN {os.path.basename(args.ckpt)} split={args.split} n={len(predictions)} "
          f"device={device} max_abs_diff_vs_saved={max_diff:.3e}")
    with open(os.path.join(REPO, "runs", "verify", "provenance.txt"), "a") as handle:
        handle.write(f"{args.ckpt}\t{args.split}\t{args.out}\t{device}\tmax_abs_diff={max_diff:.3e}\n")


if __name__ == "__main__":
    main()
