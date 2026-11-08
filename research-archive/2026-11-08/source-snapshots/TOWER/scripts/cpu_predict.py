#!/usr/bin/env python
"""Offline CPU prediction smoke test: hidden CUDA, 2 threads, timed.

Usage: env -u CUDA_VISIBLE_DEVICES .venv/bin/python scripts/cpu_predict.py \
         --ckpt runs/v2_dense/best.pt --out runs/verify/cpu_test_v2_dense.csv \
         --gpu-csv runs/v2_dense/test_submission.csv
"""
import csv
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""       # hide CUDA from torch
os.environ["HIP_VISIBLE_DEVICES"] = ""        # hide ROCm GPUs from torch
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import argparse

import torch

torch.set_num_threads(2)

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
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-csv", required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    assert not torch.cuda.is_available(), "CUDA must be hidden for this smoke test"

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = state["config"]
    model = NextSpinTransformer(state["arch"], d_model=cfg["d_model"], nhead=cfg["nhead"],
                                num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"],
                                dropout=cfg["dropout"])
    model.load_state_dict(state["state_dict"])
    model.eval()

    data = load_split(os.path.join(REPO, "kit", "public", "test"))
    started = time.perf_counter()
    predictions = predict_split(model, data, "cpu", window=cfg["window"], batch_size=32)
    seconds = time.perf_counter() - started
    write_submission_csv(args.out, predictions)

    gpu = read_probs_csv(args.gpu_csv)
    assert set(gpu) == set(predictions), "query_id sets differ"
    max_diff = max(abs(a - b) for q in gpu for a, b in zip(gpu[q], predictions[q]))
    name = args.name or os.path.basename(os.path.dirname(args.ckpt))
    line = (f"CPU_SMOKE {name} n_queries={len(predictions)} seconds={seconds:.1f} "
            f"threads=2 max_abs_diff_vs_gpu={max_diff:.3e}")
    print(line)
    with open(os.path.join(REPO, "runs", "verify", "cpu_smoke.txt"), "a") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    main()
