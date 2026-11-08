"""Score stored ensemble maps only. No training and no author writes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[3]
ENS = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108"
sys.path.insert(0, str(ENS / "source"))
import common as C


def load_npz(path: Path) -> dict:
    blob = np.load(path)
    ids = [str(item) for item in blob["query_id"].tolist()]
    values = blob["probability"]
    return {key: [float(v) for v in row] for key, row in zip(ids, values)}


names = [
    "ensemble_equal_s0_s1_s2",
    "blend_s2_control_w0.0",
    "blend_s2_control_w0.025",
    "blend_s2_control_w0.05",
    "blend_s2_control_w0.1",
    "blend_s2_control_w0.2",
    "curve_C_lr003_s1",
]
current = 0.30980333573689856
winner = 0.28468175920919014
for name in names:
    preds = load_npz(ENS / "results/predictions" / f"{name}.npz")
    errors = C.simplex_errors(preds)
    detail = C.score_tune_detail(preds)
    print(
        f"{name} R={detail['tune_R']:.17f} n_tune={detail['n_tune_queries']} "
        f"n_forward={detail['n_forward_queries']} simplex={len(errors)} "
        f"vs_current={current - detail['tune_R']:.17f} "
        f"vs_02={detail['tune_R'] - winner:.17f}",
        flush=True,
    )
print(f"improvement_02={current - winner:.17f}", flush=True)
print("SCORE_REST_OK", flush=True)
