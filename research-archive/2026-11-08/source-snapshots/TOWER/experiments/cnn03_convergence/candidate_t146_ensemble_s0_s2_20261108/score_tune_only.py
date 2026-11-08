"""Score the unpacked ensemble on the 384 tune rows only.

Uses the frozen ensemble study's tune filter and the kit scorer. Does not
write a confirm, full, or test metric and does not call evaluate.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

CANDIDATE = Path(__file__).resolve().parent
REPO = CANDIDATE.parents[2]
PUBLIC = REPO / "kit" / "public"
NPZ = CANDIDATE / "refs" / "ensemble_equal_s0_s2.npz"
NPZ_SHA256 = "eb3ad2f66ae1898a97a8aaf9c9f3c2a6a9e173a3de43e161fffc3c8f1341356c"
STUDY_NPZ = (
    REPO
    / "experiments/cnn03_convergence/ensemble_blend_20261108/results/predictions/ensemble_equal_s0_s2.npz"
)
EXPECTED_R = 0.28468175920919014
S0_SHA = "d5d714649a93e1e552ccf5419cb01adee311c082af1063f80d8b42f6d8dfe3cb"
S2_SHA = "bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71"


def fail(message: str) -> None:
    print(f"FAIL {message}")
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predict(entry: Path):
    for name in ("data", "model", "predict"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(entry))
    spec = importlib.util.spec_from_file_location("ensemble_entry_predict", entry / "predict.py")
    if spec is None or spec.loader is None:
        fail("could not load predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def map_max_abs(preds: dict, npz_path: Path) -> float:
    import numpy as np
    blob = np.load(npz_path)
    keys = [str(item) for item in blob["query_id"].tolist()]
    values = blob["probability"]
    if set(keys) != set(preds):
        fail(f"npz ids differ from predictions missing={len(set(keys)-set(preds))}")
    worst = 0.0
    for index, key in enumerate(keys):
        for left, right in zip(preds[key], values[index]):
            worst = max(worst, abs(float(left) - float(right)))
    return worst


def read_csv_map(path: Path) -> dict[str, list[float]]:
    import csv
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["query_id"]] = [float(row[name]) for name in ("p_0", "p_1", "p_2", "p_5", "p_20")]
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Tune-only score of the seed 0+2 ensemble package")
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entry = args.entry.resolve()
    val_csv = args.val_csv.resolve()
    out = args.out.resolve()
    if "candidate_t146_ensemble_s0_s2_20261108" not in out.parts:
        fail(f"refusing to write outside the candidate folder: {out}")
    s0 = sha256_file(entry / "components" / "s0" / "checkpoint.pt")
    s2 = sha256_file(entry / "components" / "s2" / "checkpoint.pt")
    if s0 != S0_SHA or s2 != S2_SHA:
        fail(f"component checkpoint drift s0={s0} s2={s2}")
    original_s0 = REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s0/checkpoint.pt"
    original_s2 = REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt"
    if (entry / "components" / "s0" / "checkpoint.pt").read_bytes() != original_s0.read_bytes():
        fail("packed seed 0 bytes differ from the original parent")
    if (entry / "components" / "s2" / "checkpoint.pt").read_bytes() != original_s2.read_bytes():
        fail("packed seed 2 bytes differ from the original parent")
    if sha256_file(NPZ) != NPZ_SHA256 or sha256_file(STUDY_NPZ) != NPZ_SHA256:
        fail("verified ensemble npz hash drifted")
    if NPZ.read_bytes() != STUDY_NPZ.read_bytes():
        fail("vendored npz bytes differ from the study npz")
    predict = load_predict(entry)
    preds, stats = predict.predict_split(str(PUBLIC), "validation")
    if stats["queries"] != 768:
        fail(f"forward queries {stats['queries']}")
    study_dir = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108/source"
    sys.path.insert(0, str(study_dir))
    import common
    detail = common.score_tune_detail(preds)
    if set(detail) & {"confirm", "full", "test"}:
        fail("tune scorer returned a non-tune bundle")
    if detail["n_tune_queries"] != 384 or detail["n_tune_episodes"] != 48:
        fail(f"tune contract {detail}")
    if detail["n_forward_queries"] != 768:
        fail(f"forward contract {detail['n_forward_queries']}")
    if detail["tune_R"] != EXPECTED_R:
        fail(f"tune R {detail['tune_R']!r} != {EXPECTED_R!r}")
    worst = map_max_abs(preds, NPZ)
    if worst > 1e-8:
        fail(f"package map vs verified npz max_abs={worst}")
    csv_map = read_csv_map(val_csv)
    csv_detail = common.score_tune_detail(csv_map)
    if csv_detail["tune_R"] != EXPECTED_R:
        fail(f"val csv tune R {csv_detail['tune_R']!r} != {EXPECTED_R!r}")
    csv_worst = map_max_abs(csv_map, NPZ)
    if csv_worst > 1e-8:
        fail(f"val csv vs verified npz max_abs={csv_worst}")
    payload = {
        "tune_R": detail["tune_R"],
        "n_tune_queries": detail["n_tune_queries"],
        "n_tune_episodes": detail["n_tune_episodes"],
        "n_forward_queries": detail["n_forward_queries"],
        "map_max_abs_vs_verified_npz": worst,
        "csv_tune_R": csv_detail["tune_R"],
        "csv_max_abs_vs_verified_npz": csv_worst,
        "s0_infer_s": stats["s0_infer_s"],
        "s2_infer_s": stats["s2_infer_s"],
        "infer_s": stats["infer_s"],
        "max_rss_kb": stats["max_rss_kb"],
        "confirm_full_test_scored": False,
        "expected_tune_R": EXPECTED_R,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"TUNE_ONLY R={detail['tune_R']:.17g} n={detail['n_tune_queries']} "
        f"forward={detail['n_forward_queries']} map_max_abs={worst:.12g} "
        f"csv_max_abs={csv_worst:.12g} "
        f"s0_infer_s={stats['s0_infer_s']:.6f} s2_infer_s={stats['s2_infer_s']:.6f} "
        f"infer_s={stats['infer_s']:.6f} max_rss_kb={stats['max_rss_kb']}"
    )


if __name__ == "__main__":
    main()
