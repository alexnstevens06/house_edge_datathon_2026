"""Score the unpacked candidate on the 384 tune rows only.

Imports the frozen study scorer. Does not modify it and does not write a
confirm, full, or test metric. The expected tune R is the already measured
parent value.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

CANDIDATE = Path(__file__).resolve().parent
REPO = CANDIDATE.parents[2]
UNPACKED = CANDIDATE / "unpacked"
ORIGINAL = REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt"
STUDY_SOURCE = REPO / "experiments/cnn03_convergence/continuation_20261108/source"
CORRECTED = (
    REPO
    / "experiments/cnn03_convergence/continuation_20261108/corrected_results/fp64_original_tune.json"
)
EXPECTED_SHA = "bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(unpacked: Path):
    sys.path.insert(0, str(unpacked))
    spec = importlib.util.spec_from_file_location("candidate_unpacked_model", unpacked / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load unpacked model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_unpacked_model"] = module
    spec.loader.exec_module(module)
    if module.USE_SIGNAL:
        raise RuntimeError("unpacked model USE_SIGNAL is true")
    return module


def predict_with(model_mod, blob, validation):
    import torch
    torch.set_num_threads(2)
    model_mod.USE_SIGNAL = False
    cfg = blob["config"]
    net = model_mod.CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
    net.load_state_dict(blob["state"])
    net.double()
    net.eval()
    if net.use_signal or net.sig is not None:
        raise RuntimeError("signal mode refused")
    query, episodes = model_mod.build_query(validation, cfg["W"])
    probs = model_mod.predict(net, query, "cpu")
    if str(probs.dtype) != "float64":
        raise RuntimeError(f"predict dtype {probs.dtype}")
    return model_mod.to_preds(probs, episodes, validation)


def main() -> None:
    import argparse
    import torch
    parser = argparse.ArgumentParser(description="Tune-only score of an unpacked candidate")
    parser.add_argument("--entry", type=Path, default=UNPACKED)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    args = parser.parse_args()
    unpacked = args.entry.resolve()
    original = args.original.resolve()
    packed_sha = sha256_file(unpacked / "checkpoint.pt")
    original_sha = sha256_file(original)
    if packed_sha != EXPECTED_SHA or original_sha != EXPECTED_SHA:
        raise SystemExit(f"checkpoint sha drifted packed={packed_sha} original={original_sha}")
    if (unpacked / "checkpoint.pt").read_bytes() != original.read_bytes():
        raise SystemExit("unpacked checkpoint bytes differ from the original parent")
    sys.path.insert(0, str(STUDY_SOURCE))
    sys.path.insert(0, str(REPO / "src"))
    from study_common import canonical_renorm, score_tune_detail
    from house_edge.data import load_split

    blob = torch.load(unpacked / "checkpoint.pt", map_location="cpu", weights_only=False)
    if set(blob) != {"config", "state"}:
        raise SystemExit("unpacked checkpoint is not weights-only")
    model_mod = load_model(unpacked)
    study_validation = load_split(REPO / "kit" / "public" / "validation")
    study_preds = predict_with(model_mod, blob, study_validation)
    study_detail = score_tune_detail(study_preds)
    from data import load_public_split
    package_validation = load_public_split(REPO / "kit" / "public", "validation")
    package_preds = predict_with(model_mod, blob, package_validation)
    package_detail = score_tune_detail(package_preds)
    stored = json.loads(CORRECTED.read_text(encoding="utf-8"))["curve_C_lr003_s2"]
    expected = stored["tune_R"]
    if study_detail["tune_R"] != expected or package_detail["tune_R"] != expected:
        raise SystemExit(
            f"tune R mismatch study={study_detail['tune_R']!r} "
            f"package={package_detail['tune_R']!r} expected={expected!r}"
        )
    if study_detail["n_tune_queries"] != 384 or package_detail["n_tune_queries"] != 384:
        raise SystemExit("tune query count is not 384")
    if study_detail["n_forward_queries"] != 768 or package_detail["n_forward_queries"] != 768:
        raise SystemExit("forward query count is not 768")
    worst = 0.0
    if set(study_preds) != set(package_preds):
        raise SystemExit("study loader and package loader query ids differ")
    for query_id in study_preds:
        left = canonical_renorm(study_preds[query_id])
        right = canonical_renorm(package_preds[query_id])
        for a, b in zip(left, right):
            worst = max(worst, abs(a - b))
    if worst != 0.0:
        raise SystemExit(f"loader probability mismatch max_abs={worst}")
    out = {
        "tune_R": study_detail["tune_R"],
        "n_tune_queries": study_detail["n_tune_queries"],
        "n_tune_episodes": study_detail["n_tune_episodes"],
        "n_forward_queries": study_detail["n_forward_queries"],
        "loader_max_abs_after_one_renorm": worst,
        "checkpoint_sha256": packed_sha,
        "original_checkpoint_sha256": original_sha,
        "scorer": "continuation_20261108/source/study_common.py score_tune_detail",
        "renorm": "canonical_renorm once, float then sum, same as house_edge.scoring.write_submission_csv",
        "confirm_full_test_metrics": "not_computed",
        "selection_use": "not a new selection; matches the already measured parent",
    }
    dest = CANDIDATE / "tune_only.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        f"TUNE_ONLY R={study_detail['tune_R']:.17g} n={study_detail['n_tune_queries']} "
        f"forward={study_detail['n_forward_queries']} loader_max_abs={worst}"
    )


if __name__ == "__main__":
    main()
