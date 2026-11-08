"""CPU float64 tune scores. Uses the pinned canonical FP64 module. No confirm/test."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from study_common import (  # noqa: E402
    DROP,
    H,
    K,
    LAYERS,
    REPO,
    STUDY,
    W,
    check_state,
    continuation_bind_errors,
    incumbent_bind_errors,
    load_curve_blob,
    load_fp64_model,
    manifest,
    parent_bind_errors,
    read_prediction_csv,
    score_tune_detail,
    sha256_file,
)


def predict_blob(blob: dict) -> dict[str, list[float]]:
    torch.set_num_threads(2)
    check_state(blob["state"])
    model_mod = load_fp64_model()
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from house_edge.data import load_split

    model_mod.USE_SIGNAL = False
    net = model_mod.CNN(H, LAYERS, K, DROP, W, use_signal=False)
    net.load_state_dict(blob["state"])
    net.double()
    net.eval()
    if net.use_signal or net.sig is not None:
        raise RuntimeError("signal mode refused")
    floating = [p for p in net.parameters() if p.is_floating_point()]
    if not floating or any(p.dtype != torch.float64 for p in floating):
        raise RuntimeError("parameters are not float64")
    validation = load_split(REPO / "kit" / "public" / "validation")
    query, episodes = model_mod.build_query(validation, W)
    probs = model_mod.predict(net, query, "cpu")
    if probs.dtype != torch.float64 and str(probs.dtype) != "float64":
        raise RuntimeError(f"predict dtype {probs.dtype}")
    return model_mod.to_preds(probs, episodes, validation)


def bind_errors(path: Path, blob: dict, arm: str, digest: str) -> list[str]:
    if "parent_checkpoint_sha256" in blob:
        return continuation_bind_errors(blob, arm, digest)
    if arm == "incumbent_final_cnn03_s0":
        return incumbent_bind_errors(path, blob, digest)
    if arm not in manifest()["checkpoints"]:
        return [f"unknown arm {arm}"]
    return parent_bind_errors(path, blob, arm, digest)


def score_path(path: Path, arm: str) -> dict:
    path = Path(path)
    digest = sha256_file(path)
    blob = load_curve_blob(path)
    errors = bind_errors(path, blob, arm, digest)
    try:
        check_state(blob["state"])
    except RuntimeError as exc:
        errors.append(str(exc))
    if errors:
        raise RuntimeError("score refused: " + "; ".join(errors))
    preds = predict_blob(blob)
    detail = score_tune_detail(preds)
    if detail["n_tune_queries"] != 384:
        raise RuntimeError(f"tune query count {detail['n_tune_queries']}")
    return {
        "tune_R": detail["tune_R"],
        "n_queries": detail["n_tune_queries"],
        "n_tune_queries": detail["n_tune_queries"],
        "n_tune_episodes": detail["n_tune_episodes"],
        "n_forward_queries": detail["n_forward_queries"],
        "device": "cpu",
        "dtype": "float64",
        "arm": arm,
        "checkpoint_sha256": digest,
    }


def score_reference_csv() -> dict:
    pin = manifest()["known_reference_csv"]
    path = REPO / pin["path"]
    measured = sha256_file(path)
    if measured != pin["sha256"]:
        raise RuntimeError(f"reference csv hash mismatch {measured}")
    preds = read_prediction_csv(path)
    detail = score_tune_detail(preds)
    again = score_tune_detail(preds)
    if detail["tune_R"] != again["tune_R"]:
        raise RuntimeError("tune scorer is not repeatable on the reference csv")
    if detail["n_tune_queries"] != 384 or detail["n_forward_queries"] != 768:
        raise RuntimeError("reference csv query counts are not 384 tune and 768 forward")
    return {
        "tune_R": detail["tune_R"],
        "n_queries": detail["n_tune_queries"],
        "n_tune_queries": detail["n_tune_queries"],
        "n_forward_queries": detail["n_forward_queries"],
        "sha256": measured,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if "corrected_results" not in out.parts:
        raise SystemExit("refusing to overwrite the original score audit")
    pin = manifest()
    rows = {}
    for name, spec in pin["checkpoints"].items():
        rows[name] = score_path(REPO / spec["path"], name)
        print(f"FP64 {name} tune_R {rows[name]['tune_R']:.8f} n_tune={rows[name]['n_queries']}", flush=True)
    rows["incumbent_final_cnn03_s0"] = score_path(REPO / pin["incumbent_checkpoint"], "incumbent_final_cnn03_s0")
    print(f"FP64 incumbent tune_R {rows['incumbent_final_cnn03_s0']['tune_R']:.8f}", flush=True)
    rows["reference_csv"] = score_reference_csv()
    print(f"FP64 reference csv tune_R {rows['reference_csv']['tune_R']:.8f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
