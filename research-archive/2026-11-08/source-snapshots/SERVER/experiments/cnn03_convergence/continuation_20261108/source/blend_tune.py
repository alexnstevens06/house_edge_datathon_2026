"""Frozen TUNE blends of the CNN03 CSV with closed hard-signal predictions.

CPU inference only. Does not train specialists and does not score confirm or test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from study_common import (  # noqa: E402
    BLEND_WEIGHTS,
    REPO,
    canonical_renorm,
    manifest,
    read_prediction_csv,
    score_tune,
    sha256_file,
)

HARD = REPO / "experiments/cnn_experts/hard_signal"
sys.path.insert(0, str(HARD / "source"))
import hard_signal as HS  # noqa: E402


PUBLISHED = {
    "routed": 0.5009909376779845,
    "control": 0.46211193532631023,
}


def materialize() -> tuple[dict, dict]:
    torch.set_num_threads(2)
    from house_edge.data import load_split

    validation = load_split(os.path.join(HS.B.PUB, "validation"))
    root = HARD / "runs" / "s0"

    def state(name: str):
        path = root / name / "checkpoint.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "sig.weight" in blob["state"]:
            raise RuntimeError(f"signal tensor in {name}")
        return blob["state"]

    specialists = {
        -1: HS._load_state(state("specialist_-1"), "cpu"),
        0: HS._load_state(state("specialist_+0"), "cpu"),
        1: HS._load_state(state("specialist_+1"), "cpu"),
    }
    routed, _routes = HS.predict_routed(
        {"shared": HS._load_state(state("control"), "cpu"), "specialists": specialists},
        {-1: "specialist", 0: "specialist", 1: "specialist"},
        validation,
        HS.W,
    )
    control = HS.predict_single(HS._load_state(state("control"), "cpu"), validation, HS.W)
    return routed, control


def mix(base: dict, other: dict, weight: float) -> dict:
    if set(base) != set(other):
        raise RuntimeError("blend query ids differ")
    out = {}
    for key, row in base.items():
        mixed = [(1.0 - weight) * float(row[i]) + weight * float(other[key][i]) for i in range(5)]
        out[key] = canonical_renorm(mixed)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if "corrected_results" not in out.parts:
        raise SystemExit("refusing to overwrite the original blend audit")
    pin = manifest()["known_reference_csv"]
    csv_path = REPO / pin["path"]
    measured = sha256_file(csv_path)
    if measured != pin["sha256"]:
        raise RuntimeError(f"reference csv hash mismatch {measured}")
    cnn = read_prediction_csv(csv_path)
    zero = score_tune(cnn)
    routed, control = materialize()
    reproduced = {
        "routed": score_tune(routed),
        "control": score_tune(control),
    }
    rows = []
    for source, preds in (("routed", routed), ("control", control)):
        for weight in BLEND_WEIGHTS:
            blended = cnn if weight == 0.0 else mix(cnn, preds, weight)
            tune_r = score_tune(blended)
            if weight == 0.0 and tune_r != zero:
                raise RuntimeError(f"w=0 {source} tune_R {tune_r} != csv {zero}")
            rows.append({"source": source, "w": weight, "tune_R": tune_r})
            print(f"BLEND {source} w={weight} tune_R={tune_r:.8f}", flush=True)
    payload = {
        "cnn03_csv_tune_R": zero,
        "materialized_tune_R": reproduced,
        "published_hard_signal_tune_R": PUBLISHED,
        "caveat": "grid scored on tune; minimum is not a confirm result",
        "reference_csv_sha256": measured,
        "weights": list(BLEND_WEIGHTS),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
