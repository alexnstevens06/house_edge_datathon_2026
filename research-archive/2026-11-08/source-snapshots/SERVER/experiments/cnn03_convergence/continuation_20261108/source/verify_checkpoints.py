"""Architecture, key, and SHA checks for the six curve checkpoints. No scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from study_common import (  # noqa: E402
    REPO,
    STUDY,
    check_state,
    load_curve_blob,
    manifest,
    parent_is_weights_only,
    sha256_file,
)


def check_arm(name: str, spec: dict) -> dict:
    path = REPO / spec["path"]
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise RuntimeError(f"{name} sha256 {digest} != pin {spec['sha256']}")
    blob = load_curve_blob(path)
    if not parent_is_weights_only(blob):
        raise RuntimeError(f"{name} is not a weights-only checkpoint: {sorted(blob)}")
    cfg = blob["config"]
    if cfg.get("h") != 128 or cfg.get("layers") != 4 or cfg.get("W") != 64 or cfg.get("drop") != 0.3:
        raise RuntimeError(f"{name} config architecture {cfg}")
    if cfg.get("bs") != 512 or cfg.get("epochs") != 30 or abs(float(cfg["wd"]) - 0.00895) > 1e-12:
        raise RuntimeError(f"{name} training config drifted {cfg}")
    if abs(float(cfg["pct_start"]) - 0.3) > 1e-12:
        raise RuntimeError(f"{name} pct_start {cfg.get('pct_start')}")
    if int(cfg["seed"]) != int(spec["seed"]) or abs(float(cfg["lr"]) - float(spec["lr"])) > 1e-12:
        raise RuntimeError(f"{name} lr/seed {cfg.get('lr')} {cfg.get('seed')} != {spec}")
    check_state(blob["state"])
    return {
        "arm": name,
        "sha256": digest,
        "keys": sorted(blob),
        "n_tensors": len(blob["state"]),
        "optimizer_saved": False,
        "scheduler_saved": False,
        "rng_saved": False,
        "warm_start_required": True,
        "lr": cfg["lr"],
        "seed": cfg["seed"],
    }


def main() -> None:
    pin = manifest()
    rows = [check_arm(name, spec) for name, spec in pin["checkpoints"].items()]
    out_dir = STUDY / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "checkpoint_validation.json"
    out.write_text(json.dumps({"arms": rows, "verdict": "PASS"}, indent=1) + "\n", encoding="utf-8")
    print(f"PASS six checkpoints weights-only n_tensors={rows[0]['n_tensors']} wrote {out}", flush=True)


if __name__ == "__main__":
    main()
