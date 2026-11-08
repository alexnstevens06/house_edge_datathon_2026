"""Pin the twelve continuation files and the incumbent after training.

This is not a pretraining manifest. MANIFEST.json is not modified.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from study_common import EXTRA_EPOCHS, POST_RUN_PATH, REPO, STUDY, manifest, sha256_file  # noqa: E402


def main() -> None:
    pin = manifest()
    continuations = {}
    for arm, spec in pin["checkpoints"].items():
        for extra in EXTRA_EPOCHS:
            name = f"{arm}_plus{extra}"
            path = STUDY / "runs" / name / "checkpoint.pt"
            continuations[name] = {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "arm": arm,
                "extra_epochs": extra,
                "parent_sha256": spec["sha256"],
            }
    incumbent = REPO / pin["incumbent_checkpoint"]
    payload = {
        "role": "post-run hash pin of checkpoints that already existed after the bounded jobs",
        "not_a_pretraining_pin": True,
        "original_manifest": "experiments/cnn03_convergence/continuation_20261108/MANIFEST.json",
        "original_manifest_not_modified": True,
        "bounded_jobs_finished_cst": "2026-11-08 00:17:20",
        "measured_at_cst": datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": "These digests were measured after training finished. They were not the pretraining pins in MANIFEST.json.",
        "continuations": continuations,
        "incumbent_checkpoint": {
            "path": pin["incumbent_checkpoint"],
            "sha256": sha256_file(incumbent),
            "note": "MANIFEST.json records this path and not a sha256. This digest is a post-run measurement.",
        },
    }
    if POST_RUN_PATH.exists():
        old = json.loads(POST_RUN_PATH.read_text(encoding="utf-8"))
        if old.get("continuations") != continuations or old.get("incumbent_checkpoint", {}).get("sha256") != payload["incumbent_checkpoint"]["sha256"]:
            raise SystemExit("refusing to change an existing post-run hash pin")
        print(f"post-run pin unchanged {POST_RUN_PATH}", flush=True)
        return
    POST_RUN_PATH.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {POST_RUN_PATH} continuations={len(continuations)}", flush=True)


if __name__ == "__main__":
    main()
