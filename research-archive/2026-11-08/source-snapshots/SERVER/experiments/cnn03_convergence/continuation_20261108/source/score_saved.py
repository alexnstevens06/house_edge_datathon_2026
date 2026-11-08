"""CPU float64 tune scores for the twelve saved continuations.

Reads POST_RUN_HASHES.json. Does not glob arbitrary checkpoints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_fp64 import score_path  # noqa: E402
from study_common import EXTRA_EPOCHS, STUDY, manifest, post_run, sha256_file  # noqa: E402


def continuation_jobs() -> list[tuple[str, str, Path]]:
    record = post_run()
    if record is None:
        raise RuntimeError("continuation byte pin missing")
    jobs = []
    for arm in manifest()["checkpoints"]:
        for extra in EXTRA_EPOCHS:
            name = f"{arm}_plus{extra}"
            spec = record.get("continuations", {}).get(name)
            if not spec:
                raise RuntimeError(f"post-run pin missing {name}")
            path = STUDY / "runs" / name / "checkpoint.pt"
            digest = sha256_file(path)
            if digest != spec["sha256"]:
                raise RuntimeError(f"hash mismatch for {name}")
            if spec.get("arm") != arm or spec.get("extra_epochs") != extra:
                raise RuntimeError(f"post-run label mismatch for {name}")
            jobs.append((name, arm, path))
    if len(jobs) != 12:
        raise RuntimeError(f"expected 12 continuation pins, found {len(jobs)}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if "corrected_results" not in out.parts:
        raise SystemExit("refusing to overwrite the original score audit")
    import json
    rows = {}
    for name, arm, path in continuation_jobs():
        rows[name] = score_path(path, arm)
        print(f"FP64 {name} tune_R {rows[name]['tune_R']:.8f} n_tune={rows[name]['n_queries']}", flush=True)
    if len(rows) != 12:
        raise SystemExit(f"expected 12 continuation checkpoints, found {len(rows)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
