"""Compare corrected tune R values with the preserved audit. Exit 1 on any change."""
from __future__ import annotations

import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    old_parents = load(STUDY / "runs" / "fp64_original_tune.json")
    new_parents = load(STUDY / "corrected_results" / "fp64_original_tune.json")
    old_cont = load(STUDY / "runs" / "fp64_continuation_tune.json")
    new_cont = load(STUDY / "corrected_results" / "fp64_continuation_tune.json")
    old_blend = load(STUDY / "runs" / "blends_tune.json")
    new_blend = load(STUDY / "corrected_results" / "blends_tune.json")
    deviations = []
    for name, row in old_parents.items():
        if name == "reference_csv":
            new_r = new_parents[name]["tune_R"]
        else:
            new_r = new_parents[name]["tune_R"]
            if new_parents[name]["n_queries"] != 384:
                deviations.append(f"{name} n_queries {new_parents[name]['n_queries']}")
            if new_parents[name]["n_forward_queries"] != 768:
                deviations.append(f"{name} forward {new_parents[name]['n_forward_queries']}")
        if new_r != row["tune_R"]:
            deviations.append(f"{name} {row['tune_R']} -> {new_r}")
    if new_parents["reference_csv"]["n_queries"] != 384:
        deviations.append("reference n_queries")
    for name, row in old_cont.items():
        if new_cont[name]["tune_R"] != row["tune_R"]:
            deviations.append(f"{name} {row['tune_R']} -> {new_cont[name]['tune_R']}")
        if new_cont[name]["n_queries"] != 384:
            deviations.append(f"{name} n_queries {new_cont[name]['n_queries']}")
    if new_blend["cnn03_csv_tune_R"] != old_blend["cnn03_csv_tune_R"]:
        deviations.append("blend baseline")
    if new_blend["materialized_tune_R"] != old_blend["materialized_tune_R"]:
        deviations.append(f"materialized {old_blend['materialized_tune_R']} -> {new_blend['materialized_tune_R']}")
    old_rows = {(row["source"], row["w"]): row["tune_R"] for row in old_blend["rows"]}
    new_rows = {(row["source"], row["w"]): row["tune_R"] for row in new_blend["rows"]}
    if old_rows != new_rows:
        deviations.append(f"blend grid {old_rows} -> {new_rows}")
    if deviations:
        print("DEVIATIONS")
        for item in deviations:
            print(item)
        raise SystemExit(1)
    print("NO_NUMERIC_DEVIATIONS", flush=True)


if __name__ == "__main__":
    main()
