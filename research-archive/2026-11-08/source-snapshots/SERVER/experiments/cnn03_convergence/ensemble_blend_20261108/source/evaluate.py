"""Materialize the frozen ensemble and blend matrix. No training."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import common as C  # noqa: E402


COMPONENT_ROLES = (
    "curve_C_lr003_s0",
    "curve_C_lr003_s1",
    "curve_C_lr003_s2",
    "hard_shared_control",
)


def _save_map(directory: Path, name: str, preds: dict[str, list[float]]) -> dict:
    errors = C.simplex_errors(preds)
    if errors:
        raise RuntimeError(f"{name} simplex failed: {errors[:3]}")
    keys = sorted(preds)
    array = np.asarray([preds[key] for key in keys], dtype=np.float64)
    path = directory / f"{name}.npz"
    np.savez(path, query_id=np.asarray(keys), probability=array)
    return {
        "path": str(path.relative_to(C.STUDY)),
        "sha256": C.sha256_file(path),
        "n_queries": len(keys),
        "dtype": "float64",
    }


def _rank(rows: list[dict]) -> dict:
    """Lowest tune R. Within 1e-6 of current best, current best wins."""
    best = C.CURRENT_BEST_TUNE_R
    eligible = []
    for row in rows:
        delta = best - float(row["tune_R"])
        eligible.append({
            "name": row["name"],
            "tune_R": row["tune_R"],
            "improvement_vs_current_best": delta,
            "clears_1e-6": delta >= C.IMPROVEMENT_THRESHOLD,
            "tune_informed": bool(row.get("tune_informed")),
            "n_components": row["n_components"],
            "blend_weight": row.get("blend_weight"),
        })
    eligible.sort(key=lambda item: (
        0 if item["clears_1e-6"] else 1,
        item["tune_R"],
        item["n_components"],
        item["blend_weight"] if item["blend_weight"] is not None else -1,
        item["name"],
    ))
    return {
        "current_best_tune_R": best,
        "threshold": C.IMPROVEMENT_THRESHOLD,
        "packaged": False,
        "clears": [item["name"] for item in eligible if item["clears_1e-6"]],
        "order_among_rows_that_clear_else_current_best_kept": eligible,
        "caveat": (
            "Tune-only. ensemble_equal_s0_s2 is tune-informed. "
            "Blend weights were predeclared, not fit. No noise or hidden-test claim. "
            "Low-rate jobs are not in this ranking."
        ),
    }


def main() -> None:
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise SystemExit("HSA_OVERRIDE_GFX_VERSION is set")
    C.assert_pinned_files()
    # Hash-check every copied parent, including the three controls that are not scored.
    for role in C.manifest()["checkpoints"]:
        path = C.STUDY / C.manifest()["checkpoints"][role]["path"]
        if role in COMPONENT_ROLES:
            continue
        blob = C.load_blob(path)
        errors = C.bind_errors(path, blob, role, C.sha256_file(path))
        if errors:
            raise RuntimeError(f"{role} refused: {errors}")
    out = C.STUDY / "results"
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    maps = {}
    files = {}
    reproduction = {}
    for role in COMPONENT_ROLES:
        blob = C.require_bind(C.STUDY / C.manifest()["checkpoints"][role]["path"], role)
        preds = C.forward_probabilities(blob["state"])
        detail = C.score_tune_detail(preds)
        if detail["n_tune_queries"] != 384 or detail["n_forward_queries"] != 768:
            raise RuntimeError(f"{role} query counts {detail}")
        published = C.PUBLISHED_TUNE_R[role]
        delta = abs(detail["tune_R"] - published)
        reproduction[role] = {
            "tune_R": detail["tune_R"],
            "published_tune_R": published,
            "abs_delta": delta,
            "within_1e-12": C.reproduction_ok(detail["tune_R"], published),
        }
        print(
            f"REPRO {role} tune_R={detail['tune_R']:.17f} delta={delta:.3e}",
            flush=True,
        )
        if role != "hard_shared_control" and not reproduction[role]["within_1e-12"]:
            payload = {"reproduction": reproduction, "blend_written": False}
            (out / "REPRODUCTION_STOP.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
            raise SystemExit(f"reproduction gate failed for {role}; blend matrix not written")
        maps[role] = preds
    control_gap = reproduction["hard_shared_control"]
    if not control_gap["within_1e-12"]:
        # The three LR seeds match at 0. The validation tensors match house_edge.data.
        # fp32 softmax is also short of 1e-12. Probabilities are not adjusted to hit the old scalar.
        control_gap["investigation"] = (
            "Canonical float64 forward of the pinned checkpoint does not reproduce "
            "0.46211193532631023 within 1e-12. The same module reproduces seeds 0, 1, and 2 "
            "with delta 0. Validation observations match house_edge.data. An fp32 softmax "
            "path is also outside 1e-12. The blend uses the unmodified canonical float64 map."
        )
        print("CONTROL_GAP", control_gap["abs_delta"], flush=True)
        files[role] = _save_map(pred_dir, role, preds)
    base = maps["curve_C_lr003_s2"]
    control = maps["hard_shared_control"]
    rows = []
    for name, members in C.ENSEMBLES.items():
        mixed = C.equal_probability_mean([maps[item] for item in members])
        detail = C.score_tune_detail(mixed)
        files[name] = _save_map(pred_dir, name, mixed)
        rows.append({
            "name": name,
            "kind": "ensemble",
            "members": list(members),
            "n_components": len(members),
            "tune_informed": name == "ensemble_equal_s0_s2",
            "tune_R": detail["tune_R"],
            "n_tune_queries": detail["n_tune_queries"],
            "delta_vs_seed2": detail["tune_R"] - C.PUBLISHED_TUNE_R["curve_C_lr003_s2"],
        })
        print(f"ROW {name} tune_R={detail['tune_R']:.17f}", flush=True)
    for weight in C.BLEND_WEIGHTS:
        mixed = C.blend_maps(base, control, weight)
        if weight == 0.0:
            for key, row in mixed.items():
                if row != base[key]:
                    raise RuntimeError("weight 0 is not an exact copy of seed 2")
        detail = C.score_tune_detail(mixed)
        name = f"blend_s2_control_w{weight}"
        files[name] = _save_map(pred_dir, name, mixed)
        rows.append({
            "name": name,
            "kind": "blend",
            "members": ["curve_C_lr003_s2", "hard_shared_control"],
            "n_components": 2,
            "blend_weight": weight,
            "tune_informed": False,
            "tune_R": detail["tune_R"],
            "n_tune_queries": detail["n_tune_queries"],
            "delta_vs_seed2": detail["tune_R"] - C.PUBLISHED_TUNE_R["curve_C_lr003_s2"],
        })
        print(f"ROW {name} tune_R={detail['tune_R']:.17f}", flush=True)
    payload = {
        "device": "cpu",
        "dtype": "float64",
        "reproduction": reproduction,
        "rows": rows,
        "prediction_files": files,
        "ranking": _rank(rows),
        "confirm_full_test_scored": False,
    }
    (out / "matrix.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    lines = [
        "# Ensemble and blend tune R",
        "",
        "CPU float64. 384 tune queries. Confirm, full, and test were not scored.",
        "",
        "## Reproduction",
        "",
    ]
    for role, item in reproduction.items():
        lines.append(
            f"- {role}: {item['tune_R']:.17f} (published {item['published_tune_R']:.17f}, abs delta {item['abs_delta']:.3e})"
        )
    lines.extend(["", "## Predeclared rows", ""])
    for row in rows:
        lines.append(f"- {row['name']}: {row['tune_R']:.17f} (delta vs seed 2 {row['delta_vs_seed2']:+.17f})")
    ranking = payload["ranking"]
    lines.extend([
        "",
        "## Ranking against the immutable current best",
        "",
        f"Current best tune R {ranking['current_best_tune_R']:.17f}. Threshold {ranking['threshold']}.",
        f"Rows that clear 1e-6: {ranking['clears'] or 'none'}.",
        "No package was written. ensemble_equal_s0_s2 is tune-informed. The blend grid was predeclared.",
        "",
    ])
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("MATRIX_OK", flush=True)


if __name__ == "__main__":
    main()
