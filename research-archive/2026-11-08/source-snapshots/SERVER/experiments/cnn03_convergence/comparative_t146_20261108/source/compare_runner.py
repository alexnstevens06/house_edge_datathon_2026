"""Authoritative SERVER comparison. Reads frozen studies. Writes only this folder.

Does not import ensemble evaluate.py, so the mirrored results are not rewritten.
Metric files are written only after the frozen hash check and the frozen tests.
"""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[2]
ENSEMBLE = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108"
LOWRATE = REPO / "experiments/cnn03_convergence/lowrate_20261108"
PINS_PATH = STUDY / "PINS.json"
SOURCE_SHA_PATH = STUDY / "SOURCE_SHA256.txt"
RESULT_PATH = STUDY / "results" / "comparison.json"
RSS_LIMIT_KB = 2 * 1024 * 1024
PUBLISHED_CONTROL_R = 0.46211193532631023
QUERY_ATOL = 1e-8

sys.path.insert(0, str(HERE))
import ranking as selector  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def refuse_if_heavy() -> None:
    used = rss_kb()
    if used > RSS_LIMIT_KB:
        raise SystemExit(f"host RSS {used} KB exceeds 2 GiB; stopping before more work")


def verify_frozen_hashes() -> None:
    if RESULT_PATH.exists():
        raise SystemExit("comparison.json already exists; refusing to recompute over it")
    lines = [line for line in SOURCE_SHA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("SOURCE_SHA256.txt is empty")
    for line in lines:
        digest, rel = line.split(None, 1)
        got = sha256_file(STUDY / rel)
        if got != digest:
            raise SystemExit(f"runner hash drift {rel}")
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    for rel, digest in pins["files"].items():
        path = REPO / rel
        if not path.is_file():
            raise SystemExit(f"pinned file missing {rel}")
        got = sha256_file(path)
        if got != digest:
            raise SystemExit(f"pinned file drift {rel}")
    if pins["coordinator_plan_sha256"] != "03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3":
        raise SystemExit("coordinator plan pin is not the protocol hash")


def run_local_tests() -> None:
    spec = importlib.util.spec_from_file_location("comparative_test_select", STUDY / "tests" / "test_select.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    frozen = importlib.util.spec_from_file_location(
        "comparative_frozen_ensemble_tests", STUDY / "tests" / "run_frozen_ensemble_tests.py"
    )
    frozen_module = importlib.util.module_from_spec(frozen)
    frozen.loader.exec_module(frozen_module)
    frozen_module.main()


def load_ensemble_common():
    sys.path.insert(0, str(ENSEMBLE / "source"))
    import common as ensemble_common  # noqa: WPS433
    return ensemble_common


def load_npz(path: Path) -> dict[str, list[float]]:
    import numpy as np
    payload = np.load(path)
    keys = [str(item) for item in payload["query_id"].tolist()]
    values = payload["probability"]
    return {key: [float(item) for item in values[index]] for index, key in enumerate(keys)}


def max_abs(left: dict[str, list[float]], right: dict[str, list[float]], renorm) -> float:
    if set(left) != set(right):
        raise RuntimeError("query id sets differ")
    worst = 0.0
    for key in left:
        a = renorm(left[key])
        b = renorm(right[key])
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    return worst


def fp32_softmax_map(common, state: dict) -> dict[str, list[float]]:
    """Same canonical windows and module, float32 forward and softmax. Not the blend map."""
    import torch
    model_mod = common.load_canonical_model()
    data_mod = common.load_canonical_data()
    model_mod.USE_SIGNAL = False
    net = model_mod.CNN(common.H, common.LAYERS, common.K, common.DROP, common.W, use_signal=False)
    net.load_state_dict(state)
    net.float().eval()
    validation = data_mod.load_public_split(REPO / "kit" / "public", "validation")
    query, episodes = model_mod.build_query(validation, common.W)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(query[0]), 4):
            batch = [item[start:start + 4] for item in query]
            logits = net(*batch)
            if logits.dtype != torch.float32:
                raise RuntimeError(f"fp32 probe logits {logits.dtype}")
            pieces.append(torch.softmax(logits.float(), dim=-1).double())
    probs = torch.cat(pieces).numpy()
    if str(probs.dtype) != "float64":
        raise RuntimeError(f"stored probe dtype {probs.dtype}")
    return common.renorm_map(model_mod.to_preds(probs, episodes, validation))


def historical_control_map(state: dict) -> dict[str, list[float]]:
    """The closed hard-signal CPU fp32 path. Read-only. Tune score is applied by the caller."""
    hard_source = REPO / "experiments/cnn_experts/hard_signal/source"
    sys.path.insert(0, str(hard_source))
    import hard_signal as hard  # noqa: WPS433
    from house_edge.data import load_split
    validation = load_split(REPO / "kit" / "public" / "validation")
    model = hard._load_state(state, "cpu")
    return hard.predict_single(model, validation, hard.W)


def lowrate_rows(common) -> list[dict]:
    sys.path.insert(0, str(LOWRATE / "source"))
    import score_fp64  # noqa: WPS433
    rows = []
    specs = (
        ("server_primary_lr2e-5", "server_primary_lr2e-5", "primary", 2e-5),
        ("server_control_lr2e-5", "server_control_lr2e-5", "control", 2e-5),
    )
    for folder, _label, arm, lr in specs:
        run = LOWRATE / "runs" / folder
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if int(summary["total_steps"]) != 730 or summary["exit_ready"] is not True:
            raise RuntimeError(f"{folder} summary is not a finished 730-step run")
        exit_code = (run / "train.exit").read_text(encoding="utf-8").strip()
        if exit_code != "0":
            raise RuntimeError(f"{folder} exit file is {exit_code}")
        for epoch in (1, 3, 5):
            saved = json.loads((run / "scores" / f"epoch_0{epoch}.json").read_text(encoding="utf-8"))
            fresh = score_fp64.score_checkpoint(run, epoch)
            if fresh["n_tune_queries"] != 384 or fresh["confirm_full_test_metrics"] != "not_computed":
                raise RuntimeError(f"{folder} epoch {epoch} rescore contract failed")
            if abs(float(fresh["tune_R"]) - float(saved["tune_R"])) > 0.0:
                raise RuntimeError(
                    f"{folder} epoch {epoch} rescore {fresh['tune_R']!r} != saved {saved['tune_R']!r}"
                )
            if fresh["checkpoint_sha256"] != saved["checkpoint_sha256"]:
                raise RuntimeError(f"{folder} epoch {epoch} checkpoint hash changed")
            rows.append({
                "name": f"{folder}_epoch{epoch}",
                "status": "scored",
                "kind": "single",
                "tune_R": float(fresh["tune_R"]),
                "lr": lr,
                "epoch": epoch,
                "n_components": 1,
                "blend_weight": None,
                "is_current_best": False,
                "host": "server",
                "arm": arm,
                "parent": "curve_C_lr003_s2" if arm == "primary" else "curve_control_s2",
                "train_batch": 128,
                "train_precision": "fp32",
                "historical_parent_batch": 512,
                "historical_parent_precision": "bf16_autocast",
                "fresh_adamw": True,
                "optimizer_resume": False,
                "train_losses": summary["losses"],
                "train_seconds": summary["train_seconds"],
                "checkpoint_sha256": fresh["checkpoint_sha256"],
                "tune_informed": False,
            })
            refuse_if_heavy()
    return rows


def blocked_rows() -> list[dict]:
    reason = (
        "Tower assert_input_pins failed before any GPU step. The frozen trainer hashes "
        "experiments/cnn_runtime/fresh_fp64_cnn03_1956 and the six original "
        "experiments/cnn03_convergence/runs parent paths. Those paths are absent on Tower. "
        "The same bytes are staged under ensemble inputs and the candidate unpacked model. "
        "The handoff --parent flag does not bypass assert_input_pins. No checkpoint was saved. "
        "This comparison does not relabel that failure as a zero score."
    )
    rows = []
    for arm in ("primary", "control"):
        for epoch in (1, 3, 5):
            rows.append({
                "name": f"tower_{arm}_lr5e-5_epoch{epoch}",
                "status": "not_run_preflight_blocked",
                "kind": "single",
                "tune_R": None,
                "lr": 5e-5,
                "epoch": epoch,
                "n_components": 1,
                "blend_weight": None,
                "is_current_best": False,
                "host": "tower",
                "arm": arm,
                "reason": reason,
            })
    return rows


def main() -> None:
    verify_frozen_hashes()
    run_local_tests()
    refuse_if_heavy()
    common = load_ensemble_common()
    common.assert_pinned_files()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing the CPU comparison")
    roles = (
        "curve_C_lr003_s0",
        "curve_C_lr003_s1",
        "curve_C_lr003_s2",
        "hard_shared_control",
    )
    maps = {}
    reproduction = {}
    query_gaps = {}
    for role in roles:
        blob = common.require_bind(ENSEMBLE / "inputs" / role / "checkpoint.pt", role)
        preds = common.forward_probabilities(blob["state"])
        detail = common.score_tune_detail(preds)
        if detail["n_tune_queries"] != 384 or detail["n_forward_queries"] != 768:
            raise SystemExit(f"{role} query contract failed")
        tower = load_npz(ENSEMBLE / "results" / "predictions" / f"{role}.npz")
        gap = max_abs(preds, tower, common.canonical_renorm)
        query_gaps[role] = gap
        if gap > QUERY_ATOL:
            raise SystemExit(f"{role} tower map gap {gap} exceeds 1e-8")
        published = common.PUBLISHED_TUNE_R[role]
        reproduction[role] = {
            "tune_R": detail["tune_R"],
            "published_tune_R": published,
            "abs_delta": abs(detail["tune_R"] - published),
            "within_1e-12": common.reproduction_ok(detail["tune_R"], published),
            "max_abs_vs_tower_npz": gap,
        }
        maps[role] = preds
        print(f"COMPONENT {role} tune_R={detail['tune_R']:.17f} map_gap={gap:.3e}", flush=True)
        refuse_if_heavy()
        if role == "hard_shared_control":
            control_state = blob["state"]
    seed2_r = reproduction["curve_C_lr003_s2"]["tune_R"]
    if seed2_r != selector.CURRENT_BEST_R:
        raise SystemExit(f"seed 2 tune R {seed2_r!r} is not the immutable current best")
    rows = []
    map_rows = {}
    for name, members in common.ENSEMBLES.items():
        mixed = common.equal_probability_mean([maps[item] for item in members])
        detail = common.score_tune_detail(mixed)
        tower = load_npz(ENSEMBLE / "results" / "predictions" / f"{name}.npz")
        gap = max_abs(mixed, tower, common.canonical_renorm)
        if gap > QUERY_ATOL:
            raise SystemExit(f"{name} tower map gap {gap} exceeds 1e-8")
        map_rows[name] = gap
        rows.append({
            "name": name,
            "status": "scored",
            "kind": "ensemble",
            "tune_R": detail["tune_R"],
            "lr": None,
            "epoch": None,
            "n_components": len(members),
            "blend_weight": None,
            "is_current_best": False,
            "members": list(members),
            "tune_informed": name == "ensemble_equal_s0_s2",
            "uses_shared_control": False,
            "max_abs_vs_tower_npz": gap,
        })
        print(f"ROW {name} tune_R={detail['tune_R']:.17f} map_gap={gap:.3e}", flush=True)
    base = maps["curve_C_lr003_s2"]
    control = maps["hard_shared_control"]
    for weight in common.BLEND_WEIGHTS:
        mixed = common.blend_maps(base, control, weight)
        if weight == 0.0:
            gap_identity = max_abs(mixed, base, common.canonical_renorm)
            if gap_identity != 0.0:
                raise SystemExit("weight 0 is not an exact seed-2 copy")
        detail = common.score_tune_detail(mixed)
        name = f"blend_s2_control_w{weight}"
        tower = load_npz(ENSEMBLE / "results" / "predictions" / f"{name}.npz")
        gap = max_abs(mixed, tower, common.canonical_renorm)
        if gap > QUERY_ATOL:
            raise SystemExit(f"{name} tower map gap {gap} exceeds 1e-8")
        map_rows[name] = gap
        rows.append({
            "name": name,
            "status": "scored",
            "kind": "blend",
            "tune_R": detail["tune_R"],
            "lr": None,
            "epoch": None,
            "n_components": 2,
            "blend_weight": weight,
            "is_current_best": False,
            "members": ["curve_C_lr003_s2", "hard_shared_control"],
            "tune_informed": False,
            "uses_shared_control": True,
            "max_abs_vs_tower_npz": gap,
        })
        print(f"ROW {name} tune_R={detail['tune_R']:.17f} map_gap={gap:.3e}", flush=True)
    fp32_map = fp32_softmax_map(common, control_state)
    historical = historical_control_map(control_state)
    fp32_detail = common.score_tune_detail(fp32_map)
    historical_detail = common.score_tune_detail(historical)
    diagnosis = {
        "checkpoint_sha256": "92233992aeb8c007a40f925885d7f85ef260cf8584ea0cc669dcd39ce57d201f",
        "same_bytes_as_hard_signal_runs_s0_control": True,
        "canonical_fp64_tune_R": reproduction["hard_shared_control"]["tune_R"],
        "historical_published_tune_R": PUBLISHED_CONTROL_R,
        "canonical_minus_published": reproduction["hard_shared_control"]["tune_R"] - PUBLISHED_CONTROL_R,
        "fp32_softmax_same_canonical_windows_tune_R": fp32_detail["tune_R"],
        "fp32_softmax_abs_delta_vs_published": abs(fp32_detail["tune_R"] - PUBLISHED_CONTROL_R),
        "fp32_softmax_abs_delta_vs_fp64": abs(fp32_detail["tune_R"] - reproduction["hard_shared_control"]["tune_R"]),
        "historical_hard_signal_fp32_loader_tune_R": historical_detail["tune_R"],
        "historical_abs_delta_vs_published": abs(historical_detail["tune_R"] - PUBLISHED_CONTROL_R),
        "historical_within_1e-12": abs(historical_detail["tune_R"] - PUBLISHED_CONTROL_R) <= 1e-12,
        "max_abs_fp64_vs_historical_map": max_abs(control, historical, common.canonical_renorm),
        "max_abs_fp64_vs_fp32_softmax_map": max_abs(control, fp32_map, common.canonical_renorm),
        "probabilities_adjusted": False,
        "blend_uses": "unmodified canonical float64 map",
        "ensemble_equal_s0_s2_uses_shared_control": False,
    }
    print(
        "CONTROL_DIAGNOSIS "
        f"fp64={diagnosis['canonical_fp64_tune_R']:.17f} "
        f"historical={diagnosis['historical_hard_signal_fp32_loader_tune_R']:.17f} "
        f"fp32softmax={diagnosis['fp32_softmax_same_canonical_windows_tune_R']:.17f}",
        flush=True,
    )
    rows.append({
        "name": "current_best",
        "status": "scored",
        "kind": "single",
        "tune_R": selector.CURRENT_BEST_R,
        "lr": 0.003,
        "epoch": None,
        "n_components": 1,
        "blend_weight": None,
        "is_current_best": True,
        "members": ["curve_C_lr003_s2"],
        "tune_informed": False,
        "uses_shared_control": False,
        "reproduced_tune_R": seed2_r,
    })
    rows.extend(lowrate_rows(common))
    rows.extend(blocked_rows())
    decision = selector.select(rows)
    payload = {
        "device": "cpu",
        "dtype": "float64",
        "interpreter": "experiments/cnn_runtime/py313_verify_1936/env/bin/python",
        "confirm_full_test_scored": False,
        "package_written": False,
        "reproduction": reproduction,
        "shared_control_diagnosis": diagnosis,
        "rows": rows,
        "selection": decision,
        "rss_kb": rss_kb(),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"COMPARISON_OK selected={decision['selected_name']} "
        f"R={decision['selected_tune_R']:.17f} package_written=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
