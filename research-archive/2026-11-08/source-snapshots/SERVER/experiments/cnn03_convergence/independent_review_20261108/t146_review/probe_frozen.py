"""Frozen-tip checks. Writes only this review directory. No training."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[3]
LOW = REPO / "experiments/cnn03_convergence/lowrate_20261108"
ENS = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108"
REPORT: dict = {"reviewed_at_utc": datetime.now(timezone.utc).isoformat()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_list(root: Path, list_name: str) -> dict:
    mismatches = []
    rows = []
    for line in (root / list_name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        got = sha256_file(root / rel)
        rows.append({"path": rel, "sha256": got, "ok": got == digest})
        if got != digest:
            mismatches.append(rel)
    return {"rows": rows, "mismatches": mismatches}


def max_abs_maps(left: dict, right: dict) -> float:
    if set(left) != set(right):
        raise RuntimeError(f"id mismatch {len(left)} {len(right)} {len(set(left) ^ set(right))}")
    worst = 0.0
    for key in left:
        for a, b in zip(left[key], right[key]):
            worst = max(worst, abs(float(a) - float(b)))
    return worst


def load_npz(path: Path) -> dict:
    import numpy as np
    blob = np.load(path)
    ids = [str(item) for item in blob["query_id"].tolist()]
    values = blob["probability"]
    return {key: [float(v) for v in row] for key, row in zip(ids, values)}


def main() -> None:
    import torch

    torch.set_num_threads(2)
    REPORT["python"] = sys.version.split()[0]
    REPORT["torch"] = torch.__version__
    REPORT["lowrate_source"] = check_list(LOW, "SOURCE_SHA256.txt")
    text = (LOW / "source/train_lowrate.py").read_text(encoding="utf-8")
    REPORT["trainer_now"] = {
        "five_arg_call": "logits = model(*batch)" in text and "logits = model(*batch[:-1])" not in text,
        "four_arg_call": "logits = model(*batch[:-1])" in text,
        "rocm_smi": "rocm-smi" in text,
        "exclusive": "O_EXCL" in text or "already exists" in text,
        "loss_append": 'open("a"' in text,
        "adamw_host_lr": "AdamW(model.parameters(), lr=lr, weight_decay=WD)" in text,
        "parent_lr_in_adam": 'blob["config"]["lr"]' in text,
    }
    pins = {
        "PINNED.json": REPO / "experiments/mlp_cnn_hpo/review/PINNED.json",
        "fp64_harness.py": REPO / "experiments/cnn_runtime/fp64_harness.py",
        "best_zip": REPO / "experiments/cnn03_convergence/candidate_lr003_s2_20261108/curve_C_lr003_s2_fp64_candidate.zip",
        "best_csv": REPO / "experiments/cnn03_convergence/candidate_lr003_s2_20261108/refs/candidate_test_a.csv",
        "fallback_zip": REPO / "experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip",
        "fallback_csv": REPO / "experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv",
        "canonical_model": REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py",
    }
    expect = {
        "PINNED.json": "cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb",
        "fp64_harness.py": "515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2",
        "best_zip": "dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29",
        "best_csv": "9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6",
        "fallback_zip": "61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599",
        "fallback_csv": "08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d",
        "canonical_model": "7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a",
    }
    REPORT["protected"] = {key: sha256_file(path) == expect[key] for key, path in pins.items()}
    REPORT["tower_run_dirs"] = [p.name for p in (LOW / "runs").iterdir() if p.is_dir()]
    server = {}
    for name in ("server_primary_lr2e-5", "server_control_lr2e-5"):
        run = LOW / "runs" / name
        loss_lines = (run / "loss.jsonl").read_text(encoding="utf-8").splitlines()
        epochs = []
        for epoch in (1, 3, 5):
            score = json.loads((run / "scores" / f"epoch_{epoch:02d}.json").read_text(encoding="utf-8"))
            ckpt = run / f"epoch_{epoch:02d}" / "checkpoint.pt"
            score["file_sha256"] = sha256_file(ckpt)
            score["sha_matches_file"] = score["file_sha256"] == score["checkpoint_sha256"]
            epochs.append(score)
        server[name] = {
            "exit": (run / "train.exit").read_text(encoding="utf-8").strip(),
            "loss_lines": len(loss_lines),
            "summary_steps": json.loads((run / "summary.json").read_text(encoding="utf-8"))["total_steps"],
            "launch_lr": json.loads((run / "launch.json").read_text(encoding="utf-8"))["train_lr"],
            "device": json.loads((run / "launch.json").read_text(encoding="utf-8"))["device"],
            "scores": epochs,
        }
    REPORT["server_artifacts"] = server

    manifest = json.loads((ENS / "MANIFEST.json").read_text(encoding="utf-8"))
    copies = {}
    for role, spec in manifest["checkpoints"].items():
        copied = ENS / spec["path"]
        original_rel = None
        if role == "hard_shared_control":
            original = REPO / "experiments/cnn_experts/hard_signal/runs/s0/control/checkpoint.pt"
        else:
            original = REPO / spec["original_out"] / "checkpoint.pt"
        copies[role] = {
            "copied": sha256_file(copied),
            "pin": spec["sha256"],
            "original": sha256_file(original) if original.is_file() else None,
            "copied_eq_pin": sha256_file(copied) == spec["sha256"],
            "copied_eq_original": original.is_file() and sha256_file(copied) == sha256_file(original),
        }
    REPORT["ensemble_checkpoint_copies"] = copies

    sys.path.insert(0, str(ENS / "source"))
    import common as C

    stored = {}
    for name in (
        "curve_C_lr003_s0",
        "curve_C_lr003_s1",
        "curve_C_lr003_s2",
        "hard_shared_control",
        "ensemble_equal_s0_s2",
        "ensemble_equal_s0_s1_s2",
        "blend_s2_control_w0.0",
        "blend_s2_control_w0.025",
        "blend_s2_control_w0.05",
        "blend_s2_control_w0.1",
        "blend_s2_control_w0.2",
    ):
        stored[name] = load_npz(ENS / "results/predictions" / f"{name}.npz")
    fresh = {}
    for role in ("curve_C_lr003_s0", "curve_C_lr003_s2", "hard_shared_control"):
        blob = C.require_bind(ENS / manifest["checkpoints"][role]["path"], role)
        fresh[role] = C.forward_probabilities(blob["state"])
        print(f"FORWARD {role} n={len(fresh[role])}", flush=True)
    mean_02 = C.equal_probability_mean([fresh["curve_C_lr003_s0"], fresh["curve_C_lr003_s2"]])
    identity = max_abs_maps(stored["blend_s2_control_w0.0"], stored["curve_C_lr003_s2"])
    maps = {
        "s0_vs_npz": max_abs_maps(fresh["curve_C_lr003_s0"], stored["curve_C_lr003_s0"]),
        "s2_vs_npz": max_abs_maps(fresh["curve_C_lr003_s2"], stored["curve_C_lr003_s2"]),
        "control_vs_npz": max_abs_maps(fresh["hard_shared_control"], stored["hard_shared_control"]),
        "fresh_mean_02_vs_npz": max_abs_maps(mean_02, stored["ensemble_equal_s0_s2"]),
        "weight0_vs_s2": identity,
        "stored_mean_02_vs_npz": max_abs_maps(
            C.equal_probability_mean([stored["curve_C_lr003_s0"], stored["curve_C_lr003_s2"]]),
            stored["ensemble_equal_s0_s2"],
        ),
        "stored_mean_012_vs_npz": max_abs_maps(
            C.equal_probability_mean([
                stored["curve_C_lr003_s0"],
                stored["curve_C_lr003_s1"],
                stored["curve_C_lr003_s2"],
            ]),
            stored["ensemble_equal_s0_s1_s2"],
        ),
    }
    for weight in (0.025, 0.05, 0.1, 0.2):
        rebuilt = C.blend_maps(stored["curve_C_lr003_s2"], stored["hard_shared_control"], weight)
        maps[f"blend_{weight}_vs_npz"] = max_abs_maps(rebuilt, stored[f"blend_s2_control_w{weight}"])
    scored = {}
    for name in ("curve_C_lr003_s0", "curve_C_lr003_s2", "ensemble_equal_s0_s2", "hard_shared_control"):
        detail = C.score_tune_detail(stored[name])
        scored[name] = {
            "tune_R": detail["tune_R"],
            "n_tune": detail["n_tune_queries"],
            "n_forward": detail["n_forward_queries"],
        }
        print(f"SCORE_NPZ {name} R={detail['tune_R']:.17f} forward={detail['n_forward_queries']}", flush=True)
    fresh_s2 = C.score_tune_detail(fresh["curve_C_lr003_s2"])
    fresh_02 = C.score_tune_detail(mean_02)
    scored["fresh_s2"] = fresh_s2["tune_R"]
    scored["fresh_mean_02"] = fresh_02["tune_R"]
    REPORT["ensemble_maps"] = maps
    REPORT["ensemble_scores"] = scored

    # Historical control path. Score tune only. Do not write or rescale maps.
    hist = {}
    try:
        hard_root = REPO / "experiments/cnn_experts/hard_signal"
        sys.path.insert(0, str(hard_root / "source"))
        import hard_signal as HS
        from house_edge.data import load_split

        validation = load_split(os.path.join(HS.B.PUB, "validation"))
        state_path = ENS / "inputs/hard_shared_control/checkpoint.pt"
        blob = torch.load(state_path, map_location="cpu", weights_only=False)
        model = HS._load_state(blob["state"], "cpu")
        hist_preds = HS.predict_single(model, validation, HS.W)
        hist_detail = C.score_tune_detail(hist_preds)
        hist = {
            "tune_R": hist_detail["tune_R"],
            "published": 0.46211193532631023,
            "abs_delta_vs_published": abs(hist_detail["tune_R"] - 0.46211193532631023),
            "max_abs_vs_canonical_npz": max_abs_maps(hist_preds, stored["hard_shared_control"]),
            "n_forward": hist_detail["n_forward_queries"],
        }
        print(
            f"HIST_CONTROL R={hist_detail['tune_R']:.17f} "
            f"delta_published={hist['abs_delta_vs_published']:.3e} "
            f"map_abs={hist['max_abs_vs_canonical_npz']:.3e}",
            flush=True,
        )
    except Exception as exc:
        hist = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"HIST_CONTROL_ERROR {hist['error']}", flush=True)
    REPORT["historical_control"] = hist

    sys.path.insert(0, str(LOW / "source"))
    import score_fp64

    rescored = {}
    for name in ("server_primary_lr2e-5", "server_control_lr2e-5"):
        rescored[name] = []
        for epoch in (1, 3, 5):
            result = score_fp64.score_checkpoint(LOW / "runs" / name, epoch)
            stored_score = next(item for item in server[name]["scores"] if item["epoch"] == epoch)
            rescored[name].append({
                "epoch": epoch,
                "tune_R": result["tune_R"],
                "n_forward": result["n_forward_queries"],
                "matches_saved": result["tune_R"] == stored_score["tune_R"],
            })
            print(
                f"RESCORE {name} epoch={epoch} R={result['tune_R']:.17f} "
                f"match={result['tune_R'] == stored_score['tune_R']}",
                flush=True,
            )
    REPORT["lowrate_rescore"] = rescored
    (REVIEW / "probe_frozen_report.json").write_text(json.dumps(REPORT, indent=1) + "\n", encoding="utf-8")
    print("PROBE_FROZEN_OK", flush=True)


if __name__ == "__main__":
    main()
