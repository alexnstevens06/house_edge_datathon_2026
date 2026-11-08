"""Read-only tip probe for the unfrozen low-rate study. CPU only. Writes here."""
from __future__ import annotations

import ast
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
STUDY = REPO / "experiments/cnn03_convergence/lowrate_20261108"
ENSEMBLE = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108"
sys.path.insert(0, str(STUDY / "source"))

REPORT: dict = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import torch

    torch.set_num_threads(2)
    REPORT["python"] = sys.version.split()[0]
    REPORT["torch"] = torch.__version__
    REPORT["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    REPORT["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    names = [
        "COORDINATOR_PLAN.md",
        "PLAN.md",
        "MANIFEST.json",
        "source/contract.py",
        "source/train_lowrate.py",
        "source/score_fp64.py",
        "source/preflight_fp32.py",
        "source/preflight_fp64.py",
    ]
    hashed = {}
    for rel in names:
        path = STUDY / rel
        hashed[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
    REPORT["source_hashes"] = hashed
    missing = [
        rel
        for rel in (
            "HANDOFF_READY",
            "SPLIT_IDS.json",
            "SOURCE_SHA256.txt",
            "tests/test_lowrate.py",
        )
        if not (STUDY / rel).exists()
    ]
    REPORT["missing"] = missing
    REPORT["ensemble_dir_exists"] = ENSEMBLE.exists()
    REPORT["protocol_sha_matches_manifest"] = (
        hashed["COORDINATOR_PLAN.md"]["sha256"]
        == json.loads((STUDY / "MANIFEST.json").read_text(encoding="utf-8"))["protocol_sha256"]
    )
    protected = {
        "experiments/mlp_cnn_hpo/review/PINNED.json": "cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb",
        "experiments/cnn_runtime/fp64_harness.py": "515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2",
        "experiments/cnn03_convergence/candidate_lr003_s2_20261108/curve_C_lr003_s2_fp64_candidate.zip": "dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29",
        "experiments/cnn03_convergence/candidate_lr003_s2_20261108/refs/candidate_test_a.csv": "9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6",
        "experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip": "61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599",
        "experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv": "08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d",
    }
    REPORT["protected_match"] = {
        rel: sha256_file(REPO / rel) == expect for rel, expect in protected.items()
    }

    tree = ast.parse((STUDY / "source/train_lowrate.py").read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"save", "AdamW"}:
            calls.append(ast.dump(node)[:240])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "model":
            calls.append(f"line {node.lineno}: model({ast.dump(node.args[0]) if node.args else ''})")
    REPORT["trainer_calls"] = calls
    text = (STUDY / "source/train_lowrate.py").read_text(encoding="utf-8")
    REPORT["trainer_flags"] = {
        "passes_star_batch": "logits = model(*batch)" in text,
        "preflight_drops_label": "model(*batch[:-1])" in (STUDY / "source/preflight_fp32.py").read_text(encoding="utf-8"),
        "canonical_trainer_unpacks_four": "lg = model(x, t, ty, sg)" in (
            REPO / "experiments/cnn03_convergence/source/conv_train.py"
        ).read_text(encoding="utf-8"),
        "uses_parent_config_lr_for_adam": "blob[\"config\"][\"lr\"]" in text or "cfg[\"lr\"]" in text,
        "adamw_uses_host_lr": "AdamW(model.parameters(), lr=lr, weight_decay=WD)" in text,
        "nvidia_smi_hard_fail": "nvidia-smi failed" in text,
        "rocm_smi": "rocm-smi" in text,
        "loss_log_append": 'loss_log.open("a"' in text,
        "exclusive_create": "O_EXCL" in text or "exist_ok=False" in text,
        "one_cycle": "OneCycle" in text,
        "autocast": "autocast" in text,
    }

    from contract import (
        ARMS,
        CONTINUATION_SEED,
        EPOCHS,
        HOST_LR,
        SAVE_EPOCHS,
        STEPS_PER_EPOCH,
        TOTAL_STEPS,
        budget_errors,
        expected_shapes,
        host_lr,
        run_errors,
        state_errors,
        tune_id_errors,
    )

    REPORT["budget_math"] = {
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": EPOCHS,
        "total_steps": TOTAL_STEPS,
        "four_job_steps": 4 * TOTAL_STEPS,
        "ceil_ok": STEPS_PER_EPOCH == 146 and TOTAL_STEPS == 730 and 4 * TOTAL_STEPS == 2920,
    }
    REPORT["host_lr"] = {key: host_lr(key) for key in HOST_LR}
    REPORT["optimizer_lr_is_not_parent"] = host_lr("server") != 0.003 and host_lr("tower") != 0.00197

    shapes = expected_shapes()
    good_state = {key: torch.zeros(shape, dtype=torch.float32) for key, shape in shapes.items()}
    good_cfg = {
        "host": "server",
        "arm": "primary",
        "train_lr": host_lr("server"),
        "continuation_seed": CONTINUATION_SEED,
        "precision": "fp32",
        "scheduler": "none",
        "autocast": False,
        "bs": 128,
        "wd": 0.00895,
        "completed_epochs": 1,
        "completed_steps": 146,
        "parent_train_bs": 512,
        "parent_precision": "bf16_autocast",
        "warm_start_from": "weights_only",
    }
    good = {
        "state": good_state,
        "optimizer": {},
        "config": good_cfg,
        "torch_rng": b"0",
        "cuda_rng": [],
        "numpy_rng": ("MT",),
        "completed_epochs": 1,
        "parent_checkpoint_sha256": "bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71",
        "parent_had_optimizer": False,
        "warm_start_from": "weights_only",
    }
    checks = {
        "clean_epoch1": run_errors(good, "abc", "abc"),
        "wrong_hash": run_errors(good, "abc", "def"),
        "parent_lr_as_train_lr": run_errors(
            {**good, "config": {**good_cfg, "train_lr": 0.003}}, "abc", "abc"
        ),
        "control_base_lr_as_train_lr": run_errors(
            {**good, "config": {**good_cfg, "train_lr": 0.00197}}, "abc", "abc"
        ),
        "wrong_parent_sha": run_errors(good, "abc", "abc")
        if False
        else run_errors({**good, "parent_checkpoint_sha256": "0" * 64}, "abc", "abc"),
        "extra_key": state_errors({**good_state, "extra.weight": torch.zeros(1)}),
        "bad_shape": state_errors(
            {**good_state, "blocks.0.conv.weight": torch.zeros(128, 128, 5)}
        ),
        "bad_dtype": state_errors(
            {**good_state, "head.2.bias": good_state["head.2.bias"].to(torch.float16)}
        ),
        "nonfinite": state_errors(
            {**good_state, "head.2.bias": torch.tensor([float("nan"), 0, 0, 0, 0])}
        ),
        "budget_epoch2": budget_errors(2, 292),
        "budget_epoch1_steps": budget_errors(1, 145),
        "budget_epoch5": budget_errors(5, 730),
        "nontune_extra": tune_id_errors({"q1", "other"}, {"q1"}, set()),
        "nontune_missing": tune_id_errors(set(), {"q1"}, set()),
    }
    # Confirm a clean control blob uses the control parent pin, not the primary pin.
    control = {
        **good,
        "config": {**good_cfg, "host": "tower", "arm": "control", "train_lr": host_lr("tower"), "completed_epochs": 5, "completed_steps": 730},
        "completed_epochs": 5,
        "parent_checkpoint_sha256": "47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9",
    }
    checks["clean_tower_control_epoch5"] = run_errors(control, "abc", "abc")
    checks["server_rate_on_tower_label"] = run_errors(
        {**control, "config": {**control["config"], "train_lr": host_lr("server")}}, "abc", "abc"
    )
    REPORT["guard_results"] = {name: list(errors) for name, errors in checks.items()}
    REPORT["guard_expect"] = {
        "clean_epoch1_empty": checks["clean_epoch1"] == [],
        "clean_tower_control_empty": checks["clean_tower_control_epoch5"] == [],
        "wrong_hash_caught": any("hash mismatch" == item for item in checks["wrong_hash"]),
        "parent_lrs_caught": any("train lr" in item for item in checks["parent_lr_as_train_lr"])
        and any("train lr" in item for item in checks["control_base_lr_as_train_lr"]),
        "shape_dtype_finite_key_caught": all(
            checks[name] for name in ("extra_key", "bad_shape", "bad_dtype", "nonfinite")
        ),
        "budget_caught": checks["budget_epoch2"] != [] and checks["budget_epoch1_steps"] != [] and checks["budget_epoch5"] == [],
        "nontune_caught": checks["nontune_extra"] != [] and checks["nontune_missing"] != [],
        "cross_host_rate_caught": checks["server_rate_on_tower_label"] != [],
    }
    REPORT["arms"] = ARMS
    REPORT["save_epochs"] = list(SAVE_EPOCHS)

    conv = REPO / "experiments/cnn03_convergence/source"
    sys.path.insert(0, str(conv))
    import base_cnn as B

    B.USE_SIGNAL = False
    model = B.CNN(128, 4, 3, 0.3, 64, use_signal=False)
    x = torch.zeros(2, 64, 8, dtype=torch.int64)
    t = torch.full((2,), 20)
    ty = torch.zeros(2, 8, dtype=torch.int64)
    sg = torch.zeros(2, 8, dtype=torch.int64)
    y = torch.zeros(2, 8, dtype=torch.int64)
    four = None
    five = None
    try:
        logits = model(x, t, ty, sg)
        four = str(logits.dtype)
    except Exception as exc:
        four = f"{type(exc).__name__}: {exc}"
    try:
        model(x, t, ty, sg, y)
        five = "accepted"
    except TypeError as exc:
        five = f"TypeError: {exc}"
    float_impl = torch.Tensor.float
    loss = B.masked_loss(torch.zeros(2, 8, 5), y)
    REPORT["forward_probe"] = {
        "four_arg_dtype": four,
        "five_arg": five,
        "masked_loss_finite": bool(torch.isfinite(loss)),
        "tensor_float_identity_unchanged": torch.Tensor.float is float_impl,
        "signal_module": model.sig is None,
    }

    fp64_entry = REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956"
    sys.path.insert(0, str(fp64_entry))
    spec = importlib.util.spec_from_file_location("t146_fp64_model", fp64_entry / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = torch.load(
        REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    module.USE_SIGNAL = False
    net = module.CNN(128, 4, 3, 0.3, 64, use_signal=False)
    net.load_state_dict(parent["state"])
    net.double()
    net.eval()
    with torch.no_grad():
        logits64 = net(x, t, ty, sg)
        probs = torch.softmax(logits64, dim=-1)
    floating = [p for p in net.parameters() if p.is_floating_point()]
    REPORT["fp64_forward"] = {
        "parent_keys": sorted(parent),
        "parent_lr": parent["config"].get("lr"),
        "parent_seed": parent["config"].get("seed"),
        "parent_has_kernel": "kernel" in parent["config"],
        "params": len(floating),
        "param_dtype": str(floating[0].dtype),
        "logit_dtype": str(logits64.dtype),
        "softmax_dtype": str(probs.dtype),
        "autocast": bool(torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu")),
        "tensor_float_unchanged": torch.Tensor.float is float_impl,
    }
    control = torch.load(
        REPO / "experiments/cnn03_convergence/runs/curve_control_s2/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    REPORT["control_parent"] = {
        "keys": sorted(control),
        "lr": control["config"].get("lr"),
        "seed": control["config"].get("seed"),
        "bs": control["config"].get("bs"),
    }
    (REVIEW / "probe_report.json").write_text(json.dumps(REPORT, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_ok": REPORT["protocol_sha_matches_manifest"],
        "missing": missing,
        "ensemble": REPORT["ensemble_dir_exists"],
        "five_arg": five,
        "four_arg": four,
        "guards": REPORT["guard_expect"],
        "fp64": REPORT["fp64_forward"],
        "flags": REPORT["trainer_flags"],
        "protected": REPORT["protected_match"],
    }, indent=1))


if __name__ == "__main__":
    main()
