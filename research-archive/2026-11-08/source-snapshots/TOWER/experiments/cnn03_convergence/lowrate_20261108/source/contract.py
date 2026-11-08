"""Frozen low-LR study contract. Reads pins. Writes nothing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
MANIFEST_PATH = STUDY / "MANIFEST.json"
SPLIT_IDS_PATH = STUDY / "SPLIT_IDS.json"
SOURCE_SHA_PATH = STUDY / "SOURCE_SHA256.txt"
FP64_ENTRY = REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956"
CONV_SOURCE = REPO / "experiments/cnn03_convergence/source"

H, LAYERS, KERNEL, DROP, W = 128, 4, 3, 0.3, 64
BS = 128
EPOCHS = 5
SAVE_EPOCHS = (1, 3, 5)
EXPECTED_WINDOWS = 18567
STEPS_PER_EPOCH = (EXPECTED_WINDOWS + BS - 1) // BS
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
WD = 0.00895
GRAD_CLIP = 1.0
CONTINUATION_SEED = 20261108
PARENT_BATCH = 512
HOST_LR = {"server": 2e-5, "tower": 5e-5}
ARMS = {
    "primary": "curve_C_lr003_s2",
    "control": "curve_control_s2",
}


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "pay.weight": (6, H),
        "mask.weight": (H, 1),
        "mask.bias": (H,),
        "typ.weight": (3, H),
        "slot.weight": (8, H),
        "tick.weight": (H, 2),
        "tick.bias": (H,),
        "head.0.weight": (H, 2 * H),
        "head.0.bias": (H,),
        "head.2.weight": (5, H),
        "head.2.bias": (5,),
    }
    for index in range(LAYERS):
        prefix = f"blocks.{index}"
        shapes[f"{prefix}.conv.weight"] = (H, H, KERNEL)
        shapes[f"{prefix}.conv.bias"] = (H,)
        shapes[f"{prefix}.mix.weight"] = (H, 2 * H)
        shapes[f"{prefix}.mix.bias"] = (H,)
        shapes[f"{prefix}.n1.weight"] = (H,)
        shapes[f"{prefix}.n1.bias"] = (H,)
        shapes[f"{prefix}.n2.weight"] = (H,)
        shapes[f"{prefix}.n2.bias"] = (H,)
    return shapes


def state_errors(state: dict) -> list[str]:
    if not isinstance(state, dict):
        return ["state is not an object"]
    leaked = [str(key) for key in state if key == "sig.weight" or str(key).startswith("sig.")]
    if leaked:
        return ["signal tensors present: " + ", ".join(leaked)]
    want = expected_shapes()
    if len(want) != 43:
        return ["internal shape table is not 43"]
    if set(state) != set(want):
        missing = sorted(set(want) - set(state))
        extra = sorted(set(state) - set(want))
        return [f"architecture keys mismatch missing={missing} extra={extra}"]
    errors = []
    for key, shape in want.items():
        tensor = state[key]
        if tuple(tensor.shape) != shape:
            errors.append(f"{key} shape {tuple(tensor.shape)} != {shape}")
        elif tensor.dtype != torch.float32:
            errors.append(f"{key} saved dtype {tensor.dtype} is not float32")
        elif not bool(torch.isfinite(tensor).all()):
            errors.append(f"{key} is not finite")
    return errors


def assert_input_pins() -> None:
    pin = manifest()
    if pin["protocol_sha256"] != sha256_file(STUDY / "COORDINATOR_PLAN.md"):
        raise RuntimeError("protocol hash drift")
    for rel, digest in pin["sources"].items():
        got = sha256_file(REPO / rel)
        if got != digest:
            raise RuntimeError(f"source hash drift {rel}")
    for name, spec in pin["checkpoints"].items():
        got = sha256_file(REPO / spec["path"])
        if got != spec["sha256"]:
            raise RuntimeError(f"parent hash drift {name}")
    if STEPS_PER_EPOCH != 146 or TOTAL_STEPS != 730:
        raise RuntimeError("step budget arithmetic drifted")


def parent_errors(blob: dict, arm: str, digest: str) -> list[str]:
    pin = manifest()["checkpoints"][ARMS[arm]]
    errors = []
    if digest != pin["sha256"]:
        errors.append("hash mismatch")
    if set(blob) != {"config", "state"}:
        errors.append("parent is not weights-only")
        return errors
    cfg = blob["config"]
    if int(cfg.get("seed", -1)) != 2:
        errors.append("parent seed is not 2")
    if abs(float(cfg.get("lr", -1)) - float(pin["lr"])) > 1e-12:
        errors.append("parent lr mismatch")
    for key, want in (("h", H), ("layers", LAYERS), ("W", W), ("bs", PARENT_BATCH), ("epochs", 30)):
        if cfg.get(key) != want:
            errors.append(f"parent config {key} mismatch")
    if abs(float(cfg.get("drop", -1)) - DROP) > 1e-12 or abs(float(cfg.get("wd", -1)) - WD) > 1e-12:
        errors.append("parent drop/wd mismatch")
    errors.extend(state_errors(blob["state"]))
    return errors


def budget_errors(completed_epochs: int, completed_steps: int) -> list[str]:
    errors = []
    if completed_epochs not in SAVE_EPOCHS:
        errors.append("epoch not in frozen save set")
    elif completed_steps != completed_epochs * STEPS_PER_EPOCH:
        errors.append("step budget")
    return errors


def refuse_if_no_cuda(cuda_available: bool) -> None:
    if not cuda_available:
        raise RuntimeError("CUDA required; refusing CPU training")


def host_lr(host: str) -> float:
    if host not in HOST_LR:
        raise RuntimeError(f"unknown host {host}")
    return HOST_LR[host]


def tune_id_errors(pred_ids: set[str], tune_ids: set[str], confirm_ids: set[str]) -> list[str]:
    errors = []
    if not tune_ids <= pred_ids:
        errors.append("missing tune ids")
    unknown = pred_ids - tune_ids - confirm_ids
    if unknown:
        errors.append("non-tune query id")
    return errors


def run_errors(blob: dict, digest: str, snapshot_digest: str) -> list[str]:
    errors = []
    if digest != snapshot_digest:
        errors.append("hash mismatch")
    if not isinstance(blob, dict):
        return ["checkpoint is not an object"]
    if "scheduler" in blob:
        errors.append("scheduler state is not part of this recipe")
    for key in ("state", "optimizer", "config", "torch_rng", "cuda_rng", "numpy_rng"):
        if key not in blob:
            errors.append(f"missing {key}")
    cfg = blob.get("config") or {}
    errors.extend(state_errors(blob.get("state") or {}))
    host = cfg.get("host")
    arm = cfg.get("arm")
    if host not in HOST_LR or arm not in ARMS:
        errors.append("host/arm mismatch")
        return errors
    if abs(float(cfg.get("train_lr", -1)) - host_lr(host)) > 1e-15:
        errors.append("train lr mismatch")
    if cfg.get("continuation_seed") != CONTINUATION_SEED:
        errors.append("continuation seed mismatch")
    if cfg.get("warm_start_from") != "weights_only" or blob.get("parent_had_optimizer") is not False:
        errors.append("warm-start disclosure mismatch")
    if blob.get("parent_checkpoint_sha256") != manifest()["checkpoints"][ARMS[arm]]["sha256"]:
        errors.append("parent hash mismatch")
    if cfg.get("precision") != "fp32" or cfg.get("scheduler") != "none" or cfg.get("autocast") is not False:
        errors.append("precision/scheduler disclosure mismatch")
    errors.extend(budget_errors(int(cfg.get("completed_epochs", -1)), int(cfg.get("completed_steps", -1))))
    if int(blob.get("completed_epochs", -1)) != int(cfg.get("completed_epochs", -2)):
        errors.append("completed epoch mismatch")
    if cfg.get("bs") != BS or abs(float(cfg.get("wd", -1)) - WD) > 1e-15:
        errors.append("config wd/bs mismatch")
    if cfg.get("parent_train_bs") != PARENT_BATCH or cfg.get("parent_precision") != "bf16_autocast":
        errors.append("historical batch/precision disclosure mismatch")
    return errors


def canonical_renorm(values) -> list[float]:
    vals = [float(v) for v in values]
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        raise RuntimeError("non-finite probability")
    total = sum(vals)
    if not total > 0.0:
        raise RuntimeError("probability mass is not positive")
    return [v / total for v in vals]
