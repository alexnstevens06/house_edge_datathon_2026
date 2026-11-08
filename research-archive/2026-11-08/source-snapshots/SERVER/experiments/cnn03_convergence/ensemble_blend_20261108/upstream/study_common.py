"""Study helpers. Reads canonical files. Writes nothing outside the continuation dir."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
MANIFEST_PATH = STUDY / "MANIFEST.json"
POST_RUN_PATH = STUDY / "POST_RUN_HASHES.json"
FP64_ENTRY = REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956"
CONV_SOURCE = REPO / "experiments/cnn03_convergence/source"
H, LAYERS, K, DROP, W, BS, WD = 128, 4, 3, 0.3, 64, 512, 0.00895
EXPECTED_WINDOWS = 18567
STEPS_PER_EPOCH = 37
EXTRA_EPOCHS = (10, 20)
BLEND_WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2)
PROBABILITY_COLUMNS = ("p_0", "p_1", "p_2", "p_5", "p_20")


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_pinned_sources() -> None:
    pin = manifest()
    for rel, digest in pin["sources"].items():
        got = sha256_file(REPO / rel)
        if got != digest:
            raise RuntimeError(f"source hash drift {rel}: {got}")
    csv_pin = pin["known_reference_csv"]
    got = sha256_file(REPO / csv_pin["path"])
    if got != csv_pin["sha256"]:
        raise RuntimeError(f"reference csv hash drift: {got}")


def expected_shapes(h: int = H, layers: int = LAYERS, k: int = K) -> dict[str, tuple[int, ...]]:
    shapes = {
        "pay.weight": (6, h),
        "mask.weight": (h, 1),
        "mask.bias": (h,),
        "typ.weight": (3, h),
        "slot.weight": (8, h),
        "tick.weight": (h, 2),
        "tick.bias": (h,),
        "head.0.weight": (h, 2 * h),
        "head.0.bias": (h,),
        "head.2.weight": (5, h),
        "head.2.bias": (5,),
    }
    for i in range(layers):
        prefix = f"blocks.{i}"
        shapes[f"{prefix}.conv.weight"] = (h, h, k)
        shapes[f"{prefix}.conv.bias"] = (h,)
        shapes[f"{prefix}.mix.weight"] = (h, 2 * h)
        shapes[f"{prefix}.mix.bias"] = (h,)
        shapes[f"{prefix}.n1.weight"] = (h,)
        shapes[f"{prefix}.n1.bias"] = (h,)
        shapes[f"{prefix}.n2.weight"] = (h,)
        shapes[f"{prefix}.n2.bias"] = (h,)
    return shapes


def check_state(state: dict) -> None:
    want = expected_shapes()
    leaked = [key for key in state if key == "sig.weight" or key.startswith("sig.")]
    if leaked:
        raise RuntimeError(f"signal tensors present: {leaked}")
    if set(state) != set(want):
        missing = sorted(set(want) - set(state))
        extra = sorted(set(state) - set(want))
        raise RuntimeError(f"architecture keys mismatch missing={missing} extra={extra}")
    for key, shape in want.items():
        got = tuple(state[key].shape)
        if got != shape:
            raise RuntimeError(f"{key} shape {got} != {shape}")
        if not state[key].dtype.is_floating_point:
            raise RuntimeError(f"{key} is not floating")
        if not bool(torch.isfinite(state[key]).all()):
            raise RuntimeError(f"{key} is not finite")


def parent_is_weights_only(blob: dict) -> bool:
    return set(blob) == {"config", "state"} and "optimizer" not in blob and "scheduler" not in blob


def continuation_seed(original_seed: int, extra_epochs: int) -> int:
    if extra_epochs not in EXTRA_EPOCHS:
        raise RuntimeError(f"extra_epochs {extra_epochs} is outside the frozen set {EXTRA_EPOCHS}")
    return 1000 + int(original_seed) + int(extra_epochs)


def canonical_renorm(values) -> list[float]:
    """Same reduction as house_edge.scoring.write_submission_csv: float() then sum."""
    vals = [float(v) for v in values]
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        raise RuntimeError("non-finite probability")
    total = sum(vals)
    if not total > 0.0:
        raise RuntimeError("probability mass is not positive")
    return [v / total for v in vals]


def renorm_predictions(preds: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: canonical_renorm(vals) for key, vals in preds.items()}


def load_fp64_model():
    assert_pinned_sources()
    name = "cnn03_continuation_canonical_fp64_model"
    if name in sys.modules:
        return sys.modules[name]
    entry = str(FP64_ENTRY)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, FP64_ENTRY / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load canonical fp64 model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.USE_SIGNAL:
        raise RuntimeError("canonical fp64 reference has USE_SIGNAL true")
    return module


def ensure_house_edge():
    src = str(REPO / "src")
    kit = str(REPO / "kit")
    for path in (src, kit):
        if path not in sys.path:
            sys.path.insert(0, path)
    from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
    return load_official_validation, score_predictions, write_submission_csv


def tune_ids() -> set[str]:
    return set(tune_contract()["tune_episodes"])


def tune_contract() -> dict:
    splits = json.loads((REPO / "runs" / "splits.json").read_text(encoding="utf-8"))
    tune = set(splits["tune"])
    confirm = set(splits["confirm"])
    if len(tune) != 48 or len(confirm) != 48 or tune & confirm:
        raise RuntimeError("tune/confirm episode contract is not 48 and 48 disjoint")
    with (REPO / "kit" / "public" / "validation" / "queries.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tune_q = {row["query_id"] for row in rows if row["episode_id"] in tune}
    confirm_q = {row["query_id"] for row in rows if row["episode_id"] in confirm}
    if len(tune_q) != 384 or len(confirm_q) != 384 or tune_q & confirm_q:
        raise RuntimeError("tune/confirm query contract is not 384 and 384 disjoint")
    if len(rows) != 768:
        raise RuntimeError(f"validation queries are {len(rows)}, want 768")
    return {
        "tune_episodes": tune,
        "confirm_episodes": confirm,
        "tune_query_ids": tune_q,
        "confirm_query_ids": confirm_q,
    }


def score_tune_detail(preds: dict[str, list[float]]) -> dict:
    """Score exactly the 384 tune queries. Confirm and test ids are refused as the scored set."""
    contract = tune_contract()
    tune_q = contract["tune_query_ids"]
    confirm_q = contract["confirm_query_ids"]
    errors = []
    if not tune_q <= set(preds):
        errors.append("missing tune query ids")
    unknown = set(preds) - tune_q - confirm_q
    if unknown:
        errors.append("non-tune query id")
    if errors:
        raise RuntimeError("score refused: " + "; ".join(errors))
    load_official_validation, score_predictions, _ = ensure_house_edge()
    official = load_official_validation(REPO / "kit" / "public" / "validation")
    targets, _groups, _reference, query_eps = official
    official_tune = {row.query_id for row in targets if query_eps[row.query_id] in contract["tune_episodes"]}
    if official_tune != tune_q:
        raise RuntimeError("tune query ids disagree with official validation targets")
    scored_preds = {key: preds[key] for key in tune_q}
    scored = score_predictions(renorm_predictions(scored_preds), official, contract["tune_episodes"])
    if set(scored) & {"confirm", "full", "test"}:
        raise RuntimeError("study scorer returned a non-tune bundle")
    return {
        "tune_R": float(scored["R"]),
        "n_tune_queries": len(scored_preds),
        "n_tune_episodes": len(contract["tune_episodes"]),
        "n_forward_queries": len(preds),
    }


def score_tune(preds: dict[str, list[float]]) -> float:
    return score_tune_detail(preds)["tune_R"]


def post_run() -> dict | None:
    if not POST_RUN_PATH.is_file():
        return None
    data = json.loads(POST_RUN_PATH.read_text(encoding="utf-8"))
    if data.get("not_a_pretraining_pin") is not True:
        raise RuntimeError("POST_RUN_HASHES.json is not marked as a post-training pin")
    return data


def _curve_config_errors(cfg: dict, arm: str) -> list[str]:
    spec = manifest()["checkpoints"][arm]
    errors = []
    if int(cfg.get("seed")) != int(spec["seed"]):
        errors.append("seed mismatch")
    if abs(float(cfg.get("lr")) - float(spec["lr"])) > 1e-12:
        errors.append("lr mismatch")
    for key, want in (("h", H), ("layers", LAYERS), ("W", W), ("bs", BS), ("epochs", 30)):
        if cfg.get(key) != want:
            errors.append(f"config {key} mismatch")
    if abs(float(cfg.get("drop", -1)) - DROP) > 1e-12:
        errors.append("config drop mismatch")
    if abs(float(cfg.get("wd", -1)) - WD) > 1e-12:
        errors.append("config wd mismatch")
    if abs(float(cfg.get("pct_start", -1)) - 0.3) > 1e-12:
        errors.append("config pct_start mismatch")
    return errors


def parent_bind_errors(path: Path, blob: dict, arm: str, digest: str) -> list[str]:
    spec = manifest()["checkpoints"][arm]
    errors = []
    if digest != spec["sha256"]:
        errors.append("hash mismatch")
    pinned = (REPO / spec["path"]).resolve()
    if path.resolve() != pinned and digest == spec["sha256"]:
        errors.append("hash matches a different path")
    if not parent_is_weights_only(blob):
        errors.append("parent is not weights-only")
    errors.extend(_curve_config_errors(blob.get("config") or {}, arm))
    return errors


def incumbent_bind_errors(path: Path, blob: dict, digest: str) -> list[str]:
    pin = manifest()
    errors = []
    pinned = (REPO / pin["incumbent_checkpoint"]).resolve()
    if path.resolve() != pinned:
        errors.append("incumbent path is not the manifest path")
    record = post_run()
    if record is None:
        errors.append("incumbent byte pin missing")
    else:
        want = record.get("incumbent_checkpoint", {}).get("sha256")
        if digest != want:
            errors.append("hash mismatch")
    cfg = blob.get("config") or {}
    if cfg.get("model") != "cnn" or cfg.get("phase") != "final" or cfg.get("kernel") != 3:
        errors.append("incumbent config mismatch")
    if cfg.get("h") != H or cfg.get("layers") != LAYERS or cfg.get("W") != W or cfg.get("seed") != 0:
        errors.append("incumbent config mismatch")
    if abs(float(cfg.get("lr", -1)) - 0.00197) > 1e-12 or abs(float(cfg.get("drop", -1)) - DROP) > 1e-12:
        errors.append("incumbent config mismatch")
    if not parent_is_weights_only(blob):
        errors.append("incumbent is not weights-only")
    return errors


def continuation_bind_errors(blob: dict, arm: str, digest: str) -> list[str]:
    cfg = blob.get("config") or {}
    errors = []
    if cfg.get("arm") != arm:
        errors.append("arm mismatch")
    if arm not in manifest()["checkpoints"]:
        errors.append("unknown arm")
        return errors
    spec = manifest()["checkpoints"][arm]
    if int(cfg.get("seed", -1)) != int(spec["seed"]):
        errors.append("seed mismatch")
    if abs(float(cfg.get("lr", -1)) - float(spec["lr"])) > 1e-12:
        errors.append("lr mismatch")
    if abs(float(cfg.get("wd", -1)) - WD) > 1e-12 or cfg.get("bs") != BS:
        errors.append("config wd/bs mismatch")
    if abs(float(cfg.get("pct_start", -1)) - 0.3) > 1e-12:
        errors.append("config pct_start mismatch")
    extra = cfg.get("extra_epochs")
    if extra not in EXTRA_EPOCHS or blob.get("completed_extra_epochs") != extra:
        errors.append("extra_epochs budget")
    else:
        if cfg.get("total_steps") != extra * STEPS_PER_EPOCH:
            errors.append("step budget")
        if blob.get("continuation_seed") != continuation_seed(int(manifest()["checkpoints"][arm]["seed"]), extra):
            errors.append("continuation seed mismatch")
        record = post_run()
        name = f"{arm}_plus{extra}"
        if record is None:
            errors.append("continuation byte pin missing")
        else:
            want = record.get("continuations", {}).get(name, {}).get("sha256")
            if digest != want:
                errors.append("hash mismatch")
    parent = blob.get("parent_checkpoint_sha256")
    if parent != manifest()["checkpoints"][arm]["sha256"]:
        errors.append("parent hash mismatch")
    if blob.get("parent_had_optimizer") is not False or blob.get("warm_start_from") != "weights_only":
        errors.append("warm-start disclosure mismatch")
    return errors


def read_prediction_csv(path: Path) -> dict[str, list[float]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("query_id", *PROBABILITY_COLUMNS):
            raise RuntimeError(f"unexpected columns in {path}")
        for row in reader:
            out[row["query_id"]] = [float(row[col]) for col in PROBABILITY_COLUMNS]
    return out


def load_curve_blob(path: Path):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "state" not in blob:
        raise RuntimeError(f"{path} has no state")
    return blob


def assert_inside_study(path: Path) -> Path:
    resolved = path.resolve()
    if STUDY.resolve() not in resolved.parents and resolved != STUDY.resolve():
        raise RuntimeError(f"refusing to write outside the study dir: {resolved}")
    return resolved
