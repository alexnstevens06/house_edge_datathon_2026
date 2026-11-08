"""Pins and CPU float64 helpers for the ensemble/blend stream.

Reads canonical files. Writes nothing. Does not set HSA_OVERRIDE and does not
enable autocast. Production PINNED.json is not imported.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

torch.set_num_threads(2)

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
MANIFEST_PATH = STUDY / "MANIFEST.json"
SPLIT_PATH = STUDY / "SPLIT_IDS.json"
CANDIDATE_UNPACKED = (
    REPO / "experiments/cnn03_convergence/candidate_lr003_s2_20261108/unpacked"
)
H, LAYERS, K, DROP, W = 128, 4, 3, 0.3, 64
BLEND_WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2)
ENSEMBLES = {
    "ensemble_equal_s0_s1_s2": ("curve_C_lr003_s0", "curve_C_lr003_s1", "curve_C_lr003_s2"),
    "ensemble_equal_s0_s2": ("curve_C_lr003_s0", "curve_C_lr003_s2"),
}
PUBLISHED_TUNE_R = {
    "curve_C_lr003_s0": 0.3101981645338761,
    "curve_C_lr003_s1": 0.3452074168357309,
    "curve_C_lr003_s2": 0.30980333573689856,
    "hard_shared_control": 0.46211193532631023,
}
CURRENT_BEST_TUNE_R = 0.30980333573689856
IMPROVEMENT_THRESHOLD = 1e-6
REPRODUCTION_ATOL = 1e-12
CURVE_CONFIG_KEYS = (
    "lr", "wd", "bs", "epochs", "drop", "h", "layers", "W", "seed", "pct_start", "eval_every", "out",
)
HARD_BLOB_KEYS = ("config", "losses", "phase", "seed", "state", "steps")
HARD_CONFIG_KEYS = ("lr", "wd", "bs", "micro_bs", "epochs")


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    leaked = [key for key in state if key == "sig.weight" or str(key).startswith("sig.")]
    if leaked:
        raise RuntimeError(f"signal tensors present: {leaked}")
    if set(state) != set(want):
        missing = sorted(set(want) - set(state))
        extra = sorted(set(state) - set(want))
        raise RuntimeError(f"architecture keys mismatch missing={missing} extra={extra}")
    for key, shape in want.items():
        tensor = state[key]
        got = tuple(tensor.shape)
        if got != shape:
            raise RuntimeError(f"{key} shape {got} != {shape}")
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"{key} saved dtype {tensor.dtype} != float32")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{key} is not finite")


def _close(got, want) -> bool:
    return abs(float(got) - float(want)) <= 1e-12


def bind_errors(path: Path, blob: dict, role: str, digest: str) -> list[str]:
    pin = manifest()["checkpoints"]
    errors: list[str] = []
    if role not in pin:
        return [f"unknown role {role}"]
    spec = pin[role]
    if digest != spec["sha256"]:
        errors.append("hash mismatch")
    pinned = (STUDY / spec["path"]).resolve()
    if path.resolve() != pinned:
        errors.append("path is not the pinned study input")
    if not isinstance(blob, dict) or "state" not in blob or "config" not in blob:
        errors.append("missing state or config")
        return errors
    cfg = blob["config"]
    if not isinstance(cfg, dict):
        errors.append("config is not a dict")
        return errors
    if role == "hard_shared_control":
        if set(blob) != set(HARD_BLOB_KEYS):
            errors.append("hard control keys mismatch")
        if set(cfg) != set(HARD_CONFIG_KEYS):
            errors.append("hard control config keys mismatch")
        if blob.get("phase") != "control" or int(blob.get("seed", -1)) != 0 or int(blob.get("steps", -1)) != 10:
            errors.append("hard control phase/seed/steps mismatch")
        losses = blob.get("losses")
        if not isinstance(losses, list) or len(losses) != 10 or any(v != v or v in (float("inf"), float("-inf")) for v in losses):
            errors.append("hard control losses mismatch")
        if not _close(cfg.get("lr", -1), 0.00197) or not _close(cfg.get("wd", -1), 0.00895):
            errors.append("hard control lr/wd mismatch")
        if cfg.get("bs") != 512 or cfg.get("micro_bs") != 128 or cfg.get("epochs") != 10:
            errors.append("hard control schedule mismatch")
    else:
        if set(blob) != {"config", "state"}:
            errors.append("curve checkpoint is not weights-only")
        if set(cfg) != set(CURVE_CONFIG_KEYS):
            errors.append("curve config keys mismatch")
        if int(cfg.get("seed", -1)) != int(spec["seed"]):
            errors.append("seed mismatch")
        if not _close(cfg.get("lr", -1), spec["lr"]):
            errors.append("lr mismatch")
        for key, want in (("h", H), ("layers", LAYERS), ("W", W), ("bs", 512), ("epochs", 30)):
            if cfg.get(key) != want:
                errors.append(f"config {key} mismatch")
        if not _close(cfg.get("drop", -1), DROP) or not _close(cfg.get("wd", -1), 0.00895):
            errors.append("config drop/wd mismatch")
        if not _close(cfg.get("pct_start", -1), 0.3) or cfg.get("eval_every") != 2:
            errors.append("config pct_start/eval_every mismatch")
        if cfg.get("out") != spec["original_out"]:
            errors.append("config out mismatch")
    try:
        check_state(blob["state"])
    except RuntimeError as exc:
        errors.append(str(exc))
    return errors


def load_blob(path: Path) -> dict:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise RuntimeError(f"{path} is not a checkpoint dict")
    return blob


def require_bind(path: Path, role: str) -> dict:
    path = Path(path)
    digest = sha256_file(path)
    blob = load_blob(path)
    errors = bind_errors(path, blob, role, digest)
    if errors:
        raise RuntimeError("checkpoint refused: " + "; ".join(errors))
    return blob


def assert_pinned_files() -> None:
    pin = manifest()
    for rel, digest in pin["files"].items():
        path = REPO / rel if not rel.startswith("experiments/cnn03_convergence/ensemble_blend") else REPO / rel
        got = sha256_file(path)
        if got != digest:
            raise RuntimeError(f"pinned file drift {rel}: {got}")
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise RuntimeError("HSA_OVERRIDE_GFX_VERSION is set")


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    unpacked = str(CANDIDATE_UNPACKED)
    sys.path.insert(0, unpacked)
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == unpacked:
            sys.path.pop(0)
    return module


def load_canonical_model():
    assert_pinned_files()
    module = _load_module("ensemble_blend_canonical_fp64_model", CANDIDATE_UNPACKED / "model.py")
    if module.USE_SIGNAL:
        raise RuntimeError("canonical module USE_SIGNAL is true")
    return module


def load_canonical_data():
    assert_pinned_files()
    return _load_module("ensemble_blend_canonical_data", CANDIDATE_UNPACKED / "data.py")


def canonical_renorm(values) -> list[float]:
    vals = [float(v) for v in values]
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        raise RuntimeError("non-finite probability")
    total = sum(vals)
    if not total > 0.0:
        raise RuntimeError("probability mass is not positive")
    return [v / total for v in vals]


def renorm_map(preds: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: canonical_renorm(row) for key, row in preds.items()}


def tune_contract() -> dict:
    splits = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    upstream = json.loads((STUDY / "upstream" / "splits.json").read_text(encoding="utf-8"))
    tune = list(splits["tune"])
    confirm = list(splits["confirm"])
    if tune != list(upstream["tune"]) or confirm != list(upstream["confirm"]):
        raise RuntimeError("SPLIT_IDS.json does not match the upstream split snapshot")
    tune_set, confirm_set = set(tune), set(confirm)
    if len(tune) != 48 or len(confirm) != 48 or len(tune_set) != 48 or len(confirm_set) != 48:
        raise RuntimeError("tune/confirm episode contract is not 48 and 48")
    if tune_set & confirm_set:
        raise RuntimeError("tune and confirm episodes overlap")
    with (REPO / "kit" / "public" / "validation" / "queries.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 768:
        raise RuntimeError(f"validation queries are {len(rows)}, want 768")
    tune_q = [row["query_id"] for row in rows if row["episode_id"] in tune_set]
    confirm_q = [row["query_id"] for row in rows if row["episode_id"] in confirm_set]
    if len(tune_q) != 384 or len(set(tune_q)) != 384 or len(confirm_q) != 384 or len(set(confirm_q)) != 384:
        raise RuntimeError("tune/confirm query contract is not 384 and 384")
    if set(tune_q) & set(confirm_q):
        raise RuntimeError("tune and confirm queries overlap")
    return {
        "tune_episodes": tune_set,
        "confirm_episodes": confirm_set,
        "tune_query_ids": set(tune_q),
        "confirm_query_ids": set(confirm_q),
    }


def score_tune_detail(preds: dict[str, list[float]]) -> dict:
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
    src = str(REPO / "src")
    kit = str(REPO / "kit")
    for path in (src, kit):
        if path not in sys.path:
            sys.path.insert(0, path)
    from house_edge.scoring import load_official_validation, score_predictions

    official = load_official_validation(REPO / "kit" / "public" / "validation")
    targets, _groups, _reference, query_eps = official
    official_tune = {row.query_id for row in targets if query_eps[row.query_id] in contract["tune_episodes"]}
    if official_tune != tune_q:
        raise RuntimeError("tune query ids disagree with validation targets")
    scored_preds = {key: preds[key] for key in tune_q}
    scored = score_predictions(renorm_map(scored_preds), official, contract["tune_episodes"])
    if set(scored) & {"confirm", "full", "test"}:
        raise RuntimeError("scorer returned a non-tune bundle")
    if scored.get("R") != scored.get("R"):
        raise RuntimeError("tune R is not finite")
    return {
        "tune_R": float(scored["R"]),
        "n_tune_queries": len(scored_preds),
        "n_tune_episodes": len(contract["tune_episodes"]),
        "n_forward_queries": len(preds),
    }


def forward_probabilities(state: dict, *, max_episodes: int | None = None) -> dict[str, list[float]]:
    """Canonical module, float64 parameters, float64 softmax. No autocast."""
    check_state(state)
    model_mod = load_canonical_model()
    data_mod = load_canonical_data()
    model_mod.USE_SIGNAL = False
    net = model_mod.CNN(H, LAYERS, K, DROP, W, use_signal=False)
    net.load_state_dict(state)
    net.double()
    net.eval()
    if net.use_signal or net.sig is not None:
        raise RuntimeError("signal mode refused")
    floating = [p for p in net.parameters() if p.is_floating_point()]
    if not floating or any(p.dtype != torch.float64 for p in floating):
        raise RuntimeError("parameters are not float64")
    validation = data_mod.load_public_split(REPO / "kit" / "public", "validation")
    if max_episodes is not None:
        keep = {ep.episode_id for ep in validation.episodes[:max_episodes]}
        validation.episodes = validation.episodes[:max_episodes]
        validation.queries = [row for row in validation.queries if row["episode_id"] in keep]
    query, episodes = model_mod.build_query(validation, W)
    probs = model_mod.predict(net, query, "cpu", bs=4)
    if str(probs.dtype) != "float64":
        raise RuntimeError(f"softmax dtype {probs.dtype}")
    preds = model_mod.to_preds(probs, episodes, validation)
    if len(preds) != len(validation.queries):
        raise RuntimeError("prediction map does not cover the loaded queries")
    return renorm_map(preds)


def equal_probability_mean(maps: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    if len(maps) < 2:
        raise RuntimeError("an ensemble needs at least two maps")
    keys = set(maps[0])
    for other in maps[1:]:
        if set(other) != keys:
            raise RuntimeError("ensemble query ids differ")
    count = float(len(maps))
    out = {}
    for key in keys:
        mixed = [sum(item[key][i] for item in maps) / count for i in range(5)]
        out[key] = canonical_renorm(mixed)
    return out


def blend_maps(base: dict[str, list[float]], other: dict[str, list[float]], weight: float) -> dict[str, list[float]]:
    if weight not in BLEND_WEIGHTS:
        raise RuntimeError(f"weight {weight} is outside the frozen set")
    if set(base) != set(other):
        raise RuntimeError("blend query ids differ")
    if weight == 0.0:
        return {key: list(row) for key, row in base.items()}
    out = {}
    for key, row in base.items():
        mixed = [(1.0 - weight) * float(row[i]) + weight * float(other[key][i]) for i in range(5)]
        out[key] = canonical_renorm(mixed)
    return out


def simplex_errors(preds: dict[str, list[float]]) -> list[str]:
    errors = []
    for key, row in preds.items():
        if len(row) != 5:
            errors.append(f"{key} width")
            continue
        if any(v != v or v in (float("inf"), float("-inf")) or v < 0.0 for v in row):
            errors.append(f"{key} not a finite simplex")
            continue
        if abs(sum(row) - 1.0) > 1e-12:
            errors.append(f"{key} sum {sum(row)}")
    return errors


def reproduction_ok(measured: float, published: float) -> bool:
    return abs(float(measured) - float(published)) <= REPRODUCTION_ATOL
