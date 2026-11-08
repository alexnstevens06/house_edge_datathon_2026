"""Matched-step utility runner for synthetic machine families.

RESEARCH_ONLY. Eligibility unresolved. No submission export.
Fits the copied no-signal CNN on real TRAIN windows and, for each available
family, on batches that replace 25% or 50% of those windows with synthetic
ones. Scores public validation tune episodes only.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import base_cnn as B

B.USE_SIGNAL = False

from house_edge.data import (  # noqa: E402
    MISSING,
    PAYOUT_TO_IDX,
    TYPE_TO_IDX,
    subset_split_data,
)
from house_edge.scoring import load_official_validation, score_predictions  # noqa: E402

CHICAGO = ZoneInfo("America/Chicago")
ELIGIBILITY = "RESEARCH_ONLY"
ELIGIBILITY_STATUS = "unresolved"
SYNTH_INDEX_SEED_OFFSET = 9176
GENERATE_ATTEMPTS = ((8, 128, 8), (8, 96, 8))
PAYOUTS = (0, 1, 2, 5, 20)


def research_doc(payload: dict) -> dict:
    out = dict(payload)
    out["eligibility"] = ELIGIBILITY
    out["eligibility_status"] = ELIGIBILITY_STATUS
    out["submission_export"] = False
    out["new_information_claim"] = False
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exc_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def parse_deadline(text: str) -> datetime:
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    stamp = datetime.fromisoformat(raw)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=CHICAGO)
    return stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-only synthetic mix runner (tune split only).")
    parser.add_argument("--deadline", default="2026-11-07T19:40:00")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--families-dir", default=str(ROOT.parent / "grok_parallel"))
    parser.add_argument("--research", default=str(ROOT / "research"))
    parser.add_argument("--plan", default=str(ROOT / "PLAN.json"))
    parser.add_argument("--max-families", type=int, default=12)
    parser.add_argument("--screen-steps", type=int, default=60)
    parser.add_argument("--promote-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--min-t", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--generate-timeout", type=float, default=90.0)
    return parser


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
        raise ValueError(f"observed int must be 0 or 1, got {value!r}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes"}:
            return True
        if text in {"0", "false", "no"}:
            return False
    raise ValueError(f"bad observed value {value!r}")


def as_machine_id(value) -> int:
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if isinstance(value, (int, np.integer)) and 0 <= int(value) <= 7:
        return int(value)
    raise ValueError(f"machine_id must be slot 0..7, got {value!r}")


def payout_index(record: dict) -> int:
    if "payout" in record and record["payout"] is not None:
        payout = int(record["payout"])
        if payout not in PAYOUT_TO_IDX:
            raise ValueError(f"observed payout {payout} is outside {PAYOUTS}")
        return PAYOUT_TO_IDX[payout]
    if record.get("payout_index") is not None:
        index = int(record["payout_index"])
        if index not in range(5):
            raise ValueError(f"payout_index {index} is outside 0..4")
        return index
    raise ValueError("observed row has no payout")


def _probs_ok(values) -> bool:
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        return False
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return False
    if any(v < -1e-8 for v in nums):
        return False
    return abs(sum(nums) - 1.0) <= 1e-4


def ingest_panel(records, truth) -> tuple[list[dict], dict]:
    """Build per-episode obs matrices. observed false is MISSING, never payout 0."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    if not isinstance(truth, list) or not truth:
        raise ValueError("truth must be a non-empty list")
    grouped: dict = {}
    seen = set()
    ignored_latent = 0
    n_observed = 0
    for row_i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"record {row_i} is not an object")
        for key in ("episode_id", "tick", "machine_id", "machine_type", "observed"):
            if key not in rec:
                raise ValueError(f"record {row_i} missing {key}")
        episode_id = rec["episode_id"]
        tick = int(rec["tick"])
        if tick < 0 or tick > 10000:
            raise ValueError(f"tick {tick} outside 0..10000")
        slot = as_machine_id(rec["machine_id"])
        observed = as_bool(rec["observed"])
        machine_type = rec["machine_type"]
        if machine_type not in TYPE_TO_IDX:
            raise ValueError(f"machine_type {machine_type!r} is not one of {tuple(TYPE_TO_IDX)}")
        key = (episode_id, tick, slot)
        if key in seen:
            raise ValueError(f"duplicate record at episode={episode_id} tick={tick} machine_id={slot}")
        seen.add(key)
        episode = grouped.setdefault(episode_id, {"types": [None] * 8, "cells": {}})
        previous = episode["types"][slot]
        if previous is None:
            episode["types"][slot] = TYPE_TO_IDX[machine_type]
        elif previous != TYPE_TO_IDX[machine_type]:
            raise ValueError(f"machine type changed inside episode {episode_id} slot {slot}")
        if observed:
            index = payout_index(rec)
            n_observed += 1
            if "observation_index" in rec and rec["observation_index"] is not None:
                if int(rec["observation_index"]) != index:
                    raise ValueError(
                        f"observation_index disagrees with payout at episode={episode_id} tick={tick} slot={slot}"
                    )
        else:
            index = MISSING
            if rec.get("payout") not in (None,):
                ignored_latent += 1
            if "observation_index" in rec and rec["observation_index"] is not None:
                if int(rec["observation_index"]) != MISSING:
                    raise ValueError(
                        f"unobserved observation_index must be -1 at episode={episode_id} tick={tick} slot={slot}"
                    )
        episode["cells"][(tick, slot)] = index

    truth_keys = set()
    for row_i, rec in enumerate(truth):
        if not isinstance(rec, dict):
            raise ValueError(f"truth {row_i} is not an object")
        for key in ("episode_id", "tick", "machine_id", "probabilities"):
            if key not in rec:
                raise ValueError(f"truth {row_i} missing {key}")
        slot = as_machine_id(rec["machine_id"])
        tick = int(rec["tick"])
        if not _probs_ok(rec["probabilities"]):
            raise ValueError(
                f"truth probability simplex failed at episode={rec['episode_id']} tick={tick} machine_id={slot}"
            )
        key = (rec["episode_id"], tick, slot)
        if key in truth_keys:
            raise ValueError(f"duplicate truth at episode={rec['episode_id']} tick={tick} machine_id={slot}")
        truth_keys.add(key)
    if truth_keys != seen:
        raise ValueError(
            f"truth keys and record keys differ (records={len(seen)} truth={len(truth_keys)} "
            f"only_records={len(seen - truth_keys)} only_truth={len(truth_keys - seen)})"
        )

    episodes = []
    payout_counts = [0, 0, 0, 0, 0]
    zero_observed = 0
    for episode_id, episode in grouped.items():
        missing_slots = [slot for slot, type_id in enumerate(episode["types"]) if type_id is None]
        if missing_slots:
            raise ValueError(f"episode {episode_id} missing machine_id {missing_slots}; all 8 machines are required")
        max_tick = max(tick for tick, _slot in episode["cells"])
        obs = np.full((max_tick + 1, 8), MISSING, dtype=np.int8)
        for (tick, slot), index in episode["cells"].items():
            obs[tick, slot] = index
            if index == 0:
                zero_observed += 1
            if index >= 0:
                payout_counts[index] += 1
        episodes.append(
            {
                "episode_id": str(episode_id),
                "types": np.array(episode["types"], dtype=np.int64),
                "obs": obs,
            }
        )
    report = {
        "n_records": len(records),
        "n_truth": len(truth),
        "n_episodes": len(episodes),
        "n_observed": n_observed,
        "n_missing": len(records) - n_observed,
        "payout_index_counts_observed": payout_counts,
        "zero_payout_observed_count": zero_observed,
        "unobserved_numeric_payouts_ignored": ignored_latent,
        "missing_code": MISSING,
        "payout_to_index": {str(k): v for k, v in PAYOUT_TO_IDX.items()},
        "type_to_index": dict(TYPE_TO_IDX),
        "machine_id_mapping": "slot 0..7, not re-sorted",
    }
    return episodes, report


def stack_windows(episodes, W: int, min_t: int):
    xs, ts, tys, ys = [], [], [], []
    for episode in episodes:
        obs = episode["obs"]
        types = episode["types"]
        limit = obs.shape[0]
        for tick in range(min_t, limit):
            target = obs[tick]
            if int((target >= 0).sum()) == 0:
                continue
            xs.append(B.window(obs, tick, W))
            ts.append(tick)
            tys.append(types)
            ys.append(target.copy())
    if not xs:
        raise ValueError(f"no labelled windows with min_t={min_t}")
    pack = (
        torch.tensor(np.stack(xs)),
        torch.tensor(np.asarray(ts, dtype=np.int64)),
        torch.tensor(np.stack(tys).astype(np.int64)),
        torch.tensor(np.stack(ys)),
    )
    return pack


def check_window_alignment(episode: dict, W: int, min_t: int) -> dict:
    obs = episode["obs"]
    if obs.shape[0] <= min_t:
        raise RuntimeError(f"episode too short for causal check min_t={min_t} length={obs.shape[0]}")
    tick = min_t
    base = B.window(obs, tick, W)
    future = obs.copy()
    future[tick:, :] = 0
    if not np.array_equal(base, B.window(future, tick, W)):
        raise RuntimeError("window changed when ticks >= t were set to payout 0")
    past = obs.copy()
    past[tick - 1, 0] = 1 if past[tick - 1, 0] != 1 else 3
    if np.array_equal(base, B.window(past, tick, W)):
        raise RuntimeError("window ignored a change at tick t-1")
    target = obs[tick]
    if int((base == 0).sum()) and int((obs[:tick] == 0).sum()) == 0:
        raise RuntimeError("window contains payout 0 that is not in ticks < t")
    return {
        "tick": int(tick),
        "target": target.tolist(),
        "missing_in_window": int((base < 0).sum()),
        "zero_payout_in_window": int((base == 0).sum()),
        "future_mutation_ignored": True,
        "past_mutation_visible": True,
        "target_matches_obs_t": True,
    }


def counts_from_targets(types: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    type_np = types.detach().cpu().numpy().astype(np.int64)
    target_np = targets.detach().cpu().numpy().astype(np.int64)
    counts = np.zeros((3, 5), dtype=np.int64)
    mask = target_np >= 0
    np.add.at(counts, (type_np[mask], target_np[mask]), 1)
    return counts


def calibration_weights(train_counts: np.ndarray, synth_counts: np.ndarray, alpha: float = 0.5,
                        clip=(0.25, 4.0)) -> np.ndarray:
    """Train-only type-conditional prior weights. Does not read validation."""
    weights = np.ones((3, 5), dtype=np.float64)
    for type_id in range(3):
        if int(synth_counts[type_id].sum()) == 0:
            continue
        train_p = train_counts[type_id].astype(np.float64) + alpha
        synth_p = synth_counts[type_id].astype(np.float64) + alpha
        train_p = train_p / train_p.sum()
        synth_p = synth_p / synth_p.sum()
        raw = np.clip(train_p / synth_p, clip[0], clip[1])
        raw = raw / np.sum(synth_p * raw)
        weights[type_id] = raw
    return weights


def slot_weights(types: torch.Tensor, targets: torch.Tensor, table: np.ndarray) -> torch.Tensor:
    type_np = types.detach().cpu().numpy().astype(np.int64)
    target_np = targets.detach().cpu().numpy().astype(np.int64)
    out = np.ones(type_np.shape, dtype=np.float32)
    mask = target_np >= 0
    out[mask] = table[type_np[mask], target_np[mask]].astype(np.float32)
    return torch.tensor(out)


def index_plan(n_pool: int, steps: int, width: int, seed: int) -> np.ndarray:
    if width == 0:
        return np.zeros((steps, 0), dtype=np.int64)
    if n_pool <= 0:
        raise ValueError("cannot draw batch indexes from an empty pool")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_pool, size=(steps, width), dtype=np.int64)


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        array = state[key].detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def draw_init(seed: int, h: int, layers: int, drop: float, W: int):
    torch.manual_seed(seed)
    model = B.CNN(h, layers, 3, drop, W, use_signal=False)
    if model.sig is not None or model.use_signal:
        raise RuntimeError("CNN init was constructed with a signal pathway")
    return copy.deepcopy(model.state_dict())


def fresh_model(init_state, seed: int, h: int, layers: int, drop: float, W: int):
    torch.manual_seed(seed)
    model = B.CNN(h, layers, 3, drop, W, use_signal=False)
    model.load_state_dict(copy.deepcopy(init_state))
    model.train()
    return model


def mix_sizes(mix: float, batch_size: int) -> tuple[int, int]:
    n_synth = int(round(mix * batch_size))
    if mix > 0 and n_synth <= 0:
        n_synth = 1
    if n_synth > batch_size:
        n_synth = batch_size
    return batch_size - n_synth, n_synth


def weighted_masked_loss(logits, targets, weights):
    labels = targets.long()
    mask = labels >= 0
    if int(mask.sum()) == 0:
        raise RuntimeError("batch has no observed targets")
    token = F.cross_entropy(logits.float()[mask], labels[mask], reduction="none")
    chosen = weights[mask].float()
    return (token * chosen).sum() / chosen.sum()


def descriptive_calibration(probs: np.ndarray, labels: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    chosen = probs[np.arange(len(labels)), labels]
    nll = float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, 6)
    ece = 0.0
    bins = []
    for i in range(5):
        lo, hi = float(edges[i]), float(edges[i + 1])
        selected = (confidence >= lo) & (confidence <= hi) if i == 0 else (confidence > lo) & (confidence <= hi)
        count = int(selected.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0})
            continue
        accuracy = float(correct[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        ece += (count / len(labels)) * abs(accuracy - mean_confidence)
        bins.append({"lo": lo, "hi": hi, "n": count, "acc": accuracy, "confidence": mean_confidence})
    return {
        "nll": nll,
        "ece_5bin": float(ece),
        "n": int(len(labels)),
        "bins": bins,
        "note": "Descriptive only. Not a confidence interval or a significance test.",
    }


def assert_tune_only(scored_episodes, tune_ids, confirm_ids) -> None:
    scored = set(scored_episodes)
    if scored & set(confirm_ids):
        raise RuntimeError("refusing to score confirm episodes")
    extra = scored - set(tune_ids)
    if extra:
        raise RuntimeError(f"scored episodes outside tune: {sorted(extra)[:5]}")


def fit_steps(model, optimizer, real_pack, real_weight, synth_pack, synth_weight, real_idx, synth_idx,
              n_real, deadline: datetime):
    losses = []
    interrupted = False
    rx, rt, rty, ry = real_pack
    for step in range(real_idx.shape[0]):
        if datetime.now(CHICAGO) >= deadline:
            interrupted = True
            break
        rows = torch.from_numpy(np.ascontiguousarray(real_idx[step, :n_real], dtype=np.int64))
        xs = [rx.index_select(0, rows)]
        ts = [rt.index_select(0, rows)]
        tys = [rty.index_select(0, rows)]
        ys = [ry.index_select(0, rows)]
        ws = [real_weight.index_select(0, rows)]
        if synth_pack is not None and synth_idx.shape[1]:
            srows = torch.from_numpy(np.ascontiguousarray(synth_idx[step], dtype=np.int64))
            sx, st, sty, sy = synth_pack
            xs.append(sx.index_select(0, srows))
            ts.append(st.index_select(0, srows))
            tys.append(sty.index_select(0, srows))
            ys.append(sy.index_select(0, srows))
            ws.append(synth_weight.index_select(0, srows))
        x = torch.cat(xs, 0)
        t = torch.cat(ts, 0)
        ty = torch.cat(tys, 0)
        y = torch.cat(ys, 0)
        w = torch.cat(ws, 0)
        signal = torch.zeros(x.shape[0], 8, dtype=torch.long)
        logits = model(x, t, ty, signal)
        loss = weighted_masked_loss(logits, y, w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses, interrupted


def read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_generate(gen_path: Path, out_dir: Path, seed: int, n_episodes: int, n_ticks: int, n_machines: int,
                 timeout: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(HERE / "call_generate.py"),
        str(gen_path),
        str(out_dir),
        str(seed),
        str(n_episodes),
        str(n_ticks),
        str(n_machines),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    try:
        completed = subprocess.run(
            command,
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"generate exceeded {timeout}s") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        last = detail[-2000:]
        raise RuntimeError(last or f"generate exited {completed.returncode}")


def file_sha_or_missing(path: Path):
    if not path.exists():
        return None
    return sha256_file(path)


def load_real(args, W: int, min_t: int):
    if torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP is visible after hiding devices; refusing to train")
    train = B.load_split(os.path.join(B.PUB, "train"), with_labels=True)
    validation = B.load_split(os.path.join(B.PUB, "validation"), with_labels=False)
    splits = json.loads(Path(B.REPO, "runs", "splits.json").read_text(encoding="utf-8"))
    tune_ids = set(splits["tune"])
    confirm_ids = set(splits["confirm"])
    if tune_ids & confirm_ids:
        raise RuntimeError("splits.json tune and confirm overlap")
    tune_data = subset_split_data(validation, tune_ids)
    windows = B.build_train(train, W, min_t=min_t, stride=1)
    x, t, ty, signal, y = windows
    if signal.shape[0] != x.shape[0]:
        raise RuntimeError("signal tensor was built; no-signal path should still return zeros without reading signal")
    if B.USE_SIGNAL:
        raise RuntimeError("USE_SIGNAL became true while loading train")
    counts = counts_from_targets(ty, y)
    ones = torch.ones(y.shape, dtype=torch.float32)
    official = load_official_validation(os.path.join(B.PUB, "validation"))
    queries = official[3]
    tune_query_episodes = {qid: episode for qid, episode in queries.items() if episode in tune_ids}
    confirm_query_ids = {qid for qid, episode in queries.items() if episode in confirm_ids}
    query_pack, episodes = B.build_query(tune_data, W)
    real_files = []
    for relative in (
        "kit/public/train/history.csv",
        "kit/public/train/queries.csv",
        "kit/public/train/labels.csv",
        "kit/public/validation/queries.csv",
        "kit/public/validation/history.csv",
        "kit/public/validation/targets.csv",
        "kit/public/validation/reference.csv",
        "kit/public/validation/conditions.csv",
        "runs/splits.json",
    ):
        path = Path(B.REPO) / relative
        real_files.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "pack": (x, t, ty, y),
        "weights": ones,
        "counts": counts,
        "tune_data": tune_data,
        "query_pack": query_pack,
        "episodes": episodes,
        "official": official,
        "tune_ids": tune_ids,
        "confirm_ids": confirm_ids,
        "tune_query_episodes": tune_query_episodes,
        "confirm_query_ids": confirm_query_ids,
        "real_files": real_files,
        "n_train_episodes": len(train.episodes),
        "n_tune_episodes": len(tune_data.episodes),
    }


def evaluate_tune(model, real, W: int) -> dict:
    model.eval()
    probabilities = B.predict(model, real["query_pack"], "cpu", bs=64)
    predictions = B.to_preds(probabilities, real["episodes"], real["tune_data"])
    scored_episodes = {real["tune_query_episodes"][qid] for qid in predictions}
    assert_tune_only(scored_episodes, real["tune_ids"], real["confirm_ids"])
    if any(qid in real["confirm_query_ids"] for qid in predictions):
        raise RuntimeError("prediction set contains a confirm query_id")
    scored = score_predictions(predictions, real["official"], real["tune_ids"])
    targets, _groups, _reference, queries = real["official"]
    labels = []
    probs = []
    for target in targets:
        episode_id = queries[target.query_id]
        if episode_id not in real["tune_ids"]:
            continue
        if target.query_id not in predictions:
            raise RuntimeError(f"missing tune prediction for {target.query_id}")
        labels.append(PAYOUT_TO_IDX[int(target.outcome)])
        probs.append(predictions[target.query_id])
    calibration = descriptive_calibration(np.asarray(probs), np.asarray(labels))
    return {
        "tune_R": float(scored["R"]),
        "tune_points_60": float(scored["points_60"]),
        "tune_loss_ratio_by_condition": {k: float(v) for k, v in scored["loss_ratio_by_condition"].items()},
        "n_tune_queries": len(predictions),
        "confirm_intersection": 0,
        "calibration": calibration,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(research_doc(payload), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def trial_id(phase: str, family: str, mix: float, seed: int) -> str:
    return f"{phase}__{family}__mix{int(round(mix * 100)):03d}__seed{seed}"


def discover(families_dir: Path, max_families: int) -> list[str]:
    names = []
    if not families_dir.is_dir():
        return names
    for path in sorted(p for p in families_dir.iterdir() if p.is_dir()):
        names.append(path.name)
    return names[:max_families]


def ingest_family(name: str, families_dir: Path, research: Path, min_t: int, W: int, timeout: float,
                  train_counts: np.ndarray) -> dict:
    gen_path = families_dir / name / "generate.py"
    source_sha = sha256_file(gen_path)
    out_dir = research / "families" / name
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_file():
                child.unlink()
    errors = []
    used = None
    for n_episodes, n_ticks, n_machines in GENERATE_ATTEMPTS:
        attempt_dir = out_dir / "attempt"
        if attempt_dir.exists():
            for child in attempt_dir.iterdir():
                child.unlink()
        else:
            attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_generate(gen_path, attempt_dir, 0, n_episodes, n_ticks, n_machines, timeout)
            used = {"seed": 0, "n_episodes": n_episodes, "n_ticks": n_ticks, "n_machines": n_machines}
            break
        except Exception as exc:
            errors.append(f"generate(seed=0,n_episodes={n_episodes},n_ticks={n_ticks},n_machines={n_machines}) -> {exc_text(exc)}")
            if "cap is" not in exc_text(exc) and "MAX_RECORDS" not in exc_text(exc) and "refusing" not in exc_text(exc).lower():
                break
    if used is None:
        raise RuntimeError(" | ".join(errors))
    records = read_jsonl(attempt_dir / "records.jsonl")
    truth = read_jsonl(attempt_dir / "truth.jsonl")
    episodes, report = ingest_panel(records, truth)
    report["alignment"] = check_window_alignment(episodes[0], W, min_t)
    report["settings"] = used
    report["attempt_errors"] = errors
    windows = stack_windows(episodes, W, min_t)
    synth_counts = counts_from_targets(windows[2], windows[3])
    table = calibration_weights(train_counts, synth_counts)
    weights = slot_weights(windows[2], windows[3], table)
    records_path = out_dir / "records.jsonl"
    truth_path = out_dir / "truth.jsonl"
    records_path.write_bytes((attempt_dir / "records.jsonl").read_bytes())
    truth_path.write_bytes((attempt_dir / "truth.jsonl").read_bytes())
    keys_path = attempt_dir / "return_keys.json"
    if keys_path.exists():
        (out_dir / "return_keys.json").write_bytes(keys_path.read_bytes())
    report["train_type_payout_counts"] = train_counts.tolist()
    report["synth_type_payout_counts"] = synth_counts.tolist()
    report["prior_weight_table"] = table.tolist()
    report["prior_note"] = (
        "Weights use TRAIN window counts only to rebalance synthetic label priors. "
        "They are not a claim of new information."
    )
    write_json(out_dir / "ingest_report.json", report)
    return {
        "name": name,
        "generate_path": str(gen_path),
        "generate_sha256": source_sha,
        "records_sha256": sha256_file(records_path),
        "truth_sha256": sha256_file(truth_path),
        "windows": windows,
        "weights": weights,
        "report": report,
        "settings": used,
    }


def run_trial(phase, family, mix, seed, steps, args, real, family_pack, init_state, init_sha, plan_sha, deadline):
    name = trial_id(phase, family, mix, seed)
    out = Path(args.research) / phase / name
    out.mkdir(parents=True, exist_ok=True)
    n_real, n_synth = mix_sizes(mix, args.batch_size)
    real_idx = index_plan(real["pack"][0].shape[0], steps, args.batch_size, seed)
    synth_idx = None
    synth_pack = None
    synth_weight = None
    if n_synth:
        if family_pack is None:
            raise RuntimeError("synthetic mix requested without a family pack")
        synth_idx = index_plan(family_pack["windows"][0].shape[0], steps, n_synth, seed + SYNTH_INDEX_SEED_OFFSET)
        synth_pack = family_pack["windows"]
        synth_weight = family_pack["weights"]
    config = {
        "phase": phase,
        "family": family,
        "mix": mix,
        "seed": seed,
        "steps": steps,
        "batch_size": args.batch_size,
        "n_real_per_batch": n_real,
        "n_synth_per_batch": n_synth,
        "exposures_planned": steps * args.batch_size,
        "h": args.h,
        "layers": args.layers,
        "kernel": 3,
        "W": args.W,
        "drop": args.drop,
        "lr": args.lr,
        "wd": args.wd,
        "optimizer": "AdamW",
        "scheduler": None,
        "grad_clip": 1.0,
        "use_signal": False,
        "device": "cpu",
        "threads": args.threads,
        "min_t": args.min_t,
        "init_sha256": init_sha,
        "plan_sha256": plan_sha,
        "generation_seed": None if family_pack is None else family_pack["settings"]["seed"],
        "index_plan": "real numpy Generator(trial seed), leading columns; synth numpy Generator(trial seed + 9176)",
    }
    write_json(out / "config.json", config)
    model = fresh_model(init_state, seed, args.h, args.layers, args.drop, args.W)
    if state_sha(model.state_dict()) != init_sha:
        raise RuntimeError("reloaded initialization does not match the frozen init hash")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    torch.manual_seed(seed)
    started = time.perf_counter()
    log_lines = []
    losses, interrupted = fit_steps(
        model, optimizer, real["pack"], real["weights"], synth_pack, synth_weight, real_idx,
        synth_idx if synth_idx is not None else np.zeros((steps, 0), dtype=np.int64),
        n_real, deadline,
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "phase": phase,
        "family": family,
        "mix": mix,
        "seed": seed,
        "steps_completed": len(losses),
        "steps_planned": steps,
        "exposures": len(losses) * args.batch_size,
        "train_loss_mean": float(np.mean(losses)) if losses else None,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "elapsed_train_sec": elapsed,
        "interrupted": interrupted or len(losses) != steps,
        "tune_R": None,
        "calibration": None,
    }
    if losses and not interrupted and len(losses) == steps:
        try:
            eval_started = time.perf_counter()
            metrics.update(evaluate_tune(model, real, args.W))
            metrics["elapsed_eval_sec"] = time.perf_counter() - eval_started
        except Exception as exc:
            metrics["eval_error"] = exc_text(exc)
            metrics["eval_traceback"] = traceback.format_exc()
    (out / "log.txt").write_text(
        "\n".join(
            [f"step {i} loss {loss:.6f}" for i, loss in enumerate(losses)]
            + [f"mean {metrics['train_loss_mean']}", f"tune_R {metrics['tune_R']}"]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(out / "metrics.json", metrics)
    torch.save(
        {
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": config,
            "eligibility": ELIGIBILITY,
            "eligibility_status": ELIGIBILITY_STATUS,
            "submission_export": False,
        },
        out / "checkpoint.pt",
    )
    output_files = []
    for filename in ("config.json", "metrics.json", "log.txt", "checkpoint.pt"):
        output_files.append({"path": str((out / filename).relative_to(ROOT)), "sha256": sha256_file(out / filename)})
    manifest = {
        "trial_id": name,
        "config": config,
        "metrics_summary": {k: metrics[k] for k in (
            "train_loss_mean", "tune_R", "elapsed_train_sec", "steps_completed", "exposures", "interrupted"
        )},
        "sources": [
            {"path": "source/base_cnn.py", "sha256": sha256_file(HERE / "base_cnn.py")},
            {"path": "source/runner.py", "sha256": sha256_file(HERE / "runner.py")},
            {"path": "source/call_generate.py", "sha256": sha256_file(HERE / "call_generate.py")},
            {"path": "PLAN.json", "sha256": plan_sha},
        ],
        "data_files": real["real_files"],
        "generator": None if family_pack is None else {
            "family": family,
            "path": str(Path(family_pack["generate_path"]).resolve()),
            "sha256": family_pack["generate_sha256"],
            "records_sha256": family_pack["records_sha256"],
            "truth_sha256": family_pack["truth_sha256"],
            "settings": family_pack["settings"],
            "alignment": family_pack["report"].get("alignment"),
            "n_synth_windows": int(family_pack["windows"][0].shape[0]),
            "unobserved_numeric_payouts_ignored": family_pack["report"].get("unobserved_numeric_payouts_ignored"),
            "prior_note": family_pack["report"].get("prior_note"),
        },
        "n_real_windows": int(real["pack"][0].shape[0]),
        "outputs": output_files,
        "confirm_scored": False,
        "test_loaded": False,
    }
    write_json(out / "manifest.json", manifest)
    manifest_path = out / "manifest.json"
    print(
        f"TRIAL_DONE phase={phase} family={family} mix={mix} seed={seed} "
        f"steps={metrics['steps_completed']} train_loss={metrics['train_loss_mean']} "
        f"tune_R={metrics['tune_R']} elapsed={elapsed:.1f} dir={out}",
        flush=True,
    )
    metrics["dir"] = str(out)
    metrics["manifest_sha256"] = sha256_file(manifest_path)
    return metrics


def rank_families(trials: list[dict]) -> list[dict]:
    best = {}
    for trial in trials:
        if trial["family"] == "real_only" or trial.get("tune_R") is None:
            continue
        key = (trial["tune_R"], trial["train_loss_mean"] if trial["train_loss_mean"] is not None else 1e9, trial["family"])
        current = best.get(trial["family"])
        if current is None or key < current["sort_key"]:
            best[trial["family"]] = {
                "family": trial["family"],
                "tune_R": trial["tune_R"],
                "mix": trial["mix"],
                "train_loss_mean": trial["train_loss_mean"],
                "sort_key": key,
            }
    ordered = sorted(best.values(), key=lambda row: row["sort_key"])
    for row in ordered:
        row.pop("sort_key", None)
    return ordered


def seconds_per_step(trials: list[dict]) -> float:
    rates = []
    for trial in trials:
        if trial.get("steps_completed") and trial.get("elapsed_train_sec"):
            rates.append(trial["elapsed_train_sec"] / trial["steps_completed"])
    if not rates:
        return 0.25
    return float(np.median(rates))


def main():
    args = build_parser().parse_args()
    if args.threads < 1:
        raise SystemExit("threads must be >= 1")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    deadline = parse_deadline(args.deadline)
    research = Path(args.research)
    research.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plan)
    plan_bytes = plan_path.read_bytes()
    plan_sha = sha256_bytes(plan_bytes)
    (research / "plan_freeze.json").write_bytes(plan_bytes)
    (research / "PLAN_SHA256.txt").write_text(plan_sha + "\n", encoding="utf-8")
    families_dir = Path(args.families_dir)
    print(
        f"PLAN_FROZEN sha256={plan_sha} deadline={deadline.isoformat()} threads={args.threads} device=cpu",
        flush=True,
    )
    results = []
    skipped = []
    ready = {}
    broken_sha = {}
    inits = {}

    def init_for(seed: int):
        if seed not in inits:
            state = draw_init(seed, args.h, args.layers, args.drop, args.W)
            digest = state_sha(state)
            path = research / "inits" / f"seed{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state": state, "sha256": digest, "eligibility": ELIGIBILITY, "eligibility_status": ELIGIBILITY_STATUS},
                path,
            )
            inits[seed] = (state, digest)
        return inits[seed]

    def heartbeat(stage: str):
        write_json(
            research / "heartbeat.json",
            {
                "stage": stage,
                "now": datetime.now(CHICAGO).isoformat(),
                "deadline": deadline.isoformat(),
                "trials": len(results),
                "ready_families": sorted(ready),
                "skipped": [{k: v for k, v in row.items() if k != "traceback"} for row in skipped],
            },
        )

    real = load_real(args, args.W, args.min_t)
    print(
        f"REAL_WINDOWS n={real['pack'][0].shape[0]} train_episodes={real['n_train_episodes']} "
        f"tune_episodes={real['n_tune_episodes']}",
        flush=True,
    )
    heartbeat("real_loaded")

    def time_left():
        return (deadline - datetime.now(CHICAGO)).total_seconds()

    def can_finish(steps: int, n_trials: int) -> bool:
        return time_left() > n_trials * steps * seconds_per_step(results) + 15

    def launch(phase, family, mix, seed, steps, pack):
        if datetime.now(CHICAGO) >= deadline:
            skipped.append({"family": family, "mix": mix, "phase": phase, "seed": seed, "error": "TimeoutError: deadline reached before trial start"})
            return None
        try:
            state, digest = init_for(seed)
            metrics = run_trial(phase, family, mix, seed, steps, args, real, pack, state, digest, plan_sha, deadline)
        except Exception as exc:
            message = exc_text(exc)
            skipped.append({
                "family": family, "mix": mix, "phase": phase, "seed": seed,
                "error": message, "traceback": traceback.format_exc(),
            })
            print(f"TRIAL_FAIL phase={phase} family={family} mix={mix} seed={seed} error={message}", flush=True)
            return None
        results.append(metrics)
        heartbeat(f"done {metrics['dir']}")
        return metrics

    launch("screen", "real_only", 0.0, 0, args.screen_steps, None)
    first_comparison = False
    names = discover(families_dir, args.max_families)

    def consider(name: str):
        if name in ready:
            return
        gen_path = families_dir / name / "generate.py"
        if not gen_path.exists():
            return
        if time.time() - gen_path.stat().st_mtime < args.settle_seconds:
            return
        digest = sha256_file(gen_path)
        prior = broken_sha.get(name)
        if prior is not None and prior.get("sha256") == digest:
            return
        try:
            print(f"INGEST family={name} sha256={digest}", flush=True)
            ready[name] = ingest_family(name, families_dir, research, args.min_t, args.W, args.generate_timeout, real["counts"])
            broken_sha.pop(name, None)
            print(
                f"INGEST_OK family={name} windows={ready[name]['windows'][0].shape[0]} "
                f"settings={ready[name]['settings']}",
                flush=True,
            )
        except Exception as exc:
            broken_sha[name] = {"sha256": digest, "error": exc_text(exc), "traceback": traceback.format_exc()}
            skipped.append({"family": name, "sha256": digest, "error": exc_text(exc), "traceback": traceback.format_exc()})
            print(f"FAMILY_SKIP family={name} error={exc_text(exc)}", flush=True)

    while True:
        names = discover(families_dir, args.max_families)
        for name in names:
            consider(name)
        launched = False
        for name in names:
            if name not in ready:
                continue
            if any(row["family"] == name and row["phase"] == "screen" for row in results):
                continue
            promote_trials = 2 * (1 + min(2, max(1, len(ready))) * 2)
            reserve_ok = can_finish(args.promote_steps, promote_trials)
            screen_ok = can_finish(args.screen_steps, 2) and reserve_ok
            if first_comparison and not screen_ok:
                continue
            if not first_comparison and time_left() < 5:
                break
            for mix in (0.25, 0.5):
                launch("screen", name, mix, 0, args.screen_steps, ready[name])
            first_comparison = True
            launched = True
            break
        pending = []
        for name in names:
            gen_path = families_dir / name / "generate.py"
            if name in ready:
                continue
            if not gen_path.exists():
                pending.append(name)
                continue
            if name not in broken_sha:
                pending.append(name)
        if first_comparison and not launched:
            promote_trials = 2 * (1 + min(2, len(ready)) * 2)
            if not pending or not can_finish(args.screen_steps, 2) or not can_finish(args.promote_steps, promote_trials):
                if pending and can_finish(args.screen_steps, 2) and can_finish(args.promote_steps, promote_trials):
                    time.sleep(args.poll_seconds)
                    continue
                break
        if not first_comparison:
            reserve = 2 * args.promote_steps * seconds_per_step(results) + 20
            nothing_left = bool(names) and all(
                name in ready or name in broken_sha or not (families_dir / name / "generate.py").exists()
                for name in names
            ) and not any((families_dir / name / "generate.py").exists() and name not in ready and name not in broken_sha for name in names)
            if not ready and (time_left() < reserve or time_left() < 5 or nothing_left and not pending):
                break
            if time_left() < 5:
                break
            time.sleep(args.poll_seconds)
            continue
        if launched:
            continue
        time.sleep(args.poll_seconds)

    for name in discover(families_dir, args.max_families):
        gen_path = families_dir / name / "generate.py"
        if name in ready or any(row.get("family") == name for row in skipped):
            continue
        if not gen_path.exists():
            message = f"FileNotFoundError: {gen_path} missing at screen close"
        else:
            message = f"FileNotFoundError: {gen_path} was not settled and ingested before screen close"
        skipped.append({"family": name, "error": message})
        print(f"FAMILY_SKIP family={name} error={message}", flush=True)

    ranking = rank_families([row for row in results if row["phase"] == "screen"])
    write_json(
        research / "screen_ranking.json",
        {
            "rule": "lowest tune_R, then lowest train_loss_mean, then family name",
            "note": "Frozen before promote metrics. Lower official R is better. This is not a significance test.",
            "ranking": ranking,
            "promote_families": [row["family"] for row in ranking[:2]],
        },
    )
    print(f"RANK promote={[row['family'] for row in ranking[:2]]} table={ranking[:5]}", flush=True)
    promote_names = [row["family"] for row in ranking[:2]]
    for seed in (0, 1):
        if not can_finish(args.promote_steps, 1) and time_left() < args.promote_steps * seconds_per_step(results) * 0.9:
            skipped.append({"family": "real_only", "phase": "promote", "seed": seed, "error": "TimeoutError: not enough wall time for a 300-step trial"})
            continue
        if time_left() < 5:
            break
        launch("promote", "real_only", 0.0, seed, args.promote_steps, None)
        for name in promote_names:
            for mix in (0.25, 0.5):
                if time_left() < 5:
                    skipped.append({"family": name, "phase": "promote", "mix": mix, "seed": seed, "error": "TimeoutError: deadline before promote trial"})
                    continue
                launch("promote", name, mix, seed, args.promote_steps, ready[name])

    summary = build_summary(results, skipped, ranking, plan_sha, deadline)
    write_json(research / "SUMMARY.json", summary)
    write_json(research / "skipped_families.json", {"skipped": skipped})
    print("RUN_DONE summary=" + str(research / "SUMMARY.json"), flush=True)
    return summary


def build_summary(results, skipped, ranking, plan_sha, deadline) -> dict:
    def brief(row):
        calibration = row.get("calibration") or {}
        return {
            "phase": row["phase"],
            "family": row["family"],
            "mix": row["mix"],
            "seed": row["seed"],
            "steps_completed": row["steps_completed"],
            "exposures": row["exposures"],
            "train_loss_mean": row["train_loss_mean"],
            "tune_R": row["tune_R"],
            "elapsed_train_sec": row["elapsed_train_sec"],
            "calibration_nll": calibration.get("nll"),
            "calibration_ece_5bin": calibration.get("ece_5bin"),
            "interrupted": row["interrupted"],
            "dir": row.get("dir"),
            "eval_error": row.get("eval_error"),
        }

    spreads = []
    groups = {}
    for row in results:
        if row["phase"] != "promote" or row.get("tune_R") is None:
            continue
        groups.setdefault((row["family"], row["mix"]), []).append(row)
    for (family, mix), rows in sorted(groups.items()):
        scores = [row["tune_R"] for row in rows]
        spreads.append(
            {
                "family": family,
                "mix": mix,
                "seeds": [row["seed"] for row in rows],
                "tune_R": scores,
                "spread_max_minus_min": (max(scores) - min(scores)) if len(scores) >= 2 else None,
                "note": "spread_max_minus_min is the range across the listed seeds. It is not a confidence interval.",
            }
        )
    return {
        "plan_sha256": plan_sha,
        "deadline": deadline.isoformat(),
        "ranking_rule": "lowest tune_R is better",
        "screen_ranking": ranking,
        "trials": [brief(row) for row in results],
        "seed_spreads": spreads,
        "skipped": [{k: v for k, v in row.items() if k != "traceback"} for row in skipped],
        "confirm_scored": False,
        "test_loaded": False,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("RUN_FAIL", flush=True)
        raise
