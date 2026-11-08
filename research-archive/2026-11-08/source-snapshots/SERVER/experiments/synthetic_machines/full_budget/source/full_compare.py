"""Full-budget CNN03 comparison. RESEARCH_ONLY. No submission export.

Canonical loop matches experiments/mlp_cnn_hpo/source/train.py: AdamW, grad clip 1,
OneCycleLR, masked token-mean cross-entropy, no signal. Tune split only.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import base_cnn as B

B.USE_SIGNAL = False

from house_edge.data import MISSING, PAYOUT_TO_IDX, TYPE_TO_IDX, subset_split_data  # noqa: E402
from house_edge.scoring import load_official_validation, score_predictions  # noqa: E402

VERIFIED_ABRUPT = "6f6e29c88e831c72084645d418bd51ba8173b20231e97d17bd70d27850bf2dde"
LR, WD, EFFECTIVE_BS = 0.00197, 0.00895, 512
CURVE_EPOCHS = (1, 4, 8, 16, 30)
ARCH = {
    "tiny": {"h": 32, "layers": 2, "W": 32, "drop": 0.3, "mix": 0.0},
    "cnn03": {"h": 128, "layers": 4, "W": 64, "drop": 0.3, "mix": 0.0},
    "mix": {"h": 128, "layers": 4, "W": 64, "drop": 0.3, "mix": 0.25},
}


def research(payload):
    payload = dict(payload)
    payload["eligibility"] = "RESEARCH_ONLY"
    payload["eligibility_status"] = "unresolved"
    payload["submission_export"] = False
    payload["flagship_claim"] = False
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(research(payload), indent=1, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Full-budget real vs abrupt mix. Tune only.")
    parser.add_argument("--case", default="tiny,cnn03,mix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "research"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-cap-gb", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="CPU import, diagnostics, and one optimizer step.")
    return parser.parse_args()


def apply_gpu_cap(gb: float) -> dict:
    if not torch.cuda.is_available():
        return {"applied": False, "reason": "cuda unavailable"}
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(gb * (1024 ** 3))
    fraction = min(1.0, cap / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {"applied": True, "cap_bytes": cap, "total_bytes": int(total), "fraction": fraction}


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"bad observed value {value!r}")


def payout_index(record: dict) -> int:
    payout = int(record["payout"])
    if payout not in PAYOUT_TO_IDX:
        raise ValueError(f"observed payout {payout} outside {tuple(PAYOUT_TO_IDX)}")
    return PAYOUT_TO_IDX[payout]


def ingest_records(records) -> tuple[list[dict], dict]:
    """Public TYPE_TO_IDX on machine_type only. Unobserved payout is not written."""
    grouped = {}
    ignored = 0
    observed = 0
    for row in records:
        episode = grouped.setdefault(row["episode_id"], {"types": [None] * 8, "cells": {}})
        slot = int(row["machine_id"])
        if not 0 <= slot <= 7:
            raise ValueError(f"machine_id {slot} is not a slot 0..7")
        kind = row["machine_type"]
        if kind not in TYPE_TO_IDX:
            raise ValueError(f"machine_type {kind!r} is not a public type")
        type_id = TYPE_TO_IDX[kind]
        if episode["types"][slot] is None:
            episode["types"][slot] = type_id
        elif episode["types"][slot] != type_id:
            raise ValueError("machine_type changed inside an episode")
        seen = as_bool(row["observed"])
        if seen:
            episode["cells"][(int(row["tick"]), slot)] = payout_index(row)
            observed += 1
        else:
            episode["cells"][(int(row["tick"]), slot)] = MISSING
            if row.get("payout") is not None:
                ignored += 1
    episodes = []
    for episode_id, episode in grouped.items():
        if any(v is None for v in episode["types"]):
            raise ValueError(f"episode {episode_id} does not contain all 8 machines")
        max_tick = max(tick for tick, _slot in episode["cells"])
        obs = np.full((max_tick + 1, 8), MISSING, dtype=np.int8)
        for (tick, slot), value in episode["cells"].items():
            obs[tick, slot] = value
        episodes.append({"episode_id": str(episode_id), "types": np.asarray(episode["types"], np.int64), "obs": obs})
    return episodes, {"n_records": len(records), "n_observed": observed, "unobserved_payouts_ignored": ignored}


def stack_windows(episodes, width: int, min_t: int = 16):
    xs, ts, tys, ys = [], [], [], []
    for episode in episodes:
        obs, types = episode["obs"], episode["types"]
        for tick in range(min_t, obs.shape[0]):
            target = obs[tick]
            if int((target >= 0).sum()) == 0:
                continue
            xs.append(B.window(obs, tick, width))
            ts.append(tick)
            tys.append(types)
            ys.append(target.copy())
    if not xs:
        raise ValueError("no synthetic windows")
    return (
        torch.tensor(np.stack(xs)),
        torch.tensor(np.asarray(ts, np.int64)),
        torch.tensor(np.stack(tys).astype(np.int64)),
        torch.zeros(len(xs), 8, dtype=torch.long),
        torch.tensor(np.stack(ys)),
    )


def class_counts(targets: torch.Tensor) -> dict:
    flat = targets.reshape(-1)
    mask = flat >= 0
    counts = torch.bincount(flat[mask].long(), minlength=5).tolist()
    return {
        "windows": int(targets.shape[0]),
        "labelled_slots": int(mask.sum()),
        "missing_slots": int((~mask).sum()),
        "class_counts_payout_index": counts,
    }


def ast_of(path: Path, names: set[str]) -> dict:
    tree = ast.parse(path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.dump(node, include_attributes=False)
    return found


def compare_sources() -> dict:
    names = {"CNN", "Block", "masked_loss", "window", "build_train", "build_query", "predict", "to_preds"}
    ours = ast_of(HERE / "base_cnn.py", names)
    canon_path = ROOT.parent.parent / "mlp_cnn_hpo" / "source" / "base_cnn.py"
    canon = ast_of(canon_path, names)
    utility = ROOT.parent / "grok_utility" / "source" / "base_cnn.py"
    report = {"canonical": str(canon_path), "equal": {}, "missing": sorted(names - set(ours))}
    for name in sorted(names):
        report["equal"][name] = ours.get(name) == canon.get(name)
    if utility.exists():
        other = ast_of(utility, names)
        report["utility_cnn_block_equal"] = ours.get("CNN") == other.get("CNN") and ours.get("Block") == other.get("Block")
    report["all_equal"] = all(report["equal"].values()) and not report["missing"]
    return report


def load_canon_module():
    path = ROOT.parent.parent / "mlp_cnn_hpo" / "source" / "base_cnn.py"
    spec = importlib.util.spec_from_file_location("canonical_base_cnn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.USE_SIGNAL = False
    return module


def alignment_report(train) -> dict:
    episode = train.episodes[0]
    tick = 40 if episode.obs.shape[0] > 40 else episode.obs.shape[0] - 1
    base = B.window(episode.obs, tick, 64)
    future = episode.obs.copy()
    future[tick:, :] = 0
    past = episode.obs.copy()
    past[tick - 1, 0] = 1 if past[tick - 1, 0] != 1 else 3
    return {
        "tick": int(tick),
        "future_mutation_ignored": bool(np.array_equal(base, B.window(future, tick, 64))),
        "past_mutation_visible": bool(not np.array_equal(base, B.window(past, tick, 64))),
        "missing_code": int(MISSING),
        "query_label_is_target_not_input": (
            "build_train writes the train query payout onto full[cutoff] as Y only. "
            "window() copies ticks < t, so that label is not an input."
        ),
    }


def load_abrupt(width: int):
    path = HERE / "abrupt_generate.py"
    digest = sha256_file(path)
    if digest != VERIFIED_ABRUPT:
        raise RuntimeError(f"abrupt_generate.py sha256 {digest} != verified {VERIFIED_ABRUPT}")
    spec = importlib.util.spec_from_file_location("abrupt_generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panel = module.generate(seed=0, n_episodes=8, n_ticks=128, n_machines=8)
    if "signal" in panel["records"][0]:
        raise RuntimeError("abrupt records contain a signal field; refusing to read it")
    episodes, info = ingest_records(panel["records"])
    # Truth is checked for presence and then discarded. It is not a feature or a label.
    if not panel.get("truth"):
        raise RuntimeError("abrupt panel has no truth list")
    pack = stack_windows(episodes, width)
    info.update({"sha256": digest, "settings": panel["settings"], "n_windows": int(pack[0].shape[0]),
                 "truth_rows_discarded": len(panel["truth"]), "queries_discarded": len(panel.get("queries") or [])})
    return pack, info


def masked_stats(logits, targets):
    labels = targets.long()
    mask = labels >= 0
    count = int(mask.sum())
    if count == 0:
        raise RuntimeError("batch has no observed targets")
    summed = F.cross_entropy(logits.float()[mask], labels[mask], reduction="sum")
    correct = int((logits.detach().argmax(-1)[mask] == labels[mask]).sum())
    return summed, count, correct


def move_batch(pack, index, device):
    index = index.cpu().long()
    batch = [tensor.index_select(0, index) for tensor in pack]
    if device != "cpu":
        batch = [tensor.to(device, non_blocking=False) for tensor in batch]
    return batch


def optimizer_step(model, opt, sch, pack, index, device, micro):
    """One canonical step. Microbatches divide the summed token loss by the full-batch label count."""
    opt.zero_grad(set_to_none=True)
    total_sum = None
    total_n = 0
    correct = 0
    windows = int(index.shape[0])
    for start in range(0, windows, micro):
        batch = move_batch(pack, index[start:start + micro], device)
        try:
            with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                logits = model(*batch[:4])
            summed, count, n_correct = masked_stats(logits, batch[4])
        except RuntimeError as exc:
            if is_oom(exc):
                opt.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
            raise
        summed.backward()
        total_sum = summed.detach() if total_sum is None else total_sum + summed.detach()
        total_n += count
        correct += n_correct
        del batch, logits, summed
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(total_n)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sch.step()
    return float(total_sum.cpu()) / total_n, windows, total_n, correct


def compose_index(real_index, n_synth, mix, rng, device):
    if mix <= 0 or n_synth <= 0:
        return real_index, None
    n_syn = int(round(mix * int(real_index.shape[0])))
    n_syn = min(max(n_syn, 0), int(real_index.shape[0]))
    if n_syn == 0:
        return real_index, None
    synth = torch.from_numpy(rng.integers(0, n_synth, size=n_syn).astype(np.int64))
    keep = real_index[:-n_syn] if n_syn < real_index.shape[0] else real_index[:0]
    return keep, synth


def new_model(spec, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = B.CNN(spec["h"], spec["layers"], 3, spec["drop"], spec["W"], use_signal=False)
    if model.use_signal or model.sig is not None:
        raise RuntimeError("signal pathway was constructed")
    return model


def state_sha(model) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def score_tune(model, query, episodes, tune_data, official, tune_ids, confirm_ids):
    copied = copy.deepcopy(model).to("cpu").float().eval()
    torch.set_num_threads(2)
    probabilities = B.predict(copied, query, "cpu", bs=64)
    predictions = B.to_preds(probabilities, episodes, tune_data)
    scored_episodes = {r["episode_id"] for r in tune_data.queries}
    if set(scored_episodes) & set(confirm_ids):
        raise RuntimeError("tune query set intersects confirm")
    if not set(scored_episodes) <= set(tune_ids):
        raise RuntimeError("scored episodes leave the tune split")
    metrics = score_predictions(predictions, official, tune_ids)
    return float(metrics["R"])


def train_case(name, spec, seed, epochs, pack, synth_pack, query, episodes, tune_data, official,
               tune_ids, confirm_ids, device, out_dir, max_steps=None):
    case_dir = out_dir / f"{name}_seed{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)
    model = new_model(spec, seed)
    init_sha = state_sha(model)
    n = int(pack[0].shape[0])
    steps_planned = epochs * ((n + EFFECTIVE_BS - 1) // EFFECTIVE_BS)
    plans = [(EFFECTIVE_BS, EFFECTIVE_BS), (128, EFFECTIVE_BS), (64, EFFECTIVE_BS)]
    if device != "cuda":
        plans = [(EFFECTIVE_BS, EFFECTIVE_BS)]
    last_oom = None
    for micro, _effective in plans:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = new_model(spec, seed)
            if state_sha(model) != init_sha:
                raise RuntimeError("reseeded initialization changed")
            model = model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps_planned)
            rng = np.random.default_rng(seed + 9176)
            joined = concat_packs(pack, synth_pack) if synth_pack is not None else None
            real_rows = int(pack[0].shape[0])
            losses, exposures, real_windows, synth_windows = [], 0, 0, 0
            labelled = 0
            started = time.perf_counter()
            done_steps = 0
            curve = []
            stopped = False
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(n)
                total = count = 0
                for start in range(0, n, EFFECTIVE_BS):
                    if max_steps is not None and done_steps >= max_steps:
                        stopped = True
                        break
                    real_idx = perm[start:start + EFFECTIVE_BS]
                    keep, synth_idx = compose_index(
                        real_idx, 0 if synth_pack is None else int(synth_pack[0].shape[0]), spec["mix"], rng, device
                    )
                    if synth_idx is None:
                        loss, windows, n_lab, _correct = optimizer_step(model, opt, sch, pack, real_idx, device, micro)
                        real_windows += windows
                    else:
                        mixed = torch.cat([keep.cpu().long(), synth_idx.long() + real_rows])
                        loss, windows, n_lab, _correct = optimizer_step(model, opt, sch, joined, mixed, device, micro)
                        real_windows += int(keep.shape[0])
                        synth_windows += int(synth_idx.shape[0])
                    total += loss * windows
                    count += windows
                    exposures += windows
                    labelled += n_lab
                    done_steps += 1
                if count:
                    losses.append(total / count)
                epoch_no = epoch + 1
                row = {"epoch": epoch_no, "train_loss": losses[-1] if losses else None}
                if epoch_no in CURVE_EPOCHS or epoch_no == epochs:
                    row["tune_R"] = score_tune(model, query, episodes, tune_data, official, tune_ids, confirm_ids)
                    curve.append(row)
                    print(f"CURVE case={name} seed={seed} epoch={epoch_no} loss={row['train_loss']:.4f} tune_R={row['tune_R']:.6f}", flush=True)
                else:
                    print(f"EPOCH case={name} seed={seed} epoch={epoch_no} loss={row['train_loss']:.4f}", flush=True)
                if stopped:
                    break
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
            metrics = {
                "case": name, "seed": seed, "init_sha256": init_sha, "steps_completed": done_steps,
                "steps_planned": steps_planned, "exposures": exposures, "real_windows": real_windows,
                "synth_windows": synth_windows, "labelled_slots_in_steps": labelled,
                "microbatch": micro, "effective_batch": EFFECTIVE_BS,
                "batching": "host-staged microbatch; token-sum divided by labelled slots in the effective batch",
                "nominal_vs_resident_512": "Same objective and OneCycle step count. Host staging plus a smaller microbatch changes only addition order if micro < 512.",
                "optimizer": "AdamW", "lr": LR, "wd": WD, "scheduler": "OneCycleLR",
                "drop": spec["drop"], "h": spec["h"], "layers": spec["layers"], "W": spec["W"], "mix": spec["mix"],
                "losses": losses, "curve": curve, "tune_R": curve[-1].get("tune_R") if curve else None,
                "elapsed_sec": elapsed, "peak_gpu_mb": peak, "device": device, "oom_fallbacks": last_oom,
            }
            write_json(case_dir / "metrics.json", metrics)
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state": cpu_state, "eligibility": "RESEARCH_ONLY", "eligibility_status": "unresolved",
                        "submission_export": False, "config": {k: metrics[k] for k in ("case", "seed", "h", "layers", "W", "drop", "mix", "lr", "wd")}},
                       case_dir / "checkpoint.pt")
            write_json(case_dir / "config.json", {"case": name, "seed": seed, "epochs": epochs, "microbatch": micro})
            print(f"CASE_DONE {name} seed={seed} steps={done_steps} exposures={exposures} tune_R={metrics['tune_R']} dir={case_dir}", flush=True)
            return metrics
        except RuntimeError as exc:
            if not is_oom(exc) or micro == plans[-1][0]:
                if is_oom(exc):
                    last_oom = str(exc)
                    write_json(case_dir / "metrics.json", {"case": name, "seed": seed, "error": f"RuntimeError: {exc}", "oom": True})
                    print(f"CASE_OOM {name} seed={seed} gave up at micro={micro}", flush=True)
                    return {"case": name, "seed": seed, "tune_R": None, "error": str(exc), "losses": []}
                raise
            last_oom = f"OOM at microbatch {micro}; retrying smaller"
            print(f"OOM_FALLBACK case={name} micro={micro} -> next", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
    return {"case": name, "seed": seed, "tune_R": None, "error": last_oom, "losses": []}


def concat_packs(real, synth):
    return tuple(torch.cat([a, b], 0) for a, b in zip(real, synth))


def memorize(pack) -> dict:
    n = min(32, int(pack[0].shape[0]))
    spec = {"h": 32, "layers": 2, "W": int(pack[0].shape[1]), "drop": 0.0, "mix": 0.0}
    model = new_model(spec, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=20)
    index = torch.arange(n)
    curve = []
    model.train()
    for step in range(20):
        loss, windows, n_lab, correct = optimizer_step(model, opt, sch, pack, index, "cpu", n)
        curve.append({"step": step, "loss": loss, "token_accuracy": correct / n_lab, "labelled_slots": n_lab, "windows": windows})
    return {"windows": n, "dropout": 0.0, "steps": curve, "final_loss": curve[-1]["loss"], "final_token_accuracy": curve[-1]["token_accuracy"],
            "note": "Unique real TRAIN windows only. No held-out fit. Dropout is off so memorization can succeed."}


def build_context(width_needed):
    B.USE_SIGNAL = False
    train = B.load_split(str(Path(B.PUB) / "train"), with_labels=True)
    validation = B.load_split(str(Path(B.PUB) / "validation"), with_labels=False)
    splits = json.loads((Path(B.REPO) / "runs" / "splits.json").read_text())
    tune_ids, confirm_ids = set(splits["tune"]), set(splits["confirm"])
    if tune_ids & confirm_ids:
        raise RuntimeError("tune and confirm overlap")
    tune_data = subset_split_data(validation, tune_ids)
    packs = {}
    for width in width_needed:
        packs[width] = tuple(B.build_train(train, width))
    canon = load_canon_module()
    canon_pack = tuple(canon.build_train(train, 64))
    tensor_equal = all(torch.equal(a, b) for a, b in zip(packs[64][:5], canon_pack[:5])) if 64 in packs else None
    hashes = {}
    for width, pack in packs.items():
        hashes[str(width)] = {
            "shape": [list(t.shape) for t in pack],
            "dtype": [str(t.dtype) for t in pack],
            "sha256": [sha256_tensor(t) for t in pack],
            "counts": class_counts(pack[4]),
        }
    report = {
        "root_cause": {
            "screen_regime": {"steps": 60, "batch": 128, "exposures": 7680, "lr": 0.002, "scheduler": "constant", "drop": 0.1, "h": 32, "layers": 2, "W": 32},
            "canonical_regime": {"steps": 1110, "batch": 512, "exposures": 557010, "lr": LR, "scheduler": "OneCycleLR", "drop": 0.3, "h": 128, "layers": 4, "W": 64,
                                 "server_tune_R": 0.33206398675779764, "server_full_R_not_recomputed": 0.36845454406496114,
                                 "server_losses": [1.226, 1.0746, 0.9709, 0.7904], "server_seconds": 73},
            "conclusion": "60 steps at batch 128 is 7680 exposures against 557010 full exposures. That gap is the regime, not evidence that the pipeline or a full mixture failed.",
        },
        "score_normalization": "Official R is the mean loss ratio versus the frozen reference on tune episodes only. Confirm and test are not scored. Loss is the mean of per-slot cross-entropy over y>=0; missing (-1) is excluded, not class 0.",
        "payout_to_index": {str(k): v for k, v in PAYOUT_TO_IDX.items()},
        "type_to_index": dict(TYPE_TO_IDX),
        "abrupt_TYPE_TO_INDEX_is_not_used": "abrupt_generate.py exports steady=0,balanced=1,jackpot=2. Rows are mapped only through the machine_type string and public TYPE_TO_IDX.",
        "alignment": alignment_report(train),
        "ast": compare_sources(),
        "canonical_tensor_equal_W64": tensor_equal,
        "packs": hashes,
        "n_train_episodes": len(train.episodes),
        "n_tune_episodes": len(tune_data.episodes),
        "expected_windows": 18567,
        "sources": {
            "base_cnn.py": sha256_file(HERE / "base_cnn.py"),
            "full_compare.py": sha256_file(HERE / "full_compare.py"),
            "abrupt_generate.py": sha256_file(HERE / "abrupt_generate.py"),
            "canonical_base_cnn.py": sha256_file(ROOT.parent.parent / "mlp_cnn_hpo" / "source" / "base_cnn.py"),
            "canonical_train.py": sha256_file(ROOT.parent.parent / "mlp_cnn_hpo" / "source" / "train.py"),
        },
    }
    official = load_official_validation(Path(B.PUB) / "validation")
    query = {}
    for width in width_needed:
        query[width] = B.build_query(tune_data, width)
    return train, packs, tune_data, official, tune_ids, confirm_ids, query, report


def main():
    args = parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = {"applied": False}
    device = args.device
    if args.smoke:
        device = "cpu"
    elif device == "cuda":
        cap = apply_gpu_cap(args.gpu_cap_gb)
        if not torch.cuda.is_available():
            write_json(out / "SUMMARY.json", {"error": "cuda requested but torch.cuda.is_available() is false", "gpu_cap": cap})
            raise SystemExit("cuda requested but unavailable")
    print(f"FULL_COMPARE device={device} cap={cap} epochs={args.epochs} cases={args.case}", flush=True)
    cases = [c.strip() for c in args.case.split(",") if c.strip()]
    widths = sorted({ARCH[c]["W"] for c in cases} | {64, 32})
    _train, packs, tune_data, official, tune_ids, confirm_ids, query, report = build_context(widths)
    report["gpu_cap"] = cap
    report["memorization"] = memorize(packs[32])
    abrupt_pack = None
    if "mix" in cases:
        abrupt_pack, abrupt_info = load_abrupt(64)
        report["abrupt"] = abrupt_info
        report["abrupt_counts"] = class_counts(abrupt_pack[4])
    write_json(out / "diagnostics.json", report)
    print(f"DIAG windows64={packs[64][0].shape[0]} tensor_equal={report['canonical_tensor_equal_W64']} "
          f"ast={report['ast']['all_equal']} memorize_loss={report['memorization']['final_loss']:.4f} "
          f"acc={report['memorization']['final_token_accuracy']:.3f}", flush=True)
    if args.smoke:
        metrics = train_case("tiny", ARCH["tiny"], args.seed, 1, packs[32], None, query[32][0], query[32][1],
                             tune_data, official, tune_ids, confirm_ids, "cpu", out / "smoke", max_steps=1)
        write_json(out / "SUMMARY.json", {"smoke": True, "diagnostics": "diagnostics.json", "one_step": metrics})
        print("SMOKE_DONE", flush=True)
        return
    results = []
    init_seen = {}

    def run_one(case, seed):
        spec = ARCH[case]
        pack = packs[spec["W"]]
        synth = abrupt_pack if spec["mix"] else None
        q, eps = query[spec["W"]]
        metrics = train_case(case, spec, seed, args.epochs, pack, synth, q, eps, tune_data, official,
                             tune_ids, confirm_ids, device, out, max_steps=None)
        key = (spec["h"], spec["layers"], spec["W"], spec["drop"], seed)
        init_seen.setdefault(key, metrics.get("init_sha256"))
        if metrics.get("init_sha256") and init_seen[key] != metrics["init_sha256"]:
            raise RuntimeError(f"init hash diverged for {case} seed {seed}")
        results.append(metrics)
        return metrics

    for case in cases:
        run_one(case, args.seed)
    by = {(m["case"], m["seed"]): m for m in results}
    decision = {"rule": "Win means mix tune_R strictly below the same-seed cnn03 tune_R. Seed 0 loss does not add seeds. Three wins are required and still are not a flagship claim."}
    if "mix" in by and "cnn03" in by and by[("mix", args.seed)].get("tune_R") is not None and by[("cnn03", args.seed)].get("tune_R") is not None:
        seed0_win = by[("mix", args.seed)]["tune_R"] < by[("cnn03", args.seed)]["tune_R"]
        decision["seed0_mix_tune_R"] = by[("mix", args.seed)]["tune_R"]
        decision["seed0_cnn03_tune_R"] = by[("cnn03", args.seed)]["tune_R"]
        decision["seed0_win"] = seed0_win
        if seed0_win and args.seed == 0 and {"cnn03", "mix"} <= set(cases):
            print("DECISION seed0 mix wins; running matched seeds 1 and 2", flush=True)
            for seed in (1, 2):
                for case in ("cnn03", "mix"):
                    run_one(case, seed)
            wins = []
            for seed in (0, 1, 2):
                mix_m = next(m for m in results if m["case"] == "mix" and m["seed"] == seed)
                ctrl = next(m for m in results if m["case"] == "cnn03" and m["seed"] == seed)
                wins.append(mix_m.get("tune_R") is not None and ctrl.get("tune_R") is not None and mix_m["tune_R"] < ctrl["tune_R"])
            decision["wins"] = wins
            decision["three_of_three"] = all(wins)
        else:
            decision["pruned"] = not seed0_win
            print(f"DECISION prune={not seed0_win} seed0_win={seed0_win}", flush=True)
    write_json(out / "SUMMARY.json", {"gpu_cap": cap, "results": results, "decision": decision, "diagnostics": "diagnostics.json"})
    print("FULL_DONE " + str(out / "SUMMARY.json"), flush=True)


if __name__ == "__main__":
    main()
