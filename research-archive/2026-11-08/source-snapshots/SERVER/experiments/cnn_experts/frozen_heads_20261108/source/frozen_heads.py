"""Frozen CNN03-style trunk plus tiny signal heads. Seed 0 only.

The trunk checkpoint is read-only. This file never writes outside
experiments/cnn_experts/frozen_heads_20261108.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "kit"))
sys.path.insert(0, os.path.join(REPO, "experiments", "mlp_cnn_hpo", "source"))

import base_cnn as B  # noqa: E402

B.USE_SIGNAL = False

from house_edge.data import (  # noqa: E402
    MACHINE_TYPES,
    MISSING,
    N_MACHINES,
    PAYOUTS,
    TYPE_TO_IDX,
    load_split,
)
from house_edge.scoring import load_official_validation, score_predictions  # noqa: E402

# Frozen before metrics. See PLAN.md.
H = 128
LAYERS = 4
KERNEL = 3
W = 64
DROP = 0.3
HEAD_EPOCHS = 10
HEAD_LR = 1e-3
HEAD_WD = 0.0
BS = 512
MIN_GROUP_ROWS = 64
MIN_T = 16
TRUNK_PARAMS = 367109
HEAD_PARAMS = 30
EXPECTED_SIGNAL_ROWS = {-1: 250, 0: 267, 1: 251}
SIGNALS = (-1, 0, 1)
VRAM_BYTES = 1 * 1024 ** 3
TRUNK_REL = "experiments/cnn_experts/hard_signal/runs/s0/pretrain/checkpoint.pt"
TRUNK_SHA = "e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760"

window = B.window


class TinyHead(nn.Module):
    """30-parameter map on frozen 5-logits. Identity at initialization."""

    def __init__(self):
        super().__init__()
        self.map = nn.Linear(5, 5, bias=True)
        nn.init.eye_(self.map.weight)
        nn.init.zeros_(self.map.bias)

    def forward(self, logits):
        return self.map(logits)


def require_seed0(seed) -> None:
    if int(seed) != 0:
        raise SystemExit(f"rejected seed {seed}: this idea allows seed 0 only")


def trainable_count(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp() -> str:
    return datetime.datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S CST")


def status(line: str) -> None:
    with open(os.path.join(ROOT, "STATUS.md"), "a", encoding="utf-8") as handle:
        handle.write(f"{stamp()} {line}\n")


def index_queries(data):
    by = {}
    for row in data.queries:
        by.setdefault(row["episode_id"], []).append(row)
    return by


def build_cutoff(data, width=W, require_labels=False):
    """One window per episode at the real cutoff. Signal is not a feature."""
    by = index_queries(data)
    xs, ts, tys, ys, sgs = [], [], [], [], []
    for ep in data.episodes:
        y = np.full(N_MACHINES, MISSING, np.int64)
        sg = np.full(N_MACHINES, 99, np.int64)
        for row in by.get(ep.episode_id, []):
            slot = ep.slot_of(row["machine_id"])
            sg[slot] = int(row["signal"])
            if data.query_labels is not None and row["query_id"] in data.query_labels:
                y[slot] = int(data.query_labels[row["query_id"]])
        xs.append(window(ep.obs, ep.cutoff, width))
        ts.append(ep.cutoff)
        tys.append(ep.machine_types)
        ys.append(y)
        sgs.append(sg)
    batch = {
        "x": torch.tensor(np.stack(xs)),
        "t": torch.tensor(np.asarray(ts, dtype=np.int64)),
        "ty": torch.tensor(np.stack(tys)),
        "y": torch.tensor(np.stack(ys)),
        "sig": torch.tensor(np.stack(sgs)),
    }
    if require_labels:
        if data.query_labels is None or bool((batch["y"] < 0).any()) or bool((batch["sig"] == 99).any()):
            raise RuntimeError("cutoff labels or signals are incomplete")
    return batch


def build_history(data, width=W, min_t=MIN_T):
    """History windows only. Does not read queries."""
    xs, ts, cutoffs = [], [], []
    for ep in data.episodes:
        obs = ep.obs
        last = min(obs.shape[0], ep.cutoff)
        for t in range(min_t, ep.cutoff):
            if t >= last or int((obs[t] >= 0).sum()) == 0:
                continue
            xs.append(window(obs, t, width))
            ts.append(t)
            cutoffs.append(ep.cutoff)
    return {
        "x": torch.tensor(np.stack(xs)),
        "t": torch.tensor(np.asarray(ts, dtype=np.int64)),
        "cutoffs": torch.tensor(np.asarray(cutoffs, dtype=np.int64)),
    }


def count_signal_rows(data) -> dict:
    counts = {s: 0 for s in SIGNALS}
    for row in data.queries:
        if data.query_labels and row["query_id"] in data.query_labels:
            counts[int(row["signal"])] += 1
    return counts


def route_table(counts, min_rows=MIN_GROUP_ROWS) -> dict:
    return {s: ("specialist" if int(counts[s]) >= int(min_rows) else "trunk") for s in SIGNALS}


def loss_mask(y, sig, signal):
    mask = y.long() >= 0
    if signal is None:
        return mask
    return mask & (sig.long() == int(signal))


def normalize_rows(probs: np.ndarray) -> np.ndarray:
    """Float64 row sums forced to 1. Matches submission renormalization."""
    arr = np.asarray(probs, dtype=np.float64)
    total = arr.sum(axis=-1, keepdims=True)
    return arr / total


def make_trunk():
    model = B.CNN(H, LAYERS, KERNEL, DROP, W, use_signal=False)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def load_trunk(path=None):
    path = path or os.path.join(REPO, TRUNK_REL)
    digest = sha256_file(path)
    if digest != TRUNK_SHA:
        raise RuntimeError(f"trunk sha256 {digest} != frozen {TRUNK_SHA}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("phase") != "pretrain" or ckpt.get("seed") != 0 or ckpt.get("config", {}).get("epochs") != 30:
        raise RuntimeError(f"trunk provenance is not the history pretrain: {ckpt.get('phase')} {ckpt.get('config')}")
    model = make_trunk()
    model.load_state_dict(ckpt["state"])
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    if n != TRUNK_PARAMS or model.sig is not None:
        raise RuntimeError(f"trunk params {n} or signal embedding unexpected")
    return model, ckpt


def snapshot_frozen(module) -> dict:
    out = {}
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        out[name] = tensor.detach().cpu().clone()
    return out


def frozen_equal(module, snap: dict) -> bool:
    current = snapshot_frozen(module)
    if set(current) != set(snap):
        return False
    return all(torch.equal(current[k], snap[k]) for k in snap)


def trunk_logits(model, x, t, ty):
    sg = torch.zeros(x.shape[0], N_MACHINES, dtype=torch.long, device=x.device)
    return model(x, t, ty, sg)


def train_head(head, trunk, batch, signal, device, epochs=HEAD_EPOCHS, lr=HEAD_LR, wd=HEAD_WD, bs=BS):
    """Train one tiny head. Trunk is eval, frozen, and detached."""
    if any(p.requires_grad for p in trunk.parameters()):
        raise RuntimeError("trunk is trainable")
    head = head.to(device)
    trunk = trunk.to(device).eval()
    x, t, ty = batch["x"].to(device), batch["t"].to(device), batch["ty"].to(device)
    y, sig = batch["y"].to(device), batch["sig"].to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    n = x.shape[0]
    losses, steps = [], 0
    t0 = time.time()
    phase = "control" if signal is None else f"specialist_{int(signal):+d}"
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        epoch_sum, epoch_n = 0.0, 0
        for start in range(0, n, bs):
            j = perm[start:start + bs]
            with torch.no_grad():
                base = trunk_logits(trunk, x[j], t[j], ty[j])
            pred = head(base.detach())
            mask = loss_mask(y[j], sig[j], signal)
            if int(mask.sum()) == 0:
                raise RuntimeError(f"{phase} empty mask")
            loss = F.cross_entropy(pred.float()[mask], y[j].long()[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            steps += 1
            epoch_sum += float(loss.detach()) * int(mask.sum())
            epoch_n += int(mask.sum())
        losses.append(epoch_sum / max(epoch_n, 1))
        print(f"epoch {ep} loss {losses[-1]:.4f} t {time.time() - t0:.0f}s phase={phase}", flush=True)
        if ep == 0 and phase == "control" and str(device).startswith("cuda"):
            status(f"first epoch control head loss {losses[-1]:.6f} pid={os.getpid()} device={device}")
    return {"losses": losses, "steps": steps, "phase": phase, "labeled_slots": int(loss_mask(y, sig, signal).sum())}


@contextmanager
def float64_casts():
    original = torch.Tensor.float

    def as_float64(self, memory_format=torch.preserve_format):
        return self.to(dtype=torch.float64, memory_format=memory_format)

    torch.Tensor.float = as_float64
    try:
        yield
    finally:
        torch.Tensor.float = original


def apply_heads(base_logits, sig_row, heads, route):
    """base_logits (8, 5) float64 numpy. Exact hard stitch, then renorm per slot."""
    out = np.array(base_logits, dtype=np.float64, copy=True)
    names = []
    for slot, raw in enumerate(np.asarray(sig_row).tolist()):
        sig = int(raw)
        if route.get(sig) == "specialist":
            head = heads[sig]
            vec = head(torch.tensor(base_logits[slot], dtype=torch.float64)).detach().cpu().numpy()
            names.append(f"specialist_{sig:+d}")
        else:
            vec = base_logits[slot]
            names.append("trunk")
        out[slot] = vec
    return out, names


def predict_split(trunk, heads, route, data, width, mode):
    """mode is baseline, control, or routed. CPU float64."""
    trunk = trunk.to("cpu").double().eval()
    if mode == "control":
        head = heads["control"].to("cpu").double().eval()
    elif mode == "routed":
        spec = {s: heads[s].to("cpu").double().eval() for s in heads if s in SIGNALS}
    cut = build_cutoff(data, width, require_labels=False)
    preds = {}
    routes = []
    with torch.no_grad(), float64_casts():
        for index, ep in enumerate(data.episodes):
            x = cut["x"][index:index + 1]
            logits = trunk_logits(trunk, x, cut["t"][index:index + 1], cut["ty"][index:index + 1])
            if logits.dtype != torch.float64:
                raise RuntimeError(f"logits dtype {logits.dtype}")
            base = logits[0].detach().cpu().numpy()
            if mode == "baseline":
                used = np.array(base, dtype=np.float64, copy=True)
                names = ["trunk"] * N_MACHINES
            elif mode == "control":
                used = head(torch.tensor(base, dtype=torch.float64)).detach().cpu().numpy()
                names = ["shared_head"] * N_MACHINES
            else:
                used, names = apply_heads(base, cut["sig"][index].numpy(), spec, route)
            probs = normalize_rows(torch.softmax(torch.tensor(used, dtype=torch.float64), dim=-1).numpy())
            for row in index_queries(data).get(ep.episode_id, []):
                slot = ep.slot_of(row["machine_id"])
                sig = int(row["signal"])
                if int(cut["sig"][index, slot]) != sig:
                    raise RuntimeError("signal drift")
                preds[row["query_id"]] = [float(v) for v in probs[slot]]
                routes.append({
                    "query_id": row["query_id"],
                    "episode_id": ep.episode_id,
                    "signal": sig,
                    "model_used": names[slot],
                })
    return preds, routes


def _other_gpu():
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        free = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, None, str(exc)
    pids = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid = line.split(",")[0].strip()
        if pid.isdigit():
            pids.append(int(pid))
    return pids, int(free), None


def _blocker(msg: str) -> None:
    with open(os.path.join(ROOT, "BLOCKER.md"), "w", encoding="utf-8") as handle:
        handle.write("# BLOCKER\n\n" + msg.strip() + "\n")
    print("BLOCKER", msg, flush=True)


def train_seed0(out: str) -> dict:
    require_seed0(0)
    torch.set_num_threads(2)
    if os.path.exists(os.path.join(out, "tune_metrics.json")):
        raise SystemExit("refusing a second full run; tune_metrics.json exists")
    pids, free_mb, err = _other_gpu()
    if err or free_mb is None:
        _blocker(f"could not read GPU memory: {err}")
        raise SystemExit(4)
    if free_mb < 1536:
        _blocker(f"GPU free {free_mb} MiB is below the 1GiB slot plus margin; other pids {pids}")
        raise SystemExit(4)
    if not torch.cuda.is_available():
        _blocker("CUDA is not available. CPU training is refused.")
        raise SystemExit(3)
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(0.99, VRAM_BYTES / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    os.makedirs(out, exist_ok=True)
    device = "cuda"
    print(
        f"train_start pid={os.getpid()} device={device} name={torch.cuda.get_device_name(0)} "
        f"vram_frac={frac:.4f} free_mib={free_mb} other_pids={pids}",
        flush=True,
    )
    status(f"training started pid={os.getpid()} device={torch.cuda.get_device_name(0)} free_mib={free_mb} other_pids={pids}")
    t0 = time.time()
    trunk, ckpt = load_trunk()
    before = snapshot_frozen(trunk)
    tr = load_split(os.path.join(B.PUB, "train"), with_labels=True)
    counts = count_signal_rows(tr)
    if counts != EXPECTED_SIGNAL_ROWS:
        _blocker(f"signal counts {counts} != {EXPECTED_SIGNAL_ROWS}")
        raise SystemExit(2)
    route = route_table(counts)
    cut = build_cutoff(tr, W, require_labels=True)
    print("signal_rows", counts, "route", route, "cutoff", len(cut["x"]), "trunk_steps", ckpt["steps"], flush=True)
    torch.manual_seed(0)
    control = TinyHead()
    if trainable_count(control) != HEAD_PARAMS or trainable_count(trunk) != 0:
        _blocker("trainable count mismatch before training")
        raise SystemExit(2)
    ctrl_stats = train_head(control, trunk, cut, None, device)
    specialists = {}
    spec_stats = {}
    for sig in SIGNALS:
        if route[sig] != "specialist":
            print(f"fallback trunk signal {sig}", flush=True)
            continue
        torch.manual_seed(0)
        head = TinyHead()
        stats = train_head(head, trunk, cut, sig, device)
        if stats["steps"] != ctrl_stats["steps"]:
            _blocker(f"step mismatch {sig} {stats['steps']} vs {ctrl_stats['steps']}")
            raise SystemExit(2)
        specialists[sig] = head
        spec_stats[str(sig)] = stats
    if not frozen_equal(trunk, before):
        _blocker("frozen trunk weights or buffers changed")
        raise SystemExit(2)
    trunk_cpu = trunk.to("cpu").eval()
    va = load_split(os.path.join(B.PUB, "validation"), with_labels=False)
    heads = {"control": control, **specialists}
    base_preds, _ = predict_split(trunk_cpu, heads, route, va, W, "baseline")
    ctrl_preds, _ = predict_split(trunk_cpu, heads, route, va, W, "control")
    routed_preds, routes = predict_split(trunk_cpu, heads, route, va, W, "routed")
    for row in routes:
        expect = f"specialist_{row['signal']:+d}" if route[row["signal"]] == "specialist" else "trunk"
        if row["model_used"] != expect:
            raise RuntimeError(f"route {row} != {expect}")
    splits = json.load(open(os.path.join(ROOT, "SPLIT_IDS.json"), encoding="utf-8"))
    official = load_official_validation(os.path.join(B.PUB, "validation"))
    tune = set(splits["tune"])
    score = {
        "baseline": score_predictions(base_preds, official, tune),
        "control": score_predictions(ctrl_preds, official, tune),
        "routed": score_predictions(routed_preds, official, tune),
    }
    br, cr, rr = score["baseline"]["R"], score["control"]["R"], score["routed"]["R"]
    if any(v < 0 for stats in (ctrl_stats, *spec_stats.values()) for v in stats["losses"]):
        verdict, reason = "negative", "a train loss was negative"
    elif rr < br and rr < cr:
        verdict, reason = "promising", "routed tune R is below the frozen trunk and the shared head; stopping"
    else:
        verdict, reason = "negative", "routed tune R is not below both the frozen trunk and the shared head"
    metrics = {
        "seed": 0,
        "tune_R_baseline": br,
        "tune_R_control": cr,
        "tune_R_routed": rr,
        "tune": {k: v["R"] for k, v in score.items()},
        "tune_detail_routed": score["routed"],
        "verdict": verdict,
        "verdict_reason": reason,
        "trunk_sha256": TRUNK_SHA,
        "trunk_phase": ckpt["phase"],
        "trunk_steps": ckpt["steps"],
        "trunk_config": ckpt["config"],
        "trunk_unchanged": True,
        "trunk_trainable": 0,
        "head_params_each": HEAD_PARAMS,
        "signal_rows": {str(k): v for k, v in counts.items()},
        "route": {str(k): v for k, v in route.items()},
        "control": ctrl_stats,
        "specialists": spec_stats,
        "schedule": {"epochs": HEAD_EPOCHS, "lr": HEAD_LR, "wd": HEAD_WD, "bs": BS},
        "eval": "cpu_fp64_softmax_row_renorm",
        "seconds": time.time() - t0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "pid": os.getpid(),
        "device": torch.cuda.get_device_name(0),
    }
    json.dump(metrics, open(os.path.join(out, "tune_metrics.json"), "w"), indent=1)
    _write_results(metrics)
    print(
        f"RESULT seed 0 baseline {br:.6f} control {cr:.6f} routed {rr:.6f} "
        f"verdict {verdict} seconds {metrics['seconds']:.0f} peakMB {metrics['peak_gpu_mb']:.0f}",
        flush=True,
    )
    status(
        f"final seed0 baseline={br:.6f} control={cr:.6f} routed={rr:.6f} verdict={verdict} "
        f"seconds={metrics['seconds']:.0f} peak_gpu_mb={metrics['peak_gpu_mb']:.0f} device={metrics['device']}"
    )
    return metrics


def _write_results(metrics) -> None:
    dest = os.path.join(ROOT, "results")
    os.makedirs(dest, exist_ok=True)
    json.dump(metrics, open(os.path.join(dest, "results.json"), "w"), indent=1)
    lines = [
        "# frozen heads seed 0",
        "",
        f"Verdict: **{metrics['verdict']}**. {metrics['verdict_reason']}",
        "",
        "Tune R only, CPU float64 softmax with row renormalization. Lower is better.",
        "",
        f"- Frozen trunk baseline: {metrics['tune_R_baseline']:.6f}",
        f"- Matched shared head: {metrics['tune_R_control']:.6f}",
        f"- Routed signal heads: {metrics['tune_R_routed']:.6f}",
        "",
        f"Trunk unchanged: {metrics['trunk_unchanged']}. Trainable trunk params: {metrics['trunk_trainable']}. "
        f"Each head: {metrics['head_params_each']} params.",
        f"Schedule: {metrics['schedule']}.",
        f"Control final loss {metrics['control']['losses'][-1]:.4f} over {metrics['control']['steps']} steps.",
    ]
    for sig, stats in metrics["specialists"].items():
        lines.append(
            f"Specialist {sig} final loss {stats['losses'][-1]:.4f}, steps {stats['steps']}, "
            f"labeled slots {stats['labeled_slots']}."
        )
    lines += [
        "",
        f"Device {metrics['device']}. PID {metrics['pid']}. Peak GPU {metrics['peak_gpu_mb']:.0f} MiB. "
        f"Seconds {metrics['seconds']:.1f}.",
        "",
        "No further seed will be launched.",
        "",
    ]
    with open(os.path.join(dest, "RESULTS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    require_seed0(args.seed)
    train_seed0(args.out)


if __name__ == "__main__":
    main()
