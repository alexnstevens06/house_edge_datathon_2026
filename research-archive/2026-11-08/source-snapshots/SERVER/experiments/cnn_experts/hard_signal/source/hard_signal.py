"""Hard signal router on the CNN03 no-signal CNN.

History pretrain never reads the query signal. Each specialist fine-tunes only on
true TRAIN cutoff labels of one signal. The matched control uses the same
pretrain weights, seed, and fine-tune schedule with no router.

ADDENDUM (Alex t83) overrides promotion: this module trains seed 0 only.
It does not score confirm or test and it does not start another seed.
Imports of kit/ and experiments/mlp_cnn_hpo are read-only.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
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

# Frozen budget. See PLAN.md. Not chosen from a validation score.
H = 128
LAYERS = 4
KERNEL = 3
W = 64
DROP = 0.3
LR = 0.00197
WD = 0.00895
EPOCHS = 30
FT_EPOCHS = 10
BS = 512
MICRO_BS = 128
MIN_GROUP_ROWS = 64
MIN_T = 16
EXPECTED_PARAMS = 367109
EXPECTED_SIGNAL_ROWS = {-1: 250, 0: 267, 1: 251}
VRAM_BYTES = 2 * 1024 ** 3
SIGNALS = (-1, 0, 1)
INCUMBENT_TUNE_R = 0.33206398675779764

window = B.window
CNN = B.CNN


class Examples:
    """Window batch. `signals` is None for history and a routing tensor at cutoff."""

    def __init__(self, X, T, TY, Y, signals, cutoffs):
        self.X = X
        self.T = T
        self.TY = TY
        self.Y = Y
        self.signals = signals
        self.cutoffs = cutoffs

    def __len__(self):
        return int(self.X.shape[0])


def make_cnn(h=H, layers=LAYERS, drop=DROP, width=W, k=KERNEL):
    return CNN(h, layers, k, drop, width, use_signal=False)


def phase_hyper(phase: str) -> dict:
    common = {"lr": LR, "wd": WD, "bs": BS, "micro_bs": MICRO_BS}
    if phase == "pretrain":
        return {**common, "epochs": EPOCHS}
    if phase in ("control_finetune", "specialist_finetune"):
        return {**common, "epochs": FT_EPOCHS}
    raise ValueError(phase)


def require_seed0(seed) -> None:
    """Entry-point guard. ADDENDUM allows one full run, seed 0 only."""
    if int(seed) != 0:
        raise SystemExit(f"rejected seed {seed}: ADDENDUM allows one full run, seed 0 only")


def assert_contract_maps() -> None:
    assert tuple(MACHINE_TYPES) == ("balanced", "jackpot", "steady")
    assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}
    assert tuple(PAYOUTS) == (0, 1, 2, 5, 20)
    assert MISSING == -1 and MISSING != 0


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


def _stack_examples(xs, ts, tys, ys, sgs, cutoffs):
    signals = None if sgs is None else torch.tensor(np.stack(sgs))
    return Examples(
        torch.tensor(np.stack(xs)),
        torch.tensor(np.asarray(ts, dtype=np.int64)),
        torch.tensor(np.stack(tys)),
        torch.tensor(np.stack(ys).astype(np.int64, copy=False)),
        signals,
        torch.tensor(np.asarray(cutoffs, dtype=np.int64)),
    )


def build_history(data, width=W, min_t=MIN_T, stride=1):
    """Pseudo-cutoff windows with t < cutoff. Does not read queries or labels."""
    xs, ts, tys, ys, cutoffs = [], [], [], [], []
    for ep in data.episodes:
        obs = ep.obs
        last = min(obs.shape[0], ep.cutoff)
        for t in range(min_t, ep.cutoff, stride):
            if t >= last:
                continue
            y = obs[t]
            if int((y >= 0).sum()) == 0:
                continue
            xs.append(window(obs, t, width))
            ts.append(t)
            tys.append(ep.machine_types)
            ys.append(y)
            cutoffs.append(ep.cutoff)
    return _stack_examples(xs, ts, tys, ys, None, cutoffs)


def build_cutoff(data, width=W, require_labels=False):
    """One window per episode at the real cutoff. Signal is metadata, not a feature."""
    by = index_queries(data)
    xs, ts, tys, ys, sgs, cutoffs = [], [], [], [], [], []
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
        cutoffs.append(ep.cutoff)
    batch = _stack_examples(xs, ts, tys, ys, sgs, cutoffs)
    if require_labels:
        if data.query_labels is None or bool((batch.Y < 0).any()) or bool((batch.signals == 99).any()):
            raise RuntimeError("cutoff labels or signals are incomplete")
        bad = [int(s) for s in torch.unique(batch.signals).tolist() if int(s) not in SIGNALS]
        if bad:
            raise RuntimeError(f"cutoff signal outside {{-1,0,+1}}: {bad}")
    return batch


def require_train_labels(data):
    if not data.query_labels:
        return "train query_labels missing"
    missing = [r["query_id"] for r in data.queries if r["query_id"] not in data.query_labels]
    if missing:
        return f"{len(missing)} train queries have no label"
    for row in data.queries:
        try:
            signal = int(row["signal"])
        except (TypeError, ValueError):
            return "non-integer signal"
        if signal not in SIGNALS:
            return f"signal {signal} outside {{-1,0,+1}}"
    by = index_queries(data)
    for ep in data.episodes:
        if len(by.get(ep.episode_id, [])) != N_MACHINES:
            return f"{ep.episode_id} does not have 8 queries"
    return None


def count_signal_rows(data) -> dict:
    counts = {s: 0 for s in SIGNALS}
    for row in data.queries:
        if data.query_labels and row["query_id"] in data.query_labels:
            counts[int(row["signal"])] += 1
    return counts


def loss_mask(y, signals, mode, signal):
    mask = y.long() >= 0
    if mode in ("control", "pretrain"):
        return mask
    if mode == "specialist":
        if signals is None:
            raise RuntimeError("specialist loss mask needs cutoff signals")
        return mask & (signals.long() == int(signal))
    raise ValueError(mode)


def ce_loss(logits, y, mask, reduction="mean"):
    keep = mask.bool()
    if int(keep.sum()) == 0:
        return None
    return F.cross_entropy(logits.float()[keep], y.long()[keep], reduction=reduction)


def route_table(counts, min_rows=MIN_GROUP_ROWS) -> dict:
    return {s: ("specialist" if int(counts[s]) >= int(min_rows) else "shared") for s in SIGNALS}


def assemble_slots(shared, spec, signals, route):
    """Exact hard stitch of one episode. Probabilities are copied, not mixed."""
    out = np.empty(np.shape(shared), dtype=np.float64)
    names = []
    for slot, raw in enumerate(np.asarray(signals).tolist()):
        sig = int(raw)
        if route[sig] == "specialist":
            out[slot] = spec[sig][slot]
            names.append(f"specialist_{sig:+d}")
        else:
            out[slot] = shared[slot]
            names.append("shared_clean")
    return out, names


def _on_device(batch, device):
    def move(value):
        return value.to(device) if torch.is_tensor(value) else value
    return Examples(move(batch.X), move(batch.T), move(batch.TY), move(batch.Y), move(batch.signals), move(batch.cutoffs))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase(model, batch, epochs, lr, wd, bs, seed, signal, phase, device, micro=MICRO_BS):
    """AdamW + OneCycle. One optimizer step per `bs` rows, microbatch `micro` under the VRAM cap.

    The gradient is the mean over labeled slots in the effective batch. Dropout still
    draws a mask per microbatch, so this is not bitwise-identical to one 512-row forward.
    """
    if signal is not None and batch.signals is None:
        raise RuntimeError("specialist finetune requires cutoff signals")
    set_seed(seed)
    device = torch.device(device)
    model = model.to(device)
    batch = _on_device(batch, device)
    mode = "specialist" if signal is not None else "control"
    labeled = int(loss_mask(batch.Y, batch.signals, mode, signal).sum())
    n = len(batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_chunks = (n + bs - 1) // bs
    total_steps = epochs * n_chunks
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(total_steps, 1))
    losses = []
    steps = 0
    t0 = time.time()
    x = batch.X
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_sum = 0.0
        epoch_n = 0
        for start in range(0, n, bs):
            chunk = perm[start:start + bs]
            opt.zero_grad(set_to_none=True)
            total_n = 0
            for k in range(0, len(chunk), micro):
                j = chunk[k:k + micro]
                slot_mask = loss_mask(batch.Y[j], None if batch.signals is None else batch.signals[j], mode, signal)
                with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                    logits = model(
                        x[j], batch.T[j], batch.TY[j],
                        torch.zeros(j.shape[0], N_MACHINES, dtype=torch.long, device=device),
                    )
                sum_loss = ce_loss(logits, batch.Y[j], slot_mask, reduction="sum")
                if sum_loss is None:
                    continue
                sum_loss.backward()
                nlab = int(slot_mask.sum())
                total_n += nlab
                epoch_sum += float(sum_loss.detach())
                epoch_n += nlab
            if total_n == 0:
                raise RuntimeError(f"{phase} epoch {ep} had an empty labeled chunk")
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(total_n)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            steps += 1
        losses.append(epoch_sum / max(epoch_n, 1))
        print(f"epoch {ep} loss {losses[-1]:.4f} t {time.time() - t0:.0f}s phase={phase}", flush=True)
        if ep == 0 and phase == "pretrain":
            status(f"first epoch pretrain loss {losses[-1]:.6f} pid={os.getpid()}")
    return {"losses": losses, "steps": steps, "windows": n, "labeled_slots": labeled, "phase": phase}


def _cpu_probs(model, data, width):
    model = model.to("cpu").float().eval()
    cut = build_cutoff(data, width, require_labels=False)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(cut), 64):
            stop = start + 64
            x = cut.X[start:stop]
            logits = model(
                x, cut.T[start:stop], cut.TY[start:stop],
                torch.zeros(x.shape[0], N_MACHINES, dtype=torch.long),
            )
            pieces.append(torch.softmax(logits.float(), -1))
    return torch.cat(pieces).double().numpy(), cut


def _row_list(vec):
    return [float(v) for v in vec]


def predict_single(model, data, width):
    probs, _ = _cpu_probs(model, data, width)
    preds = {}
    for row in data.queries:
        index = data.episode_index[row["episode_id"]]
        ep = data.episodes[index]
        preds[row["query_id"]] = _row_list(probs[index, ep.slot_of(row["machine_id"])])
    return preds


def predict_routed(models, route, data, width):
    shared_p, cut = _cpu_probs(models["shared"], data, width)
    spec_p = {}
    for sig in SIGNALS:
        if route.get(sig) == "specialist":
            if sig not in models["specialists"]:
                raise RuntimeError(f"missing specialist for signal {sig}")
            spec_p[sig], _ = _cpu_probs(models["specialists"][sig], data, width)
    preds = {}
    routes = []
    for row in data.queries:
        index = data.episode_index[row["episode_id"]]
        ep = data.episodes[index]
        slot = ep.slot_of(row["machine_id"])
        sig = int(row["signal"])
        if int(cut.signals[index, slot]) != sig:
            raise RuntimeError(f"cutoff signal drifted for {row['query_id']}")
        if route.get(sig) == "specialist":
            vec = spec_p[sig][index, slot]
            used = f"specialist_{sig:+d}"
        else:
            vec = shared_p[index, slot]
            used = "shared_clean"
        preds[row["query_id"]] = _row_list(vec)
        routes.append({
            "query_id": row["query_id"],
            "episode_id": row["episode_id"],
            "signal": sig,
            "model_used": used,
        })
    return preds, routes


def _other_gpu_pids():
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, str(exc)
    pids = []
    for line in text.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids, None


def _write_blocker(msg: str) -> None:
    with open(os.path.join(ROOT, "BLOCKER.md"), "w", encoding="utf-8") as handle:
        handle.write("# BLOCKER\n\n" + msg.strip() + "\n")
    print("BLOCKER", msg, flush=True)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state(state, device):
    model = make_cnn().to(device)
    model.load_state_dict(state)
    return model


def _save(path, model, extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": _cpu_state(model), **extra}, path)


def train_seed0(out: str) -> dict:
    """One full seed-0 run: 30-epoch history pretrain, 10-epoch control, 10-epoch specialists."""
    require_seed0(0)
    torch.set_num_threads(2)
    assert_contract_maps()
    if os.path.exists(os.path.join(out, "tune_metrics.json")):
        raise SystemExit(f"refusing a second full run; {out}/tune_metrics.json already exists")
    others, gpu_err = _other_gpu_pids()
    if gpu_err is not None or others is None:
        _write_blocker(f"could not confirm the GPU is free: {gpu_err}")
        raise SystemExit(4)
    if others:
        _write_blocker(f"GPU already has compute pid(s) {others}; not starting a competing job")
        raise SystemExit(4)
    if not torch.cuda.is_available():
        _write_blocker("CUDA is not available in this process. Full training was not started.")
        raise SystemExit(3)
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(0.99, VRAM_BYTES / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    os.makedirs(out, exist_ok=True)
    print(
        f"train_start pid={os.getpid()} device=cuda seed=0 vram_frac={frac:.4f} "
        f"micro={MICRO_BS} name={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    status(f"training started pid={os.getpid()} out={out} vram_frac={frac:.4f}")
    t0 = time.time()
    tr = load_split(os.path.join(B.PUB, "train"), with_labels=True)
    err = require_train_labels(tr)
    if err:
        _write_blocker(err)
        raise SystemExit(2)
    counts = count_signal_rows(tr)
    if counts != EXPECTED_SIGNAL_ROWS:
        _write_blocker(f"signal row counts {counts} != {EXPECTED_SIGNAL_ROWS}")
        raise SystemExit(2)
    route = route_table(counts, MIN_GROUP_ROWS)
    print("signal_rows", counts, "route", route, flush=True)
    hist = build_history(tr, W)
    if hist.signals is not None or bool((hist.T >= hist.cutoffs).any()):
        _write_blocker("history batch contains a signal or a non-causal window")
        raise SystemExit(2)
    cut = build_cutoff(tr, W, require_labels=True)
    print("history_windows", len(hist), "cutoff_episodes", len(cut), "build_s", round(time.time() - t0, 3), flush=True)
    set_seed(0)
    model = make_cnn()
    nparams = sum(p.numel() for p in model.parameters())
    print("params", nparams, flush=True)
    if nparams != EXPECTED_PARAMS or model.sig is not None or model.use_signal:
        _write_blocker(f"clean CNN params/signal embedding mismatch: {nparams}")
        raise SystemExit(2)
    pre_h = phase_hyper("pretrain")
    ft_h = phase_hyper("control_finetune")
    if ft_h != phase_hyper("specialist_finetune") or ft_h["lr"] != pre_h["lr"] or ft_h["epochs"] != FT_EPOCHS:
        raise SystemExit("fine-tune budget drifted from the frozen plan")
    pre = train_phase(
        model, hist, epochs=pre_h["epochs"], lr=pre_h["lr"], wd=pre_h["wd"], bs=pre_h["bs"],
        seed=0, signal=None, phase="pretrain", device="cuda", micro=pre_h["micro_bs"],
    )
    shared_cpu = _cpu_state(model)
    _save(os.path.join(out, "pretrain", "checkpoint.pt"), model, {
        "phase": "pretrain", "seed": 0, "losses": pre["losses"], "steps": pre["steps"], "config": pre_h,
    })
    del hist
    model.to("cpu")
    torch.cuda.empty_cache()

    control = _load_state(shared_cpu, "cuda")
    ctrl = train_phase(
        control, cut, epochs=ft_h["epochs"], lr=ft_h["lr"], wd=ft_h["wd"], bs=ft_h["bs"],
        seed=0, signal=None, phase="control", device="cuda", micro=ft_h["micro_bs"],
    )
    control_cpu = _cpu_state(control)
    _save(os.path.join(out, "control", "checkpoint.pt"), control, {
        "phase": "control", "seed": 0, "losses": ctrl["losses"], "steps": ctrl["steps"], "config": ft_h,
    })
    del control
    torch.cuda.empty_cache()

    spec_stats = {}
    spec_states = {}
    for sig in SIGNALS:
        if route[sig] != "specialist":
            print(f"fallback signal {sig} rows {counts[sig]}", flush=True)
            continue
        spec = _load_state(shared_cpu, "cuda")
        phase = f"specialist_{sig:+d}"
        stats = train_phase(
            spec, cut, epochs=ft_h["epochs"], lr=ft_h["lr"], wd=ft_h["wd"], bs=ft_h["bs"],
            seed=0, signal=sig, phase=phase, device="cuda", micro=ft_h["micro_bs"],
        )
        if stats["steps"] != ctrl["steps"] or stats["windows"] != ctrl["windows"]:
            _write_blocker(
                f"unfair step budget {phase} steps={stats['steps']} windows={stats['windows']} "
                f"control steps={ctrl['steps']} windows={ctrl['windows']}"
            )
            raise SystemExit(2)
        spec_stats[str(sig)] = stats
        spec_states[sig] = _cpu_state(spec)
        _save(os.path.join(out, phase, "checkpoint.pt"), spec, {
            "phase": phase, "signal": sig, "seed": 0, "losses": stats["losses"],
            "steps": stats["steps"], "labeled_slots": stats["labeled_slots"], "config": ft_h,
        })
        del spec
        torch.cuda.empty_cache()

    losses_all = list(pre["losses"]) + list(ctrl["losses"])
    for stats in spec_stats.values():
        losses_all.extend(stats["losses"])
    negative_loss = any(float(v) < 0 for v in losses_all)

    va = load_split(os.path.join(B.PUB, "validation"), with_labels=False)
    shared = _load_state(shared_cpu, "cpu")
    specialists = {sig: _load_state(state, "cpu") for sig, state in spec_states.items()}
    routed_preds, routes = predict_routed({"shared": shared, "specialists": specialists}, route, va, W)
    control_preds = predict_single(_load_state(control_cpu, "cpu"), va, W)
    splits = json.load(open(os.path.join(ROOT, "SPLIT_IDS.json"), encoding="utf-8"))
    tune_ids = set(splits["tune"])
    official = load_official_validation(os.path.join(B.PUB, "validation"))
    tune_routed = score_predictions(routed_preds, official, tune_ids)
    tune_control = score_predictions(control_preds, official, tune_ids)
    for row in routes:
        expect = f"specialist_{row['signal']:+d}" if route[row["signal"]] == "specialist" else "shared_clean"
        if row["model_used"] != expect:
            raise RuntimeError(f"route {row} != {expect}")
    tune_routes = [row for row in routes if row["episode_id"] in tune_ids]
    with open(os.path.join(out, "route_log_tune.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("query_id", "episode_id", "signal", "model_used"))
        writer.writeheader()
        writer.writerows(tune_routes)

    beats_control = bool(tune_routed["R"] < tune_control["R"])
    beats_incumbent = bool(tune_routed["R"] < INCUMBENT_TUNE_R)
    if negative_loss:
        verdict = "negative"
        verdict_reason = "a recorded train loss was negative; no further run"
    elif beats_control and beats_incumbent:
        verdict = "promising"
        verdict_reason = "routed tune R is lower than the matched control and CNN03 seed 0; stopping anyway"
    else:
        verdict = "negative"
        verdict_reason = "routed tune R did not beat both the matched control and CNN03 seed 0"
    metrics = {
        "seed": 0,
        "tune_R_routed": tune_routed["R"],
        "tune_R_control": tune_control["R"],
        "tune_routed": tune_routed,
        "tune_control": tune_control,
        "incumbent_tune_R": INCUMBENT_TUNE_R,
        "incumbent_source": "experiments/mlp_cnn_hpo/runs/final_cnn03_s0/metrics.json tune.R",
        "eval_dtype": "cpu_fp32_softmax_stored_float64",
        "train_autocast": "cuda_bf16_weights_fp32_loss_fp32",
        "beats_matched_control": beats_control,
        "beats_incumbent_tune": beats_incumbent,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "negative_train_loss": negative_loss,
        "signal_rows": {str(k): v for k, v in counts.items()},
        "route": {str(k): v for k, v in route.items()},
        "params": nparams,
        "pretrain": pre,
        "control_finetune": ctrl,
        "specialists": spec_stats,
        "row_exposure_note": (
            "Optimizer steps and windows match across control and each specialist. "
            "Labeled-slot exposure does not: control uses every cutoff label; "
            "each specialist uses only its signal. That difference is the intervention."
        ),
        "not_cnn03_reproduction": (
            "CNN03 trains 30 epochs on history windows plus the cutoff window in one loop, "
            "batch 512, memory fraction 0.25, and scores full/confirm/tune. This run holds "
            "cutoff rows out of pretrain, fine-tunes 10 epochs, microbatches 128 inside a "
            "2GiB cap, and scores tune only."
        ),
        "seconds": time.time() - t0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "pid": os.getpid(),
    }
    json.dump(metrics, open(os.path.join(out, "tune_metrics.json"), "w"), indent=1)
    prov = {
        "hard_signal_py": sha256_file(os.path.join(HERE, "hard_signal.py")),
        "run_seed_py": sha256_file(os.path.join(HERE, "run_seed.py")),
        "test_py": sha256_file(os.path.join(ROOT, "tests", "test_hard_signal.py")),
        "base_cnn_py": sha256_file(os.path.join(REPO, "experiments", "mlp_cnn_hpo", "source", "base_cnn.py")),
        "train_py": sha256_file(os.path.join(REPO, "experiments", "mlp_cnn_hpo", "source", "train.py")),
        "plan_md": sha256_file(os.path.join(ROOT, "PLAN.md")),
        "split_ids": sha256_file(os.path.join(ROOT, "SPLIT_IDS.json")),
        "addendum_md": sha256_file(os.path.join(REPO, "experiments", "cnn_experts", "ADDENDUM.md")),
        "seed": 0,
    }
    json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=1)
    _write_results(metrics, prov)
    print(
        f"RESULT seed 0 tune_R_routed {tune_routed['R']:.6f} "
        f"tune_R_control {tune_control['R']:.6f} incumbent {INCUMBENT_TUNE_R:.6f} "
        f"verdict {verdict} seconds {metrics['seconds']:.0f}",
        flush=True,
    )
    status(
        f"final seed0 tune_R_routed={tune_routed['R']:.6f} tune_R_control={tune_control['R']:.6f} "
        f"incumbent={INCUMBENT_TUNE_R:.6f} verdict={verdict} seconds={metrics['seconds']:.0f} "
        f"peak_gpu_mb={metrics['peak_gpu_mb']:.0f}"
    )
    return metrics


def _write_results(metrics, prov) -> None:
    dest = os.path.join(ROOT, "results")
    os.makedirs(dest, exist_ok=True)
    json.dump({"metrics": metrics, "provenance": prov}, open(os.path.join(dest, "results.json"), "w"), indent=1)
    pre_loss = metrics["pretrain"]["losses"][-1]
    ctrl_loss = metrics["control_finetune"]["losses"][-1]
    spec_lines = []
    for sig, stats in metrics["specialists"].items():
        spec_lines.append(
            f"- specialist {sig}: final train loss {stats['losses'][-1]:.4f}, "
            f"steps {stats['steps']}, labeled slots {stats['labeled_slots']}"
        )
    lines = [
        "# hard_signal seed 0",
        "",
        f"Verdict: **{metrics['verdict']}**. {metrics['verdict_reason']}",
        "",
        "Tune R only (lower is better). Confirm and test were not scored.",
        "",
        f"- Routed specialists tune R: {metrics['tune_R_routed']:.6f}",
        f"- Matched control tune R: {metrics['tune_R_control']:.6f}",
        f"- Canonical CNN03 seed 0 tune R: {metrics['incumbent_tune_R']:.6f}",
        f"- Eval: {metrics['eval_dtype']}. Train: {metrics['train_autocast']}.",
        "",
        "Train loss is mean cross-entropy over the labeled slots in each epoch.",
        "",
        f"- Pretrain final loss {pre_loss:.4f} over {metrics['pretrain']['steps']} steps "
        f"and {metrics['pretrain']['windows']} history windows.",
        f"- Control final loss {ctrl_loss:.4f} over {metrics['control_finetune']['steps']} steps, "
        f"labeled slots {metrics['control_finetune']['labeled_slots']}.",
        *spec_lines,
        "",
        metrics["row_exposure_note"],
        "",
        metrics["not_cnn03_reproduction"],
        "",
        f"Wall seconds {metrics['seconds']:.1f}. Peak GPU {metrics['peak_gpu_mb']:.0f} MiB. PID {metrics['pid']}.",
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
    ap.add_argument("--report-seen-confirm", action="store_true")
    args = ap.parse_args(argv)
    require_seed0(args.seed)
    if args.report_seen_confirm:
        raise SystemExit("rejected --report-seen-confirm: ADDENDUM forbids confirm/test scoring")
    train_seed0(args.out)


if __name__ == "__main__":
    main()
