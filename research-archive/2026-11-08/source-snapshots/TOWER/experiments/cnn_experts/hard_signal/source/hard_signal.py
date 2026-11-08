"""Hard signal router: CNN03 specialists on true TRAIN cutoff rows only.

Historical windows never see the query signal. The signal is a routing key
for fine-tune masks and for prediction, not a feature.
"""
from __future__ import annotations

import os
import sys
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from house_edge.data import MISSING, N_MACHINES, PAYOUTS, TYPE_TO_IDX, load_split  # noqa: E402

assert PAYOUTS == (0, 1, 2, 5, 20)
assert MISSING == -1 and MISSING != 0
assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}

PUB = os.path.join(REPO, "kit", "public")
SIGNALS = (-1, 0, 1)
INCUMBENT_TUNE_R = {
    0: 0.33206398675779764,
    1: 0.34654552298986674,
    2: 0.33342647148315785,
}

# Frozen in PLAN.md. Effective batch is bs; micro_bs is only the 2 GiB accumulation width.
CFG = {
    "h": 128,
    "layers": 4,
    "k": 3,
    "drop": 0.3,
    "W": 64,
    "lr": 0.00197,
    "wd": 0.00895,
    "bs": 512,
    "micro_bs": 32,
    "pretrain_epochs": 30,
    "finetune_epochs": 30,
    "min_t": 16,
    "stride": 1,
    "min_group_rows": 32,
    "grad_clip": 1.0,
    "use_signal": False,
}


class Block(nn.Module):
    def __init__(s, h, d, k, drop):
        super().__init__()
        s.pad = (k - 1) * d
        s.conv = nn.Conv1d(h, h, k, dilation=d)
        s.mix = nn.Linear(2 * h, h)
        s.n1 = nn.LayerNorm(h)
        s.n2 = nn.LayerNorm(h)
        s.dr = nn.Dropout(drop)

    def forward(s, z):  # z: (B,8,W,H)
        B, M, W, H = z.shape
        c = s.conv(F.pad(z.reshape(B * M, W, H).transpose(1, 2), (s.pad, 0))).transpose(1, 2).reshape(B, M, W, H)
        z = s.n1(z + s.dr(F.gelu(c)))
        others = (z.sum(1, keepdim=True) - z) / (M - 1)
        return s.n2(z + s.dr(F.gelu(s.mix(torch.cat([z, others], -1)))))


class CNN(nn.Module):
    """CNN03 block stack. use_signal must stay False for this experiment."""

    def __init__(s, h=64, layers=4, k=3, drop=0.1, W=64, use_signal=False):
        super().__init__()
        s.W = W
        s.use_signal = use_signal
        s.pay = nn.Embedding(6, h)  # 0..4 payouts, 5 = missing
        s.mask = nn.Linear(1, h)
        s.typ = nn.Embedding(3, h)
        s.sig = nn.Embedding(3, h) if use_signal else None
        s.slot = nn.Embedding(8, h)
        s.tick = nn.Linear(2, h)
        s.blocks = nn.ModuleList([Block(h, 2 ** i, k, drop) for i in range(layers)])
        s.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 5))

    def forward(s, x, t, ty, sg):
        B, W, M = x.shape
        x = x.long().transpose(1, 2)  # (B,8,W)
        m = (x >= 0).float().unsqueeze(-1)
        idx = torch.where(x >= 0, x, torch.full_like(x, 5))
        ticks = (t.view(B, 1, 1).float() - W + torch.arange(W, device=x.device).view(1, 1, W)).expand(B, M, W)
        tf = torch.stack([ticks / 256.0, (t.view(B, 1, 1).float() - ticks) / W], -1)
        meta = s.typ(ty) + s.slot.weight[None]
        if s.use_signal:
            meta = meta + s.sig(sg)
        z = s.pay(idx) + s.mask(m) + s.tick(tf) + meta.unsqueeze(2)
        for b in s.blocks:
            z = b(z)
        last = z[:, :, -1]
        pooled = last.mean(1, keepdim=True).expand_as(last)
        return s.head(torch.cat([last, pooled], -1))


def new_cnn03():
    m = CNN(CFG["h"], CFG["layers"], CFG["k"], CFG["drop"], CFG["W"], use_signal=False)
    n = sum(p.numel() for p in m.parameters())
    if n != 367109:
        raise RuntimeError(f"CNN03 no-signal param count {n} != 367109")
    if any("sig" in name for name, _ in m.named_parameters()):
        raise RuntimeError("clean CNN has signal parameters")
    return m


def copy_model(model):
    m = new_cnn03()
    m.load_state_dict(model.state_dict())
    return m


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def window(obs, t, W):
    """Features use ONLY ticks < t on all machines. (W, 8) payout idx, MISSING pad.

    Copied from CNN03 base_cnn.window so the right edge is tick t-1.
    """
    x = np.full((W, N_MACHINES), MISSING, np.int8)
    lo = max(0, t - W)
    seg = obs[lo:min(t, obs.shape[0])]
    if len(seg):
        x[W - (t - lo):W - (t - lo) + len(seg)] = seg
    return x


class History(NamedTuple):
    X: torch.Tensor
    T: torch.Tensor
    TY: torch.Tensor
    Y: torch.Tensor


class Cutoff(NamedTuple):
    X: torch.Tensor
    T: torch.Tensor
    TY: torch.Tensor
    Y: torch.Tensor
    signals: torch.Tensor
    episode_ids: list


def _stack_windows(xs, ts, tys, ys, W):
    if not xs:
        return (
            torch.zeros(0, W, N_MACHINES, dtype=torch.int8),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, N_MACHINES, dtype=torch.long),
            torch.zeros(0, N_MACHINES, dtype=torch.long),
        )
    return (
        torch.from_numpy(np.stack(xs)),
        torch.tensor(ts, dtype=torch.long),
        torch.from_numpy(np.stack(tys).astype(np.int64, copy=False)),
        torch.from_numpy(np.stack(ys).astype(np.int64, copy=False)),
    )


def signals_by_episode(data):
    """episode_id -> (8,) signal in {-1,0,1} from that machine's own query row."""
    out = {}
    seen = {}
    for r in data.queries:
        ep = data.episodes[data.episode_index[r["episode_id"]]]
        slot = ep.slot_of(r["machine_id"])
        eid = r["episode_id"]
        arr = out.setdefault(eid, np.full(N_MACHINES, 99, np.int64))
        got = seen.setdefault(eid, np.zeros(N_MACHINES, np.int8))
        if got[slot]:
            raise RuntimeError(f"duplicate query for {eid} slot {slot}")
        sig = int(r["signal"])
        if sig not in SIGNALS:
            raise RuntimeError(f"unexpected signal {sig}")
        arr[slot] = sig
        got[slot] = 1
    for eid, got in seen.items():
        if int(got.sum()) != N_MACHINES:
            raise RuntimeError(f"{eid} is missing a query signal")
    return out


def build_history(data, W=None, min_t=None, stride=None):
    """Pseudo-cutoffs t < cutoff. Labels are history payouts. No query signal, no query label."""
    W = CFG["W"] if W is None else W
    min_t = CFG["min_t"] if min_t is None else min_t
    stride = CFG["stride"] if stride is None else stride
    xs, ts, tys, ys = [], [], [], []
    for ep in data.episodes:
        if ep.cutoff is None:
            raise RuntimeError(f"{ep.episode_id} has no cutoff")
        obs = ep.obs
        for t in range(min_t, ep.cutoff, stride):
            if t < obs.shape[0]:
                y = obs[t].astype(np.int64, copy=True)
            else:
                y = np.full(N_MACHINES, MISSING, np.int64)
            if (y >= 0).sum() == 0:
                continue
            if (y[y >= 0] > 4).any():
                raise RuntimeError("history label is not a payout index")
            xs.append(window(obs, t, W))
            ts.append(int(t))
            tys.append(ep.machine_types)
            ys.append(y)
            if ts[-1] >= ep.cutoff:
                raise RuntimeError("history window reached the cutoff")
    X, T, TY, Y = _stack_windows(xs, ts, tys, ys, W)
    return History(X, T, TY, Y)


def build_cutoff(data, W=None):
    """One window per episode at the true cutoff. Labels are query_labels only."""
    if not data.query_labels:
        raise RuntimeError("cutoff fine-tune requires query labels")
    W = CFG["W"] if W is None else W
    sigs = signals_by_episode(data)
    by_ep = {}
    for r in data.queries:
        by_ep.setdefault(r["episode_id"], []).append(r)
    xs, ts, tys, ys, sg, eids = [], [], [], [], [], []
    for ep in data.episodes:
        y = np.full(N_MACHINES, MISSING, np.int64)
        for r in by_ep.get(ep.episode_id, []):
            qid = r["query_id"]
            if qid not in data.query_labels:
                raise RuntimeError(f"missing label {qid}")
            y[ep.slot_of(r["machine_id"])] = int(data.query_labels[qid])
        if (y[y >= 0] > 4).any():
            raise RuntimeError("cutoff label is not a payout index")
        xs.append(window(ep.obs, ep.cutoff, W))
        ts.append(int(ep.cutoff))
        tys.append(ep.machine_types)
        ys.append(y)
        sg.append(sigs[ep.episode_id])
        eids.append(ep.episode_id)
    X, T, TY, Y = _stack_windows(xs, ts, tys, ys, W)
    signals = torch.from_numpy(np.stack(sg).astype(np.int64, copy=False)) if sg else torch.zeros(0, N_MACHINES, dtype=torch.long)
    return Cutoff(X, T, TY, Y, signals, eids)


def build_eval(data, W=None):
    """Query-time windows. Signal is returned beside X and is not written into X.

    Y is entirely missing so this tensor cannot be used as a training target.
    """
    W = CFG["W"] if W is None else W
    sigs = signals_by_episode(data)
    xs, ts, tys, eids = [], [], [], []
    for ep in data.episodes:
        if ep.cutoff is None:
            raise RuntimeError(f"{ep.episode_id} has no cutoff")
        xs.append(window(ep.obs, ep.cutoff, W))
        ts.append(int(ep.cutoff))
        tys.append(ep.machine_types)
        eids.append(ep.episode_id)
    if not eids:
        X, T, TY, Y = _stack_windows([], [], [], [], W)
        return Cutoff(X, T, TY, Y, torch.zeros(0, N_MACHINES, dtype=torch.long), [])
    ys = [np.full(N_MACHINES, MISSING, np.int64) for _ in eids]
    X, T, TY, Y = _stack_windows(xs, ts, tys, ys, W)
    signals = torch.from_numpy(np.stack([sigs[e] for e in eids]).astype(np.int64, copy=False))
    return Cutoff(X, T, TY, Y, signals, eids)


def count_labeled(signals, y):
    signals = np.asarray(signals)
    y = np.asarray(y)
    return {s: int(((signals == s) & (y >= 0)).sum()) for s in SIGNALS}


def active_map(counts, min_rows=None):
    min_rows = CFG["min_group_rows"] if min_rows is None else min_rows
    return {s: int(counts[s]) >= int(min_rows) for s in SIGNALS}


def apply_signal_mask(y, signals, signal):
    """signal is None -> keep every legitimate label (control). Else keep one signal."""
    if signal is None:
        return y.clone()
    out = torch.full_like(y, -1)
    m = (signals == int(signal)) & (y >= 0)
    out[m] = y[m]
    return out


def masked_ce_sum(lg, y):
    yl = y.long()
    m = yl >= 0
    n = int(m.sum().item())
    if n == 0:
        return lg.sum() * 0.0, 0
    return F.cross_entropy(lg.float()[m], yl[m], reduction="sum"), n


def seed_all(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zeros_sg(n, device):
    return torch.zeros(n, N_MACHINES, dtype=torch.long, device=device)


def train_epochs(model, X, T, TY, Y, epochs, signals=None, signal=None, bs=None, micro_bs=None, lr=None, wd=None, device="cpu"):
    """OneCycle over effective batches of `bs`. Micro-batches accumulate to that mean CE."""
    bs = CFG["bs"] if bs is None else bs
    micro_bs = CFG["micro_bs"] if micro_bs is None else micro_bs
    lr = CFG["lr"] if lr is None else lr
    wd = CFG["wd"] if wd is None else wd
    epochs = int(epochs)
    n = int(len(X))
    if epochs <= 0 or n == 0:
        return [], 0
    device = torch.device(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per = (n + bs - 1) // bs
    total_steps = epochs * steps_per
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total_steps)
    Xd, Td, TYd, Yd = (t.to(device) for t in (X, T, TY, Y))
    Sd = signals.to(device) if signals is not None else None
    use_cuda = device.type == "cuda"
    losses = []
    steps = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        sum_ce = 0.0
        n_lab = 0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            yb = Yd[idx]
            if signal is not None:
                yb = torch.where(Sd[idx] == int(signal), yb, torch.full_like(yb, -1))
            nlab = int((yb >= 0).sum().item())
            opt.zero_grad(set_to_none=True)
            if nlab == 0:
                sch.step()
                steps += 1
                continue
            ce_acc = 0.0
            for m0 in range(0, len(idx), micro_bs):
                j = idx[m0:m0 + micro_bs]
                with torch.autocast("cuda", torch.bfloat16, enabled=use_cuda):
                    lg = model(Xd[j], Td[j], TYd[j], _zeros_sg(len(j), device))
                ce, _ = masked_ce_sum(lg, yb[m0:m0 + micro_bs])
                (ce / nlab).backward()
                ce_acc += float(ce.detach().item())
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            sch.step()
            steps += 1
            sum_ce += ce_acc
            n_lab += nlab
        loss = sum_ce / max(n_lab, 1)
        losses.append(loss)
        print(f"phase signal={signal} epoch {ep} loss {loss:.4f} steps {steps}", flush=True)
    model.to("cpu")
    if use_cuda:
        torch.cuda.empty_cache()
    return losses, steps


def fit(data, seed, pretrain_epochs=None, finetune_epochs=None, min_group_rows=None, bs=None, micro_bs=None, device="cpu"):
    """Pretrain on history, then fine-tune control and specialists from that snapshot.

    No validation argument and no scoring. device is cpu unless the server trainer passes cuda.
    """
    pretrain_epochs = CFG["pretrain_epochs"] if pretrain_epochs is None else pretrain_epochs
    finetune_epochs = CFG["finetune_epochs"] if finetune_epochs is None else finetune_epochs
    min_group_rows = CFG["min_group_rows"] if min_group_rows is None else min_group_rows
    t0 = time.perf_counter()
    hist = build_history(data)
    cut = build_cutoff(data)
    counts = count_labeled(cut.signals, cut.Y)
    active = active_map(counts, min_group_rows)
    print(f"windows history {len(hist.X)} cutoff {len(cut.X)} counts {counts} active {active}", flush=True)

    seed_all(seed)
    shared = new_cnn03().to(device)
    pre_losses, pre_steps = train_epochs(
        shared, hist.X, hist.T, hist.TY, hist.Y, pretrain_epochs, bs=bs, micro_bs=micro_bs, device=device,
    )
    snap = snapshot(shared)
    del shared
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    shared_m = new_cnn03()
    shared_m.load_state_dict(snap)
    shared_m.eval()

    control = new_cnn03()
    control.load_state_dict(snap)
    seed_all(seed + 1)  # after the copy, so fine-tune shuffle does not depend on init RNG
    control = control.to(device)
    ctrl_losses, ctrl_steps = train_epochs(
        control, cut.X, cut.T, cut.TY, cut.Y, finetune_epochs,
        signals=cut.signals, signal=None, bs=bs, micro_bs=micro_bs, device=device,
    )
    control = control.cpu().eval()

    specialists = {}
    spec_losses = {}
    spec_steps = {}
    for i, s in enumerate(SIGNALS):
        if not active[s]:
            specialists[s] = None
            spec_losses[s] = []
            spec_steps[s] = 0
            print(f"fallback signal {s} rows {counts[s]} < {min_group_rows}", flush=True)
            continue
        m = new_cnn03()
        m.load_state_dict(snap)
        seed_all(seed + 2 + i)  # -1 -> seed+2, 0 -> seed+3, +1 -> seed+4
        m = m.to(device)
        losses, steps = train_epochs(
            m, cut.X, cut.T, cut.TY, cut.Y, finetune_epochs,
            signals=cut.signals, signal=s, bs=bs, micro_bs=micro_bs, device=device,
        )
        specialists[s] = m.cpu().eval()
        spec_losses[s] = losses
        spec_steps[s] = steps

    return {
        "shared": shared_m,
        "control": control,
        "specialists": specialists,
        "active": active,
        "counts": counts,
        "losses": {"pretrain": pre_losses, "control": ctrl_losses, "specialist": spec_losses},
        "steps": {"pretrain": pre_steps, "control": ctrl_steps, "specialist": spec_steps},
        "n_history": int(len(hist.X)),
        "n_cutoff": int(len(cut.X)),
        "seconds_fit": time.perf_counter() - t0,
        "cutoff_episode_ids": cut.episode_ids,
    }


@torch.no_grad()
def predict_all(model, X, T, TY, bs=64):
    """CPU fp32 probabilities (N, 8, 5). Does not read a signal tensor."""
    model = model.cpu().eval()
    n = int(len(X))
    out = np.empty((n, N_MACHINES, 5), np.float64)
    if n == 0:
        return out
    for i in range(0, n, bs):
        xb = X[i:i + bs]
        tb = T[i:i + bs]
        tyb = TY[i:i + bs]
        lg = model(xb, tb, tyb, _zeros_sg(len(xb), xb.device))
        out[i:i + bs] = torch.softmax(lg.float(), -1).double().numpy()
    return out


def specialist_name(sig):
    return {-1: "specialist_-1", 0: "specialist_0", 1: "specialist_+1"}[int(sig)]


@torch.no_grad()
def predict_routed(specialists, shared, active, X, T, TY, signals, bs=64):
    """Hard route: each slot is exactly one model's probability vector."""
    signals_np = signals.detach().cpu().numpy() if torch.is_tensor(signals) else np.asarray(signals)
    n = int(len(X))
    slot_probs = np.empty((n, N_MACHINES, 5), np.float64)
    used = np.empty((n, N_MACHINES), dtype=object)
    cache = {}

    def probs_for(key):
        if key not in cache:
            model = shared if key == "shared" else specialists[key]
            if model is None:
                raise RuntimeError(f"router selected missing model {key}")
            cache[key] = predict_all(model, X, T, TY, bs=bs)
        return cache[key]

    for i in range(n):
        for slot in range(N_MACHINES):
            sig = int(signals_np[i, slot])
            if sig not in SIGNALS:
                raise RuntimeError(f"bad signal {sig}")
            if active[sig]:
                slot_probs[i, slot] = probs_for(sig)[i, slot]
                used[i, slot] = specialist_name(sig)
            else:
                slot_probs[i, slot] = probs_for("shared")[i, slot]
                used[i, slot] = "shared"
    return slot_probs, used


def to_preds(P, episode_ids, episodes_by_id, data):
    pred = {}
    for r in data.queries:
        i = episode_ids.index(r["episode_id"])
        ep = episodes_by_id[r["episode_id"]]
        pred[r["query_id"]] = P[i, ep.slot_of(r["machine_id"])].tolist()
    return pred


def load_train():
    return load_split(os.path.join(PUB, "train"), with_labels=True)
