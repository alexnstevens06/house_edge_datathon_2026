"""Soft mixture of CNN experts. History never sees the query signal.

Encoder matches CNN03 (experiments/mlp_cnn_hpo/source/base_cnn.py) with use_signal off.
Work only under experiments/cnn_experts/soft_moe. Imports of house_edge / kit are read-only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "kit"))

from house_edge.data import (  # noqa: E402
    MISSING,
    N_MACHINES,
    PAYOUTS,
    TYPE_TO_IDX,
    load_split,
)

PUB = os.path.join(REPO, "kit", "public")
SPLITS_PATH = os.path.join(REPO, "runs", "splits.json")

# Signal codes. ABSENT is not a real query signal and is the only history value.
SIGS = {-1: 0, 0: 1, 1: 2}
ABSENT = 3

H = 128
LAYERS = 4
KERNEL = 3
WIDTH = 64
DROP = 0.3
LR = 0.00197
WD = 0.00895
BS = 512
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 10
MIN_T = 16
EXPERTS = 3
EXPERT_H = 32
VRAM_FRACTION = 2.0 / 16.0

INCUMBENT_TUNE_R = {
    0: 0.33206398675779764,
    1: 0.34654552298986674,
    2: 0.33342647148315785,
}

assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}
assert PAYOUTS == (0, 1, 2, 5, 20)
assert MISSING == -1


def window(obs, t, W):
    """Features use ONLY ticks < t on all machines. (W, 8) payout idx, MISSING pad."""
    x = np.full((W, N_MACHINES), MISSING, np.int8)
    lo = max(0, t - W)
    seg = obs[lo:min(t, obs.shape[0])]
    if len(seg):
        x[W - (t - lo):W - (t - lo) + len(seg)] = seg
    return x


def _queries_by_episode(data):
    grouped = {}
    for row in data.queries:
        grouped.setdefault(row["episode_id"], []).append(row)
    return grouped


def build_split_windows(data, W, stage, min_t=MIN_T):
    """Build history (t < cutoff, signal ABSENT) or true-cutoff rows.

    History does not read query signal or query labels. Cutoff labels are used
    only when ``data.query_labels`` is present (TRAIN). Returns tensors
    X, T, TY, SG, Y, CUT and the episode object per row.
    """
    if stage not in ("history", "cutoff"):
        raise ValueError(stage)
    grouped = _queries_by_episode(data)
    xs, ts, tys, sgs, ys, cuts, eps = [], [], [], [], [], [], []
    for ep in data.episodes:
        obs = ep.obs
        if stage == "history":
            for t in range(min_t, ep.cutoff):
                if t < obs.shape[0]:
                    y = obs[t].astype(np.int64, copy=False)
                else:
                    y = np.full(N_MACHINES, MISSING, np.int64)
                if int((y >= 0).sum()) == 0:
                    continue
                xs.append(window(obs, t, W))
                ts.append(t)
                tys.append(ep.machine_types)
                sgs.append(np.full(N_MACHINES, ABSENT, np.int64))
                ys.append(np.array(y, dtype=np.int64, copy=True))
                cuts.append(False)
                eps.append(ep)
        else:
            y = np.full(N_MACHINES, MISSING, np.int64)
            sg = np.full(N_MACHINES, ABSENT, np.int64)
            for row in grouped.get(ep.episode_id, []):
                slot = ep.slot_of(row["machine_id"])
                sig = int(row["signal"])
                if sig not in SIGS:
                    raise RuntimeError(f"unexpected signal {sig}")
                sg[slot] = SIGS[sig]
                if data.query_labels is not None:
                    y[slot] = int(data.query_labels[row["query_id"]])
            if data.query_labels is not None and int((y >= 0).sum()) == 0:
                continue
            xs.append(window(obs, ep.cutoff, W))
            ts.append(ep.cutoff)
            tys.append(ep.machine_types)
            sgs.append(sg)
            ys.append(y)
            cuts.append(True)
            eps.append(ep)
    if not xs:
        raise RuntimeError(f"no {stage} windows")
    tensors = (
        torch.tensor(np.stack(xs)),
        torch.tensor(np.asarray(ts), dtype=torch.long),
        torch.tensor(np.stack(tys)),
        torch.tensor(np.stack(sgs), dtype=torch.long),
        torch.tensor(np.stack(ys), dtype=torch.long),
        torch.tensor(np.asarray(cuts), dtype=torch.bool),
    )
    return tensors, eps


class Block(nn.Module):
    """Causal temporal block. Same math as CNN03 Block."""

    def __init__(self, h, d, k, drop):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(h, h, k, dilation=d)
        self.mix = nn.Linear(2 * h, h)
        self.n1 = nn.LayerNorm(h)
        self.n2 = nn.LayerNorm(h)
        self.dr = nn.Dropout(drop)

    def forward(self, z):
        b, m, w, h = z.shape
        c = self.conv(F.pad(z.reshape(b * m, w, h).transpose(1, 2), (self.pad, 0))).transpose(1, 2).reshape(b, m, w, h)
        z = self.n1(z + self.dr(F.gelu(c)))
        others = (z.sum(1, keepdim=True) - z) / (m - 1)
        return self.n2(z + self.dr(F.gelu(self.mix(torch.cat([z, others], -1)))))


class HistoryEncoder(nn.Module):
    """CNN03 encoder. No signal parameters."""

    def __init__(self, h=H, layers=LAYERS, k=KERNEL, drop=DROP, W=WIDTH):
        super().__init__()
        self.W = W
        self.h = h
        self.pay = nn.Embedding(6, h)
        self.mask = nn.Linear(1, h)
        self.typ = nn.Embedding(3, h)
        self.slot = nn.Embedding(8, h)
        self.tick = nn.Linear(2, h)
        self.blocks = nn.ModuleList([Block(h, 2 ** i, k, drop) for i in range(layers)])

    def forward(self, x, t, ty):
        b, w, m = x.shape
        x = x.long().transpose(1, 2)
        observed = (x >= 0).float().unsqueeze(-1)
        idx = torch.where(x >= 0, x, torch.full_like(x, 5))
        ticks = (t.view(b, 1, 1).float() - w + torch.arange(w, device=x.device).view(1, 1, w)).expand(b, m, w)
        tf = torch.stack([ticks / 256.0, (t.view(b, 1, 1).float() - ticks) / float(w)], -1)
        meta = self.typ(ty.long()) + self.slot.weight[None]
        z = self.pay(idx) + self.mask(observed) + self.tick(tf) + meta.unsqueeze(2)
        for block in self.blocks:
            z = block(z)
        return z


class LogitHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 5))

    def forward(self, last, pooled):
        return self.net(torch.cat([last, pooled], -1))


class ExpertHead(nn.Module):
    """Small causal CNN residual on shared features z (B, 8, W, H)."""

    def __init__(self, h, he=EXPERT_H, drop=DROP):
        super().__init__()
        self.proj = nn.Linear(h, he)
        self.block = Block(he, 1, 3, drop)
        self.head = nn.Sequential(nn.Linear(2 * he, he), nn.GELU(), nn.Linear(he, 5))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z):
        z = self.block(self.proj(z))
        last = z[:, :, -1]
        pooled = last.mean(1, keepdim=True).expand_as(last)
        return self.head(torch.cat([last, pooled], -1))


class SoftGate(nn.Module):
    """Per-machine softmax over experts. Signal is forced to ABSENT off cutoff."""

    def __init__(self, h, n_exp=EXPERTS):
        super().__init__()
        self.n_exp = n_exp
        self.typ = nn.Embedding(3, 16)
        self.sig = nn.Embedding(4, 16)
        self.mlp = nn.Sequential(nn.Linear(h + h + 16 + 16 + 1, 64), nn.GELU(), nn.Linear(64, n_exp))

    def forward(self, last, pooled, ty, sg, is_cutoff):
        b, m, _ = last.shape
        cut = is_cutoff.reshape(b, 1).to(dtype=torch.bool)
        sg = sg.long()
        absent = torch.full_like(sg, ABSENT)
        # Off cutoff, every slot is ABSENT. On cutoff, keep the provided code
        # (0..2 legitimate, or ABSENT for a slot that has no query).
        sg_use = torch.where(cut.expand(b, m), sg, absent)
        cut_f = cut.expand(b, m).to(last.dtype).unsqueeze(-1)
        feat = torch.cat([last, pooled, self.typ(ty.long()), self.sig(sg_use), cut_f], -1)
        return torch.softmax(self.mlp(feat), dim=-1)


def _last_pool(z):
    last = z[:, :, -1]
    pooled = last.mean(1, keepdim=True).expand_as(last)
    return last, pooled


class ControlCNN(nn.Module):
    """Single shared head. Ignores signal, cutoff flag, and expert mask."""

    def __init__(self, h=H, layers=LAYERS, k=KERNEL, drop=DROP, W=WIDTH):
        super().__init__()
        self.enc = HistoryEncoder(h, layers, k, drop, W)
        self.head = LogitHead(h)
        self.n_exp = EXPERTS

    def forward(self, x, t, ty, sg, is_cutoff, expert_keep=None):
        del sg, is_cutoff, expert_keep
        last, pooled = _last_pool(self.enc(x, t, ty))
        probs = torch.softmax(self.head(last, pooled).float(), dim=-1)
        return probs, None

    def shared_probs(self, x, t, ty):
        last, pooled = _last_pool(self.enc(x, t, ty))
        return torch.softmax(self.head(last, pooled).float(), dim=-1)


class SoftMoE(nn.Module):
    def __init__(self, h=H, layers=LAYERS, k=KERNEL, drop=DROP, W=WIDTH, n_exp=EXPERTS):
        super().__init__()
        self.enc = HistoryEncoder(h, layers, k, drop, W)
        self.head = LogitHead(h)
        self.experts = nn.ModuleList([ExpertHead(h, EXPERT_H, drop) for _ in range(n_exp)])
        self.gate = SoftGate(h, n_exp)
        self.n_exp = n_exp

    def forward(self, x, t, ty, sg, is_cutoff, expert_keep=None):
        z = self.enc(x, t, ty)
        last, pooled = _last_pool(z)
        base = self.head(last, pooled)
        base_p = torch.softmax(base.float(), dim=-1)
        gate = self.gate(last, pooled, ty, sg, is_cutoff)
        if expert_keep is None:
            expert_keep = torch.ones(self.n_exp, dtype=torch.bool, device=x.device)
        else:
            expert_keep = expert_keep.to(device=x.device, dtype=torch.bool).reshape(self.n_exp)
        if not bool(expert_keep.any()):
            return base_p, gate
        stacked = torch.stack(
            [torch.softmax((base + expert(z)).float(), dim=-1) for expert in self.experts],
            dim=2,
        )
        g = gate * expert_keep.view(1, 1, -1).to(gate.dtype)
        mass = g.sum(-1, keepdim=True)
        fallback = mass <= 0
        g = torch.where(fallback, torch.zeros_like(g), g / mass.clamp_min(1e-12))
        mixed = (g.unsqueeze(-1) * stacked).sum(2)
        probs = torch.where(fallback, base_p, mixed)
        return probs, gate


def build_model(kind, **kwargs):
    if kind == "control":
        return ControlCNN(**kwargs)
    if kind == "soft":
        return SoftMoE(**kwargs)
    raise ValueError(kind)


def masked_nll(probs, y):
    """NLL on labelled slots. Missing targets (y < 0) add nothing."""
    yl = y.long()
    mask = yl >= 0
    if not bool(mask.any()):
        return probs.sum() * 0.0
    logp = torch.log(probs.float().clamp_min(1e-8))
    return F.nll_loss(logp[mask], yl[mask])


def train_phase(model, tensors, epochs, lr, wd, bs, device):
    """AdamW + OneCycle + grad clip 1. Same loop for control and soft."""
    x, t, ty, sg, y, cut = tensors
    n = int(x.shape[0])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = max(1, epochs * ((n + bs - 1) // bs))
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    data = [tensor.to(device) for tensor in (x, t, ty, sg, y, cut)]
    losses = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for start in range(0, n, bs):
            index = perm[start:start + bs]
            xb, tb, tyb, sgb, yb, cb = (part[index] for part in data)
            with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                probs, _gate = model(xb, tb, tyb, sgb, cb)
            loss = masked_nll(probs, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            schedule.step()
            total += float(loss.item()) * int(index.shape[0])
            seen += int(index.shape[0])
        losses.append(total / max(seen, 1))
        print(f"epoch {epoch} loss {losses[-1]:.4f}", flush=True)
    return losses


def predict_probs(model, tensors, device, bs=256):
    model.eval()
    x, t, ty, sg, _y, cut = tensors
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), bs):
            batch = [part[start:start + bs].to(device) for part in (x, t, ty, sg, cut)]
            probs, _gate = model(batch[0], batch[1], batch[2], batch[3], batch[4])
            pieces.append(probs.float().cpu())
    return torch.cat(pieces).double().numpy()


def to_preds(probs, episodes, data):
    index = {ep.episode_id: i for i, ep in enumerate(episodes)}
    pred = {}
    for row in data.queries:
        i = index[row["episode_id"]]
        ep = episodes[i]
        pred[row["query_id"]] = probs[i, ep.slot_of(row["machine_id"])].tolist()
    return pred


def assert_history_has_no_signal(tensors):
    _x, t, _ty, sg, _y, cut = tensors
    if bool(cut.any()):
        raise RuntimeError("history batch contains cutoff rows")
    if int((sg != ABSENT).sum()) != 0:
        raise RuntimeError("history signal channel is not ABSENT")
    if int(t.min()) < MIN_T:
        raise RuntimeError("history tick below min_t")


def assert_cutoff_rows(tensors, data):
    _x, t, _ty, sg, y, cut = tensors
    if not bool(cut.all()):
        raise RuntimeError("cutoff batch has a history row")
    if data.query_labels is not None:
        labelled = int((y >= 0).sum())
        if labelled == 0:
            raise RuntimeError("TRAIN cutoff rows have no labels")
    # Legitimate signals are 0..2. Unqueried slots stay ABSENT.
    bad = ~((sg == ABSENT) | ((sg >= 0) & (sg <= 2)))
    if bool(bad.any()):
        raise RuntimeError("cutoff signal codes out of range")
    del t
