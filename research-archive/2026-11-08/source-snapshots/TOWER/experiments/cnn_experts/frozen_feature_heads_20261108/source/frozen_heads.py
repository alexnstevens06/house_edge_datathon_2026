"""Frozen CNN03 trunk plus 645-parameter signal heads.

The trunk is the staged no-signal historical pretrain. Signal is a mask and a
router only. Historical windows are not built here.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
STUDY = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "kit"))

from house_edge.data import MISSING, N_MACHINES, PAYOUTS, TYPE_TO_IDX, load_split  # noqa: E402

assert PAYOUTS == (0, 1, 2, 5, 20)
assert MISSING == -1 and MISSING != 0
assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}

PUB = os.path.join(REPO, "kit", "public")
TRUNK_PATH = os.path.join(STUDY, "inputs", "history_pretrain.pt")
TRUNK_SHA = "e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760"
TRUNK_STEPS = 1110
TRUNK_PARAMS = 367109
HEAD_PARAMS = 645  # Linear(128, 5)
SIGNALS = (-1, 0, 1)
MIN_GROUP_ROWS = 32
EXPECTED_COUNTS = {-1: 250, 0: 267, 1: 251}

SCHEDULE = {
    "seed": 0,
    "epochs": 10,
    "lr": 0.001,
    "wd": 0.0,
    "grad_clip": None,
    "optimizer": "AdamW",
    "batch": "all_96_cutoff_windows",
    "feature_dim": 128,
    "head_params": HEAD_PARAMS,
    "min_group_rows": MIN_GROUP_ROWS,
}


class Block(nn.Module):
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


class CNN(nn.Module):
    """CNN03 trunk. use_signal stays False: there is no signal embedding."""

    def __init__(self, h=128, layers=4, k=3, drop=0.3, W=64, use_signal=False):
        super().__init__()
        if use_signal:
            raise RuntimeError("this study does not construct a signal embedding")
        self.W = W
        self.use_signal = False
        self.pay = nn.Embedding(6, h)
        self.mask = nn.Linear(1, h)
        self.typ = nn.Embedding(3, h)
        self.sig = None
        self.slot = nn.Embedding(8, h)
        self.tick = nn.Linear(2, h)
        self.blocks = nn.ModuleList([Block(h, 2 ** i, k, drop) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 5))

    def encode(self, x, t, ty):
        b, w, m = x.shape
        x = x.long().transpose(1, 2)
        # Follow the trunk dtype. Hardcoded float32 breaks the CPU fp64 scorer on torch 2.14.
        dtype = self.pay.weight.dtype
        observed = (x >= 0).to(dtype=dtype).unsqueeze(-1)
        idx = torch.where(x >= 0, x, torch.full_like(x, 5))
        ticks = (t.view(b, 1, 1).to(dtype=dtype) - w + torch.arange(w, device=x.device).view(1, 1, w)).expand(b, m, w)
        tf = torch.stack([ticks / 256.0, (t.view(b, 1, 1).to(dtype=dtype) - ticks) / w], -1)
        meta = self.typ(ty) + self.slot.weight[None]
        z = self.pay(idx) + self.mask(observed) + self.tick(tf) + meta.unsqueeze(2)
        for block in self.blocks:
            z = block(z)
        last = z[:, :, -1]
        pooled = last.mean(1, keepdim=True).expand_as(last)
        return torch.cat([last, pooled], -1)

    def features(self, x, t, ty):
        """128-d vector that feeds head[2]. Caller must already be in eval."""
        return self.head[1](self.head[0](self.encode(x, t, ty)))

    def forward(self, x, t, ty, sg=None):
        del sg
        return self.head[2](self.features(x, t, ty))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tensors(pairs):
    h = hashlib.sha256()
    for name, value in sorted(pairs, key=lambda kv: kv[0]):
        h.update(name.encode())
        arr = value.detach().cpu().contiguous()
        h.update(str(tuple(arr.shape)).encode())
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def state_hash(module):
    items = [(n, p) for n, p in module.state_dict().items()]
    return hash_tensors(items)


def new_trunk():
    model = CNN()
    n = sum(p.numel() for p in model.parameters())
    if n != TRUNK_PARAMS:
        raise RuntimeError(f"trunk param count {n} != {TRUNK_PARAMS}")
    if any("sig" in name for name, _ in model.named_parameters()):
        raise RuntimeError("trunk has signal parameters")
    return model


def load_trunk(path=TRUNK_PATH):
    digest = sha256_file(path)
    if digest != TRUNK_SHA:
        raise RuntimeError(f"trunk sha256 {digest} != {TRUNK_SHA}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("phase") != "pretrain" or int(ckpt.get("steps", -1)) != TRUNK_STEPS or int(ckpt.get("seed", -1)) != 0:
        raise RuntimeError(f"unexpected trunk metadata {ckpt.get('phase')} steps={ckpt.get('steps')} seed={ckpt.get('seed')}")
    if len(ckpt.get("losses", [])) != 30:
        raise RuntimeError("trunk does not have 30 pretrain losses")
    model = new_trunk()
    model.load_state_dict(ckpt["state"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head_n = sum(p.numel() for p in model.head[2].parameters())
    if head_n != HEAD_PARAMS:
        raise RuntimeError(f"head[2] has {head_n} parameters, expected {HEAD_PARAMS}")
    return model, ckpt


def make_head(trunk):
    head = nn.Linear(128, 5)
    head.load_state_dict({k: v.detach().cpu().clone() for k, v in trunk.head[2].state_dict().items()})
    if sum(p.numel() for p in head.parameters()) != HEAD_PARAMS:
        raise RuntimeError("cloned head is not 645 parameters")
    return head


def freeze_check(trunk, before_hash):
    if state_hash(trunk) != before_hash:
        raise RuntimeError("frozen trunk state changed")
    if any(p.requires_grad for p in trunk.parameters()):
        raise RuntimeError("a trunk parameter is trainable")


def window(obs, t, W):
    """Ticks < t only. Right edge is tick t-1. Missing stays -1."""
    x = np.full((W, N_MACHINES), MISSING, np.int8)
    lo = max(0, t - W)
    seg = obs[lo:min(t, obs.shape[0])]
    if len(seg):
        x[W - (t - lo):W - (t - lo) + len(seg)] = seg
    return x


def signals_by_episode(data):
    out, seen = {}, {}
    for row in data.queries:
        ep = data.episodes[data.episode_index[row["episode_id"]]]
        slot = ep.slot_of(row["machine_id"])
        eid = row["episode_id"]
        arr = out.setdefault(eid, np.full(N_MACHINES, 99, np.int64))
        got = seen.setdefault(eid, np.zeros(N_MACHINES, np.int8))
        if got[slot]:
            raise RuntimeError(f"duplicate query {eid} slot {slot}")
        sig = int(row["signal"])
        if sig not in SIGNALS:
            raise RuntimeError(f"bad signal {sig}")
        arr[slot] = sig
        got[slot] = 1
    for eid, got in seen.items():
        if int(got.sum()) != N_MACHINES:
            raise RuntimeError(f"{eid} is missing a signal")
    return out


def _stack(xs, ts, tys, ys, signals, eids, W):
    if not xs:
        empty = (
            torch.zeros(0, W, N_MACHINES, dtype=torch.int8),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, N_MACHINES, dtype=torch.long),
            torch.zeros(0, N_MACHINES, dtype=torch.long),
            torch.zeros(0, N_MACHINES, dtype=torch.long),
            [],
        )
        return empty
    return (
        torch.from_numpy(np.stack(xs)),
        torch.tensor(ts, dtype=torch.long),
        torch.from_numpy(np.stack(tys).astype(np.int64, copy=False)),
        torch.from_numpy(np.stack(ys).astype(np.int64, copy=False)),
        torch.from_numpy(np.stack(signals).astype(np.int64, copy=False)),
        eids,
    )


def build_cutoff(data, W=64):
    """True cutoff rows only. Labels are query_labels. Signal is not written into X."""
    if not data.query_labels:
        raise RuntimeError("cutoff rows require train query labels")
    sigs = signals_by_episode(data)
    by_ep = {}
    for row in data.queries:
        by_ep.setdefault(row["episode_id"], []).append(row)
    xs, ts, tys, ys, sg, eids = [], [], [], [], [], []
    for ep in data.episodes:
        if ep.cutoff is None:
            raise RuntimeError(f"{ep.episode_id} has no cutoff")
        y = np.full(N_MACHINES, MISSING, np.int64)
        for row in by_ep.get(ep.episode_id, []):
            qid = row["query_id"]
            if qid not in data.query_labels:
                raise RuntimeError(f"missing label {qid}")
            y[ep.slot_of(row["machine_id"])] = int(data.query_labels[qid])
        xs.append(window(ep.obs, ep.cutoff, W))
        ts.append(int(ep.cutoff))
        tys.append(np.asarray(ep.machine_types))
        ys.append(y)
        sg.append(sigs[ep.episode_id])
        eids.append(ep.episode_id)
        if ts[-1] != ep.cutoff:
            raise RuntimeError("cutoff window tick drifted")
    return _stack(xs, ts, tys, ys, sg, eids, W)


def build_eval(data, W=64):
    """Query windows for scoring. Y is entirely missing."""
    sigs = signals_by_episode(data)
    xs, ts, tys, ys, sg, eids = [], [], [], [], [], []
    for ep in data.episodes:
        xs.append(window(ep.obs, ep.cutoff, W))
        ts.append(int(ep.cutoff))
        tys.append(np.asarray(ep.machine_types))
        ys.append(np.full(N_MACHINES, MISSING, np.int64))
        sg.append(sigs[ep.episode_id])
        eids.append(ep.episode_id)
    return _stack(xs, ts, tys, ys, sg, eids, W)


def count_labeled(signals, y):
    signals = np.asarray(signals)
    y = np.asarray(y)
    return {s: int(((signals == s) & (y >= 0)).sum()) for s in SIGNALS}


def active_map(counts, min_rows=MIN_GROUP_ROWS):
    return {s: int(counts[s]) >= int(min_rows) for s in SIGNALS}


def apply_signal_mask(y, signals, signal):
    if signal is None:
        return y
    out = torch.full_like(y, -1)
    keep = (signals == int(signal)) & (y >= 0)
    out[keep] = y[keep]
    return out


def masked_ce(logits, y):
    yl = y.long()
    m = yl >= 0
    if int(m.sum()) == 0:
        raise RuntimeError("masked loss has no labeled slots")
    return F.cross_entropy(logits.float()[m], yl[m])


def specialist_name(sig):
    return {-1: "specialist_-1", 0: "specialist_0", 1: "specialist_+1"}[int(sig)]


def route_logits(head_logits, trunk_logits, signals, active):
    """Each slot is copied from exactly one head. No mixture."""
    signals_np = signals.detach().cpu().numpy() if torch.is_tensor(signals) else np.asarray(signals)
    out = np.empty(trunk_logits.shape, np.float64)
    used = np.empty(signals_np.shape, dtype=object)
    for i in range(signals_np.shape[0]):
        for slot in range(N_MACHINES):
            sig = int(signals_np[i, slot])
            if sig not in SIGNALS:
                raise RuntimeError(f"bad signal {sig}")
            if active[sig]:
                out[i, slot] = head_logits[sig][i, slot]
                used[i, slot] = specialist_name(sig)
            else:
                out[i, slot] = trunk_logits[i, slot]
                used[i, slot] = "trunk"
    return out, used


def row_normalize(probs):
    p = np.asarray(probs, dtype=np.float64)
    totals = p.sum(axis=-1, keepdims=True)
    if not np.isfinite(p).all() or np.any(totals <= 0):
        raise RuntimeError("probabilities are not finite and positive")
    return p / totals


def logits_to_probs(logits):
    if torch.is_tensor(logits):
        probs = torch.softmax(logits.double(), dim=-1).cpu().numpy()
    else:
        probs = np.asarray(logits, dtype=np.float64)
        probs = np.exp(probs - probs.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
    return row_normalize(probs)


def head_logits(head, feats):
    return head(feats)


@torch.no_grad()
def featurize(model, X, T, TY, device, bs=32):
    model.eval()
    pieces = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device)
        t = T[i:i + bs].to(device)
        ty = TY[i:i + bs].to(device)
        pieces.append(model.features(x, t, ty).detach())
    return torch.cat(pieces, 0) if pieces else torch.zeros(0, N_MACHINES, 128)


def require_seed(seed):
    if int(seed) != 0:
        raise RuntimeError("seed guard: this study trains seed 0 only")


def train_head(head, feats, y, signals, signal, epochs, lr, wd, device):
    """Full-batch AdamW. `signal is None` is the shared control."""
    head = head.to(device)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    losses = []
    y = y.to(device)
    signals = signals.to(device)
    feats = feats.detach()
    for _ in range(int(epochs)):
        logits = head(feats)
        target = apply_signal_mask(y, signals, signal) if signal is not None else y
        loss = masked_ce(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    head.eval()
    return head, losses


def gpu_op_check(device="cuda"):
    """One real device forward and backward. Raises if the ROCm device cannot train."""
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm CUDA device is not available")
    x = torch.randn(4, 16, device=device, dtype=torch.float32)
    layer = nn.Linear(16, 5).to(device)
    loss = layer(x).square().mean()
    loss.backward()
    if layer.weight.grad is None or not torch.isfinite(layer.weight.grad).all():
        raise RuntimeError("device backward did not produce a finite gradient")
    conv = nn.Conv1d(4, 4, 3).to(device)
    z = torch.randn(2, 4, 12, device=device)
    conv(z).sum().backward()
    if conv.weight.grad is None:
        raise RuntimeError("conv backward failed")
    return torch.cuda.get_device_name(0)


def to_preds(probs, episode_ids, data):
    by_id = {ep.episode_id: ep for ep in data.episodes}
    pos = {eid: i for i, eid in enumerate(episode_ids)}
    pred = {}
    for row in data.queries:
        i = pos[row["episode_id"]]
        slot = by_id[row["episode_id"]].slot_of(row["machine_id"])
        pred[row["query_id"]] = probs[i, slot].tolist()
    return pred
