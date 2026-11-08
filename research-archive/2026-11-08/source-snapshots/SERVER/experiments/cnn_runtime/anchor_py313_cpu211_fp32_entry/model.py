"""Causal temporal CNN forward path, copied from the no_signal anchor source.

Training, labels, and signal features are not used. USE_SIGNAL stays False.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MISSING, N_MACHINES

USE_SIGNAL = False


def episode_meta(data):
    """Signal is not read. Every episode gets a zero signal vector."""
    if not USE_SIGNAL:
        return {r["episode_id"]: np.zeros(N_MACHINES, np.int64) for r in data.queries}
    raise RuntimeError("signal mode is refused")


def window(obs, t, W):
    """Features use ONLY ticks < t on all machines. Returns (W,8) payout idx with MISSING pad."""
    x = np.full((W, N_MACHINES), MISSING, np.int8)
    lo = max(0, t - W)
    seg = obs[lo:min(t, obs.shape[0])]
    if len(seg):
        x[W - (t - lo):W - (t - lo) + len(seg)] = seg
    return x


def build_query(data, W):
    meta = episode_meta(data)
    X, T, TY, SG, ids = [], [], [], [], []
    for ep in data.episodes:
        X.append(window(ep.obs, ep.cutoff, W))
        T.append(ep.cutoff)
        TY.append(ep.machine_types)
        SG.append(meta[ep.episode_id])
        ids.append(ep)
    return [torch.tensor(np.array(a)) for a in (X, T, TY, SG)], ids


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
    def __init__(s, h=64, layers=4, k=3, drop=0.1, W=64, use_signal=True):
        super().__init__()
        s.W = W
        s.use_signal = use_signal
        s.pay = nn.Embedding(6, h)
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


def predict(model, Q, dev, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Q[0]), bs):
            b = [a[i:i + bs].to(dev) for a in Q]
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = model(*b)
            out.append(torch.softmax(lg.float(), -1).cpu())
    return torch.cat(out).double().numpy()


def to_preds(P, eps, data):
    pred = {}
    for r in data.queries:
        i = [e.episode_id for e in eps].index(r["episode_id"])
        ep = eps[i]
        pred[r["query_id"]] = P[i, ep.slot_of(r["machine_id"])].tolist()
    return pred
