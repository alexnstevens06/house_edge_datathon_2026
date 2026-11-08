"""Sliding-window training examples and dense/sparse tokenization.

Every example predicts the joint next-tick outcome of all 8 machines at a
target tick ``t`` using ONLY observations with tick < t (asserted).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import MISSING, N_MACHINES, Episode, MAX_TICK

T_START = 16  # first target tick for sliding windows


@dataclass
class Example:
    ep: int                     # index into episode list
    t: int                      # target tick (the spin being predicted)
    labels: np.ndarray | None   # (8,) payout idx or MISSING; None -> read from obs[t]


def sliding_examples(episodes: list[Episode], t_start: int = T_START) -> list[Example]:
    """One example per (episode, target tick) where at least one machine is observed at t."""
    out: list[Example] = []
    for ep_idx, episode in enumerate(episodes):
        obs = episode.obs
        for t in range(t_start, obs.shape[0]):
            if (obs[t] >= 0).any():
                out.append(Example(ep_idx, t, None))
    return out


def query_examples(episodes: list[Episode], episode_index: dict[str, int],
                   queries: list[dict[str, str]], query_labels: dict[str, int]) -> list[Example]:
    """Real query cutoffs with labels.csv outcomes (one label per machine query)."""
    by_episode: dict[str, dict[str, dict[str, str]]] = {}
    for row in queries:
        by_episode.setdefault(row["episode_id"], {})[row["machine_id"]] = row
    out: list[Example] = []
    for episode_id, machines in sorted(by_episode.items()):
        ep_idx = episode_index[episode_id]
        episode = episodes[ep_idx]
        cutoff = episode.cutoff
        assert cutoff is not None
        labels = np.full(N_MACHINES, MISSING, dtype=np.int64)
        for machine_id, row in machines.items():
            labels[episode.slot_of(machine_id)] = query_labels[row["query_id"]]
        assert (labels >= 0).all(), f"{episode_id}: missing query labels"
        out.append(Example(ep_idx, cutoff, labels))
    return out


class WindowCollator:
    """Builds dense (W x 8 grid) or sparse (observed-only) token tensors.

    Dense symbols: 0 = MISSING, 1..5 = payout {0,1,2,5,20}.
    Sparse symbols: 0..4 = payout, 5 = PAD (masked).
    Tick embeddings always use the ORIGINAL absolute tick number.
    """

    def __init__(self, episodes: list[Episode], arch: str, window: int = 64):
        assert arch in ("dense", "sparse")
        self.episodes = episodes
        self.arch = arch
        self.window = window

    def _materialize(self, example: Example) -> tuple[np.ndarray, np.ndarray]:
        labels = example.labels
        if labels is None:
            obs = self.episodes[example.ep].obs
            if example.t < obs.shape[0]:
                labels = obs[example.t].astype(np.int64)
            else:  # prediction tick (query cutoff): no labels available
                labels = np.full(N_MACHINES, MISSING, dtype=np.int64)
        return labels, labels >= 0

    def __call__(self, batch: list[Example]) -> dict:
        episodes = self.episodes
        if self.arch == "dense":
            return self._dense(batch, episodes)
        return self._sparse(batch, episodes)

    def _dense(self, batch, episodes):
        B, W = len(batch), self.window
        symbols = np.zeros((B, W, N_MACHINES), dtype=np.int64)      # 0 = MISSING
        ticks = np.zeros((B, W, N_MACHINES), dtype=np.int64)        # clamped >= 0
        label_vec = np.zeros((B, N_MACHINES), dtype=np.int64)
        label_mask = np.zeros((B, N_MACHINES), dtype=bool)
        targets = np.zeros((B,), dtype=np.int64)
        for b, ex in enumerate(batch):
            episode = episodes[ex.ep]
            t = ex.t
            targets[b] = t
            lo = max(0, t - W)
            seg = episode.obs[lo:t]                                  # (t-lo, 8)
            w0 = W - (t - lo)
            symbols[b, w0:w0 + seg.shape[0]] = seg.astype(np.int64) + 1  # MISSING(-1)+1 -> 0
            ticks[b, w0:w0 + seg.shape[0]] = np.arange(lo, t)[:, None]
            labels, mask = self._materialize(ex)
            label_vec[b] = labels
            label_mask[b] = mask
        assert (ticks < targets[:, None, None]).all(), "dense causality: input tick >= target tick"
        types = np.stack([episodes[ex.ep].machine_types for ex in batch])
        slots_tiled = np.tile(np.arange(N_MACHINES, dtype=np.int64), W)      # token order w*8+slot
        return {
            "symbols": torch.from_numpy(symbols.reshape(B, W * N_MACHINES)),
            "ticks": torch.from_numpy(ticks.reshape(B, W * N_MACHINES)),
            "slots": torch.from_numpy(np.broadcast_to(slots_tiled, (B, W * N_MACHINES)).copy()),
            "types": torch.from_numpy(np.tile(types, (1, W))),
            "target_t": torch.from_numpy(targets),
            "labels": torch.from_numpy(label_vec),
            "label_mask": torch.from_numpy(label_mask),
        }

    def _sparse(self, batch, episodes):
        B, W = len(batch), self.window
        token_lists = []
        label_vec = np.zeros((B, N_MACHINES), dtype=np.int64)
        label_mask = np.zeros((B, N_MACHINES), dtype=bool)
        targets = np.zeros((B,), dtype=np.int64)
        for b, ex in enumerate(batch):
            episode = episodes[ex.ep]
            t = ex.t
            targets[b] = t
            lo = max(0, t - W)
            seg = episode.obs[lo:t]
            pos, slot = np.nonzero(seg >= 0)
            tick = (lo + pos).astype(np.int64)
            sym = seg[pos, slot].astype(np.int64)
            token_lists.append((tick, slot.astype(np.int64), sym, np.full(len(tick), episode.machine_types[slot], dtype=np.int64)))
            labels, mask = self._materialize(ex)
            label_vec[b] = labels
            label_mask[b] = mask
        N = max(len(tick) for tick, _, _, _ in token_lists)
        symbols = np.full((B, N), 5, dtype=np.int64)                 # 5 = PAD
        ticks = np.zeros((B, N), dtype=np.int64)
        slots = np.zeros((B, N), dtype=np.int64)
        types = np.zeros((B, N), dtype=np.int64)
        pad_mask = np.ones((B, N), dtype=bool)                       # True = pad
        for b, (tick, slot, sym, typ) in enumerate(token_lists):
            symbols[b, :len(tick)] = sym
            ticks[b, :len(tick)] = tick
            slots[b, :len(tick)] = slot
            types[b, :len(tick)] = typ
            pad_mask[b, :len(tick)] = False
        assert (ticks < targets[:, None]).all(), "sparse causality: input tick >= target tick"
        slot_q = np.arange(N_MACHINES, dtype=np.int64)
        types_q = np.stack([episodes[ex.ep].machine_types for ex in batch])
        return {
            "symbols": torch.from_numpy(symbols),
            "ticks": torch.from_numpy(ticks),
            "slots": torch.from_numpy(slots),
            "types": torch.from_numpy(types),
            "pad_mask": torch.from_numpy(pad_mask),
            "slots_q": torch.from_numpy(np.broadcast_to(slot_q, (B, N_MACHINES)).copy()),
            "types_q": torch.from_numpy(types_q),
            "target_t": torch.from_numpy(targets),
            "labels": torch.from_numpy(label_vec),
            "label_mask": torch.from_numpy(label_mask),
        }
