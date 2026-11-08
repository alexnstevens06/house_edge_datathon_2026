"""Loading public data into per-episode numpy structures.

Observation model: ``obs[tick, slot]`` is the payout index (0..4) or ``MISSING``
(-1) when the tick is unobserved for that machine. A missing tick is UNKNOWN,
never a zero payout.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}
PAYOUT_IDX_TO_VALUE = {i: p for i, p in enumerate(PAYOUTS)}
MISSING = -1
N_MACHINES = 8
N_PAYOUTS = 5
MACHINE_TYPES = ("balanced", "jackpot", "steady")
TYPE_TO_IDX = {t: i for i, t in enumerate(MACHINE_TYPES)}
MAX_TICK = 300  # ticks observed 0..255, cutoffs up to 256 -> embedding table 0..300


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Episode:
    episode_id: str
    machine_ids: tuple[str, ...]        # slot order (sorted machine_id)
    machine_types: np.ndarray           # (8,) int type index per slot
    obs: np.ndarray                     # (L, 8) int8 payout idx or MISSING
    cutoff: int | None = None           # prediction tick (next spin), if known

    @property
    def n_ticks(self) -> int:
        return self.obs.shape[0]

    def slot_of(self, machine_id: str) -> int:
        return self.machine_ids.index(machine_id)


@dataclass
class SplitData:
    episodes: list[Episode]
    episode_index: dict[str, int]
    queries: list[dict[str, str]]       # raw query rows
    query_labels: dict[str, int] | None  # query_id -> payout idx, if split has labels


def load_split(split_dir: str | Path, with_labels: bool = False) -> SplitData:
    """Load queries.csv + history.csv (+ labels.csv) for train/validation/test."""
    split_dir = Path(split_dir)
    queries = read_csv_rows(split_dir / "queries.csv")
    by_episode: dict[str, dict[str, str]] = {}
    for row in queries:
        by_episode.setdefault(row["episode_id"], {})[row["machine_id"]] = row

    hist_rows = read_csv_rows(split_dir / "history.csv")
    observed: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row in hist_rows:
        key = (row["episode_id"], row["machine_id"])
        observed.setdefault(key, []).append((int(row["tick"]), PAYOUT_TO_IDX[int(row["payout"])]))

    episodes: list[Episode] = []
    episode_index: dict[str, int] = {}
    for episode_id in sorted(by_episode):
        machines = by_episode[episode_id]
        machine_ids = tuple(sorted(machines))
        assert len(machine_ids) == N_MACHINES, f"{episode_id}: expected 8 machines"
        cutoffs = {int(m["cutoff"]) for m in machines.values()}
        assert len(cutoffs) == 1, f"{episode_id}: inconsistent cutoffs {cutoffs}"
        cutoff = cutoffs.pop()
        max_tick = 0
        for machine_id in machine_ids:
            for tick, _ in observed.get((episode_id, machine_id), ()):
                assert tick < cutoff, f"history tick {tick} >= cutoff {cutoff}"
                max_tick = max(max_tick, tick)
        obs = np.full((max_tick + 1, N_MACHINES), MISSING, dtype=np.int8)
        for slot, machine_id in enumerate(machine_ids):
            for tick, payout_idx in observed.get((episode_id, machine_id), ()):
                obs[tick, slot] = payout_idx
        machine_types = np.array([TYPE_TO_IDX[machines[m]["machine_type"]] for m in machine_ids], dtype=np.int64)
        episode_index[episode_id] = len(episodes)
        episodes.append(Episode(episode_id, machine_ids, machine_types, obs, cutoff))

    query_labels = None
    if with_labels:
        label_rows = read_csv_rows(split_dir / "labels.csv")
        query_labels = {r["query_id"]: PAYOUT_TO_IDX[int(r["outcome"])] for r in label_rows}
    return SplitData(episodes, episode_index, queries, query_labels)


def subset_split_data(data: SplitData, episode_ids) -> SplitData:
    """SplitData restricted to the given episode ids (index remapped)."""
    keep = set(episode_ids)
    episodes = [e for e in data.episodes if e.episode_id in keep]
    episode_index = {e.episode_id: i for i, e in enumerate(episodes)}
    queries = [r for r in data.queries if r["episode_id"] in keep]
    return SplitData(episodes, episode_index, queries, data.query_labels)


def episode_conditions(split_dir: str | Path) -> dict[str, str]:
    """episode_id -> condition (one condition per episode)."""
    rows = read_csv_rows(Path(split_dir) / "conditions.csv")
    queries = {r["query_id"]: r["episode_id"] for r in read_csv_rows(Path(split_dir) / "queries.csv")}
    mapping: dict[str, str] = {}
    for row in rows:
        episode_id = queries[row["query_id"]]
        assert mapping.get(episode_id, row["condition"]) == row["condition"], "episode spans conditions"
        mapping[episode_id] = row["condition"]
    return mapping


def data_sha256(kit_dir: str | Path) -> str:
    import hashlib
    return hashlib.sha256(Path(kit_dir, "kit.zip").read_bytes()).hexdigest()
