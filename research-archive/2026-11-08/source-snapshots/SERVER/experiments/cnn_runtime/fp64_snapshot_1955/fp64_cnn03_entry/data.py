"""Public-split loader. Same episode construction as the anchor data.py.

Reads only history.csv and queries.csv for the requested split. The signal
column is not used. Missing ticks stay MISSING (-1), never a zero payout.
Labels, conditions, targets, and reference files are not opened.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}
MISSING = -1
N_MACHINES = 8
MACHINE_TYPES = ("balanced", "jackpot", "steady")
TYPE_TO_IDX = {t: i for i, t in enumerate(MACHINE_TYPES)}

QUERY_HEADER = ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff")
HISTORY_HEADER = ("episode_id", "machine_id", "tick", "payout")


@dataclass
class Episode:
    episode_id: str
    machine_ids: tuple[str, ...]
    machine_types: np.ndarray
    obs: np.ndarray
    cutoff: int | None = None

    def slot_of(self, machine_id: str) -> int:
        return self.machine_ids.index(machine_id)


@dataclass
class SplitData:
    episodes: list[Episode]
    episode_index: dict[str, int]
    queries: list[dict[str, str]]


def _read_table(path: Path, header: tuple[str, ...]) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            got = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"empty csv: {path}") from exc
        if got != header:
            raise ValueError(f"unexpected header in {path}: {got}")
        rows = []
        for rec in reader:
            if len(rec) != len(header):
                raise ValueError(f"bad row width in {path}")
            rows.append(rec)
    return rows


def load_public_split(data_dir: str | Path, split: str = "test") -> SplitData:
    """Load data_dir/<split>/{queries,history}.csv. split is test or validation."""
    if split not in ("test", "validation"):
        raise ValueError(f"unsupported split {split}")
    split_dir = Path(data_dir) / split
    query_rows = _read_table(split_dir / "queries.csv", QUERY_HEADER)
    by_episode: dict[str, dict[str, str]] = {}
    queries: list[dict[str, str]] = []
    for rec in query_rows:
        row = {
            "query_id": rec[0],
            "episode_id": rec[1],
            "machine_id": rec[2],
            "machine_type": rec[3],
            "cutoff": rec[5],
        }
        queries.append(row)
        by_episode.setdefault(row["episode_id"], {})[row["machine_id"]] = row

    observed: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for rec in _read_table(split_dir / "history.csv", HISTORY_HEADER):
        key = (rec[0], rec[1])
        observed.setdefault(key, []).append((int(rec[2]), PAYOUT_TO_IDX[int(rec[3])]))

    episodes: list[Episode] = []
    episode_index: dict[str, int] = {}
    for episode_id in sorted(by_episode):
        machines = by_episode[episode_id]
        machine_ids = tuple(sorted(machines))
        if len(machine_ids) != N_MACHINES:
            raise ValueError(f"{episode_id}: expected 8 machines")
        cutoffs = {int(m["cutoff"]) for m in machines.values()}
        if len(cutoffs) != 1:
            raise ValueError(f"{episode_id}: inconsistent cutoffs {cutoffs}")
        cutoff = cutoffs.pop()
        max_tick = 0
        for machine_id in machine_ids:
            for tick, _ in observed.get((episode_id, machine_id), ()):
                if tick >= cutoff:
                    raise ValueError(f"history tick {tick} >= cutoff {cutoff}")
                max_tick = max(max_tick, tick)
        obs = np.full((max_tick + 1, N_MACHINES), MISSING, dtype=np.int8)
        for slot, machine_id in enumerate(machine_ids):
            for tick, payout_idx in observed.get((episode_id, machine_id), ()):
                obs[tick, slot] = payout_idx
        machine_types = np.array(
            [TYPE_TO_IDX[machines[m]["machine_type"]] for m in machine_ids], dtype=np.int64
        )
        episode_index[episode_id] = len(episodes)
        episodes.append(Episode(episode_id, machine_ids, machine_types, obs, cutoff))
    return SplitData(episodes, episode_index, queries)


def load_test_split(data_dir: str | Path) -> SplitData:
    return load_public_split(data_dir, "test")
