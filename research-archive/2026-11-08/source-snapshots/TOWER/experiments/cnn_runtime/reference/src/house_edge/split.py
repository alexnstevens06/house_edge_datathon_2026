"""Deterministic TUNE/CONFIRM split of validation episodes, stratified by condition."""
from __future__ import annotations

import random

from .data import episode_conditions

SPLIT_SEED = 0


def tune_confirm_split(validation_dir, seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Half the validation episodes per condition -> TUNE (model selection),
    other half -> CONFIRM (reported once). Deterministic given seed."""
    conditions = episode_conditions(validation_dir)
    by_condition: dict[str, list[str]] = {}
    for episode_id in sorted(conditions):
        by_condition.setdefault(conditions[episode_id], []).append(episode_id)
    rng = random.Random(seed)
    tune, confirm = [], []
    for condition in sorted(by_condition):
        episodes = list(by_condition[condition])
        rng.shuffle(episodes)
        half = len(episodes) // 2
        tune.extend(episodes[:half])
        confirm.extend(episodes[half:])
    return {"tune": sorted(tune), "confirm": sorted(confirm), "seed": seed,
            "by_condition": {c: sorted(v) for c, v in sorted(by_condition.items())}}
