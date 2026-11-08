"""Batch prediction at query cutoffs for a trained model."""
from __future__ import annotations

import torch

from .data import SplitData
from .models import NextSpinTransformer
from .windows import Example, WindowCollator


@torch.no_grad()
def predict_split(model: NextSpinTransformer, data: SplitData, device: str,
                  window: int = 64, batch_size: int = 32) -> dict[str, list[float]]:
    """Returns {query_id: [p_0,p_1,p_2,p_5,p_20]} for every query of the split.

    Predicts one joint distribution per episode (all 8 machines at the cutoff)
    and maps each query to its machine slot. query_ids come from query rows.
    """
    was_training = model.training
    model.eval()
    arch = model.arch
    collator = WindowCollator(data.episodes, arch, window)
    by_episode: dict[str, list[dict[str, str]]] = {}
    for row in data.queries:
        by_episode.setdefault(row["episode_id"], []).append(row)
    examples: list[Example] = []
    groups: list[list[dict[str, str]]] = []
    for episode_id in sorted(by_episode):
        rows = by_episode[episode_id]
        cutoffs = {int(r["cutoff"]) for r in rows}
        assert len(cutoffs) == 1, f"{episode_id}: multiple cutoffs"
        examples.append(Example(data.episode_index[episode_id], cutoffs.pop(), None))
        groups.append(rows)
    predictions: dict[str, list[float]] = {}
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        batch = collator(chunk)
        batch = {k: v.to(device) for k, v in batch.items()}
        probs = torch.softmax(model.forward(batch), dim=-1).float().cpu().numpy()
        for i, rows in enumerate(groups[start:start + batch_size]):
            episode = data.episodes[data.episode_index[rows[0]["episode_id"]]]
            for row in rows:
                slot = episode.slot_of(row["machine_id"])
                predictions[row["query_id"]] = [float(v) for v in probs[i, slot]]
    if was_training:
        model.train()
    return predictions
