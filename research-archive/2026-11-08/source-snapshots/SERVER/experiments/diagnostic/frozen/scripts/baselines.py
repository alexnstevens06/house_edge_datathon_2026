#!/usr/bin/env python
"""Baselines: kit starter frequency methods + a shrinkage baseline, scored like the models.

Usage: .venv/bin/python scripts/baselines.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "kit"))

import numpy as np

from house_edge.data import PAYOUTS, load_split
from house_edge.scoring import (load_official_validation, score_predictions,
                                write_submission_csv)
from house_edge.split import tune_confirm_split
from house_edge.data import subset_split_data

from slot_forecasting.baseline import METHODS, FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels


def shrinkage_fit_predict(train_data, target_data, prior_strength=16.0, window=64):
    """Per-machine empirical payout distribution over the last `window` observed
    ticks before the cutoff, shrunk toward the pooled train-history marginal."""
    prior = np.zeros(len(PAYOUTS))
    for episode in train_data.episodes:
        obs = episode.obs
        prior += np.bincount(obs[obs >= 0].astype(np.int64), minlength=len(PAYOUTS))
    prior = prior / prior.sum()
    predictions = {}
    for row in target_data.queries:
        episode = target_data.episodes[target_data.episode_index[row["episode_id"]]]
        slot = episode.slot_of(row["machine_id"])
        cutoff = int(row["cutoff"])
        lo = max(0, cutoff - window)
        seg = episode.obs[lo:cutoff, slot]
        observed = seg[seg >= 0].astype(np.int64)
        counts = np.bincount(observed, minlength=len(PAYOUTS)).astype(np.float64)
        probs = (counts + prior_strength * prior) / (len(observed) + prior_strength)
        predictions[row["query_id"]] = [float(p) for p in probs]
    return predictions


def main() -> None:
    public = os.path.join(REPO, "kit", "public")
    out_dir = os.path.join(REPO, "runs", "baselines")
    os.makedirs(out_dir, exist_ok=True)

    train_data = load_split(os.path.join(public, "train"), with_labels=True)
    val_data = load_split(os.path.join(public, "validation"))
    test_data = load_split(os.path.join(public, "test"))
    official = load_official_validation(os.path.join(public, "validation"))
    split = tune_confirm_split(os.path.join(public, "validation"))
    splits_path = os.path.join(REPO, "runs", "splits.json")
    if not os.path.exists(splits_path):
        with open(splits_path, "w") as handle:
            handle.write(json.dumps(split, indent=2) + "\n")
    tune_set, confirm_set = set(split["tune"]), set(split["confirm"])

    train_queries, train_history = load_inputs(os.path.join(public, "train"))
    labels = read_labels(os.path.join(public, "train", "labels.csv"), train_queries)
    val_queries, val_history = load_inputs(os.path.join(public, "validation"))
    test_queries, test_history = load_inputs(os.path.join(public, "test"))

    results = {}
    for name in list(METHODS) + ["shrinkage64"]:
        if name == "shrinkage64":
            val_preds = shrinkage_fit_predict(train_data, val_data)
            test_preds = shrinkage_fit_predict(train_data, test_data)
        else:
            model = FrequencyForecaster(name).fit(train_queries, train_history, labels)
            val_preds = model.predict(val_queries, val_history)
            test_preds = model.predict(test_queries, test_history)
        write_submission_csv(os.path.join(out_dir, f"{name}_validation.csv"), val_preds)
        write_submission_csv(os.path.join(out_dir, f"{name}_test.csv"), test_preds)
        tune = score_predictions(val_preds, official, episode_subset=tune_set)
        confirm = score_predictions(val_preds, official, episode_subset=confirm_set)
        full = score_predictions(val_preds, official)
        results[name] = {"tune_R": tune["R"], "confirm_R": confirm["R"], "full_val_R": full["R"],
                         "full_val": full}
        print(name, {k: results[name][k] for k in ("tune_R", "confirm_R", "full_val_R")}, flush=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        handle.write(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
