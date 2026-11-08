"""Synthetic checks only. They do not score the study tune split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import analysis as A
from slot_forecasting.contest import contest_score
from slot_forecasting.schema import Target


def test_canonical_matches_contest_score() -> None:
    conditions = A.CONDITIONS
    episodes = [f"ep{i}" for i in range(4)]
    queries = [f"q{i}" for i in range(4)]
    groups = {queries[i]: conditions[i] for i in range(4)}
    episode_of = {queries[i]: episodes[i] for i in range(4)}
    truth = (1.0, 0.0, 0.0, 0.0, 0.0)
    targets = tuple(Target(queries[i], episodes[i], truth, 0) for i in range(4))
    reference = {queries[i]: (0.2, 0.2, 0.2, 0.2, 0.2) for i in range(4)}
    model = {queries[i]: (0.6, 0.1, 0.1, 0.1, 0.1) for i in range(4)}
    model[queries[1]] = (0.9, 0.025, 0.025, 0.025, 0.025)
    official = contest_score(model, targets, groups, reference, max_points=60)
    query_errors = {qid: A.squared_error(model[qid], truth) for qid in queries}
    ref_errors = {qid: A.squared_error(reference[qid], truth) for qid in queries}
    model_ep = A.episode_mean_errors(query_errors, episode_of)
    ref_ep = A.episode_mean_errors(ref_errors, episode_of)
    cond = {episodes[i]: conditions[i] for i in range(4)}
    rebuilt = A.canonical_r(model_ep, ref_ep, cond, episodes)
    if abs(rebuilt - official["mean_loss_ratio"]) > 1e-12:
        raise SystemExit(f"reconstruction {rebuilt} != contest {official['mean_loss_ratio']}")


def test_ratio_is_not_the_mean_of_episode_ratios() -> None:
    gap = A.synthetic_ratio_counterexample()
    if abs(gap["canonical_r"] - gap["mean_of_episode_ratios"]) <= 1e-12:
        raise SystemExit(gap)


def test_logit_mean_is_not_scored_and_differs() -> None:
    gap = A.synthetic_logit_gap()
    if gap <= 1e-6:
        raise SystemExit(gap)


def test_bootstrap_lock_and_same_draws() -> None:
    if hasattr(np, "default_rng") or not hasattr(np.random, "default_rng"):
        raise SystemExit(f"unexpected numpy rng surface {np.__version__}")
    draws = A.bootstrap_draws()
    again = np.random.default_rng(A.BOOTSTRAP_SEED).integers(0, 48, size=(10000, 48), dtype=np.int64)
    if draws.shape != (10000, 48) or not np.array_equal(draws, again):
        raise SystemExit("bootstrap draws are not the locked stream")
    if draws.shape[1] == 384:
        raise SystemExit("query-level bootstrap is forbidden")
    ids = [f"e{i:02d}" for i in range(48)]
    conditions = {episode: A.CONDITIONS[index % 4] for index, episode in enumerate(ids)}
    reference = {episode: 1.0 + (index % 5) for index, episode in enumerate(ids)}
    left = {episode: 0.5 + 0.01 * index for index, episode in enumerate(ids)}
    right = {episode: 0.7 + 0.02 * (index % 7) for index, episode in enumerate(ids)}
    out = A.paired_bootstrap({"left": left, "right": right}, reference, conditions, ids)
    if not np.array_equal(out["draws"], draws):
        raise SystemExit("paired models did not share draws")
    interval = A.percentile_interval(out["replicate_R"]["left"] - out["replicate_R"]["right"])
    if interval["method"] != "linear" or not (interval["low"] <= interval["high"]):
        raise SystemExit(interval)


def test_missing_condition_is_refused() -> None:
    ids = [f"e{i:02d}" for i in range(48)]
    conditions = {episode: A.CONDITIONS[min(index // 16, 2)] for index, episode in enumerate(ids)}
    # indices 0..47 map to conditions 0,1,2 only. Condition 3 is absent even on the full set.
    errors = {episode: 1.0 for episode in ids}
    try:
        A.canonical_r(errors, errors, conditions, ids)
    except RuntimeError as exc:
        if "missing conditions" not in str(exc):
            raise
        return
    raise SystemExit("missing condition was accepted")


def test_probability_rows_refuse_nonfinite() -> None:
    try:
        A.kit_probabilities((0.2, 0.2, 0.2, 0.2, float("nan")))
    except RuntimeError:
        return
    raise SystemExit("nonfinite probability was accepted")


def main() -> None:
    test_canonical_matches_contest_score()
    test_ratio_is_not_the_mean_of_episode_ratios()
    test_logit_mean_is_not_scored_and_differs()
    test_bootstrap_lock_and_same_draws()
    test_missing_condition_is_refused()
    test_probability_rows_refuse_nonfinite()
    print("ANALYSIS_TESTS_OK")


if __name__ == "__main__":
    main()
