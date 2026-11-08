"""Frozen t150 estimators. No training and no alternate ensemble score.

Canonical R is the kit contest functional: mean over the four conditions of
(mean episode squared error) / (mean reference episode squared error).
Episode squared error is the mean over that episode's queries of the
five-class squared error after kit row normalization. Resamples recombine
those episode sufficient statistics. They do not average per-episode ratios.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

CONDITIONS = ("stable", "short_history", "drift", "shared_change")
N_EPISODES = 48
N_REPLICATES = 10000
BOOTSTRAP_SEED = 15020261108
TIE_ATOL = 1e-12
REPRODUCTION_ATOL = 1e-12


def kit_probabilities(values) -> tuple[float, ...]:
    vals = tuple(float(v) for v in values)
    if len(vals) != 5 or any(not math.isfinite(v) or v < 0.0 for v in vals):
        raise RuntimeError("probability row is not five finite nonnegative values")
    total = math.fsum(vals)
    if not total > 0.0:
        raise RuntimeError("probability mass is not positive")
    if abs(total - 1.0) > 1e-6 and any(v > 1.0 for v in vals):
        raise RuntimeError("probability row is outside the kit contract")
    return tuple(v / total for v in vals)


def squared_error(prediction, truth) -> float:
    left = kit_probabilities(prediction)
    right = kit_probabilities(truth)
    return math.fsum((a - b) ** 2 for a, b in zip(left, right))


def episode_mean_errors(query_errors: dict[str, float], query_to_episode: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for query_id, error in query_errors.items():
        grouped[query_to_episode[query_id]].append(error)
    return {episode: math.fsum(rows) / len(rows) for episode, rows in grouped.items()}


def condition_mean(episode_errors: dict[str, float], episode_to_condition: dict[str, str], draws: list[str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for episode in draws:
        grouped[episode_to_condition[episode]].append(episode_errors[episode])
    if set(grouped) != set(CONDITIONS):
        missing = sorted(set(CONDITIONS) - set(grouped))
        raise RuntimeError(f"resample is missing conditions {missing}")
    return {condition: math.fsum(grouped[condition]) / len(grouped[condition]) for condition in CONDITIONS}


def canonical_r(model_errors: dict[str, float], reference_errors: dict[str, float],
                episode_to_condition: dict[str, str], draws: list[str]) -> float:
    """Ratio of means within condition, then the unweighted mean of the four ratios."""
    model = condition_mean(model_errors, episode_to_condition, draws)
    reference = condition_mean(reference_errors, episode_to_condition, draws)
    ratios = [model[condition] / max(1e-9, reference[condition]) for condition in CONDITIONS]
    return math.fsum(ratios) / len(CONDITIONS)


def incorrect_mean_of_episode_ratios(model_errors: dict[str, float], reference_errors: dict[str, float],
                                    draws: list[str]) -> float:
    """The estimator the protocol forbids. Kept only so tests can show it differs."""
    ratios = [model_errors[episode] / max(1e-9, reference_errors[episode]) for episode in draws]
    return math.fsum(ratios) / len(ratios)


def ordered_episode_ids(episode_ids) -> list[str]:
    ids = list(episode_ids)
    if len(ids) != N_EPISODES or len(set(ids)) != N_EPISODES:
        raise RuntimeError(f"expected {N_EPISODES} distinct episodes, got {len(ids)}")
    return sorted(ids)


def bootstrap_draws(n_replicates: int = N_REPLICATES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Protocol names np.default_rng. NumPy 2.5.4 has no numpy.default_rng.

    numpy.random.default_rng is that Generator constructor. Seed, replicate
    count, and the 48-episode draw size stay the predeclared values.
    """
    if n_replicates != N_REPLICATES or seed != BOOTSTRAP_SEED:
        raise RuntimeError("bootstrap size or seed is not the predeclared value")
    if hasattr(np, "default_rng"):
        raise RuntimeError("numpy.default_rng appeared; re-audit the locked constructor before any draw")
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_EPISODES, size=(n_replicates, N_EPISODES), dtype=np.int64)


def paired_bootstrap(model_errors: dict[str, dict[str, float]], reference_errors: dict[str, float],
                     episode_to_condition: dict[str, str], episode_ids: list[str]) -> dict:
    ids = ordered_episode_ids(episode_ids)
    draws = bootstrap_draws()
    names = list(model_errors)
    columns = {name: np.empty(N_REPLICATES, dtype=np.float64) for name in names}
    skipped = 0
    for replicate, choice in enumerate(draws):
        sample = [ids[int(index)] for index in choice]
        try:
            for name in names:
                columns[name][replicate] = canonical_r(model_errors[name], reference_errors, episode_to_condition, sample)
        except RuntimeError:
            skipped += 1
            for name in names:
                columns[name][replicate] = math.nan
    if skipped:
        raise RuntimeError(f"{skipped} bootstrap replicates missed a condition; no alternate interval is defined")
    return {"episode_order": ids, "draws": draws, "replicate_R": columns}


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    if values.shape != (N_REPLICATES,) or not np.isfinite(values).all():
        raise RuntimeError("interval input is not 10000 finite replicates")
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return {"low": float(low), "high": float(high), "method": "linear", "levels": [0.025, 0.975]}


def leave_one_out(model_errors: dict[str, float], other_errors: dict[str, float], reference_errors: dict[str, float],
                  episode_to_condition: dict[str, str], episode_ids: list[str]) -> list[dict]:
    ids = ordered_episode_ids(episode_ids)
    full = canonical_r(model_errors, reference_errors, episode_to_condition, ids) - canonical_r(
        other_errors, reference_errors, episode_to_condition, ids)
    rows = []
    for held in ids:
        kept = [episode for episode in ids if episode != held]
        delta = canonical_r(model_errors, reference_errors, episode_to_condition, kept) - canonical_r(
            other_errors, reference_errors, episode_to_condition, kept)
        rows.append({
            "held_out": held,
            "delta_R": delta,
            "full_delta_R": full,
            "sign_reversal": (full < 0.0 and delta > 0.0) or (full > 0.0 and delta < 0.0),
        })
    return rows


def error_gap_summary(left: dict[str, float], right: dict[str, float], episode_ids: list[str]) -> dict:
    """Descriptive gaps on the episode squared-error scale. This sum is not canonical R."""
    gaps = [(episode, left[episode] - right[episode]) for episode in ordered_episode_ids(episode_ids)]
    gross = -math.fsum(gap for _, gap in gaps if gap < 0.0)
    net = -math.fsum(gap for _, gap in gaps)
    ranked = sorted(gaps, key=lambda item: (item[1], item[0]))
    wins = sum(gap < -TIE_ATOL for _, gap in gaps)
    losses = sum(gap > TIE_ATOL for _, gap in gaps)
    ties = len(gaps) - wins - losses

    def pack(rows: list[tuple[str, float]]) -> dict:
        gained = -math.fsum(gap for _, gap in rows if gap < 0.0)
        return {
            "episodes": [episode for episode, _ in rows],
            "gaps": [gap for _, gap in rows],
            "share_of_gross_error_gain": None if gross == 0.0 else gained / gross,
            "share_of_net_error_gap_sum": None if net == 0.0 else (-math.fsum(gap for _, gap in rows)) / net,
        }

    return {
        "scale": "episode_mean_squared_error_not_canonical_R",
        "wins_left_lower_error": wins,
        "losses": losses,
        "ties_abs_le_1e-12": ties,
        "gross_error_gain": gross,
        "net_error_gap_sum_left_minus_right_negated": net,
        "largest_1": pack(ranked[:1]),
        "largest_3": pack(ranked[:3]),
    }


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


def probability_versus_logit_gap(prob_a: np.ndarray, prob_b: np.ndarray,
                                logit_a: np.ndarray, logit_b: np.ndarray) -> float:
    probability_mean = 0.5 * (prob_a + prob_b)
    logit_mean = softmax_rows(0.5 * (logit_a + logit_b))
    return float(np.max(np.abs(probability_mean - logit_mean)))


def synthetic_logit_gap() -> float:
    prob_a = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    prob_b = np.array([[0.9, 0.025, 0.025, 0.025, 0.025]])
    logit_a = np.zeros((1, 5))
    logit_b = np.array([[8.0, 0.0, 0.0, 0.0, 0.0]])
    gap = probability_versus_logit_gap(prob_a, prob_b, logit_a, logit_b)
    if gap <= 1e-12:
        raise RuntimeError("synthetic logit and probability means unexpectedly agree")
    return gap


def synthetic_ratio_counterexample() -> dict[str, float]:
    episodes = [f"e{i}" for i in range(N_EPISODES)]
    # Two conditions would be illegal; use all four, with unequal reference errors in one.
    conditions = {episode: CONDITIONS[index % 4] for index, episode in enumerate(episodes)}
    reference = {episode: 1.0 for episode in episodes}
    model = {episode: 1.0 for episode in episodes}
    reference["e0"] = 1.0
    reference["e4"] = 100.0
    model["e0"] = 1.0
    model["e4"] = 1.0
    canonical = canonical_r(model, reference, conditions, episodes)
    forbidden = incorrect_mean_of_episode_ratios(model, reference, episodes)
    if abs(canonical - forbidden) <= 1e-12:
        raise RuntimeError("synthetic ratio counterexample did not separate the estimators")
    return {"canonical_r": canonical, "mean_of_episode_ratios": forbidden}
