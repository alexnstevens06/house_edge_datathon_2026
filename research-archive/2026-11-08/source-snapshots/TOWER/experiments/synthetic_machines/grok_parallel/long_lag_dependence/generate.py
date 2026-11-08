#!/usr/bin/env python3
"""Invented long-lag slot family. RESEARCH_ONLY.

Eligibility is unresolved: contest rules require organizer-released data and
do not state a verified synthetic-data exception. Do not replace real
train, validation, or test with this output.

Assumptions (toy, not fit to organizer logs):
- Eight slots. machine_id is the slot index 0..7.
- Types are fixed by slot, not estimated.
- Each machine's latent payout at tick t depends only on its own latent
  payouts at lags 24 and 48 (skipped when t is too early).
- Ticks are synchronous: tick t reads only ticks < t, never another
  machine's outcome at t.
- A missing observation is unknown. It is not payout 0 and it is not the
  latent payout. Observable-history probabilities drop that lag. Latent-oracle
  probabilities still use it. Those two distributions differ on gaps.
- Query `signal` is not invented. Queries exist only at the prediction
  cutoff (the tick after the generated history) and carry signal null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path

FAMILY = "long_lag_dependence"
LABEL = "RESEARCH_ONLY"
ELIGIBILITY = "unresolved"
MAX_RECORDS = 8192

# Public contest index maps (schema), not fitted frequencies.
PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
INDEX_TO_PAYOUT = PAYOUTS
MISSING = -1  # observation index; payout index 0 still means a real 0
MACHINE_TYPES = ("balanced", "jackpot", "steady")
TYPE_TO_INDEX = {"balanced": 0, "jackpot": 1, "steady": 2}

# Invented slot map. machine_id == slot.
SLOT_TYPE = (
    "steady",
    "steady",
    "balanced",
    "balanced",
    "balanced",
    "jackpot",
    "jackpot",
    "jackpot",
)

LAG = 24
LAG2 = 48
W1 = Fraction(1, 2)
W2 = Fraction(1, 4)

# Percent masses, each row sums to 100. Cold (index 0) through hot (index 4).
_BASE_PCT = {
    "steady": (70, 18, 7, 4, 1),
    "balanced": (40, 28, 16, 10, 6),
    "jackpot": (50, 15, 12, 8, 15),
}
_TILT_PCT = (
    (80, 12, 5, 2, 1),
    (60, 22, 10, 6, 2),
    (45, 25, 15, 10, 5),
    (30, 20, 18, 18, 14),
    (18, 14, 16, 22, 30),
)


def _pct_row(parts: tuple[int, ...]) -> tuple[Fraction, ...]:
    if len(parts) != 5 or sum(parts) != 100 or any(x <= 0 for x in parts):
        raise RuntimeError("probability table must be 5 positive percents summing to 100")
    return tuple(Fraction(x, 100) for x in parts)


BASE = {name: _pct_row(parts) for name, parts in _BASE_PCT.items()}
TILT = tuple(_pct_row(parts) for parts in _TILT_PCT)


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def simplex_floats(parts: tuple[Fraction, ...]) -> list[float]:
    """Float vector that sums to 1. Last entry absorbs binary residue."""
    head = [float(p) for p in parts[:-1]]
    last = 1.0 - math.fsum(head)
    if last < 0.0 and abs(last) < 1e-15:
        last = 0.0
    return head + [last]


def distribution(machine_type: str, lag1: int | None, lag2: int | None) -> list[float]:
    """Convex mix of the type base with long-lag tilts.

    lag1 / lag2 are payout indexes 0..4, or None when that lag is not applied.
    None means "too early" or, for the observable filter, "unobserved".
    Do not pass MISSING (-1). A real zero payout is index 0, not None.
    """
    if machine_type not in BASE:
        raise ValueError(f"unknown machine_type {machine_type!r}")
    for name, lag in (("lag1", lag1), ("lag2", lag2)):
        if lag is not None and lag not in (0, 1, 2, 3, 4):
            raise ValueError(f"{name} must be a payout index 0..4 or None, not {lag!r}")
    probs = BASE[machine_type]
    if lag1 is not None:
        probs = tuple((1 - W1) * a + W1 * b for a, b in zip(probs, TILT[lag1]))
    if lag2 is not None:
        probs = tuple((1 - W2) * a + W2 * b for a, b in zip(probs, TILT[lag2]))
    return simplex_floats(probs)


def oracle_lags(latent_index_by_tick: list[int], tick: int) -> tuple[int | None, int | None]:
    """Latent-oracle lags. Uses the spin even when that tick was unobserved."""
    lag1 = latent_index_by_tick[tick - LAG] if tick >= LAG else None
    lag2 = latent_index_by_tick[tick - LAG2] if tick >= LAG2 else None
    return lag1, lag2


def observable_lags(observation_index_by_tick: list[int], tick: int) -> tuple[int | None, int | None]:
    """Observable-history lags. MISSING drops the lag; it is not rewritten to 0."""

    def one(lag: int) -> int | None:
        if tick < lag:
            return None
        value = observation_index_by_tick[tick - lag]
        if value == MISSING:
            return None
        return value

    return one(LAG), one(LAG2)


def _episode_stream(seed: int, episode: int) -> int:
    """64-bit stream id. Not Python's salted str hash."""
    x = (seed + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    x ^= (episode + 1) * 0xBF58476D1CE4E5B9
    return x & ((1 << 64) - 1)


def is_observed(seed: int, episode: int, tick: int, machine_id: int) -> bool:
    """Deterministic 4/5 mask. Independent of payout and of ticks > this one."""
    x = (seed + 0x9E3779B1) & 0xFFFFFFFF
    x ^= (episode + 1) * 0x85EBCA6B
    x ^= (tick + 1) * 0xC2B2AE35
    x ^= (machine_id + 1) * 0x27D4EB2F
    x &= 0xFFFFFFFF
    x ^= x >> 16
    return (x % 5) != 0


def _sample_index(rng: random.Random, probs: list[float]) -> int:
    draw = rng.random()
    acc = 0.0
    for index, prob in enumerate(probs):
        acc += prob
        if draw <= acc:
            return index
    return len(probs) - 1


def _mappings() -> dict:
    return {
        "payouts_ordered": list(PAYOUTS),
        "payout_to_index": {str(payout): index for payout, index in PAYOUT_TO_INDEX.items()},
        "index_to_payout": list(INDEX_TO_PAYOUT),
        "missing_observation_index": MISSING,
        "missing_means": "unknown_not_payout_0",
        "machine_id_is_slot": True,
        "slot_to_machine_type": {str(slot): SLOT_TYPE[slot] for slot in range(len(SLOT_TYPE))},
        "type_to_index": dict(TYPE_TO_INDEX),
        "type_index_order": list(MACHINE_TYPES),
        "lags": [LAG, LAG2],
        "mix_weights": {"lag_24": "1/2", "lag_48": "1/4"},
        "query_signal_policy": (
            "signal is only meaningful on a real prediction-cutoff query; "
            "this family does not invent signal values and does not write signal onto history"
        ),
        "truth_definition": "latent_oracle_conditional_given_own_unobserved_lags",
        "observable_definition": "same_formula_with_missing_lags_dropped",
    }


def generate(seed: int = 0, n_episodes: int = 8, n_ticks: int = 96, n_machines: int = 8) -> dict:
    """Return records, latent-oracle truth, observable-history probs, and cutoff queries.

    records include every slot-tick. `payout` is the latent spin. Consumers must
    treat `observed` false as unknown: use `observation_index` (-1), never the
    latent payout and never a filled-in 0.
    """
    seed = _as_int("seed", seed)
    n_episodes = _as_int("n_episodes", n_episodes)
    n_ticks = _as_int("n_ticks", n_ticks)
    n_machines = _as_int("n_machines", n_machines)
    if n_episodes < 1 or n_ticks < 1:
        raise ValueError("n_episodes and n_ticks must be >= 1")
    if n_machines != 8:
        raise ValueError("n_machines must be 8 so machine_id is slot 0..7")
    record_count = n_episodes * n_ticks * n_machines
    if record_count > MAX_RECORDS:
        raise ValueError(f"refusing to emit {record_count} records; cap is {MAX_RECORDS}")

    records: list[dict] = []
    truth: list[dict] = []
    observable: list[dict] = []
    queries: list[dict] = []

    for episode in range(n_episodes):
        # Own stream per episode. Extra ticks or later episodes do not move earlier draws.
        rng = random.Random(_episode_stream(seed, episode))
        episode_id = f"longlag-s{seed}-e{episode:04d}"
        # Latent payout index and observation index for ticks already finished.
        latent: list[list[int]] = []
        observed_idx: list[list[int]] = []
        for tick in range(n_ticks):
            latent_row = [0] * n_machines
            obs_row = [MISSING] * n_machines
            # All eight machines read only ticks < tick. Same-tick outcomes are not inputs.
            for machine_id in range(n_machines):
                machine_type = SLOT_TYPE[machine_id]
                own_latent = [row[machine_id] for row in latent]
                own_obs = [row[machine_id] for row in observed_idx]
                lag1, lag2 = oracle_lags(own_latent, tick)
                o1, o2 = observable_lags(own_obs, tick)
                oracle_probs = distribution(machine_type, lag1, lag2)
                obs_probs = distribution(machine_type, o1, o2)
                payout_index = _sample_index(rng, oracle_probs)
                seen = is_observed(seed, episode, tick, machine_id)
                payout = INDEX_TO_PAYOUT[payout_index]
                observation_index = payout_index if seen else MISSING
                latent_row[machine_id] = payout_index
                obs_row[machine_id] = observation_index
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "machine_type": machine_type,
                        "payout": payout,
                        "payout_index": payout_index,
                        "observed": seen,
                        "observation_index": observation_index,
                    }
                )
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "probabilities": oracle_probs,
                    }
                )
                observable.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "probabilities": obs_probs,
                    }
                )
            latent.append(latent_row)
            observed_idx.append(obs_row)
        cutoff = n_ticks  # next spin; history ticks are strictly earlier
        for machine_id in range(n_machines):
            queries.append(
                {
                    "query_id": f"{episode_id}-m{machine_id}",
                    "episode_id": episode_id,
                    "machine_id": machine_id,
                    "machine_type": SLOT_TYPE[machine_id],
                    "cutoff": cutoff,
                    "signal": None,
                }
            )

    return {
        "records": records,
        "truth": truth,
        "observable": observable,
        "queries": queries,
        "meta": {
            "family": FAMILY,
            "label": LABEL,
            "eligibility": ELIGIBILITY,
            "real_only_fallback": True,
            "seed": seed,
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
            "record_count": record_count,
            "prediction_cutoff": n_ticks,
            "mappings": _mappings(),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def write_dataset(out_dir: Path, seed: int, n_episodes: int, n_ticks: int) -> dict:
    payload = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    truth_path = out_dir / "truth.jsonl"
    observable_path = out_dir / "observable.jsonl"
    queries_path = out_dir / "queries.jsonl"
    _write_jsonl(records_path, payload["records"])
    _write_jsonl(truth_path, payload["truth"])
    _write_jsonl(observable_path, payload["observable"])
    _write_jsonl(queries_path, payload["queries"])
    here = Path(__file__).resolve().parent
    manifest = {
        "family": FAMILY,
        "label": LABEL,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": True,
        "note": (
            "Rules say use organizer-released data. No verified synthetic exception. "
            "Keep the real-only training fallback. This output is research-only."
        ),
        "seed": seed,
        "settings": {
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": 8,
            "lags": [LAG, LAG2],
            "max_records": MAX_RECORDS,
            "record_count": payload["meta"]["record_count"],
            "prediction_cutoff": n_ticks,
            "assumptions": [
                "Invented percent tables and lags 24 and 48; not fit on organizer data.",
                "Oracle probabilities condition on own latent lags, including unobserved spins.",
                "Observable probabilities drop missing lags and never treat them as payout 0.",
                "No contemporaneous cross-machine inputs.",
                "Query signal is null; no historical signal series is invented.",
            ],
        },
        "mappings": _mappings(),
        "source_hashes": {
            "generate.py": _sha256(here / "generate.py"),
            "test_generator.py": _sha256(here / "test_generator.py"),
        },
        "output_hashes": {
            "records.jsonl": _sha256(records_path),
            "truth.jsonl": _sha256(truth_path),
            "observable.jsonl": _sha256(observable_path),
            "queries.jsonl": _sha256(queries_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="long_lag_dependence synthetic generator (RESEARCH_ONLY)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = write_dataset(Path(args.out), args.seed, args.episodes, args.ticks)
    print(json.dumps({"label": manifest["label"], "record_count": manifest["settings"]["record_count"], "out": args.out}))


if __name__ == "__main__":
    main()
