#!/usr/bin/env python3
"""Abrupt-changepoint synthetic slot family.

RESEARCH_ONLY. kit/RULES.md says to use organizer-released data. No synthetic
exception is verified. Keep a real-only fallback for any submission training.

This is an invented toy. None of the probabilities, gaps, or signals are fit
to organizer history, labels, validation, or test.

Latent oracle (the ``truth`` rows)
    Each machine has a piecewise-constant payout distribution. A shared episode
    anchor plus a per-machine offset sets one changepoint. Before that tick the
    machine uses its start regime; at the changepoint the distribution flips
    once. The schedule is a function of (seed, episode, machine) only. It does
    not read payouts, so tick t does not depend on ticks >= t or on another
    machine's outcome at t.

Observable-history probabilities
    ``observable_history_probabilities`` is the empirical distribution of this
    machine's observed payouts at ticks strictly before t. Missing rows are
    skipped, not counted as payout 0. With no observations it returns a flat
    prior. That prior is not the latent oracle.

Query signal
    One signal in {-1, 0, 1} is stored on the episode query at the prediction
    cutoff only. Historical rows have no signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


FAMILY = "abrupt_changepoints"
ELIGIBILITY = "RESEARCH_ONLY"
SYNTHETIC_EXCEPTION_VERIFIED = False
REAL_ONLY_FALLBACK = True
MAX_RECORDS = 8192

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
INDEX_TO_PAYOUT = {0: 0, 1: 1, 2: 2, 3: 5, 4: 20}
PROBABILITY_COLUMNS = ("p_0", "p_1", "p_2", "p_5", "p_20")

MACHINE_TYPES = ("steady", "balanced", "jackpot")
TYPE_TO_INDEX = {"steady": 0, "balanced": 1, "jackpot": 2}
# Slot i uses this type. machine_id is the slot index.
SLOT_TYPES = (
    "steady",
    "steady",
    "balanced",
    "balanced",
    "balanced",
    "jackpot",
    "jackpot",
    "jackpot",
)

# Invented regime tables, ordered (0, 1, 2, 5, 20). Each pair sums to 1.
# Index 0 is the start regime for even slots; odd slots start in index 1.
REGIME_PROBS = {
    "steady": (
        (0.62, 0.28, 0.07, 0.02, 0.01),
        (0.48, 0.30, 0.14, 0.06, 0.02),
    ),
    "balanced": (
        (0.50, 0.22, 0.15, 0.09, 0.04),
        (0.35, 0.20, 0.22, 0.15, 0.08),
    ),
    "jackpot": (
        (0.70, 0.15, 0.08, 0.05, 0.02),
        (0.55, 0.12, 0.10, 0.08, 0.15),
    ),
}

# Flat prior used only by the observable-history estimator.
OBSERVABLE_PRIOR = (0.2, 0.2, 0.2, 0.2, 0.2)

# Dropout among ticks that are not structural gaps. Invented, not estimated.
OBSERVE_P = {"steady": 0.92, "balanced": 0.84, "jackpot": 0.75}

# Shared anchor is in [ANCHOR_LO, ANCHOR_HI). Offsets keep the jump abrupt.
ANCHOR_LO = 32
ANCHOR_HI = 64
OFFSETS = (-8, -4, 0, 0, 4, 8)

ASSUMPTIONS = (
    "One exogenous changepoint per machine. No gradual blend between regimes.",
    "Even slots start in regime 0 and odd slots start in regime 1, so the two "
    "steady slots (and other same-type pairs) begin in opposite regimes.",
    "The episode anchor is shared. Machine offsets are -8,-4,0,0,4,8 ticks.",
    "Anchor and offset ignore outcomes and do not depend on n_ticks.",
    "P(observe) and the structural gap (tick + 3*slot) % 11 == 0 are exogenous. "
    "A missing row still has a latent payout in the log; features must ignore it.",
    "The query signal is exogenous at the cutoff only. It is not a history feature.",
    "No parameter was fit on organizer train, validation, or test files.",
)


def machine_type_for_slot(machine_id: int) -> str:
    if isinstance(machine_id, bool) or not isinstance(machine_id, int):
        raise ValueError("machine_id must be an int")
    if machine_id < 0:
        raise ValueError("machine_id must be >= 0")
    return SLOT_TYPES[machine_id % len(SLOT_TYPES)]


def _mix(*parts: int) -> int:
    """Deterministic 64-bit mix. Independent streams per (episode, slot, tick)."""
    x = 0x9E3779B97F4A7C15
    for part in parts:
        x ^= part & 0xFFFFFFFFFFFFFFFF
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 30
        x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 27
    return x


def _as_int(name: str, value: int, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def episode_id_for(episode: int) -> str:
    return f"e{episode:04d}"


def parse_episode_index(episode_id: str) -> int:
    if not isinstance(episode_id, str) or len(episode_id) != 5 or episode_id[0] != "e":
        raise ValueError(f"unrecognized episode_id: {episode_id!r}")
    return int(episode_id[1:])


def latent_state(seed: int, episode: int, machine: int, tick: int) -> dict:
    """Regime at ``tick``. Uses the exogenous schedule only.

    The changepoint is applied at ``change_at`` (tick >= change_at is flipped).
    No payout is an input, including payouts at this tick or later ticks.
    """
    seed = _as_int("seed", seed)
    episode = _as_int("episode", episode, 0)
    machine = _as_int("machine", machine, 0)
    tick = _as_int("tick", tick, 0)
    machine_type = machine_type_for_slot(machine)
    anchor_rng = random.Random(_mix(seed, episode, 0xA11CE, 1))
    anchor = anchor_rng.randrange(ANCHOR_LO, ANCHOR_HI)
    offset_rng = random.Random(_mix(seed, episode, machine, 0x05FF, 2))
    offset = offset_rng.choice(OFFSETS)
    change_at = anchor + offset
    start_regime = machine % 2
    regime = start_regime if tick < change_at else 1 - start_regime
    return {
        "machine_type": machine_type,
        "slot": machine,
        "anchor": anchor,
        "offset": offset,
        "change_at": change_at,
        "start_regime": start_regime,
        "regime": regime,
    }


def probabilities_at(seed: int, episode: int, machine: int, tick: int) -> tuple[float, float, float, float, float]:
    """Latent-oracle distribution at tick. Not the observable-history estimate."""
    state = latent_state(seed, episode, machine, tick)
    return REGIME_PROBS[state["machine_type"]][state["regime"]]


def _draw_payout(rng: random.Random, probs: tuple[float, ...]) -> int:
    roll = rng.random()
    cumulative = 0.0
    for payout, prob in zip(PAYOUTS, probs):
        cumulative += prob
        if roll < cumulative:
            return payout
    return PAYOUTS[-1]


def sample_payout(seed: int, episode: int, machine: int, tick: int, probs: tuple[float, ...]) -> int:
    rng = random.Random(_mix(seed, episode, machine, tick, 0xB10))
    return _draw_payout(rng, probs)


def sample_observed(seed: int, episode: int, machine: int, tick: int, machine_type: str) -> bool:
    """Structural gaps are always missing. Other ticks use an exogenous coin.

    The coin does not depend on the payout drawn for this cell.
    """
    if (tick + 3 * machine) % 11 == 0:
        return False
    rng = random.Random(_mix(seed, episode, machine, tick, 0x0B5))
    return rng.random() < OBSERVE_P[machine_type]


def sample_signal(seed: int, episode: int, machine: int, cutoff: int) -> int:
    """Query covariate at the prediction cutoff. Not defined for earlier ticks."""
    rng = random.Random(_mix(seed, episode, machine, cutoff, 0x516))
    return (-1, 0, 1)[rng.randrange(3)]


def encode_observation(payout: int, observed: bool) -> dict:
    """Map one cell to a payout index and an explicit mask.

    Missing is index None, mask 0, and a zero one-hot. That is not payout 0,
    which is index 0, mask 1, and one-hot (1, 0, 0, 0, 0).
    """
    if not isinstance(observed, bool):
        raise ValueError("observed must be a bool")
    if not observed:
        return {"payout_index": None, "mask": 0, "one_hot": (0, 0, 0, 0, 0)}
    if payout not in PAYOUT_TO_INDEX:
        raise ValueError(f"payout must be one of {PAYOUTS}")
    index = PAYOUT_TO_INDEX[payout]
    one_hot = [0, 0, 0, 0, 0]
    one_hot[index] = 1
    return {"payout_index": index, "mask": 1, "one_hot": tuple(one_hot)}


def observable_history_probabilities(
    records: list[dict],
    episode_id: str,
    machine_id: int,
    tick: int,
) -> tuple[float, float, float, float, float]:
    """Empirical distribution from this machine's observed ticks strictly < tick.

    Missing rows are unknown and are not counted. Other machines are not used.
    Ticks >= ``tick`` are not used. This is not the latent oracle.
    """
    counts = [0, 0, 0, 0, 0]
    for row in records:
        if row["episode_id"] != episode_id or row["machine_id"] != machine_id:
            continue
        if row["tick"] >= tick or not row["observed"]:
            continue
        counts[PAYOUT_TO_INDEX[row["payout"]]] += 1
    total = sum(counts)
    if total == 0:
        return OBSERVABLE_PRIOR
    return tuple(count / total for count in counts)


def history_features(records: list[dict], cutoff: int) -> tuple:
    """Legal history for a prediction at ``cutoff``.

    Every machine contributes only ticks < cutoff. Unobserved payouts do not
    appear; the mask does. The query signal is not a history feature.
    """
    features = []
    for row in records:
        if row["tick"] >= cutoff:
            continue
        encoded = encode_observation(row["payout"], row["observed"])
        features.append(
            (
                row["episode_id"],
                row["tick"],
                row["machine_id"],
                encoded["payout_index"],
                encoded["mask"],
            )
        )
    return tuple(sorted(features))


def signal_at(queries: list[dict], episode_id: str, machine_id: int, tick: int):
    """Return the signal only when tick is that machine's query cutoff."""
    for query in queries:
        if query["episode_id"] == episode_id and query["machine_id"] == machine_id:
            if tick == query["cutoff"]:
                return query["signal"]
            return None
    raise KeyError((episode_id, machine_id))


def generate(
    seed: int = 0,
    n_episodes: int = 8,
    n_ticks: int = 96,
    n_machines: int = 8,
) -> dict:
    """Return records, latent-oracle truth, and cutoff queries.

    Default size is 8 * 96 * 8 = 6144 records, under MAX_RECORDS.
    """
    seed = _as_int("seed", seed)
    n_episodes = _as_int("n_episodes", n_episodes, 1)
    n_ticks = _as_int("n_ticks", n_ticks, 1)
    n_machines = _as_int("n_machines", n_machines, 1)
    n_records = n_episodes * n_ticks * n_machines
    if n_records > MAX_RECORDS:
        raise ValueError(
            f"refusing to generate {n_records} records; cap is {MAX_RECORDS}"
        )

    records: list[dict] = []
    truth: list[dict] = []
    queries: list[dict] = []
    cutoff = n_ticks - 1

    for episode in range(n_episodes):
        episode_id = episode_id_for(episode)
        for machine in range(n_machines):
            machine_type = machine_type_for_slot(machine)
            queries.append(
                {
                    "query_id": f"{episode_id}-m{machine}",
                    "episode_id": episode_id,
                    "machine_id": machine,
                    "slot": machine,
                    "machine_type": machine_type,
                    "signal": sample_signal(seed, episode, machine, cutoff),
                    "cutoff": cutoff,
                }
            )
        for tick in range(n_ticks):
            # Inner loop reads no payout from this tick. Streams are per cell.
            for machine in range(n_machines):
                machine_type = machine_type_for_slot(machine)
                probs = probabilities_at(seed, episode, machine, tick)
                payout = sample_payout(seed, episode, machine, tick, probs)
                observed = sample_observed(seed, episode, machine, tick, machine_type)
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine,
                        "machine_type": machine_type,
                        "payout": payout,
                        "observed": observed,
                    }
                )
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine,
                        "probabilities": [float(p) for p in probs],
                    }
                )

    return {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "synthetic_exception_verified": SYNTHETIC_EXCEPTION_VERIFIED,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "records": records,
        "truth": truth,
        "queries": queries,
        "settings": {
            "seed": seed,
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
            "max_records": MAX_RECORDS,
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_line(row: dict) -> bytes:
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_dataset(out_dir: Path, data: dict, source_paths: list[Path]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    truth_path = out_dir / "truth.jsonl"
    manifest_path = out_dir / "manifest.json"

    records_blob = b"".join(_json_line(row) for row in data["records"])
    truth_blob = b"".join(_json_line(row) for row in data["truth"])
    records_path.write_bytes(records_blob)
    truth_path.write_bytes(truth_blob)

    source_hashes = {path.name: file_sha256(path) for path in source_paths}
    output_hashes = {
        "records.jsonl": sha256_bytes(records_blob),
        "truth.jsonl": sha256_bytes(truth_blob),
    }
    manifest = {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "synthetic_exception_verified": SYNTHETIC_EXCEPTION_VERIFIED,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "rule": "Use organizer-released data. No verified synthetic exception.",
        "seed": data["settings"]["seed"],
        "settings": data["settings"],
        "n_records": len(data["records"]),
        "n_truth": len(data["truth"]),
        "n_queries": len(data["queries"]),
        "payout_order": list(PAYOUTS),
        "probability_columns": list(PROBABILITY_COLUMNS),
        "payout_to_index": {str(k): v for k, v in PAYOUT_TO_INDEX.items()},
        "missing_payout_index": None,
        "missing_is_not_payout_zero": True,
        "slot_equals_machine_id": True,
        "signal_scope": "query_cutoff_only",
        "assumptions": list(ASSUMPTIONS),
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
    }
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


def default_source_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    names = ("generate.py", "test_generator.py", "README.md")
    return [here / name for name in names if (here / name).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate abrupt_changepoints synthetic logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    data = generate(
        seed=args.seed,
        n_episodes=args.episodes,
        n_ticks=args.ticks,
        n_machines=args.machines,
    )
    manifest = write_dataset(Path(args.out), data, default_source_paths())
    print(
        json.dumps(
            {
                "out": args.out,
                "n_records": manifest["n_records"],
                "eligibility": manifest["eligibility"],
                "output_hashes": manifest["output_hashes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
