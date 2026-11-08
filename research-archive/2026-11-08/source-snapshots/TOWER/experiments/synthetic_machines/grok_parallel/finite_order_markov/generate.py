#!/usr/bin/env python3
"""Finite-order Markov synthetic slot family (RESEARCH_ONLY).

Self-authored toy. Not organizer data and not fit to train, validation, or test.
Competition use stays on organizer-released real data until a synthetic exception
is verified (real-only fallback).

Each machine is an order-2 Markov chain on its own payouts. The draw at tick t
depends only on that machine's latent payouts at t-1 and t-2. Other machines are
independent. A same-tick outcome is never an input. Unobserved ticks stay in the
latent chain but are unknown to the observable filter: missing is not payout 0.

No ``signal`` series is invented. Organizer signals exist only on real query rows
at the prediction cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
INDEX_TO_PAYOUT = {0: 0, 1: 1, 2: 2, 3: 5, 4: 20}

# Slot index is machine_id. It is not a payout index.
# Payout index 0 is the category for payout 0. A missing tick has no payout index.
MACHINE_TYPES = ("steady", "balanced", "jackpot")
TYPE_BY_MACHINE = (
    "steady",
    "steady",
    "steady",
    "balanced",
    "balanced",
    "balanced",
    "jackpot",
    "jackpot",
)

ORDER = 2
MAX_RECORDS = 8192
ELIGIBILITY = "RESEARCH_ONLY"
SYNTHETIC_EXCEPTION_VERIFIED = False
REAL_ONLY_FALLBACK = True
FAMILY = "finite_order_markov"

# Invented integer weights. Not estimated from organizer files.
BASE_WEIGHTS = {
    "steady": (70, 22, 5, 2, 1),
    "balanced": (28, 26, 24, 16, 6),
    "jackpot": (72, 10, 6, 6, 6),
}
LAG1_PERSIST = {"steady": 48, "balanced": 18, "jackpot": 10}
LAG2_PERSIST = {"steady": 16, "balanced": 6, "jackpot": 4}
# Extra (p_5, p_20) weight when that lag equals 20.
LAG1_JACKPOT_BONUS = {"steady": (2, 2), "balanced": (6, 8), "jackpot": (10, 36)}
LAG2_JACKPOT_BONUS = {"steady": (1, 1), "balanced": (3, 4), "jackpot": (6, 18)}
LAG1_FIVE_BONUS = {"steady": 6, "balanced": 8, "jackpot": 14}
LAG2_FIVE_BONUS = {"steady": 2, "balanced": 3, "jackpot": 6}

# P(observed). Independent of the payout (MCAR). Gap != structural zero.
P_OBSERVE = {"steady": 0.92, "balanced": 0.84, "jackpot": 0.70}

_START = 0
_N_LAG = 6  # START + 5 payouts


def machine_type_of(machine_id: int) -> str:
    if machine_id < 0 or machine_id > 7:
        raise ValueError("machine_id must be in 0..7")
    return TYPE_BY_MACHINE[machine_id]


def feature_payout(record: dict):
    """History feature. Unobserved -> None. Never coerce a gap to payout 0."""
    if not record["observed"]:
        return None
    return record["payout"]


def normalize(weights) -> tuple[float, ...]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("probability weights must be positive")
    return tuple(float(w) / total for w in weights)


def _apply_lag(weights: list[int], lag, persist: int, jackpot_bonus, five_bonus: int) -> None:
    if lag is None:
        return
    weights[PAYOUT_TO_INDEX[lag]] += persist
    if lag == 20:
        weights[3] += jackpot_bonus[0]
        weights[4] += jackpot_bonus[1]
    elif lag == 5:
        weights[3] += five_bonus


def kernel_weights(machine_type: str, lag2, lag1) -> tuple[int, int, int, int, int]:
    """Unnormalized order-2 weights. None lag = tick does not exist yet, not a gap."""
    if machine_type not in BASE_WEIGHTS:
        raise ValueError(f"unknown machine_type {machine_type!r}")
    for lag in (lag2, lag1):
        if lag is not None and lag not in PAYOUT_TO_INDEX:
            raise ValueError(f"lag payout must be one of {PAYOUTS} or None, got {lag!r}")
    weights = list(BASE_WEIGHTS[machine_type])
    _apply_lag(weights, lag1, LAG1_PERSIST[machine_type], LAG1_JACKPOT_BONUS[machine_type], LAG1_FIVE_BONUS[machine_type])
    _apply_lag(weights, lag2, LAG2_PERSIST[machine_type], LAG2_JACKPOT_BONUS[machine_type], LAG2_FIVE_BONUS[machine_type])
    return tuple(weights)


def oracle_probabilities(machine_type: str, lag2, lag1) -> tuple[float, ...]:
    """Latent-oracle P(Y_t | type, Y_{t-2}, Y_{t-1}).

    Lags are true spins. A randomly missing tick still has a latent payout and
    must be passed here as that payout. None is only the structural start
    (t < 2). This is not the observable-history distribution.
    """
    return normalize(kernel_weights(machine_type, lag2, lag1))


def latent_oracle_from_payouts(machine_type: str, latent_past: list) -> tuple[float, ...]:
    """Oracle at the next tick given the full latent past (ticks strictly before t)."""
    lag1 = latent_past[-1] if len(latent_past) >= 1 else None
    lag2 = latent_past[-2] if len(latent_past) >= 2 else None
    return oracle_probabilities(machine_type, lag2, lag1)


def sample_index(probs, rng: random.Random) -> int:
    """Half-open cumulative buckets. The last bin catches the remainder."""
    u = rng.random()
    cumulative = 0.0
    last = len(probs) - 1
    for i, p in enumerate(probs):
        cumulative += p
        if u < cumulative or i == last:
            return i
    return last


def sample_payout(probs, rng: random.Random) -> int:
    return INDEX_TO_PAYOUT[sample_index(probs, rng)]


def _lag_index_to_value(index: int):
    if index == _START:
        return None
    return PAYOUTS[index - 1]


def _value_to_lag_index(payout: int) -> int:
    return 1 + PAYOUT_TO_INDEX[payout]


def initial_belief() -> list[float]:
    belief = [0.0] * (_N_LAG * _N_LAG)
    belief[0] = 1.0  # (START, START)
    return belief


def predictive_from_belief(machine_type: str, belief: list[float]) -> tuple[float, ...]:
    acc = [0.0] * 5
    for a in range(_N_LAG):
        for b in range(_N_LAG):
            mass = belief[a * _N_LAG + b]
            if mass == 0.0:
                continue
            probs = oracle_probabilities(machine_type, _lag_index_to_value(a), _lag_index_to_value(b))
            for k in range(5):
                acc[k] += mass * probs[k]
    total = sum(acc)
    if total <= 0.0:
        raise RuntimeError("belief predictive has no mass")
    return tuple(v / total for v in acc)


def update_belief(machine_type: str, belief: list[float], observed_payout) -> list[float]:
    """Advance one tick. observed_payout is an int if seen, else None (marginalize)."""
    joint = [0.0] * (_N_LAG * _N_LAG)
    for a in range(_N_LAG):
        for b in range(_N_LAG):
            mass = belief[a * _N_LAG + b]
            if mass == 0.0:
                continue
            probs = oracle_probabilities(machine_type, _lag_index_to_value(a), _lag_index_to_value(b))
            for k in range(5):
                y_idx = 1 + k
                joint[b * _N_LAG + y_idx] += mass * probs[k]
    if observed_payout is None:
        total = sum(joint)
        return [v / total for v in joint]
    y_idx = _value_to_lag_index(observed_payout)
    total = 0.0
    for b in range(_N_LAG):
        total += joint[b * _N_LAG + y_idx]
    if total <= 0.0:
        raise RuntimeError("observation impossible under the kernel")
    out = [0.0] * (_N_LAG * _N_LAG)
    for b in range(_N_LAG):
        out[b * _N_LAG + y_idx] = joint[b * _N_LAG + y_idx] / total
    return out


def observable_probabilities_from_mask(machine_type: str, masked_past: list) -> tuple[float, ...]:
    """P(Y_t | observed history strictly before t).

    masked_past items are payouts or None. None is unknown, not 0. This filter
    never reads a latent payout that was not observed.
    """
    belief = initial_belief()
    for item in masked_past:
        belief = update_belief(machine_type, belief, item)
    return predictive_from_belief(machine_type, belief)


def _episode_seed(seed: int, episode: int) -> int:
    """Deterministic per-episode stream id. Does not use Python's salted hash()."""
    digest = hashlib.sha256(f"finite_order_markov|{seed}|{episode}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _check_settings(seed, n_episodes, n_ticks, n_machines) -> None:
    values = {
        "seed": seed,
        "n_episodes": n_episodes,
        "n_ticks": n_ticks,
        "n_machines": n_machines,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if n_episodes < 1 or n_ticks < 1 or n_machines < 1:
        raise ValueError("n_episodes, n_ticks, and n_machines must be >= 1")
    if n_machines > 8:
        raise ValueError("n_machines must be <= 8 so machine_id stays in 0..7")
    if n_episodes * n_ticks * n_machines > MAX_RECORDS:
        raise ValueError(f"refusing to generate more than {MAX_RECORDS} records")


def generate(seed: int = 0, n_episodes: int = 8, n_ticks: int = 96, n_machines: int = 8) -> dict:
    """Return ``{"records": [...], "truth": [...]}``.

    Each episode has its own RNG stream (sha256 of seed and episode index).
    Within an episode the order is (tick, machine): payout draw, then observation draw.
    Probabilities for every machine at tick t are fixed before any tick-t update.
    Extending n_ticks keeps the prefix of every episode. Extending n_episodes
    keeps the earlier episodes.
    """
    _check_settings(seed, n_episodes, n_ticks, n_machines)
    records: list[dict] = []
    truth: list[dict] = []
    for episode in range(n_episodes):
        # Separate stream per episode so a longer episode does not shift later ones.
        rng = random.Random(_episode_seed(seed, episode))
        episode_id = f"fom-s{seed}-e{episode}"
        types = [machine_type_of(m) for m in range(n_machines)]
        latent = [[] for _ in range(n_machines)]
        beliefs = [initial_belief() for _ in range(n_machines)]
        for tick in range(n_ticks):
            drawn = []
            for machine in range(n_machines):
                past = latent[machine]
                ora = latent_oracle_from_payouts(types[machine], past)
                obp = predictive_from_belief(types[machine], beliefs[machine])
                payout = sample_payout(ora, rng)
                observed = rng.random() < P_OBSERVE[types[machine]]
                lag1 = past[-1] if len(past) >= 1 else None
                lag2 = past[-2] if len(past) >= 2 else None
                drawn.append((payout, observed, ora, obp, lag2, lag1))
            for machine, (payout, observed, ora, obp, lag2, lag1) in enumerate(drawn):
                machine_type = types[machine]
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
                        "machine_type": machine_type,
                        "probabilities": list(ora),
                        "observable_probabilities": list(obp),
                        "latent_lag2": lag2,
                        "latent_lag1": lag1,
                        "probability_kind": "latent_oracle",
                        "observable_probability_kind": "observable_history",
                    }
                )
                latent[machine].append(payout)
                beliefs[machine] = update_belief(
                    machine_type, beliefs[machine], payout if observed else None
                )
    return {"records": records, "truth": truth}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes(directory: Path) -> dict:
    hashes = {}
    for name in ("generate.py", "test_generator.py", "README.md"):
        path = directory / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def _assumptions() -> list[str]:
    return [
        "Toy order-2 Markov kernel with hand-written integer weights, not fit to organizer data.",
        "Machines are independent. Cross-machine outcomes are not inputs, including earlier ticks.",
        "State updates are synchronous: tick t uses only latent payouts at ticks < t on the same machine.",
        "Observation masks are MCAR given machine type and are not payout 0.",
        "probabilities are latent-oracle conditionals. observable_probabilities marginalize gaps.",
        "None / unobserved is unknown. Payout index 0 means category payout 0 only.",
        "No synthetic signal series. Real query signals exist only at the organizer prediction cutoff.",
        "RESEARCH_ONLY: synthetic_exception_verified is false; competition fallback is organizer-released real data.",
    ]


def build_manifest(out_dir: Path, seed: int, n_episodes: int, n_ticks: int, n_machines: int, record_count: int) -> dict:
    records_path = out_dir / "records.jsonl"
    truth_path = out_dir / "truth.jsonl"
    return {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "synthetic_exception_verified": SYNTHETIC_EXCEPTION_VERIFIED,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "seed": seed,
        "settings": {
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
            "order": ORDER,
            "max_records": MAX_RECORDS,
            "payouts": list(PAYOUTS),
            "payout_to_index": {str(k): v for k, v in PAYOUT_TO_INDEX.items()},
            "index_to_payout": {str(k): v for k, v in INDEX_TO_PAYOUT.items()},
            "machine_types_by_id": list(TYPE_BY_MACHINE[:n_machines]),
            "base_weights": {k: list(v) for k, v in BASE_WEIGHTS.items()},
            "lag1_persist": dict(LAG1_PERSIST),
            "lag2_persist": dict(LAG2_PERSIST),
            "lag1_jackpot_bonus_p5_p20": {k: list(v) for k, v in LAG1_JACKPOT_BONUS.items()},
            "lag2_jackpot_bonus_p5_p20": {k: list(v) for k, v in LAG2_JACKPOT_BONUS.items()},
            "lag1_five_bonus": dict(LAG1_FIVE_BONUS),
            "lag2_five_bonus": dict(LAG2_FIVE_BONUS),
            "p_observe": dict(P_OBSERVE),
            "missing_policy": "unobserved is unknown, not payout 0",
            "signal_policy": "no synthetic signal; real signal exists only at the organizer prediction cutoff",
            "rng_order": "per-episode sha256 stream, then tick, machine, payout draw, observation draw",
            "probability_kinds": {
                "probabilities": "latent_oracle",
                "observable_probabilities": "observable_history",
            },
        },
        "record_count": record_count,
        "source_hashes": _source_hashes(Path(__file__).resolve().parent),
        "output_hashes": {
            "records.jsonl": sha256_file(records_path),
            "truth.jsonl": sha256_file(truth_path),
        },
        "assumptions": _assumptions(),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_run(out_dir, seed: int = 0, n_episodes: int = 8, n_ticks: int = 96, n_machines: int = 8) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
    _write_jsonl(out / "records.jsonl", data["records"])
    _write_jsonl(out / "truth.jsonl", data["truth"])
    manifest = build_manifest(out, seed, n_episodes, n_ticks, n_machines, len(data["records"]))
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out / "manifest.json").write_text(text, encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate finite-order Markov synthetic slots (RESEARCH_ONLY).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = write_run(args.out, seed=args.seed, n_episodes=args.episodes, n_ticks=args.ticks, n_machines=args.machines)
    print(
        json.dumps(
            {
                "eligibility": manifest["eligibility"],
                "record_count": manifest["record_count"],
                "out": str(Path(args.out).resolve()),
                "output_hashes": manifest["output_hashes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
