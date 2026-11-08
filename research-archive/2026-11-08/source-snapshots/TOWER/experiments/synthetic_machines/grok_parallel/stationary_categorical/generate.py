#!/usr/bin/env python3
"""Stationary categorical synthetic machines. RESEARCH_ONLY.

Each machine draws i.i.d. payouts from a fixed categorical distribution chosen
by its type. The distributions are invented toy weights, not estimates from
contest logs. Competition training must keep the real-only fallback: organizer-
released data only. Rules do not give a verified synthetic-data exception.

Oracle truth at tick t is the type distribution. It does not depend on spins.
An observable-history distribution is a separate Dirichlet-multinomial posterior
fit only to that machine's observed payouts at ticks < t. Missing ticks are
unknown and are not zeros. Query signals are not synthesized; a real signal
exists only on an organizer query at the prediction cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

FAMILY = "stationary_categorical"
RESEARCH_ONLY = True
ELIGIBILITY_STATUS = "unresolved"
COMPETITION_DATA = "organizer_released_only"
REAL_ONLY_FALLBACK = True

# Index order matches kit PAYOUTS and src/house_edge/data.py PAYOUT_TO_IDX.
PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
PROBABILITY_COLUMNS = ("p_0", "p_1", "p_2", "p_5", "p_20")
# Dense grid convention from src/house_edge/data.py: missing is -1, never payout 0.
MISSING = -1
MISSING_SYMBOL_INDEX = 5  # 6-way token: payout indices 0..4, then MISSING

# Index order matches src/house_edge/data.py TYPE_TO_IDX. Not a fitted encoding.
MACHINE_TYPES = ("balanced", "jackpot", "steady")
TYPE_TO_INDEX = {name: index for index, name in enumerate(MACHINE_TYPES)}

# Invented unnormalized weights for payouts (0, 1, 2, 5, 20). Not fit to data.
# steady: mostly losses. balanced: spread across small prizes. jackpot: fatter 20.
TYPE_WEIGHTS = {
    "steady": (72, 18, 6, 3, 1),
    "balanced": (40, 25, 18, 12, 5),
    "jackpot": (50, 15, 10, 10, 15),
}

# Uniform Dirichlet prior for the OBSERVABLE history posterior only.
# Deliberately not the oracle weights, so finite history stays distinct from truth.
HISTORY_ALPHA = (1.0, 1.0, 1.0, 1.0, 1.0)

# Invented MCAR observation rate. Missingness does not depend on the payout.
OBSERVE_PROB = 0.7

MAX_RECORDS = 8192
DEFAULT_EPISODES = 8
DEFAULT_TICKS = 96
DEFAULT_MACHINES = 8
SUM_TOLERANCE = 1e-9

SOURCE_FILES = ("generate.py", "test_generator.py", "README.md")


def eligibility() -> dict:
    return {
        "status": ELIGIBILITY_STATUS,
        "research_only": RESEARCH_ONLY,
        "label": "RESEARCH_ONLY",
        "reason": (
            "Rules say use organizer-released data. No verified synthetic exception. "
            "This family is research-only until eligibility is resolved."
        ),
        "competition_fallback": COMPETITION_DATA,
        "real_only_fallback": REAL_ONLY_FALLBACK,
    }


def _simplex(weights: tuple[int, ...]) -> tuple[float, ...]:
    if len(weights) != len(PAYOUTS):
        raise ValueError("weights must cover the five payouts in order 0,1,2,5,20")
    if any((not isinstance(w, int)) or isinstance(w, bool) or w <= 0 for w in weights):
        raise ValueError("weights must be positive integers")
    total = float(sum(weights))
    head = [w / total for w in weights[:-1]]
    last = 1.0 - math.fsum(head)
    if last <= 0.0:
        raise ValueError("simplex repair produced a non-positive last component")
    probs = tuple(head + [last])
    if abs(math.fsum(probs) - 1.0) > SUM_TOLERANCE:
        raise ValueError("probability vector is not a simplex")
    return probs


ORACLE = {name: _simplex(TYPE_WEIGHTS[name]) for name in MACHINE_TYPES}


def oracle_probabilities(machine_type: str) -> tuple[float, ...]:
    """Latent categorical probabilities. Independent of every spin and of the signal."""
    try:
        return ORACLE[machine_type]
    except KeyError as exc:
        raise ValueError(f"unknown machine_type {machine_type!r}") from exc


def sample_index(rng: random.Random, probabilities: tuple[float, ...]) -> int:
    """Inverse-CDF sample. probabilities are ordered as PAYOUTS."""
    if len(probabilities) != len(PAYOUTS):
        raise ValueError("expected 5 probabilities")
    u = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probabilities):
        cumulative += prob
        if u <= cumulative:
            return index
    return len(PAYOUTS) - 1


def _rng(*parts: object) -> random.Random:
    material = "|".join(str(part) for part in parts)
    return random.Random(material)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer")
    return value


def episode_machine_types(seed: int, episode_id: int, n_machines: int) -> list[str]:
    """Assign types before any spin. Depends on seed and episode id only."""
    rng = _rng(seed, episode_id, "types")
    cycle: list[str] = []
    while len(cycle) < n_machines:
        cycle.extend(MACHINE_TYPES)
    cycle = cycle[:n_machines]
    rng.shuffle(cycle)
    return cycle


def realize_cell(
    seed: int,
    episode_id: int,
    tick: int,
    machine_id: int,
    machine_type: str,
    observe_prob: float = OBSERVE_PROB,
) -> dict:
    """Draw one machine at one tick from its type distribution.

    The RNG stream is keyed only by (seed, episode, tick, machine). It does not
    read other machines, later ticks, or this tick's other outcomes. The
    observation bit is drawn after the payout and does not depend on its value
    except through the shared stream position, which is fixed for this cell.
    """
    if not 0.0 <= observe_prob <= 1.0:
        raise ValueError("observe_prob must be in [0, 1]")
    probabilities = oracle_probabilities(machine_type)
    rng = _rng(seed, episode_id, tick, machine_id, "cell")
    payout_index = sample_index(rng, probabilities)
    observed = rng.random() < observe_prob
    payout = PAYOUTS[payout_index] if observed else None
    stored_index = payout_index if observed else None
    return {
        "episode_id": episode_id,
        "tick": tick,
        "machine_id": machine_id,
        "slot_index": machine_id,
        "machine_type": machine_type,
        "machine_type_index": TYPE_TO_INDEX[machine_type],
        "payout": payout,
        "payout_index": stored_index,
        "observed": observed,
        "observation_mask": 1 if observed else 0,
        "dense_symbol": payout_index if observed else MISSING,
    }


def observable_history_probabilities(
    records: list[dict],
    episode_id: int,
    tick: int,
    machine_id: int,
    signal: int | None = None,
) -> tuple[float, ...]:
    """Posterior mean from observed own-machine spins at ticks strictly < tick.

    `signal` is accepted and ignored. This family does not invent historical
    signals, and a cutoff query signal is not a feature of the payout law.
    Unobserved rows are skipped even if a caller stored a numeric payout.
    Other machines and ticks >= `tick` are ignored.
    """
    del signal  # cutoff query signal is not an input to this family's law
    counts = [0, 0, 0, 0, 0]
    for record in records:
        if record["episode_id"] != episode_id or record["machine_id"] != machine_id:
            continue
        if record["tick"] >= tick:
            continue
        if not record["observed"]:
            continue
        payout = record["payout"]
        if payout not in PAYOUT_TO_INDEX:
            raise ValueError("observed record has a payout outside {0, 1, 2, 5, 20}")
        counts[PAYOUT_TO_INDEX[payout]] += 1
    total = float(sum(counts) + sum(HISTORY_ALPHA))
    return tuple((counts[i] + HISTORY_ALPHA[i]) / total for i in range(len(PAYOUTS)))


def generate(
    seed: int = 0,
    n_episodes: int = DEFAULT_EPISODES,
    n_ticks: int = DEFAULT_TICKS,
    n_machines: int = DEFAULT_MACHINES,
    observe_prob: float = OBSERVE_PROB,
) -> dict:
    """Return {'records', 'truth'} for a synchronous stationary categorical panel.

    Tick t is drawn for every machine from state that was fixed before the tick
    (the type assignment). No machine reads another machine's tick-t payout.
    After the tick is committed, state is unchanged: the family is stationary.
    """
    seed = _seed(seed)
    n_episodes = _positive_int(n_episodes, "n_episodes")
    n_ticks = _positive_int(n_ticks, "n_ticks")
    n_machines = _positive_int(n_machines, "n_machines")
    if n_machines > DEFAULT_MACHINES:
        raise ValueError("n_machines must be <= 8 so machine_id stays in 0..7")
    n_records = n_episodes * n_ticks * n_machines
    if n_records > MAX_RECORDS:
        raise ValueError(f"refusing to generate {n_records} records; cap is {MAX_RECORDS}")

    records: list[dict] = []
    truth: list[dict] = []
    for episode_id in range(n_episodes):
        # Types are episode state known before tick 0. Outcomes never update them.
        types = episode_machine_types(seed, episode_id, n_machines)
        for tick in range(n_ticks):
            committed = []
            for machine_id in range(n_machines):
                machine_type = types[machine_id]
                # State at t is `machine_type`, a function of ticks < t only
                # (it was set before any tick). The draw does not read `committed`.
                cell = realize_cell(
                    seed, episode_id, tick, machine_id, machine_type, observe_prob
                )
                probabilities = list(oracle_probabilities(machine_type))
                committed.append(cell)
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "slot_index": machine_id,
                        "machine_type": machine_type,
                        "probabilities": probabilities,
                    }
                )
            # Commit the whole tick, then advance. Stationary law: no state write.
            records.extend(committed)
    return {"records": records, "truth": truth}


def prediction_cutoff(n_ticks: int) -> int:
    """Real-style query cutoff: the next spin. History ticks are strictly below it.

    This generator does not attach a signal. Organizer queries carry signal
    in {-1, 0, 1} only at this cutoff, never as a historical series.
    """
    return _positive_int(n_ticks, "n_ticks")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_hashes(package_dir: Path | None = None) -> dict[str, str]:
    root = package_dir or Path(__file__).resolve().parent
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def settings_dict(seed: int, n_episodes: int, n_ticks: int, n_machines: int, observe_prob: float) -> dict:
    return {
        "family": FAMILY,
        "seed": seed,
        "n_episodes": n_episodes,
        "n_ticks": n_ticks,
        "n_machines": n_machines,
        "observe_prob": observe_prob,
        "prediction_cutoff": prediction_cutoff(n_ticks),
        "history_tick_range": [0, n_ticks - 1],
        "max_records": MAX_RECORDS,
        "payouts_ordered": list(PAYOUTS),
        "payout_to_index": {str(payout): index for payout, index in PAYOUT_TO_INDEX.items()},
        "probability_columns": list(PROBABILITY_COLUMNS),
        "machine_types_ordered": list(MACHINE_TYPES),
        "machine_type_to_index": dict(TYPE_TO_INDEX),
        "type_weights": {name: list(TYPE_WEIGHTS[name]) for name in MACHINE_TYPES},
        "oracle_probabilities": {name: list(ORACLE[name]) for name in MACHINE_TYPES},
        "history_alpha": list(HISTORY_ALPHA),
        "missing_dense_symbol": MISSING,
        "missing_symbol_index": MISSING_SYMBOL_INDEX,
        "observation_mask": {"1": "observed", "0": "missing_unknown"},
        "slot_index": "machine_id, integer in 0..n_machines-1, default 0..7",
        "signal_policy": (
            "No historical signal is generated. A real query signal is an "
            "organizer field in {-1, 0, 1} at the prediction cutoff only. "
            "Oracle and observable-history probabilities ignore signal."
        ),
        "assumptions": [
            "Payouts are i.i.d. given machine type, for every tick.",
            "Type is assigned from the seed before tick 0 and never updates.",
            "Machines are independent. Cross-machine outcomes are not inputs.",
            "A tick is drawn from state at ticks < t, then committed for every machine.",
            "Observation is MCAR with a constant probability, independent of payout value.",
            "Unobserved payout is null. Missing is unknown, not payout 0.",
            "Oracle truth is the type categorical, not a frequency estimate.",
            "Observable-history probabilities are a uniform-Dirichlet posterior on own observed ticks < t.",
            "Weights are invented toys, not fit on public or private contest logs.",
            "Not eligible for competition training while synthetic use is unresolved.",
        ],
    }


def build_manifest(
    seed: int,
    n_episodes: int,
    n_ticks: int,
    n_machines: int,
    observe_prob: float,
    output_hashes: dict[str, str],
    n_records: int,
    n_truth: int,
) -> dict:
    return {
        "family": FAMILY,
        "label": "RESEARCH_ONLY",
        "research_only": True,
        "eligibility": eligibility(),
        "seed": seed,
        "settings": settings_dict(seed, n_episodes, n_ticks, n_machines, observe_prob),
        "n_records": n_records,
        "n_truth": n_truth,
        "hashes": {
            "algorithm": "sha256",
            "source": source_hashes(),
            "outputs": output_hashes,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    lines = [json.dumps(row, sort_keys=True, allow_nan=False) for row in rows]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


def write_outputs(
    out_dir: Path,
    seed: int,
    n_episodes: int,
    n_ticks: int,
    n_machines: int = DEFAULT_MACHINES,
    observe_prob: float = OBSERVE_PROB,
) -> dict:
    panel = generate(seed, n_episodes, n_ticks, n_machines, observe_prob)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_bytes = _write_jsonl(out_dir / "records.jsonl", panel["records"])
    truth_bytes = _write_jsonl(out_dir / "truth.jsonl", panel["truth"])
    output_hashes = {
        "records.jsonl": sha256_bytes(records_bytes),
        "truth.jsonl": sha256_bytes(truth_bytes),
    }
    manifest = build_manifest(
        seed,
        n_episodes,
        n_ticks,
        n_machines,
        observe_prob,
        output_hashes,
        n_records=len(panel["records"]),
        n_truth=len(panel["truth"]),
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_hashes["manifest.json"] = sha256_file(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stationary_categorical generator (RESEARCH_ONLY)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = write_outputs(args.out, args.seed, args.episodes, args.ticks)
    print(
        f"RESEARCH_ONLY family={FAMILY} seed={args.seed} "
        f"records={manifest['n_records']} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
