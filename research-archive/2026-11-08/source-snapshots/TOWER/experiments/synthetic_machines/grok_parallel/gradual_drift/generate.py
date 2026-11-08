#!/usr/bin/env python3
"""Self-authored gradual_drift toy family. RESEARCH ONLY.

This is not organizer data and it is not a contest training source.
kit/RULES.md says to use organizer-released data. No synthetic-data
exception was verified, so consumers must keep the real-only fallback.

Invented assumptions (not fit to public, train, validation, or test logs):

- Payout slots, in order, are 0, 1, 2, 5, 20. Slot k is PAYOUTS[k].
- machine_id 0..7 has a fixed type. See MACHINE_TYPE_BY_ID.
- The spin at tick t is drawn from a latent categorical. Its logits are
  BASE_LOGITS[type] + phase_t * DRIFT_LOGITS[type].
- phase_t uses the seed, a smooth time ramp, this machine's true spins at
  ticks < t, and other machines' true spins at ticks < t. Spins at tick t
  are applied only after every machine's tick-t distribution is fixed.
- A latent EWMA starts at 0. That 0 is an additive shock of "no spins yet",
  not an imputed payout. Payout 0 is the different value centered_payout(0).
- Some ticks are gappy. The true spin still happens and still updates the
  latent phase. Observable-history probabilities omit those ticks. They do
  not impute payout 0. feature_mask 0 means "drop this row", not "pay 0".
- Observable-history probabilities are a Dirichlet-multinomial filter:
  type prior pseudo-counts plus this machine's observed payouts at ticks < t.
  They are not the sampling distribution. Hidden phase and unobserved spins
  are invisible to that filter, which is why the two vectors differ.
- Queries have one cutoff per machine (the last generated tick). signal is
  null. No historical signal series is invented. Real contest signals stay
  on organizer query rows at the real prediction cutoff.
- Another machine's tick-t payout is not an input to anyone's tick-t odds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FAMILY = "gradual_drift"
ELIGIBILITY = "RESEARCH_ONLY"
MAX_RECORDS = 8192

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {payout: index for index, payout in enumerate(PAYOUTS)}
INDEX_TO_PAYOUT = {index: payout for payout, index in PAYOUT_TO_INDEX.items()}

MACHINE_TYPE_BY_ID = {
    0: "steady",
    1: "steady",
    2: "balanced",
    3: "balanced",
    4: "balanced",
    5: "jackpot",
    6: "jackpot",
    7: "jackpot",
}
MACHINE_TYPES = ("steady", "balanced", "jackpot")

# Toy logits. Not estimated.
BASE_LOGITS = {
    "steady": (1.20, 0.70, 0.10, -0.80, -1.80),
    "balanced": (0.40, 0.40, 0.20, -0.20, -1.00),
    "jackpot": (0.70, 0.10, 0.00, -0.10, -0.20),
}
DRIFT_LOGITS = {
    "steady": (-0.30, -0.10, 0.05, 0.15, 0.20),
    "balanced": (-0.80, -0.20, 0.10, 0.30, 0.60),
    "jackpot": (-1.20, -0.30, 0.10, 0.40, 1.00),
}
SPEED = {"steady": 0.35, "balanced": 0.75, "jackpot": 1.15}

# Observable filter pseudo-counts. Not estimated from contest logs.
PRIOR_COUNTS = {
    "steady": (8.0, 4.0, 2.0, 1.0, 0.5),
    "balanced": (4.0, 3.0, 2.0, 1.5, 0.8),
    "jackpot": (3.0, 2.0, 1.5, 1.2, 1.0),
}

EWMA_ALPHA = 0.30
OWN_GAIN = 1.50
SHARED_GAIN = 0.50
GAP_STRIDE = 5
EXTRA_MISSING_RATE = 0.15

QUERY_COLUMNS = (
    "query_id",
    "episode_id",
    "machine_id",
    "machine_type",
    "signal",
    "cutoff",
)

MAPPINGS = {
    "payout_slot_index": {str(payout): index for payout, index in PAYOUT_TO_INDEX.items()},
    "slot_index_to_payout": {str(index): payout for index, payout in INDEX_TO_PAYOUT.items()},
    "probability_vector_slots": ["p_0", "p_1", "p_2", "p_5", "p_20"],
    "query_column_index": {name: index for index, name in enumerate(QUERY_COLUMNS)},
    "machine_type_by_id": {str(machine_id): name for machine_id, name in MACHINE_TYPE_BY_ID.items()},
    "feature_mask": {
        "1": "observed; payout may be used as a feature from earlier ticks",
        "0": "missing; omit the tick; do not impute payout 0 and do not multiply the payout by the mask",
    },
    "probability_kinds": {
        "probabilities": "latent_oracle",
        "observable_probabilities": "observable_history",
    },
}


def _require_int(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def unit_interval(*parts):
    """Stable unit sample in [0, 1) from a structured key. Independent of PYTHONHASHSEED."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def machine_type(machine_id):
    try:
        return MACHINE_TYPE_BY_ID[machine_id]
    except KeyError as exc:
        raise ValueError(f"machine_id must be in 0..7, got {machine_id}") from exc


def centered_payout(payout):
    """Map a real payout onto [-0.5, 0.5]. This is not the code for a missing tick."""
    return PAYOUT_TO_INDEX[payout] / 4.0 - 0.5


def softmax(logits):
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def is_observed(seed, episode_index, machine_id, tick):
    """Deterministic gap mask. Independent of the payout draw."""
    if (tick + 2 * machine_id + episode_index) % GAP_STRIDE == GAP_STRIDE - 1:
        return False
    return unit_interval(seed, episode_index, machine_id, tick, "obs") >= EXTRA_MISSING_RATE


def initial_phase(seed, episode_index, machine_id):
    return (unit_interval(seed, episode_index, machine_id, "phi0") - 0.5) * 0.5


def drift_direction(seed, episode_index, machine_id):
    if unit_interval(seed, episode_index, machine_id, "direction") < 0.5:
        return -1.0
    return 1.0


def time_fraction(tick, n_ticks):
    if n_ticks <= 1:
        return 0.0
    return tick / (n_ticks - 1)


def shared_shock(machine_id, ewma, seen):
    """Mean latent EWMA of other machines that already have a real spin.

    A machine with seen == 0 is omitted. The fallback 0 means "no other
    evidence yet", not "those machines paid 0".
    """
    values = [ewma[other] for other in range(len(ewma)) if other != machine_id and seen[other] > 0]
    if not values:
        return 0.0
    return math.fsum(values) / len(values)


def phase(seed, episode_index, machine_id, tick, n_ticks, ewma, seen):
    kind = machine_type(machine_id)
    value = initial_phase(seed, episode_index, machine_id)
    value += drift_direction(seed, episode_index, machine_id) * SPEED[kind] * time_fraction(tick, n_ticks)
    value += OWN_GAIN * ewma[machine_id]
    value += SHARED_GAIN * shared_shock(machine_id, ewma, seen)
    return value


def oracle_probabilities(kind, phi):
    base = BASE_LOGITS[kind]
    drift = DRIFT_LOGITS[kind]
    logits = tuple(base[slot] + phi * drift[slot] for slot in range(len(PAYOUTS)))
    return softmax(logits)


def observable_history_probabilities(kind, observed_payouts):
    """History filter. `observed_payouts` must already have dropped missing ticks."""
    counts = list(PRIOR_COUNTS[kind])
    for payout in observed_payouts:
        counts[PAYOUT_TO_INDEX[payout]] += 1.0
    total = math.fsum(counts)
    return [count / total for count in counts]


def sample_payout(seed, episode_index, machine_id, tick, probabilities):
    draw = unit_interval(seed, episode_index, machine_id, tick, "payout")
    cumulative = 0.0
    for slot, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return PAYOUTS[slot]
    return PAYOUTS[-1]


def history_before(records, episode_id, machine_id, cutoff):
    """Observed payouts with tick < cutoff.

    Unobserved rows are omitted. A row at the cutoff tick is the label, not a feature.
    """
    rows = [
        row
        for row in records
        if row["episode_id"] == episode_id
        and row["machine_id"] == machine_id
        and row["tick"] < cutoff
        and row["observed"]
    ]
    rows.sort(key=lambda row: row["tick"])
    return [row["payout"] for row in rows]


def _check_intervention(interventions, episode_id, machine_id, tick):
    if not interventions:
        return None
    key = (episode_id, machine_id, tick)
    if key not in interventions:
        return None
    payout = interventions[key]
    if payout not in PAYOUT_TO_INDEX:
        raise ValueError(f"intervention payout must be one of {PAYOUTS}, got {payout}")
    return payout


def generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8, interventions=None):
    """Return records, latent-oracle truth, and one query per machine at the last tick.

    `interventions` maps (episode_id, machine_id, tick) to a payout. The replacement
    is applied after that tick's probabilities are computed, so it can change later
    ticks only.
    """
    seed = _require_int(seed, "seed")
    n_episodes = _require_int(n_episodes, "n_episodes", minimum=1)
    n_ticks = _require_int(n_ticks, "n_ticks", minimum=1)
    n_machines = _require_int(n_machines, "n_machines", minimum=1, maximum=8)
    total = n_episodes * n_ticks * n_machines
    if total > MAX_RECORDS:
        raise ValueError(f"refusing to emit {total} records; cap is {MAX_RECORDS}")

    records = []
    truth = []
    queries = []
    for episode_index in range(n_episodes):
        episode_id = f"gd-e{episode_index:03d}"
        ewma = [0.0] * n_machines
        seen = [0] * n_machines
        observed_past = [[] for _ in range(n_machines)]
        for tick in range(n_ticks):
            ewma_before = list(ewma)
            seen_before = list(seen)
            planned = []
            for machine_id in range(n_machines):
                kind = machine_type(machine_id)
                phi = phase(seed, episode_index, machine_id, tick, n_ticks, ewma_before, seen_before)
                latent = oracle_probabilities(kind, phi)
                observable = observable_history_probabilities(kind, observed_past[machine_id])
                replacement = _check_intervention(interventions, episode_id, machine_id, tick)
                if replacement is None:
                    payout = sample_payout(seed, episode_index, machine_id, tick, latent)
                else:
                    payout = replacement
                observed = is_observed(seed, episode_index, machine_id, tick)
                planned.append((kind, phi, latent, observable, payout, observed))
            for machine_id, (kind, phi, latent, observable, payout, observed) in enumerate(planned):
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "machine_type": kind,
                        "payout": payout,
                        "payout_index": PAYOUT_TO_INDEX[payout],
                        "slot_index": PAYOUT_TO_INDEX[payout],
                        "observed": observed,
                        "feature_mask": 1 if observed else 0,
                    }
                )
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "machine_type": kind,
                        "probabilities": latent,
                        "probability_kind": "latent_oracle",
                        "observable_probabilities": observable,
                        "observable_probability_kind": "observable_history",
                        "latent_phase": phi,
                    }
                )
                center = centered_payout(payout)
                ewma[machine_id] = (1.0 - EWMA_ALPHA) * ewma[machine_id] + EWMA_ALPHA * center
                seen[machine_id] += 1
                if observed:
                    observed_past[machine_id].append(payout)
        cutoff = n_ticks - 1
        for machine_id in range(n_machines):
            kind = machine_type(machine_id)
            queries.append(
                {
                    "query_id": f"{episode_id}-m{machine_id}-c{cutoff}",
                    "episode_id": episode_id,
                    "machine_id": machine_id,
                    "machine_type": kind,
                    "signal": None,
                    "cutoff": cutoff,
                    "signal_scope": "prediction_cutoff_only",
                    "signal_source": "not_synthesized",
                }
            )

    return {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": True,
        "records": records,
        "truth": truth,
        "queries": queries,
        "mappings": MAPPINGS,
        "settings": {
            "seed": seed,
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
        },
    }


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_run(out_dir, seed=0, n_episodes=8, n_ticks=96, n_machines=8):
    result = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    truth_path = out / "truth.jsonl"
    queries_path = out / "queries.jsonl"
    _write_jsonl(records_path, result["records"])
    _write_jsonl(truth_path, result["truth"])
    _write_jsonl(queries_path, result["queries"])

    source_hashes = {"generate.py": _sha256(Path(__file__).resolve())}
    test_path = Path(__file__).resolve().parent / "test_generator.py"
    readme_path = Path(__file__).resolve().parent / "README.md"
    if test_path.is_file():
        source_hashes["test_generator.py"] = _sha256(test_path)
    if readme_path.is_file():
        source_hashes["README.md"] = _sha256(readme_path)

    manifest = {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": True,
        "organizer_rule": (
            "Use organizer-released data. No verified synthetic exception. "
            "Do not replace real train, validation, or test with this output."
        ),
        "seed": seed,
        "settings": result["settings"],
        "n_records": len(result["records"]),
        "max_records": MAX_RECORDS,
        "mappings": MAPPINGS,
        "missing_policy": "omit unobserved ticks; do not impute payout 0",
        "causal_policy": "tick t uses only spins at ticks < t; no contemporaneous cross-machine outcomes",
        "signal_policy": "signal is null and exists only on the query at cutoff; no historical signal is invented",
        "source_hashes": source_hashes,
        "output_hashes": {
            "records.jsonl": _sha256(records_path),
            "truth.jsonl": _sha256(truth_path),
            "queries.jsonl": _sha256(queries_path),
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the research-only gradual_drift toy family.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = write_run(
        args.out,
        seed=args.seed,
        n_episodes=args.episodes,
        n_ticks=args.ticks,
        n_machines=args.machines,
    )
    print(
        json.dumps(
            {
                "eligibility": manifest["eligibility"],
                "n_records": manifest["n_records"],
                "out": str(Path(args.out).resolve()),
                "output_hashes": manifest["output_hashes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
