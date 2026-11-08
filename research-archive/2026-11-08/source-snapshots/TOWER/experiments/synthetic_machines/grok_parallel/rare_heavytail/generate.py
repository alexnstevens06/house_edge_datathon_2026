"""rare_heavytail: a toy synthetic slot family with a rare, state-dependent heavy tail.

RESEARCH_ONLY. Contest rules require organizer-released data. No synthetic-data
exception is verified. training_source_policy() keeps the real-only fallback on.
Do not train a contest model on this output.

Invented process (not fit on train, validation, or test):

- Payouts are categorical on (0, 1, 2, 5, 20). Truth rows are the latent-oracle
  distribution, length 5 in that order.
- Each machine's heat is an integer in 0..3 folded from its own previous 8
  latent payouts only. A 20 adds 2, a 5 adds 1, a 0 subtracts 1, and 1 or 2
  hold. Heat stays inside 0..3. The fold starts at 0 on the oldest spin in the
  window.
- Tick t is scored for every machine before any machine's tick-t payout is
  drawn. Another machine's tick-t outcome is not an input.
- A latent payout is drawn on every tick, including ticks the log hides.
  Hidden ticks stay in the latent fold. An observable-history fold skips them.
  Skipping a hidden tick is different from writing payout 0 into the fold.
- The logging mask is invented: tick % 5 == 4 is always hidden, and other ticks
  are hidden with a type-specific coin. This is not the contest short-history
  design and was not estimated from contest logs.
- Synthetic queries set signal to null. Organizer signals belong only on real
  prediction-cutoff queries. This generator does not invent a signal series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

FAMILY = "rare_heavytail"
SCHEMA_VERSION = 1
RESEARCH_ONLY = True
ELIGIBILITY = "unresolved"
REAL_ONLY_FALLBACK = True
MAX_RECORDS = 8192
LAG = 8
HEAT_MAX = 3
PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
INDEX_TO_PAYOUT = {0: 0, 1: 1, 2: 2, 3: 5, 4: 20}
DENSE_SYMBOL_MISSING = 0
STRUCTURAL_GAP_PERIOD = 5
STRUCTURAL_GAP_PHASE = 4
WEIGHT_TOTAL = 1000

# Fixed slot map for machine_id 0..7. Prefix is used when n_machines < 8.
SLOT_MACHINE_TYPES = (
    "steady",
    "steady",
    "balanced",
    "balanced",
    "balanced",
    "jackpot",
    "jackpot",
    "jackpot",
)

# Per-thousand observation coins on ticks that are not structural gaps.
OBSERVE_PER_MILLE = {
    "steady": 900,
    "balanced": 750,
    "jackpot": 600,
}

# Heat row 0..3, weights for payouts (0, 1, 2, 5, 20). Each row sums to 1000.
# Cold jackpots rarely pay 20. Heat, fed by a 5 or a 20, thickens that tail.
TABLES: dict[str, tuple[tuple[int, ...], ...]] = {
    "steady": (
        (740, 180, 60, 15, 5),
        (700, 180, 70, 30, 20),
        (640, 180, 80, 50, 50),
        (550, 160, 90, 80, 120),
    ),
    "balanced": (
        (600, 220, 120, 40, 20),
        (550, 200, 120, 70, 60),
        (480, 180, 120, 100, 120),
        (400, 150, 120, 130, 200),
    ),
    "jackpot": (
        (900, 50, 30, 12, 8),
        (840, 50, 40, 30, 40),
        (720, 50, 50, 60, 120),
        (550, 50, 50, 100, 250),
    ),
}

ASSUMPTIONS = (
    "Toy probabilities are explicit integer weights over 1000. They were not fit on contest data.",
    "machine_id is the slot index. Types are steady, steady, balanced, balanced, balanced, jackpot, jackpot, jackpot.",
    "Latent-oracle probabilities condition on the machine's own latent payouts at ticks strictly before t, including hidden spins.",
    "Observable-history probabilities skip hidden spins. A hidden spin is unknown. It is not entered as payout 0.",
    "State updates synchronously: every machine's tick-t distribution is fixed before any tick-t payout is sampled.",
    "Dependence is own-machine and limited to the previous 8 latent payouts. Contemporaneous cross-machine outcomes are not inputs.",
    "The observation mask is an invented logging process. It is independent of the payout value.",
    "Synthetic query signal is null. This family does not invent historical signals or organizer cutoff signals.",
    "Eligibility is unresolved. Real-only contest training remains the fallback.",
)

MAPPINGS = {
    "payouts_ordered": list(PAYOUTS),
    "payout_to_index": {str(payout): index for payout, index in PAYOUT_TO_INDEX.items()},
    "index_to_payout": {str(index): payout for index, payout in INDEX_TO_PAYOUT.items()},
    "dense_symbol": {
        "missing_unknown": DENSE_SYMBOL_MISSING,
        "payout_0": 1,
        "payout_1": 2,
        "payout_2": 3,
        "payout_5": 4,
        "payout_20": 5,
    },
    "sparse_symbol": {
        "observed_payout_index": "0..4",
        "missing_unknown": "absent row; unknown has no sparse symbol",
        "pad_symbol_if_a_batcher_needs_one": 5,
    },
    "mask": {"observed": 1, "missing_unknown": 0},
    "slot_index": "machine_id is the slot index in 0..n_machines-1",
    "query_cutoff": "n_ticks; history ticks are 0..n_ticks-1 and are strictly earlier than cutoff",
    "query_signal": "null; real organizer signals stay on real prediction-cutoff queries",
    "probability_order": list(PAYOUTS),
    "truth_probability_kind": "latent_oracle",
}


def training_source_policy() -> dict:
    """Contest training stays on organizer-released real data while eligibility is open."""
    return {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "research_only": RESEARCH_ONLY,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "status": "RESEARCH_ONLY",
        "contest_training_data": "organizer_released_real_only",
        "synthetic_rows_allowed_in_contest_training": False,
        "reason": (
            "Rules say use organizer-released data. No verified synthetic exception. "
            "Keep the real-only fallback until organizers allow this family."
        ),
    }


def _validate_tables() -> None:
    for machine_type, rows in TABLES.items():
        if len(rows) != HEAT_MAX + 1:
            raise RuntimeError(f"{machine_type} needs {HEAT_MAX + 1} heat rows")
        for heat, weights in enumerate(rows):
            if len(weights) != len(PAYOUTS) or sum(weights) != WEIGHT_TOTAL:
                raise RuntimeError(f"{machine_type} heat {heat} weights {weights} are not a {WEIGHT_TOTAL}-simplex")
            if any(weight < 0 for weight in weights):
                raise RuntimeError(f"{machine_type} heat {heat} has a negative weight")


_validate_tables()


def machine_types(n_machines: int) -> tuple[str, ...]:
    if n_machines < 1 or n_machines > len(SLOT_MACHINE_TYPES):
        raise ValueError("n_machines must be in 1..8 so machine_id stays in 0..7")
    return SLOT_MACHINE_TYPES[:n_machines]


def is_structural_gap(tick: int) -> bool:
    """Invented logging gap. The latent spin still happens."""
    return tick % STRUCTURAL_GAP_PERIOD == STRUCTURAL_GAP_PHASE


def dense_symbol(payout: int, observed: bool) -> int:
    """Dense code: 0 is unknown. Observed payout 0 is code 1."""
    if not observed:
        return DENSE_SYMBOL_MISSING
    return PAYOUT_TO_INDEX[payout] + 1


def fold_heat(past_values: list[int | None]) -> int:
    """Fold the last LAG own-machine values, oldest first, starting at heat 0.

    None is an unknown log cell and is skipped. A caller who inserts 0 there is
    imputing a payout, which this fold does not do.
    """
    heat = 0
    for value in past_values[-LAG:]:
        if value is None:
            continue
        if value == 20:
            heat = min(HEAT_MAX, heat + 2)
        elif value == 5:
            heat = min(HEAT_MAX, heat + 1)
        elif value == 0:
            heat = max(0, heat - 1)
        elif value in (1, 2):
            continue
        else:
            raise ValueError(f"payout must be one of {PAYOUTS} or None for unknown; got {value!r}")
    return heat


def _distribution(machine_type: str, heat: int, kind: str) -> dict:
    weights = TABLES[machine_type][heat]
    return {
        "heat": heat,
        "weights": weights,
        "probabilities": [weight / WEIGHT_TOTAL for weight in weights],
        "probability_kind": kind,
        "payout_order": PAYOUTS,
    }


def latent_oracle_probabilities(machine_type: str, past_payouts: list[int]) -> dict:
    """P(payout at the next tick | own latent spins strictly before that tick)."""
    if machine_type not in TABLES:
        raise ValueError(f"unknown machine_type {machine_type}")
    if any(payout is None for payout in past_payouts):
        raise ValueError("latent oracle history contains an unknown; pass a latent payout for every earlier tick")
    return _distribution(machine_type, fold_heat(list(past_payouts)), "latent_oracle")


def observable_history_probabilities(machine_type: str, past_observed: list[int | None]) -> dict:
    """P under the same tables, using only logged spins. None entries are skipped."""
    if machine_type not in TABLES:
        raise ValueError(f"unknown machine_type {machine_type}")
    return _distribution(machine_type, fold_heat(list(past_observed)), "observable_history")


def oracle_at(machine_type: str, latent_series: list[int], tick: int) -> dict:
    """Latent-oracle distribution at `tick`. Only latent_series[:tick] is read."""
    if tick < 0 or tick > len(latent_series):
        raise ValueError("tick must lie in 0..len(latent_series)")
    return latent_oracle_probabilities(machine_type, latent_series[:tick])


def tick_oracle(machine_types_for_slots: tuple[str, ...] | list[str], latent_columns: list[list[int]], tick: int) -> list[dict]:
    """Per-slot latent-oracle distributions at one tick. Column contents at >= tick are ignored."""
    return [oracle_at(machine_types_for_slots[slot], latent_columns[slot], tick) for slot in range(len(latent_columns))]


def episode_random(seed: int, episode: int) -> random.Random:
    """Per-episode stream. It depends on seed and episode index, not on n_ticks."""
    return random.Random(f"rare_heavytail:{seed}:{episode}")


def sample_weight_index(rng: random.Random, weights: tuple[int, ...] | list[int]) -> int:
    """One rng.randrange(1000) draw against a 1000-weight table."""
    draw = rng.randrange(WEIGHT_TOTAL)
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    raise RuntimeError("weight table does not sum to 1000")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def check_shape(n_episodes: int, n_ticks: int, n_machines: int) -> int:
    _require_positive_int(n_episodes, "n_episodes")
    _require_positive_int(n_ticks, "n_ticks")
    _require_positive_int(n_machines, "n_machines")
    if isinstance(n_machines, int) and n_machines > len(SLOT_MACHINE_TYPES):
        raise ValueError("n_machines must be <= 8 so machine_id stays in 0..7")
    n_records = n_episodes * n_ticks * n_machines
    if n_records > MAX_RECORDS:
        raise ValueError(f"refusing to generate {n_records} records; cap is {MAX_RECORDS}")
    return n_records


def generate(seed: int = 0, n_episodes: int = 8, n_ticks: int = 96, n_machines: int = 8) -> dict:
    """Return records, latent-oracle truth, and cutoff queries.

    Each episode uses episode_random(seed, episode). Inside an episode, for
    each tick and machine_id, draw the payout with one randrange(1000). If the
    tick is not a structural gap, draw the observation coin with a second
    randrange(1000). Structural gaps consume no observation draw. A shorter
    n_ticks is a tick-prefix of a longer run with the same seed, episode count,
    and machine count, because later episodes do not reuse earlier tick draws.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    n_records = check_shape(n_episodes, n_ticks, n_machines)
    types = machine_types(n_machines)
    records: list[dict] = []
    truth: list[dict] = []
    queries: list[dict] = []
    latent_by_episode: list[list[list[int]]] = []
    visible_by_episode: list[list[list[int | None]]] = []

    for episode in range(n_episodes):
        rng = episode_random(seed, episode)
        episode_id = f"rh{episode:03d}"
        latent_columns: list[list[int]] = [[] for _ in range(n_machines)]
        visible_columns: list[list[int | None]] = [[] for _ in range(n_machines)]
        for tick in range(n_ticks):
            specs = tick_oracle(types, latent_columns, tick)
            for machine_id, spec in enumerate(specs):
                payout_index = sample_weight_index(rng, spec["weights"])
                payout = INDEX_TO_PAYOUT[payout_index]
                if is_structural_gap(tick):
                    observed = False
                else:
                    observed = rng.randrange(WEIGHT_TOTAL) < OBSERVE_PER_MILLE[types[machine_id]]
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "slot_index": machine_id,
                        "machine_type": types[machine_id],
                        "payout": payout,
                        "payout_index": payout_index,
                        "observed": observed,
                        "mask": 1 if observed else 0,
                        "dense_symbol": dense_symbol(payout, observed),
                    }
                )
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "slot_index": machine_id,
                        "probabilities": list(spec["probabilities"]),
                        "weights": list(spec["weights"]),
                        "heat": spec["heat"],
                        "probability_kind": "latent_oracle",
                        "payout_order": list(PAYOUTS),
                    }
                )
                latent_columns[machine_id].append(payout)
                visible_columns[machine_id].append(payout if observed else None)
        latent_by_episode.append(latent_columns)
        visible_by_episode.append(visible_columns)
        for machine_id in range(n_machines):
            oracle = latent_oracle_probabilities(types[machine_id], latent_columns[machine_id])
            visible = observable_history_probabilities(types[machine_id], visible_columns[machine_id])
            queries.append(
                {
                    "query_id": f"{episode_id}-m{machine_id}-c{n_ticks}",
                    "episode_id": episode_id,
                    "machine_id": machine_id,
                    "slot_index": machine_id,
                    "machine_type": types[machine_id],
                    "cutoff": n_ticks,
                    "signal": None,
                    "signal_policy": "not_invented_real_cutoff_only",
                    "latent_oracle_probabilities": list(oracle["probabilities"]),
                    "latent_oracle_heat": oracle["heat"],
                    "observable_history_probabilities": list(visible["probabilities"]),
                    "observable_history_heat": visible["heat"],
                    "probability_kind_for_scoring_if_this_were_the_simulator": "latent_oracle",
                    "payout_order": list(PAYOUTS),
                }
            )

    return {
        "family": FAMILY,
        "schema_version": SCHEMA_VERSION,
        "research_only": RESEARCH_ONLY,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "status": "RESEARCH_ONLY",
        "training_source_policy": training_source_policy(),
        "settings": {
            "seed": seed,
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
            "n_records": n_records,
            "max_records": MAX_RECORDS,
            "cutoff": n_ticks,
            "lag": LAG,
        },
        "mappings": MAPPINGS,
        "assumptions": list(ASSUMPTIONS),
        "records": records,
        "truth": truth,
        "queries": queries,
        "latent_columns": latent_by_episode,
        "visible_columns": visible_by_episode,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path | None = None) -> dict[str, str | None]:
    root = root or Path(__file__).resolve().parent
    hashes: dict[str, str | None] = {}
    for name in ("generate.py", "test_generator.py", "README.md"):
        path = root / name
        hashes[name] = sha256_file(path) if path.is_file() else None
    return hashes


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_dataset(out_dir: str | Path, seed: int = 0, n_episodes: int = 8, n_ticks: int = 96, n_machines: int = 8) -> dict:
    """Write records.jsonl, truth.jsonl, queries.jsonl, and manifest.json."""
    bundle = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records_path = out_path / "records.jsonl"
    truth_path = out_path / "truth.jsonl"
    queries_path = out_path / "queries.jsonl"
    _write_jsonl(records_path, bundle["records"])
    _write_jsonl(truth_path, bundle["truth"])
    _write_jsonl(queries_path, bundle["queries"])
    output_hashes = {
        "records.jsonl": sha256_file(records_path),
        "truth.jsonl": sha256_file(truth_path),
        "queries.jsonl": sha256_file(queries_path),
    }
    manifest = {
        "family": FAMILY,
        "schema_version": SCHEMA_VERSION,
        "status": "RESEARCH_ONLY",
        "research_only": RESEARCH_ONLY,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "use_for_contest_training": False,
        "training_source_policy": training_source_policy(),
        "seed": seed,
        "settings": bundle["settings"],
        "probability_kind": "latent_oracle",
        "assumptions": list(ASSUMPTIONS),
        "mappings": MAPPINGS,
        "source_hashes": source_hashes(),
        "output_hashes": output_hashes,
        "n_records": len(bundle["records"]),
        "n_truth": len(bundle["truth"]),
        "n_queries": len(bundle["queries"]),
    }
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate rare_heavytail synthetic rows (RESEARCH_ONLY).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = write_dataset(
        args.out,
        seed=args.seed,
        n_episodes=args.episodes,
        n_ticks=args.ticks,
        n_machines=args.machines,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "n_records": manifest["n_records"],
                "out": str(Path(args.out).resolve()),
                "output_hashes": manifest["output_hashes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
