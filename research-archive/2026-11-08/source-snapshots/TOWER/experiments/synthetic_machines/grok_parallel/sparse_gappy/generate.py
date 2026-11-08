#!/usr/bin/env python3
"""sparse_gappy: self-authored toy slot process with sparse gap masks.

Research-only. Not organizer data, and not fit to any released split.
Eligibility of synthetic episodes is unresolved: organizer instructions say
to use organizer-released data, and no synthetic exception was verified.
Callers must keep the real-only fallback (do not train on this output
unless a human explicitly opts into research-only synthetic data).

Generative assumptions (all invented, all explicit):

* Each machine is an order-1 Markov chain on its own latent payout.
  The chain does not read other machines, and the distribution at tick t
  is fixed before any payout at tick t is drawn (synchronous, from ticks < t).
* Machine type is a deterministic function of the slot. It is not sampled
  and not estimated from logs.
* A second, exogenous two-state mask decides whether the latent payout is
  revealed. The mask depends only on that machine's previous mask, never
  on payouts, so a gap is missing completely at random given the mask state.
* Missing is unknown. The observer's posterior does not become a point mass
  on payout 0, and it does not get to see the hidden payout.
* The observable signal column on real queries is not synthesized. Query
  rows exist only at the prediction cutoff (tick == n_ticks) and carry
  signal = null. Nothing backfills a signal onto history ticks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

FAMILY = "sparse_gappy"
ELIGIBILITY = "RESEARCH_ONLY"
REAL_ONLY_FALLBACK = True
MAX_RECORDS = 8192

# Column j is payout PAYOUTS[j]. Do not reorder.
PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_IDX = {payout: index for index, payout in enumerate(PAYOUTS)}
PROBABILITY_COLUMNS = ("p_0", "p_1", "p_2", "p_5", "p_20")
# Explicit mask symbol used by observation_symbol(). Not a payout index.
# Payout 0 maps to index 0 only when the tick was observed.
MISSING_INDEX = -1

# slot % 8 -> type. Slots 0..7 are the default eight machines.
# machine_id is the slot index in this toy (integer, not an organizer id).
_SLOT_TYPES = (
    "steady",
    "steady",
    "balanced",
    "balanced",
    "balanced",
    "balanced",
    "jackpot",
    "jackpot",
)

# P(Y_0) by type. Used at tick 0 by both the oracle and the observer,
# because there is no previous payout to disagree about.
PRIOR = {
    "steady": (0.50, 0.22, 0.14, 0.10, 0.04),
    "balanced": (0.28, 0.24, 0.20, 0.18, 0.10),
    "jackpot": (0.30, 0.18, 0.16, 0.20, 0.16),
}

# TRANSITION[type][prev_idx][next_idx]. Rows are invented, strictly positive,
# and separated so a different latent previous payout moves the oracle by
# more than 0.05 in L-infinity. Not estimated from any split.
TRANSITION = {
    "steady": (
        (0.82, 0.10, 0.04, 0.03, 0.01),
        (0.55, 0.28, 0.10, 0.05, 0.02),
        (0.48, 0.16, 0.24, 0.08, 0.04),
        (0.40, 0.14, 0.12, 0.26, 0.08),
        (0.30, 0.10, 0.10, 0.18, 0.32),
    ),
    "balanced": (
        (0.50, 0.22, 0.14, 0.10, 0.04),
        (0.28, 0.40, 0.18, 0.10, 0.04),
        (0.22, 0.18, 0.36, 0.16, 0.08),
        (0.18, 0.14, 0.16, 0.38, 0.14),
        (0.12, 0.10, 0.14, 0.24, 0.40),
    ),
    "jackpot": (
        (0.55, 0.18, 0.12, 0.10, 0.05),
        (0.40, 0.28, 0.14, 0.12, 0.06),
        (0.32, 0.16, 0.26, 0.16, 0.10),
        (0.22, 0.12, 0.14, 0.32, 0.20),
        (0.12, 0.08, 0.10, 0.25, 0.45),
    ),
}

# P(observed_t | previous tick observed?). Index 0 = previous was a gap
# (also the initial condition), index 1 = previous tick was observed.
# Long gaps, short looks. Independent of every payout.
OBSERVE_GIVEN_PREV = {
    "steady": (0.16, 0.45),
    "balanced": (0.12, 0.38),
    "jackpot": (0.08, 0.30),
}


def machine_type_for_slot(slot: int) -> str:
    if slot < 0:
        raise ValueError(f"slot must be >= 0, got {slot}")
    return _SLOT_TYPES[slot % 8]


def oracle_probabilities(machine_type: str, previous_payout: int | None) -> list[float]:
    """Latent-oracle P(Y_t | latent Y_{t-1}, type).

    previous_payout is the realized latent payout even if that tick was
    unobserved. None means tick 0 (no history).
    """
    if previous_payout is None:
        return list(PRIOR[machine_type])
    return list(TRANSITION[machine_type][PAYOUT_TO_IDX[previous_payout]])


def observer_predictive(machine_type: str, posterior: list[float] | None) -> list[float]:
    """Observable-history P(Y_t | revealed ticks < t, masks < t, type).

    posterior is a distribution over the latent payout at t-1, or None at
    tick 0. It must be built only from revealed payouts. A gap keeps the
    mixture; it does not insert a 0.
    """
    if posterior is None:
        return list(PRIOR[machine_type])
    matrix = TRANSITION[machine_type]
    return [
        sum(posterior[prev] * matrix[prev][nxt] for prev in range(len(PAYOUTS)))
        for nxt in range(len(PAYOUTS))
    ]


def posterior_after(predictive: list[float], payout: int, observed: bool) -> list[float]:
    """Belief over the latent payout that was just drawn.

    Unobserved -> the predictive mixture (missing is uninformative under the
    exogenous mask). Observed -> a point mass on the revealed payout.
    """
    if not observed:
        return list(predictive)
    point = [0.0] * len(PAYOUTS)
    point[PAYOUT_TO_IDX[payout]] = 1.0
    return point


def observable_payout(record: dict) -> int | None:
    """Payout a causal observer is allowed to use. None if the tick is a gap.

    None is not 0. The record's payout field still holds the latent draw so
    the oracle can be audited; do not feed that field to a model when
    observed is false.
    """
    if not record["observed"]:
        return None
    return int(record["payout"])


def observation_symbol(record: dict) -> int:
    """Mask map: MISSING_INDEX (-1) if unobserved, else payout index 0..4."""
    revealed = observable_payout(record)
    if revealed is None:
        return MISSING_INDEX
    return PAYOUT_TO_IDX[revealed]


def _check_settings(seed: int, n_episodes: int, n_ticks: int, n_machines: int) -> int:
    for name, value in (
        ("seed", seed),
        ("n_episodes", n_episodes),
        ("n_ticks", n_ticks),
        ("n_machines", n_machines),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if n_episodes < 1 or n_ticks < 1 or n_machines < 1:
        raise ValueError("n_episodes, n_ticks, and n_machines must be >= 1")
    n_records = n_episodes * n_ticks * n_machines
    if n_records > MAX_RECORDS:
        raise ValueError(
            f"{n_records} records exceeds MAX_RECORDS={MAX_RECORDS}"
        )
    return n_records


def _sample_index(probs: list[float], rng: random.Random) -> int:
    draw = rng.random()
    acc = 0.0
    last = len(probs) - 1
    for index, prob in enumerate(probs):
        acc += prob
        if draw < acc or index == last:
            return index
    return last


def generate(
    seed: int = 0,
    n_episodes: int = 8,
    n_ticks: int = 96,
    n_machines: int = 8,
    *,
    rewrite=None,
) -> dict:
    """Return records, latent-oracle truth, observer probabilities, and queries.

    rewrite(episode_id, tick, machine_id, payout, observed) -> (payout, observed)
    is an optional test hook. It runs after the draw and before the state
    commit, so it can change future transitions without rewriting the past.
    The default generator does not use it.
    """
    _check_settings(seed, n_episodes, n_ticks, n_machines)
    rng = random.Random(seed)
    records: list[dict] = []
    truth: list[dict] = []
    observable_history: list[dict] = []
    queries: list[dict] = []

    for episode_id in range(n_episodes):
        types = [machine_type_for_slot(slot) for slot in range(n_machines)]
        # Latent payout and observer belief from strictly earlier ticks.
        prev_payout: list[int | None] = [None] * n_machines
        posterior: list[list[float] | None] = [None] * n_machines
        prev_observed = [False] * n_machines  # initial condition: a gap

        for tick in range(n_ticks):
            drawn = []
            # Snapshot is the list itself: each machine reads only its own
            # previous state, never another machine's draw at this tick.
            for machine_id in range(n_machines):
                machine_type = types[machine_id]
                oracle_p = oracle_probabilities(machine_type, prev_payout[machine_id])
                history_p = observer_predictive(machine_type, posterior[machine_id])
                payout = PAYOUTS[_sample_index(oracle_p, rng)]
                see_p = OBSERVE_GIVEN_PREV[machine_type][1 if prev_observed[machine_id] else 0]
                observed = rng.random() < see_p
                if rewrite is not None:
                    payout, observed = rewrite(
                        episode_id, tick, machine_id, payout, observed
                    )
                    payout = int(payout)
                    observed = bool(observed)
                    if payout not in PAYOUT_TO_IDX:
                        raise ValueError(f"rewrite returned payout {payout}")
                drawn.append((machine_id, machine_type, oracle_p, history_p, payout, observed))

            for machine_id, machine_type, oracle_p, history_p, payout, observed in drawn:
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "machine_type": machine_type,
                        "payout": payout,
                        "observed": observed,
                    }
                )
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "probabilities": oracle_p,
                    }
                )
                observable_history.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "probabilities": history_p,
                    }
                )
                prev_payout[machine_id] = payout
                posterior[machine_id] = posterior_after(history_p, payout, observed)
                prev_observed[machine_id] = observed

        # Prediction cutoff is the next tick. No history row has tick >= cutoff.
        # signal is deliberately null: this family does not invent the
        # organizer signal, and it does not attach a signal to past ticks.
        # The two probability vectors are a research answer key for Y_cutoff,
        # not input features.
        for machine_id in range(n_machines):
            queries.append(
                {
                    "query_id": f"sg-{episode_id}-{machine_id}",
                    "episode_id": episode_id,
                    "machine_id": machine_id,
                    "slot": machine_id,
                    "machine_type": types[machine_id],
                    "signal": None,
                    "cutoff": n_ticks,
                    "answer_key": True,
                    "latent_oracle_probabilities": oracle_probabilities(
                        types[machine_id], prev_payout[machine_id]
                    ),
                    "observable_history_probabilities": observer_predictive(
                        types[machine_id], posterior[machine_id]
                    ),
                }
            )

    return {
        "records": records,
        "truth": truth,
        "observable_history": observable_history,
        "queries": queries,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, separators=(",", ":"), sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def mappings_document() -> dict:
    return {
        "payouts_in_probability_order": list(PAYOUTS),
        "payout_to_index": {str(payout): index for payout, index in PAYOUT_TO_IDX.items()},
        "probability_columns": list(PROBABILITY_COLUMNS),
        "missing_index": MISSING_INDEX,
        "missing_means": "unknown, not payout 0",
        "slot_index": "machine_id is the integer slot 0..n_machines-1",
        "machine_type_by_slot_mod_8": list(_SLOT_TYPES),
        "query": {
            "one_query_per_machine_per_episode": True,
            "cutoff": "n_ticks (history ticks are 0..n_ticks-1 and satisfy tick < cutoff)",
            "signal": "null; not synthesized; never written onto history records",
            "real_signal_values_on_organizer_data_only": [-1, 0, 1],
            "answer_key_fields": [
                "latent_oracle_probabilities",
                "observable_history_probabilities",
            ],
        },
        "truth": "latent-oracle P(Y_t | latent Y_{t-1}, type), including hidden Y_{t-1}",
        "observable_history": "filter P(Y_t | revealed payouts at ticks < t only)",
    }


def assumptions_document() -> list[str]:
    return [
        "Invented toy probabilities. Not fit to train, validation, or test.",
        "Own-machine order-1 latent Markov chain. No cross-machine dependence.",
        "State at tick t is committed from ticks < t only. Same-tick payouts are not inputs.",
        "Observation mask is an exogenous gap Markov chain (long gaps, short looks).",
        "Mask coins do not depend on payout, so a gap is not evidence for payout 0.",
        "Unobserved record['payout'] is the latent draw for audit, withheld from the observer.",
        "Organizer query signal is not modeled and is not backfilled into history.",
        "Machine types repeat the 8-slot pattern steady, steady, balanced x4, jackpot, jackpot.",
    ]


def write_bundle(out_dir: Path, seed: int, n_episodes: int, n_ticks: int, n_machines: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
    _write_jsonl(out_dir / "records.jsonl", data["records"])
    _write_jsonl(out_dir / "truth.jsonl", data["truth"])
    _write_jsonl(out_dir / "observable_history.jsonl", data["observable_history"])
    _write_jsonl(out_dir / "queries.jsonl", data["queries"])

    source_files = {
        "generate.py": Path(__file__).resolve(),
        "test_generator.py": Path(__file__).resolve().parent / "test_generator.py",
        "README.md": Path(__file__).resolve().parent / "README.md",
    }
    source_hashes = {
        name: sha256_file(path) for name, path in source_files.items() if path.is_file()
    }
    output_names = (
        "records.jsonl",
        "truth.jsonl",
        "observable_history.jsonl",
        "queries.jsonl",
    )
    output_hashes = {name: sha256_file(out_dir / name) for name in output_names}
    manifest = {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "source": (
            "self-authored toy generator in this directory; "
            "not organizer-released data; not fit to any split"
        ),
        "seed": seed,
        "settings": {
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
        },
        "n_records": len(data["records"]),
        "n_truth": len(data["truth"]),
        "n_observable_history": len(data["observable_history"]),
        "n_queries": len(data["queries"]),
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
        "mappings": mappings_document(),
        "assumptions": assumptions_document(),
        "do_not_train_by_default": True,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate sparse_gappy toy episodes (research only).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = write_bundle(
        args.out,
        seed=args.seed,
        n_episodes=args.episodes,
        n_ticks=args.ticks,
        n_machines=args.machines,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "eligibility": manifest["eligibility"],
                "n_records": manifest["n_records"],
                "output_hashes": manifest["output_hashes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
