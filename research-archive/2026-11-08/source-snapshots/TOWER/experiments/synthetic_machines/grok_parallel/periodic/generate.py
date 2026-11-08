"""Periodic toy slot family. RESEARCH_ONLY. Not a fit to organizer data.

Each machine has a phase in ``0 .. PERIOD-1``. The latent-oracle payout law at
tick ``t`` is the published catalog row for ``(machine_type, phase_t)``. Phase
at ``t`` is a function of the initial phase and of that machine's own outcomes
at ticks ``< t`` only. Other machines' outcomes, including at tick ``t``, are
not read.

Two filters share the catalog and differ in the kick they apply:

- latent oracle: every sampled payout kicks the phase, even if the cell is
  unobserved. This is ``truth``.
- observable history: an unobserved cell advances the clock and applies no
  payout kick. It is not treated as payout 0. ``KICK[0]`` is nonzero so those
  two updates cannot alias.

The prediction cutoff is ``n_ticks`` (the next spin). No per-tick query signal
is invented. ``kit/RULES.md`` requires organizer-released data; this generator
does not create a synthetic exception. Keep the real-only fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {payout: index for index, payout in enumerate(PAYOUTS)}
MISSING = -1  # unknown observation. Never a stand-in for payout 0 or index 0.

# src/house_edge/data.py alphabetical order. Not the task-list order.
MACHINE_TYPES = ("balanced", "jackpot", "steady")
TYPE_TO_INDEX = {name: index for index, name in enumerate(MACHINE_TYPES)}

# Slot i is machine_id i. Types are fixed and invented, not estimated.
MACHINE_TYPE_BY_SLOT = (
    "steady",
    "steady",
    "balanced",
    "balanced",
    "balanced",
    "jackpot",
    "jackpot",
    "jackpot",
)

PERIOD = 8
CLOCK_STEP = 1
# By payout index. Observed 0 kicks by 3, so "missing" (+0 kick) is not "saw a 0".
KICK_BY_INDEX = (3, 0, 1, 2, 5)
WEIGHT_TOTAL = 40
MAX_RECORDS = 8192

# Integer weights, each row sums to WEIGHT_TOTAL. Invented toy rows.
# steady and jackpot repeat every 4 phases. balanced uses all 8 phases.
CATALOG_WEIGHTS = {
    "steady": (
        (24, 8, 4, 3, 1),
        (22, 9, 5, 3, 1),
        (19, 9, 6, 4, 2),
        (21, 7, 6, 5, 1),
        (24, 8, 4, 3, 1),
        (22, 9, 5, 3, 1),
        (19, 9, 6, 4, 2),
        (21, 7, 6, 5, 1),
    ),
    "jackpot": (
        (20, 8, 6, 4, 2),
        (18, 8, 6, 4, 4),
        (16, 6, 6, 4, 8),
        (12, 6, 6, 4, 12),
        (20, 8, 6, 4, 2),
        (18, 8, 6, 4, 4),
        (16, 6, 6, 4, 8),
        (12, 6, 6, 4, 12),
    ),
    "balanced": (
        (18, 10, 6, 4, 2),
        (16, 12, 6, 4, 2),
        (16, 10, 8, 4, 2),
        (14, 10, 8, 6, 2),
        (14, 8, 10, 6, 2),
        (12, 10, 8, 6, 4),
        (12, 8, 10, 6, 4),
        (10, 10, 8, 8, 4),
    ),
}

# (observe, hide) odds. Independent of the payout draw.
OBSERVE_YES_NO = {
    "steady": (3, 1),
    "balanced": (2, 1),
    "jackpot": (1, 1),
}

STREAM_PAYOUT = 1
STREAM_MASK = 2

_MASK = 0xFFFFFFFFFFFFFFFF


def _normalize(weights: tuple[int, ...]) -> tuple[float, ...]:
    if sum(weights) != WEIGHT_TOTAL:
        raise ValueError(f"catalog row sums to {sum(weights)}, expected {WEIGHT_TOTAL}")
    if any(weight <= 0 for weight in weights):
        raise ValueError("catalog weights must be positive")
    head = [weight / WEIGHT_TOTAL for weight in weights[:-1]]
    return tuple(head + [1.0 - sum(head)])


PROB_TABLE = {
    machine_type: tuple(_normalize(row) for row in rows)
    for machine_type, rows in CATALOG_WEIGHTS.items()
}


def _mix(*parts: int) -> int:
    """Process-stable 64-bit mix. No ``hash()`` and no shared RNG."""
    mixed = 0x6A09E667F3BCC909
    for part in parts:
        mixed ^= int(part) & _MASK
        mixed = (mixed + 0x9E3779B97F4A7C15) & _MASK
        mixed = (mixed * 0xBF58476D1CE4E5B9) & _MASK
        mixed ^= mixed >> 30
    return mixed


def initial_phase(seed: int, episode_index: int, machine_id: int) -> int:
    return (seed * 17 + episode_index * 5 + machine_id * 3 + 1) % PERIOD


def catalog_probabilities(machine_type: str, phase: int) -> tuple[float, ...]:
    return PROB_TABLE[machine_type][phase % PERIOD]


def observation_payout_index(record: dict) -> int:
    """Observable payout index, or ``MISSING`` (-1). Never rewrites a gap as 0."""
    if not record["observed"]:
        return MISSING
    return PAYOUT_TO_INDEX[record["payout"]]


def advance_phase(phase: int, payout: int, observed: bool, mode: str) -> int:
    """Advance one own-machine step. ``payout`` is the latent draw."""
    if mode == "latent":
        kick = KICK_BY_INDEX[PAYOUT_TO_INDEX[payout]]
    elif mode == "observable":
        kick = KICK_BY_INDEX[PAYOUT_TO_INDEX[payout]] if observed else 0
    elif mode == "missing_as_zero":
        # Wrong filter: a gap is imputed as payout 0 and takes KICK[0].
        if observed:
            kick = KICK_BY_INDEX[PAYOUT_TO_INDEX[payout]]
        else:
            kick = KICK_BY_INDEX[PAYOUT_TO_INDEX[0]]
    else:
        raise ValueError(f"unknown phase mode: {mode}")
    return (phase + CLOCK_STEP + kick) % PERIOD


def replay_probabilities(
    seed: int,
    episode_index: int,
    machine_id: int,
    machine_type: str,
    rows: list[dict],
    mode: str,
) -> tuple[list[tuple[float, ...]], tuple[float, ...]]:
    """Probabilities at each row, then the next-spin law after the last row.

    Row ``t`` is scored from rows that precede it only. ``rows`` must be one
    machine in increasing tick order. Other machines are not an input.
    """
    phase = initial_phase(seed, episode_index, machine_id)
    stepped: list[tuple[float, ...]] = []
    for row in rows:
        stepped.append(catalog_probabilities(machine_type, phase))
        phase = advance_phase(phase, row["payout"], bool(row["observed"]), mode)
    terminal = catalog_probabilities(machine_type, phase)
    return stepped, terminal


def _cell_rng(seed: int, episode_index: int, tick: int, machine_id: int, stream: int) -> random.Random:
    # A fresh generator per cell. Nothing is carried across cells, episodes, or calls.
    return random.Random(_mix(seed, episode_index, tick, machine_id, stream))


def _validate_request(seed: int, n_episodes: int, n_ticks: int, n_machines: int) -> None:
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if n_episodes < 1 or n_ticks < 1:
        raise ValueError("n_episodes and n_ticks must be >= 1")
    if n_machines != 8:
        raise ValueError("n_machines must be 8 so machine_id is the slot index 0..7")
    n_records = n_episodes * n_ticks * n_machines
    if n_records > MAX_RECORDS:
        raise ValueError(f"{n_records} records exceeds MAX_RECORDS={MAX_RECORDS}")


def _validate_overrides(overrides: dict, n_episodes: int, n_ticks: int, n_machines: int) -> None:
    for key, value in overrides.items():
        episode_index, tick, machine_id = key
        if not (0 <= episode_index < n_episodes and 0 <= tick < n_ticks and 0 <= machine_id < n_machines):
            raise ValueError(f"override out of range: {key}")
        if "payout" in value and value["payout"] not in PAYOUT_TO_INDEX:
            raise ValueError(f"override payout not in {PAYOUTS}: {value['payout']}")
        if "observed" in value and not isinstance(value["observed"], bool):
            raise ValueError("override observed must be a bool")


def generate(
    seed: int = 0,
    n_episodes: int = 8,
    n_ticks: int = 96,
    n_machines: int = 8,
    overrides: dict | None = None,
) -> dict:
    """Return ``records``, latent-oracle ``truth``, and observable-history probs.

    ``overrides`` is a test hook. Keys are ``(episode_index, tick, machine_id)``.
    Values may set ``payout`` and/or ``observed``, replacing that cell's draw
    after its probability is fixed and before later ticks of the same machine
    update. Other cells keep their own streams.
    """
    _validate_request(seed, n_episodes, n_ticks, n_machines)
    overrides = overrides or {}
    _validate_overrides(overrides, n_episodes, n_ticks, n_machines)

    drafted: list[list[list[dict]]] = []
    for episode_index in range(n_episodes):
        episode_id = f"periodic-s{seed}-e{episode_index:04d}"
        ticks: list[list[dict]] = []
        for tick in range(n_ticks):
            cells: list[dict] = []
            for machine_id in range(n_machines):
                machine_type = MACHINE_TYPE_BY_SLOT[machine_id]
                payout_rng = _cell_rng(seed, episode_index, tick, machine_id, STREAM_PAYOUT)
                mask_rng = _cell_rng(seed, episode_index, tick, machine_id, STREAM_MASK)
                # The uniform draw does not depend on phase. The phase selects the
                # weight row later; the integer draw is stored so replay stays pure.
                payout_draw = payout_rng.randrange(WEIGHT_TOTAL)
                yes, no = OBSERVE_YES_NO[machine_type]
                observed = mask_rng.randrange(yes + no) < yes
                override = overrides.get((episode_index, tick, machine_id))
                if override and "observed" in override:
                    observed = override["observed"]
                cells.append(
                    {
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "tick": tick,
                        "machine_id": machine_id,
                        "slot_index": machine_id,
                        "machine_type": machine_type,
                        "machine_type_index": TYPE_TO_INDEX[machine_type],
                        "payout_draw": payout_draw,
                        "observed": observed,
                        "payout_override": None if not override or "payout" not in override else override["payout"],
                    }
                )
            ticks.append(cells)
        drafted.append(ticks)

    records: list[dict] = []
    truth: list[dict] = []
    observable: list[dict] = []
    queries: list[dict] = []

    for episode_index, ticks in enumerate(drafted):
        episode_id = ticks[0][0]["episode_id"]
        per_machine_rows: list[list[dict]] = [[] for _ in range(n_machines)]
        # Sample from the latent phase, which itself uses only earlier own outcomes.
        latent_phase = [
            initial_phase(seed, episode_index, machine_id) for machine_id in range(n_machines)
        ]
        for tick, cells in enumerate(ticks):
            resolved: list[dict] = []
            for machine_id, cell in enumerate(cells):
                machine_type = cell["machine_type"]
                weights = CATALOG_WEIGHTS[machine_type][latent_phase[machine_id]]
                if cell["payout_override"] is None:
                    payout_index = _index_from_draw(cell["payout_draw"], weights)
                    payout = PAYOUTS[payout_index]
                else:
                    payout = cell["payout_override"]
                observed = cell["observed"]
                record = {
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "tick": tick,
                    "machine_id": machine_id,
                    "slot_index": machine_id,
                    "machine_type": machine_type,
                    "machine_type_index": cell["machine_type_index"],
                    "payout": payout,
                    "payout_index": PAYOUT_TO_INDEX[payout] if observed else MISSING,
                    "observed": observed,
                }
                resolved.append(record)
                per_machine_rows[machine_id].append(record)
            # Synchronous commit: no cell's tick-t update is visible inside tick t.
            for machine_id, record in enumerate(resolved):
                latent_phase[machine_id] = advance_phase(
                    latent_phase[machine_id], record["payout"], record["observed"], "latent"
                )
            records.extend(resolved)

        for machine_id in range(n_machines):
            machine_type = MACHINE_TYPE_BY_SLOT[machine_id]
            rows = per_machine_rows[machine_id]
            latent_steps, latent_terminal = replay_probabilities(
                seed, episode_index, machine_id, machine_type, rows, "latent"
            )
            observed_steps, observed_terminal = replay_probabilities(
                seed, episode_index, machine_id, machine_type, rows, "observable"
            )
            for row, latent_probs, observed_probs in zip(rows, latent_steps, observed_steps):
                truth.append(
                    {
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "tick": row["tick"],
                        "machine_id": machine_id,
                        "slot_index": machine_id,
                        "probabilities": list(latent_probs),
                        "probability_kind": "latent_oracle",
                    }
                )
                observable.append(
                    {
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "tick": row["tick"],
                        "machine_id": machine_id,
                        "slot_index": machine_id,
                        "probabilities": list(observed_probs),
                        "probability_kind": "observable_history",
                    }
                )
            queries.append(
                {
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "machine_id": machine_id,
                    "slot_index": machine_id,
                    "machine_type": machine_type,
                    "machine_type_index": TYPE_TO_INDEX[machine_type],
                    "cutoff": n_ticks,
                    "signal": None,
                    "latent_oracle_probabilities": list(latent_terminal),
                    "observable_history_probabilities": list(observed_terminal),
                }
            )

    # truth was appended machine-major. Reorder to the record order
    # (episode, tick, machine) so zip(records, truth) is the same cell.
    truth.sort(key=lambda row: (row["episode_index"], row["tick"], row["machine_id"]))
    observable.sort(key=lambda row: (row["episode_index"], row["tick"], row["machine_id"]))

    return {
        "records": records,
        "truth": truth,
        "observable": observable,
        "queries": queries,
        "settings": {
            "family": "periodic",
            "seed": seed,
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
            "period": PERIOD,
            "cutoff": n_ticks,
            "max_records": MAX_RECORDS,
            "weight_total": WEIGHT_TOTAL,
            "clock_step": CLOCK_STEP,
            "probability_kind_truth": "latent_oracle",
            "probability_kind_observable": "observable_history",
            "n_records": len(records),
        },
    }


def _index_from_draw(draw: int, weights: tuple[int, ...]) -> int:
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    raise RuntimeError("payout draw fell outside the catalog row")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(out_dir: Path, settings: dict, test: dict | None = None) -> dict:
    here = Path(__file__).resolve().parent
    manifest = {
        "family": "periodic",
        "eligibility": "RESEARCH_ONLY",
        "research_only": True,
        "real_only_fallback": True,
        "eligibility_note": (
            "kit/RULES.md says use organizer-released data. Test answers and private "
            "organizer files are off limits. No verified synthetic exception was found. "
            "Preserve the real-only fallback. Do not use this family for scoring, "
            "model selection, or eligible training."
        ),
        "seed": settings["seed"],
        "settings": settings,
        "assumptions": [
            "Catalog weights are explicit invented integers over 40, not fit to train, validation, or test.",
            "steady and jackpot laws repeat every 4 phases; balanced uses 8 distinct phases.",
            "Phase uses only that machine's earlier latent payouts (oracle) or earlier observed payouts (history).",
            "A missing cell is unknown: payout_index is -1 and the observable filter applies no payout kick.",
            "KICK of payout 0 is 3, so imputing 0 is a different update from a gap.",
            "Cutoff equals n_ticks. signal is null. No historical query signal is synthesized.",
            "machine_type_index follows balanced=0, jackpot=1, steady=2.",
            "slot_index equals machine_id in 0..7.",
        ],
        "mappings": {
            "payouts": list(PAYOUTS),
            "payout_to_index": {str(payout): index for payout, index in PAYOUT_TO_INDEX.items()},
            "missing": MISSING,
            "machine_type_to_index": dict(TYPE_TO_INDEX),
            "machine_type_by_slot": list(MACHINE_TYPE_BY_SLOT),
            "kick_by_payout_index": list(KICK_BY_INDEX),
            "period": PERIOD,
            "clock_step": CLOCK_STEP,
            "observe_yes_no": {name: list(pair) for name, pair in OBSERVE_YES_NO.items()},
            "query_signal": None,
            "cutoff": "n_ticks; history ticks are strictly less than cutoff",
        },
        "source_hashes": {
            "generate.py": file_sha256(here / "generate.py"),
            "test_generator.py": file_sha256(here / "test_generator.py"),
            "README.md": file_sha256(here / "README.md"),
        },
        "output_hashes": {
            "records.jsonl": file_sha256(out_dir / "records.jsonl"),
            "truth.jsonl": file_sha256(out_dir / "truth.jsonl"),
        },
        "n_records": settings["n_records"],
        "n_truth": settings["n_records"],
    }
    if test is not None:
        manifest["test"] = test
    return manifest


def write_outputs(out_dir: str | Path, seed: int, n_episodes: int, n_ticks: int, test: dict | None = None) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bundle = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=8)
    _write_jsonl(out_path / "records.jsonl", bundle["records"])
    _write_jsonl(out_path / "truth.jsonl", bundle["truth"])
    manifest = build_manifest(out_path, bundle["settings"], test=test)
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the periodic RESEARCH_ONLY toy family.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args(argv)
    write_outputs(args.out, seed=args.seed, n_episodes=args.episodes, n_ticks=args.ticks)
    print(f"wrote records.jsonl truth.jsonl manifest.json in {args.out}")


if __name__ == "__main__":
    main()
