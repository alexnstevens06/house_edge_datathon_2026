#!/usr/bin/env python3
"""Hierarchical toy slot family (research only).

Invented probabilities. Not fit to House Edge train, validation, or test.
Organizer rules require organizer-released data. No verified synthetic
exception is applied here. Scored training must keep the real-only fallback.

Hierarchy, top down:
  1. Episode house vector (shared, drawn before any tick).
  2. Type-level mixture of that house with a fixed type prototype.
     All machines of a type in the episode share this mixture.
  3. Machine-level logit tilt around the type mixture.
  4. Tick payout given the machine base and that machine's own observed
     payouts at ticks strictly earlier than t.

Truth rows are latent-oracle conditionals. Observable-history probabilities
use only the public type prototype and usable past payouts. They are not the
truth rows.

Causal mask: same episode, same machine, tick < t, observed is True.
A missing row is unknown. Its payout is not a zero. Same-tick outcomes,
including other machines, are not inputs. No signal series is invented.
Real query signals exist only at an organizer prediction cutoff; this
generator leaves query signal null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

FAMILY = "machine_type_hierarchical"
RESEARCH_ONLY = True
REAL_ONLY_FALLBACK = True
ELIGIBILITY = "RESEARCH_ONLY"
ELIGIBILITY_NOTE = (
    "Unresolved. Organizer rules say use organizer-released data. "
    "No verified synthetic exception. Do not replace real train, "
    "validation, or test with these rows."
)

MAX_RECORDS = 8192
PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {payout: index for index, payout in enumerate(PAYOUTS)}
INDEX_TO_PAYOUT = PAYOUTS

# Deterministic type of machine_id. Slot k uses TYPE_BY_SLOT[k % 8].
TYPE_BY_SLOT = (
    "steady",
    "steady",
    "balanced",
    "balanced",
    "balanced",
    "jackpot",
    "jackpot",
    "jackpot",
)
MACHINE_TYPES = ("steady", "balanced", "jackpot")

# Fixed invented prototypes. Each sums to 1. Order is PAYOUTS.
TYPE_PROTOTYPE = {
    "steady": (0.62, 0.22, 0.10, 0.05, 0.01),
    "balanced": (0.40, 0.28, 0.18, 0.10, 0.04),
    "jackpot": (0.70, 0.12, 0.08, 0.04, 0.06),
}
HOUSE_VECTORS = (
    (0.58, 0.22, 0.12, 0.06, 0.02),
    (0.48, 0.24, 0.16, 0.08, 0.04),
    (0.52, 0.16, 0.12, 0.12, 0.08),
)
# Weight on the episode house inside each type-level mixture.
EPISODE_MIX = 0.20
TYPE_SIGMA = {"steady": 0.04, "balanced": 0.20, "jackpot": 0.35}
# Dirichlet prior strength around the machine base. Larger ignores history more.
TAU = {"steady": 40.0, "balanced": 6.0, "jackpot": 12.0}
# Extra jackpot mixture when the last observed payout is 5 or 20.
JACKPOT_HOT = (0.35, 0.15, 0.15, 0.15, 0.20)
HEAT_MIX = 0.50
# MCAR rate. Drawn before the payout and independent of its value.
MISSING_RATE = 0.25

USABLE_INPUT_MASK = (
    "same episode_id, same machine_id, tick < prediction tick, observed is true"
)
PROBABILITY_KIND_ORACLE = "latent_oracle"
PROBABILITY_KIND_OBSERVABLE = "observable_history"

ASSUMPTIONS = (
    "Toy probabilities are explicit constants in generate.py, not estimates from organizer data.",
    "Eligibility is RESEARCH_ONLY. Real-only fallback stays in force for scored training.",
    "Machine type is TYPE_BY_SLOT[machine_id % 8], not a fitted assignment.",
    "Episode house is drawn before ticks and does not depend on outcomes.",
    "Type-level mixture is shared by machines of that type; each machine then gets its own logit tilt.",
    "P(payout at t) uses only that machine's observed payouts with tick < t.",
    "Unobserved payouts are unknown, not 0, and do not enter counts.",
    "No contemporaneous cross-machine outcomes enter the state.",
    "Missingness is MCAR with rate 0.25, independent of the payout.",
    "The stored payout on an unobserved row is a latent bookkeeping draw. Mask it before any model input.",
    "Query signal is null. Historical signals are not invented. Real signals exist only on organizer queries at cutoff.",
    "Query cutoff is n_ticks - 1. One query per machine per episode.",
    "Truth probabilities are latent-oracle. Observable-history probabilities drop the house and the machine tilt.",
    "RNG is random.Random (MT19937) with a fixed stream mix. No clock and no hash randomization.",
)


def machine_type_for(machine_id: int) -> str:
    if isinstance(machine_id, bool) or not isinstance(machine_id, int):
        raise ValueError("machine_id must be an int")
    if machine_id < 0:
        raise ValueError("machine_id must be >= 0")
    return TYPE_BY_SLOT[machine_id % 8]


def prediction_cutoff(n_ticks: int) -> int:
    """Tick a query asks for. Inputs are usable rows with tick < cutoff."""
    if isinstance(n_ticks, bool) or not isinstance(n_ticks, int) or n_ticks < 1:
        raise ValueError("n_ticks must be an int >= 1")
    return n_ticks - 1


def mix_seed(seed: int, episode_index: int, stream_id: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise ValueError("episode_index must be an int")
    mixed = (seed & 0xFFFFFFFF) ^ ((episode_index + 1) * 0x9E3779B9) ^ (stream_id * 0x85EBCA6B)
    return mixed & 0xFFFFFFFF


def _rng(seed: int, episode_index: int, stream_id: int) -> random.Random:
    return random.Random(mix_seed(seed, episode_index, stream_id))


def _blend(left: tuple[float, ...], right: tuple[float, ...], weight_on_right: float) -> tuple[float, ...]:
    keep = 1.0 - weight_on_right
    return tuple(keep * a + weight_on_right * b for a, b in zip(left, right))


def softmax_tilt(base: tuple[float, ...], noise: tuple[float, ...], sigma: float) -> tuple[float, ...]:
    """Logit tilt. noise entries are in [0, 1)."""
    logits = [math.log(p) + sigma * (u - 0.5) for p, u in zip(base, noise)]
    peak = max(logits)
    weights = [math.exp(item - peak) for item in logits]
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


def latent_oracle_probabilities(
    machine_type: str,
    machine_base: tuple[float, ...] | list[float],
    observed_payouts_before_t: list[int],
) -> list[float]:
    """Latent-oracle P(payout_t | type, machine base, observed own payouts before t).

    observed_payouts_before_t must already drop missing rows. Do not pass a
    placeholder 0 for a gap.
    """
    if machine_type not in TAU:
        raise ValueError(f"unknown machine_type {machine_type!r}")
    # No usable history: the posterior mean is the machine base itself.
    # Skipping the tau/tau rescale keeps binary values equal to that base.
    if not observed_payouts_before_t:
        return [float(value) for value in machine_base]
    counts = [0, 0, 0, 0, 0]
    for payout in observed_payouts_before_t:
        counts[PAYOUT_TO_INDEX[payout]] += 1
    total = float(len(observed_payouts_before_t))
    tau = TAU[machine_type]
    blended = [(tau * float(machine_base[i]) + counts[i]) / (tau + total) for i in range(5)]
    if machine_type == "jackpot" and observed_payouts_before_t and observed_payouts_before_t[-1] in (5, 20):
        hot = JACKPOT_HOT
        weight = HEAT_MIX
        blended = [(1.0 - weight) * blended[i] + weight * hot[i] for i in range(5)]
    return blended


def observable_history_probabilities(
    machine_type: str,
    observed_payouts_before_t: list[int],
) -> list[float]:
    """Probabilities a history-only reader can form without episode or machine latents.

    Uses TYPE_PROTOTYPE only. This is not the latent oracle and is not written
    to truth.jsonl.
    """
    return latent_oracle_probabilities(
        machine_type,
        TYPE_PROTOTYPE[machine_type],
        observed_payouts_before_t,
    )


def usable_payouts(records: list[dict], episode_id, machine_id, before_tick: int) -> list[int]:
    """Apply the mask. Missing rows are skipped, not read as payout 0."""
    rows = [
        row
        for row in records
        if row["episode_id"] == episode_id
        and row["machine_id"] == machine_id
        and row["tick"] < before_tick
        and row["observed"] is True
    ]
    rows.sort(key=lambda row: row["tick"])
    return [row["payout"] for row in rows]


def draw_latents(seed: int, episode_index: int, n_machines: int) -> dict:
    """Draw hierarchy levels 1-3. Independent of n_ticks and of outcomes."""
    rng = _rng(seed, episode_index, stream_id=1)
    house_index = rng.randrange(len(HOUSE_VECTORS))
    house = HOUSE_VECTORS[house_index]
    type_level = {
        name: _blend(TYPE_PROTOTYPE[name], house, EPISODE_MIX) for name in MACHINE_TYPES
    }
    types = []
    noises = []
    bases = []
    for machine_id in range(n_machines):
        name = machine_type_for(machine_id)
        noise = tuple(rng.random() for _ in range(5))
        types.append(name)
        noises.append(noise)
        bases.append(softmax_tilt(type_level[name], noise, TYPE_SIGMA[name]))
    return {
        "house_index": house_index,
        "house": house,
        "type_level": type_level,
        "machine_type": types,
        "noise": noises,
        "machine_base": bases,
    }


def _check_request(n_episodes: int, n_ticks: int, n_machines: int, allow_oversize: bool) -> None:
    for name, value in (
        ("n_episodes", n_episodes),
        ("n_ticks", n_ticks),
        ("n_machines", n_machines),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an int >= 1")
    total = n_episodes * n_ticks * n_machines
    if total > MAX_RECORDS and not allow_oversize:
        raise ValueError(f"refusing {total} records; cap is {MAX_RECORDS}")


def _sample_payout(rng: random.Random, probs: list[float]) -> int:
    draw = rng.random()
    cumulative = 0.0
    last = PAYOUTS[-1]
    for payout, prob in zip(PAYOUTS, probs):
        cumulative += prob
        if draw < cumulative or payout == last:
            return payout
    return last


def generate(
    seed: int = 0,
    n_episodes: int = 8,
    n_ticks: int = 96,
    n_machines: int = 8,
    allow_oversize: bool = False,
    interventions: dict | None = None,
) -> dict:
    """Return records, latent-oracle truth, and cutoff queries.

    interventions is a test hook. Keys are (episode_id, tick, machine_id).
    Values are {"payout": int, "observed": bool}. The original random draw is
    still consumed, then replaced, so an intervention at tick t leaves ticks
    < t identical to an unintervened run with the same seed and sizes.
    """
    _check_request(n_episodes, n_ticks, n_machines, allow_oversize)
    interventions = interventions or {}
    cutoff = prediction_cutoff(n_ticks)
    records: list[dict] = []
    truth: list[dict] = []
    queries: list[dict] = []

    for episode_index in range(n_episodes):
        episode_id = episode_index
        latents = draw_latents(seed, episode_index, n_machines)
        tick_rng = _rng(seed, episode_index, stream_id=2)
        # Per-machine observed payouts with tick < current t, in time order.
        observed_state = [[] for _ in range(n_machines)]

        for tick in range(n_ticks):
            pending = []
            for machine_id in range(n_machines):
                machine_type = latents["machine_type"][machine_id]
                # State still holds only ticks < tick. Same-tick samples are pending.
                probs = latent_oracle_probabilities(
                    machine_type,
                    latents["machine_base"][machine_id],
                    observed_state[machine_id],
                )
                observed = tick_rng.random() >= MISSING_RATE
                payout = _sample_payout(tick_rng, probs)
                override = interventions.get((episode_id, tick, machine_id))
                if override is not None:
                    payout = int(override["payout"])
                    observed = override["observed"] is True
                    if payout not in PAYOUT_TO_INDEX:
                        raise ValueError(f"intervention payout {payout} is not in {PAYOUTS}")
                pending.append((machine_id, machine_type, probs, payout, observed))

            for machine_id, machine_type, probs, payout, observed in pending:
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
                        "machine_type": machine_type,
                        "probability_kind": PROBABILITY_KIND_ORACLE,
                        "probabilities": list(probs),
                    }
                )
                if observed:
                    observed_state[machine_id].append(payout)

        for machine_id in range(n_machines):
            queries.append(
                {
                    "query_id": f"mth-e{episode_id:04d}-m{machine_id:02d}-c{cutoff}",
                    "episode_id": episode_id,
                    "machine_id": machine_id,
                    "machine_type": latents["machine_type"][machine_id],
                    "cutoff": cutoff,
                    "signal": None,
                }
            )

    return {
        "family": FAMILY,
        "research_only": RESEARCH_ONLY,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "eligibility": ELIGIBILITY,
        "records": records,
        "truth": truth,
        "queries": queries,
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def mappings() -> dict:
    return {
        "payout_to_slot": {str(payout): PAYOUT_TO_INDEX[payout] for payout in PAYOUTS},
        "slot_to_payout": {str(index): payout for index, payout in enumerate(PAYOUTS)},
        "machine_id_to_type_mod8": {str(index): name for index, name in enumerate(TYPE_BY_SLOT)},
        "usable_input_mask": USABLE_INPUT_MASK,
        "query_cutoff": "n_ticks - 1",
        "query_signal": (
            "null. Not synthesized. Real signals are organizer query fields at cutoff only."
        ),
        "truth_probability_order": list(PAYOUTS),
        "truth_probability_kind": PROBABILITY_KIND_ORACLE,
        "observable_probability_kind": PROBABILITY_KIND_OBSERVABLE,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, separators=(",", ":"), ensure_ascii=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset(out_dir: Path, bundle: dict, settings: dict, source_dir: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    truth_path = out_dir / "truth.jsonl"
    queries_path = out_dir / "queries.jsonl"
    _write_jsonl(records_path, bundle["records"])
    _write_jsonl(truth_path, bundle["truth"])
    _write_jsonl(queries_path, bundle["queries"])

    source_hashes = {}
    if source_dir is not None:
        for name in ("generate.py", "test_generator.py", "README.md"):
            path = source_dir / name
            if path.is_file():
                source_hashes[name] = sha256_file(path)

    manifest = {
        "family": FAMILY,
        "status": ELIGIBILITY,
        "eligibility": "unresolved",
        "research_only": True,
        "real_only_fallback": True,
        "eligibility_note": ELIGIBILITY_NOTE,
        "seed": settings["seed"],
        "settings": settings,
        "assumptions": list(ASSUMPTIONS),
        "mappings": mappings(),
        "source_hashes": source_hashes,
        "output_hashes": {
            "records.jsonl": sha256_file(records_path),
            "truth.jsonl": sha256_file(truth_path),
            "queries.jsonl": sha256_file(queries_path),
        },
        "n_records": len(bundle["records"]),
        "n_truth": len(bundle["truth"]),
        "n_queries": len(bundle["queries"]),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate machine_type_hierarchical toy data.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-oversize", action="store_true")
    args = parser.parse_args(argv)
    bundle = generate(
        seed=args.seed,
        n_episodes=args.episodes,
        n_ticks=args.ticks,
        n_machines=args.machines,
        allow_oversize=args.allow_oversize,
    )
    settings = {
        "seed": args.seed,
        "n_episodes": args.episodes,
        "n_ticks": args.ticks,
        "n_machines": args.machines,
        "cutoff": prediction_cutoff(args.ticks),
        "max_records": MAX_RECORDS,
        "missing_rate": MISSING_RATE,
        "payouts": list(PAYOUTS),
    }
    source_dir = Path(__file__).resolve().parent
    manifest = write_dataset(Path(args.out), bundle, settings, source_dir)
    print(
        f"{FAMILY} RESEARCH_ONLY records={manifest['n_records']} "
        f"out={args.out} real_only_fallback=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
