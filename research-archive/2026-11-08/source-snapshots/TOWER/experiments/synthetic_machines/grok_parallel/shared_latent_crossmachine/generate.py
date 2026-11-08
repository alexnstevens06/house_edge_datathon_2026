#!/usr/bin/env python3
"""Shared-latent cross-machine toy generator.

Research-only synthetic family. Organizer rules say to use organizer-released
data; no synthetic exception is verified. Do not train or submit on this
output. The real-only fallback is the public train split.

Invented mechanism (not fit to any House Edge file):

* One discrete latent per episode, shared by every machine, in {0, 1, 2}
  (cold, neutral, hot). It is drawn from the seed at tick 0.
* At tick t the latent used for sampling is a function of ticks < t only.
  After all machines have been drawn, the latent steps from the lagged score
  of payout *indices* observed at tick t. Missing cells are omitted. They are
  not payout index 0. Same-tick outcomes never enter the score.
* Even slots load the latent as-is. Odd slots load the flipped latent
  (2 - z), so two machines of the same type can move in opposite directions.
* Given the latent, spins are independent across machines.
* A cell is kept with an independent 75% mask. The mask does not depend on
  the payout. Unobserved draws stay in the record for audit and are dropped
  from the state.
* ``truth["probabilities"]`` is the latent-oracle emission.
  ``truth["observable_probabilities"]`` is the predictive distribution of a
  Bayes filter that knows these toy tables, starts from a uniform latent
  prior, and sees only kept payouts at ticks < t. The filter does not see
  the true latent.
* Queries exist only at cutoff = n_ticks. ``signal`` is null. This generator
  does not invent a signal series. Real query signals live only on organizer
  rows at the prediction cutoff.

Payout order is (0, 1, 2, 5, 20) -> indices (0, 1, 2, 3, 4). Slot index
equals ``machine_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FAMILY = "shared_latent_crossmachine"
STATUS = "RESEARCH_ONLY"
MAX_RECORDS = 8192

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
INDEX_TO_PAYOUT = {0: 0, 1: 1, 2: 2, 3: 5, 4: 20}
N_OUTCOMES = 5

# Missing is unknown. It is not payout 0 and not payout index 0.
MISSING = None

MACHINE_TYPES = ("steady", "balanced", "jackpot")
# Two steady, three balanced, three jackpot. Fixed toy assignment.
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

# Integer weights over 100. Rows are cold, neutral, hot.
# Positive loading uses the row for z. Negative loading uses the row for 2 - z.
EMISSION_WEIGHTS = {
    "steady": (
        (74, 14, 7, 4, 1),
        (50, 28, 12, 7, 3),
        (30, 30, 20, 14, 6),
    ),
    "balanced": (
        (55, 20, 12, 9, 4),
        (36, 28, 18, 12, 6),
        (18, 22, 24, 22, 14),
    ),
    "jackpot": (
        (62, 16, 10, 8, 4),
        (40, 22, 16, 12, 10),
        (20, 16, 18, 22, 24),
    ),
}

OBSERVE_PERCENT = 75
LATENT_VALUES = (0, 1, 2)
UNIFORM_BELIEF = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def machine_type(machine_id: int) -> str:
    return TYPE_BY_SLOT[machine_id % 8]


def loading_sign(machine_id: int) -> int:
    """Even slots follow the latent. Odd slots follow the flipped latent."""
    return 1 if machine_id % 2 == 0 else -1


def effective_latent(latent: int, sign: int) -> int:
    if latent not in LATENT_VALUES:
        raise ValueError(f"latent must be one of {LATENT_VALUES}")
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    return latent if sign > 0 else (2 - latent)


def emission_weights(latent: int, mtype: str, sign: int) -> tuple[int, int, int, int, int]:
    return EMISSION_WEIGHTS[mtype][effective_latent(latent, sign)]


def probs_from_weights(weights: tuple[int, ...]) -> list[float]:
    if len(weights) != N_OUTCOMES or sum(weights) != 100:
        raise ValueError("emission weights must be five integers summing to 100")
    head = [w / 100.0 for w in weights[:-1]]
    # Last coordinate closes the simplex in float arithmetic.
    return head + [1.0 - math.fsum(head)]


def oracle_probabilities(latent: int, mtype: str, sign: int) -> list[float]:
    """P(payout | true latent, type, loading). Latent-oracle probabilities."""
    return probs_from_weights(emission_weights(latent, mtype, sign))


def mixture_probabilities(belief: tuple[float, ...] | list[float], mtype: str, sign: int) -> list[float]:
    """Predictive distribution under a belief over the latent. No true latent."""
    acc = [0.0] * N_OUTCOMES
    for latent, mass in enumerate(belief):
        weights = emission_weights(latent, mtype, sign)
        for i, w in enumerate(weights):
            acc[i] += mass * (w / 100.0)
    total = math.fsum(acc)
    if total <= 0.0:
        raise ValueError("belief mixture has no mass")
    head = [v / total for v in acc[:-1]]
    return head + [1.0 - math.fsum(head)]


def subint(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little")


def initial_latent(seed: int, episode_id: int) -> int:
    return subint(seed, episode_id, "z0") % 3


def is_observed(seed: int, episode_id: int, tick: int, machine_id: int) -> bool:
    return subint(seed, episode_id, tick, machine_id, "mask") % 100 < OBSERVE_PERCENT


def sample_payout_index(
    seed: int,
    episode_id: int,
    tick: int,
    machine_id: int,
    weights: tuple[int, ...],
) -> int:
    draw = subint(seed, episode_id, tick, machine_id, "payout") % 100
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    return N_OUTCOMES - 1


def lag_score(obs_row: list[int | None]) -> int | None:
    """Sum of (payout_index - 2) over observed cells. None if every cell is missing.

    ``None`` is unknown and is skipped. Integer 0 is an observed loss and counts.
    """
    total = 0
    seen = 0
    for value in obs_row:
        if value is None:
            continue
        if value is True or value is False or not isinstance(value, int) or not 0 <= value <= 4:
            raise ValueError("observed cell must be a payout index in 0..4; missing is None, not 0")
        total += value - 2
        seen += 1
    if seen == 0:
        return None
    return total


def next_latent(latent: int, obs_row: list[int | None]) -> int:
    """Deterministic step from one latent and the just-finished tick.

    Call this only with the observation row at tick t after probabilities for
    tick t have already been produced. The result is the latent for tick t+1.
    """
    if latent not in LATENT_VALUES:
        raise ValueError(f"latent must be one of {LATENT_VALUES}")
    score = lag_score(obs_row)
    if score is None or score == 0:
        return latent
    step = 1 if score > 0 else -1
    return max(0, min(2, latent + step))


def update_belief(belief: list[float] | tuple[float, ...], obs_row: list[int | None]) -> list[float]:
    """Likelihood update from this tick, then the shared latent transition.

    Missing cells contribute no factor. They are not treated as payout index 0.
    The returned belief is the prior for the *next* tick.
    """
    post = []
    for latent, mass in enumerate(belief):
        like = mass
        for machine_id, outcome in enumerate(obs_row):
            if outcome is None:
                continue
            weights = emission_weights(latent, machine_type(machine_id), loading_sign(machine_id))
            like *= weights[outcome] / 100.0
        post.append(like)
    total = math.fsum(post)
    if total <= 0.0:
        post = list(belief)
        total = math.fsum(post)
    post = [v / total for v in post]
    nxt = [0.0, 0.0, 0.0]
    for latent, mass in enumerate(post):
        nxt[next_latent(latent, obs_row)] += mass
    norm = math.fsum(nxt)
    return [v / norm for v in nxt]


def oracle_panel(z0: int, observations: list[list[int | None]]) -> list[list[dict]]:
    """Oracle probabilities for every tick. Tick t reads observations[:t] only."""
    latent = z0
    panel: list[list[dict]] = []
    for tick, row in enumerate(observations):
        tick_rows = []
        for machine_id in range(len(row)):
            mtype = machine_type(machine_id)
            sign = loading_sign(machine_id)
            tick_rows.append(
                {
                    "tick": tick,
                    "machine_id": machine_id,
                    "latent": latent,
                    "probabilities": oracle_probabilities(latent, mtype, sign),
                }
            )
        panel.append(tick_rows)
        # State update is strictly after the tick's probabilities exist.
        latent = next_latent(latent, row)
    return panel


def history_panel(observations: list[list[int | None]]) -> list[list[list[float]]]:
    """Observable-history probabilities. Uniform prior, ticks < t only."""
    belief = list(UNIFORM_BELIEF)
    panel: list[list[list[float]]] = []
    for row in observations:
        tick_probs = []
        for machine_id in range(len(row)):
            tick_probs.append(
                mixture_probabilities(belief, machine_type(machine_id), loading_sign(machine_id))
            )
        panel.append(tick_probs)
        belief = update_belief(belief, row)
    return panel


def _check_shape(n_episodes: int, n_ticks: int, n_machines: int) -> None:
    for name, value in (
        ("n_episodes", n_episodes),
        ("n_ticks", n_ticks),
        ("n_machines", n_machines),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(n_episodes, bool) or isinstance(n_ticks, bool) or isinstance(n_machines, bool):
        raise ValueError("counts must be integers")
    total = n_episodes * n_ticks * n_machines
    if total > MAX_RECORDS:
        raise ValueError(f"refusing to generate {total} records; cap is {MAX_RECORDS}")


def observations_from_records(
    records: list[dict],
    n_ticks: int,
    n_machines: int,
    episode_id: int,
) -> list[list[int | None]]:
    """Grid of payout indices. Unobserved cells stay None even if a payout was drawn."""
    grid: list[list[int | None]] = [[None] * n_machines for _ in range(n_ticks)]
    for record in records:
        if record["episode_id"] != episode_id:
            continue
        if record["observed"]:
            grid[record["tick"]][record["machine_id"]] = record["payout_index"]
    return grid


def generate(seed: int = 0, n_episodes: int = 8, n_ticks: int = 96, n_machines: int = 8) -> dict:
    """Return records, latent-oracle truth, and cutoff queries.

    Default size is 8 * 96 * 8 = 6144 records, under the 8192 cap.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _check_shape(n_episodes, n_ticks, n_machines)

    records: list[dict] = []
    truth: list[dict] = []
    queries: list[dict] = []

    for episode_id in range(n_episodes):
        latent = initial_latent(seed, episode_id)
        belief = list(UNIFORM_BELIEF)
        for tick in range(n_ticks):
            obs_row: list[int | None] = [None] * n_machines
            for machine_id in range(n_machines):
                mtype = machine_type(machine_id)
                sign = loading_sign(machine_id)
                weights = emission_weights(latent, mtype, sign)
                payout_index = sample_payout_index(seed, episode_id, tick, machine_id, weights)
                payout = INDEX_TO_PAYOUT[payout_index]
                observed = is_observed(seed, episode_id, tick, machine_id)
                # Probabilities are fixed before any same-tick payout is stored.
                oracle = probs_from_weights(weights)
                observable = mixture_probabilities(belief, mtype, sign)
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "slot": machine_id,
                        "machine_type": mtype,
                        "payout": payout,
                        "payout_index": payout_index,
                        "observed": observed,
                    }
                )
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "slot": machine_id,
                        "probabilities": oracle,
                        "probability_kind": "latent_oracle",
                        "observable_probabilities": observable,
                        "observable_probability_kind": "observable_history",
                        "latent": latent,
                        "loading_sign": sign,
                    }
                )
                if observed:
                    obs_row[machine_id] = payout_index
            # Synchronous update from this tick only, after every machine is drawn.
            latent = next_latent(latent, obs_row)
            belief = update_belief(belief, obs_row)

        cutoff = n_ticks
        for machine_id in range(n_machines):
            mtype = machine_type(machine_id)
            sign = loading_sign(machine_id)
            queries.append(
                {
                    "query_id": f"slx-{episode_id}-{machine_id}",
                    "episode_id": episode_id,
                    "machine_id": machine_id,
                    "slot": machine_id,
                    "machine_type": mtype,
                    # Real signals exist only on organizer queries at the cutoff.
                    # This family does not invent a signal value or a signal history.
                    "signal": None,
                    "cutoff": cutoff,
                    "latent_oracle_probabilities": oracle_probabilities(latent, mtype, sign),
                    "observable_history_probabilities": mixture_probabilities(belief, mtype, sign),
                    "probability_order": list(PAYOUTS),
                }
            )

    return {
        "family": FAMILY,
        "status": STATUS,
        "research_only": True,
        "real_only_fallback": True,
        "eligibility": "unresolved",
        "seed": seed,
        "n_episodes": n_episodes,
        "n_ticks": n_ticks,
        "n_machines": n_machines,
        "records": records,
        "truth": truth,
        "queries": queries,
        "mappings": mappings(),
    }


def mappings() -> dict:
    return {
        "payouts": list(PAYOUTS),
        "payout_to_index": {str(k): v for k, v in PAYOUT_TO_INDEX.items()},
        "index_to_payout": {str(k): v for k, v in INDEX_TO_PAYOUT.items()},
        "missing": None,
        "missing_is_not_payout_index_0": True,
        "machine_types": list(MACHINE_TYPES),
        "type_by_slot_mod_8": list(TYPE_BY_SLOT),
        "slot_equals_machine_id": True,
        "loading_sign": "even machine_id +1, odd machine_id -1 (flipped latent)",
        "history_ticks": "0 .. cutoff-1",
        "query_cutoff": "n_ticks; the cutoff spin is not a history row",
        "signal": (
            "null on synthetic queries; no per-tick signal is emitted; "
            "organizer signal is used only at a real prediction cutoff"
        ),
        "probability_order": list(PAYOUTS),
        "probability_kinds": {
            "probabilities": "latent_oracle",
            "observable_probabilities": "observable_history",
        },
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    lines = [json.dumps(row, separators=(",", ":"), allow_nan=False) for row in rows]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    path.write_bytes(payload)
    return payload


def write_bundle(out_dir: Path, seed: int, n_episodes: int, n_ticks: int, n_machines: int) -> dict:
    bundle = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_bytes = _write_jsonl(out_dir / "records.jsonl", bundle["records"])
    truth_bytes = _write_jsonl(out_dir / "truth.jsonl", bundle["truth"])
    queries_bytes = _write_jsonl(out_dir / "queries.jsonl", bundle["queries"])
    source = Path(__file__).resolve()
    manifest = {
        "family": FAMILY,
        "status": STATUS,
        "research_only": True,
        "real_only_fallback": True,
        "eligibility": "unresolved",
        "organizer_rule": "Use organizer-released data. No verified synthetic exception.",
        "seed": seed,
        "n_episodes": n_episodes,
        "n_ticks": n_ticks,
        "n_machines": n_machines,
        "record_count": len(bundle["records"]),
        "truth_count": len(bundle["truth"]),
        "query_count": len(bundle["queries"]),
        "max_records": MAX_RECORDS,
        "settings": {
            "observe_percent": OBSERVE_PERCENT,
            "emission_weights": {
                mtype: [list(row) for row in rows] for mtype, rows in EMISSION_WEIGHTS.items()
            },
            "type_by_slot": list(TYPE_BY_SLOT),
            "latent_values": list(LATENT_VALUES),
            "initial_belief": list(UNIFORM_BELIEF),
            "fitted_to_real_data": False,
            "assumptions": [
                "Shared latent in {cold, neutral, hot} steps from the lagged score of observed payout indices.",
                "Missing cells are skipped, not coded as payout 0.",
                "Same-tick outcomes do not affect current probabilities or each other.",
                "Odd slots emit under the flipped latent.",
                "Oracle probabilities condition on the true latent. Observable probabilities are the uniform-prior filter.",
                "Signal is not invented. Queries sit only at cutoff = n_ticks.",
            ],
        },
        "mappings": bundle["mappings"],
        "source_sha256": {"generate.py": sha256_file(source)},
        "output_sha256": {
            "records.jsonl": sha256_bytes(records_bytes),
            "truth.jsonl": sha256_bytes(truth_bytes),
            "queries.jsonl": sha256_bytes(queries_bytes),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the shared_latent_crossmachine toy family.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", required=True, help="Directory for records.jsonl, truth.jsonl, manifest.json")
    args = parser.parse_args(argv)
    manifest = write_bundle(
        Path(args.out),
        seed=args.seed,
        n_episodes=args.episodes,
        n_ticks=args.ticks,
        n_machines=args.machines,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "record_count": manifest["record_count"],
                "out": str(Path(args.out).resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
