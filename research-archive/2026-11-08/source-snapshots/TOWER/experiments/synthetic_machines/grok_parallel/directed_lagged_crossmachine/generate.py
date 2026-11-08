#!/usr/bin/env python3
"""Directed lagged cross-machine toy generator. RESEARCH_ONLY.

Invented process, not fit to organizer data. Each machine's payout at tick t
is drawn from a categorical distribution that depends only on its directed
parent's latent payout at tick t-1 (a ring: parent of i is i-1 mod n).
Tick t does not read any machine's tick-t outcome. A missing observation is
unknown, not payout 0.

Latent-oracle probabilities condition on the parent's true payout even when
that spin was unobserved. Observable-history probabilities condition on a
payout only when ``observed`` is true; otherwise they mix over the filtered
belief. Those two distributions differ exactly when the lag parent is missing.

Query ``signal`` is a constant placeholder on cutoff rows only. It is not a
function of history and is not an input to the kernel. Machine type indices
are the public map balanced=0, jackpot=1, steady=2. Each episode has its own
RNG stream, so a longer horizon does not shift later episodes. Raw ``payout``
on an unobserved row is latent only and is never a feature; use
``as_feature_row``. Eligibility of synthetic rows is unresolved; keep the
real-data fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path

FAMILY = "directed_lagged_crossmachine"
STATUS = "RESEARCH_ONLY"
ELIGIBILITY = (
    "unresolved: organizer rules require organizer-released data; "
    "no verified synthetic-data exception"
)
REAL_ONLY_FALLBACK = True
MAX_RECORDS = 8192
LAG = 1

# Payout index order is the public order 0, 1, 2, 5, 20.
PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
INDEX_TO_PAYOUT = {0: 0, 1: 1, 2: 2, 3: 5, 4: 20}
N_PAYOUTS = 5
MISSING = -1  # feature index for an unobserved spin; never means payout 0

# Authoritative public type indices. These are not positions in a name list
# and not slot numbers. balanced=0, jackpot=1, steady=2.
TYPE_TO_INDEX = {"balanced": 0, "jackpot": 1, "steady": 2}
INDEX_TO_TYPE = {0: "balanced", 1: "jackpot", 2: "steady"}

# Explicit invented base counts. Each tuple sums to 20. Not estimated.
BASE_COUNTS = {
    "steady": (10, 5, 3, 1, 1),
    "balanced": (4, 5, 6, 3, 2),
    "jackpot": (6, 3, 2, 3, 6),
}
BASE_DENOM = 20
KEEP = Fraction(7, 10)  # weight on the machine's own base
PUSH = Fraction(3, 10)  # weight on the lagged parent's payout distribution

# Cutoff queries only. Not estimated and not written onto history rows.
QUERY_SIGNAL = 0
QUERY_SIGNAL_SCOPE = "prediction_cutoff_only"

ASSUMPTIONS = [
    "Toy categorical kernel invented for this family; no fit to organizer train, validation, or test.",
    "Machine type index is the public map balanced=0, jackpot=1, steady=2. randrange(3) is that index, not a position in a display list.",
    "Each episode has its own SHA-256 RNG stream from (family, seed, episode_index). Extending n_ticks does not move other episodes or earlier ticks.",
    "Directed ring: the parent of machine i is (i - 1) mod n_machines. Lag is exactly 1 tick. Tick t reads only ticks < t.",
    "P(child | parent) = 7/10 * base[child type] + 3/10 * belief(parent payout). Tick 0 uses the base.",
    "The latent draw uses the parent's true payout. Unobserved does not change the latent payout.",
    "Observable history uses the parent's payout only if that spin is observed; otherwise the filtered belief. Missing is not payout 0 and does not carry the last seen payout forward.",
    "Observation mask is a deterministic function of (seed, episode, tick, machine), independent of payout. Residue 0 mod 5 is missing.",
    "slot_index equals machine_id. Payout index order is 0,1,2,5,20.",
    "Raw payout on an unobserved row is latent simulator state and is never a feature. as_feature_row drops payout and payout_index and sets feature_payout_index to -1.",
    "One query per machine at cutoff = n_ticks (the first tick after the generated history). signal is the constant 0 and is ignored by the kernel.",
    "RESEARCH_ONLY. Competition eligibility is unresolved. Real organizer-released data remains the fallback.",
]


def parent_slot(machine_id: int, n_machines: int) -> int:
    """Directed parent. Influence flows parent -> machine_id one tick later."""
    if n_machines < 1:
        raise ValueError("n_machines must be positive")
    if isinstance(machine_id, bool) or not isinstance(machine_id, int):
        raise ValueError("machine_id must be int")
    return (machine_id - 1) % n_machines


def directed_edges(n_machines: int) -> list[tuple[int, int]]:
    """(parent, child) edges. The child reads the parent at lag 1."""
    return [(parent_slot(child, n_machines), child) for child in range(n_machines)]


def slot_index(machine_id: int, n_machines: int) -> int:
    """Identity slot map: machine_id is the column index."""
    if isinstance(machine_id, bool) or not isinstance(machine_id, int):
        raise ValueError("machine_id must be int")
    if not 0 <= machine_id < n_machines:
        raise ValueError(f"machine_id {machine_id} outside 0..{n_machines - 1}")
    return machine_id


def episode_rng(seed: int, episode_index: int) -> random.Random:
    """RNG for one episode. Independent of n_ticks and of every other episode.

    A global stream would let a longer horizon in episode 0 consume extra
    draws and shift episode 1. The seed is a SHA-256 prefix, not hash(), so
    it does not depend on process salt.
    """
    payload = f"directed_lagged_crossmachine|{int(seed)}|{int(episode_index)}".encode("ascii")
    episode_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    return random.Random(episode_seed)


def as_feature_row(record: dict) -> dict:
    """Feature adapter. Raw payout is never a feature.

    The simulator record may still carry ``payout`` when ``observed`` is false.
    That value is latent only. This view omits ``payout`` and ``payout_index``.
    ``feature_payout_index`` is the payout index if observed, else MISSING (-1).
    """
    observed = record["observed"]
    if not isinstance(observed, bool):
        raise ValueError("observed must be a bool")
    machine_type = record["machine_type"]
    if machine_type not in TYPE_TO_INDEX:
        raise ValueError(f"unknown machine_type {machine_type}")
    if observed:
        feature = PAYOUT_TO_INDEX[record["payout"]]
    else:
        feature = MISSING
    return {
        "episode_id": record["episode_id"],
        "tick": record["tick"],
        "machine_id": record["machine_id"],
        "slot_index": record["slot_index"],
        "machine_type": machine_type,
        "machine_type_index": TYPE_TO_INDEX[machine_type],
        "observed": observed,
        "feature_payout_index": feature,
    }


def observed_flag(seed: int, episode_index: int, tick: int, machine_id: int) -> bool:
    """True when the spin is shown. Independent of the payout value.

    ``(linear form) % 5 == 0`` is missing, so a gap is a coordinate property,
    not a zero payout. Seed 0, episode 0, tick 0, machine 0 is missing.
    """
    residue = (seed * 17 + episode_index * 13 + tick * 7 + machine_id * 3) % 5
    return residue != 0


def base_fractions(machine_type: str) -> tuple[Fraction, ...]:
    counts = BASE_COUNTS[machine_type]
    return tuple(Fraction(count, BASE_DENOM) for count in counts)


def one_hot(payout_index: int) -> tuple[Fraction, ...]:
    if not 0 <= payout_index < N_PAYOUTS:
        raise ValueError(f"payout index {payout_index} out of range")
    return tuple(Fraction(1 if i == payout_index else 0) for i in range(N_PAYOUTS))


def transition(
    machine_type: str, parent_belief: tuple[Fraction, ...] | None
) -> tuple[Fraction, ...]:
    """Kernel. ``parent_belief is None`` means there is no lag parent (tick 0)."""
    base = base_fractions(machine_type)
    if parent_belief is None:
        return base
    if len(parent_belief) != N_PAYOUTS:
        raise ValueError("parent belief must have length 5")
    mixed = tuple(KEEP * base[j] + PUSH * parent_belief[j] for j in range(N_PAYOUTS))
    if sum(mixed) != 1:
        raise RuntimeError(f"kernel is not a simplex: {sum(mixed)}")
    return mixed


def latent_oracle_fractions(
    machine_type: str, parent_payout: int | None
) -> tuple[Fraction, ...]:
    """P(payout at t | latent parent payout at t-1). None parent => tick-0 base."""
    if parent_payout is None:
        return transition(machine_type, None)
    if parent_payout not in PAYOUT_TO_INDEX:
        raise ValueError(f"parent payout {parent_payout} is not in {PAYOUTS}")
    return transition(machine_type, one_hot(PAYOUT_TO_INDEX[parent_payout]))


def pack_probs(fracs: tuple[Fraction, ...]) -> tuple[tuple[int, ...], int]:
    """Exact integer weights. Numerators sum to the denominator."""
    if len(fracs) != N_PAYOUTS:
        raise ValueError("expected 5 probabilities")
    denominator = 1
    for frac in fracs:
        if frac < 0:
            raise ValueError("negative probability")
        denominator = math.lcm(denominator, frac.denominator)
    numerators = [frac.numerator * (denominator // frac.denominator) for frac in fracs]
    divisor = denominator
    for numerator in numerators:
        divisor = math.gcd(divisor, numerator)
    if divisor > 1:
        numerators = [numerator // divisor for numerator in numerators]
        denominator //= divisor
    if sum(numerators) != denominator:
        raise RuntimeError("packed probabilities are not a simplex")
    return tuple(numerators), denominator


def sample_payout(rng: random.Random, fracs: tuple[Fraction, ...]) -> int:
    numerators, denominator = pack_probs(fracs)
    draw = rng.randrange(denominator)
    cumulative = 0
    for index, numerator in enumerate(numerators):
        cumulative += numerator
        if draw < cumulative:
            return PAYOUTS[index]
    raise RuntimeError("categorical sample fell through")


def prob_fields(fracs: tuple[Fraction, ...], *, prefix: str = "") -> dict:
    numerators, denominator = pack_probs(fracs)
    floats = [float(Fraction(numerator, denominator)) for numerator in numerators]
    if prefix:
        return {
            f"{prefix}probabilities": floats,
            f"{prefix}numerator": list(numerators),
            f"{prefix}denominator": denominator,
        }
    return {
        "probabilities": floats,
        "probability_numerator": list(numerators),
        "probability_denominator": denominator,
    }


def _require_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _validate_settings(n_episodes: int, n_ticks: int, n_machines: int) -> None:
    n_episodes = _require_int("n_episodes", n_episodes)
    n_ticks = _require_int("n_ticks", n_ticks)
    n_machines = _require_int("n_machines", n_machines)
    if n_episodes < 1 or n_ticks < 1 or n_machines < 1:
        raise ValueError("n_episodes, n_ticks, and n_machines must be >= 1")
    count = n_episodes * n_ticks * n_machines
    if count > MAX_RECORDS:
        raise ValueError(f"{count} records exceeds MAX_RECORDS={MAX_RECORDS}")


def _split_episodes(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for row in records:
        episode_id = row["episode_id"]
        tick = _require_int("tick", row["tick"])
        machine_id = _require_int("machine_id", row["machine_id"])
        key = (episode_id, tick, machine_id)
        if key in seen:
            raise ValueError(f"duplicate record {key}")
        seen.add(key)
        if row["machine_type"] not in BASE_COUNTS:
            raise ValueError(f"unknown machine_type {row['machine_type']}")
        payout = _require_int("payout", row["payout"])
        if payout not in PAYOUT_TO_INDEX:
            raise ValueError(f"payout {payout} is not in {PAYOUTS}")
        if not isinstance(row["observed"], bool):
            raise ValueError("observed must be a bool")
        grouped.setdefault(episode_id, []).append(row)
    return grouped


def conditional_probabilities(
    records: list[dict], predict_cutoff: int | None = None
) -> dict:
    """Oracle and observable conditionals from a panel.

    Synchronous: the distribution at tick t is fixed from machine state at
    ticks < t, then this tick's payouts update the state. Same-tick outcomes
    are not inputs. An absent machine-tick is unknown, not payout 0. Oracle
    is None when the lag parent's latent payout is not in the panel.

    ``predict_cutoff``, when set, also returns the distribution of that future
    tick (no payout is read at or after the cutoff).
    """
    if predict_cutoff is not None:
        predict_cutoff = _require_int("predict_cutoff", predict_cutoff)
        if predict_cutoff < 0:
            raise ValueError("predict_cutoff must be >= 0")

    oracle: dict[tuple, tuple[Fraction, ...] | None] = {}
    observable: dict[tuple, tuple[Fraction, ...]] = {}
    cutoffs: dict[str, dict] = {}

    for episode_id, rows in _split_episodes(records).items():
        machine_ids = {row["machine_id"] for row in rows}
        n_machines = max(machine_ids) + 1
        if machine_ids != set(range(n_machines)):
            raise ValueError(
                f"{episode_id}: machine_id values must be exactly 0..{n_machines - 1}"
            )
        types: dict[int, str] = {}
        by_tick: dict[int, dict[int, dict]] = {}
        max_tick = 0
        for row in rows:
            machine_id = row["machine_id"]
            previous = types.get(machine_id)
            if previous is not None and previous != row["machine_type"]:
                raise ValueError(f"{episode_id} machine {machine_id} type changed")
            types[machine_id] = row["machine_type"]
            by_tick.setdefault(row["tick"], {})[machine_id] = row
            if row["tick"] > max_tick:
                max_tick = row["tick"]
        last_tick = max_tick
        if predict_cutoff is not None:
            last_tick = max(last_tick, predict_cutoff)

        # None until tick 0 has been written: there is no lag parent yet.
        previous_state: dict[int, dict] | None = None
        for tick in range(last_tick + 1):
            # Read only previous_state (ticks < tick). Do not read rows at `tick`.
            for machine_id in range(n_machines):
                if previous_state is None:
                    oracle_fracs: tuple[Fraction, ...] | None = transition(types[machine_id], None)
                    observable_fracs = oracle_fracs
                else:
                    parent = previous_state[parent_slot(machine_id, n_machines)]
                    observable_fracs = transition(types[machine_id], parent["forward_belief"])
                    if parent["latent_payout"] is None:
                        oracle_fracs = None
                    else:
                        oracle_fracs = transition(
                            types[machine_id],
                            one_hot(PAYOUT_TO_INDEX[parent["latent_payout"]]),
                        )
                key = (episode_id, tick, machine_id)
                oracle[key] = oracle_fracs
                observable[key] = observable_fracs

            new_state: dict[int, dict] = {}
            rows_here = by_tick.get(tick, {})
            for machine_id in range(n_machines):
                predictive = observable[(episode_id, tick, machine_id)]
                row = rows_here.get(machine_id)
                if row is None:
                    # Hole: latent payout is unavailable. Belief stays the
                    # predictive distribution, which is not a zero payout.
                    new_state[machine_id] = {
                        "latent_payout": None,
                        "observed": False,
                        "forward_belief": predictive,
                    }
                elif row["observed"]:
                    new_state[machine_id] = {
                        "latent_payout": row["payout"],
                        "observed": True,
                        "forward_belief": one_hot(PAYOUT_TO_INDEX[row["payout"]]),
                    }
                else:
                    new_state[machine_id] = {
                        "latent_payout": row["payout"],
                        "observed": False,
                        "forward_belief": predictive,
                    }
            previous_state = new_state

        if predict_cutoff is not None:
            cutoffs[episode_id] = {}
            for machine_id in range(n_machines):
                key = (episode_id, predict_cutoff, machine_id)
                cutoffs[episode_id][machine_id] = {
                    "cutoff": predict_cutoff,
                    "machine_type": types[machine_id],
                    "latent_oracle": oracle[key],
                    "observable_history": observable[key],
                }
    return {
        "latent_oracle": oracle,
        "observable_history": observable,
        "cutoff": cutoffs,
    }


def generate(
    seed: int = 0,
    n_episodes: int = 8,
    n_ticks: int = 96,
    n_machines: int = 8,
) -> dict:
    """Return ``records``, ``truth`` (latent oracle), and cutoff ``queries``.

    Each episode has its own RNG from ``episode_rng``. Inside an episode the
    stream is ``n_machines`` type-index draws (``randrange(3)`` mapped by
    ``INDEX_TO_TYPE``), then one categorical draw per machine per tick.
    Extending ``n_ticks`` only appends draws on that episode's stream.
    The observation mask does not consume randomness. Tick ``t`` is scored
    from tick ``t-1`` before any same-tick draw.
    """
    _validate_settings(n_episodes, n_ticks, n_machines)
    seed = _require_int("seed", seed)
    records: list[dict] = []
    for episode_index in range(n_episodes):
        episode_id = f"ep{episode_index:04d}"
        rng = episode_rng(seed, episode_index)
        types = [INDEX_TO_TYPE[rng.randrange(3)] for _ in range(n_machines)]
        previous_payouts: list[int] | None = None
        for tick in range(n_ticks):
            # Entire tick is scored from previous_payouts before any draw lands.
            drawn: list[int] = []
            for machine_id in range(n_machines):
                if previous_payouts is None:
                    parent_payout = None
                else:
                    parent_payout = previous_payouts[parent_slot(machine_id, n_machines)]
                drawn.append(
                    sample_payout(rng, latent_oracle_fractions(types[machine_id], parent_payout))
                )
            previous_payouts = drawn
            for machine_id, payout in enumerate(drawn):
                observed = observed_flag(seed, episode_index, tick, machine_id)
                payout_index = PAYOUT_TO_INDEX[payout]
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "slot_index": slot_index(machine_id, n_machines),
                        "machine_type": types[machine_id],
                        "machine_type_index": TYPE_TO_INDEX[types[machine_id]],
                        "payout": payout,
                        "payout_index": payout_index,
                        "observed": observed,
                        "payout_is_feature": observed,
                        "feature_payout_index": payout_index if observed else MISSING,
                    }
                )

    annotated = conditional_probabilities(records, predict_cutoff=n_ticks)
    truth: list[dict] = []
    for record in records:
        key = (record["episode_id"], record["tick"], record["machine_id"])
        fracs = annotated["latent_oracle"][key]
        if fracs is None:
            raise RuntimeError(f"oracle undefined for generated row {key}")
        truth.append(
            {
                "episode_id": record["episode_id"],
                "tick": record["tick"],
                "machine_id": record["machine_id"],
                "slot_index": record["slot_index"],
                "prob_kind": "latent_oracle",
                **prob_fields(fracs),
            }
        )
    queries: list[dict] = []
    seen_machines: set[tuple[str, int]] = set()
    for record in records:
        identity = (record["episode_id"], record["machine_id"])
        if identity in seen_machines:
            continue
        seen_machines.add(identity)
        cutoff_row = annotated["cutoff"][record["episode_id"]][record["machine_id"]]
        if cutoff_row["latent_oracle"] is None:
            raise RuntimeError("cutoff oracle undefined on a complete panel")
        queries.append(
            {
                "query_id": f"{record['episode_id']}-m{record['machine_id']:02d}",
                "episode_id": record["episode_id"],
                "machine_id": record["machine_id"],
                "slot_index": record["slot_index"],
                "machine_type": record["machine_type"],
                "machine_type_index": TYPE_TO_INDEX[record["machine_type"]],
                "cutoff": n_ticks,
                "signal": QUERY_SIGNAL,
                "signal_scope": QUERY_SIGNAL_SCOPE,
                **prob_fields(cutoff_row["latent_oracle"], prefix="latent_oracle_"),
                **prob_fields(cutoff_row["observable_history"], prefix="observable_history_"),
            }
        )
    queries.sort(key=lambda row: (row["episode_id"], row["machine_id"]))
    return {"records": records, "truth": truth, "queries": queries}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def mapping_document() -> dict:
    return {
        "payouts_ordered": list(PAYOUTS),
        "payout_to_index": {str(payout): index for payout, index in PAYOUT_TO_INDEX.items()},
        "index_to_payout": {str(index): payout for index, payout in INDEX_TO_PAYOUT.items()},
        "missing_feature_index": MISSING,
        "missing_means": "unknown, not payout 0",
        "unobserved_payout": (
            "Latent simulator field only, and only meaningful as hidden state when "
            "observed is false. Never a feature. as_feature_row drops payout and "
            "payout_index and sets feature_payout_index to -1."
        ),
        "type_to_index": dict(TYPE_TO_INDEX),
        "index_to_type": {str(index): name for index, name in INDEX_TO_TYPE.items()},
        "type_index_note": "Authoritative public mapping balanced=0, jackpot=1, steady=2. Not list order and not slot inference.",
        "slot_index": "identity with machine_id",
        "lag": LAG,
        "parent": "(machine_id - 1) mod n_machines, read at tick-1 only",
        "query_signal": QUERY_SIGNAL,
        "query_signal_scope": QUERY_SIGNAL_SCOPE,
        "legal_history": "record.tick < prediction tick; query signal is not a history feature",
        "keep": "7/10",
        "push": "3/10",
        "base_counts": {name: list(counts) for name, counts in BASE_COUNTS.items()},
        "base_denominator": BASE_DENOM,
    }


def write_outputs(
    out_dir: Path,
    seed: int,
    n_episodes: int,
    n_ticks: int,
    n_machines: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)

    def dump_jsonl(name: str, rows: list[dict]) -> tuple[Path, str]:
        path = out_dir / name
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ).encode("utf-8")
        path.write_bytes(payload)
        return path, sha256_bytes(payload)

    records_path, records_hash = dump_jsonl("records.jsonl", panel["records"])
    truth_path, truth_hash = dump_jsonl("truth.jsonl", panel["truth"])
    queries_path, queries_hash = dump_jsonl("queries.jsonl", panel["queries"])

    source_dir = Path(__file__).resolve().parent
    source_hashes = {}
    for name in ("generate.py", "test_generator.py", "README.md"):
        source_hashes[name] = sha256_file(source_dir / name)

    manifest = {
        "family": FAMILY,
        "status": STATUS,
        "research_only": True,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": REAL_ONLY_FALLBACK,
        "do_not_train": True,
        "seed": seed,
        "settings": {
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
            "lag": LAG,
            "max_records": MAX_RECORDS,
            "record_count": len(panel["records"]),
            "truth_count": len(panel["truth"]),
            "query_count": len(panel["queries"]),
            "cutoff": n_ticks,
        },
        "assumptions": ASSUMPTIONS,
        "mappings": mapping_document(),
        "source_hashes": source_hashes,
        "output_hashes": {
            "records.jsonl": records_hash,
            "truth.jsonl": truth_hash,
            "queries.jsonl": queries_hash,
        },
        "output_files": {
            "records": records_path.name,
            "truth": truth_path.name,
            "queries": queries_path.name,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="directed_lagged_crossmachine toy generator (RESEARCH_ONLY)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", required=True, help="Directory for jsonl outputs and manifest.json")
    args = parser.parse_args(argv)
    write_outputs(
        Path(args.out),
        seed=args.seed,
        n_episodes=args.episodes,
        n_ticks=args.ticks,
        n_machines=args.machines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
