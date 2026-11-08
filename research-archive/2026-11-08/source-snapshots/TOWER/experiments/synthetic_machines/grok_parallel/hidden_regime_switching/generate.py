"""Hidden regime-switching toy machines (RESEARCH_ONLY).

Each machine carries a latent regime in {dull, normal, hot}. The regime is a
Markov chain whose transition depends on that machine's own observation at the
current tick. The payout is drawn from a hand-written categorical table
indexed by the observable machine type and the latent regime.

Assumptions (invented, not fit to organizer data):
- Three regimes, sticky when the tick is missing, pulled by the observed payout
  when the tick is observed.
- Missing ticks are unknown. They use a dedicated transition and do not enter
  the emission likelihood. They are not payout 0.
- Machines are independent. The state at tick t is fixed before any payout at
  t is drawn, and a transition reads only that machine's own tick-t observation.
- Oracle probabilities are the emission row of the true regime.
  Observable-history probabilities are the regime-belief mixture from
  observations at ticks < t only.
- No query signal is synthesized on history. Queries exist only at the
  prediction cutoff and carry a null signal.

Eligibility: RESEARCH_ONLY. Organizer instructions require organizer-released
data, and no synthetic exception is verified. Keep a real-only training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

FAMILY = "hidden_regime_switching"
ELIGIBILITY = "RESEARCH_ONLY"
MAX_RECORDS = 8192

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_IDX = {payout: index for index, payout in enumerate(PAYOUTS)}
PAYOUT_IDX_TO_VALUE = {index: payout for index, payout in enumerate(PAYOUTS)}
MACHINE_TYPES = ("balanced", "jackpot", "steady")
TYPE_TO_IDX = {name: index for index, name in enumerate(MACHINE_TYPES)}
REGIME_NAMES = ("dull", "normal", "hot")
N_REGIMES = 3
MISSING = -1  # grid encoding of an unobserved tick; distinct from payout index 0
DENOM = 16

# Dense symbol: 0 = missing, 1..5 = payout index + 1.
# Sparse symbol: 0..4 = payout index; missing ticks are omitted.
# obs_mask: 1 = observed, 0 = missing.


def _unit(numerators: tuple[int, ...], denom: int = DENOM) -> tuple[float, ...]:
    if sum(numerators) != denom:
        raise ValueError(f"numerators {numerators} do not sum to {denom}")
    if any(part < 0 for part in numerators):
        raise ValueError(numerators)
    return tuple(part / denom for part in numerators)


# EMISSION_NUMERATORS[type_idx][regime][payout_idx], rows sum to DENOM.
# Machine type changes the payout table. Values are toy fractions.
EMISSION_NUMERATORS = (
    (  # balanced
        (5, 4, 3, 2, 2),
        (3, 4, 4, 3, 2),
        (2, 2, 3, 4, 5),
    ),
    (  # jackpot
        (6, 4, 3, 2, 1),
        (3, 3, 3, 3, 4),
        (1, 1, 2, 4, 8),
    ),
    (  # steady
        (10, 3, 1, 1, 1),
        (6, 5, 3, 1, 1),
        (3, 4, 4, 3, 2),
    ),
)
EMISSIONS = tuple(
    tuple(_unit(row) for row in type_rows) for type_rows in EMISSION_NUMERATORS
)

# Shared across types: the observation, not the type, selects the kernel.
# TRANS_OBS_NUMERATORS[payout_idx][regime] = distribution of the next regime.
# TRANS_MISS_NUMERATORS[regime] is the kernel used when the tick is unobserved.
TRANS_OBS_NUMERATORS = (
    ((14, 2, 0), (10, 5, 1), (8, 6, 2)),  # observed payout 0 -> dull
    ((8, 6, 2), (4, 10, 2), (2, 6, 8)),  # payout 1
    ((4, 8, 4), (3, 10, 3), (2, 6, 8)),  # payout 2
    ((2, 6, 8), (2, 6, 8), (1, 4, 11)),  # payout 5
    ((1, 3, 12), (1, 4, 11), (0, 2, 14)),  # payout 20 -> hot
)
TRANS_MISS_NUMERATORS = (
    (12, 3, 1),
    (2, 12, 2),
    (1, 3, 12),
)
TRANS_OBS = tuple(
    tuple(_unit(row) for row in payout_rows) for payout_rows in TRANS_OBS_NUMERATORS
)
TRANS_MISS = tuple(_unit(row) for row in TRANS_MISS_NUMERATORS)
PRIOR_NUMERATORS = (8, 5, 3)
PRIOR = _unit(PRIOR_NUMERATORS)

ASSUMPTIONS = (
    "Invented toy probabilities with denominator 16. No real-data fit.",
    "Latent regimes are dull, normal, hot. Records omit the regime.",
    "probabilities on truth rows are the latent-oracle emission of the true regime.",
    "observable_probabilities marginalize a filter belief built from ticks < t.",
    "A missing tick uses TRANS_MISS and skips the emission update.",
    "An observed 0 uses TRANS_OBS[0]. Those kernels differ.",
    "No contemporaneous cross-machine coupling.",
    "History has no signal field. Queries sit at cutoff=n_ticks with signal null.",
    "RESEARCH_ONLY: real-only training remains the fallback.",
)

MAPPINGS = {
    "payouts": list(PAYOUTS),
    "payout_to_idx": {str(payout): index for payout, index in PAYOUT_TO_IDX.items()},
    "idx_to_payout": {str(index): payout for index, payout in PAYOUT_IDX_TO_VALUE.items()},
    "machine_types": list(MACHINE_TYPES),
    "type_to_idx": dict(TYPE_TO_IDX),
    "regime_names": list(REGIME_NAMES),
    "slot": "machine_id is the slot index, range 0 .. n_machines-1",
    "query_index": "query order inside an episode is slot order; query_id suffix :s{slot}",
    "missing_grid_value": MISSING,
    "dense_symbol": "0 missing, 1..5 = payout_index + 1",
    "sparse_symbol": "0..4 payout_index; missing ticks omitted (None)",
    "obs_mask": "1 observed, 0 missing",
    "probability_column_order": list(PAYOUTS),
    "probabilities": "latent_oracle",
    "observable_probabilities": "filter predictive distribution from ticks < t",
    "signal": "null on cutoff queries only; absent from history records",
}


def obs_mask(observed: bool) -> int:
    return 1 if observed else 0


def dense_symbol(observed: bool, payout: int | None) -> int:
    """Kit dense code: missing -> 0, observed payout index i -> i + 1."""
    if not observed:
        return 0
    return PAYOUT_TO_IDX[payout] + 1


def sparse_symbol(observed: bool, payout: int | None) -> int | None:
    """Kit sparse code: observed payout index, or None when the tick is absent."""
    if not observed:
        return None
    return PAYOUT_TO_IDX[payout]


def oracle_emission(machine_type: str, regime: int) -> tuple[float, ...]:
    return EMISSIONS[TYPE_TO_IDX[machine_type]][regime]


def sample_categorical(probs: tuple[float, ...] | list[float], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    last = len(probs) - 1
    for index, prob in enumerate(probs):
        cumulative += prob
        if draw < cumulative or index == last:
            return index
    return last


def _mix_seed(*parts: object) -> int:
    digest = hashlib.sha256()
    digest.update(b"hidden_regime_switching")
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def _rng(*parts: object) -> random.Random:
    return random.Random(_mix_seed(*parts))


def scheduled_observed(seed: int, episode_index: int, machine_id: int, tick: int) -> bool:
    """Deterministic gaps. About one quarter of ticks are unobserved."""
    code = tick * 3 + machine_id * 5 + episode_index * 7 + (seed % 997)
    return code % 4 != 0


def _normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0.0:
        raise RuntimeError("belief collapsed")
    return [weight / total for weight in weights]


def _mix_emission(belief: list[float], type_idx: int) -> list[float]:
    mixed = [0.0] * len(PAYOUTS)
    for regime, weight in enumerate(belief):
        row = EMISSIONS[type_idx][regime]
        for index, prob in enumerate(row):
            mixed[index] += weight * prob
    return mixed


def _apply_kernel(belief: list[float], kernel: tuple[tuple[float, ...], ...]) -> list[float]:
    nxt = [0.0] * N_REGIMES
    for regime, weight in enumerate(belief):
        for nxt_regime, prob in enumerate(kernel[regime]):
            nxt[nxt_regime] += weight * prob
    return _normalize(nxt)


def _kernel_for(observed: bool, payout_idx: int | None) -> tuple[tuple[float, ...], ...]:
    if observed:
        if payout_idx is None:
            raise ValueError("observed tick requires a payout index")
        return TRANS_OBS[payout_idx]
    if payout_idx is not None:
        raise ValueError("missing tick has no payout; refusing to treat it as 0")
    return TRANS_MISS


def machine_type_for(seed: int, episode_index: int, machine_id: int) -> str:
    offset = (seed + episode_index) % len(MACHINE_TYPES)
    return MACHINE_TYPES[(machine_id + offset) % len(MACHINE_TYPES)]


def _parse_override(value: tuple[bool, int | None]) -> tuple[bool, int | None]:
    observed, payout_idx = value
    if observed:
        if payout_idx not in PAYOUT_TO_IDX.values():
            raise ValueError(f"payout index out of range: {payout_idx}")
        return True, int(payout_idx)
    return False, None


def rollout_machine(
    seed: int,
    episode_index: int,
    machine_id: int,
    n_ticks: int,
    machine_type: str | None = None,
    tick_overrides: dict[int, tuple[bool, int | None]] | None = None,
) -> dict:
    """Roll one machine forward.

    tick_overrides maps tick -> (observed, payout_idx or None) and replaces the
    sampled observation. Transition noise stays tied to (seed, episode, machine, tick).
    """
    if machine_type is None:
        machine_type = machine_type_for(seed, episode_index, machine_id)
    type_idx = TYPE_TO_IDX[machine_type]
    overrides = tick_overrides or {}
    regime = sample_categorical(PRIOR, _rng(seed, episode_index, machine_id, "init"))
    belief = list(PRIOR)
    steps: list[dict] = []
    for tick in range(n_ticks):
        oracle = list(EMISSIONS[type_idx][regime])
        observable = _mix_emission(belief, type_idx)
        if tick in overrides:
            observed, payout_idx = _parse_override(overrides[tick])
        else:
            payout_idx = sample_categorical(oracle, _rng(seed, episode_index, machine_id, tick, "payout"))
            observed = scheduled_observed(seed, episode_index, machine_id, tick)
            if not observed:
                payout_idx = None
        steps.append(
            {
                "tick": tick,
                "machine_type": machine_type,
                "regime": regime,
                "probabilities": oracle,
                "observable_probabilities": observable,
                "observed": observed,
                "payout_idx": payout_idx,
                "payout": None if payout_idx is None else PAYOUTS[payout_idx],
            }
        )
        kernel = _kernel_for(observed, payout_idx)
        if observed:
            posterior = _normalize(
                [belief[reg] * EMISSIONS[type_idx][reg][payout_idx] for reg in range(N_REGIMES)]
            )
        else:
            posterior = list(belief)
        belief = _apply_kernel(posterior, kernel)
        regime = sample_categorical(kernel[regime], _rng(seed, episode_index, machine_id, tick, "trans"))
    cutoff_oracle = list(EMISSIONS[type_idx][regime])
    cutoff_observable = _mix_emission(belief, type_idx)
    return {
        "machine_id": machine_id,
        "machine_type": machine_type,
        "steps": steps,
        "cutoff_regime": regime,
        "cutoff_probabilities": cutoff_oracle,
        "cutoff_observable_probabilities": cutoff_observable,
    }


def _record_count(n_episodes: int, n_ticks: int, n_machines: int) -> int:
    return n_episodes * n_ticks * n_machines


def generate(
    seed: int = 0,
    n_episodes: int = 8,
    n_ticks: int = 96,
    n_machines: int = 8,
    *,
    observation_overrides: dict[tuple[int, int, int], tuple[bool, int | None]] | None = None,
) -> dict:
    """Return records, truth, and cutoff queries.

    observation_overrides is a test hook. Keys are
    (episode_index, machine_id, tick). Values are (observed, payout_idx or None).
    """
    if n_episodes < 1 or n_ticks < 1 or not 1 <= n_machines <= 8:
        raise ValueError("n_episodes >= 1, n_ticks >= 1, 1 <= n_machines <= 8")
    n_records = _record_count(n_episodes, n_ticks, n_machines)
    if n_records > MAX_RECORDS:
        raise ValueError(f"{n_records} records exceeds cap {MAX_RECORDS}")
    overrides = observation_overrides or {}
    records: list[dict] = []
    truth: list[dict] = []
    queries: list[dict] = []
    for episode_index in range(n_episodes):
        episode_id = f"hrs-{seed}-{episode_index:04d}"
        per_machine_overrides: dict[int, dict[int, tuple[bool, int | None]]] = {}
        for episode_i, machine_id, tick in overrides:
            if episode_i != episode_index:
                continue
            per_machine_overrides.setdefault(machine_id, {})[tick] = overrides[(episode_i, machine_id, tick)]
        for machine_id in range(n_machines):
            path = rollout_machine(
                seed,
                episode_index,
                machine_id,
                n_ticks,
                tick_overrides=per_machine_overrides.get(machine_id),
            )
            for step in path["steps"]:
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": step["tick"],
                        "machine_id": machine_id,
                        "slot": machine_id,
                        "machine_type": step["machine_type"],
                        "payout": step["payout"],
                        "observed": step["observed"],
                        "obs_mask": obs_mask(step["observed"]),
                    }
                )
                truth.append(
                    {
                        "episode_id": episode_id,
                        "tick": step["tick"],
                        "machine_id": machine_id,
                        "slot": machine_id,
                        "machine_type": step["machine_type"],
                        "role": "history",
                        "regime": step["regime"],
                        "regime_name": REGIME_NAMES[step["regime"]],
                        "probabilities": step["probabilities"],
                        "probability_kind": "latent_oracle",
                        "observable_probabilities": step["observable_probabilities"],
                        "observable_kind": "history_filter",
                    }
                )
            queries.append(
                {
                    "query_id": f"{episode_id}:s{machine_id}",
                    "episode_id": episode_id,
                    "machine_id": machine_id,
                    "slot": machine_id,
                    "machine_type": path["machine_type"],
                    "cutoff": n_ticks,
                    "signal": None,
                    "signal_scope": "prediction_cutoff_only",
                }
            )
            truth.append(
                {
                    "episode_id": episode_id,
                    "tick": n_ticks,
                    "machine_id": machine_id,
                    "slot": machine_id,
                    "machine_type": path["machine_type"],
                    "role": "cutoff",
                    "regime": path["cutoff_regime"],
                    "regime_name": REGIME_NAMES[path["cutoff_regime"]],
                    "probabilities": path["cutoff_probabilities"],
                    "probability_kind": "latent_oracle",
                    "observable_probabilities": path["cutoff_observable_probabilities"],
                    "observable_kind": "history_filter",
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
        "assumptions": list(ASSUMPTIONS),
        "settings": {
            "seed": seed,
            "n_episodes": n_episodes,
            "n_ticks": n_ticks,
            "n_machines": n_machines,
        },
    }


def history_grid(records: list[dict], n_ticks: int, n_machines: int) -> list[list[int]]:
    """Payout index grid. Unobserved cells are MISSING (-1), never payout index 0."""
    grid = [[MISSING for _ in range(n_machines)] for _ in range(n_ticks)]
    for row in records:
        tick = row["tick"]
        slot = row["machine_id"]
        if tick >= n_ticks:
            continue
        if row["observed"]:
            grid[tick][slot] = PAYOUT_TO_IDX[row["payout"]]
        else:
            grid[tick][slot] = MISSING
    return grid


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def write_dataset(out_dir: Path, seed: int, n_episodes: int, n_ticks: int, n_machines: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
    records_hash = _write_jsonl(out_dir / "records.jsonl", data["records"])
    truth_hash = _write_jsonl(out_dir / "truth.jsonl", data["truth"])
    queries_hash = _write_jsonl(out_dir / "queries.jsonl", data["queries"])
    source = Path(__file__).resolve()
    manifest = {
        "family": FAMILY,
        "eligibility": ELIGIBILITY,
        "real_only_fallback": True,
        "eligibility_note": (
            "Organizer instructions require organizer-released data. "
            "No synthetic exception is verified. Preserve a real-only training path."
        ),
        "seed": seed,
        "settings": data["settings"],
        "n_records": len(data["records"]),
        "n_truth": len(data["truth"]),
        "n_queries": len(data["queries"]),
        "assumptions": data["assumptions"],
        "mappings": MAPPINGS,
        "source_hashes": {source.name: _sha256_file(source)},
        "output_hashes": {
            "records.jsonl": records_hash,
            "truth.jsonl": truth_hash,
            "queries.jsonl": queries_hash,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate hidden-regime-switching toy machines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = write_dataset(args.out, args.seed, args.episodes, args.ticks, args.machines)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_records": manifest["n_records"],
                "eligibility": manifest["eligibility"],
                "output_hashes": manifest["output_hashes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
