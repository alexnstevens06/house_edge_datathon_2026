"""Causal synthetic reward processes with a separate latent-truth channel.

The event records are an explicit internal intermediate schema. They are not
an organizer schema. Conditional payout probabilities are returned only in
the truth document so they are not model features.

No function in this module reads a file, a validation split, or an organizer
simulator. Defaults are fixed research assumptions and are not fit to data.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

PAYOUTS: tuple[int, ...] = (0, 1, 2, 5, 20)
N_MACHINES = 8
MAX_SEED = 2**31 - 1
MAX_EPISODES = 48
MAX_TICKS = 64
MAX_EXCITATION_WEIGHT = 0.85
EXCITATION_DECAY = 0.5

# Cold-start and excited categorical weights. Both rows sum to 100.
_BURST_BASE_WEIGHTS: tuple[int, ...] = (82, 10, 5, 2, 1)
_BURST_EXCITED_WEIGHTS: tuple[int, ...] = (20, 20, 20, 25, 15)
_BURST_JUMP: dict[int, float] = {0: 0.0, 1: 0.35, 2: 0.8, 5: 2.0, 20: 2.5}

_COOLDOWN_WEIGHTS: tuple[int, ...] = (97, 2, 1, 0, 0)


class ParameterError(ValueError):
    """A generator parameter is outside the documented bounded range."""


@dataclass(frozen=True)
class RefractoryMachineSpec:
    machine_id: int
    name: str
    refractory_length: int
    ready_weights: tuple[int, ...]


def _refractory_table() -> tuple[RefractoryMachineSpec, ...]:
    quick = (50, 25, 15, 7, 3)
    steady = (68, 16, 9, 5, 2)
    deep = (80, 10, 5, 3, 2)
    rows = (
        (0, "quick", 2, quick),
        (1, "quick", 2, quick),
        (2, "steady", 4, steady),
        (3, "steady", 4, steady),
        (4, "steady", 4, steady),
        (5, "deep", 8, deep),
        (6, "deep", 8, deep),
        (7, "deep", 8, deep),
    )
    return tuple(
        RefractoryMachineSpec(
            machine_id=machine_id,
            name=name,
            refractory_length=length,
            ready_weights=weights,
        )
        for machine_id, name, length, weights in rows
    )


_REFRACTORY_SPECS = _refractory_table()


def _require_machine(machine_id: int) -> None:
    if isinstance(machine_id, bool) or not isinstance(machine_id, int):
        raise ParameterError("machine_id must be an integer in 0..7")
    if machine_id < 0 or machine_id >= N_MACHINES:
        raise ParameterError("machine_id must be an integer in 0..7")


def burst_machine_type(machine_id: int) -> str:
    """Return the burst excitation class for one machine."""
    _require_machine(machine_id)
    if machine_id < 4:
        return "spark"
    return "cascade"


def _burst_jump_scale(machine_id: int) -> float:
    if burst_machine_type(machine_id) == "spark":
        return 1.0
    return 1.5


def refractory_machine_spec(machine_id: int) -> RefractoryMachineSpec:
    """Return the fixed type, cooldown length, and ready weights for a machine."""
    _require_machine(machine_id)
    return _REFRACTORY_SPECS[machine_id]


def _simplex(weights: tuple[int, ...]) -> tuple[float, ...]:
    if len(weights) != len(PAYOUTS):
        raise ParameterError("categorical weights must match the payout support")
    if any(weight < 0 for weight in weights):
        raise ParameterError("categorical weights must be nonnegative")
    total = sum(weights)
    if total <= 0:
        raise ParameterError("categorical weights must have a positive total")
    head = [weight / total for weight in weights[:-1]]
    last = 1.0 - sum(head)
    if last < 0.0:
        raise ParameterError("categorical probability construction went negative")
    probs = tuple(head + [last])
    if any((not math.isfinite(prob)) or prob < 0.0 for prob in probs):
        raise ParameterError("categorical probabilities must be nonnegative and finite")
    return probs


def _mix(base: tuple[float, ...], excited: tuple[float, ...], weight: float) -> tuple[float, ...]:
    if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ParameterError("mixture weight must lie in [0, 1]")
    mixed_head = [
        (1.0 - weight) * left + weight * right
        for left, right in zip(base[:-1], excited[:-1])
    ]
    last = 1.0 - sum(mixed_head)
    if last < -1e-12:
        raise ParameterError("mixture probability construction went negative")
    if last < 0.0:
        last = 0.0
    probs = tuple(mixed_head + [last])
    if any((not math.isfinite(prob)) or prob < 0.0 for prob in probs):
        raise ParameterError("mixture probabilities must be nonnegative and finite")
    return probs


_BURST_BASE = _simplex(_BURST_BASE_WEIGHTS)
_BURST_EXCITED = _simplex(_BURST_EXCITED_WEIGHTS)
_COOLDOWN_PROBS = _simplex(_COOLDOWN_WEIGHTS)
_READY_PROBS = tuple(_simplex(spec.ready_weights) for spec in _REFRACTORY_SPECS)


def _excitation_weight(excitation: float) -> float:
    if not math.isfinite(excitation) or excitation < 0.0:
        raise ParameterError("excitation must be a nonnegative finite value")
    if excitation == 0.0:
        return 0.0
    weight = 1.0 - math.exp(-excitation)
    if weight > MAX_EXCITATION_WEIGHT:
        return MAX_EXCITATION_WEIGHT
    return weight


def burst_conditional_probabilities(machine_id: int, excitation: float) -> tuple[float, ...]:
    """Payout probabilities at this tick given excitation entering the tick.

    ``machine_id`` selects a valid machine. The conditional distribution itself
    depends on excitation, which the type-specific jump updates after the draw.
    """
    _require_machine(machine_id)
    weight = _excitation_weight(excitation)
    if weight == 0.0:
        return _BURST_BASE
    return _mix(_BURST_BASE, _BURST_EXCITED, weight)


def burst_next_excitation(machine_id: int, excitation: float, payout: int) -> float:
    """Excitation entering the next tick after observing this tick's payout."""
    _require_machine(machine_id)
    if not math.isfinite(excitation) or excitation < 0.0:
        raise ParameterError("excitation must be a nonnegative finite value")
    if payout not in _BURST_JUMP:
        raise ParameterError("payout is outside the synthetic support")
    return EXCITATION_DECAY * excitation + _burst_jump_scale(machine_id) * _BURST_JUMP[payout]


def refractory_conditional_probabilities(machine_id: int, cooldown_remaining: int) -> tuple[float, ...]:
    """Payout probabilities given the cooldown still remaining at this tick."""
    spec = refractory_machine_spec(machine_id)
    if isinstance(cooldown_remaining, bool) or not isinstance(cooldown_remaining, int):
        raise ParameterError("cooldown_remaining must be a nonnegative integer")
    if cooldown_remaining < 0:
        raise ParameterError("cooldown_remaining must be a nonnegative integer")
    if cooldown_remaining == 0:
        return _READY_PROBS[spec.machine_id]
    return _COOLDOWN_PROBS


def refractory_next_cooldown(machine_id: int, cooldown_remaining: int, payout: int) -> int:
    """Cooldown entering the next tick after this tick's payout."""
    spec = refractory_machine_spec(machine_id)
    if isinstance(cooldown_remaining, bool) or not isinstance(cooldown_remaining, int):
        raise ParameterError("cooldown_remaining must be a nonnegative integer")
    if cooldown_remaining < 0:
        raise ParameterError("cooldown_remaining must be a nonnegative integer")
    if payout not in PAYOUTS:
        raise ParameterError("payout is outside the synthetic support")
    if payout > 0:
        return spec.refractory_length
    if cooldown_remaining > 0:
        return cooldown_remaining - 1
    return 0


def _require_int(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} must be an integer in {low}..{high}")
    if value < low or value > high:
        raise ParameterError(f"{name} must be an integer in {low}..{high}")
    return value


def _validate_config(config: Mapping[str, object]) -> tuple[int, int, int, float]:
    required = {"seed", "n_episodes", "n_ticks", "observe_probability"}
    if set(config) != required:
        raise ParameterError("config must contain seed, n_episodes, n_ticks, and observe_probability")
    seed = _require_int("seed", config["seed"], 0, MAX_SEED)
    n_episodes = _require_int("n_episodes", config["n_episodes"], 1, MAX_EPISODES)
    n_ticks = _require_int("n_ticks", config["n_ticks"], 1, MAX_TICKS)
    observe = config["observe_probability"]
    if isinstance(observe, bool) or not isinstance(observe, (int, float)):
        raise ParameterError("observe_probability must be a finite number in (0, 1]")
    observe_probability = float(observe)
    if not math.isfinite(observe_probability) or observe_probability <= 0.0 or observe_probability > 1.0:
        raise ParameterError("observe_probability must be a finite number in (0, 1]")
    return seed, n_episodes, n_ticks, observe_probability


def _sample_payout(rng: random.Random, probabilities: tuple[float, ...]) -> int:
    draw = rng.random()
    cumulative = 0.0
    for payout, probability in zip(PAYOUTS, probabilities):
        cumulative += probability
        if draw < cumulative:
            return payout
    return PAYOUTS[-1]


def _truth_document(
    process: str,
    seed: int,
    n_episodes: int,
    n_ticks: int,
    observe_probability: float,
    assumptions: list[str],
) -> dict[str, object]:
    return {
        "schema_id": "internal_intermediate_latent_truth_v1",
        "schema_status": "explicit_internal_intermediate_not_organizer_schema",
        "research_only": True,
        "eligibility": "unresolved",
        "fitted_to_real_data": False,
        "real_data_used": False,
        "process": process,
        "seed": seed,
        "n_episodes": n_episodes,
        "n_ticks": n_ticks,
        "n_machines": N_MACHINES,
        "payout_values": list(PAYOUTS),
        "probability_index_order": list(PAYOUTS),
        "observe_probability": observe_probability,
        "assumptions": assumptions,
        "episodes": [],
    }


def _generate(
    config: Mapping[str, object],
    process: str,
    assumptions: list[str],
    initial_state: Callable[[int], dict[str, float | int | str]],
    probabilities_for: Callable[[int, dict[str, float | int | str]], tuple[float, ...]],
    advance: Callable[[int, dict[str, float | int | str], int], dict[str, float | int | str]],
    machine_type_for: Callable[[int], str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed, n_episodes, n_ticks, observe_probability = _validate_config(config)
    rng = random.Random(seed)
    events: list[dict[str, object]] = []
    truth = _truth_document(process, seed, n_episodes, n_ticks, observe_probability, assumptions)
    episodes: list[dict[str, object]] = []
    for episode_id in range(n_episodes):
        states = [initial_state(machine_id) for machine_id in range(N_MACHINES)]
        records: list[dict[str, object]] = []
        for tick in range(n_ticks):
            for machine_id in range(N_MACHINES):
                state = states[machine_id]
                probabilities = probabilities_for(machine_id, state)
                true_payout = _sample_payout(rng, probabilities)
                observed = rng.random() < observe_probability
                machine_type = machine_type_for(machine_id)
                events.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "machine_type": machine_type,
                        "payout": true_payout if observed else None,
                        "observed": observed,
                    }
                )
                records.append(
                    {
                        "episode_id": episode_id,
                        "tick": tick,
                        "machine_id": machine_id,
                        "machine_type": machine_type,
                        "latent_state": dict(state),
                        "conditional_payout_probabilities": list(probabilities),
                        "true_payout": true_payout,
                        "observed": observed,
                    }
                )
                states[machine_id] = advance(machine_id, state, true_payout)
        episodes.append({"episode_id": episode_id, "records": records})
    truth["episodes"] = episodes
    return events, truth


def _burst_initial(machine_id: int) -> dict[str, float | int | str]:
    return {"excitation": 0.0, "jump_scale": _burst_jump_scale(machine_id)}


def _burst_probabilities(machine_id: int, state: dict[str, float | int | str]) -> tuple[float, ...]:
    excitation = state["excitation"]
    if not isinstance(excitation, float):
        raise ParameterError("burst excitation state is corrupt")
    return burst_conditional_probabilities(machine_id, excitation)


def _burst_advance(
    machine_id: int, state: dict[str, float | int | str], payout: int
) -> dict[str, float | int | str]:
    excitation = state["excitation"]
    if not isinstance(excitation, float):
        raise ParameterError("burst excitation state is corrupt")
    return {
        "excitation": burst_next_excitation(machine_id, excitation, payout),
        "jump_scale": _burst_jump_scale(machine_id),
    }


def _refractory_initial(machine_id: int) -> dict[str, float | int | str]:
    spec = refractory_machine_spec(machine_id)
    return {"cooldown_remaining": 0, "refractory_length": spec.refractory_length}


def _refractory_probabilities(
    machine_id: int, state: dict[str, float | int | str]
) -> tuple[float, ...]:
    cooldown = state["cooldown_remaining"]
    if not isinstance(cooldown, int):
        raise ParameterError("refractory cooldown state is corrupt")
    return refractory_conditional_probabilities(machine_id, cooldown)


def _refractory_advance(
    machine_id: int, state: dict[str, float | int | str], payout: int
) -> dict[str, float | int | str]:
    cooldown = state["cooldown_remaining"]
    if not isinstance(cooldown, int):
        raise ParameterError("refractory cooldown state is corrupt")
    spec = refractory_machine_spec(machine_id)
    return {
        "cooldown_remaining": refractory_next_cooldown(machine_id, cooldown, payout),
        "refractory_length": spec.refractory_length,
    }


_BURST_ASSUMPTIONS = [
    "Research-only synthetic process. Eligibility for the datathon is unresolved.",
    "Not fit to organizer data, validation files, or a private simulator.",
    "Eight machines evolve independently given their own excitation.",
    "Machines 0-3 are spark (jump scale 1) and machines 4-7 are cascade (jump scale 1.5).",
    "The conditional distribution at tick t uses excitation entering tick t, before the draw.",
    "After the draw, excitation decays by 0.5 and adds a payout-dependent jump.",
    "Observation is an independent Bernoulli trial and does not change the latent payout.",
    "An unobserved payout is missing. It is not written as zero.",
    "Payout support is exactly [0, 1, 2, 5, 20].",
]

_REFRACTORY_ASSUMPTIONS = [
    "Research-only synthetic process. Eligibility for the datathon is unresolved.",
    "Not fit to organizer data, validation files, or a private simulator.",
    "Machines 0-1 are quick (cooldown 2), 2-4 are steady (cooldown 4), and 5-7 are deep (cooldown 8).",
    "Each type has its own ready categorical distribution.",
    "A positive payout sets that type's cooldown. Later ticks use the shared cooldown distribution until it reaches 0.",
    "The distribution at tick t uses the cooldown entering tick t, before the draw.",
    "Observation is an independent Bernoulli trial and does not change the latent payout.",
    "An unobserved payout is missing. It is not written as zero.",
    "Payout support is exactly [0, 1, 2, 5, 20].",
]


def generate_burst(config: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate bursty self-exciting categorical rewards and latent truth."""
    return _generate(
        config,
        "burst_self_exciting",
        _BURST_ASSUMPTIONS,
        _burst_initial,
        _burst_probabilities,
        _burst_advance,
        burst_machine_type,
    )


def generate_refractory(config: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate refractory renewal rewards with machine-type variation and latent truth."""
    return _generate(
        config,
        "refractory_renewal",
        _REFRACTORY_ASSUMPTIONS,
        _refractory_initial,
        _refractory_probabilities,
        _refractory_advance,
        lambda machine_id: refractory_machine_spec(machine_id).name,
    )
