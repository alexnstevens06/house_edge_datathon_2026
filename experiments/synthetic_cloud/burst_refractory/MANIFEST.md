# Synthetic burst / refractory generator

Research only. Datathon eligibility is unresolved. This is an explicit internal intermediate schema, not an organizer schema. Nothing here was fit to organizer data, and nothing here claims a better real validation score.

## What this is

Standalone stdlib generators for eight machines (`machine_id` 0..7) and payouts exactly `[0, 1, 2, 5, 20]`.

- `burst`: self-exciting categorical rewards. Spark machines 0–3 use jump scale 1. Cascade machines 4–7 use jump scale 1.5. Excitation entering tick `t` sets the conditional distribution. After the draw it decays by 0.5 and adds a payout-dependent jump.
- `refractory`: cooldown renewal with three types. Quick machines 0–1 cool down for 2 ticks, steady machines 2–4 for 4, deep machines 5–7 for 8. A positive payout starts that cooldown. Cooldown ticks use a shared distribution with mass 0.97 on 0.

Both processes are strictly causal: tick `t` depends only on latent state entering `t`, and that state is a function of earlier payouts. Machines are independent given their own state. One local `random.Random(seed)` draws the whole run. The global RNG is not seeded or drawn.

Observation is an independent Bernoulli trial. `observed: false` stores JSON `null`. An observed zero stays `0`. Missing is not coded as zero.

## Files

Event JSONL fields, and only these fields: `episode_id`, `tick`, `machine_id`, `machine_type`, `payout`, `observed`.

Latent truth is a separate JSON document, `internal_intermediate_latent_truth_v1`. It holds conditional payout probabilities, latent state, and `true_payout`. Those probabilities are not event fields and are not model features.

Tiny samples, seed `20261108`, 1 episode, 4 ticks, observe probability 0.5 (32 rows each):

- `samples/burst/` — 17 missing payouts, 9 observed zeros
- `samples/refractory/` — 17 missing payouts, 12 observed zeros

## Bounds

API and CLI: seed `0 .. 2**31-1`, episodes `1..48`, ticks `1..64`, observe probability in `(0, 1]`. Output paths cannot contain `..`. The CLI writes only `events.jsonl` and `truth.json`.

## Assumptions

Fixed weights, decay, jumps, cooldown lengths, and the Bernoulli observation model are research choices. They were not estimated from a datathon file. The generator does not read validation files, test files, or a private simulator. There is no `signal` input.

## Tests

`PYTHONPATH=experiments/synthetic_cloud python3 -m unittest burst_refractory.tests.test_generators`

Result: 18 tests, OK (about 0.04s). Covers exact seed reproducibility, nonnegative finite normalized probabilities, bounded API/CLI output, known one-tick lag in the truth channel, missing versus observed zero, nontrivial non-constant dynamics, no global RNG mutation, and source that does not open files for reading.

Fixed-seed margins, bounds left as written:

- Burst seed 2026, 20×32: P(next payout > 0 | payout ≥ 5) = 0.704 on 873 pairs. Unconditional next-positive rate = 0.448. Checks: `≥ 0.45` and `≥ unconditional + 0.12`.
- Refractory seed 2026, 16×40: P(next payout = 0 | previous payout > 0) = 0.979 on 800 pairs. Ready-to-ready zero rate = 0.695 on 1451 pairs. Checks: `≥ 0.90` and `≤ 0.85`.

## Tooling

gitleaks was not on PATH and was not installed. No secret scan from that tool is claimed.

No network calls, paid APIs, or non-stdlib installs were used.
