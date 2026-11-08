# abrupt_changepoints

Research-only toy generator for one synthetic slot family. It does not train a model.

## Eligibility

`RESEARCH_ONLY`. `kit/RULES.md` says: use organizer-released data. This repo has no verified exception that allows synthetic rows in a submission. `synthetic_exception_verified` is false. Real-only fallback: fit and submit from organizer-released train data only. Do not read validation or test answers to build this generator. Nothing here was fit to those files.

## What it generates

`generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)` returns a dict with:

- `records`: one row per episode, tick, and machine. Fields are `episode_id`, `tick`, `machine_id` (slot, 0..7 by default), `machine_type` (`steady`, `balanced`, or `jackpot`), `payout` in `{0, 1, 2, 5, 20}`, `observed` bool.
- `truth`: latent-oracle probabilities for the same grid. `probabilities` has length 5 in payout order `0, 1, 2, 5, 20` (`p_0`, `p_1`, `p_2`, `p_5`, `p_20`).
- `queries`: one query per machine at the shared cutoff `n_ticks - 1`. `signal` is in `{-1, 0, 1}` and is not stored on history rows.

Default size is 6144 records. The generator refuses more than 8192.

## Assumptions (invented)

No frequency here is estimated from the contest logs.

Each machine makes one abrupt flip between two fixed categorical regimes. Even slots start in regime 0. Odd slots start in regime 1, so the two `steady` machines begin opposite each other. An episode draws one shared anchor tick in `[32, 64)` and a per-machine offset in `{-8, -4, 0, 0, 4, 8}`. The flip applies at `anchor + offset` and stays there. The schedule ignores outcomes and does not depend on `n_ticks`.

Regime tables, ordered `(0, 1, 2, 5, 20)`:

| type | regime 0 | regime 1 |
| --- | --- | --- |
| steady | 0.62, 0.28, 0.07, 0.02, 0.01 | 0.48, 0.30, 0.14, 0.06, 0.02 |
| balanced | 0.50, 0.22, 0.15, 0.09, 0.04 | 0.35, 0.20, 0.22, 0.15, 0.08 |
| jackpot | 0.70, 0.15, 0.08, 0.05, 0.02 | 0.55, 0.12, 0.10, 0.08, 0.15 |

Slot types for machine ids `0..7` are steady, steady, balanced, balanced, balanced, jackpot, jackpot, jackpot. Further slots cycle that list.

A tick is missing when `(tick + 3 * machine_id) % 11 == 0`. Other ticks are kept with invented probabilities 0.92 (steady), 0.84 (balanced), and 0.75 (jackpot). The log still stores the latent payout on a missing row. Feature code must treat that cell as unknown.

## Causality and masks

Tick `t` is drawn from the regime schedule alone. The sampler does not read any payout at tick `t` or later, and it does not read another machine's payout at tick `t`. Cell noise uses its own seed stream, so a longer `n_ticks` leaves the shorter prefix unchanged.

`encode_observation`:

| cell | payout index | mask | one-hot |
| --- | --- | --- | --- |
| missing | `None` | 0 | `(0, 0, 0, 0, 0)` |
| payout 0 | 0 | 1 | `(1, 0, 0, 0, 0)` |
| payout 1 | 1 | 1 | `(0, 1, 0, 0, 0)` |
| payout 2 | 2 | 1 | `(0, 0, 1, 0, 0)` |
| payout 5 | 3 | 1 | `(0, 0, 0, 1, 0)` |
| payout 20 | 4 | 1 | `(0, 0, 0, 0, 1)` |

`history_features` keeps ticks strictly below the cutoff, for every machine. `signal_at` returns the query signal only when the tick equals that machine's cutoff.

## Two probability objects

Latent-oracle probabilities are `truth` / `probabilities_at`. They jump once, from one catalog row to the other.

Observable-history probabilities are `observable_history_probabilities`. They count this machine's observed payouts at ticks `< t` only. Missing rows are skipped. If nothing was observed, the result is the flat prior `(0.2, 0.2, 0.2, 0.2, 0.2)`. That prior is not a regime row. Do not train a submission model to match the oracle unless organizers allow this synthetic family.

## CLI

```sh
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 2 --ticks 32 --machines 8 --out sample_out
```

Writes `records.jsonl`, `truth.jsonl`, and `manifest.json` with source hashes, output hashes, seed, settings, and `RESEARCH_ONLY`.

## Tests

```sh
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py
```

Covers the simplex and payout-index map, reproducibility, a different seed, the regime-catalog invariant (including an abrupt flip and a shared anchor), latent oracle versus observable history, missing versus payout 0, future-mutation invariance, prefix agreement between short and long runs, and cutoff-only signals.
