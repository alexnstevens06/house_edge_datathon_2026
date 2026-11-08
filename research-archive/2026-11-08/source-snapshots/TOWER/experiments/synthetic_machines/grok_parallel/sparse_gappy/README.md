# sparse_gappy

Research-only toy generator for one synthetic machine family. It does not
read organizer train, validation, or test files, and it does not fit those
files. Organizer instructions say to use organizer-released data. No
exception that would make synthetic episodes eligible has been verified.

**Real-only fallback:** leave this directory out of training and submission
builds. A downstream trainer should keep using organizer-released data only,
unless a person explicitly opts into research-only synthetic data.

## Process

Each machine is an order-1 Markov chain on its own latent payout in
`(0, 1, 2, 5, 20)`. The distribution at tick `t` is a disclosed function of
machine type and the latent payout at `t-1` only. Tick 0 uses a disclosed
prior. Other machines, including their payouts at the same tick, are not
inputs. Tables live in `generate.py` as `PRIOR` and `TRANSITION`. They are
invented.

A second chain is an exogenous gap mask. Given the previous mask, the next
tick is revealed with a small probability (`OBSERVE_GIVEN_PREV`), independent
of payouts. Gaps come in runs. A missing tick is unknown. It is not payout 0,
and the observer does not get the hidden payout.

`truth` is the latent-oracle distribution, conditioned on the hidden previous
payout. `observable_history` is the filter that sees only revealed payouts.
Those two rows match at tick 0 and whenever the previous tick was observed.
They differ whenever the previous tick was a gap, because the filter mixes
the transition rows and the oracle uses the hidden one.

## Interface

```python
generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8) -> dict
```

The dict has `records`, `truth`, `observable_history`, and `queries`.

Record fields: `episode_id`, `tick`, `machine_id`, `machine_type`
(`steady`, `balanced`, `jackpot`), `payout`, `observed`. `machine_id` is the
slot, `0 .. n_machines-1`. On a gap, `payout` is still the latent draw so the
oracle can be checked; `observable_payout` returns `None`, and
`observation_symbol` returns `-1`. Payout 0 maps to index 0 only when
`observed` is true.

Truth fields: `episode_id`, `tick`, `machine_id`, `probabilities` of length
5 in payout order `0, 1, 2, 5, 20` (`p_0`, `p_1`, `p_2`, `p_5`, `p_20`).

Queries exist only at the prediction cutoff `n_ticks`. History ticks satisfy
`tick < cutoff`. `signal` is `null`. This family does not invent the
organizer signal (`-1`, `0`, `1` on real queries) and does not write a signal
onto history. Query probability fields are a research answer key for the
cutoff spin, not features.

Default size is `8 * 96 * 8 = 6144` records. Generation above 8192 records
raises.

Slot types repeat `(steady, steady, balanced, balanced, balanced, balanced, jackpot, jackpot)`.

## CLI

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out sample_out
```

Writes `records.jsonl`, `truth.jsonl`, `observable_history.jsonl`,
`queries.jsonl`, and `manifest.json`. The manifest carries
`eligibility: RESEARCH_ONLY`, `real_only_fallback: true`, seed, settings,
source hashes, and output hashes.

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py -v
```

Covers the payout/mask map, simplexes, reproducibility, a different seed,
future-mutation prefix invariance (and a later oracle move larger than 0.05),
no same-tick cross-machine input, sparse gap runs, missing versus zero,
hidden-payout mutations that move the oracle and not the filter, and the
pathwise oracle-versus-filter identity.

Recorded run (see `manifest.json` for hashes and the full log):

```text
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py -v
Ran 15 tests — OK
```
