# finite_order_markov

RESEARCH_ONLY toy simulator. `synthetic_exception_verified` is false. Organizer rules say to use organizer-released data. This directory is not a verified synthetic exception. Competition training and submissions stay on real organizer data (real-only fallback). Nothing here reads validation or test files, and nothing was fit to train.

## Model

Each machine is an independent order-2 Markov chain on its own payouts. Types are fixed by slot, not drawn and not estimated:

| machine_id | type |
| --- | --- |
| 0, 1, 2 | steady |
| 3, 4, 5 | balanced |
| 6, 7 | jackpot |

Payout categories, in probability-vector order, are `0, 1, 2, 5, 20`.

| payout | probability index |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 5 | 3 |
| 20 | 4 |

Index 0 is the payout-0 category. A missing tick has no payout index. `feature_payout` returns `None` when `observed` is false, including when the latent audit payout is 0.

Tick `t` is drawn from the latent-oracle row `P(Y_t | type, Y_{t-2}, Y_{t-1})`. Lags that do not exist yet (`t < 2`) are structural `None`, which is not the same row as payout 0. The integer weights live in `generate.py` (`BASE_WEIGHTS` plus lag-1 and lag-2 persist / five / jackpot bonuses). Lag 2 changes the row when lag 1 is held fixed.

Updates are synchronous and own-machine only. The probability row at tick `t` is fixed from ticks `< t` before any machine samples tick `t`. Another machine's outcome at tick `t` is not an input. Earlier ticks of other machines are not inputs either.

## Two probability rows

| field | kind | what it conditions on |
| --- | --- | --- |
| `probabilities` | latent oracle | true spins at `t-2` and `t-1`, including spins the mask hid |
| `observable_probabilities` | observable history | observations strictly before `t`; gaps are marginalized, not filled with 0 |

`records[].payout` is always the latent spin, for audit and reproducibility. If `observed` is false, that payout is not a feature. Observation is MCAR with `P(observed)` 0.92 steady, 0.84 balanced, 0.70 jackpot.

## Signal

This generator does not invent a `signal` series. On the real task, `signal` is a query field at the organizer prediction cutoff only. Synthetic rows have no `signal` key. Do not treat a generated tick index as a historical signal.

## Interface

```python
from generate import generate
out = generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)
# out["records"], out["truth"]
```

Default size is 6144 records (cap 8192). CLI:

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out sample_out
```

Writes `records.jsonl`, `truth.jsonl`, and `manifest.json` with source hashes, output hashes, seed, settings, and `RESEARCH_ONLY`.

Each episode uses its own RNG stream (`sha256` of the seed and episode index). Inside an episode the order is tick, machine, payout draw, then observation draw. A longer `n_ticks` keeps every episode's prefix. A larger `n_episodes` keeps the earlier episodes.

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator -v
```

Covers simplex and payout-index mapping, an independent copy of the kernel, reproducibility, seed change, prefix stability, future-edit causal invariance, gap versus zero imputation, latent-oracle versus mask-only probabilities, and the CLI manifest.
