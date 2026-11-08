# shared_latent_crossmachine

Research-only toy generator. One hidden state is shared by every machine in an episode. Machines load that state with opposite signs, and the state moves only from earlier observed spins.

**Status: `RESEARCH_ONLY`.** Eligibility is unresolved. `kit/RULES.md` says "Use organizer-released data." No synthetic exception is verified. **Real-only fallback:** train and submit from organizer-released data only. Do not fit or submit a model on this generator. Nothing here was estimated from train, validation, or test.

## Generate

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out /tmp/slx_out
```

`generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)` returns `records`, `truth`, and `queries`. The default grid is 6144 records (cap 8192).

`--out` writes `records.jsonl`, `truth.jsonl`, `queries.jsonl`, and `manifest.json` with seed, settings, `RESEARCH_ONLY`, and SHA-256 of `generate.py` and of those outputs.

## Contract

History rows (`records`) and matching `truth` rows:

| Field | Meaning |
| --- | --- |
| `episode_id`, `tick`, `machine_id` | `machine_id` is `0 .. n_machines-1` |
| `slot` | Same integer as `machine_id` |
| `machine_type` | `steady`, `balanced`, or `jackpot` |
| `payout` | One of `0, 1, 2, 5, 20` |
| `payout_index` | `0,1,2,5,20` → `0,1,2,3,4` |
| `observed` | `false` means unknown, not a loss |
| `probabilities` | Latent-oracle distribution, length 5, payout order above |
| `observable_probabilities` | Filter distribution from ticks `< t` only |
| `latent` | True shared state `0, 1, 2` (truth only; not on records) |

`queries` has one row per machine at `cutoff = n_ticks`. History ticks are `0 .. cutoff-1`. The cutoff spin is not a history row. `signal` is `null`. This family does not invent a signal value and does not emit a per-tick signal. On real data, `signal` exists only on the organizer query at the prediction cutoff.

## Assumptions

Invented constants, not fit to House Edge files.

* Latent `z` is cold `0`, neutral `1`, or hot `2`. Tick 0 draws `z` from the seed. Every machine in the episode shares that value.
* After tick `t` is drawn, `z` steps by the sign of the sum of `(payout_index - 2)` over **observed** cells at tick `t` only. A missing cell is skipped. An observed `0` still counts as index `0`. No observations, or a zero score, holds `z`. The step clips to `{0,1,2}`.
* Probabilities at tick `t` are computed before any payout at tick `t` is written. Same-tick outcomes do not affect each other or the current row.
* Slot `% 8` types: steady, steady, balanced, balanced, balanced, jackpot, jackpot, jackpot. Even `machine_id` uses `z`. Odd `machine_id` uses `2 - z` (same type, opposite loading).
* Given `z`, spins are independent. The keep mask is an independent 75% coin keyed by `(seed, episode, tick, machine)`, not by the payout. Unobserved draws remain on the record for audit and are dropped from the state.
* Oracle probabilities are the emission table of the true latent. Observable-history probabilities are the one-step predictive of a Bayes filter that knows this table, starts from a uniform prior on `z`, and conditions only on kept payout indices at ticks `< t`. The filter does not see the true latent. At tick 0 the two distributions differ whenever the prior is not a point mass on the true `z`.

Emission weights (cold, neutral, hot), each summing to 100, in payout order `(0, 1, 2, 5, 20)`:

| Type | Cold | Neutral | Hot |
| --- | --- | --- | --- |
| steady | 74, 14, 7, 4, 1 | 50, 28, 12, 7, 3 | 30, 30, 20, 14, 6 |
| balanced | 55, 20, 12, 9, 4 | 36, 28, 18, 12, 6 | 18, 22, 24, 22, 14 |
| jackpot | 62, 16, 10, 8, 4 | 40, 22, 16, 12, 10 | 20, 16, 18, 22, 24 |

Negative loading reads the hot row when `z` is cold, and the cold row when `z` is hot.

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py -v
```

Captured run: Python 3.12.3, exit 0, `Ran 18 tests`, `OK`. Transcript: `test_run.txt`.

| File | SHA-256 |
| --- | --- |
| `generate.py` | `2e35812b0c74f58327debbbe478b546d560ec87d1a0d4eefc0fe6c9f16372f3c` |
| `test_generator.py` | `141ad7b0d595141991fe4e79297535fe5dc1b031be7be223953b611254ff671a` |
| `test_run.txt` | `ec9ae17c6e2d5540e24a4e6d388b55891d8e5cef216a4279815bd029ed2b91f2` |

Covered: payout simplex and index map, reproducibility, a different seed, horizon prefix equality, future-mutation invariance of the causal replay, a same-tick edit that changes the next tick and not the current tick, missing versus observed zero, shared-latent emission identity, and latent-oracle versus observable-history probabilities.
