# rare_heavytail

Toy synthetic slot family for research. Status: **RESEARCH_ONLY**.

Contest rules say to use organizer-released data. A synthetic-data exception is not verified. `training_source_policy()` keeps the real-only fallback on. Do not train a contest model on these rows. Another utility owns training. This directory does not read validation or test logs and does not fit parameters on train.

## Generate

```python
from generate import generate

bundle = generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)
```

`bundle["records"]` has one row per episode, tick, and machine. Fields include `episode_id`, `tick`, `machine_id` (slot index 0..7), `machine_type` (`steady`, `balanced`, or `jackpot`), `payout` in `{0, 1, 2, 5, 20}`, and `observed`. `mask` is 1 when the spin is logged and 0 when it is hidden. The latent `payout` is still stored on hidden rows so the oracle can be checked. Callers who build features from the log use `observed` and skip hidden rows.

`bundle["truth"]` is aligned with those rows. `probabilities` has length 5 in payout order `(0, 1, 2, 5, 20)`. `probability_kind` is `latent_oracle`: the distribution that was sampled, given that machine's own latent spins at ticks strictly before `tick`, including spins the log hid.

`bundle["queries"]` asks for the unseen spin at `cutoff = n_ticks`. History ticks are `0 .. n_ticks-1`. `signal` is null. This family does not invent a historical signal and does not copy organizer cutoff signals. Each query stores both `latent_oracle_probabilities` and `observable_history_probabilities`.

The default call writes 6144 records. The generator refuses shapes above 8192 records.

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out /tmp/rare_heavytail_out
```

The CLI writes `records.jsonl`, `truth.jsonl`, `queries.jsonl`, and `manifest.json`. The manifest carries `RESEARCH_ONLY`, the seed and settings, source hashes, and output hashes.

## Assumptions

- Weights are invented per-thousand tables, not estimates from contest data.
- Slots are `steady`, `steady`, `balanced`, `balanced`, `balanced`, `jackpot`, `jackpot`, `jackpot`. Fewer than 8 machines keep that prefix.
- Heat is an integer in 0..3. On the last 8 own-machine values, oldest first, starting from 0: a 20 adds 2, a 5 adds 1, a 0 subtracts 1, and 1 or 2 hold. Heat stays inside 0..3.
- Cold jackpot weight on 20 is 8/1000. Heat 3 raises it to 250/1000. That jump is the family signature.
- Every machine's tick-t distribution is fixed before any tick-t payout is drawn. Other machines' tick-t outcomes are not inputs.
- `tick % 5 == 4` is always hidden. Other ticks are hidden with a type coin (steady 900/1000, balanced 750/1000, jackpot 600/1000). The coin does not look at the payout. This mask is not the contest short-history rule.
- Hidden cells are unknown. The observable fold skips them. Putting payout 0 in that cell is a different process and changes heat whenever a zero would have cooled the tail, or whenever a skipped 5 or 20 would have heated it.
- Each episode has its own `random.Random` stream, seeded with `rare_heavytail:{seed}:{episode}`. Per cell it draws one `randrange(1000)` for the payout. Non-gap cells draw a second `randrange(1000)` for the coin. A shorter `n_ticks` with the same seed is a tick-prefix of a longer run.

## Mappings

| Payout | Probability index | Dense symbol when observed |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 1 | 2 |
| 2 | 2 | 3 |
| 5 | 3 | 4 |
| 20 | 4 | 5 |

Dense symbol 0 means unknown. Observed payout 0 uses dense symbol 1. `mask` 0 means unknown. Sparse batches drop unknown rows; a pad symbol, if a batcher needs one, is 5 and is not a payout.

## Checks

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator -v
```

Run that from this directory. The tests cover the simplex and index map, seed reproducibility, a changed seed, prefix equality, future-mutation invariance with a later probability that actually moves, contemporaneous cross-machine silence, hidden-versus-zero heat, latent-oracle versus observable-history rows, the handwritten heat table, null cutoff signals, the record cap, and the CLI hashes.
