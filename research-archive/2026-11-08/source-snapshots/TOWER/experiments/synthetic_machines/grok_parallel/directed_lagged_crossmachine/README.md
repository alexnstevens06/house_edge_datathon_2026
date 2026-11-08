# directed_lagged_crossmachine

RESEARCH_ONLY toy simulator for one synthetic family. Competition eligibility of synthetic rows is **unresolved**: organizer rules say to use organizer-released data, and there is no verified synthetic exception. The real-data path is the fallback. Do not train a submission model on these files. This directory does not read organizer validation or test splits and does not fit parameters to public train rows.

## Generate

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out .
```

`generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)` returns `records`, `truth`, and `queries`. Defaults write 6144 history rows (cap 8192).

| File | Contents |
| --- | --- |
| `records.jsonl` | One row per history spin. `payout` is the latent outcome. When `observed` is false, `payout_is_feature` is false and that payout is not a feature. |
| `truth.jsonl` | Latent-oracle probabilities for each history row, same order as `records`. |
| `queries.jsonl` | One row per machine at `cutoff = n_ticks`. Holds both latent-oracle and observable-history probabilities of that unseen spin. |
| `manifest.json` | `RESEARCH_ONLY`, seed, settings, mappings, source and output SHA-256. |

## Assumptions (invented)

Directed ring, lag 1: the parent of machine `i` is `(i - 1) mod n`. At tick `t` every machine is scored from tick `t - 1` only, then the tick is drawn. No contemporaneous cross-machine outcomes.

Base counts over payouts `(0, 1, 2, 5, 20)`, denominator 20:

- steady: `(10, 5, 3, 1, 1)`
- balanced: `(4, 5, 6, 3, 2)`
- jackpot: `(6, 3, 2, 3, 6)`

`P = 7/10 * base[type] + 3/10 * parent_belief`. Tick 0 uses the base (`parent_belief` absent). The latent draw uses the parent's true payout. `slot_index` equals `machine_id`.

Public machine-type indices, fixed literals, not list position or slot number: balanced `0`, jackpot `1`, steady `2`. A type draw is `randrange(3)` read through that map.

Each episode has its own SHA-256 RNG stream from `(family, seed, episode_index)`. Ticks are drawn in order inside that stream. Generating more ticks leaves every episode's earlier ticks unchanged.

## Masks and queries

`as_feature_row` is the feature adapter. It never copies `payout` or `payout_index`. `feature_payout_index` is the payout index when `observed` is true and `-1` when it is not. `-1` is not the index of payout 0.

An unobserved simulator row may still store the latent `payout` so the oracle can be checked. That field is hidden from features: `payout_is_feature` is false, and the adapter mask is `-1`. Do not train or featurize `payout` unless `observed` is true. The mask depends on `(seed, episode, tick, machine)` only: missing when that linear form is `0 mod 5`. A hole does not carry the last payout forward.

`signal` exists only on query rows, at the prediction cutoff, and is the constant `0`. It is not written on history and is not a kernel input. History features for a prediction at tick `t` are records with `tick < t`.

## Two probability layers

Latent-oracle probabilities condition on the parent's true payout even if that spin was unobserved. Observable-history probabilities use the parent's payout only when it was observed; otherwise they mix over the filtered belief. They match when the lag parent was observed (and at tick 0) and differ when it was missing.

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator -v
```

Stdlib only. The command, exit status, and hashes are recorded in `test_run.json` after a run.
