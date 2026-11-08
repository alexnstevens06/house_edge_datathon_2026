# hidden_regime_switching

Research-only toy generator for a hidden regime-switching slot family. Eligibility is `RESEARCH_ONLY`. Organizer instructions require organizer-released data, and no synthetic exception is verified. Keep a real-only training path; do not train on this output unless that exception is confirmed.

Probabilities are explicit fractions with denominator 16. They are invented. They are not fit to train, validation, or test.

## Process

Each machine has an observable type `balanced`, `jackpot`, or `steady` and a latent regime `dull`, `normal`, or `hot`. The regime is omitted from records.

At tick `t` the latent-oracle payout law is the emission row of the current regime and the machine type. The observable-history law is the same table mixed by a filter belief built only from that machine's ticks `< t`. A missing tick skips the emission update and uses `TRANS_MISS`. An observed payout uses `TRANS_OBS` at that payout's index. Those kernels differ, including payout 0 versus missing.

Machines are independent. The regime used at tick `t` is already fixed before the tick-`t` payout is drawn. The transition reads that machine's own tick-`t` observation and writes the regime for tick `t + 1`.

Queries exist only at `cutoff = n_ticks`. `signal` is null there. History records have no signal field.

## Interface

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out /tmp/hrs
```

`generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)` returns `records`, `truth`, and `queries`.

History records: `episode_id`, `tick`, `machine_id` (slot `0..7`), `machine_type`, `payout`, `observed`, `obs_mask`. Unobserved rows store `payout: null` and `obs_mask: 0`.

Truth rows: `probabilities` (length 5, latent oracle, order `0,1,2,5,20`) and `observable_probabilities` (filter). Cutoff rows use `role: "cutoff"` and have no record.

Index maps, copied from the public kit's naming:

| Map | Encoding |
| --- | --- |
| Payout index | `0,1,2,5,20` -> `0,1,2,3,4` |
| Machine type index | `balanced, jackpot, steady` -> `0,1,2` |
| Slot | `machine_id` |
| Query index | slot order inside the episode, id suffix `:s{slot}` |
| Missing grid | `-1` |
| Dense symbol | `0` missing, `1..5` = payout index + 1 |
| Sparse symbol | `0..4` payout index; missing ticks omitted |
| Mask | `obs_mask` 1 observed, 0 missing |

Default size is `8 * 96 * 8 = 6144` history records. The generator rejects more than 8192.

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py
```

The test run command, result, and file hashes are recorded in `manifest.json`.
