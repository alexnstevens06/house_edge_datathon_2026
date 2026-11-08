# machine_type_hierarchical

Research-only toy generator for one synthetic slot family. It does not train a model.

Eligibility is unresolved. Organizer rules say to use organizer-released data. There is no verified exception that lets these rows stand in for real episodes. Scored training keeps the **real-only fallback**: ignore this directory and use the released train split only. Do not read or copy validation or test outcomes into this generator. Nothing here was fit to those files.

## Interface

```python
from generate import generate

bundle = generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)
```

`bundle["records"]` and `bundle["truth"]` have one row per episode, tick, and machine. Default size is 6144, under the 8192 cap. Larger requests raise `ValueError` unless `allow_oversize=True`.

Record fields: `episode_id`, `tick`, `machine_id`, `machine_type`, `payout`, `observed`.

- `machine_id` is `0 .. n_machines-1`.
- `machine_type` is `steady`, `balanced`, or `jackpot`.
- `payout` is one of `0, 1, 2, 5, 20` even when `observed` is false. That value is the latent draw for bookkeeping. It is not an observation.

Truth fields: `episode_id`, `tick`, `machine_id`, `machine_type`, `probability_kind`, `probabilities`.

- `probability_kind` is `latent_oracle`.
- `probabilities` has length 5 in payout order `0, 1, 2, 5, 20`.

`bundle["queries"]` has one query per machine at cutoff `n_ticks - 1`. `signal` is `null`.

CLI:

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 2 --ticks 16 --out sample_out
```

Writes `records.jsonl`, `truth.jsonl`, `queries.jsonl`, and `manifest.json` with source and output SHA-256, seed, settings, and `RESEARCH_ONLY`.

## Mappings

| Payout | Slot index |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 5 | 3 |
| 20 | 4 |

Machine type for `machine_id % 8`:

| machine_id % 8 | type |
| --- | --- |
| 0, 1 | steady |
| 2, 3, 4 | balanced |
| 5, 6, 7 | jackpot |

Usable input mask for predicting tick `t` on a machine: same `episode_id`, same `machine_id`, `tick < t`, and `observed is true`. Anything else is out. A gap is unknown, not payout 0. Another machine's outcome at tick `t` is out. Query `signal` is not a feature. Real signals live only on organizer query rows at the prediction cutoff. This family does not invent a signal history.

## Generative assumptions

All numbers are constants in `generate.py`. They are not estimated from public train frequencies.

1. Episode house is one of three fixed vectors, drawn before any tick.
2. Type-level distribution = `0.80 * prototype + 0.20 * house`. Machines of the same type share it.
3. Each machine tilts that mixture: `softmax(log(type_level) + sigma * (U - 0.5))` with `U` uniform on `[0, 1)` and `sigma` 0.04 / 0.20 / 0.35 for steady / balanced / jackpot.
4. Prototypes, order `(0, 1, 2, 5, 20)`:
   - steady `(0.62, 0.22, 0.10, 0.05, 0.01)`, prior strength `40`
   - balanced `(0.40, 0.28, 0.18, 0.10, 0.04)`, prior strength `6`
   - jackpot `(0.70, 0.12, 0.08, 0.04, 0.06)`, prior strength `12`
5. With `n` usable earlier payouts and counts `c`, the base update is `(tau * machine_base + c) / (tau + n)`.
6. If the machine is jackpot and the last usable payout is `5` or `20`, mix that vector `0.50 / 0.50` with hot vector `(0.35, 0.15, 0.15, 0.15, 0.20)`.
7. Missingness is MCAR at rate `0.25`, drawn before the payout and independent of it.
8. RNG is `random.Random` (MT19937), streams mixed from `(seed, episode_index, stream_id)`.

Latent-oracle truth uses steps 2–6. Observable-history probabilities use the same count and heat rules on the **type prototype only**, with no house and no machine tilt. Those two distributions differ. Truth files store the oracle.

State updates are synchronous. Probabilities at tick `t` are computed for every machine before any tick-`t` payout is written into state.

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest -v test_generator
```

Run this from this directory. The suite checks the simplex and slot map, seed reproducibility, a changed seed, prefix stability, future-intervention invariance, past-intervention movement, cross-machine non-leakage, gappy missing versus zero, the inlined known-truth formula, and oracle versus observable-history probabilities.
