# periodic (RESEARCH_ONLY)

Toy slot family with a phase clock. It is not organizer data and it is not a
fitted model of the public train set. `kit/RULES.md` says to use
organizer-released data. No synthetic exception is verified. Keep the real-only
training and scoring fallback. Do not use these rows for eligibility, selection,
or submission.

## Generate

```text
PYTHONDONTWRITEBYTECODE=1 \
  /home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out out
```

`generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)` returns `records`,
`truth` (latent oracle), `observable` (history filter), and `queries`.
Default size is 6144 records. The cap is 8192.

`records.jsonl` is one cell per episode, tick, and machine. `truth.jsonl` is
the latent-oracle distribution at that same cell, five probabilities ordered
`0, 1, 2, 5, 20`. `manifest.json` stores `RESEARCH_ONLY`, the seed, settings,
and source and output hashes.

## Assumptions

Catalog rows are hand-written integer weights over 40. They were not estimated
from train, validation, or test. `steady` occupies slots 0–1, `balanced` slots
2–4, and `jackpot` slots 5–7.

Phase starts at `(seed * 17 + episode_index * 5 + machine_id * 3 + 1) mod 8`.
Each tick adds `1 + kick`. The kick depends on the payout index of
`(0, 1, 2, 5, 20)`, with kicks `(3, 0, 1, 2, 5)`. Only that machine's earlier
outcomes enter its phase. Same-tick outcomes of other machines are ignored.
`steady` and `jackpot` repeat every 4 phases. `balanced` has eight distinct rows.

Observation odds (observe, hide) are steady `(3, 1)`, balanced `(2, 1)`,
jackpot `(1, 1)`, drawn independently of the payout.

## Masks and indexes

| Object | Mapping |
| --- | --- |
| Payout value | `0, 1, 2, 5, 20` → index `0, 1, 2, 3, 4` |
| Missing | `payout_index = -1`. Unknown, not payout 0 and not index 0 |
| Slot | `slot_index = machine_id` for ids `0..7` |
| Type index | `balanced=0`, `jackpot=1`, `steady=2` (house-edge order) |
| Query | cutoff `= n_ticks`. `signal` is null. No historical signal column |

`payout` on an unobserved row is the latent draw used by the oracle. Models
must read `payout_index` (or skip `observed=false`) and must not impute 0.
The observable filter advances the clock on a gap and applies no payout kick.
The oracle still applies the latent payout's kick. Those two laws diverge
once a gap hides a kick the observer did not see. Imputing payout 0 applies
kick 3 and is a third, wrong, law.

## Tests

```text
PYTHONDONTWRITEBYTECODE=1 \
  /home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python \
  -m unittest test_generator.py -v
```

The tests check the simplex and index maps, seed reproducibility, a different
seed, future-mutation prefix invariance (including the other machines at the
edited tick), a prefix edit that changes only later ticks of that machine,
missing versus zero, the period catalog, and that stored truth matches the
latent phase rather than the observable-history phase.
