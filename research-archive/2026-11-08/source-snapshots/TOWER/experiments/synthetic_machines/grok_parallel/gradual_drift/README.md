# gradual_drift (research only)

Toy synthetic slot family for local research. It is **not** organizer data.

`kit/RULES.md` says to use organizer-released data. This workspace has no verified exception that would let synthetic rows replace those files. Keep the real-only fallback: fit and submit from the released train, validation, and test files. Do not train a contest model on this generator. A separate utility coordinator owns training.

## What it generates

`generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)` returns a dict with `records` and `truth`.

Each record is one machine spin:

- `episode_id`, `tick`, `machine_id` (0..7), `machine_type` (`steady`, `balanced`, or `jackpot`)
- `payout` in `{0, 1, 2, 5, 20}` and `payout_index` / `slot_index` in that same order
- `observed` and `feature_mask` (`1` usable, `0` missing)

Each truth row is the same key plus:

- `probabilities`: latent-oracle categorical, length 5, order `0, 1, 2, 5, 20`. These are the sampling odds.
- `observable_probabilities`: history filter from the type prior and this machine's **observed** payouts at ticks `< t` only.

Default size is 8 × 96 × 8 = 6144 records. The generator refuses more than 8192.

## Assumptions

These constants are invented. They are not fit to public, train, validation, or test logs.

- Logits = `BASE_LOGITS[type] + phase_t * DRIFT_LOGITS[type]`, then softmax.
- `phase_t` is a seed phase, plus `direction * SPEED[type] * t/(T-1)`, plus an EWMA of this machine's true centered payouts at ticks `< t`, plus a gain times the mean EWMA of other machines that already spun before `t`.
- Centered payout is `index/4 - 0.5`. The EWMA starts at 0, which means "no spins yet", not "payout was 0".
- Every machine's tick-`t` distribution is fixed before any tick-`t` payout is written into the EWMA. Another machine's same-tick outcome is not an input.
- Gaps are deterministic (`GAP_STRIDE` plus a hash draw). The spin still happens and still moves the latent phase. The history filter drops `feature_mask == 0` rows. Do not zero-fill those rows and do not multiply payout by the mask.
- One query per machine uses cutoff `n_ticks - 1`. `signal` is null. No historical signal series is invented. On real contest rows, use the organizer query `signal` at that row's cutoff only.

## Run

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py --seed 0 --episodes 1 --ticks 16 --machines 4 --out sample_out
```

The CLI writes `records.jsonl`, `truth.jsonl`, `queries.jsonl`, and `manifest.json` with `RESEARCH_ONLY`, the seed and settings, and SHA-256 hashes of the source files and outputs.
