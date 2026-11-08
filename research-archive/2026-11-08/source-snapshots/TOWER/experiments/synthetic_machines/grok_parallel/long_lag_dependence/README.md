# long_lag_dependence (RESEARCH_ONLY)

Invented toy slot family. Contest rules say to use organizer-released data and do not give a verified synthetic-data exception, so eligibility is **unresolved**. Keep the real-only fallback: do not train on this output in place of organizer train, validation, or test. Do not read real validation or test to fit it. No parameters here were fit to organizer logs.

`truth[].probabilities` is the **latent-oracle** conditional: the distribution the sampler used, given this machine's own latent payouts at lags 24 and 48, including spins that were not observed.

`observable[].probabilities` is the **observable-history** conditional: the same formula with missing lags dropped. A missing tick is unknown. It is not payout 0, and it is not the hidden payout.

## Generative assumptions

- `machine_id` is the slot, fixed `0..7`. Types are assigned, not estimated: slots 0–1 steady, 2–4 balanced, 5–7 jackpot.
- Payout order is `(0, 1, 2, 5, 20)` → indexes `0..4`. Observation index `-1` means missing.
- Type index order matches the public loader: balanced 0, jackpot 1, steady 2.
- Tick `t` is drawn from a mixture of the type base table and tilt tables at lags 24 (weight 1/2) and 48 (weight 1/4). Weights apply only when that lag exists. Tables are positive percents and are written in `generate.py`.
- Update is synchronous. Inputs are strictly ticks `< t` on the **same** machine. Another machine's spin at `t` is not an input.
- About one fifth of spins are masked by a deterministic function of `(seed, episode, tick, machine)`. The mask does not look at the payout or at future ticks.
- History rows still store the latent `payout` so the oracle can be checked. If `observed` is false, features must use `observation_index == -1` and must ignore `payout`.
- Queries exist only at cutoff `n_ticks` (the next spin after history `0 .. n_ticks-1`). `signal` is null. This family does not invent a historical signal series. Real query signals, when used, belong only on a real prediction cutoff.

Default `generate()` emits `8 * 96 * 8 = 6144` records (cap 8192).

## Run

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out /path/to/out
```

Writes `records.jsonl`, `truth.jsonl`, `observable.jsonl`, `queries.jsonl`, and `manifest.json` (`RESEARCH_ONLY`, seed, settings, source hashes, output hashes).

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py
```

Checks: simplex and payout/slot maps, pinned mixture, every-row known-truth identity, reproducibility, seed sensitivity, prefix stability when the horizon or episode count grows, future-tick mutation leaves the long-lag conditional unchanged while editing lag 24 changes it, missing versus filled-in zero, oracle versus observable on gaps, cutoff-only queries with null signal, the 8192 cap, and CLI manifest hashes.

Recorded run (Python 3.12.3, 7 tests, OK):

```text
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py
```

Source SHA-256:

- `generate.py` `dc924859645bfcf8ec12703115d98fe1f4744ed770e83ce38b62b2f61f952821`
- `test_generator.py` `86136d20afcc35935a48e72b6086e70b99c0eb6bb7c92e7564cceffa79af6ccc`

Example CLI (`--seed 0 --episodes 1 --ticks 64`, 512 records), output SHA-256:

- `records.jsonl` `a810ca4ef929cd0a0e8c2d4bcabeeee92c27c15adc279951916a350a2b0dcf81`
- `truth.jsonl` `ebe8379e6bca4965b10252b9fd903e91e5e6f88e7b08c21305982aad0d767115`
- `observable.jsonl` `3e9cb31edd88c68b33d9fdfafbe4dd3d8a599b663bd9488a4d1ba41a6f9dc383`
- `queries.jsonl` `a045637eb0c3d35aec436051ad3fc82ffeeaeb47a8ca26b8e863154ad6a4f15f`
