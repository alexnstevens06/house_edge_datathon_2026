# Ensemble and blend plan (frozen before new metrics)

Study directory: `experiments/cnn03_convergence/ensemble_blend_20261108`.
This file is the selection contract for this stream. It is not a submission and it does not package a candidate.

Coordinator protocol: `COORDINATOR_PLAN.md` sha256 `03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3`, read in full before evaluation. Global ranking, the 1e-6 threshold, and the caveats in that file stay in force. This stream does not add weights, seeds, rates, or components.

## What is scored

CPU float64 only, on the already installed study interpreter `.venv_gfx1031` (Python 3.12, torch 2.14). No autocast and no `.float()` cast. Forward and softmax use the unpacked canonical module whose sha256 is `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a`. Rows are renormalized by `float()` then `sum`, then `house_edge.scoring.score_predictions` applies the kit tolerance renorm. The scored ids are the frozen 48 tune episodes / 384 tune queries in `SPLIT_IDS.json`. Confirm, full, and test are not scored. The official validation checker is not called.

Checkpoints are the local copies under `inputs/`. They are not retrained. Six curve parents stay weights-only. The closed hard shared control is `inputs/hard_shared_control/checkpoint.pt` sha256 `92233992aeb8c007a40f925885d7f85ef260cf8584ea0cc669dcd39ce57d201f`.

## Predeclared rows

Ensembles, equal probability means, then the same renorm:

- `ensemble_equal_s0_s1_s2`: seeds 0, 1, and 2 of original LR 0.003.
- `ensemble_equal_s0_s2`: seeds 0 and 2. This row is tune-informed because seed 2 is the known best single and seed 1 is the known worst single.

Blend, only original LR 0.003 seed 2 with the closed hard shared control, weights exactly `0, 0.025, 0.05, 0.1, 0.2`. Weight 0 must be the exact seed-2 probability map, not merely the same R. No other component, crossover, or weight.

Published CPU float64 tune R values used as reproduction gates, not as fitted targets:

- seed 0: 0.3101981645338761
- seed 1: 0.3452074168357309
- seed 2: 0.30980333573689856
- hard shared control: 0.46211193532631023

The control gate is 1e-12 absolute, checked before any blend row is written. Seed 0, 1, and 2 use the same gate before either ensemble is written. A miss stops the matrix and is investigated as a forward bug. Coefficients are not fit and labels are not backfilled.

Immutable current best for the global threshold is the already packaged seed-2 candidate, tune R 0.30980333573689856. ZIP `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29` and test CSV `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6` stay untouched. An ensemble or blend is eligible only when it is at least 1e-6 better than that number. A tie within 1e-6 keeps the current best. Nothing in this stream is packaged. Low-rate checkpoints are a later stream and are not ranked here.

## Guards

Before a forward, refuse a wrong file hash, a config or key mismatch, a shape mismatch, a signal tensor, a saved dtype other than float32, or a non-finite weight. Scoring refuses an unknown query id and a map that misses a tune id. Source, model, data, and split hashes are pinned in `MANIFEST.json`. Production `PINNED.json` and `fp64_harness.py` are not opened for edits and are not monkeypatched.

## Low-rate handoff

This plan does not train. If `lowrate_20261108/HANDOFF_READY` is absent when the matrix is done, write `RECEIVER_READY.md` and stop. Do not poll.
