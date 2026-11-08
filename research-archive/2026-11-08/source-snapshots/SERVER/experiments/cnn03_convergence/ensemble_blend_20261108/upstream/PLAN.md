# Continuation plan (frozen before new metrics)

Written before any continuation epoch and before study FP64 scores of these checkpoints. This file is the selection and budget contract. It is not a submission.

## Hardware at freeze

One GPU is present: NVIDIA GeForce RTX 5080, 16303 MiB, 2 MiB used, utilization 0, no compute process (checked 2026-11-08 05:59 UTC / 2026-11-07 23:59 CST). There is no second GPU on this host. Concurrency is two trainers on that one device. Each trainer uses at most 2 CPU threads and a 0.40 CUDA memory fraction. Batch size stays 512 so the step count stays 37 per epoch.

## Provenance

`train_canonical.py` matches `experiments/mlp_cnn_hpo/source/train.py` by the earlier byte compare. `base_cnn.py` matches the recorded HPO hash `b4bbd9af…`. The six curve checkpoints contain only `config` and `state` (43 tensors). They do not contain AdamW moments, OneCycleLR state, or RNG state. Exact resume is impossible. Originals stay read-only.

Prior BF16 tune R (not reused as the study metric): high-LR 0.31022932 / 0.34546345 / 0.30950934 versus control 0.33199008 / 0.34547671 / 0.33332854.

## Warm-start restart (predeclared)

For every arm below, independently:

- Load `state` into `CNN(h=128, layers=4, k=3, drop=0.3, W=64, use_signal=False)` after `B.USE_SIGNAL = False` and `build_train` on train only.
- Then seed. Continuation seed is `1000 + original_seed + extra_epochs`. That stream drives shuffle and dropout only. It is not the original training RNG.
- Fresh AdamW (`lr` = that arm's saved lr, `weight_decay=0.00895`). Fresh OneCycleLR over the extra steps only, `pct_start=0.3`, library defaults otherwise (`div_factor=25`, `final_div_factor=1e4`).
- `total_steps = extra_epochs * 37`. Require `n_train_windows == 18567`.
- Two independent restarts from the **original** weights, not chained: `extra_epochs=10` (370 steps) and `extra_epochs=20` (740 steps). The +20 run is not a resume of the +10 run. Both are capped. No third tranche, no new seed, no learning-rate search.
- Training is CUDA bf16 autocast, masked loss, grad clip 1.0, the same batch loop as `conv_train.py`. Refuse CPU. Log pid, device name, and GPU utilization on epoch 0.
- Saved continuation checkpoints may include optimizer, scheduler, and RNG because those objects exist for the new run. That does not rewrite the parent into an exact resume.

Arms and parent SHA-256 values are `MANIFEST.json`. Learning rates stay 0.003 for `curve_C_lr003_s*` and 0.00197 for `curve_control_s*`. Seeds stay 0, 1, 2. Twelve jobs.

## Scoring contract

Study metric is CPU float64. Forward is the untouched canonical module `experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py` (same bytes as the anchor FP64 model). Softmax stays float64. Rows are renormalized with the canonical `sum` of `float` values, the same rule as `house_edge.scoring.write_submission_csv`. `score_predictions` then applies the kit tolerance renorm. Tune episode ids only, from `runs/splits.json` key `tune`. No confirm score, no test score, no full-validation leaderboard.

`experiments/cnn_runtime/fp64_harness.py` and `PINNED.json` are not opened for edits, not monkeypatched, and not cited as production clearance.

The incumbent comparison row is a read-only FP64 rescore of `experiments/mlp_cnn_hpo/runs/final_cnn03_s0/checkpoint.pt` on tune only. The published CPU-fp32 tune R 0.33206398675779764 is context, not the study denominator.

## What will be reported

All six parents and all twelve continuations get a tune R. The candidate is the lowest tune R among the three high-LR +20 runs. If a matched control is lower, the report says so. Nothing is trained further from that number. No submission CSV and no zip is written.

## Blends (do not block the GPU launch)

Frozen weights `0, 0.025, 0.05, 0.1, 0.2`. Two mixes, both on validation rows, scored on tune ids only:

- `(1-w) * CNN03 canonical FP64 CSV + w * hard-signal routed specialist probabilities`
- `(1-w) * CNN03 canonical FP64 CSV + w * hard-signal matched-control probabilities`

CNN03 file: `experiments/cnn_runtime/server_fp64_cnn03_val1958.csv` (SHA in the manifest). Specialist and control vectors are materialized by CPU inference from the closed `experiments/cnn_experts/hard_signal/runs/s0` checkpoints. That is not a relaunch of specialist training. `w=0` must reproduce the CNN03 CSV tune R. The grid is scored on the same tune split used as the report, so the minimum is optimistically biased and is not a confirm result. No normalization coefficient is fit on held-out rows.

## Stop rule

Stop when the twelve continuation jobs and the predeclared tune scores and blends have been written, or at the first CUDA/OOM/unrelated-GPU refusal. Deadline 2026-11-08 10:30 America/Chicago.
