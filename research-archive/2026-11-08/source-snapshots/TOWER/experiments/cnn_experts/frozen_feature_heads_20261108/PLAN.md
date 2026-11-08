# Frozen feature heads (seed 0 only) — FROZEN before any tune metric

Study folder: `experiments/cnn_experts/frozen_feature_heads_20261108`. One seed. No second seed, no follow-up configuration, no hyperparameter search, no early stopping.

This is not the logit-calibration study (a 5×5 map on the 5 logits). It does not retrain the CNN trunk.

## Trunk

`inputs/history_pretrain.pt`, required sha256 `e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760`.

Inspected before this plan was used for training: `phase=pretrain`, `seed=0`, `steps=1110`, 30 recorded losses, config `lr=0.00197`, `wd=0.00895`, `bs=512`, `micro_bs=128`, `epochs=30`. State dict has 367109 floats, no `sig` parameters. Shapes are CNN03 no-signal: h=128, 4 causal blocks, window 64, dropout 0.3, head `Linear(256,128) + GELU + Linear(128,5)`. The checkpoint does not store h/layers/W/drop as fields; those are fixed by the tensor shapes and by the CNN03 source that matches this layout (`experiments/mlp_cnn_hpo/source/base_cnn.py`, sha256 `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5`).

Abort if the file hash, step count, parameter count, or `load_state_dict(..., strict=True)` fails.

## Features and heads

The frozen extractor is every trunk weight and buffer up to and including `head[0]` and `head[1]` (the 256→128 linear and its GELU). The exposed feature is that 128-d vector, shape `(batch, 8, 128)`, not the 5 logits.

Each tiny classifier is `nn.Linear(128, 5)`: 128×5 + 5 = 645 parameters. The shared control and the three signal heads (`-1`, `0`, `+1`) are each initialized by copying `head[2]` (`weight` `(5,128)`, `bias` `(5,)`). No other initialization. Epoch 0 of every head therefore matches the trunk `eval()` logits on the same features. That match is an assert, not a selection metric.

Deviation from a train-mode trunk forward, recorded here before any metric: the extractor is held in `eval()` with `requires_grad=False` for the whole study. Dropout is therefore the identity. A train-mode trunk would change features every forward; that is disallowed. Epoch 0 matches the trunk in eval mode, which is also how the baseline is scored.

Sparse fallback: a signal with fewer than 32 labeled TRAIN cutoff rows is not given a trained head. Those slots keep the frozen trunk `head[2]`. The known counts are 250 / 267 / 251, so a real seed-0 run fits all three heads. The threshold is part of this plan, not a tuned knob.

## Data

TRAIN query labels only, one window per train episode at the true cutoff, features from ticks `< cutoff` only. Signal is the routing and loss mask, never a network input, and it is never copied onto earlier ticks. This study does not build historical pseudo-cutoff windows and does not backfill signal.

Required labeled-row counts, or abort: `{-1: 250, 0: 267, 1: 251}`.

Validation is used only to score the predeclared tune episode ids in `SPLIT_IDS.json` (copied from `runs/splits.json`). Confirm episodes are not forwarded and not scored. Test is not run.

## Schedule (one fixed setting)

| | |
| --- | --- |
| seed | 0 only. Any other seed aborts. |
| epochs | 10 |
| steps | 10. One step per epoch on all 96 cutoff windows. |
| optimizer | AdamW |
| lr | 0.001 |
| weight decay | 0 |
| grad clip | none |
| loss | masked cross-entropy, fp32, slots with `y >= 0` only |
| head batch | the same 96 windows for the control and for every signal head |

The control masks nothing (all legitimate cutoff labels). A signal head keeps only slots whose own signal equals that head. Same trunk, same windows, same init, same step count.

Trunk features for the head update are computed on the ROCm device in fp32 under `torch.no_grad()`. Training must run a CUDA/ROCm tensor op and a backward. There is no CPU training fallback. If that device check fails, write `BLOCKER.md` and stop.

Scoring forward is a separate CPU fp64 pass of the same frozen architecture, fp64 head, softmax, then per-row renorm in fp64. Tune R only, via `house_edge.scoring.score_predictions` and the tune id set. Lower R is better.

## What will be reported

Tune R for three systems, and the raw differences:

- frozen baseline: trunk `head[2]`, no head update
- shared head: the control after 10 steps
- routed heads: per-slot hard choice of the signal head (or the trunk head if that signal fell back)

No winner is promoted. Seed 1 and seed 2 are not trained, whatever the signs of the differences. Tune was already used to build CNN03, the heads see only about 250 labels, and 10 steps is a short fixed budget. Those are caveats, not a license to keep fitting.

## Hardware bound

Tower AMD Radeon RX 6700 XT, PyTorch 2.9.1+rocm6.4, at most 2 CPU threads, process memory fraction 0.25. Do not start if another compute process is on the GPU.
