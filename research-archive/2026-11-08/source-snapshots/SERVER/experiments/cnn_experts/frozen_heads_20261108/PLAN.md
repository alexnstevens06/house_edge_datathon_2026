# frozen_heads plan (FROZEN before any metric)

New idea. The hard_signal 30+10 full-model fine-tune is negative and closed. This plan does not relaunch it and does not edit that folder.

ADDENDUM rule still applies: one seed, seed 0 only. No seeds 1 or 2. No confirm, test, or full score. Tune R only. Lower is better.

## Trunk
Read-only checkpoint:

`experiments/cnn_experts/hard_signal/runs/s0/pretrain/checkpoint.pt`

sha256 `e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760`

It is the genuine history pretrain from that run: phase `pretrain`, seed 0, 1110 optimizer steps, config epochs 30, lr 0.00197, wd 0.00895, effective batch 512. Windows were `16 <= t < cutoff` only (18471 windows). Query signal was not an input. Parameter count 367109, no `sig` tensor, float32.

CNN03 `final_cnn03_s0` is not this trunk. Its `build_train` includes `t == cutoff` and the query labels in the same 30-epoch loop. This experiment does not load that checkpoint and does not retrain a trunk.

Every trunk parameter and buffer is frozen. Training must leave them bitwise unchanged.

## Heads (the only trainable weights)
`TinyHead`: `Linear(5, 5)` on the frozen trunk logits. Identity weight, zero bias, so step 0 matches the trunk. 30 parameters per head.

- Baseline: frozen trunk, no head.
- Matched control: one shared head, all true TRAIN cutoff labels.
- Signal system: one head per signal in {-1, 0, +1}, labels of that signal only. Each head is initialized from the same identity and the same seed, and loads the same trunk. Heads do not load the control.

A signal with fewer than 64 labeled TRAIN cutoff rows gets no head. Those queries use the frozen trunk. Expected counts are 250 / 267 / 251, so all three heads train on this data.

## Schedule, declared once
Adam, lr 0.001, weight decay 0, grad clip 1.0, 10 epochs. One optimizer step per epoch over all 96 cutoff episodes (batch cap 512). Control and each signal head get 10 steps on the same windows. Label exposure differs by the mask. That is the intervention. No learning-rate schedule, no early stop, no epoch picked on validation. Last epoch only.

## Eval
CPU float64. A local `Tensor.float -> float64` redirect is used only while the copied network runs, then restored. Softmax is float64. Each probability row is divided by its sum. Official `score_predictions` on `SPLIT_IDS.json` tune ids only. Two threads. No production `PINNED.json` and no production fp64 harness.

## Verdict
Promising only if the routed heads have tune R strictly below both the frozen-trunk baseline and the matched shared head. Otherwise negative. Then stop.
