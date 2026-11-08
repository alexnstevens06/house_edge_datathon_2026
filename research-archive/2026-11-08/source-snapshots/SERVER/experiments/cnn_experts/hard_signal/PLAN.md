# hard_signal plan (FROZEN before any metric)

ADDENDUM (Alex t83, 8:07 PM CST) overrides the promotion rule in CONTRACT.md and in the earlier copy of this plan (preserved in `archive_pre_addendum/`, not modified). One full training run of this idea: seed 0 only, all three signal specialists plus a matched same-seed shared control. Do not launch seeds 1 or 2 even on a win. Record promising or negative and stop. Do not score or select on confirm or test.

## Question
Does a hard router from the actual query signal (-1, 0, +1) to a CNN specialist beat both a matched shared CNN and the published CNN03 seed-0 tune R? Lower tune R is better. Tune is the only score used.

## Data
- True TRAIN cutoff labels are `kit/public/train/labels.csv`. Expected labeled rows: signal -1 = 250, 0 = 267, +1 = 251. Any other count stops the run.
- History pretrain uses TRAIN episodes only, target tick `t` with `16 <= t < cutoff`. Features are `base_cnn.window` (ticks `< t` on all 8 machines). Labels are observed history payouts. The builder does not read queries, query labels, or signal.
- Fine-tune uses one window per TRAIN episode at `t = cutoff`. Features are ticks `< cutoff`. Labels are `labels.csv`. The signal vector is a loss mask and a router key. It is not an embedding input. Zeros are passed as the unused signal argument.
- Validation is not used to fit, to stop, or to pick a checkpoint. Fixed epoch counts only.
- Payout head order is 0, 1, 2, 5, 20. Machine types are balanced=0, jackpot=1, steady=2. Missing is -1, embedding row 5, plus an observed-mask channel. Missing is not payout 0.
- `SPLIT_IDS.json` is the `tune` and `confirm` lists from `runs/splits.json`. Scoring reads `tune` only.

## Model and budget
CNN03 architecture from `experiments/mlp_cnn_hpo/source/base_cnn.py`: `CNN(h=128, layers=4, k=3, drop=0.3, W=64, use_signal=False)`, 367109 parameters, no `sig` embedding.

Shared pretrain: AdamW lr 0.00197, wd 0.00895, 30 epochs, effective batch 512, OneCycleLR default, grad clip 1.0, CUDA bf16 autocast, fp32 weights, fp32 loss, seed 0, 2 CPU threads.

Fine-tune, control and every trained specialist: the same optimizer settings for 10 epochs, restarted OneCycle at the same lr 0.00197. Each fine-tune reseeds to 0 before its first step and loads the shared pretrain weights (specialists do not load the control).

The cutoff set is 96 episodes, so each fine-tune epoch is one optimizer step. Control and each specialist therefore get 10 steps on the same 96 windows. Labeled-slot exposure is not equal: the control loss uses every cutoff label, and each specialist loss uses only its own signal. That difference is the experiment, not a hidden extra budget.

A signal with fewer than 64 labeled TRAIN cutoff rows does not get a specialist. Those queries use the shared pretrain checkpoint (`shared_clean`), not the control. The real counts are all above 64, so this run trains all three specialists and the fallback is unused on real queries.

VRAM cap is 2GiB via `set_per_process_memory_fraction`. Effective batch 512 is accumulated from microbatches of 128. There is no second microbatch size and no automatic relaunch if the allocator refuses.

## Eval
CPU fp32 softmax, stored float64, the same reduction `base_cnn.predict` uses when `dev="cpu"`. Official `score_predictions` on the tune episode ids only. Compare routed tune R with the matched control and with CNN03 seed 0 tune R 0.33206398675779764 from `final_cnn03_s0/metrics.json`.

This is not a CNN03 reproduction. CNN03's `build_train` includes `t == cutoff` inside the same 30-epoch loop, uses one 512-row backward, and sets a 0.25 memory fraction. Those differences are recorded in RECONCILIATION.md.

## Verdict
Promising only if routed tune R is strictly lower than both the matched control and CNN03 seed 0. Otherwise negative. A negative train loss is also negative. Either verdict stops the idea. No seed 1, seed 2, or replacement run.
