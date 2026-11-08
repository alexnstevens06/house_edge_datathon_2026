# soft_moe plan

ADDENDUM (Alex t83, experiments/cnn_experts/ADDENDUM.md) overrides the old promotion rule,
CONTRACT promotion language, and the earlier draft of this plan. Acknowledged: one full
training run for this idea (the 3-expert soft mixture plus the matched seed-0 control).
Do not launch seeds 1 or 2 on a win. No confirm/test selection. No submission.

No metric has been written yet. The previous draft (PLAN_HASH 995e273164cba0a9…, ft_epochs 30,
seeds 1 and 2 on a win) is superseded by this file.

Experiment (B). Reference bar, not a launch list: incumbent CNN03 seed 0 tune R
0.33206398675779764 (lower is better). Seeds 1 and 2 of the incumbent are not run.
Source of architecture and no-signal windowing: experiments/mlp_cnn_hpo/source/base_cnn.py
sha256 b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5.
CNN03 config: h128, L4, kernel 3, W64, dropout 0.3, lr 0.00197, wd 0.00895, batch 512,
AdamW, OneCycleLR, grad clip 1.0, 30 epochs, no validation early stopping.
Payout class order 0,1,2,5,20. Machine type: balanced=0, jackpot=1, steady=2.
Missing mask is -1, embedding index 5, distinct from payout 0.
CPU fp32, 2 threads, for tune scoring. Tune-split R only. Confirm is seen and is not a
selection metric. Test predictions are not written.

## Data

Reuse CNN03 `window` / train-window rule: min_t=16, stride=1, features are ticks < t only,
label is the payout at t (history) or the train query label at the true cutoff.
Historical windows (t < cutoff) store signal ABSENT (index 3). No query-signal backfill.
True TRAIN cutoff rows store the legitimate per-machine query signal {-1,0,1} -> {0,1,2}.
At scoring time the model may read the legitimate query signal on a true cutoff row.
VAL labels never train anything. Confirm and test are not selection data.
If train query labels were absent, or signal counts are not 250/267/251 for -1/0/+1, stop with BLOCKER.md.

## Models

Shared trunk is the CNN03 no-signal encoder + the CNN03 head (target 367109 params).
Soft MoE adds 3 small causal CNN residual heads (project h->32, causal conv k=3, Linear to 5).
Expert output layers are zero-initialized so residuals start at 0.
Gate is an MLP: Linear(2h+32, 64) -> GELU -> Linear(64, 3) -> softmax.
Gate inputs: causal trunk features (last state concatenated with same-time machine mean),
static machine-type embedding, and a 4-way signal embedding (zero-initialized).
Blend is a convex combination of the 5-class softmax probabilities
softmax(shared_logits + residual_k). Rows sum to 1.
Sparse-group rule (frozen, not tuned): a signal group with fewer than 32 labelled TRAIN
cutoff slots disables that expert (gate mass 0, renormalize). If every expert is disabled,
the output is exactly the shared-head softmax (CNN03-style fallback).
Matched control: SharedCNN single head, same windows, epoch counts, batch, lr, wd, clip,
OneCycle phase lengths, and cutoff fine-tune step count. Control ignores the signal tensor.

## Budget (seed 0 only, one process, no relaunch)

Phase H: 30 epochs, all TRAIN windows, signal forced ABSENT, one shared CNN03 trunk.
That single pretrained state is copied into the soft MoE trunk and into the control.
Phase C: 10 epochs, true TRAIN cutoff rows only, ft lr 0.000197 (exactly lr/10, pre-registered, not swept).
The same cutoff index set, batch, weight decay, step count, and torch seed are used for both finetunes.
MoE phase C trains only the 3 experts and the gate; the copied trunk stays frozen and in eval mode.
Control phase C trains only the single head; the copied trunk stays frozen and in eval mode.
Signal values are read only in phase C. Signal embedding rows for real signals get gradient only then.
Entry points reject seed != 0. There is no automatic promotion and no second batch-size run.
After seed 0, record tune R against the matched control and against CNN03 seed 0, then stop.
This is not claimed bit-equivalent to final_cnn03_s0: phase C is extra, and the VRAM cap is 2 GiB
rather than train.py's 0.25 memory fraction.

## Compute

This host only. CUDA_VISIBLE_DEVICES=0 when a later authorized launch happens. VRAM cap 2 GiB
via set_per_process_memory_fraction. 2 CPU threads. One training process. No automatic OOM
relaunch. No tower GPU. No ssh. This prep continuation does not start that process.

## Hash

SHA256 of this file is recorded in PLAN_HASH.txt and RECONCILIATION.md before any metric is written.
Status of this continuation: READY / PREP ONLY. Do not launch GPU training from this session.
