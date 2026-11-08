# soft_moe plan (FROZEN before any metric)

Experiment (B). Comparison target is incumbent CNN03, not a number produced here.

## Incumbent (already published; lower R is better)

CNN03: h=128, L=4, kernel=3, W=64, dropout=0.3, lr=0.00197, wd=0.00895, batch=512, 30 epochs, AdamW, OneCycle, grad clip 1.0, bf16 train / CPU fp32 eval. Source `experiments/mlp_cnn_hpo/source/base_cnn.py` sha256 `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5`. Tune R: seed0 0.33206398675779764, seed1 0.34654552298986674, seed2 0.33342647148315785.

## Data

- Read-only `kit/public/{train,validation,test}` via `house_edge.data.load_split`.
- Payout class order 0, 1, 2, 5, 20. Missing mask is -1, distinct from payout index 0.
- Machine type map (kit `TYPE_TO_IDX`): balanced=0, jackpot=1, steady=2. Slots are sorted `machine_id`, not query-row order.
- Splits: `runs/splits.json` keys `tune` and `confirm`, copied into `SPLIT_IDS.json` before training. Confirm is SEEN. It is scored only as a labeled seen number, and only quoted in RESULTS after promotion.
- No VAL fitting, no early stopping, no synthetic data, no HPO.

## Leakage

- History windows are ticks `t` with `16 <= t < cutoff` (stride 1). Features use only ticks `< t`. Labels are the observed history payout at `t`, never query labels.
- Query signal is not read while building history. History signal channel is the constant ABSENT code 3. `is_cutoff` is 0.
- True TRAIN cutoff rows (`t == cutoff`) are a separate stage. Features still use ticks `< cutoff`. Labels are `train/labels.csv` only. Signal {-1,0,1} -> {0,1,2} is attached only on these rows, and only for the soft model input.
- The gate replaces the signal id with ABSENT whenever `is_cutoff` is false, so a leaked signal tensor cannot enter historical forwards.
- VAL/TEST cutoff rows may receive their own query signal at prediction time (observable input). Their labels are never read for fitting.

## Architecture

Shared causal CNN history encoder, CNN03 widths, no signal parameters: payout embedding (6, missing index 5) + mask linear + type embedding + slot embedding + tick linear (absolute/256 and recency) + 4 causal conv blocks (dilation 1,2,4,8, kernel 3) with the same cross-machine mix as CNN03.

Shared logit head: Linear(2h, h) + GELU + Linear(h, 5) on `[last timestep, mean over machines]`.

Soft model adds 3 residual CNN experts. Each is Linear(h, 32), one causal conv block (dilation 1, kernel 3, width 32), then Linear(64, 32)+GELU+Linear(32, 5). The last linear is zero-initialized, so a fresh expert residual is 0.

Soft MLP gate, per machine: concat(last hidden h, pooled last hidden h, type embedding 16, signal embedding 16, cutoff flag). MLP  (2h+33 -> 64 -> 3) + softmax. Signal embedding sees ABSENT off cutoff.

Blend is on probabilities, not logits: `p = sum_k g_k * softmax(shared_logits + residual_k)`. `g` is nonnegative and sums to 1.

Sparse-group fallback: an expert keep-mask (default all true) drops experts and renormalizes `g`. If no expert remains, `p` is exactly `softmax(shared_logits)` from the shared head (CNN03-style single head on the same encoder state).

Matched control: the same encoder + shared head only. It is trained by the same function, rows, epoch counts, batch size, optimizer, and schedule. Its forward ignores signal, cutoff flag, and expert mask.

## Budget (both models, every seed)

1. Pretrain 30 epochs on history windows, no signal. AdamW lr 0.00197, wd 0.00895, batch 512, OneCycle over that phase's steps, grad clip 1.0, bf16 autocast on CUDA, 2 CPU threads.
2. Fine-tune 10 epochs on true TRAIN cutoff rows only (new AdamW + OneCycle, same lr/wd/batch/clip). Soft model receives legitimate signal. Control receives the same rows and ignores signal. All parameters train in both phases (gate included in phase 1, without signal).
3. No validation inside the loop. Then one CPU fp32 eval (2 threads, no autocast) of validation and test queries.

Loss: NLL of the blended (or control) probabilities on slots with `y >= 0`. Missing targets contribute nothing.

## Compute

Server GPU only, detached. One process per seed runs control then soft, so one GPU job. Cap `torch.cuda.set_per_process_memory_fraction` at 2/16. Abort if `nvidia-smi` free memory is under 2500 MiB. Do not start a third concurrent job.

## Promotion

After seed 0's full budget (both models): if soft tune R is strictly lower than incumbent seed 0 (0.33206398675779764), launch seeds 1 and 2 the same way. Else DROP and stop. A keep requires 3/3 strict wins versus the matched incumbent seed. Confirm numbers stay labeled seen.

## Tests (before any training metric)

all-8-machine future mutation invariance; label mutation; signal mutation does not change historical features; query slot order invariance; missing vs 0; soft gate normalized; sparse-group fallback equals shared head; control shares the pipeline and ignores signal.
