# hard_signal reconciliation

Written after tests passed and before any tune R. ADDENDUM (Alex t83, sha256 `e2fa1688d10ebe8dfa942dda8210bb4a57b96f5ad3c528b43851416aedfc314a`) overrides promotion: seed 0 only. `archive_pre_addendum/` was not modified.

## Inherited SHA256 (archive_pre_addendum, byte-compared at 8:45 PM CST)

| file | sha256 |
| --- | --- |
| PLAN.md | `d4530ac9e19163dc38c03e8087af1bac696a2025931376d60ffb1ffcf6a49919` |
| source/hard_signal.py | `46d8686228a8f9ff7522f6d38afe2b07ffb87f4ad83187c24c965978d7a7b88b` |
| source/run_seed.py | `db01da73e8de634fbaea11235fc5be7df8251875cc3ee7c8317dfbee30ad618c` |
| tests/test_hard_signal.py | `78106ade2d7d85936228dacf86e47a6b0f71c3ebb2854b4962a52e79485d4d5d` |

`SPLIT_IDS.json` was already present and was not archived. sha256 `57e04dc8963efcadd33285cdd08e8c886302233948e35ae8c0f3ea5fb4a645e1`. Its `tune` and `confirm` lists equal `runs/splits.json` (48 ids each). Scoring uses `tune` only.

## Corrected SHA256 (frozen before metrics)

| file | sha256 |
| --- | --- |
| PLAN.md | `36017f2f667ae60095c75babb137a9a56f36acca1c008619b5cf3aa10e56dafe` |
| SPLIT_IDS.json | `57e04dc8963efcadd33285cdd08e8c886302233948e35ae8c0f3ea5fb4a645e1` |
| source/hard_signal.py | `23147fc43ac7a9ad6defa9546f7dccfca07f9e9c50dbdeb74b38716246dc3465` |
| source/run_seed.py | `9b86c3a5b47c32ae1bd611a40c32c5978ce535f32575eae95e5e64b734510af3` |
| tests/test_hard_signal.py | `79dff23e1dee83980d4deefdd8c9a9511ddcfeda029939fa9dbaa654f8601ab6` |

Canonical, read-only:

| file | sha256 |
| --- | --- |
| experiments/mlp_cnn_hpo/source/base_cnn.py | `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5` |
| experiments/mlp_cnn_hpo/source/train.py | `916c6ee1b6e405300ddadbc5650583b250db8baa83a5227fc2437c1b1a55e066` |

`base_cnn.py` matches the contract hash. CNN03 seed 0 config: h 128, layers 4, kernel 3, W 64, drop 0.3, epochs 30, lr 0.00197, wd 0.00895, bs 512, seed 0, phase final. Published tune R 0.33206398675779764.

## What was inconsistent

- `hard_signal.py` trained with 30 fine-tune epochs at lr/10, allowed seeds 0/1/2, and scored confirm, full, and test.
- `run_seed.py` and `tests/test_hard_signal.py` expected `phase_hyper`, `Examples` batches, `train_phase`, `assemble_slots`, and `predict_routed`. Those names were not in `hard_signal.py`. `run_seed.py` also computed the repo root one directory too high.
- The inherited test helper required a second argument that its own calls did not pass. The label-mutation loss check used all-zero logits, which makes every class the same cross-entropy.
- The inherited plan and test said 30 fine-tune epochs. The directed budget is 30 history epochs plus 10 cutoff epochs.

## Fixes

- One trainer API. `run_seed.py` only checks the seed and delegates to `train_seed0`.
- Budget is 30 no-signal history epochs and 10 cutoff epochs. Control and specialist fine-tunes share lr 0.00197, wd 0.00895, effective batch 512, and seed 0. Both load the same pretrain weights. Specialists do not load the control.
- Entry points raise `rejected seed N` for any seed other than 0, and reject `--report-seen-confirm`. There is no seed 1/2 queue and no OOM retry.
- Evaluation is tune-only, CPU fp32 softmax stored as float64, same as `base_cnn.predict` on CPU. CNN03 final scoring in `train.py` is also CPU fp32. Confirm and test are not scored.
- `_data` takes an optional labels flag. The label-mutation loss uses non-uniform logits.

## Comparison with CNN03, not claimed equal

| item | CNN03 `train.py` + `base_cnn.build_train` | this run |
| --- | --- | --- |
| architecture | `CNN(..., use_signal=False)`, 367109 params | same class and flag |
| signal in features | `USE_SIGNAL=False`, so `SG` is zeros even on cutoff rows | history builder does not read queries; cutoff signal is a mask/router only; the model is passed zeros |
| windows | `t` from 16 through `cutoff` inclusive, and query labels are written into `full[cutoff]` | pretrain is `t < cutoff` only; cutoff labels are a separate 10-epoch stage |
| dtype | bf16 autocast on CUDA, fp32 loss via `lg.float()`, final eval CPU fp32 | same train autocast and CPU fp32 eval |
| steps | one backward per 512 rows, 30 epochs, memory fraction 0.25 | mean gradient over each 512-row chunk from microbatches of 128, 2GiB cap; dropout RNG is per microbatch |
| metric | full, tune, and confirm | tune only |

Optimizer steps match between the control and each specialist (10 steps, 96 windows). Labeled-slot exposure does not: control uses all cutoff labels; each specialist uses only its signal (250, 267, or 251). That is the intervention.

## Tests

Command:

```
cd /home/alexn/Documents/code/datathon_2026/house-edge
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
PYTHONPATH=experiments/random_forest/.venv/lib/python3.12/site-packages \
.venv/bin/python -m pytest experiments/cnn_experts/hard_signal/tests/test_hard_signal.py -q --tb=line
```

Result: `13 passed in 3.10s` at 2026-11-07 21:23:23 CST. No package was installed.

Covered: all-8-machine future mutation, label mutation, historical features independent of query signal, query order, missing versus 0, exact hard route, sparse fallback, shared train phase, seed and confirm-flag rejection, and TRAIN signal counts 250/267/251 with `t < cutoff` on the real history build.

GPU at that time: 16303 MiB total, 2 MiB used, 15841 MiB free, no compute process.
