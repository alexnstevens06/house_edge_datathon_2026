# House Edge main round — matched dense/sparse joint transformers + baselines

- Data: `kit/public` (sha256 of `kit/kit.zip`: `06b5246f5cc51db5db26e26aae3264bec7b085d756dd82e5a0577dcba8a44f8a`).
- Official scoring reused from the kit (`kit/slot_forecasting/contest.py`, `scoring.py`); never reimplemented. Organizer CLI `kit/check_submission.py` re-run on every CSV (logs in `runs/verify/checker_*.txt`).
- Validation episodes split deterministically (seed 0, stratified by condition) into TUNE (48 eps, 12/condition) and CONFIRM (48 eps): `runs/splits.json`. Epoch/checkpoint selected on TUNE only; CONFIRM reported once; full-validation is both halves pooled.
- Training used TRAIN split only: sliding windows over observed history (target ticks 16..max, labels = observed payouts at t, unobserved machines masked) + the 768 real train query cutoffs with `labels.csv`. No validation data, targets, or simulator used in training.
- `signal` is NOT a model input (by design; validity for historical windows unknown). Unit-tested: flipping signal never changes predictions.

## Models (matched pair)

Both are JOINT full-attention transformers over ALL tokens of all 8 machines x window ticks in one sequence — not axial/alternating: every input token (tick, machine) plus 8 learned per-machine readout queries are concatenated into a single sequence and encoded by ONE `nn.TransformerEncoder` with no causal or axial masking — `src/house_edge/models.py:53-54` (dense) and `src/house_edge/models.py:62-64` (sparse).

| property | dense | sparse |
| --- | --- | --- |
| tokens | explicit grid of last W=64 ticks x 8 machines (512 tokens), MISSING is a 6th symbol | only observed (tick, machine) positions in [t-64, t), padded + `key_padding_mask` (5 payout symbols + PAD) |
| token embedding | symbol + absolute-tick + machine-slot + machine-type | payout + ORIGINAL absolute tick + machine-slot + machine-type |
| readout | 8 learned queries (slot + machine-type) -> 8x5 softmax | same |
| loss | CE on observed next-tick outcome, machines with unobserved target masked | same |
| causality | all input ticks < target tick, asserted in collate and forward (`models.py:49`, `models.py:59`) | same |

Matched budget: identical seeds (0), splits, steps, optimizer (AdamW lr 3e-4, wd 0.01, cosine to 10%), batch 64, epochs 28, d_model 64, 2 layers, 4 heads, ff 256; parameter counts equal by construction (both 6-row symbol tables).

## Results (validation; lower R is better; R = mean over conditions of episode-averaged squared error vs frozen pooled reference)

| model / baseline | params | tokens/ex | train s | epochs (best) | peak VRAM MB | TUNE R | CONFIRM R | full-val R | pts@60 | pts@80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: pooled_recent | - | - | - | - | - | 0.9455 | 0.9476 | 0.9464 | 30.83 | 41.10 |
| baseline: recent | - | - | - | - | - | 0.9567 | 0.9473 | 0.9517 | 30.74 | 40.99 |
| sparse (v2) | 121157 | 401.22 | 580 | 28 (10) | 1924.9 | 0.9425 | 0.9843 | 0.9629 | 30.57 | 40.76 |
| baseline: shrinkage64 | - | - | - | - | - | 0.9530 | 0.9774 | 0.9646 | 30.54 | 40.72 |
| dense (v1 ablation) | 88133 | 475.38 | 290 | 16 (13) | 1887.9 | 0.9456 | 0.9853 | 0.9651 | 30.53 | 40.71 |
| sparse (v1 ablation) | 88133 | 401.22 | 312 | 16 (13) | 1887.9 | 0.9433 | 0.9893 | 0.9661 | 30.52 | 40.69 |
| dense (v2) | 121157 | 475.38 | 530 | 28 (10) | 1924.9 | 0.9503 | 0.9908 | 0.9702 | 30.45 | 40.61 |
| baseline: pooled | - | - | - | - | - | 1.0000 | 1.0000 | 1.0000 | 30.00 | 40.00 |
| reference (frozen pooled) | - | - | - | - | - | 1.0000 | 1.0000 | 1.0000 | 30.00 | 40.00 |

Baselines: `pooled`/`recent`/`pooled_recent` = kit starter `FrequencyForecaster` methods; `shrinkage64` = per-machine empirical payout distribution over last 64 observed ticks, Dirichlet-shrunk (a=16) to the train-history pooled marginal. All scored on the SAME TUNE/CONFIRM/full splits as the transformers (selection used TUNE only).

Selection by TUNE only (lower is better): **dense → v1** (TUNE 0.9456 vs v2 0.9503), **sparse → v2** (TUNE 0.9425 vs v1 0.9433). The v2 pair is kept as the headline matched pair (identical budgets), the v1 pair as an ablation; all four checkpoints are provenance-checked below. Best full-validation transformer: sparse v2 (R 0.9629, 30.57/60).

## Per-condition loss ratios (full validation)

| model | stable | short_history | drift | shared_change |
| --- | --- | --- | --- | --- |
| dense (v2) | 0.9922 | 1.0390 | 0.9874 | 0.8622 |
| sparse (v2) | 0.9725 | 1.0240 | 0.9764 | 0.8787 |
| dense (v1 ablation) | 0.9892 | 1.0262 | 0.9655 | 0.8796 |
| sparse (v1 ablation) | 1.0021 | 1.0139 | 0.9755 | 0.8728 |
| baseline: pooled_recent (best) | 0.9582 | 1.0006 | 0.9464 | 0.8805 |
| reference (frozen pooled) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Organizer checker (python3 kit/check_submission.py <csv> --split validation --data kit/public)

- dense (v2): validation points **30.453822 / 60** (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/v2_dense/validation_predictions.csv`)
- dense (v2): test CSV format check **VALID** (1536 queries) (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/v2_dense/test_submission.csv`)
- sparse (v2): validation points **30.566738 / 60** (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/v2_sparse/validation_predictions.csv`)
- sparse (v2): test CSV format check **VALID** (1536 queries) (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/v2_sparse/test_submission.csv`)
- dense (v1 ablation): validation points **30.532139 / 60** (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/archive_dense_v1/validation_predictions.csv`)
- dense (v1 ablation): test CSV format check **VALID** (1536 queries) (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/archive_dense_v1/test_submission.csv`)
- sparse (v1 ablation): validation points **30.517375 / 60** (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/archive_sparse_v1/validation_predictions.csv`)
- sparse (v1 ablation): test CSV format check **VALID** (1536 queries) (`/home/alexn/Documents/code/datathon_2026/house-edge/runs/archive_sparse_v1/test_submission.csv`)
- baseline pooled_recent: validation points **30.825854 / 60**
- baseline recent: validation points **30.743191 / 60**
- baseline shrinkage64: validation points **30.540079 / 60**
- baseline pooled: validation points **30.000000 / 60**
- Full checker logs: `runs/verify/checker_*.txt`.

## Verification

- Unit tests: 15/15 PASS (`.venv/bin/python -m unittest discover -s tests -v`, log: `runs/verify/tests.txt`): causality (perturbing any machine's token at/after the target tick leaves outputs unchanged, both archs), forward asserts on tick>=t, sparse preserves ORIGINAL tick values, dense MISSING != payout 0, output shape (B,8,5) with 8 per-machine readouts, rows sum to 1 & finite, unobserved targets masked from loss, deterministic eval, query_id alignment with kit queries.csv, signal not used as input.
- Checkpoint provenance (reload `best.pt` with current code, regenerate validation predictions, compare to saved CSV): see `runs/verify/provenance.txt` — max abs diffs ~1e-7 (GPU attention-kernel nondeterminism; same code, same checkpoint). Code hashes: `runs/verify/code_hashes.txt`.
- Offline CPU prediction smoke (CUDA hidden via `CUDA_VISIBLE_DEVICES=` `HIP_VISIBLE_DEVICES=`, `torch.set_num_threads(2)`, full test split): `runs/verify/cpu_smoke.txt` — CPU_SMOKE v2_dense n_queries=1536 seconds=1.6 threads=2 max_abs_diff_vs_gpu=3.136e-05
CPU_SMOKE v2_sparse n_queries=1536 seconds=2.7 threads=2 max_abs_diff_vs_gpu=3.783e-05
- Peak VRAM measured with `torch.cuda.max_memory_allocated` (in per-run `config.json`); both runs stayed far below the 6 GB shared-machine cap.

## Reproduce (exact commands)

```sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6700 XT (gfx1031) on ROCm
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/train_model.py --arch dense --epochs 28 --dim-ff 256 --scheduler cosine --out runs/v2_dense
.venv/bin/python scripts/train_model.py --arch sparse --epochs 28 --dim-ff 256 --scheduler cosine --out runs/v2_sparse
.venv/bin/python scripts/baselines.py
.venv/bin/python scripts/regen_validation.py --ckpt runs/v2_dense/best.pt --ref runs/v2_dense/validation_predictions.csv --out runs/verify/v2_dense_validation_regen.csv
.venv/bin/python scripts/cpu_predict.py --ckpt runs/v2_dense/best.pt --out runs/verify/cpu_test_v2_dense.csv --gpu-csv runs/v2_dense/test_submission.csv --name v2_dense
.venv/bin/python scripts/make_summary.py
python3 kit/check_submission.py runs/v2_dense/test_submission.csv --data kit/public   # format only
```

Ablation v1 (16 epochs, ff=128, constant lr) kept at `runs/archive_dense_v1`, `runs/archive_sparse_v1`. `runs/dense` contains an aborted 5-epoch duplicate of the v2 config (shell timeout killed it); superseded by `runs/v2_dense`, left untouched.

## Risks / caveats

- Reproduction tolerance: rerunning inference on GPU differs from the saved CSV by up to ~1.2e-7 per probability (attention kernel nondeterminism), above the announced 1e-8 organizer tolerance; CPU-vs-GPU differences are larger (see cpu_smoke.txt). For the real submission, regenerate the CSV with the exact reproduce command on the target runtime and submit that file.
- Competition runtime is Python 3.13 stdlib with deps via requirements.txt; torch availability there is UNVERIFIED (we only tested the local .venv, torch 2.9.1+rocm6.4). If torch is unavailable at validation time, predictions cannot be recreated — a numpy/stdlib fallback would be needed.
- Some machine state is hidden (per RULES.md); ~0.94-0.96 R may be near the observable limit. All four transformers beat the frozen reference and `pooled` on full validation but trail `recent` (0.9517) and `pooled_recent` (0.9464) overall; on `shared_change` they edge out `pooled_recent` (0.862-0.880 vs 0.8805).
- Short-history episodes: the 24 older sampled ticks mostly fall outside the W=64 window by design (spec default); only the recent 40 are guaranteed covered.
- `signal` deliberately unused; if it is informative at cutoffs, some accuracy is left on the table (allowed ablation not run under the no-new-variants constraint).

