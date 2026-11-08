# Frozen review, t146

Written 2026-11-08T02:27:31 CST (2026-11-08T08:27:31Z). Same independent reviewer. `PRELIMINARY.md` at 2026-11-08T01:44:59 CST stays the record of the unfrozen tip and is not rewritten. This file reviews the frozen source and the completed artifacts that exist now.

Scope is read-only except this review directory. No training, no tower paths, no probability fitting, no edits under `lowrate_20261108`, `ensemble_blend_20261108`, or `comparative_t146_20261108`. Confirm, full, and test metrics were not computed. The official validation checker was not called. Production `PINNED.json` and `fp64_harness.py` were hashed and left unchanged. The stale harness-diff non-issue is not reopened. Current-best ZIP `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29` / CSV `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6` and fallback ZIP `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` / CSV `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` still match those hashes.

Interpreter for every command below: `experiments/cnn_runtime/py313_verify_1936/env/bin/python` (CPython 3.13.13, torch 2.11.0+cpu). Environment: `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` empty. `python -m pytest` is unavailable in that interpreter (`ModuleNotFoundError: No module named pytest`). Ensemble tests were called in process. Pytest was not installed.

## Verdicts

| Component | Verdict |
| --- | --- |
| Frozen low-rate source hashes vs `SOURCE_SHA256.txt` | PASS |
| Current trainer call shape (the old five-argument `TypeError`) | PASS, resolved |
| Server 2e-5 primary and control artifacts, 730 steps, CPU FP64 rescore | PASS |
| Tower 5e-5 jobs as executed results | BLOCK |
| Claim that the frozen trainer is portable to Tower | BLOCK |
| Replay exclusivity, host-to-device binding, forward-count refusal, TF32 abort | CONCERN |
| Six curve parents, hard-control bytes, seeds 0/1/2 canonical maps | PASS |
| `ensemble_equal_s0_s2` candidate correctness | PASS |
| `ensemble_equal_s0_s1_s2` candidate correctness | PASS, not selected |
| Blends that include the shared control | CONCERN, not cleared |
| Comparative selection of `ensemble_equal_s0_s2` | PASS for the completed-row rule |
| Package delivery | NOT CLEAR |

**Packaging clearance.** `ensemble_equal_s0_s2` is CLEAR as the best verified completed candidate for a later packaging proof. It is NOT CLEAR as a finished package. No fresh ZIP was built in this review. `comparative_t146_20261108/COMPARISON_READY.md` (2026-11-08 02:24 CST) selects the same row and records `package_written: false`. This review did not edit that folder. The comparison runner was not re-executed. The deciding tune scores, parent pins, and server rescores were measured here.

This clearance is candidate correctness of a completed row. It is not a claim that the ensemble beat the six unrun Tower 5e-5 checkpoints. It is not organizer acceptance and not a production replacement. The original improvement rule still applies: lowest CPU FP64 tune R among eligible completed rows, package only if the improvement versus current best `0.30980333573689856` is at least 1e-6. Decimal subtraction of the printed tune values is `0.02512157652770842`. The IEEE subtraction printed by `score_rest.py` is `0.02512157652770841`. Both are far above 1e-6. The next completed row, equal seeds 0+1+2, is worse by about `0.00506826686381240`. The best server single model, primary epoch 3 at `0.30706943777760692`, is worse by about `0.02239`. The single-model tie preference does not apply.

Incomplete Tower training is a separate fact. `assert_input_pins` correctly refused to start those jobs. The protocol says to stop the assigned jobs on input-source incompatibility. That refusal does not erase the completed ensemble, and it does not authorize scoring the missing rows as worse or expanding the matrix.

## What this review ran

```
experiments/cnn03_convergence/lowrate_20261108/tests/test_lowrate.py
experiments/cnn03_convergence/independent_review_20261108/t146_review/run_ensemble_tests.py
experiments/cnn03_convergence/independent_review_20261108/t146_review/probe_frozen.py
experiments/cnn03_convergence/independent_review_20261108/t146_review/score_rest.py
experiments/cnn03_convergence/comparative_t146_20261108/tests/test_select.py
sha256sum of the ensemble PLAN, MANIFEST, SPLIT_IDS, INPUT_PINS, common.py, evaluate.py, test_matrix.py, both COORDINATOR_PLAN.md copies, and lowrate SOURCE_SHA256.txt
```

Results:

- `tests_lowrate.txt`: `LOW_RATE_TESTS_OK failures=0 windows=18567 steps=730`, exit 0. The pytest import error above that line is the missing module, before the low-rate script.
- `tests_ensemble.txt`: nine `PASS` lines, `ENSEMBLE_TESTS failed=0 ran=9`, exit 0. Coverage includes pinned files and no float-cast/autocast in the ensemble sources, wrong hash/config/keys/shape/signal/dtype/nonfinite, hard-control bind, unknown and non-tune ids, no future window and missing is not payout 0, query order does not change ids, simplex, weight-0 identity, arithmetic mean, unlisted weight 0.3 refused, one real seed-2 float64 logit/softmax episode, and the split file versus the manifest contract. The real seed-2 test truncates validation to one episode. It does not score the study matrix.
- `probe_frozen_out.txt`: `PROBE_FROZEN_OK`, exit 0, at `reviewed_at_utc` `2026-11-08T08:20:53Z`. Report: `probe_frozen_report.json`.
- `score_rest_out.txt`: `SCORE_REST_OK`. Stored equal-3, seed 1, and the five blend npz files. Simplex errors 0. Tune 384, forward 768.
- `tests_select.txt`: `SELECT_TESTS_OK`, exit 0. These tests use synthetic rows. They do not score a model.

`evaluate.main` and `score_fp64.main` were not called. Both write author results. Scoring used `score_fp64.score_checkpoint` and `common.score_tune_detail` / `forward_probabilities`. The FP32 GPU preflight was not rerun. Reviewer RSS for the forward probe was not recorded, so this review does not claim that probe stayed under 2 GiB. The comparison note records 666848 KB for the comparison author's own CPU run.

## Source hashes

`lowrate_20261108/SOURCE_SHA256.txt` matches the file bytes with zero mismatches. That list file itself hashes to `554afbdfa3ea2293366942f15282ce309f387fbe19b634183d4f81908085f021` and is not a line inside the list.

| File | SHA-256 |
| --- | --- |
| `lowrate_20261108/COORDINATOR_PLAN.md` | `03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3` |
| `ensemble_blend_20261108/COORDINATOR_PLAN.md` | same `03d141087…f9e8f3` |
| `lowrate_20261108/PLAN.md` | `463ea1dd6317a0fe64e009d8b32fd751ef756cc5fe0c7c102af1c4474f606b3e` |
| `lowrate_20261108/MANIFEST.json` | `5d1963aab728ec0d0afdcfb6434c5614f0199453de0d70ffd84d0cbf6dd4e463` |
| `lowrate_20261108/SPLIT_IDS.json` | `21d0b34335349e8348d9a3a0edc578e063b08bc7e80e719fd2c93b4139dd566a` |
| `source/contract.py` | `d908aefe6eab70cbedf70d0bf2b7fc2d2132cd5bafce7e3d844fc78cccaa3025` |
| `source/train_lowrate.py` | `206ee996bc6a48b34a5e4027e90542e5cb63a787fe36e7ab7aab1f90f0b04066` |
| `source/score_fp64.py` | `138676a7fbca1be71b9348c85aac2eed444c667b5d899a2b017df01d20e463d4` |
| `source/preflight_fp32.py` | `ffdfd3d00dcccf216db8c3712efaadc13ccb77d844c7a17afb74b88b7893bd15` |
| `source/preflight_fp64.py` | `adb42b4c65545f6d2ac89062a49f87a22ff32ff59f702b7312dad22840008325` |
| `tests/test_lowrate.py` | `b131bcc4caf8ea1b7bce442ab93ff053a0d584eb19a011ed4a37743aadf62c0d` |
| `ensemble_blend_20261108/PLAN.md` | `7c48cdef610d3985b6f7cf07c23402cf0e624457aeffc47d7c16ab9af4fcc74f` |
| `ensemble_blend_20261108/MANIFEST.json` | `f966138c7a44b7bfa7c200e42f2f0e85528c2cc54f97c754a8dc1e1b2300368f` |
| `ensemble_blend_20261108/SPLIT_IDS.json` | `75ba8ee3202102b859a28aa3ade33dbf947ce82d0ecc11a2cbfc6f92925cd356` |
| `ensemble_blend_20261108/INPUT_PINS.json` | `43e06ec083e3203625c9bf9791d6f203e23c0013b643e53214c0aa2e0f8a8b98` |
| `ensemble_blend_20261108/source/common.py` | `7b53b7f7a77193bbc844de5fb28672d07081cfae29185d691790df1c89cd20ec` |
| `ensemble_blend_20261108/source/evaluate.py` | `27a8ca737ec247809f13a300560bfa8603493f7d703a19c2e167817ee704de75` |
| `ensemble_blend_20261108/tests/test_matrix.py` | `25e5e46dcd54da1dda992c8108b16bce667fb4d27c05a3826b4c4409b989fdd4` |
| canonical `fresh_fp64_cnn03_1956/model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` |
| `PINNED.json` | `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb` |
| `fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` |

`COORDINATOR_PLAN.md` was read in full. Eligible rows are the current best, the two predeclared original ensembles, the five original seed-2 / shared-control blend weights, and epoch 1/3/5 from the four low-rate jobs. Lowest CPU FP64 tune R wins. Improvement under 1e-6 is a tie and keeps the current best. A tie then prefers a single model over an ensemble or blend, then lower learning rate, earlier epoch, fewer components, and smaller blend weight. Unrun rows stay unrun. An incompatible input source stops that host's jobs.

## Prior concerns

The preliminary tip trainer `fd0b5052…` called `model(*batch)` with five tensors. That file is not the frozen trainer. Frozen `train_lowrate.py` `206ee996…` contains `logits = model(*batch[:-1])`. The probe flags are `four_arg_call` true and `five_arg_call` false. `test_lowrate.py` passed on this file, including the 18567-window count, the 730-step budget, the refusal of train LR 0.003, the seed-after-load order, and the ban on OneCycle, autocast, and `--lr`. AdamW is constructed with `lr=host_lr(host)`, not the parent `config.lr`.

`gpu_compute_rows` now has both `nvidia-smi` and `rocm-smi`. That does not make the Tower launch succeed. `assert_input_pins` in `contract.py` hashes `REPO / spec["path"]` for protocol sources and for all six original `experiments/cnn03_convergence/runs/.../checkpoint.pt` parents, plus the canonical FP64 tree named by the manifest. `HANDOFF_READY.md` tells Tower to stage the two seed-2 parents under `ensemble_blend_20261108/inputs/` and pass those paths to `--parent`. `parent_errors` can accept those bytes. `assert_input_pins` still runs against the original repo paths before any GPU step. Those paths are absent on Tower. The same bytes exist in the approved ensemble inputs and the original server `runs/` tree. This review did not copy them, did not create fake paths, and did not edit the guard. `lowrate_20261108/runs/` contains only `server_primary_lr2e-5` and `server_control_lr2e-5`. No Tower checkpoint is claimed.

These guards remain, and they do not discard the completed server pair:

- Output is not exclusive. `dest.mkdir(exist_ok=True)` and `loss.jsonl` opened with mode `"a"`. No `O_EXCL`. Each completed `loss.jsonl` has exactly five lines and each `train.exit` is `0`, which is the evidence of one pass, not of a replay lock.
- `--host` selects the learning rate and is not checked against the device name. Both launch files record `NVIDIA GeForce RTX 5080` and `train_lr` `2e-05`, which matches the server assignment. The recorded device is after-the-fact.
- `score_fp64.py` refuses a tune set other than 384 ids and checks those ids against the official tune targets. `n_forward_queries` is `len(preds)` and is not refused when it is not 768. Every stored and rescored row in this review is 768.
- TF32 is set false and recorded. Both server launch files have `matmul_allow_tf32` false and `cudnn_allow_tf32` false. The trainer does not abort if the flag fails to stick. This review did not rerun the GPU preflight.
- The low-rate tree has no selector that applies the cross-stream 1e-6 rule. Ensemble `evaluate._rank` ranks only that stream and omits low-rate rows. The comparison selector now exists and its synthetic tests passed. Observed gaps among completed rows are much larger than 1e-6, so the winner does not depend on the tie branch.

`POST_RUN_SNAPSHOT.json` is written by the trainer before scoring. The scorer requires it. That flag is a declaration in the run directory, not an external lock. Checkpoint bytes match the SHA recorded in each score JSON.

## Server 2e-5 pair

Both jobs exited 0 on RTX 5080. Original PIDs in the launch files are 716396 (primary) and 716397 (control). Each summary has `total_steps` 730. Five epochs, 146 steps per epoch, weights-only parent, fresh AdamW, FP32 batch 128. Confirm, full, and test fields say `not_computed`. Independent `score_checkpoint` matched every saved tune R with `matches_saved` true and `n_forward` 768.

| Row | Tune R | Checkpoint SHA-256 |
| --- | --- | --- |
| primary epoch 1 | 0.30951549616992924 | `6fea985c054638ae852b59685fdbf6e7d43c44b24fa8b1db31f46ae0a577687c` |
| primary epoch 3 | 0.30706943777760692 | `17d66a052d5ccdef50af14468c24a029f77fe3943bd7f991aafcf667c7faedb5` |
| primary epoch 5 | 0.30781275392021723 | `7b8368b5a6684b2fbc65b97ecd70e255fd626840da03d06feb34664c50f0954e` |
| control epoch 1 | 0.33413621090622814 | `ad6651670b2557044c205668d6f11700b718df82d74d10db151219d96facf7cc` |
| control epoch 3 | 0.33213683848995773 | `2c250ec33982b897179ff50c832a7fefe60363e540ebdd53e4ae15046db6e4fa` |
| control epoch 5 | 0.33244487608328904 | `1407fc091fe49f30ff9e1bb8197c56aadeb7119aa7c4530254019c4f6f8a88ae` |

The `.17f` print of primary epoch 3 is `0.30706943777760692`. The JSON dump in the probe report shortens that same matched float to `0.3070694377776069`. `matches_saved` was true before the dump.

Parents used for these two jobs: `curve_C_lr003_s2` `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` and `curve_control_s2` `47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9`.

Unrun, and not ranked as worse:

- `tower_primary_lr5e-5` epochs 1, 3, 5
- `tower_control_lr5e-5` epochs 1, 3, 5

Two of four low-rate jobs completed. All four did not.

## Ensemble maps and the control gap

Copied checkpoints equal both the pin and the original bytes:

| Role | SHA-256 |
| --- | --- |
| curve C seed 0 | `d5d714649a93e1e552ccf5419cb01adee311c082af1063f80d8b42f6d8dfe3cb` |
| curve C seed 1 | `8a528179ad626fd3dede0989d047991449808b0feca9032cbda6729ceb07ce74` |
| curve C seed 2 | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` |
| curve control seed 0 | `30e4152896710be79b7f842118c4d36bf525f3c2a13205fd7c78abefb534c32b` |
| curve control seed 1 | `dbe84aafea436b6a05462dc436bb1d39868f85e2e684d4b0efa9b7b9a485ca05` |
| curve control seed 2 | `47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9` |
| hard shared control | `92233992aeb8c007a40f925885d7f85ef260cf8584ea0cc669dcd39ce57d201f` |

Fresh canonical FP64 forwards, `net.double()`, float64 softmax, then renorm:

| Comparison | Max abs |
| --- | --- |
| fresh seed 0 vs stored npz | 9.992007221626409e-16 |
| fresh seed 2 vs stored npz | 9.992007221626409e-16 |
| fresh hard control vs stored npz | 1.4432899320127035e-15 |
| fresh equal mean of seeds 0 and 2 vs stored `ensemble_equal_s0_s2` | 5.551115123125783e-16 |
| stored equal mean 0+2 vs stored npz | 0.0 |
| stored equal mean 0+1+2 vs stored npz | 0.0 |
| weight 0 vs seed 2 | 0.0 |
| rebuilt blends 0.025, 0.05, 0.1, 0.2 vs stored npz | 0.0 |

Independent tune scores, 384 tune queries, 768 forward queries. Fresh seed 2 and the fresh 0+2 mean match the stored scalars.

| Row | Tune R | Shared control |
| --- | --- | --- |
| seed 0 stored | 0.31019816453387611 | no |
| seed 1 stored | 0.34520741683573092 | no |
| seed 2 stored and fresh | 0.30980333573689856 | no |
| `ensemble_equal_s0_s2` stored and fresh | 0.28468175920919014 | no |
| `ensemble_equal_s0_s1_s2` stored | 0.28975002607300254 | no |
| blend w=0 | 0.30980333573689856 | identity of seed 2 |
| blend w=0.025 | 0.30693690199632961 | yes |
| blend w=0.05 | 0.30441273225560084 | yes |
| blend w=0.1 | 0.30039118477366389 | yes |
| blend w=0.2 | 0.29645525780787213 | yes |
| canonical FP64 hard control | 0.46211190598948326 | the control itself |

Seed 1 was scored from the stored npz. This review did not fresh-forward seed 1. The 0+2 proof uses fresh forwards of seeds 0 and 2 only. `ensemble_equal_s0_s2` is tune-informed: the protocol predeclared it after seed 2 was already the best single model and seed 1 the worst. That label has to travel with any later package. No other components, weights, or fine-tune ensembles were scored or authorized.

Historical replay of the same hard-control bytes through `hard_signal.predict_single` (CPU FP32 module and FP32 softmax, tune score only) returned `0.46211193532631023`. Absolute delta versus the published historical scalar is 0.0. Max abs versus the stored canonical FP64 npz is `3.6231444022760684e-07`. The scalar gap between canonical `0.46211190598948326` and historical `0.46211193532631023` is `2.9336826967707452e-08`, about `2.934e-8`, which misses the study's 1e-12 reproduction gate. The per-query gap also misses the later 1e-8 packaging map gate. Probabilities were not rescaled.

`evaluate.py` hard-stops if a non-control role misses 1e-12. For the control only, it records the gap and still writes the blend rows. The source comment in that branch says an FP32 softmax path was also outside 1e-12. This review did not repeat a float32 softmax on the canonical windows. `COMPARISON_READY.md` reports that a float32 forward and softmax on those windows matches the historical scalar with delta 0, and that its float64-versus-float32 map max abs is `3.655700952509555e-07`. That is the comparison author's measurement. It is consistent with the direction of this review's historical-versus-canonical gap and it is not a number this review recomputed. The two max-abs figures differ in the third significant digit. Both are about `3.6e-7` and both fail 1e-8. Neither measurement adjusts the stored probabilities.

Consequence: blend rows are internally consistent with the unmodified canonical FP64 control map, and several beat the current best. They are not cleared. The ensemble plan's reproduction gate failed for the control component, and the per-query precision gap is larger than 1e-8. `ensemble_equal_s0_s2` does not contain that map. Its pin and precision proof is the seed-0 and seed-2 canonical evidence above.

## Comparison author

`comparative_t146_20261108` now contains `COMPARISON_READY.md`, `results/comparison.json`, `PINS.json`, `SOURCE_SHA256.txt`, `source/ranking.py`, `source/compare_runner.py`, and `tests/`. This review read them and ran `tests/test_select.py` only. It did not run `compare_runner.py`.

The comparison note's scored tune table matches the rows measured here, including the six server rescores, the two ensembles, weight 0, and the current best. Their recomputed blend w=0.2 display is `0.29645525780787219` against this review's score of the stored npz, `0.29645525780787213`. They report the maps still agree within about 1e-15. That display gap is about 6e-17 and does not change the order. Their selection is `ensemble_equal_s0_s2`, improvement about `0.02512157652770841`, `would_request_package` true, `package_written` false, and the six Tower rows `not_run_preflight_blocked`. This review agrees with that selection among completed rows and agrees that the blocked rows carry no score.

Their full per-query normalized comparison of every fresh map, including seed 1, is their result. This review's fresh per-query checks are seeds 0 and 2, the hard control, the fresh 0+2 mean, and exact stored arithmetic for equal-3, weight 0, and the four positive blend weights.

## What a later package still has to prove

Clearance of `ensemble_equal_s0_s2` for packaging proof is not delivery. Still required, and not done here:

- An exact fresh ZIP of that ensemble only, built after this approval, with no silent addition of blends or low-rate checkpoints.
- Offline execution at 2 CPU threads and at most 2 GiB, network none.
- A cold repeat, schema check, and cross-host normalized per-query max abs within 1e-8 on the packaged maps.
- An independent package PASS after that ZIP exists.
- The tune-informed disclosure on the package.
- The dependency-contract conflict from the earlier candidate review (kit text versus the cached guide) remains unresolved. This review does not invent precedence.
- Production pins and both existing release ZIP/CSV pairs stay immutable.

No tool gate fired. No path was bypassed.
