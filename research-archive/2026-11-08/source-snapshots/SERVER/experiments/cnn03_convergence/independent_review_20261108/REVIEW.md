# Independent review — CNN03 higher-LR continuation study

**Verdict: CONCERN**

Reviewed at 2026-11-08 06:09–06:17 UTC (00:09–00:17 CST). This is a study-code and scoring review. It does not clear the production FP64 release, does not change `fp64_harness.py` or `PINNED.json`, and does not select a continuation candidate.

The tune metric, canonical float64 module, row normalization, six parent checkpoints, and warm-start contract check out. The study scorer still does not reject a mismatched checkpoint hash, and the shipped tests never exercise that rejection.

## What was checked

Read-only inputs: `continuation_20261108/PLAN.md`, `MANIFEST.json`, `source/*.py`, `tests/test_study.py`, plus the launch script and the artifacts those programs had written. No training was started. CPU probes used 2 threads and `CUDA_VISIBLE_DEVICES=""`. Writes stayed in this review directory.

Production guard, measured, not modified:

| File | SHA-256 | Result |
| --- | --- | --- |
| `experiments/cnn_runtime/fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` | matches `MANIFEST.json` |
| `experiments/mlp_cnn_hpo/review/PINNED.json` | `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb` | present, 486 bytes; study code does not open it |
| `experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip` | `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` | unchanged |
| `experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` | matches manifest and the anchor model hash |
| `experiments/cnn_runtime/server_fp64_cnn03_val1958.csv` | `ef71ee39f668be0ee3f6e20215390f3dd0ab5da6a14199406c590334947b6aa6` | matches manifest |

Study source does not import `fp64_harness` and does not assign `Tensor.float`.

## Passed

**Six parents are weights-only and match the pin.** `verify_checkpoints.check_arm` accepted each manifest entry and rejected a substituted SHA-256. Each blob is exactly `{config, state}`: 43 floating tensors, no optimizer, scheduler, or RNG. Learning rates are 0.003 on `curve_C_lr003_s*` and 0.00197 on `curve_control_s*`. Seeds are 0, 1, 2. `pct_start` is 0.3. `check_state` rejected a shortened `head.2.bias`, a missing `pay.weight`, and an added `sig.weight`. Config files have no `kernel` field; kernel 3 is enforced by the conv weight shape `(128, 128, 3)`.

Exact optimizer resume is impossible. `continue_train.py` refuses a parent hash or key mismatch, builds a new `CNN(..., use_signal=False)`, loads `state` only, then starts a fresh AdamW at the saved learning rate and a fresh `OneCycleLR` over `extra_epochs * 37` steps (`pct_start=0.3`). Continuation seed is `1000 + original_seed + extra_epochs` and is applied after the weight load. The +10 and +20 runs both name the original parent. All 12 saved continuation checkpoints that were on disk by 06:17 UTC store `warm_start_from=weights_only`, `parent_had_optimizer=false`, and the original parent SHA-256. Their own `state` still has 43 tensors. Saved steps are 370 for +10 and 740 for +20, with the arm learning rate unchanged. Those files also store the new run's optimizer, scheduler, and RNG, which the plan allows and which does not turn the parent into an exact resume.

**Tune scoring uses the canonical float64 module and the tune split.** `score_fp64.py` loads `fresh_fp64_cnn03_1956/model.py`, calls `model.double()`, and uses that module's `predict`, which keeps softmax in float64. The canonical module and `base_cnn.py` have the same 43 state keys; `load_state_dict` from one into the other succeeds. `base_cnn.forward` still calls `.float()`. The study scorer does not use that training forward.

`runs/splits.json` has 48 tune episodes and 48 confirm episodes, overlap 0. Validation has 768 queries: 384 tune and 384 confirm, and both subsets contain `stable`, `short_history`, `drift`, and `shared_change`. On the reference CSV, this review measured:

| Subset | R |
| --- | --- |
| tune, study `score_tune` | 0.33206398205344967 |
| tune, direct `score_predictions` | 0.33206398205344967 |
| confirm | 0.40326896128145046 |
| all validation episodes | 0.36845453658648775 |

Tune is not confirm and not the full validation score. The same function recomputed every parent and the incumbent. Each value matched `runs/fp64_original_tune.json` exactly. Incumbent tune R equals the reference CSV tune R. `w=0` rows in `runs/blends_tune.json` are that same number. Author tests `test_renorm_matches_canonical_writer` and `test_reference_csv_study_score_matches_direct_call` passed.

`canonical_renorm` is `float` then `sum`, the same reduction as `house_edge.scoring.write_submission_csv`. `score_predictions` then goes through kit `probabilities`, which requires values in `[0, 1]` and a sum within `1e-6` of 1, and divides by `math.fsum`. That second step is the kit tolerance renorm named in the plan.

Author test command, from the repo, with bytecode writes disabled:

`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/cnn03_convergence/continuation_20261108/tests/test_study.py`

Exit 0. Six tests passed.

## Findings

**1. High — the scorer does not reject a bad checkpoint hash.** `predict_checkpoint` and `score_path` call `load_curve_blob` and `check_state` only. They never compare the file SHA-256 with the manifest. `score_saved.py` globs `runs/*_plus*/checkpoint.pt`, requires 12 files, and calls `score_path`. It does not read `parent_checkpoint_sha256`, extra epochs, or learning rate. A file with the right 43 tensor shapes is scored and labeled as that run. Hash rejection exists in `verify_checkpoints.py` and in `continue_train.py` before training. It is not on the scoring path. The parent tune numbers above are still the pinned files, because those hashes were checked and the forwards were rerun. A later `score_saved.py` run does not have that binding.

**2. Medium — the tests do not exercise rejection.** `tests/test_study.py` checks a fresh canonical `state_dict`, a synthetic weights-only dict, the extra-epoch cap, writer renorm, the reference CSV against a direct tune call, a second import of the same canonical file, and a hand-rolled zero-weight mix. Nothing in that file expects `RuntimeError` for a wrong hash, a wrong shape, a signal tensor, or a non-tune id set. The six passes do not cover finding 1. Negative behavior was checked only by this review's probe.

**3. Medium — `n_queries` and the confirm guard do not describe the scored set.** `score_path` sets `n_queries` to `len(preds)` after a full validation forward. That is 768. The R is computed on 384 tune queries. `score_tune` raises only if the returned dict has a top-level `confirm` or `full` key. `score_predictions` returns `R`, `points_60`, `points_80`, `probability_error`, `reference_error`, `loss_ratio_by_condition`, and `episodes`. The guard cannot fire. The episode filter itself is doing the restriction.

**4. Low — the reference record echoes the pin.** `score_reference_csv` stores `manifest()["known_reference_csv"]["sha256"]`. It does not hash the file in that function. The file hash measured here equals the pin. A model score calls `assert_pinned_sources` first, and that function does hash the CSV.

**5. Low — parent config has no kernel key.** Shape checks still require conv weights `(128, 128, 3)`. `verify_checkpoints.py` does not look up `cfg["kernel"]`.

## Artifacts not independently recomputed

`runs/fp64_continuation_tune.log` lists 12 continuation tune numbers, and `STATUS.md` says the bounded jobs finished at 2026-11-08 00:17:20 CST. This review did not rerun those 12 forwards. It did check the 12 checkpoint states and parent hashes before that log was written. No candidate is endorsed here. The plan's selection rule, lowest tune R among the three high-LR +20 runs, is not implemented as code.

`runs/blends_tune.json` was read, not rerun. `w=0` matches the CSV tune R measured above. The materialized hard-signal tune numbers equal the constants in `blend_tune.py` (`routed` 0.5009909376779845, `control` 0.46211193532631023). Specialist inference was not repeated. The file's own caveat stands: the grid is tune-scored and is not a confirm result.

Not reviewed as study logic: `AUTHOR_PROMPT.txt`, `context_check0004.md`, `author.log`, and `__pycache__`. No new source file appeared after the hash snapshot below. `source/launch_continuations.sh` was reviewed and is absent from `runs/study_code_hashes.txt`.

## Source hashes at review

These matched `runs/study_code_hashes.txt` where that file lists them. Times are the directory listing at the start of the review (UTC).

| Path | SHA-256 | mtime |
| --- | --- | --- |
| `PLAN.md` | `086187accf10068b38770a1db98b2583150fa7bb84ac96e46426d4847febfff6` | 06:06:44 |
| `MANIFEST.json` | `c71474bcbb6caede6bb629d7db163469788054da2d48dcd4a56c1bc482edcf8b` | 06:06:44 |
| `source/study_common.py` | `ccd274e90c64761cee9322b72968efd6bf98e6fce8a7ffaec4c46ebf87024d81` | 06:06:44 |
| `source/score_fp64.py` | `4f41c3362e59c964e0704e38ccb1ff7251effedaa9c2fe64d7e7848a40195aeb` | 06:07:54 |
| `source/continue_train.py` | `dad108cc6c9339fdcf1c29765073a406d11cba9b03497814ad017982ff3fda6f` | 06:07:54 |
| `source/verify_checkpoints.py` | `95cc46882c40605dcec684d5b8f97b03a2eb32a0196e575c2bf8872274df34d2` | 06:07:54 |
| `source/blend_tune.py` | `49120295a9d1ef7b1eee8f139ac47e799dd63fd76ed038b174076ed2b6f3ad13` | 06:07:54 |
| `source/score_saved.py` | `818d9fba872a9809ffbbd1de096df4e909ec0ff0e940ac08f6b58ac912fe8feb` | 06:08:43 |
| `source/launch_continuations.sh` | `2604a042486aad8769b2c622d86665cddda6584d9d1ce748060ec2a163d6e7e5` | 06:08:43 |
| `tests/test_study.py` | `359c755b27420c0afdaecfcf7e65d231230f0646f13b046220f6d4256bd518a7` | 06:09:05 |

Probe logs in this directory: `probe_out.txt`, `probe_followup_out.txt`, `author_tests_out.txt`.

## Fix that would clear the concern

Make `score_path` reject a parent whose bytes differ from the manifest, and make `score_saved` reject a continuation whose `parent_checkpoint_sha256`, arm, learning rate, and extra-epoch budget do not match the frozen plan. Add tests that require those refusals, plus a shape/key refusal. Keep reporting the number of tune queries that entered `score_predictions`, not the full validation forward.
