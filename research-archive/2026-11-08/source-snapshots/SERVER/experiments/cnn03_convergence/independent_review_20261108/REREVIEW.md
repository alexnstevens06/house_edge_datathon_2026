# Re-review — CNN03 higher-LR continuation study

**Verdict: PASS for study verification only.**

This does not clear the production FP64 release, does not package a candidate, and does not replace `REVIEW.md` (that file stays **CONCERN**).

Re-checked 2026-11-08 after the study-only guard repair. CPU, 2 threads, `CUDA_VISIBLE_DEVICES=""`. No training. No confirm, full-validation, or test metric was computed in this pass. Writes stayed in `experiments/cnn03_convergence/independent_review_20261108`.

## Prior findings

The original high and medium findings are closed on the scoring entry points.

`score_path` hashes the file it loads, then refuses before the forward on a bad parent hash, a bad continuation hash, arm, seed, learning rate, parent hash, extra-epoch set, step budget, architecture keys, tensor shapes, non-finite weights, signal tensors, or an incumbent path that is not the manifest path. `n_queries` is 384. `n_forward_queries` is 768. `score_reference_csv` stores the SHA-256 it just measured and raises if that differs from the manifest. `score_saved.py` scores the twelve names in `POST_RUN_HASHES.json`. It does not glob, and it refuses an output path outside `corrected_results`.

Author tests, run here: exit 0, nine passes, including `test_score_path_refuses_hash_key_shape_finite_signal_lr_seed_budget_parent` and `test_score_tune_refuses_non_tune_ids`. Log: `rereview_author_tests.txt`.

## Independent mutations

Temporary files were under `/tmp`. After every mutation, `curve_C_lr003_s0/checkpoint.pt` and `curve_C_lr003_s2_plus20/checkpoint.pt` still matched their pins. Snapshots `snapshot_before.json`, `snapshot_after_guards.json`, and `snapshot_after_scores.json` are identical.

| Probe | Entrypoint result |
| --- | --- |
| Resaved parent with the same state | `hash mismatch` |
| Same state, config `kernel` set to 5 | `hash mismatch` (file not scored) |
| `blocks.0.conv.weight` shaped `(128, 128, 5)` | `hash mismatch` and `shape (128, 128, 5) != (128, 128, 3)` |
| Byte copy of the real parent at another path | `hash matches a different path` |
| Continuation `total_steps` set to 1 | `step budget` and `hash mismatch` |

Kernel semantics, measured: all six parent configs omit `kernel`, `depth`, and `width`. They declare `h=128`, `layers=4`, `W=64`, `drop=0.3`. Conv weights are `(128, 128, 3)`, so kernel 3 is the shape contract. That omission is legitimate. A declared wrong kernel does not get a tune score: the file hash no longer matches the pin, so `score_path` refuses. `parent_bind_errors`, if handed the original digest together with a different in-memory config, does not itself mention `kernel` (`helper_kernel5_with_forced_matching_digest []`). `score_path` does not call it that way. It hashes the same path it unpickles. The incumbent binder does require `kernel == 3`, and the incumbent file has that field. Continuations also omit `kernel`.

`continuation_bind_errors` on a byte-identical copy at `/tmp` returned no errors. Different bytes are still refused. `score_saved.continuation_jobs` binds each name to `runs/<arm>_plus<extra>/checkpoint.pt` and checks that file's SHA-256 before `score_path`. The twelve rescored files were those paths.

## Recomputed tune scores

`score_path` / `score_reference_csv` / `score_tune_detail`, float64 canonical module `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a`. Reference CSV tune R from the study scorer equals a direct `score_predictions` call on the 384 tune rows after the same `float`/`sum` renorm: both `0.33206398205344967`. The measured reference SHA-256 is `ef71ee39f668be0ee3f6e20215390f3dd0ab5da6a14199406c590334947b6aa6`. Every row below has `n_queries=384` and `n_forward_queries=768`. Each `tune_R` equals both `runs/fp64_original_tune.json` or `runs/fp64_continuation_tune.json` and the corrected JSON. `SCORE_MISMATCHES []`.

| Arm | Parent | +10 | +20 |
| --- | ---: | ---: | ---: |
| C s0 | 0.3101981645338761 | 0.32175874405089355 | 0.342295142373659 |
| C s1 | 0.3452074168357309 | 0.32251894300266665 | 0.39313234010633447 |
| C s2 | 0.30980333573689856 | 0.30925871088194445 | 0.3385772741455493 |
| control s0 | 0.33239084535278074 | 0.3278502730949005 | 0.3523407569601887 |
| control s1 | 0.34543239563228384 | 0.3604473040840377 | 0.3814229860445475 |
| control s2 | 0.3334821793902223 | 0.3323948474741577 | 0.338873569818254 |

Incumbent `0.33206398205344967`.

The lowest of the three high-LR +20 scores is `curve_C_lr003_s2_plus20` at `0.3385772741455493`. That is worse than the incumbent `0.33206398205344967`. The matched control +20 is `0.338873569818254`. The original high-LR seed 2 parent is `0.30980333573689856`. Its +10 score `0.30925871088194445` is lower and is not the frozen rule. Nothing in the scorer promotes a +10 run. No package was built.

Continuation checkpoints are not weights-only. Each saved file records `warm_start_from=weights_only`, `parent_had_optimizer=false`, the original parent SHA-256, the arm learning rate, seed `1000+seed+extra`, and steps `extra*37` (370 or 740). `POST_RUN_HASHES.json` is marked `not_a_pretraining_pin`, measured `2026-11-08 00:27:44 CST`, after jobs finished `2026-11-08 00:17:20 CST`. Its twelve digests and the incumbent digest `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f` match the files. `PLAN.md` and `MANIFEST.json` are still the pre-training hashes from `REVIEW.md`. The six parent pins still validate, 43 tensors, weights-only.

## Blends

Recomputed with `blend_tune.materialize` and `mix`, then `score_tune_detail`. Weights only `0, 0.025, 0.05, 0.1, 0.2`. Exact match to both `runs/blends_tune.json` and `corrected_results/blends_tune.json`.

`w=0` is `0.33206398205344967`. Materialized tune R is routed `0.5009909376779845` and control `0.46211193532631023`. At `w=0.2`, control is `0.32114647095721605` and routed is `0.32173621595454593`. Both are tune-only, above the original high-LR seed 2 parent, and are not a confirm result. The grid was not extended.

## Production bytes

| File | SHA-256 |
| --- | --- |
| `experiments/cnn_runtime/fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` |
| `experiments/mlp_cnn_hpo/review/PINNED.json` | `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb` |
| `experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip` | `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` |
| `experiments/cnn_runtime/first_submission_full2029/cnn03_test_a.csv` | `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` |

Study source does not import the harness or assign `Tensor.float`. `continue_train.py` `dad108cc6c9339fdcf1c29765073a406d11cba9b03497814ad017982ff3fda6f` and `verify_checkpoints.py` `95cc46882c40605dcec684d5b8f97b03a2eb32a0196e575c2bf8872274df34d2` match the first review. Repaired sources match `REPAIR.md`: `study_common.py` `cc63dd1b087e5986e66cafc9d04dcac47c3962fe0efd8736f5b5b38fdf6a177a`, `score_fp64.py` `a79c1c006b8b83bc35ff40af1aa7c374050dff7aa97a2c55685cfdb656df87a6`, `score_saved.py` `56a8bcedecf1133f739f8b3a3c19aa1d7e3e18ad0cfd8a3a70120874c5d12a51`, `blend_tune.py` `d9e55f6e4d91313342d3c0f2b2a33a8c8935e8fe488e8b7cfa9d7e4f5655c308`, `test_study.py` `ea1b38bfd26fff5ef6387b498788acfaa4dbce200747c4dcd1a7984a6b606b71`, `POST_RUN_HASHES.json` `1273bf82f382b8f5c61f48ecfb91d32a7b0937c7f1a148f9833eb9ce182f3eb9`.

Numbers and probe text: `rereview_scores.json`, `rereview_blends.json`, `rereview_probe_out.txt`.
