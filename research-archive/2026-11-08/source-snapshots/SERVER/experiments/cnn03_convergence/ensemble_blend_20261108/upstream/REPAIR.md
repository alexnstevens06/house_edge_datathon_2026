# Repair after independent CONCERN

2026-11-08 00:28:09 CST. No training, no new epochs, seeds, or blend weights. `PLAN.md` and `MANIFEST.json` were not edited. Their SHA-256 values are still `086187accf10068b38770a1db98b2583150fa7bb84ac96e46426d4847febfff6` and `c71474bcbb6caede6bb629d7db163469788054da2d48dcd4a56c1bc482edcf8b`. Production `fp64_harness.py` is still `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2`. ZIP `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` and CSV `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` are unchanged. No candidate was packaged.

## What the review got right

`score_path` previously loaded a checkpoint and checked tensor shapes only. `score_saved.py` globbed `*_plus*` directories and did not read the parent hash, arm, learning rate, or extra-epoch budget. `n_queries` was `len(preds)` after the full validation forward, which is 768, while tune R uses 384 queries. `score_reference_csv` stored the manifest string instead of hashing the file in that function. The top-level `confirm`/`full` key check cannot fire because `score_predictions` does not return those keys. The author tests did not require a refusal.

The kernel-key note was not a scoring bug. Conv weight shape `(128, 128, 3)` still enforces kernel 3. `verify_checkpoints.py` was left as it was.

## Guards now on the score entrypoints

`score_path(path, arm)` refuses before the forward when any of these fail: manifest SHA-256 for a parent, post-run SHA-256 for a continuation, arm, seed, learning rate, parent hash, extra-epoch set `{10, 20}`, step budget `extra * 37`, architecture keys, shapes, finite tensors, signal tensors, or an incumbent path other than the manifest path. `score_saved.py` scores only the twelve names in `POST_RUN_HASHES.json`. It does not glob.

`POST_RUN_HASHES.json` was measured at 00:28 CST, after the 00:17:20 CST jobs. It is marked `not_a_pretraining_pin`. It is not `MANIFEST.json`.

Tune scoring checks 48 disjoint tune episodes, 48 confirm episodes, and 384/384 query ids against `queries.csv` and the official targets, then passes only the 384 tune rows to `score_predictions`. A bogus id raises `non-tune query id`. A confirm-only map raises `missing tune query ids`. `n_queries` is 384. `n_forward_queries` is 768 for a full validation forward. Row renorm is unchanged.

## Tests

`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/cnn03_convergence/continuation_20261108/tests/test_study.py`

Exit 0. Nine tests. The refusal test writes temporary copies only and requires `score_path` to reject a re-saved parent (`hash mismatch`), a shortened bias (`shape`), a deleted `pay.weight` (`architecture keys`), `sig.weight` (`signal tensors`), a NaN weight (`not finite`), a continuation learning rate, seed, `extra_epochs` 30, a zeroed parent hash, and an incumbent path that is not the manifest path. `score_tune` rejects a bogus query id and a confirm-only map. The real parent `curve_C_lr003_s0` still binds with no errors.

## Rescore

CPU only, same float64 module and the same renorm. Outputs are under `corrected_results/`. The original `runs/*.json` audit was not overwritten.

`compare_corrected.py` printed `NO_NUMERIC_DEVIATIONS`.

Corrected tune R, `n_queries=384`:

| Arm | Parent | +10 | +20 |
|---|---:|---:|---:|
| C s0 | 0.3101981645338761 | 0.32175874405089355 | 0.342295142373659 |
| control s0 | 0.33239084535278074 | 0.3278502730949005 | 0.3523407569601887 |
| C s1 | 0.3452074168357309 | 0.32251894300266665 | 0.39313234010633447 |
| control s1 | 0.34543239563228384 | 0.3604473040840377 | 0.3814229860445475 |
| C s2 | 0.30980333573689856 | 0.30925871088194445 | 0.3385772741455493 |
| control s2 | 0.3334821793902223 | 0.3323948474741577 | 0.338873569818254 |

Incumbent and the reference CSV remain 0.33206398205344967. The reference CSV SHA-256 measured in the scorer is `ef71ee39f668be0ee3f6e20215390f3dd0ab5da6a14199406c590334947b6aa6`.

The frozen +20 candidate is still `curve_C_lr003_s2_plus20` at 0.3385772741455493, which is worse than the incumbent. The +10 s2 value 0.30925871088194445 is not promoted. Blend `w=0` still matches the CSV. The grid minimum is still the control mix at `w=0.2`, tune R 0.32114647095721605, and the routed mix at the same weight is 0.32173621595454593. The grid was not extended.

## Source hashes after the repair

| File | SHA-256 |
|---|---|
| `source/study_common.py` | `cc63dd1b087e5986e66cafc9d04dcac47c3962fe0efd8736f5b5b38fdf6a177a` |
| `source/score_fp64.py` | `a79c1c006b8b83bc35ff40af1aa7c374050dff7aa97a2c55685cfdb656df87a6` |
| `source/score_saved.py` | `56a8bcedecf1133f739f8b3a3c19aa1d7e3e18ad0cfd8a3a70120874c5d12a51` |
| `source/blend_tune.py` | `d9e55f6e4d91313342d3c0f2b2a33a8c8935e8fe488e8b7cfa9d7e4f5655c308` |
| `source/write_post_run_hashes.py` | `725f3ffa6ea78d516ff2a6e892390a2a2e638d05cb31b0aee8e116b0a018cd00` |
| `source/compare_corrected.py` | `6a67f74d55233194522e1b3053b414e25f47989a52570fa1bc00e64da37a7649` |
| `tests/test_study.py` | `ea1b38bfd26fff5ef6387b498788acfaa4dbce200747c4dcd1a7984a6b606b71` |
| `POST_RUN_HASHES.json` | `1273bf82f382b8f5c61f48ecfb91d32a7b0937c7f1a148f9833eb9ce182f3eb9` |

`continue_train.py` and `verify_checkpoints.py` are unchanged from the review hashes. Ready for re-review. The CONCERN is not cleared by this note.
