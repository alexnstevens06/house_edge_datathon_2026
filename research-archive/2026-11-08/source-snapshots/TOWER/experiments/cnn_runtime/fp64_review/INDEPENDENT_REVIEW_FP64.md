# Independent review — FP64 anchor and CNN03 packages

**Decision: BLOCKED RELEASE.**

Computational checks that finished support the float64 forward. This review did not finish the packaged harness, did not run the server sandbox, and the packaged requirements still do not match the 11-package freeze. Old `experiments/cnn_runtime/INDEPENDENT_REVIEW.md` is unchanged. Packages, checkpoints, and ZIPs were not edited. `fp64_review/hash_diff.json` is `{"changed": []}`.

Labels: **VERIFIED** = this review measured it. **PROVIDED** = coordinator statement, not re-run here. **NOT RUN / TIMEOUT** = no finished log.

Tower runtime actually used: CPython 3.12.3, NumPy 2.5.4, Torch **2.9.1+rocm6.4**, CPU capability AVX2, oneDNN available. That is not CPython 3.13 or `torch==2.11.0+cpu`. `py313_verify_1936/env/bin/python` exited 1: `ModuleNotFoundError: No module named 'torch'`.

## Blockers

1. **Requirements are not the freeze.** Both packages pin `numpy==2.5.4` and `torch==2.11.0+cpu` (`requirements.txt` SHA `d836e33a0f3deff0abc6d98229a1b16ac92860bf7d462c3e9a738c6f2cc2afa4`). The CPU wheel URL and SHA sit in comments only. There is no index URL and no PEP 508 direct URL. **VERIFIED** freeze file `experiments/cnn_runtime/requirements_py313_cpu211_frozen.txt` SHA `cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab` (270 bytes) lists 11 pins, with torch as `torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl`. **PROVIDED:** the code author's follow-up to copy that freeze was blocked by a headless permission prompt on `sha256sum`+`cp`, was not run, and was not retried. Package comments still say the freeze was absent. That sentence is stale.
2. **Contract is unresolved.** Kit README/RULES say organizers preprovision a root `requirements.txt` before the offline run. Staged live guide says Python 3.13 stdlib, an offered environment, or bundled wheels plus an offline setup script. No precedence is chosen here. These packages are not organizer-ready.
3. **This review's harness did not finish.** See below. Builder `FP64_INTEGRATION_STATUS.md` is not this run.
4. **Release still needs the provided server proof to stay labeled provided.** It was not re-executed here. No fresh-extract sandbox was started in this review.

## Source and artifacts (VERIFIED hashes)

Canonical inference source `reference/.../cnn.py` and `mlp_cnn_hpo/server_snapshot/source/base_cnn.py` are both `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5`.

| Item | SHA-256 |
|---|---|
| Both `model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` |
| Both `data.py` | `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133` |
| Anchor `predict.py` | `8d97923a4bf3a2f3f2df562a07571518b99489d6723db4bd3f6a02b247316f33` |
| CNN03 `predict.py` | `a2aa00a32b0af42445014fbe8980cfbbb2afc5644a474c10d48bdf43fc31e6c2` |
| Anchor checkpoint and reference `g_nosignal/checkpoint.pt` | `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80` |
| Anchor config and reference config | `c9022a4305be1c4124e6a116cd0cf3132fb5624a206f9b5571090ca3c4769d2f` |
| CNN03 checkpoint and `final_cnn03_s0/checkpoint.pt` | `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f` |
| CNN03 config and snapshot config | `6169196bc0827286d5bcbd931980727e0442b1ffdef664cb7f7b319cad4eb4f5` |
| `fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` |
| `fp64_anchor_entry.zip` | `f8441494d806581939d7be09bf29fe02c31d908c155aa1eb5443d8edaf555953` (1375476) |
| `fp64_cnn03_entry.zip` | `0ea5b0d2fd3db1b6df861037bfb6a7825a2676e988ee3d199f8e37820c7cc319` (1376720) |

Both ZIP `reproduce.json` values are exactly `image=python` and `command=["python","predict.py","--data","/data","--out","/output/submission.csv"]`. Inspect saw no `.csv` members. Checkpoint bytes match the originals above. Saved tensors are float32. `model.double()` is the inference cast.

`PINNED.json` maps `final_cnn03_s0` to `ce831b9b…` and also lists s1 `a4285dcf…` and s2 `8db2bdf0…`. The package contains s0 only. Checkpoint config has **no** `no_signal` field, `drop` 0.3, kernel 3, seed 0, phase `final`. `predict.py` does not treat a missing `no_signal` as true; it refuses `no_signal is False` and any `sig.*` state, and builds `use_signal=False`. Tune-only selection text is in `experiments/mlp_cnn_hpo/review/REVIEW_GROK_BUILD.md` (s0 by `tune_R`; s2 better on confirm and not chosen). Those R numbers were not recomputed here.

Forward versus canonical `cnn.py`: same Block/head structure. Scoped dtype edits are `pay.weight.dtype` for mask, tick, and `arange`, and softmax kept in that dtype. Integer payout indexes stay integer (`x.long()`, missing index 5). No `Tensor.float` assignment and no `set_default_dtype`. Training `main`, `build_train`, and CUDA autocast are absent. `USE_SIGNAL` defaults false and the signal branch raises.

## What this review finished

**VERIFIED whole-model causality** (package `load_pinned` + `CNN` after `double()`, all test episodes, all 8 machines poisoned at and after cutoff):

- Anchor future-logit max abs **0.0**. CNN03 future-logit max abs **0.0**. Logits `torch.float64`. Signal feature sum 0. No `sig` keys. 43/43 floating parameters float64.
- Per-machine edit of the last in-window tick changed logits on all 8 machines (anchor max abs about 0.46–3.16; CNN03 about 0.40–3.30).

**VERIFIED harness compare gate**, by calling original `require_close` (uses `sys.executable` in the subprocess helper; kit row-normalization): identical exit 0; about 1e-9 exit 0; about 2e-8 exit 1.

**Redirected harness: FAIL before package tests.** `fp64_review/harness.log` exit 1. `Path(__file__).parents[2]` on the copy resolved to `experiments/experiments/cnn_runtime/...`. No export, checker, refusal, or repeat result came from that process. A second launch was killed when the review timed out. **Full final-path harness: TIMEOUT / NOT RUN.**

Stored builder CSV `fp64_out/anchor_test_a.csv` hashes to `02f08b8d44a2d57d798c5655f1519eae1c5c23aafd4e41bb1bd6be2b68b44710` (197493 bytes), the same SHA `FP64_INTEGRATION_STATUS.md` claims. This review did not re-run that export. Other status numbers (points, RSS, server file diffs) stay **not re-verified**.

## PROVIDED server proof (not re-run)

Same immutable snapshot ZIPs, fresh-extracted on both hosts. Max normalized abs: anchor test `6.66133814775e-16`, anchor val `6.10622663544e-16`, CNN03 test `9.99200722163e-16`, CNN03 val `4.99600361081e-16`. All within 1e-8. Cold repeats bitwise. CPython 3.13.13 + Torch 2.11 CPU, Docker `--network=none`, read-only, 2 CPU, 16 GiB, tmp 128 MiB, output 32 MiB, model run about 2.628 s and 979 MB. Checkers: 1536 test and 768 val. Source hashes said to match these folders. Files named under `experiments/cnn_runtime/fp64_server_zip_proof1959`. This review did not open or execute them.

## Not approved

Release stays blocked. The float64 math that was executed matches the no-signal forward with an explicit dtype change, and the whole-model cutoff check passed. The harness process, the 3.13/2.11 pin as installed bits, the freeze copy, and this reviewer's sandbox were not completed. No FP32 sandbox is used as a substitute.
