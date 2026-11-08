# Candidate package review: curve_C_lr003_s2

**Verdict: PASS** for package quality of the frozen ZIP. Scope is the package bytes and the offline CPU checks below. Organizer acceptance, a production-pin replacement, and a blind held-out claim are outside this verdict. Cross-host CPU comparison is pending.

The frozen archive is the original `curve_C_lr003_s2` parent only. The +10 continuation, the +20 continuation, and the blends are not members. `PINNED.json` still names the canonical CNN03 checkpoints and does not name this parent. No production file was edited. `QUALITY.md` was not edited.

## What was run

Fresh unpack of the exact ZIP into `candidate_review/extract`. The author harness `main()` refuses an `--out` path outside `candidate_lr003_s2_20261108`, so that entry point was not called and nothing was written into the candidate folder. This process imported `fp64_candidate_harness.py` and rebound `WORK` and `OUT` in memory to `candidate_review/_work` and `candidate_review/harness_out`, then called the same check functions. `score_tune_only.py` `main()` was not called, because it writes `tune_only.json` into the candidate folder. The tune comparison was repeated here and written to `candidate_review/tune_only_reviewer.json`.

Interpreter: `experiments/cnn_runtime/py313_verify_1936/env/bin/python`, CPython 3.13.13, torch 2.11.0+cpu, numpy 2.5.4. `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES` empty. No GPU. Parent-process max RSS on the final probe was 634164 KB. Predict subprocess RSS peaked at 974928 KB. The Docker rerun RSS was 970948 KB.

Final command, from the repo root, exit 0 (`PROBE_OK`, `failures=[]`):

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
experiments/cnn_runtime/py313_verify_1936/env/bin/python \
  experiments/cnn03_convergence/independent_review_20261108/candidate_review/probe.py
```

| Step | Exit |
| --- | ---: |
| Probe, final run | 0 |
| `test_source_has_no_global_patch` | 0 |
| `test_window_and_missing` (`unit_ok`, 8 machines, missing vs zero, no labels/conditions) | 0 |
| `test_dtype_and_forecast_causality` (`dtype_ok`, 43 float64 params, float64 logits and softmax, no autocast, `Tensor.float` unchanged) | 0 |
| `test_query_order` (reversed queries, bitwise CSV) | 0 |
| `test_refusals` (`negative_ok`: missing data, hash flip, config newline, signal tensor, extra key, conv extent 5, float16; each refused and wrote no CSV) | 0 |
| `full_runs` cold test repeat plus validation export | 0 |
| Own-reference compare, test and validation, max abs 0, threshold 1e-8 | 0 |
| Tune-only parent rescore | 0 |
| Docker `--pull=never` rerun | 0 |

`diff -u` of `fp64_release_harness2025.py` and `fp64_candidate_harness.py`, from the repo root, matches `HARNESS_DIFF.txt` byte for byte. Tolerance remains `1e-8`. `full_runs` calls `official_check` only for the test split and `structural_coverage` for validation. It prints `VALIDATION_R not_computed`. The release harness still calls validation `official_check` and `official_r`; those calls are the ones this scoped diff removes. `official_r` remains defined in the candidate file and is not on the `main` or `full_runs` path. This review did not call it.

## Package bytes

ZIP SHA-256 `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29`, 1378957 bytes. Nine members, in order: `README.md`, `checkpoint.pt`, `config/config.json`, `data.py`, `model.py`, `predict.py`, `provenance/provenance.json`, `reproduce.json`, `requirements.txt`. No directory entries, `..`, absolute names, CSV, dataset, wheel, or virtualenv. Member hashes:

| Member | SHA-256 |
| --- | --- |
| `checkpoint.pt` | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` |
| `config/config.json` | `a715bfe05eb00494065c7dfcc3ad6d71a3b2bc4ee6788006cf07ce86f3e43695` |
| `model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` |
| `data.py` | `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133` |
| `predict.py` | `9eaa178249d6859cd1a68a979f2a8d1dcb681416de86900e9bd01f5abe6c673d` |
| `requirements.txt` | `cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab` |
| `reproduce.json` | `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1` |

`SOURCE_HASHES.txt` lists 27 digests. Each one matches the current file bytes. The file's own digest, which it does not list, is `96cc3ec167cd6959456ccc8b988c724b78bdd4417e06feebf448fef363b70647`.

The extracted checkpoint is byte-identical to `experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt`. It is weights-only `{config, state}`: 43 finite float32 tensors, no `sig.*`, no `kernel` field. Saved config is seed 2, lr 0.003, wd 0.00895, bs 512, epochs 30, drop 0.3, h 128, layers 4, W 64, pct_start 0.3, eval_every 2, out `experiments/cnn03_convergence/runs/curve_C_lr003_s2`. All four `blocks.*.conv.weight` tensors are `(128, 128, 3)`. Packaged `config.json` equals that object. `predict.py` pins those two hashes, refuses a key, shape, saved-dtype, signal, or config mismatch, and exits 2 before reading history or writing a CSV. Those refusals ran on the fresh extract. This pin is specific to this parent. It does not replace the production fixed pin.

Inside the canonical ZIP `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599`, `model.py`, `data.py`, `requirements.txt`, and `reproduce.json` are byte-identical to this package. `predict.py` and `checkpoint.pt` are not. Neither ZIP member hash equals the continuation files `7f2679b120217bc49c6628de40720c66294d38d651d61885397358e802ced205` (+10) or `887bdd3ba17083cd95753bae8e0981f61daf026c5a10b14c7fef0abb0dce84d6` (+20).

Protected files, rehashed this run and unchanged:

| File | SHA-256 |
| --- | --- |
| Canonical ZIP | `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` |
| `fp64_release_out2025/cnn03_test_a.csv` and `first_submission_full2029/cnn03_test_a.csv` | `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` |
| `PINNED.json` | `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb` |
| `fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` |
| `fp64_release_harness2025.py` | `a077e8771abdab58e3d107eeb1da3ecc3ee80ccc2ee08ce72e49b7ab36d89f2c` |
| Study `PLAN.md` / `MANIFEST.json` | `086187accf10068b38770a1db98b2583150fa7bb84ac96e46426d4847febfff6` / `c71474bcbb6caede6bb629d7db163469788054da2d48dcd4a56c1bc482edcf8b` |

## Forecast checks

Fresh test A and test B are bitwise identical: SHA-256 `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6`, 198127 bytes, 1536 ids. That matches `refs/candidate_test_a.csv`, both author Docker test CSVs, and this review's Docker rerun. Kit `probabilities` accepted every row. The test checker printed `VALID FORMAT: 1536 queries` and stated that it does not calculate an event score.

Validation export is 768 ids and 5 probability columns, SHA-256 `ee6efe67ff13d5681128281a2cff25c68c0af959077ed7977298253df46fe2ae`, matching `refs/candidate_val.csv` and the author Docker validation CSV. Coverage used `validation/queries.csv` only. Targets were not opened and no validation metric was computed.

Tune-only rescore on the fresh extract, CPU float64, one canonical renorm (`float` then `sum`), `score_tune_detail` on the 384 tune rows after a 768-query forward:

```text
TUNE_ONLY R=0.30980333573689856 n=384 forward=768 loader_max_abs=0.0
```

The study loader `house_edge.data.load_split` and the packaged `data.load_public_split` agree on every query id. Per-query probability max abs is 0 before that renorm and 0 after it. The stored parent value in `corrected_results/fp64_original_tune.json` is the same number. Confirm, full, and test metrics were not computed. The canonical tune figure 0.33206398205344967 was not remeasured in this review.

## Procedural caveat

`QUALITY.md` opens by saying this package did not compute confirm, full, or test R. That sentence describes the frozen harness and the checkpoint that was packed. It does not describe an earlier draft. The author log and `QUALITY.md` line 47 say that draft called `check_submission.py --split validation`, which prints public validation condition points, and that the call was removed before the frozen re-run. This review did not repeat that metric and does not treat the package as blind held-out proof.

File times, UTC: the original parent checkpoint is `2026-11-08T02:05:39Z`. The study tune JSON that already records this parent is `2026-11-08T06:27:52Z`. Candidate `entry/predict.py`, which contains the parent hash pin, is `2026-11-08T06:53:00Z`. The ZIP is `2026-11-08T06:53:16Z`. The frozen harness is `2026-11-08T06:54:36Z`, before the author Docker runs at `2026-11-08T06:55:08Z`. The packed checkpoint hash is the pre-existing parent, and the ZIP contains no second checkpoint. A search of the candidate tree found the validation-metric format strings only in the unused `official_r` function, in the diff that deletes the call, and in the words `VALIDATION_R not_computed` in `QUALITY.md`, `HARNESS_SCOPE.md`, and the diff. No saved numeric dump of that checker run is in the tree. The author log is an untimestamped narrative, so the draft's stdout itself is not independently recoverable. The frozen control flow does not score validation.

## Dependency caveat

Read directly: `kit/README.md` line 333 and `kit/RULES.md` line 55 say organizers prepare a pinned root `requirements.txt` before the offline run. `kit/SUBMISSION_README.md` line 23 says the same. The cached guide quotation in `experiments/cnn_runtime/contract-evidence/CONTRACT.md` says the standard `python` image is Python 3.13 plus the standard library, with extra libraries from an offered environment or bundled wheels and an offline setup script. This review did not re-fetch that guide and does not choose which text wins. The ZIP requirements file names `torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl` and does not contain a wheel. Provenance says `organizer_approval_claimed` is false. Local proof used the existing CPython 3.13.13 CPU environment. That does not establish the organizer runtime.

## Docker

Author inspect files for `cand-lr003-s2-a`, `cand-lr003-s2-b`, and `cand-lr003-s2-val` show exit 0, `OOMKilled` false, image `sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55`, network `none`, read-only root, `CapDrop ["ALL"]`, `no-new-privileges`, user `1000:1000`, 2 CPU, memory 17179869184, memory+swap 34359738368, tmpfs `/tmp` 128 MiB and `/output` 32 MiB, and read-only binds. The inspect JSON does not store the CLI flag `--pull=never`.

This review reran one test container against the fresh extract. The local image id matched. The command used `--pull=never --network=none --read-only --cpus=2 --memory=2g --memory-swap=2g --user 1000:1000 --cap-drop=ALL --security-opt no-new-privileges` and the same tmpfs sizes. Exit 0. Stderr: `infer_s=1.949496 total_s=2.659115 max_rss_kb=970948`. Streamed CSV SHA-256 `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6`, `cmp` equal to `refs/candidate_test_a.csv`. No pull and no package change.

## Cross-host

Not run from this session, and not claimed. A coordinator can copy the same ZIP (`dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29`) to TOWER, run the installed CPU runtime, and compare per-query canonical row-norm max abs at `1e-8`, with metadata recorded. Until that comparison is reported, cross-host status stays pending.
