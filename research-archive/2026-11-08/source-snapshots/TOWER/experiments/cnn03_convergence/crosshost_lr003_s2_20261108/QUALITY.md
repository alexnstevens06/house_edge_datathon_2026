# curve_C_lr003_s2 candidate quality

Staged for independent review. This is not production clearance, not an automatic replacement, and not an organizer acceptance. The study re-review PASS does not clear this package.

No training was run. Nothing was uploaded or submitted. `experiments/mlp_cnn_hpo/review/PINNED.json` was not modified and is not pointed at this candidate.

## Selection caveat

The packaged weights are the original 30-epoch `curve_C_lr003_s2` parent only. The +10 continuation, the +20 continuation, and every blend stay unpackaged. +10 seed 2 remains exploratory. The frozen +20 study pick loses to the incumbent.

Tune-only context, from the already measured study table. Lower R is better. This package recomputed only the parent tune R, on 384 rows, and it matched. It did not compute confirm, full, or test R, and it did not use any of those to choose the checkpoint.

| Item | Tune R |
| --- | --- |
| Original lr 0.003 seed 2 parent, this package | 0.30980333573689856 |
| Current canonical CNN03 tune | 0.33206398205344967 |
| Matched control seed 2 | 0.3334821793902223 |
| Original lr 0.003 seed 0 vs its control | 0.3101981645338761 vs 0.33239084535278074 |
| Original lr 0.003 seed 1 vs its control | 0.3452074168357309 vs 0.34543239563228384 |
| Frozen +20 high-LR result | 0.3385772741455493 |
| Exploratory +10 seed 2 | 0.30925871088194445 |

All three original lr 0.003 parents beat their matched controls on that study table. Blends lose to this original parent. The parent config stores `eval_every` 2. That field was not rewritten. The config has no `kernel` field. Kernel 3 is the measured `blocks.*.conv.weight` shape `(128, 128, 3)` on all four layers. All 43 tensors are finite float32 on disk and float64 after `model.double()`. `Tensor.float` is not patched.

## What was executed

Interpreter: `experiments/cnn_runtime/py313_verify_1936/env/bin/python` (CPython 3.13.13, torch 2.11.0+cpu, numpy 2.5.4). `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES` empty. The harness and the tune score ran on `unpacked/`, the extract of the ZIP, not on a loose extra tree.

Host cold repeat of the unpacked test command, 2026-11-08 00:53:29 CST, exit 0. Two test CSVs compared equal with `cmp`. Validation CSV was generated and not scored.

- Test A: infer 1.919979 s, total 2.569273 s, max RSS 973392 KB, 1536 queries
- Test B: infer 1.915275 s, total 2.585509 s, max RSS 974072 KB
- Validation export: infer 0.967496 s, total 1.489558 s, max RSS 617284 KB, 768 queries

Frozen harness re-run, 2026-11-08 00:55:08 CST, exit 0, elapsed 13.46 s, max RSS 973220 KB. `HARNESS_OK`.

- `unit_ok`: all 8 machines; future ticks do not change the cutoff window; tick t-1 does; MISSING is index 5 with mask 0; payout 0 stays index 0; slots sorted; signal column dropped; labels and conditions not opened; missing split fails
- `dtype_ok`: 43 float64 parameters, float64 logits and softmax, mask and tick inputs float64, no autocast, tensor float method not monkeypatched
- `query_order_ok`: reversed query rows, bitwise identical CSV, 8 data rows
- `negative_ok`: missing data, byte-flipped checkpoint (`hash mismatch`), config newline (`unclean config hash`), signal tensor, extra key, conv kernel extent 5, float16 saved tensor. Each refused and wrote no CSV
- Cold repeat inside the harness, bitwise SHA-256 `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6`, 198127 bytes, 1536 ids
- Kit test checker: `VALID FORMAT: 1536 queries`. That checker does not calculate an event score
- Validation coverage against `queries.csv`: 768 ids, 5 probability columns, targets not opened
- `VALIDATION_R not_computed`
- Own-reference max abs after kit row norm: test 0, validation 0, threshold 1e-8

An earlier draft of the harness called `check_submission.py --split validation`. That kit command prints public validation points. The call was removed before this frozen re-run. Those points are not a selection input and are not repeated here.

Tune score, unpacked ZIP bytes, exit 0, elapsed 2.89 s, max RSS 620968 KB:

```text
TUNE_ONLY R=0.30980333573689856 n=384 forward=768 loader_max_abs=0.0
```

The study loader (`house_edge.data.load_split`) and the packaged loader (`data.py`) match after one canonical renorm (`float` then `sum`). The scorer is the existing `score_tune_detail`. Scoring precision and normalization were not changed.

Docker, existing local image only, `--pull=never`, no new install. Image `ubuntu:24.04` is `sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55`. Three containers, 2026-11-08 00:55:08–00:55:16 CST, each exit 0, then removed. Inspect: network `none`, read-only root, 2 CPU, memory 17179869184, memory+swap 34359738368, `CapDrop ALL`, `no-new-privileges`, uid 1000. `/tmp` 128 MiB and `/output` 32 MiB were tmpfs. The existing Python 3.13 tree was mounted read-only at its own path. Output was streamed from `/output/submission.csv`. Both test runs and the validation export match the host reference bytes. Test infer about 1.99 s and 970376 KB. Validation export infer about 1.01 s and 616284 KB. The validation export was not scored.

## Artifact hashes

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `curve_C_lr003_s2_fp64_candidate.zip` | `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29` | 1378957 |
| Original and unpacked `checkpoint.pt` | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` | 1483637 |
| `config/config.json` | `a715bfe05eb00494065c7dfcc3ad6d71a3b2bc4ee6788006cf07ce86f3e43695` | 221 |
| `model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` | 4458 |
| `data.py` | `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133` | 4381 |
| `predict.py` | `9eaa178249d6859cd1a68a979f2a8d1dcb681416de86900e9bd01f5abe6c673d` | 9793 |
| `requirements.txt` | `cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab` | 270 |
| `reproduce.json` | `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1` | 117 |
| Test CSV | `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6` | 198127 |
| Validation CSV, not scored | `ee6efe67ff13d5681128281a2cff25c68c0af959077ed7977298253df46fe2ae` | 103442 |

ZIP members, in order: `README.md`, `checkpoint.pt`, `config/config.json`, `data.py`, `model.py`, `predict.py`, `provenance/provenance.json`, `reproduce.json`, `requirements.txt`. No cached forecast, public data, wheel, or cache is inside the ZIP. `model.py` and `data.py` match the canonical float64 entry. `requirements.txt` and `reproduce.json` are the canonical lock and command, copied byte-for-byte. The lock names a torch wheel URL. This run did not download it.

The candidate test CSV is not the canonical test CSV. The canonical file remains `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d`.

Protected files, hashed before staging and again after the frozen re-run:

| File | SHA-256 |
| --- | --- |
| `experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip` | `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` |
| `experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv` | `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` |
| `experiments/mlp_cnn_hpo/review/PINNED.json` | `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb` |
| `experiments/cnn_runtime/fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` |
| `experiments/cnn_runtime/fp64_release_harness2025.py` | `a077e8771abdab58e3d107eeb1da3ecc3ee80ccc2ee08ce72e49b7ab36d89f2c` |
| Original parent checkpoint | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` |

The production harness digest is `515c91bc…`. The separate release harness digest is `a077e877…`. Neither file was edited. End measurements of `continuation_20261108/PLAN.md` and `MANIFEST.json` match the previously recorded digests `086187accf10068b38770a1db98b2583150fa7bb84ac96e46426d4847febfff6` and `c71474bcbb6caede6bb629d7db163469788054da2d48dcd4a56c1bc482edcf8b`.

`HARNESS_SCOPE.md` and `HARNESS_DIFF.txt` record the scoped copy. Repo root is `parents[3]` because this harness is one directory deeper than the release harness. The numeric threshold stays `1e-8`.

## Dependency caveat

Kit README, RULES, and SUBMISSION_README say organizers preinstall the pinned requirements before the offline prediction. The cached participant guide also describes a standard-library image, an offered environment, or bundled wheels. This package does not choose which text wins. No wheel is bundled. `organizer_approval_claimed` is false. The local proof used the existing CPython 3.13.13 CPU environment inside the existing `ubuntu:24.04` image.

## Not run

Cross-host CPU was not run. This session did not use tower SSH. A coordinator can copy the ZIP and run the handoff below on another CPU. Until that happens, there is no second-host proof.

## Read-only handoff

The harness refuses an `--out` directory outside this candidate folder. A fresh extract can still be the `--entry`. Do not point `PINNED.json` at this tree. Do not pip install. Do not `docker pull`.

```sh
sha256sum \
  experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip \
  experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv \
  experiments/mlp_cnn_hpo/review/PINNED.json \
  experiments/cnn_runtime/fp64_harness.py \
  experiments/cnn_runtime/fp64_release_harness2025.py \
  experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt

rm -rf /tmp/lr003_s2_review && mkdir /tmp/lr003_s2_review
python3 -c 'import zipfile; from pathlib import Path; z=zipfile.ZipFile("experiments/cnn03_convergence/candidate_lr003_s2_20261108/curve_C_lr003_s2_fp64_candidate.zip"); d=Path("/tmp/lr003_s2_review");
[ (d/i.filename).parent.mkdir(parents=True, exist_ok=True) or (d/i.filename).write_bytes(z.read(i)) for i in z.infolist() ]'

export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=experiments/cnn_runtime/py313_verify_1936/env/bin/python
"$PY" experiments/cnn03_convergence/candidate_lr003_s2_20261108/fp64_candidate_harness.py \
  --entry /tmp/lr003_s2_review \
  --out experiments/cnn03_convergence/candidate_lr003_s2_20261108/review_out \
  --reference-test experiments/cnn03_convergence/candidate_lr003_s2_20261108/refs/candidate_test_a.csv \
  --reference-val experiments/cnn03_convergence/candidate_lr003_s2_20261108/refs/candidate_val.csv
"$PY" experiments/cnn03_convergence/candidate_lr003_s2_20261108/score_tune_only.py \
  --entry /tmp/lr003_s2_review \
  --original experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt
```

Expected tune line: `TUNE_ONLY R=0.30980333573689856 n=384 forward=768 loader_max_abs=0.0`. Expected harness tail: `HARNESS_OK`. Expected test SHA-256: `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6`.

For a second host, use an already installed CPython 3.13 and torch 2.11 CPU image. Repeat with `--pull=never --network=none --read-only --cpus=2 --memory=16g --memory-swap=32g`, tmpfs `/tmp` 128 MiB and `/output` 32 MiB, `--user 1000:1000`, `--cap-drop=ALL`, `--security-opt no-new-privileges`, and read-only mounts of the fresh extract, the public data, and that host's existing interpreter. Stream `/output/submission.csv` before the container exits. Compare it to the test SHA above. Do not score the validation CSV.
