# Comparative result, not a package

2026-11-08 02:24 CST. Protocol `COORDINATOR_PLAN.md` SHA-256 `03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3`, read in full. No training. No package. Frozen low-rate files and the mirrored ensemble tree were not edited. `evaluate.py` was not run, so the mirrored `results/` files were not rewritten.

Authoritative numbers below are a new CPU run: `experiments/cnn_runtime/py313_verify_1936/env/bin/python`, CPython 3.13.13, torch 2.11.0+cpu, GPU hidden, 2 threads. Host RSS at the end was 666848 KB, under 2 GiB. Forward and softmax for the matrix used the pinned canonical float64 module. Rows were float-then-sum renormalized, then scored with the kit scorer on the frozen 384 tune queries and 48 tune episodes. Confirm, full, and test metrics were not computed.

## Tests and freeze

Before `results/comparison.json` was written, the runner checked `SOURCE_SHA256.txt` and `PINS.json` (86 read-only files, including both immutable ZIP/CSV pairs, production `PINNED.json`, and `fp64_harness.py`). Selector tests printed `SELECT_TESTS_OK`. The nine mirrored ensemble tests passed in-process, including the real seed-2 float64 logit and softmax probe, the causality and missing-versus-zero window check, and the query-order check.

`SOURCE_SHA256.txt` SHA-256 `c083ca6c87472c675cf8d5366364d00d9775e9ed03452c8c8437bfbae798b5f9`. `results/comparison.json` SHA-256 `9341e9553020e55c8ed863d20b2c31b10be7e45c4d3ffeb4d7a7ceabf991a005`. This note is not in the frozen hash.

```
aa1e515e122322caec69b0a13ae54324b8c5b8e04619ca14d388e6ac49ab0a94  PINS.json
a396b4b3299decc4819948f91658fd8583070fc961dd992d96dee5c7ab8598d5  source/ranking.py
bde8e15adf7ba1691c652d1b2da59b0f910133fc3c85b47ffb81541d87db0202  source/compare_runner.py
c1a7d3a38af9acc2093a94491981a15926f95fcca84b190f53bd1e2f115720bf  tests/test_select.py
2de6f9a94e7d3bdba993009403a0afbbb51714bc4a37e556d21aced6ba28bef4  tests/run_frozen_ensemble_tests.py
```

## Tower maps

Recomputed canonical float64 maps match the immutable tower npz files per query after normalization. The largest absolute gap is 1.443e-15, under 1e-8. Seed 0, seed 1, and seed 2 tune R match the published scalars with delta 0. Weight 0 is an exact copy of seed 2 and scores 0.30980333573689856.

| Row | Tune R | Max abs vs tower npz |
| --- | --- | --- |
| ensemble_equal_s0_s2 | 0.28468175920919014 | 5.551e-16 |
| ensemble_equal_s0_s1_s2 | 0.28975002607300254 | 5.551e-16 |
| blend w=0.2 | 0.29645525780787219 | 6.661e-16 |
| blend w=0.1 | 0.30039118477366389 | 7.772e-16 |
| blend w=0.05 | 0.30441273225560084 | 8.882e-16 |
| blend w=0.025 | 0.30693690199632961 | 8.882e-16 |
| blend w=0 | 0.30980333573689856 | 9.992e-16 |

The w=0.2 score differs from the tower `matrix.json` display 0.29645525780787213 by about 6e-17. The probability maps still agree within 1e-15. That gap does not change the order.

## Shared-control discrepancy

The pinned checkpoint `92233992aeb8c007a40f925885d7f85ef260cf8584ea0cc669dcd39ce57d201f` is the same bytes as `experiments/cnn_experts/hard_signal/runs/s0/control/checkpoint.pt`.

| Path | Tune R | Abs delta vs 0.46211193532631023 |
| --- | --- | --- |
| Canonical float64 forward and softmax | 0.46211190598948326 | 2.934e-08 |
| Same canonical windows, float32 forward and softmax | 0.46211193532631023 | 0 |
| Original hard-signal CPU fp32 loader and `predict_single` | 0.46211193532631023 | 0 |

The historical scalar is the float32 path. It is reproduced here with delta 0 by two separate float32 implementations. The canonical float64 path is not that scalar. Per query, the float64 map and the float32 map differ by at most 3.656e-07, which is outside 1e-8, so this is a real precision difference. Probabilities were not rescaled or replaced. The blend rows use the unmodified canonical float64 control map. They therefore miss the old 1e-12 reproduction target on the control component even though the blend arithmetic itself matches the tower files. `ensemble_equal_s0_s2` does not use the shared control. Its two parents reproduce their published tune R with delta 0.

## Server low-rate scores

The saved server checkpoints were rescored with the frozen low-rate CPU float64 scorer. Each rescore matched the saved JSON value and checkpoint hash. Exits were 0 and each run has 730 steps. These are weights-only warm starts with a fresh AdamW, not an optimizer resume. Training was FP32 batch 128. The parents were bf16 autocast batch 512. No tower 5e-5 checkpoint exists.

| Row | Tune R |
| --- | --- |
| server primary epoch 1 | 0.30951549616992924 |
| server primary epoch 3 | 0.30706943777760692 |
| server primary epoch 5 | 0.30781275392021723 |
| server control epoch 1 | 0.33413621090622814 |
| server control epoch 3 | 0.33213683848995773 |
| server control epoch 5 | 0.33244487608328904 |

## Tower low-rate rows

All six tower epoch rows are `not_run_preflight_blocked`. They are not zeros and they are not missing successes. `assert_input_pins` hashes `experiments/cnn_runtime/fresh_fp64_cnn03_1956` and the six original `experiments/cnn03_convergence/runs/.../checkpoint.pt` paths before it reads `--parent`. Those paths are absent on Tower. The same parent bytes and the canonical model are already staged under the approved ensemble inputs and the candidate unpack. The handoff is not portable as written. No GPU step ran and no tower low-rate checkpoint was saved. This comparison does not edit that trainer or invent paths.

## Selection

Rule used: lowest canonical CPU float64 tune R; an improvement under 1e-6 is a tie and keeps the current best; a tie otherwise prefers a single model over an ensemble or blend, then lower learning rate, earlier epoch, fewer components, and smaller blend weight. Negative tests cover the threshold, those tie breaks, a blocked row carrying 0, and an unlisted blend weight.

Current best remains 0.30980333573689856. The order of scored rows is:

1. `ensemble_equal_s0_s2` 0.28468175920919014
2. `ensemble_equal_s0_s1_s2` 0.28975002607300254
3. blend w=0.2 0.29645525780787219
4. blend w=0.1 0.30039118477366389
5. blend w=0.05 0.30441273225560084
6. blend w=0.025 0.30693690199632961
7. server primary epoch 3 0.30706943777760692
8. server primary epoch 5 0.30781275392021723
9. server primary epoch 1 0.30951549616992924
10. current best 0.30980333573689856
11. blend w=0 0.30980333573689856
12. server control epoch 3, then epoch 5, then epoch 1

`ensemble_equal_s0_s2` clears 1e-6 by 0.02512157652770841. `would_request_package` is true. `package_written` is false. Current best stays ahead of blend w=0 because their tune R ties.

## Concerns for the reviewer

- `ensemble_equal_s0_s2` is tune-informed: seed 2 was already the best single and seed 1 the worst. The protocol still predeclared it.
- Many tune comparisons were made. There is no noise-significance claim and no hidden confirm or test claim.
- Blend rows depend on the float64 shared-control map, which is not the historical float32 scalar. That needs an explicit correctness decision. The leading ensemble does not use that map.
- Server low-rate and tower low-rate are confounded with platform, and the tower half did not run. No learning-rate superiority claim is available.
- Existing ZIP `dcb8a564…` / CSV `9bae92ea…` and fallback ZIP `61bbac85…` / CSV `08ee9e0e…` were hash-checked and left unchanged.

Stopped here. Packaging waits for independent comparative approval.
