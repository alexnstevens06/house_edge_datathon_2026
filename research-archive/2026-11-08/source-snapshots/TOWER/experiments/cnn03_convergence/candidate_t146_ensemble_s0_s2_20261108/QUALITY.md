# Quality record, ensemble_equal_s0_s2 package

2026-11-08 02:52 CST. Local QA only. This is not a delivery PASS, not organizer acceptance, and not an automatic canonical replacement.

The packaged weights are the original `curve_C_lr003_s0` and `curve_C_lr003_s2` parents, equal probability weights 0.5 and 0.5, in that order. They are not low-rate continuations, not a blend, and not the shared control. Comparative selection of this row is TUNE 384 only. The pair is tune-informed. Many predeclared TUNE comparisons were already run. There is no significance claim and no hidden-test claim. Earlier packaging printed public full-validation diagnostics, so this record does not claim blind confirmation and does not claim a held-out split was never seen.

Completed-row tune R is `0.28468175920919014`. Current best is `0.30980333573689856`. Recorded improvement is `0.02512157652770841`. Six Tower low-rate rows were not run and are not ranked as worse. Only 2 of 4 low-rate jobs completed. The GPU 5e-5 arm stopped on the native RSS guard after staging. This package does not claim a win over an unrun checkpoint. Blends remain not cleared. No probability correction was fit. The kit-versus-guide dependency contract remains unresolved.

## What was executed

SERVER interpreter `experiments/cnn_runtime/py313_verify_1936/env/bin/python`, CPython 3.13.13, torch 2.11.0+cpu. GPU hidden. 2 threads. No training, no install, no pull, no submission.

Host harness on the fresh ZIP extract exited 0 with `HARNESS_OK`. Both parents ran in one process.

| Run | s0 seconds | s2 seconds | both seconds | total seconds | max RSS KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Host test A | 1.178552 | 1.172952 | 2.351504 | 3.023496 | 336036 |
| Host test B | 1.180922 | 1.173384 | 2.354305 | 3.025918 | 336604 |
| Host validation forward | 0.587226 | 0.587305 | 1.174530 | 1.697130 | 336036 |

Peak host RSS was 336604 KiB. The frozen 2 GiB line is 2097152 KiB. Test CSV cold repeat was bitwise, 1536 ids, 198055 bytes, SHA-256 `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c`. Validation forward was 768 ids, 103390 bytes, SHA-256 `e330a3e4f6ff8155cbce01025346f04e95146fbb0ff6608c74848db5809a5f2d`. Kit format check on the test CSV printed `VALID FORMAT: 1536 queries`. The validation checker was not called. Confirm, full, and test R were not computed.

Verified ensemble map `ensemble_equal_s0_s2.npz` SHA-256 `eb3ad2f66ae1898a97a8aaf9c9f3c2a6a9e173a3de43e161fffc3c8f1341356c`. Package versus that map, per query, kit-normalized and raw: max abs `5.551115123125783e-16`. Tune score `TUNE_ONLY R=0.28468175920919014 n=384 forward=768`. The same R came from the validation CSV restricted to the 384 tune rows.

Refusals covered both components: wrong hash, wrong config hash, signal tensor, extra key, wrong shape, saved dtype other than float32, and a non-finite weight. An extra component directory and a changed weight manifest were refused. Missing data wrote no CSV.

## Docker

Existing local image `ubuntu:24.04` `sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55`. `--pull=never`. No install. Each container extracted the ZIP into a fresh tmpfs, then ran `predict.py`. Network `none`, read-only root, `--cpus=2` (`NanoCpus` 2000000000), `CapDrop ALL`, `no-new-privileges`, user `1000:1000`. `/tmp` 128 MiB and `/output` 32 MiB were tmpfs. The existing Python 3.13 tree was mounted read-only. `NPROC=2` inside the container. Containers were removed after inspect.

Three containers used the previous quality memory posture, 17179869184 bytes with swap 34359738368. All exited 0. OOMKilled false.

| Container | split | s0 seconds | s2 seconds | both seconds | total seconds | max RSS KiB | EXIT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cand-t146-ens-a | test | 1.186166 | 1.177215 | 2.363381 | 3.073312 | 334168 | 0 |
| cand-t146-ens-b | test | 1.177568 | 1.173255 | 2.350823 | 3.040266 | 334524 | 0 |
| cand-t146-ens-val | validation | 0.592621 | 0.596702 | 1.189323 | 1.741607 | 287676 | 0 |

Both test CSVs match the host test SHA above. The validation CSV matches the host validation SHA. That validation file was not scored as a round.

A fourth fresh container, `cand-t146-ens-cap2g`, used the same security flags with memory and memory-plus-swap both set to 2147483648 bytes. EXIT 0, OOMKilled false, NPROC 2. Test infer s0 1.174002 s, s2 1.179949 s, both 2.353951 s, total 3.046236 s, max RSS 334036 KiB. The CSV matches the host test SHA.

## Artifact hashes

| File | SHA-256 |
| --- | --- |
| ZIP `ensemble_equal_s0_s2_fp64_candidate.zip` | `d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991` |
| Test CSV | `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c` |
| Validation forward CSV, not a round score | `e330a3e4f6ff8155cbce01025346f04e95146fbb0ff6608c74848db5809a5f2d` |
| `components/s0/checkpoint.pt` | `d5d714649a93e1e552ccf5419cb01adee311c082af1063f80d8b42f6d8dfe3cb` |
| `components/s0/config.json` | `7ddfe3a4dd1439945c6145d1d5cc2a868653ba30f25523890bf95f5e23aa10ac` |
| `components/s2/checkpoint.pt` | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` |
| `components/s2/config.json` | `a715bfe05eb00494065c7dfcc3ad6d71a3b2bc4ee6788006cf07ce86f3e43695` |
| `model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` |
| `data.py` | `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133` |
| `predict.py` | `364bb0d69cfca6365c001a6ae7322bd8944056d6783855e22863fb59b6d45594` |
| `provenance/weights.json` | `30dad61dbcb82a33fb4a72c5296ae9ad700c2c0ff92fff629403703a40139e9f` |
| `provenance/provenance.json` | `839943476b942de8a8d16a9c209ab70fbe71f3e45a248740af3c8cc6fc5e477a` |
| `reproduce.json` | `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1` |
| `requirements.txt` | `cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab` |

ZIP size 2749917 bytes. `model.py` and `data.py` are the canonical bytes. Both checkpoint files match the original parents byte for byte.

## Untouched, hashed before staging and again after this record

| File | SHA-256 |
| --- | --- |
| `candidate_lr003_s2_20261108/curve_C_lr003_s2_fp64_candidate.zip` | `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29` |
| `candidate_lr003_s2_20261108/refs/candidate_test_a.csv` | `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6` |
| `experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip` | `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` |
| `experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv` | `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` |
| `experiments/mlp_cnn_hpo/review/PINNED.json` | `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb` |
| `experiments/cnn_runtime/fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` |
| `experiments/cnn_runtime/fp64_release_harness2025.py` | `a077e8771abdab58e3d107eeb1da3ecc3ee80ccc2ee08ce72e49b7ab36d89f2c` |
| original `curve_C_lr003_s0/checkpoint.pt` | `d5d714649a93e1e552ccf5419cb01adee311c082af1063f80d8b42f6d8dfe3cb` |
| original `curve_C_lr003_s2/checkpoint.pt` | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` |

Comparison runner, pins, source, and `results/comparison.json` were not rewritten. `comparison.json` is still `9341e9553020e55c8ed863d20b2c31b10be7e45c4d3ffeb4d7a7ceabf991a005`. The path-blocker wording is superseded only by `comparative_t146_20261108/COMPARISON_STATUS_ADDENDUM.md` (`a4d7f36435415fcbd04264edd2e31436a11fcb97a098fb39c5055568b60204fd`). Current Tower reason is the native RSS guard after successful staging.

## Qualification limit

The 3.13.13 / torch 2.11.0+cpu result is the SERVER sandbox. It is not evidence that Tower's native Python 3.12 / Torch 2.14 already has this torch build. Cross-host is not run here. An independent package review has not been run here. `delivery_pass` is false.
