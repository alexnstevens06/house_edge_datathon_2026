# Low-rate status (server pair finished, no further training)

2026-11-08 01:51 CST. The server 2e-5 pair has exited. No more training is being launched from this study. The tower 5e-5 pair was not started here. Nothing was packaged.

Frozen source is `SOURCE_SHA256.txt`. Coordinator copy instructions are `HANDOFF_READY.md`. Parents were read and not rewritten. After the runs, their SHA-256 values were still `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` and `47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9`.

## Server jobs

GPU before launch: RTX 5080, 0% utilization, 2 MiB of 16303 MiB, no compute apps. Both trainers then ran together. Process-local TF32 flags were off (`matmul` and `cudnn` `allow_tf32` false, `fp32_precision` `ieee`). Fresh AdamW at 2e-5 after a weights-only load. Not an exact optimizer resume. Batch 128 FP32, against the parents' historical bf16 autocast batch 512. Peak allocated device memory was 1145.8 MB in each process.

| Arm | PID | Exit | Steps | Train seconds | Epoch losses |
| --- | --- | --- | --- | --- | --- |
| primary | 716396 | 0 | 730 | 29.405 | 0.772911, 0.772381, 0.772355, 0.772453, 0.773390 |
| control | 716397 | 0 | 730 | 29.402 | 0.788687, 0.789173, 0.788125, 0.788984, 0.789095 |

First epoch on the primary was loss 0.772911 at 100% GPU and 3299 MiB used by the pair. First epoch on the control was loss 0.788687 on the same device snapshot. Checkpoints, optimizer state, RNG, and `POST_RUN_SNAPSHOT.json` were written at epochs 1, 3, and 5 before scoring.

| Run | Epoch | Checkpoint SHA-256 |
| --- | --- | --- |
| server_primary_lr2e-5 | 1 | `6fea985c054638ae852b59685fdbf6e7d43c44b24fa8b1db31f46ae0a577687c` |
| server_primary_lr2e-5 | 3 | `17d66a052d5ccdef50af14468c24a029f77fe3943bd7f991aafcf667c7faedb5` |
| server_primary_lr2e-5 | 5 | `7b8368b5a6684b2fbc65b97ecd70e255fd626840da03d06feb34664c50f0954e` |
| server_control_lr2e-5 | 1 | `ad6651670b2557044c205668d6f11700b718df82d74d10db151219d96facf7cc` |
| server_control_lr2e-5 | 3 | `2c250ec33982b897179ff50c832a7fefe60363e540ebdd53e4ae15046db6e4fa` |
| server_control_lr2e-5 | 5 | `1407fc091fe49f30ff9e1bb8197c56aadeb7119aa7c4530254019c4f6f8a88ae` |

## CPU float64 tune scores

Interpreter: `experiments/cnn_runtime/py313_verify_1936/env/bin/python`, CPython 3.13.13, torch 2.11.0+cpu, `CUDA_VISIBLE_DEVICES` empty, 2 threads. 384 tune queries, 768 forwarded. Confirm, full, and test metrics were not computed. JSON files are under each run's `scores/` directory. The lines below are the scorer's printed values.

| Run | Epoch 1 | Epoch 3 | Epoch 5 |
| --- | --- | --- | --- |
| server primary 2e-5 | 0.30951549616992924 | 0.30706943777760692 | 0.30781275392021723 |
| server control 2e-5 | 0.33413621090622814 | 0.33213683848995773 | 0.33244487608328904 |

Current best bar is 0.30980333573689856. The original control seed-2 tune R is 0.3334821793902223. The old canonical 0.33206398205344967 is not the bar. On this tune metric, all three saved primary checkpoints are lower than the current best, and the lowest of these six is primary epoch 3. All three saved control checkpoints remain above the current best. Inside this host, at the same 2e-5 recipe, the primary arm is lower than the control arm at epochs 1, 3, and 5.

These six checkpoints were predeclared. The tune split was used for selection, so the usual multiplicity caveat applies. The tower 5e-5 pair is the other half of the four-job set and is not in this table. Platform is confounded with learning rate, so these numbers do not support a claim that 2e-5 is a better rate than 5e-5. No candidate was packaged.
