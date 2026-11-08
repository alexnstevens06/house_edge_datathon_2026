# Tower 5e-5 stopped on the frozen 2 GiB RSS guard

2026-11-08 02:25:45 CST. The two prescribed Tower jobs were not launched. No epoch was trained. `experiments/cnn03_convergence/lowrate_20261108/runs/` does not exist. No trainer pid, utilization sample, VRAM sample, step count, or EXIT from `train_lowrate.py` exists for this pair. `TOWER_TRAINING_DONE.md` was not written. `TOWER_TRAINING_BLOCKER.md` is unchanged and records only the earlier 02:13 CST `assert_input_pins` FileNotFoundError. Those two exits were before any GPU step and are not training.

## Exact stop

Required native FP32 preflight, one forward and one backward, batch 128, no optimizer step, device-memory fraction 0.40. Interpreter `experiments/cnn_experts/frozen_feature_heads_20261108/.venv_gfx1031/bin/python` (torch 2.14.0+rocm10.1.0, HIP 7.16.26385). Probe log closed 2026-11-08 02:21:21 -0600. `HSA_OVERRIDE_GFX_VERSION` was unset. `rocm-smi --showpids` showed no KFD process before the probe.

```
FP32_PROBE_OK loss=1.593103 gpu_mb=1144.0 rss_kb=2174996 logit=torch.float32
host RSS 2174996 KB exceeds 2 GiB
FP32_EXIT:1
```

`source/preflight_fp32.py` prints that line and then exits 1 when `ru_maxrss` is above `2 * 1024 * 1024` (2097152 KB). Measured peak was 2174996 KB, 77844 KB over the cap. Logits were float32. Allocated device memory was 1144.0 MB. The loss is finite. MIOpen warned that `libMIOpenCKGroupedConv_gfx1031.so` is missing and that GemmFwdRest, GemmBwdRest, and GemmWrwUniversal wanted a 98304-byte workspace. Those warnings did not stop the forward or the backward.

The same process limit is in `source/train_lowrate.py` at line 214 (after `build_train`, before the model is moved) and line 275 (after each epoch, before the checkpoint write). The preflight is the required memory gate and it already failed, so the primary and control commands were not started. The guard, recipe, batch, thread cap, fraction, and paths were not edited.

## Checks that passed

Unit script, same interpreter: `LOW_RATE_TESTS_OK failures=0 windows=18567 steps=730` and `TESTS_EXIT:0`.

CPU float64 probe, CUDA/HIP hidden by that script: `FP64_PROBE_OK params=43 logit=torch.float64 softmax=torch.float64` and `FP64_EXIT:0`.

`sha256sum -c SOURCE_SHA256.txt` at 02:25 CST: all 10 files OK. Trainer `source/train_lowrate.py` is `206ee996bc6a48b34a5e4027e90542e5cb63a787fe36e7ab7aab1f90f0b04066`. Contract `source/contract.py` is `d908aefe6eab70cbedf70d0bf2b7fc2d2132cd5bafce7e3d844fc78cccaa3025`.

`INPUT_STAGING_VERIFIED.json` (2026-11-08T02:19:46-06:00): all 14 manifest pins rehashed, 0 bad, 0 symlinks. The six `experiments/cnn03_convergence/runs/curve_*/checkpoint.pt` files are input snapshots, not Tower training runs.

Chosen `--parent` files, regular files, hashes unchanged:

| arm | path | sha256 |
| --- | --- | --- |
| primary | `experiments/cnn03_convergence/ensemble_blend_20261108/inputs/curve_C_lr003_s2/checkpoint.pt` | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` |
| control | `experiments/cnn03_convergence/ensemble_blend_20261108/inputs/curve_control_s2/checkpoint.pt` | `47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9` |

After the probe, the same interpreter reported device `AMD Radeon RX 6700 XT`. That query was not a training job. No KFD process was running at the 02:25 CST check. The server 2e-5 pair was not started. Nothing was scored on Tower.
