# Tower 5e-5 training blocked

2026-11-08 02:13:49 CST. No epoch was trained. No `runs/` directory was created. `HSA_OVERRIDE_GFX_VERSION` was unset. Pid 64410 is gone. `rocm-smi --showpids` showed no KFD process before and after the attempts. `nvidia-smi` is not installed. Frozen `sha256sum -c SOURCE_SHA256.txt` passed for all 10 listed files. Those files were not edited.

## Exact stop

`source/preflight_fp32.py` and both prescribed `train_lowrate.py` commands exit 1 inside `assert_input_pins()` before a forward, a backward, an optimizer step, or `mkdir` of the output directory.

```
FileNotFoundError: [Errno 2] No such file or directory:
/home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py
```

`contract.py` line 105 hashes `MANIFEST.json` `sources` in order. That model path is the first missing file. The same function would next miss `experiments/cnn_runtime/fresh_fp64_cnn03_1956/data.py`, then all six parents under `experiments/cnn03_convergence/runs/`:

- `curve_C_lr003_s0/checkpoint.pt`
- `curve_C_lr003_s1/checkpoint.pt`
- `curve_C_lr003_s2/checkpoint.pt`
- `curve_control_s0/checkpoint.pt`
- `curve_control_s1/checkpoint.pt`
- `curve_control_s2/checkpoint.pt`

These present pins matched: `base_cnn.py`, `conv_train.py`, `src/house_edge/data.py`, `src/house_edge/scoring.py`, `kit/slot_forecasting/scoring.py`, `runs/splits.json`.

## Staged seed-2 parents

The `--parent` paths the handoff names do have the required bytes. The trainer never reached `parent_errors`, because `assert_input_pins()` runs first and only looks at the `runs/` paths.

| arm | staged path | sha256 |
| --- | --- | --- |
| primary | `experiments/cnn03_convergence/ensemble_blend_20261108/inputs/curve_C_lr003_s2/checkpoint.pt` | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` |
| control | `experiments/cnn03_convergence/ensemble_blend_20261108/inputs/curve_control_s2/checkpoint.pt` | `47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9` |

The same model and data bytes as the manifest (`7e63ccf7…` and `af23684b…`) are in `experiments/cnn03_convergence/candidate_lr003_s2_20261108/unpacked/`. They were not copied onto the pinned paths. No hash guard was patched.

## Commands and exits

Interpreter: `experiments/cnn_experts/frozen_feature_heads_20261108/.venv_gfx1031/bin/python`.

Primary exit 1. Log: `tower_primary_guard.log`.
Control exit 1. Log: `tower_control_guard.log`.

The server unit suite was not re-run here. Its first test calls the same `assert_input_pins()`. There is no Tower FP32 probe loss, device utilization, or trainer pid, because the process never allocated a training tensor.

The server 2e-5 pair was not started. Ensemble results were not rewritten.
