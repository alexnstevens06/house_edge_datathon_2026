# Native gfx1031 runtime

Recorded 2026-11-08 00:35:08 CST. The isolated env is `.venv_gfx1031` in this study folder. The repo `.venv` is still torch `2.9.1+rocm6.4`. No `HSA_OVERRIDE_GFX_VERSION` was set. No driver, system package, or permission change was made.

## Host, read-only, before install

- 2026-11-08 00:28:19 CST. Python 3.12.3 at `/usr/bin/python3`. Kernel `7.0.0-42-generic`. `amdgpu.ko.zst` vermagic matches that kernel.
- Disk `/` had 115G free.
- `rocm-smi --showpids`: no KFD PIDs. Nothing was stopped.

## Package

Official TheRock stable index `https://stable.repo.amd.com/rocm/whl-next/`. RELEASES.md maps Radeon RX 6750 XT / 6700 XT to `gfx1031` and the extra `device-gfx1031`. The stable resolver returned a native device extra, so the nightly index was not used.

```
python3 -m venv .venv_gfx1031
.venv_gfx1031/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1031]"
.venv_gfx1031/bin/python -m pip install numpy pytest
```

| package | version | wheel sha256 |
| --- | --- | --- |
| torch | 2.14.0+rocm10.1.0 | `3182f6f166f42d6f186fd8c4a4ea65e1ed02fba8a98c48f977b68f15cca82636` |
| amd-torch-device-gfx1031 | 2.14.0+rocm10.1.0 | `6965aa78eb8025473a20e167db6ab929cc4ca32d5743d61380232cf65c7cdafa` |
| rocm-sdk-device-gfx1031 | 10.1.0 | `ed3c38994aa4218267a3bc42706178678bee9c986085fc0c08081f35b82a1bb9` |
| rocm-sdk-core | 10.1.0 | `a1a2433334ac61a526c6bae2f58f6bbc9b525f5a9e008e76a1a48d07f64f0703` |
| rocm-sdk-libraries | 10.1.0 | `ffb36d5e1d97e96c191052a10c39a1a7a4ddb4da4a88d3166ab4bf7bfdb94cf6` |
| triton | 3.8.0+git669b31ac.rocm10.1.0 | `2f745c62aebb33167bfb6d586f569fbea67f8fc5181b03798b91a6b9187a625e` |
| rocm (built from the index sdist by pip) | 10.1.0 | `157b2970075c81510dea9e218d8fa269f1c88090213cbc104d18bc58e57bac00` |
| numpy (PyPI) | 2.5.4 | `fbde6962867ee75b48b0ee29b2b9372ec5d617799dbaf38e82dc0596f2f7738a` |
| pytest (PyPI) | 9.2.0 | `07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588` |

HIP reported by this torch: `7.16.26385`. Device: AMD Radeon RX 6700 XT, `gcnArchName` `gfx1031`, major 10 minor 3. `torch.cuda.get_arch_list()` includes `gfx1031`. `amd-torch-device-gfx1031` metadata summary is "AMD device kernels (gfx1031) for torch" and depends on `rocm-sdk-device-gfx1031 ~= 10.1.0`.

Channel note: this is the stable TheRock aggregate. SUPPORTED_GPUS.md lists Linux RDNA2 gfx1031 as build passing, sanity tested, and release ready inside that development project. That is separate from the older general ROCm matrix, which does not list consumer RX 6000.

## Kernel gate, before the recorded trial

With `HSA_OVERRIDE_GFX_VERSION` unset, this interpreter completed `torch.cuda.synchronize()` after each of:

- `torch.zeros(1, device="cuda")` returned 0
- `torch.randn` on CUDA, all finite
- `Linear` backward, finite gradient
- `Conv1d` backward, finite gradient

`pytest tests/test_frozen_heads.py`: 10 passed, including `test_frozen_trunk_survives_gpu_head_step`.

MIOpen printed `libMIOpenCKGroupedConv_gfx1031.so` missing, and later `GemmFwdRest` asked for 98304 bytes of workspace it was not given. Those are warnings. The probed Conv1d backward and the trunk feature pass still finished. No extra library was installed.

## One dtype fix, then the same schedule

Pid 32633 (00:33:03 CST) trained the heads on GPU. Epoch-0 logits matched the trunk. First control-epoch loss was 0.804721. Specialist last losses were 0.668661, 0.728859, and 0.921643. CPU fp64 scoring then raised `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != double` because `encode()` forced float32 mask and tick tensors while the scorer trunk was double. Torch 2.14 does not promote that mix. No `result.json` was written. That log is `runs/train_seed0_attempt1_prescore.log`. The smi sample from that process is under `runs/seed0_attempt1_gpu_only/`.

`source/frozen_heads.py` `encode()` now builds those tensors in `self.pay.weight.dtype`. Float32 GPU training is unchanged. The file sha256 is `4b8ecd90f1975613d0f27e4fc5929850b8e4f0ec20bd6dc5fa7c9918352e8873`. `train.py` was not edited. Schedule stayed seed 0, 10 AdamW steps, lr 0.001, wd 0, 645-parameter heads, counts 250/267/251, tune only.

Pytest after the fix: 10 passed. GPU was idle. The same command was run again.

## Recorded trial

Pid 33141, 2026-11-08 00:34:28 CST, device AMD Radeon RX 6700 XT, torch 2.14.0+rocm10.1.0.

- Before launch: no KFD PIDs (`runs/seed0/gpu_users_before.txt`).
- During featurize (`runs/seed0/gpu_during_featurize.txt`): GPU use 5%, pid 33141 VRAM used 519233536 bytes, device VRAM used 1133150208 of 12868124672. Feature pass 1.38s.
- First epoch control loss 0.804721, same as pid 32633. `epoch0_logit_match` true.
- After the shared head's 10 steps (`runs/seed0/gpu_after_first_head.txt`): GPU use 1%, pid 33141 VRAM used 630345728 bytes, device VRAM used 1253232640. Peak allocated by the process: 122.2 MiB.
- Final train loss: control 0.786920, signal -1 0.668661, signal 0 0.728859, signal +1 0.921643.
- Trunk state hash before and after `d769a81c7f65fbfae7dcda19f18d822a9031eec187e9e7e2963f775c2b33c67c`. Frozen parameter count 367109. File sha256 `e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760`.
- Tune R (384 queries, lower better): baseline 0.358981, shared 0.350971 (delta -0.008009), routed 0.365661 (delta vs baseline +0.006680, vs shared +0.014690).
- Confirm, full, and test were not scored. No second seed.

The shared head beat the frozen baseline on tune. Routing three signal heads did worse than both the shared head and the baseline. This study stops here.

`BLOCKER.md` is still the ROCm 6.4 record. A copy is `BLOCKER_rocm64.md`. The old wheel manifest is `MANIFEST.json` and `MANIFEST_rocm64.json`. This runtime is `manifest_gfx1031.json`.
