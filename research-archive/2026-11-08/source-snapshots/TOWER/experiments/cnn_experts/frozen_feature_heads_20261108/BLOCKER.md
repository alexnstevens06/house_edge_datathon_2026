2026-11-08 00:18:46 CST BLOCKER

ROCm sees the tower GPU and PyTorch reports `torch.cuda.is_available() == True`, but no device kernel in this runtime will launch. Training was not started. No tune R was computed. CPU training was not used as a substitute.

## What failed

Process: `/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python`  
Torch: `2.9.1+rocm6.4`  
HIP: `6.4.43484-123eb5128`  
`torch.cuda.get_device_name(0)`: AMD Radeon RX 6700 XT  
`torch.cuda.get_device_properties(0).gcnArchName`: `gfx1031`  
`rocminfo` agent Name: `gfx1031` (Marketing Name: AMD Radeon RX 6700 XT), ISA `amdgcn-amd-amdhsa--gfx1031`

`torch.cuda.get_arch_list()`:

`gfx900, gfx906, gfx908, gfx90a, gfx942, gfx1030, gfx1100, gfx1101, gfx1102, gfx1200, gfx1201`

`gfx1031` is not in that list. `gfx1030` is. The first device op fails before any model work:

`torch.zeros(1, device="cuda")` → `hipErrorInvalidDeviceFunction`

The same error is returned for `torch.randn`, a `Linear` backward, and a `Conv1d` backward. Full log: `runs/compat_probe.txt`. Pytest `test_frozen_trunk_survives_gpu_head_step` fails in `gpu_op_check` at `source/frozen_heads.py` on that allocation.

## What was ruled out

At 2026-11-08 00:09:35 CST, `rocm-smi` showed no KFD PIDs. VRAM total 12868124672 bytes, used 374927360 bytes. This is not a busy-GPU failure.

No package was installed. `HSA_OVERRIDE_GFX_VERSION` was not set. That override would report a different ISA than the card and is not a successful kernel run on `gfx1031`.

## What still passed

Trunk file sha256 `e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760`. State loads strict into the no-signal CNN03 (367109 parameters, `head.2` is 645). CPU tests: 9 passed. The GPU test is the failure above.
