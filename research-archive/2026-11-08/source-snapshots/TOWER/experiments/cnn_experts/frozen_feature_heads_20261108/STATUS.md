2026-11-08 00:12:00 CST plan frozen. PLAN.md sha256 4bf04d32a65c1cc8d56342c6feb48d80e81e32a772fabeb91d63483f21f1ace4. Trunk inputs/history_pretrain.pt sha256 e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760. GPU free at 00:09:35 CST: no KFD PIDs, VRAM used 374927360 of 12868124672 bytes. Device name AMD Radeon RX 6700 XT.
2026-11-08 00:16:00 CST CPU tests pass: 9 passed (leakage, query order, missing-vs-0, routing, seed guard, label counts 250/267/251, dropout eval stability, fp64 scorer invariance). GPU test not in that count.
2026-11-08 00:18:46 CST training NOT started. BLOCKER: installed torch 2.9.1+rocm6.4 arch list has gfx1030 and not gfx1031; torch.zeros on cuda raises hipErrorInvalidDeviceFunction. No tune R. See BLOCKER.md and runs/compat_probe.txt.
2026-11-08 00:33:03 CST training started pid 32633 device AMD Radeon RX 6700 XT torch 2.14.0+rocm10.1.0
2026-11-08 00:33:06 CST first epoch control loss 0.804721 pid 32633
2026-11-08 00:34:28 CST training started pid 33141 device AMD Radeon RX 6700 XT torch 2.14.0+rocm10.1.0
2026-11-08 00:34:31 CST first epoch control loss 0.804721 pid 33141
2026-11-08 00:34:36 CST final seed 0 tune R baseline 0.358981 shared 0.350971 routed 0.365661 peak_gpu_mb 122.2 pid 33141
2026-11-08 00:35:08 CST seed 0 complete on native gfx1031 torch 2.14.0+rocm10.1.0. Recorded trainer pid 33141 (same schedule as pid 32633, which died in CPU fp64 scoring before result.json; encode now follows parameter dtype; losses matched). Tune R baseline 0.358981 shared 0.350971 routed 0.365661. No confirm/test. No further seed. See NATIVE_RUNTIME.md.
