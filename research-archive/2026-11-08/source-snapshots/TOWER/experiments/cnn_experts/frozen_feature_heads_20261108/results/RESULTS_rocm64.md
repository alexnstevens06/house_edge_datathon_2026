# Frozen feature heads, seed 0

No tune R. Training did not run.

The tower GPU is an AMD Radeon RX 6700 XT (`gfx1031`). The study runtime is PyTorch `2.9.1+rocm6.4`, whose compiled architectures include `gfx1030` and do not include `gfx1031`. `torch.cuda.is_available()` is true, and `torch.zeros(1, device="cuda")` raises `hipErrorInvalidDeviceFunction`. Linear and convolution backwards fail the same way. Evidence is `runs/compat_probe.txt` and `BLOCKER.md`.

The trunk file matches sha256 `e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760` and loads as the 367109-parameter no-signal CNN. The planned head is the existing `head.2` linear layer, 645 parameters, copied into one shared head and three signal heads. That schedule was not executed.

CPU checks that do not allocate device tensors passed (9). The GPU backward test failed. No confirm, full, or test score was computed.
