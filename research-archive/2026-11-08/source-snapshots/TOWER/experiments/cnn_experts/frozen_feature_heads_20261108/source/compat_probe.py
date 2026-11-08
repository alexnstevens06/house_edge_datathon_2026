"""Report which ROCm ops this process can actually run. Does not train."""
import torch

print("torch", torch.__version__)
print("is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(2)
props = torch.cuda.get_device_properties(0)
print("name", torch.cuda.get_device_name(0))
for key in ("gcnArchName", "major", "minor", "total_memory"):
    print(key, getattr(props, key, None))


def attempt(label, fn):
    try:
        fn()
        torch.cuda.synchronize()
        print("OK", label)
    except Exception as exc:
        print("FAIL", label, type(exc).__name__, exc)


def _linear():
    x = torch.randn(4, 16, device="cuda")
    layer = torch.nn.Linear(16, 5).cuda()
    layer(x).square().mean().backward()


def _conv():
    conv = torch.nn.Conv1d(4, 4, 3).cuda()
    conv(torch.randn(2, 4, 12, device="cuda")).sum().backward()


attempt("zeros", lambda: torch.zeros(1, device="cuda"))
attempt("randn", lambda: torch.randn(4, 16, device="cuda"))
attempt("linear_backward", _linear)
attempt("conv_backward", _conv)
