"""One FP32 forward and backward at batch 128. No optimizer step and no tune score."""
from __future__ import annotations

import os
import resource
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (  # noqa: E402
    BS,
    CONV_SOURCE,
    DROP,
    EXPECTED_WINDOWS,
    H,
    KERNEL,
    LAYERS,
    W,
    assert_input_pins,
    refuse_if_no_cuda,
)

sys.path.insert(0, str(CONV_SOURCE))
import base_cnn as B  # noqa: E402

B.USE_SIGNAL = False
HOST_RSS_LIMIT_KB = 2 * 1024 * 1024


def main() -> None:
    import torch
    refuse_if_no_cuda(torch.cuda.is_available())
    assert_input_pins()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.40)
    tr = B.load_split(os.path.join(B.PUB, "train"), with_labels=True)
    X, T, TY, SG, Y = B.build_train(tr, W)
    if len(X) != EXPECTED_WINDOWS or int(SG.sum()) != 0:
        raise SystemExit(f"train construction refused n={len(X)}")
    dev = "cuda"
    model = B.CNN(H, LAYERS, KERNEL, DROP, W, use_signal=False).to(dev)
    rows = torch.arange(BS, device=dev)
    batch = [item.to(dev)[rows] for item in (X, T, TY, SG, Y)]
    logits = model(*batch[:-1])
    loss = B.masked_loss(logits, batch[-1])
    loss.backward()
    allocated = torch.cuda.max_memory_allocated() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"FP32_PROBE_OK loss={float(loss):.6f} gpu_mb={allocated:.1f} rss_kb={rss} logit={logits.dtype}")
    if rss > HOST_RSS_LIMIT_KB:
        raise SystemExit(f"host RSS {rss} KB exceeds 2 GiB")
    if logits.dtype != torch.float32:
        raise SystemExit(f"training forward dtype {logits.dtype}")


if __name__ == "__main__":
    main()
