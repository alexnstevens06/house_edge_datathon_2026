"""Real parent weights, canonical FP64 model, one window. No tune score."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import DROP, FP64_ENTRY, H, KERNEL, LAYERS, REPO, W, assert_input_pins, manifest, sha256_file, state_errors  # noqa: E402


def main() -> None:
    assert_input_pins()
    import torch
    torch.set_num_threads(2)
    spec = manifest()["checkpoints"]["curve_C_lr003_s2"]
    path = REPO / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise SystemExit("primary parent hash drift")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    errors = state_errors(blob["state"])
    if errors:
        raise SystemExit("state refused: " + "; ".join(errors))
    sys.path.insert(0, str(FP64_ENTRY))
    model_spec = importlib.util.spec_from_file_location("lowrate_fp64_probe", FP64_ENTRY / "model.py")
    module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(module)
    module.USE_SIGNAL = False
    net = module.CNN(H, LAYERS, KERNEL, DROP, W, use_signal=False)
    net.load_state_dict(blob["state"])
    net.double()
    net.eval()
    sys.path.insert(0, str(REPO / "src"))
    from house_edge.data import load_split
    validation = load_split(REPO / "kit" / "public" / "validation")
    query, _episodes = module.build_query(validation, W)
    one = [item[:1] for item in query]
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise SystemExit("autocast is enabled")
    with torch.no_grad():
        logits = net(*[item for item in one])
        probs = torch.softmax(logits, dim=-1)
    if logits.dtype != torch.float64 or probs.dtype != torch.float64:
        raise SystemExit(f"dtype {logits.dtype} {probs.dtype}")
    floating = [p for p in net.parameters() if p.is_floating_point()]
    if len(floating) != 43 or any(p.dtype != torch.float64 for p in floating):
        raise SystemExit("parameters did not stay float64")
    print(f"FP64_PROBE_OK params={len(floating)} logit={logits.dtype} softmax={probs.dtype}")


if __name__ == "__main__":
    main()
