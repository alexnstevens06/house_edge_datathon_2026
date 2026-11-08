"""Capture golden logits from the CURRENT models.py (pre TASK-B edit) for the
bit-for-bit regression test. Run once; output: readout/golden_query.npz"""
import os
import sys

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "src"))

import numpy as np
import torch

from house_edge.data import MISSING, N_MACHINES, Episode
from house_edge.models import NextSpinTransformer
from house_edge.windows import Example, WindowCollator

D_MODEL, LAYERS, HEADS, FF, WINDOW = 16, 1, 2, 32, 16


def make_episode(seed):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 5, size=(40, N_MACHINES)).astype(np.int8)
    obs[rng.random((40, N_MACHINES)) < 0.2] = MISSING
    ids = tuple(f"syn-e000-m{s:03d}" for s in range(N_MACHINES))
    types = rng.integers(0, 3, size=N_MACHINES)
    return Episode("syn-e000", ids, types, obs, 30)


def main():
    out = {}
    for pos in ("abs", "lag"):
        os.environ["HE_POS"] = pos
        ep = make_episode(5)
        for arch in ("dense", "sparse"):
            torch.manual_seed(1234)
            model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS,
                                        num_layers=LAYERS, dim_ff=FF)
            model.eval()
            collator = WindowCollator([ep], arch, WINDOW)
            batch = collator([Example(0, 20, None), Example(0, 30, None)])
            with torch.no_grad():
                logits = model.forward(batch)
            out[f"{pos}_{arch}"] = logits.numpy()
            print(pos, arch, "sum", float(logits.sum()))
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_query.npz"), **out)
    print("saved golden_query.npz")


if __name__ == "__main__":
    main()
