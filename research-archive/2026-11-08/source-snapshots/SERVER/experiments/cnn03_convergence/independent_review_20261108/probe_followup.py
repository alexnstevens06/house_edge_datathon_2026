"""Finish the architecture compare and list continuation artifacts. No training."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path("/home/alexn/Documents/code/datathon_2026/house-edge")
STUDY = REPO / "experiments/cnn03_convergence/continuation_20261108"
sys.path.insert(0, str(STUDY / "source"))
sys.path.insert(0, str(REPO / "experiments/cnn03_convergence/source"))

import torch

torch.set_num_threads(2)

import base_cnn as B
from study_common import load_fp64_model, manifest, sha256_file
from score_fp64 import score_path

B.USE_SIGNAL = False
module = load_fp64_model()
torch.manual_seed(0)
canon = module.CNN(128, 4, 3, 0.3, 64, use_signal=False).double()
other = B.CNN(128, 4, 3, 0.3, 64, use_signal=False).double()
other.load_state_dict(canon.state_dict())
print("state_dict_load_ok", True)
print("keys_equal", set(canon.state_dict()) == set(other.state_dict()))
print("n_tensors", len(canon.state_dict()))
print("base_forward_hardcodes_float", ".float()" in (REPO / "experiments/cnn03_convergence/source/base_cnn.py").read_text(encoding="utf-8"))
print("canonical_forward_uses_embedding_dtype", "s.pay.weight.dtype" in (REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py").read_text(encoding="utf-8"))

pin = manifest()["checkpoints"]
author_names = [
    "curve_C_lr003_s1",
    "curve_C_lr003_s2",
    "curve_control_s0",
    "curve_control_s1",
    "curve_control_s2",
]
import json
author = json.loads((STUDY / "runs/fp64_original_tune.json").read_text(encoding="utf-8"))
for name in author_names:
    got = score_path(REPO / pin[name]["path"])
    print(name, repr(got["tune_R"]), "author", repr(author[name]["tune_R"]), "equal", got["tune_R"] == author[name]["tune_R"])

print("-- continuation artifacts --")
for path in sorted((STUDY / "runs").rglob("*")):
    if path.is_file():
        rel = path.relative_to(STUDY)
        extra = ""
        if path.name == "checkpoint.pt":
            blob = torch.load(path, map_location="cpu", weights_only=False)
            extra = " keys=" + ",".join(sorted(blob))
            if "parent_checkpoint_sha256" in blob:
                extra += " parent_sha=" + blob["parent_checkpoint_sha256"]
                extra += " warm=" + str(blob.get("warm_start_from"))
                extra += " extra=" + str(blob.get("completed_extra_epochs"))
                extra += " opt=" + str("optimizer" in blob)
        print(f"{rel} size={path.stat().st_size}{extra}")
