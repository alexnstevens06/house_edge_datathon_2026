"""Call the frozen ensemble tests without installing pytest."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[4]
TEST = ROOT / "experiments/cnn03_convergence/ensemble_blend_20261108/tests/test_matrix.py"
spec = importlib.util.spec_from_file_location("frozen_test_matrix", TEST)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

names = [name for name in dir(module) if name.startswith("test_")]
failed = 0
for name in names:
    fn = getattr(module, name)
    try:
        if "tmp_path" in fn.__code__.co_varnames:
            with tempfile.TemporaryDirectory(prefix="t146-ens-", dir=Path(__file__).resolve().parent) as tmp:
                fn(Path(tmp))
        else:
            fn()
        print(f"PASS {name}", flush=True)
    except Exception:
        failed += 1
        print(f"FAIL {name}", flush=True)
        traceback.print_exc()
print(f"ENSEMBLE_TESTS failed={failed} ran={len(names)}")
if failed:
    raise SystemExit(1)
