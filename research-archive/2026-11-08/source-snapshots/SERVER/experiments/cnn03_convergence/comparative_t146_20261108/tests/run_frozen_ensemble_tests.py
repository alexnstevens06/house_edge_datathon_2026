"""Run the mirrored ensemble tests in-process. Does not import evaluate.py."""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

REPO = Path(__file__).resolve().parents[4]
TEST_PATH = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108/tests/test_matrix.py"


def load_tests():
    spec = importlib.util.spec_from_file_location("frozen_ensemble_test_matrix", TEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen ensemble tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_tests()
    names = [
        "test_pinned_files_and_source_have_no_float_cast",
        "test_wrong_hash_config_keys_shape_signal_dtype_and_nonfinite",
        "test_hard_control_bind_and_curve_controls_present",
        "test_unknown_and_nontune_ids_are_refused",
        "test_window_has_no_future_and_missing_is_not_zero",
        "test_query_order_does_not_change_ids",
        "test_simplex_weight0_identity_and_arithmetic_mean",
        "test_canonical_fp64_logits_softmax_on_real_seed2_state",
        "test_split_file_matches_manifest_contract",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name in names:
            fn = getattr(module, name)
            if name == "test_wrong_hash_config_keys_shape_signal_dtype_and_nonfinite":
                fn(tmp_path)
            else:
                fn()
            print(f"FROZEN_TEST_OK {name}", flush=True)
    print(f"FROZEN_ENSEMBLE_TESTS_OK n={len(names)}")


if __name__ == "__main__":
    main()
