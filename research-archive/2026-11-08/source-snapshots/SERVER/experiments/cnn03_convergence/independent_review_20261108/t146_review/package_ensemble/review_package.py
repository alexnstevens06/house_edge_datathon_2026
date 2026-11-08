"""Independent exact-ZIP review. Writes only this directory. Does not call harness main."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[4]
CAND = REPO / "experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108"
ZIP_PATH = CAND / "ensemble_equal_s0_s2_fp64_candidate.zip"
SANDBOX = REVIEW / "candidate_t146_ensemble_s0_s2_20261108"
FRESH = SANDBOX / "fresh_zip"
HARNESS_OUT = SANDBOX / "harness_out"
WORK = SANDBOX / "_work"
PY = REPO / "experiments/cnn_runtime/py313_verify_1936/env/bin/python"

EXPECT = {
    "zip": "d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991",
    "test_csv": "1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c",
    "receipt": "106e1deda225ca8902e1c69f75cb7f093910253bcfd927d2ffd83a23025806eb",
    "predict": "364bb0d69cfca6365c001a6ae7322bd8944056d6783855e22863fb59b6d45594",
    "model": "7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a",
    "data": "af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133",
    "s0": "d5d714649a93e1e552ccf5419cb01adee311c082af1063f80d8b42f6d8dfe3cb",
    "s2": "bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71",
    "best_zip": "dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29",
    "best_csv": "9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6",
    "fallback_zip": "61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599",
    "fallback_csv": "08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d",
    "pinned": "cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb",
    "prod_harness": "515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2",
    "release_harness": "a077e8771abdab58e3d107eeb1da3ecc3ee80ccc2ee08ce72e49b7ab36d89f2c",
    "addendum": "a4d7f36435415fcbd04264edd2e31436a11fcb97a098fb39c5055568b60204fd",
    "tower_rss": "a42361ef2f618ab6a6b89a43246ebf58b4dd7935846c1f55cb10ae85dc193acc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"REVIEW_FAIL {message}")
    print(f"OK {message}", flush=True)


def extract_fresh() -> list[str]:
    if FRESH.exists():
        raise SystemExit(f"REVIEW_FAIL fresh extract already exists: {FRESH}")
    FRESH.mkdir(parents=True)
    names = []
    with zipfile.ZipFile(ZIP_PATH) as blob:
        for info in blob.infolist():
            name = info.filename
            if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts:
                raise SystemExit(f"REVIEW_FAIL unsafe zip member {name}")
            if info.is_symlink() if hasattr(info, "is_symlink") else False:
                raise SystemExit(f"REVIEW_FAIL symlink member {name}")
            target = (FRESH / name).resolve()
            if FRESH.resolve() not in target.parents and target != FRESH.resolve():
                raise SystemExit(f"REVIEW_FAIL zip path escapes {name}")
            names.append(name)
        blob.extractall(FRESH)
    for path in FRESH.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"REVIEW_FAIL extracted symlink {path}")
    print(f"EXTRACT members={len(names)}", flush=True)
    for name in names:
        print(f"MEMBER {name}", flush=True)
    return names


def main() -> None:
    import torch
    torch.set_num_threads(2)
    print(f"python {sys.version.split()[0]} torch {torch.__version__}", flush=True)
    pins = {
        "zip": ZIP_PATH,
        "receipt": CAND / "SOURCE_HASHES.txt",
        "best_zip": REPO / "experiments/cnn03_convergence/candidate_lr003_s2_20261108/curve_C_lr003_s2_fp64_candidate.zip",
        "best_csv": REPO / "experiments/cnn03_convergence/candidate_lr003_s2_20261108/refs/candidate_test_a.csv",
        "fallback_zip": REPO / "experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip",
        "fallback_csv": REPO / "experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv",
        "pinned": REPO / "experiments/mlp_cnn_hpo/review/PINNED.json",
        "prod_harness": REPO / "experiments/cnn_runtime/fp64_harness.py",
        "release_harness": REPO / "experiments/cnn_runtime/fp64_release_harness2025.py",
        "addendum": REPO / "experiments/cnn03_convergence/comparative_t146_20261108/COMPARISON_STATUS_ADDENDUM.md",
        "tower_rss": REPO / "experiments/cnn03_convergence/lowrate_20261108/TOWER_FINAL_RSS_GUARD.md",
        "test_csv": CAND / "refs/candidate_test_a.csv",
    }
    for key, path in pins.items():
        got = sha256_file(path)
        require(got == EXPECT[key], f"{key} {got}")
    require(ZIP_PATH.stat().st_size == 2749917, f"zip bytes {ZIP_PATH.stat().st_size}")
    require(pins["test_csv"].stat().st_size == 198055, f"test csv bytes {pins['test_csv'].stat().st_size}")
    tune_before = sha256_file(CAND / "tune_only.json")
    print(f"AUTHOR_TUNE_ONLY_BEFORE {tune_before}", flush=True)

    listed = {}
    for line in (CAND / "SOURCE_HASHES.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        target = CAND / rel if not rel.startswith("experiments/") else REPO / rel
        got = sha256_file(target)
        listed[rel] = got == digest
        if got != digest:
            print(f"MISMATCH {rel} {got}", flush=True)
    require(all(listed.values()) and len(listed) > 0, f"SOURCE_HASHES rows={len(listed)} bad={sum(not v for v in listed.values())}")

    names = extract_fresh()
    require("components/s0/checkpoint.pt" in names and "components/s2/checkpoint.pt" in names, "both checkpoints in zip")
    require("components/s1/" not in "".join(names), "no s1 member")
    fresh_map = {
        "model.py": EXPECT["model"],
        "data.py": EXPECT["data"],
        "predict.py": EXPECT["predict"],
        "components/s0/checkpoint.pt": EXPECT["s0"],
        "components/s2/checkpoint.pt": EXPECT["s2"],
    }
    for rel, digest in fresh_map.items():
        require(sha256_file(FRESH / rel) == digest, f"fresh {rel}")
    original_s0 = REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s0/checkpoint.pt"
    original_s2 = REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt"
    require((FRESH / "components/s0/checkpoint.pt").read_bytes() == original_s0.read_bytes(), "s0 bytes match original parent")
    require((FRESH / "components/s2/checkpoint.pt").read_bytes() == original_s2.read_bytes(), "s2 bytes match original parent")
    canonical_model = REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py"
    canonical_data = REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956/data.py"
    require((FRESH / "model.py").read_bytes() == canonical_model.read_bytes(), "model.py matches canonical")
    require((FRESH / "data.py").read_bytes() == canonical_data.read_bytes(), "data.py matches canonical")

    release = REPO / "experiments/cnn_runtime/fp64_release_harness2025.py"
    harness = CAND / "fp64_candidate_harness.py"
    diff = subprocess.run(["diff", "-u", str(release), str(harness)], text=True, capture_output=True)
    (REVIEW / "HARNESS_DIFF_RERUN.txt").write_text(diff.stdout, encoding="utf-8")
    published = (CAND / "HARNESS_DIFF.txt").read_text(encoding="utf-8").splitlines()
    rerun = diff.stdout.splitlines()

    def body(lines: list[str]) -> list[str]:
        return [line for line in lines if not line.startswith("--- ") and not line.startswith("+++ ")]

    require(body(published) == body(rerun), f"harness diff body match published={len(published)} rerun={len(rerun)} rc={diff.returncode}")
    print(f"HARNESS_DIFF_BODY_MATCH lines={len(body(rerun))}", flush=True)

    HARNESS_OUT.mkdir(parents=True)
    WORK.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("cand_harness_review", harness)
    if spec is None or spec.loader is None:
        raise SystemExit("REVIEW_FAIL could not load harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORK = WORK
    module.OUT = HARNESS_OUT
    module.test_source_has_no_global_patch(FRESH)
    module.test_window_and_missing(FRESH)
    module.test_dtype_and_forecast_causality(FRESH)
    module.test_query_order(FRESH)
    module.test_refusals(FRESH)
    module.full_runs("review", FRESH)
    tune_out = SANDBOX / "tune_only.json"
    proc = subprocess.run(
        [
            str(PY),
            str(CAND / "score_tune_only.py"),
            "--entry",
            str(FRESH),
            "--val-csv",
            str(HARNESS_OUT / "review_val.csv"),
            "--out",
            str(tune_out),
        ],
        cwd=REPO,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    print(proc.stdout, flush=True)
    print(proc.stderr, flush=True)
    require(proc.returncode == 0, f"score_tune_only rc={proc.returncode}")
    tune_after = sha256_file(CAND / "tune_only.json")
    require(tune_after == tune_before, "author tune_only.json unchanged")
    print("REVIEW_PACKAGE_HOST_OK", flush=True)


if __name__ == "__main__":
    main()
