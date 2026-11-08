"""Tower-local FP32 CPU stability diagnostic for the hash-locked anchor.

Does not modify entry/, the checkpoint, or the default predict path.
Each control is a fresh process. This file never assigns ISA variables into
the parent environment beyond that child. Tolerance is max absolute difference
after row normalization, not relative error.

Reproducible command, from the repo root:

  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/stability/diagnose.py --run-all --data kit/public --out-dir experiments/cnn_runtime/stability/out

One control:

  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/stability/diagnose.py --control aten_avx2 --data kit/public --out experiments/cnn_runtime/stability/out/aten_avx2/submission.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
ENTRY = RUNTIME / "entry"
REF = RUNTIME / "reference" / "experiments" / "cnn" / "runs" / "g_nosignal" / "test_cpu_a.csv"
REF_SHA = "9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9"
ANCHOR = "43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80"
HEADER = ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"]
TOL = 1e-8
ISA_KEYS = ("ATEN_CPU_CAPABILITY", "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "MKL_CBWR")
CONTROLS = ("baseline", "aten_avx2", "onednn_avx2", "mkldnn_off")
ENTRY_FILES = ("predict.py", "model.py", "data.py", "reproduce.json", "requirements.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor()
    return platform.processor()


def apply_control(name: str) -> dict[str, str]:
    """Set this process only. Call before importing torch."""
    if name not in CONTROLS:
        raise SystemExit(f"unknown control {name}")
    for key in ISA_KEYS:
        os.environ.pop(key, None)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    if name == "aten_avx2":
        os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    elif name == "onednn_avx2":
        os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
        os.environ["MKL_CBWR"] = "AVX2"
    set_keys = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    }
    for key in ISA_KEYS:
        if key in os.environ:
            set_keys[key] = os.environ[key]
    return set_keys


def load_body(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != HEADER:
        raise SystemExit(f"bad header {path}")
    body = rows[1:]
    if len(body) != 1536:
        raise SystemExit(f"expected 1536 rows in {path}, got {len(body)}")
    return body


def pair_metrics(left: Path, right: Path) -> dict:
    a = load_body(left)
    b = load_body(right)
    if [row[0] for row in a] != [row[0] for row in b]:
        raise SystemExit(f"id mismatch {left} vs {right}")
    max_abs = 0.0
    n_differ = 0
    nonfinite = 0
    simplex = 0.0
    for ra, rb in zip(a, b):
        av = [float(x) for x in ra[1:]]
        bv = [float(x) for x in rb[1:]]
        if any(not math.isfinite(v) for v in av + bv):
            nonfinite += 1
        simplex = max(simplex, abs(sum(av) - 1.0), abs(sum(bv) - 1.0))
        for x, y in zip(av, bv):
            diff = abs(x - y)
            if diff != 0.0:
                n_differ += 1
            if diff > max_abs:
                max_abs = diff
    return {
        "bitwise": left.read_bytes() == right.read_bytes(),
        "max_abs_after_row_normalization": max_abs,
        "cells_not_bitwise": n_differ,
        "nonfinite_rows": nonfinite,
        "max_simplex_abs_err": simplex,
        "within_1e-8": max_abs <= TOL,
        "metric": "max_abs_after_row_normalization",
        "relative_error_used_as_tolerance": False,
    }


def entry_hashes() -> dict[str, str]:
    hashes = {name: sha256_file(ENTRY / name) for name in ENTRY_FILES}
    hashes["checkpoint.pt"] = sha256_file(ENTRY / "checkpoint.pt")
    hashes["config/config.json"] = sha256_file(ENTRY / "config" / "config.json")
    return hashes


def run_control(name: str, data: str, out: Path) -> None:
    set_keys = apply_control(name)
    import torch
    torch.set_num_threads(2)
    mkldnn_requested_off = name == "mkldnn_off"
    if mkldnn_requested_off:
        torch.backends.mkldnn.enabled = False
    sys.path.insert(0, str(ENTRY))
    import predict
    out.parent.mkdir(parents=True, exist_ok=True)
    predict.main(["--data", data, "--out", str(out)])
    capability = None
    getter = getattr(torch.backends.cpu, "get_cpu_capability", None)
    if getter is not None:
        capability = getter()
    manifest = {
        "control": name,
        "precision": "fp32",
        "precision_change": False,
        "process_env_set": set_keys,
        "parent_environ_not_mutated_by_design": True,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": __import__("numpy").__version__,
        "cpu_model": cpu_model(),
        "cpu_capability": capability,
        "mkldnn_enabled_after_control": bool(torch.backends.mkldnn.enabled),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "num_threads": int(torch.get_num_threads()),
        "checkpoint_sha256": sha256_file(ENTRY / "checkpoint.pt"),
        "entry_sha256": entry_hashes(),
        "csv": str(out),
        "csv_sha256": sha256_file(out),
        "csv_bytes": out.stat().st_size,
        "n_rows": 1536,
        "reference_csv_sha256": REF_SHA,
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--control",
            name,
            "--data",
            data,
            "--out",
            str(out),
        ],
    }
    if manifest["checkpoint_sha256"] != ANCHOR:
        raise SystemExit("anchor checkpoint hash changed")
    if mkldnn_requested_off and manifest["mkldnn_enabled_after_control"]:
        manifest["control_took_effect"] = False
    else:
        manifest["control_took_effect"] = True
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"control": name, "csv_sha256": manifest["csv_sha256"], "cpu_capability": capability, "mkldnn": manifest["mkldnn_enabled_after_control"]}))


def child_env(name: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in ISA_KEYS:
        env.pop(key, None)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if name == "aten_avx2":
        env["ATEN_CPU_CAPABILITY"] = "avx2"
    elif name == "onednn_avx2":
        env["ONEDNN_MAX_CPU_ISA"] = "AVX2"
        env["DNNL_MAX_CPU_ISA"] = "AVX2"
        env["MKL_CBWR"] = "AVX2"
    return env


def run_all(data: str, out_dir: Path) -> None:
    if sha256_file(ENTRY / "checkpoint.pt") != ANCHOR:
        raise SystemExit("refusing to run: entry checkpoint is not the anchor")
    if sha256_file(REF) != REF_SHA:
        raise SystemExit("refusing to run: reference CPU csv hash mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in CONTROLS:
        out = out_dir / name / "submission.csv"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--control", name, "--data", data, "--out", str(out)],
            cwd=Path.cwd(),
            env=child_env(name),
            text=True,
        )
        if proc.returncode != 0 or not out.is_file():
            failures.append({"control": name, "returncode": proc.returncode})
    pairs = []
    produced = [name for name in CONTROLS if (out_dir / name / "submission.csv").is_file()]
    targets = [("reference", REF)] + [(name, out_dir / name / "submission.csv") for name in produced]
    first = RUNTIME / "first.csv"
    if first.is_file():
        targets.append(("tower_first_csv", first))
    for i, (left_name, left_path) in enumerate(targets):
        for right_name, right_path in targets[i + 1:]:
            metrics = pair_metrics(left_path, right_path)
            metrics["left"] = left_name
            metrics["right"] = right_name
            pairs.append(metrics)
    summary = {
        "host_role": "tower_only",
        "precision": "fp32",
        "precision_change": False,
        "tolerance_max_abs_after_row_normalization": TOL,
        "relative_error_used_as_tolerance": False,
        "checkpoint_sha256": ANCHOR,
        "reference_csv_sha256": REF_SHA,
        "entry_sha256": entry_hashes(),
        "cpu_model": cpu_model(),
        "controls": list(CONTROLS),
        "failures": failures,
        "pairs": pairs,
        "fp32_all_compared_pairs_within_1e-8": all(item["within_1e-8"] for item in pairs) if pairs else False,
        "not_cross_host_proof": True,
        "entry_default_unchanged": True,
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-all",
            "--data",
            data,
            "--out-dir",
            str(out_dir),
        ],
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out_dir / "comparison.json"),
        "failures": failures,
        "fp32_all_compared_pairs_within_1e-8": summary["fp32_all_compared_pairs_within_1e-8"],
        "pairs": [
            {
                "left": item["left"],
                "right": item["right"],
                "bitwise": item["bitwise"],
                "max_abs_after_row_normalization": item["max_abs_after_row_normalization"],
                "within_1e-8": item["within_1e-8"],
            }
            for item in pairs
        ],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="FP32 CPU stability controls for the anchor CNN")
    parser.add_argument("--control", choices=CONTROLS)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--data", default="kit/public")
    parser.add_argument("--out")
    parser.add_argument("--out-dir", default=str(HERE / "out"))
    args = parser.parse_args()
    if args.run_all:
        run_all(args.data, Path(args.out_dir))
        return
    if not args.control or not args.out:
        raise SystemExit("pass --run-all or both --control and --out")
    run_control(args.control, args.data, Path(args.out))


if __name__ == "__main__":
    main()
