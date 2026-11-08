"""Read-only FP64 review runner. Writes only under experiments/cnn_runtime/fp64_review/.

The packaged harness hardcodes output under fp64_out and temp dirs under
cnn_runtime. This runner executes a path-redirected copy of that harness so
the original harness, packages, checkpoints, and fp64_out stay untouched.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path("/home/alexn/Documents/code/datathon_2026/house-edge")
RUNTIME = ROOT / "experiments/cnn_runtime"
REVIEW = RUNTIME / "fp64_review"
PY = ROOT / ".venv/bin/python"
HARNESS = RUNTIME / "fp64_harness.py"
PACKAGES = (RUNTIME / "fp64_anchor_entry", RUNTIME / "fp64_cnn03_entry")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict:
    out = {}
    for path in paths:
        key = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
        if path.is_file():
            out[key] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        else:
            out[key] = {"missing": True}
    return out


PACKAGE_PATHS = []
for entry in PACKAGES:
    for rel in (
        "predict.py", "model.py", "data.py", "README.md", "reproduce.json",
        "requirements.txt", "checkpoint.pt", "config/config.json", "provenance/provenance.json",
    ):
        PACKAGE_PATHS.append(entry / rel)
PACKAGE_PATHS.extend([
    HARNESS,
    RUNTIME / "fp64_anchor_entry.zip",
    RUNTIME / "fp64_cnn03_entry.zip",
    RUNTIME / "requirements_py313_cpu211_frozen.txt",
    RUNTIME / "reference/experiments/cnn/source/cnn.py",
    RUNTIME / "reference/experiments/cnn/runs/g_nosignal/checkpoint.pt",
    RUNTIME / "reference/experiments/cnn/runs/g_nosignal/config.json",
    ROOT / "experiments/mlp_cnn_hpo/server_snapshot/source/base_cnn.py",
    ROOT / "experiments/mlp_cnn_hpo/server_snapshot/runs/final_cnn03_s0/checkpoint.pt",
    ROOT / "experiments/mlp_cnn_hpo/server_snapshot/runs/final_cnn03_s0/config.json",
    ROOT / "experiments/mlp_cnn_hpo/review/PINNED.json",
    RUNTIME / "fp64_out/anchor_test_a.csv",
    RUNTIME / "fp64_out/anchor_test_b.csv",
    RUNTIME / "fp64_out/anchor_val.csv",
    RUNTIME / "fp64_out/cnn03_test_a.csv",
    RUNTIME / "fp64_out/cnn03_test_b.csv",
    RUNTIME / "fp64_out/cnn03_val.csv",
    ROOT / "experiments/mlp_cnn_hpo/review/fp64_tower_s0_a/test_predictions_cpu.csv",
    ROOT / "experiments/mlp_cnn_hpo/review/fp64_tower_s0_a/validation_predictions_cpu.csv",
    ROOT / "experiments/mlp_cnn_hpo/review/fp64_server_s0_a/test_predictions_cpu.csv",
    ROOT / "experiments/mlp_cnn_hpo/review/fp64_server_s0_a/validation_predictions_cpu.csv",
])


def load_harness_copy() -> Path:
    text = HARNESS.read_text(encoding="utf-8")
    text = text.replace(
        "ROOT = Path(__file__).resolve().parents[2]",
        f'ROOT = Path("{ROOT}")',
    )
    text = text.replace(
        'OUT = RUNTIME / "fp64_out"',
        'OUT = RUNTIME / "fp64_review" / "harness_out"',
    )
    text = text.replace(
        'tempfile.TemporaryDirectory(prefix="fp64-slot-", dir=RUNTIME)',
        'tempfile.TemporaryDirectory(prefix="fp64-slot-", dir=RUNTIME / "fp64_review" / "tmpwork")',
    )
    text = text.replace(
        'tempfile.TemporaryDirectory(prefix="fp64-neg-", dir=RUNTIME)',
        'tempfile.TemporaryDirectory(prefix="fp64-neg-", dir=RUNTIME / "fp64_review" / "tmpwork")',
    )
    dest = REVIEW / "harness_redirected.py"
    dest.write_text(text, encoding="utf-8")
    return dest


def zip_report(path: Path) -> dict:
    import zipfile
    info = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        info["names"] = names
        info["csv_members"] = [n for n in names if n.endswith(".csv")]
        info["bad_parts"] = [
            n for n in names
            if any(part in {".git", "venv", ".venv", "__pycache__"} for part in Path(n).parts)
        ]
        raw = archive.read("reproduce.json")
        info["reproduce_bytes"] = raw.decode("utf-8")
        info["reproduce"] = json.loads(raw)
        info["readme_nonempty"] = bool(archive.read("README.md").strip())
        info["requirements_text"] = archive.read("requirements.txt").decode("utf-8")
    sys.path.insert(0, str(ROOT / "kit"))
    from slot_forecasting.submission_contract import inspect_archive
    info["inspect"] = inspect_archive(path.read_bytes())
    info["exact_command"] = info["reproduce"] == {
        "image": "python",
        "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"],
    }
    return info


def threshold_probe() -> dict:
    spec = importlib.util.spec_from_file_location("fp64_harness_probe", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = REVIEW / "threshold_probe"
    probe.mkdir(parents=True, exist_ok=True)
    header = "query_id,p_0,p_1,p_2,p_5,p_20\n"
    base = probe / "base.csv"
    close = probe / "close_1e-9.csv"
    far = probe / "far_2e-8.csv"
    base.write_text(header + "q,0.2,0.2,0.2,0.2,0.2\n", encoding="utf-8")
    close.write_text(header + "q,0.200000001,0.199999999,0.2,0.2,0.2\n", encoding="utf-8")
    far.write_text(header + "q,0.20000002,0.19999998,0.2,0.2,0.2\n", encoding="utf-8")
    result = {"uses_sys_executable": "sys.executable" in HARNESS.read_text(encoding="utf-8")}

    def call(left: Path, right: Path) -> dict:
        try:
            module.require_close(left, right, left.name)
        except SystemExit as exc:
            return {"exit": int(exc.code or 0)}
        return {"exit": 0}

    result["identical"] = call(base, base)
    result["within_1e-9"] = call(base, close)
    result["above_2e-8"] = call(base, far)
    return result


def whole_model() -> dict:
    import numpy as np
    import torch
    reports = {}
    for entry in PACKAGES:
        for name in ("data", "model", "predict"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(entry))
        from data import load_public_split
        from model import CNN, build_query, window
        from predict import load_pinned
        torch_mod, blob, cfg = load_pinned()
        del torch_mod
        kernel = int(cfg.get("kernel", 3))
        net = CNN(cfg["h"], cfg["layers"], kernel, cfg["drop"], cfg["W"], use_signal=False)
        net.load_state_dict(blob["state"])
        net.double()
        net.eval()
        state_dtypes = sorted({str(t.dtype) for t in blob["state"].values()})
        sig = [k for k in blob["state"] if "sig" in k]
        data = load_public_split(ROOT / "kit/public", "test")
        base_q, _eps = build_query(data, cfg["W"])
        saved = [ep.obs for ep in data.episodes]
        for ep in data.episodes:
            extra = np.full((ep.cutoff + 2, 8), -1, dtype=np.int8)
            extra[: ep.obs.shape[0]] = ep.obs
            extra[ep.cutoff:, :] = 3
            ep.obs = extra
        future_q, _eps = build_query(data, cfg["W"])
        for ep, obs in zip(data.episodes, saved):
            ep.obs = obs
        with torch.no_grad():
            base_logits = net(*base_q)
            future_logits = net(*future_q)
        future_max = float((base_logits - future_logits).abs().max())
        episode = data.episodes[0]
        x0 = window(episode.obs, episode.cutoff, cfg["W"])

        def pack(obs):
            x = window(obs, episode.cutoff, cfg["W"])
            return (
                torch.tensor(x[None]),
                torch.tensor([episode.cutoff]),
                torch.tensor(episode.machine_types[None]),
                torch.zeros(1, 8, dtype=torch.int64),
            )

        with torch.no_grad():
            origin = net(*pack(episode.obs))
        per_machine = []
        last = min(episode.obs.shape[0], episode.cutoff) - 1
        for machine in range(8):
            mutated = episode.obs.copy()
            mutated[last, machine] = np.int8(0 if mutated[last, machine] != 0 else 1)
            with torch.no_grad():
                moved = net(*pack(mutated))
            per_machine.append(float((origin - moved).abs().max()))
        reports[entry.name] = {
            "checkpoint_config_no_signal": None if not isinstance(blob.get("config"), dict) else blob["config"].get("no_signal", "ABSENT"),
            "checkpoint_config_drop": blob.get("config", {}).get("drop"),
            "checkpoint_config_keys": sorted(blob.get("config", {})),
            "state_dtypes_before_double": state_dtypes,
            "sig_keys": sig,
            "n_floating_fp64": sum(p.dtype == torch.float64 for p in net.parameters() if p.is_floating_point()),
            "n_floating": sum(p.is_floating_point() for p in net.parameters()),
            "future_all_episodes_all_machines_logit_max_abs": future_max,
            "per_machine_last_tick_logit_max_abs": per_machine,
            "window_unchanged_shape": list(x0.shape),
            "signal_feature_sum": int(base_q[3].sum()),
            "logit_dtype": str(base_logits.dtype),
        }
        # drop imported modules so the next package wins
        for name in ("data", "model", "predict"):
            sys.modules.pop(name, None)
        sys.path.pop(0)
    return reports


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "tmpwork").mkdir(parents=True, exist_ok=True)
    before = snapshot(PACKAGE_PATHS)
    (REVIEW / "hashes_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    import torch
    import numpy as np
    versions = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "cpu_capability": getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unavailable")(),
    }
    try:
        versions["onednn_enabled"] = bool(torch.backends.mkldnn.is_available())
    except Exception as exc:  # noqa: BLE001
        versions["onednn_enabled_error"] = str(exc)
    (REVIEW / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print("VERSIONS", json.dumps(versions))

    copy = load_harness_copy()
    print("HARNESS_COPY", copy, "sha", sha256_file(copy), "original", sha256_file(HARNESS))
    print("THRESHOLD", json.dumps(threshold_probe()))
    try:
        print("WHOLE_MODEL", json.dumps(whole_model()))
    except Exception:
        traceback.print_exc()
        (REVIEW / "whole_model_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    for zip_path in (RUNTIME / "fp64_anchor_entry.zip", RUNTIME / "fp64_cnn03_entry.zip"):
        report = zip_report(zip_path)
        (REVIEW / f"zip_{zip_path.stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("ZIP", zip_path.name, report["sha256"], report["bytes"], "exact", report["exact_command"], "csv", report["csv_members"])

    py313 = RUNTIME / "py313_verify_1936/env/bin/python"
    if py313.is_file():
        proc = subprocess.run(
            [str(py313), "-c", "import torch,sys; print(sys.version); print(torch.__version__)"],
            text=True, capture_output=True, timeout=60, check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        (REVIEW / "py313_import.log").write_text(
            f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
            encoding="utf-8",
        )
        print("PY313_IMPORT", proc.returncode)
    else:
        print("PY313_IMPORT missing")

    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    })
    proc = subprocess.run(
        [sys.executable, str(copy)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    (REVIEW / "harness.log").write_text(
        f"exit={proc.returncode}\ncmd={sys.executable} {copy}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
        encoding="utf-8",
    )
    print("HARNESS_EXIT", proc.returncode)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr[-4000:])
    after = snapshot(PACKAGE_PATHS)
    (REVIEW / "hashes_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    changed = [key for key in before if before[key] != after.get(key)]
    (REVIEW / "hash_diff.json").write_text(json.dumps({"changed": changed}, indent=2), encoding="utf-8")
    print("PACKAGE_HASH_CHANGES", changed)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
