"""Independent package-quality probe for the frozen curve_C_lr003_s2 ZIP.

Writes only under independent_review_20261108/candidate_review/.
Does not call fp64_candidate_harness.main() and does not call
score_tune_only.main(), because both write inside the candidate folder.
The harness module is imported and its WORK/OUT globals are rebound in this
process to this review directory. The production guard in main() is not
edited and is not invoked.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import resource
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[3]
CANDIDATE = REPO / "experiments/cnn03_convergence/candidate_lr003_s2_20261108"
ZIP_PATH = CANDIDATE / "curve_C_lr003_s2_fp64_candidate.zip"
EXTRACT = REVIEW / "extract"
WORK = REVIEW / "_work"
HARNESS_OUT = REVIEW / "harness_out"
EXPECTED_ZIP = "dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29"
EXPECTED_PARENT = "bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71"
EXPECTED_TEST = "9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6"
EXPECTED_TUNE = 0.30980333573689856
EXPECTED_MEMBERS = [
    "README.md",
    "checkpoint.pt",
    "config/config.json",
    "data.py",
    "model.py",
    "predict.py",
    "provenance/provenance.json",
    "reproduce.json",
    "requirements.txt",
]
NOT_PACKAGED = {
    "curve_C_lr003_s2_plus10": "7f2679b120217bc49c6628de40720c66294d38d651d61885397358e802ced205",
    "curve_C_lr003_s2_plus20": "887bdd3ba17083cd95753bae8e0981f61daf026c5a10b14c7fef0abb0dce84d6",
}
PROTECTED = {
    "experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip": "61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599",
    "experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv": "08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d",
    "experiments/cnn_runtime/first_submission_full2029/cnn03_test_a.csv": "08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d",
    "experiments/mlp_cnn_hpo/review/PINNED.json": "cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb",
    "experiments/cnn_runtime/fp64_harness.py": "515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2",
    "experiments/cnn_runtime/fp64_release_harness2025.py": "a077e8771abdab58e3d107eeb1da3ecc3ee80ccc2ee08ce72e49b7ab36d89f2c",
    "experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt": EXPECTED_PARENT,
    "experiments/cnn03_convergence/continuation_20261108/PLAN.md": "086187accf10068b38770a1db98b2583150fa7bb84ac96e46426d4847febfff6",
    "experiments/cnn03_convergence/continuation_20261108/MANIFEST.json": "c71474bcbb6caede6bb629d7db163469788054da2d48dcd4a56c1bc482edcf8b",
}
CANONICAL_ZIP_COMPARE = ("model.py", "data.py", "requirements.txt", "reproduce.json")
REPORT = {}
FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"FAIL {message}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso(path: Path) -> str | None:
    if not path.exists():
        return None
    stamp = path.stat().st_mtime
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def call_names(fn_node: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute):
            names.append(func.attr)
    return names


def function_map(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def check_source_hashes() -> None:
    text = (CANDIDATE / "SOURCE_HASHES.txt").read_text(encoding="utf-8")
    measured = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        if rel.startswith("unpacked/") or rel.startswith("refs/") or "/" not in rel:
            path = CANDIDATE / rel
        else:
            path = REPO / rel
        if not path.is_file():
            fail(f"SOURCE_HASHES missing {rel}")
            continue
        got = sha256_file(path)
        measured[rel] = got
        if got != digest:
            fail(f"SOURCE_HASHES mismatch {rel} file={got} listed={digest}")
    own = sha256_file(CANDIDATE / "SOURCE_HASHES.txt")
    REPORT["source_hashes_file_sha256"] = own
    REPORT["source_hashes_checked"] = len(measured)
    if FAILURES:
        return
    print(f"SOURCE_HASHES file_bytes_match n={len(measured)} self={own}", flush=True)


def check_protected() -> None:
    found = {}
    for rel, expect in PROTECTED.items():
        path = REPO / rel
        if not path.is_file():
            fail(f"protected missing {rel}")
            continue
        got = sha256_file(path)
        found[rel] = got
        if got != expect:
            fail(f"protected drift {rel} {got}")
    pin = (REPO / "experiments/mlp_cnn_hpo/review/PINNED.json").read_text(encoding="utf-8")
    if "candidate_lr003" in pin or EXPECTED_PARENT in pin or "curve_C_lr003_s2" in pin:
        fail("PINNED.json names this candidate")
    if "ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f" not in pin:
        fail("PINNED.json lost the canonical CNN03 seed-0 checkpoint pin")
    harness = (REPO / "experiments/cnn_runtime/fp64_harness.py").read_text(encoding="utf-8")
    if "candidate_lr003" in harness or EXPECTED_PARENT in harness:
        fail("production fp64_harness.py names this candidate")
    REPORT["protected"] = found
    print("PROTECTED hashed", flush=True)


def check_zip_and_unpack() -> None:
    if sha256_file(ZIP_PATH) != EXPECTED_ZIP:
        fail(f"ZIP sha {sha256_file(ZIP_PATH)}")
        return
    if ZIP_PATH.stat().st_size != 1378957:
        fail(f"ZIP size {ZIP_PATH.stat().st_size}")
    forbidden_suffix = (".csv", ".whl", ".pyc", ".npy", ".npz", ".parquet")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = [info.filename for info in archive.infolist()]
        bad = []
        payloads = {}
        for info in archive.infolist():
            name = info.filename
            parts = Path(name).parts
            if info.is_dir() or name.endswith("/"):
                bad.append(f"directory {name}")
                continue
            if name.startswith("/") or ".." in parts or "\\" in name:
                bad.append(f"traversal {name}")
                continue
            lowered = name.lower()
            if lowered.endswith(forbidden_suffix) or "/venv/" in lowered or name.startswith("venv"):
                bad.append(f"forbidden member {name}")
            data = archive.read(info)
            payloads[name] = data
            member_sha = sha256_bytes(data)
            if member_sha in NOT_PACKAGED.values():
                bad.append(f"continuation checkpoint packaged as {name}")
        if names != EXPECTED_MEMBERS:
            bad.append(f"member order {names}")
        if bad:
            for item in bad:
                fail(item)
            return
        if EXTRACT.exists():
            for child in sorted(EXTRACT.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
        EXTRACT.mkdir(parents=True, exist_ok=True)
        root = EXTRACT.resolve()
        for name, data in payloads.items():
            dest = (EXTRACT / name).resolve()
            if dest != root and root not in dest.parents:
                fail(f"extract escaped {name}")
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        REPORT["zip_members"] = names
        REPORT["zip_member_sha256"] = {name: sha256_bytes(data) for name, data in payloads.items()}
    original = REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt"
    packed = EXTRACT / "checkpoint.pt"
    if packed.read_bytes() != original.read_bytes():
        fail("extract checkpoint bytes differ from the original parent")
    if sha256_file(packed) != EXPECTED_PARENT:
        fail("extract checkpoint sha is not the original parent")
    print(f"UNPACKED members={len(EXPECTED_MEMBERS)} checkpoint={EXPECTED_PARENT}", flush=True)
    compare_canonical_zip()


def compare_canonical_zip() -> None:
    canonical = REPO / "experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip"
    same = {}
    with zipfile.ZipFile(canonical) as archive:
        available = set(archive.namelist())
        for name in CANONICAL_ZIP_COMPARE:
            if name not in available:
                fail(f"canonical ZIP missing {name}")
                continue
            canonical_sha = sha256_bytes(archive.read(name))
            packed_sha = sha256_file(EXTRACT / name)
            same[name] = {"canonical": canonical_sha, "packed": packed_sha, "equal": canonical_sha == packed_sha}
            if canonical_sha != packed_sha:
                fail(f"canonical bytes differ for {name}")
        predict_sha = sha256_bytes(archive.read("predict.py")) if "predict.py" in available else None
        checkpoint_sha = sha256_bytes(archive.read("checkpoint.pt")) if "checkpoint.pt" in available else None
    packed_predict = sha256_file(EXTRACT / "predict.py")
    if predict_sha == packed_predict:
        fail("candidate predict.py is byte-identical to the canonical seed-0 predict.py")
    if checkpoint_sha == EXPECTED_PARENT:
        fail("canonical ZIP checkpoint unexpectedly equals this parent")
    REPORT["canonical_zip_compare"] = same
    REPORT["predict_differs_from_canonical_zip"] = predict_sha != packed_predict
    print("CANONICAL_ZIP model/data/lock/reproduce compared", flush=True)


def check_checkpoint_contract() -> None:
    import torch

    torch.set_num_threads(2)
    blob = torch.load(EXTRACT / "checkpoint.pt", map_location="cpu", weights_only=False)
    if set(blob) != {"config", "state"}:
        fail(f"checkpoint keys {sorted(blob)}")
        return
    cfg = blob["config"]
    expect = {
        "lr": 0.003,
        "wd": 0.00895,
        "bs": 512,
        "epochs": 30,
        "drop": 0.3,
        "h": 128,
        "layers": 4,
        "W": 64,
        "seed": 2,
        "pct_start": 0.3,
        "eval_every": 2,
        "out": "experiments/cnn03_convergence/runs/curve_C_lr003_s2",
    }
    if cfg != expect or "kernel" in cfg:
        fail(f"checkpoint config {cfg}")
    packaged = json.loads((EXTRACT / "config/config.json").read_text(encoding="utf-8"))
    if packaged != cfg:
        fail("packaged config.json differs from checkpoint config")
    state = blob["state"]
    if len(state) != 43:
        fail(f"tensor count {len(state)}")
    if any(str(key).startswith("sig.") or key == "sig.weight" for key in state):
        fail("signal tensor present")
    shapes = {}
    for key, tensor in state.items():
        shapes[key] = [tuple(tensor.shape), str(tensor.dtype), bool(torch.isfinite(tensor).all())]
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            fail(f"{key} dtype/finite {tensor.dtype}")
    for index in range(4):
        shape = tuple(state[f"blocks.{index}.conv.weight"].shape)
        if shape != (128, 128, 3):
            fail(f"conv {index} shape {shape}")
    provenance = json.loads((EXTRACT / "provenance/provenance.json").read_text(encoding="utf-8"))
    REPORT["checkpoint"] = {
        "weights_only": True,
        "tensors": len(state),
        "seed": cfg["seed"],
        "lr": cfg["lr"],
        "eval_every": cfg["eval_every"],
        "kernel_field": "kernel" in cfg,
        "conv_shape": [128, 128, 3],
        "provenance_keys": sorted(provenance),
    }
    text = json.dumps(provenance)
    if "bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71" not in text:
        fail("provenance does not name the parent sha")
    if any(sha in text for sha in NOT_PACKAGED.values()):
        fail("provenance names a continuation checkpoint")
    print("CHECKPOINT contract ok", flush=True)


def check_harness_scope() -> None:
    candidate = CANDIDATE / "fp64_candidate_harness.py"
    release = REPO / "experiments/cnn_runtime/fp64_release_harness2025.py"
    import subprocess

    proc = subprocess.run(
        ["diff", "-u", str(release.relative_to(REPO)), str(candidate.relative_to(REPO))],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    recorded = (CANDIDATE / "HARNESS_DIFF.txt").read_text(encoding="utf-8")
    REPORT["harness_diff_exit"] = proc.returncode
    REPORT["harness_diff_matches_recorded_file"] = proc.stdout == recorded
    stale = REVIEW / "harness_diff_regen.txt"
    if proc.stdout == recorded and stale.exists():
        stale.unlink()
    if proc.returncode not in (0, 1):
        fail(f"diff failed {proc.returncode} {proc.stderr}")
    if proc.stdout != recorded:
        fail("HARNESS_DIFF.txt is not the exact diff -u of the current files")
        (REVIEW / "harness_diff_regen.txt").write_text(proc.stdout, encoding="utf-8")
    cand_fns = function_map(candidate)
    rel_fns = function_map(release)
    cand_full = call_names(cand_fns["full_runs"])
    rel_full = call_names(rel_fns["full_runs"])
    cand_main = call_names(cand_fns["main"])
    REPORT["candidate_full_runs_calls"] = cand_full
    REPORT["release_full_runs_calls"] = rel_full
    REPORT["candidate_main_calls"] = cand_main
    if "official_r" in cand_full or "official_r" in cand_main:
        fail("candidate full_runs or main calls official_r")
    if "structural_coverage" not in cand_full:
        fail("candidate full_runs does not schema-check validation")
    if "official_check" not in rel_full or "official_r" not in rel_full:
        fail("release harness no longer has the validation scoring calls this diff claims to remove")
    source = candidate.read_text(encoding="utf-8")
    if "TOL = 1e-8" not in source:
        fail("candidate tolerance changed")
    if "candidate_lr003_s2_20261108" not in source:
        fail("candidate output guard missing")
    # The unused official_r function remains defined. It must not be invoked.
    if cand_fns["official_r"] is None:
        fail("official_r missing")
    print(
        f"HARNESS_DIFF exact={proc.stdout == recorded} full_runs_calls={cand_full}",
        flush=True,
    )


def check_accident_footprint() -> None:
    needles = ("mean_loss_ratio=", "technical_points=", "VALIDATION_R ")
    hits = []
    for path in CANDIDATE.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".txt", ".log", ".json", ".err"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        for needle in needles:
            if needle in text:
                hits.append({"file": str(path.relative_to(CANDIDATE)), "needle": needle})
    mtimes = {
        "original_checkpoint": iso(REPO / "experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt"),
        "corrected_tune_json": iso(
            REPO / "experiments/cnn03_convergence/continuation_20261108/corrected_results/fp64_original_tune.json"
        ),
        "candidate_zip": iso(ZIP_PATH),
        "candidate_predict_entry": iso(CANDIDATE / "entry/predict.py"),
        "candidate_harness": iso(CANDIDATE / "fp64_candidate_harness.py"),
        "candidate_quality": iso(CANDIDATE / "QUALITY.md"),
        "candidate_source_hashes": iso(CANDIDATE / "SOURCE_HASHES.txt"),
    }
    REPORT["validation_metric_string_hits"] = hits
    REPORT["mtimes_utc"] = mtimes
    print(f"ACCIDENT_STRING_HITS {len(hits)}", flush=True)


def check_docker_records() -> None:
    records = {}
    for name in ("cand-lr003-s2-a", "cand-lr003-s2-b", "cand-lr003-s2-val"):
        path = CANDIDATE / "docker_out" / f"{name}.inspect.json"
        data = json.loads(path.read_text(encoding="utf-8"))[0]
        host = data["HostConfig"]
        mounts = data.get("Mounts", [])
        records[name] = {
            "exit": data["State"]["ExitCode"],
            "oom": data["State"]["OOMKilled"],
            "image": data["Image"],
            "network": host["NetworkMode"],
            "readonly": host["ReadonlyRootfs"],
            "cap_drop": host["CapDrop"],
            "privileged": host["Privileged"],
            "security": host["SecurityOpt"],
            "memory": host["Memory"],
            "memory_swap": host["MemorySwap"],
            "nano_cpus": host["NanoCpus"],
            "user": data["Config"]["User"],
            "tmpfs": host["Tmpfs"],
            "mount_modes": [item.get("Mode") for item in mounts],
            "started": data["State"]["StartedAt"],
            "finished": data["State"]["FinishedAt"],
        }
        err = (CANDIDATE / "docker_out" / f"{name}.err").read_text(encoding="utf-8").strip()
        records[name]["stderr"] = err
        if data["State"]["ExitCode"] != 0 or host["NetworkMode"] != "none" or host["ReadonlyRootfs"] is not True:
            fail(f"docker record {name} is not a read-only network-none exit 0")
        if host["CapDrop"] != ["ALL"] or "no-new-privileges" not in host["SecurityOpt"]:
            fail(f"docker record {name} capabilities")
        if any(item.get("RW") for item in mounts):
            fail(f"docker record {name} has a writable bind")
    csvs = {}
    for name in ("test_a.csv", "test_b.csv", "val.csv"):
        path = CANDIDATE / "docker_out" / name
        csvs[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    ref_test = sha256_file(CANDIDATE / "refs/candidate_test_a.csv")
    ref_val = sha256_file(CANDIDATE / "refs/candidate_val.csv")
    if csvs["test_a.csv"]["sha256"] != EXPECTED_TEST or csvs["test_b.csv"]["sha256"] != EXPECTED_TEST:
        fail("docker test CSV bytes differ from the pinned test SHA")
    if ref_test != EXPECTED_TEST:
        fail("refs/candidate_test_a.csv sha drifted")
    if csvs["val.csv"]["sha256"] != ref_val:
        fail("docker val CSV differs from refs/candidate_val.csv")
    REPORT["docker_records"] = records
    REPORT["docker_csv"] = csvs
    print("DOCKER_RECORDS read", flush=True)


def schema_csv(path: Path, expect: int) -> None:
    sys.path.insert(0, str(REPO / "kit"))
    from slot_forecasting.io import read_submission
    from slot_forecasting.schema import probabilities

    rows = read_submission(path)
    if len(rows) != expect:
        fail(f"{path.name} ids {len(rows)}")
    for query_id, values in rows.items():
        probabilities(values)
    print(f"SCHEMA {path.name} ids={len(rows)}", flush=True)


def run_harness_functions() -> None:
    spec = importlib.util.spec_from_file_location(
        "candidate_harness_readonly",
        CANDIDATE / "fp64_candidate_harness.py",
    )
    if spec is None or spec.loader is None:
        fail("could not load candidate harness")
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORK = WORK
    module.OUT = HARNESS_OUT
    module.REVIEW_TEST = CANDIDATE / "refs/candidate_test_a.csv"
    module.REVIEW_VAL = CANDIDATE / "refs/candidate_val.csv"
    WORK.mkdir(parents=True, exist_ok=True)
    HARNESS_OUT.mkdir(parents=True, exist_ok=True)
    steps = [
        ("source_patch", lambda: module.test_source_has_no_global_patch(EXTRACT)),
        ("window_missing", lambda: module.test_window_and_missing(EXTRACT)),
        ("dtype_causality", lambda: module.test_dtype_and_forecast_causality(EXTRACT)),
        ("query_order", lambda: module.test_query_order(EXTRACT)),
        ("refusals", lambda: module.test_refusals(EXTRACT)),
        ("full_runs", lambda: module.full_runs("candidate", EXTRACT)),
        ("compare_test", lambda: module.require_close(HARNESS_OUT / "candidate_test_a.csv", module.REVIEW_TEST, "candidate_vs_own_test")),
        ("compare_val", lambda: module.require_close(HARNESS_OUT / "candidate_val.csv", module.REVIEW_VAL, "candidate_vs_own_val")),
    ]
    codes = {}
    for name, fn in steps:
        print(f"BEGIN {name}", flush=True)
        try:
            fn()
            codes[name] = 0
        except SystemExit as exc:
            codes[name] = int(exc.code) if exc.code else 1
            fail(f"harness step {name} exit {codes[name]}")
            break
        except Exception:
            codes[name] = 1
            fail(f"harness step {name} exception")
            traceback.print_exc()
            break
        print(f"END {name} exit=0", flush=True)
    REPORT["harness_step_exits"] = codes
    produced = HARNESS_OUT / "candidate_test_a.csv"
    if produced.is_file():
        digest = sha256_file(produced)
        REPORT["fresh_test_sha256"] = digest
        REPORT["fresh_test_bytes"] = produced.stat().st_size
        if digest != EXPECTED_TEST:
            fail(f"fresh test CSV sha {digest}")
        else:
            schema_csv(produced, 1536)
        repeat = HARNESS_OUT / "candidate_test_b.csv"
        if repeat.is_file() and produced.read_bytes() != repeat.read_bytes():
            fail("cold repeat CSV bytes differ")
    val = HARNESS_OUT / "candidate_val.csv"
    if val.is_file():
        schema_csv(val, 768)
        # structural coverage already ran inside full_runs; repeat the id check only
        module.structural_coverage(val, "validation", 768)


def tune_only() -> None:
    import importlib
    import torch

    torch.set_num_threads(2)
    for name in ("data", "model", "predict"):
        sys.modules.pop(name, None)
    study_source = REPO / "experiments/cnn03_convergence/continuation_20261108/source"
    sys.path.insert(0, str(EXTRACT))
    sys.path.insert(0, str(study_source))
    sys.path.insert(0, str(REPO / "src"))
    from study_common import canonical_renorm, score_tune_detail
    from house_edge.data import load_split

    spec = importlib.util.spec_from_file_location("candidate_extract_model", EXTRACT / "model.py")
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    if model_mod.USE_SIGNAL:
        fail("USE_SIGNAL")
        return
    blob = torch.load(EXTRACT / "checkpoint.pt", map_location="cpu", weights_only=False)

    def predict_with(validation):
        model_mod.USE_SIGNAL = False
        cfg = blob["config"]
        net = model_mod.CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
        net.load_state_dict(blob["state"])
        net.double()
        net.eval()
        query, episodes = model_mod.build_query(validation, cfg["W"])
        probs = model_mod.predict(net, query, "cpu")
        if str(probs.dtype) != "float64":
            fail(f"predict dtype {probs.dtype}")
        return model_mod.to_preds(probs, episodes, validation)

    study_preds = predict_with(load_split(REPO / "kit/public/validation"))
    sys.path.insert(0, str(EXTRACT))
    from data import load_public_split

    package_preds = predict_with(load_public_split(REPO / "kit/public", "validation"))
    study_detail = score_tune_detail(study_preds)
    package_detail = score_tune_detail(package_preds)
    stored = json.loads(
        (REPO / "experiments/cnn03_convergence/continuation_20261108/corrected_results/fp64_original_tune.json").read_text(
            encoding="utf-8"
        )
    )["curve_C_lr003_s2"]["tune_R"]
    raw_worst = 0.0
    renorm_worst = 0.0
    if set(study_preds) != set(package_preds):
        fail("loader query ids differ")
    else:
        for query_id in study_preds:
            left = study_preds[query_id]
            right = package_preds[query_id]
            for a, b in zip(left, right):
                raw_worst = max(raw_worst, abs(float(a) - float(b)))
            for a, b in zip(canonical_renorm(left), canonical_renorm(right)):
                renorm_worst = max(renorm_worst, abs(a - b))
    tune = {
        "study_tune_R": study_detail["tune_R"],
        "package_tune_R": package_detail["tune_R"],
        "stored_tune_R": stored,
        "expected_tune_R": EXPECTED_TUNE,
        "n_tune_queries": study_detail["n_tune_queries"],
        "n_tune_episodes": study_detail["n_tune_episodes"],
        "n_forward_queries": study_detail["n_forward_queries"],
        "package_n_tune_queries": package_detail["n_tune_queries"],
        "package_n_forward_queries": package_detail["n_forward_queries"],
        "loader_max_abs_raw": raw_worst,
        "loader_max_abs_after_one_renorm": renorm_worst,
        "confirm_full_test_metrics": "not_computed",
    }
    if study_detail["tune_R"] != EXPECTED_TUNE or package_detail["tune_R"] != EXPECTED_TUNE or stored != EXPECTED_TUNE:
        fail(
            f"tune R study={study_detail['tune_R']!r} package={package_detail['tune_R']!r} stored={stored!r}"
        )
    if study_detail["n_tune_queries"] != 384 or study_detail["n_forward_queries"] != 768:
        fail("tune coverage is not 384/768")
    if raw_worst != 0.0 or renorm_worst != 0.0:
        fail(f"loader mismatch raw={raw_worst} renorm={renorm_worst}")
    REPORT["tune"] = tune
    (REVIEW / "tune_only_reviewer.json").write_text(json.dumps(tune, indent=1) + "\n", encoding="utf-8")
    print(
        f"TUNE_ONLY R={study_detail['tune_R']:.17g} n={study_detail['n_tune_queries']} "
        f"forward={study_detail['n_forward_queries']} loader_max_abs={renorm_worst}",
        flush=True,
    )


def main() -> None:
    if REVIEW.name != "candidate_review" or REVIEW.parent.name != "independent_review_20261108":
        raise SystemExit("refusing to write outside the reviewer directory")
    print(f"python {sys.version.split()[0]} executable={sys.executable}", flush=True)
    import numpy
    import torch

    print(f"torch {torch.__version__} numpy {numpy.__version__} threads={torch.get_num_threads()}", flush=True)
    REPORT["python"] = sys.version.split()[0]
    REPORT["torch"] = torch.__version__
    REPORT["numpy"] = numpy.__version__
    REPORT["executable"] = sys.executable
    check_source_hashes()
    check_protected()
    check_zip_and_unpack()
    check_checkpoint_contract()
    check_harness_scope()
    check_accident_footprint()
    check_docker_records()
    if not any(item.startswith("ZIP") or item.startswith("extract") or "member" in item for item in FAILURES):
        run_harness_functions()
        tune_only()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    REPORT["probe_max_rss_kb"] = rss
    REPORT["failures"] = FAILURES
    (REVIEW / "probe_report.json").write_text(json.dumps(REPORT, indent=1) + "\n", encoding="utf-8")
    print(f"PROBE_RSS_KB {rss} failures={len(FAILURES)}", flush=True)
    if FAILURES:
        raise SystemExit(1)
    print("PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
