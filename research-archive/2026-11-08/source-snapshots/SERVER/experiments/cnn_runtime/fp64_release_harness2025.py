"""CNN03 float64 release checks. Exits 1 on a failed check.

Uses this process's interpreter (sys.executable). Numeric compares use max
absolute difference after kit normalization and fail when that exceeds 1e-8
or when a reference file is missing.

Path config only: --entry, --out, --reference-test, --reference-val.
Default references are this runtime's own CNN03 CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "experiments" / "cnn_runtime"
CNN03 = RUNTIME / "fp64_cnn03_entry"
PUBLIC = ROOT / "kit" / "public"
OUT = RUNTIME / "fp64_release_out2025"
TOL = 1e-8
REVIEW_TEST = RUNTIME / "cnn03_release_refs2024" / "tower_cnn03_a.csv"
REVIEW_VAL = RUNTIME / "cnn03_release_refs2024" / "tower_cnn03_val.csv"


def fail(message: str) -> None:
    print(f"FAIL {message}")
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_entry(entry: Path, data: Path, out: Path, split: str = "test") -> subprocess.CompletedProcess:
    if out.exists():
        out.unlink()
    return subprocess.run(
        [sys.executable, str(entry / "predict.py"), "--data", str(data), "--out", str(out), "--split", split],
        cwd=ROOT,
        env=child_env(),
        text=True,
        capture_output=True,
    )


def purge_modules() -> None:
    for name in ("data", "model", "predict"):
        sys.modules.pop(name, None)


def use_entry(entry: Path):
    purge_modules()
    sys.path.insert(0, str(entry))


def test_source_has_no_global_patch(entry: Path) -> None:
    text = (entry / "model.py").read_text(encoding="utf-8") + (entry / "predict.py").read_text(encoding="utf-8")
    if ".float()" in text or "Tensor.float =" in text or "torch.Tensor.float" in text:
        fail(f"{entry.name} contains a .float() call or a Tensor.float assignment")


def test_window_and_missing(entry: Path) -> None:
    use_entry(entry)
    import numpy as np
    import torch
    from data import MISSING, load_public_split
    from model import USE_SIGNAL, episode_meta, window

    if USE_SIGNAL:
        fail("USE_SIGNAL default is not False")
    obs = np.zeros((20, 8), dtype=np.int8)
    for machine in range(8):
        obs[:, machine] = (machine + 1) % 5
    base = window(obs, 10, 8)
    for machine in range(8):
        future = obs.copy()
        future[10, machine] = (int(future[10, machine]) + 1) % 5
        future[15, machine] = 4
        if not np.array_equal(window(future, 10, 8), base):
            fail(f"future tick changed the cutoff window on machine {machine}")
        past = obs.copy()
        past[9, machine] = (int(past[9, machine]) + 2) % 5
        if np.array_equal(window(past, 10, 8), base):
            fail(f"tick t-1 did not change the window on machine {machine}")
    values = torch.tensor([-1, 0])
    dtype = torch.float64
    mask = (values >= 0).to(dtype=dtype)
    idx = torch.where(values >= 0, values, torch.full_like(values, 5))
    if int(idx[0]) != 5 or float(mask[0]) != 0.0:
        fail("MISSING was not index 5 with mask 0")
    if int(idx[1]) != 0 or float(mask[1]) != 1.0:
        fail("payout 0 was confused with MISSING")
    if idx.dtype != torch.int64:
        fail("categorical index was not integer")

    class Holder:
        queries = [{"episode_id": "e", "machine_id": "m0"}]

    if int(episode_meta(Holder())["e"].sum()) != 0:
        fail("no-signal meta was not zero")

    with tempfile.TemporaryDirectory(prefix="fp64-slot-", dir=RUNTIME) as tmp:
        split = Path(tmp) / "test"
        split.mkdir()
        machines = [f"m{i}" for i in (7, 0, 3, 1, 5, 2, 6, 4)]
        with (split / "queries.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"])
            for machine in machines:
                writer.writerow([f"q-{machine}", "ep", machine, "balanced", "LEAK", "4"])
        with (split / "history.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["episode_id", "machine_id", "tick", "payout"])
            writer.writerow(["ep", "m7", "1", "20"])
            writer.writerow(["ep", "m0", "0", "0"])
        opened = []
        real_open = open

        def spy(path, *args, **kwargs):
            opened.append(str(path))
            name = Path(str(path)).name
            if name in ("labels.csv", "conditions.csv", "targets.csv", "reference.csv"):
                raise AssertionError(name)
            return real_open(path, *args, **kwargs)

        import builtins
        builtins.open = spy
        try:
            data = load_public_split(tmp, "test")
        finally:
            builtins.open = real_open
        if any(name in path for path in opened for name in ("labels.csv", "conditions.csv")):
            fail("loader opened labels or conditions")
        episode = data.episodes[0]
        if episode.machine_ids != tuple(f"m{i}" for i in range(8)):
            fail(f"slots not sorted {episode.machine_ids}")
        if int(episode.obs[1, episode.slot_of("m7")]) != 4:
            fail("payout 20 did not land on the sorted slot")
        if int(episode.obs[0, episode.slot_of("m0")]) != 0:
            fail("payout 0 was not stored as index 0")
        if int(episode.obs[0, episode.slot_of("m7")]) != MISSING:
            fail("unobserved tick was not MISSING")
        if any("signal" in row or "LEAK" in row.values() for row in data.queries):
            fail("signal column was kept")
        try:
            load_public_split(tmp, "validation")
            fail("missing validation split did not fail")
        except FileNotFoundError:
            pass
    print(f"unit_ok {entry.name}")


def test_dtype_and_forecast_causality(entry: Path) -> None:
    use_entry(entry)
    import torch
    from predict import load_pinned
    import model as model_mod
    from model import CNN, build_query, predict

    float_impl = torch.Tensor.float
    torch_mod, blob, cfg = load_pinned()
    del torch_mod
    model_mod.USE_SIGNAL = False
    kernel = int(cfg.get("kernel", 3))
    net = CNN(cfg["h"], cfg["layers"], kernel, cfg["drop"], cfg["W"], use_signal=False)
    net.load_state_dict(blob["state"])
    net.double()
    net.eval()
    if torch.Tensor.float is not float_impl:
        fail("Tensor.float was monkeypatched")
    floating = [p for p in net.parameters() if p.is_floating_point()]
    if any(p.dtype != torch.float64 for p in floating):
        fail("a floating parameter is not float64")
    seen = {}

    def hook(name):
        def fn(module, inputs, output):
            seen[name] = (str(inputs[0].dtype), str(output.dtype))
        return fn

    net.mask.register_forward_hook(hook("mask"))
    net.tick.register_forward_hook(hook("tick"))
    net.pay.register_forward_hook(hook("pay"))
    from data import load_public_split
    data = load_public_split(PUBLIC, "test")
    features, _eps = build_query(data, cfg["W"])
    if features[0].dtype not in (torch.int8, torch.int64, torch.long):
        fail(f"payout window dtype is {features[0].dtype}")
    one = [item[:1] for item in features]
    with torch.no_grad():
        logits = net(*one)
        probs = torch.softmax(logits, dim=-1)
    if logits.dtype != torch.float64 or probs.dtype != torch.float64:
        fail(f"logit/softmax dtype {logits.dtype} {probs.dtype}")
    if seen.get("mask", ("", ""))[0] != "torch.float64" or seen.get("tick", ("", ""))[0] != "torch.float64":
        fail(f"mask/tick inputs are not float64: {seen}")
    if seen.get("pay", ("", ""))[0] != "torch.int64":
        fail(f"embedding index dtype {seen.get('pay')}")
    if torch.Tensor.float is not float_impl:
        fail("Tensor.float changed during forward")
    probs_np = predict(net, [item[:2] for item in features], "cpu")
    if probs_np.dtype != torch.float64 and str(probs_np.dtype) != "float64":
        fail(f"predict numpy dtype {probs_np.dtype}")

    import numpy as np
    from model import window
    episode = data.episodes[0]
    base = window(episode.obs, episode.cutoff, cfg["W"])
    longer = np.full((episode.cutoff + 3, 8), -1, dtype=np.int8)
    longer[: episode.obs.shape[0]] = episode.obs
    for machine in range(8):
        longer[episode.cutoff, machine] = machine % 5
        longer[episode.cutoff + 1, machine] = 4
    if not np.array_equal(window(longer, episode.cutoff, cfg["W"]), base):
        fail("future history changed the cutoff window")
    print(f"dtype_ok {entry.name} params={len(floating)} logit=float64")


def test_refusals(entry: Path) -> None:
    import torch
    with tempfile.TemporaryDirectory(prefix="fp64-neg-", dir=RUNTIME) as tmp:
        tmp_path = Path(tmp)
        missing_out = tmp_path / "missing.csv"
        proc = run_entry(entry, tmp_path / "nodata", missing_out)
        if proc.returncode == 0 or missing_out.exists() or "REFUSED" not in proc.stderr:
            fail(f"missing data did not fail closed: {proc.returncode} {proc.stderr}")

        tamper = tmp_path / "tamper"
        config = tmp_path / "config"
        signal = tmp_path / "signal"
        shutil.copytree(entry, tamper)
        shutil.copytree(entry, config)
        shutil.copytree(entry, signal)
        raw = bytearray((tamper / "checkpoint.pt").read_bytes())
        raw[-20] ^= 0x5A
        (tamper / "checkpoint.pt").write_bytes(raw)
        tamper_out = tmp_path / "tamper.csv"
        proc = run_entry(tamper, PUBLIC, tamper_out)
        if proc.returncode == 0 or tamper_out.exists() or "hash mismatch" not in proc.stderr:
            fail(f"tampered checkpoint: rc={proc.returncode} err={proc.stderr}")

        cfg_path = config / "config" / "config.json"
        cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        config_out = tmp_path / "config.csv"
        proc = run_entry(config, PUBLIC, config_out)
        if proc.returncode == 0 or config_out.exists() or "unclean config hash" not in proc.stderr:
            fail(f"tampered config: rc={proc.returncode} err={proc.stderr}")

        blob = torch.load(signal / "checkpoint.pt", map_location="cpu", weights_only=False)
        blob["config"]["no_signal"] = False
        blob["state"]["sig.weight"] = torch.zeros(3, 4)
        torch.save(blob, signal / "checkpoint.pt")
        signal_out = tmp_path / "signal.csv"
        proc = run_entry(signal, PUBLIC, signal_out)
        if proc.returncode == 0 or signal_out.exists() or "signal" not in proc.stderr.lower():
            fail(f"signal mode: rc={proc.returncode} err={proc.stderr}")
    print(f"negative_ok {entry.name}")


def kit_rows(path: Path):
    sys.path.insert(0, str(ROOT / "kit"))
    from slot_forecasting.io import read_submission
    from slot_forecasting.schema import PROBABILITY_COLUMNS
    rows = read_submission(path)
    return rows, PROBABILITY_COLUMNS


def max_abs(left: Path, right: Path) -> float:
    a, cols = kit_rows(left)
    b, _cols = kit_rows(right)
    if set(a) != set(b):
        fail(f"id mismatch {left.name} vs {right.name}")
    worst = 0.0
    for query_id in a:
        for idx in range(len(cols)):
            worst = max(worst, abs(a[query_id][idx] - b[query_id][idx]))
    return worst


def require_close(left: Path, right: Path, label: str) -> None:
    if not right.is_file():
        fail(f"missing compare target {right}")
    diff = max_abs(left, right)
    status = "PASS" if diff <= TOL else "FAIL"
    print(f"COMPARE {label} max_abs_after_rownorm={diff:.12g} threshold={TOL:.0e} {status}")
    if diff > TOL:
        fail(label)


def official_check(path: Path, split: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "kit" / "check_submission.py"), str(path), "--data", str(PUBLIC), "--split", split],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        fail(f"checker {split} {path}: {proc.stderr}")


def official_r(path: Path) -> None:
    sys.path.insert(0, str(ROOT / "kit"))
    from slot_forecasting.rounds import score_round
    result = score_round(path, PUBLIC)
    print(
        f"VALIDATION_R {path.name} mean_loss_ratio={result['mean_loss_ratio']:.12g} "
        f"technical_points={result['technical_points']:.12g} max_points={result['max_points']}"
    )


def full_runs(name: str, entry: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    first = OUT / f"{name}_test_a.csv"
    second = OUT / f"{name}_test_b.csv"
    val = OUT / f"{name}_val.csv"
    for dest, split in ((first, "test"), (second, "test"), (val, "val")):
        proc = run_entry(entry, PUBLIC, dest, split)
        print(proc.stderr.strip())
        if proc.returncode != 0 or not dest.is_file():
            fail(f"{name} {split} export failed: {proc.stderr}")
    if first.read_bytes() != second.read_bytes():
        diff = max_abs(first, second)
        fail(f"{name} repeat not bitwise max_abs={diff}")
    rows, _cols = kit_rows(first)
    if len(rows) != 1536:
        fail(f"{name} test ids {len(rows)}")
    print(f"REPEAT {name} bitwise sha={sha256_file(first)} bytes={first.stat().st_size} ids=1536")
    official_check(first, "test")
    official_check(val, "validation")
    official_r(val)


def main() -> None:
    global OUT, REVIEW_TEST, REVIEW_VAL
    parser = argparse.ArgumentParser(description="CNN03 float64 release harness")
    parser.add_argument("--entry", type=Path, default=CNN03)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--reference-test", type=Path, default=REVIEW_TEST)
    parser.add_argument("--reference-val", type=Path, default=REVIEW_VAL)
    args = parser.parse_args()
    entry = args.entry.resolve()
    OUT = args.out.resolve()
    REVIEW_TEST = args.reference_test.resolve()
    REVIEW_VAL = args.reference_val.resolve()
    print(f"python {sys.version.split()[0]} executable={sys.executable}")
    print(f"entry={entry}")
    print(f"out={OUT}")
    print(f"reference_test={REVIEW_TEST}")
    print(f"reference_val={REVIEW_VAL}")
    test_source_has_no_global_patch(entry)
    test_window_and_missing(entry)
    test_dtype_and_forecast_causality(entry)
    test_refusals(entry)
    full_runs("cnn03", entry)
    require_close(OUT / "cnn03_test_a.csv", REVIEW_TEST, "cnn03_vs_own_tower_test")
    require_close(OUT / "cnn03_val.csv", REVIEW_VAL, "cnn03_vs_own_tower_val")
    print(
        f"SHA {entry.name} model={sha256_file(entry / 'model.py')} "
        f"data={sha256_file(entry / 'data.py')} predict={sha256_file(entry / 'predict.py')} "
        f"checkpoint={sha256_file(entry / 'checkpoint.pt')} "
        f"config={sha256_file(entry / 'config' / 'config.json')}"
    )
    print("HARNESS_OK")


if __name__ == "__main__":
    main()
