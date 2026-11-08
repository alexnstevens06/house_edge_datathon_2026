"""Scoped float64 checks for the seed 0+2 equal ensemble. Exits 1 on a failed check.

This file is not the production harness and it does not read or write
experiments/mlp_cnn_hpo/review/PINNED.json. It does not point that pin at
this candidate.

Repo root is parents[3]. Output must stay inside
candidate_t146_ensemble_s0_s2_20261108. Official validation-round R is not
computed. Numeric tolerance stays 1e-8.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = Path(__file__).resolve().parent
PUBLIC = ROOT / "kit" / "public"
WORK = CANDIDATE / "_work"
OUT = CANDIDATE / "harness_out"
ENTRY = CANDIDATE / "unpacked"
TOL = 1e-8
RSS_CAP_KB = 2 * 1024 * 1024
NPZ = CANDIDATE / "refs" / "ensemble_equal_s0_s2.npz"
NPZ_SHA256 = "eb3ad2f66ae1898a97a8aaf9c9f3c2a6a9e173a3de43e161fffc3c8f1341356c"
TIMING = re.compile(
    r"s0_infer_s=(?P<s0>[0-9.]+) s2_infer_s=(?P<s2>[0-9.]+) "
    r"infer_s=(?P<both>[0-9.]+) total_s=(?P<total>[0-9.]+) "
    r"max_rss_kb=(?P<rss>\d+)"
)


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
    env["OPENBLAS_NUM_THREADS"] = "2"
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
    if "components/s0" not in text or "components/s2" not in text:
        fail("predict.py does not name both component directories")


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

    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fp64-slot-", dir=WORK) as tmp:
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
    from predict import prepare
    import model as model_mod
    from model import build_query, predict

    float_impl = torch.Tensor.float
    prepared = prepare()
    if [item[0]["id"] for item in prepared] != ["s0", "s2"]:
        fail("prepare order is not s0 then s2")
    model_mod.USE_SIGNAL = False
    from data import load_public_split
    data = load_public_split(PUBLIC, "test")
    features, _eps = build_query(data, prepared[0][2]["W"])
    if features[0].dtype not in (torch.int8, torch.int64, torch.long):
        fail(f"payout window dtype is {features[0].dtype}")
    one = [item[:1] for item in features]
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        fail("autocast is enabled")
    for spec, blob, cfg, net in prepared:
        cid = spec["id"]
        if "kernel" in cfg:
            fail(f"{cid} parent config grew a kernel field")
        for index in range(4):
            weight = blob["state"][f"blocks.{index}.conv.weight"]
            if tuple(weight.shape) != (128, 128, 3):
                fail(f"{cid} blocks.{index}.conv.weight shape {tuple(weight.shape)}")
        if torch.Tensor.float is not float_impl:
            fail("Tensor.float was monkeypatched")
        floating = [p for p in net.parameters() if p.is_floating_point()]
        if len(floating) != 43 or any(p.dtype != torch.float64 for p in floating):
            fail(f"{cid} floating parameters are not 43 float64 tensors")
        seen = {}

        def hook(name):
            def fn(module, inputs, output):
                seen[name] = (str(inputs[0].dtype), str(output.dtype))
            return fn

        net.mask.register_forward_hook(hook("mask"))
        net.tick.register_forward_hook(hook("tick"))
        net.pay.register_forward_hook(hook("pay"))
        with torch.no_grad():
            logits = net(*one)
            probs = torch.softmax(logits, dim=-1)
        if logits.dtype != torch.float64 or probs.dtype != torch.float64:
            fail(f"{cid} logit/softmax dtype {logits.dtype} {probs.dtype}")
        if seen.get("mask", ("", ""))[0] != "torch.float64" or seen.get("tick", ("", ""))[0] != "torch.float64":
            fail(f"{cid} mask/tick inputs are not float64: {seen}")
        if seen.get("pay", ("", ""))[0] != "torch.int64":
            fail(f"{cid} embedding index dtype {seen.get('pay')}")
        probs_np = predict(net, [item[:2] for item in features], "cpu", bs=4)
        if str(probs_np.dtype) != "float64":
            fail(f"{cid} predict numpy dtype {probs_np.dtype}")
        print(f"dtype_ok {cid} params={len(floating)} logit=float64")
    if torch.Tensor.float is not float_impl:
        fail("Tensor.float changed during forward")

    import numpy as np
    from model import window
    episode = data.episodes[0]
    base = window(episode.obs, episode.cutoff, prepared[0][2]["W"])
    longer = np.full((episode.cutoff + 3, 8), -1, dtype=np.int8)
    longer[: episode.obs.shape[0]] = episode.obs
    for machine in range(8):
        longer[episode.cutoff, machine] = machine % 5
        longer[episode.cutoff + 1, machine] = 4
    if not np.array_equal(window(longer, episode.cutoff, prepared[0][2]["W"]), base):
        fail("future history changed the cutoff window")


def test_query_order(entry: Path) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fp64-order-", dir=WORK) as tmp:
        root = Path(tmp)
        for name, order in (("forward", list(range(8))), ("reversed", list(range(7, -1, -1)))):
            split = root / name / "test"
            split.mkdir(parents=True)
            machines = [f"m{i}" for i in order]
            with (split / "queries.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"])
                for machine in machines:
                    writer.writerow([f"q-{machine}", "ep", machine, "balanced", "LEAK", "4"])
            with (split / "history.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["episode_id", "machine_id", "tick", "payout"])
                payouts = ("0", "1", "2", "5", "20")
                for machine_index, machine in enumerate(f"m{i}" for i in range(8)):
                    writer.writerow(["ep", machine, "0", payouts[machine_index % 5]])
                    writer.writerow(["ep", machine, "1", "1"])
                    writer.writerow(["ep", machine, "2", "20"])
            out = root / f"{name}.csv"
            proc = run_entry(entry, root / name, out, "test")
            if proc.returncode != 0 or not out.is_file():
                fail(f"query-order {name} failed: {proc.stderr}")
        forward = (root / "forward.csv").read_bytes()
        reverse = (root / "reversed.csv").read_bytes()
        if forward != reverse:
            fail("query row order changed the submission bytes")
        if forward.count(b"\n") != 9:
            fail(f"query-order row count {forward.count(b'\\n')}")
    print(f"query_order_ok {entry.name}")


def expect_refuse(entry: Path, needle: str, label: str) -> None:
    out = entry.parent / f"{label}.csv"
    proc = run_entry(entry, PUBLIC, out)
    if proc.returncode == 0 or out.exists() or needle not in proc.stderr:
        fail(f"{label}: rc={proc.returncode} err={proc.stderr}")


def test_refusals(entry: Path) -> None:
    import torch
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fp64-neg-", dir=WORK) as tmp:
        tmp_path = Path(tmp)
        missing_out = tmp_path / "missing.csv"
        proc = run_entry(entry, tmp_path / "nodata", missing_out)
        if proc.returncode == 0 or missing_out.exists() or "REFUSED" not in proc.stderr:
            fail(f"missing data did not fail closed: {proc.returncode} {proc.stderr}")

        cases = ("s0_hash", "s2_hash", "s0_config", "s2_config", "s0_signal", "s2_signal",
                 "s0_key", "s2_key", "s0_shape", "s2_shape", "s0_dtype", "s2_dtype",
                 "s0_nonfinite", "s2_nonfinite", "weights", "extra")
        trees = {name: tmp_path / name for name in cases}
        for dest in trees.values():
            shutil.copytree(entry, dest)

        for cid in ("s0", "s2"):
            raw = bytearray((trees[f"{cid}_hash"] / "components" / cid / "checkpoint.pt").read_bytes())
            raw[-20] ^= 0x5A
            (trees[f"{cid}_hash"] / "components" / cid / "checkpoint.pt").write_bytes(raw)
            expect_refuse(trees[f"{cid}_hash"], f"{cid} checkpoint hash mismatch", cid + "_hash")

            cfg_path = trees[f"{cid}_config"] / "components" / cid / "config.json"
            cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            expect_refuse(trees[f"{cid}_config"], f"{cid} unclean config hash", cid + "_config")

            blob = torch.load(trees[f"{cid}_signal"] / "components" / cid / "checkpoint.pt", map_location="cpu", weights_only=False)
            blob["config"]["no_signal"] = False
            blob["state"]["sig.weight"] = torch.zeros(3, 4)
            torch.save(blob, trees[f"{cid}_signal"] / "components" / cid / "checkpoint.pt")
            expect_refuse(trees[f"{cid}_signal"], f"{cid} signal", cid + "_signal")

            blob = torch.load(trees[f"{cid}_key"] / "components" / cid / "checkpoint.pt", map_location="cpu", weights_only=False)
            blob["state"]["extra.weight"] = torch.zeros(1)
            torch.save(blob, trees[f"{cid}_key"] / "components" / cid / "checkpoint.pt")
            expect_refuse(trees[f"{cid}_key"], f"{cid} architecture keys mismatch", cid + "_key")

            blob = torch.load(trees[f"{cid}_shape"] / "components" / cid / "checkpoint.pt", map_location="cpu", weights_only=False)
            blob["state"]["blocks.0.conv.weight"] = torch.zeros(128, 128, 5)
            torch.save(blob, trees[f"{cid}_shape"] / "components" / cid / "checkpoint.pt")
            expect_refuse(trees[f"{cid}_shape"], "shape", cid + "_shape")

            blob = torch.load(trees[f"{cid}_dtype"] / "components" / cid / "checkpoint.pt", map_location="cpu", weights_only=False)
            blob["state"]["head.2.bias"] = blob["state"]["head.2.bias"].to(torch.float16)
            torch.save(blob, trees[f"{cid}_dtype"] / "components" / cid / "checkpoint.pt")
            expect_refuse(trees[f"{cid}_dtype"], "saved dtype", cid + "_dtype")

            blob = torch.load(trees[f"{cid}_nonfinite"] / "components" / cid / "checkpoint.pt", map_location="cpu", weights_only=False)
            blob["state"]["head.2.bias"] = torch.tensor([float("nan"), 0, 0, 0, 0])
            torch.save(blob, trees[f"{cid}_nonfinite"] / "components" / cid / "checkpoint.pt")
            expect_refuse(trees[f"{cid}_nonfinite"], "is not finite", cid + "_nonfinite")

        raw = bytearray((trees["weights"] / "provenance" / "weights.json").read_bytes())
        raw[-5] ^= 0x5A
        (trees["weights"] / "provenance" / "weights.json").write_bytes(raw)
        expect_refuse(trees["weights"], "weights manifest hash mismatch", "weights")

        extra = trees["extra"] / "components" / "s1"
        extra.mkdir()
        (extra / "checkpoint.pt").write_bytes(b"nope")
        expect_refuse(trees["extra"], "unexpected component directory", "extra")
    print(f"negative_ok {entry.name}")


def kit_rows(path: Path):
    sys.path.insert(0, str(ROOT / "kit"))
    from slot_forecasting.io import read_submission
    from slot_forecasting.schema import PROBABILITY_COLUMNS
    rows = read_submission(path)
    return rows, PROBABILITY_COLUMNS


def max_abs_kit(csv_path: Path, npz_path: Path) -> tuple[float, float]:
    import numpy as np
    rows, cols = kit_rows(csv_path)
    blob = np.load(npz_path)
    keys = [str(item) for item in blob["query_id"].tolist()]
    values = blob["probability"]
    if set(keys) != set(rows) or len(cols) != 5:
        fail(f"id mismatch csv={len(rows)} npz={len(keys)}")
    kit_worst = 0.0
    raw_worst = 0.0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = {row["query_id"]: [float(row[name]) for name in cols] for row in csv.DictReader(handle)}
    for index, key in enumerate(keys):
        raw_npz = [float(item) for item in values[index]]
        total = math.fsum(raw_npz)
        kit_npz = [item / total for item in raw_npz]
        for left, right in zip(rows[key], kit_npz):
            kit_worst = max(kit_worst, abs(left - right))
        for left, right in zip(raw_rows[key], raw_npz):
            raw_worst = max(raw_worst, abs(left - right))
    return kit_worst, raw_worst


def structural_coverage(path: Path, split: str, expect: int) -> None:
    rows, cols = kit_rows(path)
    with (PUBLIC / split / "queries.csv").open(newline="", encoding="utf-8") as handle:
        wanted = {row["query_id"] for row in csv.DictReader(handle)}
    if set(rows) != wanted or len(rows) != expect or len(cols) != 5:
        missing = len(wanted - set(rows))
        extra = len(set(rows) - wanted)
        fail(f"{split} coverage ids={len(rows)} missing={missing} extra={extra} columns={len(cols)}")
    finite = True
    for row in rows.values():
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            finite = False
        if abs(sum(row) - 1.0) > 1e-8:
            finite = False
    if not finite:
        fail(f"{split} row is not a finite simplex after kit norm")
    print(f"COVERAGE {split} ids={len(rows)} columns={len(cols)} finite_simplex targets_not_opened")


def official_check(path: Path, split: str) -> None:
    if split != "test":
        fail("validation checker scores held-out rows; use structural_coverage")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "kit" / "check_submission.py"), str(path), "--data", str(PUBLIC), "--split", split],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        fail(f"checker {split} {path}: {proc.stderr}")


def require_timings(stderr: str, label: str) -> None:
    match = TIMING.search(stderr)
    if match is None or "components=s0,s2" not in stderr:
        fail(f"{label} did not report both component timings: {stderr}")
    rss = int(match.group("rss"))
    print(
        f"TIMING {label} s0_infer_s={match.group('s0')} s2_infer_s={match.group('s2')} "
        f"infer_s={match.group('both')} total_s={match.group('total')} max_rss_kb={rss}"
    )
    if rss > RSS_CAP_KB:
        fail(f"{label} max_rss_kb={rss} exceeds {RSS_CAP_KB}")


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
        require_timings(proc.stderr, f"{name}_{split}_{dest.name}")
    if first.read_bytes() != second.read_bytes():
        fail(f"{name} repeat not bitwise")
    rows, _cols = kit_rows(first)
    if len(rows) != 1536:
        fail(f"{name} test ids {len(rows)}")
    val_rows, _cols = kit_rows(val)
    if len(val_rows) != 768:
        fail(f"{name} val ids {len(val_rows)}")
    print(f"REPEAT {name} bitwise sha={sha256_file(first)} bytes={first.stat().st_size} ids=1536")
    official_check(first, "test")
    structural_coverage(val, "validation", 768)
    if sha256_file(NPZ) != NPZ_SHA256:
        fail("verified ensemble npz hash drifted")
    kit_abs, raw_abs = max_abs_kit(val, NPZ)
    status = "PASS" if kit_abs <= TOL else "FAIL"
    print(f"COMPARE package_vs_verified_ensemble kit_max_abs={kit_abs:.12g} raw_max_abs={raw_abs:.12g} threshold={TOL:.0e} {status}")
    if kit_abs > TOL:
        fail("package vs verified ensemble")
    print("VALIDATION_R not_computed reason=confirm_episodes_are_held_out")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description="ensemble s0+s2 float64 candidate harness")
    parser.add_argument("--entry", type=Path, default=ENTRY)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    entry = args.entry.resolve()
    OUT = args.out.resolve()
    if "candidate_t146_ensemble_s0_s2_20261108" not in OUT.parts:
        fail(f"refusing to write harness output outside the candidate folder: {OUT}")
    if sha256_file(NPZ) != NPZ_SHA256:
        fail("vendored npz hash mismatch before checks")
    print(f"python {sys.version.split()[0]} executable={sys.executable}")
    print(f"entry={entry}")
    print(f"out={OUT}")
    print("scope=candidate_parent_not_production_pin")
    test_source_has_no_global_patch(entry)
    test_window_and_missing(entry)
    test_dtype_and_forecast_causality(entry)
    test_query_order(entry)
    test_refusals(entry)
    full_runs("candidate", entry)
    proc = subprocess.run(
        [
            sys.executable,
            str(CANDIDATE / "score_tune_only.py"),
            "--entry",
            str(entry),
            "--val-csv",
            str(OUT / "candidate_val.csv"),
            "--out",
            str(CANDIDATE / "tune_only.json"),
        ],
        cwd=ROOT,
        env=child_env(),
        text=True,
        capture_output=True,
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        fail(f"tune score failed: {proc.stderr}")
    if "TUNE_ONLY R=0.28468175920919014 " not in proc.stdout:
        fail(f"tune line missing: {proc.stdout}")
    print(
        f"SHA {entry.name} model={sha256_file(entry / 'model.py')} "
        f"data={sha256_file(entry / 'data.py')} predict={sha256_file(entry / 'predict.py')} "
        f"s0={sha256_file(entry / 'components' / 's0' / 'checkpoint.pt')} "
        f"s2={sha256_file(entry / 'components' / 's2' / 'checkpoint.pt')}"
    )
    print("HARNESS_OK")


if __name__ == "__main__":
    main()
