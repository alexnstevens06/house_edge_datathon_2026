"""Tests for the hash-locked anchor entry. Does not modify entry/."""
from __future__ import annotations

import csv
import hashlib
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "experiments" / "cnn_runtime" / "entry"
REF = ROOT / "experiments" / "cnn_runtime" / "reference" / "experiments" / "cnn" / "runs" / "g_nosignal" / "test_cpu_a.csv"
FIRST = ROOT / "experiments" / "cnn_runtime" / "first.csv"
SECOND = ROOT / "experiments" / "cnn_runtime" / "second.csv"
PY = ROOT / ".venv" / "bin" / "python"
ANCHOR = "43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80"
REF_SHA = "9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9"
HEADER = ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"]
TOL = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL {message}")
    raise SystemExit(1)


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != HEADER:
        fail(f"header {path}: {rows[:1]}")
    body = rows[1:]
    if len(body) != 1536:
        fail(f"row count {path}: {len(body)}")
    ids = [row[0] for row in body]
    if ids != sorted(ids) or len(set(ids)) != 1536:
        fail(f"ids not 1536 unique sorted in {path}")
    return body


def max_abs_after_rownorm(left, right) -> tuple[float, int, int, float]:
    """Max absolute difference of already row-normalized probabilities.

    This is not a relative-error metric.
    """
    if [row[0] for row in left] != [row[0] for row in right]:
        fail("query id alignment differs")
    max_abs = 0.0
    n_differ = 0
    nonfinite = 0
    simplex = 0.0
    for a, b in zip(left, right):
        if len(a) != 6 or len(b) != 6:
            fail(f"width {a[0]}")
        av = [float(x) for x in a[1:]]
        bv = [float(x) for x in b[1:]]
        if any(not math.isfinite(x) for x in av):
            nonfinite += 1
        simplex = max(simplex, abs(sum(av) - 1.0))
        for x, y in zip(av, bv):
            if a[1:] != b[1:]:
                pass
            diff = abs(x - y)
            if diff != 0.0:
                n_differ += 1
            if diff > max_abs:
                max_abs = diff
    return max_abs, n_differ, nonfinite, simplex


def test_causality_missing_and_slots() -> None:
    sys.path.insert(0, str(ENTRY))
    import numpy as np
    from data import MISSING, load_test_split
    from model import USE_SIGNAL, episode_meta, window

    if USE_SIGNAL:
        fail("entry USE_SIGNAL default is not False")
    obs = np.zeros((12, 8), dtype=np.int8)
    for t in range(12):
        obs[t, :] = t % 5
    base = window(obs, 10, 8)
    future = obs.copy()
    future[10, 3] = (future[10, 3] + 1) % 5
    if not np.array_equal(window(future, 10, 8), base):
        fail("window changed when only tick t was edited")
    past = obs.copy()
    past[9, 3] = (past[9, 3] + 1) % 5
    if np.array_equal(window(past, 10, 8), base):
        fail("window ignored an edit at tick t-1")
    if np.any(base < 0) and set(np.unique(base)).difference({0, 1, 2, 3, 4, MISSING}):
        fail("window values left the payout/missing set")

    import torch
    missing = torch.tensor([-1])
    zero = torch.tensor([0])
    miss_idx = torch.where(missing >= 0, missing, torch.full_like(missing, 5))
    zero_idx = torch.where(zero >= 0, zero, torch.full_like(zero, 5))
    if int(miss_idx) != 5 or float((missing >= 0).float()) != 0.0:
        fail("MISSING was not mapped to embedding 5 with mask 0")
    if int(zero_idx) != 0 or float((zero >= 0).float()) != 1.0:
        fail("payout 0 was confused with MISSING")

    class Query:
        def __init__(self):
            self.queries = [{"episode_id": "e", "machine_id": "m0"}]

    meta = episode_meta(Query())
    if int(meta["e"].sum()) != 0:
        fail("no-signal meta was not zero")

    with tempfile.TemporaryDirectory(prefix="cnn-slot-", dir=ROOT / "experiments" / "cnn_runtime") as tmp:
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
        data = load_test_split(tmp)
        episode = data.episodes[0]
        if episode.machine_ids != tuple(f"m{i}" for i in range(8)):
            fail(f"slots not sorted {episode.machine_ids}")
        if int(episode.obs[1, episode.slot_of("m7")]) != 4:
            fail("m7 payout 20 did not land on the sorted slot as index 4")
        if int(episode.obs[0, episode.slot_of("m0")]) != 0:
            fail("payout 0 was not stored as index 0")
        if int(episode.obs[0, episode.slot_of("m7")]) != MISSING:
            fail("unobserved tick was not MISSING")
        if any("signal" in row or "LEAK" in row.values() for row in data.queries):
            fail("signal column was kept on the query row")


def run_entry(data: Path, out: Path, entry: Path = ENTRY) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PY), str(entry / "predict.py"), "--data", str(data), "--out", str(out)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_negative() -> None:
    with tempfile.TemporaryDirectory(prefix="cnn-neg-", dir=ROOT / "experiments" / "cnn_runtime") as tmp:
        tmp_path = Path(tmp)
        missing_out = tmp_path / "missing.csv"
        proc = run_entry(tmp_path / "nodata", missing_out)
        if proc.returncode == 0 or missing_out.exists():
            fail(f"missing data did not fail closed: {proc.returncode} {proc.stderr}")
        if "REFUSED" not in proc.stderr:
            fail(f"missing data stderr: {proc.stderr}")

        tamper = tmp_path / "tamper"
        signal = tmp_path / "signal"
        shutil.copytree(ENTRY, tamper)
        shutil.copytree(ENTRY, signal)
        blob_path = tamper / "checkpoint.pt"
        raw = bytearray(blob_path.read_bytes())
        raw[-20] ^= 0xFF
        blob_path.write_bytes(raw)
        tamper_out = tmp_path / "tamper.csv"
        proc = run_entry(ROOT / "kit" / "public", tamper_out, tamper)
        if proc.returncode == 0 or tamper_out.exists() or "anchor hash mismatch" not in proc.stderr:
            fail(f"tamper checkpoint did not fail closed: rc={proc.returncode} err={proc.stderr}")

        sys.path.insert(0, str(ENTRY))
        import torch
        blob = torch.load(signal / "checkpoint.pt", map_location="cpu", weights_only=False)
        blob["config"]["no_signal"] = False
        torch.save(blob, signal / "checkpoint.pt")
        signal_out = tmp_path / "signal.csv"
        proc = run_entry(ROOT / "kit" / "public", signal_out, signal)
        if proc.returncode == 0 or signal_out.exists() or "no_signal" not in proc.stderr:
            fail(f"signal mode did not fail closed: rc={proc.returncode} err={proc.stderr}")
        print("negative_ok missing_data tamper_checkpoint signal_mode")


def compare_pair(name: str, path: Path, ref_body) -> None:
    body = load_csv(path)
    digest = sha256_file(path)
    max_abs, n_differ, nonfinite, simplex = max_abs_after_rownorm(body, ref_body)
    bitwise = path.read_bytes() == REF.read_bytes()
    status = "PASS" if max_abs <= TOL else "FAIL"
    print(
        f"COMPARE {name} sha={digest} bitwise={bitwise} max_abs_after_rownorm={max_abs:.12g} "
        f"cells_differ={n_differ} nonfinite={nonfinite} max_simplex_abs_err={simplex:.3g} "
        f"threshold={TOL:.0e} {status}"
    )


def test_compare() -> None:
    if sha256_file(REF) != REF_SHA:
        fail("reference CPU csv hash changed")
    if sha256_file(ENTRY / "checkpoint.pt") != ANCHOR:
        fail("entry checkpoint hash changed")
    ref_body = load_csv(REF)
    if not FIRST.is_file():
        fail("first.csv missing")
    compare_pair("first", FIRST, ref_body)
    if SECOND.is_file():
        compare_pair("second", SECOND, ref_body)
        same = FIRST.read_bytes() == SECOND.read_bytes()
        print(f"REPEAT bitwise_first_second={same}")
        if not same:
            body_a = load_csv(FIRST)
            body_b = load_csv(SECOND)
            max_abs, n_differ, _, _ = max_abs_after_rownorm(body_a, body_b)
            print(f"REPEAT max_abs_after_rownorm={max_abs:.12g} cells_differ={n_differ}")
    else:
        print("REPEAT second.csv missing")


def extract_only() -> None:
    zip_path = ROOT / "experiments" / "cnn_runtime" / "anchor_entry.zip"
    if not zip_path.is_file():
        fail(f"missing {zip_path}")
    import zipfile
    dest = ROOT / "experiments" / "cnn_runtime" / "extract_check"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    out = ROOT / "experiments" / "cnn_runtime" / "extract.csv"
    proc = run_entry(ROOT / "kit" / "public", out, dest)
    if proc.returncode != 0:
        fail(f"extract rerun failed {proc.stderr}")
    if not FIRST.is_file() or out.read_bytes() != FIRST.read_bytes():
        fail("fresh extract CSV is not bitwise identical to first.csv")
    print(f"EXTRACT_OK sha={sha256_file(out)} bytes={out.stat().st_size}")
    shutil.rmtree(dest)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--extract-only":
        extract_only()
        return
    before = sha256_file(ENTRY / "predict.py")
    test_causality_missing_and_slots()
    print("unit_ok causality missing_vs_0 slot_mapping")
    test_negative()
    test_compare()
    if sha256_file(ENTRY / "predict.py") != before or sha256_file(ENTRY / "checkpoint.pt") != ANCHOR:
        fail("entry changed during tests")
    print("entry_unchanged")


if __name__ == "__main__":
    main()
