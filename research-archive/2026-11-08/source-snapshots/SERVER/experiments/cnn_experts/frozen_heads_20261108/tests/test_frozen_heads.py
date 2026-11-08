"""CPU tests for frozen trunk + tiny signal heads. No tune metrics."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "source")))
import frozen_heads as F  # noqa: E402
from house_edge.data import MISSING, Episode, SplitData, TYPE_TO_IDX, PAYOUTS, load_split  # noqa: E402


def _episode(cutoff=24, length=24, seed=0, signals=None, labels=None):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 5, size=(length, 8), dtype=np.int8)
    obs[0, 0] = MISSING
    types = np.arange(8) % 3
    mids = tuple(f"m{i}" for i in range(8))
    ep = Episode("e0", mids, types.astype(np.int64), obs, cutoff)
    if signals is None:
        signals = [-1, -1, 0, 0, 1, 1, -1, 0]
    if labels is None:
        labels = [0, 1, 2, 3, 4, 0, 1, 2]
    rows, qlabels = [], {}
    for slot, mid in enumerate(mids):
        qid = f"e0-q{slot}"
        rows.append({
            "query_id": qid, "episode_id": "e0", "machine_id": mid,
            "machine_type": "balanced", "signal": str(signals[slot]), "cutoff": str(cutoff),
        })
        qlabels[qid] = labels[slot]
    return ep, rows, qlabels


def _data(ep, rows, qlabels, shuffle=False, labels=True):
    if shuffle:
        rows = list(reversed(rows))
    if not labels:
        qlabels = None
    return SplitData([ep], {"e0": 0}, rows, qlabels)


class _Boom:
    def __iter__(self):
        raise AssertionError("history path read queries")


def test_maps_and_head_budget():
    assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}
    assert tuple(PAYOUTS) == (0, 1, 2, 5, 20)
    assert MISSING == -1 and MISSING != 0
    head = F.TinyHead()
    assert F.trainable_count(head) == 30
    assert torch.equal(head.map.weight, torch.eye(5))
    assert torch.equal(head.map.bias, torch.zeros(5))
    trunk = F.make_trunk()
    assert F.trainable_count(trunk) == 0
    assert sum(p.numel() for p in trunk.parameters()) == 367109
    assert trunk.sig is None


def test_future_and_label_mutation():
    rng = np.random.default_rng(1)
    obs = rng.integers(0, 5, size=(40, 8), dtype=np.int8)
    t, width = 32, 16
    base = F.window(obs, t, width)
    future = obs.copy()
    future = np.concatenate([obs, rng.integers(0, 5, size=(4, 8), dtype=np.int8)])
    future[t:, :] = (future[t:, :] + 3) % 5
    assert np.array_equal(base, F.window(future, t, width))
    past = obs.copy()
    past[t - 1, :] = (past[t - 1, :] + 1) % 5
    assert not np.array_equal(base, F.window(past, t, width))
    ep, rows, labels = _episode()
    other = dict(labels)
    other[rows[3]["query_id"]] = (other[rows[3]["query_id"]] + 1) % 5
    a = F.build_cutoff(_data(ep, rows, labels), width=8, require_labels=True)
    b = F.build_cutoff(_data(ep, rows, other), width=8, require_labels=True)
    assert torch.equal(a["x"], b["x"]) and not torch.equal(a["y"], b["y"])
    logits = torch.arange(40, dtype=torch.float).view(1, 8, 5)
    mask = F.loss_mask(a["y"], a["sig"], None)
    la = torch.nn.functional.cross_entropy(logits[mask], a["y"].long()[mask])
    lb = torch.nn.functional.cross_entropy(logits[mask], b["y"].long()[mask])
    assert not torch.allclose(la, lb)


def test_history_ignores_signal_and_query_order():
    ep, rows, labels = _episode(cutoff=20, length=20)
    data = _data(ep, rows, labels)
    data.queries = _Boom()
    hist = F.build_history(data, width=8)
    assert bool((hist["t"] < hist["cutoffs"]).all())
    flipped = [dict(r, signal="1" if r["signal"] == "-1" else "-1") for r in rows]
    ca = F.build_cutoff(_data(ep, rows, labels), width=8, require_labels=True)
    cb = F.build_cutoff(_data(ep, flipped, labels), width=8, require_labels=True)
    cc = F.build_cutoff(_data(ep, rows, labels, shuffle=True), width=8, require_labels=True)
    assert torch.equal(ca["x"], cb["x"]) and torch.equal(ca["y"], cb["y"])
    assert not torch.equal(ca["sig"], cb["sig"])
    assert torch.equal(ca["x"], cc["x"]) and torch.equal(ca["sig"], cc["sig"])


def test_missing_not_zero():
    torch.manual_seed(0)
    model = F.make_trunk().eval()
    missing = torch.full((1, 8, 8), MISSING)
    zero = missing.clone()
    zero[0, -1, 0] = 0
    t = torch.tensor([16])
    ty = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        a = model(missing, t, ty, torch.zeros(1, 8, dtype=torch.long))
        b = model(zero, t, ty, torch.zeros(1, 8, dtype=torch.long))
        c = model(missing, t, ty, torch.full((1, 8), 2))
    assert not torch.allclose(a, b)
    assert torch.allclose(a, c)


def test_route_and_sparse_fallback_and_norm():
    route = F.route_table({-1: 63, 0: 64, 1: 251}, 64)
    assert route == {-1: "trunk", 0: "specialist", 1: "specialist"}
    base = np.zeros((8, 5), dtype=np.float64)
    base[:, 0] = 2
    heads = {}
    for sig, col in ((0, 2), (1, 3)):
        head = F.TinyHead().double()
        with torch.no_grad():
            head.map.weight.zero_()
            head.map.bias.zero_()
            head.map.bias[col] = 5
        heads[sig] = head
    signals = np.array([-1, 0, 1, -1, 0, 1, -1, 0])
    out, names = F.apply_heads(base, signals, heads, route)
    assert names[0] == "trunk" and np.array_equal(out[0], base[0])
    assert names[1] == "specialist_+0"
    probs = F.normalize_rows(torch.softmax(torch.tensor(out), dim=-1).numpy())
    assert np.allclose(probs.sum(-1), 1.0)
    raw = np.array([[0.2, 0.2, 0.2, 0.2, 0.4]], dtype=np.float64)
    assert abs(F.normalize_rows(raw).sum() - 1.0) < 1e-15


def test_frozen_trunk_unchanged_and_matched_steps():
    ep, rows, labels = _episode(cutoff=18, length=18)
    data = _data(ep, rows, labels)
    cut = F.build_cutoff(data, width=8, require_labels=True)
    torch.manual_seed(0)
    trunk = F.make_trunk()
    snap = F.snapshot_frozen(trunk)
    control = F.TinyHead()
    spec = F.TinyHead()
    cstats = F.train_head(control, trunk, cut, None, "cpu", epochs=1, lr=1e-3, wd=0.0, bs=512)
    sstats = F.train_head(spec, trunk, cut, -1, "cpu", epochs=1, lr=1e-3, wd=0.0, bs=512)
    assert cstats["steps"] == sstats["steps"] == 1
    assert F.frozen_equal(trunk, snap)
    assert F.trainable_count(trunk) == 0
    assert not torch.allclose(control.map.weight, spec.map.weight)


def test_real_signal_counts_and_trunk_provenance():
    data = load_split(os.path.join(F.REPO, "kit", "public", "train"), with_labels=True)
    assert F.count_signal_rows(data) == {-1: 250, 0: 267, 1: 251}
    model, ckpt = F.load_trunk()
    assert ckpt["phase"] == "pretrain" and ckpt["steps"] == 1110
    assert ckpt["config"]["epochs"] == 30
    assert F.trainable_count(model) == 0
    assert F.sha256_file(os.path.join(F.REPO, F.TRUNK_REL)) == F.TRUNK_SHA


def test_seed_guard():
    with pytest.raises(SystemExit, match="rejected seed 1"):
        F.require_seed0(1)
    F.require_seed0(0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    for script in ("frozen_heads.py", "run_seed.py"):
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "..", "source", script), "--seed", "2", "--out", "/tmp/frozen_heads_reject"],
            capture_output=True, text=True, env=env, cwd=F.REPO,
        )
        assert proc.returncode != 0
        assert "rejected seed 2" in (proc.stdout + proc.stderr)
