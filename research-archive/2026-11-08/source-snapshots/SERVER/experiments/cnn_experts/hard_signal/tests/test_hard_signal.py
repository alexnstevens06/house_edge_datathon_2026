"""CPU tests for the hard-signal CNN router. No training metrics."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "source")))
import hard_signal as H  # noqa: E402
from house_edge.data import MISSING, Episode, SplitData, TYPE_TO_IDX, PAYOUTS  # noqa: E402


def _episode(eid="e0", cutoff=24, length=24, seed=0, signals=None, labels=None):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 5, size=(length, 8), dtype=np.int8)
    obs[0, 0] = MISSING
    types = np.array([i % 3 for i in range(8)], dtype=np.int64)
    mids = tuple(f"m{i}" for i in range(8))
    ep = Episode(eid, mids, types, obs, cutoff)
    if signals is None:
        signals = [-1, -1, 0, 0, 1, 1, -1, 0]
    if labels is None:
        labels = [0, 1, 2, 3, 4, 0, 1, 2]
    rows = []
    qlabels = {}
    for slot, mid in enumerate(mids):
        qid = f"{eid}-q{slot}"
        rows.append({
            "query_id": qid, "episode_id": eid, "machine_id": mid,
            "machine_type": "balanced", "signal": str(signals[slot]), "cutoff": str(cutoff),
        })
        qlabels[qid] = labels[slot]
    return ep, rows, qlabels


def _data(episodes_rows, labels=None, shuffle=False):
    episodes, rows = [], []
    qlabels = {}
    for ep, erows, elabels in episodes_rows:
        episodes.append(ep)
        rows.extend(erows)
        qlabels.update(elabels)
    if shuffle:
        rows = list(reversed(rows))
    if labels is False:
        qlabels = None
    index = {ep.episode_id: i for i, ep in enumerate(episodes)}
    return SplitData(episodes, index, rows, qlabels)


def test_contract_maps_and_budget():
    assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}
    assert tuple(PAYOUTS) == (0, 1, 2, 5, 20)
    assert MISSING == -1 and MISSING != 0
    assert H.phase_hyper("control_finetune") == H.phase_hyper("specialist_finetune")
    assert H.phase_hyper("pretrain")["lr"] == H.phase_hyper("specialist_finetune")["lr"]
    assert H.phase_hyper("pretrain")["epochs"] == 30
    assert H.phase_hyper("control_finetune")["epochs"] == 10
    assert H.phase_hyper("control_finetune")["epochs"] == H.FT_EPOCHS
    assert H.MIN_GROUP_ROWS == 64
    model = H.make_cnn()
    assert sum(p.numel() for p in model.parameters()) == 367109
    assert model.use_signal is False and model.sig is None


def test_all8_future_mutation_invariance():
    rng = np.random.default_rng(1)
    obs = rng.integers(0, 5, size=(40, 8), dtype=np.int8)
    t, width = 32, 16
    base = H.window(obs, t, width)
    future = np.concatenate([obs, rng.integers(0, 5, size=(8, 8), dtype=np.int8)])
    future[t:, :] = (future[t:, :] + 3) % 5
    assert np.array_equal(base, H.window(future, t, width))
    past = obs.copy()
    past[t - 1, :] = (past[t - 1, :] + 1) % 5
    assert not np.array_equal(base, H.window(past, t, width))
    torch.manual_seed(0)
    model = H.B.CNN(8, 1, 3, 0.0, width, use_signal=False).eval()
    ty = torch.zeros(1, 8, dtype=torch.long)
    sg = torch.zeros(1, 8, dtype=torch.long)
    tick = torch.tensor([t])

    def logits(arr):
        x = torch.tensor(np.array(H.window(arr, t, width))).unsqueeze(0)
        return model(x, tick, ty, sg)

    assert torch.allclose(logits(obs), logits(future))
    assert not torch.allclose(logits(obs), logits(past))


def test_label_mutation_changes_targets_not_features():
    ep, rows, labels = _episode()
    other = dict(labels)
    other[rows[3]["query_id"]] = (other[rows[3]["query_id"]] + 1) % 5
    a = _data([(ep, rows, labels)])
    b = _data([(ep, rows, other)])
    ha, hb = H.build_history(a), H.build_history(b)
    assert torch.equal(ha.X, hb.X) and torch.equal(ha.Y, hb.Y) and ha.signals is None
    ca, cb = H.build_cutoff(a, require_labels=True), H.build_cutoff(b, require_labels=True)
    assert torch.equal(ca.X, cb.X)
    assert not torch.equal(ca.Y, cb.Y)
    y = ca.Y.clone()
    mask = H.loss_mask(y, ca.signals, "control", None)
    # Uniform logits give the same cross-entropy for every class, so they cannot
    # show that a changed label changes the loss.
    logits = torch.arange(40, dtype=torch.float).view(1, 8, 5)
    loss_a = H.ce_loss(logits, ca.Y, mask)
    loss_b = H.ce_loss(logits, cb.Y, mask)
    assert not torch.allclose(loss_a, loss_b)


def test_signal_mutation_does_not_change_historical_features():
    ep, rows, labels = _episode()
    flipped = []
    for row in rows:
        copy = dict(row)
        copy["signal"] = "1" if row["signal"] == "-1" else "-1"
        flipped.append(copy)
    a = _data([(ep, rows, labels)])
    b = _data([(ep, flipped, labels)])
    # History must not iterate queries. A hostile queries object proves it.
    a.queries = _Boom()
    ha = H.build_history(a)
    b_hist = H.build_history(b)
    assert torch.equal(ha.X, b_hist.X) and torch.equal(ha.Y, b_hist.Y)
    assert ha.signals is None and bool((ha.T < ha.cutoffs).all())
    a.queries = rows
    ca, cb = H.build_cutoff(a, require_labels=True), H.build_cutoff(b, require_labels=True)
    assert torch.equal(ca.X, cb.X) and torch.equal(ca.Y, cb.Y)
    assert not torch.equal(ca.signals, cb.signals)


class _Boom:
    def __iter__(self):
        raise AssertionError("history path read queries")


def test_query_row_order_invariance():
    ep, rows, labels = _episode()
    a = _data([(ep, rows, labels)])
    b = _data([(ep, rows, labels)], shuffle=True)
    ca, cb = H.build_cutoff(a, require_labels=True), H.build_cutoff(b, require_labels=True)
    assert torch.equal(ca.X, cb.X) and torch.equal(ca.Y, cb.Y) and torch.equal(ca.signals, cb.signals)
    route = {s: "specialist" for s in H.SIGNALS}
    shared = np.zeros((8, 5))
    shared[:, 0] = 1
    spec = {s: np.zeros((8, 5)) for s in H.SIGNALS}
    for i, s in enumerate(H.SIGNALS):
        spec[s][:, i + 1] = 1
    signals = ca.signals[0].numpy()
    out_a, names_a = H.assemble_slots(shared, spec, signals, route)
    out_b, names_b = H.assemble_slots(shared, spec, cb.signals[0].numpy(), route)
    assert np.array_equal(out_a, out_b) and names_a == names_b


def test_missing_distinguished_from_zero_and_signal_input_ignored():
    torch.manual_seed(0)
    model = H.B.CNN(8, 1, 3, 0.0, 8, use_signal=False).eval()
    missing = torch.full((1, 8, 8), MISSING)
    zero = missing.clone()
    zero[0, -1, 0] = 0
    t = torch.tensor([16])
    ty = torch.zeros(1, 8, dtype=torch.long)
    sg = torch.zeros(1, 8, dtype=torch.long)
    y_missing = model(missing, t, ty, sg)
    y_zero = model(zero, t, ty, sg)
    assert not torch.allclose(y_missing, y_zero)
    sg_other = torch.full_like(sg, 2)
    assert torch.allclose(y_missing, model(missing, t, ty, sg_other))


def test_exact_hard_matching_and_normalized():
    shared = np.zeros((8, 5))
    shared[:, 0] = 1
    spec = {}
    for i, s in enumerate(H.SIGNALS):
        arr = np.zeros((8, 5))
        arr[:, i + 1] = 1
        spec[s] = arr
    signals = np.array([-1, -1, 0, 0, 1, 1, -1, 0])
    route = {s: "specialist" for s in H.SIGNALS}
    out, names = H.assemble_slots(shared, spec, signals, route)
    assert np.allclose(out.sum(-1), 1)
    for slot, sig in enumerate(signals):
        assert names[slot] == f"specialist_{sig:+d}"
        assert np.array_equal(out[slot], spec[sig][slot])
        assert not np.array_equal(out[slot], shared[slot])


def test_sparse_group_fallback():
    counts = {-1: 63, 0: 64, 1: 251}
    route = H.route_table(counts, 64)
    assert route == {-1: "shared", 0: "specialist", 1: "specialist"}
    shared = np.zeros((8, 5))
    shared[:, 0] = 1
    spec = {0: np.zeros((8, 5)), 1: np.zeros((8, 5))}
    spec[0][:, 2] = 1
    spec[1][:, 3] = 1
    signals = np.array([-1, 0, 1, -1, 0, 1, -1, 0])
    out, names = H.assemble_slots(shared, spec, signals, route)
    assert names[0] == "shared_clean" and np.array_equal(out[0], shared[0])
    assert names[1] == "specialist_+0" or names[1] == "specialist_0"
    assert np.array_equal(out[1], spec[0][1])
    assert names[2] == "specialist_+1" and np.array_equal(out[2], spec[1][2])


def test_specialist_masks_partition_control_mask():
    y = torch.tensor([[0, 1, -1, 2, 3, 4, 0, 1]])
    signals = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0]])
    parts = [H.loss_mask(y, signals, "specialist", s) for s in H.SIGNALS]
    union = parts[0] | parts[1] | parts[2]
    assert torch.equal(union, H.loss_mask(y, signals, "control", None))
    assert not bool((parts[0] & parts[1]).any())
    assert not bool((parts[0] & parts[2]).any())
    assert not bool((parts[1] & parts[2]).any())
    logits = torch.randn(1, 8, 5, requires_grad=True)
    loss = H.ce_loss(logits, y, parts[0])
    loss.backward()
    grad = logits.grad.reshape(-1, 5)
    flat_mask = parts[0].reshape(-1)
    assert torch.count_nonzero(grad[~flat_mask]) == 0
    assert torch.count_nonzero(grad[flat_mask]) > 0


def test_control_and_specialist_share_train_phase():
    ep, rows, labels = _episode(cutoff=20, length=20)
    data = _data([(ep, rows, labels)])
    cut = H.build_cutoff(data, width=8, require_labels=True)
    torch.manual_seed(0)
    base = H.B.CNN(4, 1, 3, 0.0, 8, use_signal=False)
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    control = H.B.CNN(4, 1, 3, 0.0, 8, use_signal=False)
    control.load_state_dict(state)
    spec = H.B.CNN(4, 1, 3, 0.0, 8, use_signal=False)
    spec.load_state_dict(state)
    hyper = H.phase_hyper("control_finetune")
    cstats = H.train_phase(
        control, cut, epochs=1, lr=hyper["lr"], wd=hyper["wd"], bs=hyper["bs"],
        seed=0, signal=None, phase="control", device="cpu")
    sstats = H.train_phase(
        spec, cut, epochs=1, lr=hyper["lr"], wd=hyper["wd"], bs=hyper["bs"],
        seed=0, signal=-1, phase="specialist_-1", device="cpu")
    assert cstats["steps"] >= 1 and sstats["steps"] >= 1
    assert np.isfinite(cstats["losses"]).all() and np.isfinite(sstats["losses"]).all()
    # Same code path, different masks, so the weight updates are not identical.
    delta = False
    for (ck, cv), (sk, sv) in zip(control.state_dict().items(), spec.state_dict().items()):
        assert ck == sk
        if not torch.allclose(cv, sv):
            delta = True
    assert delta


def test_predict_ignores_labels_and_matches_hard_route():
    ep, rows, labels = _episode()
    data = _data([(ep, rows, labels)], labels=False)
    assert data.query_labels is None
    torch.manual_seed(0)
    shared = H.B.CNN(4, 1, 3, 0.0, 16, use_signal=False)
    specialists = {}
    for sig in H.SIGNALS:
        specialists[sig] = H.B.CNN(4, 1, 3, 0.0, 16, use_signal=False)
        # Distinct bias so the three heads cannot tie.
        with torch.no_grad():
            specialists[sig].head[-1].bias.copy_(torch.tensor([sig + 1, 0, 0, 0, 0], dtype=torch.float))
    route = {s: "specialist" for s in H.SIGNALS}
    preds, routes = H.predict_routed(
        {"shared": shared, "specialists": specialists}, route, data, width=16)
    assert len(preds) == 8 and len(routes) == 8
    for row in routes:
        assert abs(sum(preds[row["query_id"]]) - 1) < 1e-5
        assert row["model_used"] == f"specialist_{row['signal']:+d}"
    # Fallback signal uses the shared clean model, byte for byte the same slot vector.
    route_fb = {-1: "shared", 0: "specialist", 1: "specialist"}
    preds_fb, routes_fb = H.predict_routed(
        {"shared": shared, "specialists": specialists}, route_fb, data, width=16)
    single = H.predict_single(shared, data, width=16)
    for row in routes_fb:
        if row["signal"] == -1:
            assert row["model_used"] == "shared_clean"
            assert preds_fb[row["query_id"]] == single[row["query_id"]]
        else:
            assert row["model_used"] == f"specialist_{row['signal']:+d}"
            assert preds_fb[row["query_id"]] != single[row["query_id"]]


def test_train_signal_counts_and_history_causality():
    from house_edge.data import load_split
    data = load_split(os.path.join(H.REPO, "kit", "public", "train"), with_labels=True)
    assert H.require_train_labels(data) is None
    assert H.count_signal_rows(data) == {-1: 250, 0: 267, 1: 251}
    hist = H.build_history(data)
    assert hist.signals is None
    assert bool((hist.T < hist.cutoffs).all())
    assert int((hist.T >= hist.cutoffs).sum()) == 0


def test_seed_guard_and_entry_points_reject_nonzero():
    import subprocess
    import pytest
    with pytest.raises(SystemExit, match="rejected seed 1"):
        H.require_seed0(1)
    with pytest.raises(SystemExit, match="rejected seed 2"):
        H.require_seed0(2)
    H.require_seed0(0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    for script in ("hard_signal.py", "run_seed.py"):
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "..", "source", script), "--seed", "1", "--out", "/tmp/hard_signal_reject"],
            capture_output=True, text=True, env=env, cwd=H.REPO,
        )
        assert proc.returncode != 0
        assert "rejected seed 1" in (proc.stdout + proc.stderr)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "..", "source", "run_seed.py"),
         "--seed", "0", "--report-seen-confirm", "--out", "/tmp/hard_signal_reject"],
        capture_output=True, text=True, env=env, cwd=H.REPO,
    )
    assert proc.returncode != 0
    assert "rejected --report-seen-confirm" in (proc.stdout + proc.stderr)
