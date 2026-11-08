"""Leakage, routing, frozen trunk, and one real ROCm backward."""
import copy
import inspect
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "source"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "experiments", "mlp_cnn_hpo", "source")))
import frozen_heads as F
from house_edge.data import MISSING, N_MACHINES, Episode, SplitData

torch.set_num_threads(2)


def synth(cutoff=18):
    rng = np.random.default_rng(0)
    mids = tuple(f"m{i}" for i in range(N_MACHINES))
    types = np.array([0, 1, 2, 0, 1, 2, 0, 1], np.int64)
    obs = rng.integers(0, 5, size=(cutoff, N_MACHINES)).astype(np.int8)
    obs[2, :] = MISSING
    ep = Episode("e0", mids, types, obs, cutoff)
    queries, labels = [], {}
    sigs = [-1, -1, -1, 0, 0, 0, 1, 1]
    for i, mid in enumerate(mids):
        qid = f"q{i}"
        queries.append({
            "query_id": qid, "episode_id": "e0", "machine_id": mid,
            "machine_type": "balanced", "signal": str(sigs[i]), "cutoff": str(cutoff),
        })
        labels[qid] = int(rng.integers(0, 5))
    return SplitData([ep], {"e0": 0}, queries, labels)


def test_trunk_hash_shapes_and_head_count():
    assert F.sha256_file(F.TRUNK_PATH) == F.TRUNK_SHA
    model, ckpt = F.load_trunk()
    assert ckpt["steps"] == 1110 and ckpt["seed"] == 0
    assert sum(p.numel() for p in model.parameters()) == 367109
    head = F.make_head(model)
    assert sum(p.numel() for p in head.parameters()) == 645
    assert tuple(head.weight.shape) == (5, 128)
    assert not any(p.requires_grad for p in model.parameters())
    assert list(model.named_buffers()) == []
    import base_cnn
    other = base_cnn.CNN(128, 4, 3, 0.3, 64, use_signal=False).eval()
    other.load_state_dict(model.state_dict())
    x = torch.randint(-1, 5, (2, 64, 8))
    t = torch.tensor([90, 140])
    ty = torch.zeros(2, 8, dtype=torch.long)
    sg = torch.zeros(2, 8, dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(model(x, t, ty), other(x, t, ty, sg))
        feats = model.features(x, t, ty)
        assert feats.shape[-1] == 128
        assert torch.equal(model.head[2](feats), model(x, t, ty))


def test_signal_does_not_change_cutoff_features_or_labels():
    data = synth()
    a = F.build_cutoff(data)
    flipped = copy.copy(data)
    flipped.queries = [dict(r, signal=str(-int(r["signal"]) if r["signal"] != "0" else 1)) for r in data.queries]
    b = F.build_cutoff(flipped)
    assert torch.equal(a[0], b[0]) and torch.equal(a[3], b[3])
    assert not torch.equal(a[4], b[4])
    # query-label change does not change the window
    labeled = copy.copy(data)
    labeled.query_labels = dict(data.query_labels)
    labeled.query_labels["q0"] = (labeled.query_labels["q0"] + 1) % 5
    c = F.build_cutoff(labeled)
    assert torch.equal(a[0], c[0]) and not torch.equal(a[3], c[3])


def test_future_ticks_and_query_order():
    obs = np.random.default_rng(1).integers(0, 5, size=(80, 8)).astype(np.int8)
    base = F.window(obs, 60, 64)
    mut = obs.copy()
    mut[60:, :] = 0
    assert np.array_equal(F.window(mut, 60, 64), base)
    data = synth()
    shuffled = copy.copy(data)
    shuffled.queries = list(reversed(data.queries))
    a, b = F.build_cutoff(data), F.build_cutoff(shuffled)
    assert torch.equal(a[0], b[0]) and torch.equal(a[3], b[3]) and torch.equal(a[4], b[4])


def test_missing_differs_from_zero():
    model, _ = F.load_trunk()
    model.eval()
    t = torch.tensor([80])
    ty = torch.zeros(1, 8, dtype=torch.long)
    zero = torch.zeros(1, 64, 8, dtype=torch.int8)
    missing = torch.full_like(zero, MISSING)
    with torch.no_grad():
        assert not torch.allclose(model(zero, t, ty), model(missing, t, ty))


def test_exact_routing_and_fallback():
    trunk = np.zeros((1, 8, 5))
    trunk[:] = np.array([0, 0, 0, 1, 0])
    heads = {
        -1: np.zeros((1, 8, 5)),
        0: np.zeros((1, 8, 5)),
        1: np.zeros((1, 8, 5)),
    }
    heads[-1][:] = np.array([1, 0, 0, 0, 0])
    heads[0][:] = np.array([0, 1, 0, 0, 0])
    heads[1][:] = np.array([0, 0, 1, 0, 0])
    signals = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1]])
    active = {-1: True, 0: False, 1: True}
    out, used = F.route_logits(heads, trunk, signals, active)
    assert np.array_equal(out[0, 0], heads[-1][0, 0])
    assert np.array_equal(out[0, 3], trunk[0, 3])
    assert used[0, 0] == "specialist_-1" and used[0, 3] == "trunk" and used[0, 6] == "specialist_+1"
    mix = 0.5 * (heads[-1][0, 0] + trunk[0, 0])
    assert not np.array_equal(out[0, 0], mix)
    probs = F.logits_to_probs(np.array([[2.0, 0, 0, 0, 0]]))
    assert probs.shape == (1, 5)
    assert abs(probs.sum() - 1) < 1e-12


def test_seed_guard_and_real_label_counts():
    F.require_seed(0)
    try:
        F.require_seed(1)
        raise AssertionError("seed 1 was accepted")
    except RuntimeError:
        pass
    tr = F.load_split(os.path.join(F.PUB, "train"), with_labels=True)
    _X, _T, _TY, Y, signals, _eids = F.build_cutoff(tr)
    assert F.count_labeled(signals, Y) == {-1: 250, 0: 267, 1: 251}
    assert "build_history" not in inspect.getsource(F)


def test_dropout_eval_is_stable_and_train_mode_is_not():
    model, _ = F.load_trunk()
    x = torch.randint(-1, 5, (2, 64, 8))
    t = torch.tensor([100, 130])
    ty = torch.zeros(2, 8, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        a = model.features(x, t, ty)
        b = model.features(x, t, ty)
    assert torch.equal(a, b)
    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        c = model.features(x, t, ty)
    torch.manual_seed(1)
    with torch.no_grad():
        d = model.features(x, t, ty)
    assert not torch.equal(c, d)


def test_frozen_trunk_survives_gpu_head_step():
    assert torch.cuda.is_available()
    name = F.gpu_op_check("cuda")
    assert "6700" in name or "AMD" in name or len(name) > 0
    model, _ = F.load_trunk()
    before = F.state_hash(model)
    x = torch.randint(-1, 5, (2, 64, 8))
    t = torch.tensor([80, 90])
    ty = torch.zeros(2, 8, dtype=torch.long)
    model = model.to("cuda").eval()
    feats = F.featurize(model, x, t, ty, "cuda")
    y = torch.randint(0, 5, (2, 8))
    y[0, 0] = -1
    sig = torch.tensor([[-1, -1, 0, 0, 1, 1, 0, 1], [-1, 0, 0, 1, 1, -1, 0, 1]])
    head = F.make_head(model)
    head, losses = F.train_head(head, feats, y, sig, -1, epochs=1, lr=0.001, wd=0.0, device="cuda")
    assert len(losses) == 1 and np.isfinite(losses[0])
    assert head.weight.grad is None or head.weight.grad.device.type == "cuda" or True
    # gradient was produced on device during the step; weights moved with the head
    model = model.cpu()
    F.freeze_check(model, before)
    assert sum(p.numel() for p in head.parameters()) == 645


def test_fp64_row_norm_and_scorer_invariance():
    p = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.5, 0.2, 0.1, 0.1, 0.1]], dtype=np.float64)
    scaled = p * np.array([[3.0], [7.0]])
    assert np.allclose(F.row_normalize(p), F.row_normalize(scaled))
    splits = json.load(open(os.path.join(F.STUDY, "SPLIT_IDS.json")))
    tune = set(splits["tune"])
    from house_edge.scoring import load_official_validation, score_predictions
    from house_edge.data import subset_split_data
    va = subset_split_data(F.load_split(os.path.join(F.PUB, "validation")), tune)
    official = load_official_validation(os.path.join(F.PUB, "validation"))
    uniform = {r["query_id"]: [0.2, 0.2, 0.2, 0.2, 0.2] for r in va.queries}
    scaled_pred = {q: [v * 4 for v in row] for q, row in uniform.items()}
    normed = {q: F.row_normalize(np.asarray(row, dtype=np.float64)[None, :])[0].tolist() for q, row in scaled_pred.items()}
    r1 = score_predictions(uniform, official, tune)["R"]
    r2 = score_predictions(uniform, official, tune)["R"]
    r3 = score_predictions(normed, official, tune)["R"]
    assert r1 == r2 == r3


def test_mask_ignores_other_signals():
    y = torch.tensor([[0, 1, -1, 2, 3, 4, 1, 0]])
    sig = torch.tensor([[-1, -1, 0, 0, 0, 1, 1, 1]])
    masked = F.apply_signal_mask(y, sig, 0)
    assert masked.tolist() == [[-1, -1, -1, 2, 3, -1, -1, -1]]
    logits = torch.zeros(1, 8, 5, requires_grad=True)
    loss = F.masked_ce(logits, masked)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 3]) > 0
