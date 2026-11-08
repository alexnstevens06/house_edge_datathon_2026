"""Contract tests for the hard signal router. CPU only."""
import copy
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "source"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "experiments", "mlp_cnn_hpo", "source")))
import hard_signal as H
from house_edge.data import MISSING, N_MACHINES, PAYOUTS, TYPE_TO_IDX, Episode, SplitData

torch.set_num_threads(2)


def synth(cutoff=18, n_ep=1, seed=0):
    rng = np.random.default_rng(seed)
    episodes, queries, labels = [], [], {}
    sig_pattern = [-1, -1, -1, 0, 0, 0, 1, 1]
    for e in range(n_ep):
        eid = f"e{e}"
        mids = tuple(f"m{i}" for i in range(N_MACHINES))
        types = np.array([0, 1, 2, 0, 1, 2, 0, 1], np.int64)
        obs = rng.integers(0, 5, size=(cutoff, N_MACHINES)).astype(np.int8)
        obs[3, :] = MISSING
        ep = Episode(eid, mids, types, obs, cutoff)
        for i, mid in enumerate(mids):
            qid = f"q{e}-{i}"
            queries.append({
                "query_id": qid,
                "episode_id": eid,
                "machine_id": mid,
                "machine_type": ["balanced", "jackpot", "steady"][int(types[i])],
                "signal": str(sig_pattern[i]),
                "cutoff": str(cutoff),
            })
            labels[qid] = int(rng.integers(0, 5))
        episodes.append(ep)
    index = {ep.episode_id: i for i, ep in enumerate(episodes)}
    return SplitData(episodes, index, queries, labels)


class Const(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x, t, ty, sg):
        return self.logits.view(1, 1, 5).expand(x.shape[0], N_MACHINES, 5).clone()


class Boom(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("inactive specialist was called")


def test_type_and_payout_maps():
    assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}
    assert PAYOUTS == (0, 1, 2, 5, 20)
    assert MISSING == -1 and MISSING != 0


def test_architecture_matches_cnn03_nosignal():
    import base_cnn
    a = H.new_cnn03().eval()
    b = base_cnn.CNN(128, 4, 3, 0.3, 64, use_signal=False).eval()
    ka = [(n, tuple(p.shape)) for n, p in a.named_parameters()]
    kb = [(n, tuple(p.shape)) for n, p in b.named_parameters()]
    assert ka == kb
    assert sum(p.numel() for p in a.parameters()) == 367109
    assert not any("sig" in n for n, _ in ka)
    b.load_state_dict(a.state_dict())
    x = torch.randint(-1, 5, (2, 64, 8))
    t = torch.tensor([100, 180])
    ty = torch.zeros(2, 8, dtype=torch.long)
    sg0 = torch.zeros(2, 8, dtype=torch.long)
    sg1 = torch.full((2, 8), 2)
    with torch.no_grad():
        assert torch.allclose(a(x, t, ty, sg0), b(x, t, ty, sg0))
        assert torch.allclose(a(x, t, ty, sg0), a(x, t, ty, sg1))


def test_future_mutation_all_eight_machines():
    obs = np.random.default_rng(1).integers(0, 5, size=(200, 8)).astype(np.int8)
    t, W = 150, 64
    base = H.window(obs, t, W)
    mut = obs.copy()
    mut[t:, :] = (mut[t:, :] + 3) % 5
    assert np.array_equal(H.window(mut, t, W), base)
    # ticks strictly before t still matter, so the test is not vacuous
    touched = obs.copy()
    touched[t - 1, :] = (touched[t - 1, :] + 1) % 5
    assert not np.array_equal(H.window(touched, t, W), base)
    m = H.new_cnn03().eval()
    z = torch.randn(1, 8, W, m.pay.embedding_dim)
    z2 = z.clone()
    z2[:, :, W // 2:, :] += 5

    def run(z):
        for b in m.blocks:
            z = b(z)
        return z

    with torch.no_grad():
        assert torch.allclose(run(z)[:, :, :W // 2], run(z2)[:, :, :W // 2], atol=1e-5)


def test_label_mutation_changes_targets_not_features():
    data = synth()
    hist = H.build_history(data)
    cut = H.build_cutoff(data)
    mutated = copy.copy(data)
    mutated.query_labels = dict(data.query_labels)
    mutated.query_labels["q0-0"] = (mutated.query_labels["q0-0"] + 1) % 5
    hist2 = H.build_history(mutated)
    cut2 = H.build_cutoff(mutated)
    assert torch.equal(hist.X, hist2.X) and torch.equal(hist.Y, hist2.Y)
    assert torch.equal(cut.X, cut2.X)
    assert not torch.equal(cut.Y, cut2.Y)
    # history label at the prediction tick changes Y only, not features of that window
    data2 = synth()
    ep = data2.episodes[0]
    t = 17
    before = H.build_history(data2)
    ep.obs = ep.obs.copy()
    ep.obs[t, 3] = (int(ep.obs[t, 3]) + 1) % 5
    after = H.build_history(data2)
    rows = (before.T == t).nonzero().flatten()
    assert len(rows) == 1
    i = int(rows[0])
    assert torch.equal(before.X[i], after.X[i])
    assert not torch.equal(before.Y[i], after.Y[i])
    earlier = before.T < t
    assert torch.equal(before.X[earlier], after.X[earlier])


def test_signal_mutation_does_not_change_historical_features():
    data = synth()
    flipped = copy.copy(data)
    flipped.queries = []
    for r in data.queries:
        sig = int(r["signal"])
        flipped.queries.append(dict(r, signal=str(-sig if sig != 0 else 1)))
    a, b = H.build_history(data), H.build_history(flipped)
    assert torch.equal(a.X, b.X) and torch.equal(a.Y, b.Y) and torch.equal(a.T, b.T)
    ca, cb = H.build_cutoff(data), H.build_cutoff(flipped)
    assert torch.equal(ca.X, cb.X) and torch.equal(ca.Y, cb.Y)
    assert not torch.equal(ca.signals, cb.signals)
    # history builder has no signal field
    assert not hasattr(a, "signals")


def test_query_slot_order_invariance():
    data = synth()
    shuffled = copy.copy(data)
    shuffled.queries = list(reversed(data.queries))
    a, b = H.build_cutoff(data), H.build_cutoff(shuffled)
    assert torch.equal(a.X, b.X) and torch.equal(a.Y, b.Y) and torch.equal(a.signals, b.signals)
    ha, hb = H.build_history(data), H.build_history(shuffled)
    assert torch.equal(ha.X, hb.X) and torch.equal(ha.Y, hb.Y)
    ev = H.build_eval(data)
    active = {-1: True, 0: True, 1: True}
    specs = {-1: Const([3.0, 0, 0, 0, 0]), 0: Const([0, 3.0, 0, 0, 0]), 1: Const([0, 0, 3.0, 0, 0])}
    P, used = H.predict_routed(specs, Const([0, 0, 0, 3.0, 0]), active, ev.X, ev.T, ev.TY, ev.signals)
    by_id = {ep.episode_id: ep for ep in data.episodes}
    p1 = H.to_preds(P, ev.episode_ids, by_id, data)
    p2 = H.to_preds(P, ev.episode_ids, by_id, shuffled)
    assert p1.keys() == p2.keys()
    for q in p1:
        assert p1[q] == p2[q]
    # reversed query list still routes each machine to its own signal
    assert used[0, 0] == "specialist_-1"
    assert used[0, 3] == "specialist_0"
    assert used[0, 6] == "specialist_+1"


def test_missing_distinguished_from_zero():
    m = H.new_cnn03().eval()
    t = torch.tensor([80])
    ty = torch.zeros(1, 8, dtype=torch.long)
    sg = torch.zeros(1, 8, dtype=torch.long)
    zero = torch.zeros(1, 64, 8, dtype=torch.int8)
    missing = torch.full((1, 64, 8), MISSING, dtype=torch.int8)
    with torch.no_grad():
        assert not torch.allclose(m(zero, t, ty, sg), m(missing, t, ty, sg))
    obs = np.zeros((90, 8), np.int8)
    miss = obs.copy()
    miss[40:70, 2] = MISSING
    assert not np.array_equal(H.window(obs, 90, 64), H.window(miss, 90, 64))


def test_hard_router_exact_match_and_normalized():
    data = synth()
    ev = H.build_eval(data)
    specs = {
        -1: Const([2.0, 0, 0, 0, 0]),
        0: Const([0, 2.0, 0, 0, 0]),
        1: Const([0, 0, 2.0, 0, 0]),
    }
    shared = Const([0, 0, 0, 2.0, 0])
    active = {-1: True, 0: False, 1: True}
    # inactive specialist must not be called; pass a trap in its place
    specs_call = {-1: specs[-1], 0: Boom(), 1: specs[1]}
    P, used = H.predict_routed(specs_call, shared, active, ev.X, ev.T, ev.TY, ev.signals)
    expect = {
        -1: torch.softmax(torch.tensor([2.0, 0, 0, 0, 0]), -1).numpy(),
        0: torch.softmax(torch.tensor([0, 2.0, 0, 0, 0]), -1).numpy(),
        1: torch.softmax(torch.tensor([0, 0, 2.0, 0, 0]), -1).numpy(),
        "shared": torch.softmax(torch.tensor([0, 0, 0, 2.0, 0]), -1).numpy(),
    }
    signals = ev.signals[0].tolist()
    for slot, sig in enumerate(signals):
        if active[sig]:
            assert used[0, slot] == H.specialist_name(sig)
            assert np.allclose(P[0, slot], expect[sig])
            mix = 0.5 * (expect[sig] + expect["shared"])
            assert not np.allclose(P[0, slot], mix)
        else:
            assert used[0, slot] == "shared"
            assert np.allclose(P[0, slot], expect["shared"])
    assert np.allclose(P.sum(-1), 1.0, atol=1e-8)
    # control path has no signal argument
    assert "signals" not in inspect.signature(H.predict_all).parameters
    ctrl = H.predict_all(shared, ev.X, ev.T, ev.TY)
    assert np.allclose(ctrl.sum(-1), 1.0, atol=1e-8)
    assert np.allclose(ctrl[0, 0], expect["shared"])


def test_sparse_group_fallback():
    counts = {-1: 250, 0: 31, 1: 251}
    assert H.active_map(counts, 32) == {-1: True, 0: False, 1: True}
    assert H.active_map({-1: 32, 0: 32, 1: 32}, 32) == {-1: True, 0: True, 1: True}
    y = torch.tensor([[0, 1, -1, 2, 3, -1, 4, 0]])
    sig = torch.tensor([[-1, -1, 0, 0, 0, 1, 1, 1]])
    got = H.count_labeled(sig, y)
    assert got == {-1: 2, 0: 2, 1: 2}
    masked = H.apply_signal_mask(y, sig, 0)
    assert masked.tolist() == [[-1, -1, -1, 2, 3, -1, -1, -1]]
    assert torch.equal(H.apply_signal_mask(y, sig, None), y)


def test_shared_control_same_pipeline():
    assert H.CFG == {
        "h": 128, "layers": 4, "k": 3, "drop": 0.3, "W": 64,
        "lr": 0.00197, "wd": 0.00895, "bs": 512, "micro_bs": 32,
        "pretrain_epochs": 30, "finetune_epochs": 30, "min_t": 16, "stride": 1,
        "min_group_rows": 32, "grad_clip": 1.0, "use_signal": False,
    }
    params = inspect.signature(H.fit).parameters
    assert "val" not in params and "validation" not in params
    src = inspect.getsource(H.fit)
    assert "score_predictions" not in src and "load_official" not in src
    data = synth()
    hist, cut = H.build_history(data), H.build_cutoff(data)
    assert len(hist.T) and int(hist.T.max()) < data.episodes[0].cutoff
    assert torch.all(cut.T == data.episodes[0].cutoff)
    # zero-epoch copies: control and specialists start at the shared snapshot
    result = H.fit(data, seed=0, pretrain_epochs=0, finetune_epochs=0, min_group_rows=1, device="cpu")
    assert result["n_cutoff"] == 1
    assert all(result["active"].values())
    snap = result["shared"].state_dict()
    assert list(result["shared"].state_dict()) == list(result["control"].state_dict())
    for key, value in snap.items():
        assert torch.equal(value, result["control"].state_dict()[key])
        for s in (-1, 0, 1):
            assert torch.equal(value, result["specialists"][s].state_dict()[key])
    # one real step on the same windows: identical step counts, weights move apart
    trained = H.fit(data, seed=0, pretrain_epochs=1, finetune_epochs=1, min_group_rows=1, bs=2, micro_bs=1, device="cpu")
    assert trained["steps"]["control"] == trained["steps"]["specialist"][-1] == trained["steps"]["specialist"][0] == trained["steps"]["specialist"][1]
    assert trained["steps"]["control"] == 1
    assert not torch.equal(trained["control"].state_dict()["head.0.weight"], trained["shared"].state_dict()["head.0.weight"])
    assert not torch.equal(trained["specialists"][-1].state_dict()["head.0.weight"], trained["shared"].state_dict()["head.0.weight"])
    assert not torch.equal(trained["specialists"][-1].state_dict()["head.0.weight"], trained["control"].state_dict()["head.0.weight"])


def test_microbatch_grad_matches_full_batch():
    torch.manual_seed(0)
    lg = torch.randn(6, 8, 5, requires_grad=True)
    y = torch.randint(0, 5, (6, 8))
    y[0, :] = -1
    y[1, 0] = -1
    ce, n = H.masked_ce_sum(lg, y)
    assert n == int((y >= 0).sum())
    (ce / n).backward()
    full = lg.grad.detach().clone()
    lg2 = lg.detach().clone().requires_grad_(True)
    for sl in (slice(0, 2), slice(2, 6)):
        part, _ = H.masked_ce_sum(lg2[sl], y[sl])
        (part / n).backward()
    assert torch.allclose(lg2.grad, full, atol=1e-6)


def test_history_excludes_cutoff_and_query_labels():
    data = synth(cutoff=20)
    # plant a query label that does not appear as a history payout index pattern
    data.query_labels = {k: 4 for k in data.query_labels}
    hist = H.build_history(data)
    cut = H.build_cutoff(data)
    assert int(hist.T.max()) < 20
    assert set(cut.T.tolist()) == {20}
    assert torch.all(cut.Y == 4)
    assert not torch.all(hist.Y == 4)
    # deleting labels does not change history
    bare = copy.copy(data)
    bare.query_labels = None
    hist2 = H.build_history(bare)
    assert torch.equal(hist.X, hist2.X) and torch.equal(hist.Y, hist2.Y)


def test_eval_y_is_missing_and_features_ignore_signal():
    data = synth()
    ev = H.build_eval(data)
    assert torch.all(ev.Y < 0)
    assert torch.equal(ev.X, H.build_cutoff(data).X)
    flipped = copy.copy(data)
    flipped.queries = [dict(r, signal="1") for r in data.queries]
    ev2 = H.build_eval(flipped)
    assert torch.equal(ev.X, ev2.X)
    assert not torch.equal(ev.signals, ev2.signals)
