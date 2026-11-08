"""Contract tests for soft_moe. CPU only. No training metrics."""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "source"))
import soft_moe as S
from house_edge.data import MISSING, N_MACHINES, episode_conditions, load_split

torch.set_num_threads(2)
PUB_TRAIN = os.path.join(S.PUB, "train")
PUB_VAL = os.path.join(S.PUB, "validation")
TR = load_split(PUB_TRAIN, with_labels=True)
VA = load_split(PUB_VAL)


def _subset(data, episode_ids):
    kept = set(episode_ids)
    out = copy.copy(data)
    out.episodes = [ep for ep in data.episodes if ep.episode_id in kept]
    out.episode_index = {ep.episode_id: i for i, ep in enumerate(out.episodes)}
    out.queries = [row for row in data.queries if row["episode_id"] in kept]
    if data.query_labels is not None:
        ids = {row["query_id"] for row in out.queries}
        out.query_labels = {key: value for key, value in data.query_labels.items() if key in ids}
    return out


def _tiny():
    torch.manual_seed(0)
    kwargs = dict(h=16, layers=2, drop=0.0, W=16)
    return S.SoftMoE(**kwargs).eval(), S.ControlCNN(**kwargs).eval()


def test_future_mutation_all_eight_machines():
    conds = episode_conditions(PUB_VAL)
    episode = next(ep for ep in VA.episodes if conds[ep.episode_id] == "short_history")
    assert (episode.obs < 0).any()
    model, _control = _tiny()
    w = 16
    full = np.full((episode.cutoff + 1, N_MACHINES), MISSING, np.int8)
    full[:episode.obs.shape[0]] = episode.obs
    ty = torch.tensor(episode.machine_types)[None]
    sg = torch.zeros(1, N_MACHINES, dtype=torch.long)
    checked = 0
    for t in (16, max(16, episode.cutoff // 2), episode.cutoff):
        base = torch.tensor(S.window(full, t, w))[None]
        tick = torch.tensor([t])
        cut = torch.tensor([t == episode.cutoff])
        with torch.no_grad():
            p0, _g = model(base, tick, ty, sg, cut)
        mutated = full.copy()
        mutated[t:, :] = (np.clip(mutated[t:, :], 0, 4) + 1) % 5
        w2 = torch.tensor(S.window(mutated, t, w))[None]
        assert torch.equal(w2, base)
        with torch.no_grad():
            p1, _g = model(w2, tick, ty, sg, cut)
        assert torch.equal(p1, p0)
        for slot in range(N_MACHINES):
            one = full.copy()
            one[t:, slot] = 3
            assert torch.equal(torch.tensor(S.window(one, t, w))[None], base)
            checked += 1
    assert checked == 3 * N_MACHINES
    # In-window tick is visible: changing it must change the window.
    seen = full.copy()
    seen[episode.cutoff - 1, 0] = 0 if full[episode.cutoff - 1, 0] != 0 else 1
    assert not torch.equal(torch.tensor(S.window(seen, episode.cutoff, w)), torch.tensor(S.window(full, episode.cutoff, w)))


def test_label_mutation_does_not_change_features():
    ids = [ep.episode_id for ep in TR.episodes[:2]]
    sub = _subset(TR, ids)
    mutated = copy.copy(sub)
    mutated.query_labels = {key: (value + 1) % 5 for key, value in sub.query_labels.items()}
    hist_a, _ = S.build_split_windows(sub, 32, "history")
    hist_b, _ = S.build_split_windows(mutated, 32, "history")
    for left, right in zip(hist_a, hist_b):
        assert torch.equal(left, right)
    cut_a, _ = S.build_split_windows(sub, 32, "cutoff")
    cut_b, _ = S.build_split_windows(mutated, 32, "cutoff")
    # X, T, TY, SG, CUT unchanged. Only Y changes, and only on labelled slots.
    for i in (0, 1, 2, 3, 5):
        assert torch.equal(cut_a[i], cut_b[i])
    assert not torch.equal(cut_a[4], cut_b[4])
    assert torch.equal((cut_a[4] + 1) % 5, cut_b[4])


def test_signal_mutation_does_not_change_historical_features():
    ids = [ep.episode_id for ep in TR.episodes[:3]]
    sub = _subset(TR, ids)
    flipped = copy.copy(sub)
    flipped.queries = []
    for row in sub.queries:
        nxt = dict(row)
        sig = int(row["signal"])
        nxt["signal"] = str(1 if sig == 0 else -sig)
        flipped.queries.append(nxt)
    hist_a, _ = S.build_split_windows(sub, 32, "history")
    hist_b, _ = S.build_split_windows(flipped, 32, "history")
    for left, right in zip(hist_a, hist_b):
        assert torch.equal(left, right)
    assert int((hist_a[3] != S.ABSENT).sum()) == 0
    assert not bool(hist_a[5].any())
    cut_a, _ = S.build_split_windows(sub, 32, "cutoff")
    cut_b, _ = S.build_split_windows(flipped, 32, "cutoff")
    assert not torch.equal(cut_a[3], cut_b[3])  # cutoff signal channel does change
    model, control = _tiny()
    x = hist_a[0][:2]
    t = hist_a[1][:2]
    ty = hist_a[2][:2]
    ysg = torch.randint(0, 3, (2, N_MACHINES))
    cut = torch.zeros(2, dtype=torch.bool)
    with torch.no_grad():
        p0, g0 = model(x, t, ty, torch.full_like(ysg, S.ABSENT), cut)
        p1, g1 = model(x, t, ty, ysg, cut)
        c0, _ = control(x, t, ty, torch.zeros_like(ysg), cut)
        c1, _ = control(x, t, ty, ysg, torch.ones(2, dtype=torch.bool))
    assert torch.equal(p0, p1) and torch.equal(g0, g1)
    assert torch.equal(c0, c1)


def test_query_slot_order_invariance():
    ids = [ep.episode_id for ep in TR.episodes[:2]]
    sub = _subset(TR, ids)
    for ep in sub.episodes:
        assert list(ep.machine_ids) == sorted(ep.machine_ids)
    reversed_queries = copy.copy(sub)
    reversed_queries.queries = list(reversed(sub.queries))
    a, eps_a = S.build_split_windows(sub, 32, "cutoff")
    b, eps_b = S.build_split_windows(reversed_queries, 32, "cutoff")
    for left, right in zip(a, b):
        assert torch.equal(left, right)
    assert [ep.episode_id for ep in eps_a] == [ep.episode_id for ep in eps_b]
    # Slot follows machine id, not the position of the row in queries.csv.
    ep = sub.episodes[0]
    rows = [row for row in sub.queries if row["episode_id"] == ep.episode_id]
    assert ep.slot_of(rows[-1]["machine_id"]) == ep.machine_ids.index(rows[-1]["machine_id"])


def test_missing_distinguished_from_zero():
    model, _control = _tiny()
    w = 16
    missing = np.full((w, N_MACHINES), MISSING, np.int8)
    zeros = np.zeros((w, N_MACHINES), np.int8)
    ty = torch.zeros(1, N_MACHINES, dtype=torch.long)
    sg = torch.full((1, N_MACHINES), S.ABSENT)
    tick = torch.tensor([w])
    cut = torch.tensor([False])
    with torch.no_grad():
        pm, _ = model(torch.tensor(missing)[None], tick, ty, sg, cut)
        pz, _ = model(torch.tensor(zeros)[None], tick, ty, sg, cut)
    assert not torch.allclose(pm, pz)


def test_router_probabilities_normalized():
    model, _control = _tiny()
    x = torch.randint(-1, 5, (4, 16, N_MACHINES))
    t = torch.tensor([20, 40, 60, 80])
    ty = torch.randint(0, 3, (4, N_MACHINES))
    sg = torch.randint(0, 3, (4, N_MACHINES))
    cut = torch.tensor([False, False, True, True])
    with torch.no_grad():
        probs, gate = model(x, t, ty, sg, cut)
    assert probs.shape == (4, N_MACHINES, 5)
    assert gate.shape == (4, N_MACHINES, 3)
    assert torch.allclose(gate.sum(-1), torch.ones(4, N_MACHINES), atol=1e-5)
    assert torch.all(gate >= 0)
    assert torch.allclose(probs.sum(-1), torch.ones(4, N_MACHINES), atol=1e-5)
    assert torch.all(probs >= 0)


def test_sparse_group_fallback_matches_shared_head():
    model, _control = _tiny()
    x = torch.randint(-1, 5, (2, 16, N_MACHINES))
    t = torch.tensor([30, 50])
    ty = torch.randint(0, 3, (2, N_MACHINES))
    sg = torch.randint(0, 3, (2, N_MACHINES))
    cut = torch.tensor([True, False])
    keep = torch.zeros(S.EXPERTS, dtype=torch.bool)
    with torch.no_grad():
        probs, gate = model(x, t, ty, sg, cut, expert_keep=keep)
        shared = torch.softmax(model.head(*S._last_pool(model.enc(x, t, ty))).float(), dim=-1)
    assert torch.allclose(probs, shared, atol=1e-6)
    assert torch.allclose(gate.sum(-1), torch.ones(2, N_MACHINES), atol=1e-5)
    # A single kept expert renormalizes to that expert's probability.
    with torch.no_grad():
        # Make residuals differ so the blend is not trivially the base.
        model.experts[0].head[-1].bias.data[0] = 4.0
        model.experts[1].head[-1].bias.data[1] = -3.0
        only = torch.tensor([True, False, False])
        p_only, _g = model(x, t, ty, sg, cut, expert_keep=only)
        z = model.enc(x, t, ty)
        last, pooled = S._last_pool(z)
        base = model.head(last, pooled)
        expect = torch.softmax((base + model.experts[0](z)).float(), dim=-1)
    assert torch.allclose(p_only, expect, atol=1e-5)


def test_shared_control_uses_same_pipeline():
    ids = [TR.episodes[0].episode_id]
    sub = _subset(TR, ids)
    hist, _ = S.build_split_windows(sub, 16, "history")
    cut, _ = S.build_split_windows(sub, 16, "cutoff")
    S.assert_history_has_no_signal(hist)
    S.assert_cutoff_rows(cut, sub)
    control = S.build_model("control", h=16, layers=1, drop=0.0, W=16)
    soft = S.build_model("soft", h=16, layers=1, drop=0.0, W=16)
    # Same trainer, same tensors, one epoch on the tiny subset (CPU).
    loss_c = S.train_phase(control, hist, epochs=1, lr=1e-3, wd=0.0, bs=8, device="cpu")
    loss_s = S.train_phase(soft, hist, epochs=1, lr=1e-3, wd=0.0, bs=8, device="cpu")
    assert len(loss_c) == 1 and len(loss_s) == 1 and np.isfinite(loss_c[0]) and np.isfinite(loss_s[0])
    ft_c = S.train_phase(control, cut, epochs=1, lr=1e-3, wd=0.0, bs=8, device="cpu")
    ft_s = S.train_phase(soft, cut, epochs=1, lr=1e-3, wd=0.0, bs=8, device="cpu")
    assert len(ft_c) == len(ft_s) == 1
    # Control has no signal embedding. Soft gate does, and ignores it off cutoff.
    assert not any("sig" in name for name, _param in control.named_parameters())
    assert any("sig" in name for name, _param in soft.named_parameters())
    x, t, ty = cut[0][:1], cut[1][:1], cut[2][:1]
    sg0 = torch.zeros(1, N_MACHINES, dtype=torch.long)
    sg1 = torch.full((1, N_MACHINES), 2)
    off = torch.zeros(1, dtype=torch.bool)
    on = torch.ones(1, dtype=torch.bool)
    control.eval()
    soft.eval()
    with torch.no_grad():
        c0, _ = control(x, t, ty, sg0, on)
        c1, _ = control(x, t, ty, sg1, on)
        s0, g0 = soft(x, t, ty, sg0, off)
        s1, g1 = soft(x, t, ty, sg1, off)
    assert torch.equal(c0, c1)
    assert torch.equal(s0, s1) and torch.equal(g0, g1)


def test_masked_nll_ignores_missing():
    probs = torch.softmax(torch.randn(4, 8, 5), dim=-1)
    y = torch.full((4, 8), -1)
    y[0, 0] = 2
    y[1, 3] = 0
    p2 = probs.detach().clone()
    p2[0, 1:] = torch.softmax(torch.randn(7, 5), dim=-1)
    p2[2:] = torch.softmax(torch.randn(2, 8, 5), dim=-1)
    # Missing slots' probabilities do not change the loss when labelled slots match.
    p2 = p2.clone()
    p2[0, 0] = probs[0, 0]
    p2[1, 3] = probs[1, 3]
    assert torch.allclose(S.masked_nll(probs, y), S.masked_nll(p2, y))
    assert float(S.masked_nll(torch.softmax(torch.randn(2, 8, 5), -1), torch.full((2, 8), -1))) == 0.0
