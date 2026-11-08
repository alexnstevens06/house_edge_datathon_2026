"""Leakage and architecture tests for soft_moe. CPU only."""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "source"))
import soft_moe as S
from house_edge.data import MISSING, load_split

torch.set_num_threads(2)
TR = load_split(os.path.join(S.PUB, "train"), with_labels=True)
SUB = copy.copy(TR)
SUB.episodes = TR.episodes[:3]
SUB.episode_index = {e.episode_id: i for i, e in enumerate(SUB.episodes)}
SUB.queries = [r for r in TR.queries if r["episode_id"] in SUB.episode_index]
SUB.query_labels = {r["query_id"]: TR.query_labels[r["query_id"]] for r in SUB.queries}


def _flip_signals(data):
    out = copy.copy(data)
    flipped = []
    for row in data.queries:
        sig = int(row["signal"])
        nxt = 1 if sig == 0 else -sig
        flipped.append(dict(row, signal=str(nxt)))
    out.queries = flipped
    return out


def test_future_mutation_all_eight_machines():
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 5, (90, 8)).astype(np.int8)
    t, W = 60, 32
    base = S.window(obs, t, W)
    for slot in range(8):
        mutated = obs.copy()
        mutated[t:, slot] = (mutated[t:, slot] + 3) % 5
        assert np.array_equal(S.window(mutated, t, W), base)


def test_label_mutation_does_not_change_features():
    base = S.build_train(SUB, 32)
    mutated = copy.copy(SUB)
    mutated.query_labels = dict(SUB.query_labels)
    ep = SUB.episodes[0]
    qids = [r["query_id"] for r in SUB.queries if r["episode_id"] == ep.episode_id]
    assert len(qids) == 8
    for qid in qids:
        mutated.query_labels[qid] = (mutated.query_labels[qid] + 1) % 5
    other = S.build_train(mutated, 32)
    assert torch.equal(base["X"], other["X"])
    assert torch.equal(base["T"], other["T"])
    assert torch.equal(base["TY"], other["TY"])
    assert not torch.equal(base["Y"], other["Y"])
    # A history label at the prediction tick, on every machine, is a target, not a feature.
    obs = ep.obs.copy()
    t = 40
    before = S.window(obs, t, 32)
    obs = obs.copy()
    for slot in range(8):
        obs[t, slot] = (obs[t, slot] + 1) % 5
    assert np.array_equal(S.window(obs, t, 32), before)


def test_signal_mutation_does_not_change_historical_features():
    base = S.build_train(SUB, 32)
    other = S.build_train(_flip_signals(SUB), 32)
    assert torch.equal(base["X"], other["X"])
    assert torch.equal(base["T"], other["T"])
    assert torch.equal(base["TY"], other["TY"])
    hist = ~base["C"]
    assert hist.any()
    assert torch.equal(base["SG"][hist], other["SG"][hist])
    assert int((base["SG"][hist] != S.ABSENT).sum()) == 0
    assert not torch.equal(base["SG"][base["C"]], other["SG"][other["C"]])
    wiped = S.signal_for_phase(base["SG"], base["C"], "absent")
    assert int((wiped != S.ABSENT).sum()) == 0
    kept = S.signal_for_phase(base["SG"], base["C"], "cutoff")
    assert torch.equal(kept[hist], torch.full_like(kept[hist], S.ABSENT))
    assert torch.equal(kept[base["C"]], base["SG"][base["C"]])


def test_query_slot_order_invariance():
    rng = np.random.default_rng(1)
    shuffled = copy.copy(SUB)
    order = list(SUB.queries)
    rng.shuffle(order)
    shuffled.queries = order
    base, b_eps = S.build_query(SUB, 32)
    other, o_eps = S.build_query(shuffled, 32)
    for a, b in zip(base, other):
        assert torch.equal(a, b)
    assert [e.episode_id for e in b_eps] == [e.episode_id for e in o_eps]
    train_a = S.build_train(SUB, 32)
    train_b = S.build_train(shuffled, 32)
    for key in ("X", "T", "TY", "SG", "Y", "C"):
        assert torch.equal(train_a[key], train_b[key])


def test_missing_distinguished_from_zero():
    model = S.SharedCNN(h=16, layers=1, drop=0.0, W=8).eval()
    missing = torch.full((2, 8, 8), MISSING)
    zeros = torch.zeros(2, 8, 8)
    t = torch.tensor([30, 40])
    ty = torch.zeros(2, 8, dtype=torch.long)
    sg = torch.full((2, 8), S.ABSENT)
    with torch.no_grad():
        assert not torch.allclose(model(missing, t, ty, sg), model(zeros, t, ty, sg))
    moe = S.SoftMoE(h=16, layers=1, drop=0.0, W=8, expert_e=8).eval()
    with torch.no_grad():
        assert not torch.allclose(moe(missing, t, ty, sg), moe(zeros, t, ty, sg))


def test_router_probabilities_normalized_and_zero_residual_matches_shared():
    moe = S.SoftMoE(h=16, layers=1, drop=0.0, W=8, expert_e=8).eval()
    x = torch.randint(-1, 5, (4, 8, 8))
    t = torch.tensor([20, 21, 22, 23])
    ty = torch.randint(0, 3, (4, 8))
    sg = torch.full((4, 8), S.ABSENT)
    with torch.no_grad():
        probs = moe(x, t, ty, sg)
        logits, feat = moe.trunk_logits(x, t, ty)
        gate = moe.mask_gate(moe.gate(feat, ty, sg))
    assert torch.allclose(probs.sum(-1), torch.ones(4, 8), atol=1e-5)
    assert torch.allclose(gate.sum(-1), torch.ones(4, 8), atol=1e-5)
    assert torch.allclose(probs, torch.softmax(logits.float(), -1), atol=1e-5)
    # Real signal at a cutoff row may change the gate, and rows stay normalized.
    sg2 = sg.clone()
    sg2[:, 0] = 0
    with torch.no_grad():
        probs2 = moe(x, t, ty, sg2)
    assert torch.allclose(probs2.sum(-1), torch.ones(4, 8), atol=1e-5)


def test_exact_blend_matches_hand_combination():
    shared = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0]]])
    residuals = torch.zeros(3, 1, 1, 5)
    residuals[0, 0, 0, 1] = 5.0
    residuals[1, 0, 0, 2] = 4.0
    residuals[2, 0, 0, 3] = 3.0
    gate = torch.tensor([[[0.2, 0.5, 0.3]]])
    mixed = S.blend_probs(shared, residuals, gate)
    expert = torch.softmax(shared.unsqueeze(0) + residuals, dim=-1)
    hand = (gate.permute(2, 0, 1).unsqueeze(-1) * expert).sum(0)
    assert torch.allclose(mixed, hand, atol=1e-6)
    assert torch.allclose(mixed.sum(-1), torch.ones(1, 1), atol=1e-6)
    assert torch.allclose(S.blend_probs(shared, residuals, None), torch.softmax(shared.float(), -1))


def test_sparse_group_fallback_matches_shared_head():
    torch.manual_seed(0)
    moe = S.SoftMoE(h=16, layers=1, drop=0.0, W=8, expert_e=8).eval()
    with torch.no_grad():
        moe.experts[0].out.weight.normal_()
        moe.experts[1].out.bias.normal_()
    x = torch.randint(-1, 5, (3, 8, 8))
    t = torch.tensor([16, 17, 18])
    ty = torch.zeros(3, 8, dtype=torch.long)
    sg = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]]).expand(3, 8).contiguous()
    moe.expert_active[:] = False
    with torch.no_grad():
        probs = moe(x, t, ty, sg)
        logits, _ = moe.trunk_logits(x, t, ty)
    assert torch.allclose(probs, torch.softmax(logits.float(), -1), atol=1e-5)
    moe.expert_active[:] = torch.tensor([True, False, True])
    _, feat = moe.trunk.encode(x, t, ty)
    with torch.no_grad():
        gate = moe.mask_gate(moe.gate(feat, ty, sg))
    assert torch.all(gate[:, :, 1] == 0)
    assert torch.allclose(gate.sum(-1), torch.ones(3, 8), atol=1e-5)
    # Counts below the frozen minimum disable that expert.
    fresh = S.SoftMoE(h=16, layers=1, drop=0.0, W=8, expert_e=4)
    fresh.set_active_from_counts([100, 3, 40], S.PIPELINE["sparse_min"])
    assert fresh.expert_active.tolist() == [True, False, True]


def test_shared_control_same_pipeline_and_ignores_signal():
    p = S.PIPELINE
    assert (p["h"], p["layers"], p["W"], p["drop"], p["epochs"], p["ft_epochs"], p["bs"], p["lr"], p["wd"]) == (
        128, 4, 64, 0.3, 30, 10, 512, 0.00197, 0.00895)
    pack = S.build_train(SUB, 16)
    x, t, ty, sg = pack["X"][:4], pack["T"][:4], pack["TY"][:4], pack["SG"][:4]
    control = S.SharedCNN(h=16, layers=1, drop=0.0, W=16).eval()
    moe = S.SoftMoE(h=16, layers=1, drop=0.0, W=16, expert_e=8).eval()
    assert not any("sig" in name for name, _ in control.named_parameters())
    with torch.no_grad():
        a = control(x, t, ty, sg)
        b = control(x, t, ty, torch.zeros_like(sg))
        c = moe(x, t, ty, torch.full_like(sg, S.ABSENT))
    assert torch.equal(a, b)
    assert a.shape == c.shape == (4, 8, 5)
    full = S.SharedCNN()
    assert sum(param.numel() for param in full.parameters()) == 367109


def test_type_and_payout_contract():
    assert S.TYPE_TO_IDX["balanced"] == 0
    assert S.TYPE_TO_IDX["jackpot"] == 1
    assert S.TYPE_TO_IDX["steady"] == 2
    assert S.PAYOUTS == (0, 1, 2, 5, 20)


def test_seed_guard_rejects_nonzero():
    S.reject_seed(0)
    for bad in (1, 2, -1, 3):
        try:
            S.reject_seed(bad)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"seed {bad} was accepted")
    script = os.path.join(HERE, "..", "source", "launch_seed0.sh")
    import subprocess
    done = subprocess.run(["bash", script, "1"], capture_output=True, text=True)
    assert done.returncode == 2
    assert "REJECT" in done.stdout
    done2 = subprocess.run(["bash", script, "2"], capture_output=True, text=True)
    assert done2.returncode == 2


def test_train_windows_match_cnn03_and_signal_counts():
    sys.path.insert(0, os.path.join(S.REPO, "experiments", "mlp_cnn_hpo", "source"))
    import base_cnn as B
    B.USE_SIGNAL = False
    theirs = B.build_train(TR, 64)
    ours = S.build_train(TR, 64)
    assert torch.equal(ours["X"], theirs[0])
    assert torch.equal(ours["T"], theirs[1])
    assert torch.equal(ours["TY"], theirs[2])
    assert torch.equal(ours["Y"], theirs[4])
    assert tuple(int(c) for c in ours["signal_counts"].tolist()) == S.EXPECTED_SIGNAL_COUNTS
    assert int((ours["SG"][~ours["C"]] != S.ABSENT).sum()) == 0
    assert int(ours["C"].sum()) == 96


def test_copied_trunk_matches_shared_control():
    torch.manual_seed(0)
    cnn = S.SharedCNN(h=16, layers=1, drop=0.0, W=8).eval()
    moe = S.SoftMoE(h=16, layers=1, drop=0.0, W=8, expert_e=8).eval()
    moe.trunk.load_state_dict(cnn.state_dict())
    x = torch.randint(-1, 5, (2, 8, 8))
    t = torch.tensor([10, 12])
    ty = torch.zeros(2, 8, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(cnn(x, t, ty, None), moe.trunk_logits(x, t, ty)[0])
