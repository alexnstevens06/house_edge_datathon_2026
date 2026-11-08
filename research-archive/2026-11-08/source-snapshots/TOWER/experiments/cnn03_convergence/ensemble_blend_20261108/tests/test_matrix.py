"""Guards for the ensemble/blend stream. These tests do not score the study matrix."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source"))

import torch

import common as C


def _curve_blob():
    path = C.STUDY / "inputs" / "curve_C_lr003_s2" / "checkpoint.pt"
    blob = C.load_blob(path)
    return path, blob, C.sha256_file(path)


def test_pinned_files_and_source_have_no_float_cast():
    C.assert_pinned_files()
    for name in ("common.py", "evaluate.py"):
        text = (C.STUDY / "source" / name).read_text(encoding="utf-8")
        assert "torch.autocast" not in text
        assert ".float()" not in text
        assert "HSA_OVERRIDE_GFX_VERSION" in text or name == "evaluate.py"


def test_wrong_hash_config_keys_shape_signal_dtype_and_nonfinite(tmp_path):
    path, blob, digest = _curve_blob()
    assert C.bind_errors(path, blob, "curve_C_lr003_s2", digest) == []
    bad = copy.deepcopy(blob)
    bad["config"]["lr"] = 0.001
    assert "lr mismatch" in "; ".join(C.bind_errors(path, bad, "curve_C_lr003_s2", digest))
    bad = copy.deepcopy(blob)
    bad["config"]["seed"] = 0
    assert "seed mismatch" in "; ".join(C.bind_errors(path, bad, "curve_C_lr003_s2", digest))
    bad = copy.deepcopy(blob)
    bad["extra"] = 1
    assert "weights-only" in "; ".join(C.bind_errors(path, bad, "curve_C_lr003_s2", digest))
    bad = copy.deepcopy(blob)
    bad["state"]["head.2.bias"] = bad["state"]["head.2.bias"][:4]
    assert "shape" in "; ".join(C.bind_errors(path, bad, "curve_C_lr003_s2", digest))
    bad = copy.deepcopy(blob)
    bad["state"]["sig.weight"] = torch.zeros(3, 128)
    assert "signal" in "; ".join(C.bind_errors(path, bad, "curve_C_lr003_s2", digest))
    bad = copy.deepcopy(blob)
    bad["state"]["head.2.bias"] = bad["state"]["head.2.bias"].double()
    assert "saved dtype" in "; ".join(C.bind_errors(path, bad, "curve_C_lr003_s2", digest))
    bad = copy.deepcopy(blob)
    bad["state"]["head.2.bias"] = bad["state"]["head.2.bias"].clone()
    bad["state"]["head.2.bias"][0] = float("nan")
    assert "not finite" in "; ".join(C.bind_errors(path, bad, "curve_C_lr003_s2", digest))
    flipped = tmp_path / "checkpoint.pt"
    payload = path.read_bytes()
    flipped.write_bytes(payload[:-1] + bytes([payload[-1] ^ 0x01]))
    assert "hash mismatch" in "; ".join(
        C.bind_errors(path, blob, "curve_C_lr003_s2", C.sha256_file(flipped))
    )
    assert C.bind_errors(flipped, blob, "curve_C_lr003_s2", digest)[0].startswith("hash") or (
        "path is not the pinned study input" in "; ".join(C.bind_errors(flipped, blob, "curve_C_lr003_s2", digest))
    )
    assert "unknown role" in "; ".join(C.bind_errors(path, blob, "not_a_role", digest))


def test_hard_control_bind_and_curve_controls_present():
    path = C.STUDY / "inputs" / "hard_shared_control" / "checkpoint.pt"
    blob = C.load_blob(path)
    digest = C.sha256_file(path)
    assert C.bind_errors(path, blob, "hard_shared_control", digest) == []
    bad = copy.deepcopy(blob)
    bad["phase"] = "train"
    assert "phase" in "; ".join(C.bind_errors(path, bad, "hard_shared_control", digest))
    for role in ("curve_control_s0", "curve_control_s1", "curve_control_s2"):
        other = C.STUDY / "inputs" / role / "checkpoint.pt"
        other_blob = C.load_blob(other)
        assert C.bind_errors(other, other_blob, role, C.sha256_file(other)) == []


def test_unknown_and_nontune_ids_are_refused():
    contract = C.tune_contract()
    tune = sorted(contract["tune_query_ids"])
    row = [0.2, 0.2, 0.2, 0.2, 0.2]
    try:
        C.score_tune_detail({tune[0]: row, "not-a-query": row})
    except RuntimeError as exc:
        assert "non-tune query id" in str(exc)
    else:
        raise AssertionError("unknown id was scored")
    confirm = next(iter(contract["confirm_query_ids"]))
    try:
        C.score_tune_detail({confirm: row})
    except RuntimeError as exc:
        assert "missing tune query ids" in str(exc)
    else:
        raise AssertionError("confirm-only map was scored")


def test_window_has_no_future_and_missing_is_not_zero():
    model = C.load_canonical_model()
    data = C.load_canonical_data()
    obs = data.MISSING * torch.ones(8, data.N_MACHINES).numpy().astype("int8")
    # numpy path: build an int8 array
    import numpy as np
    obs = np.full((8, data.N_MACHINES), data.MISSING, np.int8)
    obs[4, 0] = 3
    obs[0, 1] = 0
    window = model.window(obs, t=4, W=64)
    assert window.shape == (64, 8)
    assert not (window == 3).any()
    assert window[-4, 1] == 0
    assert (window[-4, 0] == data.MISSING) or (window[0, 0] == data.MISSING)
    # The tick-0 payout 0 is inside the window; missing pads stay -1.
    assert (window == data.MISSING).any()
    assert (window == 0).any()
    x_missing = torch.full((1, 64, 8), data.MISSING, dtype=torch.int64)
    x_zero = torch.zeros(1, 64, 8, dtype=torch.int64)
    net = model.CNN(C.H, C.LAYERS, C.K, C.DROP, C.W, use_signal=False).double().eval()
    t = torch.tensor([64])
    ty = torch.zeros(1, 8, dtype=torch.long)
    sg = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        a = net(x_missing, t, ty, sg)
        b = net(x_zero, t, ty, sg)
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    assert not torch.equal(a, b)


def test_query_order_does_not_change_ids():
    model = C.load_canonical_model()
    data = C.load_canonical_data()
    validation = data.load_public_split(C.REPO / "kit" / "public", "validation")
    validation.episodes = validation.episodes[:2]
    keep = {ep.episode_id for ep in validation.episodes}
    validation.queries = [row for row in validation.queries if row["episode_id"] in keep]
    query, episodes = model.build_query(validation, C.W)
    net = model.CNN(C.H, C.LAYERS, C.K, C.DROP, C.W, use_signal=False).double().eval()
    probs = model.predict(net, query, "cpu", bs=2)
    first = model.to_preds(probs, episodes, validation)
    flipped = copy.copy(validation)
    flipped.queries = list(reversed(validation.queries))
    second = model.to_preds(probs, episodes, flipped)
    assert first == second
    assert list(first) != list(reversed(first)) or len(first) == 1


def test_simplex_weight0_identity_and_arithmetic_mean():
    base = {"q": [0.5, 0.2, 0.1, 0.1, 0.1], "r": [0.0, 0.0, 0.0, 0.0, 1.0]}
    other = {"q": [0.0, 0.0, 0.0, 0.5, 0.5], "r": [0.2, 0.2, 0.2, 0.2, 0.2]}
    zero = C.blend_maps(base, other, 0.0)
    assert zero == {"q": [0.5, 0.2, 0.1, 0.1, 0.1], "r": [0.0, 0.0, 0.0, 0.0, 1.0]}
    assert C.simplex_errors(zero) == []
    weight = 0.2
    mixed = C.blend_maps(base, other, weight)
    manual = []
    for i in range(5):
        manual.append((1.0 - weight) * base["q"][i] + weight * other["q"][i])
    assert mixed["q"] == C.canonical_renorm(manual)
    mean = C.equal_probability_mean([base, other, base])
    expected = [sum(item["q"][i] for item in (base, other, base)) / 3.0 for i in range(5)]
    assert mean["q"] == C.canonical_renorm(expected)
    assert C.simplex_errors(mean) == []
    try:
        C.blend_maps(base, other, 0.3)
    except RuntimeError as exc:
        assert "frozen set" in str(exc)
    else:
        raise AssertionError("an unlisted weight was accepted")


def test_canonical_fp64_logits_softmax_on_real_seed2_state():
    blob = C.require_bind(C.STUDY / "inputs" / "curve_C_lr003_s2" / "checkpoint.pt", "curve_C_lr003_s2")
    model = C.load_canonical_model()
    data = C.load_canonical_data()
    net = model.CNN(C.H, C.LAYERS, C.K, C.DROP, C.W, use_signal=False)
    net.load_state_dict(blob["state"])
    net.double().eval()
    validation = data.load_public_split(C.REPO / "kit" / "public", "validation")
    validation.episodes = validation.episodes[:1]
    keep = {validation.episodes[0].episode_id}
    validation.queries = [row for row in validation.queries if row["episode_id"] in keep]
    query, episodes = model.build_query(validation, C.W)
    batch = [item[:1] for item in query]
    with torch.no_grad():
        logits = net(*[item for item in batch])
    assert logits.dtype == torch.float64
    assert torch.isfinite(logits).all()
    probs = torch.softmax(logits, dim=-1)
    assert probs.dtype == torch.float64
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)))
    pred = model.to_preds(probs.numpy(), episodes, validation)
    renormed = C.renorm_map(pred)
    assert C.simplex_errors(renormed) == []
    assert len(renormed) == len(validation.queries)


def test_split_file_matches_manifest_contract():
    pin = json.loads((C.STUDY / "MANIFEST.json").read_text(encoding="utf-8"))
    assert pin["plan_sha256"] == C.sha256_file(C.STUDY / "PLAN.md")
    assert pin["split_ids_sha256"] == C.sha256_file(C.STUDY / "SPLIT_IDS.json")
    contract = C.tune_contract()
    assert len(contract["tune_query_ids"]) == 384
    assert pin["blend_weights"] == list(C.BLEND_WEIGHTS)
    assert pin["current_best_tune_R"] == C.CURRENT_BEST_TUNE_R
