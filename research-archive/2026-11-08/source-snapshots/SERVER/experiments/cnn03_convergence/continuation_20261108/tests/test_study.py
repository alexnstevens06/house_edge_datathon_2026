"""Study guards: architecture, canonical renorm, FP64 forward, frozen budget."""
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source"))

from study_common import (  # noqa: E402
    BLEND_WEIGHTS,
    DROP,
    EXTRA_EPOCHS,
    FP64_ENTRY,
    H,
    K,
    LAYERS,
    REPO,
    STEPS_PER_EPOCH,
    W,
    canonical_renorm,
    check_state,
    continuation_seed,
    ensure_house_edge,
    expected_shapes,
    load_curve_blob,
    load_fp64_model,
    manifest,
    parent_bind_errors,
    parent_is_weights_only,
    read_prediction_csv,
    renorm_predictions,
    score_tune,
    sha256_file,
    tune_contract,
)


def test_manifest_and_expected_architecture():
    pin = manifest()
    assert len(pin["checkpoints"]) == 6
    shapes = expected_shapes()
    assert len(shapes) == 43
    assert "sig.weight" not in shapes
    model = load_fp64_model().CNN(H, LAYERS, K, DROP, W, use_signal=False)
    check_state(model.state_dict())
    assert model.sig is None


def test_weights_only_detector_and_budget():
    assert parent_is_weights_only({"config": {}, "state": {}})
    assert not parent_is_weights_only({"config": {}, "state": {}, "optimizer": {}})
    assert continuation_seed(0, 10) == 1010
    assert continuation_seed(2, 20) == 1022
    assert continuation_seed(0, 10) == continuation_seed(0, 10)
    try:
        continuation_seed(0, 30)
    except RuntimeError:
        pass
    else:
        raise AssertionError("extra epoch cap was not enforced")
    assert EXTRA_EPOCHS == (10, 20)
    assert 10 * STEPS_PER_EPOCH == 370
    assert 20 * STEPS_PER_EPOCH == 740


def test_renorm_matches_canonical_writer():
    import tempfile
    _, _, write_submission_csv = ensure_house_edge()
    raw = {"q-b": [0.2, 0.2, 0.2, 0.2, 0.3], "q-a": [2.0, 0.0, 0.0, 0.0, 2.0]}
    path = Path(tempfile.mkdtemp(prefix="cnn03cont_")) / "sub.csv"
    write_submission_csv(path, raw)
    written = read_prediction_csv(path)
    again = renorm_predictions(raw)
    for key in raw:
        assert again[key] == written[key]
    assert canonical_renorm([2.0, 0.0, 0.0, 0.0, 2.0]) == [0.5, 0.0, 0.0, 0.0, 0.5]


def test_reference_csv_study_score_matches_direct_call():
    load_official, score_predictions, _ = ensure_house_edge()
    pin = manifest()["known_reference_csv"]
    preds = read_prediction_csv(REPO / pin["path"])
    official = load_official(REPO / "kit" / "public" / "validation")
    from study_common import tune_ids
    direct = float(score_predictions(preds, official, tune_ids())["R"])
    assert score_tune(preds) == direct


def test_fp64_forward_matches_second_load_of_canonical_file():
    first = load_fp64_model()
    spec = importlib.util.spec_from_file_location("fp64_model_again", FP64_ENTRY / "model.py")
    second = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(second)
    torch.manual_seed(0)
    net_a = first.CNN(8, 2, 3, 0.1, 4, use_signal=False).double()
    net_b = second.CNN(8, 2, 3, 0.1, 4, use_signal=False).double()
    net_b.load_state_dict(net_a.state_dict())
    net_a.eval()
    net_b.eval()
    x = torch.randint(-1, 5, (2, 4, 8))
    t = torch.tensor([6, 7])
    ty = torch.zeros(2, 8, dtype=torch.long)
    sg = torch.zeros(2, 8, dtype=torch.long)
    with torch.no_grad():
        logits_a = net_a(x, t, ty, sg)
        logits_b = net_b(x, t, ty, sg)
    assert logits_a.dtype == torch.float64
    assert torch.equal(logits_a, logits_b)
    probs = torch.softmax(logits_a, dim=-1)
    assert probs.dtype == torch.float64
    assert torch.allclose(probs.sum(-1), torch.ones(2, 8, dtype=torch.float64))


def _refuse(blob, arm, token):
    from score_fp64 import score_path
    path = Path(tempfile.mkdtemp(prefix="cnn03repair_")) / "corrupt.pt"
    torch.save(blob, path)
    try:
        score_path(path, arm)
    except RuntimeError as exc:
        text = str(exc)
    else:
        raise AssertionError(f"score_path accepted a corrupt checkpoint for {token}")
    if token not in text:
        raise AssertionError(f"wanted {token} in {text}")
    pins = manifest()["checkpoints"]
    if arm in pins and path.resolve() == (REPO / pins[arm]["path"]).resolve():
        raise AssertionError("refusal test wrote the original checkpoint")


def _parent_blob():
    spec = manifest()["checkpoints"]["curve_C_lr003_s0"]
    blob = load_curve_blob(REPO / spec["path"])
    return {
        "config": dict(blob["config"]),
        "state": {key: value.detach().cpu().clone() for key, value in blob["state"].items()},
    }


def test_real_parent_hash_binds():
    spec = manifest()["checkpoints"]["curve_C_lr003_s0"]
    path = REPO / spec["path"]
    blob = load_curve_blob(path)
    errors = parent_bind_errors(path, blob, "curve_C_lr003_s0", sha256_file(path))
    assert errors == []


def test_score_path_refuses_hash_key_shape_finite_signal_lr_seed_budget_parent():
    base = _parent_blob()
    resaved = {"config": dict(base["config"]), "state": dict(base["state"])}
    _refuse(resaved, "curve_C_lr003_s0", "hash mismatch")
    shaped = {"config": dict(base["config"]), "state": dict(base["state"])}
    shaped["state"]["head.2.bias"] = torch.zeros(4)
    _refuse(shaped, "curve_C_lr003_s0", "shape")
    missing = {"config": dict(base["config"]), "state": dict(base["state"])}
    del missing["state"]["pay.weight"]
    _refuse(missing, "curve_C_lr003_s0", "architecture keys")
    leaked = {"config": dict(base["config"]), "state": dict(base["state"])}
    leaked["state"]["sig.weight"] = torch.zeros(3, 128)
    _refuse(leaked, "curve_C_lr003_s0", "signal tensors")
    dirty = {"config": dict(base["config"]), "state": dict(base["state"])}
    dirty["state"]["pay.weight"] = dirty["state"]["pay.weight"].clone()
    dirty["state"]["pay.weight"][0, 0] = float("nan")
    _refuse(dirty, "curve_C_lr003_s0", "not finite")

    cont_path = REPO / "experiments/cnn03_convergence/continuation_20261108/runs/curve_C_lr003_s2_plus20/checkpoint.pt"
    cont = load_curve_blob(cont_path)
    bad_lr = dict(cont)
    bad_lr["config"] = dict(cont["config"])
    bad_lr["config"]["lr"] = 1.0
    _refuse(bad_lr, "curve_C_lr003_s2", "lr mismatch")
    bad_seed = dict(cont)
    bad_seed["config"] = dict(cont["config"])
    bad_seed["config"]["seed"] = 9
    _refuse(bad_seed, "curve_C_lr003_s2", "seed mismatch")
    bad_budget = dict(cont)
    bad_budget["config"] = dict(cont["config"])
    bad_budget["config"]["extra_epochs"] = 30
    bad_budget["completed_extra_epochs"] = 30
    _refuse(bad_budget, "curve_C_lr003_s2", "extra_epochs budget")
    bad_parent = dict(cont)
    bad_parent["parent_checkpoint_sha256"] = "0" * 64
    _refuse(bad_parent, "curve_C_lr003_s2", "parent hash mismatch")
    _refuse(base, "incumbent_final_cnn03_s0", "incumbent path")


def test_score_tune_refuses_non_tune_ids():
    contract = tune_contract()
    bogus = {next(iter(contract["tune_query_ids"])): [0.2, 0.2, 0.2, 0.2, 0.2], "not-a-real-query": [0.2, 0.2, 0.2, 0.2, 0.2]}
    try:
        score_tune(bogus)
    except RuntimeError as exc:
        assert "non-tune query id" in str(exc)
    else:
        raise AssertionError("bogus query id was scored")
    confirm_only = {next(iter(contract["confirm_query_ids"])): [0.2, 0.2, 0.2, 0.2, 0.2]}
    try:
        score_tune(confirm_only)
    except RuntimeError as exc:
        assert "missing tune query ids" in str(exc)
    else:
        raise AssertionError("confirm-only ids were scored")


def test_blend_zero_weight_is_identity():
    assert BLEND_WEIGHTS[0] == 0.0
    base = {"q": [0.2, 0.2, 0.2, 0.2, 0.2]}
    other = {"q": [0.0, 0.0, 0.0, 0.0, 1.0]}
    weight = 0.0
    mixed = [(1.0 - weight) * base["q"][i] + weight * other["q"][i] for i in range(5)]
    assert canonical_renorm(mixed) == canonical_renorm(base["q"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__, flush=True)
    print(f"PASS {len(tests)} tests", flush=True)
