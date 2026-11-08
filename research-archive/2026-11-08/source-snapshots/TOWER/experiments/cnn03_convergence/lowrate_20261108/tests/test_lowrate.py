"""Negative contract tests for the low-LR study. Reads pins. Writes nothing in the repo."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from contract import (  # noqa: E402
    ARMS,
    BS,
    CONTINUATION_SEED,
    CONV_SOURCE,
    DROP,
    EPOCHS,
    EXPECTED_WINDOWS,
    H,
    HOST_LR,
    KERNEL,
    LAYERS,
    PARENT_BATCH,
    REPO,
    SAVE_EPOCHS,
    SOURCE_SHA_PATH,
    STEPS_PER_EPOCH,
    STUDY,
    TOTAL_STEPS,
    W,
    WD,
    assert_input_pins,
    budget_errors,
    canonical_renorm,
    expected_shapes,
    host_lr,
    manifest,
    parent_errors,
    refuse_if_no_cuda,
    run_errors,
    sha256_file,
    state_errors,
    tune_id_errors,
)

sys.path.insert(0, str(CONV_SOURCE))
import base_cnn as B  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def expect_error(errors: list[str], needle: str, label: str) -> None:
    if not any(needle in item for item in errors):
        FAILURES.append(f"{label}: expected {needle!r} in {errors}")


def clone_state(state: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def valid_run(state: dict, arm: str, digest: str) -> dict:
    host = "server"
    config = {
        "host": host,
        "arm": arm,
        "parent": ARMS[arm],
        "train_lr": HOST_LR[host],
        "wd": WD,
        "bs": BS,
        "drop": DROP,
        "h": H,
        "layers": LAYERS,
        "kernel": KERNEL,
        "W": W,
        "parent_seed": 2,
        "continuation_seed": CONTINUATION_SEED,
        "completed_epochs": 1,
        "completed_steps": STEPS_PER_EPOCH,
        "epochs_budget": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_steps_budget": TOTAL_STEPS,
        "precision": "fp32",
        "scheduler": "none",
        "autocast": False,
        "grad_clip": 1.0,
        "warm_start_from": "weights_only",
        "parent_train_bs": PARENT_BATCH,
        "parent_precision": "bf16_autocast",
    }
    return {
        "state": state,
        "optimizer": {"state": {}, "param_groups": []},
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [],
        "numpy_rng": np.random.get_state(),
        "completed_epochs": 1,
        "completed_steps": STEPS_PER_EPOCH,
        "parent_checkpoint_sha256": manifest()["checkpoints"][ARMS[arm]]["sha256"],
        "parent_had_optimizer": False,
        "warm_start_from": "weights_only",
        "continuation_seed": CONTINUATION_SEED,
        "digest_checked": digest,
    }


def test_pins_and_parents() -> None:
    assert_input_pins()
    check(len(expected_shapes()) == 43, "shape table is not 43")
    check(STEPS_PER_EPOCH == 146 and TOTAL_STEPS == 730, "step budget")
    check(HOST_LR == {"server": 2e-5, "tower": 5e-5}, "host learning rates")
    check(host_lr("server") == 2e-5 and host_lr("tower") == 5e-5, "host_lr")
    pin = manifest()
    check(set(pin["jobs"]) == {"server_primary", "server_control", "tower_primary", "tower_control"}, "jobs")
    for arm, name in ARMS.items():
        spec = pin["checkpoints"][name]
        path = REPO / spec["path"]
        digest = sha256_file(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        errors = parent_errors(blob, arm, digest)
        check(errors == [], f"real {arm} parent refused: {errors}")
        state = clone_state(blob["state"])
        expect_error(parent_errors(blob, arm, "0" * 64), "hash mismatch", f"{arm} wrong hash")
        bad_cfg = {"config": dict(blob["config"]), "state": state}
        bad_cfg["config"]["seed"] = 0
        bad_cfg["config"]["lr"] = 0.01
        bad_cfg["config"]["bs"] = 128
        expect_error(parent_errors(bad_cfg, arm, digest), "parent seed", f"{arm} wrong seed")
        expect_error(parent_errors(bad_cfg, arm, digest), "parent lr", f"{arm} wrong parent lr")
        expect_error(parent_errors(bad_cfg, arm, digest), "bs", f"{arm} wrong parent batch")
        extra = dict(state)
        extra["sig.weight"] = torch.zeros(3, H)
        expect_error(state_errors(extra), "signal", f"{arm} signal key")
        missing = dict(state)
        missing.pop("head.2.weight")
        expect_error(state_errors(missing), "architecture keys", f"{arm} missing key")
        shaped = clone_state(state)
        shaped["blocks.0.conv.weight"] = torch.zeros(H, H, 5)
        expect_error(state_errors(shaped), "shape", f"{arm} wrong shape")
        typed = clone_state(state)
        typed["head.2.bias"] = typed["head.2.bias"].double()
        expect_error(state_errors(typed), "dtype", f"{arm} wrong dtype")
        poisoned = clone_state(state)
        poisoned["head.2.bias"] = poisoned["head.2.bias"].clone()
        poisoned["head.2.bias"][0] = float("nan")
        expect_error(state_errors(poisoned), "not finite", f"{arm} nonfinite")
        run = valid_run(state, arm, digest)
        check(run_errors(run, digest, digest) == [], f"valid run refused: {run_errors(run, digest, digest)}")
        expect_error(run_errors(run, "bad", digest), "hash mismatch", f"{arm} run hash")
        with_sched = dict(run)
        with_sched["scheduler"] = {}
        expect_error(run_errors(with_sched, digest, digest), "scheduler", f"{arm} scheduler state")
        broken = dict(run)
        broken["config"] = dict(run["config"])
        broken["config"]["train_lr"] = 0.003
        expect_error(run_errors(broken, digest, digest), "train lr", f"{arm} high restart lr")
        broken["config"]["train_lr"] = HOST_LR["server"]
        broken["config"]["completed_epochs"] = 2
        broken["completed_epochs"] = 2
        broken["config"]["completed_steps"] = 292
        broken["completed_steps"] = 292
        expect_error(run_errors(broken, digest, digest), "epoch not in frozen", f"{arm} epoch 2")
        broken["config"]["completed_epochs"] = 1
        broken["completed_epochs"] = 1
        broken["config"]["completed_steps"] = 100
        broken["completed_steps"] = 100
        expect_error(run_errors(broken, digest, digest), "step budget", f"{arm} step budget")
        broken["config"]["completed_steps"] = STEPS_PER_EPOCH
        broken["completed_steps"] = STEPS_PER_EPOCH
        broken["parent_had_optimizer"] = True
        expect_error(run_errors(broken, digest, digest), "warm-start", f"{arm} silent optimizer resume")


def test_ids_budget_and_cpu() -> None:
    tune = {"a"}
    confirm = {"b"}
    expect_error(tune_id_errors({"b"}, tune, confirm), "missing tune", "missing tune id")
    expect_error(tune_id_errors({"a", "c"}, tune, confirm), "non-tune", "non-tune id")
    check(tune_id_errors({"a", "b"}, tune, confirm) == [], "confirm id in the forward set is not a scored id")
    check(budget_errors(1, 146) == [] and budget_errors(3, 438) == [] and budget_errors(5, 730) == [], "save epochs")
    expect_error(budget_errors(4, 584), "epoch not in frozen", "epoch 4")
    expect_error(budget_errors(5, 729), "step budget", "short job")
    check(SAVE_EPOCHS == (1, 3, 5), "save tuple")
    raised = False
    try:
        refuse_if_no_cuda(False)
    except RuntimeError as exc:
        raised = "CPU" in str(exc)
    check(raised, "CPU training was not refused")
    try:
        canonical_renorm([1.0, float("nan")])
        check(False, "nonfinite renorm was accepted")
    except RuntimeError:
        pass
    try:
        canonical_renorm([0.0, 0.0])
        check(False, "zero mass was accepted")
    except RuntimeError:
        pass


def test_source_text_and_construction() -> None:
    train_text = (STUDY / "source" / "train_lowrate.py").read_text(encoding="utf-8")
    score_text = (STUDY / "source" / "score_fp64.py").read_text(encoding="utf-8")
    check("OneCycle" not in train_text, "trainer mentions OneCycle")
    check("torch.autocast" not in train_text and "autocast(" not in train_text, "trainer uses autocast")
    check('add_argument("--lr"' not in train_text, "trainer accepts an LR override")
    check(train_text.find("load_state_dict") < train_text.find("manual_seed(CONTINUATION_SEED)"), "seed is not after load")
    check("model(*batch[:-1])" in train_text, "forward does not exclude the label")
    check("set_per_process_memory_fraction(0.40)" in train_text, "memory fraction missing")
    check("allow_tf32 = False" in train_text, "TF32 flag missing")
    check("check_submission" not in score_text, "scorer calls the full submission checker")
    check('os.environ["CUDA_VISIBLE_DEVICES"] = ""' in score_text, "scorer does not hide CUDA")
    B.USE_SIGNAL = False
    obs = np.ones((40, 8), np.int8)
    obs[30] = 4
    window = B.window(obs, 30, W)
    check(int((window == 4).sum()) == 0, "window included the cutoff tick")
    check(bool((window[-1] == 1).all()), "last row is not the previous tick")
    train = B.load_split(__import__("os").path.join(B.PUB, "train"), with_labels=True)
    built = B.build_train(train, W)
    X, _T, _TY, SG, _Y = built
    check(len(X) == EXPECTED_WINDOWS, f"windows {len(X)}")
    check(int(SG.sum()) == 0, "signal channel was backfilled")
    splits = json.loads((REPO / "runs" / "splits.json").read_text(encoding="utf-8"))
    tune = list(splits["tune"])
    confirm = list(splits["confirm"])
    check(len(tune) == 48 and len(confirm) == 48 and not (set(tune) & set(confirm)), "episode split")
    with (REPO / "kit" / "public" / "validation" / "queries.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tune_q = sorted(row["query_id"] for row in rows if row["episode_id"] in set(tune))
    confirm_q = sorted(row["query_id"] for row in rows if row["episode_id"] in set(confirm))
    train_ids = sorted(ep.episode_id for ep in train.episodes)
    frozen = json.loads((STUDY / "SPLIT_IDS.json").read_text(encoding="utf-8"))
    check(frozen["tune_episodes"] == tune, "frozen tune episodes drifted")
    check(frozen["confirm_episodes"] == confirm, "frozen confirm episodes drifted")
    check(frozen["tune_query_ids"] == tune_q, "frozen tune query ids drifted")
    check(frozen["confirm_query_ids"] == confirm_q, "frozen confirm query ids drifted")
    check(frozen["train_episode_ids"] == train_ids, "frozen train episode ids drifted")
    check(len(tune_q) == 384 and len(confirm_q) == 384, "query counts")


def test_source_manifest() -> None:
    required = [
        "COORDINATOR_PLAN.md",
        "PLAN.md",
        "MANIFEST.json",
        "SPLIT_IDS.json",
        "source/contract.py",
        "source/train_lowrate.py",
        "source/score_fp64.py",
        "source/preflight_fp64.py",
        "source/preflight_fp32.py",
        "tests/test_lowrate.py",
    ]
    lines = [line for line in SOURCE_SHA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    rels = []
    for line in lines:
        digest, rel = line.split(None, 1)
        rels.append(rel)
        got = sha256_file(STUDY / rel)
        check(got == digest, f"source hash drift {rel}")
    check(rels == required, f"source manifest list {rels}")


def main() -> None:
    test_pins_and_parents()
    test_ids_budget_and_cpu()
    test_source_text_and_construction()
    test_source_manifest()
    if FAILURES:
        print("LOW_RATE_TESTS_FAIL")
        for item in FAILURES:
            print(item)
        raise SystemExit(1)
    print(f"LOW_RATE_TESTS_OK failures=0 windows={EXPECTED_WINDOWS} steps={TOTAL_STEPS}")


if __name__ == "__main__":
    main()
