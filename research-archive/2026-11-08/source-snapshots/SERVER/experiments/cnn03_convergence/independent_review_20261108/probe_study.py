"""Read-only study probe. Writes only this review directory. No training."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path("/home/alexn/Documents/code/datathon_2026/house-edge")
STUDY = REPO / "experiments/cnn03_convergence/continuation_20261108"
sys.path.insert(0, str(STUDY / "source"))

import torch  # noqa: E402

torch.set_num_threads(2)

from study_common import (  # noqa: E402
    check_state,
    continuation_seed,
    ensure_house_edge,
    load_fp64_model,
    manifest,
    parent_is_weights_only,
    read_prediction_csv,
    score_tune,
    sha256_file,
)
from verify_checkpoints import check_arm  # noqa: E402
from score_fp64 import predict_checkpoint, score_path, score_reference_csv  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def show(title: str, fn) -> None:
    print(f"\n== {title} ==")
    try:
        fn()
    except Exception:
        traceback.print_exc()


def section_hashes() -> None:
    paths = [
        STUDY / "MANIFEST.json",
        STUDY / "PLAN.md",
        STUDY / "source/study_common.py",
        STUDY / "source/score_fp64.py",
        STUDY / "source/score_saved.py",
        STUDY / "source/verify_checkpoints.py",
        STUDY / "source/continue_train.py",
        STUDY / "source/blend_tune.py",
        STUDY / "source/launch_continuations.sh",
        STUDY / "tests/test_study.py",
        REPO / "experiments/cnn_runtime/fp64_harness.py",
        REPO / "experiments/mlp_cnn_hpo/review/PINNED.json",
        REPO / "experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip",
        REPO / "experiments/cnn_runtime/server_fp64_cnn03_val1958.csv",
        REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py",
        REPO / "experiments/cnn_runtime/fresh_fp64_anchor_1956/model.py",
        REPO / "experiments/cnn03_convergence/source/base_cnn.py",
    ]
    pin = manifest()
    print("harness_matches_manifest", sha(paths[10]) == pin["production_fp64_harness_sha256"])
    print("harness", sha(paths[10]))
    print("PINNED", sha(paths[11]), "bytes", paths[11].stat().st_size)
    for path in paths:
        print(f"{path.relative_to(REPO)} {sha(path)}")


def section_checkpoints() -> None:
    pin = manifest()["checkpoints"]
    for name, spec in pin.items():
        row = check_arm(name, spec)
        print("PASS", name, "sha", row["sha256"][:12], "n", row["n_tensors"], "lr", row["lr"], "seed", row["seed"])
        bad = dict(spec)
        bad["sha256"] = "0" * 64
        try:
            check_arm(name, bad)
            print("FAIL hash rejection missing", name)
        except RuntimeError as exc:
            print("REJECT hash", name, str(exc).split("!=")[0])
        blob = __import__("study_common", fromlist=["load_curve_blob"]).load_curve_blob(REPO / spec["path"])
        print("keys", sorted(blob), "cfg_kernel", blob["config"].get("kernel"), "pct", blob["config"].get("pct_start"), "optimizer_in_blob", "optimizer" in blob)
        if name != "curve_C_lr003_s0":
            continue
        state = {k: v.clone() for k, v in blob["state"].items()}
        shape_bad = dict(state)
        shape_bad["head.2.bias"] = shape_bad["head.2.bias"][:-1]
        try:
            check_state(shape_bad)
            print("FAIL shape rejection missing")
        except RuntimeError as exc:
            print("REJECT shape", exc)
        extra = dict(state)
        extra["sig.weight"] = torch.zeros(3, 4)
        try:
            check_state(extra)
            print("FAIL signal-key rejection missing")
        except RuntimeError as exc:
            print("REJECT extra", exc)
        missing = dict(state)
        missing.pop("pay.weight")
        try:
            check_state(missing)
            print("FAIL missing-key rejection missing")
        except RuntimeError as exc:
            print("REJECT missing", exc)


def section_scorer_hash_gap() -> None:
    import inspect
    import score_fp64
    text = inspect.getsource(score_fp64.predict_checkpoint) + inspect.getsource(score_fp64.score_path) + inspect.getsource(score_fp64.score_reference_csv)
    print("score_path_source_mentions_sha256", "sha256" in inspect.getsource(score_fp64.score_path))
    print("predict_mentions_sha256", "sha256" in inspect.getsource(score_fp64.predict_checkpoint))
    print("reference_fn_mentions_sha256", "sha256" in inspect.getsource(score_fp64.score_reference_csv))
    saved = (STUDY / "source/score_saved.py").read_text(encoding="utf-8")
    print("score_saved_mentions_sha256", "sha256" in saved)
    print("score_saved_mentions_parent", "parent" in saved)


def section_tune_and_renorm() -> None:
    pin = manifest()
    csv_path = REPO / pin["known_reference_csv"]["path"]
    print("csv_sha_measured", sha256_file(csv_path))
    print("csv_sha_pin", pin["known_reference_csv"]["sha256"])
    preds = read_prediction_csv(csv_path)
    splits = json.loads((REPO / "runs/splits.json").read_text(encoding="utf-8"))
    tune = set(splits["tune"])
    confirm = set(splits["confirm"])
    print("n_tune_episodes", len(tune), "n_confirm_episodes", len(confirm), "overlap", len(tune & confirm))
    load_official, score_predictions, write_submission_csv = ensure_house_edge()
    official = load_official(REPO / "kit/public/validation")
    targets, groups, reference, queries = official
    def count(episodes):
        ids = [q for q, ep in queries.items() if ep in episodes]
        conds = sorted({groups[q] for q in ids if q in groups})
        return len(ids), conds
    print("queries_total", len(queries), "queries_tune", count(tune), "queries_confirm", count(confirm))
    neither = {ep for ep in queries.values() if ep not in tune and ep not in confirm}
    print("validation_episodes_outside_tune_and_confirm", len(neither))
    tune_r = score_tune(preds)
    direct = float(score_predictions(preds, official, tune)["R"])
    confirm_r = float(score_predictions(preds, official, confirm)["R"])
    all_eps = set(queries.values())
    all_r = float(score_predictions(preds, official, all_eps)["R"])
    bundle = score_predictions(preds, official, tune)
    print("bundle_keys", sorted(bundle))
    print("tune_R", repr(tune_r))
    print("direct_tune_R", repr(direct))
    print("confirm_R", repr(confirm_r))
    print("all_validation_R", repr(all_r))
    print("tune_eq_direct", tune_r == direct)
    print("tune_eq_confirm", tune_r == confirm_r)
    print("tune_eq_all", tune_r == all_r)
    author = json.loads((STUDY / "runs/fp64_original_tune.json").read_text(encoding="utf-8"))
    print("author_reference", repr(author["reference_csv"]["tune_R"]))
    print("matches_author_reference", tune_r == author["reference_csv"]["tune_R"])
    print("reference_json_sha_is_pin_not_remeasured_in_author_fn")
    echoed = score_reference_csv()
    print("echoed_sha", echoed["sha256"], "echo_equals_pin", echoed["sha256"] == pin["known_reference_csv"]["sha256"])


def section_forward_agreement() -> None:
    import score_fp64
    pin = manifest()
    author = json.loads((STUDY / "runs/fp64_original_tune.json").read_text(encoding="utf-8"))
    # Incumbent and one high-LR parent. Full six-file rescore is separate if this stays cheap.
    for label, path in (
        ("incumbent_final_cnn03_s0", REPO / pin["incumbent_checkpoint"]),
        ("curve_C_lr003_s0", REPO / pin["checkpoints"]["curve_C_lr003_s0"]["path"]),
    ):
        got = score_path(Path(path))
        print(label, "independent", repr(got["tune_R"]), "n_queries_reported", got["n_queries"], "author", repr(author[label]["tune_R"]), "equal", got["tune_R"] == author[label]["tune_R"])
    module = load_fp64_model()
    text = (REPO / "experiments/cnn_runtime/fp64_harness.py").read_text(encoding="utf-8")
    print("study_does_not_import_harness", "fp64_harness" not in (STUDY / "source/score_fp64.py").read_text(encoding="utf-8"))
    print("canonical_USE_SIGNAL", module.USE_SIGNAL)
    # Compare base_cnn and canonical module on one random batch after weight copy.
    sys.path.insert(0, str(REPO / "experiments/cnn03_convergence/source"))
    import base_cnn as B
    B.USE_SIGNAL = False
    torch.manual_seed(0)
    canon = module.CNN(128, 4, 3, 0.3, 64, use_signal=False).double()
    other = B.CNN(128, 4, 3, 0.3, 64, use_signal=False).double()
    other.load_state_dict(canon.state_dict())
    canon.eval()
    other.eval()
    x = torch.randint(-1, 5, (2, 64, 8))
    t = torch.tensor([20, 21])
    ty = torch.zeros(2, 8, dtype=torch.long)
    sg = torch.zeros(2, 8, dtype=torch.long)
    with torch.no_grad():
        a = canon(x, t, ty, sg)
        b = other(x, t, ty, sg)
    print("base_vs_canonical_keys", set(canon.state_dict()) == set(other.state_dict()))
    print("base_vs_canonical_logits_equal", torch.equal(a, b), "dtype", str(a.dtype))
    print("n_params", sum(p.numel() for p in canon.parameters()))
    print("continuation_seeds", continuation_seed(0, 10), continuation_seed(0, 20), continuation_seed(1, 20), continuation_seed(2, 20))
    try:
        continuation_seed(0, 30)
        print("FAIL extra cap")
    except RuntimeError as exc:
        print("REJECT extra", exc)
    print("predict_checkpoint_defined", predict_checkpoint.__name__)


def main() -> None:
    show("hashes", section_hashes)
    show("checkpoints", section_checkpoints)
    show("scorer_hash_gap", section_scorer_hash_gap)
    show("tune_and_renorm", section_tune_and_renorm)
    show("forward", section_forward_agreement)
    print("\nPROBE_DONE")


if __name__ == "__main__":
    main()
