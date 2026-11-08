"""Re-review probe. Writes only under independent_review_20261108. No training."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path("/home/alexn/Documents/code/datathon_2026/house-edge")
STUDY = REPO / "experiments/cnn03_convergence/continuation_20261108"
OUT = REPO / "experiments/cnn03_convergence/independent_review_20261108"
sys.path.insert(0, str(STUDY / "source"))

import torch

torch.set_num_threads(2)

from study_common import (  # noqa: E402
    check_state,
    continuation_bind_errors,
    load_curve_blob,
    manifest,
    parent_bind_errors,
    read_prediction_csv,
    renorm_predictions,
    score_tune_detail,
    sha256_file,
    tune_contract,
)
from score_fp64 import score_path, score_reference_csv  # noqa: E402
from verify_checkpoints import check_arm  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(label: str) -> dict:
    paths = [
        STUDY / "PLAN.md",
        STUDY / "MANIFEST.json",
        STUDY / "POST_RUN_HASHES.json",
        STUDY / "source/study_common.py",
        STUDY / "source/score_fp64.py",
        STUDY / "source/score_saved.py",
        STUDY / "source/blend_tune.py",
        STUDY / "source/continue_train.py",
        STUDY / "source/verify_checkpoints.py",
        STUDY / "source/compare_corrected.py",
        STUDY / "source/write_post_run_hashes.py",
        STUDY / "tests/test_study.py",
        REPO / "experiments/cnn_runtime/fp64_harness.py",
        REPO / "experiments/mlp_cnn_hpo/review/PINNED.json",
        REPO / "experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip",
        REPO / "experiments/cnn_runtime/first_submission_full2029/cnn03_test_a.csv",
        REPO / "experiments/cnn_runtime/server_fp64_cnn03_val1958.csv",
        REPO / "experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py",
    ]
    pin = manifest()
    rows = {}
    for path in paths:
        rows[str(path.relative_to(REPO))] = sha(path) if path.is_file() else "MISSING"
    for arm, spec in pin["checkpoints"].items():
        path = REPO / spec["path"]
        rows[spec["path"]] = sha(path)
    post = json.loads((STUDY / "POST_RUN_HASHES.json").read_text(encoding="utf-8"))
    for name, spec in post["continuations"].items():
        path = REPO / spec["path"]
        rows[spec["path"]] = sha(path)
    rows["incumbent"] = sha(REPO / pin["incumbent_checkpoint"])
    print(f"SNAPSHOT {label}")
    (OUT / f"snapshot_{label}.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return rows


def expect_refuse(label: str, fn) -> None:
    try:
        fn()
    except RuntimeError as exc:
        print(f"REJECT {label}: {exc}")
        return
    raise SystemExit(f"ACCEPTED {label}")


def guards(before: dict) -> None:
    pin = manifest()
    print("plan_unchanged", before[str((STUDY / "PLAN.md").relative_to(REPO))] == "086187accf10068b38770a1db98b2583150fa7bb84ac96e46426d4847febfff6")
    print("manifest_unchanged", before[str((STUDY / "MANIFEST.json").relative_to(REPO))] == "c71474bcbb6caede6bb629d7db163469788054da2d48dcd4a56c1bc482edcf8b")
    print("harness", before["experiments/cnn_runtime/fp64_harness.py"] == pin["production_fp64_harness_sha256"])
    print("zip", before["experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip"])
    print("submission_csv", before["experiments/cnn_runtime/first_submission_full2029/cnn03_test_a.csv"])
    print("val_csv", before["experiments/cnn_runtime/server_fp64_cnn03_val1958.csv"] == pin["known_reference_csv"]["sha256"])
    print("pinned_json", before["experiments/mlp_cnn_hpo/review/PINNED.json"])
    post = json.loads((STUDY / "POST_RUN_HASHES.json").read_text(encoding="utf-8"))
    print("post_run_flag", post.get("not_a_pretraining_pin"), post.get("measured_at_cst"), post.get("bounded_jobs_finished_cst"))
    print("post_run_file_sha", before["experiments/cnn03_convergence/continuation_20261108/POST_RUN_HASHES.json"])
    for name, spec in pin["checkpoints"].items():
        digest = before[spec["path"]]
        print("parent_pin", name, digest == spec["sha256"], "kernel_field_next")
        blob = load_curve_blob(REPO / spec["path"])
        print(" parent_cfg_keys", sorted(blob["config"]))
        print(" parent_kernel", blob["config"].get("kernel", "ABSENT"))
        print(" parent_h_layers_depth_width", blob["config"].get("h"), blob["config"].get("layers"), blob["config"].get("depth"), blob["config"].get("width"))
        row = check_arm(name, spec)
        print(" check_arm", row["n_tensors"], row["lr"], row["seed"])
        if digest != spec["sha256"]:
            raise SystemExit(f"parent bytes drifted {name}")
    for name, spec in post["continuations"].items():
        digest = before[spec["path"]]
        blob = load_curve_blob(REPO / spec["path"])
        print(
            "cont", name,
            "file", digest == spec["sha256"],
            "arm", blob["config"].get("arm") == spec["arm"],
            "extra", blob.get("completed_extra_epochs") == spec["extra_epochs"],
            "steps", blob["config"].get("total_steps") == spec["extra_epochs"] * 37,
            "parent", blob.get("parent_checkpoint_sha256") == spec["parent_sha256"] == pin["checkpoints"][spec["arm"]]["sha256"],
            "lr", blob["config"].get("lr"),
            "seed", blob["config"].get("seed"),
            "cont_seed", blob.get("continuation_seed"),
            "kernel", blob["config"].get("kernel", "ABSENT"),
        )
        if digest != spec["sha256"]:
            raise SystemExit(f"continuation bytes drifted {name}")
    print("incumbent_post", before["incumbent"] == post["incumbent_checkpoint"]["sha256"])

    contract = tune_contract()
    print("tune_episodes", len(contract["tune_episodes"]), "confirm_episodes", len(contract["confirm_episodes"]), "overlap", len(contract["tune_episodes"] & contract["confirm_episodes"]))
    print("tune_queries", len(contract["tune_query_ids"]), "confirm_queries", len(contract["confirm_query_ids"]))

    # Reference hash is measured inside score_reference_csv, then compared with a direct tune-only call.
    ref = score_reference_csv()
    print("reference_measured", ref)
    preds = read_prediction_csv(REPO / pin["known_reference_csv"]["path"])
    from study_common import ensure_house_edge
    load_official, score_predictions, _write = ensure_house_edge()
    official = load_official(REPO / "kit/public/validation")
    tune_only = {key: preds[key] for key in contract["tune_query_ids"]}
    direct = float(score_predictions(renorm_predictions(tune_only), official, contract["tune_episodes"])["R"])
    detail = score_tune_detail(preds)
    print("direct_tune_R", repr(direct), "study_tune_R", repr(detail["tune_R"]), "equal", direct == detail["tune_R"])
    print("n_tune", detail["n_tune_queries"], "n_forward", detail["n_forward_queries"], "n_episodes", detail["n_tune_episodes"])

    arm = "curve_C_lr003_s0"
    spec = pin["checkpoints"][arm]
    real = REPO / spec["path"]
    real_blob = load_curve_blob(real)
    real_digest = sha256_file(real)

    def save_mut(blob, name: str) -> Path:
        dest = Path(tempfile.mkdtemp(prefix="rereview_", dir="/tmp")) / name
        torch.save(blob, dest)
        if dest.resolve() == real.resolve():
            raise SystemExit("mutation targeted the real parent")
        return dest

    resaved = {"config": dict(real_blob["config"]), "state": {k: v.clone() for k, v in real_blob["state"].items()}}
    expect_refuse("resaved parent", lambda: score_path(save_mut(resaved, "resaved.pt"), arm))

    kernel_blob = {"config": dict(real_blob["config"]), "state": {k: v.clone() for k, v in real_blob["state"].items()}}
    kernel_blob["config"]["kernel"] = 5
    expect_refuse("declared kernel 5 file", lambda: score_path(save_mut(kernel_blob, "kernel5.pt"), arm))
    helper_errors = parent_bind_errors(real, kernel_blob, arm, real_digest)
    print("helper_kernel5_with_forced_matching_digest", helper_errors)

    conv = {"config": dict(real_blob["config"]), "state": {k: v.clone() for k, v in real_blob["state"].items()}}
    conv["state"]["blocks.0.conv.weight"] = torch.zeros(128, 128, 5)
    expect_refuse("conv kernel shape 5", lambda: score_path(save_mut(conv, "conv5.pt"), arm))

    copied = Path(tempfile.mkdtemp(prefix="rereview_copy_", dir="/tmp")) / "checkpoint.pt"
    shutil.copyfile(real, copied)
    expect_refuse("parent byte copy other path", lambda: score_path(copied, arm))

    cont_name = "curve_C_lr003_s2_plus20"
    cont_spec = post["continuations"][cont_name]
    cont_real = REPO / cont_spec["path"]
    cont_copy = Path(tempfile.mkdtemp(prefix="rereview_cont_", dir="/tmp")) / "checkpoint.pt"
    shutil.copyfile(cont_real, cont_copy)
    cont_blob = load_curve_blob(cont_copy)
    cont_errors = continuation_bind_errors(cont_blob, cont_spec["arm"], sha256_file(cont_copy))
    print("continuation_copy_bind_errors", cont_errors, "copy_path", cont_copy, "real_path", cont_real)

    bad_steps = load_curve_blob(cont_real)
    bad_steps["config"] = dict(bad_steps["config"])
    bad_steps["config"]["total_steps"] = 1
    expect_refuse("step budget", lambda: score_path(save_mut(bad_steps, "steps.pt"), cont_spec["arm"]))

    after_mut = sha(real)
    print("real_parent_unchanged_after_mutations", after_mut == real_digest)
    print("real_cont_unchanged", sha(cont_real) == cont_spec["sha256"])


def rescore() -> None:
    pin = manifest()
    post = json.loads((STUDY / "POST_RUN_HASHES.json").read_text(encoding="utf-8"))
    corrected_p = json.loads((STUDY / "corrected_results/fp64_original_tune.json").read_text(encoding="utf-8"))
    corrected_c = json.loads((STUDY / "corrected_results/fp64_continuation_tune.json").read_text(encoding="utf-8"))
    old_p = json.loads((STUDY / "runs/fp64_original_tune.json").read_text(encoding="utf-8"))
    old_c = json.loads((STUDY / "runs/fp64_continuation_tune.json").read_text(encoding="utf-8"))
    parents = {}
    for name, spec in pin["checkpoints"].items():
        row = score_path(REPO / spec["path"], name)
        parents[name] = row
        print("SCORE", name, repr(row["tune_R"]), "n", row["n_queries"], "forward", row["n_forward_queries"], "sha", row["checkpoint_sha256"] == spec["sha256"])
    inc = score_path(REPO / pin["incumbent_checkpoint"], "incumbent_final_cnn03_s0")
    parents["incumbent_final_cnn03_s0"] = inc
    print("SCORE incumbent", repr(inc["tune_R"]), "n", inc["n_queries"])
    ref = score_reference_csv()
    parents["reference_csv"] = ref
    print("SCORE reference", repr(ref["tune_R"]), ref["sha256"])
    cont = {}
    for name, spec in post["continuations"].items():
        row = score_path(REPO / spec["path"], spec["arm"])
        cont[name] = row
        print("SCORE", name, repr(row["tune_R"]), "n", row["n_queries"], "sha", row["checkpoint_sha256"] == spec["sha256"])
    payload = {"parents": parents, "continuations": cont}
    (OUT / "rereview_scores.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    mismatches = []
    for name, row in parents.items():
        if row["tune_R"] != corrected_p[name]["tune_R"] or row["tune_R"] != old_p[name]["tune_R"]:
            mismatches.append(("parent", name, row["tune_R"], corrected_p[name]["tune_R"], old_p[name]["tune_R"]))
        if name != "reference_csv" and (row["n_queries"] != 384 or row["n_forward_queries"] != 768):
            mismatches.append(("count", name, row["n_queries"], row["n_forward_queries"]))
    if ref["n_queries"] != 384 or ref["n_forward_queries"] != 768:
        mismatches.append(("reference_count", ref))
    for name, row in cont.items():
        if row["tune_R"] != corrected_c[name]["tune_R"] or row["tune_R"] != old_c[name]["tune_R"]:
            mismatches.append(("cont", name, row["tune_R"], corrected_c[name]["tune_R"], old_c[name]["tune_R"]))
        if row["n_queries"] != 384 or row["n_forward_queries"] != 768:
            mismatches.append(("cont_count", name, row["n_queries"], row["n_forward_queries"]))
    print("SCORE_MISMATCHES", mismatches)
    c20 = {f"curve_C_lr003_s{i}_plus20": cont[f"curve_C_lr003_s{i}_plus20"]["tune_R"] for i in range(3)}
    winner = min(c20, key=c20.get)
    print("C_PLUS20", {k: repr(v) for k, v in c20.items()}, "lowest", winner, repr(c20[winner]))
    print("incumbent", repr(inc["tune_R"]), "parent_s2", repr(parents["curve_C_lr003_s2"]["tune_R"]), "plus10_s2", repr(cont["curve_C_lr003_s2_plus10"]["tune_R"]))
    print("plus20_worse_than_incumbent", c20[winner] > inc["tune_R"])
    print("plus10_not_the_C20_rule", winner.endswith("_plus20"))


def blends() -> None:
    from blend_tune import BLEND_WEIGHTS, materialize, mix
    pin = manifest()
    cnn = read_prediction_csv(REPO / pin["known_reference_csv"]["path"])
    zero = score_tune_detail(cnn)
    routed, control = materialize()
    print("materialized_ids", len(routed), len(control), "cnn", len(cnn))
    rows = []
    reproduced = {}
    for source, preds in (("routed", routed), ("control", control)):
        reproduced[source] = score_tune_detail(preds)["tune_R"]
        print("MATERIAL", source, repr(reproduced[source]), "n_forward", len(preds))
        for weight in BLEND_WEIGHTS:
            blended = cnn if weight == 0.0 else mix(cnn, preds, weight)
            detail = score_tune_detail(blended)
            rows.append({"source": source, "w": weight, "tune_R": detail["tune_R"], "n_tune_queries": detail["n_tune_queries"]})
            print("BLEND", source, weight, repr(detail["tune_R"]), "n", detail["n_tune_queries"])
    payload = {"cnn03_csv_tune_R": zero["tune_R"], "materialized_tune_R": reproduced, "rows": rows, "weights": list(BLEND_WEIGHTS)}
    (OUT / "rereview_blends.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    old = json.loads((STUDY / "runs/blends_tune.json").read_text(encoding="utf-8"))
    new = json.loads((STUDY / "corrected_results/blends_tune.json").read_text(encoding="utf-8"))
    old_rows = {(r["source"], r["w"]): r["tune_R"] for r in old["rows"]}
    new_rows = {(r["source"], r["w"]): r["tune_R"] for r in new["rows"]}
    mine = {(r["source"], r["w"]): r["tune_R"] for r in rows}
    print("blend_match_old", mine == old_rows, "blend_match_corrected", mine == new_rows)
    print("zero", repr(zero["tune_R"]), "old", repr(old["cnn03_csv_tune_R"]))
    print("material_match", reproduced == old["materialized_tune_R"] == new["materialized_tune_R"])


def main() -> None:
    before = snapshot("before")
    try:
        guards(before)
    except Exception:
        traceback.print_exc()
        raise
    mid = snapshot("after_guards")
    print("bytes_same_after_guards", before == mid)
    rescore()
    blends()
    end = snapshot("after_scores")
    print("bytes_same_after_scores", before == end)
    print("REREVIEW_PROBE_DONE")


if __name__ == "__main__":
    main()
