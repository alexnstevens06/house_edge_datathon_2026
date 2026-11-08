"""CPU float64 tune scores for saved epochs 1, 3, and 5. No confirm/full/test."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (  # noqa: E402
    DROP,
    FP64_ENTRY,
    H,
    KERNEL,
    LAYERS,
    REPO,
    SAVE_EPOCHS,
    STUDY,
    W,
    canonical_renorm,
    run_errors,
    sha256_file,
    tune_id_errors,
)


def load_fp64_model():
    sys.path.insert(0, str(FP64_ENTRY))
    spec = importlib.util.spec_from_file_location("lowrate_fp64_model", FP64_ENTRY / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the canonical fp64 model")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lowrate_fp64_model"] = module
    spec.loader.exec_module(module)
    if module.USE_SIGNAL:
        raise RuntimeError("canonical fp64 model has signal enabled")
    return module


def split_contract() -> dict:
    import csv
    splits = json.loads((REPO / "runs" / "splits.json").read_text(encoding="utf-8"))
    tune = set(splits["tune"])
    confirm = set(splits["confirm"])
    if len(tune) != 48 or len(confirm) != 48 or tune & confirm:
        raise RuntimeError("tune/confirm episode contract is not 48 and 48 disjoint")
    with (REPO / "kit" / "public" / "validation" / "queries.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tune_q = {row["query_id"] for row in rows if row["episode_id"] in tune}
    confirm_q = {row["query_id"] for row in rows if row["episode_id"] in confirm}
    if len(tune_q) != 384 or len(confirm_q) != 384 or tune_q & confirm_q or len(rows) != 768:
        raise RuntimeError("tune query contract is not 384 and 384")
    frozen = json.loads((STUDY / "SPLIT_IDS.json").read_text(encoding="utf-8"))
    if set(frozen["tune_episodes"]) != tune or set(frozen["tune_query_ids"]) != tune_q:
        raise RuntimeError("SPLIT_IDS.json does not match the current split files")
    return {"tune_episodes": tune, "confirm_episodes": confirm, "tune_query_ids": tune_q, "confirm_query_ids": confirm_q}


def score_checkpoint(run_dir: Path, epoch: int) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("",):
        raise RuntimeError("scorer refuses a visible GPU")
    if epoch not in SAVE_EPOCHS:
        raise RuntimeError("epoch not in frozen save set")
    run_dir = run_dir.resolve()
    if STUDY.resolve() not in run_dir.parents:
        raise RuntimeError("refusing to score outside the study folder")
    snapshot_path = run_dir / "POST_RUN_SNAPSHOT.json"
    if not snapshot_path.is_file():
        raise RuntimeError("post-run snapshot missing")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("not_a_pretraining_pin") is not True or snapshot.get("written_before_metrics") is not True:
        raise RuntimeError("snapshot is not an honest post-run pin")
    record = snapshot.get("epochs", {}).get(str(epoch))
    if not record:
        raise RuntimeError("snapshot has no row for this epoch")
    path = run_dir / f"epoch_{epoch:02d}" / "checkpoint.pt"
    digest = sha256_file(path)
    import torch
    torch.set_num_threads(2)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    errors = run_errors(blob, digest, record["sha256"])
    if errors:
        raise RuntimeError("score refused: " + "; ".join(errors))
    model_mod = load_fp64_model()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "kit"))
    from house_edge.data import load_split
    from house_edge.scoring import load_official_validation, score_predictions

    model_mod.USE_SIGNAL = False
    net = model_mod.CNN(H, LAYERS, KERNEL, DROP, W, use_signal=False)
    net.load_state_dict(blob["state"])
    net.double()
    net.eval()
    if net.use_signal or net.sig is not None:
        raise RuntimeError("signal mode refused")
    floating = [p for p in net.parameters() if p.is_floating_point()]
    if len(floating) != 43 or any(p.dtype != torch.float64 for p in floating):
        raise RuntimeError("parameters are not 43 float64 tensors")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("autocast is enabled")
    validation = load_split(REPO / "kit" / "public" / "validation")
    query, episodes = model_mod.build_query(validation, W)
    probs = model_mod.predict(net, query, "cpu")
    if str(probs.dtype) != "float64":
        raise RuntimeError(f"predict dtype {probs.dtype}")
    preds = model_mod.to_preds(probs, episodes, validation)
    contract = split_contract()
    id_errors = tune_id_errors(set(preds), contract["tune_query_ids"], contract["confirm_query_ids"])
    if id_errors:
        raise RuntimeError("score refused: " + "; ".join(id_errors))
    official = load_official_validation(REPO / "kit" / "public" / "validation")
    targets, _groups, _reference, query_eps = official
    official_tune = {row.query_id for row in targets if query_eps[row.query_id] in contract["tune_episodes"]}
    if official_tune != contract["tune_query_ids"]:
        raise RuntimeError("tune query ids disagree with official validation targets")
    scored_preds = {key: canonical_renorm(preds[key]) for key in contract["tune_query_ids"]}
    if len(scored_preds) != 384:
        raise RuntimeError(f"tune score set is {len(scored_preds)}, want 384")
    scored = score_predictions(scored_preds, official, contract["tune_episodes"])
    if set(scored) & {"confirm", "full", "test"}:
        raise RuntimeError("scorer returned a non-tune bundle")
    return {
        "tune_R": float(scored["R"]),
        "n_tune_queries": 384,
        "n_forward_queries": len(preds),
        "epoch": epoch,
        "checkpoint_sha256": digest,
        "device": "cpu",
        "dtype": "float64",
        "confirm_full_test_metrics": "not_computed",
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if STUDY.resolve() not in out.parents:
        raise SystemExit("refusing to write a score outside the study folder")
    result = score_checkpoint(args.run, args.epoch)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        f"TUNE_ONLY epoch={result['epoch']} R={result['tune_R']:.17g} "
        f"n={result['n_tune_queries']} forward={result['n_forward_queries']}"
    )


if __name__ == "__main__":
    main()
