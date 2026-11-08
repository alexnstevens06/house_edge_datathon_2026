"""CLI: label counts, backbone eval, shared + 3 specialists, routed eval, artifacts.

Usage:
    python source/run_all.py [--out runs/v1] [--epochs 30] [--backbone runs/backbone/best.pt]
    python source/run_all.py --smoke          # 1 epoch, tiny train/tune subsets
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import signal_nets as SN  # noqa: E402
from house_edge.data import data_sha256  # noqa: E402
from house_edge.scoring import write_submission_csv  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Signal-specialist networks on the sparse backbone")
    ap.add_argument("--out", default=None, help="output dir (default: runs/v1, or runs/smoke with --smoke)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--backbone", default=str(SN.DEFAULT_BACKBONE),
                    help=f"path to backbone checkpoint (default: {SN.DEFAULT_BACKBONE})")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a tiny subset (dev check)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kit", default=str(SN.DEFAULT_KIT))
    ap.add_argument("--splits", default=str(SN.DEFAULT_SPLITS))
    ap.add_argument("--train-episode-limit", type=int, default=None)
    ap.add_argument("--tune-episode-limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(EXP_DIR / ("runs/smoke" if args.smoke else "runs/v1"))
    if args.smoke:
        args.epochs = 1
        args.train_episode_limit = args.train_episode_limit or 8
        args.tune_episode_limit = args.tune_episode_limit or 8
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "runs").mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    start_iso = datetime.now().isoformat(timespec="seconds")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, (4 * 2 ** 30) / total))
        torch.cuda.reset_peak_memory_stats()

    ctx = SN.load_context(args.kit, args.splits)
    window = SN.backbone_window(args.backbone)
    backbone_sha = SN.sha256_file(args.backbone)

    # ---- 7. label counts -> runs/label_counts.json -------------------------------- #
    counts = SN.label_counts(ctx)
    (EXP_DIR / "runs" / "label_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print("label counts:", json.dumps(counts["train_labels_by_signal_and_outcome"]), flush=True)

    systems: dict[str, dict] = {}

    def eval_simple_system(name: str, model, signals=None, extra: dict | None = None) -> dict:
        """Predict val + test with one model; write CSVs and metrics.json."""
        system_dir = out_dir / name
        system_dir.mkdir(parents=True, exist_ok=True)
        preds = SN.predict_queries(model, ctx.val_data, device, window, signals=signals)
        scores = SN.score_system(preds, ctx)
        write_submission_csv(system_dir / "validation_predictions.csv", preds)
        test_preds = SN.predict_queries(model, ctx.test_data, device, window, signals=signals)
        write_submission_csv(system_dir / "test_submission.csv", test_preds)
        metrics = {"system": name, **scores,
                   "backbone_path": str(args.backbone), "backbone_sha256": backbone_sha,
                   "window": window, "device": device, **(extra or {})}
        (system_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        entry = {"system": name, "tune_R": scores["tune"]["R"],
                 "confirm_R": scores["confirm"]["R"], "full_val_R": scores["full_val"]["R"]}
        systems[name] = entry
        print(json.dumps(entry), flush=True)
        return metrics

    # ---- backbone (no signal) ----------------------------------------------------- #
    backbone = SN.load_backbone(args.backbone).to(device)
    eval_simple_system("backbone_nosignal", backbone)

    # ---- shared signal model ------------------------------------------------------ #
    shared_model, shared_info = SN.finetune(
        "shared", out_dir / "shared", epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed, backbone_path=args.backbone,
        context=ctx, device=device, train_episode_limit=args.train_episode_limit,
        tune_episode_limit=args.tune_episode_limit)
    signals_all = {**ctx.signals_val, **ctx.signals_test}
    eval_simple_system("shared", shared_model, signals=signals_all,
                       extra={"finetune": shared_info})

    # ---- per-signal specialists --------------------------------------------------- #
    specialists: dict[int, object] = {}
    for signal in SN.SIGNALS:
        model, info = SN.finetune(
            "specialist", out_dir / SN.model_name_for_signal(signal), signal=signal,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
            backbone_path=args.backbone, context=ctx, device=device,
            train_episode_limit=args.train_episode_limit,
            tune_episode_limit=args.tune_episode_limit)
        specialists[signal] = model
        eval_simple_system(SN.model_name_for_signal(signal), model,
                           extra={"finetune": info})

    # ---- routed specialists (final system) ---------------------------------------- #
    routed_dir = out_dir / "specialists_routed"
    routed_dir.mkdir(parents=True, exist_ok=True)
    val_preds, val_routes = SN.predict_routed(specialists, ctx.val_data, ctx.signals_val, device, window)
    test_preds, test_routes = SN.predict_routed(specialists, ctx.test_data, ctx.signals_test, device, window)
    write_submission_csv(routed_dir / "validation_predictions.csv", val_preds)
    write_submission_csv(routed_dir / "test_submission.csv", test_preds)
    SN.write_route_log(routed_dir / "route_log.csv", {**val_routes, **test_routes})
    scores = SN.score_system(val_preds, ctx)
    route_counts = {}
    for route in val_routes.values():
        route_counts[route["model"]] = route_counts.get(route["model"], 0) + 1
    routed_metrics = {"system": "specialists_routed", **scores,
                      "backbone_path": str(args.backbone), "backbone_sha256": backbone_sha,
                      "window": window, "device": device,
                      "route_counts_val": route_counts,
                      "specialists": {SN.model_name_for_signal(s): str(out_dir / SN.model_name_for_signal(s) / "best.pt")
                                      for s in SN.SIGNALS}}
    (routed_dir / "metrics.json").write_text(json.dumps(routed_metrics, indent=2) + "\n")
    systems["specialists_routed"] = {"system": "specialists_routed",
                                     "tune_R": scores["tune"]["R"],
                                     "confirm_R": scores["confirm"]["R"],
                                     "full_val_R": scores["full_val"]["R"]}
    print(json.dumps(systems["specialists_routed"]), flush=True)

    # ---- summary + provenance ------------------------------------------------------ #
    wall_seconds = time.perf_counter() - started
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device == "cuda" else 0.0
    summary = {"out_dir": str(out_dir), "device": device, "epochs": args.epochs,
               "smoke": bool(args.smoke), "seed": args.seed, "lr": args.lr,
               "batch_size": args.batch_size, "window": window,
               "backbone_path": str(args.backbone), "backbone_sha256": backbone_sha,
               "systems": systems, "wall_seconds": round(wall_seconds, 2),
               "peak_vram_mb": peak_vram_mb,
               "label_counts": (EXP_DIR / "runs" / "label_counts.json").name,
               "finetune": {"shared": shared_info["best_epoch"],
                            **{SN.model_name_for_signal(s): None for s in SN.SIGNALS}}}
    for signal in SN.SIGNALS:
        info_path = out_dir / SN.model_name_for_signal(signal) / "finetune_summary.json"
        summary["finetune"][SN.model_name_for_signal(signal)] = \
            json.loads(info_path.read_text())["best_epoch"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    source_sha = {}
    for pattern_dir in ("source", "tests"):
        for py in sorted((EXP_DIR / pattern_dir).glob("*.py")):
            source_sha[str(py.relative_to(EXP_DIR))] = SN.sha256_file(py)
    provenance = {"start_time": start_iso,
                  "end_time": datetime.now().isoformat(timespec="seconds"),
                  "wall_seconds": round(wall_seconds, 2),
                  "command": sys.argv.copy(),
                  "torch_version": torch.__version__,
                  "numpy_version": np.__version__,
                  "python_version": platform.python_version(),
                  "device": device, "peak_vram_mb": peak_vram_mb,
                  "backbone_path": str(args.backbone), "backbone_sha256": backbone_sha,
                  "kit_zip_sha256": data_sha256(REPO_ROOT / "kit"),
                  "source_sha256": source_sha,
                  "out_dir": str(out_dir),
                  "env": {k: os.environ.get(k) for k in
                          ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}}
    (EXP_DIR / "runs" / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
