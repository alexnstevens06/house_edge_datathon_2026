#!/usr/bin/env python
"""Write runs/SUMMARY.md from runs/*/metrics.json + official kit checks.

Usage: .venv/bin/python scripts/summarize.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs")


def kit_check(csv_path, split):
    cmd = ["python3", os.path.join(REPO, "kit", "check_submission.py"), csv_path,
           "--data", os.path.join(REPO, "kit", "public")]
    if split == "validation":
        cmd += ["--split", "validation"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def main() -> None:
    models = ["dense", "sparse"]
    metrics = {}
    for name in models:
        path = os.path.join(RUNS, name, "metrics.json")
        if os.path.exists(path):
            metrics[name] = json.load(open(path))
    baseline_path = os.path.join(RUNS, "baselines", "metrics.json")
    baselines = json.load(open(baseline_path)) if os.path.exists(baseline_path) else {}

    lines = ["# House Edge main round: matched dense/sparse transformers", "",
             "- Official scoring imported from `kit/slot_forecasting` (contest.py/scoring.py); never reimplemented.",
             "- Model selection on TUNE (half of validation episodes per condition, seed 0); CONFIRM reported once.",
             "- Both models: identical seeds/splits/steps/optimizer/batch/hyperparameters (d_model=64, 2 layers, 4 heads, ff=128, W=64, AdamW lr=3e-4, batch 64, 16 epochs); param counts equal by construction.",
             "- Training data: train-split sliding windows + train query cutoffs only. No validation data in training.", ""]

    lines += ["## Results (validation)", "",
              "| model | params | tokens/ex | peak VRAM MB | train s | epochs (best) | TUNE R | CONFIRM R | full-val R | pts@60 | pts@80 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    for name in models:
        if name not in metrics:
            continue
        m, c = metrics[name], metrics[name]["config"]
        rows.append((m["full_val_R"], f"| {name} | {c['params']} | {c['tokens_per_example']} | "
                     f"{c['peak_vram_mb']} | {c['train_seconds']} | {c['epochs']} ({m['best_epoch']}) | "
                     f"{fmt(m['tune_R'])} | {fmt(m['confirm_R'])} | {fmt(m['full_val_R'])} | "
                     f"{60/(1+m['full_val_R']):.2f} | {80/(1+m['full_val_R']):.2f} |"))
    for name, b in sorted(baselines.items(), key=lambda kv: kv[1]["full_val_R"]):
        full = b.get("full_val", {})
        rows.append((b["full_val_R"], f"| baseline:{name} | - | - | - | - | - | {fmt(b['tune_R'])} | "
                     f"{fmt(b['confirm_R'])} | {fmt(b['full_val_R'])} | {60/(1+b['full_val_R']):.2f} | "
                     f"{80/(1+b['full_val_R']):.2f} |"))
    rows.append((1.0, "| reference (frozen pooled) | - | - | - | - | - | 1.0000 | 1.0000 | 1.0000 | 30.00 | 40.00 |"))
    for _, row in sorted(rows):
        lines.append(row)

    lines += ["", "## Per-condition loss ratios (full validation, lower is better)", "",
              "| model | stable | short_history | drift | shared_change |", "| --- | --- | --- | --- | --- |"]
    for name in models:
        if name in metrics:
            r = metrics[name]["full_val"]["loss_ratio_by_condition"]
            lines.append(f"| {name} | {r['stable']:.4f} | {r['short_history']:.4f} | "
                         f"{r['drift']:.4f} | {r['shared_change']:.4f} |")
    if baselines:
        best = min(baselines.items(), key=lambda kv: kv[1]["full_val_R"])
        r = best[1]["full_val"]["loss_ratio_by_condition"]
        lines.append(f"| baseline:{best[0]} (best) | {r['stable']:.4f} | {r['short_history']:.4f} | "
                     f"{r['drift']:.4f} | {r['shared_change']:.4f} |")
    lines.append("| reference (frozen pooled) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |")

    lines += ["", "## Ablation: first-budget runs (16 epochs, ff=128, constant lr), superseded by TUNE selection", "",
              "| run | best epoch | TUNE R | CONFIRM R | full-val R | pts@60 |", "| --- | --- | --- | --- | --- | --- |"]
    for name in sorted(os.listdir(RUNS)):
        path = os.path.join(RUNS, name, "metrics.json")
        if name.startswith("archive_") and os.path.exists(path):
            m = json.load(open(path))
            lines.append(f"| {name} | {m['best_epoch']} | {fmt(m['tune_R'])} | {fmt(m['confirm_R'])} | "
                         f"{fmt(m['full_val_R'])} | {60/(1+m['full_val_R']):.2f} |")
    lines.append("")

    lines += ["", "## Official kit checks (python3 kit/check_submission.py ...)", "", "```"]
    for name in models:
        for split, csv in (("validation", "validation_predictions.csv"), ("test", "test_submission.csv")):
            path = os.path.join(RUNS, name, csv)
            if os.path.exists(path):
                out = kit_check(path, split)
                keep = [l for l in out.splitlines() if l.startswith(("VALID", "Validation points", "  "))]
                lines.append(f"# {name} {csv}")
                lines.extend(keep)
                lines.append("")
    lines += ["```"]

    lines += ["", "## Reproduce", "```sh",
              ".venv/bin/python -m unittest discover -s tests   # 12 tests, all pass",
              ".venv/bin/python scripts/train_model.py --arch dense --epochs 16",
              ".venv/bin/python scripts/train_model.py --arch sparse --epochs 16",
              ".venv/bin/python scripts/baselines.py",
              ".venv/bin/python scripts/summarize.py",
              "python3 kit/check_submission.py runs/<name>/test_submission.csv --data kit/public",
              "```",
              f"- Validation split episode lists: `runs/splits.json` (seed 0, stratified by condition).",
              f"- Data hash (sha256 kit/kit.zip): see `config.json` per run.",
              ""]

    with open(os.path.join(RUNS, "SUMMARY.md"), "w") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
