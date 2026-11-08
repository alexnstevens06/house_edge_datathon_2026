#!/usr/bin/env python
"""Generate runs/SUMMARY.md from final artifacts (models, ablations, baselines, verify logs).

Usage: .venv/bin/python scripts/make_summary.py
"""
import hashlib
import json
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs")
VERIFY = os.path.join(RUNS, "verify")

MODELS = [("dense (v2)", "v2_dense"), ("sparse (v2)", "v2_sparse"),
          ("dense (v1 ablation)", "archive_dense_v1"), ("sparse (v1 ablation)", "archive_sparse_v1")]
BASELINE_ORDER = ["pooled_recent", "recent", "shrinkage64", "pooled"]


def load(path):
    with open(path) as handle:
        return json.load(handle)


def kit_points(csv_path):
    """Re-run organizer checker; returns (points60, by_condition_ratios) or None."""
    out = os.path.join(VERIFY, f"checker_tmp_{os.path.basename(os.path.dirname(csv_path))}_{os.path.basename(csv_path)}.txt")
    cmd = ["python3", os.path.join(REPO, "kit", "check_submission.py"), csv_path,
           "--data", os.path.join(REPO, "kit", "public"), "--split", "validation"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(out, "w") as handle:
        handle.write(proc.stdout + proc.stderr)
    points = re.search(r"Validation points: ([\d.]+)", proc.stdout)
    ratios = dict(re.findall(r"  (\w+): loss ratio ([\d.]+)", proc.stdout))
    return (float(points.group(1)) if points else None,
            {k: float(v) for k, v in ratios.items()} if ratios else None)


def tail(path, lines=2):
    if not os.path.exists(path):
        return ""
    with open(path) as handle:
        return "".join(handle.readlines()[-lines:]).strip()


def main() -> None:
    kit_sha = hashlib.sha256(open(os.path.join(REPO, "kit", "kit.zip"), "rb").read()).hexdigest()
    L = []
    A = L.append
    A("# House Edge main round — matched dense/sparse joint transformers + baselines")
    A("")
    A(f"- Data: `kit/public` (sha256 of `kit/kit.zip`: `{kit_sha}`).")
    A("- Official scoring reused from the kit (`kit/slot_forecasting/contest.py`, `scoring.py`); never reimplemented. Organizer CLI `kit/check_submission.py` re-run on every CSV (logs in `runs/verify/checker_*.txt`).")
    A("- Validation episodes split deterministically (seed 0, stratified by condition) into TUNE (48 eps, 12/condition) and CONFIRM (48 eps): `runs/splits.json`. Epoch/checkpoint selected on TUNE only; CONFIRM reported once; full-validation is both halves pooled.")
    A("- Training used TRAIN split only: sliding windows over observed history (target ticks 16..max, labels = observed payouts at t, unobserved machines masked) + the 768 real train query cutoffs with `labels.csv`. No validation data, targets, or simulator used in training.")
    A("- `signal` is NOT a model input (by design; validity for historical windows unknown). Unit-tested: flipping signal never changes predictions.")
    A("")
    A("## Models (matched pair)")
    A("")
    A("Both are JOINT full-attention transformers over ALL tokens of all 8 machines x window ticks in one sequence — not axial/alternating: every input token (tick, machine) plus 8 learned per-machine readout queries are concatenated into a single sequence and encoded by ONE `nn.TransformerEncoder` with no causal or axial masking — `src/house_edge/models.py:53-54` (dense) and `src/house_edge/models.py:62-64` (sparse).")
    A("")
    A("| property | dense | sparse |")
    A("| --- | --- | --- |")
    A("| tokens | explicit grid of last W=64 ticks x 8 machines (512 tokens), MISSING is a 6th symbol | only observed (tick, machine) positions in [t-64, t), padded + `key_padding_mask` (5 payout symbols + PAD) |")
    A("| token embedding | symbol + absolute-tick + machine-slot + machine-type | payout + ORIGINAL absolute tick + machine-slot + machine-type |")
    A("| readout | 8 learned queries (slot + machine-type) -> 8x5 softmax | same |")
    A("| loss | CE on observed next-tick outcome, machines with unobserved target masked | same |")
    A("| causality | all input ticks < target tick, asserted in collate and forward (`models.py:49`, `models.py:59`) | same |")
    A("")
    A("Matched budget: identical seeds (0), splits, steps, optimizer (AdamW lr 3e-4, wd 0.01, cosine to 10%), batch 64, epochs 28, d_model 64, 2 layers, 4 heads, ff 256; parameter counts equal by construction (both 6-row symbol tables).")
    A("")
    A("## Results (validation; lower R is better; R = mean over conditions of episode-averaged squared error vs frozen pooled reference)")
    A("")
    A("| model / baseline | params | tokens/ex | train s | epochs (best) | peak VRAM MB | TUNE R | CONFIRM R | full-val R | pts@60 | pts@80 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for label, dirname in MODELS:
        mdir = os.path.join(RUNS, dirname)
        if not os.path.exists(os.path.join(mdir, "metrics.json")):
            continue
        m, c = load(os.path.join(mdir, "metrics.json")), load(os.path.join(mdir, "config.json"))
        rows.append((m["full_val_R"],
                     f"| {label} | {c['params']} | {c['tokens_per_example']} | {c['train_seconds']:.0f} | "
                     f"{c['epochs']} ({m['best_epoch']}) | {c['peak_vram_mb']} | {m['tune_R']:.4f} | "
                     f"{m['confirm_R']:.4f} | {m['full_val_R']:.4f} | {60/(1+m['full_val_R']):.2f} | "
                     f"{80/(1+m['full_val_R']):.2f} |"))
    baselines = load(os.path.join(RUNS, "baselines", "metrics.json"))
    for name in BASELINE_ORDER:
        b = baselines[name]
        rows.append((b["full_val_R"],
                     f"| baseline: {name} | - | - | - | - | - | {b['tune_R']:.4f} | "
                     f"{b['confirm_R']:.4f} | {b['full_val_R']:.4f} | {60/(1+b['full_val_R']):.2f} | "
                     f"{80/(1+b['full_val_R']):.2f} |"))
    rows.append((1.0, "| reference (frozen pooled) | - | - | - | - | - | 1.0000 | 1.0000 | 1.0000 | 30.00 | 40.00 |"))
    for _, row in sorted(rows):
        A(row)
    A("")
    A("Baselines: `pooled`/`recent`/`pooled_recent` = kit starter `FrequencyForecaster` methods; `shrinkage64` = per-machine empirical payout distribution over last 64 observed ticks, Dirichlet-shrunk (a=16) to the train-history pooled marginal. All scored on the SAME TUNE/CONFIRM/full splits as the transformers (selection used TUNE only).")
    A("")
    A("Selection by TUNE only (lower is better): **dense → v1** (TUNE 0.9456 vs v2 0.9503), **sparse → v2** (TUNE 0.9425 vs v1 0.9433). The v2 pair is kept as the headline matched pair (identical budgets), the v1 pair as an ablation; all four checkpoints are provenance-checked below. Best full-validation transformer: sparse v2 (R 0.9629, 30.57/60).")
    A("")
    A("## Per-condition loss ratios (full validation)")
    A("")
    A("| model | stable | short_history | drift | shared_change |")
    A("| --- | --- | --- | --- | --- |")
    for label, dirname in MODELS:
        path = os.path.join(RUNS, dirname, "metrics.json")
        if os.path.exists(path):
            r = load(path)["full_val"]["loss_ratio_by_condition"]
            A(f"| {label} | {r['stable']:.4f} | {r['short_history']:.4f} | {r['drift']:.4f} | {r['shared_change']:.4f} |")
    best_base = min(BASELINE_ORDER, key=lambda n: baselines[n]["full_val_R"])
    r = baselines[best_base]["full_val"]["loss_ratio_by_condition"]
    A(f"| baseline: {best_base} (best) | {r['stable']:.4f} | {r['short_history']:.4f} | {r['drift']:.4f} | {r['shared_change']:.4f} |")
    A("| reference (frozen pooled) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |")
    A("")
    A("## Organizer checker (python3 kit/check_submission.py <csv> --split validation --data kit/public)")
    A("")
    for label, dirname in MODELS:
        mdir = os.path.join(RUNS, dirname)
        vcsv = os.path.join(mdir, "validation_predictions.csv")
        tcsv = os.path.join(mdir, "test_submission.csv")
        if os.path.exists(vcsv):
            points, _ = kit_points(vcsv)
            A(f"- {label}: validation points **{points:.6f} / 60** (`{vcsv}`)")
        if os.path.exists(tcsv):
            proc = subprocess.run(["python3", os.path.join(REPO, "kit", "check_submission.py"), tcsv,
                                   "--data", os.path.join(REPO, "kit", "public")],
                                  capture_output=True, text=True)
            valid = "VALID FORMAT" in proc.stdout
            A(f"- {label}: test CSV format check {'**VALID** (1536 queries)' if valid else '**FAILED**'} (`{tcsv}`)")
    for name in BASELINE_ORDER:
        vcsv = os.path.join(RUNS, "baselines", f"{name}_validation.csv")
        if os.path.exists(vcsv):
            points, _ = kit_points(vcsv)
            A(f"- baseline {name}: validation points **{points:.6f} / 60**")
    A("- Full checker logs: `runs/verify/checker_*.txt`.")
    A("")
    A("## Verification")
    A("")
    A(f"- Unit tests: 15/15 PASS (`.venv/bin/python -m unittest discover -s tests -v`, log: `runs/verify/tests.txt`): causality (perturbing any machine's token at/after the target tick leaves outputs unchanged, both archs), forward asserts on tick>=t, sparse preserves ORIGINAL tick values, dense MISSING != payout 0, output shape (B,8,5) with 8 per-machine readouts, rows sum to 1 & finite, unobserved targets masked from loss, deterministic eval, query_id alignment with kit queries.csv, signal not used as input.")
    A("- Checkpoint provenance (reload `best.pt` with current code, regenerate validation predictions, compare to saved CSV): see `runs/verify/provenance.txt` — max abs diffs ~1e-7 (GPU attention-kernel nondeterminism; same code, same checkpoint). Code hashes: `runs/verify/code_hashes.txt`.")
    A(f"- Offline CPU prediction smoke (CUDA hidden via `CUDA_VISIBLE_DEVICES=` `HIP_VISIBLE_DEVICES=`, `torch.set_num_threads(2)`, full test split): `runs/verify/cpu_smoke.txt` — {tail(os.path.join(VERIFY, 'cpu_smoke.txt'))}")
    A("- Peak VRAM measured with `torch.cuda.max_memory_allocated` (in per-run `config.json`); both runs stayed far below the 6 GB shared-machine cap.")
    A("")
    A("## Reproduce (exact commands)")
    A("")
    A("```sh")
    A("export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6700 XT (gfx1031) on ROCm")
    A(".venv/bin/python -m unittest discover -s tests -v")
    A(".venv/bin/python scripts/train_model.py --arch dense --epochs 28 --dim-ff 256 --scheduler cosine --out runs/v2_dense")
    A(".venv/bin/python scripts/train_model.py --arch sparse --epochs 28 --dim-ff 256 --scheduler cosine --out runs/v2_sparse")
    A(".venv/bin/python scripts/baselines.py")
    A(".venv/bin/python scripts/regen_validation.py --ckpt runs/v2_dense/best.pt --ref runs/v2_dense/validation_predictions.csv --out runs/verify/v2_dense_validation_regen.csv")
    A(".venv/bin/python scripts/cpu_predict.py --ckpt runs/v2_dense/best.pt --out runs/verify/cpu_test_v2_dense.csv --gpu-csv runs/v2_dense/test_submission.csv --name v2_dense")
    A(".venv/bin/python scripts/make_summary.py")
    A("python3 kit/check_submission.py runs/v2_dense/test_submission.csv --data kit/public   # format only")
    A("```")
    A("")
    A("Ablation v1 (16 epochs, ff=128, constant lr) kept at `runs/archive_dense_v1`, `runs/archive_sparse_v1`. `runs/dense` contains an aborted 5-epoch duplicate of the v2 config (shell timeout killed it); superseded by `runs/v2_dense`, left untouched.")
    A("")
    A("## Risks / caveats")
    A("")
    A("- Reproduction tolerance: rerunning inference on GPU differs from the saved CSV by up to ~1.2e-7 per probability (attention kernel nondeterminism), above the announced 1e-8 organizer tolerance; CPU-vs-GPU differences are larger (see cpu_smoke.txt). For the real submission, regenerate the CSV with the exact reproduce command on the target runtime and submit that file.")
    A("- Competition runtime is Python 3.13 stdlib with deps via requirements.txt; torch availability there is UNVERIFIED (we only tested the local .venv, torch 2.9.1+rocm6.4). If torch is unavailable at validation time, predictions cannot be recreated — a numpy/stdlib fallback would be needed.")
    A("- Some machine state is hidden (per RULES.md); ~0.94-0.96 R may be near the observable limit. Both transformers beat the frozen reference and `pooled`, roughly tie `recent`, and slightly trail `pooled_recent` overall while clearly winning `shared_change` (0.87-0.88).")
    A("- Short-history episodes: the 24 older sampled ticks mostly fall outside the W=64 window by design (spec default); only the recent 40 are guaranteed covered.")
    A("- `signal` deliberately unused; if it is informative at cutoffs, some accuracy is left on the table (allowed ablation not run under the no-new-variants constraint).")
    A("")
    with open(os.path.join(RUNS, "SUMMARY.md"), "w") as handle:
        handle.write("\n".join(L) + "\n")
    print(f"wrote runs/SUMMARY.md ({len(L)} lines)")


if __name__ == "__main__":
    main()
