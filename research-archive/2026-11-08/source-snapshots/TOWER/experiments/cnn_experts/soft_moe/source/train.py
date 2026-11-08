"""Full soft_moe seed: matched control, then soft model. No VAL early stopping."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import soft_moe as S


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data, W, device):
    tensors, episodes = S.build_split_windows(data, W, "cutoff")
    probs = S.predict_probs(model, tensors, device)
    return S.to_preds(probs, episodes, data)


def run_kind(kind, train_data, val_data, test_data, args, device, out_dir):
    seed_all(args.seed)
    t0 = time.time()
    hist, _eps_h = S.build_split_windows(train_data, S.WIDTH, "history")
    cut, _eps_c = S.build_split_windows(train_data, S.WIDTH, "cutoff")
    S.assert_history_has_no_signal(hist)
    S.assert_cutoff_rows(cut, train_data)
    print(f"{kind} history {len(hist[0])} cutoff {len(cut[0])}", flush=True)
    model = S.build_model(kind)
    params = sum(p.numel() for p in model.parameters())
    print(f"{kind} params {params}", flush=True)
    pre_losses = S.train_phase(model, hist, args.epochs, S.LR, S.WD, S.BS, device)
    for i, loss in enumerate(pre_losses):
        print(f"{kind} pretrain {i} loss {loss:.4f} t {time.time() - t0:.0f}s", flush=True)
    ft_losses = S.train_phase(model, cut, args.ft_epochs, S.LR, S.WD, S.BS, device)
    for i, loss in enumerate(ft_losses):
        print(f"{kind} finetune {i} loss {loss:.4f} t {time.time() - t0:.0f}s", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": model.state_dict(), "kind": kind, "config": vars(args)}, os.path.join(out_dir, "checkpoint.pt"))
    # CPU fp32, 2 threads, no autocast. VAL labels are not used.
    torch.set_num_threads(2)
    model = model.to("cpu").eval()
    from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
    official = load_official_validation(os.path.join(S.PUB, "validation"))
    splits = json.load(open(S.SPLITS_PATH))
    pv = evaluate(model, val_data, S.WIDTH, "cpu")
    pt = evaluate(model, test_data, S.WIDTH, "cpu")
    tune = score_predictions(pv, official, set(splits["tune"]))
    full = score_predictions(pv, official, None)
    confirm = score_predictions(pv, official, set(splits["confirm"]))
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    record = {
        "kind": kind,
        "seed": args.seed,
        "param_count": params,
        "n_history_windows": int(len(hist[0])),
        "n_cutoff_windows": int(len(cut[0])),
        "pretrain_epochs": args.epochs,
        "finetune_epochs": args.ft_epochs,
        "pretrain_losses": pre_losses,
        "finetune_losses": ft_losses,
        "lr": S.LR,
        "wd": S.WD,
        "batch_size": S.BS,
        "tune": tune,
        "full": full,
        "confirm_seen": confirm,
        "confirm_label": "seen_not_pristine",
        "seconds": time.time() - t0,
        "peak_gpu_mb": peak,
        "device_train": device,
        "eval": "cpu_fp32",
    }
    json.dump(record, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    write_submission_csv(os.path.join(out_dir, "validation_predictions_cpu.csv"), pv)
    write_submission_csv(os.path.join(out_dir, "test_predictions_cpu.csv"), pt)
    print(f"RESULT {kind} seed {args.seed} tune_R {tune['R']:.6f} full_R {full['R']:.6f} sec {record['seconds']:.1f}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=S.PRETRAIN_EPOCHS)
    parser.add_argument("--ft-epochs", type=int, default=S.FINETUNE_EPOCHS)
    parser.add_argument("--which", choices=["both", "control", "soft"], default="both")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("refusing to train without CUDA (tower GPU is forbidden; server GPU required)")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    fraction = min(0.5, (2 * 1024 ** 3) / total_mem)
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.cuda.reset_peak_memory_stats()
    print(f"cuda {torch.cuda.get_device_name(0)} total_mib {total_mem / 2**20:.0f} fraction {fraction:.4f}", flush=True)
    out = args.out or os.path.join(HERE, "..", "runs", f"seed{args.seed}")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    train_data = S.load_split(os.path.join(S.PUB, "train"), with_labels=True)
    val_data = S.load_split(os.path.join(S.PUB, "validation"))
    test_data = S.load_split(os.path.join(S.PUB, "test"))
    if train_data.query_labels is None or len(train_data.query_labels) != len(train_data.queries):
        raise SystemExit("BLOCKER: TRAIN cutoff queries are not fully labeled")
    kinds = ["control", "soft"] if args.which == "both" else [args.which]
    records = {}
    for kind in kinds:
        records[kind] = run_kind(kind, train_data, val_data, test_data, args, device, os.path.join(out, kind))
        if device == "cuda":
            torch.cuda.empty_cache()
    root = os.path.join(HERE, "..")
    hashes = {
        "soft_moe.py": sha256_file(os.path.join(HERE, "soft_moe.py")),
        "train.py": sha256_file(os.path.join(HERE, "train.py")),
        "PLAN.md": sha256_file(os.path.join(root, "PLAN.md")),
        "SPLIT_IDS.json": sha256_file(os.path.join(root, "SPLIT_IDS.json")),
        "base_cnn.py": sha256_file(os.path.join(S.REPO, "experiments/mlp_cnn_hpo/source/base_cnn.py")),
    }
    summary = {
        "seed": args.seed,
        "records": {k: {"tune_R": v["tune"]["R"], "full_R": v["full"]["R"], "param_count": v["param_count"], "seconds": v["seconds"]} for k, v in records.items()},
        "incumbent_tune_R": S.INCUMBENT_TUNE_R.get(args.seed),
        "hashes": hashes,
        "vram_fraction": fraction,
    }
    if "soft" in records:
        inc = S.INCUMBENT_TUNE_R[args.seed]
        summary["soft_beats_incumbent"] = records["soft"]["tune"]["R"] < inc
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary["records"]), flush=True)


if __name__ == "__main__":
    main()
