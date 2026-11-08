"""Training loop shared by the dense and sparse models."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import SplitData, data_sha256, load_split, subset_split_data
from .models import NextSpinTransformer, count_parameters, masked_ce_loss
from .predict import predict_split
from .scoring import load_official_validation, score_predictions, write_submission_csv
from .split import tune_confirm_split
from .windows import Example, T_START, WindowCollator


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Example:
        return self.examples[idx]


def _tokens_per_example(episodes, examples, arch, window) -> float:
    counts = []
    for ex in examples:
        episode = episodes[ex.ep]
        lo = max(0, ex.t - window)
        if arch == "dense":
            counts.append((ex.t - lo) * 8 + 8)
        else:
            counts.append(int((episode.obs[lo:ex.t] >= 0).sum()) + 8)
    return float(np.mean(counts))


def train_model(arch: str, out_dir: str | Path, *, kit_dir: str | Path, epochs: int = 12,
                seed: int = 0, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                dim_ff: int = 128, dropout: float = 0.1, window: int = 64, lr: float = 3e-4,
                weight_decay: float = 0.01, batch_size: int = 64, t_start: int = T_START,
                pilot: bool = False, device: str | None = None, log_every: int = 50,
                scheduler: str = "none") -> dict:
    """Train one model, select on TUNE, report CONFIRM once, write artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    public = Path(kit_dir) / "public"
    train_data = load_split(public / "train", with_labels=True)
    val_data = load_split(public / "validation")
    test_data = load_split(public / "test")
    official = load_official_validation(public / "validation")
    split = tune_confirm_split(public / "validation")
    splits_path = out_dir.parent / "splits.json"
    if not splits_path.exists():
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(split, indent=2) + "\n")
    tune_data = subset_split_data(val_data, split["tune"])
    confirm_data = subset_split_data(val_data, split["confirm"])

    from .windows import query_examples, sliding_examples
    sliding = sliding_examples(train_data.episodes, t_start=t_start)
    queries = query_examples(train_data.episodes, train_data.episode_index,
                             train_data.queries, train_data.query_labels)
    examples = sliding + queries
    if pilot:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(examples), size=min(1500, len(examples)), replace=False)
        examples = [examples[i] for i in sorted(idx)]
        epochs = 1
    tokens_per_example = _tokens_per_example(train_data.episodes, examples, arch, window)

    collator = WindowCollator(train_data.episodes, arch, window)
    loader = DataLoader(ExampleDataset(examples), batch_size=batch_size, shuffle=True,
                        collate_fn=collator, generator=torch.Generator().manual_seed(seed))

    model = NextSpinTransformer(arch, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                dim_ff=dim_ff, dropout=dropout).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(examples) + batch_size - 1) // batch_size
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch, eta_min=0.1 * lr)

    log_path = out_dir / "log.jsonl"
    best = {"tune_R": float("inf"), "epoch": -1}
    steps_done = 0
    train_seconds = 0.0

    def evaluate_tune() -> float:
        preds = predict_split(model, tune_data, device, window=window)
        return score_predictions(preds, official, episode_subset=set(split["tune"]))["R"]

    with log_path.open("w") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_start, running, running_n = time.perf_counter(), 0.0, 0
            for step, batch in enumerate(loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                amp = os.environ.get("HE_AMP", "none")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp == "bf16" and device == "cuda")):
                    logits = model.forward(batch)
                loss = masked_ce_loss(logits.float(), batch["labels"], batch["label_mask"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if sched is not None:
                    sched.step()
                steps_done += 1
                running += float(loss.detach())
                running_n += 1
                if pilot and step % 10 == 0:
                    log.write(json.dumps({"step": steps_done, "train_loss": running / running_n}) + "\n")
                    log.flush()
            epoch_seconds = time.perf_counter() - epoch_start
            train_seconds += epoch_seconds
            tune_R = evaluate_tune()
            entry = {"epoch": epoch, "train_loss": running / max(1, running_n),
                     "tune_R": tune_R, "epoch_seconds": round(epoch_seconds, 2)}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(f"[{arch}] epoch {epoch}: loss {entry['train_loss']:.4f} tune_R {tune_R:.4f}", flush=True)
            if tune_R < best["tune_R"]:
                best = {"tune_R": tune_R, "epoch": epoch}
                torch.save({"state_dict": model.state_dict(), "arch": arch,
                            "config": {"d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                                       "dim_ff": dim_ff, "dropout": dropout, "window": window}},
                           out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True)["state_dict"])
    model.to(device).eval()

    def full_eval(data: SplitData, subset=None):
        preds = predict_split(model, data, device, window=window)
        result = score_predictions(preds, official, episode_subset=subset)
        return result, preds

    tune_result, _ = full_eval(tune_data, set(split["tune"]))
    confirm_result, _ = full_eval(confirm_data, set(split["confirm"]))
    full_result, full_preds = full_eval(val_data, None)
    write_submission_csv(out_dir / "validation_predictions.csv", full_preds)
    test_preds = predict_split(model, test_data, device, window=window)
    write_submission_csv(out_dir / "test_submission.csv", test_preds)

    wall_seconds = time.perf_counter() - started
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0
    throughput = len(examples) * epochs / max(train_seconds, 1e-9)

    config = {"arch": arch, "seed": seed, "d_model": d_model, "nhead": nhead,
              "num_layers": num_layers, "dim_ff": dim_ff, "dropout": dropout,
              "window": window, "lr": lr, "weight_decay": weight_decay,
              "batch_size": batch_size, "t_start": t_start, "epochs": epochs,
              "scheduler": scheduler, "amp": os.environ.get("HE_AMP", "none"), "pilot": pilot, "params": n_params, "train_examples": len(examples),
              "sliding_examples": len(sliding), "query_examples": len(queries),
              "tokens_per_example": round(tokens_per_example, 2),
              "steps_total": steps_done, "device": device,
              "torch_version": torch.__version__, "data_sha256_kit_zip": data_sha256(kit_dir),
              "train_seconds": round(train_seconds, 2), "wall_seconds": round(wall_seconds, 2),
              "peak_vram_mb": round(peak_vram_mb, 1),
              "throughput_examples_per_sec": round(throughput, 1),
              "best_epoch": best["epoch"]}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics = {"arch": arch, "best_epoch": best["epoch"],
               "tune_R": tune_result["R"], "confirm_R": confirm_result["R"],
               "full_val_R": full_result["R"],
               "tune": tune_result, "confirm": confirm_result, "full_val": full_result,
               "config": config}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
