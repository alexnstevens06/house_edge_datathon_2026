"""Signal-specialist networks on top of the frozen sparse backbone.

Rules honored here (see ../SPEC.md):
- Signal exists ONLY at the query cutoff per machine; history tokens carry no
  signal (the collator adds exactly one key, ``signals_q``, for the 8 readout
  queries).
- Fine-tuning uses ONLY train query examples; validation is used only for
  epoch selection on the TUNE subset and for reporting.
- Every input token tick < cutoff (backbone asserts preserved).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from house_edge.data import (N_MACHINES, PAYOUT_IDX_TO_VALUE, SplitData,
                             load_split, subset_split_data)
from house_edge.models import NextSpinTransformer, masked_ce_loss
from house_edge.scoring import load_official_validation, score_predictions
from house_edge.windows import Example, WindowCollator, query_examples

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONE = EXP_DIR / "runs" / "backbone" / "best.pt"
DEFAULT_KIT = REPO_ROOT / "kit" / "public"
DEFAULT_SPLITS = REPO_ROOT / "runs" / "splits.json"
SIGNALS = (-1, 0, 1)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. signals per episode/slot
# --------------------------------------------------------------------------- #

def load_signals(data: SplitData) -> dict[str, np.ndarray]:
    """episode_id -> (8,) int signal per slot (slot = Episode.slot_of(machine_id)).

    Works for any split (train/validation/test); every machine has one query row.
    """
    signals = {e.episode_id: np.zeros(N_MACHINES, dtype=np.int64) for e in data.episodes}
    for row in data.queries:
        episode = data.episodes[data.episode_index[row["episode_id"]]]
        slot = episode.slot_of(row["machine_id"])
        s = int(row["signal"])
        assert s in SIGNALS, f"unexpected signal {s}"
        signals[row["episode_id"]][slot] = s
    return signals


# --------------------------------------------------------------------------- #
# 2. SignalTransformer (backbone + zero-init signal embedding on readouts)
# --------------------------------------------------------------------------- #

class SignalTransformer(NextSpinTransformer):
    """Backbone plus nn.Embedding(3, d_model) ``signal_emb`` (zero-initialized),
    added to the 8 readout query vectors only (index = raw signal + 1)."""

    def __init__(self, arch: str = "sparse", **kwargs):
        super().__init__(arch, **kwargs)
        self.signal_emb = nn.Embedding(3, self.d_model)
        nn.init.zeros_(self.signal_emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        assert self.arch == "sparse", "SignalTransformer is sparse-only"
        signals_q = batch["signals_q"]
        assert signals_q.dim() == 2 and signals_q.shape[1] == N_MACHINES
        assert signals_q.dtype == torch.long
        assert ((signals_q >= 0) & (signals_q <= 2)).all(), "signals_q must be in {0,1,2}"
        symbols, ticks, target_t = batch["symbols"], batch["ticks"], batch["target_t"]
        assert (ticks < target_t[:, None]).all(), "sparse causality violated"
        x = self.symbol(symbols) + self.tick(ticks) + self.slot(batch["slots"]) + self.type_emb(batch["types"])
        q = (self.readout(batch["slots_q"]) + self.type_emb(batch["types_q"])
             + self.signal_emb(signals_q))
        x = torch.cat([x, q], dim=1)
        mask = torch.cat([batch["pad_mask"],
                          batch["pad_mask"].new_zeros(batch["pad_mask"].shape[0], N_MACHINES)], dim=1)
        h = self.enc(x, src_key_padding_mask=mask)
        return self.head(h[:, -N_MACHINES:])


def _model_kwargs(config: dict) -> dict:
    return {k: config[k] for k in ("d_model", "nhead", "num_layers", "dim_ff", "dropout")}


def load_backbone(backbone_path: str | Path = DEFAULT_BACKBONE) -> NextSpinTransformer:
    """Plain NextSpinTransformer with the backbone checkpoint loaded strictly."""
    ckpt = torch.load(backbone_path, map_location="cpu", weights_only=True)
    assert ckpt["arch"] == "sparse", "backbone must be sparse"
    model = NextSpinTransformer(ckpt["arch"], **_model_kwargs(ckpt["config"]))
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not missing and not unexpected, f"backbone load: missing={missing} unexpected={unexpected}"
    return model


def backbone_window(backbone_path: str | Path = DEFAULT_BACKBONE) -> int:
    ckpt = torch.load(backbone_path, map_location="cpu", weights_only=True)
    return int(ckpt["config"]["window"])


def load_model_from_checkpoint(path: str | Path) -> nn.Module:
    """Rebuild a fine-tuned model (SignalTransformer for 'shared', plain
    NextSpinTransformer for 'specialist') from a best.pt written by finetune()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    mode = ckpt.get("mode", "specialist")
    cls = SignalTransformer if mode == "shared" else NextSpinTransformer
    model = cls(ckpt["arch"], **_model_kwargs(ckpt["config"]))
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    if mode == "shared":
        assert "signal_emb.weight" in ckpt["state_dict"], "shared checkpoint must carry signal_emb"
    return model


# --------------------------------------------------------------------------- #
# 3. collation: sparse WindowCollator + signals_q (query readouts only)
# --------------------------------------------------------------------------- #

class SignalCollator:
    """Wraps a sparse WindowCollator; adds ONLY ``signals_q`` = raw signal + 1
    (long, in {0,1,2}) for the 8 readout queries of each example's episode."""

    def __init__(self, base: WindowCollator, signals: dict[str, np.ndarray]):
        assert base.arch == "sparse"
        self.base = base
        self.signals = signals

    @property
    def episodes(self):
        return self.base.episodes

    @property
    def arch(self):
        return self.base.arch

    @property
    def window(self):
        return self.base.window

    def __call__(self, batch: list[Example]) -> dict:
        out = self.base(batch)
        sig = np.stack([self.signals[self.base.episodes[ex.ep].episode_id] for ex in batch])
        out["signals_q"] = torch.from_numpy(sig.astype(np.int64) + 1)
        return out


# --------------------------------------------------------------------------- #
# prediction helpers (per-episode probabilities, query predictions, routing)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def episode_probabilities(model, data: SplitData, device: str, window: int = 64,
                          batch_size: int = 32, signals: dict | None = None) -> dict[str, np.ndarray]:
    """episode_id -> (8,5) next-spin probabilities at the cutoff (one joint
    forward per episode). ``signals`` (raw {-1,0,1} dict) adds signals_q for
    SignalTransformer models."""
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    arch = getattr(model, "arch", "sparse")
    collator = WindowCollator(data.episodes, arch, window)
    if signals is not None:
        collator = SignalCollator(collator, signals)
    by_episode: dict[str, list[dict[str, str]]] = {}
    for row in data.queries:
        by_episode.setdefault(row["episode_id"], []).append(row)
    examples: list[Example] = []
    order: list[str] = []
    for episode_id in sorted(by_episode):
        cutoffs = {int(r["cutoff"]) for r in by_episode[episode_id]}
        assert len(cutoffs) == 1, f"{episode_id}: multiple cutoffs"
        examples.append(Example(data.episode_index[episode_id], cutoffs.pop(), None))
        order.append(episode_id)
    probs: dict[str, np.ndarray] = {}
    use_amp = device == "cuda"
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        batch = collator(chunk)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model.forward(batch)
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for i, episode_id in enumerate(order[start:start + batch_size]):
            probs[episode_id] = p[i]
    if was_training and hasattr(model, "train"):
        model.train()
    return probs


def episode_probs_to_predictions(data: SplitData, probs: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Map per-episode (8,5) probabilities to {query_id: [p0,p1,p2,p5,p20]}."""
    out: dict[str, list[float]] = {}
    for row in data.queries:
        episode = data.episodes[data.episode_index[row["episode_id"]]]
        slot = episode.slot_of(row["machine_id"])
        out[row["query_id"]] = [float(v) for v in probs[row["episode_id"]][slot]]
    return out


def predict_queries(model, data: SplitData, device: str, window: int = 64,
                    batch_size: int = 32, signals: dict | None = None) -> dict[str, list[float]]:
    return episode_probs_to_predictions(
        data, episode_probabilities(model, data, device, window, batch_size, signals))


# --------------------------------------------------------------------------- #
# 5. routed prediction over the three signal specialists
# --------------------------------------------------------------------------- #

def model_name_for_signal(signal: int) -> str:
    return f"specialist_signal_{int(signal):+d}"


def predict_routed(models_by_signal: dict[int, nn.Module], data: SplitData, signals: dict[str, np.ndarray],
                   device: str, window: int = 64, batch_size: int = 32
                   ) -> tuple[dict[str, list[float]], dict[str, dict]]:
    """Route each machine slot to the specialist matching that machine's signal.

    Returns ({query_id: 5 probs}, {query_id: {"signal": s, "model": name}}).
    """
    assert sorted(models_by_signal) == list(SIGNALS), "need specialists for signals -1,0,1"
    by_sig = {s: episode_probabilities(models_by_signal[s], data, device, window, batch_size)
              for s in SIGNALS}
    combined: dict[str, np.ndarray] = {}
    for episode in data.episodes:
        sig = signals[episode.episode_id]
        assert set(np.unique(sig).tolist()) <= set(SIGNALS)
        prob = np.empty((N_MACHINES, 5), dtype=np.float64)
        for slot in range(N_MACHINES):
            prob[slot] = by_sig[int(sig[slot])][episode.episode_id][slot]
        combined[episode.episode_id] = prob
    preds = episode_probs_to_predictions(data, combined)
    routes: dict[str, dict] = {}
    for row in data.queries:
        episode = data.episodes[data.episode_index[row["episode_id"]]]
        slot = episode.slot_of(row["machine_id"])
        s = int(signals[row["episode_id"]][slot])
        assert int(row["signal"]) == s, "query row signal disagrees with signals dict"
        routes[row["query_id"]] = {"signal": s, "model": model_name_for_signal(s)}
        # every query used exactly its own signal's model:
        assert np.allclose(np.asarray(preds[row["query_id"]]),
                           by_sig[s][row["episode_id"]][slot], atol=1e-9)
    return preds, routes


def routed_with_backbone(candidate: nn.Module, candidate_signal: int, backbone: nn.Module,
                         data: SplitData, signals: dict[str, np.ndarray], device: str,
                         window: int = 64, batch_size: int = 32,
                         base_probs: dict[str, np.ndarray] | None = None) -> dict[str, list[float]]:
    """Combined prediction used for specialist epoch selection: machines whose
    signal == candidate_signal use the candidate, others use the backbone
    (no-signal) prediction. ``base_probs`` optionally caches the backbone's
    per-episode probabilities."""
    cand = episode_probabilities(candidate, data, device, window, batch_size)
    base = base_probs if base_probs is not None else \
        episode_probabilities(backbone, data, device, window, batch_size)
    combined: dict[str, np.ndarray] = {}
    for episode in data.episodes:
        sig = signals[episode.episode_id]
        mix = base[episode.episode_id].astype(np.float64).copy()
        sel = sig == candidate_signal
        mix[sel] = cand[episode.episode_id][sel]
        combined[episode.episode_id] = mix
    return episode_probs_to_predictions(data, combined)


# --------------------------------------------------------------------------- #
# shared context (data + splits + signals), loaded once per process
# --------------------------------------------------------------------------- #

@dataclass
class Context:
    train_data: SplitData
    val_data: SplitData
    test_data: SplitData
    official: tuple
    tune: list[str]
    confirm: list[str]
    signals_train: dict[str, np.ndarray]
    signals_val: dict[str, np.ndarray]
    signals_test: dict[str, np.ndarray]


def load_context(kit_dir: str | Path = DEFAULT_KIT, splits_path: str | Path = DEFAULT_SPLITS) -> Context:
    kit_dir = Path(kit_dir)
    train_data = load_split(kit_dir / "train", with_labels=True)
    val_data = load_split(kit_dir / "validation")
    test_data = load_split(kit_dir / "test")
    official = load_official_validation(kit_dir / "validation")
    splits = json.loads(Path(splits_path).read_text())
    return Context(train_data, val_data, test_data, official,
                   list(splits["tune"]), list(splits["confirm"]),
                   load_signals(train_data), load_signals(val_data), load_signals(test_data))


def limit_episodes_stratified(episode_ids: list[str], official: tuple, limit: int | None) -> list[str]:
    """First ``limit`` tune/confirm episodes keeping every condition represented
    (the official scorer requires all conditions to occur)."""
    if limit is None or limit >= len(episode_ids):
        return list(episode_ids)
    _, groups, _, queries = official
    condition: dict[str, str] = {}
    for query_id, episode_id in queries.items():
        condition[episode_id] = groups[query_id]
    by_condition: dict[str, list[str]] = {}
    for episode_id in episode_ids:
        by_condition.setdefault(condition[episode_id], []).append(episode_id)
    pools = [list(v) for _, v in sorted(by_condition.items())]
    out: list[str] = []
    while len(out) < limit and any(pools):
        for pool in pools:
            if pool and len(out) < limit:
                out.append(pool.pop(0))
    return sorted(out)


# --------------------------------------------------------------------------- #
# 4. fine-tuning (shared signal model / per-signal specialists)
# --------------------------------------------------------------------------- #

def specialist_mask(label_mask: torch.Tensor, signals_q: torch.Tensor, signal: int) -> torch.Tensor:
    """label_mask & (machine's own signal == signal). signals_q in {0,1,2}."""
    return label_mask & (signals_q == int(signal) + 1)


class _ExampleDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Example:
        return self.examples[idx]


def finetune(mode: str, out_dir: str | Path, signal: int | None = None, epochs: int = 30,
             lr: float = 1e-4, batch_size: int = 16, seed: int = 0, *,
             backbone_path: str | Path = DEFAULT_BACKBONE, context: Context | None = None,
             device: str | None = None, train_episode_limit: int | None = None,
             tune_episode_limit: int | None = None, log=print) -> tuple[nn.Module, dict]:
    """Fine-tune from the frozen backbone on train query examples only.

    mode "shared": SignalTransformer, loss on all 8 machines of each episode.
    mode "specialist": plain backbone copy, loss mask = labels of machines whose
    own signal == ``signal`` (all 8 machines' history stays in context; batches
    with zero masked labels are skipped). Epoch selection on the TUNE subset
    (specialists: routed combined tune score vs the no-signal backbone).
    """
    assert mode in ("shared", "specialist")
    if mode == "specialist":
        assert signal in SIGNALS, "specialist needs signal in {-1,0,1}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(2)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = context if context is not None else load_context()
    window = backbone_window(backbone_path)

    train_data = ctx.train_data
    if train_episode_limit is not None:
        keep = sorted(e.episode_id for e in train_data.episodes)[:train_episode_limit]
        train_data = subset_split_data(train_data, keep)
    examples = query_examples(train_data.episodes, train_data.episode_index,
                              train_data.queries, train_data.query_labels)
    assert examples, "no train query examples"
    collator = SignalCollator(WindowCollator(train_data.episodes, "sparse", window), ctx.signals_train)
    loader = DataLoader(_ExampleDataset(examples), batch_size=batch_size, shuffle=True,
                        collate_fn=collator, generator=torch.Generator().manual_seed(seed))

    ckpt = torch.load(backbone_path, map_location="cpu", weights_only=True)
    if mode == "shared":
        model = SignalTransformer(ckpt["arch"], **_model_kwargs(ckpt["config"]))
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        assert not unexpected and missing == ["signal_emb.weight"], f"missing={missing} unexpected={unexpected}"
    else:
        model = NextSpinTransformer(ckpt["arch"], **_model_kwargs(ckpt["config"]))
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    tune_ids = limit_episodes_stratified(ctx.tune, ctx.official, tune_episode_limit)
    tune_data = subset_split_data(ctx.val_data, tune_ids)
    tune_set = set(tune_ids)
    backbone = None
    backbone_tune_probs = None
    if mode == "specialist":
        backbone = load_backbone(backbone_path).to(device)
        backbone_tune_probs = episode_probabilities(backbone, tune_data, device, window)

    def tune_score() -> float:
        if mode == "shared":
            preds = predict_queries(model, tune_data, device, window, signals=ctx.signals_val)
        else:
            preds = routed_with_backbone(model, signal, backbone, tune_data, ctx.signals_val,
                                         device, window, base_probs=backbone_tune_probs)
        return score_predictions(preds, ctx.official, episode_subset=tune_set)["R"]

    use_amp = device == "cuda"
    started = time.perf_counter()
    best = {"tune_R": float("inf"), "epoch": -1}
    train_seconds = 0.0
    steps_done = 0
    log_path = out_dir / "log.jsonl"
    with log_path.open("w") as log_file:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_start, running, running_n, skipped = time.perf_counter(), 0.0, 0, 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mask = batch["label_mask"]
                if mode == "specialist":
                    mask = specialist_mask(mask, batch["signals_q"], signal)
                    if not bool(mask.any()):
                        skipped += 1
                        continue
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model.forward(batch)
                    loss = masked_ce_loss(logits.float(), batch["labels"], mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps_done += 1
                running += float(loss.detach())
                running_n += 1
            train_seconds += time.perf_counter() - epoch_start
            tune_R = tune_score()
            entry = {"epoch": epoch, "train_loss": running / max(1, running_n),
                     "skipped_batches": skipped, "tune_R": tune_R,
                     "epoch_seconds": round(time.perf_counter() - epoch_start, 2)}
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            log(f"[{mode}{'' if signal is None else f' sig {signal:+d}'}] "
                f"epoch {epoch}: loss {entry['train_loss']:.4f} tune_R {tune_R:.4f}")
            if tune_R < best["tune_R"]:
                best = {"tune_R": tune_R, "epoch": epoch}
                torch.save({"state_dict": model.state_dict(), "arch": ckpt["arch"],
                            "config": {**ckpt["config"], "window": window},
                            "mode": mode, "signal": signal,
                            "backbone_path": str(backbone_path),
                            "backbone_sha256": sha256_file(backbone_path),
                            "best_epoch": epoch, "tune_R": tune_R},
                           out_dir / "best.pt")

    model = load_model_from_checkpoint(out_dir / "best.pt").to(device)
    info = {"mode": mode, "signal": signal, "out_dir": str(out_dir),
            "backbone_path": str(backbone_path), "backbone_sha256": sha256_file(backbone_path),
            "epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed,
            "train_examples": len(examples), "steps": steps_done,
            "best_epoch": best["epoch"], "best_tune_R": best["tune_R"],
            "tune_episodes_for_selection": len(tune_ids),
            "train_seconds": round(train_seconds, 2),
            "wall_seconds": round(time.perf_counter() - started, 2),
            "device": device,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device == "cuda" else 0.0}
    (out_dir / "finetune_summary.json").write_text(json.dumps(info, indent=2) + "\n")
    return model, info


# --------------------------------------------------------------------------- #
# 7. label counts
# --------------------------------------------------------------------------- #

def label_counts(ctx: Context) -> dict:
    by_sig_type = {s: {} for s in SIGNALS}
    by_sig_outcome = {s: {} for s in SIGNALS}
    for row in ctx.train_data.queries:
        s = int(row["signal"])
        by_sig_type[s][row["machine_type"]] = by_sig_type[s].get(row["machine_type"], 0) + 1
        outcome = str(PAYOUT_IDX_TO_VALUE[ctx.train_data.query_labels[row["query_id"]]])
        by_sig_outcome[s][outcome] = by_sig_outcome[s].get(outcome, 0) + 1
    counts = {
        "train_labels_by_signal_and_machine_type": {str(s): by_sig_type[s] for s in SIGNALS},
        "train_labels_by_signal_and_outcome": {str(s): by_sig_outcome[s] for s in SIGNALS},
        "train_label_totals_by_signal": {str(s): sum(by_sig_type[s].values()) for s in SIGNALS},
        "val_queries_by_signal": {str(s): sum(1 for r in ctx.val_data.queries if int(r["signal"]) == s)
                                  for s in SIGNALS},
        "test_queries_by_signal": {str(s): sum(1 for r in ctx.test_data.queries if int(r["signal"]) == s)
                                   for s in SIGNALS},
        "train_episodes": len(ctx.train_data.episodes),
        "val_episodes": len(ctx.val_data.episodes),
        "test_episodes": len(ctx.test_data.episodes),
        "val_queries": len(ctx.val_data.queries),
        "test_queries": len(ctx.test_data.queries),
    }
    assert sum(counts["train_label_totals_by_signal"].values()) == len(ctx.train_data.queries)
    assert sum(counts["val_queries_by_signal"].values()) == len(ctx.val_data.queries)
    assert sum(counts["test_queries_by_signal"].values()) == len(ctx.test_data.queries)
    return counts


# --------------------------------------------------------------------------- #
# 6. per-system artifacts (predictions, submission, metrics, route log)
# --------------------------------------------------------------------------- #

def score_system(preds: dict[str, list[float]], ctx: Context) -> dict:
    return {
        "tune": score_predictions(preds, ctx.official, episode_subset=set(ctx.tune)),
        "confirm": score_predictions(preds, ctx.official, episode_subset=set(ctx.confirm)),
        "full_val": score_predictions(preds, ctx.official, episode_subset=None),
    }


def write_route_log(path: str | Path, routes: dict[str, dict]) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("query_id", "signal", "model_used"))
        for query_id in sorted(routes):
            r = routes[query_id]
            writer.writerow((query_id, r["signal"], r["model"]))
