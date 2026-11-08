"""CPU-only, small tests for the signal-specialist networks (see ../SPEC.md)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP_DIR / "source"))

import signal_nets as SN  # noqa: E402
from house_edge.data import (N_MACHINES, Episode, SplitData, read_csv_rows,  # noqa: E402
                             subset_split_data)
from house_edge.models import NextSpinTransformer  # noqa: E402
from house_edge.scoring import write_submission_csv  # noqa: E402
from house_edge.windows import Example, WindowCollator, query_examples, sliding_examples  # noqa: E402

MACHINE_TYPES = ("balanced", "jackpot", "steady")
MIXED = np.array([-1, 0, 1, -1, 0, 1, -1, 0], dtype=np.int64)


def make_episode(episode_id: str, seed: int = 0, n_ticks: int = 24) -> Episode:
    rng = np.random.default_rng(seed)
    machine_ids = tuple(f"{episode_id}-m{i:02d}" for i in range(N_MACHINES))
    machine_types = rng.integers(0, 3, size=N_MACHINES).astype(np.int64)
    obs = rng.choice(np.array([-1, 0, 1, 2, 3, 4], dtype=np.int8), size=(n_ticks, N_MACHINES),
                     p=[0.4, 0.3, 0.15, 0.1, 0.04, 0.01])
    return Episode(episode_id, machine_ids, machine_types, obs, cutoff=n_ticks)


def make_split_data(episodes: list[Episode], signals: dict[str, np.ndarray] | None = None,
                    with_labels: bool = False) -> SplitData:
    queries, labels = [], {}
    for ep in episodes:
        for slot, machine_id in enumerate(ep.machine_ids):
            query_id = f"q-{ep.episode_id}-{slot}"
            signal = int(signals[ep.episode_id][slot]) if signals else 0
            queries.append({"query_id": query_id, "episode_id": ep.episode_id,
                            "machine_id": machine_id, "machine_type": MACHINE_TYPES[slot % 3],
                            "signal": str(signal), "cutoff": str(ep.cutoff)})
            labels[query_id] = (slot + 1) % 5
    return SplitData(episodes, {e.episode_id: i for i, e in enumerate(episodes)},
                     queries, labels if with_labels else None)


def small_backbone(seed: int = 1) -> NextSpinTransformer:
    torch.manual_seed(seed)
    return NextSpinTransformer("sparse", d_model=16, nhead=2, num_layers=1, dim_ff=32, dropout=0.0)


def small_signal_model(seed: int = 2) -> SN.SignalTransformer:
    torch.manual_seed(seed)
    return SN.SignalTransformer("sparse", d_model=16, nhead=2, num_layers=1, dim_ff=32, dropout=0.0)


class ConstantModel(torch.nn.Module):
    """Dummy 'specialist': arch attribute + forward whose softmax is a constant
    (8,5) distribution (distinct per model)."""

    arch = "sparse"

    def __init__(self, const):
        super().__init__()
        self.register_buffer("logits", torch.log(torch.tensor(const, dtype=torch.float32)))

    def forward(self, batch):
        b = batch["symbols"].shape[0]
        return self.logits.view(1, 1, 5).expand(b, N_MACHINES, 5).clone()


# --------------------------------------------------------------------------- #
# 1. zero-init signal_emb => identical outputs to the backbone
# --------------------------------------------------------------------------- #

def test_zero_init_signal_emb_equals_backbone():
    episodes = [make_episode("syn-a", 0), make_episode("syn-b", 1)]
    signals = {"syn-a": MIXED, "syn-b": MIXED[::-1]}
    data = make_split_data(episodes, signals)
    backbone, sig_model = small_backbone(1), small_signal_model(2)
    missing, unexpected = sig_model.load_state_dict(backbone.state_dict(), strict=False)
    assert missing == ["signal_emb.weight"] and unexpected == []
    assert torch.count_nonzero(sig_model.signal_emb.weight) == 0
    batch = SN.SignalCollator(WindowCollator(episodes, "sparse", 16), signals)(
        [Example(0, episodes[0].cutoff, None), Example(1, episodes[1].cutoff, None)])
    backbone.eval()
    sig_model.eval()
    with torch.no_grad():
        out_backbone = backbone({k: v for k, v in batch.items() if k != "signals_q"})
        out_signal = sig_model(batch)
    assert out_backbone.shape == (2, N_MACHINES, 5)
    assert torch.allclose(out_backbone, out_signal, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. routing: each query gets its own signal's model output
# --------------------------------------------------------------------------- #

def test_load_signals_slot_order():
    episode = make_episode("syn-r0", 3)
    data = make_split_data([episode], {"syn-r0": MIXED})
    signals = SN.load_signals(data)
    assert set(signals) == {episode.episode_id}
    for slot, machine_id in enumerate(episode.machine_ids):
        row = next(r for r in data.queries if r["machine_id"] == machine_id)
        assert signals[episode.episode_id][episode.slot_of(machine_id)] == int(row["signal"])


def test_routed_prediction_uses_own_signal_model():
    episode = make_episode("syn-r", 4)
    data = make_split_data([episode], {"syn-r": MIXED})
    signals = SN.load_signals(data)
    consts = {-1: [0.10, 0.20, 0.30, 0.15, 0.25],
              0: [0.25, 0.25, 0.20, 0.20, 0.10],
              1: [0.05, 0.15, 0.30, 0.35, 0.15]}
    models = {s: ConstantModel(c) for s, c in consts.items()}
    preds, routes = SN.predict_routed(models, data, signals, device="cpu", window=16)
    assert len(preds) == len(data.queries) == N_MACHINES
    for row in data.queries:
        slot = episode.slot_of(row["machine_id"])
        s = int(row["signal"])
        assert s in (-1, 0, 1)
        assert routes[row["query_id"]] == {"signal": s, "model": SN.model_name_for_signal(s)}
        assert np.allclose(preds[row["query_id"]], consts[s], atol=1e-6)
    # every signal actually appears in the mixed episode
    assert {routes[q]["signal"] for q in routes} == {-1, 0, 1}


# --------------------------------------------------------------------------- #
# 3. no backfill: signals only on the query readouts; history ticks < cutoff
# --------------------------------------------------------------------------- #

def test_no_signal_backfill_and_causality():
    episode = make_episode("syn-h", 5)
    data = make_split_data([episode], {"syn-h": MIXED})
    signals = SN.load_signals(data)
    base = WindowCollator([episode], "sparse", 16)
    collator = SN.SignalCollator(base, signals)
    sliding = sliding_examples([episode], t_start=8)[:6]
    batch = collator(sliding)
    plain = base(sliding)
    assert set(batch.keys()) == set(plain.keys()) | {"signals_q"}
    for key, value in plain.items():
        assert torch.equal(batch[key], value)
    assert ((batch["signals_q"] >= 0) & (batch["signals_q"] <= 2)).all()
    assert batch["signals_q"].shape == (len(sliding), N_MACHINES)
    for i, ex in enumerate(sliding):
        assert torch.equal(batch["signals_q"][i], torch.from_numpy(signals[episode.episode_id] + 1))
    assert (batch["ticks"] < batch["target_t"][:, None]).all(), "history ticks must precede target"
    query_batch = collator([Example(0, episode.cutoff, None)])
    assert (query_batch["ticks"] < query_batch["target_t"][:, None]).all(), "ticks must precede cutoff"
    assert int(query_batch["target_t"][0]) == episode.cutoff


# --------------------------------------------------------------------------- #
# 4. specialist loss mask only covers machines with matching signal
# --------------------------------------------------------------------------- #

def test_specialist_mask_only_matching_signal():
    label_mask = torch.tensor([[True, True, True, False, True, True, True, True]], dtype=torch.bool)
    raw = torch.tensor([[-1, 0, 1, 0, -1, 0, 1, -1]], dtype=torch.long)
    signals_q = raw + 1
    for signal in (-1, 0, 1):
        mask = SN.specialist_mask(label_mask, signals_q, signal)
        assert torch.equal(mask, label_mask & (raw == signal))
    mask0 = SN.specialist_mask(label_mask, signals_q, 0)
    assert not mask0[0, 3], "labeled-but-unobserved machine stays excluded"
    assert mask0[0, 1].item() and mask0[0, 5].item()
    assert not mask0[0, 0].item() and not mask0[0, 2].item()


def test_specialist_training_mask_on_real_query_batch():
    episode = make_episode("syn-sp", 6)
    data = make_split_data([episode], {"syn-sp": MIXED}, with_labels=True)
    signals = SN.load_signals(data)
    examples = query_examples(data.episodes, data.episode_index, data.queries, data.query_labels)
    batch = SN.SignalCollator(WindowCollator(data.episodes, "sparse", 16), signals)(examples)
    assert batch["label_mask"].all()
    for signal in (-1, 0, 1):
        mask = SN.specialist_mask(batch["label_mask"], batch["signals_q"], signal)
        expected = torch.from_numpy((signals[episode.episode_id] == signal))
        assert torch.equal(mask[0], expected)
        assert int(mask.sum()) == int(expected.sum())


# --------------------------------------------------------------------------- #
# 5. prediction CSV: finite, rows sum to 1, 5 columns, ids == queries.csv ids
# --------------------------------------------------------------------------- #

def read_csv_all_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def check_prediction_csv(path, query_ids):
    rows = read_csv_all_rows(path)
    header, body = rows[0], rows[1:]
    assert len(header) == 6 and header[0] == "query_id"  # query_id + 5 probability columns
    assert [r[0] for r in body] == sorted(query_ids)
    for row in body:
        values = [float(v) for v in row[1:]]
        assert len(values) == 5
        assert all(np.isfinite(values))
        assert abs(sum(values) - 1.0) < 1e-9
        assert all(v >= 0.0 for v in values)


def test_prediction_csv_synthetic(tmp_path):
    episodes = [make_episode("syn-csv", 7), make_episode("syn-csv2", 8)]
    data = make_split_data(episodes, {e.episode_id: MIXED for e in episodes})
    preds = {row["query_id"]: [0.2, 0.2, 0.2, 0.2, 0.2] for row in data.queries}
    path = tmp_path / "preds.csv"
    write_submission_csv(path, preds)
    check_prediction_csv(path, [r["query_id"] for r in data.queries])


@pytest.mark.skipif(not (REPO_ROOT / "kit/public/validation/queries.csv").exists(),
                    reason="kit validation queries not available")
def test_prediction_csv_full_validation_ids(tmp_path):
    queries = read_csv_rows(REPO_ROOT / "kit/public/validation/queries.csv")
    assert len(queries) == 768
    preds = {r["query_id"]: [0.3, 0.3, 0.2, 0.15, 0.05] for r in queries}
    path = tmp_path / "validation_predictions.csv"
    write_submission_csv(path, preds)
    check_prediction_csv(path, [r["query_id"] for r in queries])


# --------------------------------------------------------------------------- #
# 6. checkpoint reload reproduces predictions
# --------------------------------------------------------------------------- #

def test_checkpoint_reload_reproduces_predictions(tmp_path):
    episodes = [make_episode("syn-ck", 9)]
    signals = {"syn-ck": MIXED}
    data = make_split_data(episodes, signals)
    model = small_signal_model(3)
    model.eval()
    ckpt_path = tmp_path / "best.pt"
    torch.save({"state_dict": model.state_dict(), "arch": "sparse",
                "config": {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_ff": 32,
                           "dropout": 0.0, "window": 16},
                "mode": "shared", "signal": None,
                "backbone_path": "n/a", "backbone_sha256": "0" * 64,
                "best_epoch": 1, "tune_R": 1.0}, ckpt_path)
    reloaded = SN.load_model_from_checkpoint(ckpt_path)
    reloaded.eval()
    p1 = SN.predict_queries(model, data, "cpu", window=16, signals=signals)
    p2 = SN.predict_queries(reloaded, data, "cpu", window=16, signals=signals)
    assert p1.keys() == p2.keys()
    for query_id in p1:
        assert np.allclose(p1[query_id], p2[query_id], atol=1e-7)


KIT_READY = ((REPO_ROOT / "kit/public/train/queries.csv").exists()
             and SN.DEFAULT_BACKBONE.exists())


@pytest.mark.skipif(not KIT_READY, reason="kit/backbone checkpoint not available")
def test_finetune_smoke_shared_and_specialist(tmp_path):
    """1-epoch fine-tunes on a tiny real subset; best.pt reload reproduces
    predictions (covers the full finetune -> checkpoint -> reload path)."""
    ctx = SN.load_context()
    shared_model, shared_info = SN.finetune(
        "shared", tmp_path / "shared", epochs=1, batch_size=4, context=ctx, device="cpu",
        train_episode_limit=4, tune_episode_limit=4, log=lambda *_: None)
    assert (tmp_path / "shared" / "best.pt").exists()
    assert shared_info["best_epoch"] == 1 and shared_info["train_examples"] == 4
    reloaded = SN.load_model_from_checkpoint(tmp_path / "shared" / "best.pt")
    tune_data = subset_split_data(
        ctx.val_data, SN.limit_episodes_stratified(ctx.tune, ctx.official, 4))
    p1 = SN.predict_queries(shared_model, tune_data, "cpu", window=64, signals=ctx.signals_val)
    p2 = SN.predict_queries(reloaded, tune_data, "cpu", window=64, signals=ctx.signals_val)
    for query_id in p1:
        assert np.allclose(p1[query_id], p2[query_id], atol=1e-6)

    spec_model, spec_info = SN.finetune(
        "specialist", tmp_path / "specialist_signal_m1", signal=-1, epochs=1, batch_size=4,
        context=ctx, device="cpu", train_episode_limit=4, tune_episode_limit=4, log=lambda *_: None)
    assert spec_info["signal"] == -1
    assert (tmp_path / "specialist_signal_m1" / "best.pt").exists()
    ckpt = torch.load(tmp_path / "specialist_signal_m1" / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["mode"] == "specialist" and ckpt["signal"] == -1
    assert ckpt["backbone_sha256"] == SN.sha256_file(SN.DEFAULT_BACKBONE)
    assert isinstance(ckpt["state_dict"], dict) and "signal_emb.weight" not in ckpt["state_dict"]
    reloaded_spec = SN.load_model_from_checkpoint(tmp_path / "specialist_signal_m1" / "best.pt")
    q1 = SN.predict_queries(spec_model, tune_data, "cpu", window=64)
    q2 = SN.predict_queries(reloaded_spec, tune_data, "cpu", window=64)
    for query_id in q1:
        assert np.allclose(q1[query_id], q2[query_id], atol=1e-6)
