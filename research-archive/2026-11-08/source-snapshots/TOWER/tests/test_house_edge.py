"""Unit tests for House Edge models, tokenization, and evaluation invariants.

Run: .venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch

from house_edge.data import MISSING, N_MACHINES, SplitData, Episode
from house_edge.models import NextSpinTransformer, masked_ce_loss
from house_edge.predict import predict_split
from house_edge.windows import Example, WindowCollator

D_MODEL, LAYERS, HEADS, FF = 16, 1, 2, 32
WINDOW = 16
DEVICE = "cpu"
torch.set_num_threads(4)
torch.manual_seed(0)


def make_split_data(n_episodes=2, n_ticks=40, cutoff=30, seed=1, with_gap=True):
    """Synthetic SplitData: 8 machines/episode, scattered missing ticks."""
    rng = np.random.default_rng(seed)
    episodes, index, queries = [], {}, []
    for e in range(n_episodes):
        episode_id = f"syn-e{e:03d}"
        obs = rng.integers(0, 5, size=(n_ticks, N_MACHINES)).astype(np.int8)
        if with_gap:
            holes = rng.random((n_ticks, N_MACHINES)) < 0.25
            obs[holes] = MISSING
        machine_ids = tuple(f"{episode_id}-m{s:03d}" for s in range(N_MACHINES))
        types = rng.integers(0, 3, size=N_MACHINES)
        index[episode_id] = len(episodes)
        from house_edge.data import Episode as EpisodeCls
        episodes.append(EpisodeCls(episode_id, machine_ids, types, obs, cutoff))
        for slot, machine_id in enumerate(machine_ids):
            queries.append({"query_id": f"{episode_id}-m{slot:03d}-next", "episode_id": episode_id,
                            "machine_id": machine_id, "machine_type": "steady", "signal": "0",
                            "cutoff": str(cutoff)})
    return SplitData(episodes, index, queries, None)


def collated(arch, data, t, labels=None):
    collator = WindowCollator(data.episodes, arch, WINDOW)
    return collator([Example(0, t, labels)])


class TestCausality(unittest.TestCase):
    def setUp(self):
        self.data = make_split_data()
        self.t = 20

    def _model(self, arch):
        torch.manual_seed(0)
        return NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                   dim_ff=FF)

    def _outputs(self, arch, data, model):
        model.eval()
        with torch.no_grad():
            batch = collated(arch, data, self.t)
            probs = torch.softmax(model.forward(batch), dim=-1)
        return probs, batch

    def test_dense_future_tokens_excluded(self):
        model = self._model("dense")
        base, base_batch = self._outputs("dense", self.data, model)
        future = np.array(self.data.episodes[0].obs, copy=True)
        future[self.t:] = (future[self.t:] + 3) % 5
        self.data.episodes[0].obs = future
        again, again_batch = self._outputs("dense", self.data, model)
        torch.testing.assert_close(base_batch["symbols"], again_batch["symbols"])
        torch.testing.assert_close(base, again)

    def test_sparse_future_tokens_excluded(self):
        model = self._model("sparse")
        base, base_batch = self._outputs("sparse", self.data, model)
        future = np.array(self.data.episodes[0].obs, copy=True)
        future[self.t:] = (future[self.t:] + 3) % 5
        self.data.episodes[0].obs = future
        again, again_batch = self._outputs("sparse", self.data, model)
        torch.testing.assert_close(base_batch["symbols"], again_batch["symbols"])
        torch.testing.assert_close(base, again)

    def test_forward_asserts_on_leak(self):
        model = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                    dim_ff=FF)
        batch = collated("dense", self.data, self.t)
        bad = {k: v.clone() for k, v in batch.items()}
        bad["ticks"][:, 0] = self.t  # a token claiming to be at the target tick
        with self.assertRaises(AssertionError):
            model.forward(bad)


class TestTokenization(unittest.TestCase):
    def setUp(self):
        self.data = make_split_data(seed=7)
        self.t = 30

    def test_sparse_preserves_original_ticks(self):
        batch = collated("sparse", self.data, self.t)
        obs = self.data.episodes[0].obs
        lo = max(0, self.t - WINDOW)
        expected_ticks, expected_syms = [], []
        for tick in range(lo, self.t):
            for slot in range(N_MACHINES):
                if obs[tick, slot] >= 0:
                    expected_ticks.append(tick)
                    expected_syms.append(int(obs[tick, slot]))
        n = len(expected_ticks)
        self.assertEqual(batch["symbols"].shape[1], n)  # no padding needed (single example)
        np.testing.assert_array_equal(batch["ticks"][0].numpy(), np.array(expected_ticks))
        np.testing.assert_array_equal(batch["symbols"][0].numpy(), np.array(expected_syms))

    def test_dense_missing_is_not_payout_zero(self):
        obs = self.data.episodes[0].obs
        t = self.t
        obs[t - 1, 0] = MISSING   # machine 0 unobserved at t-1
        obs[t - 1, 1] = 0         # machine 1 observed payout 0 at t-1
        batch = collated("dense", self.data, t)
        symbols = batch["symbols"].reshape(WINDOW, N_MACHINES).numpy()
        self.assertEqual(symbols[-1, 0], 0)   # MISSING symbol
        self.assertEqual(symbols[-1, 1], 1)   # payout 0 symbol
        model = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                    dim_ff=FF)
        with torch.no_grad():
            emb_missing = model.symbol(torch.tensor([0]))
            emb_zero = model.symbol(torch.tensor([1]))
        self.assertFalse(torch.allclose(emb_missing, emb_zero))

    def test_dense_ticks_are_absolute(self):
        batch = collated("dense", self.data, self.t)
        ticks = batch["ticks"].reshape(WINDOW, N_MACHINES).numpy()
        lo = max(0, self.t - WINDOW)
        np.testing.assert_array_equal(ticks[:, 0], np.arange(lo, self.t))


class TestModel(unittest.TestCase):
    def setUp(self):
        self.data = make_split_data(seed=3)

    def _batch(self, arch):
        collator = WindowCollator(self.data.episodes, arch, WINDOW)
        labels = np.arange(N_MACHINES) % 5
        return collator([Example(0, 20, labels), Example(0, 25, None)])

    def test_output_shape_and_probabilities(self):
        for arch in ("dense", "sparse"):
            model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                        dim_ff=FF)
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model.forward(self._batch(arch)), dim=-1)
            self.assertEqual(tuple(probs.shape), (2, N_MACHINES, 5))
            self.assertTrue(torch.isfinite(probs).all())
            torch.testing.assert_close(probs.sum(-1), torch.ones(2, N_MACHINES))
            self.assertTrue((probs >= 0).all())

    def test_loss_ignores_unobserved_targets(self):
        for arch in ("dense", "sparse"):
            torch.manual_seed(0)
            model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                        dim_ff=FF)
            model.eval()
            batch = self._batch(arch)
            labels = batch["labels"].clone()
            mask = batch["label_mask"].clone()
            mask[0, 2:] = False  # machine 2..7 unobserved for example 0
            with torch.no_grad():
                logits = model.forward(batch)
                loss_a = masked_ce_loss(logits, labels, mask)
                labels[0, 2:] = (labels[0, 2:] + 2) % 5  # perturb unobserved labels
                loss_b = masked_ce_loss(logits, labels, mask)
            torch.testing.assert_close(loss_a, loss_b)

    def test_all_masked_raises(self):
        for arch in ("dense", "sparse"):
            model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                        dim_ff=FF)
            batch = self._batch(arch)
            mask = torch.zeros_like(batch["label_mask"])
            with self.assertRaises(AssertionError):
                masked_ce_loss(model.forward(batch), batch["labels"], mask)

    def test_matched_parameter_counts(self):
        dense = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                    dim_ff=FF)
        sparse = NextSpinTransformer("sparse", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                     dim_ff=FF)
        n_dense = sum(p.numel() for p in dense.parameters())
        n_sparse = sum(p.numel() for p in sparse.parameters())
        self.assertEqual(n_dense, n_sparse)
        self.assertLess(abs(n_dense - n_sparse) / n_dense, 0.10)

    def test_deterministic_eval(self):
        for arch in ("dense", "sparse"):
            torch.manual_seed(0)
            model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                        dim_ff=FF)
            model.eval()
            with torch.no_grad():
                a = torch.softmax(model.forward(self._batch(arch)), dim=-1)
                b = torch.softmax(model.forward(self._batch(arch)), dim=-1)
            torch.testing.assert_close(a, b)


class TestPredictionAlignment(unittest.TestCase):
    def test_query_id_alignment(self):
        data = make_split_data(n_episodes=2, n_ticks=40, cutoff=30, seed=11)
        model = NextSpinTransformer("sparse", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                    dim_ff=FF)
        preds = predict_split(model, data, DEVICE, window=WINDOW)
        expected_ids = {row["query_id"] for row in data.queries}
        self.assertEqual(set(preds), expected_ids)
        self.assertEqual(len(preds), len(data.queries))
        for query_id, probs in preds.items():
            self.assertEqual(len(probs), 5)
            self.assertTrue(all(np.isfinite(probs)))
            self.assertAlmostEqual(sum(probs), 1.0, places=5)
        # same cutoff inputs -> identical repeated evaluation
        again = predict_split(model, data, DEVICE, window=WINDOW)
        self.assertEqual(preds, again)


class TestSignalAndReadout(unittest.TestCase):
    def setUp(self):
        self.data = make_split_data(n_episodes=1, n_ticks=40, cutoff=30, seed=23)

    def test_signal_not_used_as_input(self):
        """Query `signal` is never a model input: flipping it cannot change predictions."""
        model = NextSpinTransformer("sparse", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                    dim_ff=FF)
        base = predict_split(model, self.data, DEVICE, window=WINDOW)
        flipped = make_split_data(n_episodes=1, n_ticks=40, cutoff=30, seed=23)
        for row in flipped.queries:
            row["signal"] = "1" if row["signal"] != "1" else "-1"
        again = predict_split(model, flipped, DEVICE, window=WINDOW)
        self.assertEqual(base, again)

    def test_eight_per_machine_readouts(self):
        model = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                    dim_ff=FF)
        self.assertEqual(model.readout.num_embeddings, N_MACHINES)
        model.eval()
        with torch.no_grad():
            batch = collated("dense", self.data, 25)
            probs = torch.softmax(model.forward(batch), dim=-1)
        self.assertEqual(tuple(probs.shape), (1, N_MACHINES, 5))
        # distinct machines get distinct readout queries -> distinct distributions
        self.assertFalse(torch.allclose(probs[0, 0], probs[0, 1]))
        self.assertFalse(torch.allclose(probs[0, 0], probs[0, 7]))


class TestKitQueryAlignment(unittest.TestCase):
    def test_predictions_align_with_kit_queries_csv(self):
        """predict_split keys are exactly the kit validation query_ids for those episodes."""
        from house_edge.data import load_split, subset_split_data
        val = load_split(os.path.join(REPO, "kit", "public", "validation"))
        episodes = sorted(val.episode_index)[:4]
        subset = subset_split_data(val, episodes)
        model = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS, num_layers=LAYERS,
                                    dim_ff=FF)
        preds = predict_split(model, subset, DEVICE, window=64)
        import csv as _csv
        with open(os.path.join(REPO, "kit", "public", "validation", "queries.csv"),
                  newline="", encoding="utf-8") as handle:
            kit_ids = {r["query_id"] for r in _csv.DictReader(handle)
                       if r["episode_id"] in set(episodes)}
        self.assertEqual(set(preds), kit_ids)
        self.assertEqual(len(preds), len(subset.queries))
        for probs in preds.values():
            self.assertAlmostEqual(sum(probs), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
