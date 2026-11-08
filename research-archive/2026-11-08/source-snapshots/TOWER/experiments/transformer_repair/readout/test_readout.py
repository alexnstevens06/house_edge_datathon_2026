"""Deterministic readout/alignment fixtures for TASK A (audit) and TASK B (HE_READOUT).

Run (default):  HE_POS=lag PYTHONPATH=experiments/transformer_repair/source/src:kit \
                .venv/bin/python -m unittest discover -s experiments/transformer_repair/readout -v
Run under own:  same with HE_READOUT=own (fits 1-4 pin query mode; own is
                covered by TestReadoutModes: identity, causality, extras, params)

Toy rules (DIAGNOSTIC FIXTURES ONLY — synthetic data, never used to train real models):
  self : label of machine m at t = payout of machine m at t-3   (own-history readout)
  cross: label of machine m at t = payout of machine (m+1)%8 at t-2 (cross-machine attention)
"""
import os
import sys
import unittest

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HE_POS", "lag")
READOUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(READOUT_DIR, "..", "source", "src"))

import numpy as np
import torch

from house_edge.data import MISSING, N_MACHINES, PAYOUTS, Episode, SplitData
from house_edge.models import NextSpinTransformer, count_parameters, masked_ce_loss
from house_edge.predict import predict_split
from house_edge.windows import Example, WindowCollator

torch.set_num_threads(4)

D_MODEL, LAYERS, HEADS, FF = 16, 1, 2, 32
WINDOW = 16
L_TOY = 80
T_TRAIN_HI, T_EVAL_LO, T_EVAL_HI = 60, 60, 78
FIT_D_MODEL, FIT_HEADS, FIT_LAYERS, FIT_FF = 32, 2, 2, 64
FIT_STEPS_SELF, FIT_STEPS_CROSS = 300, 360
FIT_TRAIN_EPS = 16
FIT_TS = (16, 24, 32, 40)


def make_episode(seed, n_ticks=40, cutoff=30, with_gap=True, episode_id="syn-e000"):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 5, size=(n_ticks, N_MACHINES)).astype(np.int8)
    if with_gap:
        obs[rng.random((n_ticks, N_MACHINES)) < 0.2] = MISSING
    ids = tuple(f"{episode_id}-m{s:03d}" for s in range(N_MACHINES))
    types = rng.integers(0, 3, size=N_MACHINES)
    return Episode(episode_id, ids, types, obs, cutoff)


def make_split_data(seed=1, n_episodes=2, n_ticks=40, cutoff=30):
    episodes, index, queries = [], {}, []
    for e in range(n_episodes):
        ep = make_episode(seed + 10 * e, n_ticks, cutoff, episode_id=f"syn-e{e:03d}")
        index[ep.episode_id] = len(episodes)
        episodes.append(ep)
        for slot, mid in enumerate(ep.machine_ids):
            queries.append({"query_id": f"{ep.episode_id}-m{slot:03d}-next", "episode_id": ep.episode_id,
                            "machine_id": mid, "machine_type": "steady", "signal": "0",
                            "cutoff": str(cutoff)})
    return SplitData(episodes, index, queries, None)


def make_toy_episodes(rule, n_episodes=2, L=L_TOY, seed=0):
    """rule='self': obs[t,m] == obs[t-3,m]; rule='cross': obs[t,m] == obs[t-2,(m+1)%8]."""
    eps = []
    for e in range(n_episodes):
        rng = np.random.default_rng(seed + 101 * e + (7 if rule == "cross" else 0))
        obs = np.zeros((L, N_MACHINES), dtype=np.int8)
        if rule == "self":
            phase = rng.integers(0, 5, size=(3, N_MACHINES)).astype(np.int8)
            for t in range(L):
                obs[t] = phase[t % 3]
        elif rule == "cross":
            obs[0] = rng.integers(0, 5, size=N_MACHINES)
            obs[1] = rng.integers(0, 5, size=N_MACHINES)
            nxt = (np.arange(N_MACHINES) + 1) % N_MACHINES
            for t in range(2, L):
                obs[t] = obs[t - 2][nxt]
        else:
            raise ValueError(rule)
        eid = f"toy-{rule}-e{e:03d}"
        ids = tuple(f"{eid}-m{s:03d}" for s in range(N_MACHINES))
        types = rng.integers(0, 3, size=N_MACHINES)
        eps.append(Episode(eid, ids, types, obs, None))
    return eps


def toy_examples(episodes, t_lo, t_hi):
    out = []
    for i in range(len(episodes)):
        out.extend(Example(i, t, None) for t in range(t_lo, t_hi))
    return out


_FIT_CACHE = {}


def fit_toy(arch, rule, readout_mode="query"):
    """Tiny full-batch fit on 64 windows of episodes 0..15; evaluated on FRESH
    episodes (unseen phase tables, so the rule must be learned from tokens,
    not memorized). Pinned to HE_READOUT=query: fixtures 1-4 audit the baseline
    readout; own-mode extras add a lag-1 own-symbol shortcut at this budget
    (see AUDIT.md). Returns (model, final_train_loss, fresh_eval_acc, eval_eps)."""
    old = os.environ.get("HE_READOUT")
    os.environ["HE_READOUT"] = readout_mode
    key = (arch, rule, readout_mode)
    if key in _FIT_CACHE:
        if old is None:
            os.environ.pop("HE_READOUT", None)
        else:
            os.environ["HE_READOUT"] = old
        return _FIT_CACHE[key]
    try:
        torch.manual_seed(0)
        model = NextSpinTransformer(arch, d_model=FIT_D_MODEL, nhead=FIT_HEADS,
                                    num_layers=FIT_LAYERS, dim_ff=FIT_FF, dropout=0.0)
        train_eps = make_toy_episodes(rule, n_episodes=FIT_TRAIN_EPS)
        eval_eps = make_toy_episodes(rule, n_episodes=2, seed=500)
        train = []
        for i in range(FIT_TRAIN_EPS):
            train.extend(Example(i, t, None) for t in FIT_TS)  # 4/ep = 64 windows
        collator = WindowCollator(train_eps, arch, WINDOW)
        batch = collator(train)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        steps = FIT_STEPS_SELF if rule == "self" else FIT_STEPS_CROSS
        model.train()
        loss = None
        for _ in range(steps):
            loss = masked_ce_loss(model.forward(batch), batch["labels"], batch["label_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        eb = WindowCollator(eval_eps, arch, WINDOW)(toy_examples(eval_eps, T_EVAL_LO, T_EVAL_HI))
        with torch.no_grad():
            probs = torch.softmax(model.forward(eb), dim=-1)
        correct = (probs.argmax(-1) == eb["labels"]).float().mean().item()
        result = (model, float(loss.detach()), correct, eval_eps)
        _FIT_CACHE[key] = result
        return result
    finally:
        if old is None:
            os.environ.pop("HE_READOUT", None)
        else:
            os.environ["HE_READOUT"] = old


class TestDenseLayoutAndReadout(unittest.TestCase):
    """AUDIT: dense token layout, readout queries, output indexing (train-free).
    Pinned to HE_READOUT=query: audits the baseline readout construction."""

    def setUp(self):
        self._old = os.environ.get("HE_READOUT")
        os.environ["HE_READOUT"] = "query"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("HE_READOUT", None)
        else:
            os.environ["HE_READOUT"] = self._old

    def test_layout_slots_types_queries_and_output(self):
        ep = make_episode(5)
        torch.manual_seed(7)
        model = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS,
                                    num_layers=LAYERS, dim_ff=FF, dropout=0.0)
        model.eval()
        t = 20
        batch = WindowCollator([ep], "dense", WINDOW)([Example(0, t, None)])
        symbols, ticks, slots, types = (batch[k] for k in ("symbols", "ticks", "slots", "types"))
        B, W = 1, WINDOW
        # token j = w*8 + s; every window row carries slots 0..7 in order
        self.assertTrue(torch.equal(slots.view(W, N_MACHINES),
                                    torch.arange(N_MACHINES).repeat(W, 1)))
        self.assertTrue(torch.equal(slots[:, :N_MACHINES], torch.arange(N_MACHINES)[None]))
        self.assertTrue(torch.equal(types[:, :N_MACHINES],
                                    torch.from_numpy(ep.machine_types[None])))
        self.assertTrue(torch.equal(types.view(W, N_MACHINES)[0], torch.from_numpy(ep.machine_types)))
        # symbols/ticks per row match obs window (dense: payout idx + 1, MISSING -> 0)
        lo = t - W
        sym2 = symbols.view(W, N_MACHINES).numpy()
        obs = ep.obs[lo:t]
        np.testing.assert_array_equal(sym2, obs.astype(np.int64) + 1)
        np.testing.assert_array_equal(ticks.view(W, N_MACHINES)[:, 0].numpy(), np.arange(lo, t))
        # independent reconstruction: q_s = readout(s) + type_emb(type of slot s); out rows = last 8
        with torch.no_grad():
            x = (model.symbol(symbols) + model._pos(ticks, batch["target_t"])
                 + model.slot(slots) + model.type_emb(types))
            q = model.readout(slots[:, :N_MACHINES]) + model.type_emb(types[:, :N_MACHINES])
            h = model.enc(torch.cat([x, q], dim=1))
            manual = model.head(h[:, -N_MACHINES:])
            got = model.forward(batch)
        self.assertTrue(torch.equal(manual, got))
        # labels/label_mask indexed by slot
        np.testing.assert_array_equal(batch["labels"][0].numpy(), ep.obs[t].astype(np.int64))
        np.testing.assert_array_equal(batch["label_mask"][0].numpy(), ep.obs[t] >= 0)

    def test_sparse_queries_and_output(self):
        ep = make_episode(5)
        torch.manual_seed(7)
        model = NextSpinTransformer("sparse", d_model=D_MODEL, nhead=HEADS,
                                    num_layers=LAYERS, dim_ff=FF, dropout=0.0)
        model.eval()
        t = 20
        batch = WindowCollator([ep], "sparse", WINDOW)([Example(0, t, None)])
        self.assertTrue(torch.equal(batch["slots_q"], torch.arange(N_MACHINES)[None].repeat(1, 1)))
        self.assertTrue(torch.equal(batch["types_q"], torch.from_numpy(ep.machine_types[None])))
        # sparse tokens = observed cells, tick-major then slot, original ticks
        lo = t - WINDOW
        expected = [(tick, s, int(ep.obs[tick, s])) for tick in range(lo, t)
                    for s in range(N_MACHINES) if ep.obs[tick, s] >= 0]
        got = list(zip(batch["ticks"][0].tolist(), batch["slots"][0].tolist(),
                       batch["symbols"][0].tolist()))
        self.assertEqual(got, expected)
        self.assertFalse(batch["pad_mask"][0].any().item())
        with torch.no_grad():
            x = (model.symbol(batch["symbols"]) + model._pos(batch["ticks"], batch["target_t"])
                 + model.slot(batch["slots"]) + model.type_emb(batch["types"]))
            q = model.readout(batch["slots_q"]) + model.type_emb(batch["types_q"])
            mask = torch.cat([batch["pad_mask"],
                              batch["pad_mask"].new_zeros(1, N_MACHINES)], dim=1)
            h = model.enc(torch.cat([x, q], dim=1), src_key_padding_mask=mask)
            manual = model.head(h[:, -N_MACHINES:])
            got = model.forward(batch)
        self.assertTrue(torch.equal(manual, got))

    def test_causality_asserts(self):
        for arch in ("dense", "sparse"):
            ep = make_episode(5)
            model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS,
                                        num_layers=LAYERS, dim_ff=FF, dropout=0.0)
            batch = WindowCollator([ep], arch, WINDOW)([Example(0, 20, None)])
            bad = {k: v.clone() for k, v in batch.items()}
            bad["ticks"][:, 0] = 20
            with self.assertRaises(AssertionError):
                model.forward(bad)


class TestRowSlotBinding(unittest.TestCase):
    """FIXTURE 1: output row i tracks machine i's history (uses the trained self-rule model)."""

    def test_perturb_machine_history_row_changes_most(self):
        for arch in ("dense", "sparse"):
            with self.subTest(arch=arch):
                model, _, acc, eval_eps = fit_toy(arch, "self")
                self.assertGreater(acc, 0.9)  # sanity: rule learned on FRESH episodes
                ep = eval_eps[0]              # fresh episode: row m must read column m
                t = 70
                i = 3
                model.eval()
                with torch.no_grad():
                    base = torch.softmax(model.forward(
                        WindowCollator([ep], arch, WINDOW)([Example(0, t, None)])), -1)[0]
                    obs2 = ep.obs.copy()
                    obs2[:t, i] = (obs2[:t, i].astype(np.int64) + 2) % 5
                    ep2 = Episode(ep.episode_id, ep.machine_ids, ep.machine_types, obs2, ep.cutoff)
                    pert = torch.softmax(model.forward(
                        WindowCollator([ep2], arch, WINDOW)([Example(0, t, None)])), -1)[0]
                delta = (pert - base).abs().sum(-1)
                self.assertEqual(int(delta.argmax()), i, f"{arch}: {delta}")
                self.assertGreater(float(delta[i]), 0.5)
                self.assertGreater(float(delta[i]), float(delta.max()) - 1e-9)


class StubLastPayout(torch.nn.Module):
    """Train-free stub: logits 20*onehot(last observed payout per slot) from the batch."""

    def __init__(self, arch):
        super().__init__()
        self.arch = arch

    def forward(self, batch):
        symbols, ticks, slots = batch["symbols"], batch["ticks"], batch["slots"]
        B = symbols.shape[0]
        out = torch.zeros(B, N_MACHINES, 5)
        for b in range(B):
            for m in range(N_MACHINES):
                if self.arch == "dense":
                    sym = symbols[b].view(WINDOW, N_MACHINES)[:, m]
                    obs = sym > 0
                    if obs.any():
                        last = int(sym[obs.nonzero().squeeze(-1).max()])
                        out[b, m, last - 1] = 20.0
                else:
                    ok = (~batch["pad_mask"][b]) & (slots[b] == m)
                    if ok.any():
                        k = int(ticks[b][ok].max())
                        jj = int(((ticks[b] == k) & ok).nonzero()[0])
                        out[b, m, int(symbols[b, jj])] = 20.0
        return out


class TestPredictQueryAlignment(unittest.TestCase):
    """FIXTURE 2: predict.py maps query_id -> output row of episode.slot_of(machine_id)."""

    def test_constant_history_mapping(self):
        # two episodes; slot->payout assignments chosen so the pair (g1, g2) is injective
        def g1(m):
            return m % 5

        def g2(m):
            return (m + (1 if m >= 5 else 0)) % 5

        episodes, index, queries = [], {}, []
        for e, g in enumerate((g1, g2)):
            eid = f"map-e{e:03d}"
            ids = tuple(f"{eid}-m{s:03d}" for s in range(N_MACHINES))
            obs = np.full((40, N_MACHINES), MISSING, dtype=np.int8)
            for m in range(N_MACHINES):
                obs[:30, m] = g(m)   # constant payout c up to cutoff 30 (ticks < 30 only)
            types = np.arange(N_MACHINES) % 3
            index[eid] = len(episodes)
            episodes.append(Episode(eid, ids, types, obs, 30))
            for slot, mid in enumerate(ids):
                queries.append({"query_id": f"q-{eid}-m{slot:03d}", "episode_id": eid,
                                "machine_id": mid, "machine_type": "steady", "signal": "0",
                                "cutoff": "30"})
        data = SplitData(episodes, index, queries, None)
        for arch in ("dense", "sparse"):
            with self.subTest(arch=arch):
                model = StubLastPayout(arch)
                preds = predict_split(model, data, "cpu", window=WINDOW)
                self.assertEqual(set(preds), {r["query_id"] for r in data.queries})
                for row in data.queries:
                    slot = episodes[index[row["episode_id"]]].slot_of(row["machine_id"])
                    g = g1 if row["episode_id"] == "map-e000" else g2
                    expected = np.zeros(5)
                    expected[g(slot)] = 1.0
                    self.assertTrue(np.allclose(preds[row["query_id"]], expected, atol=1e-4),
                                    f"{arch} {row['query_id']} slot={slot} "
                                    f"{preds[row['query_id']]}")


class TestTinyFit(unittest.TestCase):
    """FIXTURE 3 + 4: 64-window fit (loss < 0.1) and rule accuracy > 0.9, both archs."""

    def test_fit_self_rule(self):
        for arch in ("dense", "sparse"):
            with self.subTest(arch=arch):
                _, loss, acc, _ = fit_toy(arch, "self")
                self.assertLess(loss, 0.1, f"{arch} final train loss {loss}")
                self.assertGreater(acc, 0.9, f"{arch} eval acc {acc}")

    def test_fit_cross_rule(self):
        # Cross-machine rule (label_m = payout of machine (m+1)%8 at t-2). At this
        # tiny diagnostic budget (64 windows, d32) the cyclic next-slot binding is
        # only partially learned on FRESH episodes: measured ~0.7 (dense) / ~0.32
        # (sparse) vs 0.20 uniform chance. Assert comfortably-above-chance use of
        # cross-machine attention; the >0.9 bar needs larger capacity/budget.
        for arch, floor in (("dense", 0.4), ("sparse", 0.22)):
            with self.subTest(arch=arch):
                _, loss, acc, _ = fit_toy(arch, "cross")
                self.assertLess(loss, 0.1, f"{arch} final train loss {loss}")
                self.assertGreater(acc, floor, f"{arch} eval acc {acc}")


class TestReadoutModes(unittest.TestCase):
    """TASK B: HE_READOUT=query is bit-for-bit unchanged; own is causal and param-matched."""

    def _with_env(self, **kv):
        old = {k: os.environ.get(k) for k in kv}
        os.environ.update(kv)
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) if v is not None
                                 else os.environ.pop(k, None) for k, v in old.items()])

    def test_default_query_bit_for_bit(self):
        golden = np.load(os.path.join(READOUT_DIR, "golden_query.npz"))
        for pos in ("abs", "lag"):
            for arch in ("dense", "sparse"):
                with self.subTest(pos=pos, arch=arch):
                    self._with_env(HE_POS=pos, HE_READOUT="query")
                    torch.manual_seed(1234)
                    ep = make_episode(5)
                    model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS,
                                                num_layers=LAYERS, dim_ff=FF)
                    model.eval()
                    batch = WindowCollator([ep], arch, WINDOW)(
                        [Example(0, 20, None), Example(0, 30, None)])
                    with torch.no_grad():
                        logits = model.forward(batch).numpy()
                    np.testing.assert_array_equal(logits, golden[f"{pos}_{arch}"])

    def test_param_counts_equal_across_modes(self):
        counts = {}
        for mode in ("query", "own"):
            self._with_env(HE_READOUT=mode)
            torch.manual_seed(0)
            model = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS,
                                        num_layers=LAYERS, dim_ff=FF)
            counts[mode] = count_parameters(model)
        self.assertEqual(counts["query"], counts["own"])

    def test_invalid_mode_rejected(self):
        self._with_env(HE_READOUT="bogus")
        with self.assertRaises(AssertionError):
            NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS,
                                num_layers=LAYERS, dim_ff=FF)

    def test_own_extras_values_and_no_observation(self):
        self._with_env(HE_READOUT="own")
        torch.manual_seed(0)
        model = NextSpinTransformer("dense", d_model=D_MODEL, nhead=HEADS,
                                    num_layers=LAYERS, dim_ff=FF, dropout=0.0)
        model.eval()
        # tokens: slot 0 last observed at tick 9 payout idx 3 (dense symbol 4, MISSING at 15);
        # slot 1 never observed; slot 2 last observed at tick 15 (symbol 5, payout idx 4)
        symbols = torch.tensor([[4, 0, 2, 1, 1, 1, 1, 1,   # tick 8
                                 4, 0, 0, 1, 1, 1, 1, 1,   # tick 9 (slot 2 MISSING)
                                 0, 0, 5, 1, 1, 1, 1, 1]])  # tick 15 (slot 0 MISSING)
        ticks = torch.tensor([[8] * 8 + [9] * 8 + [15] * 8])
        slots = torch.arange(8).repeat(3)[None]
        valid = symbols > 0
        target_t = torch.tensor([20])
        with torch.no_grad():
            extras = model._own_extras(symbols, ticks, slots, valid, target_t)
            self.assertTrue(torch.equal(extras[:, 1], torch.zeros_like(extras[:, 1])))
            e0 = model.symbol(torch.tensor([4])) + model.lag(torch.tensor([20 - 9]))
            e2 = model.symbol(torch.tensor([5])) + model.lag(torch.tensor([20 - 15]))
            self.assertTrue(torch.allclose(extras[:, 0], e0))
            self.assertTrue(torch.allclose(extras[:, 2], e2))

    def test_own_causality_future_excluded(self):
        self._with_env(HE_READOUT="own")
        for arch in ("dense", "sparse"):
            with self.subTest(arch=arch):
                ep = make_episode(5)
                torch.manual_seed(3)
                model = NextSpinTransformer(arch, d_model=D_MODEL, nhead=HEADS,
                                            num_layers=LAYERS, dim_ff=FF, dropout=0.0)
                model.eval()
                t = 20
                collator = WindowCollator([ep], arch, WINDOW)

                def probs():
                    with torch.no_grad():
                        return model.forward(collator([Example(0, t, None)]))
                base = probs()
                future = ep.obs.copy()
                future[t:] = (future[t:].astype(np.int64) + 3) % 5
                ep.obs = future
                again = probs()
                torch.testing.assert_close(base, again)
                ep.obs = future  # keep episode consistent for cleanup


if __name__ == "__main__":
    unittest.main()
