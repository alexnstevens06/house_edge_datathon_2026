"""Correctness tests for experiments/small_cnn_data/source/scnn.py (unittest).

Run: .venv/bin/python -m unittest discover -s experiments/small_cnn_data/tests -v
Covers: output shape/normalization, payout order, causality (ticks >= target),
signal invariance, cutoff-label non-leakage into features, zero gradient for
unobserved targets, MISSING-vs-0 gap handling, query_id/slot alignment vs kit
queries.csv, deterministic eval.
"""
import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "source"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "kit"))

import scnn  # noqa: E402
from house_edge.data import load_split, MISSING, N_MACHINES, PAYOUTS, PAYOUT_TO_IDX  # noqa: E402
from house_edge.scoring import write_submission_csv  # noqa: E402
from slot_forecasting.io import read_submission  # noqa: E402
from slot_forecasting.schema import PROBABILITY_COLUMNS  # noqa: E402

PUB = os.path.join(REPO, "kit", "public")


def small_model(W=16, seed=0):
    torch.manual_seed(seed)
    return scnn.CNN(h=8, layers=2, W=W)


class TestSCNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        os.makedirs("/tmp/opencode", exist_ok=True)
        cls.va = load_split(os.path.join(PUB, "validation"))
        cls.tr = load_split(os.path.join(PUB, "train"), with_labels=True)
        cls.Qv, cls.eps_v = scnn.build_query(cls.va, 16)
        cls.model = small_model(16)
        cls.model.eval()

    def fwd(self, X, T, TY):
        return self.model(torch.from_numpy(X), torch.from_numpy(T), torch.from_numpy(TY))

    def test_output_shape_and_rows_sum_to_one(self):
        B = 5
        lg = self.fwd(self.Qv["X"][:B], self.Qv["T"][:B], self.Qv["TY"][:B])
        self.assertEqual(tuple(lg.shape), (B, N_MACHINES, 5))
        p = torch.softmax(lg, -1)
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(B, N_MACHINES), atol=1e-5))
        self.assertTrue((p >= 0).all())
        P = scnn.predict(self.model, self.Qv, "cpu")
        self.assertEqual(P.shape, (len(self.Qv["X"]), N_MACHINES, 5))
        self.assertTrue(np.allclose(P.sum(-1), 1.0, atol=1e-9))

    def test_payout_order(self):
        self.assertEqual(PAYOUTS, (0, 1, 2, 5, 20))
        self.assertEqual(PROBABILITY_COLUMNS, ("p_0", "p_1", "p_2", "p_5", "p_20"))
        for j, pay in enumerate(PAYOUTS):
            self.assertEqual(PAYOUT_TO_IDX[pay], j)  # model column j <-> payout PAYOUTS[j]
        P = np.eye(5)[np.arange(5) % 5]  # 5 rows, distinct argmax per row
        pv = {f"q{i}": P[i].tolist() for i in range(5)}
        path = "/tmp/opencode/test_scnn_order.csv"
        write_submission_csv(path, pv)
        with open(path) as fh:
            header = fh.readline().strip().split(",")
        self.assertEqual(header, ["query_id", *PROBABILITY_COLUMNS])

    def test_causality_ticks_at_or_after_target(self):
        rng = np.random.default_rng(7)
        obs = rng.integers(0, 5, size=(40, N_MACHINES)).astype(np.int8)
        obs[rng.integers(0, 40, 12), rng.integers(0, N_MACHINES, 12)] = MISSING
        t, W = 25, 16
        x0 = scnn.window(obs, t, W)
        lg0 = self.model(torch.from_numpy(x0[None]), torch.tensor([t]), torch.from_numpy(np.zeros(N_MACHINES, np.int64))[None])
        # mutate ALL machines' payouts at every tick >= t (including far future)
        obs_m = obs.copy()
        obs_m[t:] = (obs_m[t:] + 1 + rng.integers(0, 4)) % 5
        obs_m[t:] = obs_m[t:].astype(np.int8)
        self.assertTrue((scnn.window(obs_m, t, W) == x0).all())
        # per-machine spot checks at tick t and the last tick
        for m in range(N_MACHINES):
            for tick in (t, t + 1, 39):
                o = obs.copy()
                o[tick, m] = np.int8((int(o[tick, m]) + 1) % 5)
                self.assertTrue((scnn.window(o, t, W) == x0).all(),
                                f"tick {tick} machine {m} leaked into window")
        lg1 = self.model(torch.from_numpy(scnn.window(obs_m, t, W)[None]), torch.tensor([t]), torch.from_numpy(np.zeros(N_MACHINES, np.int64))[None])
        self.assertTrue(torch.equal(lg0, lg1))

    def test_query_signal_column_invariance(self):
        B = 6
        base = {k: v[:B].copy() for k, v in self.Qv.items()}
        lg0 = self.fwd(base["X"], base["T"], base["TY"])
        # Mutate copies of the query rows, then restore the live rows. build_query
        # reads self.va.queries, so the flipped values are written onto those rows
        # for the call and copied originals are put back even if an assert fails.
        saved = [dict(r) for r in self.va.queries]
        try:
            for r in self.va.queries:
                r["signal"] = "1" if r["signal"] != "1" else "-1"
            Q2, _ = scnn.build_query(self.va, 16)
            for k in base:
                self.assertTrue((Q2[k][:B] == base[k]).all(), f"signal leaked into {k}")
            lg1 = self.fwd(Q2["X"][:B], Q2["T"][:B], Q2["TY"][:B])
            self.assertTrue(torch.equal(lg0, lg1))
        finally:
            for r, orig in zip(self.va.queries, saved):
                r.clear()
                r.update(orig)
            for r, orig in zip(self.va.queries, saved):
                self.assertEqual(r["signal"], orig["signal"])

    def test_cutoff_label_never_enters_features(self):
        W = 16
        D0 = scnn.build_train(self.tr, W)
        qid = self.tr.queries[0]["query_id"]
        orig = self.tr.query_labels[qid]
        self.tr.query_labels[qid] = np.int8((int(orig) + 1) % 5)  # mutate labels.csv outcome
        D1 = scnn.build_train(self.tr, W)
        self.tr.query_labels[qid] = orig
        for k in ("X", "T", "TY", "EP", "ISQ", "OBC"):  # inputs/feature stats unchanged
            self.assertTrue((D0[k] == D1[k]).all(), f"labels leaked into {k}")
        diff = (D0["Y"] != D1["Y"])
        self.assertTrue(diff.any(), "mutated label should change its target row")
        ep0 = self.tr.queries[0]
        ei = self.tr.episode_index[ep0["episode_id"]]
        slot = self.tr.episodes[ei].slot_of(ep0["machine_id"])
        # only the query-cutoff target of that machine may differ
        not_this = (~D0["ISQ"][:, None] | (D0["EP"] != ei)[:, None]
                    | (np.arange(N_MACHINES)[None, :] != slot))
        self.assertEqual(int((diff & not_this).sum()), 0)
        # inference features on the train split never read labels at all
        Qa, _ = scnn.build_query(self.tr, W)
        self.tr.query_labels[qid] = np.int8((int(orig) + 2) % 5)
        Qb, _ = scnn.build_query(self.tr, W)
        self.tr.query_labels[qid] = orig
        for k in Qa:
            self.assertTrue((Qa[k] == Qb[k]).all())

    def test_unobserved_target_zero_gradient(self):
        # One thread: the conv weight reduction is bitwise-identical, so a true
        # gradient from an unobserved row fails atol=0 instead of hiding in a
        # multi-thread add of exact zeros.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self._unobserved_target_zero_gradient()
        finally:
            torch.set_num_threads(threads)

    def _unobserved_target_zero_gradient(self):
        torch.manual_seed(0)
        model = scnn.CNN(h=8, layers=1, W=8)
        model.eval()  # dropout must not differ across the two forwards
        rng = np.random.default_rng(3)
        X = rng.integers(0, 5, size=(2, 8, N_MACHINES)).astype(np.int8)
        T = np.array([20, 30], np.int64)
        TY = np.zeros((2, N_MACHINES), np.int64)
        Y = np.stack([rng.integers(0, 5, size=N_MACHINES), np.full(N_MACHINES, -1)]).astype(np.int8)
        xt, tt, tyt, yt = (torch.from_numpy(a) for a in (X, T, TY, Y))
        for w in (None, torch.tensor([1.0, 8.0])):
            model.zero_grad()
            loss = scnn.masked_loss(model(xt, tt, tyt), yt, w)
            loss.backward()
            g_all = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            model.zero_grad()
            loss1 = scnn.masked_loss(model(xt[:1], tt[:1], tyt[:1]), yt[:1], None if w is None else w[:1])
            loss1.backward()
            g_obs = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            self.assertEqual(len(g_all), len(g_obs))
            for ga, go in zip(g_all, g_obs):
                self.assertTrue(torch.allclose(ga, go, atol=0, rtol=0), "unobserved target contributed gradient")
            self.assertAlmostEqual(loss.item(), loss1.item(), places=6)
        # all-unobserved batch: zero loss, zero grads, even with x8 weight
        model.zero_grad()
        loss = scnn.masked_loss(model(xt, tt, tyt), torch.full_like(yt, -1), torch.tensor([8.0, 8.0]))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        for p in model.parameters():
            self.assertTrue((p.grad is None) or (p.grad == 0).all())

    def test_gap_is_missing_not_zero(self):
        W, t = 8, 10
        obs = np.full((20, N_MACHINES), MISSING, np.int8)
        obs[3:8] = 1
        obs[5, 2] = 3  # observed tick 5, machine 2, payout idx 3
        deleted = obs.copy()
        deleted[5, 2] = MISSING                      # delete the observed tick
        aszero = obs.copy()
        aszero[5, 2] = 0                             # set payout 0 instead
        wd = scnn.window(deleted, t, W)
        wz = scnn.window(aszero, t, W)
        pos = W - (t - 5)  # tick 5 maps to this row
        self.assertEqual(int(wd[pos, 2]), MISSING)
        self.assertEqual(int(wz[pos, 2]), 0)
        self.assertFalse((wd == wz).all())           # MISSING input differs from 0-payout input
        m = (wd >= 0)
        self.assertEqual(int(m[pos, 2]), 0)          # deleted -> masked channel off
        mz = (wz >= 0)
        self.assertEqual(int(mz[pos, 2]), 1)         # 0 payout -> observed
        # unobserved history before episode start pads MISSING, not 0
        early = scnn.window(obs[:3], 20, W)
        self.assertEqual(int(early[:, 2].min()), MISSING or (early[:, 2] == MISSING).any() and MISSING)
        self.assertTrue((early[: W - 3] == MISSING).all())

    def test_query_slot_alignment_vs_kit(self):
        n = len(self.eps_v)
        P = np.zeros((n, N_MACHINES, 5))
        for i in range(n):
            P[i] = np.eye(5)[np.arange(N_MACHINES) % 5]  # row m is a one-hot at class m%5
        pv = scnn.to_preds(P, self.eps_v, self.va)
        qids = {r["query_id"] for r in self.va.queries}
        self.assertEqual(set(pv), qids)  # exactly the kit queries.csv query_ids
        for r in self.va.queries:
            i = self.va.episode_index[r["episode_id"]]
            slot = self.va.episodes[i].slot_of(r["machine_id"])
            self.assertEqual(pv[r["query_id"]], P[i, slot].tolist())
            self.assertEqual(int(np.argmax(pv[r["query_id"]])), slot % 5)
        path = "/tmp/opencode/test_scnn_align.csv"
        write_submission_csv(path, pv)
        got = read_submission(path, qids)
        for q in qids:
            self.assertEqual(len(got[q]), 5)
            self.assertAlmostEqual(sum(got[q]), 1.0, places=9)

    def test_deterministic_eval(self):
        torch.manual_seed(11)
        m1 = scnn.CNN(h=8, layers=2, W=16)
        torch.manual_seed(11)
        m2 = scnn.CNN(h=8, layers=2, W=16)
        for k in m1.state_dict():
            self.assertTrue(torch.equal(m1.state_dict()[k], m2.state_dict()[k]))
        B = 7
        args = (torch.from_numpy(self.Qv["X"][:B]), torch.from_numpy(self.Qv["T"][:B]), torch.from_numpy(self.Qv["TY"][:B]))
        m1.eval()
        with torch.no_grad():
            a1 = m1(*args)
            a2 = m1(*args)
        self.assertTrue(torch.equal(a1, a2))
        p1 = scnn.predict(m1, self.Qv, "cpu")
        p2 = scnn.predict(m1, self.Qv, "cpu")
        self.assertTrue((p1 == p2).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
