"""Tests for the shared_latent_crossmachine toy generator.

Run from this directory:

    /home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py -v
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import generate as gen

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _simplex(probs, places=9):
    assert len(probs) == 5, probs
    total = math.fsum(probs)
    assert math.isclose(total, 1.0, abs_tol=10 ** (-places)), total
    assert all(p >= -1e-15 for p in probs)
    assert all(math.isfinite(p) for p in probs)


def _blank(n_ticks=4, n_machines=8):
    return [[None] * n_machines for _ in range(n_ticks)]


class TestMappingsAndSimplex(unittest.TestCase):
    def test_payout_order_and_weights(self):
        self.assertEqual(gen.PAYOUTS, (0, 1, 2, 5, 20))
        self.assertEqual([gen.INDEX_TO_PAYOUT[i] for i in range(5)], [0, 1, 2, 5, 20])
        self.assertEqual(gen.PAYOUT_TO_INDEX[20], 4)
        self.assertIsNone(gen.MISSING)
        for rows in gen.EMISSION_WEIGHTS.values():
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(sum(row), 100)
                self.assertTrue(all(w > 0 for w in row))

    def test_generated_contract(self):
        out = gen.generate(seed=0, n_episodes=2, n_ticks=12, n_machines=8)
        self.assertEqual(len(out["records"]), 2 * 12 * 8)
        self.assertEqual(len(out["truth"]), len(out["records"]))
        self.assertEqual(out["status"], "RESEARCH_ONLY")
        self.assertTrue(out["real_only_fallback"])
        self.assertEqual(out["eligibility"], "unresolved")
        kinds = {row["machine_type"] for row in out["records"]}
        self.assertEqual(kinds, {"steady", "balanced", "jackpot"})
        for record, truth in zip(out["records"], out["truth"]):
            self.assertEqual(record["episode_id"], truth["episode_id"])
            self.assertEqual(record["tick"], truth["tick"])
            self.assertEqual(record["machine_id"], truth["machine_id"])
            self.assertEqual(record["slot"], record["machine_id"])
            self.assertIn(record["machine_id"], range(8))
            self.assertEqual(record["machine_type"], gen.machine_type(record["machine_id"]))
            self.assertEqual(record["payout"], gen.INDEX_TO_PAYOUT[record["payout_index"]])
            self.assertIn(record["payout"], gen.PAYOUTS)
            self.assertIsInstance(record["observed"], bool)
            self.assertNotIn("signal", record)
            self.assertNotIn("latent", record)
            self.assertEqual(truth["probability_kind"], "latent_oracle")
            self.assertEqual(truth["observable_probability_kind"], "observable_history")
            self.assertEqual(len(truth["probabilities"]), 5)
            self.assertEqual(len(truth["observable_probabilities"]), 5)
            _simplex(truth["probabilities"])
            _simplex(truth["observable_probabilities"])
            self.assertEqual(
                truth["probabilities"],
                gen.oracle_probabilities(truth["latent"], record["machine_type"], truth["loading_sign"]),
            )
        for query in out["queries"]:
            self.assertIsNone(query["signal"])
            self.assertEqual(query["cutoff"], 12)
            self.assertEqual(query["slot"], query["machine_id"])
            _simplex(query["latent_oracle_probabilities"])
            _simplex(query["observable_history_probabilities"])
        ticks = {r["tick"] for r in out["records"]}
        self.assertTrue(all(t < 12 for t in ticks))

    def test_default_size_and_cap(self):
        out = gen.generate()
        self.assertEqual(len(out["records"]), 8 * 96 * 8)
        self.assertLessEqual(len(out["records"]), gen.MAX_RECORDS)
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=3, n_ticks=100, n_machines=30)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_matches(self):
        a = gen.generate(seed=0, n_episodes=2, n_ticks=20, n_machines=8)
        b = gen.generate(seed=0, n_episodes=2, n_ticks=20, n_machines=8)
        self.assertEqual(a["records"], b["records"])
        self.assertEqual(a["truth"], b["truth"])
        self.assertEqual(a["queries"], b["queries"])

    def test_different_seed_changes_draws(self):
        a = gen.generate(seed=0, n_episodes=1, n_ticks=16, n_machines=8)
        b = gen.generate(seed=1, n_episodes=1, n_ticks=16, n_machines=8)
        self.assertNotEqual(a["records"], b["records"])

    def test_prefix_stable_when_horizon_grows(self):
        short = gen.generate(seed=4, n_episodes=2, n_ticks=24, n_machines=8)
        long = gen.generate(seed=4, n_episodes=2, n_ticks=48, n_machines=8)
        short_recs = short["records"]
        long_prefix = [r for r in long["records"] if r["tick"] < 24]
        self.assertEqual(short_recs, long_prefix)
        short_truth = short["truth"]
        long_truth = [t for t in long["truth"] if t["tick"] < 24]
        self.assertEqual(short_truth, long_truth)


class TestCausality(unittest.TestCase):
    def test_same_tick_does_not_change_current_probs_and_does_change_next(self):
        base = _blank(4)
        base[1][0] = 4
        other = copy.deepcopy(base)
        other[1][3] = 0  # contemporaneous other machine
        panel_a = gen.oracle_panel(1, base)
        panel_b = gen.oracle_panel(1, other)
        self.assertEqual(panel_a[0], panel_b[0])
        self.assertEqual(panel_a[1], panel_b[1])
        self.assertNotEqual(
            [row["probabilities"] for row in panel_a[2]],
            [row["probabilities"] for row in panel_b[2]],
        )
        # Machine 0's next distribution moved because machine 3 was observed earlier.
        self.assertNotEqual(panel_a[2][0]["probabilities"], panel_b[2][0]["probabilities"])

    def test_future_mutation_leaves_prefix(self):
        base = _blank(6)
        base[2][1] = 4
        mutated = copy.deepcopy(base)
        for tick in range(4, 6):
            for machine in range(8):
                mutated[tick][machine] = 4
        panel_a = gen.oracle_panel(0, base)
        panel_b = gen.oracle_panel(0, mutated)
        self.assertEqual(panel_a[:4], panel_b[:4])
        # Last-tick edits do not change any probability inside the panel.
        only_last = copy.deepcopy(base)
        only_last[5][4] = 0
        self.assertEqual(gen.oracle_panel(0, base), gen.oracle_panel(0, only_last))

    def test_generated_truth_matches_causal_replay(self):
        out = gen.generate(seed=5, n_episodes=1, n_ticks=16, n_machines=8)
        obs = gen.observations_from_records(out["records"], 16, 8, 0)
        panel = gen.oracle_panel(gen.initial_latent(5, 0), obs)
        history = gen.history_panel(obs)
        for truth, oracle_row, hist_probs in zip(out["truth"], (r for tick in panel for r in tick), (p for tick in history for p in tick)):
            self.assertEqual(truth["latent"], oracle_row["latent"])
            self.assertEqual(truth["probabilities"], oracle_row["probabilities"])
            self.assertEqual(truth["observable_probabilities"], hist_probs)
        mutated = copy.deepcopy(obs)
        for tick in range(10, 16):
            for machine in range(8):
                mutated[tick][machine] = 0 if obs[tick][machine] != 0 else 4
        replay = gen.oracle_panel(gen.initial_latent(5, 0), mutated)
        self.assertEqual(replay[:10], panel[:10])

    def test_cross_machine_lag_is_shared(self):
        quiet = _blank(3)
        lagged = _blank(3)
        lagged[0][7] = 4
        quiet_panel = gen.oracle_panel(1, quiet)
        lagged_panel = gen.oracle_panel(1, lagged)
        self.assertEqual(quiet_panel[0][0]["probabilities"], lagged_panel[0][0]["probabilities"])
        self.assertNotEqual(quiet_panel[1][0]["latent"], lagged_panel[1][0]["latent"])
        # Every machine shares the new latent.
        latents = {row["latent"] for row in lagged_panel[1]}
        self.assertEqual(latents, {lagged_panel[1][0]["latent"]})


class TestMissingVersusZero(unittest.TestCase):
    def test_missing_hold_differs_from_observed_zero(self):
        missing = _blank(2)
        zero = _blank(2)
        zero[0][0] = 0
        hidden_high = _blank(2)
        # A high draw that was not observed must not enter the score.
        self.assertIsNone(hidden_high[0][0])
        hold = gen.oracle_panel(1, missing)
        down = gen.oracle_panel(1, zero)
        self.assertEqual(hold[1][0]["latent"], 1)
        self.assertEqual(down[1][0]["latent"], 0)
        self.assertNotEqual(hold[1][0]["probabilities"], down[1][0]["probabilities"])
        # Coding the hidden high payout as 0 would cool the latent; leaving it missing does not.
        as_zero = _blank(2)
        as_zero[0][2] = 0
        as_missing = _blank(2)
        self.assertEqual(gen.next_latent(1, as_missing[0]), 1)
        self.assertEqual(gen.next_latent(1, as_zero[0]), 0)
        self.assertEqual(gen.next_latent(1, [4]), 2)
        self.assertEqual(gen.next_latent(1, [None, None, None]), 1)

    def test_stream_has_gaps_and_real_zeros(self):
        out = gen.generate(seed=0, n_episodes=2, n_ticks=40, n_machines=8)
        missing = [r for r in out["records"] if not r["observed"]]
        zeros = [r for r in out["records"] if r["observed"] and r["payout"] == 0]
        missing_nonzero = [r for r in missing if r["payout"] != 0]
        self.assertTrue(missing)
        self.assertTrue(zeros)
        self.assertTrue(missing_nonzero)
        obs = gen.observations_from_records(out["records"], 40, 8, 0)
        for record in out["records"]:
            if record["episode_id"] != 0:
                continue
            cell = obs[record["tick"]][record["machine_id"]]
            if record["observed"]:
                self.assertEqual(cell, record["payout_index"])
            else:
                self.assertIsNone(cell)

    def test_zero_fill_of_gaps_changes_later_latent(self):
        # One high observed payout would heat the latent; a gap must not.
        # Filling that gap with 0 would cool it instead.
        gap = _blank(2)
        filled = _blank(2)
        filled[0][4] = 0
        self.assertNotEqual(
            gen.oracle_panel(1, gap)[1][0]["latent"],
            gen.oracle_panel(1, filled)[1][0]["latent"],
        )


class TestOracleVersusHistory(unittest.TestCase):
    def test_tick0_history_ignores_true_latent(self):
        different = []
        for episode in range(8):
            different.append(gen.initial_latent(0, episode))
        self.assertGreater(len(set(different)), 1)
        out = gen.generate(seed=0, n_episodes=8, n_ticks=2, n_machines=8)
        by_episode = {}
        for truth in out["truth"]:
            if truth["tick"] == 0 and truth["machine_id"] == 0:
                by_episode[truth["episode_id"]] = truth
        oracles = {ep: row["probabilities"] for ep, row in by_episode.items()}
        observables = {ep: row["observable_probabilities"] for ep, row in by_episode.items()}
        self.assertGreater(len(set(tuple(p) for p in oracles.values())), 1)
        self.assertEqual(len(set(tuple(p) for p in observables.values())), 1)
        for row in by_episode.values():
            self.assertNotEqual(row["probabilities"], row["observable_probabilities"])

    def test_opposite_loading_table(self):
        pos = gen.oracle_probabilities(0, "steady", 1)
        neg = gen.oracle_probabilities(0, "steady", -1)
        self.assertEqual(pos, gen.probs_from_weights(gen.EMISSION_WEIGHTS["steady"][0]))
        self.assertEqual(neg, gen.probs_from_weights(gen.EMISSION_WEIGHTS["steady"][2]))
        self.assertNotEqual(pos, neg)
        # Neutral latent is a fixed point of the flip.
        self.assertEqual(
            gen.oracle_probabilities(1, "jackpot", 1),
            gen.oracle_probabilities(1, "jackpot", -1),
        )

    def test_filter_uses_observed_index_not_gap(self):
        belief = list(gen.UNIFORM_BELIEF)
        high = [None] * 8
        high[0] = 4  # steady, positive loading, payout index of 20
        gap = [None] * 8
        after_high = gen.update_belief(belief, high)
        after_gap = gen.update_belief(belief, gap)
        self.assertNotEqual(after_high, after_gap)
        # A gap leaves the uniform prior in place (score missing => latent holds for every hypothesis).
        for mass, expected in zip(after_gap, gen.UNIFORM_BELIEF):
            self.assertTrue(math.isclose(mass, expected, abs_tol=1e-12))


class TestKnownTruthInvariant(unittest.TestCase):
    def test_shared_latent_and_emission_identity(self):
        out = gen.generate(seed=2, n_episodes=2, n_ticks=30, n_machines=8)
        grouped = {}
        saw_extreme = False
        for record, truth in zip(out["records"], out["truth"]):
            key = (record["episode_id"], record["tick"])
            grouped.setdefault(key, set()).add(truth["latent"])
            expected = gen.oracle_probabilities(truth["latent"], record["machine_type"], truth["loading_sign"])
            self.assertEqual(truth["probabilities"], expected)
            if truth["latent"] != 1 and record["machine_id"] in (0, 1):
                saw_extreme = True
        self.assertTrue(all(len(vals) == 1 for vals in grouped.values()))
        self.assertTrue(saw_extreme)
        # Same type, opposite sign, non-neutral latent => different oracle rows.
        pairs = {}
        for record, truth in zip(out["records"], out["truth"]):
            if record["machine_id"] in (0, 1) and truth["latent"] != 1:
                pairs.setdefault((record["episode_id"], record["tick"]), {})[record["machine_id"]] = truth
        self.assertTrue(pairs)
        sample = next(iter(pairs.values()))
        self.assertNotEqual(sample[0]["probabilities"], sample[1]["probabilities"])
        self.assertEqual(sample[0]["latent"], sample[1]["latent"])


class TestCLI(unittest.TestCase):
    def test_cli_writes_hashed_research_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            proc = subprocess.run(
                [
                    PY,
                    str(ROOT / "generate.py"),
                    "--seed",
                    "3",
                    "--episodes",
                    "1",
                    "--ticks",
                    "4",
                    "--machines",
                    "8",
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            self.assertEqual(summary["status"], "RESEARCH_ONLY")
            self.assertEqual(summary["record_count"], 32)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "RESEARCH_ONLY")
            self.assertTrue(manifest["research_only"])
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["eligibility"], "unresolved")
            self.assertFalse(manifest["settings"]["fitted_to_real_data"])
            self.assertEqual(manifest["seed"], 3)
            for name in ("records.jsonl", "truth.jsonl", "queries.jsonl"):
                digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
                self.assertEqual(manifest["output_sha256"][name], digest)
            source = hashlib.sha256((ROOT / "generate.py").read_bytes()).hexdigest()
            self.assertEqual(manifest["source_sha256"]["generate.py"], source)
            rows = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
            truths = [json.loads(line) for line in (out / "truth.jsonl").read_text().splitlines()]
            queries = [json.loads(line) for line in (out / "queries.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 32)
            self.assertEqual(len(truths), 32)
            self.assertEqual(len(queries), 8)
            self.assertTrue(all(q["signal"] is None and q["cutoff"] == 4 for q in queries))
            self.assertTrue(all(r["tick"] < 4 for r in rows))
            for truth in truths:
                _simplex(truth["probabilities"])
                _simplex(truth["observable_probabilities"])
                self.assertEqual(truth["probability_kind"], "latent_oracle")
                self.assertEqual(truth["observable_probability_kind"], "observable_history")


if __name__ == "__main__":
    unittest.main()
