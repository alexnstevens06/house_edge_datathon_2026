"""Tests for the machine_type_hierarchical toy generator.

Known-truth checks inline the Dirichlet / jackpot-heat formula so they do
not only call the helper that generate() uses.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import generate as gen


def _simplex(probs):
    if len(probs) != 5:
        return False
    if any((not math.isfinite(p)) or p < 0.0 or p > 1.0 for p in probs):
        return False
    return abs(math.fsum(probs) - 1.0) <= 1e-12


def _formula(machine_type, machine_base, observed_payouts):
    """Independent copy of the disclosed latent-oracle update."""
    if not observed_payouts:
        return [float(value) for value in machine_base]
    counts = [0, 0, 0, 0, 0]
    for payout in observed_payouts:
        counts[gen.PAYOUT_TO_INDEX[payout]] += 1
    total = float(len(observed_payouts))
    tau = gen.TAU[machine_type]
    blended = [
        (tau * float(machine_base[i]) + counts[i]) / (tau + total) for i in range(5)
    ]
    if machine_type == "jackpot" and observed_payouts[-1] in (5, 20):
        weight = gen.HEAT_MIX
        hot = gen.JACKPOT_HOT
        blended = [(1.0 - weight) * blended[i] + weight * hot[i] for i in range(5)]
    return blended


class MappingAndSimplexTests(unittest.TestCase):
    def test_payout_slot_and_type_maps(self):
        self.assertEqual(gen.PAYOUTS, (0, 1, 2, 5, 20))
        self.assertEqual([gen.PAYOUT_TO_INDEX[p] for p in gen.PAYOUTS], [0, 1, 2, 3, 4])
        self.assertEqual(gen.PAYOUT_TO_INDEX[5], 3)
        self.assertEqual(gen.PAYOUT_TO_INDEX[20], 4)
        self.assertEqual(gen.INDEX_TO_PAYOUT[3], 5)
        self.assertEqual(
            [gen.machine_type_for(i) for i in range(8)],
            list(gen.TYPE_BY_SLOT),
        )
        self.assertEqual(gen.prediction_cutoff(96), 95)
        for name, vector in list(gen.TYPE_PROTOTYPE.items()) + [
            ("house", house) for house in gen.HOUSE_VECTORS
        ] + [("hot", gen.JACKPOT_HOT)]:
            self.assertTrue(_simplex(vector), name)

    def test_generated_rows_match_contract(self):
        bundle = gen.generate(seed=0, n_episodes=2, n_ticks=12, n_machines=8)
        self.assertEqual(len(bundle["records"]), 2 * 12 * 8)
        self.assertEqual(len(bundle["truth"]), len(bundle["records"]))
        self.assertTrue(bundle["research_only"])
        self.assertTrue(bundle["real_only_fallback"])
        self.assertEqual(bundle["eligibility"], "RESEARCH_ONLY")
        seen = set()
        for record, truth in zip(bundle["records"], bundle["truth"]):
            self.assertIn(record["machine_type"], gen.MACHINE_TYPES)
            self.assertEqual(record["machine_type"], gen.machine_type_for(record["machine_id"]))
            self.assertIn(record["payout"], gen.PAYOUTS)
            self.assertIsInstance(record["observed"], bool)
            self.assertNotIn("signal", record)
            key = (record["episode_id"], record["tick"], record["machine_id"])
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(
                (truth["episode_id"], truth["tick"], truth["machine_id"]),
                key,
            )
            self.assertEqual(truth["probability_kind"], gen.PROBABILITY_KIND_ORACLE)
            self.assertTrue(_simplex(truth["probabilities"]))
            self.assertEqual(len(truth["probabilities"]), 5)
        self.assertEqual(len(bundle["queries"]), 16)
        cutoffs = {row["cutoff"] for row in bundle["queries"]}
        self.assertEqual(cutoffs, {11})
        for query in bundle["queries"]:
            self.assertIsNone(query["signal"])
            self.assertEqual(query["machine_type"], gen.machine_type_for(query["machine_id"]))

    def test_default_size_is_within_cap(self):
        bundle = gen.generate()
        self.assertEqual(len(bundle["records"]), 8 * 96 * 8)
        self.assertLessEqual(len(bundle["records"]), gen.MAX_RECORDS)
        self.assertTrue(all(_simplex(row["probabilities"]) for row in bundle["truth"]))

    def test_oversize_refused(self):
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=64, n_ticks=96, n_machines=8)


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_matches(self):
        left = gen.generate(seed=4, n_episodes=2, n_ticks=15, n_machines=8)
        right = gen.generate(seed=4, n_episodes=2, n_ticks=15, n_machines=8)
        self.assertEqual(left["records"], right["records"])
        self.assertEqual(left["truth"], right["truth"])
        self.assertEqual(left["queries"], right["queries"])

    def test_different_seed_changes_draws(self):
        left = gen.generate(seed=0, n_episodes=1, n_ticks=20, n_machines=8)
        right = gen.generate(seed=1, n_episodes=1, n_ticks=20, n_machines=8)
        self.assertNotEqual(left["records"], right["records"])
        left_base = gen.draw_latents(0, 0, 8)["machine_base"]
        right_base = gen.draw_latents(1, 0, 8)["machine_base"]
        self.assertNotEqual(left_base, right_base)


class CausalTests(unittest.TestCase):
    def test_prefix_matches_shorter_run(self):
        short = gen.generate(seed=3, n_episodes=2, n_ticks=9, n_machines=8)
        long = gen.generate(seed=3, n_episodes=3, n_ticks=20, n_machines=8)
        short_records = short["records"]
        long_prefix = [
            row
            for row in long["records"]
            if row["episode_id"] < 2 and row["tick"] < 9
        ]
        self.assertEqual(short_records, long_prefix)
        short_truth = short["truth"]
        long_truth = [
            row
            for row in long["truth"]
            if row["episode_id"] < 2 and row["tick"] < 9
        ]
        self.assertEqual(short_truth, long_truth)
        one = gen.generate(seed=3, n_episodes=1, n_ticks=9, n_machines=8)
        self.assertEqual(
            one["records"],
            [row for row in short_records if row["episode_id"] == 0],
        )

    def test_future_intervention_keeps_prefix_and_past_moves_future(self):
        base = gen.generate(seed=0, n_episodes=1, n_ticks=12, n_machines=8)
        target = next(
            row
            for row in base["records"]
            if row["episode_id"] == 0 and row["tick"] == 6 and row["machine_id"] == 2
        )
        new_payout = 0 if target["payout"] == 20 else 20
        mutated = gen.generate(
            seed=0,
            n_episodes=1,
            n_ticks=12,
            n_machines=8,
            interventions={(0, 6, 2): {"payout": new_payout, "observed": True}},
        )

        def take(rows, tick_test):
            return [row for row in rows if tick_test(row["tick"])]

        self.assertEqual(
            take(base["records"], lambda tick: tick < 6),
            take(mutated["records"], lambda tick: tick < 6),
        )
        self.assertEqual(
            take(base["truth"], lambda tick: tick <= 6),
            take(mutated["truth"], lambda tick: tick <= 6),
        )
        changed = next(
            row
            for row in mutated["records"]
            if row["tick"] == 6 and row["machine_id"] == 2
        )
        self.assertEqual(changed["payout"], new_payout)
        self.assertTrue(changed["observed"])
        self.assertFalse(target["observed"] is True and target["payout"] == new_payout)

        def at(bundle, tick, machine_id):
            return next(
                row["probabilities"]
                for row in bundle["truth"]
                if row["tick"] == tick and row["machine_id"] == machine_id
            )

        # Balanced machine 2 uses the new observed payout. Machine 3 does not.
        self.assertNotEqual(at(base, 7, 2), at(mutated, 7, 2))
        self.assertEqual(at(base, 7, 3), at(mutated, 7, 3))
        # The intervened tick's own oracle was fixed before that payout existed.
        self.assertEqual(at(base, 6, 2), at(mutated, 6, 2))

    def test_recomputed_oracle_ignores_future_and_other_machines(self):
        bundle = gen.generate(seed=2, n_episodes=1, n_ticks=10, n_machines=8)
        latents = gen.draw_latents(2, 0, 8)
        records = [dict(row) for row in bundle["records"]]
        for row in records:
            if row["tick"] >= 5:
                row["payout"] = 20
                row["observed"] = True
            if row["machine_id"] == 0 and row["tick"] < 5:
                row["payout"] = 5
                row["observed"] = True

        for truth in bundle["truth"]:
            if truth["tick"] >= 5:
                continue
            payouts = gen.usable_payouts(
                bundle["records"], truth["episode_id"], truth["machine_id"], truth["tick"]
            )
            # Future edits and other machines are outside usable_payouts.
            expected = _formula(
                truth["machine_type"],
                latents["machine_base"][truth["machine_id"]],
                payouts,
            )
            self.assertEqual(truth["probabilities"], expected)
            leaked = gen.usable_payouts(
                records, truth["episode_id"], truth["machine_id"], truth["tick"]
            )
            if truth["machine_id"] != 0:
                self.assertEqual(leaked, payouts)
                self.assertEqual(
                    _formula(
                        truth["machine_type"],
                        latents["machine_base"][truth["machine_id"]],
                        leaked,
                    ),
                    truth["probabilities"],
                )


class MissingnessTests(unittest.TestCase):
    def test_gap_is_not_a_zero_payout(self):
        base = (0.40, 0.28, 0.18, 0.10, 0.04)
        empty = _formula("balanced", base, [])
        missing_ignored = _formula("balanced", base, [])
        observed_zero = _formula("balanced", base, [0])
        observed_five = _formula("balanced", base, [5])
        self.assertEqual(empty, missing_ignored)
        self.assertNotEqual(empty, observed_zero)
        self.assertNotEqual(observed_zero, observed_five)
        # Slot 3 is payout 5, so a zero must not increase the 5-count.
        self.assertGreater(observed_zero[gen.PAYOUT_TO_INDEX[0]], empty[gen.PAYOUT_TO_INDEX[0]])
        self.assertGreater(observed_five[gen.PAYOUT_TO_INDEX[5]], empty[gen.PAYOUT_TO_INDEX[5]])

        records = [
            {
                "episode_id": 0,
                "tick": 0,
                "machine_id": 2,
                "machine_type": "balanced",
                "payout": 0,
                "observed": False,
            },
            {
                "episode_id": 0,
                "tick": 1,
                "machine_id": 2,
                "machine_type": "balanced",
                "payout": 0,
                "observed": True,
            },
        ]
        self.assertEqual(gen.usable_payouts(records, 0, 2, 2), [0])
        self.assertEqual(gen.usable_payouts(records, 0, 2, 1), [])
        # The gap stores payout 0, and the mask still drops it.
        as_of_tick_1 = gen.observable_history_probabilities(
            "balanced", gen.usable_payouts(records, 0, 2, 1)
        )
        self.assertEqual(as_of_tick_1, gen.observable_history_probabilities("balanced", []))
        if_coded_as_zero = gen.observable_history_probabilities("balanced", [0])
        self.assertNotEqual(as_of_tick_1, if_coded_as_zero)

    def test_generated_stream_has_gaps_and_mask_holds(self):
        bundle = gen.generate(seed=0, n_episodes=1, n_ticks=16, n_machines=8)
        self.assertTrue(any(row["observed"] is False for row in bundle["records"]))
        self.assertTrue(any(row["observed"] is True for row in bundle["records"]))
        latents = gen.draw_latents(0, 0, 8)
        for truth in bundle["truth"]:
            payouts = gen.usable_payouts(
                bundle["records"], truth["episode_id"], truth["machine_id"], truth["tick"]
            )
            expected = _formula(
                truth["machine_type"],
                latents["machine_base"][truth["machine_id"]],
                payouts,
            )
            self.assertEqual(truth["probabilities"], expected)


class HierarchyAndOracleTests(unittest.TestCase):
    def test_type_level_is_shared_and_machines_tilt_it(self):
        latents = gen.draw_latents(0, 0, 8)
        for machine_id, name in enumerate(latents["machine_type"]):
            level = latents["type_level"][name]
            self.assertNotEqual(level, gen.TYPE_PROTOTYPE[name])
            tilted = gen.softmax_tilt(
                level, latents["noise"][machine_id], gen.TYPE_SIGMA[name]
            )
            self.assertEqual(latents["machine_base"][machine_id], tilted)
        # Two steady machines share the type mixture and not the machine base.
        self.assertEqual(latents["machine_type"][0], "steady")
        self.assertEqual(latents["machine_type"][1], "steady")
        self.assertNotEqual(latents["machine_base"][0], latents["machine_base"][1])
        self.assertEqual(latents["machine_type"][5], "jackpot")

    def test_known_truth_matches_inlined_formula(self):
        bundle = gen.generate(seed=7, n_episodes=1, n_ticks=8, n_machines=8)
        latents = gen.draw_latents(7, 0, 8)
        for truth in bundle["truth"]:
            payouts = gen.usable_payouts(
                bundle["records"], 0, truth["machine_id"], truth["tick"]
            )
            expected = _formula(
                truth["machine_type"],
                latents["machine_base"][truth["machine_id"]],
                payouts,
            )
            self.assertEqual(truth["probabilities"], expected)

    def test_jackpot_heat_is_part_of_known_truth(self):
        base = gen.TYPE_PROTOTYPE["jackpot"]
        history = [0, 1, 20]
        heated = _formula("jackpot", base, history)
        cold = _formula("jackpot", base, [0, 1, 2])
        unheated = [
            (
                gen.TAU["jackpot"] * float(base[i])
                + [0, 1, 20].count(gen.PAYOUTS[i])
            )
            / (gen.TAU["jackpot"] + 3)
            for i in range(5)
        ]
        self.assertEqual(heated, gen.latent_oracle_probabilities("jackpot", base, history))
        self.assertNotEqual(heated, unheated)
        self.assertNotEqual(heated, cold)
        self.assertTrue(_simplex(heated))
        # A gap must not be appended as 0 before the heat rule looks at the last payout.
        self.assertEqual(
            gen.latent_oracle_probabilities("jackpot", base, [20]),
            _formula("jackpot", base, [20]),
        )
        self.assertNotEqual(
            gen.latent_oracle_probabilities("jackpot", base, [20]),
            gen.latent_oracle_probabilities("jackpot", base, [20, 0]),
        )

    def test_oracle_differs_from_observable_history(self):
        bundle = gen.generate(seed=0, n_episodes=1, n_ticks=6, n_machines=8)
        latents = gen.draw_latents(0, 0, 8)
        for machine_id in range(8):
            oracle = next(
                row["probabilities"]
                for row in bundle["truth"]
                if row["tick"] == 0 and row["machine_id"] == machine_id
            )
            observable = gen.observable_history_probabilities(
                gen.machine_type_for(machine_id), []
            )
            self.assertEqual(oracle, list(latents["machine_base"][machine_id]))
            self.assertTrue(_simplex(observable))
            self.assertNotEqual(oracle, observable)
            self.assertEqual(observable, list(gen.TYPE_PROTOTYPE[gen.machine_type_for(machine_id)]))
        # Observable path also drops gaps, and it still is not the oracle after history.
        for truth in bundle["truth"]:
            if truth["tick"] == 0:
                continue
            payouts = gen.usable_payouts(bundle["records"], 0, truth["machine_id"], truth["tick"])
            observable = gen.observable_history_probabilities(truth["machine_type"], payouts)
            self.assertTrue(_simplex(observable))
            self.assertNotEqual(truth["probability_kind"], gen.PROBABILITY_KIND_OBSERVABLE)
            self.assertNotEqual(truth["probabilities"], observable)


class WriterTests(unittest.TestCase):
    def test_cli_files_hashes_and_research_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = gen.main(
                ["--seed", "1", "--episodes", "1", "--ticks", "5", "--machines", "8", "--out", str(out)]
            )
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "RESEARCH_ONLY")
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["eligibility"], "unresolved")
            self.assertEqual(manifest["seed"], 1)
            self.assertEqual(manifest["settings"]["n_ticks"], 5)
            self.assertEqual(manifest["settings"]["cutoff"], 4)
            self.assertEqual(manifest["mappings"]["payout_to_slot"]["5"], 3)
            self.assertEqual(manifest["mappings"]["payout_to_slot"]["20"], 4)
            self.assertIn("generate.py", manifest["source_hashes"])
            for name in ("records.jsonl", "truth.jsonl", "queries.jsonl"):
                payload = (out / name).read_bytes()
                self.assertEqual(manifest["output_hashes"][name], gen.sha256_bytes(payload))
            record = json.loads((out / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
            truth = json.loads((out / "truth.jsonl").read_text(encoding="utf-8").splitlines()[0])
            query = json.loads((out / "queries.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["machine_id"], 0)
            self.assertEqual(len(truth["probabilities"]), 5)
            self.assertEqual(truth["probability_kind"], "latent_oracle")
            self.assertIsNone(query["signal"])
            self.assertEqual(query["cutoff"], 4)
            self.assertEqual(manifest["n_records"], 40)


if __name__ == "__main__":
    unittest.main()
