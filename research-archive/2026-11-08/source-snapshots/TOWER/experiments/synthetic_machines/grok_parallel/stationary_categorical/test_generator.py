#!/usr/bin/env python3
"""Meaningful checks for the stationary_categorical generator."""

from __future__ import annotations

import copy
import json
import math
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import generate as gen

ROOT = Path(__file__).resolve().parent
PYTHON = "/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python"


def _probs_close(left, right, tol=1e-12):
    return len(left) == len(right) and all(math.isclose(a, b, abs_tol=tol) for a, b in zip(left, right))


class FixedRNG(random.Random):
    def __init__(self, value: float):
        super().__init__(0)
        self._value = value

    def random(self) -> float:
        return self._value


class TestMappingsAndSimplex(unittest.TestCase):
    def test_payout_index_order_and_columns(self):
        self.assertEqual(gen.PAYOUTS, (0, 1, 2, 5, 20))
        self.assertEqual(list(gen.PAYOUT_TO_INDEX.values()), [0, 1, 2, 3, 4])
        self.assertEqual(gen.PROBABILITY_COLUMNS, ("p_0", "p_1", "p_2", "p_5", "p_20"))
        self.assertEqual(gen.MISSING, -1)
        self.assertNotEqual(gen.MISSING, gen.PAYOUTS[0])
        self.assertEqual(gen.TYPE_TO_INDEX, {"balanced": 0, "jackpot": 1, "steady": 2})

    def test_sample_index_respects_ordered_bins(self):
        probabilities = (0.2, 0.2, 0.2, 0.2, 0.2)
        draws = {
            0.0: 0,
            0.2: 0,
            0.2000001: 1,
            0.4: 1,
            0.4000001: 2,
            0.6: 2,
            0.6000001: 3,
            0.8: 3,
            0.8000001: 4,
            0.999: 4,
        }
        for unit, expected in draws.items():
            index = gen.sample_index(FixedRNG(unit), probabilities)
            self.assertEqual(gen.PAYOUTS[index], gen.PAYOUTS[expected])
            self.assertEqual(index, gen.PAYOUT_TO_INDEX[gen.PAYOUTS[expected]])

    def test_oracle_vectors_are_simplexes_and_type_specific(self):
        seen = []
        for machine_type in gen.MACHINE_TYPES:
            probabilities = gen.oracle_probabilities(machine_type)
            self.assertEqual(len(probabilities), 5)
            self.assertTrue(all(p > 0.0 for p in probabilities))
            self.assertTrue(math.isclose(math.fsum(probabilities), 1.0, abs_tol=gen.SUM_TOLERANCE))
            seen.append(probabilities)
        self.assertFalse(_probs_close(seen[0], seen[1]))
        self.assertFalse(_probs_close(seen[0], seen[2]))
        self.assertFalse(_probs_close(seen[1], seen[2]))

    def test_generated_truth_matches_oracle_mapping(self):
        panel = gen.generate(seed=0, n_episodes=1, n_ticks=6, n_machines=8)
        self.assertEqual(len(panel["truth"]), len(panel["records"]))
        for record, truth in zip(panel["records"], panel["truth"]):
            self.assertEqual(record["episode_id"], truth["episode_id"])
            self.assertEqual(record["tick"], truth["tick"])
            self.assertEqual(record["machine_id"], truth["machine_id"])
            self.assertEqual(record["slot_index"], record["machine_id"])
            self.assertIn(record["machine_id"], range(8))
            self.assertEqual(len(truth["probabilities"]), 5)
            self.assertTrue(math.isclose(math.fsum(truth["probabilities"]), 1.0, abs_tol=1e-12))
            self.assertTrue(
                _probs_close(truth["probabilities"], gen.oracle_probabilities(record["machine_type"]))
            )
            self.assertEqual(record["observation_mask"], 1 if record["observed"] else 0)
            if record["observed"]:
                self.assertIn(record["payout"], gen.PAYOUTS)
                self.assertEqual(record["payout_index"], gen.PAYOUT_TO_INDEX[record["payout"]])
                self.assertEqual(record["dense_symbol"], record["payout_index"])
            else:
                self.assertIsNone(record["payout"])
                self.assertIsNone(record["payout_index"])
                self.assertEqual(record["dense_symbol"], gen.MISSING)
                self.assertNotEqual(record["dense_symbol"], gen.PAYOUT_TO_INDEX[0])
            self.assertNotIn("signal", record)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_matches_including_independent_cells(self):
        first = gen.generate(seed=4, n_episodes=2, n_ticks=12, n_machines=8)
        second = gen.generate(seed=4, n_episodes=2, n_ticks=12, n_machines=8)
        self.assertEqual(first, second)
        for record in first["records"]:
            again = gen.realize_cell(
                4,
                record["episode_id"],
                record["tick"],
                record["machine_id"],
                record["machine_type"],
            )
            self.assertEqual(again, record)

    def test_different_seed_changes_panel(self):
        left = gen.generate(seed=0, n_episodes=2, n_ticks=16, n_machines=8)
        right = gen.generate(seed=1, n_episodes=2, n_ticks=16, n_machines=8)
        self.assertNotEqual(left["records"], right["records"])

    def test_default_size_is_within_cap(self):
        panel = gen.generate()
        self.assertEqual(len(panel["records"]), 8 * 96 * 8)
        self.assertLessEqual(len(panel["records"]), gen.MAX_RECORDS)
        self.assertEqual(len(panel["truth"]), len(panel["records"]))
        with self.assertRaises(ValueError):
            gen.generate(n_episodes=8, n_ticks=200, n_machines=8)


class TestKnownTruthInvariant(unittest.TestCase):
    def test_truth_is_constant_across_ticks_and_ignores_spins(self):
        panel = gen.generate(seed=3, n_episodes=2, n_ticks=20, n_machines=8)
        by_machine = {}
        for row in panel["truth"]:
            key = (row["episode_id"], row["machine_id"])
            by_machine.setdefault(key, []).append(row)
        self.assertEqual(len(by_machine), 16)
        type_rows = {name: [] for name in gen.MACHINE_TYPES}
        for rows in by_machine.values():
            vectors = {tuple(row["probabilities"]) for row in rows}
            self.assertEqual(len(vectors), 1)
            ticks = [row["tick"] for row in rows]
            self.assertEqual(ticks, list(range(20)))
            machine_type = rows[0]["machine_type"]
            self.assertTrue(_probs_close(rows[0]["probabilities"], gen.oracle_probabilities(machine_type)))
            type_rows[machine_type].append(rows[0]["probabilities"])
        for machine_type, vectors in type_rows.items():
            self.assertGreaterEqual(len(vectors), 1)
            self.assertTrue(all(_probs_close(vector, gen.oracle_probabilities(machine_type)) for vector in vectors))

    def test_oracle_does_not_read_history_or_signal(self):
        panel = gen.generate(seed=3, n_episodes=1, n_ticks=8, n_machines=8)
        observed = [row for row in panel["records"] if row["observed"]]
        self.assertGreater(len(observed), 0)
        machine_id = observed[0]["machine_id"]
        mutated = copy.deepcopy(panel["records"])
        changed = 0
        for record in mutated:
            if record["observed"] and record["machine_id"] == machine_id:
                record["payout"] = 20 if record["payout"] != 20 else 0
                record["payout_index"] = gen.PAYOUT_TO_INDEX[record["payout"]]
                record["dense_symbol"] = record["payout_index"]
                changed += 1
        self.assertGreater(changed, 0)
        for truth in panel["truth"]:
            self.assertTrue(
                _probs_close(truth["probabilities"], gen.oracle_probabilities(truth["machine_type"]))
            )
        before = gen.observable_history_probabilities(panel["records"], 0, 8, machine_id, signal=-1)
        after = gen.observable_history_probabilities(mutated, 0, 8, machine_id, signal=1)
        oracle = gen.oracle_probabilities(
            next(row["machine_type"] for row in panel["records"] if row["machine_id"] == machine_id)
        )
        self.assertFalse(_probs_close(before, after))
        self.assertFalse(_probs_close(before, oracle))
        self.assertTrue(
            all(_probs_close(row["probabilities"], oracle) for row in panel["truth"] if row["machine_id"] == machine_id)
        )


class TestHistoryCausalityAndGaps(unittest.TestCase):
    def setUp(self):
        self.panel = gen.generate(seed=7, n_episodes=1, n_ticks=24, n_machines=8, observe_prob=0.65)
        self.records = self.panel["records"]

    def test_missing_is_not_zero_and_both_exist(self):
        missing = [row for row in self.records if not row["observed"]]
        zeros = [row for row in self.records if row["observed"] and row["payout"] == 0]
        self.assertGreater(len(missing), 0)
        self.assertGreater(len(zeros), 0)
        self.assertTrue(all(row["payout"] is None for row in missing))
        self.assertTrue(all(row["dense_symbol"] == gen.MISSING for row in missing))
        self.assertTrue(all(row["dense_symbol"] == 0 and row["observation_mask"] == 1 for row in zeros))
        self.assertNotEqual(missing[0]["dense_symbol"], zeros[0]["dense_symbol"])

    def test_gappy_missing_does_not_count_as_zero(self):
        target = next(row for row in self.records if not row["observed"])
        tick = target["tick"] + 1
        baseline = gen.observable_history_probabilities(
            self.records, target["episode_id"], tick, target["machine_id"]
        )
        forged = copy.deepcopy(self.records)
        for row in forged:
            if (
                row["episode_id"] == target["episode_id"]
                and row["machine_id"] == target["machine_id"]
                and row["tick"] == target["tick"]
            ):
                row["payout"] = 0
                row["payout_index"] = 0
                row["dense_symbol"] = gen.MISSING
                row["observed"] = False
                row["observation_mask"] = 0
        ignored = gen.observable_history_probabilities(
            forged, target["episode_id"], tick, target["machine_id"]
        )
        self.assertTrue(_probs_close(baseline, ignored))
        for row in forged:
            if (
                row["episode_id"] == target["episode_id"]
                and row["machine_id"] == target["machine_id"]
                and row["tick"] == target["tick"]
            ):
                row["observed"] = True
                row["observation_mask"] = 1
                row["payout"] = 0
                row["payout_index"] = 0
                row["dense_symbol"] = 0
        counted = gen.observable_history_probabilities(
            forged, target["episode_id"], tick, target["machine_id"]
        )
        self.assertFalse(_probs_close(baseline, counted))

    def test_future_and_contemporaneous_mutation_leave_prefix_unchanged(self):
        episode_id = 0
        machine_id = 0
        tick = 11
        prefix = gen.observable_history_probabilities(self.records, episode_id, tick, machine_id)
        only_prefix = [row for row in self.records if row["tick"] < tick]
        self.assertTrue(
            _probs_close(prefix, gen.observable_history_probabilities(only_prefix, episode_id, tick, machine_id))
        )
        shuffled = copy.deepcopy(only_prefix)
        random.Random(0).shuffle(shuffled)
        self.assertTrue(
            _probs_close(prefix, gen.observable_history_probabilities(shuffled, episode_id, tick, machine_id))
        )

        future = copy.deepcopy(self.records)
        mutated_future = 0
        for row in future:
            if row["tick"] >= tick and row["observed"] and row["machine_id"] == machine_id:
                row["payout"] = 20 if row["payout"] != 20 else 5
                row["payout_index"] = gen.PAYOUT_TO_INDEX[row["payout"]]
                row["dense_symbol"] = row["payout_index"]
                mutated_future += 1
        self.assertGreater(mutated_future, 0)
        self.assertTrue(
            _probs_close(prefix, gen.observable_history_probabilities(future, episode_id, tick, machine_id))
        )

        other = copy.deepcopy(self.records)
        mutated_other = 0
        for row in other:
            if row["tick"] < tick and row["machine_id"] != machine_id and row["observed"]:
                row["payout"] = 20 if row["payout"] != 20 else 0
                row["payout_index"] = gen.PAYOUT_TO_INDEX[row["payout"]]
                row["dense_symbol"] = row["payout_index"]
                mutated_other += 1
        self.assertGreater(mutated_other, 0)
        self.assertTrue(
            _probs_close(prefix, gen.observable_history_probabilities(other, episode_id, tick, machine_id))
        )

        past = copy.deepcopy(self.records)
        changed = False
        for row in past:
            if (
                row["episode_id"] == episode_id
                and row["machine_id"] == machine_id
                and row["tick"] < tick
                and row["observed"]
            ):
                row["payout"] = 20 if row["payout"] != 20 else 0
                row["payout_index"] = gen.PAYOUT_TO_INDEX[row["payout"]]
                row["dense_symbol"] = row["payout_index"]
                changed = True
                break
        self.assertTrue(changed)
        self.assertFalse(
            _probs_close(prefix, gen.observable_history_probabilities(past, episode_id, tick, machine_id))
        )

    def test_same_prefix_matches_and_signal_is_ignored(self):
        episode_id = 0
        machine_id = 2
        tick = 9
        left = gen.observable_history_probabilities(self.records, episode_id, tick, machine_id, signal=-1)
        right = gen.observable_history_probabilities(self.records, episode_id, tick, machine_id, signal=1)
        none_signal = gen.observable_history_probabilities(self.records, episode_id, tick, machine_id, signal=None)
        self.assertEqual(left, right)
        self.assertEqual(left, none_signal)
        twin = copy.deepcopy(self.records)
        for row in twin:
            if row["tick"] >= tick:
                row["payout"] = 0
                row["payout_index"] = 0 if row["observed"] else None
                row["dense_symbol"] = 0 if row["observed"] else gen.MISSING
        self.assertEqual(
            left,
            gen.observable_history_probabilities(twin, episode_id, tick, machine_id, signal=0),
        )

    def test_latent_oracle_differs_from_observable_history(self):
        record = self.records[0]
        oracle = gen.oracle_probabilities(record["machine_type"])
        at_start = gen.observable_history_probabilities(
            self.records, record["episode_id"], 0, record["machine_id"]
        )
        uniform = tuple(1.0 / 5.0 for _ in gen.PAYOUTS)
        self.assertTrue(_probs_close(at_start, uniform))
        self.assertFalse(_probs_close(at_start, oracle))
        later = gen.observable_history_probabilities(
            self.records, record["episode_id"], 24, record["machine_id"]
        )
        self.assertFalse(_probs_close(later, oracle))
        self.assertFalse(_probs_close(later, uniform))
        stored = next(
            row["probabilities"]
            for row in self.panel["truth"]
            if row["machine_id"] == record["machine_id"] and row["tick"] == 23
        )
        self.assertTrue(_probs_close(stored, oracle))
        self.assertFalse(_probs_close(stored, later))


class TestCLI(unittest.TestCase):
    def test_cli_writes_hashed_research_only_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            completed = subprocess.run(
                [
                    PYTHON,
                    str(ROOT / "generate.py"),
                    "--seed",
                    "2",
                    "--episodes",
                    "1",
                    "--ticks",
                    "4",
                    "--out",
                    str(out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("RESEARCH_ONLY", completed.stdout)
            records = (out / "records.jsonl").read_bytes()
            truth = (out / "truth.jsonl").read_bytes()
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["label"], "RESEARCH_ONLY")
            self.assertTrue(manifest["research_only"])
            self.assertEqual(manifest["eligibility"]["competition_fallback"], "organizer_released_only")
            self.assertTrue(manifest["eligibility"]["real_only_fallback"])
            self.assertEqual(manifest["eligibility"]["status"], "unresolved")
            self.assertEqual(manifest["seed"], 2)
            self.assertEqual(manifest["settings"]["n_episodes"], 1)
            self.assertEqual(manifest["settings"]["n_ticks"], 4)
            self.assertEqual(manifest["settings"]["prediction_cutoff"], 4)
            self.assertEqual(manifest["n_records"], 32)
            self.assertEqual(manifest["hashes"]["outputs"]["records.jsonl"], gen.sha256_bytes(records))
            self.assertEqual(manifest["hashes"]["outputs"]["truth.jsonl"], gen.sha256_bytes(truth))
            self.assertEqual(manifest["hashes"]["source"]["generate.py"], gen.sha256_file(ROOT / "generate.py"))
            self.assertNotIn("signal", (out / "records.jsonl").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in records.decode("utf-8").splitlines()]
            self.assertTrue(any(row["payout"] is None for row in rows))
            self.assertTrue(any(row["observed"] and row["payout"] == 0 for row in rows))
            self.assertTrue(all(row["tick"] < 4 for row in rows))


if __name__ == "__main__":
    unittest.main()
