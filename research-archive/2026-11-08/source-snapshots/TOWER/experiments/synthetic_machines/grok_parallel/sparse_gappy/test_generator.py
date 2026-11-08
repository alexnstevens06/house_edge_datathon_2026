#!/usr/bin/env python3
"""Meaningful checks for the sparse_gappy toy generator. Standard library only."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import generate  # noqa: E402


PYTHON = "/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python"
PAYOUTS = generate.PAYOUTS


def _max_abs(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def _is_simplex(probs: list[float]) -> bool:
    if len(probs) != 5:
        return False
    if any(p < -1e-12 for p in probs):
        return False
    return abs(sum(probs) - 1.0) <= 1e-12


def _index_records(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["episode_id"], row["tick"], row["machine_id"]): row for row in rows}


class ProbabilityMappingTests(unittest.TestCase):
    def test_payout_index_map_is_explicit(self):
        self.assertEqual(PAYOUTS, (0, 1, 2, 5, 20))
        self.assertEqual(
            [generate.PAYOUT_TO_IDX[payout] for payout in PAYOUTS],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            generate.PROBABILITY_COLUMNS,
            ("p_0", "p_1", "p_2", "p_5", "p_20"),
        )
        self.assertEqual(generate.MISSING_INDEX, -1)

    def test_symbol_map_separates_missing_from_payout_zero(self):
        hidden = {
            "payout": 0,
            "observed": False,
        }
        revealed_zero = {"payout": 0, "observed": True}
        revealed_twenty = {"payout": 20, "observed": True}
        self.assertIsNone(generate.observable_payout(hidden))
        self.assertEqual(generate.observation_symbol(hidden), -1)
        self.assertEqual(generate.observable_payout(revealed_zero), 0)
        self.assertEqual(generate.observation_symbol(revealed_zero), 0)
        self.assertEqual(generate.observation_symbol(revealed_twenty), 4)
        self.assertNotEqual(
            generate.observation_symbol(hidden),
            generate.observation_symbol(revealed_zero),
        )

    def test_truth_and_observer_rows_are_simplices_aligned_to_payouts(self):
        data = generate.generate(seed=0, n_episodes=2, n_ticks=12, n_machines=8)
        self.assertEqual(len(data["records"]), len(data["truth"]))
        self.assertEqual(len(data["truth"]), len(data["observable_history"]))
        records = _index_records(data["records"])
        for truth, observed in zip(data["truth"], data["observable_history"]):
            self.assertEqual(
                (truth["episode_id"], truth["tick"], truth["machine_id"]),
                (observed["episode_id"], observed["tick"], observed["machine_id"]),
            )
            self.assertTrue(_is_simplex(truth["probabilities"]))
            self.assertTrue(_is_simplex(observed["probabilities"]))
            record = records[(truth["episode_id"], truth["tick"], truth["machine_id"])]
            self.assertIn(record["payout"], PAYOUTS)
            self.assertEqual(
                generate.PAYOUT_TO_IDX[record["payout"]],
                PAYOUTS.index(record["payout"]),
            )
            self.assertIn(record["machine_type"], {"steady", "balanced", "jackpot"})
            self.assertIs(record["observed"], True if record["observed"] else False)

    def test_slot_and_query_cutoff_mapping(self):
        n_ticks = 10
        n_machines = 8
        data = generate.generate(seed=1, n_episodes=2, n_ticks=n_ticks, n_machines=n_machines)
        self.assertEqual(len(data["queries"]), 2 * n_machines)
        cutoffs = set()
        for query in data["queries"]:
            self.assertEqual(query["slot"], query["machine_id"])
            self.assertGreaterEqual(query["machine_id"], 0)
            self.assertLess(query["machine_id"], n_machines)
            self.assertEqual(
                query["machine_type"],
                generate.machine_type_for_slot(query["slot"]),
            )
            self.assertEqual(query["cutoff"], n_ticks)
            self.assertIsNone(query["signal"])
            self.assertTrue(query["answer_key"])
            cutoffs.add((query["episode_id"], query["cutoff"]))
            self.assertTrue(_is_simplex(query["latent_oracle_probabilities"]))
            self.assertTrue(_is_simplex(query["observable_history_probabilities"]))
        for record in data["records"]:
            self.assertNotIn("signal", record)
            self.assertLess(record["tick"], n_ticks)
            self.assertGreaterEqual(record["machine_id"], 0)
            self.assertLess(record["machine_id"], n_machines)
        # One cutoff per episode, and it is strictly after every history tick.
        self.assertEqual(cutoffs, {(0, n_ticks), (1, n_ticks)})


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_matches(self):
        kwargs = {"seed": 0, "n_episodes": 2, "n_ticks": 16, "n_machines": 8}
        self.assertEqual(generate.generate(**kwargs), generate.generate(**kwargs))

    def test_different_seed_changes_draws(self):
        left = generate.generate(seed=0, n_episodes=1, n_ticks=16, n_machines=8)
        right = generate.generate(seed=1, n_episodes=1, n_ticks=16, n_machines=8)
        self.assertNotEqual(left["records"], right["records"])
        # Types and the tick-0 prior do not depend on the seed.
        self.assertEqual(
            [row["machine_type"] for row in left["records"] if row["tick"] == 0],
            [row["machine_type"] for row in right["records"] if row["tick"] == 0],
        )
        self.assertEqual(
            [row["probabilities"] for row in left["truth"] if row["tick"] == 0],
            [row["probabilities"] for row in right["truth"] if row["tick"] == 0],
        )

    def test_default_settings_stay_under_the_record_cap(self):
        data = generate.generate()
        self.assertEqual(len(data["records"]), 8 * 96 * 8)
        self.assertLessEqual(len(data["records"]), generate.MAX_RECORDS)
        with self.assertRaises(ValueError):
            generate.generate(seed=0, n_episodes=8, n_ticks=200, n_machines=8)


class CausalPrefixTests(unittest.TestCase):
    def test_future_mutation_keeps_prefix_and_moves_later_oracle(self):
        kwargs = {"seed": 7, "n_episodes": 1, "n_ticks": 24, "n_machines": 8}
        split = 10

        def rewrite(episode_id, tick, machine_id, payout, observed):
            if tick >= split:
                return (0 if payout != 0 else 20), observed
            return payout, observed

        base = generate.generate(**kwargs)
        mutated = generate.generate(**kwargs, rewrite=rewrite)

        def take(rows, pred):
            return [row for row in rows if pred(row)]

        self.assertEqual(
            take(base["records"], lambda row: row["tick"] < split),
            take(mutated["records"], lambda row: row["tick"] < split),
        )
        self.assertEqual(
            take(base["truth"], lambda row: row["tick"] <= split),
            take(mutated["truth"], lambda row: row["tick"] <= split),
        )
        self.assertEqual(
            take(base["observable_history"], lambda row: row["tick"] <= split),
            take(mutated["observable_history"], lambda row: row["tick"] <= split),
        )
        self.assertNotEqual(
            take(base["records"], lambda row: row["tick"] >= split),
            take(mutated["records"], lambda row: row["tick"] >= split),
        )
        later_base = take(base["truth"], lambda row: row["tick"] > split)
        later_mut = take(mutated["truth"], lambda row: row["tick"] > split)
        self.assertNotEqual(later_base, later_mut)
        moved = max(
            _max_abs(left["probabilities"], right["probabilities"])
            for left, right in zip(later_base, later_mut)
        )
        self.assertGreater(moved, 0.05)

    def test_same_tick_other_machine_does_not_enter_the_state(self):
        """Mutating machine 0 at tick t changes machine 0's future only."""
        kwargs = {"seed": 5, "n_episodes": 1, "n_ticks": 18, "n_machines": 8}
        base = generate.generate(**kwargs)
        target_tick = 6

        def rewrite(episode_id, tick, machine_id, payout, observed):
            if tick == target_tick and machine_id == 0:
                return (20 if payout != 20 else 0), observed
            return payout, observed

        mutated = generate.generate(**kwargs, rewrite=rewrite)
        base_truth = _index_records(base["truth"])
        mut_truth = _index_records(mutated["truth"])
        for tick in range(18):
            for machine_id in range(1, 8):
                self.assertEqual(
                    base_truth[(0, tick, machine_id)]["probabilities"],
                    mut_truth[(0, tick, machine_id)]["probabilities"],
                )
        # Tick t itself was already determined by ticks < t.
        self.assertEqual(
            base_truth[(0, target_tick, 0)]["probabilities"],
            mut_truth[(0, target_tick, 0)]["probabilities"],
        )
        self.assertGreater(
            _max_abs(
                base_truth[(0, target_tick + 1, 0)]["probabilities"],
                mut_truth[(0, target_tick + 1, 0)]["probabilities"],
            ),
            0.05,
        )
        for machine_id in range(1, 8):
            self.assertEqual(
                [row for row in base["records"] if row["machine_id"] == machine_id],
                [row for row in mutated["records"] if row["machine_id"] == machine_id],
            )


class GappyMissingTests(unittest.TestCase):
    def test_generated_masks_are_sparse_and_run_in_gaps(self):
        data = generate.generate(seed=0, n_episodes=2, n_ticks=40, n_machines=8)
        flags = [row["observed"] for row in data["records"]]
        missing_rate = 1.0 - (sum(flags) / len(flags))
        self.assertGreater(missing_rate, 0.55)
        self.assertLess(missing_rate, 0.97)
        hidden_nonzero = [
            row for row in data["records"] if (not row["observed"]) and row["payout"] != 0
        ]
        self.assertTrue(hidden_nonzero)
        longest = 0
        for machine_id in range(8):
            for episode_id in range(2):
                run = 0
                for tick in range(40):
                    row = data["records"][(episode_id * 40 + tick) * 8 + machine_id]
                    self.assertEqual(row["episode_id"], episode_id)
                    self.assertEqual(row["tick"], tick)
                    self.assertEqual(row["machine_id"], machine_id)
                    if row["observed"]:
                        run = 0
                    else:
                        run += 1
                        longest = max(longest, run)
        self.assertGreaterEqual(longest, 3)

    def test_missing_is_not_imputed_as_zero(self):
        # Constructed, not a fluke of one seed: a hidden non-zero payout.
        hidden_payout = 20
        machine_type = "steady"
        predictive = list(generate.PRIOR[machine_type])
        posterior = generate.posterior_after(predictive, hidden_payout, observed=False)
        self.assertEqual(posterior, predictive)
        self.assertNotEqual(posterior, [1.0, 0.0, 0.0, 0.0, 0.0])

        oracle = generate.oracle_probabilities(machine_type, hidden_payout)
        observer = generate.observer_predictive(machine_type, posterior)
        zero_impute = generate.oracle_probabilities(machine_type, 0)
        # No history yet, so imputing the gap as 0 is a different distribution
        # from both the latent row and the prior mixture. Margin is material.
        self.assertGreater(_max_abs(oracle, zero_impute), 0.05)
        self.assertGreater(_max_abs(observer, zero_impute), 0.05)
        self.assertGreater(_max_abs(oracle, observer), 0.05)

        # Inside a generated episode the oracle still follows the hidden payout,
        # not a filled-in 0. The filter is a mixture, so it is not that row either
        # on a witnessed non-zero gap (it may sit near the zero row only after the
        # belief has already concentrated, which is a different claim).
        data = generate.generate(seed=4, n_episodes=1, n_ticks=12, n_machines=8)
        records = _index_records(data["records"])
        truth = _index_records(data["truth"])
        observer_rows = _index_records(data["observable_history"])
        oracle_gaps = []
        filter_gaps = []
        for tick in range(1, 12):
            for machine_id in range(8):
                prev = records[(0, tick - 1, machine_id)]
                if prev["observed"] or prev["payout"] == 0:
                    continue
                machine_type = prev["machine_type"]
                zero_row = generate.oracle_probabilities(machine_type, 0)
                oracle_gaps.append(
                    _max_abs(truth[(0, tick, machine_id)]["probabilities"], zero_row)
                )
                filter_gaps.append(
                    _max_abs(observer_rows[(0, tick, machine_id)]["probabilities"], zero_row)
                )
                self.assertIsNone(generate.observable_payout(prev))
                self.assertEqual(generate.observation_symbol(prev), generate.MISSING_INDEX)
                self.assertNotEqual(
                    truth[(0, tick, machine_id)]["probabilities"],
                    zero_row,
                )
        self.assertTrue(oracle_gaps)
        self.assertGreater(min(oracle_gaps), 0.05)
        self.assertGreater(max(filter_gaps), 0.05)

    def test_hidden_latent_mutation_moves_oracle_not_the_filter(self):
        kwargs = {"seed": 3, "n_episodes": 1, "n_ticks": 20, "n_machines": 8}
        base = generate.generate(**kwargs)
        target = next(
            row
            for row in base["records"]
            if (not row["observed"]) and row["payout"] != 0 and row["tick"] < 19
        )

        def rewrite(episode_id, tick, machine_id, payout, observed):
            if (
                episode_id == target["episode_id"]
                and tick == target["tick"]
                and machine_id == target["machine_id"]
            ):
                return 0, False
            return payout, observed

        mutated = generate.generate(**kwargs, rewrite=rewrite)
        self.assertEqual(
            [row["observed"] for row in base["records"]],
            [row["observed"] for row in mutated["records"]],
        )
        base_truth = _index_records(base["truth"])
        mut_truth = _index_records(mutated["truth"])
        base_obs = _index_records(base["observable_history"])
        mut_obs = _index_records(mutated["observable_history"])
        key_now = (target["episode_id"], target["tick"], target["machine_id"])
        key_next = (target["episode_id"], target["tick"] + 1, target["machine_id"])
        self.assertEqual(base_truth[key_now]["probabilities"], mut_truth[key_now]["probabilities"])
        self.assertGreater(
            _max_abs(base_truth[key_next]["probabilities"], mut_truth[key_next]["probabilities"]),
            0.05,
        )
        # The filter at t+1 was already fixed by revealed history through t.
        # The hidden payout is not an input, so the observer row does not move.
        self.assertEqual(base_obs[key_next]["probabilities"], mut_obs[key_next]["probabilities"])
        for row in base["observable_history"]:
            if row["tick"] <= target["tick"] + 1:
                key = (row["episode_id"], row["tick"], row["machine_id"])
                self.assertEqual(row["probabilities"], mut_obs[key]["probabilities"])


class KnownTruthTests(unittest.TestCase):
    def test_oracle_matches_latent_previous_and_splits_from_the_filter(self):
        # Every disclosed row is a different distribution, so the invariant
        # is not a constant categorical in disguise.
        for machine_type, rows in generate.TRANSITION.items():
            for left in range(5):
                for right in range(left + 1, 5):
                    self.assertGreater(_max_abs(list(rows[left]), list(rows[right])), 0.05)

        data = generate.generate(seed=2, n_episodes=2, n_ticks=15, n_machines=8)
        records = _index_records(data["records"])
        saw_agreement = False
        saw_split = False
        for row in data["truth"]:
            key = (row["episode_id"], row["tick"], row["machine_id"])
            record = records[key]
            if row["tick"] == 0:
                previous = None
            else:
                previous = records[(row["episode_id"], row["tick"] - 1, row["machine_id"])]["payout"]
            expected = generate.oracle_probabilities(record["machine_type"], previous)
            self.assertEqual(row["probabilities"], expected)

        # Independent filter walk. This is the observer, not the oracle.
        replay_obs: dict[tuple, list[float]] = {}
        posterior = {}
        for record in data["records"]:
            key = (record["episode_id"], record["machine_id"])
            belief = posterior.get(key)
            predictive = generate.observer_predictive(record["machine_type"], belief)
            replay_obs[(record["episode_id"], record["tick"], record["machine_id"])] = predictive
            posterior[key] = generate.posterior_after(
                predictive, record["payout"], record["observed"]
            )
        for row in data["observable_history"]:
            key = (row["episode_id"], row["tick"], row["machine_id"])
            self.assertEqual(row["probabilities"], replay_obs[key])

        truth = _index_records(data["truth"])
        for row in data["observable_history"]:
            key = (row["episode_id"], row["tick"], row["machine_id"])
            oracle = truth[key]["probabilities"]
            gap = _max_abs(row["probabilities"], oracle)
            if row["tick"] == 0 or records[(row["episode_id"], row["tick"] - 1, row["machine_id"])]["observed"]:
                self.assertEqual(row["probabilities"], oracle)
                saw_agreement = True
            else:
                # Full-support rows make the mixture unequal to every latent row.
                self.assertGreater(gap, 1e-9)
                if gap > 0.05:
                    saw_split = True
        self.assertTrue(saw_agreement)
        self.assertTrue(saw_split)

        for query in data["queries"]:
            last = records[(query["episode_id"], query["cutoff"] - 1, query["machine_id"])]
            self.assertEqual(
                query["latent_oracle_probabilities"],
                generate.oracle_probabilities(query["machine_type"], last["payout"]),
            )
            self.assertEqual(query["tick"] if "tick" in query else query["cutoff"], query["cutoff"])
            cutoff_gap = _max_abs(
                query["latent_oracle_probabilities"],
                query["observable_history_probabilities"],
            )
            if last["observed"]:
                self.assertEqual(cutoff_gap, 0.0)
            else:
                self.assertGreater(cutoff_gap, 1e-9)

    def test_tick_zero_oracle_equals_observer_prior(self):
        data = generate.generate(seed=11, n_episodes=1, n_ticks=2, n_machines=8)
        for truth, observed, record in zip(
            data["truth"], data["observable_history"], data["records"]
        ):
            if truth["tick"] != 0:
                continue
            prior = list(generate.PRIOR[record["machine_type"]])
            self.assertEqual(truth["probabilities"], prior)
            self.assertEqual(observed["probabilities"], prior)


class CliTests(unittest.TestCase):
    def test_cli_writes_hashed_research_only_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bundle"
            proc = subprocess.run(
                [
                    PYTHON,
                    str(ROOT / "generate.py"),
                    "--seed",
                    "0",
                    "--episodes",
                    "1",
                    "--ticks",
                    "8",
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            self.assertEqual(summary["eligibility"], "RESEARCH_ONLY")
            self.assertEqual(summary["n_records"], 64)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["eligibility"], "RESEARCH_ONLY")
            self.assertTrue(manifest["real_only_fallback"])
            self.assertTrue(manifest["do_not_train_by_default"])
            self.assertEqual(manifest["seed"], 0)
            self.assertEqual(manifest["settings"]["n_episodes"], 1)
            self.assertEqual(manifest["settings"]["n_ticks"], 8)
            self.assertEqual(manifest["family"], "sparse_gappy")
            for name, digest in manifest["output_hashes"].items():
                self.assertEqual(digest, generate.sha256_file(out / name))
                self.assertEqual(digest, summary["output_hashes"][name])
            self.assertEqual(
                manifest["source_hashes"]["generate.py"],
                generate.sha256_file(ROOT / "generate.py"),
            )
            records = [
                json.loads(line)
                for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            truth = [
                json.loads(line)
                for line in (out / "truth.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 64)
            self.assertEqual(len(truth), 64)
            self.assertTrue(all(len(row["probabilities"]) == 5 for row in truth))
            queries = [
                json.loads(line)
                for line in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(query["signal"] is None for query in queries))
            self.assertTrue(all(query["cutoff"] == 8 for query in queries))
            self.assertTrue(all("signal" not in row for row in records))


if __name__ == "__main__":
    unittest.main()
