"""Behavioral tests for the rare_heavytail generator.

These tests use the generator's public functions and a handwritten weight table.
They do not read contest train, validation, or test files.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate

# Independent expected weights. Heat effects are recomputed here, not imported.
REFERENCE_TABLES = {
    "steady": (
        (740, 180, 60, 15, 5),
        (700, 180, 70, 30, 20),
        (640, 180, 80, 50, 50),
        (550, 160, 90, 80, 120),
    ),
    "balanced": (
        (600, 220, 120, 40, 20),
        (550, 200, 120, 70, 60),
        (480, 180, 120, 100, 120),
        (400, 150, 120, 130, 200),
    ),
    "jackpot": (
        (900, 50, 30, 12, 8),
        (840, 50, 40, 30, 40),
        (720, 50, 50, 60, 120),
        (550, 50, 50, 100, 250),
    ),
}
PAYOUT_ORDER = (0, 1, 2, 5, 20)


def reference_heat(past: list[int | None]) -> int:
    heat = 0
    for value in past[-8:]:
        if value is None:
            continue
        if value == 20:
            heat = min(3, heat + 2)
        elif value == 5:
            heat = min(3, heat + 1)
        elif value == 0:
            heat = max(0, heat - 1)
        elif value in (1, 2):
            continue
        else:
            raise AssertionError(value)
    return heat


def weights_of(probabilities: list[float]) -> tuple[int, ...]:
    return tuple(int(round(probability * 1000)) for probability in probabilities)


class TestSimplexAndMapping(unittest.TestCase):
    def test_tables_and_symbols_match_the_published_order(self):
        self.assertEqual(generate.PAYOUTS, PAYOUT_ORDER)
        self.assertEqual([generate.INDEX_TO_PAYOUT[index] for index in range(5)], list(PAYOUT_ORDER))
        self.assertEqual(generate.dense_symbol(0, True), 1)
        self.assertEqual(generate.dense_symbol(20, True), 5)
        self.assertEqual(generate.dense_symbol(0, False), 0)
        self.assertEqual(generate.dense_symbol(20, False), 0)
        self.assertEqual(generate.MAPPINGS["dense_symbol"]["missing_unknown"], 0)
        self.assertEqual(generate.MAPPINGS["dense_symbol"]["payout_0"], 1)
        self.assertEqual(generate.MAPPINGS["mask"]["missing_unknown"], 0)

    def test_generated_rows_are_a_simplex_on_the_payout_map(self):
        bundle = generate.generate(seed=0, n_episodes=2, n_ticks=12, n_machines=8)
        self.assertEqual(len(bundle["records"]), len(bundle["truth"]))
        seen_types = set()
        for record, truth in zip(bundle["records"], bundle["truth"]):
            self.assertEqual(record["episode_id"], truth["episode_id"])
            self.assertEqual(record["tick"], truth["tick"])
            self.assertEqual(record["machine_id"], truth["machine_id"])
            self.assertEqual(record["machine_id"], record["slot_index"])
            self.assertIn(record["machine_id"], range(8))
            self.assertIn(record["machine_type"], ("steady", "balanced", "jackpot"))
            self.assertIn(record["payout"], PAYOUT_ORDER)
            self.assertEqual(record["payout"], PAYOUT_ORDER[record["payout_index"]])
            self.assertEqual(record["mask"], 1 if record["observed"] else 0)
            self.assertEqual(record["dense_symbol"], generate.dense_symbol(record["payout"], record["observed"]))
            self.assertEqual(truth["probability_kind"], "latent_oracle")
            self.assertEqual(tuple(truth["payout_order"]), PAYOUT_ORDER)
            self.assertEqual(len(truth["probabilities"]), 5)
            self.assertEqual(sum(truth["weights"]), 1000)
            self.assertTrue(all(0.0 <= probability <= 1.0 for probability in truth["probabilities"]))
            self.assertLess(abs(math.fsum(truth["probabilities"]) - 1.0), 1e-12)
            self.assertEqual(weights_of(truth["probabilities"]), tuple(truth["weights"]))
            seen_types.add(record["machine_type"])
            self.assertNotIn("signal", record)
            self.assertNotIn("signal", truth)
        self.assertEqual(seen_types, {"steady", "balanced", "jackpot"})

    def test_default_shape_stays_inside_the_record_cap(self):
        bundle = generate.generate()
        self.assertEqual(bundle["settings"]["n_records"], 8 * 96 * 8)
        self.assertLessEqual(len(bundle["records"]), generate.MAX_RECORDS)
        self.assertEqual(len(bundle["queries"]), 8 * 8)
        self.assertTrue(bundle["research_only"])
        self.assertTrue(bundle["real_only_fallback"])
        self.assertEqual(bundle["status"], "RESEARCH_ONLY")

    def test_over_cap_and_bad_shape_are_refused(self):
        with self.assertRaises(ValueError):
            generate.generate(seed=0, n_episodes=1, n_ticks=8193, n_machines=1)
        with self.assertRaises(ValueError):
            generate.generate(seed=0, n_episodes=11, n_ticks=96, n_machines=8)
        with self.assertRaises(ValueError):
            generate.generate(seed=0, n_episodes=1, n_ticks=4, n_machines=9)
        with self.assertRaises(ValueError):
            generate.generate(seed=0, n_episodes=0, n_ticks=4, n_machines=2)
        allowed = generate.check_shape(8, 128, 8)
        self.assertEqual(allowed, 8192)


class TestReproducibilityAndSeed(unittest.TestCase):
    def test_same_seed_matches_and_other_seed_changes_payouts(self):
        first = generate.generate(seed=0, n_episodes=2, n_ticks=16, n_machines=8)
        second = generate.generate(seed=0, n_episodes=2, n_ticks=16, n_machines=8)
        other = generate.generate(seed=1, n_episodes=2, n_ticks=16, n_machines=8)
        self.assertEqual(first["records"], second["records"])
        self.assertEqual(first["truth"], second["truth"])
        self.assertEqual(first["queries"], second["queries"])
        self.assertEqual(
            [row["probabilities"] for row in first["truth"] if row["tick"] == 0],
            [row["probabilities"] for row in other["truth"] if row["tick"] == 0],
        )
        self.assertNotEqual(
            [record["payout"] for record in first["records"]],
            [record["payout"] for record in other["records"]],
        )
        self.assertNotEqual(
            [record["observed"] for record in first["records"]],
            [record["observed"] for record in other["records"]],
        )

    def test_replay_of_the_documented_stream_matches(self):
        seed, n_episodes, n_ticks, n_machines = 11, 2, 6, 8
        bundle = generate.generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
        types = generate.machine_types(n_machines)
        replayed = []
        for episode in range(n_episodes):
            rng = generate.episode_random(seed, episode)
            columns = [[] for _ in range(n_machines)]
            for tick in range(n_ticks):
                specs = generate.tick_oracle(types, columns, tick)
                for machine_id in range(n_machines):
                    payout_index = generate.sample_weight_index(rng, specs[machine_id]["weights"])
                    payout = generate.INDEX_TO_PAYOUT[payout_index]
                    if generate.is_structural_gap(tick):
                        observed = False
                    else:
                        observed = rng.randrange(1000) < generate.OBSERVE_PER_MILLE[types[machine_id]]
                    columns[machine_id].append(payout)
                    replayed.append((f"rh{episode:03d}", tick, machine_id, payout, observed))
        got = [
            (record["episode_id"], record["tick"], record["machine_id"], record["payout"], record["observed"])
            for record in bundle["records"]
        ]
        self.assertEqual(got, replayed)


class TestCausalPrefixAndFutureMutation(unittest.TestCase):
    def test_shorter_run_is_the_prefix_of_a_longer_run(self):
        long_run = generate.generate(seed=4, n_episodes=2, n_ticks=11, n_machines=8)
        short_run = generate.generate(seed=4, n_episodes=2, n_ticks=5, n_machines=8)
        self.assertEqual(short_run["records"], [row for row in long_run["records"] if row["tick"] < 5])
        self.assertEqual(short_run["truth"], [row for row in long_run["truth"] if row["tick"] < 5])
        self.assertTrue(all(query["cutoff"] == 5 for query in short_run["queries"]))
        self.assertTrue(all(query["cutoff"] == 11 for query in long_run["queries"]))

    def test_future_mutation_keeps_the_prefix_and_a_past_mutation_moves_later_heat(self):
        series = [0, 1, 2, 0, 5, 0, 1, 20, 0]
        base = [generate.oracle_at("balanced", series, tick) for tick in range(len(series) + 1)]
        mutated = series.copy()
        mutated[6] = 20
        mutated[7] = 0
        mutated[8] = 20
        for tick in range(7):
            self.assertEqual(generate.oracle_at("balanced", mutated, tick)["probabilities"], base[tick]["probabilities"])
            self.assertEqual(generate.oracle_at("balanced", mutated, tick)["heat"], reference_heat(series[:tick]))
        self.assertEqual(reference_heat(series[:7]), 0)
        self.assertEqual(reference_heat(mutated[:7]), 2)
        self.assertNotEqual(
            generate.oracle_at("balanced", mutated, 7)["probabilities"],
            base[7]["probabilities"],
        )
        self.assertGreater(
            generate.oracle_at("balanced", mutated, 7)["probabilities"][4],
            base[7]["probabilities"][4],
        )

    def test_contemporaneous_cross_machine_outcome_is_ignored(self):
        types = generate.machine_types(8)
        columns = [[0, 1, 2, 0, 5] for _ in range(8)]
        before = generate.tick_oracle(types, columns, 4)
        columns[3][4] = 20
        after_same_tick = generate.tick_oracle(types, columns, 4)
        self.assertEqual(before, after_same_tick)
        columns[3][3] = 20
        after_lag = generate.tick_oracle(types, columns, 4)
        self.assertEqual(after_lag[0], before[0])
        self.assertEqual(after_lag[7], before[7])
        self.assertNotEqual(after_lag[3]["heat"], before[3]["heat"])
        self.assertEqual(after_lag[3]["heat"], reference_heat([0, 1, 2, 20]))

    def test_generated_truth_is_invariant_to_future_latent_overwrites(self):
        bundle = generate.generate(seed=7, n_episodes=1, n_ticks=12, n_machines=8)
        types, columns = _columns(bundle, "rh000")
        mutated = [column.copy() for column in columns]
        for column in mutated:
            for tick in range(6, len(column)):
                column[tick] = 20
        for tick in range(7):
            rescored = generate.tick_oracle(types, mutated, tick)
            for machine_id in range(8):
                truth = _truth_at(bundle, "rh000", tick, machine_id)
                self.assertEqual(truth["probabilities"], rescored[machine_id]["probabilities"])
        changed = False
        for tick in range(7, 12):
            original = generate.tick_oracle(types, columns, tick)
            rescored = generate.tick_oracle(types, mutated, tick)
            if original[0]["probabilities"] != rescored[0]["probabilities"]:
                changed = True
        self.assertTrue(changed)

    def test_past_mutation_on_a_generated_series_changes_the_next_tick_only(self):
        bundle = generate.generate(seed=9, n_episodes=1, n_ticks=8, n_machines=4)
        types, columns = _columns(bundle, "rh000")
        mutated = [column.copy() for column in columns]
        mutated[0][0] = 0 if columns[0][0] == 20 else 20
        self.assertEqual(generate.tick_oracle(types, mutated, 0), generate.tick_oracle(types, columns, 0))
        self.assertNotEqual(
            generate.tick_oracle(types, mutated, 1)[0]["heat"],
            generate.tick_oracle(types, columns, 1)[0]["heat"],
        )
        self.assertEqual(
            generate.tick_oracle(types, mutated, 1)[1],
            generate.tick_oracle(types, columns, 1)[1],
        )


class TestGappyMissingVersusZero(unittest.TestCase):
    def test_skipping_unknown_is_different_from_writing_zero(self):
        hidden = [20, None]
        skipped = generate.observable_history_probabilities("jackpot", hidden)
        zero_filled = generate.latent_oracle_probabilities("jackpot", [0 if value is None else value for value in hidden])
        self.assertEqual(skipped["probability_kind"], "observable_history")
        self.assertEqual(zero_filled["probability_kind"], "latent_oracle")
        self.assertEqual(skipped["heat"], reference_heat(hidden))
        self.assertEqual(zero_filled["heat"], reference_heat([20, 0]))
        self.assertEqual(skipped["heat"], 2)
        self.assertEqual(zero_filled["heat"], 1)
        self.assertGreater(skipped["probabilities"][4], zero_filled["probabilities"][4])
        self.assertEqual(weights_of(skipped["probabilities"]), REFERENCE_TABLES["jackpot"][2])
        self.assertEqual(weights_of(zero_filled["probabilities"]), REFERENCE_TABLES["jackpot"][1])

    def test_hidden_latent_zero_still_cools_the_oracle(self):
        oracle = generate.latent_oracle_probabilities("jackpot", [20, 0])
        visible = generate.observable_history_probabilities("jackpot", [20, None])
        self.assertNotEqual(oracle["probabilities"], visible["probabilities"])
        self.assertEqual(oracle["heat"], 1)
        self.assertEqual(visible["heat"], 2)

    def test_hidden_jackpot_does_not_enter_the_observable_fold(self):
        oracle = generate.latent_oracle_probabilities("steady", [0, 20])
        visible = generate.observable_history_probabilities("steady", [0, None])
        self.assertEqual(oracle["heat"], 2)
        self.assertEqual(visible["heat"], 0)
        self.assertGreater(oracle["probabilities"][4], visible["probabilities"][4])

    def test_generated_gaps_keep_nonzero_latents_and_a_distinct_zero_code(self):
        bundle = generate.generate(seed=0, n_episodes=2, n_ticks=20, n_machines=8)
        missing = [record for record in bundle["records"] if not record["observed"]]
        observed_zeros = [record for record in bundle["records"] if record["observed"] and record["payout"] == 0]
        self.assertTrue(missing)
        self.assertTrue(any(record["payout"] != 0 for record in missing))
        self.assertTrue(any(record["payout"] == 0 for record in missing))
        self.assertTrue(observed_zeros)
        for record in missing:
            self.assertEqual(record["mask"], 0)
            self.assertEqual(record["dense_symbol"], 0)
            self.assertNotEqual(record["dense_symbol"], generate.dense_symbol(0, True))
        for record in observed_zeros:
            self.assertEqual(record["mask"], 1)
            self.assertEqual(record["dense_symbol"], 1)
        structural = [record for record in bundle["records"] if generate.is_structural_gap(record["tick"])]
        self.assertTrue(structural)
        self.assertTrue(all(record["observed"] is False for record in structural))

    def test_emitted_truth_follows_latent_spins_when_the_log_hides_them(self):
        bundle = generate.generate(seed=3, n_episodes=1, n_ticks=24, n_machines=8)
        types, latent_columns = _columns(bundle, "rh000")
        visible_columns = bundle["visible_columns"][0]
        disagreements = 0
        for machine_id in range(8):
            for tick in range(1, 24):
                oracle = generate.latent_oracle_probabilities(types[machine_id], latent_columns[machine_id][:tick])
                visible = generate.observable_history_probabilities(types[machine_id], visible_columns[machine_id][:tick])
                truth = _truth_at(bundle, "rh000", tick, machine_id)
                self.assertEqual(truth["probabilities"], oracle["probabilities"])
                self.assertEqual(truth["probability_kind"], "latent_oracle")
                if oracle["heat"] != visible["heat"]:
                    disagreements += 1
                    self.assertNotEqual(truth["probabilities"], visible["probabilities"])
        self.assertGreater(disagreements, 0)


class TestKnownTruthInvariant(unittest.TestCase):
    def test_handwritten_heat_cases(self):
        cases = (
            ([], 0),
            ([20], 2),
            ([5], 1),
            ([20, 0], 1),
            ([0, 20], 2),
            ([1, 2, 1], 0),
            ([5, 5, 5], 3),
            ([20] + [1] * 7, 2),
            ([20] + [1] * 8, 0),
            ([20, None], 2),
            ([20, 0, None, 5], 2),
        )
        for past, heat in cases:
            self.assertEqual(reference_heat(list(past)), heat, past)
            self.assertEqual(generate.fold_heat(list(past)), heat, past)

    def test_cold_start_and_family_tail_signature(self):
        self.assertEqual(REFERENCE_TABLES["jackpot"][0][4], 8)
        self.assertEqual(REFERENCE_TABLES["jackpot"][3][4], 250)
        self.assertGreater(REFERENCE_TABLES["jackpot"][3][4] / REFERENCE_TABLES["jackpot"][0][4], 20)
        cold = generate.latent_oracle_probabilities("jackpot", [])
        hot = generate.latent_oracle_probabilities("jackpot", [20, 5])
        self.assertEqual(cold["heat"], 0)
        self.assertEqual(hot["heat"], 3)
        self.assertEqual(weights_of(cold["probabilities"]), REFERENCE_TABLES["jackpot"][0])
        self.assertEqual(weights_of(hot["probabilities"]), REFERENCE_TABLES["jackpot"][3])
        steady = generate.latent_oracle_probabilities("steady", [])
        balanced = generate.latent_oracle_probabilities("balanced", [])
        self.assertEqual(weights_of(steady["probabilities"]), REFERENCE_TABLES["steady"][0])
        self.assertEqual(weights_of(balanced["probabilities"]), REFERENCE_TABLES["balanced"][0])
        self.assertLess(steady["probabilities"][4], balanced["probabilities"][4])

    def test_every_emitted_truth_row_matches_the_handwritten_table(self):
        bundle = generate.generate(seed=6, n_episodes=2, n_ticks=18, n_machines=8)
        for episode in range(2):
            episode_id = f"rh{episode:03d}"
            types, columns = _columns(bundle, episode_id)
            for tick in range(18):
                for machine_id in range(8):
                    heat = reference_heat(columns[machine_id][:tick])
                    expected = REFERENCE_TABLES[types[machine_id]][heat]
                    truth = _truth_at(bundle, episode_id, tick, machine_id)
                    self.assertEqual(truth["heat"], heat)
                    self.assertEqual(tuple(truth["weights"]), expected)
                    self.assertEqual(weights_of(truth["probabilities"]), expected)
                    self.assertEqual(truth["probability_kind"], "latent_oracle")
                    visible = [
                        payout if observed else None
                        for payout, observed in (
                            (record["payout"], record["observed"])
                            for record in bundle["records"]
                            if record["episode_id"] == episode_id
                            and record["machine_id"] == machine_id
                            and record["tick"] < tick
                        )
                    ]
                    self.assertEqual(
                        generate.observable_history_probabilities(types[machine_id], visible)["heat"],
                        reference_heat(visible),
                    )

    def test_queries_carry_null_signal_only_at_the_cutoff(self):
        bundle = generate.generate(seed=0, n_episodes=2, n_ticks=10, n_machines=8)
        self.assertEqual(len(bundle["queries"]), 16)
        for query in bundle["queries"]:
            self.assertIsNone(query["signal"])
            self.assertEqual(query["signal_policy"], "not_invented_real_cutoff_only")
            self.assertEqual(query["cutoff"], 10)
            self.assertEqual(
                query["probability_kind_for_scoring_if_this_were_the_simulator"],
                "latent_oracle",
            )
            episode_records = [row for row in bundle["records"] if row["episode_id"] == query["episode_id"]]
            self.assertTrue(episode_records)
            self.assertTrue(all(row["tick"] < query["cutoff"] for row in episode_records))
            latent = [
                row["payout"]
                for row in episode_records
                if row["machine_id"] == query["machine_id"]
            ]
            visible = [
                row["payout"] if row["observed"] else None
                for row in episode_records
                if row["machine_id"] == query["machine_id"]
            ]
            oracle = generate.latent_oracle_probabilities(query["machine_type"], latent)
            observed = generate.observable_history_probabilities(query["machine_type"], visible)
            self.assertEqual(query["latent_oracle_probabilities"], oracle["probabilities"])
            self.assertEqual(query["observable_history_probabilities"], observed["probabilities"])
        self.assertTrue(
            any(
                query["latent_oracle_heat"] != query["observable_history_heat"]
                for query in bundle["queries"]
            )
        )


class TestResearchOnlyAndCLI(unittest.TestCase):
    def test_policy_blocks_contest_training(self):
        policy = generate.training_source_policy()
        self.assertEqual(policy["eligibility"], "unresolved")
        self.assertTrue(policy["research_only"])
        self.assertTrue(policy["real_only_fallback"])
        self.assertFalse(policy["synthetic_rows_allowed_in_contest_training"])
        self.assertEqual(policy["contest_training_data"], "organizer_released_real_only")
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["family"], "rare_heavytail")
        self.assertEqual(manifest["status"], "RESEARCH_ONLY")
        self.assertTrue(manifest["research_only"])
        self.assertTrue(manifest["real_only_fallback"])
        self.assertFalse(manifest["use_for_contest_training"])
        self.assertEqual(manifest["source_hashes"], generate.source_hashes(ROOT))

    def test_cli_writes_hashed_research_only_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "generate.py"),
                    "--seed",
                    "0",
                    "--episodes",
                    "2",
                    "--ticks",
                    "8",
                    "--out",
                    tmp,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            out = Path(tmp)
            records = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
            truth = (out / "truth.jsonl").read_text(encoding="utf-8").splitlines()
            queries = (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 2 * 8 * 8)
            self.assertEqual(len(truth), len(records))
            self.assertEqual(len(queries), 2 * 8)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "RESEARCH_ONLY")
            self.assertEqual(manifest["seed"], 0)
            self.assertEqual(manifest["settings"]["n_episodes"], 2)
            self.assertEqual(manifest["settings"]["n_ticks"], 8)
            self.assertTrue(manifest["real_only_fallback"])
            self.assertFalse(manifest["use_for_contest_training"])
            self.assertEqual(manifest["output_hashes"]["records.jsonl"], generate.sha256_file(out / "records.jsonl"))
            self.assertEqual(manifest["output_hashes"]["truth.jsonl"], generate.sha256_file(out / "truth.jsonl"))
            self.assertEqual(manifest["output_hashes"]["queries.jsonl"], generate.sha256_file(out / "queries.jsonl"))
            self.assertEqual(manifest["source_hashes"]["generate.py"], generate.sha256_file(ROOT / "generate.py"))
            payload = json.loads(records[0])
            self.assertIn(payload["payout"], PAYOUT_ORDER)
            self.assertIn("observed", payload)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["status"], "RESEARCH_ONLY")
            self.assertEqual(summary["n_records"], 128)


def _columns(bundle: dict, episode_id: str) -> tuple[list[str], list[list[int]]]:
    records = [row for row in bundle["records"] if row["episode_id"] == episode_id]
    n_machines = max(row["machine_id"] for row in records) + 1
    types: list[str | None] = [None] * n_machines
    columns: list[list[int]] = [[] for _ in range(n_machines)]
    for row in records:
        types[row["machine_id"]] = row["machine_type"]
        columns[row["machine_id"]].append(row["payout"])
    return [str(machine_type) for machine_type in types], columns


def _truth_at(bundle: dict, episode_id: str, tick: int, machine_id: int) -> dict:
    for row in bundle["truth"]:
        if row["episode_id"] == episode_id and row["tick"] == tick and row["machine_id"] == machine_id:
            return row
    raise AssertionError((episode_id, tick, machine_id))


if __name__ == "__main__":
    unittest.main()
