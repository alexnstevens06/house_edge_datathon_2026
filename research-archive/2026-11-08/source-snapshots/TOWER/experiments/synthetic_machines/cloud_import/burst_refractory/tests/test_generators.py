"""Tests for the internal synthetic burst and refractory generators.

These checks define the research generator contract. They do not describe an
organizer schema and they do not use any datathon validation or test files.
"""

from __future__ import annotations

import ast
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from burst_refractory.cli import main
from burst_refractory.generators import (
    N_MACHINES,
    PAYOUTS,
    ParameterError,
    burst_conditional_probabilities,
    burst_machine_type,
    burst_next_excitation,
    generate_burst,
    generate_refractory,
    refractory_conditional_probabilities,
    refractory_machine_spec,
    refractory_next_cooldown,
)


def _config(seed: int, episodes: int, ticks: int, observe: float) -> dict:
    return {
        "seed": seed,
        "n_episodes": episodes,
        "n_ticks": ticks,
        "observe_probability": observe,
    }


class ProbabilityContractTest(unittest.TestCase):
    def test_payout_support_and_machine_count(self) -> None:
        self.assertEqual(PAYOUTS, (0, 1, 2, 5, 20))
        self.assertEqual(N_MACHINES, 8)

    def test_burst_probabilities_nonnegative_finite_and_normalized(self) -> None:
        excitations = (0.0, 0.2, 1.0, 4.0, 20.0)
        for machine_id in range(N_MACHINES):
            for excitation in excitations:
                probs = burst_conditional_probabilities(machine_id, excitation)
                self.assertEqual(len(probs), len(PAYOUTS))
                self.assertTrue(all(math.isfinite(p) and p >= 0.0 for p in probs))
                self.assertLessEqual(abs(sum(probs) - 1.0), 1e-12)

    def test_refractory_probabilities_nonnegative_finite_and_normalized(self) -> None:
        for machine_id in range(N_MACHINES):
            for cooldown in (0, 1, 3, 8):
                probs = refractory_conditional_probabilities(machine_id, cooldown)
                self.assertEqual(len(probs), len(PAYOUTS))
                self.assertTrue(all(math.isfinite(p) and p >= 0.0 for p in probs))
                self.assertLessEqual(abs(sum(probs) - 1.0), 1e-12)

    def test_refractory_machine_types_differ(self) -> None:
        specs = [refractory_machine_spec(machine_id) for machine_id in range(N_MACHINES)]
        names = {spec.name for spec in specs}
        lengths = {spec.refractory_length for spec in specs}
        ready = {spec.ready_weights for spec in specs}
        self.assertGreaterEqual(len(names), 3)
        self.assertGreaterEqual(len(lengths), 3)
        self.assertGreaterEqual(len(ready), 3)
        self.assertEqual([spec.machine_id for spec in specs], list(range(N_MACHINES)))


class KnownTruthLagTest(unittest.TestCase):
    def test_burst_positive_payout_raises_next_tick_high_payout_probability(self) -> None:
        for machine_id in range(N_MACHINES):
            baseline = burst_conditional_probabilities(machine_id, 0.0)
            after_zero = burst_next_excitation(machine_id, 0.0, 0)
            after_high = burst_next_excitation(machine_id, 0.0, 20)
            self.assertEqual(after_zero, 0.0)
            self.assertGreater(after_high, 0.0)
            probs_zero = burst_conditional_probabilities(machine_id, after_zero)
            probs_high = burst_conditional_probabilities(machine_id, after_high)
            # Index 4 is payout 20. The lag is one tick and uses only past payout.
            self.assertGreater(probs_high[4], probs_zero[4])
            self.assertGreater(probs_high[4], baseline[4])
            self.assertEqual(probs_zero, baseline)

    def test_burst_machine_type_changes_lag_magnitude(self) -> None:
        spark = burst_next_excitation(0, 0.0, 1)
        cascade = burst_next_excitation(4, 0.0, 1)
        self.assertGreater(cascade, spark)
        self.assertEqual(burst_machine_type(0), "spark")
        self.assertEqual(burst_machine_type(4), "cascade")
        spark_p = burst_conditional_probabilities(0, spark)
        cascade_p = burst_conditional_probabilities(4, cascade)
        self.assertGreater(cascade_p[4], spark_p[4])

    def test_refractory_positive_payout_suppresses_the_next_tick(self) -> None:
        for machine_id in range(N_MACHINES):
            spec = refractory_machine_spec(machine_id)
            ready = refractory_conditional_probabilities(machine_id, 0)
            next_after_zero = refractory_next_cooldown(machine_id, 0, 0)
            next_after_win = refractory_next_cooldown(machine_id, 0, 5)
            self.assertEqual(next_after_zero, 0)
            self.assertEqual(next_after_win, spec.refractory_length)
            self.assertGreaterEqual(spec.refractory_length, 2)
            cooled = refractory_conditional_probabilities(machine_id, next_after_win)
            self.assertGreater(cooled[0], ready[0])
            self.assertLess(cooled[4], ready[4])

    def test_generated_truth_matches_causal_transition(self) -> None:
        burst_events, burst_truth = generate_burst(_config(11, 2, 6, 0.7))
        self._assert_burst_causal(burst_truth)
        self._assert_events_match_truth(burst_events, burst_truth)

        ref_events, ref_truth = generate_refractory(_config(11, 2, 6, 0.7))
        self._assert_refractory_causal(ref_truth)
        self._assert_events_match_truth(ref_events, ref_truth)

    def _assert_burst_causal(self, truth: dict) -> None:
        self.assertEqual(truth["process"], "burst_self_exciting")
        for episode in truth["episodes"]:
            by_machine: dict[int, list[dict]] = {m: [] for m in range(N_MACHINES)}
            for record in episode["records"]:
                by_machine[record["machine_id"]].append(record)
            for machine_id, rows in by_machine.items():
                self.assertEqual(rows[0]["latent_state"]["excitation"], 0.0)
                for current, nxt in zip(rows, rows[1:]):
                    expected_probs = burst_conditional_probabilities(
                        machine_id, current["latent_state"]["excitation"]
                    )
                    self.assertEqual(current["conditional_payout_probabilities"], list(expected_probs))
                    expected_next = burst_next_excitation(
                        machine_id,
                        current["latent_state"]["excitation"],
                        current["true_payout"],
                    )
                    self.assertTrue(math.isclose(nxt["latent_state"]["excitation"], expected_next))

    def _assert_refractory_causal(self, truth: dict) -> None:
        self.assertEqual(truth["process"], "refractory_renewal")
        for episode in truth["episodes"]:
            by_machine: dict[int, list[dict]] = {m: [] for m in range(N_MACHINES)}
            for record in episode["records"]:
                by_machine[record["machine_id"]].append(record)
            for machine_id, rows in by_machine.items():
                self.assertEqual(rows[0]["latent_state"]["cooldown_remaining"], 0)
                for current, nxt in zip(rows, rows[1:]):
                    expected_probs = refractory_conditional_probabilities(
                        machine_id, current["latent_state"]["cooldown_remaining"]
                    )
                    self.assertEqual(current["conditional_payout_probabilities"], list(expected_probs))
                    expected_next = refractory_next_cooldown(
                        machine_id,
                        current["latent_state"]["cooldown_remaining"],
                        current["true_payout"],
                    )
                    self.assertEqual(nxt["latent_state"]["cooldown_remaining"], expected_next)

    def _assert_events_match_truth(self, events: list[dict], truth: dict) -> None:
        flat = [record for episode in truth["episodes"] for record in episode["records"]]
        self.assertEqual(len(events), len(flat))
        for event, record in zip(events, flat):
            self.assertEqual(event["episode_id"], record["episode_id"])
            self.assertEqual(event["tick"], record["tick"])
            self.assertEqual(event["machine_id"], record["machine_id"])
            self.assertEqual(event["observed"], record["observed"])
            if event["observed"]:
                self.assertEqual(event["payout"], record["true_payout"])
            else:
                self.assertIsNone(event["payout"])


class MissingnessVersusZeroTest(unittest.TestCase):
    def test_unobserved_payouts_stay_missing_and_observed_zeros_stay_zero(self) -> None:
        for generator in (generate_burst, generate_refractory):
            events, truth = generator(_config(7, 4, 16, 0.55))
            self.assertTrue(any(event["observed"] and event["payout"] == 0 for event in events))
            hidden = [event for event in events if not event["observed"]]
            self.assertGreaterEqual(len(hidden), 20)
            self.assertTrue(all(event["payout"] is None for event in hidden))
            hidden_truth = [
                record
                for episode in truth["episodes"]
                for record in episode["records"]
                if not record["observed"]
            ]
            self.assertTrue(any(record["true_payout"] == 0 for record in hidden_truth))
            self.assertTrue(all(record["true_payout"] in PAYOUTS for record in hidden_truth))


class ReproducibilityAndRngTest(unittest.TestCase):
    def test_same_seed_is_byte_identical_and_other_seed_differs(self) -> None:
        for generator in (generate_burst, generate_refractory):
            first_events, first_truth = generator(_config(123, 2, 8, 0.6))
            second_events, second_truth = generator(_config(123, 2, 8, 0.6))
            other_events, other_truth = generator(_config(124, 2, 8, 0.6))
            self.assertEqual(first_events, second_events)
            self.assertEqual(first_truth, second_truth)
            self.assertNotEqual(first_events, other_events)
            self.assertNotEqual(first_truth, other_truth)

    def test_generation_does_not_touch_global_rng(self) -> None:
        random.seed(99991)
        before = random.getstate()
        generate_burst(_config(5, 2, 5, 0.4))
        generate_refractory(_config(5, 2, 5, 0.4))
        self.assertEqual(random.getstate(), before)
        # A draw from the global generator still follows the pre-call seed.
        self.assertEqual(random.random(), random.Random(99991).random())


class BoundedOutputTest(unittest.TestCase):
    def test_api_rejects_out_of_range_parameters(self) -> None:
        invalid = [
            _config(-1, 1, 1, 0.5),
            _config(2**31, 1, 1, 0.5),
            _config(1, 0, 1, 0.5),
            _config(1, 49, 1, 0.5),
            _config(1, 1, 0, 0.5),
            _config(1, 1, 65, 0.5),
            _config(1, 1, 1, 0.0),
            _config(1, 1, 1, 1.1),
            _config(1, 1, 1, float("nan")),
        ]
        for config in invalid:
            with self.assertRaises(ParameterError):
                generate_burst(config)
            with self.assertRaises(ParameterError):
                generate_refractory(config)

    def test_event_rows_are_bounded_complete_panels(self) -> None:
        events, truth = generate_burst(_config(3, 3, 5, 1.0))
        self.assertEqual(len(events), 3 * 5 * N_MACHINES)
        self.assertEqual(truth["n_machines"], N_MACHINES)
        self.assertEqual(truth["payout_values"], list(PAYOUTS))
        self.assertFalse(truth["fitted_to_real_data"])
        self.assertTrue(truth["research_only"])
        self.assertEqual(truth["schema_status"], "explicit_internal_intermediate_not_organizer_schema")
        for event in events:
            self.assertEqual(
                set(event),
                {"episode_id", "tick", "machine_id", "machine_type", "payout", "observed"},
            )
            self.assertNotIn("signal", event)
            self.assertNotIn("conditional_payout_probabilities", event)
            self.assertIn(event["machine_id"], range(N_MACHINES))
            self.assertIn(event["payout"], PAYOUTS)
            self.assertTrue(event["observed"])

    def test_cli_bounds_and_writes_only_requested_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tiny"
            code = main(
                [
                    "burst",
                    "--seed",
                    "4",
                    "--episodes",
                    "1",
                    "--ticks",
                    "2",
                    "--observe-probability",
                    "0.5",
                    "--out-dir",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            event_lines = (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(event_lines), N_MACHINES * 2)
            truth = json.loads((out / "truth.json").read_text(encoding="utf-8"))
            self.assertEqual(truth["process"], "burst_self_exciting")
            self.assertNotIn("signal", json.dumps(truth))

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "refractory",
                        "--seed",
                        "4",
                        "--episodes",
                        "49",
                        "--ticks",
                        "2",
                        "--out-dir",
                        str(Path(tmp) / "too-big"),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((Path(tmp) / "too-big").exists())

            with self.assertRaises(SystemExit) as escaped:
                main(
                    [
                        "burst",
                        "--seed",
                        "4",
                        "--episodes",
                        "1",
                        "--ticks",
                        "1",
                        "--out-dir",
                        "../outside",
                    ]
                )
            self.assertEqual(escaped.exception.code, 2)


class NontrivialProcessTest(unittest.TestCase):
    def test_processes_are_not_iid_constants(self) -> None:
        for generator, state_key in (
            (generate_burst, "excitation"),
            (generate_refractory, "cooldown_remaining"),
        ):
            _events, truth = generator(_config(21, 2, 12, 1.0))
            records = [record for episode in truth["episodes"] for record in episode["records"]]
            payouts = {record["true_payout"] for record in records}
            states = {record["latent_state"][state_key] for record in records}
            prob_rows = {tuple(record["conditional_payout_probabilities"]) for record in records}
            self.assertGreaterEqual(len(payouts), 2)
            self.assertGreaterEqual(len(states), 2)
            self.assertGreaterEqual(len(prob_rows), 2)

    def test_burst_high_payouts_raise_the_next_positive_rate(self) -> None:
        _events, truth = generate_burst(_config(2026, 20, 32, 1.0))
        lagged_hits = 0
        lagged = 0
        positives = 0
        total = 0
        for episode in truth["episodes"]:
            by_machine: dict[int, list[dict]] = {m: [] for m in range(N_MACHINES)}
            for record in episode["records"]:
                by_machine[record["machine_id"]].append(record)
            for rows in by_machine.values():
                for current, nxt in zip(rows, rows[1:]):
                    total += 1
                    if nxt["true_payout"] > 0:
                        positives += 1
                    if current["true_payout"] >= 5:
                        lagged += 1
                        if nxt["true_payout"] > 0:
                            lagged_hits += 1
        self.assertGreaterEqual(lagged, 80)
        self.assertGreaterEqual(total, 1000)
        lagged_rate = lagged_hits / lagged
        base_rate = positives / total
        # Analytical one-step P(payout > 0 | previous payout >= 5) is about 0.70
        # from a cold start. The floor stays well above an IID draw from the
        # cold baseline (0.18) without sitting on the sample estimate.
        self.assertGreaterEqual(lagged_rate, 0.45)
        self.assertGreaterEqual(lagged_rate, base_rate + 0.12)

    def test_refractory_cooldown_elevates_the_next_zero_rate(self) -> None:
        _events, truth = generate_refractory(_config(2026, 16, 40, 1.0))
        after_win_zeros = 0
        after_win = 0
        ready_zeros = 0
        ready = 0
        for episode in truth["episodes"]:
            by_machine: dict[int, list[dict]] = {m: [] for m in range(N_MACHINES)}
            for record in episode["records"]:
                by_machine[record["machine_id"]].append(record)
            for rows in by_machine.values():
                for current, nxt in zip(rows, rows[1:]):
                    if current["true_payout"] > 0:
                        after_win += 1
                        if nxt["true_payout"] == 0:
                            after_win_zeros += 1
                    if (
                        current["true_payout"] == 0
                        and current["latent_state"]["cooldown_remaining"] == 0
                        and nxt["latent_state"]["cooldown_remaining"] == 0
                    ):
                        ready += 1
                        if nxt["true_payout"] == 0:
                            ready_zeros += 1
        self.assertGreaterEqual(after_win, 80)
        self.assertGreaterEqual(ready, 80)
        # Cooldown categorical mass on 0 is 0.97. Ready mass on 0 is at most 0.80.
        self.assertGreaterEqual(after_win_zeros / after_win, 0.90)
        self.assertLessEqual(ready_zeros / ready, 0.85)


class SourceBoundaryTest(unittest.TestCase):
    def test_generator_sources_do_not_read_data_files(self) -> None:
        package = Path(__file__).resolve().parents[1]
        for name in ("generators.py", "cli.py", "__init__.py", "__main__.py"):
            source = (package / name).read_text(encoding="utf-8")
            self.assertNotIn("signal", source)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                    mode = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        mode = node.args[1].value
                    for keyword in node.keywords:
                        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                            mode = keyword.value.value
                    self.assertIsNotNone(mode)
                    assert mode is not None
                    self.assertNotIn("r", mode)
                    self.assertTrue(mode.startswith("w") or mode.startswith("x"))


if __name__ == "__main__":
    unittest.main()
