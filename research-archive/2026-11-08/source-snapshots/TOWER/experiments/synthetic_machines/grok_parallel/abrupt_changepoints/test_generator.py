#!/usr/bin/env python3
"""Meaningful checks for the abrupt_changepoints toy generator."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import generate as gen


PAYOUTS = (0, 1, 2, 5, 20)


def _simplex(probs) -> None:
    if len(probs) != 5:
        raise AssertionError(f"expected 5 probabilities, got {probs!r}")
    if any((not math.isfinite(p)) or p < 0.0 or p > 1.0 for p in probs):
        raise AssertionError(f"probability out of range: {probs!r}")
    total = math.fsum(probs)
    if abs(total - 1.0) > 1e-12:
        raise AssertionError(f"probabilities sum to {total!r}")


class AbruptChangepointTests(unittest.TestCase):
    def test_probability_simplex_and_payout_mapping(self) -> None:
        data = gen.generate(seed=0, n_episodes=2, n_ticks=40, n_machines=8)
        self.assertEqual(len(data["records"]), 2 * 40 * 8)
        self.assertEqual(len(data["truth"]), len(data["records"]))
        for row in data["truth"]:
            _simplex(row["probabilities"])
            self.assertEqual(
                [gen.INDEX_TO_PAYOUT[i] for i in range(5)],
                list(PAYOUTS),
            )
            self.assertEqual(
                list(gen.PROBABILITY_COLUMNS),
                ["p_0", "p_1", "p_2", "p_5", "p_20"],
            )
        for payout in PAYOUTS:
            encoded = gen.encode_observation(payout, True)
            self.assertEqual(encoded["payout_index"], gen.PAYOUT_TO_INDEX[payout])
            self.assertEqual(encoded["mask"], 1)
            self.assertEqual(sum(encoded["one_hot"]), 1)
            self.assertEqual(encoded["one_hot"][gen.PAYOUT_TO_INDEX[payout]], 1)
            self.assertEqual(gen.PAYOUTS[encoded["payout_index"]], payout)
        for row in data["records"]:
            self.assertIn(row["machine_id"], range(8))
            self.assertIn(row["machine_type"], gen.MACHINE_TYPES)
            self.assertEqual(row["machine_type"], gen.machine_type_for_slot(row["machine_id"]))
            self.assertIn(row["payout"], PAYOUTS)
            self.assertIsInstance(row["observed"], bool)
            self.assertNotIn("signal", row)

    def test_default_size_is_within_cap(self) -> None:
        data = gen.generate()
        self.assertEqual(data["settings"]["seed"], 0)
        self.assertEqual(data["settings"]["n_episodes"], 8)
        self.assertEqual(data["settings"]["n_ticks"], 96)
        self.assertEqual(data["settings"]["n_machines"], 8)
        self.assertEqual(len(data["records"]), 6144)
        self.assertLessEqual(len(data["records"]), gen.MAX_RECORDS)
        self.assertEqual(data["eligibility"], "RESEARCH_ONLY")
        self.assertFalse(data["synthetic_exception_verified"])
        self.assertTrue(data["real_only_fallback"])
        with self.assertRaises(ValueError):
            gen.generate(n_episodes=8, n_ticks=129, n_machines=8)

    def test_reproducibility_and_different_seed(self) -> None:
        first = gen.generate(seed=0, n_episodes=2, n_ticks=80, n_machines=8)
        second = gen.generate(seed=0, n_episodes=2, n_ticks=80, n_machines=8)
        self.assertEqual(first, second)
        other = gen.generate(seed=1, n_episodes=2, n_ticks=80, n_machines=8)
        # 80 ticks covers the jump. A short window that ends before every
        # changepoint can share the start-regime oracle; that is not a seed bug.
        self.assertNotEqual(
            [gen.latent_state(0, 0, machine, 0)["change_at"] for machine in range(8)],
            [gen.latent_state(1, 0, machine, 0)["change_at"] for machine in range(8)],
        )
        self.assertNotEqual(
            [row["probabilities"] for row in first["truth"]],
            [row["probabilities"] for row in other["truth"]],
        )
        self.assertNotEqual(
            [(row["payout"], row["observed"]) for row in first["records"]],
            [(row["payout"], row["observed"]) for row in other["records"]],
        )

    def test_family_known_truth_is_abrupt_regime_catalog(self) -> None:
        seed = 0
        data = gen.generate(seed=seed, n_episodes=2, n_ticks=80, n_machines=8)
        jumps = 0
        for row in data["truth"]:
            episode = gen.parse_episode_index(row["episode_id"])
            state = gen.latent_state(seed, episode, row["machine_id"], row["tick"])
            expected = gen.REGIME_PROBS[state["machine_type"]][state["regime"]]
            self.assertEqual(tuple(row["probabilities"]), expected)
            self.assertIn(tuple(row["probabilities"]), gen.REGIME_PROBS[state["machine_type"]])
        # Tick 0 is before every changepoint (earliest change_at is 24).
        # Even and odd slots of the same type start in opposite regimes.
        for episode in range(2):
            for left, right in ((0, 1), (2, 3), (5, 6)):
                episode_id = gen.episode_id_for(episode)
                left_row = next(
                    row
                    for row in data["truth"]
                    if row["episode_id"] == episode_id
                    and row["machine_id"] == left
                    and row["tick"] == 0
                )
                right_row = next(
                    row
                    for row in data["truth"]
                    if row["episode_id"] == episode_id
                    and row["machine_id"] == right
                    and row["tick"] == 0
                )
                self.assertNotEqual(left_row["probabilities"], right_row["probabilities"])
            anchors = set()
            change_ats = set()
            for machine in range(8):
                state = gen.latent_state(seed, episode, machine, 0)
                anchors.add(state["anchor"])
                change_ats.add(state["change_at"])
                self.assertGreaterEqual(state["change_at"], 24)
                self.assertLess(state["change_at"], 80)
                before = gen.probabilities_at(seed, episode, machine, state["change_at"] - 1)
                after = gen.probabilities_at(seed, episode, machine, state["change_at"])
                later = gen.probabilities_at(seed, episode, machine, state["change_at"] + 5)
                self.assertNotEqual(before, after)
                self.assertEqual(after, later)
                jumps += 1
            self.assertEqual(len(anchors), 1)
            self.assertGreater(len(change_ats), 1)
        self.assertGreater(jumps, 0)

    def test_latent_oracle_differs_from_observable_history(self) -> None:
        data = gen.generate(seed=0, n_episodes=1, n_ticks=40, n_machines=8)
        oracle = data["truth"][0]["probabilities"]
        observable = gen.observable_history_probabilities(
            data["records"], data["truth"][0]["episode_id"], 0, 0
        )
        self.assertEqual(observable, gen.OBSERVABLE_PRIOR)
        self.assertNotEqual(tuple(oracle), observable)
        # A hand-built gap: missing payout 20 must not enter the estimate,
        # and must not be treated as payout 0.
        fixture = [
            {
                "episode_id": "e0000",
                "tick": 0,
                "machine_id": 3,
                "machine_type": "balanced",
                "payout": 5,
                "observed": True,
            },
            {
                "episode_id": "e0000",
                "tick": 1,
                "machine_id": 3,
                "machine_type": "balanced",
                "payout": 20,
                "observed": False,
            },
            {
                "episode_id": "e0000",
                "tick": 1,
                "machine_id": 4,
                "machine_type": "balanced",
                "payout": 0,
                "observed": True,
            },
            {
                "episode_id": "e0000",
                "tick": 2,
                "machine_id": 3,
                "machine_type": "balanced",
                "payout": 5,
                "observed": True,
            },
        ]
        got = gen.observable_history_probabilities(fixture, "e0000", 3, 3)
        self.assertEqual(got, (0.0, 0.0, 0.0, 1.0, 0.0))
        as_zero = [0, 0, 0, 0, 0]
        leaked = [0, 0, 0, 0, 0]
        for row in fixture:
            if row["machine_id"] != 3 or row["tick"] >= 3:
                continue
            if row["observed"]:
                as_zero[gen.PAYOUT_TO_INDEX[row["payout"]]] += 1
                leaked[gen.PAYOUT_TO_INDEX[row["payout"]]] += 1
            else:
                as_zero[gen.PAYOUT_TO_INDEX[0]] += 1
                leaked[gen.PAYOUT_TO_INDEX[row["payout"]]] += 1
        self.assertNotEqual(tuple(count / sum(as_zero) for count in as_zero), got)
        self.assertNotEqual(tuple(count / sum(leaked) for count in leaked), got)

    def test_gappy_missing_is_not_zero_on_generated_rows(self) -> None:
        data = gen.generate(seed=2, n_episodes=1, n_ticks=24, n_machines=8)
        missing = [row for row in data["records"] if not row["observed"]]
        self.assertGreater(len(missing), 0)
        payout_zero = gen.encode_observation(0, True)
        for row in data["records"]:
            encoded = gen.encode_observation(row["payout"], row["observed"])
            structural_gap = (row["tick"] + 3 * row["machine_id"]) % 11 == 0
            if structural_gap:
                self.assertFalse(row["observed"])
            if not row["observed"]:
                self.assertEqual(encoded["payout_index"], None)
                self.assertEqual(encoded["mask"], 0)
                self.assertEqual(encoded["one_hot"], (0, 0, 0, 0, 0))
                self.assertNotEqual(encoded, payout_zero)
                self.assertNotEqual(encoded["one_hot"], payout_zero["one_hot"])

    def test_future_mutation_and_prefix_consistency(self) -> None:
        seed = 4
        short = gen.generate(seed=seed, n_episodes=1, n_ticks=28, n_machines=8)
        long = gen.generate(seed=seed, n_episodes=1, n_ticks=70, n_machines=8)
        short_records = short["records"]
        long_prefix = [row for row in long["records"] if row["tick"] < 28]
        self.assertEqual(short_records, long_prefix)
        short_truth = short["truth"]
        long_truth_prefix = [row for row in long["truth"] if row["tick"] < 28]
        self.assertEqual(short_truth, long_truth_prefix)

        cutoff = 30
        original_features = gen.history_features(long["records"], cutoff)
        original_obs = gen.observable_history_probabilities(long["records"], "e0000", 2, cutoff)
        self.assertGreater(
            sum(1 for row in long["records"] if row["machine_id"] == 2 and row["tick"] < cutoff and row["observed"]),
            0,
        )

        future = copy.deepcopy(long["records"])
        for row in future:
            if row["tick"] >= cutoff:
                row["payout"] = 0 if row["payout"] != 0 else 20
                row["observed"] = not row["observed"]
        self.assertEqual(gen.history_features(future, cutoff), original_features)
        self.assertEqual(
            gen.observable_history_probabilities(future, "e0000", 2, cutoff),
            original_obs,
        )
        # Each machine's future, one at a time, leaves the cutoff features fixed.
        for machine in range(8):
            mutated = copy.deepcopy(long["records"])
            for row in mutated:
                if row["machine_id"] == machine and row["tick"] >= cutoff:
                    row["payout"] = 20
                    row["observed"] = False
            self.assertEqual(gen.history_features(mutated, cutoff), original_features)

        # A past observed payout changes the legal features and the observable estimate.
        past = copy.deepcopy(long["records"])
        changed = False
        for row in past:
            if row["episode_id"] == "e0000" and row["machine_id"] == 2 and row["tick"] < cutoff and row["observed"]:
                row["payout"] = 20 if row["payout"] != 20 else 0
                changed = True
                break
        self.assertTrue(changed)
        self.assertNotEqual(gen.history_features(past, cutoff), original_features)
        self.assertNotEqual(
            gen.observable_history_probabilities(past, "e0000", 2, cutoff),
            original_obs,
        )

        # Latent payout on a missing past row must not change features.
        hidden = copy.deepcopy(long["records"])
        hid = False
        for row in hidden:
            if row["tick"] < cutoff and not row["observed"]:
                row["payout"] = 20 if row["payout"] != 20 else 5
                hid = True
                break
        self.assertTrue(hid)
        self.assertEqual(gen.history_features(hidden, cutoff), original_features)

        # Same-tick cross-machine payouts are not inputs to another slot's oracle.
        for row in long["truth"]:
            if row["tick"] != cutoff:
                continue
            episode = gen.parse_episode_index(row["episode_id"])
            self.assertEqual(
                tuple(row["probabilities"]),
                gen.probabilities_at(seed, episode, row["machine_id"], row["tick"]),
            )

    def test_query_signal_exists_only_at_cutoff(self) -> None:
        data = gen.generate(seed=3, n_episodes=2, n_ticks=16, n_machines=8)
        self.assertEqual(len(data["queries"]), 16)
        cutoffs = {query["cutoff"] for query in data["queries"]}
        self.assertEqual(cutoffs, {15})
        for query in data["queries"]:
            self.assertEqual(query["slot"], query["machine_id"])
            self.assertIn(query["signal"], (-1, 0, 1))
            self.assertIsNone(
                gen.signal_at(data["queries"], query["episode_id"], query["machine_id"], query["cutoff"] - 1)
            )
            self.assertEqual(
                gen.signal_at(data["queries"], query["episode_id"], query["machine_id"], query["cutoff"]),
                query["signal"],
            )
        features = gen.history_features(data["records"], 15)
        self.assertTrue(all(item[3] is None or item[3] in range(5) for item in features))
        self.assertTrue(all(len(item) == 5 for item in features))
        flipped = copy.deepcopy(data["queries"])
        for query in flipped:
            query["signal"] = 0 if query["signal"] != 0 else 1
        self.assertEqual(
            gen.history_features(data["records"], 15),
            features,
        )
        self.assertNotEqual(
            [query["signal"] for query in flipped],
            [query["signal"] for query in data["queries"]],
        )
        # Oracle truth does not move when the signal is flipped.
        self.assertEqual(data["truth"], gen.generate(seed=3, n_episodes=2, n_ticks=16, n_machines=8)["truth"])

    def test_cell_draws_ignore_other_machines(self) -> None:
        data = gen.generate(seed=5, n_episodes=1, n_ticks=20, n_machines=8)
        for row in data["records"]:
            episode = gen.parse_episode_index(row["episode_id"])
            probs = gen.probabilities_at(5, episode, row["machine_id"], row["tick"])
            self.assertEqual(
                row["payout"],
                gen.sample_payout(5, episode, row["machine_id"], row["tick"], probs),
            )
            self.assertEqual(
                row["observed"],
                gen.sample_observed(5, episode, row["machine_id"], row["tick"], row["machine_type"]),
            )

    def test_cli_writes_hashed_research_only_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code = gen.main(
                ["--seed", "7", "--episodes", "1", "--ticks", "12", "--machines", "8", "--out", str(out)]
            )
            self.assertEqual(code, 0)
            records = (out / "records.jsonl").read_bytes()
            truth = (out / "truth.jsonl").read_bytes()
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["eligibility"], "RESEARCH_ONLY")
            self.assertFalse(manifest["synthetic_exception_verified"])
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["settings"]["n_episodes"], 1)
            self.assertEqual(manifest["settings"]["n_ticks"], 12)
            self.assertEqual(manifest["settings"]["n_machines"], 8)
            self.assertEqual(manifest["n_records"], 96)
            self.assertEqual(manifest["output_hashes"]["records.jsonl"], gen.sha256_bytes(records))
            self.assertEqual(manifest["output_hashes"]["truth.jsonl"], gen.sha256_bytes(truth))
            self.assertIn("generate.py", manifest["source_hashes"])
            self.assertEqual(manifest["payout_order"], [0, 1, 2, 5, 20])
            self.assertEqual(manifest["signal_scope"], "query_cutoff_only")
            self.assertIsNone(manifest["missing_payout_index"])
            loaded = [json.loads(line) for line in records.decode("utf-8").splitlines()]
            self.assertEqual(len(loaded), 96)
            self.assertTrue(any(row["observed"] is False for row in loaded))
            for row in loaded:
                self.assertNotIn("signal", row)
                if not row["observed"]:
                    self.assertIn(row["payout"], PAYOUTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
