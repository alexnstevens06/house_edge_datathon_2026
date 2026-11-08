#!/usr/bin/env python3
"""Meaningful checks for the long_lag_dependence toy generator."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import generate as gen

PAYOUTS = (0, 1, 2, 5, 20)
# Pinned tables. These duplicate the generator on purpose so a silent edit fails.
BASE_PCT = {
    "steady": (70, 18, 7, 4, 1),
    "balanced": (40, 28, 16, 10, 6),
    "jackpot": (50, 15, 12, 8, 15),
}
TILT_PCT = (
    (80, 12, 5, 2, 1),
    (60, 22, 10, 6, 2),
    (45, 25, 15, 10, 5),
    (30, 20, 18, 18, 14),
    (18, 14, 16, 22, 30),
)


def _expected(machine_type: str, lag1: int | None, lag2: int | None) -> list[Fraction]:
    probs = [Fraction(part, 100) for part in BASE_PCT[machine_type]]
    if lag1 is not None:
        tilt = [Fraction(part, 100) for part in TILT_PCT[lag1]]
        probs = [Fraction(1, 2) * a + Fraction(1, 2) * b for a, b in zip(probs, tilt)]
    if lag2 is not None:
        tilt = [Fraction(part, 100) for part in TILT_PCT[lag2]]
        probs = [Fraction(3, 4) * a + Fraction(1, 4) * b for a, b in zip(probs, tilt)]
    return probs


def _index_rows(rows: list[dict]) -> dict:
    return {(row["episode_id"], row["tick"], row["machine_id"]): row for row in rows}


class LongLagGeneratorTest(unittest.TestCase):
    def test_simplex_and_payout_mapping(self) -> None:
        data = gen.generate(seed=0, n_episodes=1, n_ticks=64, n_machines=8)
        self.assertEqual(len(data["records"]), 512)
        self.assertEqual(len(data["truth"]), len(data["records"]))
        self.assertEqual(len(data["observable"]), len(data["records"]))
        seen_types = set()
        for record, truth, observable in zip(data["records"], data["truth"], data["observable"]):
            self.assertEqual(record["episode_id"], truth["episode_id"])
            self.assertEqual(record["tick"], truth["tick"])
            self.assertEqual(record["machine_id"], truth["machine_id"])
            self.assertIn(record["machine_id"], range(8))
            self.assertEqual(record["machine_type"], gen.SLOT_TYPE[record["machine_id"]])
            seen_types.add(record["machine_type"])
            self.assertIn(record["payout"], PAYOUTS)
            self.assertEqual(record["payout_index"], PAYOUTS.index(record["payout"]))
            self.assertEqual(gen.PAYOUT_TO_INDEX[record["payout"]], record["payout_index"])
            if record["observed"]:
                self.assertEqual(record["observation_index"], record["payout_index"])
            else:
                self.assertEqual(record["observation_index"], gen.MISSING)
                self.assertNotEqual(record["observation_index"], 0)
            for probs in (truth["probabilities"], observable["probabilities"]):
                self.assertEqual(len(probs), 5)
                self.assertTrue(all(0.0 <= value <= 1.0 for value in probs))
                self.assertLess(abs(math.fsum(probs) - 1.0), 1e-12)
            self.assertEqual(list(gen.PAYOUTS), [0, 1, 2, 5, 20])
        self.assertEqual(seen_types, {"steady", "balanced", "jackpot"})
        self.assertTrue(any(not row["observed"] for row in data["records"]))
        self.assertTrue(any(row["observed"] for row in data["records"]))

    def test_pinned_mixture_and_family_invariant(self) -> None:
        # Known closed form: half base + half tilt, then a quarter of the second tilt.
        pinned = _expected("steady", 4, 0)
        got = gen.distribution("steady", 4, 0)
        for actual, expected in zip(got, pinned):
            self.assertLess(abs(actual - float(expected)), 1e-12)
        self.assertAlmostEqual(float(pinned[4]), 0.11875, places=12)

        cold = gen.distribution("jackpot", 0, None)
        hot = gen.distribution("jackpot", 4, None)
        self.assertGreater(hot[4], cold[4])  # lagged 20 raises P(20) vs lagged 0
        self.assertGreater(cold[0], hot[0])

        data = gen.generate(seed=2, n_episodes=1, n_ticks=80, n_machines=8)
        records = _index_rows(data["records"])
        observable_rows = _index_rows(data["observable"])
        checked_long = 0
        for row in data["truth"]:
            episode_id, tick, machine_id = row["episode_id"], row["tick"], row["machine_id"]
            machine_type = gen.SLOT_TYPE[machine_id]

            def latent(lag: int, tick: int = tick, episode_id: str = episode_id, machine_id: int = machine_id) -> int | None:
                if tick < lag:
                    return None
                return records[(episode_id, tick - lag, machine_id)]["payout_index"]

            def seen(lag: int, tick: int = tick, episode_id: str = episode_id, machine_id: int = machine_id) -> int | None:
                if tick < lag:
                    return None
                value = records[(episode_id, tick - lag, machine_id)]["observation_index"]
                if value == gen.MISSING:
                    return None
                return value

            self.assertEqual(row["probabilities"], gen.distribution(machine_type, latent(24), latent(48)))
            self.assertEqual(
                observable_rows[(episode_id, tick, machine_id)]["probabilities"],
                gen.distribution(machine_type, seen(24), seen(48)),
            )
            if tick >= gen.LAG2:
                checked_long += 1
        self.assertGreater(checked_long, 0)

    def test_reproducible_and_seed_sensitive(self) -> None:
        one = gen.generate(seed=0, n_episodes=2, n_ticks=40, n_machines=8)
        two = gen.generate(seed=0, n_episodes=2, n_ticks=40, n_machines=8)
        self.assertEqual(one["records"], two["records"])
        self.assertEqual(one["truth"], two["truth"])
        self.assertEqual(one["observable"], two["observable"])
        other = gen.generate(seed=1, n_episodes=2, n_ticks=40, n_machines=8)
        self.assertNotEqual(one["records"], other["records"])
        self.assertNotEqual(
            [row["probabilities"] for row in one["truth"]],
            [row["probabilities"] for row in other["truth"]],
        )

    def test_prefix_consistency_and_future_mutation(self) -> None:
        short = gen.generate(seed=3, n_episodes=2, n_ticks=40, n_machines=8)
        longer = gen.generate(seed=3, n_episodes=2, n_ticks=72, n_machines=8)
        self.assertEqual(short["records"], [row for row in longer["records"] if row["tick"] < 40])
        self.assertEqual(short["truth"], [row for row in longer["truth"] if row["tick"] < 40])
        self.assertEqual(short["observable"], [row for row in longer["observable"] if row["tick"] < 40])
        more_eps = gen.generate(seed=3, n_episodes=3, n_ticks=72, n_machines=8)
        first_eps = {row["episode_id"] for row in short["records"]}
        self.assertEqual(
            [row for row in longer["records"] if row["episode_id"] in first_eps],
            [row for row in more_eps["records"] if row["episode_id"] in first_eps],
        )
        # Prefix is not a constant table: lag 24 already moves probabilities.
        base = gen.distribution("steady", None, None)
        moved = [
            row
            for row in short["truth"]
            if row["tick"] >= gen.LAG and row["machine_id"] == 0 and row["probabilities"] != base
        ]
        self.assertTrue(moved)

        episode_id = short["records"][0]["episode_id"]
        by_tick = {tick: [] for tick in range(72)}
        for row in longer["records"]:
            if row["episode_id"] == episode_id and row["machine_id"] == 0:
                by_tick[row["tick"]] = row["payout_index"]
        series = [by_tick[tick] for tick in range(72)]
        tick = 60
        original = gen.distribution("steady", series[tick - 24], series[tick - 48])
        self.assertEqual(
            next(row["probabilities"] for row in longer["truth"] if row["episode_id"] == episode_id and row["tick"] == tick and row["machine_id"] == 0),
            original,
        )
        future = series[:]
        future[tick] = 4 if future[tick] != 4 else 0
        future[tick + 5] = 0 if future[tick + 5] != 0 else 4
        self.assertEqual(gen.distribution("steady", future[tick - 24], future[tick - 48]), original)
        # Recent tick is not a long lag, so it is not an input either.
        recent = series[:]
        recent[tick - 1] = 4 if recent[tick - 1] != 4 else 0
        self.assertEqual(gen.distribution("steady", recent[tick - 24], recent[tick - 48]), original)
        # Mutating the lag itself must move the oracle (tilts differ).
        lagged = series[:]
        lagged[tick - 24] = 0 if lagged[tick - 24] != 0 else 4
        self.assertNotEqual(gen.distribution("steady", lagged[tick - 24], lagged[tick - 48]), original)
        # Another machine's contemporaneous or lagged spin is not an input.
        other = [
            row["payout_index"]
            for row in longer["records"]
            if row["episode_id"] == episode_id and row["machine_id"] == 1
        ]
        self.assertEqual(len(other), 72)
        own = gen.distribution("steady", series[tick - 24], series[tick - 48])
        self.assertEqual(own, original)

    def test_missing_is_not_zero_and_oracle_differs_from_observable(self) -> None:
        base = gen.distribution("balanced", None, None)
        as_zero = gen.distribution("balanced", 0, None)
        self.assertNotEqual(base, as_zero)
        with self.assertRaises(ValueError):
            gen.distribution("balanced", gen.MISSING, None)

        data = gen.generate(seed=4, n_episodes=1, n_ticks=48, n_machines=8)
        records = _index_rows(data["records"])
        found_gap = False
        for truth, observable in zip(data["truth"], data["observable"]):
            tick = truth["tick"]
            if tick < gen.LAG or tick >= gen.LAG2:
                continue
            lag_rec = records[(truth["episode_id"], tick - gen.LAG, truth["machine_id"])]
            if lag_rec["observed"]:
                continue
            dropped = gen.distribution(lag_rec["machine_type"], None, None)
            filled_as_zero = gen.distribution(lag_rec["machine_type"], 0, None)
            # Gap: observer drops the lag. Filling payout 0 would apply TILT[0].
            self.assertEqual(observable["probabilities"], dropped)
            self.assertNotEqual(observable["probabilities"], filled_as_zero)
            # Oracle still conditions on the latent spin at the unobserved lag.
            self.assertEqual(
                truth["probabilities"],
                gen.distribution(lag_rec["machine_type"], lag_rec["payout_index"], None),
            )
            self.assertNotEqual(truth["probabilities"], observable["probabilities"])
            found_gap = True
        self.assertTrue(found_gap)

    def test_queries_only_at_cutoff_without_invented_signal(self) -> None:
        data = gen.generate(seed=0, n_episodes=2, n_ticks=36, n_machines=8)
        self.assertEqual(len(data["queries"]), 16)
        history_ticks = {row["tick"] for row in data["records"]}
        for query in data["queries"]:
            self.assertEqual(query["cutoff"], 36)
            self.assertIsNone(query["signal"])
            self.assertNotIn("signal_history", query)
            self.assertTrue(all(tick < query["cutoff"] for tick in history_ticks))
        for row in data["records"]:
            self.assertNotIn("signal", row)
        self.assertEqual(data["meta"]["label"], "RESEARCH_ONLY")
        self.assertTrue(data["meta"]["real_only_fallback"])

    def test_default_cap_and_cli_manifest(self) -> None:
        data = gen.generate()
        self.assertEqual(len(data["records"]), 8 * 96 * 8)
        self.assertLessEqual(len(data["records"]), gen.MAX_RECORDS)
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=8, n_ticks=200, n_machines=8)
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=1, n_ticks=16, n_machines=7)

        root = Path(__file__).resolve().parent
        manifest_card = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_card["label"], "RESEARCH_ONLY")
        self.assertTrue(manifest_card["real_only_fallback"])
        self.assertEqual(manifest_card["family"], "long_lag_dependence")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    __import__("sys").executable,
                    str(root / "generate.py"),
                    "--seed",
                    "5",
                    "--episodes",
                    "1",
                    "--ticks",
                    "64",
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("RESEARCH_ONLY", proc.stdout)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["label"], "RESEARCH_ONLY")
            self.assertEqual(manifest["seed"], 5)
            self.assertEqual(manifest["eligibility"], "unresolved")
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["settings"]["record_count"], 512)
            for name in ("records.jsonl", "truth.jsonl", "observable.jsonl", "queries.jsonl"):
                digest = __import__("hashlib").sha256((out / name).read_bytes()).hexdigest()
                self.assertEqual(manifest["output_hashes"][name], digest)
            self.assertEqual(len((out / "records.jsonl").read_text(encoding="utf-8").splitlines()), 512)
            truth_row = json.loads((out / "truth.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(list(truth_row), ["episode_id", "machine_id", "probabilities", "tick"])
            self.assertEqual(len(truth_row["probabilities"]), 5)
            query = json.loads((out / "queries.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(query["cutoff"], 64)
            self.assertIsNone(query["signal"])
            self.assertNotIn("signal", json.loads((out / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]))


if __name__ == "__main__":
    unittest.main()
