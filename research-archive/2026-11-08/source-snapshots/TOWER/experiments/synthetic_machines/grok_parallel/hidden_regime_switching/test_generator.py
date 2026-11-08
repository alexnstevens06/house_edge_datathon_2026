"""Contract tests for the hidden_regime_switching toy generator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import generate as gen

PAYOUTS = (0, 1, 2, 5, 20)
PY = "/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python"


def _simplex(probs, length=5, places=9):
    if len(probs) != length:
        raise AssertionError(f"expected {length} probabilities, got {len(probs)}")
    if any(prob < -1e-12 for prob in probs):
        raise AssertionError(f"negative probability {probs}")
    total = sum(probs)
    if abs(total - 1.0) > 10 ** (-places):
        raise AssertionError(f"probabilities sum to {total}")


def _close(left, right, tol=1e-12):
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= tol for a, b in zip(left, right))


class HiddenRegimeTests(unittest.TestCase):
    def test_tables_and_mappings(self):
        self.assertEqual(gen.PAYOUTS, PAYOUTS)
        self.assertEqual([gen.PAYOUT_TO_IDX[p] for p in PAYOUTS], [0, 1, 2, 3, 4])
        self.assertEqual([gen.PAYOUT_IDX_TO_VALUE[i] for i in range(5)], list(PAYOUTS))
        self.assertEqual(gen.MACHINE_TYPES, ("balanced", "jackpot", "steady"))
        self.assertEqual(gen.TYPE_TO_IDX["balanced"], 0)
        self.assertEqual(gen.MISSING, -1)
        self.assertNotEqual(gen.MISSING, gen.PAYOUT_TO_IDX[0])
        for type_rows in gen.EMISSIONS:
            self.assertEqual(len(type_rows), gen.N_REGIMES)
            for row in type_rows:
                _simplex(row)
        _simplex(gen.PRIOR, length=gen.N_REGIMES)
        for payout_rows in gen.TRANS_OBS:
            for row in payout_rows:
                _simplex(row, length=gen.N_REGIMES)
        for row in gen.TRANS_MISS:
            _simplex(row, length=gen.N_REGIMES)
        for regime in range(gen.N_REGIMES):
            self.assertNotEqual(gen.TRANS_MISS[regime], gen.TRANS_OBS[0][regime])
        self.assertEqual(gen.dense_symbol(False, None), 0)
        self.assertEqual(gen.dense_symbol(True, 0), 1)
        self.assertNotEqual(gen.dense_symbol(False, None), gen.dense_symbol(True, 0))
        self.assertIsNone(gen.sparse_symbol(False, None))
        self.assertEqual(gen.sparse_symbol(True, 0), 0)
        self.assertEqual(gen.sparse_symbol(True, 20), 4)
        self.assertEqual(gen.obs_mask(True), 1)
        self.assertEqual(gen.obs_mask(False), 0)
        # u high enough to land in the last bin of the missing kernel only.
        miss = gen.sample_categorical(gen.TRANS_MISS[0], _Fixed(0.99))
        zero = gen.sample_categorical(gen.TRANS_OBS[0][0], _Fixed(0.99))
        self.assertNotEqual(miss, zero)

    def test_probability_simplex_and_oracle_mapping(self):
        data = gen.generate(seed=0, n_episodes=1, n_ticks=12, n_machines=4)
        self.assertLessEqual(len(data["records"]), gen.MAX_RECORDS)
        seen_types = set()
        oracle_differs = False
        for row in data["truth"]:
            _simplex(row["probabilities"])
            _simplex(row["observable_probabilities"])
            expected = list(gen.oracle_emission(row["machine_type"], row["regime"]))
            self.assertEqual(row["probabilities"], expected)
            self.assertEqual(row["probability_kind"], "latent_oracle")
            self.assertEqual(row["observable_kind"], "history_filter")
            self.assertEqual(len(row["probabilities"]), 5)
            if row["probabilities"] != row["observable_probabilities"]:
                oracle_differs = True
            seen_types.add(row["machine_type"])
        self.assertTrue(oracle_differs)
        self.assertEqual(seen_types, set(gen.MACHINE_TYPES))
        for row in data["records"]:
            self.assertIn(row["machine_type"], gen.MACHINE_TYPES)
            self.assertEqual(row["slot"], row["machine_id"])
            self.assertGreaterEqual(row["machine_id"], 0)
            self.assertLess(row["machine_id"], 4)
            if row["observed"]:
                self.assertIn(row["payout"], PAYOUTS)
                self.assertEqual(row["obs_mask"], 1)
                self.assertEqual(gen.dense_symbol(True, row["payout"]), gen.PAYOUT_TO_IDX[row["payout"]] + 1)
            else:
                self.assertIsNone(row["payout"])
                self.assertEqual(row["obs_mask"], 0)
                self.assertEqual(gen.dense_symbol(False, None), 0)

    def test_reproducible_and_seed_sensitive(self):
        first = gen.generate(seed=2, n_episodes=2, n_ticks=10, n_machines=3)
        second = gen.generate(seed=2, n_episodes=2, n_ticks=10, n_machines=3)
        self.assertEqual(first["records"], second["records"])
        self.assertEqual(first["truth"], second["truth"])
        self.assertEqual(first["queries"], second["queries"])
        other = gen.generate(seed=3, n_episodes=2, n_ticks=10, n_machines=3)
        self.assertNotEqual(first["records"], other["records"])

    def test_known_truth_matches_emission_and_filter_replay(self):
        seed, n_episodes, n_ticks, n_machines = 4, 1, 9, 3
        data = gen.generate(seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines)
        for machine_id in range(n_machines):
            path = gen.rollout_machine(seed, 0, machine_id, n_ticks)
            rows = [
                row for row in data["truth"]
                if row["machine_id"] == machine_id and row["role"] == "history"
            ]
            self.assertEqual(len(rows), n_ticks)
            for step, row in zip(path["steps"], rows):
                self.assertEqual(row["regime"], step["regime"])
                self.assertEqual(row["probabilities"], list(gen.oracle_emission(row["machine_type"], row["regime"])))
                self.assertTrue(_close(row["observable_probabilities"], step["observable_probabilities"]))
            cutoff = next(
                row for row in data["truth"]
                if row["machine_id"] == machine_id and row["role"] == "cutoff"
            )
            self.assertEqual(cutoff["tick"], n_ticks)
            self.assertEqual(cutoff["probabilities"], list(gen.oracle_emission(cutoff["machine_type"], cutoff["regime"])))
            self.assertEqual(cutoff["probabilities"], path["cutoff_probabilities"])

    def test_future_mutation_keeps_prefix_and_moves_later_oracle(self):
        seed, n_ticks, n_machines = 1, 16, 4
        base = gen.generate(seed=seed, n_episodes=1, n_ticks=n_ticks, n_machines=n_machines)
        base_truth = {(row["machine_id"], row["tick"]): row for row in base["truth"]}
        moved = False
        for machine_id in range(n_machines):
            for tick in range(n_ticks):
                record = next(
                    row for row in base["records"]
                    if row["machine_id"] == machine_id and row["tick"] == tick
                )
                flipped = _flip(record)
                alt = gen.generate(
                    seed=seed,
                    n_episodes=1,
                    n_ticks=n_ticks,
                    n_machines=n_machines,
                    observation_overrides={(0, machine_id, tick): flipped},
                )
                alt_truth = {(row["machine_id"], row["tick"]): row for row in alt["truth"]}
                for slot in range(n_machines):
                    last_same = n_ticks if slot != machine_id else tick
                    for earlier in range(last_same + 1):
                        left = base_truth[(slot, earlier)]
                        right = alt_truth[(slot, earlier)]
                        self.assertEqual(left["regime"], right["regime"], f"slot {slot} tick {earlier}")
                        self.assertEqual(left["probabilities"], right["probabilities"])
                        self.assertTrue(_close(left["observable_probabilities"], right["observable_probabilities"]))
                    if slot != machine_id:
                        self.assertEqual(
                            base_truth[(slot, n_ticks)]["probabilities"],
                            alt_truth[(slot, n_ticks)]["probabilities"],
                        )
                if flipped[0] and not record["observed"] and flipped[1] == 0:
                    later = alt_truth[(machine_id, tick + 1)]
                    current = base_truth[(machine_id, tick + 1)]
                    if later["regime"] != current["regime"] or later["probabilities"] != current["probabilities"]:
                        moved = True
        self.assertTrue(moved, "missing-vs-zero never changed a later regime; kernels are too weak")

    def test_gappy_missing_is_not_stored_as_zero(self):
        data = gen.generate(seed=0, n_episodes=1, n_ticks=8, n_machines=4)
        grid = gen.history_grid(data["records"], n_ticks=8, n_machines=4)
        missing_cells = [
            (tick, slot)
            for tick, row in enumerate(grid)
            for slot, value in enumerate(row)
            if value == gen.MISSING
        ]
        zero_cells = [
            (tick, slot)
            for tick, row in enumerate(grid)
            for slot, value in enumerate(row)
            if value == gen.PAYOUT_TO_IDX[0]
        ]
        self.assertTrue(missing_cells)
        self.assertTrue(zero_cells)
        tick, slot = missing_cells[0]
        self.assertNotEqual(grid[tick][slot], gen.PAYOUT_TO_IDX[0])
        record = next(row for row in data["records"] if row["tick"] == tick and row["machine_id"] == slot)
        self.assertFalse(record["observed"])
        self.assertIsNone(record["payout"])
        self.assertEqual(record["obs_mask"], 0)
        observed_zero = next(row for row in data["records"] if row["observed"] and row["payout"] == 0)
        self.assertEqual(gen.dense_symbol(True, observed_zero["payout"]), 1)
        self.assertEqual(gen.dense_symbol(False, None), 0)

    def test_queries_are_cutoff_only_and_history_has_no_signal(self):
        n_ticks, n_machines = 6, 8
        data = gen.generate(seed=5, n_episodes=1, n_ticks=n_ticks, n_machines=n_machines)
        self.assertEqual(len(data["queries"]), n_machines)
        self.assertEqual(len(data["records"]), n_ticks * n_machines)
        for record in data["records"]:
            self.assertNotIn("signal", record)
            self.assertLess(record["tick"], n_ticks)
        for index, query in enumerate(data["queries"]):
            self.assertEqual(query["slot"], index)
            self.assertEqual(query["machine_id"], index)
            self.assertEqual(query["cutoff"], n_ticks)
            self.assertIsNone(query["signal"])
            self.assertEqual(query["signal_scope"], "prediction_cutoff_only")
            self.assertTrue(query["query_id"].endswith(f":s{index}"))
        history_ticks = {row["tick"] for row in data["truth"] if row["role"] == "history"}
        self.assertTrue(all(tick < n_ticks for tick in history_ticks))
        self.assertEqual(data["eligibility"], "RESEARCH_ONLY")
        self.assertTrue(data["real_only_fallback"])

    def test_default_size_and_cap(self):
        self.assertLessEqual(8 * 96 * 8, gen.MAX_RECORDS)
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=8, n_ticks=200, n_machines=8)

    def test_cli_writes_hashed_research_only_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            completed = subprocess.run(
                [
                    PY, str(Path(gen.__file__).resolve()),
                    "--seed", "7", "--episodes", "1", "--ticks", "5", "--machines", "2",
                    "--out", str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["eligibility"], "RESEARCH_ONLY")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["eligibility"], "RESEARCH_ONLY")
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["settings"]["n_ticks"], 5)
            self.assertIn("generate.py", manifest["source_hashes"])
            for name, digest in manifest["output_hashes"].items():
                payload = (out / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
            records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
            truth = [json.loads(line) for line in (out / "truth.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 10)
            self.assertTrue(any(row["role"] == "cutoff" for row in truth))
            self.assertTrue(all(len(row["probabilities"]) == 5 for row in truth))


class _Fixed:
    """Stand-in with random() pinned, used to separate kernel bins."""

    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


def _flip(record: dict) -> tuple[bool, int | None]:
    if not record["observed"]:
        return True, 0
    if record["payout"] == 0:
        return False, None
    return True, 0


if __name__ == "__main__":
    unittest.main()
