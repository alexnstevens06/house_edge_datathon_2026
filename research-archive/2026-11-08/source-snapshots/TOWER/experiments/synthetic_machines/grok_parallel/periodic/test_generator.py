"""Tests for the periodic toy generator. Stdlib only."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import generate as gen

PAYOUTS = (0, 1, 2, 5, 20)
KICK = (3, 0, 1, 2, 5)
PERIOD = 8


def _norm(weights: tuple[int, ...]) -> tuple[float, ...]:
    head = [weight / 40.0 for weight in weights[:-1]]
    return tuple(head + [1.0 - sum(head)])


def _phase(seed: int, episode_index: int, machine_id: int) -> int:
    return (seed * 17 + episode_index * 5 + machine_id * 3 + 1) % PERIOD


def _kick(payout: int, observed: bool, mode: str) -> int:
    index = PAYOUTS.index(payout)
    if mode == "latent":
        return KICK[index]
    if mode == "observable":
        return KICK[index] if observed else 0
    if mode == "missing_as_zero":
        return KICK[index] if observed else KICK[0]
    raise AssertionError(mode)


def _advance(phase: int, payout: int, observed: bool, mode: str) -> int:
    return (phase + 1 + _kick(payout, observed, mode)) % PERIOD


def _rows(bundle: dict, episode_index: int = 0, machine_id: int | None = None) -> list[dict]:
    rows = [row for row in bundle["records"] if row["episode_index"] == episode_index]
    if machine_id is not None:
        rows = [row for row in rows if row["machine_id"] == machine_id]
    return rows


def _truth_map(bundle: dict) -> dict[tuple, list[float]]:
    return {
        (row["episode_index"], row["tick"], row["machine_id"]): row["probabilities"]
        for row in bundle["truth"]
    }


class PeriodicGeneratorTest(unittest.TestCase):
    def test_probability_simplex_and_mapping(self) -> None:
        bundle = gen.generate(seed=0, n_episodes=2, n_ticks=12, n_machines=8)
        self.assertEqual(len(bundle["records"]), len(bundle["truth"]))
        self.assertEqual(len(bundle["observable"]), len(bundle["records"]))
        self.assertEqual(gen.TYPE_TO_INDEX, {"balanced": 0, "jackpot": 1, "steady": 2})
        self.assertNotEqual(gen.TYPE_TO_INDEX["steady"], 0)
        seen_types = set()
        for record, truth, observed in zip(bundle["records"], bundle["truth"], bundle["observable"]):
            self.assertEqual(record["episode_id"], truth["episode_id"])
            self.assertEqual(record["tick"], truth["tick"])
            self.assertEqual(record["machine_id"], truth["machine_id"])
            self.assertIn(record["machine_id"], range(8))
            self.assertEqual(record["slot_index"], record["machine_id"])
            self.assertEqual(record["machine_type"], gen.MACHINE_TYPE_BY_SLOT[record["machine_id"]])
            self.assertEqual(record["machine_type_index"], gen.TYPE_TO_INDEX[record["machine_type"]])
            self.assertIn(record["payout"], PAYOUTS)
            self.assertIsInstance(record["observed"], bool)
            if record["observed"]:
                self.assertEqual(record["payout_index"], PAYOUTS.index(record["payout"]))
            else:
                self.assertEqual(record["payout_index"], -1)
                self.assertNotEqual(record["payout_index"], 0)
            self.assertEqual(gen.observation_payout_index(record), record["payout_index"])
            seen_types.add(record["machine_type"])
            for block in (truth, observed):
                probs = block["probabilities"]
                self.assertEqual(len(probs), 5)
                self.assertTrue(all(prob > 0.0 for prob in probs))
                self.assertEqual(sum(probs), 1.0)
            self.assertEqual(truth["probability_kind"], "latent_oracle")
            self.assertEqual(observed["probability_kind"], "observable_history")
        self.assertEqual(seen_types, {"steady", "balanced", "jackpot"})
        for row in gen.CATALOG_WEIGHTS["steady"]:
            self.assertEqual(len(row), 5)
            self.assertEqual(sum(row), 40)

    def test_reproducibility_and_no_shared_state(self) -> None:
        first = gen.generate(seed=2, n_episodes=2, n_ticks=10, n_machines=8)
        other = gen.generate(seed=3, n_episodes=1, n_ticks=5, n_machines=8)
        second = gen.generate(seed=2, n_episodes=2, n_ticks=10, n_machines=8)
        self.assertEqual(first, second)
        self.assertNotEqual(other["settings"]["seed"], first["settings"]["seed"])
        first["records"][0]["payout"] = 999
        third = gen.generate(seed=2, n_episodes=2, n_ticks=10, n_machines=8)
        self.assertEqual(second, third)
        self.assertEqual(first["records"][0]["payout"], 999)

    def test_different_seed_changes_draws_and_opening_law(self) -> None:
        left = gen.generate(seed=0, n_episodes=1, n_ticks=8, n_machines=8)
        right = gen.generate(seed=1, n_episodes=1, n_ticks=8, n_machines=8)
        left_draws = [(row["tick"], row["machine_id"], row["payout"], row["observed"]) for row in left["records"]]
        right_draws = [(row["tick"], row["machine_id"], row["payout"], row["observed"]) for row in right["records"]]
        self.assertNotEqual(left_draws, right_draws)
        # Seed shifts the initial phase, so the tick-0 oracle law moves.
        self.assertNotEqual(left["truth"][2]["probabilities"], right["truth"][2]["probabilities"])

    def test_future_and_same_tick_mutation_leaves_prefix(self) -> None:
        base = gen.generate(seed=4, n_episodes=2, n_ticks=16, n_machines=8)
        target = next(row for row in base["records"] if row["episode_index"] == 0 and row["tick"] == 9 and row["machine_id"] == 2)
        forced = 0 if target["payout"] != 0 else 20
        mutated = gen.generate(
            seed=4,
            n_episodes=2,
            n_ticks=16,
            n_machines=8,
            overrides={(0, 9, 2): {"payout": forced, "observed": True}},
        )
        base_prefix = [row for row in base["records"] if not (row["episode_index"] == 0 and row["tick"] >= 9)]
        mutated_prefix = [row for row in mutated["records"] if not (row["episode_index"] == 0 and row["tick"] >= 9)]
        self.assertEqual(base_prefix, mutated_prefix)
        same_tick_others = [
            row for row in base["records"]
            if row["episode_index"] == 0 and row["tick"] == 9 and row["machine_id"] != 2
        ]
        mutated_same_tick = [
            row for row in mutated["records"]
            if row["episode_index"] == 0 and row["tick"] == 9 and row["machine_id"] != 2
        ]
        self.assertEqual(same_tick_others, mutated_same_tick)
        forced_row = next(row for row in mutated["records"] if row["episode_index"] == 0 and row["tick"] == 9 and row["machine_id"] == 2)
        self.assertEqual(forced_row["payout"], forced)
        base_truth = _truth_map(base)
        mutated_truth = _truth_map(mutated)
        for key, probs in base_truth.items():
            episode_index, tick, machine_id = key
            if episode_index != 0 or tick <= 9 or (tick == 9 and machine_id == 2):
                # Probabilities at tick 9 do not read the tick-9 payout.
                if tick < 9 or episode_index != 0:
                    self.assertEqual(mutated_truth[key], probs)
            if tick == 9:
                self.assertEqual(mutated_truth[key], probs)
        self.assertNotEqual(base_truth[(0, 10, 2)], mutated_truth[(0, 10, 2)])
        self.assertEqual(base_truth[(0, 10, 5)], mutated_truth[(0, 10, 5)])
        self.assertEqual(base_truth[(1, 15, 2)], mutated_truth[(1, 15, 2)])

        # In-memory future edits do not change replayed probabilities at or before t.
        edited = copy.deepcopy(base["records"])
        for row in edited:
            if row["episode_index"] == 0 and row["tick"] >= 6:
                row["payout"] = 20
                row["observed"] = False
                row["payout_index"] = -1
        for machine_id in range(8):
            for tick in range(7):
                original = _replay_one(base["records"], 4, 0, machine_id, tick, "latent")
                after = _replay_one(edited, 4, 0, machine_id, tick, "latent")
                self.assertEqual(original, after)
                self.assertEqual(original, base_truth[(0, tick, machine_id)])

    def test_prefix_mutation_changes_later_oracle_only_after_the_edit(self) -> None:
        base = gen.generate(seed=5, n_episodes=1, n_ticks=12, n_machines=8)
        current = next(row for row in base["records"] if row["tick"] == 3 and row["machine_id"] == 2)
        forced = 20 if current["payout"] != 20 else 0
        self.assertNotEqual(KICK[PAYOUTS.index(current["payout"])], KICK[PAYOUTS.index(forced)])
        mutated = gen.generate(
            seed=5,
            n_episodes=1,
            n_ticks=12,
            n_machines=8,
            overrides={(0, 3, 2): {"payout": forced, "observed": True}},
        )
        early_base = [row for row in base["records"] if row["tick"] < 3]
        early_mut = [row for row in mutated["records"] if row["tick"] < 3]
        self.assertEqual(early_base, early_mut)
        base_truth = _truth_map(base)
        mutated_truth = _truth_map(mutated)
        for tick in range(4):
            for machine_id in range(8):
                self.assertEqual(base_truth[(0, tick, machine_id)], mutated_truth[(0, tick, machine_id)])
        self.assertNotEqual(base_truth[(0, 4, 2)], mutated_truth[(0, 4, 2)])
        for machine_id in range(8):
            if machine_id == 2:
                continue
            self.assertEqual(base_truth[(0, 4, machine_id)], mutated_truth[(0, 4, machine_id)])
            self.assertEqual(base_truth[(0, 11, machine_id)], mutated_truth[(0, 11, machine_id)])

    def test_gappy_missing_is_not_zero(self) -> None:
        hidden_zero = [
            {"payout": 1, "observed": True},
            {"payout": 0, "observed": False},
        ]
        hidden_jackpot = [
            {"payout": 1, "observed": True},
            {"payout": 20, "observed": False},
        ]
        seen_zero = [
            {"payout": 1, "observed": True},
            {"payout": 0, "observed": True},
        ]
        latent_z, obs_z, zero_z = _terminals(0, 0, 2, "balanced", hidden_zero)
        self.assertEqual(latent_z, zero_z)
        self.assertNotEqual(obs_z, zero_z)
        self.assertNotEqual(obs_z, latent_z)
        latent_j, obs_j, zero_j = _terminals(0, 0, 2, "balanced", hidden_jackpot)
        self.assertNotEqual(latent_j, obs_j)
        self.assertNotEqual(latent_j, zero_j)
        self.assertNotEqual(obs_j, zero_j)
        latent_s, obs_s, zero_s = _terminals(0, 0, 2, "balanced", seen_zero)
        self.assertEqual(latent_s, obs_s)
        self.assertEqual(obs_s, zero_s)

        bundle = gen.generate(seed=0, n_episodes=1, n_ticks=24, n_machines=8)
        gaps = [row for row in bundle["records"] if not row["observed"]]
        self.assertTrue(gaps)
        self.assertTrue(any(row["payout"] != 0 for row in gaps))
        self.assertTrue(any(row["payout"] == 0 and row["observed"] for row in bundle["records"]))
        self.assertTrue(all(row["payout_index"] == -1 for row in gaps))
        wrong = []
        right = []
        for row in bundle["records"]:
            if row["observed"]:
                wrong.append(row["payout_index"])
                right.append(row["payout_index"])
            else:
                wrong.append(0)  # the mistake the mask exists to prevent
        self.assertNotEqual(wrong.count(0), right.count(0))
        # A multiple of 8 gaps puts balanced back on the same row, so scan every
        # prefix. One gap is enough to split observable history from zero-imputation.
        split_zero = False
        split_oracle = False
        for machine_id in range(8):
            machine_type = gen.MACHINE_TYPE_BY_SLOT[machine_id]
            rows = _rows(bundle, 0, machine_id)
            latent_steps, latent_terminal = gen.replay_probabilities(0, 0, machine_id, machine_type, rows, "latent")
            obs_steps, obs_terminal = gen.replay_probabilities(0, 0, machine_id, machine_type, rows, "observable")
            zero_steps, zero_terminal = gen.replay_probabilities(0, 0, machine_id, machine_type, rows, "missing_as_zero")
            if obs_steps != zero_steps or obs_terminal != zero_terminal:
                split_zero = True
            if latent_steps != obs_steps or latent_terminal != obs_terminal:
                split_oracle = True
        self.assertTrue(split_zero)
        self.assertTrue(split_oracle)

    def test_oracle_truth_matches_independent_phase_and_not_history(self) -> None:
        bundle = gen.generate(seed=7, n_episodes=2, n_ticks=20, n_machines=8)
        truth = _truth_map(bundle)
        observable = {
            (row["episode_index"], row["tick"], row["machine_id"]): row["probabilities"]
            for row in bundle["observable"]
        }
        diverged = False
        for episode_index in range(2):
            for machine_id in range(8):
                machine_type = gen.MACHINE_TYPE_BY_SLOT[machine_id]
                phase_l = _phase(7, episode_index, machine_id)
                phase_o = phase_l
                for row in _rows(bundle, episode_index, machine_id):
                    expected = list(_norm(gen.CATALOG_WEIGHTS[machine_type][phase_l]))
                    self.assertEqual(truth[(episode_index, row["tick"], machine_id)], expected)
                    self.assertEqual(
                        observable[(episode_index, row["tick"], machine_id)],
                        list(_norm(gen.CATALOG_WEIGHTS[machine_type][phase_o])),
                    )
                    # Period-4 types can share a row when phases differ by 4.
                    # Inequality is required only when the catalog rows differ.
                    if gen.CATALOG_WEIGHTS[machine_type][phase_l] != gen.CATALOG_WEIGHTS[machine_type][phase_o]:
                        diverged = True
                        self.assertNotEqual(
                            truth[(episode_index, row["tick"], machine_id)],
                            observable[(episode_index, row["tick"], machine_id)],
                        )
                    phase_l = _advance(phase_l, row["payout"], row["observed"], "latent")
                    phase_o = _advance(phase_o, row["payout"], row["observed"], "observable")
                query = next(
                    item for item in bundle["queries"]
                    if item["episode_index"] == episode_index and item["machine_id"] == machine_id
                )
                self.assertEqual(query["cutoff"], 20)
                self.assertIsNone(query["signal"])
                self.assertEqual(query["latent_oracle_probabilities"], list(_norm(gen.CATALOG_WEIGHTS[machine_type][phase_l])))
                self.assertEqual(
                    query["observable_history_probabilities"],
                    list(_norm(gen.CATALOG_WEIGHTS[machine_type][phase_o])),
                )
                self.assertTrue(all(row["tick"] < query["cutoff"] for row in bundle["records"]))
        self.assertTrue(diverged)
        self.assertTrue(all("signal" not in row for row in bundle["records"]))

    def test_family_period_invariant(self) -> None:
        for phase in range(4):
            self.assertEqual(gen.CATALOG_WEIGHTS["steady"][phase], gen.CATALOG_WEIGHTS["steady"][phase + 4])
            self.assertEqual(gen.CATALOG_WEIGHTS["jackpot"][phase], gen.CATALOG_WEIGHTS["jackpot"][phase + 4])
        balanced = gen.CATALOG_WEIGHTS["balanced"]
        self.assertEqual(len(set(balanced)), 8)
        for phase in range(4):
            self.assertNotEqual(balanced[phase], balanced[phase + 4])
        # Same type and same latent phase => same law, including across machines.
        self.assertEqual(gen.catalog_probabilities("steady", 1), gen.catalog_probabilities("steady", 5))
        self.assertNotEqual(gen.catalog_probabilities("balanced", 1), gen.catalog_probabilities("balanced", 5))
        bundle = gen.generate(seed=8, n_episodes=1, n_ticks=16, n_machines=8)
        by_phase: dict[tuple[str, int], tuple[float, ...]] = {}
        for machine_id in range(8):
            machine_type = gen.MACHINE_TYPE_BY_SLOT[machine_id]
            phase = _phase(8, 0, machine_id)
            for row in _rows(bundle, 0, machine_id):
                probs = tuple(next(
                    item["probabilities"] for item in bundle["truth"]
                    if item["tick"] == row["tick"] and item["machine_id"] == machine_id
                ))
                key = (machine_type, phase)
                if key in by_phase:
                    self.assertEqual(by_phase[key], probs)
                by_phase[key] = probs
                if machine_type in {"steady", "jackpot"}:
                    twin = (machine_type, (phase + 4) % PERIOD)
                    if twin in by_phase:
                        self.assertEqual(by_phase[twin], probs)
                phase = _advance(phase, row["payout"], row["observed"], "latent")
        self.assertTrue(any(phase >= 4 for (machine_type, phase) in by_phase if machine_type == "steady"))

    def test_record_cap_and_default_shape(self) -> None:
        bundle = gen.generate()
        self.assertEqual(bundle["settings"]["n_records"], 8 * 96 * 8)
        self.assertLessEqual(bundle["settings"]["n_records"], 8192)
        self.assertEqual(len(bundle["queries"]), 8 * 8)
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=9, n_ticks=128, n_machines=8)
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=1, n_ticks=4, n_machines=7)

    def test_cli_research_only_manifest_and_hashes(self) -> None:
        root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                os.path.join(root, "generate.py"),
                "--seed", "1",
                "--episodes", "1",
                "--ticks", "8",
                "--out", tmp,
            ]
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0)
            manifest = json.loads(Path(tmp, "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["eligibility"], "RESEARCH_ONLY")
            self.assertTrue(manifest["research_only"])
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["seed"], 1)
            self.assertEqual(manifest["n_records"], 64)
            for name in ("records.jsonl", "truth.jsonl"):
                digest = gen.file_sha256(os.path.join(tmp, name))  # type: ignore[arg-type]
                self.assertEqual(manifest["output_hashes"][name], digest)
            self.assertEqual(manifest["source_hashes"]["generate.py"], gen.file_sha256(os.path.join(root, "generate.py")))  # type: ignore[arg-type]
            records = [json.loads(line) for line in Path(tmp, "records.jsonl").read_text(encoding="utf-8").splitlines()]
            truth = [json.loads(line) for line in Path(tmp, "truth.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 64)
            self.assertEqual(len(truth), 64)
            self.assertTrue(all("signal" not in row for row in records))
            self.assertTrue(all(sum(row["probabilities"]) == 1.0 for row in truth))
            self.assertTrue(all(row["probability_kind"] == "latent_oracle" for row in truth))


def _replay_one(records: list[dict], seed: int, episode_index: int, machine_id: int, tick: int, mode: str) -> list[float]:
    rows = [
        row for row in records
        if row["episode_index"] == episode_index and row["machine_id"] == machine_id and row["tick"] < tick
    ]
    machine_type = gen.MACHINE_TYPE_BY_SLOT[machine_id]
    _steps, terminal = gen.replay_probabilities(seed, episode_index, machine_id, machine_type, rows, mode)
    return list(terminal)


def _terminals(seed: int, episode_index: int, machine_id: int, machine_type: str, rows: list[dict]):
    out = []
    for mode in ("latent", "observable", "missing_as_zero"):
        _steps, terminal = gen.replay_probabilities(seed, episode_index, machine_id, machine_type, rows, mode)
        out.append(terminal)
    return tuple(out)


if __name__ == "__main__":
    unittest.main()
