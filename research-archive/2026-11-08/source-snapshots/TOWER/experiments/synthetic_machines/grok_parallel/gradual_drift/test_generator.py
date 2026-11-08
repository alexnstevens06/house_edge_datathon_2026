#!/usr/bin/env python3
"""Invariants for the gradual_drift toy. Stdlib only."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import generate as gd


PAYOUTS = (0, 1, 2, 5, 20)


def _simplex(probabilities):
    if len(probabilities) != 5:
        return False
    if any((not math.isfinite(p)) or p < 0.0 or p > 1.0 for p in probabilities):
        return False
    return abs(math.fsum(probabilities) - 1.0) <= 1e-12


def _index_rows(rows):
    return {(row["episode_id"], row["tick"], row["machine_id"]): row for row in rows}


def _max_abs_diff(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


def _replay_latent(bundle, seed):
    """Independent recurrence of the disclosed latent oracle.

    Uses constants from the module, but writes the tick order itself:
    probabilities at t are fixed from EWMA state built only on ticks < t,
    then every machine's true payout (observed or not) updates that EWMA.
    """
    settings = bundle["settings"]
    n_ticks = settings["n_ticks"]
    n_machines = settings["n_machines"]
    records = _index_rows(bundle["records"])
    expected = {}
    by_episode = []
    for episode_index in range(settings["n_episodes"]):
        episode_id = f"gd-e{episode_index:03d}"
        by_episode.append(episode_id)
        ewma = [0.0] * n_machines
        seen = [0] * n_machines
        for tick in range(n_ticks):
            snapshot_ewma = list(ewma)
            snapshot_seen = list(seen)
            updates = []
            for machine_id in range(n_machines):
                kind = gd.machine_type(machine_id)
                phi = gd.initial_phase(seed, episode_index, machine_id)
                phi += (
                    gd.drift_direction(seed, episode_index, machine_id)
                    * gd.SPEED[kind]
                    * gd.time_fraction(tick, n_ticks)
                )
                phi += gd.OWN_GAIN * snapshot_ewma[machine_id]
                others = [
                    snapshot_ewma[other]
                    for other in range(n_machines)
                    if other != machine_id and snapshot_seen[other] > 0
                ]
                shared = 0.0 if not others else math.fsum(others) / len(others)
                phi += gd.SHARED_GAIN * shared
                logits = tuple(
                    gd.BASE_LOGITS[kind][slot] + phi * gd.DRIFT_LOGITS[kind][slot] for slot in range(5)
                )
                expected[(episode_id, tick, machine_id)] = gd.softmax(logits)
                payout = records[(episode_id, tick, machine_id)]["payout"]
                updates.append(gd.centered_payout(payout))
            for machine_id, center in enumerate(updates):
                ewma[machine_id] = (1.0 - gd.EWMA_ALPHA) * ewma[machine_id] + gd.EWMA_ALPHA * center
                seen[machine_id] += 1
    return expected


def _replay_latent_missing_as_zero(bundle, seed):
    """Wrong model: an unobserved tick contributes centered payout 0 instead of the true spin."""
    settings = bundle["settings"]
    n_ticks = settings["n_ticks"]
    n_machines = settings["n_machines"]
    records = _index_rows(bundle["records"])
    expected = {}
    for episode_index in range(settings["n_episodes"]):
        episode_id = f"gd-e{episode_index:03d}"
        ewma = [0.0] * n_machines
        seen = [0] * n_machines
        for tick in range(n_ticks):
            snapshot_ewma = list(ewma)
            snapshot_seen = list(seen)
            updates = []
            for machine_id in range(n_machines):
                kind = gd.machine_type(machine_id)
                phi = gd.initial_phase(seed, episode_index, machine_id)
                phi += (
                    gd.drift_direction(seed, episode_index, machine_id)
                    * gd.SPEED[kind]
                    * gd.time_fraction(tick, n_ticks)
                )
                phi += gd.OWN_GAIN * snapshot_ewma[machine_id]
                others = [
                    snapshot_ewma[other]
                    for other in range(n_machines)
                    if other != machine_id and snapshot_seen[other] > 0
                ]
                shared = 0.0 if not others else math.fsum(others) / len(others)
                phi += gd.SHARED_GAIN * shared
                logits = tuple(
                    gd.BASE_LOGITS[kind][slot] + phi * gd.DRIFT_LOGITS[kind][slot] for slot in range(5)
                )
                expected[(episode_id, tick, machine_id)] = gd.softmax(logits)
                row = records[(episode_id, tick, machine_id)]
                payout = 0 if not row["observed"] else row["payout"]
                updates.append(gd.centered_payout(payout))
            for machine_id, center in enumerate(updates):
                ewma[machine_id] = (1.0 - gd.EWMA_ALPHA) * ewma[machine_id] + gd.EWMA_ALPHA * center
                seen[machine_id] += 1
    return expected


class GradualDriftTests(unittest.TestCase):
    def test_simplex_and_slot_mapping(self):
        bundle = gd.generate(seed=0, n_episodes=2, n_ticks=24, n_machines=8)
        self.assertLessEqual(len(bundle["records"]), gd.MAX_RECORDS)
        self.assertEqual(len(bundle["records"]), len(bundle["truth"]))
        self.assertEqual(bundle["eligibility"], "RESEARCH_ONLY")
        self.assertTrue(bundle["real_only_fallback"])
        self.assertEqual(
            bundle["mappings"]["payout_slot_index"],
            {"0": 0, "1": 1, "2": 2, "5": 3, "20": 4},
        )
        self.assertEqual(
            bundle["mappings"]["probability_vector_slots"],
            ["p_0", "p_1", "p_2", "p_5", "p_20"],
        )
        self.assertEqual(bundle["mappings"]["query_column_index"]["signal"], 4)
        self.assertEqual(bundle["mappings"]["query_column_index"]["cutoff"], 5)
        seen_ids = set()
        for record, truth in zip(bundle["records"], bundle["truth"]):
            self.assertEqual(record["episode_id"], truth["episode_id"])
            self.assertEqual(record["tick"], truth["tick"])
            self.assertEqual(record["machine_id"], truth["machine_id"])
            self.assertIn(record["machine_id"], range(8))
            self.assertEqual(record["machine_type"], gd.MACHINE_TYPE_BY_ID[record["machine_id"]])
            self.assertIn(record["machine_type"], ("steady", "balanced", "jackpot"))
            self.assertEqual(PAYOUTS[record["payout_index"]], record["payout"])
            self.assertEqual(record["slot_index"], record["payout_index"])
            self.assertIn(record["payout"], PAYOUTS)
            self.assertIsInstance(record["observed"], bool)
            self.assertEqual(record["feature_mask"], 1 if record["observed"] else 0)
            self.assertNotIn("signal", record)
            self.assertTrue(_simplex(truth["probabilities"]))
            self.assertTrue(_simplex(truth["observable_probabilities"]))
            self.assertEqual(truth["probability_kind"], "latent_oracle")
            self.assertEqual(truth["observable_probability_kind"], "observable_history")
            seen_ids.add(record["machine_id"])
        self.assertEqual(seen_ids, set(range(8)))

    def test_sampler_respects_slot_order(self):
        probabilities = [0.50, 0.20, 0.15, 0.10, 0.05]
        counts = Counter(
            gd.sample_payout(seed=0, episode_index=0, machine_id=0, tick=tick, probabilities=probabilities)
            for tick in range(4000)
        )
        self.assertEqual(set(counts), set(PAYOUTS))
        for payout, probability in zip(PAYOUTS, probabilities):
            frequency = counts[payout] / 4000
            self.assertLess(abs(frequency - probability), 0.04)

    def test_reproducible_and_seed_sensitive(self):
        first = gd.generate(seed=0, n_episodes=1, n_ticks=16, n_machines=4)
        second = gd.generate(seed=0, n_episodes=1, n_ticks=16, n_machines=4)
        other = gd.generate(seed=1, n_episodes=1, n_ticks=16, n_machines=4)
        self.assertEqual(first["records"], second["records"])
        self.assertEqual(first["truth"], second["truth"])
        self.assertNotEqual(first["records"], other["records"])
        latent_moved = any(
            _max_abs_diff(left["probabilities"], right["probabilities"]) > 1e-9
            for left, right in zip(first["truth"], other["truth"])
        )
        self.assertTrue(latent_moved)

    def test_known_truth_matches_independent_replay(self):
        seed = 3
        bundle = gd.generate(seed=seed, n_episodes=2, n_ticks=20, n_machines=4)
        expected = _replay_latent(bundle, seed)
        self.assertEqual(len(expected), len(bundle["truth"]))
        for row in bundle["truth"]:
            got = row["probabilities"]
            replayed = expected[(row["episode_id"], row["tick"], row["machine_id"])]
            self.assertLess(_max_abs_diff(got, replayed), 1e-12)

    def test_latent_oracle_differs_from_observable_history(self):
        bundle = gd.generate(seed=0, n_episodes=1, n_ticks=30, n_machines=4)
        gaps = 0
        distinguished = 0
        for record, truth in zip(bundle["records"], bundle["truth"]):
            if not record["observed"]:
                gaps += 1
            if _max_abs_diff(truth["probabilities"], truth["observable_probabilities"]) > 1e-3:
                distinguished += 1
        self.assertGreater(gaps, 0)
        self.assertGreater(distinguished, 0)
        # Tick 0 has no payout history. The oracle still has a hidden phase;
        # the history filter is only the type prior.
        first = bundle["truth"][0]
        prior = gd.observable_history_probabilities(first["machine_type"], [])
        self.assertLess(_max_abs_diff(first["observable_probabilities"], prior), 1e-12)
        self.assertGreater(_max_abs_diff(first["probabilities"], prior), 1e-3)

    def test_future_mutation_leaves_prefix_and_changes_later_odds(self):
        seed = 0
        n_ticks = 18
        n_machines = 3
        base = gd.generate(seed=seed, n_episodes=2, n_ticks=n_ticks, n_machines=n_machines)
        episode_id = "gd-e000"
        other_episode = "gd-e001"
        target = next(
            row
            for row in base["records"]
            if row["episode_id"] == episode_id and row["tick"] == 8 and row["machine_id"] == 1
        )
        replacement = 20 if target["payout"] != 20 else 0
        mutated = gd.generate(
            seed=seed,
            n_episodes=2,
            n_ticks=n_ticks,
            n_machines=n_machines,
            interventions={(episode_id, 1, 8): replacement},
        )
        base_records = _index_rows(base["records"])
        new_records = _index_rows(mutated["records"])
        base_truth = _index_rows(base["truth"])
        new_truth = _index_rows(mutated["truth"])

        for tick in range(8):
            for machine_id in range(n_machines):
                key = (episode_id, tick, machine_id)
                self.assertEqual(base_records[key], new_records[key])
                self.assertLess(
                    _max_abs_diff(base_truth[key]["probabilities"], new_truth[key]["probabilities"]),
                    1e-12,
                )
        # The spin at the intervention tick does not change anyone's odds at that same tick.
        for machine_id in range(n_machines):
            key = (episode_id, 8, machine_id)
            self.assertLess(
                _max_abs_diff(base_truth[key]["probabilities"], new_truth[key]["probabilities"]),
                1e-12,
            )
            self.assertLess(
                _max_abs_diff(
                    base_truth[key]["observable_probabilities"],
                    new_truth[key]["observable_probabilities"],
                ),
                1e-12,
            )
        self.assertEqual(new_records[(episode_id, 8, 1)]["payout"], replacement)
        self.assertNotEqual(base_records[(episode_id, 8, 1)]["payout"], replacement)
        for machine_id in range(n_machines):
            if machine_id == 1:
                continue
            self.assertEqual(base_records[(episode_id, 8, machine_id)], new_records[(episode_id, 8, machine_id)])

        later_moved = _max_abs_diff(
            base_truth[(episode_id, 9, 1)]["probabilities"],
            new_truth[(episode_id, 9, 1)]["probabilities"],
        )
        self.assertGreater(later_moved, 1e-4)

        # The other episode shares no state with the mutated episode.
        for row in base["records"]:
            if row["episode_id"] != other_episode:
                continue
            key = (row["episode_id"], row["tick"], row["machine_id"])
            self.assertEqual(base_records[key], new_records[key])
            self.assertLess(
                _max_abs_diff(base_truth[key]["probabilities"], new_truth[key]["probabilities"]),
                1e-12,
            )

    def test_gappy_missing_is_not_zero(self):
        seed = 4
        bundle = gd.generate(seed=seed, n_episodes=1, n_ticks=24, n_machines=4)
        records = bundle["records"]
        truth = _index_rows(bundle["truth"])
        missing_nonzero = [
            row for row in records if (not row["observed"]) and row["payout"] != 0 and row["tick"] < 23
        ]
        self.assertGreater(len(missing_nonzero), 0)
        gap = missing_nonzero[0]
        later_tick = gap["tick"] + 1
        key = (gap["episode_id"], later_tick, gap["machine_id"])
        kind = gap["machine_type"]

        omitted = gd.history_before(records, gap["episode_id"], gap["machine_id"], later_tick)
        zero_filled = []
        for row in records:
            if row["episode_id"] != gap["episode_id"] or row["machine_id"] != gap["machine_id"]:
                continue
            if row["tick"] >= later_tick:
                continue
            if row["observed"]:
                zero_filled.append(row["payout"])
            else:
                zero_filled.append(0)
        proper = gd.observable_history_probabilities(kind, omitted)
        imputed = gd.observable_history_probabilities(kind, zero_filled)
        self.assertLess(_max_abs_diff(truth[key]["observable_probabilities"], proper), 1e-12)
        self.assertGreater(_max_abs_diff(proper, imputed), 1e-6)
        # Mask 0 must not be implemented as payout * mask, which is the zero imputation above.
        self.assertEqual(gap["feature_mask"], 0)

        wrong_latent = _replay_latent_missing_as_zero(bundle, seed)
        matched_wrong = 0
        diverged_wrong = 0
        for row in bundle["truth"]:
            distance = _max_abs_diff(row["probabilities"], wrong_latent[(row["episode_id"], row["tick"], row["machine_id"])])
            if distance < 1e-12:
                matched_wrong += 1
            if distance > 1e-6:
                diverged_wrong += 1
        self.assertGreater(matched_wrong, 0)
        self.assertGreater(diverged_wrong, 0)

        # An unobserved true payout changes later oracle odds and leaves the history filter unchanged.
        hidden = gap
        forced = 20 if hidden["payout"] != 20 else 5
        mutated = gd.generate(
            seed=seed,
            n_episodes=1,
            n_ticks=24,
            n_machines=4,
            interventions={(hidden["episode_id"], hidden["machine_id"], hidden["tick"]): forced},
        )
        new_truth = _index_rows(mutated["truth"])
        self.assertGreater(
            _max_abs_diff(truth[key]["probabilities"], new_truth[key]["probabilities"]),
            1e-4,
        )
        self.assertLess(
            _max_abs_diff(
                truth[key]["observable_probabilities"],
                new_truth[key]["observable_probabilities"],
            ),
            1e-12,
        )

    def test_query_cutoff_excludes_label_and_does_not_invent_signal(self):
        bundle = gd.generate(seed=2, n_episodes=1, n_ticks=12, n_machines=2)
        self.assertEqual(len(bundle["queries"]), 2)
        cutoffs = {query["cutoff"] for query in bundle["queries"]}
        self.assertEqual(cutoffs, {11})
        for query in bundle["queries"]:
            self.assertIsNone(query["signal"])
            self.assertEqual(query["signal_source"], "not_synthesized")
            self.assertEqual(tuple(gd.QUERY_COLUMNS), (
                "query_id",
                "episode_id",
                "machine_id",
                "machine_type",
                "signal",
                "cutoff",
            ))
            cutoff = query["cutoff"]
            kind = query["machine_type"]
            usable = gd.history_before(bundle["records"], query["episode_id"], query["machine_id"], cutoff)
            label = next(
                row
                for row in bundle["records"]
                if row["episode_id"] == query["episode_id"]
                and row["machine_id"] == query["machine_id"]
                and row["tick"] == cutoff
            )
            leaked = usable + [label["payout"]]
            truth = next(
                row
                for row in bundle["truth"]
                if row["episode_id"] == query["episode_id"]
                and row["machine_id"] == query["machine_id"]
                and row["tick"] == cutoff
            )
            self.assertLess(
                _max_abs_diff(
                    truth["observable_probabilities"],
                    gd.observable_history_probabilities(kind, usable),
                ),
                1e-12,
            )
            self.assertGreater(
                _max_abs_diff(
                    gd.observable_history_probabilities(kind, usable),
                    gd.observable_history_probabilities(kind, leaked),
                ),
                1e-8,
            )
            self.assertGreater(
                _max_abs_diff(truth["probabilities"], truth["observable_probabilities"]),
                1e-3,
            )

    def test_cli_writes_research_only_manifest_and_refuses_huge_runs(self):
        with self.assertRaises(ValueError):
            gd.generate(seed=0, n_episodes=8, n_ticks=200, n_machines=8)
        with self.assertRaises(ValueError):
            gd.generate(seed=0, n_episodes=1, n_ticks=4, n_machines=9)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = gd.write_run(tmp, seed=7, n_episodes=1, n_ticks=8, n_machines=2)
            self.assertEqual(manifest["eligibility"], "RESEARCH_ONLY")
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["n_records"], 16)
            out = Path(tmp)
            for name in ("records.jsonl", "truth.jsonl", "queries.jsonl", "manifest.json"):
                self.assertTrue((out / name).is_file())
            record_lines = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
            truth_lines = (out / "truth.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(record_lines), 16)
            self.assertEqual(len(truth_lines), 16)
            self.assertEqual(len(json.loads(truth_lines[0])["probabilities"]), 5)
            self.assertEqual(gd._sha256(out / "records.jsonl"), manifest["output_hashes"]["records.jsonl"])
            self.assertIn("generate.py", manifest["source_hashes"])


if __name__ == "__main__":
    unittest.main()
