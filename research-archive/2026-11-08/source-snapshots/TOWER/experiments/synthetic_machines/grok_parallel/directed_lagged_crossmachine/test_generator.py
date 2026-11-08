"""Behavioral tests for the directed lagged cross-machine toy generator.

Run from this directory:

    /home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator -v
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import generate
from generate import (
    BASE_COUNTS,
    MAX_RECORDS,
    MISSING,
    PAYOUT_TO_INDEX,
    PAYOUTS,
    QUERY_SIGNAL,
    TYPE_TO_INDEX,
    as_feature_row,
    conditional_probabilities,
    directed_edges,
    episode_rng,
    generate as generate_panel,
    observed_flag,
    pack_probs,
    parent_slot,
)

# Literals. Not enumerate(MACHINE_TYPES) and not a slot number.
PUBLIC_TYPE_TO_INDEX = {"balanced": 0, "jackpot": 1, "steady": 2}
PUBLIC_INDEX_TO_TYPE = {0: "balanced", 1: "jackpot", 2: "steady"}

SPEC_BASE = {
    "steady": (10, 5, 3, 1, 1),
    "balanced": (4, 5, 6, 3, 2),
    "jackpot": (6, 3, 2, 3, 6),
}
SPEC_PAYOUT_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
ROOT = Path(__file__).resolve().parent


def spec_fractions(machine_type: str, parent_payout: int | None) -> tuple[Fraction, ...]:
    """Independent 7/10 base + 3/10 parent atom. Not a call into the kernel helper."""
    counts = SPEC_BASE[machine_type]
    base = tuple(Fraction(count, 20) for count in counts)
    if parent_payout is None:
        return base
    index = SPEC_PAYOUT_INDEX[parent_payout]
    return tuple(
        Fraction(7, 10) * base[j] + (Fraction(3, 10) if j == index else Fraction(0))
        for j in range(5)
    )


def spec_from_belief(machine_type: str, belief: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    counts = SPEC_BASE[machine_type]
    base = tuple(Fraction(count, 20) for count in counts)
    return tuple(Fraction(7, 10) * base[j] + Fraction(3, 10) * belief[j] for j in range(5))


def make_row(tick, machine_id, machine_type, payout, observed, episode_id="ep"):
    payout_index = SPEC_PAYOUT_INDEX[payout]
    return {
        "episode_id": episode_id,
        "tick": tick,
        "machine_id": machine_id,
        "slot_index": machine_id,
        "machine_type": machine_type,
        "payout": payout,
        "payout_index": payout_index,
        "observed": observed,
        "feature_payout_index": payout_index if observed else MISSING,
    }


def two_machine_panel(parent_payout, parent_observed, drop_parent_row=False):
    """Machine 0 (steady) is the lag parent of machine 1 (jackpot)."""
    rows = [make_row(0, 1, "jackpot", 0, True), make_row(1, 1, "jackpot", 2, True)]
    if drop_parent_row:
        rows.append(make_row(1, 0, "steady", 1, True))
    else:
        rows.append(make_row(0, 0, "steady", parent_payout, parent_observed))
        rows.append(make_row(1, 0, "steady", 1, True))
    return rows


def index_records(records):
    return {(row["episode_id"], row["tick"], row["machine_id"]): row for row in records}


class DirectedLaggedTests(unittest.TestCase):
    def test_mappings_simplex_and_alignment(self):
        self.assertEqual(list(PAYOUTS), [0, 1, 2, 5, 20])
        self.assertEqual(PAYOUT_TO_INDEX, SPEC_PAYOUT_INDEX)
        self.assertEqual(TYPE_TO_INDEX, PUBLIC_TYPE_TO_INDEX)
        self.assertEqual(TYPE_TO_INDEX["balanced"], 0)
        self.assertEqual(TYPE_TO_INDEX["jackpot"], 1)
        self.assertEqual(TYPE_TO_INDEX["steady"], 2)
        self.assertEqual(BASE_COUNTS, SPEC_BASE)
        self.assertEqual(MISSING, -1)
        self.assertNotEqual(MISSING, SPEC_PAYOUT_INDEX[0])
        self.assertEqual(directed_edges(8), [(7, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])

        panel = generate_panel(seed=0, n_episodes=2, n_ticks=8, n_machines=8)
        records, truth = panel["records"], panel["truth"]
        self.assertEqual(len(records), 128)
        self.assertEqual(len(truth), 128)
        gaps = []
        for record, row in zip(records, truth):
            self.assertEqual(record["episode_id"], row["episode_id"])
            self.assertEqual(record["tick"], row["tick"])
            self.assertEqual(record["machine_id"], row["machine_id"])
            self.assertEqual(record["slot_index"], record["machine_id"])
            self.assertEqual(row["slot_index"], record["machine_id"])
            self.assertIn(record["machine_type"], SPEC_BASE)
            self.assertEqual(record["machine_type_index"], PUBLIC_TYPE_TO_INDEX[record["machine_type"]])
            self.assertEqual(record["payout_index"], SPEC_PAYOUT_INDEX[record["payout"]])
            self.assertIs(record["payout_is_feature"], record["observed"])
            feature = as_feature_row(record)
            self.assertNotIn("payout", feature)
            self.assertNotIn("payout_index", feature)
            if record["observed"]:
                self.assertEqual(feature["feature_payout_index"], record["payout_index"])
            else:
                self.assertEqual(feature["feature_payout_index"], MISSING)
            self.assertEqual(row["prob_kind"], "latent_oracle")
            self.assertEqual(len(row["probabilities"]), 5)
            numerators = row["probability_numerator"]
            denominator = row["probability_denominator"]
            self.assertEqual(len(numerators), 5)
            self.assertGreater(denominator, 0)
            self.assertEqual(sum(numerators), denominator)
            self.assertTrue(all(part >= 0 for part in numerators))
            self.assertAlmostEqual(sum(row["probabilities"]), 1.0, places=9)
            for value, numerator in zip(row["probabilities"], numerators):
                self.assertAlmostEqual(value, numerator / denominator, places=12)
            if record["observed"]:
                self.assertEqual(record["feature_payout_index"], record["payout_index"])
            else:
                self.assertEqual(record["feature_payout_index"], MISSING)
                gaps.append(record)
            self.assertNotIn("signal", record)
        self.assertTrue(gaps)
        self.assertTrue(any(row["payout"] != 0 for row in gaps))
        self.assertFalse(any(row["tick"] >= panel["queries"][0]["cutoff"] for row in records))

    def test_reproducible_and_seed_changes_mask(self):
        kwargs = {"n_episodes": 2, "n_ticks": 5, "n_machines": 8}
        self.assertEqual(generate_panel(seed=7, **kwargs), generate_panel(seed=7, **kwargs))
        first = generate_panel(seed=0, n_episodes=1, n_ticks=4, n_machines=4)
        second = generate_panel(seed=1, n_episodes=1, n_ticks=4, n_machines=4)
        self.assertNotEqual(first["records"], second["records"])
        self.assertFalse(observed_flag(0, 0, 0, 0))
        self.assertTrue(observed_flag(1, 0, 0, 0))
        self.assertFalse(first["records"][0]["observed"])
        self.assertTrue(second["records"][0]["observed"])
        signature = inspect.signature(generate_panel)
        self.assertEqual(signature.parameters["seed"].default, 0)
        self.assertEqual(signature.parameters["n_episodes"].default, 8)
        self.assertEqual(signature.parameters["n_ticks"].default, 96)
        self.assertEqual(signature.parameters["n_machines"].default, 8)

    def test_rng_stream_and_known_truth_kernel(self):
        seed, n_episodes, n_ticks, n_machines = 4, 2, 6, 8
        panel = generate_panel(
            seed=seed, n_episodes=n_episodes, n_ticks=n_ticks, n_machines=n_machines
        )
        by_record = index_records(panel["records"])
        for episode_index in range(n_episodes):
            rng = episode_rng(seed, episode_index)
            types = [PUBLIC_INDEX_TO_TYPE[rng.randrange(3)] for _ in range(n_machines)]
            previous = None
            for tick in range(n_ticks):
                drawn = []
                for machine_id in range(n_machines):
                    parent_payout = None if previous is None else previous[parent_slot(machine_id, n_machines)]
                    numerators, denominator = pack_probs(spec_fractions(types[machine_id], parent_payout))
                    draw = rng.randrange(denominator)
                    cumulative = 0
                    payout = None
                    for index, weight in enumerate(numerators):
                        cumulative += weight
                        if draw < cumulative:
                            payout = (0, 1, 2, 5, 20)[index]
                            break
                    record = by_record[(f"ep{episode_index:04d}", tick, machine_id)]
                    self.assertEqual(record["machine_type"], types[machine_id])
                    self.assertEqual(record["payout"], payout)
                    self.assertIs(record["observed"], observed_flag(seed, episode_index, tick, machine_id))
                    drawn.append(payout)
                previous = drawn

        annotated = conditional_probabilities(panel["records"], predict_cutoff=n_ticks)
        for record, row in zip(panel["records"], panel["truth"]):
            if record["tick"] == 0:
                parent_payout = None
                parent_observed = None
            else:
                parent = by_record[(record["episode_id"], record["tick"] - 1, parent_slot(record["machine_id"], n_machines))]
                parent_payout = parent["payout"]
                parent_observed = parent["observed"]
            expected = spec_fractions(record["machine_type"], parent_payout)
            numerators, denominator = pack_probs(expected)
            self.assertEqual(row["probability_numerator"], list(numerators))
            self.assertEqual(row["probability_denominator"], denominator)
            key = (record["episode_id"], record["tick"], record["machine_id"])
            self.assertEqual(annotated["latent_oracle"][key], expected)
            observable = annotated["observable_history"][key]
            if record["tick"] == 0 or parent_observed:
                self.assertEqual(observable, expected)
            else:
                self.assertNotEqual(observable, expected)

        for query in panel["queries"]:
            self.assertEqual(query["cutoff"], n_ticks)
            self.assertEqual(query["signal"], 0)
            self.assertEqual(query["signal_scope"], "prediction_cutoff_only")
            parent = by_record[(query["episode_id"], n_ticks - 1, parent_slot(query["machine_id"], n_machines))]
            expected = spec_fractions(query["machine_type"], parent["payout"])
            numerators, denominator = pack_probs(expected)
            self.assertEqual(query["latent_oracle_numerator"], list(numerators))
            self.assertEqual(query["latent_oracle_denominator"], denominator)
            if parent["observed"]:
                self.assertEqual(query["observable_history_numerator"], list(numerators))
            else:
                self.assertNotEqual(query["observable_history_numerator"], list(numerators))

    def test_future_mutation_prefix_and_directed_lag(self):
        panel = generate_panel(seed=2, n_episodes=2, n_ticks=8, n_machines=8)
        original = panel["records"]
        before = conditional_probabilities(original, predict_cutoff=8)

        mutated = copy.deepcopy(original)
        for row in mutated:
            if row["episode_id"] == "ep0000" and row["tick"] >= 4:
                payout = 0 if row["payout"] != 0 else 20
                row["payout"] = payout
                row["payout_index"] = SPEC_PAYOUT_INDEX[payout]
                row["observed"] = not row["observed"]
                row["feature_payout_index"] = row["payout_index"] if row["observed"] else MISSING
        after = conditional_probabilities(mutated, predict_cutoff=8)
        mutated_by = index_records(mutated)
        original_by = index_records(original)

        for tick in range(8):
            for machine_id in range(8):
                key = ("ep0001", tick, machine_id)
                self.assertEqual(before["latent_oracle"][key], after["latent_oracle"][key])
                self.assertEqual(before["observable_history"][key], after["observable_history"][key])
        for tick in range(5):
            for machine_id in range(8):
                key = ("ep0000", tick, machine_id)
                self.assertEqual(before["latent_oracle"][key], after["latent_oracle"][key])
                self.assertEqual(before["observable_history"][key], after["observable_history"][key])
                self.assertEqual(
                    before["latent_oracle"][key],
                    spec_fractions(
                        original_by[key]["machine_type"],
                        None if tick == 0 else original_by[("ep0000", tick - 1, parent_slot(machine_id, 8))]["payout"],
                    ),
                )
        changed = 0
        for machine_id in range(8):
            key = ("ep0000", 5, machine_id)
            self.assertNotEqual(before["latent_oracle"][key], after["latent_oracle"][key])
            parent_id = parent_slot(machine_id, 8)
            self.assertEqual(
                after["latent_oracle"][key],
                spec_fractions(mutated_by[key]["machine_type"], mutated_by[("ep0000", 4, parent_id)]["payout"]),
            )
            self.assertEqual(
                before["latent_oracle"][key],
                spec_fractions(original_by[key]["machine_type"], original_by[("ep0000", 4, parent_id)]["payout"]),
            )
            changed += 1
        self.assertEqual(changed, 8)

        # Same-tick sibling is not a parent. Previous-tick parent is.
        sibling = copy.deepcopy(original)
        for row in sibling:
            if row["episode_id"] == "ep0000" and row["tick"] == 5 and row["machine_id"] == 3:
                row["payout"] = 0 if row["payout"] != 0 else 20
        sibling_probs = conditional_probabilities(sibling, predict_cutoff=8)
        target = ("ep0000", 5, 0)
        child_of_sibling = ("ep0000", 5, 4)
        self.assertEqual(before["latent_oracle"][target], sibling_probs["latent_oracle"][target])
        self.assertEqual(before["latent_oracle"][child_of_sibling], sibling_probs["latent_oracle"][child_of_sibling])

        lagged_other = copy.deepcopy(original)
        for row in lagged_other:
            if row["episode_id"] == "ep0000" and row["tick"] == 4 and row["machine_id"] == 3:
                row["payout"] = 0 if row["payout"] != 0 else 20
        lagged_probs = conditional_probabilities(lagged_other, predict_cutoff=8)
        self.assertEqual(before["latent_oracle"][target], lagged_probs["latent_oracle"][target])
        self.assertNotEqual(before["latent_oracle"][("ep0000", 5, 4)], lagged_probs["latent_oracle"][("ep0000", 5, 4)])

        lagged_parent = copy.deepcopy(original)
        for row in lagged_parent:
            if row["episode_id"] == "ep0000" and row["tick"] == 4 and row["machine_id"] == 7:
                row["payout"] = 0 if row["payout"] != 0 else 20
        parent_probs = conditional_probabilities(lagged_parent, predict_cutoff=8)
        self.assertNotEqual(before["latent_oracle"][target], parent_probs["latent_oracle"][target])
        self.assertEqual(
            parent_probs["latent_oracle"][target],
            spec_fractions(
                original_by[target]["machine_type"],
                0 if original_by[("ep0000", 4, 7)]["payout"] != 0 else 20,
            ),
        )

        prefix = [row for row in original if row["tick"] < 4]
        prefix_probs = conditional_probabilities(prefix, predict_cutoff=4)
        junk = copy.deepcopy(prefix)
        for tick in (4, 6):
            for machine_id in range(8):
                donor = original_by[("ep0000", 0, machine_id)]
                junk.append(make_row(tick, machine_id, donor["machine_type"], 20, True, "ep0000"))
                donor_b = original_by[("ep0001", 0, machine_id)]
                junk.append(make_row(tick, machine_id, donor_b["machine_type"], 20, True, "ep0001"))
        junk_probs = conditional_probabilities(junk, predict_cutoff=4)
        for episode_id in ("ep0000", "ep0001"):
            for tick in range(5):
                for machine_id in range(8):
                    key = (episode_id, tick, machine_id)
                    self.assertEqual(prefix_probs["latent_oracle"][key], junk_probs["latent_oracle"][key])
                    self.assertEqual(prefix_probs["observable_history"][key], junk_probs["observable_history"][key])

    def test_missing_is_not_zero_and_oracle_uses_latent(self):
        hidden_20 = two_machine_panel(20, False)
        hidden_0 = two_machine_panel(0, False)
        seen_0 = two_machine_panel(0, True)
        seen_20 = two_machine_panel(20, True)
        absent = two_machine_panel(20, False, drop_parent_row=True)

        hidden_20_p = conditional_probabilities(hidden_20)
        hidden_0_p = conditional_probabilities(hidden_0)
        seen_0_p = conditional_probabilities(seen_0)
        seen_20_p = conditional_probabilities(seen_20)
        absent_p = conditional_probabilities(absent)
        key = ("ep", 1, 1)

        steady = tuple(Fraction(count, 20) for count in SPEC_BASE["steady"])
        gap_belief = spec_from_belief("jackpot", steady)
        self.assertEqual(hidden_20_p["observable_history"][key], gap_belief)
        self.assertEqual(hidden_0_p["observable_history"][key], gap_belief)
        self.assertEqual(absent_p["observable_history"][key], gap_belief)
        self.assertNotEqual(gap_belief, spec_fractions("jackpot", 0))
        self.assertNotEqual(gap_belief, spec_fractions("jackpot", 20))
        self.assertEqual(hidden_20_p["latent_oracle"][key], spec_fractions("jackpot", 20))
        self.assertEqual(hidden_0_p["latent_oracle"][key], spec_fractions("jackpot", 0))
        self.assertEqual(seen_0_p["observable_history"][key], spec_fractions("jackpot", 0))
        self.assertEqual(seen_20_p["latent_oracle"][key], spec_fractions("jackpot", 20))
        self.assertEqual(seen_20_p["observable_history"][key], seen_20_p["latent_oracle"][key])
        self.assertIsNone(absent_p["latent_oracle"][key])
        self.assertNotEqual(hidden_20_p["observable_history"][key], seen_0_p["observable_history"][key])

        # The feature channel is not an input. Flipping it, or stamping a historical signal, changes nothing.
        polluted = copy.deepcopy(hidden_20)
        for row in polluted:
            row["feature_payout_index"] = 0
            row["signal"] = -1 if row["tick"] else 1
        polluted_p = conditional_probabilities(polluted)
        self.assertEqual(polluted_p["latent_oracle"][key], hidden_20_p["latent_oracle"][key])
        self.assertEqual(polluted_p["observable_history"][key], hidden_20_p["observable_history"][key])

        unobserved_parent = next(row for row in hidden_20 if row["tick"] == 0 and row["machine_id"] == 0)
        self.assertFalse(unobserved_parent["observed"])
        self.assertEqual(unobserved_parent["payout"], 20)
        self.assertEqual(unobserved_parent["feature_payout_index"], MISSING)

    def test_horizon_prefix_holds_for_every_episode(self):
        """A longer horizon must not move later episodes.

        One global RNG fails this: extra ticks in episode 0 are consumed before
        episode 1 is drawn, so episode 1's prefix changes. Independent streams
        keep every episode's ticks < horizon identical, and tick `horizon`
        still depends only on that episode's lag parent.
        """
        horizon, extra, n_episodes, n_machines = 5, 4, 3, 6
        short = generate_panel(seed=11, n_episodes=n_episodes, n_ticks=horizon, n_machines=n_machines)
        long = generate_panel(
            seed=11, n_episodes=n_episodes, n_ticks=horizon + extra, n_machines=n_machines
        )
        short_rows = index_records(short["records"])
        long_rows = index_records(long["records"])
        short_truth = index_records(short["truth"])
        long_truth = index_records(long["truth"])
        self.assertEqual(set(short_rows), set(short_truth))
        for key, row in short_rows.items():
            self.assertEqual(row, long_rows[key])
            self.assertEqual(short_truth[key], long_truth[key])
            self.assertLess(key[1], horizon)
        self.assertTrue(any(key[1] >= horizon for key in long_rows))
        # Later episodes are real streams, not a copy of episode 0.
        episode_payouts = []
        for episode_index in range(n_episodes):
            episode_id = f"ep{episode_index:04d}"
            episode_payouts.append(
                [row["payout"] for row in short["records"] if row["episode_id"] == episode_id]
            )
            for machine_id in range(n_machines):
                parent = short_rows[(episode_id, horizon - 1, parent_slot(machine_id, n_machines))]
                child = long_rows[(episode_id, horizon, machine_id)]
                expected = spec_fractions(child["machine_type"], parent["payout"])
                numerators, denominator = pack_probs(expected)
                self.assertEqual(long_truth[(episode_id, horizon, machine_id)]["probability_numerator"], list(numerators))
                self.assertEqual(long_truth[(episode_id, horizon, machine_id)]["probability_denominator"], denominator)
        self.assertEqual(len({tuple(payouts) for payouts in episode_payouts}), n_episodes)

        mutated = copy.deepcopy(long["records"])
        for row in mutated:
            if row["tick"] >= horizon:
                payout = 0 if row["payout"] != 0 else 20
                row["payout"] = payout
                row["payout_index"] = SPEC_PAYOUT_INDEX[payout]
                row["observed"] = not row["observed"]
                row["payout_is_feature"] = row["observed"]
                row["feature_payout_index"] = row["payout_index"] if row["observed"] else MISSING
        before = conditional_probabilities(long["records"])
        after = conditional_probabilities(mutated)
        for episode_index in range(n_episodes):
            episode_id = f"ep{episode_index:04d}"
            for tick in range(horizon + 1):
                for machine_id in range(n_machines):
                    key = (episode_id, tick, machine_id)
                    self.assertEqual(before["latent_oracle"][key], after["latent_oracle"][key])
                    self.assertEqual(before["observable_history"][key], after["observable_history"][key])

    def test_unobserved_payout_is_never_a_feature(self):
        hidden_high = make_row(0, 0, "steady", 20, False)
        hidden_zero = make_row(0, 0, "steady", 0, False)
        seen_high = make_row(0, 0, "steady", 20, True)
        self.assertEqual(as_feature_row(hidden_high), as_feature_row(hidden_zero))
        self.assertNotEqual(hidden_high["payout"], hidden_zero["payout"])
        self.assertNotIn("payout", as_feature_row(hidden_high))
        self.assertEqual(as_feature_row(hidden_high)["feature_payout_index"], MISSING)
        self.assertEqual(as_feature_row(seen_high)["feature_payout_index"], SPEC_PAYOUT_INDEX[20])
        self.assertNotEqual(as_feature_row(hidden_high), as_feature_row(seen_high))

    def test_record_cap_and_default_shape(self):
        with self.assertRaises(ValueError):
            generate_panel(n_episodes=0, n_ticks=1, n_machines=1)
        with self.assertRaises(ValueError):
            generate_panel(n_episodes=1, n_ticks=8193, n_machines=1)
        edge = generate_panel(seed=0, n_episodes=1, n_ticks=8192, n_machines=1)
        self.assertEqual(len(edge["records"]), MAX_RECORDS)
        opening = spec_fractions(edge["records"][0]["machine_type"], None)
        numerators, denominator = pack_probs(opening)
        self.assertEqual(edge["truth"][0]["probability_numerator"], list(numerators))
        self.assertEqual(edge["truth"][0]["probability_denominator"], denominator)

        panel = generate_panel()
        self.assertEqual(len(panel["records"]), 8 * 96 * 8)
        self.assertLessEqual(len(panel["records"]), MAX_RECORDS)
        self.assertEqual(len(panel["truth"]), len(panel["records"]))
        self.assertEqual(len(panel["queries"]), 64)
        machines = {row["machine_id"] for row in panel["records"]}
        self.assertEqual(machines, set(range(8)))
        self.assertTrue(all(row["prob_kind"] == "latent_oracle" for row in panel["truth"]))
        self.assertTrue(all(query["signal"] == QUERY_SIGNAL and query["cutoff"] == 96 for query in panel["queries"]))
        self.assertTrue(all("signal" not in row for row in panel["records"]))

    def test_cli_hashes_and_research_only(self):
        python = sys.executable
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            command = [
                python,
                str(ROOT / "generate.py"),
                "--seed",
                "3",
                "--episodes",
                "1",
                "--ticks",
                "4",
                "--machines",
                "4",
                "--out",
                str(out),
            ]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0)
            snapshot = {name: (out / name).read_bytes() for name in ("records.jsonl", "truth.jsonl", "queries.jsonl", "manifest.json")}
            subprocess.run(command, check=True, capture_output=True, text=True)
            for name, payload in snapshot.items():
                self.assertEqual((out / name).read_bytes(), payload)

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["family"], "directed_lagged_crossmachine")
            self.assertEqual(manifest["status"], "RESEARCH_ONLY")
            self.assertTrue(manifest["research_only"])
            self.assertTrue(manifest["real_only_fallback"])
            self.assertTrue(manifest["do_not_train"])
            self.assertEqual(manifest["seed"], 3)
            self.assertEqual(manifest["settings"]["record_count"], 16)
            self.assertEqual(manifest["settings"]["cutoff"], 4)
            for name, payload in snapshot.items():
                if name == "manifest.json":
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                self.assertEqual(manifest["output_hashes"][name], digest)
            self.assertEqual(
                manifest["source_hashes"]["generate.py"],
                hashlib.sha256((ROOT / "generate.py").read_bytes()).hexdigest(),
            )
            records = [json.loads(line) for line in snapshot["records.jsonl"].decode().splitlines()]
            truth = [json.loads(line) for line in snapshot["truth.jsonl"].decode().splitlines()]
            queries = [json.loads(line) for line in snapshot["queries.jsonl"].decode().splitlines()]
            self.assertEqual(len(records), 16)
            self.assertTrue(all("signal" not in row for row in records))
            self.assertTrue(all(row["cutoff"] == 4 and row["signal"] == 0 for row in queries))
            self.assertTrue(all(sum(row["probability_numerator"]) == row["probability_denominator"] for row in truth))
            gaps = [row for row in records if not row["observed"]]
            self.assertTrue(gaps)
            self.assertTrue(all(row["feature_payout_index"] == MISSING for row in gaps))


if __name__ == "__main__":
    unittest.main()
