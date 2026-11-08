"""Meaningful checks for the finite-order Markov synthetic family.

The enumeration filter below is independent of generate.update_belief. It
recomputes observable-history probabilities by summing latent paths.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import generate as gen

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_INDEX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
TYPES = ("steady", "balanced", "jackpot")
LAGS = (None, 0, 1, 2, 5, 20)


def independent_oracle(machine_type, lag2, lag1):
    """Contract copy of the disclosed integer kernel. Kept separate on purpose."""
    base = {
        "steady": [70, 22, 5, 2, 1],
        "balanced": [28, 26, 24, 16, 6],
        "jackpot": [72, 10, 6, 6, 6],
    }[machine_type]
    persist = {1: {"steady": 48, "balanced": 18, "jackpot": 10}, 2: {"steady": 16, "balanced": 6, "jackpot": 4}}
    jackpot_bonus = {
        1: {"steady": (2, 2), "balanced": (6, 8), "jackpot": (10, 36)},
        2: {"steady": (1, 1), "balanced": (3, 4), "jackpot": (6, 18)},
    }
    five_bonus = {1: {"steady": 6, "balanced": 8, "jackpot": 14}, 2: {"steady": 2, "balanced": 3, "jackpot": 6}}
    weights = base[:]

    def add(lag, which):
        if lag is None:
            return
        weights[PAYOUT_TO_INDEX[lag]] += persist[which][machine_type]
        if lag == 20:
            weights[3] += jackpot_bonus[which][machine_type][0]
            weights[4] += jackpot_bonus[which][machine_type][1]
        elif lag == 5:
            weights[3] += five_bonus[which][machine_type]

    add(lag1, 1)
    add(lag2, 2)
    total = float(sum(weights))
    return tuple(w / total for w in weights)


def enumerate_observable(machine_type, masked_past):
    """P(Y_next | mask) by enumerating latent paths. None stays marginalized."""
    acc = [0.0] * 5
    total = 0.0

    def rec(pos, mass, lag2, lag1):
        nonlocal total
        if mass == 0.0:
            return
        if pos == len(masked_past):
            probs = independent_oracle(machine_type, lag2, lag1)
            total += mass
            for k in range(5):
                acc[k] += mass * probs[k]
            return
        probs = independent_oracle(machine_type, lag2, lag1)
        seen = masked_past[pos]
        if seen is None:
            for k, pk in enumerate(probs):
                rec(pos + 1, mass * pk, lag1, PAYOUTS[k])
        else:
            k = PAYOUT_TO_INDEX[seen]
            rec(pos + 1, mass * probs[k], lag1, PAYOUTS[k])

    rec(0, 1.0, None, None)
    if total <= 0.0:
        raise AssertionError("enumeration lost all mass")
    return tuple(v / total for v in acc)


def assert_simplex(testcase, probs):
    testcase.assertEqual(len(probs), 5)
    testcase.assertTrue(all(isinstance(p, float) for p in probs))
    testcase.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
    testcase.assertTrue(math.isclose(sum(probs), 1.0, abs_tol=1e-12))


def close_tuple(left, right, tol=1e-12):
    return len(left) == len(right) and all(math.isclose(a, b, abs_tol=tol, rel_tol=0.0) for a, b in zip(left, right))


class KernelTests(unittest.TestCase):
    def test_payout_and_slot_mappings(self):
        self.assertEqual(gen.PAYOUTS, PAYOUTS)
        self.assertEqual(gen.PAYOUT_TO_INDEX, PAYOUT_TO_INDEX)
        self.assertEqual(gen.INDEX_TO_PAYOUT[0], 0)
        self.assertEqual(gen.INDEX_TO_PAYOUT[3], 5)
        self.assertEqual(gen.INDEX_TO_PAYOUT[4], 20)
        self.assertEqual([gen.INDEX_TO_PAYOUT[i] for i in range(5)], list(PAYOUTS))
        self.assertEqual([gen.machine_type_of(i) for i in range(8)], list(gen.TYPE_BY_MACHINE))
        self.assertEqual(gen.TYPE_BY_MACHINE.count("steady"), 3)
        self.assertEqual(gen.TYPE_BY_MACHINE.count("balanced"), 3)
        self.assertEqual(gen.TYPE_BY_MACHINE.count("jackpot"), 2)
        # Index 0 is a real payout category. It is not the missing code.
        self.assertIsNone(gen.feature_payout({"observed": False, "payout": 20}))
        self.assertIsNone(gen.feature_payout({"observed": False, "payout": 0}))
        self.assertEqual(gen.feature_payout({"observed": True, "payout": 0}), 0)

    def test_simplex_and_independent_kernel(self):
        for machine_type in TYPES:
            for lag2 in LAGS:
                for lag1 in LAGS:
                    probs = gen.oracle_probabilities(machine_type, lag2, lag1)
                    assert_simplex(self, probs)
                    self.assertEqual(probs, independent_oracle(machine_type, lag2, lag1))
        # Order 2 is real: lag2 changes the row when lag1 is held fixed.
        self.assertNotEqual(
            gen.oracle_probabilities("jackpot", 0, 1),
            gen.oracle_probabilities("jackpot", 20, 1),
        )
        # Structural absence is not payout 0.
        self.assertNotEqual(
            gen.oracle_probabilities("steady", None, None),
            gen.oracle_probabilities("steady", 0, 0),
        )
        self.assertNotEqual(
            gen.oracle_probabilities("jackpot", 20, 0),
            gen.oracle_probabilities("jackpot", 20, 20),
        )

    def test_sample_bucket_mapping(self):
        class Fixed:
            def __init__(self, u):
                self.u = u

            def random(self):
                return self.u

        flat = (0.2, 0.2, 0.2, 0.2, 0.2)
        self.assertEqual(gen.sample_payout(flat, Fixed(0.0)), 0)
        self.assertEqual(gen.sample_payout(flat, Fixed(0.1)), 0)
        self.assertEqual(gen.sample_payout(flat, Fixed(0.3)), 1)
        self.assertEqual(gen.sample_payout(flat, Fixed(0.5)), 2)
        self.assertEqual(gen.sample_payout(flat, Fixed(0.7)), 5)
        self.assertEqual(gen.sample_payout(flat, Fixed(0.9)), 20)
        self.assertEqual(gen.sample_payout((0.0, 0.0, 0.0, 1.0, 0.0), Fixed(0.0)), 5)
        self.assertEqual(gen.sample_payout((0.0, 0.0, 0.0, 0.0, 1.0), Fixed(0.3)), 20)

    def test_eligibility_flags(self):
        self.assertEqual(gen.ELIGIBILITY, "RESEARCH_ONLY")
        self.assertFalse(gen.SYNTHETIC_EXCEPTION_VERIFIED)
        self.assertTrue(gen.REAL_ONLY_FALLBACK)
        self.assertEqual(gen.ORDER, 2)


class FilterTests(unittest.TestCase):
    def test_full_observation_matches_oracle_point_mass(self):
        latent = [20, 0, 5, 1]
        masked = list(latent)
        for t in range(len(latent) + 1):
            ora = gen.latent_oracle_from_payouts("jackpot", latent[:t])
            obp = gen.observable_probabilities_from_mask("jackpot", masked[:t])
            self.assertEqual(obp, ora)

    def test_gappy_missing_is_not_zero(self):
        # Latent jackpot then jackpot, but the second spin is unseen.
        gap_mask = [20, None]
        zero_mask = [20, 0]
        gap = gen.observable_probabilities_from_mask("jackpot", gap_mask)
        as_zero = gen.observable_probabilities_from_mask("jackpot", zero_mask)
        enumerated_gap = enumerate_observable("jackpot", gap_mask)
        enumerated_zero = enumerate_observable("jackpot", zero_mask)
        self.assertTrue(close_tuple(gap, enumerated_gap))
        self.assertTrue(close_tuple(as_zero, enumerated_zero))
        self.assertFalse(close_tuple(gap, as_zero, tol=1e-9))
        # Oracle knows the hidden 20. The observer does not, and must not impute 0.
        hidden = gen.latent_oracle_from_payouts("jackpot", [20, 20])
        imputed = gen.oracle_probabilities("jackpot", 20, 0)
        self.assertFalse(close_tuple(gap, hidden, tol=1e-9))
        self.assertFalse(close_tuple(gap, imputed, tol=1e-9))
        self.assertEqual(as_zero, imputed)
        self.assertIsNone(gen.feature_payout({"observed": False, "payout": 20}))

    def test_enumeration_matches_filter_on_mixed_mask(self):
        masked = [0, None, 5, None, 20]
        for machine_type in TYPES:
            got = gen.observable_probabilities_from_mask(machine_type, masked)
            exp = enumerate_observable(machine_type, masked)
            self.assertTrue(close_tuple(got, exp, tol=1e-9), msg=f"{machine_type} {got} vs {exp}")
            assert_simplex(self, got)

    def test_future_and_cross_machine_invariance(self):
        past = [1, 2, 5, 0, 20]
        base = [gen.latent_oracle_from_payouts("jackpot", past[:t]) for t in range(len(past) + 1)]
        edited = past[:]
        edited[4] = 0 if edited[4] != 0 else 1
        edited_probs = [gen.latent_oracle_from_payouts("jackpot", edited[:t]) for t in range(len(edited) + 1)]
        # Ticks whose lag window ends before the edit (edit at index 4 touches predictions from tick 5 on).
        for t in range(5):
            self.assertEqual(base[t], edited_probs[t])
        self.assertNotEqual(base[5], edited_probs[5])
        # The row at tick t ignores Y_t. Writing a different Y_t changes later rows only.
        self.assertEqual(
            gen.latent_oracle_from_payouts("jackpot", past[:4]),
            gen.latent_oracle_from_payouts("jackpot", edited[:4]),
        )
        self.assertNotEqual(
            gen.latent_oracle_from_payouts("steady", past),
            gen.latent_oracle_from_payouts("steady", [20, 20, 20, 20]),
        )
        # Observable prefix ignores a future mask/outcome edit.
        mask = [1, 2, None, 0]
        future_edit = [1, 2, None, 0, 20]
        self.assertEqual(
            gen.observable_probabilities_from_mask("balanced", mask),
            gen.observable_probabilities_from_mask("balanced", future_edit[:4]),
        )
        self.assertEqual(
            gen.observable_probabilities_from_mask("balanced", mask[:2]),
            gen.observable_probabilities_from_mask("balanced", [1, 2, 0, 20][:2]),
        )


class GenerateTests(unittest.TestCase):
    def _episode(self, data, episode_id):
        records = [r for r in data["records"] if r["episode_id"] == episode_id]
        truth = [r for r in data["truth"] if r["episode_id"] == episode_id]
        return records, truth

    def test_shape_simplex_and_known_truth(self):
        data = gen.generate(seed=0, n_episodes=2, n_ticks=12, n_machines=8)
        self.assertEqual(len(data["records"]), 2 * 12 * 8)
        self.assertEqual(len(data["truth"]), len(data["records"]))
        seen_missing = False
        seen_present = False
        by_machine = {}
        for rec, tru in zip(data["records"], data["truth"]):
            self.assertEqual(rec["episode_id"], tru["episode_id"])
            self.assertEqual(rec["tick"], tru["tick"])
            self.assertEqual(rec["machine_id"], tru["machine_id"])
            self.assertIn(rec["machine_type"], TYPES)
            self.assertEqual(rec["machine_type"], gen.machine_type_of(rec["machine_id"]))
            self.assertIn(rec["payout"], PAYOUTS)
            self.assertIsInstance(rec["observed"], bool)
            self.assertEqual(tru["probability_kind"], "latent_oracle")
            self.assertEqual(tru["observable_probability_kind"], "observable_history")
            assert_simplex(self, tru["probabilities"])
            assert_simplex(self, tru["observable_probabilities"])
            self.assertNotIn("signal", rec)
            self.assertNotIn("signal", tru)
            seen_missing = seen_missing or (not rec["observed"])
            seen_present = seen_present or rec["observed"]
            by_machine.setdefault((rec["episode_id"], rec["machine_id"]), []).append((rec, tru))
        self.assertTrue(seen_missing)
        self.assertTrue(seen_present)
        for rows in by_machine.values():
            latent = []
            masked = []
            for rec, tru in rows:
                ora = gen.latent_oracle_from_payouts(rec["machine_type"], latent)
                obp = gen.observable_probabilities_from_mask(rec["machine_type"], masked)
                self.assertEqual(tuple(tru["probabilities"]), ora)
                self.assertTrue(close_tuple(tru["observable_probabilities"], obp, tol=1e-12))
                self.assertEqual(tru["latent_lag1"], latent[-1] if latent else None)
                self.assertEqual(tru["latent_lag2"], latent[-2] if len(latent) >= 2 else None)
                # Current and other-machine outcomes are not in this prefix.
                latent.append(rec["payout"])
                masked.append(rec["payout"] if rec["observed"] else None)
                if not rec["observed"]:
                    self.assertIsNone(gen.feature_payout(rec))

    def test_reproducible_and_seed_and_prefix(self):
        first = gen.generate(seed=1, n_episodes=2, n_ticks=9, n_machines=4)
        random.seed(999)
        second = gen.generate(seed=1, n_episodes=2, n_ticks=9, n_machines=4)
        self.assertEqual(first, second)
        other = gen.generate(seed=2, n_episodes=2, n_ticks=9, n_machines=4)
        payouts = [r["payout"] for r in first["records"]]
        self.assertNotEqual(payouts, [r["payout"] for r in other["records"]])

        longer = gen.generate(seed=1, n_episodes=2, n_ticks=14, n_machines=4)
        short_ids = {(r["episode_id"], r["tick"], r["machine_id"]) for r in first["records"]}
        longer_prefix = [
            r for r in longer["records"] if (r["episode_id"], r["tick"], r["machine_id"]) in short_ids and r["tick"] < 9
        ]
        self.assertEqual(longer_prefix, first["records"])
        truth_prefix = [
            r for r in longer["truth"] if r["tick"] < 9 and r["episode_id"] in {x["episode_id"] for x in first["truth"]}
        ]
        self.assertEqual(truth_prefix, first["truth"])

        one_ep = gen.generate(seed=1, n_episodes=1, n_ticks=9, n_machines=4)
        first_ep = first["records"][0]["episode_id"]
        self.assertEqual(one_ep["records"], [r for r in first["records"] if r["episode_id"] == first_ep])
        self.assertEqual(one_ep["truth"], [r for r in first["truth"] if r["episode_id"] == first_ep])

    def test_future_mutation_keeps_generated_prefix(self):
        data = gen.generate(seed=3, n_episodes=1, n_ticks=8, n_machines=8)
        rows = {}
        for rec, tru in zip(data["records"], data["truth"]):
            rows.setdefault(rec["machine_id"], []).append((rec, tru))
        machine = 6
        self.assertEqual(rows[machine][0][0]["machine_type"], "jackpot")
        latent = [rec["payout"] for rec, _ in rows[machine]]
        original = [tru["probabilities"] for _, tru in rows[machine]]
        edited = latent[:]
        edited[5] = 0 if edited[5] != 0 else 20
        for t in range(6):
            self.assertEqual(list(gen.latent_oracle_from_payouts("jackpot", edited[:t])), original[t])
        self.assertNotEqual(list(gen.latent_oracle_from_payouts("jackpot", edited[:6])), original[6])
        # Stored rows match own-machine lags only, so another machine's payouts are not inputs.
        for t in range(8):
            self.assertEqual(list(gen.latent_oracle_from_payouts("jackpot", latent[:t])), original[t])
            self.assertEqual(rows[7][t][1]["machine_id"], 7)
            self.assertEqual(tuple(rows[machine][t][1]["probabilities"]), gen.latent_oracle_from_payouts("jackpot", latent[:t]))
        # Observable probabilities at t depend only on this machine's mask before t.
        masked = [rec["payout"] if rec["observed"] else None for rec, _ in rows[machine]]
        for t in range(8):
            self.assertTrue(
                close_tuple(
                    rows[machine][t][1]["observable_probabilities"],
                    gen.observable_probabilities_from_mask("jackpot", masked[:t]),
                )
            )
        mutated_mask = masked[:]
        mutated_mask[5] = 0 if mutated_mask[5] != 0 else None
        for t in range(6):
            self.assertTrue(
                close_tuple(
                    gen.observable_probabilities_from_mask("jackpot", masked[:t]),
                    gen.observable_probabilities_from_mask("jackpot", mutated_mask[:t]),
                )
            )

    def test_cap_and_cli_manifest(self):
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=20, n_ticks=100, n_machines=8)
        with self.assertRaises(ValueError):
            gen.generate(seed=0, n_episodes=1, n_ticks=4, n_machines=9)
        with tempfile.TemporaryDirectory() as tmp:
            code = gen.main(["--seed", "5", "--episodes", "1", "--ticks", "6", "--machines", "8", "--out", tmp])
            self.assertEqual(code, 0)
            out = Path(tmp)
            record_lines = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
            truth_lines = (out / "truth.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(record_lines), 48)
            self.assertEqual(len(truth_lines), 48)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["eligibility"], "RESEARCH_ONLY")
            self.assertFalse(manifest["synthetic_exception_verified"])
            self.assertTrue(manifest["real_only_fallback"])
            self.assertEqual(manifest["family"], "finite_order_markov")
            self.assertEqual(manifest["seed"], 5)
            self.assertEqual(manifest["settings"]["n_ticks"], 6)
            self.assertEqual(manifest["settings"]["payout_to_index"]["5"], 3)
            self.assertEqual(manifest["settings"]["payout_to_index"]["20"], 4)
            self.assertEqual(manifest["record_count"], 48)
            self.assertEqual(manifest["output_hashes"]["records.jsonl"], gen.sha256_file(out / "records.jsonl"))
            self.assertEqual(manifest["output_hashes"]["truth.jsonl"], gen.sha256_file(out / "truth.jsonl"))
            self.assertEqual(manifest["source_hashes"]["generate.py"], gen.sha256_file(Path(gen.__file__)))
            first = json.loads(record_lines[0])
            self.assertEqual(set(first), {"episode_id", "tick", "machine_id", "machine_type", "payout", "observed"})
            self.assertNotIn("signal", first)
            parsed_truth = json.loads(truth_lines[0])
            self.assertEqual(parsed_truth["probability_kind"], "latent_oracle")
            assert_simplex(self, parsed_truth["probabilities"])
            assert_simplex(self, parsed_truth["observable_probabilities"])


if __name__ == "__main__":
    unittest.main()
