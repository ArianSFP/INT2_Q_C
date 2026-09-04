from __future__ import annotations

from fractions import Fraction
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_PATH = Path(__file__).resolve().with_name("tetrapath4_oracle.py")
SPEC = importlib.util.spec_from_file_location("tetrapath4_oracle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
t = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t
SPEC.loader.exec_module(t)


class TetraPathSourceTests(unittest.TestCase):
    def test_tuple_roundtrip(self) -> None:
        ids = np.arange(256, dtype=np.int64)
        np.testing.assert_array_equal(t.tuple_ids(t.labels_from_ids(ids)), ids)

    def test_xor_synergy_pairwise_null_full_gain_quarter_bpw(self) -> None:
        rows = []
        for a, b, c in itertools.product((0, 1), repeat=3):
            rows.append((a, b, c, a ^ b ^ c))
        labels = np.repeat(np.asarray(rows, dtype=np.uint8), 64, axis=0)
        census = t.fixed_assignment_census(labels)
        for value in census["pairwise_mi"].values():
            self.assertAlmostEqual(value, 0.0, places=14)
        self.assertAlmostEqual(census["independent_bpw"], 1.0, places=14)
        self.assertAlmostEqual(census["best_pair_bpw"], 1.0, places=14)
        self.assertAlmostEqual(census["tree_bpw"], 1.0, places=14)
        self.assertAlmostEqual(census["full_bpw"], 0.75, places=14)
        self.assertAlmostEqual(census["fourway_gain_over_best_factorized_bpw"], 0.25,
                               places=14)
        self.assertAlmostEqual(census["total_correlation_bpw"], 0.25, places=14)
        self.assertAlmostEqual(census["best_2plus2_saving_bpw"], 0.0, places=14)
        self.assertAlmostEqual(census["best_chow_liu_saving_bpw"], 0.0, places=14)
        self.assertAlmostEqual(census["pairwise_maxent_bpw"], 1.0, places=12)
        self.assertAlmostEqual(census["residual_connected_information_bpw"], 0.25,
                               places=12)
        self.assertLess(abs(census["parity_bpw"] - 0.75), 0.006)
        fiber = t.fiber_fixed_ledger(labels, "fiber_gray_low")
        self.assertTrue(fiber["encodable"])
        self.assertAlmostEqual(fiber["common_bits_per_tetrad"], 1.0, places=14)
        self.assertEqual([round(x, 14) for x in fiber["private_bits_per_tetrad"]],
                         [1.0, 1.0])
        self.assertAlmostEqual(fiber["total_bpw"], 0.75, places=14)
        self.assertAlmostEqual(fiber["max_logical_read_amplification"], 4 / 3,
                               places=14)

    def test_balanced_iid_control_hard_kills(self) -> None:
        labels = np.repeat(t.QTABLE, 16, axis=0)
        census = t.fixed_assignment_census(labels)
        self.assertAlmostEqual(census["independent_bpw"], 2.0, places=14)
        self.assertAlmostEqual(census["best_pair_bpw"], 2.0, places=14)
        self.assertAlmostEqual(census["tree_bpw"], 2.0, places=14)
        self.assertAlmostEqual(census["full_bpw"], 2.0, places=14)
        self.assertAlmostEqual(census["pairwise_maxent_bpw"], 2.0, places=12)
        self.assertAlmostEqual(census["residual_connected_information_bpw"], 0.0,
                               places=12)
        self.assertLess(census["fourway_gain_over_best_factorized_bpw"], t.EARLY_KILL_BPW)

    def test_explicit_down_transposed_alignment(self) -> None:
        shape = (2, 3)
        up_e = np.arange(6, dtype=np.float64).reshape(shape)
        down_e_t = up_e + 10
        up_f = up_e + 20
        down_f_t = up_e + 30
        got = t.aligned_up_down_values(up_e, down_e_t, up_f, down_f_t)
        np.testing.assert_array_equal(got[:, 0], up_e.ravel())
        np.testing.assert_array_equal(got[:, 1], down_e_t.ravel())
        np.testing.assert_array_equal(got[:, 2], up_f.ravel())
        np.testing.assert_array_equal(got[:, 3], down_f_t.ravel())

    def test_expanded_distortion_matches_scalar_bruteforce(self) -> None:
        rng = np.random.default_rng(123)
        costs = rng.random((7, 4, 4), dtype=np.float64)
        expanded = t.expanded_tuple_costs(costs)
        for i in range(costs.shape[0]):
            for qid, labels in enumerate(t.QTABLE):
                expected = sum(costs[i, v, labels[v]] for v in range(4))
                self.assertAlmostEqual(expanded[i, qid], expected, places=15)

    def test_assignment_equals_exhaustive_tiny_global_search(self) -> None:
        rng = np.random.default_rng(456)
        costs = rng.random((2, 256), dtype=np.float64)
        p = rng.random(256, dtype=np.float64)
        p /= p.sum()
        bit_weight = 0.13
        got = t.assign_given_probability(costs, p, bit_weight)
        best = None
        for a in range(256):
            for b in range(256):
                objective = (costs[0, a] + costs[1, b] -
                             bit_weight * (math.log2(p[a]) + math.log2(p[b])))
                candidate = (objective, a, b)
                if best is None or candidate < best:
                    best = candidate
        np.testing.assert_array_equal(got, np.asarray(best[1:], dtype=np.uint16))

    def test_chow_liu_matches_exhaustive_tree_search(self) -> None:
        rng = np.random.default_rng(789)
        labels = rng.integers(0, 4, size=(4096, 4), dtype=np.uint8)
        mi = {(a, b): t._mutual_information(labels, a, b)
              for a in range(4) for b in range(a + 1, 4)}
        expected = min(t.TREES, key=lambda tree: (-sum(mi[e] for e in tree), tree))
        self.assertEqual(t.chow_liu_tree(labels), expected)

    def test_all_pairings_are_exhaustive_and_symmetric(self) -> None:
        got = {tuple(sorted(tuple(sorted(e)) for e in pairing))
               for pairing in t.PAIRINGS.values()}
        expected = {
            ((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))
        }
        self.assertEqual(got, expected)
        nearest = np.zeros((3, 4), dtype=np.uint8)
        starts = t.symmetric_multistarts(nearest)
        offsets = {tuple(row) for start in starts for row in start[:1]}
        for offset in tuple(offsets):
            for permutation in itertools.permutations(range(4)):
                self.assertIn(tuple(offset[i] for i in permutation), offsets)

    def test_convex_hull_and_equal_rate_comparison(self) -> None:
        baseline = [(1.0, 0.5), (1.5, 0.30), (2.0, 0.25)]
        challenger = [(1.0, 0.40), (1.5, 0.28), (2.0, 0.20)]
        comparison = t.compare_frontiers(baseline, challenger)
        self.assertGreater(comparison["equal_rate_best_equivalent_bpw"], 0)
        self.assertGreater(comparison["optimistic_best_equivalent_bpw"], 0)

    def test_small_oracle_runs_all_families_and_uses_one_grid(self) -> None:
        rng = np.random.default_rng(101112)
        values = rng.normal(size=(64, 4)).astype(np.float64)
        levels = np.broadcast_to(np.asarray((-1.5, -0.5, 0.5, 1.5), dtype=np.float64),
                                 (64, 4, 4)).copy()
        costs, energy = t.fourway_distortion_costs(values, levels)
        result = t.run_dominant_oracle(costs, energy,
                                       (Fraction(0), Fraction(1, 64), Fraction(1, 8)))
        self.assertEqual(set(result["frontiers"]), set(t.FAMILIES))
        for points in result["points"].values():
            self.assertEqual([p["lambda"] for p in points], [[0, 1], [1, 64], [1, 8]])
        self.assertTrue(result["kill_only"])
        self.assertFalse(result["tables_and_time_sharing_charged"])
        self.assertIn("full_joint_best_G4_bpw", result)
        for key in ("full_over_independent_codec_relevance", "full_over_best_2plus2",
                    "full_over_chow_liu_tree",
                    "full_over_pairwise_maxent_residual_synergy"):
            self.assertIn(key, result)

    def test_payload_gate_is_unconditionally_disabled(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source-only"):
            t.payload_execution_gate("anything")

    def test_validation_rejects_nonfinite_and_negative(self) -> None:
        costs = np.zeros((2, 4, 4), dtype=np.float64)
        with self.assertRaises(t.OracleError):
            t.validate_costs(costs, 0.0)
        costs[0, 0, 0] = -1
        with self.assertRaises(t.OracleError):
            t.validate_costs(costs, 1.0)
        costs[0, 0, 0] = np.nan
        with self.assertRaises(t.OracleError):
            t.validate_costs(costs, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
