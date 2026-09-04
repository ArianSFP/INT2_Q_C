import itertools
import importlib.util
import math
import pathlib
import sys
import unittest

import numpy as np

MODULE_PATH = pathlib.Path(__file__).resolve().with_name("renorm_q_oracle.py")
SPEC = importlib.util.spec_from_file_location("renorm_q_oracle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
rq = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rq
SPEC.loader.exec_module(rq)


class RenormQSourceTests(unittest.TestCase):
    def test_public_map_bank_is_frozen_and_total(self):
        bank = rq.public_map_bank(4, 4)
        self.assertEqual(len(bank), 8)
        self.assertEqual(len({m.name for m in bank}), len(bank))
        for spec in bank:
            spec.validate(4 ** 4)
            self.assertEqual(spec.outputs.dtype, np.uint8)

    def test_rsmi_xor_detects_suffix_invisible_collective(self):
        blocks = np.asarray(list(itertools.product(range(2), repeat=4)), dtype=np.uint8)
        environment = np.bitwise_xor.reduce(blocks, axis=1)[:, None]
        rows = rq.collective_variable_census(blocks, environment, 2, beta=0.0,
                                             charge_descriptor=False)
        by_name = {r["map"]: r for r in rows}
        self.assertAlmostEqual(by_name["gray_low_parity"]["mutual_information_bits_per_cell"],
                               1.0, places=12)
        self.assertAlmostEqual(rows[0]["mutual_information_bits_per_cell"], 1.0,
                               places=12)

    def test_balanced_iid_control_is_exactly_zero(self):
        block_rows = list(itertools.product(range(2), repeat=4))
        blocks, env = [], []
        for row in block_rows:
            for e in itertools.product(range(2), repeat=2):
                blocks.append(row)
                env.append(e)
        rows = rq.collective_variable_census(np.asarray(blocks, dtype=np.uint8),
                                             np.asarray(env, dtype=np.uint8), 2,
                                             beta=0.0, charge_descriptor=False)
        self.assertTrue(all(abs(r["mutual_information_bits_per_cell"]) < 1e-12
                            for r in rows))

    def test_min_sum_matches_exhaustive_global_enumeration(self):
        bank = {m.name: m for m in rq.public_map_bank(2, 2)}
        spec = bank["gray_low_parity"]
        costs = np.asarray([
            [[0.0, 0.7], [0.1, 0.0]],
            [[0.0, 0.4], [0.0, 0.5]],
        ], dtype=np.float64)
        root = np.asarray([0.4, 1.1], dtype=np.float64)
        transition = np.asarray([[[0.1, 1.6], [1.3, 0.2]]], dtype=np.float64)
        leaf = rq.uniform_fiber_leaf_nll(spec, 4)
        dp = rq.exact_tree_min_sum(costs, 2, spec, root, transition, leaf, 0.37)
        brute = rq.brute_force_tree(costs, 2, spec, root, transition, leaf, 0.37)
        self.assertAlmostEqual(dp.objective, brute.objective, places=12)
        self.assertAlmostEqual(dp.distortion, brute.distortion, places=12)
        self.assertAlmostEqual(dp.modeled_bits, brute.modeled_bits, places=12)
        np.testing.assert_array_equal(dp.tuple_ids, brute.tuple_ids)
        np.testing.assert_array_equal(dp.leaf_states, brute.leaf_states)

    def test_flexible_labels_can_trade_distortion_for_hierarchy_rate(self):
        spec = {m.name: m for m in rq.public_map_bank(2, 2)}["gray_low_parity"]
        costs = np.asarray([
            [[0.0, 0.3], [0.0, 0.3]],
            [[0.0, 0.3], [0.3, 0.0]],
        ], dtype=np.float64)
        root = np.zeros(2, dtype=np.float64)
        transition = np.asarray([[[0.0, 8.0], [8.0, 0.0]]], dtype=np.float64)
        leaf = rq.uniform_fiber_leaf_nll(spec, 4)
        result = rq.exact_tree_min_sum(costs, 2, spec, root, transition, leaf, 1.0)
        # Nearest labels have opposite parity; the optimum moves one label so
        # both collective states agree rather than merely entropy-coding NN labels.
        self.assertEqual(int(result.leaf_states[0]), int(result.leaf_states[1]))
        self.assertGreater(result.distortion, 0.0)
        self.assertLess(result.distortion, 1.0)

    def test_logical_read_projection_and_strict_boundary(self):
        self.assertAlmostEqual(rq.logical_common_private_read_amplification(2, 1, 1),
                               4.0 / 3.0, places=12)
        self.assertLess(rq.logical_common_private_read_amplification(16, 1, 3), 2.0)
        self.assertEqual(rq.kill_decision(0.02, 0.0, 0.01), "HARD_KILL")
        self.assertEqual(rq.kill_decision(0.04, 0.0, 0.01), "SCIENTIFIC_SIGNAL_ONLY")
        self.assertEqual(rq.kill_decision(0.06, 0.0, 0.01),
                         "PROMOTE_TO_SEPARATE_FINITE_PROJECTION")

    def test_rejects_noncanonical_and_large_inputs(self):
        with self.assertRaises(rq.OracleError):
            rq.tuple_table(7, 2)
        with self.assertRaises(rq.OracleError):
            rq.collective_variable_census(np.zeros((4, 4), dtype=np.float64),
                                          np.zeros((4, 1), dtype=np.uint8), 2)
        spec = rq.public_map_bank(2, 2)[0]
        costs = np.zeros((3, 2, 2), dtype=np.float64)
        with self.assertRaises(rq.OracleError):
            rq.exact_tree_min_sum(costs, 2, spec, np.zeros(2),
                                  np.zeros((1, 2, 2)),
                                  rq.uniform_fiber_leaf_nll(spec, 4), 1.0)

    def test_source_closure(self):
        root = pathlib.Path(__file__).resolve().parent
        source = (root / "renorm_q_oracle.py").read_text(encoding="utf-8").lower()
        forbidden = ("huggingface", "requests", "subprocess", "socket", "cupy",
                     "torch", "ssh", "runpod")
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
