from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cochain_q_oracle as c


class CochainQTests(unittest.TestCase):
    def test_bijection_and_zero_fixed_label_saving(self) -> None:
        for dimension in (2, 3):
            q = c.all_patterns(dimension)
            boundary, syndrome = c.cochain_coordinates(q)
            self.assertTrue(np.array_equal(c.inverse_cochain_coordinates(boundary, syndrome), q))
            audit = c.fixed_reparameterization_audit(q)
            self.assertEqual(audit["saved_bits"], 0)
            self.assertEqual(audit["raw_bits"], audit["cochain_total_bits"])

    def test_exact_solver_matches_global_brute_force(self) -> None:
        rng = np.random.default_rng(903)
        for dimension, cells in ((2, 2), (3, 1)):
            costs = rng.uniform(0.01, 3.0, size=(cells, 1 << dimension, 2)).astype(np.float64)
            for syndrome in (None, 0, 1):
                fast_q, fast_sse = c.best_labels(costs, dimension, syndrome)
                brute_q, brute_sse = c.brute_force_global(costs, dimension, syndrome)
                self.assertTrue(np.array_equal(fast_q, brute_q))
                self.assertEqual(fast_sse, brute_sse)

    def test_low_degree_plaquette_survives_at_exact_quarter_bpw(self) -> None:
        q = c.low_degree_even_ensemble(2)
        costs, energy = c.preference_costs(q)
        result = c.run_oracle(costs, energy, 2, public_syndrome=0)
        self.assertEqual(result["status"],
                         "SURVIVES_SOURCE_ONLY_REQUIRES_QWEN_CONTROLS_AND_SIX_PLANE_PACKET")
        self.assertEqual(result["public_fiber"]["relative_mse"],
                         result["baseline"]["relative_mse"])
        self.assertEqual(result["public_fiber"]["ideal_equivalent_gain_bpw"], 0.25)
        self.assertTrue(all(abs(v) < 1e-15 for v in c.pairwise_mutual_information(q).values()))

    def test_cube_pure_higher_order_fixture(self) -> None:
        q = c.low_degree_even_ensemble(3)
        costs, energy = c.preference_costs(q)
        result = c.run_oracle(costs, energy, 3, public_syndrome=0)
        self.assertEqual(result["public_fiber"]["ideal_equivalent_gain_bpw"], 0.125)
        self.assertTrue(all(abs(v) < 1e-15 for v in c.pairwise_mutual_information(q).values()))
        self.assertGreater(result["public_fiber"]["physical_equivalent_gain_bpw"], 0.12)

    def test_balanced_iid_fixed_labels_have_no_lossless_syndrome_gain(self) -> None:
        q = c.all_patterns(3)
        syndrome = c.mixed_syndrome(q)
        self.assertEqual(np.bincount(syndrome, minlength=2).tolist(), [128, 128])
        audit = c.fixed_reparameterization_audit(q)
        self.assertEqual(audit["saved_bits"], 0)
        costs, energy = c.preference_costs(q, preferred_cost=1.0, alternate_cost=1001.0)
        result = c.run_oracle(costs, energy, 3, public_syndrome=0)
        self.assertEqual(result["status"], "HARD_KILL_PUBLIC_COCHAIN_BELOW_0P045_BPW")
        self.assertLess(result["public_fiber"]["physical_equivalent_gain_bpw"], 0.045)

    def test_literal_packets_and_canonical_padding(self) -> None:
        for dimension in (2, 3):
            q = c.low_degree_even_ensemble(dimension)[:5]
            packet = c.encode_public_fiber(q, 0)
            self.assertTrue(np.array_equal(c.decode_public_fiber(packet, len(q), dimension, 0), q))
            if len(q) * ((1 << dimension) - 1) % 8:
                bad = bytearray(packet)
                bad[-1] |= 0x80
                with self.assertRaises(c.OracleError):
                    c.decode_public_fiber(bytes(bad), len(q), dimension, 0)

    def test_public_map_has_expert_local_read_one(self) -> None:
        ledger = c.physical_ledger(1 << 20, 2)
        self.assertEqual(ledger["fixed_map_or_selector_bits"], 0)
        self.assertTrue(ledger["expert_local"])
        self.assertEqual(ledger["cross_expert_bytes"], 0)
        self.assertEqual(ledger["logical_routed_read_amplification"], 1.0)
        self.assertEqual(ledger["page_rounding_over_payload"], 1.0)

    def test_payload_execution_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source-only"):
            c.payload_execution_gate()

    def test_deterministic_receipt_summary(self) -> None:
        summary = {}
        for dimension in (2, 3):
            q = c.low_degree_even_ensemble(dimension)
            costs, energy = c.preference_costs(q)
            result = c.run_oracle(costs, energy, dimension)
            summary[str(dimension)] = {
                "patterns": len(q),
                "gain": result["public_fiber"]["physical_equivalent_gain_bpw"],
                "status": result["status"],
                "fixed_saved": result["fixed_label_reparameterization"]["saved_bits"],
            }
        text = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        self.assertIn('"fixed_saved":0', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
