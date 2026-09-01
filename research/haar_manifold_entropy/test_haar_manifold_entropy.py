#!/usr/bin/env python3
"""Unit tests for the canonical Haar/ACG gate (NumPy only, no GPU)."""

from __future__ import annotations

import math
import unittest

import numpy as np

import haar_manifold_entropy as hm


class HaarManifoldEntropyTests(unittest.TestCase):
    def test_dimension_partition(self) -> None:
        self.assertEqual(hm.orientation_dof(), 1_277_568)
        self.assertEqual(hm.triangular_dof(), 295_296)
        self.assertEqual(hm.orientation_dof() + hm.triangular_dof(), hm.ROWS * hm.COLS)

    def test_raw_qr_chart_is_invertible(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.standard_normal((11, 5))
        h, tau = np.linalg.qr(source, mode="raw")
        rebuilt = hm.reconstruct_from_raw_qr(h, tau)
        np.testing.assert_allclose(rebuilt, source, rtol=2e-13, atol=2e-13)

    def test_householder_bins_recover_unit_spheres(self) -> None:
        rng = np.random.default_rng(5)
        matrix = rng.standard_normal((5, 11)).astype(np.float32)
        energy, counts, diagnostics = hm.householder_sphere_bins(matrix, coord_bins=4)
        self.assertEqual(energy.shape, (5, 4))
        self.assertEqual(counts.shape, (5, 4))
        np.testing.assert_allclose(np.sum(energy, axis=1), np.ones(5), rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(np.sum(counts, axis=1), 11 - np.arange(5))
        self.assertLess(diagnostics["max_sphere_norm_abs_error"], 2e-6)

    def test_acg_identity_and_scale_invariance(self) -> None:
        energy = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
        counts = np.asarray([[2, 2, 3, 3], [2, 2, 2, 2]])
        zero = hm.acg_log_ratio_nats(energy, counts, np.zeros(4))
        np.testing.assert_allclose(zero, np.zeros(2), atol=1e-15)
        shape = np.asarray([-0.3, 0.1, 0.25, -0.05])
        left = hm.acg_log_ratio_nats(energy, counts, shape)
        right = hm.acg_log_ratio_nats(energy, counts, shape + 7.25)
        np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-14)

    def test_fit_improves_training_likelihood_and_serializes_fp16(self) -> None:
        rng = np.random.default_rng(6)
        experts, rows, bins = 7, 8, 4
        counts = np.full((rows, bins), 4, dtype=np.int64)
        # An intentionally anisotropic Dirichlet proxy makes the optimum clear.
        features = rng.dirichlet([9.0, 4.0, 2.0, 1.0], size=(experts, rows))
        table, diagnostics = hm.fit_shape_table(
            features,
            counts,
            reflector_bands=2,
            ridge=1.0,
            iterations=24,
            quantize_fp16=True,
        )
        self.assertEqual(table.shape, (2, bins))
        self.assertTrue(np.all(np.isfinite(table)))
        self.assertEqual(table.astype(np.float16).tobytes(), table.astype(np.float16).tobytes())
        self.assertTrue(all(row["final_log_likelihood_nats"] >= row["initial_log_likelihood_nats"] for row in diagnostics))
        fitted, _ = hm.score_shape_table(features, counts, table)
        identity, _ = hm.score_shape_table(features, counts, np.zeros_like(table))
        self.assertGreater(fitted, identity)

    def test_leave_one_out_is_whole_expert_and_finite(self) -> None:
        rng = np.random.default_rng(7)
        experts, roles, rows, bins = 5, 2, 8, 4
        counts = np.full((rows, bins), 3, dtype=np.int64)
        features = rng.dirichlet([5.0, 3.0, 2.0, 1.0], size=(experts, roles, rows))
        result = hm.leave_one_expert_out(features, counts, ridge=1.0, iterations=12)
        self.assertEqual(len(result["groups"]), experts)
        self.assertEqual([row["heldout_index"] for row in result["groups"]], list(range(experts)))
        self.assertTrue(all(math.isfinite(row["gain_bpw"]) for row in result["groups"]))
        self.assertEqual(len(bytes.fromhex(result["final_fp16_shape_table_sha256"])), 32)

    def test_exact_rate_and_read_ledgers(self) -> None:
        ledgers = hm.exact_rate_read_ledgers()
        self.assertEqual([row["requested_rate_bpw"] for row in ledgers], list(hm.DEFAULT_RATES))
        for row in ledgers:
            self.assertLessEqual(row["physical_rate_bpw"], row["requested_rate_bpw"])
            self.assertLess(row["max_cold_4k_amplification"], 2.0)
            self.assertEqual(sum(row["frame_bytes"]) + row["global_prefix_bytes"], row["container_bytes"])
            self.assertTrue(row["below_2x"])

    def test_required_rate_identity(self) -> None:
        self.assertAlmostEqual(hm.REQUIRED_S, 0.16096404744368115, places=15)
        self.assertAlmostEqual(2.0 ** (-2.0 * hm.REQUIRED_S), 0.8, places=15)


if __name__ == "__main__":
    unittest.main()
