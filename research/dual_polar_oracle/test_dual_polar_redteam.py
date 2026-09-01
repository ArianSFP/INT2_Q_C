#!/usr/bin/env python3
"""CPU-only unit tests for the dual-polar matched-control machinery."""

from __future__ import annotations

import math
import unittest

import numpy as np

import dual_polar_matched_gaussian as control


class DualPolarRedteamTests(unittest.TestCase):
    def test_full_dimensions_close(self) -> None:
        self.assertEqual(control.STIEFEL_DOF, 2_620_416)
        self.assertEqual(control.SYMMETRIC_DOF, 2_098_176)
        self.assertEqual(control.STIEFEL_DOF + control.SYMMETRIC_DOF, control.VALUES_PER_EXPERT)

    def test_small_rank_curve_closes_every_partition(self) -> None:
        values = 6 * 4
        stiefel = values - 4 * 5 // 2
        rows = control.rank_curve(np.asarray([4.0, 1.0, 3.0, 2.0]), cols=4, values=values, stiefel=stiefel)
        self.assertGreater(len(rows), 0)
        for rank, row in enumerate(rows):
            self.assertEqual(row["rank"], rank)
            self.assertEqual(row["model_dof"] + row["normal_dof"], values)
            self.assertGreaterEqual(row["residual_energy"], 0.0)

    def test_moment_matched_control(self) -> None:
        matrix, receipt = control.moment_matched_gaussian((32, 48), 0.0125, 987.0, 12345)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertLess(receipt["absolute_mean_error"], 2e-8)
        self.assertLess(receipt["relative_centered_energy_error"], 2e-8)

    def test_required_identity_and_mp_support(self) -> None:
        self.assertAlmostEqual(2.0 ** (-2.0 * control.REQUIRED_S), 0.8, places=15)
        aspect = control.COLS / control.STACK_ROWS
        root = math.sqrt(aspect)
        self.assertAlmostEqual(1.0 - root, 0.057190958417936644, places=15)
        self.assertAlmostEqual(1.0 + root, 1.9428090415820634, places=15)

    def test_seed_is_stable_and_role_separated(self) -> None:
        first = control.deterministic_seed(0, 5, 18, "gate")
        self.assertEqual(first, control.deterministic_seed(0, 5, 18, "gate"))
        self.assertNotEqual(first, control.deterministic_seed(0, 5, 18, "up"))
        self.assertNotEqual(first, control.deterministic_seed(1, 5, 18, "gate"))


if __name__ == "__main__":
    unittest.main()
