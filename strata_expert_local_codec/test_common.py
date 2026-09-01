#!/usr/bin/env python3

import math
import struct
import unittest

import numpy as np

from strata_expert_local_codec import common


class ExpertAffineCommonTests(unittest.TestCase):
    def synthetic_route(self) -> bytes:
        rows = bytearray()
        for expert in range(common.EXPERTS):
            for role, axis in ((0, 0), (1, 0), (2, 1)):
                rows.extend(struct.pack(">HHBBH", expert, 100 + expert, role, axis, 768))
        return bytes(rows)

    def test_exact_physical_ledger(self) -> None:
        self.assertEqual(common.PHYSICAL_BYTES, 8_847_360)
        self.assertEqual(common.PHYSICAL_BITS, 70_778_880)
        self.assertEqual(common.PHYSICAL_BITS / common.WEIGHTS, 2.5)
        self.assertEqual(common.DIRECTORY_BYTES, 105)
        self.assertEqual(common.RESERVOIR_BYTES, 8_841_799)

    def test_labels_and_expert_affine_coverage(self) -> None:
        labels = np.repeat(np.arange(8, dtype=np.uint8), 1728)
        packed = common.pack_labels(labels)
        self.assertTrue(np.array_equal(common.unpack_labels(packed), labels))
        blocks = common.expected_block_group_ordinals(labels)
        self.assertEqual([len(row) for row in blocks], list(common.BLOCK_GROUPS))
        self.assertTrue(
            np.array_equal(
                np.sort(np.concatenate(blocks)), np.arange(common.GROUPS, dtype=np.int64)
            )
        )
        for expert in range(common.EXPERTS):
            first, second, tail = common.expert_required_blocks(expert)
            self.assertEqual(common.block_owner_experts(first), [expert])
            self.assertEqual(common.block_owner_experts(second), [expert])
            self.assertIn(expert, common.block_owner_experts(tail))

    def test_header_and_seed_domain(self) -> None:
        labels = common.pack_labels(np.repeat(np.arange(8, dtype=np.uint8), 1728))
        route = self.synthetic_route()
        coefficients = [(np.float32(1.0), np.float32(0.0))] * common.EXPERTS
        header = common.build_header(coefficients, [0] * common.EXPERTS, route, labels)
        common.validate_header(header, route, labels)
        profiles = bytes(range(common.BLOCKS))
        rows = [common.derive_seeds(header, route, labels, profiles, i) for i in range(common.BLOCKS)]
        self.assertEqual(len({row[2] for row in rows}), common.BLOCKS)

    def test_allocator_fits_reserved_budget(self) -> None:
        energy = np.linspace(1.0, 15.0, common.BLOCKS, dtype=np.float64)
        profiles, report = common.allocate_profiles(energy)
        self.assertEqual(profiles.shape, (common.BLOCKS,))
        self.assertLessEqual(report["nominal_profile_bits"], common.NOMINAL_PROFILE_BUDGET_BITS)
        self.assertGreaterEqual(report["nominal_unused_bits"], 0)
        self.assertTrue(math.isfinite(report["projected_relative_mse"]))


if __name__ == "__main__":
    unittest.main()
