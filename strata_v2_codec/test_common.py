#!/usr/bin/env python3
"""Source-free unit tests for the frozen STRATA-v2 format primitives."""

from __future__ import annotations

import math
import struct
import unittest

import numpy as np

from strata_v2_codec import common


class CommonTests(unittest.TestCase):
    def fixture(self):
        route = b"".join(
            struct.pack(">HHBBH", 2 * triplet, 7 + triplet, role, 1 if role == 2 else 0, 768)
            for triplet in range(6)
            for role in range(3)
        )
        labels = common.pack_labels(np.repeat(np.arange(8, dtype=np.uint8), 1728))
        codes = [-16384, -8192, -1, 0, 8192, 16384]
        coefficients = []
        for code in codes:
            theta = code * math.pi / 32768.0
            coefficients.append(
                (np.float32(math.cos(theta)), np.float32(math.sin(theta)))
            )
        return route, labels, codes, coefficients

    def test_ledger(self):
        self.assertEqual(sum(common.BLOCK_SIZES), common.WEIGHTS)
        self.assertEqual(common.PHYSICAL_BITS, 8 * common.PHYSICAL_BYTES)
        self.assertEqual(common.PHYSICAL_BITS, 60_869_832)
        self.assertLessEqual(common.PHYSICAL_BITS, common.INTEGER_CAP_BITS)
        self.assertEqual(
            common.RESERVOIR_BYTES * 8 - common.GLOBAL_RESERVE_BITS,
            common.NOMINAL_PROFILE_BUDGET_BITS,
        )

    def test_header_and_coefficient_integrity(self):
        route, labels, codes, coefficients = self.fixture()
        header = common.build_header(coefficients, codes, route, labels)
        common.validate_header(header, route, labels)
        changed = bytearray(header)
        changed[32] ^= 1
        # Repair CRC so rejection is specifically coefficient regeneration.
        import zlib

        struct.pack_into("<I", changed, 124, zlib.crc32(changed[:124]) & 0xFFFFFFFF)
        with self.assertRaisesRegex(ValueError, "bit-exact"):
            common.validate_header(bytes(changed), route, labels)

    def test_route_and_label_mutations(self):
        route, labels, codes, coefficients = self.fixture()
        header = common.build_header(coefficients, codes, route, labels)
        with self.assertRaises(ValueError):
            common.validate_header(header, route[:-1] + bytes([route[-1] ^ 1]), labels)
        with self.assertRaises(ValueError):
            common.validate_header(header, route, labels[:-1] + bytes([labels[-1] ^ 1]))

    def test_seed_is_header_route_label_profile_bound(self):
        route, labels, codes, coefficients = self.fixture()
        header = common.build_header(coefficients, codes, route, labels)
        profiles = bytes(range(14))
        first = common.derive_seeds(header, route, labels, profiles, 0)
        self.assertEqual(first, common.derive_seeds(header, route, labels, profiles, 0))
        changed = bytearray(profiles)
        changed[0] ^= 1
        self.assertNotEqual(first, common.derive_seeds(header, route, labels, bytes(changed), 0))
        self.assertNotEqual(first, common.derive_seeds(header, route, labels, profiles, 1))

    def test_dp_is_deterministic_and_feasible(self):
        energy = np.geomspace(1.0, 32.0, 14)
        q1, audit1 = common.allocate_profiles(energy)
        q2, audit2 = common.allocate_profiles(energy.copy())
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(audit1, audit2)
        self.assertLessEqual(
            audit1["nominal_profile_bits"], common.NOMINAL_PROFILE_BUDGET_BITS
        )
        self.assertEqual(q1.dtype, np.uint8)
        self.assertEqual(q1.size, 14)

    def test_label_roundtrip(self):
        labels = (np.arange(common.GROUPS, dtype=np.uint16) % 8).astype(np.uint8)
        self.assertTrue(np.array_equal(labels, common.unpack_labels(common.pack_labels(labels))))


if __name__ == "__main__":
    unittest.main()
