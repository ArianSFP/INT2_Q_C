#!/usr/bin/env python3
"""Hostile/source-free tests for the nonlocal WFA prototype."""

from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from nonlocal_wfa import (
    BLOCK_LENGTH,
    CHECKS,
    Q16_TOTAL,
    SparseUnifilarWFA,
    capacity_sanity,
    decode_blocks,
    encode_blocks,
    exact_normalization_probe,
    fit_candidate,
    generate_syndrome_blocks,
    logical_codelength_bits,
    model_byte_ledger,
    select_model,
    suffix_cross_entropy,
    worst_unaligned_page_union,
)


class SourceOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = generate_syndrome_blocks(1024, 101, True)
        self.valid = generate_syndrome_blocks(2048, 102, True)

    def test_marginals_are_fair(self) -> None:
        means = self.train.mean(axis=0)
        self.assertLess(float(np.max(np.abs(means - 0.5))), 0.06)

    def test_six_checks_are_exact(self) -> None:
        for group in range(CHECKS):
            expected = np.bitwise_xor.reduce(self.train[:, group:26:CHECKS], axis=1)
            np.testing.assert_array_equal(expected, self.train[:, 26 + group])

    def test_dense_rows_normalized_exactly(self) -> None:
        model = fit_candidate(self.train, CHECKS)
        dense = model.expand_dense()
        np.testing.assert_array_equal(
            dense.sum(axis=(1, 3), dtype=np.uint64),
            np.full((12, model.chi), Q16_TOTAL, dtype=np.uint64),
        )

    def test_fraction_predictive_normalization(self) -> None:
        model = fit_candidate(self.train, CHECKS)
        probe = exact_normalization_probe(model, self.valid[0, :8].tolist())
        self.assertTrue(probe["all_exactly_normalized"])
        alpha = tuple(Fraction(1 if index == 0 else 0, 1) for index in range(model.chi))
        self.assertEqual(
            model.predictive_fraction(alpha, 0, 0) + model.predictive_fraction(alpha, 0, 1),
            Fraction(1, 1),
        )

    def test_packet_roundtrip_and_formula(self) -> None:
        model = fit_candidate(self.train, CHECKS)
        packet = model.serialize()
        restored = SparseUnifilarWFA.deserialize(packet)
        self.assertEqual(restored.serialize(), packet)
        ledger = model_byte_ledger(model)
        self.assertEqual(len(packet), ledger["sparse_physical_bytes"])
        self.assertEqual(ledger["sparse_page_bytes"], 4096)
        self.assertGreater(ledger["dense_equivalent_physical_bytes"], len(packet))

    def test_arithmetic_roundtrip(self) -> None:
        model = fit_candidate(self.train, CHECKS)
        sample = self.valid[:64]
        payload, meaningful = encode_blocks(model, sample)
        decoded = decode_blocks(model, payload, sample.shape[0])
        np.testing.assert_array_equal(decoded, sample)
        self.assertGreater(meaningful, 0)

    def test_heldout_model_selection_detects_hidden_state(self) -> None:
        model, rows = select_model(self.train, self.valid, expert_count=8)
        self.assertEqual(model.syndrome_bits, CHECKS)
        self.assertEqual(model.chi, 64)
        self.assertLess(logical_codelength_bits(model, self.valid) / self.valid.size, 0.84)
        self.assertEqual(len(rows), CHECKS + 1)

    def test_iid_control_does_not_gain(self) -> None:
        train = generate_syndrome_blocks(2048, 201, False)
        valid = generate_syndrome_blocks(4096, 202, False)
        model, _ = select_model(train, valid, expert_count=8)
        logical_bps = logical_codelength_bits(model, valid) / valid.size
        self.assertLessEqual(model.syndrome_bits, 1)
        self.assertGreater(logical_bps, 0.999)

    def test_suffix_depths_miss(self) -> None:
        for depth in (0, 1, 2, 4, 8, 16):
            row = suffix_cross_entropy(self.train, self.valid, depth)
            self.assertGreater(float(row["logical_bps"]), 0.99)

    def test_single_parity_capacity_is_tiny(self) -> None:
        row = capacity_sanity(2048)
        self.assertAlmostEqual(row["single_parity_saving_bpw"], 1 / 2048)
        self.assertGreater(row["standalone_required_bits_per_block"], 313.0)
        self.assertEqual(row["fixture_constraints_per_2048_if_reset_every_32"], 384)

    def test_worst_unaligned_pages(self) -> None:
        self.assertEqual(worst_unaligned_page_union(1), 4096)
        self.assertEqual(worst_unaligned_page_union(4096), 8192)
        self.assertEqual(worst_unaligned_page_union(4097), 8192)

    def test_no_existing_output_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                path.open("xb")


if __name__ == "__main__":
    unittest.main()

