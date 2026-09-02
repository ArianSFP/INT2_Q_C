#!/usr/bin/env python3
"""Hostile source-only tests for the STRATA-RM6 gate."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import unittest

import numpy as np

from exact_oracle import exact_joint_oracle, unconstrained_64way_bound
from packet_codec import _build_packet, _header, decode_packet, encode_packet
from rm6_core import (LOCAL_N, ORDER_BANK, RM6Error, assemble_indices,
                      bank_dimensions, dimension_ledger, exact_distortion_costs,
                      frozen_external_from_seed, generator_matrix,
                      plane_from_information, reconstruction_levels, rm_dimension,
                      rm_information_positions, selected_distortion)
from strata_rm_sc import (classify_selected_dimension, replay_six_greedy,
                          replay_six_prescribed, rm_ordered_positions)
from strata_semantics import authenticate_auditor, verify_rm_orientation


AUDITOR = Path(os.environ.get(
    "STRATA_AUDITOR",
    str(Path(__file__).resolve().parents[2] /
        "strata_v2_klt_mixed_independent_auditor_v1.py")))


class RMGeometryTests(unittest.TestCase):
    def test_rm5_12_dimension(self) -> None:
        self.assertEqual(rm_dimension(5, 12), 1586)
        self.assertEqual(sum(bank_dimensions(0)), 9516)
        self.assertEqual(9516 / 4096.0, 2.3232421875)

    def test_exact_orientation_and_row_weights(self) -> None:
        receipt = verify_rm_orientation()
        self.assertEqual(receipt["dimension"], 1586)
        self.assertEqual(receipt["minimum_row_weight"], 128)
        matrix = generator_matrix(5, 2)
        positions = rm_information_positions(5, 2)
        self.assertTrue(np.array_equal(matrix.sum(1),
                                       np.asarray([1 << int(i).bit_count()
                                                   for i in positions])))

    def test_all_banks_are_exact_rm_orders_and_fit_dimension_ledger(self) -> None:
        previous = None
        for bank_id, orders in ORDER_BANK.items():
            self.assertTrue(all(left <= right for left, right in zip(orders, orders[1:])))
            row = dimension_ledger(bank_id)
            self.assertTrue(row["is_exact_rm_not_dimension_truncated"])
            self.assertLessEqual(row["packet_bytes"], 1280)
            if previous is not None:
                self.assertLessEqual(row["information_bits"], previous)
            previous = row["information_bits"]

    def test_global_k_is_not_mislabeled_exact_rm(self) -> None:
        row = classify_selected_dimension(21, 700000)
        self.assertFalse(row["exact_rm"])
        self.assertEqual(row["name"], "RM-ordered truncated polar set")
        exact = classify_selected_dimension(12, 1586)
        self.assertTrue(exact["exact_rm"])
        positions = rm_ordered_positions(1 << 12, 1586)
        self.assertTrue(np.array_equal(np.sort(positions),
                                       rm_information_positions(12, 5)))


class SixPlaneAndCosetTests(unittest.TestCase):
    def test_plane_assembly_exact_0_63(self) -> None:
        rng = np.random.default_rng(73)
        planes = [rng.integers(0, 2, 32, dtype=np.uint8) for _ in range(6)]
        indices = assemble_indices(planes)
        expected = sum(planes[level].astype(np.uint8) << np.uint8(level)
                       for level in range(6))
        self.assertTrue(np.array_equal(indices, expected))
        self.assertTrue(np.all(indices < 64))

    def test_four_level_abstraction_fails_closed(self) -> None:
        with self.assertRaises(RM6Error):
            assemble_indices([np.zeros(8, dtype=np.uint8) for _ in range(4)])

    def test_zero_and_current_random_cosets_differ(self) -> None:
        variables, order, seed = 5, 2, 19
        info = np.zeros(rm_dimension(order, variables), dtype=np.uint8)
        zero = plane_from_information(info, variables, order,
                                      np.zeros(1 << variables, dtype=np.uint8))
        random = plane_from_information(info, variables, order,
                                        frozen_external_from_seed(1 << variables,
                                                                  seed, 1))
        self.assertFalse(np.array_equal(zero, random))

    def test_direct_plane_matches_sc_replay(self) -> None:
        bank, profile, seed, mode = 7, 80, 1234, "zero"
        decisions = [np.zeros(dimension, dtype=np.uint8)
                     for dimension in bank_dimensions(bank)]
        replay = replay_six_prescribed(bank, profile, seed, mode, decisions)
        direct = []
        for level, (order, bits) in enumerate(zip(ORDER_BANK[bank], decisions,
                                                  strict=True), start=1):
            direct.append(plane_from_information(bits, 12, order,
                                                  np.zeros(LOCAL_N, dtype=np.uint8)))
        self.assertTrue(np.array_equal(replay["indices"], assemble_indices(direct)))


class LiteralPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bank, cls.profile, cls.seed = 0, 96, 0x13579BDF
        cls.greedy = replay_six_greedy(cls.bank, cls.profile, cls.seed, "zero")
        cls.packet, cls.encoded = encode_packet(
            cls.greedy["decisions"], bank_id=cls.bank, scale_fp16_bits=0x3C00,
            profile_q=cls.profile, coset_mode="zero", sc_seed=cls.seed,
            rht_seed=0x0123456789ABCDEF)

    def test_canonical_packet_and_actual_rate(self) -> None:
        decoded = decode_packet(self.packet)
        self.assertTrue(decoded["canonical_reencode_match"])
        self.assertTrue(np.array_equal(decoded["indices"], self.encoded["indices"]))
        self.assertEqual(len(self.packet) % 128, 0)
        self.assertLessEqual(len(self.packet), 1280)
        self.assertTrue(decoded["ledger"]["actual_passes_2_5_bpw"])
        self.assertFalse(decoded["ledger"]["actual_target_rate_eligible"])
        self.assertEqual(decoded["ledger"]["promotion_status"],
                         "MECHANISM_FIXTURE_BELOW_2_15_BPW")
        self.assertEqual(decoded["information_bits"], 9516)
        self.assertEqual(decoded["logical_bits"],
                         decoded["ledger"]["emitted_arithmetic_bits"])
        self.assertNotEqual(decoded["information_bits"], decoded["logical_bits"])

    def test_bank0_arithmetic_overflow_fails_closed(self) -> None:
        logical_bits = 10_000
        header = _header(0, logical_bits, 0x3C00, self.profile, "zero",
                         self.seed, 0x0123456789ABCDEF)
        with self.assertRaisesRegex(RM6Error, "exceeds 2.5 bpw"):
            _build_packet(header, bytes(logical_bits // 8), logical_bits)

    def test_dimension_screen_is_not_emitted_rate_or_promotion(self) -> None:
        row = dimension_ledger(7)
        self.assertTrue(row["dimension_screen_not_emitted_arithmetic_bits"])
        self.assertFalse(row["dimension_screen_target_rate_eligible"])
        self.assertEqual(row["physical_bpw"], 1.25)

    def test_crc_tamper_rejected(self) -> None:
        tampered = bytearray(self.packet)
        tampered[50] ^= 1
        with self.assertRaises(RM6Error):
            decode_packet(bytes(tampered))

    def test_alignment_tamper_rejected(self) -> None:
        tampered = bytearray(self.packet)
        tampered[-1] = 1
        with self.assertRaises(RM6Error):
            decode_packet(bytes(tampered))

    def test_bank_header_tamper_rejected(self) -> None:
        tampered = bytearray(self.packet)
        tampered[7] = 7
        with self.assertRaises(RM6Error):
            decode_packet(bytes(tampered))


class DistortionAndOracleTests(unittest.TestCase):
    def test_exact_64way_costs(self) -> None:
        levels = reconstruction_levels(0x3C00)
        target = np.asarray([levels[0], levels[17] + 0.1, levels[63]], dtype=np.float64)
        costs = exact_distortion_costs(target, 0x3C00)
        self.assertEqual(costs.shape, (3, 64))
        indices = np.asarray([0, 17, 63], dtype=np.uint8)
        expected = float(np.sum((target - levels[indices]) ** 2))
        self.assertAlmostEqual(selected_distortion(costs, indices), expected, places=15)

    def test_bounded_joint_oracle_is_legal_and_dominates_candidate(self) -> None:
        variables, orders = 3, (1, 1, 0, 0, 0, 0)
        coordinate = np.arange(1 << variables)
        levels = reconstruction_levels(0x3C00)
        target = levels[((13 * coordinate) & 63).astype(np.uint8)] + 0.01
        costs = exact_distortion_costs(target, 0x3C00)
        exact = exact_joint_oracle(costs, variables, orders, sc_seed=7,
                                   coset_mode="zero")
        unconstrained = unconstrained_64way_bound(costs)
        self.assertEqual(exact["candidate_messages"], 4096)
        self.assertGreaterEqual(exact["distortion"] + 1e-12,
                                unconstrained["distortion"])


class AuthenticationAndClosureTests(unittest.TestCase):
    def test_authenticated_current_semantics(self) -> None:
        row = authenticate_auditor(AUDITOR)
        self.assertEqual(row["current_block_log2"], [20, 21])
        self.assertTrue(row["random_frozen_coset_authenticated"])

    def test_no_payload_cli(self) -> None:
        from run_gate import make_receipt
        self.assertEqual(tuple(inspect.signature(make_receipt).parameters), ("auditor",))
        source = inspect.getsource(make_receipt)
        self.assertNotIn("qwen_path", source)
        self.assertNotIn("coarse_path", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
