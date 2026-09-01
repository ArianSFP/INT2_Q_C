from __future__ import annotations

import hashlib
import math
import unittest

import numpy as np

from . import tail_peeling_composite as oracle


class ExactCodeTests(unittest.TestCase):
    def test_enumerative_lengths_are_exact(self) -> None:
        for n in range(1, 18):
            for k in range(n + 1):
                cardinality = math.comb(n, k)
                expected = 0 if cardinality <= 1 else math.ceil(math.log2(cardinality))
                self.assertEqual(oracle.ceil_log2_binomial(n, k), expected)

    def test_huffman_weighted_path_length(self) -> None:
        self.assertEqual(oracle.huffman_payload_bits([]), 0)
        self.assertEqual(oracle.huffman_payload_bits([9]), 0)
        self.assertEqual(oracle.huffman_payload_bits([1, 1]), 2)
        self.assertEqual(oracle.huffman_payload_bits([5, 2, 1]), 11)
        self.assertEqual(oracle.huffman_payload_bits([4, 3, 2, 1]), 19)

    def test_lossless_value_mode_never_exceeds_literal(self) -> None:
        words = np.asarray(
            [0x3F80] * 100 + [0xBF80] * 100 + [0x4000] * 3,
            dtype=np.uint16,
        )
        counts = np.bincount(words.astype(np.int64), minlength=1 << 16)
        row = oracle.best_value_code(counts)
        self.assertLessEqual(row["total_bits"], 16 * words.size)
        self.assertIn(row["mode"], row["all_mode_total_bits"])

    def test_bf16_rne_is_deterministic(self) -> None:
        values = np.asarray([0.0, -0.0, 1.0, -2.25, 1.001], dtype=np.float32)
        left = oracle.float32_to_bf16_rne_words(values)
        right = oracle.float32_to_bf16_rne_words(values.copy())
        np.testing.assert_array_equal(left, right)
        self.assertTrue(np.all(np.isfinite(oracle.bf16_words_to_float32(left))))


class GeometryTests(unittest.TestCase):
    def test_stable_tail_order_breaks_ties_by_ordinal(self) -> None:
        values = np.asarray([2.0, -3.0, 3.0, -2.0, 1.0], dtype=np.float32)
        self.assertEqual(oracle.stable_tail_order(values, np).tolist(), [1, 2, 0, 3, 4])

    def test_support_pattern_xklt_conserves_dimension_and_energy(self) -> None:
        categories = 3
        bins = categories**3
        counts = np.zeros(bins, dtype=np.int64)
        grams = np.zeros((bins, 3, 3), dtype=np.float64)
        # All three roles survive choice zero in this category.
        code = (2 * categories + 2) * categories + 2
        counts[code] = 10
        covariance = np.asarray(
            [[20.0, 2.0, 1.0], [2.0, 10.0, -0.5], [1.0, -0.5, 5.0]]
        )
        grams[code] = covariance
        table = oracle.SupportTable(0, categories, counts, grams, {})
        components, angles = table.components((0, 0, 0))
        self.assertEqual(angles, 3)
        self.assertEqual(sum(row.dimension for row in components), 30)
        self.assertAlmostEqual(sum(row.energy for row in components), np.trace(covariance))

    def test_integer_waterfill_closes_exactly(self) -> None:
        components = [
            oracle.Component("a", 0, 13, 20.0),
            oracle.Component("b", 1, 17, 7.0),
            oracle.Component("c", 2, 19, 1.0),
        ]
        for budget in (1, 17, 101, 1000):
            row = oracle.integer_waterfill(components, budget)
            self.assertEqual(row["payload_bits"], budget)
            self.assertEqual(
                sum(item["payload_bits"] for item in row["allocations"]), budget
            )
            self.assertGreater(row["distortion_sse"], 0.0)

    def test_lagrange_dual_is_below_every_discrete_primal(self) -> None:
        choices = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.uint8)
        banks = []
        for expert in range(oracle.EXPERTS):
            banks.append(
                oracle.DualOptionBank(
                    expert_ordinal=expert,
                    geometry="fixture",
                    choices=choices,
                    side_bits=np.asarray([10.0, 30.0]),
                    dimensions=np.asarray([[100.0], [90.0]]),
                    energies=np.asarray([[100.0], [80.0]]),
                )
            )
        physical_bits = oracle.COMMON_PREFIX_BITS + 1200
        dual = oracle.dual_point(banks, theta=0.25, physical_bits=physical_bits)
        primal_values = []
        for selection in range(1 << oracle.EXPERTS):
            components = []
            side = oracle.COMMON_PREFIX_BITS
            for expert in range(oracle.EXPERTS):
                option = (selection >> expert) & 1
                side += int(banks[expert].side_bits[option])
                components.append(
                    oracle.Component(
                        f"e{expert}",
                        expert,
                        int(banks[expert].dimensions[option, 0]),
                        float(banks[expert].energies[option, 0]),
                    )
                )
            budget = physical_bits - side
            primal_values.append(oracle.integer_waterfill(components, budget)["distortion_sse"])
        self.assertLessEqual(
            dual["dual_lower_bound_sse"], min(primal_values) + 1e-10
        )


class PhysicalLedgerTests(unittest.TestCase):
    @staticmethod
    def fake_panel() -> oracle.PanelStats:
        matrix_candidates = []
        for _ in range(oracle.MATRICES):
            rows = []
            for index, k in enumerate(oracle.TAIL_COUNTS):
                fraction = k / oracle.VALUES_PER_MATRIX
                rows.append(
                    oracle.TailCandidate(
                        candidate_index=index,
                        k=k,
                        tail_energy=fraction,
                        residual_energy=1.0 - fraction,
                        mask_bits=oracle.ceil_log2_binomial(oracle.VALUES_PER_MATRIX, k),
                        value_bits=16 * k,
                        value_mode="literal16",
                        value_detail={
                            "mode": "literal16",
                            "total_bits": 16 * k,
                            "payload_bits": 16 * k,
                            "model_bits": 0,
                            "distinct_symbols": min(k, 1),
                            "all_mode_total_bits": {"literal16": 16 * k},
                            "mode_bits_in_matrix_descriptor": 2,
                        },
                    )
                )
            matrix_candidates.append(rows)
        return oracle.PanelStats(
            label="fixture",
            matrix_candidates=matrix_candidates,
            matrix_energies=[1.0] * oracle.MATRICES,
            matrix_receipts=[{"matrix_ordinal": i} for i in range(oracle.MATRICES)],
            support_tables=[],
            total_energy=float(oracle.MATRICES),
        )

    def test_raw_physical_and_read_closure(self) -> None:
        row = oracle.score_configuration(
            self.fake_panel(),
            [0] * oracle.MATRICES,
            geometry="raw",
            requested_rate=2.15,
            include_allocations=True,
        )
        self.assertTrue(row["valid"])
        self.assertLessEqual(row["physical_rate_bpw"], 2.15)
        self.assertEqual(row["side_bits"] + row["payload_bits"], row["physical_bits"])
        self.assertEqual(row["read_ledger"]["bit_closure"], row["physical_bits"])
        self.assertTrue(row["read_ledger"]["below_2x"])
        self.assertLess(row["read_ledger"]["maximum_cold_amplification"], 1.1)

    def test_seed_is_domain_separated(self) -> None:
        first = oracle.stable_seed("domain-a", 0)
        second = oracle.stable_seed("domain-b", 0)
        self.assertNotEqual(first, second)
        self.assertEqual(first, oracle.stable_seed("domain-a", 0))
        self.assertEqual(
            hashlib.sha256(b"domain-a\x000").digest()[:8], first.to_bytes(8, "little")
        )


if __name__ == "__main__":
    unittest.main()
