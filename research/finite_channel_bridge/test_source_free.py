from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

import bridge_ledger as ledger
import synthetic_rotated_dither_gate as gate


class SourceFreeTests(unittest.TestCase):
    def test_exact_ledgers_stay_inside_rate_and_read_caps(self) -> None:
        table_bytes = 32 * gate.INTERNAL_NODES * 2 + 256
        for model in (ledger.SILWARP_FP16_BYTES, ledger.SILWARP_INT8_BYTES):
            for experts in (6, 128):
                for rate in ledger.RATE_FRACTIONS.values():
                    row = ledger.exact_budget(
                        rate, experts=experts, global_bytes=model + table_bytes
                    )
                    self.assertLessEqual(row["physical_bpw"], float(rate))
                    self.assertLess(row["cold_read_amplification"], 2.0)
                    self.assertGreaterEqual(row["unused_cap_bytes"], 0)

    def test_known_fp16_nine_stream_ledger_without_table(self) -> None:
        row = ledger.exact_budget(
            Fraction(5, 2), experts=128, global_bytes=ledger.SILWARP_FP16_BYTES
        )
        self.assertEqual(row["block_capacity_bytes"], 163_412)
        self.assertAlmostEqual(row["physical_bpw"], 2.499986516104804)
        self.assertAlmostEqual(row["cold_read_amplification"], 1.320055134116949)

    def test_rht_roundtrip(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.standard_normal(1 << 12).astype(np.float32)
        _dither, signs = gate.public_dither_and_signs(x.size, 9)
        y = gate.inverse_rht(gate.forward_rht(x, signs, np), signs, np)
        np.testing.assert_allclose(y, x, rtol=2e-6, atol=2e-6)

    def test_arithmetic_symbol_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        n = 4096
        dither, _signs = gate.public_dither_and_signs(n, 12)
        phases = gate.phases_for_dither(dither, 16)
        table = gate.build_frequency_table(16, quadrature=8)
        q = np.clip(np.rint(rng.standard_normal(n) * 1.3), -gate.K, gate.K).astype(
            np.int16
        )
        symbols, escapes = gate.q_to_symbols(q)
        self.assertEqual(escapes, 0)
        bits, frequencies = gate.decisions_for_symbols(symbols, phases, table)
        payload, logical = gate.arithmetic_encode_binary(bits, frequencies)
        decoded = gate.decode_symbols(n, phases, table, payload, logical)
        np.testing.assert_array_equal(decoded, symbols)

    def test_scalar_shaping_gap(self) -> None:
        self.assertAlmostEqual(
            ledger.SHAPING_GAPS["scalar_z_G_1over12"], 0.25461433482006296
        )


if __name__ == "__main__":
    unittest.main()
