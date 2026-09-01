#!/usr/bin/env python3
"""Hostile standard-library tests for the independent CCQ result audit."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "research/ccq_raw_mse_stage0_v0_runpod_result_20260901"
_SPEC = importlib.util.spec_from_file_location("ccq_independent_audit", AUDIT / "verify_audit.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load independent verifier")
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


class IndependentAuditTests(unittest.TestCase):
    def test_full_evidence_without_audit_self_manifest(self) -> None:
        count = verifier.verify(ROOT, AUDIT, verify_audit_closure=False)
        self.assertGreater(count, 500)

    def test_exact_packets(self) -> None:
        source = (RESULT / "source_prefix.bin").read_bytes()
        gaussian = (RESULT / "gaussian_prefix.bin").read_bytes()
        self.assertEqual(verifier.parse_packet(source, "source")["bytes"], 7_518_592)
        self.assertEqual(verifier.parse_packet(gaussian, "gaussian")["bytes"], 7_518_592)

    def test_rate_arithmetic(self) -> None:
        rows6, prefix6, required6 = verifier.rate_ledger(6)
        self.assertAlmostEqual(prefix6, 2.1245298032407407)
        self.assertAlmostEqual(required6, 0.0420722358191473)
        self.assertEqual([row["physical_bytes"] for row in rows6], [7_608_730, 8_139_572, 8_847_360])
        self.assertTrue(all(float(row["cold_read_amplification"]) < 2.0 for row in rows6))
        rows128, prefix128, required128 = verifier.rate_ledger(128)
        self.assertAlmostEqual(prefix128, 2.1234266493055554)
        self.assertAlmostEqual(required128, 0.04213662594769002)
        self.assertTrue(all(float(row["cold_read_amplification"]) < 2.0 for row in rows128))

    def test_packet_trailing_byte_rejected(self) -> None:
        payload = (RESULT / "source_prefix.bin").read_bytes() + b"\0"
        with self.assertRaises(verifier.Failure):
            verifier.parse_packet(payload, "source")

    def test_packet_global_padding_rejected(self) -> None:
        payload = bytearray((RESULT / "source_prefix.bin").read_bytes())
        payload[4095] ^= 1
        with self.assertRaises(verifier.Failure):
            verifier.parse_packet(bytes(payload), "source")

    def test_packet_expert_padding_rejected(self) -> None:
        payload = bytearray((RESULT / "source_prefix.bin").read_bytes())
        payload[4096 + 63] ^= 1
        with self.assertRaises(verifier.Failure):
            verifier.parse_packet(bytes(payload), "source")

    def test_packet_label_rejected(self) -> None:
        payload = (RESULT / "source_prefix.bin").read_bytes()
        with self.assertRaises(verifier.Failure):
            verifier.parse_packet(payload, "gaussian")

    def test_packet_nonfinite_float32_rejected(self) -> None:
        payload = bytearray((RESULT / "source_prefix.bin").read_bytes())
        first_code_scale = 4096 + 64 + 393_216 + 12_288
        struct.pack_into("<I", payload, first_code_scale, 0x7F800000)
        with self.assertRaises(verifier.Failure):
            verifier.parse_packet(bytes(payload), "source")

    def test_packet_nonpositive_float16_rejected(self) -> None:
        payload = bytearray((RESULT / "source_prefix.bin").read_bytes())
        first_super_scale = 4096 + 64 + 393_216 + 12_288 + 768 * 8
        struct.pack_into("<H", payload, first_super_scale, 0)
        with self.assertRaises(verifier.Failure):
            verifier.parse_packet(bytes(payload), "source")

    def test_json_duplicate_key_rejected(self) -> None:
        with self.assertRaises(verifier.Failure):
            verifier.parse_json_bytes(b'{"a":1,"a":2}', "duplicate fixture")

    def test_json_exponent_overflow_rejected(self) -> None:
        with self.assertRaises(verifier.Failure):
            verifier.parse_json_bytes(b'{"a":1e9999}', "overflow fixture")

    def test_json_nonstandard_constant_rejected(self) -> None:
        with self.assertRaises(verifier.Failure):
            verifier.parse_json_bytes(b'{"a":NaN}', "NaN fixture")

    def test_manifest_without_final_lf_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "MANIFEST.sha256"
            path.write_bytes(b"0" * 64 + b"  member")
            with self.assertRaises(verifier.Failure):
                verifier.parse_manifest(path)

    def test_result_extra_member_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)
            for name in verifier.RESULT_HASHES:
                (path / name).write_bytes(b"")
            (path / "extra").write_bytes(b"")
            with self.assertRaises(verifier.Failure):
                verifier.check_exact_closure(path, set(verifier.RESULT_HASHES), verifier.Checks(), "fixture")

    def test_real_symlink_rejected_or_explicit_skip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)
            target = path / "target"
            target.write_bytes(b"x")
            link = path / "source_prefix.bin"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"platform does not permit test symlink: {exc}")
            (path / "gaussian_prefix.bin").write_bytes(b"")
            (path / "result.json").write_bytes(b"")
            with self.assertRaises(verifier.Failure):
                verifier.check_exact_closure(path, set(verifier.RESULT_HASHES), verifier.Checks(), "fixture")


if __name__ == "__main__":
    unittest.main(verbosity=2)
