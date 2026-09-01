#!/usr/bin/env python3
"""Hostile tests for the independent RAVEL-v1 result audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ravel_v1_independent_result_audit", HERE / "verify_audit.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load verifier")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditTests(unittest.TestCase):
    def copied(self, root: Path) -> Path:
        target = root / "audit"
        shutil.copytree(HERE, target)
        return target

    def test_exact_audit(self) -> None:
        count = AUDIT.verify(HERE)
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_packet_parser_recomputes_finite_payload(self) -> None:
        raw = (HERE / "evidence" / "result" / "fit_table_packet.bin").read_bytes()
        checks = AUDIT.Checks()
        header = AUDIT.parse_packet(raw, checks)
        self.assertEqual(header["entries"], 6144)
        self.assertGreater(checks.count, 10)

    def test_nonzero_packet_padding_rejected(self) -> None:
        packet = bytearray((HERE / "evidence" / "result" / "fit_table_packet.bin").read_bytes())
        newline = packet.index(10)
        packet[newline + 1] = 1
        with self.assertRaises(AUDIT.Failure):
            AUDIT.parse_packet(bytes(packet), AUDIT.Checks())

    def test_nonfinite_fp16_rejected_even_with_updated_table_hash(self) -> None:
        packet = bytearray((HERE / "evidence" / "result" / "fit_table_packet.bin").read_bytes())
        packet[4096:4098] = b"\x00\x7c"
        table_hash = hashlib.sha256(packet[4096:]).hexdigest()
        newline = packet.index(10)
        header = json.loads(packet[:newline].decode("ascii"))
        header["table_sha256"] = table_hash
        encoded = AUDIT.canonical(header) + b"\n"
        packet[:4096] = encoded + bytes(4096 - len(encoded))
        with self.assertRaises(AUDIT.Failure):
            AUDIT.parse_packet(bytes(packet), AUDIT.Checks())

    def test_duplicate_and_nonfinite_json_rejected(self) -> None:
        with self.assertRaises(AUDIT.Failure):
            AUDIT.strict_json(b'{"a":1,"a":2}')
        with self.assertRaises(AUDIT.Failure):
            AUDIT.strict_json(b'{"a":NaN}')

    def test_extra_directory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copied(Path(raw))
            (copy / "extra-empty-directory").mkdir()
            with self.assertRaises(AUDIT.Failure):
                AUDIT.verify(copy)

    def test_evidence_mutation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copied(Path(raw))
            with (copy / "evidence" / "result" / "result.json").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaises(AUDIT.Failure):
                AUDIT.verify(copy)

    def test_symlink_evidence_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            copy = self.copied(root)
            member = copy / "evidence" / "result" / "COMPLETE.json"
            outside = root / "outside.json"
            outside.write_bytes(member.read_bytes())
            member.unlink()
            try:
                os.symlink(outside, member)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(AUDIT.Failure):
                AUDIT.verify(copy)

    def test_receipt_records_oracle_replay_limitation(self) -> None:
        receipt = AUDIT.strict_json((HERE / "audit_receipt.json").read_bytes())
        self.assertFalse(receipt["limitations"]["weighted_oracle_sse_replayable_without_payload"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
