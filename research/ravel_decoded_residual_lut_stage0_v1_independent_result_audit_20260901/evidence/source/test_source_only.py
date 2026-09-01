#!/usr/bin/env python3
"""Hostile source-only tests for RAVEL-6144-v1."""
from __future__ import annotations

import importlib.util
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = load("ravel_v1_verify_source", HERE / "verify_source.py")
PACKET = load("ravel_v1_packet_codec_tests", HERE / "packet_codec.py")


class SourceTests(unittest.TestCase):
    def copied(self, root: Path) -> Path:
        target = root / "package"
        shutil.copytree(HERE, target)
        return target

    def test_exact_source_closure(self) -> None:
        self.assertEqual(VERIFY.verify(HERE), 89)

    def test_weighted_raw_sse_counterexample(self) -> None:
        scales = (1.0, 2.0)
        residuals = (1.0, 1.0)
        exact = sum(s * r for s, r in zip(scales, residuals)) / sum(s * s for s in scales)
        old = sum(r / s for s, r in zip(scales, residuals)) / len(scales)
        exact_sse = sum((r - exact * s) ** 2 for s, r in zip(scales, residuals))
        old_sse = sum((r - old * s) ** 2 for s, r in zip(scales, residuals))
        self.assertEqual(exact, 0.6)
        self.assertEqual(exact_sse, 0.2)
        self.assertGreater(old_sse, exact_sse)

    def test_packet_roundtrip_alignment_and_padding(self) -> None:
        values = [math.sin(index) for index in range(PACKET.ENTRIES)]
        packet = PACKET.build_packet(values)
        parsed = PACKET.parse_packet(packet)
        self.assertEqual(len(packet), 16384)
        self.assertEqual(parsed["header"]["table_offset"], 4096)
        self.assertEqual(len(parsed["values"]), 6144)
        self.assertTrue(all(math.isfinite(value) for value in parsed["values"]))
        self.assertEqual(packet[parsed["header_json_bytes"] + 1:4096],
                         bytes(4096 - parsed["header_json_bytes"] - 1))

    def test_nonzero_packet_padding_rejected(self) -> None:
        packet = bytearray(PACKET.build_packet([0.0] * PACKET.ENTRIES))
        newline = packet.index(10)
        packet[newline + 1] = 1
        with self.assertRaises(PACKET.PacketError):
            PACKET.parse_packet(bytes(packet))

    def test_packet_table_mutation_rejected(self) -> None:
        packet = bytearray(PACKET.build_packet([0.0] * PACKET.ENTRIES))
        packet[-1] ^= 1
        with self.assertRaises(PACKET.PacketError):
            PACKET.parse_packet(bytes(packet))

    def test_nonfinite_and_overflow_packet_values_rejected(self) -> None:
        with self.assertRaises(PACKET.PacketError):
            PACKET.build_packet([float("inf")] + [0.0] * (PACKET.ENTRIES - 1))
        with self.assertRaises(PACKET.PacketError):
            PACKET.build_packet([1.0e100] + [0.0] * (PACKET.ENTRIES - 1))

    def test_noncyclic_boundary_and_role_stride(self) -> None:
        first = PACKET.reference_scalar_index(0, [1.0, 2.0, 3.0], 0, 2.0)
        changed_last_sign = PACKET.reference_scalar_index(0, [1.0, 2.0, -3.0], 0, 2.0)
        self.assertEqual(first, changed_last_sign)
        self.assertEqual(PACKET.reference_scalar_index(1, [1.0, 2.0, 3.0], 0, 2.0) - first, 2048)

    def test_extra_entry_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            package = self.copied(Path(raw))
            (package / "extra").mkdir()
            with self.assertRaises(VERIFY.Failure):
                VERIFY.verify(package)

    def test_source_mutation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            package = self.copied(Path(raw))
            with (package / "ravel_stage0.py").open("ab") as stream:
                stream.write(b"\n# mutation\n")
            with self.assertRaises(VERIFY.Failure):
                VERIFY.verify(package)

    def test_symlink_member_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            package = self.copied(root)
            target = root / "outside.json"
            member = package / "design_lock.json"
            target.write_bytes(member.read_bytes())
            member.unlink()
            try:
                os.symlink(target, member)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(VERIFY.Failure):
                VERIFY.verify(package)


if __name__ == "__main__":
    unittest.main(verbosity=2)
