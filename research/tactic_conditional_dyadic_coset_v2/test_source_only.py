#!/usr/bin/env python3
"""Hostile standard-library tests for the TACTIC-DH384 v2 source freeze."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tactic_v2_common import (
    ContractError,
    HeldAbsolute,
    HeldOutput,
    cpu_projection,
    selector_packet,
    sha256_bytes,
    strict_json_loads,
    universal_selector_table,
)
from verify_source import verify_design, verify_package


PACKAGE = Path(__file__).resolve().parent


class SourceOnlyTests(unittest.TestCase):
    def test_01_frozen_package_passes(self) -> None:
        result = verify_package(PACKAGE)
        self.assertEqual(result["status"], "PASS_SOURCE_ONLY_NO_EXECUTION_AUTHORITY")
        self.assertEqual(result["arithmetic"]["physical_bpw"], 2.5)
        self.assertLess(result["arithmetic"]["cold_read_amplification"], 2.0)

    def test_02_design_arithmetic_direct(self) -> None:
        design = strict_json_loads((PACKAGE / "design_lock.json").read_bytes())
        result = verify_design(design)
        self.assertAlmostEqual(result["planning_c_required"], 0.2972443434920543, places=14)
        self.assertEqual(result["coarse_bpw"], 307 / 128)

    def test_03_duplicate_and_nonfinite_json_rejected(self) -> None:
        for payload in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":-1e999}'):
            with self.subTest(payload=payload):
                with self.assertRaises(ContractError):
                    strict_json_loads(payload)

    def test_04_selector_is_bounded_and_deterministic(self) -> None:
        first = universal_selector_table()
        self.assertEqual(first, universal_selector_table())
        self.assertEqual(len(first), 3072)
        self.assertTrue(all(0 <= value < 8 for value in first))
        self.assertEqual(len(selector_packet(first)), 16_384)
        changed = bytearray(first)
        changed[0] ^= 1
        with self.assertRaises(ContractError):
            selector_packet(bytes(changed))

    def test_05_cpu_dyadic_norm_and_projection(self) -> None:
        table = universal_selector_table()
        symbols = [((index * 73 + 19) % 511) - 255 for index in range(64)]
        error = [((index * 29 + 11) % 257 - 128) / 127.0 for index in range(64)]
        result = cpu_projection(symbols, error, role=1, table=table, rank=6)
        self.assertAlmostEqual(result["energy"], result["transformed_energy"], places=11)
        self.assertAlmostEqual(result["energy"] - result["projected_energy"],
                               result["residual_energy"], places=11)
        self.assertGreaterEqual(result["projected_energy"], 0.0)

    def test_06_extra_member_fails_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "package"
            shutil.copytree(PACKAGE, copy)
            (copy / "unexpected.bin").write_bytes(b"x")
            with self.assertRaises(ContractError):
                verify_package(copy)

    def test_07_member_mutation_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "package"
            shutil.copytree(PACKAGE, copy)
            with (copy / "README.md").open("ab") as handle:
                handle.write(b"\nmutation\n")
            with self.assertRaises(ContractError):
                verify_package(copy)

    def test_08_manifest_row_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "package"
            shutil.copytree(PACKAGE, copy)
            path = copy / "SOURCE_MANIFEST.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["files"][0]["bytes"] += 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ContractError):
                verify_package(copy)

    def test_09_symlink_member_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "package"
            shutil.copytree(PACKAGE, copy)
            target = copy / "README.md"
            backup = copy / "README.real"
            target.rename(backup)
            try:
                target.symlink_to(backup.name)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ContractError):
                verify_package(copy)

    @unittest.skipUnless(os.name == "posix", "held-FD contract is RunPod/POSIX-only")
    def test_10_held_regular_descriptor_and_create_new(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.bin"
            source.write_bytes(b"held-data")
            with HeldAbsolute(source, want_directory=False) as held:
                self.assertGreaterEqual(held.fd, 0)
            output_path = root / "new-output"
            with HeldOutput(output_path) as output:
                output.write_new("receipt.bin", b"ok")
                with self.assertRaises(FileExistsError):
                    output.write_new("receipt.bin", b"again")
            self.assertEqual((output_path / "receipt.bin").read_bytes(), b"ok")

    def test_11_source_hash_is_stable(self) -> None:
        payload = (PACKAGE / "design_lock.json").read_bytes()
        self.assertEqual(len(sha256_bytes(payload)), 64)
        self.assertEqual(sha256_bytes(payload), sha256_bytes(payload))


if __name__ == "__main__":
    unittest.main()
