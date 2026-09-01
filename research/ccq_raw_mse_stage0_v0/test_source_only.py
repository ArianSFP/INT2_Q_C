#!/usr/bin/env python3
"""Hostile standard-library tests for the CCQ source-only verifier."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


_VERIFIER_PATH = Path(__file__).resolve().with_name("verify_source.py")
_SPEC = importlib.util.spec_from_file_location("ccq_source_verifier", _VERIFIER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load local verifier")
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


class SourceOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.package = Path(__file__).resolve().parent

    def copy_package(self, target: Path) -> Path:
        destination = target / "package"
        shutil.copytree(self.package, destination)
        return destination

    def test_exact_package_without_external_lock(self) -> None:
        self.assertGreater(verifier.verify(self.package.parent.parent, self.package, verify_lock=False), 80)

    def test_rate_arithmetic(self) -> None:
        rows, prefix, required = verifier.rate_ledger(6)
        self.assertAlmostEqual(prefix, 2.1245298032407407)
        self.assertAlmostEqual(required, 0.0420722358191473)
        self.assertEqual([row["physical"] for row in rows], [7608730, 8139572, 8847360])
        self.assertTrue(all(float(row["amp"]) < 2.0 for row in rows))

    def test_extra_member_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copy_package(Path(raw))
            (copy / "unexpected.txt").write_text("x", encoding="ascii")
            with self.assertRaises(verifier.Failure):
                verifier.verify(copy.parent, copy, verify_lock=False)

    def test_manifest_mutation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copy_package(Path(raw))
            with (copy / "README.md").open("ab") as stream:
                stream.write(b"mutation")
            with self.assertRaises(verifier.Failure):
                verifier.verify(copy.parent, copy, verify_lock=False)

    def test_directory_member_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copy_package(Path(raw))
            (copy / "nested").mkdir()
            with self.assertRaises(verifier.Failure):
                verifier.verify(copy.parent, copy, verify_lock=False)

    def test_real_symlink_rejected_or_explicit_skip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copy_package(Path(raw))
            link = copy / "link"
            try:
                link.symlink_to(copy / "README.md")
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"platform does not permit test symlink: {exc}")
            with self.assertRaises(verifier.Failure):
                verifier.verify(copy.parent, copy, verify_lock=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
