#!/usr/bin/env python3
"""Hostile source-only tests for the RAVEL package verifier."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().with_name("verify_source.py")
SPEC = importlib.util.spec_from_file_location("ravel_verify", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load verifier")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class SourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.package = Path(__file__).resolve().parent

    def copy(self, root: Path) -> Path:
        target = root / "package"
        shutil.copytree(self.package, target)
        return target

    def test_exact(self) -> None:
        self.assertGreater(verifier.verify(self.package), 40)

    def test_extra_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            package = self.copy(Path(raw))
            (package / "extra").write_text("x", encoding="ascii")
            with self.assertRaises(verifier.Failure):
                verifier.verify(package)

    def test_mutation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            package = self.copy(Path(raw))
            with (package / "ravel_stage0.py").open("ab") as stream:
                stream.write(b"\n# mutation\n")
            with self.assertRaises(verifier.Failure):
                verifier.verify(package)

    def test_design_arithmetic(self) -> None:
        design = __import__("json").loads((self.package / "design_lock.json").read_text(encoding="utf-8"))
        self.assertEqual(design["architecture"]["table_entries"], 6144)
        self.assertLess(design["rate_and_read"]["conservative_cold_page_read_amplification"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

