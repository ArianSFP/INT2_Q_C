#!/usr/bin/env python3
"""Source-only regression tests; these never import CuPy or open Qwen payloads."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("kbvq_verify_design", HERE / "verify_design.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load source verifier")
verify_design = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_design
SPEC.loader.exec_module(verify_design)


class SourceOnlyTests(unittest.TestCase):
    def test_sealed_package_verifies(self) -> None:
        result = verify_design.verify(HERE)
        self.assertEqual(result["status"], "PASS")
        self.assertGreater(result["best_two_role_F"], 0.8)
        self.assertLess(result["best_two_role_cold_page_amplification"], 2.0)

    def test_cupy_is_not_top_level(self) -> None:
        tree = ast.parse((HERE / "stage0_gate.py").read_text(encoding="utf-8"))
        top = {
            alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("cupy", top)

    def test_mutated_design_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "package"
            target.mkdir()
            for source in HERE.iterdir():
                (target / source.name).write_bytes(source.read_bytes())
            design = json.loads((target / "design_lock.json").read_text(encoding="utf-8"))
            design["objective"]["F_max"] = 0.81
            (target / "design_lock.json").write_text(json.dumps(design), encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify_design.verify(target)


if __name__ == "__main__":
    unittest.main()
