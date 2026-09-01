#!/usr/bin/env python3
"""Hostile standard-library tests for the independent RAVEL-v0 audit."""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ravel_independent_audit", HERE / "verify_audit.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load independent verifier")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditTests(unittest.TestCase):
    def copied(self, root: Path) -> Path:
        target = root / "audit"
        shutil.copytree(HERE, target)
        return target

    def test_exact_sealed_audit(self) -> None:
        self.assertEqual(AUDIT.verify(HERE), AUDIT.strict_json((HERE / "audit_receipt.json").read_bytes())["verifier_check_count"])

    def test_projection_counterexample_is_strict(self) -> None:
        row = AUDIT.projection_counterexample()
        self.assertEqual(row["unweighted_normalized_mean"], 0.75)
        self.assertEqual(row["weighted_raw_sse_optimum"], 0.6)
        self.assertGreater(row["unweighted_raw_sse"], row["optimal_raw_sse"])

    def test_packet_layout_is_odd_and_unroundtripped(self) -> None:
        self.assertEqual(AUDIT.packet_layout(), {
            "header_bytes": 79,
            "table_bytes": 12288,
            "padding_bytes": 4017,
            "table_offset_mod_2": 1,
        })

    def test_extra_directory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copied(Path(raw))
            (copy / "empty-extra-directory").mkdir()
            with self.assertRaises(AUDIT.AuditFailure):
                AUDIT.verify(copy)

    def test_evidence_mutation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copied(Path(raw))
            with (copy / "evidence" / "ravel_stage0.py").open("ab") as stream:
                stream.write(b"\n# hostile mutation\n")
            with self.assertRaises(AUDIT.AuditFailure):
                AUDIT.verify(copy)

    def test_manifest_substitution_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copied(Path(raw))
            text = (copy / "AUDIT_MANIFEST.json").read_text(encoding="utf-8")
            (copy / "AUDIT_MANIFEST.json").write_text(text.replace("closed_world", "closed_w0rld", 1), encoding="utf-8")
            with self.assertRaises(AUDIT.AuditFailure):
                AUDIT.verify(copy)

    def test_symlink_member_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = self.copied(Path(raw))
            victim = copy / "evidence" / "README.md"
            replacement = Path(raw) / "replacement.md"
            replacement.write_bytes(victim.read_bytes())
            victim.unlink()
            try:
                os.symlink(replacement, victim)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(AUDIT.AuditFailure):
                AUDIT.verify(copy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
