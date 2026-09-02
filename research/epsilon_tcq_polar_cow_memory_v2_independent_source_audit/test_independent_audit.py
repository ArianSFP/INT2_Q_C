#!/usr/bin/env python3
"""Hostile tests for the independent v2 audit."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from independent_audit import (AuditError, authenticate_cupy_receipt,
                               authenticate_decoder, authenticate_source,
                               independent_memory, independent_work)


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "epsilon_tcq_polar_cow_memory_v2"
AUDITOR = HERE.parents[1] / "strata_v2_klt_mixed_independent_auditor_v1.py"
CUPY = HERE / "cupy_receipt.json"


class FormulaTests(unittest.TestCase):
    def test_expected_memory_table(self) -> None:
        expected = {4: 211952896, 8: 402794240,
                    16: 797059840, 32: 1610756864}
        for beam, allocated in expected.items():
            row = independent_memory(beam)
            self.assertEqual(row["aligned_peak_bytes"], allocated)
            self.assertTrue(row["passes_4gib_cap"])
            self.assertEqual(row["buffer_rows"], 34)

    def test_b32_work_counts(self) -> None:
        row = independent_work(32)
        self.assertEqual(row["likelihood_node_updates"], 8455716864)
        self.assertEqual(row["partial_sum_state_writes"], 4227858624)
        self.assertEqual(row["partial_sum_xors"], 3825205440)
        self.assertEqual(row["level_end_polar_xors"], 4227858432)
        self.assertEqual(row["lower_index_adds"], 402653184)
        self.assertEqual(row["branch_candidates_scored"], 805306110)
        self.assertEqual(row["survivor_tape_symbols_written"], 402653086)
        self.assertEqual(row["winner_replay_likelihood_node_updates"], 264241152)


class AuthenticationTests(unittest.TestCase):
    def test_real_pins(self) -> None:
        self.assertEqual(authenticate_source(SOURCE)["members"], 9)
        self.assertTrue(authenticate_decoder(AUDITOR)["six_level_major_passes_authenticated"])
        self.assertEqual(authenticate_cupy_receipt(CUPY)["maximum_pool_bytes"], 1610762240)

    def test_source_member_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source"
            shutil.copytree(SOURCE, target)
            path = target / "memory_plan.py"
            path.write_bytes(path.read_bytes() + b"\n# tamper\n")
            with self.assertRaises(AuditError):
                authenticate_source(target)

    def test_source_manifest_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source"
            shutil.copytree(SOURCE, target)
            path = target / "SOURCE_MANIFEST.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["source_root_sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(AuditError):
                authenticate_source(target)

    def test_decoder_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auditor.py"
            path.write_bytes(AUDITOR.read_bytes() + b"\n")
            with self.assertRaises(AuditError):
                authenticate_decoder(path)

    def test_cupy_receipt_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            value = json.loads(CUPY.read_text(encoding="utf-8"))
            value["beams"][-1]["passes_4gib_actual_pool"] = False
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(AuditError):
                authenticate_cupy_receipt(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
