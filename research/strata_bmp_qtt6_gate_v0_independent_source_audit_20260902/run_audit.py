#!/usr/bin/env python3
"""Run the independent payload-free hostile audit and emit one receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest

from independent_auth import authenticate_source


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_names: list[str] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_names.append(test.id())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = authenticate_source(args.source)
    os.environ["STRATA_BMP_QTT6_FROZEN_SOURCE"] = str(args.source.resolve())
    audit_dir = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        str(audit_dir), pattern="test_hostile_audit.py",
        top_level_dir=str(audit_dir),
    )
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result: RecordingResult = runner.run(suite)  # type: ignore[assignment]
    receipt = {
        "schema": "strata-bmp-qtt6-independent-source-audit-v0",
        "source_auth": auth,
        "tests_run": result.testsRun,
        "passed_tests": sorted(result.passed_names),
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "expected_boundary_exposures": [
            "BMP packets admit semantically equivalent factor gauges and ranks",
            "QTT packets admit semantically equivalent ranks and zero-state gauges",
            "canonical re-encode is byte/syntax canonicality, not function canonicality",
            "Geometry.validate accepts powers of two beyond uint16 packet fields",
            "CuPy import/device/build identity is not authenticated by the producer smoke",
            "the bounded search is NumPy; CuPy covers generated primitive smoke only",
            "the mechanism packet omits STRATA transform/scale/framing/page bytes",
            "there is no complete-codec 2.15--2.5 bpw or routed-read gate",
        ],
        "payloads_opened": 0,
        "qwen_claim": False,
        "passed_hostile_mechanics": result.wasSuccessful(),
        "disposition": (
            "HOLD_PRODUCTION__CORRECT_SEMANTIC_CANONICALITY_AND_UINT16_GEOMETRY__"
            "THEN_SEPARATELY_BIND_STRATA_CONTROL_SCORER_AND_READ_LEDGER"
            if result.wasSuccessful() else "FAIL_INDEPENDENT_SOURCE_AUDIT"
        ),
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
