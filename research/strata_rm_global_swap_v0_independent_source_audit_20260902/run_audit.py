#!/usr/bin/env python3
"""Execute the independent no-payload audit and emit a structured receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT_DIR))
from independent_auth import authenticate_external, authenticate_source


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
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_auth = authenticate_source(args.source)
    external_auth = authenticate_external(args.external_root)
    os.environ["STRATA_RM_GLOBAL_SOURCE"] = str(args.source.resolve())
    os.environ["STRATA_RM_GLOBAL_EXTERNAL_ROOT"] = str(args.external_root.resolve())
    suite = unittest.defaultTestLoader.discover(
        str(AUDIT_DIR), pattern="test_hostile_audit.py", top_level_dir=str(AUDIT_DIR))
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result: RecordingResult = runner.run(suite)  # type: ignore[assignment]
    receipt = {
        "schema": "strata-rm-global-swap-v0-independent-source-audit",
        "source_auth": source_auth,
        "external_auth": external_auth,
        "tests_run": result.testsRun,
        "passed_tests": sorted(result.passed_names),
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "expected_boundary_exposures": [
            "producer source verifier ignores unmanifested directories",
            "install hook does not authenticate module/function object identity",
            "physical result validator accepts self-declared no-packet receipts",
            "canonical replay is declared hash equality rather than byte replay",
            "physical result validator has no MSE/F/control/read/universality authority",
            "CuPy smoke accepts an injected NumPy facade",
        ],
        "payloads_opened": 0,
        "passed": result.wasSuccessful(),
        "status": (
            "PASS_STATED_SOURCE_MECHANISM__HOLD_PAYLOAD_INTEGRATION_AND_PHYSICAL_RESULT"
            if result.wasSuccessful() else "FAIL_INDEPENDENT_SOURCE_AUDIT"
        ),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
