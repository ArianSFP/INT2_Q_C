#!/usr/bin/env python3
"""Run the frozen source-only reproducibility review and emit one receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW))
from independent_auth import (authenticate_external_sources,
                              authenticate_producer)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_names: list[str] = []
        self.skipped_names: list[dict[str, str]] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_names.append(test.id())

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.skipped_names.append({"test": test.id(), "reason": str(reason)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    producer_auth = authenticate_producer(args.producer)
    external_auth = authenticate_external_sources(args.external_root)
    os.environ["STRATA_RM_V1_REVIEW_PRODUCER"] = str(args.producer.resolve())
    os.environ["STRATA_RM_V1_REVIEW_EXTERNAL_ROOT"] = str(
        args.external_root.resolve())
    suite = unittest.defaultTestLoader.discover(
        str(REVIEW), pattern="test_reproducibility_review.py",
        top_level_dir=str(REVIEW))
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result: RecordingResult = runner.run(suite)  # type: ignore[assignment]
    findings = [
        "standalone verify_source resolves the package before checking whether the supplied root is a symlink; authority.authenticate_v1_package does reject it",
        "current external modules are hash-checked and then imported from their original mutable paths rather than from an authenticated snapshot",
        "the physical decoder is hash-checked but executed from its external path rather than an immutable snapshot",
        "the production decoder-audit manifest/root are supplied inside the same experiment commitment; no separate out-of-band decoder-audit pin or audit-execution receipt is required",
        "decoder source-access and packet-read claims are receipt/trace fields, not operating-system-enforced file-access instrumentation",
        "Qwen identity, Gaussian-control generation, architecture-family identity, and model-family agnosticism are commitment labels without checkpoint or generator provenance",
        "target rate/F/read aggregation is over Qwen cases only; the second architecture family is required to exist but need not itself meet the target",
        "matched controls are required to pair geometrically but are not used in a Qwen-minus-control acceptance comparison",
        "the real-CuPy worker has strong isolated-runtime checks, but its CPU and GPU order implementations share the same frozen rm_order module and remain trusted-runner evidence, not hardware attestation",
    ]
    passed = result.wasSuccessful()
    receipt = {
        "schema": "strata-rm-global-swap-v1-hardened-reproducibility-review",
        "producer_auth": producer_auth, "external_auth": external_auth,
        "tests_run": result.testsRun, "passed_tests": sorted(result.passed_names),
        "skipped_tests": result.skipped_names,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "correctness_and_authority_gaps": findings,
        "payloads_opened": 0, "model_data_accessed": False,
        "network_accessed": False, "qwen_result": None,
        "passed": passed,
        "status": (
            "PASS_SOURCE_REPRODUCIBILITY_REVIEW__GAPS_RECORDED__HOLD_PAYLOAD_AND_PHYSICAL_RESULT"
            if passed else "FAIL_SOURCE_REPRODUCIBILITY_REVIEW"),
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True,
                         ensure_ascii=True, allow_nan=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
