#!/usr/bin/env python3
"""Run the source-only SILT hostile suite without accepting an input payload."""

from __future__ import annotations

import argparse
import json
import os
import time
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail rather than skip if the mandatory CuPy optimization path is unavailable",
    )
    parser.add_argument(
        "--require-rtx-5090",
        action="store_true",
        help="also require that the CUDA device reports RTX 5090",
    )
    arguments = parser.parse_args()
    if arguments.require_rtx_5090 and not arguments.require_gpu:
        parser.error("--require-rtx-5090 requires --require-gpu")
    os.environ["SILT_REQUIRE_CUPY_TEST"] = "1" if arguments.require_gpu else "0"
    os.environ["SILT_REQUIRE_RTX5090"] = "1" if arguments.require_rtx_5090 else "0"
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromName("test_source_only")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "schema": "silt-source-only-verifier-receipt-v0",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "gpu_required": arguments.require_gpu,
        "rtx_5090_required": arguments.require_rtx_5090,
        "elapsed_seconds": time.perf_counter() - started,
        "payload_input_accepted": False,
        "source_gain_claim": False,
        "result_frozen": False,
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

