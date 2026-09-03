#!/usr/bin/env python3
"""Run only the v4 source gate; accept no runtime, packet, or model path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from authority_v4 import (authenticate_v3_and_review, authenticate_v4_package,
                          canonical_json, require)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--v3-package", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v4 = authenticate_v4_package(args.package, args.expected_manifest_sha256)
    lineage = authenticate_v3_and_review(args.v3_package, args.review_package)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(v4["path"]) /
                                         "test_source_only.py")],
        cwd=Path(v4["path"]), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, check=False)
    require(completed.returncode == 0,
            "v4 hostile tests failed: " +
            completed.stderr.decode("utf-8", errors="replace")[-4000:])
    record = {
        "schema": "strata-rm-global-swap-v4-source-gate-receipt",
        "v4_manifest_sha256": args.expected_manifest_sha256,
        "v4_source_root_sha256": v4["source_root_sha256"],
        "lineage": lineage,
        "hostile_tests": {"executed": True,
                          "stdout_tail": completed.stdout.decode(
                              "utf-8", errors="replace")[-3000:],
                          "status": "PASS_V4_SOURCE_ONLY_HOSTILE_TESTS"},
        "wasmtime_imported": False, "wasm_guest_executed": False,
        "runtime_audit_package_opened": False,
        "semantic_decoder_audit_package_opened": False,
        "scientific_audit_package_opened": False,
        "model_data_accessed": False, "payloads_opened": 0,
        "qwen_result": None, "physical_result": None,
        "status": "PASS_V4_SOURCE_ONLY__RUNTIME_AND_PAYLOAD_HELD"}
    args.output.write_bytes(canonical_json(record) + b"\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
