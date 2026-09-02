#!/usr/bin/env python3
"""Execute the no-payload v2 source gate and optional RunPod workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from authority_v2 import (V1_MANIFEST_SHA256, authenticate_v1_and_review,
                          authenticate_v2_package, canonical_json,
                          real_directory, regular_bytes, require,
                          run_current_integration_snapshot, run_snapshot_worker,
                          sha256)


V1_RM_ORDER_SHA256 = "e5d85d844633d206125a775efcd35711d02bf9eec5060715c17e8e7d50df0f92"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--v1-package", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--run-current-integration", action="store_true")
    parser.add_argument("--run-real-cupy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package_auth = authenticate_v2_package(args.package,
                                           args.expected_manifest_sha256)
    lineage = authenticate_v1_and_review(args.v1_package, args.review_package)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(package_auth["path"]) /
                                         "test_source_only.py")],
        cwd=Path(package_auth["path"]), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=600, check=False)
    require(completed.returncode == 0,
            "source-only hostile tests failed: " +
            completed.stderr.decode("utf-8", errors="replace")[-4000:])
    current = None
    parity = None
    if args.run_current_integration:
        require(args.external_root is not None,
                "current integration requires external root")
        current = run_current_integration_snapshot(
            package=args.package,
            expected_manifest_sha256=args.expected_manifest_sha256,
            external_root=args.external_root)
    if args.run_real_cupy:
        v1 = real_directory(args.v1_package, "v1 package for parity")
        payload = regular_bytes(v1 / "rm_order.py", "v1 rm_order.py")
        require(sha256(payload) == V1_RM_ORDER_SHA256,
                "v1 rm-order source pin")
        with tempfile.TemporaryDirectory(prefix="strata-rm-v2-parity-") as directory:
            root = Path(directory).resolve(strict=True)
            production = root / "v1_rm_order.py"
            production.write_bytes(payload)
            os.chmod(production, 0o444)
            parity = run_snapshot_worker(
                package=args.package,
                expected_manifest_sha256=args.expected_manifest_sha256,
                worker_name="parity_worker.py",
                worker_args=["--production-rm-order", str(production)],
                timeout_seconds=3600)
            require(sha256(regular_bytes(production,
                                         "post-parity v1 rm_order.py")) ==
                    V1_RM_ORDER_SHA256, "immutable v1 parity source")
    record = {
        "schema": "strata-rm-global-swap-v2-source-gate-receipt",
        "v2_manifest_sha256": args.expected_manifest_sha256,
        "v2_source_root_sha256": package_auth["source_root_sha256"],
        "v1_manifest_sha256": V1_MANIFEST_SHA256,
        "lineage": lineage,
        "source_only_tests": {"executed": True,
                              "stdout_tail": completed.stdout.decode(
                                  "utf-8", errors="replace")[-3000:],
                              "status": "PASS_SOURCE_ONLY_HOSTILE_TESTS"},
        "current_snapshot_integration": current,
        "independent_real_cupy_parity": parity,
        "payloads_opened": 0, "qwen_result": None, "rd_claim": False,
        "status": ("PASS_V2_SOURCE_AND_RUNPOD_MECHANISMS__HOLD_PAYLOAD"
                   if current is not None and parity is not None else
                   "PASS_V2_SOURCE_ONLY__OPTIONAL_RUNPOD_MECHANISMS_NOT_ALL_RUN"),
    }
    args.output.write_bytes(canonical_json(record) + b"\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
