#!/usr/bin/env python3
"""Authenticate source dependencies and optionally run isolated workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from authority import (authenticate_current_external_root,
                       authenticate_dependencies, run_isolated_worker)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--v0-package", type=Path, required=True)
    parser.add_argument("--v0-audit-package", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-current-integration", action="store_true")
    parser.add_argument("--run-real-cupy", action="store_true")
    args = parser.parse_args()
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(PACKAGE / "verify_source.py"),
         "--package", str(args.package.resolve()),
         "--expected-manifest-sha256", args.expected_manifest_sha256],
        check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ValueError("v1 source authentication: " + completed.stderr[-2000:])
    dependencies = authenticate_dependencies(args.v0_package,
                                               args.v0_audit_package)
    external = authenticate_current_external_root(args.external_root)
    workers = {}
    if args.run_current_integration:
        workers["current_integration"] = run_isolated_worker(
            args.package, expected_manifest_sha256=args.expected_manifest_sha256,
            worker_name="current_integration_worker.py",
            external_root=args.external_root)
    if args.run_real_cupy:
        workers["real_cupy"] = run_isolated_worker(
            args.package, expected_manifest_sha256=args.expected_manifest_sha256,
            worker_name="real_cupy_worker.py",
            external_root=args.external_root)
    receipt = {
        "schema": "strata-rm-global-swap-v1-source-gate-receipt",
        "source_manifest_sha256": args.expected_manifest_sha256,
        "dependencies": dependencies, "external": external,
        "workers": workers, "payloads_opened": 0, "rd_claim": False,
        "status": ("PASS_SOURCE_AND_REQUESTED_WORKERS__HOLD_PAYLOAD"
                   if workers else
                   "PASS_STATIC_SOURCE_AUTH__WORKERS_UNEXECUTED__HOLD_PAYLOAD"),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
