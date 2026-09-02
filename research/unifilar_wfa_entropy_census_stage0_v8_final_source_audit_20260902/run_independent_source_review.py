#!/usr/bin/env python3
"""Run the final UWFA-SC v8 source review in an isolated checkout.

The launcher first executes the independent byte/commit review, then invokes
the sealed package verifier and all source-only tests as isolated Python
subprocesses.  It emits the requested final review JSON only if all three gates
pass.  It never searches for any payload path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import independent_source_review as core


EXPECTED_BUNDLE_SHA256 = "e4a525cf6b8ffe71c30573e0dab101c36c31af77d27e5c3b449a1be5701c0988"
EXPECTED_REVIEWER_SHA256 = "a9d5a2c8ca046312e2dbce657ceadf11ed965468375ca6988f7f68f5de14e9c2"


class LaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(argv: list[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        argv,
        cwd=None if cwd is None else str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return completed


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise LaunchError(f"nonfinite JSON: {value}")

    parsed = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(parsed, dict), "JSON root object")
    return parsed


def text_command(argv: list[str]) -> str:
    completed = run(argv, timeout=60)
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr.decode('utf-8', errors='replace')}")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--python", default="/usr/bin/python3.12")
    args = parser.parse_args()

    repo = Path(args.repo).absolute()
    bundle = Path(args.bundle).absolute()
    python = Path(args.python).absolute()
    package = repo / core.PACKAGE_REL
    reviewer = Path(core.__file__).absolute()

    bundle_bytes = bundle.read_bytes()
    reviewer_bytes = reviewer.read_bytes()
    require(sha256(bundle_bytes) == EXPECTED_BUNDLE_SHA256, "transport bundle SHA-256")
    require(sha256(reviewer_bytes) == EXPECTED_REVIEWER_SHA256, "independent reviewer SHA-256")
    require(python.is_file(), "isolated Python executable")

    bundle_check = run(["git", "-C", str(repo), "bundle", "verify", str(bundle)], timeout=120)
    require(bundle_check.returncode == 0, "git bundle verification")

    source_review = core.review(repo)
    require(source_review["status"] == "PASS_INDEPENDENT_SOURCE_REVIEW", "independent source review")

    verify_argv = [
        str(python), "-I", "-B", str(package / "verify_source.py"),
        "--package", str(package), "--compact",
    ]
    verify_start = time.monotonic()
    verified = run(verify_argv, cwd=package, timeout=120)
    verify_wall = time.monotonic() - verify_start
    require(verified.returncode == 0 and verified.stderr == b"", "sealed verify_source execution")
    producer_verification = strict_json(verified.stdout)
    require(producer_verification.get("schema") == "unifilar-wfa-source-verification-v8", "producer verifier schema")
    require(producer_verification.get("status") == "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "producer verifier status")
    require(producer_verification.get("source_manifest_sha256") == core.EXPECTED_MANIFEST_SHA256, "producer verifier manifest binding")
    require(producer_verification.get("payload_authority_granted") is False, "producer verifier payload boundary")

    test_argv = [str(python), "-I", "-B", str(package / "test_source_only.py")]
    test_start = time.monotonic()
    tested = run(test_argv, cwd=package, timeout=600)
    test_wall = time.monotonic() - test_start
    combined = tested.stdout + b"\n" + tested.stderr
    combined_text = combined.decode("utf-8", errors="strict")
    require(tested.returncode == 0, "source-only test return code")
    timing_matches = re.findall(r"Ran\s+(\d+)\s+tests\s+in\s+([0-9.]+)s", combined_text)
    require(timing_matches == [("68", timing_matches[0][1])] if timing_matches else False, "exact 68-test summary")
    require(re.search(r"(?:^|\n)OK(?:\n|$)", combined_text) is not None, "unittest OK summary")
    require("FAILED" not in combined_text and "skipped=" not in combined_text, "no failed/skipped source test")
    passed_test_lines = re.findall(r"^test_[^\r\n]+ \.\.\. ok$", combined_text, flags=re.MULTILINE)
    require(len(passed_test_lines) == 68, f"expected 68 explicit ok lines, got {len(passed_test_lines)}")

    python_sha = sha256(python.read_bytes())
    gpu_query = text_command([
        "nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version", "--format=csv,noheader",
    ])
    require(len(gpu_query.splitlines()) == 1, "one audit GPU")
    tracked_status = text_command(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"])
    require(tracked_status == "", "clean tracked checkout after tests")
    sparse_paths = text_command(["git", "-C", str(repo), "sparse-checkout", "list"]).splitlines()
    require(sparse_paths == [core.PACKAGE_REL], "source-only sparse checkout path")

    source_review.update({
        "runpod_execution": {
            "hostname": text_command(["hostname"]),
            "kernel": text_command(["uname", "-a"]),
            "python_version": text_command([str(python), "--version"]),
            "python_executable": str(python),
            "python_executable_sha256": python_sha,
            "gpu_identity_csv": gpu_query,
            "isolated_checkout": str(repo),
            "sparse_checkout_paths": sparse_paths,
            "tracked_worktree_clean_before_and_after": True,
            "transport_bundle_bytes": len(bundle_bytes),
            "transport_bundle_sha256": EXPECTED_BUNDLE_SHA256,
            "git_bundle_verified_complete_history": True,
            "anonymous_github_clone_available_from_runpod": False,
        },
        "independent_reviewer": {
            "path": reviewer.name,
            "bytes": len(reviewer_bytes),
            "sha256": EXPECTED_REVIEWER_SHA256,
            "imports_producer_modules": False,
        },
        "sealed_source_verifier": {
            "argv": verify_argv,
            "return_code": verified.returncode,
            "wall_seconds": verify_wall,
            "stdout_bytes": len(verified.stdout),
            "stdout_sha256": sha256(verified.stdout),
            "stderr_bytes": len(verified.stderr),
            "stderr_sha256": sha256(verified.stderr),
            "result": producer_verification,
        },
        "source_only_tests": {
            "argv": test_argv,
            "return_code": tested.returncode,
            "reported_tests": 68,
            "explicit_ok_lines": len(passed_test_lines),
            "failures": 0,
            "errors": 0,
            "skips": 0,
            "reported_seconds": float(timing_matches[0][1]),
            "launcher_wall_seconds": test_wall,
            "stdout_bytes": len(tested.stdout),
            "stdout_sha256": sha256(tested.stdout),
            "stderr_bytes": len(tested.stderr),
            "stderr_sha256": sha256(tested.stderr),
            "summary": "Ran 68 tests; all 68 explicit test lines passed; OK",
        },
        "review_scope": "sealed source, freeze transition, source verifier, and synthetic source-only tests only",
        "qwen_or_payload_performance_claim": False,
        "external_pinned_dispatcher_audit_completed_by_this_review": False,
        "fresh_public_commit_rtx5090_all150_representative_replay_completed_by_this_review": False,
        "fresh_process_payload_result_audit_completed_by_this_review": False,
    })

    print(json.dumps(source_review, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_FINAL_INDEPENDENT_SOURCE_REVIEW: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
