#!/usr/bin/env python3
"""Authenticate frozen producer source, then run the independent CPU audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import unittest


EXPECTED_MANIFEST_SHA256 = (
    "916aaca15620e3bf033e849b74a73604015fab280dfe8953683d6cbe04e0d2e4"
)
EXPECTED_SOURCE_ROOT_SHA256 = (
    "369e01b30173977a5d8227e71104c8515f1b68ef440198dccd1488050e865203"
)
EXPECTED_MEMBER_NAMES = {
    "README.md", "SOURCE_MANIFEST.json", "THREAT_MODEL.md", "codec.py",
    "cupy_backend.py", "cupy_worker.py", "design_lock.json",
    "production_hooks.py", "run_cupy_smoke.py", "run_source_free_fixture.py",
    "search.py", "test_source_only.py", "verify_source.py",
}
EXPECTED_TESTS = 21


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def read_regular(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_size, before.st_mtime_ns, before.st_mode,
             getattr(before, "st_ino", 0)) ==
            (after.st_size, after.st_mtime_ns, after.st_mode,
             getattr(after, "st_ino", 0)), f"{label} changed during read")
    return payload


def authenticate_source(source: Path) -> dict:
    root = source.resolve(strict=True)
    require(root.is_dir(), "producer source directory")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json", "manifest")
    require(sha256(manifest_payload) == EXPECTED_MANIFEST_SHA256,
            "externally pinned producer manifest")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    require(manifest.get("source_root_sha256") == EXPECTED_SOURCE_ROOT_SHA256,
            "externally pinned producer root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "producer members")
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "producer member schema")
        name = row["name"]
        require(isinstance(name, str) and "/" not in name and "\\" not in name,
                "producer member name")
        payload = read_regular(root / name, f"producer member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"producer member pin {name}")
        observed.append(item)
    require(sha256(canonical_json(observed)) == EXPECTED_SOURCE_ROOT_SHA256,
            "producer source-root closure")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == EXPECTED_MEMBER_NAMES and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "producer exact package closure")
    return {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_root_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "member_count": len(observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_receipt = authenticate_source(args.source)
    os.environ["STRATA_BMP_QTT6_V1_FROZEN_SOURCE"] = str(
        args.source.resolve(strict=True)
    )
    test_path = Path(__file__).resolve().with_name("test_benign_audit.py")
    spec = importlib.util.spec_from_file_location("independent_v1_benign_audit",
                                                  test_path)
    require(spec is not None and spec.loader is not None, "audit test import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS, "independent test count")
    record = {
        "schema": "strata-bmp-qtt6-v1-independent-cpu-audit-receipt-v1",
        "status": (
            "PASS_21_TESTS__V0_REPAIRS_CONFIRMED__"
            "HOLD_EXACT_WORKSPACE_LEDGER_AND_REPLAY_DOC__HOLD_PAYLOAD"
            if result.wasSuccessful() else "FAIL_INDEPENDENT_CPU_AUDIT"
        ),
        "producer": source_receipt,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "isolated_flag": bool(sys.flags.isolated),
            "dont_write_bytecode_flag": bool(sys.dont_write_bytecode),
        },
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "unittest_transcript": stream.getvalue(),
        "findings_reproduced": [
            "A1_candidate_packet_capacity_is_not_exact_runtime_ownership",
            "A1_stable_order_dtype_is_platform_intp_not_forced_i32",
            "A2_readme_option_and_manifest_order_block_frozen_verifier",
            "A3_production_hooks_are_syntactic_launch_prerequisites",
        ],
        "qwen_or_other_model_payload_accessed": False,
        "strata_or_coarse_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "network_accessed": False,
        "payload_authority": False,
        "claim_boundary": (
            "CPU source-only audit. No Qwen result, F<=0.8 result, complete "
            "physical codec, routed-read result or production authority."
        ),
    }
    payload = json.dumps(record, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
