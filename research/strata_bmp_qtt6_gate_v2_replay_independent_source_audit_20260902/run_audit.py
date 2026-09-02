#!/usr/bin/env python3
"""Authenticate frozen replay-v2 source, then run the independent CPU audit."""

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


EXPECTED_MANIFEST_SHA256 = "84df0d32a55682f6565ac9d144f7de850acf77cde27bffdefa77a151211906f8"
EXPECTED_SOURCE_ROOT_SHA256 = "b518b203c43fd401c94e1bfcf67e029a85a95f1f7ce244fcd864a96d0780da47"
EXPECTED_MEMBERS = {
    "README.md", "STATIC_REVIEW.json", "THREAT_MODEL.md", "codec.py",
    "cupy_backend.py", "cupy_worker.py", "design_lock.json",
    "production_hooks.py", "run_cupy_smoke.py", "run_source_free_fixture.py",
    "search.py", "test_source_only.py", "verify_source.py",
}
EXPECTED_TESTS = 19


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
            "external producer manifest pin")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    require(manifest.get("source_root_sha256") == EXPECTED_SOURCE_ROOT_SHA256,
            "external producer root pin")
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == 13, "producer member count")
    names = [row.get("name") for row in rows]
    require(names == sorted(names, key=lambda value: value.encode("utf-8")),
            "producer canonical UTF-8 order")
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "producer member schema")
        name = row["name"]
        require(name in EXPECTED_MEMBERS and "/" not in name and "\\" not in name,
                "producer member name")
        payload = read_regular(root / name, f"producer member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"producer member pin {name}")
        observed.append(item)
    require(sha256(canonical_json(observed)) == EXPECTED_SOURCE_ROOT_SHA256,
            "producer source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == EXPECTED_MEMBERS |
            {"SOURCE_MANIFEST.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "producer exact regular closure")
    return {"manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_root_sha256": EXPECTED_SOURCE_ROOT_SHA256,
            "member_count": len(rows), "canonical_utf8_order": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    producer = authenticate_source(args.source)
    os.environ["STRATA_BMP_QTT6_V2_FROZEN_SOURCE"] = str(
        args.source.resolve(strict=True)
    )
    test_path = Path(__file__).resolve().with_name("test_benign_audit.py")
    spec = importlib.util.spec_from_file_location("independent_v2_benign_audit",
                                                  test_path)
    require(spec is not None and spec.loader is not None, "audit test import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS, "independent test count")
    success = result.wasSuccessful()
    record = {
        "schema": "strata-bmp-qtt6-independent-audit-receipt-v2",
        "passed": success,
        "status": (
            "PASS_19_TESTS__V1_REPLAY_AND_WORKSPACE_REPAIRS__"
            "HOLD_TRUSTED_PRODUCTION_CAPABILITY__HOLD_PAYLOAD"
            if success else "FAIL_INDEPENDENT_V2_CPU_AUDIT"
        ),
        "producer_source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "producer_source_root_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "producer": producer,
        "python": {"executable": sys.executable, "version": sys.version,
                   "isolated_flag": bool(sys.flags.isolated),
                   "dont_write_bytecode_flag": bool(sys.dont_write_bytecode)},
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "unittest_transcript": stream.getvalue(),
        "v1_repairs_confirmed": [
            "canonical_utf8_manifest_and_self_replay",
            "geometry_derived_candidate_capacities",
            "actual_numpy_intp_runtime_events",
            "logical_host_and_measured_cupy_ledgers_separated",
        ],
        "production_trust_hold": (
            "caller-authenticated bytes and self-attested receipts do not form "
            "an external trust anchor"
        ),
        "qwen_or_other_model_payload_accessed": False,
        "strata_or_coarse_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "network_accessed": False,
        "payload_authority": False,
        "claim_boundary": (
            "Independent source-only CPU audit; no Qwen, F<=0.8, complete codec, "
            "routed-read or trusted production result."
        ),
    }
    payload = json.dumps(record, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

