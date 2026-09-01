#!/usr/bin/env python3
"""Isolated stdlib-only v7 stage/lock audit; no runtime/payload/output API."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ALLOWED = {
    "authorization_contract.json", "audit_lock_entrypoint.py",
    "launch_manifest.json", "lossy_tail_core.py", "lossy_tail_oracle.py",
    "preflight_launch.py", "protocol_lock.json", "repair_lock.json",
    "runtime_calibrate.py", "runtime_contract.json", "source_bindings.json",
}


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(f"V7_AUDIT_REJECT: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            fail(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except Exception as exc:
        fail(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def checked_path(raw: str, label: str) -> Path:
    if not raw or "\x00" in raw or not os.path.isabs(raw) or raw != os.path.normpath(raw):
        fail(f"{label} must be absolute and lexically canonical")
    path = Path(raw)
    probe = Path(path.parts[0])
    for component in path.parts[1:]:
        probe = probe / component
        try:
            metadata = os.lstat(probe)
        except FileNotFoundError:
            fail(f"{label} component missing: {probe}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains symlink component: {probe}")
    if os.path.realpath(raw) != raw:
        fail(f"{label} does not name actual target")
    return path


def read_regular(path: Path, label: str) -> bytes:
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"{label} is not regular")
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size:
            fail(f"{label} changed during read")
        return payload
    finally:
        os.close(descriptor)


def main() -> None:
    raw = sys.argv[1:]
    if len(raw) != 4 or raw[::2] != ["--manifest", "--manifest-sha256"]:
        fail("exact grammar is --manifest PATH --manifest-sha256 HEX")
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize != 0:
        fail("requires python -B -I without optimization")
    launcher = checked_path(sys.argv[0], "raw argv0")
    if launcher.name != "audit_lock_entrypoint.py" or sys.argv[0] != os.fspath(Path(__file__)):
        fail("raw argv0 must exactly equal executing __file__")
    left = os.stat(launcher, follow_symlinks=False)
    right = os.stat(Path(__file__), follow_symlinks=False)
    if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
        fail("raw argv0/executing identity mismatch")
    stage = checked_path(os.fspath(launcher.parent), "stage")
    manifest_path = checked_path(raw[1], "manifest")
    if manifest_path != stage / "launch_manifest.json":
        fail("manifest must use exact immediate-stage spelling")
    expected_sha = raw[3].lower()
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        fail("invalid manifest SHA-256")
    manifest_payload = read_regular(manifest_path, "launch manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != expected_sha:
        fail("manifest SHA-256 mismatch")
    manifest = strict(manifest_payload, "launch manifest")
    if set(manifest) != {"schema", "status", "allowed_members", "members", "source_audit_invocation", "runtime_calibration_invocation_after_independent_source_pass_only", "production_invocation_after_independent_runtime_receipt_audit_and_separate_authorization_only", "production_child_grammar", "authorization"}:
        fail("manifest key drift")
    if manifest.get("schema") != "lossy-tail-v7-launch-manifest-v1":
        fail("manifest schema mismatch")
    if manifest.get("status") != "FROZEN_V7_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        fail("manifest status mismatch")
    allowed = manifest.get("allowed_members")
    if not isinstance(allowed, list) or len(allowed) != len(ALLOWED) or len(set(allowed)) != len(allowed) or set(allowed) != ALLOWED:
        fail("allowed-member closure mismatch")
    observed = set()
    with os.scandir(stage) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                fail(f"forbidden stage entry: {entry.name}")
            observed.add(entry.name)
    if observed != ALLOWED:
        fail("stage closure mismatch")
    rows = manifest.get("members")
    expected_rows = ALLOWED - {"launch_manifest.json"}
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        fail("manifest row cardinality mismatch")
    row_paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(row_paths) != len(rows) or len(set(row_paths)) != len(row_paths) or set(row_paths) != expected_rows:
        fail("manifest row closure mismatch")
    payloads = {"launch_manifest.json": manifest_payload}
    for row in rows:
        if set(row) != {"path", "bytes", "sha256"}:
            fail(f"manifest row key drift: {row.get('path')}")
        payload = read_regular(stage / row["path"], f"member {row['path']}")
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            fail(f"member identity mismatch: {row['path']}")
        payloads[row["path"]] = payload
    repair = strict(payloads["repair_lock.json"], "repair lock")
    if repair.get("schema") != "lossy-tail-release-repair-lock-v7" or repair.get("status") != "FROZEN_V7_SOURCE_PACKAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        fail("repair-lock status mismatch")
    repair_copy = dict(repair)
    internal = repair_copy.pop("repair_lock_sha256", None)
    if internal is None or hashlib.sha256(canonical(repair_copy)).hexdigest() != internal:
        fail("repair-lock internal seal mismatch")
    live = {
        "scientific_protocol_sha256": hashlib.sha256(payloads["protocol_lock.json"]).hexdigest(),
        "source_bindings_sha256": hashlib.sha256(payloads["source_bindings.json"]).hexdigest(),
        "runtime_contract_sha256": hashlib.sha256(payloads["runtime_contract.json"]).hexdigest(),
        "authorization_contract_sha256": hashlib.sha256(payloads["authorization_contract.json"]).hexdigest(),
        "oracle_bootstrap_sha256": hashlib.sha256(payloads["lossy_tail_oracle.py"]).hexdigest(),
        "scientific_core_sha256": hashlib.sha256(payloads["lossy_tail_core.py"]).hexdigest(),
        "preflight_sha256": hashlib.sha256(payloads["preflight_launch.py"]).hexdigest(),
        "audit_entrypoint_sha256": hashlib.sha256(payloads["audit_lock_entrypoint.py"]).hexdigest(),
        "runtime_calibrate_sha256": hashlib.sha256(payloads["runtime_calibrate.py"]).hexdigest(),
    }
    if repair.get("authenticated_identities") != live:
        fail("repair-lock current identity mismatch")
    protocol = strict(payloads["protocol_lock.json"], "protocol")
    runtime_contract = strict(payloads["runtime_contract.json"], "runtime contract")
    authorization_contract = strict(payloads["authorization_contract.json"], "authorization contract")
    if protocol.get("status") != "FROZEN_V7_BEFORE_ANY_RUNTIME_CALIBRATION_PAYLOAD_OR_GPU_EXECUTION":
        fail("protocol not frozen")
    if runtime_contract.get("status") != "FROZEN_SOURCE_FREE_BEFORE_RUNTIME_CALIBRATION":
        fail("runtime contract not frozen")
    if authorization_contract.get("status") != "FROZEN_TEMPLATE_ONLY_NO_AUTHORIZATION_EXISTS":
        fail("authorization contract status mismatch")
    print(json.dumps({
        "event": "V7_SOURCE_ONLY_STAGE_AND_LOCK_PASS",
        "manifest_sha256": expected_sha,
        "repair_lock_internal_sha256": internal,
        "stage_member_count": len(observed),
        "cupy_imported": False,
        "cuda_initialized": False,
        "payload_paths_supplied": 0,
        "payload_files_opened": 0,
        "runtime_receipts_opened": 0,
        "production_authorizations_opened": 0,
        "result_files_written": 0,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
