#!/usr/bin/env python3
"""Source-free v7 runtime calibration entrypoint.

This is not production authority.  It has no source/model argument and emits
an untrusted create-new receipt which must later pass an independent audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


STAGE_MEMBERS = {
    "authorization_contract.json", "audit_lock_entrypoint.py",
    "launch_manifest.json", "lossy_tail_core.py", "lossy_tail_oracle.py",
    "preflight_launch.py", "protocol_lock.json", "repair_lock.json",
    "runtime_calibrate.py", "runtime_contract.json", "source_bindings.json",
}


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(f"V7_RUNTIME_CALIBRATE_REJECT: {message}")


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            fail(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_object)
    except Exception as exc:
        fail(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_original(raw: str, label: str, *, allow_missing_tail: bool) -> Path:
    if not raw or "\x00" in raw or not os.path.isabs(raw) or raw != os.path.normpath(raw):
        fail(f"{label} must be absolute and lexically canonical")
    path = Path(raw)
    probe = Path(path.parts[0])
    missing = False
    for component in path.parts[1:]:
        probe = probe / component
        if missing:
            continue
        try:
            metadata = os.lstat(probe)
        except FileNotFoundError:
            if not allow_missing_tail:
                fail(f"{label} component missing: {probe}")
            missing = True
            continue
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains symlink component: {probe}")
    if os.path.realpath(raw) != raw:
        fail(f"{label} does not name its actual target")
    return path


def read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"{label} is not regular")
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size:
            fail(f"{label} changed during descriptor read")
        return payload
    finally:
        os.close(descriptor)


def contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_stage(stage: Path, manifest_path: Path, manifest_sha256: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest_bytes = read_regular(manifest_path, "launch manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        fail("launch-manifest file SHA-256 mismatch")
    manifest = strict_json(manifest_bytes, "launch manifest")
    if set(manifest) != {"schema", "status", "allowed_members", "members", "source_audit_invocation", "runtime_calibration_invocation_after_independent_source_pass_only", "production_invocation_after_independent_runtime_receipt_audit_and_separate_authorization_only", "production_child_grammar", "authorization"}:
        fail("launch-manifest key drift")
    if manifest.get("schema") != "lossy-tail-v7-launch-manifest-v1":
        fail("launch-manifest schema mismatch")
    if manifest.get("status") != "FROZEN_V7_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        fail("launch-manifest status mismatch")
    allowed = manifest.get("allowed_members")
    if not isinstance(allowed, list) or len(allowed) != len(STAGE_MEMBERS) or len(set(allowed)) != len(allowed) or set(allowed) != STAGE_MEMBERS:
        fail("allowed-member closure mismatch")
    observed = set()
    with os.scandir(stage) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                fail(f"forbidden stage entry: {entry.name}")
            observed.add(entry.name)
    if observed != STAGE_MEMBERS:
        fail("stage entry closure mismatch")
    rows = manifest.get("members")
    expected = STAGE_MEMBERS - {"launch_manifest.json"}
    if not isinstance(rows, list) or len(rows) != len(expected):
        fail("member-row cardinality mismatch")
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(paths) != len(rows) or len(set(paths)) != len(paths) or set(paths) != expected:
        fail("member-row closure mismatch")
    payloads = {"launch_manifest.json": manifest_bytes}
    for row in rows:
        if set(row) != {"path", "bytes", "sha256"}:
            fail(f"member-row key drift: {row.get('path')}")
        payload = read_regular(stage / row["path"], f"stage member {row['path']}")
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            fail(f"member identity mismatch: {row['path']}")
        payloads[row["path"]] = payload
    return manifest, payloads


def write_receipt(path: Path, value: dict[str, Any]) -> None:
    clean = dict(value)
    clean["runtime_receipt_sha256"] = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    payload = (json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    os.mkdir(path.parent, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    raw = sys.argv[1:]
    flags = ["--manifest", "--manifest-sha256", "--output"]
    if len(raw) != 6 or raw[::2] != flags:
        fail("exact grammar is --manifest PATH --manifest-sha256 HEX --output PATH")
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize != 0:
        fail("requires python -B -I without optimization")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        fail("requires an explicit single effective CUDA_VISIBLE_DEVICES=0")
    launcher = canonical_original(sys.argv[0], "raw argv0", allow_missing_tail=False)
    module_file = os.fspath(Path(__file__))
    if launcher.name != "runtime_calibrate.py" or sys.argv[0] != module_file:
        fail("raw argv0 must exactly equal runtime_calibrate.py __file__")
    launcher_stat = os.stat(launcher, follow_symlinks=False)
    module_stat = os.stat(module_file, follow_symlinks=False)
    if (launcher_stat.st_dev, launcher_stat.st_ino) != (module_stat.st_dev, module_stat.st_ino):
        fail("raw argv0 and executing calibrator identity differ")
    stage = canonical_original(os.fspath(launcher.parent), "stage", allow_missing_tail=False)
    manifest = canonical_original(raw[1], "manifest", allow_missing_tail=False)
    if manifest != stage / "launch_manifest.json":
        fail("manifest must use exact immediate-stage spelling")
    manifest_sha256 = raw[3].lower()
    if len(manifest_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in manifest_sha256):
        fail("invalid launch-manifest SHA-256")
    output = canonical_original(raw[5], "runtime receipt output", allow_missing_tail=True)
    if output.name != "runtime_receipt.json" or output.parent.exists() or output.exists() or not output.parent.parent.is_dir():
        fail("runtime receipt must be absent run_root/runtime_receipt.json below existing parent")
    if contains(stage, output.parent.parent) or contains(output.parent.parent, stage):
        fail("runtime receipt existing output parent overlaps sealed stage")
    _, payloads = validate_stage(stage, manifest, manifest_sha256)
    repair = strict_json(payloads["repair_lock.json"], "repair lock")
    if repair.get("schema") != "lossy-tail-release-repair-lock-v7" or repair.get("status") != "FROZEN_V7_SOURCE_PACKAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        fail("repair-lock status mismatch")
    repair_copy = dict(repair)
    repair_internal = repair_copy.pop("repair_lock_sha256", None)
    if not isinstance(repair_internal, str) or hashlib.sha256(canonical_bytes(repair_copy)).hexdigest() != repair_internal:
        fail("repair-lock internal seal mismatch")
    live_identities = {
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
    if repair.get("authenticated_identities") != live_identities:
        fail("repair-lock authenticated identities mismatch")
    protocol = strict_json(payloads["protocol_lock.json"], "protocol")
    authorization_contract = strict_json(payloads["authorization_contract.json"], "authorization contract")
    bindings = strict_json(payloads["source_bindings.json"], "source bindings")
    if protocol.get("status") != "FROZEN_V7_BEFORE_ANY_RUNTIME_CALIBRATION_PAYLOAD_OR_GPU_EXECUTION":
        fail("scientific protocol status mismatch")
    if authorization_contract.get("status") != "FROZEN_TEMPLATE_ONLY_NO_AUTHORIZATION_EXISTS":
        fail("authorization contract status mismatch")
    frozen_source = Path(bindings.get("source_directory_at_execution", ""))
    if not frozen_source.is_absolute() or contains(frozen_source, output.parent.parent) or contains(output.parent.parent, frozen_source):
        fail("runtime receipt existing parent overlaps frozen source spelling")
    contract = strict_json(payloads["runtime_contract.json"], "runtime contract")
    if contract.get("schema") != "lossy-tail-v7-runtime-calibration-contract-v1" or contract.get("status") != "FROZEN_SOURCE_FREE_BEFORE_RUNTIME_CALIBRATION":
        fail("runtime contract status mismatch")

    oracle_module = types.ModuleType("lossy_tail_v7_runtime_core")
    oracle_module.__file__ = os.fspath(stage / "lossy_tail_core.py")
    oracle_module.__package__ = None
    oracle_module.__dict__["__V7_CORE_CONTEXT__"] = {
        "schema": "lossy-tail-v7-core-context-v1",
        "mode": "runtime_calibration",
        "launch_manifest_sha256": manifest_sha256,
    }
    sys.modules[oracle_module.__name__] = oracle_module
    exec(compile(payloads["lossy_tail_core.py"], oracle_module.__file__, "exec"), oracle_module.__dict__)
    import cupy as cp  # type: ignore
    live_tuple = oracle_module.exact_runtime_tuple(cp)
    for key, expected in contract["expected_runtime_tuple"].items():
        if live_tuple.get(key) != expected:
            fail(f"runtime tuple mismatch for {key}")
    probe = oracle_module.runtime_probe(cp)
    receipt = {
        "schema": "lossy-tail-v7-source-free-runtime-receipt-v1",
        "status": "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT",
        "launch_manifest": {"path": os.fspath(manifest), "sha256": manifest_sha256},
        "runtime_contract": {
            "path": os.fspath(stage / "runtime_contract.json"),
            "sha256": hashlib.sha256(payloads["runtime_contract.json"]).hexdigest(),
        },
        "calibrator": {
            "path": os.fspath(launcher),
            "sha256": hashlib.sha256(payloads["runtime_calibrate.py"]).hexdigest(),
        },
        "scientific_core": {
            "path": os.fspath(stage / "lossy_tail_core.py"),
            "sha256": hashlib.sha256(payloads["lossy_tail_core.py"]).hexdigest(),
        },
        "invocation": {"argv": list(sys.argv), "cwd": os.getcwd(), "python_flags": ["-B", "-I"]},
        "runtime_probe": probe,
        "access_ledger": {
            "model_or_qwen_paths_supplied": 0,
            "model_or_qwen_paths_opened": 0,
            "payload_files_opened": 0,
            "production_results_opened": 0,
            "production_outputs_created": 0,
            "runtime_receipts_created": 1,
        },
        "authorization": "NOT_PRODUCTION_AUTHORITY_UNTIL_INDEPENDENT_AUDIT_AND_LATER_ONE_SHOT_AUTHORIZATION",
    }
    write_receipt(output, receipt)
    print(json.dumps({
        "runtime_calibration": "RECEIPT_CREATED_UNTRUSTED",
        "output": os.fspath(output),
        "probe_aggregate_sha256": probe["probe_aggregate_sha256"],
        "payload_files_opened": 0,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
