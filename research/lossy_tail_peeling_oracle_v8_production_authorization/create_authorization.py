#!/usr/bin/env python3
"""Create the external, one-shot lossy-tail-v8 production authorization.

The emitted JSON is not a result and does not weaken any v8 check.  It merely
collects the exact already-audited evidence and live filesystem identities
that both ``preflight_launch.py`` and ``lossy_tail_core.py`` independently
revalidate before the first source file is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload, value


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_existing(raw: str, *, directory: bool = False) -> Path:
    if not raw or "\x00" in raw or not os.path.isabs(raw) or raw != os.path.normpath(raw):
        raise ValueError(f"path is not absolute and lexical-canonical: {raw!r}")
    path = Path(raw)
    if os.path.realpath(raw) != raw:
        raise ValueError(f"path is not its own realpath: {raw}")
    metadata = os.stat(path, follow_symlinks=False)
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"expected directory: {path}")
    if not directory and not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected regular file: {path}")
    return path


def exact_absent_file(raw: str) -> Path:
    if not raw or "\x00" in raw or not os.path.isabs(raw) or raw != os.path.normpath(raw):
        raise ValueError(f"output is not absolute and lexical-canonical: {raw!r}")
    path = Path(raw)
    if path.exists() or not path.parent.is_dir() or os.path.realpath(path.parent) != os.fspath(path.parent):
        raise ValueError("authorization output must be absent below an existing canonical parent")
    return path


def exact_executable_spelling(raw: str) -> Path:
    """Preserve the venv launcher spelling reported by ``sys.executable``.

    The pinned venv's ``bin/python`` is intentionally a symlink.  V8 binds the
    interpreter spelling via ``sys.executable`` but does not include it among
    the protected filesystem identity rows, so resolving that spelling here
    would make an otherwise valid authorization fail exact comparison.
    """
    if not raw or "\x00" in raw or not os.path.isabs(raw) or raw != os.path.normpath(raw):
        raise ValueError(f"interpreter is not absolute and lexical-canonical: {raw!r}")
    path = Path(raw)
    if not path.exists() or not os.access(path, os.X_OK):
        raise ValueError(f"interpreter is missing or not executable: {path}")
    return path


def decode_mount_path(raw: str) -> str:
    return (
        raw.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def mount_snapshot() -> tuple[bytes, list[tuple[int, Path]]]:
    payload = Path("/proc/self/mountinfo").read_bytes()
    rows: list[tuple[int, Path]] = []
    for line in payload.decode("utf-8").splitlines():
        fields = line.split(" ")
        if len(fields) < 10 or "-" not in fields:
            raise ValueError("malformed /proc/self/mountinfo row")
        rows.append((int(fields[0], 10), Path(decode_mount_path(fields[4]))))
    return payload, rows


def mount_id_for(path: Path, rows: list[tuple[int, Path]]) -> int:
    resolved = Path(os.path.realpath(path))
    matches: list[tuple[int, int]] = []
    for mount_id, mount_point in rows:
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append((len(mount_point.parts), mount_id))
    if not matches:
        raise ValueError(f"no mount row contains {path}")
    return max(matches)[1]


def file_identity(label: str, path: Path, mounts: list[tuple[int, Path]]) -> dict[str, Any]:
    metadata = os.stat(path, follow_symlinks=False)
    return {
        "label": label,
        "path": os.fspath(path),
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "mount_id": mount_id_for(path, mounts),
    }


def internal_seal(value: dict[str, Any], field: str) -> str:
    seal = value.get(field)
    if not isinstance(seal, str) or len(seal) != 64:
        raise ValueError(f"missing internal seal {field}")
    clean = dict(value)
    clean.pop(field)
    if sha256(canonical_bytes(clean)) != seal:
        raise ValueError(f"internal seal mismatch: {field}")
    return seal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--source-audit-manifest", required=True)
    parser.add_argument("--source-audit-receipt", required=True)
    parser.add_argument("--runtime-receipt", required=True)
    parser.add_argument("--runtime-audit-manifest", required=True)
    parser.add_argument("--runtime-audit-receipt", required=True)
    parser.add_argument("--runtime-audit-internal-field", default="audit_receipt_sha256")
    parser.add_argument("--output-existing-parent", required=True)
    parser.add_argument("--run-root-name", required=True)
    parser.add_argument("--authorization-output", required=True)
    parser.add_argument("--python-executable", required=True)
    args = parser.parse_args()

    stage = exact_existing(args.stage, directory=True)
    launch_manifest_path = exact_existing(os.fspath(stage / "launch_manifest.json"))
    source_bindings_path = exact_existing(os.fspath(stage / "source_bindings.json"))
    runtime_contract_path = exact_existing(os.fspath(stage / "runtime_contract.json"))
    launcher = exact_existing(os.fspath(stage / "preflight_launch.py"))
    source_audit_manifest = exact_existing(args.source_audit_manifest)
    source_audit_receipt = exact_existing(args.source_audit_receipt)
    runtime_receipt_path = exact_existing(args.runtime_receipt)
    runtime_audit_manifest = exact_existing(args.runtime_audit_manifest)
    runtime_audit_receipt = exact_existing(args.runtime_audit_receipt)
    output_parent = exact_existing(args.output_existing_parent, directory=True)
    authorization_output = exact_absent_file(args.authorization_output)
    python_executable = exact_executable_spelling(args.python_executable)

    if (
        not args.run_root_name
        or args.run_root_name in {".", ".."}
        or os.sep in args.run_root_name
        or (os.altsep is not None and os.altsep in args.run_root_name)
    ):
        raise ValueError("run-root name must be one canonical path component")
    run_root = output_parent / args.run_root_name
    if run_root.exists():
        raise ValueError(f"run root already exists: {run_root}")

    launch_payload, launch_manifest = read_json(launch_manifest_path)
    bindings_payload, bindings = read_json(source_bindings_path)
    runtime_contract_payload, _ = read_json(runtime_contract_path)
    source_manifest_payload, source_manifest = read_json(source_audit_manifest)
    source_receipt_payload, source_receipt = read_json(source_audit_receipt)
    runtime_payload, runtime_receipt = read_json(runtime_receipt_path)
    runtime_audit_manifest_payload, runtime_audit_manifest_value = read_json(runtime_audit_manifest)
    runtime_audit_receipt_payload, runtime_audit_receipt_value = read_json(runtime_audit_receipt)

    if len(launch_manifest.get("allowed_members", [])) != 11:
        raise ValueError("launch stage member count drift")
    if source_manifest.get("schema") != "lossy-tail-v8-independent-source-audit-manifest-v1" or source_manifest.get("status") != "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET":
        raise ValueError("source-audit manifest is not the frozen PASS object")
    if source_receipt.get("schema") != "lossy-tail-v8-independent-source-audit-receipt-v1" or source_receipt.get("status") != "PASS_V8_INDEPENDENT_SOURCE_AUDIT":
        raise ValueError("source-audit receipt is not the frozen PASS object")
    if runtime_receipt.get("schema") != "lossy-tail-v8-source-free-runtime-receipt-v1" or runtime_receipt.get("status") != "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT":
        raise ValueError("runtime receipt schema/status drift")
    if runtime_audit_manifest_value.get("schema") != "lossy-tail-v8-independent-runtime-audit-manifest-v1" or runtime_audit_manifest_value.get("status") != "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET":
        raise ValueError("runtime-audit manifest is not the frozen PASS object")
    if runtime_audit_receipt_value.get("schema") != "lossy-tail-v8-independent-runtime-audit-receipt-v1" or runtime_audit_receipt_value.get("status") != "PASS_V8_INDEPENDENT_RUNTIME_AUDIT":
        raise ValueError("runtime-audit receipt is not the frozen PASS object")

    source_receipt_internal = internal_seal(source_receipt, "audit_receipt_sha256")
    runtime_internal = internal_seal(runtime_receipt, "runtime_receipt_sha256")
    runtime_audit_internal = internal_seal(runtime_audit_receipt_value, args.runtime_audit_internal_field)

    source = exact_existing(bindings["source_directory_at_execution"], directory=True)
    mount_payload, mounts = mount_snapshot()
    live_paths = {
        "stage": stage,
        "source": source,
        "output_existing_parent": output_parent,
        "authorization_parent": authorization_output.parent,
        "source_audit_manifest": source_audit_manifest,
        "source_audit_receipt": source_audit_receipt,
        "runtime_receipt": runtime_receipt_path,
        "runtime_audit_manifest": runtime_audit_manifest,
        "runtime_audit_receipt": runtime_audit_receipt,
    }
    identities = [file_identity(label, path, mounts) for label, path in live_paths.items()]
    inode_ids = {(row["st_dev"], row["st_ino"]) for row in identities}
    if len(inode_ids) != len(identities):
        raise ValueError("filesystem evidence contains inode aliases")

    authorization: dict[str, Any] = {
        "schema": "lossy-tail-v8-one-shot-production-authorization-v1",
        "status": "AUTHORIZED_ONCE_AFTER_INDEPENDENT_SOURCE_AND_RUNTIME_AUDITS",
        "authorization_path": os.fspath(authorization_output),
        "authorization_nonce": secrets.token_hex(32),
        "action": "CREATE_NEW_RUN_ROOT_AND_RESULT_JSON",
        "stage": {
            "path": os.fspath(stage),
            "launch_manifest_file_sha256": sha256(launch_payload),
            "launch_manifest_internal_stage_member_count": 11,
        },
        "source": {
            "path": os.fspath(source),
            "bindings_file_sha256": sha256(bindings_payload),
        },
        "output": {
            "run_root": os.fspath(run_root),
            "result_path": os.fspath(run_root / "result.json"),
        },
        "source_audit": {
            "manifest_path": os.fspath(source_audit_manifest),
            "manifest_file_sha256": sha256(source_manifest_payload),
            "receipt_path": os.fspath(source_audit_receipt),
            "receipt_file_sha256": sha256(source_receipt_payload),
            "receipt_internal_field": "audit_receipt_sha256",
            "receipt_internal_sha256": source_receipt_internal,
        },
        "runtime_receipt": {
            "path": os.fspath(runtime_receipt_path),
            "file_sha256": sha256(runtime_payload),
            "internal_sha256": runtime_internal,
            "runtime_contract_file_sha256": sha256(runtime_contract_payload),
        },
        "runtime_audit": {
            "manifest_path": os.fspath(runtime_audit_manifest),
            "manifest_file_sha256": sha256(runtime_audit_manifest_payload),
            "receipt_path": os.fspath(runtime_audit_receipt),
            "receipt_file_sha256": sha256(runtime_audit_receipt_payload),
            "receipt_internal_field": args.runtime_audit_internal_field,
            "receipt_internal_sha256": runtime_audit_internal,
        },
        "execution": {
            "python_executable": os.fspath(python_executable),
            "raw_launcher_path": os.fspath(launcher),
            "cuda_visible_devices": "0",
            "runtime_tuple": runtime_receipt["runtime_probe"]["runtime_tuple"],
        },
        "filesystem": {
            "mountinfo_path": "/proc/self/mountinfo",
            "mountinfo_file_sha256": sha256(mount_payload),
            "identities": identities,
        },
        "fixed_scientific_arguments": {
            "control_replicates": 4,
            "maximum_coordinate_passes": 4,
        },
    }
    authorization["authorization_sha256"] = sha256(canonical_bytes(authorization))
    output_payload = (json.dumps(authorization, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(
        authorization_output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(output_payload):
            offset += os.write(descriptor, output_payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({
        "status": "CREATED_ONE_SHOT_AUTHORIZATION_NO_PAYLOAD_OPENED",
        "path": os.fspath(authorization_output),
        "file_sha256": sha256(output_payload),
        "internal_sha256": authorization["authorization_sha256"],
        "run_root": os.fspath(run_root),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
