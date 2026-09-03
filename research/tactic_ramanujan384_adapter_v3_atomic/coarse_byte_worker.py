#!/usr/bin/env python3
"""Authenticate and execute a pathless zero-import coarse byte program."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


ROLE_ORDER = ("gate", "up", "down_transposed")
UINT32_MAX = (1 << 32) - 1


class CoarseWorkerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoarseWorkerError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def strict_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("ascii"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CoarseWorkerError(f"{label} nonfinite {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoarseWorkerError(f"{label} JSON") from exc
    require(isinstance(value, dict) and canonical_json(value) == payload,
            f"{label} canonical object")
    return value


def _read_regular(path: Path, maximum: int) -> bytes:
    absolute = path.resolve(strict=True)
    require(absolute == path.absolute(), "canonical nonsymlink input")
    descriptor = os.open(
        os.fspath(absolute), os.O_RDONLY | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "regular single-link input")
        require(0 < before.st_size <= maximum, "bounded nonempty input")
        output = bytearray()
        while len(output) < before.st_size:
            row = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(row), "short input read")
            output.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_nlink), "input identity drift")
        return bytes(output)
    finally:
        os.close(descriptor)


def _authenticate_closure(*, directory: Path, manifest_path: Path,
                          expected_manifest_sha256: str,
                          expected_root_sha256: str, schema: str) -> Mapping[str, bytes]:
    root = directory.resolve(strict=True)
    require(root == directory.absolute() and root.is_dir(), "canonical closure directory")
    manifest_payload = _read_regular(manifest_path, 4 << 20)
    require(sha256(manifest_payload) == expected_manifest_sha256, "closure manifest SHA256")
    document = strict_canonical_json(manifest_payload, "closure manifest")
    require(set(document) == {"schema", "source_root_sha256", "members"}
            and document["schema"] == schema
            and document["source_root_sha256"] == expected_root_sha256,
            "closure manifest identity")
    rows = document["members"]
    require(isinstance(rows, list) and bool(rows), "closure members")
    names = []
    canonical_rows = []
    payloads = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "closure member schema")
        name, size, digest = row["name"], row["bytes"], row["sha256"]
        require(isinstance(name, str) and name and Path(name).name == name
                and name not in names, "flat unique closure member")
        require(type(size) is int and size > 0 and isinstance(digest, str)
                and len(digest) == 64, "closure member fields")
        payload = _read_regular(root / name, 16 << 20)
        require(len(payload) == size and sha256(payload) == digest,
                f"closure member {name}")
        names.append(name)
        payloads[name] = payload
        canonical_rows.append({"name": name, "bytes": size, "sha256": digest})
    canonical_rows.sort(key=lambda row: row["name"])
    require(rows == canonical_rows, "canonical closure rows")
    actual = sorted(path.name for path in root.iterdir())
    require(actual == sorted(names) and all((root / name).is_file()
                                            and not (root / name).is_symlink()
                                            for name in names), "exact closure entries")
    require(sha256(canonical_json(canonical_rows)) == expected_root_sha256,
            "closure source root")
    return MappingProxyType(payloads)


def _execute_program(program_payload: bytes, coarse_payload: bytes,
                     intermediate: int, hidden: int,
                     role_order: Sequence[str]) -> Mapping[str, bytes]:
    """The worker ABI has byte/int inputs only and exposes no paths or callbacks."""
    require(type(program_payload) is bytes and type(coarse_payload) is bytes,
            "worker byte buffers")
    require(type(intermediate) is int and 0 < intermediate <= UINT32_MAX
            and type(hidden) is int and 0 < hidden <= UINT32_MAX,
            "worker uint32 geometry")
    roles = tuple(role_order)
    require(roles == ROLE_ORDER, "worker role order")
    program = strict_canonical_json(program_payload, "coarse worker program")
    require(set(program) == {
        "schema", "version", "imports", "opcode", "coarse_sha256", "shape", "roles"
    }, "coarse worker program exact schema")
    require(program["schema"] == "tactic-coarse-byte-worker-program-v3"
            and program["version"] == 1 and program["imports"] == []
            and program["opcode"] == "ZERO_F32_LE", "zero-import worker identity")
    require(program["coarse_sha256"] == sha256(coarse_payload)
            and program["shape"] == [intermediate, hidden]
            and program["roles"] == list(ROLE_ORDER), "worker input binding")
    role_bytes = bytes(4 * intermediate * hidden)
    return MappingProxyType({role: role_bytes for role in ROLE_ORDER})


def authenticate_and_decode(
    *, capability_path: Path, expected_capability_sha256: str,
    worker_source_directory: Path, worker_source_manifest_path: Path,
    expected_worker_source_manifest_sha256: str, expected_worker_source_root_sha256: str,
    auditor_source_directory: Path, auditor_source_manifest_path: Path,
    expected_auditor_source_manifest_sha256: str, expected_auditor_source_root_sha256: str,
    independent_audit_receipt_path: Path, expected_independent_audit_receipt_sha256: str,
    coarse_payload: bytes, intermediate: int, hidden: int,
    role_order: Sequence[str],
) -> dict[str, Any]:
    """Authenticate external closures and run the built-in byte-only VM."""
    pins = (
        expected_capability_sha256, expected_worker_source_manifest_sha256,
        expected_worker_source_root_sha256, expected_auditor_source_manifest_sha256,
        expected_auditor_source_root_sha256, expected_independent_audit_receipt_sha256,
    )
    require(all(isinstance(value, str) and len(value) == 64 for value in pins),
            "six distinct external SHA256 pins")
    capability_payload = _read_regular(capability_path, 2 << 20)
    require(sha256(capability_payload) == expected_capability_sha256,
            "capability external pin")
    capability = strict_canonical_json(capability_payload, "coarse worker capability")
    require(set(capability) == {
        "schema", "status", "capability_id", "program_name", "program_sha256",
        "worker_source_manifest_sha256", "worker_source_root_sha256",
        "auditor_source_manifest_sha256", "auditor_source_root_sha256",
        "independent_audit_receipt_sha256",
    }, "capability exact schema")
    require(capability["schema"] == "tactic-coarse-byte-worker-capability-v3"
            and capability["status"] == "INDEPENDENT_ZERO_IMPORT_WORKER_AUDIT_REQUIRED",
            "capability identity")
    expected = {
        "worker_source_manifest_sha256": expected_worker_source_manifest_sha256,
        "worker_source_root_sha256": expected_worker_source_root_sha256,
        "auditor_source_manifest_sha256": expected_auditor_source_manifest_sha256,
        "auditor_source_root_sha256": expected_auditor_source_root_sha256,
        "independent_audit_receipt_sha256": expected_independent_audit_receipt_sha256,
    }
    require(all(capability[key] == value for key, value in expected.items()),
            "capability external closure pins")
    worker = _authenticate_closure(
        directory=worker_source_directory, manifest_path=worker_source_manifest_path,
        expected_manifest_sha256=expected_worker_source_manifest_sha256,
        expected_root_sha256=expected_worker_source_root_sha256,
        schema="tactic-coarse-worker-source-manifest-v3",
    )
    _authenticate_closure(
        directory=auditor_source_directory, manifest_path=auditor_source_manifest_path,
        expected_manifest_sha256=expected_auditor_source_manifest_sha256,
        expected_root_sha256=expected_auditor_source_root_sha256,
        schema="tactic-coarse-worker-auditor-source-manifest-v3",
    )
    require(capability["program_name"] in worker
            and sha256(worker[capability["program_name"]]) == capability["program_sha256"],
            "capability program member")
    receipt_payload = _read_regular(independent_audit_receipt_path, 4 << 20)
    require(sha256(receipt_payload) == expected_independent_audit_receipt_sha256,
            "independent audit receipt external pin")
    receipt = strict_canonical_json(receipt_payload, "coarse worker audit receipt")
    require(set(receipt) == {
        "schema", "status", "capability_id", "program_sha256",
        "worker_source_manifest_sha256", "worker_source_root_sha256",
        "auditor_source_manifest_sha256", "auditor_source_root_sha256",
        "coarse_sha256", "shape", "role_order", "output_f32_sha256_by_role",
        "literal_payload_only_pass", "zero_import_no_path_pass",
        "deterministic_output_hashes_recorded", "hostile_tests_passed"
    }, "audit receipt exact schema")
    require(receipt["schema"] == "tactic-independent-coarse-worker-audit-receipt-v3"
            and receipt["status"] == "INDEPENDENT_COARSE_WORKER_AUDIT_PASS"
            and receipt["capability_id"] == capability["capability_id"],
            "audit receipt identity")
    for key in ("program_sha256", "worker_source_manifest_sha256",
                "worker_source_root_sha256", "auditor_source_manifest_sha256",
                "auditor_source_root_sha256"):
        require(receipt[key] == capability[key], f"audit receipt {key}")
    require(receipt["literal_payload_only_pass"] is True
            and receipt["zero_import_no_path_pass"] is True
            and receipt["deterministic_output_hashes_recorded"] is True
            and type(receipt["hostile_tests_passed"]) is int
            and receipt["hostile_tests_passed"] >= 12, "audit capability evidence")
    require(receipt["coarse_sha256"] == sha256(coarse_payload)
            and receipt["shape"] == [intermediate, hidden]
            and receipt["role_order"] == list(ROLE_ORDER), "audit input binding")
    outputs = _execute_program(
        worker[capability["program_name"]], coarse_payload,
        intermediate, hidden, role_order,
    )
    output_hashes = {role: sha256(outputs[role]) for role in ROLE_ORDER}
    require(receipt["output_f32_sha256_by_role"] == output_hashes,
            "independently recorded worker output hashes")
    return {
        "coarse_f32_bytes": outputs,
        "capability_id": capability["capability_id"],
        "capability_sha256": expected_capability_sha256,
        "worker_source_root_sha256": expected_worker_source_root_sha256,
        "auditor_source_root_sha256": expected_auditor_source_root_sha256,
        "independent_audit_receipt_sha256": expected_independent_audit_receipt_sha256,
        "zero_import_no_path_byte_worker": True,
        "mutable_decoder_object_used": False,
    }
