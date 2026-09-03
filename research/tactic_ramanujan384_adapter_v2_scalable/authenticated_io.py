#!/usr/bin/env python3
"""Self-contained explicit-path source/coarse authentication for scalable v2."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "tactic-ramanujan384-input-binding-v2"
STATUS = "INDEPENDENT_SOURCE_COARSE_AND_DECODER_AUDITS_REQUIRED"
ROLE_ORDER = ("gate", "up", "down_transposed")


class AuthenticationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthenticationError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    result = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AuthenticationError(f"{label} nonfinite {token}")
        ),
    )
    require(isinstance(result, dict), f"{label} object")
    return result


def read_regular(path: Path, maximum: int) -> bytes:
    absolute = path.resolve(strict=True)
    require(absolute == path.absolute(), "canonical nonsymlink input path")
    descriptor = os.open(os.fspath(absolute), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "regular single-link input")
        require(0 < before.st_size <= maximum, "bounded nonempty input")
        output = bytearray()
        while len(output) < before.st_size:
            row = os.read(descriptor, min(8 << 20, before.st_size - len(output)))
            require(bool(row), "short authenticated read")
            output.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
                "authenticated input identity drift")
        return bytes(output)
    finally:
        os.close(descriptor)


def _record(value: Any, label: str) -> tuple[int, str]:
    require(isinstance(value, Mapping) and set(value) == {"bytes", "sha256"},
            f"{label} record")
    size, digest = value["bytes"], value["sha256"]
    require(type(size) is int and size > 0 and isinstance(digest, str) and len(digest) == 64,
            f"{label} fields")
    return size, digest


def authenticate_role(
    *,
    binding_path: Path,
    expected_binding_sha256: str,
    source_audit_receipt_path: Path,
    expected_source_audit_receipt_sha256: str,
    input_manifest_path: Path,
    auditor_source_manifest_path: Path,
    coarse_artifact_path: Path,
    source_bf16_path: Path,
    coarse_reconstruction_f32_path: Path,
) -> dict[str, Any]:
    binding_payload = read_regular(binding_path, 2 << 20)
    require(sha256(binding_payload) == expected_binding_sha256, "binding SHA256")
    binding = strict_json(binding_payload, "binding")
    require(set(binding) == {
        "schema", "status", "role", "shape", "normalized_layout", "source",
        "coarse_artifact", "coarse_reconstruction", "source_audit",
        "input_manifest_sha256", "auditor_source_manifest_sha256",
    }, "binding exact schema")
    require(binding["schema"] == SCHEMA and binding["status"] == STATUS, "binding identity")
    role = binding["role"]
    shape = binding["shape"]
    require(role in ROLE_ORDER, "role")
    require(isinstance(shape, list) and len(shape) == 2
            and all(type(value) is int and value > 0 for value in shape), "shape")
    require(binding["normalized_layout"] == "intermediate_by_hidden_row_major", "layout")
    weights = shape[0] * shape[1]
    source_size, source_sha = _record(binding["source"], "source")
    coarse_size, coarse_sha = _record(binding["coarse_artifact"], "coarse")
    reconstruction_size, reconstruction_sha = _record(
        binding["coarse_reconstruction"], "coarse reconstruction"
    )
    require(source_size == 2 * weights and reconstruction_size == 4 * weights,
            "role byte geometry")
    require(binding["source_audit"] == expected_source_audit_receipt_sha256,
            "binding source audit receipt pin")

    audit_payload = read_regular(source_audit_receipt_path, 8 << 20)
    require(sha256(audit_payload) == expected_source_audit_receipt_sha256,
            "source audit receipt SHA256")
    audit = strict_json(audit_payload, "source audit")
    input_payload = read_regular(input_manifest_path, 32 << 20)
    auditor_payload = read_regular(auditor_source_manifest_path, 32 << 20)
    require(sha256(input_payload) == binding["input_manifest_sha256"]
            == audit.get("input_manifest_sha256"), "actual input manifest SHA256")
    require(sha256(auditor_payload) == binding["auditor_source_manifest_sha256"]
            == audit.get("auditor_source_manifest_sha256"),
            "actual auditor source manifest SHA256")
    strict_json(input_payload, "actual input manifest")
    strict_json(auditor_payload, "actual auditor source manifest")
    require(audit.get("status") == "INDEPENDENT_SOURCE_AND_COARSE_RECONSTRUCTION_AUDIT_PASS",
            "source audit PASS")
    publication = {row.get("name"): row for row in audit.get("publication_members", [])}
    require(len(publication) == len(audit.get("publication_members", [])),
            "unique publication members")
    require(publication.get("COARSE.bin") == {
        "name": "COARSE.bin", "bytes": coarse_size, "sha256": coarse_sha,
    }, "independent coarse publication")
    roles = {row.get("role"): row for row in audit.get("input_roles", [])}
    require(len(roles) == len(audit.get("input_roles", [])), "unique audited roles")
    require(roles.get(role) == {
        "role": role, "bytes": source_size, "sha256": source_sha,
    }, "independent source role")
    require(audit.get("reconstruction_f32_sha256", {}).get(role) == reconstruction_sha,
            "independent coarse reconstruction")

    coarse_payload = read_regular(coarse_artifact_path, coarse_size)
    source_payload = read_regular(source_bf16_path, source_size)
    reconstruction_payload = read_regular(coarse_reconstruction_f32_path, reconstruction_size)
    require(len(coarse_payload) == coarse_size and sha256(coarse_payload) == coarse_sha,
            "literal coarse artifact")
    require(len(source_payload) == source_size and sha256(source_payload) == source_sha,
            "literal source")
    require(len(reconstruction_payload) == reconstruction_size
            and sha256(reconstruction_payload) == reconstruction_sha,
            "literal coarse reconstruction")
    source_u16 = np.frombuffer(source_payload, dtype="<u2")
    source = (source_u16.astype("<u4") << 16).view("<f4").reshape(tuple(shape))
    coarse = np.frombuffer(reconstruction_payload, dtype="<f4").reshape(tuple(shape))
    require(np.all(np.isfinite(source)) and np.all(np.isfinite(coarse)), "finite role arrays")
    return {
        "role": role, "shape": tuple(shape), "source": source.astype(np.float64),
        "coarse": coarse.astype(np.float64), "source_sha256": source_sha,
        "coarse_artifact_payload": coarse_payload, "coarse_artifact_sha256": coarse_sha,
        "coarse_reconstruction_sha256": reconstruction_sha,
        "binding_sha256": expected_binding_sha256,
        "actual_input_manifest_opened": True,
        "actual_auditor_source_manifest_opened": True,
    }
