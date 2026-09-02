#!/usr/bin/env python3
"""Fail-closed binding of source, independent coarse artifact and reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "tactic-ramanujan384-input-binding-v0"
STATUS = "INDEPENDENT_COARSE_AND_SOURCE_HASHES_REQUIRED"
ROLES = ("gate", "up", "down_transposed")


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
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(AuthenticationError(f"{label} nonfinite {token}")),
    )
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, maximum: int) -> bytes:
    absolute = path.resolve(strict=True)
    require(absolute == path.absolute(), "path must be canonical and contain no symlink")
    descriptor = os.open(
        os.fspath(absolute),
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "regular single-link input")
        require(0 < before.st_size <= maximum, "bounded nonempty input")
        output = bytearray()
        while len(output) < before.st_size:
            row = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(row), "short authenticated read")
            output.extend(row)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
            "authenticated input identity drift",
        )
        return bytes(output)
    finally:
        os.close(descriptor)


def _digest_record(record: Mapping[str, Any], label: str) -> tuple[int, str]:
    require(isinstance(record, Mapping) and set(record) == {"bytes", "sha256"}, f"{label} digest record")
    size = record["bytes"]
    digest = record["sha256"]
    require(type(size) is int and size > 0 and isinstance(digest, str) and len(digest) == 64,
            f"{label} digest fields")
    return size, digest


def authenticate_role(
    *,
    binding_path: Path,
    expected_binding_sha256: str,
    audit_receipt_path: Path,
    coarse_artifact_path: Path,
    source_bf16_path: Path,
    coarse_reconstruction_f32_path: Path,
) -> dict[str, Any]:
    require(isinstance(expected_binding_sha256, str) and len(expected_binding_sha256) == 64,
            "expected binding SHA256 required")
    binding_payload = read_regular(binding_path, 1 << 20)
    require(sha256(binding_payload) == expected_binding_sha256, "binding SHA256")
    binding = strict_json(binding_payload, "binding")
    require(set(binding) == {
        "schema", "status", "role", "shape", "normalized_layout", "source",
        "coarse_artifact", "coarse_reconstruction", "independent_audit",
        "input_manifest_sha256",
    }, "binding exact schema")
    require(binding["schema"] == SCHEMA and binding["status"] == STATUS, "binding identity")
    role = binding["role"]
    shape = binding["shape"]
    require(role in ROLES, "binding role")
    require(isinstance(shape, list) and len(shape) == 2
            and all(type(value) is int and value > 0 for value in shape), "binding universal shape")
    require(binding["normalized_layout"] == "intermediate_by_hidden_row_major", "normalized role layout")
    weights = shape[0] * shape[1]
    source_size, source_sha = _digest_record(binding["source"], "source")
    coarse_size, coarse_sha = _digest_record(binding["coarse_artifact"], "coarse artifact")
    reconstruction_size, reconstruction_sha = _digest_record(binding["coarse_reconstruction"], "coarse reconstruction")
    require(source_size == 2 * weights and reconstruction_size == 4 * weights, "role byte geometry")
    independent = binding["independent_audit"]
    require(isinstance(independent, dict) and set(independent) == {
        "receipt_sha256", "auditor_source_manifest_sha256"
    }, "independent audit pins")
    require(all(isinstance(independent[key], str) and len(independent[key]) == 64 for key in independent),
            "independent audit SHA256 pins")
    require(isinstance(binding["input_manifest_sha256"], str) and len(binding["input_manifest_sha256"]) == 64,
            "input manifest SHA256 pin")

    audit_payload = read_regular(audit_receipt_path, 4 << 20)
    require(sha256(audit_payload) == independent["receipt_sha256"], "independent audit receipt SHA256")
    audit = strict_json(audit_payload, "independent audit")
    require(audit.get("auditor_source_manifest_sha256") == independent["auditor_source_manifest_sha256"],
            "auditor source-manifest pin")
    require(audit.get("input_manifest_sha256") == binding["input_manifest_sha256"], "input manifest pin")
    require(audit.get("independent_packet_parser_and_causal_CPU_decoder_used") is True,
            "independent causal decoder evidence")
    require(audit.get("literal_COARSE_canonical_reencode_matches") is True,
            "independent canonical coarse reencode")
    publication_rows = audit.get("publication_members", [])
    require(isinstance(publication_rows, list) and all(isinstance(row, dict) for row in publication_rows),
            "independent publication rows")
    publication_names = [row.get("name") for row in publication_rows]
    require(len(publication_names) == len(set(publication_names)), "duplicate independent publication member")
    publication = {row.get("name"): row for row in publication_rows}
    require(publication.get("COARSE.bin", {}).get("bytes") == coarse_size
            and publication.get("COARSE.bin", {}).get("sha256") == coarse_sha,
            "coarse artifact independent publication pin")
    role_rows = audit.get("input_roles", [])
    require(isinstance(role_rows, list) and all(isinstance(row, dict) for row in role_rows),
            "independent input role rows")
    role_names = [row.get("role") for row in role_rows]
    require(len(role_names) == len(set(role_names)), "duplicate independent input role")
    audit_roles = {row.get("role"): row for row in role_rows}
    require(audit_roles.get(role, {}).get("bytes") == source_size
            and audit_roles.get(role, {}).get("sha256") == source_sha,
            "source role independent pin")
    recomputed = audit.get("recomputed", {})
    require(isinstance(recomputed, dict)
            and recomputed.get("reconstruction_f32_sha256", {}).get(role) == reconstruction_sha,
            "coarse reconstruction independent pin")

    coarse_payload = read_regular(coarse_artifact_path, coarse_size)
    source_payload = read_regular(source_bf16_path, source_size)
    reconstruction_payload = read_regular(coarse_reconstruction_f32_path, reconstruction_size)
    require(len(coarse_payload) == coarse_size and sha256(coarse_payload) == coarse_sha,
            "literal coarse artifact hash")
    require(len(source_payload) == source_size and sha256(source_payload) == source_sha,
            "literal source role hash")
    require(len(reconstruction_payload) == reconstruction_size and sha256(reconstruction_payload) == reconstruction_sha,
            "literal coarse reconstruction hash")

    import numpy as np
    source_u16 = np.frombuffer(source_payload, dtype="<u2")
    source_f32 = (source_u16.astype("<u4") << 16).view("<f4")
    reconstruction = np.frombuffer(reconstruction_payload, dtype="<f4")
    require(source_f32.size == weights and reconstruction.size == weights, "decoded role geometry")
    source = source_f32.reshape(tuple(shape)).astype(np.float64)
    coarse = reconstruction.reshape(tuple(shape)).astype(np.float64)
    return {
        "role": role,
        "shape": tuple(shape),
        "weights": weights,
        "source": source,
        "coarse": coarse,
        "coarse_artifact_bytes": coarse_size,
        "coarse_artifact_payload": coarse_payload,
        "binding_sha256": expected_binding_sha256,
        "audit_receipt_sha256": independent["receipt_sha256"],
        "source_sha256": source_sha,
        "coarse_reconstruction_sha256": reconstruction_sha,
        "qwen_or_model_identity_used": False,
    }
