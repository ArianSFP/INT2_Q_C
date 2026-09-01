#!/usr/bin/env python3
"""Authorized CuPy opportunity gate for FOSP-ARX-v2-DIRECT.

This distinct v2 deliberately has no marginal-KLT early gate.  Its first
scientific gate is the full cross-role 3x3 predecessor-to-target family that
the v1 gate failed to contain.  The module has no heavy top-level imports, no
selectable source manifest, and no pinned-panel argument.  Source access is
possible only after an external one-shot authorization binds an independently
audited package, an independently audited source-free runtime calibration,
the exact fixed auxiliary bindings, the Python runtime, and a create-new
output path.

The calculation is an optimistic source oracle, not a compressed-stream
result.  It charges the inline 783-byte permutation and 767*9 FP16
coefficients, evaluates an actually FP16-rounded legal path, and compares
against identically optimized moment-matched Gaussian controls.  It does not
quantize the residual payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "free_order_swiglu_path_auxiliary_result_v2"
AUTHORIZATION_SCHEMA = "free_order_swiglu_path_one_shot_authorization_v2"
ROWS = 768
COLS = 2048
ROLES = 3
WEIGHTS_PER_EXPERT = ROWS * COLS * ROLES
REQUIRED_S = -0.5 * math.log2(0.8)
RATES = (2.15, 2.30, 2.50)
HEADER_BYTES = 64
FACTORADIC_BYTES = 783
FACTORADIC_BITS = FACTORADIC_BYTES * 8
FP16_COEFFICIENTS_PER_EDGE = 9
PATH_EDGES = ROWS - 1
FP16_COEFFICIENT_BITS = PATH_EDGES * FP16_COEFFICIENTS_PER_EDGE * 16
TOTAL_SIDE_BITS = HEADER_BYTES * 8 + FACTORADIC_BITS + FP16_COEFFICIENT_BITS
SIDE_BPW = TOTAL_SIDE_BITS / WEIGHTS_PER_EXPERT
REQUIRED_GROSS_S = REQUIRED_S + SIDE_BPW
CONTROL_SEEDS = (
    26_090_101,
    26_090_119,
    26_090_143,
    26_090_171,
    26_090_207,
    26_090_231,
    26_090_263,
    26_090_299,
)
BINDINGS_NAME = "source_bindings.json"
BINDINGS_SHA256 = "3454b718a65efc02c32463f955c10ff393f4218fac04f358107960ff3735990d"
MANIFEST_NAME = "ARTIFACT_SHA256SUMS.txt"
SOURCE_RECEIPT_NAME = "source_only_receipt.json"
FROZEN_FILES = (
    "README.md",
    "calibrate_runtime.py",
    "create_authorization.py",
    "free_order_oracle_v2.py",
    "protocol_lock.json",
    "source_bindings.json",
    "source_only_receipt.json",
    "test_source_only.py",
    "verify_package.py",
)
SHA_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


class ProtocolError(RuntimeError):
    """Fail-closed protocol violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ceil_log2_factorial(n: int) -> int:
    if n < 0:
        raise ValueError("negative factorial")
    return (math.factorial(n) - 1).bit_length()


def rank_permutation(permutation: Sequence[int]) -> int:
    n = len(permutation)
    if sorted(permutation) != list(range(n)):
        raise ValueError("not a permutation")
    available = list(range(n))
    rank = 0
    for index, value in enumerate(permutation):
        position = available.index(value)
        rank += position * math.factorial(n - index - 1)
        del available[position]
    return rank


def unrank_permutation(n: int, rank: int) -> tuple[int, ...]:
    if n < 0 or rank < 0 or rank >= math.factorial(n):
        raise ValueError("factoradic rank outside domain")
    available = list(range(n))
    output: list[int] = []
    for remaining in range(n, 0, -1):
        factorial = math.factorial(remaining - 1)
        position, rank = divmod(rank, factorial)
        output.append(available.pop(position))
    return tuple(output)


def serialize_permutation(permutation: Sequence[int]) -> bytes:
    rank = rank_permutation(permutation)
    encoded = rank.to_bytes(FACTORADIC_BYTES, "big", signed=False)
    if len(encoded) != FACTORADIC_BYTES:
        raise AssertionError("factoradic physical width drift")
    if unrank_permutation(len(permutation), int.from_bytes(encoded, "big")) != tuple(permutation):
        raise AssertionError("factoradic roundtrip failed")
    return encoded


def frame_ledger(rate: float) -> dict[str, Any]:
    if rate not in RATES:
        raise ValueError("rate is not frozen")
    frame_bytes = math.floor(WEIGHTS_PER_EXPERT * rate / 8.0)
    frame_bits = frame_bytes * 8
    payload_bits = frame_bits - TOTAL_SIDE_BITS
    if payload_bits <= 0:
        raise AssertionError("side ledger exhausts frame")
    cold_page_bytes = (math.ceil(frame_bytes / 4096) + 1) * 4096
    return {
        "requested_rate_bpw": rate,
        "frame_bytes": frame_bytes,
        "actual_rate_bpw": frame_bits / WEIGHTS_PER_EXPERT,
        "header_bits": HEADER_BYTES * 8,
        "factoradic_bits": FACTORADIC_BITS,
        "fp16_coefficient_bits": FP16_COEFFICIENT_BITS,
        "total_side_bits": TOTAL_SIDE_BITS,
        "side_bpw": SIDE_BPW,
        "residual_payload_bits": payload_bits,
        "residual_payload_bpw": payload_bits / WEIGHTS_PER_EXPERT,
        "required_gross_s_bpw": REQUIRED_GROSS_S,
        "logical_byte_read_amplification": 1.0,
        "cold_page_bytes_including_one_shared_page": cold_page_bytes,
        "cold_page_amplification": cold_page_bytes / frame_bytes,
        "strictly_below_2x": cold_page_bytes / frame_bytes < 2.0,
    }


def _regular_bytes_no_follow(path: Path, expected_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProtocolError(f"not a regular file: {path}")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ProtocolError(f"wrong byte count for {path}: {before.st_size} != {expected_bytes}")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ProtocolError(f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProtocolError(f"file grew during read: {path}")
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise ProtocolError(f"file changed while open: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _canonical_existing_path(path: Path, kind: str) -> Path:
    """Reject aliases, dot segments, case aliases, and symlinked components."""
    raw = os.fspath(path)
    if not path.is_absolute() or raw != os.path.normpath(raw):
        raise ProtocolError(f"{kind} must use one normalized absolute spelling")
    anchor = Path(path.anchor)
    cursor = anchor
    for part in path.parts[1:]:
        cursor = cursor / part
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ProtocolError(f"{kind} contains a symlinked component: {cursor}")
    resolved = path.resolve(strict=True)
    if os.path.normcase(os.fspath(resolved)) != os.path.normcase(raw):
        raise ProtocolError(f"{kind} spelling is not canonical")
    return resolved


def _path_from_frozen_spelling(raw: str, kind: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ProtocolError(f"{kind} path spelling missing")
    if raw != os.path.normpath(raw) or not os.path.isabs(raw):
        raise ProtocolError(f"{kind} must use one normalized absolute spelling")
    return Path(raw)


def _canonical_new_path(path: Path, kind: str) -> tuple[Path, Path]:
    raw = os.fspath(path)
    if not path.is_absolute() or raw != os.path.normpath(raw):
        raise ProtocolError(f"{kind} must use one normalized absolute spelling")
    if path.exists() or path.is_symlink():
        raise ProtocolError(f"{kind} already exists")
    if path.name in ("", ".", ".."):
        raise ProtocolError(f"{kind} filename invalid")
    parent = _canonical_existing_path(path.parent, f"{kind} parent")
    canonical = parent / path.name
    if os.path.normcase(os.fspath(canonical)) != os.path.normcase(raw):
        raise ProtocolError(f"{kind} spelling is not canonical")
    return canonical, parent


def _directory_descriptor(path: Path, kind: str) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(path), flags)
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise ProtocolError(f"{kind} descriptor is not a directory")
    path_info = path.lstat()
    if (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino):
        os.close(descriptor)
        raise ProtocolError(f"{kind} changed while opening")
    return descriptor


def _regular_bytes_at(
    root_descriptor: int,
    relative: Path,
    expected_bytes: int | None = None,
) -> bytes:
    """Descriptor-relative O_NOFOLLOW traversal below a held directory."""
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ProtocolError("unsafe descriptor-relative source path")
    cursor = os.dup(root_descriptor)
    try:
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        for part in relative.parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=cursor)
            os.close(cursor)
            cursor = next_descriptor
        file_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = os.open(relative.parts[-1], file_flags, dir_fd=cursor)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ProtocolError(f"descriptor-relative source is not regular: {relative}")
            if expected_bytes is not None and before.st_size != expected_bytes:
                raise ProtocolError(
                    f"wrong byte count for {relative}: {before.st_size} != {expected_bytes}"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1 << 20, remaining))
                if not chunk:
                    raise ProtocolError(f"short read: {relative}")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ProtocolError(f"file grew during read: {relative}")
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ProtocolError(f"source changed while open: {relative}")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(cursor)


def _verify_canonical_seal(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    observed = str(unsigned.pop(field, ""))
    if re.fullmatch(r"[0-9a-f]{64}", observed) is None:
        raise ProtocolError(f"{label} canonical seal missing")
    if canonical_sha256(unsigned) != observed:
        raise ProtocolError(f"{label} canonical seal mismatch")
    return observed


def _audit_manifest_binds(raw: bytes, receipt_name: str, receipt_sha256: str, label: str) -> None:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = SHA_LINE.fullmatch(line)
        if match is None:
            raise ProtocolError(f"malformed {label} manifest line {line_number}")
        digest, name = match.groups()
        if name in rows:
            raise ProtocolError(f"duplicate {label} manifest entry: {name}")
        rows[name] = digest
    if rows.get(receipt_name) != receipt_sha256:
        raise ProtocolError(f"{label} manifest does not bind exact receipt")
    if "verify_audit.py" not in rows or len(rows) < 2:
        raise ProtocolError(f"{label} manifest lacks an independent verifier closure")


def _artifact_rows(package: Path) -> tuple[dict[str, str], bytes]:
    raw = _regular_bytes_no_follow(package / MANIFEST_NAME)
    rows: dict[str, str] = {}
    for line_number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = SHA_LINE.fullmatch(line)
        if match is None:
            raise ProtocolError(f"malformed artifact manifest line {line_number}")
        digest, name = match.groups()
        if name in rows:
            raise ProtocolError(f"duplicate artifact manifest entry: {name}")
        rows[name] = digest
    if set(rows) != set(FROZEN_FILES):
        raise ProtocolError("artifact manifest closure mismatch")
    observed = {path.name for path in package.iterdir() if path.is_file() and path.name != MANIFEST_NAME}
    if observed != set(FROZEN_FILES):
        raise ProtocolError("package contains an unsealed or missing regular file")
    for name, expected in rows.items():
        if sha256_bytes(_regular_bytes_no_follow(package / name)) != expected:
            raise ProtocolError(f"artifact hash mismatch: {name}")
    return rows, raw


def _load_bindings(package: Path) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes_no_follow(package / BINDINGS_NAME)
    if sha256_bytes(raw) != BINDINGS_SHA256:
        raise ProtocolError("fixed source-bindings hash mismatch")
    value = json.loads(raw.decode("utf-8"))
    if value.get("schema") != "free_order_swiglu_path_auxiliary_bindings_v1":
        raise ProtocolError("wrong source-bindings schema")
    experts = value.get("experts")
    if not isinstance(experts, list) or len(experts) != 2:
        raise ProtocolError("exactly two fixed auxiliary experts are required")
    if [int(row["ordinal"]) for row in experts] != [0, 1]:
        raise ProtocolError("expert ordinals drifted")
    for expert in experts:
        if [row.get("role") for row in expert.get("roles", [])] != ["gate", "up", "down"]:
            raise ProtocolError("all three roles are required in frozen order")
    return value, raw


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _pairwise_disjoint(paths: Sequence[Path]) -> bool:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if _is_within(left, right) or _is_within(right, left):
                return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    return parser.parse_args()


def _load_authorization(
    args: argparse.Namespace,
    package: Path,
    artifact_rows: dict[str, str],
    artifact_raw: bytes,
) -> tuple[dict[str, Any], str, Path, int, Path, int]:
    if re.fullmatch(r"[0-9a-f]{64}", args.authorization_sha256) is None:
        raise ProtocolError("authorization SHA-256 syntax invalid")
    authorization_path = _canonical_existing_path(
        _path_from_frozen_spelling(args.authorization, "authorization"), "authorization"
    )
    if not authorization_path.is_file():
        raise ProtocolError("authorization is not a regular file")
    raw = _regular_bytes_no_follow(authorization_path)
    authorization_sha = sha256_bytes(raw)
    if authorization_sha != args.authorization_sha256:
        raise ProtocolError("authorization file hash mismatch")
    value = json.loads(raw.decode("utf-8"))
    if value.get("schema") != AUTHORIZATION_SCHEMA:
        raise ProtocolError("wrong authorization schema")
    if value.get("status") != "AUTHORIZED_ONE_SHOT_AUXILIARY_SOURCE_RUN":
        raise ProtocolError("authorization status is not executable")
    if value.get("one_shot") is not True or value.get("pinned_panel_authorized") is not False:
        raise ProtocolError("authorization scope drift")
    if value.get("scope_literal") != "FOSP_V2_AUXILIARY_DISCOVERY_ONLY_NO_PINNED_PANEL":
        raise ProtocolError("authorization discovery-only scope literal mismatch")
    _verify_canonical_seal(value, "canonical_unsigned_sha256", "authorization")

    binding = value.get("artifact_binding", {})
    required_binding = {
        "artifact_manifest_sha256": sha256_bytes(artifact_raw),
        "source_only_receipt_sha256": artifact_rows[SOURCE_RECEIPT_NAME],
        "runner_sha256": artifact_rows[Path(__file__).name],
        "source_bindings_sha256": BINDINGS_SHA256,
        "runtime_calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
    }
    if binding != required_binding:
        raise ProtocolError("authorization does not bind this exact package")

    root = _canonical_existing_path(
        _path_from_frozen_spelling(args.workspace_root, "workspace root"), "workspace root"
    )
    if not root.is_dir():
        raise ProtocolError("workspace root is not a directory")
    output, output_parent = _canonical_new_path(
        _path_from_frozen_spelling(args.output, "production output"), "production output"
    )
    authorization_parent = _canonical_existing_path(authorization_path.parent, "authorization parent")

    path_binding = value.get("path_binding", {})
    if path_binding != {
        "workspace_root": os.fspath(root),
        "output": os.fspath(output),
        "authorization_parent": os.fspath(authorization_parent),
    }:
        raise ProtocolError("authorization path binding mismatch")

    runtime = value.get("runtime_binding", {})
    if runtime.get("python_executable_resolved") != os.fspath(Path(sys.executable).resolve(strict=True)):
        raise ProtocolError("Python executable does not match audited runtime")
    if runtime.get("python_version") != platform.python_version():
        raise ProtocolError("Python version does not match audited runtime")
    if runtime.get("cuda_visible_devices") != "0":
        raise ProtocolError("authorization CUDA device drift")

    audit_path_values = value.get("audit_paths", {})
    expected_audit_keys = {
        "source_audit_manifest",
        "source_audit_receipt",
        "runtime_receipt",
        "runtime_audit_manifest",
        "runtime_audit_receipt",
    }
    if set(audit_path_values) != expected_audit_keys:
        raise ProtocolError("authorization audit path closure mismatch")
    audit_paths = {
        key: _canonical_existing_path(
            _path_from_frozen_spelling(str(audit_path_values[key]), key.replace("_", " ")),
            key.replace("_", " "),
        )
        for key in sorted(expected_audit_keys)
    }
    if audit_paths["source_audit_manifest"].name != "AUDIT_SHA256SUMS.txt":
        raise ProtocolError("source audit manifest filename is not frozen")
    if audit_paths["source_audit_receipt"].name != "audit_receipt.json":
        raise ProtocolError("source audit receipt filename is not frozen")
    if audit_paths["runtime_receipt"].name != "runtime_receipt.json":
        raise ProtocolError("runtime receipt filename is not frozen")
    if audit_paths["runtime_audit_manifest"].name != "AUDIT_SHA256SUMS.txt":
        raise ProtocolError("runtime audit manifest filename is not frozen")
    if audit_paths["runtime_audit_receipt"].name != "audit_receipt.json":
        raise ProtocolError("runtime audit receipt filename is not frozen")
    source_audit_parent = audit_paths["source_audit_manifest"].parent
    runtime_audit_parent = audit_paths["runtime_audit_manifest"].parent
    runtime_receipt_parent = audit_paths["runtime_receipt"].parent
    if audit_paths["source_audit_receipt"].parent != source_audit_parent:
        raise ProtocolError("source audit closure must share one directory")
    if audit_paths["runtime_audit_receipt"].parent != runtime_audit_parent:
        raise ProtocolError("runtime audit closure must share one directory")
    protected = (
        package,
        root,
        output_parent,
        authorization_parent,
        source_audit_parent,
        runtime_receipt_parent,
        runtime_audit_parent,
    )
    if not _pairwise_disjoint(protected):
        raise ProtocolError("package, source, output, authorization, and evidence roots must be disjoint")

    evidence_raw = {key: _regular_bytes_no_follow(path) for key, path in audit_paths.items()}
    evidence_sha = {key: sha256_bytes(payload) for key, payload in evidence_raw.items()}
    _audit_manifest_binds(
        evidence_raw["source_audit_manifest"],
        "audit_receipt.json",
        evidence_sha["source_audit_receipt"],
        "source audit",
    )
    _audit_manifest_binds(
        evidence_raw["runtime_audit_manifest"],
        "audit_receipt.json",
        evidence_sha["runtime_audit_receipt"],
        "runtime audit",
    )

    source_audit = json.loads(evidence_raw["source_audit_receipt"].decode("utf-8"))
    runtime_receipt = json.loads(evidence_raw["runtime_receipt"].decode("utf-8"))
    runtime_audit = json.loads(evidence_raw["runtime_audit_receipt"].decode("utf-8"))
    if not all(isinstance(item, dict) for item in (source_audit, runtime_receipt, runtime_audit)):
        raise ProtocolError("an external receipt root is not an object")
    source_internal = _verify_canonical_seal(
        source_audit, "canonical_unsigned_sha256", "source audit receipt"
    )
    runtime_internal = _verify_canonical_seal(
        runtime_receipt, "canonical_unsigned_sha256", "runtime receipt"
    )
    runtime_audit_internal = _verify_canonical_seal(
        runtime_audit, "canonical_unsigned_sha256", "runtime audit receipt"
    )

    expected_package = {
        "artifact_manifest_sha256": sha256_bytes(artifact_raw),
        "source_only_receipt_sha256": artifact_rows[SOURCE_RECEIPT_NAME],
        "runner_sha256": artifact_rows[Path(__file__).name],
        "runtime_calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
        "source_bindings_sha256": BINDINGS_SHA256,
    }
    if source_audit.get("schema") != "free-order-swiglu-path-v2-independent-source-audit-receipt-v1":
        raise ProtocolError("wrong independent source audit schema")
    if source_audit.get("status") != "PASS_V2_INDEPENDENT_SOURCE_AUDIT":
        raise ProtocolError("independent source audit is not PASS")
    if source_audit.get("artifact_set_status") != "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET":
        raise ProtocolError("independent source audit closure is not immutable PASS")
    if source_audit.get("audited_package") != expected_package:
        raise ProtocolError("independent source audit target mismatch")
    if source_audit.get("v1_counterexample_replay", {}).get("status") != "PASS_COUNTEREXAMPLE_REACHES_DIRECT_STAGE":
        raise ProtocolError("source audit did not replay the v1 containment counterexample")
    source_zero = source_audit.get("zero_access_ledger", {})
    for key in (
        "qwen_or_model_payload_files_opened",
        "qwen_or_model_payload_bytes_read",
        "pinned_panel_files_opened",
        "validation_files_opened",
        "cupy_imports",
        "cuda_api_calls",
        "gpu_device_calls",
        "external_data_fetches",
    ):
        if int(source_zero.get(key, -1)) != 0:
            raise ProtocolError(f"source audit does not prove zero {key}")

    if runtime_receipt.get("schema") != "free_order_swiglu_path_runtime_calibration_v2":
        raise ProtocolError("wrong runtime receipt schema")
    if runtime_receipt.get("status") != "PASS_SOURCE_FREE_FULL_GEOMETRY_RUNTIME_CALIBRATION":
        raise ProtocolError("source-free runtime calibration is not PASS")
    if runtime_receipt.get("artifact_binding") != {
        "artifact_manifest_sha256": sha256_bytes(artifact_raw),
        "runner_sha256": artifact_rows[Path(__file__).name],
        "calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
    }:
        raise ProtocolError("runtime receipt package target mismatch")
    runtime_zero = runtime_receipt.get("zero_access_ledger", {})
    for key in (
        "workspace_or_source_arguments_supported",
        "source_bindings_loaded",
        "qwen_or_model_payload_files_opened",
        "qwen_or_model_payload_bytes_read",
        "pinned_panel_files_opened",
        "validation_files_opened",
        "external_data_fetches",
        "production_result_files_opened",
        "production_gpu_jobs",
    ):
        if int(runtime_zero.get(key, -1)) != 0:
            raise ProtocolError(f"runtime receipt does not prove zero {key}")
    if int(runtime_zero.get("synthetic_gpu_jobs", -1)) != 1:
        raise ProtocolError("runtime receipt did not execute exactly one synthetic GPU job")

    if runtime_audit.get("schema") != "free-order-swiglu-path-v2-independent-runtime-audit-receipt-v1":
        raise ProtocolError("wrong independent runtime audit schema")
    if runtime_audit.get("status") != "PASS_V2_INDEPENDENT_RUNTIME_AUDIT":
        raise ProtocolError("independent runtime audit is not PASS")
    if runtime_audit.get("artifact_set_status") != "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET":
        raise ProtocolError("runtime audit closure is not immutable PASS")
    if runtime_audit.get("audited_package") != expected_package:
        raise ProtocolError("runtime audit package target mismatch")
    if runtime_audit.get("audited_runtime_receipt") != {
        "file_sha256": evidence_sha["runtime_receipt"],
        "internal_sha256": runtime_internal,
    }:
        raise ProtocolError("runtime audit did not bind the exact runtime receipt")
    runtime_audit_zero = runtime_audit.get("zero_access_ledger", {})
    for key in (
        "qwen_or_model_payload_files_opened",
        "qwen_or_model_payload_bytes_read",
        "pinned_panel_files_opened",
        "validation_files_opened",
        "production_result_files_opened",
        "production_gpu_jobs",
        "cupy_imports",
        "cuda_api_calls",
        "gpu_device_calls",
    ):
        if int(runtime_audit_zero.get(key, -1)) != 0:
            raise ProtocolError(f"runtime audit does not prove zero {key}")

    audits = value.get("audit_binding", {})
    expected_audits = {
        "source_audit_status": "PASS_V2_INDEPENDENT_SOURCE_AUDIT",
        "source_audit_manifest_sha256": evidence_sha["source_audit_manifest"],
        "source_audit_receipt_sha256": evidence_sha["source_audit_receipt"],
        "source_audit_receipt_internal_sha256": source_internal,
        "runtime_receipt_sha256": evidence_sha["runtime_receipt"],
        "runtime_receipt_internal_sha256": runtime_internal,
        "runtime_audit_status": "PASS_V2_INDEPENDENT_RUNTIME_AUDIT",
        "runtime_audit_manifest_sha256": evidence_sha["runtime_audit_manifest"],
        "runtime_audit_receipt_sha256": evidence_sha["runtime_audit_receipt"],
        "runtime_audit_receipt_internal_sha256": runtime_audit_internal,
    }
    if audits != expected_audits:
        raise ProtocolError("authorization audit binding does not match opened evidence")
    runtime_backend = runtime_receipt.get("backend", {})
    if value.get("runtime_binding") != runtime_backend:
        raise ProtocolError("authorization runtime tuple differs from opened calibration receipt")
    run_material = {
        "artifact_binding": value["artifact_binding"],
        "audit_binding": value["audit_binding"],
        "audit_paths": value["audit_paths"],
        "path_binding": value["path_binding"],
        "runtime_binding": value["runtime_binding"],
    }
    if value.get("run_id") != canonical_sha256(run_material):
        raise ProtocolError("authorization run-id seal mismatch")

    source_root_descriptor = _directory_descriptor(root, "workspace root")
    try:
        output_parent_descriptor = _directory_descriptor(output_parent, "output parent")
    except Exception:
        os.close(source_root_descriptor)
        raise
    return (
        value,
        authorization_sha,
        root,
        source_root_descriptor,
        output,
        output_parent_descriptor,
    )


def _decode_bf16(raw: bytes, shape: tuple[int, int], np: Any, cp: Any) -> Any:
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != math.prod(shape):
        raise ProtocolError("BF16 shape mismatch")
    gpu_words = cp.asarray(words, dtype=cp.uint16)
    values = (gpu_words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
    return values.reshape(shape).astype(cp.float64)


def _load_sources(
    root_descriptor: int,
    bindings: dict[str, Any],
    np: Any,
    cp: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    experts: list[Any] = []
    receipts: list[dict[str, Any]] = []
    expected_bytes = int(bindings["expected_geometry"]["bf16_bytes_per_matrix"])
    for expert in bindings["experts"]:
        role_arrays: list[Any] = []
        for role in expert["roles"]:
            relative = Path(str(role["relative_path"]))
            raw = _regular_bytes_at(root_descriptor, relative, expected_bytes)
            digest = sha256_bytes(raw)
            if digest != str(role["sha256"]):
                raise ProtocolError(f"source hash mismatch: {relative.as_posix()}")
            shape = tuple(int(x) for x in role["shape"])
            array = _decode_bf16(raw, shape, np, cp)
            if role["role"] == "down":
                array = cp.ascontiguousarray(array.T)
            if tuple(array.shape) != (ROWS, COLS):
                raise ProtocolError("canonical role shape mismatch")
            role_arrays.append(array)
            receipts.append(
                {
                    "ordinal": int(expert["ordinal"]),
                    "layer": int(expert["layer"]),
                    "expert": int(expert["expert"]),
                    "role": str(role["role"]),
                    "relative_path": relative.as_posix(),
                    "bytes": len(raw),
                    "sha256": digest,
                }
            )
        experts.append(cp.stack(role_arrays, axis=1))
    return experts, receipts


def _matched_control(source: Any, seed: int, cp: Any) -> tuple[Any, dict[str, float]]:
    random = cp.random.RandomState(seed)
    raw = random.standard_normal(source.shape, dtype=cp.float64)
    source_mean = cp.mean(source, axis=2)
    source_centered = source - source_mean[:, :, None]
    source_gram = cp.einsum("nrd,nsd->nrs", source_centered, source_centered)

    raw -= cp.mean(raw, axis=2)[:, :, None]
    raw_gram = cp.einsum("nrd,nsd->nrs", raw, raw)
    source_eval, source_evec = cp.linalg.eigh(source_gram)
    raw_eval, raw_evec = cp.linalg.eigh(raw_gram)
    if bool(cp.any(source_eval <= 0.0)) or bool(cp.any(raw_eval <= 0.0)):
        raise ProtocolError("degenerate matched-control Gram")
    source_root = cp.einsum("nri,ni,nsi->nrs", source_evec, cp.sqrt(source_eval), source_evec)
    raw_invroot = cp.einsum("nri,ni,nsi->nrs", raw_evec, 1.0 / cp.sqrt(raw_eval), raw_evec)
    transform = cp.einsum("nri,nis->nrs", source_root, raw_invroot)
    control = cp.einsum("nrs,nsd->nrd", transform, raw) + source_mean[:, :, None]

    control_mean = cp.mean(control, axis=2)
    control_centered = control - control_mean[:, :, None]
    control_gram = cp.einsum("nrd,nsd->nrs", control_centered, control_centered)
    mean_error = float(cp.max(cp.abs(control_mean - source_mean)).item())
    gram_relative = float(
        (cp.max(cp.abs(control_gram - source_gram)) / cp.max(cp.abs(source_gram))).item()
    )
    if mean_error > 2e-13 or gram_relative > 2e-12:
        raise ProtocolError(f"matched-control closure failed: mean={mean_error}, gram={gram_relative}")
    return control, {
        "maximum_absolute_role_mean_error": mean_error,
        "maximum_relative_centered_gram_error": gram_relative,
    }


def _pair_scores(expert: Any, cp: Any) -> tuple[Any, Any, Any]:
    if expert.ndim != 3 or int(expert.shape[1]) != ROLES:
        raise ProtocolError("pair-score tensor must be neuron x 3 roles x coordinates")
    neurons = int(expert.shape[0])
    gram = cp.einsum("nrd,nsd->nrs", expert, expert)
    inverse = cp.linalg.inv(gram)
    cross = cp.empty((neurons, neurons, ROLES, ROLES), dtype=cp.float64)
    for target_role in range(ROLES):
        for predecessor_role in range(ROLES):
            cross[:, :, target_role, predecessor_role] = (
                expert[:, target_role, :] @ expert[:, predecessor_role, :].T
            )
    full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)
    indices = cp.arange(neurons)
    full[indices, indices] = -cp.inf
    return full, cross, inverse


def _cycles_from_assignment(predecessor: Sequence[int]) -> list[list[int]]:
    neurons = len(predecessor)
    successor = [-1] * neurons
    for target, pred in enumerate(predecessor):
        if target == pred or pred < 0 or pred >= neurons or successor[pred] != -1:
            raise ProtocolError("assignment is not a non-self cycle cover")
        successor[pred] = target
    cycles: list[list[int]] = []
    unseen = set(range(neurons))
    while unseen:
        start = min(unseen)
        cycle: list[int] = []
        node = start
        while node not in cycle:
            if node not in unseen:
                raise ProtocolError("cycle-cover traversal collision")
            cycle.append(node)
            unseen.remove(node)
            node = successor[node]
        if node != start:
            raise ProtocolError("cycle does not close at its start")
        cycles.append(cycle)
    return cycles


def _legal_path_from_cycle_cover(scores: Any, np: Any, linear_sum_assignment: Any) -> dict[str, Any]:
    matrix = np.asarray(scores, dtype=np.float64)
    neurons = int(matrix.shape[0])
    if matrix.shape != (neurons, neurons):
        raise ProtocolError("score matrix is not square")
    rows, columns = linear_sum_assignment(-matrix)
    if not np.array_equal(rows, np.arange(neurons)):
        raise ProtocolError("unexpected assignment row order")
    predecessor = [int(value) for value in columns]
    cycles = _cycles_from_assignment(predecessor)
    successor = {pred: target for target, pred in enumerate(predecessor)}
    segments: list[list[int]] = []
    dropped: list[dict[str, Any]] = []
    for cycle in cycles:
        weakest_target = min(cycle, key=lambda target: (matrix[target, predecessor[target]], target))
        segment = [weakest_target]
        while len(segment) < len(cycle):
            segment.append(successor[segment[-1]])
        segments.append(segment)
        dropped.append(
            {
                "predecessor": predecessor[weakest_target],
                "target": weakest_target,
                "capture": float(matrix[weakest_target, predecessor[weakest_target]]),
            }
        )
    segments.sort(key=lambda row: row[0])
    path = [node for segment in segments for node in segment]
    if sorted(path) != list(range(neurons)):
        raise ProtocolError("cycle-cover construction did not produce a permutation")
    captures = [float(matrix[target, pred]) for pred, target in zip(path[:-1], path[1:])]
    return {
        "cycle_count": len(cycles),
        "cycle_cover_capture": float(
            math.fsum(float(matrix[target, predecessor[target]]) for target in range(neurons))
        ),
        "dropped_edges": dropped,
        "path": path,
        "legal_path_capture": math.fsum(captures),
    }


def _metric_from_residual(residual: float, energy: float) -> dict[str, float]:
    if not (math.isfinite(residual) and math.isfinite(energy) and energy > 0.0 and residual > 0.0):
        raise ProtocolError("invalid residual metric inputs")
    ratio = residual / energy
    if ratio > 1.0 + 1e-10:
        # A rounded predictor can be worse than zero.  That is a valid miss,
        # but report its negative s rather than hiding it.
        pass
    s_value = -0.5 * math.log2(ratio)
    return {
        "residual_energy": residual,
        "residual_ratio": ratio,
        "energy_reduction": 1.0 - ratio,
        "s_bpw": s_value,
        "net_s_after_side_bpw": s_value - SIDE_BPW,
        "projected_F_after_side": 2.0 ** (-2.0 * (s_value - SIDE_BPW)),
    }


def _metric_from_capture(capture: float, energy: float) -> dict[str, float]:
    residual = energy - capture
    if residual <= 0.0:
        raise ProtocolError("optimistic capture exhausted source energy")
    value = _metric_from_residual(residual, energy)
    value["capture"] = capture
    return value


def _fp16_path_replay(
    expert: Any,
    path: Sequence[int],
    cross: Any,
    inverse: Any,
    np: Any,
    cp: Any,
) -> dict[str, Any]:
    neurons = int(expert.shape[0])
    if len(path) != neurons or sorted(path) != list(range(neurons)):
        raise ProtocolError("FP16 replay path is not a permutation")
    predecessors = cp.asarray(path[:-1], dtype=cp.int64)
    targets = cp.asarray(path[1:], dtype=cp.int64)
    selected_cross = cross[targets, predecessors]
    exact_coefficients = cp.einsum("eab,ebc->eac", selected_cross, inverse[predecessors])
    fp16_coefficients = exact_coefficients.astype(cp.float16)
    replay_coefficients = fp16_coefficients.astype(cp.float64)
    predicted = cp.einsum(
        "eab,ebd->ead",
        replay_coefficients,
        expert[predecessors],
    )
    residuals = expert[targets] - predicted
    anchor_energy = float(cp.sum(expert[int(path[0])] ** 2, dtype=cp.float64).item())
    edge_residual_energy = float(cp.sum(residuals * residuals, dtype=cp.float64).item())
    coefficient_host = cp.asnumpy(fp16_coefficients)
    coefficient_bytes = coefficient_host.astype("<f2", copy=False).tobytes()
    if len(coefficient_bytes) != (neurons - 1) * ROLES * ROLES * 2:
        raise ProtocolError("FP16 coefficient byte ledger mismatch")
    return {
        "anchor_energy": anchor_energy,
        "edge_residual_energy": edge_residual_energy,
        "residual_energy": anchor_energy + edge_residual_energy,
        "coefficient_count": (neurons - 1) * ROLES * ROLES,
        "coefficient_bytes": len(coefficient_bytes),
        "coefficient_sha256_f16le": sha256_bytes(coefficient_bytes),
        "maximum_absolute_exact_coefficient": float(cp.max(cp.abs(exact_coefficients)).item()),
        "maximum_absolute_rounding_error": float(
            cp.max(cp.abs(replay_coefficients - exact_coefficients)).item()
        ),
    }


def _pair_panel(experts: Sequence[Any], np: Any, cp: Any, linear_sum_assignment: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_energy = 0.0
    total_relaxed_capture = 0.0
    total_legal_capture = 0.0
    total_fp16_residual = 0.0
    for ordinal, expert in enumerate(experts):
        if tuple(expert.shape) != (ROWS, ROLES, COLS):
            raise ProtocolError("production expert geometry drift")
        energy = float(cp.sum(expert * expert, dtype=cp.float64).item())
        scores, cross, inverse = _pair_scores(expert, cp)
        score_host = cp.asnumpy(scores)
        relaxed_capture = float(np.sum(np.max(score_host, axis=1), dtype=np.float64))
        legal = _legal_path_from_cycle_cover(score_host, np, linear_sum_assignment)
        path = [int(value) for value in legal.pop("path")]
        permutation_bytes = serialize_permutation(path)
        fp16 = _fp16_path_replay(expert, path, cross, inverse, np, cp)
        path_u16 = np.asarray(path, dtype="<u2").tobytes()
        row = {
            "expert_ordinal": ordinal,
            "source_energy": energy,
            "relaxed_reuse_exact_capture": relaxed_capture,
            "legal_exact": legal,
            "legal_fp16": fp16,
            "path_sha256_u16le": sha256_bytes(path_u16),
            "factoradic_rank_sha256_783be": sha256_bytes(permutation_bytes),
            "factoradic_physical_bytes": len(permutation_bytes),
        }
        rows.append(row)
        total_energy += energy
        total_relaxed_capture += relaxed_capture
        total_legal_capture += float(legal["legal_path_capture"])
        total_fp16_residual += float(fp16["residual_energy"])
        del scores, cross, inverse
        cp.get_default_memory_pool().free_all_blocks()
    return {
        "experts": rows,
        "total_source_energy": total_energy,
        "relaxed_reuse_exact": _metric_from_capture(total_relaxed_capture, total_energy),
        "legal_path_exact": _metric_from_capture(total_legal_capture, total_energy),
        "legal_path_fp16": _metric_from_residual(total_fp16_residual, total_energy),
    }


def _metric_for_subset(panel: dict[str, Any], metric: str, kept: Sequence[int]) -> float:
    rows = [panel["experts"][index] for index in kept]
    energy = math.fsum(float(row["source_energy"]) for row in rows)
    if metric == "relaxed_reuse_exact":
        capture = math.fsum(float(row["relaxed_reuse_exact_capture"]) for row in rows)
        return float(_metric_from_capture(capture, energy)["s_bpw"])
    if metric == "legal_path_exact":
        capture = math.fsum(float(row["legal_exact"]["legal_path_capture"]) for row in rows)
        return float(_metric_from_capture(capture, energy)["s_bpw"])
    if metric == "legal_path_fp16":
        residual = math.fsum(float(row["legal_fp16"]["residual_energy"]) for row in rows)
        return float(_metric_from_residual(residual, energy)["s_bpw"])
    raise ValueError("unknown metric")


def _jackknife_se(values: Sequence[float]) -> float:
    count = len(values)
    if count < 2:
        return 0.0
    mean = math.fsum(values) / count
    return math.sqrt((count - 1.0) / count * math.fsum((value - mean) ** 2 for value in values))


def _controlled_statistic(
    qwen: dict[str, Any],
    controls: Sequence[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    qwen_s = float(qwen[metric]["s_bpw"])
    control_s = [float(row[metric]["s_bpw"]) for row in controls]
    control_mean = math.fsum(control_s) / len(control_s)
    control_mc_se = math.sqrt(
        math.fsum((value - control_mean) ** 2 for value in control_s)
        / (len(control_s) * (len(control_s) - 1))
    )
    delete_estimates: list[float] = []
    expert_count = len(qwen["experts"])
    for omitted in range(expert_count):
        kept = [index for index in range(expert_count) if index != omitted]
        qwen_delete = _metric_for_subset(qwen, metric, kept)
        control_delete = math.fsum(_metric_for_subset(row, metric, kept) for row in controls) / len(controls)
        delete_estimates.append(qwen_delete - control_delete)
    jackknife_se = _jackknife_se(delete_estimates)
    combined_se = math.hypot(control_mc_se, jackknife_se)
    excess = qwen_s - control_mean
    optimistic = excess + 3.0 * combined_se
    net_optimistic = optimistic - SIDE_BPW
    return {
        "metric": metric,
        "qwen_gross_s_bpw": qwen_s,
        "control_s_bpw": control_s,
        "control_mean_s_bpw": control_mean,
        "control_mc_se_bpw": control_mc_se,
        "delete_one_expert_estimates_s_bpw": delete_estimates,
        "delete_one_expert_jackknife_se_bpw": jackknife_se,
        "combined_se_bpw": combined_se,
        "qwen_specific_excess_s_bpw": excess,
        "optimistic_excess_plus_3se_s_bpw": optimistic,
        "net_optimistic_s_after_side_bpw": net_optimistic,
        "projected_optimistic_F_after_side": 2.0 ** (-2.0 * net_optimistic),
        "required_gross_s_bpw": REQUIRED_GROSS_S,
        "upper_confidence_survives_target": optimistic >= REQUIRED_GROSS_S,
    }


def _direct_stage(experts: Sequence[Any], np: Any, cp: Any, linear_sum_assignment: Any) -> dict[str, Any]:
    qwen = _pair_panel(experts, np, cp, linear_sum_assignment)
    qwen_relaxed_s = float(qwen["relaxed_reuse_exact"]["s_bpw"])
    if qwen_relaxed_s < REQUIRED_GROSS_S:
        return {
            "qwen": qwen,
            "controls": [],
            "control_moment_closure": [],
            "statistics": {},
            "decision": "HARD_KILL_GROSS_RELAXED_UPPER_BOUND_BELOW_SIDE_ADJUSTED_TARGET",
            "early_stop": True,
            "reason": (
                "Even the illegal exact-coefficient reuse relaxation is below the target after "
                "the physical header, permutation, and FP16 coefficient charge. Controls cannot "
                "turn this gross deterministic miss into an achievable path."
            ),
        }

    controls: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    for replicate, base_seed in enumerate(CONTROL_SEEDS):
        arrays: list[Any] = []
        replicate_closure: list[dict[str, float]] = []
        for ordinal, expert in enumerate(experts):
            control, closure = _matched_control(expert, base_seed + 1009 * ordinal, cp)
            arrays.append(control)
            replicate_closure.append(closure)
        controls.append(_pair_panel(arrays, np, cp, linear_sum_assignment))
        closures.append({"replicate": replicate, "seed": base_seed, "experts": replicate_closure})
        del arrays
        cp.get_default_memory_pool().free_all_blocks()

    statistics = {
        metric: _controlled_statistic(qwen, controls, metric)
        for metric in ("relaxed_reuse_exact", "legal_path_exact", "legal_path_fp16")
    }
    relaxed = statistics["relaxed_reuse_exact"]
    legal_fp16 = statistics["legal_path_fp16"]
    if not relaxed["upper_confidence_survives_target"]:
        decision = "HARD_KILL_CONTROL_CORRECTED_RELAXED_UPPER_BOUND"
    elif legal_fp16["upper_confidence_survives_target"] and float(qwen["legal_path_fp16"]["s_bpw"]) >= REQUIRED_GROSS_S:
        decision = "SURVIVE_SOURCE_ORACLE_FP16_PATH_RESIDUAL_CODEC_REQUIRED"
    else:
        decision = "AMBIGUOUS_RELAXATION_GAP_STRONGER_PATH_SOLVER_REQUIRED"
    return {
        "qwen": qwen,
        "controls": controls,
        "control_moment_closure": closures,
        "statistics": statistics,
        "decision": decision,
        "early_stop": False,
        "claim_boundary": (
            "The relaxed statistic is a containing optimistic upper bound for the frozen path "
            "family. The cycle-cover path is only an achievable heuristic. Exact FP64 fitting is "
            "used only for selection; the legal replay rounds and charges every coefficient as FP16."
        ),
    }


def _write_create_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(
        os.fspath(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProtocolError("short result write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_new_at(
    parent_descriptor: int,
    parent_path: Path,
    filename: str,
    value: dict[str, Any],
) -> None:
    """Exclusive descriptor-relative write after rechecking held-parent identity."""
    if filename in ("", ".", "..") or Path(filename).name != filename:
        raise ProtocolError("unsafe output filename")
    held = os.fstat(parent_descriptor)
    current = parent_path.lstat()
    if not stat.S_ISDIR(held.st_mode) or stat.S_ISLNK(current.st_mode):
        raise ProtocolError("output parent is no longer a real directory")
    if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
        raise ProtocolError("output parent identity changed before commit")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProtocolError("short descriptor-relative result write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if hasattr(os, "fsync"):
        os.fsync(parent_descriptor)


def main() -> None:
    started = time.time()
    args = _parse_args()
    package = Path(__file__).absolute().parent.resolve(strict=True)
    artifact_rows, artifact_raw = _artifact_rows(package)
    bindings, bindings_raw = _load_bindings(package)
    (
        authorization,
        authorization_sha,
        root,
        source_root_descriptor,
        output,
        output_parent_descriptor,
    ) = _load_authorization(
        args, package, artifact_rows, artifact_raw
    )
    try:
        # Heavy imports occur only after every artifact, opened external audit,
        # path, runtime, and create-new-output precondition available to the
        # standard library has passed.
        import numpy as np  # type: ignore
        import cupy as cp  # type: ignore
        import scipy  # type: ignore
        from scipy.optimize import linear_sum_assignment  # type: ignore

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ProtocolError("CUDA_VISIBLE_DEVICES must be exactly 0")
        runtime_binding = authorization["runtime_binding"]
        observed_versions = {
            "numpy_version": np.__version__,
            "cupy_version": cp.__version__,
            "scipy_version": scipy.__version__,
        }
        for key, observed in observed_versions.items():
            if runtime_binding.get(key) != observed:
                raise ProtocolError(f"runtime version mismatch: {key}")
        cp.cuda.Device(0).use()
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        cuda_runtime = int(cp.cuda.runtime.runtimeGetVersion())
        if runtime_binding.get("device_name") != device_name or int(runtime_binding.get("cuda_runtime")) != cuda_runtime:
            raise ProtocolError("GPU runtime tuple differs from independent calibration")

        experts, source_receipts = _load_sources(source_root_descriptor, bindings, np, cp)
        direct = _direct_stage(experts, np, cp, linear_sum_assignment)
        result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_AUXILIARY_DIRECT_CROSS_ROLE_OPPORTUNITY_GATE",
        "architecture": "FOSP-ARX-v2-DIRECT",
        "metric": "original-coordinate pooled BF16 squared error",
        "objective": {
            "target_F_maximum": 0.8,
            "required_net_s_bpw": REQUIRED_S,
            "required_gross_s_after_all_side_bits_bpw": REQUIRED_GROSS_S,
            "rates_bpw": list(RATES),
            "maximum_cold_read_amplification": 2.0,
        },
        "zero_bit_permutation_variant": {
            "decision": "INELIGIBLE_IMMEDIATE_KILL",
            "reason": "original arbitrary labels are not decoder-visible from an orbit representative",
        },
        "eligible_physical_bridge": {
            "factoradic_information_bits": ceil_log2_factorial(ROWS),
            "factoradic_physical_bits": FACTORADIC_BITS,
            "factoradic_physical_bytes": FACTORADIC_BYTES,
            "fp16_coefficients": PATH_EDGES * FP16_COEFFICIENTS_PER_EDGE,
            "fp16_coefficient_bits": FP16_COEFFICIENT_BITS,
            "total_side_bits": TOTAL_SIDE_BITS,
            "side_bpw": SIDE_BPW,
            "scatter_to_original_coordinates_before_score": True,
        },
        "rate_read_ledgers": [frame_ledger(rate) for rate in RATES],
        "direct_cross_role_stage": direct,
        "source_receipts": source_receipts,
        "bindings_sha256": sha256_bytes(bindings_raw),
        "authorization": {
            "file_sha256": authorization_sha,
            "run_id": authorization["run_id"],
            "source_audit_receipt_sha256": authorization["audit_binding"]["source_audit_receipt_sha256"],
            "runtime_audit_receipt_sha256": authorization["audit_binding"]["runtime_audit_receipt_sha256"],
        },
        "backend": {
            "python_executable_resolved": os.fspath(Path(sys.executable).resolve(strict=True)),
            "python_version": platform.python_version(),
            **observed_versions,
            "device_name": device_name,
            "cuda_runtime": cuda_runtime,
        },
        "execution": {
            "artifact_manifest_sha256": sha256_bytes(artifact_raw),
            "runner_sha256": artifact_rows[Path(__file__).name],
            "elapsed_seconds": time.time() - started,
            "pinned_panel_opened": False,
            "package_mutated": False,
            "output_create_new": True,
        },
        "claim_boundary": (
            "This is a discovery-only auxiliary source oracle. It emits no residual bitstream, "
            "does not establish finite-rate MSE, and cannot authorize a pinned-panel run."
        ),
        }
        result["canonical_unsigned_sha256"] = canonical_sha256(result)
        _write_create_new_at(output_parent_descriptor, output.parent, output.name, result)
        print(json.dumps({"output": os.fspath(output), "decision": direct["decision"]}, sort_keys=True))
    finally:
        os.close(source_root_descriptor)
        os.close(output_parent_descriptor)


if __name__ == "__main__":
    main()
