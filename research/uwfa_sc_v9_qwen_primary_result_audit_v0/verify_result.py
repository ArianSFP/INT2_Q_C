#!/usr/bin/env python3
"""Independent, fail-closed result audit for UWFA-SC v9 primary Qwen runs.

The package contains no result pins or model payloads.  A run is authorized by
an externally hashed JSON pin bundle.  The verifier authenticates the complete
source closure and the exact seven-member publication, independently rebuilds
the owner-component folds and decision predicate, and causally replays the
literal candidate container in a fresh CPU process.  It never initializes
CUDA and it never grants positive claim authority.

Directory races are handled deliberately: broad ancestors are retained by
device/inode/type and name binding only, while the actual source-package and
publication directories retain full metadata.  Thus an unrelated sibling
created under /workspace cannot invalidate an otherwise immutable audit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import statistics
import struct
import sys
import types
import zlib
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


AUTHORIZATION = "AUDIT_EXACT_UWFA_SC_V9_PRIMARY_NONPROMOTING_RESULT_V0"
AUDIT_SCHEMA = "uwfa-sc-v9-primary-independent-result-audit-v0"
PINS_SCHEMA = "uwfa-sc-v9-primary-external-result-pins-v0"
RESULT_SCHEMA = "uwfa-sc-v9-qwen-primary-gate-v0"
COMPLETION_SCHEMA = "uwfa-sc-v9-qwen-primary-completion-v0"
MAX_JSON_BYTES = 256 * (1 << 20)
MAX_SOURCE_BYTES = 2 * (1 << 20)
MAX_CONTAINER_BYTES = 512 * (1 << 20)
MAX_JSON_NODES = 2_000_000
MAX_JSON_DEPTH = 96
MAX_JSON_STRING = 2 * (1 << 20)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")

PUBLICATION_MEMBERS = frozenset({
    "COMPLETE.json",
    "RESULT.json",
    "BOUND_BASELINE_SCORE.json",
    "SOURCE_PREFLIGHT.json",
    "DECODER_BUNDLE.json",
    "UWFCV8.bin",
    "IDENTITY_FRAMING.bin",
})
DATA_MEMBERS = PUBLICATION_MEMBERS - {"COMPLETE.json"}

KNOWN_V9_MANIFEST_SHA256 = "d1e3eaff6762df2e273f6e3f4216ff9110abe74a7534a0098544a4ceef632c5e"
KNOWN_V9_SOURCE_ROOT_SHA256 = "4f99644a8d36eb15d6ff966db25f01e3e10f6d0f481af5fe0fd507c647eadca5"
KNOWN_V9_RUNNER_SHA256 = "d1ff04ce3c2cc36208e464eaed943d6c94eb91a47e9d3c460b2d562b7162cc4d"
KNOWN_SUPPORT_SHA256 = "399cb25260d34ec299cc91a17f129da9be5ba5b799c961e43f0c1b0637ee0174"
KNOWN_V8_MANIFEST_SHA256 = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"
KNOWN_V8_SOURCE_ROOT_SHA256 = "be06cf4d6c474a01517c4062f448b0c41c7f59d31724d6d5af380b8c064de4fa"
KNOWN_STRATA_COMMON_SHA256 = "3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1"
KNOWN_FROZEN_AUDITOR_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
KNOWN_ARTIFACT_SHA256 = "4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b"
KNOWN_ARTIFACT_BYTES = 8_847_360
SOURCE_WEIGHTS = 28_311_552
PINNED_PRIMARY_UPDATES = 38_621_316_130
PINNED_DEFERRED_MAXIMUM_UPDATES = 286_625_070_746
PINNED_DEFERRED_COORDINATE_UPDATES = 93_518_490_096
PINNED_PANEL_SYMBOLS = 126_627_266
PINNED_PANEL_STREAMS = 15
PRIMARY_KERNEL_BUDGET_SECONDS = 21_600.0
CONSERVATIVE_THROUGHPUT_MIN = 1_800_000.0
CONSERVATIVE_THROUGHPUT_MAX = 4_500_000.0
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090"
INDEPENDENT_STANDALONE_REQUIRED_SAVING_BPW = (
    -0.5 * math.log2(0.8) - 0.008074080480766676
)
PINNED_FOLD_UPDATES = (
    (0, (0, 1), 12_865_688_966),
    (1, (2, 3), 12_794_875_916),
    (2, (4, 5), 12_707_496_716),
)

# Independent literal-container grammar.  These values are restated here and
# parsed without calling any sealed-v8 container helper.
IC_MAGIC = b"UWFCV8\x00\x00"
IC_DIRECTORY_MAGIC = b"UWFDIR4\x00"
IC_REGION_MAGIC = b"UWFREG4\x00"
IC_FRAME_MAGIC = b"UWFFRM4\x00"
IC_VERSION = 4
IC_HEADER_BYTES = 4096
IC_DIRECTORY_RECORD_BYTES = 256
IC_REGION_HEADER_BYTES = 256
IC_FRAME_HEADER_BYTES = 256
IC_CONTRIBUTION_RECORD_BYTES = 24
IC_OWNER_SET_BYTES = 32
IC_PAGE_BYTES = 4096
IC_HEADER_SEAL_BEGIN = 416
IC_HEADER_SEAL_END = 448
IC_BINDINGS = (
    "baseline_plan_sha256", "baseline_score_sha256",
    "universal_decoder_sha256", "producer_manifest_sha256",
    "audit_bootstrap_sha256", "source_full_geometry_sha256",
    "source_structural_geometry_sha256", "extraction_program_sha256",
    "universal_adapter_sha256", "pipeline_sha256",
    "source_snapshot_root_sha256", "source_preflight_receipt_sha256",
)
IC_BINDINGS_BEGIN = 448
IC_CRC_OFFSET = IC_BINDINGS_BEGIN + 32 * len(IC_BINDINGS)
IC_TOPOLOGIES = (
    "suffix", "xor_sketch", "modular_ones", "rolling_affine",
    "signed_saturating",
)
IC_STATE_SIZES = (2, 4, 8, 16, 32, 64)
IC_RESET_LENGTHS = (32, 128, 512, 2048, 4096)

V9_REQUIRED_MEMBERS = (
    "README.md", "design_lock.json", "primary_gate.py", "test_source_only.py",
)
V8_REQUIRED_MEMBERS = (
    "INDEPENDENT_BOOTSTRAP_ABI.md", "README.md", "container_codec.py",
    "cupy_backend.py", "design_lock.json", "dispatcher_contract.py",
    "fixture_long_memory.py", "fixture_portability.py", "protocol.py",
    "result_envelope.py", "run_source_free_gpu_dev.py", "stage0_census.py",
    "strata_sc_adapter.py", "test_source_only.py", "universal_adapter.py",
    "uwfa_common.py", "verify_source.py",
)
AUDIT_REQUIRED_MEMBERS = (
    "README.md", "UNRESOLVED_EXTERNAL_PINS.json", "design_lock.json",
    "test_source_only.py", "verify_result.py",
)

KNOWN_V8_MEMBERS = {
    "INDEPENDENT_BOOTSTRAP_ABI.md": (11025, "b46b2703121d2e50460025bc0c5ff53ca28fffb94a1a2b23e58a52ce41bd2160"),
    "README.md": (16213, "253b89f19c041118fb4148d8cbf76ebc71301b701e20f8cfefa421d77df68d0c"),
    "container_codec.py": (93379, "645debb547a76818a880bfc346a2dd6230af97b07dc832afb3548a83d6920fed"),
    "cupy_backend.py": (40964, "7904a5e122686487d89fb684b70052507089bfe3bbfe4f1f02520df6ce3fb1ba"),
    "design_lock.json": (11554, "da0514b2e1fa0f033b113912bbe05e7ae640c3a606fa5386ee202d45dcc71805"),
    "dispatcher_contract.py": (9205, "747db5747b75074c1191e17055d615df3cddc54da00e29ba03edfd99ddb2a243"),
    "fixture_long_memory.py": (4307, "d72e7c109920f7d2c6a64bcbf9de0c6463ae80b40cbdb3e772af44c30b3a8c38"),
    "fixture_portability.py": (16350, "b8e9c8d0741f5c7de44ad9ae2bedf8ea6b0fba3ec6fa58df80d8d08fb5a8a1db"),
    "protocol.py": (21051, "9e18675a1e646eb10c0900aa3767bff96666943309dbd8db3953c745888d2cc1"),
    "result_envelope.py": (19002, "ad568758b318a9a6f298da2dc17edcd7f7639e2f772511ae680798f301bc4601"),
    "run_source_free_gpu_dev.py": (8263, "888c5420353951d164a76015e6563154df119f1481da29621154a01347791838"),
    "stage0_census.py": (123776, "7b7c2e0fcb6593805e6b2c8234ae59cb42d90fbb7dcf945a35aa5dfe331ae618"),
    "strata_sc_adapter.py": (36184, "08fc8808ac168f6930ee9482e160f25f2bd087829fca4630553aea3510d722c6"),
    "test_source_only.py": (135687, "5dc3730b629dc3c05a1353d036c6a9049013b6c163540c31f2cb8275d5a68383"),
    "universal_adapter.py": (11577, "a5ab2e1919af98c2aa9b3032faa0ba5552efe05cca250bd6844fd48c76aabbc8"),
    "uwfa_common.py": (58875, "db53567ab6d71d5150cc92ef4a78fa9ce5cca01f5474fa2ca32edc8711cc4325"),
    "verify_source.py": (15907, "c9ccbcd0b68681400dab97636bad7e4d445a83f2446d032b53863a8ab77b7714"),
}

RESULT_FIELDS = frozenset({
    "schema", "status", "positive_claim_authority",
    "positive_claim_even_if_all_primary_gates_pass", "controls_run",
    "controls_may_be_opened_or_inferred_from_this_result", "shuffles_run",
    "coordinate_disjoint_diagnostic_run", "deferred_stages", "claim_boundary",
    "artifact_identity", "baseline_score", "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "recomputed_panel_reconstruction_f64_sha256", "source_free_review",
    "runtime_admission", "original_v8_full_survivor_projection",
    "scientific_primary_nested_holdout", "source_final", "physical",
    "exploratory_panel_cache", "decoder_bundle", "decoder_bundle_sha256",
    "pipeline_record", "pipeline_sha256", "telemetry",
    "total_observed_launch_wall_seconds",
    "runtime_projection_was_not_total_wall_time",
    "evaluation_workload_pins_are_decoder_identity_inputs", "source_hashes",
})


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            f"{label}: lowercase SHA-256")
    return value


def exact_int(value: Any, label: str, minimum: int = 0,
              maximum: int = (1 << 63) - 1) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{label}: exact integer bound")
    return value


def safe_name(value: Any, label: str) -> str:
    require(isinstance(value, str) and SAFE_NAME.fullmatch(value) is not None,
            f"{label}: safe filename")
    require(value not in {".", ".."}, f"{label}: reserved filename")
    return value


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AuditError(f"noncanonical JSON value: {exc}") from exc


def pretty_json(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True,
                           allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AuditError(f"noncanonical pretty JSON value: {exc}") from exc


def strict_json(data: bytes, label: str,
                maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    require(isinstance(data, bytes) and len(data) <= maximum_bytes,
            f"{label}: JSON byte bound")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key {key!r}")
            result[key] = value
        return result

    def finite(text: str) -> float:
        value = float(text)
        require(math.isfinite(value), f"{label}: nonfinite float")
        return value

    def reject(text: str) -> None:
        raise AuditError(f"{label}: nonfinite constant {text}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_float=finite,
                           parse_constant=reject)
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"{label}: invalid JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label}: root object")
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        require(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH,
                f"{label}: JSON complexity")
        if isinstance(item, dict):
            for key, child in item.items():
                require(isinstance(key, str) and len(key) <= MAX_JSON_STRING,
                        f"{label}: key bound")
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            require(len(item) <= MAX_JSON_STRING, f"{label}: string bound")
        elif isinstance(item, float):
            require(math.isfinite(item), f"{label}: finite tree")
        else:
            require(item is None or isinstance(item, (bool, int)),
                    f"{label}: scalar type")
    return value


def exact_fields(record: Any, fields: Sequence[str] | set[str] | frozenset[str],
                 label: str) -> dict[str, Any]:
    require(isinstance(record, dict) and set(record) == set(fields),
            f"{label}: exact fields")
    return record


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def require_float_equal(observed: Any, expected: float, label: str) -> None:
    require(type(observed) is float and math.isfinite(observed) and
            _float_bits(observed) == _float_bits(expected),
            f"{label}: binary64 mismatch")


def require_deep_equal(observed: Any, expected: Any, label: str) -> None:
    stack: list[tuple[Any, Any, str]] = [(observed, expected, label)]
    while stack:
        left, right, path = stack.pop()
        require(type(left) is type(right), f"{path}: type mismatch")
        if isinstance(right, dict):
            require(set(left) == set(right), f"{path}: field mismatch")
            stack.extend((left[key], right[key], f"{path}.{key}") for key in right)
        elif isinstance(right, list):
            require(len(left) == len(right), f"{path}: length mismatch")
            stack.extend((a, b, f"{path}[{i}]")
                         for i, (a, b) in enumerate(zip(left, right)))
        elif isinstance(right, float):
            require_float_equal(left, right, path)
        else:
            require(left == right, f"{path}: value mismatch")


def weak_directory_identity(info: os.stat_result) -> tuple[int, int, int]:
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


def strong_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def pread_exact(fd: int, size: int, cap: int, label: str) -> bytes:
    require(0 <= size <= cap, f"{label}: bounded bytes")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1 << 20, size - offset), offset)
        require(bool(chunk), f"{label}: short descriptor read")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


class RetainedDirectory:
    """No-follow path walk; broad ancestors ignore mtime/ctime churn."""

    def __init__(self, path: str, label: str, *, strict_leaf: bool) -> None:
        require(os.name == "posix", "audit requires POSIX openat/pread")
        pure = PurePosixPath(path)
        require(pure.is_absolute() and str(pure) == path and path != "/" and
                "//" not in path, f"{label}: canonical absolute directory")
        parts = path.split("/")[1:]
        require(parts and all(part and part not in {".", ".."} for part in parts),
                f"{label}: canonical components")
        self.label = label
        self.strict_leaf = strict_leaf
        self.fds: list[int] = []
        self.names: list[str] = []
        self.weak: list[tuple[int, int, int]] = []
        self.leaf_strong: tuple[int, int, int, int, int, int, int] | None = None
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY |
                          getattr(os, "O_NOFOLLOW", 0))
        self.fds.append(current)
        self.weak.append(weak_directory_identity(os.fstat(current)))
        try:
            for part in parts:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                                getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
                info = os.fstat(child)
                require(stat.S_ISDIR(info.st_mode), f"{label}: directory component")
                self.names.append(part)
                self.fds.append(child)
                self.weak.append(weak_directory_identity(info))
                current = child
            if strict_leaf:
                self.leaf_strong = strong_identity(os.fstat(self.fds[-1]))
        except Exception:
            self.close(verify=False)
            raise

    @property
    def fd(self) -> int:
        return self.fds[-1]

    def verify_final(self) -> None:
        for index, (fd, expected) in enumerate(zip(self.fds, self.weak)):
            require(weak_directory_identity(os.fstat(fd)) == expected,
                    f"{self.label}: held directory identity changed [{index}]")
            if index:
                named = os.stat(self.names[index - 1], dir_fd=self.fds[index - 1],
                                follow_symlinks=False)
                require(weak_directory_identity(named) == expected,
                        f"{self.label}: directory name rebound [{index}]")
        if self.strict_leaf:
            require(strong_identity(os.fstat(self.fds[-1])) == self.leaf_strong,
                    f"{self.label}: actual leaf directory metadata changed")

    def close(self, *, verify: bool = True) -> None:
        pending: BaseException | None = None
        if verify and self.fds:
            try:
                self.verify_final()
            except BaseException as exc:
                pending = exc
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds = []
        if pending is not None:
            raise pending

    def __enter__(self) -> "RetainedDirectory":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(verify=True)


class HeldRegularAt:
    def __init__(self, parent_fd: int, name: str, cap: int, label: str,
                 expected_sha256: str | None = None,
                 expected_bytes: int | None = None) -> None:
        self.parent_fd = parent_fd
        self.name = safe_name(name, label)
        self.cap = cap
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.name, flags, dir_fd=parent_fd)
        try:
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode), f"{label}: regular file")
            self.before = strong_identity(info)
            if expected_bytes is not None:
                require(info.st_size == expected_bytes, f"{label}: exact bytes")
            self.data = pread_exact(self.fd, info.st_size, cap, label)
            self.sha256 = sha256(self.data)
            if expected_sha256 is not None:
                require(self.sha256 == digest(expected_sha256, f"{label} expected"),
                        f"{label}: digest mismatch")
            self.verify_final()
        except Exception:
            os.close(self.fd)
            self.fd = -1
            raise

    def verify_final(self) -> None:
        require(self.fd >= 0, f"{self.label}: closed descriptor")
        require(strong_identity(os.fstat(self.fd)) == self.before,
                f"{self.label}: held descriptor changed")
        named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        require(strong_identity(named) == self.before,
                f"{self.label}: name/inode rebound")
        require(pread_exact(self.fd, self.before[3], self.cap, self.label) == self.data,
                f"{self.label}: bytes changed")

    def close(self, *, verify: bool = True) -> None:
        if self.fd < 0:
            return
        pending: BaseException | None = None
        if verify:
            try:
                self.verify_final()
            except BaseException as exc:
                pending = exc
        os.close(self.fd)
        self.fd = -1
        if pending is not None:
            raise pending

    def __enter__(self) -> "HeldRegularAt":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(verify=True)


class HeldAbsoluteRegular:
    def __init__(self, path: str, cap: int, label: str,
                 expected_sha256: str, expected_bytes: int | None = None) -> None:
        pure = PurePosixPath(path)
        require(pure.is_absolute() and pure.name not in {"", ".", ".."} and
                str(pure) == path and "//" not in path,
                f"{label}: canonical absolute file")
        self.parent = RetainedDirectory(str(pure.parent), f"{label} parent",
                                        strict_leaf=False)
        try:
            self.held = HeldRegularAt(self.parent.fd, pure.name, cap, label,
                                      expected_sha256, expected_bytes)
        except Exception:
            self.parent.close(verify=False)
            raise

    @property
    def data(self) -> bytes:
        return self.held.data

    @property
    def sha256(self) -> str:
        return self.held.sha256

    @property
    def fd(self) -> int:
        return self.held.fd

    @property
    def before(self) -> tuple[int, int, int, int, int, int, int]:
        return self.held.before

    def verify_final(self) -> None:
        self.held.verify_final()
        self.parent.verify_final()

    def close(self, *, verify: bool = True) -> None:
        pending: BaseException | None = None
        try:
            self.held.close(verify=verify)
        except BaseException as exc:
            pending = exc
        try:
            self.parent.close(verify=verify and pending is None)
        except BaseException as exc:
            if pending is None:
                pending = exc
        if pending is not None:
            raise pending

    def __enter__(self) -> "HeldAbsoluteRegular":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(verify=True)


@dataclass(frozen=True)
class Pins:
    output_parent: str
    final_name: str
    v9_package: str
    support_path: str
    v8_package: str
    strata_common_path: str
    frozen_auditor_path: str
    artifact_path: str
    publication_members: Mapping[str, Mapping[str, Any]]
    original_source_identity: Mapping[str, Any]
    v8_members: Mapping[str, Mapping[str, Any]]
    raw: Mapping[str, Any]


def validate_absolute_path(path: Any, label: str, *, directory: bool) -> str:
    require(isinstance(path, str) and path.startswith("/") and "//" not in path,
            f"{label}: absolute lexical path")
    pure = PurePosixPath(path)
    require(pure.is_absolute() and str(pure) == path, f"{label}: canonical path")
    if not directory:
        require(pure.name not in {"", ".", ".."}, f"{label}: file leaf")
    return path


def parse_pins(record: dict[str, Any]) -> Pins:
    exact_fields(record, {"schema", "status", "paths", "source_hashes",
                          "publication_members", "original_source_identity"},
                 "external pins")
    require(record["schema"] == PINS_SCHEMA, "external pins schema")
    require(record["status"] == "EXTERNALLY_RECORDED_AFTER_PUBLICATION",
            "external pins unresolved/status")
    paths = exact_fields(record["paths"], {
        "output_parent", "final_name", "v9_package", "support_path",
        "v8_package", "strata_common_path", "frozen_auditor_path",
        "artifact_path"}, "external paths")
    safe_name(paths["final_name"], "final name")
    for key in ("output_parent", "v9_package", "v8_package"):
        validate_absolute_path(paths[key], key, directory=True)
    for key in ("support_path", "strata_common_path", "frozen_auditor_path",
                "artifact_path"):
        validate_absolute_path(paths[key], key, directory=False)
    hashes = exact_fields(record["source_hashes"], {
        "v9_manifest_sha256", "v9_source_root_sha256", "primary_gate_sha256",
        "support_sha256", "v8_manifest_sha256", "v8_source_root_sha256",
        "v8_members", "strata_common_sha256", "frozen_auditor_sha256",
        "artifact_sha256", "artifact_bytes"}, "source hashes")
    known = {
        "v9_manifest_sha256": KNOWN_V9_MANIFEST_SHA256,
        "v9_source_root_sha256": KNOWN_V9_SOURCE_ROOT_SHA256,
        "primary_gate_sha256": KNOWN_V9_RUNNER_SHA256,
        "support_sha256": KNOWN_SUPPORT_SHA256,
        "v8_manifest_sha256": KNOWN_V8_MANIFEST_SHA256,
        "v8_source_root_sha256": KNOWN_V8_SOURCE_ROOT_SHA256,
        "strata_common_sha256": KNOWN_STRATA_COMMON_SHA256,
        "frozen_auditor_sha256": KNOWN_FROZEN_AUDITOR_SHA256,
        "artifact_sha256": KNOWN_ARTIFACT_SHA256,
    }
    for key, expected in known.items():
        require(digest(hashes[key], key) == expected, f"{key}: frozen pin mismatch")
    require(exact_int(hashes["artifact_bytes"], "artifact bytes", 1, 1 << 34)
            == KNOWN_ARTIFACT_BYTES, "artifact byte pin")
    rows = hashes["v8_members"]
    require(isinstance(rows, dict) and set(rows) == set(V8_REQUIRED_MEMBERS),
            "external v8 complete member pins")
    for name in V8_REQUIRED_MEMBERS:
        row = exact_fields(rows[name], {"bytes", "sha256"}, f"v8 pin {name}")
        expected_bytes, expected_hash = KNOWN_V8_MEMBERS[name]
        require(exact_int(row["bytes"], f"v8 pin bytes {name}", 1,
                          MAX_SOURCE_BYTES) == expected_bytes,
                f"v8 pin bytes {name}")
        require(digest(row["sha256"], f"v8 pin hash {name}") == expected_hash,
                f"v8 pin hash {name}")
    publication = record["publication_members"]
    require(isinstance(publication, dict) and set(publication) == PUBLICATION_MEMBERS,
            "external publication pins exact seven members")
    for name in PUBLICATION_MEMBERS:
        row = exact_fields(publication[name], {"bytes", "sha256"},
                           f"publication pin {name}")
        maximum = MAX_CONTAINER_BYTES if name.endswith(".bin") else MAX_JSON_BYTES
        exact_int(row["bytes"], f"publication bytes {name}", 1, maximum)
        digest(row["sha256"], f"publication hash {name}")
    identity = exact_fields(record["original_source_identity"], {
        "source_full_geometry_sha256", "source_structural_geometry_sha256",
        "reconstruction_f64_sha256", "score_receipt_sha256",
        "relative_mse", "sse_fp64", "source_energy_fp64"},
        "original source identity")
    for key in ("source_full_geometry_sha256", "source_structural_geometry_sha256",
                "reconstruction_f64_sha256", "score_receipt_sha256"):
        digest(identity[key], f"original source {key}")
    for key in ("relative_mse", "sse_fp64", "source_energy_fp64"):
        require(type(identity[key]) is float and math.isfinite(identity[key]) and
                identity[key] > 0.0, f"original source {key}: positive binary64")
    expected_mse = identity["sse_fp64"] / identity["source_energy_fp64"]
    require(math.isfinite(expected_mse) and expected_mse > 0.0 and
            abs(expected_mse - identity["relative_mse"]) <=
            4.0 * math.ulp(expected_mse), "original source MSE closure")
    return Pins(
        output_parent=paths["output_parent"], final_name=paths["final_name"],
        v9_package=paths["v9_package"], support_path=paths["support_path"],
        v8_package=paths["v8_package"],
        strata_common_path=paths["strata_common_path"],
        frozen_auditor_path=paths["frozen_auditor_path"],
        artifact_path=paths["artifact_path"], publication_members=publication,
        original_source_identity=identity, v8_members=rows, raw=record,
    )


def literal_assignments(source: bytes, names: set[str], label: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source, filename=f"<authenticated:{label}>")
    except SyntaxError as exc:
        raise AuditError(f"{label}: syntax") from exc
    values: dict[str, Any] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name)):
            name = node.targets[0].id
            if name in names:
                require(name not in values, f"{label}: duplicate literal {name}")
                try:
                    values[name] = ast.literal_eval(node.value)
                except Exception as exc:
                    raise AuditError(f"{label}: nonliteral {name}") from exc
    require(set(values) == names, f"{label}: required literal assignments")
    return values


def load_module(name: str, source: bytes, expected_sha256: str) -> types.ModuleType:
    require(sha256(source) == expected_sha256, f"{name}: source digest")
    require(name not in sys.modules, f"{name}: namespace collision")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated-v9-result-audit:{name}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(source, module.__file__, "exec", dont_inherit=True,
                       optimize=0)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def normalize_strata_group_ordinals(strata_common: Any) -> dict[str, Any]:
    original = getattr(strata_common, "expected_block_group_ordinals", None)
    require(callable(original), "STRATA group-ordinal entrypoint")

    def normalized(labels: Any) -> list[list[int]]:
        rows = original(labels)
        require(isinstance(rows, list), "STRATA group rows")
        converted: list[list[int]] = []
        for row in rows:
            values = [int(value) for value in row]
            require(all(type(value) is int for value in values),
                    "STRATA exact Python-int ABI")
            converted.append(values)
        return converted

    strata_common.expected_block_group_ordinals = normalized
    receipt = {
        "schema": "uwfa-sc-v8-qwen-early-gate-strata-group-ordinal-abi-v0",
        "status": "EXPLORATORY_VALUE_PRESERVING_NUMPY_INTEGER_TO_PYTHON_INT",
        "operation": "for every group ordinal emitted by the pinned STRATA helper, apply built-in int without reordering",
        "positive_claim_authority": False,
    }
    receipt["receipt_sha256"] = sha256(canonical_json(receipt))
    return receipt


def authenticate_package(
    stack: ExitStack,
    path: str,
    *,
    label: str,
    expected_manifest_sha256: str,
    expected_root_sha256: str,
    manifest_schema: str,
    required_members: Sequence[str],
    external_members: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    package = stack.enter_context(RetainedDirectory(path, label, strict_leaf=True))
    manifest_held = stack.enter_context(HeldRegularAt(
        package.fd, "SOURCE_MANIFEST.json", 1 << 20, f"{label} manifest",
        expected_manifest_sha256,
    ))
    manifest = strict_json(manifest_held.data, f"{label} manifest", 1 << 20)
    required_manifest_fields = (
        {"schema", "status", "members", "access_attestation", "claim_boundary"}
        if manifest_schema == "uwfa-sc-v9-primary-source-manifest-v0"
        else {"schema", "status", "members", "access_attestation",
              "post_freeze_requirements"}
    )
    exact_fields(manifest, required_manifest_fields, f"{label} manifest")
    require(manifest["schema"] == manifest_schema, f"{label} manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY",
            f"{label} manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(required_members),
            f"{label} manifest rows")
    require([row.get("name") if isinstance(row, dict) else None for row in rows]
            == list(required_members), f"{label} canonical member order")
    require({entry.name for entry in os.scandir(package.fd)} ==
            set(required_members) | {"SOURCE_MANIFEST.json"},
            f"{label} exact member set")
    root = sha256(canonical_json(rows))
    require(root == expected_root_sha256, f"{label} external source root")
    sources: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    identities: dict[tuple[int, int], str] = {
        (manifest_held.before[0], manifest_held.before[1]): "SOURCE_MANIFEST.json"
    }
    for row in rows:
        exact_fields(row, {"name", "bytes", "sha256"}, f"{label} member row")
        name = safe_name(row["name"], f"{label} member name")
        size = exact_int(row["bytes"], f"{label} bytes {name}", 1,
                         MAX_SOURCE_BYTES)
        expected = digest(row["sha256"], f"{label} digest {name}")
        if external_members is not None:
            external = external_members[name]
            require(external["bytes"] == size and external["sha256"] == expected,
                    f"{label} external member pin {name}")
        held = stack.enter_context(HeldRegularAt(
            package.fd, name, MAX_SOURCE_BYTES, f"{label} member {name}",
            expected, size,
        ))
        inode = (held.before[0], held.before[1])
        require(inode not in identities,
                f"{label} inode alias {name}/{identities.get(inode)}")
        identities[inode] = name
        sources[name] = held.data
        hashes[name] = held.sha256
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_held.sha256,
        "source_root_sha256": root,
        "sources": sources,
        "hashes": hashes,
        "identities": identities,
    }


def authenticate_audit_source(
    stack: ExitStack, expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind the executing auditor to an externally pinned source manifest."""
    expected_manifest_sha256 = digest(
        expected_manifest_sha256, "audit source manifest external pin")
    script_path = os.path.abspath(__file__)
    package_path = os.path.dirname(script_path)
    validate_absolute_path(package_path, "audit source package", directory=True)
    package = stack.enter_context(RetainedDirectory(
        package_path, "audit source package", strict_leaf=True))
    held_manifest = stack.enter_context(HeldRegularAt(
        package.fd, "SOURCE_MANIFEST.json", 1 << 20,
        "audit source manifest", expected_manifest_sha256))
    manifest = strict_json(held_manifest.data, "audit source manifest", 1 << 20)
    exact_fields(manifest, {
        "schema", "status", "members", "access_attestation",
        "claim_boundary",
    }, "audit source manifest")
    require(manifest["schema"] ==
            "uwfa-sc-v9-primary-independent-result-audit-source-manifest-v0" and
            manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY",
            "audit source manifest schema/status")
    require(manifest["claim_boundary"] ==
            "nonpromoting integrity audit only; no controls, shuffles, compression claim, or universal SwiGLU-MoE authority",
            "audit source manifest claim boundary")
    require_deep_equal(manifest["access_attestation"], {
        "runpod_accessed": False,
        "qwen_or_model_payload_accessed": False,
        "live_v9_result_accessed": False,
        "gaussian_control_accessed": False,
        "cuda_or_cupy_initialized": False,
    }, "audit source build-access attestation")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(AUDIT_REQUIRED_MEMBERS),
            "audit source manifest rows")
    require([row.get("name") if isinstance(row, dict) else None for row in rows]
            == list(AUDIT_REQUIRED_MEMBERS),
            "audit source canonical member order")
    require({entry.name for entry in os.scandir(package.fd)} ==
            set(AUDIT_REQUIRED_MEMBERS) | {"SOURCE_MANIFEST.json"},
            "audit source exact member set")
    hashes: dict[str, str] = {}
    running_identity = os.lstat(script_path)
    for row in rows:
        exact_fields(row, {"name", "bytes", "sha256"},
                     "audit source member row")
        name = safe_name(row["name"], "audit source member name")
        size = exact_int(row["bytes"], f"audit source member {name} bytes",
                         1, MAX_SOURCE_BYTES)
        expected = digest(row["sha256"],
                          f"audit source member {name} digest")
        held = stack.enter_context(HeldRegularAt(
            package.fd, name, MAX_SOURCE_BYTES,
            f"audit source member {name}", expected, size))
        hashes[name] = held.sha256
        if name == "verify_result.py":
            require((held.before[0], held.before[1]) ==
                    (running_identity.st_dev, running_identity.st_ino),
                    "executing verifier/source-manifest inode binding")
    return {
        "manifest_sha256": held_manifest.sha256,
        "source_root_sha256": sha256(canonical_json(rows)),
        "member_hashes": hashes,
    }


def load_authenticated_modules(
    closure: Mapping[str, Any],
    strata_source: bytes,
    strata_hash: str,
    auditor_source: bytes,
    auditor_hash: str,
) -> dict[str, Any]:
    # Deliberately CPU only. NumPy is imported after all source and artifact
    # descriptors have been authenticated by the caller.
    import numpy as np

    tag = sha256(canonical_json({
        "root": closure["source_root_sha256"],
        "strata": strata_hash,
        "auditor": auditor_hash,
    }))[:16]
    sources = closure["sources"]
    hashes = closure["hashes"]
    common = load_module(f"uwfa_v9_result_{tag}_common", sources["uwfa_common.py"],
                         hashes["uwfa_common.py"])
    protocol = load_module(f"uwfa_v9_result_{tag}_protocol", sources["protocol.py"],
                           hashes["protocol.py"])
    semantic = load_module(f"uwfa_v9_result_{tag}_semantic",
                           sources["universal_adapter.py"],
                           hashes["universal_adapter.py"])
    codec = load_module(f"uwfa_v9_result_{tag}_codec",
                        sources["container_codec.py"],
                        hashes["container_codec.py"])
    stage = load_module(f"uwfa_v9_result_{tag}_stage",
                        sources["stage0_census.py"],
                        hashes["stage0_census.py"])
    adapter_source = load_module(f"uwfa_v9_result_{tag}_adapter",
                                 sources["strata_sc_adapter.py"],
                                 hashes["strata_sc_adapter.py"])
    strata = load_module(f"uwfa_v9_result_{tag}_strata", strata_source,
                         strata_hash)
    frozen = load_module(f"uwfa_v9_result_{tag}_frozen", auditor_source,
                         auditor_hash)
    bridge = normalize_strata_group_ordinals(strata)
    adapter = adapter_source.StrataSCAdapter(
        common=common, semantic_codec=semantic, np=np,
        frozen_auditor=frozen, strata_common=strata, device="numpy",
    )
    return {
        "np": np, "common": common, "protocol": protocol,
        "semantic": semantic, "codec": codec, "stage": stage,
        "adapter_source": adapter_source, "strata": strata,
        "frozen": frozen, "bridge": bridge, "adapter": adapter,
    }


def verify_internal_seal(record: Mapping[str, Any], field: str,
                         label: str) -> str:
    require(isinstance(record, dict), f"{label}: object")
    claimed = digest(record.get(field), f"{label}.{field}")
    clean = dict(record)
    clean.pop(field)
    require(sha256(canonical_json(clean)) == claimed, f"{label}: internal seal")
    return claimed


def verify_completion(
    complete: dict[str, Any],
    observed: Mapping[str, HeldRegularAt],
    expected_source_root: str,
) -> None:
    exact_fields(complete, {
        "schema", "status", "positive_claim_authority", "controls_run",
        "shuffles_run", "coordinate_diagnostic_run",
        "v9_source_snapshot_root_sha256", "members", "completion_sha256"},
        "COMPLETE")
    require(complete["schema"] == COMPLETION_SCHEMA, "COMPLETE schema")
    for name in ("positive_claim_authority", "controls_run", "shuffles_run",
                 "coordinate_diagnostic_run"):
        require(complete[name] is False, f"COMPLETE {name}")
    require(complete["v9_source_snapshot_root_sha256"] == expected_source_root,
            "COMPLETE v9 source root")
    verify_internal_seal(complete, "completion_sha256", "COMPLETE")
    rows = complete["members"]
    require(isinstance(rows, list), "COMPLETE member rows")
    expected_order = sorted(DATA_MEMBERS, key=lambda value: value.encode("utf-8"))
    require([row.get("name") if isinstance(row, dict) else None for row in rows]
            == expected_order, "COMPLETE canonical exact data-member order")
    for row in rows:
        exact_fields(row, {"name", "bytes", "sha256"}, "COMPLETE member row")
        held = observed[row["name"]]
        require(row["bytes"] == len(held.data),
                f"COMPLETE member bytes {row['name']}")
        require(row["sha256"] == held.sha256,
                f"COMPLETE member digest {row['name']}")


def open_publication(stack: ExitStack, pins: Pins) -> dict[str, Any]:
    # Output parent is intentionally weak-metadata: unrelated siblings may be
    # created while this long CPU replay runs. The named final directory itself
    # remains strongly held and rebound.
    parent = stack.enter_context(RetainedDirectory(
        pins.output_parent, "publication parent", strict_leaf=False,
    ))
    final_fd = os.open(
        pins.final_name,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent.fd,
    )
    stack.callback(os.close, final_fd)
    final_info = os.fstat(final_fd)
    require(stat.S_ISDIR(final_info.st_mode), "publication final directory")
    final_identity = strong_identity(final_info)
    actual = {entry.name for entry in os.scandir(final_fd)}
    require(actual == PUBLICATION_MEMBERS, "publication exact seven members")
    held: dict[str, HeldRegularAt] = {}
    output_inodes: dict[tuple[int, int], str] = {}
    caps = {
        "COMPLETE.json": 1 << 20,
        "RESULT.json": MAX_JSON_BYTES,
        "BOUND_BASELINE_SCORE.json": 1 << 20,
        "SOURCE_PREFLIGHT.json": 64 << 20,
        "DECODER_BUNDLE.json": 1 << 20,
        "UWFCV8.bin": MAX_CONTAINER_BYTES,
        "IDENTITY_FRAMING.bin": MAX_CONTAINER_BYTES,
    }
    for name in ["COMPLETE.json"] + sorted(DATA_MEMBERS,
                                                  key=lambda value: value.encode("utf-8")):
        pin = pins.publication_members[name]
        item = stack.enter_context(HeldRegularAt(
            final_fd, name, caps[name], f"publication member {name}",
            pin["sha256"], pin["bytes"],
        ))
        require(item.before[6] == 1, f"publication sole link {name}")
        inode = (item.before[0], item.before[1])
        require(inode not in output_inodes,
                f"publication inode alias {name}/{output_inodes.get(inode)}")
        output_inodes[inode] = name
        held[name] = item
    complete = strict_json(held["COMPLETE.json"].data, "COMPLETE", 1 << 20)
    require(held["COMPLETE.json"].data == pretty_json(complete),
            "COMPLETE canonical pretty encoding")
    verify_completion(complete, held, KNOWN_V9_SOURCE_ROOT_SHA256)
    # Static evidence cannot prove syscall history, but it can reject a
    # completion seal that is observably older than any sealed data member.
    complete_meta = held["COMPLETE.json"].before
    for name in DATA_MEMBERS:
        require(complete_meta[4] >= held[name].before[4] and
                complete_meta[5] >= held[name].before[5],
                f"COMPLETE observably predates {name}")
    return {
        "parent": parent, "final_fd": final_fd,
        "final_identity": final_identity, "held": held,
        "complete": complete, "output_inodes": output_inodes,
        "actual_members": actual,
    }


def final_publication_rebind(publication: Mapping[str, Any], pins: Pins) -> None:
    for held in publication["held"].values():
        held.verify_final()
    final_fd = publication["final_fd"]
    require(strong_identity(os.fstat(final_fd)) == publication["final_identity"],
            "publication directory descriptor changed")
    named = os.stat(pins.final_name, dir_fd=publication["parent"].fd,
                    follow_symlinks=False)
    require(strong_identity(named) == publication["final_identity"],
            "publication final name/inode rebound")
    require({entry.name for entry in os.scandir(final_fd)} ==
            publication["actual_members"], "publication member set changed")
    publication["parent"].verify_final()


def verify_nonpromotion_counters(value: Any, label: str = "RESULT") -> None:
    must_be_false = {
        "positive_claim_authority",
        "positive_claim_even_if_all_primary_gates_pass",
        "positive_promotion",
        "positive_claim_use_permitted",
        "controls_run",
        "controls_opened",
        "controls_may_be_opened_or_inferred_from_this_result",
        "shuffles_run",
        "coordinate_disjoint_diagnostic_run",
        "coordinate_diagnostic_run",
    }
    stack: list[tuple[Any, str]] = [(value, label)]
    while stack:
        item, path = stack.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                if key in must_be_false:
                    require(child is False,
                            f"{child_path}: nonpromotion counter must be false")
                stack.append((child, child_path))
        elif isinstance(item, list):
            stack.extend((child, f"{path}[{index}]")
                         for index, child in enumerate(item))


def recompute_primary_status(scientific: Mapping[str, Any],
                             source_final: Mapping[str, Any]) -> str:
    metrics = source_final["parsed_metrics"]
    standalone = source_final["standalone_decode"]
    integrity = bool(
        standalone["all_payloads_canonically_reencoded"]
        and source_final["identical_reconstruction_proved_by_full_f64_digest"]
        and source_final["all_adapted_values_deserialized_from_transmitted_model"]
    )
    if not integrity:
        return "FAIL_EVIDENCE_INTEGRITY_PRIMARY_CONTAINER"
    if not bool(metrics["passes_rate_interval"] and metrics["passes_F_target"]):
        return "HARD_KILL_PRIMARY_PHYSICAL_RATE_OR_F"
    if not bool(metrics["passes_cold_read_below_2x"]):
        return "FAIL_PRIMARY_STRICT_COLD_READ"
    if not bool(scientific["passes_heldout_gate"]):
        return "NO_PROMOTION_PRIMARY_NESTED_HELDOUT"
    return "PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED"


def verify_claim_boundary(result: dict[str, Any], complete: dict[str, Any]) -> str:
    exact_fields(result, RESULT_FIELDS, "RESULT")
    require(result["schema"] == RESULT_SCHEMA, "RESULT schema")
    for name in (
        "positive_claim_authority",
        "positive_claim_even_if_all_primary_gates_pass",
        "controls_run",
        "controls_may_be_opened_or_inferred_from_this_result",
        "shuffles_run",
        "coordinate_disjoint_diagnostic_run",
    ):
        require(result[name] is False, f"RESULT {name}")
    require(result["claim_boundary"] ==
            "runtime repair for exact sealed-v8 primary Qwen estimand only; never a compression claim or universal SwiGLU-MoE result",
            "RESULT claim boundary")
    require(result["runtime_projection_was_not_total_wall_time"] is True,
            "RESULT runtime projection disclosure")
    require(result["evaluation_workload_pins_are_decoder_identity_inputs"] is False,
            "RESULT evaluation pin disclosure")
    deferred = exact_fields(result["deferred_stages"], {
        "survivor_shuffles", "coordinate_disjoint_diagnostic",
        "matched_gaussian_controls", "independent_result_audit"},
        "RESULT deferred stages")
    require(deferred == {
        "survivor_shuffles": "NOT_RUN_REQUIRES_SEPARATE_REVIEW_AND_AUTHORIZATION",
        "coordinate_disjoint_diagnostic": "NOT_RUN_REQUIRES_SEPARATE_REVIEW_AND_AUTHORIZATION",
        "matched_gaussian_controls": "NOT_RUN_REQUIRES_SEPARATE_REVIEW_AND_AUTHORIZATION",
        "independent_result_audit": "REQUIRED_BEFORE_ANY_CLAIM",
    }, "RESULT exact deferred-stage boundary")
    expected = recompute_primary_status(
        result["scientific_primary_nested_holdout"], result["source_final"])
    require(result["status"] == expected, "RESULT independently recomputed status")
    require(complete["status"] == expected, "COMPLETE independently recomputed status")
    verify_nonpromotion_counters(result)
    return expected


def length_prefixed_digest(parts: Sequence[str | bytes | int], *,
                           domain: bytes) -> str:
    digest_state = hashlib.sha256(domain)
    for value in parts:
        if isinstance(value, str):
            payload, tag = value.encode("utf-8", "strict"), 1
        elif isinstance(value, bytes):
            payload, tag = value, 2
        elif type(value) is int:
            payload, tag = str(value).encode("ascii"), 3
        else:
            raise AuditError("unsupported split-digest value")
        digest_state.update(bytes((tag,)))
        digest_state.update(struct.pack("<Q", len(payload)))
        digest_state.update(payload)
    return digest_state.hexdigest()


def independent_component_plan(panel: Mapping[str, Any],
                               inner_modulus: int) -> list[dict[str, Any]]:
    experts = exact_int(panel["experts"], "panel experts", 1, 256)
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    require(len(identities) == experts and len(set(map(tuple, identities))) == experts,
            "semantic identity uniqueness")
    require(inner_modulus == 5, "independent frozen inner-validation modulus")
    owner_rows: list[tuple[int, ...]] = []
    for ordinal, row in enumerate(streams):
        owners = _ic_owner_ordinals(bytes(row["owner_set"]), experts)
        require(row["owner_set_hex"] == bytes(row["owner_set"]).hex() and
                tuple(row["owner_expert_ordinals"]) == owners and
                tuple(row["owner_identity_indices"]) == owners,
                f"stream[{ordinal}] raw owner-set/derived owner binding")
        owner_rows.append(owners)
    parent = list(range(experts))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for owners in owner_rows:
        for owner in owners[1:]:
            union(owners[0], owner)
    grouped: dict[int, list[int]] = {}
    for expert in range(experts):
        grouped.setdefault(find(expert), []).append(expert)
    components = [tuple(grouped[root]) for root in sorted(grouped)]
    plans: list[dict[str, Any]] = []
    for ordinal, members in enumerate(components):
        member_set = set(members)
        test = [index for index, owners in enumerate(owner_rows)
                if member_set.intersection(owners)]
        development = [index for index, owners in enumerate(owner_rows)
                       if not member_set.intersection(owners)]
        require(all(set(owner_rows[index]).issubset(member_set)
                    for index in test), "component shared-stream closure")
        require(test and len(development) >= 2, "component fold estimable")
        component_key = [value for member in members for value in identities[member]]
        ranked = sorted(
            development,
            key=lambda index: (
                length_prefixed_digest(
                    component_key + [int(streams[index]["stream_ordinal"]),
                                     bytes(streams[index]["owner_set"])],
                    domain=b"UWFA-SC-V8-DISJOINT-COMPONENT-SPLIT-2026-09-02\x00",
                ),
                int(streams[index]["stream_ordinal"]),
            ),
        )
        validation_count = min(max(1, len(ranked) // inner_modulus),
                               len(ranked) - 1)
        validation = sorted(ranked[:validation_count])
        train = sorted(ranked[validation_count:])
        plans.append({
            "component_ordinal": ordinal,
            "identity_indices": list(members),
            "identities": [tuple(identities[index]) for index in members],
            "test_indices": test,
            "development_indices": development,
            "validation_indices": validation,
            "train_indices": train,
        })
    return plans


def validate_candidate_bank(common: Any) -> dict[int, dict[str, Any]]:
    bank = common.candidate_bank()
    require(len(bank) == 150, "candidate bank exact 150 cells")
    require([row.selector_ordinal for row in bank] == list(range(150)),
            "candidate bank canonical selector order")
    output: dict[int, dict[str, Any]] = {}
    independently_frozen = []
    for topology_id, topology in enumerate(IC_TOPOLOGIES):
        for state_index, states in enumerate(IC_STATE_SIZES):
            for reset_index, reset_length in enumerate(IC_RESET_LENGTHS):
                selector = ((topology_id * len(IC_STATE_SIZES) + state_index) *
                            len(IC_RESET_LENGTHS) + reset_index)
                independently_frozen.append({
                    "topology": topology, "topology_id": topology_id,
                    "states": states, "reset_length": reset_length,
                    "selector_ordinal": selector,
                })
    require(len(independently_frozen) == 150,
            "independent frozen candidate Cartesian product")
    for ordinal, (row, expected) in enumerate(zip(
            bank, independently_frozen, strict=True)):
        record = row.as_dict()
        exact_fields(record, {"topology", "topology_id", "states",
                              "reset_length", "selector_ordinal"},
                     f"candidate[{ordinal}]")
        require(record["selector_ordinal"] == ordinal,
                f"candidate[{ordinal}] selector")
        require_deep_equal(record, expected,
                           f"candidate[{ordinal}] independent frozen cell")
        output[ordinal] = record
    require(len({canonical_json(row) for row in output.values()}) == 150,
            "candidate bank unique cells")
    return output


def recompute_workload(plans: Sequence[Mapping[str, Any]],
                       streams: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    folds = []
    fold_details = []
    total = 0
    count_updates = 0
    length_updates = 0
    for plan in plans:
        train = sum(int(streams[index]["symbols"])
                    for index in plan["train_indices"])
        validation = sum(int(streams[index]["symbols"])
                         for index in plan["validation_indices"])
        development = sum(int(streams[index]["symbols"])
                          for index in plan["development_indices"])
        test = sum(int(streams[index]["symbols"])
                   for index in plan["test_indices"])
        updates = 150 * (train + validation) + development + test
        fold_count_updates = 150 * train + development
        fold_length_updates = 150 * validation + test
        require(fold_count_updates + fold_length_updates == updates,
                "independent fold count/length partition")
        total += updates
        count_updates += fold_count_updates
        length_updates += fold_length_updates
        folds.append((int(plan["component_ordinal"]),
                      tuple(int(v) for v in plan["identity_indices"]), updates))
        fold_details.append({
            "component_ordinal": int(plan["component_ordinal"]),
            "train_symbols": train,
            "validation_symbols": validation,
            "development_symbols": development,
            "test_symbols": test,
            "train_streams": len(plan["train_indices"]),
            "validation_streams": len(plan["validation_indices"]),
            "development_streams": len(plan["development_indices"]),
            "test_streams": len(plan["test_indices"]),
            "count_updates": fold_count_updates,
            "length_updates": fold_length_updates,
        })
    full = sum(int(row["symbols"]) for row in streams)
    # projected_updates() admits a conservative final fit plus exact final
    # scoring pass.  The literal source path actually launches only the final
    # fit on the GPU; arithmetic encoding/scoring is host-side.  Keep the
    # admission workload and the independently derived observed CUDA workload
    # as two distinct quantities.
    total += 2 * full
    count_updates += full
    return {"folds": tuple(folds), "fold_details": tuple(fold_details),
            "full_symbols": full, "full_streams": len(streams),
            "exact_primary_updates": total,
            "expected_observed_count_updates": count_updates,
            "expected_observed_length_updates": length_updates,
            "expected_observed_cuda_updates": count_updates + length_updates,
            "admission_minus_observed_cuda_updates": full}


def verify_scientific(
    scientific: dict[str, Any],
    modules: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    exact_fields(scientific, {
        "kind", "primary_policy", "status", "folds", "skipped_folds",
        "estimable", "pooled_exact_heldout_saving_bpw",
        "minimum_fold_exact_saving_bpw",
        "dependence_component_mean_saving_bpw_diagnostic_only",
        "confidence_rule", "independent_component_count",
        "all_dependence_components_positive",
        "leave_one_component_out_pooled_saving_bpw_diagnostic_only",
        "candidate_vote_counts", "final_topology_selected_from_nested_fold_votes",
        "passes_pooled_standalone_threshold",
        "passes_every_disjoint_component_positive", "passes_heldout_gate",
        "positive_promotion"}, "scientific primary")
    require(scientific["kind"] ==
            "disjoint_stream_owner_dependence_component_holdout",
            "scientific kind")
    require(scientific["primary_policy"] == "exact_identity",
            "scientific primary policy")
    require(scientific["status"] ==
            "PASS_DISJOINT_DEPENDENCE_COMPONENT_HOLDOUT",
            "scientific status")
    require(scientific["estimable"] is True and
            scientific["positive_promotion"] is False,
            "scientific nonpromotion")
    require(scientific["skipped_folds"] == [], "scientific skipped folds")
    require(scientific["confidence_rule"] ==
            "no iid confidence interval; promotion requires disjoint owner-stream components, pooled literal saving at target, and every component strictly positive",
            "scientific frozen confidence rule")
    common = modules["common"]
    codec = modules["codec"]
    stage = modules["stage"]
    protocol = modules["protocol"]
    plans = independent_component_plan(panel, int(common.INNER_VALIDATION_MODULUS))
    require(len(plans) == 3, "independent three dependence components")
    bank = validate_candidate_bank(common)
    workload = recompute_workload(plans, panel["streams"])
    require(workload["folds"] == PINNED_FOLD_UPDATES,
            "independent exact fold update pins")
    require(workload["full_symbols"] == PINNED_PANEL_SYMBOLS,
            "independent panel symbols")
    require(workload["exact_primary_updates"] == PINNED_PRIMARY_UPDATES,
            "independent primary update count")
    folds = scientific["folds"]
    require(isinstance(folds, list) and len(folds) == len(plans),
            "scientific fold count")
    fold_fields = {
        "outer_dependence_component_ordinal", "outer_identity_indices",
        "outer_identities_from_artifact", "development_exclusion_policy",
        "test_stream_ordinals", "development_stream_ordinals",
        "inner_train_stream_ordinals", "inner_validation_stream_ordinals",
        "selected_by_inner_validation_only",
        "inner_validation_exact_charged_bits",
        "literal_authenticated_current_baseline_container_bits",
        "literal_candidate_container_bits",
        "literal_selected_model_aligned_increment_bits",
        "literal_test_saving_after_exact_container_delta_bits",
        "allocated_test_weights", "allocated_baseline_bits",
        "allocated_candidate_bits", "exact_test_saving_bpw",
    }
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    baseline_bits = stage.literal_current_baseline_score(protocol, codec, panel)
    require(baseline_bits == independent_literal_layout_bits(panel, 0, {}),
            "scientific independent current-baseline literal layout")
    allocated_total = 0.0
    pooled_saved = 0.0
    values: list[float] = []
    votes: dict[int, int] = {}
    for ordinal, (fold, plan) in enumerate(zip(folds, plans, strict=True)):
        exact_fields(fold, fold_fields, f"scientific fold[{ordinal}]")
        require(fold["outer_dependence_component_ordinal"] == ordinal,
                f"fold[{ordinal}] component ordinal")
        require_deep_equal(fold["outer_identity_indices"],
                           plan["identity_indices"],
                           f"fold[{ordinal}] identity indices")
        require_deep_equal(fold["outer_identities_from_artifact"],
                           [list(identities[index])
                            for index in plan["identity_indices"]],
                           f"fold[{ordinal}] semantic identities")
        require(fold["development_exclusion_policy"] == "exact_identity",
                f"fold[{ordinal}] exclusion policy")
        for field, plan_field in (
            ("test_stream_ordinals", "test_indices"),
            ("development_stream_ordinals", "development_indices"),
            ("inner_train_stream_ordinals", "train_indices"),
            ("inner_validation_stream_ordinals", "validation_indices"),
        ):
            expected = [int(streams[index]["stream_ordinal"])
                        for index in plan[plan_field]]
            require_deep_equal(fold[field], expected,
                               f"fold[{ordinal}] {field}")
        selected = fold["selected_by_inner_validation_only"]
        require(isinstance(selected, dict) and
                selected.get("selector_ordinal") in bank,
                f"fold[{ordinal}] selected candidate")
        selected_ordinal = int(selected["selector_ordinal"])
        require_deep_equal(selected, bank[selected_ordinal],
                           f"fold[{ordinal}] candidate bank identity")
        votes[selected_ordinal] = votes.get(selected_ordinal, 0) + 1
        require(fold["literal_authenticated_current_baseline_container_bits"]
                == baseline_bits, f"fold[{ordinal}] baseline container bits")
        zero_frequencies = common.q16_frequencies_from_counts(
            common.zero_counts(common.candidate_bank()[selected_ordinal]))
        model_only = stage.literal_validation_score(
            common, protocol, codec, panel, [], [],
            common.candidate_bank()[selected_ordinal], zero_frequencies,
        )
        model_increment = model_only - baseline_bits
        zero_model_packet = common.serialize_model(
            common.candidate_bank()[selected_ordinal], zero_frequencies)
        require(model_only == independent_literal_layout_bits(
            panel, len(zero_model_packet), {}),
            f"fold[{ordinal}] independent model-only literal layout")
        require(fold["literal_selected_model_aligned_increment_bits"] ==
                model_increment, f"fold[{ordinal}] exact model/layout cost")
        expected_weights = float(sum(int(streams[index]["weight_charge"])
                                     for index in plan["test_indices"]))
        expected_baseline_allocated = float(sum(
            8 * int(streams[index]["baseline_payload_bytes"])
            for index in plan["test_indices"]))
        require_float_equal(fold["allocated_test_weights"], expected_weights,
                            f"fold[{ordinal}] allocated weights")
        require_float_equal(fold["allocated_baseline_bits"],
                            expected_baseline_allocated,
                            f"fold[{ordinal}] baseline allocation")
        candidate_allocated = fold["allocated_candidate_bits"]
        require(type(candidate_allocated) is float and
                math.isfinite(candidate_allocated) and
                candidate_allocated >= 0.0 and candidate_allocated.is_integer(),
                f"fold[{ordinal}] candidate allocation exact-float integer")
        # The complete literal layout is not additive in payload bytes because
        # frame, region, page and rate-floor alignment can change.  The exact
        # selected replay below can recompute it from logical lengths; without
        # that expensive replay we verify only its integer geometry and the
        # exact downstream subtraction, never an invalid additive shortcut.
        literal_candidate = exact_int(
            fold["literal_candidate_container_bits"],
            f"fold[{ordinal}] literal candidate bits", 1, 1 << 50)
        saved = baseline_bits - literal_candidate
        require(fold["literal_test_saving_after_exact_container_delta_bits"] == saved,
                f"fold[{ordinal}] exact literal saving")
        saving = saved / expected_weights
        require_float_equal(fold["exact_test_saving_bpw"], saving,
                            f"fold[{ordinal}] exact saving bpw")
        exact_int(fold["inner_validation_exact_charged_bits"],
                  f"fold[{ordinal}] validation charged bits", 1, 1 << 50)
        allocated_total += expected_weights
        pooled_saved += float(saved)
        values.append(saving)
    require(abs(allocated_total - int(panel["weights"])) <= 1e-6,
            "scientific folds partition weights")
    pooled = pooled_saved / allocated_total
    require_float_equal(scientific["pooled_exact_heldout_saving_bpw"], pooled,
                        "scientific pooled saving")
    require_float_equal(scientific["minimum_fold_exact_saving_bpw"], min(values),
                        "scientific minimum saving")
    require_float_equal(
        scientific["dependence_component_mean_saving_bpw_diagnostic_only"],
        statistics.fmean(values), "scientific component mean")
    component_positive = all(value > 0.0 for value in values)
    require(scientific["all_dependence_components_positive"] is component_positive,
            "scientific all components positive")
    require(scientific["passes_every_disjoint_component_positive"] is
            component_positive, "scientific positive-component gate")
    require_float_equal(float(common.STANDALONE_REQUIRED_SAVING_BPW),
                        INDEPENDENT_STANDALONE_REQUIRED_SAVING_BPW,
                        "independent standalone threshold/source equality")
    threshold = pooled >= INDEPENDENT_STANDALONE_REQUIRED_SAVING_BPW
    require(scientific["passes_pooled_standalone_threshold"] is threshold,
            "scientific standalone threshold")
    heldout_gate = threshold and component_positive
    require(scientific["passes_heldout_gate"] is heldout_gate,
            "scientific heldout gate")
    require(scientific["independent_component_count"] == len(plans),
            "scientific component count")
    leave_one = []
    for omitted, fold in enumerate(folds):
        kept_weights = allocated_total - float(fold["allocated_test_weights"])
        kept_bits = pooled_saved - float(
            fold["literal_test_saving_after_exact_container_delta_bits"])
        leave_one.append({"omitted_component_ordinal": omitted,
                          "pooled_saving_bpw": kept_bits / kept_weights})
    require_deep_equal(
        scientific["leave_one_component_out_pooled_saving_bpw_diagnostic_only"],
        leave_one, "scientific leave-one-component-out")
    require_deep_equal(scientific["candidate_vote_counts"],
                       {str(key): value for key, value in votes.items()},
                       "scientific candidate votes")
    winner = min(votes, key=lambda value: (-votes[value], value))
    require_deep_equal(scientific["final_topology_selected_from_nested_fold_votes"],
                       bank[winner], "scientific vote-selected final topology")
    return {
        "plans": plans, "workload": workload, "candidate_bank": bank,
        "heldout_gate": heldout_gate, "winner": bank[winner],
        "pooled_saving_bpw": pooled,
        "unreplayed_gpu_fact":
            "alternative-candidate Q0.16 fits and their 150-way ordering are not emitted; selected cells, exact splits, model-only layout cost, fold allocations, vote and all terminal gates are replayed",
    }


def verify_source_hashes(result: Mapping[str, Any],
                         v9: Mapping[str, Any],
                         v8: Mapping[str, Any]) -> None:
    expected = {
        "v9_source_manifest_sha256": KNOWN_V9_MANIFEST_SHA256,
        "v9_source_snapshot_root_sha256": KNOWN_V9_SOURCE_ROOT_SHA256,
        "v9_members": v9["hashes"],
        "sealed_v8_manifest_sha256": KNOWN_V8_MANIFEST_SHA256,
        "sealed_v8_source_snapshot_root_sha256": KNOWN_V8_SOURCE_ROOT_SHA256,
        "pinned_support_sha256": KNOWN_SUPPORT_SHA256,
        "strata_expert_local_codec_common_sha256": KNOWN_STRATA_COMMON_SHA256,
        "strata_v2_klt_mixed_independent_auditor_sha256":
            KNOWN_FROZEN_AUDITOR_SHA256,
    }
    require_deep_equal(result["source_hashes"], expected,
                       "RESULT source hash closure")
    require(v8["hashes"] == {name: KNOWN_V8_MEMBERS[name][1]
                             for name in V8_REQUIRED_MEMBERS},
            "independently pinned every v8 member")


def verify_decoder_pipeline(
    result: Mapping[str, Any],
    decoder_file: Mapping[str, Any],
    v8: Mapping[str, Any],
    bridge: Mapping[str, Any],
    support_constants: Mapping[str, Any],
) -> dict[str, Any]:
    expected_decoder = {
        "schema": "uwfa-sc-v9-primary-decoder-bundle-v0",
        "members": [
            {"name": "strata_expert_local_codec/common.py",
             "sha256": KNOWN_STRATA_COMMON_SHA256},
            {"name": "strata_v2_klt_mixed_independent_auditor_v1.py",
             "sha256": KNOWN_FROZEN_AUDITOR_SHA256},
            {"name": "strata_sc_adapter.py",
             "sha256": v8["hashes"]["strata_sc_adapter.py"]},
            {"name": "universal_adapter.py",
             "sha256": v8["hashes"]["universal_adapter.py"]},
            {"name": "container_codec.py",
             "sha256": v8["hashes"]["container_codec.py"]},
        ],
        "exploratory_semantic_bridge": dict(bridge),
        "semantic_bridge_container_abi": "list[list[int]]",
        "tuple_rows_forbidden_by_numpy_advanced_index_semantics": True,
        "single_artifact_panel_cache_required": True,
    }
    require_deep_equal(decoder_file, expected_decoder, "DECODER_BUNDLE")
    require_deep_equal(result["decoder_bundle"], expected_decoder,
                       "RESULT.decoder_bundle")
    decoder_sha = sha256(canonical_json(expected_decoder))
    require(result["decoder_bundle_sha256"] == decoder_sha,
            "decoder bundle canonical aggregate")
    pipeline = exact_fields(result["pipeline_record"], {
        "schema", "v9_source_snapshot_root_sha256",
        "v8_source_snapshot_root_sha256", "sealed_v8_manifest_sha256",
        "pinned_support_sha256", "runner_sha256", "decoder_bundle_sha256",
        "baseline_plan_sha256", "scope"}, "pipeline record")
    require(pipeline == {
        "schema": "uwfa-sc-v9-primary-pipeline-v0",
        "v9_source_snapshot_root_sha256": KNOWN_V9_SOURCE_ROOT_SHA256,
        "v8_source_snapshot_root_sha256": KNOWN_V8_SOURCE_ROOT_SHA256,
        "sealed_v8_manifest_sha256": KNOWN_V8_MANIFEST_SHA256,
        "pinned_support_sha256": KNOWN_SUPPORT_SHA256,
        "runner_sha256": KNOWN_V9_RUNNER_SHA256,
        "decoder_bundle_sha256": decoder_sha,
        "baseline_plan_sha256": support_constants["BASELINE_PLAN_SHA256"],
        "scope": "exact primary nested holdout plus final physical container only",
    }, "pipeline exact source closure")
    pipeline_sha = sha256(canonical_json(pipeline))
    require(result["pipeline_sha256"] == pipeline_sha,
            "pipeline canonical aggregate")
    return {"decoder_sha256": decoder_sha, "pipeline_sha256": pipeline_sha}


def verify_score(
    result: Mapping[str, Any],
    score: dict[str, Any],
    score_bytes: bytes,
    pins: Pins,
    panel: Mapping[str, Any],
    decoder_sha256: str,
) -> None:
    exact_fields(score, {
        "schema", "status", "artifact_sha256", "artifact_bytes", "weights",
        "relative_mse", "sse_fp64", "source_energy_fp64", "normalization",
        "reconstruction_f64_sha256", "original_source_panel_sha256",
        "independent_decoder_source_sha256", "score_receipt_sha256"},
        "BOUND_BASELINE_SCORE")
    require(score["schema"] == "uwfa-bound-baseline-score-v8",
            "score schema")
    require(score["status"] == "PASS_INDEPENDENT_BASELINE_SCORE",
            "score status")
    require(score["normalization"] ==
            "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
            "score frozen normalization")
    require(score["artifact_sha256"] == KNOWN_ARTIFACT_SHA256 and
            score["artifact_bytes"] == KNOWN_ARTIFACT_BYTES and
            score["weights"] == SOURCE_WEIGHTS, "score artifact geometry")
    identity = pins.original_source_identity
    for key in ("relative_mse", "sse_fp64", "source_energy_fp64"):
        require_float_equal(score[key], float(identity[key]), f"score {key}")
    require(score["reconstruction_f64_sha256"] ==
            identity["reconstruction_f64_sha256"], "score reconstruction")
    require(score["original_source_panel_sha256"] ==
            identity["source_full_geometry_sha256"], "score source panel")
    require(score["independent_decoder_source_sha256"] == decoder_sha256,
            "score decoder bundle")
    require(score["score_receipt_sha256"] == identity["score_receipt_sha256"],
            "score external receipt pin")
    verify_internal_seal(score, "score_receipt_sha256", "score")
    require_deep_equal(result["baseline_score"], score,
                       "RESULT baseline score")
    require(score_bytes == pretty_json(score), "score canonical pretty bytes")
    require(panel["reconstruction"]["full_reconstruction_f64_sha256"] ==
            score["reconstruction_f64_sha256"],
            "score independently replayed reconstruction")


TELEMETRY_INTEGER_FIELDS = (
    "h2d_bytes", "h2d_payload_bytes", "h2d_root_descriptor_bytes",
    "h2d_subset_descriptor_bytes", "h2d_launch_descriptor_bytes",
    "h2d_model_table_bytes", "h2d_kernel_scalar_bytes", "d2h_bytes",
    "d2d_descriptor_bytes", "device_output_allocation_bytes", "kernel_count",
    "count_kernel_count", "length_kernel_count", "count_cell_symbol_updates",
    "length_cell_symbol_updates", "pack_calls", "subset_calls", "to_host_calls",
    "telemetry_samples", "peak_process_tree_rss_bytes", "peak_process_hwm_bytes",
    "incremental_peak_process_tree_rss_bytes", "peak_vram_incremental_bytes",
    "peak_default_pool_used_bytes", "peak_default_pool_total_bytes",
    "peak_pinned_pool_free_blocks", "baseline_free_vram_bytes",
    "total_vram_bytes", "resource_preflight_calls",
)


def verify_environment_telemetry(
    environment: Any, identity: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    """Independently validate the frozen CUDA receipt's literal conservation.

    This is deliberately separate from sealed-v8's validation routine.  It
    does not attest that a GPU performed the work; it proves that the
    externally pinned receipt is internally complete, device-bound and
    arithmetically conservative before phase deltas are considered.
    """
    row = exact_fields(environment, {
        "cupy_version", "cuda_runtime_version", "cuda_driver_version",
        "python_version", "platform", "device_id", "device_name",
        "device_uuid", "pci_bus_id", "compute_capability",
        "current_free_vram_bytes", "total_vram_bytes", "statistics",
        "telemetry_samples", "host_byteorder",
        "explicit_device_synchronization_at_phase_boundaries_and_after_every_kernel",
        "fatal_telemetry_sampling", "transfer_formula",
    }, label)
    require(row["device_name"] == identity["device_name"] ==
            EXPECTED_DEVICE_NAME, f"{label}: expected device name")
    require(row["device_uuid"] == identity["device_uuid"] and
            row["pci_bus_id"] == identity["pci_bus_id"],
            f"{label}: independent device identity")
    require(row["host_byteorder"] == "little" and
            row["fatal_telemetry_sampling"] is True and
            row["explicit_device_synchronization_at_phase_boundaries_and_after_every_kernel"]
            is True, f"{label}: fail-closed telemetry flags")
    exact_int(row["device_id"], f"{label}.device_id", 0, 255)
    exact_int(row["cuda_runtime_version"], f"{label}.runtime", 1, (1 << 31) - 1)
    exact_int(row["cuda_driver_version"], f"{label}.driver", 1, (1 << 31) - 1)
    capability = row["compute_capability"]
    require(isinstance(capability, list) and len(capability) == 2,
            f"{label}: compute capability")
    exact_int(capability[0], f"{label}.compute_major", 1, 99)
    exact_int(capability[1], f"{label}.compute_minor", 0, 99)
    total_vram = exact_int(row["total_vram_bytes"], f"{label}.total_vram", 1)
    exact_int(row["current_free_vram_bytes"], f"{label}.free_vram", 1,
              total_vram)
    require_deep_equal(row["transfer_formula"], {
        "root_pack_h2d": "4*N_symbols + 16*S_root",
        "subset_h2d": "16*S_subset per descriptor materialization",
        "launch_descriptor_h2d": "16*S_launch on every fit-count or exact-length launch",
        "model_h2d": "2*states*384 per exact-length call",
        "kernel_scalars_h2d": "16 per RawKernel launch",
        "d2h": "8*states*384*2 per count result plus 8*S_subset per length result",
    }, f"{label}: frozen transfer formula")
    statistics_row = exact_fields(
        row["statistics"], set(TELEMETRY_INTEGER_FIELDS) |
        {"jit_compile_seconds", "kernel_wall_seconds", "last_pack_resource_plan"},
        f"{label}.statistics")
    clean: dict[str, Any] = {}
    for name in TELEMETRY_INTEGER_FIELDS:
        clean[name] = exact_int(statistics_row[name], f"{label}.{name}")
    require(clean["telemetry_samples"] > 0 and
            clean["peak_process_tree_rss_bytes"] > 0 and
            clean["peak_process_hwm_bytes"] > 0 and
            clean["total_vram_bytes"] == total_vram and
            0 < clean["baseline_free_vram_bytes"] <= total_vram and
            0 <= clean["peak_vram_incremental_bytes"] <=
            clean["baseline_free_vram_bytes"],
            f"{label}: fatal telemetry present")
    require(clean["h2d_bytes"] == sum(clean[name] for name in (
        "h2d_payload_bytes", "h2d_root_descriptor_bytes",
        "h2d_subset_descriptor_bytes", "h2d_launch_descriptor_bytes",
        "h2d_model_table_bytes", "h2d_kernel_scalar_bytes")),
        f"{label}: H2D category conservation")
    require(clean["kernel_count"] == clean["count_kernel_count"] +
            clean["length_kernel_count"],
            f"{label}: kernel category conservation")
    for name in ("jit_compile_seconds", "kernel_wall_seconds"):
        value = statistics_row[name]
        require(type(value) is float and math.isfinite(value) and value >= 0.0,
                f"{label}: finite nonnegative {name}")
    plan = exact_fields(statistics_row["last_pack_resource_plan"], {
        "symbols", "streams", "payload_host_and_device_bytes",
        "root_descriptor_device_bytes",
        "additional_host_bytes_including_reserve",
        "device_required_bytes_including_aux_and_reserve",
        "current_process_tree_rss_bytes", "current_process_hwm_bytes",
        "projected_process_tree_rss_bytes", "current_free_vram_bytes",
        "current_total_vram_bytes", "host_cap_bytes", "vram_cap_bytes",
        "passes", "checked_before_blob_concatenation_or_cupy_allocation",
    }, f"{label}.last_pack_resource_plan")
    plan_symbols = exact_int(plan["symbols"], f"{label}.plan.symbols", 1,
                             1 << 54)
    plan_streams = exact_int(plan["streams"], f"{label}.plan.streams", 1,
                             65_536)
    max_host = 96 * (1 << 30)
    max_vram = 28 * (1 << 30)
    host_reserve = 1 << 30
    vram_reserve = 2 * (1 << 30)
    maximum_auxiliary = (64 * 384 * 2 * 8 + 64 * 384 * 2 +
                         40 * 65_536)
    payload_bytes = 4 * plan_symbols
    root_bytes = 16 * plan_streams
    additional_host = payload_bytes + 64 * plan_streams + host_reserve
    required_device = (payload_bytes + root_bytes + maximum_auxiliary +
                       vram_reserve)
    require(plan["payload_host_and_device_bytes"] == payload_bytes and
            plan["root_descriptor_device_bytes"] == root_bytes and
            plan["additional_host_bytes_including_reserve"] == additional_host and
            plan["device_required_bytes_including_aux_and_reserve"] ==
            required_device and plan["host_cap_bytes"] == max_host and
            plan["vram_cap_bytes"] == max_vram,
            f"{label}: resource-plan frozen arithmetic")
    current_rss = exact_int(plan["current_process_tree_rss_bytes"],
                            f"{label}.plan.current_rss", 1)
    free_vram = exact_int(plan["current_free_vram_bytes"],
                          f"{label}.plan.free_vram", 1)
    require(plan["projected_process_tree_rss_bytes"] ==
            current_rss + additional_host and
            plan["current_total_vram_bytes"] == total_vram and
            plan["current_total_vram_bytes"] >= free_vram and
            plan["passes"] is True and
            plan["checked_before_blob_concatenation_or_cupy_allocation"] is True and
            current_rss + additional_host <= max_host and
            required_device <= min(max_vram, free_vram),
            f"{label}: resource-plan pass predicate")
    samples = row["telemetry_samples"]
    require(isinstance(samples, list) and len(samples) == clean["telemetry_samples"],
            f"{label}: telemetry sample count")
    sample_fields = {
        "phase", "process_tree_rss_bytes", "process_hwm_bytes",
        "free_vram_bytes", "total_vram_bytes", "default_pool_used_bytes",
        "default_pool_total_bytes", "pinned_pool_free_blocks",
    }
    for index, sample in enumerate(samples):
        exact_fields(sample, sample_fields, f"{label}.sample[{index}]")
        require(isinstance(sample["phase"], str) and sample["phase"],
                f"{label}.sample[{index}]: phase")
        for name in sample_fields - {"phase"}:
            minimum = 1 if name in {
                "process_tree_rss_bytes", "process_hwm_bytes",
                "free_vram_bytes", "total_vram_bytes",
            } else 0
            exact_int(sample[name], f"{label}.sample[{index}].{name}", minimum)
        require(sample["total_vram_bytes"] == total_vram and
                0 < sample["free_vram_bytes"] <= sample["total_vram_bytes"],
                f"{label}.sample[{index}]: VRAM conservation")
        require(sample["default_pool_used_bytes"] <=
                sample["default_pool_total_bytes"],
                f"{label}.sample[{index}]: default-pool conservation")

    # Every statistic below is updated by _sample in the frozen backend.  Bind
    # the literal trace to those cumulative extrema rather than accepting a
    # well-typed but incomplete list of samples.
    require(samples[0]["phase"] == "post_jit",
            f"{label}: telemetry begins at post_jit")
    require(clean["peak_process_tree_rss_bytes"] ==
            max(sample["process_tree_rss_bytes"] for sample in samples) and
            clean["peak_process_hwm_bytes"] ==
            max(sample["process_hwm_bytes"] for sample in samples),
            f"{label}: host peak/sample conservation")
    require(clean["peak_default_pool_used_bytes"] ==
            max(sample["default_pool_used_bytes"] for sample in samples) and
            clean["peak_default_pool_total_bytes"] ==
            max(sample["default_pool_total_bytes"] for sample in samples) and
            clean["peak_pinned_pool_free_blocks"] ==
            max(sample["pinned_pool_free_blocks"] for sample in samples),
            f"{label}: pool peak/sample conservation")
    require(clean["peak_vram_incremental_bytes"] ==
            max(0, clean["baseline_free_vram_bytes"] -
                min(sample["free_vram_bytes"] for sample in samples)),
            f"{label}: VRAM peak/sample conservation")
    baseline_rss = (clean["peak_process_tree_rss_bytes"] -
                    clean["incremental_peak_process_tree_rss_bytes"])
    require(0 < baseline_rss <= samples[0]["process_tree_rss_bytes"] and
            clean["incremental_peak_process_tree_rss_bytes"] ==
            clean["peak_process_tree_rss_bytes"] - baseline_rss,
            f"{label}: host baseline/peak conservation")
    clean["_samples"] = samples
    clean["_statistics"] = statistics_row
    return clean


def expected_primary_telemetry_phases(
    fold_count: int, candidate_count: int,
) -> list[str]:
    """Restate the exact frozen primary backend call/sample order."""
    exact_int(fold_count, "primary telemetry fold count", 1, 64)
    exact_int(candidate_count, "primary telemetry candidate count", 1, 10_000)

    def operation(kernel: str) -> list[str]:
        require(kernel in {"count_kernel", "length_kernel"},
                "primary telemetry kernel phase")
        return ["subset_descriptors", kernel, "device_to_host"]

    phases = ["pack_streams"]
    for _fold in range(fold_count):
        for _candidate in range(candidate_count):
            phases.extend(operation("count_kernel"))
            phases.extend(operation("length_kernel"))
        # Refit the validation winner on the development side, then length
        # score exactly that fold's held-out test side.
        phases.extend(operation("count_kernel"))
        phases.extend(operation("length_kernel"))
    # The vote-selected final topology is fitted once on the full panel.
    phases.extend(operation("count_kernel"))
    phases.append("environment_receipt")
    return phases


def verify_primary_telemetry_trace(
    before: Mapping[str, Any], after: Mapping[str, Any], *,
    fold_count: int, candidate_count: int,
) -> dict[str, Any]:
    """Bind the final cumulative trace to the exact primary-only suffix."""
    before_samples = before.get("_samples")
    after_samples = after.get("_samples")
    require(isinstance(before_samples, list) and isinstance(after_samples, list),
            "primary telemetry retained sample lists")
    require(len(after_samples) >= len(before_samples),
            "primary telemetry cumulative sample length")
    require_deep_equal(after_samples[:len(before_samples)], before_samples,
                       "primary telemetry immutable preflight prefix")
    suffix = after_samples[len(before_samples):]
    expected = expected_primary_telemetry_phases(fold_count, candidate_count)
    require([sample["phase"] for sample in suffix] == expected,
            "primary telemetry exact ordered phase suffix")
    require(len(suffix) == len(expected),
            "primary telemetry exact suffix sample count")

    before_stats = before.get("_statistics")
    after_stats = after.get("_statistics")
    require(isinstance(before_stats, dict) and isinstance(after_stats, dict),
            "primary telemetry retained statistics")
    peak_fields = {
        "peak_process_tree_rss_bytes": "process_tree_rss_bytes",
        "peak_process_hwm_bytes": "process_hwm_bytes",
        "peak_default_pool_used_bytes": "default_pool_used_bytes",
        "peak_default_pool_total_bytes": "default_pool_total_bytes",
        "peak_pinned_pool_free_blocks": "pinned_pool_free_blocks",
    }
    for statistic, sample_field in peak_fields.items():
        require(after_stats[statistic] == max(
            before_stats[statistic],
            max(sample[sample_field] for sample in suffix),
        ), f"primary telemetry suffix peak conservation: {statistic}")
    require(after_stats["peak_vram_incremental_bytes"] == max(
        before_stats["peak_vram_incremental_bytes"],
        after_stats["baseline_free_vram_bytes"] -
        min(sample["free_vram_bytes"] for sample in suffix),
    ), "primary telemetry suffix peak conservation: VRAM")
    baseline_rss = (before_stats["peak_process_tree_rss_bytes"] -
                    before_stats["incremental_peak_process_tree_rss_bytes"])
    require(after_stats["incremental_peak_process_tree_rss_bytes"] ==
            after_stats["peak_process_tree_rss_bytes"] - baseline_rss,
            "primary telemetry suffix peak conservation: incremental host")
    require(after_stats["baseline_free_vram_bytes"] ==
            before_stats["baseline_free_vram_bytes"] and
            after_stats["total_vram_bytes"] == before_stats["total_vram_bytes"],
            "primary telemetry immutable device baselines")
    return {
        "preflight_samples": len(before_samples),
        "primary_suffix_samples": len(suffix),
        "primary_suffix_sha256": sha256(canonical_json(suffix)),
        "exact_ordered_phase_suffix": True,
    }


def verify_preflight_runtime(
    result: Mapping[str, Any],
    preflight: dict[str, Any],
    modules: Mapping[str, Any],
    scientific_audit: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    exact_fields(preflight, {"schema", "source_snapshot_root_sha256", "all150",
                             "representative", "independent_gpu_identity",
                             "receipt_sha256"}, "SOURCE_PREFLIGHT")
    require(preflight["schema"] == "uwfa-sc-v8-bound-source-preflight",
            "preflight schema")
    require(preflight["source_snapshot_root_sha256"] ==
            KNOWN_V8_SOURCE_ROOT_SHA256, "preflight v8 source root")
    verify_internal_seal(preflight, "receipt_sha256", "SOURCE_PREFLIGHT")
    all150 = preflight["all150"]
    representative = preflight["representative"]
    identity = exact_fields(preflight["independent_gpu_identity"], {
        "schema", "status", "device_uuid", "pci_bus_id", "device_name",
        "provider", "identity_receipt_sha256",
    }, "independent GPU identity")
    require(identity["schema"] == "uwfa-sc-v8-independent-gpu-identity" and
            identity["status"] == "PASS_INDEPENDENT_GPU_IDENTITY" and
            identity["device_name"] == EXPECTED_DEVICE_NAME and
            identity["provider"] == "nvidia-smi",
            "independent GPU identity schema/device")
    identity_clean = dict(identity)
    identity_receipt = identity_clean.pop("identity_receipt_sha256")
    digest(identity_receipt, "independent GPU identity receipt")
    require(sha256(canonical_json(identity_clean)) == identity_receipt,
            "independent GPU identity seal")
    require(all150["status"] == "PASS_ALL_150_CPU_CUPY_EXACT_REPEATED",
            "preflight all150 status")
    require(representative["status"] == "PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD",
            "preflight representative status")
    review = exact_fields(result["source_free_review"], {
        "status", "v9_source_snapshot_root_sha256",
        "v8_source_snapshot_root_sha256", "preflight_receipt_sha256",
        "support_sha256", "measured_updates_per_second",
        "conservative_updates_per_second", "device_name", "device_uuid",
        "pci_bus_id", "receipt_sha256"}, "source-free review")
    review_clean = dict(review)
    review_receipt = review_clean.pop("receipt_sha256")
    require(review["status"] == "PASS_AUTHENTICATED_SOURCE_FREE_REVIEW",
            "source-free review status")
    require(review["v9_source_snapshot_root_sha256"] ==
            KNOWN_V9_SOURCE_ROOT_SHA256, "review v9 root")
    require(review["v8_source_snapshot_root_sha256"] ==
            KNOWN_V8_SOURCE_ROOT_SHA256, "review v8 root")
    require(review["preflight_receipt_sha256"] == preflight["receipt_sha256"],
            "review preflight binding")
    require(review["support_sha256"] == KNOWN_SUPPORT_SHA256,
            "review support binding")
    require(review["device_name"] == identity["device_name"] and
            review["device_uuid"] == identity["device_uuid"] and
            review["pci_bus_id"] == identity["pci_bus_id"],
            "review independent GPU identity")
    measured = float(review["measured_updates_per_second"])
    conservative = float(review["conservative_updates_per_second"])
    require(math.isfinite(measured) and math.isfinite(conservative) and
            abs(conservative - 0.5 * measured) <=
            8.0 * math.ulp(0.5 * measured),
            "review conservative throughput derivation")
    require(CONSERVATIVE_THROUGHPUT_MIN <= conservative <=
            CONSERVATIVE_THROUGHPUT_MAX,
            "review frozen conservative throughput bounds")
    require(sha256(canonical_json(review_clean)) == review_receipt,
            "source-free review seal")
    runtime = representative["runtime_projection"]
    require_float_equal(runtime["measured_updates_per_second"], measured,
                        "representative/review measured throughput")
    require_float_equal(runtime["conservative_updates_per_second"], conservative,
                        "representative/review conservative throughput")
    admission = exact_fields(result["runtime_admission"], {
        "schema", "status", "exact_primary_cell_symbol_updates",
        "authenticated_live_measured_updates_per_second",
        "authenticated_live_conservative_updates_per_second",
        "conservative_fraction_of_measured", "throughput_tamper_bounds",
        "projected_primary_gpu_kernel_work_seconds",
        "primary_gpu_kernel_budget_seconds", "passes",
        "is_total_launch_wall_time_projection", "unmodeled_wall_components",
        "evaluation_runner_pins_are_decoder_identity_inputs",
        "deferred_not_counted_or_executed", "admission_sha256"},
        "runtime admission")
    verify_internal_seal(admission, "admission_sha256", "runtime admission")
    require(admission["schema"] == "uwfa-sc-v9-primary-runtime-admission-v0" and
            admission["status"] == "PASS_PRIMARY_KERNEL_WORKLOAD_ADMITTED",
            "runtime admission schema/status")
    require(admission["exact_primary_cell_symbol_updates"] ==
            scientific_audit["workload"]["exact_primary_updates"] ==
            PINNED_PRIMARY_UPDATES, "runtime independent update count")
    require_float_equal(admission["authenticated_live_measured_updates_per_second"],
                        measured, "admission measured throughput")
    require_float_equal(
        admission["authenticated_live_conservative_updates_per_second"],
        conservative, "admission conservative throughput")
    require_float_equal(admission["conservative_fraction_of_measured"], 0.5,
                        "admission conservative fraction")
    require_deep_equal(admission["throughput_tamper_bounds"],
                       [CONSERVATIVE_THROUGHPUT_MIN,
                        CONSERVATIVE_THROUGHPUT_MAX],
                       "admission frozen throughput bounds")
    expected_seconds = PINNED_PRIMARY_UPDATES / conservative
    require_float_equal(admission["projected_primary_gpu_kernel_work_seconds"],
                        expected_seconds, "admission projected kernel work")
    require_float_equal(admission["primary_gpu_kernel_budget_seconds"],
                        PRIMARY_KERNEL_BUDGET_SECONDS,
                        "admission frozen primary kernel budget")
    require(admission["passes"] is True and
            (expected_seconds <= PRIMARY_KERNEL_BUDGET_SECONDS),
            "admission pass predicate")
    require(admission["is_total_launch_wall_time_projection"] is False and
            admission["evaluation_runner_pins_are_decoder_identity_inputs"] is False,
            "runtime scope disclosures")
    require_deep_equal(admission["unmodeled_wall_components"], [
        "one authenticated STRATA panel decode before primary fitting",
        "host-side final arithmetic encode/decode and canonical rebuild",
        "routed and standalone final causal decode plus physical metrics",
        "filesystem publication and synchronization",
    ], "runtime unmodeled wall disclosure")
    require_deep_equal(admission["deferred_not_counted_or_executed"], {
        "four_survivor_shuffles_and_coordinate_diagnostic_maximum_updates":
            PINNED_DEFERRED_MAXIMUM_UPDATES - PINNED_PRIMARY_UPDATES,
        "coordinate_diagnostic_updates": PINNED_DEFERRED_COORDINATE_UPDATES,
        "matched_controls":
            "separate authorization; no control path exists in this runner",
    }, "runtime exact deferred workload disclosure")
    projection = result["original_v8_full_survivor_projection"]
    fresh_projection = modules["stage"].projected_updates(
        modules["common"], modules["protocol"], panel)
    require_deep_equal(projection, fresh_projection,
                       "v8 projection fresh authenticated replay")
    require(projection["exact_cell_symbol_updates"] == PINNED_PRIMARY_UPDATES,
            "v8 projection exact primary updates")
    require(projection["maximum_source_survivor_updates_including_four_shuffles"]
            == PINNED_DEFERRED_MAXIMUM_UPDATES,
            "v8 projection deferred maximum updates")
    require(projection["coordinate_disjoint_diagnostic_cell_symbol_updates"]
            == PINNED_DEFERRED_COORDINATE_UPDATES and
            projection["coordinate_disjoint_diagnostic_estimable_folds"] == 6,
            "v8 projection coordinate diagnostic pins")
    require(projection["passes_pre_fit_resource_budget"] is True and
            projection["passes_pre_fit_runtime_budget"] is False and
            projection["primary_exact_identity_estimable"] is True and
            projection["disjoint_dependence_component_count"] == 3 and
            projection["primary_fold_policy"] ==
            "disjoint_stream_owner_dependence_components",
            "v8 projection exact admission disposition")
    require(projection["static_resource_admission"]["symbols"] ==
            PINNED_PANEL_SYMBOLS and
            projection["static_resource_admission"]["streams"] ==
            PINNED_PANEL_STREAMS and
            projection["static_resource_admission"]["passes"] is True and
            projection["static_resource_admission"]
            ["checked_before_backend_pack_or_cupy_allocation"] is True,
            "v8 projection panel geometry/resource pass")
    observed_folds = tuple(
        (int(row["component_ordinal"]),
         tuple(int(v) for v in row["identity_indices"]),
         int(row["cell_symbol_updates"]))
        for row in projection["folds"]
    )
    require(observed_folds == PINNED_FOLD_UPDATES,
            "v8 projection independent fold updates")
    before = verify_environment_telemetry(
        representative["telemetry"], identity, "representative telemetry")
    require(representative["telemetry"]["statistics"]
            ["last_pack_resource_plan"]["symbols"] ==
            representative["fixture"]["symbols"] and
            representative["telemetry"]["statistics"]
            ["last_pack_resource_plan"]["streams"] == 15,
            "representative last resource-plan fixture binding")
    telemetry = result["telemetry"]
    after = verify_environment_telemetry(
        telemetry, identity, "final primary telemetry")
    require(telemetry["statistics"]["last_pack_resource_plan"]["symbols"] ==
            PINNED_PANEL_SYMBOLS and
            telemetry["statistics"]["last_pack_resource_plan"]["streams"] ==
            PINNED_PANEL_STREAMS,
            "final resource-plan panel binding")

    workload = scientific_audit["workload"]
    fold_details = workload["fold_details"]
    primary = result["scientific_primary_nested_holdout"]
    fold_selected_states = [
        exact_int(row["selected_by_inner_validation_only"]["states"],
                  f"telemetry fold[{index}] selected states", 1, 64)
        for index, row in enumerate(primary["folds"])
    ]
    final_states = exact_int(scientific_audit["winner"]["states"],
                             "telemetry final selected states", 1, 64)
    candidate_state_sum = sum(exact_int(row["states"],
                                        "telemetry candidate states", 1, 64)
                              for row in scientific_audit["candidate_bank"].values())
    count_calls = 150 * len(fold_details) + len(fold_details) + 1
    length_calls = 150 * len(fold_details) + len(fold_details)
    count_stream_uses = sum(
        150 * int(row["train_streams"]) + int(row["development_streams"])
        for row in fold_details) + int(workload["full_streams"])
    length_stream_uses = sum(
        150 * int(row["validation_streams"]) + int(row["test_streams"])
        for row in fold_details)
    count_state_uses = (len(fold_details) * candidate_state_sum +
                        sum(fold_selected_states) + final_states)
    length_state_uses = (len(fold_details) * candidate_state_sum +
                         sum(fold_selected_states))
    length_result_stream_uses = length_stream_uses
    trace = verify_primary_telemetry_trace(
        before, after, fold_count=len(fold_details),
        candidate_count=len(scientific_audit["candidate_bank"]),
    )
    require(trace["primary_suffix_samples"] == 2_723 and
            after["telemetry_samples"] - before["telemetry_samples"] == 2_723,
            "primary telemetry exact 2723-sample suffix")
    expected_delta = {
        "h2d_payload_bytes": 4 * int(workload["full_symbols"]),
        "h2d_root_descriptor_bytes": 16 * int(workload["full_streams"]),
        "h2d_subset_descriptor_bytes":
            16 * (count_stream_uses + length_stream_uses),
        "h2d_launch_descriptor_bytes":
            16 * (count_stream_uses + length_stream_uses),
        "h2d_model_table_bytes": 2 * 384 * length_state_uses,
        "h2d_kernel_scalar_bytes": 16 * (count_calls + length_calls),
        "d2h_bytes": (384 * 2 * 8 * count_state_uses +
                      8 * length_result_stream_uses),
        "d2d_descriptor_bytes": 0,
        "kernel_count": count_calls + length_calls,
        "count_kernel_count": count_calls,
        "length_kernel_count": length_calls,
        "count_cell_symbol_updates":
            int(workload["expected_observed_count_updates"]),
        "length_cell_symbol_updates":
            int(workload["expected_observed_length_updates"]),
        "pack_calls": 1,
        "subset_calls": count_calls + length_calls,
        "to_host_calls": count_calls + length_calls,
        "resource_preflight_calls": 1,
    }
    expected_delta["h2d_bytes"] = sum(expected_delta[name] for name in (
        "h2d_payload_bytes", "h2d_root_descriptor_bytes",
        "h2d_subset_descriptor_bytes", "h2d_launch_descriptor_bytes",
        "h2d_model_table_bytes", "h2d_kernel_scalar_bytes"))
    expected_delta["device_output_allocation_bytes"] = expected_delta["d2h_bytes"]
    for name, expected in expected_delta.items():
        require(after[name] - before[name] == expected,
                f"primary telemetry exact delta: {name}")
    require((after["count_cell_symbol_updates"] -
             before["count_cell_symbol_updates"] +
             after["length_cell_symbol_updates"] -
             before["length_cell_symbol_updates"]) ==
            workload["expected_observed_cuda_updates"],
            "primary observed CUDA update conservation")
    require(workload["exact_primary_updates"] -
            workload["expected_observed_cuda_updates"] ==
            workload["full_symbols"],
            "admission/observed final host-score disclosure")
    for name in TELEMETRY_INTEGER_FIELDS:
        if name not in expected_delta:
            require(after[name] >= before[name],
                    f"primary telemetry monotone cumulative field: {name}")
    return {
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "device_uuid": review["device_uuid"],
        "admitted_primary_updates": PINNED_PRIMARY_UPDATES,
        "observed_primary_count_updates":
            expected_delta["count_cell_symbol_updates"],
        "observed_primary_length_updates":
            expected_delta["length_cell_symbol_updates"],
        "observed_primary_cuda_updates":
            workload["expected_observed_cuda_updates"],
        "admission_minus_observed_host_scoring_symbols":
            workload["admission_minus_observed_cuda_updates"],
        "primary_count_kernel_count": count_calls,
        "primary_length_kernel_count": length_calls,
        "primary_kernel_count": count_calls + length_calls,
        "primary_telemetry_trace": trace,
        "exact_transfer_delta": expected_delta,
    }


def fraction_from_record(record: Any, label: str) -> Fraction:
    exact_fields(record, {"numerator", "denominator", "exact", "float"}, label)
    numerator = exact_int(record["numerator"], f"{label}.numerator",
                          -(1 << 63), (1 << 63) - 1)
    denominator = exact_int(record["denominator"], f"{label}.denominator",
                            1, (1 << 63) - 1)
    value = Fraction(numerator, denominator)
    require(record["exact"] == f"{value.numerator}/{value.denominator}",
            f"{label}: exact text")
    require_float_equal(record["float"], float(value), f"{label}: float")
    return value


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {"numerator": value.numerator, "denominator": value.denominator,
            "exact": f"{value.numerator}/{value.denominator}",
            "float": float(value)}


def _ic_align(value: int, alignment: int) -> int:
    exact_int(value, "independent container alignment value", 0,
              MAX_CONTAINER_BYTES)
    exact_int(alignment, "independent container alignment", 1,
              IC_PAGE_BYTES)
    return ((value + alignment - 1) // alignment) * alignment


def _ic_owner_ordinals(owner_set: bytes, experts: int) -> tuple[int, ...]:
    require(isinstance(owner_set, bytes) and len(owner_set) == IC_OWNER_SET_BYTES,
            "independent owner-set geometry")
    exact_int(experts, "independent expert count", 1, 256)
    used = (experts + 7) // 8
    require(not any(owner_set[used:]), "independent owner-set high bytes")
    if experts & 7:
        require(not (owner_set[used - 1] &
                     ~((1 << (experts & 7)) - 1)),
                "independent owner-set high bits")
    owners = tuple(index for index in range(experts)
                   if owner_set[index >> 3] & (1 << (index & 7)))
    require(bool(owners), "independent nonempty owner set")
    return owners


def _ic_all_owners(experts: int) -> bytes:
    raw = bytearray(IC_OWNER_SET_BYTES)
    for expert in range(experts):
        raw[expert >> 3] |= 1 << (expert & 7)
    return bytes(raw)


def _ic_role(raw: bytes) -> str:
    require(len(raw) == 32, "independent role geometry")
    end = raw.find(b"\x00")
    require(end > 0 and not any(raw[end:]), "independent canonical role bytes")
    try:
        role = raw[:end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise AuditError("independent role ASCII") from exc
    require(role in {"gate", "up", "down", "mixed"},
            "independent role value")
    require(raw == role.encode("ascii") + bytes(32 - len(role)),
            "independent role canonical padding")
    return role


def _ic_zero(raw: bytes, begin: int, end: int, label: str) -> None:
    require(0 <= begin <= end <= len(raw), f"{label}: range")
    require(not any(raw[begin:end]), f"{label}: nonzero padding")


def _ic_interval(begin: int, end: int, kind: str, owner_set: bytes,
                 *, padding: bool) -> dict[str, Any]:
    require(0 <= begin < end <= MAX_CONTAINER_BYTES,
            "independent ledger interval")
    return {"begin": begin, "end": end, "bytes": end - begin,
            "kind": kind, "owner_set": owner_set, "padding": padding}


def _ic_read_summary(ranges: Sequence[tuple[int, int]]) -> dict[str, Any]:
    requested = 0
    normalized: list[tuple[int, int]] = []
    pages: set[int] = set()
    for begin, end in ranges:
        require(type(begin) is int and type(end) is int and
                0 <= begin <= end, "independent routed range")
        normalized.append((begin, end))
        requested += end - begin
        if end > begin:
            pages.update(range(begin // IC_PAGE_BYTES,
                               (end - 1) // IC_PAGE_BYTES + 1))
    unique = 0
    nonempty = sorted((begin, end) for begin, end in normalized if end > begin)
    if nonempty:
        union_begin, union_end = nonempty[0]
        for begin, end in nonempty[1:]:
            if begin > union_end:
                unique += union_end - union_begin
                union_begin, union_end = begin, end
            else:
                union_end = max(union_end, end)
        unique += union_end - union_begin
    return {
        "ranges": normalized,
        "read_request_count": len(normalized),
        "requested_bytes_with_repetition": requested,
        "unique_requested_bytes": unique,
        "overlap_bytes_requested_again": requested - unique,
        "touched_page_indices": sorted(pages),
        "touched_page_bytes": len(pages) * IC_PAGE_BYTES,
    }


def independent_parse_container(raw: bytes, label: str) -> dict[str, Any]:
    """Parse and ledger UWFCV8 v4 without importing sealed-v8 parser code."""
    require(isinstance(raw, bytes) and
            IC_HEADER_BYTES <= len(raw) <= MAX_CONTAINER_BYTES,
            f"{label}: independent byte envelope")
    header_raw = raw[:IC_HEADER_BYTES]
    fields = struct.unpack_from("<8sHHIIQIIHHII", header_raw, 0)
    (magic, version, header_bytes, page_bytes, flags, weights, experts,
     streams, owner_bytes, directory_record_bytes, region_count,
     reserved) = fields
    require((magic, version, header_bytes, page_bytes, flags, owner_bytes,
             directory_record_bytes, reserved) ==
            (IC_MAGIC, IC_VERSION, IC_HEADER_BYTES, IC_PAGE_BYTES, 0,
             IC_OWNER_SET_BYTES, IC_DIRECTORY_RECORD_BYTES, 0),
            f"{label}: independent header constants")
    weights = exact_int(weights, f"{label}.weights", 1, 1 << 50)
    experts = exact_int(experts, f"{label}.experts", 1, 256)
    streams = exact_int(streams, f"{label}.streams", 1, 65_536)
    region_count = exact_int(region_count, f"{label}.regions", 1, streams)
    baseline_bytes, audited_mse = struct.unpack_from("<Qd", header_raw, 48)
    baseline_bytes = exact_int(baseline_bytes, f"{label}.baseline_bytes", 1,
                               MAX_CONTAINER_BYTES)
    require(math.isfinite(audited_mse) and audited_mse > 0.0,
            f"{label}: finite positive audited MSE")
    section_values = struct.unpack_from("<QQQQQQQQQQ", header_raw, 64)
    (semantic_offset, semantic_bytes, immutable_offset, immutable_bytes,
     model_offset, model_bytes, directory_offset, directory_bytes,
     shared_bytes, total_bytes) = section_values
    rate_num, rate_den = struct.unpack_from("<QQ", header_raw, 144)
    require((rate_num, rate_den) == (43, 20),
            f"{label}: frozen rate floor")
    require(total_bytes == len(raw), f"{label}: header total bytes")
    require(directory_bytes == streams * IC_DIRECTORY_RECORD_BYTES,
            f"{label}: directory byte count")
    require(1 <= semantic_bytes <= 1 << 26 and
            0 <= immutable_bytes <= 1 << 26 and
            1 <= model_bytes <= 1 << 26,
            f"{label}: global section bounds")
    expected_semantic = IC_HEADER_BYTES
    expected_immutable = _ic_align(expected_semantic + semantic_bytes, 64)
    expected_model = _ic_align(expected_immutable + immutable_bytes,
                               IC_PAGE_BYTES)
    expected_directory = _ic_align(expected_model + model_bytes,
                                   IC_PAGE_BYTES)
    expected_shared = _ic_align(expected_directory + directory_bytes,
                                IC_PAGE_BYTES)
    require((semantic_offset, immutable_offset, model_offset,
             directory_offset, shared_bytes) ==
            (expected_semantic, expected_immutable, expected_model,
             expected_directory, expected_shared),
            f"{label}: canonical shared placement")
    require(shared_bytes < total_bytes and shared_bytes % IC_PAGE_BYTES == 0,
            f"{label}: shared/region envelope")

    clean = bytearray(header_raw)
    observed_seal = bytes(clean[IC_HEADER_SEAL_BEGIN:IC_HEADER_SEAL_END])
    observed_crc = struct.unpack_from("<I", clean, IC_CRC_OFFSET)[0]
    struct.pack_into("<I", clean, IC_CRC_OFFSET, 0)
    require(observed_crc == (zlib.crc32(clean) & 0xFFFFFFFF),
            f"{label}: independent header CRC")
    clean[IC_HEADER_SEAL_BEGIN:IC_HEADER_SEAL_END] = bytes(32)
    require(observed_seal == hashlib.sha256(clean).digest(),
            f"{label}: independent header SHA seal")
    _ic_zero(header_raw, IC_CRC_OFFSET + 4, IC_HEADER_BYTES,
             f"{label}: header reserved")
    semantic_packet = raw[semantic_offset:semantic_offset + semantic_bytes]
    immutable_state = raw[immutable_offset:immutable_offset + immutable_bytes]
    model_packet = raw[model_offset:model_offset + model_bytes]
    directory_blob = raw[directory_offset:directory_offset + directory_bytes]
    expected_section_hashes = (
        header_raw[256:288], header_raw[288:320], header_raw[320:352],
        header_raw[352:384], header_raw[384:416],
    )
    observed_section_hashes = (
        hashlib.sha256(semantic_packet).digest(),
        hashlib.sha256(immutable_state).digest(),
        hashlib.sha256(model_packet).digest(),
        hashlib.sha256(directory_blob).digest(),
        hashlib.sha256(raw[IC_HEADER_BYTES:]).digest(),
    )
    require(observed_section_hashes == expected_section_hashes,
            f"{label}: independent global/body digests")
    _ic_zero(raw, semantic_offset + semantic_bytes, immutable_offset,
             f"{label}: semantic alignment")
    _ic_zero(raw, immutable_offset + immutable_bytes, model_offset,
             f"{label}: immutable alignment")
    _ic_zero(raw, model_offset + model_bytes, directory_offset,
             f"{label}: model alignment")
    _ic_zero(raw, directory_offset + directory_bytes, shared_bytes,
             f"{label}: directory alignment")
    binding_hashes = {
        name: header_raw[IC_BINDINGS_BEGIN + 32 * index:
                         IC_BINDINGS_BEGIN + 32 * (index + 1)].hex()
        for index, name in enumerate(IC_BINDINGS)
    }
    all_owners = _ic_all_owners(experts)
    ledger = [_ic_interval(0, IC_HEADER_BYTES, "container_header",
                           all_owners, padding=False)]
    ledger.append(_ic_interval(semantic_offset, semantic_offset + semantic_bytes,
                               "universal_semantics", all_owners,
                               padding=False))
    if immutable_offset > semantic_offset + semantic_bytes:
        ledger.append(_ic_interval(semantic_offset + semantic_bytes,
                                   immutable_offset,
                                   "semantic_alignment_padding", all_owners,
                                   padding=True))
    if immutable_bytes:
        ledger.append(_ic_interval(immutable_offset,
                                   immutable_offset + immutable_bytes,
                                   "evaluation_plugin_immutable_state",
                                   all_owners, padding=False))
    if model_offset > immutable_offset + immutable_bytes:
        ledger.append(_ic_interval(immutable_offset + immutable_bytes,
                                   model_offset, "immutable_alignment_padding",
                                   all_owners, padding=True))
    ledger.append(_ic_interval(model_offset, model_offset + model_bytes,
                               "serialized_unifilar_model", all_owners,
                               padding=False))
    if directory_offset > model_offset + model_bytes:
        ledger.append(_ic_interval(model_offset + model_bytes, directory_offset,
                                   "model_alignment_padding", all_owners,
                                   padding=True))
    ledger.append(_ic_interval(directory_offset,
                               directory_offset + directory_bytes,
                               "stream_directory", all_owners, padding=False))
    if shared_bytes > directory_offset + directory_bytes:
        ledger.append(_ic_interval(directory_offset + directory_bytes,
                                   shared_bytes, "directory_alignment_padding",
                                   all_owners, padding=True))

    directory: list[dict[str, Any]] = []
    for index in range(streams):
        packet = directory_blob[index * IC_DIRECTORY_RECORD_BYTES:
                                (index + 1) * IC_DIRECTORY_RECORD_BYTES]
        values = struct.unpack_from("<8sIIHHIQQQQQQQQQQQd", packet, 0)
        (row_magic, ordinal, region_ordinal, profile_q, contribution_count,
         row_reserved, symbols, logical_bits, payload_offset, payload_bytes,
         region_offset, region_bytes, frame_offset, frame_bytes,
         source_weights, group_rows, group_cols, decoder_scale) = values
        require(row_magic == IC_DIRECTORY_MAGIC and row_reserved == 0 and
                ordinal == index, f"{label}: directory[{index}] constants/order")
        exact_int(region_ordinal, f"{label}.directory[{index}].region", 0,
                  region_count - 1)
        exact_int(profile_q, f"{label}.directory[{index}].profile", 0, 65535)
        exact_int(contribution_count,
                  f"{label}.directory[{index}].contributions", 1,
                  3 * experts)
        exact_int(symbols, f"{label}.directory[{index}].symbols", 1,
                  1 << 54)
        exact_int(logical_bits, f"{label}.directory[{index}].logical", 1,
                  1 << 56)
        require(payload_bytes == (logical_bits + 7) // 8,
                f"{label}: directory[{index}] payload/logical")
        exact_int(group_rows, f"{label}.directory[{index}].group_rows", 1,
                  1 << 24)
        exact_int(group_cols, f"{label}.directory[{index}].group_cols", 1,
                  1 << 24)
        exact_int(frame_bytes, f"{label}.directory[{index}].frame_bytes",
                  IC_FRAME_HEADER_BYTES, 1 << 26)
        require(group_rows >= 1 and group_cols >= 1 and
                source_weights == group_rows * group_cols,
                f"{label}: directory[{index}] source shape")
        require(math.isfinite(decoder_scale) and decoder_scale > 0.0,
                f"{label}: directory[{index}] decoder scale")
        owner_set = bytes(packet[120:152])
        owners = _ic_owner_ordinals(owner_set, experts)
        require(len(owners) <= contribution_count,
                f"{label}: directory[{index}] owner/contribution count")
        require(packet[248:256] == hashlib.sha256(packet[:248]).digest()[:8],
                f"{label}: directory[{index}] seal")
        require(region_offset >= shared_bytes and
                region_offset % IC_PAGE_BYTES == 0 and
                region_bytes >= IC_PAGE_BYTES and
                region_bytes % IC_PAGE_BYTES == 0 and
                frame_offset % 64 == 0 and frame_bytes % 64 == 0,
                f"{label}: directory[{index}] placement alignment")
        region_end = region_offset + region_bytes
        frame_end = frame_offset + frame_bytes
        payload_end = payload_offset + payload_bytes
        require(region_offset + IC_REGION_HEADER_BYTES <= frame_offset <
                frame_end <= region_end <= total_bytes and
                frame_offset + IC_FRAME_HEADER_BYTES <= payload_offset <
                payload_end <= frame_end,
                f"{label}: directory[{index}] nested ranges")
        directory.append({
            "ordinal": ordinal, "region_ordinal": region_ordinal,
            "profile_q": profile_q, "contribution_count": contribution_count,
            "symbols": symbols, "logical_bits": logical_bits,
            "payload_offset": payload_offset, "payload_bytes": payload_bytes,
            "region_offset": region_offset, "region_bytes": region_bytes,
            "frame_offset": frame_offset, "frame_bytes": frame_bytes,
            "source_weights": source_weights, "group_rows": group_rows,
            "group_cols": group_cols, "decoder_scale": decoder_scale,
            "owner_set": owner_set, "owner_set_hex": owner_set.hex(),
            "owners": owners, "source_digest": packet[152:184].hex(),
            "payload_sha256": packet[184:216].hex(),
            "role": _ic_role(bytes(packet[216:248])),
        })

    grouped: dict[int, dict[str, Any]] = {}
    union = bytearray(IC_OWNER_SET_BYTES)
    seen_owner_sets: set[bytes] = set()
    for row in directory:
        identity = (row["region_offset"], row["region_bytes"], row["owner_set"])
        region = grouped.setdefault(row["region_ordinal"],
                                    {"identity": identity, "rows": []})
        require(region["identity"] == identity,
                f"{label}: consistent region identity")
        region["rows"].append(row)
        seen_owner_sets.add(row["owner_set"])
        for offset, value in enumerate(row["owner_set"]):
            union[offset] |= value
    require(sorted(grouped) == list(range(region_count)) and
            len(seen_owner_sets) == region_count and bytes(union) == all_owners,
            f"{label}: complete unique region/owner coverage")
    owner_order = [grouped[index]["identity"][2]
                   for index in range(region_count)]
    require(owner_order == sorted(
        owner_order,
        key=lambda value: (len(_ic_owner_ordinals(value, experts)) != 1,
                           _ic_owner_ordinals(value, experts))),
        f"{label}: canonical region owner order")

    parsed_regions: list[dict[str, Any]] = []
    cursor = shared_bytes
    for region_ordinal in range(region_count):
        region = grouped[region_ordinal]
        region_offset, region_bytes, owner_set = region["identity"]
        require(region_offset == cursor,
                f"{label}: region[{region_ordinal}] contiguous placement")
        packet_header = raw[region_offset:
                            region_offset + IC_REGION_HEADER_BYTES]
        values = struct.unpack_from("<8sIIHHIQQQ", packet_header, 0)
        (region_magic, observed_ordinal, stream_count, observed_owner_bytes,
         reserved_h, reserved_i, header_region_bytes, content_bytes,
         frame_area_bytes) = values
        rows = sorted(region["rows"], key=lambda row: row["ordinal"])
        require((region_magic, observed_ordinal, stream_count,
                 observed_owner_bytes, reserved_h, reserved_i,
                 header_region_bytes) ==
                (IC_REGION_MAGIC, region_ordinal, len(rows),
                 IC_OWNER_SET_BYTES, 0, 0, region_bytes),
                f"{label}: region[{region_ordinal}] constants")
        require(packet_header[48:80] == owner_set and
                packet_header[112:144] ==
                hashlib.sha256(packet_header[:112]).digest(),
                f"{label}: region[{region_ordinal}] owner/seal")
        _ic_zero(packet_header, 144, IC_REGION_HEADER_BYTES,
                 f"{label}: region[{region_ordinal}] reserved")
        require(content_bytes == IC_REGION_HEADER_BYTES + frame_area_bytes and
                IC_FRAME_HEADER_BYTES <= frame_area_bytes <= region_bytes,
                f"{label}: region[{region_ordinal}] content geometry")
        frame_area_begin = region_offset + IC_REGION_HEADER_BYTES
        require(hashlib.sha256(raw[frame_area_begin:
                                   frame_area_begin + frame_area_bytes]).digest()
                == packet_header[80:112],
                f"{label}: region[{region_ordinal}] frame-area digest")
        ledger.append(_ic_interval(region_offset, frame_area_begin,
                                   "region_header", owner_set, padding=False))
        frame_cursor = frame_area_begin
        for row in rows:
            ordinal = row["ordinal"]
            require(row["owner_set"] == owner_set and
                    row["frame_offset"] == frame_cursor,
                    f"{label}: frame[{ordinal}] owner/contiguity")
            frame_end = frame_cursor + row["frame_bytes"]
            frame = raw[frame_cursor:frame_end]
            require(len(frame) == row["frame_bytes"],
                    f"{label}: frame[{ordinal}] bounded read")
            frame_header = frame[:IC_FRAME_HEADER_BYTES]
            frame_values = struct.unpack_from(
                "<8sIIHHIQQQQQQdQQ", frame_header, 0)
            (frame_magic, frame_ordinal, frame_region, frame_profile,
             frame_contributions, frame_reserved, frame_symbols,
             frame_logical, frame_payload_bytes, frame_source_weights,
             frame_rows, frame_cols, frame_scale, metadata_bytes,
             observed_frame_bytes) = frame_values
            require((frame_magic, frame_ordinal, frame_region, frame_profile,
                     frame_contributions, frame_reserved, frame_symbols,
                     frame_logical, frame_payload_bytes, frame_source_weights,
                     frame_rows, frame_cols) ==
                    (IC_FRAME_MAGIC, ordinal, region_ordinal, row["profile_q"],
                     row["contribution_count"], 0, row["symbols"],
                     row["logical_bits"], row["payload_bytes"],
                     row["source_weights"], row["group_rows"],
                     row["group_cols"]),
                    f"{label}: frame[{ordinal}] repeated scalars")
            require(_float_bits(frame_scale) == _float_bits(row["decoder_scale"]),
                    f"{label}: frame[{ordinal}] scale bits")
            expected_metadata = _ic_align(
                IC_FRAME_HEADER_BYTES +
                IC_CONTRIBUTION_RECORD_BYTES * row["contribution_count"], 64)
            require(metadata_bytes == expected_metadata and
                    observed_frame_bytes == row["frame_bytes"],
                    f"{label}: frame[{ordinal}] metadata/bytes")
            require(frame_header[96:128] == owner_set and
                    frame_header[128:160].hex() == row["source_digest"] and
                    frame_header[160:192].hex() == row["payload_sha256"] and
                    _ic_role(frame_header[192:224]) == row["role"] and
                    frame_header[224:256] ==
                    hashlib.sha256(frame_header[:224]).digest(),
                    f"{label}: frame[{ordinal}] repeated hashes/role/seal")
            contributions = []
            previous: tuple[int, int, int] | None = None
            contribution_total = 0
            contribution_cursor = IC_FRAME_HEADER_BYTES
            for contribution_index in range(row["contribution_count"]):
                expert, role_id, source_offset, weight_count = struct.unpack_from(
                    "<IIQQ", frame, contribution_cursor)
                exact_int(expert,
                          f"{label}.frame[{ordinal}].contribution.expert", 0,
                          experts - 1)
                exact_int(role_id,
                          f"{label}.frame[{ordinal}].contribution.role", 1, 3)
                exact_int(source_offset,
                          f"{label}.frame[{ordinal}].contribution.offset", 0,
                          (1 << 50) - 1)
                exact_int(weight_count,
                          f"{label}.frame[{ordinal}].contribution.weights", 1,
                          1 << 50)
                require(source_offset <= (1 << 50) - weight_count,
                        f"{label}: frame[{ordinal}] contribution interval")
                key = (expert, role_id, source_offset)
                require(previous is None or key > previous,
                        f"{label}: frame[{ordinal}] canonical contributions")
                previous = key
                contribution_total += weight_count
                contributions.append({
                    "expert": expert,
                    "role": ("gate", "up", "down")[role_id - 1],
                    "source_offset": source_offset,
                    "weight_count": weight_count,
                })
                contribution_cursor += IC_CONTRIBUTION_RECORD_BYTES
            require(tuple(sorted({item["expert"] for item in contributions})) ==
                    row["owners"] and contribution_total == row["source_weights"],
                    f"{label}: frame[{ordinal}] owner/source conservation")
            _ic_zero(frame, contribution_cursor, metadata_bytes,
                     f"{label}: frame[{ordinal}] metadata padding")
            payload_end = metadata_bytes + row["payload_bytes"]
            payload = frame[metadata_bytes:payload_end]
            require(hashlib.sha256(payload).hexdigest() == row["payload_sha256"],
                    f"{label}: frame[{ordinal}] payload hash")
            if row["logical_bits"] & 7:
                require(not (payload[-1] &
                             ((1 << (8 - (row["logical_bits"] & 7))) - 1)),
                        f"{label}: frame[{ordinal}] terminal arithmetic padding")
            _ic_zero(frame, payload_end, len(frame),
                     f"{label}: frame[{ordinal}] alignment padding")
            require(frame_cursor + metadata_bytes == row["payload_offset"],
                    f"{label}: frame[{ordinal}] payload offset")
            row["owner_contributions"] = contributions
            row["payload"] = payload
            row["metadata_bytes"] = metadata_bytes
            contribution_end = (frame_cursor + IC_FRAME_HEADER_BYTES +
                                IC_CONTRIBUTION_RECORD_BYTES *
                                row["contribution_count"])
            metadata_end = frame_cursor + metadata_bytes
            absolute_payload_end = row["payload_offset"] + row["payload_bytes"]
            ledger.append(_ic_interval(frame_cursor,
                                       frame_cursor + IC_FRAME_HEADER_BYTES,
                                       "frame_header", owner_set,
                                       padding=False))
            ledger.append(_ic_interval(frame_cursor + IC_FRAME_HEADER_BYTES,
                                       contribution_end,
                                       "owner_contribution_records", owner_set,
                                       padding=False))
            if metadata_end > contribution_end:
                ledger.append(_ic_interval(contribution_end, metadata_end,
                                           "frame_metadata_padding", owner_set,
                                           padding=True))
            ledger.append(_ic_interval(row["payload_offset"],
                                       absolute_payload_end,
                                       "arithmetic_payload", owner_set,
                                       padding=False))
            if frame_end > absolute_payload_end:
                ledger.append(_ic_interval(absolute_payload_end, frame_end,
                                           "frame_alignment_padding", owner_set,
                                           padding=True))
            frame_cursor = frame_end
        require(frame_cursor == frame_area_begin + frame_area_bytes ==
                region_offset + content_bytes,
                f"{label}: region[{region_ordinal}] frame coverage")
        region_end = region_offset + region_bytes
        if region_end > frame_cursor:
            _ic_zero(raw, frame_cursor, region_end,
                     f"{label}: region[{region_ordinal}] rate padding")
            ledger.append(_ic_interval(frame_cursor, region_end,
                                       "owner_region_rate_padding", owner_set,
                                       padding=True))
        parsed_regions.append({
            "ordinal": region_ordinal, "offset": region_offset,
            "bytes": region_bytes, "content_bytes": content_bytes,
            "owner_set": owner_set, "owner_set_hex": owner_set.hex(),
            "owners": _ic_owner_ordinals(owner_set, experts),
            "row_ordinals": [row["ordinal"] for row in rows],
        })
        cursor = region_end
    base_lengths = [_ic_align(row["content_bytes"], IC_PAGE_BYTES)
                    for row in parsed_regions]
    base_total = shared_bytes + sum(base_lengths)
    minimum_total = (weights * 43 + 8 * 20 - 1) // (8 * 20)
    padding_pages = (max(0, minimum_total - base_total) + IC_PAGE_BYTES - 1) // IC_PAGE_BYTES
    pages_each, leading_extra = divmod(padding_pages, len(base_lengths))
    canonical_lengths = [
        length + IC_PAGE_BYTES * (pages_each + (1 if index < leading_extra else 0))
        for index, length in enumerate(base_lengths)
    ]
    require([row["bytes"] for row in parsed_regions] == canonical_lengths,
            f"{label}: canonical rate-padding distribution")
    require(cursor == len(raw), f"{label}: exact EOF coverage")
    require(sum(row["source_weights"] for row in directory) == weights,
            f"{label}: independent directory source-weight conservation")
    expected_begin = 0
    for entry in ledger:
        require(entry["begin"] == expected_begin,
                f"{label}: independent ledger overlap/hole")
        expected_begin = entry["end"]
    require(expected_begin == len(raw), f"{label}: independent ledger EOF")

    attributed_total = [Fraction(0, 1) for _ in range(experts)]
    attributed_nonpadding = [Fraction(0, 1) for _ in range(experts)]
    allocation = Fraction(0, 1)
    for entry in ledger:
        owners = _ic_owner_ordinals(entry["owner_set"], experts)
        share = Fraction(entry["bytes"], len(owners))
        for owner in owners:
            attributed_total[owner] += share
            if not entry["padding"]:
                attributed_nonpadding[owner] += share
            allocation += share
    require(allocation == len(raw),
            f"{label}: independent owner allocation conservation")
    routed_rows = []
    maximum_cold = Fraction(0, 1)
    for expert in range(experts):
        selected = [row for row in directory if expert in row["owners"]]
        require(bool(selected), f"{label}: expert[{expert}] has stream")
        ranges = [
            (0, IC_HEADER_BYTES),
            (semantic_offset, semantic_offset + semantic_bytes),
            (immutable_offset, immutable_offset + immutable_bytes),
            (model_offset, model_offset + model_bytes),
            (directory_offset, directory_offset + directory_bytes),
        ]
        selected_regions: dict[int, list[dict[str, Any]]] = {}
        for row in selected:
            selected_regions.setdefault(row["region_ordinal"], []).append(row)
        for region_ordinal in sorted(selected_regions):
            region_rows = sorted(selected_regions[region_ordinal],
                                 key=lambda row: row["frame_offset"])
            first = region_rows[0]
            ranges.append((first["region_offset"],
                           first["region_offset"] + IC_REGION_HEADER_BYTES))
            frame_cursor = first["region_offset"] + IC_REGION_HEADER_BYTES
            for row in region_rows:
                require(row["frame_offset"] == frame_cursor,
                        f"{label}: expert[{expert}] routed frame contiguity")
                ranges.append((row["frame_offset"],
                               row["frame_offset"] + row["frame_bytes"]))
                frame_cursor += row["frame_bytes"]
            region = parsed_regions[region_ordinal]
            require(frame_cursor == region["offset"] + region["content_bytes"],
                    f"{label}: expert[{expert}] routed region coverage")
        summary = _ic_read_summary(ranges)
        strict = max(Fraction(summary["touched_page_bytes"], 1) /
                     attributed_total[expert],
                     Fraction(summary["touched_page_bytes"], 1) /
                     attributed_nonpadding[expert])
        maximum_cold = max(maximum_cold, strict)
        source_weights_for_expert = sum(
            item["weight_count"] for row in selected
            for item in row["owner_contributions"]
            if item["expert"] == expert)
        routed_rows.append({
            "expert_ordinal": expert,
            **summary,
            "routed_modeled_symbols": sum(row["symbols"] for row in selected),
            "expert_source_weights": source_weights_for_expert,
            "attributable_total_physical_bytes": attributed_total[expert],
            "attributable_nonpadding_decodable_bytes":
                attributed_nonpadding[expert],
            "strict_cold_amplification": strict,
        })
    return {
        "container_sha256": sha256(raw),
        "weights": weights, "experts": experts, "streams": streams,
        "region_count": region_count, "baseline_object_bytes": baseline_bytes,
        "audited_relative_mse": audited_mse,
        "semantic_offset": semantic_offset, "semantic_bytes": semantic_bytes,
        "immutable_offset": immutable_offset, "immutable_bytes": immutable_bytes,
        "model_offset": model_offset, "model_bytes": model_bytes,
        "directory_offset": directory_offset, "directory_bytes": directory_bytes,
        "shared_bytes": shared_bytes, "total_bytes": total_bytes,
        "minimum_rate_numerator": rate_num,
        "minimum_rate_denominator": rate_den,
        "baseline_artifact_sha256": header_raw[160:192].hex(),
        "reconstruction_sha256": header_raw[192:224].hex(),
        "audit_binding_sha256": header_raw[224:256].hex(),
        "binding_hashes": binding_hashes,
        "semantic_packet": semantic_packet, "immutable_state": immutable_state,
        "model_packet": model_packet, "directory_blob": directory_blob,
        "directory": directory, "regions": parsed_regions,
        "byte_ledger": ledger, "ownership_allocated_bytes": allocation,
        "routed": routed_rows, "maximum_strict_cold": maximum_cold,
    }


def audit_literal_container(
    modules: Mapping[str, Any], held: HeldRegularAt, *, semantic_decode: bool,
    label: str,
) -> dict[str, Any]:
    codec = modules["codec"]
    common = modules["common"]
    semantic = modules["semantic"]
    adapter = modules["adapter"]
    raw = held.data
    independently_parsed = independent_parse_container(raw, label)
    parsed = codec.parse_container(common, semantic, raw)
    require(isinstance(parsed, dict) and bytes(parsed.get("raw", b"")) == raw,
            f"{label}: parser literal-byte rebind")
    for field in (
        "weights", "experts", "streams", "region_count",
        "baseline_object_bytes", "audited_relative_mse", "semantic_offset",
        "semantic_bytes", "immutable_offset", "immutable_bytes", "model_offset",
        "model_bytes", "directory_offset", "directory_bytes", "shared_bytes",
        "total_bytes", "minimum_rate_numerator", "minimum_rate_denominator",
        "baseline_artifact_sha256", "reconstruction_sha256",
        "audit_binding_sha256", "binding_hashes",
    ):
        if field == "audited_relative_mse":
            require_float_equal(parsed[field], independently_parsed[field],
                                f"{label}: independent/producer {field}")
        else:
            require_deep_equal(parsed[field], independently_parsed[field],
                               f"{label}: independent/producer {field}")
    for field in ("semantic_packet", "immutable_state", "model_packet",
                  "directory_blob"):
        require(bytes(parsed[field]) == bytes(independently_parsed[field]),
                f"{label}: independent/producer {field}")
    require(len(parsed["directory"]) == len(independently_parsed["directory"]),
            f"{label}: independent/producer directory count")
    scalar_fields = (
        "ordinal", "region_ordinal", "profile_q", "contribution_count",
        "symbols", "logical_bits", "payload_offset", "payload_bytes",
        "region_offset", "region_bytes", "frame_offset", "frame_bytes",
        "source_weights", "group_rows", "group_cols", "owner_set_hex",
        "owners", "source_digest", "payload_sha256", "role",
    )
    for ordinal, (producer_row, independent_row) in enumerate(zip(
            parsed["directory"], independently_parsed["directory"],
            strict=True)):
        for field in scalar_fields:
            require_deep_equal(producer_row[field], independent_row[field],
                               f"{label}: directory[{ordinal}].{field}")
        require(_float_bits(producer_row["decoder_scale"]) ==
                _float_bits(independent_row["decoder_scale"]),
                f"{label}: directory[{ordinal}].decoder_scale")
        require(bytes(producer_row["owner_set"]) ==
                bytes(independent_row["owner_set"]) and
                bytes(producer_row["payload"]) ==
                bytes(independent_row["payload"]),
                f"{label}: directory[{ordinal}] owner/payload bytes")
        require_deep_equal([dict(row) for row in
                            producer_row["owner_contributions"]],
                           independent_row["owner_contributions"],
                           f"{label}: directory[{ordinal}] contributions")
    rebuilt = codec.canonical_rebuild(common, semantic, parsed)
    require(rebuilt == raw, f"{label}: canonical rebuild")
    if semantic_decode:
        standalone = adapter.decode_new_container(parsed)
        descriptor = codec.AuthenticatedDescriptorSource(held.fd, held.sha256)
        try:
            metrics = codec.physical_metrics(
                common, semantic, parsed,
                routed_descriptor_source=descriptor,
                externally_authenticated_container_sha256=held.sha256,
                routed_decoder=adapter.new_routed_decoder(),
            )
        finally:
            descriptor.close()
        require(standalone["all_payloads_canonically_reencoded"] is True,
                f"{label}: causal canonical re-encode")
        require(standalone["reconstruction"]["groups_covered_once"] is True,
                f"{label}: exact group/role reconstruction coverage")
    else:
        standalone = None
        metrics = codec.physical_metrics(common, semantic, parsed)
    return {"parsed": parsed, "independent": independently_parsed,
            "standalone": standalone, "metrics": metrics,
            "canonical_rebuild_sha256": sha256(rebuilt)}


def verify_bandwidth_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows = metrics["experts"]
    require(isinstance(rows, list) and rows, "physical expert rows")
    maximum = Fraction(0, 1)
    maximum_repeated = Fraction(0, 1)
    for ordinal, row in enumerate(rows):
        require(row["expert_ordinal"] == ordinal, "physical expert order")
        total = fraction_from_record(row["attributable_total_physical_bytes"],
                                     f"expert[{ordinal}] total")
        nonpadding = fraction_from_record(
            row["attributable_nonpadding_decodable_bytes"],
            f"expert[{ordinal}] nonpadding")
        require(total > 0 and nonpadding > 0, "physical positive attribution")
        cold = exact_int(row["touched_page_bytes"], f"expert[{ordinal}] pages")
        strict = max(Fraction(cold, 1) / total,
                     Fraction(cold, 1) / nonpadding)
        require(fraction_from_record(row["strict_cold_amplification"],
                                     f"expert[{ordinal}] strict cold") == strict,
                f"expert[{ordinal}] cold amplification")
        repeated_bytes = exact_int(
            row["instrumented_routed_requested_bytes_with_repetition"],
            f"expert[{ordinal}] requested bytes")
        repeated = max(Fraction(repeated_bytes, 1) / total,
                       Fraction(repeated_bytes, 1) / nonpadding)
        maximum = max(maximum, strict)
        maximum_repeated = max(maximum_repeated, repeated)
        require(row["causal_decode_reencode_reconstruction"] is not None,
                f"expert[{ordinal}] routed causal decode evidence")
    require(fraction_from_record(metrics["maximum_strict_cold_read_amplification"],
                                 "maximum strict cold") == maximum,
            "maximum strict cold recomputation")
    require(metrics["passes_cold_read_below_2x"] is
            (metrics["routed_io_authoritative_descriptor_backed"] is True and
             maximum < 2), "cold gate recomputation")
    return {
        "maximum_strict_cold": fraction_record(maximum),
        "maximum_requested_with_repetition": fraction_record(maximum_repeated),
        "passes_page_cold_below_2x": maximum < 2,
        "passes_repeated_requested_below_2x": maximum_repeated < 2,
    }


def verify_independent_bandwidth(
    independent: Mapping[str, Any], metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check every routed range/page/allocation against raw-byte grammar."""
    rows = metrics["experts"]
    expected_rows = independent["routed"]
    require(isinstance(rows, list) and len(rows) == len(expected_rows),
            "independent routed expert count")
    maximum_requested = Fraction(0, 1)
    for expert, (row, expected) in enumerate(zip(rows, expected_rows,
                                                 strict=True)):
        require(row["expert_ordinal"] == expected["expert_ordinal"] == expert,
                f"independent route[{expert}] ordinal")
        require_deep_equal(row["touched_page_indices"],
                           expected["touched_page_indices"],
                           f"independent route[{expert}] page indices")
        require(row["touched_page_bytes"] == expected["touched_page_bytes"],
                f"independent route[{expert}] page bytes")
        require_deep_equal(row["instrumented_routed_read_ranges"],
                           [list(value) for value in expected["ranges"]],
                           f"independent route[{expert}] literal read ranges")
        for observed_name, expected_name in (
            ("instrumented_routed_read_request_count", "read_request_count"),
            ("instrumented_routed_requested_bytes_with_repetition",
             "requested_bytes_with_repetition"),
            ("instrumented_routed_unique_requested_bytes",
             "unique_requested_bytes"),
            ("instrumented_routed_overlap_bytes_requested_again",
             "overlap_bytes_requested_again"),
            ("routed_modeled_symbols", "routed_modeled_symbols"),
            ("expert_source_weights", "expert_source_weights"),
        ):
            require(row[observed_name] == expected[expected_name],
                    f"independent route[{expert}] {observed_name}")
        total = fraction_from_record(row["attributable_total_physical_bytes"],
                                     f"independent route[{expert}] total")
        nonpadding = fraction_from_record(
            row["attributable_nonpadding_decodable_bytes"],
            f"independent route[{expert}] nonpadding")
        require(total == expected["attributable_total_physical_bytes"] and
                nonpadding ==
                expected["attributable_nonpadding_decodable_bytes"],
                f"independent route[{expert}] attribution")
        strict = fraction_from_record(row["strict_cold_amplification"],
                                      f"independent route[{expert}] strict")
        require(strict == expected["strict_cold_amplification"],
                f"independent route[{expert}] strict cold")
        repeated = max(
            Fraction(expected["requested_bytes_with_repetition"], 1) / total,
            Fraction(expected["requested_bytes_with_repetition"], 1) /
            nonpadding)
        maximum_requested = max(maximum_requested, repeated)
        decoded = exact_fields(row["causal_decode_reencode_reconstruction"], {
            "expert_ordinal", "decoded_streams",
            "all_payloads_canonically_reencoded",
            "all_three_roles_reconstructed",
            "routed_expert_reconstruction_sha256",
        }, f"independent route[{expert}] causal decode")
        require(decoded["expert_ordinal"] == expert and
                decoded["decoded_streams"] > 0 and
                decoded["all_payloads_canonically_reencoded"] is True and
                decoded["all_three_roles_reconstructed"] is True,
                f"independent route[{expert}] causal decode flags")
        digest(decoded["routed_expert_reconstruction_sha256"],
               f"independent route[{expert}] reconstruction digest")
    maximum = fraction_from_record(
        metrics["maximum_strict_cold_read_amplification"],
        "independent maximum strict cold")
    require(maximum == independent["maximum_strict_cold"],
            "independent maximum strict cold equality")
    require(fraction_from_record(metrics["ownership_allocated_bytes_sum"],
                                 "independent allocation total") ==
            independent["ownership_allocated_bytes"] ==
            independent["total_bytes"],
            "independent ownership allocation total")
    require(metrics["complete_byte_partition_entries"] ==
            len(independent["byte_ledger"]) and
            metrics["complete_byte_partition_exact"] is True,
            "independent complete byte ledger")
    routed_full = exact_fields(metrics["routed_full_reconstruction"], {
        "experts", "full_reconstruction_f64_sha256",
        "matches_container_reconstruction",
    }, "independent routed full reconstruction")
    require(routed_full["experts"] == independent["experts"] and
            routed_full["matches_container_reconstruction"] is True and
            routed_full["full_reconstruction_f64_sha256"] ==
            independent["reconstruction_sha256"],
            "independent routed full reconstruction binding")
    require(metrics["passes_cold_read_below_2x"] is
            (independent["maximum_strict_cold"] < 2),
            "independent cold-read predicate")
    require(metrics["cold_gate_definition"] ==
            "maximum of routed touched-pages/attributable-total and routed touched-pages/attributable-nonpadding",
            "independent cold-gate definition")
    unique_symbols = sum(row["symbols"] for row in independent["directory"])
    routed_symbols = sum(row["routed_modeled_symbols"] for row in expected_rows)
    routed_weights = sum(row["expert_source_weights"] for row in expected_rows)
    require_deep_equal(metrics["modeled_symbol_density"], {
        "unique_directory_modeled_symbols": unique_symbols,
        "source_weights": independent["weights"],
        "unique_directory_modeled_symbols_per_source_weight":
            fraction_record(Fraction(unique_symbols, independent["weights"])),
        "routed_modeled_symbols_sum_across_experts": routed_symbols,
        "routed_source_weights_sum_across_experts": routed_weights,
        "routed_modeled_symbols_per_source_weight_sum_across_experts":
            fraction_record(Fraction(routed_symbols, routed_weights)),
        "shared_stream_symbol_reuse_across_expert_routes":
            routed_symbols - unique_symbols,
    }, "independent modeled-symbol density")
    requested_sum = sum(row["requested_bytes_with_repetition"]
                        for row in expected_rows)
    unique_sum = sum(row["unique_requested_bytes"] for row in expected_rows)
    overlap_sum = sum(row["overlap_bytes_requested_again"]
                      for row in expected_rows)
    page_sum = sum(row["touched_page_bytes"] for row in expected_rows)
    request_count_sum = sum(row["read_request_count"] for row in expected_rows)
    maximum_raw_requested = max(row["requested_bytes_with_repetition"]
                                for row in expected_rows)
    require_deep_equal(metrics["routed_read_request_aggregates"], {
        "read_request_count_sum_across_experts": request_count_sum,
        "requested_bytes_with_repetition_sum_across_experts": requested_sum,
        "unique_requested_bytes_sum_across_experts": unique_sum,
        "overlap_bytes_requested_again_sum_across_experts": overlap_sum,
        "unique_touched_page_bytes_sum_across_experts": page_sum,
        "maximum_requested_bytes_with_repetition_per_expert":
            maximum_raw_requested,
        "mean_requested_bytes_with_repetition_per_expert":
            fraction_record(Fraction(requested_sum, independent["experts"])),
        "frozen_cold_gate_uses_unique_touched_page_bytes_only": True,
    }, "independent routed read aggregates")
    installation_ranges = []
    cursor = 0
    while cursor < independent["total_bytes"]:
        end = min(independent["total_bytes"], cursor + (1 << 20))
        installation_ranges.append((cursor, end))
        cursor = end
    install_summary = _ic_read_summary(installation_ranges)
    installation = metrics["installation_authentication_reported_separately"]
    require_deep_equal(installation, {
        "container_sha256": independent["container_sha256"],
        "read_ranges": [list(value) for value in installation_ranges],
        "read_request_count": install_summary["read_request_count"],
        "requested_bytes_with_repetition":
            install_summary["requested_bytes_with_repetition"],
        "unique_requested_bytes": install_summary["unique_requested_bytes"],
        "overlap_bytes_requested_again":
            install_summary["overlap_bytes_requested_again"],
        "scan_bytes": independent["total_bytes"],
        "touched_page_indices": list(range(
            (independent["total_bytes"] + IC_PAGE_BYTES - 1) //
            IC_PAGE_BYTES)),
        "touched_page_bytes": _ic_align(independent["total_bytes"],
                                         IC_PAGE_BYTES),
        "excluded_from_per_expert_cold_numerator": True,
    }, "independent installation-authentication scan")
    return {
        "maximum_strict_cold":
            fraction_record(independent["maximum_strict_cold"]),
        "maximum_requested_with_repetition":
            fraction_record(maximum_requested),
        "passes_page_cold_below_2x":
            independent["maximum_strict_cold"] < 2,
        "passes_repeated_requested_below_2x": maximum_requested < 2,
        "independent_parser_and_page_ledger": True,
    }


def verify_physical(
    result: Mapping[str, Any], publication: Mapping[str, Any],
    modules: Mapping[str, Any], panel: Mapping[str, Any], pins: Pins,
    score: Mapping[str, Any], bindings: Mapping[str, str],
    scientific_audit: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = audit_literal_container(
        modules, publication["held"]["UWFCV8.bin"], semantic_decode=True,
        label="UWFCV8")
    identity = audit_literal_container(
        modules, publication["held"]["IDENTITY_FRAMING.bin"],
        semantic_decode=False, label="IDENTITY_FRAMING")
    parsed = candidate["parsed"]
    identity_parsed = identity["parsed"]
    independent = candidate["independent"]
    identity_independent = identity["independent"]
    metrics = candidate["metrics"]
    identity_metrics = identity["metrics"]
    standalone = candidate["standalone"]
    source_final = exact_fields(result["source_final"], {
        "parsed_metrics", "identity_framing_metrics",
        "absolute_saving_vs_bound_current_artifact_bpw",
        "incremental_same_framing_WFA_saving_bpw",
        "raw_payload_minus_full_model_saving_bpw", "payload_rows",
        "standalone_decode", "model_packet_sha256", "container_sha256",
        "identity_framing_container_sha256", "candidate",
        "posterior_diagnostic_handoff",
        "all_adapted_values_deserialized_from_transmitted_model",
        "identical_reconstruction_proved_by_full_f64_digest"},
        "source final")
    for label, item in (("UWFCV8", parsed),
                        ("IDENTITY_FRAMING", identity_parsed)):
        require(item["baseline_artifact_sha256"] == KNOWN_ARTIFACT_SHA256,
                f"{label}: artifact binding")
        require(item["weights"] == SOURCE_WEIGHTS == panel["weights"],
                f"{label}: source weights")
        require(item["experts"] == panel["experts"],
                f"{label}: expert count")
        require(item["reconstruction_sha256"] ==
                pins.original_source_identity["reconstruction_f64_sha256"],
                f"{label}: reconstruction binding")
        require(item["audit_binding_sha256"] ==
                publication["held"]["BOUND_BASELINE_SCORE.json"].sha256,
                f"{label}: score-file binding")
        require_deep_equal(item["binding_hashes"], dict(bindings),
                           f"{label}: complete header bindings")
        require(item["baseline_object_bytes"] == KNOWN_ARTIFACT_BYTES,
                f"{label}: externally pinned baseline object bytes")
        require_float_equal(item["audited_relative_mse"],
                            float(score["relative_mse"]),
                            f"{label}: externally pinned audited MSE")
    for label, item in (("UWFCV8", independent),
                        ("IDENTITY_FRAMING", identity_independent)):
        require(item["baseline_object_bytes"] == KNOWN_ARTIFACT_BYTES and
                item["baseline_artifact_sha256"] == KNOWN_ARTIFACT_SHA256,
                f"{label}: independent baseline binding")
        require_float_equal(item["audited_relative_mse"],
                            float(score["relative_mse"]),
                            f"{label}: independent audited MSE")
        require(item["reconstruction_sha256"] ==
                pins.original_source_identity["reconstruction_f64_sha256"] and
                item["audit_binding_sha256"] ==
                publication["held"]["BOUND_BASELINE_SCORE.json"].sha256,
                f"{label}: independent reconstruction/score binding")
        require_deep_equal(item["binding_hashes"], dict(bindings),
                           f"{label}: independent header bindings")
        require(bytes(item["semantic_packet"]) ==
                bytes(panel["semantic_packet"]) and
                bytes(item["immutable_state"]) ==
                bytes(panel["immutable_state"]),
                f"{label}: independent panel semantics/immutable state")
    for key in ("semantic_packet", "immutable_state", "model_packet"):
        require(bytes(parsed[key]) == bytes(identity_parsed[key]),
                f"candidate/identity shared {key}")
    require(bytes(parsed["semantic_packet"]) == bytes(panel["semantic_packet"]) and
            bytes(parsed["immutable_state"]) == bytes(panel["immutable_state"]),
            "candidate semantics/immutable state direct panel binding")
    require(len(parsed["directory"]) == len(identity_parsed["directory"]) ==
            len(panel["streams"]), "candidate/identity/panel stream count")
    shared_fields = (
        "ordinal", "symbols", "source_weights", "group_rows", "group_cols",
        "profile_q", "decoder_scale", "owner_set_hex", "owners",
        "source_digest", "role", "owner_contributions",
    )
    for ordinal, (candidate_row, identity_row, panel_row) in enumerate(zip(
            parsed["directory"], identity_parsed["directory"], panel["streams"],
            strict=True)):
        require(candidate_row["ordinal"] == identity_row["ordinal"] == ordinal,
                f"directory[{ordinal}] ordinal")
        for field in shared_fields:
            require_deep_equal(candidate_row[field], identity_row[field],
                               f"directory[{ordinal}].{field}")
        expected_contributions = [
            {
                "expert": int(row["expert"]), "role": str(row["role"]),
                "source_offset": int(row["source_offset"]),
                "weight_count": int(row["weight_count"]),
            }
            for row in panel_row["owner_contributions"]
        ]
        expected_panel_fields = {
            "ordinal": int(panel_row["stream_ordinal"]),
            "symbols": int(panel_row["symbols"]),
            "source_weights": int(panel_row["weight_charge"]),
            "group_rows": int(panel_row["shape_rows"]),
            "group_cols": int(panel_row["shape_cols"]),
            "profile_q": int(panel_row["profile_q"]),
            "owner_set_hex": str(panel_row["owner_set_hex"]),
            "owners": tuple(int(value) for value in
                            panel_row["owner_expert_ordinals"]),
            "source_digest": str(panel_row["source_digest"]),
            "role": str(panel_row["role"]),
            "owner_contributions": expected_contributions,
        }
        for field, expected in expected_panel_fields.items():
            observed = ([dict(row) for row in
                         candidate_row["owner_contributions"]]
                        if field == "owner_contributions" else
                        candidate_row[field])
            require_deep_equal(observed, expected,
                               f"candidate directory[{ordinal}] panel {field}")
            observed_identity = ([dict(row) for row in
                                  identity_row["owner_contributions"]]
                                 if field == "owner_contributions" else
                                 identity_row[field])
            require_deep_equal(observed_identity, expected,
                               f"identity directory[{ordinal}] panel {field}")
        require(bytes(candidate_row["owner_set"]) ==
                bytes(panel_row["owner_set"]) ==
                bytes(identity_row["owner_set"]),
                f"directory[{ordinal}] direct panel owner-set bytes")
        require(_float_bits(candidate_row["decoder_scale"]) ==
                _float_bits(float(panel_row["decoder_scale"])) ==
                _float_bits(identity_row["decoder_scale"]),
                f"directory[{ordinal}] direct panel decoder-scale bits")
        require(identity_row["logical_bits"] == panel_row["baseline_logical_bits"],
                f"identity[{ordinal}] baseline logical bits")
        require(bytes(identity_row["payload"]) ==
                bytes(panel_row["baseline_payload"]),
                f"identity[{ordinal}] baseline payload")
    require(standalone["reconstruction"]["full_reconstruction_f64_sha256"] ==
            pins.original_source_identity["reconstruction_f64_sha256"],
            "candidate standalone reconstruction")
    require(len(standalone["streams"]) == len(panel["streams"]),
            "candidate standalone stream count")
    for ordinal, (decoded_row, panel_row) in enumerate(zip(
            standalone["streams"], panel["streams"], strict=True)):
        require(decoded_row["ordinal"] == ordinal and
                decoded_row["source_digest"] == panel_row["source_digest"] and
                decoded_row["canonical_reencode_matches"] is True,
                f"standalone stream[{ordinal}] decoded panel commitment")
    candidate_bytes = publication["held"]["UWFCV8.bin"].data
    identity_bytes = publication["held"]["IDENTITY_FRAMING.bin"].data
    require(metrics["actual_container_bytes"] == len(candidate_bytes),
            "candidate literal bytes")
    require(identity_metrics["actual_container_bytes"] == len(identity_bytes),
            "identity literal bytes")
    exact_rate = Fraction(8 * len(candidate_bytes), SOURCE_WEIGHTS)
    require(fraction_from_record(metrics["actual_physical_rate_rational"],
                                 "candidate physical rate") == exact_rate,
            "candidate rational physical rate")
    require_float_equal(metrics["actual_physical_rate_bpw"], float(exact_rate),
                        "candidate physical rate float")
    require_float_equal(metrics["audited_identical_reconstruction_relative_mse"],
                        float(score["relative_mse"]),
                        "candidate independently pinned metric MSE")
    expected_net_saving = (8.0 *
        (KNOWN_ARTIFACT_BYTES - len(candidate_bytes)) / SOURCE_WEIGHTS)
    require_float_equal(metrics["net_physical_saving_bpw"],
                        expected_net_saving,
                        "candidate independent net physical saving")
    identity_rate = Fraction(8 * len(identity_bytes), SOURCE_WEIGHTS)
    require(fraction_from_record(identity_metrics["actual_physical_rate_rational"],
                                 "identity physical rate") == identity_rate,
            "identity independent rational rate")
    require_float_equal(identity_metrics["actual_physical_rate_bpw"],
                        float(identity_rate), "identity physical rate float")
    expected_f = float(score["relative_mse"]) * math.pow(
        2.0, 2.0 * float(exact_rate))
    require_float_equal(metrics["F_from_actual_bytes_and_identical_reconstruction"],
                        expected_f, "candidate literal F")
    require(metrics["passes_rate_interval"] is
            (Fraction(43, 20) <= exact_rate <= Fraction(5, 2)),
            "candidate rate gate")
    require(metrics["passes_F_target"] is (expected_f <= 0.8),
            "candidate F gate")
    require(metrics["routed_io_authoritative_descriptor_backed"] is True,
            "candidate descriptor-backed routed evidence")
    producer_bandwidth = verify_bandwidth_metrics(metrics)
    bandwidth = verify_independent_bandwidth(independent, metrics)
    require_deep_equal(producer_bandwidth, {
        key: bandwidth[key] for key in producer_bandwidth
    }, "independent/producer bandwidth recomputation")
    require(source_final["container_sha256"] ==
            publication["held"]["UWFCV8.bin"].sha256,
            "source-final candidate hash")
    require(source_final["identity_framing_container_sha256"] ==
            publication["held"]["IDENTITY_FRAMING.bin"].sha256,
            "source-final identity hash")
    require(source_final["model_packet_sha256"] ==
            sha256(bytes(parsed["model_packet"])), "source-final model hash")
    require_deep_equal(source_final["parsed_metrics"], metrics,
                       "source-final physical metrics")
    require_deep_equal(source_final["identity_framing_metrics"], identity_metrics,
                       "source-final identity metrics")
    require_deep_equal(source_final["standalone_decode"], standalone,
                       "source-final standalone decode")
    handoff = modules["codec"].posterior_diagnostic_handoff(
        modules["common"], parsed)
    require_deep_equal(source_final["posterior_diagnostic_handoff"], handoff,
                       "source-final selected-decision commitments")
    independent_decision_rows = [{
        "ordinal": row["ordinal"], "symbols": row["symbols"],
        "logical_bits": row["logical_bits"],
        "decoded_selected_decision_triplet_sha256": row["source_digest"],
        "payload_sha256": hashlib.sha256(row["payload"]).hexdigest(),
        "profile_q": row["profile_q"], "role": row["role"],
        "owner_set_hex": row["owner_set_hex"],
        "owner_contributions": [dict(value) for value in
                                row["owner_contributions"]],
    } for row in independent["directory"]]
    independent_decision_commitment = sha256(canonical_json(
        independent_decision_rows))
    require_deep_equal(handoff["stream_decision_triplet_commitments"],
                       independent_decision_rows,
                       "independent decision commitment rows")
    require(handoff["decoded_sc_decision_triplet_commitment_sha256"] ==
            independent_decision_commitment,
            "independent decision commitment aggregate")
    require_deep_equal(source_final["candidate"], parsed["candidate"].as_dict(),
                       "source-final deserialized candidate")
    require_deep_equal(source_final["candidate"], scientific_audit["winner"],
                       "physical candidate/vote-selected topology")
    payload_rows = []
    for candidate_row, panel_row in zip(parsed["directory"], panel["streams"],
                                        strict=True):
        payload_rows.append({
            "ordinal": int(panel_row["stream_ordinal"]),
            "baseline_payload_bytes": int(panel_row["baseline_payload_bytes"]),
            "new_payload_bytes": len(bytes(candidate_row["payload"])),
            "baseline_logical_bits": int(panel_row["baseline_logical_bits"]),
            "new_logical_bits": int(candidate_row["logical_bits"]),
        })
    require_deep_equal(source_final["payload_rows"], payload_rows,
                       "source-final payload ledger")
    raw_saving = (sum(8 * (row["baseline_payload_bytes"] -
                            row["new_payload_bytes"]) for row in payload_rows)
                  - 8 * len(bytes(parsed["model_packet"])))
    scalar_expectations = {
        "absolute_saving_vs_bound_current_artifact_bpw":
            8.0 * (KNOWN_ARTIFACT_BYTES - len(candidate_bytes)) / SOURCE_WEIGHTS,
        "incremental_same_framing_WFA_saving_bpw":
            8.0 * (len(identity_bytes) - len(candidate_bytes)) / SOURCE_WEIGHTS,
        "raw_payload_minus_full_model_saving_bpw": raw_saving / SOURCE_WEIGHTS,
    }
    for name, value in scalar_expectations.items():
        require_float_equal(source_final[name], value, f"source-final {name}")
    require(source_final["all_adapted_values_deserialized_from_transmitted_model"]
            is True, "source-final transmitted-model flag")
    require(source_final["identical_reconstruction_proved_by_full_f64_digest"]
            is True, "source-final identical reconstruction flag")
    compact = {
        "container_bytes": metrics["actual_container_bytes"],
        "physical_rate_bpw": metrics["actual_physical_rate_bpw"],
        "physical_rate_rational": metrics["actual_physical_rate_rational"],
        "relative_mse": metrics["audited_identical_reconstruction_relative_mse"],
        "F": metrics["F_from_actual_bytes_and_identical_reconstruction"],
        "net_physical_saving_bpw": metrics["net_physical_saving_bpw"],
        "passes_rate_interval": metrics["passes_rate_interval"],
        "passes_F_target": metrics["passes_F_target"],
        "passes_cold_read_below_2x": metrics["passes_cold_read_below_2x"],
        "container_sha256": source_final["container_sha256"],
        "identity_framing_container_sha256":
            source_final["identity_framing_container_sha256"],
        "model_packet_sha256": source_final["model_packet_sha256"],
    }
    require_deep_equal(result["physical"], compact, "RESULT compact physical")
    return {
        "candidate_sha256": publication["held"]["UWFCV8.bin"].sha256,
        "identity_sha256": publication["held"]["IDENTITY_FRAMING.bin"].sha256,
        "model_packet_sha256": sha256(bytes(parsed["model_packet"])),
        "directory_sha256": sha256(bytes(parsed["directory_blob"])),
        "identity_directory_sha256":
            sha256(bytes(identity_parsed["directory_blob"])),
        "decision_commitment_sha256":
            handoff["decoded_sc_decision_triplet_commitment_sha256"],
        "reconstruction_sha256":
            standalone["reconstruction"]["full_reconstruction_f64_sha256"],
        "rate": fraction_record(exact_rate), "F": expected_f,
        "physical_pass": bool(metrics["passes_rate_interval"] and
                              metrics["passes_F_target"]),
        "cold_pass": bool(metrics["passes_cold_read_below_2x"]),
        "bandwidth": bandwidth,
        "independent_container_parser": True,
        "independent_byte_ledger_entries": len(independent["byte_ledger"]),
        "independent_selected_stream_causal_replay": True,
        "identity_rate": fraction_record(identity_rate),
        "identity_semantic_decode":
            "IMPOSSIBLE_FROM_INTENTIONALLY_MISMATCHED_COUNTERFACTUAL_BY_SEALED_V8_ABI",
    }


def verify_panel_cache(result: Mapping[str, Any]) -> None:
    cache = exact_fields(result["exploratory_panel_cache"], {
        "schema", "status", "artifact_bytes", "artifact_sha256",
        "extract_calls", "delegate_extract_calls", "same_panel_object_reused",
        "positive_claim_authority", "receipt_sha256"}, "panel cache")
    verify_internal_seal(cache, "receipt_sha256", "panel cache")
    require(cache["schema"] ==
            "uwfa-sc-v8-qwen-early-gate-single-artifact-panel-cache-v0" and
            cache["status"] == "EXPLORATORY_EXACT_IDENTITY_REUSE",
            "panel-cache schema/status")
    require(cache["artifact_bytes"] == KNOWN_ARTIFACT_BYTES and
            cache["artifact_sha256"] == KNOWN_ARTIFACT_SHA256,
            "panel-cache artifact")
    require(cache["extract_calls"] == 2 and
            cache["delegate_extract_calls"] == 1 and
            cache["same_panel_object_reused"] is True and
            cache["positive_claim_authority"] is False,
            "panel-cache exact counters")


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def independent_public_context(level: int, base_frequency: int,
                               within_reset: int) -> int:
    require(0 <= level < 6 and 1 <= base_frequency <= 65535 and
            within_reset >= 0, "independent public context input")
    prior = min(15, base_frequency * 16 // 65536)
    return ((level * 16 + prior) * 4) + (within_reset & 3)


def independent_transition(candidate: Any, state: int, bit: int,
                           context: int, within_reset: int) -> int:
    states = int(candidate.states)
    mask = states - 1
    topology = str(candidate.topology)
    require(0 <= state < states and bit in (0, 1),
            "independent transition state/bit")
    if topology == "suffix":
        return ((state << 1) | bit) & mask
    if topology == "xor_sketch":
        if bit == 0:
            return state
        sketch = _mix32(0xA511E9B3 ^ context ^
                        (within_reset * 0x9E3779B1)) & mask
        return state ^ (sketch or 1)
    if topology == "modular_ones":
        weight = (_mix32(0x63D83595 ^ context ^
                         ((within_reset & 3) << 20)) & mask) | 1
        return (state + weight * bit) & mask
    if topology == "rolling_affine":
        multiplier = (5 if states >= 8 else 1) & mask
        addend = _mix32(0xB5297A4D ^ context ^
                       ((within_reset & 3) << 24)) & mask
        return (multiplier * state + addend + bit) & mask
    if topology == "signed_saturating":
        return min(mask, state + 1) if bit else max(0, state - 1)
    raise AuditError("independent transition topology")


def independent_add_counts(bits: Sequence[int], levels: Sequence[int],
                           base: Sequence[int], candidate: Any,
                           counts: list[int]) -> None:
    require(len(bits) == len(levels) == len(base) and len(bits) > 0,
            "independent count stream geometry")
    require(len(counts) == int(candidate.states) * 384 * 2,
            "independent count table geometry")
    state = 0
    reset = int(candidate.reset_length)
    for position, (raw_bit, raw_level, raw_base) in enumerate(
            zip(bits, levels, base, strict=True)):
        within = position % reset
        if within == 0:
            state = 0
        bit = int(raw_bit)
        context = independent_public_context(int(raw_level), int(raw_base),
                                             within)
        counts[(state * 384 + context) * 2 + bit] += 1
        state = independent_transition(candidate, state, bit, context, within)


def independent_q16_frequencies(counts: Sequence[int]) -> list[int]:
    require(len(counts) > 0 and len(counts) % 2 == 0,
            "independent Q0.16 count geometry")
    output = []
    for index in range(0, len(counts), 2):
        c0, c1 = int(counts[index]), int(counts[index + 1])
        require(c0 >= 0 and c1 >= 0, "independent nonnegative counts")
        # Jeffreys p1=(c1+1/2)/(c0+c1+1), evaluated with integer rounding.
        numerator = 65536 * (2 * c1 + 1)
        denominator = 2 * (c0 + c1 + 1)
        value = (numerator + denominator // 2) // denominator
        output.append(min(65535, max(1, value)))
    return output


def independent_stream_frequencies(
    bits: Sequence[int], levels: Sequence[int], base: Sequence[int],
    candidate: Any, frequencies: Sequence[int],
) -> list[int]:
    require(len(bits) == len(levels) == len(base) and len(bits) > 0,
            "independent stream-frequency geometry")
    require(len(frequencies) == int(candidate.states) * 384,
            "independent model-frequency geometry")
    state = 0
    output = []
    reset = int(candidate.reset_length)
    for position, (raw_bit, raw_level, raw_base) in enumerate(zip(
            bits, levels, base, strict=True)):
        within = position % reset
        if within == 0:
            state = 0
        bit = int(raw_bit)
        context = independent_public_context(int(raw_level), int(raw_base),
                                             within)
        output.append(int(frequencies[state * 384 + context]))
        state = independent_transition(candidate, state, bit, context, within)
    require(all(1 <= value <= 65535 for value in output),
            "independent stream frequencies bounded")
    return output


def independent_arithmetic_encode(
    bits: Sequence[int], frequencies: Sequence[int],
) -> tuple[bytes, int]:
    require(len(bits) == len(frequencies) and len(bits) > 0,
            "independent arithmetic geometry")
    full, half, quarter, three_quarters = (
        1 << 32, 1 << 31, 1 << 30, 3 << 30)
    low, high, pending = 0, full - 1, 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        if pending:
            output.extend([1 - bit] * pending)
            pending = 0

    for raw_bit, raw_frequency in zip(bits, frequencies, strict=True):
        bit = int(raw_bit)
        frequency = int(raw_frequency)
        require(bit in (0, 1) and 1 <= frequency <= 65535,
                "independent arithmetic symbol")
        width = high - low + 1
        split = low + width * (65536 - frequency) // 65536 - 1
        require(low <= split < high, "independent arithmetic split")
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
    pending += 1
    emit(0 if low < quarter else 1)
    payload = bytearray((len(output) + 7) // 8)
    for index, bit in enumerate(output):
        payload[index >> 3] |= bit << (7 - (index & 7))
    return bytes(payload), len(output)


def independent_unifilar_encode(
    bits: Sequence[int], levels: Sequence[int], base: Sequence[int],
    candidate: Any, frequencies: Sequence[int],
) -> tuple[bytes, int]:
    return independent_arithmetic_encode(
        bits, independent_stream_frequencies(
            bits, levels, base, candidate, frequencies))


def independent_unifilar_decode(
    payload: bytes, logical_bits: int, levels: Sequence[int],
    base: Sequence[int], candidate: Any, frequencies: Sequence[int],
) -> list[int]:
    require(isinstance(payload, bytes) and 0 < logical_bits <= len(payload) * 8,
            "independent arithmetic payload geometry")
    require(len(levels) == len(base) and len(levels) > 0,
            "independent arithmetic decode contexts")
    if logical_bits & 7:
        require(not (payload[-1] &
                     ((1 << (8 - (logical_bits & 7))) - 1)),
                "independent arithmetic terminal padding")
    bit_position = 0

    def read() -> int:
        nonlocal bit_position
        if bit_position >= logical_bits:
            return 0
        value = ((payload[bit_position >> 3] >>
                  (7 - (bit_position & 7))) & 1)
        bit_position += 1
        return value

    full, half, quarter, three_quarters = (
        1 << 32, 1 << 31, 1 << 30, 3 << 30)
    low, high, code = 0, full - 1, 0
    for _ in range(32):
        code = ((code << 1) & (full - 1)) | read()
    state = 0
    output: list[int] = []
    used_frequencies: list[int] = []
    reset = int(candidate.reset_length)
    for index in range(len(levels)):
        within = index % reset
        if within == 0:
            state = 0
        context = independent_public_context(int(levels[index]),
                                             int(base[index]), within)
        frequency = int(frequencies[state * 384 + context])
        require(1 <= frequency <= 65535,
                "independent arithmetic decode frequency")
        width = high - low + 1
        split = low + width * (65536 - frequency) // 65536 - 1
        require(low <= split < high, "independent arithmetic decode split")
        bit = 0 if code <= split else 1
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarters:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
            code = ((code << 1) & (full - 1)) | read()
        output.append(bit)
        used_frequencies.append(frequency)
        state = independent_transition(candidate, state, bit, context, within)
    replay, replay_bits = independent_arithmetic_encode(output,
                                                         used_frequencies)
    require(replay == payload and replay_bits == logical_bits,
            "independent arithmetic canonical replay")
    return output


def independent_deserialize_model(packet: bytes) -> tuple[Any, list[int]]:
    require(isinstance(packet, bytes) and len(packet) >= 66,
            "independent model packet envelope")
    (magic, version, topology_id, states, selector_bytes, reset_length,
     contexts, candidate_hash, tensor_checksum) = struct.unpack(
         "<8sHHHHII32s8s", packet[:64])
    require(magic == b"UWFAV8\x00\x00" and version == 3 and
            selector_bytes == 2 and contexts == 384 and
            0 <= topology_id < len(IC_TOPOLOGIES) and
            states in IC_STATE_SIZES and reset_length in IC_RESET_LENGTHS,
            "independent model header")
    topology = IC_TOPOLOGIES[topology_id]
    state_index = IC_STATE_SIZES.index(states)
    reset_index = IC_RESET_LENGTHS.index(reset_length)
    selector = ((topology_id * len(IC_STATE_SIZES) + state_index) *
                len(IC_RESET_LENGTHS) + reset_index)
    candidate_record = {
        "topology": topology, "topology_id": topology_id, "states": states,
        "reset_length": reset_length, "selector_ordinal": selector,
    }
    require(candidate_hash == hashlib.sha256(
        canonical_json(candidate_record)).digest(),
        "independent model candidate hash")
    require(struct.unpack_from("<H", packet, 64)[0] == selector,
            "independent model selector")
    tensor = packet[66:]
    require(len(tensor) == states * 384 * 2 and
            tensor_checksum == hashlib.sha256(tensor).digest()[:8],
            "independent model tensor geometry/checksum")
    frequencies = list(struct.unpack(f"<{states * 384}H", tensor))
    require(all(1 <= value <= 65535 for value in frequencies),
            "independent model frequency bounds")
    return types.SimpleNamespace(**candidate_record), frequencies


def independent_triplet_sha256(
    bits: bytes, levels: bytes, base_u16le: bytes,
) -> str:
    require(bits and len(levels) == len(bits) and
            len(base_u16le) == 2 * len(bits),
            "independent decision triplet geometry")
    digest_state = hashlib.sha256()
    for value in (bits, levels, base_u16le):
        digest_state.update(struct.pack("<Q", len(value)))
        digest_state.update(value)
    return digest_state.hexdigest()


def independent_literal_layout_bits(
    panel: Mapping[str, Any], model_bytes: int,
    replacements: Mapping[int, int],
) -> int:
    """Recompute the v4 literal layout from geometry, not producer helpers."""
    exact_int(model_bytes, "independent layout model bytes", 0, 1 << 26)
    streams = panel["streams"]
    require(isinstance(streams, list) and streams,
            "independent layout stream list")
    require(set(replacements).issubset(set(range(len(streams)))),
            "independent layout replacement ordinals")
    grouped: dict[bytes, list[int]] = {}
    experts = exact_int(panel["experts"], "independent layout experts", 1, 256)
    for ordinal, row in enumerate(streams):
        owner_set = bytes(row["owner_set"])
        _ic_owner_ordinals(owner_set, experts)
        grouped.setdefault(owner_set, []).append(ordinal)
    owner_sets = sorted(grouped,
                        key=lambda value:
                        (len(_ic_owner_ordinals(value, experts)) != 1,
                         _ic_owner_ordinals(value, experts)))
    semantic_bytes = len(bytes(panel["semantic_packet"]))
    immutable_bytes = len(bytes(panel["immutable_state"]))
    directory_bytes = len(streams) * IC_DIRECTORY_RECORD_BYTES
    immutable_offset = _ic_align(IC_HEADER_BYTES + semantic_bytes, 64)
    model_offset = _ic_align(immutable_offset + immutable_bytes, IC_PAGE_BYTES)
    directory_offset = _ic_align(model_offset + model_bytes, IC_PAGE_BYTES)
    shared_bytes = _ic_align(directory_offset + directory_bytes, IC_PAGE_BYTES)
    region_lengths = []
    for owner_set in owner_sets:
        frame_area = 0
        for ordinal in grouped[owner_set]:
            row = streams[ordinal]
            logical = exact_int(replacements.get(
                ordinal, int(row["baseline_logical_bits"])),
                f"independent layout stream[{ordinal}] logical", 1, 1 << 56)
            contribution_count = len(row["owner_contributions"])
            metadata = _ic_align(
                IC_FRAME_HEADER_BYTES +
                IC_CONTRIBUTION_RECORD_BYTES * contribution_count, 64)
            frame_area += _ic_align(metadata + (logical + 7) // 8, 64)
        region_lengths.append(_ic_align(
            IC_REGION_HEADER_BYTES + frame_area, IC_PAGE_BYTES))
    base_total = shared_bytes + sum(region_lengths)
    weights = exact_int(panel["weights"], "independent layout weights", 1,
                        1 << 50)
    minimum_total = (weights * 43 + 8 * 20 - 1) // (8 * 20)
    padding_pages = ((max(0, minimum_total - base_total) +
                      IC_PAGE_BYTES - 1) // IC_PAGE_BYTES)
    return 8 * (base_total + padding_pages * IC_PAGE_BYTES)


def replay_selected_models_cpu(
    scientific: Mapping[str, Any],
    scientific_audit: Mapping[str, Any],
    modules: Mapping[str, Any],
    panel: Mapping[str, Any],
    literal_container: bytes,
) -> dict[str, Any]:
    """Mandatory exact selected-cell replay; intentionally never all-150 GPU work."""

    common = modules["common"]
    stage = modules["stage"]
    protocol = modules["protocol"]
    codec = modules["codec"]
    streams = panel["streams"]

    def fit(indices: Sequence[int], candidate: Any) -> list[int]:
        counts = [0] * (int(candidate.states) * 384 * 2)
        for index in indices:
            row = streams[index]
            independent_add_counts(row["bits"], row["levels"], row["base"],
                                   candidate, counts)
        frequencies = independent_q16_frequencies(counts)
        require(frequencies == common.q16_frequencies_from_counts(counts),
                "independent/production Jeffreys Q0.16 equality")
        return frequencies

    def lengths(indices: Sequence[int], candidate: Any,
                frequencies: Sequence[int]) -> list[int]:
        output = []
        for index in indices:
            row = streams[index]
            independent_payload, independent_logical = independent_unifilar_encode(
                row["bits"], row["levels"], row["base"], candidate,
                frequencies)
            producer_payload, producer_logical = common.encode_unifilar_stream(
                row["bits"], row["levels"], row["base"], candidate, frequencies)
            require(independent_payload == producer_payload and
                    independent_logical == producer_logical,
                    f"independent/producer causal arithmetic stream[{index}]")
            output.append(int(independent_logical))
        return output

    fold_receipts = []
    for ordinal, (fold, plan) in enumerate(zip(
            scientific["folds"], scientific_audit["plans"], strict=True)):
        selected_ordinal = int(
            fold["selected_by_inner_validation_only"]["selector_ordinal"])
        candidate = common.candidate_bank()[selected_ordinal]
        train_frequencies = fit(plan["train_indices"], candidate)
        validation_lengths = lengths(plan["validation_indices"], candidate,
                                     train_frequencies)
        validation_score = stage.literal_validation_score(
            common, protocol, codec, panel, plan["validation_indices"],
            validation_lengths, candidate, train_frequencies)
        independent_validation_score = independent_literal_layout_bits(
            panel, len(common.serialize_model(candidate, train_frequencies)),
            dict(zip(plan["validation_indices"], validation_lengths,
                     strict=True)))
        require(validation_score == independent_validation_score,
                f"selected CPU replay fold[{ordinal}] independent validation layout")
        require(validation_score == fold["inner_validation_exact_charged_bits"],
                f"selected CPU replay fold[{ordinal}] validation score")
        development_frequencies = fit(plan["development_indices"], candidate)
        test_lengths = lengths(plan["test_indices"], candidate,
                               development_frequencies)
        candidate_score = stage.literal_validation_score(
            common, protocol, codec, panel, plan["test_indices"], test_lengths,
            candidate, development_frequencies)
        independent_candidate_score = independent_literal_layout_bits(
            panel,
            len(common.serialize_model(candidate, development_frequencies)),
            dict(zip(plan["test_indices"], test_lengths, strict=True)))
        require(candidate_score == independent_candidate_score,
                f"selected CPU replay fold[{ordinal}] independent test layout")
        require(candidate_score == fold["literal_candidate_container_bits"],
                f"selected CPU replay fold[{ordinal}] test score")
        allocated_candidate_bits = float(sum(
            8 * ((logical + 7) // 8) for logical in test_lengths))
        require_float_equal(fold["allocated_candidate_bits"],
                            allocated_candidate_bits,
                            f"selected CPU replay fold[{ordinal}] allocated candidate bits")
        fold_receipts.append({
            "component_ordinal": ordinal,
            "selector_ordinal": selected_ordinal,
            "train_q016_sha256": sha256(struct.pack(
                f"<{len(train_frequencies)}H", *train_frequencies)),
            "development_q016_sha256": sha256(struct.pack(
                f"<{len(development_frequencies)}H", *development_frequencies)),
            "validation_exact_charged_bits": validation_score,
            "independent_validation_layout_bits":
                independent_validation_score,
            "test_literal_candidate_bits": candidate_score,
            "independent_test_layout_bits": independent_candidate_score,
            "test_logical_lengths": test_lengths,
            "allocated_candidate_bits": int(allocated_candidate_bits),
        })
    winner_ordinal = int(scientific_audit["winner"]["selector_ordinal"])
    winner = common.candidate_bank()[winner_ordinal]
    final_frequencies = fit(list(range(len(streams))), winner)
    serialized = common.serialize_model(winner, final_frequencies)
    independently_parsed = independent_parse_container(
        literal_container, "selected CPU replay final container")
    require(bytes(independently_parsed["model_packet"]) == serialized,
            "selected CPU replay final Q0.16 model packet")
    parsed_candidate, parsed_frequencies = independent_deserialize_model(
        bytes(independently_parsed["model_packet"]))
    require_deep_equal({
        "topology": parsed_candidate.topology,
        "topology_id": parsed_candidate.topology_id,
        "states": parsed_candidate.states,
        "reset_length": parsed_candidate.reset_length,
        "selector_ordinal": parsed_candidate.selector_ordinal,
    }, winner.as_dict(), "selected CPU replay independent final candidate")
    require(parsed_frequencies == final_frequencies,
            "selected CPU replay independent final frequencies")
    final_lengths = []
    for ordinal, (row, directory_row) in enumerate(zip(
            streams, independently_parsed["directory"], strict=True)):
        payload, logical = independent_unifilar_encode(
            row["bits"], row["levels"], row["base"], parsed_candidate,
            parsed_frequencies)
        require(payload == directory_row["payload"] and
                logical == directory_row["logical_bits"],
                f"selected CPU replay final stream[{ordinal}] bytes/length")
        decoded = independent_unifilar_decode(
            directory_row["payload"], directory_row["logical_bits"],
            row["levels"], row["base"], parsed_candidate, parsed_frequencies)
        require(decoded == [int(value) for value in row["bits"]],
                f"selected CPU replay final stream[{ordinal}] causal decode")
        decoded_triplet = independent_triplet_sha256(
            bytes(decoded), bytes(int(value) for value in row["levels"]),
            struct.pack(f"<{len(row['base'])}H",
                        *(int(value) for value in row["base"])))
        require(decoded_triplet == directory_row["source_digest"] ==
                row["source_digest"],
                f"selected CPU replay final stream[{ordinal}] triplet commitment")
        final_lengths.append(logical)
    require(independent_literal_layout_bits(
        panel, len(serialized),
        {index: logical for index, logical in enumerate(final_lengths)}) ==
        8 * len(literal_container),
        "selected CPU replay independent final literal layout")
    return {
        "status": "PASS_EXACT_SELECTED_CELL_Q016_CPU_REPLAY",
        "folds": fold_receipts,
        "final_selector_ordinal": winner_ordinal,
        "final_q016_sha256": sha256(struct.pack(
            f"<{len(final_frequencies)}H", *final_frequencies)),
        "serialized_model_sha256": sha256(serialized),
        "final_logical_lengths_sha256": sha256(struct.pack(
            f"<{len(final_lengths)}Q", *final_lengths)),
        "independent_final_container_layout_bits": 8 * len(literal_container),
        "all_150_alternative_candidate_optimality_replayed": False,
    }


def verify(authorization: str, pins: Pins) -> dict[str, Any]:
    require(authorization == AUTHORIZATION, "explicit result-audit authorization")
    require(os.name == "posix", "result audit requires POSIX descriptors")
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "invoke result audit with CPython -I -B")
    with ExitStack() as stack:
        v9 = authenticate_package(
            stack, pins.v9_package, label="sealed v9 package",
            expected_manifest_sha256=KNOWN_V9_MANIFEST_SHA256,
            expected_root_sha256=KNOWN_V9_SOURCE_ROOT_SHA256,
            manifest_schema="uwfa-sc-v9-primary-source-manifest-v0",
            required_members=V9_REQUIRED_MEMBERS,
        )
        require(v9["hashes"]["primary_gate.py"] == KNOWN_V9_RUNNER_SHA256,
                "v9 primary runner external pin")
        runner_constants = literal_assignments(
            v9["sources"]["primary_gate.py"], {
                "SCHEMA", "AUTHORIZATION", "SEALED_V8_MANIFEST_SHA256",
                "PINNED_SUPPORT_SHA256", "CURRENT_ARTIFACT_SHA256",
                "CURRENT_ARTIFACT_BYTES", "SOURCE_WEIGHTS",
                "PINNED_PRIMARY_CELL_SYMBOL_UPDATES",
                "PINNED_DEFERRED_MAXIMUM_UPDATES",
                "PINNED_DEFERRED_COORDINATE_UPDATES", "PINNED_PANEL_SYMBOLS",
                "PINNED_PANEL_STREAMS", "PINNED_FOLD_UPDATES",
                "PRIMARY_KERNEL_BUDGET_SECONDS",
                "CONSERVATIVE_THROUGHPUT_MIN",
                "CONSERVATIVE_THROUGHPUT_MAX", "EXPECTED_DEVICE_NAME"},
            "v9 primary runner")
        require(runner_constants == {
            "SCHEMA": RESULT_SCHEMA,
            "AUTHORIZATION": "RUN_EXACT_QWEN_PRIMARY_ONLY_NONPROMOTING_V0",
            "SEALED_V8_MANIFEST_SHA256": KNOWN_V8_MANIFEST_SHA256,
            "PINNED_SUPPORT_SHA256": KNOWN_SUPPORT_SHA256,
            "CURRENT_ARTIFACT_SHA256": KNOWN_ARTIFACT_SHA256,
            "CURRENT_ARTIFACT_BYTES": KNOWN_ARTIFACT_BYTES,
            "SOURCE_WEIGHTS": SOURCE_WEIGHTS,
            "PINNED_PRIMARY_CELL_SYMBOL_UPDATES": PINNED_PRIMARY_UPDATES,
            "PINNED_DEFERRED_MAXIMUM_UPDATES":
                PINNED_DEFERRED_MAXIMUM_UPDATES,
            "PINNED_DEFERRED_COORDINATE_UPDATES":
                PINNED_DEFERRED_COORDINATE_UPDATES,
            "PINNED_PANEL_SYMBOLS": PINNED_PANEL_SYMBOLS,
            "PINNED_PANEL_STREAMS": PINNED_PANEL_STREAMS,
            "PINNED_FOLD_UPDATES": PINNED_FOLD_UPDATES,
            "PRIMARY_KERNEL_BUDGET_SECONDS": PRIMARY_KERNEL_BUDGET_SECONDS,
            "CONSERVATIVE_THROUGHPUT_MIN": CONSERVATIVE_THROUGHPUT_MIN,
            "CONSERVATIVE_THROUGHPUT_MAX": CONSERVATIVE_THROUGHPUT_MAX,
            "EXPECTED_DEVICE_NAME": EXPECTED_DEVICE_NAME,
        }, "v9 runner frozen scientific constants")
        support = stack.enter_context(HeldAbsoluteRegular(
            pins.support_path, MAX_SOURCE_BYTES, "pinned support",
            KNOWN_SUPPORT_SHA256))
        support_constants = literal_assignments(
            support.data, {"SEALED_V8_MANIFEST_SHA256", "STRATA_COMMON_SHA256",
                           "FROZEN_AUDITOR_SHA256", "CURRENT_ARTIFACT_SHA256",
                           "CURRENT_ARTIFACT_BYTES", "SOURCE_WEIGHTS",
                           "BASELINE_PLAN_SHA256"}, "pinned support")
        require(support_constants["SEALED_V8_MANIFEST_SHA256"] ==
                KNOWN_V8_MANIFEST_SHA256, "support v8 manifest pin")
        require(support_constants["STRATA_COMMON_SHA256"] ==
                KNOWN_STRATA_COMMON_SHA256, "support STRATA pin")
        require(support_constants["FROZEN_AUDITOR_SHA256"] ==
                KNOWN_FROZEN_AUDITOR_SHA256, "support auditor pin")
        require(support_constants["CURRENT_ARTIFACT_SHA256"] ==
                KNOWN_ARTIFACT_SHA256 and
                support_constants["CURRENT_ARTIFACT_BYTES"] ==
                KNOWN_ARTIFACT_BYTES and
                support_constants["SOURCE_WEIGHTS"] == SOURCE_WEIGHTS,
                "support artifact geometry")
        digest(support_constants["BASELINE_PLAN_SHA256"],
               "support baseline plan")
        v8 = authenticate_package(
            stack, pins.v8_package, label="sealed v8 package",
            expected_manifest_sha256=KNOWN_V8_MANIFEST_SHA256,
            expected_root_sha256=KNOWN_V8_SOURCE_ROOT_SHA256,
            manifest_schema="unifilar-wfa-source-manifest-v8",
            required_members=V8_REQUIRED_MEMBERS,
            external_members=pins.v8_members,
        )
        strata = stack.enter_context(HeldAbsoluteRegular(
            pins.strata_common_path, MAX_SOURCE_BYTES, "STRATA common",
            KNOWN_STRATA_COMMON_SHA256))
        frozen = stack.enter_context(HeldAbsoluteRegular(
            pins.frozen_auditor_path, MAX_SOURCE_BYTES, "frozen auditor",
            KNOWN_FROZEN_AUDITOR_SHA256))
        artifact = stack.enter_context(HeldAbsoluteRegular(
            pins.artifact_path, KNOWN_ARTIFACT_BYTES, "Qwen artifact",
            KNOWN_ARTIFACT_SHA256, KNOWN_ARTIFACT_BYTES))

        # NumPy and numerical source modules are loaded only after all source
        # and artifact bytes have authenticated. No CuPy import occurs.
        modules = load_authenticated_modules(
            v8, strata.data, strata.sha256, frozen.data, frozen.sha256)
        panel = modules["adapter"].extract_from_current(artifact.data)
        require(isinstance(panel, dict), "independently decoded panel")
        modules["stage"].attach_semantic_owners(modules["protocol"], panel)
        modules["protocol"].panel_geometry(panel)
        require(panel["artifact"]["raw_sha256"] == KNOWN_ARTIFACT_SHA256 and
                panel["artifact"]["raw_bytes"] == KNOWN_ARTIFACT_BYTES,
                "panel artifact binding")
        require(panel["weights"] == SOURCE_WEIGHTS and
                len(panel["streams"]) == PINNED_PANEL_STREAMS,
                "panel source geometry")
        full_geometry = modules["protocol"].geometry_sha256(
            modules["common"], panel)
        structural_geometry = modules["protocol"].structural_geometry_sha256(
            modules["common"], panel)
        reconstruction = panel["reconstruction"][
            "full_reconstruction_f64_sha256"]
        require(full_geometry ==
                pins.original_source_identity["source_full_geometry_sha256"],
                "external full source geometry")
        require(structural_geometry ==
                pins.original_source_identity["source_structural_geometry_sha256"],
                "external structural geometry")
        require(reconstruction ==
                pins.original_source_identity["reconstruction_f64_sha256"],
                "external reconstruction closure")

        publication = open_publication(stack, pins)
        held = publication["held"]
        result = strict_json(held["RESULT.json"].data, "RESULT")
        score = strict_json(held["BOUND_BASELINE_SCORE.json"].data,
                            "BOUND_BASELINE_SCORE", 1 << 20)
        preflight = strict_json(held["SOURCE_PREFLIGHT.json"].data,
                                "SOURCE_PREFLIGHT", 64 << 20)
        decoder = strict_json(held["DECODER_BUNDLE.json"].data,
                              "DECODER_BUNDLE", 1 << 20)
        for name, record in (
            ("RESULT.json", result),
            ("BOUND_BASELINE_SCORE.json", score),
            ("SOURCE_PREFLIGHT.json", preflight),
            ("DECODER_BUNDLE.json", decoder),
        ):
            require(held[name].data == pretty_json(record),
                    f"{name}: canonical pretty encoding")
        status = verify_claim_boundary(result, publication["complete"])
        verify_source_hashes(result, v9, v8)
        artifact_identity = exact_fields(result["artifact_identity"],
            {"st_dev", "st_ino", "bytes", "mtime_ns", "sha256"},
            "RESULT artifact identity")
        for name in ("st_dev", "st_ino", "bytes", "mtime_ns"):
            exact_int(artifact_identity[name], f"RESULT artifact {name}", 0,
                      (1 << 63) - 1)
        require(artifact_identity["bytes"] == artifact.before[3] ==
                KNOWN_ARTIFACT_BYTES and
                artifact_identity["sha256"] == artifact.sha256 ==
                KNOWN_ARTIFACT_SHA256,
                "RESULT artifact content identity")
        require(result["source_full_geometry_sha256"] == full_geometry and
                result["source_structural_geometry_sha256"] ==
                structural_geometry and
                result["recomputed_panel_reconstruction_f64_sha256"] ==
                reconstruction, "RESULT independently recomputed source closure")
        require(type(result["total_observed_launch_wall_seconds"]) is float and
                math.isfinite(result["total_observed_launch_wall_seconds"]) and
                result["total_observed_launch_wall_seconds"] >= 0.0,
                "RESULT observed wall telemetry")
        pipeline = verify_decoder_pipeline(
            result, decoder, v8, modules["bridge"], support_constants)
        verify_score(result, score, held["BOUND_BASELINE_SCORE.json"].data,
                     pins, panel, pipeline["decoder_sha256"])
        science = verify_scientific(
            result["scientific_primary_nested_holdout"], modules, panel)
        runtime = verify_preflight_runtime(
            result, preflight, modules, science, panel)
        verify_panel_cache(result)
        bindings = {
            "baseline_plan_sha256": support_constants["BASELINE_PLAN_SHA256"],
            "baseline_score_sha256": held["BOUND_BASELINE_SCORE.json"].sha256,
            "universal_decoder_sha256": pipeline["decoder_sha256"],
            "producer_manifest_sha256": KNOWN_V8_MANIFEST_SHA256,
            "audit_bootstrap_sha256": KNOWN_V9_RUNNER_SHA256,
            "source_full_geometry_sha256": full_geometry,
            "source_structural_geometry_sha256": structural_geometry,
            "extraction_program_sha256":
                v8["hashes"]["strata_sc_adapter.py"],
            "universal_adapter_sha256":
                v8["hashes"]["universal_adapter.py"],
            "pipeline_sha256": pipeline["pipeline_sha256"],
            "source_snapshot_root_sha256": KNOWN_V8_SOURCE_ROOT_SHA256,
            "source_preflight_receipt_sha256": preflight["receipt_sha256"],
        }
        typed_preflight = modules["stage"].SourcePreflightEvidence(
            preflight["all150"], preflight["representative"],
            preflight["independent_gpu_identity"], preflight["receipt_sha256"])
        typed_bindings = modules["stage"].BoundEvidence(**bindings)
        validated_preflight = modules["stage"].validate_source_preflight(
            modules["common"], modules["protocol"], typed_preflight,
            typed_bindings)
        require(validated_preflight["receipt_sha256"] ==
                preflight["receipt_sha256"],
                "sealed preflight validation replay")
        # A PASS receipt is impossible without exact selected-cell Q0.16
        # refits, lengths, aligned layout scores and final model bytes.
        selected_replay = replay_selected_models_cpu(
            result["scientific_primary_nested_holdout"], science, modules,
            panel, held["UWFCV8.bin"].data)
        physical = verify_physical(
            result, publication, modules, panel, pins, score, bindings, science)
        require(recompute_primary_status(
            result["scientific_primary_nested_holdout"], result["source_final"])
            == status, "terminal status after physical replay")

        protected: dict[tuple[int, int], str] = {
            (support.before[0], support.before[1]): "support",
            (strata.before[0], strata.before[1]): "STRATA common",
            (frozen.before[0], frozen.before[1]): "frozen auditor",
            (artifact.before[0], artifact.before[1]): "Qwen artifact",
        }
        for closure_label, closure in (("v9", v9), ("v8", v8)):
            for inode, name in closure["identities"].items():
                require(inode not in protected,
                        f"input inode alias {closure_label}:{name}/"
                        f"{protected.get(inode)}")
                protected[inode] = f"{closure_label}:{name}"
        require(not (set(protected) & set(publication["output_inodes"])),
                "input/output inode-domain alias")
        final_publication_rebind(publication, pins)
        support.verify_final()
        strata.verify_final()
        frozen.verify_final()
        artifact.verify_final()

        limitations = [
            {
                "dependency": "MATCHED_CONTROLS_AND_SHUFFLES",
                "status": "NOT_PRESENT_AND_NOT_RUN_BY_V9_PRIMARY_CONTRACT",
                "consequence": "even a primary survivor is nonpromoting and is not universal SwiGLU-MoE evidence",
            },
            {
                "dependency": "ALL_150_QWEN_GPU_FIT_OPTIMALITY",
                "status": "ORDERED_BANK_AND_EMITTED_SELECTED_CELL_COMMITMENTS_REPLAYED_BUT_ALTERNATIVE_Q016_TABLES_NOT_PUBLISHED",
                "consequence": "the CPU audit cannot prove no unreported alternative scored lower without repeating the full 38.6B-update search",
            },
            {
                "dependency": "IDENTITY_FRAMING_SEMANTIC_DECODE",
                "status": "IMPOSSIBLE_FOR_INTENTIONALLY_MISMATCHED_COUNTERFACTUAL",
                "consequence": "identity framing is audited only as canonical bytes and physical cost, never as a reconstruction",
            },
            {
                "dependency": "ORIGINAL_FP64_QWEN_SOURCE_TENSORS",
                "status": "EXTERNAL_SCORE_AND_SOURCE_IDENTITY_CLOSURE_REQUIRED",
                "consequence": "the artifact reproduces reconstruction but does not contain original tensors needed to recompute SSE and energy",
            },
            {
                "dependency": "AUDIT_NATIVE_FP64_SEMANTIC_INVERSE",
                "status": "INDEPENDENT_RAW_CONTAINER_ARITHMETIC_DECISION_AND_TRIPLET_REPLAY_WITH_AUTHENTICATED_PANEL_CONTEXTS; FP64_INVERSE_CROSS_CHECK_USES_FROZEN_DECODER",
                "consequence": "the audit independently parses every byte, refits Q0.16, decodes/re-encodes every selected arithmetic decision and binds every panel field/page; it does not duplicate the large STRATA polar/RHT FP64 inverse and instead requires its externally pinned full digest and frozen-decoder standalone/routed agreement",
            },
            {
                "dependency": "STRICT_HISTORICAL_COMPLETION_SYSCALL_ORDER",
                "status": "STATIC_FILES_CAN_ONLY_PROVE_COMPLETE_IS_NOT_OBSERVABLY_OLDER",
                "consequence": "external pins, exact set, completion seal, retained descriptors and final rebinding prove the audited snapshot; equal timestamp granularity cannot prove syscall history",
            },
        ]
        return {
            "schema": AUDIT_SCHEMA,
            "status": "PASS_FAIL_CLOSED_NONPROMOTING_PRIMARY_RESULT_AUDIT",
            "positive_claim_authority": False,
            "controls_run_by_this_audit": False,
            "shuffles_run_by_this_audit": False,
            "coordinate_diagnostic_run_by_this_audit": False,
            "primary_result_status": status,
            "publication_members": {
                name: {"bytes": len(held[name].data), "sha256": held[name].sha256}
                for name in sorted(PUBLICATION_MEMBERS)
            },
            "source_closure": {
                "v9_manifest_sha256": KNOWN_V9_MANIFEST_SHA256,
                "v9_source_root_sha256": KNOWN_V9_SOURCE_ROOT_SHA256,
                "v9_runner_sha256": KNOWN_V9_RUNNER_SHA256,
                "support_sha256": KNOWN_SUPPORT_SHA256,
                "v8_manifest_sha256": KNOWN_V8_MANIFEST_SHA256,
                "v8_source_root_sha256": KNOWN_V8_SOURCE_ROOT_SHA256,
                "strata_common_sha256": KNOWN_STRATA_COMMON_SHA256,
                "frozen_auditor_sha256": KNOWN_FROZEN_AUDITOR_SHA256,
                "artifact_sha256": KNOWN_ARTIFACT_SHA256,
                "artifact_bytes": KNOWN_ARTIFACT_BYTES,
            },
            "scientific_replay": {
                "dependence_components": [plan["identity_indices"]
                                          for plan in science["plans"]],
                "exact_primary_updates": science["workload"][
                    "exact_primary_updates"],
                "candidate_bank_cells": len(science["candidate_bank"]),
                "winner": science["winner"],
                "pooled_saving_bpw": science["pooled_saving_bpw"],
                "heldout_gate": science["heldout_gate"],
                "selected_q016_cpu_replay": selected_replay,
            },
            "runtime_replay": runtime,
            "literal_container_audit": physical,
            "evidence_limitations": limitations,
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--external-pins", required=True)
    result.add_argument("--expected-external-pins-sha256", required=True)
    result.add_argument("--expected-audit-source-manifest-sha256", required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    # The authorization and pins-file hash/path are validated before first
    # path access. The pins file itself is then retained through the audit.
    require(arguments.authorization == AUTHORIZATION,
            "explicit result-audit authorization")
    digest(arguments.expected_external_pins_sha256,
           "external pins file SHA-256")
    digest(arguments.expected_audit_source_manifest_sha256,
           "audit source manifest SHA-256")
    validate_absolute_path(arguments.external_pins, "external pins file",
                           directory=False)
    require(os.name == "posix", "result audit requires POSIX descriptors")
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "invoke result audit with CPython -I -B")
    with ExitStack() as stack:
        audit_source = authenticate_audit_source(
            stack, arguments.expected_audit_source_manifest_sha256)
        held_pins = stack.enter_context(HeldAbsoluteRegular(
            arguments.external_pins, 1 << 20, "external pins file",
            arguments.expected_external_pins_sha256))
        pins_record = strict_json(held_pins.data, "external pins", 1 << 20)
        require(held_pins.data == pretty_json(pins_record),
                "external pins canonical pretty encoding")
        pins = parse_pins(pins_record)
        receipt = verify(arguments.authorization, pins)
        held_pins.verify_final()
        receipt["audit_source_closure"] = audit_source
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_UWFA_SC_V9_PRIMARY_RESULT_AUDIT: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
