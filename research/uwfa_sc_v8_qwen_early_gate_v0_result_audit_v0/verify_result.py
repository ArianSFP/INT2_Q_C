#!/usr/bin/env python3
"""Fail-closed independent result audit for the exploratory Qwen early gate.

This module contains no resolved run/output pins.  The CLI refuses all path
access until an exact authorization string and every required external pin
have passed lexical validation.  It is intended to run as ``python3 -I -B``
in a fresh POSIX process after the producer has finished publishing.

The audit authenticates source and artifact inputs through retained no-follow
descriptors, authenticates the completion-last publication, parses hostile
JSON strictly, and rebinds every descriptor/name/byte snapshot after all
semantic work.  A candidate UWFCV8 container is independently parsed,
canonically rebuilt, causally decoded/re-encoded on CPU, reconstructed, and
measured from its literal bytes.  The identity-framing object is independently
parsed and rebuilt, but the sealed producer explicitly defines it as a byte-
cost counterfactual rather than a UWFA-coded object; no false semantic-decode
claim is manufactured for it.
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
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


AUTHORIZATION = "AUDIT_EXACT_QWEN_EARLY_GATE_RESULT_NO_PROMOTION_V0"
RESULT_SCHEMA = "uwfa-sc-v8-qwen-early-gate-v0"
COMPLETION_SCHEMA = "uwfa-sc-v8-qwen-early-gate-completion-v0"
SOURCE_SCHEMA = "uwfa-sc-v8-source-phase-result"
MAX_JSON_BYTES = 256 * (1 << 20)
MAX_SOURCE_BYTES = 2 * (1 << 20)
MAX_CONTAINER_BYTES = 512 * (1 << 20)
MAX_JSON_NODES = 2_000_000
MAX_JSON_DEPTH = 96
MAX_JSON_STRING = 2 * (1 << 20)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")

BASE_MEMBERS = {
    "BOUND_BASELINE_SCORE.json",
    "DECODER_BUNDLE.json",
    "RESULT.json",
    "SOURCE_PREFLIGHT.json",
    "COMPLETE.json",
}
CONTAINER_MEMBERS = {"UWFCV8.bin", "IDENTITY_FRAMING.bin"}

V8_REQUIRED_MEMBERS = (
    "INDEPENDENT_BOOTSTRAP_ABI.md",
    "README.md",
    "container_codec.py",
    "cupy_backend.py",
    "design_lock.json",
    "dispatcher_contract.py",
    "fixture_long_memory.py",
    "fixture_portability.py",
    "protocol.py",
    "result_envelope.py",
    "run_source_free_gpu_dev.py",
    "stage0_census.py",
    "strata_sc_adapter.py",
    "test_source_only.py",
    "universal_adapter.py",
    "uwfa_common.py",
    "verify_source.py",
)

RESULT_FIELDS = {
    "schema",
    "status",
    "underlying_exact_v8_source_status",
    "positive_claim_authority",
    "controls_run",
    "controls_may_not_be_inferred_or_added",
    "claim_boundary",
    "binding_authority_disclosure",
    "artifact_identity",
    "baseline_score",
    "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "recomputed_panel_reconstruction_f64_sha256",
    "winner",
    "pooled_exact_heldout_saving_bpw",
    "per_dependence_component_saving",
    "physical",
    "bandwidth",
    "canonical_decode_reencode",
    "source_preflight_receipt_sha256",
    "telemetry",
    "bindings",
    "decoder_bundle",
    "decoder_bundle_sha256",
    "exploratory_panel_cache",
    "pipeline_record",
    "pipeline_sha256",
    "source_hashes",
    "exact_v8_source_result",
}

BINDING_FIELDS = (
    "baseline_plan_sha256",
    "baseline_score_sha256",
    "universal_decoder_sha256",
    "producer_manifest_sha256",
    "audit_bootstrap_sha256",
    "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "extraction_program_sha256",
    "universal_adapter_sha256",
    "pipeline_sha256",
    "source_snapshot_root_sha256",
    "source_preflight_receipt_sha256",
)

POST_FIT_STATUSES = {
    "FAIL_EVIDENCE_INTEGRITY_SOURCE_STANDALONE_DECODE",
    "HARD_KILL_PHYSICAL_RATE_OR_F",
    "FAIL_STRICT_COLD_READ",
    "NO_PROMOTION_NESTED_HELDOUT",
    "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED",
}
PRE_FIT_STATUSES = {
    "NO_PROMOTION_UNESTIMABLE_EXACT_IDENTITY_HOLDOUT",
    "ABORT_RESOURCE_BUDGET_BEFORE_BACKEND_PACK",
    "ABORT_RUNTIME_BUDGET_BEFORE_FIT",
}


class ResultAuditError(RuntimeError):
    """A fail-closed audit rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultAuditError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{label}: lowercase SHA-256")
    return value


def exact_int(value: Any, label: str, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    require(type(value) is int and minimum <= value <= maximum, f"{label}: exact integer bound")
    return value


def safe_name(value: Any, label: str) -> str:
    require(isinstance(value, str) and SAFE_NAME.fullmatch(value) is not None, f"{label}: safe name")
    require(value not in {".", ".."}, f"{label}: reserved name")
    return value


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ResultAuditError(f"noncanonical JSON value: {exc}") from exc


def pretty_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ResultAuditError(f"noncanonical pretty JSON value: {exc}") from exc


def strict_json(data: bytes, label: str, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    require(isinstance(data, bytes) and len(data) <= maximum_bytes, f"{label}: JSON byte bound")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def finite(text: str) -> float:
        value = float(text)
        require(math.isfinite(value), f"{label}: nonfinite JSON float")
        return value

    def reject(text: str) -> None:
        raise ResultAuditError(f"{label}: nonfinite JSON constant {text}")

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_float=finite,
            parse_constant=reject,
        )
    except ResultAuditError:
        raise
    except Exception as exc:
        raise ResultAuditError(f"{label}: invalid JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label}: JSON root object")
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        require(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH, f"{label}: JSON complexity bound")
        if isinstance(item, dict):
            for key, child in item.items():
                require(isinstance(key, str) and len(key) <= MAX_JSON_STRING, f"{label}: JSON key bound")
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            require(len(item) <= MAX_JSON_STRING, f"{label}: JSON string bound")
        elif isinstance(item, float):
            require(math.isfinite(item), f"{label}: finite float tree")
        else:
            require(item is None or isinstance(item, (bool, int)), f"{label}: JSON scalar type")
    return value


def exact_fields(record: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(record, dict) and set(record) == set(fields), f"{label}: exact fields")
    return record


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def require_deep_equal(left: Any, right: Any, label: str) -> None:
    """Strict recursive equality, including types and binary64 sign/encoding."""
    stack: list[tuple[Any, Any, str]] = [(left, right, label)]
    while stack:
        observed, expected, path = stack.pop()
        require(type(observed) is type(expected), f"{path}: type mismatch")
        if isinstance(expected, dict):
            require(set(observed) == set(expected), f"{path}: field mismatch")
            stack.extend((observed[key], expected[key], f"{path}.{key}") for key in expected)
        elif isinstance(expected, list):
            require(len(observed) == len(expected), f"{path}: list length")
            stack.extend((a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(observed, expected)))
        elif isinstance(expected, float):
            require(math.isfinite(observed) and _float_bits(observed) == _float_bits(expected), f"{path}: binary64 mismatch")
        else:
            require(observed == expected, f"{path}: value mismatch")


def stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


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
    """No-follow absolute POSIX directory walk retaining every component."""

    def __init__(self, path: str, label: str) -> None:
        require(os.name == "posix", "descriptor audit requires POSIX openat/pread semantics")
        require(isinstance(path, str) and path.startswith("/") and not path.endswith("/"), f"{label}: canonical absolute path")
        require("//" not in path, f"{label}: duplicate separator")
        parts = path.split("/")[1:]
        require(parts and all(part and part not in {".", ".."} for part in parts), f"{label}: canonical components")
        self.path = path
        self.label = label
        self.fds: list[int] = []
        self.names: list[str] = []
        self.identities: list[tuple[int, int, int, int, int, int, int]] = []
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        self.fds.append(current)
        self.identities.append(stat_identity(os.fstat(current)))
        try:
            for part in parts:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
                info = os.fstat(child)
                require(stat.S_ISDIR(info.st_mode), f"{label}: component directory")
                self.names.append(part)
                self.fds.append(child)
                self.identities.append(stat_identity(info))
                current = child
        except Exception:
            self.close(verify=False)
            raise

    @property
    def fd(self) -> int:
        return self.fds[-1]

    def verify_final(self) -> None:
        for index, (fd, expected) in enumerate(zip(self.fds, self.identities)):
            require(stat_identity(os.fstat(fd)) == expected, f"{self.label}: held directory changed [{index}]")
            if index:
                named = os.stat(self.names[index - 1], dir_fd=self.fds[index - 1], follow_symlinks=False)
                require(stat_identity(named) == expected, f"{self.label}: directory name rebound [{index}]")

    def close(self, *, verify: bool = True) -> None:
        pending: BaseException | None = None
        if verify and self.fds:
            try:
                self.verify_final()
            except BaseException as exc:  # preserve cleanup on hostile mutation
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
    """A bounded regular child retained through final name/inode/byte rebind."""

    def __init__(self, parent_fd: int, name: str, cap: int, label: str, expected_sha256: str | None = None) -> None:
        self.parent_fd = parent_fd
        self.name = safe_name(name, label)
        self.cap = cap
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.name, flags, dir_fd=parent_fd)
        try:
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode), f"{label}: regular file")
            self.before = stat_identity(info)
            self.data = pread_exact(self.fd, info.st_size, cap, label)
            self.sha256 = sha256(self.data)
            if expected_sha256 is not None:
                require(self.sha256 == digest(expected_sha256, f"{label} expected"), f"{label}: digest mismatch")
            self.verify_final()
        except Exception:
            os.close(self.fd)
            self.fd = -1
            raise

    def verify_final(self) -> None:
        require(self.fd >= 0, f"{self.label}: closed descriptor")
        require(stat_identity(os.fstat(self.fd)) == self.before, f"{self.label}: held descriptor changed")
        named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        require(stat_identity(named) == self.before, f"{self.label}: name/inode rebound")
        require(pread_exact(self.fd, self.before[3], self.cap, self.label) == self.data, f"{self.label}: held bytes changed")

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
    def __init__(self, path: str, cap: int, label: str, expected_sha256: str) -> None:
        require(isinstance(path, str), f"{label}: path text")
        pure = PurePosixPath(path)
        require(pure.is_absolute() and pure.name not in {"", ".", ".."}, f"{label}: canonical absolute file")
        require(str(pure) == path and "//" not in path, f"{label}: normalized absolute file")
        self.parent = RetainedDirectory(str(pure.parent), f"{label} parent")
        try:
            self.held = HeldRegularAt(self.parent.fd, pure.name, cap, label, expected_sha256)
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
    runner_path: str
    runner_sha256: str
    v8_package: str
    v8_manifest_sha256: str
    v8_source_root_sha256: str
    strata_common_path: str
    strata_common_sha256: str
    frozen_auditor_path: str
    frozen_auditor_sha256: str
    artifact_path: str
    artifact_sha256: str
    artifact_bytes: int
    complete_file_sha256: str
    result_file_sha256: str
    baseline_score_sha256: str
    source_preflight_sha256: str


def validate_pins(authorization: str, pins: Pins) -> None:
    """Pure validation: this is deliberately called before the first stat/open."""
    require(authorization == AUTHORIZATION, "explicit result-audit authorization")
    require(os.name == "posix", "result audit requires POSIX descriptor semantics")
    safe_name(pins.final_name, "final name")
    for label, path, directory in (
        ("output parent", pins.output_parent, True),
        ("early-gate runner", pins.runner_path, False),
        ("v8 package", pins.v8_package, True),
        ("STRATA common", pins.strata_common_path, False),
        ("frozen auditor", pins.frozen_auditor_path, False),
        ("Qwen artifact", pins.artifact_path, False),
    ):
        require(isinstance(path, str) and path.startswith("/") and "//" not in path, f"{label}: absolute lexical path")
        pure = PurePosixPath(path)
        require(str(pure) == path and pure.is_absolute(), f"{label}: canonical lexical path")
        if not directory:
            require(pure.name not in {"", ".", ".."}, f"{label}: file leaf")
    for label, value in (
        ("runner", pins.runner_sha256),
        ("v8 manifest", pins.v8_manifest_sha256),
        ("v8 source root", pins.v8_source_root_sha256),
        ("STRATA common", pins.strata_common_sha256),
        ("frozen auditor", pins.frozen_auditor_sha256),
        ("Qwen artifact", pins.artifact_sha256),
        ("COMPLETE file", pins.complete_file_sha256),
        ("RESULT file", pins.result_file_sha256),
        ("baseline score", pins.baseline_score_sha256),
        ("source preflight", pins.source_preflight_sha256),
    ):
        digest(value, f"expected {label}")
    exact_int(pins.artifact_bytes, "expected artifact bytes", 1, 1 << 34)


def literal_assignments(source: bytes, names: set[str], label: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source, filename=f"<authenticated:{label}>")
    except SyntaxError as exc:
        raise ResultAuditError(f"{label}: source syntax") from exc
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                require(name not in values, f"{label}: duplicate literal {name}")
                try:
                    values[name] = ast.literal_eval(node.value)
                except Exception as exc:
                    raise ResultAuditError(f"{label}: nonliteral {name}") from exc
    require(set(values) == names, f"{label}: required literal assignments")
    return values


def load_module(name: str, source: bytes, expected_sha256: str) -> types.ModuleType:
    require(sha256(source) == expected_sha256, f"{name}: authenticated module digest")
    require(name not in sys.modules, f"{name}: module namespace collision")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated-result-audit:{name}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(source, module.__file__, "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def normalize_strata_group_ordinals(strata_common: Any) -> dict[str, Any]:
    original = getattr(strata_common, "expected_block_group_ordinals", None)
    require(callable(original), "STRATA expected-block-group entrypoint")

    def normalized(labels: Any) -> list[list[int]]:
        rows = original(labels)
        require(isinstance(rows, list), "STRATA group rows")
        converted: list[list[int]] = []
        for row in rows:
            require(hasattr(row, "__iter__"), "STRATA group row iterable")
            # Preserve the original one-axis NumPy advanced-index semantics.
            # A tuple of integers would instead mean one index per axis.
            values = [int(value) for value in row]
            require(all(type(value) is int for value in values), "STRATA group Python-int ABI")
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


def authenticate_v8_package(stack: ExitStack, pins: Pins) -> dict[str, Any]:
    package = stack.enter_context(RetainedDirectory(pins.v8_package, "sealed v8 package"))
    manifest_held = stack.enter_context(
        HeldRegularAt(package.fd, "SOURCE_MANIFEST.json", 1 << 20, "sealed v8 manifest", pins.v8_manifest_sha256)
    )
    manifest = strict_json(manifest_held.data, "sealed v8 manifest", 1 << 20)
    exact_fields(
        manifest,
        {"schema", "status", "members", "access_attestation", "post_freeze_requirements"},
        "sealed v8 manifest",
    )
    require(manifest["schema"] == "unifilar-wfa-source-manifest-v8", "sealed v8 manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "sealed v8 manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(V8_REQUIRED_MEMBERS), "sealed v8 manifest rows")
    require(
        [row.get("name") if isinstance(row, dict) else None for row in rows] == list(V8_REQUIRED_MEMBERS),
        "sealed v8 member order",
    )
    require(
        {entry.name for entry in os.scandir(package.fd)} == set(V8_REQUIRED_MEMBERS) | {"SOURCE_MANIFEST.json"},
        "sealed v8 exact member set",
    )
    sources: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    identities: dict[tuple[int, int], str] = {(manifest_held.before[0], manifest_held.before[1]): "SOURCE_MANIFEST.json"}
    for row in rows:
        exact_fields(row, {"name", "bytes", "sha256"}, "sealed v8 member row")
        name = safe_name(row["name"], "sealed v8 member name")
        size = exact_int(row["bytes"], f"sealed v8 member bytes {name}", 1, MAX_SOURCE_BYTES)
        expected = digest(row["sha256"], f"sealed v8 member digest {name}")
        held = stack.enter_context(HeldRegularAt(package.fd, name, MAX_SOURCE_BYTES, f"sealed v8 member {name}", expected))
        require(held.before[3] == size, f"sealed v8 member size {name}")
        inode = (held.before[0], held.before[1])
        require(inode not in identities, f"sealed v8 inode alias {name}/{identities.get(inode)}")
        identities[inode] = name
        sources[name] = held.data
        hashes[name] = held.sha256
    source_root = sha256(canonical_json(rows))
    require(source_root == pins.v8_source_root_sha256, "sealed v8 externally pinned source root")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_held.sha256,
        "source_root_sha256": source_root,
        "sources": sources,
        "hashes": hashes,
        "identities": identities,
    }


def load_authenticated_modules(closure: Mapping[str, Any], strata_source: bytes, strata_hash: str, auditor_source: bytes, auditor_hash: str) -> dict[str, Any]:
    # NumPy is imported only after authorization and all source/artifact pins
    # have authenticated.  CuPy/CUDA is intentionally not used by this audit.
    import numpy as np

    tag = sha256(canonical_json({"root": closure["source_root_sha256"], "runner": strata_hash, "audit": auditor_hash}))[:16]
    sources = closure["sources"]
    hashes = closure["hashes"]
    common = load_module(f"uwfa_result_audit_{tag}_common", sources["uwfa_common.py"], hashes["uwfa_common.py"])
    protocol = load_module(f"uwfa_result_audit_{tag}_protocol", sources["protocol.py"], hashes["protocol.py"])
    semantic = load_module(f"uwfa_result_audit_{tag}_semantic", sources["universal_adapter.py"], hashes["universal_adapter.py"])
    codec = load_module(f"uwfa_result_audit_{tag}_codec", sources["container_codec.py"], hashes["container_codec.py"])
    stage = load_module(f"uwfa_result_audit_{tag}_stage", sources["stage0_census.py"], hashes["stage0_census.py"])
    adapter_source = load_module(f"uwfa_result_audit_{tag}_adapter", sources["strata_sc_adapter.py"], hashes["strata_sc_adapter.py"])
    strata = load_module(f"uwfa_result_audit_{tag}_strata", strata_source, strata_hash)
    frozen = load_module(f"uwfa_result_audit_{tag}_frozen", auditor_source, auditor_hash)
    bridge = normalize_strata_group_ordinals(strata)
    adapter = adapter_source.StrataSCAdapter(
        common=common,
        semantic_codec=semantic,
        np=np,
        frozen_auditor=frozen,
        strata_common=strata,
        device="numpy",
    )
    return {
        "np": np,
        "common": common,
        "protocol": protocol,
        "semantic": semantic,
        "codec": codec,
        "stage": stage,
        "adapter_source": adapter_source,
        "strata": strata,
        "frozen": frozen,
        "bridge": bridge,
        "adapter": adapter,
    }


def verify_internal_seal(record: Mapping[str, Any], field: str, label: str) -> str:
    require(isinstance(record, dict), f"{label}: object")
    claimed = digest(record.get(field), f"{label} {field}")
    clean = dict(record)
    clean.pop(field)
    require(sha256(canonical_json(clean)) == claimed, f"{label}: internal seal")
    return claimed


def verify_completion_record(complete: dict[str, Any], actual_members: set[str], observed: Mapping[str, "HeldRegularAt"], expected_source_root: str) -> None:
    exact_fields(
        complete,
        {"schema", "status", "positive_claim_authority", "controls_run", "source_snapshot_root_sha256", "members", "completion_sha256"},
        "COMPLETE",
    )
    require(complete["schema"] == COMPLETION_SCHEMA, "COMPLETE schema")
    require(complete["positive_claim_authority"] is False and complete["controls_run"] is False, "COMPLETE nonpromoting counters")
    require(complete["source_snapshot_root_sha256"] == expected_source_root, "COMPLETE source root")
    verify_internal_seal(complete, "completion_sha256", "COMPLETE")
    rows = complete["members"]
    expected_without_complete = actual_members - {"COMPLETE.json"}
    require(isinstance(rows, list), "COMPLETE member rows")
    require(
        [row.get("name") if isinstance(row, dict) else None for row in rows]
        == sorted(expected_without_complete, key=lambda value: value.encode("utf-8")),
        "COMPLETE canonical exact member order",
    )
    for row in rows:
        exact_fields(row, {"name", "bytes", "sha256"}, "COMPLETE member row")
        name = safe_name(row["name"], "COMPLETE member name")
        require(name in observed, f"COMPLETE undeclared observed member {name}")
        held = observed[name]
        require(type(row["bytes"]) is int and row["bytes"] == len(held.data), f"COMPLETE bytes {name}")
        require(row["sha256"] == held.sha256, f"COMPLETE digest {name}")


def open_publication(stack: ExitStack, pins: Pins) -> dict[str, Any]:
    parent = stack.enter_context(RetainedDirectory(pins.output_parent, "output parent"))
    final_fd = os.open(
        pins.final_name,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent.fd,
    )
    stack.callback(os.close, final_fd)
    final_info = os.fstat(final_fd)
    require(stat.S_ISDIR(final_info.st_mode), "final output directory")
    final_identity = stat_identity(final_info)
    actual = {entry.name for entry in os.scandir(final_fd)}
    require(actual in (BASE_MEMBERS, BASE_MEMBERS | CONTAINER_MEMBERS), "publication exact member set")
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
    # COMPLETE is opened first but all member descriptors remain held until the
    # final rebind after parsing/decoding.
    order = ["COMPLETE.json"] + sorted(actual - {"COMPLETE.json"}, key=lambda value: value.encode("utf-8"))
    for name in order:
        expected_digest = (
            pins.complete_file_sha256 if name == "COMPLETE.json"
            else pins.result_file_sha256 if name == "RESULT.json"
            else None
        )
        item = stack.enter_context(HeldRegularAt(final_fd, name, caps[name], f"publication member {name}", expected_digest))
        require(item.before[6] == 1, f"publication member sole link {name}")
        inode = (item.before[0], item.before[1])
        require(inode not in output_inodes, f"publication member inode alias {name}/{output_inodes.get(inode)}")
        output_inodes[inode] = name
        held[name] = item
    complete = strict_json(held["COMPLETE.json"].data, "COMPLETE", 1 << 20)
    require(held["COMPLETE.json"].data == pretty_json(complete), "COMPLETE canonical pretty encoding")
    verify_completion_record(complete, actual, held, pins.v8_source_root_sha256)
    return {
        "parent": parent,
        "final_fd": final_fd,
        "final_identity": final_identity,
        "actual_members": actual,
        "held": held,
        "complete": complete,
        "output_inodes": output_inodes,
    }


def final_publication_rebind(publication: Mapping[str, Any], pins: Pins) -> None:
    for held in publication["held"].values():
        held.verify_final()
    final_fd = publication["final_fd"]
    require(stat_identity(os.fstat(final_fd)) == publication["final_identity"], "final directory descriptor changed")
    named = os.stat(pins.final_name, dir_fd=publication["parent"].fd, follow_symlinks=False)
    require(stat_identity(named) == publication["final_identity"], "final directory name/inode rebound")
    require({entry.name for entry in os.scandir(final_fd)} == publication["actual_members"], "final member set changed")
    publication["parent"].verify_final()


def expected_wrapper_status(source_status: str) -> str:
    if source_status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED":
        return "EARLY_DIAGNOSTIC_SOURCE_SURVIVOR_REQUIRES_CONTROLS_AND_INDEPENDENT_AUDIT"
    return f"EARLY_DIAGNOSTIC_{source_status}"


def verify_claim_boundary(result: dict[str, Any], complete: dict[str, Any]) -> str:
    exact_fields(result, RESULT_FIELDS, "RESULT")
    require(result["schema"] == RESULT_SCHEMA, "RESULT schema")
    source_status = result["underlying_exact_v8_source_status"]
    require(isinstance(source_status, str) and source_status in POST_FIT_STATUSES | PRE_FIT_STATUSES, "RESULT source status")
    wrapper = expected_wrapper_status(source_status)
    require(result["status"] == wrapper and complete["status"] == wrapper, "RESULT/COMPLETE status binding")
    require(result["positive_claim_authority"] is False, "RESULT positive-claim boundary")
    require(result["controls_run"] is False, "RESULT controls counter")
    require(result["controls_may_not_be_inferred_or_added"] is True, "RESULT controls inference boundary")
    require(
        result["claim_boundary"] == "early-kill Qwen diagnostic using exact sealed v8 source; never a positive compression claim",
        "RESULT claim-boundary text",
    )
    disclosure = exact_fields(
        result["binding_authority_disclosure"],
        {
            "status",
            "externally_pinned_dispatcher_receipt_present",
            "baseline_score_receipt",
            "decoder_bundle_sha256",
            "audit_bootstrap_sha256",
            "pipeline_sha256",
            "positive_claim_use_permitted",
        },
        "binding authority disclosure",
    )
    require(disclosure["status"] == "EARLY_DIAGNOSTIC_LOCAL_BINDINGS_NOT_PRODUCTION_DISPATCHER_AUTHORITY", "binding disclosure status")
    require(disclosure["externally_pinned_dispatcher_receipt_present"] is False, "dispatcher receipt counter")
    require(disclosure["positive_claim_use_permitted"] is False, "positive-claim use counter")
    require(disclosure["baseline_score_receipt"] == "constructed locally from the fixed audited D/SSE/energy and recomputed artifact identities", "baseline-score authority disclosure")
    require(disclosure["decoder_bundle_sha256"] == "canonical aggregate of the exact hash-pinned decoder source members", "decoder-bundle authority disclosure")
    require(disclosure["audit_bootstrap_sha256"] == "self-reported hash of this unsealed exploratory runner", "runner authority disclosure")
    require(disclosure["pipeline_sha256"] == "canonical aggregate constructed by this exploratory runner", "pipeline authority disclosure")
    return source_status


def verify_nonpromotion_counters(value: Any, label: str = "RESULT") -> None:
    stack: list[tuple[Any, str]] = [(value, label)]
    must_be_false = {
        "positive_claim_authority",
        "positive_claim_use_permitted",
        "positive_promotion",
        "controls_run",
        "controls_opened",
    }
    while stack:
        item, path = stack.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                if key in must_be_false:
                    require(child is False, f"{child_path}: nonpromoting counter must be false")
                stack.append((child, child_path))
        elif isinstance(item, list):
            stack.extend((child, f"{path}[{index}]") for index, child in enumerate(item))


def fraction_from_record(record: Any, label: str) -> Fraction:
    exact_fields(record, {"numerator", "denominator", "exact", "float"}, label)
    numerator = exact_int(record["numerator"], f"{label} numerator", -(1 << 63), (1 << 63) - 1)
    denominator = exact_int(record["denominator"], f"{label} denominator", 1, (1 << 63) - 1)
    value = Fraction(numerator, denominator)
    require(record["exact"] == f"{value.numerator}/{value.denominator}", f"{label}: exact text")
    require(type(record["float"]) is float and _float_bits(record["float"]) == _float_bits(float(value)), f"{label}: float projection")
    return value


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "float": float(value),
    }


def bandwidth_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    maximum_page = Fraction(0, 1)
    maximum_repeated = Fraction(0, 1)
    maximum_coalesced = Fraction(0, 1)
    for expected_ordinal, row in enumerate(metrics["experts"]):
        require(row["expert_ordinal"] == expected_ordinal, "physical expert canonical order")
        total = fraction_from_record(row["attributable_total_physical_bytes"], "attributable total")
        nonpadding = fraction_from_record(row["attributable_nonpadding_decodable_bytes"], "attributable nonpadding")
        require(total > 0 and nonpadding > 0, "positive bandwidth denominator")
        touched = exact_int(row["touched_page_bytes"], "touched page bytes")
        repeated_bytes = exact_int(row["instrumented_routed_requested_bytes_with_repetition"], "repeated requested bytes")
        unique_bytes = exact_int(row["instrumented_routed_unique_requested_bytes"], "unique requested bytes")
        page = max(Fraction(touched, 1) / total, Fraction(touched, 1) / nonpadding)
        repeated = max(Fraction(repeated_bytes, 1) / total, Fraction(repeated_bytes, 1) / nonpadding)
        coalesced = max(Fraction(unique_bytes, 1) / total, Fraction(unique_bytes, 1) / nonpadding)
        maximum_page = max(maximum_page, page)
        maximum_repeated = max(maximum_repeated, repeated)
        maximum_coalesced = max(maximum_coalesced, coalesced)
        rows.append({
            "expert_ordinal": expected_ordinal,
            "descriptor_backed_unique_page_ratio_strict": fraction_record(page),
            "requested_with_repetition_ratio_strict": fraction_record(repeated),
            "ideal_coalesced_unique_requested_ratio_strict": fraction_record(coalesced),
            "touched_page_bytes": touched,
            "requested_bytes_with_repetition": repeated_bytes,
            "unique_requested_bytes": unique_bytes,
            "overlap_bytes_requested_again": int(row["instrumented_routed_overlap_bytes_requested_again"]),
            "read_request_count": int(row["instrumented_routed_read_request_count"]),
            "causal_decode_reencode_reconstruction": row["causal_decode_reencode_reconstruction"],
            "passes_descriptor_backed_unique_page_below_2x": page < 2,
            "passes_requested_with_repetition_below_2x": repeated < 2,
            "passes_ideal_coalesced_unique_requested_below_2x": coalesced < 2,
            "passes_all_reported_bandwidth_ratios_below_2x": page < 2 and repeated < 2 and coalesced < 2,
        })
    return {
        "definition": {
            "unique_page": "union of descriptor-backed 4096-byte pages touched divided by the stricter owner-local denominator",
            "requested_with_repetition": "sum of literal read-call lengths including overlap divided by the stricter owner-local denominator",
            "ideal_coalesced_unique_requested": "union of requested byte intervals divided by the stricter owner-local denominator; diagnostic, not the frozen cold gate",
        },
        "experts": rows,
        "maximum_descriptor_backed_unique_page_ratio_strict": fraction_record(maximum_page),
        "maximum_requested_with_repetition_ratio_strict": fraction_record(maximum_repeated),
        "maximum_ideal_coalesced_unique_requested_ratio_strict": fraction_record(maximum_coalesced),
        "passes_frozen_unique_page_below_2x": maximum_page < 2,
        "passes_strict_requested_with_repetition_below_2x": maximum_repeated < 2,
        "passes_strict_ideal_coalesced_unique_requested_below_2x": maximum_coalesced < 2,
        "passes_all_reported_bandwidth_ratios_below_2x": maximum_page < 2 and maximum_repeated < 2 and maximum_coalesced < 2,
    }


def compact_component_rows(scientific: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    folds = scientific.get("folds", [])
    require(isinstance(folds, list), "scientific folds list")
    for row in folds:
        require(isinstance(row, dict), "scientific fold object")
        result.append({
            "component_ordinal": row.get("outer_dependence_component_ordinal"),
            "identity_indices": row.get("outer_identity_indices"),
            "identities": row.get("outer_identities_from_artifact"),
            "test_stream_ordinals": row.get("test_stream_ordinals"),
            "allocated_test_weights": row.get("allocated_test_weights"),
            "selected": row.get("selected_by_inner_validation_only"),
            "literal_baseline_bits": row.get("literal_authenticated_current_baseline_container_bits"),
            "literal_candidate_bits": row.get("literal_candidate_container_bits"),
            "literal_model_aligned_increment_bits": row.get("literal_selected_model_aligned_increment_bits"),
            "literal_saved_bits": row.get("literal_test_saving_after_exact_container_delta_bits"),
            "exact_saving_bpw": row.get("exact_test_saving_bpw"),
        })
    return result


def verify_scientific_counters(scientific: dict[str, Any], modules: Mapping[str, Any], panel: Mapping[str, Any]) -> bool:
    exact_fields(
        scientific,
        {
            "kind", "primary_policy", "status", "folds", "skipped_folds", "estimable",
            "pooled_exact_heldout_saving_bpw", "minimum_fold_exact_saving_bpw",
            "dependence_component_mean_saving_bpw_diagnostic_only", "confidence_rule",
            "independent_component_count", "all_dependence_components_positive",
            "leave_one_component_out_pooled_saving_bpw_diagnostic_only", "candidate_vote_counts",
            "final_topology_selected_from_nested_fold_votes", "passes_pooled_standalone_threshold",
            "passes_every_disjoint_component_positive", "passes_heldout_gate", "positive_promotion",
        },
        "scientific nested holdout",
    )
    require(scientific["kind"] == "disjoint_stream_owner_dependence_component_holdout", "scientific kind")
    require(scientific["primary_policy"] == "exact_identity", "scientific policy")
    require(scientific["status"] == "PASS_DISJOINT_DEPENDENCE_COMPONENT_HOLDOUT", "scientific status")
    require(scientific["estimable"] is True and scientific["positive_promotion"] is False, "scientific estimable/nonpromoting counters")
    require(scientific["skipped_folds"] == [], "scientific skipped folds")
    require(
        scientific["confidence_rule"]
        == "no iid confidence interval; promotion requires disjoint owner-stream components, pooled literal saving at target, and every component strictly positive",
        "scientific confidence rule",
    )
    plans = modules["stage"]._component_fold_plan(modules["common"], modules["protocol"], panel)
    valid_plans = [plan for plan in plans if plan["estimable"]]
    require(len(valid_plans) == len(plans) >= 2, "scientific independently estimable component plans")
    folds = scientific["folds"]
    require(isinstance(folds, list) and len(folds) == len(valid_plans), "scientific fold count")
    fold_fields = {
        "outer_dependence_component_ordinal", "outer_identity_indices", "outer_identities_from_artifact",
        "development_exclusion_policy", "test_stream_ordinals", "development_stream_ordinals",
        "inner_train_stream_ordinals", "inner_validation_stream_ordinals", "selected_by_inner_validation_only",
        "inner_validation_exact_charged_bits", "literal_authenticated_current_baseline_container_bits",
        "literal_candidate_container_bits", "literal_selected_model_aligned_increment_bits",
        "literal_test_saving_after_exact_container_delta_bits", "allocated_test_weights",
        "allocated_baseline_bits", "allocated_candidate_bits", "exact_test_saving_bpw",
    }
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    candidate_bank = {row.selector_ordinal: row.as_dict() for row in modules["common"].candidate_bank()}
    baseline_bits = modules["stage"].literal_current_baseline_score(modules["protocol"], modules["codec"], panel)
    allocated_total = 0.0
    pooled_saved_bits = 0.0
    values: list[float] = []
    votes: dict[int, int] = {}
    for ordinal, (fold, plan) in enumerate(zip(folds, valid_plans, strict=True)):
        exact_fields(fold, fold_fields, f"scientific fold[{ordinal}]")
        require(fold["outer_dependence_component_ordinal"] == int(plan["component_ordinal"]), f"scientific fold[{ordinal}] component")
        identity_indices = [int(value) for value in plan["identity_indices"]]
        require_deep_equal(fold["outer_identity_indices"], identity_indices, f"scientific fold[{ordinal}] identities")
        require_deep_equal(fold["outer_identities_from_artifact"], [list(identities[index]) for index in identity_indices], f"scientific fold[{ordinal}] semantic identities")
        require(fold["development_exclusion_policy"] == "exact_identity", f"scientific fold[{ordinal}] policy")
        for field, plan_field in (
            ("test_stream_ordinals", "test_indices"),
            ("development_stream_ordinals", "development_indices"),
            ("inner_train_stream_ordinals", "train_indices"),
            ("inner_validation_stream_ordinals", "validation_indices"),
        ):
            expected_ordinals = [int(streams[index]["stream_ordinal"]) for index in plan[plan_field]]
            require_deep_equal(fold[field], expected_ordinals, f"scientific fold[{ordinal}] {field}")
        selected = fold["selected_by_inner_validation_only"]
        require(isinstance(selected, dict) and selected.get("selector_ordinal") in candidate_bank, f"scientific fold[{ordinal}] selected candidate")
        require_deep_equal(selected, candidate_bank[selected["selector_ordinal"]], f"scientific fold[{ordinal}] candidate fields")
        selected_ordinal = int(selected["selector_ordinal"])
        votes[selected_ordinal] = votes.get(selected_ordinal, 0) + 1
        require(fold["literal_authenticated_current_baseline_container_bits"] == baseline_bits, f"scientific fold[{ordinal}] literal baseline")
        literal_candidate = exact_int(fold["literal_candidate_container_bits"], f"scientific fold[{ordinal}] literal candidate", 1, 1 << 50)
        literal_saved = baseline_bits - literal_candidate
        require(fold["literal_test_saving_after_exact_container_delta_bits"] == literal_saved, f"scientific fold[{ordinal}] literal saving")
        expected_weights = float(sum(int(streams[index]["weight_charge"]) for index in plan["test_indices"]))
        require(type(fold["allocated_test_weights"]) is float and _float_bits(fold["allocated_test_weights"]) == _float_bits(expected_weights), f"scientific fold[{ordinal}] allocated weights")
        expected_baseline_allocated = float(sum(8 * int(streams[index]["baseline_payload_bytes"]) for index in plan["test_indices"]))
        require(type(fold["allocated_baseline_bits"]) is float and _float_bits(fold["allocated_baseline_bits"]) == _float_bits(expected_baseline_allocated), f"scientific fold[{ordinal}] allocated baseline bits")
        require(type(fold["allocated_candidate_bits"]) is float and math.isfinite(fold["allocated_candidate_bits"]) and fold["allocated_candidate_bits"] > 0.0, f"scientific fold[{ordinal}] allocated candidate bits")
        expected_saving = literal_saved / expected_weights
        require(type(fold["exact_test_saving_bpw"]) is float and _float_bits(fold["exact_test_saving_bpw"]) == _float_bits(expected_saving), f"scientific fold[{ordinal}] exact saving")
        exact_int(fold["inner_validation_exact_charged_bits"], f"scientific fold[{ordinal}] validation bits", 1, 1 << 50)
        exact_int(fold["literal_selected_model_aligned_increment_bits"], f"scientific fold[{ordinal}] model increment", -(1 << 50), 1 << 50)
        allocated_total += expected_weights
        pooled_saved_bits += float(literal_saved)
        values.append(expected_saving)
    require(abs(allocated_total - int(panel["weights"])) <= 1e-6, "scientific folds partition source weights")
    pooled = pooled_saved_bits / allocated_total
    require(_float_bits(scientific["pooled_exact_heldout_saving_bpw"]) == _float_bits(pooled), "scientific pooled saving")
    require(_float_bits(scientific["minimum_fold_exact_saving_bpw"]) == _float_bits(min(values)), "scientific minimum saving")
    require(_float_bits(scientific["dependence_component_mean_saving_bpw_diagnostic_only"]) == _float_bits(statistics.fmean(values)), "scientific mean diagnostic")
    component_positive = all(value > 0.0 for value in values)
    require(scientific["all_dependence_components_positive"] is component_positive, "scientific component-positive counter")
    require(scientific["passes_every_disjoint_component_positive"] is component_positive, "scientific component-positive gate")
    threshold_pass = pooled >= modules["common"].STANDALONE_REQUIRED_SAVING_BPW
    require(scientific["passes_pooled_standalone_threshold"] is threshold_pass, "scientific pooled threshold")
    expected_gate = threshold_pass and component_positive
    require(scientific["passes_heldout_gate"] is expected_gate, "scientific heldout gate")
    require(scientific["independent_component_count"] == len(folds), "scientific independent component count")
    leave_one = []
    for omitted, fold in enumerate(folds):
        kept_weights = allocated_total - float(fold["allocated_test_weights"])
        kept_bits = pooled_saved_bits - float(fold["literal_test_saving_after_exact_container_delta_bits"])
        leave_one.append({"omitted_component_ordinal": omitted, "pooled_saving_bpw": kept_bits / kept_weights})
    require_deep_equal(scientific["leave_one_component_out_pooled_saving_bpw_diagnostic_only"], leave_one, "scientific leave-one-component-out")
    expected_votes = {str(key): value for key, value in votes.items()}
    require_deep_equal(scientific["candidate_vote_counts"], expected_votes, "scientific candidate votes")
    selected_ordinal = min(votes, key=lambda value: (-votes[value], value))
    require_deep_equal(scientific["final_topology_selected_from_nested_fold_votes"], candidate_bank[selected_ordinal], "scientific final candidate vote")
    return expected_gate


def audit_literal_container(
    modules: Mapping[str, Any],
    held: HeldRegularAt,
    *,
    semantic_decode: bool,
    label: str,
) -> dict[str, Any]:
    codec = modules["codec"]
    common = modules["common"]
    semantic = modules["semantic"]
    adapter = modules["adapter"]
    raw = held.data
    parsed = codec.parse_container(common, semantic, raw)
    require(isinstance(parsed, dict) and bytes(parsed.get("raw", b"")) == raw, f"{label}: parser literal-byte rebind")
    rebuilt = codec.canonical_rebuild(common, semantic, parsed)
    require(rebuilt == raw, f"{label}: canonical rebuild mismatch")
    if semantic_decode:
        standalone = adapter.decode_new_container(parsed)
        descriptor_source = codec.AuthenticatedDescriptorSource(held.fd, held.sha256)
        try:
            metrics = codec.physical_metrics(
                common,
                semantic,
                parsed,
                routed_descriptor_source=descriptor_source,
                externally_authenticated_container_sha256=held.sha256,
                routed_decoder=adapter.new_routed_decoder(),
            )
        finally:
            descriptor_source.close()
        require(standalone["all_payloads_canonically_reencoded"] is True, f"{label}: canonical re-encode")
        require(standalone["all_three_roles_reconstructed"] is True, f"{label}: role reconstruction")
    else:
        # Identity framing deliberately carries original arithmetic payloads
        # beside the selected candidate model.  The sealed producer calls only
        # parse + non-authoritative physical_metrics for this counterfactual.
        standalone = None
        metrics = codec.physical_metrics(common, semantic, parsed)
    return {
        "parsed": parsed,
        "standalone": standalone,
        "metrics": metrics,
        "canonical_rebuild_sha256": sha256(rebuilt),
    }


def verify_decoder_and_pipeline(
    result: dict[str, Any],
    decoder_file: dict[str, Any],
    closure: Mapping[str, Any],
    pins: Pins,
    runner_constants: Mapping[str, Any],
    bridge: Mapping[str, Any],
) -> dict[str, Any]:
    expected_decoder = {
        "schema": "uwfa-sc-v8-qwen-early-gate-decoder-bundle-v0",
        "members": [
            {"name": "strata_expert_local_codec/common.py", "sha256": pins.strata_common_sha256},
            {"name": "strata_v2_klt_mixed_independent_auditor_v1.py", "sha256": pins.frozen_auditor_sha256},
            {"name": "strata_sc_adapter.py", "sha256": closure["hashes"]["strata_sc_adapter.py"]},
            {"name": "universal_adapter.py", "sha256": closure["hashes"]["universal_adapter.py"]},
            {"name": "container_codec.py", "sha256": closure["hashes"]["container_codec.py"]},
        ],
        "exploratory_semantic_bridge": dict(bridge),
    }
    require_deep_equal(decoder_file, expected_decoder, "DECODER_BUNDLE")
    require_deep_equal(result["decoder_bundle"], expected_decoder, "RESULT.decoder_bundle")
    decoder_sha = sha256(canonical_json(expected_decoder))
    require(result["decoder_bundle_sha256"] == decoder_sha, "decoder bundle aggregate")
    pipeline = exact_fields(
        result["pipeline_record"],
        {"schema", "sealed_v8_manifest_sha256", "source_snapshot_root_sha256", "decoder_bundle_sha256", "runner_sha256", "baseline_plan_sha256"},
        "pipeline record",
    )
    require(pipeline["schema"] == "uwfa-sc-v8-qwen-early-gate-pipeline-v0", "pipeline schema")
    require(pipeline["sealed_v8_manifest_sha256"] == pins.v8_manifest_sha256, "pipeline v8 manifest")
    require(pipeline["source_snapshot_root_sha256"] == pins.v8_source_root_sha256, "pipeline v8 source root")
    require(pipeline["decoder_bundle_sha256"] == decoder_sha, "pipeline decoder bundle")
    require(pipeline["runner_sha256"] == pins.runner_sha256, "pipeline runner")
    require(pipeline["baseline_plan_sha256"] == runner_constants["BASELINE_PLAN_SHA256"], "pipeline baseline plan")
    pipeline_sha = sha256(canonical_json(pipeline))
    require(result["pipeline_sha256"] == pipeline_sha, "pipeline aggregate")
    return {"decoder_bundle_sha256": decoder_sha, "pipeline_sha256": pipeline_sha}


def verify_score_and_bindings(
    result: dict[str, Any],
    score: dict[str, Any],
    score_bytes: bytes,
    preflight: dict[str, Any],
    preflight_bytes: bytes,
    panel: Mapping[str, Any],
    modules: Mapping[str, Any],
    closure: Mapping[str, Any],
    pins: Pins,
    runner_constants: Mapping[str, Any],
    pipeline: Mapping[str, str],
) -> dict[str, Any]:
    require(score_bytes == pretty_json(score), "BOUND_BASELINE_SCORE canonical pretty encoding")
    require(preflight_bytes == pretty_json(preflight), "SOURCE_PREFLIGHT canonical pretty encoding")
    require(sha256(score_bytes) == pins.baseline_score_sha256, "externally pinned baseline score file")
    require(sha256(preflight_bytes) == pins.source_preflight_sha256, "externally pinned source preflight file")
    require_deep_equal(result["baseline_score"], score, "RESULT baseline score copy")
    exact_fields(
        score,
        {
            "schema", "status", "artifact_sha256", "artifact_bytes", "weights", "relative_mse", "sse_fp64",
            "source_energy_fp64", "normalization", "reconstruction_f64_sha256", "original_source_panel_sha256",
            "independent_decoder_source_sha256", "score_receipt_sha256",
        },
        "baseline score",
    )
    require(score["schema"] == "uwfa-bound-baseline-score-v8" and score["status"] == "PASS_INDEPENDENT_BASELINE_SCORE", "baseline score schema/status")
    verify_internal_seal(score, "score_receipt_sha256", "baseline score")
    require(score["artifact_sha256"] == pins.artifact_sha256 and score["artifact_bytes"] == pins.artifact_bytes, "baseline score artifact binding")
    require(score["weights"] == runner_constants["SOURCE_WEIGHTS"], "baseline score weight binding")
    for field in ("AUDITED_RELATIVE_MSE", "AUDITED_SSE_FP64", "AUDITED_SOURCE_ENERGY_FP64"):
        require(type(runner_constants[field]) is float and math.isfinite(runner_constants[field]), f"runner {field}")
    require(_float_bits(score["relative_mse"]) == _float_bits(runner_constants["AUDITED_RELATIVE_MSE"]), "baseline relative MSE pin")
    require(_float_bits(score["sse_fp64"]) == _float_bits(runner_constants["AUDITED_SSE_FP64"]), "baseline SSE pin")
    require(_float_bits(score["source_energy_fp64"]) == _float_bits(runner_constants["AUDITED_SOURCE_ENERGY_FP64"]), "baseline energy pin")
    require(score["normalization"] == "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM", "baseline normalization")
    expected_ratio = float(score["sse_fp64"]) / float(score["source_energy_fp64"])
    require(abs(expected_ratio - float(score["relative_mse"])) <= 4.0 * math.ulp(expected_ratio), "baseline score arithmetic")

    reconstruction = digest(panel["reconstruction"]["full_reconstruction_f64_sha256"], "panel reconstruction")
    full_geometry = result["source_full_geometry_sha256"]
    require(score["reconstruction_f64_sha256"] == reconstruction, "score/panel reconstruction")
    require(score["original_source_panel_sha256"] == full_geometry, "score/full geometry")
    require(score["independent_decoder_source_sha256"] == pipeline["decoder_bundle_sha256"], "score/decoder bundle")
    modules["protocol"].validate_score_receipt(
        score,
        artifact_sha256=pins.artifact_sha256,
        artifact_bytes=pins.artifact_bytes,
        weights=int(panel["weights"]),
        reconstruction_sha256=reconstruction,
        original_source_panel_sha256=full_geometry,
        independent_decoder_source_sha256=pipeline["decoder_bundle_sha256"],
    )

    exact_fields(preflight, {"schema", "source_snapshot_root_sha256", "all150", "representative", "independent_gpu_identity", "receipt_sha256"}, "source preflight")
    require(preflight["schema"] == "uwfa-sc-v8-bound-source-preflight", "source preflight schema")
    require(preflight["source_snapshot_root_sha256"] == pins.v8_source_root_sha256, "source preflight root")
    verify_internal_seal(preflight, "receipt_sha256", "source preflight")
    require(result["source_preflight_receipt_sha256"] == preflight["receipt_sha256"], "RESULT source preflight seal")

    bindings = exact_fields(result["bindings"], set(BINDING_FIELDS), "RESULT bindings")
    for field in BINDING_FIELDS:
        digest(bindings[field], f"binding {field}")
    expected = {
        "baseline_plan_sha256": runner_constants["BASELINE_PLAN_SHA256"],
        "baseline_score_sha256": pins.baseline_score_sha256,
        "universal_decoder_sha256": pipeline["decoder_bundle_sha256"],
        "producer_manifest_sha256": pins.v8_manifest_sha256,
        "audit_bootstrap_sha256": pins.runner_sha256,
        "source_full_geometry_sha256": result["source_full_geometry_sha256"],
        "source_structural_geometry_sha256": result["source_structural_geometry_sha256"],
        "extraction_program_sha256": closure["hashes"]["strata_sc_adapter.py"],
        "universal_adapter_sha256": closure["hashes"]["universal_adapter.py"],
        "pipeline_sha256": pipeline["pipeline_sha256"],
        "source_snapshot_root_sha256": pins.v8_source_root_sha256,
        "source_preflight_receipt_sha256": preflight["receipt_sha256"],
    }
    require_deep_equal(bindings, expected, "RESULT binding commitments")
    typed_preflight = modules["stage"].SourcePreflightEvidence(
        preflight["all150"],
        preflight["representative"],
        preflight["independent_gpu_identity"],
        preflight["receipt_sha256"],
    )
    typed_bindings = modules["stage"].BoundEvidence(**bindings)
    validated_preflight = modules["stage"].validate_source_preflight(
        modules["common"], modules["protocol"], typed_preflight, typed_bindings,
    )
    require(validated_preflight["receipt_sha256"] == preflight["receipt_sha256"], "sealed preflight validator receipt")
    return {
        "reconstruction_sha256": reconstruction,
        "full_geometry_sha256": result["source_full_geometry_sha256"],
        "structural_geometry_sha256": result["source_structural_geometry_sha256"],
        "bindings": bindings,
    }


def expected_physical_summary(metrics: Mapping[str, Any], source_final: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "container_bytes": metrics["actual_container_bytes"],
        "physical_rate_bpw": metrics["actual_physical_rate_bpw"],
        "physical_rate_rational": metrics["actual_physical_rate_rational"],
        "relative_mse": metrics["audited_identical_reconstruction_relative_mse"],
        "F": metrics["F_from_actual_bytes_and_identical_reconstruction"],
        "net_physical_saving_bpw": metrics["net_physical_saving_bpw"],
        "passes_rate_interval": metrics["passes_rate_interval"],
        "passes_F_target": metrics["passes_F_target"],
        "passes_descriptor_backed_cold_below_2x": metrics["passes_cold_read_below_2x"],
        "container_sha256": source_final["container_sha256"],
        "identity_framing_container_sha256": source_final["identity_framing_container_sha256"],
        "model_packet_sha256": source_final["model_packet_sha256"],
    }


def verify_container_commitments(
    result: dict[str, Any],
    source: dict[str, Any],
    publication: Mapping[str, Any],
    modules: Mapping[str, Any],
    panel: Mapping[str, Any],
    score: Mapping[str, Any],
    pins: Pins,
) -> dict[str, Any]:
    candidate = audit_literal_container(modules, publication["held"]["UWFCV8.bin"], semantic_decode=True, label="UWFCV8")
    identity = audit_literal_container(modules, publication["held"]["IDENTITY_FRAMING.bin"], semantic_decode=False, label="IDENTITY_FRAMING")
    parsed = candidate["parsed"]
    identity_parsed = identity["parsed"]
    metrics = candidate["metrics"]
    identity_metrics = identity["metrics"]
    standalone = candidate["standalone"]
    source_final = source["source_final"]
    require(isinstance(source_final, dict), "source final object")
    exact_fields(
        source_final,
        {
            "parsed_metrics", "identity_framing_metrics", "absolute_saving_vs_bound_current_artifact_bpw",
            "incremental_same_framing_WFA_saving_bpw", "raw_payload_minus_full_model_saving_bpw", "payload_rows",
            "standalone_decode", "model_packet_sha256", "container_sha256", "identity_framing_container_sha256",
            "candidate", "posterior_diagnostic_handoff", "all_adapted_values_deserialized_from_transmitted_model",
            "identical_reconstruction_proved_by_full_f64_digest",
        },
        "source final",
    )

    for label, item in (("UWFCV8", parsed), ("IDENTITY_FRAMING", identity_parsed)):
        require(item["baseline_artifact_sha256"] == pins.artifact_sha256, f"{label}: artifact commitment")
        require(item["weights"] == panel["weights"], f"{label}: source weights")
        require(item["experts"] == panel["experts"], f"{label}: expert count")
        require(item["reconstruction_sha256"] == panel["reconstruction"]["full_reconstruction_f64_sha256"], f"{label}: reconstruction commitment")
        require(item["audit_binding_sha256"] == pins.baseline_score_sha256, f"{label}: score commitment")
        require_deep_equal(item["binding_hashes"], result["bindings"], f"{label}: header bindings")
    for key in ("semantic_packet", "immutable_state", "model_packet"):
        require(bytes(parsed[key]) == bytes(identity_parsed[key]), f"identity/candidate shared {key}")
    require(len(parsed["directory"]) == len(identity_parsed["directory"]) == len(panel["streams"]), "candidate/identity/panel stream count")
    shared_directory_fields = (
        "ordinal", "symbols", "source_weights", "group_rows", "group_cols", "profile_q", "decoder_scale",
        "owner_set_hex", "owners", "source_digest", "role", "owner_contributions",
    )
    for ordinal, (candidate_row, identity_row, panel_row) in enumerate(zip(parsed["directory"], identity_parsed["directory"], panel["streams"], strict=True)):
        require(candidate_row["ordinal"] == identity_row["ordinal"] == ordinal, "candidate/identity ordinal")
        for field in shared_directory_fields:
            require_deep_equal(candidate_row[field], identity_row[field], f"candidate/identity directory[{ordinal}].{field}")
        require(identity_row["logical_bits"] == panel_row["baseline_logical_bits"], f"identity baseline logical bits [{ordinal}]")
        require(bytes(identity_row["payload"]) == bytes(panel_row["baseline_payload"]), f"identity baseline payload [{ordinal}]")
    require(
        standalone["reconstruction"]["full_reconstruction_f64_sha256"]
        == panel["reconstruction"]["full_reconstruction_f64_sha256"],
        "candidate standalone/panel reconstruction",
    )
    require(metrics["actual_container_bytes"] == len(publication["held"]["UWFCV8.bin"].data), "candidate literal byte count")
    exact_rate = Fraction(8 * len(publication["held"]["UWFCV8.bin"].data), int(panel["weights"]))
    require(fraction_from_record(metrics["actual_physical_rate_rational"], "candidate rate") == exact_rate, "candidate literal rate rational")
    require(_float_bits(metrics["actual_physical_rate_bpw"]) == _float_bits(float(exact_rate)), "candidate literal rate float")
    expected_f = float(score["relative_mse"]) * math.pow(2.0, 2.0 * float(exact_rate))
    require(_float_bits(metrics["F_from_actual_bytes_and_identical_reconstruction"]) == _float_bits(expected_f), "candidate literal F")
    require(metrics["routed_io_authoritative_descriptor_backed"] is True, "candidate descriptor-backed bandwidth")

    require(source_final["container_sha256"] == publication["held"]["UWFCV8.bin"].sha256, "source-final candidate hash")
    require(source_final["identity_framing_container_sha256"] == publication["held"]["IDENTITY_FRAMING.bin"].sha256, "source-final identity hash")
    require(source_final["model_packet_sha256"] == sha256(bytes(parsed["model_packet"])), "source-final model hash")
    require_deep_equal(source_final["parsed_metrics"], metrics, "source-final physical metrics")
    require_deep_equal(source_final["identity_framing_metrics"], identity_metrics, "source-final identity metrics")
    require_deep_equal(source_final["standalone_decode"], standalone, "source-final standalone decode")
    handoff = modules["codec"].posterior_diagnostic_handoff(modules["common"], parsed)
    require_deep_equal(source_final["posterior_diagnostic_handoff"], handoff, "source-final decision commitments")
    require(source_final["candidate"] == parsed["candidate"].as_dict(), "source-final deserialized model")

    payload_rows = []
    for candidate_row, panel_row in zip(parsed["directory"], panel["streams"], strict=True):
        payload_rows.append({
            "ordinal": int(panel_row["stream_ordinal"]),
            "baseline_payload_bytes": int(panel_row["baseline_payload_bytes"]),
            "new_payload_bytes": len(bytes(candidate_row["payload"])),
            "baseline_logical_bits": int(panel_row["baseline_logical_bits"]),
            "new_logical_bits": int(candidate_row["logical_bits"]),
        })
    require_deep_equal(source_final["payload_rows"], payload_rows, "source-final payload rows")
    weights = int(panel["weights"])
    candidate_size = len(publication["held"]["UWFCV8.bin"].data)
    identity_size = len(publication["held"]["IDENTITY_FRAMING.bin"].data)
    raw_saving = sum(8 * (row["baseline_payload_bytes"] - row["new_payload_bytes"]) for row in payload_rows) - 8 * len(bytes(parsed["model_packet"]))
    expected_scalars = {
        "absolute_saving_vs_bound_current_artifact_bpw": 8.0 * (pins.artifact_bytes - candidate_size) / weights,
        "incremental_same_framing_WFA_saving_bpw": 8.0 * (identity_size - candidate_size) / weights,
        "raw_payload_minus_full_model_saving_bpw": raw_saving / weights,
    }
    for field, value in expected_scalars.items():
        require(type(source_final[field]) is float and _float_bits(source_final[field]) == _float_bits(value), f"source-final {field}")
    require(source_final["all_adapted_values_deserialized_from_transmitted_model"] is True, "source-final model decode flag")
    require(source_final["identical_reconstruction_proved_by_full_f64_digest"] is True, "source-final reconstruction flag")

    physical = expected_physical_summary(metrics, source_final)
    require_deep_equal(result["physical"], physical, "RESULT compact physical")
    bandwidth = bandwidth_summary(metrics)
    require_deep_equal(result["bandwidth"], bandwidth, "RESULT bandwidth")
    canonical = {
        "standalone_all_payloads_canonically_reencoded": standalone["all_payloads_canonically_reencoded"],
        "standalone_all_three_roles_reconstructed": standalone["all_three_roles_reconstructed"],
        "full_reconstruction_f64_sha256": standalone["reconstruction"]["full_reconstruction_f64_sha256"],
        "matches_recomputed_panel_reconstruction": standalone["reconstruction"]["full_reconstruction_f64_sha256"] == panel["reconstruction"]["full_reconstruction_f64_sha256"],
        "routed_full_reconstruction": metrics["routed_full_reconstruction"],
        "literal_container_canonical_rebuild_was_enforced_by_exact_v8_final_container": True,
    }
    require_deep_equal(result["canonical_decode_reencode"], canonical, "RESULT canonical decode/re-encode")
    return {
        "candidate_sha256": publication["held"]["UWFCV8.bin"].sha256,
        "identity_sha256": publication["held"]["IDENTITY_FRAMING.bin"].sha256,
        "model_packet_sha256": sha256(bytes(parsed["model_packet"])),
        "directory_sha256": sha256(bytes(parsed["directory_blob"])),
        "identity_directory_sha256": sha256(bytes(identity_parsed["directory_blob"])),
        "decision_commitment_sha256": handoff["decoded_sc_decision_triplet_commitment_sha256"],
        "reconstruction_sha256": standalone["reconstruction"]["full_reconstruction_f64_sha256"],
        "rate": fraction_record(exact_rate),
        "F": expected_f,
        "physical_pass": bool(metrics["passes_rate_interval"] and metrics["passes_F_target"]),
        "cold_pass": bool(metrics["passes_cold_read_below_2x"]),
        "identity_semantic_decode": "IMPOSSIBLE_FROM_THIS_COUNTERFACTUAL_BY_SEALED_V8_ABI",
    }


def classify_decision(source_status: str, physical_pass: bool, cold_pass: bool, heldout_pass: bool, integrity_pass: bool) -> dict[str, Any]:
    if not integrity_pass:
        expected = "FAIL_EVIDENCE_INTEGRITY_SOURCE_STANDALONE_DECODE"
    elif not physical_pass:
        expected = "HARD_KILL_PHYSICAL_RATE_OR_F"
    elif not cold_pass:
        expected = "FAIL_STRICT_COLD_READ"
    elif not heldout_pass:
        expected = "NO_PROMOTION_NESTED_HELDOUT"
    else:
        expected = "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED"
    require(source_status == expected, "source decision does not follow sealed gate ordering")
    if expected == "HARD_KILL_PHYSICAL_RATE_OR_F":
        return {
            "classification": "VERIFIED_HARD_KILL_FINAL_REGARDLESS_OF_CONTROLS",
            "controls_required_before_any_positive_claim": False,
            "positive_claim_authority": False,
        }
    if expected == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED":
        return {
            "classification": "VERIFIED_SOURCE_SURVIVOR_CONTROLS_AND_FRESH_INDEPENDENT_AUDIT_REQUIRED",
            "controls_required_before_any_positive_claim": True,
            "positive_claim_authority": False,
        }
    return {
        "classification": "VERIFIED_NON_SURVIVOR_NONPROMOTING_DIAGNOSTIC",
        "controls_required_before_any_positive_claim": False,
        "positive_claim_authority": False,
    }


def verify_source_result(
    result: dict[str, Any],
    source_status: str,
    publication: Mapping[str, Any],
    modules: Mapping[str, Any],
    panel: Mapping[str, Any],
    score: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pins: Pins,
) -> dict[str, Any]:
    source = result["exact_v8_source_result"]
    require(isinstance(source, dict), "exact v8 source result object")
    require(source.get("schema") == SOURCE_SCHEMA and source.get("status") == source_status, "exact v8 source schema/status")
    runtime_projection = modules["stage"].projected_updates(modules["common"], modules["protocol"], panel)
    require_deep_equal(source.get("runtime_projection"), runtime_projection, "source runtime projection")
    if source_status in PRE_FIT_STATUSES:
        require(publication["actual_members"] == BASE_MEMBERS, "pre-fit abort must not emit containers")
        if not runtime_projection["primary_exact_identity_estimable"]:
            expected = "NO_PROMOTION_UNESTIMABLE_EXACT_IDENTITY_HOLDOUT"
        elif not runtime_projection["passes_pre_fit_resource_budget"]:
            expected = "ABORT_RESOURCE_BUDGET_BEFORE_BACKEND_PACK"
        elif not runtime_projection["passes_pre_fit_runtime_budget"]:
            expected = "ABORT_RUNTIME_BUDGET_BEFORE_FIT"
        else:
            raise ResultAuditError("pre-fit status claimed although all pre-fit gates pass")
        require(source_status == expected, "pre-fit decision ordering")
        require(result["physical"] is None and result["bandwidth"] is None and result["canonical_decode_reencode"] is None, "pre-fit physical absence")
        require(result["winner"] is None and result["pooled_exact_heldout_saving_bpw"] is None and result["per_dependence_component_saving"] == [], "pre-fit scientific summary absence")
        require(source.get("controls_may_be_opened") is False, "pre-fit controls boundary")
        if "positive_promotion" in source:
            require(source["positive_promotion"] is False, "pre-fit promotion counter")
        return {
            "decision": {
                "classification": "VERIFIED_PRE_FIT_ABORT_NONPROMOTING_DIAGNOSTIC",
                "controls_required_before_any_positive_claim": False,
                "positive_claim_authority": False,
            },
            "container": None,
        }

    require(publication["actual_members"] == BASE_MEMBERS | CONTAINER_MEMBERS, "post-fit exact container member set")
    post_fields = {
        "schema", "status", "source_full_geometry_sha256", "source_structural_geometry_sha256", "source_pipeline_sha256",
        "source_artifact_sha256", "score_receipt_sha256", "source_preflight_receipt_sha256", "source_preflight_summary",
        "runtime_projection", "scientific_nested_holdout", "coordinate_disjoint_nonpromoting_diagnostic",
        "predeclared_shuffle_diagnostics", "source_final", "source_phase_elapsed_seconds", "controls_may_be_opened",
        "physical_Qwen_failure_is_final_regardless_of_controls", "claim_boundary",
        "requires_external_fresh_process_independent_result_audit",
    }
    exact_fields(source, post_fields, "post-fit exact v8 source result")
    require(source["source_artifact_sha256"] == pins.artifact_sha256, "source result artifact")
    require(source["score_receipt_sha256"] == pins.baseline_score_sha256, "source result score")
    require(source["source_preflight_receipt_sha256"] == preflight["receipt_sha256"], "source result preflight")
    expected_preflight_summary = {
        "source_snapshot_root_sha256": preflight["source_snapshot_root_sha256"],
        "all150_status": preflight["all150"]["status"],
        "representative_status": preflight["representative"]["status"],
        "device_uuid": preflight["independent_gpu_identity"]["device_uuid"],
        "pci_bus_id": preflight["independent_gpu_identity"]["pci_bus_id"],
    }
    require_deep_equal(source["source_preflight_summary"], expected_preflight_summary, "source preflight summary")
    require(source["source_pipeline_sha256"] == result["pipeline_sha256"], "source result pipeline")
    require(source["source_full_geometry_sha256"] == result["source_full_geometry_sha256"], "source full geometry copy")
    require(source["source_structural_geometry_sha256"] == result["source_structural_geometry_sha256"], "source structural geometry copy")
    require(type(source["source_phase_elapsed_seconds"]) is float and math.isfinite(source["source_phase_elapsed_seconds"]) and source["source_phase_elapsed_seconds"] >= 0.0, "source elapsed telemetry")
    require(source["physical_Qwen_failure_is_final_regardless_of_controls"] is True, "physical failure final boundary")
    require(source["requires_external_fresh_process_independent_result_audit"] is True, "external result audit counter")
    require(source["claim_boundary"] == "frozen selected-SC-decision recoder only; Qwen is an evaluation panel, not a universal performance proof", "source claim boundary")

    scientific = source["scientific_nested_holdout"]
    require(isinstance(scientific, dict), "scientific nested holdout object")
    require_deep_equal(result["winner"], scientific.get("final_topology_selected_from_nested_fold_votes"), "RESULT winner projection")
    require_deep_equal(result["pooled_exact_heldout_saving_bpw"], scientific.get("pooled_exact_heldout_saving_bpw"), "RESULT pooled saving projection")
    require_deep_equal(result["per_dependence_component_saving"], compact_component_rows(scientific), "RESULT component projection")
    heldout_pass = verify_scientific_counters(scientific, modules, panel)

    container = verify_container_commitments(result, source, publication, modules, panel, score, pins)
    standalone = container["reconstruction_sha256"] == panel["reconstruction"]["full_reconstruction_f64_sha256"]
    integrity_pass = bool(
        standalone
        and source["source_final"]["standalone_decode"]["all_payloads_canonically_reencoded"]
        and source["source_final"]["identical_reconstruction_proved_by_full_f64_digest"]
        and source["source_final"]["all_adapted_values_deserialized_from_transmitted_model"]
    )
    decision = classify_decision(source_status, container["physical_pass"], container["cold_pass"], heldout_pass, integrity_pass)
    require(source["controls_may_be_opened"] is (source_status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED"), "source controls-may-open decision")
    return {"decision": decision, "container": container}


def verify_source_hash_disclosures(result: Mapping[str, Any], closure: Mapping[str, Any], pins: Pins) -> None:
    source_hashes = exact_fields(
        result["source_hashes"],
        {
            "sealed_v8_manifest_sha256", "sealed_v8_source_snapshot_root_sha256", "sealed_v8_members",
            "strata_expert_local_codec_common_sha256", "strata_v2_klt_mixed_independent_auditor_sha256",
            "early_gate_runner_sha256",
        },
        "RESULT source hashes",
    )
    expected = {
        "sealed_v8_manifest_sha256": pins.v8_manifest_sha256,
        "sealed_v8_source_snapshot_root_sha256": pins.v8_source_root_sha256,
        "sealed_v8_members": closure["hashes"],
        "strata_expert_local_codec_common_sha256": pins.strata_common_sha256,
        "strata_v2_klt_mixed_independent_auditor_sha256": pins.frozen_auditor_sha256,
        "early_gate_runner_sha256": pins.runner_sha256,
    }
    require_deep_equal(source_hashes, expected, "RESULT authenticated source hashes")


def verify(authorization: str, pins: Pins) -> dict[str, Any]:
    validate_pins(authorization, pins)
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "invoke verifier with CPython -I -B")
    with ExitStack() as stack:
        runner = stack.enter_context(HeldAbsoluteRegular(pins.runner_path, MAX_SOURCE_BYTES, "early-gate runner", pins.runner_sha256))
        constants = literal_assignments(
            runner.data,
            {
                "SCHEMA", "AUTHORIZATION", "SEALED_V8_MANIFEST_SHA256", "STRATA_COMMON_SHA256",
                "FROZEN_AUDITOR_SHA256", "CURRENT_ARTIFACT_SHA256", "CURRENT_ARTIFACT_BYTES",
                "SOURCE_WEIGHTS", "BASELINE_PLAN_SHA256", "AUDITED_RELATIVE_MSE", "AUDITED_SSE_FP64",
                "AUDITED_SOURCE_ENERGY_FP64",
            },
            "early-gate runner",
        )
        require(constants["SCHEMA"] == RESULT_SCHEMA, "runner/result schema")
        require(constants["SEALED_V8_MANIFEST_SHA256"] == pins.v8_manifest_sha256, "runner/pinned v8 manifest")
        require(constants["STRATA_COMMON_SHA256"] == pins.strata_common_sha256, "runner/pinned STRATA common")
        require(constants["FROZEN_AUDITOR_SHA256"] == pins.frozen_auditor_sha256, "runner/pinned frozen auditor")
        require(constants["CURRENT_ARTIFACT_SHA256"] == pins.artifact_sha256, "runner/pinned Qwen artifact")
        require(constants["CURRENT_ARTIFACT_BYTES"] == pins.artifact_bytes, "runner/pinned Qwen artifact bytes")

        closure = authenticate_v8_package(stack, pins)
        strata = stack.enter_context(HeldAbsoluteRegular(pins.strata_common_path, MAX_SOURCE_BYTES, "STRATA common source", pins.strata_common_sha256))
        frozen = stack.enter_context(HeldAbsoluteRegular(pins.frozen_auditor_path, MAX_SOURCE_BYTES, "frozen auditor source", pins.frozen_auditor_sha256))
        artifact = stack.enter_context(HeldAbsoluteRegular(pins.artifact_path, pins.artifact_bytes, "Qwen artifact", pins.artifact_sha256))
        require(artifact.before[3] == pins.artifact_bytes, "Qwen artifact exact bytes")

        modules = load_authenticated_modules(closure, strata.data, strata.sha256, frozen.data, frozen.sha256)
        panel = modules["adapter"].extract_from_current(artifact.data)
        require(isinstance(panel, dict), "independent artifact panel")
        full_geometry = modules["protocol"].geometry_sha256(modules["common"], panel)
        structural_geometry = modules["protocol"].structural_geometry_sha256(modules["common"], panel)
        require(panel["artifact"]["raw_sha256"] == pins.artifact_sha256 and panel["artifact"]["raw_bytes"] == pins.artifact_bytes, "independent panel artifact binding")

        publication = open_publication(stack, pins)
        held = publication["held"]
        result = strict_json(held["RESULT.json"].data, "RESULT")
        score = strict_json(held["BOUND_BASELINE_SCORE.json"].data, "BOUND_BASELINE_SCORE", 1 << 20)
        preflight = strict_json(held["SOURCE_PREFLIGHT.json"].data, "SOURCE_PREFLIGHT", 64 << 20)
        decoder = strict_json(held["DECODER_BUNDLE.json"].data, "DECODER_BUNDLE", 1 << 20)
        for name, record in (
            ("RESULT.json", result),
            ("BOUND_BASELINE_SCORE.json", score),
            ("SOURCE_PREFLIGHT.json", preflight),
            ("DECODER_BUNDLE.json", decoder),
        ):
            require(held[name].data == pretty_json(record), f"{name}: canonical pretty encoding")
        source_status = verify_claim_boundary(result, publication["complete"])
        verify_nonpromotion_counters(result)
        require(result["source_full_geometry_sha256"] == full_geometry, "RESULT independently recomputed full geometry")
        require(result["source_structural_geometry_sha256"] == structural_geometry, "RESULT independently recomputed structural geometry")
        require(result["recomputed_panel_reconstruction_f64_sha256"] == panel["reconstruction"]["full_reconstruction_f64_sha256"], "RESULT independently recomputed reconstruction")
        artifact_identity = exact_fields(result["artifact_identity"], {"st_dev", "st_ino", "bytes", "mtime_ns", "sha256"}, "RESULT artifact identity")
        require(all(type(artifact_identity[field]) is int for field in ("st_dev", "st_ino", "bytes", "mtime_ns")), "RESULT artifact identity integers")
        require(artifact_identity["bytes"] == pins.artifact_bytes and artifact_identity["sha256"] == pins.artifact_sha256, "RESULT artifact identity hash/bytes")

        pipeline = verify_decoder_and_pipeline(result, decoder, closure, pins, constants, modules["bridge"])
        score_bindings = verify_score_and_bindings(
            result, score, held["BOUND_BASELINE_SCORE.json"].data, preflight,
            held["SOURCE_PREFLIGHT.json"].data, panel, modules, closure, pins, constants, pipeline,
        )
        require(score_bindings["full_geometry_sha256"] == full_geometry, "score binding independent full geometry")
        require(score_bindings["structural_geometry_sha256"] == structural_geometry, "binding independent structural geometry")
        verify_source_hash_disclosures(result, closure, pins)
        source_audit = verify_source_result(result, source_status, publication, modules, panel, score, preflight, pins)

        panel_cache = exact_fields(
            result["exploratory_panel_cache"],
            {"schema", "status", "artifact_bytes", "artifact_sha256", "extract_calls", "delegate_extract_calls", "same_panel_object_reused", "positive_claim_authority", "receipt_sha256"},
            "panel-cache receipt",
        )
        verify_internal_seal(panel_cache, "receipt_sha256", "panel-cache receipt")
        require(panel_cache["schema"] == "uwfa-sc-v8-qwen-early-gate-single-artifact-panel-cache-v0", "panel-cache schema")
        require(panel_cache["status"] == "EXPLORATORY_EXACT_IDENTITY_REUSE", "panel-cache status")
        require(panel_cache["artifact_bytes"] == pins.artifact_bytes and panel_cache["artifact_sha256"] == pins.artifact_sha256, "panel-cache artifact")
        require(panel_cache["extract_calls"] == 2 and panel_cache["delegate_extract_calls"] == 1 and panel_cache["same_panel_object_reused"] is True, "panel-cache counters")
        require(panel_cache["positive_claim_authority"] is False, "panel-cache claim authority")

        protected_inputs = {
            (runner.before[0], runner.before[1]): "runner",
            (strata.before[0], strata.before[1]): "strata_common",
            (frozen.before[0], frozen.before[1]): "frozen_auditor",
            (artifact.before[0], artifact.before[1]): "qwen_artifact",
        }
        for inode, label in closure["identities"].items():
            require(inode not in protected_inputs, f"authenticated input inode alias {label}/{protected_inputs.get(inode)}")
            protected_inputs[inode] = f"sealed_v8:{label}"
        require(not (set(protected_inputs) & set(publication["output_inodes"])), "input/output inode-domain alias")
        final_publication_rebind(publication, pins)
        runner.verify_final()
        strata.verify_final()
        frozen.verify_final()
        artifact.verify_final()

        limitations = [
            {
                "dependency": "PRODUCER_PARENT_COMMIT_MARKER",
                "status": "ABSENT_BY_EARLY_GATE_PUBLICATION_SCHEMA_EXTERNAL_COMPLETE_AND_RESULT_PINS_REQUIRED",
                "reason": "the direct completion-last publisher has no parent marker; this audit compensates with exact out-of-band file pins plus retained final-name rebinding",
            },
            {
                "dependency": "EXECUTED_EARLY_GATE_RUNNER_BYTE_PROVENANCE",
                "status": "IMPOSSIBLE_TO_PROVE_FROM_PUBLICATION_SELF_REPORTED_PATH_HASH_ONLY",
                "reason": "the exploratory runner is not launched from a retained immutable snapshot and hashes Path(__file__) late; the external runner pin authenticates the disclosed file bytes, not historical executed bytecode",
            },
            {
                "dependency": "IDENTITY_FRAMING_SEMANTIC_RECONSTRUCTION",
                "status": "IMPOSSIBLE_FROM_EMITTED_COUNTERFACTUAL_BY_SEALED_V8_ABI",
                "reason": "the sealed producer pairs original arithmetic payloads with the selected candidate model and explicitly treats the object as byte-cost framing only",
            },
            {
                "dependency": "GPU_PREFLIGHT_AND_TELEMETRY_REPLAY",
                "status": "EXTERNALLY_PINNED_BYTES_ONLY_NOT_REPLAYED_BY_CPU_RESULT_AUDIT",
                "reason": "SOURCE_PREFLIGHT is hash-authenticated; this result audit does not initialize CUDA",
            },
            {
                "dependency": "NESTED_HOLDOUT_GPU_REFIT",
                "status": "SPLITS_AND_ALL_EMITTED_ARITHMETIC_COUNTERS_VERIFIED_GPU_REFIT_NOT_REPLAYED",
                "reason": "the audit reconstructs the fold plan, partitions, votes, and gates but does not repeat the CuPy candidate fits",
            },
            {
                "dependency": "AUDITED_QWEN_MSE_SOURCE_TENSORS",
                "status": "EXTERNAL_BASELINE_SCORE_PIN_REQUIRED_AND_AUTHENTICATED",
                "reason": "the emitted artifact supports reconstruction replay but does not contain the original FP64 source tensors needed to recompute SSE/energy",
            },
        ]
        if source_status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED":
            limitations.append({
                "dependency": "MATCHED_CONTROLS",
                "status": "REQUIRED_BUT_NOT_PRESENT_BY_EARLY_GATE_CONTRACT",
                "reason": "a source survivor is not a positive compression result until separately authorized controls and another fresh independent audit complete",
            })
        return {
            "schema": "uwfa-sc-v8-qwen-early-gate-independent-result-audit-v0",
            "status": "PASS_FAIL_CLOSED_NONPROMOTING_RESULT_AUDIT",
            "positive_claim_authority": False,
            "controls_run_by_this_audit": False,
            "authorization": AUTHORIZATION,
            "output_parent": pins.output_parent,
            "final_name": pins.final_name,
            "completion_sha256": publication["complete"]["completion_sha256"],
            "complete_file_sha256": pins.complete_file_sha256,
            "result_sha256": pins.result_file_sha256,
            "runner_sha256": pins.runner_sha256,
            "sealed_v8_manifest_sha256": pins.v8_manifest_sha256,
            "sealed_v8_source_root_sha256": pins.v8_source_root_sha256,
            "artifact_sha256": pins.artifact_sha256,
            "artifact_bytes": pins.artifact_bytes,
            "independent_panel_reconstruction_f64_sha256": panel["reconstruction"]["full_reconstruction_f64_sha256"],
            "decision": source_audit["decision"],
            "literal_container_audit": source_audit["container"],
            "evidence_limitations": limitations,
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--output-parent", required=True)
    result.add_argument("--final-name", required=True)
    result.add_argument("--runner", required=True)
    result.add_argument("--expected-runner-sha256", required=True)
    result.add_argument("--v8-package", required=True)
    result.add_argument("--expected-v8-manifest-sha256", required=True)
    result.add_argument("--expected-v8-source-root-sha256", required=True)
    result.add_argument("--strata-common", required=True)
    result.add_argument("--expected-strata-common-sha256", required=True)
    result.add_argument("--frozen-auditor", required=True)
    result.add_argument("--expected-frozen-auditor-sha256", required=True)
    result.add_argument("--artifact", required=True)
    result.add_argument("--expected-artifact-sha256", required=True)
    result.add_argument("--expected-artifact-bytes", required=True, type=int)
    result.add_argument("--expected-complete-file-sha256", required=True)
    result.add_argument("--expected-result-file-sha256", required=True)
    result.add_argument("--expected-baseline-score-sha256", required=True)
    result.add_argument("--expected-source-preflight-sha256", required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    pins = Pins(
        output_parent=arguments.output_parent,
        final_name=arguments.final_name,
        runner_path=arguments.runner,
        runner_sha256=arguments.expected_runner_sha256,
        v8_package=arguments.v8_package,
        v8_manifest_sha256=arguments.expected_v8_manifest_sha256,
        v8_source_root_sha256=arguments.expected_v8_source_root_sha256,
        strata_common_path=arguments.strata_common,
        strata_common_sha256=arguments.expected_strata_common_sha256,
        frozen_auditor_path=arguments.frozen_auditor,
        frozen_auditor_sha256=arguments.expected_frozen_auditor_sha256,
        artifact_path=arguments.artifact,
        artifact_sha256=arguments.expected_artifact_sha256,
        artifact_bytes=arguments.expected_artifact_bytes,
        complete_file_sha256=arguments.expected_complete_file_sha256,
        result_file_sha256=arguments.expected_result_file_sha256,
        baseline_score_sha256=arguments.expected_baseline_score_sha256,
        source_preflight_sha256=arguments.expected_source_preflight_sha256,
    )
    receipt = verify(arguments.authorization, pins)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_QWEN_EARLY_GATE_RESULT_AUDIT: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
