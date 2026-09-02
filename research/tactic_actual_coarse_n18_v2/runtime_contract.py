#!/usr/bin/env python3
"""Authenticated dependency/import graph and CuPy telemetry contract.

This module deliberately imports only the standard library.  Numeric/CUDA
imports are forbidden until a future environment lock and independent review
receipt both pass.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from n18_common import (
    ENVIRONMENT_SCHEMA,
    REVIEW_SCHEMA,
    canonical_json,
    is_sha256,
    require,
    strict_json_loads,
)
from secure_io import HeldFileSet, HeldRegularFile


DEPENDENCY_GRAPH_SCHEMA = "tactic_actual_coarse_n18_dependency_graph_v2"


def _seal_valid(value: dict[str, Any], field: str) -> bool:
    declared = value.get(field)
    if not is_sha256(declared):
        return False
    clone = dict(value)
    clone.pop(field, None)
    return hashlib.sha256(canonical_json(clone)).hexdigest() == declared


def validate_review_receipt(raw: bytes, manifest_sha256: str, action: str) -> dict[str, Any]:
    value = strict_json_loads(raw)
    require(isinstance(value, dict), "review receipt object")
    require(
        set(value)
        == {
            "schema",
            "status",
            "source_manifest_sha256",
            "allowed_actions",
            "findings_sha256",
            "receipt_sha256",
        },
        "review receipt exact keys",
    )
    require(value["schema"] == REVIEW_SCHEMA, "review receipt schema")
    require(value["status"] == "PASS_INDEPENDENT_SOURCE_REVIEW", "independent review status")
    require(value["source_manifest_sha256"] == manifest_sha256, "review/source manifest binding")
    require(is_sha256(value["findings_sha256"]), "review findings digest")
    actions = value["allowed_actions"]
    require(
        isinstance(actions, list)
        and actions == sorted(set(actions))
        and set(actions) <= {"synthetic", "pilot", "full"},
        "review action set",
    )
    require(action in actions, "review does not authorize requested action")
    require(_seal_valid(value, "receipt_sha256"), "review receipt internal seal")
    return value


def validate_environment_lock(raw: bytes) -> dict[str, Any]:
    value = strict_json_loads(raw)
    require(isinstance(value, dict), "environment lock object")
    require(value.get("schema") == ENVIRONMENT_SCHEMA, "environment schema")
    require(
        value.get("status") == "FROZEN_AUTHENTICATED_RUNTIME_READY",
        "runtime environment intentionally unfrozen; CUDA/payload remains blocked",
    )
    require(_seal_valid(value, "lock_sha256"), "environment internal seal")
    interpreter = value.get("interpreter")
    require(isinstance(interpreter, dict), "interpreter lock")
    require(
        set(interpreter) == {"absolute_path", "bytes", "sha256", "python_version"},
        "interpreter exact lock",
    )
    require(interpreter["absolute_path"] == sys.executable, "interpreter absolute path")
    require(is_sha256(interpreter["sha256"]), "interpreter digest")
    distributions = value.get("distributions")
    require(isinstance(distributions, list) and distributions, "runtime distributions")
    required = {"numpy", "cupy-cuda12x", "scipy", "nvidia-ml-py"}
    observed = {row.get("name") for row in distributions if isinstance(row, dict)}
    require(observed == required, "exact required runtime distributions")
    for row in distributions:
        require(
            set(row) == {"name", "version", "record_sha256", "files_root_sha256"}
            and isinstance(row["version"], str)
            and is_sha256(row["record_sha256"])
            and is_sha256(row["files_root_sha256"]),
            "runtime distribution lock row",
        )
    return value


def _imports(source: bytes) -> set[str]:
    tree = ast.parse(source.decode("utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add((node.module or "").split(".")[0])
    result.discard("")
    result.discard("__future__")
    return result


@dataclass
class AuthenticatedDependencies:
    """Held external source descriptors and private immutable import copies."""

    held: HeldFileSet
    snapshot: tempfile.TemporaryDirectory[str]
    rows: tuple[dict[str, Any], ...]

    def close(self) -> None:
        self.held.close()
        self.snapshot.cleanup()

    def __enter__(self) -> "AuthenticatedDependencies":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def authenticate_dependencies(raw_lock: bytes, absolute_repo_root: str) -> AuthenticatedDependencies:
    """Hash sources by held FD, verify imports, then copy to a private snapshot."""
    require(os.name == "posix" and os.path.isabs(absolute_repo_root), "absolute POSIX repository root")
    value = strict_json_loads(raw_lock)
    require(isinstance(value, dict), "dependency graph object")
    require(value.get("schema") == DEPENDENCY_GRAPH_SCHEMA, "dependency graph schema")
    require(value.get("status") == "SOURCE_PINNED_RUNTIME_ENVIRONMENT_SEPARATELY_LOCKED", "dependency graph status")
    rows = value.get("external_python_sources")
    require(isinstance(rows, list) and len(rows) == 2, "two external source dependencies")
    held = HeldFileSet()
    snapshot = tempfile.TemporaryDirectory(prefix="tactic-n18-v2-dependencies-")
    try:
        observed_rows: list[dict[str, Any]] = []
        for row in rows:
            require(
                isinstance(row, dict)
                and set(row) == {"id", "relative_path", "bytes", "sha256", "allowed_import_roots"},
                "dependency row schema",
            )
            relative = row["relative_path"]
            require(
                isinstance(relative, str)
                and not relative.startswith("/")
                and ".." not in relative.split("/"),
                "dependency relative path",
            )
            absolute = os.path.normpath(os.path.join(absolute_repo_root, *relative.split("/")))
            require(absolute.startswith(os.path.normpath(absolute_repo_root) + os.sep), "dependency root containment")
            file = held.add(
                HeldRegularFile(
                    absolute,
                    maximum_bytes=1 << 20,
                    expected_bytes=row["bytes"],
                    expected_sha256=row["sha256"],
                )
            )
            packet = file.read()
            imports = _imports(packet)
            declared = row["allowed_import_roots"]
            require(
                isinstance(declared, list)
                and declared == sorted(set(declared))
                and imports == set(declared),
                f"dependency import graph drift: {row['id']}",
            )
            destination = os.path.join(snapshot.name, f"{row['id']}.py")
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            try:
                offset = 0
                while offset < len(packet):
                    written = os.write(descriptor, packet[offset:])
                    require(written > 0, "dependency snapshot short write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            observed_rows.append(
                {
                    "id": row["id"],
                    "bytes": len(packet),
                    "sha256": hashlib.sha256(packet).hexdigest(),
                    "imports": sorted(imports),
                }
            )
        held.verify_stable()
        return AuthenticatedDependencies(held, snapshot, tuple(observed_rows))
    except Exception:
        held.close()
        snapshot.cleanup()
        raise


def validate_telemetry_receipt(value: dict[str, Any]) -> None:
    """Validate the exact mandatory receipt schema for a future CuPy run."""
    require(
        set(value)
        == {
            "cuda_visible_devices",
            "cuda_logical_index",
            "cuda_uuid",
            "cuda_pci_bus_id",
            "nvml_physical_index",
            "nvml_uuid",
            "nvml_pci_bus_id",
            "device_name",
            "compute_capability",
            "driver_version",
            "cuda_runtime_version",
            "cupy_version",
            "numpy_version",
            "scipy_version",
            "pynvml_version",
            "logical_h2d_bytes",
            "logical_d2h_bytes",
            "model_h2d_bytes",
            "cuda_event_h2d_ms",
            "cuda_event_kernel_ms",
            "cuda_event_d2h_ms",
            "wall_seconds",
            "host_rss_baseline_bytes",
            "host_rss_peak_bytes",
            "host_rss_delta_bytes",
            "nvml_process_baseline_bytes",
            "nvml_process_peak_bytes",
            "nvml_process_delta_bytes",
            "nvml_device_baseline_bytes",
            "nvml_device_peak_bytes",
            "nvml_device_delta_bytes",
            "cupy_pool_baseline_bytes",
            "cupy_pool_peak_bytes",
            "cupy_pool_delta_bytes",
            "telemetry_sampling_interval_ms",
            "transfer_definition",
        },
        "telemetry exact fields",
    )
    require(value["cuda_logical_index"] == 0, "CUDA logical device zero")
    require(value["cuda_visible_devices"] == "0", "CUDA_VISIBLE_DEVICES exact zero")
    require(value["device_name"] == "NVIDIA GeForce RTX 5090", "mandatory RTX 5090 device")
    require(value["cuda_uuid"] == value["nvml_uuid"], "CUDA/NVML UUID mapping")
    require(value["cuda_pci_bus_id"] == value["nvml_pci_bus_id"], "CUDA/NVML PCI mapping")
    for prefix in ("host_rss", "nvml_process", "nvml_device", "cupy_pool"):
        baseline = value[f"{prefix}_baseline_bytes"]
        peak = value[f"{prefix}_peak_bytes"]
        delta = value[f"{prefix}_delta_bytes"]
        require(type(baseline) is int and type(peak) is int and type(delta) is int, f"{prefix} integer bytes")
        require(0 <= baseline <= peak and delta == peak - baseline, f"{prefix} baseline/peak/delta")
    for field in ("logical_h2d_bytes", "logical_d2h_bytes", "model_h2d_bytes"):
        require(type(value[field]) is int and value[field] >= 0, f"{field} nonnegative")
    for field in (
        "cuda_event_h2d_ms",
        "cuda_event_kernel_ms",
        "cuda_event_d2h_ms",
        "wall_seconds",
        "telemetry_sampling_interval_ms",
    ):
        require(
            isinstance(value[field], (int, float))
            and not isinstance(value[field], bool)
            and math.isfinite(float(value[field]))
            and float(value[field]) >= 0.0,
            f"{field} finite nonnegative",
        )
    require(
        value["transfer_definition"]
        == "exact logical nbytes of explicitly enumerated host/device arrays; not claimed physical PCIe traffic",
        "telemetry transfer definition",
    )


class LogicalTransferLedger:
    """Explicit transfer-byte ledger used by an instrumented future core."""

    def __init__(self) -> None:
        self.h2d = 0
        self.d2h = 0
        self.model_h2d = 0
        self.rows: list[dict[str, Any]] = []

    def record(self, direction: str, nbytes: int, label: str, *, model: bool = False) -> None:
        require(direction in ("h2d", "d2h"), "transfer direction")
        require(type(nbytes) is int and nbytes >= 0, "transfer nbytes")
        require(isinstance(label, str) and label, "transfer label")
        require(not model or direction == "h2d", "model transfer direction")
        self.rows.append({"direction": direction, "bytes": nbytes, "label": label, "model": model})
        if direction == "h2d":
            self.h2d += nbytes
            if model:
                self.model_h2d += nbytes
        else:
            self.d2h += nbytes
