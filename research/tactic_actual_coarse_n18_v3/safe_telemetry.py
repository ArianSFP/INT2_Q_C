#!/usr/bin/env python3
"""Strict semantic validator for a future instrumented CuPy/NVML receipt."""

from __future__ import annotations

import math
from typing import Any

from v3_common import (
    PCI_PATTERN,
    TELEMETRY_SCHEMA,
    UUID_PATTERN,
    VERSION_PATTERN,
    exact_keys,
    require,
    valid_sha256,
)


FIELDS = {
    "schema",
    "authenticated_source_root",
    "runtime_lock_sha256",
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
    "transfers",
    "kernel_launches",
    "cuda_events_synchronized",
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
    "sampling_limit",
}


def _positive_number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (float(value) >= 0.0 if allow_zero else float(value) > 0.0),
        label,
    )
    return float(value)


def _version(value: Any, label: str) -> str:
    require(isinstance(value, str) and VERSION_PATTERN.fullmatch(value) is not None, label)
    return value


def validate_telemetry(value: Any) -> dict[str, Any]:
    value = exact_keys(value, FIELDS, "telemetry receipt")
    require(value["schema"] == TELEMETRY_SCHEMA, "telemetry schema")
    require(valid_sha256(value["authenticated_source_root"], nonzero=True), "telemetry source root")
    require(valid_sha256(value["runtime_lock_sha256"], nonzero=True), "telemetry runtime lock")
    require(value["cuda_visible_devices"] == "0" and value["cuda_logical_index"] == 0, "CUDA logical mapping")
    require(type(value["nvml_physical_index"]) is int and value["nvml_physical_index"] >= 0, "NVML physical index")
    require(
        isinstance(value["cuda_uuid"], str)
        and UUID_PATTERN.fullmatch(value["cuda_uuid"])
        and value["cuda_uuid"] == value["nvml_uuid"],
        "nonempty equal CUDA/NVML UUID",
    )
    require(
        isinstance(value["cuda_pci_bus_id"], str)
        and PCI_PATTERN.fullmatch(value["cuda_pci_bus_id"])
        and value["cuda_pci_bus_id"].lower() == str(value["nvml_pci_bus_id"]).lower(),
        "nonempty equal CUDA/NVML PCI",
    )
    require(value["device_name"] == "NVIDIA GeForce RTX 5090", "mandatory future device")
    _version(value["compute_capability"], "compute capability")
    for field in (
        "driver_version",
        "cuda_runtime_version",
        "cupy_version",
        "numpy_version",
        "scipy_version",
        "pynvml_version",
    ):
        _version(value[field], field)
    for field in ("logical_h2d_bytes", "logical_d2h_bytes"):
        require(type(value[field]) is int and value[field] > 0, f"positive {field}")
    require(
        type(value["model_h2d_bytes"]) is int
        and 0 <= value["model_h2d_bytes"] <= value["logical_h2d_bytes"],
        "model H2D subset",
    )
    transfers = value["transfers"]
    require(isinstance(transfers, list) and 1 <= len(transfers) <= 4096, "bounded transfer provenance")
    h2d = d2h = model_h2d = 0
    labels: set[str] = set()
    for row in transfers:
        row = exact_keys(row, {"ordinal", "direction", "label", "bytes", "model", "buffer_sha256"}, "transfer row")
        require(type(row["ordinal"]) is int and row["ordinal"] == len(labels), "transfer ordinal")
        require(row["direction"] in ("h2d", "d2h"), "transfer direction")
        require(isinstance(row["label"], str) and 0 < len(row["label"]) <= 128 and row["label"] not in labels, "unique transfer label")
        labels.add(row["label"])
        require(type(row["bytes"]) is int and row["bytes"] > 0, "positive transfer bytes")
        require(type(row["model"]) is bool and (not row["model"] or row["direction"] == "h2d"), "model transfer semantics")
        require(valid_sha256(row["buffer_sha256"], nonzero=True), "transfer buffer digest")
        if row["direction"] == "h2d":
            h2d += row["bytes"]
            if row["model"]:
                model_h2d += row["bytes"]
        else:
            d2h += row["bytes"]
    require((h2d, d2h, model_h2d) == (value["logical_h2d_bytes"], value["logical_d2h_bytes"], value["model_h2d_bytes"]), "transfer totals/provenance")
    require(type(value["kernel_launches"]) is int and value["kernel_launches"] > 0, "positive kernel launches")
    require(value["cuda_events_synchronized"] is True, "synchronized CUDA events")
    phase_ms = sum(
        _positive_number(value[field], field)
        for field in ("cuda_event_h2d_ms", "cuda_event_kernel_ms", "cuda_event_d2h_ms")
    )
    wall = _positive_number(value["wall_seconds"], "positive wall seconds")
    require(wall * 1000.0 + 1e-9 >= phase_ms, "wall time contains CUDA phases")
    _positive_number(value["telemetry_sampling_interval_ms"], "positive telemetry sampling interval")
    for prefix in ("host_rss", "nvml_process", "nvml_device", "cupy_pool"):
        baseline = value[f"{prefix}_baseline_bytes"]
        peak = value[f"{prefix}_peak_bytes"]
        delta = value[f"{prefix}_delta_bytes"]
        require(
            type(baseline) is int
            and type(peak) is int
            and type(delta) is int
            and 0 <= baseline <= peak
            and delta == peak - baseline,
            f"{prefix} baseline/peak/delta",
        )
    require(
        value["transfer_definition"]
        == "exact logical nbytes of every enumerated host/device buffer; not claimed physical PCIe traffic",
        "transfer definition",
    )
    require(
        value["sampling_limit"]
        == "RSS/NVML peaks are sampled and may miss sub-interval transients; logical transfer totals are exact",
        "sampling limitation disclosure",
    )
    return value
