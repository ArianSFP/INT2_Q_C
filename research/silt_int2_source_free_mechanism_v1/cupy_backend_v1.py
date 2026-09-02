#!/usr/bin/env python3
"""Mandatory CuPy metadata search with complete measured v1 telemetry."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Sequence

import numpy as np

from silt_v1 import (
    ExpertInput,
    FormatError,
    LiftedTensor,
    deterministic_permutation,
    deterministic_selectors,
    encode_coefficients,
    fit_model,
    flatten_details,
    lift_forward,
    level_pair_counts,
    level_sizes,
    pack_selectors,
    permutation_byte_count,
    require,
    serialize_permutation,
    validate_alphabet,
    validate_geometry,
    validate_permutation,
    validate_selector,
)


class GpuEnvironmentError(FormatError):
    pass


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _normalize_bus_id(value: str) -> str:
    value = value.strip().lower()
    # CUDA may use a four-digit domain and NVML an eight-digit domain.
    pieces = value.split(":")
    require(len(pieces) >= 3, "PCI bus ID syntax")
    return f"{int(pieces[-3], 16):04x}:{int(pieces[-2], 16):02x}:{pieces[-1]}"


def _process_vram(pynvml: Any, handle: Any, pid: int) -> int:
    getter = getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None)
    if getter is None:
        getter = pynvml.nvmlDeviceGetComputeRunningProcesses
    total = 0
    try:
        for process in getter(handle):
            if int(process.pid) == pid:
                used = int(process.usedGpuMemory)
                if used >= 0 and used < (1 << 63):
                    total += used
    except pynvml.NVMLError:
        return 0
    return total


def require_gpu_environment(require_rtx_5090: bool = True) -> tuple[Any, Any, Any, dict[str, object]]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as exc:
        raise GpuEnvironmentError("pynvml/NVML is mandatory for v1 telemetry") from exc
    pid = os.getpid()
    pre_by_uuid: dict[str, int] = {}
    physical_count = int(pynvml.nvmlDeviceGetCount())
    for physical in range(physical_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
        uuid = _text(pynvml.nvmlDeviceGetUUID(handle))
        pre_by_uuid[uuid] = _process_vram(pynvml, handle, pid)
    try:
        import cupy as cp
    except Exception as exc:
        pynvml.nvmlShutdown()
        raise GpuEnvironmentError("CuPy is mandatory for v1 search") from exc
    try:
        logical = int(cp.cuda.Device().id)
        raw_bus = _text(cp.cuda.runtime.deviceGetPCIBusId(logical))
        normalized_cuda_bus = _normalize_bus_id(raw_bus)
        matched: tuple[int, Any, str, str] | None = None
        for physical in range(physical_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            nvml_bus = _text(pci.busId)
            if _normalize_bus_id(nvml_bus) == normalized_cuda_bus:
                matched = (physical, handle, _text(pynvml.nvmlDeviceGetUUID(handle)), nvml_bus)
                break
        require(matched is not None, "CUDA logical device must map to one NVML physical device")
        physical, handle, uuid, nvml_bus = matched
        properties = cp.cuda.runtime.getDeviceProperties(logical)
        name = _text(properties["name"])
        if require_rtx_5090:
            require("5090" in name, f"canonical device must be RTX 5090, found {name!r}")
        pre_context_process = int(pre_by_uuid.get(uuid, 0))
        cp.cuda.Device(logical).use()
        cp.cuda.runtime.deviceSynchronize()
        post_context_process = _process_vram(pynvml, handle, pid)
        pool = cp.get_default_memory_pool()
        pre_search_pool = int(pool.used_bytes())
        pre_search_process = _process_vram(pynvml, handle, pid)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mapping = {
            "cuda_logical_index": logical,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_pci_bus_id": raw_bus,
            "cuda_pci_bus_id_normalized": normalized_cuda_bus,
            "nvml_physical_index": int(physical),
            "nvml_pci_bus_id": nvml_bus,
            "gpu_uuid": uuid,
            "gpu_mapping_asserted_by_pci_bus_id": True,
            "device_name": name,
            "compute_capability": str(cp.cuda.Device(logical).compute_capability),
            "device_total_bytes": int(memory.total),
            "driver_version": _text(pynvml.nvmlSystemGetDriverVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cupy_version": str(cp.__version__),
            "pynvml_package_version": importlib_metadata.version("nvidia-ml-py"),
            "nvml_library_version": _text(pynvml.nvmlSystemGetNVMLVersion()),
            "pre_context_process_vram_bytes": pre_context_process,
            "post_context_process_vram_bytes": int(post_context_process),
            "pre_search_process_vram_bytes": int(pre_search_process),
            "pre_search_cupy_pool_used_bytes": pre_search_pool,
        }
        return cp, pynvml, handle, mapping
    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        raise


def _host_rss_bytes() -> int:
    with open("/proc/self/statm", "r", encoding="ascii") as handle:
        fields = handle.read().split()
    require(len(fields) >= 2, "procfs statm")
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


class ResourceSampler:
    def __init__(self, cp: Any, pynvml: Any, nvml_handle: Any, interval_seconds: float = 0.002) -> None:
        self.cp = cp
        self.pynvml = pynvml
        self.handle = nvml_handle
        self.interval = interval_seconds
        self.pid = os.getpid()
        self.host_baseline = _host_rss_bytes()
        info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        self.device_baseline = int(info.used)
        self.process_baseline = _process_vram(pynvml, nvml_handle, self.pid)
        self.pool_baseline = int(cp.get_default_memory_pool().used_bytes())
        self.host_peak = self.host_baseline
        self.device_peak = self.device_baseline
        self.process_peak = self.process_baseline
        self.pool_peak = self.pool_baseline
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> None:
        self.host_peak = max(self.host_peak, _host_rss_bytes())
        info = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.device_peak = max(self.device_peak, int(info.used))
        self.process_peak = max(self.process_peak, _process_vram(self.pynvml, self.handle, self.pid))
        self.pool_peak = max(self.pool_peak, int(self.cp.get_default_memory_pool().used_bytes()))
        self.samples += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.sample()
        self.thread.start()

    def finish(self) -> dict[str, object]:
        self.sample()
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        return {
            "host_rss_baseline_bytes": self.host_baseline,
            "host_rss_peak_bytes": self.host_peak,
            "host_rss_delta_bytes": max(0, self.host_peak - self.host_baseline),
            "vram_device_baseline_used_bytes": self.device_baseline,
            "vram_device_peak_used_bytes": self.device_peak,
            "vram_device_delta_bytes": max(0, self.device_peak - self.device_baseline),
            "vram_process_baseline_used_bytes": self.process_baseline,
            "vram_process_peak_used_bytes": self.process_peak,
            "vram_process_delta_bytes": max(0, self.process_peak - self.process_baseline),
            "cupy_pool_baseline_used_bytes": self.pool_baseline,
            "cupy_pool_peak_used_bytes": self.pool_peak,
            "cupy_pool_delta_bytes": max(0, self.pool_peak - self.pool_baseline),
            "resource_sample_count": self.samples,
            "resource_poll_interval_ms": self.interval * 1000.0,
            "vram_measurement": "NVML total-device and current-process samples; deltas baseline-subtracted",
            "host_rss_measurement": "/proc/self/statm sampled RSS",
        }


def _cuda_timed(cp: Any, operation: Any) -> tuple[Any, float]:
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    output = operation()
    stop.record()
    stop.synchronize()
    return output, float(cp.cuda.get_elapsed_time(start, stop))


class TransferLedger:
    def __init__(self, cp: Any) -> None:
        self.cp = cp
        self.rows: list[dict[str, object]] = []
        self.h2d_ms = 0.0
        self.d2h_ms = 0.0

    def h2d(self, name: str, array: np.ndarray) -> Any:
        require(isinstance(array, np.ndarray) and array.flags.c_contiguous, "H2D contiguous numpy array")
        output, elapsed = _cuda_timed(self.cp, lambda: self.cp.asarray(array))
        self.h2d_ms += elapsed
        self.rows.append(
            {
                "phase": name,
                "direction": "H2D",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "logical_array_bytes": int(array.nbytes),
                "cuda_event_ms": elapsed,
            }
        )
        return output

    def d2h(self, name: str, array: Any) -> np.ndarray:
        output, elapsed = _cuda_timed(self.cp, lambda: self.cp.asnumpy(array))
        self.d2h_ms += elapsed
        self.rows.append(
            {
                "phase": name,
                "direction": "D2H",
                "dtype": str(array.dtype),
                "shape": [int(value) for value in array.shape],
                "logical_array_bytes": int(array.nbytes),
                "cuda_event_ms": elapsed,
            }
        )
        return output

    def summary(self) -> dict[str, object]:
        h2d = sum(int(row["logical_array_bytes"]) for row in self.rows if row["direction"] == "H2D")
        d2h = sum(int(row["logical_array_bytes"]) for row in self.rows if row["direction"] == "D2H")
        return {
            "h2d_bytes": h2d,
            "d2h_bytes": d2h,
            "h2d_ms": self.h2d_ms,
            "d2h_ms": self.d2h_ms,
            "model_h2d_bytes": 0,
            "transfer_byte_semantics": "exact logical array bytes, not inferred physical PCIe transactions",
            "array_transfers": self.rows,
        }


@dataclass(frozen=True)
class GpuLifted:
    roots: Any
    detail_levels: tuple[Any, ...]


def lift_forward_device(cp: Any, leaves: Any, alphabet: int, permutation: Any, selectors: Any) -> GpuLifted:
    alphabet = validate_alphabet(alphabet)
    vectors, lanes = (int(value) for value in leaves.shape)
    validate_geometry(lanes, vectors)
    current = cp.ascontiguousarray(leaves[:, permutation.astype(cp.int64, copy=False)])
    levels: list[Any] = []
    offset = 0
    for pair_count in level_pair_counts(lanes):
        codes = selectors[offset : offset + pair_count]
        offset += pair_count
        left = current[:, 0 : 2 * pair_count : 2].astype(cp.int16)
        right = current[:, 1 : 2 * pair_count : 2].astype(cp.int16)
        swap = (((codes >> 2) & 1).astype(cp.bool_))[None, :]
        p = (((codes >> 1) & 1).astype(cp.int16))[None, :]
        u = ((codes & 1).astype(cp.int16))[None, :]
        x = cp.where(swap, right, left)
        y = cp.where(swap, left, right)
        detail = cp.mod(y - p * x, alphabet).astype(cp.uint8)
        coarse = cp.mod(x + u * detail.astype(cp.int16), alphabet).astype(cp.uint8)
        current = cp.concatenate((coarse, current[:, -1:]), axis=1) if int(current.shape[1]) & 1 else coarse
        levels.append(cp.ascontiguousarray(detail))
    require(offset == lanes - 1 and current.shape == (vectors, 1), "GPU forward coverage")
    return GpuLifted(cp.ascontiguousarray(current[:, 0]), tuple(levels))


def flatten_details_device(cp: Any, lifted: GpuLifted) -> Any:
    if not lifted.detail_levels:
        return cp.empty(0, dtype=cp.uint8)
    return cp.ascontiguousarray(cp.concatenate([level.reshape(-1) for level in reversed(lifted.detail_levels)])).astype(cp.uint8, copy=False)


def lift_inverse_device(cp: Any, lifted: GpuLifted, lanes: int, alphabet: int, permutation: Any, selectors: Any) -> Any:
    count_rows = level_pair_counts(lanes)
    offsets = np.cumsum([0] + count_rows).tolist()
    width_rows = level_sizes(lanes)
    current = cp.ascontiguousarray(lifted.roots[:, None])
    vectors = int(lifted.roots.shape[0])
    for depth in range(len(count_rows) - 1, -1, -1):
        pair_count = count_rows[depth]
        codes = selectors[offsets[depth] : offsets[depth + 1]]
        detail = lifted.detail_levels[depth].astype(cp.int16)
        coarse = current[:, :pair_count].astype(cp.int16)
        swap = (((codes >> 2) & 1).astype(cp.bool_))[None, :]
        p = (((codes >> 1) & 1).astype(cp.int16))[None, :]
        u = ((codes & 1).astype(cp.int16))[None, :]
        x = cp.mod(coarse - u * detail, alphabet)
        y = cp.mod(detail + p * x, alphabet)
        left = cp.where(swap, y, x).astype(cp.uint8)
        right = cp.where(swap, x, y).astype(cp.uint8)
        previous = cp.empty((vectors, width_rows[depth]), dtype=cp.uint8)
        previous[:, 0 : 2 * pair_count : 2] = left
        previous[:, 1 : 2 * pair_count : 2] = right
        if width_rows[depth] & 1:
            previous[:, -1] = current[:, -1]
        current = previous
    output = cp.empty_like(current)
    output[:, permutation.astype(cp.int64, copy=False)] = current
    return output


@dataclass(frozen=True)
class SearchResult:
    selected_seed: int
    selected_permutation: tuple[int, ...]
    selected_selectors: tuple[int, ...]
    candidate_rows: tuple[dict[str, object], ...]
    telemetry: dict[str, object]


def search_metadata_cupy(
    train_leaves: np.ndarray,
    validation_leaves: np.ndarray,
    alphabet: int,
    candidate_seeds: Sequence[int],
    *,
    require_rtx_5090: bool = True,
) -> SearchResult:
    alphabet = validate_alphabet(alphabet)
    require(train_leaves.ndim == validation_leaves.ndim == 2, "search matrices")
    require(train_leaves.dtype == validation_leaves.dtype == np.uint8, "search dtypes")
    require(train_leaves.shape[1] == validation_leaves.shape[1], "search lane agreement")
    validate_geometry(int(train_leaves.shape[1]), int(train_leaves.shape[0]))
    validate_geometry(int(validation_leaves.shape[1]), int(validation_leaves.shape[0]))
    seeds = [int(seed) for seed in candidate_seeds]
    require(seeds and len(seeds) == len(set(seeds)), "unique candidate seeds")
    host_wall_start = time.perf_counter()
    cp, pynvml, nvml_handle, mapping = require_gpu_environment(require_rtx_5090=require_rtx_5090)
    sampler = ResourceSampler(cp, pynvml, nvml_handle)
    sampler.start()
    ledger = TransferLedger(cp)
    kernel_ms = 0.0
    cpu_score_ms = 0.0
    lanes = int(train_leaves.shape[1])
    candidate_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    try:
        gpu_train = ledger.h2d("source/train", np.ascontiguousarray(train_leaves))
        gpu_validation = ledger.h2d("source/validation", np.ascontiguousarray(validation_leaves))
        for seed in seeds:
            permutation = np.ascontiguousarray(np.asarray(deterministic_permutation(lanes, seed), dtype=np.int64))
            selectors = np.ascontiguousarray(np.asarray(deterministic_selectors(lanes, alphabet, seed ^ 0x5A17), dtype=np.uint8))
            gpu_permutation = ledger.h2d(f"candidate/{seed}/permutation", permutation)
            gpu_selectors = ledger.h2d(f"candidate/{seed}/selectors", selectors)
            train_lifted, elapsed = _cuda_timed(cp, lambda: lift_forward_device(cp, gpu_train, alphabet, gpu_permutation, gpu_selectors))
            kernel_ms += elapsed
            validation_lifted, elapsed = _cuda_timed(cp, lambda: lift_forward_device(cp, gpu_validation, alphabet, gpu_permutation, gpu_selectors))
            kernel_ms += elapsed
            train_roots = ledger.d2h(f"candidate/{seed}/train_roots", train_lifted.roots)
            train_details = ledger.d2h(f"candidate/{seed}/train_details", flatten_details_device(cp, train_lifted))
            validation_roots = ledger.d2h(f"candidate/{seed}/validation_roots", validation_lifted.roots)
            validation_details = ledger.d2h(f"candidate/{seed}/validation_details", flatten_details_device(cp, validation_lifted))
            scoring_start = time.perf_counter()
            model = fit_model(alphabet, train_roots, train_details)
            finite_packet, meaningful = encode_coefficients(model, validation_roots, validation_details)
            cpu_score_ms += 1000.0 * (time.perf_counter() - scoring_start)
            metadata_bytes = len(serialize_permutation(permutation.tolist())) + len(pack_selectors(selectors.tolist(), alphabet))
            rows.append(
                {
                    "seed": seed,
                    "finite_meaningful_bits": meaningful,
                    "finite_payload_bytes": len(finite_packet),
                    "charged_metadata_bytes": metadata_bytes,
                    "charged_objective_bits": meaningful + 8 * metadata_bytes,
                    "validation_symbols": int(validation_leaves.size),
                }
            )
            candidate_cache[seed] = (train_roots, train_details, validation_roots, validation_details)
        rows.sort(key=lambda row: (int(row["charged_objective_bits"]), int(row["seed"])))
        selected = int(rows[0]["seed"])
        selected_permutation = deterministic_permutation(lanes, selected)
        selected_selectors = deterministic_selectors(lanes, alphabet, selected ^ 0x5A17)
        cpu_train = lift_forward(train_leaves, alphabet, selected_permutation, selected_selectors)
        cpu_validation = lift_forward(validation_leaves, alphabet, selected_permutation, selected_selectors)
        cached = candidate_cache[selected]
        require(np.array_equal(cpu_train.roots, cached[0]), "selected train roots CPU/CuPy")
        require(np.array_equal(flatten_details(cpu_train), cached[1]), "selected train details CPU/CuPy")
        require(np.array_equal(cpu_validation.roots, cached[2]), "selected validation roots CPU/CuPy")
        require(np.array_equal(flatten_details(cpu_validation), cached[3]), "selected validation details CPU/CuPy")
        selected_perm_array = np.ascontiguousarray(np.asarray(selected_permutation, dtype=np.int64))
        selected_selector_array = np.ascontiguousarray(np.asarray(selected_selectors, dtype=np.uint8))
        gpu_permutation = ledger.h2d("selected/permutation", selected_perm_array)
        gpu_selectors = ledger.h2d("selected/selectors", selected_selector_array)
        selected_lifted, elapsed = _cuda_timed(cp, lambda: lift_forward_device(cp, gpu_validation, alphabet, gpu_permutation, gpu_selectors))
        kernel_ms += elapsed
        inverse, elapsed = _cuda_timed(cp, lambda: lift_inverse_device(cp, selected_lifted, lanes, alphabet, gpu_permutation, gpu_selectors))
        kernel_ms += elapsed
        inverse_host = ledger.d2h("selected/inverse_validation", inverse)
        require(np.array_equal(inverse_host, validation_leaves), "selected GPU inverse")
        cp.cuda.runtime.deviceSynchronize()
        resources = sampler.finish()
        transfers = ledger.summary()
        wall_ms = 1000.0 * (time.perf_counter() - host_wall_start)
        telemetry = {
            **mapping,
            **resources,
            **transfers,
            "kernel_ms": kernel_ms,
            "cpu_reference_scoring_ms": cpu_score_ms,
            "wall_ms": wall_ms,
            "candidate_count": len(seeds),
            "train_shape": list(train_leaves.shape),
            "validation_shape": list(validation_leaves.shape),
            "cpu_cupy_selected_coefficients_equal": True,
            "cupy_inverse_roundtrip_equal": True,
            "timings_measured_not_inferred": True,
        }
        return SearchResult(selected, tuple(selected_permutation), tuple(selected_selectors), tuple(rows), telemetry)
    finally:
        if sampler.thread.is_alive():
            sampler.finish()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
