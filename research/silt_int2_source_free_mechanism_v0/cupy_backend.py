#!/usr/bin/env python3
"""Mandatory CuPy optimization/search backend for the source-free SILT prototype.

The CPU implementation in :mod:`silt_mechanism` is the bit-exact reference.
This module is deliberately fail-closed: metadata optimization is not allowed
to silently fall back to the CPU.  The canonical replay additionally requires
the provided RTX 5090 and records measured CUDA-event transfer/kernel timings,
wall time, and sampled device-memory use.

No function accepts a file path or external payload.  Arrays must be supplied
by the synthetic-only caller.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from silt_mechanism import (
    ContractError,
    LiftedTensor,
    deterministic_permutation,
    deterministic_selectors,
    encode_coefficients,
    fit_model,
    flatten_details,
    lift_forward,
    level_pair_counts,
    level_sizes,
    permutation_byte_count,
    require,
)


class CuPyRequiredError(ContractError):
    """The mandatory GPU search contract is unavailable."""


def require_cupy(require_rtx_5090: bool = False) -> tuple[Any, dict[str, object]]:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - exercised on CPU-only hosts
        raise CuPyRequiredError("CuPy/CUDA is mandatory for SILT metadata search") from exc

    try:
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        raw_name = properties["name"]
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception as exc:
        raise CuPyRequiredError("a working CUDA device is mandatory for SILT metadata search") from exc
    if require_rtx_5090 and "5090" not in name:
        raise CuPyRequiredError(f"canonical replay requires RTX 5090, found {name!r}")
    return cp, {
        "device_index": int(device.id),
        "device_name": name,
        "compute_capability": str(device.compute_capability),
        "device_total_bytes": int(total_bytes),
        "device_free_bytes_at_start": int(free_bytes),
        "cupy_version": str(cp.__version__),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "canonical_rtx_5090_required": bool(require_rtx_5090),
    }


class _PeakVramSampler:
    """Poll NVML when present; otherwise record synchronized CUDA samples.

    Both paths are measurements.  The fallback is explicitly named a
    phase-boundary sample and is not represented as a continuous high-water
    mark.
    """

    def __init__(self, cp: Any, device_index: int, interval_seconds: float = 0.002) -> None:
        self.cp = cp
        self.device_index = int(device_index)
        self.interval_seconds = float(interval_seconds)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak = 0
        self.samples = 0
        self.method = "cuda_memGetInfo_phase_boundary"
        self._nvml: Any | None = None
        self._handle: Any | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.method = "NVML_poll"
        except Exception:
            self._nvml = None
            self._handle = None

    def _sample_nvml(self) -> None:
        assert self._nvml is not None and self._handle is not None
        while not self.stop_event.is_set():
            used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            self.peak = max(self.peak, used)
            self.samples += 1
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.sample_boundary()
        if self._nvml is not None:
            self.thread = threading.Thread(target=self._sample_nvml, daemon=True)
            self.thread.start()

    def sample_boundary(self) -> None:
        if self._nvml is not None and self._handle is not None:
            used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        else:
            free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
            used = int(total_bytes - free_bytes)
        self.peak = max(self.peak, used)
        self.samples += 1

    def finish(self) -> dict[str, object]:
        self.sample_boundary()
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return {
            "sampled_peak_device_used_bytes": int(self.peak),
            "peak_vram_measurement_method": self.method,
            "peak_vram_sample_count": int(self.samples),
            "peak_vram_poll_interval_ms": 1000.0 * self.interval_seconds
            if self.method == "NVML_poll"
            else None,
        }


def _cuda_timed(cp: Any, operation: Any) -> tuple[Any, float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    return output, float(cp.cuda.get_elapsed_time(start, end))


@dataclass(frozen=True)
class GpuLifted:
    roots: Any
    detail_levels: tuple[Any, ...]


def lift_forward_device(
    cp: Any,
    leaves: Any,
    alphabet: int,
    permutation: Any,
    selectors: Any,
) -> GpuLifted:
    """Exact modular tree lifting on already-resident CuPy arrays."""

    require(alphabet in (2, 4), "GPU alphabet")
    require(leaves.ndim == 2 and leaves.dtype == cp.uint8, "GPU leaf array")
    vectors, lanes = (int(value) for value in leaves.shape)
    require(vectors > 0 and lanes > 0, "GPU positive geometry")
    require(permutation.shape == (lanes,) and selectors.shape == (lanes - 1,), "GPU metadata")
    current = cp.ascontiguousarray(leaves[:, permutation.astype(cp.int64, copy=False)])
    levels: list[Any] = []
    selector_offset = 0
    for pairs in level_pair_counts(lanes):
        codes = selectors[selector_offset : selector_offset + pairs]
        selector_offset += pairs
        left = current[:, 0 : 2 * pairs : 2].astype(cp.int16)
        right = current[:, 1 : 2 * pairs : 2].astype(cp.int16)
        swap = (((codes >> 2) & 1).astype(cp.bool_))[None, :]
        p = (((codes >> 1) & 1).astype(cp.int16))[None, :]
        u = ((codes & 1).astype(cp.int16))[None, :]
        x = cp.where(swap, right, left)
        y = cp.where(swap, left, right)
        detail = cp.mod(y - p * x, alphabet).astype(cp.uint8)
        coarse = cp.mod(x + u * detail.astype(cp.int16), alphabet).astype(cp.uint8)
        if int(current.shape[1]) & 1:
            current = cp.concatenate((coarse, current[:, -1:]), axis=1)
        else:
            current = coarse
        levels.append(cp.ascontiguousarray(detail))
    require(selector_offset == lanes - 1 and current.shape == (vectors, 1), "GPU tree coverage")
    return GpuLifted(cp.ascontiguousarray(current[:, 0]), tuple(levels))


def flatten_details_device(cp: Any, lifted: GpuLifted) -> Any:
    if not lifted.detail_levels:
        return cp.empty(0, dtype=cp.uint8)
    return cp.ascontiguousarray(
        cp.concatenate([cp.ascontiguousarray(level).reshape(-1) for level in reversed(lifted.detail_levels)])
    ).astype(cp.uint8, copy=False)


def lift_inverse_device(
    cp: Any,
    lifted: GpuLifted,
    lanes: int,
    alphabet: int,
    permutation: Any,
    selectors: Any,
) -> Any:
    """Exact inverse, including every odd carry, on CuPy arrays."""

    vectors = int(lifted.roots.shape[0])
    counts = level_pair_counts(lanes)
    offsets = np.cumsum([0] + counts).tolist()
    widths = level_sizes(lanes)
    current = cp.ascontiguousarray(lifted.roots[:, None])
    for depth in range(len(counts) - 1, -1, -1):
        pairs = counts[depth]
        previous_width = widths[depth]
        codes = selectors[offsets[depth] : offsets[depth + 1]]
        detail = lifted.detail_levels[depth].astype(cp.int16)
        coarse = current[:, :pairs].astype(cp.int16)
        swap = (((codes >> 2) & 1).astype(cp.bool_))[None, :]
        p = (((codes >> 1) & 1).astype(cp.int16))[None, :]
        u = ((codes & 1).astype(cp.int16))[None, :]
        x = cp.mod(coarse - u * detail, alphabet)
        y = cp.mod(detail + p * x, alphabet)
        left = cp.where(swap, y, x).astype(cp.uint8)
        right = cp.where(swap, x, y).astype(cp.uint8)
        previous = cp.empty((vectors, previous_width), dtype=cp.uint8)
        previous[:, 0 : 2 * pairs : 2] = left
        previous[:, 1 : 2 * pairs : 2] = right
        if previous_width & 1:
            require(current.shape[1] == pairs + 1, "GPU odd carry geometry")
            previous[:, -1] = current[:, -1]
        else:
            require(current.shape[1] == pairs, "GPU even geometry")
        current = previous
    result = cp.empty_like(current)
    result[:, permutation.astype(cp.int64, copy=False)] = current
    return result


@dataclass(frozen=True)
class MetadataSearchResult:
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
    require_rtx_5090: bool = False,
) -> MetadataSearchResult:
    """Search metadata by exact held-out finite arithmetic codelength.

    Candidate transforms *must* execute through CuPy.  Q16 fitting and the
    finite arithmetic score use the CPU reference after D2H so the objective is
    the real serialized-symbol codelength, not a differential-entropy proxy.
    Every candidate has identical metadata/model byte counts, but those fixed
    charged bits are included in the reported objective.
    """

    cp, device = require_cupy(require_rtx_5090=require_rtx_5090)
    require(train_leaves.ndim == 2 and validation_leaves.ndim == 2, "search matrices")
    require(train_leaves.dtype == np.uint8 and validation_leaves.dtype == np.uint8, "search dtype")
    require(train_leaves.shape[1] == validation_leaves.shape[1], "search lanes")
    require(train_leaves.shape[0] > 0 and validation_leaves.shape[0] > 0, "search vectors")
    require(bool(np.all(train_leaves < alphabet)) and bool(np.all(validation_leaves < alphabet)), "search alphabet")
    seeds = [int(value) for value in candidate_seeds]
    require(len(seeds) > 0 and len(set(seeds)) == len(seeds), "unique search candidates")
    lanes = int(train_leaves.shape[1])

    sampler = _PeakVramSampler(cp, int(device["device_index"]))
    sampler.start()
    wall_start = time.perf_counter()
    h2d_ms = 0.0
    kernel_ms = 0.0
    d2h_ms = 0.0
    cpu_score_ms = 0.0

    (gpu_train, gpu_validation), elapsed = _cuda_timed(
        cp,
        lambda: (
            cp.asarray(np.ascontiguousarray(train_leaves)),
            cp.asarray(np.ascontiguousarray(validation_leaves)),
        ),
    )
    h2d_ms += elapsed
    sampler.sample_boundary()
    rows: list[dict[str, object]] = []
    candidate_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    metadata_bits = 8 * (
        permutation_byte_count(lanes) + (3 * (lanes - 1) + 7) // 8
    )

    for seed in seeds:
        permutation = np.asarray(deterministic_permutation(lanes, seed), dtype=np.int64)
        selectors = np.asarray(deterministic_selectors(lanes, seed ^ 0x5A17), dtype=np.uint8)
        (gpu_permutation, gpu_selectors), elapsed = _cuda_timed(
            cp, lambda: (cp.asarray(permutation), cp.asarray(selectors))
        )
        h2d_ms += elapsed
        (train_lifted, validation_lifted), elapsed = _cuda_timed(
            cp,
            lambda: (
                lift_forward_device(cp, gpu_train, alphabet, gpu_permutation, gpu_selectors),
                lift_forward_device(cp, gpu_validation, alphabet, gpu_permutation, gpu_selectors),
            ),
        )
        kernel_ms += elapsed
        (train_roots, train_details, validation_roots, validation_details), elapsed = _cuda_timed(
            cp,
            lambda: (
                cp.asnumpy(train_lifted.roots),
                cp.asnumpy(flatten_details_device(cp, train_lifted)),
                cp.asnumpy(validation_lifted.roots),
                cp.asnumpy(flatten_details_device(cp, validation_lifted)),
            ),
        )
        d2h_ms += elapsed
        sampler.sample_boundary()
        score_start = time.perf_counter()
        model = fit_model(alphabet, train_roots, train_details)
        finite_packet, meaningful_bits = encode_coefficients(
            model, validation_roots, validation_details
        )
        cpu_score_ms += 1000.0 * (time.perf_counter() - score_start)
        objective_bits = int(meaningful_bits + metadata_bits)
        rows.append(
            {
                "seed": seed,
                "finite_meaningful_bits": int(meaningful_bits),
                "finite_payload_bytes": len(finite_packet),
                "charged_metadata_bits": int(metadata_bits),
                "charged_objective_bits": objective_bits,
                "validation_symbols": int(validation_leaves.size),
                "charged_objective_bits_per_symbol": objective_bits
                / float(validation_leaves.size),
            }
        )
        candidate_cache[seed] = (
            train_roots,
            train_details,
            validation_roots,
            validation_details,
        )

    rows.sort(key=lambda row: (int(row["charged_objective_bits"]), int(row["seed"])))
    selected_seed = int(rows[0]["seed"])
    selected_permutation = deterministic_permutation(lanes, selected_seed)
    selected_selectors = deterministic_selectors(lanes, selected_seed ^ 0x5A17)

    # Mandatory cross-backend equality for the selected candidate.
    cpu_train = lift_forward(train_leaves, alphabet, selected_permutation, selected_selectors)
    cpu_validation = lift_forward(validation_leaves, alphabet, selected_permutation, selected_selectors)
    cached = candidate_cache[selected_seed]
    require(np.array_equal(cpu_train.roots, cached[0]), "CPU/CuPy train roots")
    require(np.array_equal(flatten_details(cpu_train), cached[1]), "CPU/CuPy train details")
    require(np.array_equal(cpu_validation.roots, cached[2]), "CPU/CuPy validation roots")
    require(np.array_equal(flatten_details(cpu_validation), cached[3]), "CPU/CuPy validation details")

    (gpu_permutation, gpu_selectors), elapsed = _cuda_timed(
        cp,
        lambda: (
            cp.asarray(np.asarray(selected_permutation, dtype=np.int64)),
            cp.asarray(np.asarray(selected_selectors, dtype=np.uint8)),
        ),
    )
    h2d_ms += elapsed
    selected_gpu, elapsed = _cuda_timed(
        cp,
        lambda: lift_forward_device(
            cp, gpu_validation, alphabet, gpu_permutation, gpu_selectors
        ),
    )
    kernel_ms += elapsed
    inverse_gpu, elapsed = _cuda_timed(
        cp,
        lambda: lift_inverse_device(
            cp,
            selected_gpu,
            lanes,
            alphabet,
            gpu_permutation,
            gpu_selectors,
        ),
    )
    kernel_ms += elapsed
    inverse_cpu, elapsed = _cuda_timed(cp, lambda: cp.asnumpy(inverse_gpu))
    d2h_ms += elapsed
    require(np.array_equal(inverse_cpu, validation_leaves), "CuPy inverse leaf roundtrip")
    sampler.sample_boundary()

    wall_ms = 1000.0 * (time.perf_counter() - wall_start)
    memory = sampler.finish()
    telemetry: dict[str, object] = {
        **device,
        **memory,
        "h2d_ms": h2d_ms,
        "kernel_ms": kernel_ms,
        "d2h_ms": d2h_ms,
        "cpu_reference_scoring_ms": cpu_score_ms,
        "wall_ms": wall_ms,
        "candidate_count": len(seeds),
        "train_shape": list(train_leaves.shape),
        "validation_shape": list(validation_leaves.shape),
        "cpu_cupy_selected_coefficients_equal": True,
        "cupy_inverse_roundtrip_equal": True,
        "timing_source": "CUDA events for H2D/kernel/D2H; perf_counter for wall",
        "telemetry_values_measured_not_inferred": True,
    }
    return MetadataSearchResult(
        selected_seed,
        tuple(selected_permutation),
        tuple(selected_selectors),
        tuple(rows),
        telemetry,
    )

