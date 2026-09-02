#!/usr/bin/env python3
"""CuPy RawKernel backend for the frozen sparse unifilar bank.

This file must only be imported after external source-manifest bootstrap and
independent-review authentication.  It intentionally supports CuPy only: no
Torch, JAX, NumPy fitting path, or dense chi-by-chi contraction exists here.
Each CUDA thread processes one canonical stream sequentially.  Across a fixed
150-cell topology bank, work is O(N) in the number of selected symbols.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import time
import weakref
from typing import Any


CUDA_SOURCE = r'''
extern "C" {

__device__ __forceinline__ unsigned int mix32(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    return value ^ (value >> 16);
}

__device__ __forceinline__ unsigned int context_of(
    unsigned char level, unsigned short base_freq, unsigned long long within
) {
    unsigned int bucket = (((unsigned int)base_freq) * 16u) >> 16;
    if (bucket > 15u) bucket = 15u;
    return ((((unsigned int)level) * 16u + bucket) * 4u) + ((unsigned int)within & 3u);
}

__device__ __forceinline__ unsigned int next_state(
    int topology,
    unsigned int states,
    unsigned int state,
    unsigned int bit,
    unsigned int context,
    unsigned long long within
) {
    const unsigned int mask = states - 1u;
    if (topology == 0) {
        return ((state << 1) | bit) & mask;
    }
    if (topology == 1) {
        if (bit == 0u) return state;
        unsigned int sketch = mix32(0xA511E9B3u ^ context ^ ((unsigned int)within * 0x9E3779B1u)) & mask;
        if (sketch == 0u) sketch = 1u;
        return state ^ sketch;
    }
    if (topology == 2) {
        unsigned int weight = (mix32(0x63D83595u ^ context ^ (((unsigned int)within & 3u) << 20)) & mask) | 1u;
        return (state + weight * bit) & mask;
    }
    if (topology == 3) {
        unsigned int multiplier = (states >= 8u ? 5u : 1u) & mask;
        unsigned int addend = mix32(0xB5297A4Du ^ context ^ (((unsigned int)within & 3u) << 24)) & mask;
        return (multiplier * state + addend + bit) & mask;
    }
    if (bit) return state < mask ? state + 1u : mask;
    return state > 0u ? state - 1u : 0u;
}

__global__ void count_streams(
    const unsigned char* bits,
    const unsigned char* levels,
    const unsigned short* base_freq,
    const unsigned long long* offsets,
    const unsigned long long* lengths,
    unsigned int stream_count,
    int topology,
    unsigned int states,
    unsigned int reset_length,
    unsigned long long* counts
) {
    const unsigned int stream = blockDim.x * blockIdx.x + threadIdx.x;
    if (stream >= stream_count) return;
    const unsigned long long start = offsets[stream];
    const unsigned long long length = lengths[stream];
    unsigned int state = 0u;
    for (unsigned long long position = 0; position < length; ++position) {
        const unsigned long long within = position % (unsigned long long)reset_length;
        if (within == 0ull) state = 0u;
        const unsigned long long index = start + position;
        const unsigned int bit = (unsigned int)bits[index];
        const unsigned int context = context_of(levels[index], base_freq[index], within);
        const unsigned long long count_index = (((unsigned long long)state * 384ull + context) << 1) + bit;
        atomicAdd(&counts[count_index], 1ull);
        state = next_state(topology, states, state, bit, context, within);
    }
}

__global__ void exact_arithmetic_lengths(
    const unsigned char* bits,
    const unsigned char* levels,
    const unsigned short* base_freq,
    const unsigned long long* offsets,
    const unsigned long long* lengths,
    unsigned int stream_count,
    int topology,
    unsigned int states,
    unsigned int reset_length,
    const unsigned short* frequencies,
    unsigned long long* output_lengths
) {
    const unsigned int stream = blockDim.x * blockIdx.x + threadIdx.x;
    if (stream >= stream_count) return;
    const unsigned long long start = offsets[stream];
    const unsigned long long length = lengths[stream];
    unsigned int state = 0u;
    unsigned long long low = 0ull;
    unsigned long long high = 0xFFFFFFFFull;
    unsigned long long pending = 0ull;
    unsigned long long emitted = 0ull;
    for (unsigned long long position = 0; position < length; ++position) {
        const unsigned long long within = position % (unsigned long long)reset_length;
        if (within == 0ull) state = 0u;
        const unsigned long long index = start + position;
        const unsigned int bit = (unsigned int)bits[index];
        const unsigned int context = context_of(levels[index], base_freq[index], within);
        const unsigned int f1 = (unsigned int)frequencies[(unsigned long long)state * 384ull + context];
        const unsigned int f0 = 65536u - f1;
        const unsigned long long width = high - low + 1ull;
        const unsigned long long split = low + (width * (unsigned long long)f0 / 65536ull) - 1ull;
        if (bit == 0u) high = split; else low = split + 1ull;
        while (true) {
            if (high < 0x80000000ull) {
                emitted += 1ull + pending;
                pending = 0ull;
            } else if (low >= 0x80000000ull) {
                emitted += 1ull + pending;
                pending = 0ull;
                low -= 0x80000000ull;
                high -= 0x80000000ull;
            } else if (low >= 0x40000000ull && high < 0xC0000000ull) {
                pending += 1ull;
                low -= 0x40000000ull;
                high -= 0x40000000ull;
            } else {
                break;
            }
            low = (low << 1) & 0xFFFFFFFFull;
            high = ((high << 1) & 0xFFFFFFFFull) | 1ull;
        }
        state = next_state(topology, states, state, bit, context, within);
    }
    // The canonical encoder performs pending += 1 followed by one emit.
    output_lengths[stream] = emitted + pending + 2ull;
}

} // extern C
'''


def _kernels(cp: Any) -> tuple[Any, Any]:
    module = cp.RawModule(
        code=CUDA_SOURCE,
        options=("--std=c++11",),
        name_expressions=("count_streams", "exact_arithmetic_lengths"),
    )
    return module.get_function("count_streams"), module.get_function("exact_arithmetic_lengths")


MAX_STREAMS = 65_536
MAX_HOST_BYTES = 96 * (1 << 30)
MAX_VRAM_BYTES = 28 * (1 << 30)
HOST_ALLOCATION_RESERVE_BYTES = 1 << 30
VRAM_ALLOCATION_RESERVE_BYTES = 2 * (1 << 30)
MAX_AUXILIARY_DEVICE_BYTES = 64 * 384 * 2 * 8 + 64 * 384 * 2 + 40 * MAX_STREAMS
MAX_PACKED_SYMBOLS = (MAX_VRAM_BYTES - VRAM_ALLOCATION_RESERVE_BYTES - MAX_AUXILIARY_DEVICE_BYTES) // 4
LEGAL_STATES = (2, 4, 8, 16, 32, 64)
LEGAL_RESETS = (32, 128, 512, 2048, 4096)
CONTEXTS = 384
LEVELS = 6


_PACKED_KEYS = frozenset({
    "bits",
    "levels",
    "base_freq",
    "offsets",
    "lengths",
    "stream_count",
    "symbol_count",
    "root_stream_indices",
    "host_offsets",
    "host_lengths",
})


class _PackedHandle(dict):
    """Backend-registered mapping whose public fields cannot be reassigned.

    ``dict.__setitem__`` can still be invoked explicitly on a subclass, so the
    backend also compares every field with its private registration before any
    CuPy operation.  Immutability is therefore a convenience, not the trust
    boundary; registration and full revalidation are the trust boundary.
    """

    def __init__(self, values: dict[str, Any], token: object):
        dict.__init__(self, values)
        self._uwfa_backend_token = token

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("packed stream handles are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _linux_process_tree_rss() -> tuple[int, int]:
    """Return current process-tree RSS and maximum observed VmHWM bytes."""
    if os.name != "posix" or not os.path.isdir("/proc"):
        return 0, 0
    pending = [os.getpid()]
    seen: set[int] = set()
    rss_total = 0
    hwm_max = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/status", "r", encoding="ascii") as handle:
                values = {}
                for line in handle:
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        name, value, _unit = line.split()
                        values[name[:-1]] = int(value) * 1024
                rss_total += values.get("VmRSS", 0)
                hwm_max = max(hwm_max, values.get("VmHWM", 0))
            with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="ascii") as handle:
                pending.extend(int(value) for value in handle.read().split())
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return rss_total, hwm_max


class CuPyUnifilarBackend:
    """Validated exact CUDA path with complete transfer and peak telemetry."""

    def __init__(self, cp: Any):
        if sys.byteorder != "little":
            raise RuntimeError("UWFA CuPy backend requires little-endian host")
        self.cp = cp
        self._packed_backend_token = object()
        self._packed_registry: dict[int, dict[str, Any]] = {}
        started = time.perf_counter()
        self.count_kernel, self.length_kernel = _kernels(cp)
        cp.cuda.runtime.deviceSynchronize()
        compile_seconds = time.perf_counter() - started
        free, total = cp.cuda.runtime.memGetInfo()
        host_rss, host_hwm = _linux_process_tree_rss()
        self._baseline_free_vram = int(free)
        self._minimum_free_vram = int(free)
        self._baseline_host_rss = int(host_rss)
        self.stats = {
            "h2d_bytes": 0,
            "h2d_payload_bytes": 0,
            "h2d_root_descriptor_bytes": 0,
            "h2d_subset_descriptor_bytes": 0,
            "h2d_launch_descriptor_bytes": 0,
            "h2d_model_table_bytes": 0,
            "h2d_kernel_scalar_bytes": 0,
            "d2h_bytes": 0,
            "d2d_descriptor_bytes": 0,
            "device_output_allocation_bytes": 0,
            "kernel_count": 0,
            "count_kernel_count": 0,
            "length_kernel_count": 0,
            "count_cell_symbol_updates": 0,
            "length_cell_symbol_updates": 0,
            "pack_calls": 0,
            "subset_calls": 0,
            "to_host_calls": 0,
            "jit_compile_seconds": compile_seconds,
            "kernel_wall_seconds": 0.0,
            "telemetry_samples": 0,
            "peak_process_tree_rss_bytes": int(host_rss),
            "peak_process_hwm_bytes": int(host_hwm),
            "incremental_peak_process_tree_rss_bytes": 0,
            "peak_vram_incremental_bytes": 0,
            "peak_default_pool_used_bytes": int(cp.get_default_memory_pool().used_bytes()),
            "peak_default_pool_total_bytes": int(cp.get_default_memory_pool().total_bytes()),
            "peak_pinned_pool_free_blocks": int(cp.get_default_pinned_memory_pool().n_free_blocks()),
            "baseline_free_vram_bytes": int(free),
            "total_vram_bytes": int(total),
        }
        self.samples: list[dict[str, Any]] = []
        self._sample("post_jit")

    def _registry(self) -> tuple[object, dict[int, dict[str, Any]]]:
        """Return the private registry, including for minimal test doubles."""
        token = getattr(self, "_packed_backend_token", None)
        registry = getattr(self, "_packed_registry", None)
        if token is None or not isinstance(registry, dict):
            token = object()
            registry = {}
            self._packed_backend_token = token
            self._packed_registry = registry
        return token, registry

    @staticmethod
    def _exact_nonnegative(value: Any, label: str, maximum: int) -> int:
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(label)
        return value

    @staticmethod
    def _device_vector(
        value: Any,
        *,
        expected_identity: Any,
        dtype_name: str,
        count: int,
        itemsize: int,
        label: str,
    ) -> None:
        """Validate device-array metadata without copying or touching CUDA."""
        if value is not expected_identity:
            raise ValueError(f"{label} identity")
        if str(getattr(value, "dtype", "")) != dtype_name:
            raise ValueError(f"{label} dtype")
        if type(getattr(value, "ndim", None)) is not int or value.ndim != 1:
            raise ValueError(f"{label} rank")
        if type(getattr(value, "size", None)) is not int or value.size != count:
            raise ValueError(f"{label} element count")
        if tuple(getattr(value, "shape", ())) != (count,):
            raise ValueError(f"{label} shape")
        if type(getattr(value, "nbytes", None)) is not int or value.nbytes != count * itemsize:
            raise ValueError(f"{label} byte count")
        flags = getattr(value, "flags", None)
        if flags is None or not bool(getattr(flags, "c_contiguous", False)):
            raise ValueError(f"{label} contiguous layout")

    def _register_packed(
        self,
        values: dict[str, Any],
        *,
        kind: str,
        root_offsets: tuple[int, ...],
        root_lengths: tuple[int, ...],
        payload_symbol_count: int,
    ) -> _PackedHandle:
        token, registry = self._registry()
        handle = _PackedHandle(values, token)
        handle_id = id(handle)

        def forget(_reference: Any, *, key: int = handle_id, table: dict[int, dict[str, Any]] = registry) -> None:
            table.pop(key, None)

        record = {
            "handle_ref": weakref.ref(handle, forget),
            "kind": kind,
            "bits": values["bits"],
            "levels": values["levels"],
            "base_freq": values["base_freq"],
            "offsets": values["offsets"],
            "lengths": values["lengths"],
            "stream_count": values["stream_count"],
            "symbol_count": values["symbol_count"],
            "root_stream_indices": values["root_stream_indices"],
            "host_offsets": values["host_offsets"],
            "host_lengths": values["host_lengths"],
            "root_offsets": root_offsets,
            "root_lengths": root_lengths,
            "payload_symbol_count": payload_symbol_count,
        }
        registry[handle_id] = record
        try:
            self._validate_packed(handle)
        except Exception:
            registry.pop(handle_id, None)
            raise
        return handle

    def _validate_packed(self, packed: Any) -> dict[str, Any]:
        """Authenticate and completely validate a packed handle on the host."""
        token, registry = self._registry()
        if type(packed) is not _PackedHandle or getattr(packed, "_uwfa_backend_token", None) is not token:
            raise ValueError("unregistered GPU packed handle")
        record = registry.get(id(packed))
        if record is None or record["handle_ref"]() is not packed:
            raise ValueError("unknown GPU packed provenance")
        if set(dict.keys(packed)) != _PACKED_KEYS:
            raise ValueError("GPU packed field set")
        for name in ("bits", "levels", "base_freq", "offsets", "lengths"):
            if dict.__getitem__(packed, name) is not record[name]:
                raise ValueError(f"GPU packed {name} changed")
        for name in ("stream_count", "symbol_count", "root_stream_indices", "host_offsets", "host_lengths"):
            if dict.__getitem__(packed, name) != record[name]:
                raise ValueError(f"GPU packed {name} changed")

        stream_count = self._exact_nonnegative(record["stream_count"], "GPU packed stream count", MAX_STREAMS)
        if stream_count == 0:
            raise ValueError("GPU packed stream count")
        symbol_count = self._exact_nonnegative(record["symbol_count"], "GPU packed symbol count", MAX_PACKED_SYMBOLS)
        if symbol_count == 0:
            raise ValueError("GPU packed symbol count")
        payload_symbols = self._exact_nonnegative(record["payload_symbol_count"], "GPU packed payload symbols", MAX_PACKED_SYMBOLS)
        if payload_symbols == 0:
            raise ValueError("GPU packed payload symbols")

        root_offsets = record["root_offsets"]
        root_lengths = record["root_lengths"]
        root_indices = record["root_stream_indices"]
        host_offsets = record["host_offsets"]
        host_lengths = record["host_lengths"]
        if not all(type(value) is tuple for value in (root_offsets, root_lengths, root_indices, host_offsets, host_lengths)):
            raise ValueError("GPU packed descriptor tuple types")
        if len(root_offsets) != len(root_lengths) or not 1 <= len(root_offsets) <= MAX_STREAMS:
            raise ValueError("GPU packed root descriptor geometry")
        if not (len(root_indices) == len(host_offsets) == len(host_lengths) == stream_count):
            raise ValueError("GPU packed launch descriptor geometry")
        if any(type(index) is not int for index in root_indices) or tuple(sorted(set(root_indices))) != root_indices:
            raise ValueError("GPU packed root indices")

        root_cursor = 0
        for begin, length in zip(root_offsets, root_lengths, strict=True):
            begin = self._exact_nonnegative(begin, "GPU root offset", payload_symbols)
            length = self._exact_nonnegative(length, "GPU root length", payload_symbols)
            if length == 0 or begin != root_cursor or length > payload_symbols - begin:
                raise ValueError("GPU root descriptor coverage")
            root_cursor = begin + length
        if root_cursor != payload_symbols:
            raise ValueError("GPU root descriptor terminal coverage")

        selected_symbols = 0
        for root_index, begin, length in zip(root_indices, host_offsets, host_lengths, strict=True):
            if not 0 <= root_index < len(root_offsets):
                raise ValueError("GPU packed root index bound")
            begin = self._exact_nonnegative(begin, "GPU packed offset", payload_symbols)
            length = self._exact_nonnegative(length, "GPU packed length", payload_symbols)
            if length == 0 or length > payload_symbols - begin:
                raise ValueError("GPU packed offset/length range")
            if begin != root_offsets[root_index] or length != root_lengths[root_index]:
                raise ValueError("GPU packed descriptor/root mismatch")
            if selected_symbols > MAX_PACKED_SYMBOLS - length:
                raise ValueError("GPU packed selected-symbol sum")
            selected_symbols += length
        if selected_symbols != symbol_count:
            raise ValueError("GPU packed selected-symbol count")

        self._device_vector(record["bits"], expected_identity=record["bits"], dtype_name="uint8", count=payload_symbols, itemsize=1, label="GPU bits")
        self._device_vector(record["levels"], expected_identity=record["levels"], dtype_name="uint8", count=payload_symbols, itemsize=1, label="GPU levels")
        self._device_vector(record["base_freq"], expected_identity=record["base_freq"], dtype_name="uint16", count=payload_symbols, itemsize=2, label="GPU base frequencies")
        self._device_vector(record["offsets"], expected_identity=record["offsets"], dtype_name="uint64", count=stream_count, itemsize=8, label="GPU stored offsets")
        self._device_vector(record["lengths"], expected_identity=record["lengths"], dtype_name="uint64", count=stream_count, itemsize=8, label="GPU stored lengths")
        return record

    def _fresh_launch_descriptors(self, record: dict[str, Any]) -> tuple[Any, Any]:
        """Copy fresh validated uint64 launch descriptors and charge both copies."""
        cp = self.cp
        stream_count = record["stream_count"]
        offsets = cp.asarray(list(record["host_offsets"]), dtype=cp.uint64)
        self._add_h2d("h2d_launch_descriptor_bytes", 8 * stream_count)
        self._device_vector(offsets, expected_identity=offsets, dtype_name="uint64", count=stream_count, itemsize=8, label="GPU launch offsets")
        lengths = cp.asarray(list(record["host_lengths"]), dtype=cp.uint64)
        self._add_h2d("h2d_launch_descriptor_bytes", 8 * stream_count)
        self._device_vector(lengths, expected_identity=lengths, dtype_name="uint64", count=stream_count, itemsize=8, label="GPU launch lengths")
        return offsets, lengths

    def _add_h2d(self, category: str, amount: int) -> None:
        if type(amount) is not int or amount < 0:
            raise ValueError("H2D accounting amount")
        self.stats.setdefault("h2d_bytes", 0)
        self.stats.setdefault(category, 0)
        self.stats["h2d_bytes"] += amount
        self.stats[category] += amount

    def _sample(self, phase: str) -> None:
        cp = self.cp
        cp.cuda.runtime.deviceSynchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        host_rss, host_hwm = _linux_process_tree_rss()
        if int(free) <= 0 or int(total) <= 0 or int(free) > int(total):
            raise RuntimeError("fatal invalid CUDA memory telemetry")
        if int(host_rss) <= 0 or int(host_hwm) <= 0:
            raise RuntimeError("fatal unavailable process RSS/HWM telemetry")
        self._minimum_free_vram = min(self._minimum_free_vram, int(free))
        pool = cp.get_default_memory_pool()
        pinned = cp.get_default_pinned_memory_pool()
        self.stats["telemetry_samples"] += 1
        self.stats["peak_process_tree_rss_bytes"] = max(self.stats["peak_process_tree_rss_bytes"], int(host_rss))
        self.stats["peak_process_hwm_bytes"] = max(self.stats["peak_process_hwm_bytes"], int(host_hwm))
        self.stats["incremental_peak_process_tree_rss_bytes"] = max(0, self.stats["peak_process_tree_rss_bytes"] - self._baseline_host_rss)
        self.stats["peak_vram_incremental_bytes"] = max(self.stats["peak_vram_incremental_bytes"], self._baseline_free_vram - self._minimum_free_vram)
        self.stats["peak_default_pool_used_bytes"] = max(self.stats["peak_default_pool_used_bytes"], int(pool.used_bytes()))
        self.stats["peak_default_pool_total_bytes"] = max(self.stats["peak_default_pool_total_bytes"], int(pool.total_bytes()))
        self.stats["peak_pinned_pool_free_blocks"] = max(self.stats["peak_pinned_pool_free_blocks"], int(pinned.n_free_blocks()))
        self.samples.append({
            "phase": str(phase),
            "process_tree_rss_bytes": int(host_rss),
            "process_hwm_bytes": int(host_hwm),
            "free_vram_bytes": int(free),
            "total_vram_bytes": int(total),
            "default_pool_used_bytes": int(pool.used_bytes()),
            "default_pool_total_bytes": int(pool.total_bytes()),
            "pinned_pool_free_blocks": int(pinned.n_free_blocks()),
        })

    @staticmethod
    def _validate_candidate(topology_id: Any, states: Any, reset_length: Any) -> tuple[int, int, int]:
        if type(topology_id) is not int or not 0 <= topology_id <= 4:
            raise ValueError("GPU topology id")
        if type(states) is not int or states not in LEGAL_STATES:
            raise ValueError("GPU state count")
        if type(reset_length) is not int or reset_length not in LEGAL_RESETS:
            raise ValueError("GPU reset length")
        return topology_id, states, reset_length

    def pack_streams(self, streams: list[tuple[bytes, bytes, bytes]]) -> dict[str, Any]:
        if not isinstance(streams, list) or not 1 <= len(streams) <= MAX_STREAMS:
            raise ValueError("packed stream count")
        offsets: list[int] = []
        lengths: list[int] = []
        bit_parts: list[bytes] = []
        level_parts: list[bytes] = []
        frequency_parts: list[bytes] = []
        offset = 0
        for bits, levels, base_frequency in streams:
            if not isinstance(bits, bytes) or not isinstance(levels, bytes) or not isinstance(base_frequency, bytes):
                raise ValueError("packed stream byte types")
            if not bits or len(bits) != len(levels) or len(base_frequency) != 2 * len(bits):
                raise ValueError("packed stream geometry")
            if any(value > 1 for value in bits) or any(value >= LEVELS for value in levels):
                raise ValueError("packed bit/level alphabet")
            if offset > MAX_PACKED_SYMBOLS - len(bits):
                raise ValueError("packed symbol bound")
            for position in range(0, len(base_frequency), 2):
                frequency = base_frequency[position] | (base_frequency[position + 1] << 8)
                if not 1 <= frequency <= 65535:
                    raise ValueError("packed base frequency")
            offsets.append(offset)
            lengths.append(len(bits))
            offset += len(bits)
            bit_parts.append(bits)
            level_parts.append(levels)
            frequency_parts.append(base_frequency)
        descriptor_cursor = 0
        for begin, length in zip(offsets, lengths, strict=True):
            if type(begin) is not int or type(length) is not int or length <= 0:
                raise ValueError("packed descriptor integer geometry")
            if begin != descriptor_cursor or begin > offset or length > offset - begin:
                raise ValueError("packed descriptor coverage")
            descriptor_cursor = begin + length
        if descriptor_cursor != offset or not 1 <= offset <= MAX_PACKED_SYMBOLS:
            raise ValueError("packed descriptor terminal coverage")
        resource_plan = self.pack_resource_plan(offset, len(streams))
        if resource_plan["passes"] is not True:
            raise MemoryError("GPU pack resource admission failed before host/device allocation")
        cp = self.cp
        bit_blob = b"".join(bit_parts)
        level_blob = b"".join(level_parts)
        frequency_blob = b"".join(frequency_parts)
        device_bits = cp.frombuffer(bit_blob, dtype=cp.uint8)
        self._add_h2d("h2d_payload_bytes", len(bit_blob))
        device_levels = cp.frombuffer(level_blob, dtype=cp.uint8)
        self._add_h2d("h2d_payload_bytes", len(level_blob))
        device_base = cp.frombuffer(frequency_blob, dtype=cp.uint16)
        self._add_h2d("h2d_payload_bytes", len(frequency_blob))
        device_offsets = cp.asarray(offsets, dtype=cp.uint64)
        self._add_h2d("h2d_root_descriptor_bytes", 8 * len(streams))
        device_lengths = cp.asarray(lengths, dtype=cp.uint64)
        self._add_h2d("h2d_root_descriptor_bytes", 8 * len(streams))
        result = self._register_packed({
            "bits": device_bits,
            "levels": device_levels,
            "base_freq": device_base,
            "offsets": device_offsets,
            "lengths": device_lengths,
            "stream_count": len(streams),
            "symbol_count": offset,
            "root_stream_indices": tuple(range(len(streams))),
            "host_offsets": tuple(offsets),
            "host_lengths": tuple(lengths),
        }, kind="root", root_offsets=tuple(offsets), root_lengths=tuple(lengths), payload_symbol_count=offset)
        self.stats["pack_calls"] += 1
        self.stats["resource_preflight_calls"] = int(self.stats.get("resource_preflight_calls", 0)) + 1
        self.stats["last_pack_resource_plan"] = resource_plan
        self._sample("pack_streams")
        return result

    def pack_resource_plan(self, symbol_count: Any, stream_count: Any) -> dict[str, Any]:
        """Exact fail-closed admission before concatenation or CuPy allocation."""
        symbols = self._exact_nonnegative(symbol_count, "resource symbols", MAX_PACKED_SYMBOLS)
        streams = self._exact_nonnegative(stream_count, "resource streams", MAX_STREAMS)
        if symbols == 0 or streams == 0:
            raise ValueError("resource plan nonempty geometry")
        payload_bytes = 4 * symbols
        descriptor_bytes = 16 * streams
        additional_host_bytes = payload_bytes + 64 * streams + HOST_ALLOCATION_RESERVE_BYTES
        device_required_bytes = (
            payload_bytes + descriptor_bytes + MAX_AUXILIARY_DEVICE_BYTES
            + VRAM_ALLOCATION_RESERVE_BYTES
        )
        host_rss, host_hwm = _linux_process_tree_rss()
        if host_rss <= 0 or host_hwm <= 0:
            raise RuntimeError("fatal unavailable host telemetry before allocation")
        free_vram, total_vram = self.cp.cuda.runtime.memGetInfo()
        if int(free_vram) <= 0 or int(total_vram) <= 0 or int(free_vram) > int(total_vram):
            raise RuntimeError("fatal unavailable VRAM telemetry before allocation")
        host_after = host_rss + additional_host_bytes
        passes = (
            host_after <= MAX_HOST_BYTES
            and device_required_bytes <= MAX_VRAM_BYTES
            and device_required_bytes <= int(free_vram)
        )
        return {
            "symbols": symbols,
            "streams": streams,
            "payload_host_and_device_bytes": payload_bytes,
            "root_descriptor_device_bytes": descriptor_bytes,
            "additional_host_bytes_including_reserve": additional_host_bytes,
            "device_required_bytes_including_aux_and_reserve": device_required_bytes,
            "current_process_tree_rss_bytes": int(host_rss),
            "current_process_hwm_bytes": int(host_hwm),
            "projected_process_tree_rss_bytes": int(host_after),
            "current_free_vram_bytes": int(free_vram),
            "current_total_vram_bytes": int(total_vram),
            "host_cap_bytes": MAX_HOST_BYTES,
            "vram_cap_bytes": MAX_VRAM_BYTES,
            "passes": bool(passes),
            "checked_before_blob_concatenation_or_cupy_allocation": True,
        }

    def subset(self, packed: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        registration = self._validate_packed(packed)
        root_count = registration["stream_count"]
        if not isinstance(indices, list) or not 1 <= len(indices) <= root_count:
            raise ValueError("GPU subset count")
        if any(type(index) is not int or not 0 <= index < root_count for index in indices):
            raise ValueError("GPU subset index")
        if indices != sorted(set(indices)):
            raise ValueError("GPU subset indices must be unique and sorted")
        host_offsets = [registration["host_offsets"][index] for index in indices]
        host_lengths = [registration["host_lengths"][index] for index in indices]
        root_stream_indices = tuple(registration["root_stream_indices"][index] for index in indices)
        symbol_count = 0
        for length in host_lengths:
            if symbol_count > MAX_PACKED_SYMBOLS - length:
                raise ValueError("GPU subset symbol count")
            symbol_count += length
        if not 1 <= symbol_count <= MAX_PACKED_SYMBOLS:
            raise ValueError("GPU subset symbol count")
        cp = self.cp
        device_offsets = cp.asarray(host_offsets, dtype=cp.uint64)
        self._add_h2d("h2d_subset_descriptor_bytes", 8 * len(indices))
        device_lengths = cp.asarray(host_lengths, dtype=cp.uint64)
        self._add_h2d("h2d_subset_descriptor_bytes", 8 * len(indices))
        result = self._register_packed({
            "bits": registration["bits"],
            "levels": registration["levels"],
            "base_freq": registration["base_freq"],
            "offsets": device_offsets,
            "lengths": device_lengths,
            "stream_count": len(indices),
            "symbol_count": symbol_count,
            "root_stream_indices": root_stream_indices,
            "host_offsets": tuple(host_offsets),
            "host_lengths": tuple(host_lengths),
        }, kind="subset", root_offsets=registration["root_offsets"], root_lengths=registration["root_lengths"], payload_symbol_count=registration["payload_symbol_count"])
        self.stats["subset_calls"] += 1
        self._sample("subset_descriptors")
        return result

    def fit_counts(self, packed: dict[str, Any], topology_id: int, states: int, reset_length: int) -> Any:
        topology_id, states, reset_length = self._validate_candidate(topology_id, states, reset_length)
        registration = self._validate_packed(packed)
        stream_count = registration["stream_count"]
        threads = 128
        blocks = (stream_count + threads - 1) // threads
        if not 1 <= blocks <= (1 << 31) - 1:
            raise ValueError("GPU count grid")
        cp = self.cp
        launch_offsets, launch_lengths = self._fresh_launch_descriptors(registration)
        output_count = states * CONTEXTS * 2
        counts = cp.zeros(output_count, dtype=cp.uint64)
        self._device_vector(counts, expected_identity=counts, dtype_name="uint64", count=output_count, itemsize=8, label="GPU count output")
        self.stats["device_output_allocation_bytes"] += counts.nbytes
        self._add_h2d("h2d_kernel_scalar_bytes", 16)
        started = time.perf_counter()
        self.count_kernel(
            (blocks,), (threads,),
            (registration["bits"], registration["levels"], registration["base_freq"], launch_offsets, launch_lengths, cp.uint32(stream_count), cp.int32(topology_id), cp.uint32(states), cp.uint32(reset_length), counts),
        )
        cp.cuda.runtime.deviceSynchronize()
        self.stats["kernel_wall_seconds"] += time.perf_counter() - started
        self.stats["kernel_count"] += 1
        self.stats["count_kernel_count"] += 1
        self.stats["count_cell_symbol_updates"] += registration["symbol_count"]
        self._sample("count_kernel")
        return counts

    def exact_lengths(self, packed: dict[str, Any], topology_id: int, states: int, reset_length: int, frequencies: Any) -> Any:
        topology_id, states, reset_length = self._validate_candidate(topology_id, states, reset_length)
        registration = self._validate_packed(packed)
        stream_count = registration["stream_count"]
        expected = states * CONTEXTS
        if not isinstance(frequencies, (tuple, list)) or len(frequencies) != expected:
            raise ValueError("GPU frequency geometry")
        host_model = []
        for value in frequencies:
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError("GPU Q0.16 frequency")
            host_model.append(value)
        threads = 128
        blocks = (stream_count + threads - 1) // threads
        if not 1 <= blocks <= (1 << 31) - 1:
            raise ValueError("GPU length grid")
        cp = self.cp
        launch_offsets, launch_lengths = self._fresh_launch_descriptors(registration)
        model = cp.asarray(host_model, dtype=cp.uint16)
        self._add_h2d("h2d_model_table_bytes", 2 * expected)
        self._device_vector(model, expected_identity=model, dtype_name="uint16", count=expected, itemsize=2, label="GPU model table")
        result = cp.zeros(stream_count, dtype=cp.uint64)
        self._device_vector(result, expected_identity=result, dtype_name="uint64", count=stream_count, itemsize=8, label="GPU length output")
        self.stats["device_output_allocation_bytes"] += result.nbytes
        self._add_h2d("h2d_kernel_scalar_bytes", 16)
        started = time.perf_counter()
        self.length_kernel(
            (blocks,), (threads,),
            (registration["bits"], registration["levels"], registration["base_freq"], launch_offsets, launch_lengths, cp.uint32(stream_count), cp.int32(topology_id), cp.uint32(states), cp.uint32(reset_length), model, result),
        )
        cp.cuda.runtime.deviceSynchronize()
        self.stats["kernel_wall_seconds"] += time.perf_counter() - started
        self.stats["kernel_count"] += 1
        self.stats["length_kernel_count"] += 1
        self.stats["length_cell_symbol_updates"] += registration["symbol_count"]
        self._sample("length_kernel")
        return result

    def to_host_list(self, value: Any) -> list[int]:
        if not hasattr(value, "get") or not hasattr(value, "nbytes"):
            if hasattr(value, "tolist"):
                return [int(item) for item in value.tolist()]
            return [int(item) for item in value]
        self.cp.cuda.runtime.deviceSynchronize()
        self.stats["d2h_bytes"] += int(value.nbytes)
        self.stats["to_host_calls"] += 1
        host = value.get()
        self._sample("device_to_host")
        return [int(item) for item in host.tolist()]

    def statistics_snapshot(self) -> dict[str, Any]:
        self._sample("statistics_snapshot")
        return dict(self.stats)

    @staticmethod
    def _canonical_device_uuid(raw: Any) -> str:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("GPU-"):
                text = text[4:]
            compact = text.replace("-", "").lower()
            if re.fullmatch(r"[0-9a-f]{32}", compact) is None:
                raise RuntimeError("fatal unavailable CUDA device UUID")
        else:
            try:
                packed = bytes(raw)
            except Exception as exc:
                raise RuntimeError("fatal unavailable CUDA device UUID") from exc
            if len(packed) != 16:
                raise RuntimeError("fatal unavailable CUDA device UUID")
            compact = packed.hex()
        return "GPU-" + "-".join((compact[:8], compact[8:12], compact[12:16], compact[16:20], compact[20:]))

    @staticmethod
    def _canonical_pci_bus_id(raw: Any) -> str:
        if isinstance(raw, bytes):
            try:
                text = raw.decode("ascii")
            except UnicodeDecodeError as exc:
                raise RuntimeError("fatal unavailable CUDA PCI bus id") from exc
        else:
            text = str(raw)
        match = re.fullmatch(r"([0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])", text.strip())
        if match is None:
            raise RuntimeError("fatal unavailable CUDA PCI bus id")
        domain, bus, device, function = match.groups()
        return f"{int(domain, 16):08x}:{bus.lower()}:{device.lower()}.{function}"

    def _device_identity(self, properties: dict[Any, Any], device_id: int) -> tuple[str, str]:
        runtime = self.cp.cuda.runtime
        raw_uuid = None
        get_uuid = getattr(runtime, "deviceGetUuid", None)
        if callable(get_uuid):
            raw_uuid = get_uuid(device_id)
        if raw_uuid is None:
            raw_uuid = properties.get("uuid", properties.get(b"uuid"))
        get_pci = getattr(runtime, "deviceGetPCIBusId", None)
        if not callable(get_pci):
            raise RuntimeError("fatal CUDA runtime lacks deviceGetPCIBusId")
        raw_pci = get_pci(device_id)
        return self._canonical_device_uuid(raw_uuid), self._canonical_pci_bus_id(raw_pci)

    def environment_receipt(self) -> dict[str, Any]:
        self._sample("environment_receipt")
        cp = self.cp
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        device_uuid, pci_bus_id = self._device_identity(properties, int(device.id))
        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "cupy_version": str(cp.__version__),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "python_version": sys.version,
            "platform": platform.platform(),
            "device_id": int(device.id),
            "device_name": properties["name"].decode("utf-8") if isinstance(properties["name"], bytes) else str(properties["name"]),
            "device_uuid": device_uuid,
            "pci_bus_id": pci_bus_id,
            "compute_capability": [int(properties["major"]), int(properties["minor"])],
            "current_free_vram_bytes": int(free),
            "total_vram_bytes": int(total),
            "statistics": dict(self.stats),
            "telemetry_samples": list(self.samples),
            "host_byteorder": sys.byteorder,
            "explicit_device_synchronization_at_phase_boundaries_and_after_every_kernel": True,
            "fatal_telemetry_sampling": True,
            "transfer_formula": {
                "root_pack_h2d": "4*N_symbols + 16*S_root",
                "subset_h2d": "16*S_subset per descriptor materialization",
                "launch_descriptor_h2d": "16*S_launch on every fit-count or exact-length launch",
                "model_h2d": "2*states*384 per exact-length call",
                "kernel_scalars_h2d": "16 per RawKernel launch",
                "d2h": "8*states*384*2 per count result plus 8*S_subset per length result",
            },
        }


def build_backend(cp: Any) -> CuPyUnifilarBackend:
    return CuPyUnifilarBackend(cp)
