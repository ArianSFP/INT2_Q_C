#!/usr/bin/env python3
"""Authenticated, bounded CuPy search for the hardened source-only gate.

Only the explicit fresh-process runner imports this module. It performs a
real rank-0/rank-1 label-flexible BMP search on device; it is not a generated
distortion-table smoke disguised as a search.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import types

import numpy as np


GPU_WORKSPACE_CAP_BYTES = 128 * 1024 * 1024


def _text(value):
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def require_cupy():
    import cupy as cp
    if not isinstance(cp, types.ModuleType):
        raise RuntimeError("CuPy must be a real imported module")
    origin = Path(cp.__file__).resolve(strict=True)
    distributions = importlib.metadata.packages_distributions().get("cupy", [])
    allowed = [name for name in distributions if name == "cupy" or
               name.startswith("cupy-cuda") or name.startswith("cupy-rocm")]
    if not allowed:
        raise RuntimeError("CuPy distribution ownership missing")
    versions = {name: importlib.metadata.version(name) for name in allowed}
    if str(cp.__version__) not in versions.values():
        raise RuntimeError("CuPy module/distribution version mismatch")
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA device required")
    return cp


def compiled_kernel_probe(cp) -> None:
    probe = cp.arange(257, dtype=cp.int64)
    if int((probe * probe).sum().item()) != sum(i * i for i in range(257)):
        raise RuntimeError("CuPy compiled-kernel identity probe")


def runtime_identity(cp) -> dict:
    origin = Path(cp.__file__).resolve(strict=True)
    active = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(active)
    distributions = importlib.metadata.packages_distributions().get("cupy", [])
    versions = {name: importlib.metadata.version(name) for name in distributions
                if name == "cupy" or name.startswith("cupy-")}
    return {
        "module": cp.__name__,
        "module_origin": str(origin),
        "module_file_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "module_version": str(cp.__version__),
        "owning_distributions": versions,
        "cuda_visible_device_count": int(cp.cuda.runtime.getDeviceCount()),
        "active_device_id": active,
        "active_device_name": _text(properties.get("name", "")),
        "active_device_pci_bus_id": _text(cp.cuda.runtime.deviceGetPCIBusId(active)),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "compiled_kernel_identity_probe": True,
    }


class PoolReceipt:
    """Exact dedicated-pool accounting inside the isolated worker."""

    def __init__(self, cp):
        self.cp = cp
        self.pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(self.pool.malloc)
        self.samples = []
        self.sample("fresh_pool")

    def sample(self, label: str) -> None:
        self.cp.cuda.get_current_stream().synchronize()
        row = {"label": label,
               "used_bytes": int(self.pool.used_bytes()),
               "total_reserved_bytes": int(self.pool.total_bytes())}
        self.samples.append(row)
        if row["total_reserved_bytes"] > GPU_WORKSPACE_CAP_BYTES:
            raise MemoryError("GPU workspace cap")

    @property
    def peak_reserved_bytes(self) -> int:
        return max(row["total_reserved_bytes"] for row in self.samples)


def _packet_gpu_metrics(cp, table, packet: bytes, lambda_bit: float) -> dict:
    from codec import decode_packet
    decoded = decode_packet(packet, allow_small=True)
    index = cp.asarray(decoded["indices"], dtype=cp.int64)
    rows = cp.arange(index.size, dtype=cp.int64)
    sse = float(table[rows, index].sum(dtype=cp.float64).item())
    bits = int(decoded["physical_bits"])
    return {"sse": sse, "physical_bits": bits,
            "objective": sse + lambda_bit * bits}


def _rank_one_plane_gpu(cp, c0, c1, nr: int, nc: int, receipt: PoolReceipt):
    preferred = (c1 < c0).reshape(nr, nc).astype(cp.uint8)
    v = preferred[0].copy()
    u = cp.zeros(nr, dtype=cp.uint8)
    for iteration in range(4):
        out1 = cp.broadcast_to(v[None, :], (nr, nc))
        score0 = c0.reshape(nr, nc).sum(axis=1, dtype=cp.float64)
        score1 = cp.where(out1, c1.reshape(nr, nc),
                          c0.reshape(nr, nc)).sum(axis=1, dtype=cp.float64)
        u = (score1 < score0).astype(cp.uint8)
        out1_col = cp.broadcast_to(u[:, None], (nr, nc))
        score0_col = c0.reshape(nr, nc).sum(axis=0, dtype=cp.float64)
        score1_col = cp.where(out1_col, c1.reshape(nr, nc),
                              c0.reshape(nr, nc)).sum(axis=0, dtype=cp.float64)
        v = (score1_col < score0_col).astype(cp.uint8)
        receipt.sample(f"rank1_als_{iteration}")
    plane = (u[:, None] * v[None, :]).astype(cp.uint8).reshape(-1)
    receipt.sample("rank1_plane")
    return plane


def search_bmp_rank01_cupy(distortion, geometry, lambda_bit: float, rate_cap):
    """Run a literal device-backed bounded rank-0/rank-1 BMP search."""
    from codec import (FAMILY_BMP, canonical_gf2_factor, encode_packet,
                       planes_to_indices, validate_distortion_table)
    from search import CompleteRateCap, add_joint_exceptions

    cp = require_cupy()
    if not isinstance(rate_cap, CompleteRateCap):
        raise RuntimeError("explicit complete-rate cap required")
    rate_cap.validate()
    geometry.validate()
    host = validate_distortion_table(np.asarray(distortion), geometry.count)
    receipt = PoolReceipt(cp)
    compiled_kernel_probe(cp)
    receipt.sample("compiled_kernel_identity_probe")
    table = cp.asarray(host, dtype=cp.float64)
    nearest = cp.argmin(table, axis=1).astype(cp.uint8)
    receipt.sample("distortion_and_nearest")
    candidates = []
    for requested_rank in (0, 1):
        current = nearest.copy()
        factors = []
        planes = []
        for level in range(6):
            clear = current & cp.uint8(63 ^ (1 << level))
            one = clear | cp.uint8(1 << level)
            rows = cp.arange(geometry.count, dtype=cp.int64)
            c0 = table[rows, clear.astype(cp.int64)]
            c1 = table[rows, one.astype(cp.int64)]
            if requested_rank == 0:
                plane_gpu = cp.zeros(geometry.count, dtype=cp.uint8)
            else:
                plane_gpu = _rank_one_plane_gpu(
                    cp, c0, c1, geometry.row_count, geometry.col_count, receipt)
            plane = cp.asnumpy(plane_gpu)
            U, V = canonical_gf2_factor(
                plane, geometry.row_count, geometry.col_count)
            factors.append((U, V))
            planes.append(plane)
            current = clear | (plane_gpu << level)
            receipt.sample(f"rank{requested_rank}_level{level}")
        model = {"ranks": [U.shape[1] for U, _ in factors], "factors": factors}
        base = planes_to_indices(np.stack(planes, axis=0))
        exceptions = add_joint_exceptions(host, base, lambda_bit)
        packet = encode_packet(FAMILY_BMP, 0, geometry, model, exceptions)
        metrics = _packet_gpu_metrics(cp, table, packet, lambda_bit)
        candidates.append({"family": "GF2_MATRIX_FACTOR",
                           "requested_rank": requested_rank,
                           "packet": packet, **metrics})
        receipt.sample(f"rank{requested_rank}_packet_score")
    admitted = [row for row in candidates
                if rate_cap.admit_packet(row["physical_bits"])]
    if not admitted:
        raise RuntimeError("all GPU candidates exceed complete-rate cap")
    winner = min(admitted, key=lambda row: (row["objective"], row["packet"]))
    return {
        "winner": winner,
        "candidates": candidates,
        "backend_scope": "actual_cupy_rank0_rank1_bmp_bounded_search",
        "runtime_identity": runtime_identity(cp),
        "workspace": {
            "allocator": "fresh dedicated cupy MemoryPool",
            "samples": receipt.samples,
            "peak_total_reserved_bytes": receipt.peak_reserved_bytes,
            "cap_bytes": GPU_WORKSPACE_CAP_BYTES,
            "unobserved_allocator_peak_claimed": False,
        },
        "held_families": ["ROBDD GPU search", "canonical QTT GPU search"],
    }
