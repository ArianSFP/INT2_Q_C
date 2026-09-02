#!/usr/bin/env python3
"""Optional bounded CuPy primitives; imported only by the explicit GPU smoke."""

from __future__ import annotations


GPU_WORKSPACE_CAP_BYTES = 128 * 1024 * 1024


def require_cupy():
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA device required")
    return cp


def distortion_table_gpu(source, levels):
    cp = require_cupy()
    x = cp.asarray(source, dtype=cp.float64)
    q = cp.asarray(levels, dtype=cp.float64)
    if x.ndim != 1 or x.size != 4096 or q.shape != (64,):
        raise ValueError("source[4096], levels[64]")
    workspace = int(x.size * q.size * 8)
    if workspace > GPU_WORKSPACE_CAP_BYTES:
        raise MemoryError("GPU workspace cap")
    result = (x[:, None] - q[None, :]) ** 2
    cp.cuda.get_current_stream().synchronize()
    return result


def assemble_six_planes_gpu(planes):
    cp = require_cupy()
    value = cp.asarray(planes, dtype=cp.uint8)
    if value.shape != (6, 4096):
        raise ValueError("six completed 4096 planes")
    if not bool(cp.all((value == 0) | (value == 1)).item()):
        raise ValueError("binary planes")
    output = cp.zeros(4096, dtype=cp.uint8)
    for level in range(6):
        output |= value[level] << level
    cp.cuda.get_current_stream().synchronize()
    return output


def bmp_planes_gpu(factors):
    cp = require_cupy()
    if len(factors) != 6:
        raise ValueError("six BMP planes")
    rows = []
    peak = 0
    for U, V in factors:
        left = cp.asarray(U, dtype=cp.uint8)
        right = cp.asarray(V, dtype=cp.uint8)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("BMP factors")
        peak = max(peak, int(left.nbytes + right.nbytes + left.shape[0] *
                             right.shape[0] * 2))
        if peak > GPU_WORKSPACE_CAP_BYTES:
            raise MemoryError("GPU workspace cap")
        rows.append(((left.astype(cp.uint16) @ right.astype(cp.uint16).T) & 1)
                    .astype(cp.uint8).reshape(-1))
    result = cp.stack(rows, axis=0)
    cp.cuda.get_current_stream().synchronize()
    return result
