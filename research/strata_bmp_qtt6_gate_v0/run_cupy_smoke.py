#!/usr/bin/env python3
"""Explicit source-free CuPy smoke; it has no payload locator or network path."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

from codec import bmp_plane, indices_to_planes, planes_to_indices, sha256
from cupy_backend import (GPU_WORKSPACE_CAP_BYTES, assemble_six_planes_gpu,
                          bmp_planes_gpu, distortion_table_gpu, require_cupy)


def main() -> None:
    cp = require_cupy()
    levels = np.float64(0.0625) * (-31.0 + np.arange(64, dtype=np.float64))
    source = levels[np.arange(4096, dtype=np.uint16) % 64]
    source += 0.001 * np.sin(np.arange(4096, dtype=np.float64))
    gpu_d = distortion_table_gpu(source, levels)
    cpu_d = (source[:, None] - levels[None, :]) ** 2
    indices = np.argmin(cpu_d, axis=1).astype(np.uint8)
    planes = indices_to_planes(indices)
    gpu_indices = assemble_six_planes_gpu(planes)
    factors = []
    for level in range(6):
        U = np.zeros((16, 1), dtype=np.uint8)
        V = np.zeros((256, 1), dtype=np.uint8)
        U[:, 0] = (np.arange(16) >> (level % 4)) & 1
        V[:, 0] = (np.arange(256) >> ((level + 1) % 8)) & 1
        factors.append((U, V))
    gpu_bmp = bmp_planes_gpu(factors)
    cpu_bmp = np.stack([bmp_plane(U, V) for U, V in factors])
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    result = {
        "schema": "strata-bmp-obdd-qtt6-cupy-smoke-v0",
        "status": "PASS_SOURCE_FREE_CUPY_SMOKE",
        "cupy_version": cp.__version__,
        "device_name": str(device_name),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "distortion_max_abs_error": float(cp.max(cp.abs(
            gpu_d - cp.asarray(cpu_d))).item()),
        "six_plane_index_equal": bool(cp.array_equal(
            gpu_indices, cp.asarray(planes_to_indices(planes))).item()),
        "bmp_equal": bool(cp.array_equal(gpu_bmp, cp.asarray(cpu_bmp)).item()),
        "distortion_sha256": sha256(cp.asnumpy(gpu_d).tobytes()),
        "gpu_workspace_cap_bytes": GPU_WORKSPACE_CAP_BYTES,
        "model_or_qwen_payload_accessed": False,
        "network_accessed": False,
    }
    cp.cuda.get_current_stream().synchronize()
    if not (result["distortion_max_abs_error"] <= 1e-15 and
            result["six_plane_index_equal"] and result["bmp_equal"]):
        raise RuntimeError("CuPy parity")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
