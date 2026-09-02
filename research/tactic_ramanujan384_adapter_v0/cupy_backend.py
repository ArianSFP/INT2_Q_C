#!/usr/bin/env python3
"""Lazy source-free CuPy smoke for the finite Ramanujan-384 codec."""

from __future__ import annotations

import hashlib
from typing import Any


AUTHORIZATION = "RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_CUPY_SMOKE_V0"


class BackendError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendError(message)


def load_cupy() -> Any:
    import cupy as cp
    require(cp.cuda.runtime.getDeviceCount() > 0, "CUDA device")
    return cp


def source_free_smoke(codec: Any, *, authorization: str) -> dict[str, Any]:
    require(authorization == AUTHORIZATION, "explicit source-free CuPy authorization")
    cp = load_cupy()
    coordinate = cp.arange(codec.BLOCK_VALUES, dtype=cp.int64)
    period7 = cp.asarray([codec.ramanujan_sum(7, index) for index in range(7)], dtype=cp.float64)
    period11 = cp.asarray([codec.ramanujan_sum(11, index) for index in range(11)], dtype=cp.float64)
    residual = cp.stack([
        0.013 * period7[(coordinate - block) % 7]
        + 0.004 * period11[(coordinate - 2 * block) % 11]
        for block in range(2)
    ])
    coarse = cp.zeros_like(residual)
    source_energy = float(cp.sum(residual * residual, dtype=cp.float64).item()) / codec.COARSE_RELATIVE_MSE
    result = codec.run_finite_panel(cp, residual, coarse, role="gate", source_energy=source_energy)
    require(result["controls_rerun"] and len(result.get("gaussian_controls", [])) == 8,
            "complete source-free control contract")
    cp.cuda.Stream.null.synchronize()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "schema": "tactic-ramanujan384-source-free-cupy-smoke-v0",
        "status": "PASS_SOURCE_FREE_CUPY_FINITE_PACKET_AND_ALL_CONTROLS",
        "device_id": int(cp.cuda.Device().id),
        "device_name": str(name),
        "source_f64_sha256": hashlib.sha256(
            cp.asnumpy(residual).astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "finite_status": result["status"],
        "relative_mse": result["relative_mse"],
        "controls_rerun": result["controls_rerun"],
        "gaussian_controls": len(result.get("gaussian_controls", [])),
        "literal_bits_per_block": 384,
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "network_accessed": False,
    }
