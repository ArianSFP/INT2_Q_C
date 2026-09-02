#!/usr/bin/env python3
"""Lazy CuPy entrypoints for the MOSAIC secondary-oracle heavy paths."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence


AUTHORIZATION = "RUN_SOURCE_FREE_MOSAIC_SECONDARY_CUPY_SMOKE_V0"


class BackendError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendError(message)


def load_cupy() -> Any:
    import cupy as cp

    require(cp.cuda.runtime.getDeviceCount() > 0, "CUDA device")
    return cp


def source_free_smoke(
    residual_core: Any,
    *,
    authorization: str,
    periods: Sequence[int],
) -> dict[str, Any]:
    require(authorization == AUTHORIZATION, "explicit CuPy smoke authorization")
    cp = load_cupy()
    length = 256
    blocks = 8
    basis = residual_core.build_ramanujan_basis(
        cp,
        length=length,
        periods=periods,
        maximum_columns=64,
    )
    coordinate = cp.arange(length, dtype=cp.float64)
    rows = []
    for ordinal in range(blocks):
        phase = 0.13 * ordinal
        rows.append(
            cp.sin((2.0 * math.pi / 7.0) * coordinate + phase)
            + 0.07 * cp.cos((4.0 * math.pi / 11.0) * coordinate - phase)
        )
    residual = cp.stack(rows)
    energy = float(cp.sum(residual * residual, dtype=cp.float64).item()) / 0.04
    ramanujan = residual_core.ramanujan_panel_metrics(
        cp,
        residual,
        basis,
        source_energy=energy,
    )
    ar = residual_core.ar_hankel_panel_metrics(
        cp,
        residual,
        source_energy=energy,
        orders=(1, 2, 4),
    )
    permuted = residual_core.phase_destroyed_blocks(cp, residual, 10619863)
    gaussian = residual_core.moment_matched_gaussian_blocks(cp, residual, 10619863)
    source_means = cp.mean(residual, axis=1, dtype=cp.float64)
    source_centered_energy = cp.sum(
        (residual - source_means[:, None]) ** 2, axis=1, dtype=cp.float64
    )
    gaussian_means = cp.mean(gaussian, axis=1, dtype=cp.float64)
    gaussian_centered_energy = cp.sum(
        (gaussian - gaussian_means[:, None]) ** 2, axis=1, dtype=cp.float64
    )
    cp.cuda.Stream.null.synchronize()
    digest = hashlib.sha256(
        cp.asnumpy(residual).astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()
    return {
        "schema": "mosaic-secondary-source-free-cupy-smoke-v0",
        "status": "PASS_SOURCE_FREE_CUPY_HEAVY_PATH",
        "device_id": int(cp.cuda.Device().id),
        "residual_f64_sha256": digest,
        "ramanujan_relative_mse": ramanujan["ideal_public_basis_waterfill_remaining_sse"] / energy,
        "ar_relative_mse": ar["winner"]["relative_mse"],
        "permutation_preserves_values": bool(
            cp.all(cp.sort(permuted, axis=1) == cp.sort(residual, axis=1)).item()
        ),
        "gaussian_mean_max_abs_error": float(cp.max(cp.abs(gaussian_means - source_means)).item()),
        "gaussian_centered_energy_max_relative_error": float(
            cp.max(
                cp.abs(gaussian_centered_energy - source_centered_energy)
                / cp.maximum(source_centered_energy, 1.0)
            ).item()
        ),
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "network_accessed": False,
    }
