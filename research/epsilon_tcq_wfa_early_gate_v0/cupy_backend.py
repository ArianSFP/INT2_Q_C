#!/usr/bin/env python3
"""Lazy CuPy kernels and accounting for epsilon-TCQ beam expansion."""

from __future__ import annotations

import math
from typing import Any


class BackendError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendError(message)


class CuPyBeamBackend:
    def __init__(self) -> None:
        import cupy as cp
        self.cp = cp
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.kernel_calls = 0
        self.synchronized = False

    def branch_distortion(
        self, targets: Any, nominal: Any, centroid: Any,
    ) -> Any:
        cp = self.cp
        target_array = cp.asarray(targets, dtype=cp.float64)
        nominal_array = cp.asarray(nominal, dtype=cp.float64)
        centroid_array = cp.asarray(centroid, dtype=cp.float64)
        require(target_array.ndim == 1 and nominal_array.ndim == 2 and
                centroid_array.shape == nominal_array.shape and
                target_array.shape[0] == nominal_array.shape[0],
                "beam distortion geometry")
        self.h2d_bytes += int(target_array.nbytes + nominal_array.nbytes +
                              centroid_array.nbytes)
        output = (target_array[:, None] - nominal_array - centroid_array) ** 2
        self.kernel_calls += 1
        return output

    def score_flat_squared_error(self, targets: Any,
                                 reconstructions: Any) -> tuple[float, ...]:
        """Batch the production beam's source-domain FP64 branch metric."""
        cp = self.cp
        target_array = cp.asarray(targets, dtype=cp.float64).reshape(-1)
        reconstruction_array = cp.asarray(
            reconstructions, dtype=cp.float64).reshape(-1)
        require(target_array.size > 0 and
                target_array.shape == reconstruction_array.shape,
                "flat beam distortion geometry")
        self.h2d_bytes += int(target_array.nbytes + reconstruction_array.nbytes)
        output = (target_array - reconstruction_array) ** 2
        self.kernel_calls += 1
        host = self.to_host(output)
        return tuple(float(value) for value in host)

    def bounded_hankel_spectrum(self, bits: Any, prefix: int = 3,
                                suffix: int = 3) -> tuple[float, ...]:
        """Bounded spectral diagnostic; it cannot add a model topology."""
        cp = self.cp
        require(type(prefix) is int and type(suffix) is int and
                1 <= prefix <= 3 and 1 <= suffix <= 3,
                "bounded Hankel word lengths")
        host_bits = tuple(int(value) for value in bits)
        require(len(host_bits) >= prefix + suffix and
                all(value in (0, 1) for value in host_bits),
                "bounded Hankel bits")
        rows = 1 << prefix
        columns = 1 << suffix
        counts = cp.zeros((rows, columns), dtype=cp.float64)
        # The bounded matrix is at most 8x8; construction is deterministic,
        # while the requested spectral factorization is the CuPy heavy path.
        host_counts = [[0.0] * columns for _ in range(rows)]
        for start in range(len(host_bits) - prefix - suffix + 1):
            left = 0
            right = 0
            for bit in host_bits[start:start + prefix]:
                left = (left << 1) | bit
            for bit in host_bits[start + prefix:start + prefix + suffix]:
                right = (right << 1) | bit
            host_counts[left][right] += 1.0
        counts[...] = cp.asarray(host_counts, dtype=cp.float64)
        self.h2d_bytes += rows * columns * 8
        singular = cp.linalg.svd(counts, compute_uv=False)
        self.kernel_calls += 1
        host = self.to_host(singular)
        return tuple(float(value) for value in host)

    def topk_paths(self, objective: Any, width: int) -> Any:
        cp = self.cp
        values = cp.asarray(objective, dtype=cp.float64).reshape(-1)
        require(type(width) is int and width > 0, "beam top-k width")
        count = min(width, int(values.size))
        if count == values.size:
            indices = cp.argsort(values)
        else:
            selected = cp.argpartition(values, count - 1)[:count]
            indices = selected[cp.argsort(values[selected])]
        self.kernel_calls += 2
        return indices

    def to_host(self, value: Any) -> Any:
        array = self.cp.asnumpy(value)
        self.d2h_bytes += int(array.nbytes)
        return array

    def receipt(self) -> dict[str, Any]:
        cp = self.cp
        cp.cuda.Stream.null.synchronize()
        self.synchronized = True
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return {
            "schema": "epsilon-tcq-cupy-beam-backend-receipt-v0",
            "device_name": str(name),
            "compute_capability": str(cp.cuda.Device(0).compute_capability),
            "cupy_version": cp.__version__,
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "kernel_calls": self.kernel_calls,
            "synchronized": self.synchronized,
            "storage_bytes_counted_as_hbm": False,
            "hbm_bytes_counted_as_storage": False,
        }


def source_free_smoke() -> dict[str, Any]:
    backend = CuPyBeamBackend()
    targets = [0.0, 1.0, -1.0]
    nominal = [[-0.25, 0.0, 0.25], [0.5, 1.0, 1.5],
               [-1.5, -1.0, -0.5]]
    centroids = [[0.0, 0.0, 0.0]] * 3
    distortion = backend.branch_distortion(targets, nominal, centroids)
    indices = backend.topk_paths(distortion, 4)
    host = backend.to_host(indices)
    require(host.size == 4 and all(int(value) >= 0 for value in host),
            "source-free top-k")
    flat = backend.score_flat_squared_error(
        (0.0, 1.0, -1.0), (-0.25, 1.0, -0.5))
    require(flat == (0.0625, 0.0, 0.25), "source-free flat metric")
    spectrum = backend.bounded_hankel_spectrum(
        tuple((index ^ (index >> 1)) & 1 for index in range(128)))
    require(len(spectrum) == 8 and all(math.isfinite(value) and value >= 0.0
                                      for value in spectrum),
            "source-free bounded Hankel spectrum")
    receipt = backend.receipt()
    receipt.update({
        "status": "PASS_SOURCE_FREE_CUPY_BEAM_SMOKE",
        "payload_accessed": False,
        "network_accessed": False,
    })
    return receipt
