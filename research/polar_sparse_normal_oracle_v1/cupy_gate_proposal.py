"""Inert CuPy-compatible primitives for the PSNO-v1 sparse-normal gate.

The caller must supply an already authenticated CuPy module and the 18 frozen
normal matrices.  This file has no tensor paths, imports no CuPy on import,
creates no device context on import, writes nothing, and grants no authority.
"""

from __future__ import annotations

import math
from typing import Any


AUTHORITY_GRANTED = False
N = 768
MATRICES = 18
UNIQUE = N * (N + 1) // 2
CONTROL_SEEDS = (
    27_090_101,
    27_090_119,
    27_090_143,
    27_090_171,
    27_090_207,
    27_090_231,
    27_090_263,
    27_090_299,
)


def source_only_status() -> dict[str, Any]:
    return {
        "status": "SOURCE_ONLY_NO_TENSOR_OR_GPU_AUTHORITY",
        "cupy_imported_by_module": False,
        "tensor_paths_present": False,
        "gpu_execution_authorized": False,
        "authorization": False,
    }


def _shape(normal_matrices: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in normal_matrices.shape)


def validate_normal_batch(normal_matrices: Any, cp: Any, *, symmetry_tolerance: float = 2e-10) -> None:
    if _shape(normal_matrices) != (MATRICES, N, N):
        raise ValueError("normal batch must have shape (18,768,768)")
    if normal_matrices.dtype != cp.float64:
        raise ValueError("binary64 normal batch required")
    if not bool(cp.all(cp.isfinite(normal_matrices)).item()):
        raise ValueError("non-finite normal coefficient")
    scale = float(cp.max(cp.abs(normal_matrices)).item())
    mismatch = float(cp.max(cp.abs(normal_matrices - normal_matrices.transpose(0, 2, 1))).item())
    if mismatch > symmetry_tolerance * max(1.0, scale):
        raise ValueError("normal symmetry closure failed")


def orthonormal_symmetric_coefficients(normal_matrices: Any, cp: Any) -> tuple[Any, Any, Any]:
    """Return a_ii=N_ii, a_ij=sqrt(2)N_ij and their upper indices."""

    validate_normal_batch(normal_matrices, cp)
    row, col = cp.triu_indices(N)
    scale = cp.where(row == col, 1.0, math.sqrt(2.0)).astype(cp.float64)
    coefficients = normal_matrices[:, row, col] * scale[None, :]
    if _shape(coefficients) != (MATRICES, UNIQUE):
        raise AssertionError("symmetric coordinate count")
    source_energy = cp.sum(cp.square(normal_matrices), axis=(1, 2), dtype=cp.float64)
    coordinate_energy = cp.sum(cp.square(coefficients), axis=1, dtype=cp.float64)
    relative = cp.max(cp.abs(source_energy - coordinate_energy) / cp.maximum(source_energy, 1e-300))
    if float(relative.item()) > 3e-13:
        raise AssertionError("orthonormal symmetric energy closure")
    return coefficients, row, col


def coordinate_energy_prefix(coefficients: Any, cp: Any) -> Any:
    """Exact best-k capture prefix for every matrix, including k=0."""

    if _shape(coefficients) != (MATRICES, UNIQUE):
        raise ValueError("coefficient batch shape")
    descending = cp.sort(cp.square(coefficients), axis=1)[:, ::-1]
    zero = cp.zeros((MATRICES, 1), dtype=cp.float64)
    return cp.concatenate((zero, cp.cumsum(descending, axis=1, dtype=cp.float64)), axis=1)


def triangular_tile_groups(
    coefficients: Any, row: Any, col: Any, block_size: int, cp: Any
) -> dict[str, Any]:
    """Fixed upper-triangular BxB groups with exact unique-coordinate sizes."""

    if block_size not in (8, 16, 32, 64):
        raise ValueError("unfrozen tile size")
    tiles = (N + block_size - 1) // block_size
    tile_row = row // block_size
    tile_col = col // block_size
    group_id = tile_row * tiles - tile_row * (tile_row - 1) // 2 + (tile_col - tile_row)
    group_count = tiles * (tiles + 1) // 2
    sizes = cp.bincount(group_id, minlength=group_count).astype(cp.int64)
    energies = cp.empty((MATRICES, group_count), dtype=cp.float64)
    squared = cp.square(coefficients)
    for matrix in range(MATRICES):
        energies[matrix] = cp.bincount(
            group_id, weights=squared[matrix], minlength=group_count
        )
    if int(cp.sum(sizes).item()) != UNIQUE:
        raise AssertionError("tile group coordinate closure")
    return {
        "family": "upper_triangular_tiles",
        "block_size": block_size,
        "group_count": group_count,
        "group_sizes": sizes,
        "group_energies": energies,
    }


def offset_segment_groups(
    coefficients: Any, row: Any, col: Any, segment_length: int, cp: Any
) -> dict[str, Any]:
    """Contiguous segments within each upper diagonal offset."""

    if segment_length not in (8, 16, 32, 64):
        raise ValueError("unfrozen segment length")
    prefix: list[int] = []
    total = 0
    for offset in range(N):
        prefix.append(total)
        total += (N - offset + segment_length - 1) // segment_length
    prefix_device = cp.asarray(prefix, dtype=cp.int64)
    offset = col - row
    group_id = prefix_device[offset] + row // segment_length
    sizes = cp.bincount(group_id, minlength=total).astype(cp.int64)
    energies = cp.empty((MATRICES, total), dtype=cp.float64)
    squared = cp.square(coefficients)
    for matrix in range(MATRICES):
        energies[matrix] = cp.bincount(group_id, weights=squared[matrix], minlength=total)
    if int(cp.sum(sizes).item()) != UNIQUE:
        raise AssertionError("offset group coordinate closure")
    return {
        "family": "diagonal_offset_segments",
        "segment_length": segment_length,
        "group_count": total,
        "group_sizes": sizes,
        "group_energies": energies,
    }


def gaussian_rank_heavy_tail_control(coefficients: Any, seed: int, cp: Any) -> Any:
    """Gaussian-rank spatial null preserving every absolute coefficient."""

    if seed not in CONTROL_SEEDS:
        raise ValueError("unfrozen control seed")
    if _shape(coefficients) != (MATRICES, UNIQUE):
        raise ValueError("coefficient batch shape")
    output = cp.empty_like(coefficients)
    for matrix in range(MATRICES):
        rng = cp.random.RandomState(seed + 1009 * matrix)
        gaussian = rng.standard_normal(UNIQUE, dtype=cp.float64)
        order = cp.argsort(cp.abs(gaussian))
        magnitudes = cp.sort(cp.abs(coefficients[matrix]))
        signs = cp.where(gaussian[order] < 0.0, -1.0, 1.0)
        output[matrix, order] = magnitudes * signs
    original_sorted = cp.sort(cp.square(coefficients), axis=1)
    control_sorted = cp.sort(cp.square(output), axis=1)
    if not bool(cp.array_equal(original_sorted, control_sorted).item()):
        raise AssertionError("heavy-tail control marginal closure")
    return output


def diagonal_stratified_control(
    coefficients: Any, row: Any, col: Any, seed: int, cp: Any
) -> Any:
    """Preserve every offset marginal while destroying within-offset locality."""

    if seed not in CONTROL_SEEDS:
        raise ValueError("unfrozen control seed")
    output = cp.empty_like(coefficients)
    offsets = col - row
    for matrix in range(MATRICES):
        rng = cp.random.RandomState(seed + 2027 * matrix)
        for offset in range(N):
            positions = cp.flatnonzero(offsets == offset)
            permutation = rng.permutation(len(positions))
            output[matrix, positions] = coefficients[matrix, positions[permutation]]
    original_energy = cp.sum(cp.square(coefficients), axis=1, dtype=cp.float64)
    control_energy = cp.sum(cp.square(output), axis=1, dtype=cp.float64)
    if not bool(cp.array_equal(original_energy, control_energy).item()):
        # CuPy scatter order can change the final reduction roundoff.  Require
        # an exact multiset and a tight energy check instead of accepting drift.
        for matrix in range(MATRICES):
            if not bool(
                cp.array_equal(
                    cp.sort(cp.square(coefficients[matrix])),
                    cp.sort(cp.square(output[matrix])),
                ).item()
            ):
                raise AssertionError("diagonal control marginal closure")
    return output


def build_gate_measurements(normal_matrices: Any, cp: Any) -> dict[str, Any]:
    """Measurement-only primitive; deliberately does not score or authorize."""

    coefficients, row, col = orthonormal_symmetric_coefficients(normal_matrices, cp)
    return {
        "coordinate_prefix": coordinate_energy_prefix(coefficients, cp),
        "tiles": [
            triangular_tile_groups(coefficients, row, col, size, cp)
            for size in (8, 16, 32, 64)
        ],
        "offset_segments": [
            offset_segment_groups(coefficients, row, col, size, cp)
            for size in (8, 16, 32, 64)
        ],
        "coefficients": coefficients,
        "row": row,
        "col": col,
        "authorization": False,
    }


if __name__ == "__main__":
    raise SystemExit("PSNO_V1_SOURCE_ONLY_NO_GPU_AUTHORITY")
