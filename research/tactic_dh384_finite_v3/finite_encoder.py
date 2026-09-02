#!/usr/bin/env python3
"""CuPy-heavy encoder for the frozen finite TACTIC-DH384 v3 codebook.

This module is loaded only from authenticated source bytes after launch review,
v6 result, runtime, and input bindings pass.  It contains no model discovery,
training, fitted table, or selector search.
"""

from __future__ import annotations

import math
from typing import Any


class EncoderError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EncoderError(message)


def _conditional_analysis(cp: Any, symbols: Any, residual: Any,
                          role_ordinal: int, table: bytes,
                          block: int, stages: int) -> tuple[Any, list[Any]]:
    symbol = cp.asarray(symbols, dtype=cp.int64).reshape(-1, block)
    error = cp.asarray(residual, dtype=cp.float64).reshape(-1, block)
    table_gpu = cp.asarray(bytearray(table), dtype=cp.uint8).reshape(stages, 256)
    mean_abs = cp.sum(cp.abs(symbol), axis=1, dtype=cp.int64) // cp.int64(block)
    shadow = symbol
    schedules: list[Any] = []
    for stage in range(stages):
        stride = 1 << stage
        paired = shadow.reshape(shadow.shape[0], -1, 2, stride)
        left, right = paired[:, :, 0, :], paired[:, :, 1, :]
        absolute_left, absolute_right = cp.abs(left), cp.abs(right)
        threshold = mean_abs[:, None, None]
        feature = (
            cp.int64(role_ordinal << 6)
            | ((left < 0).astype(cp.int64) << cp.int64(5))
            | ((right < 0).astype(cp.int64) << cp.int64(4))
            | ((absolute_left > absolute_right).astype(cp.int64) << cp.int64(3))
            | (((absolute_left + absolute_right) > 2 * threshold).astype(cp.int64)
               << cp.int64(2))
            | ((absolute_left > threshold).astype(cp.int64) << cp.int64(1))
            | (absolute_right > threshold).astype(cp.int64)
        )
        operation = table_gpu[stage, feature]
        swap = (operation & cp.uint8(1)) != 0
        a = cp.where(swap, right, left)
        b = cp.where(swap, left, right)
        a = cp.where((operation & cp.uint8(2)) != 0, -a, a)
        b = cp.where((operation & cp.uint8(4)) != 0, -b, b)
        shadow = cp.stack((a + b, a - b), axis=2).reshape(shadow.shape)
        schedules.append(operation)

    transformed = error
    for stage in reversed(range(stages)):
        stride = 1 << stage
        paired = transformed.reshape(transformed.shape[0], -1, 2, stride)
        left, right = paired[:, :, 0, :], paired[:, :, 1, :]
        x0, x1 = left + right, left - right
        operation = schedules[stage]
        x0 = cp.where((operation & cp.uint8(2)) != 0, -x0, x0)
        x1 = cp.where((operation & cp.uint8(4)) != 0, -x1, x1)
        swap = (operation & cp.uint8(1)) != 0
        u = cp.where(swap, x1, x0)
        v = cp.where(swap, x0, x1)
        transformed = cp.stack((u, v), axis=2).reshape(transformed.shape)
    return transformed / cp.float64(64.0), schedules


def _synthesis(cp: Any, coefficients: Any, schedules: list[Any],
               stages: int) -> Any:
    values = coefficients
    for stage in range(stages):
        stride = 1 << stage
        paired = values.reshape(values.shape[0], -1, 2, stride)
        left, right = paired[:, :, 0, :], paired[:, :, 1, :]
        operation = schedules[stage]
        swap = (operation & cp.uint8(1)) != 0
        a = cp.where(swap, right, left)
        b = cp.where(swap, left, right)
        a = cp.where((operation & cp.uint8(2)) != 0, -a, a)
        b = cp.where((operation & cp.uint8(4)) != 0, -b, b)
        values = cp.stack((a + b, a - b), axis=2).reshape(values.shape)
    return values / cp.float64(64.0)


def continuous_tile(cp: Any, symbols: Any, residual: Any,
                    role_ordinal: int, spec: Any) -> dict[str, float]:
    coefficients, _schedules = _conditional_analysis(
        cp, symbols, residual, role_ordinal, spec.universal_selector_table(),
        spec.BLOCK_VALUES, spec.STAGES)
    error = cp.asarray(residual, dtype=cp.float64).reshape(
        -1, spec.BLOCK_VALUES)
    energy = float(cp.sum(error * error, dtype=cp.float64).item())
    transformed_energy = float(
        cp.sum(coefficients * coefficients, dtype=cp.float64).item())
    require(math.isclose(energy, transformed_energy,
                         rel_tol=2e-11, abs_tol=2e-9),
            "CuPy conditional transform norm identity")
    parent = float(cp.sum(
        coefficients[:, :spec.AUDITED_PARENT_RANK] ** 2,
        dtype=cp.float64).item())
    active = float(cp.sum(
        coefficients[:, :spec.ACTIVE_RANK] ** 2,
        dtype=cp.float64).item())
    require(0.0 <= active <= parent * (1.0 + 2e-12) and
            parent <= energy * (1.0 + 2e-12),
            "continuous containment bounds")
    return {
        "error_energy_fp64": energy,
        "transformed_energy_fp64": transformed_energy,
        "audited_parent_rank384_projected_energy_fp64": min(parent, energy),
        "active_rank376_projected_energy_fp64": min(active, energy),
    }


def encode_tile(cp: Any, np: Any, symbols: Any, residual: Any,
                reconstruction: Any, role_ordinal: int,
                spec: Any) -> tuple[bytes, Any, dict[str, Any]]:
    coefficients, schedules = _conditional_analysis(
        cp, symbols, residual, role_ordinal, spec.universal_selector_table(),
        spec.BLOCK_VALUES, spec.STAGES)
    coarse = cp.asarray(reconstruction, dtype=cp.float64).reshape(
        -1, spec.BLOCK_VALUES)
    error = cp.asarray(residual, dtype=cp.float64).reshape(
        -1, spec.BLOCK_VALUES)
    block_count = int(error.shape[0])
    require(block_count == spec.COARSE_TILE_VALUES // spec.BLOCK_VALUES,
            "tile microblock count")
    coarse_max = cp.max(cp.abs(coarse), axis=1)
    absolute_sum = cp.sum(
        cp.abs(coefficients[:, :spec.ACTIVE_RANK]), axis=1,
        dtype=cp.float64)
    codes = cp.arange(256, dtype=cp.float64)[None, :]
    alphas = (
        coarse_max[:, None] * codes * codes /
        cp.float64(spec.SCALE_DENOMINATOR)
    )
    deltas = (
        cp.float64(spec.ACTIVE_RANK) * alphas * alphas
        - cp.float64(2.0) * alphas * absolute_sum[:, None]
    )
    scale_codes = cp.argmin(deltas, axis=1).astype(cp.uint8)
    # Exact same-codebook local comparator.  With fixed length, an orthogonal
    # span and one shared nonnegative amplitude, the locally nearest signs are
    # sign(y), while the continuous nearest amplitude is mean(|y|).  Rounding
    # that amplitude to the frozen 256-entry dyadic grid must equal exhaustive
    # D+lambda*R search because R is exactly 384 bits for every legal record.
    local_target_alpha = absolute_sum / cp.float64(spec.ACTIVE_RANK)
    forced_local_scale_codes = cp.argmin(
        cp.abs(alphas - local_target_alpha[:, None]), axis=1).astype(cp.uint8)
    local_joint_mismatches = int(cp.sum(
        forced_local_scale_codes != scale_codes, dtype=cp.int64).item())
    require(local_joint_mismatches == 0,
            "joint/exact same-codebook local decision equivalence")
    selected_alpha = alphas[
        cp.arange(block_count, dtype=cp.int64),
        scale_codes.astype(cp.int64)]
    signs = coefficients[:, :spec.ACTIVE_RANK] >= cp.float64(0.0)
    signs = cp.where(scale_codes[:, None] == cp.uint8(0), False, signs)
    # CuPy 14.2 implements packbits only for a flattened input.  A row holds
    # exactly 376 = 47*8 bits, so flatten/pack/reshape cannot cross a partial
    # record boundary and is byte-identical to row-wise LSB-first packing.
    packed_signs = cp.packbits(
        signs.reshape(-1), bitorder="little").reshape(block_count, 47)
    require(tuple(packed_signs.shape) == (block_count, 47),
            "packed sign geometry")
    records_gpu = cp.concatenate((scale_codes[:, None], packed_signs), axis=1)
    records = cp.asnumpy(records_gpu).astype(np.uint8, copy=False).tobytes()
    require(len(records) == block_count * spec.FINE_RECORD_BYTES,
            "tile fine bytes")
    # CPU reference parser provides an independent canonicality check before
    # any record can enter a physical container.
    for begin in range(0, len(records), spec.FINE_RECORD_BYTES):
        record = records[begin:begin + spec.FINE_RECORD_BYTES]
        code, decoded_signs = spec.unpack_record(record)
        require(code == records[begin] and spec.pack_record(
            code, decoded_signs) == record, "record CPU canonical reencode")

    finite_coefficients = cp.zeros_like(coefficients, dtype=cp.float64)
    finite_coefficients[:, :spec.ACTIVE_RANK] = cp.where(
        signs, selected_alpha[:, None], -selected_alpha[:, None])
    finite_coefficients[:, :spec.ACTIVE_RANK] = cp.where(
        scale_codes[:, None] == cp.uint8(0), cp.float64(0.0),
        finite_coefficients[:, :spec.ACTIVE_RANK])
    correction = _synthesis(cp, finite_coefficients, schedules, spec.STAGES)
    # The finite coefficient tail is allocated and asserted as literal zero;
    # the synthesis therefore lies in B[:,0:376] subset B[:,0:384].
    require(not bool(cp.any(
        finite_coefficients[:, spec.ACTIVE_RANK:] != cp.float64(0.0)).item()),
        "finite correction coefficient tail")
    coarse_sse = float(cp.sum(error * error, dtype=cp.float64).item())
    corrected_error = error - correction
    fine_sse = float(cp.sum(
        corrected_error * corrected_error, dtype=cp.float64).item())
    require(math.isfinite(coarse_sse) and math.isfinite(fine_sse) and
            fine_sse <= coarse_sse * (1.0 + 5e-13),
            "finite code never worsens tile SSE")
    return records, correction.reshape(-1), {
        "blocks": block_count,
        "fine_bytes": len(records),
        "coarse_sse_fp64": coarse_sse,
        "finite_sse_fp64": fine_sse,
        "finite_error_capture_fp64": coarse_sse - fine_sse,
        "zero_scale_records": int(cp.sum(
            scale_codes == cp.uint8(0), dtype=cp.int64).item()),
        "minimum_scale_code": int(cp.min(scale_codes).item()),
        "maximum_scale_code": int(cp.max(scale_codes).item()),
        "record_reencode_matches": True,
        "correction_in_active_rank376_subset_parent_rank384": True,
        "fine_encoder_search": (
            "exhaustive 256-scale D+lambda*R search with all 376 legal sign "
            "labels source-selected; lambda*R is constant at 384 bits"),
        "coarse_codeword_reoptimized_jointly": False,
        "forced_same_codebook_local_comparator": (
            "sign(y) plus nearest frozen scale to mean(abs(y))"),
        "forced_local_vs_joint_record_mismatches": local_joint_mismatches,
        "forced_local_decisions_equal_exact_joint_search": True,
    }
