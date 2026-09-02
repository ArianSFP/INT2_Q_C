#!/usr/bin/env python3
"""Pure mechanics for the frozen TACTIC-CAGE graph/Krylov oracle.

The module imports only the standard library. Numerical backends are injected
by the authenticated runner after every source/result binding has passed.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping


BLOCK_VALUES = 4096
FINE_BITS_PER_BLOCK = 384
TOPK = 384
TARGET_RELATIVE_MSE = 0.025
TARGET_TOLERANCE = 1e-12
MIN_GRAPH_ADVANTAGE_BPW = 0.03
MIN_SOURCE_SPECIFIC_ADVANTAGE_BPW = 0.03
COMPOSITE_GAP_BPW = 0.11356063

PUBLIC_FAMILY = "public_coordinate_path_dct"
GRAPH_FAMILIES = (
    "coarse_signed_path_dct",
    "coarse_magnitude_path_dct",
    "coarse_context_path_dct",
)
ALL_FAMILIES = (PUBLIC_FAMILY,) + GRAPH_FAMILIES


class OracleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def exact_budget_record() -> dict[str, Any]:
    coarse_numerator = 307
    coarse_denominator = 128
    cap_numerator = 5
    cap_denominator = 2
    remaining_numerator = (
        cap_numerator * coarse_denominator
        - coarse_numerator * cap_denominator
    )
    remaining_denominator = cap_denominator * coarse_denominator
    divisor = math.gcd(remaining_numerator, remaining_denominator)
    remaining_numerator //= divisor
    remaining_denominator //= divisor
    total_remaining_bits = BLOCK_VALUES * remaining_numerator
    require(total_remaining_bits % remaining_denominator == 0,
            "integral total remaining bits")
    return {
        "coarse_bpw_exact": "307/128",
        "cap_bpw_exact": "5/2",
        "remaining_bpw_exact":
            f"{remaining_numerator}/{remaining_denominator}",
        "remaining_bits_per_4096":
            total_remaining_bits // remaining_denominator,
        "oracle_fine_bits_per_4096": FINE_BITS_PER_BLOCK,
        "reserved_noncoefficient_bits_per_4096":
            total_remaining_bits // remaining_denominator
            - FINE_BITS_PER_BLOCK,
    }


def required_capture(coarse_relative_mse: float) -> float:
    require(math.isfinite(coarse_relative_mse) and coarse_relative_mse >= 0.0,
            "finite nonnegative coarse relative MSE")
    if coarse_relative_mse == 0.0:
        return 0.0
    return max(0.0, 1.0 - TARGET_RELATIVE_MSE / coarse_relative_mse)


def rate_gain_bpw(remaining_sse: float, input_sse: float) -> float:
    require(math.isfinite(input_sse) and input_sse >= 0.0,
            "finite nonnegative input SSE")
    require(math.isfinite(remaining_sse) and remaining_sse >= 0.0,
            "finite nonnegative remaining SSE")
    require(remaining_sse <= input_sse * (1.0 + 2e-10) + 1e-20,
            "remaining SSE cannot exceed input SSE")
    if input_sse == 0.0:
        return 0.0
    if remaining_sse == 0.0:
        return math.inf
    return -0.5 * math.log2(remaining_sse / input_sse)


def graph_advantage_bpw(graph_remaining: float, public_remaining: float) -> float:
    require(graph_remaining >= 0.0 and public_remaining >= 0.0,
            "nonnegative graph/public remaining SSE")
    if graph_remaining == 0.0:
        return math.inf if public_remaining > 0.0 else 0.0
    if public_remaining == 0.0:
        return -math.inf
    return 0.5 * math.log2(public_remaining / graph_remaining)


def waterfill_reference(energies: Iterable[float], bits: float) -> dict[str, float]:
    """Scalar reverse waterfilling reference used by source-only tests."""
    rows = [float(value) for value in energies]
    require(rows and math.isfinite(bits) and bits >= 0.0,
            "waterfill reference inputs")
    require(all(math.isfinite(value) and value >= 0.0 for value in rows),
            "waterfill nonnegative finite energies")
    positive = [value for value in rows if value > 0.0]
    if not positive:
        return {"distortion": 0.0, "bits": 0.0, "theta": 0.0}
    if bits == 0.0:
        return {
            "distortion": math.fsum(rows), "bits": 0.0,
            "theta": max(positive),
        }
    lo = min(math.log(value) for value in positive) - 2.0 * bits * math.log(2.0) - 32.0
    hi = max(math.log(value) for value in positive) + 1.0
    for _ in range(160):
        middle = 0.5 * (lo + hi)
        allocated = 0.5 / math.log(2.0) * math.fsum(
            max(math.log(value) - middle, 0.0) for value in positive
        )
        if allocated > bits:
            lo = middle
        else:
            hi = middle
    log_theta = 0.5 * (lo + hi)
    theta = math.exp(log_theta)
    distortion = math.fsum(min(value, theta) for value in rows)
    allocated = 0.5 / math.log(2.0) * math.fsum(
        max(math.log(value) - log_theta, 0.0) for value in positive
    )
    return {"distortion": distortion, "bits": allocated, "theta": theta}


def _splitmix64(values: Any, xp: Any) -> Any:
    mask = xp.uint64(0xFFFFFFFFFFFFFFFF)
    def evaluate() -> Any:
        z = (values + xp.uint64(0x9E3779B97F4A7C15)) & mask
        z = ((z ^ (z >> xp.uint64(30)))
             * xp.uint64(0xBF58476D1CE4E5B9)) & mask
        z = ((z ^ (z >> xp.uint64(27)))
             * xp.uint64(0x94D049BB133111EB)) & mask
        z = z ^ (z >> xp.uint64(31))
        return z
    errstate = getattr(xp, "errstate", None)
    if errstate is None:
        return evaluate()
    with errstate(over="ignore"):
        return evaluate()


def _stable_permutation(symbols: Any, family: str, xp: Any) -> Any:
    require(family in ALL_FAMILIES, "known transform family")
    require(symbols.ndim == 2 and symbols.shape[1] == BLOCK_VALUES,
            "symbol block matrix")
    rows = symbols.shape[0]
    coordinate = xp.arange(BLOCK_VALUES, dtype=xp.int64)
    coordinates = xp.broadcast_to(coordinate, (rows, BLOCK_VALUES))
    q = symbols.astype(xp.int64, copy=False)
    if family == PUBLIC_FAMILY:
        return coordinates
    if family == "coarse_signed_path_dct":
        key = q
    elif family == "coarse_magnitude_path_dct":
        key = xp.abs(q) * xp.int64(2) + (q < 0).astype(xp.int64)
    else:
        # Frozen coarse-only XOR-neighbour signature. All terms remain within
        # I64 because v6 proves |q| <= 8,388,608.
        key = (
            q * xp.int64(257)
            + xp.take(q, coordinate ^ xp.int64(1), axis=1) * xp.int64(67)
            + xp.take(q, coordinate ^ xp.int64(2), axis=1) * xp.int64(31)
            + xp.take(q, coordinate ^ xp.int64(4), axis=1) * xp.int64(13)
            + xp.take(q, coordinate ^ xp.int64(8), axis=1) * xp.int64(5)
        )
    # Coordinate is an explicit deterministic tie breaker. The proven v6 I32
    # bound and frozen key coefficients keep this composite strictly inside
    # I64. This avoids CuPy 14.2's lack of batched lexsort.
    composite = key * xp.int64(BLOCK_VALUES) + coordinates
    return xp.argsort(composite, axis=1)


def graph_coefficients(
    residual_blocks: Any,
    symbol_blocks: Any,
    family: str,
    xp: Any,
    dct: Any,
) -> tuple[Any, dict[str, Any]]:
    require(residual_blocks.ndim == 2 and
            residual_blocks.shape == symbol_blocks.shape and
            residual_blocks.shape[1] == BLOCK_VALUES,
            "aligned residual/symbol block matrices")
    require(str(residual_blocks.dtype) == "float64",
            "FP64 residual blocks")
    permutation = _stable_permutation(symbol_blocks, family, xp)
    ordered = xp.take_along_axis(residual_blocks, permutation, axis=1)
    coefficients = dct(ordered, type=2, axis=1, norm="ortho")
    require(str(coefficients.dtype) == "float64", "FP64 DCT coefficients")
    input_energy = xp.sum(residual_blocks * residual_blocks, axis=1, dtype=xp.float64)
    spectral_energy = xp.sum(coefficients * coefficients, axis=1, dtype=xp.float64)
    maximum_error = float(xp.max(xp.abs(input_energy - spectral_energy)).item())
    maximum_scale = float(xp.max(input_energy).item())
    tolerance = 2e-10 * max(1.0, maximum_scale)
    require(maximum_error <= tolerance, "orthonormal DCT energy parity")
    return coefficients, {
        "family": family,
        "graph_is_decoder_visible_coarse_function": family != PUBLIC_FAMILY,
        "basis": "orthonormal DCT-II path-graph Fourier basis",
        "krylov_interpretation":
            "spectral envelope of path-Laplacian Lanczos/Chebyshev filters",
        "maximum_fp64_parseval_abs_error": maximum_error,
        "parseval_tolerance": tolerance,
    }


def _waterfill_rows(energies: Any, bits: float, xp: Any) -> tuple[Any, Any, Any]:
    require(energies.ndim == 2 and energies.shape[1] == BLOCK_VALUES,
            "waterfill energy matrix")
    require(bits >= 0.0, "waterfill bits")
    positive = energies > 0.0
    safe = xp.where(positive, energies, xp.float64(1.0))
    logs = xp.where(positive, xp.log(safe), -xp.inf)
    maximum = xp.max(logs, axis=1)
    minimum = xp.min(xp.where(positive, logs, xp.inf), axis=1)
    all_zero = ~xp.any(positive, axis=1)
    minimum = xp.where(all_zero, xp.float64(0.0), minimum)
    maximum = xp.where(all_zero, xp.float64(0.0), maximum)
    lo = minimum - xp.float64(2.0 * bits * math.log(2.0) + 32.0)
    hi = maximum + xp.float64(1.0)
    for _ in range(96):
        middle = 0.5 * (lo + hi)
        allocated = 0.5 / math.log(2.0) * xp.sum(
            xp.maximum(logs - middle[:, None], 0.0), axis=1,
            dtype=xp.float64,
        )
        too_many = allocated > bits
        lo = xp.where(too_many, middle, lo)
        hi = xp.where(too_many, hi, middle)
    log_theta = 0.5 * (lo + hi)
    theta = xp.exp(log_theta)
    distortion = xp.sum(
        xp.minimum(energies, theta[:, None]), axis=1, dtype=xp.float64)
    allocated = 0.5 / math.log(2.0) * xp.sum(
        xp.maximum(logs - log_theta[:, None], 0.0), axis=1,
        dtype=xp.float64,
    )
    distortion = xp.where(all_zero, xp.float64(0.0), distortion)
    allocated = xp.where(all_zero, xp.float64(0.0), allocated)
    theta = xp.where(all_zero, xp.float64(0.0), theta)
    return distortion, allocated, theta


def score_family(
    residual_blocks: Any,
    symbol_blocks: Any,
    family: str,
    xp: Any,
    dct: Any,
) -> dict[str, Any]:
    coefficients, graph_receipt = graph_coefficients(
        residual_blocks, symbol_blocks, family, xp, dct)
    energies = coefficients * coefficients
    input_per_block = xp.sum(
        residual_blocks * residual_blocks, axis=1, dtype=xp.float64)
    fixed_remaining = input_per_block - xp.sum(
        energies[:, :TOPK], axis=1, dtype=xp.float64)
    split = BLOCK_VALUES - TOPK
    partitioned = xp.partition(energies, split, axis=1)
    topk_captured = xp.sum(
        partitioned[:, split:], axis=1, dtype=xp.float64)
    topk_remaining = input_per_block - topk_captured
    waterfill_remaining, allocated, theta = _waterfill_rows(
        energies, float(FINE_BITS_PER_BLOCK), xp)
    fixed_cumulative = xp.sum(
        xp.cumsum(energies[:, :TOPK], axis=1, dtype=xp.float64),
        axis=0, dtype=xp.float64)
    descending = xp.flip(xp.sort(energies, axis=1), axis=1)[:, :TOPK]
    topk_cumulative = xp.sum(
        xp.cumsum(descending, axis=1, dtype=xp.float64),
        axis=0, dtype=xp.float64)
    input_sse = float(xp.sum(input_per_block, dtype=xp.float64).item())
    fixed_sse = max(0.0, float(xp.sum(fixed_remaining, dtype=xp.float64).item()))
    topk_sse = max(0.0, float(xp.sum(topk_remaining, dtype=xp.float64).item()))
    waterfill_sse = max(0.0, float(xp.sum(waterfill_remaining, dtype=xp.float64).item()))
    maximum_bit_error = float(
        xp.max(xp.abs(allocated - xp.float64(FINE_BITS_PER_BLOCK))).item())
    zero_blocks = int(xp.sum(input_per_block == 0.0).item())
    if zero_blocks:
        nonzero = input_per_block > 0.0
        if bool(xp.any(nonzero).item()):
            maximum_bit_error = float(xp.max(
                xp.abs(allocated[nonzero] - xp.float64(FINE_BITS_PER_BLOCK))
            ).item())
        else:
            maximum_bit_error = 0.0
    require(maximum_bit_error <= 2e-9, "waterfill exact bit budget")
    for value in (fixed_sse, topk_sse, waterfill_sse):
        require(value <= input_sse * (1.0 + 2e-10) + 1e-20,
                "oracle remaining SSE bound")
    return {
        "family": family,
        "blocks": int(residual_blocks.shape[0]),
        "input_sse_fp64": input_sse,
        "fixed_first384_exact_amplitudes_free_remaining_sse_fp64": fixed_sse,
        "free_support_top384_exact_amplitudes_free_remaining_sse_fp64": topk_sse,
        "ideal_384bit_gaussian_waterfill_remaining_sse_fp64": waterfill_sse,
        "fixed_first384_capture_fraction":
            (1.0 - fixed_sse / input_sse) if input_sse else 0.0,
        "free_support_top384_capture_fraction":
            (1.0 - topk_sse / input_sse) if input_sse else 0.0,
        "ideal_waterfill_rate_gain_bpw": rate_gain_bpw(waterfill_sse, input_sse),
        "waterfill_bits_per_nonzero_block": FINE_BITS_PER_BLOCK,
        "waterfill_maximum_abs_bit_budget_error": maximum_bit_error,
        "waterfill_zero_energy_blocks": zero_blocks,
        "waterfill_theta_min": float(xp.min(theta).item()),
        "waterfill_theta_max": float(xp.max(theta).item()),
        "fixed_rank_1_to_384_captured_sse_fp64": [
            float(value) for value in xp.asnumpy(fixed_cumulative)],
        "free_support_rank_1_to_384_captured_sse_fp64": [
            float(value) for value in xp.asnumpy(topk_cumulative)],
        "graph_receipt": graph_receipt,
    }


def new_accumulator() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for family in ALL_FAMILIES:
        result[family] = {
            "family": family,
            "blocks": 0,
            "input_sse_fp64": 0.0,
            "fixed_first384_exact_amplitudes_free_remaining_sse_fp64": 0.0,
            "free_support_top384_exact_amplitudes_free_remaining_sse_fp64": 0.0,
            "ideal_384bit_gaussian_waterfill_remaining_sse_fp64": 0.0,
            "maximum_fp64_parseval_abs_error": 0.0,
            "maximum_waterfill_abs_bit_budget_error": 0.0,
            "waterfill_zero_energy_blocks": 0,
            "fixed_rank_1_to_384_captured_sse_fp64": [0.0] * TOPK,
            "free_support_rank_1_to_384_captured_sse_fp64": [0.0] * TOPK,
        }
    return result


def add_score(accumulator: dict[str, dict[str, Any]], score: Mapping[str, Any]) -> None:
    family = str(score["family"])
    require(family in accumulator, "accumulator family")
    row = accumulator[family]
    row["blocks"] += int(score["blocks"])
    for name in (
        "input_sse_fp64",
        "fixed_first384_exact_amplitudes_free_remaining_sse_fp64",
        "free_support_top384_exact_amplitudes_free_remaining_sse_fp64",
        "ideal_384bit_gaussian_waterfill_remaining_sse_fp64",
    ):
        row[name] += float(score[name])
    row["maximum_fp64_parseval_abs_error"] = max(
        row["maximum_fp64_parseval_abs_error"],
        float(score["graph_receipt"]["maximum_fp64_parseval_abs_error"]),
    )
    row["maximum_waterfill_abs_bit_budget_error"] = max(
        row["maximum_waterfill_abs_bit_budget_error"],
        float(score["waterfill_maximum_abs_bit_budget_error"]),
    )
    row["waterfill_zero_energy_blocks"] += int(score["waterfill_zero_energy_blocks"])
    for name in (
        "fixed_rank_1_to_384_captured_sse_fp64",
        "free_support_rank_1_to_384_captured_sse_fp64",
    ):
        values = score[name]
        require(len(values) == TOPK, "complete rank curve")
        row[name] = [left + float(right) for left, right in zip(row[name], values)]


def finalize_accumulator(
    accumulator: Mapping[str, Mapping[str, Any]], source_energy: float,
) -> dict[str, dict[str, Any]]:
    require(math.isfinite(source_energy) and source_energy > 0.0,
            "positive pooled source energy")
    output: dict[str, dict[str, Any]] = {}
    reference_input: float | None = None
    for family in ALL_FAMILIES:
        row = dict(accumulator[family])
        input_sse = float(row["input_sse_fp64"])
        if reference_input is None:
            reference_input = input_sse
        else:
            require(abs(input_sse - reference_input) <=
                    2e-10 * max(1.0, reference_input),
                    "candidate input SSE identity")
        for prefix, field in (
            ("fixed_first384", "fixed_first384_exact_amplitudes_free_remaining_sse_fp64"),
            ("free_support_top384", "free_support_top384_exact_amplitudes_free_remaining_sse_fp64"),
            ("ideal_384bit_waterfill", "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"),
        ):
            remaining = float(row[field])
            row[f"{prefix}_capture_fraction"] = (
                1.0 - remaining / input_sse if input_sse else 0.0)
            row[f"{prefix}_rate_gain_bpw"] = rate_gain_bpw(remaining, input_sse)
            row[f"{prefix}_final_relative_mse"] = remaining / source_energy
        ranks = list(range(1, TOPK + 1))
        isotropic = [rank / BLOCK_VALUES for rank in ranks]
        fixed_curve = [
            float(value) / input_sse if input_sse else 0.0
            for value in row["fixed_rank_1_to_384_captured_sse_fp64"]]
        topk_curve = [
            float(value) / input_sse if input_sse else 0.0
            for value in row["free_support_rank_1_to_384_captured_sse_fp64"]]
        row["rank_curve"] = {
            "ranks_1_to_384": ranks,
            "isotropic_fixed_subspace_expected_capture_fraction": isotropic,
            "fixed_capture_fraction": fixed_curve,
            "fixed_minus_isotropic_capture_fraction": [
                observed - baseline
                for observed, baseline in zip(fixed_curve, isotropic)],
            "free_support_topk_capture_fraction": topk_curve,
            "free_support_is_source_leaking_and_has_no_isotropic_rank_identity": True,
        }
        output[family] = row
    return output


def source_gate(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    graph_top = min(
        GRAPH_FAMILIES,
        key=lambda name: (
            float(metrics[name]["free_support_top384_final_relative_mse"]), name),
    )
    graph_waterfill = min(
        GRAPH_FAMILIES,
        key=lambda name: (
            float(metrics[name]["ideal_384bit_waterfill_final_relative_mse"]), name),
    )
    top_relative = float(
        metrics[graph_top]["free_support_top384_final_relative_mse"])
    waterfill_relative = float(
        metrics[graph_waterfill]["ideal_384bit_waterfill_final_relative_mse"])
    top_survives = top_relative <= TARGET_RELATIVE_MSE + TARGET_TOLERANCE
    waterfill_survives = waterfill_relative <= TARGET_RELATIVE_MSE + TARGET_TOLERANCE
    if not top_survives:
        status = "HARD_KILL_CONTINUOUS_GRAPH_ENVELOPE_MISSES_TARGET"
    elif not waterfill_survives:
        status = "SOURCE_SURVIVES_CONTINUOUS_ENVELOPE_OPEN_CONTROLS_WATERFILL_AT_RISK"
    else:
        status = "SOURCE_SURVIVES_CONTINUOUS_AND_IDEAL_WATERFILL_OPEN_CONTROLS"
    return {
        "status": status,
        "controls_may_open": top_survives,
        "best_free_support_family": graph_top,
        "best_free_support_final_relative_mse": top_relative,
        "best_waterfill_family": graph_waterfill,
        "best_waterfill_final_relative_mse": waterfill_relative,
        "continuous_target_pass": top_survives,
        "ideal_waterfill_target_pass": waterfill_survives,
    }


def affine_permutation_control(
    residual_blocks: Any, global_block_base: int, xp: Any,
) -> tuple[Any, dict[str, Any]]:
    require(residual_blocks.ndim == 2 and residual_blocks.shape[1] == BLOCK_VALUES,
            "permutation residual blocks")
    rows = residual_blocks.shape[0]
    block = xp.arange(rows, dtype=xp.int64) + xp.int64(global_block_base)
    a = (((block * xp.int64(2654435761) + xp.int64(2246822519))
          & xp.int64(BLOCK_VALUES - 1)) | xp.int64(1))
    b = ((block * xp.int64(3266489917) + xp.int64(668265263))
         & xp.int64(BLOCK_VALUES - 1))
    coordinate = xp.arange(BLOCK_VALUES, dtype=xp.int64)
    permutation = (
        a[:, None] * coordinate[None, :] + b[:, None]
    ) & xp.int64(BLOCK_VALUES - 1)
    controlled = xp.take_along_axis(residual_blocks, permutation, axis=1)
    before = xp.sum(residual_blocks * residual_blocks, axis=1, dtype=xp.float64)
    after = xp.sum(controlled * controlled, axis=1, dtype=xp.float64)
    error = float(xp.max(xp.abs(before - after)).item())
    require(error <= 2e-12 * max(1.0, float(xp.max(before).item())),
            "permutation energy parity")
    return controlled, {
        "schema": "tactic-cage-affine-permutation-control-v0",
        "global_block_base": global_block_base,
        "rows": int(rows),
        "maximum_fp64_energy_error": error,
        "odd_multiplier_proves_bijection_mod_4096": True,
    }


def gaussian_moment_control(
    residual_blocks: Any, global_block_base: int, xp: Any,
) -> tuple[Any, dict[str, Any]]:
    require(residual_blocks.ndim == 2 and residual_blocks.shape[1] == BLOCK_VALUES,
            "Gaussian residual blocks")
    rows = residual_blocks.shape[0]
    coordinate = xp.arange(BLOCK_VALUES, dtype=xp.uint64)[None, :]
    block = (xp.arange(rows, dtype=xp.uint64)
             + xp.uint64(global_block_base))[:, None]
    counter = coordinate + block * xp.uint64(BLOCK_VALUES)
    raw1 = _splitmix64(counter ^ xp.uint64(0xD1B54A32D192ED03), xp)
    raw2 = _splitmix64(counter ^ xp.uint64(0x94D049BB133111EB), xp)
    scale = float(1 << 53)
    u1 = ((raw1 >> xp.uint64(11)).astype(xp.float64) + xp.float64(0.5)) / scale
    u2 = ((raw2 >> xp.uint64(11)).astype(xp.float64) + xp.float64(0.5)) / scale
    gaussian = xp.sqrt(-2.0 * xp.log(u1)) * xp.cos(2.0 * math.pi * u2)
    source_mean = xp.mean(residual_blocks, axis=1, dtype=xp.float64)
    centered_source = residual_blocks - source_mean[:, None]
    source_centered_sse = xp.sum(
        centered_source * centered_source, axis=1, dtype=xp.float64)
    gaussian_mean = xp.mean(gaussian, axis=1, dtype=xp.float64)
    centered_gaussian = gaussian - gaussian_mean[:, None]
    gaussian_sse = xp.sum(
        centered_gaussian * centered_gaussian, axis=1, dtype=xp.float64)
    require(bool(xp.all(gaussian_sse > 0.0).item()), "Gaussian control energy")
    ratio = xp.sqrt(xp.where(
        source_centered_sse > 0.0,
        source_centered_sse / gaussian_sse,
        xp.float64(0.0),
    ))
    controlled = source_mean[:, None] + ratio[:, None] * centered_gaussian
    observed_mean = xp.mean(controlled, axis=1, dtype=xp.float64)
    observed_centered = controlled - observed_mean[:, None]
    observed_sse = xp.sum(
        observed_centered * observed_centered, axis=1, dtype=xp.float64)
    mean_error = float(xp.max(xp.abs(observed_mean - source_mean)).item())
    energy_error = float(xp.max(xp.abs(observed_sse - source_centered_sse)).item())
    scale_mean = max(1.0, float(xp.max(xp.abs(source_mean)).item()))
    scale_energy = max(1.0, float(xp.max(source_centered_sse).item()))
    require(mean_error <= 2e-12 * scale_mean, "Gaussian matched mean")
    require(energy_error <= 2e-10 * scale_energy, "Gaussian matched centered energy")
    return controlled, {
        "schema": "tactic-cage-block-moment-gaussian-control-v0",
        "global_block_base": global_block_base,
        "rows": int(rows),
        "maximum_fp64_mean_error": mean_error,
        "maximum_fp64_centered_energy_error": energy_error,
        "counter_generator": "SplitMix64 plus Box-Muller",
    }


def controls_gate(
    source_metrics: Mapping[str, Mapping[str, Any]],
    control_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    winning_family: str,
) -> dict[str, Any]:
    require(winning_family in GRAPH_FAMILIES, "winning graph family")
    qwen_advantage = graph_advantage_bpw(
        float(source_metrics[winning_family][
            "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"]),
        float(source_metrics[PUBLIC_FAMILY][
            "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"]),
    )
    controls: dict[str, float] = {}
    qwen_graph_rate_gain = rate_gain_bpw(
        float(source_metrics[winning_family][
            "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"]),
        float(source_metrics[winning_family]["input_sse_fp64"]),
    )
    control_graph_rate_gain: dict[str, float] = {}
    rank_excess: dict[str, Any] = {}
    qwen_curve = source_metrics[winning_family]["rank_curve"]
    for name in ("permutation", "gaussian"):
        metrics = control_metrics[name]
        controls[name] = graph_advantage_bpw(
            float(metrics[winning_family][
                "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"]),
            float(metrics[PUBLIC_FAMILY][
                "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"]),
        )
        control_graph_rate_gain[name] = rate_gain_bpw(
            float(metrics[winning_family][
                "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"]),
            float(metrics[winning_family]["input_sse_fp64"]),
        )
        control_curve = metrics[winning_family]["rank_curve"]
        rank_excess[name] = {
            "fixed_qwen_minus_control_capture_fraction": [
                float(left) - float(right) for left, right in zip(
                    qwen_curve["fixed_capture_fraction"],
                    control_curve["fixed_capture_fraction"],
                )
            ],
            "free_support_qwen_minus_control_capture_fraction": [
                float(left) - float(right) for left, right in zip(
                    qwen_curve["free_support_topk_capture_fraction"],
                    control_curve["free_support_topk_capture_fraction"],
                )
            ],
        }
    largest_control = max(controls.values())
    largest_control_rate_gain = max(control_graph_rate_gain.values())
    source_specific = qwen_graph_rate_gain - largest_control_rate_gain
    target_pass = float(source_metrics[winning_family][
        "ideal_384bit_waterfill_final_relative_mse"]) <= (
            TARGET_RELATIVE_MSE + TARGET_TOLERANCE)
    graph_pass = qwen_advantage + 1e-15 >= MIN_GRAPH_ADVANTAGE_BPW
    specific_pass = source_specific + 1e-15 >= MIN_SOURCE_SPECIFIC_ADVANTAGE_BPW
    if not target_pass:
        status = "HARD_KILL_IDEAL_384BIT_WATERFILL_MISSES_TARGET"
    elif not graph_pass or not specific_pass:
        status = "HARD_KILL_COARSE_GRAPH_NOT_SOURCE_SPECIFIC_0P03_BPW"
    else:
        status = "ELIGIBLE_FOR_FINITE_GRAPH_LIFTING_BUILD"
    return {
        "status": status,
        "winning_family": winning_family,
        "qwen_graph_over_public_advantage_bpw": qwen_advantage,
        "control_graph_over_public_advantage_bpw": controls,
        "largest_control_advantage_bpw": largest_control,
        "qwen_control_subtracted_advantage_bpw": source_specific,
        "qwen_graph_ideal_waterfill_rate_gain_bpw": qwen_graph_rate_gain,
        "control_graph_ideal_waterfill_rate_gain_bpw": control_graph_rate_gain,
        "largest_control_graph_rate_gain_bpw": largest_control_rate_gain,
        "rank_1_to_384_qwen_minus_control_excess_capture": rank_excess,
        "promotion_uses_only_qwen_minus_matched_control_excess_gain": True,
        "target_pass": target_pass,
        "minimum_graph_advantage_pass": graph_pass,
        "minimum_source_specific_advantage_pass": specific_pass,
        "composite_gap_eligible":
            status == "ELIGIBLE_FOR_FINITE_GRAPH_LIFTING_BUILD" and
            source_specific + 1e-15 >= COMPOSITE_GAP_BPW,
    }
