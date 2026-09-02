#!/usr/bin/env python3
"""Non-dyadic Ramanujan and pullback-charged AR/Hankel source oracles."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


class OracleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def _as_host(xp: Any, value: Any) -> Any:
    return xp.asnumpy(value) if hasattr(xp, "asnumpy") else value


def primitive_real_frequencies(period: int) -> tuple[tuple[int, str], ...]:
    """Canonical real basis labels spanning the exact-period-p subspace."""

    require(type(period) is int and period >= 3, "period")
    rows = []
    for frequency in range(1, period // 2 + 1):
        if math.gcd(frequency, period) != 1:
            continue
        partner = period - frequency
        if frequency > partner:
            continue
        rows.append((frequency, "cos"))
        if frequency != partner:
            rows.append((frequency, "sin"))
    require(len(rows) == sum(math.gcd(value, period) == 1 for value in range(1, period)), "Ramanujan dimension")
    return tuple(rows)


def build_ramanujan_basis(
    xp: Any,
    *,
    length: int,
    periods: Sequence[int],
    maximum_columns: int,
    dtype: Any | None = None,
) -> dict[str, Any]:
    """Build one public QR-orthonormalized non-dyadic real dictionary."""

    require(type(length) is int and length >= 64, "basis length")
    require(type(maximum_columns) is int and 1 <= maximum_columns < length, "basis columns")
    require(bool(periods) and len(set(periods)) == len(periods), "unique periods")
    dtype = xp.float64 if dtype is None else dtype
    coordinate = xp.arange(length, dtype=dtype)
    atoms = []
    labels = []
    for period in periods:
        for frequency, kind in primitive_real_frequencies(int(period)):
            angle = (2.0 * math.pi * frequency / period) * coordinate
            atom = xp.cos(angle) if kind == "cos" else xp.sin(angle)
            atoms.append(atom)
            labels.append({"period": int(period), "frequency": frequency, "kind": kind})
            if len(atoms) == maximum_columns:
                break
        if len(atoms) == maximum_columns:
            break
    require(len(atoms) == maximum_columns, "period bank supplies requested columns")
    dictionary = xp.stack(atoms, axis=1)
    q, r = xp.linalg.qr(dictionary, mode="reduced")
    diagonal = xp.diag(r)
    signs = xp.where(diagonal < 0, -1.0, 1.0).astype(dtype, copy=False)
    q = q * signs[None, :]
    gram = q.T @ q
    identity = xp.eye(maximum_columns, dtype=dtype)
    max_error = float(xp.max(xp.abs(gram - identity)).item())
    require(math.isfinite(max_error) and max_error <= 5e-9, "Ramanujan QR orthogonality")
    return {
        "basis": q,
        "labels": tuple(labels),
        "length": length,
        "columns": maximum_columns,
        "orthogonality_max_abs_error": max_error,
        "basis_is_public_and_source_independent": True,
    }


def ceil_log2_binomial(n: int, k: int) -> int:
    require(type(n) is int and type(k) is int and 0 <= k <= n, "binomial geometry")
    count = math.comb(n, k)
    return 0 if count <= 1 else (count - 1).bit_length()


def maximum_literal_support(columns: int, bits: int = 384) -> int:
    """Largest per-block support fitting rank, combinatorial support and FP16 amplitudes."""

    require(type(columns) is int and columns > 0 and type(bits) is int and bits > 0, "support budget")
    best = 0
    for rank in range(1, min(columns, bits // 16) + 1):
        used = 9 + ceil_log2_binomial(columns, rank) + 16 * rank
        if used <= bits:
            best = rank
    return best


def _waterfill(
    dimensions: Sequence[float],
    energies: Sequence[float],
    total_bits: float,
) -> dict[str, Any]:
    require(len(dimensions) == len(energies) and bool(dimensions), "waterfill geometry")
    require(math.isfinite(total_bits) and total_bits >= 0.0, "waterfill bits")
    d = [float(value) for value in dimensions]
    e = [float(value) for value in energies]
    require(all(value > 0.0 and math.isfinite(value) for value in d), "waterfill dimensions")
    require(all(value >= 0.0 and math.isfinite(value) for value in e), "waterfill energies")
    positive = [index for index, value in enumerate(e) if value > 0.0]
    if not positive:
        return {"distortion": 0.0, "used_bits": 0.0, "allocations": [0.0] * len(e)}
    log_variances = [math.log2(e[index] / d[index]) for index in positive]
    low = min(log_variances) - 2.0 * total_bits / min(d) - 4.0
    high = max(log_variances)
    for _ in range(200):
        level = 0.5 * (low + high)
        used = sum(0.5 * d[index] * max(0.0, math.log2(e[index] / d[index]) - level) for index in positive)
        if used > total_bits:
            low = level
        else:
            high = level
    log_level = high
    allocations = [0.0] * len(e)
    distortion = 0.0
    used_bits = 0.0
    for index, (dimension, energy) in enumerate(zip(d, e, strict=True)):
        if energy == 0.0:
            continue
        bits = 0.5 * dimension * max(0.0, math.log2(energy / dimension) - log_level)
        allocations[index] = bits
        used_bits += bits
        distortion += energy * 2.0 ** (-2.0 * bits / dimension)
    require(abs(used_bits - total_bits) <= 1e-7 * max(1.0, total_bits), "waterfill closure")
    return {"distortion": distortion, "used_bits": used_bits, "allocations": allocations}


def ramanujan_panel_metrics(
    xp: Any,
    residual_blocks: Any,
    basis_record: dict[str, Any],
    *,
    source_energy: float,
    fine_bits_per_block: int = 384,
) -> dict[str, Any]:
    residual = xp.asarray(residual_blocks, dtype=xp.float64)
    q = xp.asarray(basis_record["basis"], dtype=xp.float64)
    require(residual.ndim == 2 and residual.shape[1] == q.shape[0], "residual block geometry")
    blocks, length = map(int, residual.shape)
    columns = int(q.shape[1])
    require(blocks > 0 and 1 <= columns < length, "Ramanujan panel geometry")
    input_sse = float(xp.sum(residual * residual, dtype=xp.float64).item())
    require(input_sse > 0.0 and math.isfinite(source_energy) and source_energy > 0.0, "Ramanujan energies")
    coefficients = residual @ q
    mode_energy = xp.sum(coefficients * coefficients, axis=0, dtype=xp.float64)
    captured_all = float(xp.sum(mode_energy, dtype=xp.float64).item())
    outside = max(0.0, input_sse - captured_all)

    prefix = min(384, columns)
    free_prefix_remaining = max(
        0.0,
        input_sse - float(xp.sum(mode_energy[:prefix], dtype=xp.float64).item()),
    )
    fixed_fp16_rank = min(fine_bits_per_block // 16, columns)
    rounded = coefficients[:, :fixed_fp16_rank].astype(xp.float16).astype(xp.float64)
    fp16_error = float(xp.sum((coefficients[:, :fixed_fp16_rank] - rounded) ** 2, dtype=xp.float64).item())
    fixed_fp16_remaining = max(
        0.0,
        input_sse - float(xp.sum(mode_energy[:fixed_fp16_rank], dtype=xp.float64).item()) + fp16_error,
    )

    literal_rank = maximum_literal_support(columns, fine_bits_per_block)
    if literal_rank:
        order = xp.argsort(xp.abs(coefficients), axis=1)[:, ::-1][:, :literal_rank]
        selected = xp.take_along_axis(coefficients, order, axis=1)
        selected_rounded = selected.astype(xp.float16).astype(xp.float64)
        literal_captured = float(xp.sum(selected * selected, dtype=xp.float64).item())
        literal_error = float(xp.sum((selected - selected_rounded) ** 2, dtype=xp.float64).item())
        literal_remaining = max(0.0, input_sse - literal_captured + literal_error)
        literal_bits = 9 + ceil_log2_binomial(columns, literal_rank) + 16 * literal_rank
    else:
        literal_remaining = input_sse
        literal_bits = 0

    dimensions = [float(blocks)] * columns + [float(blocks * (length - columns))]
    energies = [float(value) for value in _as_host(xp, mode_energy)] + [outside]
    if outside == 0.0:
        dimensions.pop()
        energies.pop()
    waterfill = _waterfill(dimensions, energies, float(blocks * fine_bits_per_block))
    ideal_remaining = float(waterfill["distortion"])
    return {
        "blocks": blocks,
        "block_values": length,
        "basis_columns": columns,
        "input_sse": input_sse,
        "source_energy": source_energy,
        "fixed_free_prefix_rank": prefix,
        "fixed_free_prefix_remaining_sse": free_prefix_remaining,
        "fixed_fp16_rank": fixed_fp16_rank,
        "fixed_fp16_descriptor_bits_per_block": 16 * fixed_fp16_rank,
        "fixed_fp16_remaining_sse": fixed_fp16_remaining,
        "source_selected_literal_rank": literal_rank,
        "source_selected_literal_bits_per_block": literal_bits,
        "source_selected_literal_remaining_sse": literal_remaining,
        "ideal_public_basis_waterfill_remaining_sse": ideal_remaining,
        "ideal_waterfill_has_finite_backend": False,
        "source_selected_support_is_not_free": True,
        "all_metrics_scored_in_source_coordinates": True,
    }


def fit_ar_filter(xp: Any, values: Any, order: int) -> Any:
    sequence = xp.asarray(values, dtype=xp.float64).reshape(-1)
    length = int(sequence.size)
    require(type(order) is int and 1 <= order <= 12 and length >= 8 * order, "AR order")
    columns = [sequence[order - lag:length - lag] for lag in range(1, order + 1)]
    design = xp.stack(columns, axis=1)
    target = -sequence[order:]
    gram = design.T @ design
    trace = float(xp.trace(gram).item())
    ridge = max(trace / order, 1e-30) * 2.0 ** -24
    coefficients = xp.linalg.solve(gram + ridge * xp.eye(order, dtype=xp.float64), design.T @ target)
    return coefficients.astype(xp.float16).astype(xp.float64)


def ar_innovations(xp: Any, values: Any, coefficients: Any) -> Any:
    sequence = xp.asarray(values, dtype=xp.float64).reshape(-1)
    coefficients = xp.asarray(coefficients, dtype=xp.float64).reshape(-1)
    order = int(coefficients.size)
    output = sequence.copy()
    for lag in range(1, order + 1):
        output[order:] += coefficients[lag - 1] * sequence[order - lag:-lag]
    return output


def inverse_noise_gain(coefficients: Sequence[float], length: int) -> float:
    values = tuple(float(value) for value in coefficients)
    order = len(values)
    require(1 <= order <= 12 and type(length) is int and length >= 8 * order, "inverse gain geometry")
    impulse = [0.0] * length
    impulse[0] = 1.0
    for index in range(1, length):
        value = 0.0
        for lag in range(1, min(order, index) + 1):
            value -= values[lag - 1] * impulse[index - lag]
        impulse[index] = value
        if not math.isfinite(value) or abs(value) > 1e100:
            return math.inf
    return sum((length - lag) * value * value for lag, value in enumerate(impulse)) / length


def ar_hankel_panel_metrics(
    xp: Any,
    residual_blocks: Any,
    *,
    source_energy: float,
    orders: Sequence[int] = (1, 2, 4, 8, 12),
    fine_bits_per_block: int = 384,
) -> dict[str, Any]:
    residual = xp.asarray(residual_blocks, dtype=xp.float64)
    require(residual.ndim == 2 and residual.shape[0] > 0, "AR residual panel")
    blocks, length = map(int, residual.shape)
    input_sse = float(xp.sum(residual * residual, dtype=xp.float64).item())
    require(input_sse > 0.0 and source_energy > 0.0, "AR energies")
    rows = []
    for order in orders:
        require(type(order) is int and 1 <= order <= 12, "AR order bank")
        descriptor_bits = 4 + 16 * order
        require(descriptor_bits < fine_bits_per_block, "AR descriptor budget")
        innovation_sse = 0.0
        weighted_noise_sse = 0.0
        maximum_gain = 0.0
        for block in residual:
            coefficients = fit_ar_filter(xp, block, order)
            innovations = ar_innovations(xp, block, coefficients)
            energy = float(xp.sum(innovations * innovations, dtype=xp.float64).item())
            gain = inverse_noise_gain([float(value) for value in _as_host(xp, coefficients)], length)
            innovation_sse += energy
            weighted_noise_sse += energy * gain
            maximum_gain = max(maximum_gain, gain)
        payload_rate = (fine_bits_per_block - descriptor_bits) / length
        ideal_remaining = weighted_noise_sse * 2.0 ** (-2.0 * payload_rate)
        rows.append({
            "order": order,
            "coefficient_encoding": "IEEE binary16",
            "order_selector_bits_per_block": 4,
            "coefficient_bits_per_block": 16 * order,
            "descriptor_bits_per_block": descriptor_bits,
            "innovation_bits_per_block": fine_bits_per_block - descriptor_bits,
            "innovation_rate_bpw": payload_rate,
            "innovation_sse": innovation_sse,
            "innovation_energy_fraction": innovation_sse / input_sse,
            "finite_length_inverse_noise_gain_weighted_sse": weighted_noise_sse,
            "maximum_block_inverse_noise_gain": maximum_gain,
            "ideal_iid_gaussian_innovation_remaining_sse": ideal_remaining,
            "relative_mse": ideal_remaining / source_energy,
            "finite_innovation_codec_executed": False,
            "pullback_noise_amplification_charged": True,
        })
    winner = min(rows, key=lambda row: (row["ideal_iid_gaussian_innovation_remaining_sse"], row["order"]))
    baseline = input_sse * 2.0 ** (-2.0 * fine_bits_per_block / length)
    return {
        "blocks": blocks,
        "block_values": length,
        "input_sse": input_sse,
        "source_energy": source_energy,
        "no_predictor_ideal_gaussian_remaining_sse": baseline,
        "orders": rows,
        "winner": winner,
        "source_metric_pullback_is_not_ignored": True,
        "dominant_oracle_only": True,
    }


def odd_affine_permutation(length: int, seed: int) -> tuple[int, ...]:
    require(type(length) is int and length > 1 and length & (length - 1) == 0, "permutation length")
    multiplier = (2 * (int(seed) % (length // 2)) + 1) % length
    offset = (int(seed) >> 17) % length
    result = tuple((multiplier * index + offset) % length for index in range(length))
    require(len(set(result)) == length, "odd-affine bijection")
    return result


def phase_destroyed_blocks(xp: Any, residual_blocks: Any, seed: int) -> Any:
    """Apply one frozen within-block odd-affine permutation to every block."""

    residual = xp.asarray(residual_blocks, dtype=xp.float64)
    require(residual.ndim == 2 and residual.shape[0] > 0, "permutation control geometry")
    permutation = odd_affine_permutation(int(residual.shape[1]), int(seed))
    return residual[:, xp.asarray(permutation, dtype=xp.int64)]


def moment_matched_gaussian_blocks(xp: Any, residual_blocks: Any, seed: int) -> Any:
    """Generate blockwise Gaussian controls matching mean and centered energy."""

    residual = xp.asarray(residual_blocks, dtype=xp.float64)
    require(residual.ndim == 2 and residual.shape[0] > 0, "Gaussian control geometry")
    require(type(seed) is int and 0 <= seed < 2**32, "Gaussian control seed")
    generator = xp.random.RandomState(seed)
    gaussian = generator.standard_normal(residual.shape).astype(xp.float64, copy=False)
    source_mean = xp.mean(residual, axis=1, keepdims=True, dtype=xp.float64)
    source_centered = residual - source_mean
    source_centered_energy = xp.sum(source_centered * source_centered, axis=1, keepdims=True, dtype=xp.float64)
    gaussian_centered = gaussian - xp.mean(gaussian, axis=1, keepdims=True, dtype=xp.float64)
    gaussian_energy = xp.sum(gaussian_centered * gaussian_centered, axis=1, keepdims=True, dtype=xp.float64)
    require(bool(xp.all(gaussian_energy > 0.0).item()), "Gaussian control nonzero energy")
    scale = xp.sqrt(source_centered_energy / gaussian_energy)
    matched = source_mean + gaussian_centered * scale
    observed_mean = xp.mean(matched, axis=1, keepdims=True, dtype=xp.float64)
    observed_centered = matched - observed_mean
    observed_energy = xp.sum(observed_centered * observed_centered, axis=1, keepdims=True, dtype=xp.float64)
    mean_error = float(xp.max(xp.abs(observed_mean - source_mean)).item())
    energy_scale = xp.maximum(source_centered_energy, 1.0)
    energy_error = float(xp.max(xp.abs(observed_energy - source_centered_energy) / energy_scale).item())
    require(mean_error <= 2e-12 and energy_error <= 2e-12, "Gaussian moment closure")
    return matched
