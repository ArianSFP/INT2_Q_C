#!/usr/bin/env python3
"""CPU-only diagnostic for MAP-SC versus rate-matched polar codeword search.

This probe deliberately uses only procedural Gaussian sources and NumPy.  It
does not import CuPy, open Qwen weights, or modify a codec artifact.  At the
default N=16 and D=2^-5, levels 3--6 of the frozen BEC-surrogate construction
are full-rate.  We therefore enumerate *every* legal assignment in the two
constrained low bitplanes, then optimize the full-rate upper bitplanes over a
wide rate/distortion Lagrange grid.  Every improving candidate is replayed
through the exact causal prior model and the literal Q16 arithmetic coder.

The Lagrange family is not an exhaustive constrained nearest-codeword solver;
the result is a direct opportunity diagnostic, not a proof of global
optimality.  The unrestricted nearest reconstruction is exact for this small
construction.  A retained rate-matched reconstruction needs no selector: its
own information-bit stream identifies it, and the stream is zero-extended to
the MAP-SC control length with an arithmetic round-trip check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


UPSTREAM_ENCODER_SHA256 = "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0"
UPSTREAM_BEC_SHA256 = "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bit_reverse_indices(n: int) -> np.ndarray:
    width = int(math.log2(n))
    if 1 << width != n:
        raise ValueError("N must be a power of two")
    source = np.arange(n, dtype=np.uint32)
    result = np.zeros(n, dtype=np.uint32)
    for _ in range(width):
        result = (result << np.uint32(1)) | (source & np.uint32(1))
        source >>= np.uint32(1)
    return result.astype(np.int64)


def sc_layers(n: int) -> np.ndarray:
    depth = int(math.log2(n))
    result = np.ones(n + 1, dtype=np.int32)
    result[0] = depth
    for one_based in range(2, n + 1):
        end = 1
        cursor = one_based
        while cursor % 2 == 1:
            end += 1
            cursor = (cursor + 1) // 2
        result[one_based - 1] = end
    return result


def polar_transform(bits: np.ndarray) -> np.ndarray:
    result = np.asarray(bits, dtype=np.uint8).copy()
    step = 1
    while step < result.shape[-1]:
        view = result.reshape(*result.shape[:-1], -1, 2 * step)
        view[..., :step] ^= view[..., step:]
        step *= 2
    return result


def periodic_binary_capacity(
    sigma: float, grid: int = 1 << 17, neighbors: int = 16
) -> float:
    y = (np.arange(grid, dtype=np.float64) + 0.5) * (2.0 / grid)
    ks = np.arange(-neighbors, neighbors + 1, dtype=np.float64)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    p0 = (
        np.exp(-0.5 * ((y[:, None] + 2.0 * ks[None, :]) / sigma) ** 2).sum(1)
        * norm
    )
    p1 = (
        np.exp(-0.5 * ((y[:, None] - 1.0 + 2.0 * ks[None, :]) / sigma) ** 2).sum(1)
        * norm
    )
    mixture = 0.5 * (p0 + p1)
    posterior = np.divide(
        p1, p0 + p1, out=np.full_like(p1, 0.5), where=(p0 + p1) > 0
    )
    entropy = -(
        posterior * np.log2(np.maximum(posterior, 1e-300))
        + (1.0 - posterior)
        * np.log2(np.maximum(1.0 - posterior, 1e-300))
    )
    return float(1.0 - np.sum(mixture * entropy) * (2.0 / grid))


def bec_synthesized_z(capacity: float, n: int) -> np.ndarray:
    full = 1 << 31
    capacity_q31 = min(full, max(0, int(round(float(capacity) * full))))
    values = np.full(n, full - capacity_q31, dtype=np.uint64)
    step = 1
    while step < n:
        view = values.reshape(-1, 2 * step)
        left = view[:, :step].copy()
        right = view[:, step:].copy()
        product = (left * right + np.uint64(1 << 30)) >> np.uint64(31)
        view[:, :step] = left + right - product
        view[:, step:] = product
        step *= 2
    return values


def bec_flags(n: int, capacities: list[float]) -> list[np.ndarray]:
    reverse = bit_reverse_indices(n)
    result: list[np.ndarray] = []
    for capacity in capacities:
        keep = min(n, max(0, int(math.ceil(n * float(capacity)))))
        scores = bec_synthesized_z(capacity, n)
        order = np.lexsort((np.arange(n, dtype=np.int64), scores))
        external = np.ones(n, dtype=np.uint8)
        external[order[:keep]] = 0
        result.append(external[reverse].copy())
    return result


@dataclass
class SCResult:
    external_u: np.ndarray
    internal_u: np.ndarray
    selected_nll_bits: float
    selected_bits: np.ndarray
    selected_freq1_u16: np.ndarray


def sc_encode_ratio(
    leaf_lr: np.ndarray,
    freeze_flag: np.ndarray,
    frozen_external: np.ndarray,
    reverse: np.ndarray,
    layers: np.ndarray,
    *,
    forced_internal: np.ndarray | None = None,
    score_selected: bool = False,
) -> SCResult:
    """Pure-NumPy byte-for-byte arithmetic translation of the frozen SC core."""
    lr_in = np.clip(np.asarray(leaf_lr, dtype=np.float64), 1e-30, 1e30)
    n = lr_in.size
    depth = int(math.log2(n))
    lr_reg = np.ones((n // 2, depth), dtype=np.float64)
    mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)
    internal = np.zeros(n, dtype=np.uint8)
    nll = 0.0
    selected_bits: list[int] = []
    selected_freq1: list[int] = []

    def adjust(value: np.ndarray | float) -> np.ndarray | float:
        return np.clip(value, 1e-30, 1e30)

    for index in range(n):
        one_based = index + 1
        if one_based == 1:
            end = int(layers[index])
            column = end - 1
            left = lr_in[0::2]
            right = lr_in[1::2]
            lr_reg[:, column] = adjust((left * right + 1.0) / (left + right))
            for level_one in range(end - 1, 0, -1):
                source_column = level_one
                destination_column = level_one - 1
                count = 1 << level_one
                left = lr_reg[0:count:2, source_column]
                right = lr_reg[1:count:2, source_column]
                lr_reg[: count // 2, destination_column] = adjust(
                    (left * right + 1.0) / (left + right)
                )
        elif one_based == n // 2 + 1:
            end = int(layers[index])
            column = end - 1
            left = lr_in[0::2]
            right = lr_in[1::2]
            used = mu_reg[:, -1].astype(np.int8)
            lr_reg[:, column] = adjust(np.power(left, 1 - 2 * used) * right)
            for level_one in range(end - 1, 0, -1):
                source_column = level_one
                destination_column = level_one - 1
                count = 1 << level_one
                left = lr_reg[0:count:2, source_column]
                right = lr_reg[1:count:2, source_column]
                lr_reg[: count // 2, destination_column] = adjust(
                    (left * right + 1.0) / (left + right)
                )
        elif one_based % 2 == 0:
            end = int(layers[index])
            destination_column = end - 1
            source_column = end
            left = float(lr_reg[0, source_column])
            right = float(lr_reg[1, source_column])
            used = int(mu_reg[0, 0])
            lr_reg[0, destination_column] = adjust(
                (left ** (1 - 2 * used)) * right
            )
        else:
            end = int(layers[index])
            destination_column = end - 1
            source_column = end
            count = 1 << end
            left = lr_reg[0:count:2, source_column]
            right = lr_reg[1:count:2, source_column]
            used = mu_reg[: count // 2, destination_column].astype(np.int8)
            lr_reg[: count // 2, destination_column] = adjust(
                np.power(left, 1 - 2 * used) * right
            )
            for level_one in range(end - 1, 0, -1):
                source_column_2 = level_one
                destination_column_2 = level_one - 1
                count_2 = 1 << level_one
                left_2 = lr_reg[0:count_2:2, source_column_2]
                right_2 = lr_reg[1:count_2:2, source_column_2]
                lr_reg[: count_2 // 2, destination_column_2] = adjust(
                    (left_2 * right_2 + 1.0) / (left_2 + right_2)
                )

        root_lr = float(np.clip(lr_reg[0, 0], 1e-30, 1e30))
        probability_one = 1.0 / (1.0 + root_lr)
        if forced_internal is not None:
            bit = int(forced_internal[index])
        elif freeze_flag[index]:
            bit = int(frozen_external[reverse[index]])
        else:
            bit = int(root_lr < 1.0)
        internal[index] = bit

        if score_selected and not freeze_flag[index]:
            probability = probability_one if bit else 1.0 - probability_one
            nll -= math.log2(max(probability, 1e-300))
            selected_bits.append(bit)
            selected_freq1.append(
                min(
                    65535,
                    max(1, int(math.floor(probability_one * 65536.0 + 0.5))),
                )
            )

        if one_based % 2 == 1:
            mu_reg[0, 0] = bit
        else:
            end = int(layers[one_based])
            temporary = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temporary[0] = bit
            for level_one in range(1, end):
                length = 1 << (level_one - 1)
                left = mu_reg[:length, level_one - 1]
                right = temporary[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2] = left ^ right
                merged[1::2] = right
                temporary[: 2 * length] = merged
            mu_reg[: 1 << max(end - 1, 0), end - 1] = temporary

    return SCResult(
        external_u=internal[reverse].copy(),
        internal_u=internal,
        selected_nll_bits=nll,
        selected_bits=np.asarray(selected_bits, dtype=np.uint8),
        selected_freq1_u16=np.asarray(selected_freq1, dtype=np.uint16),
    )


def arithmetic_encode_binary(
    bits: np.ndarray, frequencies_one: np.ndarray
) -> tuple[bytes, int]:
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    low = 0
    high = full - 1
    pending = 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        if pending:
            output.extend([1 - bit] * pending)
            pending = 0

    for bit_u8, frequency_one_u16 in zip(bits, frequencies_one, strict=True):
        bit = int(bit_u8)
        frequency_one = int(frequency_one_u16)
        frequency_zero = 65536 - frequency_one
        width = high - low + 1
        split = low + (width * frequency_zero // 65536) - 1
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
    pending += 1
    emit(0 if low < quarter else 1)
    packed = np.packbits(np.asarray(output, dtype=np.uint8), bitorder="big").tobytes()
    return packed, len(output)


def arithmetic_decode_binary(
    payload: bytes,
    logical_bits: int,
    frequencies_one: np.ndarray,
) -> np.ndarray:
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    packed_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    cursor = 0

    def read_bit() -> int:
        nonlocal cursor
        if cursor >= logical_bits:
            return 0
        result = int(packed_bits[cursor])
        cursor += 1
        return result

    low = 0
    high = full - 1
    code = 0
    for _ in range(32):
        code = ((code << 1) & (full - 1)) | read_bit()
    result = np.empty(frequencies_one.size, dtype=np.uint8)
    for index, frequency_one_u16 in enumerate(frequencies_one):
        frequency_one = int(frequency_one_u16)
        frequency_zero = 65536 - frequency_one
        width = high - low + 1
        split = low + (width * frequency_zero // 65536) - 1
        if code <= split:
            result[index] = 0
            high = split
        else:
            result[index] = 1
            low = split + 1
        while True:
            if high < half:
                pass
            elif low >= half:
                code -= half
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                code -= quarter
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
            code = ((code << 1) & (full - 1)) | read_bit()
    return result


def leaf_likelihood_ratios(
    y: np.ndarray,
    alphabet: np.ndarray,
    weights: np.ndarray,
    distortion: float,
    previous: np.ndarray,
    level: int,
) -> np.ndarray:
    density = (
        np.exp(-0.5 * ((y[:, None] - alphabet[None, :]) ** 2) / distortion)
        * weights[None, :]
    )
    lower_modulus = 1 << (level - 1)
    bit_value = 1 << (level - 1)
    result = np.empty(y.size, dtype=np.float64)
    for context in range(lower_modulus):
        positions = np.flatnonzero(previous == context)
        if positions.size == 0:
            continue
        zero = np.asarray(
            [
                j
                for j in range(alphabet.size)
                if j % lower_modulus == context and (j & bit_value) == 0
            ],
            dtype=np.int64,
        )
        one = np.asarray(
            [
                j
                for j in range(alphabet.size)
                if j % lower_modulus == context and (j & bit_value) != 0
            ],
            dtype=np.int64,
        )
        probability_zero = density[np.ix_(positions, zero)].sum(axis=1)
        probability_one = density[np.ix_(positions, one)].sum(axis=1)
        result[positions] = np.clip(
            probability_zero / np.maximum(probability_one, 1e-300), 1e-30, 1e30
        )
    return result


def leaf_prior_ratios(
    weights: np.ndarray, previous: np.ndarray, level: int
) -> np.ndarray:
    lower_modulus = 1 << (level - 1)
    bit_value = 1 << (level - 1)
    values = np.empty(lower_modulus, dtype=np.float64)
    for context in range(lower_modulus):
        zero = [
            j
            for j in range(weights.size)
            if j % lower_modulus == context and (j & bit_value) == 0
        ]
        one = [
            j
            for j in range(weights.size)
            if j % lower_modulus == context and (j & bit_value) != 0
        ]
        values[context] = weights[zero].sum() / max(weights[one].sum(), 1e-300)
    return np.clip(values[previous], 1e-30, 1e30)


def map_sc(
    y: np.ndarray,
    distortion: float,
    alphabet: np.ndarray,
    weights: np.ndarray,
    flags: list[np.ndarray],
    frozen_external: list[np.ndarray],
    reverse: np.ndarray,
    layers: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    previous = np.zeros(y.size, dtype=np.int16)
    internals: list[np.ndarray] = []
    for level in range(1, 7):
        likelihood = leaf_likelihood_ratios(
            y, alphabet, weights, distortion, previous, level
        )
        chosen = sc_encode_ratio(
            likelihood,
            flags[level - 1],
            frozen_external[level - 1],
            reverse,
            layers,
        )
        plane = polar_transform(chosen.external_u)
        previous += np.int16(1 << (level - 1)) * plane.astype(np.int16)
        internals.append(chosen.internal_u)
    return previous, internals


def causal_stream(
    indices: np.ndarray,
    weights: np.ndarray,
    flags: list[np.ndarray],
    frozen_external: list[np.ndarray],
    reverse: np.ndarray,
    layers: np.ndarray,
) -> dict[str, object]:
    previous = np.zeros(indices.size, dtype=np.int16)
    all_bits: list[np.ndarray] = []
    all_frequencies: list[np.ndarray] = []
    ideal_nll = 0.0
    internals: list[np.ndarray] = []
    for level in range(1, 7):
        plane = ((indices >> np.int16(level - 1)) & np.int16(1)).astype(np.uint8)
        external = polar_transform(plane)
        internal = external[reverse]
        expected_frozen = frozen_external[level - 1][reverse]
        if np.any(internal[flags[level - 1] == 1] != expected_frozen[flags[level - 1] == 1]):
            raise AssertionError(f"candidate violates frozen coset at level {level}")
        prior = leaf_prior_ratios(weights, previous, level)
        scored = sc_encode_ratio(
            prior,
            flags[level - 1],
            frozen_external[level - 1],
            reverse,
            layers,
            forced_internal=internal,
            score_selected=True,
        )
        ideal_nll += scored.selected_nll_bits
        all_bits.append(scored.selected_bits)
        all_frequencies.append(scored.selected_freq1_u16)
        previous += np.int16(1 << (level - 1)) * plane.astype(np.int16)
        internals.append(internal)
    if not np.array_equal(previous, indices):
        raise AssertionError("causal replay indices mismatch")
    bits = np.concatenate(all_bits)
    frequencies = np.concatenate(all_frequencies)
    payload, logical_bits = arithmetic_encode_binary(bits, frequencies)
    if not np.array_equal(
        arithmetic_decode_binary(payload, logical_bits, frequencies), bits
    ):
        raise AssertionError("native arithmetic round trip failed")
    return {
        "ideal_nll_bits": float(ideal_nll),
        "logical_bits": int(logical_bits),
        "payload": payload,
        "bits": bits,
        "frequencies": frequencies,
        "internals": internals,
    }


def enumerate_low_residues(
    n: int,
    constrained_levels: int,
    flags: list[np.ndarray],
    frozen_external: list[np.ndarray],
    reverse: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    counts = [int(np.count_nonzero(flags[level] == 0)) for level in range(constrained_levels)]
    information_bits = sum(counts)
    if information_bits > 20:
        raise ValueError(f"small-N enumeration would require 2^{information_bits} rows")
    rows = 1 << information_bits
    identifiers = np.arange(rows, dtype=np.uint64)
    residues = np.zeros((rows, n), dtype=np.uint8)
    internals: list[np.ndarray] = []
    offset = 0
    for level, count in enumerate(counts):
        flag = flags[level]
        internal = np.broadcast_to(
            frozen_external[level][reverse], (rows, n)
        ).copy()
        positions = np.flatnonzero(flag == 0)
        bit_numbers = np.arange(count, dtype=np.uint64) + np.uint64(offset)
        values = ((identifiers[:, None] >> bit_numbers[None, :]) & np.uint64(1)).astype(np.uint8)
        internal[:, positions] = values
        external = internal[:, reverse]
        plane = polar_transform(external)
        residues |= plane << np.uint8(level)
        internals.append(internal)
        offset += count
    return residues, internals, information_bits


def candidate_family(
    y: np.ndarray,
    alphabet: np.ndarray,
    weights: np.ndarray,
    residues: np.ndarray,
    constrained_levels: int,
    lambdas: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    modulus = 1 << constrained_levels
    coordinates = np.arange(y.size, dtype=np.int64)
    choices = [np.arange(residue, 64, modulus, dtype=np.int16) for residue in range(modulus)]
    error = np.square(y[:, None] - alphabet[None, :])
    conditional_cost = np.empty((modulus, 64), dtype=np.float64)
    conditional_cost.fill(np.inf)
    for residue, allowed in enumerate(choices):
        probability = weights[allowed] / weights[allowed].sum()
        conditional_cost[residue, allowed] = -np.log2(np.maximum(probability, 1e-300))

    nearest_error = np.empty((y.size, modulus), dtype=np.float64)
    nearest_index = np.empty((y.size, modulus), dtype=np.int16)
    for residue, allowed in enumerate(choices):
        local = error[:, allowed]
        selected = np.argmin(local, axis=1)
        nearest_error[:, residue] = local[coordinates, selected]
        nearest_index[:, residue] = allowed[selected]
    unrestricted_sse = float(
        np.min(np.sum(nearest_error[coordinates[None, :], residues], axis=1))
    )

    candidates: dict[bytes, np.ndarray] = {}
    for multiplier in lambdas:
        best = np.empty((y.size, modulus), dtype=np.int16)
        for residue, allowed in enumerate(choices):
            objective = error[:, allowed] + multiplier * conditional_cost[residue, allowed][None, :]
            best[:, residue] = allowed[np.argmin(objective, axis=1)]
        selected = best[coordinates[None, :], residues]
        for row in selected:
            key = row.astype(np.uint8, copy=False).tobytes()
            candidates.setdefault(key, row.astype(np.int16, copy=True))
    return list(candidates.values()), unrestricted_sse


def zero_extend_and_check(stream: dict[str, object], target_bits: int) -> str:
    native = int(stream["logical_bits"])
    if native > target_bits:
        raise ValueError("cannot zero-extend a longer stream")
    unpacked = np.unpackbits(
        np.frombuffer(stream["payload"], dtype=np.uint8), bitorder="big"
    )[:native]
    extended = np.concatenate(
        [unpacked, np.zeros(target_bits - native, dtype=np.uint8)]
    )
    payload = np.packbits(extended, bitorder="big").tobytes()
    decoded = arithmetic_decode_binary(
        payload, target_bits, stream["frequencies"]
    )
    if not np.array_equal(decoded, stream["bits"]):
        raise AssertionError("zero-extended arithmetic round trip failed")
    return hashlib.sha256(payload).hexdigest()


def one_trial(
    trial: int,
    seed: int,
    n: int,
    distortion: float,
    alphabet: np.ndarray,
    weights: np.ndarray,
    flags: list[np.ndarray],
    frozen_external: list[np.ndarray],
    reverse: np.ndarray,
    layers: np.ndarray,
    residues: np.ndarray,
    constrained_levels: int,
    lambdas: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + trial)
    y = rng.normal(size=n).astype(np.float64)
    y /= math.sqrt(float(np.mean(y * y)))
    energy = float(np.sum(y * y))
    control_indices, _ = map_sc(
        y, distortion, alphabet, weights, flags, frozen_external, reverse, layers
    )
    control_sse = float(np.sum(np.square(y - alphabet[control_indices])))
    control_stream = causal_stream(
        control_indices, weights, flags, frozen_external, reverse, layers
    )
    family, unrestricted_sse = candidate_family(
        y, alphabet, weights, residues, constrained_levels, lambdas
    )
    family.append(control_indices.copy())
    unique = {row.astype(np.uint8).tobytes(): row for row in family}
    ordered = sorted(
        unique.values(), key=lambda row: float(np.sum(np.square(y - alphabet[row])))
    )
    best_indices = control_indices
    best_stream = control_stream
    best_sse = control_sse
    scored = 0
    for indices in ordered:
        sse = float(np.sum(np.square(y - alphabet[indices])))
        if sse >= best_sse - 1e-15:
            break
        stream = causal_stream(
            indices, weights, flags, frozen_external, reverse, layers
        )
        scored += 1
        if int(stream["logical_bits"]) <= int(control_stream["logical_bits"]):
            best_indices = indices
            best_stream = stream
            best_sse = sse
            break
    equalized_sha = zero_extend_and_check(
        best_stream, int(control_stream["logical_bits"])
    )
    return {
        "trial": trial,
        "source_sha256": hashlib.sha256(y.astype("<f8").tobytes()).hexdigest(),
        "energy": energy,
        "control": {
            "sse": control_sse,
            "relative_mse": control_sse / energy,
            "ideal_nll_bits": control_stream["ideal_nll_bits"],
            "logical_bits": control_stream["logical_bits"],
            "indices_sha256": hashlib.sha256(control_indices.astype(np.uint8).tobytes()).hexdigest(),
        },
        "unrestricted_exact_nearest": {
            "sse": unrestricted_sse,
            "relative_mse": unrestricted_sse / energy,
            "optimistic_sse_reduction": (control_sse - unrestricted_sse) / control_sse,
            "rate_ignored": True,
        },
        "rate_matched_family_winner": {
            "sse": best_sse,
            "relative_mse": best_sse / energy,
            "relative_sse_reduction": (control_sse - best_sse) / control_sse,
            "native_ideal_nll_bits": best_stream["ideal_nll_bits"],
            "native_logical_bits": best_stream["logical_bits"],
            "equalized_logical_bits": control_stream["logical_bits"],
            "equalized_payload_sha256": equalized_sha,
            "indices_sha256": hashlib.sha256(best_indices.astype(np.uint8).tobytes()).hexdigest(),
            "selector_bits": 0,
            "arithmetic_roundtrip_after_zero_extension": True,
        },
        "family_unique_codewords": len(unique),
        "better_codewords_exactly_rate_scored": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--distortion", type=float, default=0.03125)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0x504F4C41524D4150)
    parser.add_argument("--lambda-points", type=int, default=65)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n != 16:
        raise ValueError("the audited default geometry currently requires N=16")
    if not (0.0 < args.distortion < 1.0):
        raise ValueError("distortion must be in (0,1)")
    if args.trials < 1 or args.lambda_points < 3:
        raise ValueError("positive trials and at least three lambda points required")

    root = Path(__file__).resolve().parents[3]
    upstream_encoder = root / "agent_polaris_qwen_rht_encoder.py"
    upstream_bec = root / "bg_codec_bec_encoder.py"
    actual_encoder_sha = sha256_file(upstream_encoder)
    actual_bec_sha = sha256_file(upstream_bec)
    if actual_encoder_sha != UPSTREAM_ENCODER_SHA256:
        raise RuntimeError(f"upstream encoder hash changed: {actual_encoder_sha}")
    if actual_bec_sha != UPSTREAM_BEC_SHA256:
        raise RuntimeError(f"upstream BEC hash changed: {actual_bec_sha}")

    started = time.perf_counter()
    n = args.n
    distortion = args.distortion
    eta = 0.25
    sigma_reconstruction = math.sqrt(1.0 - distortion)
    tilde_sigma = sigma_reconstruction * math.sqrt(distortion)
    capacities = [
        periodic_binary_capacity(tilde_sigma / eta / (1 << level))
        for level in range(6)
    ]
    flags = bec_flags(n, capacities)
    constrained_levels = next(
        (level for level in range(6) if all(np.count_nonzero(row) == 0 for row in flags[level:])),
        6,
    )
    if constrained_levels != 2:
        raise RuntimeError(
            f"expected exactly two constrained low levels, got {constrained_levels}"
        )
    reverse = bit_reverse_indices(n)
    layers = sc_layers(n)
    frozen_external = [
        np.random.default_rng(args.seed + 1_000_003 * level).integers(
            0, 2, size=n, dtype=np.uint8
        )
        for level in range(1, 7)
    ]
    residues, _, enumerated_information_bits = enumerate_low_residues(
        n, constrained_levels, flags, frozen_external, reverse
    )
    alphabet = eta * np.arange(-31, 33, dtype=np.float64)
    weights = np.exp(-0.5 * np.square(alphabet / sigma_reconstruction))
    gaussian_lagrange = 2.0 * math.log(2.0) * distortion
    lambdas = np.concatenate(
        [
            np.asarray([0.0]),
            gaussian_lagrange
            * np.exp2(np.linspace(-5.0, 5.0, args.lambda_points - 1)),
        ]
    )
    trials = [
        one_trial(
            trial,
            args.seed,
            n,
            distortion,
            alphabet,
            weights,
            flags,
            frozen_external,
            reverse,
            layers,
            residues,
            constrained_levels,
            lambdas,
        )
        for trial in range(args.trials)
    ]
    control_sse = sum(float(row["control"]["sse"]) for row in trials)
    winner_sse = sum(
        float(row["rate_matched_family_winner"]["sse"]) for row in trials
    )
    unrestricted_sse = sum(
        float(row["unrestricted_exact_nearest"]["sse"]) for row in trials
    )
    reductions = np.asarray(
        [
            float(row["rate_matched_family_winner"]["relative_sse_reduction"])
            for row in trials
        ],
        dtype=np.float64,
    )
    required = 0.16096404744368117
    result = {
        "schema": "strata_polar_map_sc_small_n_rate_matched_probe_v1",
        "scope": "procedural Gaussian CPU-only development diagnostic",
        "claim_boundary": (
            "exact enumeration of constrained low-plane assignments and exact "
            "unrestricted nearest reconstruction at N=16; the rate-matched "
            "upper-plane Lagrange family is broad but not exhaustive"
        ),
        "parameters": {
            "n": n,
            "distortion": distortion,
            "trials": args.trials,
            "seed": args.seed,
            "eta": eta,
            "capacities": capacities,
            "kept_internal_bits": [int(np.count_nonzero(row == 0)) for row in flags],
            "constrained_levels": constrained_levels,
            "enumerated_information_bits": enumerated_information_bits,
            "enumerated_low_codewords": int(residues.shape[0]),
            "lambda_points": int(lambdas.size),
            "lambda_min_positive": float(lambdas[1]),
            "lambda_max": float(lambdas[-1]),
        },
        "aggregate": {
            "control_sse": control_sse,
            "rate_matched_family_sse": winner_sse,
            "rate_matched_relative_sse_reduction": (control_sse - winner_sse) / control_sse,
            "rate_matched_trial_reduction_mean": float(np.mean(reductions)),
            "rate_matched_trial_reduction_median": float(np.median(reductions)),
            "rate_matched_trial_reduction_max": float(np.max(reductions)),
            "rate_matched_improved_trials": int(np.count_nonzero(reductions > 0.0)),
            "unrestricted_exact_nearest_sse": unrestricted_sse,
            "unrestricted_exact_nearest_reduction": (control_sse - unrestricted_sse) / control_sse,
            "required_equivalent_gain_bpw": required,
            "hard_kill_threshold_relative_mse_reduction": 0.20,
            "hard_kill": bool((control_sse - winner_sse) / control_sse < 0.20),
        },
        "bindings": {
            "upstream_encoder": str(upstream_encoder),
            "upstream_encoder_sha256": actual_encoder_sha,
            "upstream_bec": str(upstream_bec),
            "upstream_bec_sha256": actual_bec_sha,
            "script_sha256_before_result_write": sha256_file(Path(__file__)),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "processor": platform.processor(),
            "gpu_used": False,
            "seconds": time.perf_counter() - started,
        },
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
