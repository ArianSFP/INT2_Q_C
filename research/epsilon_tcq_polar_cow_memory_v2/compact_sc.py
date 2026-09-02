#!/usr/bin/env python3
"""Dense-reference and exact ragged STRATA SC level implementations."""

from __future__ import annotations

import math
from typing import Any, Sequence


class SCError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SCError(message)


def bit_reverse_indices(np: Any, n: int) -> Any:
    depth = int(math.log2(n))
    require(1 << depth == n, "bit-reverse power of two")
    source = np.arange(n, dtype=np.uint32)
    result = np.zeros(n, dtype=np.uint32)
    for _ in range(depth):
        result = (result << np.uint32(1)) | (source & np.uint32(1))
        source >>= np.uint32(1)
    return result.astype(np.int64)


def sc_layers(np: Any, n: int) -> Any:
    depth = int(math.log2(n))
    result = np.ones(n + 1, dtype=np.int32)
    result[0] = depth
    for one_based in range(2, n + 1):
        layer, cursor = 1, one_based
        while cursor % 2 == 1:
            layer += 1
            cursor = (cursor + 1) // 2
        result[one_based - 1] = layer
    return result


def polar_transform(np: Any, bits: Any) -> Any:
    result = np.asarray(bits, dtype=np.uint8).copy()
    stride = 1
    while stride < result.size:
        rows = result.reshape(-1, 2 * stride)
        rows[:, :stride] ^= rows[:, stride:]
        stride *= 2
    return result


class PrescribedArithmetic:
    def __init__(self, decisions: Sequence[int]) -> None:
        self.decisions = tuple(int(value) for value in decisions)
        require(all(value in (0, 1) for value in self.decisions),
                "prescribed binary decisions")
        self.cursor = 0
        self.frequencies: list[int] = []

    def decode(self, frequency: int) -> int:
        require(self.cursor < len(self.decisions) and 1 <= int(frequency) <= 65535,
                "prescribed arithmetic event")
        self.frequencies.append(int(frequency))
        value = self.decisions[self.cursor]
        self.cursor += 1
        return value


def _common_inputs(np: Any, leaf_lr: Any, freeze_flag: Any,
                   frozen_external: Any) -> tuple[int, int, Any, Any, Any, Any]:
    leaf = np.clip(np.asarray(leaf_lr, dtype=np.float64), 1e-30, 1e30)
    n = int(leaf.size)
    depth = int(math.log2(n))
    require(1 << depth == n and n >= 8, "SC block geometry")
    freeze = np.asarray(freeze_flag, dtype=np.uint8)
    frozen = np.asarray(frozen_external, dtype=np.uint8)
    require(freeze.shape == frozen.shape == leaf.shape and
            not np.any(freeze > 1) and not np.any(frozen > 1), "SC flag geometry")
    reverse = bit_reverse_indices(np, n)
    layers = sc_layers(np, n)
    return n, depth, leaf, freeze, frozen, reverse, layers


def dense_decode_level(np: Any, leaf_lr: Any, freeze_flag: Any,
                       frozen_external: Any, decisions: Sequence[int]) -> dict[str, Any]:
    """Literal dense state layout matching the authenticated decoder."""

    n, depth, lr_in, freeze, frozen, reverse, layers = _common_inputs(
        np, leaf_lr, freeze_flag, frozen_external)
    arithmetic = PrescribedArithmetic(decisions)
    lr_reg = np.ones((n // 2, depth), dtype=np.float64)
    mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)
    internal = np.zeros(n, dtype=np.uint8)
    selected_count = int(np.count_nonzero(freeze == 0))
    frequencies = np.empty(selected_count, dtype=np.uint16)
    selected = np.empty(selected_count, dtype=np.uint8)
    cursor = 0

    def bounded(value):
        return np.clip(value, 1e-30, 1e30)

    for i0 in range(n):
        one_based = i0 + 1
        if one_based == 1:
            end = int(layers[i0])
            col = end - 1
            left, right = lr_in[0::2], lr_in[1::2]
            lr_reg[:, col] = bounded((left * right + 1.0) / (left + right))
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left = lr_reg[0:count:2, layer]
                right = lr_reg[1:count:2, layer]
                lr_reg[:count // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right))
        elif one_based == n // 2 + 1:
            end = int(layers[i0])
            col = end - 1
            left, right = lr_in[0::2], lr_in[1::2]
            used = mu_reg[:, -1].astype(np.int8)
            lr_reg[:, col] = bounded(np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left = lr_reg[0:count:2, layer]
                right = lr_reg[1:count:2, layer]
                lr_reg[:count // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right))
        elif one_based % 2 == 0:
            end = int(layers[i0])
            left, right = float(lr_reg[0, end]), float(lr_reg[1, end])
            used = int(mu_reg[0, 0])
            lr_reg[0, end - 1] = bounded(left ** (1 - 2 * used) * right)
        else:
            end = int(layers[i0])
            count = 1 << end
            left = lr_reg[0:count:2, end]
            right = lr_reg[1:count:2, end]
            used = mu_reg[:count // 2, end - 1].astype(np.int8)
            lr_reg[:count // 2, end - 1] = bounded(
                np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                count2 = 1 << layer
                left = lr_reg[0:count2:2, layer]
                right = lr_reg[1:count2:2, layer]
                lr_reg[:count2 // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right))
        root_lr = float(np.clip(lr_reg[0, 0], 1e-30, 1e30))
        if freeze[i0]:
            value = int(frozen[reverse[i0]])
        else:
            probability = 1.0 / (1.0 + root_lr)
            frequency = min(65535, max(1, int(math.floor(
                probability * 65536.0 + 0.5))))
            frequencies[cursor] = frequency
            value = arithmetic.decode(frequency)
            selected[cursor] = value
            cursor += 1
        internal[i0] = value
        if one_based % 2 == 1:
            mu_reg[0, 0] = value
        else:
            end = int(layers[one_based])
            temp = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temp[0] = value
            for layer in range(1, end):
                length = 1 << (layer - 1)
                left = mu_reg[:length, layer - 1]
                right = temp[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2] = left ^ right
                merged[1::2] = right
                temp[:2 * length] = merged
            mu_reg[:1 << max(end - 1, 0), end - 1] = temp
    require(cursor == selected_count and arithmetic.cursor == len(decisions),
            "dense selected event coverage")
    return {"output": polar_transform(np, internal[reverse]),
            "frequencies": frequencies, "selected": selected,
            "internal": internal, "lr_bytes": int(lr_reg.nbytes),
            "mu_bytes": int(mu_reg.nbytes)}


def ragged_decode_level(np: Any, leaf_lr: Any, freeze_flag: Any,
                        frozen_external: Any, decisions: Sequence[int]) -> dict[str, Any]:
    """Exact schedule with only N-1 active cells in each state bank."""

    n, depth, lr_in, freeze, frozen, reverse, layers = _common_inputs(
        np, leaf_lr, freeze_flag, frozen_external)
    arithmetic = PrescribedArithmetic(decisions)
    lr_flat = np.ones(n - 1, dtype=np.float64)
    mu_flat = np.zeros(n - 1, dtype=np.uint8)
    internal = np.zeros(n, dtype=np.uint8)
    selected_count = int(np.count_nonzero(freeze == 0))
    frequencies = np.empty(selected_count, dtype=np.uint16)
    selected = np.empty(selected_count, dtype=np.uint8)
    cursor = 0

    def lr_col(column):
        begin = (1 << column) - 1
        return lr_flat[begin:begin + (1 << column)]

    def mu_col(column):
        begin = (1 << column) - 1
        return mu_flat[begin:begin + (1 << column)]

    def bounded(value):
        return np.clip(value, 1e-30, 1e30)

    for i0 in range(n):
        one_based = i0 + 1
        if one_based == 1:
            end = int(layers[i0])
            left, right = lr_in[0::2], lr_in[1::2]
            lr_col(end - 1)[:] = bounded((left * right + 1.0) / (left + right))
            for layer in range(end - 1, 0, -1):
                source = lr_col(layer)
                lr_col(layer - 1)[:] = bounded(
                    (source[0::2] * source[1::2] + 1.0) /
                    (source[0::2] + source[1::2]))
        elif one_based == n // 2 + 1:
            end = int(layers[i0])
            left, right = lr_in[0::2], lr_in[1::2]
            used = mu_col(depth - 1).astype(np.int8)
            lr_col(end - 1)[:] = bounded(np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                source = lr_col(layer)
                lr_col(layer - 1)[:] = bounded(
                    (source[0::2] * source[1::2] + 1.0) /
                    (source[0::2] + source[1::2]))
        elif one_based % 2 == 0:
            end = int(layers[i0])
            source = lr_col(end)
            left, right = float(source[0]), float(source[1])
            used = int(mu_col(0)[0])
            lr_col(end - 1)[0] = bounded(left ** (1 - 2 * used) * right)
        else:
            end = int(layers[i0])
            source = lr_col(end)
            used = mu_col(end - 1).astype(np.int8)
            lr_col(end - 1)[:] = bounded(
                np.power(source[0::2], 1 - 2 * used) * source[1::2])
            for layer in range(end - 1, 0, -1):
                source2 = lr_col(layer)
                lr_col(layer - 1)[:] = bounded(
                    (source2[0::2] * source2[1::2] + 1.0) /
                    (source2[0::2] + source2[1::2]))
        root_lr = float(np.clip(lr_col(0)[0], 1e-30, 1e30))
        if freeze[i0]:
            value = int(frozen[reverse[i0]])
        else:
            probability = 1.0 / (1.0 + root_lr)
            frequency = min(65535, max(1, int(math.floor(
                probability * 65536.0 + 0.5))))
            frequencies[cursor] = frequency
            value = arithmetic.decode(frequency)
            selected[cursor] = value
            cursor += 1
        internal[i0] = value
        if one_based % 2 == 1:
            mu_col(0)[0] = value
        else:
            end = int(layers[one_based])
            temp = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temp[0] = value
            for layer in range(1, end):
                length = 1 << (layer - 1)
                left = mu_col(layer - 1)[:length]
                right = temp[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2] = left ^ right
                merged[1::2] = right
                temp[:2 * length] = merged
            mu_col(end - 1)[:1 << max(end - 1, 0)] = temp
    require(cursor == selected_count and arithmetic.cursor == len(decisions),
            "ragged selected event coverage")
    return {"output": polar_transform(np, internal[reverse]),
            "frequencies": frequencies, "selected": selected,
            "internal": internal, "lr_bytes": int(lr_flat.nbytes),
            "mu_bytes": int(mu_flat.nbytes)}


def leaf_prior_ratios(np: Any, weights: Any, previous: Any, level: int) -> Any:
    """Literal decoder-visible 64-symbol conditional prior."""

    source_weights = np.asarray(weights, dtype=np.float64)
    prior_index = np.asarray(previous, dtype=np.int16)
    require(source_weights.shape == (64,) and prior_index.ndim == 1 and
            1 <= level <= 6 and np.all(prior_index >= 0) and
            np.all(prior_index < (1 << (level - 1))), "six-level prior geometry")
    lower_mod = 1 << (level - 1)
    bit_value = 1 << (level - 1)
    ratios = np.empty(lower_mod, dtype=np.float64)
    indices = np.arange(64)
    for context in range(lower_mod):
        matching = indices % lower_mod == context
        mass0 = source_weights[matching & ((indices & bit_value) == 0)].sum()
        mass1 = source_weights[matching & ((indices & bit_value) != 0)].sum()
        ratios[context] = mass0 / max(float(mass1), 1e-300)
    return np.clip(ratios[prior_index], 1e-30, 1e30)


def arithmetic_encode_binary(np: Any, bits: Any, frequencies: Any) -> tuple[bytes, int]:
    """Canonical authenticated 32-bit/uint16 binary arithmetic encoder."""

    source_bits = np.asarray(bits, dtype=np.uint8)
    source_freq = np.asarray(frequencies, dtype=np.uint16)
    require(source_bits.shape == source_freq.shape and source_bits.ndim == 1,
            "arithmetic replay geometry")
    full, half, quarter, three_quarters = 1 << 32, 1 << 31, 1 << 30, 3 << 30
    low, high, pending = 0, full - 1, 0
    output: list[int] = []

    def emit(value: int) -> None:
        nonlocal pending
        output.append(value)
        if pending:
            output.extend([1 - value] * pending)
            pending = 0

    for bit_u8, freq_u16 in zip(source_bits, source_freq, strict=True):
        bit, freq1 = int(bit_u8), int(freq_u16)
        require(bit in (0, 1) and 1 <= freq1 <= 65535, "arithmetic replay symbol")
        freq0 = 65536 - freq1
        width = high - low + 1
        split = low + width * freq0 // 65536 - 1
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
    logical_bits = len(output)
    packed = np.packbits(np.asarray(output, dtype=np.uint8), bitorder="big").tobytes()
    return packed, logical_bits


def replay_six_levels(np: Any, weights: Any, freeze_flags: Sequence[Any],
                      frozen_vectors: Sequence[Any], decisions: Sequence[Sequence[int]],
                      *, layout: str) -> dict[str, Any]:
    """Source-free six-pass harness preserving current 64-index semantics."""

    require(layout in ("dense", "ragged") and len(freeze_flags) == 6 and
            len(frozen_vectors) == 6 and len(decisions) == 6, "six-level replay")
    n = int(np.asarray(freeze_flags[0]).size)
    previous = np.zeros(n, dtype=np.int16)
    level_rows = []
    all_frequency, all_selected = [], []
    decoder = dense_decode_level if layout == "dense" else ragged_decode_level
    for level0 in range(6):
        prior = leaf_prior_ratios(np, weights, previous, level0 + 1)
        row = decoder(np, prior, freeze_flags[level0], frozen_vectors[level0],
                      decisions[level0])
        previous += (1 << level0) * row["output"].astype(np.int16)
        all_frequency.append(row["frequencies"])
        all_selected.append(row["selected"])
        level_rows.append(row)
    frequencies = np.concatenate(all_frequency)
    selected = np.concatenate(all_selected)
    payload, logical_bits = arithmetic_encode_binary(np, selected, frequencies)
    return {"indices": previous, "levels": level_rows, "frequencies": frequencies,
            "selected": selected, "payload": payload, "logical_bits": logical_bits}
