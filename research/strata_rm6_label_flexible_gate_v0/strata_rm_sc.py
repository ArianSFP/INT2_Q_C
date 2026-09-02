#!/usr/bin/env python3
"""Exact single-path STRATA SC replay under local RM frozen sets."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from rm6_core import (ETA, ORDER_BANK, PLANES, RM6Error, assemble_indices,
                      bit_reverse_indices, frozen_external_from_seed,
                      polar_transform, require, rm_dimension, rm_freeze_flag)


def sc_layers(n: int) -> np.ndarray:
    depth = int(math.log2(n))
    require(1 << depth == n, "SC layer geometry")
    result = np.ones(n + 1, dtype=np.int32)
    result[0] = depth
    for one_based in range(2, n + 1):
        layer, cursor = 1, one_based
        while cursor % 2 == 1:
            layer += 1
            cursor = (cursor + 1) // 2
        result[one_based - 1] = layer
    return result


def profile_weights(profile_q: int) -> np.ndarray:
    require(0 <= profile_q <= 255, "profile q")
    rate = 1.75 + profile_q / 256.0
    distortion = math.exp2(-2.0 * rate)
    sigma_reconstruction = math.sqrt(1.0 - distortion)
    alphabet = ETA * np.arange(-31, 33, dtype=np.float64)
    return np.exp(-0.5 * (alphabet / sigma_reconstruction) ** 2)


def leaf_prior_ratios(weights: Any, previous: Any, level: int) -> np.ndarray:
    source_weights = np.asarray(weights, dtype=np.float64)
    prior_index = np.asarray(previous, dtype=np.int16)
    require(source_weights.shape == (64,) and prior_index.ndim == 1 and
            1 <= level <= PLANES and np.all(prior_index >= 0) and
            np.all(prior_index < (1 << (level - 1))), "leaf prior")
    lower_mod, bit_value = 1 << (level - 1), 1 << (level - 1)
    ratios = np.empty(lower_mod, dtype=np.float64)
    indices = np.arange(64)
    for context in range(lower_mod):
        matching = indices % lower_mod == context
        mass0 = source_weights[matching & ((indices & bit_value) == 0)].sum()
        mass1 = source_weights[matching & ((indices & bit_value) != 0)].sum()
        ratios[context] = mass0 / max(float(mass1), 1e-300)
    return np.clip(ratios[prior_index], 1e-30, 1e30)


class PrescribedBits:
    def __init__(self, bits: Sequence[int]) -> None:
        self.bits = tuple(int(value) for value in bits)
        require(all(value in (0, 1) for value in self.bits), "prescribed bits")
        self.cursor = 0

    def __call__(self, frequency: int, phase: int) -> int:
        del frequency, phase
        require(self.cursor < len(self.bits), "prescribed bit underflow")
        value = self.bits[self.cursor]
        self.cursor += 1
        return value


class GreedyProbabilityBits:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def __call__(self, frequency: int, phase: int) -> int:
        del phase
        value = int(frequency >= 32768)
        self.bits.append(value)
        return value


class ArithmeticBinaryDecoder:
    def __init__(self, raw: bytes, logical_bits: int) -> None:
        require(0 <= logical_bits <= len(raw) * 8, "arithmetic window")
        self.raw, self.logical_bits, self.cursor = raw, logical_bits, 0
        self.full, self.half, self.quarter = 1 << 32, 1 << 31, 1 << 30
        self.three_quarters = 3 << 30
        self.low, self.high, self.code = 0, (1 << 32) - 1, 0
        for _ in range(32):
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()

    def _read(self) -> int:
        if self.cursor >= self.logical_bits:
            return 0
        position = self.cursor
        self.cursor += 1
        return (self.raw[position >> 3] >> (7 - (position & 7))) & 1

    def __call__(self, frequency: int, phase: int) -> int:
        del phase
        freq1 = min(65535, max(1, int(frequency)))
        freq0 = 65536 - freq1
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * 65536 - 1) // width
        split = self.low + width * freq0 // 65536 - 1
        if scaled < freq0:
            value, self.high = 0, split
        else:
            value, self.low = 1, split + 1
        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.code -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.low -= self.quarter
                self.high -= self.quarter
                self.code -= self.quarter
            else:
                break
            self.low = (self.low << 1) & (self.full - 1)
            self.high = ((self.high << 1) & (self.full - 1)) | 1
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()
        return value


def arithmetic_encode_binary(bits: Any, frequencies: Any) -> tuple[bytes, int]:
    source_bits = np.asarray(bits, dtype=np.uint8)
    source_freq = np.asarray(frequencies, dtype=np.uint16)
    require(source_bits.shape == source_freq.shape and source_bits.ndim == 1,
            "arithmetic encode geometry")
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
        require(bit in (0, 1) and 1 <= freq1 <= 65535, "arithmetic symbol")
        freq0, width = 65536 - freq1, high - low + 1
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
    packed = np.packbits(np.asarray(output, dtype=np.uint8), bitorder="big").tobytes()
    return packed, len(output)


def run_sc_level(leaf_lr: Any, freeze_flag: Any, frozen_external: Any,
                 decision: Callable[[int, int], int]) -> dict[str, Any]:
    """Authenticated dense SC schedule with injected selected-bit source."""

    lr_in = np.clip(np.asarray(leaf_lr, dtype=np.float64), 1e-30, 1e30)
    freeze = np.asarray(freeze_flag, dtype=np.uint8)
    frozen = np.asarray(frozen_external, dtype=np.uint8)
    n, depth = lr_in.size, int(math.log2(lr_in.size))
    require(1 << depth == n and freeze.shape == frozen.shape == (n,), "SC geometry")
    reverse, layers = bit_reverse_indices(n), sc_layers(n)
    lr_reg = np.ones((n // 2, depth), dtype=np.float64)
    mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)
    internal = np.zeros(n, dtype=np.uint8)
    selected_count = int(np.count_nonzero(freeze == 0))
    frequencies = np.empty(selected_count, dtype=np.uint16)
    selected = np.empty(selected_count, dtype=np.uint8)
    frequency_cursor = 0

    def bounded(value: Any) -> Any:
        return np.clip(value, 1e-30, 1e30)

    for i0 in range(n):
        one_based = i0 + 1
        if one_based == 1:
            end, left, right = int(layers[i0]), lr_in[0::2], lr_in[1::2]
            lr_reg[:, end - 1] = bounded((left * right + 1.0) / (left + right))
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left, right = lr_reg[0:count:2, layer], lr_reg[1:count:2, layer]
                lr_reg[:count // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right))
        elif one_based == n // 2 + 1:
            end, left, right = int(layers[i0]), lr_in[0::2], lr_in[1::2]
            used = mu_reg[:, -1].astype(np.int8)
            lr_reg[:, end - 1] = bounded(np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left, right = lr_reg[0:count:2, layer], lr_reg[1:count:2, layer]
                lr_reg[:count // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right))
        elif one_based % 2 == 0:
            end = int(layers[i0])
            left, right = float(lr_reg[0, end]), float(lr_reg[1, end])
            used = int(mu_reg[0, 0])
            lr_reg[0, end - 1] = bounded(left ** (1 - 2 * used) * right)
        else:
            end, count = int(layers[i0]), 1 << int(layers[i0])
            left, right = lr_reg[0:count:2, end], lr_reg[1:count:2, end]
            used = mu_reg[:count // 2, end - 1].astype(np.int8)
            lr_reg[:count // 2, end - 1] = bounded(
                np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                count2 = 1 << layer
                left, right = lr_reg[0:count2:2, layer], lr_reg[1:count2:2, layer]
                lr_reg[:count2 // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right))
        root_lr = float(np.clip(lr_reg[0, 0], 1e-30, 1e30))
        if freeze[i0]:
            value = int(frozen[reverse[i0]])
        else:
            p1 = 1.0 / (1.0 + root_lr)
            frequency = min(65535, max(1, int(math.floor(p1 * 65536.0 + 0.5))))
            frequencies[frequency_cursor] = frequency
            value = int(decision(frequency, i0))
            require(value in (0, 1), "decision value")
            selected[frequency_cursor] = value
            frequency_cursor += 1
        internal[i0] = value
        if one_based % 2 == 1:
            mu_reg[0, 0] = value
        else:
            end = int(layers[one_based])
            temp = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temp[0] = value
            for layer in range(1, end):
                length = 1 << (layer - 1)
                left, right = mu_reg[:length, layer - 1], temp[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2], merged[1::2] = left ^ right, right
                temp[:2 * length] = merged
            mu_reg[:1 << max(end - 1, 0), end - 1] = temp
    require(frequency_cursor == selected_count, "selected count")
    return {"output": polar_transform(internal[reverse]), "internal": internal,
            "frequencies": frequencies, "selected": selected}


def _frozen(n: int, sc_seed: int, level: int, coset_mode: str) -> np.ndarray:
    require(coset_mode in ("zero", "current_random"), "coset mode")
    if coset_mode == "zero":
        return np.zeros(n, dtype=np.uint8)
    return frozen_external_from_seed(n, sc_seed, level)


def replay_six_prescribed(bank_id: int, profile_q: int, sc_seed: int,
                          coset_mode: str, decisions: Sequence[Sequence[int]],
                          variables: int = 12) -> dict[str, Any]:
    require(bank_id in ORDER_BANK and len(decisions) == PLANES, "six replay bank")
    n, weights = 1 << variables, profile_weights(profile_q)
    previous = np.zeros(n, dtype=np.int16)
    rows, all_selected, all_frequency = [], [], []
    for level0, order in enumerate(ORDER_BANK[bank_id]):
        expected = rm_dimension(order, variables)
        require(len(decisions[level0]) == expected, "level information dimension")
        policy = PrescribedBits(decisions[level0])
        row = run_sc_level(leaf_prior_ratios(weights, previous, level0 + 1),
                           rm_freeze_flag(variables, order),
                           _frozen(n, sc_seed, level0 + 1, coset_mode), policy)
        require(policy.cursor == expected, "prescribed coverage")
        previous += (1 << level0) * row["output"].astype(np.int16)
        rows.append(row)
        all_selected.append(row["selected"])
        all_frequency.append(row["frequencies"])
    selected, frequencies = np.concatenate(all_selected), np.concatenate(all_frequency)
    payload, logical_bits = arithmetic_encode_binary(selected, frequencies)
    return {"indices": previous.astype(np.uint8), "levels": rows,
            "selected": selected, "frequencies": frequencies,
            "payload": payload, "logical_bits": logical_bits}


def replay_six_greedy(bank_id: int, profile_q: int, sc_seed: int,
                      coset_mode: str, variables: int = 12) -> dict[str, Any]:
    n, weights = 1 << variables, profile_weights(profile_q)
    previous = np.zeros(n, dtype=np.int16)
    decisions, rows, all_selected, all_frequency = [], [], [], []
    for level0, order in enumerate(ORDER_BANK[bank_id]):
        policy = GreedyProbabilityBits()
        row = run_sc_level(leaf_prior_ratios(weights, previous, level0 + 1),
                           rm_freeze_flag(variables, order),
                           _frozen(n, sc_seed, level0 + 1, coset_mode), policy)
        previous += (1 << level0) * row["output"].astype(np.int16)
        decisions.append(tuple(policy.bits))
        rows.append(row)
        all_selected.append(row["selected"])
        all_frequency.append(row["frequencies"])
    selected, frequencies = np.concatenate(all_selected), np.concatenate(all_frequency)
    payload, logical_bits = arithmetic_encode_binary(selected, frequencies)
    return {"indices": previous.astype(np.uint8), "levels": rows,
            "decisions": decisions, "selected": selected, "frequencies": frequencies,
            "payload": payload, "logical_bits": logical_bits}


def decode_six_payload(bank_id: int, profile_q: int, sc_seed: int,
                       coset_mode: str, payload: bytes, logical_bits: int,
                       variables: int = 12) -> dict[str, Any]:
    n, weights = 1 << variables, profile_weights(profile_q)
    previous = np.zeros(n, dtype=np.int16)
    arithmetic = ArithmeticBinaryDecoder(payload, logical_bits)
    rows, all_selected, all_frequency = [], [], []
    for level0, order in enumerate(ORDER_BANK[bank_id]):
        row = run_sc_level(leaf_prior_ratios(weights, previous, level0 + 1),
                           rm_freeze_flag(variables, order),
                           _frozen(n, sc_seed, level0 + 1, coset_mode), arithmetic)
        previous += (1 << level0) * row["output"].astype(np.int16)
        rows.append(row)
        all_selected.append(row["selected"])
        all_frequency.append(row["frequencies"])
    selected, frequencies = np.concatenate(all_selected), np.concatenate(all_frequency)
    canonical, bits = arithmetic_encode_binary(selected, frequencies)
    require(bits == logical_bits and canonical == payload, "canonical arithmetic replay")
    return {"indices": previous.astype(np.uint8), "levels": rows,
            "selected": selected, "frequencies": frequencies,
            "logical_bits": bits, "payload": canonical}


def rm_ordered_positions(n: int, selected: int) -> np.ndarray:
    """Cheap global STRATA swap: RM ordering, often not an exact RM code."""

    variables = int(math.log2(n))
    require(1 << variables == n and 0 <= selected <= n, "global RM ordering")
    order = sorted(range(n), key=lambda index: (-index.bit_count(), index))
    return np.asarray(order[:selected], dtype=np.int64)


def classify_selected_dimension(variables: int, selected: int) -> dict[str, Any]:
    exact_order = next((order for order in range(variables + 1)
                        if rm_dimension(order, variables) == selected), None)
    return {"variables": variables, "selected": selected,
            "exact_rm": exact_order is not None, "exact_rm_order": exact_order,
            "name": (f"RM({exact_order},{variables})" if exact_order is not None
                     else "RM-ordered truncated polar set")}
