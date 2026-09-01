#!/usr/bin/env python3
"""Source-free nonlocal WFA/HMM prototype.

The model is a tied, nonnegative, unifilar weighted finite automaton.  Its
serialized sparse form expands exactly to symbol-conditioned chi-by-chi
matrices.  It is deliberately not a suffix context: a hidden state may retain
parities across an arbitrarily long random gap.

This module has no model-checkpoint input path and no GPU dependency.  A future
GPU evaluator must use CuPy; this synthetic reference remains CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCHEMA = "nonlocal-wfa-global-state-synthetic-v0"
ALPHABET = 2
BLOCK_LENGTH = 32
BODY_SYMBOLS = 26
CHECKS = 6
CONTEXTS = 2 * CHECKS
Q16_TOTAL = 65535
HEADER_BYTES = 256
PAGE_BYTES = 4096
LOCAL_FRAME_HEADER_BYTES = 256
MAX_SUFFIX_DEPTH = 16
STANDALONE_REQUIRED_BPW = 0.1528899669629145
TARGET_SYNTHETIC_GROSS_BPS = CHECKS / BLOCK_LENGTH
MAGIC = b"NLWFA0\0\0"


class ContractError(RuntimeError):
    """A source-free prototype invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def page_ceil(byte_count: int) -> int:
    require(byte_count >= 0, "nonnegative byte count")
    return ((byte_count + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES


def worst_unaligned_page_union(byte_count: int) -> int:
    """Maximum page union for a nonempty contiguous range of known length."""
    require(byte_count > 0, "positive byte range")
    pages = (byte_count + PAGE_BYTES - 2) // PAGE_BYTES + 1
    return pages * PAGE_BYTES


def public_context(position_in_block: int) -> int:
    """Shape/coordinate-only context, never a model/layer/expert identity."""
    require(0 <= position_in_block < BLOCK_LENGTH, "block position")
    if position_in_block < BODY_SYMBOLS:
        return position_in_block % CHECKS
    return CHECKS + (position_in_block - BODY_SYMBOLS)


def generate_syndrome_blocks(
    block_count: int, seed: int, constrained: bool
) -> np.ndarray:
    """Generate matched-marginal binary blocks.

    The first 26 symbols are iid Bernoulli(1/2).  In the structured source,
    each of the final six symbols is the parity of a disjoint residue class of
    body coordinates.  In the matched control, the final six are independent
    Bernoulli(1/2).  Every site marginal is therefore identical.
    """
    require(block_count > 0, "positive block count")
    rng = np.random.default_rng(seed)
    body = rng.integers(0, 2, size=(block_count, BODY_SYMBOLS), dtype=np.uint8)
    if constrained:
        checks = np.empty((block_count, CHECKS), dtype=np.uint8)
        for group in range(CHECKS):
            checks[:, group] = np.bitwise_xor.reduce(body[:, group::CHECKS], axis=1)
    else:
        checks = rng.integers(0, 2, size=(block_count, CHECKS), dtype=np.uint8)
    return np.concatenate((body, checks), axis=1)


def flatten_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(blocks.ndim == 2 and blocks.shape[1] == BLOCK_LENGTH, "block geometry")
    symbols = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1)
    contexts = np.tile(
        np.asarray([public_context(t) for t in range(BLOCK_LENGTH)], dtype=np.uint8),
        blocks.shape[0],
    )
    return symbols, contexts


@dataclass(frozen=True)
class SparseUnifilarWFA:
    """A sparse serialization of nonnegative A[c,y] in R_+^(chi x chi)."""

    syndrome_bits: int
    successors: np.ndarray  # uint8 [context, symbol, state]
    freq1: np.ndarray  # uint16 [context, state], with 1 <= f < Q
    initial_state: int = 0

    @property
    def chi(self) -> int:
        return 1 << self.syndrome_bits

    def validate(self) -> None:
        require(0 <= self.syndrome_bits <= CHECKS, "syndrome bits")
        require(self.chi <= 255, "u8 successor format")
        require(self.successors.shape == (CONTEXTS, ALPHABET, self.chi), "successor shape")
        require(self.successors.dtype == np.uint8, "successor dtype")
        require(np.all(self.successors < self.chi), "successor range")
        require(self.freq1.shape == (CONTEXTS, self.chi), "frequency shape")
        require(self.freq1.dtype == np.uint16, "frequency dtype")
        require(np.all(self.freq1 >= 1) and np.all(self.freq1 < Q16_TOTAL), "frequency range")
        require(0 <= self.initial_state < self.chi, "initial state")
        dense = self.expand_dense()
        row_sums = dense.sum(axis=(1, 3), dtype=np.uint64)
        require(np.all(row_sums == Q16_TOTAL), "exact normalized dense rows")

    def expand_dense(self) -> np.ndarray:
        """Return integer A[c,y,i,j], whose (y,j) row sum is exactly Q."""
        dense = np.zeros((CONTEXTS, ALPHABET, self.chi, self.chi), dtype=np.uint16)
        for context in range(CONTEXTS):
            for state in range(self.chi):
                f1 = int(self.freq1[context, state])
                dense[context, 0, state, int(self.successors[context, 0, state])] = Q16_TOTAL - f1
                dense[context, 1, state, int(self.successors[context, 1, state])] = f1
        return dense

    def predictive_fraction(
        self, alpha: Sequence[Fraction], context: int, symbol: int
    ) -> Fraction:
        """Exact causal p(symbol|prefix,context) for an arbitrary state belief."""
        require(len(alpha) == self.chi, "alpha geometry")
        require(sum(alpha, Fraction(0, 1)) == 1, "normalized alpha")
        require(0 <= context < CONTEXTS and symbol in (0, 1), "context/symbol")
        dense = self.expand_dense()
        numerator = Fraction(0, 1)
        denominator = Fraction(0, 1)
        for state, weight in enumerate(alpha):
            for out_symbol in (0, 1):
                row = int(dense[context, out_symbol, state, :].sum(dtype=np.uint64))
                term = weight * row
                denominator += term
                if out_symbol == symbol:
                    numerator += term
        require(denominator == Q16_TOTAL, "exact predictive denominator")
        return numerator / denominator

    def update_fraction(
        self, alpha: Sequence[Fraction], context: int, symbol: int
    ) -> tuple[Fraction, ...]:
        """Exact Bayes/filter update under the serialized integer matrices."""
        dense = self.expand_dense()
        unnormalized = []
        for next_state in range(self.chi):
            value = sum(
                alpha[state] * int(dense[context, symbol, state, next_state])
                for state in range(self.chi)
            )
            unnormalized.append(value)
        normalizer = sum(unnormalized, Fraction(0, 1))
        require(normalizer > 0, "positive observed-symbol probability")
        return tuple(value / normalizer for value in unnormalized)

    def serialize(self) -> bytes:
        self.validate()
        meta = struct.pack(
            "<8sHHHHHHI",
            MAGIC,
            0,
            self.syndrome_bits,
            self.chi,
            CONTEXTS,
            ALPHABET,
            self.initial_state,
            Q16_TOTAL,
        )
        require(len(meta) <= HEADER_BYTES, "header fit")
        header = meta + bytes(HEADER_BYTES - len(meta))
        pi = np.zeros(self.chi, dtype="<u2")
        pi[self.initial_state] = Q16_TOTAL
        return (
            header
            + pi.tobytes(order="C")
            + self.successors.astype(np.uint8, copy=False).tobytes(order="C")
            + self.freq1.astype("<u2", copy=False).tobytes(order="C")
        )

    @classmethod
    def deserialize(cls, packet: bytes) -> "SparseUnifilarWFA":
        require(len(packet) >= HEADER_BYTES, "packet header")
        fields = struct.unpack("<8sHHHHHHI", packet[:24])
        magic, version, bits, chi, contexts, alphabet, initial, total = fields
        require(magic == MAGIC and version == 0, "packet magic/version")
        require(contexts == CONTEXTS and alphabet == ALPHABET and total == Q16_TOTAL, "packet constants")
        require(chi == 1 << bits and chi <= 255, "packet chi")
        offset = HEADER_BYTES
        pi_bytes = 2 * chi
        successor_bytes = CONTEXTS * ALPHABET * chi
        frequency_bytes = 2 * CONTEXTS * chi
        expected = offset + pi_bytes + successor_bytes + frequency_bytes
        require(len(packet) == expected, "packet length")
        pi = np.frombuffer(packet[offset : offset + pi_bytes], dtype="<u2").copy()
        require(int(pi.sum(dtype=np.uint64)) == Q16_TOTAL, "serialized pi normalization")
        require(int(pi[initial]) == Q16_TOTAL and np.count_nonzero(pi) == 1, "serialized initial state")
        offset += pi_bytes
        successors = np.frombuffer(
            packet[offset : offset + successor_bytes], dtype=np.uint8
        ).copy().reshape(CONTEXTS, ALPHABET, chi)
        offset += successor_bytes
        freq1 = np.frombuffer(packet[offset:], dtype="<u2").copy().reshape(CONTEXTS, chi)
        model = cls(int(bits), successors, freq1, int(initial))
        model.validate()
        return model


def candidate_successors(syndrome_bits: int) -> np.ndarray:
    """Universal parity-bank topology candidate, expanded as successor maps."""
    require(0 <= syndrome_bits <= CHECKS, "candidate syndrome bits")
    chi = 1 << syndrome_bits
    result = np.empty((CONTEXTS, ALPHABET, chi), dtype=np.uint8)
    for context in range(CONTEXTS):
        for symbol in (0, 1):
            for state in range(chi):
                next_state = state
                if context < syndrome_bits and symbol == 1:
                    next_state ^= 1 << context
                result[context, symbol, state] = next_state
    return result


def state_trace(blocks: np.ndarray, syndrome_bits: int) -> np.ndarray:
    """State before each observed symbol; reset is public at every block."""
    require(blocks.ndim == 2 and blocks.shape[1] == BLOCK_LENGTH, "trace geometry")
    successors = candidate_successors(syndrome_bits)
    trace = np.empty_like(blocks, dtype=np.uint8)
    for row in range(blocks.shape[0]):
        state = 0
        for position in range(BLOCK_LENGTH):
            trace[row, position] = state
            context = public_context(position)
            symbol = int(blocks[row, position])
            state = int(successors[context, symbol, state])
    return trace


def fit_candidate(blocks: np.ndarray, syndrome_bits: int) -> SparseUnifilarWFA:
    """Learn quantized transition weights for one selected global-state topology."""
    require(blocks.ndim == 2 and blocks.shape[1] == BLOCK_LENGTH, "fit geometry")
    chi = 1 << syndrome_bits
    states = state_trace(blocks, syndrome_bits)
    count0 = np.zeros((CONTEXTS, chi), dtype=np.int64)
    count1 = np.zeros((CONTEXTS, chi), dtype=np.int64)
    for position in range(BLOCK_LENGTH):
        context = public_context(position)
        position_states = states[:, position].astype(np.int64)
        position_symbols = blocks[:, position].astype(np.int64)
        keys = position_states * 2 + position_symbols
        counts = np.bincount(keys, minlength=2 * chi).reshape(chi, 2)
        count0[context, :] += counts[:, 0]
        count1[context, :] += counts[:, 1]
    # Jeffreys 1/2 smoothing, deterministically quantized.  Algebraic form
    # avoids floating point in the fitted packet.
    numerator = Q16_TOTAL * (2 * count1 + 1)
    denominator = 2 * (count0 + count1 + 1)
    freq1 = (numerator + denominator // 2) // denominator
    freq1 = np.clip(freq1, 1, Q16_TOTAL - 1).astype(np.uint16)
    model = SparseUnifilarWFA(
        syndrome_bits=syndrome_bits,
        successors=candidate_successors(syndrome_bits),
        freq1=freq1,
    )
    model.validate()
    return model


def logical_codelength_bits(model: SparseUnifilarWFA, blocks: np.ndarray) -> float:
    """Exact-model ideal arithmetic length, before finite termination."""
    states = state_trace(blocks, model.syndrome_bits)
    bits = 0.0
    for position in range(BLOCK_LENGTH):
        context = public_context(position)
        position_states = states[:, position].astype(np.int64)
        position_symbols = blocks[:, position].astype(np.int64)
        f1 = model.freq1[context, position_states].astype(np.int64)
        selected = np.where(position_symbols == 1, f1, Q16_TOTAL - f1)
        require(bool(np.all(selected > 0)), "positive selected frequencies")
        bits -= float(np.log2(selected.astype(np.float64) / Q16_TOTAL).sum(dtype=np.float64))
    return bits


def model_byte_ledger(model: SparseUnifilarWFA) -> dict[str, int | float]:
    model.validate()
    chi = model.chi
    sparse_successor_bytes = CONTEXTS * ALPHABET * chi
    sparse_frequency_bytes = 2 * CONTEXTS * chi
    initial_bytes = 2 * chi
    sparse_bytes = HEADER_BYTES + initial_bytes + sparse_successor_bytes + sparse_frequency_bytes
    dense_matrix_bytes = 2 * CONTEXTS * ALPHABET * chi * chi
    dense_bytes = HEADER_BYTES + initial_bytes + dense_matrix_bytes
    packet = model.serialize()
    require(len(packet) == sparse_bytes, "sparse model formula")
    return {
        "chi": chi,
        "contexts": CONTEXTS,
        "header_bytes": HEADER_BYTES,
        "initial_u16_bytes": initial_bytes,
        "sparse_successor_u8_bytes": sparse_successor_bytes,
        "sparse_frequency_u16_bytes": sparse_frequency_bytes,
        "sparse_physical_bytes": sparse_bytes,
        "sparse_page_bytes": page_ceil(sparse_bytes),
        "dense_equivalent_matrix_u16_bytes": dense_matrix_bytes,
        "dense_equivalent_physical_bytes": dense_bytes,
        "dense_equivalent_page_bytes": page_ceil(dense_bytes),
        "sparse_packet_sha256": sha256_bytes(packet),
    }


def select_model(train: np.ndarray, validation: np.ndarray, expert_count: int) -> tuple[SparseUnifilarWFA, list[dict[str, float | int]]]:
    """Select chi by held-out physical length, charging one shared model."""
    require(expert_count > 0, "positive expert count")
    rows: list[dict[str, float | int]] = []
    candidates: list[SparseUnifilarWFA] = []
    for bits in range(CHECKS + 1):
        model = fit_candidate(train, bits)
        logical = logical_codelength_bits(model, validation)
        model_bits_per_expert = 8.0 * page_ceil(len(model.serialize())) / expert_count
        termination_bits = 32.0
        physical_proxy = logical + model_bits_per_expert + termination_bits
        rows.append(
            {
                "syndrome_bits": bits,
                "chi": model.chi,
                "validation_symbols": int(validation.size),
                "logical_bits": logical,
                "logical_bps": logical / validation.size,
                "model_page_bits_per_expert": model_bits_per_expert,
                "termination_bits_per_expert": termination_bits,
                "physical_proxy_bits_per_expert": physical_proxy,
                "physical_proxy_bps": physical_proxy / validation.size,
            }
        )
        candidates.append(model)
    best = min(
        range(len(rows)),
        key=lambda i: (
            float(rows[i]["physical_proxy_bits_per_expert"]),
            int(rows[i]["chi"]),
        ),
    )
    return candidates[best], rows


def suffix_cross_entropy(
    train: np.ndarray, validation: np.ndarray, depth: int
) -> dict[str, float | int]:
    """Held-out KT-smoothed suffix model, reset at every public block boundary."""
    require(0 <= depth <= MAX_SUFFIX_DEPTH, "suffix depth")
    state_count = 1 << depth
    count0 = np.zeros((CONTEXTS, state_count), dtype=np.int64)
    count1 = np.zeros((CONTEXTS, state_count), dtype=np.int64)

    def visit(blocks: np.ndarray, accumulate: bool) -> float:
        total_bits = 0.0
        mask = state_count - 1
        for position in range(BLOCK_LENGTH):
            context = public_context(position)
            if depth == 0:
                states = np.zeros(blocks.shape[0], dtype=np.int64)
            else:
                states = np.zeros(blocks.shape[0], dtype=np.int64)
                start = max(0, position - depth)
                for previous in range(start, position):
                    states = ((states << 1) | blocks[:, previous].astype(np.int64)) & mask
            symbols = blocks[:, position].astype(np.int64)
            if accumulate:
                keys = states * 2 + symbols
                counts = np.bincount(keys, minlength=2 * state_count).reshape(state_count, 2)
                count0[context, :] += counts[:, 0]
                count1[context, :] += counts[:, 1]
            else:
                n0 = count0[context, states]
                n1 = count1[context, states]
                denominator = n0 + n1 + 1.0
                p1 = (n1 + 0.5) / denominator
                selected = np.where(symbols == 1, p1, 1.0 - p1)
                total_bits -= float(np.log2(selected).sum(dtype=np.float64))
        return total_bits

    visit(train, True)
    bits = visit(validation, False)
    raw_table_bytes = HEADER_BYTES + 2 * CONTEXTS * state_count
    return {
        "depth": depth,
        "states": state_count,
        "logical_bits": bits,
        "logical_bps": bits / validation.size,
        "raw_q16_table_bytes": raw_table_bytes,
        "page_table_bytes": page_ceil(raw_table_bytes),
    }


class ArithmeticEncoder:
    """Reference 32-bit binary arithmetic encoder."""

    FULL = 1 << 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE_QUARTER = 3 << 30

    def __init__(self) -> None:
        self.low = 0
        self.high = self.FULL - 1
        self.pending = 0
        self.bits: list[int] = []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def write(self, symbol: int, freq1: int) -> None:
        require(symbol in (0, 1), "arithmetic symbol")
        require(1 <= freq1 < Q16_TOTAL, "arithmetic frequency")
        f0 = Q16_TOTAL - freq1
        cumulative_low = 0 if symbol == 0 else f0
        cumulative_high = f0 if symbol == 0 else Q16_TOTAL
        width = self.high - self.low + 1
        self.high = self.low + (width * cumulative_high // Q16_TOTAL) - 1
        self.low = self.low + (width * cumulative_low // Q16_TOTAL)
        while True:
            if self.high < self.HALF:
                self._emit(0)
            elif self.low >= self.HALF:
                self._emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> tuple[bytes, int]:
        self.pending += 1
        self._emit(0 if self.low < self.QUARTER else 1)
        meaningful_bits = len(self.bits)
        while len(self.bits) % 8:
            self.bits.append(0)
        payload = bytearray(len(self.bits) // 8)
        for index, bit in enumerate(self.bits):
            payload[index // 8] |= bit << (7 - (index & 7))
        return bytes(payload), meaningful_bits


class ArithmeticDecoder:
    FULL = ArithmeticEncoder.FULL
    HALF = ArithmeticEncoder.HALF
    QUARTER = ArithmeticEncoder.QUARTER
    THREE_QUARTER = ArithmeticEncoder.THREE_QUARTER

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.bit_index = 0
        self.low = 0
        self.high = self.FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self.bit_index >= len(self.payload) * 8:
            self.bit_index += 1
            return 0
        value = (self.payload[self.bit_index // 8] >> (7 - (self.bit_index & 7))) & 1
        self.bit_index += 1
        return value

    def read(self, freq1: int) -> int:
        require(1 <= freq1 < Q16_TOTAL, "arithmetic decode frequency")
        f0 = Q16_TOTAL - freq1
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * Q16_TOTAL - 1) // width
        symbol = 0 if scaled < f0 else 1
        cumulative_low = 0 if symbol == 0 else f0
        cumulative_high = f0 if symbol == 0 else Q16_TOTAL
        self.high = self.low + (width * cumulative_high // Q16_TOTAL) - 1
        self.low = self.low + (width * cumulative_low // Q16_TOTAL)
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.code -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.code -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = ((self.code << 1) | self._read_bit()) & (self.FULL - 1)
        return symbol


def encode_blocks(model: SparseUnifilarWFA, blocks: np.ndarray) -> tuple[bytes, int]:
    model.validate()
    encoder = ArithmeticEncoder()
    for row in range(blocks.shape[0]):
        state = model.initial_state
        for position in range(BLOCK_LENGTH):
            context = public_context(position)
            symbol = int(blocks[row, position])
            frequency = int(model.freq1[context, state])
            encoder.write(symbol, frequency)
            state = int(model.successors[context, symbol, state])
    return encoder.finish()


def decode_blocks(
    model: SparseUnifilarWFA, payload: bytes, block_count: int
) -> np.ndarray:
    model.validate()
    decoder = ArithmeticDecoder(payload)
    blocks = np.empty((block_count, BLOCK_LENGTH), dtype=np.uint8)
    for row in range(block_count):
        state = model.initial_state
        for position in range(BLOCK_LENGTH):
            context = public_context(position)
            frequency = int(model.freq1[context, state])
            symbol = decoder.read(frequency)
            blocks[row, position] = symbol
            state = int(model.successors[context, symbol, state])
    return blocks


def physical_stream_ledger(
    model: SparseUnifilarWFA,
    payload_bytes: int,
    symbols_per_expert: int,
    expert_count: int,
) -> dict[str, float | int | bool]:
    require(payload_bytes > 0 and symbols_per_expert > 0 and expert_count > 0, "ledger inputs")
    model_pages = page_ceil(len(model.serialize()))
    local_bytes = LOCAL_FRAME_HEADER_BYTES + payload_bytes
    baseline_local_bytes = LOCAL_FRAME_HEADER_BYTES + (symbols_per_expert + 7) // 8
    local_cold = worst_unaligned_page_union(local_bytes)
    baseline_cold = worst_unaligned_page_union(baseline_local_bytes)
    cold = model_pages + local_cold
    aggregate_bytes = model_pages + expert_count * local_bytes
    baseline_aggregate = expert_count * baseline_local_bytes
    return {
        "expert_count": expert_count,
        "symbols_per_expert": symbols_per_expert,
        "global_model_page_bytes": model_pages,
        "local_frame_header_bytes": LOCAL_FRAME_HEADER_BYTES,
        "payload_bytes_per_expert": payload_bytes,
        "local_physical_bytes_per_expert": local_bytes,
        "aggregate_physical_bytes": aggregate_bytes,
        "baseline_raw_one_bit_aggregate_bytes": baseline_aggregate,
        "aggregate_saving_bps": 8.0 * (baseline_aggregate - aggregate_bytes) / (expert_count * symbols_per_expert),
        "worst_unaligned_local_page_union_bytes": local_cold,
        "worst_unaligned_baseline_page_union_bytes": baseline_cold,
        "synthetic_cold_read_bytes": cold,
        "synthetic_cold_read_amplification_vs_raw_one_bit_frame": cold / baseline_cold,
        "cold_below_two": cold / baseline_cold < 2.0,
        "scope": "synthetic binary-stream ledger only; not a Qwen/current-codec read claim",
    }


def capacity_sanity(block_length: int = 2048) -> dict[str, float | int | str]:
    required_bits = STANDALONE_REQUIRED_BPW * block_length
    return {
        "block_length": block_length,
        "standalone_required_bpw": STANDALONE_REQUIRED_BPW,
        "standalone_required_bits_per_block": required_bits,
        "single_parity_saving_bits_per_block": 1,
        "single_parity_saving_bpw": 1.0 / block_length,
        "chi64_nonnegative_cut_state_bits": math.log2(64),
        "chi64_born_rough_cut_bound_bits": 2.0 * math.log2(64),
        "fixture_constraints_per_32_symbols": CHECKS,
        "fixture_gross_saving_bps": TARGET_SYNTHETIC_GROSS_BPS,
        "fixture_constraints_per_2048_if_reset_every_32": CHECKS * (block_length // BLOCK_LENGTH),
        "interpretation": (
            "I(left;right)<=log2(chi) for one nonnegative factorization cut. "
            "This limits independent information crossing that cut, not total sequence savings: "
            "a small state may be queried repeatedly or reset and reused across blocks."
        ),
    }


def exact_normalization_probe(model: SparseUnifilarWFA, symbols: Sequence[int]) -> dict[str, object]:
    """Short exact-Fraction causal replay for the source-only receipt."""
    require(0 < len(symbols) <= BLOCK_LENGTH, "probe length")
    alpha = tuple(Fraction(1 if state == model.initial_state else 0, 1) for state in range(model.chi))
    rows = []
    for position, symbol in enumerate(symbols):
        context = public_context(position)
        p0 = model.predictive_fraction(alpha, context, 0)
        p1 = model.predictive_fraction(alpha, context, 1)
        require(p0 + p1 == 1, "exact predictive normalization")
        rows.append(
            {
                "position": position,
                "context": context,
                "p0": [p0.numerator, p0.denominator],
                "p1": [p1.numerator, p1.denominator],
            }
        )
        alpha = model.update_fraction(alpha, context, int(symbol))
        require(sum(alpha, Fraction(0, 1)) == 1, "exact posterior normalization")
    return {"steps": rows, "all_exactly_normalized": True}


def write_bytes_create_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)

