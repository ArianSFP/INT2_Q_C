#!/usr/bin/env python3
"""Exact finite arithmetic search and statistical core for epsilon-TCQ v0."""

from __future__ import annotations

import hashlib
import itertools
import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


TOTAL = 1 << 16
CONTEXTS = 16
MASK32 = (1 << 32) - 1
HALF = 1 << 31
QUARTER = 1 << 30
THREE_QUARTER = 3 << 30
TOPOLOGIES = (
    "suffix", "xor_sketch", "modular_ones", "rolling_affine",
    "signed_saturating",
)
FROZEN_BANK = (
    ("suffix", 4, 32),
    ("suffix", 8, 128),
    ("xor_sketch", 8, 64),
    ("modular_ones", 8, 64),
    ("rolling_affine", 8, 128),
    ("signed_saturating", 8, 64),
)
EPSILONS = (1, 2)
LAMBDA_EXPONENTS = (-16, -12, -8, -4)
BEAM_WIDTHS = (32, 128, 512)
EXACT_MAX_LABELS = 12
EXACT_MAX_CHOICES = 3


class CoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoreError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pack_bits(bits: Sequence[int]) -> bytes:
    output = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        require(bit in (0, 1), "pack binary bit")
        output[index >> 3] |= int(bit) << (7 - (index & 7))
    return bytes(output)


def unpack_bit(payload: bytes, logical_bits: int, index: int) -> int:
    require(type(logical_bits) is int and 0 <= logical_bits <= 8 * len(payload),
            "logical bit bound")
    if index >= logical_bits:
        return 0
    return (payload[index >> 3] >> (7 - (index & 7))) & 1


@dataclass(frozen=True)
class ModelCandidate:
    topology: str
    states: int
    reset: int

    def validate(self) -> None:
        require((self.topology, self.states, self.reset) in FROZEN_BANK,
                "candidate belongs to frozen bank")
        require(self.states > 0 and self.states & (self.states - 1) == 0,
                "power-of-two states")

    @property
    def selector(self) -> int:
        self.validate()
        return FROZEN_BANK.index((self.topology, self.states, self.reset))


@dataclass(frozen=True)
class FittedModel:
    candidate: ModelCandidate
    frequencies_q16: tuple[int, ...]

    def validate(self) -> None:
        self.candidate.validate()
        require(len(self.frequencies_q16) == self.candidate.states * CONTEXTS,
                "complete model frequency table")
        require(all(type(value) is int and 1 <= value <= 65535
                    for value in self.frequencies_q16), "Q0.16 frequencies")

    def frequency(self, state: int, context: int) -> int:
        self.validate()
        require(0 <= state < self.candidate.states and 0 <= context < CONTEXTS,
                "frequency lookup")
        return self.frequencies_q16[state * CONTEXTS + context]


def transition(candidate: ModelCandidate, state: int, bit: int,
               context: int, event_ordinal: int) -> int:
    candidate.validate()
    require(0 <= state < candidate.states and bit in (0, 1) and
            0 <= context < CONTEXTS and event_ordinal >= 0,
            "transition input")
    mask = candidate.states - 1
    if candidate.topology == "suffix":
        return ((state << 1) | bit) & mask
    if candidate.topology == "xor_sketch":
        sketch = ((context * 5 + event_ordinal * 3 + 1) & mask) if bit else 0
        return state ^ sketch
    if candidate.topology == "modular_ones":
        return (state + ((context | 1) & mask) * bit) & mask
    if candidate.topology == "rolling_affine":
        multiplier = 5 if candidate.states >= 8 else 1
        return (multiplier * state + context + event_ordinal + bit) & mask
    if candidate.topology == "signed_saturating":
        return min(mask, state + 1) if bit else max(0, state - 1)
    raise CoreError("unreachable topology")


@dataclass
class ArithmeticEncoder:
    low: int = 0
    high: int = MASK32
    pending: int = 0
    bits: tuple[int, ...] = ()

    def clone(self) -> "ArithmeticEncoder":
        return ArithmeticEncoder(self.low, self.high, self.pending, self.bits)

    def _emit(self, bit: int) -> None:
        require(bit in (0, 1), "arithmetic emit bit")
        self.bits = self.bits + (bit,) + (1 - bit,) * self.pending
        self.pending = 0

    def encode(self, bit: int, frequency_one: int) -> None:
        require(bit in (0, 1) and 1 <= frequency_one <= 65535,
                "arithmetic encode input")
        width = self.high - self.low + 1
        frequency_zero = TOTAL - frequency_one
        split = self.low + (width * frequency_zero // TOTAL) - 1
        require(self.low <= split < self.high, "arithmetic nonempty split")
        if bit == 0:
            self.high = split
        else:
            self.low = split + 1
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK32
            self.high = ((self.high << 1) & MASK32) | 1

    def finish(self) -> tuple[bytes, int]:
        final = self.clone()
        final.pending += 1
        final._emit(0 if final.low < QUARTER else 1)
        return pack_bits(final.bits), len(final.bits)

    def state_key(self) -> tuple[Any, ...]:
        return self.low, self.high, self.pending, self.bits


class ArithmeticDecoder:
    def __init__(self, payload: bytes, logical_bits: int) -> None:
        require(type(payload) is bytes and 0 <= logical_bits <= 8 * len(payload),
                "arithmetic decoder payload")
        if logical_bits & 7:
            mask = (1 << (8 - (logical_bits & 7))) - 1
            require(not (payload[-1] & mask), "canonical zero tail bits")
        self.payload = payload
        self.logical_bits = logical_bits
        self.cursor = 0
        self.low = 0
        self.high = MASK32
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) & MASK32) | self._read()

    def _read(self) -> int:
        bit = unpack_bit(self.payload, self.logical_bits, self.cursor)
        self.cursor += 1
        return bit

    def decode(self, frequency_one: int) -> int:
        require(1 <= frequency_one <= 65535, "arithmetic decode frequency")
        width = self.high - self.low + 1
        split = self.low + (width * (TOTAL - frequency_one) // TOTAL) - 1
        if self.code <= split:
            bit = 0
            self.high = split
        else:
            bit = 1
            self.low = split + 1
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK32
            self.high = ((self.high << 1) & MASK32) | 1
            self.code = ((self.code << 1) & MASK32) | self._read()
        return bit


def fit_model(candidate: ModelCandidate,
              trajectories: Iterable[Sequence[Any]]) -> FittedModel:
    candidate.validate()
    zeros = [0] * (candidate.states * CONTEXTS)
    ones = [0] * (candidate.states * CONTEXTS)
    for trajectory in trajectories:
        state = 0
        event_ordinal = 0
        for position, choice in enumerate(trajectory):
            choice.validate(choice.interface if hasattr(choice, "interface") else
                            ("strata_sc_6bit_legal_replay"
                             if len(choice.event_bits) == 6 else
                             "direct_int2_4level_new_codec"))
            if position % candidate.reset == 0:
                state = 0
            for bit, context in zip(choice.event_bits, choice.event_contexts):
                index = state * CONTEXTS + context
                (ones if bit else zeros)[index] += 1
                state = transition(candidate, state, bit, context, event_ordinal)
                event_ordinal += 1
    frequencies = []
    for zero, one in zip(zeros, ones):
        # Jeffreys half-count with exact nearest-integer rational rounding.
        numerator = 2 * one + 1
        denominator = 2 * (zero + one + 1)
        value = (numerator * TOTAL + denominator // 2) // denominator
        frequencies.append(min(65535, max(1, int(value))))
    model = FittedModel(candidate, tuple(frequencies))
    model.validate()
    return model


@dataclass(frozen=True)
class CentroidHead:
    mode: str
    states: int
    labels: int
    values: tuple[float, ...]

    def validate(self) -> None:
        require(self.mode in {"nominal", "local", "state", "state_permuted"},
                "centroid mode")
        expected = 0 if self.mode == "nominal" else (
            self.labels if self.mode == "local" else self.states * self.labels)
        require(len(self.values) == expected and
                all(math.isfinite(value) for value in self.values),
                "centroid table")

    def _mapped_state(self, state: int, stream_ordinal: int) -> int:
        require(0 <= state < self.states, "centroid state")
        if self.mode != "state_permuted":
            return state
        mask = self.states - 1
        multiplier = ((stream_ordinal * 6 + 5) & mask) | 1
        offset = (stream_ordinal * 11 + 3) & mask
        return (multiplier * state + offset) & mask

    def correction(self, pre_state: int, label: int,
                   stream_ordinal: int = 0) -> float:
        self.validate()
        require(0 <= label < self.labels, "centroid label")
        if self.mode == "nominal":
            return 0.0
        if self.mode == "local":
            return self.values[label]
        state = self._mapped_state(pre_state, stream_ordinal)
        return self.values[state * self.labels + label]


def _half(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def fit_centroid_head(mode: str, states: int, labels: int,
                      samples: Iterable[tuple[int, int, int, float, float]]) -> CentroidHead:
    require(mode in {"nominal", "local", "state", "state_permuted"} and
            states > 0 and labels in (4, 64), "centroid fit geometry")
    if mode == "nominal":
        return CentroidHead(mode, states, labels, ())
    cells = labels if mode == "local" else states * labels
    sums = [0.0] * cells
    counts = [0] * cells
    provisional = CentroidHead(mode, states, labels, (0.0,) * cells)
    for stream, state, label, target, nominal in samples:
        require(math.isfinite(target) and math.isfinite(nominal),
                "centroid sample")
        mapped = 0 if mode == "local" else provisional._mapped_state(state, stream)
        index = label if mode == "local" else mapped * labels + label
        sums[index] += target - nominal
        counts[index] += 1
    values = tuple(_half(sums[index] / counts[index]) if counts[index] else 0.0
                   for index in range(cells))
    result = CentroidHead(mode, states, labels, values)
    result.validate()
    return result


@dataclass
class SearchPath:
    legal_state: int
    wfa_state: int
    coder: ArithmeticEncoder
    distortion: float
    labels: tuple[int, ...]
    pre_states: tuple[int, ...]

    def terminal(self, fixed_packet_bytes: int, rate_lambda: float) -> tuple[float, bytes, int]:
        payload, logical = self.coder.finish()
        physical_bits = 8 * (fixed_packet_bytes + len(payload))
        return self.distortion + rate_lambda * physical_bits, payload, logical


@dataclass(frozen=True)
class SearchResult:
    labels: tuple[int, ...]
    pre_states: tuple[int, ...]
    payload: bytes
    logical_bits: int
    distortion: float
    physical_bytes: int
    objective: float
    exact: bool
    beam_width: int | None


def _pre_state(path: SearchPath, position: int, model: FittedModel) -> int:
    return 0 if position % model.candidate.reset == 0 else path.wfa_state


def _reconstruction(path: SearchPath, choice: Any, position: int,
                    model: FittedModel, head: CentroidHead,
                    stream_ordinal: int) -> float:
    state = _pre_state(path, position, model)
    return choice.nominal + head.correction(
        state, choice.label, stream_ordinal)


def _expand(path: SearchPath, choice: Any, target: float, position: int,
            model: FittedModel, head: CentroidHead, stream_ordinal: int,
            branch_squared_error: float | None = None) -> SearchPath:
    choice.validate("strata_sc_6bit_legal_replay"
                    if len(choice.event_bits) == 6 else
                    "direct_int2_4level_new_codec")
    state = _pre_state(path, position, model)
    pre_state = state
    coder = path.coder.clone()
    event_base = position * len(choice.event_bits)
    for level, (bit, context) in enumerate(zip(choice.event_bits,
                                               choice.event_contexts)):
        coder.encode(bit, model.frequency(state, context))
        state = transition(model.candidate, state, bit, context,
                           event_base + level)
    reconstruction = choice.nominal + head.correction(
        pre_state, choice.label, stream_ordinal)
    error = float(target) - reconstruction
    squared_error = error * error if branch_squared_error is None else float(
        branch_squared_error)
    require(math.isfinite(squared_error) and squared_error >= 0.0,
            "finite nonnegative branch error")
    return SearchPath(
        choice.next_legal_state, state, coder,
        path.distortion + squared_error,
        path.labels + (choice.label,), path.pre_states + (pre_state,))


def _path_sort_key(path: SearchPath, fixed_packet_bytes: int,
                   rate_lambda: float) -> tuple[Any, ...]:
    objective, payload, logical = path.terminal(fixed_packet_bytes, rate_lambda)
    return objective, path.distortion, len(payload), logical, path.labels,


def search_labels(
    targets: Sequence[float], nearest_labels: Sequence[int], adapter: Any,
    model: FittedModel, head: CentroidHead, *, epsilon: int,
    rate_lambda: float, fixed_packet_bytes: int,
    exact: bool, beam_width: int | None = None, stream_ordinal: int = 0,
    cupy_backend: Any | None = None,
) -> SearchResult:
    require(len(targets) == len(nearest_labels) and len(targets) > 0 and
            all(math.isfinite(float(value)) for value in targets),
            "search source geometry")
    require(epsilon in EPSILONS and math.isfinite(rate_lambda) and
            rate_lambda >= 0.0 and fixed_packet_bytes >= 0,
            "search hyperparameters")
    if exact:
        require(len(targets) <= EXACT_MAX_LABELS, "exact length cap")
    else:
        require(beam_width in BEAM_WIDTHS, "frozen beam width")
    model.validate()
    head.validate()
    paths = [SearchPath(adapter.initial_state(), 0, ArithmeticEncoder(),
                        0.0, (), ())]
    for position, (target, nearest) in enumerate(zip(targets, nearest_labels)):
        pending = []
        for path in paths:
            choices = tuple(adapter.encode_choices(
                position, path.legal_state, int(nearest), epsilon))
            require(choices and any(choice.label == int(nearest) for choice in choices),
                    "nearest label is a legal candidate")
            if exact:
                require(len(choices) <= EXACT_MAX_CHOICES,
                        "exact candidate cap")
            for choice in choices:
                pending.append((path, choice))
        branch_errors = None
        if cupy_backend is not None:
            target_values = [float(target)] * len(pending)
            reconstructions = [
                _reconstruction(path, choice, position, model, head,
                                stream_ordinal)
                for path, choice in pending
            ]
            branch_errors = tuple(float(value) for value in
                                  cupy_backend.score_flat_squared_error(
                                      target_values, reconstructions))
            require(len(branch_errors) == len(pending),
                    "CuPy branch score geometry")
        expanded = [
            _expand(path, choice, float(target), position, model, head,
                    stream_ordinal,
                    None if branch_errors is None else branch_errors[index])
            for index, (path, choice) in enumerate(pending)
        ]
        require(expanded, "nonempty search frontier")
        # Exact-state merging is permitted only when every decoder and coder
        # state, including emitted prefix, is identical.
        merged: dict[tuple[Any, ...], SearchPath] = {}
        for path in expanded:
            key = (path.legal_state, path.wfa_state, path.coder.state_key(),
                   path.labels)
            incumbent = merged.get(key)
            if incumbent is None or path.distortion < incumbent.distortion:
                merged[key] = path
        paths = list(merged.values())
        if not exact and len(paths) > int(beam_width):
            paths.sort(key=lambda row: _path_sort_key(
                row, fixed_packet_bytes, rate_lambda))
            paths = paths[:int(beam_width)]
    winner = min(paths, key=lambda row: _path_sort_key(
        row, fixed_packet_bytes, rate_lambda))
    objective, payload, logical = winner.terminal(fixed_packet_bytes, rate_lambda)
    return SearchResult(
        winner.labels, winner.pre_states, payload, logical,
        winner.distortion, fixed_packet_bytes + len(payload), objective,
        exact, None if exact else int(beam_width))


def fixed_nearest_search(
    targets: Sequence[float], nearest_labels: Sequence[int], adapter: Any,
    model: FittedModel, head: CentroidHead, *, rate_lambda: float,
    fixed_packet_bytes: int, stream_ordinal: int = 0,
) -> SearchResult:
    model.validate()
    path = SearchPath(adapter.initial_state(), 0, ArithmeticEncoder(), 0.0, (), ())
    for position, (target, nearest) in enumerate(zip(targets, nearest_labels)):
        choices = adapter.encode_choices(position, path.legal_state,
                                          int(nearest), 1)
        rows = [choice for choice in choices if choice.label == int(nearest)]
        require(len(rows) == 1, "unique fixed nearest legal label")
        path = _expand(path, rows[0], float(target), position,
                       model, head, stream_ordinal)
    objective, payload, logical = path.terminal(fixed_packet_bytes, rate_lambda)
    return SearchResult(path.labels, path.pre_states, payload, logical,
                        path.distortion, fixed_packet_bytes + len(payload),
                        objective, True, None)


def decode_payload(
    count: int, payload: bytes, logical_bits: int, adapter: Any,
    model: FittedModel, head: CentroidHead, *, stream_ordinal: int = 0,
) -> dict[str, Any]:
    require(type(count) is int and count > 0, "decode count")
    events = 6 if adapter.interface == "strata_sc_6bit_legal_replay" else 2
    decoder = ArithmeticDecoder(payload, logical_bits)
    legal_state = adapter.initial_state()
    state = 0
    labels = []
    pre_states = []
    reconstructions = []
    choices = []
    for position in range(count):
        if position % model.candidate.reset == 0:
            state = 0
        pre_state = state
        bits = []
        contexts = []
        for level in range(events):
            context = adapter.decode_context(position, legal_state, tuple(bits))
            bit = decoder.decode(model.frequency(state, context))
            bits.append(bit)
            contexts.append(context)
            state = transition(model.candidate, state, bit, context,
                               position * events + level)
        choice = adapter.decode_events(position, legal_state, tuple(bits))
        require(tuple(choice.event_contexts) == tuple(contexts) and
                tuple(choice.event_bits) == tuple(bits),
                "adapter causal context/event replay")
        legal_state = choice.next_legal_state
        labels.append(choice.label)
        pre_states.append(pre_state)
        reconstructions.append(choice.nominal + head.correction(
            pre_state, choice.label, stream_ordinal))
        choices.append(choice)
    replay = ArithmeticEncoder()
    replay_state = 0
    for position, choice in enumerate(choices):
        if position % model.candidate.reset == 0:
            replay_state = 0
        for level, (bit, context) in enumerate(zip(choice.event_bits,
                                                   choice.event_contexts)):
            replay.encode(bit, model.frequency(replay_state, context))
            replay_state = transition(model.candidate, replay_state, bit,
                                      context, position * events + level)
    canonical_payload, canonical_logical = replay.finish()
    require(canonical_payload == payload and canonical_logical == logical_bits,
            "literal arithmetic payload reencode")
    return {
        "labels": tuple(labels), "pre_states": tuple(pre_states),
        "reconstructions": tuple(reconstructions),
        "payload_sha256": sha256(payload),
        "logical_bits": logical_bits,
        "literal_payload_reencode_matches": True,
    }


def connected_owner_components(owner_sets: Iterable[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    rows = [tuple(sorted(set(int(value) for value in owners)))
            for owners in owner_sets]
    require(rows and all(row and all(value >= 0 for value in row) for row in rows),
            "nonempty owner sets")
    parents: dict[int, int] = {}
    def find(value: int) -> int:
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)
    for row in rows:
        for value in row:
            find(value)
        for value in row[1:]:
            union(row[0], value)
    groups: dict[int, list[int]] = {}
    for value in sorted(parents):
        groups.setdefault(find(value), []).append(value)
    return tuple(tuple(group) for _, group in sorted(groups.items()))


def bounded_hankel_cssr_diagnostic(bits: Sequence[int], prefix: int = 3,
                                    suffix: int = 3) -> dict[str, Any]:
    require(1 <= prefix <= 3 and 1 <= suffix <= 3 and
            len(bits) >= prefix + suffix and all(bit in (0, 1) for bit in bits),
            "bounded Hankel input")
    rows, columns = 1 << prefix, 1 << suffix
    matrix = [[0.0] * columns for _ in range(rows)]
    for start in range(len(bits) - prefix - suffix + 1):
        left = 0
        right = 0
        for bit in bits[start:start + prefix]:
            left = (left << 1) | int(bit)
        for bit in bits[start + prefix:start + prefix + suffix]:
            right = (right << 1) | int(bit)
        matrix[left][right] += 1.0
    total = math.fsum(math.fsum(row) for row in matrix)
    require(total > 0.0, "nonempty Hankel counts")
    work = [[value / total for value in row] for row in matrix]
    rank = 0
    column = 0
    tolerance = 1e-12
    while rank < rows and column < columns:
        pivot = max(range(rank, rows), key=lambda index: abs(work[index][column]))
        if abs(work[pivot][column]) <= tolerance:
            column += 1
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [left - factor * right
                             for left, right in zip(work[row], work[rank])]
        rank += 1
        column += 1
    require(rank <= 16, "bounded CSSR numerical rank")
    ordering = sorted(
        (ModelCandidate(*row) for row in FROZEN_BANK),
        key=lambda candidate: (abs(candidate.states - max(1, rank)),
                               candidate.selector))
    return {
        "prefix_length": prefix, "suffix_length": suffix,
        "matrix_rows": rows, "matrix_columns": columns,
        "numerical_rank": rank, "maximum_rank": 16,
        "frozen_candidate_ranking": [candidate.selector for candidate in ordering],
        "new_topology_created": False,
    }
