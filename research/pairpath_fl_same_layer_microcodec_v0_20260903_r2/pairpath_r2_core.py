"""Source-only PAIRPATH-P2 r2 executable microcodec.

The module contains no model locator, payload path, network, GPU, or deployment
entry point.  It is a literal two-expert experiment over caller-supplied finite
FP64 arrays.  Every source-derived model is serialized in the packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
import json
import math
import struct
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


ROLES = ("gate", "up", "down")
OPTIMIZED_ROLES = (1, 2)
ALPHABET = 4
STATE_COUNT = 2
BLOCK_VALUES = 2048
FOLD_COUNT = 8
PAGE_BYTES = 4096
RATE_MIN = Fraction(43, 20)
RATE_MAX = Fraction(5, 2)
F0 = 0.9888693569009007
TARGET_F = 0.8
REQUIRED_GAIN_BPW = 0.1528899669629145
REQUIRED_UPDOWN_GAIN_BPW = 0.22933495044437174
FIXED_ASSIGNMENT_MI_REQUIRED_BITS_PER_PAIR = 0.4586699008887435
ORACLE_EARLY_KILL_BPW = 0.045
ORACLE_ENGINEERING_MARGIN_BPW = 0.27
MAX_ALTERNATIONS = 8
LEVELS_RMS = np.asarray(
    (-1.510417608, -0.452780039, 0.452780039, 1.510417608), dtype=np.float64
)
LAMBDA_GRID = tuple(Fraction(n, 4096) for n in
                    (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))
CANDIDATES = ("independent_fixed", "pair_k2_fixed", "pair_k2_flexible")
CONTROL_SEEDS = (
    0x5041495200000001, 0x5041495200000003,
    0x5041495200000007, 0x504149520000000B,
    0x5041495200000011, 0x5041495200000013,
    0x5041495200000017, 0x504149520000001D,
)
GAUSSIAN_SEED = 0x4741555353504149
BOOTSTRAP_SEED = 0x424F4F5450325232
MAGIC = b"PPR2C001"
PRIVATE_MAGIC = b"P2PR"


class CodecError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CodecError(message)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ceil_div(a: int, b: int) -> int:
    require(isinstance(a, int) and isinstance(b, int) and a >= 0 and b > 0,
            "ceil_div domain")
    return (a + b - 1) // b


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (int(value) + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def _validate_values(values: np.ndarray, expert_counts: tuple[int, ...] = (1, 2)) -> np.ndarray:
    x = np.asarray(values)
    require(x.dtype == np.float64 and x.ndim == 3 and x.shape[0] in expert_counts and
            x.shape[1] == len(ROLES) and x.shape[2] >= FOLD_COUNT * BLOCK_VALUES and
            x.shape[2] % BLOCK_VALUES == 0 and bool(np.all(np.isfinite(x))),
            "canonical FP64 [expert,role,coordinate] source")
    return np.ascontiguousarray(x.astype("<f8", copy=False))


def canonical_source_bytes(values: np.ndarray) -> bytes:
    x = _validate_values(values)
    header = canonical_json({"dtype": "<f8", "roles": list(ROLES),
                             "shape": list(x.shape)})
    return struct.pack("<I", len(header)) + header + x.tobytes(order="C")


def source_sha256(values: np.ndarray) -> str:
    return sha256_bytes(canonical_source_bytes(values))


def estimate_scale_bits(values: np.ndarray) -> np.ndarray:
    x = _validate_values(values)
    blocks = x.shape[2] // BLOCK_VALUES
    result = np.empty((x.shape[0], len(ROLES), blocks), dtype=np.uint16)
    smallest = np.float16(np.finfo(np.float16).tiny)
    for e in range(x.shape[0]):
        for r in range(len(ROLES)):
            for b in range(blocks):
                v = x[e, r, b * BLOCK_VALUES:(b + 1) * BLOCK_VALUES]
                rms = math.sqrt(float(np.dot(v, v)) / BLOCK_VALUES)
                q = np.float16(rms) if rms > 0 else smallest
                if not np.isfinite(q) or q <= 0:
                    q = np.float16(np.finfo(np.float16).max)
                result[e, r, b] = np.asarray(q, dtype=np.float16).view(np.uint16)
    return result


def levels_per_coordinate(scale_bits: np.ndarray, coordinates: int) -> np.ndarray:
    s = np.asarray(scale_bits)
    require(s.dtype == np.uint16 and s.ndim == 1 and
            coordinates == s.size * BLOCK_VALUES, "scale geometry")
    decoded = s.view(np.float16).astype(np.float64)
    require(bool(np.all(np.isfinite(decoded))) and bool(np.all(decoded > 0)),
            "decoded scales")
    return np.repeat(decoded, BLOCK_VALUES)[:, None] * LEVELS_RMS[None, :]


def nearest_labels(values: np.ndarray, scale_bits: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    levels = levels_per_coordinate(scale_bits, x.size)
    return np.argmin((x[:, None] - levels) ** 2, axis=1).astype(np.uint8)


@dataclass(frozen=True)
class PrefixModel:
    counts: tuple[int, ...]
    lengths: tuple[int, ...]
    codes: tuple[int, ...]


def prefix_model(counts: Iterable[int]) -> PrefixModel:
    c = tuple(int(v) for v in counts)
    require(len(c) >= 2 and all(v >= 0 for v in c), "prefix counts")
    heap = []
    serial = 0
    lengths = [0] * len(c)
    for symbol, count in enumerate(c):
        heap.append((2 * count + 1, symbol, serial, (symbol,)))
        serial += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        left, right = heapq.heappop(heap), heapq.heappop(heap)
        symbols = left[3] + right[3]
        for symbol in symbols:
            lengths[symbol] += 1
        heapq.heappush(heap, (left[0] + right[0], min(symbols), serial, symbols))
        serial += 1
    order = sorted(range(len(c)), key=lambda s: (lengths[s], s))
    codes = [0] * len(c)
    code = 0
    previous = lengths[order[0]]
    for symbol in order:
        length = lengths[symbol]
        code <<= length - previous
        codes[symbol] = code
        code += 1
        previous = length
    return PrefixModel(c, tuple(lengths), tuple(codes))


class BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.valid_bits = 0

    def write(self, code: int, bits: int) -> None:
        require(isinstance(code, int) and isinstance(bits, int) and bits >= 0 and
                0 <= code < (1 << bits), "bit write")
        for shift in range(bits - 1, -1, -1):
            if self.valid_bits % 8 == 0:
                self.data.append(0)
            if (code >> shift) & 1:
                self.data[-1] |= 1 << (7 - (self.valid_bits & 7))
            self.valid_bits += 1

    def finish(self) -> tuple[bytes, int]:
        return bytes(self.data), self.valid_bits


class BitReader:
    def __init__(self, payload: bytes, valid_bits: int) -> None:
        require(isinstance(payload, bytes) and isinstance(valid_bits, int) and
                0 <= valid_bits <= len(payload) * 8, "bit reader")
        self.payload = payload
        self.valid_bits = valid_bits
        self.position = 0

    def bit(self) -> int:
        require(self.position < self.valid_bits, "truncated bitstream")
        result = ((self.payload[self.position >> 3] >>
                  (7 - (self.position & 7))) & 1)
        self.position += 1
        return result


def _write_symbol(writer: BitWriter, symbol: int, model: PrefixModel) -> None:
    require(0 <= int(symbol) < len(model.counts), "symbol range")
    writer.write(model.codes[int(symbol)], model.lengths[int(symbol)])


def _read_symbol(reader: BitReader, model: PrefixModel) -> int:
    lookup = {(model.lengths[s], model.codes[s]): s for s in range(len(model.counts))}
    code = 0
    for length in range(1, max(model.lengths) + 1):
        code = (code << 1) | reader.bit()
        if (length, code) in lookup:
            return lookup[(length, code)]
    raise CodecError("invalid prefix code")


def encode_contextual(symbols: np.ndarray, model_ids: np.ndarray,
                      models: Sequence[PrefixModel]) -> tuple[bytes, int]:
    q, ids = np.asarray(symbols), np.asarray(model_ids)
    require(q.ndim == ids.ndim == 1 and q.size == ids.size and
            np.issubdtype(q.dtype, np.integer) and np.issubdtype(ids.dtype, np.integer) and
            bool(np.all((ids >= 0) & (ids < len(models)))), "contextual geometry")
    writer = BitWriter()
    for symbol, model_id in zip(q, ids):
        _write_symbol(writer, int(symbol), models[int(model_id)])
    return writer.finish()


def decode_contextual(payload: bytes, valid_bits: int, model_ids: np.ndarray,
                      models: Sequence[PrefixModel]) -> np.ndarray:
    ids = np.asarray(model_ids)
    require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and
            bool(np.all((ids >= 0) & (ids < len(models)))), "context ids")
    reader = BitReader(payload, valid_bits)
    result = np.empty(ids.size, dtype=np.uint8)
    for i, model_id in enumerate(ids):
        result[i] = _read_symbol(reader, models[int(model_id)])
    require(reader.position == valid_bits, "trailing contextual bits")
    return result


def fold_ids(coordinates: int) -> np.ndarray:
    require(coordinates >= FOLD_COUNT * BLOCK_VALUES and coordinates % BLOCK_VALUES == 0,
            "fold geometry")
    return ((np.arange(coordinates, dtype=np.int64) // BLOCK_VALUES) % FOLD_COUNT).astype(np.uint8)


def _counts(values: np.ndarray, cardinality: int) -> tuple[int, ...]:
    require(cardinality >= 2, "cardinality")
    return tuple(int(x) for x in np.bincount(np.asarray(values, dtype=np.int64),
                                            minlength=cardinality)[:cardinality])


def _fit_pair_fold(values: np.ndarray, levels: np.ndarray, nearest: np.ndarray,
                   train: np.ndarray, flexible: bool, bit_weight: float) -> dict:
    require(values.shape == (2, values.shape[1]) and levels.shape ==
            (2, values.shape[1], ALPHABET) and nearest.shape == values.shape and
            train.shape == (values.shape[1],) and bool(np.any(train)), "pair fit geometry")
    q = nearest[:, train].copy()
    state = ((q[0].astype(np.int16) + 2 * q[1].astype(np.int16)) % STATE_COUNT).astype(np.uint8)
    x = values[:, train]
    lv = levels[:, train]
    for _ in range(MAX_ALTERNATIONS):
        state_counts = _counts(state, STATE_COUNT)
        state_model = prefix_model(state_counts)
        label_counts = [[_counts(q[e, state == s], ALPHABET)
                         for s in range(STATE_COUNT)] for e in range(2)]
        label_models = [[prefix_model(label_counts[e][s]) for s in range(STATE_COUNT)]
                        for e in range(2)]
        best_cost = np.full(x.shape[1], np.inf, dtype=np.float64)
        best_state = np.zeros(x.shape[1], dtype=np.uint8)
        best_q0 = np.zeros(x.shape[1], dtype=np.uint8)
        best_q1 = np.zeros(x.shape[1], dtype=np.uint8)
        for s in range(STATE_COUNT):
            states_bits = state_model.lengths[s]
            q0_values = range(ALPHABET) if flexible else (None,)
            q1_values = range(ALPHABET) if flexible else (None,)
            for a0 in q0_values:
                for a1 in q1_values:
                    qq0 = q[0] if a0 is None else np.full(q.shape[1], a0, dtype=np.uint8)
                    qq1 = q[1] if a1 is None else np.full(q.shape[1], a1, dtype=np.uint8)
                    if not flexible:
                        qq0, qq1 = nearest[0, train], nearest[1, train]
                    distortion = ((x[0] - lv[0, np.arange(x.shape[1]), qq0]) ** 2 +
                                  (x[1] - lv[1, np.arange(x.shape[1]), qq1]) ** 2)
                    lengths0 = np.asarray(label_models[0][s].lengths)[qq0]
                    lengths1 = np.asarray(label_models[1][s].lengths)[qq1]
                    cost = distortion + bit_weight * (states_bits + lengths0 + lengths1)
                    better = cost < best_cost
                    best_cost[better] = cost[better]
                    best_state[better] = s
                    best_q0[better] = qq0[better]
                    best_q1[better] = qq1[better]
        new_q = np.stack((best_q0, best_q1))
        if np.array_equal(state, best_state) and np.array_equal(q, new_q):
            break
        state, q = best_state, new_q
    return {"state_counts": list(_counts(state, STATE_COUNT)),
            "label_counts": [[list(_counts(q[e, state == s], ALPHABET))
                              for s in range(STATE_COUNT)] for e in range(2)]}


def _apply_pair_fold(values: np.ndarray, levels: np.ndarray, nearest: np.ndarray,
                     select: np.ndarray, model_row: Mapping, flexible: bool,
                     bit_weight: float) -> tuple[np.ndarray, np.ndarray]:
    state_model = prefix_model(model_row["state_counts"])
    label_models = [[prefix_model(model_row["label_counts"][e][s])
                     for s in range(STATE_COUNT)] for e in range(2)]
    x, lv = values[:, select], levels[:, select]
    fixed = nearest[:, select]
    best_cost = np.full(x.shape[1], np.inf, dtype=np.float64)
    best_state = np.zeros(x.shape[1], dtype=np.uint8)
    best_q0 = np.zeros(x.shape[1], dtype=np.uint8)
    best_q1 = np.zeros(x.shape[1], dtype=np.uint8)
    for s in range(STATE_COUNT):
        for a0 in range(ALPHABET):
            if not flexible and a0 > 0:
                break
            for a1 in range(ALPHABET):
                if not flexible and a1 > 0:
                    break
                q0 = fixed[0] if not flexible else np.full(x.shape[1], a0, dtype=np.uint8)
                q1 = fixed[1] if not flexible else np.full(x.shape[1], a1, dtype=np.uint8)
                distortion = ((x[0] - lv[0, np.arange(x.shape[1]), q0]) ** 2 +
                              (x[1] - lv[1, np.arange(x.shape[1]), q1]) ** 2)
                cost = distortion + bit_weight * (
                    state_model.lengths[s] + np.asarray(label_models[0][s].lengths)[q0] +
                    np.asarray(label_models[1][s].lengths)[q1])
                better = cost < best_cost
                best_cost[better] = cost[better]
                best_state[better] = s
                best_q0[better] = q0[better]
                best_q1[better] = q1[better]
    require(bool(np.all(best_state < STATE_COUNT)), "selected state range")
    return best_state, np.stack((best_q0, best_q1))


def choose_pair_labels(values: np.ndarray, scale_bits: np.ndarray, lagrange: Fraction,
                       flexible: bool) -> dict:
    x = np.asarray(values, dtype=np.float64)
    s = np.asarray(scale_bits)
    require(x.ndim == 2 and x.shape[0] == 2 and s.shape ==
            (2, x.shape[1] // BLOCK_VALUES) and s.dtype == np.uint16,
            "pair role geometry")
    require(isinstance(lagrange, Fraction) and lagrange in LAMBDA_GRID,
            "frozen lambda")
    folds = fold_ids(x.shape[1])
    levels = np.stack([levels_per_coordinate(s[e], x.shape[1]) for e in range(2)])
    nearest = np.stack([nearest_labels(x[e], s[e]) for e in range(2)])
    source_energy = float(np.sum(x * x, dtype=np.float64))
    bit_weight = float(lagrange) * max(source_energy, np.finfo(np.float64).tiny) / x.size
    models, states, labels = [], np.empty(x.shape[1], np.uint8), np.empty(x.shape, np.uint8)
    for fold in range(FOLD_COUNT):
        train, held = folds != fold, folds == fold
        model = _fit_pair_fold(x, levels, nearest, train, flexible, bit_weight)
        state, q = _apply_pair_fold(x, levels, nearest, held, model, flexible, bit_weight)
        models.append(model)
        states[held], labels[:, held] = state, q
    require(bool(np.all(states < STATE_COUNT)) and bool(np.all(labels < ALPHABET)),
            "pair output range")
    return {"models": models, "states": states, "labels": labels,
            "nearest": nearest}


def _independent_models_and_labels(values: np.ndarray, scale_bits: np.ndarray) -> tuple[list, np.ndarray]:
    x = _validate_values(values)
    folds = fold_ids(x.shape[2])
    labels = np.empty(x.shape, np.uint8)
    models = []
    for r in range(len(ROLES)):
        role_models = []
        for fold in range(FOLD_COUNT):
            held, train = folds == fold, folds != fold
            fold_models = []
            for e in range(x.shape[0]):
                q = nearest_labels(x[e, r], scale_bits[e, r])
                counts = _counts(q[train], ALPHABET)
                fold_models.append(list(counts))
                labels[e, r, held] = q[held]
            role_models.append(fold_models)
        models.append(role_models)
    return models, labels


def materialize_pair_first_tree(expert_count: int, pairs: Sequence[Sequence[int]],
                                merge_ranks: Sequence[int]):
    require(isinstance(expert_count, int) and expert_count >= 2 and expert_count % 2 == 0,
            "even expert count")
    normalized = tuple(sorted(tuple(sorted((int(p[0]), int(p[1])))) for p in pairs))
    require(len(normalized) * 2 == expert_count and
            sorted(v for pair in normalized for v in pair) == list(range(expert_count)),
            "perfect matching")
    active: list[object] = [tuple(pair) for pair in normalized]
    active.sort(key=flatten_tree)
    require(len(merge_ranks) == max(0, len(active) - 2), "merge rank count")
    for digit in merge_ranks:
        choices = [(i, j) for i in range(len(active)) for j in range(i + 1, len(active))]
        require(isinstance(digit, int) and 0 <= digit < len(choices), "merge rank")
        i, j = choices[digit]
        left, right = active[i], active[j]
        merged = (left, right) if flatten_tree(left) < flatten_tree(right) else (right, left)
        active = [node for k, node in enumerate(active) if k not in (i, j)] + [merged]
        active.sort(key=flatten_tree)
    require(len(active) == 2 or expert_count == 2, "tree active clusters")
    if expert_count == 2:
        return active[0]
    return (active[0], active[1]) if flatten_tree(active[0]) < flatten_tree(active[1]) else (active[1], active[0])


def flatten_tree(tree) -> tuple[int, ...]:
    if isinstance(tree, int):
        return (tree,)
    require(isinstance(tree, tuple) and len(tree) == 2, "binary tree")
    leaves = flatten_tree(tree[0]) + flatten_tree(tree[1])
    require(len(leaves) == len(set(leaves)), "duplicate tree leaf")
    return tuple(sorted(leaves))


def odd_double_factorial(value: int) -> int:
    require(value >= -1 and value % 2 == 1, "odd double factorial")
    return 1 if value <= 0 else math.prod(range(1, value + 1, 2))


def rank_matching(pairs: Sequence[Sequence[int]], expert_count: int) -> int:
    normalized = tuple(sorted(tuple(sorted(map(int, p))) for p in pairs))
    require(len(normalized) * 2 == expert_count and
            sorted(v for p in normalized for v in p) == list(range(expert_count)),
            "matching")
    remaining, result = list(range(expert_count)), 0
    while remaining:
        first = remaining[0]
        mate = next((b for a, b in normalized if a == first), None)
        require(mate in remaining[1:], "matching canonicality")
        result += remaining[1:].index(mate) * odd_double_factorial(len(remaining) - 3)
        remaining.remove(mate)
        remaining.remove(first)
    return result


def unrank_matching(rank: int, expert_count: int) -> tuple[tuple[int, int], ...]:
    total = odd_double_factorial(expert_count - 1)
    require(0 <= rank < total, "matching rank")
    remaining, result = list(range(expert_count)), []
    while remaining:
        first = remaining.pop(0)
        stride = odd_double_factorial(len(remaining) - 2)
        digit, rank = divmod(rank, stride)
        require(digit < len(remaining), "matching digit")
        result.append((first, remaining.pop(digit)))
    return tuple(result)


def tree_descriptor_bits(expert_count: int) -> int:
    require(expert_count >= 2 and expert_count % 2 == 0, "tree expert count")
    bits = 0 if expert_count == 2 else (odd_double_factorial(expert_count - 1) - 1).bit_length()
    for active in range(expert_count // 2, 2, -1):
        bits += (math.comb(active, 2) - 1).bit_length()
    return bits


def encode_tree_descriptor(expert_count: int, pairs: Sequence[Sequence[int]],
                           merge_ranks: Sequence[int]) -> tuple[int, int]:
    packed, width = rank_matching(pairs, expert_count), (0 if expert_count == 2 else
                      (odd_double_factorial(expert_count - 1) - 1).bit_length())
    active = expert_count // 2
    require(len(merge_ranks) == max(0, active - 2), "merge digits")
    for digit in merge_ranks:
        states = math.comb(active, 2)
        bits = (states - 1).bit_length()
        require(0 <= int(digit) < states, "merge digit")
        packed, width, active = (packed << bits) | int(digit), width + bits, active - 1
    require(width == tree_descriptor_bits(expert_count), "tree descriptor width")
    return packed, width


def decode_tree_descriptor(packed: int, expert_count: int) -> dict:
    width = tree_descriptor_bits(expert_count)
    require(isinstance(packed, int) and 0 <= packed < (1 << width if width else 1),
            "tree codeword")
    digits = []
    for active in range(3, expert_count // 2 + 1):
        bits = (math.comb(active, 2) - 1).bit_length()
        digit = packed & ((1 << bits) - 1)
        require(digit < math.comb(active, 2), "unused merge codeword")
        digits.append(digit)
        packed >>= bits
    require(packed < odd_double_factorial(expert_count - 1), "unused matching codeword")
    pairs = unrank_matching(packed, expert_count)
    merges = tuple(reversed(digits))
    tree = materialize_pair_first_tree(expert_count, pairs, merges)
    return {"pairs": pairs, "merge_ranks": merges, "tree": tree}


def _models_to_json(models) -> object:
    return models


def _append_stream(buffer: bytearray, payload: bytes, valid_bits: int, count: int) -> dict:
    row = {"offset": len(buffer), "bytes": len(payload), "valid_bits": valid_bits,
           "count": count}
    buffer.extend(payload)
    return row


def _stream_slice(payload: bytes, row: Mapping) -> bytes:
    offset, size = int(row["offset"]), int(row["bytes"])
    require(0 <= offset <= len(payload) and 0 <= size <= len(payload) - offset,
            "stream directory")
    return payload[offset:offset + size]


def _make_plan(values: np.ndarray, candidate: str, lagrange: Fraction) -> dict:
    x = _validate_values(values, (2,))
    require(candidate in CANDIDATES and lagrange in LAMBDA_GRID, "candidate request")
    scales = estimate_scale_bits(x)
    independent_models, independent_labels = _independent_models_and_labels(x, scales)
    labels = independent_labels.copy()
    pair_models, states = {}, {}
    if candidate != "independent_fixed":
        for role in OPTIMIZED_ROLES:
            result = choose_pair_labels(x[:, role], scales[:, role], lagrange,
                                        candidate == "pair_k2_flexible")
            pair_models[str(role)] = result["models"]
            states[str(role)] = result["states"]
            labels[:, role] = result["labels"]
    return {"candidate": candidate, "lambda": str(lagrange), "scales": scales,
            "labels": labels, "independent_models": independent_models,
            "pair_models": pair_models, "states": states}


def _encode_plan(values: np.ndarray, plan: Mapping) -> bytes:
    x = _validate_values(values, (2,))
    coordinates, blocks = x.shape[2], x.shape[2] // BLOCK_VALUES
    folds = fold_ids(coordinates)
    common_payload = bytearray()
    common_streams = {}
    private_payloads = [bytearray(PRIVATE_MAGIC + struct.pack("<I", e)) for e in range(2)]
    private_streams = [{}, {}]
    # All scales are expert-private and literal little-endian uint16.
    for e in range(2):
        raw = np.ascontiguousarray(plan["scales"][e].astype("<u2", copy=False)).tobytes()
        private_streams[e]["scales"] = {"offset": len(private_payloads[e]), "bytes": len(raw)}
        private_payloads[e].extend(raw)
    for role in range(len(ROLES)):
        if plan["candidate"] == "independent_fixed" or role == 0:
            for e in range(2):
                models = [prefix_model(plan["independent_models"][role][f][e])
                          for f in range(FOLD_COUNT)]
                payload, valid = encode_contextual(plan["labels"][e, role], folds, models)
                private_streams[e][str(role)] = _append_stream(
                    private_payloads[e], payload, valid, coordinates)
        else:
            rows = plan["pair_models"][str(role)]
            state_models = [prefix_model(row["state_counts"]) for row in rows]
            state = np.asarray(plan["states"][str(role)], dtype=np.uint8)
            require(bool(np.all(state < STATE_COUNT)), "state range before encode")
            payload, valid = encode_contextual(state, folds, state_models)
            common_streams[str(role)] = _append_stream(common_payload, payload, valid, coordinates)
            for e in range(2):
                models = [prefix_model(rows[f]["label_counts"][e][s])
                          for f in range(FOLD_COUNT) for s in range(STATE_COUNT)]
                ids = folds.astype(np.int64) * STATE_COUNT + state
                payload, valid = encode_contextual(plan["labels"][e, role], ids, models)
                private_streams[e][str(role)] = _append_stream(
                    private_payloads[e], payload, valid, coordinates)
    base_header = {
        "schema": "pairpath_p2_literal_packet_v2", "candidate": plan["candidate"],
        "lambda": plan["lambda"], "roles": list(ROLES), "shape": list(x.shape),
        "source_sha256": source_sha256(x), "block_values": BLOCK_VALUES,
        "fold_count": FOLD_COUNT, "state_count": STATE_COUNT,
        "tree_descriptor": {"packed": 0, "bits": 0, "pairs": [[0, 1]],
                            "merge_ranks": [], "materialized": [0, 1]},
        "independent_models": _models_to_json(plan["independent_models"]),
        "pair_models": _models_to_json(plan["pair_models"]),
        "common_streams": common_streams, "private_streams": private_streams,
        "common_payload_bytes": len(common_payload),
        "private_payload_bytes": [len(p) for p in private_payloads],
        "common_pages": 0, "private_pages": [0, 0],
    }
    # Header size and page counts are mutually dependent only through decimal
    # fields. Iterate to the unique fixed point, then allocate minimum-rate
    # surplus to private expert pages in index order.
    header = dict(base_header)
    for _ in range(16):
        h = canonical_json(header)
        common_pages = ceil_div(len(MAGIC) + 4 + len(h) + len(common_payload), PAGE_BYTES)
        private_pages = [ceil_div(len(p), PAGE_BYTES) for p in private_payloads]
        total_weights = int(np.prod(x.shape))
        min_pages = ceil_div(ceil_div(RATE_MIN.numerator * total_weights,
                                     RATE_MIN.denominator * 8), PAGE_BYTES)
        while common_pages + sum(private_pages) < min_pages:
            target = 0 if private_pages[0] <= private_pages[1] else 1
            private_pages[target] += 1
        updated = dict(base_header)
        updated["common_pages"] = common_pages
        updated["private_pages"] = private_pages
        if updated == header:
            break
        header = updated
    else:
        raise CodecError("packet header did not converge")
    h = canonical_json(header)
    common_raw = MAGIC + struct.pack("<I", len(h)) + h + bytes(common_payload)
    common_size = header["common_pages"] * PAGE_BYTES
    require(len(common_raw) <= common_size, "common allocation")
    segments = [common_raw + bytes(common_size - len(common_raw))]
    for e, payload in enumerate(private_payloads):
        size = header["private_pages"][e] * PAGE_BYTES
        require(len(payload) <= size, "private allocation")
        segments.append(bytes(payload) + bytes(size - len(payload)))
    packet = b"".join(segments)
    rate = Fraction(len(packet) * 8, total_weights)
    require(RATE_MIN <= rate <= RATE_MAX, "literal packet outside rate interval")
    return packet


def _parse_packet(packet: bytes) -> tuple[dict, bytes, list[bytes]]:
    require(isinstance(packet, bytes) and len(packet) >= 12 and packet[:8] == MAGIC,
            "packet magic")
    header_len = struct.unpack("<I", packet[8:12])[0]
    require(0 < header_len <= len(packet) - 12, "header length")
    raw_header = packet[12:12 + header_len]
    require(raw_header.endswith(b"\n"), "header newline")
    header = json.loads(raw_header)
    require(canonical_json(header) == raw_header and
            header.get("schema") == "pairpath_p2_literal_packet_v2", "canonical header")
    require(header.get("roles") == list(ROLES) and header.get("candidate") in CANDIDATES and
            Fraction(header.get("lambda")) in LAMBDA_GRID and
            header.get("block_values") == BLOCK_VALUES and header.get("fold_count") == FOLD_COUNT and
            header.get("state_count") == STATE_COUNT, "header constants")
    shape = header.get("shape")
    require(isinstance(shape, list) and len(shape) == 3 and shape[0] == 2 and
            shape[1] == 3 and shape[2] >= FOLD_COUNT * BLOCK_VALUES and
            shape[2] % BLOCK_VALUES == 0, "packet geometry")
    common_size = int(header["common_pages"]) * PAGE_BYTES
    private_sizes = [int(v) * PAGE_BYTES for v in header["private_pages"]]
    require(common_size > 0 and all(v > 0 for v in private_sizes) and
            common_size + sum(private_sizes) == len(packet), "segment sizes")
    common_payload_start = 12 + header_len
    common_payload_end = common_payload_start + int(header["common_payload_bytes"])
    require(common_payload_end <= common_size and not any(packet[common_payload_end:common_size]),
            "common padding")
    common_payload = packet[common_payload_start:common_payload_end]
    privates, offset = [], common_size
    for e, size in enumerate(private_sizes):
        segment = packet[offset:offset + size]
        raw_size = int(header["private_payload_bytes"][e])
        require(8 <= raw_size <= size and segment[:4] == PRIVATE_MAGIC and
                struct.unpack("<I", segment[4:8])[0] == e and not any(segment[raw_size:]),
                "private segment")
        privates.append(segment[:raw_size])
        offset += size
    return header, common_payload, privates


def decode_packet(packet: bytes) -> dict:
    header, common_payload, privates = _parse_packet(packet)
    coordinates = int(header["shape"][2])
    blocks, folds = coordinates // BLOCK_VALUES, fold_ids(coordinates)
    scales = np.empty((2, len(ROLES), blocks), np.uint16)
    for e in range(2):
        row = header["private_streams"][e]["scales"]
        raw = _stream_slice(privates[e], row)
        require(len(raw) == len(ROLES) * blocks * 2, "scale bytes")
        scales[e] = np.frombuffer(raw, dtype="<u2").reshape(len(ROLES), blocks)
        levels_per_coordinate(scales[e, 0], coordinates)
    labels = np.empty((2, len(ROLES), coordinates), np.uint8)
    states = {}
    for role in range(len(ROLES)):
        if header["candidate"] == "independent_fixed" or role == 0:
            for e in range(2):
                models = [prefix_model(header["independent_models"][role][f][e])
                          for f in range(FOLD_COUNT)]
                row = header["private_streams"][e][str(role)]
                labels[e, role] = decode_contextual(
                    _stream_slice(privates[e], row), int(row["valid_bits"]), folds, models)
        else:
            rows = header["pair_models"][str(role)]
            state_models = [prefix_model(row["state_counts"]) for row in rows]
            row = header["common_streams"][str(role)]
            state = decode_contextual(_stream_slice(common_payload, row),
                                      int(row["valid_bits"]), folds, state_models)
            require(bool(np.all(state < STATE_COUNT)), "decoded state range")
            states[str(role)] = state
            ids = folds.astype(np.int64) * STATE_COUNT + state
            for e in range(2):
                models = [prefix_model(rows[f]["label_counts"][e][s])
                          for f in range(FOLD_COUNT) for s in range(STATE_COUNT)]
                prow = header["private_streams"][e][str(role)]
                labels[e, role] = decode_contextual(
                    _stream_slice(privates[e], prow), int(prow["valid_bits"]), ids, models)
    reconstructed = np.empty(labels.shape, np.float64)
    for e in range(2):
        for role in range(len(ROLES)):
            levels = levels_per_coordinate(scales[e, role], coordinates)
            reconstructed[e, role] = levels[np.arange(coordinates), labels[e, role]]
    return {"header": header, "labels": labels, "scale_bits": scales,
            "states": states, "reconstruction": reconstructed,
            "packet_sha256": sha256_bytes(packet)}


def packet_read_ledger(packet: bytes) -> dict:
    header, _, _ = _parse_packet(packet)
    common_physical = int(header["common_pages"]) * PAGE_BYTES
    private_physical = [int(v) * PAGE_BYTES for v in header["private_pages"]]
    common_raw = 12 + len(canonical_json(header)) + int(header["common_payload_bytes"])
    private_raw = [int(v) for v in header["private_payload_bytes"]]
    physical_amp, conservative_amp = [], []
    for e in range(2):
        touched = common_physical + private_physical[e]
        owned_physical = Fraction(common_physical, 2) + private_physical[e]
        owned_raw = Fraction(common_raw, 2) + private_raw[e]
        physical_amp.append(Fraction(touched, 1) / owned_physical)
        conservative_amp.append(Fraction(touched, 1) / owned_raw)
    strict = all(v < 2 for v in physical_amp + conservative_amp)
    return {"status": "PASS_STRICT_READ" if strict else "FAIL_STRICT_READ",
            "strictly_below_2x": strict, "common_physical_bytes": common_physical,
            "private_physical_bytes": private_physical, "common_raw_bytes": common_raw,
            "private_raw_bytes": private_raw,
            "amplification_physical": [str(v) for v in physical_amp],
            "amplification_conservative": [str(v) for v in conservative_amp],
            "max_amplification": float(max(physical_amp + conservative_amp))}


def evaluate_packet(values: np.ndarray, packet: bytes) -> dict:
    x = _validate_values(values, (2,))
    decoded = decode_packet(packet)
    require(decoded["header"]["source_sha256"] == source_sha256(x), "packet/source binding")
    error = x - decoded["reconstruction"]
    sse_roles = np.sum(error * error, axis=(0, 2), dtype=np.float64)
    source_energy = float(np.sum(x * x, dtype=np.float64))
    require(source_energy > 0 and bool(np.all(np.isfinite(sse_roles))) and
            bool(np.all(sse_roles >= 0)), "score energies")
    total_weights = int(x.size)
    rate = Fraction(len(packet) * 8, total_weights)
    require(RATE_MIN <= rate <= RATE_MAX, "score rate")
    relative = float(np.sum(sse_roles, dtype=np.float64)) / source_energy
    f = relative * 2.0 ** (2.0 * float(rate))
    ledger = packet_read_ledger(packet)
    return {"rate_fraction": str(rate), "rate_bpw": float(rate),
            "physical_bytes": len(packet), "total_weights": total_weights,
            "sse_by_role": [float(v) for v in sse_roles], "source_energy": source_energy,
            "relative_mse": relative, "F": f, "passes_F": f <= TARGET_F,
            "read_ledger": ledger}


def score_from_energies(*, sse_by_role: Sequence[float], source_energy: float,
                        physical_bytes: int, total_weights: int) -> dict:
    require(len(sse_by_role) == len(ROLES) and all(math.isfinite(float(v)) and
            float(v) >= 0 for v in sse_by_role), "nonnegative finite SSE")
    require(math.isfinite(float(source_energy)) and source_energy > 0 and
            isinstance(physical_bytes, int) and physical_bytes >= 0 and
            isinstance(total_weights, int) and total_weights > 0, "score domain")
    rate = Fraction(physical_bytes * 8, total_weights)
    require(RATE_MIN <= rate <= RATE_MAX, "score interval")
    relative = sum(float(v) for v in sse_by_role) / float(source_energy)
    return {"relative_mse": relative, "rate_fraction": str(rate),
            "F": relative * 2.0 ** (2.0 * float(rate))}


def make_binding(values: np.ndarray, packet: bytes) -> dict:
    x = _validate_values(values, (2,))
    decoded = decode_packet(packet)
    return {"schema": "pairpath_p2_source_binding_v2", "source_sha256": source_sha256(x),
            "packet_sha256": sha256_bytes(packet), "decoded_labels_sha256":
            sha256_bytes(np.ascontiguousarray(decoded["labels"]).tobytes()),
            "decoded_scales_sha256": sha256_bytes(np.ascontiguousarray(
                decoded["scale_bits"].astype("<u2", copy=False)).tobytes()),
            "shape": list(x.shape), "packet_bytes": len(packet)}


def validate_binding(binding: Mapping, values: np.ndarray, packet: bytes) -> dict:
    x = _validate_values(values, (2,))
    require(binding.get("schema") == "pairpath_p2_source_binding_v2" and
            binding.get("shape") == list(x.shape) and binding.get("packet_bytes") == len(packet),
            "binding schema/geometry")
    actual = make_binding(x, packet)
    for key in ("source_sha256", "packet_sha256", "decoded_labels_sha256",
                "decoded_scales_sha256"):
        require(isinstance(binding.get(key), str) and binding[key] == actual[key],
                f"binding hash {key}")
    return {"status": "PASS_REAL_BYTE_BINDING", "source_sha256": actual["source_sha256"],
            "packet_sha256": actual["packet_sha256"]}


def run_micro_oracle(values: np.ndarray, lambda_grid: Sequence[Fraction] = LAMBDA_GRID) -> dict:
    x = _validate_values(values, (2,))
    lambdas = tuple(lambda_grid)
    require(bool(lambdas) and all(isinstance(v, Fraction) and v in LAMBDA_GRID for v in lambdas),
            "lambda grid subset")
    rows = []
    for lambda_index, lagrange in enumerate(lambdas):
        for candidate_index, candidate in enumerate(CANDIDATES):
            plan = _make_plan(x, candidate, lagrange)
            try:
                packet = _encode_plan(x, plan)
                score = evaluate_packet(x, packet)
                rd_objective = score["relative_mse"] + float(lagrange) * score["rate_bpw"]
                row = {"candidate": candidate, "lambda": str(lagrange), "packet": packet,
                       "score": score, "rd_objective_relative_mse_plus_lambda_bpw": rd_objective,
                       "eligible": score["read_ledger"]["strictly_below_2x"]}
            except CodecError as error:
                row = {"candidate": candidate, "lambda": str(lagrange), "eligible": False,
                       "error": str(error)}
            row["tie"] = [candidate_index, lambda_index]
            rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    require(bool(eligible), "no rate/read-legal candidate")
    selected = min(eligible, key=lambda row: (row["score"]["F"], row["score"]["physical_bytes"],
                                              row["tie"]))
    independent = min((r for r in eligible if r["candidate"] == "independent_fixed"),
                      key=lambda row: (row["score"]["F"], row["tie"]))
    gain = 0.5 * math.log2(independent["score"]["F"] / selected["score"]["F"])
    summary_rows = [{k: v for k, v in row.items() if k != "packet"} for row in rows]
    return {"schema": "pairpath_p2_micro_oracle_result_v2",
            "status": "SURVIVE_SOURCE_ONLY" if gain >= REQUIRED_GAIN_BPW else
                      "HARD_KILL_BELOW_REQUIRED_GAIN",
            "source_sha256": source_sha256(x), "selected_candidate": selected["candidate"],
            "selected_lambda": selected["lambda"], "selected_packet": selected["packet"],
            "selected_packet_sha256": sha256_bytes(selected["packet"]),
            "selected_score": selected["score"], "independent_score": independent["score"],
            "equivalent_gain_bpw": gain, "required_gain_bpw": REQUIRED_GAIN_BPW,
            "candidate_rows": summary_rows}


def affine_value_control(values: np.ndarray, seed: int) -> np.ndarray:
    x = _validate_values(values, (2,))
    result = np.empty_like(x)
    base = np.arange(BLOCK_VALUES, dtype=np.int64)
    for e in range(2):
        for role in range(len(ROLES)):
            for block in range(x.shape[2] // BLOCK_VALUES):
                material = int(seed) ^ (e << 48) ^ (role << 40) ^ block
                a = _splitmix64(material) % BLOCK_VALUES or 1
                while math.gcd(a, BLOCK_VALUES) != 1:
                    a = (a + 1) % BLOCK_VALUES or 1
                b = _splitmix64(material ^ 0xD1B54A32D192ED03) % BLOCK_VALUES
                lo = block * BLOCK_VALUES
                result[e, role, lo:lo + BLOCK_VALUES] = x[
                    e, role, lo + (a * base + b) % BLOCK_VALUES]
    return result


def gaussian_value_control(values: np.ndarray, seed: int = GAUSSIAN_SEED) -> np.ndarray:
    x = _validate_values(values, (2,))
    result = np.empty_like(x)
    for e in range(2):
        for role in range(len(ROLES)):
            for block in range(x.shape[2] // BLOCK_VALUES):
                lo = block * BLOCK_VALUES
                source = x[e, role, lo:lo + BLOCK_VALUES]
                mean = float(np.mean(source, dtype=np.float64))
                centered = source - mean
                rms = math.sqrt(float(np.dot(centered, centered)) / BLOCK_VALUES)
                rng = np.random.Generator(np.random.PCG64DXSM(
                    int(seed) ^ (e << 48) ^ (role << 40) ^ block))
                z = rng.standard_normal(BLOCK_VALUES, dtype=np.float64)
                z -= float(np.mean(z, dtype=np.float64))
                z /= math.sqrt(float(np.dot(z, z)) / BLOCK_VALUES)
                result[e, role, lo:lo + BLOCK_VALUES] = mean + rms * z
    return result


def run_complete_controls(values: np.ndarray, *,
                          control_seeds: Sequence[int] = CONTROL_SEEDS,
                          include_gaussian: bool = True,
                          lambda_grid: Sequence[Fraction] = LAMBDA_GRID) -> dict:
    x = _validate_values(values, (2,))
    source = run_micro_oracle(x, lambda_grid)
    controls = []
    for seed in control_seeds:
        controlled = affine_value_control(x, int(seed))
        before_scales = estimate_scale_bits(controlled)
        result = run_micro_oracle(controlled, lambda_grid)
        after_scales = decode_packet(result["selected_packet"])["scale_bits"]
        require(np.array_equal(before_scales, after_scales), "control scale/requantize closure")
        controls.append({"kind": "affine", "seed_hex": f"{int(seed):016x}",
                         "equivalent_gain_bpw": result["equivalent_gain_bpw"],
                         "selected_candidate": result["selected_candidate"]})
    if include_gaussian:
        controlled = gaussian_value_control(x)
        before_scales = estimate_scale_bits(controlled)
        result = run_micro_oracle(controlled, lambda_grid)
        after_scales = decode_packet(result["selected_packet"])["scale_bits"]
        require(np.array_equal(before_scales, after_scales), "Gaussian scale/requantize closure")
        controls.append({"kind": "gaussian", "seed_hex": f"{GAUSSIAN_SEED:016x}",
                         "equivalent_gain_bpw": result["equivalent_gain_bpw"],
                         "selected_candidate": result["selected_candidate"]})
    max_positive = max([0.0] + [float(row["equivalent_gain_bpw"]) for row in controls])
    corrected = float(source["equivalent_gain_bpw"]) - max_positive
    return {"schema": "pairpath_p2_complete_control_result_v2",
            "source_result": {k: v for k, v in source.items() if k != "selected_packet"},
            "controls": controls, "control_count": len(controls),
            "max_positive_control_gain_bpw": max_positive,
            "control_corrected_gain_bpw": corrected,
            "status": "SURVIVE_CONTROLS_SOURCE_ONLY" if corrected >= REQUIRED_GAIN_BPW else
                      "HARD_KILL_CONTROL_CORRECTED_BELOW_REQUIRED_GAIN"}


def _resample_joint_blocks(values: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    x = _validate_values(values, (2,))
    blocks = x.shape[2] // BLOCK_VALUES
    require(len(indices) == blocks and all(0 <= int(v) < blocks for v in indices),
            "bootstrap indices")
    return np.concatenate([x[:, :, int(v) * BLOCK_VALUES:(int(v) + 1) * BLOCK_VALUES]
                           for v in indices], axis=2)


def bootstrap_full_refit(values: np.ndarray, replicates: int,
                         lambda_grid: Sequence[Fraction] = LAMBDA_GRID) -> dict:
    x = _validate_values(values, (2,))
    require(isinstance(replicates, int) and replicates >= 1, "bootstrap replicates")
    blocks, gains = x.shape[2] // BLOCK_VALUES, []
    for replicate in range(replicates):
        rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED ^ replicate))
        indices = rng.integers(0, blocks, size=blocks)
        gains.append(float(run_micro_oracle(_resample_joint_blocks(x, indices),
                                             lambda_grid)["equivalent_gain_bpw"]))
    ordered = sorted(gains)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {"replicates": replicates, "gains_bpw": gains,
            "nearest_rank_upper95_bpw": ordered[rank],
            "full_pipeline_refit_each_replicate": True,
            "joint_block_resampling": True}


def _single_expert_refit(values: np.ndarray) -> dict:
    x = _validate_values(values, (1,))
    scales = estimate_scale_bits(x)
    _, labels = _independent_models_and_labels(x, scales)
    reconstruction = np.empty_like(x)
    for role in range(len(ROLES)):
        lv = levels_per_coordinate(scales[0, role], x.shape[2])
        reconstruction[0, role] = lv[np.arange(x.shape[2]), labels[0, role]]
    sse = float(np.sum((x - reconstruction) ** 2, dtype=np.float64))
    energy = float(np.sum(x * x, dtype=np.float64))
    return {"candidate": "independent_singleton_refit", "relative_mse": sse / energy,
            "source_sha256": source_sha256(x)}


def leave_one_expert_out_refit(values: np.ndarray) -> dict:
    x = _validate_values(values, (2,))
    rows = []
    for omitted in range(2):
        kept = x[[1 - omitted]]
        row = _single_expert_refit(kept)
        row["omitted_expert"] = omitted
        rows.append(row)
    return {"status": "DIAGNOSTIC_ONLY_PAIR_STRUCTURE_REMOVED", "panels": rows,
            "full_refit": True}


def leave_one_layer_out_refit(layers: Sequence[np.ndarray],
                              lambda_grid: Sequence[Fraction] = LAMBDA_GRID) -> dict:
    sources = [_validate_values(v, (2,)) for v in layers]
    require(len(sources) >= 2 and len({v.shape for v in sources}) == 1,
            "whole-layer panel")
    rows = []
    for omitted in range(len(sources)):
        retained = [run_micro_oracle(v, lambda_grid) for i, v in enumerate(sources)
                    if i != omitted]
        rows.append({"omitted_layer": omitted,
                     "mean_gain_bpw": float(np.mean([r["equivalent_gain_bpw"] for r in retained],
                                                     dtype=np.float64)),
                     "retained_layer_count": len(retained)})
    return {"status": "PASS_WHOLE_LAYER_LOO_DIAGNOSTIC", "panels": rows,
            "full_refit": True, "promotion_confidence_available": len(sources) >= 2}


def _entropy_bits(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(c))
    require(c.ndim == 1 and total > 0 and bool(np.all(c >= 0)), "entropy counts")
    p = c[c > 0] / total
    return float(-np.sum(p * np.log2(p), dtype=np.float64))


def fixed_assignment_mi_ceiling(values: np.ndarray) -> dict:
    """Role-conditioned nearest-label MI ceiling per Up/Down weight.

    Role is decoder-visible.  Pooling Up and Down before computing mutual
    information can manufacture mixture dependence, so the authoritative
    ceiling is the coordinate-weighted mean of the per-role mutual
    informations.
    """
    x = _validate_values(values, (2,))
    scales = estimate_scale_bits(x)
    role_rows = []
    for role in OPTIMIZED_ROLES:
        q0 = nearest_labels(x[0, role], scales[0, role])
        q1 = nearest_labels(x[1, role], scales[1, role])
        joint_counts = np.zeros((ALPHABET, ALPHABET), dtype=np.int64)
        np.add.at(joint_counts, (q0, q1), 1)
        marginal0 = np.bincount(q0, minlength=ALPHABET)
        marginal1 = np.bincount(q1, minlength=ALPHABET)
        pair_entropy = _entropy_bits(joint_counts.ravel())
        role_mi = (_entropy_bits(marginal0) + _entropy_bits(marginal1) -
                   pair_entropy)
        role_rows.append({"role": ROLES[role], "coordinates": int(q0.size),
                          "mutual_information_bits_per_coordinate_pair": role_mi})
    total_coordinates = sum(row["coordinates"] for row in role_rows)
    mutual_information = sum(
        row["coordinates"] * row["mutual_information_bits_per_coordinate_pair"]
        for row in role_rows) / total_coordinates
    ceiling_bpw = mutual_information / 2.0
    return {"mutual_information_bits_per_coordinate_pair": mutual_information,
            "fixed_assignment_ceiling_bpw": ceiling_bpw,
            "conditioning": "decoder-visible role",
            "role_rows": role_rows,
            "required_mutual_information_bits_per_pair":
                FIXED_ASSIGNMENT_MI_REQUIRED_BITS_PER_PAIR,
            "passes_fixed_assignment_standalone_necessary_screen":
                mutual_information >= FIXED_ASSIGNMENT_MI_REQUIRED_BITS_PER_PAIR,
            "claim": "necessary ceiling for the frozen nearest assignment only; flexible labels remain open"}


def _ideal_initializations(values: np.ndarray, levels: np.ndarray) -> list[np.ndarray]:
    """Deterministic symmetric starts for the independent and joint solvers."""
    distortion = (values[:, :, None] - levels) ** 2
    nearest = np.argmin(distortion, axis=2).astype(np.uint8)
    equal = np.argmin(distortion[0] + distortion[1], axis=1).astype(np.uint8)
    starts = [nearest, np.stack((equal, equal)).astype(np.uint8)]
    for a0 in range(ALPHABET):
        for a1 in range(ALPHABET):
            starts.append(np.stack((
                np.full(values.shape[1], a0, dtype=np.uint8),
                np.full(values.shape[1], a1, dtype=np.uint8),
            )))
    unique, seen = [], set()
    for start in starts:
        key = start.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(start)
    return unique


def _ideal_flexible_role(values: np.ndarray, levels: np.ndarray, bit_weight: float,
                         joint: bool) -> tuple[np.ndarray, float, float]:
    """Multistart model-free single-letter alternating oracle for one role."""
    require(values.shape[0] == 2 and levels.shape == values.shape + (ALPHABET,),
            "ideal role geometry")
    require(math.isfinite(bit_weight) and bit_weight >= 0, "ideal bit weight")

    def score(labels: np.ndarray) -> tuple[float, float, float]:
        reconstruction = np.take_along_axis(levels, labels[:, :, None], axis=2)[:, :, 0]
        sse = float(np.sum((values - reconstruction) ** 2, dtype=np.float64))
        if joint:
            index = labels[0].astype(np.int16) * ALPHABET + labels[1]
            rate = _entropy_bits(np.bincount(
                index, minlength=ALPHABET * ALPHABET)) / 2.0
        else:
            rate = sum(_entropy_bits(np.bincount(labels[e], minlength=ALPHABET))
                       for e in range(2)) / 2.0
        total_bits = rate * values.size
        return sse + bit_weight * total_bits, sse, rate

    best = None
    for start_index, start in enumerate(_ideal_initializations(values, levels)):
        q = start.copy()
        visited = set()
        for iteration in range(MAX_ALTERNATIONS + 1):
            objective, sse, rate = score(q)
            key = (objective, sse, rate, start_index, iteration)
            if best is None or key < best[0]:
                best = (key, q.copy(), sse, rate)
            packed = q.tobytes()
            if iteration == MAX_ALTERNATIONS or packed in visited:
                break
            visited.add(packed)
            if joint:
                index = q[0].astype(np.int16) * ALPHABET + q[1]
                counts = np.bincount(
                    index, minlength=ALPHABET * ALPHABET).astype(np.float64)
                length = -np.log2((counts + 0.5) /
                                  (index.size + 0.5 * counts.size))
                costs = np.empty((values.shape[1], ALPHABET * ALPHABET), np.float64)
                for a0 in range(ALPHABET):
                    for a1 in range(ALPHABET):
                        k = a0 * ALPHABET + a1
                        costs[:, k] = ((values[0] - levels[0, :, a0]) ** 2 +
                                       (values[1] - levels[1, :, a1]) ** 2 +
                                       bit_weight * length[k])
                selected = np.argmin(costs, axis=1)
                new_q = np.stack((selected // ALPHABET,
                                  selected % ALPHABET)).astype(np.uint8)
            else:
                new_q = np.empty_like(q)
                for e in range(2):
                    counts = np.bincount(q[e], minlength=ALPHABET).astype(np.float64)
                    length = -np.log2((counts + 0.5) /
                                      (q.shape[1] + 0.5 * ALPHABET))
                    costs = ((values[e, :, None] - levels[e]) ** 2 +
                             bit_weight * length[None, :])
                    new_q[e] = np.argmin(costs, axis=1).astype(np.uint8)
            if np.array_equal(q, new_q):
                break
            q = new_q
    require(best is not None, "ideal multistart result")
    return best[1], best[2], best[3]


def _pareto_rd(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower convex rate-distortion envelope; time sharing is granted free."""
    best = {}
    for rate, distortion in points:
        require(math.isfinite(rate) and rate >= 0 and math.isfinite(distortion) and
                distortion > 0, "RD point")
        best[rate] = min(best.get(rate, math.inf), distortion)
    ordered = []
    running = math.inf
    for rate, distortion in sorted(best.items()):
        if distortion < running:
            ordered.append((rate, distortion))
            running = distortion
    hull: list[tuple[float, float]] = []
    for point in ordered:
        hull.append(point)
        while len(hull) >= 3:
            a, b, c = hull[-3:]
            slope_ab = (b[1] - a[1]) / (b[0] - a[0])
            slope_bc = (c[1] - b[1]) / (c[0] - b[0])
            if slope_bc <= slope_ab:
                hull.pop(-2)
            else:
                break
    return hull


def _interp_distortion(hull: Sequence[tuple[float, float]], rate: float) -> float:
    require(len(hull) >= 1 and hull[0][0] <= rate <= hull[-1][0], "rate interpolation")
    for left, right in zip(hull, hull[1:]):
        if left[0] <= rate <= right[0]:
            alpha = (rate - left[0]) / (right[0] - left[0])
            return left[1] + alpha * (right[1] - left[1])
    return hull[-1][1]


def _interp_rate(hull: Sequence[tuple[float, float]], distortion: float) -> float:
    require(hull[-1][1] <= distortion <= hull[0][1], "distortion interpolation")
    for left, right in zip(hull, hull[1:]):
        if right[1] <= distortion <= left[1]:
            alpha = (left[1] - distortion) / (left[1] - right[1])
            return left[0] + alpha * (right[0] - left[0])
    return hull[-1][0]


def optimistic_single_letter_joint_gate(values: np.ndarray,
                                        lambda_grid: Sequence[Fraction] = LAMBDA_GRID) -> dict:
    """Equal-flexibility independent/joint oracle with convexified RD frontiers.

    It grants source-derived distributions and time-sharing schedules free, so
    it can kill but never promote a finite codec.
    """
    x = _validate_values(values, (2,))
    scales = estimate_scale_bits(x)
    grid = (Fraction(0, 1),) + tuple(lambda_grid)
    require(all(v == 0 or v in LAMBDA_GRID for v in grid), "oracle lambda grid")
    independent_points, pair_points = [], []
    energy = float(np.sum(x[:, OPTIMIZED_ROLES] ** 2, dtype=np.float64))
    for lagrange in grid:
        bit_weight = (float(lagrange) * max(energy, np.finfo(np.float64).tiny) /
                      x[:, OPTIMIZED_ROLES].size)
        independent_sse = pair_sse = 0.0
        independent_rate_sum = pair_rate_sum = 0.0
        for role in OPTIMIZED_ROLES:
            levels = np.stack([levels_per_coordinate(scales[e, role], x.shape[2])
                               for e in range(2)])
            _, sse_i, rate_i = _ideal_flexible_role(
                x[:, role], levels, bit_weight, False)
            _, sse_p, rate_p = _ideal_flexible_role(
                x[:, role], levels, bit_weight, True)
            independent_sse += sse_i
            pair_sse += sse_p
            independent_rate_sum += rate_i
            pair_rate_sum += rate_p
        independent_points.append((independent_rate_sum / len(OPTIMIZED_ROLES),
                                   independent_sse / energy))
        pair_points.append((pair_rate_sum / len(OPTIMIZED_ROLES), pair_sse / energy))
    independent_hull, pair_hull = _pareto_rd(independent_points), _pareto_rd(pair_points)
    lo = max(independent_hull[0][0], pair_hull[0][0])
    hi = min(independent_hull[-1][0], pair_hull[-1][0])
    rates = sorted({lo, hi} | {r for r, _ in independent_hull if lo <= r <= hi} |
                   {r for r, _ in pair_hull if lo <= r <= hi})
    equal_rate = []
    if lo <= hi:
        for rate in rates:
            di = _interp_distortion(independent_hull, rate)
            dp = _interp_distortion(pair_hull, rate)
            equal_rate.append({"rate_bpw": rate, "D_ind": di, "D_pair": dp,
                               "G_eq_bpw": 0.5 * math.log2(di / dp)})
    dlo = max(independent_hull[-1][1], pair_hull[-1][1])
    dhi = min(independent_hull[0][1], pair_hull[0][1])
    distortions = sorted({dlo, dhi} |
                         {d for _, d in independent_hull if dlo <= d <= dhi} |
                         {d for _, d in pair_hull if dlo <= d <= dhi}, reverse=True)
    equal_mse = []
    if dlo <= dhi:
        for distortion in distortions:
            ri = _interp_rate(independent_hull, distortion)
            rp = _interp_rate(pair_hull, distortion)
            equal_mse.append({"relative_D": distortion, "R_ind_bpw": ri,
                              "R_pair_bpw": rp, "G_eq_bpw": ri - rp})
    gains = [row["G_eq_bpw"] for row in equal_rate + equal_mse]
    best_gain = max(gains) if gains else -math.inf
    if best_gain < ORACLE_EARLY_KILL_BPW:
        status = "HARD_KILL_OPTIMISTIC_JOINT_GATE_BELOW_0P045"
    elif best_gain >= ORACLE_ENGINEERING_MARGIN_BPW:
        status = "SURVIVE_OPTIMISTIC_GATE_WITH_PHYSICAL_MARGIN"
    elif best_gain >= REQUIRED_UPDOWN_GAIN_BPW:
        status = "SURVIVE_OPTIMISTIC_GATE_STANDALONE_THRESHOLD"
    else:
        status = "INTERESTING_BUT_INSUFFICIENT_STANDALONE"
    return {"schema": "pairpath_p2_optimistic_single_letter_gate_v2", "status": status,
            "claim": "kill-only; source-derived probabilities and time sharing are free",
            "fixed_assignment_mi": fixed_assignment_mi_ceiling(x),
            "independent_hull": independent_hull, "pair_hull": pair_hull,
            "equal_rate": equal_rate, "equal_mse": equal_mse,
            "best_G_eq_UD_bpw": best_gain, "early_kill_bpw": ORACLE_EARLY_KILL_BPW,
            "standalone_required_bpw": REQUIRED_UPDOWN_GAIN_BPW,
            "physical_engineering_margin_bpw": ORACLE_ENGINEERING_MARGIN_BPW}


def hard_kill_contract() -> dict:
    return {
        "known_fixed_label_cbib": "net ideal gain 0.000010730760043135371 bpw; fixed-label same-layer branch is dead",
        "fixed_assignment_mi": "standalone requires I(A;B)>=0.4586699008887435 bits per coordinate pair before overhead",
        "optimistic_joint": "hard-kill if convexified equal-rate/equal-MSE G_eq,UD<0.045 bpw; standalone >=0.22933495044437174; engineering margin >=0.27",
        "stage_1_64": "if favorable gain <0.015 bpw, require 1/8 confirmation",
        "stage_1_8": "kill if nearest-rank upper95 full-refit bootstrap gain <0.03 bpw",
        "source_specific": f"kill unless control-corrected gain >= {REQUIRED_GAIN_BPW:.16f} bpw",
        "finite": "kill unless one decoded packet has F<=0.8, 2.15<=R<=2.5, max read <2",
        "promotion": "requires at least two whole held-out layers; this source-only package cannot promote",
    }
