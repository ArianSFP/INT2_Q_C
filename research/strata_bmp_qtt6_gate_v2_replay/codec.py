#!/usr/bin/env python3
"""Canonical six-plane coordinate-function packet for the source-only gate.

This module deliberately operates on completed STRATA reconstruction planes.
It never accepts a four-level alphabet or internal polar/SC decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
import zlib
from typing import Iterable

import numpy as np


MAGIC = b"SQF6V1\x00\x00"
VERSION = 1
FAMILY_BMP = 0
FAMILY_OBDD = 1
FAMILY_QTT = 2
FAMILY_NAMES = {FAMILY_BMP: "GF2_MATRIX_FACTOR", FAMILY_OBDD: "ROBDD",
                FAMILY_QTT: "BMP_QTT_GF2"}
HEADER = struct.Struct("<8sBBBBHHHHHHBBI")
NODE = struct.Struct("<BHH")
EXCEPTION = struct.Struct("<HB")
CRC = struct.Struct("<I")
MAX_BLOCK_WEIGHTS = 4096
MAX_ACTIVE_FEATURES = 12
MAX_EXCEPTIONS = 64
MAX_BMP_RANK = 4
MAX_QTT_RANK = 2
MAX_OBDD_NODES = 240
ORDER_BANK_SIZE = 4
UINT16_MAX = (1 << 16) - 1


class CodecError(RuntimeError):
    """Fail-closed packet or geometry error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CodecError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class Geometry:
    rows: int
    cols: int
    role: int
    row_start: int
    row_count: int
    col_start: int
    col_count: int

    @property
    def count(self) -> int:
        return self.row_count * self.col_count

    def validate(self, *, allow_small: bool = False) -> None:
        # The serialized ABI owns six uint16 geometry fields.  Validate that
        # ABI *before* struct.pack so an invalid public shape always fails with
        # CodecError, never a platform struct.error.
        for name, value in (
                ("rows", self.rows), ("cols", self.cols),
                ("row_start", self.row_start), ("row_count", self.row_count),
                ("col_start", self.col_start), ("col_count", self.col_count)):
            require(isinstance(value, int) and 0 <= value <= UINT16_MAX,
                    f"{name} outside uint16 packet ABI")
        require(self.rows > 0 and self.cols > 0, "positive tensor shape")
        require(isinstance(self.role, int) and 0 <= self.role < 3, "role trit")
        require(is_power_of_two(self.row_count) and
                is_power_of_two(self.col_count), "tile powers of two")
        require(self.row_start % self.row_count == 0 and
                self.col_start % self.col_count == 0, "aligned tile")
        require(0 <= self.row_start <= self.rows - self.row_count and
                0 <= self.col_start <= self.cols - self.col_count,
                "tile in tensor")
        require(self.count <= MAX_BLOCK_WEIGHTS, "block cap")
        if not allow_small:
            require(self.count == MAX_BLOCK_WEIGHTS, "production gate block N=4096")


def _feature_columns(geometry: Geometry) -> tuple[list[str], np.ndarray]:
    """Return public mixed-radix coordinate bits in raster order.

    The domain is ``role(base 3) x row(base rows) x column(base cols)``.  Rows
    and columns are deliberately *not* restricted to powers of two.  Aligned
    power-of-two tiles still expose an exact Boolean cube to the algebraic
    families, while arbitrary SwiGLU widths remain representable in the
    packet header.
    """
    geometry.validate(allow_small=True)
    row_k = max(1, int(math.ceil(math.log2(geometry.rows))))
    col_k = max(1, int(math.ceil(math.log2(geometry.cols))))
    rr = np.repeat(
        np.arange(geometry.row_start, geometry.row_start + geometry.row_count,
                  dtype=np.uint32), geometry.col_count)
    cc = np.tile(
        np.arange(geometry.col_start, geometry.col_start + geometry.col_count,
                  dtype=np.uint32), geometry.row_count)
    names: list[str] = ["role.0", "role.1"]
    columns = [
        np.full(geometry.count, geometry.role & 1, dtype=np.uint8),
        np.full(geometry.count, (geometry.role >> 1) & 1, dtype=np.uint8),
    ]
    for bit in range(row_k):
        names.append(f"row_bin.{bit}")
        columns.append(((rr >> bit) & 1).astype(np.uint8))
    for bit in range(col_k):
        names.append(f"col_bin.{bit}")
        columns.append(((cc >> bit) & 1).astype(np.uint8))
    return names, np.stack(columns, axis=1)


def _order_key(name: str, order_id: int) -> tuple[int, int, str]:
    stem, number = name.split(".")
    bit = int(number)
    semantic = {"role": 0, "row_bin": 1, "col_bin": 2}[stem]
    if order_id == 0:  # semantic, most-significant first inside each radix
        return semantic, -bit, name
    if order_id == 1:  # polyphase/interleaved by significance
        return -bit, semantic, name
    if order_id == 2:  # column-first, least-significant first
        return (0 if stem == "col_bin" else 1 if stem == "row_bin" else 2), bit, name
    if order_id == 3:  # reversed semantic order
        return -semantic, bit, name
    raise CodecError("order selector")


def active_features(geometry: Geometry, order_id: int) -> tuple[list[str], np.ndarray]:
    require(0 <= order_id < ORDER_BANK_SIZE, "frozen order selector")
    names, values = _feature_columns(geometry)
    active = [j for j in range(values.shape[1])
              if int(values[:, j].min()) != int(values[:, j].max())]
    active.sort(key=lambda j: _order_key(names[j], order_id))
    selected = values[:, active]
    feature_names = [names[j] for j in active]
    require(selected.shape[1] <= MAX_ACTIVE_FEATURES, "active feature cap")
    require(1 << selected.shape[1] == geometry.count,
            "active coordinate bits must enumerate tile")
    packed = np.packbits(selected, axis=1, bitorder="little")
    require(np.unique(packed, axis=0).shape[0] == geometry.count,
            "coordinate function domain is bijective")
    return feature_names, selected


def validate_distortion_table(distortion: np.ndarray, n: int) -> np.ndarray:
    value = np.asarray(distortion, dtype=np.float64)
    require(value.shape == (n, 64), "exact D[i,0..63] required")
    require(np.isfinite(value).all() and (value >= 0).all(),
            "finite nonnegative distortion table")
    return value


def indices_to_planes(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices)
    require(values.ndim == 1 and values.size > 0 and
            np.issubdtype(values.dtype, np.integer), "integer 0..63 indices")
    require(int(values.min()) >= 0 and int(values.max()) < 64, "index range")
    return np.stack([((values.astype(np.uint8) >> level) & 1)
                     for level in range(6)], axis=0)


def planes_to_indices(planes: np.ndarray) -> np.ndarray:
    value = np.asarray(planes, dtype=np.uint8)
    require(value.ndim == 2 and value.shape[0] == 6,
            "six completed planes")
    require(((value == 0) | (value == 1)).all(), "binary completed planes")
    result = np.zeros(value.shape[1], dtype=np.uint8)
    for level in range(6):
        result |= value[level] << level
    return result


def pack_bits(bits: np.ndarray) -> bytes:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    require(((flat == 0) | (flat == 1)).all(), "packed binary bits")
    return np.packbits(flat, bitorder="little").tobytes()


def unpack_bits(payload: bytes, count: int) -> np.ndarray:
    require(count >= 0 and len(payload) == (count + 7) // 8, "packed length")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8),
                         bitorder="little")
    if count & 7:
        require(not bits[count:].any(), "nonzero packed tail")
    return bits[:count].astype(np.uint8, copy=True)


def bmp_plane(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    left = np.asarray(U, dtype=np.uint8)
    right = np.asarray(V, dtype=np.uint8)
    require(left.ndim == right.ndim == 2 and left.shape[1] == right.shape[1],
            "BMP factors")
    require(((left <= 1).all() and (right <= 1).all()), "BMP GF(2) bits")
    if left.shape[1] == 0:
        return np.zeros(left.shape[0] * right.shape[0], dtype=np.uint8)
    return ((left.astype(np.uint16) @ right.astype(np.uint16).T) & 1).astype(
        np.uint8).reshape(-1)


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Deterministic inverse of a small nonsingular GF(2) matrix."""
    value = np.asarray(matrix, dtype=np.uint8)
    require(value.ndim == 2 and value.shape[0] == value.shape[1],
            "GF2 inverse square")
    n = value.shape[0]
    augmented = np.concatenate([value.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        pivots = np.flatnonzero(augmented[col:, col])
        require(pivots.size > 0, "GF2 singular")
        pivot = col + int(pivots[0])
        if pivot != col:
            augmented[[col, pivot]] = augmented[[pivot, col]]
        for row in range(n):
            if row != col and augmented[row, col]:
                augmented[row] ^= augmented[col]
    require(np.array_equal(augmented[:, :n], np.eye(n, dtype=np.uint8)),
            "GF2 inverse closure")
    return augmented[:, n:]


def canonical_gf2_factor(plane: np.ndarray, rows: int, cols: int,
                         max_rank: int = MAX_BMP_RANK
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Unique minimum-rank ``U,V`` factor for a binary matrix.

    Columns are scanned in public raster order.  The lexicographically first
    independent matrix columns form ``U``; the lexicographically first pivot
    rows of ``U`` then define the unique coordinates in ``V``.  Consequently
    rank inflation, zero components, column swaps and every GL(r,2) gauge are
    rejected by strict re-encoding.
    """
    matrix = np.asarray(plane, dtype=np.uint8)
    require(matrix.shape == (rows * cols,) and
            ((matrix == 0) | (matrix == 1)).all(), "binary BMP plane")
    matrix = matrix.reshape(rows, cols)
    if not matrix.any():
        return (np.zeros((rows, 0), dtype=np.uint8),
                np.zeros((cols, 0), dtype=np.uint8))

    basis_columns: list[int] = []
    basis = np.zeros((rows, 0), dtype=np.uint8)
    current_rank = 0
    for col in range(cols):
        proposal = np.concatenate([basis, matrix[:, col:col + 1]], axis=1)
        # Tiny deterministic GF(2) rank; no floating-point rank decisions.
        work = proposal.copy()
        rank = 0
        for bit_col in range(work.shape[1]):
            pivots = np.flatnonzero(work[rank:, bit_col])
            if pivots.size:
                pivot = rank + int(pivots[0])
                work[[rank, pivot]] = work[[pivot, rank]]
                for row in range(rows):
                    if row != rank and work[row, bit_col]:
                        work[row] ^= work[rank]
                rank += 1
        if rank > current_rank:
            basis_columns.append(col)
            basis = proposal
            current_rank = rank
            require(current_rank <= max_rank, "BMP semantic rank cap")

    U = matrix[:, basis_columns].copy()
    rank = U.shape[1]
    pivot_rows: list[int] = []
    row_basis = np.zeros((0, rank), dtype=np.uint8)
    current_rank = 0
    for row in range(rows):
        proposal = np.concatenate([row_basis, U[row:row + 1]], axis=0)
        work = proposal.copy()
        r = 0
        for col in range(rank):
            pivots = np.flatnonzero(work[r:, col])
            if pivots.size:
                pivot = r + int(pivots[0])
                work[[r, pivot]] = work[[pivot, r]]
                for rr in range(work.shape[0]):
                    if rr != r and work[rr, col]:
                        work[rr] ^= work[r]
                r += 1
        if r > current_rank:
            pivot_rows.append(row)
            row_basis = proposal
            current_rank = r
            if current_rank == rank:
                break
    require(len(pivot_rows) == rank, "BMP pivot closure")
    inverse = _gf2_inverse(U[pivot_rows, :])
    coefficients = (inverse.astype(np.uint16) @
                    matrix[pivot_rows, :].astype(np.uint16)) & 1
    V = coefficients.T.astype(np.uint8)
    require(np.array_equal((U.astype(np.uint16) @ V.astype(np.uint16).T) & 1,
                           matrix), "BMP canonical factor closure")
    return U, V


def qtt_shapes(d: int, ranks: Iterable[int]) -> list[tuple[int, int, int]]:
    internal = tuple(int(value) for value in ranks)
    require(d >= 1 and len(internal) == max(0, d - 1), "QTT rank vector")
    require(all(1 <= value <= MAX_QTT_RANK for value in internal),
            "QTT rank cap")
    chain = (1,) + internal + (1,)
    return [(chain[axis], 2, chain[axis + 1]) for axis in range(d)]


def qtt_core_bit_count(d: int, ranks: Iterable[int]) -> int:
    return sum(int(np.prod(shape)) for shape in qtt_shapes(d, ranks))


def split_qtt_cores(bits: np.ndarray, d: int,
                    ranks: Iterable[int]) -> list[np.ndarray]:
    internal = tuple(int(value) for value in ranks)
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    require(flat.size == qtt_core_bit_count(d, internal),
            "QTT core bit count")
    result = []
    offset = 0
    for shape in qtt_shapes(d, internal):
        count = int(np.prod(shape))
        result.append(flat[offset:offset + count].reshape(shape).copy())
        offset += count
    return result


def qtt_plane(core_bits: np.ndarray, feature_bits: np.ndarray,
              ranks: Iterable[int]) -> np.ndarray:
    x = np.asarray(feature_bits, dtype=np.uint8)
    require(x.ndim == 2, "QTT feature matrix")
    internal = tuple(int(value) for value in ranks)
    cores = split_qtt_cores(core_bits, x.shape[1], internal)
    state = np.ones((x.shape[0], 1), dtype=np.uint8)
    rows = np.arange(x.shape[0])
    for axis, core in enumerate(cores):
        selected = np.transpose(core[:, x[:, axis], :], (1, 0, 2))
        state = (np.einsum("ni,nij->nj", state.astype(np.uint16),
                           selected.astype(np.uint16)) & 1).astype(np.uint8)
    require(state.shape[1] == 1, "QTT scalar output")
    return state[:, 0]


def _gf2_matrix_factor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical full-rank factor ``matrix = left @ right`` over GF(2)."""
    value = np.asarray(matrix, dtype=np.uint8)
    require(value.ndim == 2, "GF2 matrix")
    rows, cols = value.shape
    if not value.any():
        return (np.zeros((rows, 0), dtype=np.uint8),
                np.zeros((0, cols), dtype=np.uint8))
    selected: list[int] = []
    basis = np.zeros((rows, 0), dtype=np.uint8)
    current = 0
    for col in range(cols):
        proposal = np.concatenate([basis, value[:, col:col + 1]], axis=1)
        work = proposal.copy()
        rank = 0
        for axis in range(work.shape[1]):
            pivots = np.flatnonzero(work[rank:, axis])
            if pivots.size:
                pivot = rank + int(pivots[0])
                work[[rank, pivot]] = work[[pivot, rank]]
                for row in range(rows):
                    if row != rank and work[row, axis]:
                        work[row] ^= work[rank]
                rank += 1
        if rank > current:
            selected.append(col)
            basis = proposal
            current = rank
    left = value[:, selected].copy()
    rank = left.shape[1]
    pivot_rows: list[int] = []
    row_basis = np.zeros((0, rank), dtype=np.uint8)
    current = 0
    for row in range(rows):
        proposal = np.concatenate([row_basis, left[row:row + 1]], axis=0)
        work = proposal.copy()
        found = 0
        for col in range(rank):
            pivots = np.flatnonzero(work[found:, col])
            if pivots.size:
                pivot = found + int(pivots[0])
                work[[found, pivot]] = work[[pivot, found]]
                for rr in range(work.shape[0]):
                    if rr != found and work[rr, col]:
                        work[rr] ^= work[found]
                found += 1
        if found > current:
            pivot_rows.append(row)
            row_basis = proposal
            current = found
            if current == rank:
                break
    inverse = _gf2_inverse(left[pivot_rows, :])
    right = ((inverse.astype(np.uint16) @
              value[pivot_rows, :].astype(np.uint16)) & 1).astype(np.uint8)
    require(np.array_equal((left.astype(np.uint16) @
                            right.astype(np.uint16)) & 1, value),
            "GF2 factor closure")
    return left, right


def canonical_qtt(plane: np.ndarray, feature_bits: np.ndarray
                  ) -> tuple[tuple[int, ...], np.ndarray] | None:
    """Canonical minimum TT-rank representation of a Boolean truth table.

    A zero function has the sole representation ``None``.  Nonzero functions
    are decomposed by deterministic GF(2) rank factorizations at every cut;
    this fixes all bond gauges and rejects inflated ranks.  The gate admits a
    function only when every exact Schmidt rank is at most two.
    """
    target = np.asarray(plane, dtype=np.uint8).reshape(-1)
    features = np.asarray(feature_bits, dtype=np.uint8)
    require(features.ndim == 2 and features.shape[0] == target.size and
            ((target == 0) | (target == 1)).all(), "QTT truth table")
    d = features.shape[1]
    require(d >= 1 and (1 << d) == target.size, "complete QTT Boolean cube")
    tensor = np.zeros((2,) * d, dtype=np.uint8)
    tensor[tuple(features[:, axis] for axis in range(d))] = target
    if not tensor.any():
        return None
    residual = tensor.reshape(2, -1)
    previous_rank = 1
    ranks: list[int] = []
    cores: list[np.ndarray] = []
    for axis in range(d - 1):
        matrix = residual.reshape(previous_rank * 2, -1)
        left, right = _gf2_matrix_factor(matrix)
        rank = left.shape[1]
        require(1 <= rank <= MAX_QTT_RANK, "QTT semantic rank cap")
        ranks.append(rank)
        cores.append(left.reshape(previous_rank, 2, rank))
        residual = right.reshape(rank * 2, -1)
        previous_rank = rank
    require(residual.shape == (previous_rank * 2, 1), "QTT terminal shape")
    cores.append(residual.reshape(previous_rank, 2, 1).astype(np.uint8))
    packed = np.concatenate([core.reshape(-1) for core in cores]).astype(np.uint8)
    reconstructed = qtt_plane(packed, features, ranks)
    require(np.array_equal(reconstructed, target), "canonical QTT closure")
    return tuple(ranks), packed


def qtt_rank_code(ranks: Iterable[int] | None, d: int) -> int:
    """Zero is the unique zero-function code; positive codes own rank masks."""
    if ranks is None:
        return 0
    internal = tuple(int(value) for value in ranks)
    require(len(internal) == d - 1 and all(value in (1, 2) for value in internal),
            "QTT rank vector")
    mask = sum((value - 1) << axis for axis, value in enumerate(internal))
    code = 1 + mask
    require(code <= UINT16_MAX, "QTT rank code uint16")
    return code


def qtt_ranks_from_code(code: int, d: int) -> tuple[int, ...] | None:
    require(0 <= code <= UINT16_MAX, "QTT rank code")
    if code == 0:
        return None
    mask = code - 1
    require(mask < (1 << max(0, d - 1)), "QTT unused rank bits")
    return tuple(1 + ((mask >> axis) & 1) for axis in range(d - 1))


def eval_obdd(root: int, nodes: list[tuple[int, int, int]],
              feature_bits: np.ndarray) -> np.ndarray:
    features = np.asarray(feature_bits, dtype=np.uint8)
    result = np.empty(features.shape[0], dtype=np.uint8)
    for row in range(features.shape[0]):
        ref = int(root)
        steps = 0
        while ref >= 2:
            index = ref - 2
            require(index < len(nodes), "OBDD reference")
            variable, low, high = nodes[index]
            require(variable < features.shape[1], "OBDD variable")
            ref = high if features[row, variable] else low
            steps += 1
            require(steps <= len(nodes), "OBDD cycle")
        result[row] = ref
    return result


def build_robdd(bits: np.ndarray, feature_bits: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
    target = np.asarray(bits, dtype=np.uint8).reshape(-1)
    features = np.asarray(feature_bits, dtype=np.uint8)
    require(features.shape == (target.size, features.shape[1]) and
            1 << features.shape[1] == target.size, "complete OBDD domain")
    unique: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[int, int, int]] = []

    def visit(indices: np.ndarray, depth: int) -> int:
        first = int(target[indices[0]])
        if np.all(target[indices] == first):
            return first
        require(depth < features.shape[1], "OBDD leaf conflict")
        mask = features[indices, depth] == 0
        require(mask.any() and (~mask).any(), "OBDD incomplete branch")
        low = visit(indices[mask], depth + 1)
        high = visit(indices[~mask], depth + 1)
        if low == high:
            return low
        key = (depth, low, high)
        if key in unique:
            return unique[key]
        ref = len(nodes) + 2
        nodes.append(key)
        unique[key] = ref
        return ref

    root = visit(np.arange(target.size, dtype=np.int32), 0)
    require(len(nodes) <= 65533, "OBDD format node count")
    return root, nodes


def _encode_bmp(model: dict, geometry: Geometry) -> bytes:
    ranks = model["ranks"]
    factors = model["factors"]
    require(len(ranks) == len(factors) == 6, "six BMP planes")
    canonical_ranks = []
    for rank in ranks:
        require(isinstance(rank, (int, np.integer)) and
                0 <= int(rank) <= MAX_BMP_RANK, "BMP rank cap")
        canonical_ranks.append(int(rank))
    payload = bytearray(bytes(canonical_ranks))
    for rank, pair in zip(canonical_ranks, factors):
        U, V = pair
        require(np.asarray(U).shape == (geometry.row_count, rank) and
                np.asarray(V).shape == (geometry.col_count, rank),
                "BMP factor dimensions")
        decoded_plane = bmp_plane(U, V)
        canonical_u, canonical_v = canonical_gf2_factor(
            decoded_plane, geometry.row_count, geometry.col_count)
        require(rank == canonical_u.shape[1] and
                np.array_equal(np.asarray(U, dtype=np.uint8), canonical_u) and
                np.array_equal(np.asarray(V, dtype=np.uint8), canonical_v),
                "canonical minimum-rank BMP factors")
        payload.extend(pack_bits(np.concatenate([
            np.asarray(U, dtype=np.uint8).reshape(-1),
            np.asarray(V, dtype=np.uint8).reshape(-1)])))
    return bytes(payload)


def _decode_bmp(payload: bytes, geometry: Geometry) -> tuple[dict, int]:
    require(len(payload) >= 6, "BMP payload ranks")
    ranks = list(payload[:6])
    offset = 6
    factors = []
    for rank in ranks:
        require(rank <= MAX_BMP_RANK, "BMP rank cap")
        count = rank * (geometry.row_count + geometry.col_count)
        size = (count + 7) // 8
        bits = unpack_bits(payload[offset:offset + size], count)
        offset += size
        cut = geometry.row_count * rank
        factors.append((bits[:cut].reshape(geometry.row_count, rank),
                        bits[cut:].reshape(geometry.col_count, rank)))
    return {"ranks": ranks, "factors": factors}, offset


def _encode_obdd(model: dict, geometry: Geometry, feature_bits: np.ndarray) -> bytes:
    roots = model["roots"]
    diagrams = model["nodes"]
    require(len(roots) == len(diagrams) == 6, "six OBDD planes")
    require(sum(len(nodes) for nodes in diagrams) <= MAX_OBDD_NODES,
            "OBDD node cap")
    payload = bytearray()
    for root, nodes in zip(roots, diagrams):
        require(isinstance(root, (int, np.integer)) and
                0 <= int(root) <= UINT16_MAX, "OBDD root uint16")
        require(len(nodes) <= UINT16_MAX, "OBDD count uint16")
        payload.extend(struct.pack("<HH", int(root), len(nodes)))
        for variable, low, high in nodes:
            require(all(isinstance(value, (int, np.integer)) for value in
                        (variable, low, high)) and
                    0 <= int(variable) <= 255 and
                    0 <= int(low) <= UINT16_MAX and
                    0 <= int(high) <= UINT16_MAX,
                    "OBDD node packet range")
            payload.extend(NODE.pack(variable, low, high))
        decoded = eval_obdd(root, nodes, feature_bits)
        canonical_root, canonical_nodes = build_robdd(decoded, feature_bits)
        require(root == canonical_root and nodes == canonical_nodes,
                "canonical reduced OBDD")
    return bytes(payload)


def _decode_obdd(payload: bytes, geometry: Geometry,
                 feature_bits: np.ndarray) -> tuple[dict, int]:
    roots = []
    diagrams = []
    offset = 0
    for _ in range(6):
        require(offset + 4 <= len(payload), "OBDD plane header")
        root, count = struct.unpack_from("<HH", payload, offset)
        offset += 4
        require(count <= MAX_OBDD_NODES and
                offset + NODE.size * count <= len(payload), "OBDD node count")
        nodes = []
        for index in range(count):
            variable, low, high = NODE.unpack_from(payload, offset)
            offset += NODE.size
            require(variable < feature_bits.shape[1] and low < index + 2 and
                    high < index + 2 and low != high, "OBDD topological node")
            nodes.append((variable, low, high))
        require(root < count + 2, "OBDD root")
        decoded = eval_obdd(root, nodes, feature_bits)
        canonical_root, canonical_nodes = build_robdd(decoded, feature_bits)
        require(root == canonical_root and nodes == canonical_nodes,
                "canonical reduced OBDD")
        roots.append(root)
        diagrams.append(nodes)
    require(sum(len(nodes) for nodes in diagrams) <= MAX_OBDD_NODES,
            "OBDD total node cap")
    return {"roots": roots, "nodes": diagrams}, offset


def _encode_qtt(model: dict, feature_bits: np.ndarray) -> bytes:
    ranks = model["rank_vectors"]
    cores = model["cores"]
    require(len(ranks) == len(cores) == 6, "six QTT planes")
    d = feature_bits.shape[1]
    payload = bytearray()
    for rank_vector, bits in zip(ranks, cores):
        if rank_vector is None:
            require(np.asarray(bits, dtype=np.uint8).size == 0,
                    "zero QTT has no gauge bits")
            payload.extend(struct.pack("<H", 0))
            continue
        internal = tuple(int(value) for value in rank_vector)
        flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
        require(flat.size == qtt_core_bit_count(d, internal),
                "QTT descriptor count")
        plane = qtt_plane(flat, feature_bits, internal)
        canonical = canonical_qtt(plane, feature_bits)
        require(canonical is not None and canonical[0] == internal and
                np.array_equal(canonical[1], flat),
                "canonical minimum-rank QTT")
        payload.extend(struct.pack("<H", qtt_rank_code(internal, d)))
        payload.extend(pack_bits(flat))
    return bytes(payload)


def _decode_qtt(payload: bytes, feature_bits: np.ndarray) -> tuple[dict, int]:
    require(len(payload) >= 12, "QTT payload rank codes")
    d = feature_bits.shape[1]
    ranks = []
    cores = []
    offset = 0
    for _ in range(6):
        require(offset + 2 <= len(payload), "QTT rank code extent")
        code, = struct.unpack_from("<H", payload, offset)
        offset += 2
        internal = qtt_ranks_from_code(code, d)
        ranks.append(internal)
        if internal is None:
            cores.append(np.zeros(0, dtype=np.uint8))
            continue
        count = qtt_core_bit_count(d, internal)
        size = (count + 7) // 8
        flat = unpack_bits(payload[offset:offset + size], count)
        offset += size
        plane = qtt_plane(flat, feature_bits, internal)
        canonical = canonical_qtt(plane, feature_bits)
        require(canonical is not None and canonical[0] == internal and
                np.array_equal(canonical[1], flat),
                "canonical minimum-rank QTT")
        cores.append(flat)
    return {"rank_vectors": ranks, "cores": cores}, offset


def base_planes(family: int, model: dict, geometry: Geometry,
                order_id: int) -> np.ndarray:
    _, features = active_features(geometry, order_id)
    if family == FAMILY_BMP:
        return np.stack([bmp_plane(*pair) for pair in model["factors"]], axis=0)
    if family == FAMILY_OBDD:
        return np.stack([eval_obdd(root, nodes, features)
                         for root, nodes in zip(model["roots"], model["nodes"])],
                        axis=0)
    if family == FAMILY_QTT:
        rows = []
        for bits, rank_vector in zip(model["cores"], model["rank_vectors"]):
            if rank_vector is None:
                rows.append(np.zeros(geometry.count, dtype=np.uint8))
            else:
                rows.append(qtt_plane(bits, features, rank_vector))
        return np.stack(rows, axis=0)
    raise CodecError("family")


def _encode_model(family: int, model: dict, geometry: Geometry,
                  order_id: int) -> bytes:
    _, features = active_features(geometry, order_id)
    if family == FAMILY_BMP:
        return _encode_bmp(model, geometry)
    if family == FAMILY_OBDD:
        return _encode_obdd(model, geometry, features)
    if family == FAMILY_QTT:
        return _encode_qtt(model, features)
    raise CodecError("family")


def _decode_model(family: int, payload: bytes, geometry: Geometry,
                  order_id: int) -> tuple[dict, int]:
    _, features = active_features(geometry, order_id)
    if family == FAMILY_BMP:
        return _decode_bmp(payload, geometry)
    if family == FAMILY_OBDD:
        return _decode_obdd(payload, geometry, features)
    if family == FAMILY_QTT:
        return _decode_qtt(payload, features)
    raise CodecError("family")


def encode_packet(family: int, order_id: int, geometry: Geometry, model: dict,
                  exceptions: Iterable[tuple[int, int]]) -> bytes:
    geometry.validate(allow_small=True)
    require(isinstance(family, int) and family in FAMILY_NAMES,
            "family selector")
    require(isinstance(order_id, int) and 0 <= order_id < ORDER_BANK_SIZE,
            "order selector")
    feature_names, _ = active_features(geometry, order_id)
    exception_rows = [(int(position), int(index)) for position, index in exceptions]
    require(len(exception_rows) <= MAX_EXCEPTIONS, "exception cap")
    require(exception_rows == sorted(exception_rows) and
            len({position for position, _ in exception_rows}) == len(exception_rows),
            "sorted unique exceptions")
    for position, index in exception_rows:
        require(0 <= position < geometry.count and 0 <= index < 64,
                "exception range")
    model_payload = _encode_model(family, model, geometry, order_id)
    require(len(model_payload) <= (1 << 32) - 1, "model payload uint32")
    base = planes_to_indices(base_planes(family, model, geometry, order_id))
    for position, index in exception_rows:
        require(int(base[position]) != index, "redundant exception")
    payload = bytearray(model_payload)
    for row in exception_rows:
        payload.extend(EXCEPTION.pack(*row))
    header = HEADER.pack(
        MAGIC, VERSION, family, order_id, geometry.role, geometry.rows,
        geometry.cols, geometry.row_start, geometry.row_count,
        geometry.col_start, geometry.col_count, len(feature_names),
        len(exception_rows), len(payload))
    body = header + bytes(payload)
    return body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_packet(packet: bytes, *, allow_small: bool = False) -> dict:
    require(isinstance(packet, bytes) and len(packet) >= HEADER.size + CRC.size,
            "packet bytes")
    expected_crc, = CRC.unpack_from(packet, len(packet) - CRC.size)
    body = packet[:-CRC.size]
    require(zlib.crc32(body) & 0xFFFFFFFF == expected_crc, "CRC32")
    values = HEADER.unpack_from(body, 0)
    (magic, version, family, order_id, role, rows, cols, row_start, row_count,
     col_start, col_count, feature_count, exception_count, payload_bytes) = values
    require(magic == MAGIC and version == VERSION, "magic/version")
    require(family in FAMILY_NAMES and 0 <= order_id < ORDER_BANK_SIZE,
            "family/order")
    geometry = Geometry(rows, cols, role, row_start, row_count,
                        col_start, col_count)
    geometry.validate(allow_small=allow_small)
    names, _ = active_features(geometry, order_id)
    require(feature_count == len(names), "active feature receipt")
    require(exception_count <= MAX_EXCEPTIONS and
            len(body) == HEADER.size + payload_bytes, "payload extent")
    payload = body[HEADER.size:]
    model, offset = _decode_model(family, payload, geometry, order_id)
    require(len(payload) - offset == exception_count * EXCEPTION.size,
            "exception payload extent")
    exceptions = []
    for _ in range(exception_count):
        exceptions.append(EXCEPTION.unpack_from(payload, offset))
        offset += EXCEPTION.size
    require(exceptions == sorted(exceptions) and
            len({position for position, _ in exceptions}) == len(exceptions),
            "canonical exception order")
    planes = base_planes(family, model, geometry, order_id)
    indices = planes_to_indices(planes)
    for position, index in exceptions:
        require(position < geometry.count and index < 64 and
                int(indices[position]) != index, "exception canonicality")
        indices[position] = index
    completed = indices_to_planes(indices)
    reencoded = encode_packet(family, order_id, geometry, model, exceptions)
    require(reencoded == packet, "canonical re-encode")
    return {
        "family": family,
        "family_name": FAMILY_NAMES[family],
        "order_id": order_id,
        "geometry": geometry,
        "feature_names": names,
        "model": model,
        "exceptions": exceptions,
        "base_planes": planes,
        "completed_planes": completed,
        "indices": indices,
        "packet_bytes": len(packet),
        "physical_bits": len(packet) * 8,
        "physical_rate_bpw": len(packet) * 8 / geometry.count,
        "packet_sha256": sha256(packet),
        "plane_sha256": [sha256(np.ascontiguousarray(row).tobytes())
                          for row in completed],
        "index_sha256": sha256(np.ascontiguousarray(indices).tobytes()),
    }


def descriptor_formula(decoded: dict) -> dict:
    geometry: Geometry = decoded["geometry"]
    family = decoded["family"]
    model = decoded["model"]
    exceptions = len(decoded["exceptions"])
    fixed = 8 * (HEADER.size + CRC.size)
    if family == FAMILY_BMP:
        component_bytes = [
            (rank * (geometry.row_count + geometry.col_count) + 7) // 8
            for rank in model["ranks"]]
        model_bytes = 6 + sum(component_bytes)
        symbolic = (
            "8*(HEADER[30]+CRC[4]+6 rank bytes+"
            "sum_l ceil(r_l*(row_count+col_count)/8)+3*E)")
    elif family == FAMILY_OBDD:
        counts = [len(nodes) for nodes in model["nodes"]]
        model_bytes = sum(4 + NODE.size * count for count in counts)
        symbolic = "8*(HEADER[30]+CRC[4]+sum_l(4+5*n_l)+3*E)"
        component_bytes = [4 + NODE.size * count for count in counts]
    else:
        d = len(decoded["feature_names"])
        component_bytes = [
            2 if rank_vector is None else
            2 + (qtt_core_bit_count(d, rank_vector) + 7) // 8
            for rank_vector in model["rank_vectors"]]
        model_bytes = sum(component_bytes)
        symbolic = (
            "8*(HEADER[30]+CRC[4]+"
            "sum_l(2-byte canonical rank code+ceil(core_bits_l/8))+3*E)")
    calculated = fixed + 8 * model_bytes + 8 * EXCEPTION.size * exceptions
    require(calculated == decoded["physical_bits"], "descriptor formula closure")
    return {
        "symbolic": symbolic,
        "fixed_header_and_crc_bits": fixed,
        "component_model_bytes": component_bytes,
        "model_payload_bytes": model_bytes,
        "exception_bytes": EXCEPTION.size * exceptions,
        "selector_physical_header_bits": 16,
        "total_physical_bits": calculated,
    }
