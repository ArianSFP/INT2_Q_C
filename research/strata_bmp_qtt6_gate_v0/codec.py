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


MAGIC = b"SQF6V0\x00\x00"
VERSION = 0
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
        require(self.rows % 3 == 0 and is_power_of_two(self.rows // 3),
                "rows must have mixed radix 3*2^k")
        require(is_power_of_two(self.cols), "hidden width must be power of two")
        require(0 <= self.role < 3, "role trit")
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
    """Return full mixed-radix coordinate bits in canonical raster order."""
    geometry.validate(allow_small=True)
    row_k = int(math.log2(geometry.rows // 3))
    col_k = int(math.log2(geometry.cols))
    rr = np.repeat(
        np.arange(geometry.row_start, geometry.row_start + geometry.row_count,
                  dtype=np.uint32), geometry.col_count)
    cc = np.tile(
        np.arange(geometry.col_start, geometry.col_start + geometry.col_count,
                  dtype=np.uint32), geometry.row_count)
    row_trit = rr >> row_k
    row_low = rr & ((1 << row_k) - 1)
    names: list[str] = ["role.0", "role.1", "row_trit.0", "row_trit.1"]
    columns = [
        np.full(geometry.count, geometry.role & 1, dtype=np.uint8),
        np.full(geometry.count, (geometry.role >> 1) & 1, dtype=np.uint8),
        (row_trit & 1).astype(np.uint8),
        ((row_trit >> 1) & 1).astype(np.uint8),
    ]
    for bit in range(row_k):
        names.append(f"row_bin.{bit}")
        columns.append(((row_low >> bit) & 1).astype(np.uint8))
    for bit in range(col_k):
        names.append(f"col_bin.{bit}")
        columns.append(((cc >> bit) & 1).astype(np.uint8))
    return names, np.stack(columns, axis=1)


def _order_key(name: str, order_id: int) -> tuple[int, int, str]:
    stem, number = name.split(".")
    bit = int(number)
    semantic = {"role": 0, "row_trit": 1, "row_bin": 2, "col_bin": 3}[stem]
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


def qtt_core_bit_count(d: int, rank: int) -> int:
    require(d >= 1 and 1 <= rank <= MAX_QTT_RANK, "QTT rank")
    if d == 1:
        return 2
    return 4 * rank + 2 * (d - 2) * rank * rank


def qtt_shapes(d: int, rank: int) -> list[tuple[int, int, int]]:
    if d == 1:
        return [(1, 2, 1)]
    return [(1, 2, rank)] + [(rank, 2, rank)] * (d - 2) + [(rank, 2, 1)]


def split_qtt_cores(bits: np.ndarray, d: int, rank: int) -> list[np.ndarray]:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    require(flat.size == qtt_core_bit_count(d, rank), "QTT core bit count")
    result = []
    offset = 0
    for shape in qtt_shapes(d, rank):
        count = int(np.prod(shape))
        result.append(flat[offset:offset + count].reshape(shape).copy())
        offset += count
    return result


def qtt_plane(core_bits: np.ndarray, feature_bits: np.ndarray, rank: int) -> np.ndarray:
    x = np.asarray(feature_bits, dtype=np.uint8)
    require(x.ndim == 2, "QTT feature matrix")
    cores = split_qtt_cores(core_bits, x.shape[1], rank)
    state = np.ones((x.shape[0], 1), dtype=np.uint8)
    rows = np.arange(x.shape[0])
    for axis, core in enumerate(cores):
        selected = np.transpose(core[:, x[:, axis], :], (1, 0, 2))
        state = (np.einsum("ni,nij->nj", state.astype(np.uint16),
                           selected.astype(np.uint16)) & 1).astype(np.uint8)
    require(state.shape[1] == 1, "QTT scalar output")
    return state[:, 0]


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
    payload = bytearray(bytes(ranks))
    for rank, pair in zip(ranks, factors):
        require(0 <= rank <= MAX_BMP_RANK, "BMP rank cap")
        U, V = pair
        require(np.asarray(U).shape == (geometry.row_count, rank) and
                np.asarray(V).shape == (geometry.col_count, rank),
                "BMP factor dimensions")
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
        payload.extend(struct.pack("<HH", int(root), len(nodes)))
        for variable, low, high in nodes:
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


def _encode_qtt(model: dict, d: int) -> bytes:
    ranks = model["ranks"]
    cores = model["cores"]
    require(len(ranks) == len(cores) == 6, "six QTT planes")
    payload = bytearray(bytes(ranks))
    for rank, bits in zip(ranks, cores):
        require(1 <= rank <= MAX_QTT_RANK, "QTT rank cap")
        require(np.asarray(bits).size == qtt_core_bit_count(d, rank),
                "QTT descriptor count")
        payload.extend(pack_bits(bits))
    return bytes(payload)


def _decode_qtt(payload: bytes, d: int) -> tuple[dict, int]:
    require(len(payload) >= 6, "QTT payload ranks")
    ranks = list(payload[:6])
    offset = 6
    cores = []
    for rank in ranks:
        require(1 <= rank <= MAX_QTT_RANK, "QTT rank cap")
        count = qtt_core_bit_count(d, rank)
        size = (count + 7) // 8
        cores.append(unpack_bits(payload[offset:offset + size], count))
        offset += size
    return {"ranks": ranks, "cores": cores}, offset


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
        return np.stack([qtt_plane(bits, features, rank)
                         for bits, rank in zip(model["cores"], model["ranks"])],
                        axis=0)
    raise CodecError("family")


def _encode_model(family: int, model: dict, geometry: Geometry,
                  order_id: int) -> bytes:
    _, features = active_features(geometry, order_id)
    if family == FAMILY_BMP:
        return _encode_bmp(model, geometry)
    if family == FAMILY_OBDD:
        return _encode_obdd(model, geometry, features)
    if family == FAMILY_QTT:
        return _encode_qtt(model, features.shape[1])
    raise CodecError("family")


def _decode_model(family: int, payload: bytes, geometry: Geometry,
                  order_id: int) -> tuple[dict, int]:
    _, features = active_features(geometry, order_id)
    if family == FAMILY_BMP:
        return _decode_bmp(payload, geometry)
    if family == FAMILY_OBDD:
        return _decode_obdd(payload, geometry, features)
    if family == FAMILY_QTT:
        return _decode_qtt(payload, features.shape[1])
    raise CodecError("family")


def encode_packet(family: int, order_id: int, geometry: Geometry, model: dict,
                  exceptions: Iterable[tuple[int, int]]) -> bytes:
    geometry.validate(allow_small=True)
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
        component_bytes = [(qtt_core_bit_count(d, rank) + 7) // 8
                           for rank in model["ranks"]]
        model_bytes = 6 + sum(component_bytes)
        symbolic = (
            "8*(HEADER[30]+CRC[4]+6 rank bytes+"
            "sum_l ceil((4*r_l+2*(d-2)*r_l^2)/8)+3*E)")
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
