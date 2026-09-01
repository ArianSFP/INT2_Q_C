#!/usr/bin/env python3
"""Frozen CuPy CCQ Code-Cluster raw-source-MSE stage-0 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


AUTHORIZATION = "OPEN_AUTHENTICATED_18_MATRIX_PANEL_FOR_CCQ_RAW_MSE_STAGE0_V0"
SOURCE_LOCK_RELPATH = Path("blind_protocol_v2/unblinded/source_hashes.lock.json")
SOURCE_LOCK_SHA256 = "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23"
SOURCE_LOCK_BYTES = 46013
SOURCE_LOCK_INTERNAL = "5a82dac742110d4f48bbd73ae82081e1622b10b660b7850dadfe613ff475cc5b"

ROWS = 768
COLS = 2048
ROLES = ("gate", "up", "down")
EXPERTS = 6
MATRICES = 18
VALUES_PER_MATRIX = ROWS * COLS
VALUES_PER_EXPERT = 3 * VALUES_PER_MATRIX
PANEL_VALUES = EXPERTS * VALUES_PER_EXPERT
FIT_SLOTS = (0, 2, 3, 5)
HOLDOUT_SLOTS = (1, 4)
EXPECTED_SLOTS = ((5, 18), (12, 7), (18, 20), (28, 83), (36, 76), (45, 41))

GROUP = 64
VECTOR = 4
STATE_COUNT = 64
BYTE_CODES = 256
SHIFTS = (9, 6, 3, 0)
CONTINUOUS_PASSES = 3
CLUSTER_PASSES = 2
VITERBI_BATCH = 65536
COLUMN_TILE = 64
GAUSSIAN_SEED_BASE = 250707145

GLOBAL_HEADER_BYTES = 4096
EXPERT_HEADER_BYTES = 64
INDEX_BYTES_PER_MATRIX = VALUES_PER_MATRIX // VECTOR
LOCAL_SCALE_BYTES_PER_MATRIX = VALUES_PER_MATRIX // 128
INDEX_BYTES_PER_EXPERT = 3 * INDEX_BYTES_PER_MATRIX
LOCAL_SCALE_BYTES_PER_EXPERT = 3 * LOCAL_SCALE_BYTES_PER_MATRIX
OUTPUT_CHANNELS = (768, 768, 2048)
CODE_FLOAT32_BYTES_PER_EXPERT = sum(OUTPUT_CHANNELS) * 2 * 4
SUPER_FP16_BYTES_PER_EXPERT = sum(OUTPUT_CHANNELS) * 2
PARAMETER_BYTES_PER_EXPERT = CODE_FLOAT32_BYTES_PER_EXPERT + SUPER_FP16_BYTES_PER_EXPERT
EXPERT_FIXED_BYTES = (
    EXPERT_HEADER_BYTES + INDEX_BYTES_PER_EXPERT + LOCAL_SCALE_BYTES_PER_EXPERT + PARAMETER_BYTES_PER_EXPERT
)
PANEL_FIXED_BYTES = GLOBAL_HEADER_BYTES + EXPERTS * EXPERT_FIXED_BYTES

RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
PROMOTION_S = TARGET_S + 0.02


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {name}")
    return result


def validate_source_lock(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.resolve()
    lock_path = (root / SOURCE_LOCK_RELPATH).resolve()
    try:
        lock_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("source lock escaped root") from exc
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError("source lock is missing, non-file, or symlink")
    if lock_path.stat().st_size != SOURCE_LOCK_BYTES or sha256_file(lock_path) != SOURCE_LOCK_SHA256:
        raise RuntimeError("source lock byte identity changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "int2-qwen-blind-source-finalization-v2":
        raise RuntimeError("source lock schema changed")
    if lock.get("lock_sha256") != SOURCE_LOCK_INTERNAL:
        raise RuntimeError("source lock internal identity changed")
    if lock.get("matrix_count") != MATRICES or lock.get("source_values") != PANEL_VALUES:
        raise RuntimeError("source lock panel geometry changed")
    rows = lock.get("matrices")
    if not isinstance(rows, list) or len(rows) != MATRICES:
        raise RuntimeError("source lock matrix list changed")
    for ordinal, row in enumerate(rows):
        slot = ordinal // 3
        role = ROLES[ordinal % 3]
        layer, expert = EXPECTED_SLOTS[slot]
        expected_shape = [2048, 768] if role == "down" else [768, 2048]
        expected = {
            "matrix_ordinal": ordinal,
            "layer": layer,
            "expert": expert,
            "role": role,
            "shape": expected_shape,
            "nvalues": VALUES_PER_MATRIX,
            "nbytes": VALUES_PER_MATRIX * 2,
            "dtype": "BF16",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise RuntimeError(f"source lock mismatch {ordinal}:{key}")
    return lock_path, lock


def load_sources(cp: Any, lock_path: Path, lock: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    source_root = lock_path.parent.resolve()
    arrays: list[Any] = []
    receipts: list[dict[str, Any]] = []
    for row in lock["matrices"]:
        ordinal = int(row["matrix_ordinal"])
        path = (source_root / str(row["output_relpath"])).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError("source payload path escaped authenticated lock parent") from exc
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source payload missing, non-file, or symlink at {ordinal}")
        observed = sha256_file(path)
        if path.stat().st_size != VALUES_PER_MATRIX * 2 or observed != row["source_bf16_sha256"]:
            raise RuntimeError(f"source payload byte identity changed at {ordinal}")
        words = np.fromfile(path, dtype="<u2")
        if words.size != VALUES_PER_MATRIX:
            raise RuntimeError(f"source payload length changed at {ordinal}")
        host = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
        host = host.reshape(tuple(int(item) for item in row["shape"]))
        if row["role"] == "down":
            host = host.T
        host = np.ascontiguousarray(host, dtype=np.float32)
        if host.shape != (ROWS, COLS) or not np.isfinite(host).all():
            raise RuntimeError(f"source payload values invalid at {ordinal}")
        arrays.append(cp.asarray(host))
        receipts.append(
            {
                "matrix_ordinal": ordinal,
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "role": str(row["role"]),
                "bytes": int(path.stat().st_size),
                "declared_sha256": str(row["source_bf16_sha256"]),
                "observed_sha256": observed,
            }
        )
    return arrays, receipts


def gaussian_controls(cp: Any, source: list[Any]) -> tuple[list[Any], list[dict[str, float]]]:
    controls: list[Any] = []
    reports: list[dict[str, float]] = []
    for ordinal, values in enumerate(source):
        rng = cp.random.RandomState(GAUSSIAN_SEED_BASE + ordinal)
        z = rng.standard_normal(values.shape).astype(cp.float32)
        z_mean = cp.mean(z, axis=1, keepdims=True, dtype=cp.float64)
        centered = z - z_mean.astype(cp.float32)
        z_std = cp.sqrt(cp.mean(centered.astype(cp.float64) ** 2, axis=1, keepdims=True))
        source_mean = cp.mean(values, axis=1, keepdims=True, dtype=cp.float64)
        source_centered = values.astype(cp.float64) - source_mean
        source_std = cp.sqrt(cp.mean(source_centered * source_centered, axis=1, keepdims=True))
        control = centered * (source_std / z_std).astype(cp.float32) + source_mean.astype(cp.float32)
        control = cp.ascontiguousarray(control, dtype=cp.float32)
        observed_mean = cp.mean(control, axis=1, keepdims=True, dtype=cp.float64)
        observed_centered = control.astype(cp.float64) - observed_mean
        observed_std = cp.sqrt(cp.mean(observed_centered * observed_centered, axis=1, keepdims=True))
        mean_error = finite(cp.asnumpy(cp.max(cp.abs(observed_mean - source_mean))), "Gaussian mean error")
        std_error = finite(cp.asnumpy(cp.max(cp.abs(observed_std - source_std))), "Gaussian RMS error")
        controls.append(control)
        reports.append({"matrix_ordinal": ordinal, "max_abs_mean_error": mean_error, "max_abs_centered_rms_error": std_error})
    return controls, reports


def released_orientation(cp: Any, natural: Any, role: str) -> Any:
    if role in ("gate", "up"):
        result = cp.ascontiguousarray(natural.T, dtype=cp.float32)
        expected = (2048, 768)
    else:
        result = cp.ascontiguousarray(natural, dtype=cp.float32)
        expected = (768, 2048)
    if result.shape != expected or result.shape[0] % 128 != 0:
        raise RuntimeError(f"released orientation mismatch for {role}")
    return result


def vector_view(matrix: Any) -> Any:
    k, n = matrix.shape
    return matrix.reshape(k // GROUP, GROUP // VECTOR, VECTOR, n).transpose(0, 1, 3, 2).reshape(-1, n, VECTOR)


def decoded_states(cp: Any, codes: Any) -> Any:
    return cp.stack([((codes >> shift) & 63).astype(cp.int16) - 32 for shift in SHIFTS], axis=-1)


def viterbi_codes(cp: Any, matrix: Any, scales: Any) -> Any:
    """Exact nearest overlapping-window 15-bit code for each scaled four-vector."""
    k, n = matrix.shape
    groups = k // GROUP
    vectors = vector_view(matrix).reshape(-1, VECTOR)
    vector_scales = cp.broadcast_to(scales[:, None, :], (groups, GROUP // VECTOR, n)).reshape(-1)
    states = cp.arange(STATE_COUNT, dtype=cp.float32)
    output = cp.empty(vectors.shape[0], dtype=cp.uint16)
    for start in range(0, vectors.shape[0], VITERBI_BATCH):
        stop = min(start + VITERBI_BATCH, vectors.shape[0])
        positive = vector_scales[start:stop] > 0
        safe_scale = cp.where(positive, vector_scales[start:stop], cp.float32(1.0))
        target = vectors[start:stop] / safe_scale[:, None]
        cost = (states[None, :] - (target[:, 0, None] + cp.float32(32.0))) ** 2
        predecessors: list[Any] = []
        for position in range(1, VECTOR):
            shaped = cost.reshape(cost.shape[0], 8, 8)
            arg_high = cp.argmin(shaped, axis=1).astype(cp.uint8)
            best = cp.min(shaped, axis=1)
            overlap = (cp.arange(STATE_COUNT, dtype=cp.int16) >> 3).astype(cp.int16)
            cost = best[:, overlap] + (states[None, :] - (target[:, position, None] + cp.float32(32.0))) ** 2
            predecessors.append(arg_high)
        row = cp.arange(cost.shape[0], dtype=cp.int64)
        s3 = cp.argmin(cost, axis=1).astype(cp.int16)
        overlap3 = s3 >> 3
        s2 = predecessors[2][row, overlap3].astype(cp.int16) * 8 + overlap3
        overlap2 = s2 >> 3
        s1 = predecessors[1][row, overlap2].astype(cp.int16) * 8 + overlap2
        overlap1 = s1 >> 3
        s0 = predecessors[0][row, overlap1].astype(cp.int16) * 8 + overlap1
        code = (s0.astype(cp.uint16) << 9) | ((s1.astype(cp.uint16) & 7) << 6)
        code |= ((s2.astype(cp.uint16) & 7) << 3) | (s3.astype(cp.uint16) & 7)
        output[start:stop] = cp.where(positive, code, cp.uint16(0))
    return output.reshape(groups, GROUP // VECTOR, n)


def states_to_matrix(cp: Any, states: Any, k: int, n: int) -> Any:
    return cp.ascontiguousarray(states.reshape(k // GROUP, GROUP // VECTOR, n, VECTOR).transpose(0, 1, 3, 2).reshape(k, n))


def least_squares_scales(cp: Any, matrix: Any, state_matrix: Any) -> Any:
    k, n = matrix.shape
    w = matrix.reshape(k // GROUP, GROUP, n).astype(cp.float64)
    z = state_matrix.reshape(k // GROUP, GROUP, n).astype(cp.float64)
    numerator = cp.sum(w * z, axis=1)
    denominator = cp.sum(z * z, axis=1)
    result = cp.maximum(numerator / cp.maximum(denominator, 1.0e-30), 1.0e-12)
    return result.astype(cp.float32)


def initialize_scales(cp: Any, matrix: Any) -> Any:
    k, n = matrix.shape
    rms = cp.sqrt(cp.mean(matrix.reshape(k // GROUP, GROUP, n).astype(cp.float64) ** 2, axis=1))
    code_rms = math.sqrt(sum((value - 32) ** 2 for value in range(64)) / 64.0)
    return cp.maximum((rms / code_rms).astype(cp.float32), cp.float32(1.0e-12))


def quantize_scales(cp: Any, continuous: Any) -> tuple[Any, Any, Any]:
    super_scale = cp.max(continuous.astype(cp.float64), axis=0) / 15.0
    super_scale = cp.maximum(super_scale, 1.0e-12)
    for _ in range(6):
        q = cp.clip(cp.rint(continuous / super_scale[None, :]), 0, 15).astype(cp.uint8)
        numerator = cp.sum(continuous.astype(cp.float64) * q.astype(cp.float64), axis=0)
        denominator = cp.sum(q.astype(cp.float64) ** 2, axis=0)
        super_scale = cp.maximum(numerator / cp.maximum(denominator, 1.0), 1.0e-12)
    canonical = cp.asarray(cp.asnumpy(super_scale).astype("<f2").astype(np.float32))
    q = cp.clip(cp.rint(continuous / canonical[None, :]), 0, 15).astype(cp.uint8)
    decoded = q.astype(cp.float32) * canonical[None, :]
    return q, canonical, decoded


def cluster_fields(cp: Any, codes: Any) -> tuple[Any, Any]:
    minimum = cp.min(codes, axis=(0, 1)).astype(cp.float32)
    maximum = cp.max(codes, axis=(0, 1)).astype(cp.float32)
    scale = cp.where(maximum > minimum, (maximum - minimum) / cp.float32(255.0), cp.float32(1.0))
    return cp.ascontiguousarray(scale, dtype=cp.float32), cp.ascontiguousarray(minimum, dtype=cp.float32)


def fmaf_decoder(cp: Any) -> Any:
    return cp.ElementwiseKernel(
        "uint8 q, float32 code_scale, float32 code_zp",
        "int32 decoded",
        "decoded = __float2int_rd(fmaf((float)q, code_scale, code_zp + 0.5f));",
        "ccq_stage0_fmaf_decode_v0",
    )


def code_table(cp: Any, decoder: Any, code_scale: Any, code_zp: Any) -> Any:
    byte = cp.arange(BYTE_CODES, dtype=cp.uint8)[:, None]
    return decoder(byte, code_scale[None, :], code_zp[None, :])


def assign_cluster_bytes(cp: Any, decoder: Any, matrix: Any, decoded_scale: Any, code_scale: Any, code_zp: Any) -> Any:
    k, n = matrix.shape
    groups = k // GROUP
    vectors = vector_view(matrix).reshape(groups * (GROUP // VECTOR), n, VECTOR)
    vector_scales = cp.broadcast_to(decoded_scale[:, None, :], (groups, GROUP // VECTOR, n)).reshape(-1, n)
    result = cp.empty((vectors.shape[0], n), dtype=cp.uint8)
    for column in range(0, n, COLUMN_TILE):
        stop = min(column + COLUMN_TILE, n)
        scale = vector_scales[:, column:stop]
        target = vectors[:, column:stop, :]
        table_codes = code_table(cp, decoder, code_scale[column:stop], code_zp[column:stop])
        table = decoded_states(cp, table_codes).astype(cp.float32)
        reconstruction = scale[:, None, :, None] * table[None, :, :, :]
        distance = cp.sum((target[:, None, :, :] - reconstruction) ** 2, axis=3)
        result[:, column:stop] = cp.argmin(distance, axis=1).astype(cp.uint8)
        del target, table_codes, table, reconstruction, distance
    return result.reshape(groups, GROUP // VECTOR, n)


def bytes_to_state_matrix(cp: Any, decoder: Any, indices: Any, code_scale: Any, code_zp: Any, k: int, n: int) -> Any:
    codes = decoder(indices, code_scale[None, None, :], code_zp[None, None, :])
    return states_to_matrix(cp, decoded_states(cp, codes), k, n)


def encode_matrix(cp: Any, decoder: Any, matrix: Any, ordinal: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    k, n = matrix.shape
    scales = initialize_scales(cp, matrix)
    trace: list[dict[str, Any]] = []
    codes = None
    for pass_index in range(CONTINUOUS_PASSES):
        codes = viterbi_codes(cp, matrix, scales)
        state_matrix = states_to_matrix(cp, decoded_states(cp, codes), k, n)
        scales = least_squares_scales(cp, matrix, state_matrix)
        recon = state_matrix.astype(cp.float32) * scales.repeat(GROUP, axis=0)
        q = finite(cp.asnumpy(cp.sum((matrix.astype(cp.float64) - recon.astype(cp.float64)) ** 2) / cp.sum(matrix.astype(cp.float64) ** 2)), "continuous q")
        trace.append({"stage": f"continuous_{pass_index + 1}", "relative_residual_energy": q})
    assert codes is not None
    qscale, super_scale, decoded_scale = quantize_scales(cp, scales)
    codes = viterbi_codes(cp, matrix, decoded_scale)
    code_scale, code_zp = cluster_fields(cp, codes)
    indices = None
    for pass_index in range(CLUSTER_PASSES):
        indices = assign_cluster_bytes(cp, decoder, matrix, decoded_scale, code_scale, code_zp)
        state_matrix = bytes_to_state_matrix(cp, decoder, indices, code_scale, code_zp, k, n)
        continuous = least_squares_scales(cp, matrix, state_matrix)
        qscale, super_scale, decoded_scale = quantize_scales(cp, continuous)
        recon = state_matrix.astype(cp.float32) * decoded_scale.repeat(GROUP, axis=0)
        q = finite(cp.asnumpy(cp.sum((matrix.astype(cp.float64) - recon.astype(cp.float64)) ** 2) / cp.sum(matrix.astype(cp.float64) ** 2)), "cluster q")
        trace.append({"stage": f"cluster_refine_{pass_index + 1}", "relative_residual_energy": q})
    indices = assign_cluster_bytes(cp, decoder, matrix, decoded_scale, code_scale, code_zp)
    fields = {
        "indices": cp.asnumpy(indices.reshape(k // VECTOR, n)).astype(np.uint8, copy=False),
        "local_scale": cp.asnumpy(qscale).astype(np.uint8, copy=False),
        "code_scale": cp.asnumpy(code_scale).astype("<f4", copy=False),
        "code_zp": cp.asnumpy(code_zp).astype("<f4", copy=False),
        "super_scale": cp.asnumpy(super_scale).astype("<f2", copy=False),
    }
    return fields, {"matrix_ordinal": ordinal, "K": k, "N": n, "trace": trace}


def pack_local_scale(values: np.ndarray) -> bytes:
    if values.ndim != 2 or values.shape[0] % 2 or np.any(values > 15):
        raise RuntimeError("invalid uint4 local scale field")
    packed = values[0::2] | (values[1::2] << np.uint8(4))
    return np.ascontiguousarray(packed, dtype=np.uint8).tobytes(order="C")


def unpack_local_scale(payload: bytes, groups: int, n: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8).copy().reshape(groups // 2, n)
    values = np.empty((groups, n), dtype=np.uint8)
    values[0::2] = packed & np.uint8(15)
    values[1::2] = packed >> np.uint8(4)
    return values


def serialize_prefix(records: list[dict[str, np.ndarray]], label: str) -> bytes:
    if len(records) != MATRICES:
        raise RuntimeError("prefix record count changed")
    metadata = canonical_json_bytes(
        {"format": "CCQ-RMSE-S0-v0", "label": label, "experts": EXPERTS, "matrices": MATRICES}
    )
    header = struct.pack("<8sIIII", b"CCQRM0\0\0", 0, EXPERTS, MATRICES, EXPERT_FIXED_BYTES) + metadata
    if len(header) > GLOBAL_HEADER_BYTES:
        raise RuntimeError("global header overflow")
    output = bytearray(header + bytes(GLOBAL_HEADER_BYTES - len(header)))
    for slot in range(EXPERTS):
        layer, expert = EXPECTED_SLOTS[slot]
        local_header = struct.pack("<8sIIIIII", b"CCQEXP0\0", 0, slot, layer, expert, 3, EXPERT_FIXED_BYTES)
        output.extend(local_header + bytes(EXPERT_HEADER_BYTES - len(local_header)))
        for role_index, n in enumerate(OUTPUT_CHANNELS):
            row = records[3 * slot + role_index]
            k = VALUES_PER_MATRIX // n
            if row["indices"].shape != (k // VECTOR, n) or row["local_scale"].shape != (k // GROUP, n):
                raise RuntimeError("matrix field geometry changed")
            output.extend(np.ascontiguousarray(row["indices"], dtype=np.uint8).tobytes(order="C"))
            output.extend(pack_local_scale(row["local_scale"]))
            output.extend(np.ascontiguousarray(row["code_scale"], dtype="<f4").tobytes(order="C"))
            output.extend(np.ascontiguousarray(row["code_zp"], dtype="<f4").tobytes(order="C"))
            output.extend(np.ascontiguousarray(row["super_scale"], dtype="<f2").tobytes(order="C"))
    if len(output) != PANEL_FIXED_BYTES:
        raise RuntimeError("serialized prefix length changed")
    return bytes(output)


def parse_prefix(payload: bytes) -> tuple[str, list[dict[str, np.ndarray]]]:
    if len(payload) != PANEL_FIXED_BYTES:
        raise RuntimeError("prefix byte length changed")
    magic, version, experts, matrices, expert_bytes = struct.unpack_from("<8sIIII", payload, 0)
    if (magic, version, experts, matrices, expert_bytes) != (b"CCQRM0\0\0", 0, EXPERTS, MATRICES, EXPERT_FIXED_BYTES):
        raise RuntimeError("global prefix header changed")
    zero = payload.find(b"\0", struct.calcsize("<8sIIII"))
    if zero < 0:
        raise RuntimeError("global metadata terminator missing")
    metadata = json.loads(payload[struct.calcsize("<8sIIII"):zero].decode("utf-8"))
    if metadata.get("format") != "CCQ-RMSE-S0-v0" or metadata.get("experts") != EXPERTS:
        raise RuntimeError("global metadata changed")
    offset = GLOBAL_HEADER_BYTES
    records: list[dict[str, np.ndarray]] = []
    for slot in range(EXPERTS):
        fields = struct.unpack_from("<8sIIIIII", payload, offset)
        layer, expert = EXPECTED_SLOTS[slot]
        if fields != (b"CCQEXP0\0", 0, slot, layer, expert, 3, EXPERT_FIXED_BYTES):
            raise RuntimeError(f"expert header changed at {slot}")
        offset += EXPERT_HEADER_BYTES
        for n in OUTPUT_CHANNELS:
            k = VALUES_PER_MATRIX // n
            index_bytes = k * n // VECTOR
            local_bytes = k * n // 128
            indices = np.frombuffer(payload, dtype=np.uint8, count=index_bytes, offset=offset).copy().reshape(k // VECTOR, n)
            offset += index_bytes
            local = unpack_local_scale(payload[offset:offset + local_bytes], k // GROUP, n)
            offset += local_bytes
            code_scale = np.frombuffer(payload, dtype="<f4", count=n, offset=offset).copy()
            offset += n * 4
            code_zp = np.frombuffer(payload, dtype="<f4", count=n, offset=offset).copy()
            offset += n * 4
            super_scale = np.frombuffer(payload, dtype="<f2", count=n, offset=offset).copy()
            offset += n * 2
            if not np.isfinite(code_scale).all() or not np.isfinite(code_zp).all() or not np.isfinite(super_scale).all():
                raise RuntimeError("non-finite serialized parameter")
            if np.any(code_scale <= 0) or np.any(super_scale <= 0):
                raise RuntimeError("non-positive serialized parameter")
            records.append(
                {"indices": indices, "local_scale": local, "code_scale": code_scale, "code_zp": code_zp, "super_scale": super_scale}
            )
    if offset != len(payload):
        raise RuntimeError("trailing prefix member")
    return str(metadata["label"]), records


def reconstruct(cp: Any, decoder: Any, record: dict[str, np.ndarray], k: int, n: int) -> Any:
    indices = cp.asarray(record["indices"].reshape(k // GROUP, GROUP // VECTOR, n))
    code_scale = cp.asarray(record["code_scale"].astype(np.float32))
    code_zp = cp.asarray(record["code_zp"].astype(np.float32))
    state_matrix = bytes_to_state_matrix(cp, decoder, indices, code_scale, code_zp, k, n)
    local = cp.asarray(record["local_scale"])
    super_scale = cp.asarray(record["super_scale"].astype(np.float32))
    scale = local.astype(cp.float32) * super_scale[None, :]
    return state_matrix.astype(cp.float32) * scale.repeat(GROUP, axis=0)


def score_records(cp: Any, decoder: Any, arrays: list[Any], records: list[dict[str, np.ndarray]], prefix_bpw: float) -> dict[str, Any]:
    matrix_rows: list[dict[str, Any]] = []
    expert_sse = [0.0] * EXPERTS
    expert_energy = [0.0] * EXPERTS
    total_holdout_energy = 0.0
    for slot in HOLDOUT_SLOTS:
        for role_index in range(3):
            ordinal = 3 * slot + role_index
            total_holdout_energy += finite(cp.asnumpy(cp.sum(arrays[ordinal].astype(cp.float64) ** 2)), "heldout total energy")
    accumulated_holdout_sse = 0.0
    early_certificate = None
    for ordinal, natural in enumerate(arrays):
        role_index = ordinal % 3
        role = ROLES[role_index]
        oriented = released_orientation(cp, natural, role)
        k, n = oriented.shape
        decoded = reconstruct(cp, decoder, records[ordinal], k, n)
        sse = finite(cp.asnumpy(cp.sum((oriented.astype(cp.float64) - decoded.astype(cp.float64)) ** 2)), "matrix SSE")
        energy = finite(cp.asnumpy(cp.sum(oriented.astype(cp.float64) ** 2)), "matrix energy")
        slot = ordinal // 3
        expert_sse[slot] += sse
        expert_energy[slot] += energy
        split = "holdout" if slot in HOLDOUT_SLOTS else "fit"
        row = {"matrix_ordinal": ordinal, "slot": slot, "role": role, "split": split, "sse": sse, "source_energy": energy, "relative_residual_energy": sse / energy}
        if split == "holdout":
            accumulated_holdout_sse += sse
            lower_q = accumulated_holdout_sse / total_holdout_energy
            lower_f = lower_q * 2.0 ** (2.0 * prefix_bpw)
            row["accumulated_no_recovery_F_lower_bound"] = lower_f
            if early_certificate is None and lower_f > TARGET_F:
                early_certificate = {"after_matrix_ordinal": ordinal, "F_lower_bound": lower_f, "why": "remaining SSE is nonnegative"}
        matrix_rows.append(row)
        del oriented, decoded
    fit_slots = list(FIT_SLOTS)
    holdout_slots = list(HOLDOUT_SLOTS)
    fit_sse = sum(expert_sse[slot] for slot in fit_slots)
    fit_energy = sum(expert_energy[slot] for slot in fit_slots)
    held_sse = sum(expert_sse[slot] for slot in holdout_slots)
    held_energy = sum(expert_energy[slot] for slot in holdout_slots)
    factor = 2.0 ** (2.0 * prefix_bpw)
    held_q = held_sse / held_energy
    experts = []
    for slot in holdout_slots:
        q = expert_sse[slot] / expert_energy[slot]
        f_value = q * factor
        experts.append(
            {
                "slot": slot,
                "layer": EXPECTED_SLOTS[slot][0],
                "expert": EXPECTED_SLOTS[slot][1],
                "sse": expert_sse[slot],
                "source_energy": expert_energy[slot],
                "relative_residual_energy": q,
                "F_oracle": f_value,
                "s_oracle": -0.5 * math.log2(f_value),
            }
        )
    held_f = held_q * factor
    return {
        "fit_relative_residual_energy": fit_sse / fit_energy,
        "holdout_relative_residual_energy": held_q,
        "holdout_F_oracle": held_f,
        "holdout_s_oracle": -0.5 * math.log2(held_f),
        "heldout_experts": experts,
        "strict_early_kill_certificate": early_certificate,
        "matrices": matrix_rows,
    }


def rate_ledger(expert_count: int) -> tuple[list[dict[str, Any]], float, float]:
    values = expert_count * VALUES_PER_EXPERT
    fixed = GLOBAL_HEADER_BYTES + expert_count * EXPERT_FIXED_BYTES
    prefix_bpw = fixed * 8.0 / values
    required_q = TARGET_F / 2.0 ** (2.0 * prefix_bpw)
    rows = []
    for requested in RATES:
        physical = math.ceil(requested * values / 8.0)
        local_total = physical - GLOBAL_HEADER_BYTES
        local_min, remainder = divmod(local_total, expert_count)
        local_max = local_min + int(remainder != 0)
        residual = physical - fixed
        cold = GLOBAL_HEADER_BYTES + 4096 * math.ceil(local_max / 4096.0)
        rows.append(
            {
                "requested_bpw": requested,
                "actual_bpw": physical * 8.0 / values,
                "physical_bytes": physical,
                "fixed_prefix_bytes": fixed,
                "ideal_residual_bytes": residual,
                "local_frame_min_bytes": local_min,
                "local_frame_max_bytes": local_max,
                "large_frame_count": remainder,
                "cold_expert_bytes_4k": cold,
                "cold_read_amplification": cold / (physical / expert_count),
            }
        )
    return rows, prefix_bpw, required_q


def encode_panel(cp: Any, decoder: Any, arrays: list[Any]) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    fields: list[dict[str, np.ndarray]] = []
    traces: list[dict[str, Any]] = []
    for ordinal, natural in enumerate(arrays):
        role = ROLES[ordinal % 3]
        matrix = released_orientation(cp, natural, role)
        record, trace = encode_matrix(cp, decoder, matrix, ordinal)
        fields.append(record)
        traces.append(trace)
        del matrix
        cp.get_default_memory_pool().free_all_blocks()
    return fields, traces


def decide(source: dict[str, Any], gaussian: dict[str, Any]) -> tuple[str, list[str]]:
    if float(source["holdout_F_oracle"]) > TARGET_F:
        return "KILL", ["pooled held-out canonical CCQ first stage fails the ideal-residual oracle"]
    every_expert = all(float(row["F_oracle"]) <= TARGET_F for row in source["heldout_experts"])
    matched = float(source["holdout_s_oracle"]) - float(gaussian["holdout_s_oracle"])
    if every_expert and float(source["holdout_s_oracle"]) >= PROMOTION_S and matched > 0.0:
        return "PROMOTE_TO_FRESH_AUXILIARY_CONFIRMATION_ONLY", [
            "pooled and both held-out experts clear F<=0.8",
            "pooled source clears the frozen 0.02-bpw margin",
            "matched source-minus-Gaussian s is positive",
        ]
    reasons = []
    if not every_expert:
        reasons.append("at least one held-out expert exceeds F=0.8")
    if float(source["holdout_s_oracle"]) < PROMOTION_S:
        reasons.append("pooled source does not clear the frozen promotion margin")
    if matched <= 0.0:
        reasons.append("matched source-minus-Gaussian s is non-positive")
    return "HOLD_INCONCLUSIVE", reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authorization", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.authorization != AUTHORIZATION:
        raise SystemExit("authorization literal mismatch; no payload bytes opened")
    lock_path, lock = validate_source_lock(args.root)
    required_env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0", "CUPY_ACCELERATORS": ""}
    for key, value in required_env.items():
        if os.environ.get(key) != value:
            raise SystemExit(f"required environment mismatch for {key}; no payload bytes opened")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("output must be absent or an empty directory; no payload bytes opened")
    output.mkdir(parents=True, exist_ok=True)

    import cupy as cp

    cp.cuda.Device(0).use()
    decoder = fmaf_decoder(cp)
    started = time.monotonic()
    source, source_receipts = load_sources(cp, lock_path, lock)
    gaussian, gaussian_match = gaussian_controls(cp, source)
    panel_ledger, prefix_bpw, required_q = rate_ledger(EXPERTS)
    layer_ledger, layer_prefix_bpw, layer_required_q = rate_ledger(128)

    source_fields, source_trace = encode_panel(cp, decoder, source)
    source_payload = serialize_prefix(source_fields, "source")
    source_path = output / "source_prefix.bin"
    source_path.write_bytes(source_payload)
    source_label, source_parsed = parse_prefix(source_path.read_bytes())
    if source_label != "source":
        raise RuntimeError("source packet label changed")
    source_score = score_records(cp, decoder, source, source_parsed, prefix_bpw)

    gaussian_fields, gaussian_trace = encode_panel(cp, decoder, gaussian)
    gaussian_payload = serialize_prefix(gaussian_fields, "gaussian")
    gaussian_path = output / "gaussian_prefix.bin"
    gaussian_path.write_bytes(gaussian_payload)
    gaussian_label, gaussian_parsed = parse_prefix(gaussian_path.read_bytes())
    if gaussian_label != "gaussian":
        raise RuntimeError("Gaussian packet label changed")
    gaussian_score = score_records(cp, decoder, gaussian, gaussian_parsed, prefix_bpw)

    status, reasons = decide(source_score, gaussian_score)
    matched_advantage = float(source_score["holdout_s_oracle"]) - float(gaussian_score["holdout_s_oracle"])
    device = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "schema": "ccq-raw-mse-stage0-result-v0",
        "status": status,
        "decision_reasons": reasons,
        "claim_boundary": (
            "Frozen paper-derived CCQ Code-Cluster cell only. Not an official encoder reproduction, finite residual codec, "
            "fresh-validation result, model-wide result, or target achievement."
        ),
        "source_lock": {
            "path": str(lock_path),
            "bytes": lock_path.stat().st_size,
            "file_sha256": sha256_file(lock_path),
            "internal_sha256": lock["lock_sha256"],
        },
        "split": {"fit_slots": list(FIT_SLOTS), "holdout_slots": list(HOLDOUT_SLOTS)},
        "source_receipts": source_receipts,
        "gaussian_moment_match": gaussian_match,
        "source_prefix": {"bytes": len(source_payload), "sha256": hashlib.sha256(source_payload).hexdigest()},
        "gaussian_prefix": {"bytes": len(gaussian_payload), "sha256": hashlib.sha256(gaussian_payload).hexdigest()},
        "source_trace": source_trace,
        "gaussian_trace": gaussian_trace,
        "source_score": source_score,
        "gaussian_score": gaussian_score,
        "matched_source_minus_gaussian_s": matched_advantage,
        "six_expert_rate_ledger": panel_ledger,
        "six_expert_fixed_prefix_bpw": prefix_bpw,
        "six_expert_required_first_stage_q": required_q,
        "hypothetical_128_expert_ledger": {
            "claim_boundary": "Arithmetic projection only; no 128-expert generalization evidence.",
            "fixed_prefix_bpw": layer_prefix_bpw,
            "required_first_stage_q": layer_required_q,
            "rates": layer_ledger,
        },
        "target": {"F": TARGET_F, "s": TARGET_S, "promotion_s": PROMOTION_S},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "device_name": bytes(device["name"]).decode("ascii", errors="replace") if isinstance(device["name"], bytes) else str(device["name"]),
            "device_total_memory": int(device["totalGlobalMem"]),
            "elapsed_seconds": time.monotonic() - started,
            "environment": {key: os.environ.get(key) for key in required_env},
        },
    }
    result["result_lock_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    write_json(output / "result.json", result)
    print(json.dumps({"status": status, "result": str(output / "result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
