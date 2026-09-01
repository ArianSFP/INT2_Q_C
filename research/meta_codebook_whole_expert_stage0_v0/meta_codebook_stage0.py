#!/usr/bin/env python3
"""Frozen CuPy stage-0 nonlinear meta-codebook experiment.

This program is intentionally inert without its literal authorization flag.
It reads exactly the authenticated 18-matrix source lock and its declared BF16
payloads, trains on four whole experts, and scores two held-out whole experts.
There is no network, subprocess, model loader, or alternate input mode.
"""

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


AUTHORIZATION = "OPEN_AUTHENTICATED_18_MATRIX_PANEL_FOR_META_CODEBOOK_STAGE0_V0"
SOURCE_LOCK_RELPATH = Path("blind_protocol_v2/unblinded/source_hashes.lock.json")
SOURCE_LOCK_SHA256 = "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23"
SOURCE_LOCK_BYTES = 46013
SOURCE_LOCK_INTERNAL = "5a82dac742110d4f48bbd73ae82081e1622b10b660b7850dadfe613ff475cc5b"

ROWS = 768
COLS = 2048
ROLES = ("gate", "up", "down")
MATRICES = 18
EXPERTS = 6
VALUES_PER_MATRIX = ROWS * COLS
VALUES_PER_EXPERT = 3 * VALUES_PER_MATRIX
PANEL_VALUES = EXPERTS * VALUES_PER_EXPERT
VECTOR_DIM = 8
VECTORS_PER_MATRIX = VALUES_PER_MATRIX // VECTOR_DIM
LATENT_DIM = 4
CONDITION_DIM = 4
HIDDEN = 64
CODE_COUNT = 32768
INDEX_BITS = 15
INDEX_BYTES_PER_EXPERT = VALUES_PER_EXPERT // VECTOR_DIM * INDEX_BITS // 8
GLOBAL_SIDE_BYTES = 278528
GLOBAL_HEADER_BYTES = 4096
LOCAL_HEADER_BYTES = 64
DECODER_FP16_BYTES = 10512
ROW_MOMENT_BYTES_PER_MATRIX = ROWS * 2 * 2
ROW_MOMENT_BYTES_PER_EXPERT = 3 * ROW_MOMENT_BYTES_PER_MATRIX
PANEL_MOMENT_BYTES = EXPERTS * ROW_MOMENT_BYTES_PER_EXPERT
GLOBAL_PADDING_BYTES = 1776
RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
PROMOTION_MARGIN_S = 0.02
FIT_SLOTS = (0, 2, 3, 5)
HOLDOUT_SLOTS = (1, 4)
SEEDS = (2026090101, 2026090102)
GAUSSIAN_SEED_BASE = 70707001
STEPS = 512
BATCH = 2048
EVAL_BATCH = 4096
CODE_TILE = 2048
LEARNING_RATE = 1.0e-3
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1.0e-8
COMMITMENT_BETA = 0.25

EXPECTED_SLOTS = (
    (5, 18),
    (12, 7),
    (18, 20),
    (28, 83),
    (36, 76),
    (45, 41),
)

DECODER_SHAPES = (
    ("dec_w1", (LATENT_DIM + CONDITION_DIM, HIDDEN)),
    ("dec_b1", (HIDDEN,)),
    ("dec_w2", (HIDDEN, HIDDEN)),
    ("dec_b2", (HIDDEN,)),
    ("dec_w3", (HIDDEN, VECTOR_DIM)),
    ("dec_b3", (VECTOR_DIM,)),
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


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


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {name}")
    return result


def validate_source_lock(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.resolve()
    lock_path = (root / SOURCE_LOCK_RELPATH).resolve()
    if lock_path.stat().st_size != SOURCE_LOCK_BYTES:
        raise RuntimeError("source lock byte length changed")
    if sha256_file(lock_path) != SOURCE_LOCK_SHA256:
        raise RuntimeError("source lock file hash changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "int2-qwen-blind-source-finalization-v2":
        raise RuntimeError("unexpected source-lock schema")
    if lock.get("lock_sha256") != SOURCE_LOCK_INTERNAL:
        raise RuntimeError("source-lock internal identity changed")
    if lock.get("matrix_count") != MATRICES or lock.get("source_values") != PANEL_VALUES:
        raise RuntimeError("source-lock panel size changed")
    rows = lock.get("matrices")
    if not isinstance(rows, list) or len(rows) != MATRICES:
        raise RuntimeError("source lock does not contain 18 matrices")
    for ordinal, row in enumerate(rows):
        slot = ordinal // 3
        role = ROLES[ordinal % 3]
        layer, expert = EXPECTED_SLOTS[slot]
        expected_shape = [2048, 768] if role == "down" else [768, 2048]
        required = {
            "matrix_ordinal": ordinal,
            "layer": layer,
            "expert": expert,
            "role": role,
            "shape": expected_shape,
            "nvalues": VALUES_PER_MATRIX,
            "nbytes": VALUES_PER_MATRIX * 2,
            "dtype": "BF16",
        }
        for key, expected in required.items():
            if row.get(key) != expected:
                raise RuntimeError(f"source-lock identity mismatch at {ordinal}:{key}")
        digest = row.get("source_bf16_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"bad source hash at ordinal {ordinal}")
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
            raise RuntimeError("source path escaped authenticated root") from exc
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source is missing, non-file, or symlink: {path}")
        observed = sha256_file(path)
        if observed != row["source_bf16_sha256"]:
            raise RuntimeError(f"source payload hash mismatch at ordinal {ordinal}")
        words = np.fromfile(path, dtype="<u2")
        if words.size != VALUES_PER_MATRIX:
            raise RuntimeError(f"source payload length mismatch at ordinal {ordinal}")
        host = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
        host = host.reshape(tuple(int(v) for v in row["shape"]))
        if row["role"] == "down":
            host = host.T
        host = np.ascontiguousarray(host, dtype=np.float32)
        if host.shape != (ROWS, COLS) or not np.isfinite(host).all():
            raise RuntimeError(f"invalid decoded source at ordinal {ordinal}")
        arrays.append(cp.asarray(host))
        receipts.append(
            {
                "matrix_ordinal": ordinal,
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "role": str(row["role"]),
                "payload_bytes": int(path.stat().st_size),
                "declared_sha256": str(row["source_bf16_sha256"]),
                "observed_sha256": observed,
            }
        )
    return arrays, receipts


def matrix_condition(cp: Any, row: dict[str, Any]) -> Any:
    role = ROLES.index(str(row["role"]))
    condition = np.zeros(CONDITION_DIM, dtype=np.float32)
    condition[role] = 1.0
    condition[3] = 2.0 * float(row["layer"]) / 47.0 - 1.0
    return cp.asarray(condition)


def compute_moments(cp: Any, arrays: list[Any]) -> tuple[np.ndarray, list[dict[str, float]]]:
    moments = np.empty((MATRICES, ROWS, 2), dtype="<f2")
    reports: list[dict[str, float]] = []
    for ordinal, values in enumerate(arrays):
        mean = cp.mean(values, axis=1, dtype=cp.float64)
        square = cp.multiply(values, values, dtype=cp.float64)
        rms = cp.sqrt(cp.mean(square, axis=1, dtype=cp.float64))
        mean_host = cp.asnumpy(mean)
        rms_host = cp.asnumpy(rms)
        if not np.isfinite(mean_host).all() or not np.isfinite(rms_host).all() or np.any(rms_host <= 0.0):
            raise RuntimeError(f"invalid row moments at ordinal {ordinal}")
        moments[ordinal, :, 0] = mean_host.astype("<f2")
        moments[ordinal, :, 1] = rms_host.astype("<f2")
        serialized = moments[ordinal].tobytes(order="C")
        reports.append(
            {
                "matrix_ordinal": ordinal,
                "row_count": ROWS,
                "mean_min": float(np.min(mean_host)),
                "mean_max": float(np.max(mean_host)),
                "rms_min": float(np.min(rms_host)),
                "rms_max": float(np.max(rms_host)),
                "stored_fp16_bytes": len(serialized),
                "stored_fp16_sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )
    if moments.nbytes != PANEL_MOMENT_BYTES:
        raise RuntimeError("moment serialization length changed")
    return moments, reports


def decoded_location(cp: Any, moments: np.ndarray, ordinal: int) -> tuple[Any, Any]:
    mean = cp.asarray(moments[ordinal, :, 0].astype(np.float32))[:, None]
    rms = cp.asarray(moments[ordinal, :, 1].astype(np.float32))[:, None]
    variance = cp.maximum(rms * rms - mean * mean, cp.float32(1.0e-20))
    return mean, cp.sqrt(variance)


def normalize_vectors(cp: Any, values: Any, moments: np.ndarray, ordinal: int) -> Any:
    mean, scale = decoded_location(cp, moments, ordinal)
    normalized = (values - mean) / scale
    return cp.ascontiguousarray(normalized.reshape(-1, VECTOR_DIM), dtype=cp.float32)


def gaussian_controls(cp: Any, arrays: list[Any]) -> tuple[list[Any], list[dict[str, float]]]:
    controls: list[Any] = []
    reports: list[dict[str, float]] = []
    for ordinal, source in enumerate(arrays):
        rng = cp.random.RandomState(GAUSSIAN_SEED_BASE + ordinal)
        z = rng.standard_normal(source.shape).astype(cp.float32)
        z_mean = cp.mean(z, axis=1, keepdims=True, dtype=cp.float64)
        z_center = z - z_mean.astype(cp.float32)
        z_std = cp.sqrt(cp.mean(cp.multiply(z_center, z_center, dtype=cp.float64), axis=1, keepdims=True))
        source_mean = cp.mean(source, axis=1, keepdims=True, dtype=cp.float64)
        source_rms = cp.sqrt(cp.mean(cp.multiply(source, source, dtype=cp.float64), axis=1, keepdims=True))
        source_std = cp.sqrt(cp.maximum(source_rms * source_rms - source_mean * source_mean, 1.0e-30))
        control = z_center * (source_std / z_std).astype(cp.float32) + source_mean.astype(cp.float32)
        control = cp.ascontiguousarray(control, dtype=cp.float32)
        observed_mean = cp.mean(control, axis=1, keepdims=True, dtype=cp.float64)
        observed_rms = cp.sqrt(cp.mean(cp.multiply(control, control, dtype=cp.float64), axis=1, keepdims=True))
        mean_error = finite_float(cp.asnumpy(cp.max(cp.abs(observed_mean - source_mean))), "control mean error")
        rms_error = finite_float(cp.asnumpy(cp.max(cp.abs(observed_rms - source_rms))), "control rms error")
        max_source_rms = finite_float(cp.asnumpy(cp.max(source_rms)), "source rms")
        tolerance = 5.0e-6 * max(1.0, max_source_rms)
        if mean_error > tolerance or rms_error > tolerance:
            raise RuntimeError(f"Gaussian moment match failed at ordinal {ordinal}")
        controls.append(control)
        reports.append(
            {
                "matrix_ordinal": ordinal,
                "maximum_row_absolute_mean_error": mean_error,
                "maximum_row_absolute_rms_error": rms_error,
                "tolerance": tolerance,
            }
        )
    return controls, reports


def xavier(cp: Any, rng: Any, rows: int, cols: int) -> Any:
    scale = math.sqrt(2.0 / float(rows + cols))
    return (rng.standard_normal((rows, cols)) * scale).astype(cp.float32)


def initialize_parameters(cp: Any, rng: Any) -> dict[str, Any]:
    params = {
        "enc_w1": xavier(cp, rng, VECTOR_DIM + CONDITION_DIM, HIDDEN),
        "enc_b1": cp.zeros(HIDDEN, dtype=cp.float32),
        "enc_w2": xavier(cp, rng, HIDDEN, HIDDEN),
        "enc_b2": cp.zeros(HIDDEN, dtype=cp.float32),
        "enc_w3": xavier(cp, rng, HIDDEN, LATENT_DIM),
        "enc_b3": cp.zeros(LATENT_DIM, dtype=cp.float32),
        "dec_w1": xavier(cp, rng, LATENT_DIM + CONDITION_DIM, HIDDEN),
        "dec_b1": cp.zeros(HIDDEN, dtype=cp.float32),
        "dec_w2": xavier(cp, rng, HIDDEN, HIDDEN),
        "dec_b2": cp.zeros(HIDDEN, dtype=cp.float32),
        "dec_w3": xavier(cp, rng, HIDDEN, VECTOR_DIM),
        "dec_b3": cp.zeros(VECTOR_DIM, dtype=cp.float32),
    }
    return params


def encoder_forward(cp: Any, x: Any, condition: Any, params: dict[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    cond = cp.broadcast_to(condition, (x.shape[0], CONDITION_DIM))
    a0 = cp.concatenate((x, cond), axis=1)
    h1 = cp.tanh(a0 @ params["enc_w1"] + params["enc_b1"])
    h2 = cp.tanh(h1 @ params["enc_w2"] + params["enc_b2"])
    latent = h2 @ params["enc_w3"] + params["enc_b3"]
    return latent, (a0, h1, h2)


def decoder_forward(cp: Any, latent: Any, condition: Any, params: dict[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    cond = cp.broadcast_to(condition, (latent.shape[0], CONDITION_DIM))
    a0 = cp.concatenate((latent, cond), axis=1)
    h1 = cp.tanh(a0 @ params["dec_w1"] + params["dec_b1"])
    h2 = cp.tanh(h1 @ params["dec_w2"] + params["dec_b2"])
    output = h2 @ params["dec_w3"] + params["dec_b3"]
    return output, (a0, h1, h2)


def nearest_latent(cp: Any, latent: Any, codebook: Any) -> Any:
    x2 = cp.sum(latent * latent, axis=1, keepdims=True)
    c2 = cp.sum(codebook * codebook, axis=1)[None, :]
    distance = x2 + c2 - cp.float32(2.0) * (latent @ codebook.T)
    return cp.argmin(distance, axis=1).astype(cp.int32)


def backprop_mlp(cp: Any, grad_output: Any, cache: tuple[Any, ...], params: dict[str, Any], prefix: str) -> tuple[Any, dict[str, Any]]:
    a0, h1, h2 = cache
    w1 = params[f"{prefix}_w1"]
    w2 = params[f"{prefix}_w2"]
    w3 = params[f"{prefix}_w3"]
    grads: dict[str, Any] = {}
    grads[f"{prefix}_w3"] = h2.T @ grad_output
    grads[f"{prefix}_b3"] = cp.sum(grad_output, axis=0)
    dh2 = grad_output @ w3.T
    da2 = dh2 * (cp.float32(1.0) - h2 * h2)
    grads[f"{prefix}_w2"] = h1.T @ da2
    grads[f"{prefix}_b2"] = cp.sum(da2, axis=0)
    dh1 = da2 @ w2.T
    da1 = dh1 * (cp.float32(1.0) - h1 * h1)
    grads[f"{prefix}_w1"] = a0.T @ da1
    grads[f"{prefix}_b1"] = cp.sum(da1, axis=0)
    return da1 @ w1.T, grads


def selected_codebook_gradient(cp: Any, indices: Any, selected_gradient: Any) -> Any:
    order = cp.argsort(indices)
    sorted_indices = indices[order]
    sorted_gradient = selected_gradient[order]
    unique, starts = cp.unique(sorted_indices, return_index=True)
    sums = cp.add.reduceat(sorted_gradient, starts, axis=0)
    gradient = cp.zeros((CODE_COUNT, LATENT_DIM), dtype=cp.float32)
    gradient[unique] = sums
    return gradient


def adam_step(cp: Any, params: dict[str, Any], grads: dict[str, Any], state: dict[str, tuple[Any, Any]], step: int) -> None:
    correction1 = 1.0 - BETA1**step
    correction2 = 1.0 - BETA2**step
    for name in sorted(grads):
        grad = grads[name]
        m, v = state[name]
        m *= cp.float32(BETA1)
        m += cp.float32(1.0 - BETA1) * grad
        v *= cp.float32(BETA2)
        v += cp.float32(1.0 - BETA2) * grad * grad
        params[name] -= cp.float32(LEARNING_RATE) * (m / cp.float32(correction1)) / (
            cp.sqrt(v / cp.float32(correction2)) + cp.float32(ADAM_EPS)
        )


def initialize_codebook(cp: Any, rng: Any, arrays: list[Any], rows: list[dict[str, Any]], moments: np.ndarray, params: dict[str, Any]) -> Any:
    chunks: list[Any] = []
    per_matrix = math.ceil(CODE_COUNT / (len(FIT_SLOTS) * 3))
    for slot in FIT_SLOTS:
        for role_index in range(3):
            ordinal = 3 * slot + role_index
            vectors = normalize_vectors(cp, arrays[ordinal], moments, ordinal)
            indices = rng.randint(0, VECTORS_PER_MATRIX, size=per_matrix, dtype=cp.int32)
            latent, _ = encoder_forward(cp, vectors[indices], matrix_condition(cp, rows[ordinal]), params)
            chunks.append(latent)
    return cp.ascontiguousarray(cp.concatenate(chunks, axis=0)[:CODE_COUNT], dtype=cp.float32)


def train_codec(cp: Any, arrays: list[Any], rows: list[dict[str, Any]], moments: np.ndarray, seed: int) -> tuple[dict[str, Any], Any, list[dict[str, float]]]:
    rng = cp.random.RandomState(seed)
    params = initialize_parameters(cp, rng)
    params["codebook"] = initialize_codebook(cp, rng, arrays, rows, moments, params)
    state = {name: (cp.zeros_like(value), cp.zeros_like(value)) for name, value in params.items()}
    trace: list[dict[str, float]] = []
    fit_ordinals = tuple(3 * slot + role for slot in FIT_SLOTS for role in range(3))
    for step0 in range(STEPS):
        ordinal = fit_ordinals[step0 % len(fit_ordinals)]
        vectors = normalize_vectors(cp, arrays[ordinal], moments, ordinal)
        sample = rng.randint(0, VECTORS_PER_MATRIX, size=BATCH, dtype=cp.int32)
        x = vectors[sample]
        condition = matrix_condition(cp, rows[ordinal])
        latent, enc_cache = encoder_forward(cp, x, condition, params)
        indices = nearest_latent(cp, latent, params["codebook"])
        quantized = params["codebook"][indices]
        reconstructed, dec_cache = decoder_forward(cp, quantized, condition, params)
        difference = reconstructed - x
        grad_reconstruction = cp.float32(2.0 / float(BATCH * VECTOR_DIM)) * difference
        grad_decoder_input, dec_grads = backprop_mlp(cp, grad_reconstruction, dec_cache, params, "dec")
        grad_latent = grad_decoder_input[:, :LATENT_DIM]
        grad_latent += cp.float32(2.0 * COMMITMENT_BETA / float(BATCH * LATENT_DIM)) * (latent - quantized)
        grad_encoder_input, enc_grads = backprop_mlp(cp, grad_latent, enc_cache, params, "enc")
        del grad_encoder_input
        selected = cp.float32(2.0 / float(BATCH * LATENT_DIM)) * (quantized - latent)
        codebook_grad = selected_codebook_gradient(cp, indices, selected)
        grads = dict(dec_grads)
        grads.update(enc_grads)
        grads["codebook"] = codebook_grad
        adam_step(cp, params, grads, state, step0 + 1)
        if step0 == 0 or (step0 + 1) % 64 == 0:
            mse = finite_float(cp.asnumpy(cp.mean(difference * difference, dtype=cp.float64)), "training MSE")
            used = int(cp.unique(indices).size)
            trace.append({"step": step0 + 1, "normalized_mse": mse, "codes_used_in_batch": used})
    return params, params["codebook"], trace


def serialize_global_side(params: dict[str, Any], codebook: Any, seed: int, control: bool) -> bytes:
    codebook_bytes = np.asarray(codebook, dtype="<f2").tobytes(order="C")
    if len(codebook_bytes) != CODE_COUNT * LATENT_DIM * 2:
        raise RuntimeError("codebook serialization length changed")
    decoder_parts: list[bytes] = []
    for name, shape in DECODER_SHAPES:
        value = np.asarray(params[name], dtype="<f2")
        if value.shape != shape:
            raise RuntimeError(f"decoder shape changed: {name}")
        decoder_parts.append(value.tobytes(order="C"))
    decoder_bytes = b"".join(decoder_parts)
    if len(decoder_bytes) != DECODER_FP16_BYTES:
        raise RuntimeError("decoder serialization length changed")
    metadata = canonical_json_bytes(
        {
            "code_count": CODE_COUNT,
            "control": bool(control),
            "format": "MCB-WE-S0-v0",
            "index_bits": INDEX_BITS,
            "latent_dimension": LATENT_DIM,
            "seed": seed,
            "vector_dimension": VECTOR_DIM,
        }
    )
    header_prefix = struct.pack("<8sIIII", b"MCBWES0\0", 0, CODE_COUNT, LATENT_DIM, INDEX_BITS)
    header = header_prefix + metadata
    if len(header) > GLOBAL_HEADER_BYTES:
        raise RuntimeError("global header overflow")
    header += bytes(GLOBAL_HEADER_BYTES - len(header))
    payload = header + codebook_bytes + decoder_bytes
    if len(payload) > GLOBAL_SIDE_BYTES:
        raise RuntimeError("global side overflow")
    payload += bytes(GLOBAL_SIDE_BYTES - len(payload))
    if len(payload) != GLOBAL_SIDE_BYTES or payload[-GLOBAL_PADDING_BYTES:] != bytes(GLOBAL_PADDING_BYTES):
        raise RuntimeError("non-canonical global-side padding")
    return payload


def serialize_row_moments(moments: np.ndarray) -> bytes:
    payload = np.asarray(moments, dtype="<f2").tobytes(order="C")
    if len(payload) != PANEL_MOMENT_BYTES:
        raise RuntimeError("row-moment serialization length changed")
    return payload


def parse_side(cp: Any, global_payload: bytes, moment_payload: bytes) -> tuple[dict[str, Any], Any, np.ndarray]:
    if len(global_payload) != GLOBAL_SIDE_BYTES:
        raise RuntimeError("global side length mismatch")
    if len(moment_payload) != PANEL_MOMENT_BYTES:
        raise RuntimeError("row-moment side length mismatch")
    magic, version, count, latent, bits = struct.unpack_from("<8sIIII", global_payload, 0)
    if (magic, version, count, latent, bits) != (b"MCBWES0\0", 0, CODE_COUNT, LATENT_DIM, INDEX_BITS):
        raise RuntimeError("side header mismatch")
    offset = GLOBAL_HEADER_BYTES
    count_values = CODE_COUNT * LATENT_DIM
    codebook_host = np.frombuffer(global_payload, dtype="<f2", count=count_values, offset=offset).astype(np.float32).reshape(CODE_COUNT, LATENT_DIM)
    offset += count_values * 2
    params: dict[str, Any] = {}
    for name, shape in DECODER_SHAPES:
        size = math.prod(shape)
        host = np.frombuffer(global_payload, dtype="<f2", count=size, offset=offset).astype(np.float32).reshape(shape)
        params[name] = cp.asarray(host)
        offset += size * 2
    moments = np.frombuffer(
        moment_payload,
        dtype="<f2",
        count=MATRICES * ROWS * 2,
        offset=0,
    ).copy().reshape(MATRICES, ROWS, 2)
    if offset + GLOBAL_PADDING_BYTES != GLOBAL_SIDE_BYTES or any(global_payload[offset:]):
        raise RuntimeError("global side trailing padding mismatch")
    return params, cp.asarray(codebook_host), moments


def decoded_centers(cp: Any, params: dict[str, Any], codebook: Any, moments: np.ndarray, ordinal: int, row: dict[str, Any]) -> Any:
    normalized, _ = decoder_forward(cp, codebook, matrix_condition(cp, row), params)
    return cp.ascontiguousarray(normalized, dtype=cp.float32)


def exact_nearest_sse(cp: Any, values: Any, centers: Any, moments: np.ndarray, ordinal: int) -> tuple[float, float]:
    vectors = normalize_vectors(cp, values, moments, ordinal)
    _, row_scale = decoded_location(cp, moments, ordinal)
    vector_scale2 = cp.repeat(row_scale[:, 0] * row_scale[:, 0], COLS // VECTOR_DIM)
    center_norm = cp.sum(centers * centers, axis=1)
    sse = 0.0
    energy = 0.0
    for start in range(0, vectors.shape[0], EVAL_BATCH):
        x = vectors[start : start + EVAL_BATCH]
        x_norm = cp.sum(x * x, axis=1)
        best = cp.full(x.shape[0], cp.inf, dtype=cp.float32)
        for code_start in range(0, CODE_COUNT, CODE_TILE):
            c = centers[code_start : code_start + CODE_TILE]
            distance = x_norm[:, None] + center_norm[None, code_start : code_start + c.shape[0]]
            distance -= cp.float32(2.0) * (x @ c.T)
            best = cp.minimum(best, cp.min(distance, axis=1))
        best = cp.maximum(best, cp.float32(0.0))
        weighted = best * vector_scale2[start : start + x.shape[0]]
        sse += finite_float(cp.asnumpy(cp.sum(weighted, dtype=cp.float64)), "nearest SSE")
        raw = values.reshape(-1, VECTOR_DIM)[start : start + x.shape[0]]
        energy += finite_float(cp.asnumpy(cp.sum(cp.multiply(raw, raw, dtype=cp.float64))), "source energy")
    return sse, energy


def evaluate_codec(cp: Any, arrays: list[Any], rows: list[dict[str, Any]], global_payload: bytes, moment_payload: bytes) -> dict[str, Any]:
    decoder, codebook, moments = parse_side(cp, global_payload, moment_payload)
    by_expert: list[dict[str, Any]] = []
    total_sse = 0.0
    total_energy = 0.0
    for slot in HOLDOUT_SLOTS:
        expert_sse = 0.0
        expert_energy = 0.0
        matrix_rows: list[dict[str, Any]] = []
        for role_index in range(3):
            ordinal = 3 * slot + role_index
            centers = decoded_centers(cp, decoder, codebook, moments, ordinal, rows[ordinal])
            sse, energy = exact_nearest_sse(cp, arrays[ordinal], centers, moments, ordinal)
            expert_sse += sse
            expert_energy += energy
            matrix_rows.append(
                {
                    "matrix_ordinal": ordinal,
                    "relative_residual_energy": sse / energy,
                    "sse": sse,
                    "source_energy": energy,
                }
            )
        total_sse += expert_sse
        total_energy += expert_energy
        by_expert.append(
            {
                "slot": slot,
                "layer": int(rows[3 * slot]["layer"]),
                "expert": int(rows[3 * slot]["expert"]),
                "relative_residual_energy": expert_sse / expert_energy,
                "sse": expert_sse,
                "source_energy": expert_energy,
                "matrices": matrix_rows,
            }
        )
    return {
        "relative_residual_energy": total_sse / total_energy,
        "sse": total_sse,
        "source_energy": total_energy,
        "heldout_experts": by_expert,
    }


def rate_ledger(expert_count: int) -> tuple[list[dict[str, Any]], float, float]:
    values = expert_count * VALUES_PER_EXPERT
    fixed_bits = (
        GLOBAL_SIDE_BYTES * 8
        + expert_count * (LOCAL_HEADER_BYTES + ROW_MOMENT_BYTES_PER_EXPERT + INDEX_BYTES_PER_EXPERT) * 8
    )
    prefix_bpw = fixed_bits / values
    threshold_q = TARGET_F / (2.0 ** (2.0 * prefix_bpw))
    rows: list[dict[str, Any]] = []
    for requested in RATES:
        total_bytes = math.ceil(requested * values / 8.0)
        local_total = total_bytes - GLOBAL_SIDE_BYTES
        local_min, remainder = divmod(local_total, expert_count)
        local_max = local_min + int(remainder != 0)
        residual_bytes = total_bytes - GLOBAL_SIDE_BYTES - expert_count * (
            INDEX_BYTES_PER_EXPERT + ROW_MOMENT_BYTES_PER_EXPERT + LOCAL_HEADER_BYTES
        )
        if residual_bytes < 0:
            raise RuntimeError("negative residual budget")
        cold = GLOBAL_SIDE_BYTES + 4096 * math.ceil(local_max / 4096.0)
        actual_rate = total_bytes * 8.0 / values
        residual_bpw = residual_bytes * 8.0 / values
        rows.append(
            {
                "requested_bpw": requested,
                "actual_bpw": actual_rate,
                "physical_bytes": total_bytes,
                "expert_count": expert_count,
                "global_side_bytes": GLOBAL_SIDE_BYTES,
                "local_total_bytes": local_total,
                "local_frame_min_bytes": local_min,
                "local_frame_max_bytes": local_max,
                "large_frame_count": remainder,
                "index_bytes_per_expert": INDEX_BYTES_PER_EXPERT,
                "row_moment_bytes_per_expert": ROW_MOMENT_BYTES_PER_EXPERT,
                "local_header_bytes_per_expert": LOCAL_HEADER_BYTES,
                "total_residual_bytes": residual_bytes,
                "residual_bpw": residual_bpw,
                "cold_expert_bytes_4k": cold,
                "cold_read_amplification": cold / (total_bytes / expert_count),
            }
        )
    return rows, prefix_bpw, threshold_q


def score_oracle(evaluation: dict[str, Any], prefix_bpw: float) -> dict[str, Any]:
    q = float(evaluation["relative_residual_energy"])
    factor = 2.0 ** (2.0 * prefix_bpw)
    f_value = q * factor
    s_value = -0.5 * math.log2(f_value)
    experts = []
    for row in evaluation["heldout_experts"]:
        expert = dict(row)
        expert_f = float(row["relative_residual_energy"]) * factor
        expert["F_oracle"] = expert_f
        expert["s_oracle"] = -0.5 * math.log2(expert_f)
        experts.append(expert)
    return {
        "relative_residual_energy": q,
        "F_oracle": f_value,
        "s_oracle": s_value,
        "heldout_experts": experts,
    }


def run_one(cp: Any, output: Path, seed: int, source: list[Any], control: list[Any], rows: list[dict[str, Any]], moments: np.ndarray, prefix_bpw: float) -> dict[str, Any]:
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir()
    reports: dict[str, Any] = {"seed": seed}
    for label, arrays, is_control in (("source", source, False), ("gaussian", control, True)):
        started = time.monotonic()
        params, codebook, trace = train_codec(cp, arrays, rows, moments, seed)
        host_params = {name: cp.asnumpy(params[name]) for name, _ in DECODER_SHAPES}
        host_codebook = cp.asnumpy(codebook)
        global_payload = serialize_global_side(host_params, host_codebook, seed, is_control)
        moment_payload = serialize_row_moments(moments)
        global_path = seed_dir / f"{label}_global_side.bin"
        moment_path = seed_dir / f"{label}_row_moments.bin"
        global_path.write_bytes(global_payload)
        moment_path.write_bytes(moment_payload)
        evaluation = evaluate_codec(cp, arrays, rows, global_payload, moment_payload)
        oracle = score_oracle(evaluation, prefix_bpw)
        reports[label] = {
            "training_trace": trace,
            "global_side_relpath": str(global_path.relative_to(output)).replace("\\", "/"),
            "global_side_bytes": len(global_payload),
            "global_side_sha256": hashlib.sha256(global_payload).hexdigest(),
            "row_moments_relpath": str(moment_path.relative_to(output)).replace("\\", "/"),
            "row_moments_bytes": len(moment_payload),
            "row_moments_sha256": hashlib.sha256(moment_payload).hexdigest(),
            "evaluation": evaluation,
            "oracle": oracle,
            "elapsed_seconds": time.monotonic() - started,
        }
        del params, codebook, host_params, host_codebook, global_payload, moment_payload, evaluation
        cp.get_default_memory_pool().free_all_blocks()
    reports["matched_advantage_s"] = (
        reports["source"]["oracle"]["s_oracle"] - reports["gaussian"]["oracle"]["s_oracle"]
    )
    return reports


def decide(seed_reports: list[dict[str, Any]]) -> tuple[str, list[str]]:
    source_s = [float(row["source"]["oracle"]["s_oracle"]) for row in seed_reports]
    if max(source_s) < TARGET_S:
        return "KILL", ["even the better predeclared seed fails the favorable source oracle"]
    promoted_s = TARGET_S + PROMOTION_MARGIN_S
    both_margin = min(source_s) >= promoted_s
    every_expert = all(
        float(expert["F_oracle"]) <= TARGET_F
        for row in seed_reports
        for expert in row["source"]["oracle"]["heldout_experts"]
    )
    positive_matched = all(float(row["matched_advantage_s"]) > 0.0 for row in seed_reports)
    if both_margin and every_expert and positive_matched:
        return "PROMOTE_TO_FRESH_AUXILIARY_CONFIRMATION_ONLY", [
            "both seeds clear the exact 0.02-s margin",
            "every held-out expert clears F<=0.8",
            "both matched Gaussian advantages are positive",
        ]
    reasons = []
    if not both_margin:
        reasons.append("both seeds do not clear the promotion margin")
    if not every_expert:
        reasons.append("at least one held-out expert exceeds F=0.8")
    if not positive_matched:
        reasons.append("at least one matched Gaussian advantage is non-positive")
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
        raise SystemExit("authorization literal mismatch; no panel bytes opened")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    lock_path, lock = validate_source_lock(args.root)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    os.environ.setdefault("CUPY_ACCELERATORS", "")
    import cupy as cp

    cp.cuda.Device(0).use()
    started = time.monotonic()
    source, source_receipts = load_sources(cp, lock_path, lock)
    moments, moment_report = compute_moments(cp, source)
    control, gaussian_report = gaussian_controls(cp, source)
    rows = lock["matrices"]
    panel_ledger, prefix_bpw, threshold_q = rate_ledger(EXPERTS)
    layer_ledger, layer_prefix_bpw, layer_threshold_q = rate_ledger(128)
    reports = [run_one(cp, output, seed, source, control, rows, moments, prefix_bpw) for seed in SEEDS]
    decision, reasons = decide(reports)
    device = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "schema": "meta-codebook-whole-expert-stage0-result-v0",
        "status": decision,
        "decision_reasons": reasons,
        "claim_boundary": (
            "Six-expert frozen stage-0 only. A kill rejects this cell. A promotion is an ideal-Gaussian-residual "
            "feasibility result on two held-out experts, not a finite residual codec, model-wide generalization, "
            "fresh-validation result, or target achievement."
        ),
        "source_lock": {
            "path": str(lock_path),
            "bytes": lock_path.stat().st_size,
            "file_sha256": sha256_file(lock_path),
            "internal_sha256": lock["lock_sha256"],
        },
        "split": {"fit_slots": list(FIT_SLOTS), "holdout_slots": list(HOLDOUT_SLOTS)},
        "source_receipts": source_receipts,
        "source_moments": moment_report,
        "gaussian_moment_match": gaussian_report,
        "six_expert_rate_ledger": panel_ledger,
        "six_expert_fixed_prefix_bpw": prefix_bpw,
        "six_expert_required_first_stage_relative_residual_energy": threshold_q,
        "hypothetical_128_expert_layer_ledger": {
            "claim_boundary": "Arithmetic projection only; the six-expert experiment does not establish this layer-wide distribution or generalization.",
            "fixed_prefix_bpw": layer_prefix_bpw,
            "required_first_stage_relative_residual_energy": layer_threshold_q,
            "rates": layer_ledger,
        },
        "target": {
            "F": TARGET_F,
            "s": TARGET_S,
            "promotion_s": TARGET_S + PROMOTION_MARGIN_S,
        },
        "seed_reports": reports,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "device_name": bytes(device["name"]).decode("ascii", errors="replace") if isinstance(device["name"], bytes) else str(device["name"]),
            "device_total_memory": int(device["totalGlobalMem"]),
            "elapsed_seconds": time.monotonic() - started,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        },
    }
    result["result_lock_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    write_json(output / "result.json", result)
    print(json.dumps({"status": decision, "result": str(output / "result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
