#!/usr/bin/env python3
"""Frozen CuPy direct-output K=32768,d=8 whole-expert stage-0 gate."""

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


AUTHORIZATION = "OPEN_AUTHENTICATED_18_MATRIX_PANEL_FOR_DIRECT_OUTPUT_CODEBOOK_STAGE0_V0"
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
VECTORS_PER_ROW = COLS // VECTOR_DIM
VECTORS_PER_MATRIX = VALUES_PER_MATRIX // VECTOR_DIM
CODE_COUNT = 32768
INDEX_BITS = 15
INDEX_BYTES_PER_EXPERT = VALUES_PER_EXPERT // VECTOR_DIM * INDEX_BITS // 8
GLOBAL_HEADER_BYTES = 4096
CODEBOOK_BYTES = CODE_COUNT * VECTOR_DIM * 2
GLOBAL_SIDE_BYTES = GLOBAL_HEADER_BYTES + CODEBOOK_BYTES
LOCAL_HEADER_BYTES = 64
ROW_MOMENT_BYTES_PER_MATRIX = ROWS * 2 * 2
ROW_MOMENT_BYTES_PER_EXPERT = 3 * ROW_MOMENT_BYTES_PER_MATRIX
PANEL_MOMENT_BYTES = EXPERTS * ROW_MOMENT_BYTES_PER_EXPERT
FIT_SLOTS = (0, 2, 3, 5)
HOLDOUT_SLOTS = (1, 4)
FIT_ORDINALS = tuple(3 * slot + role for slot in FIT_SLOTS for role in range(3))
SEEDS = (2026090111, 2026090112)
GAUSSIAN_SEED_BASE = 70707101
PROBE_SEED = 90909101
STEPS = 1024
MINIBATCH = 4096
PROBE_PER_MATRIX = 4096
EVAL_BATCH = 4096
CODE_TILE = 2048
RESEED_PER_STEP = 128
CHECKPOINTS = {
    128: (0.30, 2048),
    256: (0.25, 4096),
    512: (0.20, 8192),
    1024: (0.16, 12288),
}
LLOYD_CHECK = (0.14, 16000)
RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
PROMOTION_MARGIN_S = 0.02

EXPECTED_SLOTS = (
    (5, 18),
    (12, 7),
    (18, 20),
    (28, 83),
    (36, 76),
    (45, 41),
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
    matrix_rows = lock.get("matrices")
    if not isinstance(matrix_rows, list) or len(matrix_rows) != MATRICES:
        raise RuntimeError("source lock does not contain 18 matrices")
    for ordinal, row in enumerate(matrix_rows):
        slot = ordinal // 3
        role = ROLES[ordinal % 3]
        layer, expert = EXPECTED_SLOTS[slot]
        shape = [2048, 768] if role == "down" else [768, 2048]
        expected = {
            "matrix_ordinal": ordinal,
            "layer": layer,
            "expert": expert,
            "role": role,
            "shape": shape,
            "nvalues": VALUES_PER_MATRIX,
            "nbytes": VALUES_PER_MATRIX * 2,
            "dtype": "BF16",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise RuntimeError(f"source-lock identity mismatch at {ordinal}:{key}")
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
            raise RuntimeError(f"source missing, non-file, or symlink: {path}")
        observed = sha256_file(path)
        if observed != row["source_bf16_sha256"]:
            raise RuntimeError(f"source payload hash mismatch at ordinal {ordinal}")
        words = np.fromfile(path, dtype="<u2")
        if words.size != VALUES_PER_MATRIX:
            raise RuntimeError(f"source payload length mismatch at ordinal {ordinal}")
        host = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
        host = host.reshape(tuple(int(value) for value in row["shape"]))
        if row["role"] == "down":
            host = host.T
        host = np.ascontiguousarray(host, dtype=np.float32)
        if host.shape != (ROWS, COLS) or not np.isfinite(host).all():
            raise RuntimeError(f"invalid source at ordinal {ordinal}")
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


def compute_moments(cp: Any, arrays: list[Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    moments = np.empty((MATRICES, ROWS, 2), dtype="<f2")
    report: list[dict[str, Any]] = []
    for ordinal, values in enumerate(arrays):
        mean = cp.mean(values, axis=1, dtype=cp.float64)
        rms = cp.sqrt(cp.mean(cp.multiply(values, values, dtype=cp.float64), axis=1, dtype=cp.float64))
        mean_host = cp.asnumpy(mean)
        rms_host = cp.asnumpy(rms)
        if not np.isfinite(mean_host).all() or not np.isfinite(rms_host).all() or np.any(rms_host <= 0.0):
            raise RuntimeError(f"invalid row moments at ordinal {ordinal}")
        moments[ordinal, :, 0] = mean_host.astype("<f2")
        moments[ordinal, :, 1] = rms_host.astype("<f2")
        body = moments[ordinal].tobytes(order="C")
        report.append(
            {
                "matrix_ordinal": ordinal,
                "row_count": ROWS,
                "stored_fp16_bytes": len(body),
                "stored_fp16_sha256": hashlib.sha256(body).hexdigest(),
                "mean_min": float(np.min(mean_host)),
                "mean_max": float(np.max(mean_host)),
                "rms_min": float(np.min(rms_host)),
                "rms_max": float(np.max(rms_host)),
            }
        )
    if moments.nbytes != PANEL_MOMENT_BYTES:
        raise RuntimeError("row-moment serialization length changed")
    return moments, report


def decoded_location(cp: Any, moments: np.ndarray, ordinal: int) -> tuple[Any, Any]:
    mean = cp.asarray(moments[ordinal, :, 0].astype(np.float32))[:, None]
    rms = cp.asarray(moments[ordinal, :, 1].astype(np.float32))[:, None]
    variance = cp.maximum(rms * rms - mean * mean, cp.float32(1.0e-20))
    return mean, cp.sqrt(variance)


def normalized_vectors(cp: Any, values: Any, moments: np.ndarray, ordinal: int) -> Any:
    mean, scale = decoded_location(cp, moments, ordinal)
    return cp.ascontiguousarray(((values - mean) / scale).reshape(-1, VECTOR_DIM), dtype=cp.float32)


def gaussian_controls(cp: Any, arrays: list[Any]) -> tuple[list[Any], list[dict[str, float]]]:
    controls: list[Any] = []
    reports: list[dict[str, float]] = []
    for ordinal, source in enumerate(arrays):
        rng = cp.random.RandomState(GAUSSIAN_SEED_BASE + ordinal)
        z = rng.standard_normal(source.shape).astype(cp.float32)
        z_mean = cp.mean(z, axis=1, keepdims=True, dtype=cp.float64)
        centered = z - z_mean.astype(cp.float32)
        z_std = cp.sqrt(cp.mean(cp.multiply(centered, centered, dtype=cp.float64), axis=1, keepdims=True))
        source_mean = cp.mean(source, axis=1, keepdims=True, dtype=cp.float64)
        source_rms = cp.sqrt(cp.mean(cp.multiply(source, source, dtype=cp.float64), axis=1, keepdims=True))
        source_std = cp.sqrt(cp.maximum(source_rms * source_rms - source_mean * source_mean, 1.0e-30))
        control = centered * (source_std / z_std).astype(cp.float32) + source_mean.astype(cp.float32)
        control = cp.ascontiguousarray(control, dtype=cp.float32)
        observed_mean = cp.mean(control, axis=1, keepdims=True, dtype=cp.float64)
        observed_rms = cp.sqrt(cp.mean(cp.multiply(control, control, dtype=cp.float64), axis=1, keepdims=True))
        mean_error = finite_float(cp.asnumpy(cp.max(cp.abs(observed_mean - source_mean))), "control mean error")
        rms_error = finite_float(cp.asnumpy(cp.max(cp.abs(observed_rms - source_rms))), "control rms error")
        tolerance = 5.0e-6 * max(1.0, finite_float(cp.asnumpy(cp.max(source_rms)), "source rms"))
        if mean_error > tolerance or rms_error > tolerance:
            raise RuntimeError(f"Gaussian row-moment match failed at ordinal {ordinal}")
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


def modular_indices(count: int, modulo: int, seed: int, stride: int) -> np.ndarray:
    if math.gcd(stride, modulo) != 1:
        raise RuntimeError("index stride is not coprime")
    start = seed % modulo
    values = (start + np.arange(count, dtype=np.int64) * stride) % modulo
    if np.unique(values).size != count:
        raise RuntimeError("deterministic index schedule repeated")
    return values.astype(np.int32)


def frozen_probe_indices() -> tuple[dict[int, np.ndarray], str]:
    probe: dict[int, np.ndarray] = {}
    digest = hashlib.sha256()
    for position, ordinal in enumerate(FIT_ORDINALS):
        indices = modular_indices(PROBE_PER_MATRIX, VECTORS_PER_MATRIX, PROBE_SEED + 7919 * position, 65537)
        probe[ordinal] = indices
        digest.update(np.asarray(indices, dtype="<u4").tobytes())
    return probe, digest.hexdigest()


def initialize_centroids(cp: Any, arrays: list[Any], moments: np.ndarray, seed: int) -> Any:
    per_matrix = math.ceil(CODE_COUNT / len(FIT_ORDINALS))
    chunks: list[Any] = []
    for position, ordinal in enumerate(FIT_ORDINALS):
        indices = modular_indices(per_matrix, VECTORS_PER_MATRIX, seed + 104729 * position, 65537)
        vectors = normalized_vectors(cp, arrays[ordinal], moments, ordinal)
        chunks.append(vectors[cp.asarray(indices)])
    centroids = cp.concatenate(chunks, axis=0)[:CODE_COUNT]
    if centroids.shape != (CODE_COUNT, VECTOR_DIM):
        raise RuntimeError("initial codebook shape changed")
    return cp.ascontiguousarray(centroids, dtype=cp.float32)


def exact_assign(cp: Any, vectors: Any, centroids: Any) -> tuple[Any, Any]:
    vector_norm = cp.sum(vectors * vectors, axis=1)
    centroid_norm = cp.sum(centroids * centroids, axis=1)
    best = cp.full(vectors.shape[0], cp.inf, dtype=cp.float32)
    best_index = cp.zeros(vectors.shape[0], dtype=cp.int32)
    for start in range(0, CODE_COUNT, CODE_TILE):
        block = centroids[start : start + CODE_TILE]
        distance = vector_norm[:, None] + centroid_norm[None, start : start + block.shape[0]]
        distance -= cp.float32(2.0) * (vectors @ block.T)
        local_index = cp.argmin(distance, axis=1).astype(cp.int32)
        local_best = distance[cp.arange(vectors.shape[0]), local_index]
        better = local_best < best
        best = cp.where(better, local_best, best)
        best_index = cp.where(better, local_index + start, best_index)
    return best_index, cp.maximum(best, cp.float32(0.0))


def segmented_sums(cp: Any, indices: Any, vectors: Any) -> tuple[Any, Any, Any]:
    order = cp.argsort(indices)
    sorted_index = indices[order]
    sorted_vectors = vectors[order]
    unique, starts, counts = cp.unique(sorted_index, return_index=True, return_counts=True)
    sums = cp.add.reduceat(sorted_vectors, starts, axis=0)
    return unique, counts.astype(cp.float64), sums.astype(cp.float64)


def reservoir_chunk(cp: Any, vectors: Any, errors: Any) -> tuple[Any, Any]:
    take = min(RESEED_PER_STEP, int(errors.size))
    selected = cp.argpartition(errors, errors.size - take)[-take:]
    order = cp.argsort(errors[selected])[::-1]
    selected = selected[order]
    return vectors[selected].copy(), errors[selected].copy()


def repair_clusters(cp: Any, centroids: Any, total_counts: Any, missed: Any, window_counts: Any, reservoir_vectors: list[Any], reservoir_errors: list[Any], fallback: Any) -> int:
    missed[:] = cp.where(window_counts == 0, missed + 1, 0)
    stale = cp.where((total_counts == 0) | (missed >= 2))[0]
    count = int(stale.size)
    if count == 0:
        return 0
    candidates = cp.concatenate(reservoir_vectors, axis=0) if reservoir_vectors else fallback
    errors = cp.concatenate(reservoir_errors, axis=0) if reservoir_errors else cp.zeros(candidates.shape[0], dtype=cp.float32)
    order = cp.argsort(errors)[::-1]
    candidates = candidates[order]
    if candidates.shape[0] < count:
        repeats = math.ceil(count / candidates.shape[0])
        candidates = cp.tile(candidates, (repeats, 1))
    centroids[stale] = candidates[:count]
    total_counts[stale] = cp.float64(1.0)
    missed[stale] = 0
    return count


def probe_stats(cp: Any, arrays: list[Any], moments: np.ndarray, centroids: Any, probe: dict[int, np.ndarray]) -> dict[str, Any]:
    total_sse = 0.0
    total_energy = 0.0
    used = cp.zeros(CODE_COUNT, dtype=cp.bool_)
    for ordinal in FIT_ORDINALS:
        all_vectors = normalized_vectors(cp, arrays[ordinal], moments, ordinal)
        indices = cp.asarray(probe[ordinal])
        vectors = all_vectors[indices]
        assigned, distance = exact_assign(cp, vectors, centroids)
        used[assigned] = True
        _, row_scale = decoded_location(cp, moments, ordinal)
        vector_scale2 = cp.repeat(row_scale[:, 0] * row_scale[:, 0], VECTORS_PER_ROW)
        raw = arrays[ordinal].reshape(-1, VECTOR_DIM)[indices]
        total_sse += finite_float(cp.asnumpy(cp.sum(distance * vector_scale2[indices], dtype=cp.float64)), "probe SSE")
        total_energy += finite_float(cp.asnumpy(cp.sum(cp.multiply(raw, raw, dtype=cp.float64))), "probe energy")
    return {
        "relative_residual_energy": total_sse / total_energy,
        "codes_used": int(cp.asnumpy(cp.count_nonzero(used))),
        "sse": total_sse,
        "source_energy": total_energy,
    }


def full_lloyd_pass(cp: Any, arrays: list[Any], moments: np.ndarray, centroids: Any, fallback: Any) -> tuple[Any, dict[str, Any]]:
    sums = cp.zeros((CODE_COUNT, VECTOR_DIM), dtype=cp.float64)
    counts = cp.zeros(CODE_COUNT, dtype=cp.float64)
    reservoir_vectors: list[Any] = []
    reservoir_errors: list[Any] = []
    objective = 0.0
    vectors_seen = 0
    for ordinal in FIT_ORDINALS:
        vectors = normalized_vectors(cp, arrays[ordinal], moments, ordinal)
        for start in range(0, VECTORS_PER_MATRIX, EVAL_BATCH):
            batch = vectors[start : start + EVAL_BATCH]
            assigned, error = exact_assign(cp, batch, centroids)
            unique, local_counts, local_sums = segmented_sums(cp, assigned, batch)
            counts[unique] += local_counts
            sums[unique] += local_sums
            rv, re = reservoir_chunk(cp, batch, error)
            reservoir_vectors.append(rv)
            reservoir_errors.append(re)
            objective += finite_float(cp.asnumpy(cp.sum(error, dtype=cp.float64)), "Lloyd objective")
            vectors_seen += int(batch.shape[0])
    occupied = counts > 0
    centroids[occupied] = (sums[occupied] / counts[occupied, None]).astype(cp.float32)
    empty = cp.where(~occupied)[0]
    empty_count = int(empty.size)
    if empty_count:
        candidates = cp.concatenate(reservoir_vectors, axis=0)
        errors = cp.concatenate(reservoir_errors, axis=0)
        candidates = candidates[cp.argsort(errors)[::-1]]
        if candidates.shape[0] < empty_count:
            candidates = cp.concatenate((candidates, fallback), axis=0)
        centroids[empty] = candidates[:empty_count]
    return centroids, {
        "vectors_seen": vectors_seen,
        "normalized_objective_per_vector": objective / vectors_seen,
        "occupied_before_reseed": CODE_COUNT - empty_count,
        "empty_clusters_reseeded": empty_count,
    }


def train_kmeans(cp: Any, arrays: list[Any], moments: np.ndarray, seed: int, probe: dict[int, np.ndarray]) -> tuple[Any, list[dict[str, Any]], list[str]]:
    centroids = initialize_centroids(cp, arrays, moments, seed)
    fallback = centroids.copy()
    rng = cp.random.RandomState(seed)
    total_counts = cp.zeros(CODE_COUNT, dtype=cp.float64)
    window_counts = cp.zeros(CODE_COUNT, dtype=cp.int64)
    missed = cp.zeros(CODE_COUNT, dtype=cp.int8)
    reservoir_vectors: list[Any] = []
    reservoir_errors: list[Any] = []
    trace: list[dict[str, Any]] = []
    collapse: list[str] = []
    for step0 in range(STEPS):
        ordinal = FIT_ORDINALS[step0 % len(FIT_ORDINALS)]
        vectors = normalized_vectors(cp, arrays[ordinal], moments, ordinal)
        sample = rng.randint(0, VECTORS_PER_MATRIX, size=MINIBATCH, dtype=cp.int32)
        batch = vectors[sample]
        assigned, error = exact_assign(cp, batch, centroids)
        unique, local_counts, local_sums = segmented_sums(cp, assigned, batch)
        old = total_counts[unique]
        new = old + local_counts
        centroids[unique] = ((old[:, None] * centroids[unique] + local_sums) / new[:, None]).astype(cp.float32)
        total_counts[unique] = new
        window_counts[unique] += local_counts.astype(cp.int64)
        rv, re = reservoir_chunk(cp, batch, error)
        reservoir_vectors.append(rv)
        reservoir_errors.append(re)
        step = step0 + 1
        if step in CHECKPOINTS:
            repaired = repair_clusters(
                cp,
                centroids,
                total_counts,
                missed,
                window_counts,
                reservoir_vectors,
                reservoir_errors,
                fallback,
            )
            stats = probe_stats(cp, arrays, moments, centroids, probe)
            maximum_q, minimum_used = CHECKPOINTS[step]
            passed = stats["relative_residual_energy"] <= maximum_q and stats["codes_used"] >= minimum_used
            trace.append(
                {
                    "checkpoint": step,
                    "maximum_q": maximum_q,
                    "minimum_codes_used": minimum_used,
                    "clusters_reseeded": repaired,
                    "passed": passed,
                    **stats,
                }
            )
            window_counts.fill(0)
            reservoir_vectors.clear()
            reservoir_errors.clear()
            if not passed:
                collapse.append(f"collapse checkpoint {step} failed")
                break
    if not collapse:
        centroids, lloyd = full_lloyd_pass(cp, arrays, moments, centroids, fallback)
        stats = probe_stats(cp, arrays, moments, centroids, probe)
        maximum_q, minimum_used = LLOYD_CHECK
        passed = stats["relative_residual_energy"] <= maximum_q and stats["codes_used"] >= minimum_used
        trace.append(
            {
                "checkpoint": "full_lloyd_1",
                "maximum_q": maximum_q,
                "minimum_codes_used": minimum_used,
                "passed": passed,
                **lloyd,
                **stats,
            }
        )
        if not passed:
            collapse.append("collapse checkpoint full_lloyd_1 failed")
    return cp.ascontiguousarray(centroids, dtype=cp.float32), trace, collapse


def serialize_global(codebook: np.ndarray, seed: int, control: bool) -> bytes:
    table = np.asarray(codebook, dtype="<f2").tobytes(order="C")
    if len(table) != CODEBOOK_BYTES:
        raise RuntimeError("codebook serialization length changed")
    metadata = canonical_json_bytes(
        {
            "code_count": CODE_COUNT,
            "control": bool(control),
            "format": "DOCB-WE-S0-v0",
            "index_bits": INDEX_BITS,
            "seed": seed,
            "vector_dimension": VECTOR_DIM,
        }
    )
    prefix = struct.pack("<8sIIII", b"DOCBS0\0\0", 0, CODE_COUNT, VECTOR_DIM, INDEX_BITS)
    header = prefix + metadata
    if len(header) > GLOBAL_HEADER_BYTES:
        raise RuntimeError("global header overflow")
    header += bytes(GLOBAL_HEADER_BYTES - len(header))
    payload = header + table
    if len(payload) != GLOBAL_SIDE_BYTES:
        raise RuntimeError("global side length changed")
    return payload


def serialize_moments(moments: np.ndarray) -> bytes:
    payload = np.asarray(moments, dtype="<f2").tobytes(order="C")
    if len(payload) != PANEL_MOMENT_BYTES:
        raise RuntimeError("moment side length changed")
    return payload


def parse_global(cp: Any, payload: bytes) -> Any:
    if len(payload) != GLOBAL_SIDE_BYTES:
        raise RuntimeError("global side length mismatch")
    fields = struct.unpack_from("<8sIIII", payload, 0)
    if fields != (b"DOCBS0\0\0", 0, CODE_COUNT, VECTOR_DIM, INDEX_BITS):
        raise RuntimeError("global side header mismatch")
    host = np.frombuffer(payload, dtype="<f2", count=CODE_COUNT * VECTOR_DIM, offset=GLOBAL_HEADER_BYTES)
    host = host.astype(np.float32).reshape(CODE_COUNT, VECTOR_DIM)
    if not np.isfinite(host).all():
        raise RuntimeError("non-finite canonical FP16 codebook")
    return cp.asarray(host)


def parse_moments(payload: bytes) -> np.ndarray:
    if len(payload) != PANEL_MOMENT_BYTES:
        raise RuntimeError("moment side length mismatch")
    moments = np.frombuffer(payload, dtype="<f2").copy().reshape(MATRICES, ROWS, 2)
    if not np.isfinite(moments).all():
        raise RuntimeError("non-finite canonical FP16 moments")
    return moments


def exact_evaluation(cp: Any, arrays: list[Any], codebook: Any, moments: np.ndarray) -> dict[str, Any]:
    by_expert: list[dict[str, Any]] = []
    total_sse = 0.0
    total_energy = 0.0
    for slot in HOLDOUT_SLOTS:
        expert_sse = 0.0
        expert_energy = 0.0
        matrix_rows: list[dict[str, Any]] = []
        for role in range(3):
            ordinal = 3 * slot + role
            vectors = normalized_vectors(cp, arrays[ordinal], moments, ordinal)
            _, row_scale = decoded_location(cp, moments, ordinal)
            vector_scale2 = cp.repeat(row_scale[:, 0] * row_scale[:, 0], VECTORS_PER_ROW)
            raw = arrays[ordinal].reshape(-1, VECTOR_DIM)
            sse = 0.0
            energy = 0.0
            used = cp.zeros(CODE_COUNT, dtype=cp.bool_)
            for start in range(0, VECTORS_PER_MATRIX, EVAL_BATCH):
                batch = vectors[start : start + EVAL_BATCH]
                assigned, distance = exact_assign(cp, batch, codebook)
                used[assigned] = True
                stop = start + batch.shape[0]
                sse += finite_float(cp.asnumpy(cp.sum(distance * vector_scale2[start:stop], dtype=cp.float64)), "heldout SSE")
                energy += finite_float(cp.asnumpy(cp.sum(cp.multiply(raw[start:stop], raw[start:stop], dtype=cp.float64))), "heldout energy")
            expert_sse += sse
            expert_energy += energy
            matrix_rows.append(
                {
                    "matrix_ordinal": ordinal,
                    "relative_residual_energy": sse / energy,
                    "codes_used": int(cp.asnumpy(cp.count_nonzero(used))),
                    "sse": sse,
                    "source_energy": energy,
                }
            )
        total_sse += expert_sse
        total_energy += expert_energy
        by_expert.append(
            {
                "slot": slot,
                "layer": EXPECTED_SLOTS[slot][0],
                "expert": EXPECTED_SLOTS[slot][1],
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
    fixed_bits = GLOBAL_SIDE_BYTES * 8 + expert_count * (
        INDEX_BYTES_PER_EXPERT + ROW_MOMENT_BYTES_PER_EXPERT + LOCAL_HEADER_BYTES
    ) * 8
    prefix_bpw = fixed_bits / values
    required_q = TARGET_F / (2.0 ** (2.0 * prefix_bpw))
    rows: list[dict[str, Any]] = []
    for requested in RATES:
        physical = math.ceil(requested * values / 8.0)
        local_total = physical - GLOBAL_SIDE_BYTES
        local_min, remainder = divmod(local_total, expert_count)
        local_max = local_min + int(remainder != 0)
        residual = physical - GLOBAL_SIDE_BYTES - expert_count * (
            INDEX_BYTES_PER_EXPERT + ROW_MOMENT_BYTES_PER_EXPERT + LOCAL_HEADER_BYTES
        )
        cold = GLOBAL_SIDE_BYTES + 4096 * math.ceil(local_max / 4096.0)
        rows.append(
            {
                "requested_bpw": requested,
                "actual_bpw": physical * 8.0 / values,
                "expert_count": expert_count,
                "physical_bytes": physical,
                "global_side_bytes": GLOBAL_SIDE_BYTES,
                "local_total_bytes": local_total,
                "local_frame_min_bytes": local_min,
                "local_frame_max_bytes": local_max,
                "large_frame_count": remainder,
                "index_bytes_per_expert": INDEX_BYTES_PER_EXPERT,
                "row_moment_bytes_per_expert": ROW_MOMENT_BYTES_PER_EXPERT,
                "local_header_bytes_per_expert": LOCAL_HEADER_BYTES,
                "total_residual_bytes": residual,
                "residual_bpw": residual * 8.0 / values,
                "cold_expert_bytes_4k": cold,
                "cold_read_amplification": cold / (physical / expert_count),
            }
        )
    return rows, prefix_bpw, required_q


def oracle(evaluation: dict[str, Any], prefix_bpw: float) -> dict[str, Any]:
    factor = 2.0 ** (2.0 * prefix_bpw)
    q = float(evaluation["relative_residual_energy"])
    f_value = q * factor
    experts = []
    for row in evaluation["heldout_experts"]:
        item = dict(row)
        item["F_oracle"] = float(row["relative_residual_energy"]) * factor
        item["s_oracle"] = -0.5 * math.log2(item["F_oracle"])
        experts.append(item)
    return {
        "relative_residual_energy": q,
        "F_oracle": f_value,
        "s_oracle": -0.5 * math.log2(f_value),
        "heldout_experts": experts,
    }


def run_one(cp: Any, output: Path, seed: int, source: list[Any], control: list[Any], moments: np.ndarray, probe: dict[int, np.ndarray], prefix_bpw: float) -> dict[str, Any]:
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir()
    report: dict[str, Any] = {"seed": seed}
    for label, arrays, is_control in (("source", source, False), ("gaussian", control, True)):
        started = time.monotonic()
        codebook, trace, collapse = train_kmeans(cp, arrays, moments, seed, probe)
        global_payload = serialize_global(cp.asnumpy(codebook), seed, is_control)
        moment_payload = serialize_moments(moments)
        global_path = seed_dir / f"{label}_global_side.bin"
        moment_path = seed_dir / f"{label}_row_moments.bin"
        global_path.write_bytes(global_payload)
        moment_path.write_bytes(moment_payload)
        finite_codebook = parse_global(cp, global_payload)
        finite_moments = parse_moments(moment_payload)
        evaluation = exact_evaluation(cp, arrays, finite_codebook, finite_moments)
        report[label] = {
            "collapse_reasons": collapse,
            "training_trace": trace,
            "global_side_relpath": str(global_path.relative_to(output)).replace("\\", "/"),
            "global_side_bytes": len(global_payload),
            "global_side_sha256": hashlib.sha256(global_payload).hexdigest(),
            "row_moments_relpath": str(moment_path.relative_to(output)).replace("\\", "/"),
            "row_moments_bytes": len(moment_payload),
            "row_moments_sha256": hashlib.sha256(moment_payload).hexdigest(),
            "evaluation": evaluation,
            "oracle": oracle(evaluation, prefix_bpw),
            "elapsed_seconds": time.monotonic() - started,
        }
        del codebook, global_payload, moment_payload, finite_codebook, finite_moments, evaluation
        cp.get_default_memory_pool().free_all_blocks()
    report["matched_advantage_s"] = (
        report["source"]["oracle"]["s_oracle"] - report["gaussian"]["oracle"]["s_oracle"]
    )
    return report


def decide(reports: list[dict[str, Any]]) -> tuple[str, list[str]]:
    source = [row["source"] for row in reports]
    if min(float(row["oracle"]["F_oracle"]) for row in source) > TARGET_F:
        return "KILL", ["both fixed direct-output seeds fail the favorable held-out oracle"]
    margin = TARGET_S + PROMOTION_MARGIN_S
    no_collapse = all(not row["collapse_reasons"] for row in source)
    both_margin = all(float(row["oracle"]["s_oracle"]) >= margin for row in source)
    every_expert = all(
        float(expert["F_oracle"]) <= TARGET_F
        for row in source
        for expert in row["oracle"]["heldout_experts"]
    )
    positive_matched = all(float(row["matched_advantage_s"]) > 0.0 for row in reports)
    if no_collapse and both_margin and every_expert and positive_matched:
        return "PROMOTE_TO_FRESH_AUXILIARY_CONFIRMATION_ONLY", [
            "both seeds clear the fixed margin without collapse",
            "every held-out expert clears F<=0.8",
            "both matched Gaussian advantages are positive",
        ]
    reasons = []
    if not no_collapse:
        reasons.append("at least one source seed tripped a fixed collapse checkpoint")
    if not both_margin:
        reasons.append("both source seeds do not clear the promotion margin")
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
    probe, probe_sha256 = frozen_probe_indices()
    panel_ledger, prefix_bpw, required_q = rate_ledger(EXPERTS)
    layer_ledger, layer_prefix, layer_required_q = rate_ledger(128)
    reports = [run_one(cp, output, seed, source, control, moments, probe, prefix_bpw) for seed in SEEDS]
    status, reasons = decide(reports)
    device = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "schema": "direct-output-codebook-whole-expert-stage0-result-v0",
        "status": status,
        "decision_reasons": reasons,
        "claim_boundary": (
            "Six-expert direct-output K-means cell only. A kill is not a converse for arbitrary VQ. "
            "A promotion is ideal-Gaussian-residual feasibility, not a finite codec, model-wide generalization, "
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
        "probe_indices_sha256": probe_sha256,
        "six_expert_rate_ledger": panel_ledger,
        "six_expert_fixed_prefix_bpw": prefix_bpw,
        "six_expert_required_first_stage_relative_residual_energy": required_q,
        "hypothetical_128_expert_layer_ledger": {
            "claim_boundary": "Arithmetic projection only; no layer-wide generalization evidence.",
            "fixed_prefix_bpw": layer_prefix,
            "required_first_stage_relative_residual_energy": layer_required_q,
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
    print(json.dumps({"status": status, "result": str(output / "result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
