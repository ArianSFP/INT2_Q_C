#!/usr/bin/env python3
"""Cross-fitted nonparametric rate--distortion oracle for the Qwen MoE panel.

This is a source-only, CPU-only early-kill experiment.  It deliberately does
not claim to be a serialized codec.  It asks a narrower question: after
conditioning on a whole-matrix mean/RMS and after exposing either raw or the
already-sealed XKLT coordinates, does a nonparametric scalar/2-D/4-D test
channel reveal the 0.160964... bit/weight advantage required for distortion
20% below an iid Gaussian at the same physical rate?

Leakage boundary
----------------
For each of six outer folds, every reconstruction table, coordinate rotation,
and BA output prior is learned from the other five experts.  All three matrices
of the held-out expert are then evaluated without refitting.  The held-out rate
is the variational rate

    E_x KL(P_beta(y|x) || q_train(y)),

not mutual information recomputed with a held-out output marginal.  Thus an
output-prior mismatch is charged.  A deterministic moment-matched Gaussian
control uses exactly the same folds, sample counts, table geometry, and solver.

Bound interpretation
--------------------
The empirical BA channel is an achievability oracle on deterministic samples,
not a converse for every conceivable codec.  To make the *kill* decision
deliberately favorable to the candidate, the result also reports a calibrated
gain that divides out the complete measured loss of the matched-Gaussian
control.  It then adds a fold-bootstrap upper allowance and a fixed numerical
allowance.  Only if even that free-side optimistic quantity is far below the
required gain is the branch killed.

No CuPy, torch, CUDA, or GPU subprocess is imported or invoked here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


WEIGHTS = 28_311_552
EXPERTS = 6
COMPONENTS = 3
ROWS = 768
COLS = 2048
VALUES = ROWS * COLS
SOURCE_BYTES = VALUES * 2
PHYSICAL_BYTES_AT_2P5 = 8_847_360
TARGET_GAIN_BPW = -0.5 * math.log2(0.8)
TARGET_RATES = (2.15, 2.5)
DEFAULT_BETAS = (3.0, 4.5, 6.0, 8.0, 10.0, 13.0, 17.0, 23.0, 32.0, 48.0)
REPRESENTATIONS = ("raw", "xklt")
ROLE_NAMES = ("gate", "up_or_k0", "down_or_k1")
TENSOR_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.memmap(path, dtype="<u2", mode="r", shape=shape)
    result = (np.asarray(words, dtype=np.uint32) << np.uint32(16)).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite BF16 source: {path}")
    return result


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def coprime_stride(modulus: int, seed: int) -> int:
    candidate = int(seed % modulus) | 1
    while math.gcd(candidate, modulus) != 1:
        candidate += 2
        if candidate >= modulus:
            candidate = 1
    return candidate


def deterministic_indices(total: int, count: int, seed: int) -> np.ndarray:
    count = min(total, count)
    start = seed % total
    stride = coprime_stride(total, seed >> 17)
    result = (start + stride * np.arange(count, dtype=np.uint64)) % total
    return result.astype(np.int64)


def parse_int_map(text: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for field in text.split(","):
        key, value = field.split(":", 1)
        result[int(key)] = int(value)
    return result


def source_matrix(path: Path, role: str) -> np.ndarray:
    if role in ("gate", "up"):
        return np.asarray(bf16(path, (ROWS, COLS)), dtype=np.float32, order="C")
    if role == "down":
        return np.asarray(bf16(path, (COLS, ROWS)).T, dtype=np.float32, order="C")
    raise ValueError(role)


def normalized_sample(
    matrix: np.ndarray,
    dimension: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    flat = np.ravel(matrix)
    total_sum = float(np.sum(flat, dtype=np.float64))
    total_energy = float(np.dot(flat.astype(np.float64), flat.astype(np.float64)))
    mean = total_sum / float(flat.size)
    centered_energy = total_energy - float(flat.size) * mean * mean
    if not centered_energy > 0.0:
        raise ValueError("non-positive centered source energy")
    rms = math.sqrt(centered_energy / float(flat.size))
    per_row = COLS // dimension
    vectors = matrix.reshape(ROWS, per_row, dimension)
    indices = deterministic_indices(ROWS * per_row, count, seed)
    row = indices // per_row
    column = indices % per_row
    selected = np.asarray(vectors[row, column], dtype=np.float64)
    selected = ((selected - mean) / rms).astype(np.float32)
    sample_hash = hashlib.sha256(selected.astype("<f4", copy=False).tobytes()).hexdigest()
    return selected, {
        "source_values": int(flat.size),
        "sample_vectors": int(selected.shape[0]),
        "mean_fp64": mean,
        "rms_about_mean_fp64": rms,
        "source_energy_fp64": total_energy,
        "centered_energy_fp64": centered_energy,
        "sample_fp32_sha256": sample_hash,
        "selection_start": int(seed % (ROWS * per_row)),
        "selection_stride": int(coprime_stride(ROWS * per_row, seed >> 17)),
    }


@dataclass
class MatrixSamples:
    expert: int
    component: int
    representation: str
    dimension: int
    values: np.ndarray
    metadata: dict[str, Any]


def load_plan(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    clean = dict(plan)
    expected = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(
        json.dumps(
            clean,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise ValueError("plan internal seal mismatch")
    if plan.get("coverage", {}).get("weights") != WEIGHTS:
        raise ValueError("plan is not the pinned 28,311,552-weight panel")
    if len(plan.get("sources", [])) != EXPERTS * COMPONENTS:
        raise ValueError("plan source count changed")
    return plan


def parse_xklt_coefficients(header_path: Path) -> list[tuple[float, float]]:
    payload = header_path.read_bytes()
    if len(payload) != 128 or payload[:8] != b"PLRLOC3\0":
        raise ValueError("wrong expert-affine header")
    coefficients = struct.unpack_from("<12f", payload, 32)
    codes = struct.unpack_from("<6h", payload, 80)
    result = []
    for expert, code in enumerate(codes):
        theta = code * math.pi / 32768.0
        expected = (np.float32(math.cos(theta)), np.float32(math.sin(theta)))
        pair = (np.float32(coefficients[2 * expert]), np.float32(coefficients[2 * expert + 1]))
        if pair[0].tobytes() != expected[0].tobytes() or pair[1].tobytes() != expected[1].tobytes():
            raise ValueError("header XKLT coefficient/code mismatch")
        result.append((float(pair[0]), float(pair[1])))
    return result


def build_sample_bank(
    plan_path: Path,
    dimensions: Iterable[int],
    sample_counts: dict[int, int],
) -> tuple[dict[tuple[str, int, int, int], MatrixSamples], dict[str, Any]]:
    plan = load_plan(plan_path)
    plan_dir = plan_path.parent
    source_root = Path(plan["source_root"]).resolve(strict=True)
    header_path = plan_dir / plan["assets"]["header.bin"]["relpath"]
    if sha256_file(header_path) != plan["assets"]["header.bin"]["sha256"]:
        raise ValueError("header hash mismatch")
    coefficients = parse_xklt_coefficients(header_path)
    sources = plan["sources"]
    source_rows = []
    bank: dict[tuple[str, int, int, int], MatrixSamples] = {}
    for expert in range(EXPERTS):
        triplet = sources[3 * expert : 3 * expert + 3]
        if [row["role"] for row in triplet] != ["gate", "up", "down"]:
            raise ValueError(f"source role order changed for expert {expert}")
        opened = []
        for row in triplet:
            match = TENSOR_RE.fullmatch(row["tensor"])
            if match is None:
                raise ValueError(f"unexpected tensor: {row['tensor']}")
            path = (source_root / row["source_relpath"]).resolve(strict=True)
            if source_root not in path.parents or path.stat().st_size != SOURCE_BYTES:
                raise ValueError("source escaped root or has wrong length")
            actual_hash = sha256_file(path)
            if actual_hash != row["source_bf16_sha256"]:
                raise ValueError(f"source hash mismatch: {row['tensor']}")
            opened.append(source_matrix(path, row["role"]))
            source_rows.append(
                {
                    "matrix_ordinal": row["matrix_ordinal"],
                    "tensor": row["tensor"],
                    "bytes": row["bytes"],
                    "sha256": actual_hash,
                }
            )
        gate, up, down_t = opened
        co, si = coefficients[expert]
        # Compute in FP32 exactly as an inference-friendly header transform;
        # do not round the transformed values back to BF16 in this oracle.
        k0 = (np.float32(co) * up + np.float32(si) * down_t).astype(np.float32)
        k1 = (-np.float32(si) * up + np.float32(co) * down_t).astype(np.float32)
        matrices = {
            "raw": (gate, up, down_t),
            "xklt": (gate, k0, k1),
        }
        source_digest = hashlib.sha256(
            "".join(row["source_bf16_sha256"] for row in triplet).encode("ascii")
        ).hexdigest()
        for representation, components in matrices.items():
            for component, matrix in enumerate(components):
                for dimension in dimensions:
                    seed = stable_seed(
                        "QWEN-NONPARAMETRIC-BA-v1",
                        source_digest,
                        representation,
                        component,
                        dimension,
                    )
                    values, metadata = normalized_sample(
                        matrix, dimension, sample_counts[dimension], seed
                    )
                    metadata.update(
                        {
                            "expert_ordinal": expert,
                            "component_ordinal": component,
                            "component_name": ROLE_NAMES[component],
                            "representation": representation,
                            "dimension": dimension,
                        }
                    )
                    bank[(representation, dimension, expert, component)] = MatrixSamples(
                        expert, component, representation, dimension, values, metadata
                    )
        del opened, gate, up, down_t, k0, k1
    provenance = {
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_lock_sha256": plan["lock_sha256"],
        "header_path": str(header_path.resolve()),
        "header_sha256": sha256_file(header_path),
        "xklt_coefficients_fp32": [list(pair) for pair in coefficients],
        "source_root": str(source_root),
        "sources": source_rows,
    }
    return bank, provenance


def orthogonal_rotation(train: np.ndarray) -> np.ndarray:
    dimension = train.shape[1]
    covariance = np.asarray(train, dtype=np.float64).T @ np.asarray(train, dtype=np.float64)
    covariance /= float(train.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    rotation = eigenvectors[:, order]
    # Canonicalize eigenvector signs for literal reproducibility.
    for column in range(dimension):
        pivot = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[pivot, column] < 0.0:
            rotation[:, column] *= -1.0
    return rotation.astype(np.float64)


def quantile_centers(values: np.ndarray, levels: int) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64), axis=None)
    boundaries = np.linspace(0, sorted_values.size, levels + 1, dtype=np.int64)
    result = np.empty(levels, dtype=np.float64)
    for index in range(levels):
        begin, end = int(boundaries[index]), int(boundaries[index + 1])
        if begin == end:
            result[index] = sorted_values[min(begin, sorted_values.size - 1)]
        else:
            result[index] = float(np.mean(sorted_values[begin:end], dtype=np.float64))
    return result


def product_reproductions(train: np.ndarray, levels: int) -> np.ndarray:
    axes = [quantile_centers(train[:, axis], levels) for axis in range(train.shape[1])]
    result = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
    if result.shape != (levels ** train.shape[1], train.shape[1]):
        raise AssertionError("Cartesian reproduction geometry changed")
    return result.astype(np.float32)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    distance = (
        np.sum(x64 * x64, axis=1, dtype=np.float64)[:, None]
        + np.sum(y64 * y64, axis=1, dtype=np.float64)[None, :]
        - 2.0 * (x64 @ y64.T)
    )
    np.maximum(distance, 0.0, out=distance)
    return distance.astype(np.float32)


def ba_update(
    values: np.ndarray,
    reproduction: np.ndarray,
    beta: float,
    q: np.ndarray,
    chunk: int,
    collect_score: bool,
) -> tuple[np.ndarray, float, float]:
    q64 = np.maximum(np.asarray(q, dtype=np.float64), 1e-300)
    q64 /= float(np.sum(q64))
    logq = np.log(q64)
    qsum = np.zeros_like(q64)
    total_distortion = 0.0
    total_kl = 0.0
    total = values.shape[0]
    for begin in range(0, total, chunk):
        sample = values[begin : begin + chunk]
        distance = np.asarray(squared_distances(sample, reproduction), dtype=np.float64)
        logits = logq[None, :] - beta * distance
        maximum = np.max(logits, axis=1, keepdims=True)
        probability = np.exp(logits - maximum)
        probability /= np.sum(probability, axis=1, keepdims=True)
        qsum += np.sum(probability, axis=0, dtype=np.float64)
        if collect_score:
            total_distortion += float(np.sum(probability * distance, dtype=np.float64))
            log_probability = np.log(np.maximum(probability, 1e-300))
            total_kl += float(
                np.sum(probability * (log_probability - logq[None, :]), dtype=np.float64)
            )
    return (
        qsum / float(total),
        total_distortion / float(total),
        total_kl / (float(total) * math.log(2.0)),
    )


def fit_prior(
    train: np.ndarray,
    reproduction: np.ndarray,
    beta: float,
    max_iterations: int,
    tolerance: float,
    chunk: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    q = np.full(reproduction.shape[0], 1.0 / reproduction.shape[0], dtype=np.float64)
    converged = False
    delta = math.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        q_new, _, _ = ba_update(train, reproduction, beta, q, chunk, False)
        # A tiny floor permits a coordinate to reactivate at larger beta and
        # avoids an initialization-dependent absorbing zero.
        q_new = np.maximum(q_new, 1e-15)
        q_new /= float(np.sum(q_new))
        delta = float(np.sum(np.abs(q_new - q), dtype=np.float64))
        q = q_new
        if delta <= tolerance:
            converged = True
            break
    _, train_distortion, train_rate = ba_update(
        train, reproduction, beta, q, chunk, True
    )
    return q, {
        "iterations": iteration,
        "converged": converged,
        "terminal_l1_delta": delta,
        "active_probability_entries_gt_1e_9": int(np.sum(q > 1e-9)),
        "train_rate_bits_per_vector": train_rate,
        "train_distortion_sum_per_vector": train_distortion,
    }


def evaluate_prior(
    test: np.ndarray,
    reproduction: np.ndarray,
    beta: float,
    q: np.ndarray,
    chunk: int,
) -> tuple[float, float]:
    _, distortion, rate = ba_update(test, reproduction, beta, q, chunk, True)
    return rate / test.shape[1], distortion / test.shape[1]


def exact_moment_gaussian(
    count: int,
    covariance: np.ndarray,
    seed: int,
) -> np.ndarray:
    dimension = covariance.shape[0]
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    z = rng.standard_normal((count, dimension), dtype=np.float64)
    z -= np.mean(z, axis=0, keepdims=True, dtype=np.float64)
    zcov = (z.T @ z) / float(count)
    zeval, zevec = np.linalg.eigh(zcov)
    zwhite = z @ (zevec @ np.diag(1.0 / np.sqrt(np.maximum(zeval, 1e-12))) @ zevec.T)
    ceval, cevec = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    color = cevec @ np.diag(np.sqrt(np.maximum(ceval, 0.0))) @ cevec.T
    result = zwhite @ color
    return result.astype(np.float32)


def lower_time_share(points: list[dict[str, float]], target_rate: float) -> dict[str, Any]:
    candidates = [(0.0, 1.0, "zero")]
    for index, point in enumerate(points):
        rate = float(point["rate_bpw"])
        distortion = float(point["distortion_per_coordinate"])
        if math.isfinite(rate) and math.isfinite(distortion) and rate >= 0.0:
            candidates.append((rate, distortion, str(index)))
    best = (math.inf, None)
    for left in range(len(candidates)):
        r0, d0, label0 = candidates[left]
        if r0 <= target_rate and d0 < best[0]:
            best = (d0, {"left": label0, "right": label0, "right_weight": 0.0})
        for right in range(left + 1, len(candidates)):
            r1, d1, label1 = candidates[right]
            if (r0 - target_rate) * (r1 - target_rate) > 0.0 or r0 == r1:
                continue
            weight = (target_rate - r0) / (r1 - r0)
            if not 0.0 <= weight <= 1.0:
                continue
            distortion = (1.0 - weight) * d0 + weight * d1
            if distortion < best[0]:
                best = (
                    distortion,
                    {"left": label0, "right": label1, "right_weight": weight},
                )
    if best[1] is None:
        raise ValueError(f"curve does not cover target rate {target_rate}")
    return {"distortion": best[0], "time_share": best[1]}


def fold_curve(
    train: np.ndarray,
    test: np.ndarray,
    levels: int,
    betas: Iterable[float],
    max_iterations: int,
    tolerance: float,
    chunk: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rotation = orthogonal_rotation(train)
    train_rotated = (np.asarray(train, dtype=np.float64) @ rotation).astype(np.float32)
    test_rotated = (np.asarray(test, dtype=np.float64) @ rotation).astype(np.float32)
    reproduction = product_reproductions(train_rotated, levels)
    rows = []
    for beta in betas:
        started = time.monotonic()
        q, fit = fit_prior(
            train_rotated,
            reproduction,
            float(beta),
            max_iterations,
            tolerance,
            chunk,
        )
        rate, distortion = evaluate_prior(
            test_rotated, reproduction, float(beta), q, chunk
        )
        rows.append(
            {
                "beta": float(beta),
                "rate_bpw": rate,
                "distortion_per_coordinate": distortion,
                "fit": fit,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    model = {
        "rotation_fp64": rotation.tolist(),
        "reproduction_points": int(reproduction.shape[0]),
        "reproduction_dimension": int(reproduction.shape[1]),
        "reproduction_fp32_sha256": hashlib.sha256(
            reproduction.astype("<f4", copy=False).tobytes()
        ).hexdigest(),
    }
    return rows, model


def covariance(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return (x.T @ x) / float(x.shape[0])


def run_branch(
    bank: dict[tuple[str, int, int, int], MatrixSamples],
    representation: str,
    dimension: int,
    levels: int,
    betas: Iterable[float],
    max_iterations: int,
    tolerance: float,
    chunk: int,
    target_rates: Iterable[float],
) -> dict[str, Any]:
    folds = []
    for heldout in range(EXPERTS):
        print(
            json.dumps(
                {
                    "event": "fold_start",
                    "representation": representation,
                    "dimension": dimension,
                    "heldout_expert": heldout,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        component_rows = []
        for component in range(COMPONENTS):
            training_parts = [
                bank[(representation, dimension, expert, component)].values
                for expert in range(EXPERTS)
                if expert != heldout
            ]
            train = np.concatenate(training_parts, axis=0)
            test_item = bank[(representation, dimension, heldout, component)]
            test = test_item.values
            source_curve, source_model = fold_curve(
                train,
                test,
                levels,
                betas,
                max_iterations,
                tolerance,
                chunk,
            )
            train_cov = covariance(train)
            test_cov = covariance(test)
            gaussian_train = exact_moment_gaussian(
                train.shape[0],
                train_cov,
                stable_seed("gaussian-train", representation, dimension, heldout, component),
            )
            gaussian_test = exact_moment_gaussian(
                test.shape[0],
                test_cov,
                stable_seed("gaussian-test", representation, dimension, heldout, component),
            )
            gaussian_curve, gaussian_model = fold_curve(
                gaussian_train,
                gaussian_test,
                levels,
                betas,
                max_iterations,
                tolerance,
                chunk,
            )
            scores = {}
            for target_rate in target_rates:
                scores[format(target_rate, ".12g")] = {
                    "source": lower_time_share(source_curve, target_rate),
                    "matched_gaussian": lower_time_share(gaussian_curve, target_rate),
                }
            component_rows.append(
                {
                    "component": component,
                    "component_name": ROLE_NAMES[component],
                    "test_metadata": test_item.metadata,
                    "train_vectors": int(train.shape[0]),
                    "test_vectors": int(test.shape[0]),
                    "train_covariance_fp64": train_cov.tolist(),
                    "test_covariance_fp64": test_cov.tolist(),
                    "source_curve": source_curve,
                    "matched_gaussian_curve": gaussian_curve,
                    "source_model": source_model,
                    "matched_gaussian_model": gaussian_model,
                    "scores": scores,
                }
            )
        folds.append({"heldout_expert": heldout, "components": component_rows})
    return {
        "representation": representation,
        "dimension": dimension,
        "levels_per_axis": levels,
        "reproduction_points": levels**dimension,
        "folds": folds,
    }


def physical_side_ledger(dimension: int, points: int) -> dict[str, Any]:
    normalization_bits = EXPERTS * COMPONENTS * 2 * 32
    rotation_bits = COMPONENTS * dimension * dimension * 32
    reproduction_bits = COMPONENTS * points * dimension * 16
    prior_frequency_bits = COMPONENTS * points * 16
    global_choice_bits = 64
    total = (
        normalization_bits
        + rotation_bits
        + reproduction_bits
        + prior_frequency_bits
        + global_choice_bits
    )
    common_bytes = (total + 7) // 8
    expert_share = PHYSICAL_BYTES_AT_2P5 / EXPERTS
    return {
        "normalization": {
            "description": "FP32 mean and RMS for all 18 matrices",
            "bits": normalization_bits,
        },
        "orthogonal_rotations": {
            "description": "three dense FP32 d-by-d coordinate rotations",
            "bits": rotation_bits,
        },
        "reproduction_tables": {
            "description": "three full FP16 vector reconstruction tables",
            "bits": reproduction_bits,
        },
        "prior_frequency_tables": {
            "description": "three uint16 BA output-frequency tables",
            "bits": prior_frequency_bits,
        },
        "global_choice_and_header": {"bits": global_choice_bits},
        "per_value_or_per_group_label_bits": 0,
        "total_bits": total,
        "total_bytes_rounded": common_bytes,
        "bpw": total / WEIGHTS,
        "read_amplification": {
            "accounting_scope": (
                "conservative graft onto expert-affine layout; common model is read "
                "cold for every routed expert"
            ),
            "base_geometric_expert_affine": 10.0 / 9.0,
            "additional_common_table_amplification_at_2p5": common_bytes / expert_share,
            "conservative_total": 10.0 / 9.0 + common_bytes / expert_share,
            "strictly_below_2x": 10.0 / 9.0 + common_bytes / expert_share < 2.0,
        },
    }


def aggregate_branch(branch: dict[str, Any], physical_rates: Iterable[float]) -> dict[str, Any]:
    ledger = physical_side_ledger(branch["dimension"], branch["reproduction_points"])
    side_bpw = float(ledger["bpw"])
    rate_rows = {}
    expert_gain_rows: dict[str, list[float]] = {}
    for physical_rate in physical_rates:
        payload_rate = physical_rate - side_bpw
        if payload_rate <= 0.0:
            raise ValueError("side model consumes entire physical rate")
        source_sse = 0.0
        gaussian_sse = 0.0
        centered_energy = 0.0
        source_sse_by_expert = np.zeros(EXPERTS, dtype=np.float64)
        gaussian_sse_by_expert = np.zeros(EXPERTS, dtype=np.float64)
        energy_by_expert = np.zeros(EXPERTS, dtype=np.float64)
        for fold in branch["folds"]:
            expert = int(fold["heldout_expert"])
            for component in fold["components"]:
                energy = float(component["test_metadata"]["centered_energy_fp64"])
                source = lower_time_share(component["source_curve"], payload_rate)["distortion"]
                gaussian = lower_time_share(
                    component["matched_gaussian_curve"], payload_rate
                )["distortion"]
                source_sse += energy * source
                gaussian_sse += energy * gaussian
                centered_energy += energy
                source_sse_by_expert[expert] += energy * source
                gaussian_sse_by_expert[expert] += energy * gaussian
                energy_by_expert[expert] += energy
        source_distortion = source_sse / centered_energy
        gaussian_control_distortion = gaussian_sse / centered_energy
        calibrated_f = source_distortion / gaussian_control_distortion
        calibrated_gain = -0.5 * math.log2(calibrated_f)
        exact_f = source_distortion / math.pow(2.0, -2.0 * physical_rate)
        exact_gain = -0.5 * math.log2(exact_f)
        expert_gains = []
        for expert in range(EXPERTS):
            ratio = source_sse_by_expert[expert] / gaussian_sse_by_expert[expert]
            expert_gains.append(-0.5 * math.log2(ratio))
        key = format(physical_rate, ".12g")
        expert_gain_rows[key] = expert_gains
        rate_rows[key] = {
            "physical_rate_bpw": physical_rate,
            "payload_rate_after_model_bpw": payload_rate,
            "source_normalized_distortion": source_distortion,
            "matched_gaussian_control_distortion": gaussian_control_distortion,
            "free_side_calibrated_F": calibrated_f,
            "free_side_calibrated_rate_advantage_bpw": calibrated_gain,
            "charged_F_prediction": calibrated_f * math.pow(2.0, 2.0 * side_bpw),
            "charged_rate_advantage_bpw": calibrated_gain - side_bpw,
            "direct_F_vs_exact_iid_gaussian_at_physical_rate": exact_f,
            "direct_rate_advantage_vs_exact_iid_gaussian_bpw": exact_gain,
            "target_F": 0.8,
            "target_rate_advantage_bpw": TARGET_GAIN_BPW,
            "target_met_by_charged_prediction": calibrated_gain - side_bpw >= TARGET_GAIN_BPW,
            "heldout_expert_calibrated_gain_bpw": expert_gains,
        }
    # Six experts are the independent outer-fold units.  This standard error is
    # descriptive; the final kill allowance is intentionally two-sided and then
    # adds a separate numerical allowance.
    all_gain = np.asarray(
        [value for values in expert_gain_rows.values() for value in values],
        dtype=np.float64,
    )
    return {
        "side_information_ledger": ledger,
        "rates": rate_rows,
        "heldout_gain_descriptive": {
            "minimum_bpw": float(np.min(all_gain)),
            "maximum_bpw": float(np.max(all_gain)),
            "mean_bpw": float(np.mean(all_gain)),
            "standard_error_bpw": float(np.std(all_gain, ddof=1) / math.sqrt(all_gain.size)),
        },
    }


def self_test() -> None:
    rng = np.random.default_rng(123)
    train = rng.standard_normal((2048, 1)).astype(np.float32)
    test = rng.standard_normal((1024, 1)).astype(np.float32)
    curve, model = fold_curve(
        train,
        test,
        levels=16,
        betas=(3.0, 8.0, 20.0),
        max_iterations=30,
        tolerance=1e-7,
        chunk=256,
    )
    rates = [row["rate_bpw"] for row in curve]
    distortions = [row["distortion_per_coordinate"] for row in curve]
    if not all(math.isfinite(value) for value in rates + distortions):
        raise AssertionError("non-finite BA self-test")
    if model["reproduction_points"] != 16:
        raise AssertionError("wrong self-test support")
    score = lower_time_share(curve, min(1.0, max(rates)))
    if not 0.0 < score["distortion"] < 1.0:
        raise AssertionError("invalid self-test distortion")
    print("self-test passed", json.dumps({"rates": rates, "distortions": distortions}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dimensions", default="1,2,4")
    parser.add_argument("--samples", default="1:16384,2:8192,4:4096")
    parser.add_argument("--levels", default="1:64,2:20,4:6")
    parser.add_argument(
        "--betas", default=",".join(format(value, ".12g") for value in DEFAULT_BETAS)
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=2e-7)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.plan is None or args.output is None:
        parser.error("--plan and --output are required unless --self-test is used")
    dimensions = tuple(int(value) for value in args.dimensions.split(","))
    if any(value not in (1, 2, 4) or COLS % value for value in dimensions):
        raise ValueError("dimensions must be a subset of 1,2,4")
    samples = parse_int_map(args.samples)
    levels = parse_int_map(args.levels)
    if any(value not in samples or value not in levels for value in dimensions):
        raise ValueError("sample/level map does not cover dimensions")
    representations = tuple(args.representations.split(","))
    if any(value not in REPRESENTATIONS for value in representations):
        raise ValueError("unknown representation")
    betas = tuple(float(value) for value in args.betas.split(","))
    if any(value <= 0.0 for value in betas):
        raise ValueError("betas must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    started = time.monotonic()
    bank, provenance = build_sample_bank(args.plan.resolve(strict=True), dimensions, samples)
    branches = []
    for dimension in dimensions:
        for representation in representations:
            branch = run_branch(
                bank,
                representation,
                dimension,
                levels[dimension],
                betas,
                args.max_iterations,
                args.tolerance,
                args.chunk,
                TARGET_RATES,
            )
            branch["aggregate"] = aggregate_branch(branch, TARGET_RATES)
            branches.append(branch)
            write_json(
                args.output.with_suffix(args.output.suffix + ".partial"),
                {
                    "schema": "qwen-nonparametric-ba-oracle-partial-v1",
                    "provenance": provenance,
                    "branches": branches,
                },
            )
    candidates = []
    for branch in branches:
        for rate, row in branch["aggregate"]["rates"].items():
            candidates.append(
                {
                    "representation": branch["representation"],
                    "dimension": branch["dimension"],
                    "physical_rate_bpw": float(rate),
                    "charged_gain_bpw": row["charged_rate_advantage_bpw"],
                    "free_side_gain_bpw": row["free_side_calibrated_rate_advantage_bpw"],
                    "charged_F": row["charged_F_prediction"],
                    "standard_error_bpw": branch["aggregate"]["heldout_gain_descriptive"][
                        "standard_error_bpw"
                    ],
                }
            )
    best = max(candidates, key=lambda row: row["free_side_gain_bpw"])
    numerical_allowance = 0.005
    optimistic_upper_gain = (
        best["free_side_gain_bpw"] + 2.0 * best["standard_error_bpw"] + numerical_allowance
    )
    conclusion = {
        "required_rate_advantage_bpw": TARGET_GAIN_BPW,
        "required_F": 0.8,
        "best_branch": best,
        "free_side_two_standard_error_plus_numerical_allowance_bpw": optimistic_upper_gain,
        "numerical_allowance_bpw": numerical_allowance,
        "remaining_shortfall_bpw": TARGET_GAIN_BPW - optimistic_upper_gain,
        "hard_kill_nonparametric_scalar_vector_branch": optimistic_upper_gain < TARGET_GAIN_BPW,
        "claim_boundary": (
            "A kill is strong evidence against stationary scalar/adjacent-2D/adjacent-4D "
            "non-Gaussian test channels under the tested raw/XKLT normalizations. It is not "
            "a universal converse for arbitrary long-range or semantic deterministic structure."
        ),
    }
    result = {
        "schema": "qwen-nonparametric-ba-oracle-v1",
        "created_unix_seconds": int(time.time()),
        "runtime_seconds": time.monotonic() - started,
        "cpu_only": True,
        "gpu_imports_or_subprocesses": False,
        "method": {
            "outer_split": "leave one complete expert triplet out; six folds",
            "heldout_rate": "E KL(P_beta(y|x)||q_train(y)) / vector dimension",
            "source_model": "nonparametric Cartesian reconstruction support plus BA joint prior",
            "gaussian_control": "deterministic exact-moment Gaussian with identical solver geometry",
            "representations": list(representations),
            "dimensions": list(dimensions),
            "sample_vectors_per_matrix": {str(k): samples[k] for k in dimensions},
            "levels_per_axis": {str(k): levels[k] for k in dimensions},
            "betas": list(betas),
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "chunk": args.chunk,
            "target_physical_rates_bpw": list(TARGET_RATES),
        },
        "provenance": provenance,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "branches": branches,
        "conclusion": conclusion,
    }
    result["result_payload_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    write_json(args.output, result)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    print(json.dumps(conclusion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
