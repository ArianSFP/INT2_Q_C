#!/usr/bin/env python3
"""CPU-only hidden-negentropy / orthogonal-ICA oracle for the pinned Qwen panel.

The experiment is deliberately favorable to the hypothesis.  It searches
orthogonal block rotations with KLT and three symmetric projection-pursuit
contrasts, first on the complete panel (a free-side, leaky screen), and then
refits the selected contrast in six leave-one-expert-out folds.  The held-out
fold evaluates both a cross-fitted marginal density and an entropy-coded
uniform scalar-quantizer curve.  Independently generated Gaussian samples with
the same counts and apparatus remove finite-histogram / scalar-shaping bias.

No CuPy, torch, CUDA API, or GPU subprocess is imported or invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


WEIGHTS = 28_311_552
EXPERTS = 6
ROLES = 3
ROWS = 768
COLS = 2048
VALUES = ROWS * COLS
SOURCE_BYTES = VALUES * 2
REQUIRED_GAIN_BPW = -0.5 * math.log2(0.8)
TARGET_RATES = (2.15, 2.5)
ROLE_NAMES = ("gate", "up_or_k0", "down_or_k1")
TENSOR_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_indices(total: int, count: int, seed: int) -> np.ndarray:
    count = min(total, count)
    stride = int(seed % total) | 1
    while math.gcd(stride, total) != 1:
        stride += 2
        if stride >= total:
            stride = 1
    start = int((seed >> 19) % total)
    return ((start + stride * np.arange(count, dtype=np.uint64)) % total).astype(np.int64)


def load_bf16(path: Path, role: str) -> np.ndarray:
    words = np.memmap(path, dtype="<u2", mode="r", shape=(VALUES,))
    values = (np.asarray(words, dtype=np.uint32) << np.uint32(16)).view(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite source: {path}")
    if role in ("gate", "up"):
        return np.asarray(values.reshape(ROWS, COLS), dtype=np.float32, order="C")
    if role == "down":
        return np.asarray(values.reshape(COLS, ROWS).T, dtype=np.float32, order="C")
    raise ValueError(role)


def validate_plan(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    clean = dict(plan)
    expected = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_json(clean)).hexdigest()
    if actual != expected:
        raise ValueError("plan internal lock mismatch")
    if plan.get("coverage", {}).get("weights") != WEIGHTS:
        raise ValueError("wrong panel weight count")
    if len(plan.get("sources", [])) != EXPERTS * ROLES:
        raise ValueError("wrong source count")
    return plan


def parse_header(plan_path: Path, plan: dict[str, Any]) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    asset = plan["assets"]["header.bin"]
    path = plan_path.parent / asset["relpath"]
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if len(payload) != 128 or payload[:8] != b"PLRLOC3\0" or actual != asset["sha256"]:
        raise ValueError("expert-affine header mismatch")
    coefficients = struct.unpack_from("<12f", payload, 32)
    codes = struct.unpack_from("<6h", payload, 80)
    pairs = []
    for expert, code in enumerate(codes):
        theta = code * math.pi / 32768.0
        expected = (np.float32(math.cos(theta)), np.float32(math.sin(theta)))
        actual_pair = (
            np.float32(coefficients[2 * expert]),
            np.float32(coefficients[2 * expert + 1]),
        )
        if expected[0].tobytes() != actual_pair[0].tobytes() or expected[1].tobytes() != actual_pair[1].tobytes():
            raise ValueError("header coefficient/code mismatch")
        pairs.append((float(actual_pair[0]), float(actual_pair[1])))
    return pairs, {"path": str(path.resolve()), "bytes": len(payload), "sha256": actual}


def matrix_stats(matrix: np.ndarray) -> tuple[float, float, float]:
    flat = matrix.reshape(-1).astype(np.float64)
    total = float(np.sum(flat, dtype=np.float64))
    energy = float(np.dot(flat, flat))
    mean = total / flat.size
    centered = energy - flat.size * mean * mean
    if centered <= 0.0:
        raise ValueError("non-positive centered energy")
    return mean, math.sqrt(centered / flat.size), energy


def load_sample_bank(
    plan_path: Path,
    dimensions: list[int],
    vectors_per_matrix: int,
) -> tuple[dict[tuple[str, int, int, int], np.ndarray], dict[str, Any]]:
    plan = validate_plan(plan_path)
    coefficients, header = parse_header(plan_path, plan)
    source_root = Path(plan["source_root"]).resolve(strict=True)
    bank: dict[tuple[str, int, int, int], np.ndarray] = {}
    source_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    for expert in range(EXPERTS):
        triplet = plan["sources"][3 * expert : 3 * expert + 3]
        if [row["role"] for row in triplet] != ["gate", "up", "down"]:
            raise ValueError(f"role order mismatch at expert {expert}")
        matrices = []
        for row in triplet:
            match = TENSOR_RE.fullmatch(row["tensor"])
            if match is None:
                raise ValueError(f"unexpected tensor name: {row['tensor']}")
            path = (source_root / row["source_relpath"]).resolve(strict=True)
            if source_root not in path.parents or path.stat().st_size != SOURCE_BYTES:
                raise ValueError("source escaped root or has wrong size")
            actual_hash = sha256_file(path)
            if actual_hash != row["source_bf16_sha256"]:
                raise ValueError(f"source hash mismatch: {row['tensor']}")
            matrix = load_bf16(path, row["role"])
            matrices.append(matrix)
            mean, rms, energy = matrix_stats(matrix)
            source_rows.append(
                {
                    "matrix_ordinal": int(row["matrix_ordinal"]),
                    "tensor": row["tensor"],
                    "bytes": int(row["bytes"]),
                    "sha256": actual_hash,
                    "mean_fp64": mean,
                    "rms_about_mean_fp64": rms,
                    "energy_fp64": energy,
                }
            )
        gate, up, down_t = matrices
        co, si = coefficients[expert]
        k0 = (np.float32(co) * up + np.float32(si) * down_t).astype(np.float32)
        k1 = (-np.float32(si) * up + np.float32(co) * down_t).astype(np.float32)
        representations = {"raw": (gate, up, down_t), "xklt": (gate, k0, k1)}
        for representation, components in representations.items():
            for role, matrix in enumerate(components):
                mean, rms, energy = matrix_stats(matrix)
                energy_rows.append(
                    {
                        "representation": representation,
                        "expert": expert,
                        "role": role,
                        "mean_fp64": mean,
                        "rms_fp64": rms,
                        "energy_fp64": energy,
                    }
                )
                normalized = ((matrix.astype(np.float64) - mean) / rms).astype(np.float32)
                for dimension in dimensions:
                    vectors = normalized.reshape(-1, dimension)
                    seed = stable_seed("QWEN-ICA-SAMPLE-v1", plan["lock_sha256"], expert, role, dimension)
                    indices = deterministic_indices(vectors.shape[0], vectors_per_matrix, seed)
                    bank[(representation, dimension, expert, role)] = np.asarray(
                        vectors[indices], dtype=np.float32, order="C"
                    )
        del matrices, gate, up, down_t, k0, k1, representations
    provenance = {
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_lock_sha256": plan["lock_sha256"],
        "source_root": str(source_root),
        "header": header,
        "xklt_coefficients_fp32": [list(pair) for pair in coefficients],
        "sources": source_rows,
        "representation_energy": energy_rows,
    }
    return bank, provenance


def canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def symmetric_orthogonalize(matrix: np.ndarray) -> np.ndarray:
    gram = matrix.T @ matrix
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, 1e-14)
    return matrix @ ((vectors * (1.0 / np.sqrt(values))) @ vectors.T)


def klt_rotation(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0, dtype=np.float64)
    covariance = centered.T @ centered / float(centered.shape[0])
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    return canonicalize_columns(vectors[:, order])


def initial_rotation(dimension: int, seed: int, klt: np.ndarray | None) -> np.ndarray:
    if seed == 0:
        return np.eye(dimension, dtype=np.float64) if klt is None else klt.copy()
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    q *= np.sign(np.diag(r))[None, :]
    if klt is not None:
        q = klt @ q
    return q


def fast_orthogonal_projection(
    values: np.ndarray,
    contrast: str,
    seed: int,
    iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float64)
    x = x - np.mean(x, axis=0, dtype=np.float64)
    global_rms = math.sqrt(float(np.mean(x * x, dtype=np.float64)))
    x /= global_rms
    klt = klt_rotation(x)
    w = initial_rotation(x.shape[1], seed, klt)
    converged = False
    last_delta = float("inf")
    completed = 0
    for iteration in range(iterations):
        y = x @ w
        if contrast == "tanh":
            g = np.tanh(y)
            gp = 1.0 - g * g
        elif contrast == "gauss":
            exponential = np.exp(-0.5 * y * y)
            g = y * exponential
            gp = (1.0 - y * y) * exponential
        elif contrast == "cube":
            g = y * y * y
            gp = 3.0 * y * y
        else:
            raise ValueError(contrast)
        update = (x.T @ g) / float(x.shape[0]) - w * np.mean(gp, axis=0)[None, :]
        w_new = symmetric_orthogonalize(update)
        alignment = np.abs(np.diag(w.T @ w_new))
        last_delta = float(np.max(np.abs(1.0 - alignment)))
        w = w_new
        completed = iteration + 1
        if last_delta < tolerance:
            converged = True
            break
    w = canonicalize_columns(w)
    return w, {
        "contrast": contrast,
        "seed": seed,
        "iterations": completed,
        "converged": converged,
        "last_alignment_delta": last_delta,
        "global_rms": global_rms,
        "orthogonality_max_abs": float(np.max(np.abs(w.T @ w - np.eye(w.shape[1])))),
        "rotation_f64_sha256": hashlib.sha256(w.astype("<f8").tobytes()).hexdigest(),
    }


def self_histogram_negentropy(values: np.ndarray, bins: int, edge: float) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    mean = np.mean(x, axis=0, dtype=np.float64)
    sigma = np.std(x, axis=0, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)
    z = (x - mean) / sigma
    width = 2.0 * edge / bins
    indices = np.floor((z + edge) / width).astype(np.int64)
    np.clip(indices, 0, bins - 1, out=indices)
    entropies = []
    skews = []
    kurtoses = []
    for column in range(z.shape[1]):
        counts = np.bincount(indices[:, column], minlength=bins).astype(np.float64)
        probability = counts[counts > 0.0] / float(z.shape[0])
        h = -float(np.sum(probability * np.log2(probability / width)))
        entropies.append(h)
        zc = z[:, column]
        skews.append(float(np.mean(zc ** 3, dtype=np.float64)))
        kurtoses.append(float(np.mean(zc ** 4, dtype=np.float64) - 3.0))
    gaussian_entropy = 0.5 * math.log2(2.0 * math.pi * math.e)
    histogram_j = gaussian_entropy - float(np.mean(entropies))
    edgeworth_j = float(
        np.mean(np.square(skews) / 12.0 + np.square(kurtoses) / 48.0) / math.log(2.0)
    )
    variances = sigma * sigma
    variance_gain = 0.5 * math.log2(float(np.mean(variances)) / math.exp(float(np.mean(np.log(variances)))))
    return {
        "histogram_negentropy_bpw": histogram_j,
        "edgeworth_negentropy_bpw": edgeworth_j,
        "variance_allocation_gain_bpw": variance_gain,
        "max_abs_skew": float(np.max(np.abs(skews))),
        "max_abs_excess_kurtosis": float(np.max(np.abs(kurtoses))),
    }


def fit_histogram_score(
    train: np.ndarray,
    test: np.ndarray,
    bins: int,
    edge: float,
    pseudocount: float,
) -> dict[str, float]:
    train64 = np.asarray(train, dtype=np.float64)
    test64 = np.asarray(test, dtype=np.float64)
    mean = np.mean(train64, axis=0, dtype=np.float64)
    sigma = np.std(train64, axis=0, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)
    z_train = (train64 - mean) / sigma
    z_test = (test64 - mean) / sigma
    width = 2.0 * edge / bins
    train_index = np.floor((z_train + edge) / width).astype(np.int64)
    test_index = np.floor((z_test + edge) / width).astype(np.int64)
    np.clip(train_index, 0, bins - 1, out=train_index)
    np.clip(test_index, 0, bins - 1, out=test_index)
    histogram_log_likelihood = 0.0
    gaussian_log_likelihood = 0.0
    count = test64.size
    for column in range(train64.shape[1]):
        counts = np.bincount(train_index[:, column], minlength=bins).astype(np.float64)
        probability = (counts + pseudocount) / (train64.shape[0] + pseudocount * bins)
        histogram_log_likelihood += float(
            np.sum(np.log2(probability[test_index[:, column]] / width), dtype=np.float64)
        )
        gaussian_log_likelihood += float(
            np.sum((-0.5 * z_test[:, column] ** 2 - 0.5 * math.log(2.0 * math.pi)) / math.log(2.0), dtype=np.float64)
        )
    return {
        "histogram_nll_bpw": -histogram_log_likelihood / count,
        "gaussian_nll_bpw": -gaussian_log_likelihood / count,
        "shape_gain_bpw": (histogram_log_likelihood - gaussian_log_likelihood) / count,
        "train_scalars": int(train64.size),
        "test_scalars": int(test64.size),
    }


def quantizer_curve(
    train: np.ndarray,
    test: np.ndarray,
    deltas: list[float],
    qmax: int,
    pseudocount: float,
) -> list[dict[str, float]]:
    train64 = np.asarray(train, dtype=np.float64)
    test64 = np.asarray(test, dtype=np.float64)
    mean = np.mean(train64, axis=0, dtype=np.float64)
    sigma = np.std(train64, axis=0, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)
    z_train = (train64 - mean) / sigma
    z_test = (test64 - mean) / sigma
    levels = 2 * qmax + 1
    result = []
    for delta in deltas:
        q_train = np.rint(z_train / delta).astype(np.int64)
        q_test = np.rint(z_test / delta).astype(np.int64)
        np.clip(q_train, -qmax, qmax, out=q_train)
        np.clip(q_test, -qmax, qmax, out=q_test)
        bits = 0.0
        for column in range(train64.shape[1]):
            counts = np.bincount(q_train[:, column] + qmax, minlength=levels).astype(np.float64)
            probability = (counts + pseudocount) / (train64.shape[0] + pseudocount * levels)
            bits -= float(np.sum(np.log2(probability[q_test[:, column] + qmax]), dtype=np.float64))
        reconstructed = mean + sigma * (q_test.astype(np.float64) * delta)
        squared_error = float(np.sum((test64 - reconstructed) ** 2, dtype=np.float64))
        result.append(
            {
                "delta": float(delta),
                "rate_bpw": bits / test64.size,
                "mse": squared_error / test64.size,
                "bits": bits,
                "squared_error": squared_error,
                "scalars": int(test64.size),
            }
        )
    return result


def aggregate_curves(curves: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    if not curves:
        raise ValueError("no curves")
    result = []
    for index in range(len(curves[0])):
        rows = [curve[index] for curve in curves]
        scalars = sum(int(row["scalars"]) for row in rows)
        bits = sum(float(row["bits"]) for row in rows)
        error = sum(float(row["squared_error"]) for row in rows)
        result.append(
            {
                "delta": float(rows[0]["delta"]),
                "rate_bpw": bits / scalars,
                "mse": error / scalars,
                "bits": bits,
                "squared_error": error,
                "scalars": scalars,
            }
        )
    return result


def interpolate_mse(curve: list[dict[str, float]], target_rate: float) -> dict[str, float]:
    rows = sorted(curve, key=lambda row: row["rate_bpw"])
    for left, right in zip(rows, rows[1:]):
        if left["rate_bpw"] <= target_rate <= right["rate_bpw"]:
            fraction = (target_rate - left["rate_bpw"]) / (right["rate_bpw"] - left["rate_bpw"])
            log_mse = math.log(left["mse"]) + fraction * (math.log(right["mse"]) - math.log(left["mse"]))
            return {
                "rate_bpw": target_rate,
                "mse": math.exp(log_mse),
                "left_delta": left["delta"],
                "right_delta": right["delta"],
                "fraction": fraction,
            }
    raise ValueError(f"target rate {target_rate} outside curve [{rows[0]['rate_bpw']}, {rows[-1]['rate_bpw']}]")


def select_fit_values(
    bank: dict[tuple[str, int, int, int], np.ndarray],
    representation: str,
    dimension: int,
    experts: list[int],
    fit_vectors_per_matrix: int,
) -> np.ndarray:
    arrays = []
    for expert in experts:
        for role in range(ROLES):
            arrays.append(bank[(representation, dimension, expert, role)][:fit_vectors_per_matrix])
    return np.concatenate(arrays, axis=0)


def all_values(
    bank: dict[tuple[str, int, int, int], np.ndarray],
    representation: str,
    dimension: int,
    experts: list[int],
) -> np.ndarray:
    return np.concatenate(
        [bank[(representation, dimension, expert, role)] for expert in experts for role in range(ROLES)],
        axis=0,
    )


def screen_candidates(
    values_fit: np.ndarray,
    values_eval: np.ndarray,
    gaussian_eval: np.ndarray,
    iterations: int,
    tolerance: float,
    bins: int,
    edge: float,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    dimension = values_fit.shape[1]
    candidates: list[tuple[str, int, np.ndarray, dict[str, Any]]] = []
    identity = np.eye(dimension, dtype=np.float64)
    candidates.append(("identity", 0, identity, {"contrast": "identity", "seed": 0, "iterations": 0}))
    klt = klt_rotation(values_fit)
    candidates.append(("klt", 0, klt, {"contrast": "klt", "seed": 0, "iterations": 0}))
    for contrast, seed in (("tanh", 0), ("tanh", 1), ("gauss", 1), ("cube", 0)):
        rotation, metadata = fast_orthogonal_projection(values_fit, contrast, seed, iterations, tolerance)
        candidates.append((contrast, seed, rotation, metadata))
    control = self_histogram_negentropy(gaussian_eval, bins, edge)
    rows = []
    for contrast, seed, rotation, metadata in candidates:
        transformed = np.asarray(values_eval, dtype=np.float64) @ rotation
        score = self_histogram_negentropy(transformed, bins, edge)
        calibrated_hist = score["histogram_negentropy_bpw"] - control["histogram_negentropy_bpw"]
        optimistic_proxy = calibrated_hist + score["variance_allocation_gain_bpw"]
        rows.append(
            {
                "contrast": contrast,
                "seed": seed,
                "metadata": metadata,
                "score": score,
                "matched_gaussian_score": control,
                "calibrated_histogram_negentropy_bpw": calibrated_hist,
                "optimistic_shape_plus_variance_bpw": optimistic_proxy,
                "rotation_f64_sha256": hashlib.sha256(rotation.astype("<f8").tobytes()).hexdigest(),
                "_rotation": rotation,
            }
        )
    best = max(rows, key=lambda row: row["optimistic_shape_plus_variance_bpw"])
    selected_rotation = best["_rotation"]
    selected_public = {key: value for key, value in best.items() if key != "_rotation"}
    public_rows = [{key: value for key, value in row.items() if key != "_rotation"} for row in rows]
    return selected_rotation, selected_public, public_rows


def side_ledger(dimension: int, histogram_bins: int) -> dict[str, Any]:
    rotation_bytes = dimension * dimension * 4
    mean_scale_bytes = 2 * dimension * 4
    density_table_bytes = dimension * histogram_bins * 2
    quantizer_table_bytes = dimension * 63 * 2
    framing_bytes = 128
    total = rotation_bytes + mean_scale_bytes + density_table_bytes + quantizer_table_bytes + framing_bytes
    rates = {}
    expert_weights = WEIGHTS // EXPERTS
    for rate in TARGET_RATES:
        ideal_expert_bytes = expert_weights * rate / 8.0
        cold_bytes = (10.0 / 9.0) * ideal_expert_bytes + total
        rates[str(rate)] = {
            "ideal_expert_bytes": ideal_expert_bytes,
            "cold_read_bytes": cold_bytes,
            "cold_read_amplification": cold_bytes / ideal_expert_bytes,
        }
    return {
        "assumption": "one common FP32 orthogonal block rotation and shared uint16 density/quantizer tables, all reread cold per expert",
        "rotation_bytes": rotation_bytes,
        "mean_scale_bytes": mean_scale_bytes,
        "density_table_bytes": density_table_bytes,
        "quantizer_table_bytes": quantizer_table_bytes,
        "framing_bytes": framing_bytes,
        "total_common_bytes": total,
        "side_bpw": total * 8.0 / WEIGHTS,
        "rates": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimensions", default="8,16,32,64")
    parser.add_argument("--representations", default="raw,xklt")
    parser.add_argument("--vectors-per-matrix", type=int, default=2048)
    parser.add_argument("--fit-vectors-per-matrix", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--histogram-bins", type=int, default=128)
    parser.add_argument("--histogram-edge", type=float, default=8.0)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--deltas", default="0.40,0.48,0.56,0.64,0.72,0.80,0.90,1.00,1.12,1.26,1.42")
    parser.add_argument("--qmax", type=int, default=31)
    args = parser.parse_args()

    started = time.time()
    dimensions = [int(item) for item in args.dimensions.split(",") if item]
    representations = [item for item in args.representations.split(",") if item]
    deltas = [float(item) for item in args.deltas.split(",") if item]
    if any(COLS % dimension for dimension in dimensions):
        raise ValueError("all dimensions must divide 2048")
    if not set(representations).issubset({"raw", "xklt"}):
        raise ValueError("unknown representation")
    bank, provenance = load_sample_bank(args.plan, dimensions, args.vectors_per_matrix)
    script_path = Path(__file__).resolve()
    results = []
    best_crossfit_gain = -float("inf")
    best_free_gain = -float("inf")
    best_descriptor: dict[str, Any] | None = None

    for representation in representations:
        for dimension in dimensions:
            print(f"screen {representation} d={dimension}", flush=True)
            fit = select_fit_values(bank, representation, dimension, list(range(EXPERTS)), args.fit_vectors_per_matrix)
            evaluation = all_values(bank, representation, dimension, list(range(EXPERTS)))
            rng = np.random.default_rng(stable_seed("QWEN-ICA-GAUSS-SCREEN-v1", representation, dimension))
            gaussian_evaluation = rng.standard_normal(evaluation.shape).astype(np.float64)
            _, selected, candidate_rows = screen_candidates(
                fit,
                evaluation,
                gaussian_evaluation,
                args.iterations,
                args.tolerance,
                args.histogram_bins,
                args.histogram_edge,
            )
            free_gain = float(selected["optimistic_shape_plus_variance_bpw"])
            best_free_gain = max(best_free_gain, free_gain)
            selected_contrast = selected["contrast"]
            selected_seed = int(selected["seed"])
            fold_rows = []
            real_curves = []
            gaussian_curves = []
            weighted_real_hist_gain = 0.0
            weighted_gaussian_hist_gain = 0.0
            total_hist_scalars = 0
            for heldout in range(EXPERTS):
                train_experts = [expert for expert in range(EXPERTS) if expert != heldout]
                train_fit = select_fit_values(
                    bank, representation, dimension, train_experts, args.fit_vectors_per_matrix
                )
                if selected_contrast == "identity":
                    rotation = np.eye(dimension, dtype=np.float64)
                    rotation_meta = {"contrast": "identity", "seed": 0, "iterations": 0}
                elif selected_contrast == "klt":
                    rotation = klt_rotation(train_fit)
                    rotation_meta = {"contrast": "klt", "seed": 0, "iterations": 0}
                else:
                    rotation, rotation_meta = fast_orthogonal_projection(
                        train_fit,
                        selected_contrast,
                        selected_seed,
                        args.iterations,
                        args.tolerance,
                    )
                train = all_values(bank, representation, dimension, train_experts).astype(np.float64) @ rotation
                test = all_values(bank, representation, dimension, [heldout]).astype(np.float64) @ rotation
                seed = stable_seed("QWEN-ICA-GAUSS-FOLD-v1", representation, dimension, heldout)
                rng = np.random.default_rng(seed)
                gaussian_train = rng.standard_normal(train.shape)
                gaussian_test = rng.standard_normal(test.shape)
                real_hist = fit_histogram_score(
                    train, test, args.histogram_bins, args.histogram_edge, args.pseudocount
                )
                gaussian_hist = fit_histogram_score(
                    gaussian_train,
                    gaussian_test,
                    args.histogram_bins,
                    args.histogram_edge,
                    args.pseudocount,
                )
                real_curve = quantizer_curve(train, test, deltas, args.qmax, args.pseudocount)
                gaussian_curve = quantizer_curve(
                    gaussian_train, gaussian_test, deltas, args.qmax, args.pseudocount
                )
                real_curves.append(real_curve)
                gaussian_curves.append(gaussian_curve)
                scalars = int(real_hist["test_scalars"])
                weighted_real_hist_gain += scalars * real_hist["shape_gain_bpw"]
                weighted_gaussian_hist_gain += scalars * gaussian_hist["shape_gain_bpw"]
                total_hist_scalars += scalars
                fold_rows.append(
                    {
                        "heldout_expert": heldout,
                        "selected_contrast": selected_contrast,
                        "selected_seed": selected_seed,
                        "rotation": rotation_meta,
                        "rotation_f64_sha256": hashlib.sha256(rotation.astype("<f8").tobytes()).hexdigest(),
                        "real_histogram": real_hist,
                        "matched_gaussian_histogram": gaussian_hist,
                        "calibrated_histogram_gain_bpw": real_hist["shape_gain_bpw"] - gaussian_hist["shape_gain_bpw"],
                        "real_quantizer_curve": real_curve,
                        "matched_gaussian_quantizer_curve": gaussian_curve,
                    }
                )
            aggregate_real = aggregate_curves(real_curves)
            aggregate_gaussian = aggregate_curves(gaussian_curves)
            rate_results = []
            for target_rate in TARGET_RATES:
                real_point = interpolate_mse(aggregate_real, target_rate)
                gaussian_point = interpolate_mse(aggregate_gaussian, target_rate)
                gain = -0.5 * math.log2(real_point["mse"] / gaussian_point["mse"])
                rate_results.append(
                    {
                        "target_rate_bpw": target_rate,
                        "real": real_point,
                        "matched_gaussian": gaussian_point,
                        "rate_gain_bpw": gain,
                        "distortion_factor": 2.0 ** (-2.0 * gain),
                        "fraction_of_required_gain": gain / REQUIRED_GAIN_BPW,
                    }
                )
                if gain > best_crossfit_gain:
                    best_crossfit_gain = gain
                    best_descriptor = {
                        "representation": representation,
                        "dimension": dimension,
                        "target_rate_bpw": target_rate,
                        "rate_gain_bpw": gain,
                    }
            aggregate_hist_gain = (
                weighted_real_hist_gain - weighted_gaussian_hist_gain
            ) / total_hist_scalars
            result = {
                "representation": representation,
                "dimension": dimension,
                "free_side_full_panel": {
                    "selection_rule": "max calibrated in-panel histogram negentropy plus diagonal-variance allocation gain; all transform/table bits free",
                    "selected": selected,
                    "candidates": candidate_rows,
                },
                "crossfit": {
                    "selection_leakage": "contrast family and seed selected by full-panel screen; each numeric rotation and every marginal probability model refit without held-out expert",
                    "folds": fold_rows,
                    "aggregate_calibrated_histogram_gain_bpw": aggregate_hist_gain,
                    "aggregate_real_quantizer_curve": aggregate_real,
                    "aggregate_matched_gaussian_quantizer_curve": aggregate_gaussian,
                    "rate_matched_results": rate_results,
                },
                "side_and_read_ledger": side_ledger(dimension, args.histogram_bins),
            }
            results.append(result)

    if best_descriptor is None:
        raise AssertionError("no result")
    largest_empirical = max(best_free_gain, best_crossfit_gain)
    numerical_allowance = 0.005
    optimistic_with_allowance = largest_empirical + numerical_allowance
    decision = "promote" if optimistic_with_allowance >= REQUIRED_GAIN_BPW else "kill"
    output: dict[str, Any] = {
        "schema": "qwen-hidden-ica-projection-oracle/v1",
        "hypothesis": "A block-orthogonal projection-pursuit transform exposes independent non-Gaussian components hidden by training-space mixing, enabling class-matched coding.",
        "claim_boundary": "A source-only empirical early-kill, not a universal rate-distortion converse. It covers repeated contiguous block rotations of d=8/16/32/64, the listed contrasts, cross-fitted marginal models, and an entropy-coded uniform scalar-quantizer proxy; it does not cover arbitrary 2048-D nonlinear manifolds or functional equivalence.",
        "required_gain_bpw": REQUIRED_GAIN_BPW,
        "required_distortion_factor": 0.8,
        "configuration": {
            "dimensions": dimensions,
            "representations": representations,
            "vectors_per_matrix": args.vectors_per_matrix,
            "fit_vectors_per_matrix": args.fit_vectors_per_matrix,
            "iterations": args.iterations,
            "tolerance": args.tolerance,
            "histogram_bins": args.histogram_bins,
            "histogram_edge": args.histogram_edge,
            "pseudocount": args.pseudocount,
            "deltas": deltas,
            "qmax": args.qmax,
            "gpu_used": False,
        },
        "provenance": provenance,
        "results": results,
        "summary": {
            "best_free_side_shape_plus_variance_bpw": best_free_gain,
            "best_crossfit_rate_matched_gain_bpw": best_crossfit_gain,
            "best_crossfit_descriptor": best_descriptor,
            "fixed_numerical_allowance_bpw": numerical_allowance,
            "optimistic_gain_with_allowance_bpw": optimistic_with_allowance,
            "shortfall_to_required_bpw": REQUIRED_GAIN_BPW - optimistic_with_allowance,
            "fraction_of_required_gain": optimistic_with_allowance / REQUIRED_GAIN_BPW,
            "decision": decision,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pid": os.getpid(),
            "elapsed_seconds": time.time() - started,
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
        },
    }
    output["result_seal_sha256"] = hashlib.sha256(canonical_json(output)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
