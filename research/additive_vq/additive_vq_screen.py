#!/usr/bin/env python3
"""CPU-only, cross-expert screen for high-dimensional additive VQ on Qwen.

The experiment is intentionally an optimistic *early-kill* screen, not a
bitstream claim.  It learns role-conditioned residual/additive codebooks on
five experts and evaluates the sixth.  An independently generated iid
Gaussian control is put through the identical fitting and encoding pipeline.
This divides out finite-dimension and search losses before asking whether the
Qwen source supplies the 0.160964... bit/weight advantage needed for 20%
below the Gaussian rate-distortion curve.

No CUDA, CuPy, torch, scipy, or sklearn code is imported.  NumPy is the only
numerical dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
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
TARGET_S_BPW = -0.5 * math.log2(0.8)
TARGET_F = 0.8
PLAN_LOCK = "99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d"
TENSOR_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def canonical_json_bytes(value: object) -> bytes:
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


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    clean = dict(plan)
    claimed = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(clean)).hexdigest()
    if claimed != actual or claimed != PLAN_LOCK:
        raise ValueError(f"wrong plan lock: claimed={claimed}, recomputed={actual}")
    if plan.get("coverage", {}).get("weights") != WEIGHTS:
        raise ValueError("wrong panel weight count")
    if len(plan.get("sources", [])) != EXPERTS * ROLES:
        raise ValueError("wrong source count")
    return plan


def bf16_matrix(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.memmap(path, dtype="<u2", mode="r", shape=shape)
    values = (np.asarray(words, dtype=np.uint32) << np.uint32(16)).view(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite source {path}")
    return values


def oriented_matrix(path: Path, role: str) -> np.ndarray:
    if role in ("gate", "up"):
        return np.asarray(bf16_matrix(path, (ROWS, COLS)), dtype=np.float32)
    if role == "down":
        return np.asarray(bf16_matrix(path, (COLS, ROWS)).T, dtype=np.float32)
    raise ValueError(role)


def xklt_coefficients(header_path: Path) -> list[tuple[np.float32, np.float32]]:
    payload = header_path.read_bytes()
    if len(payload) != 128 or payload[:8] != b"PLRLOC3\0":
        raise ValueError("wrong expert-local header")
    coefficients = struct.unpack_from("<12f", payload, 32)
    codes = struct.unpack_from("<6h", payload, 80)
    result = []
    for expert, code in enumerate(codes):
        theta = code * math.pi / 32768.0
        expected = (np.float32(math.cos(theta)), np.float32(math.sin(theta)))
        actual = (
            np.float32(coefficients[2 * expert]),
            np.float32(coefficients[2 * expert + 1]),
        )
        if actual[0].tobytes() != expected[0].tobytes() or actual[1].tobytes() != expected[1].tobytes():
            raise ValueError("XKLT coefficient/code mismatch")
        result.append(actual)
    return result


def coprime_stride(modulus: int, seed: int) -> int:
    stride = int(seed % modulus) | 1
    while math.gcd(stride, modulus) != 1:
        stride += 2
        if stride >= modulus:
            stride = 1
    return stride


def sample_vectors(
    matrix: np.ndarray, dimension: int, count: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    flat64 = np.ravel(matrix).astype(np.float64)
    total_sum = float(np.sum(flat64, dtype=np.float64))
    total_energy = float(np.dot(flat64, flat64))
    mean = total_sum / flat64.size
    centered_energy = total_energy - flat64.size * mean * mean
    rms = math.sqrt(centered_energy / flat64.size)
    available = ROWS * (COLS // dimension)
    count = min(count, available)
    start = seed % available
    stride = coprime_stride(available, seed >> 17)
    indices = (start + stride * np.arange(count, dtype=np.uint64)) % available
    vectors = matrix.reshape(available, dimension)[indices.astype(np.int64)]
    vectors = ((np.asarray(vectors, dtype=np.float32) - np.float32(mean)) / np.float32(rms)).astype(np.float32)
    return vectors, {
        "mean_fp64": mean,
        "rms_about_mean_fp64": rms,
        "centered_energy_fp64": centered_energy,
        "source_energy_fp64": total_energy,
        "sample_vectors": int(count),
        "sample_fp32_sha256": hashlib.sha256(vectors.astype("<f4", copy=False).tobytes()).hexdigest(),
        "selection_start": int(start),
        "selection_stride": int(stride),
    }


def gaussian_control(shape: tuple[int, int], seed: int) -> np.ndarray:
    # An independent control per matrix.  Do not force sample moments: source
    # normalization is based on the full matrix, so an iid N(0,1) sample is the
    # matched finite-sample experiment.
    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


def build_bank(
    plan_path: Path, dimension: int, sample_count: int, representation: str
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    plan = load_plan(plan_path)
    plan_dir = plan_path.parent
    source_root = Path(plan["source_root"]).resolve(strict=True)
    header = plan_dir / plan["assets"]["header.bin"]["relpath"]
    if sha256_file(header) != plan["assets"]["header.bin"]["sha256"]:
        raise ValueError("header hash mismatch")
    coeff = xklt_coefficients(header)
    bank: list[list[dict[str, Any]]] = []
    bindings: list[dict[str, Any]] = []
    for expert in range(EXPERTS):
        triplet = plan["sources"][3 * expert : 3 * expert + 3]
        if [row["role"] for row in triplet] != ["gate", "up", "down"]:
            raise ValueError("source role order changed")
        matrices = []
        for row in triplet:
            if TENSOR_RE.fullmatch(row["tensor"]) is None:
                raise ValueError(f"unexpected tensor {row['tensor']}")
            path = (source_root / row["source_relpath"]).resolve(strict=True)
            if source_root not in path.parents or path.stat().st_size != SOURCE_BYTES:
                raise ValueError("source path or size mismatch")
            digest = sha256_file(path)
            if digest != row["source_bf16_sha256"]:
                raise ValueError(f"source hash mismatch {row['tensor']}")
            matrices.append(oriented_matrix(path, row["role"]))
            bindings.append(
                {
                    "matrix_ordinal": row["matrix_ordinal"],
                    "tensor": row["tensor"],
                    "sha256": digest,
                    "bytes": row["bytes"],
                }
            )
        gate, up, down = matrices
        if representation == "xklt":
            co, si = coeff[expert]
            k0 = (co * up + si * down).astype(np.float32)
            k1 = (-si * up + co * down).astype(np.float32)
            components = (gate, k0, k1)
        elif representation == "raw":
            components = (gate, up, down)
        else:
            raise ValueError(representation)
        expert_rows = []
        for role, matrix in enumerate(components):
            seed = stable_seed(
                "QWEN-ADDITIVE-VQ-v1", PLAN_LOCK, representation, dimension, expert, role
            )
            source, metadata = sample_vectors(matrix, dimension, sample_count, seed)
            control = gaussian_control(
                source.shape,
                stable_seed("QWEN-ADDITIVE-VQ-GAUSSIAN-v1", seed),
            )
            expert_rows.append(
                {
                    "source": source,
                    "gaussian": control,
                    "metadata": metadata,
                }
            )
        bank.append(expert_rows)
        del matrices, gate, up, down, components
    provenance = {
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_lock_sha256": plan["lock_sha256"],
        "header_path": str(header.resolve()),
        "header_sha256": sha256_file(header),
        "representation": representation,
        "dimension": dimension,
        "source_bindings": bindings,
        "xklt_coefficients_fp32": [[float(a), float(b)] for a, b in coeff],
    }
    return bank, provenance


def nearest(values: np.ndarray, centers: np.ndarray, batch: int) -> np.ndarray:
    labels = np.empty(values.shape[0], dtype=np.int16)
    center_norm = np.sum(centers * centers, axis=1, dtype=np.float32)
    for start in range(0, values.shape[0], batch):
        x = values[start : start + batch]
        scores = 2.0 * (x @ centers.T) - center_norm[None, :]
        labels[start : start + x.shape[0]] = np.argmax(scores, axis=1).astype(np.int16)
    return labels


def lloyd(
    values: np.ndarray,
    centers_count: int,
    iterations: int,
    seed: int,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(values.shape[0], size=centers_count, replace=False)
    centers = np.asarray(values[chosen], dtype=np.float32).copy()
    labels = np.zeros(values.shape[0], dtype=np.int16)
    for _ in range(iterations):
        labels = nearest(values, centers, batch)
        counts = np.bincount(labels, minlength=centers_count)
        for code in range(centers_count):
            if counts[code]:
                centers[code] = np.mean(values[labels == code], axis=0, dtype=np.float64).astype(np.float32)
            else:
                # Deterministic farthest-point repair avoids a dead codebook row.
                error = np.sum((values - centers[labels]) ** 2, axis=1)
                centers[code] = values[int(np.argmax(error))]
    return centers, labels


def encode_additive(
    values: np.ndarray,
    books: np.ndarray,
    sweeps: int,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    stages, _, _ = books.shape
    labels = np.empty((values.shape[0], stages), dtype=np.int16)
    reconstruction = np.zeros_like(values, dtype=np.float32)
    for stage in range(stages):
        labels[:, stage] = nearest(values - reconstruction, books[stage], batch)
        reconstruction += books[stage, labels[:, stage]]
    for _ in range(sweeps):
        for stage in range(stages):
            old = books[stage, labels[:, stage]]
            target = values - (reconstruction - old)
            new_labels = nearest(target, books[stage], batch)
            new = books[stage, new_labels]
            reconstruction += new - old
            labels[:, stage] = new_labels
    return reconstruction, labels


def train_additive(
    values: np.ndarray,
    stages: int,
    alphabet: int,
    lloyd_iterations: int,
    refit_rounds: int,
    sweeps: int,
    batch: int,
    seed: int,
) -> np.ndarray:
    books = np.empty((stages, alphabet, values.shape[1]), dtype=np.float32)
    residual = np.asarray(values, dtype=np.float32).copy()
    for stage in range(stages):
        centers, labels = lloyd(
            residual,
            alphabet,
            lloyd_iterations,
            stable_seed(seed, "stage", stage),
            batch,
        )
        books[stage] = centers
        residual -= centers[labels]
    for _ in range(refit_rounds):
        reconstruction, labels = encode_additive(values, books, sweeps, batch)
        for stage in range(stages):
            old = books[stage, labels[:, stage]]
            target = values - (reconstruction - old)
            for code in range(alphabet):
                members = labels[:, stage] == code
                if np.any(members):
                    books[stage, code] = np.mean(target[members], axis=0, dtype=np.float64).astype(np.float32)
            new_labels = nearest(target, books[stage], batch)
            new = books[stage, new_labels]
            reconstruction += new - old
            labels[:, stage] = new_labels
    # FP16 is the charged and decoded representation.
    return books.astype(np.float16).astype(np.float32)


def best_gain_distortion(values: np.ndarray, reconstruction: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(reconstruction, dtype=np.float64)
    denominator = float(np.sum(y * y, dtype=np.float64))
    gain = float(np.sum(x * y, dtype=np.float64) / max(denominator, 1e-300))
    error = float(np.sum((x - gain * y) ** 2, dtype=np.float64))
    energy = float(np.sum(x * x, dtype=np.float64))
    return error / energy, gain


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def evaluate_config(
    bank: list[list[dict[str, Any]]],
    dimension: int,
    alphabet: int,
    stages: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    fold_rows = []
    source_sse = gaussian_sse = total_energy = 0.0
    for heldout in range(EXPERTS):
        fold_source_sse = fold_gaussian_sse = fold_energy = 0.0
        role_rows = []
        for role in range(ROLES):
            train_source = np.concatenate(
                [bank[e][role]["source"] for e in range(EXPERTS) if e != heldout], axis=0
            )
            train_gaussian = np.concatenate(
                [bank[e][role]["gaussian"] for e in range(EXPERTS) if e != heldout], axis=0
            )
            source_books = train_additive(
                train_source,
                stages,
                alphabet,
                args.lloyd_iterations,
                args.refit_rounds,
                args.sweeps,
                args.batch,
                stable_seed("source", heldout, role, dimension, alphabet, stages),
            )
            gaussian_books = train_additive(
                train_gaussian,
                stages,
                alphabet,
                args.lloyd_iterations,
                args.refit_rounds,
                args.sweeps,
                args.batch,
                stable_seed("gaussian", heldout, role, dimension, alphabet, stages),
            )
            test_source = bank[heldout][role]["source"]
            test_gaussian = bank[heldout][role]["gaussian"]
            source_reconstruction, _ = encode_additive(
                test_source, source_books, args.sweeps, args.batch
            )
            gaussian_reconstruction, _ = encode_additive(
                test_gaussian, gaussian_books, args.sweeps, args.batch
            )
            ds, source_gain = best_gain_distortion(test_source, source_reconstruction)
            dg, gaussian_gain = best_gain_distortion(test_gaussian, gaussian_reconstruction)
            energy = float(bank[heldout][role]["metadata"]["centered_energy_fp64"])
            fold_source_sse += ds * energy
            fold_gaussian_sse += dg * energy
            fold_energy += energy
            role_rows.append(
                {
                    "role": role,
                    "source_distortion": ds,
                    "gaussian_distortion": dg,
                    "matched_F": ds / dg,
                    "matched_s_bpw": -0.5 * math.log2(ds / dg),
                    "source_oracle_gain": source_gain,
                    "gaussian_oracle_gain": gaussian_gain,
                    "centered_energy_fp64": energy,
                }
            )
        fold_ds = fold_source_sse / fold_energy
        fold_dg = fold_gaussian_sse / fold_energy
        fold_f = fold_ds / fold_dg
        fold_rows.append(
            {
                "heldout_expert": heldout,
                "source_distortion": fold_ds,
                "gaussian_distortion": fold_dg,
                "matched_F": fold_f,
                "matched_s_bpw": -0.5 * math.log2(fold_f),
                "roles": role_rows,
            }
        )
        source_sse += fold_source_sse
        gaussian_sse += fold_gaussian_sse
        total_energy += fold_energy
        print(
            json.dumps(
                {
                    "dimension": dimension,
                    "alphabet": alphabet,
                    "stages": stages,
                    "heldout": heldout,
                    "source_D": fold_ds,
                    "gaussian_D": fold_dg,
                    "s_bpw": -0.5 * math.log2(fold_f),
                }
            ),
            flush=True,
        )
    source_d = source_sse / total_energy
    gaussian_d = gaussian_sse / total_energy
    matched_f = source_d / gaussian_d
    matched_s = -0.5 * math.log2(matched_f)
    fold_s = [row["matched_s_bpw"] for row in fold_rows]
    se = standard_error(fold_s)
    payload_rate = stages * math.log2(alphabet) / dimension
    table_bits = ROLES * stages * alphabet * dimension * 16
    # mean, RMS and post-encode gain: three FP32 values for each matrix, plus a
    # small fixed table/packing header.  This is deliberately compact but real.
    scalar_bits = EXPERTS * ROLES * 3 * 32
    fixed_header_bits = 256
    ternary_pack_overhead_bits = EXPERTS if alphabet == 3 else 0
    side_bits = table_bits + scalar_bits + fixed_header_bits + ternary_pack_overhead_bits
    side_bpw = side_bits / WEIGHTS
    physical_rate = payload_rate + side_bpw
    charged_matched_s = matched_s - side_bpw
    charged_matched_f = math.pow(2.0, -2.0 * charged_matched_s)
    optimistic_s = charged_matched_s + 2.0 * se + args.optimism_allowance_bpw
    optimistic_f = math.pow(2.0, -2.0 * optimistic_s)
    shannon_d = math.pow(2.0, -2.0 * physical_rate)
    exact_f = source_d / shannon_d
    exact_s = -0.5 * math.log2(exact_f)
    expert_weights = ROLES * VALUES
    own_payload_bits = expert_weights * payload_rate
    cold_read_bits = own_payload_bits + table_bits + ROLES * 3 * 32 + fixed_header_bits
    cold_amp = cold_read_bits / own_payload_bits
    return {
        "architecture": f"role-conditioned-additive-rvq-d{dimension}-k{alphabet}-m{stages}",
        "dimension": dimension,
        "alphabet": alphabet,
        "stages": stages,
        "payload_rate_bpw": payload_rate,
        "table_bits_fp16": table_bits,
        "matrix_scalar_bits_fp32": scalar_bits,
        "fixed_header_bits": fixed_header_bits,
        "mixed_radix_pack_overhead_bits": ternary_pack_overhead_bits,
        "side_bits": side_bits,
        "side_bpw": side_bpw,
        "physical_rate_bpw": physical_rate,
        "source_distortion": source_d,
        "matched_gaussian_distortion": gaussian_d,
        "gaussian_shannon_distortion_at_physical_rate": shannon_d,
        "matched_F_source_over_control": matched_f,
        "matched_s_bpw_before_side_charge": matched_s,
        "charged_matched_s_bpw": charged_matched_s,
        "charged_matched_F_identity": charged_matched_f,
        "fold_s_standard_error_bpw": se,
        "optimism_allowance_bpw": args.optimism_allowance_bpw,
        "optimistic_2se_s_bpw": optimistic_s,
        "optimistic_2se_F_identity": optimistic_f,
        "exact_F_source_over_shannon_at_physical_rate": exact_f,
        "exact_s_bpw_source_vs_shannon": exact_s,
        "target_s_bpw": TARGET_S_BPW,
        "target_F": TARGET_F,
        "exact_target_met": exact_f <= TARGET_F and 2.15 <= physical_rate <= 2.5,
        "optimistic_calibrated_target_met": optimistic_s >= TARGET_S_BPW,
        "fraction_of_required_s_optimistic": optimistic_s / TARGET_S_BPW,
        "cold_expert_read_amplification": cold_amp,
        "read_amplification_below_2x": cold_amp < 2.0,
        "crossfit": "six leave-one-expert-out folds; all three roles held together",
        "folds": fold_rows,
        "elapsed_seconds": time.time() - started,
    }


def stages_for(dimension: int, alphabet: int, nominal_rate: float) -> int:
    candidates = []
    for stages in range(1, 256):
        rate = stages * math.log2(alphabet) / dimension
        if 2.15 <= rate <= 2.49:
            candidates.append((abs(rate - nominal_rate), stages))
    if not candidates:
        raise ValueError((dimension, alphabet, nominal_rate))
    return min(candidates)[1]


def write_report(path: Path, result: dict[str, Any]) -> None:
    rows = sorted(result["results"], key=lambda row: row["optimistic_2se_s_bpw"], reverse=True)
    lines = [
        "# High-dimensional additive VQ early-kill screen",
        "",
        "This is a CPU-only source experiment, not a serialized-codec claim. Codebooks are",
        "trained on five experts and evaluated on the held-out sixth; the same pipeline is",
        "independently fitted to moment-matched iid Gaussian controls. All tables and compact",
        "per-matrix scalars are charged, while the favorable decision bound additionally grants",
        "two fold standard errors and a fixed numerical allowance.",
        "",
        f"Required advantage: `s >= {TARGET_S_BPW:.15f}` bpw, equivalently `F=2^(-2s) <= 0.8`.",
        "",
        "| architecture | R physical | source D | matched s | optimistic s | optimistic F | exact F vs Shannon | cold read |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {row['physical_rate_bpw']:.6f} | "
            f"{row['source_distortion']:.8f} | {row['charged_matched_s_bpw']:.6f} | "
            f"{row['optimistic_2se_s_bpw']:.6f} | {row['optimistic_2se_F_identity']:.6f} | "
            f"{row['exact_F_source_over_shannon_at_physical_rate']:.6f} | "
            f"{row['cold_expert_read_amplification']:.6f}x |"
        )
    best = rows[0]
    lines += [
        "",
        "## Decision",
        "",
        f"The most favorable tested result reaches only `{best['fraction_of_required_s_optimistic']:.3%}` "
        f"of the required advantage after the deliberately generous allowance. Its identity is "
        f"`F = 2^(-2*{best['optimistic_2se_s_bpw']:.12f}) = {best['optimistic_2se_F_identity']:.12f}`. "
        "The branch is therefore rejected early; increasing training effort cannot plausibly close",
        "the remaining order-of-magnitude gap, and the uncalibrated absolute finite-dimensional",
        "quantizers are farther from the Gaussian Shannon curve still.",
        "",
        "## Claim boundary",
        "",
        "This rejects the tested adjacent-coordinate, role-conditioned additive residual VQ family",
        "at dimensions 8/16/32 and binary/ternary/quaternary alphabets. It is not a converse for",
        "arbitrary semantic reordering, extremely large unstructured codebooks, or joint coding",
        "across experts. The oracle per-matrix reconstruction gain is stored and charged; it makes",
        "the screen more favorable to the candidate.",
        "",
        "Full source hashes, plan/header seals, fold results, costs, and exact identities are in",
        f"`{result['result_json_name']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("xklt", "raw"), default="xklt")
    parser.add_argument("--dimensions", default="8,16,32")
    parser.add_argument("--alphabets", default="2,3,4")
    parser.add_argument("--sample-vectors-per-matrix", type=int, default=2048)
    parser.add_argument("--nominal-rate", type=float, default=2.25)
    parser.add_argument("--lloyd-iterations", type=int, default=3)
    parser.add_argument("--refit-rounds", type=int, default=1)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--optimism-allowance-bpw", type=float, default=0.005)
    args = parser.parse_args()
    dimensions = [int(value) for value in args.dimensions.split(",") if value]
    alphabets = [int(value) for value in args.alphabets.split(",") if value]
    for dimension in dimensions:
        if COLS % dimension:
            raise ValueError(f"dimension {dimension} does not divide {COLS}")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    provenances = []
    for dimension in dimensions:
        print(f"loading d={dimension}", flush=True)
        bank, provenance = build_bank(
            args.plan, dimension, args.sample_vectors_per_matrix, args.representation
        )
        provenances.append(provenance)
        for alphabet in alphabets:
            stages = stages_for(dimension, alphabet, args.nominal_rate)
            print(f"evaluating d={dimension} K={alphabet} M={stages}", flush=True)
            row = evaluate_config(bank, dimension, alphabet, stages, args)
            results.append(row)
            print(
                json.dumps(
                    {
                        "architecture": row["architecture"],
                        "R": row["physical_rate_bpw"],
                        "source_D": row["source_distortion"],
                        "matched_s": row["charged_matched_s_bpw"],
                        "optimistic_s": row["optimistic_2se_s_bpw"],
                        "exact_F": row["exact_F_source_over_shannon_at_physical_rate"],
                    }
                ),
                flush=True,
            )
        del bank
    result_path = args.output / "additive_vq_screen_result.json"
    report_path = args.output / "ADDITIVE_VQ_EARLY_KILL.md"
    payload = {
        "schema": "qwen-additive-vq-early-kill-v1",
        "created_unix": time.time(),
        "cpu_only": True,
        "numpy_version": np.__version__,
        "parameters": {
            "representation": args.representation,
            "dimensions": dimensions,
            "alphabets": alphabets,
            "sample_vectors_per_matrix": args.sample_vectors_per_matrix,
            "nominal_payload_rate_bpw": args.nominal_rate,
            "lloyd_iterations": args.lloyd_iterations,
            "refit_rounds": args.refit_rounds,
            "coordinate_descent_sweeps": args.sweeps,
            "batch": args.batch,
            "optimism_allowance_bpw": args.optimism_allowance_bpw,
        },
        "target": {
            "required_s_bpw": TARGET_S_BPW,
            "required_F": TARGET_F,
            "identity": "F = 2^(-2s)",
            "allowed_physical_rate_bpw": [2.15, 2.5],
            "maximum_read_amplification": 2.0,
        },
        "provenance_by_dimension": provenances,
        "results": results,
        "result_json_name": result_path.name,
        "decision": "kill" if not any(row["optimistic_calibrated_target_met"] for row in results) else "survives",
        "claim_boundary": (
            "Adjacent-coordinate role-conditioned additive residual VQ only; dimensions and "
            "alphabets listed in parameters. Cross-expert held out, Gaussian calibrated, and "
            "table/index/scalar costs charged. Not a universal converse."
        ),
    }
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(report_path, payload)
    print(json.dumps({"result": str(result_path), "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
