#!/usr/bin/env python3
"""CPU-only procedural union-of-subspaces oracle on the pinned Qwen panel.

This is an intentionally favorable early-kill experiment.  Every sampled block
receives its exact norm for free, every candidate subspace receives continuous
least-squares coefficients for free, and the encoder chooses the best PRG seed
for free.  An iid Gaussian direction with exactly the same block energy is run
through the identical library.  The ratio of the two residuals is therefore a
matched test for Qwen-specific directional structure, not a codec claim.

The script also evaluates an optimistic, rate-accounted construction.  It
charges a fixed seed, an 8-bit block scale, a private 512-bit expert header,
and all coefficient payload at the exact physical rate.  Coefficient error is
then modeled with the ideal Gaussian RD factor 2**(-2*q).  This is more
favorable than an ordinary scalar codec, but it is an engineering screen rather
than an information-theoretic lower bound for non-Gaussian coefficients.

No torch, CuPy, CUDA API, or GPU subprocess is imported or invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
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
WEIGHTS_PER_EXPERT = ROLES * VALUES
SOURCE_BYTES = VALUES * 2
REQUIRED_S_BPW = -0.5 * math.log2(0.8)
TARGET_RATES = (2.15, 2.5)
HEADER_BITS_PER_EXPERT = 512
SCALE_BITS_PER_BLOCK = 8
LIBRARY_SCHEMA = "procedural-union-subspaces-prg-v1"
RESULT_SCHEMA = "qwen-procedural-subspace-oracle-v1"
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
    start = int((seed >> 21) % total)
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
    claimed = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_json(clean)).hexdigest()
    if claimed != actual:
        raise ValueError(f"plan lock mismatch: claimed={claimed}, actual={actual}")
    if plan.get("coverage", {}).get("weights") != WEIGHTS:
        raise ValueError("wrong pinned-panel weight count")
    if len(plan.get("sources", [])) != EXPERTS * ROLES:
        raise ValueError("wrong pinned-panel source count")
    return plan


def parse_xklt(plan_path: Path, plan: dict[str, Any]) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    asset = plan["assets"]["header.bin"]
    path = plan_path.parent / asset["relpath"]
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if len(payload) != 128 or payload[:8] != b"PLRLOC3\0" or actual_hash != asset["sha256"]:
        raise ValueError("expert-affine header mismatch")
    coefficients = struct.unpack_from("<12f", payload, 32)
    codes = struct.unpack_from("<6h", payload, 80)
    pairs: list[tuple[float, float]] = []
    for expert, code in enumerate(codes):
        theta = code * math.pi / 32768.0
        expected = (np.float32(math.cos(theta)), np.float32(math.sin(theta)))
        actual = (
            np.float32(coefficients[2 * expert]),
            np.float32(coefficients[2 * expert + 1]),
        )
        if expected[0].tobytes() != actual[0].tobytes() or expected[1].tobytes() != actual[1].tobytes():
            raise ValueError("XKLT coefficient/code mismatch")
        pairs.append((float(actual[0]), float(actual[1])))
    return pairs, {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": actual_hash,
        "coefficients_fp32": [list(pair) for pair in pairs],
    }


def read_panel(
    plan_path: Path,
) -> tuple[dict[str, list[list[np.ndarray]]], dict[str, Any]]:
    plan = validate_plan(plan_path)
    xklt, header = parse_xklt(plan_path, plan)
    source_root = Path(plan["source_root"]).resolve(strict=True)
    raw: list[list[np.ndarray]] = []
    transformed: list[list[np.ndarray]] = []
    source_rows: list[dict[str, Any]] = []
    for expert in range(EXPERTS):
        triplet = plan["sources"][3 * expert : 3 * expert + 3]
        if [row["role"] for row in triplet] != ["gate", "up", "down"]:
            raise ValueError(f"role order mismatch for expert ordinal {expert}")
        matrices: list[np.ndarray] = []
        for row in triplet:
            if TENSOR_RE.fullmatch(row["tensor"]) is None:
                raise ValueError(f"unexpected tensor: {row['tensor']}")
            path = (source_root / row["source_relpath"]).resolve(strict=True)
            if source_root not in path.parents or path.stat().st_size != SOURCE_BYTES:
                raise ValueError("source escaped root or has wrong length")
            actual_hash = sha256_file(path)
            if actual_hash != row["source_bf16_sha256"]:
                raise ValueError(f"source hash mismatch: {row['tensor']}")
            matrix = load_bf16(path, row["role"])
            matrices.append(matrix)
            source_rows.append(
                {
                    "matrix_ordinal": int(row["matrix_ordinal"]),
                    "expert_ordinal": expert,
                    "role": row["role"],
                    "tensor": row["tensor"],
                    "bytes": int(path.stat().st_size),
                    "sha256": actual_hash,
                }
            )
        gate, up, down_t = matrices
        co, si = xklt[expert]
        k0 = (np.float32(co) * up + np.float32(si) * down_t).astype(np.float32)
        k1 = (-np.float32(si) * up + np.float32(co) * down_t).astype(np.float32)
        raw.append([gate, up, down_t])
        transformed.append([gate, k0, k1])
    return {"raw": raw, "xklt": transformed}, {
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_lock_sha256": plan["lock_sha256"],
        "source_root": str(source_root),
        "header": header,
        "sources": sorted(source_rows, key=lambda row: row["matrix_ordinal"]),
    }


def sample_bank(
    panel: list[list[np.ndarray]],
    representation: str,
    dimension: int,
    count_per_matrix: int,
    plan_lock: str,
) -> dict[str, np.ndarray]:
    directions: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    experts: list[np.ndarray] = []
    roles: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    max_energy_match_error = 0.0
    for expert in range(EXPERTS):
        for role in range(ROLES):
            matrix = panel[expert][role]
            blocks = matrix.reshape(-1, dimension)
            seed = stable_seed("PSUOS-SAMPLE-v1", plan_lock, expert, role, dimension)
            indices = deterministic_indices(blocks.shape[0], count_per_matrix, seed)
            vectors64 = np.asarray(blocks[indices], dtype=np.float64)
            energy = np.einsum("ij,ij->i", vectors64, vectors64, dtype=np.float64)
            if np.any(energy <= 0.0):
                raise ValueError("zero-energy sampled block")
            unit = (vectors64 / np.sqrt(energy)[:, None]).astype(np.float32)

            control_seed = stable_seed(
                "PSUOS-GAUSSIAN-CONTROL-v1", plan_lock, representation, expert, role, dimension
            )
            rng = np.random.Generator(np.random.PCG64DXSM(control_seed))
            gaussian64 = rng.standard_normal(vectors64.shape, dtype=np.float64)
            gaussian64 /= np.linalg.norm(gaussian64, axis=1)[:, None]
            gaussian = gaussian64.astype(np.float32)
            unit_norm = np.einsum("ij,ij->i", unit.astype(np.float64), unit.astype(np.float64))
            gaussian_norm = np.einsum(
                "ij,ij->i", gaussian.astype(np.float64), gaussian.astype(np.float64)
            )
            max_energy_match_error = max(
                max_energy_match_error,
                float(np.max(np.abs(unit_norm - 1.0))),
                float(np.max(np.abs(gaussian_norm - 1.0))),
            )
            n = len(indices)
            directions.append(unit)
            controls.append(gaussian)
            energies.append(energy)
            experts.append(np.full(n, expert, dtype=np.int8))
            roles.append(np.full(n, role, dtype=np.int8))
            ordinals.append(indices.astype(np.int64))
    return {
        "source": np.concatenate(directions, axis=0),
        "gaussian": np.concatenate(controls, axis=0),
        "energy": np.concatenate(energies, axis=0),
        "expert": np.concatenate(experts, axis=0),
        "role": np.concatenate(roles, axis=0),
        "block_ordinal": np.concatenate(ordinals, axis=0),
        "max_unit_norm_error_fp64": np.asarray(max_energy_match_error, dtype=np.float64),
    }


def canonicalize_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64, order="C")
    for column in range(q.shape[1]):
        pivot = int(np.argmax(np.abs(q[:, column])))
        if q[pivot, column] < 0.0:
            q[:, column] *= -1.0
    return q.astype(np.float32)


def hadamard(dimension: int) -> np.ndarray:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    result = np.ones((1, 1), dtype=np.float32)
    while result.shape[0] < dimension:
        result = np.block([[result, result], [result, -result]])
    return result / np.float32(math.sqrt(dimension))


def procedural_basis(
    dimension: int,
    rank: int,
    family: str,
    seed_ordinal: int,
    mode: str,
    hadamard_cache: dict[int, np.ndarray],
) -> np.ndarray:
    seed = stable_seed(
        "PSUOS-LIBRARY-v1", LIBRARY_SCHEMA, dimension, rank, family, seed_ordinal, mode
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    if family == "hadamard":
        base = hadamard_cache.setdefault(dimension, hadamard(dimension))
        rows = rng.permutation(dimension)
        columns = rng.permutation(dimension)[:rank]
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=dimension)
        return np.asarray(base[rows][:, columns] * signs[:, None], dtype=np.float32, order="C")
    if family == "gaussian_qr":
        z = rng.standard_normal((dimension, rank), dtype=np.float64)
    elif family == "rademacher_qr":
        z = rng.choice(np.asarray([-1.0, 1.0]), size=(dimension, rank))
    elif family == "power_half_qr":
        raw = rng.standard_normal((dimension, rank), dtype=np.float64)
        z = np.sign(raw) * np.sqrt(np.abs(raw))
    elif family == "sparse_quarter_qr":
        z = rng.choice(
            np.asarray([-1.0, 0.0, 1.0]),
            size=(dimension, rank),
            p=np.asarray([0.125, 0.75, 0.125]),
        )
        # The deterministic dense dither prevents the rare rank-deficient draw.
        z += 1e-7 * rng.standard_normal((dimension, rank), dtype=np.float64)
    else:
        raise ValueError(f"unknown family: {family}")
    q, _ = np.linalg.qr(z, mode="reduced")
    return canonicalize_q(q)


def aggregate_metric(
    residual: np.ndarray,
    energy: np.ndarray,
    expert_index: np.ndarray,
) -> list[float]:
    weighted = residual.astype(np.float64) * energy
    return [float(np.sum(weighted[expert_index == expert], dtype=np.float64)) for expert in range(EXPERTS)]


def metric_record(
    *,
    representation: str,
    family: str,
    dimension: int,
    coefficient_count: int,
    seeds: int,
    source_sse: list[float],
    gaussian_sse: list[float],
    energy: list[float],
) -> dict[str, Any]:
    total_source = float(sum(source_sse))
    total_gaussian = float(sum(gaussian_sse))
    total_energy = float(sum(energy))
    d_source = total_source / total_energy
    d_gaussian = total_gaussian / total_energy
    ratio = d_source / d_gaussian
    s_bpw = -0.5 * math.log2(ratio)
    mode = "retained_basis" if coefficient_count <= dimension // 2 else "excluded_complement"
    evaluated_rank = min(coefficient_count, dimension - coefficient_count)
    return {
        "candidate_id": (
            f"{representation}:d{dimension}:k{coefficient_count}:"
            f"{family}:K{seeds}"
        ),
        "representation": representation,
        "family": family,
        "dimension": dimension,
        "coefficient_count": coefficient_count,
        "evaluated_mode": mode,
        "evaluated_rank": evaluated_rank,
        "library_seeds": seeds,
        "seed_bits_per_block": int(math.log2(seeds)),
        "source_residual_sse_by_expert": source_sse,
        "gaussian_residual_sse_by_expert": gaussian_sse,
        "sample_energy_by_expert": energy,
        "source_relative_mse": d_source,
        "gaussian_relative_mse": d_gaussian,
        "source_over_matched_gaussian": ratio,
        "matched_percent_below_gaussian": 100.0 * (1.0 - ratio),
        "matched_structural_advantage_bpw": s_bpw,
        "F_ratio_identity": 2.0 ** (-2.0 * s_bpw),
    }


def evaluate_bank(
    bank: dict[str, np.ndarray],
    representation: str,
    dimension: int,
    coefficient_counts: list[int],
    families: list[str],
    max_seeds: int,
    checkpoints: list[int],
) -> list[dict[str, Any]]:
    source = bank["source"]
    gaussian = bank["gaussian"]
    combined = np.concatenate([source, gaussian], axis=0)
    count = source.shape[0]
    energy = bank["energy"].astype(np.float64)
    expert_index = bank["expert"]
    energy_by_expert = [
        float(np.sum(energy[expert_index == expert], dtype=np.float64))
        for expert in range(EXPERTS)
    ]
    records: list[dict[str, Any]] = []
    hcache: dict[int, np.ndarray] = {}
    for family in families:
        for coefficient_count in coefficient_counts:
            if not 0 < coefficient_count < dimension:
                continue
            retained = coefficient_count <= dimension // 2
            rank = coefficient_count if retained else dimension - coefficient_count
            mode = "retained_basis" if retained else "excluded_complement"
            minimum = np.ones(combined.shape[0], dtype=np.float64)
            for seed_ordinal in range(max_seeds):
                basis = procedural_basis(
                    dimension, rank, family, seed_ordinal, mode, hcache
                )
                projection = combined @ basis
                projected_energy = np.einsum(
                    "ij,ij->i",
                    projection.astype(np.float64),
                    projection.astype(np.float64),
                    dtype=np.float64,
                )
                if retained:
                    residual = 1.0 - projected_energy
                else:
                    residual = projected_energy
                residual = np.clip(residual, 0.0, 1.0)
                np.minimum(minimum, residual, out=minimum)
                seeds = seed_ordinal + 1
                if seeds in checkpoints:
                    source_sse = aggregate_metric(minimum[:count], energy, expert_index)
                    gaussian_sse = aggregate_metric(minimum[count:], energy, expert_index)
                    records.append(
                        metric_record(
                            representation=representation,
                            family=family,
                            dimension=dimension,
                            coefficient_count=coefficient_count,
                            seeds=seeds,
                            source_sse=source_sse,
                            gaussian_sse=gaussian_sse,
                            energy=energy_by_expert,
                        )
                    )
    return records


def rate_account(candidate: dict[str, Any], requested_rate: float) -> dict[str, Any]:
    dimension = int(candidate["dimension"])
    coefficient_count = int(candidate["coefficient_count"])
    blocks_per_expert = WEIGHTS_PER_EXPERT // dimension
    if blocks_per_expert * dimension != WEIGHTS_PER_EXPERT:
        raise ValueError("dimension does not tile one expert")
    physical_bytes_per_expert = math.ceil(WEIGHTS_PER_EXPERT * requested_rate / 8.0)
    physical_bits_per_expert = physical_bytes_per_expert * 8
    actual_rate = physical_bits_per_expert / WEIGHTS_PER_EXPERT
    seed_bits = int(candidate["seed_bits_per_block"])
    fixed_block_bits = seed_bits + SCALE_BITS_PER_BLOCK
    coefficient_payload_bits = (
        physical_bits_per_expert
        - HEADER_BITS_PER_EXPERT
        - blocks_per_expert * fixed_block_bits
    )
    coefficient_bits_each = coefficient_payload_bits / (blocks_per_expert * coefficient_count)
    if coefficient_bits_each < 0.0:
        quantization_factor = 1.0
        valid = False
    else:
        quantization_factor = 2.0 ** (-2.0 * coefficient_bits_each)
        valid = True
    source_residual = np.asarray(candidate["source_residual_sse_by_expert"], dtype=np.float64)
    gaussian_residual = np.asarray(candidate["gaussian_residual_sse_by_expert"], dtype=np.float64)
    energy = np.asarray(candidate["sample_energy_by_expert"], dtype=np.float64)
    source_sse = source_residual + (energy - source_residual) * quantization_factor
    gaussian_sse = gaussian_residual + (energy - gaussian_residual) * quantization_factor
    source_d = float(np.sum(source_sse) / np.sum(energy))
    gaussian_d = float(np.sum(gaussian_sse) / np.sum(energy))
    structural_ratio = source_d / gaussian_d
    structural_s = -0.5 * math.log2(structural_ratio)
    gaussian_limit = 2.0 ** (-2.0 * actual_rate)
    f_source = source_d / gaussian_limit
    return {
        "requested_rate_bpw": requested_rate,
        "physical_rate_bpw": actual_rate,
        "physical_bytes_per_expert": physical_bytes_per_expert,
        "physical_bits_per_expert": physical_bits_per_expert,
        "weights_per_expert": WEIGHTS_PER_EXPERT,
        "blocks_per_expert": blocks_per_expert,
        "private_header_bits_per_expert": HEADER_BITS_PER_EXPERT,
        "framing_table_bits_per_expert": HEADER_BITS_PER_EXPERT,
        "shared_table_bits": 0,
        "seed_bits_per_block": seed_bits,
        "scale_bits_per_block": SCALE_BITS_PER_BLOCK,
        "coefficient_payload_bits_per_expert": coefficient_payload_bits,
        "ideal_coefficient_bits_each": coefficient_bits_each,
        "ideal_coefficient_distortion_factor": quantization_factor,
        "valid_payload": valid,
        "optimistic_source_sse_by_expert": source_sse.tolist(),
        "optimistic_gaussian_sse_by_expert": gaussian_sse.tolist(),
        "optimistic_source_relative_mse": source_d,
        "optimistic_matched_gaussian_relative_mse": gaussian_d,
        "source_over_matched_gaussian": structural_ratio,
        "matched_structural_advantage_bpw": structural_s,
        "F_ratio_identity": 2.0 ** (-2.0 * structural_s),
        "gaussian_assumed_limit": gaussian_limit,
        "source_F_equals_D_times_2pow2R": f_source,
        "target_F": 0.8,
        "passes_twenty_percent_below_gaussian_limit": bool(f_source <= 0.8),
        "cold_expert_payload_bytes": physical_bytes_per_expert,
        "cold_expert_bytes_read": physical_bytes_per_expert,
        "cold_expert_read_amplification": 1.0,
        "cold_read_strictly_below_2x": True,
    }


def crossfit_free(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    held_source = 0.0
    held_gaussian = 0.0
    held_energy = 0.0
    for heldout in range(EXPERTS):
        def train_ratio(candidate: dict[str, Any]) -> tuple[float, str]:
            source = sum(
                value for expert, value in enumerate(candidate["source_residual_sse_by_expert"])
                if expert != heldout
            )
            gaussian = sum(
                value for expert, value in enumerate(candidate["gaussian_residual_sse_by_expert"])
                if expert != heldout
            )
            return source / gaussian, candidate["candidate_id"]

        winner = min(candidates, key=train_ratio)
        source = float(winner["source_residual_sse_by_expert"][heldout])
        gaussian = float(winner["gaussian_residual_sse_by_expert"][heldout])
        energy = float(winner["sample_energy_by_expert"][heldout])
        held_source += source
        held_gaussian += gaussian
        held_energy += energy
        selections.append(
            {
                "heldout_expert": heldout,
                "selected_candidate_id": winner["candidate_id"],
                "training_source_over_gaussian": train_ratio(winner)[0],
                "heldout_source_sse": source,
                "heldout_gaussian_sse": gaussian,
                "heldout_energy": energy,
            }
        )
    ratio = held_source / held_gaussian
    s_bpw = -0.5 * math.log2(ratio)
    return {
        "selection_rule": "minimize train-five-expert source/matched-Gaussian residual ratio",
        "selections": selections,
        "pooled_heldout_source_relative_mse": held_source / held_energy,
        "pooled_heldout_gaussian_relative_mse": held_gaussian / held_energy,
        "pooled_heldout_source_over_gaussian": ratio,
        "pooled_heldout_percent_below_gaussian": 100.0 * (1.0 - ratio),
        "pooled_heldout_structural_advantage_bpw": s_bpw,
        "F_ratio_identity": 2.0 ** (-2.0 * s_bpw),
    }


def crossfit_rate(candidates: list[dict[str, Any]], requested_rate: float) -> dict[str, Any]:
    accounted = [(candidate, rate_account(candidate, requested_rate)) for candidate in candidates]
    selections: list[dict[str, Any]] = []
    held_source = 0.0
    held_gaussian = 0.0
    held_energy = 0.0
    actual_rates: list[float] = []
    for heldout in range(EXPERTS):
        def training_distortion(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, str]:
            candidate, rate = item
            source = sum(
                value for expert, value in enumerate(rate["optimistic_source_sse_by_expert"])
                if expert != heldout
            )
            energy = sum(
                value for expert, value in enumerate(candidate["sample_energy_by_expert"])
                if expert != heldout
            )
            return source / energy, candidate["candidate_id"]

        candidate, rate = min(accounted, key=training_distortion)
        source = float(rate["optimistic_source_sse_by_expert"][heldout])
        gaussian = float(rate["optimistic_gaussian_sse_by_expert"][heldout])
        energy = float(candidate["sample_energy_by_expert"][heldout])
        held_source += source
        held_gaussian += gaussian
        held_energy += energy
        actual_rates.append(float(rate["physical_rate_bpw"]))
        selections.append(
            {
                "heldout_expert": heldout,
                "selected_candidate_id": candidate["candidate_id"],
                "training_source_relative_mse": training_distortion((candidate, rate))[0],
                "heldout_source_sse": source,
                "heldout_gaussian_sse": gaussian,
                "heldout_energy": energy,
                "physical_rate_bpw": rate["physical_rate_bpw"],
                "cold_expert_read_amplification": rate["cold_expert_read_amplification"],
            }
        )
    if max(actual_rates) - min(actual_rates) > 1e-15:
        raise ValueError("fold physical-rate mismatch")
    actual_rate = actual_rates[0]
    source_d = held_source / held_energy
    gaussian_d = held_gaussian / held_energy
    ratio = source_d / gaussian_d
    s_bpw = -0.5 * math.log2(ratio)
    gaussian_limit = 2.0 ** (-2.0 * actual_rate)
    f_source = source_d / gaussian_limit
    return {
        "requested_rate_bpw": requested_rate,
        "physical_rate_bpw": actual_rate,
        "selection_rule": "minimize optimistic source distortion on train-five experts",
        "selections": selections,
        "pooled_heldout_source_relative_mse": source_d,
        "pooled_heldout_matched_gaussian_relative_mse": gaussian_d,
        "pooled_heldout_source_over_matched_gaussian": ratio,
        "pooled_heldout_structural_advantage_bpw": s_bpw,
        "F_ratio_identity": 2.0 ** (-2.0 * s_bpw),
        "gaussian_assumed_limit": gaussian_limit,
        "source_F_equals_D_times_2pow2R": f_source,
        "target_F": 0.8,
        "passes_twenty_percent_below_gaussian_limit": bool(f_source <= 0.8),
        "cold_expert_read_amplification": 1.0,
        "cold_read_strictly_below_2x": True,
    }


def parse_csv(text: str, cast: Any) -> list[Any]:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def default_k(dimension: int) -> list[int]:
    return sorted({1, dimension // 16, dimension // 4, dimension // 2, dimension - 8, dimension - 4})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vectors-per-matrix", type=int, default=96)
    parser.add_argument("--dimensions", default="64,128,256")
    parser.add_argument("--representations", default="raw,xklt")
    parser.add_argument(
        "--families",
        default="gaussian_qr,rademacher_qr,power_half_qr,sparse_quarter_qr,hadamard",
    )
    parser.add_argument("--max-seeds", type=int, default=64)
    parser.add_argument("--seed-checkpoints", default="1,16,64")
    args = parser.parse_args()

    dimensions = parse_csv(args.dimensions, int)
    representations = parse_csv(args.representations, str)
    families = parse_csv(args.families, str)
    checkpoints = sorted(set(parse_csv(args.seed_checkpoints, int)))
    if not checkpoints or checkpoints[-1] != args.max_seeds:
        raise ValueError("largest seed checkpoint must equal --max-seeds")
    if any(seed <= 0 or seed & (seed - 1) for seed in checkpoints):
        raise ValueError("seed checkpoints must be positive powers of two")
    if any(d not in (64, 128, 256) for d in dimensions):
        raise ValueError("this frozen protocol permits only d=64/128/256")
    if any(rep not in ("raw", "xklt") for rep in representations):
        raise ValueError("representation must be raw or xklt")
    if args.vectors_per_matrix <= 0:
        raise ValueError("vectors-per-matrix must be positive")

    started = time.time()
    panel, provenance = read_panel(args.plan.resolve(strict=True))
    candidates: list[dict[str, Any]] = []
    bank_receipts: list[dict[str, Any]] = []
    for representation in representations:
        for dimension in dimensions:
            bank = sample_bank(
                panel[representation],
                representation,
                dimension,
                args.vectors_per_matrix,
                provenance["plan_lock_sha256"],
            )
            ks = default_k(dimension)
            print(
                f"evaluate representation={representation} d={dimension} "
                f"vectors={len(bank['energy'])} ks={ks}",
                flush=True,
            )
            rows = evaluate_bank(
                bank,
                representation,
                dimension,
                ks,
                families,
                args.max_seeds,
                checkpoints,
            )
            candidates.extend(rows)
            bank_receipts.append(
                {
                    "representation": representation,
                    "dimension": dimension,
                    "vectors_per_matrix": args.vectors_per_matrix,
                    "total_vectors": int(len(bank["energy"])),
                    "sample_energy_fp64": float(np.sum(bank["energy"], dtype=np.float64)),
                    "max_unit_norm_error_fp64": float(bank["max_unit_norm_error_fp64"]),
                    "sample_ordinal_sha256": hashlib.sha256(
                        np.asarray(bank["block_ordinal"], dtype="<u8").tobytes()
                    ).hexdigest(),
                }
            )

    if not candidates:
        raise ValueError("no candidates evaluated")
    panel_best_free = max(
        candidates,
        key=lambda row: (row["matched_structural_advantage_bpw"], row["candidate_id"]),
    )
    free_crossfit = crossfit_free(candidates)
    rate_results: list[dict[str, Any]] = []
    for requested_rate in TARGET_RATES:
        accounted = [(candidate, rate_account(candidate, requested_rate)) for candidate in candidates]
        best_candidate, best_rate = min(
            accounted,
            key=lambda item: (
                item[1]["source_F_equals_D_times_2pow2R"],
                item[0]["candidate_id"],
            ),
        )
        rate_results.append(
            {
                "requested_rate_bpw": requested_rate,
                "panel_leaky_best_candidate_id": best_candidate["candidate_id"],
                "panel_leaky_optimistic_accounting": best_rate,
                "leave_one_expert_out": crossfit_rate(candidates, requested_rate),
            }
        )

    max_free_s = float(panel_best_free["matched_structural_advantage_bpw"])
    crossfit_s = float(free_crossfit["pooled_heldout_structural_advantage_bpw"])
    early_kill = bool(max_free_s < REQUIRED_S_BPW / 4.0 and crossfit_s < REQUIRED_S_BPW / 4.0)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "EARLY_KILL" if early_kill else "SURVIVES_FAVORABLE_SCREEN",
        "claim_boundary": (
            "Sampled directional-structure screen only; free projection coefficients and the "
            "modeled ideal-Gaussian coefficient RD are not measured codec MSE. The rate model "
            "is not a universal RD lower bound for non-Gaussian coefficients."
        ),
        "gpu_policy": {
            "cpu_only": True,
            "imports_torch": False,
            "imports_cupy": False,
            "invokes_cuda": False,
        },
        "protocol": {
            "library_schema": LIBRARY_SCHEMA,
            "dimensions": dimensions,
            "representations": representations,
            "families": families,
            "coefficient_counts_by_dimension": {str(d): default_k(d) for d in dimensions},
            "max_library_seeds": args.max_seeds,
            "seed_checkpoints": checkpoints,
            "vectors_per_matrix": args.vectors_per_matrix,
            "sampling": "deterministic coprime stride, balanced across all 18 matrices",
            "control": "iid Gaussian direction, exact matched source-block energy",
            "free_oracle_grants": [
                "exact per-block norm",
                "continuous least-squares coefficients",
                "best seed among K with no seed bits",
                "procedural library with no table bits",
            ],
            "crossfit": "six leave-one-expert-out folds; no heldout expert selects hyperparameters",
            "rate_accounting": {
                "private_header_bits_per_expert": HEADER_BITS_PER_EXPERT,
                "scale_bits_per_block": SCALE_BITS_PER_BLOCK,
                "shared_table_bits": 0,
                "coefficient_model": (
                    "ideal Gaussian RD factor 2**(-2*q); engineering comparator, not a universal "
                    "lower bound for non-Gaussian coefficients"
                ),
                "physical_byte_rounding": "ceil each expert stream to whole bytes",
            },
        },
        "target": {
            "required_percent_below_gaussian": 20.0,
            "required_F": 0.8,
            "required_structural_advantage_bpw": REQUIRED_S_BPW,
            "identity_s_equals_minus_half_log2_F": -0.5 * math.log2(0.8),
            "rates_bpw": list(TARGET_RATES),
        },
        "provenance": {
            **provenance,
            "algorithm_path": str(Path(__file__).resolve()),
            "algorithm_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            },
        },
        "sample_banks": bank_receipts,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "favorable_free_oracle": {
            "panel_leaky_best": panel_best_free,
            "leave_one_expert_out": free_crossfit,
            "required_advantage_bpw": REQUIRED_S_BPW,
            "panel_leaky_shortfall_multiple": (
                REQUIRED_S_BPW / max(max_free_s, 1e-300) if max_free_s > 0.0 else None
            ),
        },
        "optimistic_rate_accounted": rate_results,
        "decision": {
            "early_kill": early_kill,
            "criterion": (
                "both panel-leaky and LOO favorable free-oracle s must be below one quarter "
                "of required s=0.160964047443681"
            ),
            "quarter_threshold_bpw": REQUIRED_S_BPW / 4.0,
            "panel_leaky_free_s_bpw": max_free_s,
            "loo_free_s_bpw": crossfit_s,
            "reason": (
                "The deliberately free directional oracle is far short, so the tested procedural "
                "subspace-selection mechanism is stopped. This does not exclude a separate "
                "non-Gaussian coefficient code."
                if early_kill
                else "The favorable structural screen is large enough to justify confirmation."
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    result["result_lock_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": len(candidates),
                "panel_leaky_free_s_bpw": max_free_s,
                "loo_free_s_bpw": crossfit_s,
                "output": str(args.output),
                "result_lock_sha256": result["result_lock_sha256"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
