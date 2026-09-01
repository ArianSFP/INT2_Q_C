#!/usr/bin/env python3
"""CuPy stage-0 genie for a decoded rank-3 local SVD tangent.

The source is used only to score the exact orthogonal projection.  The tangent
itself is a deterministic function of the already decoded coarse block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np


EXPERTS = 6
ROLES = 3
ROWS = 768
COLS = 2048
GROUP = 2048
GROUPS_PER_EXPERT = 2304
BLOCK_VALUES = 4096
BLOCK_SIDE = 64
BLOCKS_PER_MATRIX = ROWS * COLS // BLOCK_VALUES
TANGENT_RANK = 3
TANGENT_DIMENSION = TANGENT_RANK * (2 * BLOCK_SIDE - TANGENT_RANK)
COSET_BITS_PER_BLOCK = 384
PANEL_VALUES = EXPERTS * ROLES * ROWS * COLS
EXPECTED_BASE_SSE = 500.39553685426534
EXPECTED_BASE_ENERGY = 16192.89450885593
BASE_F = 0.9888693569009007
COARSE_BPW = 2.3984375
COSET_BPW = 0.09375
METADATA_BPW = 0.0078125
TOTAL_BPW = 2.5
REQUIRED_CAPTURE = 0.2972443434920543
READ_AMPLIFICATION = 73.0 / 72.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def check_plan_seal(plan: dict[str, Any]) -> None:
    claimed = plan.get("lock_sha256")
    clean = dict(plan)
    clean.pop("lock_sha256", None)
    observed = hashlib.sha256(canonical(clean)).hexdigest()
    if observed != claimed:
        raise ValueError("plan internal seal mismatch")


def bf16_matrix(path: Path, shape: tuple[int, int], expected_hash: str) -> np.ndarray:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"source hash mismatch: {path}")
    words = np.fromfile(path, dtype="<u2")
    if words.size != math.prod(shape):
        raise ValueError(f"source geometry mismatch: {path}")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite source: {path}")
    return values.astype(np.float64).reshape(shape)


def tangent_projection_energy(
    source: np.ndarray,
    decoded: np.ndarray,
    batch_blocks: int,
) -> tuple[float, float]:
    """Return exact error SSE and its rank-3 tangent projection energy."""

    source_blocks = np.ascontiguousarray(source).reshape(-1, BLOCK_SIDE, BLOCK_SIDE)
    decoded_blocks = np.ascontiguousarray(decoded).reshape(-1, BLOCK_SIDE, BLOCK_SIDE)
    if source_blocks.shape != (BLOCKS_PER_MATRIX, BLOCK_SIDE, BLOCK_SIDE):
        raise AssertionError(source_blocks.shape)
    error_sum = 0.0
    projection_sum = 0.0
    for begin in range(0, BLOCKS_PER_MATRIX, batch_blocks):
        stop = min(BLOCKS_PER_MATRIX, begin + batch_blocks)
        x = cp.asarray(source_blocks[begin:stop], dtype=cp.float64)
        y = cp.asarray(decoded_blocks[begin:stop], dtype=cp.float64)
        error = x - y
        u, _, vh = cp.linalg.svd(y, full_matrices=False)
        u = u[:, :, :TANGENT_RANK]
        v = vh[:, :TANGENT_RANK, :]

        # Orthogonal projection onto {U A^T + B V^T}:
        # P_U E + E P_V - P_U E P_V.
        left = cp.matmul(u, cp.matmul(cp.swapaxes(u, 1, 2), error))
        right = cp.matmul(cp.matmul(error, cp.swapaxes(v, 1, 2)), v)
        cross = cp.matmul(cp.matmul(left, cp.swapaxes(v, 1, 2)), v)
        projection = left + right - cross

        error_energy = cp.sum(error * error, dtype=cp.float64)
        projection_energy = cp.sum(projection * projection, dtype=cp.float64)
        # Orthogonal-projection identity is a useful independent numeric guard.
        inner = cp.sum(error * projection, dtype=cp.float64)
        pe = float(projection_energy.item())
        if abs(float(inner.item()) - pe) > 3e-10 * max(1.0, pe):
            raise FloatingPointError("tangent projection lost orthogonality")
        ee = float(error_energy.item())
        if pe < -1e-12 or pe > ee * (1.0 + 2e-12):
            raise FloatingPointError("invalid tangent projection energy")
        error_sum += ee
        projection_sum += pe
        del x, y, error, u, vh, v, left, right, cross, projection
    return error_sum, projection_sum


def jackknife_ratio_upper_three_se(error: np.ndarray, captured: np.ndarray) -> dict[str, Any]:
    total_error = float(np.sum(error, dtype=np.float64))
    total_capture = float(np.sum(captured, dtype=np.float64))
    estimate = total_capture / total_error
    delete_one = np.asarray(
        [
            (total_capture - float(captured[i])) / (total_error - float(error[i]))
            for i in range(EXPERTS)
        ],
        dtype=np.float64,
    )
    center = float(np.mean(delete_one, dtype=np.float64))
    se = math.sqrt(
        (EXPERTS - 1.0)
        / EXPERTS
        * float(np.sum(np.square(delete_one - center), dtype=np.float64))
    )
    return {
        "estimate": estimate,
        "delete_one_expert": delete_one.tolist(),
        "jackknife_center": center,
        "jackknife_se": se,
        "upper_three_se": estimate + 3.0 * se,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-blocks", type=int, default=32)
    args = parser.parse_args()
    started = time.time()
    plan_dir = args.plan_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if args.batch_blocks < 1 or args.batch_blocks > BLOCKS_PER_MATRIX:
        raise ValueError("batch-blocks outside closed range")

    plan_path = plan_dir / "plan.lock.json"
    header_path = plan_dir / "header.bin"
    post_path = plan_dir / "independent_audit/post_klt_canonical_groups.f64.bin"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    check_plan_seal(plan)
    if len(plan.get("sources", [])) != EXPERTS * ROLES:
        raise ValueError("plan source count mismatch")
    if post_path.stat().st_size != PANEL_VALUES * 8:
        raise ValueError("post-KLT reconstruction byte count mismatch")
    header = header_path.read_bytes()
    if len(header) != 128:
        raise ValueError("header byte count mismatch")
    coefficients = struct.unpack_from("<12f", header, 32)
    post = np.memmap(post_path, dtype="<f8", mode="r", shape=(EXPERTS * GROUPS_PER_EXPERT, GROUP))
    source_root = Path(plan["source_root"])

    expert_error = np.zeros(EXPERTS, dtype=np.float64)
    expert_capture = np.zeros(EXPERTS, dtype=np.float64)
    expert_source_energy = np.zeros(EXPERTS, dtype=np.float64)
    role_error = np.zeros(ROLES, dtype=np.float64)
    role_capture = np.zeros(ROLES, dtype=np.float64)
    matrix_rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []

    for expert_ordinal in range(EXPERTS):
        base = expert_ordinal * GROUPS_PER_EXPERT
        gate_hat = np.asarray(post[base : base + ROWS], dtype=np.float64)
        z0 = np.asarray(post[base + ROWS : base + 2 * ROWS], dtype=np.float64)
        z1 = np.asarray(post[base + 2 * ROWS : base + 3 * ROWS], dtype=np.float64)
        cosine = float(coefficients[2 * expert_ordinal])
        sine = float(coefficients[2 * expert_ordinal + 1])
        norm2 = cosine * cosine + sine * sine
        reconstructions = (
            gate_hat,
            (cosine * z0 - sine * z1) / norm2,
            (sine * z0 + cosine * z1) / norm2,
        )
        for role_ordinal, reconstruction in enumerate(reconstructions):
            matrix_ordinal = 3 * expert_ordinal + role_ordinal
            row = plan["sources"][matrix_ordinal]
            if int(row.get("matrix_ordinal", -1)) != matrix_ordinal:
                raise ValueError("source ordinal mismatch")
            role = str(row["role"])
            expected_role = ("gate", "up", "down")[role_ordinal]
            if role != expected_role:
                raise ValueError("role order mismatch")
            shape = (COLS, ROWS) if role == "down" else (ROWS, COLS)
            path = source_root / row["source_relpath"]
            source_raw = bf16_matrix(path, shape, str(row["source_bf16_sha256"]))
            source = source_raw.T.copy() if role == "down" else source_raw
            source_energy = float(np.sum(source * source, dtype=np.float64))
            error, capture = tangent_projection_energy(
                source,
                np.asarray(reconstruction, dtype=np.float64),
                args.batch_blocks,
            )
            expert_error[expert_ordinal] += error
            expert_capture[expert_ordinal] += capture
            expert_source_energy[expert_ordinal] += source_energy
            role_error[role_ordinal] += error
            role_capture[role_ordinal] += capture
            matrix_rows.append(
                {
                    "matrix_ordinal": matrix_ordinal,
                    "expert_ordinal": expert_ordinal,
                    "role": role,
                    "source_relpath": row["source_relpath"],
                    "source_bf16_sha256": row["source_bf16_sha256"],
                    "source_energy_fp64": source_energy,
                    "coarse_error_sse_fp64": error,
                    "tangent_projection_energy_fp64": capture,
                    "capture_fraction": capture / error,
                    "blocks": BLOCKS_PER_MATRIX,
                }
            )
            source_receipts.append(
                {
                    "matrix_ordinal": matrix_ordinal,
                    "path": str(path),
                    "sha256": row["source_bf16_sha256"],
                    "bytes": path.stat().st_size,
                }
            )
            print(
                f"matrix {matrix_ordinal:02d}/17 {role:4s} "
                f"capture={capture / error:.9f}",
                flush=True,
            )

    total_error = float(np.sum(expert_error, dtype=np.float64))
    total_energy = float(np.sum(expert_source_energy, dtype=np.float64))
    total_capture = float(np.sum(expert_capture, dtype=np.float64))
    if not math.isclose(total_error, EXPECTED_BASE_SSE, rel_tol=0.0, abs_tol=3e-9):
        raise AssertionError(("base SSE mismatch", total_error, EXPECTED_BASE_SSE))
    if not math.isclose(total_energy, EXPECTED_BASE_ENERGY, rel_tol=0.0, abs_tol=3e-9):
        raise AssertionError(("base energy mismatch", total_energy, EXPECTED_BASE_ENERGY))
    uncertainty = jackknife_ratio_upper_three_se(expert_error, expert_capture)
    decision = (
        "POLICY_REJECT_MALT64_R3_FAR_SHORT_STOP_BEFORE_CONTROLS"
        if uncertainty["upper_three_se"] < REQUIRED_CAPTURE
        else "POLICY_HOLD_MALT64_R3_FOR_CONTROLS_AND_FINITE_DESIGN"
    )
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu_name = properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8", errors="strict")
    result = {
        "schema": "malt64_decoded_svd_tangent_stage0_result_v0",
        "status": "complete",
        "decision": decision,
        "architecture": {
            "name": "MALT64-r3",
            "block_shape": [BLOCK_SIDE, BLOCK_SIDE],
            "decoded_svd_rank": TANGENT_RANK,
            "continuous_tangent_dimension": TANGENT_DIMENSION,
            "coarse_block_values": BLOCK_VALUES,
            "coset_bits_per_block": COSET_BITS_PER_BLOCK,
            "tangent_rank_fraction": TANGENT_DIMENSION / BLOCK_VALUES,
            "null_isotropic_capture": TANGENT_DIMENSION / BLOCK_VALUES,
            "stage0_grant": "arbitrary exact real coefficients in each decoder-derived tangent",
            "source_dependency": "none in tangent construction; source used only for projection score",
        },
        "physical_planning_ledger": {
            "coarse_bpw": COARSE_BPW,
            "coset_bpw": COSET_BPW,
            "metadata_bpw": METADATA_BPW,
            "total_bpw": TOTAL_BPW,
            "cold_page_read_amplification": READ_AMPLIFICATION,
            "required_coarse_error_capture": REQUIRED_CAPTURE,
            "favourable_base_F_transfer": BASE_F,
        },
        "aggregate": {
            "source_energy_fp64": total_energy,
            "coarse_error_sse_fp64": total_error,
            "coarse_relative_mse": total_error / total_energy,
            "tangent_projection_energy_fp64": total_capture,
            "capture_fraction": total_capture / total_error,
            "uncertainty": uncertainty,
            "required_capture": REQUIRED_CAPTURE,
            "fraction_of_required_at_upper_three_se": uncertainty["upper_three_se"] / REQUIRED_CAPTURE,
        },
        "experts": [
            {
                "expert_ordinal": i,
                "coarse_error_sse_fp64": float(expert_error[i]),
                "tangent_projection_energy_fp64": float(expert_capture[i]),
                "capture_fraction": float(expert_capture[i] / expert_error[i]),
            }
            for i in range(EXPERTS)
        ],
        "roles": [
            {
                "role": ("gate", "up", "down")[i],
                "coarse_error_sse_fp64": float(role_error[i]),
                "tangent_projection_energy_fp64": float(role_capture[i]),
                "capture_fraction": float(role_capture[i] / role_error[i]),
            }
            for i in range(ROLES)
        ],
        "matrices": matrix_rows,
        "bindings": {
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_internal_lock_sha256": plan["lock_sha256"],
            "header_path": str(header_path),
            "header_sha256": sha256_file(header_path),
            "post_klt_path": str(post_path),
            "post_klt_sha256": sha256_file(post_path),
            "sources": source_receipts,
        },
        "execution": {
            "backend": "cupy-fp64",
            "cupy_version": cp.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_id": int(cp.cuda.Device().id),
            "device_name": gpu_name,
            "batch_blocks": args.batch_blocks,
            "elapsed_seconds": time.time() - started,
        },
        "claim_boundary": (
            "Architecture-scoped continuous-span upper bound on MALT64-r3. "
            "It is not a finite codec, not a result at the lower coarse rate, "
            "and not a converse for arbitrary nonlinear decoder-conditioned frames."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "capture_fraction": total_capture / total_error,
                "upper_three_se": uncertainty["upper_three_se"],
                "required_capture": REQUIRED_CAPTURE,
                "elapsed_seconds": result["execution"]["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
