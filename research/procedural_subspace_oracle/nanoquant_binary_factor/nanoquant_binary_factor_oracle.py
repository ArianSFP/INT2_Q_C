#!/usr/bin/env python3
"""CPU-only spectral gate for NanoQuant-style binary matrix factorization.

The tested architecture follows NanoQuant's storage form

    W_hat = diag(s1) U_{+-1} V_{+-1}^T diag(s2)

with factor bits r(n+m) and two FP16 scale vectors costing 16(n+m).
Before attempting its expensive discrete ADMM optimization, this script tests
the necessary motivating hypothesis: that the pinned Qwen matrices are more
continuously low-rank/spectrally compressible than independently optimized,
moment-matched Gaussian matrices.  Exact continuous SVD tails are evaluated on
the full matrices, both directly and after a free full-precision row/column RMS
equilibration.  The latter is a scale-normalized structural diagnostic, not a
source-domain reconstruction bound.

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
RESULT_SCHEMA = "qwen-nanoquant-binary-factor-spectral-gate-v1"
PAPER_URL = "https://arxiv.org/abs/2602.06694"
PAPER_VERSION = "2602.06694v3"
REQUIRED_USER_S = -0.5 * math.log2(0.8)
PROMOTION_S = 0.153
HEADER_BITS_PER_EXPERT = 512
TARGET_RATES = (2.15, 2.5)
RANK_MULTIPLE = 32
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
        raise ValueError("plan canonical lock mismatch")
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


def read_panel(plan_path: Path) -> tuple[dict[str, list[list[np.ndarray]]], dict[str, Any]]:
    plan = validate_plan(plan_path)
    xklt, header = parse_xklt(plan_path, plan)
    source_root = Path(plan["source_root"]).resolve(strict=True)
    raw: list[list[np.ndarray]] = []
    transformed: list[list[np.ndarray]] = []
    source_rows: list[dict[str, Any]] = []
    for expert in range(EXPERTS):
        triplet = plan["sources"][3 * expert : 3 * expert + 3]
        if [row["role"] for row in triplet] != ["gate", "up", "down"]:
            raise ValueError("source role order mismatch")
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
            matrices.append(load_bf16(path, row["role"]))
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
    provenance = {
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_lock_sha256": plan["lock_sha256"],
        "source_root": str(source_root),
        "header": header,
        "sources": sorted(source_rows, key=lambda row: row["matrix_ordinal"]),
    }
    return {"raw": raw, "xklt": transformed}, provenance


def center_and_match_gaussian(
    matrix: np.ndarray,
    plan_lock: str,
    representation: str,
    expert: int,
    role: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source = matrix.astype(np.float64)
    source_mean = float(np.mean(source, dtype=np.float64))
    source -= source_mean
    source_energy = float(np.sum(source * source, dtype=np.float64))
    if source_energy <= 0.0:
        raise ValueError("non-positive centered source energy")
    seed = stable_seed(
        "NQ-BINARY-GAUSSIAN-CONTROL-v1", plan_lock, representation, expert, role
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    gaussian = rng.standard_normal(source.shape, dtype=np.float64)
    gaussian -= float(np.mean(gaussian, dtype=np.float64))
    gaussian_energy = float(np.sum(gaussian * gaussian, dtype=np.float64))
    gaussian *= math.sqrt(source_energy / gaussian_energy)
    matched_energy = float(np.sum(gaussian * gaussian, dtype=np.float64))
    return source, gaussian, {
        "source_mean_fp64": source_mean,
        "source_centered_energy_fp64": source_energy,
        "gaussian_centered_energy_fp64": matched_energy,
        "relative_energy_match_error": abs(matched_energy / source_energy - 1.0),
    }


def equilibrate(matrix: np.ndarray, iterations: int) -> tuple[np.ndarray, dict[str, float]]:
    z = np.asarray(matrix, dtype=np.float64, order="C").copy()
    row_scale = np.ones(z.shape[0], dtype=np.float64)
    col_scale = np.ones(z.shape[1], dtype=np.float64)
    for _ in range(iterations):
        row_rms = np.sqrt(np.mean(z * z, axis=1, dtype=np.float64))
        row_rms = np.maximum(row_rms, 1e-300)
        z /= row_rms[:, None]
        row_scale *= row_rms
        col_rms = np.sqrt(np.mean(z * z, axis=0, dtype=np.float64))
        col_rms = np.maximum(col_rms, 1e-300)
        z /= col_rms[None, :]
        col_scale *= col_rms
    energy = float(np.sum(z * z, dtype=np.float64))
    z /= math.sqrt(energy)
    return z, {
        "iterations": iterations,
        "row_scale_min": float(np.min(row_scale)),
        "row_scale_max": float(np.max(row_scale)),
        "column_scale_min": float(np.min(col_scale)),
        "column_scale_max": float(np.max(col_scale)),
        "normalized_energy_before_unit_scaling": energy,
    }


def normalized_spectrum(matrix: np.ndarray) -> np.ndarray:
    # Gram is only 768x768 and gives an exact full singular spectrum.
    gram = matrix @ matrix.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
    total = float(np.sum(eigenvalues, dtype=np.float64))
    if total <= 0.0:
        raise ValueError("non-positive spectral energy")
    return eigenvalues / total


def residual_at_rank(spectrum: np.ndarray, rank: int) -> float:
    return float(np.sum(spectrum[rank:], dtype=np.float64))


def candidate_record(
    representation: str,
    scaling: str,
    rank: int,
    source_residual: list[float],
    gaussian_residual: list[float],
    energy: list[float],
) -> dict[str, Any]:
    source_sse = [source_residual[i] * energy[i] for i in range(EXPERTS)]
    gaussian_sse = [gaussian_residual[i] * energy[i] for i in range(EXPERTS)]
    source_d = sum(source_sse) / sum(energy)
    gaussian_d = sum(gaussian_sse) / sum(energy)
    ratio = source_d / gaussian_d
    s_bpw = -0.5 * math.log2(ratio)
    return {
        "candidate_id": f"{representation}:{scaling}:rank{rank}",
        "representation": representation,
        "scaling": scaling,
        "continuous_rank": rank,
        "source_residual_fraction_by_expert": source_residual,
        "gaussian_residual_fraction_by_expert": gaussian_residual,
        "source_residual_sse_by_expert": source_sse,
        "gaussian_residual_sse_by_expert": gaussian_sse,
        "source_energy_by_expert": energy,
        "source_relative_residual": source_d,
        "gaussian_relative_residual": gaussian_d,
        "source_over_matched_gaussian": ratio,
        "percent_below_matched_gaussian": 100.0 * (1.0 - ratio),
        "structural_advantage_bpw": s_bpw,
        "F_ratio_identity": 2.0 ** (-2.0 * s_bpw),
    }


def crossfit(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    total_source = total_gaussian = total_energy = 0.0
    for heldout in range(EXPERTS):
        def training_key(candidate: dict[str, Any]) -> tuple[float, str]:
            source = sum(
                value for i, value in enumerate(candidate["source_residual_sse_by_expert"])
                if i != heldout
            )
            gaussian = sum(
                value for i, value in enumerate(candidate["gaussian_residual_sse_by_expert"])
                if i != heldout
            )
            return source / gaussian, candidate["candidate_id"]

        winner = min(candidates, key=training_key)
        source = float(winner["source_residual_sse_by_expert"][heldout])
        gaussian = float(winner["gaussian_residual_sse_by_expert"][heldout])
        energy = float(winner["source_energy_by_expert"][heldout])
        total_source += source
        total_gaussian += gaussian
        total_energy += energy
        selections.append(
            {
                "heldout_expert": heldout,
                "selected_candidate_id": winner["candidate_id"],
                "training_source_over_gaussian": training_key(winner)[0],
                "heldout_source_sse": source,
                "heldout_gaussian_sse": gaussian,
                "heldout_energy": energy,
            }
        )
    source_d = total_source / total_energy
    gaussian_d = total_gaussian / total_energy
    ratio = source_d / gaussian_d
    s_bpw = -0.5 * math.log2(ratio)
    return {
        "selection_rule": "minimize train-five-expert source/matched-Gaussian residual ratio",
        "selections": selections,
        "pooled_heldout_source_relative_residual": source_d,
        "pooled_heldout_gaussian_relative_residual": gaussian_d,
        "pooled_heldout_source_over_gaussian": ratio,
        "pooled_heldout_percent_below_gaussian": 100.0 * (1.0 - ratio),
        "pooled_heldout_structural_advantage_bpw": s_bpw,
        "F_ratio_identity": 2.0 ** (-2.0 * s_bpw),
    }


def factor_ledger(requested_rate: float) -> dict[str, Any]:
    n, m = ROWS, COLS
    physical_bytes = math.ceil(WEIGHTS_PER_EXPERT * requested_rate / 8.0)
    physical_bits = physical_bytes * 8
    bits_after_header = physical_bits - HEADER_BITS_PER_EXPERT
    factor_axis = n + m
    scale_bits_per_matrix = 16 * factor_axis

    # Radical overcomplete extension: use the largest multiple-of-32 r whose
    # three matrix factors plus scales fit in the private expert stream.
    raw_rank = (bits_after_header / ROLES - scale_bits_per_matrix) / factor_axis
    overcomplete_rank = max(0, int(raw_rank) // RANK_MULTIPLE * RANK_MULTIPLE)
    matrix_bits = (overcomplete_rank + 16) * factor_axis
    used_bits = HEADER_BITS_PER_EXPERT + ROLES * matrix_bits
    padding_bits = physical_bits - used_bits
    if padding_bits < 0:
        raise ValueError("negative factor-ledger padding")

    official_rank = min(overcomplete_rank, min(n, m))
    official_matrix_bits = (official_rank + 16) * factor_axis
    official_used_bits = HEADER_BITS_PER_EXPERT + ROLES * official_matrix_bits
    return {
        "requested_rate_bpw": requested_rate,
        "physical_rate_bpw": physical_bits / WEIGHTS_PER_EXPERT,
        "physical_bytes_per_expert": physical_bytes,
        "physical_bits_per_expert": physical_bits,
        "private_header_bits_per_expert": HEADER_BITS_PER_EXPERT,
        "matrices_per_expert": ROLES,
        "matrix_shape": [n, m],
        "factor_bits_per_matrix": overcomplete_rank * factor_axis,
        "fp16_scale_bits_per_matrix": scale_bits_per_matrix,
        "overcomplete_rank_multiple_32": overcomplete_rank,
        "overcomplete_used_bits_per_expert": used_bits,
        "overcomplete_padding_bits_per_expert": padding_bits,
        "overcomplete_useful_bpw": (used_bits - HEADER_BITS_PER_EXPERT) / WEIGHTS_PER_EXPERT,
        "official_code_rank_cap": min(n, m),
        "official_capped_rank": official_rank,
        "official_capped_used_bits_per_expert": official_used_bits,
        "official_capped_useful_bpw_excluding_header": (
            official_used_bits - HEADER_BITS_PER_EXPERT
        ) / WEIGHTS_PER_EXPERT,
        "shared_table_bits": 0,
        "cold_expert_payload_bytes": physical_bytes,
        "cold_expert_bytes_read": physical_bytes,
        "cold_expert_read_amplification": 1.0,
        "cold_read_strictly_below_2x": True,
    }


def parse_csv(text: str, cast: Any) -> list[Any]:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representations", default="raw,xklt")
    parser.add_argument("--scalings", default="plain,rowcol_rms_free")
    parser.add_argument(
        "--ranks",
        default="1,2,4,8,16,32,64,128,256,384,512,640,704,736,752,760,764",
    )
    parser.add_argument("--equilibration-iters", type=int, default=8)
    args = parser.parse_args()

    representations = parse_csv(args.representations, str)
    scalings = parse_csv(args.scalings, str)
    ranks = sorted(set(parse_csv(args.ranks, int)))
    if any(rep not in ("raw", "xklt") for rep in representations):
        raise ValueError("representation must be raw or xklt")
    if any(scale not in ("plain", "rowcol_rms_free") for scale in scalings):
        raise ValueError("unknown scale relaxation")
    if not ranks or any(not 0 < rank < ROWS for rank in ranks):
        raise ValueError("continuous rank grid must lie in 1..767")

    started = time.time()
    panel, provenance = read_panel(args.plan.resolve(strict=True))
    spectra: dict[tuple[str, str, int, int, str], np.ndarray] = {}
    energy_by_rep: dict[str, list[float]] = {}
    matrix_receipts: list[dict[str, Any]] = []
    for representation in representations:
        expert_energy = [0.0] * EXPERTS
        for expert in range(EXPERTS):
            for role in range(ROLES):
                print(
                    f"spectrum representation={representation} expert={expert} role={role}",
                    flush=True,
                )
                source, gaussian, moment = center_and_match_gaussian(
                    panel[representation][expert][role],
                    provenance["plan_lock_sha256"],
                    representation,
                    expert,
                    role,
                )
                expert_energy[expert] += moment["source_centered_energy_fp64"]
                for scaling in scalings:
                    if scaling == "plain":
                        source_work = source / math.sqrt(moment["source_centered_energy_fp64"])
                        gaussian_work = gaussian / math.sqrt(moment["gaussian_centered_energy_fp64"])
                        source_scale_receipt = {"iterations": 0}
                        gaussian_scale_receipt = {"iterations": 0}
                    else:
                        source_work, source_scale_receipt = equilibrate(
                            source, args.equilibration_iters
                        )
                        gaussian_work, gaussian_scale_receipt = equilibrate(
                            gaussian, args.equilibration_iters
                        )
                    spectra[(representation, scaling, expert, role, "source")] = normalized_spectrum(
                        source_work
                    )
                    spectra[(representation, scaling, expert, role, "gaussian")] = normalized_spectrum(
                        gaussian_work
                    )
                    matrix_receipts.append(
                        {
                            "representation": representation,
                            "expert": expert,
                            "role": role,
                            "scaling": scaling,
                            "moments": moment,
                            "source_scaling_receipt": source_scale_receipt,
                            "gaussian_scaling_receipt": gaussian_scale_receipt,
                            "source_spectrum_sum": float(
                                np.sum(
                                    spectra[(representation, scaling, expert, role, "source")],
                                    dtype=np.float64,
                                )
                            ),
                            "gaussian_spectrum_sum": float(
                                np.sum(
                                    spectra[(representation, scaling, expert, role, "gaussian")],
                                    dtype=np.float64,
                                )
                            ),
                        }
                    )
        energy_by_rep[representation] = expert_energy

    candidates: list[dict[str, Any]] = []
    for representation in representations:
        for scaling in scalings:
            for rank in ranks:
                source_by_expert: list[float] = []
                gaussian_by_expert: list[float] = []
                for expert in range(EXPERTS):
                    energy = energy_by_rep[representation][expert]
                    source_weighted = gaussian_weighted = 0.0
                    role_energies = []
                    for role in range(ROLES):
                        moment = next(
                            row["moments"]
                            for row in matrix_receipts
                            if row["representation"] == representation
                            and row["expert"] == expert
                            and row["role"] == role
                            and row["scaling"] == scaling
                        )
                        role_energy = float(moment["source_centered_energy_fp64"])
                        role_energies.append(role_energy)
                        source_weighted += role_energy * residual_at_rank(
                            spectra[(representation, scaling, expert, role, "source")], rank
                        )
                        gaussian_weighted += role_energy * residual_at_rank(
                            spectra[(representation, scaling, expert, role, "gaussian")], rank
                        )
                    source_by_expert.append(source_weighted / energy)
                    gaussian_by_expert.append(gaussian_weighted / energy)
                candidates.append(
                    candidate_record(
                        representation,
                        scaling,
                        rank,
                        source_by_expert,
                        gaussian_by_expert,
                        energy_by_rep[representation],
                    )
                )

    panel_best = max(candidates, key=lambda row: (row["structural_advantage_bpw"], row["candidate_id"]))
    loo = crossfit(candidates)
    panel_s = float(panel_best["structural_advantage_bpw"])
    loo_s = float(loo["pooled_heldout_structural_advantage_bpw"])
    early_kill = bool(panel_s < PROMOTION_S and loo_s < PROMOTION_S)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "EARLY_KILL_BEFORE_DISCRETE_ADMM" if early_kill else "PROMOTE_TO_DISCRETE_ADMM",
        "claim_boundary": (
            "Full-matrix continuous spectral structural gate, not binary-factor codec MSE. Plain "
            "SVD tails are source-domain; row/column-equilibrated tails are normalized-domain "
            "diagnostics. A negative gate rejects this research branch but is not a universal "
            "lower bound on all binary factorizations."
        ),
        "reference": {
            "paper": "NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models",
            "arxiv": PAPER_URL,
            "version_read": PAPER_VERSION,
            "paper_equation": "W_hat=diag(s1) U_{+-1} V_{+-1}^T diag(s2)",
            "paper_storage_bits_per_matrix": "r(n+m)+16(n+m)",
            "official_repository": "https://github.com/SamsungLabs/NanoQuant",
        },
        "gpu_policy": {
            "cpu_only": True,
            "imports_torch": False,
            "imports_cupy": False,
            "invokes_cuda": False,
        },
        "protocol": {
            "matrix_shape": [ROWS, COLS],
            "representations": representations,
            "scale_relaxations": scalings,
            "continuous_ranks": ranks,
            "equilibration_iterations": args.equilibration_iters,
            "moment_match": "per-matrix exact mean and centered FP64 energy",
            "spectrum": "exact eigvalsh of 768x768 Gram matrix",
            "matrix_coverage": "all 18 matrices, no sampling",
            "crossfit": "six leave-one-expert-out folds",
        },
        "target": {
            "user_required_F": 0.8,
            "user_required_structural_advantage_bpw": REQUIRED_USER_S,
            "promotion_threshold_bpw": PROMOTION_S,
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
        "matrix_receipts": matrix_receipts,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "favorable_continuous_gate": {
            "panel_leaky_best": panel_best,
            "leave_one_expert_out": loo,
            "panel_shortfall_multiple_vs_promotion": (
                PROMOTION_S / panel_s if panel_s > 0.0 else None
            ),
            "loo_shortfall_multiple_vs_promotion": (
                PROMOTION_S / loo_s if loo_s > 0.0 else None
            ),
        },
        "physical_factor_ledgers": [factor_ledger(rate) for rate in TARGET_RATES],
        "decision": {
            "early_kill_before_discrete_admm": early_kill,
            "promotion_threshold_bpw": PROMOTION_S,
            "panel_leaky_s_bpw": panel_s,
            "loo_s_bpw": loo_s,
            "criterion": "panel-leaky and LOO continuous structural s are both below 0.153 bpw",
            "reason": (
                "The matched continuous spectrum supplies too little Qwen-specific advantage to "
                "justify the much more expensive binary ADMM branch."
                if early_kill
                else "The continuous structural gate is large enough to justify binary ADMM."
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
                "panel_leaky_s_bpw": panel_s,
                "loo_s_bpw": loo_s,
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
