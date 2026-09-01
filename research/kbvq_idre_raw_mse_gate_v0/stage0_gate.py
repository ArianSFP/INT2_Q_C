#!/usr/bin/env python3
"""Source-locked KBVQ-IDRE raw-MSE gate.

The default path only rederives a sufficient-statistic hard kill from an
already authenticated aggregate result.  It neither imports CuPy nor opens a
numeric Qwen tensor.  A separately coordinated flag+token can independently
replay the same stacked-right-SVD statistic on the authorized auxiliary cache.

This is an optimistic architecture oracle, not a finite codec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "design_lock.json").read_text(encoding="utf-8"))
EVIDENCE = json.loads((HERE / "prior_evidence.json").read_text(encoding="utf-8"))

EXPERTS = 128
ROLES = ("up", "down")
ROWS = 768
COLS = 2048
N_MATRIX = ROWS * COLS
N_EXPERT = len(ROLES) * N_MATRIX
N_LAYER = EXPERTS * N_EXPERT
PAGE = 4096
SHARED_HEADER = 4096
EXPERT_HEADER = 512
FIT = tuple(LOCK["auxiliary_panel"]["fit_experts"])
VALIDATION = tuple(LOCK["auxiliary_panel"]["untouched_validation_experts"])
MANIFEST_SHA256 = LOCK["auxiliary_panel"]["authorized_manifest_sha256"]
REQUIRED_S = float(LOCK["objective"]["required_s_min_bpw"])
EXPECTED_TOKEN = LOCK["execution_interlock"]["independent_replay_requires_token"]
RATE_FRACTIONS = (Fraction(43, 20), Fraction(23, 10), Fraction(5, 2))


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def strict_regular(path: Path) -> None:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError(f"not a regular non-link file: {path}")


def load_json(path: Path) -> dict[str, Any]:
    strict_regular(path)
    return json.loads(path.read_text(encoding="utf-8"))


def shared_packet_bytes(mode: str, rank: int) -> int:
    copies = 1 if mode == "joint_role" else len(ROLES)
    return SHARED_HEADER + 2 * copies * COLS * rank


def private_factor_bytes(rank: int) -> int:
    return 2 * len(ROLES) * ROWS * rank


def union_page_count(shared_bytes: int, frame_start: int, frame_bytes: int) -> int:
    """Exact page union for [0,shared) and one [start,start+frame)."""
    if shared_bytes <= 0 or frame_bytes <= 0:
        raise ValueError("positive byte ranges required")
    shared_first, shared_last = 0, (shared_bytes - 1) // PAGE
    frame_first = frame_start // PAGE
    frame_last = (frame_start + frame_bytes - 1) // PAGE
    shared_pages = shared_last - shared_first + 1
    frame_pages = frame_last - frame_first + 1
    overlap_first = max(shared_first, frame_first)
    overlap_last = min(shared_last, frame_last)
    overlap = max(0, overlap_last - overlap_first + 1)
    return shared_pages + frame_pages - overlap


def layout(mode: str, rank: int, cap: Fraction) -> dict[str, Any]:
    shared = shared_packet_bytes(mode, rank)
    cap_bytes = (N_LAYER * cap.numerator) // (8 * cap.denominator)
    if shared >= cap_bytes:
        return {"legal": False, "reason": "shared packet exceeds container cap"}
    frame = (cap_bytes - shared) // EXPERTS
    emitted = shared + EXPERTS * frame
    private = private_factor_bytes(rank)
    residual = frame - EXPERT_HEADER - private
    if residual < 0:
        return {"legal": False, "reason": "FP16 private factors exceed expert frame"}
    worst_pages = 0
    worst_expert = None
    for expert in range(EXPERTS):
        start = shared + expert * frame
        pages = union_page_count(shared, start, frame)
        if pages > worst_pages:
            worst_pages, worst_expert = pages, expert
    equal_share = emitted / EXPERTS
    page_amp = worst_pages * PAGE / equal_share
    byte_amp = (shared + frame) / equal_share
    side_bytes = shared + EXPERTS * (EXPERT_HEADER + private)
    payload_bytes = EXPERTS * residual
    if side_bytes + payload_bytes != emitted:
        raise AssertionError("container byte closure failed")
    actual_bpw = 8.0 * emitted / N_LAYER
    side_bpw = 8.0 * side_bytes / N_LAYER
    payload_bpw = 8.0 * payload_bytes / N_LAYER
    return {
        "legal": page_amp < 2.0,
        "mode": mode,
        "rank": rank,
        "requested_cap_bpw": float(cap),
        "cap_bytes_floor": cap_bytes,
        "shared_packet_bytes": shared,
        "private_factor_bytes_per_expert": private,
        "expert_frame_bytes": frame,
        "residual_payload_bytes_per_expert": residual,
        "emitted_bytes": emitted,
        "unused_cap_bytes": cap_bytes - emitted,
        "actual_bpw": actual_bpw,
        "component_header_side_bpw": side_bpw,
        "residual_payload_bpw": payload_bpw,
        "closure_error_bpw": actual_bpw - side_bpw - payload_bpw,
        "cold_byte_amplification": byte_amp,
        "cold_worst_expert": worst_expert,
        "cold_worst_pages": worst_pages,
        "cold_page_bytes": worst_pages * PAGE,
        "cold_page_amplification": page_amp,
        "passes_page_read_lt_2": page_amp < 2.0,
    }


def layout_envelope() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("joint_role", "role_specific"):
        rows = []
        for cap in RATE_FRACTIONS:
            legal = [layout(mode, rank, cap) for rank in range(1, 257)]
            legal = [row for row in legal if row.get("legal")]
            winner = max(legal, key=lambda row: row["rank"])
            rows.append(winner)
        result[mode] = rows
    return result


def reverse_waterfill(
    energies: list[float], dimensions: list[int], payload_bits: float
) -> dict[str, Any]:
    """Exact diagonal-Gaussian reverse waterfill over declared components."""
    if len(energies) != len(dimensions) or not energies:
        raise ValueError("waterfill component closure")
    if payload_bits < 0 or any(e <= 0.0 for e in energies) or any(d <= 0 for d in dimensions):
        raise ValueError("positive waterfill inputs required")
    variances = [energy / dimension for energy, dimension in zip(energies, dimensions)]
    lo = min(variances) * 2.0 ** -120
    hi = max(variances)
    for _ in range(192):
        theta = math.sqrt(lo * hi)
        used = 0.5 * sum(
            dimension * max(math.log2(variance / theta), 0.0)
            for variance, dimension in zip(variances, dimensions)
        )
        if used > payload_bits:
            lo = theta
        else:
            hi = theta
    theta = hi
    rates = [0.5 * max(math.log2(variance / theta), 0.0) for variance in variances]
    allocated = sum(dimension * rate for dimension, rate in zip(dimensions, rates))
    distortion = sum(
        dimension * min(variance, theta)
        for variance, dimension in zip(variances, dimensions)
    )
    return {
        "water_level": theta,
        "component_variances": variances,
        "component_rates_bpw": rates,
        "allocated_bits": allocated,
        "distortion_energy": distortion,
    }


def physical_rows_from_prior(prior: dict[str, Any]) -> list[dict[str, Any]]:
    """Two-role Gaussian waterfill after the literal IDRE factor ledger.

    The captured subspace is granted exact despite its FP16 serialization.
    All bytes left in the expert frame are optimally split between the Up and
    Down residual bands.  This is favorable to the actual FP16 component, but
    it is only a two-band Gaussian gate; it is not a converse for within-band
    residual anisotropy.
    """
    validation_values = len(VALIDATION) * len(ROLES) * N_MATRIX
    rows = []
    for cell in prior["candidates"]:
        if cell["family"] != "right" or cell["left_rank"] != 0:
            continue
        mode = str(cell["mode"])
        rank = int(cell["right_rank"])
        by_role = cell["validation_component_energies"]
        source_energy = sum(float(by_role[role]["total"]) for role in ROLES)
        captured_energy = sum(float(by_role[role]["right"]) for role in ROLES)
        residual_energies = [
            float(by_role[role]["total"]) - float(by_role[role]["right"])
            for role in ROLES
        ]
        residual_dimensions = [len(VALIDATION) * ROWS * (COLS - rank) for _ in ROLES]
        for cap in RATE_FRACTIONS:
            ledger = layout(mode, rank, cap)
            if not ledger.get("legal"):
                continue
            payload_bits = ledger["residual_payload_bpw"] * validation_values
            waterfill = reverse_waterfill(residual_energies, residual_dimensions, payload_bits)
            relative_mse = waterfill["distortion_energy"] / source_energy
            f_value = relative_mse * 2.0 ** (2.0 * ledger["actual_bpw"])
            rows.append({
                "mode": mode,
                "rank": rank,
                "requested_cap_bpw": float(cap),
                "actual_bpw": ledger["actual_bpw"],
                "component_header_side_bpw": ledger["component_header_side_bpw"],
                "residual_payload_bpw": ledger["residual_payload_bpw"],
                "cold_page_amplification": ledger["cold_page_amplification"],
                "captured_dimensions": len(VALIDATION) * len(ROLES) * ROWS * rank,
                "residual_dimensions_by_role": residual_dimensions,
                "source_energy": source_energy,
                "captured_energy_exact_free_after_side_charge": captured_energy,
                "residual_energies_by_role": residual_energies,
                "water_level": waterfill["water_level"],
                "residual_role_variances": waterfill["component_variances"],
                "residual_role_rates_bpw": waterfill["component_rates_bpw"],
                "payload_bits_validation": payload_bits,
                "allocated_bits_validation": waterfill["allocated_bits"],
                "relative_mse_two_role_gaussian_oracle": relative_mse,
                "F_two_role_gaussian_oracle": f_value,
                "s_two_role_gaussian_oracle_bpw": -0.5 * math.log2(f_value),
                "passes_F_0p8": f_value <= 0.8,
                "fp16_factor_rounding_error_granted_zero": True,
            })
    rows.sort(key=lambda row: (row["F_two_role_gaussian_oracle"], row["actual_bpw"], row["rank"]))
    return rows


def derive_from_prior(result_path: Path, verification_path: Path) -> dict[str, Any]:
    expected = LOCK["prior_sufficient_statistic"]
    if sha256_file(result_path) != expected["artifact_sha256"]:
        raise RuntimeError("prior result SHA-256 mismatch")
    if sha256_file(verification_path) != expected["independent_verification_sha256"]:
        raise RuntimeError("prior verification SHA-256 mismatch")
    prior = load_json(result_path)
    receipt = load_json(verification_path)
    if prior.get("result_sha256") != expected["internal_result_seal"]:
        raise RuntimeError("prior internal result seal mismatch")
    if receipt.get("status") != "PASS" or receipt.get("decision") != prior.get("decision"):
        raise RuntimeError("prior independent verification is not a matching PASS")
    if tuple(prior["split"]["fit_experts"]) != FIT:
        raise RuntimeError("prior fit split mismatch")
    if tuple(prior["split"]["validation_experts"]) != VALIDATION:
        raise RuntimeError("prior validation split mismatch")

    rows = []
    for mode in ("joint_role", "role_specific"):
        matches = [
            cell for cell in prior["candidates"]
            if cell["mode"] == mode and cell["family"] == "right"
            and cell["right_rank"] == 256 and cell["left_rank"] == 0
        ]
        if len(matches) != 1:
            raise RuntimeError(f"missing unique prior rank-256 row: {mode}")
        energies = matches[0]["validation_component_energies"]
        total = sum(float(energies[role]["total"]) for role in ROLES)
        captured = sum(float(energies[role]["right"]) for role in ROLES)
        q = 1.0 - captured / total
        s = -0.5 * math.log2(q)
        rows.append({
            "mode": mode,
            "rank": 256,
            "source_energy": total,
            "captured_energy": captured,
            "capture": captured / total,
            "residual_energy_ratio_q": q,
            "free_s_bpw": s,
            "passes_required_s": s >= REQUIRED_S,
        })
    ledgers = layout_envelope()
    max_legal = max(row["rank"] for mode_rows in ledgers.values() for row in mode_rows)
    if max_legal >= 256:
        raise RuntimeError("rank-256 statistic does not dominate every legal rank")
    physical_rows = physical_rows_from_prior(prior)
    if not physical_rows:
        raise RuntimeError("no physically legal prior-IDRE waterfill rows")
    best = physical_rows[0]
    bounded_kill = best["F_two_role_gaussian_oracle"] > 0.8
    return {
        "decision": (
            "HARD_KILL_BOUNDED_TWO_ROLE_GAUSSIAN_IDRE_CELL"
            if bounded_kill else "SURVIVE_TWO_ROLE_GATE_REQUIRES_PER_MODE_REPLAY"
        ),
        "numeric_qwen_payload_files_opened": 0,
        "gpu_imported": False,
        "rank256_exact_free_envelopes": rows,
        "serialized_layout_maxima": ledgers,
        "maximum_legal_rank": max_legal,
        "physical_two_role_waterfill": {
            "scope": "exact factor/header bytes; exact free component after its side charge; optimal Gaussian bit allocation between Up and Down residual bands",
            "not_claimed": "not an exact per-mode residual waterfill and not a universal rate-distortion converse",
            "best": best,
            "rows": physical_rows,
        },
        "free_capture_warning": (
            "s=-0.5*log2(1-capture) is reported only as gross signal. It is not the physical score: "
            "the physical score must charge factors and reallocate remaining bits over residual dimensions."
        ),
        "composite_assessment": {
            "existing_composite_s_bpw": 0.0474034129,
            "incremental_s_needed_bpw": REQUIRED_S - 0.0474034129,
            "hypothetical_illegal_free_role_specific_sum_s_bpw": 0.0474034129 + rows[1]["free_s_bpw"],
            "hypothetical_illegal_free_role_specific_F": 2.0 ** (-2.0 * (0.0474034129 + rows[1]["free_s_bpw"])),
            "verdict": "NO_ADDITIVE_CLAIM",
            "reason": (
                "The auxiliary IDRE projection and pinned composite can address overlapping source energy, use different samples, "
                "and compete for the same physical bits. A valid nesting test must apply the basis to the literal composite residual "
                "and redo one joint byte allocation; neither containment nor disjointness follows from rank monotonicity."
            ),
        },
    }


def load_manifest(root: Path, manifest_path: Path) -> dict[tuple[int, str], Path]:
    """Authorized numeric access: hash every source before decoding it."""
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("authorized auxiliary manifest SHA-256 mismatch")
    manifest = load_json(manifest_path)
    lookup: dict[tuple[int, str], Path] = {}
    for row in manifest["tensors"]:
        key = (int(row["expert"]), str(row["role"]))
        if key in lookup:
            raise RuntimeError(f"duplicate tensor binding: {key}")
        path = (root / row["local_path"]).resolve()
        strict_regular(path)
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"source byte mismatch: {key}")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"source hash mismatch: {key}")
        lookup[key] = path
    expected = {(expert, role) for expert in FIT + VALIDATION for role in ROLES}
    if set(lookup) != expected:
        raise RuntimeError("auxiliary manifest identity closure mismatch")
    return lookup


def load_bf16(path: Path, role: str, cp: Any) -> Any:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != N_MATRIX:
        raise RuntimeError(f"wrong source element count: {path}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    matrix = values.reshape((ROWS, COLS) if role == "up" else (COLS, ROWS))
    if role == "down":
        matrix = matrix.T
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"nonfinite source: {path}")
    return cp.asarray(matrix, dtype=cp.float32)


def fit_bases(lookup: dict[tuple[int, str], Path], cp: Any) -> dict[str, Any]:
    cov_joint = cp.zeros((COLS, COLS), dtype=cp.float32)
    cov_roles = {role: cp.zeros((COLS, COLS), dtype=cp.float32) for role in ROLES}
    for expert in FIT:
        for role in ROLES:
            matrix = load_bf16(lookup[(expert, role)], role, cp)
            gram = matrix.T @ matrix
            cov_joint += gram
            cov_roles[role] += gram
            del matrix, gram
    _, joint = cp.linalg.eigh(cov_joint)
    bases = {"joint_role": {role: cp.ascontiguousarray(joint[:, ::-1]) for role in ROLES}}
    bases["role_specific"] = {}
    for role in ROLES:
        _, basis = cp.linalg.eigh(cov_roles[role])
        bases["role_specific"][role] = cp.ascontiguousarray(basis[:, ::-1])
    return bases


def replay_mode_spectra(
    lookup: dict[tuple[int, str], Path], cp: Any
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    bases = fit_bases(lookup, cp)
    spectra = {
        mode: {role: cp.zeros(COLS, dtype=cp.float64) for role in ROLES}
        for mode in ("joint_role", "role_specific")
    }
    for expert in VALIDATION:
        for role in ROLES:
            matrix = load_bf16(lookup[(expert, role)], role, cp)
            for mode in spectra:
                projected = matrix @ bases[mode][role]
                spectra[mode][role] += cp.sum(projected.astype(cp.float64) ** 2, axis=0)
                del projected
            del matrix
            cp.get_default_memory_pool().free_all_blocks()
    rows = []
    host_spectra = {
        mode: {role: cp.asnumpy(values).astype(float).tolist() for role, values in by_role.items()}
        for mode, by_role in spectra.items()
    }
    for mode, by_role in host_spectra.items():
        total = sum(sum(by_role[role]) for role in ROLES)
        captured = sum(sum(by_role[role][:256]) for role in ROLES)
        q = 1.0 - captured / total
        s = -0.5 * math.log2(q)
        rows.append({
            "mode": mode,
            "rank": 256,
            "source_energy": total,
            "captured_energy": captured,
            "capture": captured / total,
            "residual_energy_ratio_q": q,
            "free_s_bpw": s,
            "passes_required_s": s >= REQUIRED_S,
        })
    return rows, host_spectra


def per_mode_physical_rows(
    spectra: dict[str, dict[str, list[float]]]
) -> list[dict[str, Any]]:
    """Exact validation per-mode diagonal-Gaussian waterfill for a replay."""
    validation_values = len(VALIDATION) * len(ROLES) * N_MATRIX
    mode_dimension = len(VALIDATION) * ROWS
    frozen_ranks = (8, 16, 32, 64, 96, 128, 192, 256)
    rows = []
    for mode, by_role in spectra.items():
        source_energy = sum(sum(by_role[role]) for role in ROLES)
        for rank in frozen_ranks:
            captured = sum(sum(by_role[role][:rank]) for role in ROLES)
            residual_energies = [
                float(energy)
                for role in ROLES
                for energy in by_role[role][rank:]
            ]
            residual_dimensions = [mode_dimension] * len(residual_energies)
            for cap in RATE_FRACTIONS:
                ledger = layout(mode, rank, cap)
                if not ledger.get("legal"):
                    continue
                payload_bits = ledger["residual_payload_bpw"] * validation_values
                waterfill = reverse_waterfill(residual_energies, residual_dimensions, payload_bits)
                relative_mse = waterfill["distortion_energy"] / source_energy
                f_value = relative_mse * 2.0 ** (2.0 * ledger["actual_bpw"])
                rows.append({
                    "mode": mode,
                    "rank": rank,
                    "requested_cap_bpw": float(cap),
                    "actual_bpw": ledger["actual_bpw"],
                    "component_header_side_bpw": ledger["component_header_side_bpw"],
                    "residual_payload_bpw": ledger["residual_payload_bpw"],
                    "cold_page_amplification": ledger["cold_page_amplification"],
                    "captured_energy_exact_free_after_side_charge": captured,
                    "residual_mode_count": len(residual_energies),
                    "residual_mode_dimension": mode_dimension,
                    "water_level": waterfill["water_level"],
                    "payload_bits_validation": payload_bits,
                    "allocated_bits_validation": waterfill["allocated_bits"],
                    "relative_mse_per_mode_gaussian_oracle": relative_mse,
                    "F_per_mode_gaussian_oracle": f_value,
                    "s_per_mode_gaussian_oracle_bpw": -0.5 * math.log2(f_value),
                    "passes_F_0p8": f_value <= 0.8,
                    "fp16_factor_rounding_error_granted_zero": True,
                })
    rows.sort(key=lambda row: (row["F_per_mode_gaussian_oracle"], row["actual_bpw"], row["rank"]))
    return rows


def independent_replay(root: Path, manifest_path: Path) -> dict[str, Any]:
    started = time.time()
    lookup = load_manifest(root, manifest_path)
    import cupy as cp  # Deliberately unreachable without the coordinated CLI interlock.

    rows, spectra = replay_mode_spectra(lookup, cp)
    physical_rows = per_mode_physical_rows(spectra)
    best = physical_rows[0]
    hard_kill = best["F_per_mode_gaussian_oracle"] > 0.8
    return {
        "decision": (
            "HARD_KILL_IDRE_RAW_MSE_EXACT_PER_MODE_GAUSSIAN_REPLAY"
            if hard_kill else "DISCREPANCY_SURVIVOR_STOP_AND_RESEAL_BEFORE_ANY_FINITE_WORK"
        ),
        "numeric_qwen_payload_files_opened": len(lookup),
        "gpu_imported": True,
        "rank256_exact_free_envelopes": rows,
        "serialized_layout_maxima": layout_envelope(),
        "physical_per_mode_waterfill": {
            "scope": "exact validation energies in every fitted right mode; FP16 component error still granted zero after literal side charge",
            "best": best,
            "rows": physical_rows,
        },
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--prior-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-independent-payload-replay", action="store_true")
    parser.add_argument("--authorization-token", default="")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()

    if args.authorize_independent_payload_replay:
        if args.authorization_token != EXPECTED_TOKEN:
            raise RuntimeError("coordinated replay token mismatch")
        if args.manifest is None or args.root is None:
            raise RuntimeError("independent replay requires --manifest and --root")
        result = independent_replay(args.root.resolve(), args.manifest.resolve())
    else:
        if args.authorization_token or args.manifest is not None or args.root is not None:
            raise RuntimeError("payload arguments are forbidden without the coordinated replay flag")
        result = derive_from_prior(args.prior_result.resolve(), args.prior_verification.resolve())

    payload = {
        "schema": "kbvq_idre_raw_mse_gate_result_v0",
        "status": "COMPLETE_OPTIMISTIC_GATE_NOT_A_CODEC",
        "design_lock_sha256": sha256_file(HERE / "design_lock.json"),
        "stage0_script_sha256": sha256_file(Path(__file__).resolve()),
        "required_s_bpw": REQUIRED_S,
        "required_F_max": 0.8,
        **result,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    payload["internal_result_seal"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
