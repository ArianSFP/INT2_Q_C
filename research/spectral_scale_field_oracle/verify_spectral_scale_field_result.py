#!/usr/bin/env python3
"""Dependency-free verifier for the spectral scale-field oracle receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TOTAL_VALUES = 28_311_552
EXPERTS = 6
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
SCHEMA = "qwen-spectral-scale-field-oracle-v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float, tolerance: float = 2e-11) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def verify_internal(value: dict[str, Any], field: str) -> str:
    clean = dict(value)
    declared = clean.pop(field, None)
    actual = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    require(declared == actual, f"{field} mismatch")
    return actual


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "representation",
            "model_id",
            "family",
            "parameters",
            "accounting",
            "physical_rate_bpw",
            "coefficient_rate_bpw",
            "relative_mse",
            "F",
            "s_bpw",
            "passes_F_le_0p8",
            "side_bits_total",
            "cold_expert_read_amplification",
            "cold_read_strictly_below_2x",
            "matched_gaussian_control",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    result_lock = verify_internal(result, "result_lock_sha256")
    require(result["schema"] == SCHEMA, "unexpected schema")
    require(close(result["target"]["required_F"], TARGET_F), "target F changed")
    require(close(result["target"]["required_s_bpw"], TARGET_S), "target s changed")
    require(result["protocol"]["cpu_only"] is True, "CPU-only flag missing")
    require(result["protocol"]["imports_cupy"] is False, "CuPy import claimed")
    require(result["protocol"]["invokes_cuda"] is False, "CUDA invocation claimed")

    script = args.result.parent / "spectral_scale_field_oracle.py"
    require(script.is_file(), "sibling algorithm script missing")
    require(
        sha256_file(script) == result["provenance"]["algorithm_sha256"],
        "algorithm hash mismatch",
    )

    rows = result["candidate_rows"]
    require(len(rows) > 0, "candidate table is empty")
    for index, row in enumerate(rows):
        prefix = f"candidate_rows[{index}]"
        rate = float(row["physical_rate_bpw"])
        mse = float(row["relative_mse"])
        require(2.15 <= rate <= 2.5, f"{prefix}: rate outside interval")
        require(mse > 0.0 and math.isfinite(mse), f"{prefix}: invalid MSE")
        expected_f = mse * 2.0 ** (2.0 * rate)
        expected_s = -0.5 * math.log2(expected_f)
        require(close(row["F"], expected_f), f"{prefix}: F formula mismatch")
        require(close(row["s_bpw"], expected_s), f"{prefix}: s formula mismatch")
        require(
            bool(row["passes_F_le_0p8"]) == (expected_f <= TARGET_F),
            f"{prefix}: pass flag mismatch",
        )
        expected_coefficient_rate = rate - float(row["side_bits_total"]) / TOTAL_VALUES
        require(
            close(row["coefficient_rate_bpw"], expected_coefficient_rate),
            f"{prefix}: side-rate ledger mismatch",
        )
        expert_bits = [float(value) for value in row["expert_physical_bits"]]
        require(len(expert_bits) == EXPERTS, f"{prefix}: expert ledger length")
        total = sum(expert_bits)
        amp = max(expert_bits) / (total / EXPERTS)
        require(close(row["physical_total_bits"], total), f"{prefix}: total bits mismatch")
        require(close(row["cold_expert_read_amplification"], amp), f"{prefix}: read amp mismatch")
        require(
            bool(row["cold_read_strictly_below_2x"]) == (amp < 2.0),
            f"{prefix}: read pass mismatch",
        )
        # FP64 summation over up to 28,311,552 independently allocated cells.
        # One millibit is <3.6e-11 bpw and is a deliberately strict closure.
        require(
            abs(float(row["rate_closure_error_bits"])) < 1e-3,
            f"{prefix}: physical rate does not close",
        )
        control = row["matched_gaussian_control"]
        samples = [float(value) for value in control["replicate_relative_mse"]]
        require(len(samples) >= 2, f"{prefix}: too few Gaussian controls")
        mean = sum(samples) / len(samples)
        ratio = mse / mean
        require(close(control["mean_relative_mse"], mean), f"{prefix}: control mean mismatch")
        require(close(control["source_over_control_mean"], ratio), f"{prefix}: control ratio mismatch")
        require(
            close(control["matched_structural_s_bpw"], -0.5 * math.log2(ratio)),
            f"{prefix}: matched s mismatch",
        )

    eligible = [
        row for row in rows
        if row["cold_read_strictly_below_2x"] and 2.15 <= row["physical_rate_bpw"] <= 2.5
    ]
    free = min(
        (row for row in eligible if row["accounting"] == "free_source_leaky_side"),
        key=lambda row: (row["F"], -row["matched_gaussian_control"]["matched_structural_s_bpw"], row["model_id"]),
    )
    charged = min(
        (row for row in eligible if row["accounting"] == "charged"),
        key=lambda row: (row["F"], -row["matched_gaussian_control"]["matched_structural_s_bpw"], row["model_id"]),
    )
    require(
        canonical_bytes(compact(free)) == canonical_bytes(result["decision"]["best_free_source_leaky"]),
        "best free row mismatch",
    )
    require(
        canonical_bytes(compact(charged)) == canonical_bytes(result["decision"]["best_physically_charged"]),
        "best charged row mismatch",
    )
    passes = bool(
        charged["F"] <= TARGET_F
        and charged["matched_gaussian_control"]["matched_structural_s_bpw"] >= TARGET_S
    )
    require(result["decision"]["passes_hard_gate"] is passes, "decision pass mismatch")
    expected_status = "PROMOTE_TO_OPERATIONAL_CODEC" if passes else "EARLY_KILL_VARIANCE_FIELD_BRANCH"
    require(result["status"] == expected_status, "status mismatch")

    if args.source_lock is not None:
        lock_payload = args.source_lock.read_bytes()
        require(
            hashlib.sha256(lock_payload).hexdigest()
            == result["provenance"]["source_lock_file_sha256"],
            "source lock file hash mismatch",
        )
        lock = json.loads(lock_payload)
        lock_internal = verify_internal(lock, "lock_sha256")
        require(
            lock_internal == result["provenance"]["source_lock_internal_sha256"],
            "source lock internal hash mismatch",
        )

    if args.source_root is not None:
        require(args.source_lock is not None, "--source-root requires --source-lock")
        lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
        metadata = {int(row["matrix_ordinal"]): row for row in lock["matrices"]}
        for receipt in result["provenance"]["sources"]:
            meta = metadata[int(receipt["matrix_ordinal"])]
            path = args.source_root / meta["output_relpath"]
            require(path.stat().st_size == receipt["bytes"], f"{path}: byte size mismatch")
            require(sha256_file(path) == receipt["sha256"], f"{path}: source hash mismatch")

    print(
        json.dumps(
            {
                "verified": True,
                "status": result["status"],
                "result_lock_sha256": result_lock,
                "best_free_F": free["F"],
                "best_free_s_bpw": free["s_bpw"],
                "best_charged_F": charged["F"],
                "best_charged_s_bpw": charged["s_bpw"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
