#!/usr/bin/env python3
"""Fail-closed verifier and merger for the split scalar/joint BA runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TARGET = -0.5 * math.log2(0.8)
NUMERICAL_ALLOWANCE = 0.005
EXPECTED_WEIGHTS = 28_311_552
EXPECTED_SOURCES = 18
MIN_PHYSICAL_RATE = 2.15


def canonical_bytes(value: object) -> bytes:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checked(path: Path, expected_script: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "qwen-nonparametric-ba-oracle-v1":
        raise ValueError(f"wrong schema: {path}")
    if value.get("cpu_only") is not True or value.get("gpu_imports_or_subprocesses") is not False:
        raise ValueError(f"CPU/GPU declaration failed: {path}")
    expected_payload = value.pop("result_payload_sha256", None)
    actual_payload = hashlib.sha256(canonical_bytes(value)).hexdigest()
    value["result_payload_sha256"] = expected_payload
    if expected_payload != actual_payload:
        raise ValueError(f"internal result seal mismatch: {path}")
    if value["script"]["sha256"] != expected_script:
        raise ValueError(f"executed script hash mismatch: {path}")
    sources = value["provenance"]["sources"]
    if len(sources) != EXPECTED_SOURCES:
        raise ValueError("source count changed")
    hashes = [row["sha256"] for row in sources]
    tensors = [row["tensor"] for row in sources]
    if len(set(hashes)) != EXPECTED_SOURCES or len(set(tensors)) != EXPECTED_SOURCES:
        raise ValueError("sources are not 18 unique hash/tensor bindings")
    return value


def close(left: float, right: float, tolerance: float = 2e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_hash = sha256_file(args.script)
    scalar = load_checked(args.scalar, script_hash)
    joint = load_checked(args.joint, script_hash)
    if scalar["provenance"] != joint["provenance"]:
        raise ValueError("scalar and joint runs do not bind the same sources/plan/header")
    branches = scalar["branches"] + joint["branches"]
    keys = {(row["representation"], int(row["dimension"])) for row in branches}
    expected_keys = {(representation, dimension) for representation in ("raw", "xklt") for dimension in (1, 2, 4)}
    if keys != expected_keys or len(branches) != len(expected_keys):
        raise ValueError(f"branch coverage mismatch: {keys}")
    candidates = []
    single_fold_gains = []
    max_read = 0.0
    max_common_bytes = 0
    for branch in branches:
        ledger = branch["aggregate"]["side_information_ledger"]
        read = float(ledger["read_amplification"]["conservative_total"])
        if not ledger["read_amplification"]["strictly_below_2x"] or not read < 2.0:
            raise ValueError("read-amplification gate failed")
        max_read = max(max_read, read)
        max_common_bytes = max(max_common_bytes, int(ledger["total_bytes_rounded"]))
        side = float(ledger["bpw"])
        for rate, row in branch["aggregate"]["rates"].items():
            free_gain = float(row["free_side_calibrated_rate_advantage_bpw"])
            charged_gain = float(row["charged_rate_advantage_bpw"])
            charged_f = float(row["charged_F_prediction"])
            if not close(charged_gain, free_gain - side):
                raise ValueError("side-rate subtraction mismatch")
            if not close(charged_f, math.pow(2.0, -2.0 * charged_gain)):
                raise ValueError("gain/F identity mismatch")
            expert_gains = np.asarray(row["heldout_expert_calibrated_gain_bpw"], dtype=np.float64)
            if expert_gains.shape != (6,) or not np.isfinite(expert_gains).all():
                raise ValueError("held-out expert gain geometry changed")
            single_fold_gains.extend(expert_gains.tolist())
            per_rate_se = float(np.std(expert_gains, ddof=1) / math.sqrt(6.0))
            candidates.append(
                {
                    "representation": branch["representation"],
                    "dimension": int(branch["dimension"]),
                    "physical_rate_bpw": float(rate),
                    "free_side_gain_bpw": free_gain,
                    "free_side_F": math.pow(2.0, -2.0 * free_gain),
                    "charged_gain_bpw": charged_gain,
                    "charged_F": charged_f,
                    "per_rate_six_fold_standard_error_bpw": per_rate_se,
                    "two_se_plus_numerical_upper_gain_bpw": free_gain + 2.0 * per_rate_se + NUMERICAL_ALLOWANCE,
                }
            )
    best_point = max(candidates, key=lambda row: row["free_side_gain_bpw"])
    best_two_se = max(candidates, key=lambda row: row["two_se_plus_numerical_upper_gain_bpw"])
    max_fold_gain = max(single_fold_gains)
    max_fold_plus_allowance = max_fold_gain + NUMERICAL_ALLOWANCE
    strongest_optimistic_upper = max(
        best_two_se["two_se_plus_numerical_upper_gain_bpw"],
        max_fold_plus_allowance,
    )
    minimum_rate_expert_share_bytes = EXPECTED_WEIGHTS * MIN_PHYSICAL_RATE / 8.0 / 6.0
    minimum_rate_worst_read = 10.0 / 9.0 + max_common_bytes / minimum_rate_expert_share_bytes
    summary: dict[str, Any] = {
        "schema": "qwen-nonparametric-ba-summary-v1",
        "decision": {
            "required_gain_bpw": TARGET,
            "required_F": 0.8,
            "best_panel_aggregate_free_side_point": best_point,
            "best_corrected_two_se_plus_numerical_upper": best_two_se,
            "largest_single_heldout_fold_gain_bpw": max_fold_gain,
            "largest_single_fold_plus_numerical_allowance_bpw": max_fold_plus_allowance,
            "largest_single_fold_plus_allowance_F": math.pow(2.0, -2.0 * max_fold_plus_allowance),
            "strongest_optimistic_upper_gain_bpw": strongest_optimistic_upper,
            "strongest_optimistic_upper_F": math.pow(2.0, -2.0 * strongest_optimistic_upper),
            "remaining_gain_shortfall_bpw": TARGET - strongest_optimistic_upper,
            "hard_kill": strongest_optimistic_upper < TARGET,
        },
        "coverage": {
            "representations": ["raw", "xklt"],
            "dimensions": [1, 2, 4],
            "outer_folds": 6,
            "whole_matrices_held_out_per_fold": 3,
            "source_weights": EXPECTED_WEIGHTS,
            "source_count": EXPECTED_SOURCES,
            "source_bindings": scalar["provenance"]["sources"],
            "plan_file_sha256": scalar["provenance"]["plan_file_sha256"],
            "plan_lock_sha256": scalar["provenance"]["plan_lock_sha256"],
            "header_sha256": scalar["provenance"]["header_sha256"],
        },
        "read_accounting": {
            "maximum_conservative_amplification_at_2p5_bpw": max_read,
            "maximum_common_model_bytes": max_common_bytes,
            "worst_permitted_rate_bpw": MIN_PHYSICAL_RATE,
            "maximum_conservative_amplification_at_2p15_bpw": minimum_rate_worst_read,
            "below_2x_over_permitted_rate_interval": minimum_rate_worst_read < 2.0,
            "note": "Base 10/9 expert-affine coefficient traffic plus cold common model-table bytes; 2.15 bpw is worst because its expert payload share is smallest.",
        },
        "evidence": {
            "executed_script": {"path": str(args.script), "sha256": script_hash},
            "scalar_result": {"path": str(args.scalar), "sha256": sha256_file(args.scalar)},
            "joint_result": {"path": str(args.joint), "sha256": sha256_file(args.joint)},
            "scalar_internal_payload_sha256": scalar["result_payload_sha256"],
            "joint_internal_payload_sha256": joint["result_payload_sha256"],
        },
        "claim_boundary": (
            "This rejects stationary scalar and adjacent 2-D/4-D nonparametric test channels under "
            "the tested whole-matrix normalization and raw/XKLT coordinates. It is not a universal "
            "converse for long-range deterministic or semantic structure."
        ),
    }
    summary["summary_payload_sha256"] = hashlib.sha256(canonical_bytes(summary)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
