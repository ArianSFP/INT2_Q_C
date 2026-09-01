#!/usr/bin/env python3
"""Pure-stdlib threshold replay for PSNO-v1.

This script reads only the prior sealed JSON result.  It never opens a tensor,
imports NumPy/CuPy/Torch, calls a GPU, or writes an artifact.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any


N = 768
VALUES = 768 * 2048
M = N * (N + 1) // 2
RATE = 2.5
TARGET_F = 0.8
EXPECTED_PARENT_SHA256 = "e4fecac5f676d84739972bbf0e04467027aeae1356e62e1dc3cd2b84bff67026"
DEFAULT_K = (0, 64, 256, 1024, 4096, 8192, 16384, 24576, 32768, 40960, 45056, 48018, 48019)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@functools.lru_cache(maxsize=None)
def exact_support_bits(k: int) -> int:
    if not 0 <= k <= M:
        raise ValueError("k outside symmetric support universe")
    combinations = math.comb(M, k)
    return (combinations - 1).bit_length()


def approximate_support_bits(k: int) -> int:
    if k in (0, M):
        return 0
    value = (
        math.lgamma(M + 1) - math.lgamma(k + 1) - math.lgamma(M - k + 1)
    ) / math.log(2.0)
    return math.ceil(value)


def waterfill(dimensions: list[float], energies: list[float], rate: float) -> dict[str, Any]:
    rows = [(d, e) for d, e in zip(dimensions, energies, strict=True) if d > 0.0 and e > 0.0]
    if rate <= 0.0 or not rows:
        return {"valid": False}
    rows.sort(key=lambda row: math.log2(row[1] / row[0]), reverse=True)
    cumulative_d = 0.0
    cumulative_d_logv = 0.0
    active = len(rows)
    log_level = 0.0
    for index, (dimension, energy) in enumerate(rows, start=1):
        logv = math.log2(energy / dimension)
        cumulative_d += dimension
        cumulative_d_logv += dimension * logv
        candidate = (cumulative_d_logv - 2.0 * rate) / cumulative_d
        next_logv = (
            math.log2(rows[index][1] / rows[index][0]) if index < len(rows) else -math.inf
        )
        if candidate <= logv + 2e-14 and candidate >= next_logv - 2e-14:
            active = index
            log_level = candidate
            break
    level = 2.0**log_level
    distortion = 0.0
    used = 0.0
    for dimension, energy in rows:
        logv = math.log2(energy / dimension)
        if logv > log_level:
            distortion += dimension * level
            used += dimension * 0.5 * (logv - log_level)
        else:
            distortion += energy
    if not math.isclose(used, rate, rel_tol=4e-10, abs_tol=4e-10):
        raise AssertionError(("waterfill rate", used, rate, active))
    return {
        "valid": True,
        "distortion": distortion,
        "payload_rate_bpw": rate,
        "active_components": active,
    }


def score(
    records: list[dict[str, Any]],
    base_side_bpw: float,
    k: int,
    captured_fraction: float,
    value_bits: int,
    *,
    exact_support: bool,
) -> dict[str, Any]:
    if not 0.0 <= captured_fraction <= 1.0:
        raise ValueError("capture outside [0,1]")
    support = exact_support_bits(k) if exact_support else approximate_support_bits(k)
    field_bits = support + value_bits * k
    side = base_side_bpw + field_bits / VALUES
    total_energy = math.fsum(float(row["source_energy"]) for row in records)
    dimensions: list[float] = []
    energies: list[float] = []
    for row in records:
        dimensions.append(float(row["model_dof"]) / (18 * VALUES))
        energies.append(float(row["model_energy"]) / total_energy)
        residual = float(row["normal_energy"]) * (1.0 - captured_fraction)
        if residual > 0.0:
            removed = min(k, int(row["normal_dof"]) - 1)
            dimensions.append(max(1, int(row["normal_dof"]) - removed) / (18 * VALUES))
            energies.append(residual / total_energy)
    payload = RATE - side
    result = waterfill(dimensions, energies, payload)
    if not result["valid"]:
        return {"valid": False, "side_bpw": side, "field_bits_per_matrix": field_bits}
    f_value = float(result["distortion"]) * 2.0 ** (2.0 * RATE)
    return {
        **result,
        "valid": True,
        "F": f_value,
        "side_bpw": side,
        "support_bits_per_matrix": support,
        "value_bits_per_matrix": value_bits * k,
        "field_bits_per_matrix": field_bits,
    }


def required_capture(
    records: list[dict[str, Any]], base_side_bpw: float, k: int, value_bits: int
) -> float | None:
    full = score(records, base_side_bpw, k, 1.0, value_bits, exact_support=True)
    if not full["valid"] or float(full["F"]) > TARGET_F:
        return None
    empty = score(records, base_side_bpw, k, 0.0, value_bits, exact_support=True)
    if float(empty["F"]) <= TARGET_F:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2.0
        candidate = score(
            records, base_side_bpw, k, middle, value_bits, exact_support=True
        )
        if float(candidate["F"]) <= TARGET_F:
            high = middle
        else:
            low = middle
    return high


def gaussian_topk_capture(k: int) -> float:
    if k <= 0:
        return 0.0
    if k >= M:
        return 1.0
    probability = k / M
    threshold = NormalDist().inv_cdf(1.0 - probability / 2.0)
    density = math.exp(-0.5 * threshold * threshold) / math.sqrt(2.0 * math.pi)
    return probability + 2.0 * threshold * density


def last_sparse_full_capture_survivor(
    records: list[dict[str, Any]], base_side_bpw: float, value_bits: int
) -> int:
    # Only the sparse half of the support universe is in scope.  The
    # approximate search locates the transition; exact integer binomials then
    # certify the boundary.
    low, high = 0, M // 2
    while low + 1 < high:
        middle = (low + high) // 2
        candidate = score(
            records, base_side_bpw, middle, 1.0, value_bits, exact_support=False
        )
        if candidate["valid"] and float(candidate["F"]) <= TARGET_F:
            low = middle
        else:
            high = middle
    start = max(0, low - 8)
    stop = min(M // 2, high + 8)
    exact_last = start
    for k in range(start, stop + 1):
        candidate = score(records, base_side_bpw, k, 1.0, value_bits, exact_support=True)
        if candidate["valid"] and float(candidate["F"]) <= TARGET_F:
            exact_last = k
    return exact_last


def build_report(parent: Path) -> dict[str, Any]:
    raw = parent.read_bytes()
    observed = sha256(raw)
    if observed != EXPECTED_PARENT_SHA256:
        raise AssertionError(("parent result hash", observed))
    report = json.loads(raw.decode("utf-8"))
    if report["schema"] != "qwen_polar_normal_predictor_oracle_v1":
        raise AssertionError("parent schema")
    records = report["source_normal_records"]
    if len(records) != 18:
        raise AssertionError("normal record count")
    base_side = float(report["scope"]["base_polar_explicit_side_bpw"])
    baseline = score(records, base_side, 0, 0.0, 0, exact_support=True)
    if not math.isclose(float(baseline["F"]), 0.9520339564260487, rel_tol=0.0, abs_tol=3e-15):
        raise AssertionError(("baseline", baseline))

    rows: list[dict[str, Any]] = []
    for k in DEFAULT_K:
        support = exact_support_bits(k)
        gaussian_capture = gaussian_topk_capture(k)
        row: dict[str, Any] = {
            "k": k,
            "support_bits_per_matrix": support,
            "support_bpw": support / VALUES,
            "iid_gaussian_capture_fraction": gaussian_capture,
            "value_ledgers": {},
        }
        for value_bits in (0, 1, 16):
            needed = required_capture(records, base_side, k, value_bits)
            gaussian_score = score(
                records,
                base_side,
                k,
                gaussian_capture,
                value_bits,
                exact_support=True,
            )
            full_score = score(
                records, base_side, k, 1.0, value_bits, exact_support=True
            )
            row["value_ledgers"][str(value_bits)] = {
                "required_capture_fraction": needed,
                "iid_gaussian_F": gaussian_score.get("F"),
                "full_capture_F": full_score.get("F"),
                "total_side_bpw": full_score["side_bpw"],
            }
        rows.append(row)

    boundaries: dict[str, Any] = {}
    for value_bits in (0, 1, 16):
        last = last_sparse_full_capture_survivor(records, base_side, value_bits)
        current = score(records, base_side, last, 1.0, value_bits, exact_support=True)
        following = score(
            records, base_side, last + 1, 1.0, value_bits, exact_support=True
        )
        boundaries[str(value_bits)] = {
            "last_viable_k": last,
            "last_support_bits_per_matrix": exact_support_bits(last),
            "last_total_side_bpw": current["side_bpw"],
            "last_full_capture_F": current["F"],
            "next_k": last + 1,
            "next_support_bits_per_matrix": exact_support_bits(last + 1),
            "next_total_side_bpw": following["side_bpw"],
            "next_full_capture_F": following["F"],
        }
    if boundaries["0"]["last_viable_k"] != 48018:
        raise AssertionError(("free-value boundary", boundaries["0"]))

    return {
        "schema": "polar_sparse_normal_source_only_thresholds_v1",
        "parent_result_sha256": observed,
        "tensor_files_opened": 0,
        "third_party_imports": 0,
        "cupy_imports": 0,
        "gpu_operations": 0,
        "geometry": {
            "n": N,
            "unique_symmetric_coordinates": M,
            "values_per_matrix": VALUES,
        },
        "base_polar_side_bpw": base_side,
        "baseline_F": baseline["F"],
        "boundaries_by_exact_value_bits": boundaries,
        "equal_k_rows": rows,
        "interpretation": (
            "Rows are exact equal-k source-only thresholds, not a Qwen top-k result. "
            "The future adaptive gate uses per-matrix prefix curves and a Lagrangian dual."
        ),
        "authorization": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-result",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "polar_normal_predictor/result.json",
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.parent_result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
