#!/usr/bin/env python3
"""Independent arithmetic, seal, ledger, and optional source verifier.

This does not rerun LB-ADMM.  It verifies the frozen observation-level output,
re-aggregates every reported MSE, rebuilds the physical bit ledger, and can
rehash the pinned plan/header/source files on the RunPod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "qwen-nanoquant-discrete-tile-pilot-v1"


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


def close(a: float, b: float, *, tolerance: float = 3e-13) -> None:
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=2e-14):
        raise ValueError(f"numeric mismatch: {a} != {b}")


def ledger(tile_rows: int, tile_cols: int, rate: float, rank_scale_bits: int) -> dict[str, Any]:
    values = tile_rows * tile_cols
    axis = tile_rows + tile_cols
    rank = max(int(math.floor((rate * values - 16 * axis) / (axis + rank_scale_bits))), 1)
    useful = rank * (axis + rank_scale_bits) + 16 * axis
    physical = 8 * math.ceil(rate * values / 8.0)
    return {
        "requested_rate_bpw": rate,
        "physical_rate_bpw": physical / values,
        "tile_shape": [tile_rows, tile_cols],
        "rank": rank,
        "rank_exceeds_min_dimension": rank > min(tile_rows, tile_cols),
        "binary_factor_bits": rank * axis,
        "rank_scale_bits": rank * rank_scale_bits,
        "rank_scale_storage_bits_each": rank_scale_bits,
        "fp16_scale_bits": 16 * axis,
        "useful_bits": useful,
        "physical_bits": physical,
        "padding_bits": physical - useful,
        "useful_bpw": useful / values,
        "shared_table_bits": 0,
        "cold_read_amplification": 1.0,
    }


def verify_plan(plan_path: Path, result: dict[str, Any], rehash_sources: bool) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    clean = dict(plan)
    claimed_lock = clean.pop("lock_sha256", None)
    actual_lock = hashlib.sha256(canonical_json(clean)).hexdigest()
    if claimed_lock != actual_lock:
        raise ValueError("plan canonical lock mismatch")
    provenance = result["provenance"]
    if actual_lock != provenance["plan_lock_sha256"]:
        raise ValueError("result/plan lock mismatch")
    if sha256_file(plan_path) != provenance["plan_file_sha256"]:
        raise ValueError("result/plan file hash mismatch")

    header_row = plan["assets"]["header.bin"]
    header_path = (plan_path.parent / header_row["relpath"]).resolve(strict=True)
    if sha256_file(header_path) != header_row["sha256"]:
        raise ValueError("header hash mismatch")
    source_root = Path(plan["source_root"]).resolve(strict=True)
    result_sources = {row["tensor"]: row for row in provenance["sources"]}
    source_receipts: list[dict[str, Any]] = []
    for row in plan["sources"]:
        receipt = result_sources.get(row["tensor"])
        if receipt is None or receipt["sha256"] != row["source_bf16_sha256"]:
            raise ValueError("source manifest mismatch")
        item: dict[str, Any] = {
            "tensor": row["tensor"],
            "declared_sha256": row["source_bf16_sha256"],
            "rehash_performed": rehash_sources,
        }
        if rehash_sources:
            path = (source_root / row["source_relpath"]).resolve(strict=True)
            if source_root not in path.parents:
                raise ValueError("source escaped root")
            observed = sha256_file(path)
            if observed != row["source_bf16_sha256"]:
                raise ValueError("source hash mismatch")
            item["observed_sha256"] = observed
            item["bytes"] = path.stat().st_size
        source_receipts.append(item)
    return {
        "plan_file_sha256": sha256_file(plan_path),
        "plan_lock_sha256": actual_lock,
        "header_sha256": header_row["sha256"],
        "source_count": len(source_receipts),
        "all_sources_rehashed": rehash_sources,
        "sources": source_receipts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--algorithm", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--rehash-sources", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_path = args.result.resolve(strict=True)
    algorithm_path = args.algorithm.resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA:
        raise ValueError("unexpected schema")
    clean = dict(result)
    claimed_lock = clean.pop("result_lock_sha256", None)
    actual_lock = hashlib.sha256(canonical_json(clean)).hexdigest()
    if claimed_lock != actual_lock:
        raise ValueError("result canonical lock mismatch")
    algorithm_hash = sha256_file(algorithm_path)
    if algorithm_hash != result["provenance"]["algorithm_sha256"]:
        raise ValueError("algorithm hash mismatch")

    protocol = result["protocol"]
    tile_rows, tile_cols = map(int, protocol["tile_shape"])
    rank_scale_bits = int(protocol["rank_scale_storage_bits_each"])
    expected_ledgers = [
        ledger(tile_rows, tile_cols, float(row["requested_rate_bpw"]), rank_scale_bits)
        for row in result["ledgers"]
    ]
    if canonical_json(expected_ledgers) != canonical_json(result["ledgers"]):
        raise ValueError("physical ledger mismatch")

    observations = result["observations"]
    expected_observations = (
        len(expected_ledgers)
        * int(protocol["matrix_count"])
        * int(protocol["tiles_per_matrix"])
    )
    if len(observations) != expected_observations:
        raise ValueError("observation count mismatch")
    for observation in observations:
        source = observation["source"]
        gaussian = observation["gaussian"]
        close(float(source["energy_fp64"]), float(gaussian["energy_fp64"]))
        for fit in (source, gaussian):
            close(
                float(fit["fp16_scale_relative_mse"]),
                float(fit["fp16_scale_sse"]) / float(fit["energy_fp64"]),
            )
            close(
                float(fit["fp64_scale_relative_mse"]),
                float(fit["fp64_scale_sse"]) / float(fit["energy_fp64"]),
            )

    aggregate_receipts: list[dict[str, Any]] = []
    for row in result["aggregate"]:
        rate = float(row["ledger"]["requested_rate_bpw"])
        selected = [item for item in observations if float(item["requested_rate_bpw"]) == rate]
        source_sse = [0.0] * 6
        gaussian_sse = [0.0] * 6
        energy = [0.0] * 6
        for observation in selected:
            expert = int(observation["expert"])
            source_sse[expert] += float(observation["source"]["fp16_scale_sse"])
            gaussian_sse[expert] += float(observation["gaussian"]["fp16_scale_sse"])
            energy[expert] += float(observation["source"]["energy_fp64"])
        for actual, expected in zip(source_sse, row["source_sse_by_expert"]):
            close(actual, float(expected))
        for actual, expected in zip(gaussian_sse, row["gaussian_sse_by_expert"]):
            close(actual, float(expected))
        for actual, expected in zip(energy, row["energy_by_expert"]):
            close(actual, float(expected))
        source_d = math.fsum(source_sse) / math.fsum(energy)
        gaussian_d = math.fsum(gaussian_sse) / math.fsum(energy)
        ratio = source_d / gaussian_d
        s_bpw = -0.5 * math.log2(ratio)
        physical_rate = float(row["ledger"]["physical_rate_bpw"])
        codec_f = source_d * 2.0 ** (2.0 * physical_rate)
        close(source_d, float(row["source_relative_mse"]))
        close(gaussian_d, float(row["gaussian_relative_mse"]))
        close(ratio, float(row["source_over_matched_gaussian"]))
        close(s_bpw, float(row["structural_advantage_bpw"]))
        close(codec_f, float(row["source_F_equals_D_times_2pow2R"]))
        aggregate_receipts.append(
            {
                "physical_rate_bpw": physical_rate,
                "source_relative_mse": source_d,
                "matched_gaussian_relative_mse": gaussian_d,
                "source_over_matched_gaussian": ratio,
                "structural_advantage_bpw": s_bpw,
                "source_codec_F": codec_f,
                "cold_read_amplification": float(row["ledger"]["cold_read_amplification"]),
            }
        )

    plan_receipt = None
    if args.plan is not None:
        plan_receipt = verify_plan(
            args.plan.resolve(strict=True), result, bool(args.rehash_sources)
        )
    elif args.rehash_sources:
        raise ValueError("--rehash-sources requires --plan")

    receipt: dict[str, Any] = {
        "schema": "qwen-nanoquant-discrete-verification-receipt-v1",
        "status": "PASS",
        "result_file_sha256": sha256_file(result_path),
        "result_lock_sha256": actual_lock,
        "algorithm_file_sha256": algorithm_hash,
        "observation_count": len(observations),
        "aggregate": aggregate_receipts,
        "plan": plan_receipt,
        "checks": [
            "canonical result lock",
            "algorithm hash binding",
            "physical factor/scale/padding ledger",
            "paired source/Gaussian energy identity per observation",
            "observation-level SSE/MSE identities",
            "full pooled aggregate recomputation",
            "D, Gaussian ratio, s, and codec-F identities",
            "optional plan/header/source rehash",
        ],
    }
    receipt["receipt_lock_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
