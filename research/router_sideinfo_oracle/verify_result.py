#!/usr/bin/env python3
"""Dependency-free integrity/formula verifier for the router-side oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(observed: float, expected: float, tolerance: float = 2e-11) -> None:
    if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"formula mismatch: {observed} != {expected}")


def reverse_waterfill(dim: float, energy: float, rate: float) -> tuple[float, float, float]:
    dims = (dim, 1.0 - dim)
    energies = (energy, 1.0 - energy)
    variances = tuple(e / d for e, d in zip(energies, dims))
    lo = math.log2(min(variances)) - 2.0 * rate / min(dims) - 8.0
    hi = math.log2(max(variances))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        log_terms = [max(0.0, 0.5 * (math.log2(v) - mid)) for v in variances]
        used = sum(d * term for d, term in zip(dims, log_terms))
        if used > rate:
            lo = mid
        else:
            hi = mid
    log_water = 0.5 * (lo + hi)
    distortion = sum(
        d * (2.0 ** min(math.log2(v), log_water))
        for d, v in zip(dims, variances)
    )
    factor = distortion * (2.0 ** (2.0 * rate))
    return distortion, factor, -0.5 * math.log2(factor)


def verify_plan(plan_path: Path, expected_lock: str) -> dict[str, Any]:
    plan = json.loads(plan_path.read_bytes())
    clean = dict(plan)
    lock = clean.pop("lock_sha256", None)
    observed = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    if lock != expected_lock or observed != expected_lock:
        raise ValueError("plan seal mismatch")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    result_path = args.result.resolve(strict=True)
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(f"refusing to replace receipt: {receipt_path}")
    result = json.loads(result_path.read_bytes())
    if result.get("schema") != "qwen_router_sideinfo_oracle_v1":
        raise ValueError("unexpected result schema")
    if result.get("decision") != "hard_kill":
        raise ValueError("unexpected decision")
    if result["target"] != {"F_max": 0.8, "rates_bpw": [2.15, 2.3, 2.5]}:
        raise ValueError("target changed")

    plan_path = args.plan_dir.resolve(strict=True) / "plan.lock.json"
    if sha256_file(plan_path) != result["plan"]["sha256"]:
        raise ValueError("plan byte hash mismatch")
    plan = verify_plan(plan_path, result["plan"]["lock_sha256"])
    source_root = Path(plan["source_root"]).resolve(strict=True)
    by_tensor = {row["tensor"]: row for row in result["sources"]}
    if len(by_tensor) != 18 or len(plan["sources"]) != 18:
        raise ValueError("source coverage mismatch")
    for source in plan["sources"]:
        result_row = by_tensor.get(source["tensor"])
        if result_row is None or result_row["sha256"] != source["source_bf16_sha256"]:
            raise ValueError(f"source binding mismatch: {source['tensor']}")
        if sha256_file(source_root / source["source_relpath"]) != result_row["sha256"]:
            raise ValueError(f"source byte mismatch: {source['tensor']}")

    manifests = sorted(args.router_dir.resolve(strict=True).glob("*.manifest.json"))
    if len(manifests) != 6 or len(result["router_bindings"]) != 6:
        raise ValueError("router coverage mismatch")
    verified_routers = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_bytes())
        layer = str(manifest["tensor"].split(".")[2])
        binding = result["router_bindings"].get(layer)
        if binding is None or binding["tensor"] != manifest["tensor"]:
            raise ValueError(f"router binding mismatch: {manifest_path}")
        if sha256_file(manifest_path) != binding["manifest_sha256"]:
            raise ValueError(f"router manifest mismatch: {manifest_path}")
        data_path = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".bin"))
        if sha256_file(data_path) != binding["sha256"] or binding["sha256"] != manifest["sha256"]:
            raise ValueError(f"router byte mismatch: {data_path}")
        verified_routers += 1

    curves = result["curves"]
    if len(curves) != 32 or len(result["matrix_projections"]) != 144:
        raise ValueError("curve/projection coverage mismatch")
    best_row: dict[str, Any] | None = None
    best_factor = math.inf
    checked_rates = 0
    for row in curves:
        dim = float(row["dimension_fraction"])
        energy = float(row["captured_energy_fraction"])
        close(row["energy_enrichment"], energy / dim)
        row_best_factor = math.inf
        row_best_s = -math.inf
        for rate_row in row["ideal_rates"]:
            distortion, factor, advantage = reverse_waterfill(dim, energy, float(rate_row["rate_bpw"]))
            close(rate_row["relative_mse_oracle"], distortion)
            close(rate_row["F"], factor)
            close(rate_row["s_bpw"], advantage)
            row_best_factor = min(row_best_factor, factor)
            row_best_s = max(row_best_s, advantage)
            checked_rates += 1
        close(row["best_F"], row_best_factor)
        close(row["best_s_bpw"], row_best_s)
        if row_best_factor < best_factor:
            best_factor = row_best_factor
            best_row = row
    if best_row is None:
        raise ValueError("no curves")
    for key in ("mode", "k"):
        if result["best"][key] != best_row[key]:
            raise ValueError("best-row selection mismatch")
    close(result["best"]["best_F"], best_factor)
    if best_factor <= result["target"]["F_max"]:
        raise ValueError("hard-kill decision inconsistent with factor")

    receipt = {
        "schema": "qwen_router_sideinfo_verification_v1",
        "status": "PASS",
        "result_sha256": sha256_file(result_path),
        "plan_sha256": sha256_file(plan_path),
        "verified_sources": len(by_tensor),
        "verified_routers": verified_routers,
        "verified_curves": len(curves),
        "verified_rate_cells": checked_rates,
        "best_F": best_factor,
        "target_F_max": result["target"]["F_max"],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
