#!/usr/bin/env python3
"""Source-free structural verifier for the Qwen hidden-ICA oracle artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK = "99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d"
WEIGHTS = 28_311_552
REQUIRED = -0.5 * math.log2(0.8)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_seal(payload: dict, label: str) -> None:
    clean = dict(payload)
    expected = clean.pop("result_seal_sha256")
    require(hashlib.sha256(canonical(clean)).hexdigest() == expected, f"{label} result seal")


def close(left: float, right: float, label: str, tolerance: float = 2e-12) -> None:
    require(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance), label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-sources",
        action="store_true",
        help="also stream and hash all 18 BF16 files at plan.source_root",
    )
    args = parser.parse_args()
    plan_path = ROOT / "plan.lock.json"
    header_path = ROOT / "header.bin"
    free_path = ROOT / "qwen_ica_free_side_screen.json"
    full_path = ROOT / "qwen_ica_crossfit_confirmation.json"
    engine_path = ROOT / "ica_projection_oracle.py"
    free_script_path = ROOT / "free_side_screen.py"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_clean = dict(plan)
    plan_lock = plan_clean.pop("lock_sha256")
    require(hashlib.sha256(canonical(plan_clean)).hexdigest() == plan_lock == LOCK, "plan lock")
    require(plan["coverage"]["weights"] == WEIGHTS, "plan weights")
    require(len(plan["sources"]) == 18, "source count")
    require(len({row["source_bf16_sha256"] for row in plan["sources"]}) == 18, "source hash uniqueness")
    require(all(row["bytes"] == 3_145_728 for row in plan["sources"]), "source sizes")
    require(sha(header_path) == plan["assets"]["header.bin"]["sha256"], "header hash")
    require(header_path.read_bytes()[:8] == b"PLRLOC3\0", "header magic")
    physical_sources_checked = False
    if args.check_sources:
        source_root = Path(plan["source_root"]).resolve(strict=True)
        for row in plan["sources"]:
            source_path = (source_root / row["source_relpath"]).resolve(strict=True)
            require(source_root in source_path.parents, "source escaped root")
            require(source_path.stat().st_size == row["bytes"], "physical source size")
            require(sha(source_path) == row["source_bf16_sha256"], "physical source hash")
        physical_sources_checked = True

    free = json.loads(free_path.read_text(encoding="utf-8"))
    full = json.loads(full_path.read_text(encoding="utf-8"))
    verify_seal(free, "free")
    verify_seal(full, "full")
    for payload, label in ((free, "free"), (full, "full")):
        close(payload["required_gain_bpw"], REQUIRED, f"{label} required")
        provenance = payload["provenance"]
        require(provenance["plan_lock_sha256"] == LOCK, f"{label} plan binding")
        require(provenance["plan_file_sha256"] == sha(plan_path), f"{label} plan file")
        require(provenance["header"]["sha256"] == sha(header_path), f"{label} header binding")
        observed = [(row["matrix_ordinal"], row["sha256"]) for row in provenance["sources"]]
        expected = [(row["matrix_ordinal"], row["source_bf16_sha256"]) for row in plan["sources"]]
        require(observed == expected, f"{label} source binding")
        require(payload["configuration"]["gpu_used"] is False, f"{label} CPU-only marker")

    require(free["runtime"]["script_sha256"] == sha(free_script_path), "free script hash")
    require(free["runtime"]["engine_script_sha256"] == sha(engine_path), "free engine hash")
    require(full["runtime"]["script_sha256"] == sha(engine_path), "full script hash")
    free_keys = {(row["representation"], row["dimension"]) for row in free["results"]}
    require(free_keys == {(rep, dim) for rep in ("raw", "xklt") for dim in (8, 16, 32, 64)}, "free coverage")
    full_keys = {(row["representation"], row["dimension"]) for row in full["results"]}
    require(full_keys == {(rep, dim) for rep in ("raw", "xklt") for dim in (16, 64)}, "crossfit coverage")

    free_best = max(
        row["selected"]["optimistic_shape_plus_variance_bpw"]
        for branch in free["results"]
        for row in [branch]
    )
    close(free_best, free["summary"]["best_optimistic_gain_bpw"], "free maximum")
    close(free_best, full["summary"]["best_free_side_shape_plus_variance_bpw"], "free/full agreement")
    gains = []
    for branch in full["results"]:
        require(len(branch["crossfit"]["folds"]) == 6, "six folds")
        ledger = branch["side_and_read_ledger"]
        close(ledger["side_bpw"], ledger["total_common_bytes"] * 8.0 / WEIGHTS, "side bpw")
        for rate in (2.15, 2.5):
            rate_ledger = ledger["rates"][str(rate)]
            close(
                rate_ledger["cold_read_amplification"],
                rate_ledger["cold_read_bytes"] / rate_ledger["ideal_expert_bytes"],
                "read amp identity",
            )
            require(rate_ledger["cold_read_amplification"] < 2.0, "read amp gate")
        for row in branch["crossfit"]["rate_matched_results"]:
            gain = -0.5 * math.log2(row["real"]["mse"] / row["matched_gaussian"]["mse"])
            close(gain, row["rate_gain_bpw"], "rate gain")
            close(2.0 ** (-2.0 * gain), row["distortion_factor"], "distortion factor")
            gains.append(gain)
    best_crossfit = max(gains)
    close(best_crossfit, full["summary"]["best_crossfit_rate_matched_gain_bpw"], "crossfit maximum")
    optimistic = max(free_best, best_crossfit) + full["summary"]["fixed_numerical_allowance_bpw"]
    close(optimistic, full["summary"]["optimistic_gain_with_allowance_bpw"], "allowance total")
    close(REQUIRED - optimistic, full["summary"]["shortfall_to_required_bpw"], "shortfall")
    require(full["summary"]["decision"] == "kill" and optimistic < REQUIRED, "kill decision")
    print(
        json.dumps(
            {
                "status": "PASS",
                "plan_lock_sha256": LOCK,
                "free_artifact_sha256": sha(free_path),
                "crossfit_artifact_sha256": sha(full_path),
                "engine_sha256": sha(engine_path),
                "free_script_sha256": sha(free_script_path),
                "best_free_gain_bpw": free_best,
                "best_crossfit_gain_bpw": best_crossfit,
                "optimistic_with_allowance_bpw": optimistic,
                "required_gain_bpw": REQUIRED,
                "physical_sources_checked": physical_sources_checked,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
