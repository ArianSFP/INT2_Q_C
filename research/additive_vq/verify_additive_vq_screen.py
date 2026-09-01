#!/usr/bin/env python3
"""Dependency-free verifier for additive_vq_screen_result.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


WEIGHTS = 28_311_552
EXPERTS = 6
ROLES = 3
TARGET_S = -0.5 * math.log2(0.8)
PLAN_LOCK = "99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-15):
        raise ValueError(f"{name}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if result["schema"] != "qwen-additive-vq-early-kill-v1":
        raise ValueError("wrong result schema")
    if not result["cpu_only"]:
        raise ValueError("run does not claim CPU-only")
    if plan.get("lock_sha256") != PLAN_LOCK:
        raise ValueError("wrong plan lock")
    clean = dict(plan)
    clean.pop("lock_sha256")
    recomputed_lock = hashlib.sha256(
        json.dumps(
            clean,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if recomputed_lock != PLAN_LOCK:
        raise ValueError("plan internal seal mismatch")
    close(result["target"]["required_s_bpw"], TARGET_S, "target s")
    if result["target"]["required_F"] != 0.8:
        raise ValueError("wrong target F")
    dimensions = result["parameters"]["dimensions"]
    alphabets = result["parameters"]["alphabets"]
    expected_pairs = {(d, k) for d in dimensions for k in alphabets}
    rows = result["results"]
    if {(row["dimension"], row["alphabet"]) for row in rows} != expected_pairs:
        raise ValueError("dimension/alphabet grid incomplete")
    if len(rows) != len(expected_pairs):
        raise ValueError("duplicate result grid cells")
    for row in rows:
        d = row["dimension"]
        k = row["alphabet"]
        m = row["stages"]
        payload = m * math.log2(k) / d
        table = ROLES * m * k * d * 16
        scalars = EXPERTS * ROLES * 3 * 32
        header = 256
        pack = EXPERTS if k == 3 else 0
        side_bits = table + scalars + header + pack
        side_bpw = side_bits / WEIGHTS
        physical = payload + side_bpw
        close(row["payload_rate_bpw"], payload, "payload")
        if row["table_bits_fp16"] != table or row["matrix_scalar_bits_fp32"] != scalars:
            raise ValueError("table/scalar ledger mismatch")
        if row["side_bits"] != side_bits:
            raise ValueError("side-bit ledger mismatch")
        close(row["side_bpw"], side_bpw, "side bpw")
        close(row["physical_rate_bpw"], physical, "physical rate")
        if not 2.15 <= physical <= 2.5:
            raise ValueError("rate out of range")
        ds = row["source_distortion"]
        dg = row["matched_gaussian_distortion"]
        matched_f = ds / dg
        matched_s = -0.5 * math.log2(matched_f)
        charged_s = matched_s - side_bpw
        charged_f = math.pow(2.0, -2.0 * charged_s)
        optimistic_s = (
            charged_s
            + 2.0 * row["fold_s_standard_error_bpw"]
            + row["optimism_allowance_bpw"]
        )
        optimistic_f = math.pow(2.0, -2.0 * optimistic_s)
        shannon_d = math.pow(2.0, -2.0 * physical)
        exact_f = ds / shannon_d
        exact_s = -0.5 * math.log2(exact_f)
        close(row["matched_F_source_over_control"], matched_f, "matched F")
        close(row["matched_s_bpw_before_side_charge"], matched_s, "matched s")
        close(row["charged_matched_s_bpw"], charged_s, "charged s")
        close(row["charged_matched_F_identity"], charged_f, "charged F identity")
        close(row["optimistic_2se_s_bpw"], optimistic_s, "optimistic s")
        close(row["optimistic_2se_F_identity"], optimistic_f, "optimistic F identity")
        close(row["gaussian_shannon_distortion_at_physical_rate"], shannon_d, "Shannon D")
        close(row["exact_F_source_over_shannon_at_physical_rate"], exact_f, "exact F")
        close(row["exact_s_bpw_source_vs_shannon"], exact_s, "exact s")
        if len(row["folds"]) != EXPERTS:
            raise ValueError("fold count mismatch")
        if sorted(fold["heldout_expert"] for fold in row["folds"]) != list(range(EXPERTS)):
            raise ValueError("held-out expert coverage mismatch")
        if any(len(fold["roles"]) != ROLES for fold in row["folds"]):
            raise ValueError("role coverage mismatch")
        if not row["cold_expert_read_amplification"] < 2.0:
            raise ValueError("read amplification gate failed")
        if row["exact_target_met"] or row["optimistic_calibrated_target_met"]:
            raise ValueError("result unexpectedly survives")
    source_root = Path(plan["source_root"]).resolve(strict=True)
    for provenance in result["provenance_by_dimension"]:
        if provenance["plan_lock_sha256"] != PLAN_LOCK:
            raise ValueError("provenance lock mismatch")
        if provenance["plan_file_sha256"] != sha256_file(args.plan):
            raise ValueError("provenance plan-file hash mismatch")
        header = Path(provenance["header_path"]).resolve(strict=True)
        if provenance["header_sha256"] != sha256_file(header):
            raise ValueError("header hash mismatch")
        bindings = provenance["source_bindings"]
        if len(bindings) != EXPERTS * ROLES:
            raise ValueError("binding count mismatch")
        if sorted(row["matrix_ordinal"] for row in bindings) != list(range(EXPERTS * ROLES)):
            raise ValueError("binding ordinal mismatch")
        for binding in bindings:
            source_row = plan["sources"][binding["matrix_ordinal"]]
            if binding["tensor"] != source_row["tensor"]:
                raise ValueError("tensor binding mismatch")
            source = (source_root / source_row["source_relpath"]).resolve(strict=True)
            if source_root not in source.parents:
                raise ValueError("source escaped root")
            if binding["sha256"] != source_row["source_bf16_sha256"]:
                raise ValueError("recorded source digest mismatch")
            if sha256_file(source) != binding["sha256"]:
                raise ValueError("live source digest mismatch")
    best = max(rows, key=lambda row: row["optimistic_2se_s_bpw"])
    if result["decision"] != "kill":
        raise ValueError("decision is not kill")
    receipt = {
        "schema": "qwen-additive-vq-verification-v1",
        "verified": True,
        "result_sha256": sha256_file(args.result),
        "plan_sha256": sha256_file(args.plan),
        "plan_lock_sha256": PLAN_LOCK,
        "verified_grid_cells": len(rows),
        "verified_live_source_files": EXPERTS * ROLES,
        "best_architecture": best["architecture"],
        "best_optimistic_s_bpw": best["optimistic_2se_s_bpw"],
        "best_optimistic_F_identity": best["optimistic_2se_F_identity"],
        "required_s_bpw": TARGET_S,
        "decision": "kill",
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
