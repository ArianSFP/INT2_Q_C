#!/usr/bin/env python3
"""Build the compact, source-free expert-affine checkpoint bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_expert_local_codec import common
from strata_expert_local_codec import verify_checkpoint


def copy_row(
    source: Path,
    output: Path,
    relative: str,
    role: str,
    classification: str,
) -> dict[str, Any]:
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": relative,
        "bytes": destination.stat().st_size,
        "sha256": common.sha256_file(destination),
        "role": role,
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_dir = args.plan_dir.resolve(strict=True)
    audit_path = args.audit_report.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory must not exist: {output}")
    output.mkdir(parents=True)

    plan = json.loads((plan_dir / "plan.lock.json").read_text(encoding="utf-8"))
    if not common.verify_internal_seal(plan):
        raise ValueError("plan seal mismatch")
    summary = json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if summary.get("status") != "encoded_once_and_packed":
        raise ValueError("summary is not a completed one-shot run")
    if audit.get("status") != "passed" or not audit.get("milestone_gate", {}).get("passed"):
        raise ValueError("independent milestone audit has not passed")
    if summary.get("plan_lock_sha256") != plan.get("lock_sha256"):
        raise ValueError("summary/plan lock binding mismatch")
    bindings = audit.get("bindings", {})
    if bindings.get("plan_lock_sha256") != plan.get("lock_sha256"):
        raise ValueError("audit/plan lock binding mismatch")
    expected_sources = hashlib.sha256(common.canonical_bytes(plan["sources"])).hexdigest()
    if bindings.get("sources_canonical_sha256") != expected_sources:
        raise ValueError("audit/source binding mismatch")
    artifact = plan_dir / str(summary["artifact"]["relpath"])
    if common.sha256_file(artifact) != summary["artifact"]["sha256"]:
        raise ValueError("summary/container binding mismatch")
    if audit["container"]["sha256"] != summary["artifact"]["sha256"]:
        raise ValueError("audit/container binding mismatch")

    rows = [
        copy_row(
            plan_dir / "plan.lock.json",
            output,
            "plan.lock.json",
            "plan",
            "sealed_preencoding_plan",
        ),
        copy_row(
            plan_dir / "summary.json",
            output,
            "summary.json",
            "summary",
            "one_shot_execution_evidence",
        ),
        copy_row(
            artifact,
            output,
            artifact.name,
            "container",
            "byte_bound_binary_evidence",
        ),
        copy_row(
            audit_path,
            output,
            "independent_audit.json",
            "independent_audit",
            "independent_decode_and_source_score",
        ),
    ]
    for name in ("header.bin", "route.bin", "labels_3bit.bin", "profiles.bin"):
        rows.append(
            copy_row(
                plan_dir / name,
                output,
                f"assets/{name}",
                f"asset_{name.replace('.', '_')}",
                "sealed_format_asset",
            )
        )
    encoded = sorted((plan_dir / "encoded").glob("block_*.json"))
    if len(encoded) != common.BLOCKS:
        raise ValueError("encoder metadata coverage mismatch")
    for ordinal, path in enumerate(encoded):
        if path.name != f"block_{ordinal:02d}.json":
            raise ValueError("encoder metadata ordinal mismatch")
        rows.append(
            copy_row(
                path,
                output,
                f"encoder_metadata/{path.name}",
                f"encoder_block_{ordinal:02d}",
                "one_shot_execution_evidence",
            )
        )

    mse = float(audit["source_score"]["energy_weighted_relative_mse"])
    rate = float(audit["container"]["physical_bpw"])
    max_read = float(audit["read_amplification"]["max_4k"])
    final_pass = bool(
        audit["rate_relative"]["passes_20_percent_below_same_rate_gaussian"]
    )
    manifest = {
        "schema": verify_checkpoint.MANIFEST_SCHEMA,
        "artifact": {
            "architecture": "STRATA expert-affine N20/N21 checkpoint",
            "format_magic": "PLRLOC3\\0",
            "model": "Qwen/Qwen3-30B-A3B",
            "panel": "six precommitted expert triplets / 18 full matrices",
            "weights": common.WEIGHTS,
            "physical_bytes": common.PHYSICAL_BYTES,
            "physical_bits": common.PHYSICAL_BITS,
            "physical_bpw": rate,
            "container_sha256": summary["artifact"]["sha256"],
            "scope": (
                "source-free checkpoint evidence; excludes raw BF16 sources, "
                "duplicate per-block payloads, and decoded FP64 scratch"
            ),
        },
        "claim": {
            "checkpoint_passed": True,
            "final_rate_relative_gate_passed": final_pass,
            "energy_weighted_relative_mse": mse,
            "physical_bpw": rate,
            "max_4k_read_amplification": max_read,
            "gaussian_assumed_mse": float(
                audit["rate_relative"]["gaussian_assumed_mse"]
            ),
            "target_mse_at_same_rate": float(
                audit["rate_relative"]["target_mse_at_same_rate"]
            ),
            "claim_boundary": (
                "one deterministic precommitted 18-matrix Qwen panel; external "
                "compressed-byte reads, not a fused inference-kernel benchmark"
            ),
        },
        "files": rows,
    }
    common.write_json(output / "checkpoint_manifest.json", manifest)
    parsed = verify_checkpoint.parse_container(output / artifact.name)
    roles = verify_checkpoint.verify_manifest(
        output, output / "checkpoint_manifest.json", manifest
    )
    result = verify_checkpoint.verify_evidence(roles, parsed, manifest)
    print(json.dumps({"status": "published_and_verified", **result}, indent=2))


if __name__ == "__main__":
    main()
