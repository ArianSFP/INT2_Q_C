#!/usr/bin/env python3
"""Independent standard-library verifier for the downloaded meta result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


EXPECTED_RESULT_FILES = {
    "result.json": (47607, "9d3f43c8c417e0c9f84849e7a27e6feebbd952a97a74ef29a2307cb933f2a5a0"),
    "seed_2026090101/gaussian_global_side.bin": (278528, "1707403480f12ebbf2ac0c492910077843cd718f00ff887401524e8a46a93b9f"),
    "seed_2026090101/gaussian_row_moments.bin": (55296, "f734342eafe04669d634f8dd0c520c029875b942d66d7c714e4055b6f2acfe69"),
    "seed_2026090101/source_global_side.bin": (278528, "968880462be9e86167168bb453c289f7b35d9d9e2816452cd8af71fb1c96e19e"),
    "seed_2026090101/source_row_moments.bin": (55296, "f734342eafe04669d634f8dd0c520c029875b942d66d7c714e4055b6f2acfe69"),
    "seed_2026090102/gaussian_global_side.bin": (278528, "d5505c43f8a4c5b7c0eb4feba9c5e55d47847c54713666e667bdf52e6ade0179"),
    "seed_2026090102/gaussian_row_moments.bin": (55296, "f734342eafe04669d634f8dd0c520c029875b942d66d7c714e4055b6f2acfe69"),
    "seed_2026090102/source_global_side.bin": (278528, "71f1723c9d01bffbc9815e65104dc03a1d46044323e57550b30adc27e8193f69"),
    "seed_2026090102/source_row_moments.bin": (55296, "f734342eafe04669d634f8dd0c520c029875b942d66d7c714e4055b6f2acfe69"),
}
EXPECTED_AUDIT_FILES = {"README.md", "SOURCE_MANIFEST.json", "audit_receipt.json", "verify_audit.py"}
PRODUCER_MANIFEST_SHA256 = "d62ee98c3c76fd25b81b50d83abe945f620972a630354aa9ec2606ee2783d9fa"
EXPECTED_RESULT_LOCK = "38970e2aafaddd224cd5a103a332b738a99cad555615fe3208eb18ec6a4a6509"


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise RuntimeError(message)

    def close(self, left: float, right: float, message: str, tolerance: float = 1.0e-12) -> None:
        self.require(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance), message)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def relative_files(root: Path) -> set[str]:
    return {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
    }


def check_finite_half(checks: Checks, payload: bytes, start: int, stop: int, label: str) -> None:
    checks.require((stop - start) % 2 == 0, f"{label} half alignment")
    for offset in range(start, stop, 2):
        word = struct.unpack_from("<H", payload, offset)[0]
        checks.require((word & 0x7C00) != 0x7C00, f"non-finite FP16 in {label} at {offset}")


def ledger(experts: int) -> tuple[float, float, list[dict[str, Any]]]:
    values = experts * 3 * 768 * 2048
    side = 278528
    fixed = 1105920 + 9216 + 64
    prefix = (side * 8 + experts * fixed * 8) / values
    required_q = 0.8 / (2.0 ** (2.0 * prefix))
    rows = []
    for requested in (2.15, 2.30, 2.50):
        physical = math.ceil(requested * values / 8.0)
        local_total = physical - side
        local_min, remainder = divmod(local_total, experts)
        local_max = local_min + int(remainder != 0)
        residual = physical - side - experts * fixed
        cold = side + 4096 * math.ceil(local_max / 4096.0)
        rows.append(
            {
                "physical_bytes": physical,
                "local_total_bytes": local_total,
                "local_frame_min_bytes": local_min,
                "local_frame_max_bytes": local_max,
                "large_frame_count": remainder,
                "total_residual_bytes": residual,
                "residual_bpw": residual * 8.0 / values,
                "cold_expert_bytes_4k": cold,
                "cold_read_amplification": cold / (physical / experts),
                "actual_bpw": physical * 8.0 / values,
            }
        )
    return prefix, required_q, rows


def check_evaluation(checks: Checks, evaluation: dict[str, Any], oracle: dict[str, Any], prefix: float) -> None:
    checks.close(evaluation["sse"] / evaluation["source_energy"], evaluation["relative_residual_energy"], "pooled q")
    checks.close(evaluation["relative_residual_energy"], oracle["relative_residual_energy"], "oracle q")
    expected_f = evaluation["relative_residual_energy"] * 2.0 ** (2.0 * prefix)
    checks.close(expected_f, oracle["F_oracle"], "pooled F", 2.0e-12)
    checks.close(-0.5 * math.log2(expected_f), oracle["s_oracle"], "pooled s")
    expert_sse = 0.0
    expert_energy = 0.0
    for expert, oracle_expert in zip(evaluation["heldout_experts"], oracle["heldout_experts"], strict=True):
        checks.require(expert["slot"] == oracle_expert["slot"], "expert order")
        checks.close(expert["sse"] / expert["source_energy"], expert["relative_residual_energy"], "expert q")
        checks.close(expert["relative_residual_energy"] * 2.0 ** (2.0 * prefix), oracle_expert["F_oracle"], "expert F")
        checks.close(-0.5 * math.log2(oracle_expert["F_oracle"]), oracle_expert["s_oracle"], "expert s")
        matrix_sse = 0.0
        matrix_energy = 0.0
        for matrix in expert["matrices"]:
            checks.close(matrix["sse"] / matrix["source_energy"], matrix["relative_residual_energy"], "matrix q")
            matrix_sse += matrix["sse"]
            matrix_energy += matrix["source_energy"]
        checks.close(matrix_sse, expert["sse"], "expert SSE sum")
        checks.close(matrix_energy, expert["source_energy"], "expert energy sum")
        expert_sse += expert["sse"]
        expert_energy += expert["source_energy"]
    checks.close(expert_sse, evaluation["sse"], "pooled SSE sum")
    checks.close(expert_energy, evaluation["source_energy"], "pooled energy sum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = Checks()
    project = args.root.resolve()
    producer = project / "research/meta_codebook_whole_expert_stage0_v0"
    result_root = project / "research/meta_codebook_whole_expert_stage0_v0_runpod_result_20260901"
    audit = Path(__file__).resolve().parent

    checks.require(audit.parent == project / "research", "audit location")
    checks.require(relative_files(audit) == EXPECTED_AUDIT_FILES, "exact audit closure")
    checks.require(all(not path.is_symlink() for path in audit.rglob("*")), "audit contains no links")
    audit_manifest = json.loads((audit / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    checks.require(audit_manifest["schema"] == "meta-codebook-stage0-independent-result-audit-source-manifest-v1",
                   "audit manifest schema")
    checks.require(audit_manifest["closure"] == sorted(EXPECTED_AUDIT_FILES), "audit manifest closure")
    for row in audit_manifest["files"]:
        path = audit / row["path"]
        checks.require(path.stat().st_size == row["bytes"], f"audit bytes {row['path']}")
        checks.require(sha256_file(path) == row["sha256"], f"audit hash {row['path']}")
    producer_manifest = producer / "SOURCE_MANIFEST.json"
    checks.require(sha256_file(producer_manifest) == PRODUCER_MANIFEST_SHA256, "producer manifest hash")
    manifest = json.loads(producer_manifest.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = producer / row["path"]
        checks.require(path.stat().st_size == row["bytes"], f"producer bytes {row['path']}")
        checks.require(sha256_file(path) == row["sha256"], f"producer hash {row['path']}")

    checks.require(relative_files(result_root) == set(EXPECTED_RESULT_FILES), "result closure")
    for relative, (size, digest) in EXPECTED_RESULT_FILES.items():
        path = result_root / relative
        checks.require(path.is_file() and not path.is_symlink(), f"result member {relative}")
        checks.require(path.stat().st_size == size, f"result bytes {relative}")
        checks.require(sha256_file(path) == digest, f"result hash {relative}")

    result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
    declared_lock = result.pop("result_lock_sha256")
    checks.require(declared_lock == EXPECTED_RESULT_LOCK, "declared result lock")
    checks.require(hashlib.sha256(canonical_json_bytes(result)).hexdigest() == EXPECTED_RESULT_LOCK, "recomputed result lock")
    result["result_lock_sha256"] = declared_lock
    checks.require(result["schema"] == "meta-codebook-whole-expert-stage0-result-v0", "result schema")
    checks.require(result["status"] == "KILL", "result status")
    checks.require(result["decision_reasons"] == ["even the better predeclared seed fails the favorable source oracle"], "decision reason")
    checks.require(result["split"] == {"fit_slots": [0, 2, 3, 5], "holdout_slots": [1, 4]}, "split")
    checks.require(result["source_lock"]["file_sha256"] == "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23", "source lock")
    checks.require("Six-expert frozen stage-0 only" in result["claim_boundary"], "claim boundary")

    prefix6, q6, ledger6 = ledger(6)
    prefix128, q128, ledger128 = ledger(128)
    checks.close(prefix6, result["six_expert_fixed_prefix_bpw"], "six prefix")
    checks.close(q6, result["six_expert_required_first_stage_relative_residual_energy"], "six q")
    checks.close(prefix128, result["hypothetical_128_expert_layer_ledger"]["fixed_prefix_bpw"], "128 prefix")
    checks.close(q128, result["hypothetical_128_expert_layer_ledger"]["required_first_stage_relative_residual_energy"], "128 q")
    for expected, observed in zip(ledger6, result["six_expert_rate_ledger"], strict=True):
        for key, value in expected.items():
            checks.close(value, observed[key], f"six ledger {key}")
    for expected, observed in zip(ledger128, result["hypothetical_128_expert_layer_ledger"]["rates"], strict=True):
        for key, value in expected.items():
            checks.close(value, observed[key], f"128 ledger {key}")
    checks.require(all(row["cold_read_amplification"] < 2.0 for row in result["six_expert_rate_ledger"]), "six reads")
    checks.require(all(row["cold_read_amplification"] < 2.0 for row in result["hypothetical_128_expert_layer_ledger"]["rates"]), "128 reads")

    moment_hashes = set()
    source_s = []
    for seed_row in result["seed_reports"]:
        seed = int(seed_row["seed"])
        for label in ("source", "gaussian"):
            run = seed_row[label]
            global_path = (result_root / run["global_side_relpath"]).resolve()
            moment_path = (result_root / run["row_moments_relpath"]).resolve()
            checks.require(global_path.is_relative_to(result_root.resolve()), "global path containment")
            checks.require(moment_path.is_relative_to(result_root.resolve()), "moment path containment")
            checks.require(global_path.stat().st_size == run["global_side_bytes"] == 278528, "global bytes")
            checks.require(moment_path.stat().st_size == run["row_moments_bytes"] == 55296, "moment bytes")
            checks.require(sha256_file(global_path) == run["global_side_sha256"], "declared global hash")
            checks.require(sha256_file(moment_path) == run["row_moments_sha256"], "declared moment hash")
            moment_hashes.add(run["row_moments_sha256"])
            global_bytes = global_path.read_bytes()
            fields = struct.unpack_from("<8sIIII", global_bytes, 0)
            checks.require(fields == (b"MCBWES0\0", 0, 32768, 4, 15), "global header")
            metadata_end = global_bytes.find(b"\0", 24)
            checks.require(metadata_end > 24, "metadata terminator")
            metadata = json.loads(global_bytes[24:metadata_end].decode("utf-8"))
            checks.require(metadata["seed"] == seed and metadata["control"] == (label == "gaussian"), "header metadata")
            check_finite_half(checks, global_bytes, 4096, 276752, f"{seed}/{label}/global")
            checks.require(global_bytes[276752:] == bytes(1776), "global zero padding")
            moment_bytes = moment_path.read_bytes()
            check_finite_half(checks, moment_bytes, 0, len(moment_bytes), f"{seed}/{label}/moments")
            for offset in range(0, len(moment_bytes), 4):
                mean = struct.unpack_from("<e", moment_bytes, offset)[0]
                rms = struct.unpack_from("<e", moment_bytes, offset + 2)[0]
                checks.require(rms > 0.0 and rms * rms > mean * mean, "valid row RMS")
            check_evaluation(checks, run["evaluation"], run["oracle"], prefix6)
        checks.close(
            seed_row["source"]["oracle"]["s_oracle"] - seed_row["gaussian"]["oracle"]["s_oracle"],
            seed_row["matched_advantage_s"],
            "matched advantage",
        )
        checks.require(seed_row["source"]["training_trace"][-1]["step"] == 512, "source final step")
        checks.require(seed_row["source"]["training_trace"][-1]["codes_used_in_batch"] < 2048, "source utilization collapse")
        source_s.append(float(seed_row["source"]["oracle"]["s_oracle"]))
    checks.require(moment_hashes == {"f734342eafe04669d634f8dd0c520c029875b942d66d7c714e4055b6f2acfe69"}, "identical moment side")
    checks.require(max(source_s) < result["target"]["s"], "KILL recomputation")
    checks.require(all(row["declared_sha256"] == row["observed_sha256"] for row in result["source_receipts"]), "source receipt equality")
    checks.require(len(result["source_receipts"]) == 18, "source receipt count")
    checks.require(all(row["maximum_row_absolute_mean_error"] <= row["tolerance"] and row["maximum_row_absolute_rms_error"] <= row["tolerance"] for row in result["gaussian_moment_match"]), "Gaussian moment match")

    receipt = json.loads((audit / "audit_receipt.json").read_text(encoding="utf-8"))
    checks.require(receipt["verdict"] == "PASS_KILL_CONFIRMED_LATENT4_ONLY", "audit receipt verdict")
    checks.require(receipt["result_origin_path"] ==
                   "research/meta_codebook_whole_expert_stage0_v0_runpod_result_20260901",
                   "sibling result origin")
    checks.require("Direct K=32768,d=8 output-space K-means" in receipt["claim_boundary"], "narrow audit scope")
    print(json.dumps({"status": "PASS_KILL_CONFIRMED_LATENT4_ONLY", "checks": checks.count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
