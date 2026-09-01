#!/usr/bin/env python3
"""Dependency-free verifier for a BiSCo auxiliary raw-MSE gate result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


PARENT_PROTOCOL_SHA256 = "28c2bd6656f31ce7315601d0048d0b43759a7f2859142f745465e8fa0fe83164"
ASSESSMENT_SHA256 = "859ba01b285ad497fbcca63c9ef47c6e4c079c7e549ba33ca22ac24fab54f581"
LEDGER_SCRIPT_SHA256 = "0c8be46df79b42e15d1435a4d4edd60a511201786f21dcd5920e97b5e0d70cc0"
LAUNCH_PROTOCOL_SHA256 = "0d79a1b8e3cacbc345bdea464986279b0935c4cf2e20290dea75507f7fbfcd4c"
WEIGHTS_PER_EXPERT = 4_718_592
D = 16
HIDDEN = 64
BITS = 18
HEADER_BYTES = 256
LOCAL_SCALE_BYTES = 12
TARGET_S = -0.5 * math.log2(0.8)
VALIDATION_EXPERTS = (24, 56, 88, 120)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 2e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError({"actual": actual, "expected": expected})


def decoder_parameters() -> int:
    return 3 * HIDDEN * (2 * BITS + 2 * D + 2) + 6 * D


def ledger(experts: int) -> dict[str, float | int]:
    params = decoder_parameters()
    decoder_bytes = 2 * params
    global_bytes = decoder_bytes + HEADER_BYTES
    code_bits = (WEIGHTS_PER_EXPERT // D) * (2 * BITS)
    if code_bits % 8:
        raise AssertionError(code_bits)
    code_bytes = code_bits // 8
    attributed = code_bytes + global_bytes / experts + LOCAL_SCALE_BYTES
    cold = code_bytes + global_bytes + LOCAL_SCALE_BYTES
    rate = 8.0 * attributed / WEIGHTS_PER_EXPERT
    side = rate - 2 * BITS / D
    return {
        "decoder_parameters": params,
        "decoder_bytes": decoder_bytes,
        "header_bytes": HEADER_BYTES,
        "local_scale_bytes_per_expert": LOCAL_SCALE_BYTES,
        "code_bits_per_expert": code_bits,
        "code_bytes_per_expert": code_bytes,
        "attributed_physical_bytes_per_expert": attributed,
        "cold_bytes_per_expert": cold,
        "physical_bpw": rate,
        "side_bpw": side,
        "cold_read_amplification": cold / attributed,
        "minimum_matched_s_if_gaussian_code_is_ideal": TARGET_S + side,
        "target_relative_mse": 0.8 * 2.0 ** (-2.0 * rate),
    }


def recompute_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    rows = evaluation["per_expert"]
    if tuple(int(row["expert"]) for row in rows) != VALIDATION_EXPERTS:
        raise AssertionError("wrong whole-expert fold identities")
    matrices = evaluation["per_matrix"]
    identities = [(int(row["expert"]), row["role"]) for row in matrices]
    expected_identities = [(expert, role) for role in ("up", "down") for expert in VALIDATION_EXPERTS]
    if identities != expected_identities:
        raise AssertionError({"per_matrix_identities": identities, "expected": expected_identities})
    matrix_sums = {
        expert: {"qwen_sse": 0.0, "qwen_energy": 0.0, "gaussian_sse": 0.0, "gaussian_energy": 0.0}
        for expert in VALIDATION_EXPERTS
    }
    for matrix in matrices:
        target = matrix_sums[int(matrix["expert"])]
        for field in target:
            value = float(matrix[field])
            if not math.isfinite(value) or value <= 0.0:
                raise AssertionError({"nonpositive_matrix_statistic": field, "row": matrix})
            target[field] += value
    for row in rows:
        expert = int(row["expert"])
        for field, expected in matrix_sums[expert].items():
            close(float(row[field]), expected)
    q_sse = sum(float(row["qwen_sse"]) for row in rows)
    q_energy = sum(float(row["qwen_energy"]) for row in rows)
    g_sse = sum(float(row["gaussian_sse"]) for row in rows)
    g_energy = sum(float(row["gaussian_energy"]) for row in rows)
    d_q = q_sse / q_energy
    d_g = g_sse / g_energy
    s_values = []
    for row in rows:
        fold_dq = float(row["qwen_sse"]) / float(row["qwen_energy"])
        fold_dg = float(row["gaussian_sse"]) / float(row["gaussian_energy"])
        fold_s = -0.5 * math.log2(fold_dq / fold_dg)
        close(float(row["D_Qwen"]), fold_dq)
        close(float(row["D_Gaussian"]), fold_dg)
        close(float(row["s_match"]), fold_s)
        s_values.append(fold_s)
    s_match = -0.5 * math.log2(d_q / d_g)
    se = statistics.stdev(s_values) / math.sqrt(len(s_values))
    close(float(evaluation["D_Qwen"]), d_q)
    close(float(evaluation["D_Gaussian"]), d_g)
    close(float(evaluation["s_match"]), s_match)
    close(float(evaluation["whole_expert_standard_error"]), se)
    close(float(evaluation["upper_s_match_2se"]), s_match + 2.0 * se)
    if bool(evaluation["all_whole_expert_folds_positive"]) != all(value > 0.0 for value in s_values):
        raise AssertionError("whole-expert positivity gate mismatch")
    gaussian_f = d_g * 2.0 ** (2.0 * (2 * BITS / D))
    close(float(evaluation["Gaussian_operational_gap"]["F_gaussian"]), gaussian_f)
    close(float(evaluation["Gaussian_operational_gap"]["s_gaussian"]), -0.5 * math.log2(gaussian_f))
    for name, experts in (("production_128", 128), ("self_contained_panel_6", 6)):
        row = evaluation["absolute"][name]
        rate = float(ledger(experts)["physical_bpw"])
        f_value = d_q * 2.0 ** (2.0 * rate)
        close(float(row["physical_R"]), rate)
        close(float(row["F"]), f_value)
        close(float(row["s_absolute"]), -0.5 * math.log2(f_value))
        if bool(row["passes_F_0p8"]) != (f_value <= 0.8):
            raise AssertionError("absolute target gate mismatch")
    return {"D_Qwen": d_q, "D_Gaussian": d_g, "s_match": s_match, "upper_s_match_2se": s_match + 2.0 * se}


def verify(result_path: Path, aux_dir: Path | None = None) -> dict[str, Any]:
    result_path = result_path.resolve()
    result_dir = result_path.parent
    package_dir = Path(__file__).resolve().parent
    redteam = package_dir.parent / "breakthrough_redteam"
    expected_bindings = {
        "launch": LAUNCH_PROTOCOL_SHA256,
        "parent": PARENT_PROTOCOL_SHA256,
        "assessment": ASSESSMENT_SHA256,
        "ledger": LEDGER_SCRIPT_SHA256,
        "executed_script_sha256": sha256_file(package_dir / "bisco_raw_mse_oracle.py"),
    }
    paths = {
        "launch": package_dir / "launch_protocol.json",
        "parent": redteam / "bisco_protocol_freeze.json",
        "assessment": redteam / "BISCO_BSQ_ASSESSMENT.md",
        "ledger": redteam / "bisco_bsq_ledger.py",
    }
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != expected_bindings[name]:
            raise AssertionError({"binding": name, "actual": actual, "expected": expected_bindings[name]})
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["bindings"] != expected_bindings:
        raise AssertionError({"result_bindings": result["bindings"], "expected": expected_bindings})
    if result["protocol"] != "bisco_raw_mse_aux_d16_launch_v1":
        raise AssertionError("wrong protocol")
    if result["pinned_panel"]["opened"] or result["pinned_panel"]["path_argument_supported"]:
        raise AssertionError("pinned-panel firewall was not preserved")
    if result["backend"]["name"] != "cupy":
        raise AssertionError("non-CuPy output is not a production gate")
    for role, hashes in result["paired_control"]["initialization_sha256"].items():
        if hashes["qwen"] != hashes["gaussian"]:
            raise AssertionError(f"paired initialization mismatch: {role}")
    expected_ledgers = {"production_128": ledger(128), "self_contained_panel_6": ledger(6)}
    for name, expected in expected_ledgers.items():
        actual = result["physical_ledger"][name]
        for field, expected_value in expected.items():
            if isinstance(expected_value, float):
                close(float(actual[field]), expected_value)
            elif actual[field] != expected_value:
                raise AssertionError({"ledger": name, "field": field})
        if not (2.15 <= float(actual["physical_bpw"]) <= 2.5 and float(actual["cold_read_amplification"]) < 2.0):
            raise AssertionError("rate/read gate failed")
    history = result["training"]["history"]
    recomputed_history = []
    for item in history:
        recomputed_history.append({"update": int(item["update"]), **recompute_evaluation(item["evaluation"])})
    if int(result["training"]["stopped_update"]) == 512:
        by_update = {item["update"]: item for item in recomputed_history}
        improvement = by_update[512]["upper_s_match_2se"] - by_update[256]["upper_s_match_2se"]
        killed = by_update[512]["upper_s_match_2se"] < 0.08 and improvement < 0.01
        if not killed or result["decision"] != "HARD_KILL_D16_SHALLOW_BEFORE_PINNED":
            raise AssertionError("early-kill decision mismatch")
        close(float(result["training"]["early_kill"]["late_improvement"]), improvement)
    final_recomputed = recompute_evaluation(result["final_evaluation"])
    for key in ("D_Qwen", "D_Gaussian", "s_match", "upper_s_match_2se"):
        close(float(final_recomputed[key]), float(recomputed_history[-1][key]))
    for domain, domain_artifacts in result["artifacts"].items():
        for artifact_name, artifact in domain_artifacts.items():
            path = result_dir / artifact["file"]
            if path.stat().st_size != int(artifact["bytes"]):
                raise AssertionError(f"artifact size mismatch: {path}")
            if sha256_file(path) != artifact["sha256"]:
                raise AssertionError(f"artifact hash mismatch: {path}")
            bytes_per_value = 2 if artifact_name == "auxiliary_two_role_decoder" else 4
            schema_values = sum(int(row["values"]) for row in artifact["schema"])
            if schema_values * bytes_per_value != int(artifact["bytes"]):
                raise AssertionError(f"artifact schema/byte mismatch: {path}")
            expected_offset = 0
            for schema_row in artifact["schema"]:
                if int(schema_row["offset_values"]) != expected_offset:
                    raise AssertionError(f"noncontiguous artifact schema: {path}")
                shape_values = math.prod(int(value) for value in schema_row["shape"])
                if shape_values != int(schema_row["values"]):
                    raise AssertionError(f"artifact shape/value mismatch: {path}")
                expected_offset += shape_values
        if int(domain_artifacts["auxiliary_two_role_decoder"]["bytes"]) != 18_048:
            raise AssertionError(f"wrong two-role decoder bytes: {domain}")
    source_checks = 0
    if aux_dir is not None:
        expected_files = result["data_firewall"]["source_sha256"]
        actual_names = {path.name for path in aux_dir.resolve().glob("*.bf16.bin")}
        if actual_names != set(expected_files):
            raise AssertionError("auxiliary directory file set mismatch")
        for name, expected_hash in expected_files.items():
            if sha256_file(aux_dir / name) != expected_hash:
                raise AssertionError(f"auxiliary hash mismatch: {name}")
            source_checks += 1
    return {
        "verified": True,
        "result_sha256": sha256_file(result_path),
        "decision": result["decision"],
        "backend": result["backend"],
        "final": final_recomputed,
        "physical_ledger": expected_ledgers,
        "artifacts_verified": 4,
        "auxiliary_sources_rehashed": source_checks,
        "pinned_panel_opened": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--aux-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = verify(args.result, args.aux_dir)
    payload = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if args.receipt:
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
