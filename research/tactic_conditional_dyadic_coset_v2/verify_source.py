#!/usr/bin/env python3
"""Standard-library source verifier for TACTIC-DH384 v2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

from tactic_v2_common import (
    DESIGN_SCHEMA,
    PAYLOAD_AUTHORIZATION,
    SYNTHETIC_AUTHORIZATION,
    ContractError,
    canonical_json,
    require,
    selector_packet,
    sha256_file,
    strict_json_loads,
    universal_selector_table,
)


MANIFEST_NAME = "SOURCE_MANIFEST.json"
EXPECTED_FILES = {
    "README.md",
    "design_lock.json",
    "tactic_v2_common.py",
    "stage0_gate.py",
    "cupy_preflight.py",
    "verify_source.py",
    "test_source_only.py",
}


def _close(actual: float, expected: float, label: str, tolerance: float = 3e-15) -> None:
    require(math.isfinite(actual), f"{label}: non-finite")
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            f"{label}: {actual!r} != {expected!r}")


def verify_design(data: dict[str, Any]) -> dict[str, Any]:
    require(data.get("schema") == DESIGN_SCHEMA, "design schema")
    require(data.get("status") == "FROZEN_SOURCE_ONLY_NO_PAYLOAD_NO_GPU_AUTHORITY",
            "design status")
    objective = data["objective"]
    geometry = data["geometry"]
    frame = data["conditional_dyadic_frame"]
    physical = data["physical_ledger"]
    read = data["cold_read_ledger"]
    gate = data["planning_gate"]
    auth = data["authorization"]
    coarse_lock = data["coarse_lock_contract"]

    rows, columns = int(geometry["rows"]), int(geometry["columns"])
    matrix = rows * columns
    expert = matrix * len(geometry["roles"])
    total = expert * int(geometry["experts"])
    require(matrix == int(geometry["weights_per_matrix"]) == 1_572_864, "matrix geometry")
    require(expert == int(geometry["weights_per_expert"]) == 4_718_592, "expert geometry")
    require(total == int(geometry["weights_total"]) == 28_311_552, "total geometry")
    require(int(geometry["streams_per_expert"]) * int(geometry["stream_values"]) == expert,
            "stream coverage")
    require(int(geometry["blocks_per_stream"]) * int(geometry["block_values"]) ==
            int(geometry["stream_values"]), "block/stream coverage")
    require(int(geometry["blocks_per_expert"]) ==
            int(geometry["streams_per_expert"]) * int(geometry["blocks_per_stream"]),
            "expert block count")
    require(int(geometry["tangent_rank"]) == 384 < int(geometry["block_values"]), "rank")

    require(int(frame["stages"]) == 12 and int(frame["feature_states"]) == 256,
            "conditional frame geometry")
    require(int(frame["selector_searches"]) == 0, "selector search must be absent")
    require(int(frame["universal_selector_ordinal"]) == 17, "universal selector ordinal")
    require(int(frame["selector_active_bytes"]) == 12 * 256, "selector active bytes")
    require(int(frame["selector_packet_bytes"]) == 16_384, "selector packet bytes")
    require(int(frame["q12_abs_max"]) * 4096 ==
            int(frame["worst_unscaled_accumulator_abs_inclusive"]), "accumulator bound")
    require(int(frame["worst_unscaled_accumulator_abs_inclusive"]) < 1 << 24,
            "accumulator safety")
    require("no intermediate rounding or saturation" in frame["finite_linearity"],
            "linear dominance boundary")
    universal = data["universality_contract"]
    require(universal["model_identity_inputs"] == [], "model identity input")
    require(universal["provenance_inputs"] == [], "provenance input")
    require(universal["external_reference_reads"] == [], "external reference input")
    require(universal["source_fitted_parameters"] == [], "source-fitted parameter")
    require("ordinal 17" in universal["packet_construction"], "universal packet construction")
    require("same 4096-value block" in universal["allowed_decoder_inputs"][0],
            "same-block symbol contract")
    evaluation = data["evaluation"]
    require(evaluation["decision_expert_ordinals"] == list(range(6)), "decision clusters")
    require(evaluation["selector_fit_expert_ordinals"] == [], "selector fit split")
    require(evaluation["architecture_parameters_selected_from_evaluation"] is False,
            "evaluation-selected architecture")
    packet = selector_packet(universal_selector_table())
    require(len(packet) == 16_384, "universal selector packet bytes")
    require(hashlib.sha256(packet).hexdigest() == frame["selector_packet_sha256"] ==
            "0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad",
            "universal selector packet hash")
    require(coarse_lock["panel_kinds_in_order"] == [
        "source", "decoded_gaussian", "structure_destroyed"
    ], "coarse panel order")
    require(int(coarse_lock["must_bind_streams_per_panel"]) == 108,
            "coarse reservoirs per panel")
    require(int(coarse_lock["must_bind_source_reconstruction_and_symbol_records_per_panel"]) == 18,
            "coarse matrix records per panel")
    require(coarse_lock["decode_reencode_required_for_every_panel"] is True,
            "coarse roundtrip per panel")
    require(coarse_lock["all_stream_reservoirs_required_for_every_panel"] is True,
            "coarse reservoir binding")

    global_bytes = sum(int(physical[key]) for key in (
        "global_schema_bytes", "global_selector_packet_bytes", "global_qc_tables_bytes",
        "global_seed_fixture_bytes",
    ))
    require(global_bytes == int(physical["global_packet_bytes"]) == 24_576, "global bytes")
    coset = int(geometry["blocks_per_expert"]) * int(physical["coset_bytes_per_block"])
    require(coset == int(physical["coset_bytes_per_expert"]) == 55_296, "coset bytes")
    coarse = int(geometry["streams_per_expert"]) * int(physical["coarse_bytes_per_stream"])
    require(coarse == int(physical["coarse_bytes_per_expert"]) == 1_414_656, "coarse bytes")
    frame_bytes = int(physical["expert_header_bytes"]) + coset + coarse
    require(frame_bytes == int(physical["expert_frame_bytes"]) == 1_470_464, "frame bytes")
    container = global_bytes + int(geometry["experts"]) * frame_bytes
    require(container == int(physical["container_bytes"]) == 8_847_360, "container bytes")
    require(8 * container == int(physical["container_bits"]) == 70_778_880, "container bits")

    coarse_rate = Fraction(8 * coarse, expert)
    coset_rate = Fraction(8 * coset, expert)
    metadata_rate = Fraction(
        8 * (global_bytes + int(geometry["experts"]) * int(physical["expert_header_bytes"])),
        total,
    )
    total_rate = Fraction(8 * container, total)
    require(coarse_rate == Fraction(307, 128), "coarse rate")
    require(coset_rate == Fraction(12, 128), "coset rate")
    require(metadata_rate == Fraction(1, 128), "metadata rate")
    require(total_rate == Fraction(5, 2), "total rate")
    _close(float(physical["coarse_bpw"]), float(coarse_rate), "coarse_bpw")
    _close(float(physical["coset_bpw"]), float(coset_rate), "coset_bpw")
    _close(float(physical["metadata_bpw"]), float(metadata_rate), "metadata_bpw")
    _close(float(physical["physical_bpw"]), float(total_rate), "physical_bpw")
    require(bool(physical["all_padding_charged"]), "padding not charged")
    require(bool(physical["fixed_reservoir_overflow_is_failure"]), "overflow policy")

    page = int(read["page_bytes"])
    require(global_bytes % page == 0 and frame_bytes % page == 0, "page alignment")
    global_pages, frame_pages = global_bytes // page, frame_bytes // page
    cold_pages = global_pages + frame_pages
    share_pages = Fraction(container, int(geometry["experts"]) * page)
    amplification = Fraction(cold_pages, 1) / share_pages
    require((global_pages, frame_pages, cold_pages, share_pages) == (6, 359, 365, 360),
            "page ledger")
    require(amplification == Fraction(73, 72), "read amplification fraction")
    _close(float(read["amplification"]), float(amplification), "read amplification")
    require(float(amplification) < float(objective["cold_read_limit_exclusive"]), "read gate")
    require(not read["warm_cache_assumed"] and int(read["compressed_frame_reads"]) == 1,
            "cold read semantics")

    base_f = float(gate["published_finite_base_F_at_2p5"])
    planning_d0 = base_f * 2.0 ** (-2.0 * float(coarse_rate))
    c_required = 1.0 - float(objective["target_relative_mse_at_cell_rate"]) / planning_d0
    isotropic = int(geometry["tangent_rank"]) / int(geometry["block_values"])
    _close(float(gate["planning_D0_at_coarse_rate"]), planning_d0, "planning D0")
    _close(float(gate["planning_c_required"]), c_required, "planning c")
    _close(float(gate["isotropic_rank_capture"]), isotropic, "isotropic capture")
    _close(float(gate["required_over_isotropic"]), c_required / isotropic,
           "capture concentration")
    require(gate["planning_factor_transfer_only"] is True, "planning transfer boundary")
    require("measured_actual_lower_rate_D0" in gate["runtime_c_required"],
            "actual lower-rate requirement absent")
    require("capture + 3*whole_expert_SE" in gate["hard_reject"], "upper gate")
    require("capture - 3*whole_expert_SE" in gate["promote"], "lower gate")

    require(auth["synthetic_preflight"] == SYNTHETIC_AUTHORIZATION, "synthetic token")
    require(auth["future_payload_gate"] == PAYLOAD_AUTHORIZATION, "payload token")
    require(auth["source_package_authorizes_payload"] is False, "payload authority")
    require(auth["source_package_authorizes_gpu"] is False, "GPU authority")
    require(auth["independent_source_review_required_before_synthetic_gpu"] is True,
            "independent review boundary")
    require(data["controls"]["open_only_after_absolute_source_oracle_survives"] is True,
            "control access order")
    require(data["controls"]["not_a_converse"] is True, "control claim boundary")
    return {
        "physical_bpw": float(total_rate),
        "coarse_bpw": float(coarse_rate),
        "coset_bpw": float(coset_rate),
        "metadata_bpw": float(metadata_rate),
        "cold_read_amplification": float(amplification),
        "planning_c_required": c_required,
    }


def _imports(tree: ast.AST) -> list[tuple[str, int]]:
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append((node.module or "", node.lineno))
    return result


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def verify_static(package: Path) -> dict[str, Any]:
    common_text = (package / "tactic_v2_common.py").read_text(encoding="utf-8")
    stage_text = (package / "stage0_gate.py").read_text(encoding="utf-8")
    preflight_text = (package / "cupy_preflight.py").read_text(encoding="utf-8")
    common_tree = ast.parse(common_text, filename="tactic_v2_common.py")
    stage_tree = ast.parse(stage_text, filename="stage0_gate.py")
    preflight_tree = ast.parse(preflight_text, filename="cupy_preflight.py")
    forbidden_common = {"numpy", "cupy", "torch", "requests", "urllib", "socket"}
    require(not (_top_level_imports(common_tree) & forbidden_common), "common external import")
    require("cupy" not in _top_level_imports(stage_tree) and "numpy" not in _top_level_imports(stage_tree),
            "stage imports numeric runtime before authorization")
    require("cupy" not in _top_level_imports(preflight_tree) and "numpy" not in _top_level_imports(preflight_tree),
            "preflight imports CUDA before authorization")
    all_imports = {name.split(".")[0] for name, _line in _imports(common_tree)}
    require(not (all_imports & {"requests", "urllib", "socket"}), "common network import")
    require("O_NOFOLLOW" in common_text and "O_EXCL" in common_text and "fstat" in common_text,
            "held-FD/create-new primitives absent")
    require("AUTHORIZATION = PAYLOAD_AUTHORIZATION" in stage_text,
            "stage payload authorization binding absent")
    require("AUTHORIZATION = SYNTHETIC_AUTHORIZATION" in preflight_text,
            "preflight authorization binding absent")
    require("--coarse-lock" in stage_text and "_validate_rate" in stage_text and
            "BASE_BPW = 307.0 / 128.0" in stage_text,
            "actual coarse input boundary")
    require("--coarse-lock" not in preflight_text and "--root" not in preflight_text and
            "--manifest" not in preflight_text, "synthetic preflight accepts payload path")
    require("gpu_projection" in stage_text and "transformed / cp.float64(64.0)" in stage_text,
            "dyadic projection implementation absent")
    require("candidate_selector_table" not in stage_text and "candidate_selector_table" not in preflight_text,
            "source-adaptive selector API imported by executable")
    require("candidate_selector_table" not in common_text,
            "selector-candidate API present in common source")
    require("universal_selector_table" in stage_text and "selector_searches\": 0" in stage_text,
            "universal no-search stage absent")
    return {
        "common_imports": sorted(all_imports),
        "stage_ast_nodes": sum(1 for _ in ast.walk(stage_tree)),
        "preflight_ast_nodes": sum(1 for _ in ast.walk(preflight_tree)),
    }


def verify_manifest(package: Path) -> dict[str, Any]:
    manifest_path = package / MANIFEST_NAME
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "source manifest missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest = strict_json_loads(manifest_bytes)
    require(manifest.get("schema") == "tactic_dh384_source_manifest_v2", "manifest schema")
    records = manifest.get("files")
    require(isinstance(records, list) and len(records) == len(EXPECTED_FILES), "manifest rows")
    names = [row.get("name") for row in records]
    require(names == sorted(EXPECTED_FILES) and len(names) == len(set(names)), "manifest order/names")
    actual = {entry.name for entry in package.iterdir()}
    require(actual == EXPECTED_FILES | {MANIFEST_NAME}, f"package closure: {sorted(actual)}")
    hashes = {}
    for row in records:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "manifest row schema")
        path = package / row["name"]
        require(path.is_file() and not path.is_symlink(), f"invalid member {row['name']}")
        require(path.stat().st_size == int(row["bytes"]), f"member size {row['name']}")
        observed = sha256_file(path)
        require(observed == row["sha256"], f"member hash {row['name']}")
        hashes[row["name"]] = observed
    return {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "file_hashes": hashes}


def verify_readme(package: Path) -> None:
    text = (package / "README.md").read_text(encoding="utf-8")
    required = [
        "73/72", "0.2972443434920543", "actual", "held regular-file descriptors",
        "RAVEL is not this cell", "MALT64 does not contain it", "SILWARP does not contain it",
        "No `s` values are added", "not a converse", "source-only frozen candidate",
        "Frozen universality contract", "no Qwen-specific table", "There is no trainer",
        "Stage 0 is therefore not runnable as sealed", "108 independently decodable",
    ]
    for phrase in required:
        require(phrase.lower() in text.lower(), f"README boundary missing: {phrase}")


def verify_package(package: Path) -> dict[str, Any]:
    package = package.resolve(strict=True)
    require(package.is_dir() and not package.is_symlink(), "package directory")
    manifest = verify_manifest(package)
    design = strict_json_loads((package / "design_lock.json").read_bytes())
    arithmetic = verify_design(design)
    static = verify_static(package)
    verify_readme(package)
    return {
        "schema": "tactic_dh384_source_verification_v2",
        "status": "PASS_SOURCE_ONLY_NO_EXECUTION_AUTHORITY",
        **manifest,
        "arithmetic": arithmetic,
        "static": static,
        "claim_boundary": "Source closure/arithmetic/static verification only; no payload or GPU action.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = verify_package(args.package)
    print(json.dumps(result, sort_keys=True, separators=(",", ":") if args.compact else None,
                     indent=None if args.compact else 2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        raise SystemExit(f"source verification failed: {exc}") from exc
