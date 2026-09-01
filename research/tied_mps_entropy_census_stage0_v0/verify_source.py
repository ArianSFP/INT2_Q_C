#!/usr/bin/env python3
"""Independent-friendly source-only verifier for tied MPS census v0."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

_COMMON_PATH = Path(__file__).resolve().with_name("mps_common.py")
_COMMON_SPEC = importlib.util.spec_from_file_location("mps_common", _COMMON_PATH)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load sealed same-directory mps_common.py")
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules["mps_common"] = _COMMON_MODULE
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)

from mps_common import (
    AUTHORIZATION,
    CONTROL_SEEDS,
    DESIGN_SCHEMA,
    EM_ITERATIONS,
    FIT_SEEDS,
    HIDDEN_DIMENSIONS,
    PERIODS,
    REVIEW_SCHEMA,
    STANDALONE_REQUIRED_SAVING_BPW,
    SUFFIX_DEPTHS,
    TARGET_F,
    hmm_model_ledger,
    require,
    sha256_file,
    strict_json_loads,
    suffix_model_ledger,
)


MANIFEST_NAME = "SOURCE_MANIFEST.json"
EXPECTED_FILES = {
    "README.md",
    "design_lock.json",
    "evidence_bindings.json",
    "mps_common.py",
    "stage0_census.py",
    "test_source_only.py",
    "verify_source.py",
}


def _top_imports(tree: ast.AST) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def _all_imports(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def verify_manifest(package: Path) -> dict[str, Any]:
    path = package / MANIFEST_NAME
    require(path.is_file() and not path.is_symlink(), "source manifest missing")
    raw = path.read_bytes()
    data = strict_json_loads(raw)
    require(data.get("schema") == "tied-mps-entropy-census-source-manifest-v0", "manifest schema")
    rows = data.get("files")
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_FILES), "manifest rows")
    require([row["name"] for row in rows] == sorted(EXPECTED_FILES), "manifest order")
    require({entry.name for entry in package.iterdir()} == EXPECTED_FILES | {MANIFEST_NAME}, "exact package closure")
    hashes = {}
    for row in rows:
        require(set(row) == {"name", "bytes", "sha256"}, "manifest row schema")
        member = package / row["name"]
        require(member.is_file() and not member.is_symlink(), f"regular member {row['name']}")
        require(member.stat().st_size == int(row["bytes"]), f"member bytes {row['name']}")
        observed = sha256_file(member)
        require(observed == row["sha256"], f"member hash {row['name']}")
        hashes[row["name"]] = observed
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "file_hashes": hashes}


def verify_evidence(package: Path) -> dict[str, Any]:
    bindings = strict_json_loads((package / "evidence_bindings.json").read_bytes())
    require(bindings.get("schema") == "tied-mps-entropy-census-evidence-bindings-v0", "evidence schema")
    rows = bindings.get("rows")
    require(isinstance(rows, list) and len(rows) == 9, "evidence rows")
    repo = package.parents[1]
    checked = []
    for row in rows:
        require(set(row) == {"path", "bytes", "sha256", "use"}, "evidence row schema")
        path = repo / row["path"]
        require(path.is_file() and not path.is_symlink(), f"evidence regular file {row['path']}")
        require(path.stat().st_size == int(row["bytes"]), f"evidence bytes {row['path']}")
        require(sha256_file(path) == row["sha256"], f"evidence hash {row['path']}")
        checked.append(row["path"])
    access = bindings.get("seal_time_access")
    require(access == {
        "bound_metadata_artifacts_opened_or_hashed": 9,
        "model_payloads": 0,
        "decoded_finite_streams": 0,
        "control_streams": 0,
        "gpu_jobs": 0,
    }, "evidence access attestation")
    return {"checked": checked}


def verify_design(package: Path) -> dict[str, Any]:
    design = strict_json_loads((package / "design_lock.json").read_bytes())
    require(design.get("schema") == DESIGN_SCHEMA, "design schema")
    require(design.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "design status")
    objective = design["objective"]
    required = -0.5 * math.log2(TARGET_F)
    require(math.isclose(objective["total_required_s_bpw"], required, rel_tol=0.0, abs_tol=2e-16), "total s")
    require(math.isclose(objective["standalone_net_physical_saving_required_bpw"], STANDALONE_REQUIRED_SAVING_BPW, rel_tol=0.0, abs_tol=2e-16), "standalone gap")
    require(objective["speculative_composite_gap_bpw_not_applicable_to_this_lossless_cell"] == 0.11356063457, "composite distinction")
    a0 = design["gate_A0_local_subclass"]
    require(tuple(a0["suffix_depths"]) == SUFFIX_DEPTHS, "A0 depths")
    require("cannot close" in a0["claim_boundary"], "A0 local boundary")
    a1 = design["gate_A1_true_hidden_state"]
    require(tuple(a1["hidden_dimensions_chi"]) == HIDDEN_DIMENSIONS, "A1 chi")
    require(a1["fit"]["iterations"] == EM_ITERATIONS, "EM iterations")
    require(tuple(a1["fit"]["seeds"]) == FIT_SEEDS, "EM seeds")
    require("A_(c,y)[i,j]" in a1["matrix_definition"], "symbol-conditioned matrices")
    require("alpha_" in a1["causal_probability"] and "prefix" in a1["causal_probability"], "causal probability")
    require(a1["physical_parameters"]["all_tensors_and_rounding_are_decoder_visible"] is True, "decoder-visible tensors")
    context = design["public_decoder_context"]
    require(tuple(context["allowed_periods"]) == PERIODS, "public periods")
    forbidden = " ".join(context["forbidden_keys"]).lower()
    for phrase in ("model identity", "layer identity", "expert identity", "absolute site", "future decisions"):
        require(phrase in forbidden, f"forbidden context {phrase}")
    controls = design["gaussian_controls"]
    require(tuple(controls["seeds"]) == CONTROL_SEEDS, "control seeds")
    require(controls["opened_only_after_absolute_source_survival"] is True, "control order")
    require(
        controls["model_refit"].startswith("the source-selected A1 chi and P are independently refit"),
        "control refit",
    )
    critique = design["information_identities_and_proposal_critique"]
    require("H(q)-H(q|b)=H(b)" in critique["deterministic_statistic"], "deterministic statistic identity")
    require(
        critique["Gray_Wyner_evidence"]["prior_same_layer_favourable_capture"] == 0.016534903625203354,
        "Gray-Wyner evidence",
    )
    access = design["access_attestation"]
    require(all(value is False for value in access.values()), "source-only access attestation")
    require(design["lifecycle"]["authorization"] == AUTHORIZATION, "authorization binding")
    ledgers = []
    for period in PERIODS:
        for depth in SUFFIX_DEPTHS:
            row = suffix_model_ledger(depth, period)
            require(row["physical_model_bytes"] == 256 + 2 * row["contexts"] * row["states"], "suffix bytes")
        for chi in HIDDEN_DIMENSIONS:
            row = hmm_model_ledger(chi, period)
            require(row["physical_model_bytes"] == 256 + 2 * (chi + chi * chi + row["contexts"] * chi), "HMM bytes")
            ledgers.append(row)
    return {"model_cells": len(ledgers), "largest_model_bytes": max(row["physical_model_bytes"] for row in ledgers)}


def verify_static(package: Path) -> dict[str, Any]:
    common_text = (package / "mps_common.py").read_text(encoding="utf-8")
    stage_text = (package / "stage0_census.py").read_text(encoding="utf-8")
    common_tree = ast.parse(common_text, filename="mps_common.py")
    stage_tree = ast.parse(stage_text, filename="stage0_census.py")
    forbidden = {"numpy", "torch", "tensorflow", "requests", "urllib", "socket"}
    require(not (_all_imports(common_tree) & forbidden), "common forbidden dependency")
    require(not (_all_imports(stage_tree) & forbidden), "stage forbidden dependency")
    require("cupy" not in _top_imports(stage_tree), "CuPy top import")
    require(stage_text.index("authorization mismatch") < stage_text.index("import cupy as cp"), "authorization before CuPy")
    source_call = stage_text.index("source = _dynamic_verify_source()")
    require(stage_text.index("with CompletionLastOutput(output_path)") < source_call, "output before verifier")
    require(source_call < stage_text.index("import cupy as cp"), "source verification before CuPy")
    require(stage_text.index("_review(review_file") < stage_text.index("import cupy as cp"), "review before CuPy")
    require("_controls_after_survival" in stage_text and "if a1[\"status\"] == \"SURVIVE_EXACT_SOURCE_REQUIRES_HOLDOUT_AND_CONTROLS\"" in stage_text, "controls survivor gate")
    require("HeldFileSet" in stage_text and "held_files.verify_stable()" in stage_text, "held descriptors")
    require("fit_hmm" in stage_text and "_expectation_batch" in stage_text, "true HMM fit")
    require("model.transition" in stage_text and "model.emission1" in stage_text, "learned transition/emission")
    require("_crossfit_selected" in stage_text and "layer_group !=" in stage_text and "expert_group !=" in stage_text, "disjoint holdout")
    require("arithmetic_encode_binary" in stage_text and "quantize_hmm" in stage_text, "finite quantized replay")
    require("payload == trace.original_payload" in stage_text, "byte-exact original payload replay")
    require("_trace_inventory_sha256" in stage_text and "extraction inventory binding" in stage_text, "inventory binding")
    require("persistent-regime" in (package / "design_lock.json").read_text(encoding="utf-8"), "HMM subclass boundary")
    require("output.complete(" in stage_text and "COMPLETE.json" in common_text, "completion last")
    return {
        "common_ast_nodes": sum(1 for _ in ast.walk(common_tree)),
        "stage_ast_nodes": sum(1 for _ in ast.walk(stage_tree)),
        "stage_imports": sorted(_all_imports(stage_tree)),
    }


def verify_readme(package: Path) -> None:
    text = (package / "README.md").read_text(encoding="utf-8").lower()
    required = [
        "0.1528899669629145", "0.11356063457", "local-only", "chi={4,8,16,32,64}",
        "symbol-conditioned", "whole-layer", "independently encoded gaussian", "no control subtraction",
        "h(q)-h(q|b)=h(b)", "gray--wyner", "rcc", "fiber", "complete.json",
        "not a global hmm-mle proof", "no model payload", "separate independent source review",
    ]
    for phrase in required:
        require(phrase in text, f"README boundary missing: {phrase}")


def verify_package(package: Path) -> dict[str, Any]:
    package = package.resolve(strict=True)
    require(package.is_dir() and not package.is_symlink(), "package directory")
    manifest = verify_manifest(package)
    design = verify_design(package)
    evidence = verify_evidence(package)
    static = verify_static(package)
    verify_readme(package)
    return {
        "schema": "tied-mps-entropy-census-source-verification-v0",
        "status": "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY",
        **manifest,
        "design": design,
        "evidence": evidence,
        "static": static,
        "review_schema": REVIEW_SCHEMA,
        "authorization": AUTHORIZATION,
        "claim_boundary": "Source closure, arithmetic and static verification only; no decoded stream, model payload, control or CUDA action.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = verify_package(args.package)
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
