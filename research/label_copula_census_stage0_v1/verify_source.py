#!/usr/bin/env python3
"""Independent-friendly verifier for the sealed label-copula source package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_COMMON_PATH = Path(os.path.abspath(__file__)).with_name("label_copula_common.py")
_COMMON_SPEC = importlib.util.spec_from_file_location("label_copula_common", _COMMON_PATH)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load same-directory label_copula_common.py")
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules["label_copula_common"] = _COMMON_MODULE
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)

from label_copula_common import (
    AUTHORIZATION,
    CONTEXT_COUNT,
    CONTROL_SEEDS,
    DESIGN_SCHEMA,
    MODEL_HEADER_BYTES,
    MIN_TEST_LAYERS,
    MIN_TOTAL_LAYERS,
    RESET_SYMBOLS,
    REVIEW_SCHEMA,
    STANDALONE_REQUIRED_SAVING_BPW,
    STATE_SIZES,
    TOPOLOGIES,
    Candidate,
    QuantizedModel,
    candidate_bank,
    model_ledger,
    require,
    reject_symlink_path_and_ancestors,
    sha256_file,
    strict_json_loads,
)


MANIFEST_NAME = "SOURCE_MANIFEST.json"
MANIFEST_SCHEMA = "label-copula-census-source-manifest-v1"
EXPECTED_FILES = {
    "README.md",
    "design_lock.json",
    "label_copula_common.py",
    "run_source_free_fixture.py",
    "stage0_census.py",
    "test_source_only.py",
    "verify_source.py",
}


def _all_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def _top_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def verify_manifest(package: Path) -> dict[str, Any]:
    path = package / MANIFEST_NAME
    require(path.is_file() and not path.is_symlink(), "source manifest missing")
    raw = path.read_bytes()
    record = strict_json_loads(raw)
    require(isinstance(record, dict) and record.get("schema") == MANIFEST_SCHEMA, "manifest schema")
    rows = record.get("files")
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_FILES), "manifest rows")
    require([row.get("name") for row in rows] == sorted(EXPECTED_FILES), "manifest sorted closure")
    require({path.name for path in package.iterdir()} == EXPECTED_FILES | {MANIFEST_NAME}, "exact package closure")
    hashes: dict[str, str] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest row schema")
        member = package / str(row["name"])
        require(member.is_file() and not member.is_symlink(), f"regular member {member.name}")
        require(member.stat().st_size == int(row["bytes"]), f"member bytes {member.name}")
        observed = sha256_file(member)
        require(observed == row["sha256"], f"member hash {member.name}")
        hashes[member.name] = observed
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "file_hashes": hashes}


def verify_design(package: Path) -> dict[str, Any]:
    design = strict_json_loads((package / "design_lock.json").read_bytes())
    require(design.get("schema") == DESIGN_SCHEMA, "design schema")
    require(design.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "design status")
    objective = design["objective"]
    require(math.isclose(
        float(objective["standalone_net_physical_saving_required_bpw"]),
        STANDALONE_REQUIRED_SAVING_BPW,
        rel_tol=0.0,
        abs_tol=2e-16,
    ), "standalone threshold")
    require(objective["gate_uses_source_lower_95_result_not_control_subtraction"] is True, "source-first lower-bound gate")
    scope = design["scope"]
    require(scope["canonical_orientation"] == "Gate[j,k],Up[j,k],Down[k,j]", "canonical orientation")
    require("not a complete" in scope["diagnostic_only"], "diagnostic boundary")
    require("deferred" in scope["deferred_stream_B"].lower(), "view B deferred")
    context = design["universal_decoder_context"]
    forbidden = " ".join(context["forbidden"]).lower()
    for phrase in ("model identity", "layer identity", "expert identity", "absolute tensor site", "future label"):
        require(phrase in forbidden, f"forbidden probability key {phrase}")
    bank = design["candidate_bank"]
    require(tuple(bank["topologies"]) == TOPOLOGIES, "topology lock")
    require(tuple(bank["state_sizes_chi"]) == STATE_SIZES, "state-size lock")
    require(tuple(bank["reset_symbols"]) == RESET_SYMBOLS, "reset lock")
    require(bank["cells"] == 240 and len(candidate_bank()) == 240, "bank closure")
    controls = design["controls"]
    require(tuple(controls["seeds"]) == CONTROL_SEEDS, "control seeds")
    require(controls["opened_or_generated_only_after_source_absolute_survival"] is True, "control order")
    require("never turn an absolute source miss into survival" in controls["no_control_created_pass"].lower(), "control no-rescue")
    require(design["nested_protocol"]["no_full_panel_selection"] is True, "no full-panel selection")
    require(design["lifecycle"]["authorization"] == AUTHORIZATION, "authorization binding")
    require(design["lifecycle"]["v1_payload_authority"] is False, "no payload authority")
    require(design["lifecycle"]["v1_claim_authority"] is False, "no claim authority")
    require(design["nested_protocol"]["minimum_total_layers"] == MIN_TOTAL_LAYERS, "minimum total layers")
    require(design["nested_protocol"]["minimum_test_layers"] == MIN_TEST_LAYERS, "minimum test clusters")
    require(design["controls"]["complete_nonlocal_candidate_cells_per_control"] == 240, "complete control search")
    require(all(value is False for value in design["access_attestation"].values()), "source-only access attestation")
    ledgers = [model_ledger(candidate) for candidate in candidate_bank()]
    require(all(row["physical_model_bytes"] == MODEL_HEADER_BYTES + 2 * CONTEXT_COUNT * int(row["chi"]) for row in ledgers), "model ledgers")
    return {
        "candidate_cells": len(ledgers),
        "smallest_model_bytes": min(int(row["physical_model_bytes"]) for row in ledgers),
        "largest_model_bytes": max(int(row["physical_model_bytes"]) for row in ledgers),
    }


def verify_static(package: Path) -> dict[str, Any]:
    common_text = (package / "label_copula_common.py").read_text(encoding="utf-8")
    stage_text = (package / "stage0_census.py").read_text(encoding="utf-8")
    fixture_text = (package / "run_source_free_fixture.py").read_text(encoding="utf-8")
    common_tree = ast.parse(common_text, filename="label_copula_common.py")
    stage_tree = ast.parse(stage_text, filename="stage0_census.py")
    fixture_tree = ast.parse(fixture_text, filename="run_source_free_fixture.py")
    forbidden = {"numpy", "torch", "tensorflow", "jax", "requests", "urllib", "socket", "paramiko"}
    require(not (_all_imports(common_tree) & forbidden), "common forbidden dependency")
    require(not (_all_imports(stage_tree) & forbidden), "stage forbidden dependency")
    require(not (_all_imports(fixture_tree) & forbidden), "fixture forbidden dependency")
    require("cupy" not in _top_imports(stage_tree), "CuPy top-level import")
    bad = stage_text.index("authorization mismatch")
    output = stage_text.index("with CompletionLastOutput")
    source = stage_text.index("source = _dynamic_verify_source()")
    review = stage_text.index("review = _review")
    metadata = stage_text.index("metadata = _input_metadata")
    cupy = stage_text.index("import cupy as cp")
    require(bad < output < source < review < metadata < cupy, "fail-closed launch order")
    require("payloads_opened\": 0" in stage_text and "cuda_kernels_launched\": 0" in stage_text, "zero-access preflight")
    require("output.complete(manifest_sha256)" in stage_text, "completion last call")
    require("entrypoint_sha256" in stage_text and "payload_authority\": False" in stage_text, "entrypoint binding/no authority")
    require("def next_state" in common_text and "parity_sketch" in common_text, "unifilar topology implementation")
    require("def encode_stream" in common_text and "def decode_stream" in common_text, "finite arithmetic roundtrip surface")
    require("def nested_partition" in common_text and "select_on_validation" in common_text, "nested selection surface")
    require("control_panels" not in common_text, "arbitrary prebuilt control-panel interface forbidden")
    require("build_matched_gaussian_control_panel" in common_text, "bound Gaussian-control generation")
    require("B_shared/E + B_private_i" in common_text, "per-expert cold denominator")
    require("not self._completed" in common_text, "irrevocable completion state")
    require("layer_group" not in common_text[common_text.index("def public_context"):common_text.index("@dataclass(frozen=True, order=True)")], "identity-free context")
    return {
        "common_ast_nodes": sum(1 for _ in ast.walk(common_tree)),
        "stage_ast_nodes": sum(1 for _ in ast.walk(stage_tree)),
        "stage_imports": sorted(_all_imports(stage_tree)),
    }


def verify_integer_packet() -> dict[str, Any]:
    candidate = Candidate("parity_sketch", 64, 4096)
    frequencies = tuple(32768 for _ in range(CONTEXT_COUNT * candidate.chi))
    model = QuantizedModel(candidate, frequencies)
    packet = model.serialize()
    restored = QuantizedModel.deserialize(packet)
    require(restored == model, "integer model packet roundtrip")
    return {"packet_bytes": len(packet), "packet_sha256": hashlib.sha256(packet).hexdigest()}


def verify_readme(package: Path) -> None:
    text = (package / "README.md").read_text(encoding="utf-8").lower()
    required = [
        "0.1528899669629145",
        "gate[j,k], up[j,k], down[k,j]",
        "not yet a complete 2.15–2.5 bpw codec",
        "view b",
        "exact-integer unifilar",
        "exactly 240",
        "no control-created pass",
        "whole-layer",
        "checkpoint/model identity",
        "complete.json",
        "source-only",
        "cupy-only",
        "does not close arbitrary",
        "identical reusable",
        "five test-layer clusters",
        "immutable raw swiglu panel",
        "all 240",
        "b_shared/e + b_private_i",
        "no payload authority and no claim authority",
        "independently pinned bootstrap",
    ]
    for phrase in required:
        require(phrase in text, f"README boundary missing: {phrase}")


def verify_package(package: Path) -> dict[str, Any]:
    package = reject_symlink_path_and_ancestors(package)
    require(package.is_dir() and not package.is_symlink(), "package directory")
    manifest = verify_manifest(package)
    design = verify_design(package)
    static = verify_static(package)
    packet = verify_integer_packet()
    verify_readme(package)
    return {
        "schema": "label-copula-census-source-verification-v1",
        "status": "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY",
        **manifest,
        "design": design,
        "static": static,
        "integer_packet": packet,
        "review_schema": REVIEW_SCHEMA,
        "authorization": AUTHORIZATION,
        "claim_boundary": "Source closure/static/integer packet only; no checkpoint/current stream/control/CuPy/CUDA action.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(os.path.abspath(__file__)).parent)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = verify_package(args.package)
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
