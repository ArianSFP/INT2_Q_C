#!/usr/bin/env python3
"""Independent, payload-free hostile audit for label-copula stage0 v1.

The producer package is hashed before any of its Python is imported.  This
program never accepts or resolves a model/checkpoint/current-codec/control
payload.  All dynamic probes use synthetic in-memory streams or temporary
copies of the already authenticated producer source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PINNED_MANIFEST_SHA256 = "81fc371df5db5f654815d9fe0c673c34cb403d959b313425ae91c09c37fc5bd7"
EXPECTED_MEMBERS = {
    "README.md",
    "design_lock.json",
    "label_copula_common.py",
    "run_source_free_fixture.py",
    "stage0_census.py",
    "test_source_only.py",
    "verify_source.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            value.update(block)
    return value.hexdigest()


def authenticate(package: Path) -> dict[str, Any]:
    manifest_path = package / "SOURCE_MANIFEST.json"
    manifest_raw = manifest_path.read_bytes()
    observed = hashlib.sha256(manifest_raw).hexdigest()
    require(observed == PINNED_MANIFEST_SHA256, "pinned producer manifest mismatch")
    manifest = json.loads(manifest_raw)
    rows = manifest["files"]
    require({row["name"] for row in rows} == EXPECTED_MEMBERS, "producer member closure")
    require({path.name for path in package.iterdir()} == EXPECTED_MEMBERS | {"SOURCE_MANIFEST.json"}, "producer directory closure")
    for row in rows:
        member = package / row["name"]
        require(member.is_file() and not member.is_symlink(), f"producer regular member: {row['name']}")
        require(member.stat().st_size == row["bytes"], f"producer byte mismatch: {row['name']}")
        require(digest(member) == row["sha256"], f"producer digest mismatch: {row['name']}")
    return {"source_manifest_sha256": observed, "authenticated_members": len(rows)}


def import_exact_common(package: Path):
    path = package / "label_copula_common.py"
    spec = importlib.util.spec_from_file_location("label_copula_audited_common", path)
    require(spec is not None and spec.loader is not None, "common import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_stream(common, layer: str, slot: str, salt: int, weights: int = 257):
    count = 2 * weights
    symbols = tuple((((index * 13 + salt * 7) ^ (index >> 1) ^ (index >> 5)) & 1) for index in range(count))
    roles = tuple((index // 2 + salt) % 3 for index in range(count))
    planes = tuple(index & 1 for index in range(count))
    return common.SymbolStream(layer, slot, symbols, roles, planes, weights)


def exhaustive_integer_codec(common) -> dict[str, Any]:
    training = tuple(tiny_stream(common, "train", f"slot-{index}", index) for index in range(3))
    target = tiny_stream(common, "test", "slot-x", 17, 389)
    cells = common.candidate_bank() + common.factorized_bank()
    packet_hash = hashlib.sha256()
    for candidate in cells:
        model = common.fit_model(training, candidate)
        packet = model.serialize()
        restored = common.QuantizedModel.deserialize(packet)
        require(restored == model, f"serialized model mismatch: {candidate}")
        for frequency in restored.freq1:
            require(1 <= frequency < common.Q_TOTAL, f"Q0.16 range: {candidate}")
            require(frequency + (common.Q_TOTAL - frequency) == common.Q_TOTAL, f"Q0.16 row sum: {candidate}")
        payload, meaningful = common.encode_stream(restored, target)
        decoded = common.decode_stream(restored, target.roles, target.planes, payload, meaningful)
        require(decoded == target.symbols, f"serialized arithmetic roundtrip: {candidate}")
        packet_hash.update(hashlib.sha256(packet).digest())
    return {
        "cells": len(cells),
        "nonlocal_cells": len(common.candidate_bank()),
        "factorized_cells": len(common.factorized_bank()),
        "all_packets_and_frames_roundtrip_from_deserialized_models": True,
        "packet_digest_accumulator_sha256": packet_hash.hexdigest(),
    }


def split_and_slot_checks(common) -> dict[str, Any]:
    panel = tuple(
        tiny_stream(common, f"layer-{layer}", f"slot-{slot}", 10 * layer + slot, 32)
        for layer in range(12)
        for slot in range(5)
    )
    folds = common.nested_partition(panel)
    train_layers = {row.layer_group for row in folds["train"]}
    test_layers = {row.layer_group for row in folds["test"]}
    train_slots = {row.expert_group for row in folds["train"]}
    validation_slots = {row.expert_group for row in folds["validation"]}
    require(not train_layers & test_layers, "whole-layer leakage")
    require(not train_slots & validation_slots, "whole-slot leakage")
    require(len(test_layers) >= 5, "test cluster minimum")
    irregular = panel[:-1]
    rejected = False
    try:
        common.nested_partition(irregular)
    except common.ContractError:
        rejected = True
    require(rejected, "irregular reusable-slot universe accepted")
    return {
        "train_layers": len(train_layers),
        "test_layers": len(test_layers),
        "train_slots": len(train_slots),
        "validation_slots": len(validation_slots),
        "irregular_panel_rejected": rejected,
    }


def ledger_checks(common) -> dict[str, Any]:
    training = tuple(tiny_stream(common, "train", f"slot-{index}", index, 512) for index in range(3))
    model = common.QuantizedModel.deserialize(
        common.fit_model(training, common.Candidate("rolling", 64, 4096)).serialize()
    )
    streams = tuple(
        tiny_stream(common, "test", f"slot-{index}", 31 + index, weights)
        for index, weights in enumerate((1, 31, 2049, 5003, 8191))
    )
    encoded = tuple((row, *common.encode_stream(model, row)) for row in streams)
    ledger = common.container_ledger(len(model.serialize()), encoded)
    offset = common.CONTAINER_HEADER_BYTES + ledger["stored_model_page_bytes"] + ledger["stored_directory_page_bytes"]
    denominators = 0.0
    for row, (_, payload, meaningful) in zip(ledger["frame_rows"], encoded, strict=True):
        require(row["frame_offset"] == offset, "frame offset")
        frame_bytes = common.FRAME_HEADER_BYTES + len(payload)
        stored = common.align_up(frame_bytes, common.FRAME_ALIGNMENT)
        require(row["frame_bytes"] == frame_bytes and row["stored_frame_bytes"] == stored, "frame byte ledger")
        pages = set(row["header_page_indices"]) | set(row["model_page_indices"]) | set(row["addressed_directory_page_indices"]) | set(row["frame_page_indices"])
        require(pages == set(row["cold_page_indices"]), "cold page union")
        require(row["cold_read_bytes"] == len(pages) * common.PAGE_BYTES, "cold page bytes")
        require(len(payload) == (meaningful + 7) // 8, "arithmetic byte padding")
        denominators += row["physical_denominator_bytes"]
        offset += stored
    require(ledger["total_physical_bytes"] == common.page_ceil(offset), "final physical page rounding")
    require(math.isclose(denominators, ledger["total_physical_bytes"], rel_tol=0.0, abs_tol=1e-9), "denominator closure")
    return {
        "synthetic_experts": len(streams),
        "total_physical_bytes": ledger["total_physical_bytes"],
        "denominator_closes": True,
        "literal_page_union_math_closes": True,
    }


def synthetic_raw_panel(common):
    rows = []
    for layer in range(common.MIN_TOTAL_LAYERS):
        for slot in range(3):
            x = (layer * 3 + slot + 1) / 97.0
            rows.append(common.RawSwiGLUExpert(
                layer_group=f"layer-{layer}",
                expert_group=f"slot-{slot}",
                gate=((x,),),
                up=((-2.0 * x,),),
                down=((3.0 * x,),),
            ))
    return tuple(rows)


def one_cell_scientific_override_counterexample(common) -> dict[str, Any]:
    candidate = common.Candidate("suffix", 2, 32)
    result = common.evaluate_raw_source_panel(synthetic_raw_panel(common), (candidate,))
    require(result["schema"] == common.RESULT_SCHEMA, "one-cell result schema")
    require(result["nonlocal_candidate_cells"] == 1, "one-cell override was unexpectedly rejected")
    require(len(result["selection_rows"]) == 1, "one-cell selection row")
    return {
        "counterexample_observed": True,
        "returned_scientific_result_schema": result["schema"],
        "accepted_nonlocal_candidate_cells": result["nonlocal_candidate_cells"],
        "frozen_requirement_cells": 240,
    }


def serialized_model_bypass_counterexample(common) -> dict[str, Any]:
    panel = common.synthetic_parity_streams(
        layers=common.MIN_TOTAL_LAYERS,
        experts=3,
        blocks_per_stream=1,
        seed=424242,
        constrained=True,
    )
    original = common.QuantizedModel.serialize

    def invalid_packet(self):
        self.validate()
        return bytes(common.MODEL_HEADER_BYTES + 2 * common.CONTEXT_COUNT * self.candidate.chi)

    common.QuantizedModel.serialize = invalid_packet
    try:
        result = common.evaluate_nested(panel, (common.Candidate("suffix", 2, 32),))
        sample = common.fit_model(panel[:3], common.Candidate("suffix", 2, 32)).serialize()
        deserialize_rejected = False
        try:
            common.QuantizedModel.deserialize(sample)
        except common.ContractError:
            deserialize_rejected = True
    finally:
        common.QuantizedModel.serialize = original
    require(deserialize_rejected, "invalid serialized packet accepted")
    require(result["schema"] == common.RESULT_SCHEMA, "in-memory bypass did not return a scored result")
    return {
        "counterexample_observed": True,
        "serialized_packet_is_invalid": deserialize_rejected,
        "evaluate_nested_still_returned_scored_result": True,
        "cause": "encode_panel charges only len(model.serialize()); scored decode_stream receives the original in-memory model",
    }


def preauthorization_execution_counterexample(package: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clone = root / "producer-clone"
        shutil.copytree(package, clone)
        marker = root / "executed-before-authorization.txt"
        common_path = clone / "label_copula_common.py"
        text = common_path.read_text(encoding="utf-8")
        needle = "from __future__ import annotations\n"
        injection = (
            needle
            + "import os as _audit_os\n"
            + "from pathlib import Path as _AuditPath\n"
            + "_AuditPath(_audit_os.environ['LC_AUDIT_MARKER']).write_text('executed', encoding='ascii')\n"
        )
        require(text.count(needle) == 1, "future-import injection point")
        common_path.write_text(text.replace(needle, injection, 1), encoding="utf-8", newline="\n")
        output = root / "must-not-exist"
        absent = root / "absent.json"
        environment = dict(os.environ)
        environment["LC_AUDIT_MARKER"] = str(marker)
        run = subprocess.run(
            [
                sys.executable,
                "-B",
                "-I",
                str(clone / "stage0_census.py"),
                "--authorization",
                "WRONG",
                "--review-receipt",
                str(absent),
                "--input-lock",
                str(absent),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
            check=False,
        )
        require(marker.is_file(), "same-directory producer code did not execute before wrong-token branch")
        require(not output.exists(), "wrong-token output unexpectedly created")
        return {
            "counterexample_observed": True,
            "wrong_token_returncode": run.returncode,
            "unmanifested_common_code_executed_before_token_check": True,
            "output_created": False,
        }


def unauthenticated_review_counterexample(package: Path) -> dict[str, Any]:
    stage_path = package / "stage0_census.py"
    spec = importlib.util.spec_from_file_location("label_copula_audited_stage", stage_path)
    require(spec is not None and spec.loader is not None, "stage import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest_sha = PINNED_MANIFEST_SHA256
    entry_sha = digest(stage_path)
    forged = {
        "schema": module.REVIEW_SCHEMA,
        "status": "PASS_INDEPENDENT_SOURCE_REVIEW",
        "source_manifest_sha256": manifest_sha,
        "entrypoint_sha256": entry_sha,
        "payloads_opened": 0,
        "cuda_jobs": 0,
        "payload_authority": False,
        "receipt_sha256": "attacker-controlled-string",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "forged-review.json"
        path.write_text(json.dumps(forged), encoding="ascii")
        accepted = module._review(path, manifest_sha, entry_sha)
    require(accepted == forged, "synthetic unsigned review was not accepted as expected")
    return {
        "counterexample_observed": True,
        "unsigned_caller_authored_review_accepted": True,
        "signature_or_external_trust_root_checked": False,
        "producer_declares_payload_authority": False,
    }


def static_contract_findings(common) -> dict[str, Any]:
    raw_signature = tuple(inspect.signature(common.evaluate_raw_source_panel).parameters)
    control_builder_signature = tuple(inspect.signature(common.build_matched_gaussian_control_panel).parameters)
    controls_text = inspect.getsource(common.evaluate_independent_matched_controls)
    builder_text = inspect.getsource(common.build_matched_gaussian_control_panel)
    completion_text = inspect.getsource(common.CompletionLastOutput)
    require("full_bank = candidate_bank()" in controls_text, "controls do not freeze full bank")
    require("evaluate_nested(panel, full_bank)" in controls_text, "controls do not rerun full nested path")
    return {
        "raw_source_api_parameters": raw_signature,
        "raw_source_bank_override_exposed": "bank" in raw_signature,
        "control_builder_parameters": control_builder_signature,
        "control_builder_has_source_survival_argument": "source_result" in control_builder_signature,
        "control_builder_body_has_gate": "matched_control_gate" in builder_text,
        "control_evaluator_reruns_full_240_bank": True,
        "control_result_has_source_specific_excess_point": "source_specific_excess_bpw" in controls_text,
        "control_result_has_source_specific_confidence_gate": "source_specific_lower" in controls_text or "source_specific_survival" in controls_text,
        "completion_writer_rejects_post_complete_method_calls": "not self._completed" in completion_text,
        "literal_container_writer_present": hasattr(common, "write_container") or hasattr(common, "serialize_container"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = Path(os.path.abspath(args.package))
    authentication = authenticate(package)
    common = import_exact_common(package)
    result = {
        "schema": "label-copula-census-stage0-v1-independent-source-audit-probe-v1",
        "status": "BLOCK_SOURCE_PREFLIGHT_READINESS",
        "payloads_opened": 0,
        "model_or_checkpoint_payloads_opened": 0,
        "current_codec_payloads_opened": 0,
        "control_payloads_opened_or_generated": 0,
        "network_used": False,
        "authentication": authentication,
        "passing_mechanism_checks": {
            "complete_integer_codec": exhaustive_integer_codec(common),
            "split_and_slots": split_and_slot_checks(common),
            "ledger_math": ledger_checks(common),
        },
        "blocking_counterexamples": {
            "scientific_source_bank_override": one_cell_scientific_override_counterexample(common),
            "serialized_model_decode_bypass": serialized_model_bypass_counterexample(common),
            "producer_code_executes_before_authorization": preauthorization_execution_counterexample(package),
            "review_receipt_has_no_external_authentication": unauthenticated_review_counterexample(package),
        },
        "static_contract_findings": static_contract_findings(common),
        "claim_boundary": "Source-only hostile audit. BLOCK is not evidence about Qwen weights and grants no payload authority.",
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
