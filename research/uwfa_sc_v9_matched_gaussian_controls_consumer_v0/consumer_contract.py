#!/usr/bin/env python3
"""Pure contracts for the v9 full-PTQ matched-Gaussian consumer/auditor.

Importing this module opens no path and imports no numeric or CUDA package.
All dynamic bytes must be supplied by an independently authenticated
dispatcher after the primary authorization and run pins have passed.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import struct
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "uwfa-sc-v9-matched-gaussian-controls-consumer-v0"
PRIMARY_STATUS = "PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED"
CONTROL_SEEDS = (
    10619863,
    10619881,
    10619909,
    10619927,
    10619953,
    10619971,
    10619999,
    10620017,
)
ROLES = ("gate", "up", "down")
EXPERTS = 6
HIDDEN = 2048
INTERMEDIATE = 768
VALUES_PER_MATRIX = HIDDEN * INTERMEDIATE
WEIGHTS = 3 * EXPERTS * VALUES_PER_MATRIX
ARTIFACT_BYTES = 8_847_360
SOURCE_BYTES_PER_MATRIX = 2 * VALUES_PER_MATRIX
SOURCE_CHUNK_BYTES = 2 * (1 << 18)
ALL150 = 150
PRODUCER_SOURCE_MANIFEST_SHA256 = "20cd2cd8b2a0e41f68e5fcf58a1b2ebe8d0e09c984bdbbd786a1057e869c9eb1"
V8_SOURCE_MANIFEST_SHA256 = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"
V9_PRIMARY_SOURCE_MANIFEST_SHA256 = "d1e3eaff6762df2e273f6e3f4216ff9110abe74a7534a0098544a4ceef632c5e"
V9_PRIMARY_RUNNER_SHA256 = "d1ff04ce3c2cc36208e464eaed943d6c94eb91a47e9d3c460b2d562b7162cc4d"

SYMMETRIC_CODEC_CLOSURE_FIELDS = (
    "runtime_snapshot_root_sha256",
    "numpy_runtime_closure_sha256",
    "cupy_runtime_closure_sha256",
    "polar_repository_tree_sha256",
    "strata_v2_emitter_sha256",
    "strata_v2_common_sha256",
    "strata_polar_wrapper_sha256",
    "polaris_base_encoder_sha256",
    "bec_encoder_sha256",
    "expert_common_sha256",
    "run_and_pack_sha256",
    "independent_auditor_sha256",
    "frozen_auditor_sha256",
    "v8_adapter_sha256",
    "v8_protocol_sha256",
    "v8_common_sha256",
)

V8_RUNTIME_CLOSURE_FIELDS = (
    "v8_source_manifest_sha256",
    "v8_uwfa_common_sha256",
    "v8_stage0_census_sha256",
    "v8_protocol_sha256",
    "v8_container_codec_sha256",
    "v8_cupy_backend_sha256",
    "v8_strata_adapter_sha256",
    "v8_universal_adapter_sha256",
)

V8_RUNTIME_EXPECTED = {
    "v8_source_manifest_sha256": "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6",
    "v8_uwfa_common_sha256": "db53567ab6d71d5150cc92ef4a78fa9ce5cca01f5474fa2ca32edc8711cc4325",
    "v8_stage0_census_sha256": "7b7c2e0fcb6593805e6b2c8234ae59cb42d90fbb7dcf945a35aa5dfe331ae618",
    "v8_protocol_sha256": "9e18675a1e646eb10c0900aa3767bff96666943309dbd8db3953c745888d2cc1",
    "v8_container_codec_sha256": "645debb547a76818a880bfc346a2dd6230af97b07dc832afb3548a83d6920fed",
    "v8_cupy_backend_sha256": "7904a5e122686487d89fb684b70052507089bfe3bbfe4f1f02520df6ce3fb1ba",
    "v8_strata_adapter_sha256": "08fc8808ac168f6930ee9482e160f25f2bd087829fca4630553aea3510d722c6",
    "v8_universal_adapter_sha256": "a5ab2e1919af98c2aa9b3032faa0ba5552efe05cca250bd6844fd48c76aabbc8",
}

SOURCE_CLOSURE_FIELDS = (
    "source_artifact_sha256",
    "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "source_pipeline_sha256",
    "source_score_receipt_sha256",
    "source_moment_auditor_sha256",
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64, f"{label} digest width")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ContractError(f"{label} digest encoding") from exc
    require(len(raw) == 32 and value == value.lower(), f"{label} canonical digest")
    return value


def exact_int(value: Any, label: str, low: int = 0, high: int = (1 << 63) - 1) -> int:
    require(type(value) is int and low <= value <= high, label)
    return value


def finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float) and not isinstance(value, bool), label)
    observed = float(value)
    require(math.isfinite(observed), f"{label} finite")
    if positive:
        require(observed > 0.0, f"{label} positive")
    return observed


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        require(name not in result, f"duplicate JSON field {name}")
        result[name] = value
    return result


def _json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant {value}")


def _json_float(value: str) -> float:
    observed = float(value)
    require(math.isfinite(observed), "finite JSON float")
    return observed


def strict_json(payload: bytes, label: str, *, maximum: int = 64 << 20) -> dict[str, Any]:
    require(isinstance(payload, bytes) and 0 < len(payload) <= maximum, f"{label} bytes")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_json_pairs,
            parse_constant=_json_constant,
            parse_float=_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    clean = dict(value)
    require(field not in clean, f"duplicate seal {field}")
    clean[field] = sha256(canonical_json(clean))
    return clean


def validate_seal(value: Mapping[str, Any], field: str) -> None:
    claimed = digest(value.get(field), field)
    clean = dict(value)
    clean.pop(field)
    require(sha256(canonical_json(clean)) == claimed, f"{field} integrity")


def universal_format_geometry() -> dict[str, Any]:
    logn = [21, 21] * 6 + [20, 20, 20]
    owners = [[slot] for slot in range(6) for _ in range(2)] + [[0, 1], [2, 3], [4, 5]]
    return {
        "schema": "uwfa-sc-v9-universal-strata15-format-geometry-v1",
        "experts": EXPERTS,
        "weights": WEIGHTS,
        "expert_shapes": [
            {"slot": slot, "hidden": HIDDEN, "intermediate": INTERMEDIATE}
            for slot in range(EXPERTS)
        ],
        "roles": list(ROLES),
        "blocks": [
            {"ordinal": ordinal, "logn": size, "owner_slots": owner}
            for ordinal, (size, owner) in enumerate(zip(logn, owners, strict=True))
        ],
        "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
    }


def universal_format_geometry_sha256() -> str:
    return sha256(canonical_json(universal_format_geometry()))


def validate_universal_geometry(value: Any) -> dict[str, Any]:
    require(value == universal_format_geometry(), "pre-frozen universal format geometry")
    encoded = canonical_json(value).lower()
    for forbidden in (b"profile", b"symbol", b"label", b"payload", b"logical_bits", b"qwen", b"model.layers"):
        require(forbidden not in encoded, "source-derived field in universal geometry")
    return dict(value)


def validate_named_closure(value: Any, fields: Sequence[str], label: str) -> dict[str, str]:
    require(isinstance(value, dict) and set(value) == set(fields), f"{label} fields")
    return {name: digest(value[name], f"{label} {name}") for name in fields}


def validate_primary_authorization(
    value: Any,
    *,
    expected_auditor_manifest_sha256: str,
    expected_audit_receipt_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema", "status", "source_status", "positive_claim_authority",
        "controls_authorized", "source_artifact_sha256",
        "source_full_geometry_sha256", "source_structural_geometry_sha256",
        "source_pipeline_sha256", "source_score_receipt_sha256",
        "source_moment_auditor_sha256", "source_reconstruction_sha256",
        "source_absolute_saving_bpw", "source_gates",
        "universal_format_geometry_sha256", "symmetric_codec_closure",
        "v9_primary_source_manifest_sha256", "v9_primary_runner_sha256",
        "independent_result_auditor_manifest_sha256",
        "independent_result_audit_receipt_sha256", "authorization_sha256",
    }
    require(isinstance(value, dict) and set(value) == required, "primary authorization fields")
    require(value["schema"] == "uwfa-sc-v9-primary-controls-authorization-v1", "primary authorization schema")
    require(value["status"] == "PASS_INDEPENDENT_PRIMARY_SURVIVOR_CONTROLS_AUTHORIZED", "primary authorization status")
    require(value["source_status"] == PRIMARY_STATUS, "primary survivor status")
    require(value["positive_claim_authority"] is False, "primary remains nonpromoting")
    require(value["controls_authorized"] is True, "primary controls gate")
    for field in SOURCE_CLOSURE_FIELDS:
        digest(value[field], field)
    digest(value["source_reconstruction_sha256"], "source reconstruction")
    finite_float(value["source_absolute_saving_bpw"], "source saving", positive=True)
    gates = value["source_gates"]
    required_gates = {
        "rate_interval", "F_target", "cold_read_below_2x", "heldout",
        "standalone_decode", "integrity", "independent_result_audit",
    }
    require(isinstance(gates, dict) and set(gates) == required_gates, "source gate fields")
    require(all(gates[name] is True for name in required_gates), "all source gates pass")
    require(value["universal_format_geometry_sha256"] == universal_format_geometry_sha256(), "primary universal geometry")
    validate_named_closure(value["symmetric_codec_closure"], SYMMETRIC_CODEC_CLOSURE_FIELDS, "symmetric codec closure")
    require(value["v9_primary_source_manifest_sha256"] == V9_PRIMARY_SOURCE_MANIFEST_SHA256, "primary source manifest pin")
    require(value["v9_primary_runner_sha256"] == V9_PRIMARY_RUNNER_SHA256, "primary runner pin")
    require(
        value["independent_result_auditor_manifest_sha256"]
        == digest(expected_auditor_manifest_sha256, "expected primary auditor manifest"),
        "external primary auditor manifest pin",
    )
    require(
        value["independent_result_audit_receipt_sha256"]
        == digest(expected_audit_receipt_sha256, "expected primary audit receipt"),
        "external primary audit receipt pin",
    )
    validate_seal(value, "authorization_sha256")
    return dict(value)


def validate_run_authorization(
    value: Any,
    *,
    expected_consumer_source_manifest_sha256: str,
    expected_consumer_auditor_manifest_sha256: str,
    expected_consumer_audit_receipt_sha256: str,
    expected_producer_auditor_manifest_sha256: str,
    expected_producer_audit_receipt_sha256: str,
    expected_v8_all150_preflight_receipt_sha256: str,
    expected_gpu_identity_receipt_sha256: str,
    expected_source_snapshot_root_sha256: str,
    expected_descriptor_source_builder_sha256: str,
    expected_moment_replayer_source_sha256: str,
    expected_audit_bootstrap_sha256: str,
    expected_root_complete_sha256: str,
    expected_primary_authorization_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema", "status", "payload_access_authority", "positive_claim_authority",
        "consumer_source_manifest_sha256", "consumer_auditor_manifest_sha256",
        "consumer_audit_receipt_sha256", "producer_source_manifest_sha256",
        "producer_auditor_manifest_sha256", "producer_audit_receipt_sha256",
        "eight_control_root_complete_sha256", "primary_authorization_sha256",
        "v8_runtime_closure", "v8_all150_preflight_receipt_sha256",
        "independent_gpu_identity_receipt_sha256", "source_snapshot_root_sha256",
        "descriptor_source_builder_sha256", "moment_replayer_source_sha256",
        "audit_bootstrap_sha256", "all_eight_authenticate_before_fit",
        "all_150_independent_per_executed_control", "source_winner_reuse_forbidden",
        "member_loader_rejects_symlinks_and_path_escape",
        "immutable_snapshot_held_through_run",
        "run_authorization_sha256",
    }
    require(isinstance(value, dict) and set(value) == required, "run authorization fields")
    require(value["schema"] == "uwfa-sc-v9-matched-controls-run-authorization-v1", "run authorization schema")
    require(value["status"] == "PASS_EXTERNAL_PINS_CONTROL_PAYLOAD_ACCESS_AUTHORIZED", "run authorization status")
    require(value["payload_access_authority"] is True and value["positive_claim_authority"] is False, "run authority boundary")
    exact = {
        "consumer_source_manifest_sha256": expected_consumer_source_manifest_sha256,
        "consumer_auditor_manifest_sha256": expected_consumer_auditor_manifest_sha256,
        "consumer_audit_receipt_sha256": expected_consumer_audit_receipt_sha256,
        "producer_source_manifest_sha256": PRODUCER_SOURCE_MANIFEST_SHA256,
        "producer_auditor_manifest_sha256": expected_producer_auditor_manifest_sha256,
        "producer_audit_receipt_sha256": expected_producer_audit_receipt_sha256,
        "eight_control_root_complete_sha256": expected_root_complete_sha256,
        "primary_authorization_sha256": expected_primary_authorization_sha256,
        "v8_all150_preflight_receipt_sha256": expected_v8_all150_preflight_receipt_sha256,
        "independent_gpu_identity_receipt_sha256": expected_gpu_identity_receipt_sha256,
        "source_snapshot_root_sha256": expected_source_snapshot_root_sha256,
        "descriptor_source_builder_sha256": expected_descriptor_source_builder_sha256,
        "moment_replayer_source_sha256": expected_moment_replayer_source_sha256,
        "audit_bootstrap_sha256": expected_audit_bootstrap_sha256,
    }
    for name, expected in exact.items():
        require(value[name] == digest(expected, f"expected {name}"), f"run pin {name}")
    runtime = validate_named_closure(value["v8_runtime_closure"], V8_RUNTIME_CLOSURE_FIELDS, "v8 runtime closure")
    require(runtime == V8_RUNTIME_EXPECTED, "exact v8 runtime source pins")
    require(value["all_eight_authenticate_before_fit"] is True, "all-eight preauthentication")
    require(value["all_150_independent_per_executed_control"] is True, "all150 run gate")
    require(value["source_winner_reuse_forbidden"] is True, "source winner reuse gate")
    require(
        value["member_loader_rejects_symlinks_and_path_escape"] is True
        and value["immutable_snapshot_held_through_run"] is True,
        "immutable safe member loader",
    )
    validate_seal(value, "run_authorization_sha256")
    return dict(value)


def safe_member_name(value: Any) -> str:
    require(isinstance(value, str) and 0 < len(value) <= 512, "member name")
    require(
        "\\" not in value
        and ":" not in value
        and "\x00" not in value
        and not value.startswith("/"),
        "member path syntax",
    )
    normalized = posixpath.normpath(value)
    require(normalized == value and value not in (".", "..") and not value.startswith("../"), "member path traversal")
    return value


def validate_member_rows(value: Any, label: str) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{label} list")
    rows = []
    names = []
    for raw in value:
        require(isinstance(raw, dict) and set(raw) == {"name", "bytes", "sha256"}, f"{label} row")
        name = safe_member_name(raw["name"])
        rows.append({
            "name": name,
            "bytes": exact_int(raw["bytes"], f"{label} bytes", 0, 1 << 40),
            "sha256": digest(raw["sha256"], f"{label} digest"),
        })
        names.append(name)
    require(names == sorted(names) and len(names) == len(set(names)), f"{label} canonical unique order")
    return rows


def members_root(rows: Sequence[Mapping[str, Any]], domain: bytes) -> str:
    state = hashlib.sha256(domain)
    for row in rows:
        name = str(row["name"]).encode("utf-8")
        state.update(struct.pack("<Q", len(name)))
        state.update(name)
        state.update(struct.pack("<Q", int(row["bytes"])))
        state.update(bytes.fromhex(str(row["sha256"])))
    return state.hexdigest()


def role_shape(role: str) -> list[int]:
    require(role in ROLES, "role")
    return [HIDDEN, INTERMEDIATE] if role == "down" else [INTERMEDIATE, HIDDEN]


def _validate_source_panel(value: Any, *, seed: int, prefix: str) -> tuple[dict[str, Any], dict[str, list[str]]]:
    required = {
        "schema", "status", "seed", "identity_semantics", "matrices",
        "source_bytes", "source_weights", "source_panel_manifest_sha256",
    }
    require(isinstance(value, dict) and set(value) == required, "source panel fields")
    require(value["schema"] == "uwfa-sc-v9-control-bf16-source-panel-v1", "source panel schema")
    require(value["status"] == "COMPLETE_RETAINED_ENCODER_INPUT" and value["seed"] == seed, "source panel status/seed")
    require(value["identity_semantics"] == "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY", "source panel identity")
    matrices = value["matrices"]
    require(isinstance(matrices, list) and len(matrices) == 18, "source matrix coverage")
    chunks: dict[str, list[str]] = {}
    row_fields = {
        "matrix_ordinal", "slot", "role", "shape", "relpath", "bytes",
        "sha256", "n18_chunk_sha256s",
    }
    for ordinal, row in enumerate(matrices):
        require(isinstance(row, dict) and set(row) == row_fields, f"source row {ordinal}")
        slot, role = divmod(ordinal, 3)
        role_name = ROLES[role]
        require(row["matrix_ordinal"] == ordinal and row["slot"] == slot and row["role"] == role_name, f"source identity {ordinal}")
        require(row["shape"] == role_shape(role_name), f"source shape {ordinal}")
        require(row["bytes"] == SOURCE_BYTES_PER_MATRIX, f"source bytes {ordinal}")
        relpath = safe_member_name(row["relpath"])
        require(relpath == f"bf16_sources/slot_{slot:02d}_{role_name}.bf16.bin", f"source relpath {ordinal}")
        digest(row["sha256"], f"source hash {ordinal}")
        raw_chunks = row["n18_chunk_sha256s"]
        require(isinstance(raw_chunks, list) and len(raw_chunks) == 6, f"source chunks {ordinal}")
        chunks[f"{prefix}/{relpath}"] = [digest(item, f"source chunk {ordinal}") for item in raw_chunks]
    require(value["source_bytes"] == 18 * SOURCE_BYTES_PER_MATRIX, "source total bytes")
    require(value["source_weights"] == WEIGHTS, "source total weights")
    clean = dict(value)
    claimed = digest(clean.pop("source_panel_manifest_sha256"), "source panel seal")
    require(sha256(canonical_json(clean)) == claimed, "source panel integrity")
    return dict(value), chunks


def _validate_control_binding(
    value: Any,
    *,
    seed: int,
    primary: Mapping[str, Any],
    universal_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema", "seed", *SOURCE_CLOSURE_FIELDS,
        "pipeline_sha256", "universal_format_geometry_sha256",
        "generator_capsule_sha256", "moment_match_receipt_sha256",
        "source_panel_manifest_sha256", "control_artifact_sha256",
        "control_full_geometry_sha256", "control_structural_geometry_sha256",
        "symmetric_codec_closure", "binding_sha256",
    }
    # source_pipeline is serialized under the historical pipeline field only.
    required.remove("source_pipeline_sha256")
    require(isinstance(value, dict) and set(value) == required, "control binding fields")
    require(value["schema"] == "uwfa-matched-gaussian-control-binding-v9" and value["seed"] == seed, "control binding schema/seed")
    mapping = {
        "source_artifact_sha256": "source_artifact_sha256",
        "source_full_geometry_sha256": "source_full_geometry_sha256",
        "source_structural_geometry_sha256": "source_structural_geometry_sha256",
        "pipeline_sha256": "source_pipeline_sha256",
        "source_score_receipt_sha256": "source_score_receipt_sha256",
        "source_moment_auditor_sha256": "source_moment_auditor_sha256",
    }
    for control_name, primary_name in mapping.items():
        require(value[control_name] == primary[primary_name], f"binding source closure {control_name}")
    require(value["universal_format_geometry_sha256"] == universal_sha256, "binding universal geometry")
    for name in (
        "generator_capsule_sha256", "moment_match_receipt_sha256",
        "source_panel_manifest_sha256", "control_artifact_sha256",
        "control_full_geometry_sha256", "control_structural_geometry_sha256",
    ):
        digest(value[name], name)
    closure = validate_named_closure(value["symmetric_codec_closure"], SYMMETRIC_CODEC_CLOSURE_FIELDS, "control symmetric codec closure")
    require(closure == primary["symmetric_codec_closure"], "control/source symmetric codec closure equality")
    validate_seal(value, "binding_sha256")
    return dict(value)


def _validate_simple_sealed(value: Any, *, schema: str, seal: str, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and value.get("schema") == schema, f"{label} schema")
    validate_seal(value, seal)
    return dict(value)


def authenticate_eight_control_root(
    *,
    root_complete_bytes: bytes,
    expected_root_complete_sha256: str,
    observed_member_names: Sequence[str],
    member_loader: Callable[[str], bytes],
    primary_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate every declared byte and all eight control metadata sets.

    No candidate-fit callable is accepted here. The caller cannot begin a fit
    until this function returns a complete eight-control authentication row.
    """
    require(sha256(root_complete_bytes) == digest(expected_root_complete_sha256, "control root COMPLETE"), "control root COMPLETE pin")
    root = strict_json(root_complete_bytes, "control root COMPLETE")
    required = {
        "schema", "status", "controls", "generator_capsule_sha256",
        "moment_contract_sha256", "universal_format_geometry_sha256",
        "members", "members_root_sha256", "all_control_sources_retained",
        "all_control_artifacts_literal_current_format", "all_150_wfa_search_run",
        "positive_claim_authority",
    }
    require(set(root) == required, "control root fields")
    require(root["schema"] == "uwfa-sc-v9-full-ptq-eight-control-root-v1", "control root schema")
    require(root["status"] == "COMPLETE_EIGHT_FULL_PTQ_CONTROLS_AWAITING_V9_ALL150_CONSUMER", "control root status")
    require(root["all_control_sources_retained"] is True and root["all_control_artifacts_literal_current_format"] is True, "control root completeness")
    require(root["all_150_wfa_search_run"] is False and root["positive_claim_authority"] is False, "producer claim boundary")
    rows = validate_member_rows(root["members"], "root members")
    require(members_root(rows, b"UWFA-SC-V9-EIGHT-CONTROL-ROOT-v1\x00") == root["members_root_sha256"], "root member declaration integrity")
    declared_names = [row["name"] for row in rows]
    observed = [safe_member_name(name) for name in observed_member_names]
    require(sorted(observed) == sorted(declared_names + ["COMPLETE.json"]), "observed/declaration exact member set")
    require("INCOMPLETE" not in observed, "incomplete marker present")
    index = {row["name"]: row for row in rows}

    controls = root["controls"]
    require(isinstance(controls, list) and len(controls) == 8, "root eight controls")
    metadata_names = {"GENERATOR_CAPSULE.json", "MOMENT_CONTRACT.json", "UNIVERSAL_FORMAT_GEOMETRY.json"}
    control_rows = []
    control_fields = {
        "index", "seed", "relpath", "complete_sha256", "artifact_sha256",
        "source_panel_manifest_sha256", "control_full_geometry_sha256",
        "control_structural_geometry_sha256",
    }
    for index_value, (expected_seed, row) in enumerate(zip(CONTROL_SEEDS, controls, strict=True)):
        require(isinstance(row, dict) and set(row) == control_fields, "root control row fields")
        prefix = f"control_{index_value:02d}_{expected_seed}"
        require(row["index"] == index_value and row["seed"] == expected_seed and row["relpath"] == prefix, "root control order")
        for name in control_fields - {"index", "seed", "relpath"}:
            digest(row[name], f"root control {name}")
        metadata_names.update({
            f"{prefix}/COMPLETE.json",
            f"{prefix}/SOURCE_PANEL.json",
            f"{prefix}/CONTROL_BINDING.json",
            f"{prefix}/MOMENT_MATCH_RECEIPT.json",
            f"{prefix}/INDEPENDENT_SCORE.json",
            f"{prefix}/SCORE_RECEIPT.json",
            f"{prefix}/current_plan/summary.json",
            f"{prefix}/current_plan/plan.lock.json",
        })
        control_rows.append((expected_seed, prefix, row))
    require(metadata_names.issubset(index), "required control metadata coverage")

    cache: dict[str, bytes] = {}
    for name in sorted(metadata_names):
        payload = member_loader(name)
        require(isinstance(payload, bytes), f"member loader bytes {name}")
        declaration = index[name]
        require(len(payload) == declaration["bytes"] and sha256(payload) == declaration["sha256"], f"metadata member digest {name}")
        cache[name] = payload
    geometry = validate_universal_geometry(strict_json(cache["UNIVERSAL_FORMAT_GEOMETRY.json"], "universal geometry", maximum=1 << 20))
    universal_sha = universal_format_geometry_sha256()
    require(root["universal_format_geometry_sha256"] == universal_sha, "root universal geometry digest")
    require(sha256(canonical_json(geometry)) == universal_sha, "literal universal geometry digest")
    require(sha256(cache["GENERATOR_CAPSULE.json"]) == root["generator_capsule_sha256"], "root generator capsule")
    require(sha256(cache["MOMENT_CONTRACT.json"]) == root["moment_contract_sha256"], "root moment contract")

    source_chunks: dict[str, list[str]] = {}
    prepared = []
    for seed, prefix, root_row in control_rows:
        complete_name = f"{prefix}/COMPLETE.json"
        require(sha256(cache[complete_name]) == root_row["complete_sha256"], "control COMPLETE root binding")
        complete = strict_json(cache[complete_name], f"control {seed} COMPLETE")
        complete_fields = {
            "schema", "status", "seed", "members", "members_root_sha256",
            "artifact", "source_panel_manifest_sha256", "moment_match_receipt_sha256",
            "score_receipt_sha256", "binding_sha256", "all_150_wfa_search_run",
            "requires_separately_audited_v9_controls_bridge",
        }
        require(set(complete) == complete_fields, "control COMPLETE fields")
        require(complete["schema"] == "uwfa-sc-v9-full-ptq-control-complete-v1" and complete["seed"] == seed, "control COMPLETE schema/seed")
        require(complete["status"] == "COMPLETE_FULL_BF16_TO_CURRENT_STRATA_ARTIFACT_NONPROMOTING", "control COMPLETE status")
        require(complete["all_150_wfa_search_run"] is False and complete["requires_separately_audited_v9_controls_bridge"] is True, "control producer boundary")
        local_rows = validate_member_rows(complete["members"], f"control {seed} members")
        require(members_root(local_rows, b"UWFA-SC-V9-CONTROL-MEMBERS-v1\x00") == complete["members_root_sha256"], "control member root")
        root_under_prefix = {
            name[len(prefix) + 1:]: {
                "name": name[len(prefix) + 1:],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for name, row in index.items()
            if name.startswith(prefix + "/") and name != complete_name
        }
        require(root_under_prefix == {row["name"]: row for row in local_rows}, "control/root member declaration equality")
        panel, chunks = _validate_source_panel(
            strict_json(cache[f"{prefix}/SOURCE_PANEL.json"], f"control {seed} source panel"),
            seed=seed,
            prefix=prefix,
        )
        source_chunks.update(chunks)
        require(panel["source_panel_manifest_sha256"] == root_row["source_panel_manifest_sha256"] == complete["source_panel_manifest_sha256"], "source panel binding")
        binding = _validate_control_binding(
            strict_json(cache[f"{prefix}/CONTROL_BINDING.json"], f"control {seed} binding"),
            seed=seed,
            primary=primary_authorization,
            universal_sha256=universal_sha,
        )
        require(binding["binding_sha256"] == complete["binding_sha256"], "control binding COMPLETE")
        require(binding["generator_capsule_sha256"] == root["generator_capsule_sha256"], "control generator root binding")
        require(binding["source_panel_manifest_sha256"] == panel["source_panel_manifest_sha256"], "control source panel binding")
        moment = _validate_simple_sealed(
            strict_json(cache[f"{prefix}/MOMENT_MATCH_RECEIPT.json"], f"control {seed} moment receipt"),
            schema="uwfa-sc-v9-bf16-moment-replay-receipt-v1",
            seal="moment_match_receipt_sha256",
            label="moment receipt",
        )
        require(moment.get("seed") == seed and moment["moment_match_receipt_sha256"] == complete["moment_match_receipt_sha256"] == binding["moment_match_receipt_sha256"], "moment receipt binding")
        score_receipt = _validate_simple_sealed(
            strict_json(cache[f"{prefix}/SCORE_RECEIPT.json"], f"control {seed} score receipt"),
            schema="uwfa-bound-baseline-score-v8",
            seal="score_receipt_sha256",
            label="score receipt",
        )
        require(score_receipt["score_receipt_sha256"] == complete["score_receipt_sha256"], "score receipt COMPLETE")
        score = strict_json(cache[f"{prefix}/INDEPENDENT_SCORE.json"], f"control {seed} independent score")
        require(score.get("schema") == "uwfa-sc-v9-universal-control-independent-score-v1", "independent score schema")
        require(score.get("same_reconstruction_as_exact_v8_adapter") is True and score.get("all_payloads_canonically_reencoded") is True, "independent score decode flags")
        for geometry_name in ("control_full_geometry", "control_structural_geometry"):
            require(isinstance(score.get(geometry_name), dict), f"score {geometry_name}")
            require(sha256(canonical_json(score[geometry_name])) == score[f"{geometry_name}_sha256"], f"score {geometry_name} digest")
        require(score["control_full_geometry_sha256"] == root_row["control_full_geometry_sha256"] == binding["control_full_geometry_sha256"], "control full geometry binding")
        require(score["control_structural_geometry_sha256"] == root_row["control_structural_geometry_sha256"] == binding["control_structural_geometry_sha256"], "control structural geometry binding")
        require(score["universal_format_geometry_sha256"] == universal_sha, "score universal geometry")
        summary = strict_json(cache[f"{prefix}/current_plan/summary.json"], f"control {seed} summary")
        artifact_row = summary.get("artifact")
        require(isinstance(artifact_row, dict), "summary artifact")
        require(complete["artifact"] == artifact_row, "control COMPLETE/summary artifact equality")
        artifact_member = safe_member_name(f"{prefix}/current_plan/{artifact_row.get('relpath')}")
        require(artifact_member in index, "literal control artifact member")
        require(index[artifact_member]["bytes"] == ARTIFACT_BYTES, "literal control artifact bytes")
        require(index[artifact_member]["sha256"] == root_row["artifact_sha256"] == binding["control_artifact_sha256"] == score["artifact_sha256"], "literal artifact digest binding")
        require(score["artifact_bytes"] == ARTIFACT_BYTES, "independent score artifact bytes")
        require(score_receipt["artifact_sha256"] == score["artifact_sha256"] and score_receipt["artifact_bytes"] == ARTIFACT_BYTES, "score/artifact binding")
        require(score_receipt["reconstruction_f64_sha256"] == score["reconstruction_f64_sha256"], "score reconstruction binding")
        plan_member = f"{prefix}/current_plan/plan.lock.json"
        prepared.append({
            "seed": seed,
            "prefix": prefix,
            "artifact_member": artifact_member,
            "plan_member": plan_member,
            "plan_sha256": index[plan_member]["sha256"],
            "score_receipt_member": f"{prefix}/SCORE_RECEIPT.json",
            "score_receipt_sha256": index[f"{prefix}/SCORE_RECEIPT.json"]["sha256"],
            "source_panel": panel,
            "binding": binding,
            "moment_receipt": moment,
            "moment_receipt_file_sha256": index[
                f"{prefix}/MOMENT_MATCH_RECEIPT.json"
            ]["sha256"],
            "score": score,
            "score_receipt": score_receipt,
            "complete_sha256": root_row["complete_sha256"],
        })

    # Authenticate every byte only after all eight declarations and metadata
    # are closed. Source chunks are additionally replayed from retained BF16.
    for row in rows:
        name = row["name"]
        payload = cache.get(name)
        if payload is None:
            payload = member_loader(name)
        require(isinstance(payload, bytes), f"member bytes {name}")
        require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"], f"member authentication {name}")
        if name in source_chunks:
            chunks = [
                sha256(payload[offset:offset + SOURCE_CHUNK_BYTES])
                for offset in range(0, len(payload), SOURCE_CHUNK_BYTES)
            ]
            require(chunks == source_chunks[name], f"retained BF16 chunk authentication {name}")
    require(len(prepared) == 8 and [row["seed"] for row in prepared] == list(CONTROL_SEEDS), "all-eight prepared order")
    return {
        "schema": "uwfa-sc-v9-eight-control-input-authentication-v1",
        "status": "PASS_ALL_EIGHT_FULL_PTQ_BUNDLES_AUTHENTICATED_BEFORE_ANY_FIT",
        "root_complete_sha256": sha256(root_complete_bytes),
        "members_root_sha256": root["members_root_sha256"],
        "universal_format_geometry_sha256": universal_sha,
        "generator_capsule_sha256": root["generator_capsule_sha256"],
        "moment_contract_sha256": root["moment_contract_sha256"],
        "controls": prepared,
        "all_declared_members_authenticated": True,
        "all_retained_bf16_chunks_authenticated": True,
        "candidate_fit_calls": 0,
    }


def validate_all150_scientific(value: Any, common: Any) -> dict[str, Any]:
    require(isinstance(value, dict) and value.get("estimable") is True, "scientific estimable")
    require(value.get("source_winner_reused") is False, "source winner reuse forbidden")
    require(value.get("complete_150_cell_search_recorded_every_fold") is True, "all150 recorded flag")
    folds = value.get("folds")
    require(isinstance(folds, list) and folds, "scientific folds")
    bank = list(common.candidate_bank())
    require(len(bank) == ALL150 and [int(row.selector_ordinal) for row in bank] == list(range(ALL150)), "canonical 150 bank")
    for fold in folds:
        cells = fold.get("all_150_inner_validation_cells")
        require(isinstance(cells, list) and len(cells) == ALL150, "fold exact 150 cells")
        choices = []
        for ordinal, (cell, candidate) in enumerate(zip(cells, bank, strict=True)):
            expected = candidate.as_dict()
            required_cell = set(expected) | {
                "validation_charged_bits", "fitted_frequency_u16_sha256",
                "validation_lengths_u64_sha256", "validation_stream_ordinals",
                "trained_only_on_inner_train_streams", "source_winner_reused",
                "cell_result_sha256",
            }
            require(
                isinstance(cell, dict)
                and set(cell) == required_cell
                and all(cell.get(name) == expected[name] for name in expected),
                "cell candidate identity",
            )
            require(cell.get("selector_ordinal") == ordinal, "cell canonical selector")
            require(
                cell["trained_only_on_inner_train_streams"] is True
                and cell["source_winner_reused"] is False,
                "cell source-independent selection flags",
            )
            require(
                isinstance(cell["validation_stream_ordinals"], list)
                and all(type(item) is int and item >= 0 for item in cell["validation_stream_ordinals"]),
                "cell validation stream ordinals",
            )
            charged = exact_int(cell.get("validation_charged_bits"), "cell validation charged bits", 0)
            digest(cell.get("fitted_frequency_u16_sha256"), "cell frequency")
            digest(cell.get("validation_lengths_u64_sha256"), "cell lengths")
            validate_seal(cell, "cell_result_sha256")
            choices.append((charged, ordinal))
        require(fold.get("all_150_cell_results_sha256") == sha256(canonical_json(cells)), "fold all150 result list")
        selected = min(choices)[1]
        require(fold["selected_by_inner_validation_only"]["selector_ordinal"] == selected, "fold independent selector")
    return dict(value)


def early_null_decision(
    *,
    source_saving_bpw: float,
    executed_controls: Sequence[Mapping[str, Any]],
    authenticated_controls: int,
) -> dict[str, Any]:
    source = finite_float(source_saving_bpw, "source saving", positive=True)
    require(authenticated_controls == 8, "all controls authenticate before decision")
    require(len(executed_controls) <= 8, "executed control count")
    for index, row in enumerate(executed_controls):
        require(row.get("seed") == CONTROL_SEEDS[index], "executed control order")
        gain = finite_float(row.get("absolute_saving_bpw"), "control saving")
        if gain >= source:
            return {
                "status": "HARD_KILL_MATCHED_GAUSSIAN_NOT_SPECIFIC",
                "specificity_pass": False,
                "decisive_seed": CONTROL_SEEDS[index],
                "decisive_control_saving_bpw": gain,
                "source_saving_bpw": source,
                "source_minus_strongest_executed_null_bpw": source - max(float(item["absolute_saving_bpw"]) for item in executed_controls),
                "controls_executed": len(executed_controls),
                "early_stop_required": True,
                "positive_claim_authority": False,
            }
    if len(executed_controls) < 8:
        return {
            "status": "BLOCK_INCOMPLETE_MATCHED_NULL_SEQUENCE",
            "specificity_pass": False,
            "source_saving_bpw": source,
            "controls_executed": len(executed_controls),
            "early_stop_required": False,
            "positive_claim_authority": False,
        }
    strongest = max(float(row["absolute_saving_bpw"]) for row in executed_controls)
    return {
        "status": "PASS_ALL_EIGHT_MATCHED_NULLS_NONPROMOTING_AWAITING_INDEPENDENT_RESULT_AUDIT",
        "specificity_pass": source > strongest,
        "decisive_seed": None,
        "strongest_matched_null_bpw": strongest,
        "source_saving_bpw": source,
        "source_minus_strongest_executed_null_bpw": source - strongest,
        "controls_executed": 8,
        "early_stop_required": False,
        "positive_claim_authority": False,
    }


def audit_terminal_result(
    *,
    result: Any,
    control_rows: Sequence[Mapping[str, Any]],
    common: Any,
) -> dict[str, Any]:
    """Independently recompute all retained-cell and early-null conclusions."""
    required = {
        "schema", "status", "input_authentication_sha256",
        "primary_authorization_sha256", "run_authorization_sha256",
        "controls_authenticated", "controls_executed", "executed_seeds",
        "unexecuted_seeds", "decision",
        "every_executed_control_recorded_all_150_per_fold",
        "source_winner_reuse_forbidden_and_not_observed", "specificity_pass",
        "positive_claim_authority", "fresh_independent_result_audit_required",
        "result_sha256",
    }
    require(isinstance(result, dict) and set(result) == required, "terminal result fields")
    require(result["schema"] == "uwfa-sc-v9-matched-gaussian-controls-result-v1", "terminal result schema")
    require(result["controls_authenticated"] == 8, "terminal all-eight authentication")
    require(result["controls_executed"] == len(control_rows), "terminal control count")
    require(result["executed_seeds"] == list(CONTROL_SEEDS[: len(control_rows)]), "terminal executed seed order")
    require(result["unexecuted_seeds"] == list(CONTROL_SEEDS[len(control_rows):]), "terminal unexecuted seed suffix")
    require(result["positive_claim_authority"] is False and result["fresh_independent_result_audit_required"] is True, "terminal claim boundary")
    validate_seal(result, "result_sha256")
    decision_inputs = []
    row_fields = {
        "schema", "index", "seed", "scientific_nested_holdout", "final",
        "absolute_saving_bpw", "repeated_complete_150_cell_selection_fit_pack_decode",
        "source_winner_reused", "moment_replay", "positive_claim_authority",
        "control_result_sha256",
    }
    for index, row in enumerate(control_rows):
        require(isinstance(row, dict) and set(row) == row_fields, "control output fields")
        require(row["schema"] == "uwfa-sc-v9-matched-control-all150-result-v1", "control output schema")
        require(row["index"] == index and row["seed"] == CONTROL_SEEDS[index], "control output order")
        require(row["positive_claim_authority"] is False and row["source_winner_reused"] is False, "control output claim/selection boundary")
        require(row["repeated_complete_150_cell_selection_fit_pack_decode"] is True, "control complete all150 pipeline")
        validate_all150_scientific(row["scientific_nested_holdout"], common)
        gain = finite_float(row["absolute_saving_bpw"], "control output saving")
        require(
            gain
            == finite_float(
                row["final"]["absolute_saving_vs_bound_current_artifact_bpw"],
                "control final saving",
            ),
            "control public/final saving equality",
        )
        validate_seal(row["moment_replay"], "replay_receipt_sha256")
        validate_seal(row, "control_result_sha256")
        decision_inputs.append({"seed": row["seed"], "absolute_saving_bpw": gain})
    require(
        result["every_executed_control_recorded_all_150_per_fold"]
        is all(row["repeated_complete_150_cell_selection_fit_pack_decode"] is True for row in control_rows),
        "terminal all150 conjunction",
    )
    require(
        result["source_winner_reuse_forbidden_and_not_observed"]
        is all(row["source_winner_reused"] is False for row in control_rows),
        "terminal source-winner conjunction",
    )
    if result["status"] in {
        "HARD_KILL_MATCHED_GAUSSIAN_NOT_SPECIFIC",
        "PASS_ALL_EIGHT_MATCHED_NULLS_NONPROMOTING_AWAITING_INDEPENDENT_RESULT_AUDIT",
        "BLOCK_INCOMPLETE_MATCHED_NULL_SEQUENCE",
    }:
        source = finite_float(result["decision"].get("source_saving_bpw"), "decision source saving", positive=True)
        expected = early_null_decision(
            source_saving_bpw=source,
            executed_controls=decision_inputs,
            authenticated_controls=8,
        )
        require(result["decision"] == expected and result["status"] == expected["status"], "terminal early-null recomputation")
        require(result["specificity_pass"] is (expected.get("specificity_pass") is True), "terminal specificity conclusion")
    else:
        require(result["specificity_pass"] is False, "blocked terminal cannot pass specificity")
    return dict(result)


def direct_main() -> int:
    print("BLOCK_DIRECT_EXECUTION_REQUIRES_EXTERNAL_AUDITED_RUN_AUTHORIZATION", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(direct_main())
