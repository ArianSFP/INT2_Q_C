#!/usr/bin/env python3
"""Dependency-free verification of a STRATA-XKLT-SC v2 release bundle.

The verifier is source-free: it checks byte-bound evidence, cross-document
hashes/seals, frozen artifact hashes, physical container invariants, one-shot
summary bindings, independent-audit gate consistency, and the 10/10 tamper
report without opening source payloads or rerunning the heavy auditor.

Expected manifest shape:

{
  "artifact": {
    "schema": "strata_xklt_sc_v2_release_manifest_v1",
    "protocol_mode": "blind"
  },
  "claim": {
    "audit_passed": true,
    "primary_claim_passed": true,
    "audit_execution_passed": true,
    "tamper_report_passed": true
  },
  "files": [
    {
      "path": "relative/path/from/repo/root",
      "bytes": 123,
      "sha256": "lowercase64hex...",
      "role": "container",
      "classification": "byte_bound_evidence"
    }
  ]
}

The manifest intentionally excludes itself and mutable prose documents.
It may declare a narrowly allow-listed historical frozen dependency as
withheld when redistribution rights are unavailable; the verifier still binds
its exact path and SHA-256 through the original freeze and runtime receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("release/strata_v2_release_manifest.json")
MANIFEST_SCHEMA = "strata_xklt_sc_v2_release_manifest_v1"

HEADER_BYTES = 128
ROUTE_BYTES = 144
LABEL_BYTES = 5_184
DIRECTORY_RECORD = struct.Struct("<BeI")
BLOCKS = 14
DIRECTORY_BYTES = BLOCKS * DIRECTORY_RECORD.size
RESERVOIR_BYTES = 7_603_175
CONTAINER_BYTES = HEADER_BYTES + ROUTE_BYTES + LABEL_BYTES + DIRECTORY_BYTES + RESERVOIR_BYTES
WEIGHTS = 28_311_552
PHYSICAL_BITS = CONTAINER_BYTES * 8
PHYSICAL_BPW = PHYSICAL_BITS / WEIGHTS
INTEGER_CAP_BITS = (43 * WEIGHTS) // 20
HEADROOM_BITS = INTEGER_CAP_BITS - PHYSICAL_BITS
GLOBAL_RESERVE_BITS = 65_536
NOMINAL_PROFILE_BUDGET_BITS = RESERVOIR_BYTES * 8 - GLOBAL_RESERVE_BITS
PRIMARY_RULE = (
    "physical_bpw<=2.15 AND complete blind lineage/source-derived metadata audit "
    "AND pooled source-domain MSE<2^-4.3"
)
GAUSSIAN_LIMIT = math.exp2(-4.3)
REQUIRED_ROLES = (
    "selection_lock",
    "source_lock",
    "codec_freeze",
    "codec_freeze_validation",
    "format_freeze",
    "preencoding_manifest",
    "allocation_lock",
    "one_shot_intent",
    "one_shot_summary",
    "container",
    "independent_audit",
    "tamper_report",
)
REQUIRED_AUDIT_CONDITIONS = (
    "physical_rate_at_most_2p15",
    "complete_source_lineage_present",
    "blind_protocol_mode",
    "source_staging_label_scale_and_dp_audit_passed",
    "source_domain_mse_below_gaussian_limit",
)
RUNTIME_ARTIFACT_FREEZE_KEYS = {
    "python_interpreter": "python_interpreter",
    "runner": "one_shot_runner",
    "polar_encoder": "polar_encoder",
    "base_encoder": "base_cupy_encoder",
    "procedural_bec_builder": "procedural_q31_bec",
    "common": "common",
    "emitter": "emitter",
    "format": "format",
    "independent_auditor": "independent_auditor",
}
EXPECTED_TAMPER_NAMES = {
    "selection_role_resealed",
    "source_nested_hash_resealed",
    "codec_threshold_resealed",
    "manifest_source_binding",
    "allocation_profile_resealed",
    "intent_retry_permission",
    "summary_artifact_hash",
    "coefficient_regeneration",
    "nonzero_stream_padding",
    "directory_scale",
}
EXPECTED_VALIDATION_KEYS = {
    "schema",
    "status",
    "passed",
    "freeze_path",
    "freeze_file_sha256",
    "freeze_internal_lock_sha256",
    "executing_validator_sha256",
    "frozen_artifact_count",
    "development_pooled_relative_mse",
    "gaussian_mse_reference",
    "physical_bits",
    "physical_bpw",
    "preaccess_state",
    "lock_sha256",
}
ALLOWED_WITHHELD_FROZEN_ARTIFACTS = {"base_cupy_encoder"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def require_sha256(value: Any, description: str) -> str:
    require(is_sha256(value), f"{description} must be lowercase 64-hex sha256")
    return str(value)


def same_float(observed: Any, expected: float, *, atol: float = 1e-15) -> bool:
    try:
        return math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def verify_internal_lock(value: dict[str, Any], description: str) -> str:
    expected = require_sha256(value.get("lock_sha256"), f"{description} lock_sha256")
    clean = dict(value)
    clean.pop("lock_sha256", None)
    require(sha256_bytes(canonical_json_bytes(clean)) == expected, f"{description} internal seal mismatch")
    return expected


def load_json_object(path: Path, description: str) -> tuple[dict[str, Any], str]:
    payload = path.resolve(strict=True).read_bytes()
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), f"{description} must be a JSON object")
    return value, sha256_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_path(value: Any, description: str) -> Path:
    require(isinstance(value, (str, Path)) and str(value), f"{description} must be a non-empty relative path")
    path = Path(str(value))
    require(not path.is_absolute(), f"{description} must be relative")
    require(path.drive == "", f"{description} must not include a drive")
    require(".." not in path.parts, f"{description} must not escape the repo root")
    return path


def resolve_under(root: Path, relative: Any, description: str) -> Path:
    path = (root / relative_path(relative, description)).resolve()
    root_resolved = root.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise AssertionError(f"{description} escapes the repo root") from exc
    return path


def expect_int(value: Any, description: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{description} must be an integer")
    return value


def expect_bool(value: Any, description: str) -> bool:
    require(isinstance(value, bool), f"{description} must be boolean")
    return value


def block_log2_for_ordinal(ordinal: int) -> int:
    return 21 if ordinal < 13 else 20


def block_values_for_ordinal(ordinal: int) -> int:
    return 1 << block_log2_for_ordinal(ordinal)


def verify_manifest_file_table(
    repo_root: Path, manifest_path: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    artifact = manifest.get("artifact")
    claim = manifest.get("claim")
    files = manifest.get("files")
    require(isinstance(artifact, dict), "manifest.artifact must be an object")
    require(isinstance(claim, dict), "manifest.claim must be an object")
    require(isinstance(files, list) and files, "manifest.files must be a non-empty list")
    require(artifact.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    if "weights" in artifact:
        require(expect_int(artifact["weights"], "artifact.weights") == WEIGHTS, "artifact weights mismatch")

    rows: list[dict[str, Any]] = []
    roles: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    manifest_real = manifest_path.resolve(strict=True)
    for index, row in enumerate(files):
        require(isinstance(row, dict), f"manifest.files[{index}] must be an object")
        for key in ("path", "bytes", "sha256", "role", "classification"):
            require(key in row, f"manifest.files[{index}] missing {key}")
        relative = relative_path(row["path"], f"manifest.files[{index}].path")
        role = str(row["role"])
        classification = str(row["classification"])
        size = expect_int(row["bytes"], f"manifest.files[{index}].bytes")
        digest = require_sha256(row["sha256"], f"manifest.files[{index}].sha256")
        require(classification.lower() != "mutable_prose", f"manifest.files[{index}] must not list mutable prose")
        path = resolve_under(repo_root, relative, f"manifest.files[{index}].path")
        require(path.resolve() != manifest_real, "manifest must not list itself in files")
        path_key = str(relative).replace("\\", "/")
        require(path_key not in seen_paths, f"duplicate manifest file path: {path_key}")
        seen_paths.add(path_key)
        require(path.is_file(), f"listed file is missing: {path_key}")
        require(path.stat().st_size == size, f"listed byte count mismatch: {path_key}")
        require(sha256_file(path) == digest, f"listed sha256 mismatch: {path_key}")
        item = {
            "path": path,
            "relative_path": path_key,
            "bytes": size,
            "sha256": digest,
            "role": role,
            "classification": classification,
        }
        rows.append(item)
        require(role not in roles, f"duplicate manifest role: {role}")
        roles[role] = item

    for role in REQUIRED_ROLES:
        require(role in roles, f"manifest missing required role: {role}")
    return rows, roles


def protocol_mode_from_manifest(manifest: dict[str, Any]) -> str:
    artifact = manifest["artifact"]
    claim = manifest["claim"]
    artifact_mode = artifact.get("protocol_mode", "blind")
    claim_mode = claim.get("protocol_mode", artifact_mode)
    require(artifact_mode in ("blind", "development"), "artifact.protocol_mode must be blind or development")
    require(claim_mode == artifact_mode, "claim.protocol_mode mismatch")
    return str(artifact_mode)


def parse_directory(container_path: Path) -> dict[str, Any]:
    raw = container_path.read_bytes()
    require(len(raw) == CONTAINER_BYTES, "container physical byte count mismatch")
    header = raw[:HEADER_BYTES]
    route = raw[HEADER_BYTES : HEADER_BYTES + ROUTE_BYTES]
    labels = raw[HEADER_BYTES + ROUTE_BYTES : HEADER_BYTES + ROUTE_BYTES + LABEL_BYTES]
    directory_offset = HEADER_BYTES + ROUTE_BYTES + LABEL_BYTES
    directory_raw = raw[directory_offset : directory_offset + DIRECTORY_BYTES]
    reservoir = raw[directory_offset + DIRECTORY_BYTES :]
    require(len(reservoir) == RESERVOIR_BYTES, "container reservoir byte count mismatch")

    rows: list[dict[str, Any]] = []
    profile_bytes = bytearray()
    valid_payload_bits = 0
    used_payload_bytes = 0
    cursor = 0
    for ordinal in range(BLOCKS):
        begin = ordinal * DIRECTORY_RECORD.size
        profile_q, decoder_scale, logical_bits = DIRECTORY_RECORD.unpack(
            directory_raw[begin : begin + DIRECTORY_RECORD.size]
        )
        payload_bytes = (logical_bits + 7) // 8
        require(cursor + payload_bytes <= RESERVOIR_BYTES, f"container payload overflow at block {ordinal}")
        payload = reservoir[cursor : cursor + payload_bytes]
        padding_bits = payload_bytes * 8 - logical_bits
        if padding_bits and payload:
            mask = (1 << padding_bits) - 1
            require((payload[-1] & mask) == 0, f"nonzero low padding bits in block {ordinal}")
        rows.append(
            {
                "block_ordinal": ordinal,
                "block_log2": block_log2_for_ordinal(ordinal),
                "block_values": block_values_for_ordinal(ordinal),
                "profile_q": int(profile_q),
                "decoder_scale": float(decoder_scale),
                "decoder_scale_fp16_hex": struct.pack("<e", decoder_scale).hex(),
                "logical_bits": int(logical_bits),
                "payload_bytes": int(payload_bytes),
                "payload_terminal_padding_bits": int(padding_bits),
                "payload_terminal_padding_all_zero": True,
            }
        )
        profile_bytes.append(profile_q)
        valid_payload_bits += logical_bits
        used_payload_bytes += payload_bytes
        cursor += payload_bytes
    require(all(byte == 0 for byte in reservoir[used_payload_bytes:]), "nonzero zero-tail bytes in reservoir")
    return {
        "raw_sha256": sha256_bytes(raw),
        "header_sha256": sha256_bytes(header),
        "route_sha256": sha256_bytes(route),
        "labels_sha256": sha256_bytes(labels),
        "directory_sha256": sha256_bytes(directory_raw),
        "reservoir_sha256": sha256_bytes(reservoir),
        "profile_bytes_hex": bytes(profile_bytes).hex(),
        "rows": rows,
        "valid_payload_bits": valid_payload_bits,
        "used_payload_bytes": used_payload_bytes,
        "zero_terminal_bytes": RESERVOIR_BYTES - used_payload_bytes,
    }


def verify_release(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_file_hash = load_json_object(manifest_path, "release manifest")
    del manifest_file_hash
    file_rows, role_rows = verify_manifest_file_table(repo_root, manifest_path, manifest)
    protocol_mode = protocol_mode_from_manifest(manifest)

    selection, selection_file_hash = load_json_object(role_rows["selection_lock"]["path"], "selection lock")
    source_lock, source_file_hash = load_json_object(role_rows["source_lock"]["path"], "source lock")
    codec_freeze, codec_file_hash = load_json_object(role_rows["codec_freeze"]["path"], "codec freeze")
    validation, validation_file_hash = load_json_object(
        role_rows["codec_freeze_validation"]["path"], "codec-freeze validation receipt"
    )
    format_hash = sha256_file(role_rows["format_freeze"]["path"])
    preencoding_manifest, preencoding_manifest_file_hash = load_json_object(
        role_rows["preencoding_manifest"]["path"], "preencoding manifest"
    )
    allocation_lock, allocation_file_hash = load_json_object(role_rows["allocation_lock"]["path"], "allocation lock")
    one_shot_intent, intent_file_hash = load_json_object(
        role_rows["one_shot_intent"]["path"], "one-shot intent"
    )
    one_shot_summary, summary_file_hash = load_json_object(
        role_rows["one_shot_summary"]["path"], "one-shot summary"
    )
    independent_audit, independent_audit_file_hash = load_json_object(
        role_rows["independent_audit"]["path"], "independent audit"
    )
    tamper_report, tamper_report_file_hash = load_json_object(role_rows["tamper_report"]["path"], "tamper report")
    del independent_audit_file_hash, tamper_report_file_hash

    selection_internal = verify_internal_lock(selection, "selection lock")
    source_internal = verify_internal_lock(source_lock, "source lock")
    codec_internal = verify_internal_lock(codec_freeze, "codec freeze")
    validation_internal = verify_internal_lock(validation, "codec-freeze validation receipt")
    allocation_internal = verify_internal_lock(allocation_lock, "allocation lock")

    expected_selection_contract = (
        (
            "int2-qwen-blind-selection-proposal-v2",
            "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
        )
        if protocol_mode == "blind"
        else (
            "int2-qwen-blind-selection-v1",
            "selected_and_header_validated_tensor_payload_unopened",
        )
    )
    require(
        (selection.get("schema"), selection.get("status")) == expected_selection_contract,
        "selection lock schema/status mismatch",
    )

    expected_source_lock = codec_freeze.get("expected_source_lock")
    require(isinstance(expected_source_lock, dict), "codec freeze missing expected_source_lock")
    require(
        (source_lock.get("schema"), source_lock.get("status"))
        == (expected_source_lock.get("schema"), expected_source_lock.get("status")),
        "source lock schema/status mismatch",
    )
    for key in ("matrix_count", "block_count", "source_values", "source_bytes", "dtype"):
        if key in expected_source_lock:
            require(source_lock.get(key) == expected_source_lock.get(key), f"source lock {key} mismatch")
    require(source_lock.get("selection_lock_sha256") == selection_internal, "source lock selection seal mismatch")

    require(
        (codec_freeze.get("schema"), codec_freeze.get("status"))
        == ("strata_xklt_sc_v2_codec_freeze_v1", "frozen_before_blind_source_access"),
        "codec freeze schema/status mismatch",
    )
    require(codec_freeze.get("selection_lock_file_sha256") == selection_file_hash, "codec freeze selection file hash mismatch")
    require(codec_freeze.get("selection_lock_sha256") == selection_internal, "codec freeze selection seal mismatch")
    require(same_float(codec_freeze.get("physical_rate_limit_bpw"), 2.15), "codec freeze physical rate limit mismatch")
    require(same_float(codec_freeze.get("primary_mse_threshold"), GAUSSIAN_LIMIT), "codec freeze primary MSE threshold mismatch")
    require(same_float(codec_freeze.get("gaussian_mse_reference"), GAUSSIAN_LIMIT), "codec freeze gaussian reference mismatch")
    require(codec_freeze.get("allocator_frozen") is True, "codec freeze allocator_frozen must be true")
    require(codec_freeze.get("architecture_frozen") is True, "codec freeze architecture_frozen must be true")
    require(
        codec_freeze.get("no_retry_resume_or_postaccess_tuning") is True,
        "codec freeze no_retry_resume_or_postaccess_tuning must be true",
    )
    physical_ledger = codec_freeze.get("physical_ledger")
    require(isinstance(physical_ledger, dict), "codec freeze physical_ledger missing")
    expected_ledger = {
        "weights": WEIGHTS,
        "physical_bytes": CONTAINER_BYTES,
        "physical_bits": PHYSICAL_BITS,
        "physical_bpw": PHYSICAL_BPW,
        "integer_2p15_cap_bits": INTEGER_CAP_BITS,
        "headroom_bits": HEADROOM_BITS,
        "header_bits": HEADER_BYTES * 8,
        "route_bits": ROUTE_BYTES * 8,
        "label_bits": LABEL_BYTES * 8,
        "directory_bits": DIRECTORY_BYTES * 8,
        "reservoir_bytes": RESERVOIR_BYTES,
        "reservoir_bits": RESERVOIR_BYTES * 8,
        "global_no_retry_reserve_bits": GLOBAL_RESERVE_BITS,
        "nominal_profile_budget_bits": NOMINAL_PROFILE_BUDGET_BITS,
    }
    for key, expected in expected_ledger.items():
        observed = physical_ledger.get(key)
        if isinstance(expected, float):
            require(same_float(observed, expected), f"codec freeze physical_ledger[{key}] mismatch")
        else:
            require(observed == expected, f"codec freeze physical_ledger[{key}] mismatch")

    frozen_artifact_paths = codec_freeze.get("frozen_artifact_paths")
    frozen_artifacts = codec_freeze.get("frozen_artifact_sha256s")
    require(isinstance(frozen_artifact_paths, dict), "codec freeze frozen_artifact_paths missing")
    require(isinstance(frozen_artifacts, dict), "codec freeze frozen_artifact_sha256s missing")
    withheld_rows = manifest["artifact"].get("withheld_frozen_artifacts", [])
    require(isinstance(withheld_rows, list), "artifact.withheld_frozen_artifacts must be a list")
    withheld_by_name: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(withheld_rows):
        require(isinstance(row, dict), f"withheld frozen artifact {index} must be an object")
        require(set(row) == {"name", "path", "sha256", "reason"}, f"withheld frozen artifact {index} key set mismatch")
        name = str(row["name"])
        require(name in ALLOWED_WITHHELD_FROZEN_ARTIFACTS, f"withheld frozen artifact is not allow-listed: {name}")
        require(name not in withheld_by_name, f"duplicate withheld frozen artifact: {name}")
        require(isinstance(row["reason"], str) and row["reason"], f"withheld frozen artifact {name} reason missing")
        withheld_by_name[name] = row
    local_frozen_hashes: dict[str, str] = {}
    withheld_frozen_hashes: dict[str, str] = {}
    for name, expected_hash in frozen_artifacts.items():
        require_sha256(expected_hash, f"codec freeze frozen artifact {name}")
    for name, relative in frozen_artifact_paths.items():
        require(name in frozen_artifacts, f"codec freeze path {name} missing hash entry")
        path = resolve_under(repo_root, relative, f"codec freeze frozen_artifact_paths[{name}]")
        if name in withheld_by_name:
            row = withheld_by_name[name]
            require(row["path"] == relative, f"withheld frozen artifact path mismatch: {name}")
            require(require_sha256(row["sha256"], f"withheld frozen artifact {name} sha256") == frozen_artifacts[name], f"withheld frozen artifact sha256 mismatch: {name}")
            require(not path.exists(), f"withheld frozen artifact is unexpectedly redistributed: {name}")
            withheld_frozen_hashes[name] = frozen_artifacts[name]
            continue
        require(path.is_file(), f"frozen artifact file missing: {name}")
        observed_hash = sha256_file(path)
        require(observed_hash == frozen_artifacts[name], f"frozen artifact sha256 mismatch: {name}")
        local_frozen_hashes[name] = observed_hash
    require(set(withheld_by_name) == set(withheld_frozen_hashes), "withheld frozen artifact declaration mismatch")
    external_only_frozen = sorted(
        set(frozen_artifacts) - set(local_frozen_hashes) - set(withheld_frozen_hashes)
    )
    route_binding = preencoding_manifest.get("bindings", {}).get("route")
    if isinstance(route_binding, dict) and "sha256" in route_binding:
        require(codec_freeze.get("route_file_sha256") == route_binding.get("sha256"), "codec freeze route hash mismatch")
    require(frozen_artifacts.get("format") == format_hash, "codec freeze format hash mismatch")

    require(set(validation) == EXPECTED_VALIDATION_KEYS, "codec-freeze validation key set mismatch")
    require(
        (validation.get("schema"), validation.get("status"), validation.get("passed"))
        == ("strata_xklt_sc_v2_codec_freeze_validation_v1", "validated_before_blind_source_access", True),
        "codec-freeze validation schema/status/pass mismatch",
    )
    require(
        validation.get("freeze_path") == "blind_protocol_v2/codec_freeze.lock.json",
        "codec-freeze validation freeze_path mismatch",
    )
    require(validation.get("freeze_file_sha256") == codec_file_hash, "codec-freeze validation file hash mismatch")
    require(
        validation.get("freeze_internal_lock_sha256") == codec_internal,
        "codec-freeze validation internal lock mismatch",
    )
    require(
        validation.get("executing_validator_sha256") == frozen_artifacts.get("freeze_validator"),
        "codec-freeze validation validator hash mismatch",
    )
    require(
        expect_int(validation.get("frozen_artifact_count"), "validation.frozen_artifact_count")
        == len(frozen_artifacts),
        "codec-freeze validation artifact count mismatch",
    )
    require(
        math.isfinite(float(validation["development_pooled_relative_mse"]))
        and float(validation["development_pooled_relative_mse"]) < GAUSSIAN_LIMIT,
        "codec-freeze validation development pooled MSE must be finite and below the gaussian limit",
    )
    require(same_float(validation.get("gaussian_mse_reference"), GAUSSIAN_LIMIT), "codec-freeze validation gaussian reference mismatch")
    require(validation.get("physical_bits") == PHYSICAL_BITS, "codec-freeze validation physical_bits mismatch")
    require(same_float(validation.get("physical_bpw"), PHYSICAL_BPW), "codec-freeze validation physical_bpw mismatch")
    require(
        validation.get("preaccess_state") == codec_freeze.get("preaccess_state"),
        "codec-freeze validation preaccess_state mismatch",
    )

    source_freeze_binding = source_lock.get("codec_freeze")
    require(isinstance(source_freeze_binding, dict), "source lock codec_freeze binding missing")
    require(source_freeze_binding.get("file_sha256") == codec_file_hash, "source lock codec freeze file hash mismatch")
    require(
        source_freeze_binding.get("internal_lock_sha256") == codec_internal,
        "source lock codec freeze internal lock mismatch",
    )
    if protocol_mode == "blind":
        source_validation_binding = source_lock.get("codec_freeze_validation")
        require(isinstance(source_validation_binding, dict), "blind source lock missing codec_freeze_validation")
        require(
            source_validation_binding.get("file_sha256") == validation_file_hash
            and source_validation_binding.get("internal_lock_sha256") == validation_internal,
            "source lock codec-freeze validation binding mismatch",
        )

    require(
        (preencoding_manifest.get("schema"), preencoding_manifest.get("status"))
        == ("strata_xklt_sc_v2_preencoding_manifest_v1", "complete_and_allocation_sealed_before_encoding"),
        "preencoding manifest schema/status mismatch",
    )
    require(preencoding_manifest.get("protocol_mode") == protocol_mode, "preencoding manifest protocol_mode mismatch")
    bindings = preencoding_manifest.get("bindings")
    require(isinstance(bindings, dict), "preencoding manifest bindings missing")
    require(bindings.get("protocol_mode") == protocol_mode, "preencoding bindings protocol_mode mismatch")
    expected_binding_fields = {
        "selection_lock": {"file_sha256": selection_file_hash, "internal_lock_sha256": selection_internal},
        "source_lock": {
            "file_sha256": source_file_hash,
            "internal_lock_sha256": source_internal,
            "selection_lock_sha256": selection_internal,
        },
        "codec_freeze": {"file_sha256": codec_file_hash, "internal_lock_sha256": codec_internal},
        "codec_freeze_validation": {
            "file_sha256": validation_file_hash,
            "internal_lock_sha256": validation_internal,
        },
        "format_freeze": {"sha256": format_hash},
    }
    for name, expected_fields in expected_binding_fields.items():
        actual = bindings.get(name)
        require(isinstance(actual, dict), f"preencoding manifest missing binding {name}")
        require(all(actual.get(key) == value for key, value in expected_fields.items()), f"preencoding manifest binding mismatch: {name}")
    for executable_name in ("emitter", "common"):
        if executable_name in bindings:
            actual = bindings.get(executable_name)
            require(isinstance(actual, dict), f"preencoding manifest binding {executable_name} must be an object")
            expected_hash = frozen_artifacts.get(executable_name)
            if expected_hash is not None:
                require(actual.get("sha256") == expected_hash, f"preencoding manifest {executable_name} hash mismatch")

    assets = preencoding_manifest.get("assets")
    require(isinstance(assets, dict), "preencoding manifest assets missing")
    container = parse_directory(role_rows["container"]["path"])
    asset_expectations = {
        "header.bin": (HEADER_BYTES, container["header_sha256"]),
        "route.bin": (ROUTE_BYTES, container["route_sha256"]),
        "labels_3bit.bin": (LABEL_BYTES, container["labels_sha256"]),
        "profiles.bin": (BLOCKS, container["profile_bytes_hex"]),
    }
    for name, (expected_bytes, expected_hash_or_hex) in asset_expectations.items():
        row = assets.get(name)
        require(isinstance(row, dict), f"preencoding manifest asset missing: {name}")
        require(expect_int(row.get("bytes"), f"preencoding asset {name}.bytes") == expected_bytes, f"preencoding asset bytes mismatch: {name}")
        if name == "profiles.bin":
            require(require_sha256(row.get("sha256"), f"preencoding asset {name}.sha256") == sha256_bytes(bytes.fromhex(expected_hash_or_hex)), f"preencoding asset sha256 mismatch: {name}")
        else:
            require(require_sha256(row.get("sha256"), f"preencoding asset {name}.sha256") == expected_hash_or_hex, f"preencoding asset sha256 mismatch: {name}")

    manifest_blocks = preencoding_manifest.get("blocks")
    require(isinstance(manifest_blocks, list) and len(manifest_blocks) == BLOCKS, "preencoding manifest block count mismatch")
    for ordinal, (block, directory_row) in enumerate(zip(manifest_blocks, container["rows"], strict=True)):
        require(isinstance(block, dict), f"preencoding manifest block {ordinal} must be an object")
        require(expect_int(block.get("block_ordinal"), f"manifest block {ordinal}.block_ordinal") == ordinal, f"manifest block ordinal mismatch: {ordinal}")
        require(expect_int(block.get("block_log2"), f"manifest block {ordinal}.block_log2") == directory_row["block_log2"], f"manifest block log2 mismatch: {ordinal}")
        require(expect_int(block.get("values"), f"manifest block {ordinal}.values") == directory_row["block_values"], f"manifest block values mismatch: {ordinal}")
        require(expect_int(block.get("profile_id"), f"manifest block {ordinal}.profile_id") == directory_row["profile_q"], f"manifest block profile mismatch: {ordinal}")
        if "logical_bits" in block:
            require(expect_int(block["logical_bits"], f"manifest block {ordinal}.logical_bits") == directory_row["logical_bits"], f"manifest block logical_bits mismatch: {ordinal}")

    allocation = preencoding_manifest.get("allocation")
    require(isinstance(allocation, dict), "preencoding manifest allocation missing")
    require(list(allocation.get("profile_ids", [])) == [row["profile_q"] for row in container["rows"]], "preencoding manifest allocation profile_ids mismatch")

    require(
        (allocation_lock.get("schema"), allocation_lock.get("status"))
        == ("strata_xklt_sc_v2_allocation_lock_v1", "allocation_sealed_before_first_encoder_invocation"),
        "allocation lock schema/status mismatch",
    )
    require(allocation_lock.get("protocol_mode") == protocol_mode, "allocation lock protocol_mode mismatch")
    require(allocation_lock.get("manifest_sha256") == preencoding_manifest_file_hash, "allocation lock manifest sha mismatch")
    for key in ("bindings", "assets", "physical_format", "allocation", "blocks"):
        require(
            canonical_json_bytes(allocation_lock.get(key)) == canonical_json_bytes(preencoding_manifest.get(key)),
            f"allocation lock mismatch for {key}",
        )

    require(
        (one_shot_intent.get("schema"), one_shot_intent.get("status"))
        == ("strata_xklt_sc_v2_one_shot_intent_v1", "sealed_before_first_encoder_invocation"),
        "one-shot intent schema/status mismatch",
    )
    require(one_shot_intent.get("protocol_mode") == protocol_mode, "one-shot intent protocol_mode mismatch")
    require(
        one_shot_intent.get("allocation_lock_file_sha256") == allocation_file_hash,
        "one-shot intent allocation lock file hash mismatch",
    )
    require(
        one_shot_intent.get("allocation_lock_internal_sha256") == allocation_internal,
        "one-shot intent allocation lock internal hash mismatch",
    )
    require(one_shot_intent.get("manifest_sha256") == preencoding_manifest_file_hash, "one-shot intent manifest sha mismatch")
    require(
        expect_int(one_shot_intent.get("encoder_invocations_planned"), "one-shot intent encoder_invocations_planned") == BLOCKS,
        "one-shot intent encoder_invocations_planned mismatch",
    )
    require(
        one_shot_intent.get("retry_resume_or_adaptive_rate_change_allowed") is False,
        "one-shot intent retry/resume permission mismatch",
    )
    runtime_freeze = one_shot_intent.get("runtime_freeze")
    require(isinstance(runtime_freeze, dict), "one-shot intent runtime_freeze missing")
    runtime_codec = runtime_freeze.get("codec_freeze")
    require(isinstance(runtime_codec, dict), "one-shot intent runtime_freeze.codec_freeze missing")
    require(runtime_codec.get("file_sha256") == codec_file_hash, "one-shot intent runtime codec-freeze file hash mismatch")
    require(
        runtime_codec.get("internal_lock_sha256") == codec_internal,
        "one-shot intent runtime codec-freeze internal hash mismatch",
    )
    runtime_artifacts = runtime_freeze.get("artifacts")
    require(isinstance(runtime_artifacts, dict), "one-shot intent runtime_freeze.artifacts missing")
    require(set(runtime_artifacts) == set(RUNTIME_ARTIFACT_FREEZE_KEYS), "one-shot intent runtime artifact key set mismatch")
    for runtime_name, freeze_name in RUNTIME_ARTIFACT_FREEZE_KEYS.items():
        row = runtime_artifacts[runtime_name]
        require(isinstance(row, dict), f"one-shot intent runtime artifact {runtime_name} must be an object")
        require(isinstance(row.get("path"), str) and row["path"], f"one-shot intent runtime artifact {runtime_name} path missing")
        require(
            require_sha256(row.get("sha256"), f"one-shot intent runtime artifact {runtime_name}.sha256")
            == frozen_artifacts.get(freeze_name),
            f"one-shot intent runtime artifact {runtime_name} hash mismatch",
        )
    require(one_shot_intent.get("runner_sha256") == frozen_artifacts.get("one_shot_runner"), "one-shot intent runner_sha256 mismatch")
    require(one_shot_intent.get("encoder_sha256") == frozen_artifacts.get("polar_encoder"), "one-shot intent encoder_sha256 mismatch")

    require(
        (one_shot_summary.get("schema"), one_shot_summary.get("status"))
        == ("strata_xklt_sc_v2_one_shot_summary_v1", "one-shot physical artifact complete"),
        "one-shot summary schema/status mismatch",
    )
    require(one_shot_summary.get("protocol_mode") == protocol_mode, "one-shot summary protocol_mode mismatch")
    require(one_shot_summary.get("allocation_lock_file_sha256") == allocation_file_hash, "one-shot summary allocation lock file hash mismatch")
    require(one_shot_summary.get("allocation_lock_internal_sha256") == allocation_internal, "one-shot summary allocation lock internal hash mismatch")
    require(one_shot_summary.get("intent_sha256") == intent_file_hash, "one-shot summary intent sha mismatch")
    require(expect_int(one_shot_summary.get("encoder_invocations"), "one-shot summary encoder_invocations") == BLOCKS, "one-shot summary encoder_invocations mismatch")
    require(expect_int(one_shot_summary.get("retries"), "one-shot summary retries") == 0, "one-shot summary retries mismatch")
    require(expect_int(one_shot_summary.get("resumes"), "one-shot summary resumes") == 0, "one-shot summary resumes mismatch")
    require(expect_int(one_shot_summary.get("postencoding_profile_changes"), "one-shot summary postencoding_profile_changes") == 0, "one-shot summary profile change mismatch")
    encoded_blocks = one_shot_summary.get("encoded_blocks")
    require(isinstance(encoded_blocks, list) and len(encoded_blocks) == BLOCKS, "one-shot summary encoded_blocks mismatch")
    for ordinal, (encoded, directory_row) in enumerate(zip(encoded_blocks, container["rows"], strict=True)):
        require(isinstance(encoded, dict), f"one-shot summary encoded block {ordinal} must be an object")
        require(expect_int(encoded.get("block_ordinal"), f"encoded block {ordinal}.block_ordinal") == ordinal, f"one-shot summary block ordinal mismatch: {ordinal}")
        require(expect_int(encoded.get("encoder_invocations"), f"encoded block {ordinal}.encoder_invocations") == 1, f"one-shot summary block encoder invocation mismatch: {ordinal}")
        require(expect_int(encoded.get("logical_bits"), f"encoded block {ordinal}.logical_bits") == directory_row["logical_bits"], f"one-shot summary block logical_bits mismatch: {ordinal}")

    summary_physical = one_shot_summary.get("physical")
    require(isinstance(summary_physical, dict), "one-shot summary physical missing")
    nominal_bits = expect_int(allocation.get("nominal_profile_bits"), "preencoding allocation nominal_profile_bits")
    require(summary_physical.get("artifact_sha256") == container["raw_sha256"], "one-shot summary artifact sha mismatch")
    require(summary_physical.get("physical_bytes") == CONTAINER_BYTES, "one-shot summary physical_bytes mismatch")
    require(summary_physical.get("physical_bits") == PHYSICAL_BITS, "one-shot summary physical_bits mismatch")
    require(same_float(summary_physical.get("physical_bpw"), PHYSICAL_BPW), "one-shot summary physical_bpw mismatch")
    require(summary_physical.get("integer_2p15_gate_passed") is True, "one-shot summary integer gate mismatch")
    require(summary_physical.get("directory_bytes") == DIRECTORY_BYTES, "one-shot summary directory_bytes mismatch")
    require(summary_physical.get("logical_payload_bits") == container["valid_payload_bits"], "one-shot summary logical_payload_bits mismatch")
    require(summary_physical.get("nominal_profile_bits") == nominal_bits, "one-shot summary nominal_profile_bits mismatch")
    require(
        summary_physical.get("observed_logical_minus_nominal_bits") == container["valid_payload_bits"] - nominal_bits,
        "one-shot summary observed logical minus nominal mismatch",
    )
    require(summary_physical.get("payload_byte_count") == container["used_payload_bytes"], "one-shot summary payload_byte_count mismatch")
    require(
        summary_physical.get("zero_reservoir_tail_bytes") == container["zero_terminal_bytes"],
        "one-shot summary zero_reservoir_tail_bytes mismatch",
    )
    require(summary_physical.get("reservoir_fit") is True, "one-shot summary reservoir_fit mismatch")

    require(
        (independent_audit.get("schema"), independent_audit.get("audit_execution_passed"))
        == ("strata_v2_klt_mixed_independent_decode_audit_v1", True),
        "independent audit schema/audit_execution_passed mismatch",
    )
    primary_claim_gate = independent_audit.get("primary_claim_gate")
    require(isinstance(primary_claim_gate, dict), "independent audit primary_claim_gate missing")
    require(independent_audit.get("passed") == primary_claim_gate.get("passed"), "independent audit passed mismatch with primary gate")
    require(primary_claim_gate.get("rule") == PRIMARY_RULE, "independent audit primary rule mismatch")
    conditions = primary_claim_gate.get("conditions")
    require(isinstance(conditions, dict), "independent audit primary conditions missing")
    require(set(conditions) == set(REQUIRED_AUDIT_CONDITIONS), "independent audit primary condition keys mismatch")

    inspection = independent_audit.get("container_inspection")
    require(isinstance(inspection, dict), "independent audit container_inspection missing")
    require(inspection.get("schema") == "strata_v2_klt_mixed_independent_container_inspection_v1", "independent audit inspection schema mismatch")
    require(inspection.get("passed") is True, "independent audit inspection passed mismatch")
    require(inspection.get("container_sha256") == container["raw_sha256"], "independent audit container sha mismatch")
    physical_rate = inspection.get("physical_rate")
    require(isinstance(physical_rate, dict), "independent audit physical_rate missing")
    require(physical_rate.get("weights") == WEIGHTS, "independent audit physical_rate weights mismatch")
    require(physical_rate.get("bits") == PHYSICAL_BITS, "independent audit physical_rate bits mismatch")
    require(same_float(physical_rate.get("bpw"), PHYSICAL_BPW), "independent audit physical_rate bpw mismatch")
    require(physical_rate.get("integer_cap_floor_bits") == INTEGER_CAP_BITS, "independent audit physical_rate integer cap mismatch")
    require(
        physical_rate.get("headroom_bits_to_integer_floor") == HEADROOM_BITS,
        "independent audit physical_rate headroom mismatch",
    )
    require(physical_rate.get("exact_gate") == "bits*20 <= 43*weights", "independent audit physical_rate exact_gate mismatch")
    require(physical_rate.get("passes_2p15") is True, "independent audit physical_rate passes_2p15 mismatch")

    audit_directory = inspection.get("directory")
    require(isinstance(audit_directory, dict), "independent audit directory inspection missing")
    require(audit_directory.get("profile_bytes_hex") == container["profile_bytes_hex"], "independent audit directory profile bytes mismatch")
    audit_directory_rows = audit_directory.get("rows")
    require(isinstance(audit_directory_rows, list) and len(audit_directory_rows) == BLOCKS, "independent audit directory rows mismatch")
    for ordinal, (audit_row, container_row) in enumerate(zip(audit_directory_rows, container["rows"], strict=True)):
        require(isinstance(audit_row, dict), f"independent audit directory row {ordinal} must be an object")
        require(audit_row.get("block_ordinal") == ordinal, f"independent audit directory block ordinal mismatch: {ordinal}")
        require(audit_row.get("block_log2") == container_row["block_log2"], f"independent audit directory block_log2 mismatch: {ordinal}")
        require(audit_row.get("block_values") == container_row["block_values"], f"independent audit directory block_values mismatch: {ordinal}")
        require(audit_row.get("profile_q") == container_row["profile_q"], f"independent audit directory profile_q mismatch: {ordinal}")
        require(audit_row.get("logical_bits") == container_row["logical_bits"], f"independent audit directory logical_bits mismatch: {ordinal}")
        require(audit_row.get("payload_bytes") == container_row["payload_bytes"], f"independent audit directory payload_bytes mismatch: {ordinal}")
        require(audit_row.get("payload_terminal_padding_bits") == container_row["payload_terminal_padding_bits"], f"independent audit directory payload padding mismatch: {ordinal}")
        require(audit_row.get("payload_terminal_padding_all_zero") is True, f"independent audit directory zero-padding mismatch: {ordinal}")

    audit_reservoir = inspection.get("reservoir")
    require(isinstance(audit_reservoir, dict), "independent audit reservoir inspection missing")
    require(audit_reservoir.get("valid_payload_bits") == container["valid_payload_bits"], "independent audit reservoir valid payload bits mismatch")
    require(audit_reservoir.get("used_payload_bytes") == container["used_payload_bytes"], "independent audit reservoir used payload bytes mismatch")
    require(audit_reservoir.get("zero_terminal_bytes") == container["zero_terminal_bytes"], "independent audit reservoir zero terminal bytes mismatch")
    require(audit_reservoir.get("terminal_fill_all_zero") is True, "independent audit reservoir zero-fill mismatch")

    source_lineage = independent_audit.get("source_lineage")
    require(isinstance(source_lineage, dict), "independent audit source_lineage missing")
    require(source_lineage.get("all_checks_passed") is True, "independent audit source_lineage all_checks_passed mismatch")
    require(source_lineage.get("protocol_mode") == protocol_mode, "independent audit source_lineage protocol_mode mismatch")
    require(
        source_lineage.get("blind_positive_claim_eligible") == (protocol_mode == "blind"),
        "independent audit source_lineage blind_positive_claim_eligible mismatch",
    )
    expected_lineage_bindings = {
        "selection_lock": {"file_sha256": selection_file_hash, "internal_lock_sha256": selection_internal},
        "source_lock": {"file_sha256": source_file_hash, "internal_lock_sha256": source_internal},
        "codec_freeze": {"file_sha256": codec_file_hash, "internal_lock_sha256": codec_internal},
        "codec_freeze_validation": {
            "file_sha256": validation_file_hash,
            "internal_lock_sha256": validation_internal,
        },
        "format_freeze": {"sha256": format_hash},
        "preencoding_manifest": {"sha256": preencoding_manifest_file_hash},
        "allocation_lock": {"file_sha256": allocation_file_hash, "internal_lock_sha256": allocation_internal},
        "one_shot_intent": {"sha256": intent_file_hash},
        "one_shot_summary": {"sha256": summary_file_hash},
    }
    for name, expected_fields in expected_lineage_bindings.items():
        actual = source_lineage.get(name)
        require(isinstance(actual, dict), f"independent audit source_lineage missing {name}")
        require(all(actual.get(key) == value for key, value in expected_fields.items()), f"independent audit source_lineage mismatch: {name}")
    require(
        source_lineage.get("executing_independent_auditor_sha256") == frozen_artifacts.get("independent_auditor"),
        "independent audit executing auditor hash mismatch",
    )

    scale_audit = independent_audit.get("source_staging_and_scale_audit")
    require(isinstance(scale_audit, dict), "independent audit source_staging_and_scale_audit missing")
    require(scale_audit.get("all_checks_passed") is True, "independent audit source_staging_and_scale_audit must pass")

    source_score = independent_audit.get("source_score")
    require(isinstance(source_score, dict), "independent audit source_score missing")
    matrices = source_score.get("matrices")
    require(isinstance(matrices, list) and matrices, "independent audit source_score.matrices missing")
    energy_total = float(source_score.get("source_energy_sum_fp64"))
    sse_total = float(source_score.get("sse_sum_fp64"))
    reported_rel_mse = float(source_score.get("energy_weighted_relative_mse"))
    require(math.isfinite(energy_total) and energy_total > 0.0, "independent audit source energy must be positive finite")
    require(math.isfinite(sse_total) and sse_total >= 0.0, "independent audit SSE must be finite nonnegative")
    recomputed_sse = 0.0
    recomputed_energy = 0.0
    for index, row in enumerate(matrices):
        require(isinstance(row, dict), f"independent audit source_score matrix {index} must be an object")
        matrix_sse = float(row.get("sse_fp64"))
        matrix_energy = float(row.get("source_energy_fp64"))
        matrix_rel = float(row.get("relative_mse"))
        require(math.isfinite(matrix_energy) and matrix_energy > 0.0, f"independent audit source_score matrix energy invalid: {index}")
        require(math.isfinite(matrix_sse) and matrix_sse >= 0.0, f"independent audit source_score matrix sse invalid: {index}")
        require(same_float(matrix_rel, matrix_sse / matrix_energy, atol=2e-15), f"independent audit matrix relative_mse mismatch: {index}")
        recomputed_sse += matrix_sse
        recomputed_energy += matrix_energy
    require(same_float(recomputed_sse, sse_total, atol=2e-12), "independent audit source_score sse_sum mismatch")
    require(same_float(recomputed_energy, energy_total, atol=2e-12), "independent audit source_score energy_sum mismatch")
    require(same_float(reported_rel_mse, sse_total / energy_total, atol=2e-15), "independent audit pooled relative MSE mismatch")
    require(same_float(source_score.get("gaussian_limit_at_2p15"), GAUSSIAN_LIMIT), "independent audit gaussian limit mismatch")
    require(
        source_score.get("beats_gaussian_limit") == (reported_rel_mse < GAUSSIAN_LIMIT),
        "independent audit gaussian pass flag mismatch",
    )
    if "original_source_energy_fp64" in scale_audit:
        require(
            same_float(scale_audit.get("original_source_energy_fp64"), energy_total, atol=2e-12),
            "independent audit scale audit/source score energy mismatch",
        )

    require(conditions["physical_rate_at_most_2p15"] == (physical_rate["passes_2p15"] is True), "independent audit primary condition mismatch: physical_rate_at_most_2p15")
    require(conditions["complete_source_lineage_present"] is True, "independent audit primary condition mismatch: complete_source_lineage_present")
    require(conditions["blind_protocol_mode"] == (protocol_mode == "blind"), "independent audit primary condition mismatch: blind_protocol_mode")
    require(
        conditions["source_staging_label_scale_and_dp_audit_passed"] == (scale_audit["all_checks_passed"] is True),
        "independent audit primary condition mismatch: source_staging_label_scale_and_dp_audit_passed",
    )
    require(
        conditions["source_domain_mse_below_gaussian_limit"] == (source_score["beats_gaussian_limit"] is True),
        "independent audit primary condition mismatch: source_domain_mse_below_gaussian_limit",
    )

    claim = manifest["claim"]
    require(expect_bool(claim.get("audit_passed"), "claim.audit_passed") == independent_audit.get("passed"), "manifest claim audit_passed mismatch")
    require(
        expect_bool(claim.get("primary_claim_passed"), "claim.primary_claim_passed") == primary_claim_gate.get("passed"),
        "manifest claim primary_claim_passed mismatch",
    )
    require(
        expect_bool(claim.get("audit_execution_passed"), "claim.audit_execution_passed") == independent_audit.get("audit_execution_passed"),
        "manifest claim audit_execution_passed mismatch",
    )

    require(
        tamper_report.get("schema") == "strata_v2_klt_independent_lineage_tamper_tests_v1",
        "tamper report schema mismatch",
    )
    require(tamper_report.get("protocol_mode") == protocol_mode, "tamper report protocol_mode mismatch")
    require(tamper_report.get("container_sha256") == container["raw_sha256"], "tamper report container sha mismatch")
    require(
        tamper_report.get("auditor_sha256") == frozen_artifacts.get("independent_auditor"),
        "tamper report auditor sha mismatch",
    )
    require(
        tamper_report.get("executing_tamper_harness_sha256") == frozen_artifacts.get("lineage_tamper_test"),
        "tamper report tamper harness sha mismatch",
    )
    require(expect_int(tamper_report.get("tamper_count"), "tamper_report.tamper_count") == 10, "tamper report tamper_count mismatch")
    require(tamper_report.get("exact_unique_tamper_name_set") is True, "tamper report exact_unique_tamper_name_set mismatch")
    observed_expected_names = tamper_report.get("expected_tamper_names")
    require(isinstance(observed_expected_names, list), "tamper report expected_tamper_names missing")
    require(set(observed_expected_names) == EXPECTED_TAMPER_NAMES, "tamper report expected_tamper_names mismatch")
    tamper_rows = tamper_report.get("tamper_rows")
    require(isinstance(tamper_rows, list) and len(tamper_rows) == 10, "tamper report tamper_rows mismatch")
    observed_names: set[str] = set()
    for index, row in enumerate(tamper_rows):
        require(isinstance(row, dict), f"tamper report row {index} must be an object")
        name = str(row.get("tamper"))
        observed_names.add(name)
        require(name in EXPECTED_TAMPER_NAMES, f"tamper report unexpected tamper name: {name}")
        require(row.get("rejected") is True, f"tamper report row {index} rejected flag mismatch")
        require(isinstance(row.get("error"), str) and row["error"], f"tamper report row {index} error missing")
    require(observed_names == EXPECTED_TAMPER_NAMES, "tamper report observed tamper-name set mismatch")
    require(expect_bool(claim.get("tamper_report_passed"), "claim.tamper_report_passed") == tamper_report.get("passed"), "manifest claim tamper_report_passed mismatch")

    return {
        "status": "verified",
        "manifest": str(manifest_path.resolve()),
        "protocol_mode": protocol_mode,
        "files_verified": len(file_rows),
        "container_sha256": container["raw_sha256"],
        "container_bytes": CONTAINER_BYTES,
        "physical_bpw": PHYSICAL_BPW,
        "headroom_bits_to_integer_floor": HEADROOM_BITS,
        "logical_payload_bits": container["valid_payload_bits"],
        "payload_byte_count": container["used_payload_bytes"],
        "zero_reservoir_tail_bytes": container["zero_terminal_bytes"],
        "independent_audit_passed": independent_audit["passed"],
        "primary_claim_gate_passed": primary_claim_gate["passed"],
        "audit_execution_passed": independent_audit["audit_execution_passed"],
        "source_energy_sum_fp64": energy_total,
        "sse_sum_fp64": sse_total,
        "energy_weighted_relative_mse": reported_rel_mse,
        "beats_gaussian_limit": source_score["beats_gaussian_limit"],
        "tamper_report_passed": tamper_report["passed"],
        "tamper_count": tamper_report["tamper_count"],
        "local_frozen_artifacts_verified": len(local_frozen_hashes),
        "withheld_frozen_artifacts_hash_bound": sorted(withheld_frozen_hashes),
        "external_frozen_hashes_presence_checked": external_only_frozen,
    }


def _seal(document: dict[str, Any]) -> dict[str, Any]:
    clean = dict(document)
    clean.pop("lock_sha256", None)
    result = dict(clean)
    result["lock_sha256"] = sha256_bytes(canonical_json_bytes(clean))
    return result


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def build_self_test_bundle(root: Path) -> Path:
    repo_root = root.resolve()
    release_dir = repo_root / "release"
    bundle_dir = repo_root / "bundle"
    release_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    frozen_artifact_paths = {
        "format": "docs/FORMAT.md",
        "common": "strata_v2_codec/common.py",
        "emitter": "strata_v2_codec/emit_and_lock.py",
        "polar_encoder": "strata_v2_codec/polar_encoder.py",
        "one_shot_runner": "strata_v2_codec/run_one_shot.py",
        "base_cupy_encoder": "agent_polaris_qwen_rht_encoder.py",
        "procedural_q31_bec": "bg_codec_bec_encoder.py",
        "independent_auditor": "strata_v2_klt_mixed_independent_auditor_v1.py",
        "lineage_tamper_test": "strata_v2_klt_lineage_tamper_tests_v1.py",
        "freeze_validator": "blind_protocol_v2/validate_codec_freeze.py",
    }
    frozen_artifact_sha256s: dict[str, str] = {}
    for name, relative in frozen_artifact_paths.items():
        frozen_artifact_sha256s[name] = _write_text(repo_root / relative, f"synthetic {name}\n")
    for name in ("numpy_distribution_tree", "cupy_distribution_tree", "scipy_distribution_tree", "cuda_pathfinder_distribution_tree", "python_interpreter"):
        frozen_artifact_sha256s[name] = hashlib.sha256(f"synthetic {name}".encode("utf-8")).hexdigest()

    profile_ids = [11 + index for index in range(BLOCKS)]
    logical_bits = [8 + (index % 3) for index in range(BLOCKS)]
    directory = bytearray()
    payloads = []
    for ordinal, (profile_id, logical) in enumerate(zip(profile_ids, logical_bits, strict=True)):
        directory.extend(DIRECTORY_RECORD.pack(profile_id, 1.0 + ordinal / 16.0, logical))
        payload_bytes = (logical + 7) // 8
        payload = bytes(((0xA0 + ordinal) & 0xFF,) * payload_bytes)
        if logical % 8:
            mask = ~((1 << (payload_bytes * 8 - logical)) - 1) & 0xFF
            payload = payload[:-1] + bytes((payload[-1] & mask,))
        payloads.append(payload)
    streams = b"".join(payloads)
    reservoir = streams + bytes(RESERVOIR_BYTES - len(streams))
    header = bytes((index % 251 for index in range(HEADER_BYTES)))
    route = bytes((17 + index) % 251 for index in range(ROUTE_BYTES))
    labels = bytes((index % 8) for index in range(LABEL_BYTES))
    container_bytes = header + route + labels + bytes(directory) + reservoir
    container_path = bundle_dir / "strata_xklt_sc_v2.bin"
    container_path.write_bytes(container_bytes)
    container_hash = sha256_file(container_path)
    profiles_blob = bytes(profile_ids)
    header_hash = sha256_bytes(header)
    route_hash = sha256_bytes(route)
    labels_hash = sha256_bytes(labels)
    profiles_hash = sha256_bytes(profiles_blob)
    used_payload_bytes = len(streams)
    valid_payload_bits = sum(logical_bits)
    nominal_profile_bits = valid_payload_bits + 5

    selection = _seal(
        {
            "schema": "int2-qwen-blind-selection-proposal-v2",
            "status": "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
            "checkpoint": {"repo": "synthetic/repo", "revision": "deadbeef"},
        }
    )
    selection_path = bundle_dir / "selection.lock.json"
    write_json(selection_path, selection)
    selection_file_hash = sha256_file(selection_path)

    codec_freeze = _seal(
        {
            "schema": "strata_xklt_sc_v2_codec_freeze_v1",
            "status": "frozen_before_blind_source_access",
            "selection_lock_file_sha256": selection_file_hash,
            "selection_lock_sha256": selection["lock_sha256"],
            "route_file_sha256": route_hash,
            "physical_rate_limit_bpw": 2.15,
            "primary_mse_threshold": GAUSSIAN_LIMIT,
            "gaussian_mse_reference": GAUSSIAN_LIMIT,
            "allocator_frozen": True,
            "architecture_frozen": True,
            "no_retry_resume_or_postaccess_tuning": True,
            "physical_ledger": {
                "weights": WEIGHTS,
                "physical_bytes": CONTAINER_BYTES,
                "physical_bits": PHYSICAL_BITS,
                "physical_bpw": PHYSICAL_BPW,
                "integer_2p15_cap_bits": INTEGER_CAP_BITS,
                "headroom_bits": HEADROOM_BITS,
                "header_bits": HEADER_BYTES * 8,
                "route_bits": ROUTE_BYTES * 8,
                "label_bits": LABEL_BYTES * 8,
                "directory_bits": DIRECTORY_BYTES * 8,
                "reservoir_bytes": RESERVOIR_BYTES,
                "reservoir_bits": RESERVOIR_BYTES * 8,
                "global_no_retry_reserve_bits": GLOBAL_RESERVE_BITS,
                "nominal_profile_budget_bits": NOMINAL_PROFILE_BUDGET_BITS,
            },
            "expected_source_lock": {
                "schema": "int2-qwen-blind-source-finalization-v2",
                "status": "all_locked_sources_materialized_and_hash_finalized",
                "matrix_count": 18,
                "block_count": 108,
                "source_values": WEIGHTS,
                "source_bytes": 2 * WEIGHTS,
                "dtype": "BF16",
            },
            "preaccess_state": {"frozen": True},
            "frozen_artifact_paths": frozen_artifact_paths,
            "frozen_artifact_sha256s": frozen_artifact_sha256s,
        }
    )
    codec_freeze_path = bundle_dir / "codec_freeze.lock.json"
    write_json(codec_freeze_path, codec_freeze)
    codec_freeze_file_hash = sha256_file(codec_freeze_path)

    validation = _seal(
        {
            "schema": "strata_xklt_sc_v2_codec_freeze_validation_v1",
            "status": "validated_before_blind_source_access",
            "passed": True,
            "freeze_path": "blind_protocol_v2/codec_freeze.lock.json",
            "freeze_file_sha256": codec_freeze_file_hash,
            "freeze_internal_lock_sha256": codec_freeze["lock_sha256"],
            "executing_validator_sha256": frozen_artifact_sha256s["freeze_validator"],
            "frozen_artifact_count": len(frozen_artifact_sha256s),
            "development_pooled_relative_mse": 0.04,
            "gaussian_mse_reference": GAUSSIAN_LIMIT,
            "physical_bits": PHYSICAL_BITS,
            "physical_bpw": PHYSICAL_BPW,
            "preaccess_state": codec_freeze["preaccess_state"],
        }
    )
    validation_path = bundle_dir / "codec_freeze.validation.json"
    write_json(validation_path, validation)
    validation_file_hash = sha256_file(validation_path)

    source_lock = _seal(
        {
            "schema": "int2-qwen-blind-source-finalization-v2",
            "status": "all_locked_sources_materialized_and_hash_finalized",
            "matrix_count": 18,
            "block_count": 108,
            "source_values": WEIGHTS,
            "source_bytes": 2 * WEIGHTS,
            "dtype": "BF16",
            "selection_lock_sha256": selection["lock_sha256"],
            "codec_freeze": {
                "file_sha256": codec_freeze_file_hash,
                "internal_lock_sha256": codec_freeze["lock_sha256"],
            },
            "codec_freeze_validation": {
                "file_sha256": validation_file_hash,
                "internal_lock_sha256": validation["lock_sha256"],
            },
        }
    )
    source_lock_path = bundle_dir / "source.lock.json"
    write_json(source_lock_path, source_lock)
    source_lock_file_hash = sha256_file(source_lock_path)

    preencoding_manifest = {
        "schema": "strata_xklt_sc_v2_preencoding_manifest_v1",
        "status": "complete_and_allocation_sealed_before_encoding",
        "protocol_mode": "blind",
        "bindings": {
            "protocol_mode": "blind",
            "selection_lock": {
                "file_sha256": selection_file_hash,
                "internal_lock_sha256": selection["lock_sha256"],
            },
            "source_lock": {
                "file_sha256": source_lock_file_hash,
                "internal_lock_sha256": source_lock["lock_sha256"],
                "selection_lock_sha256": selection["lock_sha256"],
            },
            "codec_freeze": {
                "file_sha256": codec_freeze_file_hash,
                "internal_lock_sha256": codec_freeze["lock_sha256"],
            },
            "codec_freeze_validation": {
                "file_sha256": validation_file_hash,
                "internal_lock_sha256": validation["lock_sha256"],
            },
            "route": {"sha256": route_hash, "bytes": ROUTE_BYTES},
            "format_freeze": {"sha256": frozen_artifact_sha256s["format"]},
            "emitter": {"sha256": frozen_artifact_sha256s["emitter"]},
            "common": {"sha256": frozen_artifact_sha256s["common"]},
        },
        "assets": {
            "header.bin": {"bytes": HEADER_BYTES, "sha256": header_hash},
            "route.bin": {"bytes": ROUTE_BYTES, "sha256": route_hash},
            "labels_3bit.bin": {"bytes": LABEL_BYTES, "sha256": labels_hash},
            "profiles.bin": {"bytes": BLOCKS, "sha256": profiles_hash},
        },
        "physical_format": {"container_bytes": CONTAINER_BYTES},
        "allocation": {
            "profile_ids": profile_ids,
            "nominal_profile_bits": nominal_profile_bits,
        },
        "blocks": [
            {
                "block_ordinal": ordinal,
                "block_log2": block_log2_for_ordinal(ordinal),
                "values": block_values_for_ordinal(ordinal),
                "profile_id": profile_id,
                "logical_bits": logical,
            }
            for ordinal, (profile_id, logical) in enumerate(zip(profile_ids, logical_bits, strict=True))
        ],
    }
    preencoding_manifest_path = bundle_dir / "preencoding_manifest.json"
    write_json(preencoding_manifest_path, preencoding_manifest)
    preencoding_manifest_file_hash = sha256_file(preencoding_manifest_path)

    allocation_lock = _seal(
        {
            "schema": "strata_xklt_sc_v2_allocation_lock_v1",
            "status": "allocation_sealed_before_first_encoder_invocation",
            "protocol_mode": "blind",
            "manifest_sha256": preencoding_manifest_file_hash,
            "bindings": preencoding_manifest["bindings"],
            "assets": preencoding_manifest["assets"],
            "physical_format": preencoding_manifest["physical_format"],
            "allocation": preencoding_manifest["allocation"],
            "blocks": preencoding_manifest["blocks"],
        }
    )
    allocation_lock_path = bundle_dir / "allocation.lock.json"
    write_json(allocation_lock_path, allocation_lock)
    allocation_lock_file_hash = sha256_file(allocation_lock_path)

    runtime_artifacts = {
        runtime_name: {
            "path": f"synthetic/{runtime_name}",
            "sha256": frozen_artifact_sha256s[freeze_name],
        }
        for runtime_name, freeze_name in RUNTIME_ARTIFACT_FREEZE_KEYS.items()
    }
    one_shot_intent = {
        "schema": "strata_xklt_sc_v2_one_shot_intent_v1",
        "status": "sealed_before_first_encoder_invocation",
        "allocation_lock_file_sha256": allocation_lock_file_hash,
        "allocation_lock_internal_sha256": allocation_lock["lock_sha256"],
        "manifest_sha256": preencoding_manifest_file_hash,
        "runner_sha256": frozen_artifact_sha256s["one_shot_runner"],
        "encoder_sha256": frozen_artifact_sha256s["polar_encoder"],
        "protocol_mode": "blind",
        "runtime_freeze": {
            "codec_freeze": {
                "path": str(codec_freeze_path),
                "file_sha256": codec_freeze_file_hash,
                "internal_lock_sha256": codec_freeze["lock_sha256"],
            },
            "artifacts": runtime_artifacts,
            "packages": {"synthetic": True},
            "cuda": {"synthetic": True},
        },
        "workers": 1,
        "encoder_invocations_planned": BLOCKS,
        "retry_resume_or_adaptive_rate_change_allowed": False,
    }
    one_shot_intent_path = bundle_dir / "ONE_SHOT_INTENT.json"
    write_json(one_shot_intent_path, one_shot_intent)
    one_shot_intent_file_hash = sha256_file(one_shot_intent_path)

    one_shot_summary = {
        "schema": "strata_xklt_sc_v2_one_shot_summary_v1",
        "status": "one-shot physical artifact complete",
        "protocol_mode": "blind",
        "allocation_lock_file_sha256": allocation_lock_file_hash,
        "allocation_lock_internal_sha256": allocation_lock["lock_sha256"],
        "intent_sha256": one_shot_intent_file_hash,
        "encoder_invocations": BLOCKS,
        "retries": 0,
        "resumes": 0,
        "postencoding_profile_changes": 0,
        "encoded_blocks": [
            {"block_ordinal": ordinal, "encoder_invocations": 1, "logical_bits": logical}
            for ordinal, logical in enumerate(logical_bits)
        ],
        "physical": {
            "artifact_relpath": "strata_xklt_sc_v2.bin",
            "artifact_sha256": container_hash,
            "physical_bytes": CONTAINER_BYTES,
            "physical_bits": PHYSICAL_BITS,
            "physical_bpw": PHYSICAL_BPW,
            "integer_2p15_gate_passed": True,
            "directory_bytes": DIRECTORY_BYTES,
            "logical_payload_bits": valid_payload_bits,
            "nominal_profile_bits": nominal_profile_bits,
            "observed_logical_minus_nominal_bits": valid_payload_bits - nominal_profile_bits,
            "payload_byte_count": used_payload_bytes,
            "zero_reservoir_tail_bytes": RESERVOIR_BYTES - used_payload_bytes,
            "reservoir_fit": True,
            "encoder_side_staging_energy_relative_mse": 0.02,
        },
    }
    one_shot_summary_path = bundle_dir / "summary.json"
    write_json(one_shot_summary_path, one_shot_summary)
    one_shot_summary_file_hash = sha256_file(one_shot_summary_path)

    audit_directory_rows = []
    cursor = 0
    for ordinal, logical in enumerate(logical_bits):
        payload_bytes = (logical + 7) // 8
        audit_directory_rows.append(
            {
                "block_ordinal": ordinal,
                "block_log2": block_log2_for_ordinal(ordinal),
                "block_values": block_values_for_ordinal(ordinal),
                "profile_q": profile_ids[ordinal],
                "logical_bits": logical,
                "payload_bytes": payload_bytes,
                "payload_terminal_padding_bits": payload_bytes * 8 - logical,
                "payload_terminal_padding_all_zero": True,
                "reservoir_byte_begin": cursor,
                "reservoir_byte_end_exclusive": cursor + payload_bytes,
            }
        )
        cursor += payload_bytes

    source_score = {
        "source_root": str(bundle_dir),
        "matrices": [
            {
                "matrix_ordinal": 0,
                "tensor": "synthetic.0",
                "source_path": "synthetic/0",
                "source_sha256": hashlib.sha256(b"m0").hexdigest(),
                "sse_fp64": 1.5,
                "source_energy_fp64": 50.0,
                "relative_mse": 0.03,
            },
            {
                "matrix_ordinal": 1,
                "tensor": "synthetic.1",
                "source_path": "synthetic/1",
                "source_sha256": hashlib.sha256(b"m1").hexdigest(),
                "sse_fp64": 2.5,
                "source_energy_fp64": 50.0,
                "relative_mse": 0.05,
            },
        ],
        "sse_sum_fp64": 4.0,
        "source_energy_sum_fp64": 100.0,
        "energy_weighted_relative_mse": 0.04,
        "gaussian_limit_at_2p15": GAUSSIAN_LIMIT,
        "beats_gaussian_limit": True,
    }
    independent_audit = {
        "schema": "strata_v2_klt_mixed_independent_decode_audit_v1",
        "passed": True,
        "audit_execution_passed": True,
        "primary_claim_gate": {
            "passed": True,
            "rule": PRIMARY_RULE,
            "conditions": {
                "physical_rate_at_most_2p15": True,
                "complete_source_lineage_present": True,
                "blind_protocol_mode": True,
                "source_staging_label_scale_and_dp_audit_passed": True,
                "source_domain_mse_below_gaussian_limit": True,
            },
        },
        "container_inspection": {
            "schema": "strata_v2_klt_mixed_independent_container_inspection_v1",
            "passed": True,
            "container": str(container_path),
            "container_sha256": container_hash,
            "directory": {
                "profile_bytes_hex": profiles_blob.hex(),
                "rows": audit_directory_rows,
            },
            "reservoir": {
                "valid_payload_bits": valid_payload_bits,
                "used_payload_bytes": used_payload_bytes,
                "zero_terminal_bytes": RESERVOIR_BYTES - used_payload_bytes,
                "terminal_fill_all_zero": True,
            },
            "physical_rate": {
                "weights": WEIGHTS,
                "bits": PHYSICAL_BITS,
                "bpw": PHYSICAL_BPW,
                "integer_cap_floor_bits": INTEGER_CAP_BITS,
                "headroom_bits_to_integer_floor": HEADROOM_BITS,
                "exact_gate": "bits*20 <= 43*weights",
                "passes_2p15": True,
            },
        },
        "source_lineage": {
            "all_checks_passed": True,
            "protocol_mode": "blind",
            "blind_positive_claim_eligible": True,
            "selection_lock": {
                "path": str(selection_path),
                "file_sha256": selection_file_hash,
                "internal_lock_sha256": selection["lock_sha256"],
            },
            "source_lock": {
                "path": str(source_lock_path),
                "file_sha256": source_lock_file_hash,
                "internal_lock_sha256": source_lock["lock_sha256"],
            },
            "codec_freeze": {
                "path": str(codec_freeze_path),
                "file_sha256": codec_freeze_file_hash,
                "internal_lock_sha256": codec_freeze["lock_sha256"],
            },
            "codec_freeze_validation": {
                "path": str(validation_path),
                "file_sha256": validation_file_hash,
                "internal_lock_sha256": validation["lock_sha256"],
            },
            "format_freeze": {
                "path": str(repo_root / frozen_artifact_paths["format"]),
                "sha256": frozen_artifact_sha256s["format"],
            },
            "preencoding_manifest": {"path": str(preencoding_manifest_path), "sha256": preencoding_manifest_file_hash},
            "allocation_lock": {
                "path": str(allocation_lock_path),
                "file_sha256": allocation_lock_file_hash,
                "internal_lock_sha256": allocation_lock["lock_sha256"],
            },
            "one_shot_intent": {"path": str(one_shot_intent_path), "sha256": one_shot_intent_file_hash},
            "one_shot_summary": {"path": str(one_shot_summary_path), "sha256": one_shot_summary_file_hash},
            "executing_independent_auditor_sha256": frozen_artifact_sha256s["independent_auditor"],
        },
        "source_staging_and_scale_audit": {
            "all_checks_passed": True,
            "original_source_energy_fp64": 100.0,
        },
        "source_score": source_score,
    }
    independent_audit_path = bundle_dir / "independent_decode_audit.json"
    write_json(independent_audit_path, independent_audit)

    tamper_report = {
        "schema": "strata_v2_klt_independent_lineage_tamper_tests_v1",
        "passed": True,
        "protocol_mode": "blind",
        "container_sha256": container_hash,
        "auditor_sha256": frozen_artifact_sha256s["independent_auditor"],
        "executing_tamper_harness_sha256": frozen_artifact_sha256s["lineage_tamper_test"],
        "tamper_count": 10,
        "exact_unique_tamper_name_set": True,
        "expected_tamper_names": sorted(EXPECTED_TAMPER_NAMES),
        "tamper_rows": [
            {"tamper": name, "rejected": True, "error": f"synthetic rejection for {name}"}
            for name in sorted(EXPECTED_TAMPER_NAMES)
        ],
    }
    tamper_report_path = bundle_dir / "independent_lineage_tamper_tests.json"
    write_json(tamper_report_path, tamper_report)

    manifest = {
        "artifact": {"schema": MANIFEST_SCHEMA, "protocol_mode": "blind", "weights": WEIGHTS},
        "claim": {
            "audit_passed": True,
            "primary_claim_passed": True,
            "audit_execution_passed": True,
            "tamper_report_passed": True,
        },
        "files": [
            {
                "path": str(path.relative_to(repo_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
                "classification": "byte_bound_evidence",
            }
            for role, path in (
                ("selection_lock", selection_path),
                ("source_lock", source_lock_path),
                ("codec_freeze", codec_freeze_path),
                ("codec_freeze_validation", validation_path),
                ("format_freeze", repo_root / frozen_artifact_paths["format"]),
                ("preencoding_manifest", preencoding_manifest_path),
                ("allocation_lock", allocation_lock_path),
                ("one_shot_intent", one_shot_intent_path),
                ("one_shot_summary", one_shot_summary_path),
                ("container", container_path),
                ("independent_audit", independent_audit_path),
                ("tamper_report", tamper_report_path),
            )
        ],
    }
    manifest_path = release_dir / "strata_v2_release_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def run_self_test() -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="strata_v2_release_selftest_"))
    try:
        manifest_path = build_self_test_bundle(temp_root)
        result = verify_release(temp_root, manifest_path)
        result["self_test"] = True
        result["temp_root"] = str(temp_root)
        return result
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="repository root; defaults to INT2_Q_C")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="release manifest path relative to repo root unless absolute",
    )
    parser.add_argument("--self-test", action="store_true", help="run a synthetic dependency-free self-test")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return

    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else (repo_root / args.manifest)
    print(json.dumps(verify_release(repo_root, manifest_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
