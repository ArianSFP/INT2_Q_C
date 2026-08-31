#!/usr/bin/env python3
"""Build the STRATA-XKLT-SC v2 codec freeze before blind-v2 access.

The builder reads metadata, source-free code, and already-opened development
evidence only.  It refuses to run once any v2 materializer, source lock,
unblinded directory, selected BF16 payload, or full checkpoint shard exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_v2_codec import common


SELECTION_FILE_SHA256 = "528250d8c6bac52dfdf64958d7f4929a115ff68d907a47880cab85d532aade14"
SELECTION_INTERNAL_SHA256 = "cd8cb70ca7509d2ddd4899df8a7047b7b8f47d381b637e2eb497db9ecd4eb9f8"
ROUTE_FILE_SHA256 = "94feb3564fe0c3eddfc745703f1f6001b5ae316e7146209e6b45323cdf81697c"
GAUSSIAN_MSE = math.exp2(-4.3)
WEIGHTS = 28_311_552
PHYSICAL_BYTES = 7_608_729
PHYSICAL_BITS = 60_869_832
INTEGER_CAP_BITS = (43 * WEIGHTS) // 20
AUTHORIZATION_PHRASE = "AUTHORIZE SEALED QWEN V2 ONE-SHOT MATERIALIZATION"
CANONICAL_OUTPUT = "blind_protocol_v2/codec_freeze.lock.json"
CANONICAL_BUILDER = "blind_protocol_v2/build_codec_freeze_v2.py"
PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "cupy": "cupy-cuda12x",
    "scipy": "scipy",
    "cuda.pathfinder": "cuda-pathfinder",
}
TAMPER_NAMES = {
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
ENCODER_CHECK_KEYS = {
    "schema",
    "block_length",
    "distortion",
    "eta",
    "sc_seed",
    "rht_seed",
    "source_hash",
    "roundtrip_bits",
    "causal_frequencies",
    "reconstruction_indices",
}

STATIC_ARTIFACTS = {
    "format": "strata_v2_codec/FORMAT.md",
    "common": "strata_v2_codec/common.py",
    "emitter": "strata_v2_codec/emit_and_lock.py",
    "polar_encoder": "strata_v2_codec/polar_encoder.py",
    "one_shot_runner": "strata_v2_codec/run_one_shot.py",
    "base_cupy_encoder": "agent_polaris_qwen_rht_encoder.py",
    "procedural_q31_bec": "bg_codec_bec_encoder.py",
    "independent_auditor": "strata_v2_klt_mixed_independent_auditor_v1.py",
    "common_tests": "strata_v2_codec/test_common.py",
    "emitter_contract_tests": "strata_v2_codec/test_emitter_contract.py",
    "emitter_synthetic_test": "strata_v2_codec/test_emitter_synthetic.py",
    "lineage_tamper_test": "strata_v2_klt_lineage_tamper_tests_v1.py",
    "freeze_builder": "blind_protocol_v2/build_codec_freeze_v2.py",
    "freeze_validator": "blind_protocol_v2/validate_codec_freeze_v2.py",
    "selection_proposal": "blind_protocol_v2/selection.proposal.lock.json",
    "literal_route": "blind_protocol_v2/route_table.proposal.bin",
    "route_audit": "blind_protocol_v2/route_table.proposal.audit.json",
    "unopened_snapshot": "blind_protocol_v2/unopened_snapshot.audit.json",
    "v1_failure_audit": "blind_protocol_v2/v1_failure_independent_audit.json",
    "proposal_validator": "blind_protocol_v2/validate_proposal.py",
}

DEVELOPMENT_EVIDENCE = {
    "high_rate_n20_probe": "strata_v2_klt_n20_rate243359_probe.json",
    "low_rate_n21_probe": "strata_v2_klt_n21_lowrate_q12_probe.json",
    "mixed_geometry_projection": (
        "strata_v2_allocation_research/fp32_klt_projection_n20_n21_reserve65536.json"
    ),
    "cross_projection_transfer": (
        "strata_v2_cross_projection_transfer_audit_v1.json"
    ),
    "triplet_covariance_early_kill": (
        "strata_v2_allocation_research/triplet_covariance_probe_v1.json"
    ),
    "physical_savings_audit": (
        "strata_v2_allocation_research/physical_savings_audit_v1_v5/audit.json"
    ),
}


class FreezeError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeError(f"JSON object required: {path}")
    return value


def load_object_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise FreezeError(f"JSON object required for {label}: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def verify_internal_seal(value: dict[str, Any], label: str) -> str:
    clean = dict(value)
    declared = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    if declared != actual:
        raise FreezeError(f"{label} internal seal mismatch")
    return actual


def seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["lock_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_json_create_only(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def checked_file(workspace: Path, relpath: str) -> Path:
    path = (workspace / relpath).resolve(strict=True)
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise FreezeError(f"artifact escapes workspace: {path}") from exc
    if not path.is_file():
        raise FreezeError(f"artifact is not a file: {path}")
    return path


def workspace_relative(workspace: Path, path: Path, label: str) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise FreezeError(f"{label} escapes the workspace: {resolved}") from exc
    return relative.as_posix()


def distribution_receipt(module_name: str, distribution_name: str) -> dict[str, Any]:
    return common.distribution_tree_receipt(module_name, distribution_name)


def build_runtime_environment() -> tuple[dict[str, Any], dict[str, str]]:
    import cupy as cp

    python_invocation = Path(sys.executable).absolute()
    require(python_invocation.is_file(), "Python invocation path is not a file")
    python_resolved = python_invocation.resolve(strict=True)
    packages = {
        name: distribution_receipt(name, distribution)
        for name, distribution in PACKAGE_DISTRIBUTIONS.items()
    }
    device = cp.cuda.runtime.getDeviceProperties(0)
    environment = {
        "python_interpreter": {
            "invocation_path": str(python_invocation),
            "resolved_path": str(python_resolved),
            "sha256": sha256_file(python_invocation),
            "version": sys.version.split()[0],
        },
        "packages": packages,
        "cuda": {
            "cupy_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device_name": device["name"].decode(),
            "compute_capability": [int(device["major"]), int(device["minor"])],
        },
    }
    external_hashes = {"python_interpreter": sha256_file(python_invocation)}
    external_paths = {"python_interpreter": str(python_invocation)}
    for name, row in packages.items():
        external_hashes[f"{name}_import_origin"] = row["import_origin_sha256"]
        external_hashes[f"{name}_wheel_record"] = row["record_sha256"]
        external_hashes[f"{name}_distribution_tree"] = row[
            "distribution_tree_sha256"
        ]
        external_paths[f"{name}_import_origin"] = row["import_origin"]
        external_paths[f"{name}_wheel_record"] = row["record_path"]
    environment["external_frozen_artifact_paths"] = external_paths
    return environment, external_hashes


def validate_unopened(workspace: Path, selection: dict[str, Any]) -> dict[str, Any]:
    forbidden = [
        workspace / "blind_protocol_v2/codec_freeze.lock.json",
        workspace / "blind_protocol_v2/codec_freeze.validation.json",
        workspace / "blind_protocol_v2/unblinded",
        workspace / "blind_protocol_v2/materialize_full_tensors_v2.py",
        workspace / "blind_protocol_v2/materialize_full_tensors.py",
        workspace / "blind_protocol_v2/source_hashes.lock.json",
        workspace / "blind_protocol_v2/source_materialization.receipt.json",
    ]
    present = [str(path.relative_to(workspace)) for path in forbidden if path.exists()]
    if present:
        raise FreezeError(f"pre-access freeze refused; forbidden state exists: {present}")
    tensor_names = {str(row["tensor"]) for row in selection["matrices"]}
    expected_payloads: list[Path] = []
    for row in selection["matrices"]:
        relpath = Path(str(row.get("future_output_relpath", "")))
        require(
            str(relpath) not in ("", ".") and not relpath.is_absolute() and ".." not in relpath.parts,
            "proposal future_output_relpath is not a safe relative path",
        )
        candidate = (workspace / "blind_protocol_v2/unblinded" / relpath).resolve()
        try:
            candidate.relative_to((workspace / "blind_protocol_v2/unblinded").resolve())
        except ValueError as exc:
            raise FreezeError(f"proposal payload path escapes unblinded root: {relpath}") from exc
        expected_payloads.append(candidate)
    present_expected = [str(path) for path in expected_payloads if path.exists()]
    require(not present_expected, f"proposal-selected payload paths exist: {present_expected}")
    payloads: list[str] = []
    shards: list[str] = []
    for directory, _, filenames in os.walk(workspace):
        for filename in filenames:
            path = Path(directory) / filename
            if filename.startswith("model-") and filename.endswith("-of-00016.safetensors"):
                shards.append(str(path))
            if any(tensor in filename for tensor in tensor_names) and filename.endswith(
                (".bf16", ".bf16.bin", ".safetensors")
            ):
                payloads.append(str(path))
    if payloads or shards:
        raise FreezeError(
            f"selected payload or full shard already exists: payloads={payloads}, shards={shards}"
        )
    snapshot = load_object(
        checked_file(workspace, "blind_protocol_v2/unopened_snapshot.audit.json")
    )
    require(
        (snapshot.get("schema"), snapshot.get("status"), snapshot.get("passed"))
        == (
            "int2-qwen-second-panel-unopened-snapshot-v1",
            "metadata_only_snapshot_candidates_have_no_workspace_payload_evidence",
            True,
        ),
        "historical unopened-snapshot contract mismatch",
    )
    require(
        int(snapshot.get("selector_network_calls", -1)) == 0
        and int(snapshot.get("selector_tensor_payload_bytes_read", -1)) == 0
        and int(snapshot.get("selector_tensor_payload_files_opened", -1)) == 0
        and snapshot.get("candidate_tensor_path_hits_before_proposal") == []
        and snapshot.get("full_safetensors_shards_present_in_workspace") == [],
        "historical unopened-snapshot evidence failed",
    )
    return {
        "current_state_only_not_global_historical_proof": True,
        "selection_matrix_and_nested_hashes_all_null": True,
        "unblinded_directory_absent": True,
        "materializer_absent": True,
        "source_finalization_outputs_absent": True,
        "all_expected_future_output_paths_absent": True,
        "selected_payload_filename_scan_clear": True,
        "full_checkpoint_shards_absent": True,
        "historical_unopened_snapshot_semantics_validated": True,
    }


def validate_selection(workspace: Path) -> tuple[dict[str, Any], Path]:
    path = checked_file(workspace, "blind_protocol_v2/selection.proposal.lock.json")
    require(sha256_file(path) == SELECTION_FILE_SHA256, "selection file hash drift")
    selection = load_object(path)
    require(
        verify_internal_seal(selection, "selection") == SELECTION_INTERNAL_SHA256,
        "selection internal hash drift",
    )
    require(
        (selection.get("schema"), selection.get("status"))
        == (
            "int2-qwen-blind-selection-proposal-v2",
            "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
        ),
        "selection is not the sealed unopened v2 proposal",
    )
    matrices = selection.get("matrices")
    require(isinstance(matrices, list) and len(matrices) == 18, "selection matrix count")
    require(
        [int(row.get("matrix_ordinal", -1)) for row in matrices] == list(range(18)),
        "selection matrix ordinals",
    )
    require(
        all(
            str(row.get("dtype", "")).upper() == "BF16"
            and int(row.get("nvalues", -1)) == 1_572_864
            and int(row.get("nbytes", -1)) == 3_145_728
            and int(row.get("block_count", -1)) == 6
            and isinstance(row.get("blocks"), list)
            and len(row["blocks"]) == 6
            for row in matrices
        ),
        "selection matrix geometry",
    )
    totals = selection.get("panel_totals", {})
    require(
        (
            int(totals.get("matrix_count", -1)),
            int(totals.get("block_count", -1)),
            int(totals.get("source_values", -1)),
            int(totals.get("source_bytes", -1)),
        )
        == (18, 108, WEIGHTS, 2 * WEIGHTS),
        "selection aggregate geometry",
    )
    require(
        all(row.get("source_bf16_sha256") is None for row in matrices),
        "selection contains source hashes",
    )
    require(
        all(
            block.get("source_bf16_sha256") is None
            for row in matrices
            for block in row.get("blocks", [])
        ),
        "selection contains nested source hashes",
    )
    return selection, path


def validate_development_run(
    workspace: Path,
    run_dir: Path,
    audit_path: Path,
    tamper_path: Path,
    frozen_hashes: dict[str, str],
    runtime_environment: dict[str, Any],
) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    require(run_dir.parent == workspace, "development run must be a direct workspace child")
    require(
        run_dir.name.startswith("strata_v2_dev_final_exact_runtime_"),
        "development run is not in the explicit final-runtime evidence namespace",
    )
    run_relpath = workspace_relative(workspace, run_dir, "development run")
    audit_relpath = workspace_relative(workspace, audit_path, "development audit")
    tamper_relpath = workspace_relative(workspace, tamper_path, "tamper audit")
    try:
        audit_path.resolve(strict=True).relative_to(run_dir)
        tamper_path.resolve(strict=True).relative_to(run_dir)
    except ValueError as exc:
        raise FreezeError("development audit evidence must be contained by the run root") from exc
    require(audit_path.name == "independent_decode_audit.json", "noncanonical audit filename")
    require(
        tamper_path.resolve(strict=True)
        == (run_dir / "independent_lineage_tamper_tests.json").resolve(strict=True),
        "noncanonical tamper-audit path",
    )
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "preencoding_manifest.json"
    allocation_path = run_dir / "allocation.lock.json"
    intent_path = run_dir / "ONE_SHOT_INTENT.json"
    container_path = run_dir / "strata_xklt_sc_v2.bin"
    for path in (summary_path, manifest_path, allocation_path, intent_path, container_path):
        require(path.is_file(), f"development artifact missing: {path}")
    summary, summary_digest = load_object_snapshot(summary_path, "development summary")
    manifest, manifest_digest = load_object_snapshot(manifest_path, "development manifest")
    allocation, allocation_digest = load_object_snapshot(
        allocation_path, "development allocation"
    )
    intent, intent_digest = load_object_snapshot(intent_path, "development intent")
    audit, audit_digest = load_object_snapshot(
        audit_path.resolve(strict=True), "development independent audit"
    )
    tamper, tamper_digest = load_object_snapshot(
        tamper_path.resolve(strict=True), "development tamper audit"
    )
    container_digest = sha256_file(container_path)
    verify_internal_seal(allocation, "development allocation")
    require(
        (summary.get("schema"), summary.get("status"))
        == (
            "strata_xklt_sc_v2_one_shot_summary_v1",
            "one-shot physical artifact complete",
        ),
        "dev summary schema/status",
    )
    require(
        (manifest.get("schema"), manifest.get("status"))
        == (
            "strata_xklt_sc_v2_preencoding_manifest_v1",
            "complete_and_allocation_sealed_before_encoding",
        ),
        "dev manifest schema/status",
    )
    require(
        (allocation.get("schema"), allocation.get("status"))
        == (
            "strata_xklt_sc_v2_allocation_lock_v1",
            "allocation_sealed_before_first_encoder_invocation",
        ),
        "dev allocation schema/status",
    )
    require(
        (intent.get("schema"), intent.get("status"))
        == (
            "strata_xklt_sc_v2_one_shot_intent_v1",
            "sealed_before_first_encoder_invocation",
        ),
        "dev intent schema/status",
    )
    require(summary.get("protocol_mode") == "development", "dev run is not development mode")
    require(manifest.get("protocol_mode") == "development", "dev manifest mode")
    require(allocation.get("protocol_mode") == "development", "dev allocation mode")
    require(intent.get("protocol_mode") == "development", "dev intent mode")
    require(
        allocation.get("manifest_sha256") == manifest_digest,
        "dev allocation/manifest hash binding",
    )
    for key in ("bindings", "assets", "physical_format", "allocation", "blocks"):
        require(
            canonical_bytes(allocation.get(key)) == canonical_bytes(manifest.get(key)),
            f"dev allocation/manifest mismatch: {key}",
        )
    require(
        intent.get("allocation_lock_file_sha256") == allocation_digest
        and intent.get("allocation_lock_internal_sha256") == allocation["lock_sha256"]
        and intent.get("manifest_sha256") == manifest_digest
        and int(intent.get("encoder_invocations_planned", -1)) == 14
        and intent.get("retry_resume_or_adaptive_rate_change_allowed") is False,
        "dev intent lineage",
    )
    require(
        summary.get("allocation_lock_file_sha256") == allocation_digest
        and summary.get("allocation_lock_internal_sha256") == allocation["lock_sha256"]
        and summary.get("intent_sha256") == intent_digest,
        "dev summary lineage",
    )
    require(int(summary.get("encoder_invocations", -1)) == 14, "dev invocation count")
    require(int(summary.get("retries", -1)) == 0, "dev retries are nonzero")
    require(int(summary.get("resumes", -1)) == 0, "dev resumes are nonzero")
    require(int(summary.get("postencoding_profile_changes", -1)) == 0, "dev profile changes")
    encoded_rows = summary.get("encoded_blocks", [])
    require(
        isinstance(encoded_rows, list)
        and len(encoded_rows) == 14
        and [int(row.get("block_ordinal", -1)) for row in encoded_rows] == list(range(14))
        and all(
            int(row.get("encoder_invocations", -1)) == 1
            and isinstance(row.get("checks"), dict)
            and set(row["checks"]) == ENCODER_CHECK_KEYS
            and all(value is True for value in row["checks"].values())
            for row in encoded_rows
        ),
        "development encoded-block receipts",
    )
    physical = summary.get("physical", {})
    require(int(physical.get("physical_bytes", -1)) == PHYSICAL_BYTES, "dev physical bytes")
    require(int(physical.get("physical_bits", -1)) == PHYSICAL_BITS, "dev physical bits")
    require(physical.get("integer_2p15_gate_passed") is True, "dev rate gate failed")
    require(physical.get("reservoir_fit") is True, "dev reservoir overflow")
    logical_bits = sum(int(row["logical_bits"]) for row in encoded_rows)
    payload_bytes = sum((int(row["logical_bits"]) + 7) // 8 for row in encoded_rows)
    require(
        all(
            0 < int(row["logical_bits"]) <= 6 * int(block["values"])
            for row, block in zip(encoded_rows, manifest["blocks"], strict=True)
        )
        and payload_bytes <= 7_603_175
        and int(physical.get("logical_payload_bits", -1)) == logical_bits
        and int(physical.get("payload_byte_count", -1)) == payload_bytes
        and 0 <= int(physical.get("zero_reservoir_tail_bytes", -1))
        and int(physical.get("zero_reservoir_tail_bytes", -1)) == 7_603_175 - payload_bytes
        and int(physical.get("directory_bytes", -1)) == 98
        and logical_bits <= 60_759_864 + 65_536,
        "development payload/reserve accounting",
    )
    require(
        container_path.stat().st_size == PHYSICAL_BYTES
        and container_digest == physical.get("artifact_sha256"),
        "dev container size/hash",
    )
    runtime = intent.get("runtime_freeze", {}).get("artifacts", {})
    expected_runtime = {
        "python_interpreter": frozen_hashes["python_interpreter"],
        "runner": frozen_hashes["one_shot_runner"],
        "polar_encoder": frozen_hashes["polar_encoder"],
        "base_encoder": frozen_hashes["base_cupy_encoder"],
        "procedural_bec_builder": frozen_hashes["procedural_q31_bec"],
        "common": frozen_hashes["common"],
        "emitter": frozen_hashes["emitter"],
        "format": frozen_hashes["format"],
        "independent_auditor": frozen_hashes["independent_auditor"],
    }
    require(set(runtime) == set(expected_runtime), "dev runtime artifact key set")
    require(
        all(runtime[name].get("sha256") == digest for name, digest in expected_runtime.items()),
        "development rehearsal did not execute the final frozen runtime",
    )
    require(
        runtime["python_interpreter"].get("path")
        == runtime_environment["python_interpreter"]["invocation_path"],
        "development rehearsal used a different lexical Python entry path",
    )
    require(
        intent.get("runtime_freeze", {}).get("packages")
        == runtime_environment.get("packages"),
        "development rehearsal package trees differ from the final freeze",
    )
    require(
        intent.get("runtime_freeze", {}).get("cuda")
        == runtime_environment.get("cuda"),
        "development rehearsal CUDA runtime differs from the final freeze",
    )
    require(
        intent.get("runner_sha256") == expected_runtime["runner"]
        and intent.get("encoder_sha256") == expected_runtime["polar_encoder"],
        "dev intent executable hashes",
    )
    manifest_bindings = manifest.get("bindings", {})
    require(
        manifest_bindings.get("common", {}).get("sha256") == expected_runtime["common"]
        and manifest_bindings.get("emitter", {}).get("sha256") == expected_runtime["emitter"]
        and manifest_bindings.get("format_freeze", {}).get("sha256")
        == expected_runtime["format"],
        "dev manifest did not bind the final frozen emitter/common/FORMAT",
    )
    require(
        audit.get("schema") == "strata_v2_klt_mixed_independent_decode_audit_v1"
        and audit.get("audit_execution_passed") is True
        and audit.get("passed") is False,
        "development independent-audit semantics",
    )
    claim = audit.get("primary_claim_gate", {})
    conditions = claim.get("conditions", {})
    require(
        claim.get("passed") is False
        and conditions
        == {
            "physical_rate_at_most_2p15": True,
            "complete_source_lineage_present": True,
            "blind_protocol_mode": False,
            "source_staging_label_scale_and_dp_audit_passed": True,
            "source_domain_mse_below_gaussian_limit": True,
        },
        "development audit primary-condition semantics",
    )
    lineage = audit.get("source_lineage", {})
    require(
        lineage.get("all_checks_passed") is True
        and lineage.get("protocol_mode") == "development"
        and lineage.get("blind_positive_claim_eligible") is False
        and lineage.get("executing_independent_auditor_sha256")
        == expected_runtime["independent_auditor"],
        "development source-lineage audit",
    )
    require(
        lineage.get("preencoding_manifest", {}).get("sha256") == manifest_digest
        and lineage.get("allocation_lock", {}).get("file_sha256") == allocation_digest
        and lineage.get("allocation_lock", {}).get("internal_lock_sha256")
        == allocation["lock_sha256"]
        and lineage.get("one_shot_intent", {}).get("sha256") == intent_digest
        and lineage.get("one_shot_summary", {}).get("sha256") == summary_digest,
        "independent audit does not bind the supplied development lineage files",
    )
    require(
        audit.get("source_staging_and_scale_audit", {}).get("all_checks_passed") is True,
        "development source staging/label/scale/DP audit",
    )
    require(
        lineage.get("independently_measured_runtime_environment")
        == runtime_environment,
        "development auditor runtime differs from the final freeze environment",
    )
    inspection = audit.get("container_inspection", {})
    require(
        inspection.get("container_sha256") == physical.get("artifact_sha256"),
        "development audit/container binding mismatch",
    )
    score = audit.get("source_score")
    require(isinstance(score, dict), "development audit has no source-domain score")
    energy = float(score.get("source_energy_sum_fp64", float("nan")))
    sse = float(score.get("sse_sum_fp64", float("nan")))
    mse = float(score.get("energy_weighted_relative_mse", float("nan")))
    require(
        math.isfinite(energy)
        and energy > 0.0
        and math.isfinite(sse)
        and sse >= 0.0
        and math.isfinite(mse)
        and mse == sse / energy
        and mse < GAUSSIAN_MSE,
        "development pooled FP64 MSE does not beat Gaussian",
    )
    require(score.get("beats_gaussian_limit") is True, "development Gaussian gate flag")
    score_matrices = score.get("matrices", [])
    source_bindings = manifest_bindings.get("sources", [])
    score_root = Path(str(score.get("source_root", ""))).resolve(strict=True)
    require(
        isinstance(score_matrices, list)
        and isinstance(source_bindings, list)
        and len(score_matrices) == len(source_bindings) == 18
        and [int(row.get("matrix_ordinal", -1)) for row in score_matrices]
        == list(range(18)),
        "development score matrix ordinals",
    )
    for row, source in zip(score_matrices, source_bindings, strict=True):
        row_energy = float(row.get("source_energy_fp64", float("nan")))
        row_sse = float(row.get("sse_fp64", float("nan")))
        row_mse = float(row.get("relative_mse", float("nan")))
        expected_source_path = (score_root / str(source["source_relpath"])).resolve(strict=True)
        require(
            math.isfinite(row_energy)
            and row_energy > 0.0
            and math.isfinite(row_sse)
            and row_sse >= 0.0
            and math.isfinite(row_mse)
            and row_mse == row_sse / row_energy
            and row.get("tensor") == source.get("tensor")
            and row.get("source_sha256") == source.get("source_bf16_sha256")
            and Path(str(row.get("source_path", ""))).resolve(strict=True)
            == expected_source_path,
            f"development matrix-score binding {row.get('matrix_ordinal')}",
        )
    require(
        sum(float(row["source_energy_fp64"]) for row in score_matrices) == energy
        and sum(float(row["sse_fp64"]) for row in score_matrices) == sse,
        "development matrix-score pooled sums",
    )
    decode = audit.get("decode", {})
    require(decode.get("all_blocks_decoded") is True, "development decode incomplete")
    blocks = decode.get("blocks", [])
    require(
        len(blocks) == 14
        and [int(row.get("block_ordinal", -1)) for row in blocks] == list(range(14))
        and all(
            int(row.get("block_log2", -1)) == (21 if ordinal < 13 else 20)
            and int(row.get("values", -1)) == (1 << (21 if ordinal < 13 else 20))
            and int(row.get("logical_bits", -1))
            == int(encoded_rows[ordinal]["logical_bits"])
            and int(row.get("payload_terminal_padding_bits", -1))
            == ((-int(row.get("logical_bits", -1))) % 8)
            and 0 <= int(row.get("payload_terminal_padding_bits", -1)) <= 7
            and row.get("canonical_reencode_logical_length_match") is True
            and row.get("canonical_reencode_payload_bytes_match") is True
            for ordinal, row in enumerate(blocks)
        ),
        "development canonical re-encode gate failed",
    )
    tamper_rows = tamper.get("tamper_rows", [])
    tamper_names = [str(row.get("tamper", "")) for row in tamper_rows]
    require(
        tamper.get("schema") == "strata_v2_klt_independent_lineage_tamper_tests_v1"
        and tamper.get("passed") is True
        and tamper.get("protocol_mode") == "development"
        and tamper.get("container_sha256") == sha256_file(container_path)
        and tamper.get("auditor_sha256") == expected_runtime["independent_auditor"]
        and tamper.get("executing_tamper_harness_sha256")
        == frozen_hashes["lineage_tamper_test"]
        and int(tamper.get("tamper_count", -1)) == 10
        and len(tamper_rows) == 10
        and len(set(tamper_names)) == 10
        and set(tamper_names) == TAMPER_NAMES
        and all(
            row.get("rejected") is True and bool(str(row.get("error", "")).strip())
            for row in tamper_rows
        ),
        "development lineage-tamper test evidence",
    )
    return {
        "run_directory": run_relpath,
        "summary_sha256": summary_digest,
        "preencoding_manifest_sha256": manifest_digest,
        "allocation_lock_file_sha256": allocation_digest,
        "allocation_lock_internal_sha256": allocation["lock_sha256"],
        "one_shot_intent_sha256": intent_digest,
        "container_sha256": container_digest,
        "independent_audit_path": audit_relpath,
        "independent_audit_sha256": audit_digest,
        "lineage_tamper_audit_path": tamper_relpath,
        "lineage_tamper_audit_sha256": tamper_digest,
        "physical_bits": PHYSICAL_BITS,
        "physical_bpw": PHYSICAL_BITS / WEIGHTS,
        "source_energy_fp64": energy,
        "sse_fp64": sse,
        "pooled_relative_mse": mse,
        "gaussian_reference": GAUSSIAN_MSE,
        "relative_margin_below_gaussian": 1.0 - mse / GAUSSIAN_MSE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--development-tamper-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    require(
        Path(__file__).resolve(strict=True)
        == (workspace / CANONICAL_BUILDER).resolve(strict=True),
        "noncanonical codec-freeze builder was executed",
    )
    output = args.output.resolve()
    expected_output = (workspace / CANONICAL_OUTPUT).resolve()
    require(output == expected_output, f"freeze output must be {expected_output}")
    require(output.parent.is_dir(), "freeze output parent must already exist")
    require(not output.exists(), f"refusing to overwrite freeze: {output}")
    selection, selection_path = validate_selection(workspace)
    preaccess = validate_unopened(workspace, selection)
    route_path = checked_file(workspace, "blind_protocol_v2/route_table.proposal.bin")
    require(route_path.stat().st_size == 144, "route byte count")
    require(sha256_file(route_path) == ROUTE_FILE_SHA256, "route hash drift")

    artifact_hashes = {
        name: sha256_file(checked_file(workspace, relpath))
        for name, relpath in STATIC_ARTIFACTS.items()
    }
    runtime_environment, external_hashes = build_runtime_environment()
    require(not (set(artifact_hashes) & set(external_hashes)), "frozen hash key collision")
    artifact_hashes.update(external_hashes)
    evidence_hashes = {}
    for name, relpath in DEVELOPMENT_EVIDENCE.items():
        path = checked_file(workspace, relpath)
        evidence_hashes[name] = {"path": relpath, "sha256": sha256_file(path)}
    development = validate_development_run(
        workspace,
        args.development_run_dir,
        args.development_audit,
        args.development_tamper_audit,
        artifact_hashes,
        runtime_environment,
    )

    freeze = seal(
        {
            "schema": "strata_xklt_sc_v2_codec_freeze_v1",
            "status": "frozen_before_blind_source_access",
            "selection_lock_sha256": SELECTION_INTERNAL_SHA256,
            "selection_lock_file_sha256": SELECTION_FILE_SHA256,
            "route_file_sha256": ROUTE_FILE_SHA256,
            "selection_path": str(selection_path.relative_to(workspace)).replace("\\", "/"),
            "checkpoint": selection["checkpoint"],
            "physical_rate_limit_bpw": 2.15,
            "gaussian_mse_reference": GAUSSIAN_MSE,
            "primary_mse_threshold": GAUSSIAN_MSE,
            "success_criterion": "pooled source-domain FP64 SSE/energy < 2^-4.3",
            "allocator_frozen": True,
            "architecture_frozen": True,
            "no_retry_resume_or_postaccess_tuning": True,
            "blind_materializer_authorization_phrase": AUTHORIZATION_PHRASE,
            "expected_source_lock": {
                "schema": "int2-qwen-blind-source-finalization-v2",
                "status": "all_locked_sources_materialized_and_hash_finalized",
                "matrix_count": 18,
                "block_count": 108,
                "source_values": WEIGHTS,
                "source_bytes": 2 * WEIGHTS,
                "dtype": "BF16",
                "exact_codec_freeze_validation_binding_required": True,
                "required_matrix_fields": [
                    "matrix_ordinal",
                    "tensor",
                    "role",
                    "layer",
                    "expert",
                    "dtype",
                    "shape",
                    "nvalues",
                    "nbytes",
                    "block_count",
                    "shard",
                    "http_range_inclusive",
                    "http_response",
                    "output_relpath",
                    "source_bf16_sha256",
                    "blocks",
                ],
                "required_http_response_fields": [
                    "status",
                    "request_url",
                    "requested_range",
                    "content_range",
                    "content_length",
                    "content_encoding",
                    "body_bytes",
                    "body_sha256",
                ],
                "required_block_fields": [
                    "canonical_block_index",
                    "nvalues",
                    "nbytes",
                    "source_bf16_sha256",
                ],
            },
            "physical_ledger": {
                "weights": WEIGHTS,
                "header_bits": 1_024,
                "route_bits": 1_152,
                "label_bits": 41_472,
                "directory_bits": 784,
                "reservoir_bits": 60_825_400,
                "reservoir_bytes": 7_603_175,
                "physical_bits": PHYSICAL_BITS,
                "physical_bytes": PHYSICAL_BYTES,
                "physical_bpw": PHYSICAL_BITS / WEIGHTS,
                "integer_2p15_cap_bits": INTEGER_CAP_BITS,
                "headroom_bits": INTEGER_CAP_BITS - PHYSICAL_BITS,
                "global_no_retry_reserve_bits": 65_536,
                "nominal_profile_budget_bits": 60_759_864,
            },
            "frozen_artifact_paths": STATIC_ARTIFACTS,
            "frozen_artifact_sha256s": artifact_hashes,
            "development_evidence": development,
            "development_evidence_hashes": evidence_hashes,
            "runtime_environment": runtime_environment,
            "preaccess_state": preaccess,
            "claim_boundary": (
                "One precommitted 18-matrix Qwen expert panel; strict PTQ and exact physical "
                "rate. Not a full-checkpoint, perplexity, inference-speed, or universal RD claim."
            ),
        }
    )
    # Re-evaluate every mutable input immediately before the exclusive write.
    require(validate_unopened(workspace, selection) == preaccess, "preaccess state drift")
    require(sha256_file(selection_path) == SELECTION_FILE_SHA256, "selection changed")
    require(sha256_file(route_path) == ROUTE_FILE_SHA256, "route changed")
    final_artifact_hashes = {
        name: sha256_file(checked_file(workspace, relpath))
        for name, relpath in STATIC_ARTIFACTS.items()
    }
    final_environment, final_external_hashes = build_runtime_environment()
    final_artifact_hashes.update(final_external_hashes)
    require(final_artifact_hashes == artifact_hashes, "frozen runtime changed before write")
    require(final_environment == runtime_environment, "runtime environment changed before write")
    for name, row in evidence_hashes.items():
        require(
            sha256_file(checked_file(workspace, row["path"])) == row["sha256"],
            f"development evidence changed before write: {name}",
        )
    run_root = (workspace / development["run_directory"]).resolve(strict=True)
    direct_development = {
        "summary_sha256": run_root / "summary.json",
        "preencoding_manifest_sha256": run_root / "preencoding_manifest.json",
        "allocation_lock_file_sha256": run_root / "allocation.lock.json",
        "one_shot_intent_sha256": run_root / "ONE_SHOT_INTENT.json",
        "container_sha256": run_root / "strata_xklt_sc_v2.bin",
        "independent_audit_sha256": workspace / development["independent_audit_path"],
        "lineage_tamper_audit_sha256": workspace
        / development["lineage_tamper_audit_path"],
    }
    for field, path in direct_development.items():
        resolved = path.resolve(strict=True)
        resolved.relative_to(run_root)
        require(
            sha256_file(resolved) == development[field],
            f"direct development artifact changed before write: {field}",
        )
    require(validate_unopened(workspace, selection) == preaccess, "final preaccess drift")
    require(sha256_file(selection_path) == SELECTION_FILE_SHA256, "final selection drift")
    require(sha256_file(route_path) == ROUTE_FILE_SHA256, "final route drift")
    write_json_create_only(output, freeze)
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(output),
                "file_sha256": sha256_file(output),
                "internal_lock_sha256": freeze["lock_sha256"],
                "development_relative_mse": development["pooled_relative_mse"],
                "physical_bpw": PHYSICAL_BITS / WEIGHTS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
