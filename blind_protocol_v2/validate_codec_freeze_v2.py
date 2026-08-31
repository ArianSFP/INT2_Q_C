#!/usr/bin/env python3
"""Independent fresh-process validator for the pre-access XKLT-v2 freeze."""

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
WEIGHTS = 28_311_552
PHYSICAL_BYTES = 7_608_729
PHYSICAL_BITS = 60_869_832
AUTHORIZATION_PHRASE = "AUTHORIZE SEALED QWEN V2 ONE-SHOT MATERIALIZATION"
CANONICAL_FREEZE = "blind_protocol_v2/codec_freeze.lock.json"
CANONICAL_RECEIPT = "blind_protocol_v2/codec_freeze.validation.json"
CANONICAL_VALIDATOR = "blind_protocol_v2/validate_codec_freeze_v2.py"

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
    "cross_projection_transfer": "strata_v2_cross_projection_transfer_audit_v1.json",
    "triplet_covariance_early_kill": (
        "strata_v2_allocation_research/triplet_covariance_probe_v1.json"
    ),
    "physical_savings_audit": (
        "strata_v2_allocation_research/physical_savings_audit_v1_v5/audit.json"
    ),
}
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_exact(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, payload, hashlib.sha256(payload).hexdigest()


def verify_seal(value: dict[str, Any], label: str) -> str:
    clean = dict(value)
    declared = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    require(declared == actual, f"{label} internal seal mismatch")
    return actual


def seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["lock_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def safe_file(workspace: Path, relpath: str) -> Path:
    candidate = Path(relpath)
    require(not candidate.is_absolute() and ".." not in candidate.parts, f"unsafe path: {relpath}")
    path = (workspace / candidate).resolve(strict=True)
    path.relative_to(workspace)
    require(path.is_file(), f"required file is absent: {relpath}")
    return path


def create_only_json(path: Path, value: Any) -> None:
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


def current_runtime_environment() -> tuple[dict[str, Any], dict[str, str]]:
    import cupy as cp

    invocation = Path(sys.executable).absolute()
    require(invocation.is_file(), "Python invocation path is not a file")
    packages = {
        name: common.distribution_tree_receipt(name, distribution)
        for name, distribution in PACKAGE_DISTRIBUTIONS.items()
    }
    device = cp.cuda.runtime.getDeviceProperties(0)
    environment = {
        "python_interpreter": {
            "invocation_path": str(invocation),
            "resolved_path": str(invocation.resolve(strict=True)),
            "sha256": sha256_file(invocation),
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
    paths = {"python_interpreter": str(invocation)}
    hashes = {"python_interpreter": sha256_file(invocation)}
    for name, row in packages.items():
        paths[f"{name}_import_origin"] = row["import_origin"]
        paths[f"{name}_wheel_record"] = row["record_path"]
        hashes[f"{name}_import_origin"] = row["import_origin_sha256"]
        hashes[f"{name}_wheel_record"] = row["record_sha256"]
        hashes[f"{name}_distribution_tree"] = row["distribution_tree_sha256"]
    environment["external_frozen_artifact_paths"] = paths
    return environment, hashes


def validate_selection(selection: dict[str, Any]) -> None:
    require(
        (selection.get("schema"), selection.get("status"))
        == (
            "int2-qwen-blind-selection-proposal-v2",
            "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
        ),
        "selection schema/status",
    )
    require(verify_seal(selection, "selection") == SELECTION_INTERNAL_SHA256, "selection seal")
    matrices = selection.get("matrices")
    require(isinstance(matrices, list) and len(matrices) == 18, "selection matrix count")
    require([int(row.get("matrix_ordinal", -1)) for row in matrices] == list(range(18)), "ordinals")
    require(
        all(
            str(row.get("dtype", "")).upper() == "BF16"
            and int(row.get("nvalues", -1)) == 1_572_864
            and int(row.get("nbytes", -1)) == 3_145_728
            and int(row.get("block_count", -1)) == 6
            and isinstance(row.get("blocks"), list)
            and len(row["blocks"]) == 6
            and row.get("source_bf16_sha256") is None
            and all(block.get("source_bf16_sha256") is None for block in row["blocks"])
            for row in matrices
        ),
        "selection geometry or null hashes",
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
        "selection totals",
    )


def preaccess_state(workspace: Path, selection: dict[str, Any]) -> dict[str, Any]:
    forbidden = [
        workspace / "blind_protocol_v2/unblinded",
        workspace / "blind_protocol_v2/materialize_full_tensors_v2.py",
        workspace / "blind_protocol_v2/materialize_full_tensors.py",
        workspace / "blind_protocol_v2/source_hashes.lock.json",
        workspace / "blind_protocol_v2/source_materialization.receipt.json",
    ]
    require(not any(path.exists() for path in forbidden), "post-access v2 state exists")
    tensor_names = {str(row["tensor"]) for row in selection["matrices"]}
    unblinded = (workspace / "blind_protocol_v2/unblinded").resolve()
    for row in selection["matrices"]:
        relpath = Path(str(row.get("future_output_relpath", "")))
        require(
            str(relpath) not in ("", ".") and not relpath.is_absolute() and ".." not in relpath.parts,
            "unsafe future_output_relpath",
        )
        path = (unblinded / relpath).resolve()
        path.relative_to(unblinded)
        require(not path.exists(), "proposal-selected payload exists")
    payloads: list[str] = []
    shards: list[str] = []
    for directory, _, filenames in os.walk(workspace):
        for filename in filenames:
            if filename.startswith("model-") and filename.endswith("-of-00016.safetensors"):
                shards.append(str(Path(directory) / filename))
            if any(tensor in filename for tensor in tensor_names) and filename.endswith(
                (".bf16", ".bf16.bin", ".safetensors")
            ):
                payloads.append(str(Path(directory) / filename))
    require(not payloads and not shards, "selected payload filename or full shard exists")
    snapshot, _, _ = load_exact(
        safe_file(workspace, "blind_protocol_v2/unopened_snapshot.audit.json"),
        "unopened snapshot",
    )
    require(
        (snapshot.get("schema"), snapshot.get("status"), snapshot.get("passed"))
        == (
            "int2-qwen-second-panel-unopened-snapshot-v1",
            "metadata_only_snapshot_candidates_have_no_workspace_payload_evidence",
            True,
        )
        and int(snapshot.get("selector_network_calls", -1)) == 0
        and int(snapshot.get("selector_tensor_payload_bytes_read", -1)) == 0
        and int(snapshot.get("selector_tensor_payload_files_opened", -1)) == 0
        and snapshot.get("candidate_tensor_path_hits_before_proposal") == []
        and snapshot.get("full_safetensors_shards_present_in_workspace") == [],
        "historical unopened-snapshot semantics",
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


def validate_development(
    workspace: Path,
    record: dict[str, Any],
    frozen_hashes: dict[str, str],
    environment: dict[str, Any],
) -> dict[Path, str]:
    run_dir = (workspace / str(record.get("run_directory", ""))).resolve(strict=True)
    run_dir.relative_to(workspace)
    require(run_dir.parent == workspace, "development root containment")
    require(run_dir.name.startswith("strata_v2_dev_final_exact_runtime_"), "development namespace")
    paths = {
        "summary_sha256": run_dir / "summary.json",
        "preencoding_manifest_sha256": run_dir / "preencoding_manifest.json",
        "allocation_lock_file_sha256": run_dir / "allocation.lock.json",
        "one_shot_intent_sha256": run_dir / "ONE_SHOT_INTENT.json",
        "container_sha256": run_dir / "strata_xklt_sc_v2.bin",
        "independent_audit_sha256": safe_file(workspace, str(record.get("independent_audit_path", ""))),
        "lineage_tamper_audit_sha256": safe_file(
            workspace, str(record.get("lineage_tamper_audit_path", ""))
        ),
    }
    require(
        paths["independent_audit_sha256"].name == "independent_decode_audit.json",
        "audit filename",
    )
    require(
        paths["lineage_tamper_audit_sha256"].resolve()
        == (run_dir / "independent_lineage_tamper_tests.json").resolve(),
        "tamper filename",
    )
    for field, path in paths.items():
        path.resolve(strict=True).relative_to(run_dir)
        require(path.is_file() and sha256_file(path) == record.get(field), f"development hash: {field}")
    summary, _, summary_hash = load_exact(paths["summary_sha256"], "development summary")
    manifest, _, manifest_hash = load_exact(paths["preencoding_manifest_sha256"], "development manifest")
    allocation, _, allocation_hash = load_exact(paths["allocation_lock_file_sha256"], "development allocation")
    intent, _, intent_hash = load_exact(paths["one_shot_intent_sha256"], "development intent")
    audit, _, audit_hash = load_exact(paths["independent_audit_sha256"], "development audit")
    tamper, _, tamper_hash = load_exact(paths["lineage_tamper_audit_sha256"], "tamper audit")
    require(
        (summary_hash, manifest_hash, allocation_hash, intent_hash, audit_hash, tamper_hash)
        == tuple(
            record[key]
            for key in (
                "summary_sha256",
                "preencoding_manifest_sha256",
                "allocation_lock_file_sha256",
                "one_shot_intent_sha256",
                "independent_audit_sha256",
                "lineage_tamper_audit_sha256",
            )
        ),
        "development snapshot hashes",
    )
    allocation_internal = verify_seal(allocation, "development allocation")
    require(
        (summary.get("schema"), summary.get("status"), summary.get("protocol_mode"))
        == (
            "strata_xklt_sc_v2_one_shot_summary_v1",
            "one-shot physical artifact complete",
            "development",
        ),
        "development summary contract",
    )
    require(
        (manifest.get("schema"), manifest.get("status"), manifest.get("protocol_mode"))
        == (
            "strata_xklt_sc_v2_preencoding_manifest_v1",
            "complete_and_allocation_sealed_before_encoding",
            "development",
        ),
        "development manifest contract",
    )
    require(
        (allocation.get("schema"), allocation.get("status"), allocation.get("protocol_mode"))
        == (
            "strata_xklt_sc_v2_allocation_lock_v1",
            "allocation_sealed_before_first_encoder_invocation",
            "development",
        ),
        "development allocation contract",
    )
    require(
        (intent.get("schema"), intent.get("status"), intent.get("protocol_mode"))
        == (
            "strata_xklt_sc_v2_one_shot_intent_v1",
            "sealed_before_first_encoder_invocation",
            "development",
        ),
        "development intent contract",
    )
    require(allocation.get("manifest_sha256") == manifest_hash, "allocation binding")
    for key in ("bindings", "assets", "physical_format", "allocation", "blocks"):
        require(canonical_bytes(allocation.get(key)) == canonical_bytes(manifest.get(key)), f"manifest/allocation {key}")
    require(
        intent.get("allocation_lock_file_sha256") == allocation_hash
        and intent.get("allocation_lock_internal_sha256") == allocation_internal
        and intent.get("manifest_sha256") == manifest_hash
        and int(intent.get("encoder_invocations_planned", -1)) == 14
        and intent.get("retry_resume_or_adaptive_rate_change_allowed") is False,
        "development intent lineage",
    )
    require(
        summary.get("allocation_lock_file_sha256") == allocation_hash
        and summary.get("allocation_lock_internal_sha256") == allocation_internal
        and summary.get("intent_sha256") == intent_hash
        and int(summary.get("encoder_invocations", -1)) == 14
        and int(summary.get("retries", -1)) == 0
        and int(summary.get("resumes", -1)) == 0
        and int(summary.get("postencoding_profile_changes", -1)) == 0,
        "development summary lineage/no-retry",
    )
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
    require(
        paths["container_sha256"].stat().st_size == PHYSICAL_BYTES
        and int(physical.get("physical_bytes", -1)) == PHYSICAL_BYTES
        and int(physical.get("physical_bits", -1)) == PHYSICAL_BITS
        and physical.get("artifact_sha256") == record["container_sha256"]
        and physical.get("integer_2p15_gate_passed") is True
        and physical.get("reservoir_fit") is True,
        "development physical artifact",
    )
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
    runtime_freeze = intent.get("runtime_freeze", {})
    runtime = runtime_freeze.get("artifacts", {})
    require(set(runtime) == set(expected_runtime), "development runtime keys")
    require(all(runtime[name].get("sha256") == digest for name, digest in expected_runtime.items()), "development runtime hashes")
    require(
        runtime["python_interpreter"].get("path")
        == environment["python_interpreter"]["invocation_path"],
        "development Python invocation path",
    )
    require(runtime_freeze.get("packages") == environment.get("packages"), "development package trees")
    require(runtime_freeze.get("cuda") == environment.get("cuda"), "development CUDA receipt")
    require(
        intent.get("runner_sha256") == expected_runtime["runner"]
        and intent.get("encoder_sha256") == expected_runtime["polar_encoder"],
        "development runner/encoder",
    )
    bindings = manifest.get("bindings", {})
    require(
        bindings.get("common", {}).get("sha256") == expected_runtime["common"]
        and bindings.get("emitter", {}).get("sha256") == expected_runtime["emitter"]
        and bindings.get("format_freeze", {}).get("sha256") == expected_runtime["format"],
        "development manifest runtime bindings",
    )
    require(
        audit.get("schema") == "strata_v2_klt_mixed_independent_decode_audit_v1"
        and audit.get("audit_execution_passed") is True
        and audit.get("passed") is False,
        "development audit semantics",
    )
    claim = audit.get("primary_claim_gate", {})
    require(
        claim.get("passed") is False
        and claim.get("conditions")
        == {
            "physical_rate_at_most_2p15": True,
            "complete_source_lineage_present": True,
            "blind_protocol_mode": False,
            "source_staging_label_scale_and_dp_audit_passed": True,
            "source_domain_mse_below_gaussian_limit": True,
        },
        "development primary-condition semantics",
    )
    lineage = audit.get("source_lineage", {})
    require(
        lineage.get("all_checks_passed") is True
        and lineage.get("protocol_mode") == "development"
        and lineage.get("blind_positive_claim_eligible") is False
        and lineage.get("executing_independent_auditor_sha256") == expected_runtime["independent_auditor"]
        and lineage.get("preencoding_manifest", {}).get("sha256") == manifest_hash
        and lineage.get("allocation_lock", {}).get("file_sha256") == allocation_hash
        and lineage.get("allocation_lock", {}).get("internal_lock_sha256") == allocation_internal
        and lineage.get("one_shot_intent", {}).get("sha256") == intent_hash
        and lineage.get("one_shot_summary", {}).get("sha256") == summary_hash,
        "development source lineage",
    )
    require(audit.get("source_staging_and_scale_audit", {}).get("all_checks_passed") is True, "staging/scale audit")
    require(
        lineage.get("independently_measured_runtime_environment") == environment,
        "development auditor runtime receipt",
    )
    require(
        audit.get("container_inspection", {}).get("container_sha256") == record["container_sha256"],
        "audit/container binding",
    )
    blocks = audit.get("decode", {}).get("blocks", [])
    require(
        audit.get("decode", {}).get("all_blocks_decoded") is True
        and len(blocks) == 14
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
        "canonical independent decode/reencode",
    )
    score = audit.get("source_score", {})
    energy = float(score.get("source_energy_sum_fp64", float("nan")))
    sse = float(score.get("sse_sum_fp64", float("nan")))
    mse = float(score.get("energy_weighted_relative_mse", float("nan")))
    require(
        math.isfinite(energy)
        and energy > 0
        and math.isfinite(sse)
        and sse >= 0
        and math.isfinite(mse)
        and mse == sse / energy
        and mse < math.exp2(-4.3)
        and score.get("beats_gaussian_limit") is True,
        "development pooled source score",
    )
    score_matrices = score.get("matrices", [])
    source_bindings = bindings.get("sources", [])
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
    require(
        record.get("source_energy_fp64") == energy
        and record.get("sse_fp64") == sse
        and record.get("pooled_relative_mse") == mse
        and record.get("gaussian_reference") == math.exp2(-4.3)
        and record.get("relative_margin_below_gaussian") == 1.0 - mse / math.exp2(-4.3)
        and int(record.get("physical_bits", -1)) == PHYSICAL_BITS
        and record.get("physical_bpw") == PHYSICAL_BITS / WEIGHTS,
        "development freeze summary values",
    )
    tamper_rows = tamper.get("tamper_rows", [])
    tamper_names = [str(row.get("tamper", "")) for row in tamper_rows]
    require(
        tamper.get("schema") == "strata_v2_klt_independent_lineage_tamper_tests_v1"
        and tamper.get("passed") is True
        and tamper.get("protocol_mode") == "development"
        and tamper.get("container_sha256") == record["container_sha256"]
        and tamper.get("auditor_sha256") == expected_runtime["independent_auditor"]
        and tamper.get("executing_tamper_harness_sha256") == frozen_hashes["lineage_tamper_test"]
        and int(tamper.get("tamper_count", -1)) == 10
        and len(tamper_rows) == 10
        and len(set(tamper_names)) == 10
        and set(tamper_names) == TAMPER_NAMES
        and all(
            row.get("rejected") is True and bool(str(row.get("error", "")).strip())
            for row in tamper_rows
        ),
        "lineage tamper evidence",
    )
    return {path.resolve(): str(record[field]) for field, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    require(
        Path(__file__).resolve(strict=True) == (workspace / CANONICAL_VALIDATOR).resolve(strict=True),
        "noncanonical freeze validator was executed",
    )
    freeze_path = args.freeze.resolve(strict=True)
    output = args.output.resolve()
    require(freeze_path == (workspace / CANONICAL_FREEZE).resolve(strict=True), "noncanonical freeze path")
    require(output == (workspace / CANONICAL_RECEIPT).resolve(), "noncanonical validation receipt path")
    require(output.parent.is_dir() and not output.exists(), "validation receipt already exists or parent absent")
    freeze, freeze_payload, freeze_file_hash = load_exact(freeze_path, "codec freeze")
    internal = verify_seal(freeze, "codec freeze")
    require(
        set(freeze)
        == {
            "schema",
            "status",
            "selection_lock_sha256",
            "selection_lock_file_sha256",
            "route_file_sha256",
            "selection_path",
            "checkpoint",
            "physical_rate_limit_bpw",
            "gaussian_mse_reference",
            "primary_mse_threshold",
            "success_criterion",
            "allocator_frozen",
            "architecture_frozen",
            "no_retry_resume_or_postaccess_tuning",
            "blind_materializer_authorization_phrase",
            "expected_source_lock",
            "physical_ledger",
            "frozen_artifact_paths",
            "frozen_artifact_sha256s",
            "development_evidence",
            "development_evidence_hashes",
            "runtime_environment",
            "preaccess_state",
            "claim_boundary",
            "lock_sha256",
        },
        "codec-freeze top-level key set",
    )
    require(
        (freeze.get("schema"), freeze.get("status"))
        == ("strata_xklt_sc_v2_codec_freeze_v1", "frozen_before_blind_source_access"),
        "codec-freeze schema/status",
    )
    require(
        freeze.get("selection_lock_sha256") == SELECTION_INTERNAL_SHA256
        and freeze.get("selection_lock_file_sha256") == SELECTION_FILE_SHA256
        and freeze.get("route_file_sha256") == ROUTE_FILE_SHA256,
        "selection/route freeze binding",
    )
    require(
        freeze.get("physical_rate_limit_bpw") == 2.15
        and freeze.get("gaussian_mse_reference") == math.exp2(-4.3)
        and freeze.get("primary_mse_threshold") == math.exp2(-4.3)
        and freeze.get("allocator_frozen") is True
        and freeze.get("architecture_frozen") is True
        and freeze.get("no_retry_resume_or_postaccess_tuning") is True
        and freeze.get("blind_materializer_authorization_phrase") == AUTHORIZATION_PHRASE,
        "primary frozen-control fields",
    )
    require(
        freeze.get("selection_path")
        == "blind_protocol_v2/selection.proposal.lock.json"
        and freeze.get("success_criterion")
        == "pooled source-domain FP64 SSE/energy < 2^-4.3"
        and freeze.get("claim_boundary")
        == (
            "One precommitted 18-matrix Qwen expert panel; strict PTQ and exact physical "
            "rate. Not a full-checkpoint, perplexity, inference-speed, or universal RD claim."
        ),
        "selection path, criterion, or claim boundary",
    )
    require(
        freeze.get("expected_source_lock")
        == {
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
        "expected source-finalization contract",
    )
    expected_ledger = {
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
        "integer_2p15_cap_bits": (43 * WEIGHTS) // 20,
        "headroom_bits": 4,
        "global_no_retry_reserve_bits": 65_536,
        "nominal_profile_budget_bits": 60_759_864,
    }
    ledger = freeze.get("physical_ledger", {})
    require(ledger == expected_ledger, "exact physical ledger")
    require(
        ledger["header_bits"]
        + ledger["route_bits"]
        + ledger["label_bits"]
        + ledger["directory_bits"]
        + ledger["reservoir_bits"]
        == PHYSICAL_BITS
        and ledger["reservoir_bytes"] * 8 == ledger["reservoir_bits"]
        and PHYSICAL_BYTES * 8 == PHYSICAL_BITS
        and ledger["nominal_profile_budget_bits"] + ledger["global_no_retry_reserve_bits"]
        == ledger["reservoir_bits"]
        and PHYSICAL_BITS * 20 <= 43 * WEIGHTS,
        "physical-ledger arithmetic",
    )
    paths = freeze.get("frozen_artifact_paths")
    hashes = freeze.get("frozen_artifact_sha256s")
    require(paths == STATIC_ARTIFACTS and isinstance(hashes, dict), "frozen artifact maps")
    environment, external_hashes = current_runtime_environment()
    require(set(hashes) == set(STATIC_ARTIFACTS) | set(external_hashes), "frozen hash key set")
    tracked: dict[Path, str] = {}
    for name, relpath in STATIC_ARTIFACTS.items():
        path = safe_file(workspace, relpath)
        digest = sha256_file(path)
        require(digest == hashes[name], f"frozen artifact drift: {name}")
        tracked[path] = digest
    require(freeze.get("runtime_environment") == environment, "runtime environment receipt")
    require(all(hashes[name] == digest for name, digest in external_hashes.items()), "external runtime hashes")
    selection_path = safe_file(workspace, STATIC_ARTIFACTS["selection_proposal"])
    selection, _, selection_file_hash = load_exact(selection_path, "selection proposal")
    require(selection_file_hash == SELECTION_FILE_SHA256, "selection file hash")
    validate_selection(selection)
    require(freeze.get("checkpoint") == selection.get("checkpoint"), "checkpoint binding")
    route = safe_file(workspace, STATIC_ARTIFACTS["literal_route"])
    require(route.stat().st_size == 144 and sha256_file(route) == ROUTE_FILE_SHA256, "route bytes")
    state = preaccess_state(workspace, selection)
    require(freeze.get("preaccess_state") == state, "recorded preaccess state")
    evidence = freeze.get("development_evidence_hashes")
    require(isinstance(evidence, dict) and set(evidence) == set(DEVELOPMENT_EVIDENCE), "evidence key set")
    for name, relpath in DEVELOPMENT_EVIDENCE.items():
        row = evidence[name]
        require(row.get("path") == relpath and set(row) == {"path", "sha256"}, f"evidence row {name}")
        path = safe_file(workspace, relpath)
        digest = sha256_file(path)
        require(digest == row["sha256"], f"evidence drift: {name}")
        tracked[path] = digest
    development = freeze.get("development_evidence")
    require(isinstance(development, dict), "development evidence record")
    tracked.update(validate_development(workspace, development, hashes, environment))
    require(freeze_path.read_bytes() == freeze_payload, "freeze bytes changed during validation")
    for path, digest in tracked.items():
        require(sha256_file(path) == digest, f"validated input changed: {path}")
    final_environment, final_external_hashes = current_runtime_environment()
    require(final_environment == environment and final_external_hashes == external_hashes, "runtime changed")
    require(preaccess_state(workspace, selection) == state, "preaccess state changed")
    receipt = seal(
        {
            "schema": "strata_xklt_sc_v2_codec_freeze_validation_v1",
            "status": "validated_before_blind_source_access",
            "passed": True,
            "freeze_path": CANONICAL_FREEZE,
            "freeze_file_sha256": freeze_file_hash,
            "freeze_internal_lock_sha256": internal,
            "executing_validator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "frozen_artifact_count": len(hashes),
            "development_pooled_relative_mse": development["pooled_relative_mse"],
            "gaussian_mse_reference": math.exp2(-4.3),
            "physical_bits": PHYSICAL_BITS,
            "physical_bpw": PHYSICAL_BITS / WEIGHTS,
            "preaccess_state": state,
        }
    )
    create_only_json(output, receipt)
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "receipt_internal_lock_sha256": receipt["lock_sha256"],
                "freeze_file_sha256": freeze_file_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
