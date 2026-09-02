#!/usr/bin/env python3
"""Pure validation contract for externally stored v6 source-free receipts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


SCHEMA = "tactic-actual-coarse-n18-v6-source-free-cupy-smoke-v1"
STATUS = "PASS_SOURCE_FREE_V6_REPAIRS_CUPY_SOURCE_BOUND"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class SmokeContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            f"{label}: SHA-256")
    return value


def _validate_frame_ledger(
    ledger: Any, *, frame_bytes: int, passes: int, mode: str,
) -> None:
    require(isinstance(ledger, dict) and set(ledger) == {
                "schema", "frame_bytes", "external_compressed_read",
                "host_memory_parse_and_integrity", "scratch_lower_bound",
                "accelerator_hbm", "target_eligible_exact_307_over_128",
            }, "smoke complete frame ledger")
    require(ledger["schema"] ==
            "tactic-actual-coarse-n18-v6-frame-ledger-v1" and
            ledger["frame_bytes"] == frame_bytes,
            "smoke frame ledger schema/bytes")
    external = ledger["external_compressed_read"]
    require(isinstance(external, dict) and set(external) == {
                "mode", "passes", "first_pass_bytes", "total_read_bytes",
                "reread_bytes", "total_read_amplification",
                "reread_amplification", "one_external_pass",
                "start_offset_mod_page",
                "unique_page_bytes_if_externally_read",
            } and
            external.get("mode") == mode and
            external.get("passes") == passes and
            external.get("first_pass_bytes") ==
            (frame_bytes if passes else 0) and
            external.get("total_read_bytes") == passes * frame_bytes and
            external.get("reread_bytes") == max(0, passes - 1) * frame_bytes and
            external.get("total_read_amplification") == float(passes) and
            external.get("reread_amplification") == float(max(0, passes - 1)) and
            external.get("one_external_pass") is (passes == 1) and
            external.get("start_offset_mod_page") == 0 and
            external.get("unique_page_bytes_if_externally_read") ==
            (((frame_bytes + 4095) // 4096) * 4096 if passes else 0),
            "smoke external frame read identities")
    host = ledger["host_memory_parse_and_integrity"]
    require(isinstance(host, dict) and
            host.get("causal_packet_decode_input_frame_passes") == 1 and
            host.get("causal_packet_decode_input_frame_bytes") == frame_bytes and
            host.get("aggregate_equality_input_frame_passes") == 1 and
            host.get("aggregate_equality_input_frame_bytes") == frame_bytes and
            host.get("frame_sha256_input_frame_passes") == 1 and
            host.get("frame_sha256_input_frame_bytes") == frame_bytes and
            host.get("minimum_input_frame_full_scan_equivalents") == 3 and
            host.get("minimum_input_frame_bytes_touched") == 3 * frame_bytes and
            host.get("arithmetic_and_array_internal_traffic_measured") is False,
            "smoke host-memory parse ledger")
    require(frame_bytes % 78_592 == 0, "smoke whole reservoir frame")
    records = frame_bytes // 78_592
    expected_i32 = records * (1 << 18) * 4
    expected_reconstruction = 3 * (1 << 18) * 4
    scratch = ledger["scratch_lower_bound"]
    require(isinstance(scratch, dict) and
            scratch.get("canonical_packet_buffers_bytes") == frame_bytes and
            scratch.get("aggregate_reencode_buffer_bytes") == frame_bytes and
            scratch.get("canonical_symbols_i32_bytes") == expected_i32 and
            scratch.get("reconstruction_f32_bytes") == expected_reconstruction and
            scratch.get("minimum_numeric_and_packet_scratch_bytes") ==
            2 * frame_bytes + scratch["canonical_symbols_i32_bytes"] +
            scratch["reconstruction_f32_bytes"] and
            scratch.get("python_object_and_decoder_internal_scratch_measured")
            is False,
            "smoke scratch lower-bound ledger")
    require(ledger["accelerator_hbm"] == {
                "measured": False, "read_bytes": None,
                "read_amplification": None,
                "below_2x_claim_authority": False,
            }, "smoke accelerator-HBM nonclaim ledger")
    require(ledger["target_eligible_exact_307_over_128"] is True,
            "smoke target-eligible ledger")


def validate_smoke_receipt(
    record: Any,
    *,
    source_manifest_sha256: str,
    source_root_sha256: str,
    predecessor_lock_sha256: str,
    runtime_lock_sha256: str,
    source_member_hashes: Mapping[str, str],
) -> dict[str, Any]:
    require(isinstance(record, dict), "smoke receipt object")
    required = {
        "schema", "status", "source_closure", "runtime_closure",
        "numeric_tile", "i32_stress_lifetime", "aggregate_zero_frame",
        "traffic_ledgers", "payload_accessed",
        "model_or_qwen_path_discovered_or_enumerated", "claim_boundary",
        "receipt_sha256",
    }
    require(set(record) == required, "smoke receipt exact fields")
    require(record["schema"] == SCHEMA and record["status"] == STATUS,
            "smoke receipt schema/status")
    clean = dict(record)
    claimed = _digest(clean.pop("receipt_sha256"), "smoke receipt seal")
    require(sha256(canonical_json(clean)) == claimed,
            "smoke receipt internal seal")
    source = record["source_closure"]
    require(isinstance(source, dict) and set(source) == {
                "source_manifest_sha256", "source_root_sha256",
                "member_hashes", "retained_no_follow_descriptors",
                "executing_entry_inode_bound", "executing_entry_name",
            } and
            source.get("source_manifest_sha256") == source_manifest_sha256 and
            source.get("source_root_sha256") == source_root_sha256 and
            source.get("member_hashes") == dict(source_member_hashes) and
            source.get("retained_no_follow_descriptors") is True and
            source.get("executing_entry_inode_bound") is True and
            source.get("executing_entry_name") ==
            "synthetic_cupy_smoke.py",
            "smoke/source retained closure")
    runtime = record["runtime_closure"]
    require(isinstance(runtime, dict) and
            runtime.get("predecessor_lock_sha256") ==
            predecessor_lock_sha256 and
            runtime.get("runtime_lock_sha256") == runtime_lock_sha256 and
            runtime.get("inverse_transient_dtype") == "<i4" and
            runtime.get("inverse_override_installed_before_any_reservoir_decode")
            is True,
            "smoke runtime closure")
    tile = record["numeric_tile"]
    require(isinstance(tile, dict) and tile.get("packet_bytes") == 78_592 and
            tile.get("all_encoder_self_checks_required_and_passed") is True and
            tile.get("canonical_reencode_matches") is True and
            tile.get("inverse_i32_dtype_verified_before_facade_cast") is True and
            tile.get("inverse_transient_dtype") == "<i4" and
            type(tile.get("relative_mse_original_coordinates")) is float and
            math.isfinite(tile["relative_mse_original_coordinates"]) and
            tile["relative_mse_original_coordinates"] >= 0.0,
            "smoke numeric tile")
    stress = record["i32_stress_lifetime"]
    require(isinstance(stress, dict) and
            stress.get("input_index") == 63 and
            stress.get("expected_abs_max") == 8_388_608 and
            stress.get("observed_abs_max") == 8_388_608 and
            stress.get("observed_abs_max") > 32_767 and
            stress.get("installed_in_inherited_decoder_before_call") is True and
            stress.get("inverse_output_dtype_before_facade") == "<i4" and
            stress.get("facade_retained_dtype") == "<i4" and
            stress.get("no_copy_or_downcast") is True and
            stress.get("downstream_reconstruction_float64_abs_max") ==
            4_096.0 and
            stress.get("downstream_reconstruction_expected_abs_max") ==
            4_096.0,
            "smoke >I16 I32 lifetime stress")
    aggregate = record["aggregate_zero_frame"]
    require(isinstance(aggregate, dict) and
            aggregate.get("roles") == ["gate", "up", "down_transposed"] and
            aggregate.get("literal_aggregate_reencode_matches") is True and
            aggregate.get("exact_inherited_role_abi") is True,
            "smoke aggregate role/reencode")
    traffic = record["traffic_ledgers"]
    require(isinstance(traffic, dict) and set(traffic) == {
                "prebuffered_decode", "modeled_one_external_pass",
                "modeled_two_external_passes",
            }, "smoke traffic ledgers")
    prebuffered = traffic.get("prebuffered_decode")
    one = traffic.get("modeled_one_external_pass")
    two = traffic.get("modeled_two_external_passes")
    frame_bytes = aggregate.get("frame_bytes")
    require(type(frame_bytes) is int and frame_bytes == 3 * 78_592,
            "smoke aggregate frame bytes")
    _validate_frame_ledger(
        prebuffered, frame_bytes=frame_bytes, passes=0,
        mode="prebuffered_encoder_output")
    _validate_frame_ledger(
        one, frame_bytes=frame_bytes, passes=1,
        mode="one_pass_external_file")
    _validate_frame_ledger(
        two, frame_bytes=frame_bytes, passes=2,
        mode="modeled_external_file_reread")
    require(record["payload_accessed"] is False and
            record["model_or_qwen_path_discovered_or_enumerated"] is False and
            record["claim_boundary"] ==
            "source-free mechanics/runtime only; authorizes a separately bound Qwen pilot but is not a Qwen, MSE, universal-tail, fine-code, or inference-HBM result",
            "smoke claim boundary")
    return {
        "receipt_sha256": claimed,
        "source_manifest_sha256": source_manifest_sha256,
        "source_root_sha256": source_root_sha256,
        "i32_stress_above_i16": True,
        "qwen_payload_accessed": False,
        "positive_claim_authority": False,
    }
