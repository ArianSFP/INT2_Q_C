#!/usr/bin/env python3
"""Authenticated-snapshot producer core for universal UWFA-SC v6.

This file is intentionally *not* a directly executable payload launcher.  A
separate independently reviewed dispatcher must pin the producer manifest and
review digest, read every producer member once through retained descriptors,
and compile/exec this exact entrypoint snapshot.  A public token or self-sealed
JSON cannot call the producer.

The functions below contain the complete source/control scientific pipeline.
They accept already-authenticated byte snapshots and injected modules; no
repository-relative import or dynamic input pathname is used.
"""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PRODUCER_ABI = "uwfa-sc-v6-authenticated-snapshot-producer-abi-1"
DIRECT_LAUNCH_STATUS = "BLOCK_DIRECT_EXECUTION_REQUIRES_EXTERNALLY_PINNED_DISPATCHER"
MAX_PROJECTED_WALL_SECONDS = 21_600.0
MAX_HOST_BYTES = 96 * (1 << 30)
MAX_VRAM_BYTES = 28 * (1 << 30)
HOST_ALLOCATION_RESERVE_BYTES = 1 << 30
VRAM_ALLOCATION_RESERVE_BYTES = 2 * (1 << 30)
MAX_BACKEND_STREAMS = 65_536
MAX_AUXILIARY_DEVICE_BYTES = 64 * 384 * 2 * 8 + 64 * 384 * 2 + 40 * MAX_BACKEND_STREAMS
MAX_PACKED_SYMBOLS = (MAX_VRAM_BYTES - VRAM_ALLOCATION_RESERVE_BYTES - MAX_AUXILIARY_DEVICE_BYTES) // 4
MIN_CALIBRATED_CELL_SYMBOLS_PER_SECOND = 1_000_000.0
T95_TWO_SIDED = (
    math.inf, 12.706204736432095, 4.302652729696142, 3.182446305284263,
    2.7764451051977987, 2.570581835636305, 2.446911848791681,
    2.3646242510102993, 2.3060041350333704, 2.2621571627409915,
    2.2281388519649385, 2.200985160082949, 2.178812829663418,
    2.1603686564610127, 2.1447866879169273, 2.131449545559323,
    2.1199052992210112, 2.109815577833181, 2.10092204024096,
    2.093024054408263, 2.0859634472658364, 2.079613844727662,
    2.0738730679040147, 2.0686576104190406, 2.063898561628021,
    2.0595385527532946, 2.055529438642871, 2.0518305164802833,
    2.048407141795244, 2.045229642132703, 2.0422724563012373,
)


@dataclass(frozen=True)
class BoundEvidence:
    baseline_plan_sha256: str
    baseline_score_sha256: str
    universal_decoder_sha256: str
    producer_manifest_sha256: str
    audit_bootstrap_sha256: str
    source_full_geometry_sha256: str
    source_structural_geometry_sha256: str
    extraction_program_sha256: str
    universal_adapter_sha256: str
    pipeline_sha256: str
    source_snapshot_root_sha256: str
    source_preflight_receipt_sha256: str

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be SHA-256 hex")
            try:
                raw = bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} not hexadecimal") from exc
            if len(raw) != 32:
                raise ValueError(f"{name} digest geometry")

    def container_hashes(self) -> dict[str, str]:
        return {
            "baseline_plan_sha256": self.baseline_plan_sha256,
            "baseline_score_sha256": self.baseline_score_sha256,
            "universal_decoder_sha256": self.universal_decoder_sha256,
            "producer_manifest_sha256": self.producer_manifest_sha256,
            "audit_bootstrap_sha256": self.audit_bootstrap_sha256,
            "source_full_geometry_sha256": self.source_full_geometry_sha256,
            "source_structural_geometry_sha256": self.source_structural_geometry_sha256,
            "extraction_program_sha256": self.extraction_program_sha256,
            "universal_adapter_sha256": self.universal_adapter_sha256,
            "pipeline_sha256": self.pipeline_sha256,
            "source_snapshot_root_sha256": self.source_snapshot_root_sha256,
            "source_preflight_receipt_sha256": self.source_preflight_receipt_sha256,
        }

    def symmetric_control_closure(self) -> dict[str, str]:
        """Fields that must be byte-identical for source and every null."""
        return {
            "baseline_plan_sha256": self.baseline_plan_sha256,
            "universal_decoder_sha256": self.universal_decoder_sha256,
            "producer_manifest_sha256": self.producer_manifest_sha256,
            "audit_bootstrap_sha256": self.audit_bootstrap_sha256,
            "extraction_program_sha256": self.extraction_program_sha256,
            "universal_adapter_sha256": self.universal_adapter_sha256,
            "pipeline_sha256": self.pipeline_sha256,
            "source_snapshot_root_sha256": self.source_snapshot_root_sha256,
            "source_preflight_receipt_sha256": self.source_preflight_receipt_sha256,
        }


@dataclass(frozen=True)
class SourcePreflightEvidence:
    """Typed source-free all150/representative/device evidence bundle."""

    all150: Mapping[str, Any]
    representative: Mapping[str, Any]
    independent_gpu_identity: Mapping[str, Any]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, dict) for value in (self.all150, self.representative, self.independent_gpu_identity)):
            raise ValueError("preflight nested receipts must be exact dictionaries")
        if not isinstance(self.receipt_sha256, str) or len(self.receipt_sha256) != 64:
            raise ValueError("preflight receipt digest geometry")
        try:
            raw = bytes.fromhex(self.receipt_sha256)
        except ValueError as exc:
            raise ValueError("preflight receipt digest encoding") from exc
        if len(raw) != 32:
            raise ValueError("preflight receipt digest width")


def _validate_environment_identity(protocol: Any, environment: Any, independent: Mapping[str, Any]) -> dict[str, Any]:
    row = protocol.strict_fields(
        environment,
        required=(
            "cupy_version", "cuda_runtime_version", "cuda_driver_version",
            "python_version", "platform", "device_id", "device_name",
            "device_uuid", "pci_bus_id", "compute_capability",
            "current_free_vram_bytes", "total_vram_bytes", "statistics",
            "telemetry_samples", "host_byteorder",
            "explicit_device_synchronization_at_phase_boundaries_and_after_every_kernel",
            "fatal_telemetry_sampling", "transfer_formula",
        ),
        label="CUDA environment receipt",
    )
    if row["fatal_telemetry_sampling"] is not True or row["explicit_device_synchronization_at_phase_boundaries_and_after_every_kernel"] is not True:
        raise ValueError("CUDA telemetry fail-closed flags")
    device_uuid = protocol.identifier(row["device_uuid"], "CUDA device UUID")
    pci_bus_id = protocol.identifier(row["pci_bus_id"], "CUDA PCI bus id")
    device_name = str(row["device_name"])
    if not device_name or len(device_name) > 256 or any(ord(character) < 32 or ord(character) > 126 for character in device_name):
        raise ValueError("CUDA device name")
    independent_row = protocol.strict_fields(
        independent,
        required=("schema", "status", "device_uuid", "pci_bus_id", "device_name", "provider", "identity_receipt_sha256"),
        label="independent GPU identity",
    )
    if independent_row["schema"] != "uwfa-sc-v6-independent-gpu-identity" or independent_row["status"] != "PASS_INDEPENDENT_GPU_IDENTITY":
        raise ValueError("independent GPU identity schema/status")
    clean = dict(independent_row)
    claimed = protocol.sha256_hex(clean.pop("identity_receipt_sha256"), "independent GPU identity seal")
    import json
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != claimed:
        raise ValueError("independent GPU identity seal")
    if (
        protocol.identifier(independent_row["device_uuid"], "independent GPU UUID") != device_uuid
        or protocol.identifier(independent_row["pci_bus_id"], "independent GPU PCI") != pci_bus_id
        or str(independent_row["device_name"]) != device_name
    ):
        raise ValueError("CUDA/NVML UUID, PCI, or device-name mismatch")
    protocol.identifier(independent_row["provider"], "independent GPU provider")
    statistics_row = _validate_telemetry_statistics(protocol, row["statistics"], row["telemetry_samples"])
    if statistics_row["total_vram_bytes"] != protocol.exact_int(row["total_vram_bytes"], "CUDA total VRAM", 1, (1 << 63) - 1):
        raise ValueError("CUDA environment/statistics VRAM mismatch")
    protocol.exact_int(row["current_free_vram_bytes"], "CUDA current free VRAM", 1, statistics_row["total_vram_bytes"])
    if row["host_byteorder"] != "little":
        raise ValueError("CUDA environment host byte order")
    capability = row["compute_capability"]
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("CUDA compute capability")
    protocol.exact_int(capability[0], "CUDA compute major", 1, 99)
    protocol.exact_int(capability[1], "CUDA compute minor", 0, 99)
    protocol.exact_int(row["device_id"], "CUDA device id", 0, 255)
    for name in ("cuda_runtime_version", "cuda_driver_version"):
        protocol.exact_int(row[name], name, 1, (1 << 31) - 1)
    transfer = protocol.strict_fields(
        row["transfer_formula"],
        required=("root_pack_h2d", "subset_h2d", "launch_descriptor_h2d", "model_h2d", "kernel_scalars_h2d", "d2h"),
        label="CUDA transfer formula",
    )
    if not all(isinstance(value, str) and value for value in transfer.values()):
        raise ValueError("CUDA transfer formula values")
    return dict(row)


def _validate_telemetry_statistics(protocol: Any, statistics: Any, samples: Any) -> dict[str, Any]:
    required_integers = (
        "h2d_bytes", "h2d_payload_bytes", "h2d_root_descriptor_bytes",
        "h2d_subset_descriptor_bytes", "h2d_launch_descriptor_bytes",
        "h2d_model_table_bytes", "h2d_kernel_scalar_bytes", "d2h_bytes",
        "d2d_descriptor_bytes", "device_output_allocation_bytes", "kernel_count",
        "count_kernel_count", "length_kernel_count", "count_cell_symbol_updates",
        "length_cell_symbol_updates", "pack_calls", "subset_calls", "to_host_calls",
        "telemetry_samples", "peak_process_tree_rss_bytes", "peak_process_hwm_bytes",
        "incremental_peak_process_tree_rss_bytes", "peak_vram_incremental_bytes",
        "peak_default_pool_used_bytes", "peak_default_pool_total_bytes",
        "peak_pinned_pool_free_blocks", "baseline_free_vram_bytes", "total_vram_bytes",
        "resource_preflight_calls",
    )
    required = required_integers + ("jit_compile_seconds", "kernel_wall_seconds", "last_pack_resource_plan")
    row = protocol.strict_fields(statistics, required=required, label="CUDA statistics")
    clean: dict[str, Any] = {}
    for name in required_integers:
        clean[name] = protocol.exact_int(row[name], f"CUDA statistics {name}", 0, (1 << 63) - 1)
    for name in ("jit_compile_seconds", "kernel_wall_seconds"):
        clean[name] = protocol.finite_float(row[name], f"CUDA statistics {name}")
        if clean[name] < 0.0:
            raise ValueError(f"CUDA statistics negative {name}")
    if clean["telemetry_samples"] <= 0 or clean["peak_process_tree_rss_bytes"] <= 0 or clean["peak_process_hwm_bytes"] <= 0 or clean["total_vram_bytes"] <= 0:
        raise ValueError("CUDA fatal telemetry absent")
    if clean["h2d_bytes"] != sum(clean[name] for name in (
        "h2d_payload_bytes", "h2d_root_descriptor_bytes", "h2d_subset_descriptor_bytes",
        "h2d_launch_descriptor_bytes", "h2d_model_table_bytes", "h2d_kernel_scalar_bytes",
    )):
        raise ValueError("CUDA H2D category conservation")
    if clean["kernel_count"] != clean["count_kernel_count"] + clean["length_kernel_count"]:
        raise ValueError("CUDA kernel category conservation")
    if not isinstance(samples, list) or len(samples) != clean["telemetry_samples"]:
        raise ValueError("CUDA telemetry sample count")
    for sample in samples:
        item = protocol.strict_fields(
            sample,
            required=("phase", "process_tree_rss_bytes", "process_hwm_bytes", "free_vram_bytes", "total_vram_bytes", "default_pool_used_bytes", "default_pool_total_bytes", "pinned_pool_free_blocks"),
            label="CUDA telemetry sample",
        )
        if not isinstance(item["phase"], str) or not item["phase"]:
            raise ValueError("CUDA telemetry phase")
        for name in set(item) - {"phase"}:
            minimum = 1 if name in {"process_tree_rss_bytes", "process_hwm_bytes", "free_vram_bytes", "total_vram_bytes"} else 0
            protocol.exact_int(item[name], f"CUDA telemetry sample {name}", minimum, (1 << 63) - 1)
    clean["last_pack_resource_plan"] = row["last_pack_resource_plan"]
    return clean


def _validate_resource_plan(protocol: Any, plan: Any, *, symbols: int, streams: int) -> dict[str, Any]:
    row = protocol.strict_fields(
        plan,
        required=(
            "symbols", "streams", "payload_host_and_device_bytes",
            "root_descriptor_device_bytes", "additional_host_bytes_including_reserve",
            "device_required_bytes_including_aux_and_reserve",
            "current_process_tree_rss_bytes", "current_process_hwm_bytes",
            "projected_process_tree_rss_bytes", "current_free_vram_bytes",
            "current_total_vram_bytes", "host_cap_bytes", "vram_cap_bytes",
            "passes", "checked_before_blob_concatenation_or_cupy_allocation",
        ),
        label="GPU resource plan",
    )
    expected_symbols = protocol.exact_int(symbols, "resource expected symbols", 1, MAX_PACKED_SYMBOLS)
    expected_streams = protocol.exact_int(streams, "resource expected streams", 1, MAX_BACKEND_STREAMS)
    if row["symbols"] != expected_symbols or row["streams"] != expected_streams:
        raise ValueError("GPU resource plan geometry")
    payload = 4 * expected_symbols
    descriptors = 16 * expected_streams
    additional_host = payload + 64 * expected_streams + HOST_ALLOCATION_RESERVE_BYTES
    required_device = payload + descriptors + MAX_AUXILIARY_DEVICE_BYTES + VRAM_ALLOCATION_RESERVE_BYTES
    if (
        row["payload_host_and_device_bytes"] != payload
        or row["root_descriptor_device_bytes"] != descriptors
        or row["additional_host_bytes_including_reserve"] != additional_host
        or row["device_required_bytes_including_aux_and_reserve"] != required_device
        or row["host_cap_bytes"] != MAX_HOST_BYTES
        or row["vram_cap_bytes"] != MAX_VRAM_BYTES
    ):
        raise ValueError("GPU resource plan arithmetic")
    current_rss = protocol.exact_int(row["current_process_tree_rss_bytes"], "resource current RSS", 1, MAX_HOST_BYTES)
    protocol.exact_int(row["current_process_hwm_bytes"], "resource current HWM", 1, (1 << 63) - 1)
    free = protocol.exact_int(row["current_free_vram_bytes"], "resource free VRAM", 1, (1 << 63) - 1)
    total = protocol.exact_int(row["current_total_vram_bytes"], "resource total VRAM", free, (1 << 63) - 1)
    if row["projected_process_tree_rss_bytes"] != current_rss + additional_host:
        raise ValueError("GPU resource projected RSS")
    expected_pass = current_rss + additional_host <= MAX_HOST_BYTES and required_device <= MAX_VRAM_BYTES and required_device <= free
    if row["passes"] is not expected_pass or row["passes"] is not True or row["checked_before_blob_concatenation_or_cupy_allocation"] is not True:
        raise ValueError("GPU resource admission flags")
    return dict(row)


def _u64_list_sha256(protocol: Any, values: Any, *, count: int, label: str) -> tuple[list[int], str]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{label} geometry")
    clean = [protocol.exact_int(value, label, 1, (1 << 63) - 1) for value in values]
    return clean, hashlib.sha256(b"".join(value.to_bytes(8, "little") for value in clean)).hexdigest()


def _expected_all150_fixture(common: Any) -> tuple[list[int], int, str]:
    lengths = [4097, 2053, 1031, 521]
    digest = hashlib.sha256()
    for stream_ordinal, length in enumerate(lengths):
        bits = bytes((((index * (17 + stream_ordinal)) ^ (index >> (1 + stream_ordinal)) ^ (index // 31)) & 1) for index in range(length))
        levels = bytes((index + 3 * stream_ordinal) % common.LEVELS for index in range(length))
        base = []
        for index in range(length):
            if index % 257 == 0:
                base.append(1)
            elif index % 257 == 1:
                base.append(65535)
            else:
                bucket = (index // 4 + stream_ordinal) % common.PRIOR_BINS
                base.append(min(65535, max(1, bucket * 4096 + 2048)))
        digest.update(bits)
        digest.update(levels)
        digest.update(struct.pack(f"<{length}H", *base))
    return lengths, sum(lengths), digest.hexdigest()


def _validate_all150_receipt(common: Any, protocol: Any, value: Any, bindings: BoundEvidence, independent: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    row = protocol.strict_fields(
        value,
        required=(
            "schema", "source_snapshot_root_sha256", "status", "cells", "cell_count",
            "streams", "symbols_per_complete_bank", "fixture_stream_lengths",
            "fixture_sha256", "resource_plan", "elapsed_seconds", "environment",
            "frequency_extremes_tested", "reset_boundaries_through_4096_tested",
            "candidate_selector_sha256", "cell_results_sha256",
        ),
        label="all150 preflight",
    )
    if row["schema"] != "uwfa-sc-v6-all150-source-free-preflight" or row["status"] != "PASS_ALL_150_CPU_CUPY_EXACT_REPEATED":
        raise ValueError("all150 preflight schema/status")
    if protocol.sha256_hex(row["source_snapshot_root_sha256"], "all150 source root") != bindings.source_snapshot_root_sha256:
        raise ValueError("all150 source snapshot root")
    fixture_lengths, fixture_symbols, fixture_sha = _expected_all150_fixture(common)
    if row["streams"] != 4 or row["symbols_per_complete_bank"] != fixture_symbols or row["fixture_stream_lengths"] != fixture_lengths or protocol.sha256_hex(row["fixture_sha256"], "all150 fixture") != fixture_sha:
        raise ValueError("all150 deterministic fixture binding")
    _validate_resource_plan(protocol, row["resource_plan"], symbols=fixture_symbols, streams=4)
    protocol.finite_float(row["elapsed_seconds"], "all150 elapsed", positive=True)
    if row["frequency_extremes_tested"] != [1, 65535] or row["reset_boundaries_through_4096_tested"] is not True:
        raise ValueError("all150 boundary coverage")
    cells = row["cells"]
    if row["cell_count"] != 150 or not isinstance(cells, list) or len(cells) != 150:
        raise ValueError("all150 exact cell count")
    expected_cells = common.candidate_bank()
    selectors = []
    cell_fields = (
        "topology", "topology_id", "states", "reset_length", "selector_ordinal",
        "count_tensor_sha256", "count_entries", "count_total", "fitted_q016_sha256",
        "logical_lengths", "logical_length_count", "logical_lengths_sha256",
        "frequency_extreme_logical_lengths", "frequency_extreme_logical_lengths_sha256",
        "cpu_gpu_first_count_exact", "cpu_gpu_second_count_exact", "fitted_q016_cpu_gpu_exact",
        "cpu_gpu_first_logical_lengths_exact", "cpu_gpu_second_logical_lengths_exact",
        "frequency_extreme_cpu_gpu_exact", "repeated_gpu_run_exact", "cell_result_sha256",
    )
    for ordinal, (cell, candidate) in enumerate(zip(cells, expected_cells, strict=True)):
        item = protocol.strict_fields(cell, required=cell_fields, label="all150 cell")
        if {name: item[name] for name in candidate.as_dict()} != candidate.as_dict() or item["selector_ordinal"] != ordinal:
            raise ValueError("all150 canonical candidate ordering")
        selectors.append(item["selector_ordinal"])
        if item["count_entries"] != candidate.states * common.CONTEXTS * 2 or item["count_total"] != fixture_symbols:
            raise ValueError("all150 count tensor metadata")
        for name in ("count_tensor_sha256", "fitted_q016_sha256"):
            protocol.sha256_hex(item[name], f"all150 {name}")
        logical, logical_sha = _u64_list_sha256(protocol, item["logical_lengths"], count=4, label="all150 logical length")
        extreme, extreme_sha = _u64_list_sha256(protocol, item["frequency_extreme_logical_lengths"], count=4, label="all150 extreme logical length")
        if item["logical_length_count"] != 4 or item["logical_lengths_sha256"] != logical_sha or item["frequency_extreme_logical_lengths_sha256"] != extreme_sha:
            raise ValueError("all150 logical-length commitments")
        for name in (
            "cpu_gpu_first_count_exact", "cpu_gpu_second_count_exact", "fitted_q016_cpu_gpu_exact",
            "cpu_gpu_first_logical_lengths_exact", "cpu_gpu_second_logical_lengths_exact",
            "frequency_extreme_cpu_gpu_exact", "repeated_gpu_run_exact",
        ):
            if item[name] is not True:
                raise ValueError(f"all150 equality flag: {name}")
        clean = dict(item)
        claimed = protocol.sha256_hex(clean.pop("cell_result_sha256"), "all150 cell result")
        if hashlib.sha256(common.canonical_json(clean)).hexdigest() != claimed:
            raise ValueError("all150 cell result seal")
    if selectors != list(range(150)) or row["candidate_selector_sha256"] != hashlib.sha256(common.canonical_json(selectors)).hexdigest():
        raise ValueError("all150 unique canonical selector coverage")
    if protocol.sha256_hex(row["cell_results_sha256"], "all150 results") != hashlib.sha256(common.canonical_json(cells)).hexdigest():
        raise ValueError("all150 result-list seal")
    environment = _validate_environment_identity(protocol, row["environment"], independent)
    return dict(row), environment


def _expected_representative_fixture(common: Any) -> tuple[list[int], int, str]:
    lengths = [131_071 + 257 * index for index in range(12)] + [65_537 + 509 * index for index in range(3)]
    digest = hashlib.sha256()
    for ordinal, length in enumerate(lengths):
        bits = bytes((((position * (17 + ordinal)) ^ (position >> 3) ^ (position // 31) ^ ordinal) & 1) for position in range(length))
        levels = bytes((position + ordinal) % common.LEVELS for position in range(length))
        base = [
            1 if position % 257 == 0 else 65535 if position % 257 == 1 else 2048 + 4096 * ((position // 4 + ordinal) % common.PRIOR_BINS)
            for position in range(length)
        ]
        digest.update(bits)
        digest.update(levels)
        digest.update(struct.pack(f"<{length}H", *base))
    return lengths, sum(lengths), digest.hexdigest()


def _validate_representative_receipt(common: Any, protocol: Any, value: Any, bindings: BoundEvidence, independent: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    row = protocol.strict_fields(
        value,
        required=(
            "schema", "source_snapshot_root_sha256", "status", "fixture", "outer_fold",
            "runtime_projection", "telemetry", "measured_phase_statistics_delta",
            "model_h2d_bytes_nonzero", "d2h_bytes_nonzero", "peak_host_ram_recorded", "peak_vram_recorded",
        ),
        label="representative preflight",
    )
    if row["schema"] != "uwfa-sc-v6-representative-source-free-preflight" or row["status"] != "PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD":
        raise ValueError("representative preflight schema/status")
    if protocol.sha256_hex(row["source_snapshot_root_sha256"], "representative source root") != bindings.source_snapshot_root_sha256:
        raise ValueError("representative source snapshot root")
    fixture = protocol.strict_fields(
        row["fixture"],
        required=("streams", "semantic_owners", "private_streams", "shared_tail_streams", "stream_lengths", "symbols", "source", "fixture_sha256", "resource_plan"),
        label="representative fixture",
    )
    expected_lengths, full_symbols, expected_fixture_sha = _expected_representative_fixture(common)
    if (
        (fixture["streams"], fixture["semantic_owners"], fixture["private_streams"], fixture["shared_tail_streams"]) != (15, 6, 12, 3)
        or fixture["stream_lengths"] != expected_lengths or fixture["symbols"] != full_symbols
        or fixture["source"] != "public frozen block geometry proxy; no observed model symbols"
        or protocol.sha256_hex(fixture["fixture_sha256"], "representative fixture") != expected_fixture_sha
    ):
        raise ValueError("representative deterministic fixture binding")
    _validate_resource_plan(protocol, fixture["resource_plan"], symbols=full_symbols, streams=15)
    outer = protocol.strict_fields(
        row["outer_fold"],
        required=(
            "train_streams", "validation_streams", "test_streams", "all_150_candidates_fit_and_scored",
            "candidate_count", "candidate_scores", "candidate_scores_sha256", "winner",
            "validation_charged_bits", "test_logical_lengths", "test_logical_lengths_sha256",
            "final_full_panel_logical_lengths", "final_full_panel_logical_lengths_sha256",
            "winner_development_q016_sha256", "final_full_panel_q016_sha256", "serialized_model_sha256",
            "winner_refit_on_complete_development", "final_full_panel_fit",
            "literal_container_parse_decode_reencode_rebuild", "container_sha256", "canonical_rebuild_sha256",
            "decoded_reencoded_stream_count", "decoded_triplet_sha256s", "decoded_triplet_commitment_sha256",
        ),
        label="representative outer fold",
    )
    expected_development = list(range(2, 12)) + [13, 14]
    if outer["train_streams"] != list(range(4, 12)) + [13, 14] or outer["validation_streams"] != [2, 3] or outer["test_streams"] != [0, 1, 12]:
        raise ValueError("representative exact split")
    scores = outer["candidate_scores"]
    if outer["candidate_count"] != 150 or not isinstance(scores, list) or len(scores) != 150 or outer["all_150_candidates_fit_and_scored"] is not True:
        raise ValueError("representative candidate coverage")
    clean_scores = []
    for ordinal, (score, candidate) in enumerate(zip(scores, common.candidate_bank(), strict=True)):
        item = protocol.strict_fields(score, required=tuple(candidate.as_dict()) + ("validation_charged_bits",), label="representative candidate score")
        if {name: item[name] for name in candidate.as_dict()} != candidate.as_dict() or item["selector_ordinal"] != ordinal:
            raise ValueError("representative canonical candidate ordering")
        protocol.exact_int(item["validation_charged_bits"], "representative validation bits", 1, (1 << 63) - 1)
        clean_scores.append(dict(item))
    if outer["candidate_scores_sha256"] != hashlib.sha256(common.canonical_json(clean_scores)).hexdigest():
        raise ValueError("representative candidate-score seal")
    winning_score = min(clean_scores, key=lambda item: (item["validation_charged_bits"], item["selector_ordinal"]))
    winner = protocol.strict_fields(outer["winner"], required=tuple(common.candidate_bank()[0].as_dict()), label="representative winner")
    expected_winner = common.candidate_bank()[int(winning_score["selector_ordinal"])].as_dict()
    if dict(winner) != expected_winner or outer["validation_charged_bits"] != winning_score["validation_charged_bits"]:
        raise ValueError("representative winner derivation")
    test_lengths, test_sha = _u64_list_sha256(protocol, outer["test_logical_lengths"], count=3, label="representative test length")
    full_lengths, full_sha = _u64_list_sha256(protocol, outer["final_full_panel_logical_lengths"], count=15, label="representative full length")
    if outer["test_logical_lengths_sha256"] != test_sha or outer["final_full_panel_logical_lengths_sha256"] != full_sha:
        raise ValueError("representative logical-length commitments")
    for name in ("winner_development_q016_sha256", "final_full_panel_q016_sha256", "serialized_model_sha256", "container_sha256", "canonical_rebuild_sha256"):
        protocol.sha256_hex(outer[name], f"representative {name}")
    if outer["container_sha256"] != outer["canonical_rebuild_sha256"]:
        raise ValueError("representative canonical rebuild identity")
    for name in ("winner_refit_on_complete_development", "final_full_panel_fit", "literal_container_parse_decode_reencode_rebuild"):
        if outer[name] is not True:
            raise ValueError(f"representative pipeline flag: {name}")
    triplets = outer["decoded_triplet_sha256s"]
    if outer["decoded_reencoded_stream_count"] != 15 or not isinstance(triplets, list) or len(triplets) != 15:
        raise ValueError("representative decoded triplet count")
    for digest in triplets:
        protocol.sha256_hex(digest, "representative decoded triplet")
    if outer["decoded_triplet_commitment_sha256"] != hashlib.sha256(common.canonical_json(triplets)).hexdigest():
        raise ValueError("representative decoded triplet commitment")
    runtime = protocol.strict_fields(
        row["runtime_projection"],
        required=("measured_seconds", "warmup_seconds_excluded", "measured_cell_symbol_updates", "measured_updates_per_second", "conservative_updates_per_second", "projection_formula", "pipelines_source_plus_four_shuffles_plus_eight_controls", "projected_cell_symbol_updates", "projected_wall_seconds", "budget_seconds", "passes"),
        label="representative runtime projection",
    )
    train_symbols = sum(expected_lengths[index] for index in outer["train_streams"])
    validation_symbols = sum(expected_lengths[index] for index in outer["validation_streams"])
    development_symbols = sum(expected_lengths[index] for index in expected_development)
    test_symbols = sum(expected_lengths[index] for index in outer["test_streams"])
    one_fold_updates = 150 * (train_symbols + validation_symbols) + development_symbols + test_symbols
    measured_updates = one_fold_updates + 2 * full_symbols
    projected_updates = 13 * (6 * one_fold_updates + 2 * full_symbols)
    measured_seconds = protocol.finite_float(runtime["measured_seconds"], "representative measured seconds", positive=True)
    warmup_seconds = protocol.finite_float(runtime["warmup_seconds_excluded"], "representative warmup seconds", positive=True)
    throughput = protocol.finite_float(runtime["measured_updates_per_second"], "representative throughput", positive=True)
    conservative = protocol.finite_float(runtime["conservative_updates_per_second"], "representative conservative throughput", positive=True)
    projected_seconds = protocol.finite_float(runtime["projected_wall_seconds"], "representative projected seconds", positive=True)
    def close(observed: float, expected: float) -> bool:
        return abs(observed - expected) <= 8.0 * math.ulp(expected)
    if (
        runtime["measured_cell_symbol_updates"] != measured_updates
        or runtime["projected_cell_symbol_updates"] != projected_updates
        or runtime["pipelines_source_plus_four_shuffles_plus_eight_controls"] != 13
        or runtime["projection_formula"] != "13 pipelines * (6 complete outer folds + final fit/score), at 50% measured throughput"
        or runtime["budget_seconds"] != MAX_PROJECTED_WALL_SECONDS
        or runtime["passes"] is not True or projected_seconds > MAX_PROJECTED_WALL_SECONDS
        or not close(throughput, measured_updates / measured_seconds)
        or not close(conservative, throughput * 0.5)
        or not close(projected_seconds, projected_updates / conservative)
    ):
        raise ValueError("representative runtime derivation")
    environment = _validate_environment_identity(protocol, row["telemetry"], independent)
    delta_names = (
        "h2d_bytes", "h2d_payload_bytes", "h2d_root_descriptor_bytes", "h2d_subset_descriptor_bytes",
        "h2d_launch_descriptor_bytes", "h2d_model_table_bytes", "h2d_kernel_scalar_bytes", "d2h_bytes",
        "d2d_descriptor_bytes", "device_output_allocation_bytes", "kernel_count", "count_kernel_count",
        "length_kernel_count", "count_cell_symbol_updates", "length_cell_symbol_updates", "pack_calls",
        "subset_calls", "to_host_calls",
    )
    delta = protocol.strict_fields(row["measured_phase_statistics_delta"], required=delta_names, label="representative phase telemetry")
    clean_delta = {name: protocol.exact_int(delta[name], f"representative phase {name}", 0, (1 << 63) - 1) for name in delta_names}
    if clean_delta["h2d_bytes"] != sum(clean_delta[name] for name in (
        "h2d_payload_bytes", "h2d_root_descriptor_bytes", "h2d_subset_descriptor_bytes",
        "h2d_launch_descriptor_bytes", "h2d_model_table_bytes", "h2d_kernel_scalar_bytes",
    )) or clean_delta["kernel_count"] != clean_delta["count_kernel_count"] + clean_delta["length_kernel_count"]:
        raise ValueError("representative phase telemetry conservation")
    stats = environment["statistics"]
    expected_flags = {
        "model_h2d_bytes_nonzero": clean_delta["h2d_model_table_bytes"] > 0,
        "d2h_bytes_nonzero": clean_delta["d2h_bytes"] > 0,
        "peak_host_ram_recorded": int(stats["peak_process_tree_rss_bytes"]) > 0,
        "peak_vram_recorded": int(stats["peak_vram_incremental_bytes"]) > 0 and int(stats["telemetry_samples"]) > 0,
    }
    if any(row[name] is not expected for name, expected in expected_flags.items()) or not all(expected_flags.values()):
        raise ValueError("representative telemetry gates")
    return dict(row), environment


def validate_source_preflight(common: Any, protocol: Any, evidence: Any, bindings: BoundEvidence) -> dict[str, Any]:
    if not isinstance(evidence, SourcePreflightEvidence):
        raise ValueError("typed SourcePreflightEvidence required")
    all150 = dict(evidence.all150)
    representative = dict(evidence.representative)
    independent = dict(evidence.independent_gpu_identity)
    all150, all150_environment = _validate_all150_receipt(common, protocol, all150, bindings, independent)
    representative, representative_environment = _validate_representative_receipt(common, protocol, representative, bindings, independent)
    if (all150_environment["device_uuid"], all150_environment["pci_bus_id"]) != (representative_environment["device_uuid"], representative_environment["pci_bus_id"]):
        raise ValueError("all150/representative device mismatch")
    record = {
        "schema": "uwfa-sc-v6-bound-source-preflight",
        "source_snapshot_root_sha256": bindings.source_snapshot_root_sha256,
        "all150": all150,
        "representative": representative,
        "independent_gpu_identity": independent,
    }
    observed = hashlib.sha256(common.canonical_json(record)).hexdigest()
    if observed != evidence.receipt_sha256 or observed != bindings.source_preflight_receipt_sha256:
        raise ValueError("source preflight receipt binding")
    return {**record, "receipt_sha256": observed}


def _host_list(value: Any, backend: Any | None = None) -> list[int]:
    if backend is not None and hasattr(backend, "to_host_list"):
        return backend.to_host_list(value)
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def packed_rows(streams: Sequence[Mapping[str, Any]]) -> list[tuple[bytes, bytes, bytes]]:
    return [(bytes(row["bits_bytes"]), bytes(row["levels_bytes"]), bytes(row["base_bytes"])) for row in streams]


def prepare_backend_cache(backend: Any, panel: Mapping[str, Any]) -> Any:
    return backend.pack_streams(packed_rows(panel["streams"]))


def _subset(backend: Any, cache: Any, indices: Sequence[int]) -> Any:
    if not indices:
        raise ValueError("empty stream subset")
    if hasattr(backend, "subset"):
        return backend.subset(cache, list(indices))
    # Source-free CPU reference backends may expose rows directly.
    if isinstance(cache, dict) and "rows" in cache:
        return {"rows": [cache["rows"][index] for index in indices]}
    raise TypeError("backend must provide descriptor-only subset selection")


def fit_candidate(common: Any, backend: Any, cache: Any, indices: Sequence[int], candidate: Any) -> list[int]:
    subset = _subset(backend, cache, indices)
    counts = _host_list(backend.fit_counts(subset, candidate.topology_id, candidate.states, candidate.reset_length), backend)
    return common.q16_frequencies_from_counts(counts)


def exact_lengths(
    common: Any,
    backend: Any,
    cache: Any,
    indices: Sequence[int],
    candidate: Any,
    frequencies: Sequence[int],
) -> list[int]:
    subset = _subset(backend, cache, indices)
    return _host_list(backend.exact_lengths(subset, candidate.topology_id, candidate.states, candidate.reset_length, frequencies), backend)


def validation_score(common: Any, lengths: Sequence[int], candidate: Any) -> int:
    return 8 * sum((int(value) + 7) // 8 for value in lengths) + 8 * common.model_ledger(candidate)["physical_model_bytes"]


def gpu_preflight_all_150(common: Any, backend: Any, source_snapshot_root_sha256: str) -> dict[str, Any]:
    """Exact all-cell CPU/CuPy source-free replay on the actual runtime."""
    from array import array

    if not isinstance(source_snapshot_root_sha256, str) or len(source_snapshot_root_sha256) != 64:
        raise ValueError("all150 source snapshot root")
    bytes.fromhex(source_snapshot_root_sha256)
    rows = []
    for stream_ordinal, length in enumerate((4097, 2053, 1031, 521)):
        bits = bytes((((index * (17 + stream_ordinal)) ^ (index >> (1 + stream_ordinal)) ^ (index // 31)) & 1) for index in range(length))
        levels = bytes((index + 3 * stream_ordinal) % common.LEVELS for index in range(length))
        # Visit every prior bin and both arithmetic extremes.
        base_values = []
        for index in range(length):
            if index % 257 == 0:
                base_values.append(1)
            elif index % 257 == 1:
                base_values.append(65535)
            else:
                bucket = (index // 4 + stream_ordinal) % common.PRIOR_BINS
                base_values.append(min(65535, max(1, bucket * 4096 + 2048)))
        packed_u16 = array("H", base_values)
        if sys.byteorder != "little":
            packed_u16.byteswap()
        rows.append((bits, levels, packed_u16.tobytes(), base_values))
    fixture_lengths = [len(row[0]) for row in rows]
    fixture_sha256 = hashlib.sha256(b"".join(
        bits + levels + base_bytes for bits, levels, base_bytes, _base in rows
    )).hexdigest()
    resource_plan = backend.pack_resource_plan(sum(fixture_lengths), len(rows))
    if resource_plan.get("passes") is not True:
        raise MemoryError("all150 source-free fixture resource admission")
    cache = backend.pack_streams([(bits, levels, base_bytes) for bits, levels, base_bytes, _base in rows])
    checked = []
    started = time.perf_counter()
    for candidate in common.candidate_bank():
        cpu_counts = common.merge_counts(
            common.count_stream_cpu(list(bits), list(levels), base_values, candidate)
            for bits, levels, _base_bytes, base_values in rows
        )
        first_counts = _host_list(backend.fit_counts(cache, candidate.topology_id, candidate.states, candidate.reset_length), backend)
        second_counts = _host_list(backend.fit_counts(cache, candidate.topology_id, candidate.states, candidate.reset_length), backend)
        if first_counts != cpu_counts or second_counts != cpu_counts:
            raise ValueError(f"all-150 CPU/CuPy count mismatch: {candidate}")
        fitted_cpu = common.q16_frequencies_from_counts(cpu_counts)
        fitted_gpu = common.q16_frequencies_from_counts(first_counts)
        if fitted_gpu != fitted_cpu:
            raise ValueError(f"all-150 fitted Q0.16 mismatch: {candidate}")
        cpu_lengths = [
            common.exact_stream_length_cpu(list(bits), list(levels), base_values, candidate, fitted_cpu)
            for bits, levels, _base_bytes, base_values in rows
        ]
        first_lengths = _host_list(backend.exact_lengths(cache, candidate.topology_id, candidate.states, candidate.reset_length, fitted_cpu), backend)
        second_lengths = _host_list(backend.exact_lengths(cache, candidate.topology_id, candidate.states, candidate.reset_length, fitted_cpu), backend)
        if first_lengths != cpu_lengths or second_lengths != cpu_lengths:
            raise ValueError(f"all-150 CPU/CuPy arithmetic mismatch: {candidate}")
        extreme = [1 if index & 1 else 65535 for index in range(common.model_frequency_count(candidate))]
        extreme_cpu = [
            common.exact_stream_length_cpu(list(bits), list(levels), base_values, candidate, extreme)
            for bits, levels, _base_bytes, base_values in rows
        ]
        extreme_gpu = _host_list(backend.exact_lengths(cache, candidate.topology_id, candidate.states, candidate.reset_length, extreme), backend)
        if extreme_gpu != extreme_cpu:
            raise ValueError(f"frequency-extreme CPU/CuPy mismatch: {candidate}")
        logical_bytes = b"".join(int(value).to_bytes(8, "little") for value in cpu_lengths)
        extreme_bytes = b"".join(int(value).to_bytes(8, "little") for value in extreme_cpu)
        cell = {
            **candidate.as_dict(),
            "count_tensor_sha256": hashlib.sha256(b"".join(int(value).to_bytes(8, "little") for value in cpu_counts)).hexdigest(),
            "count_entries": len(cpu_counts),
            "count_total": sum(cpu_counts),
            "fitted_q016_sha256": hashlib.sha256(b"".join(int(value).to_bytes(2, "little") for value in fitted_cpu)).hexdigest(),
            "logical_lengths": cpu_lengths,
            "logical_length_count": len(cpu_lengths),
            "logical_lengths_sha256": hashlib.sha256(logical_bytes).hexdigest(),
            "frequency_extreme_logical_lengths": extreme_cpu,
            "frequency_extreme_logical_lengths_sha256": hashlib.sha256(extreme_bytes).hexdigest(),
            "cpu_gpu_first_count_exact": True,
            "cpu_gpu_second_count_exact": True,
            "fitted_q016_cpu_gpu_exact": True,
            "cpu_gpu_first_logical_lengths_exact": True,
            "cpu_gpu_second_logical_lengths_exact": True,
            "frequency_extreme_cpu_gpu_exact": True,
            "repeated_gpu_run_exact": True,
        }
        cell["cell_result_sha256"] = hashlib.sha256(common.canonical_json(cell)).hexdigest()
        checked.append(cell)
    elapsed = time.perf_counter() - started
    return {
        "schema": "uwfa-sc-v6-all150-source-free-preflight",
        "source_snapshot_root_sha256": source_snapshot_root_sha256.lower(),
        "status": "PASS_ALL_150_CPU_CUPY_EXACT_REPEATED",
        "cells": checked,
        "cell_count": len(checked),
        "streams": len(rows),
        "symbols_per_complete_bank": sum(len(row[0]) for row in rows),
        "fixture_stream_lengths": fixture_lengths,
        "fixture_sha256": fixture_sha256,
        "resource_plan": resource_plan,
        "elapsed_seconds": elapsed,
        "environment": backend.environment_receipt() if hasattr(backend, "environment_receipt") else {},
        "frequency_extremes_tested": [1, 65535],
        "reset_boundaries_through_4096_tested": True,
        "candidate_selector_sha256": hashlib.sha256(common.canonical_json(list(range(150)))).hexdigest(),
        "cell_results_sha256": hashlib.sha256(common.canonical_json(checked)).hexdigest(),
    }


def representative_outer_fold_benchmark(
    common: Any,
    protocol: Any,
    container_codec: Any,
    semantic_codec: Any,
    backend: Any,
    source_snapshot_root_sha256: str,
) -> dict[str, Any]:
    """Source-free 15-stream/6-owner complete nested outer-fold benchmark."""
    from array import array

    if not isinstance(source_snapshot_root_sha256, str) or len(source_snapshot_root_sha256) != 64:
        raise ValueError("representative source snapshot root")
    bytes.fromhex(source_snapshot_root_sha256)

    warmup_started = time.perf_counter()
    warmup_cache = backend.pack_streams([(b"\x00", b"\x00", b"\x00\x80")])
    warmup_candidate = common.candidate_bank()[0]
    warmup_frequencies = fit_candidate(common, backend, warmup_cache, [0], warmup_candidate)
    exact_lengths(common, backend, warmup_cache, [0], warmup_candidate, warmup_frequencies)
    warmup_seconds = time.perf_counter() - warmup_started
    telemetry_before = backend.statistics_snapshot() if hasattr(backend, "statistics_snapshot") else {}

    experts = 6
    hidden, intermediate = 37, 43
    matrix_weights = hidden * intermediate
    shapes = tuple(semantic_codec.ExpertShape(index, hidden, intermediate) for index in range(experts))
    semantic_packet = semantic_codec.build_semantic_packet(shapes, b"representative-outer-fold-v6")
    owner_layout = [(index // 2,) for index in range(12)] + [(0, 1), (2, 3), (4, 5)]
    roles = [("gate" if index % 2 == 0 else "up") for index in range(12)] + ["down"] * 3
    public_lengths = tuple(131_071 + 257 * index for index in range(12)) + tuple(65_537 + 509 * index for index in range(3))
    rows = []
    packed_rows_input = []
    for ordinal, (owners, role, length) in enumerate(zip(owner_layout, roles, public_lengths, strict=True)):
        bits = bytes((((position * (17 + ordinal)) ^ (position >> 3) ^ (position // 31) ^ ordinal) & 1) for position in range(length))
        levels = bytes((position + ordinal) % common.LEVELS for position in range(length))
        base_values = [
            1 if position % 257 == 0 else 65535 if position % 257 == 1 else 2048 + 4096 * ((position // 4 + ordinal) % common.PRIOR_BINS)
            for position in range(length)
        ]
        packed_base = array("H", base_values)
        if sys.byteorder != "little":
            packed_base.byteswap()
        base_bytes = packed_base.tobytes()
        owner_set = protocol.owner_set_from_ordinals(experts, list(owners))
        contributions = tuple(
            {"expert": owner, "role": role, "source_offset": 0, "weight_count": matrix_weights}
            for owner in owners
        )
        rows.append({
            "stream_ordinal": ordinal,
            "owner_set": owner_set,
            "owner_set_hex": owner_set.hex(),
            "owners": tuple(owners),
            "role": role,
            "owner_contributions": contributions,
            "weight_charge": matrix_weights * len(owners),
            "shape_rows": 1,
            "shape_cols": matrix_weights * len(owners),
            "symbols": length,
            "bits": list(bits),
            "levels": list(levels),
            "base": base_values,
            "bits_bytes": bits,
            "levels_bytes": levels,
            "base_bytes": base_bytes,
        })
        packed_rows_input.append((bits, levels, base_bytes))
    if len(rows) != 15 or sum(len(row["owners"]) == 1 for row in rows) != 12 or sum(len(row["owners"]) > 1 for row in rows) != 3:
        raise AssertionError("representative fixture topology")
    representative_resource_plan = backend.pack_resource_plan(sum(public_lengths), len(rows))
    if representative_resource_plan.get("passes") is not True:
        raise MemoryError("representative source-free fixture resource admission")
    started = time.perf_counter()
    cache = backend.pack_streams(packed_rows_input)
    # The representative benchmark exercises the same total primary rule as
    # production: exclude only the exact held-out semantic identity.  The
    # stricter coordinate-disjoint rule remains a separate nonpromoting
    # diagnostic and is never silently substituted here.
    development = [index for index, row in enumerate(rows) if 0 not in row["owners"]]
    test = [index for index, row in enumerate(rows) if 0 in row["owners"]]
    validation = development[:2]
    train = development[2:]
    if not train or not validation or not test or set(train) & set(validation) or set(development) & set(test):
        raise AssertionError("representative outer-fold split")
    choices = []
    candidate_scores = []
    for candidate in common.candidate_bank():
        frequencies = fit_candidate(common, backend, cache, train, candidate)
        lengths = exact_lengths(common, backend, cache, validation, candidate, frequencies)
        charged = validation_score(common, lengths, candidate)
        choices.append((charged, candidate.selector_ordinal, candidate))
        candidate_scores.append({**candidate.as_dict(), "validation_charged_bits": charged})
    validation_bits, _selector, selected = min(choices)
    fitted_development = fit_candidate(common, backend, cache, development, selected)
    test_lengths = exact_lengths(common, backend, cache, test, selected, fitted_development)
    final_frequencies = fit_candidate(common, backend, cache, list(range(15)), selected)
    full_lengths = exact_lengths(common, backend, cache, list(range(15)), selected, final_frequencies)
    model_packet = common.serialize_model(selected, final_frequencies)
    stream_specs = []
    for row in rows:
        payload, logical = common.encode_unifilar_stream(row["bits"], row["levels"], row["base"], selected, final_frequencies)
        decoded = common.decode_unifilar_stream(payload, logical, row["levels"], row["base"], selected, final_frequencies)
        if decoded != row["bits"]:
            raise ValueError("representative serialized-model decode")
        contributions = tuple(
            container_codec.OwnerContribution(int(item["expert"]), str(item["role"]), int(item["source_offset"]), int(item["weight_count"]))
            for item in row["owner_contributions"]
        )
        stream_specs.append((
            container_codec.StreamSpec(
                ordinal=int(row["stream_ordinal"]),
                symbols=int(row["symbols"]),
                logical_bits=int(logical),
                payload=payload,
                source_digest=common.selected_decision_triplet_sha256(
                    bytes(row["bits_bytes"]), bytes(row["levels_bytes"]), bytes(row["base_bytes"])
                ),
                profile_q=0,
                decoder_scale=1.0,
                role=str(row["role"]),
                group_rows=int(row["shape_rows"]),
                group_cols=int(row["shape_cols"]),
                owner_contributions=contributions,
            ),
            row["owner_set"],
        ))
    regions = _regions_from_specs(container_codec, stream_specs, experts)
    bindings = {name: hashlib.sha256(("representative:" + name).encode("ascii")).hexdigest() for name in container_codec._HEADER_BINDINGS}
    container, _metrics = container_codec.build_container(
        common,
        semantic_codec,
        model_packet=model_packet,
        semantic_packet=semantic_packet,
        immutable_state=b"",
        regions=regions,
        weights=experts * 3 * matrix_weights,
        experts=experts,
        baseline_object_bytes=1_000_000,
        audited_relative_mse=0.025,
        baseline_artifact_sha256="11" * 32,
        reconstruction_sha256="22" * 32,
        audit_binding_sha256="33" * 32,
        binding_hashes=bindings,
    )
    parsed = container_codec.parse_container(common, semantic_codec, container)
    rebuilt = container_codec.canonical_rebuild(common, semantic_codec, parsed)
    if rebuilt != container:
        raise ValueError("representative literal canonical rebuild")
    decoded_triplets = []
    for row, parsed_row in zip(rows, parsed["directory"], strict=True):
        decoded = common.decode_unifilar_stream(
            parsed_row["payload"], int(parsed_row["logical_bits"]), row["levels"], row["base"], parsed["candidate"], parsed["frequencies"]
        )
        replay, replay_bits = common.encode_unifilar_stream(decoded, row["levels"], row["base"], parsed["candidate"], parsed["frequencies"])
        if replay != parsed_row["payload"] or replay_bits != parsed_row["logical_bits"]:
            raise ValueError("representative literal causal re-encode")
        decoded_triplets.append(str(parsed_row["source_digest"]))
    measured_seconds = time.perf_counter() - started
    train_symbols = sum(rows[index]["symbols"] for index in train)
    validation_symbols = sum(rows[index]["symbols"] for index in validation)
    development_symbols = sum(rows[index]["symbols"] for index in development)
    test_symbols = sum(rows[index]["symbols"] for index in test)
    full_symbols = sum(row["symbols"] for row in rows)
    one_fold_updates = 150 * (train_symbols + validation_symbols) + development_symbols + test_symbols
    measured_updates = one_fold_updates + 2 * full_symbols
    throughput = measured_updates / measured_seconds
    conservative_throughput = throughput * 0.5
    pipelines = 1 + 4 + len(common.CONTROL_SEEDS)
    projected_updates = pipelines * (6 * one_fold_updates + 2 * full_symbols)
    projected_seconds = projected_updates / conservative_throughput
    telemetry = backend.environment_receipt()
    stats = telemetry["statistics"]
    delta_keys = (
        "h2d_bytes", "h2d_payload_bytes", "h2d_root_descriptor_bytes",
        "h2d_subset_descriptor_bytes", "h2d_launch_descriptor_bytes",
        "h2d_model_table_bytes", "h2d_kernel_scalar_bytes", "d2h_bytes",
        "d2d_descriptor_bytes", "device_output_allocation_bytes", "kernel_count",
        "count_kernel_count", "length_kernel_count", "count_cell_symbol_updates",
        "length_cell_symbol_updates", "pack_calls", "subset_calls", "to_host_calls",
    )
    phase_delta = {
        key: int(stats[key]) - int(telemetry_before.get(key, 0))
        for key in delta_keys
    }
    return {
        "schema": "uwfa-sc-v6-representative-source-free-preflight",
        "source_snapshot_root_sha256": source_snapshot_root_sha256.lower(),
        "status": "PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD" if projected_seconds <= MAX_PROJECTED_WALL_SECONDS else "FAIL_PROJECTED_RUNTIME_BUDGET",
        "fixture": {
            "streams": 15,
            "semantic_owners": 6,
            "private_streams": 12,
            "shared_tail_streams": 3,
            "stream_lengths": list(public_lengths),
            "symbols": full_symbols,
            "source": "public frozen block geometry proxy; no observed model symbols",
            "fixture_sha256": hashlib.sha256(b"".join(row["bits_bytes"] + row["levels_bytes"] + row["base_bytes"] for row in rows)).hexdigest(),
            "resource_plan": representative_resource_plan,
        },
        "outer_fold": {
            "train_streams": train,
            "validation_streams": validation,
            "test_streams": test,
            "all_150_candidates_fit_and_scored": True,
            "candidate_count": len(candidate_scores),
            "candidate_scores": candidate_scores,
            "candidate_scores_sha256": hashlib.sha256(common.canonical_json(candidate_scores)).hexdigest(),
            "winner": selected.as_dict(),
            "validation_charged_bits": validation_bits,
            "test_logical_lengths": test_lengths,
            "test_logical_lengths_sha256": hashlib.sha256(b"".join(int(value).to_bytes(8, "little") for value in test_lengths)).hexdigest(),
            "final_full_panel_logical_lengths": full_lengths,
            "final_full_panel_logical_lengths_sha256": hashlib.sha256(b"".join(int(value).to_bytes(8, "little") for value in full_lengths)).hexdigest(),
            "winner_development_q016_sha256": hashlib.sha256(b"".join(int(value).to_bytes(2, "little") for value in fitted_development)).hexdigest(),
            "final_full_panel_q016_sha256": hashlib.sha256(b"".join(int(value).to_bytes(2, "little") for value in final_frequencies)).hexdigest(),
            "serialized_model_sha256": hashlib.sha256(model_packet).hexdigest(),
            "winner_refit_on_complete_development": True,
            "final_full_panel_fit": True,
            "literal_container_parse_decode_reencode_rebuild": True,
            "container_sha256": hashlib.sha256(container).hexdigest(),
            "canonical_rebuild_sha256": hashlib.sha256(rebuilt).hexdigest(),
            "decoded_reencoded_stream_count": len(decoded_triplets),
            "decoded_triplet_sha256s": decoded_triplets,
            "decoded_triplet_commitment_sha256": hashlib.sha256(common.canonical_json(decoded_triplets)).hexdigest(),
        },
        "runtime_projection": {
            "measured_seconds": measured_seconds,
            "warmup_seconds_excluded": warmup_seconds,
            "measured_cell_symbol_updates": measured_updates,
            "measured_updates_per_second": throughput,
            "conservative_updates_per_second": conservative_throughput,
            "projection_formula": "13 pipelines * (6 complete outer folds + final fit/score), at 50% measured throughput",
            "pipelines_source_plus_four_shuffles_plus_eight_controls": pipelines,
            "projected_cell_symbol_updates": projected_updates,
            "projected_wall_seconds": projected_seconds,
            "budget_seconds": MAX_PROJECTED_WALL_SECONDS,
            "passes": projected_seconds <= MAX_PROJECTED_WALL_SECONDS,
        },
        "telemetry": telemetry,
        "measured_phase_statistics_delta": phase_delta,
        "model_h2d_bytes_nonzero": int(phase_delta["h2d_model_table_bytes"]) > 0,
        "d2h_bytes_nonzero": int(phase_delta["d2h_bytes"]) > 0,
        "peak_host_ram_recorded": int(stats["peak_process_tree_rss_bytes"]) > 0,
        "peak_vram_recorded": int(stats["peak_vram_incremental_bytes"]) > 0 and int(stats["telemetry_samples"]) > 0,
    }


def attach_semantic_owners(protocol: Any, panel: dict[str, Any]) -> None:
    if not isinstance(panel, dict):
        raise ValueError("panel must be an object before semantic ownership")
    experts = protocol.validate_experts(panel.get("experts"))
    protocol.exact_int(panel.get("weights"), "panel weights before semantic ownership", 1, protocol.MAX_WEIGHTS)
    if not isinstance(panel.get("streams"), list) or not 1 <= len(panel["streams"]) <= protocol.MAX_STREAMS:
        raise ValueError("panel streams before semantic ownership")
    if "semantic_identities" in panel:
        raw_identities = panel["semantic_identities"]
        if not isinstance(raw_identities, (tuple, list)) or len(raw_identities) != experts:
            raise ValueError("semantic identity count before conversion")
        identities = []
        for row in raw_identities:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("semantic identity geometry")
            identities.append((
                protocol.exact_int(row[0], "semantic layer identity", 0, (1 << 31) - 1),
                protocol.exact_int(row[1], "semantic expert identity", 0, (1 << 31) - 1),
            ))
    else:
        route = panel["artifact"]["route_rows"]
        if len(route) != 3 * experts:
            raise ValueError("evaluation-plugin route geometry")
        identities = []
        for expert in range(experts):
            gate = route[3 * expert]
            identities.append((
                protocol.exact_int(gate["layer"], "route layer identity", 0, (1 << 31) - 1),
                protocol.exact_int(gate["expert"], "route expert identity", 0, (1 << 31) - 1),
            ))
    if len(identities) != experts:
        raise ValueError("semantic identity count")
    if len(set(identities)) != experts:
        raise ValueError("artifact route identities are not unique")
    owned_counts = [0] * experts
    owned_weights = [0] * experts
    for row in panel["streams"]:
        owner_set = bytes.fromhex(protocol.owner_set_hex(row["owner_set_hex"], experts))
        owners = list(protocol.owner_ordinals(owner_set, experts))
        if not owners:
            raise ValueError("stream has no semantic owner")
        contributions = row.get("owner_contributions")
        if not isinstance(contributions, (tuple, list)) or not contributions:
            raise ValueError("stream lacks exact semantic contributions")
        weight_by_owner = {owner: 0 for owner in owners}
        for contribution in contributions:
            owner = protocol.exact_int(contribution["expert"], "contribution expert", 0, experts - 1)
            count = protocol.exact_int(contribution["weight_count"], "contribution weight", 1, protocol.MAX_WEIGHTS)
            if owner not in weight_by_owner:
                raise ValueError("contribution expert outside owner set")
            weight_by_owner[owner] += count
        if any(value <= 0 for value in weight_by_owner.values()) or sum(weight_by_owner.values()) != int(row["weight_charge"]):
            raise ValueError("per-owner weight conservation")
        for expert in owners:
            owned_counts[expert] += 1
            owned_weights[expert] += weight_by_owner[expert]
        row["owner_expert_ordinals"] = owners
        row["owner_identity_indices"] = owners
        row["owner_weight_contributions"] = weight_by_owner
        row["owner_set"] = owner_set
    if any(value == 0 for value in owned_counts) or any(value == 0 for value in owned_weights):
        raise ValueError("empty expert may not amortize shared bytes")
    if sum(owned_weights) != int(panel["weights"]):
        raise ValueError("panel semantic source weights do not conserve")
    panel["semantic_identities"] = identities


def _split_digest(protocol: Any, identity: tuple[int, int], row: Mapping[str, Any]) -> bytes:
    digest = protocol.length_prefixed_digest(
        [identity[0], identity[1], int(row["stream_ordinal"]), bytes(row["owner_set"])],
        domain=b"UWFA-SC-V6-NESTED-SPLIT-2026-09-02\x00",
    )
    return bytes.fromhex(digest)


def _fold_plan(common: Any, protocol: Any, panel: Mapping[str, Any], *, policy: str) -> list[dict[str, Any]]:
    if policy not in {"exact_identity", "coordinate_disjoint"}:
        raise ValueError("unknown fold policy")
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    plans = []
    for identity_index, identity in enumerate(identities):
        layer, expert = identity
        test_indices = [index for index, row in enumerate(streams) if identity_index in row["owner_identity_indices"]]
        if policy == "exact_identity":
            development_indices = [
                index for index, row in enumerate(streams)
                if identity_index not in row["owner_identity_indices"]
            ]
        else:
            development_indices = [
                index for index, row in enumerate(streams)
                if all(
                    identities[owner][0] != layer and identities[owner][1] != expert
                    for owner in row["owner_identity_indices"]
                )
            ]
        estimable = bool(test_indices) and len(development_indices) >= 2
        ranked: list[int] = []
        validation: list[int] = []
        train: list[int] = []
        if estimable:
            ranked = sorted(
                development_indices,
                key=lambda index: (_split_digest(protocol, identity, streams[index]), int(streams[index]["stream_ordinal"])),
            )
            validation_count = min(max(1, len(ranked) // common.INNER_VALIDATION_MODULUS), len(ranked) - 1)
            validation = ranked[:validation_count]
            train = ranked[validation_count:]
        plans.append({
            "identity_index": identity_index,
            "identity": tuple(identity),
            "policy": policy,
            "test_indices": test_indices,
            "development_indices": development_indices,
            "validation_indices": validation,
            "train_indices": train,
            "estimable": estimable,
            "reason": None if estimable else "requires nonempty test and at least two independent development streams",
        })
    return plans


def projected_updates(common: Any, protocol: Any, panel: Mapping[str, Any]) -> dict[str, Any]:
    total = 0
    folds = []
    streams = panel["streams"]
    exact_plans = _fold_plan(common, protocol, panel, policy="exact_identity")
    strict_plans = _fold_plan(common, protocol, panel, policy="coordinate_disjoint")
    for plan in exact_plans:
        identity_index = int(plan["identity_index"])
        if not plan["estimable"]:
            folds.append({"identity_index": identity_index, "estimable": False, "cell_symbol_updates": 0})
            continue
        development_indices = plan["development_indices"]
        test_indices = plan["test_indices"]
        validation = plan["validation_indices"]
        train = plan["train_indices"]
        train_symbols = sum(int(streams[index]["symbols"]) for index in train)
        validation_symbols = sum(int(streams[index]["symbols"]) for index in validation)
        development_symbols = sum(int(streams[index]["symbols"]) for index in development_indices)
        test_symbols = sum(int(streams[index]["symbols"]) for index in test_indices)
        updates = len(common.candidate_bank()) * (train_symbols + validation_symbols) + development_symbols + test_symbols
        total += updates
        folds.append({"identity_index": identity_index, "estimable": True, "cell_symbol_updates": updates})
    final_symbols = sum(int(row["symbols"]) for row in streams)
    stream_count = len(streams)
    payload_bytes = 4 * final_symbols
    descriptor_bytes = 16 * stream_count
    additional_host_bytes = payload_bytes + 64 * stream_count + HOST_ALLOCATION_RESERVE_BYTES
    device_required_bytes = payload_bytes + descriptor_bytes + MAX_AUXILIARY_DEVICE_BYTES + VRAM_ALLOCATION_RESERVE_BYTES
    static_resource_pass = bool(
        1 <= stream_count <= MAX_BACKEND_STREAMS
        and 1 <= final_symbols <= MAX_PACKED_SYMBOLS
        and additional_host_bytes <= MAX_HOST_BYTES
        and device_required_bytes <= MAX_VRAM_BYTES
    )
    total += 2 * final_symbols  # final fit plus exact final scoring
    strict_diagnostic_updates = 0
    strict_estimable = 0
    for plan in strict_plans:
        if not plan["estimable"]:
            continue
        strict_estimable += 1
        train_symbols = sum(int(streams[index]["symbols"]) for index in plan["train_indices"])
        validation_symbols = sum(int(streams[index]["symbols"]) for index in plan["validation_indices"])
        development_symbols = sum(int(streams[index]["symbols"]) for index in plan["development_indices"])
        test_symbols = sum(int(streams[index]["symbols"]) for index in plan["test_indices"])
        strict_diagnostic_updates += len(common.candidate_bank()) * (train_symbols + validation_symbols) + development_symbols + test_symbols
    maximum_survivor_updates = 5 * total + strict_diagnostic_updates  # source + four shuffles + optional stricter diagnostic
    projected_seconds = maximum_survivor_updates / MIN_CALIBRATED_CELL_SYMBOLS_PER_SECOND
    return {
        "exact_cell_symbol_updates": total,
        "maximum_source_survivor_updates_including_four_shuffles": maximum_survivor_updates,
        "folds": folds,
        "primary_fold_policy": "exact_identity_exclusion",
        "primary_exact_identity_estimable": all(bool(row["estimable"]) for row in folds),
        "coordinate_disjoint_diagnostic_estimable_folds": strict_estimable,
        "coordinate_disjoint_diagnostic_cell_symbol_updates": strict_diagnostic_updates,
        "conservative_minimum_throughput_updates_per_second": MIN_CALIBRATED_CELL_SYMBOLS_PER_SECOND,
        "projected_wall_seconds": projected_seconds,
        "frozen_maximum_wall_seconds": MAX_PROJECTED_WALL_SECONDS,
        "passes_pre_fit_runtime_budget": projected_seconds <= MAX_PROJECTED_WALL_SECONDS,
        "static_resource_admission": {
            "symbols": final_symbols,
            "streams": stream_count,
            "payload_host_and_device_bytes": payload_bytes,
            "root_descriptor_device_bytes": descriptor_bytes,
            "additional_host_bytes_including_reserve": additional_host_bytes,
            "device_required_bytes_including_aux_and_reserve": device_required_bytes,
            "maximum_packed_symbols": MAX_PACKED_SYMBOLS,
            "host_cap_bytes": MAX_HOST_BYTES,
            "vram_cap_bytes": MAX_VRAM_BYTES,
            "passes": static_resource_pass,
            "checked_before_backend_pack_or_cupy_allocation": True,
        },
        "passes_pre_fit_resource_budget": static_resource_pass,
        "host_byte_budget": MAX_HOST_BYTES,
        "vram_byte_budget": MAX_VRAM_BYTES,
    }


def _clone_panel_with_streams(panel: Mapping[str, Any], streams: list[dict[str, Any]]) -> dict[str, Any]:
    clone = dict(panel)
    clone["streams"] = streams
    return clone


def within_context_permutation(common: Any, protocol: Any, panel: Mapping[str, Any], seed: int = 0x5C11A2) -> dict[str, Any]:
    """Preserve exact public-context bit counts while destroying sequence order."""
    output = []
    for row in panel["streams"]:
        clone = dict(row)
        bits = list(row["bits_bytes"])
        levels = list(row["levels_bytes"])
        base_values = list(row["base"])
        groups: dict[int, list[int]] = {}
        for position, (level, base) in enumerate(zip(levels, base_values, strict=True)):
            context = common.public_context(level, base, position & 3)
            groups.setdefault(context, []).append(position)
        shuffled = list(bits)
        for context, positions in groups.items():
            ranked_sources = sorted(
                positions,
                key=lambda position: protocol.length_prefixed_digest(
                    [seed, int(row["stream_ordinal"]), context, position],
                    domain=b"UWFA-SC-V6-WITHIN-CONTEXT-PERMUTATION\x00",
                ),
            )
            values = [bits[position] for position in ranked_sources]
            for destination, value in zip(positions, values, strict=True):
                shuffled[destination] = value
        clone["bits"] = shuffled
        clone["bits_bytes"] = bytes(shuffled)
        output.append(clone)
    return _clone_panel_with_streams(panel, output)


def multiscale_chunk_shuffle(protocol: Any, panel: Mapping[str, Any], chunk: int, seed: int = 0x71A55E) -> dict[str, Any]:
    if chunk not in {32, 128, 512}:
        raise ValueError("frozen multiscale chunk")
    output = []
    for row in panel["streams"]:
        length = int(row["symbols"])
        spans = [(begin, min(length, begin + chunk)) for begin in range(0, length, chunk)]
        ranked = sorted(
            enumerate(spans),
            key=lambda item: protocol.length_prefixed_digest(
                [seed, chunk, int(row["stream_ordinal"]), item[0]],
                domain=b"UWFA-SC-V6-MULTISCALE-CHUNK-SHUFFLE\x00",
            ),
        )
        order = [spans[index] for index, _span in ranked]
        bits = b"".join(bytes(row["bits_bytes"])[begin:end] for begin, end in order)
        levels = b"".join(bytes(row["levels_bytes"])[begin:end] for begin, end in order)
        base_parts = []
        base_raw = bytes(row["base_bytes"])
        for begin, end in order:
            base_parts.append(base_raw[2 * begin:2 * end])
        base_bytes = b"".join(base_parts)
        base_values = list(int.from_bytes(base_bytes[index:index + 2], "little") for index in range(0, len(base_bytes), 2))
        clone = dict(row)
        clone.update({
            "bits": list(bits), "bits_bytes": bits,
            "levels": list(levels), "levels_bytes": levels,
            "base": base_values, "base_bytes": base_bytes,
        })
        output.append(clone)
    return _clone_panel_with_streams(panel, output)


def survivor_shuffle_diagnostics(common: Any, protocol: Any, backend: Any, panel: Mapping[str, Any]) -> dict[str, Any]:
    variants = [("within_public_context", within_context_permutation(common, protocol, panel))]
    variants.extend((f"chunk_{chunk}", multiscale_chunk_shuffle(protocol, panel, chunk)) for chunk in (32, 128, 512))
    rows = []
    for name, variant in variants:
        cache = prepare_backend_cache(backend, variant)
        scientific = nested_holdout(common, protocol, backend, cache, variant)
        rows.append({
            "variant": name,
            "pooled_exact_heldout_saving_bpw": scientific["pooled_exact_heldout_saving_bpw"],
            "selected": scientific["final_topology_selected_from_nested_fold_votes"],
            "complete_150_cell_nested_search_repeated": True,
        })
    return {
        "status": "PASS_PREDECLARED_SHUFFLE_PIPELINES_EXECUTED",
        "variants": rows,
        "claim_use": "source minus shuffle gaps diagnose nonlocal order; diagnostics never rescue a physical failure",
    }


def nested_holdout(
    common: Any,
    protocol: Any,
    backend: Any,
    cache: Any,
    panel: Mapping[str, Any],
    *,
    policy: str = "exact_identity",
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    plans = _fold_plan(common, protocol, panel, policy=policy)
    skipped = [
        {
            "outer_identity_index": int(plan["identity_index"]),
            "outer_identity": list(plan["identity"]),
            "reason": plan["reason"],
        }
        for plan in plans if not plan["estimable"]
    ]
    if skipped and not diagnostic_only:
        return {
            "kind": "whole-artifact-semantic-expert-folds_with_owner_attribution",
            "primary_policy": "exact_identity_exclusion",
            "status": "NOT_ESTIMABLE_EXACT_IDENTITY_HOLDOUT",
            "folds": [],
            "skipped_folds": skipped,
            "estimable": False,
            "passes_heldout_gate": False,
            "positive_promotion": False,
        }
    valid_plans = [plan for plan in plans if plan["estimable"]]
    if not valid_plans:
        return {
            "kind": "coordinate-disjoint-nonpromoting-diagnostic",
            "primary_policy": policy,
            "status": "NOT_RUN_NO_NONEMPTY_COORDINATE_DISJOINT_FOLDS",
            "folds": [],
            "skipped_folds": skipped,
            "estimable": False,
            "passes_heldout_gate": False,
            "positive_promotion": False,
        }
    fold_rows = []
    for plan in valid_plans:
        identity_index = int(plan["identity_index"])
        identity = identities[identity_index]
        layer, expert = identity
        test_indices = plan["test_indices"]
        development_indices = plan["development_indices"]
        validation_indices = plan["validation_indices"]
        train_indices = plan["train_indices"]
        selections = []
        for candidate in common.candidate_bank():
            frequencies = fit_candidate(common, backend, cache, train_indices, candidate)
            lengths = exact_lengths(common, backend, cache, validation_indices, candidate, frequencies)
            selections.append((validation_score(common, lengths, candidate), candidate.selector_ordinal, candidate))
        validation_bits, _ordinal, selected = min(selections)
        frequencies = fit_candidate(common, backend, cache, development_indices, selected)
        test_lengths = exact_lengths(common, backend, cache, test_indices, selected, frequencies)
        baseline_allocated_bits = 0.0
        candidate_allocated_bits = 0.0
        allocated_weights = 0.0
        for index, logical in zip(test_indices, test_lengths, strict=True):
            row = streams[index]
            owner_count = len(row["owner_identity_indices"])
            baseline_allocated_bits += 8.0 * int(row["baseline_payload_bytes"]) / owner_count
            candidate_allocated_bits += 8.0 * ((int(logical) + 7) // 8) / owner_count
            allocated_weights += int(row["owner_weight_contributions"][identity_index])
        # Each scientific fold must carry a full independent model.
        model_bits = 8 * common.model_ledger(selected)["physical_model_bytes"]
        saving = (baseline_allocated_bits - candidate_allocated_bits - model_bits) / allocated_weights
        fold_rows.append({
            "outer_identity_index": identity_index,
            "outer_layer_from_artifact": layer,
            "outer_expert_from_artifact": expert,
            "development_exclusion_policy": policy,
            "test_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in test_indices],
            "development_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in development_indices],
            "inner_train_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in train_indices],
            "inner_validation_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in validation_indices],
            "selected_by_inner_validation_only": selected.as_dict(),
            "inner_validation_exact_charged_bits": validation_bits,
            "charged_full_fold_model_bits": model_bits,
            "allocated_test_weights": allocated_weights,
            "allocated_baseline_bits": baseline_allocated_bits,
            "allocated_candidate_bits": candidate_allocated_bits,
            "exact_test_saving_bpw": saving,
        })
    allocated_total = sum(float(row["allocated_test_weights"]) for row in fold_rows)
    if not diagnostic_only and abs(allocated_total - int(panel["weights"])) > 1e-6:
        raise ValueError("owner-attributed outer folds do not partition weights")
    pooled_saved_bits = sum(
        float(row["allocated_baseline_bits"]) - float(row["allocated_candidate_bits"]) - float(row["charged_full_fold_model_bits"])
        for row in fold_rows
    )
    pooled = pooled_saved_bits / allocated_total
    values = [float(row["exact_test_saving_bpw"]) for row in fold_rows]
    mean = statistics.fmean(values)
    if len(values) > 1:
        degrees_of_freedom = len(values) - 1
        critical = T95_TWO_SIDED[min(degrees_of_freedom, 30)]
        margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    else:
        degrees_of_freedom = 0
        critical = math.inf
        margin = math.inf
    candidate_votes: dict[int, int] = {}
    for row in fold_rows:
        ordinal = int(row["selected_by_inner_validation_only"]["selector_ordinal"])
        candidate_votes[ordinal] = candidate_votes.get(ordinal, 0) + 1
    selected_ordinal = min(candidate_votes, key=lambda ordinal: (-candidate_votes[ordinal], ordinal))
    selected = common.candidate_bank()[selected_ordinal]
    return {
        "kind": "coordinate-disjoint-nonpromoting-diagnostic" if diagnostic_only else "whole-artifact-semantic-expert-folds_with_owner_attribution",
        "primary_policy": policy,
        "status": "PASS_NONPROMOTING_COORDINATE_DISJOINT_DIAGNOSTIC" if diagnostic_only else "PASS_EXACT_IDENTITY_PRIMARY_HOLDOUT",
        "folds": fold_rows,
        "skipped_folds": skipped,
        "estimable": True,
        "pooled_exact_heldout_saving_bpw": pooled,
        "minimum_fold_exact_saving_bpw": min(values),
        "whole_expert_mean_saving_bpw": mean,
        "whole_expert_t95_lower_bpw": mean - margin,
        "whole_expert_t95_upper_bpw": mean + margin,
        "confidence_rule": "two-sided Student-t interval with df derived from actual valid fold count; df>30 uses conservative df=30 critical",
        "confidence_degrees_of_freedom": degrees_of_freedom,
        "confidence_critical_95": critical,
        "candidate_vote_counts": candidate_votes,
        "final_topology_selected_from_nested_fold_votes": selected.as_dict(),
        "passes_pooled_standalone_threshold": pooled >= common.STANDALONE_REQUIRED_SAVING_BPW,
        "passes_positive_whole_expert_lower_confidence": mean - margin > 0.0,
        "passes_heldout_gate": (not diagnostic_only) and pooled >= common.STANDALONE_REQUIRED_SAVING_BPW and mean - margin > 0.0,
        "positive_promotion": False,
    }


def coordinate_disjoint_diagnostic(common: Any, protocol: Any, backend: Any, cache: Any, panel: Mapping[str, Any]) -> dict[str, Any]:
    return nested_holdout(
        common, protocol, backend, cache, panel,
        policy="coordinate_disjoint", diagnostic_only=True,
    )


def _regions_from_specs(container_codec: Any, stream_specs: Sequence[Any], experts: int) -> list[Any]:
    grouped: dict[bytes, list[Any]] = {}
    for spec, owner_set in stream_specs:
        owner_set = bytes(owner_set)
        container_codec.owner_ordinals(owner_set, experts)
        grouped.setdefault(owner_set, []).append(spec)
    # Private expert regions first in expert order, then explicitly shared
    # owner regions.  Within each region stream ordinals remain canonical.
    owner_sets = sorted(
        grouped,
        key=lambda value: (
            len(container_codec.owner_ordinals(value, experts)) != 1,
            container_codec.owner_ordinals(value, experts),
        ),
    )
    return [
        container_codec.RegionSpec(owner_set, tuple(sorted(grouped[owner_set], key=lambda row: row.ordinal)))
        for owner_set in owner_sets
    ]


def final_container(
    common: Any,
    container_codec: Any,
    semantic_codec: Any,
    adapter: Any,
    backend: Any,
    cache: Any,
    panel: Mapping[str, Any],
    candidate: Any,
    score: Mapping[str, Any],
    bindings: BoundEvidence,
    authenticated_descriptor_source_builder: Any,
) -> dict[str, Any]:
    indices = list(range(len(panel["streams"])))
    fitted = fit_candidate(common, backend, cache, indices, candidate)
    transmitted_model = common.serialize_model(candidate, fitted)
    # Everything below consumes only the serialized model copy.
    decoded_candidate, decoded_frequencies = common.deserialize_model(transmitted_model)
    stream_specs = []
    identity_specs = []
    payload_rows = []
    for row in panel["streams"]:
        payload, logical = common.encode_unifilar_stream(
            row["bits"], row["levels"], row["base"], decoded_candidate, decoded_frequencies
        )
        decoded = common.decode_unifilar_stream(
            payload, logical, row["levels"], row["base"], decoded_candidate, decoded_frequencies
        )
        if decoded != row["bits"]:
            raise ValueError("serialized-model causal decode mismatch")
        contributions = tuple(
            container_codec.OwnerContribution(
                int(item["expert"]),
                str(item["role"]),
                int(item["source_offset"]),
                int(item["weight_count"]),
            )
            for item in row["owner_contributions"]
        )
        spec = container_codec.StreamSpec(
            ordinal=int(row["stream_ordinal"]),
            symbols=int(row["symbols"]),
            logical_bits=int(logical),
            payload=payload,
            source_digest=str(row["source_digest"]),
            profile_q=int(row["profile_q"]),
            decoder_scale=float(row["decoder_scale"]),
            role=str(row["role"]),
            group_rows=int(row["shape_rows"]),
            group_cols=int(row["shape_cols"]),
            owner_contributions=contributions,
        )
        identity = container_codec.StreamSpec(
            ordinal=int(row["stream_ordinal"]),
            symbols=int(row["symbols"]),
            logical_bits=int(row["baseline_logical_bits"]),
            payload=bytes(row["baseline_payload"]),
            source_digest=str(row["source_digest"]),
            profile_q=int(row["profile_q"]),
            decoder_scale=float(row["decoder_scale"]),
            role=str(row["role"]),
            group_rows=int(row["shape_rows"]),
            group_cols=int(row["shape_cols"]),
            owner_contributions=contributions,
        )
        stream_specs.append((spec, bytes(row["owner_set"])))
        identity_specs.append((identity, bytes(row["owner_set"])))
        payload_rows.append({
            "ordinal": int(row["stream_ordinal"]),
            "baseline_payload_bytes": int(row["baseline_payload_bytes"]),
            "new_payload_bytes": len(payload),
            "baseline_logical_bits": int(row["baseline_logical_bits"]),
            "new_logical_bits": int(logical),
        })
    regions = _regions_from_specs(container_codec, stream_specs, int(panel["experts"]))
    identity_regions = _regions_from_specs(container_codec, identity_specs, int(panel["experts"]))
    artifact_sha = str(panel["artifact"]["raw_sha256"])
    reconstruction_sha = str(panel["reconstruction"]["full_reconstruction_f64_sha256"])
    build_args = dict(
        common=common,
        semantic_codec=semantic_codec,
        model_packet=transmitted_model,
        semantic_packet=bytes(panel["semantic_packet"]),
        immutable_state=bytes(panel["immutable_state"]),
        weights=int(panel["weights"]),
        experts=int(panel["experts"]),
        baseline_object_bytes=int(panel["artifact"]["raw_bytes"]),
        audited_relative_mse=float(score["relative_mse"]),
        baseline_artifact_sha256=artifact_sha,
        reconstruction_sha256=reconstruction_sha,
        audit_binding_sha256=bindings.baseline_score_sha256,
        binding_hashes=bindings.container_hashes(),
        minimum_rate_numerator=container_codec.RATE_MIN_NUMERATOR,
        minimum_rate_denominator=container_codec.RATE_MIN_DENOMINATOR,
    )
    container, _predicted = container_codec.build_container(regions=regions, **build_args)
    parsed = container_codec.parse_container(common, semantic_codec, container)
    if not callable(authenticated_descriptor_source_builder):
        raise ValueError("external authenticated descriptor source builder required")
    route_source = authenticated_descriptor_source_builder(container)
    if not isinstance(route_source, container_codec.AuthenticatedDescriptorSource):
        raise ValueError("routed reader builder must return AuthenticatedDescriptorSource")
    try:
        routed_decoder = adapter.new_routed_decoder()
        metrics = container_codec.physical_metrics(
            common,
            semantic_codec,
            parsed,
            routed_descriptor_source=route_source,
            externally_authenticated_container_sha256=hashlib.sha256(container).hexdigest(),
            routed_decoder=routed_decoder,
        )
    finally:
        route_source.close()
    standalone = adapter.decode_new_container(parsed)
    if standalone["reconstruction"]["full_reconstruction_f64_sha256"] != reconstruction_sha:
        raise ValueError("candidate/current full reconstruction differs")
    rebuilt = container_codec.canonical_rebuild(common, semantic_codec, parsed)
    if rebuilt != container:
        raise ValueError("literal container does not canonically rebuild")
    identity_container, _ = container_codec.build_container(regions=identity_regions, **build_args)
    identity_parsed = container_codec.parse_container(common, semantic_codec, identity_container)
    # Identity framing is a byte-cost counterfactual, not a UWFA-coded object;
    # it therefore receives only non-authoritative memory-routing diagnostics.
    identity_metrics = container_codec.physical_metrics(common, semantic_codec, identity_parsed)
    payload_saved_bits = sum(
        8 * (int(row["baseline_payload_bytes"]) - int(row["new_payload_bytes"])) for row in payload_rows
    ) - 8 * len(transmitted_model)
    return {
        "container": container,
        "identity_framing_container": identity_container,
        "parsed_metrics": metrics,
        "identity_framing_metrics": identity_metrics,
        "absolute_saving_vs_bound_current_artifact_bpw": 8.0 * (int(panel["artifact"]["raw_bytes"]) - len(container)) / int(panel["weights"]),
        "incremental_same_framing_WFA_saving_bpw": 8.0 * (len(identity_container) - len(container)) / int(panel["weights"]),
        "raw_payload_minus_full_model_saving_bpw": payload_saved_bits / int(panel["weights"]),
        "payload_rows": payload_rows,
        "standalone_decode": standalone,
        "model_packet_sha256": hashlib.sha256(transmitted_model).hexdigest(),
        "container_sha256": hashlib.sha256(container).hexdigest(),
        "identity_framing_container_sha256": hashlib.sha256(identity_container).hexdigest(),
        "candidate": decoded_candidate.as_dict(),
        "posterior_diagnostic_handoff": container_codec.posterior_diagnostic_handoff(common, parsed),
        "all_adapted_values_deserialized_from_transmitted_model": True,
        "identical_reconstruction_proved_by_full_f64_digest": True,
    }


def prepare_panel(protocol: Any, adapter: Any, artifact_bytes: bytes) -> dict[str, Any]:
    panel = adapter.extract_from_current(artifact_bytes)
    if not isinstance(panel, dict):
        raise ValueError("adapter panel must be an object")
    protocol.validate_experts(panel.get("experts"))
    attach_semantic_owners(protocol, panel)
    protocol.panel_geometry(panel)
    return panel


def _result_without_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"container", "identity_framing_container"}}


def source_phase(
    *,
    common: Any,
    protocol: Any,
    container_codec: Any,
    semantic_codec: Any,
    adapter: Any,
    backend: Any,
    artifact_bytes: bytes,
    score_receipt_bytes: bytes,
    bindings: BoundEvidence,
    source_preflight: SourcePreflightEvidence,
    authenticated_descriptor_source_builder: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    panel = prepare_panel(protocol, adapter, artifact_bytes)
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha != panel["artifact"]["raw_sha256"] or len(artifact_bytes) != panel["artifact"]["raw_bytes"]:
        raise ValueError("held artifact descriptor binding")
    source_full_geometry = protocol.geometry_sha256(common, panel)
    source_structural_geometry = protocol.structural_geometry_sha256(common, panel)
    if source_full_geometry != bindings.source_full_geometry_sha256:
        raise ValueError("recomputed source full geometry/BoundEvidence mismatch")
    if source_structural_geometry != bindings.source_structural_geometry_sha256:
        raise ValueError("recomputed source structural geometry/BoundEvidence mismatch")
    validated_preflight = validate_source_preflight(common, protocol, source_preflight, bindings)
    score_record = common.strict_json_loads(score_receipt_bytes)
    score_sha = hashlib.sha256(score_receipt_bytes).hexdigest()
    if score_sha != bindings.baseline_score_sha256:
        raise ValueError("externally authorized score receipt digest")
    score = protocol.validate_score_receipt(
        score_record,
        artifact_sha256=artifact_sha,
        artifact_bytes=len(artifact_bytes),
        weights=int(panel["weights"]),
        reconstruction_sha256=str(panel["reconstruction"]["full_reconstruction_f64_sha256"]),
        original_source_panel_sha256=source_full_geometry,
        independent_decoder_source_sha256=bindings.universal_decoder_sha256,
    )
    runtime_projection = projected_updates(common, protocol, panel)
    if not runtime_projection["primary_exact_identity_estimable"]:
        return {
            "schema": "uwfa-sc-v6-source-phase-result",
            "status": "NO_PROMOTION_UNESTIMABLE_EXACT_IDENTITY_HOLDOUT",
            "source_full_geometry_sha256": source_full_geometry,
            "source_structural_geometry_sha256": source_structural_geometry,
            "source_preflight_receipt_sha256": validated_preflight["receipt_sha256"],
            "runtime_projection": runtime_projection,
            "controls_may_be_opened": False,
            "positive_promotion": False,
        }
    if not runtime_projection["passes_pre_fit_resource_budget"]:
        return {
            "schema": "uwfa-sc-v6-source-phase-result",
            "status": "ABORT_RESOURCE_BUDGET_BEFORE_BACKEND_PACK",
            "source_full_geometry_sha256": source_full_geometry,
            "source_structural_geometry_sha256": source_structural_geometry,
            "source_preflight_receipt_sha256": validated_preflight["receipt_sha256"],
            "runtime_projection": runtime_projection,
            "controls_may_be_opened": False,
            "positive_promotion": False,
        }
    if not runtime_projection["passes_pre_fit_runtime_budget"]:
        return {
            "schema": "uwfa-sc-v6-source-phase-result",
            "status": "ABORT_RUNTIME_BUDGET_BEFORE_FIT",
            "runtime_projection": runtime_projection,
            "controls_may_be_opened": False,
            "artifact_was_parsed_and_baseline_replayed": True,
        }
    cache = prepare_backend_cache(backend, panel)
    scientific = nested_holdout(common, protocol, backend, cache, panel)
    if scientific.get("estimable") is not True:
        raise ValueError("exact-identity primary unexpectedly became unestimable after projection")
    selected = common.candidate_bank()[int(scientific["final_topology_selected_from_nested_fold_votes"]["selector_ordinal"])]
    physical = final_container(
        common, container_codec, semantic_codec, adapter, backend, cache, panel,
        selected, score, bindings, authenticated_descriptor_source_builder,
    )
    metrics = physical["parsed_metrics"]
    physical_pass = bool(metrics["passes_rate_interval"] and metrics["passes_F_target"])
    cold_pass = bool(metrics["passes_cold_read_below_2x"])
    heldout_pass = bool(scientific["passes_heldout_gate"])
    integrity_pass = bool(
        physical["standalone_decode"]["all_payloads_canonically_reencoded"]
        and physical["identical_reconstruction_proved_by_full_f64_digest"]
        and physical["all_adapted_values_deserialized_from_transmitted_model"]
    )
    if not integrity_pass:
        status = "FAIL_EVIDENCE_INTEGRITY_SOURCE_STANDALONE_DECODE"
    elif not physical_pass:
        status = "HARD_KILL_PHYSICAL_RATE_OR_F"
    elif not cold_pass:
        status = "FAIL_STRICT_COLD_READ"
    elif not heldout_pass:
        status = "NO_PROMOTION_NESTED_HELDOUT"
    else:
        status = "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED"
    shuffles = None
    stricter_diagnostic = {
        "status": "NOT_RUN_SOURCE_DID_NOT_SURVIVE_PRIMARY_GATES",
        "positive_promotion": False,
    }
    if status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED":
        shuffles = survivor_shuffle_diagnostics(common, protocol, backend, panel)
        stricter_diagnostic = coordinate_disjoint_diagnostic(common, protocol, backend, cache, panel)
    elapsed = time.perf_counter() - started
    result = {
        "schema": "uwfa-sc-v6-source-phase-result",
        "status": status,
        "source_full_geometry_sha256": source_full_geometry,
        "source_structural_geometry_sha256": source_structural_geometry,
        "source_pipeline_sha256": bindings.pipeline_sha256,
        "source_artifact_sha256": artifact_sha,
        "score_receipt_sha256": score_sha,
        "source_preflight_receipt_sha256": validated_preflight["receipt_sha256"],
        "source_preflight_summary": {
            "source_snapshot_root_sha256": validated_preflight["source_snapshot_root_sha256"],
            "all150_status": validated_preflight["all150"]["status"],
            "representative_status": validated_preflight["representative"]["status"],
            "device_uuid": validated_preflight["independent_gpu_identity"]["device_uuid"],
            "pci_bus_id": validated_preflight["independent_gpu_identity"]["pci_bus_id"],
        },
        "runtime_projection": runtime_projection,
        "scientific_nested_holdout": scientific,
        "coordinate_disjoint_nonpromoting_diagnostic": stricter_diagnostic,
        "predeclared_shuffle_diagnostics": shuffles,
        "source_final": _result_without_payload(physical),
        "source_phase_elapsed_seconds": elapsed,
        "controls_may_be_opened": status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED",
        "physical_Qwen_failure_is_final_regardless_of_controls": True,
        "claim_boundary": "frozen selected-SC-decision recoder only; Qwen is an evaluation panel, not a universal performance proof",
        "requires_external_fresh_process_independent_result_audit": True,
    }
    result["_container"] = physical["container"]
    result["_identity_framing_container"] = physical["identity_framing_container"]
    result["_panel"] = panel
    result["_bindings"] = bindings
    return result


def controls_phase(
    *,
    common: Any,
    protocol: Any,
    container_codec: Any,
    semantic_codec: Any,
    adapter_factory: Any,
    backend_factory: Any,
    source_result: Mapping[str, Any],
    source_artifact_sha256: str,
    controls: Sequence[Mapping[str, Any]],
    authenticated_descriptor_source_builder: Any,
    moment_match_replayer: Any,
) -> dict[str, Any]:
    if source_result.get("controls_may_be_opened") is not True:
        raise ValueError("control payload access forbidden before all source gates")
    if len(controls) != len(common.CONTROL_SEEDS):
        raise ValueError("exact eight controls required")
    source_full_geometry = protocol.sha256_hex(source_result["source_full_geometry_sha256"], "source result full geometry")
    source_structural_geometry = protocol.sha256_hex(source_result["source_structural_geometry_sha256"], "source result structural geometry")
    source_pipeline = protocol.sha256_hex(source_result["source_pipeline_sha256"], "source result pipeline")
    source_panel = source_result.get("_panel")
    if not isinstance(source_panel, dict):
        raise ValueError("source panel unavailable for symmetric control moment replay")
    source_bindings = source_result.get("_bindings")
    if not isinstance(source_bindings, BoundEvidence):
        raise ValueError("authenticated source BoundEvidence unavailable")
    if (
        protocol.geometry_sha256(common, source_panel) != source_full_geometry
        or protocol.structural_geometry_sha256(common, source_panel) != source_structural_geometry
        or source_bindings.source_full_geometry_sha256 != source_full_geometry
        or source_bindings.source_structural_geometry_sha256 != source_structural_geometry
        or source_bindings.pipeline_sha256 != source_pipeline
    ):
        raise ValueError("authenticated source state geometry/pipeline mismatch")
    source_artifact = source_panel.get("artifact")
    if not isinstance(source_artifact, dict):
        raise ValueError("authenticated source artifact state absent")
    authenticated_source_artifact_sha256 = protocol.sha256_hex(
        source_artifact.get("raw_sha256"), "authenticated source artifact digest"
    )
    if protocol.sha256_hex(source_result.get("source_artifact_sha256"), "source result artifact digest") != authenticated_source_artifact_sha256:
        raise ValueError("source result/authenticated artifact digest mismatch")
    if protocol.sha256_hex(source_artifact_sha256, "caller source artifact digest") != authenticated_source_artifact_sha256:
        raise ValueError("caller source artifact digest differs from authenticated source state")
    symmetric_source_closure = source_bindings.symmetric_control_closure()
    if not callable(moment_match_replayer):
        raise ValueError("independently authenticated moment-match replayer required")
    # First parse, authenticate, replay, and geometry-check all controls.  No
    # candidate fit occurs until this complete loop succeeds.
    prepared = []
    for expected_seed, item in zip(common.CONTROL_SEEDS, controls, strict=True):
        if set(item) != {
            "artifact_bytes", "score_receipt_bytes", "binding_record", "bindings",
            "moment_match_receipt_bytes", "generator_source_bytes",
        }:
            raise ValueError("unknown/missing control bundle field")
        adapter = adapter_factory()
        panel = prepare_panel(protocol, adapter, bytes(item["artifact_bytes"]))
        full_geometry = protocol.geometry_sha256(common, panel)
        structural_geometry = protocol.structural_geometry_sha256(common, panel)
        if structural_geometry != source_structural_geometry:
            raise ValueError("Gaussian control structural geometry differs from source")
        artifact_sha = hashlib.sha256(item["artifact_bytes"]).hexdigest()
        binding_record = protocol.validate_control_binding(
            common,
            item["binding_record"],
            seed=int(expected_seed),
            source_artifact_sha256=authenticated_source_artifact_sha256,
            source_full_geometry_sha256=source_full_geometry,
            source_structural_geometry_sha256=source_structural_geometry,
            source_pipeline_sha256=source_pipeline,
            control_artifact_sha256=artifact_sha,
            control_full_geometry_sha256=full_geometry,
            control_structural_geometry_sha256=structural_geometry,
            symmetric_closure=symmetric_source_closure,
        )
        moment_bytes = bytes(item["moment_match_receipt_bytes"])
        generator_bytes = bytes(item["generator_source_bytes"])
        if hashlib.sha256(moment_bytes).hexdigest() != binding_record["moment_match_receipt_sha256"]:
            raise ValueError("control moment receipt byte binding")
        if hashlib.sha256(generator_bytes).hexdigest() != binding_record["generator_source_sha256"]:
            raise ValueError("control generator source byte binding")
        moment_replay = moment_match_replayer(
            source_panel=source_panel,
            control_panel=panel,
            seed=int(expected_seed),
            generator_source_bytes=generator_bytes,
            moment_match_receipt_bytes=moment_bytes,
        )
        if not isinstance(moment_replay, dict) or set(moment_replay) != {
            "status", "seed", "source_moments_sha256", "control_moments_sha256", "moment_match_receipt_sha256",
        }:
            raise ValueError("control moment replay result schema")
        if moment_replay["status"] != "PASS_RECOMPUTED_MOMENT_MATCH" or moment_replay["seed"] != int(expected_seed) or moment_replay["moment_match_receipt_sha256"] != binding_record["moment_match_receipt_sha256"]:
            raise ValueError("control moment replay did not pass exact binding")
        protocol.sha256_hex(moment_replay["source_moments_sha256"], "replayed source moments")
        protocol.sha256_hex(moment_replay["control_moments_sha256"], "replayed control moments")
        score = common.strict_json_loads(bytes(item["score_receipt_bytes"]))
        evidence = item["bindings"]
        if not isinstance(evidence, BoundEvidence):
            raise ValueError("control BoundEvidence type")
        score_receipt_sha = hashlib.sha256(bytes(item["score_receipt_bytes"])).hexdigest()
        if score_receipt_sha != evidence.baseline_score_sha256:
            raise ValueError("control score bytes/BoundEvidence digest mismatch")
        if (
            evidence.pipeline_sha256 != source_pipeline
            or evidence.source_full_geometry_sha256 != full_geometry
            or evidence.source_structural_geometry_sha256 != structural_geometry
            or evidence.symmetric_control_closure() != symmetric_source_closure
        ):
            raise ValueError("control symmetric closure/pipeline/geometry BoundEvidence mismatch")
        score = protocol.validate_score_receipt(
            score,
            artifact_sha256=artifact_sha,
            artifact_bytes=len(item["artifact_bytes"]),
            weights=int(panel["weights"]),
            reconstruction_sha256=str(panel["reconstruction"]["full_reconstruction_f64_sha256"]),
            original_source_panel_sha256=full_geometry,
            independent_decoder_source_sha256=evidence.universal_decoder_sha256,
        )
        prepared.append((expected_seed, adapter, panel, score, evidence, moment_replay))
    rows = []
    for seed, adapter, panel, score, bindings, moment_replay in prepared:
        backend = backend_factory()
        projection = projected_updates(common, protocol, panel)
        if not projection["primary_exact_identity_estimable"]:
            return {
                "status": "NO_PROMOTION_CONTROL_UNESTIMABLE_EXACT_IDENTITY_HOLDOUT",
                "seed": seed,
                "positive_promotion": False,
            }
        if not projection["passes_pre_fit_resource_budget"]:
            return {
                "status": "ABORT_CONTROL_RESOURCE_BUDGET_BEFORE_BACKEND_PACK",
                "seed": seed,
                "positive_promotion": False,
            }
        if not projection["passes_pre_fit_runtime_budget"]:
            return {
                "status": "ABORT_CONTROL_RUNTIME_BUDGET_BEFORE_FIT",
                "seed": seed,
                "positive_promotion": False,
            }
        cache = prepare_backend_cache(backend, panel)
        scientific = nested_holdout(common, protocol, backend, cache, panel)
        if scientific.get("estimable") is not True:
            raise ValueError("control exact-identity holdout unexpectedly unestimable")
        selected = common.candidate_bank()[int(scientific["final_topology_selected_from_nested_fold_votes"]["selector_ordinal"])]
        physical = final_container(
            common, container_codec, semantic_codec, adapter, backend, cache, panel,
            selected, score, bindings, authenticated_descriptor_source_builder,
        )
        rows.append({
            "seed": seed,
            "scientific_nested_holdout": scientific,
            "final": _result_without_payload(physical),
            "repeated_complete_150_cell_selection_fit_pack_decode": True,
            "recomputed_moment_match": moment_replay,
        })
    source_gain = float(source_result["source_final"]["absolute_saving_vs_bound_current_artifact_bpw"])
    strongest = max(float(row["final"]["absolute_saving_vs_bound_current_artifact_bpw"]) for row in rows)
    specificity = source_gain > strongest
    return {
        "schema": "uwfa-sc-v6-control-phase-result",
        "status": "PASS_MATCHED_NULL_SPECIFICITY_AWAITING_EXTERNAL_RESULT_AUDIT" if specificity else "NO_PROMOTION_GAUSSIAN_SPECIFICITY",
        "controls": rows,
        "specificity_statistic": "absolute physical saving bpw from each independently selected and packed full pipeline",
        "G_operational_source_bpw": source_gain,
        "strongest_matched_null_bpw": strongest,
        "source_minus_strongest_null_bpw": source_gain - strongest,
        "specificity_pass": specificity,
        "positive_promotion": False,
        "promotion_withheld_until_external_fresh_process_result_audit": True,
    }


def promotion_conjunction(
    *,
    physical: bool,
    cold: bool,
    heldout: bool,
    specificity: bool,
    standalone_decode: bool,
    integrity: bool,
    independent_result_audit: bool,
) -> bool:
    return all((physical, cold, heldout, specificity, standalone_decode, integrity, independent_result_audit))


def direct_main() -> int:
    # Do not parse arguments: a direct invocation must not touch any dynamic
    # input or output and imports no sibling module, NumPy, CuPy, or CUDA.
    print(DIRECT_LAUNCH_STATUS, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(direct_main())
