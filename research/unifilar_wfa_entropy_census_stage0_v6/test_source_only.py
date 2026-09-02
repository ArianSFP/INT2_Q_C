#!/usr/bin/env python3
"""Hostile source-only UWFA-SC v6 producer tests.

No test discovers or opens model, current-artifact, extracted-stream, or
Gaussian-control paths. All codec bytes are deterministic synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
import copy
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest import mock


PACKAGE = Path(__file__).absolute().parent


def load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("uwfa_v6_test_common", "uwfa_common.py")
protocol = load("uwfa_v6_test_protocol", "protocol.py")
semantic = load("uwfa_v6_test_semantic", "universal_adapter.py")
codec = load("uwfa_v6_test_container", "container_codec.py")
fixture = load("uwfa_v6_test_fixture", "fixture_portability.py")
strata = load("uwfa_v6_test_strata", "strata_sc_adapter.py")
stage = load("uwfa_v6_test_stage", "stage0_census.py")
cupy_backend = load("uwfa_v6_test_cupy", "cupy_backend.py")
dispatcher = load("uwfa_v6_test_dispatcher", "dispatcher_contract.py")
envelope = load("uwfa_v6_test_envelope", "result_envelope.py")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bindings() -> dict[str, str]:
    return {name: sha(("binding:" + name).encode("ascii")) for name in codec._HEADER_BINDINGS}


def bound_evidence(
    *,
    full: str = "aa" * 32,
    structural: str = "bb" * 32,
    pipeline: str = "cc" * 32,
    score_bytes: bytes = b"{}",
    source_root: str = "18" * 32,
    preflight_receipt: str = "19" * 32,
    decoder: str = "12" * 32,
) -> Any:
    return stage.BoundEvidence(
        baseline_plan_sha256="10" * 32,
        baseline_score_sha256=sha(score_bytes),
        universal_decoder_sha256=decoder,
        producer_manifest_sha256="13" * 32,
        audit_bootstrap_sha256="14" * 32,
        source_full_geometry_sha256=full,
        source_structural_geometry_sha256=structural,
        extraction_program_sha256="16" * 32,
        universal_adapter_sha256="17" * 32,
        pipeline_sha256=pipeline,
        source_snapshot_root_sha256=source_root,
        source_preflight_receipt_sha256=preflight_receipt,
    )


def _synthetic_resource_plan(symbols: int, streams: int) -> dict[str, Any]:
    payload = 4 * symbols
    descriptor = 16 * streams
    additional_host = payload + 64 * streams + stage.HOST_ALLOCATION_RESERVE_BYTES
    required_device = payload + descriptor + stage.MAX_AUXILIARY_DEVICE_BYTES + stage.VRAM_ALLOCATION_RESERVE_BYTES
    current_rss = 1
    free_vram = 32 * (1 << 30)
    return {
        "symbols": symbols,
        "streams": streams,
        "payload_host_and_device_bytes": payload,
        "root_descriptor_device_bytes": descriptor,
        "additional_host_bytes_including_reserve": additional_host,
        "device_required_bytes_including_aux_and_reserve": required_device,
        "current_process_tree_rss_bytes": current_rss,
        "current_process_hwm_bytes": 1,
        "projected_process_tree_rss_bytes": current_rss + additional_host,
        "current_free_vram_bytes": free_vram,
        "current_total_vram_bytes": free_vram,
        "host_cap_bytes": stage.MAX_HOST_BYTES,
        "vram_cap_bytes": stage.MAX_VRAM_BYTES,
        "passes": True,
        "checked_before_blob_concatenation_or_cupy_allocation": True,
    }


def _synthetic_environment(last_pack_resource_plan: dict[str, Any]) -> dict[str, Any]:
    device_uuid = "GPU-c06e0fe0-9836-2f98-8f10-0514d085f722"
    pci_bus_id = "00000000:16:00.0"
    statistics = {
        "h2d_bytes": 6,
        "h2d_payload_bytes": 1,
        "h2d_root_descriptor_bytes": 1,
        "h2d_subset_descriptor_bytes": 1,
        "h2d_launch_descriptor_bytes": 1,
        "h2d_model_table_bytes": 1,
        "h2d_kernel_scalar_bytes": 1,
        "d2h_bytes": 1,
        "d2d_descriptor_bytes": 0,
        "device_output_allocation_bytes": 1,
        "kernel_count": 2,
        "count_kernel_count": 1,
        "length_kernel_count": 1,
        "count_cell_symbol_updates": 1,
        "length_cell_symbol_updates": 1,
        "pack_calls": 1,
        "subset_calls": 1,
        "to_host_calls": 1,
        "telemetry_samples": 1,
        "peak_process_tree_rss_bytes": 1,
        "peak_process_hwm_bytes": 1,
        "incremental_peak_process_tree_rss_bytes": 1,
        "peak_vram_incremental_bytes": 1,
        "peak_default_pool_used_bytes": 1,
        "peak_default_pool_total_bytes": 1,
        "peak_pinned_pool_free_blocks": 1,
        "baseline_free_vram_bytes": 32 * (1 << 30),
        "total_vram_bytes": 32 * (1 << 30),
        "resource_preflight_calls": 1,
        "jit_compile_seconds": 0.0,
        "kernel_wall_seconds": 0.0,
        "last_pack_resource_plan": copy.deepcopy(last_pack_resource_plan),
    }
    sample = {
        "phase": "source-only-fixture",
        "process_tree_rss_bytes": 1,
        "process_hwm_bytes": 1,
        "free_vram_bytes": 1,
        "total_vram_bytes": 32 * (1 << 30),
        "default_pool_used_bytes": 0,
        "default_pool_total_bytes": 0,
        "pinned_pool_free_blocks": 0,
    }
    return {
        "cupy_version": "13.6.0",
        "cuda_runtime_version": 12090,
        "cuda_driver_version": 12090,
        "python_version": "3.12",
        "platform": "Linux",
        "device_id": 0,
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_uuid": device_uuid,
        "pci_bus_id": pci_bus_id,
        "compute_capability": [12, 0],
        "current_free_vram_bytes": 1,
        "total_vram_bytes": 32 * (1 << 30),
        "statistics": statistics,
        "telemetry_samples": [sample],
        "host_byteorder": "little",
        "explicit_device_synchronization_at_phase_boundaries_and_after_every_kernel": True,
        "fatal_telemetry_sampling": True,
        "transfer_formula": {
            "root_pack_h2d": "fixture",
            "subset_h2d": "fixture",
            "launch_descriptor_h2d": "fixture",
            "model_h2d": "fixture",
            "kernel_scalars_h2d": "fixture",
            "d2h": "fixture",
        },
    }


_VALID_PREFLIGHT_CACHE: tuple[dict[str, Any], dict[str, Any], dict[str, Any], str] | None = None


def _valid_source_preflight_payload() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    global _VALID_PREFLIGHT_CACHE
    if _VALID_PREFLIGHT_CACHE is not None:
        return copy.deepcopy(_VALID_PREFLIGHT_CACHE)
    source_root = "18" * 32
    device_uuid = "GPU-c06e0fe0-9836-2f98-8f10-0514d085f722"
    pci_bus_id = "00000000:16:00.0"
    independent = {
        "schema": "uwfa-sc-v6-independent-gpu-identity",
        "status": "PASS_INDEPENDENT_GPU_IDENTITY",
        "device_uuid": device_uuid,
        "pci_bus_id": pci_bus_id,
        "device_name": "NVIDIA GeForce RTX 5090",
        "provider": "nvidia-smi",
    }
    independent["identity_receipt_sha256"] = sha(common.canonical_json(independent))

    fixture_lengths, fixture_symbols, fixture_sha = stage._expected_all150_fixture(common)
    all150_resource = _synthetic_resource_plan(fixture_symbols, 4)
    cells = []
    for candidate in common.candidate_bank():
        ordinal = candidate.selector_ordinal
        logical_lengths = [100 + ordinal + index for index in range(4)]
        extreme_lengths = [200 + ordinal + index for index in range(4)]
        item = {
            **candidate.as_dict(),
            "count_tensor_sha256": sha(f"count:{ordinal}".encode("ascii")),
            "count_entries": candidate.states * common.CONTEXTS * 2,
            "count_total": fixture_symbols,
            "fitted_q016_sha256": sha(f"fit:{ordinal}".encode("ascii")),
            "logical_lengths": logical_lengths,
            "logical_length_count": 4,
            "logical_lengths_sha256": sha(b"".join(value.to_bytes(8, "little") for value in logical_lengths)),
            "frequency_extreme_logical_lengths": extreme_lengths,
            "frequency_extreme_logical_lengths_sha256": sha(b"".join(value.to_bytes(8, "little") for value in extreme_lengths)),
            "cpu_gpu_first_count_exact": True,
            "cpu_gpu_second_count_exact": True,
            "fitted_q016_cpu_gpu_exact": True,
            "cpu_gpu_first_logical_lengths_exact": True,
            "cpu_gpu_second_logical_lengths_exact": True,
            "frequency_extreme_cpu_gpu_exact": True,
            "repeated_gpu_run_exact": True,
        }
        item["cell_result_sha256"] = sha(common.canonical_json(item))
        cells.append(item)
    selectors = list(range(150))
    all150 = {
        "schema": "uwfa-sc-v6-all150-source-free-preflight",
        "source_snapshot_root_sha256": source_root,
        "status": "PASS_ALL_150_CPU_CUPY_EXACT_REPEATED",
        "cells": cells,
        "cell_count": 150,
        "streams": 4,
        "symbols_per_complete_bank": fixture_symbols,
        "fixture_stream_lengths": fixture_lengths,
        "fixture_sha256": fixture_sha,
        "resource_plan": all150_resource,
        "elapsed_seconds": 1.0,
        "environment": _synthetic_environment(all150_resource),
        "frequency_extremes_tested": [1, 65535],
        "reset_boundaries_through_4096_tested": True,
        "candidate_selector_sha256": sha(common.canonical_json(selectors)),
        "cell_results_sha256": sha(common.canonical_json(cells)),
    }

    representative_lengths, representative_symbols, representative_sha = stage._expected_representative_fixture(common)
    representative_resource = _synthetic_resource_plan(representative_symbols, 15)
    candidate_scores = [
        {**candidate.as_dict(), "validation_charged_bits": 1_000 + candidate.selector_ordinal}
        for candidate in common.candidate_bank()
    ]
    train_streams = list(range(4, 12)) + [13, 14]
    validation_streams = [2, 3]
    test_streams = [0, 1, 12]
    development_streams = list(range(2, 12)) + [13, 14]
    train_symbols = sum(representative_lengths[index] for index in train_streams)
    validation_symbols = sum(representative_lengths[index] for index in validation_streams)
    development_symbols = sum(representative_lengths[index] for index in development_streams)
    test_symbols = sum(representative_lengths[index] for index in test_streams)
    one_fold_updates = 150 * (train_symbols + validation_symbols) + development_symbols + test_symbols
    measured_updates = one_fold_updates + 2 * representative_symbols
    projected_updates = 13 * (6 * one_fold_updates + 2 * representative_symbols)
    measured_seconds = 1.0
    throughput = measured_updates / measured_seconds
    conservative = throughput * 0.5
    projected_seconds = projected_updates / conservative
    test_lengths = [300 + index for index in range(3)]
    full_lengths = [400 + index for index in range(15)]
    triplets = [sha(f"triplet:{index}".encode("ascii")) for index in range(15)]
    container_sha = sha(b"representative-container")
    outer_fold = {
        "train_streams": train_streams,
        "validation_streams": validation_streams,
        "test_streams": test_streams,
        "all_150_candidates_fit_and_scored": True,
        "candidate_count": 150,
        "candidate_scores": candidate_scores,
        "candidate_scores_sha256": sha(common.canonical_json(candidate_scores)),
        "winner": common.candidate_bank()[0].as_dict(),
        "validation_charged_bits": 1_000,
        "test_logical_lengths": test_lengths,
        "test_logical_lengths_sha256": sha(b"".join(value.to_bytes(8, "little") for value in test_lengths)),
        "final_full_panel_logical_lengths": full_lengths,
        "final_full_panel_logical_lengths_sha256": sha(b"".join(value.to_bytes(8, "little") for value in full_lengths)),
        "winner_development_q016_sha256": sha(b"development-q016"),
        "final_full_panel_q016_sha256": sha(b"full-q016"),
        "serialized_model_sha256": sha(b"serialized-model"),
        "winner_refit_on_complete_development": True,
        "final_full_panel_fit": True,
        "literal_container_parse_decode_reencode_rebuild": True,
        "container_sha256": container_sha,
        "canonical_rebuild_sha256": container_sha,
        "decoded_reencoded_stream_count": 15,
        "decoded_triplet_sha256s": triplets,
        "decoded_triplet_commitment_sha256": sha(common.canonical_json(triplets)),
    }
    phase_delta = {
        "h2d_bytes": 6,
        "h2d_payload_bytes": 1,
        "h2d_root_descriptor_bytes": 1,
        "h2d_subset_descriptor_bytes": 1,
        "h2d_launch_descriptor_bytes": 1,
        "h2d_model_table_bytes": 1,
        "h2d_kernel_scalar_bytes": 1,
        "d2h_bytes": 1,
        "d2d_descriptor_bytes": 0,
        "device_output_allocation_bytes": 1,
        "kernel_count": 2,
        "count_kernel_count": 1,
        "length_kernel_count": 1,
        "count_cell_symbol_updates": 1,
        "length_cell_symbol_updates": 1,
        "pack_calls": 1,
        "subset_calls": 1,
        "to_host_calls": 1,
    }
    representative = {
        "schema": "uwfa-sc-v6-representative-source-free-preflight",
        "source_snapshot_root_sha256": source_root,
        "status": "PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD",
        "fixture": {
            "streams": 15,
            "semantic_owners": 6,
            "private_streams": 12,
            "shared_tail_streams": 3,
            "stream_lengths": representative_lengths,
            "symbols": representative_symbols,
            "source": "public frozen block geometry proxy; no observed model symbols",
            "fixture_sha256": representative_sha,
            "resource_plan": representative_resource,
        },
        "outer_fold": outer_fold,
        "runtime_projection": {
            "measured_seconds": measured_seconds,
            "warmup_seconds_excluded": 1.0,
            "measured_cell_symbol_updates": measured_updates,
            "measured_updates_per_second": throughput,
            "conservative_updates_per_second": conservative,
            "projection_formula": "13 pipelines * (6 complete outer folds + final fit/score), at 50% measured throughput",
            "pipelines_source_plus_four_shuffles_plus_eight_controls": 13,
            "projected_cell_symbol_updates": projected_updates,
            "projected_wall_seconds": projected_seconds,
            "budget_seconds": stage.MAX_PROJECTED_WALL_SECONDS,
            "passes": True,
        },
        "telemetry": _synthetic_environment(representative_resource),
        "measured_phase_statistics_delta": phase_delta,
        "model_h2d_bytes_nonzero": True,
        "d2h_bytes_nonzero": True,
        "peak_host_ram_recorded": True,
        "peak_vram_recorded": True,
    }
    preflight_record = {
        "schema": "uwfa-sc-v6-bound-source-preflight",
        "source_snapshot_root_sha256": source_root,
        "all150": all150,
        "representative": representative,
        "independent_gpu_identity": independent,
    }
    receipt = sha(common.canonical_json(preflight_record))
    _VALID_PREFLIGHT_CACHE = (all150, representative, independent, receipt)
    return copy.deepcopy(_VALID_PREFLIGHT_CACHE)


def valid_source_preflight(*, full: str = "aa" * 32, structural: str = "bb" * 32, score_bytes: bytes = b"{}") -> tuple[Any, Any]:
    all150, representative, independent, receipt = _valid_source_preflight_payload()
    source_root = "18" * 32
    evidence = bound_evidence(
        full=full, structural=structural, score_bytes=score_bytes,
        source_root=source_root, preflight_receipt=receipt,
    )
    return stage.SourcePreflightEvidence(all150, representative, independent, receipt), evidence


def reseal_source_preflight(preflight: Any, evidence: Any) -> tuple[Any, Any]:
    record = {
        "schema": "uwfa-sc-v6-bound-source-preflight",
        "source_snapshot_root_sha256": evidence.source_snapshot_root_sha256,
        "all150": dict(preflight.all150),
        "representative": dict(preflight.representative),
        "independent_gpu_identity": dict(preflight.independent_gpu_identity),
    }
    receipt = sha(common.canonical_json(record))
    return (
        stage.SourcePreflightEvidence(
            dict(preflight.all150),
            dict(preflight.representative),
            dict(preflight.independent_gpu_identity),
            receipt,
        ),
        replace(evidence, source_preflight_receipt_sha256=receipt),
    )


def build_fixture(experts: int = 2, hidden: int = 3, intermediate: int = 5, *, shared: bool = True) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    groups = ()
    if shared and experts >= 2:
        groups = (("gate", ((0, 2), (experts - 1, 3))),)
    source = fixture.make_fixture(common, codec, semantic, experts=experts, hidden=hidden, intermediate=intermediate, shared_groups=groups)
    raw, metrics = codec.build_container(
        common,
        semantic,
        model_packet=source["model_packet"],
        semantic_packet=source["semantic_packet"],
        immutable_state=b"source-only-v6-fixture",
        regions=source["regions"],
        weights=source["weights"],
        experts=experts,
        baseline_object_bytes=10_000_000,
        audited_relative_mse=0.025,
        baseline_artifact_sha256=sha(b"baseline"),
        reconstruction_sha256=source["reconstruction_sha256"],
        audit_binding_sha256=sha(b"audit"),
        binding_hashes=bindings(),
    )
    return raw, metrics, source


def reseal_header(raw: bytes, mutate: Any) -> bytes:
    packet = bytearray(raw)
    header = bytearray(packet[:codec.HEADER_BYTES])
    mutate(header)
    header[codec._HEADER_SEAL_BEGIN:codec._HEADER_SEAL_END] = bytes(32)
    struct.pack_into("<I", header, codec._HEADER_CRC_OFFSET, 0)
    header[codec._HEADER_SEAL_BEGIN:codec._HEADER_SEAL_END] = hashlib.sha256(header).digest()
    struct.pack_into("<I", header, codec._HEADER_CRC_OFFSET, zlib.crc32(header) & 0xFFFFFFFF)
    packet[:codec.HEADER_BYTES] = header
    return bytes(packet)


def reseal_body_and_header(raw: bytes, mutate: Any, *, directory_changed: bool = True) -> bytes:
    packet = bytearray(raw)
    header = bytearray(packet[:codec.HEADER_BYTES])
    directory_offset = struct.unpack_from("<Q", header, 112)[0]
    directory_bytes = struct.unpack_from("<Q", header, 120)[0]
    mutate(packet, header, directory_offset, directory_bytes)
    if directory_changed:
        header[352:384] = hashlib.sha256(packet[directory_offset:directory_offset + directory_bytes]).digest()
    header[384:416] = hashlib.sha256(packet[codec.HEADER_BYTES:]).digest()
    header[codec._HEADER_SEAL_BEGIN:codec._HEADER_SEAL_END] = bytes(32)
    struct.pack_into("<I", header, codec._HEADER_CRC_OFFSET, 0)
    header[codec._HEADER_SEAL_BEGIN:codec._HEADER_SEAL_END] = hashlib.sha256(header).digest()
    struct.pack_into("<I", header, codec._HEADER_CRC_OFFSET, zlib.crc32(header) & 0xFFFFFFFF)
    packet[:codec.HEADER_BYTES] = header
    return bytes(packet)


def reseal_directory_row(packet: bytearray, directory_offset: int, ordinal: int) -> None:
    begin = directory_offset + ordinal * codec.DIRECTORY_RECORD_BYTES
    packet[begin + 248:begin + 256] = hashlib.sha256(packet[begin:begin + 248]).digest()[:8]


def reseal_frame_header(packet: bytearray, frame_offset: int) -> None:
    packet[frame_offset + 224:frame_offset + 256] = hashlib.sha256(
        packet[frame_offset:frame_offset + 224]
    ).digest()


def reseal_region(packet: bytearray, region_offset: int) -> None:
    header = bytearray(packet[region_offset:region_offset + codec.REGION_HEADER_BYTES])
    frame_area_bytes = struct.unpack_from("<Q", header, 40)[0]
    area_begin = region_offset + codec.REGION_HEADER_BYTES
    header[80:112] = hashlib.sha256(packet[area_begin:area_begin + frame_area_bytes]).digest()
    header[112:144] = hashlib.sha256(header[:112]).digest()
    packet[region_offset:region_offset + codec.REGION_HEADER_BYTES] = header


class FrozenMathTests(unittest.TestCase):
    def test_threshold_bank_and_reset_law(self) -> None:
        self.assertAlmostEqual(common.STANDALONE_REQUIRED_SAVING_BPW, 0.15288996696291447, places=15)
        bank = common.candidate_bank()
        self.assertEqual(len(bank), 150)
        self.assertEqual([row.selector_ordinal for row in bank], list(range(150)))
        for candidate in bank:
            self.assertEqual(common.transition(candidate, 0, 0, 0, 0), common.transition(candidate, 0, 0, 0, 0))

    def test_exhaustive_small_arithmetic_and_model_tamper(self) -> None:
        candidate = common.Candidate("xor_sketch", 4, 32)
        frequencies = [1 if index & 1 else 65535 for index in range(common.model_frequency_count(candidate))]
        for length in range(1, 8):
            levels = [index % common.LEVELS for index in range(length)]
            base = [1 if index & 1 else 65535 for index in range(length)]
            for bits in itertools.product((0, 1), repeat=length):
                payload, logical = common.encode_unifilar_stream(list(bits), levels, base, candidate, frequencies)
                self.assertEqual(common.decode_unifilar_stream(payload, logical, levels, base, candidate, frequencies), list(bits))
        packet = common.serialize_model(candidate, frequencies)
        self.assertEqual(common.serialize_model(*common.deserialize_model(packet)), packet)
        corrupt = bytearray(packet)
        corrupt[-1] ^= 1
        with self.assertRaises(Exception):
            common.deserialize_model(bytes(corrupt))

    def test_bounded_json_size_and_depth(self) -> None:
        with self.assertRaises(Exception):
            common.strict_json_loads(b" " * (16 * (1 << 20) + 1))
        deeply_nested = ("[" * 66 + "0" + "]" * 66).encode("ascii")
        with self.assertRaises(Exception):
            common.strict_json_loads(deeply_nested)


class OwnerAbiTests(unittest.TestCase):
    def test_positive_boundaries_and_high_bits(self) -> None:
        boundaries = (1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 255, 256)
        probes = (0, 7, 8, 31, 32, 127, 128, 254, 255)
        for experts in boundaries:
            owners = sorted({value for value in probes if value < experts} | {experts - 1})
            encoded = protocol.owner_set_from_ordinals(experts, owners)
            self.assertEqual(len(encoded), 32)
            self.assertEqual(protocol.owner_ordinals(encoded, experts), tuple(owners))
            self.assertEqual(codec.owner_ordinals(encoded, experts), tuple(owners))
        self.assertTrue(protocol.owner_set_from_ordinals(128, [127])[15] & 0x80)
        self.assertTrue(protocol.owner_set_from_ordinals(256, [255])[31] & 0x80)

    def test_zero_high_unused_duplicate_and_oversized_experts_reject(self) -> None:
        with self.assertRaises(ValueError):
            protocol.owner_ordinals(bytes(32), 128)
        invalid = bytearray(32)
        invalid[0] = 1
        invalid[16] = 1
        with self.assertRaises(ValueError):
            protocol.owner_ordinals(bytes(invalid), 128)
        with self.assertRaises(ValueError):
            protocol.owner_set_from_ordinals(8, [1, 1])
        for experts in (0, 257, 4096, 4097):
            started = time.perf_counter()
            with self.assertRaises(ValueError):
                protocol.validate_experts(experts)
            self.assertLess(time.perf_counter() - started, 0.2)

    def test_panel_conserves_shapes_contributions_and_universe(self) -> None:
        owner0 = protocol.owner_set_from_ordinals(2, [0]).hex()
        owner1 = protocol.owner_set_from_ordinals(2, [1]).hex()
        panel = {
            "weights": 12,
            "experts": 2,
            "semantic_identities": [(0, 0), (0, 1)],
            "expert_shapes": [
                {"expert": 0, "hidden": 1, "intermediate": 2},
                {"expert": 1, "hidden": 1, "intermediate": 2},
            ],
            "artifact": {"raw_bytes": 100},
            "immutable_state": b"",
            "streams": [
                {"stream_ordinal": 0, "owner_set_hex": owner0, "weight_charge": 6, "shape_rows": 2, "shape_cols": 3, "role": "gate", "owner_contributions": ({"expert": 0, "role": "gate", "source_offset": 0, "weight_count": 6},), "symbols": 2, "logn": 1, "profile_q": 0, "baseline_payload_bytes": 1, "baseline_logical_bits": 1},
                {"stream_ordinal": 1, "owner_set_hex": owner1, "weight_charge": 6, "shape_rows": 2, "shape_cols": 3, "role": "gate", "owner_contributions": ({"expert": 1, "role": "gate", "source_offset": 0, "weight_count": 6},), "symbols": 2, "logn": 1, "profile_q": 0, "baseline_payload_bytes": 1, "baseline_logical_bits": 1},
            ],
        }
        self.assertEqual(protocol.panel_geometry(panel)["weights"], 12)
        panel["streams"][1]["owner_set_hex"] = owner0
        with self.assertRaises(ValueError):
            protocol.panel_geometry(panel)

    def test_untrusted_expert_count_rejects_before_semantics_or_allocation(self) -> None:
        raw, _metrics, _source = build_fixture()

        class UntouchedSemanticCodec:
            def __init__(self) -> None:
                self.called = False

            def parse_semantic_packet(self, _packet: bytes) -> Any:
                self.called = True
                raise AssertionError("semantic parser reached after invalid E")

        for experts in (0, 257, 4096, 4097, 0xFFFFFFFF):
            hostile = reseal_header(
                raw,
                lambda header, experts=experts: struct.pack_into("<I", header, 28, experts),
            )
            untouched = UntouchedSemanticCodec()
            with self.subTest(experts=experts):
                with self.assertRaises(ValueError):
                    codec.parse_container(common, untouched, hostile)
                self.assertFalse(untouched.called)


class LiteralContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw, cls.metrics, cls.source = build_fixture()
        cls.parsed = codec.parse_container(common, semantic, cls.raw)

    def test_roundtrip_decode_reencode_integer_rebuild_and_partition(self) -> None:
        self.assertEqual(codec.canonical_rebuild(common, semantic, self.parsed), self.raw)
        self.assertEqual(fixture.decode_and_reencode(common, self.parsed)["decoded_streams"], len(self.parsed["directory"]))
        cursor = 0
        for entry in self.parsed["byte_ledger"]:
            self.assertEqual(entry["begin"], cursor)
            cursor = entry["end"]
        self.assertEqual(cursor, len(self.raw))
        self.assertTrue(self.metrics["complete_byte_partition_exact"])
        self.assertEqual(self.metrics["ownership_allocated_bytes_sum"]["numerator"], len(self.raw))
        self.assertEqual(self.metrics["ownership_allocated_bytes_sum"]["denominator"], 1)
        handoff = codec.posterior_diagnostic_handoff(common, self.parsed)
        self.assertEqual(handoff["literal_container_sha256"], sha(self.raw))
        self.assertEqual(handoff["source_artifact_sha256"], sha(b"baseline"))
        self.assertEqual(handoff["full_reconstruction_f64_sha256"], self.source["reconstruction_sha256"])
        self.assertEqual(handoff["stream_count"], len(self.parsed["directory"]))
        self.assertTrue(handoff["requires_literal_redecode"])
        self.assertFalse(handoff["contains_posterior_or_MMSE_result"])
        self.assertIn("decoded_sc_decision_triplet_commitment_sha256", handoff)
        self.assertNotIn("decoded_symbol_bits_sha256", json.dumps(handoff, sort_keys=True))
        first = handoff["stream_decision_triplet_commitments"][0]
        self.assertEqual(
            first["decoded_selected_decision_triplet_sha256"],
            self.parsed["directory"][0]["source_digest"],
        )

    def test_triplet_commitment_semantics_match_strata_and_are_not_bit_only(self) -> None:
        bits = bytes((0, 1, 1, 0))
        levels = bytes((0, 1, 2, 3))
        base = struct.pack("<4H", 1, 32768, 65535, 17)
        expected = common.selected_decision_triplet_sha256(bits, levels, base)
        self.assertEqual(strata._stream_digest(bits, levels, base), expected)
        self.assertNotEqual(sha(bits), expected)

    def test_fresh_routed_read_and_dual_denominator(self) -> None:
        trace = codec.instrument_expert_pages(common, semantic, self.raw, 0)
        self.assertTrue(trace["routed_read_ranges"])
        self.assertNotIn((0, len(self.raw)), trace["routed_read_ranges"])
        self.assertEqual(trace["installation_authentication_scan_bytes"], len(self.raw))
        self.assertFalse(self.metrics["routed_io_authoritative_descriptor_backed"])
        self.assertFalse(self.metrics["passes_cold_read_below_2x"])
        with tempfile.TemporaryFile() as handle:
            handle.write(self.raw)
            handle.flush()
            os.fsync(handle.fileno())
            descriptor_parsed = codec.parse_container_descriptor(common, semantic, handle.fileno())
            self.assertEqual(codec.canonical_rebuild(common, semantic, descriptor_parsed), self.raw)
            source = codec.AuthenticatedDescriptorSource(handle.fileno(), sha(self.raw))
            try:
                measured = codec.physical_metrics(
                    common,
                    semantic,
                    self.parsed,
                    routed_descriptor_source=source,
                    externally_authenticated_container_sha256=sha(self.raw),
                    routed_decoder=fixture.FixtureRoutedDecoder(common),
                )
            finally:
                source.close()
        self.assertTrue(measured["routed_io_authoritative_descriptor_backed"])
        installation = measured["installation_authentication_reported_separately"]
        self.assertEqual(installation["scan_bytes"], len(self.raw))
        self.assertEqual(installation["touched_page_bytes"], len(self.raw))
        self.assertEqual(installation["read_ranges"][0][0], 0)
        self.assertEqual(installation["read_ranges"][-1][1], len(self.raw))
        self.assertTrue(installation["excluded_from_per_expert_cold_numerator"])
        row = measured["experts"][0]
        self.assertGreaterEqual(row["cold_amplification_nonpadding"]["float"], row["cold_amplification_total_physical"]["float"])

    def test_explicit_modeled_symbol_density_and_repeated_read_aggregates(self) -> None:
        expected_unique_symbols = sum(int(row["symbols"]) for row in self.parsed["directory"])
        density = self.metrics["modeled_symbol_density"]
        self.assertEqual(density["unique_directory_modeled_symbols"], expected_unique_symbols)
        self.assertEqual(density["source_weights"], self.parsed["weights"])
        expected_density = Fraction(expected_unique_symbols, int(self.parsed["weights"]))
        self.assertEqual(density["unique_directory_modeled_symbols_per_source_weight"]["numerator"], expected_density.numerator)
        self.assertEqual(density["unique_directory_modeled_symbols_per_source_weight"]["denominator"], expected_density.denominator)

        requested_sum = 0
        unique_requested_sum = 0
        overlap_sum = 0
        request_count_sum = 0
        touched_page_sum = 0
        routed_symbol_sum = 0
        for expert_row in self.metrics["experts"]:
            expert = int(expert_row["expert_ordinal"])
            ranges = expert_row["instrumented_routed_read_ranges"]
            expected_requested = sum(int(end) - int(begin) for begin, end in ranges)
            self.assertEqual(expert_row["instrumented_routed_read_request_count"], len(ranges))
            self.assertEqual(expert_row["instrumented_routed_requested_bytes_with_repetition"], expected_requested)
            self.assertEqual(
                expert_row["instrumented_routed_requested_bytes_with_repetition"],
                expert_row["instrumented_routed_unique_requested_bytes"]
                + expert_row["instrumented_routed_overlap_bytes_requested_again"],
            )
            selected = [row for row in self.parsed["directory"] if expert in row["owners"]]
            expected_routed_symbols = sum(int(row["symbols"]) for row in selected)
            expected_expert_weights = sum(
                int(contribution["weight_count"])
                for row in selected
                for contribution in row["owner_contributions"]
                if int(contribution["expert"]) == expert
            )
            self.assertEqual(expert_row["routed_modeled_symbols"], expected_routed_symbols)
            self.assertEqual(expert_row["expert_source_weights"], expected_expert_weights)
            per_expert_density = Fraction(expected_routed_symbols, expected_expert_weights)
            self.assertEqual(expert_row["routed_modeled_symbols_per_source_weight"]["exact"], f"{per_expert_density.numerator}/{per_expert_density.denominator}")
            requested_sum += expected_requested
            unique_requested_sum += expert_row["instrumented_routed_unique_requested_bytes"]
            overlap_sum += expert_row["instrumented_routed_overlap_bytes_requested_again"]
            request_count_sum += len(ranges)
            touched_page_sum += expert_row["touched_page_bytes"]
            routed_symbol_sum += expected_routed_symbols

        aggregate = self.metrics["routed_read_request_aggregates"]
        self.assertEqual(aggregate["requested_bytes_with_repetition_sum_across_experts"], requested_sum)
        self.assertEqual(aggregate["unique_requested_bytes_sum_across_experts"], unique_requested_sum)
        self.assertEqual(aggregate["overlap_bytes_requested_again_sum_across_experts"], overlap_sum)
        self.assertEqual(aggregate["read_request_count_sum_across_experts"], request_count_sum)
        self.assertEqual(aggregate["unique_touched_page_bytes_sum_across_experts"], touched_page_sum)
        self.assertTrue(aggregate["frozen_cold_gate_uses_unique_touched_page_bytes_only"])
        self.assertEqual(density["routed_modeled_symbols_sum_across_experts"], routed_symbol_sum)
        self.assertEqual(density["shared_stream_symbol_reuse_across_expert_routes"], routed_symbol_sum - expected_unique_symbols)

        maximum = Fraction(
            self.metrics["maximum_strict_cold_read_amplification"]["numerator"],
            self.metrics["maximum_strict_cold_read_amplification"]["denominator"],
        )
        self.assertEqual(
            self.metrics["passes_cold_read_below_2x"],
            self.metrics["routed_io_authoritative_descriptor_backed"] and maximum < 2,
        )

        with tempfile.TemporaryFile() as handle:
            handle.write(self.raw)
            handle.flush()
            os.fsync(handle.fileno())
            source = codec.AuthenticatedDescriptorSource(handle.fileno(), sha(self.raw))
            try:
                measured = codec.physical_metrics(
                    common, semantic, self.parsed,
                    routed_descriptor_source=source,
                    externally_authenticated_container_sha256=sha(self.raw),
                    routed_decoder=fixture.FixtureRoutedDecoder(common),
                )
            finally:
                source.close()
        installation = measured["installation_authentication_reported_separately"]
        self.assertEqual(
            installation["requested_bytes_with_repetition"],
            sum(int(end) - int(begin) for begin, end in installation["read_ranges"]),
        )
        self.assertEqual(installation["read_request_count"], len(installation["read_ranges"]))
        self.assertTrue(measured["routed_read_request_aggregates"]["frozen_cold_gate_uses_unique_touched_page_bytes_only"])

    def test_owner_copy_directory_mutation_rejects_after_reseal(self) -> None:
        def mutate(packet: bytearray, _header: bytearray, directory_offset: int, _directory_bytes: int) -> None:
            packet[directory_offset + 120] ^= 0x02
            reseal_directory_row(packet, directory_offset, 0)
        hostile = reseal_body_and_header(self.raw, mutate)
        with self.assertRaises(ValueError):
            codec.parse_container(common, semantic, hostile)

    def test_semantic_scalar_and_range_mutations_reject(self) -> None:
        mutations = [
            (24, struct.pack("<Q", 0)),
            (32, struct.pack("<Q", 0)),
            (32, struct.pack("<Q", (1 << 56) + 1)),
            (40, struct.pack("<Q", (1 << 40) - 1)),
            (112, struct.pack("<d", math.nan)),
        ]
        for offset, payload in mutations:
            def mutate(packet: bytearray, _header: bytearray, directory_offset: int, _directory_bytes: int, offset: int = offset, payload: bytes = payload) -> None:
                packet[directory_offset + offset:directory_offset + offset + len(payload)] = payload
                reseal_directory_row(packet, directory_offset, 0)
            hostile = reseal_body_and_header(self.raw, mutate)
            with self.subTest(offset=offset, payload=payload.hex()):
                with self.assertRaises(ValueError):
                    codec.parse_container(common, semantic, hostile)

    def test_header_bounds_reject_before_dependent_work(self) -> None:
        cases = ((28, "<I", 257), (28, "<I", 4096), (28, "<I", 4097), (32, "<I", 0xFFFFFFFF), (40, "<I", 0xFFFFFFFF), (104, "<Q", 1 << 40), (36, "<H", 16))
        for offset, fmt, value in cases:
            hostile = reseal_header(self.raw, lambda header, offset=offset, fmt=fmt, value=value: struct.pack_into(fmt, header, offset, value))
            started = time.perf_counter()
            with self.subTest(offset=offset, value=value):
                with self.assertRaises(ValueError):
                    codec.parse_container(common, semantic, hostile)
            self.assertLess(time.perf_counter() - started, 0.5)

    def test_frozen_rate_flags_and_streaming_chunk_bound(self) -> None:
        for offset, fmt, value in ((16, "<I", 1), (144, "<Q", 10_000), (152, "<Q", 1)):
            hostile = reseal_header(
                self.raw,
                lambda header, offset=offset, fmt=fmt, value=value: struct.pack_into(fmt, header, offset, value),
            )
            with self.subTest(offset=offset):
                with self.assertRaises(ValueError):
                    codec.parse_container(common, semantic, hostile)

        class SpyReader:
            def __init__(self) -> None:
                self.maximum = 0

            def read(self, _begin: int, length: int) -> bytes:
                self.maximum = max(self.maximum, length)
                return bytes(length)

        spy = SpyReader()
        codec._hash_reader_range(spy, 0, 5 * (1 << 20) + 17)
        self.assertLessEqual(spy.maximum, codec.MAX_READER_CHUNK_BYTES)

    def test_resealed_padding_and_routed_source_digest_tamper_reject(self) -> None:
        padding = next(entry for entry in self.parsed["byte_ledger"] if entry["kind"] == "owner_region_rate_padding")
        hostile_padding = reseal_body_and_header(
            self.raw,
            lambda packet, _header, _directory_offset, _directory_bytes: packet.__setitem__(int(padding["begin"]), 1),
            directory_changed=False,
        )
        with self.assertRaises(ValueError):
            codec.parse_container(common, semantic, hostile_padding)

        row = self.parsed["directory"][0]
        def mutate_digest(packet: bytearray, _header: bytearray, directory_offset: int, _directory_bytes: int) -> None:
            replacement = bytes.fromhex("99" * 32)
            packet[directory_offset + 152:directory_offset + 184] = replacement
            reseal_directory_row(packet, directory_offset, 0)
            frame_offset = int(row["frame_offset"])
            packet[frame_offset + 128:frame_offset + 160] = replacement
            reseal_frame_header(packet, frame_offset)
            reseal_region(packet, int(row["region_offset"]))
        hostile_digest = reseal_body_and_header(self.raw, mutate_digest)
        parsed = codec.parse_container(common, semantic, hostile_digest)
        with tempfile.TemporaryFile() as handle:
            handle.write(hostile_digest)
            handle.flush()
            os.fsync(handle.fileno())
            source = codec.AuthenticatedDescriptorSource(handle.fileno(), sha(hostile_digest))
            try:
                with self.assertRaises(ValueError):
                    codec.physical_metrics(
                        common, semantic, parsed,
                        routed_descriptor_source=source,
                        externally_authenticated_container_sha256=sha(hostile_digest),
                        routed_decoder=fixture.FixtureRoutedDecoder(common),
                    )
            finally:
                source.close()

    def test_routed_decode_never_reads_or_hashes_an_unselected_corrupt_frame(self) -> None:
        unselected = next(row for row in self.parsed["directory"] if tuple(row["owners"]) == (1,))
        corrupt_offset = int(unselected["payload_offset"])
        packet = bytearray(self.raw)
        packet[corrupt_offset] ^= 1
        hostile = bytes(packet)
        with self.assertRaises(ValueError):
            codec.parse_container(common, semantic, hostile)
        with tempfile.TemporaryFile() as handle:
            handle.write(hostile)
            handle.flush()
            os.fsync(handle.fileno())
            source = codec.AuthenticatedDescriptorSource(handle.fileno(), sha(hostile))
            reader = source.fresh_reader()
            try:
                route = codec.routed_read_expert(
                    common, semantic, reader,
                    file_size=reader.size, expert=0,
                    externally_authenticated_container_sha256=sha(hostile),
                    decode_routed_expert=fixture.FixtureRoutedDecoder(common).decode_expert,
                )
                reader.verify_stable()
                source.verify_stable()
            finally:
                reader.close()
                source.close()
        self.assertTrue(route["causal_decode_reencode_reconstruction"]["all_payloads_canonically_reencoded"])
        self.assertTrue(all(not (begin <= corrupt_offset < end) for begin, end in route["routed_read_ranges"]))
        self.assertNotIn(corrupt_offset // codec.PAGE_BYTES, route["touched_page_indices"])

    def test_duplicate_region_and_invalid_contribution_reject_builder(self) -> None:
        region = self.source["regions"][0]
        with self.assertRaises(ValueError):
            codec.build_container(
                common, semantic,
                model_packet=self.source["model_packet"], semantic_packet=self.source["semantic_packet"], immutable_state=b"",
                regions=(region, region), weights=self.source["weights"], experts=2, baseline_object_bytes=1_000_000,
                audited_relative_mse=0.025, baseline_artifact_sha256="11" * 32, reconstruction_sha256="22" * 32,
                audit_binding_sha256="33" * 32, binding_hashes=bindings(),
            )

    def test_truncation_trailing_padding_and_model_tamper_reject(self) -> None:
        with self.assertRaises(ValueError):
            codec.parse_container(common, semantic, self.raw[:-codec.PAGE_BYTES])
        with self.assertRaises(ValueError):
            codec.parse_container(common, semantic, self.raw + bytes(codec.PAGE_BYTES))
        semantic_end = int(self.parsed["semantic_offset"] + self.parsed["semantic_bytes"])
        if semantic_end < int(self.parsed["immutable_offset"]):
            hostile = reseal_body_and_header(
                self.raw,
                lambda packet, _header, _directory_offset, _directory_bytes: packet.__setitem__(semantic_end, 1),
                directory_changed=False,
            )
            with self.assertRaises(ValueError):
                codec.parse_container(common, semantic, hostile)
        model_offset = int(self.parsed["model_offset"])
        hostile_model = reseal_body_and_header(
            self.raw,
            lambda packet, _header, _directory_offset, _directory_bytes: packet.__setitem__(
                model_offset + int(self.parsed["model_bytes"]) - 1,
                packet[model_offset + int(self.parsed["model_bytes"]) - 1] ^ 1,
            ),
            directory_changed=False,
        )
        with self.assertRaises(ValueError):
            codec.parse_container(common, semantic, hostile_model)


class PortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        groups = (("gate", ((0, 2), (7, 3))), ("up", ((8, 4), (31, 5))), ("down", ((32, 6), (127, 7))))
        cls.source128 = fixture.make_fixture(common, codec, semantic, experts=128, hidden=17, intermediate=19, shared_groups=groups)
        cls.raw128, cls.metrics128 = codec.build_container(
            common, semantic,
            model_packet=cls.source128["model_packet"], semantic_packet=cls.source128["semantic_packet"],
            immutable_state=b"awkward-128", regions=cls.source128["regions"], weights=cls.source128["weights"], experts=128,
            baseline_object_bytes=10_000_000, audited_relative_mse=0.025,
            baseline_artifact_sha256="11" * 32, reconstruction_sha256=cls.source128["reconstruction_sha256"],
            audit_binding_sha256="33" * 32, binding_hashes=bindings(),
        )
        cls.parsed128 = codec.parse_container(common, semantic, cls.raw128)

    def test_128_expert_different_shape_full_portability(self) -> None:
        parsed = self.parsed128
        self.assertEqual(parsed["experts"], 128)
        self.assertGreater(parsed["directory_bytes"], codec.PAGE_BYTES)
        self.assertTrue(any(127 in row["owners"] for row in parsed["directory"]))
        self.assertEqual(parsed["coverage"]["source_weights"], 128 * 3 * 17 * 19)
        self.assertEqual(codec.canonical_rebuild(common, semantic, parsed), self.raw128)
        self.assertEqual(fixture.decode_and_reencode(common, parsed)["decoded_streams"], len(parsed["directory"]))
        trace = codec.instrument_expert_pages(common, semantic, self.raw128, 127)
        self.assertGreater(trace["touched_page_bytes"], 0)

    def test_128_expert_descriptor_routed_decode_reencode_and_reconstruction(self) -> None:
        with tempfile.TemporaryFile() as handle:
            handle.write(self.raw128)
            handle.flush()
            os.fsync(handle.fileno())
            source = codec.AuthenticatedDescriptorSource(handle.fileno(), sha(self.raw128))
            try:
                metrics = codec.physical_metrics(
                    common,
                    semantic,
                    self.parsed128,
                    routed_descriptor_source=source,
                    externally_authenticated_container_sha256=sha(self.raw128),
                    routed_decoder=fixture.FixtureRoutedDecoder(common),
                )
            finally:
                source.close()
        self.assertTrue(metrics["routed_io_authoritative_descriptor_backed"])
        self.assertEqual(metrics["routed_full_reconstruction"]["full_reconstruction_f64_sha256"], self.source128["reconstruction_sha256"])
        self.assertEqual(len(metrics["experts"]), 128)
        self.assertTrue(all(row["causal_decode_reencode_reconstruction"]["all_payloads_canonically_reencoded"] for row in metrics["experts"]))
        expert127 = metrics["experts"][127]
        self.assertTrue(expert127["causal_decode_reencode_reconstruction"]["all_three_roles_reconstructed"])
        self.assertTrue(expert127["instrumented_routed_read_ranges"])

    def test_e250_unequal_shape_shared_tail_descriptor_portability(self) -> None:
        source = fixture.make_unequal_shape_e250_fixture(common, codec, semantic)
        self.assertEqual(source["experts"], 250)
        self.assertEqual(len(source["regions"]), source["expected_regions"])
        self.assertEqual(sum(len(region.streams) for region in source["regions"]), source["expected_streams"])
        raw, _diagnostic = codec.build_container(
            common, semantic,
            model_packet=source["model_packet"], semantic_packet=source["semantic_packet"],
            immutable_state=b"e250-unequal-v6", regions=source["regions"],
            weights=source["weights"], experts=250, baseline_object_bytes=10_000_000,
            audited_relative_mse=0.025, baseline_artifact_sha256="11" * 32,
            reconstruction_sha256=source["reconstruction_sha256"],
            audit_binding_sha256="33" * 32, binding_hashes=bindings(),
        )
        parsed = codec.parse_container(common, semantic, raw)
        self.assertEqual(codec.canonical_rebuild(common, semantic, parsed), raw)
        shared = [row for row in parsed["directory"] if tuple(row["owners"]) == (0, 249)]
        self.assertEqual(len(shared), 1)
        self.assertEqual(
            [(row["expert"], row["weight_count"]) for row in shared[0]["owner_contributions"]],
            [(0, 1), (249, 2)],
        )
        with tempfile.TemporaryFile() as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            source_fd = codec.AuthenticatedDescriptorSource(handle.fileno(), sha(raw))
            try:
                metrics = codec.physical_metrics(
                    common, semantic, parsed,
                    routed_descriptor_source=source_fd,
                    externally_authenticated_container_sha256=sha(raw),
                    routed_decoder=fixture.FixtureRoutedDecoder(common),
                )
            finally:
                source_fd.close()
        self.assertTrue(metrics["routed_full_reconstruction"]["matches_container_reconstruction"])
        self.assertTrue(all(
            row["causal_decode_reencode_reconstruction"]["all_payloads_canonically_reencoded"]
            for row in metrics["experts"]
        ))

    def test_all_boundary_expert_containers_roundtrip(self) -> None:
        for experts in (1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 255, 256):
            source = fixture.make_fixture(common, codec, semantic, experts=experts, hidden=1, intermediate=1)
            raw, _metrics = codec.build_container(
                common, semantic,
                model_packet=source["model_packet"], semantic_packet=source["semantic_packet"], immutable_state=b"",
                regions=source["regions"], weights=source["weights"], experts=experts, baseline_object_bytes=10_000_000,
                audited_relative_mse=0.025, baseline_artifact_sha256="11" * 32, reconstruction_sha256="22" * 32,
                audit_binding_sha256="33" * 32, binding_hashes=bindings(),
            )
            parsed = codec.parse_container(common, semantic, raw)
            self.assertEqual(parsed["experts"], experts)
            self.assertTrue(any(experts - 1 in row["owners"] for row in parsed["directory"]))
            self.assertEqual(codec.canonical_rebuild(common, semantic, parsed), raw)
        self.assertTrue(protocol.owner_set_from_ordinals(256, [255])[31] & 0x80)

    def test_256_expert_shared_owner_sets_cross_every_high_boundary(self) -> None:
        groups = (
            ("gate", ((63, 1), (64, 1))),
            ("up", ((127, 1), (128, 1))),
            ("down", ((254, 1), (255, 1))),
        )
        source = fixture.make_fixture(
            common, codec, semantic,
            experts=256, hidden=1, intermediate=2, shared_groups=groups,
        )
        raw, _metrics = codec.build_container(
            common, semantic,
            model_packet=source["model_packet"], semantic_packet=source["semantic_packet"],
            immutable_state=b"owner-boundaries-256", regions=source["regions"],
            weights=source["weights"], experts=256, baseline_object_bytes=10_000_000,
            audited_relative_mse=0.025, baseline_artifact_sha256="11" * 32,
            reconstruction_sha256=source["reconstruction_sha256"],
            audit_binding_sha256="33" * 32, binding_hashes=bindings(),
        )
        parsed = codec.parse_container(common, semantic, raw)
        observed = {tuple(row["owners"]): row["owner_set"] for row in parsed["directory"]}
        for owners in ((63, 64), (127, 128), (254, 255)):
            owner_set = observed[owners]
            self.assertEqual(len(owner_set), 32)
            self.assertEqual(codec.owner_ordinals(owner_set, 256), owners)
        self.assertTrue(observed[(127, 128)][15] & 0x80)
        self.assertTrue(observed[(127, 128)][16] & 0x01)
        self.assertTrue(observed[(254, 255)][31] & 0xC0)
        self.assertEqual(codec.canonical_rebuild(common, semantic, parsed), raw)

    def test_empty_expert_and_scalar_overlap_reject(self) -> None:
        semantics = semantic.parse_semantic_packet(self.source128["semantic_packet"])
        row = {
            "ordinal": 0,
            "role": "gate",
            "owners": (0,),
            "owner_contributions": ({"expert": 0, "role": "gate", "source_offset": 0, "weight_count": 323},),
            "source_weights": 323,
            "group_rows": 1,
            "group_cols": 323,
        }
        with self.assertRaises(ValueError):
            semantic.validate_stream_coverage(semantics, [row])
        small_semantics = semantic.parse_semantic_packet(
            semantic.build_semantic_packet((semantic.ExpertShape(0, 1, 2),))
        )
        overlap = [
            {"ordinal": 0, "role": "mixed", "owners": (0,), "owner_contributions": (
                {"expert": 0, "role": "gate", "source_offset": 0, "weight_count": 2},
                {"expert": 0, "role": "up", "source_offset": 0, "weight_count": 2},
                {"expert": 0, "role": "down", "source_offset": 0, "weight_count": 1},
            ), "source_weights": 5, "group_rows": 1, "group_cols": 5},
            {"ordinal": 1, "role": "down", "owners": (0,), "owner_contributions": (
                {"expert": 0, "role": "down", "source_offset": 0, "weight_count": 1},
            ), "source_weights": 1, "group_rows": 1, "group_cols": 1},
        ]
        with self.assertRaises(ValueError):
            semantic.validate_stream_coverage(small_semantics, overlap)

    def test_owner_relabel_probability_equivariance(self) -> None:
        source = fixture.make_fixture(
            common, codec, semantic,
            experts=3, hidden=3, intermediate=5,
            shared_groups=(("gate", ((0, 2), (2, 3))),),
        )
        permutation = {0: 2, 1: 0, 2: 1}
        relabeled_specs = []
        for region in source["regions"]:
            for spec in region.streams:
                contributions = tuple(sorted(
                    (
                        codec.OwnerContribution(
                            permutation[item.expert], item.role,
                            item.source_offset, item.weight_count,
                        )
                        for item in spec.owner_contributions
                    ),
                    key=lambda item: (item.expert, ("gate", "up", "down").index(item.role), item.source_offset),
                ))
                relabeled_specs.append(codec.StreamSpec(
                    ordinal=spec.ordinal, symbols=spec.symbols,
                    logical_bits=spec.logical_bits, payload=spec.payload,
                    source_digest=spec.source_digest, profile_q=spec.profile_q,
                    decoder_scale=spec.decoder_scale, role=spec.role,
                    group_rows=spec.group_rows, group_cols=spec.group_cols,
                    owner_contributions=contributions,
                ))
        grouped: dict[bytes, list[Any]] = {}
        for spec in relabeled_specs:
            owner_set = codec.owner_set_from_ordinals(3, [item.expert for item in spec.owner_contributions])
            grouped.setdefault(owner_set, []).append(spec)
        owner_sets = sorted(grouped, key=lambda value: (len(codec.owner_ordinals(value, 3)) != 1, codec.owner_ordinals(value, 3)))
        relabeled_regions = tuple(
            codec.RegionSpec(owner_set, tuple(sorted(grouped[owner_set], key=lambda item: item.ordinal)))
            for owner_set in owner_sets
        )
        raw_a, metrics_a = codec.build_container(
            common, semantic,
            model_packet=source["model_packet"], semantic_packet=source["semantic_packet"],
            immutable_state=b"relabel", regions=source["regions"], weights=source["weights"], experts=3,
            baseline_object_bytes=1_000_000, audited_relative_mse=0.025,
            baseline_artifact_sha256="11" * 32, reconstruction_sha256="22" * 32,
            audit_binding_sha256="33" * 32, binding_hashes=bindings(),
        )
        raw_b, metrics_b = codec.build_container(
            common, semantic,
            model_packet=source["model_packet"], semantic_packet=source["semantic_packet"],
            immutable_state=b"relabel", regions=relabeled_regions, weights=source["weights"], experts=3,
            baseline_object_bytes=1_000_000, audited_relative_mse=0.025,
            baseline_artifact_sha256="11" * 32, reconstruction_sha256="22" * 32,
            audit_binding_sha256="33" * 32, binding_hashes=bindings(),
        )
        parsed_a = codec.parse_container(common, semantic, raw_a)
        parsed_b = codec.parse_container(common, semantic, raw_b)
        stream_law_a = [(row["ordinal"], row["logical_bits"], row["payload"]) for row in parsed_a["directory"]]
        stream_law_b = [(row["ordinal"], row["logical_bits"], row["payload"]) for row in parsed_b["directory"]]
        self.assertEqual(stream_law_a, stream_law_b)
        self.assertEqual(parsed_a["model_packet"], parsed_b["model_packet"])
        self.assertEqual(metrics_a["actual_container_bytes"], metrics_b["actual_container_bytes"])
        self.assertEqual(codec.canonical_rebuild(common, semantic, parsed_b), raw_b)


class CpuBackend:
    def pack_streams(self, rows: list[tuple[bytes, bytes, bytes]]) -> dict[str, Any]:
        return {"rows": rows}

    def subset(self, packed: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        return {"rows": [packed["rows"][index] for index in indices]}

    def fit_counts(self, packed: dict[str, Any], topology_id: int, states: int, reset: int) -> list[int]:
        candidate = common.Candidate(common.TOPOLOGIES[topology_id], states, reset)
        tensors = []
        for bits, levels, base_bytes in packed["rows"]:
            base = list(struct.unpack(f"<{len(base_bytes) // 2}H", base_bytes))
            tensors.append(common.count_stream_cpu(list(bits), list(levels), base, candidate))
        return common.merge_counts(tensors)

    def exact_lengths(self, packed: dict[str, Any], topology_id: int, states: int, reset: int, frequencies: list[int]) -> list[int]:
        candidate = common.Candidate(common.TOPOLOGIES[topology_id], states, reset)
        output = []
        for bits, levels, base_bytes in packed["rows"]:
            base = list(struct.unpack(f"<{len(base_bytes) // 2}H", base_bytes))
            output.append(common.exact_stream_length_cpu(list(bits), list(levels), base, candidate, frequencies))
        return output


def scientific_panel() -> dict[str, Any]:
    rows = []
    for expert in range(6):
        for role_index, role in enumerate(("gate", "up", "down")):
            ordinal = 3 * expert + role_index
            length = 31 + ordinal
            bits = bytes(((position * 13 + ordinal * 7 + (position >> 2)) & 1) for position in range(length))
            levels = bytes((position + ordinal) % common.LEVELS for position in range(length))
            base = [1 + ((position * 7919 + ordinal * 811) % 65535) for position in range(length)]
            base_bytes = struct.pack(f"<{length}H", *base)
            owner = protocol.owner_set_from_ordinals(6, [expert])
            rows.append({
                "stream_ordinal": ordinal,
                "owner_set_hex": owner.hex(),
                "owner_set": owner,
                "owner_contributions": ({"expert": expert, "role": role, "source_offset": 0, "weight_count": 1},),
                "owner_expert_ordinals": [expert],
                "owner_identity_indices": [expert],
                "owner_weight_contributions": {expert: 1},
                "weight_charge": 1,
                "shape_rows": 1,
                "shape_cols": 1,
                "role": role,
                "symbols": length,
                "bits": list(bits),
                "levels": list(levels),
                "base": base,
                "bits_bytes": bits,
                "levels_bytes": levels,
                "base_bytes": base_bytes,
                "baseline_payload_bytes": (length + 7) // 8 + 20,
                "baseline_logical_bits": length,
                "profile_q": 0,
                "decoder_scale": 1.0,
                "logn": 1,
            })
    return {
        "streams": rows,
        "weights": 18,
        "experts": 6,
        "artifact": {"raw_bytes": 1_000_000, "raw_sha256": "cc" * 32},
        "immutable_state": b"",
        "semantic_identities": [(index, index) for index in range(6)],
        "expert_shapes": [
            {"expert": index, "hidden": 1, "intermediate": 1}
            for index in range(6)
        ],
        "reconstruction": {"full_reconstruction_f64_sha256": "44" * 32},
    }


class ScientificPipelineTests(unittest.TestCase):
    def test_nested_150_dynamic_confidence_and_exact_weights(self) -> None:
        panel = scientific_panel()
        panel["semantic_identities"] = [(0, index) for index in range(6)]
        backend = CpuBackend()
        cache = stage.prepare_backend_cache(backend, panel)
        result = stage.nested_holdout(common, protocol, backend, cache, panel)
        self.assertEqual(len(result["folds"]), 6)
        self.assertEqual(result["status"], "PASS_EXACT_IDENTITY_PRIMARY_HOLDOUT")
        self.assertEqual(result["confidence_degrees_of_freedom"], 5)
        self.assertAlmostEqual(sum(row["allocated_test_weights"] for row in result["folds"]), 18)
        self.assertTrue(all(len(row["development_stream_ordinals"]) == 15 for row in result["folds"]))
        strict = stage.coordinate_disjoint_diagnostic(common, protocol, backend, cache, panel)
        self.assertEqual(strict["status"], "NOT_RUN_NO_NONEMPTY_COORDINATE_DISJOINT_FOLDS")
        self.assertFalse(strict["positive_promotion"])
        source = (PACKAGE / "stage0_census.py").read_text(encoding="utf-8")
        self.assertNotIn("df=5", source)

    def test_legal_single_expert_panel_fails_safe_before_any_fit(self) -> None:
        panel = scientific_panel()
        panel["streams"] = panel["streams"][:3]
        panel["weights"] = 3
        panel["experts"] = 1
        panel["semantic_identities"] = [(0, 0)]
        panel["expert_shapes"] = panel["expert_shapes"][:1]
        projection = stage.projected_updates(common, protocol, panel)
        self.assertFalse(projection["primary_exact_identity_estimable"])
        backend = mock.Mock()
        result = stage.nested_holdout(common, protocol, backend, object(), panel)
        self.assertEqual(result["status"], "NOT_ESTIMABLE_EXACT_IDENTITY_HOLDOUT")
        self.assertFalse(result["estimable"])
        backend.fit_counts.assert_not_called()
        backend.exact_lengths.assert_not_called()

    def test_geometry_binds_semantic_identities_and_exact_shapes(self) -> None:
        panel = scientific_panel()
        original = protocol.geometry_sha256(common, panel)
        original_structural = protocol.structural_geometry_sha256(common, panel)
        arithmetic_change = copy.deepcopy(panel)
        arithmetic_change["streams"][0]["baseline_payload_bytes"] += 1
        arithmetic_change["streams"][0]["baseline_logical_bits"] += 7
        self.assertNotEqual(original, protocol.geometry_sha256(common, arithmetic_change))
        self.assertEqual(original_structural, protocol.structural_geometry_sha256(common, arithmetic_change))
        relabeled = copy.deepcopy(panel)
        relabeled["semantic_identities"] = [(99 + index, 199 + index) for index in range(6)]
        self.assertNotEqual(original, protocol.geometry_sha256(common, relabeled))
        malformed_shape = copy.deepcopy(panel)
        malformed_shape["expert_shapes"][5]["intermediate"] = 2
        with self.assertRaises(ValueError):
            protocol.panel_geometry(malformed_shape)
        missing_shape = copy.deepcopy(panel)
        missing_shape["expert_shapes"].pop()
        with self.assertRaises(ValueError):
            protocol.panel_geometry(missing_shape)

    def test_typed_preflight_rejects_status_geometry_and_device_forgery(self) -> None:
        preflight, evidence = valid_source_preflight()
        validated = stage.validate_source_preflight(common, protocol, preflight, evidence)
        self.assertEqual(validated["receipt_sha256"], evidence.source_preflight_receipt_sha256)

        bad_status = copy.deepcopy(dict(preflight.all150))
        bad_status["status"] = "FORGED_PRETEND_PASS"
        with self.assertRaises(ValueError):
            stage.validate_source_preflight(
                common, protocol,
                stage.SourcePreflightEvidence(bad_status, dict(preflight.representative), dict(preflight.independent_gpu_identity), preflight.receipt_sha256),
                evidence,
            )

        bad_cells = copy.deepcopy(dict(preflight.all150))
        bad_cells["cells"].pop()
        with self.assertRaises(ValueError):
            stage.validate_source_preflight(
                common, protocol,
                stage.SourcePreflightEvidence(bad_cells, dict(preflight.representative), dict(preflight.independent_gpu_identity), preflight.receipt_sha256),
                evidence,
            )

        bad_identity = dict(preflight.independent_gpu_identity)
        bad_identity["device_uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
        clean = dict(bad_identity)
        clean.pop("identity_receipt_sha256")
        bad_identity["identity_receipt_sha256"] = sha(json.dumps(
            clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii"))
        with self.assertRaises(ValueError):
            stage.validate_source_preflight(
                common, protocol,
                stage.SourcePreflightEvidence(dict(preflight.all150), dict(preflight.representative), bad_identity, preflight.receipt_sha256),
                evidence,
            )

    def test_preflight_rejects_honestly_resealed_duplicate_cells_and_sparse_representative(self) -> None:
        preflight, evidence = valid_source_preflight()

        duplicate_all150 = copy.deepcopy(dict(preflight.all150))
        duplicate_all150["cells"] = [copy.deepcopy(duplicate_all150["cells"][0]) for _ in range(150)]
        duplicate_all150["candidate_selector_sha256"] = sha(common.canonical_json([0] * 150))
        duplicate_all150["cell_results_sha256"] = sha(common.canonical_json(duplicate_all150["cells"]))
        duplicate_bundle, duplicate_evidence = reseal_source_preflight(
            stage.SourcePreflightEvidence(
                duplicate_all150,
                dict(preflight.representative),
                dict(preflight.independent_gpu_identity),
                preflight.receipt_sha256,
            ),
            evidence,
        )
        with self.assertRaisesRegex(ValueError, "canonical candidate ordering"):
            stage.validate_source_preflight(common, protocol, duplicate_bundle, duplicate_evidence)

        sparse_representative = {
            "schema": "uwfa-sc-v6-representative-source-free-preflight",
            "source_snapshot_root_sha256": evidence.source_snapshot_root_sha256,
            "status": "PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD",
            "fixture": {"streams": 15, "semantic_owners": 6, "private_streams": 12, "shared_tail_streams": 3},
            "outer_fold": {"all_150_candidates_fit_and_scored": True, "literal_container_parse_decode_reencode_rebuild": True},
            "runtime_projection": {"passes": True},
            "telemetry": dict(preflight.all150["environment"]),
            "model_h2d_bytes_nonzero": True,
            "d2h_bytes_nonzero": True,
            "peak_host_ram_recorded": True,
            "peak_vram_recorded": True,
        }
        sparse_bundle, sparse_evidence = reseal_source_preflight(
            stage.SourcePreflightEvidence(
                dict(preflight.all150),
                sparse_representative,
                dict(preflight.independent_gpu_identity),
                preflight.receipt_sha256,
            ),
            evidence,
        )
        with self.assertRaisesRegex(ValueError, "representative preflight"):
            stage.validate_source_preflight(common, protocol, sparse_bundle, sparse_evidence)

    def test_source_geometry_preflight_and_resource_gates_precede_any_fit(self) -> None:
        artifact = b"held-source-fixture"
        panel = scientific_panel()
        panel["artifact"] = {"raw_bytes": len(artifact), "raw_sha256": sha(artifact)}
        preflight, evidence = valid_source_preflight(score_bytes=b"score")
        backend = mock.Mock()
        with (
            mock.patch.object(stage, "prepare_panel", return_value=panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="ff" * 32),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="bb" * 32),
            mock.patch.object(stage, "prepare_backend_cache") as pack,
        ):
            with self.assertRaises(ValueError):
                stage.source_phase(
                    common=common, protocol=protocol, container_codec=codec, semantic_codec=semantic,
                    adapter=object(), backend=backend, artifact_bytes=artifact,
                    score_receipt_bytes=b"score", bindings=evidence,
                    source_preflight=preflight,
                    authenticated_descriptor_source_builder=lambda _raw: None,
                )
            pack.assert_not_called()

        with (
            mock.patch.object(stage, "prepare_panel", return_value=panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="aa" * 32),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="bb" * 32),
            mock.patch.object(common, "strict_json_loads", return_value={}),
            mock.patch.object(protocol, "validate_score_receipt", return_value={"relative_mse": 0.025}),
            mock.patch.object(stage, "projected_updates", return_value={
                "primary_exact_identity_estimable": True,
                "passes_pre_fit_resource_budget": False,
                "passes_pre_fit_runtime_budget": True,
            }),
            mock.patch.object(stage, "prepare_backend_cache") as pack,
        ):
            result = stage.source_phase(
                common=common, protocol=protocol, container_codec=codec, semantic_codec=semantic,
                adapter=object(), backend=backend, artifact_bytes=artifact,
                score_receipt_bytes=b"score", bindings=evidence,
                source_preflight=preflight,
                authenticated_descriptor_source_builder=lambda _raw: None,
            )
        self.assertEqual(result["status"], "ABORT_RESOURCE_BUDGET_BEFORE_BACKEND_PACK")
        pack.assert_not_called()
        backend.fit_counts.assert_not_called()

    def test_score_receipt_binds_full_geometry_and_independent_decoder(self) -> None:
        expected_geometry = "aa" * 32
        expected_decoder = "12" * 32
        artifact = b"score-source"
        record = {
            "schema": "uwfa-bound-baseline-score-v6",
            "status": "PASS_INDEPENDENT_BASELINE_SCORE",
            "artifact_sha256": sha(artifact),
            "artifact_bytes": len(artifact),
            "weights": 18,
            "relative_mse": 0.025,
            "sse_fp64": 0.025,
            "source_energy_fp64": 1.0,
            "normalization": "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
            "reconstruction_f64_sha256": "44" * 32,
            "original_source_panel_sha256": expected_geometry,
            "independent_decoder_source_sha256": expected_decoder,
        }
        record["score_receipt_sha256"] = sha(json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii"))
        protocol.validate_score_receipt(
            record, artifact_sha256=sha(artifact), artifact_bytes=len(artifact), weights=18,
            reconstruction_sha256="44" * 32,
            original_source_panel_sha256=expected_geometry,
            independent_decoder_source_sha256=expected_decoder,
        )
        for field in ("original_source_panel_sha256", "independent_decoder_source_sha256"):
            hostile = dict(record)
            hostile[field] = "ff" * 32
            hostile.pop("score_receipt_sha256")
            hostile["score_receipt_sha256"] = sha(json.dumps(
                hostile, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
            ).encode("ascii"))
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_score_receipt(
                    hostile, artifact_sha256=sha(artifact), artifact_bytes=len(artifact), weights=18,
                    reconstruction_sha256="44" * 32,
                    original_source_panel_sha256=expected_geometry,
                    independent_decoder_source_sha256=expected_decoder,
                )

    def test_controls_repeat_all_eight_complete_pipelines(self) -> None:
        events = []
        panel = scientific_panel()
        source_evidence = bound_evidence(full="aa" * 32, structural="dd" * 32, pipeline="bb" * 32)
        source_result = {
            "controls_may_be_opened": True,
            "source_full_geometry_sha256": "aa" * 32,
            "source_structural_geometry_sha256": "dd" * 32,
            "source_pipeline_sha256": "bb" * 32,
            "source_artifact_sha256": "cc" * 32,
            "source_final": {"absolute_saving_vs_bound_current_artifact_bpw": 1.0},
            "_panel": panel,
            "_bindings": source_evidence,
        }
        evidence = bound_evidence(full="ee" * 32, structural="dd" * 32, pipeline="bb" * 32)
        controls = [
            {
                "artifact_bytes": bytes((index + 1,)),
                "score_receipt_bytes": b"{}",
                "binding_record": {},
                "bindings": evidence,
                "moment_match_receipt_bytes": b"moment",
                "generator_source_bytes": b"generator",
            }
            for index in range(8)
        ]
        nested_result = {"estimable": True, "final_topology_selected_from_nested_fold_votes": {"selector_ordinal": 0}}
        final_result = {"absolute_saving_vs_bound_current_artifact_bpw": 0.5}
        class Adapter:
            pass
        with (
            mock.patch.object(stage, "prepare_panel", side_effect=lambda *_args: events.append("prepare") or panel),
            mock.patch.object(protocol, "geometry_sha256", side_effect=["aa" * 32] + ["ee" * 32] * 8),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="dd" * 32),
            mock.patch.object(protocol, "validate_control_binding", return_value={
                "moment_match_receipt_sha256": sha(b"moment"),
                "generator_source_sha256": sha(b"generator"),
            }),
            mock.patch.object(protocol, "validate_score_receipt", return_value={"relative_mse": 0.025}),
            mock.patch.object(common, "strict_json_loads", return_value={}),
            mock.patch.object(stage, "projected_updates", return_value={
                "primary_exact_identity_estimable": True,
                "passes_pre_fit_resource_budget": True,
                "passes_pre_fit_runtime_budget": True,
            }),
            mock.patch.object(stage, "prepare_backend_cache", return_value={}),
            mock.patch.object(stage, "nested_holdout", side_effect=lambda *_args: events.append("nested150") or nested_result),
            mock.patch.object(stage, "final_container", side_effect=lambda *_args: events.append("pack_decode") or final_result),
        ):
            result = stage.controls_phase(
                common=common, protocol=protocol, container_codec=codec, semantic_codec=semantic,
                adapter_factory=Adapter, backend_factory=CpuBackend, source_result=source_result,
                source_artifact_sha256="cc" * 32, controls=controls,
                authenticated_descriptor_source_builder=lambda _raw: None,
                moment_match_replayer=lambda **kwargs: {
                    "status": "PASS_RECOMPUTED_MOMENT_MATCH",
                    "seed": kwargs["seed"],
                    "source_moments_sha256": "55" * 32,
                    "control_moments_sha256": "66" * 32,
                    "moment_match_receipt_sha256": sha(b"moment"),
                },
            )
        self.assertEqual(events[:8], ["prepare"] * 8)
        self.assertEqual(events.count("nested150"), 8)
        self.assertEqual(events.count("pack_decode"), 8)
        self.assertEqual(len(result["controls"]), 8)

    def test_control_moment_and_score_bytes_veto_before_any_candidate_fit(self) -> None:
        panel = scientific_panel()
        evidence = bound_evidence(full="aa" * 32, structural="dd" * 32, pipeline="bb" * 32)
        source_result = {
            "controls_may_be_opened": True,
            "source_full_geometry_sha256": "aa" * 32,
            "source_structural_geometry_sha256": "dd" * 32,
            "source_pipeline_sha256": "bb" * 32,
            "source_artifact_sha256": "cc" * 32,
            "source_final": {"absolute_saving_vs_bound_current_artifact_bpw": 1.0},
            "_panel": panel,
            "_bindings": evidence,
        }
        good = {
            "artifact_bytes": b"control",
            "score_receipt_bytes": b"{}",
            "binding_record": {},
            "bindings": evidence,
            "moment_match_receipt_bytes": b"moment",
            "generator_source_bytes": b"generator",
        }
        controls = [dict(good) for _ in range(8)]
        controls[-1]["moment_match_receipt_bytes"] = b"forged-moment"
        with (
            mock.patch.object(stage, "prepare_panel", return_value=panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="aa" * 32),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="dd" * 32),
            mock.patch.object(protocol, "validate_control_binding", return_value={
                "moment_match_receipt_sha256": sha(b"moment"),
                "generator_source_sha256": sha(b"generator"),
            }),
            mock.patch.object(stage, "nested_holdout") as nested,
        ):
            with self.assertRaises(ValueError):
                stage.controls_phase(
                    common=common, protocol=protocol, container_codec=codec,
                    semantic_codec=semantic, adapter_factory=object,
                    backend_factory=CpuBackend, source_result=source_result,
                    source_artifact_sha256="cc" * 32, controls=controls,
                    authenticated_descriptor_source_builder=lambda _raw: None,
                    moment_match_replayer=lambda **_kwargs: {
                        "status": "PASS_RECOMPUTED_MOMENT_MATCH", "seed": 0,
                        "source_moments_sha256": "55" * 32,
                        "control_moments_sha256": "66" * 32,
                        "moment_match_receipt_sha256": sha(b"moment"),
                    },
                )
            nested.assert_not_called()

    def test_controls_reject_wrong_caller_artifact_and_foreign_symmetric_closure_before_fit(self) -> None:
        panel = scientific_panel()
        source_evidence = bound_evidence(full="aa" * 32, structural="dd" * 32, pipeline="bb" * 32)
        source_result = {
            "controls_may_be_opened": True,
            "source_full_geometry_sha256": "aa" * 32,
            "source_structural_geometry_sha256": "dd" * 32,
            "source_pipeline_sha256": "bb" * 32,
            "source_artifact_sha256": "cc" * 32,
            "source_final": {"absolute_saving_vs_bound_current_artifact_bpw": 1.0},
            "_panel": panel,
            "_bindings": source_evidence,
        }
        base_control = {
            "artifact_bytes": b"control",
            "score_receipt_bytes": b"{}",
            "binding_record": {},
            "bindings": source_evidence,
            "moment_match_receipt_bytes": b"moment",
            "generator_source_bytes": b"generator",
        }
        with (
            mock.patch.object(protocol, "geometry_sha256", return_value="aa" * 32),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="dd" * 32),
            mock.patch.object(stage, "prepare_panel") as prepare,
            mock.patch.object(stage, "nested_holdout") as nested,
        ):
            with self.assertRaisesRegex(ValueError, "caller source artifact digest"):
                stage.controls_phase(
                    common=common, protocol=protocol, container_codec=codec,
                    semantic_codec=semantic, adapter_factory=object,
                    backend_factory=CpuBackend, source_result=source_result,
                    source_artifact_sha256="fe" * 32,
                    controls=[dict(base_control) for _ in range(8)],
                    authenticated_descriptor_source_builder=lambda _raw: None,
                    moment_match_replayer=lambda **_kwargs: {},
                )
            prepare.assert_not_called()
            nested.assert_not_called()

        foreign_evidence = replace(source_evidence, baseline_plan_sha256="fe" * 32)
        foreign_control = dict(base_control)
        foreign_control["bindings"] = foreign_evidence
        with (
            mock.patch.object(stage, "prepare_panel", return_value=panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="aa" * 32),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="dd" * 32),
            mock.patch.object(protocol, "validate_control_binding", return_value={
                "moment_match_receipt_sha256": sha(b"moment"),
                "generator_source_sha256": sha(b"generator"),
            }),
            mock.patch.object(common, "strict_json_loads", return_value={}),
            mock.patch.object(stage, "nested_holdout") as nested,
        ):
            with self.assertRaisesRegex(ValueError, "symmetric closure"):
                stage.controls_phase(
                    common=common, protocol=protocol, container_codec=codec,
                    semantic_codec=semantic, adapter_factory=object,
                    backend_factory=CpuBackend, source_result=source_result,
                    source_artifact_sha256="cc" * 32,
                    controls=[dict(foreign_control) for _ in range(8)],
                    authenticated_descriptor_source_builder=lambda _raw: None,
                    moment_match_replayer=lambda **kwargs: {
                        "status": "PASS_RECOMPUTED_MOMENT_MATCH",
                        "seed": kwargs["seed"],
                        "source_moments_sha256": "55" * 32,
                        "control_moments_sha256": "66" * 32,
                        "moment_match_receipt_sha256": sha(b"moment"),
                    },
                )
            nested.assert_not_called()

        controls = [dict(base_control) for _ in range(8)]
        controls[-1]["score_receipt_bytes"] = b"forged-score"
        with (
            mock.patch.object(stage, "prepare_panel", return_value=panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="aa" * 32),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="dd" * 32),
            mock.patch.object(protocol, "validate_control_binding", return_value={
                "moment_match_receipt_sha256": sha(b"moment"),
                "generator_source_sha256": sha(b"generator"),
            }),
            mock.patch.object(common, "strict_json_loads", return_value={}),
            mock.patch.object(stage, "nested_holdout") as nested,
        ):
            with self.assertRaises(ValueError):
                stage.controls_phase(
                    common=common, protocol=protocol, container_codec=codec,
                    semantic_codec=semantic, adapter_factory=object,
                    backend_factory=CpuBackend, source_result=source_result,
                    source_artifact_sha256="cc" * 32, controls=controls,
                    authenticated_descriptor_source_builder=lambda _raw: None,
                    moment_match_replayer=lambda **kwargs: {
                        "status": "PASS_RECOMPUTED_MOMENT_MATCH",
                        "seed": kwargs["seed"],
                        "source_moments_sha256": "55" * 32,
                        "control_moments_sha256": "66" * 32,
                        "moment_match_receipt_sha256": sha(b"moment"),
                    },
                )
            nested.assert_not_called()

    def test_promotion_requires_every_gate(self) -> None:
        names = ("physical", "cold", "heldout", "specificity", "standalone_decode", "integrity", "independent_result_audit")
        self.assertTrue(stage.promotion_conjunction(**{name: True for name in names}))
        for missing in names:
            values = {name: True for name in names}
            values[missing] = False
            self.assertFalse(stage.promotion_conjunction(**values))


class ColdReadAttributionTests(unittest.TestCase):
    def test_asymmetric_shared_tail_uses_owner_local_denominator(self) -> None:
        experts = 8
        all_owners = protocol.owner_set_from_ordinals(experts, list(range(experts)))
        ledger = [{"bytes": 8192, "owner_set": all_owners, "padding": False}]
        private_sizes = (4096,) + (8192,) * 7
        for expert, amount in enumerate(private_sizes):
            ledger.append({
                "bytes": amount,
                "owner_set": protocol.owner_set_from_ordinals(experts, [expert]),
                "padding": False,
            })
        total = 8192 + sum(private_sizes)
        directory = [
            {
                "symbols": 1,
                "source_weights": 1,
                "owners": (expert,),
                "owner_contributions": ({"expert": expert, "weight_count": 1},),
            }
            for expert in range(experts)
        ]
        parsed = {
            "raw": bytes(total),
            "weights": experts,
            "experts": experts,
            "directory": directory,
            "byte_ledger": ledger,
            "audited_relative_mse": 0.025,
            "baseline_object_bytes": total,
        }

        def routed(_common: Any, _semantic: Any, _raw: bytes, expert: int) -> dict[str, Any]:
            # Two 4 KiB shared pages plus the expert-private page range. Expert
            # zero's 12 KiB logical route touches four pages due to alignment.
            touched = 16384 if expert == 0 else 20480
            return {
                "touched_page_bytes": touched,
                "touched_page_indices": tuple(range(touched // codec.PAGE_BYTES)),
                "routed_read_ranges": (),
                "rows": (directory[expert],),
                "installation_authentication_scan_bytes": total,
            }

        with mock.patch.object(codec, "instrument_expert_pages", side_effect=routed):
            metrics = codec.physical_metrics(common, semantic, parsed)
        row0 = metrics["experts"][0]
        owner_local = Fraction(4096 + 8192 // experts, 1)
        self.assertEqual(row0["attributable_nonpadding_decodable_bytes"]["exact"], "5120/1")
        self.assertEqual(Fraction(12288, 1) / owner_local, Fraction(12, 5))  # 2.4 before page rounding
        self.assertEqual(row0["strict_cold_amplification"]["exact"], "16/5")  # 3.2 with exact touched pages
        naive = Fraction(16384, 1) / Fraction(total, experts)
        self.assertAlmostEqual(float(naive), 1.8823529411764706)
        self.assertLess(naive, 2)
        self.assertFalse(metrics["passes_cold_read_below_2x"])


class _FakeFlags:
    c_contiguous = True


class _FakeArray:
    def __init__(self, values: Any, itemsize: int, dtype: str):
        self.values = list(values) if not isinstance(values, int) else [0] * values
        self.nbytes = len(self.values) * itemsize
        self.dtype = dtype
        self.ndim = 1
        self.size = len(self.values)
        self.shape = (len(self.values),)
        self.flags = _FakeFlags()

    def get(self) -> "_FakeArray":
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class _FakeRuntime:
    def __init__(self, calls: list[str]):
        self.calls = calls

    def deviceSynchronize(self) -> None:
        self.calls.append("sync")

    def memGetInfo(self) -> tuple[int, int]:
        self.calls.append("meminfo")
        return 30 * (1 << 30), 32 * (1 << 30)


class _FakeCuda:
    def __init__(self, calls: list[str]):
        self.runtime = _FakeRuntime(calls)


class _FakeCP:
    uint8 = "uint8"
    uint16 = "uint16"
    uint64 = "uint64"
    uint32 = int
    int32 = int

    def __init__(self):
        self.calls: list[str] = []
        self.cuda = _FakeCuda(self.calls)

    def asarray(self, values: Any, *, dtype: str) -> _FakeArray:
        self.calls.append("asarray")
        return _FakeArray(values, 2 if dtype == self.uint16 else 8, dtype)

    def frombuffer(self, values: bytes, *, dtype: str) -> _FakeArray:
        self.calls.append("frombuffer")
        itemsize = 2 if dtype == self.uint16 else 1
        count = len(values) // itemsize
        return _FakeArray(count, itemsize, dtype)

    def zeros(self, count: int, *, dtype: str) -> _FakeArray:
        self.calls.append("zeros")
        return _FakeArray(count, 8, dtype)


class TelemetryContractTests(unittest.TestCase):
    @staticmethod
    def backend() -> tuple[Any, _FakeCP]:
        fake = _FakeCP()
        backend = cupy_backend.CuPyUnifilarBackend.__new__(cupy_backend.CuPyUnifilarBackend)
        backend.cp = fake
        backend.length_kernel = lambda _grid, _block, _args: fake.calls.append("kernel")
        backend._sample = lambda phase: fake.calls.append("sample:" + phase)
        backend.stats = {
            "h2d_bytes": 0,
            "h2d_payload_bytes": 0,
            "h2d_root_descriptor_bytes": 0,
            "h2d_subset_descriptor_bytes": 0,
            "h2d_launch_descriptor_bytes": 0,
            "h2d_model_table_bytes": 0,
            "h2d_kernel_scalar_bytes": 0,
            "device_output_allocation_bytes": 0,
            "kernel_wall_seconds": 0.0,
            "kernel_count": 0,
            "length_kernel_count": 0,
            "length_cell_symbol_updates": 0,
            "pack_calls": 0,
        }
        return backend, fake

    def test_model_h2d_and_kernel_accounting_are_exact(self) -> None:
        backend, fake = self.backend()
        packed = backend.pack_streams([
            (b"\x00\x01", b"\x00\x01", struct.pack("<2H", 1, 65535)),
            (b"\x01\x00\x01", b"\x02\x01\x00", struct.pack("<3H", 32768, 1, 65535)),
        ])
        before_h2d = backend.stats["h2d_bytes"]
        frequencies = [32768] * (2 * common.CONTEXTS)
        result = backend.exact_lengths(packed, 0, 2, 32, frequencies)
        self.assertEqual(result.nbytes, 16)
        self.assertEqual(backend.stats["h2d_model_table_bytes"], 2 * len(frequencies))
        self.assertEqual(backend.stats["h2d_kernel_scalar_bytes"], 16)
        self.assertEqual(backend.stats["h2d_launch_descriptor_bytes"], 32)
        self.assertEqual(backend.stats["h2d_bytes"] - before_h2d, 2 * len(frequencies) + 16 + 32)
        self.assertEqual(backend.stats["kernel_count"], 1)
        self.assertEqual(backend.stats["length_cell_symbol_updates"], 5)
        self.assertIn("kernel", fake.calls)

    def test_invalid_geometry_rejects_before_any_device_call(self) -> None:
        backend, fake = self.backend()
        forged = {"stream_count": 1, "symbol_count": 1}
        with self.assertRaises(ValueError):
            backend.exact_lengths(forged, 0, 3, 32, [1])
        self.assertEqual(fake.calls, [])
        with self.assertRaises(ValueError):
            backend.exact_lengths(forged, 0, 2, 32, [1])
        self.assertEqual(fake.calls, [])
        packed = backend.pack_streams([(b"\x00", b"\x00", struct.pack("<H", 1))])
        calls = list(fake.calls)
        dict.__setitem__(packed, "symbol_count", -1)
        with self.assertRaises(ValueError):
            backend.exact_lengths(packed, 0, 2, 32, [1] * (2 * common.CONTEXTS))
        self.assertEqual(fake.calls, calls)

    def test_forged_and_cross_backend_handles_reject_before_device_call(self) -> None:
        backend_a, fake_a = self.backend()
        backend_b, fake_b = self.backend()
        packed = backend_a.pack_streams([(b"\x00\x01", b"\x00\x01", struct.pack("<2H", 1, 65535))])
        before_a = list(fake_a.calls)
        before_b = list(fake_b.calls)
        forged = dict(packed)
        with self.assertRaises(ValueError):
            backend_a.exact_lengths(forged, 0, 2, 32, [32768] * (2 * common.CONTEXTS))
        self.assertEqual(fake_a.calls, before_a)
        with self.assertRaises(ValueError):
            backend_b.exact_lengths(packed, 0, 2, 32, [32768] * (2 * common.CONTEXTS))
        self.assertEqual(fake_b.calls, before_b)

        original_offsets = packed["host_offsets"]
        dict.__setitem__(packed, "host_offsets", (1,))
        with self.assertRaises(ValueError):
            backend_a.exact_lengths(packed, 0, 2, 32, [32768] * (2 * common.CONTEXTS))
        self.assertEqual(fake_a.calls, before_a)
        dict.__setitem__(packed, "host_offsets", original_offsets)

    def test_peak_fields_and_explicit_d2h_are_present(self) -> None:
        source = (PACKAGE / "cupy_backend.py").read_text(encoding="utf-8")
        for field in (
            "peak_process_tree_rss_bytes", "peak_process_hwm_bytes", "peak_vram_incremental_bytes",
            "peak_default_pool_used_bytes", "peak_default_pool_total_bytes", "peak_pinned_pool_free_blocks",
            "h2d_model_table_bytes", "d2h_bytes", "kernel_count",
        ):
            self.assertIn(field, source)
        self.assertIn("def to_host_list", source)

    def test_static_caps_match_stage_and_reject_oversize_before_cuda(self) -> None:
        self.assertEqual(cupy_backend.MAX_HOST_BYTES, stage.MAX_HOST_BYTES)
        self.assertEqual(cupy_backend.MAX_VRAM_BYTES, stage.MAX_VRAM_BYTES)
        self.assertEqual(cupy_backend.MAX_PACKED_SYMBOLS, stage.MAX_PACKED_SYMBOLS)
        backend, fake = self.backend()
        with self.assertRaises(ValueError):
            backend.pack_resource_plan(cupy_backend.MAX_PACKED_SYMBOLS + 1, 1)
        self.assertEqual(fake.calls, [])

    def test_uuid_and_pci_canonicalization_is_exact_and_fail_closed(self) -> None:
        raw = bytes.fromhex("c06e0fe098362f988f100514d085f722")
        self.assertEqual(
            cupy_backend.CuPyUnifilarBackend._canonical_device_uuid(raw),
            "GPU-c06e0fe0-9836-2f98-8f10-0514d085f722",
        )
        self.assertEqual(
            cupy_backend.CuPyUnifilarBackend._canonical_pci_bus_id("0000:16:00.0"),
            "00000000:16:00.0",
        )
        with self.assertRaises(RuntimeError):
            cupy_backend.CuPyUnifilarBackend._canonical_device_uuid(b"short")
        with self.assertRaises(RuntimeError):
            cupy_backend.CuPyUnifilarBackend._canonical_pci_bus_id("16:00.0")


class SecurityAndSealTests(unittest.TestCase):
    def test_direct_stage_is_inert(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(PACKAGE / "stage0_census.py"), "--payload", "forbidden"],
            cwd=str(PACKAGE), capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("BLOCK_DIRECT_EXECUTION_REQUIRES_EXTERNALLY_PINNED_DISPATCHER", completed.stderr)
        self.assertNotIn("cupy", completed.stdout.lower() + completed.stderr.lower())

    def test_internal_dispatcher_cannot_self_authorize(self) -> None:
        with self.assertRaises(dispatcher.DispatchContractError):
            dispatcher.reject_direct_payload_launch()

    def test_completion_last_faults_never_delete_members_or_leave_empty_final(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "ab" * 32)
            try:
                transaction_id = "01" * 16
                staging = parent_path / f".uwfa-incomplete.{transaction_id}.incomplete"
                with self.assertRaises(RuntimeError):
                    with common.CompletionLastOutput(retained, "incomplete", transaction_id) as transaction:
                        transaction.write_new("RESULT.bin", b"durable")
                        raise RuntimeError("injected post-member fault")
                self.assertFalse((parent_path / "incomplete").exists())
                self.assertEqual((staging / "RESULT.bin").read_bytes(), b"durable")
                self.assertTrue((staging / "RUN_STATE.json").is_file())
                self.assertFalse((staging / "COMPLETE.json").exists())

                with common.CompletionLastOutput(retained, "completed", "02" * 16) as transaction:
                    member = transaction.write_new("RESULT.bin", b"durable")
                    transaction.complete(list(transaction.members), "ab" * 32)
                    self.assertEqual(member["sha256"], sha(b"durable"))
                self.assertEqual((parent_path / "completed" / "RESULT.bin").read_bytes(), b"durable")
                self.assertTrue((parent_path / "completed" / "COMPLETE.json").is_file())
                marker = parent_path / common.parent_commit_marker_name("completed")
                self.assertTrue(marker.is_file())
                verified = envelope.verify_completed_under_parent(
                    common, retained, "completed", expected_source_manifest_sha256="ab" * 32,
                )
                self.assertEqual(verified["status"], "PASS_PARENT_MARKER_COMMITTED_ENVELOPE")
            finally:
                retained.close()

    def test_parent_marker_rejects_staging_name_substitution_before_named_move(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "a1" * 32)
            real_move = common._rename_directory_noreplace
            final_name = "before-move"
            transaction_id = "21" * 16

            def substitute(parent_fd: int, source: str, destination: str) -> None:
                self.assertEqual((source, destination), (
                    f".uwfa-{final_name}.{transaction_id}.incomplete", final_name,
                ))
                os.rename(source, "verified-staging-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.mkdir(source, 0o700, dir_fd=parent_fd)
                attacker_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                try:
                    fd = os.open("ATTACKER.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=attacker_fd)
                    os.write(fd, b"substitute-before-move")
                    os.close(fd)
                finally:
                    os.close(attacker_fd)
                real_move(parent_fd, source, destination)

            try:
                with mock.patch.object(common, "_rename_directory_noreplace", side_effect=substitute):
                    with self.assertRaisesRegex(common.ContractError, "retained staging directory"):
                        with common.CompletionLastOutput(retained, final_name, transaction_id) as transaction:
                            transaction.write_new("RESULT.bin", b"authentic")
                            transaction.complete(list(transaction.members), "ab" * 32)
                self.assertFalse((parent_path / common.parent_commit_marker_name(final_name)).exists())
                self.assertEqual((parent_path / final_name / "ATTACKER.bin").read_bytes(), b"substitute-before-move")
                with self.assertRaises((FileNotFoundError, ValueError, common.ContractError)):
                    envelope.verify_completed_under_parent(
                        common, retained, final_name, expected_source_manifest_sha256="ab" * 32,
                    )
            finally:
                retained.close()

    def test_parent_marker_rejects_final_substitution_after_move_before_marker(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "a2" * 32)
            real_open_marker = common._open_held_commit_authority_file
            final_name = "before-marker"

            def substitute(parent_fd: int, anchor_name: str) -> tuple[int, str | None]:
                os.rename(final_name, "verified-final-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.mkdir(final_name, 0o700, dir_fd=parent_fd)
                return real_open_marker(parent_fd, anchor_name)

            try:
                with mock.patch.object(common, "_open_held_commit_authority_file", side_effect=substitute):
                    with self.assertRaisesRegex(common.ContractError, "retained staging directory"):
                        with common.CompletionLastOutput(retained, final_name, "22" * 16) as transaction:
                            transaction.write_new("RESULT.bin", b"authentic")
                            transaction.complete(list(transaction.members), "ab" * 32)
                self.assertFalse((parent_path / common.parent_commit_marker_name(final_name)).exists())
                with self.assertRaises((FileNotFoundError, ValueError, common.ContractError)):
                    envelope.verify_completed_under_parent(
                        common, retained, final_name, expected_source_manifest_sha256="ab" * 32,
                    )
            finally:
                retained.close()

    def test_parent_marker_makes_after_link_directory_substitution_unverifiable(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "a3" * 32)
            real_link = common._link_held_unnamed_file_noreplace
            final_name = "after-marker"

            def link_then_substitute(source_fd: int, parent_fd: int, destination: str) -> None:
                real_link(source_fd, parent_fd, destination)
                os.rename(final_name, "committed-final-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.mkdir(final_name, 0o700, dir_fd=parent_fd)

            try:
                with mock.patch.object(common, "_link_held_unnamed_file_noreplace", side_effect=link_then_substitute):
                    with self.assertRaisesRegex(common.ContractError, "retained staging directory"):
                        with common.CompletionLastOutput(retained, final_name, "23" * 16) as transaction:
                            transaction.write_new("RESULT.bin", b"authentic")
                            transaction.complete(list(transaction.members), "ab" * 32)
                self.assertTrue((parent_path / common.parent_commit_marker_name(final_name)).is_file())
                with self.assertRaisesRegex(ValueError, "final-directory inode mismatch"):
                    envelope.verify_completed_under_parent(
                        common, retained, final_name, expected_source_manifest_sha256="ab" * 32,
                    )
            finally:
                retained.close()

    def test_complete_json_without_parent_marker_is_never_complete(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "a4" * 32)
            try:
                result = parent_path / "markerless"
                result.mkdir()
                (result / "COMPLETE.json").write_bytes(common.pretty_json(common.seal_record({
                    "schema": "unifilar-wfa-completion-v6",
                    "status": "COMPLETE_LAST",
                    "source_manifest_sha256": "ab" * 32,
                    "members": [],
                }, "completion_sha256")))
                with self.assertRaises(FileNotFoundError):
                    envelope.verify_completed_under_parent(
                        common, retained, "markerless", expected_source_manifest_sha256="ab" * 32,
                    )
            finally:
                retained.close()

    def test_output_symlink_ancestor_and_parent_substitution_are_descriptor_safe(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises((OSError, common.ContractError)):
                common.RetainedOutputParent.open_path_source_only(alias, "ab" * 32)

            held_path = root / "held"
            moved_path = root / "moved"
            held_path.mkdir()
            retained = common.RetainedOutputParent.open_path_source_only(held_path, "cd" * 32)
            try:
                held_path.rename(moved_path)
                held_path.mkdir()
                with common.CompletionLastOutput(retained, "result", "03" * 16) as transaction:
                    transaction.write_new("RESULT.bin", b"pinned-parent")
                    transaction.complete(list(transaction.members), "ab" * 32)
                self.assertTrue((moved_path / "result" / "COMPLETE.json").is_file())
                self.assertFalse((held_path / "result").exists())
            finally:
                retained.close()

    def test_completion_rehash_rejects_mutated_declared_member_before_publication(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "aa" * 32)
            transaction_id = "06" * 16
            try:
                with common.CompletionLastOutput(retained, "mutated", transaction_id) as transaction:
                    transaction.write_new("RESULT.json", b"original")
                    fd = os.open("RESULT.json", os.O_WRONLY | os.O_TRUNC, dir_fd=transaction.dir_fd)
                    try:
                        os.write(fd, b"changed")
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    with self.assertRaises(common.ContractError):
                        transaction.complete(list(transaction.members), "ab" * 32)
                self.assertFalse((parent_path / "mutated").exists())
                staging = parent_path / f".uwfa-mutated.{transaction_id}.incomplete"
                self.assertTrue((staging / "RESULT.json").is_file())
                self.assertFalse((staging / "COMPLETE.json").exists())
            finally:
                retained.close()

    def test_completion_reenumeration_rejects_undeclared_member_before_publication(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "bb" * 32)
            transaction_id = "07" * 16
            try:
                with common.CompletionLastOutput(retained, "undeclared", transaction_id) as transaction:
                    transaction.write_new("RESULT.json", b"original")
                    fd = os.open(
                        "UNDECLARED.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600, dir_fd=transaction.dir_fd,
                    )
                    os.close(fd)
                    with self.assertRaises(common.ContractError):
                        transaction.complete(list(transaction.members), "ab" * 32)
                self.assertFalse((parent_path / "undeclared").exists())
                staging = parent_path / f".uwfa-undeclared.{transaction_id}.incomplete"
                self.assertTrue((staging / "UNDECLARED.bin").is_file())
                self.assertFalse((staging / "COMPLETE.json").exists())
            finally:
                retained.close()

    def test_publication_is_no_replace_and_post_rename_fault_cannot_delete_final(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication is a POSIX production contract")
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory)
            retained = common.RetainedOutputParent.open_path_source_only(parent_path, "ef" * 32)
            try:
                with self.assertRaises(OSError):
                    with common.CompletionLastOutput(retained, "raced", "04" * 16) as transaction:
                        transaction.write_new("RESULT.bin", b"new")
                        (parent_path / "raced").mkdir()
                        (parent_path / "raced" / "ATTACKER.bin").write_bytes(b"old")
                        transaction.complete(list(transaction.members), "ab" * 32)
                self.assertEqual((parent_path / "raced" / "ATTACKER.bin").read_bytes(), b"old")
                self.assertFalse((parent_path / "raced" / "RESULT.bin").exists())

                real_fsync = common.os.fsync
                final_path = parent_path / "durable-final"

                def fail_only_after_publish(fd: int) -> None:
                    if final_path.is_dir():
                        raise OSError("injected parent fsync after durable rename")
                    real_fsync(fd)

                with mock.patch.object(common.os, "fsync", side_effect=fail_only_after_publish):
                    with self.assertRaises(OSError):
                        with common.CompletionLastOutput(retained, "durable-final", "05" * 16) as transaction:
                            transaction.write_new("RESULT.bin", b"irrevocable")
                            transaction.complete(list(transaction.members), "ab" * 32)
                self.assertEqual((final_path / "RESULT.bin").read_bytes(), b"irrevocable")
                self.assertTrue((final_path / "COMPLETE.json").is_file())
                self.assertGreater(len(list(final_path.iterdir())), 0)
            finally:
                retained.close()

    def test_no_legacy_owner_or_float_geometry_shortcuts(self) -> None:
        forbidden = ("owner_mask", "minimum_rate_bpw", "df=5")
        for path in PACKAGE.glob("*.py"):
            if path.name == "test_source_only.py":
                continue
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{token} in {path.name}")

    def test_manifest_verifier_only_after_manifest_exists(self) -> None:
        manifest = PACKAGE / "SOURCE_MANIFEST.json"
        if not manifest.exists():
            self.skipTest("pre-review source tree intentionally has no manifest")
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(PACKAGE / "verify_source.py"), "--package", str(PACKAGE), "--compact"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
