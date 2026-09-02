#!/usr/bin/env python3
"""Hostile source-only UWFA-SC v3 producer tests.

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


common = load("uwfa_v3_test_common", "uwfa_common.py")
protocol = load("uwfa_v3_test_protocol", "protocol.py")
semantic = load("uwfa_v3_test_semantic", "universal_adapter.py")
codec = load("uwfa_v3_test_container", "container_codec.py")
fixture = load("uwfa_v3_test_fixture", "fixture_portability.py")
stage = load("uwfa_v3_test_stage", "stage0_census.py")
cupy_backend = load("uwfa_v3_test_cupy", "cupy_backend.py")
dispatcher = load("uwfa_v3_test_dispatcher", "dispatcher_contract.py")
envelope = load("uwfa_v3_test_envelope", "result_envelope.py")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bindings() -> dict[str, str]:
    return {name: sha(("binding:" + name).encode("ascii")) for name in codec._HEADER_BINDINGS}


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
        immutable_state=b"source-only-v3-fixture",
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
        "artifact": {"raw_bytes": 1_000_000},
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
        backend = CpuBackend()
        cache = stage.prepare_backend_cache(backend, panel)
        result = stage.nested_holdout(common, protocol, backend, cache, panel)
        self.assertEqual(len(result["folds"]), 6)
        self.assertEqual(result["confidence_degrees_of_freedom"], 5)
        self.assertAlmostEqual(sum(row["allocated_test_weights"] for row in result["folds"]), 18)
        source = (PACKAGE / "stage0_census.py").read_text(encoding="utf-8")
        self.assertNotIn("df=5", source)

    def test_geometry_binds_semantic_identities_and_exact_shapes(self) -> None:
        panel = scientific_panel()
        original = protocol.geometry_sha256(common, panel)
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

    def test_controls_repeat_all_eight_complete_pipelines(self) -> None:
        events = []
        panel = scientific_panel()
        source_result = {
            "controls_may_be_opened": True,
            "source_geometry_sha256": "aa" * 32,
            "source_pipeline_sha256": "bb" * 32,
            "source_final": {"absolute_saving_vs_bound_current_artifact_bpw": 1.0},
            "_panel": panel,
        }
        evidence = stage.BoundEvidence(
            baseline_plan_sha256="10" * 32,
            baseline_score_sha256=sha(b"{}"),
            universal_decoder_sha256="12" * 32,
            producer_manifest_sha256="13" * 32,
            audit_bootstrap_sha256="14" * 32,
            source_panel_sha256="aa" * 32,
            extraction_program_sha256="16" * 32,
            universal_adapter_sha256="17" * 32,
            pipeline_sha256="bb" * 32,
        )
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
        nested_result = {"final_topology_selected_from_nested_fold_votes": {"selector_ordinal": 0}}
        final_result = {"absolute_saving_vs_bound_current_artifact_bpw": 0.5}
        class Adapter:
            pass
        with (
            mock.patch.object(stage, "prepare_panel", side_effect=lambda *_args: events.append("prepare") or panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="aa" * 32),
            mock.patch.object(protocol, "validate_control_binding", return_value={
                "moment_match_receipt_sha256": sha(b"moment"),
                "generator_source_sha256": sha(b"generator"),
            }),
            mock.patch.object(protocol, "validate_score_receipt", return_value={"relative_mse": 0.025}),
            mock.patch.object(common, "strict_json_loads", return_value={}),
            mock.patch.object(stage, "projected_updates", return_value={"passes_pre_fit_runtime_budget": True}),
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
        evidence = stage.BoundEvidence(
            baseline_plan_sha256="10" * 32,
            baseline_score_sha256=sha(b"{}"),
            universal_decoder_sha256="12" * 32,
            producer_manifest_sha256="13" * 32,
            audit_bootstrap_sha256="14" * 32,
            source_panel_sha256="aa" * 32,
            extraction_program_sha256="16" * 32,
            universal_adapter_sha256="17" * 32,
            pipeline_sha256="bb" * 32,
        )
        source_result = {
            "controls_may_be_opened": True,
            "source_geometry_sha256": "aa" * 32,
            "source_pipeline_sha256": "bb" * 32,
            "source_final": {"absolute_saving_vs_bound_current_artifact_bpw": 1.0},
            "_panel": panel,
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

        controls = [dict(good) for _ in range(8)]
        controls[-1]["score_receipt_bytes"] = b"forged-score"
        with (
            mock.patch.object(stage, "prepare_panel", return_value=panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="aa" * 32),
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
        parsed = {
            "raw": bytes(total),
            "weights": 1,
            "experts": experts,
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
        with tempfile.TemporaryDirectory() as parent:
            empty_target = Path(parent) / "open-fault"
            with mock.patch.object(common.os, "open", side_effect=OSError("injected open fault")):
                with self.assertRaises(OSError):
                    with common.CompletionLastOutput(empty_target):
                        pass
            self.assertFalse(empty_target.exists())

            incomplete = Path(parent) / "incomplete"
            with self.assertRaises(RuntimeError):
                with common.CompletionLastOutput(incomplete) as transaction:
                    transaction.write_new("RESULT.bin", b"durable")
                    raise RuntimeError("injected post-member fault")
            self.assertEqual((incomplete / "RESULT.bin").read_bytes(), b"durable")
            self.assertTrue((incomplete / "RUN_STATE.json").is_file())
            self.assertFalse((incomplete / "COMPLETE.json").exists())

            completed = Path(parent) / "completed"
            with self.assertRaises(RuntimeError):
                with common.CompletionLastOutput(completed) as transaction:
                    member = transaction.write_new("RESULT.bin", b"durable")
                    transaction.complete(list(transaction.members), "ab" * 32)
                    self.assertEqual(member["sha256"], sha(b"durable"))
                    raise RuntimeError("injected post-completion fault")
            self.assertEqual((completed / "RESULT.bin").read_bytes(), b"durable")
            self.assertTrue((completed / "COMPLETE.json").is_file())

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
