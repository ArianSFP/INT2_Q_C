#!/usr/bin/env python3
"""Payload-free hostile tests for the v4 packet and accounting contract."""

from __future__ import annotations

import hashlib
import inspect
import math
import struct
import sys
import unittest
import zlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import independent_decoder
import numeric_encoder
import packet_format
from packet_format import (
    HEADER_BYTES,
    N,
    PAGE_BYTES,
    PAYLOAD_BITS,
    RESERVOIR_BYTES,
    ContractError,
    ExpertGeometry,
    bits_to_payload,
    canonical_packet_reencode,
    frame_ledger,
    pack_reservoir,
    parse_expert_frame,
    parse_reservoir,
    payload_to_bits,
    qwen_frozen_ledgers,
    seed_pair,
)


REPO_ROOT = PACKAGE.parents[1]


def zero_frame(geometry: ExpertGeometry) -> bytes:
    packets = []
    for role in range(3):
        for tile in range(geometry.streams_per_role):
            packets.append(
                pack_reservoir(b"", 0, 1.0, geometry, role, tile, zero_tile=True)
            )
    return b"".join(packets)


def mutate_and_recrc(packet: bytes, offset: int, replacement: bytes) -> bytes:
    value = bytearray(packet)
    value[offset : offset + len(replacement)] = replacement
    struct.pack_into("<I", value, 124, zlib.crc32(value[:124]) & 0xFFFFFFFF)
    return bytes(value)


class PacketGrammarTests(unittest.TestCase):
    def test_exact_constants_and_qwen_rate(self) -> None:
        self.assertEqual(RESERVOIR_BYTES, 78_592)
        self.assertEqual(PAYLOAD_BITS, 627_712)
        self.assertEqual(PAYLOAD_BITS - packet_format.NOMINAL_BITS, 1_024)
        geometry = ExpertGeometry(768, 2048)
        self.assertEqual(geometry.role_values, 6 * N)
        self.assertEqual(geometry.records, 18)
        self.assertTrue(geometry.target_eligible)
        self.assertEqual(geometry.frame_bytes, 1_414_656)
        self.assertEqual(8 * geometry.frame_bytes * 128, geometry.values * 307)

    def test_nonzero_packet_roundtrip_and_canonical_bits(self) -> None:
        geometry = ExpertGeometry(N, 1)
        payload, logical_bits = bits_to_payload([1, 0, 1, 1, 0, 0, 1])
        packet = pack_reservoir(payload, logical_bits, 0.125, geometry, 1, 0)
        parsed = parse_reservoir(packet)
        self.assertEqual(parsed.payload, payload)
        self.assertEqual(parsed.logical_bits, logical_bits)
        self.assertEqual(payload_to_bits(payload, logical_bits), (1, 0, 1, 1, 0, 0, 1))
        self.assertEqual(
            canonical_packet_reencode(packet, [1, 0, 1, 1, 0, 0, 1]),
            packet,
        )

    def test_zero_tile_is_literal_and_self_canonical(self) -> None:
        geometry = ExpertGeometry(N, 1)
        packet = pack_reservoir(b"", 0, 99.0, geometry, 0, 0, zero_tile=True)
        parsed = parse_reservoir(packet)
        self.assertTrue(parsed.zero_tile)
        self.assertEqual(parsed.decoder_scale_fp32, 1.0)
        self.assertEqual(canonical_packet_reencode(packet, []), packet)

    def test_terminal_padding_and_capacity_fail_closed(self) -> None:
        geometry = ExpertGeometry(N, 1)
        with self.assertRaises(ContractError):
            pack_reservoir(b"\x01", 1, 1.0, geometry, 0, 0)

        def too_many_bits():
            for _ in range(PAYLOAD_BITS + 1):
                yield 0

        with self.assertRaises(ContractError):
            bits_to_payload(too_many_bits())

    def test_payload_corruption_fill_and_truncation_rejected(self) -> None:
        geometry = ExpertGeometry(N, 1)
        payload, logical_bits = bits_to_payload([1] * 64)
        packet = pack_reservoir(payload, logical_bits, 0.5, geometry, 0, 0)
        corrupted = bytearray(packet)
        corrupted[HEADER_BYTES] ^= 0x80
        with self.assertRaises(ContractError):
            parse_reservoir(bytes(corrupted))
        fill = bytearray(packet)
        fill[-1] = 1
        with self.assertRaises(ContractError):
            parse_reservoir(bytes(fill))
        with self.assertRaises(ContractError):
            parse_reservoir(packet[:-1])
        with self.assertRaises(ContractError):
            parse_reservoir(packet + b"\0")

    def test_header_seed_scale_flags_and_crc_rejected(self) -> None:
        geometry = ExpertGeometry(N, 1)
        payload, logical_bits = bits_to_payload([0, 1] * 16)
        packet = pack_reservoir(payload, logical_bits, 0.25, geometry, 0, 0)
        with self.assertRaises(ContractError):
            parse_reservoir(mutate_and_recrc(packet, 36, b"\0\0\0\0"))
        with self.assertRaises(ContractError):
            parse_reservoir(mutate_and_recrc(packet, 48, struct.pack("<f", math.nan)))
        flags = struct.unpack_from("<H", packet, 18)[0] ^ 0x80
        with self.assertRaises(ContractError):
            parse_reservoir(mutate_and_recrc(packet, 18, struct.pack("<H", flags)))
        bad_crc = bytearray(packet)
        bad_crc[124] ^= 1
        with self.assertRaises(ContractError):
            parse_reservoir(bytes(bad_crc))

    def test_seed_is_shape_role_coordinate_only(self) -> None:
        base = seed_pair(0, 768, 2048, 0)
        self.assertEqual(base, seed_pair(0, 768, 2048, 0))
        self.assertNotEqual(base, seed_pair(1, 768, 2048, 0))
        self.assertNotEqual(base, seed_pair(0, 769, 2048, 0))
        self.assertNotEqual(base, seed_pair(0, 768, 2048, 1))


class FrameAndTailTests(unittest.TestCase):
    def test_target_eligible_frame_is_self_describing(self) -> None:
        geometry = ExpertGeometry(N, 1)
        frame = zero_frame(geometry)
        parsed = parse_expert_frame(frame)
        self.assertEqual(parsed.geometry, geometry)
        self.assertEqual(len(parsed.records), 3)
        self.assertTrue(parsed.target_eligible)
        self.assertEqual(parsed.frame_sha256, hashlib.sha256(frame).hexdigest())
        self.assertEqual(8 * len(frame) * 128, geometry.values * 307)

    def test_tail_is_zero_padded_compatibility_not_target_cell(self) -> None:
        geometry = ExpertGeometry(1, N + 17)
        frame = zero_frame(geometry)
        parsed = parse_expert_frame(frame)
        self.assertFalse(parsed.target_eligible)
        self.assertEqual(geometry.streams_per_role, 2)
        tails = [row for row in parsed.records if row.tile_ordinal == 1]
        self.assertEqual(len(tails), 3)
        self.assertTrue(all(row.padded_tail and row.valid_values == 17 for row in tails))
        self.assertGreater(8 * len(frame) / geometry.values, 307 / 128)

    def test_encoder_tail_extraction_is_literal(self) -> None:
        geometry = ExpertGeometry(1, N + 3)
        raw = bytes(2 * N) + b"\x01\0\x02\0\x03\0"
        tile, valid, zero = numeric_encoder.canonical_padded_tile(raw, geometry, 1)
        self.assertEqual(valid, 3)
        self.assertFalse(zero)
        self.assertEqual(tile[:6], raw[-6:])
        self.assertEqual(tile[6:], bytes(2 * (N - 3)))
        negative_zero = bytes(2 * N) + b"\0\x80" * 3
        _, _, zero = numeric_encoder.canonical_padded_tile(negative_zero, geometry, 1)
        self.assertTrue(zero)

    def test_reordered_missing_and_extra_records_rejected(self) -> None:
        geometry = ExpertGeometry(N, 1)
        frame = zero_frame(geometry)
        records = [
            frame[index : index + RESERVOIR_BYTES]
            for index in range(0, len(frame), RESERVOIR_BYTES)
        ]
        with self.assertRaises(ContractError):
            parse_expert_frame(records[1] + records[0] + records[2])
        with self.assertRaises(ContractError):
            parse_expert_frame(b"".join(records[:-1]))
        with self.assertRaises(ContractError):
            parse_expert_frame(frame + records[0])
        with self.assertRaises(ContractError):
            parse_expert_frame(frame + b"\0")

    def test_geometry_caps_and_tiny_shapes_are_explicitly_noneligible(self) -> None:
        with self.assertRaises(ContractError):
            ExpertGeometry(0, 1)
        with self.assertRaises(ContractError):
            ExpertGeometry(packet_format.MAX_DIMENSION + 1, 1)
        tiny = ExpertGeometry(1, 1)
        self.assertFalse(tiny.target_eligible)
        self.assertGreater(8 * tiny.frame_bytes / tiny.values, 2.5)


class ReadAndModelLedgerTests(unittest.TestCase):
    def test_one_pass_and_worst_page_ledgers(self) -> None:
        geometry = ExpertGeometry(768, 2048)
        aligned = frame_ledger(geometry, start_offset_mod_page=0, compressed_passes=1)
        hostile = frame_ledger(
            geometry, start_offset_mod_page=PAGE_BYTES - 1, compressed_passes=1
        )
        second = frame_ledger(geometry, compressed_passes=2)
        self.assertEqual(aligned["repeated_byte_amplification"], 1.0)
        self.assertLess(hostile["unique_page_amplification"], 2.0)
        self.assertEqual(second["repeated_byte_amplification"], 2.0)
        self.assertFalse(second["one_pass_schedule"])

    def test_frozen_final_topology_requires_buffered_coarse_state(self) -> None:
        ledger = qwen_frozen_ledgers()
        final = ledger["frozen_final_planning_topology_not_implemented_here"]
        self.assertEqual(final["cold_page_amplification"], 73 / 72)
        self.assertEqual(final["forbidden_second_private_frame_pass_pages"], 724)
        self.assertEqual(
            final["forbidden_second_private_frame_pass_amplification"], 724 / 360
        )
        self.assertGreater(final["forbidden_second_private_frame_pass_amplification"], 2.0)
        self.assertFalse(final["second_private_frame_pass_satisfies_strict_below_2x"])

    def test_no_trained_model_or_identity_selector_in_format(self) -> None:
        source = inspect.getsource(packet_format)
        self.assertNotIn("checkpoint_id", source)
        self.assertNotIn("expert_ordinal", source)
        self.assertNotIn("layer_ordinal", source)
        self.assertEqual(qwen_frozen_ledgers()["coarse"]["model_bytes"], 0)

    def test_independent_decoder_does_not_import_encoder(self) -> None:
        source = inspect.getsource(independent_decoder)
        self.assertNotIn("import numeric_encoder", source)
        self.assertNotIn("from numeric_encoder", source)
        self.assertIn("compressed_frame_passes", source)
        self.assertIn("residual_f64_sha256", source)

    def test_numeric_dependency_sources_are_exactly_pinned(self) -> None:
        rows = (
            (
                REPO_ROOT / numeric_encoder.ENCODER_RELATIVE,
                numeric_encoder.ENCODER_BYTES,
                numeric_encoder.ENCODER_SHA256,
            ),
            (
                REPO_ROOT / numeric_encoder.DECODER_RELATIVE,
                numeric_encoder.DECODER_BYTES,
                numeric_encoder.DECODER_SHA256,
            ),
        )
        for path, size, digest in rows:
            raw = path.read_bytes()
            self.assertEqual(len(raw), size)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        self.assertEqual(
            independent_decoder.DECODER_SHA256, numeric_encoder.DECODER_SHA256
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
