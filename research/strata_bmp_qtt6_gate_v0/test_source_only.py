#!/usr/bin/env python3
"""Hostile source-only tests for the six-plane packet and bounded search."""

from __future__ import annotations

import struct
from pathlib import Path
import sys
import unittest
import zlib

import numpy as np


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import codec
from codec import (CodecError, Geometry, active_features, build_robdd,
                   decode_packet, descriptor_formula, encode_packet,
                   indices_to_planes, planes_to_indices, qtt_core_bit_count,
                   validate_distortion_table)
from search import exhaustive_small_indices


def geometry() -> Geometry:
    return Geometry(768, 2048, 0, 0, 16, 0, 256)


def bmp_model(rank: int = 1) -> dict:
    factors = []
    for level in range(6):
        U = np.zeros((16, rank), dtype=np.uint8)
        V = np.zeros((256, rank), dtype=np.uint8)
        if rank:
            U[:, 0] = (np.arange(16) >> (level % 4)) & 1
            V[:, 0] = (np.arange(256) >> ((level + 1) % 8)) & 1
        factors.append((U, V))
    return {"ranks": [rank] * 6, "factors": factors}


def recalc_crc(packet: bytes) -> bytes:
    body = packet[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


class SourceOnlyTests(unittest.TestCase):
    def test_six_plane_lsb_semantics(self):
        indices = np.arange(64, dtype=np.uint8)
        planes = indices_to_planes(indices)
        self.assertEqual(planes.shape, (6, 64))
        self.assertTrue(np.array_equal(planes_to_indices(planes), indices))
        self.assertEqual(planes[:, 37].tolist(), [1, 0, 1, 0, 0, 1])

    def test_four_level_table_rejected(self):
        with self.assertRaisesRegex(CodecError, "D\[i,0..63\]"):
            validate_distortion_table(np.zeros((4096, 4)), 4096)

    def test_mixed_radix_geometry(self):
        names, bits = active_features(geometry(), 0)
        self.assertEqual(bits.shape, (4096, 12))
        self.assertEqual(len(names), 12)
        with self.assertRaisesRegex(CodecError, "3\*2\^k"):
            active_features(Geometry(704, 2048, 0, 0, 16, 0, 256), 0)
        with self.assertRaisesRegex(CodecError, "role trit"):
            active_features(Geometry(768, 2048, 3, 0, 16, 0, 256), 0)

    def test_order_bank_bijective(self):
        hashes = []
        for order in range(4):
            _, bits = active_features(geometry(), order)
            self.assertEqual(np.unique(np.packbits(bits, axis=1), axis=0).shape[0],
                             4096)
            hashes.append(bits.tobytes())
        self.assertGreaterEqual(len(set(hashes)), 3)

    def test_bmp_packet_roundtrip_and_formula(self):
        packet = encode_packet(codec.FAMILY_BMP, 0, geometry(), bmp_model(), [])
        decoded = decode_packet(packet)
        self.assertEqual(decoded["completed_planes"].shape, (6, 4096))
        self.assertEqual(descriptor_formula(decoded)["total_physical_bits"],
                         len(packet) * 8)

    def test_crc_attack(self):
        packet = bytearray(encode_packet(codec.FAMILY_BMP, 0, geometry(),
                                         bmp_model(), []))
        packet[20] ^= 1
        with self.assertRaisesRegex(CodecError, "CRC32"):
            decode_packet(bytes(packet))

    def test_unsorted_exceptions_rejected(self):
        model = bmp_model(0)
        with self.assertRaisesRegex(CodecError, "sorted unique"):
            encode_packet(codec.FAMILY_BMP, 0, geometry(), model,
                          [(2, 1), (1, 2)])

    def test_redundant_exception_rejected(self):
        with self.assertRaisesRegex(CodecError, "redundant"):
            encode_packet(codec.FAMILY_BMP, 0, geometry(), bmp_model(0),
                          [(1, 0)])

    def test_exception_roundtrip(self):
        packet = encode_packet(codec.FAMILY_BMP, 0, geometry(), bmp_model(0),
                               [(0, 63), (4095, 17)])
        decoded = decode_packet(packet)
        self.assertEqual(decoded["indices"][[0, 4095]].tolist(), [63, 17])
        self.assertTrue(np.array_equal(decoded["completed_planes"],
                                       indices_to_planes(decoded["indices"])))

    def test_bmp_padding_attack(self):
        small = Geometry(6, 8, 0, 0, 2, 0, 4)
        factors = [(np.zeros((2, 1), np.uint8),
                    np.zeros((4, 1), np.uint8)) for _ in range(6)]
        packet = bytearray(encode_packet(codec.FAMILY_BMP, 0, small,
                                         {"ranks": [1] * 6,
                                          "factors": factors}, []))
        # Header (30), ranks (6), then first one-byte six-bit factor payload.
        packet[codec.HEADER.size + 6] |= 0x80
        packet = bytearray(recalc_crc(bytes(packet)))
        with self.assertRaisesRegex(CodecError, "tail"):
            decode_packet(bytes(packet), allow_small=True)

    def test_obdd_canonical_roundtrip(self):
        _, features = active_features(geometry(), 1)
        roots, diagrams = [], []
        for level in range(6):
            target = features[:, level % features.shape[1]]
            root, nodes = build_robdd(target, features)
            roots.append(root)
            diagrams.append(nodes)
        packet = encode_packet(codec.FAMILY_OBDD, 1, geometry(),
                               {"roots": roots, "nodes": diagrams}, [])
        decoded = decode_packet(packet)
        self.assertEqual(sum(map(len, decoded["model"]["nodes"])), 6)

    def test_obdd_node_cap(self):
        nodes = [[(0, 0, 1)] * 41 for _ in range(6)]
        with self.assertRaisesRegex(CodecError, "node cap"):
            encode_packet(codec.FAMILY_OBDD, 0, geometry(),
                          {"roots": [2] * 6, "nodes": nodes}, [])

    def test_qtt_packet_roundtrip(self):
        _, features = active_features(geometry(), 0)
        cores = []
        for _ in range(6):
            bits = np.zeros(qtt_core_bit_count(features.shape[1], 2), np.uint8)
            cores.append(bits)
        packet = encode_packet(codec.FAMILY_QTT, 0, geometry(),
                               {"ranks": [2] * 6, "cores": cores}, [])
        decoded = decode_packet(packet)
        self.assertEqual(decoded["indices"].max(), 0)
        self.assertEqual(descriptor_formula(decoded)["total_physical_bits"],
                         len(packet) * 8)

    def test_qtt_rank_cap(self):
        _, features = active_features(geometry(), 0)
        cores = [np.zeros(qtt_core_bit_count(features.shape[1], 2), np.uint8)
                 for _ in range(6)]
        with self.assertRaisesRegex(CodecError, "rank cap"):
            encode_packet(codec.FAMILY_QTT, 0, geometry(),
                          {"ranks": [3] * 6, "cores": cores}, [])

    def test_exact_small_exhaustive(self):
        table = np.full((2, 64), 10.0)
        table[0, 7] = 0.0
        table[1, 9] = 0.0
        result = exhaustive_small_indices(
            table, 0.1, lambda q: 4 if int(q[0]) == int(q[1]) else 12)
        self.assertEqual(result["evaluated"], 4096)
        # Independent brute force is deliberately repeated in the hostile test.
        expected = min(
            (float(table[0, a] + table[1, b] +
                   0.1 * (4 if a == b else 12)), a, b)
            for a in range(64) for b in range(64))
        self.assertEqual(result["indices"], [expected[1], expected[2]])
        self.assertAlmostEqual(result["objective"], expected[0])

    def test_no_cupy_initialized(self):
        self.assertNotIn("cupy", sys.modules)

    def test_no_payload_locators_in_core(self):
        for name in ("codec.py", "search.py", "run_source_free_fixture.py",
                     "test_source_only.py"):
            text = (PACKAGE / name).read_text(encoding="utf-8")
            self.assertNotIn("safetensors", text)
            self.assertNotIn("root@", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
