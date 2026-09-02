#!/usr/bin/env python3
"""Hostile source-only regressions for hardened BMP/OBDD/QTT6 v1."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import struct
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import zlib

import numpy as np


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import codec
import cupy_backend
import run_source_free_fixture as fixture
import search
from codec import (CodecError, Geometry, active_features, build_robdd,
                   canonical_gf2_factor, canonical_qtt, decode_packet,
                   descriptor_formula, encode_packet, indices_to_planes,
                   planes_to_indices, qtt_core_bit_count, qtt_plane, qtt_shapes,
                   validate_distortion_table)
from production_hooks import held_source_only_hooks
from search import CompleteRateCap, exact_workspace_plan, exhaustive_small_indices


def geometry() -> Geometry:
    return Geometry(704, 2304, 0, 320, 16, 1024, 256)


def zero_bmp_model() -> dict:
    return {"ranks": [0] * 6,
            "factors": [(np.zeros((16, 0), np.uint8),
                         np.zeros((256, 0), np.uint8)) for _ in range(6)]}


def canonical_bmp_model() -> dict:
    factors = []
    for level in range(6):
        u = ((np.arange(16) >> (level % 4)) & 1).astype(np.uint8)
        v = ((np.arange(256) >> ((level + 1) % 8)) & 1).astype(np.uint8)
        factors.append(canonical_gf2_factor(np.outer(u, v).reshape(-1), 16, 256))
    return {"ranks": [u.shape[1] for u, _ in factors], "factors": factors}


def canonical_qtt_model(g=None, order=0) -> dict:
    g = g or geometry()
    _, features = active_features(g, order)
    rank_vectors, cores = [], []
    for level in range(6):
        target = (features[:, level % features.shape[1]] ^
                  (features[:, (level + 1) % features.shape[1]] &
                   features[:, (level + 4) % features.shape[1]])).astype(np.uint8)
        canonical = canonical_qtt(target, features)
        assert canonical is not None
        rank_vectors.append(canonical[0])
        cores.append(canonical[1])
    return {"rank_vectors": rank_vectors, "cores": cores}


def recalc_crc(packet: bytes) -> bytes:
    body = packet[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


class HardenedSourceOnlyTests(unittest.TestCase):
    def test_six_plane_exact_semantics_and_literal_d64(self):
        indices = np.arange(64, dtype=np.uint8)
        planes = indices_to_planes(indices)
        self.assertTrue(np.array_equal(planes_to_indices(planes), indices))
        self.assertEqual(planes[:, 37].tolist(), [1, 0, 1, 0, 0, 1])
        with self.assertRaisesRegex(CodecError, r"D\[i,0..63\]"):
            validate_distortion_table(np.zeros((4096, 4)), 4096)

    def test_variable_mixed_radix_swiglu_geometry(self):
        for g in (geometry(), Geometry(11008, 4096, 1, 4096, 8, 2048, 512),
                  Geometry(5760, 3584, 2, 5120, 16, 3072, 256)):
            for order in range(codec.ORDER_BANK_SIZE):
                _, bits = active_features(g, order)
                self.assertEqual(bits.shape, (4096, 12))
                self.assertEqual(np.unique(np.packbits(bits, axis=1), axis=0).shape[0],
                                 4096)
        with self.assertRaisesRegex(CodecError, "role trit"):
            Geometry(704, 2304, 3, 0, 16, 0, 256).validate()

    def test_uint16_geometry_boundaries_fail_as_codec_error(self):
        boundary = Geometry(65535, 65535, 2, 65504, 16, 65024, 256)
        packet = encode_packet(codec.FAMILY_BMP, 0, boundary,
                               zero_bmp_model(), [])
        self.assertEqual(decode_packet(packet)["geometry"], boundary)
        for g in (Geometry(65536, 2048, 0, 0, 16, 0, 256),
                  Geometry(768, 65536, 0, 0, 16, 0, 256),
                  Geometry(768, 2048, 0, 65536, 16, 0, 256),
                  Geometry(768, 2048, 0, 0, 16, 65536, 256)):
            with self.assertRaisesRegex(CodecError, "uint16 packet ABI"):
                encode_packet(codec.FAMILY_BMP, 0, g, zero_bmp_model(), [])

    def test_bmp_minimum_rank_roundtrip(self):
        packet = encode_packet(codec.FAMILY_BMP, 0, geometry(),
                               canonical_bmp_model(), [])
        decoded = decode_packet(packet)
        self.assertEqual(descriptor_formula(decoded)["total_physical_bits"],
                         len(packet) * 8)

    def test_bmp_zero_rank_inflation_rejected(self):
        inflated = {"ranks": [1] * 6,
                    "factors": [(np.zeros((16, 1), np.uint8),
                                 np.zeros((256, 1), np.uint8)) for _ in range(6)]}
        with self.assertRaisesRegex(CodecError, "canonical minimum-rank BMP"):
            encode_packet(codec.FAMILY_BMP, 0, geometry(), inflated, [])

    def test_bmp_column_gauge_swap_rejected(self):
        rr = np.arange(16)
        cc = np.arange(256)
        plane = (np.outer((rr >> 0) & 1, (cc >> 0) & 1) ^
                 np.outer((rr >> 1) & 1, (cc >> 1) & 1)).astype(np.uint8)
        u, v = canonical_gf2_factor(plane.reshape(-1), 16, 256)
        self.assertEqual(u.shape[1], 2)
        swapped = (u[:, ::-1].copy(), v[:, ::-1].copy())
        model = {"ranks": [2] * 6, "factors": [swapped] * 6}
        with self.assertRaisesRegex(CodecError, "canonical minimum-rank BMP"):
            encode_packet(codec.FAMILY_BMP, 0, geometry(), model, [])

    def test_bmp_gauge_transform_rejected(self):
        rr = np.arange(16)
        cc = np.arange(256)
        plane = (np.outer((rr >> 0) & 1, (cc >> 0) & 1) ^
                 np.outer((rr >> 1) & 1, (cc >> 1) & 1)).astype(np.uint8)
        u, v = canonical_gf2_factor(plane.reshape(-1), 16, 256)
        transform = np.asarray([[1, 1], [0, 1]], dtype=np.uint8)
        inverse_t = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
        alias_u = ((u.astype(np.uint16) @ transform.astype(np.uint16)) & 1).astype(np.uint8)
        alias_v = ((v.astype(np.uint16) @ inverse_t.astype(np.uint16)) & 1).astype(np.uint8)
        self.assertTrue(np.array_equal(codec.bmp_plane(alias_u, alias_v),
                                       plane.reshape(-1)))
        with self.assertRaisesRegex(CodecError, "canonical minimum-rank BMP"):
            encode_packet(codec.FAMILY_BMP, 0, geometry(),
                          {"ranks": [2] * 6,
                           "factors": [(alias_u, alias_v)] * 6}, [])

    def test_qtt_canonical_roundtrip(self):
        packet = encode_packet(codec.FAMILY_QTT, 0, geometry(),
                               canonical_qtt_model(), [])
        decoded = decode_packet(packet)
        self.assertEqual(descriptor_formula(decoded)["total_physical_bits"],
                         len(packet) * 8)

    def test_qtt_zero_has_one_representation(self):
        zero = {"rank_vectors": [None] * 6,
                "cores": [np.zeros(0, np.uint8) for _ in range(6)]}
        packet = encode_packet(codec.FAMILY_QTT, 0, geometry(), zero, [])
        self.assertEqual(int(decode_packet(packet)["indices"].max()), 0)
        _, features = active_features(geometry(), 0)
        ranks = (1,) * (features.shape[1] - 1)
        inflated = {"rank_vectors": [ranks] * 6,
                    "cores": [np.zeros(qtt_core_bit_count(features.shape[1], ranks),
                                       np.uint8) for _ in range(6)]}
        with self.assertRaisesRegex(CodecError, "canonical minimum-rank QTT"):
            encode_packet(codec.FAMILY_QTT, 0, geometry(), inflated, [])

    def test_qtt_inflated_state_path_rejected(self):
        _, features = active_features(geometry(), 0)
        ranks = (2,) * (features.shape[1] - 1)
        bits = np.zeros(qtt_core_bit_count(features.shape[1], ranks), np.uint8)
        offset = 0
        for shape in qtt_shapes(features.shape[1], ranks):
            core = bits[offset:offset + int(np.prod(shape))].reshape(shape)
            core[0, :, 0] = 1
            offset += core.size
        self.assertTrue(qtt_plane(bits, features, ranks).all())
        with self.assertRaisesRegex(CodecError, "canonical minimum-rank QTT"):
            encode_packet(codec.FAMILY_QTT, 0, geometry(),
                          {"rank_vectors": [ranks] * 6, "cores": [bits] * 6}, [])

    def test_qtt_unused_rank_mask_rejected(self):
        packet = bytearray(encode_packet(codec.FAMILY_QTT, 0, geometry(),
                                         canonical_qtt_model(), []))
        _, features = active_features(geometry(), 0)
        bad_code = 1 + (1 << (features.shape[1] - 1))
        struct.pack_into("<H", packet, codec.HEADER.size, bad_code)
        packet = bytearray(recalc_crc(bytes(packet)))
        with self.assertRaisesRegex(CodecError, "unused rank bits"):
            decode_packet(bytes(packet))

    def test_obdd_still_reduced_and_canonical(self):
        _, features = active_features(geometry(), 1)
        roots, diagrams = [], []
        for level in range(6):
            root, nodes = build_robdd(features[:, level], features)
            roots.append(root); diagrams.append(nodes)
        packet = encode_packet(codec.FAMILY_OBDD, 1, geometry(),
                               {"roots": roots, "nodes": diagrams}, [])
        self.assertEqual(sum(map(len, decode_packet(packet)["model"]["nodes"])), 6)

    def test_crc_tail_and_exception_canonicality(self):
        packet = bytearray(encode_packet(codec.FAMILY_BMP, 0, geometry(),
                                         zero_bmp_model(), [(0, 63)]))
        packet[20] ^= 1
        with self.assertRaisesRegex(CodecError, "CRC32"):
            decode_packet(bytes(packet))
        with self.assertRaisesRegex(CodecError, "sorted unique"):
            encode_packet(codec.FAMILY_BMP, 0, geometry(), zero_bmp_model(),
                          [(2, 1), (1, 2)])
        with self.assertRaisesRegex(CodecError, "redundant"):
            encode_packet(codec.FAMILY_BMP, 0, geometry(), zero_bmp_model(), [(0, 0)])

    def test_complete_rate_cap_exact_boundaries(self):
        cap = CompleteRateCap(total_weights=4096, outer_bits=1000,
                              already_committed_bits=2000,
                              reserved_future_bits=3000)
        self.assertEqual(cap.min_total_bits, 8807)
        self.assertEqual(cap.max_total_bits, 10240)
        self.assertEqual(cap.available_packet_bits, 4240)
        self.assertTrue(cap.admit_packet(4240))
        self.assertFalse(cap.admit_packet(4241))
        with self.assertRaisesRegex(CodecError, "retains reserved"):
            cap.assert_complete(4240)
        final = CompleteRateCap(total_weights=4096, outer_bits=1000,
                                already_committed_bits=5000)
        self.assertEqual(final.assert_complete(4240)["total_bits"], 10240)
        with self.assertRaisesRegex(CodecError, "below 2.15"):
            CompleteRateCap(4096, 0).assert_complete(1)

    def test_search_requires_complete_rate_cap(self):
        table = np.zeros((4096, 64), np.float64)
        with self.assertRaisesRegex(CodecError, "explicit complete-rate"):
            search.search_bank(table, geometry(), 0.0, None)
        with self.assertRaisesRegex(CodecError, "nonpacket fields exceed"):
            search.search_bank(table, geometry(), 0.0,
                               CompleteRateCap(4096, 10241))

    def test_exact_workspace_named_ledger(self):
        plan = exact_workspace_plan(geometry())
        self.assertEqual(len({name for name, _ in plan}), len(plan))
        self.assertEqual(sum(size for _, size in plan), 2314240)
        ledger = search.WorkspaceLedger(cap_bytes=10)
        ledger.own("x", 10)
        with self.assertRaisesRegex(CodecError, "workspace cap"):
            ledger.own("y", 1)

    def test_exact_small_exhaustive(self):
        table = np.full((2, 64), 10.0)
        table[0, 7] = 0.0; table[1, 9] = 0.0
        result = exhaustive_small_indices(table, 0.1,
                                          lambda q: 4 if q[0] == q[1] else 12)
        self.assertEqual(result["evaluated"], 4096)

    def test_production_hooks_fail_closed(self):
        with self.assertRaisesRegex(CodecError, "unbound production hooks"):
            held_source_only_hooks().authorize()

    def test_fake_cupy_facade_rejected(self):
        fake = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, {"cupy": fake}):
            with self.assertRaisesRegex(RuntimeError, "real imported module"):
                cupy_backend.require_cupy()

    def test_fresh_process_runner_contract_is_literal(self):
        launcher = (PACKAGE / "run_cupy_smoke.py").read_text("utf-8")
        worker = (PACKAGE / "cupy_worker.py").read_text("utf-8")
        for phrase in ('sys.executable, "-I", "-B"', "nonce/PID authentication",
                       "worker_stdout_sha256"):
            self.assertIn(phrase, launcher)
        for phrase in ("active_device_id", "payload_authority"):
            self.assertIn(phrase, (PACKAGE / "cupy_backend.py").read_text("utf-8") + worker)

    def test_fixture_reuses_same_search_and_cap(self):
        calls = []
        def fake_search(table, g, lambda_bit, rate_cap):
            calls.append((np.asarray(table).shape, g, lambda_bit, rate_cap))
            return {"tag": len(calls), "caps": {},
                    "complete_rate_cap": {}, "workspace": {}}
        with mock.patch.object(fixture, "search_bank", side_effect=fake_search), \
             mock.patch.object(fixture, "summarize", return_value={"ok": True}), \
             redirect_stdout(io.StringIO()):
            fixture.main()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], (4096, 64))
        self.assertIs(calls[0][3], calls[1][3])

    def test_no_payload_locator_or_authority(self):
        for name in ("codec.py", "search.py", "cupy_backend.py",
                     "production_hooks.py", "run_source_free_fixture.py"):
            text = (PACKAGE / name).read_text("utf-8")
            self.assertNotIn("safetensors", text)
            self.assertNotIn("root@", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
