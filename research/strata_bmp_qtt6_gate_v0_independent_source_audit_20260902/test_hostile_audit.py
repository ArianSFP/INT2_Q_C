#!/usr/bin/env python3
"""Independent hostile tests for the frozen source-only BMP/QTT6 mechanism."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest import mock

import numpy as np


SOURCE = Path(os.environ["STRATA_BMP_QTT6_FROZEN_SOURCE"]).resolve()
sys.path.insert(0, str(SOURCE))
for _name in ("codec", "search", "cupy_backend", "run_source_free_fixture"):
    sys.modules.pop(_name, None)
codec = importlib.import_module("codec")
search = importlib.import_module("search")
cupy_backend = importlib.import_module("cupy_backend")
fixture = importlib.import_module("run_source_free_fixture")


def geometry() -> object:
    return codec.Geometry(768, 2048, 2, 512, 16, 512, 256)


def bmp_model(rank: int) -> dict:
    factors = []
    for _ in range(6):
        factors.append((np.zeros((16, rank), dtype=np.uint8),
                        np.zeros((256, rank), dtype=np.uint8)))
    return {"ranks": [rank] * 6, "factors": factors}


def qtt_model(rank: int, g=None, order_id: int = 0) -> dict:
    if g is None:
        g = geometry()
    _, features = codec.active_features(g, order_id)
    count = codec.qtt_core_bit_count(features.shape[1], rank)
    return {"ranks": [rank] * 6,
            "cores": [np.zeros(count, dtype=np.uint8) for _ in range(6)]}


class FrozenMechanismAudit(unittest.TestCase):
    def test_exact_six_completed_plane_semantics(self):
        indices = np.arange(64, dtype=np.uint8)
        planes = codec.indices_to_planes(indices)
        self.assertEqual(planes.shape, (6, 64))
        self.assertTrue(np.array_equal(codec.planes_to_indices(planes), indices))
        independently = sum(
            planes[level].astype(np.uint16) * (1 << level) for level in range(6)
        )
        self.assertTrue(np.array_equal(independently.astype(np.uint8), indices))
        self.assertEqual(planes[:, 37].tolist(), [1, 0, 1, 0, 0, 1])

    def test_distortion_abi_is_literal_n_by_64(self):
        table = np.arange(4096 * 64, dtype=np.float64).reshape(4096, 64)
        self.assertIs(codec.validate_distortion_table(table, 4096), table)
        for invalid in (np.zeros((4096, 4)), np.zeros((4095, 64)),
                        np.full((4096, 64), np.nan),
                        -np.ones((4096, 64))):
            with self.assertRaises(codec.CodecError):
                codec.validate_distortion_table(invalid, 4096)

    def test_mixed_radix_coordinate_bijection_across_shape_family(self):
        cases = [
            codec.Geometry(768, 2048, 0, 0, 16, 0, 256),
            codec.Geometry(768, 2048, 1, 256, 16, 512, 256),
            codec.Geometry(768, 2048, 2, 512, 16, 1536, 256),
            codec.Geometry(768, 2048, 0, 0, 512, 0, 8),
            codec.Geometry(1536, 4096, 1, 512, 8, 2048, 512),
            codec.Geometry(96, 512, 2, 64, 8, 0, 512),
        ]
        for g in cases:
            for order_id in range(codec.ORDER_BANK_SIZE):
                names, bits = codec.active_features(g, order_id)
                self.assertEqual(bits.shape, (4096, 12))
                self.assertEqual(len(names), 12)
                packed = np.packbits(bits, axis=1, bitorder="little")
                self.assertEqual(np.unique(packed, axis=0).shape[0], 4096)

    def test_portability_is_explicitly_not_arbitrary_swiglu_width(self):
        with self.assertRaisesRegex(codec.CodecError, r"3\*2\^k"):
            codec.Geometry(704, 2048, 0, 0, 16, 0, 256).validate()
        with self.assertRaisesRegex(codec.CodecError, "power of two"):
            codec.Geometry(768, 2304, 0, 0, 16, 0, 256).validate()

    def test_geometry_validation_exceeds_uint16_packet_abi(self):
        # This is an expected audit exposure: validate() accepts the shape, but
        # HEADER's unsigned-short field cannot serialize it fail-closed as a
        # CodecError. Production must bound rows/cols before payload authority.
        g = codec.Geometry(768, 65536, 0, 0, 16, 0, 256)
        g.validate()
        with self.assertRaises(struct.error):
            codec.encode_packet(codec.FAMILY_BMP, 0, g, bmp_model(0), [])
        g2 = codec.Geometry(98304, 2048, 0, 0, 16, 0, 256)
        g2.validate()
        with self.assertRaises(struct.error):
            codec.encode_packet(codec.FAMILY_BMP, 0, g2, bmp_model(0), [])

    def test_bmp_packet_roundtrip_and_exact_formula(self):
        for rank in (0, 1, 2, 4):
            packet = codec.encode_packet(
                codec.FAMILY_BMP, 0, geometry(), bmp_model(rank), []
            )
            decoded = codec.decode_packet(packet)
            formula = codec.descriptor_formula(decoded)
            expected = 34 + 6 + 6 * (
                (rank * (16 + 256) + 7) // 8
            )
            self.assertEqual(len(packet), expected)
            self.assertEqual(formula["total_physical_bits"], 8 * len(packet))
            self.assertEqual(decoded["completed_planes"].shape, (6, 4096))

    def test_bmp_semantic_aliases_survive_canonical_reencode(self):
        rank0 = codec.encode_packet(
            codec.FAMILY_BMP, 0, geometry(), bmp_model(0), []
        )
        rank1 = codec.encode_packet(
            codec.FAMILY_BMP, 0, geometry(), bmp_model(1), []
        )
        self.assertNotEqual(rank0, rank1)
        self.assertTrue(np.array_equal(
            codec.decode_packet(rank0)["indices"],
            codec.decode_packet(rank1)["indices"],
        ))
        self.assertEqual(codec.decode_packet(rank0)["indices"].max(), 0)

    def test_bmp_column_gauge_alias_same_rate(self):
        first = bmp_model(2)
        for level, (u, v) in enumerate(first["factors"]):
            u[:, 0] = (np.arange(16) >> (level % 4)) & 1
            u[:, 1] = (np.arange(16) >> ((level + 1) % 4)) & 1
            v[:, 0] = (np.arange(256) >> (level % 8)) & 1
            v[:, 1] = (np.arange(256) >> ((level + 2) % 8)) & 1
        second = {"ranks": [2] * 6, "factors": [
            (u[:, ::-1].copy(), v[:, ::-1].copy())
            for u, v in first["factors"]
        ]}
        p1 = codec.encode_packet(codec.FAMILY_BMP, 0, geometry(), first, [])
        p2 = codec.encode_packet(codec.FAMILY_BMP, 0, geometry(), second, [])
        self.assertNotEqual(p1, p2)
        self.assertEqual(len(p1), len(p2))
        self.assertTrue(np.array_equal(
            codec.decode_packet(p1)["indices"],
            codec.decode_packet(p2)["indices"],
        ))

    def test_qtt_roundtrip_formula_and_rank_alias(self):
        packets = []
        for rank in (1, 2):
            packet = codec.encode_packet(
                codec.FAMILY_QTT, 0, geometry(), qtt_model(rank), []
            )
            decoded = codec.decode_packet(packet)
            self.assertEqual(
                codec.descriptor_formula(decoded)["total_physical_bits"],
                len(packet) * 8,
            )
            self.assertEqual(decoded["indices"].max(), 0)
            packets.append(packet)
        self.assertNotEqual(packets[0], packets[1])
        self.assertTrue(np.array_equal(
            codec.decode_packet(packets[0])["indices"],
            codec.decode_packet(packets[1])["indices"],
        ))

    def test_obdd_reduction_rejects_unreachable_semantic_alias(self):
        roots = [2] + [0] * 5
        diagrams = [[(0, 0, 1), (1, 0, 1)]] + [[] for _ in range(5)]
        with self.assertRaisesRegex(codec.CodecError, "canonical reduced"):
            codec.encode_packet(
                codec.FAMILY_OBDD, 0, geometry(),
                {"roots": roots, "nodes": diagrams}, [],
            )

    def test_exception_threshold_is_exact_24_physical_bits(self):
        distortion = np.full((4096, 64), 100.0, dtype=np.float64)
        distortion[:, 0] = 0.0
        distortion[0, 0], distortion[0, 5] = 4.0, 0.0
        distortion[1, 0], distortion[1, 6] = 3.0, 0.0
        exceptions = search.add_joint_exceptions(
            distortion, np.zeros(4096, dtype=np.uint8), lambda_bit=0.125
        )
        self.assertEqual(exceptions, [(0, 5)])
        packet = codec.encode_packet(
            codec.FAMILY_BMP, 0, geometry(), bmp_model(0), exceptions
        )
        base_packet = codec.encode_packet(
            codec.FAMILY_BMP, 0, geometry(), bmp_model(0), []
        )
        self.assertEqual(len(packet) - len(base_packet), 3)

    def test_objective_uses_decoded_packet_bits_and_literal_d_table(self):
        distortion = np.full((4096, 64), 9.0, dtype=np.float64)
        distortion[:, 0] = np.arange(4096, dtype=np.float64) / 4096.0
        packet = codec.encode_packet(
            codec.FAMILY_BMP, 0, geometry(), bmp_model(0), []
        )
        value, metrics = search.objective(distortion, packet, 0.25)
        independent_sse = float(distortion[:, 0].sum(dtype=np.float64))
        self.assertEqual(metrics["physical_bits"], len(packet) * 8)
        self.assertEqual(metrics["sse"], independent_sse)
        self.assertEqual(value, independent_sse + 0.25 * len(packet) * 8)

    def test_conditional_plane_costs_use_other_five_completed_bits(self):
        rng = np.random.default_rng(91)
        table = rng.random((17, 64))
        indices = rng.integers(0, 64, size=17, dtype=np.uint8)
        rows = np.arange(17)
        for level in range(6):
            c0, c1 = search.conditional_plane_costs(table, indices, level)
            clear = indices & np.uint8(63 ^ (1 << level))
            self.assertTrue(np.array_equal(c0, table[rows, clear]))
            self.assertTrue(np.array_equal(
                c1, table[rows, clear | np.uint8(1 << level)]
            ))

    def test_exact_small_oracle_really_visits_64_power_n(self):
        table = np.full((3, 64), 5.0)
        table[0, 7], table[1, 11], table[2, 63] = 0.0, 0.0, 0.0
        result = search.exhaustive_small_indices(table, 0.0, lambda _: 0)
        self.assertEqual(result["evaluated"], 64 ** 3)
        self.assertEqual(result["indices"], [7, 11, 63])

    def test_packet_bound_arithmetic_and_missing_complete_rate_gate(self):
        # Independent arithmetic for the documented N=4096 mechanism bounds.
        self.assertEqual(34 + 6, 40)
        self.assertEqual(34 + 6 * 4, 58)
        self.assertEqual(34 + 6 + 6 * 136 + 64 * 3, 1048)
        self.assertEqual(34 + 6 * 4 + 240 * 5 + 64 * 3, 1450)
        self.assertEqual(34 + 6 + 6 * 11 + 64 * 3, 298)
        self.assertAlmostEqual(1450 * 8 / 4096, 2.83203125)
        design = json.loads((SOURCE / "design_lock.json").read_text("utf-8"))
        self.assertFalse(design["packet"]["complete_production_strata_framing_present"])
        self.assertFalse(design["payload_authority"]["qwen"])

    def test_hard_count_caps_fail_closed(self):
        with self.assertRaisesRegex(codec.CodecError, "exception cap"):
            codec.encode_packet(
                codec.FAMILY_BMP, 0, geometry(), bmp_model(0),
                [(i, 1) for i in range(65)],
            )
        too_many = [[(0, 0, 1)] * 41 for _ in range(6)]
        with self.assertRaisesRegex(codec.CodecError, "node cap"):
            codec.encode_packet(
                codec.FAMILY_OBDD, 0, geometry(),
                {"roots": [2] * 6, "nodes": too_many}, [],
            )
        with self.assertRaisesRegex(codec.CodecError, "rank cap"):
            codec.encode_packet(
                codec.FAMILY_QTT, 0, geometry(),
                {"ranks": [3] * 6,
                 "cores": [np.zeros(2, dtype=np.uint8) for _ in range(6)]},
                [],
            )

    def test_search_and_workspace_caps_are_active_guards(self):
        counter = search.Counter()
        with mock.patch.object(search, "MAX_SEARCH_EVALUATIONS", 0):
            with self.assertRaisesRegex(codec.CodecError, "evaluation cap"):
                counter.add()
        table = np.zeros((4096, 64), dtype=np.float64)
        with mock.patch.object(search, "MAX_WORKSPACE_BYTES", 1):
            with self.assertRaisesRegex(codec.CodecError, "workspace cap"):
                search.search_bank(table, geometry(), 0.0)

    def test_gaussian_fixture_reuses_identical_search_callable(self):
        calls = []

        def fake_search(table, g, lambda_bit):
            calls.append((np.asarray(table).copy(), g, lambda_bit))
            return {"tag": len(calls)}

        with mock.patch.object(fixture, "search_bank", side_effect=fake_search), \
             mock.patch.object(fixture, "summarize", return_value={"ok": True}), \
             redirect_stdout(io.StringIO()):
            fixture.main()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0].shape, (4096, 64))
        self.assertEqual(calls[1][0].shape, (4096, 64))
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][2], calls[1][2])

    def test_gaussian_fixture_moments_are_empirically_matched(self):
        g = geometry()
        source, _, _ = fixture.synthetic_source(g)
        rng = np.random.default_rng(0x5A17A6)
        gaussian = rng.standard_normal(g.count).astype(np.float64)
        gaussian = (gaussian - gaussian.mean()) / gaussian.std()
        control = gaussian * source.std() + source.mean()
        self.assertLessEqual(abs(float(control.mean() - source.mean())), 1e-14)
        self.assertLessEqual(abs(float(control.std() - source.std())), 1e-14)

    def test_cupy_import_identity_can_be_spoofed(self):
        fake = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
            )
        )
        with mock.patch.dict(sys.modules, {"cupy": fake}):
            self.assertIs(cupy_backend.require_cupy(), fake)

    def test_cupy_is_only_optional_generated_primitive_smoke(self):
        search_text = (SOURCE / "search.py").read_text(encoding="utf-8")
        smoke_text = (SOURCE / "run_cupy_smoke.py").read_text(encoding="utf-8")
        self.assertNotIn("cupy", search_text)
        self.assertIn("getDeviceProperties(0)", smoke_text)
        self.assertNotIn("getDevice()", smoke_text)
        self.assertIn("model_or_qwen_payload_accessed", smoke_text)

    def test_missing_strata_outer_packet_and_read_ledger_are_explicit(self):
        readme = (SOURCE / "README.md").read_text(encoding="utf-8")
        for phrase in (
            "authenticated STRATA scale, RHT/KLT",
            "component/expert framing, or page padding",
            "not a complete production rate",
            "cold reads are below 2x",
        ):
            self.assertIn(phrase, readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
