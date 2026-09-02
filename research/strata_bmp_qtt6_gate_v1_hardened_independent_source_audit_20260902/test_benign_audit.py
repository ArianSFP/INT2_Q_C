#!/usr/bin/env python3
"""Independent benign correctness tests for frozen hardened BMP/QTT6 v1."""

from __future__ import annotations

import ast
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import sys
import unittest
import zlib

import numpy as np


SOURCE = Path(os.environ["STRATA_BMP_QTT6_V1_FROZEN_SOURCE"]).resolve(strict=True)
sys.path.insert(0, str(SOURCE))
for _name in ("codec", "search", "production_hooks", "cupy_backend"):
    sys.modules.pop(_name, None)
codec = importlib.import_module("codec")
search = importlib.import_module("search")
production_hooks = importlib.import_module("production_hooks")


def geometry() -> object:
    return codec.Geometry(704, 2304, 2, 512, 16, 1792, 256)


def zero_bmp_model(g=None) -> dict:
    g = geometry() if g is None else g
    return {
        "ranks": [0] * 6,
        "factors": [
            (np.zeros((g.row_count, 0), np.uint8),
             np.zeros((g.col_count, 0), np.uint8))
            for _ in range(6)
        ],
    }


def gf2_rank(matrix: np.ndarray) -> int:
    """Independent exact GF(2) rank."""
    work = np.asarray(matrix, dtype=np.uint8).copy()
    rows, cols = work.shape
    rank = 0
    for col in range(cols):
        candidates = np.flatnonzero(work[rank:, col])
        if not candidates.size:
            continue
        pivot = rank + int(candidates[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(rank + 1, rows):
            if work[row, col]:
                work[row] ^= work[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def boolean_cube(d: int) -> np.ndarray:
    return np.asarray(list(itertools.product((0, 1), repeat=d)), dtype=np.uint8)


def recalc_crc(packet: bytes) -> bytes:
    body = packet[:-codec.CRC.size]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


class IndependentBenignAudit(unittest.TestCase):
    def test_01_exact_six_plane_d64_semantics(self):
        values = np.arange(64, dtype=np.uint8)
        planes = codec.indices_to_planes(values)
        independent = sum(
            planes[level].astype(np.uint16) << level for level in range(6)
        ).astype(np.uint8)
        self.assertTrue(np.array_equal(independent, values))
        self.assertTrue(np.array_equal(codec.planes_to_indices(planes), values))
        with self.assertRaisesRegex(codec.CodecError, r"D\[i,0..63\]"):
            codec.validate_distortion_table(np.zeros((4096, 4)), 4096)

    def test_02_mixed_radix_arbitrary_uint16_geometry_bijections(self):
        cases = (
            geometry(),
            codec.Geometry(11008, 4096, 0, 4096, 8, 2048, 512),
            codec.Geometry(5760, 3584, 1, 5120, 16, 3072, 256),
            codec.Geometry(65535, 65535, 2, 65504, 16, 65024, 256),
            codec.Geometry(1, 4096, 0, 0, 1, 0, 4096),
            codec.Geometry(4096, 1, 1, 0, 4096, 0, 1),
        )
        for g in cases:
            for order_id in range(codec.ORDER_BANK_SIZE):
                names, bits = codec.active_features(g, order_id)
                self.assertEqual(bits.shape, (4096, 12))
                self.assertEqual(len(names), 12)
                packed = np.packbits(bits, axis=1, bitorder="little")
                self.assertEqual(np.unique(packed, axis=0).shape[0], 4096)

    def test_03_uint16_boundaries_fail_before_struct_pack(self):
        boundary = codec.Geometry(65535, 65535, 2, 65504, 16, 65024, 256)
        packet = codec.encode_packet(
            codec.FAMILY_BMP, 0, boundary, zero_bmp_model(boundary), []
        )
        self.assertEqual(codec.decode_packet(packet)["geometry"], boundary)
        invalid = (
            codec.Geometry(65536, 2048, 0, 0, 16, 0, 256),
            codec.Geometry(768, 65536, 0, 0, 16, 0, 256),
            codec.Geometry(768, 2048, 0, 65536, 16, 0, 256),
            codec.Geometry(768, 2048, 0, 0, 65536, 0, 256),
            codec.Geometry(768, 2048, 0, 0, 16, 65536, 256),
            codec.Geometry(768, 2048, 0, 0, 16, 0, 65536),
        )
        for g in invalid:
            with self.assertRaisesRegex(codec.CodecError, "uint16 packet ABI"):
                codec.encode_packet(codec.FAMILY_BMP, 0, g,
                                    zero_bmp_model(geometry()), [])

    def test_04_bmp_exhaustive_small_minimum_rank_and_determinism(self):
        rows, cols = 3, 4
        seen = set()
        for word in range(1 << (rows * cols)):
            flat = np.asarray([(word >> bit) & 1
                               for bit in range(rows * cols)], dtype=np.uint8)
            u, v = codec.canonical_gf2_factor(flat, rows, cols)
            self.assertEqual(u.shape[1], gf2_rank(flat.reshape(rows, cols)))
            self.assertTrue(np.array_equal(codec.bmp_plane(u, v), flat))
            u2, v2 = codec.canonical_gf2_factor(
                codec.bmp_plane(u, v), rows, cols
            )
            self.assertTrue(np.array_equal(u, u2))
            self.assertTrue(np.array_equal(v, v2))
            descriptor = (u.shape[1], u.tobytes(), v.tobytes())
            self.assertNotIn(descriptor, seen)
            seen.add(descriptor)
        self.assertEqual(len(seen), 1 << (rows * cols))

    def test_05_bmp_zero_inflation_swap_and_gl_gauge_rejected(self):
        g = geometry()
        inflated = {
            "ranks": [1] * 6,
            "factors": [(np.zeros((16, 1), np.uint8),
                         np.zeros((256, 1), np.uint8)) for _ in range(6)],
        }
        with self.assertRaisesRegex(codec.CodecError, "canonical minimum-rank"):
            codec.encode_packet(codec.FAMILY_BMP, 0, g, inflated, [])

        rr, cc = np.arange(16), np.arange(256)
        plane = (np.outer((rr >> 0) & 1, (cc >> 0) & 1) ^
                 np.outer((rr >> 1) & 1, (cc >> 1) & 1)).astype(np.uint8)
        u, v = codec.canonical_gf2_factor(plane.reshape(-1), 16, 256)
        aliases = [
            (u[:, ::-1].copy(), v[:, ::-1].copy()),
            (((u.astype(np.uint16) @ np.asarray([[1, 1], [0, 1]], np.uint16)) & 1)
             .astype(np.uint8),
             ((v.astype(np.uint16) @ np.asarray([[1, 0], [1, 1]], np.uint16)) & 1)
             .astype(np.uint8)),
        ]
        for alias_u, alias_v in aliases:
            self.assertTrue(np.array_equal(codec.bmp_plane(alias_u, alias_v),
                                           plane.reshape(-1)))
            model = {"ranks": [2] * 6,
                     "factors": [(alias_u, alias_v)] * 6}
            with self.assertRaisesRegex(codec.CodecError,
                                        "canonical minimum-rank"):
                codec.encode_packet(codec.FAMILY_BMP, 0, g, model, [])

    def test_06_qtt_all_three_bit_truth_tables_are_canonical_and_injective(self):
        features = boolean_cube(3)
        seen = set()
        for word in range(1 << (1 << 3)):
            target = np.asarray([(word >> bit) & 1 for bit in range(8)],
                                dtype=np.uint8)
            canonical = codec.canonical_qtt(target, features)
            if word == 0:
                self.assertIsNone(canonical)
                descriptor = (None, b"")
            else:
                self.assertIsNotNone(canonical)
                ranks, cores = canonical
                tensor = np.zeros((2, 2, 2), np.uint8)
                tensor[tuple(features[:, axis] for axis in range(3))] = target
                expected = (gf2_rank(tensor.reshape(2, 4)),
                            gf2_rank(tensor.reshape(4, 2)))
                self.assertEqual(ranks, expected)
                self.assertTrue(np.array_equal(
                    codec.qtt_plane(cores, features, ranks), target
                ))
                self.assertEqual(codec.canonical_qtt(target, features)[0], ranks)
                self.assertTrue(np.array_equal(
                    codec.canonical_qtt(target, features)[1], cores
                ))
                descriptor = (ranks, cores.tobytes())
            self.assertNotIn(descriptor, seen)
            seen.add(descriptor)
        self.assertEqual(len(seen), 256)

    def test_07_qtt_zero_rank_inflation_and_unused_mask_rejected(self):
        g = geometry()
        _, features = codec.active_features(g, 0)
        ranks = (2,) * (features.shape[1] - 1)
        cores = np.zeros(codec.qtt_core_bit_count(features.shape[1], ranks),
                         np.uint8)
        with self.assertRaisesRegex(codec.CodecError, "canonical minimum-rank"):
            codec.encode_packet(codec.FAMILY_QTT, 0, g,
                                {"rank_vectors": [ranks] * 6,
                                 "cores": [cores] * 6}, [])

        zero = {"rank_vectors": [None] * 6,
                "cores": [np.zeros(0, np.uint8) for _ in range(6)]}
        packet = bytearray(codec.encode_packet(codec.FAMILY_QTT, 0, g, zero, []))
        struct.pack_into("<H", packet, codec.HEADER.size,
                         1 + (1 << (features.shape[1] - 1)))
        with self.assertRaisesRegex(codec.CodecError, "unused rank bits"):
            codec.decode_packet(recalc_crc(bytes(packet)))

    def test_08_robdd_all_three_bit_truth_tables_are_canonical_and_injective(self):
        features = boolean_cube(3)
        seen = set()
        for word in range(256):
            target = np.asarray([(word >> bit) & 1 for bit in range(8)],
                                dtype=np.uint8)
            root, nodes = codec.build_robdd(target, features)
            self.assertTrue(np.array_equal(
                codec.eval_obdd(root, nodes, features), target
            ))
            root2, nodes2 = codec.build_robdd(
                codec.eval_obdd(root, nodes, features), features
            )
            self.assertEqual((root, nodes), (root2, nodes2))
            descriptor = (root, tuple(nodes))
            self.assertNotIn(descriptor, seen)
            seen.add(descriptor)
        self.assertEqual(len(seen), 256)

    def test_09_packet_crc_extent_tail_and_exception_canonicality(self):
        g = geometry()
        packet = codec.encode_packet(codec.FAMILY_BMP, 0, g,
                                     zero_bmp_model(g), [(0, 63)])
        self.assertEqual(codec.encode_packet(
            codec.FAMILY_BMP, 0, g,
            codec.decode_packet(packet)["model"], [(0, 63)]
        ), packet)
        corrupt = bytearray(packet)
        corrupt[20] ^= 1
        with self.assertRaisesRegex(codec.CodecError, "CRC32"):
            codec.decode_packet(bytes(corrupt))
        with self.assertRaisesRegex(codec.CodecError, "sorted unique"):
            codec.encode_packet(codec.FAMILY_BMP, 0, g, zero_bmp_model(g),
                                [(2, 1), (1, 2)])
        with self.assertRaisesRegex(codec.CodecError, "redundant"):
            codec.encode_packet(codec.FAMILY_BMP, 0, g, zero_bmp_model(g),
                                [(0, 0)])

    def test_10_descriptor_formula_matches_literal_packets(self):
        g = geometry()
        packet = codec.encode_packet(codec.FAMILY_BMP, 0, g,
                                     zero_bmp_model(g), [(0, 63), (1, 62)])
        decoded = codec.decode_packet(packet)
        formula = codec.descriptor_formula(decoded)
        self.assertEqual(formula["total_physical_bits"], 8 * len(packet))
        self.assertEqual(formula["exception_bytes"], 6)

    def test_11_complete_rate_cap_exact_integer_boundaries(self):
        for n in (1, 2, 3, 19, 20, 21, 4095, 4096, 4097):
            cap = search.CompleteRateCap(n, 0)
            self.assertEqual(cap.min_total_bits, math.ceil(43 * n / 20))
            self.assertEqual(cap.max_total_bits, (5 * n) // 2)
        cap = search.CompleteRateCap(4096, 1000, 2000, 3000)
        self.assertEqual(cap.available_packet_bits, 4240)
        self.assertTrue(cap.admit_packet(4240))
        self.assertFalse(cap.admit_packet(4241))
        with self.assertRaisesRegex(codec.CodecError, "retains reserved"):
            cap.assert_complete(4240)
        self.assertEqual(
            search.CompleteRateCap(4096, 1000, 5000).assert_complete(4240)
            ["total_bits"], 10240
        )

    def test_12_search_requires_explicit_complete_rate_cap(self):
        table = np.zeros((4096, 64), np.float64)
        with self.assertRaisesRegex(codec.CodecError, "explicit complete-rate"):
            search.search_bank(table, geometry(), 0.0, None)
        with self.assertRaisesRegex(codec.CodecError,
                                    "nonpacket fields exceed"):
            search.search_bank(table, geometry(), 0.0,
                               search.CompleteRateCap(4096, 10241))

    def test_13_workspace_packet_factor_is_not_a_per_candidate_maximum(self):
        g = codec.Geometry(4, 1024, 0, 0, 4, 0, 1024)
        rr, cc = np.arange(4), np.arange(1024)
        u_seed = np.eye(4, dtype=np.uint8)
        v_seed = np.stack([((cc >> bit) & 1).astype(np.uint8)
                           for bit in range(4)], axis=1)
        plane = ((u_seed.astype(np.uint16) @ v_seed.astype(np.uint16).T) & 1)
        u, v = codec.canonical_gf2_factor(plane.reshape(-1), 4, 1024)
        self.assertEqual(u.shape[1], 4)
        model = {"ranks": [4] * 6, "factors": [(u, v)] * 6}
        base = codec.planes_to_indices(codec.base_planes(
            codec.FAMILY_BMP, model, g, 0
        ))
        exceptions = [(position, (int(base[position]) + 1) & 63)
                      for position in range(codec.MAX_EXCEPTIONS)]
        packet = codec.encode_packet(codec.FAMILY_BMP, 0, g, model, exceptions)
        self.assertEqual(len(packet), 3316)
        plan = dict(search.exact_workspace_plan(g))
        self.assertEqual(plan["candidate_packet_bank"],
                         search.MAX_FAMILY_CANDIDATES * 2048)
        self.assertGreater(len(packet), 2048)

    def test_14_workspace_stable_order_dtype_exposure(self):
        g = geometry()
        planned = dict(search.exact_workspace_plan(g))["stable_order_i32"]
        actual_order = np.argsort(np.arange(g.count, dtype=np.float64),
                                  kind="stable")
        self.assertEqual(planned, g.count * 4)
        self.assertEqual(actual_order.dtype, np.dtype(np.intp))
        if np.dtype(np.intp).itemsize > 4:
            self.assertGreater(actual_order.nbytes, planned)

    def test_15_search_rescores_decoded_packet_and_caps_evaluations(self):
        g = geometry()
        table = np.full((g.count, 64), 9.0, np.float64)
        table[:, 0] = np.arange(g.count, dtype=np.float64) / g.count
        packet = codec.encode_packet(codec.FAMILY_BMP, 0, g,
                                     zero_bmp_model(g), [])
        value, metrics = search.objective(table, packet, 0.25)
        expected = float(table[:, 0].sum(dtype=np.float64))
        self.assertEqual(metrics["sse"], expected)
        self.assertEqual(value, expected + 0.25 * len(packet) * 8)
        counter = search.Counter(search.MAX_SEARCH_EVALUATIONS)
        with self.assertRaisesRegex(codec.CodecError, "evaluation cap"):
            counter.add()

    def test_16_production_hooks_fail_closed_but_are_syntactic_only(self):
        with self.assertRaisesRegex(codec.CodecError, "unbound production hooks"):
            production_hooks.held_source_only_hooks().authorize()
        digest = "a" * 64
        hooks = production_hooks.ProductionHooks(
            strata_packet_sha256=digest,
            scale_decoder_sha256=digest,
            forward_transform_sha256=digest,
            inverse_transform_sha256=digest,
            original_bf16_scorer_sha256=digest,
            gaussian_control_factory_sha256=digest,
            gaussian_control_count=8,
            component_framer_sha256=digest,
            routed_read_ledger_sha256=digest,
            independent_audit_receipt_sha256=digest,
        )
        self.assertTrue(hooks.authorize()["authorized"])

    def test_17_cupy_source_contains_real_device_search_and_explicit_holds(self):
        backend = (SOURCE / "cupy_backend.py").read_text("utf-8")
        worker = (SOURCE / "cupy_worker.py").read_text("utf-8")
        launcher = (SOURCE / "run_cupy_smoke.py").read_text("utf-8")
        for phrase in ("cp.asarray(host", "cp.argmin", "cp.where",
                       "cp.asnumpy", "sum(dtype=cp.float64)",
                       "canonical_gf2_factor", "add_joint_exceptions"):
            self.assertIn(phrase, backend)
        for phrase in ("MemoryPool", "set_allocator", "synchronize",
                       "total_reserved_bytes", "active_device_id",
                       "compiled_kernel_identity_probe"):
            self.assertIn(phrase, backend)
        self.assertIn('"ROBDD GPU search"', backend)
        self.assertIn('"canonical QTT GPU search"', backend)
        self.assertIn('sys.executable, "-I", "-B"', launcher)
        self.assertIn("payload_authority", worker)

    def test_18_cupy_import_provenance_checks_are_not_simple_facade_checks(self):
        backend = (SOURCE / "cupy_backend.py").read_text("utf-8")
        tree = ast.parse(backend)
        top_level_imported = {
            alias.name for node in tree.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        require_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "require_cupy"
        )
        local_imported = {
            alias.name for node in ast.walk(require_function)
            if isinstance(node, ast.Import) for alias in node.names
        }
        self.assertNotIn("cupy", top_level_imported)
        self.assertIn("cupy", local_imported)
        for phrase in ("types.ModuleType", "packages_distributions",
                       "distribution version mismatch", "resolve(strict=True)",
                       "module_file_sha256"):
            self.assertIn(phrase, backend)

    def test_19_readme_replay_cli_mismatch_is_frozen(self):
        readme = (SOURCE / "README.md").read_text("utf-8")
        verifier = (SOURCE / "verify_source.py").read_text("utf-8")
        tree = ast.parse(verifier)
        options = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        self.assertIn("--manifest", readme)
        self.assertNotIn("--manifest", options)
        self.assertIn("--package", options)
        self.assertIn("--expected-manifest-sha256", options)
        manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text("utf-8"))
        names = [row["name"] for row in manifest["members"]]
        self.assertNotEqual(names, sorted(names, key=lambda value: value.encode("utf-8")))

    def test_20_no_payload_locator_or_network_import_boundary(self):
        forbidden = {"requests", "socket", "urllib", "http", "paramiko",
                     "safetensors", "transformers", "huggingface_hub", "torch"}
        for path in SOURCE.glob("*.py"):
            source = path.read_text("utf-8")
            tree = ast.parse(source)
            roots = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".", 1)[0])
            self.assertFalse(roots & forbidden, path.name)
            if path.name not in {"test_source_only.py", "verify_source.py"}:
                self.assertNotIn("root@", source)
                self.assertNotIn(".safetensors", source)

    def test_21_claim_boundary_and_runtime_pending_are_literal(self):
        design = json.loads((SOURCE / "design_lock.json").read_text("utf-8"))
        manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text("utf-8"))
        self.assertFalse(design["payload_authority"]["qwen"])
        self.assertFalse(design["execution_attestation"]["runtime_pass_claimed"])
        self.assertFalse(manifest["test_attestation"]["fresh_cupy_search_executed"])
        self.assertFalse(manifest["test_attestation"]["independent_source_audit_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
