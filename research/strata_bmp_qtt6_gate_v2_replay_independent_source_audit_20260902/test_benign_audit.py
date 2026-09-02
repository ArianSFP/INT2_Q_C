#!/usr/bin/env python3
"""Independent benign correctness tests for frozen BMP/QTT6 replay v2."""

from __future__ import annotations

import ast
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SOURCE = Path(os.environ["STRATA_BMP_QTT6_V2_FROZEN_SOURCE"]).resolve(strict=True)
EXPECTED_MANIFEST = "84df0d32a55682f6565ac9d144f7de850acf77cde27bffdefa77a151211906f8"
EXPECTED_ROOT = "b518b203c43fd401c94e1bfcf67e029a85a95f1f7ce244fcd864a96d0780da47"
sys.path.insert(0, str(SOURCE))
for _name in ("codec", "search", "production_hooks", "cupy_backend"):
    sys.modules.pop(_name, None)
codec = importlib.import_module("codec")
search = importlib.import_module("search")
hooks_module = importlib.import_module("production_hooks")


def geometry() -> object:
    return codec.Geometry(704, 2304, 2, 512, 16, 1792, 256)


def zero_bmp_model(g=None) -> dict:
    g = geometry() if g is None else g
    return {
        "ranks": [0] * 6,
        "factors": [(np.zeros((g.row_count, 0), np.uint8),
                     np.zeros((g.col_count, 0), np.uint8)) for _ in range(6)],
    }


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, np.uint8).copy()
    rank = 0
    for col in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, col])
        if not pivots.size:
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(rank + 1, work.shape[0]):
            if work[row, col]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def boolean_cube(d: int) -> np.ndarray:
    return np.asarray(list(itertools.product((0, 1), repeat=d)), np.uint8)


def canonical_rows(rows: list[dict]) -> bytes:
    return json.dumps(rows, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def write_binding(path: Path, value) -> object:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
               if isinstance(value, dict) else bytes(value))
    path.write_bytes(payload)
    return hooks_module.ArtifactBinding(path, hashlib.sha256(payload).hexdigest())


def dummy_hooks(root: Path, source_manifest: str = "1" * 64,
                source_root: str = "2" * 64) -> object:
    binaries = {name: write_binding(root / name, name.encode("ascii"))
                for name in ("strata", "scale", "forward", "inverse",
                             "scorer", "factory", "framer")}
    read = write_binding(root / "read.json", {
        "schema": hooks_module.READ_SCHEMA,
        "maximum_routed_read_amplification": 1.25,
    })
    audit = write_binding(root / "audit.json", {
        "schema": hooks_module.AUDIT_SCHEMA,
        "passed": True,
        "producer_source_manifest_sha256": source_manifest,
        "producer_source_root_sha256": source_root,
    })
    controls = tuple(write_binding(root / f"control-{index}.json", {
        "schema": hooks_module.CONTROL_SCHEMA,
        "identical_complete_selection_replayed": True,
        "selected_control": True,
        "control_id": f"control-{index}",
    }) for index in range(8))
    return hooks_module.ProductionHooks(
        strata_packet=binaries["strata"], scale_decoder=binaries["scale"],
        forward_transform=binaries["forward"],
        inverse_transform=binaries["inverse"],
        original_bf16_scorer=binaries["scorer"],
        gaussian_control_factory=binaries["factory"],
        component_framer=binaries["framer"], routed_read_ledger=read,
        independent_audit_receipt=audit,
        gaussian_control_receipts=controls,
        expected_source_manifest_sha256=source_manifest,
        expected_source_root_sha256=source_root,
    )


class IndependentV2Audit(unittest.TestCase):
    def test_01_manifest_order_hashes_root_and_member_count(self):
        payload = (SOURCE / "SOURCE_MANIFEST.json").read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), EXPECTED_MANIFEST)
        manifest = json.loads(payload)
        rows = manifest["members"]
        names = [row["name"] for row in rows]
        self.assertEqual(len(rows), 13)
        self.assertEqual(names, sorted(names, key=lambda value: value.encode("utf-8")))
        observed = []
        for row in rows:
            member = (SOURCE / row["name"]).read_bytes()
            item = {"name": row["name"], "bytes": len(member),
                    "sha256": hashlib.sha256(member).hexdigest()}
            self.assertEqual(item, row)
            observed.append(item)
        self.assertEqual(hashlib.sha256(canonical_rows(observed)).hexdigest(),
                         EXPECTED_ROOT)

    def test_02_verifier_exact_cli_self_replays(self):
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(SOURCE / "verify_source.py"),
             "--package", str(SOURCE),
             "--expected-manifest-sha256", EXPECTED_MANIFEST],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        record = json.loads(completed.stdout)
        self.assertEqual(record["source_root_sha256"], EXPECTED_ROOT)
        self.assertTrue(record["canonical_utf8_member_order"])

    def test_03_six_completed_planes_and_literal_d64(self):
        values = np.arange(64, dtype=np.uint8)
        planes = codec.indices_to_planes(values)
        self.assertTrue(np.array_equal(codec.planes_to_indices(planes), values))
        independent = sum(planes[level].astype(np.uint16) << level
                          for level in range(6)).astype(np.uint8)
        self.assertTrue(np.array_equal(independent, values))
        with self.assertRaisesRegex(codec.CodecError, r"D\[i,0..63\]"):
            codec.validate_distortion_table(np.zeros((4096, 4)), 4096)

    def test_04_arbitrary_uint16_geometry_and_boundaries(self):
        cases = (
            geometry(),
            codec.Geometry(11008, 4096, 0, 4096, 8, 2048, 512),
            codec.Geometry(5760, 3584, 1, 5120, 16, 3072, 256),
            codec.Geometry(65535, 65535, 2, 65504, 16, 65024, 256),
            codec.Geometry(1, 4096, 0, 0, 1, 0, 4096),
        )
        for g in cases:
            for order in range(codec.ORDER_BANK_SIZE):
                _, bits = codec.active_features(g, order)
                self.assertEqual(bits.shape, (4096, 12))
                self.assertEqual(np.unique(np.packbits(bits, axis=1), axis=0)
                                 .shape[0], 4096)
        invalid = codec.Geometry(65536, 2048, 0, 0, 16, 0, 256)
        with self.assertRaisesRegex(codec.CodecError, "uint16 packet ABI"):
            codec.encode_packet(codec.FAMILY_BMP, 0, invalid,
                                zero_bmp_model(), [])

    def test_05_bmp_exhaustive_small_canonical_minimum_rank(self):
        rows, cols = 3, 4
        seen = set()
        for word in range(1 << (rows * cols)):
            plane = np.asarray([(word >> bit) & 1
                                for bit in range(rows * cols)], np.uint8)
            u, v = codec.canonical_gf2_factor(plane, rows, cols)
            self.assertEqual(u.shape[1], gf2_rank(plane.reshape(rows, cols)))
            self.assertTrue(np.array_equal(codec.bmp_plane(u, v), plane))
            descriptor = (u.shape[1], u.tobytes(), v.tobytes())
            self.assertNotIn(descriptor, seen)
            seen.add(descriptor)

    def test_06_qtt_all_three_bit_truth_tables_canonical(self):
        features = boolean_cube(3)
        seen = set()
        for word in range(256):
            target = np.asarray([(word >> bit) & 1 for bit in range(8)], np.uint8)
            canonical = codec.canonical_qtt(target, features)
            if word == 0:
                descriptor = (None, b"")
                self.assertIsNone(canonical)
            else:
                ranks, cores = canonical
                tensor = target.reshape(2, 2, 2)
                self.assertEqual(ranks, (gf2_rank(tensor.reshape(2, 4)),
                                         gf2_rank(tensor.reshape(4, 2))))
                self.assertTrue(np.array_equal(
                    codec.qtt_plane(cores, features, ranks), target))
                descriptor = (ranks, cores.tobytes())
            self.assertNotIn(descriptor, seen)
            seen.add(descriptor)

    def test_07_robdd_all_three_bit_truth_tables_canonical(self):
        features = boolean_cube(3)
        seen = set()
        for word in range(256):
            target = np.asarray([(word >> bit) & 1 for bit in range(8)], np.uint8)
            root, nodes = codec.build_robdd(target, features)
            self.assertTrue(np.array_equal(codec.eval_obdd(root, nodes, features),
                                           target))
            self.assertEqual(codec.build_robdd(target, features), (root, nodes))
            self.assertNotIn((root, tuple(nodes)), seen)
            seen.add((root, tuple(nodes)))

    def test_08_packet_formula_and_complete_rate_integer_bounds(self):
        g = geometry()
        packet = codec.encode_packet(codec.FAMILY_BMP, 0, g,
                                     zero_bmp_model(g), [(0, 63)])
        decoded = codec.decode_packet(packet)
        self.assertEqual(codec.descriptor_formula(decoded)["total_physical_bits"],
                         len(packet) * 8)
        for n in (19, 20, 21, 4095, 4096, 4097):
            cap = search.CompleteRateCap(n, 0)
            self.assertEqual(cap.min_total_bits, (43 * n + 19) // 20)
            self.assertEqual(cap.max_total_bits, (5 * n) // 2)

    def test_09_geometry_derived_candidate_capacities(self):
        for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                     2048, 4096):
            cols = 4096 // rows
            g = codec.Geometry(rows, cols, 0, 0, rows, 0, cols)
            receipt = search.candidate_serialized_capacity(g)
            self.assertEqual(receipt["candidate_count"], 16)
            self.assertEqual(receipt["aggregate_maximum_packet_bytes"],
                             sum(row["maximum_packet_bytes"]
                                 for row in receipt["candidates"]))
            for row in receipt["candidates"][:4]:
                rank = min(row["requested_rank"], rows, cols)
                expected = (codec.HEADER.size + codec.CRC.size + 6 +
                            6 * ((rank * (rows + cols) + 7) // 8) +
                            codec.MAX_EXCEPTIONS * codec.EXCEPTION.size)
                self.assertEqual(row["maximum_packet_bytes"], expected)
        skew = search.candidate_serialized_capacity(
            codec.Geometry(1, 4096, 0, 0, 1, 0, 4096))
        rank_one = next(row for row in skew["candidates"]
                        if row["family"] == "GF2_MATRIX_FACTOR" and
                        row["requested_rank"] == 1)
        self.assertEqual(rank_one["maximum_packet_bytes"], 3310)
        self.assertGreater(rank_one["maximum_packet_bytes"], 2048)

    def test_10_logical_capacity_is_explicitly_not_runtime_ownership(self):
        plan = dict(search.logical_capacity_plan(geometry()))
        self.assertEqual(plan["stable_order_intp_capacity"],
                         4096 * np.dtype(np.intp).itemsize)
        self.assertNotIn("stable_order_i32", plan)
        self.assertEqual(search.exact_workspace_plan(geometry()),
                         search.logical_capacity_plan(geometry()))
        source = (SOURCE / "search.py").read_text("utf-8")
        self.assertIn("Conservative logical capacities", source)
        self.assertIn("runtime_allocation_claimed", source)

    def test_11_runtime_owned_events_use_actual_intp_and_packet_lengths(self):
        table = np.zeros((4096, 64), np.float64)
        with mock.patch.object(search, "search_obdd", return_value=[]), \
             mock.patch.object(search, "search_qtt", return_value=[]):
            run = search.search_bank(table, geometry(), 0.0,
                                     search.CompleteRateCap(4096, 0))
        receipt = run["workspace"]["runtime_owned_objects"]
        owns = [row for row in receipt["events"] if row["event"] == "own"]
        order = [row for row in owns
                 if row["name"].startswith("stable_order_intp:")]
        self.assertEqual(len(order), 4)
        self.assertTrue(all(row["dtype"] == str(np.dtype(np.intp)) and
                            row["bytes"] == 4096 * np.dtype(np.intp).itemsize
                            for row in order))
        packet_bytes = sum(len(row["packet"]) for row in run["candidates"])
        self.assertEqual(receipt["live_owned_bytes"], packet_bytes)
        self.assertEqual(sum(row["bytes"] for row in receipt["live_objects"]),
                         packet_bytes)

    def test_12_workspace_ledger_exact_event_guards(self):
        ledger = search.WorkspaceLedger(cap_bytes=10)
        ledger.own("x", 8, dtype="bytes", shape=(8,))
        with self.assertRaisesRegex(codec.CodecError, "workspace cap"):
            ledger.own("y", 3, dtype="bytes", shape=(3,))
        ledger = search.WorkspaceLedger(cap_bytes=10)
        ledger.own("x", 8, dtype="bytes", shape=(8,))
        with self.assertRaisesRegex(codec.CodecError, "exact ownership"):
            ledger.release("x", 7)

    def test_13_cupy_ledger_separates_capacity_pool_and_cross_allocator(self):
        backend = (SOURCE / "cupy_backend.py").read_text("utf-8")
        for phrase in ("logical_capacity_separate_from_measured_cupy_pool",
                       "logical_serialized_capacity",
                       "actual_retained_candidate_packet_bytes",
                       "measured_cupy_pool", "cross_allocator_peak_claimed",
                       "peak_total_reserved_bytes"):
            self.assertIn(phrase, backend)
        self.assertIn("cp.asarray(host", backend)
        self.assertIn("cp.argmin", backend)
        self.assertIn("cp.where", backend)
        self.assertIn("cp.asnumpy", backend)

    def test_14_artifact_binding_rejects_false_digest_and_non_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "object"
            path.write_bytes(b"value")
            with self.assertRaisesRegex(codec.CodecError, "object SHA-256"):
                hooks_module.ArtifactBinding(path, "0" * 64).authenticate("x")
            with self.assertRaisesRegex(codec.CodecError, "path object"):
                hooks_module.ArtifactBinding(str(path), "0" * 64).authenticate("x")

    def test_15_self_authored_dummy_receipts_still_return_authorized(self):
        with tempfile.TemporaryDirectory() as raw:
            hooks = dummy_hooks(Path(raw))
            receipt = hooks.authorize()
            self.assertTrue(receipt["authorized"])
            self.assertEqual(receipt["producer_source_manifest_sha256"], "1" * 64)
            self.assertEqual(receipt["producer_source_root_sha256"], "2" * 64)
            self.assertEqual(receipt["gaussian_control_count"], 8)
            self.assertFalse(receipt["digest_syntax_only_authority"])

    def test_16_receipt_parsers_reject_obvious_semantic_failures(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            hooks = dummy_hooks(root)
            bad_read = write_binding(root / "bad-read.json", {
                "schema": hooks_module.READ_SCHEMA,
                "maximum_routed_read_amplification": 2.0,
            })
            with self.assertRaisesRegex(codec.CodecError, "routed read"):
                hooks_module.ProductionHooks(
                    **{**hooks.__dict__, "routed_read_ledger": bad_read}
                ).authorize()
            bad_audit = write_binding(root / "bad-audit.json", {
                "schema": hooks_module.AUDIT_SCHEMA, "passed": False,
                "producer_source_manifest_sha256": "1" * 64,
                "producer_source_root_sha256": "2" * 64,
            })
            with self.assertRaisesRegex(codec.CodecError, "independent audit"):
                hooks_module.ProductionHooks(
                    **{**hooks.__dict__, "independent_audit_receipt": bad_audit}
                ).authorize()

    def test_17_no_payload_or_network_import_boundary(self):
        forbidden = {"requests", "socket", "urllib", "http", "paramiko",
                     "safetensors", "transformers", "huggingface_hub", "torch"}
        for path in SOURCE.glob("*.py"):
            tree = ast.parse(path.read_text("utf-8"))
            roots = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".", 1)[0])
            self.assertFalse(roots & forbidden, path.name)

    def test_18_source_claims_runtime_and_payload_remain_held(self):
        manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text("utf-8"))
        design = json.loads((SOURCE / "design_lock.json").read_text("utf-8"))
        self.assertFalse(manifest["test_attestation"]["source_only_tests_executed"])
        self.assertFalse(manifest["test_attestation"]["fresh_cupy_search_executed"])
        self.assertFalse(manifest["production_attestation"]
                         ["production_launch_authorized"])
        self.assertFalse(design["payload_authority"]["qwen"])

    def test_19_readme_only_inventory_count_is_stale(self):
        readme = (SOURCE / "README.md").read_text("utf-8")
        self.assertIn("twelve-member source inventory", readme)
        manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text("utf-8"))
        self.assertEqual(len(manifest["members"]), 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)

