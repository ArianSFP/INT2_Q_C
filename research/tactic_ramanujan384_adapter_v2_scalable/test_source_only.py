#!/usr/bin/env python3
"""Bounded source-only tests for scalable v2; no model payload aperture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("tactic_ramanujan384_v2_test_core", "scalable_core.py")
io = load("tactic_ramanujan384_v2_test_io", "authenticated_io.py")
capability = load("tactic_ramanujan384_v2_test_capability", "coarse_capability.py")
fixture = load("tactic_ramanujan384_v2_test_fixture", "source_free_fixture.py")
decoder_module = load("tactic_ramanujan384_v2_test_decoder", "fixture_coarse_decoder.py")


class PacketTests(unittest.TestCase):
    def test_canonical_packet_roundtrip_and_crc(self):
        packet = core.encode_packet("gate", (0, 17, 383), (-1023, 7, 991), 2.0 ** -10)
        self.assertEqual(len(packet), 48)
        row = core.decode_packet(packet)
        self.assertEqual(row["support"], (0, 17, 383))
        hostile = bytearray(packet)
        hostile[-1] ^= 1
        with self.assertRaisesRegex(core.ScalableCodecError, "CRC"):
            core.decode_packet(bytes(hostile))


class ShapeTests(unittest.TestCase):
    def test_qwen_and_target_fixture_ledgers(self):
        qwen = core.define_shape(768, 2048)
        self.assertEqual(core.physical_ledger(qwen)["physical_rate_bpw"], 359 / 144)
        target = core.define_shape(128, 2048)
        self.assertEqual(core.physical_ledger(target)["physical_rate_bpw"], 2.5)
        self.assertTrue(core.physical_ledger(target)["target_rate_eligible"])

    def test_uint32_block_overflow_is_rejected(self):
        with self.assertRaisesRegex(core.ScalableCodecError, "uint32"):
            core.define_shape(core.UINT32_MAX, 4097)


class GaussianTests(unittest.TestCase):
    def test_canonical_box_muller_is_repeatable_and_gaussian_not_irwin_hall(self):
        first = core.canonical_gaussian_f64((8, 4096), 7)
        second = core.canonical_gaussian_f64((8, 4096), 7)
        self.assertEqual(first.tobytes(order="C"), second.tobytes(order="C"))
        self.assertLess(abs(float(np.mean(first))), 0.03)
        self.assertLess(abs(float(np.std(first)) - 1.0), 0.03)
        source = (ROOT / "scalable_core.py").read_text(encoding="utf-8")
        self.assertNotIn("Irwin", source)
        self.assertNotIn("xp.random", source)
        self.assertNotIn("np.random", source)


class BatchTests(unittest.TestCase):
    def test_all_ranks_use_one_solve_and_one_candidate_einsum(self):
        shape = core.define_shape(32, 32)
        coordinate = np.arange(shape.role_values)
        lookup = np.asarray([core.ramanujan_sum(7, index) for index in range(7)])
        source = (2.0 ** -7) * lookup[coordinate % 7]
        prepared = core.prepare_dictionary(np)
        row = core.encode_role_batched(
            np, source, np.zeros_like(source), shape, "gate", prepared
        )
        self.assertEqual(row["candidate_ranks_batched"], 15)
        self.assertEqual(row["batched_linear_solve_calls"], 1)
        self.assertEqual(row["batched_candidate_einsum_calls"], 1)
        self.assertEqual(row["per_candidate_host_scalar_syncs"], 0)
        self.assertEqual(row["per_candidate_solves"], 0)
        self.assertEqual(row["per_candidate_matmuls"], 0)
        self.assertTrue(row["decoded_packet_state_equals_selected_candidate_state"])
        self.assertFalse(row["tail_padding_in_candidate_generation_or_score"])

    def test_hot_core_has_no_dynamic_import_or_item_sync(self):
        source = (ROOT / "scalable_core.py").read_text(encoding="utf-8")
        self.assertNotIn("importlib", source)
        self.assertNotIn(".item(", source)
        self.assertEqual(source.count("xp.linalg.solve("), 1)


class CapabilityTests(unittest.TestCase):
    def test_independently_audited_coarse_capability_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            role_inputs, arguments = fixture.build(
                Path(temporary), core=core, io=io,
                decoder_class=decoder_module.SourceFreeZeroCoarseDecoder,
                decoder_source_path=ROOT / "fixture_coarse_decoder.py",
                intermediate=32, hidden=32,
            )
            row = capability.authenticate(**arguments, io=io)
            self.assertTrue(row["independently_audited"])
            hostile = dict(arguments)
            hostile["expected_capability_sha256"] = "0" * 64
            with self.assertRaisesRegex(capability.CoarseCapabilityError, "capability SHA256"):
                capability.authenticate(**hostile, io=io)
            self.assertEqual(len(role_inputs), 3)


class ClosureTests(unittest.TestCase):
    def test_external_bootstrap_is_outside_package_and_rejects_extras(self):
        bootstrap = ROOT.parent / "tactic_ramanujan384_adapter_v2_scalable_bootstrap_verify.py"
        self.assertTrue(bootstrap.is_file())
        source = bootstrap.read_text(encoding="utf-8")
        self.assertIn("extra, missing, or nested package entry", source)
        self.assertNotIn("importlib", source)

    def test_v1_and_review_manifests_are_pinned(self):
        lock = json.loads((ROOT / "dependency_lock.json").read_text(encoding="ascii"))
        for key, directory in (
            ("v1", "tactic_ramanujan384_adapter_v1_authority"),
            ("v1_review", "tactic_ramanujan384_adapter_v1_authority_independent_review_20260902"),
        ):
            path = ROOT.parent / directory / "SOURCE_MANIFEST.json"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             lock[key]["source_manifest_sha256"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    )
    print(json.dumps({
        "schema": "tactic-ramanujan384-v2-source-only-tests",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "passed": result.wasSuccessful(),
        "qwen_payload_accessed": False, "coarse_model_payload_accessed": False,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
