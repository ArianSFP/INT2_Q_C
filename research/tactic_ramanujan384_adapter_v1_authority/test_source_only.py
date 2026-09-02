#!/usr/bin/env python3
"""Source-only hostile tests for the v1 authority repair."""

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
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


manifest = load("tactic_ramanujan384_authority_test_manifest", "manifest.py")
contract = load("tactic_ramanujan384_authority_test_contract", "contract.py")
controls = load("tactic_ramanujan384_authority_test_controls", "stable_controls.py")
trace = load("tactic_ramanujan384_authority_test_trace", "read_trace.py")
codec = load("tactic_ramanujan384_authority_test_codec", "codec_authority.py")
fixture = load("tactic_ramanujan384_authority_test_fixture", "run_source_free_fixture.py")


class ManifestTests(unittest.TestCase):
    def test_one_sorted_key_root_algorithm_is_order_invariant(self):
        rows = [
            {"name": "b", "bytes": 2, "sha256": "2" * 64},
            {"sha256": "1" * 64, "bytes": 1, "name": "a"},
        ]
        self.assertEqual(manifest.source_root(rows), manifest.source_root(reversed(rows)))
        expected = hashlib.sha256(manifest.canonical_json([
            {"bytes": 1, "name": "a", "sha256": "1" * 64},
            {"bytes": 2, "name": "b", "sha256": "2" * 64},
        ])).hexdigest()
        self.assertEqual(manifest.source_root(rows), expected)


class ContractTests(unittest.TestCase):
    def test_qwen_shape_exact_integrality_tail_and_ledger(self):
        shape = contract.define_shape(768, 2048)
        self.assertTrue(shape.tail_free)
        self.assertEqual(shape.coarse_bytes, 1414656)
        row = contract.physical_ledger(shape)
        self.assertEqual(row["physical_bytes"], 1470464)
        self.assertAlmostEqual(row["physical_rate_bpw"], 359 / 144)
        self.assertTrue(row["target_rate_eligible"])

    def test_tail_is_explicit_and_nonintegral_coarse_shape_rejected(self):
        shape = contract.define_shape(32, 160)
        self.assertFalse(shape.tail_free)
        self.assertEqual(shape.last_block_valid_values, 1024)
        self.assertEqual(shape.tail_values_per_role, 3072)
        with self.assertRaisesRegex(contract.ContractError, "nonintegral"):
            contract.define_shape(1, 1)


class StableControlTests(unittest.TestCase):
    def test_frozen_integer_prng_and_control_bytes_repeat(self):
        first = controls.splitmix64_words(7, 8)
        second = controls.splitmix64_words(7, 8)
        self.assertTrue(np.array_equal(first, second))
        reference = np.arange(32, dtype=np.float64).reshape(2, 16)
        a = controls.moment_matched_blocks(np, reference, 99)
        b = controls.moment_matched_blocks(np, reference, 99)
        self.assertEqual(controls.host_bytes(np, a), controls.host_bytes(np, b))
        for index in range(2):
            self.assertAlmostEqual(float(np.mean(a[index])), float(np.mean(reference[index])), places=12)
            self.assertAlmostEqual(float(np.sum((a[index] - np.mean(a[index])) ** 2)),
                                   float(np.sum((reference[index] - np.mean(reference[index])) ** 2)),
                                   places=9)


class CodecTests(unittest.TestCase):
    def test_every_defined_rank_candidate_is_packet_replayed_and_tail_not_scored(self):
        shape = contract.define_shape(32, 160)
        coordinate = np.arange(shape.role_values, dtype=np.float64)
        source = 0.01 * ((coordinate % 7) - 3) + 0.002 * ((coordinate % 11) - 5)
        row = codec.encode_role(np, source, np.zeros_like(source), shape, "gate")
        self.assertTrue(row["all_defined_candidates_packet_replayed_before_selection"])
        self.assertTrue(row["winner_stream_replayed_after_selection"])
        self.assertFalse(row["tail_padding_scored"])
        self.assertEqual(len(row["candidate_packet_replays_per_block"]), shape.blocks_per_role)
        self.assertTrue(all(value >= 1 for value in row["candidate_packet_replays_per_block"]))

    def test_source_free_fixture_literal_weight_replay(self):
        row = fixture.run(np)
        self.assertEqual(row["status"], "PASS_SOURCE_FREE_LITERAL_COARSE_FINE_WEIGHT_REPLAY")
        self.assertTrue(row["literal_composite_reconstructed_to_weights"])
        self.assertTrue(row["independent_source_domain_fp64_rescore"])
        self.assertTrue(row["every_defined_candidate_packet_replayed_before_selection"])
        self.assertTrue(row["actual_input_manifests_opened"])
        self.assertTrue(row["actual_auditor_manifests_opened"])
        self.assertEqual(row["instrumented_file_read_amplification"], 1.0)
        self.assertTrue(row["layout_bound_is_not_a_measurement"])
        self.assertFalse(row["qwen_payload_accessed"])


class ReadTraceTests(unittest.TestCase):
    def test_trace_is_observation_and_layout_bound_is_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object.bin"
            path.write_bytes(b"authority")
            payload, row = trace.read_once(path, 9)
            self.assertEqual(payload, b"authority")
            self.assertEqual(row["instrumented_file_read_amplification"], 1.0)
            self.assertTrue(row["layout_bound_is_not_a_measurement"])
            self.assertTrue(row["instrumented_trace_is_not_physical_storage_or_hbm_telemetry"])
            self.assertFalse(row["accelerator_hbm_measured"])


class ClosureTests(unittest.TestCase):
    def test_v0_and_audit_pins_are_literal(self):
        lock = json.loads((ROOT / "dependency_lock.json").read_text(encoding="utf-8"))
        v0 = ROOT.parent / "tactic_ramanujan384_adapter_v0" / "SOURCE_MANIFEST.json"
        audit = (ROOT.parent / "tactic_ramanujan384_adapter_v0_independent_source_audit_20260902"
                 / "SOURCE_MANIFEST.json")
        self.assertEqual(hashlib.sha256(v0.read_bytes()).hexdigest(),
                         lock["producer_v0"]["source_manifest_sha256"])
        self.assertEqual(hashlib.sha256(audit.read_bytes()).hexdigest(),
                         lock["independent_audit_v0"]["source_manifest_sha256"])

    def test_cupy_runner_has_no_payload_aperture(self):
        source = (ROOT / "run_source_free_cupy_smoke.py").read_text(encoding="utf-8")
        for forbidden in ("--payload", "--qwen", "--coarse", "COARSE.bin"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "tactic-ramanujan384-authority-source-test-v1",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "qwen_payload_accessed": False,
        "coarse_model_payload_accessed": False,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
