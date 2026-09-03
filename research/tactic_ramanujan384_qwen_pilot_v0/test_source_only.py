#!/usr/bin/env python3
"""Source-only tests for the held Qwen pilot runner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aperture  # noqa: E402
import capability  # noqa: E402
import pilot_runner  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def fixture_document() -> dict:
    closures = {}
    for key, expected in capability.KNOWN_CLOSURES.items():
        closures[key] = {"path": "/source-test/" + key, **expected}
    audits = []
    for kind in ("v3_atomic_cupy_runtime", "pilot_runner_independent_source"):
        audits.append({
            "kind": kind,
            "package_path": "/source-test/audit-" + kind,
            "manifest_sha256": digest("manifest-" + kind),
            "source_root_sha256": digest("root-" + kind),
            "root_field": "source_root_sha256",
            "root_domain_hex": "",
            "receipt_name": "AUDIT_RECEIPT.json",
            "receipt_sha256": digest("receipt-" + kind),
            "auditor_authority_id": "fixture-auditor-" + kind,
            "executed": True,
            "status": "PASS_INDEPENDENT_AUDIT",
            "dummy": True,
            "self_authored": True,
        })
    role_hashes = {
        "gate": ("fe4fd2b8438d868a4b118df31f2886d36c2178c93132e5738e64008d1717a51c",
                 "fe5cffd27348e97dfcfa9ad03cc53699627aee94842374a285c659f9648fecbd"),
        "up": ("857b57d1d37140bf10dbc582884c73c632f5a58cd1367f342c6903900e2b376b",
               "4c226db5a0469c06ce38ebda194fde79017f62e7a5296d7"),
        "down_transposed": (
            "a6820fb9d6efd4d58bf5b6a3d861e0deb64d9daaa2e5e73235a0d6db15d08c03",
            "4e9c82667e18fb77df9ae1a17be9f787942beb4da194fde79017f62e7a5296d7"),
    }
    # Correct the intentionally hand-written up reconstruction to a full digest.
    role_hashes["up"] = (role_hashes["up"][0], digest("up-reconstruction-fixture"))
    roles = [{
        "role": role,
        "source_bf16_path": "/source-test/" + role + ".bf16",
        "source_bytes": 3_145_728,
        "source_sha256": role_hashes[role][0],
        "coarse_reconstruction_f32_path": "/source-test/" + role + ".f32",
        "coarse_reconstruction_bytes": 6_291_456,
        "coarse_reconstruction_sha256": role_hashes[role][1],
    } for role in capability.ROLE_ORDER]
    return {
        "schema": capability.CAPABILITY_SCHEMA,
        "status": capability.CAPABILITY_STATUS,
        "evidence_class": "SOURCE_TEST_FIXTURE",
        "issuer_authority_id": "fixture-independent-issuer",
        "issued_before_payload_access": True,
        "pilot_source_pins": {"manifest_sha256": digest("pilot-manifest"),
                              "source_root_sha256": digest("pilot-root")},
        "closures": closures,
        "runtime_audits": audits,
        "cupy_runtime": {"version": "fixture", "module_file_sha256": digest("cupy"),
                         "device_ordinal": 0, "device_name": "fixture-gpu",
                         "compute_capability": [9, 0], "runtime_version": 12080,
                         "driver_version": 12080},
        "coarse_result": {
            "publication_directory": "/source-test/coarse-publication",
            "audit_receipt_path": "/source-test/coarse-audit.json",
            "audit_receipt_sha256": capability.COARSE_RESULT_AUDIT_FILE_SHA256,
            "coarse_member": {"name": "COARSE.bin",
                              "bytes": capability.COARSE_FRAME_BYTES,
                              "sha256": capability.COARSE_FRAME_SHA256},
            "completion_member": {"name": "COMPLETE.json",
                                  "sha256": "6b5e96c42518a29493e68237d649daad2e25f44a509ce7535425f83fd79fbb37"},
            "input_manifest_sha256":
                "6f6a0f174cd5b9c2b52ef29efd612e4520ef77afa6cc950ebec8c7e055fedcaa",
        },
        "roles": roles,
        "pilot": {
            "geometry": capability.GEOMETRY,
            "sample_blocks": {role: list(capability.SAMPLE_BLOCKS[role])
                              for role in capability.ROLE_ORDER},
            "sample_fixed_before_access": True,
            "bootstrap_replicates": 4096,
            "bootstrap_alpha": 0.05,
            "required_capture": capability.REQUIRED_CAPTURE,
            "coarse_relative_mse": capability.COARSE_RELATIVE_MSE,
            "target_d": capability.TARGET_D,
            "physical_bytes": capability.EXPECTED_PHYSICAL_BYTES,
            "weights": capability.EXPECTED_WEIGHTS,
            "rate_numerator": capability.EXPECTED_RATE_NUMERATOR,
            "rate_denominator": capability.EXPECTED_RATE_DENOMINATOR,
            "controls": {"phase": 1, "gaussian": 8},
        },
        "output_parent": "/source-test/output",
    }


class PilotSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (HERE.parent / "tactic_ramanujan384_adapter_v2_scalable" /
                  "scalable_core.py").read_bytes()
        cls.core = pilot_runner.load_snapshot_module(
            source, "tactic_r384_pilot_source_test_core", "<source-test>/scalable_core.py")

    def test_01_production_hold_precedes_path_access(self) -> None:
        self.assertIsNone(capability.TRUSTED_CAPABILITY_SHA256)
        with self.assertRaisesRegex(capability.CapabilityError, "HOLD"):
            capability.authorize_production(Path("definitely-does-not-exist"))

    def test_02_explicit_runner_authorization(self) -> None:
        with self.assertRaisesRegex(pilot_runner.PilotError, "explicit"):
            pilot_runner.execute(Path("unused"), Path("unused"), "wrong")

    def test_03_fixed_sample_unique_and_in_range(self) -> None:
        for role in capability.ROLE_ORDER:
            rows = capability.SAMPLE_BLOCKS[role]
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(set(rows)), 16)
            self.assertTrue(all(0 <= value < 384 for value in rows))

    def test_04_required_capture_arithmetic(self) -> None:
        expected = 1.0 - capability.TARGET_D / capability.COARSE_RELATIVE_MSE
        self.assertAlmostEqual(expected, capability.REQUIRED_CAPTURE, places=15)

    def test_05_literal_rate_fraction(self) -> None:
        self.assertEqual(
            8 * capability.EXPECTED_PHYSICAL_BYTES * capability.EXPECTED_RATE_DENOMINATOR,
            capability.EXPECTED_RATE_NUMERATOR * capability.EXPECTED_WEIGHTS)
        self.assertAlmostEqual(capability.EXPECTED_RATE_NUMERATOR /
                               capability.EXPECTED_RATE_DENOMINATOR,
                               2.4930555555555554)

    def test_06_low_capture_hard_kills(self) -> None:
        rows = {role: {"input_sse_by_block": np.ones(16),
                       "remaining_sse_by_block": np.full(16, 0.8)}
                for role in capability.ROLE_ORDER}
        result = aperture.bootstrap_capture_gate(rows)
        self.assertFalse(result["survives_to_full_expert"])
        self.assertFalse(result["controls_permitted"])

    def test_07_high_capture_survives(self) -> None:
        rows = {role: {"input_sse_by_block": np.ones(16),
                       "remaining_sse_by_block": np.full(16, 0.6)}
                for role in capability.ROLE_ORDER}
        result = aperture.bootstrap_capture_gate(rows)
        self.assertTrue(result["survives_to_full_expert"])
        self.assertTrue(result["full_expert_permitted"])
        self.assertFalse(result["controls_permitted"])

    def test_08_weak_owner_kills_strong_pool(self) -> None:
        rows = {}
        for role in capability.ROLE_ORDER:
            remaining = 0.75 if role == "gate" else 0.45
            rows[role] = {"input_sse_by_block": np.ones(16),
                          "remaining_sse_by_block": np.full(16, remaining)}
        result = aperture.bootstrap_capture_gate(rows)
        self.assertGreater(result["pooled_point_capture"], capability.REQUIRED_CAPTURE)
        self.assertFalse(result["survives_to_full_expert"])

    def test_09_bf16_literal_decode(self) -> None:
        values = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
        bits = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
        decoded = aperture.bf16_le_to_f64(bits, 3)
        np.testing.assert_array_equal(decoded, values.astype(np.float64))

    def test_10_coarse_f32_literal_decode(self) -> None:
        values = np.asarray([1.25, -3.5], dtype="<f4")
        decoded = aperture.coarse_f32_le_to_f64(values.tobytes(), 2)
        np.testing.assert_array_equal(decoded, values.astype(np.float64))

    def test_11_literal_rank_packet_search(self) -> None:
        prepared = self.core.prepare_dictionary(np)
        source = np.asarray(prepared["dictionary"][:, 17], dtype=np.float64)[None, :] * 0.25
        coarse = np.zeros_like(source)
        result = aperture.score_literal_rank_packets(
            np, self.core, source, coarse, "gate", prepared)
        self.assertEqual(result["candidate_states"], 15)
        self.assertEqual(result["candidate_ranks"], tuple(range(15)))
        self.assertTrue(result["all_scored_candidates_literal_packet_replayed"])
        self.assertEqual(result["per_candidate_solve_calls"], 0)
        self.assertLess(result["remaining_sse_by_block"][0],
                        result["input_sse_by_block"][0])

    def test_12_zero_residual_rank_zero_literal(self) -> None:
        prepared = self.core.prepare_dictionary(np)
        zeros = np.zeros((1, 4096), dtype=np.float64)
        result = aperture.score_literal_rank_packets(
            np, self.core, zeros, zeros, "up", prepared)
        self.assertEqual(int(result["winner_rank_by_block"][0]), 0)
        self.assertEqual(float(result["remaining_sse_by_block"][0]), 0.0)
        self.assertIsNotNone(result["packet_sha256_by_block_rank"][0][0])

    def test_13_one_pass_page_trace(self) -> None:
        payload = bytes(range(256)) * 32
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packet.bin"
            path.write_bytes(payload)
            replay, receipt = pilot_runner.one_pass_page_trace(path, payload)
        self.assertEqual(replay, payload)
        self.assertEqual(len(receipt["events"]), 2)
        self.assertEqual(receipt["read_amplification"], 1.0)
        self.assertFalse(receipt["accelerator_hbm_measured"])

    def test_14_independent_fp64_score(self) -> None:
        source = {role: np.asarray([1.0, 2.0]) for role in capability.ROLE_ORDER}
        reconstruction = {role: np.asarray([1.0, 1.0])
                          for role in capability.ROLE_ORDER}
        result = pilot_runner.independent_decoded_score(np, source, reconstruction, 2)
        self.assertEqual(result["pooled_sse_fp64"], 3.0)
        self.assertEqual(result["pooled_source_energy_fp64"], 15.0)
        self.assertTrue(result["decoded_bytes_rescored_in_fp64"])

    def test_15_source_fixture_capability_schema(self) -> None:
        document = fixture_document()
        capability._validate_source_test_document(document)

    def test_16_mutated_sample_precommit_rejected(self) -> None:
        document = fixture_document()
        document["pilot"]["sample_blocks"]["gate"][0] ^= 1
        with self.assertRaisesRegex(capability.CapabilityError, "aperture"):
            capability._validate_source_test_document(document)

    def test_17_dummy_production_audit_rejected(self) -> None:
        document = fixture_document()
        document["evidence_class"] = "PRODUCTION"
        with self.assertRaisesRegex(capability.CapabilityError, "audit capability"):
            capability._validate_source_test_document(document)


if __name__ == "__main__":
    unittest.main(verbosity=2)
