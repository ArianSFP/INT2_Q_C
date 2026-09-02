#!/usr/bin/env python3
"""Independent no-model reproducibility tests for the frozen v1 package."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[1]
EXTERNAL = REPO.parent
PRODUCER = REVIEW.parent / "strata_rm_global_swap_v1_hardened"
V0 = REVIEW.parent / "strata_rm_global_swap_v0"
V0_AUDIT = REVIEW.parent / "strata_rm_global_swap_v0_independent_source_audit_20260902"
sys.path.insert(0, str(REVIEW))
from independent_auth import (EXTERNAL_PINS, PRODUCER_MANIFEST_SHA256,
                              PRODUCER_SOURCE_ROOT_SHA256, ReviewError,
                              authenticate_external_sources,
                              authenticate_producer)
from fixture_tools import bf16, make_fixture

sys.path.insert(0, str(PRODUCER))
import authority as producer_authority
import physical_authority


class ClosureAndSerializationTests(unittest.TestCase):
    def test_independent_exact_producer_closure(self):
        row = authenticate_producer(PRODUCER)
        self.assertEqual(row["source_root_sha256"], PRODUCER_SOURCE_ROOT_SHA256)
        self.assertEqual(row["manifest_sha256"], PRODUCER_MANIFEST_SHA256)
        self.assertEqual(row["payloads_opened"], 0)

    def test_producer_authenticator_agrees_on_root(self):
        row = producer_authority.authenticate_v1_package(
            PRODUCER, PRODUCER_MANIFEST_SHA256)
        self.assertEqual(row["source_root_sha256"], PRODUCER_SOURCE_ROOT_SHA256)

    def test_unmanifested_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            clone = Path(td) / "producer"
            shutil.copytree(PRODUCER, clone)
            (clone / "cupy").mkdir()
            with self.assertRaises(Exception):
                authenticate_producer(clone)
            with self.assertRaises(Exception):
                producer_authority.authenticate_v1_package(
                    clone, PRODUCER_MANIFEST_SHA256)

    def test_manifest_member_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            clone = Path(td) / "producer"
            shutil.copytree(PRODUCER, clone)
            with (clone / "rm_order.py").open("ab") as stream:
                stream.write(b"\n# review mutation\n")
            with self.assertRaises(Exception):
                authenticate_producer(clone)

    def test_standalone_verifier_root_link_gap_when_links_are_supported(self):
        """Record the static verifier's resolve-before-link-check divergence."""
        with tempfile.TemporaryDirectory() as td:
            link = Path(td) / "producer-link"
            try:
                link.symlink_to(PRODUCER, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks unavailable")
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(PRODUCER / "verify_source.py"),
                 "--package", str(link), "--expected-manifest-sha256",
                 PRODUCER_MANIFEST_SHA256], check=False, capture_output=True,
                text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with self.assertRaises(ReviewError):
                authenticate_producer(link)

    def test_canonical_json_and_strict_parser(self):
        left = producer_authority.canonical_json({"b": 2, "a": 1})
        right = producer_authority.canonical_json({"a": 1, "b": 2})
        self.assertEqual(left, b'{"a":1,"b":2}')
        self.assertEqual(left, right)
        with self.assertRaises(Exception):
            producer_authority.strict_json(b'{"a":1,"a":2}', "duplicate")
        with self.assertRaises(Exception):
            producer_authority.strict_json(b'{"a":NaN}', "nonfinite")


class DependencyAndWorkerTests(unittest.TestCase):
    def test_exact_dependency_and_external_source_pins(self):
        dependencies = producer_authority.authenticate_dependencies(V0, V0_AUDIT)
        external = authenticate_external_sources(EXTERNAL)
        self.assertEqual(dependencies["status"],
                         "PASS_PINNED_V0_AND_AUDIT_CLOSURE")
        self.assertEqual(external["external_pins"], EXTERNAL_PINS)

    def test_worker_is_launched_from_reauthenticated_snapshot(self):
        captured = {}

        def fake_run(command, **_kwargs):
            captured["command"] = list(command)
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(producer_authority.canonical_json({
                "external_pins": dict(EXTERNAL_PINS),
                "fresh_interpreter": True, "python_isolated_flag": True,
                "pythonpath_inherited": False, "payloads_opened": 0,
            }) + b"\n")
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        with mock.patch.object(producer_authority.subprocess, "run", fake_run):
            row = producer_authority.run_isolated_worker(
                PRODUCER, expected_manifest_sha256=PRODUCER_MANIFEST_SHA256,
                worker_name="current_integration_worker.py",
                external_root=EXTERNAL)
        worker = Path(captured["command"][3]).resolve()
        self.assertNotEqual(worker, (PRODUCER / "current_integration_worker.py").resolve())
        self.assertEqual(worker.parent.name, "source")
        self.assertEqual(captured["command"][1:3], ["-I", "-B"])
        self.assertEqual(row["payloads_opened"], 0)

    def test_worker_environment_drops_python_import_controls(self):
        row = producer_authority.sanitized_worker_environment({
            "PATH": "safe", "PYTHONPATH": "hostile", "PYTHONHOME": "hostile",
            "CUDA_VISIBLE_DEVICES": "0", "UNRELATED": "hostile",
        })
        self.assertEqual(row["PATH"], "safe")
        self.assertNotIn("PYTHONPATH", row)
        self.assertNotIn("PYTHONHOME", row)
        self.assertNotIn("UNRELATED", row)
        self.assertEqual(row["PYTHONNOUSERSITE"], "1")

    def test_current_hook_worker_has_no_injected_hook_cli(self):
        text = (PRODUCER / "current_integration_worker.py").read_text(
            encoding="utf-8")
        self.assertNotIn('add_argument("--hook"', text)
        self.assertNotIn('add_argument("--module"', text)
        self.assertIn("bg.base is base", text)
        self.assertIn("base.reliability_freeze_flags is bg.bec_flags", text)
        self.assertIn("final authenticated hook identity", text)


class PacketAndNumericalAuthorityTests(unittest.TestCase):
    def test_synthetic_packet_decodes_and_reencodes_byte_identically(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = make_fixture(Path(td), producer=PRODUCER,
                                   external_root=EXTERNAL)
            row = physical_authority.validate_physical_bundle(
                evidence_root=fixture["evidence"], external_root=EXTERNAL,
                commitment_path=fixture["commitment_path"],
                expected_commitment_sha256=fixture["commitment_sha256"],
                authorization=physical_authority.FIXTURE_AUTHORIZATION)
        self.assertTrue(row["canonical_packets_compared_as_bytes"])
        self.assertTrue(row["source_metrics_recomputed_from_exact_bf16"])
        self.assertEqual(row["pooled"]["relative_mse"], 0.0)
        self.assertEqual(row["pooled"]["maximum_read_amplification"], 1.0)
        self.assertIn("NO_QWEN", row["status"])

    def test_recommitted_corrupt_packet_is_rejected_by_decoder(self):
        def mutate(payload):
            result = bytearray(payload)
            result[-8] ^= 1
            return bytes(result)

        with tempfile.TemporaryDirectory() as td:
            fixture = make_fixture(Path(td), producer=PRODUCER,
                                   external_root=EXTERNAL,
                                   mutate_packet=mutate)
            with self.assertRaises(Exception):
                physical_authority.validate_physical_bundle(
                    evidence_root=fixture["evidence"], external_root=EXTERNAL,
                    commitment_path=fixture["commitment_path"],
                    expected_commitment_sha256=fixture["commitment_sha256"],
                    authorization=physical_authority.FIXTURE_AUTHORIZATION)

    def test_source_mutation_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = make_fixture(Path(td), producer=PRODUCER,
                                   external_root=EXTERNAL)
            path = fixture["evidence"] / "source-0.bf16"
            payload = bytearray(path.read_bytes())
            payload[0] ^= 1
            path.write_bytes(payload)
            with self.assertRaises(Exception):
                physical_authority.validate_physical_bundle(
                    evidence_root=fixture["evidence"], external_root=EXTERNAL,
                    commitment_path=fixture["commitment_path"],
                    expected_commitment_sha256=fixture["commitment_sha256"],
                    authorization=physical_authority.FIXTURE_AUTHORIZATION)

    def test_exact_bf16_to_f64_recomputation_fields(self):
        source = bf16([1.0, -2.0, 4.0])
        reconstruction = struct.pack("<ddd", 1.0, -1.0, 2.0)
        row = physical_authority.exact_bf16_f64_score(source, reconstruction)
        self.assertEqual(row["weights"], 3)
        self.assertEqual(float.fromhex(row["sse_fp64_hex"]), 5.0)
        self.assertEqual(float.fromhex(row["energy_fp64_hex"]), 21.0)
        self.assertAlmostEqual(row["relative_mse"], 5.0 / 21.0)

    def test_nonfinite_bf16_source_is_rejected(self):
        source = struct.pack("<H", 0x7F80)
        reconstruction = struct.pack("<d", 0.0)
        with self.assertRaises(Exception):
            physical_authority.exact_bf16_f64_score(source, reconstruction)


class ScopeAndProvenanceBoundaryTests(unittest.TestCase):
    def test_execution_status_truthfully_holds_all_payload_authority(self):
        status = json.loads((PRODUCER / "EXECUTION_STATUS.json").read_text(
            encoding="utf-8"))
        self.assertFalse(status["python_source_and_hostile_suites"]["executed"])
        self.assertFalse(status["isolated_current_hook_worker"]["executed"])
        self.assertFalse(status["isolated_real_cupy_worker"]["executed"])
        self.assertEqual(status["payloads_opened"], 0)
        self.assertIsNone(status["qwen_result"])
        self.assertIn("HOLD_RUNPOD_AND_PAYLOAD", status["status"])

    def test_read_amplification_is_derived_from_decoder_reported_trace(self):
        row = physical_authority._validate_trace({
            "schema": "strata-rm-v1-read-trace", "packet_bytes": 8,
            "operations": [{"object": "packet", "offset": 0, "length": 8}],
        }, 8)
        self.assertEqual(row["read_amplification"], 1.0)
        self.assertTrue(row["one_expert_local_object"])

    def test_case_schema_has_geometry_but_no_model_provenance_field(self):
        sources = []
        for ordinal, (role, shape) in enumerate((
                ("gate", [2, 3]), ("up", [2, 3]), ("down", [3, 2]))):
            sources.append({
                "ordinal": ordinal, "role": role, "layer": 0, "expert": 0,
                "shape": shape, "source_relative_path": f"s{ordinal}.bf16",
                "source_bytes": 2 * shape[0] * shape[1],
                "source_sha256": "0" * 64,
            })
        case = {
            "case_id": "schema-only", "kind": "qwen_bf16",
            "architecture_family": "declared-family", "pipeline_id": "declared",
            "matched_case_id": "control", "packet": {
                "relative_path": "packet.bin", "bytes": 1, "sha256": "0" * 64},
            "sources": sources, "charged_shared_bytes": 0,
        }
        physical_authority._case_schema(case)
        self.assertNotIn("checkpoint_sha256", case)
        self.assertNotIn("model_manifest", case)

    def test_production_decoder_audit_pin_is_commitment_supplied(self):
        parameters = inspect.signature(
            physical_authority.validate_physical_bundle).parameters
        self.assertNotIn("expected_decoder_audit_manifest_sha256", parameters)
        self.assertNotIn("expected_decoder_audit_source_root_sha256", parameters)
        text = inspect.getsource(physical_authority.validate_physical_bundle)
        self.assertIn('worker_row["independent_audit"]', text)

    def test_physical_decoder_command_uses_external_worker_path_directly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = root / "worker.py"
            request = root / "request.json"
            packet = root / "packet.bin"
            output = root / "output"
            command = physical_authority._decoder_command(
                worker, request, packet, output)
        self.assertEqual(Path(command[3]), worker)
        self.assertEqual(command[1:3], ["-I", "-B"])

    def test_target_pool_and_read_max_are_qwen_only_in_current_source(self):
        text = inspect.getsource(physical_authority.validate_physical_bundle)
        self.assertIn('total_weights = sum(row["weights"] for row in qwen)', text)
        self.assertIn('for row in qwen)', text)
        self.assertIn('families == {row["architecture_family"] for row in model_cases}',
                      text)


class AcceleratorProvenanceTests(unittest.TestCase):
    def test_backend_inside_controlled_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            name = "rm_swap_review_fake_backend"
            (root / f"{name}.py").write_text("fabricated = True\n",
                                             encoding="utf-8")
            sys.path.insert(0, str(root))
            try:
                with self.assertRaises(Exception):
                    producer_authority.module_origin_outside_controlled_roots(
                        name, [root])
            finally:
                sys.path.pop(0)
                sys.modules.pop(name, None)

    def test_real_cupy_worker_has_required_runtime_and_sync_checks(self):
        text = (PRODUCER / "real_cupy_worker.py").read_text(encoding="utf-8")
        for required in (
                'require("cupy" not in sys.modules and "numpy" not in sys.modules',
                "getDeviceCount", "runtimeGetVersion", "driverGetVersion",
                "cp.cuda.Stream.null.synchronize()", "observed_probe == expected_probe",
                "np.array_equal(actual, expected)"):
            self.assertIn(required, text)
        self.assertIn(
            "from rm_order import TARGET_N, rm_full_order_cupy, rm_full_order_numpy",
            text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
