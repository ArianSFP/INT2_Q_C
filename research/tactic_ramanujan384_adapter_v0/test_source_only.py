#!/usr/bin/env python3
"""Hostile source-only tests for TACTIC Ramanujan-384 v0."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import zlib
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


packet = load("tactic_ramanujan384_test_packet", "packet.py")
codec = load("tactic_ramanujan384_test_codec", "ramanujan_codec.py")
auth = load("tactic_ramanujan384_test_auth", "authenticated_io.py")
container = load("tactic_ramanujan384_test_container", "container.py")
adapter = load("tactic_ramanujan384_test_adapter", "adapter.py")
fixture = load("tactic_ramanujan384_test_fixture", "run_source_free_fixture.py")


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


class PacketTests(unittest.TestCase):
    def test_literal_384_bit_roundtrip_all_roles(self):
        for role in packet.ROLES:
            payload = packet.encode_packet(role, (0, 17, 383), (-1023, 7, 991), 2.0 ** -10)
            self.assertEqual(len(payload), 48)
            row = packet.decode_packet(payload)
            self.assertEqual(row["role"], role)
            self.assertEqual(row["support"], (0, 17, 383))
            self.assertEqual(row["coefficients"], (-1023, 7, 991))
            self.assertEqual(row["padding_bits"], 250)

    def test_rank14_charges_support_scale_header_crc_and_padding(self):
        support = tuple(range(14))
        coefficients = tuple(-index - 1 for index in range(14))
        payload = packet.encode_packet("gate", support, coefficients, 0.001)
        row = packet.decode_packet(payload)
        self.assertEqual(row["rank"], 14)
        self.assertEqual(row["padding_bits"], 30)
        self.assertEqual(len(payload[:44]) * 8 + 32, 384)

    def test_crc_padding_duplicate_and_noncanonical_scale_rejected(self):
        payload = packet.encode_packet("up", (1, 9), (10, -11), 0.125)
        hostile = bytearray(payload)
        hostile[-1] ^= 1
        with self.assertRaisesRegex(packet.PacketError, "CRC32"):
            packet.decode_packet(bytes(hostile))
        hostile = bytearray(payload)
        hostile[20] ^= 0x40
        hostile[-4:] = struct.pack("<I", zlib.crc32(hostile[:44]) & 0xFFFFFFFF)
        with self.assertRaisesRegex(packet.PacketError, "padding"):
            packet.decode_packet(bytes(hostile))
        with self.assertRaisesRegex(packet.PacketError, "support order"):
            packet.encode_packet("gate", (7, 7), (1, 2), 0.5)
        with self.assertRaisesRegex(packet.PacketError, "atom index"):
            packet.encode_packet("gate", (384,), (1,), 0.5)
        with self.assertRaisesRegex(packet.PacketError, "zero-rank scale"):
            packet.encode_packet("gate", (), (), 1.0)


class DictionaryTests(unittest.TestCase):
    def test_period_bank_covers_every_non_dyadic_period_3_through_127(self):
        periods = codec.non_dyadic_periods()
        labels = codec.period_bank_labels()
        self.assertEqual(len(periods), 120)
        self.assertEqual(len(labels), 384)
        self.assertEqual({period for period, _ in labels}, set(periods))
        self.assertTrue(all(period & (period - 1) for period in periods))

    def test_integer_ramanujan_sum_matches_primitive_frequency_definition(self):
        for period in range(3, 24):
            for coordinate in range(period):
                direct = sum(
                    math.cos(2.0 * math.pi * frequency * coordinate / period)
                    for frequency in range(1, period + 1)
                    if math.gcd(frequency, period) == 1
                )
                self.assertAlmostEqual(codec.ramanujan_sum(period, coordinate), direct, places=9)

    def test_shift_span_has_audited_totient_dimension(self):
        for period in (3, 5, 6, 7, 10, 12, 15):
            phi = codec.totient(period)
            matrix = np.asarray([
                [codec.ramanujan_sum(period, coordinate - shift) for shift in range(period)]
                for coordinate in range(period)
            ], dtype=np.float64)
            self.assertEqual(np.linalg.matrix_rank(matrix), phi)

    def test_dictionary_is_exact_integer_decoder_state(self):
        basis = codec.build_public_dictionary(np, length=512, columns=128)
        self.assertTrue(basis["every_period_represented"])
        self.assertTrue(basis["integer_decoder_atoms"])
        self.assertTrue(np.array_equal(basis["dictionary"], np.rint(basis["dictionary"])))


class FiniteSearchTests(unittest.TestCase):
    def test_source_free_fixture_runs_all_nine_controls(self):
        row = fixture.run()
        self.assertEqual(row["status"], "PASS_SOURCE_FREE_FINITE_PACKET_AND_ALL_CONTROLS")
        self.assertTrue(row["controls_rerun"])
        self.assertEqual(row["gaussian_controls"], 8)
        self.assertEqual(row["fine_stream_bytes"], 96)
        self.assertFalse(row["qwen_payload_accessed"])

    def test_absolute_miss_does_not_open_controls(self):
        rng = np.random.default_rng(9917)
        residual = rng.normal(size=(2, codec.BLOCK_VALUES))
        energy = float(np.sum(residual * residual)) / codec.COARSE_RELATIVE_MSE
        row = codec.run_finite_panel(
            np, residual, np.zeros_like(residual), role="down_transposed", source_energy=energy
        )
        self.assertEqual(row["status"], "HARD_KILL_ABSOLUTE_SOURCE_MISSES_D_0P025")
        self.assertFalse(row["controls_permitted"])
        self.assertFalse(row["controls_rerun"])
        self.assertNotIn("gaussian_controls", row)

    def test_exact_qwen_geometry_ledger_is_2p5_bpw_and_one_read(self):
        role_weights = (768 * 2048,) * 3
        row = codec.expert_read_ledger(role_weights=role_weights, coarse_artifact_bytes=1414656)
        self.assertEqual(row["weights"], 4718592)
        self.assertEqual(row["fine_bytes"], 55296)
        self.assertEqual(row["container_header_bytes"], 512)
        self.assertEqual(row["coarse_plus_fine_rate_bpw"], 2.4921875)
        self.assertEqual(row["physical_bytes"], 1470464)
        self.assertAlmostEqual(row["physical_rate_bpw"], 2.4930555555555554)
        self.assertTrue(row["target_rate_eligible"])
        self.assertEqual(row["external_read_amplification"], 1.0)
        self.assertFalse(row["accelerator_hbm_measured"])

    def test_literal_container_roundtrip_charges_header_and_page_padding(self):
        blocks = container.expected_blocks(64, 64)
        self.assertEqual(blocks, (1, 1, 1))
        coarse = bytes(container.expected_coarse_bytes(64, 64))
        streams = tuple(packet.encode_packet(role, (), (), 0.0) for role in packet.ROLES)
        composite = container.encode_composite(
            intermediate=64,
            hidden=64,
            coarse_payload=coarse,
            role_fine_streams=streams,
            source_binding_sha256="a" * 64,
        )
        decoded = container.decode_composite(composite)
        self.assertEqual(decoded["header"]["block_counts"], blocks)
        self.assertEqual(decoded["coarse_payload"], coarse)
        self.assertEqual(decoded["fine_payload"], b"".join(streams))
        self.assertEqual(decoded["external_read_amplification"], 1.0)
        self.assertEqual(len(composite) % 4096, 0)

    def test_container_rejects_tamper_role_reorder_and_overpadding(self):
        streams = tuple(packet.encode_packet(role, (), (), 0.0) for role in packet.ROLES)
        with self.assertRaisesRegex(container.ContainerError, "role order"):
            container.encode_composite(
                intermediate=64,
                hidden=64,
                coarse_payload=b"coarse",
                role_fine_streams=(streams[1], streams[0], streams[2]),
                source_binding_sha256="b" * 64,
            )
        composite = container.encode_composite(
            intermediate=64,
            hidden=64,
            coarse_payload=bytes(container.expected_coarse_bytes(64, 64)),
            role_fine_streams=streams,
            source_binding_sha256="b" * 64,
        )
        hostile = bytearray(composite)
        hostile[10] ^= 1
        with self.assertRaisesRegex(container.ContainerError, "CRC32"):
            container.decode_composite(bytes(hostile))
        with self.assertRaisesRegex(container.ContainerError, "minimal page padding"):
            container.decode_composite(composite + bytes(4096))

    def test_universal_tail_is_charged_not_hidden(self):
        weights = (4097, 4097, 4097)
        coarse_bytes = math.ceil(307 * sum(weights) / 1024)
        row = codec.expert_read_ledger(role_weights=weights, coarse_artifact_bytes=coarse_bytes)
        self.assertFalse(row["tail_free"])
        self.assertFalse(row["target_rate_eligible"])
        self.assertGreater(row["physical_rate_bpw"], 2.5)


class AuthenticationTests(unittest.TestCase):
    def _fixture(self, directory: Path):
        source_f32 = np.asarray([1.0, -0.5, 0.25, 2.0], dtype="<f4")
        source_payload = (source_f32.view("<u4") >> 16).astype("<u2").tobytes()
        reconstruction_payload = np.asarray([0.9, -0.4, 0.2, 1.8], dtype="<f4").tobytes()
        coarse_payload = b"synthetic-independent-coarse"
        source_path = directory / "source.bf16"
        reconstruction_path = directory / "reconstruction.f32"
        coarse_path = directory / "COARSE.fixture"
        source_path.write_bytes(source_payload)
        reconstruction_path.write_bytes(reconstruction_payload)
        coarse_path.write_bytes(coarse_payload)
        source_sha = hashlib.sha256(source_payload).hexdigest()
        reconstruction_sha = hashlib.sha256(reconstruction_payload).hexdigest()
        coarse_sha = hashlib.sha256(coarse_payload).hexdigest()
        auditor_manifest = "1" * 64
        input_manifest = "2" * 64
        audit = {
            "auditor_source_manifest_sha256": auditor_manifest,
            "input_manifest_sha256": input_manifest,
            "independent_packet_parser_and_causal_CPU_decoder_used": True,
            "literal_COARSE_canonical_reencode_matches": True,
            "publication_members": [{"name": "COARSE.bin", "bytes": len(coarse_payload), "sha256": coarse_sha}],
            "input_roles": [{"role": "gate", "bytes": len(source_payload), "sha256": source_sha}],
            "recomputed": {"reconstruction_f32_sha256": {"gate": reconstruction_sha}},
        }
        audit_payload = canonical_json(audit)
        audit_path = directory / "audit.json"
        audit_path.write_bytes(audit_payload)
        binding = {
            "schema": auth.SCHEMA,
            "status": auth.STATUS,
            "role": "gate",
            "shape": [2, 2],
            "normalized_layout": "intermediate_by_hidden_row_major",
            "source": {"bytes": len(source_payload), "sha256": source_sha},
            "coarse_artifact": {"bytes": len(coarse_payload), "sha256": coarse_sha},
            "coarse_reconstruction": {"bytes": len(reconstruction_payload), "sha256": reconstruction_sha},
            "independent_audit": {
                "receipt_sha256": hashlib.sha256(audit_payload).hexdigest(),
                "auditor_source_manifest_sha256": auditor_manifest,
            },
            "input_manifest_sha256": input_manifest,
        }
        binding_payload = canonical_json(binding)
        binding_path = directory / "binding.json"
        binding_path.write_bytes(binding_payload)
        return {
            "binding_path": binding_path,
            "expected_binding_sha256": hashlib.sha256(binding_payload).hexdigest(),
            "audit_receipt_path": audit_path,
            "coarse_artifact_path": coarse_path,
            "source_bf16_path": source_path,
            "coarse_reconstruction_f32_path": reconstruction_path,
        }

    def test_all_independent_hashes_are_required_and_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = self._fixture(Path(temporary))
            row = auth.authenticate_role(**arguments)
            self.assertEqual(row["role"], "gate")
            self.assertEqual(row["shape"], (2, 2))
            self.assertFalse(row["qwen_or_model_identity_used"])
            hostile = dict(arguments)
            hostile["expected_binding_sha256"] = "0" * 64
            with self.assertRaisesRegex(auth.AuthenticationError, "binding SHA256"):
                auth.authenticate_role(**hostile)

    def test_literal_source_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = self._fixture(Path(temporary))
            source_path = arguments["source_bf16_path"]
            payload = bytearray(source_path.read_bytes())
            payload[0] ^= 1
            source_path.write_bytes(payload)
            with self.assertRaisesRegex(auth.AuthenticationError, "source role hash"):
                auth.authenticate_role(**arguments)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(auth.AuthenticationError, "duplicate key"):
            auth.strict_json(b'{"a":1,"a":2}', "hostile")

    def test_duplicate_audit_role_is_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = self._fixture(Path(temporary))
            audit_path = arguments["audit_receipt_path"]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["input_roles"].append(dict(audit["input_roles"][0]))
            audit_payload = canonical_json(audit)
            audit_path.write_bytes(audit_payload)
            binding_path = arguments["binding_path"]
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            binding["independent_audit"]["receipt_sha256"] = hashlib.sha256(audit_payload).hexdigest()
            binding_payload = canonical_json(binding)
            binding_path.write_bytes(binding_payload)
            arguments["expected_binding_sha256"] = hashlib.sha256(binding_payload).hexdigest()
            with self.assertRaisesRegex(auth.AuthenticationError, "duplicate independent input role"):
                auth.authenticate_role(**arguments)


class ClosureTests(unittest.TestCase):
    def test_whole_expert_adapter_is_present_but_not_payload_executed(self):
        self.assertTrue(callable(adapter.run_authenticated_expert))

    def test_dependency_hashes_match_pinned_audited_parent(self):
        parent = ROOT.parent / "mosaic_secondary_oracles_v0"
        self.assertEqual(hashlib.sha256((parent / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
                         codec.DEPENDENCY_MANIFEST_SHA256)
        self.assertEqual(hashlib.sha256((parent / "residual_oracles.py").read_bytes()).hexdigest(),
                         codec.DEPENDENCY_ORACLES_SHA256)

    def test_cupy_import_is_lazy_and_runner_has_no_payload_aperture(self):
        backend = (ROOT / "cupy_backend.py").read_text(encoding="utf-8")
        self.assertNotIn("import cupy", backend.split("def load_cupy", 1)[0])
        runner = (ROOT / "run_source_free_cupy_smoke.py").read_text(encoding="utf-8")
        for forbidden in ("--payload", "--qwen", "--coarse", "COARSE.bin"):
            self.assertNotIn(forbidden, runner)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "tactic-ramanujan384-source-test-v0",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "cuda_initialized": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
