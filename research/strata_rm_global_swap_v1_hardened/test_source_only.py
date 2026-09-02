#!/usr/bin/env python3
"""Source-only mechanism and physical-authority fixture tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
EXTERNAL = REPO.parent
V0 = PACKAGE.parent / "strata_rm_global_swap_v0"
AUDIT = PACKAGE.parent / "strata_rm_global_swap_v0_independent_source_audit_20260902"
sys.path.insert(0, str(PACKAGE))
from authority import (V0_AUDIT_SOURCE_ROOT_SHA256, V0_SOURCE_ROOT_SHA256,
                       authenticate_current_external_root,
                       authenticate_dependencies, authenticate_flat_package,
                       isolated_worker_command, sanitized_worker_environment)
from physical_authority import (FIXTURE_AUTHORIZATION, exact_bf16_f64_score,
                                validate_physical_bundle)
from rm_order import phase_key


MAGIC = b"SRMGF1\0\0"
PREFIX = struct.Struct("<8sII")
TRAILER = struct.Struct("<I")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def bf16(values) -> bytes:
    rows = []
    for value in values:
        bits, = struct.unpack("<I", struct.pack("<f", float(value)))
        rows.append((bits >> 16) & 0xFFFF)
    return struct.pack("<" + "H" * len(rows), *rows)


def fixture_packet(reconstructions) -> bytes:
    chunks = [struct.pack("<" + "d" * len(row), *row) for row in reconstructions]
    header = canonical({"schema": "strata-rm-v1-authority-fixture-packet",
                        "reconstruction_f64_bytes": [len(row) for row in chunks]})
    payload = b"".join(chunks)
    packet = PREFIX.pack(MAGIC, len(header), len(payload)) + header + payload
    return packet + TRAILER.pack(zlib.crc32(packet) & 0xFFFFFFFF)


def make_fixture(root: Path, *, packet_mutator=None, extra_commitment=None):
    evidence = root / "evidence"
    evidence.mkdir()
    values = ([1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5],
              [2.0, 2.0, -1.0, -1.0])
    sources = []
    reconstructions = []
    for ordinal, (role, row) in enumerate(zip(("gate", "up", "down"), values,
                                               strict=True)):
        payload = bf16(row)
        name = f"source-{ordinal}.bf16"
        (evidence / name).write_bytes(payload)
        sources.append({"ordinal": ordinal, "role": role, "layer": 0,
                        "expert": 0, "shape": [2, 2],
                        "source_relative_path": name, "source_bytes": len(payload),
                        "source_sha256": hashlib.sha256(payload).hexdigest()})
        reconstructions.append(list(row))
    packet = fixture_packet(reconstructions)
    if packet_mutator is not None:
        packet = packet_mutator(packet)
    (evidence / "packet.bin").write_bytes(packet)
    worker_rel = PACKAGE.relative_to(EXTERNAL).as_posix() + "/fixture_decoder_worker.py"
    worker = PACKAGE / "fixture_decoder_worker.py"
    commitment = {
        "schema": "strata-rm-global-swap-v1-physical-commitment",
        "mode": "synthetic_authority_fixture",
        "v0_source_root_sha256": V0_SOURCE_ROOT_SHA256,
        "v0_audit_source_root_sha256": V0_AUDIT_SOURCE_ROOT_SHA256,
        "external_pins": {
            "agent_polaris_qwen_rht_encoder.py":
                "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
            "bg_codec_bec_encoder.py":
                "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267",
            "strata_v2_klt_mixed_independent_auditor_v1.py":
                "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"},
        "decoder_worker": {"relative_path": worker_rel,
                           "sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
                           "protocol": "strata-rm-v1-decoder-worker-protocol",
                           "independent_from_encoder": True,
                           "independent_audit": None},
        "cases": [{"case_id": "fixture-0", "kind": "synthetic_fixture",
                   "architecture_family": "synthetic", "pipeline_id": "fixture-v1",
                   "matched_case_id": None,
                   "packet": {"relative_path": "packet.bin", "bytes": len(packet),
                              "sha256": hashlib.sha256(packet).hexdigest()},
                   "sources": sources, "charged_shared_bytes": 0}],
        "universal_contract": {"roles": ["gate", "up", "down"],
            "shape_parameterized": True, "qwen_specific_tables": False,
            "model_family_agnostic": True,
            "architecture_families": ["synthetic"]},
        "shared_model_bytes": 0, "selection_frozen_before_test": True,
        "test_bytes_opened_during_selection": False,
    }
    if extra_commitment:
        extra_commitment(commitment)
    commitment_path = evidence / "commitment.json"
    payload = canonical(commitment) + b"\n"
    commitment_path.write_bytes(payload)
    return evidence, commitment_path, hashlib.sha256(payload).hexdigest(), commitment


class DependencyAuthorityTests(unittest.TestCase):
    def test_exact_v0_and_audit_roots(self):
        row = authenticate_dependencies(V0, AUDIT)
        self.assertEqual(row["v0"]["source_root_sha256"], V0_SOURCE_ROOT_SHA256)
        self.assertEqual(row["v0_independent_audit"]["source_root_sha256"],
                         V0_AUDIT_SOURCE_ROOT_SHA256)

    def test_current_external_sources_and_semantics(self):
        row = authenticate_current_external_root(EXTERNAL)
        self.assertEqual(row["reference_hook"], "bec_flags")

    def test_unmanifested_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            clone = Path(td) / "v0"
            shutil.copytree(V0, clone)
            (clone / "cupy").mkdir()
            with self.assertRaises(Exception):
                authenticate_dependencies(clone, AUDIT)

    def test_symlink_member_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as td:
            clone = Path(td) / "v0"
            shutil.copytree(V0, clone)
            target = clone / "rm_order.py"
            saved = clone / "saved.py"
            target.rename(saved)
            try:
                target.symlink_to(saved.name)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(Exception):
                authenticate_dependencies(clone, AUDIT)


class HookAndBackendBoundaryTests(unittest.TestCase):
    def test_worker_command_is_fixed_isolated_and_has_no_hook_argument(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "out.json"
            command = isolated_worker_command(
                PACKAGE, worker_name="current_integration_worker.py",
                external_root=EXTERNAL, output=output)
        self.assertEqual(command[1:3], ["-I", "-B"])
        self.assertNotIn("--hook", command)
        self.assertNotIn("--module", command)

    def test_python_import_facade_environment_is_removed(self):
        environment = {"PATH": "safe", "PYTHONPATH": "hostile",
                       "PYTHONHOME": "hostile", "CUDA_VISIBLE_DEVICES": "0"}
        result = sanitized_worker_environment(environment)
        self.assertNotIn("PYTHONPATH", result)
        self.assertNotIn("PYTHONHOME", result)
        self.assertEqual(result["PYTHONNOUSERSITE"], "1")

    def test_normative_rm_tie_order(self):
        for m in range(1, 10):
            n = 1 << m
            order = sorted(range(n), key=lambda phase: phase_key(phase, n))
            self.assertEqual(len(order), n)
            self.assertEqual(len(set(order)), n)


class PhysicalAuthorityTests(unittest.TestCase):
    def test_exact_bf16_source_metric(self):
        source = bf16([1.0, -2.0, 4.0])
        reconstruction = struct.pack("<ddd", 1.0, -1.0, 2.0)
        row = exact_bf16_f64_score(source, reconstruction)
        self.assertEqual(row["weights"], 3)
        self.assertEqual(float.fromhex(row["sse_fp64_hex"]), 5.0)
        self.assertEqual(float.fromhex(row["energy_fp64_hex"]), 21.0)

    def test_literal_fixture_decode_and_byte_reencode(self):
        with tempfile.TemporaryDirectory() as td:
            evidence, commitment, digest, _ = make_fixture(Path(td))
            row = validate_physical_bundle(
                evidence_root=evidence, external_root=EXTERNAL,
                commitment_path=commitment, expected_commitment_sha256=digest,
                authorization=FIXTURE_AUTHORIZATION)
        self.assertTrue(row["canonical_packets_compared_as_bytes"])
        self.assertEqual(row["pooled"]["relative_mse"], 0.0)
        self.assertEqual(row["pooled"]["maximum_read_amplification"], 1.0)
        self.assertIn("NO_QWEN", row["status"])

    def test_fabricated_no_packet_receipt_is_impossible(self):
        with tempfile.TemporaryDirectory() as td:
            evidence, commitment, digest, _ = make_fixture(Path(td))
            (evidence / "packet.bin").unlink()
            with self.assertRaises(Exception):
                validate_physical_bundle(
                    evidence_root=evidence, external_root=EXTERNAL,
                    commitment_path=commitment, expected_commitment_sha256=digest,
                    authorization=FIXTURE_AUTHORIZATION)

    def test_self_declared_metric_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            evidence, commitment, digest, _ = make_fixture(
                Path(td), extra_commitment=lambda row: row.__setitem__("F", 0.0))
            with self.assertRaises(Exception):
                validate_physical_bundle(
                    evidence_root=evidence, external_root=EXTERNAL,
                    commitment_path=commitment, expected_commitment_sha256=digest,
                    authorization=FIXTURE_AUTHORIZATION)

    def test_packet_mutation_even_when_recommitted_fails_decoder(self):
        def mutate(packet):
            row = bytearray(packet)
            row[-8] ^= 1
            return bytes(row)
        with tempfile.TemporaryDirectory() as td:
            evidence, commitment, digest, _ = make_fixture(Path(td), packet_mutator=mutate)
            with self.assertRaises(Exception):
                validate_physical_bundle(
                    evidence_root=evidence, external_root=EXTERNAL,
                    commitment_path=commitment, expected_commitment_sha256=digest,
                    authorization=FIXTURE_AUTHORIZATION)

    def test_source_mutation_fails_external_pin(self):
        with tempfile.TemporaryDirectory() as td:
            evidence, commitment, digest, _ = make_fixture(Path(td))
            path = evidence / "source-0.bf16"
            row = bytearray(path.read_bytes())
            row[0] ^= 1
            path.write_bytes(row)
            with self.assertRaises(Exception):
                validate_physical_bundle(
                    evidence_root=evidence, external_root=EXTERNAL,
                    commitment_path=commitment, expected_commitment_sha256=digest,
                    authorization=FIXTURE_AUTHORIZATION)

    def test_production_commitment_requires_matched_and_universal_fields(self):
        with tempfile.TemporaryDirectory() as td:
            evidence, path, digest, _ = make_fixture(
                Path(td), extra_commitment=lambda row: row.__setitem__(
                    "mode", "production_global_rm_swap"))
            with self.assertRaises(Exception):
                validate_physical_bundle(
                    evidence_root=evidence, external_root=EXTERNAL,
                    commitment_path=path, expected_commitment_sha256=digest,
                    authorization="AUDIT_LITERAL_GLOBAL_RM_SWAP_RESULT_V1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
