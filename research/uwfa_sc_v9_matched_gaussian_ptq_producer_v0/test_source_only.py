#!/usr/bin/env python3
"""Source-only tests. No model/artifact paths and no CUDA imports."""

from __future__ import annotations

import hashlib
import base64
import importlib.util
import json
import math
import struct
import sys
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
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


contract = load("gaussian_contract_test", "producer_contract.py")


def fake_moment_contract() -> dict:
    rows = []
    for ordinal, (slot, role) in enumerate(contract.expected_matrix_order()):
        mean = (ordinal - 8.5) * 1.0e-6
        centered = 0.0004 * contract.VALUES_PER_MATRIX * (1.0 + ordinal / 100.0)
        energy = centered + contract.VALUES_PER_MATRIX * mean * mean
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "slot": slot,
                "role": role,
                "shape": list(contract.role_shape(role)),
                "values": contract.VALUES_PER_MATRIX,
                "mean_f64_hex": contract.f64_hex(mean),
                "centered_sse_f64_hex": contract.f64_hex(centered),
                "energy_f64_hex": contract.f64_hex(energy),
                "source_matrix_bf16_sha256": hashlib.sha256(
                    f"source:{ordinal}".encode("ascii")
                ).hexdigest(),
            }
        )
    clean = {
        "schema": contract.MOMENT_CONTRACT_SCHEMA,
        "status": "AUTHENTICATED_EXTERNAL_SOURCE_MOMENTS",
        "moment_semantics": "ORIGINAL_BF16_MATRIX_STORAGE_ORIENTATION_BINARY64_MEAN_AND_CENTERED_SSE",
        "panel": {
            "experts": 6,
            "hidden": 2048,
            "intermediate": 768,
            "roles": ["gate", "up", "down"],
            "weights": 28_311_552,
            "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
        },
        "source_closure": {
            name: hashlib.sha256(name.encode("ascii")).hexdigest()
            for name in (
                "source_artifact_sha256",
                "source_full_geometry_sha256",
                "source_structural_geometry_sha256",
                "source_pipeline_sha256",
                "source_score_receipt_sha256",
                "source_moment_auditor_sha256",
            )
        },
        "matrices": rows,
    }
    return contract.sealed(clean, "moment_contract_sha256")


class ContractTests(unittest.TestCase):
    def test_generator_capsule_contains_replayable_source_bytes(self) -> None:
        members = []
        for name in ("producer_contract.py", "full_ptq_producer.py"):
            payload = (PACKAGE / name).read_bytes()
            members.append(
                {
                    "name": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source_base64": base64.b64encode(payload).decode("ascii"),
                }
            )
        capsule = contract.sealed(
            {
                "schema": "uwfa-sc-v9-gaussian-generator-source-capsule-v1",
                "status": "AUTHENTICATED_SOURCE_BYTES_FOR_INDEPENDENT_REPLAY",
                "members": members,
                "runtime_distribution_closure_sha256": "ab" * 32,
            },
            "capsule_sha256",
        )
        encoded = contract.canonical_json(capsule)
        self.assertEqual(contract.validate_generator_capsule(encoded), capsule)

    def test_contract_and_binary64_roundtrip(self) -> None:
        record, moments = contract.validate_moment_contract(fake_moment_contract())
        self.assertEqual(len(moments), 18)
        self.assertEqual(record["panel"]["identity_semantics"], "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY")
        self.assertEqual(moments[0].shape, (768, 2048))
        self.assertEqual(moments[2].shape, (2048, 768))
        self.assertEqual(struct.unpack("<d", bytes.fromhex(contract.f64_hex(moments[0].mean)))[0], moments[0].mean)

    def test_contract_tamper_rejected(self) -> None:
        value = fake_moment_contract()
        value["matrices"][0]["role"] = "up"
        with self.assertRaisesRegex(ValueError, "slot/role"):
            contract.validate_moment_contract(value)

    def test_generator_key_has_no_model_identity(self) -> None:
        _, moments = contract.validate_moment_contract(fake_moment_contract())
        encoded = contract.canonical_json(moments[0].public_generator_key()).lower()
        for forbidden in (b"qwen", b"model.layers", b"expert_id", b"checkpoint"):
            self.assertNotIn(forbidden, encoded)
        first = contract.derive_matrix_seed(contract.CONTROL_SEEDS[0], moments[0])
        self.assertEqual(first, contract.derive_matrix_seed(contract.CONTROL_SEEDS[0], moments[0]))
        self.assertNotEqual(first, contract.derive_matrix_seed(contract.CONTROL_SEEDS[1], moments[0]))
        self.assertNotEqual(first, contract.derive_matrix_seed(contract.CONTROL_SEEDS[0], moments[1]))

    def test_universal_route_is_slot_role_only(self) -> None:
        payload = contract.build_universal_route()
        self.assertEqual(len(payload), 144)
        for ordinal in range(18):
            namespace, slot, role, axis, groups = struct.unpack_from(">HHBBH", payload, 8 * ordinal)
            self.assertEqual(namespace, 0)
            self.assertEqual(slot, ordinal // 3)
            self.assertEqual(role, ordinal % 3)
            self.assertEqual(axis, 1 if role == 2 else 0)
            self.assertEqual(groups, 768)

    def test_universal_geometry_excludes_source_derived_fields(self) -> None:
        geometry = contract.universal_format_geometry()
        encoded = contract.canonical_json(geometry)
        for forbidden in (b"profile", b"symbols", b"label", b"payload", b"layer", b"qwen"):
            self.assertNotIn(forbidden, encoded.lower())
        self.assertEqual(len(geometry["blocks"]), 15)
        self.assertEqual(geometry["blocks"][-1]["owner_slots"], [4, 5])

    def test_score_and_binding_seals(self) -> None:
        source = fake_moment_contract()["source_closure"]
        score = contract.build_score_receipt(
            artifact_sha256="11" * 32,
            artifact_bytes=8_847_360,
            reconstruction_sha256="22" * 32,
            control_full_geometry_sha256="33" * 32,
            independent_decoder_source_sha256="44" * 32,
            sse=2.5,
            energy=100.0,
        )
        contract.validate_seal(score, "score_receipt_sha256")
        symmetric = {
            name: hashlib.sha256(name.encode("ascii")).hexdigest()
            for name in contract.SYMMETRIC_CODEC_CLOSURE_FIELDS
        }
        binding = contract.build_control_binding_v9(
            seed=contract.CONTROL_SEEDS[0],
            source_closure=source,
            generator_capsule_sha256="55" * 32,
            moment_match_receipt_sha256="66" * 32,
            source_panel_manifest_sha256="77" * 32,
            control_artifact_sha256="11" * 32,
            control_full_geometry_sha256="33" * 32,
            control_structural_geometry_sha256="88" * 32,
            symmetric_codec_closure=symmetric,
        )
        contract.validate_seal(binding, "binding_sha256")
        self.assertEqual(binding["universal_format_geometry_sha256"], contract.universal_format_geometry_sha256())

    def test_symmetric_codec_closure_cannot_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "symmetric codec closure fields"):
            contract.build_control_binding_v9(
                seed=contract.CONTROL_SEEDS[0],
                source_closure=fake_moment_contract()["source_closure"],
                generator_capsule_sha256="55" * 32,
                moment_match_receipt_sha256="66" * 32,
                source_panel_manifest_sha256="77" * 32,
                control_artifact_sha256="11" * 32,
                control_full_geometry_sha256="33" * 32,
                control_structural_geometry_sha256="88" * 32,
                symmetric_codec_closure={},
            )

    def test_direct_entrypoints_are_inert(self) -> None:
        runtime = load("gaussian_runtime_test", "full_ptq_producer.py")
        self.assertEqual(contract.direct_main(), 3)
        self.assertEqual(runtime.direct_main(), 3)

    def test_runtime_source_uses_exact_production_boundaries(self) -> None:
        source = (PACKAGE / "full_ptq_producer.py").read_text(encoding="utf-8")
        required = (
            "v2_emitter.build_staging",
            "run_and_pack.run_block",
            "run_and_pack.pack",
            "independent_auditor.decode_block_worker",
            "adapter.extract_from_current",
            "all_150_wfa_search_run\": False",
        )
        for fragment in required:
            self.assertIn(fragment, source)
        self.assertNotIn("model.layers.", source)


class NumericalGeneratorTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy unavailable")
    def test_small_bf16_generator_determinism_and_moments(self) -> None:
        import numpy as np

        values = 262144
        mean = -0.00125
        centered = values * 0.021 ** 2
        energy = centered + values * mean * mean
        moment = contract.MatrixMoment(
            ordinal=0,
            slot=0,
            role="gate",
            shape=(512, 512),
            values=values,
            mean=mean,
            centered_sse=centered,
            energy=energy,
            source_matrix_bf16_sha256="aa" * 32,
        )
        first, receipt = contract.generate_matrix_bf16(np, moment, contract.CONTROL_SEEDS[0])
        second, second_receipt = contract.generate_matrix_bf16(np, moment, contract.CONTROL_SEEDS[0])
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(receipt, second_receipt)
        self.assertLessEqual(receipt["mean_error_over_source_rms"], contract.MEAN_RMS_TOLERANCE)
        self.assertLessEqual(receipt["centered_rms_relative_error"], contract.CENTERED_RMS_RELATIVE_TOLERANCE)
        self.assertEqual(hashlib.sha256(first.tobytes()).hexdigest(), receipt["control_bf16_sha256"])


if __name__ == "__main__":
    unittest.main()
