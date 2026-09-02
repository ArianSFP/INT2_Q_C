#!/usr/bin/env python3
"""Bounded source-only tests; synthetic values only and no payload discovery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
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


contract = load("uwfa_source_moment_contract_tests", "moment_contract.py")


def fake_authorization() -> dict:
    rows = []
    for ordinal, (slot, role) in enumerate(contract.expected_matrix_order()):
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "slot": slot,
                "role": role,
                "shape": list(contract.role_shape(role)),
                "values": contract.VALUES_PER_MATRIX,
                "bytes": contract.BYTES_PER_MATRIX,
                "source_relpath": contract.canonical_source_relpath(ordinal, slot, role),
                "source_matrix_bf16_sha256": hashlib.sha256(
                    f"synthetic-source-row:{ordinal}".encode("ascii")
                ).hexdigest(),
            }
        )
    clean = {
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "panel": contract.panel_record(),
        "source_closure": {
            name: hashlib.sha256(f"synthetic:{name}".encode("ascii")).hexdigest()
            for name in contract.SOURCE_CLOSURE_FIELDS
        },
        "matrices": rows,
        "source_set_sha256": contract.source_set_sha256(rows),
    }
    return contract.sealed(clean, "authorization_sha256")


def fake_moment_contract() -> dict:
    authorization = fake_authorization()
    rows = []
    for ordinal, source in enumerate(authorization["matrices"]):
        mean = (ordinal - 8.5) * 1.0e-6
        centered = contract.VALUES_PER_MATRIX * 0.0004 * (1.0 + ordinal / 100.0)
        energy = centered + contract.VALUES_PER_MATRIX * mean * mean
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "slot": source["slot"],
                "role": source["role"],
                "shape": source["shape"],
                "values": source["values"],
                "mean_f64_hex": contract.f64_hex(mean),
                "centered_sse_f64_hex": contract.f64_hex(centered),
                "energy_f64_hex": contract.f64_hex(energy),
                "source_matrix_bf16_sha256": source["source_matrix_bf16_sha256"],
            }
        )
    return contract.sealed(
        {
            "schema": contract.MOMENT_CONTRACT_SCHEMA,
            "status": contract.MOMENT_CONTRACT_STATUS,
            "moment_semantics": contract.MOMENT_SEMANTICS,
            "panel": contract.panel_record(),
            "source_closure": authorization["source_closure"],
            "matrices": rows,
        },
        "moment_contract_sha256",
    )


class UniversalContractTests(unittest.TestCase):
    def test_exact_eighteen_matrix_order_and_geometry(self) -> None:
        authorization = contract.validate_authorization_record(fake_authorization())
        self.assertEqual(len(authorization["matrices"]), 18)
        self.assertEqual(sum(row["values"] for row in authorization["matrices"]), 28_311_552)
        self.assertEqual(sum(row["bytes"] for row in authorization["matrices"]), 56_623_104)
        for ordinal, row in enumerate(authorization["matrices"]):
            self.assertEqual(row["matrix_ordinal"], ordinal)
            self.assertEqual(row["slot"], ordinal // 3)
            self.assertEqual(row["role"], contract.ROLES[ordinal % 3])
            expected = [2048, 768] if row["role"] == "down" else [768, 2048]
            self.assertEqual(row["shape"], expected)

    def test_external_file_digest_is_separate_from_self_seal(self) -> None:
        payload = contract.pretty_json(fake_authorization())
        observed = hashlib.sha256(payload).hexdigest()
        self.assertEqual(
            contract.parse_external_authorization(payload, observed),
            fake_authorization(),
        )
        with self.assertRaisesRegex(ValueError, "out-of-band authorization"):
            contract.parse_external_authorization(payload, "00" * 32)

    def test_runtime_records_carry_no_private_model_identity(self) -> None:
        authorization = contract.pretty_json(fake_authorization()).lower()
        moment_record = contract.pretty_json(fake_moment_contract()).lower()
        generator_key = contract.canonical_json(
            contract.validate_moment_contract_record(fake_moment_contract())[1][0].public_generator_key()
        ).lower()
        for payload in (authorization, moment_record, generator_key):
            for forbidden in (
                b"qwen",
                b"model.layers",
                b"checkpoint",
                b"tensor_name",
                b"original_expert",
            ):
                self.assertNotIn(forbidden, payload)

    def test_moment_contract_matches_consumer_abi(self) -> None:
        record, moments = contract.validate_moment_contract_record(fake_moment_contract())
        self.assertEqual(record["schema"], "uwfa-sc-v9-bf16-matrix-moment-contract-v1")
        self.assertEqual(len(moments), 18)
        self.assertEqual(moments[0].shape, (768, 2048))
        self.assertEqual(moments[2].shape, (2048, 768))

    def test_template_is_inert_and_complete(self) -> None:
        template = json.loads((PACKAGE / "AUTHORIZATION_TEMPLATE.json").read_text(encoding="ascii"))
        self.assertEqual(template["status"], "TEMPLATE_ONLY_NOT_AUTHORIZATION")
        self.assertEqual(len(template["matrices"]), 18)
        self.assertNotIn("authorization_sha256", template)
        self.assertTrue(
            all(row["source_matrix_bf16_sha256"].startswith("__") for row in template["matrices"])
        )

    def test_runtime_pins_are_canonical_and_exact(self) -> None:
        pins, payload = contract.load_runtime_pins(PACKAGE)
        self.assertEqual(payload, contract.pretty_json(pins))
        self.assertEqual(pins["moment_runtime"]["python_version"], "3.12.3")
        self.assertEqual(pins["moment_runtime"]["numpy"]["version"], "2.5.2")
        self.assertEqual(
            pins["consumer_source_pins"]["source_manifest"]["sha256"],
            "20cd2cd8b2a0e41f68e5fcf58a1b2ebe8d0e09c984bdbbd786a1057e869c9eb1",
        )

    def test_direct_entrypoints_are_inert(self) -> None:
        publisher = load("uwfa_source_moment_publisher_tests", "source_moment_publisher.py")
        self.assertEqual(contract.direct_main(), 3)
        self.assertEqual(publisher.direct_main(), 3)


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy unavailable")
class NumericalReferenceTests(unittest.TestCase):
    def test_exact_bf16_decode_and_two_pass_moments(self) -> None:
        import numpy as np

        values = np.asarray([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
        words = contract.fp32_to_bf16_rne(np, values)
        mean, centered_sse, energy = contract.measured_moments(np, words)
        self.assertEqual(mean, 0.0)
        self.assertEqual(centered_sse, 10.0)
        self.assertEqual(energy, 10.0)

    def test_binary64_serialization_roundtrip(self) -> None:
        value = -math.pi / 17.0
        self.assertEqual(contract.from_f64_hex(contract.f64_hex(value), "value"), value)

    def test_small_gaussian_reference_is_deterministic(self) -> None:
        import numpy as np

        values = 262_144
        mean = -0.00125
        centered_sse = values * 0.021 ** 2
        energy = centered_sse + values * mean * mean
        moment = contract.MatrixMoment(
            ordinal=0,
            slot=0,
            role="gate",
            shape=(512, 512),
            values=values,
            mean=mean,
            centered_sse=centered_sse,
            energy=energy,
            source_matrix_bf16_sha256="aa" * 32,
        )
        first, first_receipt = contract.regenerate_gaussian_bf16(
            np, moment, contract.CONTROL_SEEDS[0]
        )
        second, second_receipt = contract.regenerate_gaussian_bf16(
            np, moment, contract.CONTROL_SEEDS[0]
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_receipt, second_receipt)
        self.assertLessEqual(
            first_receipt["mean_error_over_source_rms"], contract.MEAN_RMS_TOLERANCE
        )
        self.assertLessEqual(
            first_receipt["centered_rms_relative_error"],
            contract.CENTERED_RMS_RELATIVE_TOLERANCE,
        )
        self.assertEqual(
            hashlib.sha256(first.tobytes(order="C")).hexdigest(),
            first_receipt["control_bf16_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
