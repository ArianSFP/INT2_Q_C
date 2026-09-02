#!/usr/bin/env python3
"""Hostile synthetic tests for authorization and fail-closed publication."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).resolve().parent


def load_contract():
    name = "uwfa_source_moment_contract_hostile_tests"
    spec = importlib.util.spec_from_file_location(name, PACKAGE / "moment_contract.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("moment_contract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


contract = load_contract()


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
                    f"hostile-fixture:{ordinal}".encode("ascii")
                ).hexdigest(),
            }
        )
    clean = {
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "panel": contract.panel_record(),
        "source_closure": {
            name: hashlib.sha256(name.encode("ascii")).hexdigest()
            for name in contract.SOURCE_CLOSURE_FIELDS
        },
        "matrices": rows,
        "source_set_sha256": contract.source_set_sha256(rows),
    }
    return contract.sealed(clean, "authorization_sha256")


def reseal_authorization(value: dict, *, refresh_source_set: bool = True) -> dict:
    value = copy.deepcopy(value)
    value.pop("authorization_sha256", None)
    if refresh_source_set:
        value["source_set_sha256"] = contract.source_set_sha256(value["matrices"])
    return contract.sealed(value, "authorization_sha256")


def fake_moment_contract() -> dict:
    auth = fake_authorization()
    rows = []
    for ordinal, row in enumerate(auth["matrices"]):
        mean = 1.0e-6 * ordinal
        centered = contract.VALUES_PER_MATRIX * (0.02 + 0.0001 * ordinal) ** 2
        energy = centered + contract.VALUES_PER_MATRIX * mean * mean
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "slot": row["slot"],
                "role": row["role"],
                "shape": row["shape"],
                "values": row["values"],
                "mean_f64_hex": contract.f64_hex(mean),
                "centered_sse_f64_hex": contract.f64_hex(centered),
                "energy_f64_hex": contract.f64_hex(energy),
                "source_matrix_bf16_sha256": row["source_matrix_bf16_sha256"],
            }
        )
    return contract.sealed(
        {
            "schema": contract.MOMENT_CONTRACT_SCHEMA,
            "status": contract.MOMENT_CONTRACT_STATUS,
            "moment_semantics": contract.MOMENT_SEMANTICS,
            "panel": contract.panel_record(),
            "source_closure": auth["source_closure"],
            "matrices": rows,
        },
        "moment_contract_sha256",
    )


class AuthorizationHostileTests(unittest.TestCase):
    def assert_auth_rejected(self, value: dict, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            contract.validate_authorization_record(value)

    def test_swapped_order_rejected_even_when_attacker_reseals(self) -> None:
        value = fake_authorization()
        value["matrices"][0], value["matrices"][1] = value["matrices"][1], value["matrices"][0]
        self.assert_auth_rejected(reseal_authorization(value), "source ordinal")

    def test_role_substitution_rejected_even_when_attacker_reseals(self) -> None:
        value = fake_authorization()
        value["matrices"][0]["role"] = "up"
        self.assert_auth_rejected(reseal_authorization(value), "slot/role")

    def test_down_storage_transpose_rejected(self) -> None:
        value = fake_authorization()
        value["matrices"][2]["shape"] = [768, 2048]
        self.assert_auth_rejected(reseal_authorization(value), "source shape")

    def test_boolean_ordinal_rejected(self) -> None:
        value = fake_authorization()
        value["matrices"][0]["matrix_ordinal"] = False
        self.assert_auth_rejected(reseal_authorization(value), "ordinal type")

    def test_path_traversal_rejected(self) -> None:
        value = fake_authorization()
        value["matrices"][0]["source_relpath"] = "../matrix_00_slot_00_gate.bf16"
        self.assert_auth_rejected(reseal_authorization(value), "canonical source path")

    def test_identity_field_injection_rejected(self) -> None:
        value = fake_authorization()
        value["matrices"][0]["tensor_name"] = "private"
        self.assert_auth_rejected(reseal_authorization(value), "source row fields")

    def test_uppercase_digest_rejected(self) -> None:
        value = fake_authorization()
        value["matrices"][0]["source_matrix_bf16_sha256"] = value["matrices"][0][
            "source_matrix_bf16_sha256"
        ].upper()
        self.assert_auth_rejected(reseal_authorization(value), "lowercase SHA-256")

    def test_source_set_root_tamper_rejected(self) -> None:
        value = fake_authorization()
        value["source_set_sha256"] = "11" * 32
        self.assert_auth_rejected(
            reseal_authorization(value, refresh_source_set=False), "source set closure"
        )

    def test_noncanonical_file_serialization_rejected(self) -> None:
        value = fake_authorization()
        payload = json.dumps(value).encode("ascii")
        with self.assertRaisesRegex(ValueError, "canonical pretty"):
            contract.parse_external_authorization(payload, hashlib.sha256(payload).hexdigest())

    def test_oversized_authorization_rejected_before_json_parse(self) -> None:
        payload = b" " * (contract.MAX_AUTHORIZATION_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "authorization file size bound"):
            contract.parse_external_authorization(payload, hashlib.sha256(payload).hexdigest())


class MomentHostileTests(unittest.TestCase):
    def reseal(self, value: dict) -> dict:
        value = copy.deepcopy(value)
        value.pop("moment_contract_sha256", None)
        return contract.sealed(value, "moment_contract_sha256")

    def test_energy_identity_tamper_rejected(self) -> None:
        value = fake_moment_contract()
        value["matrices"][0]["energy_f64_hex"] = contract.f64_hex(12345.0)
        with self.assertRaisesRegex(ValueError, "moment energy identity"):
            contract.validate_moment_contract_record(self.reseal(value))

    def test_nonfinite_moment_rejected(self) -> None:
        value = fake_moment_contract()
        value["matrices"][0]["mean_f64_hex"] = struct.pack("<d", float("nan")).hex()
        with self.assertRaisesRegex(ValueError, "mean 0 finite"):
            contract.validate_moment_contract_record(self.reseal(value))

    def test_zero_centered_sse_rejected(self) -> None:
        value = fake_moment_contract()
        value["matrices"][0]["centered_sse_f64_hex"] = contract.f64_hex(0.0)
        with self.assertRaisesRegex(ValueError, "centered SSE 0 positive"):
            contract.validate_moment_contract_record(self.reseal(value))


class FileAndPublicationHostileTests(unittest.TestCase):
    def test_regular_file_digest_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            path = root / "synthetic.dat"
            path.write_bytes(b"\x00\x01\x02\x03")
            row = {
                "source_relpath": path.name,
                "bytes": 4,
                "source_matrix_bf16_sha256": "00" * 32,
            }
            with self.assertRaisesRegex(ValueError, "source digest"):
                contract._read_bound_regular_file(root, row)

    def test_existing_publication_is_never_overwritten(self) -> None:
        records = {
            "MOMENT_CONTRACT.json": b"{}\n",
            "PUBLICATION.json": b"{}\n",
            "RUNTIME_PINS.json": b"{}\n",
            "SOURCE_AUTHORIZATION.json": b"{}\n",
        }
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "final"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "must not exist"):
                contract._atomic_publish_records(output, records)
            self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_write_failure_leaves_final_absent_and_incomplete_stage(self) -> None:
        records = {
            "MOMENT_CONTRACT.json": b"{}\n",
            "PUBLICATION.json": b"{}\n",
            "RUNTIME_PINS.json": b"{}\n",
            "SOURCE_AUTHORIZATION.json": b"{}\n",
        }
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "final"
            original = contract._write_new

            def hostile_write(path: Path, payload: bytes) -> None:
                if path.name == "PUBLICATION.json":
                    raise OSError("synthetic write failure")
                original(path, payload)

            with mock.patch.object(contract, "_write_new", side_effect=hostile_write):
                with self.assertRaisesRegex(OSError, "synthetic write failure"):
                    contract._atomic_publish_records(output, records)
            self.assertFalse(output.exists())
            stages = list(parent.glob(".final.incomplete-*"))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / "INCOMPLETE").is_file())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unavailable")
    def test_source_symlink_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            target = root / "target.dat"
            target.write_bytes(b"\x00\x01")
            link = root / "link.dat"
            try:
                link.symlink_to(target.name)
            except OSError as exc:
                self.skipTest(f"symlink privilege unavailable: {exc}")
            row = {
                "source_relpath": link.name,
                "bytes": 2,
                "source_matrix_bf16_sha256": hashlib.sha256(b"\x00\x01").hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "symlink forbidden"):
                contract._read_bound_regular_file(root, row)


if __name__ == "__main__":
    unittest.main()
