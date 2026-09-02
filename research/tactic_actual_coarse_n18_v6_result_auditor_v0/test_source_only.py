#!/usr/bin/env python3
"""Hostile standard-library tests for the N18-v6 result auditor."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest


ROOT = Path(__file__).resolve().parent


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


numeric_before = {name for name in sys.modules if name == "numpy" or name.startswith("cupy")}
core = load("tacn18_v6_audit_core_test", ROOT / "audit_core.py")
audit = load("tacn18_v6_result_auditor_test", ROOT / "result_auditor.py")
numeric_after = {name for name in sys.modules if name == "numpy" or name.startswith("cupy")}

H = "11" * 32


def valid_pins() -> dict:
    return {
        "schema": audit.PINS_SCHEMA,
        "status": "EXTERNALLY_RECORDED_AFTER_TERMINAL_PUBLICATION",
        "paths": {
            "v6_package": "/repo/research/tactic_actual_coarse_n18_v6",
            "v4_package": "/repo/research/tactic_actual_coarse_n18_v4",
            "frozen_decoder_core": "/repo/strata_v2_klt_mixed_independent_auditor_v1.py",
            "polaris_encoder_source": "/repo/src/polaris_sc_v2_rht_encoder.py",
            "smoke_receipt": "/evidence/smoke.json",
            "input_manifest": "/input/manifest.json",
            "publication_directory": "/output/result",
        },
        "hashes": {
            "v6_source_manifest_sha256": audit.KNOWN_V6_MANIFEST_SHA256,
            "v6_source_root_sha256": audit.KNOWN_V6_SOURCE_ROOT_SHA256,
            "predecessor_lock_sha256": audit.KNOWN_PREDECESSOR_LOCK_SHA256,
            "runtime_lock_sha256": audit.KNOWN_RUNTIME_LOCK_SHA256,
            "v4_source_root_sha256": audit.KNOWN_V4_SOURCE_ROOT_SHA256,
            "frozen_decoder_core_sha256": audit.KNOWN_FROZEN_DECODER_SHA256,
            "polaris_encoder_source_sha256": audit.KNOWN_POLARIS_ENCODER_SHA256,
            "smoke_receipt_sha256": audit.KNOWN_SMOKE_RECEIPT_FILE_SHA256,
            "input_manifest_sha256": H,
        },
        "input_roles": {
            role: {
                "absolute_path": f"/input/{role}.bf16",
                "bytes": 3_145_728,
                "sha256": H,
            }
            for role in core.ROLES
        },
        "publication_members": {
            name: {"bytes": 17, "sha256": H}
            for name in audit.PUBLICATION_MEMBERS
        },
    }


class ImportBoundaryTests(unittest.TestCase):
    def test_import_does_not_load_numpy_or_cupy(self) -> None:
        self.assertEqual(numeric_before, numeric_after)

    def test_unresolved_template_is_rejected(self) -> None:
        record = valid_pins()
        record["status"] = "UNRESOLVED_TEMPLATE_NOT_AUDIT_AUTHORITY"
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(record)


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_key_rejected(self) -> None:
        with self.assertRaises(core.AuditError):
            core.strict_json(b'{"a":1,"a":2}', "duplicate")

    def test_nonfinite_rejected(self) -> None:
        for value in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e9999}'):
            with self.assertRaises(core.AuditError):
                core.strict_json(value, "nonfinite")

    def test_pretty_and_canonical_are_deterministic(self) -> None:
        value = {"z": False, "a": [1, 2]}
        self.assertNotEqual(core.pretty_json(value), core.canonical_json(value))
        self.assertEqual(core.strict_json(core.pretty_json(value), "pretty"), value)


class PinTests(unittest.TestCase):
    def test_complete_pins_parse(self) -> None:
        parsed = audit.parse_pins(valid_pins())
        self.assertEqual(set(parsed["publication_members"]), audit.PUBLICATION_MEMBERS)
        self.assertEqual(set(parsed["input_roles"]), set(core.ROLES))

    def test_missing_publication_member_rejected(self) -> None:
        value = valid_pins()
        value["publication_members"].pop("COARSE.bin")
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(value)

    def test_down_alias_rejected(self) -> None:
        value = valid_pins()
        value["input_roles"]["down"] = value["input_roles"].pop("down_transposed")
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(value)

    def test_frozen_source_hash_weakening_rejected(self) -> None:
        value = valid_pins()
        value["hashes"]["runtime_lock_sha256"] = H
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(value)

    def test_relative_path_rejected(self) -> None:
        value = valid_pins()
        value["paths"]["publication_directory"] = "output/result"
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(value)


class PacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = core.Geometry(512, 512)
        self.packet = core.make_zero_packet(self.geometry, 0, 0)

    def test_zero_packet_roundtrip(self) -> None:
        parsed = core.parse_reservoir(self.packet)
        self.assertTrue(parsed.zero_tile)
        self.assertEqual(core.canonical_packet(parsed), self.packet)
        self.assertEqual(len(self.packet), core.RESERVOIR_BYTES)

    def test_header_crc_tamper_rejected(self) -> None:
        hostile = bytearray(self.packet)
        hostile[20] ^= 1
        with self.assertRaises(core.AuditError):
            core.parse_reservoir(bytes(hostile))

    def test_reservoir_fill_tamper_rejected(self) -> None:
        hostile = bytearray(self.packet)
        hostile[-1] = 1
        with self.assertRaises(core.AuditError):
            core.parse_reservoir(bytes(hostile))

    def test_role_seed_laundering_rejected(self) -> None:
        hostile = bytearray(self.packet)
        hostile[17] = 1
        struct.pack_into("<I", hostile, 124, 0)
        struct.pack_into("<I", hostile, 124, __import__("zlib").crc32(hostile[:124]) & 0xFFFFFFFF)
        with self.assertRaises(core.AuditError):
            core.parse_reservoir(bytes(hostile))

    def test_i32_contract_exceeds_i16_without_overflow(self) -> None:
        maximum = core.i32_inverse_contract_max(63)
        self.assertEqual(maximum, 8_388_608)
        self.assertGreater(maximum, 32_767)
        self.assertLess(maximum, 2**31)


class RateAndTrafficTests(unittest.TestCase):
    def test_exact_qwen_rate(self) -> None:
        rate = core.exact_rate(1_414_656, 4_718_592)
        self.assertEqual(rate["exact"], "307/128")
        self.assertTrue(rate["equals_307_over_128"])

    def test_tail_rate_is_above_target(self) -> None:
        geometry = core.Geometry(769, 2048)
        rate = core.exact_rate(geometry.frame_bytes, geometry.values)
        self.assertFalse(rate["equals_307_over_128"])
        self.assertGreater(rate["float"], 307 / 128)

    def test_audit_file_read_is_one_pass_and_hbm_unmeasured(self) -> None:
        ledgers = core.traffic_ledgers(core.Geometry(768, 2048))
        external = ledgers["external_compressed_file_read_executed_by_auditor"]
        self.assertEqual(external["passes"], 1)
        self.assertEqual(external["reread_bytes"], 0)
        self.assertFalse(ledgers["accelerator_hbm"]["measured"])
        self.assertFalse(ledgers["strict_below_2x_inference_HBM_authority"])

    def test_second_file_pass_cannot_be_laundered_as_one(self) -> None:
        ledgers = core.traffic_ledgers(core.Geometry(768, 2048))
        external = ledgers["external_compressed_file_read_executed_by_auditor"]
        hostile = dict(external)
        hostile["total_read_bytes"] *= 2
        self.assertNotEqual(hostile["total_read_bytes"], hostile["first_pass_bytes"] + hostile["reread_bytes"])


class CompletionTests(unittest.TestCase):
    def make(self):
        members = {
            name: {"bytes": len(name), "sha256": core.sha256(name.encode("ascii"))}
            for name in audit.PUBLICATION_DATA_MEMBERS
        }
        rows = [{"name": name, **members[name]} for name in sorted(members, key=lambda item: item.encode("ascii"))]
        value = {
            "schema": "tactic-actual-coarse-n18-v6-completion-v1",
            "status": "PASS_V6_BOUND_TARGET_ELIGIBLE_FRAME_NONPROMOTING_INDEPENDENT_RESULT_AUDIT_REQUIRED",
            "positive_claim_authority": False,
            "source_root_sha256": audit.KNOWN_V6_SOURCE_ROOT_SHA256,
            "source_free_smoke_file_sha256": audit.KNOWN_SMOKE_RECEIPT_FILE_SHA256,
            "frame_sha256": H,
            "members": rows,
            "members_root_sha256": core.sha256(core.canonical_json(rows)),
        }
        value["completion_claim_sha256"] = core.sha256(core.canonical_json(value))
        return value, members

    def test_valid_completion(self) -> None:
        value, members = self.make()
        core.verify_completion(
            value, members, value["status"],
            source_root_sha256=audit.KNOWN_V6_SOURCE_ROOT_SHA256,
            smoke_sha256=audit.KNOWN_SMOKE_RECEIPT_FILE_SHA256,
            frame_sha256=H,
        )

    def test_reordered_completion_rejected_even_when_resealed(self) -> None:
        value, members = self.make()
        value["members"].reverse()
        value["members_root_sha256"] = core.sha256(core.canonical_json(value["members"]))
        value.pop("completion_claim_sha256")
        value["completion_claim_sha256"] = core.sha256(core.canonical_json(value))
        with self.assertRaises(core.AuditError):
            core.verify_completion(
                value, members, value["status"],
                source_root_sha256=audit.KNOWN_V6_SOURCE_ROOT_SHA256,
                smoke_sha256=audit.KNOWN_SMOKE_RECEIPT_FILE_SHA256,
                frame_sha256=H,
            )

    def test_positive_authority_laundering_rejected(self) -> None:
        value, members = self.make()
        value["positive_claim_authority"] = True
        value.pop("completion_claim_sha256")
        value["completion_claim_sha256"] = core.sha256(core.canonical_json(value))
        with self.assertRaises(core.AuditError):
            core.verify_completion(
                value, members, value["status"],
                source_root_sha256=audit.KNOWN_V6_SOURCE_ROOT_SHA256,
                smoke_sha256=audit.KNOWN_SMOKE_RECEIPT_FILE_SHA256,
                frame_sha256=H,
            )


class QwenGateTests(unittest.TestCase):
    def test_exact_shape_gate(self) -> None:
        record = {
            "geometry": {
                "intermediate": 768, "hidden": 2048, "weights": 4_718_592,
                "role_values": 1_572_864, "records": 18, "streams_per_role": 6,
            },
            "frame_bytes": 1_414_656,
            "rate": {"exact": "307/128", "equals_307_over_128": True},
        }
        self.assertIn("NONPROMOTING", core.qwen_geometry_gate(record))

    def test_shape_only_never_proves_model_identity(self) -> None:
        value = json.loads((ROOT / "design_lock.json").read_text(encoding="utf-8"))
        self.assertFalse(value["claim_boundary"]["qwen_checkpoint_provenance_from_shape"])
        self.assertFalse(value["claim_boundary"]["strict_below_2x_inference_hbm_authority"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
