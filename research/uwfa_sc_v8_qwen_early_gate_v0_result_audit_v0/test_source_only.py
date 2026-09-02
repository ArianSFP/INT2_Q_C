#!/usr/bin/env python3
"""Synthetic hostile tests; no Qwen, control, or producer-result payloads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VERIFY_PATH = Path(__file__).with_name("verify_result.py")
SPEC = importlib.util.spec_from_file_location("uwfa_qwen_early_gate_result_audit_under_test", VERIFY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load verifier source")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def sealed(record: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(record)
    result[field] = audit.sha256(audit.canonical_json(result))
    return result


class FakeHeld:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.fd = 7


def claim_result(status: str) -> dict[str, Any]:
    wrapper = audit.expected_wrapper_status(status)
    result = {key: None for key in audit.RESULT_FIELDS}
    result.update({
        "schema": audit.RESULT_SCHEMA,
        "status": wrapper,
        "underlying_exact_v8_source_status": status,
        "positive_claim_authority": False,
        "controls_run": False,
        "controls_may_not_be_inferred_or_added": True,
        "claim_boundary": "early-kill Qwen diagnostic using exact sealed v8 source; never a positive compression claim",
        "binding_authority_disclosure": {
            "status": "EARLY_DIAGNOSTIC_LOCAL_BINDINGS_NOT_PRODUCTION_DISPATCHER_AUTHORITY",
            "externally_pinned_dispatcher_receipt_present": False,
            "baseline_score_receipt": "constructed locally from the fixed audited D/SSE/energy and recomputed artifact identities",
            "decoder_bundle_sha256": "canonical aggregate of the exact hash-pinned decoder source members",
            "audit_bootstrap_sha256": "self-reported hash of this unsealed exploratory runner",
            "pipeline_sha256": "canonical aggregate constructed by this exploratory runner",
            "positive_claim_use_permitted": False,
        },
    })
    return result


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_key_rejected(self) -> None:
        with self.assertRaises(audit.ResultAuditError):
            audit.strict_json(b'{"a":1,"a":2}', "synthetic")

    def test_nonfinite_rejected(self) -> None:
        with self.assertRaises(audit.ResultAuditError):
            audit.strict_json(b'{"a":NaN}', "synthetic")

    def test_pretty_encoding_is_distinguished(self) -> None:
        value = audit.strict_json(b'{"a":1}', "synthetic")
        self.assertNotEqual(b'{"a":1}', audit.pretty_json(value))


class CompletionTests(unittest.TestCase):
    def make_completion(self) -> tuple[dict[str, Any], set[str], dict[str, FakeHeld], str]:
        source_root = "ab" * 32
        data = {
            "BOUND_BASELINE_SCORE.json": b"score",
            "DECODER_BUNDLE.json": b"decoder",
            "RESULT.json": b"result",
            "SOURCE_PREFLIGHT.json": b"preflight",
        }
        observed = {name: FakeHeld(payload) for name, payload in data.items()}
        rows = [
            {"name": name, "bytes": len(observed[name].data), "sha256": observed[name].sha256}
            for name in sorted(data, key=lambda value: value.encode("utf-8"))
        ]
        complete = sealed({
            "schema": audit.COMPLETION_SCHEMA,
            "status": "EARLY_DIAGNOSTIC_HARD_KILL_PHYSICAL_RATE_OR_F",
            "positive_claim_authority": False,
            "controls_run": False,
            "source_snapshot_root_sha256": source_root,
            "members": rows,
        }, "completion_sha256")
        actual = set(data) | {"COMPLETE.json"}
        return complete, actual, observed, source_root

    def test_valid_synthetic_completion(self) -> None:
        complete, actual, observed, root = self.make_completion()
        audit.verify_completion_record(complete, actual, observed, root)

    def test_hostile_extra_member_rejected(self) -> None:
        complete, actual, observed, root = self.make_completion()
        actual.add("ATTACKER.bin")
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_completion_record(complete, actual, observed, root)

    def test_hostile_member_digest_rejected(self) -> None:
        complete, actual, observed, root = self.make_completion()
        complete["members"][0]["sha256"] = "00" * 32
        complete = sealed({key: value for key, value in complete.items() if key != "completion_sha256"}, "completion_sha256")
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_completion_record(complete, actual, observed, root)

    def test_hostile_controls_counter_rejected(self) -> None:
        complete, actual, observed, root = self.make_completion()
        complete["controls_run"] = True
        complete = sealed({key: value for key, value in complete.items() if key != "completion_sha256"}, "completion_sha256")
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_completion_record(complete, actual, observed, root)


class ClaimBoundaryTests(unittest.TestCase):
    def completion(self, status: str) -> dict[str, Any]:
        return {"status": audit.expected_wrapper_status(status)}

    def test_hard_kill_nonpromoting_boundary(self) -> None:
        status = "HARD_KILL_PHYSICAL_RATE_OR_F"
        self.assertEqual(audit.verify_claim_boundary(claim_result(status), self.completion(status)), status)

    def test_survivor_positive_claim_flip_rejected(self) -> None:
        status = "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED"
        result = claim_result(status)
        result["positive_claim_authority"] = True
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_claim_boundary(result, self.completion(status))

    def test_survivor_controls_flip_rejected(self) -> None:
        status = "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED"
        result = claim_result(status)
        result["controls_run"] = True
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_claim_boundary(result, self.completion(status))

    def test_wrapper_status_confusion_rejected(self) -> None:
        status = "HARD_KILL_PHYSICAL_RATE_OR_F"
        result = claim_result(status)
        result["status"] = audit.expected_wrapper_status("SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED")
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_claim_boundary(result, self.completion(status))

    def test_disclosure_string_laundering_rejected(self) -> None:
        status = "HARD_KILL_PHYSICAL_RATE_OR_F"
        result = claim_result(status)
        result["binding_authority_disclosure"]["baseline_score_receipt"] = "synthetic placeholder"
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_claim_boundary(result, self.completion(status))

    def test_deep_positive_promotion_counter_rejected(self) -> None:
        with self.assertRaises(audit.ResultAuditError):
            audit.verify_nonpromotion_counters({"nested": [{"positive_promotion": True}]})

    def test_deep_false_nonpromotion_counters_pass(self) -> None:
        audit.verify_nonpromotion_counters({"positive_claim_authority": False, "nested": {"controls_opened": False}})


class DecisionTests(unittest.TestCase):
    def test_literal_physical_failure_is_hard_kill(self) -> None:
        row = audit.classify_decision("HARD_KILL_PHYSICAL_RATE_OR_F", False, True, True, True)
        self.assertEqual(row["classification"], "VERIFIED_HARD_KILL_FINAL_REGARDLESS_OF_CONTROLS")
        self.assertFalse(row["positive_claim_authority"])

    def test_survivor_requires_controls(self) -> None:
        row = audit.classify_decision("SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED", True, True, True, True)
        self.assertTrue(row["controls_required_before_any_positive_claim"])
        self.assertFalse(row["positive_claim_authority"])

    def test_status_laundering_rejected(self) -> None:
        with self.assertRaises(audit.ResultAuditError):
            audit.classify_decision("SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED", False, True, True, True)


class LiteralContainerHarnessTests(unittest.TestCase):
    class DescriptorSource:
        def __init__(self, fd: int, expected: str) -> None:
            self.fd = fd
            self.expected = expected

        def close(self) -> None:
            pass

    class Codec:
        AuthenticatedDescriptorSource = None

        def __init__(self, *, bad_rebuild: bool = False) -> None:
            self.bad_rebuild = bad_rebuild
            self.AuthenticatedDescriptorSource = LiteralContainerHarnessTests.DescriptorSource

        def parse_container(self, common: Any, semantic: Any, raw: bytes) -> dict[str, Any]:
            return {"raw": raw}

        def canonical_rebuild(self, common: Any, semantic: Any, parsed: dict[str, Any]) -> bytes:
            return parsed["raw"] + (b"x" if self.bad_rebuild else b"")

        def physical_metrics(self, common: Any, semantic: Any, parsed: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            return {"descriptor": bool(kwargs)}

    class Adapter:
        def __init__(self) -> None:
            self.decode_calls = 0

        def decode_new_container(self, parsed: dict[str, Any]) -> dict[str, Any]:
            self.decode_calls += 1
            return {
                "all_payloads_canonically_reencoded": True,
                "all_three_roles_reconstructed": True,
            }

        def new_routed_decoder(self) -> object:
            return object()

    def modules(self, codec: Any, adapter: Any) -> dict[str, Any]:
        return {"codec": codec, "common": object(), "semantic": object(), "adapter": adapter}

    def test_candidate_runs_semantic_decode_and_descriptor_metrics(self) -> None:
        adapter = self.Adapter()
        result = audit.audit_literal_container(self.modules(self.Codec(), adapter), FakeHeld(b"UWFCV8-synthetic"), semantic_decode=True, label="synthetic")
        self.assertEqual(adapter.decode_calls, 1)
        self.assertTrue(result["metrics"]["descriptor"])

    def test_identity_counterfactual_does_not_invent_decode(self) -> None:
        adapter = self.Adapter()
        result = audit.audit_literal_container(self.modules(self.Codec(), adapter), FakeHeld(b"identity-synthetic"), semantic_decode=False, label="synthetic identity")
        self.assertEqual(adapter.decode_calls, 0)
        self.assertFalse(result["metrics"]["descriptor"])

    def test_noncanonical_rebuild_rejected(self) -> None:
        with self.assertRaises(audit.ResultAuditError):
            audit.audit_literal_container(self.modules(self.Codec(bad_rebuild=True), self.Adapter()), FakeHeld(b"tamper"), semantic_decode=False, label="tamper")


@unittest.skipUnless(os.name == "posix", "retained openat test requires POSIX")
class DescriptorHostilityTests(unittest.TestCase):
    def test_final_name_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uwfa-audit-synthetic-") as temporary:
            parent = audit.RetainedDirectory(temporary, "synthetic parent")
            target = Path(temporary) / "member.bin"
            target.write_bytes(b"original")
            held = audit.HeldRegularAt(parent.fd, "member.bin", 1024, "synthetic member")
            try:
                os.rename(target, Path(temporary) / "original-aside.bin")
                target.write_bytes(b"replacement")
                with self.assertRaises(audit.ResultAuditError):
                    held.verify_final()
            finally:
                held.close(verify=False)
                parent.close(verify=False)


class PinTemplateTests(unittest.TestCase):
    def test_pin_template_contains_no_resolved_value(self) -> None:
        path = Path(__file__).with_name("UNRESOLVED_EXTERNAL_PINS.json")
        pins = json.loads(path.read_text(encoding="ascii"))
        self.assertTrue(pins)
        self.assertTrue(all(value is None for value in pins.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
