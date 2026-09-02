#!/usr/bin/env python3
"""Hostile synthetic tests for the posterior-centroid v0 result auditor."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("uwfa_pc_result_audit_test_core", "audit_core.py")
runner = load("uwfa_pc_result_audit_test_runner", "result_auditor.py")


def assert_reject(test: unittest.TestCase, callback) -> None:
    with test.assertRaises((core.AuditError, runner.ResultAuditError, ValueError, KeyError, TypeError)):
        callback()


def valid_pins() -> dict:
    digest = "1" * 64
    v9 = {name: {"bytes": 1, "sha256": digest} for name in runner.V9_MEMBERS}
    posterior = {name: {"bytes": 1, "sha256": digest} for name in runner.POSTERIOR_BASE_MEMBERS}
    return {
        "schema": runner.PINS_SCHEMA,
        "status": "RESOLVED_IMMUTABLE_EXTERNAL_AUTHORITY",
        "paths": {
            "posterior_producer_package": "/absolute/posterior-producer",
            "v9_publication": "/absolute/v9",
            "v9_result_audit_receipt": "/absolute/v9-audit.json",
            "v8_package": "/absolute/v8",
            "strata_common": "/absolute/common.py",
            "frozen_auditor": "/absolute/frozen.py",
            "source_manifest": "/absolute/source.json",
            "posterior_publication": "/absolute/posterior",
            "audit_output_parent": "/absolute/output",
            "audit_output_name": "audit-v0",
        },
        "hashes": {
            "posterior_producer_manifest_sha256": runner.KNOWN_POSTERIOR_MANIFEST_SHA256,
            "v9_result_audit_source_manifest_sha256": runner.KNOWN_V9_RESULT_AUDIT_MANIFEST_SHA256,
            "v9_result_audit_receipt_sha256": digest,
            "v8_manifest_sha256": runner.KNOWN_V8_MANIFEST_SHA256,
            "strata_common_sha256": runner.KNOWN_STRATA_SHA256,
            "frozen_auditor_sha256": runner.KNOWN_FROZEN_SHA256,
            "source_manifest_sha256": digest,
        },
        "v9_publication_members": v9,
        "posterior_publication_members": posterior,
        "access_authorization": {
            "may_open_completed_v9_publication": True,
            "may_open_completed_posterior_publication": True,
            "may_open_bf16_score_panel": True,
            "may_initialize_cupy_for_exact_rht_replay": True,
            "positive_claim_authority": False,
        },
    }


class SourceOnlyTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        assert_reject(self, lambda: runner.strict_json(b'{"x":1,"x":2}', "duplicate"))
        assert_reject(self, lambda: runner.strict_json(b'{"x":NaN}', "nonfinite"))

    def test_external_pins_require_frozen_closures(self) -> None:
        row = valid_pins()
        parsed = runner.parse_pins(row)
        self.assertEqual(set(parsed["posterior_rows"]), set(runner.POSTERIOR_BASE_MEMBERS))
        damaged = json.loads(json.dumps(row))
        damaged["hashes"]["v8_manifest_sha256"] = "2" * 64
        assert_reject(self, lambda: runner.parse_pins(damaged))

    def test_source_root_is_domain_separated_and_ascii_ordered(self) -> None:
        rows = [
            {"name": "audit.py", "bytes": 1, "sha256": "a" * 64},
            {"name": "BLOCK.json", "bytes": 2, "sha256": "b" * 64},
        ]
        observed = runner.auditor_source_root(rows)
        expected_rows = [rows[1], rows[0]]
        expected = runner.sha256(runner.AUDITOR_SOURCE_ROOT_DOMAIN + runner.canonical_json(expected_rows))
        self.assertEqual(observed, expected)
        self.assertNotEqual(observed, runner.sha256(runner.canonical_json(expected_rows)))

    def test_external_pins_cannot_grant_positive_authority(self) -> None:
        row = valid_pins()
        row["access_authorization"]["positive_claim_authority"] = True
        assert_reject(self, lambda: runner.parse_pins(row))

    def test_external_member_set_is_exact(self) -> None:
        row = valid_pins()
        row["posterior_publication_members"]["UNAUTHENTICATED.bin"] = {"bytes": 1, "sha256": "1" * 64}
        assert_reject(self, lambda: runner.parse_pins(row))

    def test_final_member_is_only_optional_member(self) -> None:
        row = valid_pins()
        row["posterior_publication_members"][runner.POSTERIOR_FINAL_MEMBER] = {"bytes": 1, "sha256": "1" * 64}
        parsed = runner.parse_pins(row)
        self.assertIn(runner.POSTERIOR_FINAL_MEMBER, parsed["posterior_rows"])

    def test_head_and_wrapper_independent_roundtrip(self) -> None:
        handoff = "2" * 64
        parameters = np.linspace(-0.125, 0.125, core.parameter_count(core.LAW_STATE, 4), dtype=np.float64)
        head = core.serialize_head(np, parameters, law=core.LAW_STATE, states=4, ridge_exponent=-12, handoff_root_sha256=handoff)
        parsed_head = core.parse_head(np, head, expected_handoff_root_sha256=handoff)
        self.assertEqual(parsed_head["law_name"], "state-aware")
        inner = b"I" * core.PAGE_BYTES
        wrapper = core.build_wrapper(inner, head, weights=100_000, experts=3, fold_ordinal=1, handoff_root_sha256=handoff)
        parsed = core.parse_wrapper(np, wrapper, expected_handoff_root_sha256=handoff)
        self.assertEqual(parsed["inner"], inner)
        damaged = bytearray(wrapper)
        damaged[-core.WRAPPER_FOOTER_BYTES - 1] ^= 1
        assert_reject(self, lambda: core.parse_wrapper(np, bytes(damaged), expected_handoff_root_sha256=handoff))

    def test_head_binding_tamper_rejected(self) -> None:
        handoff = "3" * 64
        head = core.serialize_head(np, np.zeros(2), law=core.LAW_LOCAL, states=4, ridge_exponent=0, handoff_root_sha256=handoff)
        assert_reject(self, lambda: core.parse_head(np, head, expected_handoff_root_sha256="4" * 64))

    def test_state_permutation_is_nontrivial_and_deterministic(self) -> None:
        first = core.state_permutation(9, 2, 32)
        self.assertEqual(first, core.state_permutation(9, 2, 32))
        self.assertEqual(sorted(first), list(range(32)))
        self.assertNotEqual(first, tuple(range(32)))

    def test_ridge_fit_and_binary16_packet_are_deterministic(self) -> None:
        occupancy = np.zeros((core.LEVELS, 4), dtype=np.float64)
        occupancy[:, 0] = 0.75
        occupancy[:, 1:] = -0.25
        indices = np.tile(np.arange(8, dtype=np.int16), 4)
        q = 0.25 * (indices.astype(np.float64) - 31.0)
        block = core.Observation(0, (0,), indices, q + 0.01 + 0.02 * q, occupancy, "5" * 64)
        parameters = core.fit_head(np, (block,), law=core.LAW_LOCAL, states=4, ridge_exponent=-28)
        self.assertTrue(np.all(np.isfinite(parameters)))
        head_a = core.serialize_head(np, parameters, law=core.LAW_LOCAL, states=4, ridge_exponent=-28, handoff_root_sha256="6" * 64)
        head_b = core.serialize_head(np, parameters.copy(), law=core.LAW_LOCAL, states=4, ridge_exponent=-28, handoff_root_sha256="6" * 64)
        self.assertEqual(head_a, head_b)

    def test_read_projection_explicitly_is_not_routed_posterior(self) -> None:
        handoff = "7" * 64
        head = core.serialize_head(np, np.zeros(2), law=core.LAW_LOCAL, states=2, ridge_exponent=0, handoff_root_sha256=handoff)
        parsed = core.parse_wrapper(np, core.build_wrapper(b"A" * core.PAGE_BYTES, head, weights=8192, experts=2, fold_ordinal=0, handoff_root_sha256=handoff), expected_handoff_root_sha256=handoff)
        fraction = core.fraction_record(Fraction(4096, 1))
        nonpadding = core.fraction_record(Fraction(4000, 1))
        metrics = {
            "actual_container_bytes": core.PAGE_BYTES,
            "experts": [
                {
                    "expert_ordinal": expert,
                    "instrumented_routed_read_ranges": [[0, 512]],
                    "instrumented_routed_read_request_count": 1,
                    "attributable_total_physical_bytes": fraction,
                    "attributable_nonpadding_decodable_bytes": nonpadding,
                    "causal_decode_reencode_reconstruction": {
                        "all_payloads_canonically_reencoded": True,
                        "all_three_roles_reconstructed": True,
                    },
                }
                for expert in range(2)
            ],
        }
        ledger = core.wrapper_ledger(inner_metrics=metrics, wrapper=parsed, weights_by_expert=(4096, 4096))
        self.assertTrue(ledger["actual_inner_routed_decode_executed"])
        self.assertFalse(ledger["actual_posterior_wrapper_routed_decode_executed"])
        self.assertFalse(ledger["posterior_head_applied_to_routed_reconstruction"])
        self.assertTrue(ledger["read_claim_is_nonpromoting_projection_from_instrumented_inner_decode_plus_literal_suffix"])

    def test_strict_read_equality_at_two_fails(self) -> None:
        gates = core.fold_gate(delta_s_value=0.1, g_state_value=0.01, candidate_rate_bpw=2.4, candidate_f=0.7, cold_read_below_2x=False)
        self.assertFalse(gates["passes_all_fold_gates"])

    def test_source_manifest_identity_field_rejected(self) -> None:
        manifest = {
            "schema": runner.SOURCE_PANEL_SCHEMA,
            "bound_artifact_sha256": "8" * 64,
            "experts": 1,
            "source_record_set_sha256": "9" * 64,
            "model_name": "forbidden",
            "matrices": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.json"
            path.write_bytes(b"{}")
            assert_reject(self, lambda: runner.authenticate_source_manifest(core, json.dumps(manifest).encode(), path, artifact_sha256="8" * 64, experts=1, intermediate=2, hidden=2))

    def test_output_publication_rejects_completion_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {"RESULT.json": b"{}\n", "COMPLETE.json": b"{}\n"}
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
            rows = {name: {"bytes": len(payload), "sha256": runner.sha256(payload)} for name, payload in payloads.items()}
            assert_reject(self, lambda: runner.authenticate_publication(root, rows, "hostile"))

    def test_no_producer_numerical_core_import(self) -> None:
        source = (ROOT / "result_auditor.py").read_text(encoding="utf-8")
        self.assertNotIn("import posterior_core", source)
        self.assertNotIn("from posterior_core", source)
        self.assertNotIn("diagnostic.py\"],", source)

    def test_backend_import_occurs_after_input_authentication(self) -> None:
        source = (ROOT / "result_auditor.py").read_text(encoding="utf-8")
        numpy_position = source.index("    import numpy as np")
        self.assertLess(source.index("posterior = authenticate_publication"), numpy_position)
        self.assertLess(source.index("source_manifest_payload = regular_bytes"), numpy_position)
        self.assertLess(source.index("producer = authenticate_source_package"), numpy_position)

    def test_direct_execution_without_authority_is_inert(self) -> None:
        completed = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "result_auditor.py")], capture_output=True, timeout=20, check=False)
        self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceOnlyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"schema": "uwfa-sc-posterior-centroid-v0-result-auditor-source-test-v0", "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "status": "OK" if result.wasSuccessful() else "FAIL"}, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
