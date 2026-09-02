#!/usr/bin/env python3
"""Hostile standard-library tests; no payload, model or CUDA imports."""

from __future__ import annotations

import hashlib
import importlib.util
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


contract = load("uwfa_v9_matched_consumer_contract_test", "consumer_contract.py")
runtime = load("uwfa_v9_matched_consumer_runtime_test", "matched_controls_consumer.py")


def d(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def symmetric_closure() -> dict[str, str]:
    return {name: d(name) for name in contract.SYMMETRIC_CODEC_CLOSURE_FIELDS}


def primary_authorization() -> dict:
    clean = {
        "schema": "uwfa-sc-v9-primary-controls-authorization-v1",
        "status": "PASS_INDEPENDENT_PRIMARY_SURVIVOR_CONTROLS_AUTHORIZED",
        "source_status": contract.PRIMARY_STATUS,
        "positive_claim_authority": False,
        "controls_authorized": True,
        "source_artifact_sha256": d("source artifact"),
        "source_full_geometry_sha256": d("source full geometry"),
        "source_structural_geometry_sha256": d("source structural geometry"),
        "source_pipeline_sha256": d("source pipeline"),
        "source_score_receipt_sha256": d("source score"),
        "source_moment_auditor_sha256": d("source moment auditor"),
        "source_reconstruction_sha256": d("source reconstruction"),
        "source_absolute_saving_bpw": 0.1675415,
        "source_gates": {
            "rate_interval": True,
            "F_target": True,
            "cold_read_below_2x": True,
            "heldout": True,
            "standalone_decode": True,
            "integrity": True,
            "independent_result_audit": True,
        },
        "universal_format_geometry_sha256": contract.universal_format_geometry_sha256(),
        "symmetric_codec_closure": symmetric_closure(),
        "v9_primary_source_manifest_sha256": contract.V9_PRIMARY_SOURCE_MANIFEST_SHA256,
        "v9_primary_runner_sha256": contract.V9_PRIMARY_RUNNER_SHA256,
        "independent_result_auditor_manifest_sha256": d("primary auditor"),
        "independent_result_audit_receipt_sha256": d("primary receipt"),
    }
    return contract.sealed(clean, "authorization_sha256")


def run_authorization(
    *, primary_sha: str = d("primary authorization file"), root_sha: str = d("root complete")
) -> dict:
    runtime_closure = dict(contract.V8_RUNTIME_EXPECTED)
    clean = {
        "schema": "uwfa-sc-v9-matched-controls-run-authorization-v1",
        "status": "PASS_EXTERNAL_PINS_CONTROL_PAYLOAD_ACCESS_AUTHORIZED",
        "payload_access_authority": True,
        "positive_claim_authority": False,
        "consumer_source_manifest_sha256": d("consumer manifest"),
        "consumer_auditor_manifest_sha256": d("consumer auditor"),
        "consumer_audit_receipt_sha256": d("consumer receipt"),
        "producer_source_manifest_sha256": contract.PRODUCER_SOURCE_MANIFEST_SHA256,
        "producer_auditor_manifest_sha256": d("producer auditor"),
        "producer_audit_receipt_sha256": d("producer receipt"),
        "eight_control_root_complete_sha256": root_sha,
        "primary_authorization_sha256": primary_sha,
        "v8_runtime_closure": runtime_closure,
        "v8_all150_preflight_receipt_sha256": d("all150 preflight"),
        "independent_gpu_identity_receipt_sha256": d("gpu identity"),
        "source_snapshot_root_sha256": d("source snapshot"),
        "descriptor_source_builder_sha256": d("descriptor builder"),
        "moment_replayer_source_sha256": d("moment replayer"),
        "audit_bootstrap_sha256": d("audit bootstrap"),
        "all_eight_authenticate_before_fit": True,
        "all_150_independent_per_executed_control": True,
        "source_winner_reuse_forbidden": True,
        "member_loader_rejects_symlinks_and_path_escape": True,
        "immutable_snapshot_held_through_run": True,
    }
    return contract.sealed(clean, "run_authorization_sha256")


class GeometryTests(unittest.TestCase):
    def test_universal_geometry_is_source_independent(self) -> None:
        geometry = contract.validate_universal_geometry(
            contract.universal_format_geometry()
        )
        encoded = contract.canonical_json(geometry).lower()
        for forbidden in (
            b"profile", b"symbol", b"label", b"payload", b"qwen", b"model.layers"
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(len(geometry["blocks"]), 15)
        self.assertEqual(geometry["blocks"][-1]["owner_slots"], [4, 5])

    def test_source_derived_geometry_injection_is_rejected(self) -> None:
        geometry = contract.universal_format_geometry()
        geometry["profile_q"] = 7
        with self.assertRaisesRegex(contract.ContractError, "pre-frozen"):
            contract.validate_universal_geometry(geometry)


class AuthorizationTests(unittest.TestCase):
    def test_primary_requires_external_auditor_pins(self) -> None:
        value = primary_authorization()
        observed = contract.validate_primary_authorization(
            value,
            expected_auditor_manifest_sha256=d("primary auditor"),
            expected_audit_receipt_sha256=d("primary receipt"),
        )
        self.assertFalse(observed["positive_claim_authority"])
        with self.assertRaisesRegex(contract.ContractError, "external primary"):
            contract.validate_primary_authorization(
                value,
                expected_auditor_manifest_sha256=d("self-authored"),
                expected_audit_receipt_sha256=d("primary receipt"),
            )

    def test_primary_gate_tamper_is_rejected(self) -> None:
        value = primary_authorization()
        value["source_gates"]["cold_read_below_2x"] = False
        with self.assertRaisesRegex(contract.ContractError, "all source gates"):
            contract.validate_primary_authorization(
                value,
                expected_auditor_manifest_sha256=d("primary auditor"),
                expected_audit_receipt_sha256=d("primary receipt"),
            )

    def test_run_authorization_exact_closures(self) -> None:
        value = run_authorization()
        contract.validate_run_authorization(
            value,
            expected_consumer_source_manifest_sha256=d("consumer manifest"),
            expected_consumer_auditor_manifest_sha256=d("consumer auditor"),
            expected_consumer_audit_receipt_sha256=d("consumer receipt"),
            expected_producer_auditor_manifest_sha256=d("producer auditor"),
            expected_producer_audit_receipt_sha256=d("producer receipt"),
            expected_v8_all150_preflight_receipt_sha256=d("all150 preflight"),
            expected_gpu_identity_receipt_sha256=d("gpu identity"),
            expected_source_snapshot_root_sha256=d("source snapshot"),
            expected_descriptor_source_builder_sha256=d("descriptor builder"),
            expected_moment_replayer_source_sha256=d("moment replayer"),
            expected_audit_bootstrap_sha256=d("audit bootstrap"),
            expected_root_complete_sha256=d("root complete"),
            expected_primary_authorization_sha256=d("primary authorization file"),
        )
        value = run_authorization()
        value["v8_runtime_closure"].pop("v8_cupy_backend_sha256")
        with self.assertRaisesRegex(contract.ContractError, "v8 runtime closure fields"):
            contract.validate_run_authorization(
                value,
                expected_consumer_source_manifest_sha256=d("consumer manifest"),
                expected_consumer_auditor_manifest_sha256=d("consumer auditor"),
                expected_consumer_audit_receipt_sha256=d("consumer receipt"),
                expected_producer_auditor_manifest_sha256=d("producer auditor"),
                expected_producer_audit_receipt_sha256=d("producer receipt"),
                expected_v8_all150_preflight_receipt_sha256=d("all150 preflight"),
                expected_gpu_identity_receipt_sha256=d("gpu identity"),
                expected_source_snapshot_root_sha256=d("source snapshot"),
                expected_descriptor_source_builder_sha256=d("descriptor builder"),
                expected_moment_replayer_source_sha256=d("moment replayer"),
                expected_audit_bootstrap_sha256=d("audit bootstrap"),
                expected_root_complete_sha256=d("root complete"),
                expected_primary_authorization_sha256=d("primary authorization file"),
            )


class BundleGrammarTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_rejected(self) -> None:
        with self.assertRaisesRegex(contract.ContractError, "duplicate JSON field"):
            contract.strict_json(b'{"a":1,"a":2}', "duplicate")
        with self.assertRaisesRegex(contract.ContractError, "non-finite"):
            contract.strict_json(b'{"a":NaN}', "nonfinite")

    def test_member_path_traversal_and_noncanonical_order_rejected(self) -> None:
        with self.assertRaisesRegex(contract.ContractError, "traversal"):
            contract.safe_member_name("control/../../payload")
        rows = [
            {"name": "b", "bytes": 0, "sha256": d("b")},
            {"name": "a", "bytes": 0, "sha256": d("a")},
        ]
        with self.assertRaisesRegex(contract.ContractError, "canonical unique order"):
            contract.validate_member_rows(rows, "hostile")

    def test_control_binding_does_not_require_source_structural_equality(self) -> None:
        primary = primary_authorization()
        clean = {
            "schema": "uwfa-matched-gaussian-control-binding-v9",
            "seed": contract.CONTROL_SEEDS[0],
            "source_artifact_sha256": primary["source_artifact_sha256"],
            "source_full_geometry_sha256": primary["source_full_geometry_sha256"],
            "source_structural_geometry_sha256": primary["source_structural_geometry_sha256"],
            "pipeline_sha256": primary["source_pipeline_sha256"],
            "source_score_receipt_sha256": primary["source_score_receipt_sha256"],
            "source_moment_auditor_sha256": primary["source_moment_auditor_sha256"],
            "universal_format_geometry_sha256": contract.universal_format_geometry_sha256(),
            "generator_capsule_sha256": d("generator"),
            "moment_match_receipt_sha256": d("moment"),
            "source_panel_manifest_sha256": d("control source panel"),
            "control_artifact_sha256": d("control artifact"),
            "control_full_geometry_sha256": d("control full geometry"),
            "control_structural_geometry_sha256": d("different control structure"),
            "symmetric_codec_closure": symmetric_closure(),
        }
        binding = contract.sealed(clean, "binding_sha256")
        observed = contract._validate_control_binding(
            binding,
            seed=contract.CONTROL_SEEDS[0],
            primary=primary,
            universal_sha256=contract.universal_format_geometry_sha256(),
        )
        self.assertNotEqual(
            observed["control_structural_geometry_sha256"],
            primary["source_structural_geometry_sha256"],
        )


class Candidate:
    def __init__(self, selector: int):
        self.selector_ordinal = selector

    def as_dict(self):
        return {
            "selector_ordinal": self.selector_ordinal,
            "topology_id": self.selector_ordinal % 5,
            "states": 4 << (self.selector_ordinal % 6),
            "reset_length": (32, 128, 512, 2048, 8192)[self.selector_ordinal % 5],
        }


class Common:
    def candidate_bank(self):
        return tuple(Candidate(index) for index in range(150))


def scientific_fixture() -> dict:
    cells = []
    for candidate in Common().candidate_bank():
        cells.append(
            contract.sealed(
                {
                    **candidate.as_dict(),
                    "validation_charged_bits": 1000 + candidate.selector_ordinal,
                    "fitted_frequency_u16_sha256": d(
                        f"frequency {candidate.selector_ordinal}"
                    ),
                    "validation_lengths_u64_sha256": d(
                        f"length {candidate.selector_ordinal}"
                    ),
                    "validation_stream_ordinals": [0, 1],
                    "trained_only_on_inner_train_streams": True,
                    "source_winner_reused": False,
                },
                "cell_result_sha256",
            )
        )
    return {
        "estimable": True,
        "source_winner_reused": False,
        "complete_150_cell_search_recorded_every_fold": True,
        "folds": [
            {
                "all_150_inner_validation_cells": cells,
                "all_150_cell_results_sha256": contract.sha256(
                    contract.canonical_json(cells)
                ),
                "selected_by_inner_validation_only": Candidate(0).as_dict(),
            }
        ],
    }


class All150AndDecisionTests(unittest.TestCase):
    def test_all_150_exact_receipt(self) -> None:
        contract.validate_all150_scientific(scientific_fixture(), Common())

    def test_149_cells_and_source_winner_reuse_are_rejected(self) -> None:
        value = scientific_fixture()
        value["folds"][0]["all_150_inner_validation_cells"].pop()
        with self.assertRaisesRegex(contract.ContractError, "exact 150"):
            contract.validate_all150_scientific(value, Common())
        value = scientific_fixture()
        value["source_winner_reused"] = True
        with self.assertRaisesRegex(contract.ContractError, "source winner"):
            contract.validate_all150_scientific(value, Common())

    def test_first_equal_null_is_decisive(self) -> None:
        rows = [
            {
                "seed": contract.CONTROL_SEEDS[0],
                "absolute_saving_bpw": 0.20,
            }
        ]
        decision = contract.early_null_decision(
            source_saving_bpw=0.20,
            executed_controls=rows,
            authenticated_controls=8,
        )
        self.assertEqual(decision["status"], "HARD_KILL_MATCHED_GAUSSIAN_NOT_SPECIFIC")
        self.assertTrue(decision["early_stop_required"])

    def test_positive_requires_all_eight_strictly_below(self) -> None:
        seven = [
            {"seed": seed, "absolute_saving_bpw": 0.10}
            for seed in contract.CONTROL_SEEDS[:7]
        ]
        self.assertEqual(
            contract.early_null_decision(
                source_saving_bpw=0.20,
                executed_controls=seven,
                authenticated_controls=8,
            )["status"],
            "BLOCK_INCOMPLETE_MATCHED_NULL_SEQUENCE",
        )
        eight = seven + [
            {"seed": contract.CONTROL_SEEDS[7], "absolute_saving_bpw": 0.199}
        ]
        self.assertTrue(
            contract.early_null_decision(
                source_saving_bpw=0.20,
                executed_controls=eight,
                authenticated_controls=8,
            )["specificity_pass"]
        )


class StaticBoundaryTests(unittest.TestCase):
    def test_direct_entrypoints_are_inert(self) -> None:
        self.assertEqual(contract.direct_main(), 3)
        self.assertEqual(runtime.direct_main(), 3)

    def test_runtime_authenticates_everything_before_backend(self) -> None:
        source = (PACKAGE / "matched_controls_consumer.py").read_text(
            encoding="utf-8"
        )
        consume = source[source.index("def consume_controls(") :]
        self.assertLess(
            consume.index("authenticate_eight_control_root"),
            consume.index("backend_factory()"),
        )
        self.assertLess(
            consume.index("_prepare_all_controls"),
            consume.index("backend_factory()"),
        )
        self.assertIn("all_150_inner_validation_cells", source)
        self.assertIn("source_winner_reused", source)
        self.assertNotIn("control_structural_geometry_sha256 == primary", source)

    def test_no_numeric_or_model_imports(self) -> None:
        source = (
            (PACKAGE / "consumer_contract.py").read_text(encoding="utf-8")
            + (PACKAGE / "matched_controls_consumer.py").read_text(encoding="utf-8")
        ).lower()
        self.assertNotIn("import numpy", source)
        self.assertNotIn("import cupy", source)
        self.assertNotIn("model.layers.", source)


if __name__ == "__main__":
    unittest.main()
