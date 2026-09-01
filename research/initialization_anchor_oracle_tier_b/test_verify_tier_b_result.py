"""CPU-only tests for the independent Tier-B result verifier."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import common
import verify_tier_b_result as verifier


class TierBVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = Path(__file__).resolve().parent
        cls.result_path = cls.package / "result.json"
        cls.calibration_path = cls.package / "tier_b_source_free_calibration.json"
        cls.recovery_path = cls.package / "recovery_orphan_stage0_136.json"
        cls.result = json.loads(cls.result_path.read_text(encoding="utf-8"))
        cls.calibration = json.loads(cls.calibration_path.read_text(encoding="utf-8"))
        cls.recovery = json.loads(cls.recovery_path.read_text(encoding="utf-8"))
        cls.rows = common.load_source_rows()
        cls.full_plan = common.make_plan(cls.rows, stage0=False)

    def test_01_published_artifacts_pass_without_payload_or_state(self) -> None:
        receipt = verifier.verify_result(self.result_path)
        self.assertEqual("PASS", receipt["status"])
        self.assertEqual("HARD_KILL_BOUNDED_TIER_B_PROCEDURAL_SET", receipt["decision_state"])
        self.assertFalse(receipt["journal_verification"]["performed"])
        self.assertFalse(receipt["payload_verification"]["rehash_performed"])
        self.assertEqual(0, receipt["excluded_payloads_opened"])

    def test_02_published_hash_inventory(self) -> None:
        self.assertEqual(verifier.PUBLISHED_RESULT_SHA256, common.sha256_file(self.result_path))
        self.assertEqual(verifier.PUBLISHED_CALIBRATION_SHA256, common.sha256_file(self.calibration_path))
        self.assertEqual(verifier.PUBLISHED_RECOVERY_SHA256, common.sha256_file(self.recovery_path))
        self.assertEqual(verifier.RUNNER_SHA256, common.sha256_file(self.package / "tier_b_gate.py"))
        self.assertEqual(verifier.COMMON_SHA256, common.sha256_file(self.package / "common.py"))
        self.assertEqual(verifier.KERNELS_SHA256, common.sha256_file(self.package / "kernels.py"))

    def test_03_calibration_and_production_parity_are_exact(self) -> None:
        verifier._verify_calibration(self.calibration, common.load_candidate_lock())
        self.assertEqual(self.calibration["parity"], self.result["backend"]["parity"])
        self.assertEqual(45, len(self.calibration["parity"]["descriptor_checks"]))
        self.assertEqual(810, len(self.calibration["parity"]["candidate_coordinate_checks"]))
        self.assertEqual(9, len(self.calibration["parity"]["persistent_packing_checks"]))

    def test_04_embedded_event_chain_and_recovery(self) -> None:
        events, final_hash = verifier._verify_embedded_events(self.result)
        self.assertEqual(392, len(events))
        self.assertEqual(verifier.LAST_EVENT_SHA256, final_hash)
        verifier._verify_recovery_evidence(self.recovery, events)

    def test_05_embedded_event_mutation_fails(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["resume_state"]["events"][17]["file_bytes"] += 1
        with self.assertRaises(common.ProtocolError):
            verifier._verify_embedded_events(mutated)

    def test_06_winner_candidate_bindings_and_tie_representatives(self) -> None:
        winners = self.result["search"]["stage1_winners"]
        self.assertEqual(set(common.DOMAIN_IDS), set(winners))
        for domain_id in common.DOMAIN_IDS:
            record = winners[domain_id]
            candidate = common.decode_ordinal(int(record["candidate"]["ordinal"]))
            self.assertEqual(candidate.to_json(), record["candidate"])
            self.assertIn(candidate.pp_index, (0, 2, 3))
            self.assertAlmostEqual(
                float(record["selection_q"]),
                float(self.result["search"]["selection_folds"][domain_id]["pooled"]["q"]),
                places=7,
            )

    def test_07_detail_moment_algebra_detects_mutation(self) -> None:
        selection_plan = [row for row in self.full_plan if row.source.split == "candidate_selection"]
        row = self.result["search"]["selection_details"]["source"][0]
        verifier._verify_detail_row(row, selection_plan[0], "source[0]")
        mutated = copy.deepcopy(row)
        mutated["score"]["sse"] += 0.01
        with self.assertRaises(common.ProtocolError):
            verifier._verify_detail_row(mutated, selection_plan[0], "mutated")

    def test_08_null_correction_and_physical_ledger_recompute(self) -> None:
        expected_ledger = common.physical_ledger()
        self.assertEqual(expected_ledger, self.result["physical_ledger"])
        expected_decision = common.make_decision(
            self.result["validation"]["folds"]["source"],
            self.result["validation"]["null_captures"],
        )
        self.assertEqual(expected_decision, self.result["decision"])
        self.assertLess(
            expected_decision["bias_corrected_upper_3se"],
            expected_decision["metadata_adjusted_composite_required_capture"],
        )

    def test_09_firewall_has_exact_order_and_excluded_payload_unopened(self) -> None:
        firewall = self.result["data_firewall"]
        eligible = [row for row in self.rows if not row.excluded]
        self.assertEqual(31, len(eligible))
        self.assertEqual(0, firewall["excluded_payloads_opened"])
        self.assertEqual(
            [row.tensor_name for row in eligible],
            [row["tensor_name"] for row in firewall["eligible"]],
        )
        self.assertEqual(24, firewall["access_log"][24]["sequence"])
        self.assertEqual(
            "all_33_global_winners_state_backed_before_validation_payload_access",
            firewall["access_log"][24]["event"],
        )
        self.assertTrue(all(not row["payload_opened"] for row in firewall["excluded"]))
        self.assertFalse(self.result["pinned_panel"]["opened"])

    def test_10_import_is_cuda_library_free(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("cupy", sys.modules)
        self.assertFalse(common.environment_has_cuda_imports())

    def test_11_external_manifest_runtime_status_is_environmental(self) -> None:
        recorded = copy.deepcopy(self.result["data_firewall"]["exclusion_binding"])
        current = copy.deepcopy(recorded)
        current["full_external_manifest_revalidated_at_runtime"] = not recorded[
            "full_external_manifest_revalidated_at_runtime"
        ]
        receipt = verifier._verify_exclusion_binding(recorded, current)
        self.assertTrue(receipt["immutable_binding_verified"])
        self.assertEqual(
            recorded["full_external_manifest_revalidated_at_runtime"],
            receipt["recorded_full_external_manifest_revalidated_at_runtime"],
        )
        self.assertEqual(
            current["full_external_manifest_revalidated_at_runtime"],
            receipt["current_full_external_manifest_revalidated_at_runtime"],
        )

    def test_12_immutable_exclusion_binding_mutations_fail(self) -> None:
        recorded = copy.deepcopy(self.result["data_firewall"]["exclusion_binding"])
        current = copy.deepcopy(recorded)
        mutations = {
            "packaged_intersection_lock_sha256": "0" * 64,
            "source_exclusion_manifest_sha256": "1" * 64,
            "excluded_tensor_identities": ["model.layers.15.mlp.experts.8.up_proj.weight"],
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                mutated = copy.deepcopy(recorded)
                mutated[key] = value
                with self.assertRaises(common.ProtocolError):
                    verifier._verify_exclusion_binding(mutated, current)


if __name__ == "__main__":
    unittest.main()
