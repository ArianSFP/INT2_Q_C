from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common
import kernels
import tier_b_gate


class TierBFrozenProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = common.default_workspace_root()

    def test_lock_seals_counts_sources_and_nulls(self) -> None:
        lock = common.load_candidate_lock()
        self.assertEqual(common.CANDIDATE_LOCK_FILE_SHA256, common.sha256_file(common.CANDIDATE_LOCK_PATH))
        self.assertEqual(37_748_736, lock["logical_key_space"]["logical_candidate_count"])
        self.assertEqual(28_311_552, lock["equivalence_deduplication"]["effective_candidate_count"])
        self.assertEqual(33, lock["search_cascade"]["search_domain_count"])
        self.assertEqual(32, len(common.NULL_DOMAIN_IDS))
        self.assertEqual(16, sum(value.startswith("gaussian_") for value in common.NULL_DOMAIN_IDS))
        self.assertEqual(16, sum(value.startswith("scramble_") for value in common.NULL_DOMAIN_IDS))
        self.assertIn("100*pipeline_parallel_rank", lock["procedural_family"]["expert_seed_formula"])

    def test_candidate_ordinal_roundtrip_and_shard(self) -> None:
        first = common.decode_ordinal(0)
        last = common.decode_ordinal(common.LOGICAL_CANDIDATES - 1)
        self.assertEqual(0, first.base_seed)
        self.assertEqual(65_535, last.base_seed)
        self.assertEqual("fused_up_gate_then_down", last.projection_packing)
        for ordinal in (0, 1, 575, 576, 1234567, common.LOGICAL_CANDIDATES - 1):
            candidate = common.decode_ordinal(ordinal)
            self.assertEqual(
                ordinal,
                common.logical_ordinal(
                    candidate.base_seed,
                    candidate.pp_index,
                    candidate.ep_index,
                    candidate.etp_index,
                    candidate.assignment_index,
                    candidate.packing_index,
                ),
            )
        shard = common.representative_ordinals(0, 256)
        self.assertEqual(110_592, len(shard))
        self.assertEqual(len(shard), len(np.unique(shard)))
        self.assertTrue(all(common.decode_ordinal(int(value)).pp_index in (0, 2, 3) for value in shard[:1000]))

    def test_pp_equivalence_and_end_to_end_seed(self) -> None:
        arguments = dict(base_seed=1234, ep_index=3, etp_index=2, assignment_index=1, packing_index=1)
        candidates = []
        for pp_index in range(4):
            ordinal = common.logical_ordinal(
                arguments["base_seed"], pp_index, arguments["ep_index"], arguments["etp_index"],
                arguments["assignment_index"], arguments["packing_index"],
            )
            candidates.append(common.decode_ordinal(ordinal))
        descriptors = [
            kernels.coordinate_descriptor(candidate, 57, "up", 987654, 170, 1536)
            for candidate in candidates
        ]
        self.assertEqual(descriptors[0], descriptors[1])
        self.assertEqual(descriptors[0].seed + 100, descriptors[2].seed)
        self.assertEqual(descriptors[0].seed + 200, descriptors[3].seed)
        self.assertEqual(15, descriptors[0].local_layer)
        self.assertEqual(3, descriptors[2].local_layer)
        self.assertEqual("fc353f4bc6e5431b7c9891d3e87a5778c9c21342aafd8ab5a8fea007c50cf0c8", common.equivalence_map_sha256())

    def test_execution_policy(self) -> None:
        for numel in (1, 257, 393216, 786432, 1572864, 3145728):
            grid = min((numel + 255) // 256, 170 * 6)
            expected = ((numel - 1) // (256 * grid * 4) + 1) * 4
            self.assertEqual(expected, kernels.policy_increment(numel, 170, 1536))

    def test_coordinate_plans_are_bound_and_disjoint(self) -> None:
        rows = common.load_source_rows(self.workspace)
        stage0 = common.make_plan(rows, stage0=True)
        full = common.make_plan(rows, stage0=False)
        self.assertEqual("f1f4fc00dab8ba5b9856e14679fb9f571b035af483d6b5655a8e030ca35873bd", common.plan_sha256(stage0))
        self.assertEqual("61082d53eb472edb2c0a51f1c3a970c2bf207147140562572fdb9db1b1b68dcf", common.plan_sha256(full))
        self.assertEqual(512, sum(len(row.fit) + len(row.score) for row in stage0))
        self.assertEqual(65536, sum(len(row.fit) + len(row.score) for row in full))
        for plan in (stage0, full):
            for row in plan:
                self.assertFalse(set(row.fit) & set(row.score))

    def test_null_domains_are_deterministic_and_distinct(self) -> None:
        coordinates = [1, 22, 333, 4444]
        first = common.stateless_normals("gaussian_00", "tensor", "fit", coordinates)
        second = common.stateless_normals("gaussian_00", "tensor", "fit", coordinates)
        other = common.stateless_normals("gaussian_01", "tensor", "fit", coordinates)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        permutation, signs = common.permutation_and_sign("scramble_00", "tensor", "fit", 100)
        np.testing.assert_array_equal(np.arange(100), np.sort(permutation))
        self.assertTrue(set(signs.tolist()) <= {-1.0, 1.0})

    def test_physical_and_promotion_boundaries(self) -> None:
        ledger = common.physical_ledger()
        self.assertAlmostEqual(0.0000226056134259259, ledger["side_bpw"], places=18)
        self.assertAlmostEqual(0.145688848385821, ledger["metadata_adjusted_composite_required_capture"], places=14)
        self.assertAlmostEqual(0.191020609160754, ledger["metadata_adjusted_standalone_required_capture"], places=14)
        self.assertLess(ledger["conservative_appended_cold_read_amplification"], 2.0)
        nulls = {domain: 0.0 for domain in common.NULL_DOMAIN_IDS}

        def folds(capture):
            return {
                "pooled": {"capture": capture},
                "whole_expert_capture_standard_error": 0.0,
                "whole_experts": [{"capture": capture}] * 4,
                "roles": [{"capture": capture}] * 2,
            }

        composite = ledger["metadata_adjusted_composite_required_capture"]
        standalone = ledger["metadata_adjusted_standalone_required_capture"]
        self.assertEqual(
            "HARD_KILL_BOUNDED_TIER_B_PROCEDURAL_SET",
            common.make_decision(folds(math.nextafter(composite, -math.inf)), nulls)["state"],
        )
        self.assertEqual(
            "COMPOSITE_PROCEDURAL_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL",
            common.make_decision(folds(composite), nulls)["state"],
        )
        self.assertEqual(
            "STANDALONE_PROCEDURAL_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL",
            common.make_decision(folds(standalone), nulls)["state"],
        )

    def test_state_journal_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = tier_b_gate.StateJournal(root)
            first = journal.write_json("header", "one", {"value": 1})
            journal.write_npz("array", "two", values=np.arange(10))
            resumed = tier_b_gate.StateJournal(root)
            self.assertEqual(2, len(resumed.events_list))
            first.write_text("corrupt", encoding="utf-8")
            with self.assertRaises(common.ProtocolError):
                tier_b_gate.StateJournal(root)

    def test_kernel_module_import_is_cuda_free(self) -> None:
        self.assertNotIn("cupy", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertIn("curand_normal4", kernels.CUDA_SOURCE)
        self.assertIn("__float2bfloat16_rn", kernels.CUDA_SOURCE)

    def test_cpu_preflight(self) -> None:
        report = tier_b_gate.cpu_preflight(self.workspace)
        self.assertEqual("PASS_CUDA_NOT_IMPORTED_OR_TOUCHED", report["status"])
        self.assertEqual(37_748_736, report["logical_candidates"])
        self.assertEqual(28_311_552, report["effective_candidates"])
        self.assertEqual(48_624, report["selection_full_coordinates"])
        self.assertFalse(report["cuda_modules_imported"])


if __name__ == "__main__":
    unittest.main()
