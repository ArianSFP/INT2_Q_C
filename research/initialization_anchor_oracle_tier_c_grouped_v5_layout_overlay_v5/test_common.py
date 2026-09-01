from __future__ import annotations

import math
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import common
import kernels


class TierCGroupedV5LayoutOverlayCommonTests(unittest.TestCase):
    def test_boundary_rejects_output_below_or_above_any_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            source.mkdir()
            with self.assertRaisesRegex(common.ProtocolError, "ancestry overlap"):
                common.BoundaryGuard(
                    "TEST_NESTED_OUTPUT",
                    outputs=(("output", source / "nested", "directory", False),),
                    inputs=(("input", source, "directory"),),
                )
            with self.assertRaisesRegex(common.ProtocolError, "ancestry overlap"):
                common.BoundaryGuard(
                    "TEST_ANCESTOR_OUTPUT",
                    outputs=(("output", root, "directory", True),),
                    inputs=(("input", source, "directory"),),
                )

    def test_boundary_rejects_hardlink_device_inode_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.bin"
            alias = root / "alias.bin"
            source.write_bytes(b"identity fixture")
            try:
                alias.hardlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest("hard links unavailable")
            with self.assertRaisesRegex(common.ProtocolError, "device/inode alias"):
                common.BoundaryGuard(
                    "TEST_HARDLINK",
                    outputs=(("output", alias, "file", True),),
                    inputs=(("input", source, "file"),),
                )

    def test_boundary_rejects_synthetic_bind_mount_coordinate_alias(self):
        left = {
            "label": "output alias",
            "path": "/tmp/outside-a",
            "exists": False,
            "device": None,
            "inode": None,
            "mount": {"major_minor": "8:1", "filesystem_path": "/shared/tree"},
        }
        right = {
            "label": "input alias",
            "path": "/mnt/outside-b",
            "exists": True,
            "device": 9,
            "inode": 10,
            "mount": {"major_minor": "8:1", "filesystem_path": "/shared/tree/input"},
        }
        with self.assertRaisesRegex(common.ProtocolError, "mount-coordinate alias"):
            common._assert_boundary_pair_disjoint(left, right)

    def test_boundary_revalidation_rejects_output_inode_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            source.mkdir()
            output = root / "output"
            guard = common.BoundaryGuard(
                "TEST_REPLACEMENT",
                outputs=(("output", output, "directory", False),),
                inputs=(("input", source, "directory"),),
            )
            output.mkdir()
            guard.revalidate("bind created output")
            output.rename(root / "old-output")
            output.mkdir()
            with self.assertRaisesRegex(common.ProtocolError, "object identity changed"):
                guard.revalidate("detect replacement")

    def test_boundary_requires_raw_absolute_canonical_and_no_symlink_components(self):
        with self.assertRaisesRegex(common.ProtocolError, "absolute path"):
            common.require_canonical_absolute_spelling(Path("relative"), "fixture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            real = root / "real"
            real.mkdir()
            link = root / "LINK"
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(common.ProtocolError, "symlink component"):
                common.BoundaryGuard(
                    "TEST_SYMLINK_COMPONENT",
                    outputs=(("output", root / "out", "directory", False),),
                    inputs=(("input", link, "directory"),),
                )

    def test_cpu_import_firewall(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("cupy", sys.modules)
        self.assertNotIn("transformer_engine", sys.modules)
        self.assertFalse(common.environment_has_cuda_imports())

    def test_stored_u16_maps_to_first_65536_positive_cli_seeds(self):
        first = common.decode_ordinal(common.logical_ordinal(0, 0, 0, 0, 0, 0, 0))
        last = common.decode_ordinal(common.logical_ordinal(65_535, 9, 7, 3, 1, 1, 1))
        self.assertEqual((first.stored_seed_u16, first.cli_base_seed), (0, 1))
        self.assertEqual((last.stored_seed_u16, last.cli_base_seed), (65_535, 65_536))
        self.assertEqual(common.LOGICAL_CANDIDATES, 167_772_160)

    def test_roundtrip_all_axis_boundaries(self):
        for axes in ((0, 0, 0, 0, 0, 0, 0), (65_535, 9, 7, 3, 1, 1, 1),
                     (3407, 6, 3, 2, 1, 0, 1)):
            candidate = common.decode_ordinal(common.logical_ordinal(*axes))
            self.assertEqual((candidate.stored_seed_u16, candidate.pp_index,
                              candidate.ep_index, candidate.etp_index,
                              candidate.assignment_index, candidate.half_index,
                              candidate.abi_index), axes)

    def test_exact_global_equivalence_count(self):
        audit = common.equivalence_audit()
        self.assertEqual(audit["seed_pp_rows_exhausted"], 655_360)
        self.assertEqual(audit["full_union_distinct_anchor_count"], 58_720_256)
        self.assertEqual(audit["new_distinct_anchor_count"], 42_205_184)
        self.assertEqual(len(common.representative_ordinals(0, 256)), 164_864)
        self.assertEqual(len(common.full_representative_ordinals(0, 256)), 229_376)
        self.assertEqual(256 * 164_864, common.NEW_EFFECTIVE_CANDIDATES)
        self.assertEqual(256 * 229_376, common.FULL_EFFECTIVE_CANDIDATES)

    def test_v5_equivalence_map_is_frozen(self):
        self.assertEqual(
            common.equivalence_map_sha256(),
            "f48f49d85fa0284a6fdacafbcfda67e613e5eb34203e52d2269ebfb195b93a68",
        )
        self.assertEqual(
            common.equivalence_map_object()["schema"],
            "tier_c_grouped_v5_layout_overlay_equivalence_map_v1",
        )

    def test_ep_endpoint_assignment_and_abi_relations(self):
        for ep_index in (0, 7):
            a = common.logical_ordinal(7, 0, ep_index, 3, 0, 1, 0)
            b = common.logical_ordinal(7, 0, ep_index, 3, 1, 1, 1)
            self.assertEqual(common.representative_ordinal(a), common.representative_ordinal(b))
        for ep_index in range(1, 7):
            a = common.logical_ordinal(7, 0, ep_index, 3, 0, 1, 0)
            b = common.logical_ordinal(7, 0, ep_index, 3, 1, 1, 0)
            self.assertNotEqual(common.representative_ordinal(a), common.representative_ordinal(b))

    def test_only_pp1_pp2_pp3_collapse_and_cross_seed_dedup_is_absent(self):
        def rep(seed, pp):
            return common.representative_ordinal(common.logical_ordinal(seed, pp, 4, 3, 1, 1, 0))
        for seed in (0, 1, 99, 100, 65_535):
            self.assertEqual(rep(seed, 0), rep(seed, 1))
            self.assertEqual(rep(seed, 0), rep(seed, 2))
            self.assertEqual(len({rep(seed, pp) for pp in (0, 3, 4, 5, 6, 7, 8, 9)}), 8)
        for seed in range(1, common.STORED_SEED_COUNT):
            self.assertNotEqual(rep(seed, 3), rep(seed - 1, 3))
        for seed in (0, 99, 100, 65_535):
            for pp in range(10):
                ordinal = common.logical_ordinal(seed, pp, 4, 3, 1, 1, 1)
                self.assertLessEqual(common.representative_ordinal(ordinal), ordinal)

    def test_new_only_enumeration_is_exact_disjoint_and_strictly_sorted(self):
        new = common.representative_ordinals(0, 3)
        full = common.full_representative_ordinals(0, 3)
        self.assertEqual((len(new), len(full)), (3 * 644, 3 * 896))
        self.assertTrue(np.all(new[1:] > new[:-1]))
        self.assertTrue(np.all(full[1:] > full[:-1]))
        translated_old = []
        for seed in range(3):
            for pp in (0, 2, 3):
                for ep in range(8):
                    for etp in range(3):
                        for assignment in ((0,) if ep in (0, 7) else (0, 1)):
                            for half in range(2):
                                old = common.v4_logical_ordinal(seed, pp, ep, etp, assignment, half, 0)
                                translated_old.append(common.translate_v4_ordinal(old))
        translated_old = np.asarray(translated_old, dtype=np.uint64)
        self.assertEqual(len(translated_old), 3 * 252)
        self.assertTrue(np.all(translated_old[1:] > translated_old[:-1]))
        self.assertEqual(np.intersect1d(new, translated_old).size, 0)
        self.assertEqual(np.union1d(new, translated_old).size, len(full))

    def test_create_new_output_rejects_existing_and_dangling_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "receipt.json"
            common.write_json_create_new(output, {"ok": True}, "fixture output")
            with self.assertRaises(common.ProtocolError):
                common.write_json_create_new(output, {"ok": False}, "fixture output")
            dangling_target = root / "must-not-be-created.json"
            dangling = root / "dangling.json"
            try:
                dangling.symlink_to(dangling_target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaises(common.ProtocolError):
                common.write_json_create_new(dangling, {"bad": True}, "fixture output")
            self.assertFalse(dangling_target.exists())

    def test_output_parent_and_directory_symlinks_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            parent_link = root / "parent-link"
            dangling_dir = root / "dangling-dir"
            try:
                parent_link.symlink_to(real, target_is_directory=True)
                dangling_dir.symlink_to(root / "missing-dir", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaises(common.ProtocolError):
                common.preflight_create_new_file(parent_link / "x.json", "parent-link output")
            with self.assertRaises(common.ProtocolError):
                common.preflight_output_directory(
                    dangling_dir, allow_existing=True, label="dangling output directory"
                )

    def test_original_symlink_plus_parent_traversal_is_rejected_before_abspath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "LINK"
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            disguised = link / ".." / "create-new"
            self.assertIn("..", disguised.parts)
            with self.assertRaisesRegex(common.ProtocolError, "parent traversal"):
                common.preflight_output_directory(
                    disguised, allow_existing=False, label="disguised output"
                )
            with self.assertRaisesRegex(common.ProtocolError, "parent traversal"):
                common.require_regular_file_before_resolve(
                    disguised / "receipt.json", "disguised input"
                )

    def test_existing_ancestor_symlink_is_rejected_without_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            receipt = real / "receipt.json"
            receipt.write_text("{}", encoding="utf-8")
            link = root / "LINK"
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaisesRegex(common.ProtocolError, "symlink component"):
                common.require_regular_file_before_resolve(
                    link / receipt.name, "linked input"
                )

    def test_qwen_workspace_and_aux_symlink_components_fail_before_consumer_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "LINK"
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with mock.patch.object(
                common, "_read_bound_json_with_events",
                side_effect=AssertionError("manifest consumer reached"),
            ):
                with self.assertRaisesRegex(common.ProtocolError, "symlink component"):
                    common.load_source_rows(link, [])
            with mock.patch.object(
                common.os, "listdir", side_effect=AssertionError("directory consumer reached")
            ):
                with self.assertRaisesRegex(common.ProtocolError, "symlink component"):
                    common.validate_aux_directory(link, (), [])

    def test_projection_major_descriptor_uses_positive_cli_seed(self):
        candidate = common.decode_ordinal(common.logical_ordinal(0, 7, 3, 3, 1, 0, 0))
        descriptor = kernels.coordinate_descriptor(candidate, 57, "up", 786_431, 170, 1536)
        self.assertEqual(candidate.cli_base_seed, 1)
        self.assertGreaterEqual(descriptor.seed, 1 + 1024)
        self.assertGreater(kernels.policy_increment(descriptor.target_numel, 170, 1536), 0)

    def test_half_assignment_changes_only_up_native_index(self):
        gate_up = common.decode_ordinal(common.logical_ordinal(1, 0, 0, 0, 0, 0, 0))
        up_gate = common.decode_ordinal(common.logical_ordinal(1, 0, 0, 0, 0, 1, 0))
        coordinate = 123 * common.COLUMNS + 17
        a_up = kernels.coordinate_descriptor(gate_up, 9, "up", coordinate, 170, 1536)
        b_up = kernels.coordinate_descriptor(up_gate, 9, "up", coordinate, 170, 1536)
        a_down = kernels.coordinate_descriptor(gate_up, 9, "down", coordinate, 170, 1536)
        b_down = kernels.coordinate_descriptor(up_gate, 9, "down", coordinate, 170, 1536)
        self.assertEqual(abs(a_up.native_index - b_up.native_index), common.ROWS * common.COLUMNS)
        self.assertEqual(a_up.target_offset, b_up.target_offset)
        self.assertEqual(a_down, b_down)

    def test_fp16_affine_codec_is_exact_four_byte_little_endian_roundtrip(self):
        payload, alpha, mu = common.quantize_affine_f16le(0.123456789, -0.000987654)
        self.assertEqual(len(payload), 4)
        decoded = np.frombuffer(payload, dtype="<f2").astype(np.float64)
        self.assertEqual((alpha, mu), (float(decoded[0]), float(decoded[1])))
        fit = common.fit_affine_moments([1.0, 3.0, 2.0], [0.5, -0.25, 1.0])
        self.assertEqual(fit["affine_storage_bytes"], 4)
        self.assertEqual(bytes.fromhex(fit["alpha_mu_f16le_hex"]),
                         common.quantize_affine_f16le(fit["alpha"], fit["mu"])[0])
        score = common.score_affine_moments([0.0, 1.0], [0.25, -0.5],
                                            fit["alpha"], fit["mu"], fit["fit_mean_w"])
        self.assertTrue(score["score_uses_decoded_fp16"])

    def test_non_fp16_decoded_score_coefficients_fail_closed(self):
        with self.assertRaises(common.ProtocolError):
            common.score_affine_moments([0.0], [0.0], 0.123456789, 0.0, 0.0)
        with self.assertRaises(common.ProtocolError):
            common.quantize_affine_f16le(math.inf, 0.0)

    def test_matched_null_gate_is_descriptive_not_randomization_p(self):
        source = {"pooled": {"capture": 0.2}, "whole_expert_capture_standard_error": 0.01,
                  "whole_experts": [{"capture": 0.1}] * 4,
                  "roles": [{"capture": 0.1}] * 2}
        controls = {domain: -0.01 for domain in common.NULL_DOMAIN_IDS}
        decision = common.make_decision(source, controls)
        self.assertFalse(decision["randomization_p_value_claimed"])
        self.assertEqual(decision["empirical_control_rank_denominator"], 33)
        self.assertIn("not_exchangeable", decision["control_rank_interpretation"])

    def _decision_at_capture(self, capture):
        source = {
            "pooled": {"capture": float(capture)},
            "whole_expert_capture_standard_error": 0.0,
            "whole_experts": [{"capture": 0.01}] * 4,
            "roles": [{"capture": 0.01}] * 2,
        }
        controls = {domain: -0.01 for domain in common.NULL_DOMAIN_IDS}
        return common.make_decision(source, controls)

    def test_composite_screen_exact_boundary_never_claims_final_target(self):
        composite = common.physical_ledger()["metadata_adjusted_composite_required_capture"]
        below = self._decision_at_capture(np.nextafter(composite, -np.inf))
        at = self._decision_at_capture(composite)
        self.assertEqual(
            below["state"], "HARD_KILL_BOUNDED_TIER_C_GROUPED_V5_LAYOUT_OVERLAY_SET"
        )
        self.assertEqual(
            at["state"],
            "COMPOSITE_SCREEN_ONLY_REQUIRES_FINITE_AUDITED_COMPOSITION_NOT_FINAL_TARGET_CLAIM",
        )
        self.assertFalse(at["composite_screen_is_final_20_percent_below_gaussian_claim"])
        self.assertTrue(
            at["final_rate_distortion_claim_requires_separately_finite_audited_composition"]
        )

    def test_standalone_screen_exact_boundary_and_predecessor_status(self):
        ledger = common.physical_ledger()
        composite = ledger["metadata_adjusted_composite_required_capture"]
        standalone = ledger["metadata_adjusted_standalone_required_capture"]
        below = self._decision_at_capture(np.nextafter(standalone, -np.inf))
        at = self._decision_at_capture(standalone)
        self.assertGreater(below["bias_corrected_lower_3se"], composite)
        self.assertEqual(
            below["state"],
            "COMPOSITE_SCREEN_ONLY_REQUIRES_FINITE_AUDITED_COMPOSITION_NOT_FINAL_TARGET_CLAIM",
        )
        self.assertEqual(
            at["state"],
            "STANDALONE_ENERGY_SCREEN_SURVIVOR_REQUIRES_SEPARATELY_FROZEN_CODEC_COMPOSITION",
        )
        self.assertFalse(at["standalone_screen_is_final_20_percent_below_gaussian_claim"])

    def test_lock_rejects_dimensionally_wrong_capture_gate_semantics(self):
        lock = common.load_candidate_lock()
        chance = lock["chance_search_control"]
        self.assertNotIn("primary_gate", chance)
        self.assertTrue(chance["deprecated_0_160964_design_value_is_not_a_capture_gate"])
        self.assertFalse(chance["final_20_percent_below_gaussian_claim_emitted_by_this_gate"])
        self.assertEqual(
            chance["nested_reference"],
            common.physical_ledger()["metadata_adjusted_composite_required_capture"],
        )
        self.assertEqual(
            chance["standalone_reference"],
            common.physical_ledger()["metadata_adjusted_standalone_required_capture"],
        )

    def test_rate_and_read_ledger(self):
        ledger = common.physical_ledger()
        self.assertEqual(ledger["metadata_bytes_total"], 80)
        self.assertEqual(ledger["per_matrix_affine_bytes"], 4)
        self.assertTrue(ledger["scientific_scores_use_decoded_fp16_affines"])
        self.assertEqual(ledger["strict_bpw_cap"], 2.15)
        self.assertTrue(ledger["upstream_read_baseline_requires_later_composition_receipt"])
        self.assertTrue(ledger["passes_read_gate_arithmetic_only"])

    def test_exclusion_binding_never_resolves_or_stats_external_manifest(self):
        self.assertEqual(common.exclusion_binding(), {
            "mode": "packaged_identity_only",
            "excluded_tensor_identities": [common.EXCLUDED_TENSOR],
            "external_heldout_manifest_path_resolved": False,
            "external_heldout_manifest_existence_checked": False,
            "external_heldout_manifest_statted": False,
            "external_heldout_manifest_opened_or_read": False,
        })


if __name__ == "__main__":
    unittest.main()
