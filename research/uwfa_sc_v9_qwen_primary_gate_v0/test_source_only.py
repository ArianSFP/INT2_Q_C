#!/usr/bin/env python3
"""Hostile source-only tests; never discover or open Qwen/control payloads."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).absolute().parent
REPOSITORY = PACKAGE.parents[1]
RUNNER = PACKAGE / "primary_gate.py"
V8 = REPOSITORY / "research" / "unifilar_wfa_entropy_census_stage0_v8"
SUPPORT = REPOSITORY / "research" / "uwfa_sc_v8_qwen_early_gate_v0" / "early_gate.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("uwfa_sc_v9_primary_source_test", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("runner loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def make_review(conservative: float = 3_242_398.2106118356):
    clean = {
        "status": "PASS_AUTHENTICATED_SOURCE_FREE_REVIEW",
        "v9_source_snapshot_root_sha256": "11" * 32,
        "v8_source_snapshot_root_sha256": "22" * 32,
        "preflight_receipt_sha256": "33" * 32,
        "support_sha256": runner.PINNED_SUPPORT_SHA256,
        "measured_updates_per_second": 2.0 * conservative,
        "conservative_updates_per_second": conservative,
        "device_name": runner.EXPECTED_DEVICE_NAME,
        "device_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "pci_bus_id": "00000000:00:00.0",
    }
    return runner.SourceFreeReview(
        **clean,
        receipt_sha256=runner.sha256(runner.canonical_json(clean)),
    )


class FakeCandidate:
    def __init__(self, ordinal: int) -> None:
        self.selector_ordinal = ordinal

    def as_dict(self):
        return {
            "topology": "suffix",
            "topology_id": 0,
            "states": 2,
            "reset_length": 32,
            "selector_ordinal": self.selector_ordinal,
        }


class FakeCommon:
    @staticmethod
    def candidate_bank():
        return tuple(FakeCandidate(index) for index in range(150))


def exact_projection():
    return {
        "primary_fold_policy": "disjoint_stream_owner_dependence_components",
        "primary_exact_identity_estimable": True,
        "disjoint_dependence_component_count": 3,
        "exact_cell_symbol_updates": runner.PINNED_PRIMARY_CELL_SYMBOL_UPDATES,
        "maximum_source_survivor_updates_including_four_shuffles": runner.PINNED_DEFERRED_MAXIMUM_UPDATES,
        "coordinate_disjoint_diagnostic_cell_symbol_updates": runner.PINNED_DEFERRED_COORDINATE_UPDATES,
        "coordinate_disjoint_diagnostic_estimable_folds": 6,
        "passes_pre_fit_resource_budget": True,
        "passes_pre_fit_runtime_budget": False,
        "static_resource_admission": {
            "streams": runner.PINNED_PANEL_STREAMS,
            "symbols": runner.PINNED_PANEL_SYMBOLS,
            "passes": True,
        },
        "folds": [
            {
                "component_ordinal": component,
                "identity_indices": list(identities),
                "cell_symbol_updates": updates,
                "estimable": True,
            }
            for component, identities, updates in runner.PINNED_FOLD_UPDATES
        ],
    }


class FakeProjectionStage:
    @staticmethod
    def projected_updates(common, protocol, panel):
        if protocol != "exact-protocol" or panel != {"panel": "exact"}:
            raise AssertionError("projection arguments changed")
        return exact_projection()


class PrimaryGateSourceTests(unittest.TestCase):
    def test_import_is_inert_and_numerical_imports_are_inside_run(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        numerical = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name in {"numpy", "cupy"} for alias in node.names)
        ]
        self.assertEqual(len(numerical), 2)
        run_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        run_descendants = set(ast.walk(run_node))
        self.assertTrue(all(node in run_descendants for node in numerical))
        allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef, ast.If)
        for ordinal, node in enumerate(tree.body):
            if ordinal == 0 and isinstance(node, ast.Expr):
                self.assertIsInstance(node.value, ast.Constant)
            else:
                self.assertIsInstance(node, allowed)
        self.assertNotIn("glob(", source)
        self.assertNotIn("rglob(", source)
        self.assertNotIn("os.walk", source)

    def test_wrong_authorization_fails_before_any_source_review_or_path_access(self) -> None:
        arguments = argparse.Namespace(
            authorization="wrong",
            v9_package="/abs/missing-v9",
            pinned_support="/abs/missing-support",
            v8_package="/abs/missing-v8",
            strata_common="/abs/missing-common",
            frozen_auditor="/abs/missing-auditor",
            artifact="/abs/missing-qwen",
            output_dir="/abs/missing-output",
        )
        with mock.patch.object(runner, "authenticate_v9_package") as authenticate:
            with self.assertRaisesRegex(runner.PrimaryGateError, "authorization"):
                runner.run(arguments)
        authenticate.assert_not_called()

    def test_review_capability_tamper_rejects_before_artifact_accessor(self) -> None:
        calls = []

        class FakeSupport:
            @staticmethod
            def HeldRegularInput(*args, **kwargs):
                calls.append((args, kwargs))
                raise AssertionError("artifact accessor must remain unreachable")

        bad = dataclasses.replace(make_review(), receipt_sha256="00" * 32)
        with self.assertRaisesRegex(runner.PrimaryGateError, "capability seal"):
            runner.held_qwen_artifact_after_review(FakeSupport(), bad, Path("/abs/qwen.bin"))
        self.assertEqual(calls, [])

    def test_review_throughput_upper_tamper_bound_rejects(self) -> None:
        high = make_review(runner.CONSERVATIVE_THROUGHPUT_MAX + 1.0)
        with self.assertRaisesRegex(runner.PrimaryGateError, "tamper bounds"):
            runner.verify_source_free_review(high)

    def test_primary_runtime_admission_uses_only_authenticated_primary_work(self) -> None:
        projection, admission = runner.primary_runtime_admission(
            FakeProjectionStage(), FakeCommon(), "exact-protocol", {"panel": "exact"}, make_review()
        )
        self.assertEqual(projection["exact_cell_symbol_updates"], 38_621_316_130)
        self.assertAlmostEqual(
            admission["projected_primary_gpu_kernel_work_seconds"],
            38_621_316_130 / 3_242_398.2106118356,
        )
        self.assertTrue(admission["passes"])
        self.assertFalse(admission["is_total_launch_wall_time_projection"])
        self.assertEqual(len(admission["unmodeled_wall_components"]), 4)
        self.assertFalse(admission["evaluation_runner_pins_are_decoder_identity_inputs"])
        deferred = admission["deferred_not_counted_or_executed"]
        self.assertEqual(
            deferred["four_survivor_shuffles_and_coordinate_diagnostic_maximum_updates"],
            286_625_070_746 - 38_621_316_130,
        )
        self.assertEqual(deferred["coordinate_diagnostic_updates"], 93_518_490_096)

    def test_any_fold_workload_weakening_is_fatal(self) -> None:
        weakened = exact_projection()
        weakened["folds"][1]["cell_symbol_updates"] -= 1

        class WeakStage:
            @staticmethod
            def projected_updates(*args):
                return weakened

        with self.assertRaisesRegex(runner.PrimaryGateError, "fold workload"):
            runner.primary_runtime_admission(
                WeakStage(), FakeCommon(), object(), object(), make_review()
            )

    def test_candidate_bank_filtering_or_reordering_is_fatal(self) -> None:
        class FilteredCommon:
            @staticmethod
            def candidate_bank():
                return tuple(FakeCandidate(index) for index in range(149))

        with self.assertRaisesRegex(runner.PrimaryGateError, "candidate bank"):
            runner.primary_runtime_admission(
                FakeProjectionStage(), FilteredCommon(), "exact-protocol", {"panel": "exact"}, make_review()
            )

    def test_exact_primary_calls_only_nested_exact_then_final_container(self) -> None:
        events = []
        bank = FakeCommon.candidate_bank()

        class Stage:
            @staticmethod
            def prepare_backend_cache(backend, panel):
                events.append(("prepare", backend, panel))
                return "cache"

            @staticmethod
            def nested_holdout(common, protocol, codec, backend, cache, panel, **kwargs):
                events.append(("nested", kwargs))
                return {
                    "estimable": True,
                    "primary_policy": "exact_identity",
                    "final_topology_selected_from_nested_fold_votes": bank[17].as_dict(),
                }

            @staticmethod
            def final_container(*args):
                events.append(("final", args[7].selector_ordinal))
                return {"literal": "container"}

        modules = {
            "stage": Stage(),
            "common": FakeCommon(),
            "protocol": object(),
            "codec": object(),
            "semantic": object(),
        }
        scientific, physical = runner.execute_exact_primary(
            modules, "backend", "adapter", "panel", "score", "bindings", "descriptor"
        )
        self.assertEqual(scientific["primary_policy"], "exact_identity")
        self.assertEqual(physical, {"literal": "container"})
        self.assertEqual(events[1], ("nested", {"policy": "exact_identity", "diagnostic_only": False}))
        self.assertEqual(events[2], ("final", 17))

    def test_forbidden_survivor_and_control_entrypoints_are_unreachable(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        forbidden = {
            "source_phase",
            "survivor_shuffle_diagnostics",
            "coordinate_disjoint_diagnostic",
            "controls_phase",
            "within_context_permutation",
            "multiscale_chunk_shuffle",
        }
        self.assertTrue(forbidden.isdisjoint(called_attributes | called_names))
        options = {option for action in runner.parser()._actions for option in action.option_strings}
        self.assertFalse(any("control" in option or "shuffle" in option for option in options))

    def test_survivor_status_is_still_explicitly_nonpromoting(self) -> None:
        physical = {
            "parsed_metrics": {
                "passes_rate_interval": True,
                "passes_F_target": True,
                "passes_cold_read_below_2x": True,
            },
            "standalone_decode": {"all_payloads_canonically_reencoded": True},
            "identical_reconstruction_proved_by_full_f64_digest": True,
            "all_adapted_values_deserialized_from_transmitted_model": True,
        }
        status = runner.primary_status({"passes_heldout_gate": True}, physical)
        self.assertEqual(status, "PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED")
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"positive_claim_even_if_all_primary_gates_pass": False', source)
        self.assertIn('"controls_may_be_opened_or_inferred_from_this_result": False', source)

    def test_pinned_support_preserves_list_of_builtin_int_rows_and_one_decode_cache(self) -> None:
        support, record = runner.load_authenticated_support(SUPPORT.absolute())
        self.assertEqual(record["semantic_bridge_abi"], "list[list[int]]")

        class NumpyLikeInt(int):
            pass

        class FakeStrata:
            @staticmethod
            def expected_block_group_ordinals(labels):
                return [[NumpyLikeInt(7), NumpyLikeInt(2)], [NumpyLikeInt(11)]]

        receipt = support.normalize_strata_group_ordinal_abi(FakeStrata)
        rows = FakeStrata.expected_block_group_ordinals(object())
        self.assertEqual(rows, [[7, 2], [11]])
        self.assertTrue(all(type(row) is list for row in rows))
        self.assertTrue(all(type(value) is int for row in rows for value in row))
        self.assertFalse(receipt["positive_claim_authority"])

        class Delegate:
            calls = 0

            def extract_from_current(self, raw):
                self.calls += 1
                return {"raw": raw}

        delegate = Delegate()
        cache = support.SingleArtifactPanelCache(delegate)
        first = cache.extract_from_current(b"synthetic-only")
        second = cache.extract_from_current(b"synthetic-only")
        self.assertIs(first, second)
        self.assertEqual(delegate.calls, 1)
        self.assertTrue(cache.receipt()["same_panel_object_reused"])

    def test_source_manifest_and_pinned_support_hash_authenticate(self) -> None:
        closure = runner.authenticate_v9_package(PACKAGE.absolute())
        self.assertEqual(tuple(closure["member_hashes"]), runner.V9_REQUIRED_MEMBERS)
        self.assertEqual(
            hashlib.sha256(SUPPORT.read_bytes()).hexdigest(),
            runner.PINNED_SUPPORT_SHA256,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
