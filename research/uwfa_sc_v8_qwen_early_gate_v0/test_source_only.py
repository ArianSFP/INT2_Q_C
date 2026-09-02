#!/usr/bin/env python3
"""Source-only tests; never discover or open the Qwen artifact."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import ast
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).absolute().parent
REPOSITORY = PACKAGE.parents[1]
V8 = REPOSITORY / "research" / "unifilar_wfa_entropy_census_stage0_v8"
RUNNER = PACKAGE / "early_gate.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("uwfa_sc_v8_qwen_early_gate_source_test", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("runner loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class FakeCommon:
    @staticmethod
    def pretty_json(value):
        return runner.pretty_json(value)


class EarlyGateSourceTests(unittest.TestCase):
    def test_constants_are_exact_audited_inputs(self) -> None:
        self.assertEqual(runner.CURRENT_ARTIFACT_BYTES, 8_847_360)
        self.assertEqual(runner.SOURCE_WEIGHTS, 28_311_552)
        self.assertEqual(
            runner.CURRENT_ARTIFACT_SHA256,
            "4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b",
        )
        expected = runner.AUDITED_SSE_FP64 / runner.AUDITED_SOURCE_ENERGY_FP64
        self.assertLessEqual(abs(expected - runner.AUDITED_RELATIVE_MSE), 4.0 * runner.math.ulp(expected))

    def test_import_is_inert(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        numerical_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name in {"numpy", "cupy"} for alias in node.names)
        ]
        self.assertEqual(len(numerical_imports), 2)
        run_function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        self.assertTrue(all(node in ast.walk(run_function) for node in numerical_imports))
        allowed_top_level = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef, ast.If)
        for ordinal, node in enumerate(tree.body):
            if ordinal == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                self.assertIsInstance(node.value.value, str)
            else:
                self.assertIsInstance(node, allowed_top_level)
        self.assertNotIn("glob(", source)
        self.assertNotIn("rglob(", source)
        self.assertNotIn("os.walk", source)

    def test_sealed_v8_manifest_authenticates_without_numeric_imports(self) -> None:
        closure = runner.authenticate_v8_package(V8.absolute())
        self.assertEqual(closure["manifest_sha256"], runner.SEALED_V8_MANIFEST_SHA256)
        self.assertEqual(tuple(closure["member_hashes"]), runner.V8_REQUIRED_MEMBERS)
        self.assertEqual(len(closure["source_snapshot_root_sha256"]), 64)

    def test_exact_v8_snapshots_load_with_required_entrypoints(self) -> None:
        closure = runner.authenticate_v8_package(V8.absolute())
        modules = runner.load_v8_modules(closure)
        required = (
            (modules["stage"], "gpu_preflight_all_150"),
            (modules["stage"], "representative_outer_fold_benchmark"),
            (modules["stage"], "nested_holdout"),
            (modules["stage"], "final_container"),
            (modules["stage"], "source_phase"),
            (modules["codec"], "physical_metrics"),
            (modules["backend_source"], "build_backend"),
            (modules["adapter_source"], "StrataSCAdapter"),
        )
        for module, name in required:
            self.assertTrue(callable(getattr(module, name)))

    def test_wrong_authorization_fails_before_any_source_or_payload_access(self) -> None:
        arguments = runner.argparse.Namespace(
            authorization="wrong",
            v8_package="/does/not/exist/source",
            strata_common="/does/not/exist/common.py",
            frozen_auditor="/does/not/exist/auditor.py",
            artifact="/does/not/exist/qwen.bin",
            output_dir="/does/not/exist/result",
        )
        with mock.patch.object(runner, "authenticate_v8_package") as authenticate:
            with self.assertRaises(runner.EarlyGateError):
                runner.run(arguments)
        authenticate.assert_not_called()

    def test_score_receipt_is_protocol_compatible_and_self_sealed(self) -> None:
        receipt, encoded = runner.build_score_receipt(
            FakeCommon(),
            artifact_sha256="01" * 32,
            reconstruction_sha256="02" * 32,
            full_geometry_sha256="03" * 32,
            decoder_bundle_sha256="04" * 32,
        )
        clean = dict(receipt)
        claimed = clean.pop("score_receipt_sha256")
        self.assertEqual(claimed, hashlib.sha256(runner.canonical_json(clean)).hexdigest())
        self.assertEqual(json.loads(encoded), receipt)
        self.assertEqual(receipt["normalization"], "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM")

    def test_bandwidth_ratios_separate_pages_repetition_and_coalescing(self) -> None:
        def fr(value: Fraction):
            return runner.fraction_record(value)

        metrics = {
            "experts": [
                {
                    "expert_ordinal": 0,
                    "attributable_total_physical_bytes": fr(Fraction(100, 1)),
                    "attributable_nonpadding_decodable_bytes": fr(Fraction(80, 1)),
                    "touched_page_bytes": 120,
                    "instrumented_routed_requested_bytes_with_repetition": 112,
                    "instrumented_routed_unique_requested_bytes": 96,
                    "instrumented_routed_overlap_bytes_requested_again": 16,
                    "instrumented_routed_read_request_count": 7,
                    "causal_decode_reencode_reconstruction": {"all_payloads_canonically_reencoded": True},
                }
            ]
        }
        result = runner.bandwidth_summary(metrics)
        row = result["experts"][0]
        self.assertEqual(row["descriptor_backed_unique_page_ratio_strict"]["exact"], "3/2")
        self.assertEqual(row["requested_with_repetition_ratio_strict"]["exact"], "7/5")
        self.assertEqual(row["ideal_coalesced_unique_requested_ratio_strict"]["exact"], "6/5")
        self.assertTrue(result["passes_frozen_unique_page_below_2x"])
        self.assertTrue(result["passes_strict_requested_with_repetition_below_2x"])
        self.assertTrue(result["passes_strict_ideal_coalesced_unique_requested_below_2x"])
        self.assertTrue(result["passes_all_reported_bandwidth_ratios_below_2x"])

    def test_descriptor_builder_owns_backing_file_after_return(self) -> None:
        closure = runner.authenticate_v8_package(V8.absolute())
        codec = runner.load_snapshot_module(
            "uwfa_sc_v8_eg_codec_descriptor_test",
            closure["snapshots"]["container_codec.py"],
            closure["member_hashes"]["container_codec.py"],
        )
        raw = bytes((ordinal * 29 + 7) & 0xFF for ordinal in range(codec.HEADER_BYTES))
        source = runner.descriptor_source_builder(codec)(raw)
        try:
            reader = source.fresh_reader()
            try:
                self.assertEqual(reader.read(0, len(raw)), raw)
                reader.verify_stable()
            finally:
                reader.close()
            source.verify_stable()
        finally:
            source.close()

    def test_group_ordinal_abi_normalization_is_value_and_order_preserving(self) -> None:
        class ForeignInteger(int):
            pass

        module = type("FakeStrataCommon", (), {})()
        module.expected_block_group_ordinals = lambda labels: [
            (ForeignInteger(7), ForeignInteger(2)),
            (ForeignInteger(11),),
        ]
        receipt = runner.normalize_strata_group_ordinal_abi(module)
        rows = module.expected_block_group_ordinals(None)
        self.assertEqual(rows, [(7, 2), (11,)])
        self.assertTrue(all(type(value) is int for row in rows for value in row))
        self.assertEqual(
            receipt["status"],
            "EXPLORATORY_VALUE_PRESERVING_NUMPY_INTEGER_TO_PYTHON_INT",
        )
        clean = dict(receipt)
        claimed = clean.pop("receipt_sha256")
        self.assertEqual(claimed, hashlib.sha256(runner.canonical_json(clean)).hexdigest())

    def test_single_artifact_panel_cache_reuses_exact_object_and_rejects_substitution(self) -> None:
        class Delegate:
            def __init__(self):
                self.calls = 0
                self.marker = "delegated"

            def extract_from_current(self, raw):
                self.calls += 1
                return {"raw": raw, "call": self.calls}

        delegate = Delegate()
        cache = runner.SingleArtifactPanelCache(delegate)
        first = cache.extract_from_current(b"same-artifact")
        second = cache.extract_from_current(b"same-artifact")
        self.assertIs(first, second)
        self.assertEqual(delegate.calls, 1)
        self.assertEqual(cache.marker, "delegated")
        receipt = cache.receipt()
        self.assertEqual(receipt["extract_calls"], 2)
        self.assertEqual(receipt["delegate_extract_calls"], 1)
        self.assertTrue(receipt["same_panel_object_reused"])
        with self.assertRaisesRegex(runner.EarlyGateError, "digest identity"):
            cache.extract_from_current(b"evil-artifact")
        self.assertFalse(cache.receipt()["same_panel_object_reused"])

    def test_publication_is_exclusive_and_complete_last(self) -> None:
        if os.name != "posix":
            self.skipTest("descriptor-relative publication test is POSIX")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            first = runner.publish(
                output,
                {"RESULT.json": b"{}\n", "UWFCV8.bin": b"payload"},
                source_root="ab" * 32,
                status="EARLY_DIAGNOSTIC_TEST",
            )
            self.assertTrue((output / "COMPLETE.json").is_file())
            self.assertEqual([row["name"] for row in first["members"]], ["RESULT.json", "UWFCV8.bin", "COMPLETE.json"])
            with self.assertRaises(FileExistsError):
                runner.publish(
                    output,
                    {"RESULT.json": b"{}\n"},
                    source_root="ab" * 32,
                    status="EARLY_DIAGNOSTIC_TEST",
                )

    def test_claim_boundary_is_literal(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for token in (
            '"positive_claim_authority": False',
            '"controls_run": False',
            '"controls_may_not_be_inferred_or_added": True',
            "EARLY_DIAGNOSTIC_SOURCE_SURVIVOR_REQUIRES_CONTROLS_AND_INDEPENDENT_AUDIT",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
