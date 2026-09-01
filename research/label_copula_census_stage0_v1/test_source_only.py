#!/usr/bin/env python3
"""Hostile source-only tests: no model payload, network, CuPy, or CUDA."""

from __future__ import annotations

import math
import os
import inspect
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from label_copula_common import (
    AUTHORIZATION,
    CONTEXT_COUNT,
    CONTROL_SEEDS,
    MIN_TEST_LAYERS,
    MIN_TOTAL_LAYERS,
    Q_TOTAL,
    RESET_SYMBOLS,
    STANDALONE_REQUIRED_SAVING_BPW,
    Candidate,
    CompletionLastOutput,
    ContractError,
    HeldRegularFile,
    QuantizedModel,
    RawSwiGLUExpert,
    SymbolStream,
    candidate_bank,
    build_matched_gaussian_control_panel,
    canonical_raw_lloyd4_stream,
    container_ledger,
    decode_stream,
    encode_stream,
    evaluate_independent_matched_controls,
    evaluate_raw_source_panel,
    factorized_bank,
    fit_model,
    gray_bits,
    lloyd4_label,
    matched_control_gate,
    nested_partition,
    next_state,
    public_context,
    raw_panel_provenance,
    reject_symlink_path_and_ancestors,
    sha256_bytes,
    strict_json_loads,
    synthetic_parity_streams,
)


PACKAGE = Path(__file__).resolve().parent


class QuantizerAndTopologyTests(unittest.TestCase):
    def test_standalone_threshold(self) -> None:
        expected = -0.5 * math.log2(0.8) - 0.008074080480766676
        self.assertAlmostEqual(STANDALONE_REQUIRED_SAVING_BPW, expected, places=15)

    def test_gray_order_and_zero_block(self) -> None:
        self.assertEqual([gray_bits(index) for index in range(4)], [(0, 0), (0, 1), (1, 1), (1, 0)])
        self.assertEqual(lloyd4_label(0.0, 0.0), 2)
        self.assertEqual(lloyd4_label(-1.0, 1.0), 0)
        self.assertEqual(lloyd4_label(1.0, 1.0), 3)

    def test_arbitrary_shape_canonical_orientation(self) -> None:
        gate = [[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]
        up = [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
        down = [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]
        stream = canonical_raw_lloyd4_stream(
            gate, up, down, layer_group="L", expert_group="E"
        )
        self.assertEqual(stream.source_weights, 18)
        self.assertEqual(len(stream.symbols), 36)
        self.assertEqual(stream.roles[:12], (0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2))
        self.assertEqual(stream.planes[:8], (0, 1, 0, 1, 0, 1, 0, 1))
        with self.assertRaises(ContractError):
            canonical_raw_lloyd4_stream(gate, up, [[1.0]], layer_group="L", expert_group="E")

    def test_bank_exactly_240(self) -> None:
        bank = candidate_bank()
        self.assertEqual(len(bank), 240)
        self.assertEqual(len(set(bank)), 240)
        self.assertEqual(len(factorized_bank()), len(RESET_SYMBOLS))

    def test_all_updates_are_bounded_deterministic(self) -> None:
        for candidate in candidate_bank():
            for state in (0, candidate.chi - 1):
                for symbol in (0, 1):
                    first = next_state(candidate, state, symbol, 2, 1, candidate.reset - 3)
                    second = next_state(candidate, state, symbol, 2, 1, candidate.reset - 3)
                    self.assertEqual(first, second)
                    self.assertGreaterEqual(first, 0)
                    self.assertLess(first, candidate.chi)

    def test_parity_sketch_separates_suffix_collision(self) -> None:
        prefix_a = tuple([0] * 20 + [1, 0, 1, 1, 0, 1])
        prefix_b = tuple([1] + [0] * 19 + [1, 0, 1, 1, 0, 1])

        def trace(candidate: Candidate, row: tuple[int, ...]) -> int:
            state = 0
            for position, symbol in enumerate(row):
                state = next_state(candidate, state, symbol, 0, 0, position)
            return state

        self.assertEqual(trace(Candidate("suffix", 64, 32), prefix_a), trace(Candidate("suffix", 64, 32), prefix_b))
        self.assertNotEqual(trace(Candidate("parity_sketch", 64, 32), prefix_a), trace(Candidate("parity_sketch", 64, 32), prefix_b))

    def test_public_context_has_no_identity_argument(self) -> None:
        seen = {
            public_context(role, plane, position, 4096)
            for role in range(3)
            for plane in range(2)
            for position in range(4096)
        }
        self.assertTrue(seen)
        self.assertLess(max(seen), CONTEXT_COUNT)


class IntegerModelAndArithmeticTests(unittest.TestCase):
    @staticmethod
    def _tiny_stream(layer: str, expert: str, salt: int) -> SymbolStream:
        symbols = tuple(((index * 5 + salt) ^ (index >> 2)) & 1 for index in range(256))
        roles = tuple((index // 2) % 3 for index in range(256))
        planes = tuple(index & 1 for index in range(256))
        return SymbolStream(layer, expert, symbols, roles, planes, 128)

    def test_model_packet_exact_integer_roundtrip(self) -> None:
        streams = [self._tiny_stream("L0", f"E{index}", index) for index in range(3)]
        model = fit_model(streams, Candidate("rolling", 8, 64))
        packet = model.serialize()
        restored = QuantizedModel.deserialize(packet)
        self.assertEqual(restored, model)
        self.assertEqual(len(packet), 256 + 2 * CONTEXT_COUNT * 8)
        self.assertTrue(all(1 <= value < Q_TOTAL for value in model.freq1))

    def test_real_arithmetic_roundtrip_all_topology_families(self) -> None:
        training = [self._tiny_stream("L0", f"E{index}", index) for index in range(4)]
        target = self._tiny_stream("L1", "E9", 11)
        for topology in ("factorized", "suffix", "parity_sketch", "modular", "rolling", "regime"):
            with self.subTest(topology=topology):
                candidate = Candidate(topology, 1 if topology == "factorized" else 8, 64)
                model = fit_model(training, candidate)
                payload, meaningful = encode_stream(model, target)
                self.assertEqual(len(payload), (meaningful + 7) // 8)
                decoded = decode_stream(model, target.roles, target.planes, payload, meaningful)
                self.assertEqual(decoded, target.symbols)

    def test_container_charges_model_frames_padding_and_reads(self) -> None:
        streams = [self._tiny_stream("L0", f"E{index}", index) for index in range(4)]
        model = fit_model(streams, Candidate("suffix", 4, 32))
        encoded = tuple((stream, *encode_stream(model, stream)) for stream in streams)
        ledger = container_ledger(len(model.serialize()), encoded)
        self.assertEqual(ledger["container_header_bytes"], 4096)
        self.assertEqual(ledger["raw_directory_bytes"], 64 * len(streams))
        self.assertEqual(len(ledger["frame_rows"]), len(streams))
        self.assertEqual(ledger["total_physical_bytes"] % 4096, 0)
        self.assertGreater(ledger["maximum_cold_read_amplification"], 0.0)
        self.assertAlmostEqual(
            math.fsum(row["physical_denominator_bytes"] for row in ledger["frame_rows"]),
            ledger["total_physical_bytes"],
            places=9,
        )
        for row in ledger["frame_rows"]:
            expected = set(row["header_page_indices"])
            expected.update(row["model_page_indices"])
            expected.update(row["addressed_directory_page_indices"])
            expected.update(row["frame_page_indices"])
            self.assertEqual(set(row["cold_page_indices"]), expected)
            self.assertEqual(row["cold_read_bytes"], 4096 * len(expected))


class SplitAndControlTests(unittest.TestCase):
    @staticmethod
    def _raw_panel(*, perturb: float = 0.0) -> tuple[RawSwiGLUExpert, ...]:
        rows = []
        for layer in range(MIN_TOTAL_LAYERS):
            for slot in range(3):
                salt = float(10 * layer + slot) / 100.0
                rows.append(RawSwiGLUExpert(
                    layer_group=f"layer-{layer}",
                    expert_group=f"slot-{slot}",
                    gate=(((-2.0 + salt + (perturb if layer == 0 and slot == 0 else 0.0)), 0.5 + salt),),
                    up=((-0.5 - salt, 2.0 - salt),),
                    down=((1.5 + salt,), (-1.5 - salt,)),
                ))
        return tuple(rows)

    def test_whole_layer_and_whole_expert_isolation(self) -> None:
        rows = synthetic_parity_streams(layers=MIN_TOTAL_LAYERS, experts=5, blocks_per_stream=2, seed=7, constrained=True)
        folds = nested_partition(rows)
        train_layers = {row.layer_group for row in folds["train"]}
        test_layers = {row.layer_group for row in folds["test"]}
        train_experts = {row.expert_group for row in folds["train"]}
        validation_experts = {row.expert_group for row in folds["validation"]}
        self.assertFalse(train_layers & test_layers)
        self.assertFalse(train_experts & validation_experts)
        self.assertTrue(folds["train"] and folds["validation"] and folds["test"])
        self.assertGreaterEqual(len(test_layers), MIN_TEST_LAYERS)

    def test_degenerate_confidence_and_irregular_slot_panels_rejected(self) -> None:
        too_few = synthetic_parity_streams(layers=3, experts=3, blocks_per_stream=2, seed=5, constrained=True)
        with self.assertRaisesRegex(ContractError, "whole layers"):
            nested_partition(too_few)

        regular = list(synthetic_parity_streams(
            layers=MIN_TOTAL_LAYERS,
            experts=3,
            blocks_per_stream=1,
            seed=17,
            constrained=True,
        ))
        irregular = tuple(SymbolStream(
            row.layer_group,
            f"unique-{index}",
            row.symbols,
            row.roles,
            row.planes,
            row.source_weights,
        ) for index, row in enumerate(regular))
        with self.assertRaisesRegex(ContractError, "identical reusable expert-slot universe"):
            nested_partition(irregular)

        with self.assertRaisesRegex(ContractError, "identical reusable expert-slot universe"):
            nested_partition(tuple(regular[:-1]))

    def test_control_cannot_create_source_pass(self) -> None:
        miss = {
            "net_nonlocal_physical_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW - 1e-6,
            "absolute_source_survival_before_controls": False,
            "paired_whole_layer_uncertainty": {"lower_95_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW + 1.0},
        }
        self.assertFalse(matched_control_gate(miss))
        pass_row = {
            "net_nonlocal_physical_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW,
            "absolute_source_survival_before_controls": True,
            "paired_whole_layer_uncertainty": {"lower_95_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW},
        }
        self.assertTrue(matched_control_gate(pass_row))
        point_only = dict(pass_row)
        point_only["paired_whole_layer_uncertainty"] = {
            "lower_95_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW - 1e-9
        }
        self.assertFalse(matched_control_gate(point_only))
        self.assertEqual(len(CONTROL_SEEDS), 8)

    def test_gaussian_control_provenance_is_source_seed_and_pipeline_bound(self) -> None:
        raw = self._raw_panel()
        source = raw_panel_provenance(raw)
        first_panel, first = build_matched_gaussian_control_panel(raw, CONTROL_SEEDS[0])
        again_panel, again = build_matched_gaussian_control_panel(raw, CONTROL_SEEDS[0])
        second_panel, second = build_matched_gaussian_control_panel(raw, CONTROL_SEEDS[1])
        changed_panel, changed = build_matched_gaussian_control_panel(
            self._raw_panel(perturb=0.125), CONTROL_SEEDS[0]
        )
        self.assertEqual(first, again)
        self.assertEqual(first_panel, again_panel)
        self.assertEqual(first["source_tensor_sha256"], source["source_tensor_sha256"])
        self.assertEqual(first["source_geometry_sha256"], source["source_geometry_sha256"])
        self.assertEqual(first["source_block_moments_sha256"], source["source_block_moments_sha256"])
        self.assertNotEqual(first["generated_tensor_sha256"], second["generated_tensor_sha256"])
        self.assertNotEqual(first["source_tensor_sha256"], changed["source_tensor_sha256"])
        self.assertNotEqual(first["generated_label_panel_sha256"], "")
        self.assertEqual(len(first_panel), len(second_panel))
        self.assertEqual(len(first_panel), len(changed_panel))

    def test_scientific_control_evaluator_cannot_accept_prebuilt_panels(self) -> None:
        signature = inspect.signature(evaluate_independent_matched_controls)
        self.assertEqual(tuple(signature.parameters), ("source_result", "raw_panel"))
        source_text = inspect.getsource(evaluate_independent_matched_controls)
        self.assertIn("full_bank = candidate_bank()", source_text)
        self.assertIn("complete 240-cell", source_text)
        self.assertNotIn("control_panels", source_text)

    def test_raw_source_result_binds_complete_reference_pipeline(self) -> None:
        result = evaluate_raw_source_panel(
            self._raw_panel(),
            (Candidate("suffix", 2, 32),),
        )
        self.assertEqual(result["raw_source_provenance"]["experts"], MIN_TOTAL_LAYERS * 3)
        self.assertEqual(result["nonlocal_candidate_cells"], 1)
        self.assertEqual(len(result["complete_source_pipeline_sha256"]), 64)


class LifecycleTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
            strict_json_loads('{"x":1,"x":2}')
        for payload in ("NaN", "Infinity", "1e9999"):
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                strict_json_loads(payload)

    def test_held_regular_file_hash_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"source-only")
            with HeldRegularFile(target, 11, sha256_bytes(b"source-only")) as held:
                self.assertEqual(held.read_all(), b"source-only")
                held.verify_stable()
            link = root / "link"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises((ContractError, OSError)):
                with HeldRegularFile(link):
                    pass

    def test_symlinked_leaf_and_ancestor_rejected_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_parent = root / "real-parent"
            target_parent.mkdir()
            package = target_parent / "package"
            package.mkdir()
            leaf_link = root / "leaf-link"
            ancestor_link = root / "ancestor-link"
            try:
                leaf_link.symlink_to(package, target_is_directory=True)
                ancestor_link.symlink_to(target_parent, target_is_directory=True)
            except (OSError, NotImplementedError):
                return
            with self.assertRaisesRegex(ContractError, "symlink/reparse"):
                reject_symlink_path_and_ancestors(leaf_link)
            with self.assertRaisesRegex(ContractError, "symlink/reparse"):
                reject_symlink_path_and_ancestors(ancestor_link / "package")

    def test_completion_is_exclusive_and_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with CompletionLastOutput(output) as writer:
                writer.write_new("result.json", b"{}\n")
                self.assertNotIn("COMPLETE.json", {path.name for path in output.iterdir()})
                writer.complete("a" * 64)
                with self.assertRaisesRegex(ContractError, "not writable after completion"):
                    writer.write_new("AFTER_COMPLETE.json", b"{}\n")
                with self.assertRaisesRegex(ContractError, "irrevocable"):
                    writer.complete("b" * 64)
            self.assertIn("COMPLETE.json", {path.name for path in output.iterdir()})
            self.assertNotIn("AFTER_COMPLETE.json", {path.name for path in output.iterdir()})
            with self.assertRaises(FileExistsError):
                with CompletionLastOutput(output):
                    pass

    def test_wrong_token_rejects_before_output_and_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "must-not-exist"
            absent = root / "absent.json"
            command = [
                sys.executable,
                "-B",
                str(PACKAGE / "stage0_census.py"),
                "--authorization",
                "WRONG",
                "--review-receipt",
                str(absent),
                "--input-lock",
                str(absent),
                "--output",
                str(output),
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            run = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=20, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("CuPy not imported", run.stdout + run.stderr)
            self.assertFalse(output.exists())

    def test_valid_token_reserves_output_before_missing_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reserved"
            absent = root / "absent.json"
            command = [
                sys.executable,
                "-B",
                str(PACKAGE / "stage0_census.py"),
                "--authorization",
                AUTHORIZATION,
                "--review-receipt",
                str(absent),
                "--input-lock",
                str(absent),
                "--output",
                str(output),
            ]
            run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertTrue(output.is_dir())
            self.assertEqual({path.name for path in output.iterdir()}, {"RUN_STATE.json"})


class SealedPackageTests(unittest.TestCase):
    def test_native_verifier_and_tamper(self) -> None:
        command = [sys.executable, "-B", str(PACKAGE / "verify_source.py"), "--package", str(PACKAGE), "--compact"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        receipt = strict_json_loads(run.stdout)
        self.assertEqual(receipt["status"], "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY")

        from verify_source import verify_package

        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "package"
            shutil.copytree(PACKAGE, clone)
            target = clone / "README.md"
            target.write_bytes(target.read_bytes() + b"tamper")
            # verify_source intentionally reloads its same-directory common
            # module, so its ContractError has a distinct Python class identity.
            with self.assertRaisesRegex(Exception, "member (bytes|hash)"):
                verify_package(clone)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real"
            real_parent.mkdir()
            clone = real_parent / "package"
            shutil.copytree(PACKAGE, clone)
            leaf = root / "leaf"
            ancestor = root / "ancestor"
            try:
                leaf.symlink_to(clone, target_is_directory=True)
                ancestor.symlink_to(real_parent, target_is_directory=True)
            except (OSError, NotImplementedError):
                return
            with self.assertRaisesRegex(Exception, "symlink/reparse"):
                verify_package(leaf)
            with self.assertRaisesRegex(Exception, "symlink/reparse"):
                verify_package(ancestor / "package")


if __name__ == "__main__":
    unittest.main(verbosity=2)
