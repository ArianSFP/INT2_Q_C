#!/usr/bin/env python3
"""Hostile standard-library tests for finite TACTIC-DH384 v3 sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def load(name: str, filename: str):
    source = (ROOT / filename).read_bytes()
    module = types.ModuleType(name)
    module.__file__ = f"<source-test:{filename}:{hashlib.sha256(source).hexdigest()}>"
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True,
                 optimize=0), module.__dict__)
    return module


SPEC = load("tactic_dh384_v3_test_spec", "format_spec.py")
EXTERNAL = load("tactic_dh384_v3_test_external", "external_contract.py")


class FiniteFormatTests(unittest.TestCase):
    def test_01_exact_single_expert_rate(self):
        self.assertEqual(str(SPEC.COARSE_BPW), "307/128")
        self.assertEqual(str(SPEC.FINE_BPW), "3/32")
        self.assertEqual(str(SPEC.HEADER_BPW), "1/128")
        self.assertEqual(str(SPEC.COMPOSITE_BPW), "5/2")
        self.assertEqual(SPEC.COMPOSITE_BYTES, 1_474_560)
        self.assertEqual(SPEC.COMPOSITE_BYTES // 4096, 360)

    def test_02_record_uses_all_384_bits(self):
        self.assertEqual(SPEC.FINE_RECORD_BYTES, 48)
        self.assertEqual(8 + SPEC.ACTIVE_RANK, 384)
        signs = tuple((index % 3) == 0 for index in range(SPEC.ACTIVE_RANK))
        record = SPEC.pack_record(73, signs)
        self.assertEqual(len(record), 48)
        self.assertEqual(SPEC.unpack_record(record), (73, signs))

    def test_03_zero_scale_is_canonical_only_once(self):
        zero = SPEC.pack_record(0, (False,) * SPEC.ACTIVE_RANK)
        self.assertEqual(zero, bytes(48))
        with self.assertRaises(SPEC.FormatError):
            SPEC.pack_record(0, (True,) + (False,) * (SPEC.ACTIVE_RANK - 1))
        with self.assertRaises(SPEC.FormatError):
            SPEC.unpack_record(b"\x00\x01" + bytes(46))

    def test_04_scale_law_is_frozen_dyadic(self):
        self.assertEqual(SPEC.SCALE_DENOMINATOR, 1 << 18)
        self.assertEqual(SPEC.scale_alpha(0, 4.0), 0.0)
        self.assertEqual(SPEC.scale_alpha(128, 4.0), 0.25)
        self.assertEqual(SPEC.scale_alpha(255, 0.0), 0.0)
        with self.assertRaises(SPEC.FormatError):
            SPEC.scale_alpha(256, 1.0)

    def test_05_audited_selector_identity(self):
        packet = SPEC.universal_selector_packet()
        self.assertEqual(len(packet), 16_384)
        self.assertEqual(hashlib.sha256(packet).hexdigest(),
                         SPEC.SELECTOR_PACKET_SHA256)
        self.assertEqual(SPEC.UNIVERSAL_SELECTOR_ORDINAL, 17)

    def test_06_cpu_transform_is_orthogonal(self):
        symbols = [((index * 29 + 7) % 41) - 20
                   for index in range(SPEC.BLOCK_VALUES)]
        values = [math.sin(index * 0.017) + 0.01 * (index % 11)
                  for index in range(SPEC.BLOCK_VALUES)]
        schedule = SPEC.conditional_schedule(symbols, 2)
        coefficients = SPEC.analysis_cpu(values, schedule)
        rebuilt = SPEC.synthesis_cpu(coefficients, schedule)
        self.assertTrue(math.isclose(
            math.fsum(value * value for value in values),
            math.fsum(value * value for value in coefficients),
            rel_tol=3e-13, abs_tol=3e-11))
        self.assertLess(max(abs(a - b) for a, b in zip(values, rebuilt)),
                        2e-12)

    def test_07_finite_correction_is_inside_parent_span(self):
        symbols = [((index * 13) % 31) - 15
                   for index in range(SPEC.BLOCK_VALUES)]
        coarse = [0.125 + (index % 17) / 1024.0
                  for index in range(SPEC.BLOCK_VALUES)]
        signs = tuple((index & 1) == 0 for index in range(SPEC.ACTIVE_RANK))
        record = SPEC.pack_record(91, signs)
        correction = SPEC.correction_cpu(record, symbols, coarse, 0)
        schedule = SPEC.conditional_schedule(symbols, 0)
        recovered = SPEC.analysis_cpu(correction, schedule)
        self.assertLess(max(abs(value) for value in recovered[376:]), 2e-12)
        self.assertEqual(SPEC.ACTIVE_RANK, 376)
        self.assertLess(SPEC.ACTIVE_RANK, SPEC.AUDITED_PARENT_RANK)

    def test_08_record_selector_never_worsens_fixed_span(self):
        residual = [0.01 * math.sin(index * 0.11)
                    for index in range(SPEC.BLOCK_VALUES)]
        coarse = [0.25 + 0.01 * math.cos(index * 0.03)
                  for index in range(SPEC.BLOCK_VALUES)]
        record = SPEC.select_record_cpu(residual, coarse)
        code, signs = SPEC.unpack_record(record)
        alpha = SPEC.scale_alpha(code, max(abs(value) for value in coarse))
        before = math.fsum(value * value for value in residual[:376])
        after = math.fsum(
            (value - (alpha if positive else -alpha)) ** 2
            for value, positive in zip(residual[:376], signs))
        self.assertLessEqual(after, before + 1e-18)

    def test_09_header_and_composite_roundtrip(self):
        digest = "1" * 64
        bindings = {
            "coarse_sha256": hashlib.sha256(bytes(SPEC.COARSE_BYTES)).hexdigest(),
            "fine_sha256": hashlib.sha256(bytes(SPEC.FINE_BYTES)).hexdigest(),
            "input_manifest_sha256": digest,
            "v6_complete_sha256": "2" * 64,
            "producer_source_manifest_sha256": "3" * 64,
            "producer_source_root_sha256": "4" * 64,
        }
        header = SPEC.make_header(bindings)
        self.assertEqual(len(header), SPEC.PILOT_HEADER_BYTES)
        self.assertEqual(SPEC.parse_header(header)["bindings"], bindings)
        composite = header + bytes(SPEC.COARSE_BYTES) + bytes(SPEC.FINE_BYTES)
        parsed, coarse, fine = SPEC.split_composite(composite)
        self.assertEqual(parsed["bindings"], bindings)
        self.assertEqual(len(coarse), SPEC.COARSE_BYTES)
        self.assertEqual(len(fine), SPEC.FINE_BYTES)

    def test_10_header_padding_tamper_fails(self):
        bindings = {
            "coarse_sha256": "0" * 64,
            "fine_sha256": "1" * 64,
            "input_manifest_sha256": "2" * 64,
            "v6_complete_sha256": "3" * 64,
            "producer_source_manifest_sha256": "4" * 64,
            "producer_source_root_sha256": "5" * 64,
        }
        header = bytearray(SPEC.make_header(bindings))
        header[-1] = 1
        with self.assertRaises(SPEC.FormatError):
            SPEC.parse_header(bytes(header))

    def test_11_traffic_does_not_infer_six_expert_packet(self):
        traffic = SPEC.single_expert_traffic(0, 1)
        self.assertEqual(traffic["literal_pilot"]["total_read_bytes"],
                         SPEC.COMPOSITE_BYTES)
        self.assertEqual(traffic["literal_pilot"]["unique_pages"], 360)
        self.assertEqual(traffic["literal_pilot"][
            "amplification_over_literal_pilot_bytes"], 1.0)
        six = traffic["six_expert_amortized_tactic_layout"]
        self.assertFalse(six["emitted"])
        self.assertIsNone(six["global_packet_bytes"])
        self.assertFalse(six["seventy_three_over_seventy_two_claim"])

    def test_12_repeated_read_is_separate(self):
        traffic = SPEC.single_expert_traffic(31, 2)["literal_pilot"]
        self.assertEqual(traffic["first_pass_bytes"], SPEC.COMPOSITE_BYTES)
        self.assertEqual(traffic["reread_bytes"], SPEC.COMPOSITE_BYTES)
        self.assertEqual(traffic["total_read_bytes"],
                         2 * SPEC.COMPOSITE_BYTES)


class ReviewAndSourceTests(unittest.TestCase):
    def _review(self):
        record = {
            "schema": EXTERNAL.REVIEW_SCHEMA,
            "status": EXTERNAL.REVIEW_STATUS,
            "package_manifest_sha256": "1" * 64,
            "package_source_root_sha256": "2" * 64,
            "v6_source_manifest_sha256": "3" * 64,
            "v6_source_root_sha256": "4" * 64,
            "v6_complete_sha256": "5" * 64,
            "input_manifest_sha256": "6" * 64,
            "allowed_scope": {
                "experts": 1,
                "geometry": [768, 2048],
                "qwen_or_model_identity_available_to_codec": False,
                "universal_tail_claim": False,
            },
            "independent_audit": {
                "finite_source_reviewed": True,
                "v6_completed_result_reviewed": True,
                "payload_launch_explicitly_authorized": True,
            },
        }
        record["review_claim_sha256"] = hashlib.sha256(
            EXTERNAL.canonical_json(record)).hexdigest()
        return record

    def test_13_review_exact_binding_passes(self):
        record = self._review()
        parsed = EXTERNAL.validate_launch_review(
            EXTERNAL.canonical_json(record),
            package_manifest_sha256="1" * 64,
            package_source_root_sha256="2" * 64,
            v6_source_manifest_sha256="3" * 64,
            v6_source_root_sha256="4" * 64)
        self.assertEqual(parsed, record)

    def test_14_review_tamper_fails(self):
        record = self._review()
        record["allowed_scope"]["experts"] = 2
        with self.assertRaises(EXTERNAL.ExternalError):
            EXTERNAL.validate_launch_review(
                EXTERNAL.canonical_json(record),
                package_manifest_sha256="1" * 64,
                package_source_root_sha256="2" * 64,
                v6_source_manifest_sha256="3" * 64,
                v6_source_root_sha256="4" * 64)

    def test_15_strict_json_duplicate_fails(self):
        with self.assertRaises(EXTERNAL.ExternalError):
            EXTERNAL.strict_json(b'{"x":1,"x":2}', "duplicate")

    def test_16_v6_lock_matches_live_source_only_tree(self):
        lock = json.loads((ROOT / "V6_LOCK.json").read_text("utf-8"))
        v6 = REPO / lock["relative_directory"]
        manifest = v6 / "SOURCE_MANIFEST.json"
        self.assertEqual(manifest.stat().st_size,
                         lock["source_manifest"]["bytes"])
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         lock["source_manifest"]["sha256"])
        for row in lock["members"]:
            payload = (v6 / row["name"]).read_bytes()
            self.assertEqual(len(payload), row["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])

    def test_17_v2_and_v6_are_not_modified_by_package(self):
        design = json.loads((ROOT / "design_lock.json").read_text("utf-8"))
        boundary = design["predecessor_boundary"]
        self.assertFalse(boundary[
            "tactic_conditional_dyadic_coset_v2_modified"])
        self.assertFalse(boundary["tactic_actual_coarse_n18_v6_modified"])

    def test_18_hard_gate_and_containment_are_executable(self):
        dispatcher = (ROOT / "dispatcher.py").read_text("utf-8")
        self.assertIn("exact_nested_required_error_capture", dispatcher)
        self.assertIn("HARD_REJECT_PARENT_RANK384", dispatcher)
        self.assertIn("HARD_REJECT_ACTIVE_RANK376", dispatcher)
        self.assertIn("384.0 / 4096.0", dispatcher)
        self.assertIn("(2.0 / math.pi)", dispatcher)

    def test_19_review_precedes_result_and_input_open(self):
        dispatcher = (ROOT / "dispatcher.py").read_text("utf-8")
        review = dispatcher.index("validate_launch_review")
        runtime = dispatcher.index("v6_package.load_runtime", review)
        result = dispatcher.index("HeldCompletedV6Result", runtime)
        inputs = dispatcher.index("authenticate_inputs", result)
        self.assertLess(review, runtime)
        self.assertLess(runtime, result)
        self.assertLess(result, inputs)

    def test_20_isolated_and_gpu_guard_precede_review(self):
        dispatcher = (ROOT / "dispatcher.py").read_text("utf-8")
        isolated = dispatcher.index("sys.flags.isolated")
        cuda = dispatcher.index('CUDA_VISIBLE_DEVICES") == "0"')
        review = dispatcher.index("validate_launch_review")
        self.assertLess(isolated, review)
        self.assertLess(cuda, review)

    def test_21_encoder_selects_fine_labels_but_not_coarse(self):
        encoder = (ROOT / "finite_encoder.py").read_text("utf-8")
        self.assertIn("cp.argmin(deltas", encoder)
        self.assertIn("forced_local_scale_codes", encoder)
        self.assertIn("local_joint_mismatches == 0", encoder)
        design = json.loads((ROOT / "design_lock.json").read_text("utf-8"))
        self.assertFalse(design["encoder_objective"][
            "coarse_codeword_reoptimized"])
        self.assertTrue(design["encoder_objective"][
            "fine_labels_source_selected"])

    def test_22_independent_decoder_does_not_import_encoder(self):
        source = (ROOT / "independent_decoder.py").read_text("utf-8")
        self.assertNotIn("finite_encoder", source)
        self.assertIn("fine_records_independently_decode_reencode", source)
        self.assertIn("correction dyadic-span containment", source)

    def test_23_atomic_publication_is_terminal(self):
        source = (ROOT / "atomic_publish.py").read_text("utf-8")
        self.assertIn("RENAME_NOREPLACE", source)
        self.assertIn('pending_name = ".COMPLETE.pending"', source)
        self.assertLess(source.index("_rehash(final_fd, pending_row"),
                        source.index('pending_name, "COMPLETE.json"'))

    def test_24_no_payload_result_or_gpu_in_source_attestation(self):
        manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text("utf-8"))
        self.assertEqual(manifest["access_attestation"], {
            "cuda_or_cupy_initialized_during_source_build": False,
            "network_accessed": False,
            "qwen_or_model_payload_accessed": False,
            "runpod_accessed": False,
            "v6_live_result_accessed": False,
        })

    def test_25_exact_source_manifest_closure(self):
        manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text("utf-8"))
        rows = manifest["members"]
        self.assertEqual([row["name"] for row in rows],
                         sorted((row["name"] for row in rows),
                                key=lambda value: value.encode("utf-8")))
        self.assertEqual({path.name for path in ROOT.iterdir()},
                         {row["name"] for row in rows} |
                         {"SOURCE_MANIFEST.json"})
        observed = []
        for row in rows:
            payload = (ROOT / row["name"]).read_bytes()
            self.assertEqual(len(payload), row["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])
            observed.append(row)
        self.assertEqual(
            hashlib.sha256(SPEC.canonical_json(observed)).hexdigest(),
            manifest["source_root_sha256"])


if __name__ == "__main__":
    if sys.flags.isolated != 1 or not sys.dont_write_bytecode:
        raise SystemExit("invoke with CPython -I -B")
    unittest.main(verbosity=2)
