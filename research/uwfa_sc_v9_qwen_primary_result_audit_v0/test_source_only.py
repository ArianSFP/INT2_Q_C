#!/usr/bin/env python3
"""Hostile source-only tests for the v9 independent result auditor."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest


_AUDIT_PATH = Path(__file__).with_name("verify_result.py")
_AUDIT_SPEC = importlib.util.spec_from_file_location("verify_result", _AUDIT_PATH)
if _AUDIT_SPEC is None or _AUDIT_SPEC.loader is None:
    raise RuntimeError("cannot load sibling verifier")
audit = importlib.util.module_from_spec(_AUDIT_SPEC)
sys.modules[_AUDIT_SPEC.name] = audit
_AUDIT_SPEC.loader.exec_module(audit)


H = "11" * 32


def valid_pins_record() -> dict:
    return {
        "schema": audit.PINS_SCHEMA,
        "status": "EXTERNALLY_RECORDED_AFTER_PUBLICATION",
        "paths": {
            "output_parent": "/audit/output",
            "final_name": "v9-result",
            "v9_package": "/repo/v9",
            "support_path": "/repo/support.py",
            "v8_package": "/repo/v8",
            "strata_common_path": "/repo/common.py",
            "frozen_auditor_path": "/repo/auditor.py",
            "artifact_path": "/data/qwen.bin",
        },
        "source_hashes": {
            "v9_manifest_sha256": audit.KNOWN_V9_MANIFEST_SHA256,
            "v9_source_root_sha256": audit.KNOWN_V9_SOURCE_ROOT_SHA256,
            "primary_gate_sha256": audit.KNOWN_V9_RUNNER_SHA256,
            "support_sha256": audit.KNOWN_SUPPORT_SHA256,
            "v8_manifest_sha256": audit.KNOWN_V8_MANIFEST_SHA256,
            "v8_source_root_sha256": audit.KNOWN_V8_SOURCE_ROOT_SHA256,
            "v8_members": {
                name: {"bytes": size, "sha256": digest}
                for name, (size, digest) in audit.KNOWN_V8_MEMBERS.items()
            },
            "strata_common_sha256": audit.KNOWN_STRATA_COMMON_SHA256,
            "frozen_auditor_sha256": audit.KNOWN_FROZEN_AUDITOR_SHA256,
            "artifact_sha256": audit.KNOWN_ARTIFACT_SHA256,
            "artifact_bytes": audit.KNOWN_ARTIFACT_BYTES,
        },
        "publication_members": {
            name: {"bytes": 17, "sha256": H}
            for name in audit.PUBLICATION_MEMBERS
        },
        "original_source_identity": {
            "source_full_geometry_sha256": H,
            "source_structural_geometry_sha256": "22" * 32,
            "reconstruction_f64_sha256": "33" * 32,
            "score_receipt_sha256": "44" * 32,
            "relative_mse": 0.5,
            "sse_fp64": 1.0,
            "source_energy_fp64": 2.0,
        },
    }


class FakeHeld:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.sha256 = audit.sha256(data)
        self.before = (1, 2, 0, len(data), 10, 10, 1)


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_key_rejected(self) -> None:
        with self.assertRaises(audit.AuditError):
            audit.strict_json(b'{"a":1,"a":2}', "duplicate")

    def test_nonfinite_rejected(self) -> None:
        for raw in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e9999}'):
            with self.assertRaises(audit.AuditError):
                audit.strict_json(raw, "nonfinite")

    def test_canonical_pretty_is_distinct(self) -> None:
        row = {"z": 1, "a": False}
        self.assertNotEqual(audit.canonical_json(row), audit.pretty_json(row))
        self.assertEqual(audit.strict_json(audit.pretty_json(row), "pretty"), row)


class ExternalPinTests(unittest.TestCase):
    def test_complete_external_pin_bundle_passes_pure_parse(self) -> None:
        pins = audit.parse_pins(valid_pins_record())
        self.assertEqual(set(pins.publication_members), audit.PUBLICATION_MEMBERS)
        self.assertEqual(set(pins.v8_members), set(audit.V8_REQUIRED_MEMBERS))

    def test_unresolved_status_rejected(self) -> None:
        row = valid_pins_record()
        row["status"] = "UNRESOLVED"
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(row)

    def test_missing_publication_member_rejected(self) -> None:
        row = valid_pins_record()
        row["publication_members"].pop("UWFCV8.bin")
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(row)

    def test_publication_pin_is_not_copied_from_complete(self) -> None:
        row = valid_pins_record()
        row["publication_members"]["RESULT.json"]["sha256"] = None
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(row)

    def test_v8_member_pin_weakening_rejected(self) -> None:
        row = valid_pins_record()
        row["source_hashes"]["v8_members"].pop("stage0_census.py")
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(row)

    def test_original_score_identity_does_not_close_rejected(self) -> None:
        row = valid_pins_record()
        row["original_source_identity"]["relative_mse"] = 0.4
        with self.assertRaises(audit.AuditError):
            audit.parse_pins(row)


class CompletionTests(unittest.TestCase):
    def build(self) -> tuple[dict, dict[str, FakeHeld]]:
        observed = {name: FakeHeld(name.encode("ascii"))
                    for name in audit.DATA_MEMBERS}
        clean = {
            "schema": audit.COMPLETION_SCHEMA,
            "status": "NO_PROMOTION_PRIMARY_NESTED_HELDOUT",
            "positive_claim_authority": False,
            "controls_run": False,
            "shuffles_run": False,
            "coordinate_diagnostic_run": False,
            "v9_source_snapshot_root_sha256": audit.KNOWN_V9_SOURCE_ROOT_SHA256,
            "members": [
                {"name": name, "bytes": len(observed[name].data),
                 "sha256": observed[name].sha256}
                for name in sorted(audit.DATA_MEMBERS,
                                   key=lambda value: value.encode("utf-8"))
            ],
        }
        return {**clean, "completion_sha256": audit.sha256(
            audit.canonical_json(clean))}, observed

    def test_valid_completion(self) -> None:
        complete, observed = self.build()
        audit.verify_completion(complete, observed,
                                audit.KNOWN_V9_SOURCE_ROOT_SHA256)

    def test_reordered_member_rows_rejected(self) -> None:
        complete, observed = self.build()
        complete["members"].reverse()
        clean = dict(complete)
        clean.pop("completion_sha256")
        complete["completion_sha256"] = audit.sha256(audit.canonical_json(clean))
        with self.assertRaises(audit.AuditError):
            audit.verify_completion(complete, observed,
                                    audit.KNOWN_V9_SOURCE_ROOT_SHA256)

    def test_controls_counter_laundering_rejected(self) -> None:
        complete, observed = self.build()
        complete["controls_run"] = True
        clean = dict(complete)
        clean.pop("completion_sha256")
        complete["completion_sha256"] = audit.sha256(audit.canonical_json(clean))
        with self.assertRaises(audit.AuditError):
            audit.verify_completion(complete, observed,
                                    audit.KNOWN_V9_SOURCE_ROOT_SHA256)


def minimal_scientific(heldout: bool) -> dict:
    return {"passes_heldout_gate": heldout}


def minimal_source_final(*, integrity: bool = True, rate: bool = True,
                         f_gate: bool = True, cold: bool = True) -> dict:
    return {
        "parsed_metrics": {
            "passes_rate_interval": rate,
            "passes_F_target": f_gate,
            "passes_cold_read_below_2x": cold,
        },
        "standalone_decode": {
            "all_payloads_canonically_reencoded": integrity,
        },
        "identical_reconstruction_proved_by_full_f64_digest": integrity,
        "all_adapted_values_deserialized_from_transmitted_model": integrity,
    }


class DecisionTests(unittest.TestCase):
    def test_decision_order(self) -> None:
        self.assertEqual(audit.recompute_primary_status(
            minimal_scientific(True), minimal_source_final(integrity=False)),
            "FAIL_EVIDENCE_INTEGRITY_PRIMARY_CONTAINER")
        self.assertEqual(audit.recompute_primary_status(
            minimal_scientific(True), minimal_source_final(rate=False)),
            "HARD_KILL_PRIMARY_PHYSICAL_RATE_OR_F")
        self.assertEqual(audit.recompute_primary_status(
            minimal_scientific(True), minimal_source_final(cold=False)),
            "FAIL_PRIMARY_STRICT_COLD_READ")
        self.assertEqual(audit.recompute_primary_status(
            minimal_scientific(False), minimal_source_final()),
            "NO_PROMOTION_PRIMARY_NESTED_HELDOUT")
        self.assertEqual(audit.recompute_primary_status(
            minimal_scientific(True), minimal_source_final()),
            "PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED")

    def test_deep_positive_counter_rejected(self) -> None:
        with self.assertRaises(audit.AuditError):
            audit.verify_nonpromotion_counters(
                {"nested": [{"positive_promotion": True}]})


class FilesystemIdentityTests(unittest.TestCase):
    @staticmethod
    def info(*, mtime: int, ctime: int):
        return types.SimpleNamespace(
            st_dev=1, st_ino=2, st_mode=0o040755, st_size=4096,
            st_mtime_ns=mtime, st_ctime_ns=ctime, st_nlink=3,
        )

    def test_broad_ancestor_identity_ignores_unrelated_sibling_churn(self) -> None:
        before = self.info(mtime=10, ctime=10)
        after = self.info(mtime=20, ctime=20)
        self.assertEqual(audit.weak_directory_identity(before),
                         audit.weak_directory_identity(after))
        self.assertNotEqual(audit.strong_identity(before),
                            audit.strong_identity(after))


class ComponentPlanTests(unittest.TestCase):
    def panel(self) -> dict:
        streams = []
        ordinal = 0
        for owners in ((0, 1), (2, 3), (4, 5)):
            raw_owner_set = bytes([sum(1 << value for value in owners)]) + bytes(31)
            for _ in range(2):
                streams.append({
                    "stream_ordinal": ordinal,
                    "owner_identity_indices": list(owners),
                    "owner_expert_ordinals": list(owners),
                    "owner_set": raw_owner_set,
                    "owner_set_hex": raw_owner_set.hex(),
                    "symbols": 10 + ordinal,
                })
                ordinal += 1
        return {
            "experts": 6,
            "semantic_identities": [(15, index) for index in range(6)],
            "streams": streams,
        }

    def test_three_disjoint_owner_components_and_workload(self) -> None:
        panel = self.panel()
        plans = audit.independent_component_plan(panel, 5)
        self.assertEqual([row["identity_indices"] for row in plans],
                         [[0, 1], [2, 3], [4, 5]])
        workload = audit.recompute_workload(plans, panel["streams"])
        expected = sum(
            150 * (sum(panel["streams"][i]["symbols"]
                       for i in plan["train_indices"] +
                       plan["validation_indices"]) )
            + sum(panel["streams"][i]["symbols"]
                  for i in plan["development_indices"] + plan["test_indices"])
            for plan in plans
        ) + 2 * sum(row["symbols"] for row in panel["streams"])
        self.assertEqual(workload["exact_primary_updates"], expected)
        self.assertEqual(
            workload["expected_observed_cuda_updates"],
            expected - workload["full_symbols"])


class IndependentQ016Tests(unittest.TestCase):
    def test_jeffreys_half_count_integer_rounding(self) -> None:
        self.assertEqual(audit.independent_q16_frequencies([0, 0]), [32768])
        self.assertEqual(audit.independent_q16_frequencies([1, 0]), [16384])
        self.assertEqual(audit.independent_q16_frequencies([0, 1]), [49152])

    def test_reset_and_suffix_transition_count(self) -> None:
        candidate = types.SimpleNamespace(
            states=2, reset_length=2, topology="suffix")
        counts = [0] * (2 * 384 * 2)
        bits = [1, 0, 1, 1]
        levels = [0, 0, 0, 0]
        base = [32768, 32768, 32768, 32768]
        audit.independent_add_counts(bits, levels, base, candidate, counts)
        self.assertEqual(sum(counts), 4)
        context0 = audit.independent_public_context(0, 32768, 0)
        context1 = audit.independent_public_context(0, 32768, 1)
        self.assertEqual(counts[(0 * 384 + context0) * 2 + 1], 2)
        self.assertEqual(counts[(1 * 384 + context1) * 2 + 0], 1)

    def test_independent_arithmetic_roundtrip_and_canonical_payload(self) -> None:
        candidate = types.SimpleNamespace(
            states=2, reset_length=32, topology="suffix")
        bits = [0, 1, 1, 0, 1, 0, 0, 1] * 9
        levels = [index % 6 for index in range(len(bits))]
        base = [1 if index % 2 == 0 else 65535
                for index in range(len(bits))]
        frequencies = [32768] * (2 * 384)
        payload, logical = audit.independent_unifilar_encode(
            bits, levels, base, candidate, frequencies)
        decoded = audit.independent_unifilar_decode(
            payload, logical, levels, base, candidate, frequencies)
        self.assertEqual(decoded, bits)

    def test_independent_triplet_is_length_delimited(self) -> None:
        digest = audit.independent_triplet_sha256(
            b"\x00\x01", b"\x02\x03", b"\x01\x00\xff\xff")
        self.assertEqual(len(digest), 64)


class IndependentContainerTests(unittest.TestCase):
    def test_zero_header_is_rejected_before_any_producer_parser(self) -> None:
        with self.assertRaises(audit.AuditError):
            audit.independent_parse_container(bytes(audit.IC_HEADER_BYTES),
                                              "hostile-zero")


class BandwidthTests(unittest.TestCase):
    @staticmethod
    def frac(value: int) -> dict:
        return audit.fraction_record(audit.Fraction(value, 1))

    def test_owner_aware_maximum_recomputed(self) -> None:
        metrics = {
            "routed_io_authoritative_descriptor_backed": True,
            "passes_cold_read_below_2x": True,
            "maximum_strict_cold_read_amplification":
                audit.fraction_record(audit.Fraction(3, 2)),
            "experts": [
                {
                    "expert_ordinal": 0,
                    "attributable_total_physical_bytes": self.frac(8),
                    "attributable_nonpadding_decodable_bytes": self.frac(8),
                    "touched_page_bytes": 12,
                    "strict_cold_amplification":
                        audit.fraction_record(audit.Fraction(3, 2)),
                    "instrumented_routed_requested_bytes_with_repetition": 13,
                    "causal_decode_reencode_reconstruction": {"ok": True},
                }
            ],
        }
        result = audit.verify_bandwidth_metrics(metrics)
        self.assertTrue(result["passes_page_cold_below_2x"])
        self.assertTrue(result["passes_repeated_requested_below_2x"])


class TelemetryTraceTests(unittest.TestCase):
    @staticmethod
    def sample(phase: str) -> dict:
        return {
            "phase": phase,
            "process_tree_rss_bytes": 10,
            "process_hwm_bytes": 11,
            "free_vram_bytes": 100,
            "total_vram_bytes": 120,
            "default_pool_used_bytes": 1,
            "default_pool_total_bytes": 2,
            "pinned_pool_free_blocks": 0,
        }

    @staticmethod
    def statistics(samples: list[dict]) -> dict:
        return {
            "peak_process_tree_rss_bytes": 10,
            "peak_process_hwm_bytes": 11,
            "incremental_peak_process_tree_rss_bytes": 0,
            "peak_vram_incremental_bytes": 0,
            "peak_default_pool_used_bytes": 1,
            "peak_default_pool_total_bytes": 2,
            "peak_pinned_pool_free_blocks": 0,
            "baseline_free_vram_bytes": 100,
            "total_vram_bytes": 120,
            "telemetry_samples": len(samples),
        }

    def test_exact_primary_phase_schedule_has_2723_samples(self) -> None:
        phases = audit.expected_primary_telemetry_phases(3, 150)
        self.assertEqual(len(phases), 2723)
        self.assertEqual(phases[0], "pack_streams")
        self.assertEqual(phases[-1], "environment_receipt")
        self.assertEqual(phases.count("count_kernel"), 454)
        self.assertEqual(phases.count("length_kernel"), 453)
        self.assertEqual(phases.count("subset_descriptors"), 907)
        self.assertEqual(phases.count("device_to_host"), 907)

    def test_missing_primary_sample_is_rejected(self) -> None:
        before_samples = [self.sample("post_jit")]
        suffix = [self.sample(phase) for phase in
                  audit.expected_primary_telemetry_phases(1, 1)]
        after_samples = before_samples + suffix
        before = {
            "_samples": before_samples,
            "_statistics": self.statistics(before_samples),
        }
        after = {
            "_samples": after_samples,
            "_statistics": self.statistics(after_samples),
        }
        receipt = audit.verify_primary_telemetry_trace(
            before, after, fold_count=1, candidate_count=1)
        self.assertEqual(receipt["primary_suffix_samples"], len(suffix))
        hostile = copy.deepcopy(after)
        hostile["_samples"].pop(-2)
        with self.assertRaises(audit.AuditError):
            audit.verify_primary_telemetry_trace(
                before, hostile, fold_count=1, candidate_count=1)


class SourceInertnessTests(unittest.TestCase):
    def test_no_cupy_or_network_import(self) -> None:
        source = open(audit.__file__, "r", encoding="utf-8").read()
        self.assertNotIn("import cupy", source)
        self.assertNotIn("import requests", source)
        self.assertNotIn("subprocess", source)

    def test_exact_publication_member_set(self) -> None:
        self.assertEqual(len(audit.PUBLICATION_MEMBERS), 7)
        self.assertEqual(audit.DATA_MEMBERS | {"COMPLETE.json"},
                         audit.PUBLICATION_MEMBERS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
