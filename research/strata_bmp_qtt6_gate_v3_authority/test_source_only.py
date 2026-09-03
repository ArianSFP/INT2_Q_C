#!/usr/bin/env python3
"""Source-only hostile tests for the v3 evidence authority.

All generated objects are explicitly marked SOURCE_TEST_FIXTURE/dummy.  The
production path is tested only for rejection and never receives model data.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import authority as a  # noqa: E402


def d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def emit_json(path: Path, value: dict) -> None:
    path.write_bytes(a.canonical_json(value) + b"\n")


def score_row(route_id: str, role_hashes: dict[str, str], recon: str,
              *, packet_bytes: int = 4096, weight_count: int = 13108) -> dict:
    sse = 2.0
    energy = 100.0
    bits = packet_bytes * 8
    relative = sse / energy
    rate = bits / weight_count
    return {
        "route_id": route_id,
        "source_role_sha256": role_hashes,
        "decoded_reconstruction_sha256": recon,
        "weight_count": weight_count,
        "physical_bits": bits,
        "sse_fp64": sse,
        "source_energy_fp64": energy,
        "relative_mse": relative,
        "physical_rate_bpw": rate,
        "f_value": relative * 2.0 ** (2.0 * rate),
    }


def adapter_row(route_id: str, packet_sha: str, recon: str) -> dict:
    return {
        "route_id": route_id,
        "packet_sha256": packet_sha,
        "packet_bytes": 4096,
        "scale_payload_sha256": d("scale-" + route_id),
        "scale_bytes": 64,
        "scale_payload_inside_packet": True,
        "forward_transform_id": "strata-rht-v-current",
        "forward_transform_sha256": d("forward-current"),
        "inverse_transform_sha256": d("inverse-current"),
        "framing_header_bytes": 128,
        "framing_payload_bytes": 3900,
        "framing_trailer_bytes": 68,
        "framing_padding_bytes": 0,
        "canonical_reencode_equal": True,
        "decoded_reconstruction_sha256": recon,
        "decoded_weight_count": 13108,
    }


def read_row(route_id: str, packet_sha: str, layer: str, expert: str) -> dict:
    return {
        "route_id": route_id,
        "layer": layer,
        "expert": expert,
        "packet_sha256": packet_sha,
        "literal_packet_bytes": 4096,
        "events": [{
            "sequence": 0,
            "page_index": 0,
            "file_offset": 0,
            "bytes_read": 4096,
            "page_sha256": packet_sha,
        }],
        "unique_page_indices": [0],
        "physical_page_bytes_read": 4096,
        "cold_read_amplification": 1.0,
        "one_routed_expert_only": True,
    }


def make_capability(root: Path, kind: str, details: dict) -> dict:
    folder = root / ("cap-" + kind)
    folder.mkdir()
    implementation = ("source-test implementation for " + kind + "\n").encode()
    (folder / "implementation.bin").write_bytes(implementation)
    implementation_sha = a.sha256(implementation)
    manifest_stub = {
        "kind": kind,
        "capability_id": "cap-" + kind,
        "producer_authority_id": "producer-" + kind,
        "executor_authority_id": "executor-" + kind,
        "auditor_authority_id": "auditor-" + kind,
    }
    execution = {
        "schema": a.EXECUTION_SCHEMA,
        "status": a.EXECUTION_STATUS[kind],
        "executed": True,
        "evidence_class": "SOURCE_TEST_FIXTURE",
        "kind": kind,
        "capability_id": manifest_stub["capability_id"],
        "producer_authority_id": manifest_stub["producer_authority_id"],
        "executor_authority_id": manifest_stub["executor_authority_id"],
        "implementation_sha256": implementation_sha,
        "invocation_sha256": d("invocation-" + kind),
        "input_manifest_sha256": d("input-" + kind),
        "output_sha256": d("output-" + kind),
        "started_utc": "2026-09-03T00:00:00Z",
        "finished_utc": "2026-09-03T00:00:01Z",
        "test_fixture": True,
        "dummy": True,
        "self_authored": True,
        "details": details,
    }
    emit_json(folder / "EXECUTION_RECEIPT.json", execution)
    execution_sha = a.sha256((folder / "EXECUTION_RECEIPT.json").read_bytes())
    audit = {
        "schema": a.AUDIT_SCHEMA,
        "status": "PASS_INDEPENDENT_EXECUTED_CAPABILITY_AUDIT",
        "executed": True,
        "evidence_class": "SOURCE_TEST_FIXTURE",
        "kind": kind,
        "capability_id": manifest_stub["capability_id"],
        "producer_authority_id": manifest_stub["producer_authority_id"],
        "auditor_authority_id": manifest_stub["auditor_authority_id"],
        "implementation_sha256": implementation_sha,
        "execution_receipt_sha256": execution_sha,
        "exact_closure_verified": True,
        "semantic_replay_verified": True,
        "hostile_tests": 1,
        "test_fixture": True,
        "dummy": True,
        "self_authored": True,
        "findings": [],
    }
    emit_json(folder / "AUDIT_RECEIPT.json", audit)
    rows = []
    for name in sorted(("AUDIT_RECEIPT.json", "EXECUTION_RECEIPT.json",
                        "implementation.bin"), key=lambda value: value.encode("utf-8")):
        payload = (folder / name).read_bytes()
        rows.append({"name": name, "bytes": len(payload), "sha256": a.sha256(payload)})
    source_root = a.sha256(a.canonical_json(rows))
    manifest = {
        "schema": a.CAPABILITY_SCHEMA,
        "status": "SEALED_EXECUTED_CAPABILITY",
        "kind": kind,
        "capability_id": manifest_stub["capability_id"],
        "evidence_class": "SOURCE_TEST_FIXTURE",
        "producer_authority_id": manifest_stub["producer_authority_id"],
        "executor_authority_id": manifest_stub["executor_authority_id"],
        "auditor_authority_id": manifest_stub["auditor_authority_id"],
        "source_root_sha256": source_root,
        "members": rows,
        "implementation_name": "implementation.bin",
        "execution_receipt_name": "EXECUTION_RECEIPT.json",
        "audit_receipt_name": "AUDIT_RECEIPT.json",
    }
    emit_json(folder / "CAPABILITY_MANIFEST.json", manifest)
    return {
        "kind": kind,
        "relative_path": folder.name,
        "manifest_sha256": a.sha256((folder / "CAPABILITY_MANIFEST.json").read_bytes()),
        "source_root_sha256": source_root,
        "execution_receipt_sha256": execution_sha,
        "audit_receipt_sha256": a.sha256((folder / "AUDIT_RECEIPT.json").read_bytes()),
    }


def build_fixture(root: Path) -> tuple[Path, str, list[dict]]:
    route_specs = (("model-0", "model_bf16", 11),
                   ("control-0", "matched_gaussian_bf16", 31))
    routes = []
    route_meta = {}
    for ordinal, (route_id, kind, seed) in enumerate(route_specs):
        sources = []
        role_hashes = {}
        for role_index, (role, size) in enumerate(
                (("gate", 8738), ("up", 8738), ("down_transposed", 8740))):
            payload = bytes([(seed + role_index) % 251 + 1]) * size
            name = f"{route_id}-{role}.bf16"
            (root / name).write_bytes(payload)
            digest = a.sha256(payload)
            sources.append({"role": role, "relative_path": name,
                            "bytes": size, "sha256": digest})
            role_hashes[role] = digest
        packet = bytes([90 + ordinal]) * 4096
        packet_name = route_id + ".strata"
        (root / packet_name).write_bytes(packet)
        packet_sha = a.sha256(packet)
        recon = d("reconstruction-" + route_id)
        routes.append({
            "route_id": route_id,
            "kind": kind,
            "architecture_family": "universal-swiglu",
            "layer": "layer-0",
            "expert": "expert-0" if kind == "model_bf16" else "control-0",
            "sources": sources,
            "packet": {"relative_path": packet_name, "bytes": 4096,
                       "sha256": packet_sha},
            "required_control_route_ids": ["control-0"] if kind == "model_bf16" else [],
        })
        route_meta[route_id] = {"role_hashes": role_hashes, "packet_sha": packet_sha,
                                "recon": recon, "kind": kind}

    predecessor = {
        "v2_manifest_sha256": a.V2_MANIFEST_SHA256,
        "v2_source_root_sha256": a.V2_SOURCE_ROOT_SHA256,
        "v2_exact_closure_verified": True,
        "v2_audit_manifest_sha256": a.V2_AUDIT_MANIFEST_SHA256,
        "v2_audit_source_root_sha256": a.V2_AUDIT_SOURCE_ROOT_SHA256,
        "v2_audit_exact_closure_verified": True,
        "producer_source_tests_executed": True,
        "independent_cpu_audit_executed": True,
        "independent_cupy_audit_executed": True,
    }
    controls = {
        "generator_sha256": d("control-generator"),
        "integer_prng_spec_sha256": d("integer-prng"),
        "backend_byte_identical": True,
        "complete_selection_replayed": True,
        "control_routes": [{
            "route_id": "control-0",
            "source_role_sha256": route_meta["control-0"]["role_hashes"],
            "source_identity_sha256": d("control-source-identity"),
            "selected_packet_sha256": route_meta["control-0"]["packet_sha"],
            "selected_reconstruction_sha256": route_meta["control-0"]["recon"],
        }],
        "model_route_ids": ["model-0"],
    }
    adapter = {
        "adapter_abi": "CURRENT_STRATA_SIX_PLANE_INDEX64_V1",
        "literal_current_strata": True,
        "completed_planes": 6,
        "decoded_index_min": 0,
        "decoded_index_max": 63,
        "routes": [adapter_row(route_id, meta["packet_sha"], meta["recon"])
                   for route_id, meta in route_meta.items()],
    }
    score_rows = [score_row(route_id, meta["role_hashes"], meta["recon"])
                  for route_id, meta in route_meta.items()]
    total_sse = sum(row["sse_fp64"] for row in score_rows)
    total_energy = sum(row["source_energy_fp64"] for row in score_rows)
    total_count = sum(row["weight_count"] for row in score_rows)
    total_bits = sum(row["physical_bits"] for row in score_rows)
    pooled_relative = total_sse / total_energy
    pooled_rate = total_bits / total_count
    scorer = {
        "source_dtype": "BF16_LE",
        "accumulation": "FP64",
        "independent_from_adapter": True,
        "routes": score_rows,
        "pooled": {"sse_fp64": total_sse, "source_energy_fp64": total_energy,
                   "weight_count": total_count, "physical_bits": total_bits,
                   "relative_mse": pooled_relative,
                   "physical_rate_bpw": pooled_rate,
                   "f_value": pooled_relative * 2.0 ** (2.0 * pooled_rate)},
    }
    reader = {
        "page_bytes": 4096,
        "instrumented_reads": True,
        "layout_only": False,
        "routes": [read_row(route_id, meta["packet_sha"], "layer-0",
                            "expert-0" if route_id == "model-0" else "control-0")
                   for route_id, meta in route_meta.items()],
    }
    details = {
        "predecessor_source_audit": predecessor,
        "gaussian_control_generator": controls,
        "current_strata_adapter": adapter,
        "independent_bf16_scorer": scorer,
        "routed_page_reader": reader,
    }
    pins = [make_capability(root, kind, details[kind]) for kind in details]
    other_pins = sorted(pins, key=lambda item: item["kind"].encode("utf-8"))
    launch_audit = {
        "other_capability_pin_set_sha256": a.sha256(a.canonical_json(other_pins)),
        "launch_schema_verified": True,
        "route_closure_verified": True,
        "model_control_alias_checks_replayed": True,
        "strata_adapter_replayed": True,
        "bf16_scorer_replayed": True,
        "per_expert_read_trace_replayed": True,
    }
    pins.append(make_capability(root, "independent_launch_audit", launch_audit))
    pins.sort(key=lambda item: item["kind"].encode("utf-8"))
    manifest = {
        "schema": a.LAUNCH_SCHEMA,
        "status": "SEALED_BEFORE_PAYLOAD_LAUNCH",
        "evidence_class": "SOURCE_TEST_FIXTURE",
        "issuer_authority_id": "fixture-launch-issuer",
        "v3_source_manifest_sha256": d("future-v3-manifest"),
        "v3_source_root_sha256": d("future-v3-root"),
        "predecessor_pins": {
            "v2_manifest_sha256": a.V2_MANIFEST_SHA256,
            "v2_source_root_sha256": a.V2_SOURCE_ROOT_SHA256,
            "v2_audit_manifest_sha256": a.V2_AUDIT_MANIFEST_SHA256,
            "v2_audit_source_root_sha256": a.V2_AUDIT_SOURCE_ROOT_SHA256,
        },
        "capability_pins": pins,
        "routes": routes,
    }
    path = root / "LAUNCH_MANIFEST.json"
    emit_json(path, manifest)
    return path, a.sha256(path.read_bytes()), pins


class AuthorityTests(unittest.TestCase):
    def test_01_actual_predecessor_closures(self) -> None:
        result = a.authenticate_pinned_predecessors(
            HERE.parent / "strata_bmp_qtt6_gate_v2_replay",
            HERE.parent / "strata_bmp_qtt6_gate_v2_replay_independent_source_audit_20260902")
        self.assertEqual(result["producer"]["members"], 13)
        self.assertEqual(result["independent_source_audit"]["members"], 7)

    def test_02_production_is_compiled_hold(self) -> None:
        self.assertIsNone(a.TRUSTED_LAUNCH_MANIFEST_SHA256)
        with self.assertRaisesRegex(a.AuthorityError, "HOLD"):
            a.authorize_production(Path("unused"), Path("unused"))

    def test_03_complete_source_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest, _ = build_fixture(root)
            result = a.verify_precommitted_evidence(
                root, path, digest, allow_source_test_fixture=True)
            self.assertTrue(result["verified"])
            self.assertFalse(result["production_authorized"])
            self.assertLessEqual(result["pooled_model_f_value"], 0.8)

    def test_04_fixture_cannot_be_production_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, pins = build_fixture(root)
            with self.assertRaises(a.AuthorityError):
                a.authenticate_capability(root, pins[0], allow_source_test_fixture=False)

    def test_05_extra_capability_member_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, pins = build_fixture(root)
            pin = pins[0]
            (root / pin["relative_path"] / "SURPLUS").write_bytes(b"x")
            with self.assertRaisesRegex(a.AuthorityError, "exact regular closure"):
                a.authenticate_capability(root, pin, allow_source_test_fixture=True)

    def test_06_tampered_receipt_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, pins = build_fixture(root)
            pin = pins[0]
            target = root / pin["relative_path"] / "EXECUTION_RECEIPT.json"
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaises(a.AuthorityError):
                a.authenticate_capability(root, pin, allow_source_test_fixture=True)

    def test_07_self_authored_receipt_rejected(self) -> None:
        manifest = {"kind": "routed_page_reader", "capability_id": "x",
                    "producer_authority_id": "p", "executor_authority_id": "e"}
        record = {
            "schema": a.EXECUTION_SCHEMA, "status": a.EXECUTION_STATUS["routed_page_reader"],
            "executed": True, "evidence_class": "PRODUCTION_EXECUTION",
            "kind": "routed_page_reader", "capability_id": "x",
            "producer_authority_id": "p", "executor_authority_id": "e",
            "implementation_sha256": d("i"), "invocation_sha256": d("v"),
            "input_manifest_sha256": d("m"), "output_sha256": d("o"),
            "started_utc": "x", "finished_utc": "y", "test_fixture": False,
            "dummy": False, "self_authored": True, "details": {},
        }
        with self.assertRaisesRegex(a.AuthorityError, "fixture, dummy, or self-authored"):
            a._validate_common_execution(record, manifest, d("i"),
                                         allow_source_test_fixture=False)

    def test_08_adapter_requires_literal_scale_bytes(self) -> None:
        row = adapter_row("x", d("packet"), d("recon"))
        row["scale_bytes"] = 0
        details = {"adapter_abi": "CURRENT_STRATA_SIX_PLANE_INDEX64_V1",
                   "literal_current_strata": True, "completed_planes": 6,
                   "decoded_index_min": 0, "decoded_index_max": 63, "routes": [row]}
        with self.assertRaises(a.AuthorityError):
            a._validate_adapter_details(details, fixture=True)

    def test_09_bf16_scorer_arithmetic_recomputed(self) -> None:
        row = score_row("x", {role: d(role) for role in
                              ("gate", "up", "down_transposed")}, d("recon"))
        pooled = {key: row[key] for key in
                  ("sse_fp64", "source_energy_fp64", "weight_count", "physical_bits",
                   "relative_mse", "physical_rate_bpw", "f_value")}
        details = {"source_dtype": "BF16_LE", "accumulation": "FP64",
                   "independent_from_adapter": True, "routes": [row], "pooled": pooled}
        a._validate_scorer_details(details, fixture=True)
        details["routes"][0]["f_value"] += 0.01
        with self.assertRaisesRegex(a.AuthorityError, "recomputation"):
            a._validate_scorer_details(details, fixture=True)

    def test_10_layout_is_not_read_trace(self) -> None:
        details = {"page_bytes": 4096, "instrumented_reads": True,
                   "layout_only": True,
                   "routes": [read_row("x", d("packet"), "l", "e")]}
        with self.assertRaisesRegex(a.AuthorityError, "not layout"):
            a._validate_read_details(details, fixture=True)

    def test_11_read_amplification_is_recomputed(self) -> None:
        row = read_row("x", d("packet"), "l", "e")
        row["literal_packet_bytes"] = 2000
        row["cold_read_amplification"] = 4096 / 2000
        details = {"page_bytes": 4096, "instrumented_reads": True,
                   "layout_only": False, "routes": [row]}
        with self.assertRaisesRegex(a.AuthorityError, "amplification"):
            a._validate_read_details(details, fixture=True)

    def test_12_wrong_launch_precommit_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, _ = build_fixture(root)
            with self.assertRaisesRegex(a.AuthorityError, "precommit"):
                a.verify_precommitted_evidence(
                    root, path, d("wrong"), allow_source_test_fixture=True)

    def test_13_model_control_byte_alias_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, _ = build_fixture(root)
            record = json.loads(path.read_text("ascii"))
            model_source = record["routes"][0]["sources"][0]
            record["routes"][1]["sources"][0] = deepcopy(model_source)
            emit_json(path, record)
            with self.assertRaisesRegex(a.AuthorityError, "alias"):
                a.verify_precommitted_evidence(
                    root, path, a.sha256(path.read_bytes()),
                    allow_source_test_fixture=True)

    def test_14_noncanonical_launch_json_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, _ = build_fixture(root)
            record = json.loads(path.read_text("ascii"))
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
            with self.assertRaisesRegex(a.AuthorityError, "canonical JSON"):
                a.verify_precommitted_evidence(
                    root, path, a.sha256(path.read_bytes()),
                    allow_source_test_fixture=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
