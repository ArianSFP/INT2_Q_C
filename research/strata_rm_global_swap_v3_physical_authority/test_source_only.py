#!/usr/bin/env python3
"""Source-only hostile tests for v3 routed physical authority."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import authority_v3 as auth


def digest(character: str) -> str:
    return character * 64


def canonical_file(path: Path, value) -> str:
    payload = auth.canonical_json(value) + b"\n"
    path.write_bytes(payload)
    return auth.sha256(payload)


def sources(prefix: str, family_index: int, *, expert: int = 0):
    hashes = [format(family_index * 3 + index, "x") * 64
              for index in range(3)]
    return [
        {"ordinal": 0, "role": "gate", "layer": 1, "expert": expert,
         "shape": [2, 4], "relative_path": f"{prefix}-gate.bf16",
         "bytes": 16, "sha256": hashes[0]},
        {"ordinal": 1, "role": "up", "layer": 1, "expert": expert,
         "shape": [2, 4], "relative_path": f"{prefix}-up.bf16",
         "bytes": 16, "sha256": hashes[1]},
        {"ordinal": 2, "role": "down", "layer": 1, "expert": expert,
         "shape": [4, 2], "relative_path": f"{prefix}-down.bf16",
         "bytes": 16, "sha256": hashes[2]},
    ]


def capability_record():
    pipeline = digest("a")
    rows = []
    families = ["Qwen-SwiGLU-MoE", "Independent-SwiGLU-MoE"]
    for index, family in enumerate(families):
        model_id = f"model-{index}"
        control_id = f"control-{index}"
        rows.append({
            "route_id": model_id,
            "kind": "qwen_bf16" if index == 0 else "swiglu_moe_bf16",
            "architecture_family": family, "pipeline_sha256": pipeline,
            "checkpoint_manifest_sha256": digest(str(index + 1)),
            "tensor_manifest_sha256": digest(chr(ord("b") + index)),
            "checkpoint_identity_sha256": digest(chr(ord("d") + index)),
            "architecture_schema_sha256": digest(str(8 + index)),
            "control_family": None, "paired_model_route_id": None,
            "generator_sha256": None, "seed_sha256": None,
            "moments_sha256": None,
            "required_control_route_ids": [control_id],
            "sources": sources(model_id, index * 2),
        })
        rows.append({
            "route_id": control_id, "kind": "matched_gaussian_bf16",
            "architecture_family": family, "pipeline_sha256": pipeline,
            "checkpoint_manifest_sha256": None,
            "tensor_manifest_sha256": None,
            "checkpoint_identity_sha256": None,
            "architecture_schema_sha256": None,
            "control_family": "moment_matched_pipeline_replay",
            "paired_model_route_id": model_id,
            "generator_sha256": digest(chr(ord("a") + index)),
            "seed_sha256": digest(chr(ord("c") + index)),
            "moments_sha256": digest(chr(ord("e") + index)),
            "required_control_route_ids": [],
            "sources": sources(control_id, index * 2 + 1),
        })
    return {"schema": "strata-rm-global-swap-v3-scientific-capability",
            "selection": {"pipeline_sha256": pipeline,
                          "frozen_before_test": True,
                          "test_bytes_opened": 0,
                          "search_replayed_on_every_control": True},
            "architecture_families": families, "routes": rows}


def make_scientific_audit(root: Path, capability=None, *, executed=True):
    capability = capability_record() if capability is None else capability
    capability_sha = canonical_file(root / "SCIENTIFIC_CAPABILITY.json", capability)
    audit_source = b"# independent scientific provenance audit\n"
    (root / "audit.py").write_bytes(audit_source)
    members = [{"name": "audit.py", "bytes": len(audit_source),
                "sha256": auth.sha256(audit_source)}]
    source_root = auth._member_root(members)
    receipt = {
        "schema": "strata-rm-global-swap-v3-scientific-independent-audit-receipt",
        "executed": executed,
        "status": "PASS_INDEPENDENT_SCIENTIFIC_PROVENANCE_AUDIT_V3",
        "audit_source_root_sha256": source_root,
        "scientific_capability_sha256": capability_sha,
        "checkpoint_manifests_opened": True,
        "tensor_manifests_opened": True,
        "source_hashes_recomputed": True,
        "control_generator_replayed": True,
        "control_moments_recomputed": True,
        "family_identity_verified": True,
        "cross_family_aliases_rejected": True,
        "selection_replay_verified": True, "hostile_tests": 14,
    }
    receipt_sha = canonical_file(root / "AUDIT_RECEIPT.json", receipt)
    manifest = {
        "schema": "strata-rm-global-swap-v3-scientific-independent-audit-manifest",
        "source_root_sha256": source_root,
        "receipt_name": "AUDIT_RECEIPT.json",
        "capability_name": "SCIENTIFIC_CAPABILITY.json",
        "capability_sha256": capability_sha, "members": members}
    manifest_sha = canonical_file(root / "source_manifest.json", manifest)
    return {"expected_manifest_sha256": manifest_sha,
            "expected_source_root_sha256": source_root,
            "expected_receipt_sha256": receipt_sha,
            "expected_capability_sha256": capability_sha}


def make_decoder_audit(root: Path, sandbox_sha: str, *, executed=True):
    module = b"\x00asm\x01\x00\x00\x00"
    module_sha = auth.sha256(module)
    (root / "DECODER.wasm").write_bytes(module)
    audit_source = b"# independent WebAssembly decoder audit\n"
    (root / "audit.py").write_bytes(audit_source)
    members = [{"name": "audit.py", "bytes": len(audit_source),
                "sha256": auth.sha256(audit_source)}]
    source_root = auth._member_root(members)
    receipt = {
        "schema": "strata-rm-global-swap-v3-wasm-decoder-independent-audit-receipt",
        "executed": executed,
        "status": "PASS_INDEPENDENT_ZERO_IMPORT_WASM_DECODER_AUDIT_V3",
        "audit_source_root_sha256": source_root,
        "decoder_module_sha256": module_sha, "sandbox_sha256": sandbox_sha,
        "zero_imports_verified": True, "no_wasi_verified": True,
        "abi_verified": True, "input_immutability_verified": True,
        "canonical_replay_verified": True, "fixed_universal_decoder": True,
        "qwen_specific_tables_absent": True, "hostile_tests": 14,
        "payloads_opened": 0,
    }
    receipt_sha = canonical_file(root / "AUDIT_RECEIPT.json", receipt)
    manifest = {
        "schema": "strata-rm-global-swap-v3-wasm-decoder-independent-audit-manifest",
        "source_root_sha256": source_root,
        "receipt_name": "AUDIT_RECEIPT.json",
        "decoder_module_name": "DECODER.wasm",
        "decoder_module_sha256": module_sha,
        "sandbox_sha256": sandbox_sha, "members": members}
    manifest_sha = canonical_file(root / "source_manifest.json", manifest)
    return {"expected_manifest_sha256": manifest_sha,
            "expected_source_root_sha256": source_root,
            "expected_receipt_sha256": receipt_sha,
            "expected_decoder_module_sha256": module_sha,
            "expected_sandbox_sha256": sandbox_sha}


def result_row(route_id: str, kind: str, family: str, factor: float,
               *, control_family=None, paired=None):
    weights = 27307
    packet_bytes = 8192
    rate = 8.0 * packet_bytes / weights
    relative = factor / (2.0 ** (2.0 * rate))
    return {"route_id": route_id, "kind": kind,
            "architecture_family": family, "control_family": control_family,
            "paired_model_route_id": paired, "weights": weights,
            "literal_packet_bytes": packet_bytes, "physical_rate_bpw": rate,
            "sse_fp64_hex": relative.hex(), "energy_fp64_hex": (1.0).hex(),
            "relative_mse": relative, "F": factor,
            "saving_bpw": -0.5 * math.log2(factor),
            "cold_read": {"cold_read_amplification": 1.0}}


class V3AuthorityTests(unittest.TestCase):
    def test_root_symlink_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlink unavailable")
            with self.assertRaises(auth.AuthorityError):
                auth.real_directory(link, "linked root")

    def test_scientific_audit_package_and_receipt_are_both_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_scientific_audit(root)
            result = auth.authenticate_scientific_audit_package(root, **pins)
            self.assertEqual(result["status"],
                             "PASS_SEPARATELY_PINNED_SCIENTIFIC_AUDIT_AND_CAPABILITY")

    def test_unexecuted_scientific_audit_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_scientific_audit(root, executed=False)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_scientific_audit_package(root, **pins)

    def test_cross_family_checkpoint_alias_fails(self):
        record = capability_record()
        record["routes"][2]["checkpoint_manifest_sha256"] = (
            record["routes"][0]["checkpoint_manifest_sha256"])
        with self.assertRaises(auth.AuthorityError):
            auth._validate_scientific_capability(record)

    def test_cross_family_tensor_alias_fails(self):
        record = capability_record()
        record["routes"][2]["tensor_manifest_sha256"] = (
            record["routes"][0]["tensor_manifest_sha256"])
        with self.assertRaises(auth.AuthorityError):
            auth._validate_scientific_capability(record)

    def test_cross_family_source_hash_alias_fails(self):
        record = capability_record()
        record["routes"][2]["sources"][0]["sha256"] = (
            record["routes"][0]["sources"][0]["sha256"])
        with self.assertRaises(auth.AuthorityError):
            auth._validate_scientific_capability(record)

    def test_cross_family_source_path_alias_fails(self):
        record = capability_record()
        record["routes"][2]["sources"][0]["relative_path"] = (
            record["routes"][0]["sources"][0]["relative_path"])
        with self.assertRaises(auth.AuthorityError):
            auth._validate_scientific_capability(record)

    def test_model_control_source_hash_alias_fails(self):
        record = capability_record()
        record["routes"][1]["sources"][0]["sha256"] = (
            record["routes"][0]["sources"][0]["sha256"])
        with self.assertRaises(auth.AuthorityError):
            auth._validate_scientific_capability(record)

    def test_route_may_contain_only_one_expert_triplet(self):
        record = capability_record()
        record["routes"][0]["sources"][2]["expert"] = 9
        with self.assertRaises(auth.AuthorityError):
            auth._validate_scientific_capability(record)

    def test_control_must_be_exactly_paired_to_model_route(self):
        record = capability_record()
        record["routes"][1]["paired_model_route_id"] = "model-1"
        with self.assertRaises(auth.AuthorityError):
            auth._validate_scientific_capability(record)

    def test_decoder_audit_package_requires_executed_zero_import_receipt(self):
        sandbox_sha = auth.sha256((PACKAGE / "wasm_decoder_sandbox.py").read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_decoder_audit(root, sandbox_sha)
            result = auth.authenticate_decoder_audit_package(root, **pins)
            self.assertEqual(result["status"],
                             "PASS_SEPARATELY_PINNED_ZERO_IMPORT_WASM_DECODER_AUDIT")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_decoder_audit(root, sandbox_sha, executed=False)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_decoder_audit_package(root, **pins)

    def test_commitment_rejects_packet_alias_across_routes(self):
        record = {
            "schema": "strata-rm-global-swap-v3-routed-expert-physical-commitment",
            "mode": "production_routed_expert",
            "v2_source_root_sha256": auth.V2_SOURCE_ROOT_SHA256,
            "v2_review_source_root_sha256": auth.V2_REVIEW_SOURCE_ROOT_SHA256,
            "decoder_module_sha256": digest("a"), "sandbox_sha256": digest("b"),
            "route_packets": [
                {"route_id": "one", "relative_path": "one.bin", "bytes": 9,
                 "sha256": digest("c")},
                {"route_id": "two", "relative_path": "two.bin", "bytes": 9,
                 "sha256": digest("c")}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commitment.json"
            pin = canonical_file(path, record)
            with self.assertRaises(auth.AuthorityError):
                auth._strict_commitment(path, pin)

    def test_sandbox_receipt_charges_page_rounding_for_one_expert(self):
        packet = bytes(5000)
        module_sha = digest("a")
        sandbox_sha = digest("b")
        pages = [
            {"page_index": 0, "literal_offset": 0, "literal_bytes": 4096,
             "supplied_bytes": 4096},
            {"page_index": 1, "literal_offset": 4096, "literal_bytes": 904,
             "supplied_bytes": 4096},
        ]
        receipt = {
            "schema": "strata-rm-global-swap-v3-zero-import-wasm-sandbox-receipt",
            "route_id": "route", "decoder_module_sha256": module_sha,
            "sandbox_sha256": sandbox_sha, "module_imports": [],
            "wasi_enabled": False, "filesystem_api_exposed": False,
            "descriptor_api_exposed": False, "native_io_imports_exposed": False,
            "packet_buffer_preopened": True, "packet_input_unchanged": True,
            "packet_sha256": auth.sha256(packet),
            "literal_packet_bytes_supplied": 5000, "page_bytes": 4096,
            "pages_supplied": pages, "zero_padding_bytes_supplied": 3192,
            "physical_page_bytes_supplied": 8192, "decode_status": 0,
            "canonical_reencode_bytes": 5000,
            "status": "PASS_ZERO_IMPORT_WASM_EXPERT_PACKET_BUFFER_DECODE",
        }
        result = auth._validate_sandbox_receipt(
            receipt, route_id="route", packet=packet,
            module_sha256=module_sha, sandbox_sha256=sandbox_sha)
        self.assertAlmostEqual(result["cold_read_amplification"], 8192 / 5000)
        self.assertTrue(result["one_independently_routed_expert"])

    def test_every_family_and_strongest_control_must_pass(self):
        scientific = capability_record()
        results = []
        for index, family in enumerate(scientific["architecture_families"]):
            results.append(result_row(
                f"model-{index}", "qwen_bf16" if index == 0 else
                "swiglu_moe_bf16", family, 0.70))
            results.append(result_row(
                f"control-{index}", "matched_gaussian_bf16", family, 1.0,
                control_family="moment_matched_pipeline_replay",
                paired=f"model-{index}"))
        accepted = auth.evaluate_acceptance(results, scientific)
        self.assertTrue(accepted["all_families_passed"])
        failed_family = [dict(row) for row in results]
        failed_family[2] = result_row(
            "model-1", "swiglu_moe_bf16",
            scientific["architecture_families"][1], 0.81)
        with self.assertRaises(auth.AuthorityError):
            auth.evaluate_acceptance(failed_family, scientific)
        control_artifact = [dict(row) for row in results]
        control_artifact[1] = result_row(
            "control-0", "matched_gaussian_bf16",
            scientific["architecture_families"][0], 0.69,
            control_family="moment_matched_pipeline_replay", paired="model-0")
        with self.assertRaises(auth.AuthorityError):
            auth.evaluate_acceptance(control_artifact, scientific)

    def test_wasm_sandbox_exposes_no_python_handle_proxy(self):
        source = (PACKAGE / "wasm_decoder_sandbox.py").read_text(encoding="utf-8")
        self.assertIn("module_imports == []", source)
        self.assertIn("wasmtime.Instance(store, module, [])", source)
        self.assertNotIn("class PacketReader", source)
        self.assertNotIn("__getattr__", source)
        self.assertNotIn("fileno(", source)

    def test_physical_entry_rejects_wrong_authorization_before_io(self):
        with self.assertRaises(auth.AuthorityError):
            auth.validate_physical_bundle(
                v3_package=PACKAGE, expected_v3_manifest_sha256=digest("0"),
                evidence_root=PACKAGE, commitment_path=PACKAGE / "README.md",
                expected_commitment_sha256=digest("0"),
                scientific_audit_package=PACKAGE,
                expected_scientific_manifest_sha256=digest("0"),
                expected_scientific_source_root_sha256=digest("0"),
                expected_scientific_receipt_sha256=digest("0"),
                expected_scientific_capability_sha256=digest("0"),
                decoder_audit_package=PACKAGE,
                expected_decoder_manifest_sha256=digest("0"),
                expected_decoder_source_root_sha256=digest("0"),
                expected_decoder_receipt_sha256=digest("0"),
                expected_decoder_module_sha256=digest("0"),
                authorization="WRONG")


if __name__ == "__main__":
    unittest.main(verbosity=2)
