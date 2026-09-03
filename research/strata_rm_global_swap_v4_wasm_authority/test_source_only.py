#!/usr/bin/env python3
"""Hostile source-only tests for the final v4 Wasmtime authority repair."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import authority_v4 as auth


def canonical_file(path: Path, value) -> str:
    payload = auth.canonical_json(value) + b"\n"
    path.write_bytes(payload)
    return auth.sha256(payload)


def member_row(path: Path) -> dict:
    payload = path.read_bytes()
    return {"name": path.name, "bytes": len(payload),
            "sha256": auth.sha256(payload)}


def make_runtime_audit(root: Path, *, executed=True, fuel_probe=True,
                       versions_match=True, extra_directory=False):
    runtime = root / "runtime"
    package = runtime / "wasmtime"
    native_dir = package / "linux-x86_64"
    metadata_dir = runtime / "wasmtime-99.1.0.dist-info"
    native_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    files = {
        "runtime/wasmtime-99.1.0.dist-info/METADATA":
            b"Metadata-Version: 2.1\nName: wasmtime\nVersion: 99.1.0\n",
        "runtime/wasmtime-99.1.0.dist-info/RECORD": b"pinned-record\n",
        "runtime/wasmtime/__init__.py": b"# pinned Wasmtime binding\n",
        "runtime/wasmtime/_bindings.py": b"# generated binding\n",
        "runtime/wasmtime/linux-x86_64/libwasmtime.so": b"ELF-fixture-not-run\n",
    }
    module_paths = sorted(["runtime/wasmtime/__init__.py",
                           "runtime/wasmtime/_bindings.py"])
    native_paths = ["runtime/wasmtime/linux-x86_64/libwasmtime.so"]
    metadata_path = "runtime/wasmtime-99.1.0.dist-info/METADATA"
    rows = []
    for logical, payload in sorted(files.items()):
        path = root.joinpath(*logical.split("/"))
        path.write_bytes(payload)
        if logical in module_paths:
            kind = "python_module"
        elif logical in native_paths:
            kind = "native_library"
        elif logical == metadata_path:
            kind = "metadata"
        else:
            kind = "resource"
        rows.append({"path": logical, "bytes": len(payload),
                     "sha256": auth.sha256(payload), "kind": kind})
    if extra_directory:
        (runtime / "unmanifested-empty").mkdir()
    by_path = {row["path"]: row for row in rows}
    tree_root = auth._row_root(rows)
    module_root = auth._row_root([by_path[path] for path in module_paths])
    native_root = auth._row_root([by_path[path] for path in native_paths])
    runtime_version = "99.1.0" if versions_match else "99.1.1"
    capability = {
        "schema": "strata-rm-global-swap-v4-wasmtime-runtime-capability",
        "distribution_name": "wasmtime",
        "python_distribution_version": "99.1.0",
        "wasmtime_runtime_version": runtime_version,
        "python_abi": "cp312", "platform_tag": "manylinux_2_28_x86_64",
        "target": "x86_64-unknown-linux-gnu",
        "module_entry_relative_path": "runtime/wasmtime/__init__.py",
        "metadata_relative_path": metadata_path,
        "runtime_tree_root_sha256": tree_root,
        "module_tree_root_sha256": module_root,
        "native_library_root_sha256": native_root,
        "python_module_files": module_paths,
        "native_libraries": native_paths,
        "engine_limits": auth._runtime_limits(),
    }
    capability_sha = canonical_file(root / "RUNTIME_CAPABILITY.json", capability)
    audit_path = root / "audit_runtime.py"
    audit_path.write_bytes(b"# independent runtime audit source\n")
    members = [member_row(audit_path)]
    source_root = auth._row_root(members)
    receipt = {
        "schema": "strata-rm-global-swap-v4-wasmtime-runtime-audit-receipt",
        "executed": executed,
        "status": "PASS_PINNED_WASMTIME_RUNTIME_AUDIT_V4",
        "audit_source_root_sha256": source_root,
        "runtime_capability_sha256": capability_sha,
        "runtime_tree_root_sha256": tree_root,
        "python_distribution_version_observed": "99.1.0",
        "wasmtime_runtime_version_observed": runtime_version,
        "module_tree_rehashed": True,
        "native_libraries_loaded_and_rehashed": True,
        "module_origin_from_snapshot": True,
        "target_observed": capability["target"],
        "engine_compile_probe": True,
        "store_memory_limit_probe": True,
        "fuel_exhaustion_probe": fuel_probe,
        "hostile_tests": 12, "payloads_opened": 0,
    }
    receipt_sha = canonical_file(root / "AUDIT_RECEIPT.json", receipt)
    manifest = {
        "schema": "strata-rm-global-swap-v4-wasmtime-runtime-audit-manifest",
        "source_root_sha256": source_root,
        "receipt_name": "AUDIT_RECEIPT.json",
        "capability_name": "RUNTIME_CAPABILITY.json",
        "capability_sha256": capability_sha,
        "runtime_tree_root_sha256": tree_root,
        "members": members, "runtime_files": rows,
    }
    manifest_sha = canonical_file(root / "source_manifest.json", manifest)
    return {"expected_manifest_sha256": manifest_sha,
            "expected_source_root_sha256": source_root,
            "expected_receipt_sha256": receipt_sha,
            "expected_capability_sha256": capability_sha,
            "expected_runtime_tree_root_sha256": tree_root}


def make_semantic_audit(root: Path, sandbox_sha: str, *, executed=True,
                        raw_packet=False, alias_rejection=True,
                        packet_capability=False, same_modules=False):
    decoder = b"\x00asm\x01\x00\x00\x00decoder"
    encoder = decoder if same_modules else b"\x00asm\x01\x00\x00\x00encoder"
    (root / "DECODER.wasm").write_bytes(decoder)
    (root / "CANONICAL_ENCODER.wasm").write_bytes(encoder)
    schema = {
        "schema": "strata-rm-v4-decoded-semantic-state", "version": 1,
        "state_fields": ["header", "quantizer_decisions", "centroids"],
        "raw_packet_bytes_permitted": raw_packet,
        "complete_quantizer_decisions": True,
        "canonical_field_order": ["header", "quantizer_decisions", "centroids"],
        "maximum_state_bytes_formula":
            "min(8*packet_bytes+1048576,536870912)",
    }
    schema_sha = canonical_file(root / "SEMANTIC_SCHEMA.json", schema)
    audit_path = root / "audit_semantics.py"
    audit_path.write_bytes(b"# independent semantic audit source\n")
    members = [member_row(audit_path)]
    source_root = auth._row_root(members)
    decoder_sha = auth.sha256(decoder)
    encoder_sha = auth.sha256(encoder)
    receipt = {
        "schema": "strata-rm-global-swap-v4-semantic-decoder-audit-receipt",
        "executed": executed,
        "status": "PASS_INDEPENDENT_SEMANTIC_CANONICALITY_AUDIT_V4",
        "audit_source_root_sha256": source_root,
        "decoder_sha256": decoder_sha,
        "canonical_encoder_sha256": encoder_sha,
        "semantic_schema_sha256": schema_sha,
        "sandbox_sha256": sandbox_sha,
        "decoder_only_safe_packet_import": True,
        "canonical_encoder_zero_imports": True,
        "semantic_decode_complete": True,
        "semantic_state_excludes_raw_packet": not raw_packet,
        "canonical_encoder_independent_from_decoder": not same_modules,
        "canonical_encoder_no_packet_capability": not packet_capability,
        "causal_decisions_regenerated": True,
        "complete_packet_consumption_verified": True,
        "trailing_bytes_rejected": True,
        "noncanonical_alias_rejection_verified": alias_rejection,
        "decode_then_independent_encode_verified": True,
        "canonical_uniqueness_verified": True,
        "fixed_universal_swiglu_moe_decoder": True,
        "qwen_specific_tables_absent": True,
        "hostile_tests": 20, "payloads_opened": 0,
    }
    receipt_sha = canonical_file(root / "AUDIT_RECEIPT.json", receipt)
    manifest = {
        "schema": "strata-rm-global-swap-v4-semantic-decoder-audit-manifest",
        "source_root_sha256": source_root,
        "receipt_name": "AUDIT_RECEIPT.json",
        "decoder_name": "DECODER.wasm", "decoder_sha256": decoder_sha,
        "canonical_encoder_name": "CANONICAL_ENCODER.wasm",
        "canonical_encoder_sha256": encoder_sha,
        "semantic_schema_name": "SEMANTIC_SCHEMA.json",
        "semantic_schema_sha256": schema_sha,
        "sandbox_sha256": sandbox_sha, "members": members,
    }
    manifest_sha = canonical_file(root / "source_manifest.json", manifest)
    return {"expected_manifest_sha256": manifest_sha,
            "expected_source_root_sha256": source_root,
            "expected_receipt_sha256": receipt_sha,
            "expected_decoder_sha256": decoder_sha,
            "expected_canonical_encoder_sha256": encoder_sha,
            "expected_semantic_schema_sha256": schema_sha,
            "expected_sandbox_sha256": sandbox_sha}


def sandbox_receipt(packet: bytes, runtime: dict, semantic: dict,
                    operations: list[dict[str, int]]) -> dict:
    page_count = (len(packet) + auth.PAGE_BYTES - 1) // auth.PAGE_BYTES
    fuel = min(auth.MAX_FUEL, auth.FUEL_BASE +
               auth.FUEL_PER_PACKET_BYTE * len(packet))
    capability = runtime["capability"]
    return {
        "schema": "strata-rm-global-swap-v4-pinned-wasmtime-sandbox-receipt",
        "route_id": "route-0", "sandbox_sha256": "a" * 64,
        "runtime_capability_sha256": runtime["capability_sha256"],
        "runtime_tree_root_sha256": runtime["runtime_tree_root_sha256"],
        "python_distribution_version": capability["python_distribution_version"],
        "wasmtime_runtime_version": capability["wasmtime_runtime_version"],
        "module_tree_root_sha256": capability["module_tree_root_sha256"],
        "native_library_root_sha256": capability["native_library_root_sha256"],
        "native_libraries_loaded": capability["native_libraries"],
        "decoder_imports": [{"module": "authority", "name": "read_packet",
                             "kind": "func"}],
        "canonical_encoder_imports": [], "wasi_enabled": False,
        "store_limits_installed_before_instantiation": True,
        "store_memory_limit_bytes": auth.STORE_MEMORY_LIMIT_BYTES,
        "fuel_budget": fuel, "decoder_fuel_remaining": fuel - 1,
        "encoder_fuel_remaining": fuel - 1,
        "packet_host_buffer_immutable": True,
        "packet_capability": "read-only bounded host callback",
        "packet_sha256": auth.sha256(packet), "packet_bytes": len(packet),
        "packet_read_operations": operations,
        "literal_bytes_supplied_total": sum(row["length"] for row in operations),
        "unique_literal_bytes_supplied": len(packet),
        "pages_supplied": list(range(page_count)),
        "physical_page_bytes_supplied": page_count * auth.PAGE_BYTES,
        "semantic_state_bytes": 100,
        "semantic_schema_sha256": semantic["semantic_schema_sha256"],
        "canonical_encoder_received_packet_capability": False,
        "canonical_packet_bytes": len(packet),
        "canonical_packet_sha256": auth.sha256(packet), "decode_status": 0,
        "status": "PASS_PINNED_BOUNDED_IMMUTABLE_SEMANTIC_WASM_DECODE"}


class V4AuthorityTests(unittest.TestCase):
    def test_01_v3_and_review_roots_are_exactly_pinned(self):
        self.assertEqual(auth.V3_SOURCE_ROOT_SHA256,
                         "83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad")
        self.assertEqual(auth.V3_REVIEW_SOURCE_ROOT_SHA256,
                         "3113631a5c64255d919f2bb5c545436452c8a721eb4130fcd32d7ffc4b2cdfe0")

    def test_02_package_root_symlink_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual"
            actual.mkdir()
            linked = root / "linked"
            try:
                linked.symlink_to(actual, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(auth.AuthorityError):
                auth.real_directory(linked, "test root")

    def test_03_complete_runtime_audit_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_runtime_audit(root)
            result = auth.authenticate_runtime_audit_package(root, **pins)
            self.assertEqual(result["status"],
                             "PASS_SEPARATELY_PINNED_WASMTIME_DISTRIBUTION_AND_RUNTIME")

    def test_04_runtime_distribution_and_runtime_versions_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_runtime_audit(root, versions_match=False)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_runtime_audit_package(root, **pins)

    def test_05_runtime_fuel_exhaustion_probe_is_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_runtime_audit(root, fuel_probe=False)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_runtime_audit_package(root, **pins)

    def test_06_unexecuted_runtime_receipt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_runtime_audit(root, executed=False)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_runtime_audit_package(root, **pins)

    def test_07_runtime_native_mutation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_runtime_audit(root)
            (root / "runtime/wasmtime/linux-x86_64/libwasmtime.so").write_bytes(
                b"mutated\n")
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_runtime_audit_package(root, **pins)

    def test_08_unmanifested_runtime_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_runtime_audit(root, extra_directory=True)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_runtime_audit_package(root, **pins)

    def test_09_semantic_decoder_and_encoder_audit_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_semantic_audit(root, "b" * 64)
            result = auth.authenticate_semantic_decoder_audit_package(root, **pins)
            self.assertEqual(result["status"],
                             "PASS_SEPARATELY_PINNED_SEMANTIC_DECODER_AND_ENCODER_AUDIT")

    def test_10_same_decoder_and_encoder_binary_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_semantic_audit(root, "b" * 64, same_modules=True)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_semantic_decoder_audit_package(root, **pins)

    def test_11_semantic_state_may_not_contain_raw_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_semantic_audit(root, "b" * 64, raw_packet=True)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_semantic_decoder_audit_package(root, **pins)

    def test_12_noncanonical_alias_rejection_is_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_semantic_audit(root, "b" * 64, alias_rejection=False)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_semantic_decoder_audit_package(root, **pins)

    def test_13_encoder_packet_capability_is_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_semantic_audit(root, "b" * 64, packet_capability=True)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_semantic_decoder_audit_package(root, **pins)

    def test_14_unexecuted_semantic_receipt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_semantic_audit(root, "b" * 64, executed=False)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_semantic_decoder_audit_package(root, **pins)

    def test_15_route_packet_aliases_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commitment.json"
            record = {
                "schema": "strata-rm-global-swap-v4-routed-expert-physical-commitment",
                "mode": "production_routed_expert",
                "v3_source_root_sha256": auth.V3_SOURCE_ROOT_SHA256,
                "v3_review_source_root_sha256": auth.V3_REVIEW_SOURCE_ROOT_SHA256,
                "runtime_capability_sha256": "1" * 64,
                "decoder_sha256": "2" * 64,
                "canonical_encoder_sha256": "3" * 64,
                "semantic_schema_sha256": "4" * 64,
                "sandbox_sha256": "5" * 64,
                "route_packets": [
                    {"route_id": "one", "relative_path": "one.bin",
                     "bytes": 9, "sha256": "6" * 64},
                    {"route_id": "two", "relative_path": "two.bin",
                     "bytes": 9, "sha256": "6" * 64}]}
            pin = canonical_file(path, record)
            with self.assertRaises(auth.AuthorityError):
                auth._strict_commitment(path, pin)

    def test_16_exactly_once_packet_callback_ledger_succeeds(self):
        packet = bytes(5000)
        runtime = {"capability_sha256": "1" * 64,
                   "runtime_tree_root_sha256": "2" * 64,
                   "capability": {
                       "python_distribution_version": "99.1.0",
                       "wasmtime_runtime_version": "99.1.0",
                       "module_tree_root_sha256": "3" * 64,
                       "native_library_root_sha256": "4" * 64,
                       "native_libraries": ["runtime/libwasmtime.so"]}}
        semantic = {"semantic_schema_sha256": "5" * 64}
        operations = [{"offset": 0, "length": 173},
                      {"offset": 173, "length": len(packet) - 173}]
        receipt = sandbox_receipt(packet, runtime, semantic, operations)
        result = auth._validate_sandbox_receipt(
            receipt, route_id="route-0", packet=packet, runtime=runtime,
            semantic=semantic, sandbox_sha="a" * 64)
        self.assertLess(result["cold_read_amplification"], 2.0)

    def test_17_overlapping_packet_callback_reads_fail(self):
        packet = bytes(5000)
        runtime = {"capability_sha256": "1" * 64,
                   "runtime_tree_root_sha256": "2" * 64,
                   "capability": {
                       "python_distribution_version": "99.1.0",
                       "wasmtime_runtime_version": "99.1.0",
                       "module_tree_root_sha256": "3" * 64,
                       "native_library_root_sha256": "4" * 64,
                       "native_libraries": ["runtime/libwasmtime.so"]}}
        semantic = {"semantic_schema_sha256": "5" * 64}
        operations = [{"offset": 0, "length": 3000},
                      {"offset": 2000, "length": 3000}]
        receipt = sandbox_receipt(packet, runtime, semantic, operations)
        with self.assertRaises(auth.AuthorityError):
            auth._validate_sandbox_receipt(
                receipt, route_id="route-0", packet=packet, runtime=runtime,
                semantic=semantic, sandbox_sha="a" * 64)

    def test_18_sandbox_has_preinstantiation_limits_fuel_and_host_callback(self):
        source = (PACKAGE / "wasm_runtime_sandbox.py").read_text("utf-8")
        configure = source.index("def configure_store")
        instantiate = source.index("decoder_instance = linker.instantiate")
        self.assertLess(configure, instantiate)
        self.assertIn("config.consume_fuel = True", source)
        self.assertIn("store.set_limits(memory_size=STORE_MEMORY_LIMIT_BYTES", source)
        self.assertIn("store.set_fuel(fuel)", source)
        self.assertIn("def read_packet(caller, offset, destination, length)", source)
        self.assertIn("packet_payload = regular_bytes", source)
        self.assertNotIn("memory.write(store, packet_payload", source)
        self.assertNotIn("linker.define_wasi", source)

    def test_19_independent_encoder_has_no_packet_callback(self):
        source = (PACKAGE / "wasm_runtime_sandbox.py").read_text("utf-8")
        encoder_section = source[source.index("encoder_store = configure_store"):]
        self.assertIn("wasmtime.Instance(encoder_store, encoder_module, [])",
                      encoder_section)
        self.assertNotIn("linker.define(encoder_store", encoder_section)
        self.assertNotIn("packet_payload, encoder_semantic_offset", encoder_section)

    def test_20_production_requires_explicit_v4_authorization(self):
        with self.assertRaises(auth.AuthorityError):
            auth.validate_physical_bundle(
                v4_package=Path("missing"), expected_v4_manifest_sha256="0" * 64,
                v3_package=Path("missing"), v3_review_package=Path("missing"),
                evidence_root=Path("missing"), commitment_path=Path("missing"),
                expected_commitment_sha256="0" * 64,
                scientific_audit_package=Path("missing"),
                expected_scientific_manifest_sha256="0" * 64,
                expected_scientific_source_root_sha256="0" * 64,
                expected_scientific_receipt_sha256="0" * 64,
                expected_scientific_capability_sha256="0" * 64,
                runtime_audit_package=Path("missing"),
                expected_runtime_manifest_sha256="0" * 64,
                expected_runtime_source_root_sha256="0" * 64,
                expected_runtime_receipt_sha256="0" * 64,
                expected_runtime_capability_sha256="0" * 64,
                expected_runtime_tree_root_sha256="0" * 64,
                semantic_decoder_audit_package=Path("missing"),
                expected_decoder_manifest_sha256="0" * 64,
                expected_decoder_source_root_sha256="0" * 64,
                expected_decoder_receipt_sha256="0" * 64,
                expected_decoder_sha256="0" * 64,
                expected_canonical_encoder_sha256="0" * 64,
                expected_semantic_schema_sha256="0" * 64,
                authorization="WRONG")


if __name__ == "__main__":
    unittest.main(verbosity=2)
