#!/usr/bin/env python3
"""Independent benign source tests for the frozen v4 authority.

These tests authenticate source and exercise only synthetic temporary audit
packages. They never import Wasmtime, open a packet/model path, or connect to a
network. Some tests intentionally document conditional trust boundaries.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
PRODUCER = REVIEW.parent / "strata_rm_global_swap_v4_wasm_authority"
V3 = REVIEW.parent / "strata_rm_global_swap_v3_physical_authority"
V3_REVIEW = (
    REVIEW.parent /
    "strata_rm_global_swap_v3_physical_authority_independent_source_review_20260902")
EXPECTED_MANIFEST = "62bf04cd413317e2e8b98635713419c84394db7b7d2bd4567afddf56957a5e2f"
EXPECTED_ROOT = "f535699c4828a02e5769b916b1207309768f7381db5f92a0fb58e10915ae8a25"
EXPECTED_V3_MANIFEST = "9105dd69a2a82d1eaf14e176e4334189a4c31be840dafee467d243c231788e83"
EXPECTED_V3_REVIEW_MANIFEST = (
    "ebe65fcf1abd73263be0176cdb70244ebca4f0a883eb6815c24c8956b0d0d89c")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auth = load_module("reviewed_v4_authority", PRODUCER / "authority_v4.py")
fixtures = load_module("reviewed_v4_fixture_helpers", PRODUCER / "test_source_only.py")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class IndependentV4SourceReview(unittest.TestCase):
    def test_01_exact_producer_manifest_and_canonical_root(self):
        manifest_payload = (PRODUCER / "source_manifest.json").read_bytes()
        self.assertEqual(sha(manifest_payload), EXPECTED_MANIFEST)
        manifest = json.loads(manifest_payload)
        observed = []
        for row in manifest["members"]:
            payload = (PRODUCER / row["name"]).read_bytes()
            item = {"name": row["name"], "bytes": len(payload),
                    "sha256": sha(payload)}
            self.assertEqual(item, row)
            observed.append(item)
        self.assertEqual(sha(json.dumps(
            observed, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")), EXPECTED_ROOT)
        self.assertEqual({path.name for path in PRODUCER.iterdir()},
                         {row["name"] for row in observed} |
                         {"source_manifest.json"})
        self.assertEqual(len(observed), 9)

    def test_02_v3_and_review_lineage_are_literal(self):
        self.assertEqual(sha((V3 / "source_manifest.json").read_bytes()),
                         EXPECTED_V3_MANIFEST)
        self.assertEqual(sha((V3_REVIEW / "source_manifest.json").read_bytes()),
                         EXPECTED_V3_REVIEW_MANIFEST)
        self.assertEqual(auth.V3_SOURCE_ROOT_SHA256,
                         "83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad")
        self.assertEqual(auth.V3_REVIEW_SOURCE_ROOT_SHA256,
                         "3113631a5c64255d919f2bb5c545436452c8a721eb4130fcd32d7ffc4b2cdfe0")

    def test_03_runtime_audit_closes_recursive_distribution_tree(self):
        source = (PRODUCER / "authority_v4.py").read_text("utf-8")
        for phrase in (
                "runtime package exact recursive file closure",
                "runtime package exact recursive directory closure",
                "complete runtime module/native inventories",
                "runtime distribution METADATA version",
                "native_libraries_loaded_and_rehashed",
                "fuel_exhaustion_probe"):
            self.assertIn(phrase, source)

    def test_04_limits_and_fuel_precede_both_instantiations(self):
        source = (PRODUCER / "wasm_runtime_sandbox.py").read_text("utf-8")
        self.assertLess(source.index("config.consume_fuel = True"),
                        source.index("decoder_module = wasmtime.Module"))
        self.assertIn("store.set_limits(memory_size=STORE_MEMORY_LIMIT_BYTES", source)
        self.assertIn("store.set_fuel(fuel)", source)
        self.assertLess(source.index("decoder_store = configure_store"),
                        source.index("decoder_instance = linker.instantiate"))
        self.assertLess(source.index("encoder_store = configure_store"),
                        source.index("encoder_instance = wasmtime.Instance"))

    def test_05_packet_is_host_bytes_behind_bounded_exactly_once_callback(self):
        source = (PRODUCER / "wasm_runtime_sandbox.py").read_text("utf-8")
        for phrase in (
                "packet_payload = regular_bytes(args.packet",
                "def read_packet(caller, offset, destination, length)",
                "offset + length <= len(packet_payload)",
                "for left, right in read_intervals",
                "memory.write(caller, packet_payload[offset:end], destination)",
                "decoder consumed every literal packet byte exactly once"):
            self.assertIn(phrase, source)
        self.assertNotIn("linker.define_wasi", source)

    def test_06_encoder_is_distinct_zero_import_and_has_no_direct_callback(self):
        source = (PRODUCER / "wasm_runtime_sandbox.py").read_text("utf-8")
        section = source[source.index("encoder_store = configure_store"):]
        self.assertIn("encoder_import_list == []", source)
        self.assertIn("wasmtime.Instance(encoder_store, encoder_module, [])", section)
        self.assertNotIn("linker.define(encoder_store", section)
        authority = (PRODUCER / "authority_v4.py").read_text("utf-8")
        self.assertIn("expected_decoder_sha256 != expected_canonical_encoder_sha256",
                      authority)

    def test_07_semantic_state_is_opaque_at_runtime(self):
        source = (PRODUCER / "wasm_runtime_sandbox.py").read_text("utf-8")
        start = source.index("semantic_state = bytes(decoder_memory.read")
        end = source.index(
            "encoder_memory.write(encoder_store, semantic_state", start)
        bridge = source[start:end]
        self.assertNotIn("strict_json(semantic_state", bridge)
        self.assertNotIn("raw_packet_bytes_permitted", bridge)
        self.assertIn("semantic_state", bridge)

    def test_08_caller_authored_runtime_attestation_is_conditionally_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = fixtures.make_runtime_audit(root)
            result = auth.authenticate_runtime_audit_package(root, **pins)
            self.assertEqual(result["status"],
                             "PASS_SEPARATELY_PINNED_WASMTIME_DISTRIBUTION_AND_RUNTIME")
            # The fixture's native object is text, never a runnable ELF. This
            # proves the authenticator trusts the externally pinned receipt;
            # it does not itself execute the audit or native object.
            native = root / "runtime/wasmtime/linux-x86_64/libwasmtime.so"
            self.assertEqual(native.read_bytes(), b"ELF-fixture-not-run\n")

    def test_09_caller_authored_invalid_wasm_attestation_is_conditionally_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = fixtures.make_semantic_audit(root, "b" * 64)
            result = auth.authenticate_semantic_decoder_audit_package(root, **pins)
            self.assertEqual(result["status"],
                             "PASS_SEPARATELY_PINNED_SEMANTIC_DECODER_AND_ENCODER_AUDIT")
            # These fixture bytes are not independently compiled or exercised
            # by the authenticator. External pin provenance is mandatory.
            self.assertTrue(result["decoder_payload"].endswith(b"decoder"))
            self.assertTrue(result["canonical_encoder_payload"].endswith(b"encoder"))

    def test_10_current_host_abi_platform_and_target_are_not_reobserved(self):
        source = (PRODUCER / "wasm_runtime_sandbox.py").read_text("utf-8")
        self.assertIn('"python_abi"', source)
        self.assertIn('"platform_tag"', source)
        self.assertIn('"target"', source)
        self.assertNotIn("sys.implementation.cache_tag", source)
        self.assertNotIn("platform.machine", source)
        self.assertNotIn("sysconfig.get_platform", source)

    def test_11_native_mapping_check_is_snapshot_scoped(self):
        source = (PRODUCER / "wasm_runtime_sandbox.py").read_text("utf-8")
        self.assertIn("expected <= observed", source)
        self.assertIn("loaded_from_snapshot == expected", source)
        self.assertNotIn("observed == expected", source)

    def test_12_v3_scientific_and_per_family_acceptance_is_reused(self):
        source = (PRODUCER / "authority_v4.py").read_text("utf-8")
        self.assertIn(
            'v3.evaluate_acceptance(results, scientific["record"], enforce=True)',
            source)
        v3_source = (V3 / "authority_v3.py").read_text("utf-8")
        for phrase in (
                "for family in scientific_record[\"architecture_families\"]",
                "RATE_MIN <= model_pool[\"physical_rate_bpw\"] <= RATE_MAX",
                "model_pool[\"F\"] <= TARGET_F",
                "advantage >= MIN_SOURCE_SPECIFIC_BPW",
                "qwen[\"F\"] <= TARGET_F"):
            self.assertIn(phrase, v3_source)

    def test_13_distinct_expert_packets_and_narrow_page_ledger_preserved(self):
        source = (PRODUCER / "authority_v4.py").read_text("utf-8")
        self.assertIn("distinct expert packet per route", source)
        self.assertIn("one packet for every audited route", source)
        self.assertIn("physical_page_bytes", source)
        self.assertIn("cold_read_amplification", source)
        # The actual harness also reads the evidence packet, writes a snapshot,
        # and the child rereads it. Those transfers are outside the reported
        # callback page ratio.
        self.assertIn("packet = _read_pinned", source)
        self.assertIn("_write_immutable(packet_path, packet", source)
        sandbox = (PRODUCER / "wasm_runtime_sandbox.py").read_text("utf-8")
        self.assertIn("regular_bytes(args.packet", sandbox)

    def test_14_source_gate_has_no_payload_or_runtime_inputs(self):
        tree = ast.parse((PRODUCER / "run_source_gate.py").read_text("utf-8"))
        flags = {node.args[0].value for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute) and
                 node.func.attr == "add_argument" and node.args and
                 isinstance(node.args[0], ast.Constant) and
                 isinstance(node.args[0].value, str)}
        self.assertEqual(flags, {"--package", "--expected-manifest-sha256",
                                 "--v3-package", "--review-package", "--output"})

    def test_15_no_network_client_imports(self):
        forbidden = {"socket", "requests", "urllib", "http", "ftplib",
                     "paramiko", "asyncssh"}
        for name in ("authority_v4.py", "wasm_runtime_sandbox.py",
                     "run_source_gate.py", "verify_source.py"):
            tree = ast.parse((PRODUCER / name).read_text("utf-8"))
            roots = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".")[0])
            self.assertFalse(roots & forbidden, name)

    def test_16_runtime_payload_and_rd_are_explicitly_held(self):
        status = json.loads((PRODUCER / "EXECUTION_STATUS.json").read_bytes())
        self.assertFalse(status["wasmtime_runtime_imported"])
        self.assertFalse(status["wasm_guest_executed"])
        self.assertFalse(status["model_data_accessed"])
        self.assertEqual(status["payloads_opened"], 0)
        self.assertIsNone(status["qwen_result"])
        self.assertIsNone(status["physical_result"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

