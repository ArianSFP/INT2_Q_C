#!/usr/bin/env python3
"""Adversarial source-only tests for the N18 v3 closure repair."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dependency_auth
import dispatcher_contract
import immutable_bootstrap
import runtime_auth
import safe_telemetry
import secure_io
import universal_layout
from immutable_bootstrap import BootstrapError
from v3_common import (
    ContractError,
    MAX_SOURCE_MEMBER_BYTES,
    MICRO,
    N18,
    N18_COARSE_RESERVOIR_BYTES,
    PAGE_BYTES,
    canonical_json,
)
from verify_source import source_root, verify_packets


_AUTH_CONTEXT: Mapping[str, Any] | None = None


def _context() -> Mapping[str, Any]:
    if _AUTH_CONTEXT is None:
        raise AssertionError("authenticated_main did not install context")
    return _AUTH_CONTEXT


def _packets() -> Mapping[str, bytes]:
    return _context()["packets"]


def _inventory_value() -> dict[str, Any]:
    rows = []
    for name in sorted(_packets(), key=lambda value: value.encode("utf-8")):
        packet = _packets()[name]
        rows.append({"name": name, "bytes": len(packet), "sha256": hashlib.sha256(packet).hexdigest()})
    return {
        "schema": immutable_bootstrap.INVENTORY_SCHEMA,
        "status": "EXTERNAL_INVENTORY_NO_EXECUTION_AUTHORITY",
        "files": rows,
        "authority_boundary": "test inventory only; not an execution or review authority",
    }


def _runtime_value() -> dict[str, Any]:
    nonzero = "1" * 64
    root = os.path.abspath("runtime-test-root")
    rows = []
    for name in runtime_auth.REQUIRED_DISTRIBUTIONS:
        rows.append(
            {
                "name": name,
                "version": "1.2.3",
                "installation_root": root,
                "record_path": f"{name}.dist-info/RECORD",
                "record_bytes": 99,
                "record_sha256": nonzero,
                "tree_files": 3,
                "tree_bytes": 1234,
                "tree_sha256": nonzero,
            }
        )
    return {
        "schema": "tactic_actual_coarse_n18_runtime_lock_v3",
        "status": "FROZEN_EXTERNAL_RUNTIME_AUTHORITY",
        "lock_id": "external-test-lock",
        "interpreter": {
            "absolute_path": os.path.abspath(sys.executable),
            "bytes": 1234,
            "sha256": nonzero,
            "python_version": "3.12-test",
        },
        "distributions": rows,
        "tree_algorithm": "SHA256 domain + bytewise metadata-path + uint64 bytes + file SHA256; every importlib.metadata file",
        "claim_boundary": "schema fixture only",
    }


def _telemetry_value() -> dict[str, Any]:
    nonzero = "2" * 64
    value = {
        "schema": "tactic_actual_coarse_n18_telemetry_v3",
        "authenticated_source_root": nonzero,
        "runtime_lock_sha256": nonzero,
        "cuda_visible_devices": "0",
        "cuda_logical_index": 0,
        "cuda_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "cuda_pci_bus_id": "0000:01:00.0",
        "nvml_physical_index": 0,
        "nvml_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "nvml_pci_bus_id": "0000:01:00.0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "driver_version": "600.1",
        "cuda_runtime_version": "12.8",
        "cupy_version": "14.0",
        "numpy_version": "2.2",
        "scipy_version": "1.15",
        "pynvml_version": "12.0",
        "logical_h2d_bytes": 100,
        "logical_d2h_bytes": 20,
        "model_h2d_bytes": 100,
        "transfers": [
            {"ordinal": 0, "direction": "h2d", "label": "model", "bytes": 100, "model": True, "buffer_sha256": nonzero},
            {"ordinal": 1, "direction": "d2h", "label": "score", "bytes": 20, "model": False, "buffer_sha256": nonzero},
        ],
        "kernel_launches": 2,
        "cuda_events_synchronized": True,
        "cuda_event_h2d_ms": 1.0,
        "cuda_event_kernel_ms": 2.0,
        "cuda_event_d2h_ms": 1.0,
        "wall_seconds": 0.01,
        "telemetry_sampling_interval_ms": 2.0,
        "transfer_definition": "exact logical nbytes of every enumerated host/device buffer; not claimed physical PCIe traffic",
        "sampling_limit": "RSS/NVML peaks are sampled and may miss sub-interval transients; logical transfer totals are exact",
    }
    for prefix in ("host_rss", "nvml_process", "nvml_device", "cupy_pool"):
        value[f"{prefix}_baseline_bytes"] = 10
        value[f"{prefix}_peak_bytes"] = 15
        value[f"{prefix}_delta_bytes"] = 5
    return value


class ArithmeticAndLayoutTests(unittest.TestCase):
    def test_exact_n18_and_dh384_handoff(self) -> None:
        self.assertEqual(64 * MICRO, N18)
        self.assertEqual(64 * 1228, N18_COARSE_RESERVOIR_BYTES)
        self.assertEqual(8 * 1228 / MICRO, 307 / 128)
        self.assertEqual(1228 + 48 + 4, 1280)
        self.assertEqual(8 * 1280 / MICRO, 2.5)

    def test_qwen_reference_geometry(self) -> None:
        receipt = universal_layout.qwen_evaluation_ledger()
        self.assertEqual(receipt["coarse_bpw"], 307 / 128)
        self.assertEqual(receipt["final_bpw"], 2.5)
        self.assertEqual(receipt["n18_streams_per_expert"], 18)
        self.assertEqual(receipt["maximum_unique_page_read_amplification"], 1.0)

    def test_odd_769x2051_tail_is_rate_and_page_legal(self) -> None:
        geometry = universal_layout.ExpertGeometry(769, 2051)
        partition = universal_layout.partition_expert(geometry)
        panel = universal_layout.layout_panel([geometry])
        self.assertTrue(partition.explicit)
        self.assertGreater(partition.tail_values, 0)
        self.assertLess(partition.tail_coarse_bytes, N18_COARSE_RESERVOIR_BYTES)
        self.assertLessEqual(16 * panel.physical_bytes, 5 * panel.weights)
        self.assertLess(panel.maximum_read_amplification, 2.0)
        owner = panel.owners[0]
        exact_pages = set(range(owner.first_page, owner.last_page + 1))
        self.assertEqual(owner.unique_pages, len(exact_pages))
        self.assertEqual(owner.unique_page_bytes, len(exact_pages) * PAGE_BYTES)

    def test_tiny_shapes_use_zero_byte_zero_read_fallback(self) -> None:
        for geometry in (
            universal_layout.ExpertGeometry(1, 1),
            universal_layout.ExpertGeometry(1, 4096),
            universal_layout.ExpertGeometry(7, 31),
        ):
            partition = universal_layout.partition_expert(geometry)
            panel = universal_layout.layout_panel([geometry])
            self.assertFalse(partition.explicit)
            self.assertEqual(partition.physical_frame_bytes, 0)
            self.assertEqual(panel.physical_bytes, 0)
            self.assertEqual(panel.owners[0].unique_pages, 0)
            self.assertEqual(panel.owners[0].read_amplification, 0.0)

    def test_explicit_threshold_all_page_offsets(self) -> None:
        partition = universal_layout.partition_expert(universal_layout.ExpertGeometry(1, 8738))
        self.assertEqual(partition.physical_frame_bytes, 8191)
        self.assertTrue(partition.explicit)
        for offset in range(PAGE_BYTES):
            owner = universal_layout._owner(0, partition, offset)
            expected = ((offset + owner.byte_length - 1) // PAGE_BYTES - offset // PAGE_BYTES + 1) * PAGE_BYTES
            self.assertEqual(owner.unique_page_bytes, expected)
            self.assertLess(owner.read_amplification, 2.0)

    def test_unequal_panel_uses_per_owner_page_union(self) -> None:
        shapes = [
            universal_layout.ExpertGeometry(1, 1),
            universal_layout.ExpertGeometry(769, 2051),
            universal_layout.ExpertGeometry(768, 2048),
            universal_layout.ExpertGeometry(17, 8191),
        ]
        panel = universal_layout.layout_panel(shapes)
        self.assertLessEqual(16 * panel.physical_bytes, 5 * panel.weights)
        for owner in panel.owners:
            if owner.byte_length == 0:
                self.assertEqual(owner.unique_page_bytes, 0)
            else:
                page_union = set(range(owner.first_page, owner.last_page + 1))
                self.assertEqual(owner.unique_page_bytes, PAGE_BYTES * len(page_union))
                self.assertLess(owner.unique_page_bytes, 2 * owner.byte_length)

    def test_shape_bounds_before_products(self) -> None:
        with self.assertRaises(ContractError):
            universal_layout.ExpertGeometry(0, 1)
        with self.assertRaises(ContractError):
            universal_layout.ExpertGeometry(1 << 24, (1 << 24) + 1)
        with self.assertRaises(ContractError):
            universal_layout.layout_panel([])


class InventoryAndLoaderTests(unittest.TestCase):
    def test_valid_inventory_and_caps(self) -> None:
        value = _inventory_value()
        parsed = immutable_bootstrap.validate_inventory_bytes(canonical_json(value))
        self.assertEqual(len(parsed["files"]), len(immutable_bootstrap.EXPECTED_SOURCE_FILES))

    def test_inventory_rejects_traversal_duplicate_extra_and_aggregate(self) -> None:
        cases = []
        traversal = _inventory_value()
        traversal["files"][0]["name"] = "../escape.py"
        cases.append(traversal)
        duplicate = _inventory_value()
        duplicate["files"][1]["name"] = duplicate["files"][0]["name"]
        cases.append(duplicate)
        extra = _inventory_value()
        extra["files"].append({"name": "z.py", "bytes": 1, "sha256": "1" * 64})
        cases.append(extra)
        aggregate = _inventory_value()
        for row in aggregate["files"]:
            row["bytes"] = MAX_SOURCE_MEMBER_BYTES
        cases.append(aggregate)
        for value in cases:
            with self.subTest(kind=value["files"][0]["name"]):
                with self.assertRaises(BootstrapError):
                    immutable_bootstrap.validate_inventory_bytes(canonical_json(value))

    def test_inventory_rejects_bad_digest_name_and_duplicate_json_key(self) -> None:
        bad = _inventory_value()
        bad["files"][0]["sha256"] = "g" * 64
        with self.assertRaises(BootstrapError):
            immutable_bootstrap.validate_inventory_bytes(canonical_json(bad))
        long_name = _inventory_value()
        long_name["files"][0]["name"] = "a" * 129
        with self.assertRaises(BootstrapError):
            immutable_bootstrap.validate_inventory_bytes(canonical_json(long_name))
        with self.assertRaises(BootstrapError):
            immutable_bootstrap.validate_inventory_bytes(b'{"schema":1,"schema":2}')

    def test_authenticated_loader_prefers_packet_and_never_writes_pycache(self) -> None:
        module_name = "v3_loader_hostile_fixture"
        packet = b"VALUE = 7\n"
        packets = MappingProxyType({f"{module_name}.py": packet})
        finder = immutable_bootstrap.AuthenticatedBytesFinder(packets)
        old_dont_write = sys.dont_write_bytecode
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, f"{module_name}.py").write_text("VALUE = 99\n", encoding="utf-8")
            sys.path.insert(0, directory)
            sys.meta_path.insert(0, finder)
            sys.dont_write_bytecode = True
            try:
                module = importlib.import_module(module_name)
                self.assertEqual(module.VALUE, 7)
                self.assertTrue(module.__file__.startswith("<authenticated-bytes:"))
                self.assertIsNone(module.__cached__)
                self.assertEqual(module.__authenticated_source_sha256__, hashlib.sha256(packet).hexdigest())
                self.assertFalse(Path(directory, "__pycache__").exists())
            finally:
                sys.modules.pop(module_name, None)
                sys.meta_path.remove(finder)
                sys.path.remove(directory)
                sys.dont_write_bytecode = old_dont_write

    def test_executor_rejects_preloaded_sibling(self) -> None:
        trap = "v3_preloaded_trap"
        sys.modules[trap] = types.ModuleType(trap)
        try:
            with self.assertRaises(BootstrapError):
                immutable_bootstrap.execute_authenticated_entry(
                    MappingProxyType({f"{trap}.py": b"VALUE=1\n"}),
                    "1" * 64,
                    "2" * 64,
                    "verify",
                    [],
                )
        finally:
            sys.modules.pop(trap, None)

    def test_live_path_bootstrap_launch_is_rejected(self) -> None:
        with self.assertRaises(BootstrapError):
            immutable_bootstrap.authenticate_bootstrap_launch(3, "1" * 64)


class DependencyAndRuntimeTests(unittest.TestCase):
    def test_dependency_graph_valid(self) -> None:
        rows = dependency_auth.validate_dependency_graph(_packets()["dependency_graph.json"])
        self.assertEqual([row["id"] for row in rows], sorted(row["id"] for row in rows))

    def test_dependency_ids_paths_imports_and_count_are_hostile(self) -> None:
        base = json.loads(_packets()["dependency_graph.json"])
        mutations = []
        for identifier in ("../escape", "bad/id", "é", "a" * 65):
            value = copy.deepcopy(base)
            value["external_python_sources"][0]["id"] = identifier
            mutations.append(value)
        traversal = copy.deepcopy(base)
        traversal["external_python_sources"][0]["relative_path"] = "../escape.py"
        mutations.append(traversal)
        duplicate_path = copy.deepcopy(base)
        duplicate_path["external_python_sources"][1]["relative_path"] = duplicate_path["external_python_sources"][0]["relative_path"]
        mutations.append(duplicate_path)
        duplicate_import = copy.deepcopy(base)
        duplicate_import["external_python_sources"][0]["allowed_import_roots"].append("argparse")
        mutations.append(duplicate_import)
        too_many = copy.deepcopy(base)
        too_many["external_python_sources"] = [copy.deepcopy(base["external_python_sources"][0]) for _ in range(17)]
        mutations.append(too_many)
        for value in mutations:
            with self.assertRaises((ContractError, UnicodeError)):
                dependency_auth.validate_dependency_graph(canonical_json(value))

    def test_runtime_schema_accepts_only_real_positive_lock_shape(self) -> None:
        value = _runtime_value()
        parsed = runtime_auth.validate_runtime_lock_schema(canonical_json(value))
        self.assertEqual(tuple(row["name"] for row in parsed["distributions"]), runtime_auth.REQUIRED_DISTRIBUTIONS)

    def test_runtime_rejects_v2_fabricated_fields(self) -> None:
        base = _runtime_value()
        mutations: list[dict[str, Any]] = []
        for field, replacement in (("bytes", -1), ("bytes", 0), ("sha256", "0" * 64), ("python_version", "")):
            value = copy.deepcopy(base)
            value["interpreter"][field] = replacement
            mutations.append(value)
        for field, replacement in (
            ("version", ""),
            ("record_bytes", 0),
            ("record_sha256", "0" * 64),
            ("tree_files", 0),
            ("tree_bytes", -1),
            ("tree_sha256", "0" * 64),
        ):
            value = copy.deepcopy(base)
            value["distributions"][0][field] = replacement
            mutations.append(value)
        for value in mutations:
            with self.assertRaises(ContractError):
                runtime_auth.validate_runtime_lock_schema(canonical_json(value))

    def test_runtime_has_no_in_package_frozen_lock(self) -> None:
        self.assertFalse(any(name.startswith("runtime_lock") for name in _packets()))
        self.assertIn("importlib.metadata", _packets()["runtime_auth.py"].decode("utf-8"))
        self.assertIn("HeldRegularFile", _packets()["runtime_auth.py"].decode("utf-8"))


class TelemetryAndAuthorityTests(unittest.TestCase):
    def test_strict_telemetry_fixture(self) -> None:
        self.assertIs(safe_telemetry.validate_telemetry(_telemetry_value())["cuda_events_synchronized"], True)

    def test_telemetry_rejects_identity_transfer_timing_and_peak_fabrication(self) -> None:
        base = _telemetry_value()
        mutations = []
        for field, replacement in (
            ("authenticated_source_root", "0" * 64),
            ("runtime_lock_sha256", "0" * 64),
            ("cuda_uuid", ""),
            ("nvml_pci_bus_id", "0000:02:00.0"),
            ("cupy_version", ""),
            ("logical_h2d_bytes", -1),
            ("cuda_events_synchronized", False),
            ("wall_seconds", 0.001),
            ("telemetry_sampling_interval_ms", 0),
            ("host_rss_delta_bytes", 4),
        ):
            value = copy.deepcopy(base)
            value[field] = replacement
            mutations.append(value)
        duplicate = copy.deepcopy(base)
        duplicate["transfers"][1]["label"] = "model"
        mutations.append(duplicate)
        wrong_total = copy.deepcopy(base)
        wrong_total["logical_d2h_bytes"] = 21
        mutations.append(wrong_total)
        for value in mutations:
            with self.assertRaises(ContractError):
                safe_telemetry.validate_telemetry(value)

    def test_package_cannot_self_authorize_review(self) -> None:
        with self.assertRaises(RuntimeError):
            dispatcher_contract.no_standalone_authority()
        packet = _packets()["dispatcher_contract.py"].decode("utf-8")
        self.assertIn("external dispatcher", packet)
        self.assertNotIn("def issue_", packet)
        design = json.loads(_packets()["design_lock.json"])
        self.assertEqual(design["authority_contract"]["producer_authority"], "NONE")


@unittest.skipUnless(os.name == "posix", "POSIX no-follow/renameat2 fault tests are authored but not executable on this Windows review host")
class PosixSecurityAndPublicationTests(unittest.TestCase):
    def test_held_input_rejects_symlink_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            real = os.path.join(directory, "real")
            link = os.path.join(directory, "link")
            Path(real).write_bytes(b"abc")
            os.symlink(real, link)
            with self.assertRaises(ContractError):
                secure_io.HeldRegularFile(link, maximum_bytes=10)
            source = secure_io.HeldRegularFile(real, maximum_bytes=10)
            try:
                Path(real).write_bytes(b"abd")
                with self.assertRaises(ContractError):
                    source.verify()
            finally:
                source.close()

    def test_constructor_faults_leave_no_public_or_staging_tree(self) -> None:
        for target in (
            secure_io.PublicationPhase.PARENT_HELD,
            secure_io.PublicationPhase.STAGING_CREATED,
            secure_io.PublicationPhase.STAGING_HELD,
        ):
            with self.subTest(phase=target.value), tempfile.TemporaryDirectory() as directory:
                output = os.path.join(directory, "result")
                def fault(phase: secure_io.PublicationPhase, publisher: secure_io.VerifiedPublisher) -> None:
                    if phase is target:
                        raise RuntimeError("injected")
                with self.assertRaises(RuntimeError):
                    secure_io.VerifiedPublisher(output, "1" * 64, fault_hook=fault)
                self.assertFalse(os.path.exists(output))
                self.assertFalse(any(".staging." in name for name in os.listdir(directory)))

    def test_every_prepublication_fault_has_no_public_complete_tree(self) -> None:
        targets = (
            secure_io.PublicationPhase.WRITING,
            secure_io.PublicationPhase.INDEX_WRITTEN,
            secure_io.PublicationPhase.COMPLETE_WRITTEN,
            secure_io.PublicationPhase.STAGING_SYNCED,
            secure_io.PublicationPhase.STAGING_VERIFIED,
        )
        for target in targets:
            with self.subTest(phase=target.value), tempfile.TemporaryDirectory() as directory:
                output = os.path.join(directory, "result")
                def fault(phase: secure_io.PublicationPhase, publisher: secure_io.VerifiedPublisher) -> None:
                    if phase is target:
                        raise RuntimeError("injected")
                with self.assertRaises(RuntimeError):
                    with secure_io.VerifiedPublisher(output, "1" * 64, fault_hook=fault) as publisher:
                        publisher.write("payload.bin", b"payload")
                        publisher.complete({"ok": True})
                self.assertFalse(os.path.exists(output))
                self.assertFalse(any(".staging." in name for name in os.listdir(directory)))

    def test_corrupt_staging_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "result")
            def corrupt(phase: secure_io.PublicationPhase, publisher: secure_io.VerifiedPublisher) -> None:
                if phase is secure_io.PublicationPhase.STAGING_SYNCED:
                    path = os.path.join(directory, publisher.staging_name, "payload.bin")
                    os.chmod(path, 0o600)
                    Path(path).write_bytes(b"PAYLOAD")
            with self.assertRaises(ContractError):
                with secure_io.VerifiedPublisher(output, "1" * 64, fault_hook=corrupt) as publisher:
                    publisher.write("payload.bin", b"payload")
                    publisher.complete({"ok": True})
            self.assertFalse(os.path.exists(output))
            self.assertFalse(any(".staging." in name for name in os.listdir(directory)))

    def test_postrename_fault_can_only_leave_verified_complete_tree(self) -> None:
        for target in (secure_io.PublicationPhase.PUBLISHED, secure_io.PublicationPhase.PARENT_SYNCED):
            with self.subTest(phase=target.value), tempfile.TemporaryDirectory() as directory:
                output = os.path.join(directory, "result")
                def fault(phase: secure_io.PublicationPhase, publisher: secure_io.VerifiedPublisher) -> None:
                    if phase is target:
                        raise RuntimeError("injected")
                with self.assertRaises(RuntimeError):
                    with secure_io.VerifiedPublisher(output, "1" * 64, fault_hook=fault) as publisher:
                        publisher.write("payload.bin", b"payload")
                        publisher.complete({"ok": True})
                self.assertTrue(os.path.isfile(os.path.join(output, "COMPLETE.json")))
                index_raw = Path(output, "ARTIFACTS.json").read_bytes()
                complete = json.loads(Path(output, "COMPLETE.json").read_bytes())
                self.assertEqual(complete["artifact_index_sha256"], hashlib.sha256(index_raw).hexdigest())

    def test_success_publishes_complete_last_verified_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "result")
            with secure_io.VerifiedPublisher(output, "1" * 64) as publisher:
                publisher.write("payload.bin", b"payload")
                receipt = publisher.complete({"ok": True})
            self.assertEqual(receipt.phase, "closed")
            self.assertEqual(set(os.listdir(output)), {"payload.bin", "ARTIFACTS.json", "COMPLETE.json"})


class WholeSourceVerifierTests(unittest.TestCase):
    def test_authenticated_packet_verifier(self) -> None:
        receipt = verify_packets(_context())
        self.assertEqual(receipt["status"], "SOURCE_ONLY_VERIFIED_NOT_EXTERNALLY_SEALED")
        self.assertEqual(receipt["source_root"], _context()["source_root"])
        self.assertLessEqual(receipt["odd_769x2051"]["physical_bpw"], 2.5)
        self.assertEqual(receipt["tiny_1x1"]["physical_bytes"], 0)

    def test_no_forbidden_freeze_result_or_manifest_member(self) -> None:
        names = set(_packets())
        self.assertNotIn("SOURCE_MANIFEST.json", names)
        self.assertNotIn("runtime_lock.json", names)
        self.assertFalse(any(name.startswith("RESULT") for name in names))
        self.assertEqual(source_root(_packets()), _context()["source_root"])


def authenticated_main(context: Mapping[str, Any], argv: Sequence[str]) -> int:
    global _AUTH_CONTEXT
    _AUTH_CONTEXT = context
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    verbosity = 2 if "--verbose" in argv or "-v" in argv else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    print(
        json.dumps(
            {
                "schema": "tactic_actual_coarse_n18_source_tests_v3",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "source_root": context["source_root"],
                "scope": "source-only; skipped tests require a future POSIX independent review host",
            },
            sort_keys=True,
        )
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit("test_source_only.py must execute through immutable_bootstrap.py")
