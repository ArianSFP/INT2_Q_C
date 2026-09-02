#!/usr/bin/env python3
"""Hostile source-only tests for the external dispatcher.

No test discovers, stats, hashes, or opens a Qwen/current/control payload.
Temporary files contain only small deterministic synthetic bytes.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path


PACKAGE = Path(__file__).absolute().parent


def load_bootstrap() -> types.ModuleType:
    data = (PACKAGE / "bootstrap.py").read_bytes()
    module = types.ModuleType("uwfa_v8_external_dispatcher_test_snapshot")
    module.__file__ = "<authenticated-test-snapshot:bootstrap.py>"
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(data, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


b = load_bootstrap()


def load_ordinal_bridge() -> types.ModuleType:
    data = (PACKAGE / "strata_ordinal_bridge.py").read_bytes()
    module = types.ModuleType("uwfa_v8_ordinal_bridge_test_snapshot")
    module.__file__ = "<authenticated-test-snapshot:strata_ordinal_bridge.py>"
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(data, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


ordinal_bridge = load_ordinal_bridge()


def load_output_verifier() -> types.ModuleType:
    data = (PACKAGE / "verify_output.py").read_bytes()
    module = types.ModuleType("uwfa_v8_output_verifier_test_snapshot")
    module.__file__ = "<authenticated-test-snapshot:verify_output.py>"
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(data, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


output_verifier = load_output_verifier()


def write(path: Path, data: bytes) -> tuple[int, str]:
    path.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def fraction(value: Fraction | int) -> dict[str, object]:
    value = Fraction(value)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "float": float(value),
    }


class _SyntheticFramingCodec:
    @staticmethod
    def parse_container(_common: object, _semantic: object, raw: bytes) -> dict[str, object]:
        return {
            "raw": raw,
            "experts": 1,
            "byte_ledger": [{"bytes": len(raw), "padding": False, "owner_set": b"\x01"}],
        }

    @staticmethod
    def owner_ordinals(owner_set: bytes, experts: int) -> tuple[int, ...]:
        if owner_set != b"\x01" or experts != 1:
            raise ValueError("synthetic owner set")
        return (0,)


def metrics_for_ranges(ranges: list[list[int]]) -> tuple[bytes, object, dict[str, object]]:
    denominator = 8192
    literal_container = b"F" * denominator
    repeated = sum(end - begin for begin, end in ranges)
    ordered = sorted((begin, end) for begin, end in ranges)
    unique = 0
    if ordered:
        left, right = ordered[0]
        for begin, end in ordered[1:]:
            if begin > right:
                unique += right - left
                left, right = begin, end
            else:
                right = max(right, end)
        unique += right - left
    pages: set[int] = set()
    for begin, end in ranges:
        if end > begin:
            pages.update(range(begin // 4096, (end - 1) // 4096 + 1))
    touched = len(pages) * 4096
    page_ratio = Fraction(touched, denominator)
    row = {
        "expert_ordinal": 0,
        "instrumented_routed_read_ranges": ranges,
        "instrumented_routed_read_request_count": len(ranges),
        "instrumented_routed_requested_bytes_with_repetition": repeated,
        "instrumented_routed_unique_requested_bytes": unique,
        "instrumented_routed_overlap_bytes_requested_again": repeated - unique,
        "touched_page_indices": sorted(pages),
        "touched_page_bytes": touched,
        "attributable_total_physical_bytes": fraction(denominator),
        "attributable_nonpadding_decodable_bytes": fraction(denominator),
        "strict_cold_amplification": fraction(page_ratio),
    }
    metrics = {
        "actual_container_bytes": len(literal_container),
        "routed_io_authoritative_descriptor_backed": True,
        "installation_authentication_reported_separately": {"excluded_from_per_expert_cold_numerator": True},
        "experts": [row],
        "routed_read_request_aggregates": {
            "read_request_count_sum_across_experts": len(ranges),
            "requested_bytes_with_repetition_sum_across_experts": repeated,
            "unique_requested_bytes_sum_across_experts": unique,
            "overlap_bytes_requested_again_sum_across_experts": repeated - unique,
            "unique_touched_page_bytes_sum_across_experts": touched,
            "frozen_cold_gate_uses_unique_touched_page_bytes_only": True,
        },
        "maximum_strict_cold_read_amplification": fraction(page_ratio),
        "passes_cold_read_below_2x": page_ratio < 2,
    }
    framing = b.derive_authenticated_container_framing(
        literal_container,
        common=object(),
        semantic_codec=object(),
        container_codec=_SyntheticFramingCodec,
        universal_decoder_sha256="75" * 32,
    )
    return literal_container, framing, metrics


def baseline_fixture() -> tuple[bytes, dict[str, object], bytes, bytes]:
    artifact = b"synthetic-current-artifact"
    matrices = []
    legacy_matrices = []
    for ordinal in range(18):
        role = ("gate", "up", "down")[ordinal % 3]
        shape = [2048, 768] if role == "down" else [768, 2048]
        source_hash = hashlib.sha256(f"source-{ordinal}".encode()).hexdigest()
        matrices.append({
            "matrix_ordinal": ordinal,
            "tensor": f"synthetic.tensor.{ordinal}",
            "role": role,
            "shape": shape,
            "source_bf16_sha256": source_hash,
            "source_energy_fp64": 1.0,
        })
        legacy_matrices.append({
            "matrix_ordinal": ordinal,
            "tensor": f"synthetic.tensor.{ordinal}",
            "role": role,
            "shape": shape,
            "source_bf16_sha256": source_hash,
            "source_energy_fp64": 1.0,
        })
    legacy = {
        "schema": "strata_expert_affine_independent_audit_v1",
        "status": "passed",
        "bindings": {"sources_canonical_sha256": "71" * 32},
        "container": {"sha256": hashlib.sha256(artifact).hexdigest(), "physical_bytes": len(artifact)},
        "decode": {"canonical_reencode_all_match": True, "every_group_once": True},
        "source_score": {
            "sse_sum_fp64": 1.0,
            "source_energy_sum_fp64": 18.0,
            "energy_weighted_relative_mse": 1.0 / 18.0,
            "matrices": legacy_matrices,
        },
    }
    legacy_bytes = b.canonical_json(legacy)
    source = {
        "schema": "uwfa-qwen-original-source-panel-binding-v8",
        "status": "PASS_INDEPENDENT_ORIGINAL_SOURCE_BINDING",
        "weights": 18 * 768 * 2048,
        "matrices": matrices,
        "sources_canonical_sha256": "71" * 32,
        "source_energy_fp64": 18.0,
        "legacy_independent_audit_sha256": hashlib.sha256(legacy_bytes).hexdigest(),
    }
    source["source_binding_sha256"] = hashlib.sha256(b.canonical_json(source)).hexdigest()
    panel = {
        "weights": 18 * 768 * 2048,
        "reconstruction": {"full_reconstruction_f64_sha256": "72" * 32},
    }
    return artifact, panel, legacy_bytes, b.canonical_json(source)


def baseline_plan_bytes(
    *,
    artifact_bytes: bytes,
    legacy_sha256: str,
    source_binding_sha256: str,
    public_commit: str = "11" * 20,
) -> bytes:
    record = {
        "schema": "uwfa-sc-v8-authenticated-baseline-plan-v1",
        "status": "FROZEN_INDEPENDENTLY_REVIEWED_BASELINE_PLAN",
        "producer_public_git_commit": public_commit,
        "artifact": {"bytes": len(artifact_bytes), "sha256": hashlib.sha256(artifact_bytes).hexdigest()},
        "weights": 18 * 768 * 2048,
        "matrix_ordinals": list(range(18)),
        "matrix_roles": [("gate", "up", "down")[index % 3] for index in range(18)],
        "score_normalization": "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
        "legacy_independent_audit_sha256": legacy_sha256,
        "original_source_binding_sha256": source_binding_sha256,
    }
    record["plan_sha256"] = hashlib.sha256(b.canonical_json(record)).hexdigest()
    return b.canonical_json(record)


def external_authority(
    *,
    request_sha256: str,
    baseline_plan_sha256: str,
    legacy_independent_audit_sha256: str,
    original_source_binding_sha256: str,
) -> object:
    return b.ExternalLaunchAuthority(
        dispatcher_manifest_sha256="31" * 32,
        dispatcher_audit_sha256="32" * 32,
        dispatcher_public_git_commit="33" * 20,
        launcher_source_sha256="34" * 32,
        launcher_review_sha256="35" * 32,
        request_sha256=request_sha256,
        baseline_plan_sha256=baseline_plan_sha256,
        legacy_independent_audit_sha256=legacy_independent_audit_sha256,
        original_source_binding_sha256=original_source_binding_sha256,
        native_audit_event_fd=7,
        native_audit_session_nonce="3a" * 32,
    )


@unittest.skipUnless(os.name == "posix" and os.open in os.supports_dir_fd, "POSIX descriptor-relative test")
class DescriptorHostileTests(unittest.TestCase):
    def test_authenticated_loader_rejects_path_substitution_between_resolution_and_exec(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "victim.py"
            source.write_bytes(b"VALUE = 7\n")
            directory = b.HeldDirectory.open_absolute(str(root), label="loader root")
            try:
                held = directory.open_member("victim.py", cap=100, label="victim")
                absolute = b.HeldAbsoluteFile(directory, held)
                ledger = b.AppendOnlyImportNativeLedger({str(source): absolute})
                finder = b.AuthenticatedManifestFinder(import_roots=[str(root)], held_by_path={str(source): absolute})
                finder.attach_ledger(ledger)
                ledger._active = True
                spec = finder.find_spec("victim")
                source.rename(root / "victim.authenticated-old")
                source.write_bytes(b"VALUE = 99\n")
                module = types.ModuleType("victim")
                module.__spec__ = spec
                # The retained parent-directory identity normally detects the
                # rename/create first. A filesystem with unchanged directory
                # metadata must still fail the later leaf name/inode rebind.
                with self.assertRaisesRegex(b.DispatchError, r"held (directory changed|name substituted)"):
                    finder.exec_module(module)
                self.assertFalse(hasattr(module, "VALUE"), "substituted or held code must not execute after drift")
            finally:
                directory.close()

    def test_import_then_delete_sys_modules_remains_in_ledger_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "ephemeral.py"
            path.write_bytes(b"VALUE = 1\n")
            directory = b.HeldDirectory.open_absolute(str(root), label="ephemeral root")
            try:
                held = directory.open_member("ephemeral.py", cap=100, label="ephemeral")
                absolute = b.HeldAbsoluteFile(directory, held)
                ledger = b.AppendOnlyImportNativeLedger({str(path): absolute})
                module = types.ModuleType("ephemeral_authenticated")
                sys.modules[module.__name__] = module
                ledger.bind_authenticated_module(
                    module.__name__, module, held, member_path=str(path), execution_kind="source-test"
                )
                module.__file__ = "/hostile/mutable-presentation-only.py"
                self.assertEqual(ledger._module_provenance[module.__name__][1].member_path, str(path))
                sys.modules.pop(module.__name__)
                with self.assertRaisesRegex(b.DispatchError, "removed/replaced"):
                    ledger._verify_module_closure()
                actions = [row["action"] for row in ledger.events]
                self.assertIn("PYTHON_MODULE_BOUND", actions)
                self.assertIn("PYTHON_MODULE_REMOVED_OR_REPLACED", actions)
            finally:
                sys.modules.pop("ephemeral_authenticated", None)
                directory.close()

    def test_output_verifier_retains_fd_and_finally_rebinds_name_inode_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "member").write_bytes(b"authenticated")
            directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            held = output_verifier.read_regular_at(directory_fd, "member", 100)
            try:
                (root / "member").rename(root / "old")
                (root / "member").write_bytes(b"authenticated")
                with self.assertRaisesRegex(output_verifier.OutputVerificationError, "name/inode substituted"):
                    held.verify_final()
            finally:
                held.close()
                os.close(directory_fd)

    def test_no_follow_rejects_symlink_leaf_and_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            real = root / "real"
            real.mkdir()
            (real / "item").write_bytes(b"x")
            (root / "alias").symlink_to(real, target_is_directory=True)
            with self.assertRaises(Exception):
                b.HeldDirectory.open_absolute(str(root / "alias"), label="symlink ancestor")
            directory = b.HeldDirectory.open_absolute(str(real), label="real")
            try:
                (real / "link").symlink_to(real / "item")
                with self.assertRaises(Exception):
                    directory.open_member("link", cap=10, label="symlink leaf")
            finally:
                directory.close()

    def test_held_descriptor_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "item").write_bytes(b"before")
            directory = b.HeldDirectory.open_absolute(str(root), label="root")
            try:
                directory.open_member("item", cap=100, label="item")
                (root / "item").write_bytes(b"after-longer")
                with self.assertRaises(b.DispatchError):
                    directory.verify_stable()
            finally:
                directory.close()

    def test_held_descriptor_detects_name_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "item").write_bytes(b"before")
            directory = b.HeldDirectory.open_absolute(str(root), label="root")
            try:
                directory.open_member("item", cap=100, label="item")
                (root / "item").rename(root / "old")
                (root / "item").write_bytes(b"before")
                with self.assertRaises(b.DispatchError):
                    directory.verify_stable()
            finally:
                directory.close()

    def test_request_is_inaccessible_before_preflight_then_held(self) -> None:
        journal = b.AccessJournal()
        journal.authority_passed()
        with self.assertRaisesRegex(b.DispatchError, "before authority/preflight"):
            b.open_source_inputs_after_preflight(
                request_path="/definitely/not/opened/request.json",
                request_sha256="00" * 32,
                public_commit_pin="11" * 20,
                external_authority=external_authority(
                    request_sha256="00" * 32,
                    baseline_plan_sha256="01" * 32,
                    legacy_independent_audit_sha256="02" * 32,
                    original_source_binding_sha256="03" * 32,
                ),
                journal=journal,
            )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            specs = {}
            artifact = b"synthetic-artifact"
            legacy = b"synthetic-reviewed-legacy-audit"
            source = b"synthetic-reviewed-source-binding"
            payloads = {
                "artifact": artifact,
                "legacy_independent_audit": legacy,
                "original_source_binding": source,
                "baseline_plan": baseline_plan_bytes(
                    artifact_bytes=artifact,
                    legacy_sha256=hashlib.sha256(legacy).hexdigest(),
                    source_binding_sha256=hashlib.sha256(source).hexdigest(),
                ),
            }
            for name in sorted(b.SOURCE_INPUT_NAMES):
                data = payloads[name]
                path = root / name
                size, digest = write(path, data)
                specs[name] = {"path": str(path), "bytes": size, "sha256": digest}
            request_record = {
                "schema": "uwfa-sc-v8-external-source-phase-request-v1",
                "status": "AUTHORIZED_SOURCE_ONLY_NO_CONTROLS",
                "producer_public_commit": "11" * 20,
                "transaction_id": "22" * 16,
                "output_parent": str(root),
                "final_name": "synthetic-result",
                "inputs": specs,
            }
            request_bytes = b.canonical_json(request_record)
            request_path = root / "request.json"
            request_path.write_bytes(request_bytes)
            authority = external_authority(
                request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                baseline_plan_sha256=specs["baseline_plan"]["sha256"],
                legacy_independent_audit_sha256=specs["legacy_independent_audit"]["sha256"],
                original_source_binding_sha256=specs["original_source_binding"]["sha256"],
            )
            journal.preflight_started()
            journal.preflight_passed()
            opened = b.open_source_inputs_after_preflight(
                request_path=str(request_path),
                request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                public_commit_pin="11" * 20,
                external_authority=authority,
                journal=journal,
            )
            try:
                self.assertEqual(set(opened.inputs), set(b.SOURCE_INPUT_NAMES))
                self.assertLess(
                    journal.events.index("FRESH_TYPED_SOURCE_FREE_GPU_PREFLIGHT_PASSED"),
                    journal.events.index("POST_PREFLIGHT_OPEN:source-request"),
                )
            finally:
                opened.close()

    def test_request_rejects_pretty_noncanonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            specs = {}
            artifact = b"artifact"
            legacy = b"legacy"
            source = b"source"
            payloads = {
                "artifact": artifact,
                "legacy_independent_audit": legacy,
                "original_source_binding": source,
                "baseline_plan": baseline_plan_bytes(
                    artifact_bytes=artifact,
                    legacy_sha256=hashlib.sha256(legacy).hexdigest(),
                    source_binding_sha256=hashlib.sha256(source).hexdigest(),
                ),
            }
            for name in sorted(b.SOURCE_INPUT_NAMES):
                data = payloads[name]
                path = root / name
                size, digest = write(path, data)
                specs[name] = {"path": str(path), "bytes": size, "sha256": digest}
            record = {
                "schema": "uwfa-sc-v8-external-source-phase-request-v1",
                "status": "AUTHORIZED_SOURCE_ONLY_NO_CONTROLS",
                "producer_public_commit": "11" * 20,
                "transaction_id": "22" * 16,
                "output_parent": str(root),
                "final_name": "result",
                "inputs": specs,
            }
            data = json.dumps(record, indent=2).encode()
            request = root / "request.json"
            request.write_bytes(data)
            authority = external_authority(
                request_sha256=hashlib.sha256(data).hexdigest(),
                baseline_plan_sha256=specs["baseline_plan"]["sha256"],
                legacy_independent_audit_sha256=specs["legacy_independent_audit"]["sha256"],
                original_source_binding_sha256=specs["original_source_binding"]["sha256"],
            )
            journal = b.AccessJournal()
            journal.authority_passed()
            journal.preflight_started()
            journal.preflight_passed()
            with self.assertRaisesRegex(b.DispatchError, "canonical JSON"):
                b.open_source_inputs_after_preflight(
                    request_path=str(request),
                    request_sha256=hashlib.sha256(data).hexdigest(),
                    public_commit_pin="11" * 20,
                    external_authority=authority,
                    journal=journal,
                )


class ContractTests(unittest.TestCase):
    def test_preloaded_numeric_module_is_rejected_before_loader_install(self) -> None:
        sys.modules["safetensors.hostile_preload"] = types.ModuleType("safetensors.hostile_preload")
        try:
            with self.assertRaisesRegex(b.DispatchError, "preloaded numeric"):
                b.reject_preloaded_numeric_modules()
        finally:
            sys.modules.pop("safetensors.hostile_preload", None)

    def test_hostile_meta_path_and_path_hook_are_rejected(self) -> None:
        frozen = sys.modules["_frozen_importlib"]
        external = sys.modules["_frozen_importlib_external"]

        class HostileFinder:
            pass

        canonical_meta = [frozen.BuiltinImporter, frozen.FrozenImporter, external.PathFinder]
        b.reject_ambient_import_hooks(meta_path=canonical_meta, path_hooks=[])
        with self.assertRaisesRegex(b.DispatchError, "sys.meta_path"):
            b.reject_ambient_import_hooks(meta_path=[HostileFinder(), *canonical_meta], path_hooks=[])
        with self.assertRaisesRegex(b.DispatchError, "sys.path_hooks"):
            b.reject_ambient_import_hooks(meta_path=canonical_meta, path_hooks=[lambda _path: None])

    def test_transient_native_load_unload_events_are_append_only_and_unmanifested_load_fails(self) -> None:
        class FakeMember:
            identity = (5, 7, 0, 1, 2, 3, 1)
            data = b"native"
            sha256 = hashlib.sha256(data).hexdigest()

        class FakeHeld:
            member = FakeMember()

            @staticmethod
            def verify_stable() -> None:
                return None

        ledger = b.AppendOnlyImportNativeLedger({"/held/native.so": FakeHeld()})
        ledger._native_feed_nonce = "ab" * 32
        ledger._native_auditor_path = "/held/native.so"
        ledger.ingest_native_event({
            "schema": "uwfa-native-loader-event-v1", "sequence": 0,
            "action": "READY", "nonce": "ab" * 32,
            "auditor_path": "/held/native.so", "auditor_sha256": FakeMember.sha256,
            "device": 5, "inode": 7,
        })
        ledger.ingest_native_event({
            "schema": "uwfa-native-loader-event-v1", "sequence": 1,
            "action": "LOAD", "path": "/held/native.so", "device": 5, "inode": 7,
        })
        ledger.ingest_native_event({
            "schema": "uwfa-native-loader-event-v1", "sequence": 2,
            "action": "UNLOAD", "path": "/held/native.so", "device": 5, "inode": 7,
        })
        with self.assertRaisesRegex(b.DispatchError, "unmanifested transient native load"):
            ledger.ingest_native_event({
                "schema": "uwfa-native-loader-event-v1", "sequence": 3,
                "action": "LOAD", "path": "/hostile/transient.so", "device": 9, "inode": 11,
            })
        native_rows = [row for row in ledger.events if row["action"] == "NATIVE_LOADER_EVENT"]
        self.assertEqual([row["native_event"]["action"] for row in native_rows], ["READY", "LOAD", "UNLOAD", "LOAD"])
        snapshot = ledger.events
        self.assertEqual([row["sequence"] for row in snapshot], list(range(len(snapshot))))
        for prior, current in zip(snapshot, snapshot[1:]):
            self.assertEqual(current["prior_chain_sha256"], prior["chain_sha256"])
        snapshot[0]["action"] = "FORGED_COPY"
        self.assertNotEqual(ledger.events[0]["action"], "FORGED_COPY")

    def test_import_native_enforcement_order_precedes_snapshots_numeric_import_and_preflight(self) -> None:
        source = (PACKAGE / "bootstrap.py").read_text(encoding="utf-8")
        body = source[source.index("def dispatch_production("):source.index("def _parser()")]
        self.assertLess(body.index("activate_import_native_enforcement("), body.index("compile_producer_snapshots("))
        self.assertLess(body.index("activate_import_native_enforcement("), body.index("enable_python_runtime()"))
        self.assertLess(body.index("activate_import_native_enforcement("), body.index("run_fresh_typed_source_free_preflight("))
        preflight = source[source.index("def run_fresh_typed_source_free_preflight("):source.index("class InputSpec")]
        self.assertLess(preflight.index('checkpoint("BEFORE_AUTHORITATIVE_PREFLIGHT")'), preflight.index("journal.preflight_started()"))
        self.assertLess(preflight.index('checkpoint("AFTER_AUTHORITATIVE_PREFLIGHT")'), preflight.index("journal.preflight_passed()"))

    def test_numpy_scalar_ordinal_bridge_preserves_value_and_source_order(self) -> None:
        class FakeDType:
            kind = "i"

            def type(self, value: int) -> object:
                return FakeInteger(value)

        class FakeInteger:
            def __init__(self, value: int) -> None:
                self.value = value
                self.dtype = FakeDType()

            def item(self) -> int:
                return self.value

            def __int__(self) -> int:
                return self.value

        class FakeBool(FakeInteger):
            pass

        class FakeNP:
            integer = FakeInteger
            bool_ = FakeBool

        source_order = [FakeInteger(2), FakeInteger(0), FakeInteger(1)]
        bridged = ordinal_bridge.bridge_canonical_ordinal_sequence(FakeNP, source_order, count=3)
        self.assertEqual(bridged, [2, 0, 1])
        self.assertIs(type(bridged), list)
        self.assertTrue(all(type(value) is int for value in bridged))
        with self.assertRaisesRegex(ordinal_bridge.OrdinalBridgeError, "duplicates"):
            ordinal_bridge.bridge_canonical_ordinal_sequence(
                FakeNP, [FakeInteger(0), FakeInteger(0), FakeInteger(2)], count=3
            )
        with self.assertRaisesRegex(ordinal_bridge.OrdinalBridgeError, "NumPy integer scalar"):
            ordinal_bridge.numpy_scalar_to_builtin_int(FakeNP, 1)
        with self.assertRaisesRegex(ordinal_bridge.OrdinalBridgeError, "tuple ordinal rows"):
            ordinal_bridge.bridge_canonical_ordinal_sequence(
                FakeNP, (FakeInteger(0), FakeInteger(1), FakeInteger(2)), count=3
            )

    def test_numpy_2d_advanced_index_bridge_requires_list_rows(self) -> None:
        np = __import__("numpy")

        class ListRows:
            GROUPS = 3

            @staticmethod
            def expected_block_group_ordinals(_labels: object) -> list[list[object]]:
                return [[np.int64(2), np.int64(0), np.int64(1)]]

        original_row = ListRows.expected_block_group_ordinals(None)[0]
        converted_row = ordinal_bridge.wrap_strata_common(ListRows, np).expected_block_group_ordinals(None)[0]
        post = np.arange(12).reshape(3, 4)
        self.assertIs(type(converted_row), list)
        self.assertTrue(all(type(value) is int for value in converted_row))
        self.assertTrue(np.array_equal(post[converted_row], post[np.asarray(original_row)]))
        with self.assertRaises(IndexError):
            _ = post[tuple(converted_row)]

        class TupleRows:
            GROUPS = 3

            @staticmethod
            def expected_block_group_ordinals(_labels: object) -> list[tuple[object, ...]]:
                return [(np.int64(2), np.int64(0), np.int64(1))]

        with self.assertRaisesRegex(ordinal_bridge.OrdinalBridgeError, "rows must be built-in lists"):
            ordinal_bridge.wrap_strata_common(TupleRows, np).expected_block_group_ordinals(None)

    def test_decoder_logical_to_producer_member_map_is_exact_and_injective(self) -> None:
        expected = {
            "fixed_strata_sc_adapter": "strata_sc_adapter.py",
            "universal_semantic_adapter": "universal_adapter.py",
        }
        self.assertEqual(b.validate_logical_to_producer_member_map(expected), expected)
        hostile = dict(expected)
        hostile["fixed_strata_sc_adapter"] = "universal_adapter.py"
        with self.assertRaisesRegex(b.DispatchError, "exact logical"):
            b.validate_logical_to_producer_member_map(hostile)

    def test_ordinal_bridge_source_and_abi_are_exactly_decoder_bundle_bound(self) -> None:
        bundle = json.loads((PACKAGE / "decoder_bundle.json").read_bytes())
        row = next(item for item in bundle["members"] if item["logical_name"] == "numpy_strata_ordinal_bridge")
        bridge_bytes = (PACKAGE / "strata_ordinal_bridge.py").read_bytes()
        self.assertEqual(row["sha256"], hashlib.sha256(bridge_bytes).hexdigest())
        self.assertEqual(row["bridge_abi_sha256"], ordinal_bridge.BRIDGE_ABI_SHA256)

    def test_checked_in_production_pins_remain_fail_closed(self) -> None:
        with self.assertRaisesRegex(b.DispatchError, "BLOCK_UNRESOLVED"):
            b.ProductionPins.embedded()

    def test_direct_execution_blocks_before_argument_or_payload_handling(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(PACKAGE / "bootstrap.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("BLOCK_DIRECT_PRODUCTION_LAUNCH_REQUIRES_OUT_OF_TREE_PINNED_AUTHORITY", completed.stderr)
        self.assertNotIn("--request", completed.stderr)

    def test_external_launcher_authority_is_typed_and_non_self_pinned(self) -> None:
        authority = b.ExternalLaunchAuthority(
            dispatcher_manifest_sha256="31" * 32,
            dispatcher_audit_sha256="32" * 32,
            dispatcher_public_git_commit="33" * 20,
            launcher_source_sha256="34" * 32,
            launcher_review_sha256="35" * 32,
            request_sha256="36" * 32,
            baseline_plan_sha256="37" * 32,
            legacy_independent_audit_sha256="38" * 32,
            original_source_binding_sha256="39" * 32,
            native_audit_event_fd=7,
            native_audit_session_nonce="3a" * 32,
        )
        self.assertEqual(authority.dispatcher_public_git_commit, "33" * 20)
        with self.assertRaises(b.DispatchError):
            b.ExternalLaunchAuthority(
                "bad", "32" * 32, "33" * 20, "34" * 32, "35" * 32,
                "36" * 32, "37" * 32, "38" * 32, "39" * 32, 7, "3a" * 32,
            )

    def test_dispatch_has_no_caller_production_pins_parameter(self) -> None:
        self.assertEqual(list(inspect.signature(b.dispatch_production).parameters), ["arguments", "external_authority"])
        source = (PACKAGE / "bootstrap.py").read_text(encoding="utf-8")
        body = source[source.index("def dispatch_production("):source.index("def _parser()")]
        self.assertIn("pins = ProductionPins.embedded()", body)

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(b.DispatchError, "duplicate key"):
            b.strict_json(b'{"a":1,"a":2}')
        with self.assertRaisesRegex(b.DispatchError, "nonfinite"):
            b.strict_json(b'{"a":NaN}')

    def test_noncanonical_and_unsafe_paths_reject(self) -> None:
        for path in ("relative", "/tmp/../x", "/tmp//x", "/tmp/x/", "/tmp\\x"):
            with self.assertRaises(b.DispatchError, msg=path):
                b._absolute_components(path, label="hostile")

    def test_preloaded_snapshot_rejected(self) -> None:
        sys.modules["uwfa_v8_dispatch_common"] = types.ModuleType("uwfa_v8_dispatch_common")
        try:
            with self.assertRaisesRegex(b.DispatchError, "preloaded"):
                b.reject_preloaded_snapshot_modules()
        finally:
            sys.modules.pop("uwfa_v8_dispatch_common", None)

    def test_snapshot_compiles_exact_authenticated_bytes_only(self) -> None:
        data = b"VALUE = 7\n"
        module = b.exec_snapshot_module("synthetic_snapshot", data, hashlib.sha256(data).hexdigest())
        try:
            self.assertEqual(module.VALUE, 7)
            self.assertTrue(module.__file__.startswith("<authenticated-snapshot:"))
        finally:
            sys.modules.pop("synthetic_snapshot", None)
        with self.assertRaisesRegex(b.DispatchError, "digest mismatch"):
            b.exec_snapshot_module("synthetic_bad", data, "00" * 32)

    def test_blocked_runtime_lock_rejects_before_external_path(self) -> None:
        data = (PACKAGE / "runtime_lock.json").read_bytes()
        with self.assertRaisesRegex(b.DispatchError, "runtime lock status"):
            b.authenticate_runtime(data, expected_lock_sha256=hashlib.sha256(data).hexdigest())

    def test_blocked_decoder_bundle_rejects_before_member_paths(self) -> None:
        data = (PACKAGE / "decoder_bundle.json").read_bytes()
        with self.assertRaisesRegex(b.DispatchError, "decoder bundle status"):
            b.authenticate_decoder_bundle(data, expected_bundle_sha256=hashlib.sha256(data).hexdigest(), producer=None)

    def test_access_journal_cannot_skip_authority_or_preflight(self) -> None:
        journal = b.AccessJournal()
        with self.assertRaises(b.DispatchError):
            journal.preflight_started()
        with self.assertRaises(b.DispatchError):
            journal.before_payload_path_access("artifact")
        journal.authority_passed()
        journal.preflight_started()
        with self.assertRaises(b.DispatchError):
            journal.before_payload_path_access("artifact")
        journal.preflight_passed()
        journal.before_payload_path_access("artifact")
        self.assertEqual(journal.events[-1], "POST_PREFLIGHT_OPEN:artifact")

    def test_manifest_rows_require_exact_order_and_names(self) -> None:
        rows = [
            {"name": "a", "bytes": 1, "sha256": "11" * 32},
            {"name": "b", "bytes": 2, "sha256": "22" * 32},
        ]
        b._manifest_rows(rows, expected_names={"a", "b"}, label="fixture")
        with self.assertRaisesRegex(b.DispatchError, "canonical order"):
            b._manifest_rows(list(reversed(rows)), expected_names={"a", "b"}, label="fixture")
        with self.assertRaisesRegex(b.DispatchError, "exact member set"):
            b._manifest_rows(rows, expected_names={"a", "b", "c"}, label="fixture")

    def test_runtime_loaded_image_closure_rejects_unheld_python_or_native_image(self) -> None:
        b.validate_loaded_image_path_set(
            observed={"/held/python", "/held/native.so"},
            held_manifest_paths={"/held/python", "/held/native.so", "/held/unused"},
        )
        with self.assertRaisesRegex(b.DispatchError, "unmanifested"):
            b.validate_loaded_image_path_set(
                observed={"/held/python", "/hostile/injected.so"},
                held_manifest_paths={"/held/python"},
            )

    def test_authority_exact_request_pin_rejects_before_path_access(self) -> None:
        journal = b.AccessJournal()
        journal.authority_passed()
        journal.preflight_started()
        journal.preflight_passed()
        authority = external_authority(
            request_sha256="aa" * 32,
            baseline_plan_sha256="bb" * 32,
            legacy_independent_audit_sha256="cc" * 32,
            original_source_binding_sha256="dd" * 32,
        )
        with self.assertRaisesRegex(b.DispatchError, "external authority mismatch"):
            b.open_source_inputs_after_preflight(
                request_path="/this/path/must/not/be/opened",
                request_sha256="ee" * 32,
                public_commit_pin="11" * 20,
                external_authority=authority,
                journal=journal,
            )
        self.assertNotIn("POST_PREFLIGHT_OPEN:source-request", journal.events)

    def test_baseline_plan_is_parsed_sealed_and_cross_bound(self) -> None:
        artifact = b"artifact"
        legacy_sha = "ab" * 32
        source_sha = "cd" * 32
        encoded = baseline_plan_bytes(
            artifact_bytes=artifact,
            legacy_sha256=legacy_sha,
            source_binding_sha256=source_sha,
        )
        spec = b.InputSpec("/synthetic/artifact", len(artifact), hashlib.sha256(artifact).hexdigest())
        record = b.validate_baseline_plan(
            encoded,
            public_git_commit="11" * 20,
            artifact=spec,
            legacy_independent_audit_sha256=legacy_sha,
            original_source_binding_sha256=source_sha,
        )
        self.assertEqual(record["matrix_ordinals"], list(range(18)))
        hostile = json.loads(encoded)
        hostile["matrix_ordinals"][0], hostile["matrix_ordinals"][1] = 1, 0
        hostile.pop("plan_sha256")
        hostile["plan_sha256"] = hashlib.sha256(b.canonical_json(hostile)).hexdigest()
        with self.assertRaisesRegex(b.DispatchError, "exact matrix order"):
            b.validate_baseline_plan(
                b.canonical_json(hostile),
                public_git_commit="11" * 20,
                artifact=spec,
                legacy_independent_audit_sha256=legacy_sha,
                original_source_binding_sha256=source_sha,
            )

    def test_score_is_derived_from_adapter_geometry_not_request(self) -> None:
        artifact, panel, legacy, source = baseline_fixture()
        record, encoded = b.construct_bound_baseline_score(
            artifact_bytes=artifact,
            panel=panel,
            source_full_geometry_sha256="73" * 32,
            universal_decoder_sha256="74" * 32,
            legacy_audit_bytes=legacy,
            original_source_binding_bytes=source,
        )
        self.assertEqual(record["artifact_sha256"], hashlib.sha256(artifact).hexdigest())
        self.assertEqual(record["reconstruction_f64_sha256"], "72" * 32)
        self.assertEqual(record["original_source_panel_sha256"], "73" * 32)
        self.assertEqual(record["independent_decoder_source_sha256"], "74" * 32)
        self.assertEqual(encoded, b.canonical_json(record))
        clean = dict(record)
        claimed = clean.pop("score_receipt_sha256")
        self.assertEqual(claimed, hashlib.sha256(b.canonical_json(clean)).hexdigest())

    def test_score_rejects_legacy_arithmetic_or_source_substitution(self) -> None:
        artifact, panel, legacy_bytes, source_bytes = baseline_fixture()
        legacy = json.loads(legacy_bytes)
        legacy["source_score"]["energy_weighted_relative_mse"] *= 1.01
        with self.assertRaisesRegex(b.DispatchError, "relative MSE arithmetic"):
            b.construct_bound_baseline_score(
                artifact_bytes=artifact,
                panel=panel,
                source_full_geometry_sha256="73" * 32,
                universal_decoder_sha256="74" * 32,
                legacy_audit_bytes=b.canonical_json(legacy),
                original_source_binding_bytes=source_bytes,
            )
        source = json.loads(source_bytes)
        source["matrices"][0]["source_bf16_sha256"] = "99" * 32
        with self.assertRaisesRegex(b.DispatchError, "internal seal"):
            b.construct_bound_baseline_score(
                artifact_bytes=artifact,
                panel=panel,
                source_full_geometry_sha256="73" * 32,
                universal_decoder_sha256="74" * 32,
                legacy_audit_bytes=legacy_bytes,
                original_source_binding_bytes=b.canonical_json(source),
            )

    def test_unique_page_and_repeated_coalesced_gate_pass(self) -> None:
        literal, framing, metrics = metrics_for_ranges([[0, 4096], [4096, 8192]])
        result = b.validate_external_bandwidth_gate(literal, framing, metrics)
        self.assertEqual(result["status"], "PASS_STRICT_ALL_BANDWIDTH_GATES")
        self.assertTrue(result["passes_all_bandwidth_gates"])

    def test_repeated_request_gate_catches_unique_page_pass(self) -> None:
        literal, framing, metrics = metrics_for_ranges([[0, 8192], [0, 8192]])
        self.assertTrue(metrics["passes_cold_read_below_2x"])
        result = b.validate_external_bandwidth_gate(literal, framing, metrics)
        self.assertFalse(result["repeated_request_gate_passed"])
        self.assertTrue(result["coalesced_request_gate_passed"])
        self.assertTrue(result["independently_recomputed_unique_page_gate_passed"])
        self.assertEqual(result["maximum_strict_repeated_request_amplification"]["exact"], "2/1")

    def test_bandwidth_gate_rejects_resealed_summary_and_fraction_tamper(self) -> None:
        literal, framing, metrics = metrics_for_ranges([[0, 4096]])
        hostile = copy.deepcopy(metrics)
        hostile["routed_read_request_aggregates"]["requested_bytes_with_repetition_sum_across_experts"] += 1
        with self.assertRaisesRegex(b.DispatchError, "aggregate repeated"):
            b.validate_external_bandwidth_gate(literal, framing, hostile)
        hostile = copy.deepcopy(metrics)
        hostile["experts"][0]["attributable_total_physical_bytes"]["exact"] = "forged"
        with self.assertRaisesRegex(b.DispatchError, "exact spelling"):
            b.validate_external_bandwidth_gate(literal, framing, hostile)
        hostile = copy.deepcopy(metrics)
        hostile["routed_io_authoritative_descriptor_backed"] = False
        with self.assertRaisesRegex(b.DispatchError, "authoritative descriptor"):
            b.validate_external_bandwidth_gate(literal, framing, hostile)

    def test_bandwidth_gate_rejects_literal_length_and_authenticated_denominator_substitution(self) -> None:
        literal, framing, metrics = metrics_for_ranges([[0, 4096]])
        with self.assertRaisesRegex(b.DispatchError, "framing/container digest"):
            b.validate_external_bandwidth_gate(literal + b"x", framing, metrics)
        hostile = copy.deepcopy(metrics)
        hostile["experts"][0]["attributable_total_physical_bytes"] = fraction(4096)
        with self.assertRaisesRegex(b.DispatchError, "denominator/framing mismatch"):
            b.validate_external_bandwidth_gate(literal, framing, hostile)

    def test_authority_request_output_inode_aliases_fail_closed(self) -> None:
        b.reject_authority_request_output_inode_aliasing(
            authority={"audit": (1, 10)},
            request={"request": (1, 11)},
            output={"parent": (1, 12)},
        )
        with self.assertRaisesRegex(b.DispatchError, "inode aliasing rejected"):
            b.reject_authority_request_output_inode_aliasing(
                authority={"audit": (1, 10)},
                request={"request": (1, 10)},
                output={"parent": (1, 12)},
            )

    def test_publication_member_schema_is_exact_and_conditional(self) -> None:
        base = b.expected_publication_member_names(False)
        full = b.expected_publication_member_names(True)
        self.assertIn("RUN_STATE.json", base)
        self.assertIn("COMPLETE.json", base)
        self.assertNotIn("UWFCV8.bin", base)
        self.assertEqual(full - base, {"UWFCV8.bin", "IDENTITY_FRAMING.bin", "POSTERIOR_HANDOFF.json"})

    def test_source_phase_has_no_control_input_name(self) -> None:
        self.assertEqual(
            set(b.SOURCE_INPUT_NAMES),
            {"artifact", "baseline_plan", "legacy_independent_audit", "original_source_binding"},
        )
        self.assertFalse(any("control" in name for name in b.SOURCE_INPUT_NAMES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
