#!/usr/bin/env python3
"""Hostile standard-library tests for the immutable N18 v6 sibling."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def load(name: str, filename: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


source_auth = load("tacn18_v6_test_source_auth", "source_auth.py")
closure = load("tacn18_v6_test_closure", "runtime_closure.py")
codec = load("tacn18_v6_test_codec", "successor_codec.py")
smoke_contract = load("tacn18_v6_test_smoke_contract", "smoke_contract.py")
smoke_script = load("tacn18_v6_test_smoke_script", "synthetic_cupy_smoke.py")
dispatcher = load("tacn18_v6_test_dispatcher", "dispatcher.py")
source_verifier = load("tacn18_v6_test_source_verifier", "verify_source.py")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_manifest_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["name"].encode("utf-8"))
    return {
        "schema": source_auth.MANIFEST_SCHEMA,
        "status": source_auth.MANIFEST_STATUS,
        "source_root_sha256": digest(source_auth.canonical_json(ordered)),
        "members": ordered,
        "claim_boundary":
            "source-free mechanics only; Qwen pilot, universal-tail, MSE, fine-code and inference-HBM claims require separate authorization and audit",
        "access_attestation": {
            "runpod_accessed": False,
            "qwen_or_model_payload_accessed": False,
            "cuda_or_cupy_initialized_during_source_build": False,
            "network_accessed": False,
        },
    }


def write_temp_source_package(root: Path, *, hardlink: bool = False) -> Path:
    entry = root / "entry.py"
    other = root / "other.py"
    entry.write_bytes(b"entry\n")
    if hardlink:
        os.link(entry, other)
    else:
        other.write_bytes(b"other\n")
    rows = []
    for member in (entry, other):
        payload = member.read_bytes()
        rows.append({"name": member.name, "bytes": len(payload),
                     "sha256": digest(payload)})
    manifest = source_manifest_record(rows)
    (root / "SOURCE_MANIFEST.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("ascii") + b"\n")
    return entry


class SourceClosureTests(unittest.TestCase):
    def test_current_manifest_is_exact_retained_closure(self) -> None:
        payload = (ROOT / "SOURCE_MANIFEST.json").read_bytes()
        with source_auth.HeldSourcePackage(
            ROOT, digest(payload), executing_path=ROOT / "test_source_only.py",
        ) as package:
            self.assertEqual(package.manifest_sha256, digest(payload))
            self.assertEqual(package.receipt()["executing_entry_name"],
                             "test_source_only.py")
            package.verify_final()

    def test_independent_retained_verifier(self) -> None:
        result = source_verifier.verify()
        self.assertEqual(
            result["status"],
            "PASS_RETAINED_SOURCE_CLOSURE_AWAITING_EXTERNAL_CUPY_SMOKE")
        self.assertFalse(result["source_free_smoke_run"])
        self.assertFalse(result["payload_accessed"])

    def test_temp_package_rejects_extra_symlink_entry(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlink unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            entry = write_temp_source_package(root)
            try:
                os.symlink(entry.name, root / "extra-link")
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            manifest = (root / "SOURCE_MANIFEST.json").read_bytes()
            with self.assertRaises(source_auth.SourceAuthError):
                source_auth.HeldSourcePackage(
                    root, digest(manifest), executing_path=entry)

    def test_temp_package_rejects_member_inode_alias(self) -> None:
        if not hasattr(os, "link"):
            self.skipTest("hardlink unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            try:
                entry = write_temp_source_package(root, hardlink=True)
            except OSError:
                self.skipTest("hardlink unavailable")
            manifest = (root / "SOURCE_MANIFEST.json").read_bytes()
            with self.assertRaises(source_auth.SourceAuthError):
                source_auth.HeldSourcePackage(
                    root, digest(manifest), executing_path=entry)

    def test_predecessor_and_numerical_dependency_pins(self) -> None:
        lock = json.loads(
            (ROOT / "PREDECESSOR_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["schema"],
                         "tactic-actual-coarse-n18-v6-predecessor-lock-v1")
        self.assertEqual(
            lock["source_root_sha256"],
            "1f9f2c92df3796f5f23b7e3a6b0826d6d8a2ea53bc70014fb75e61e7bc8a9fbf")
        rows = []
        v4 = REPO / lock["relative_directory"]
        for row in lock["members"]:
            payload = (v4 / row["name"]).read_bytes()
            self.assertEqual((len(payload), digest(payload)),
                             (row["bytes"], row["sha256"]))
            rows.append({"name": row["name"], "bytes": len(payload),
                         "sha256": digest(payload)})
        self.assertEqual(closure._v4_source_root(rows),
                         lock["source_root_sha256"])
        for dependency in lock["numerical_dependencies"]:
            payload = (REPO / dependency["relative_path"]).read_bytes()
            self.assertEqual((len(payload), digest(payload)),
                             (dependency["bytes"], dependency["sha256"]))

    def test_runtime_lock_is_exact_not_a_range(self) -> None:
        lock = json.loads(
            (ROOT / "RUNTIME_LOCK.json").read_text(encoding="utf-8"))
        expected = {
            "python": "3.12.3", "implementation": "CPython",
            "system": "Linux", "machine": "x86_64", "numpy": "2.5.2",
            "cupy": "14.2.0", "cuda_runtime": 12090,
            "cuda_driver": 13000, "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 5090",
            "compute_capability": "120",
        }
        for key, value in expected.items():
            self.assertEqual(lock[key], value)

    def test_isolated_entry_is_inert_and_help_runs_without_cuda(self) -> None:
        source = (ROOT / "synthetic_cupy_smoke.py").read_text(encoding="utf-8")
        self.assertNotIn("from successor_codec", source)
        self.assertNotIn("import successor_codec", source)
        self.assertNotIn("import cupy", source)
        completed = subprocess.run(
            [sys.executable, "-I", "-B", os.fspath(ROOT / "synthetic_cupy_smoke.py"),
             "--help"], capture_output=True, text=True, timeout=30,
            check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--package-manifest-sha256", completed.stdout)


class CodecContractTests(unittest.TestCase):
    @staticmethod
    def fake_encoder_runtime(flags: tuple[bool, bool, bool], *, zero: bool = False):
        packet = b"packet-v6"

        class Numeric:
            @staticmethod
            def encode_tile(*_args: Any):
                if zero:
                    report = {"status": "PASS_EXACT_ZERO_TILE",
                              "packet_sha256": digest(packet)}
                else:
                    report = {
                        "status": "PASS_FINITE_TILE_WITHIN_RESERVOIR",
                        "packet_sha256": digest(packet),
                        "arithmetic_roundtrip_bits_match": flags[0],
                        "causal_decoder_frequencies_match": flags[1],
                        "reconstruction_indices_match": flags[2],
                    }
                return packet, report

        return SimpleNamespace(numeric_encoder=Numeric(),
                               encoder_runtime=object())

    def test_encoder_requires_all_three_nonzero_checks(self) -> None:
        packet, report = codec.encode_tile_v6(
            b"x", object(), 0, 0,
            self.fake_encoder_runtime((True, True, True)))
        self.assertEqual(packet, b"packet-v6")
        self.assertTrue(report["all_encoder_self_checks_required_and_passed"])
        for failed in range(3):
            flags = [True, True, True]
            flags[failed] = False
            with self.subTest(failed=failed), self.assertRaises(codec.CodecError):
                codec.encode_tile_v6(
                    b"x", object(), 0, 0,
                    self.fake_encoder_runtime(tuple(flags)))

    def test_zero_tile_marks_checks_not_applicable(self) -> None:
        _packet, report = codec.encode_tile_v6(
            b"x", object(), 0, 0,
            self.fake_encoder_runtime((False, False, False), zero=True))
        self.assertTrue(report["all_encoder_self_checks_required_and_passed"])
        self.assertFalse(report["encoder_self_checks_applicable"])

    def test_role_abi_survives_frame_encoder_end_to_end(self) -> None:
        geometry = SimpleNamespace(
            role_values=2, streams_per_role=1, records=3, frame_bytes=30,
            values=6, target_eligible=False, intermediate=1, hidden=2)

        class Numeric:
            @staticmethod
            def encode_tile(_raw: bytes, _geometry: Any, role: int,
                            _tile: int, _runtime: Any):
                packet = bytes([role + 1]) * 10
                return packet, {
                    "status": "PASS_FINITE_TILE_WITHIN_RESERVOIR",
                    "packet_sha256": digest(packet),
                    "arithmetic_roundtrip_bits_match": True,
                    "causal_decoder_frequencies_match": True,
                    "reconstruction_indices_match": True,
                }

        packet_module = SimpleNamespace(
            ROLES=("gate", "up", "down_transposed"),
            parse_expert_frame=lambda _frame: SimpleNamespace(geometry=geometry))
        runtime = SimpleNamespace(
            numeric_encoder=Numeric(), encoder_runtime=object(),
            packet=packet_module, receipt={"test": True})
        role_bytes = {"gate": b"\x00" * 4, "up": b"\x01" * 4,
                      "down_transposed": b"\x02" * 4}
        frame, receipt = codec.encode_expert_frame_from_bf16_v6(
            role_bytes, geometry, runtime)
        self.assertEqual(frame, b"\x01" * 10 + b"\x02" * 10 + b"\x03" * 10)
        self.assertEqual(set(receipt["source_bf16_sha256"]),
                         {"gate", "up", "down_transposed"})
        with self.assertRaises(codec.CodecError):
            codec.encode_expert_frame_from_bf16_v6(
                {"gate": b"\0" * 4, "up": b"\0" * 4,
                 "down": b"\0" * 4}, geometry, runtime)

    def test_target_rate_is_exact_rational_and_tail_is_separate(self) -> None:
        target = SimpleNamespace(
            frame_bytes=1_414_656, values=4_718_592,
            target_eligible=True)
        result = codec.exact_rate_record(target.frame_bytes, target)
        self.assertEqual(result["exact"], "307/128")
        self.assertTrue(result["equals_307_over_128"])
        tail = SimpleNamespace(frame_bytes=235_776, values=3,
                               target_eligible=False)
        tail_result = codec.exact_rate_record(tail.frame_bytes, tail)
        self.assertFalse(tail_result["equals_307_over_128"])
        self.assertGreater(tail_result["float"], 307 / 128)
        mismatch = SimpleNamespace(frame_bytes=1_414_656, values=4_718_592,
                                   target_eligible=False)
        with self.assertRaises(codec.CodecError):
            codec.exact_rate_record(mismatch.frame_bytes, mismatch)

    def test_external_host_scratch_and_hbm_ledgers_are_disjoint(self) -> None:
        geometry = SimpleNamespace(
            frame_bytes=1_414_656, records=18, values=4_718_592,
            target_eligible=True)
        pre = codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=0,
            external_read_mode="prebuffered_encoder_output")
        one = codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=1,
            external_read_mode="one_pass_external_file")
        two = codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=2,
            external_read_mode="modeled_external_file_reread")
        self.assertEqual(pre["external_compressed_read"]["total_read_bytes"], 0)
        self.assertEqual(one["external_compressed_read"]["first_pass_bytes"],
                         geometry.frame_bytes)
        self.assertEqual(one["external_compressed_read"]["reread_bytes"], 0)
        self.assertEqual(two["external_compressed_read"]["total_read_bytes"],
                         2 * geometry.frame_bytes)
        self.assertEqual(two["external_compressed_read"]["reread_bytes"],
                         geometry.frame_bytes)
        self.assertEqual(
            one["host_memory_parse_and_integrity"]
            ["minimum_input_frame_full_scan_equivalents"], 3)
        self.assertEqual(one["scratch_lower_bound"]
                         ["canonical_symbols_i32_bytes"], 18 * (1 << 18) * 4)
        self.assertEqual(one["accelerator_hbm"]["measured"], False)
        self.assertIsNone(one["accelerator_hbm"]["read_bytes"])

    def test_ledger_rejects_inconsistent_modes(self) -> None:
        geometry = SimpleNamespace(
            frame_bytes=235_776, records=3, values=3 * (1 << 18),
            target_eligible=True)
        bad = ((0, "one_pass_external_file"),
               (1, "prebuffered_encoder_output"),
               (1, "modeled_external_file_reread"))
        for passes, mode in bad:
            with self.subTest(passes=passes, mode=mode), \
                    self.assertRaises(codec.CodecError):
                codec.frame_ledger_v6(
                    geometry, external_compressed_read_passes=passes,
                    external_read_mode=mode)

    def test_mixed_record_geometry_is_rejected_without_second_scan(self) -> None:
        first_geometry = SimpleNamespace(
            frame_bytes=3, records=3, streams_per_role=1, role_values=1,
            values=3, target_eligible=False, intermediate=1, hidden=1)
        other_geometry = SimpleNamespace(**vars(first_geometry))
        other_geometry.hidden = 2
        calls = 0

        def fake_decode(packet: bytes, _runtime: Any):
            nonlocal calls
            ordinal = calls
            calls += 1
            geometry = first_geometry if ordinal == 0 else other_geometry
            parsed = SimpleNamespace(
                geometry=geometry, role_ordinal=ordinal, tile_ordinal=0,
                zero_tile=False)
            return codec.DecodedTileV6(parsed, None, None, packet, {})

        original = codec.decode_tile_v6
        codec.decode_tile_v6 = fake_decode
        try:
            runtime = SimpleNamespace(packet=SimpleNamespace(
                ROLES=("gate", "up", "down_transposed"), RESERVOIR_BYTES=1))
            with self.assertRaises(codec.CodecError):
                codec.decode_expert_frame_bytes_v6(b"abc", runtime)
        finally:
            codec.decode_tile_v6 = original

    def test_i32_facade_refuses_i16_and_retains_i32_without_copy(self) -> None:
        class Array:
            def __init__(self, dtype: str):
                self.dtype = SimpleNamespace(str=dtype)
                self.ndim = 1
                self.flags = SimpleNamespace(c_contiguous=True)

            def astype(self, dtype: str, copy: bool = True):
                if dtype == "<i4" and not copy and self.dtype.str == "<i4":
                    return self
                return Array(dtype)

        class FakeNP:
            @staticmethod
            def asarray(value: Any):
                return value

            @staticmethod
            def shares_memory(left: Any, right: Any) -> bool:
                return left is right

        i32 = Array("<i4")
        self.assertIs(codec.retain_canonical_symbols_i32(i32, FakeNP), i32)
        with self.assertRaises(codec.CodecError):
            codec.retain_canonical_symbols_i32(Array("<i2"), FakeNP)


class InputDispatcherTests(unittest.TestCase):
    def make_input(
        self, root: Path, *, extra_identity: bool = False,
        bad_role: str | None = None, nonfinite_word: int | None = None,
    ) -> tuple[Path, bytes]:
        rows = []
        roles = ("gate", "up", "down_transposed")
        for index, role in enumerate(roles):
            filename = f"role-{index}.bf16"
            words = [0x3F80] * 12
            if nonfinite_word is not None and index == 0:
                words[0] = nonfinite_word
            payload = b"".join(struct.pack("<H", word) for word in words)
            (root / filename).write_bytes(payload)
            emitted_role = bad_role if index == 2 and bad_role else role
            rows.append({"role": emitted_role, "relative_path": filename,
                         "bytes": len(payload), "sha256": digest(payload)})
        record: dict[str, Any] = {
            "schema": dispatcher.INPUT_SCHEMA,
            "geometry": {"intermediate": 3, "hidden": 4},
            "roles": rows,
            "output_directory_name": "result",
        }
        if extra_identity:
            record["model_name"] = "forbidden"
        payload = json.dumps(record, sort_keys=True).encode("utf-8")
        path = root / "INPUT.json"
        path.write_bytes(payload)
        return path, payload

    def test_input_binding_is_identity_free_and_exact_role_abi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path, payload = self.make_input(root)
            result = dispatcher.authenticate_inputs(
                path, expected_manifest_sha256=digest(payload))
            self.assertFalse(result["identity_fields_available_to_codec"])
            self.assertEqual(set(result["role_bytes"]),
                             {"gate", "up", "down_transposed"})

    def test_input_rejects_identity_field_and_down_alias(self) -> None:
        for kwargs in ({"extra_identity": True}, {"bad_role": "down"}):
            with self.subTest(kwargs=kwargs), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path, payload = self.make_input(root, **kwargs)
                with self.assertRaises(dispatcher.DispatchError):
                    dispatcher.authenticate_inputs(
                        path, expected_manifest_sha256=digest(payload))

    def test_input_rejects_bf16_nan_and_infinity_before_encoder(self) -> None:
        for word in (0x7F80, 0xFF80, 0x7FC1):
            with self.subTest(word=hex(word)), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path, payload = self.make_input(root, nonfinite_word=word)
                with self.assertRaises(dispatcher.DispatchError):
                    dispatcher.authenticate_inputs(
                        path, expected_manifest_sha256=digest(payload))

    def test_input_symlink_chain_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlink unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            real.mkdir()
            path, payload = self.make_input(real)
            link = root / "link"
            try:
                os.symlink(real, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaises(dispatcher.DispatchError):
                dispatcher.authenticate_inputs(
                    link / path.name, expected_manifest_sha256=digest(payload))

    def test_output_namespace_inside_source_package_is_rejected(self) -> None:
        with self.assertRaises(dispatcher.DispatchError):
            dispatcher.require_output_outside_package(ROOT / "result", ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary).resolve() / "result"
            dispatcher.require_output_outside_package(outside, ROOT)


def valid_smoke_record() -> tuple[dict[str, Any], dict[str, str]]:
    member_hashes = {"synthetic_cupy_smoke.py": "a" * 64,
                     "source_auth.py": "b" * 64}
    geometry = SimpleNamespace(
        frame_bytes=3 * 78_592, records=3, values=3 * (1 << 18),
        target_eligible=True)
    traffic = {
        "prebuffered_decode": codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=0,
            external_read_mode="prebuffered_encoder_output"),
        "modeled_one_external_pass": codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=1,
            external_read_mode="one_pass_external_file"),
        "modeled_two_external_passes": codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=2,
            external_read_mode="modeled_external_file_reread"),
    }
    record: dict[str, Any] = {
        "schema": smoke_contract.SCHEMA,
        "status": smoke_contract.STATUS,
        "source_closure": {
            "source_manifest_sha256": "1" * 64,
            "source_root_sha256": "2" * 64,
            "member_hashes": member_hashes,
            "retained_no_follow_descriptors": True,
            "executing_entry_inode_bound": True,
            "executing_entry_name": "synthetic_cupy_smoke.py",
        },
        "runtime_closure": {
            "predecessor_lock_sha256": "3" * 64,
            "runtime_lock_sha256": "4" * 64,
            "inverse_transient_dtype": "<i4",
            "inverse_override_installed_before_any_reservoir_decode": True,
        },
        "numeric_tile": {
            "packet_bytes": 78_592,
            "all_encoder_self_checks_required_and_passed": True,
            "canonical_reencode_matches": True,
            "inverse_i32_dtype_verified_before_facade_cast": True,
            "inverse_transient_dtype": "<i4",
            "relative_mse_original_coordinates": 0.1,
        },
        "i32_stress_lifetime": {
            "input_index": 63,
            "expected_abs_max": 8_388_608,
            "observed_abs_max": 8_388_608,
            "installed_in_inherited_decoder_before_call": True,
            "inverse_output_dtype_before_facade": "<i4",
            "facade_retained_dtype": "<i4",
            "no_copy_or_downcast": True,
            "downstream_reconstruction_float64_abs_max": 4_096.0,
            "downstream_reconstruction_expected_abs_max": 4_096.0,
        },
        "aggregate_zero_frame": {
            "roles": ["gate", "up", "down_transposed"],
            "literal_aggregate_reencode_matches": True,
            "exact_inherited_role_abi": True,
            "frame_bytes": 3 * 78_592,
        },
        "traffic_ledgers": traffic,
        "payload_accessed": False,
        "model_or_qwen_path_discovered_or_enumerated": False,
        "claim_boundary":
            "source-free mechanics/runtime only; authorizes a separately bound Qwen pilot but is not a Qwen, MSE, universal-tail, fine-code, or inference-HBM result",
    }
    record["receipt_sha256"] = digest(smoke_contract.canonical_json(record))
    return record, member_hashes


def reseal(record: dict[str, Any]) -> None:
    record.pop("receipt_sha256", None)
    record["receipt_sha256"] = digest(smoke_contract.canonical_json(record))


class SmokeReceiptTests(unittest.TestCase):
    def validate(self, record: dict[str, Any], hashes: dict[str, str]) -> Any:
        return smoke_contract.validate_smoke_receipt(
            record, source_manifest_sha256="1" * 64,
            source_root_sha256="2" * 64,
            predecessor_lock_sha256="3" * 64,
            runtime_lock_sha256="4" * 64,
            source_member_hashes=hashes)

    def test_valid_receipt_binds_source_entry_i32_and_complete_ledgers(self) -> None:
        record, hashes = valid_smoke_record()
        result = self.validate(record, hashes)
        self.assertTrue(result["i32_stress_above_i16"])
        self.assertFalse(result["positive_claim_authority"])

    def test_receipt_seal_is_mandatory(self) -> None:
        record, hashes = valid_smoke_record()
        record["receipt_sha256"] = "0" * 64
        with self.assertRaises(smoke_contract.SmokeContractError):
            self.validate(record, hashes)

    def test_semantic_tampering_fails_even_when_resealed(self) -> None:
        base, hashes = valid_smoke_record()
        attacks = []
        wrong_entry = copy.deepcopy(base)
        wrong_entry["source_closure"]["executing_entry_name"] = "dispatcher.py"
        attacks.append(wrong_entry)
        weak_i32 = copy.deepcopy(base)
        weak_i32["i32_stress_lifetime"]["observed_abs_max"] = 32_767
        attacks.append(weak_i32)
        role_alias = copy.deepcopy(base)
        role_alias["aggregate_zero_frame"]["roles"][-1] = "down"
        attacks.append(role_alias)
        weak_host = copy.deepcopy(base)
        host = weak_host["traffic_ledgers"]["prebuffered_decode"] \
            ["host_memory_parse_and_integrity"]
        host["minimum_input_frame_full_scan_equivalents"] = 2
        host["minimum_input_frame_bytes_touched"] = 2 * 3 * 78_592
        attacks.append(weak_host)
        i16_scratch = copy.deepcopy(base)
        scratch = i16_scratch["traffic_ledgers"]["prebuffered_decode"] \
            ["scratch_lower_bound"]
        scratch["canonical_symbols_i32_bytes"] //= 2
        scratch["minimum_numeric_and_packet_scratch_bytes"] = (
            2 * 3 * 78_592 + scratch["canonical_symbols_i32_bytes"] +
            scratch["reconstruction_f32_bytes"])
        attacks.append(i16_scratch)
        fake_hbm = copy.deepcopy(base)
        fake_hbm["traffic_ledgers"]["prebuffered_decode"]["accelerator_hbm"] = {
            "measured": True, "read_bytes": 1,
            "read_amplification": 0.1, "below_2x_claim_authority": True}
        attacks.append(fake_hbm)
        for attack in attacks:
            reseal(attack)
            with self.subTest(attack=attacks.index(attack)), \
                    self.assertRaises(smoke_contract.SmokeContractError):
                self.validate(attack, hashes)

    def test_external_receipt_publication_is_exclusive_and_outside_package(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("frozen publication runtime is Linux")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            package = root / "package"
            outside = root / "receipts"
            package.mkdir()
            outside.mkdir()
            target = outside / "smoke.json"
            result = smoke_script.write_receipt_exclusive(
                target, b"{}\n", package)
            self.assertEqual(result["sha256"], digest(b"{}\n"))
            self.assertEqual(target.read_bytes(), b"{}\n")
            with self.assertRaises(FileExistsError):
                smoke_script.write_receipt_exclusive(target, b"x", package)
            descendant = package / "receipts"
            descendant.mkdir()
            with self.assertRaises(RuntimeError):
                smoke_script.write_receipt_exclusive(
                    descendant / "bad.json", b"x", package)


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("frozen renameat2 runtime is Linux")

    def test_no_replace_terminal_publication_and_rebind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "result"
            result = dispatcher.publish_atomic(
                output, {"A.bin": b"a", "RESULT.json": b"{}\n"},
                {"schema": dispatcher.COMPLETE_SCHEMA, "status": "TEST"})
            self.assertTrue(result["rename_noreplace"])
            self.assertTrue(result["completion_rename_noreplace"])
            self.assertTrue(result["final_members_rehashed_and_name_bound"])
            self.assertEqual(result["complete"]["name"], "COMPLETE.json")
            self.assertTrue((output / "COMPLETE.json").is_file())
            before = (output / "A.bin").read_bytes()
            with self.assertRaises(dispatcher.DispatchError):
                dispatcher.publish_atomic(
                    output, {"A.bin": b"replacement"},
                    {"schema": dispatcher.COMPLETE_SCHEMA, "status": "TEST"})
            self.assertEqual((output / "A.bin").read_bytes(), before)
            self.assertFalse(any(
                entry.name.startswith(".result.partial")
                for entry in parent.iterdir()))

    def test_failed_staging_is_narrowly_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "result"
            with self.assertRaises(dispatcher.DispatchError):
                dispatcher.publish_atomic(
                    output, {"../escape": b"x"},
                    {"schema": dispatcher.COMPLETE_SCHEMA, "status": "TEST"})
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.iterdir()), [])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "tactic-actual-coarse-n18-v6-source-test-v1",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "payload_accessed": False,
        "cuda_initialized": False,
        "network_accessed": False,
        "source_free_cupy_smoke_run": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
