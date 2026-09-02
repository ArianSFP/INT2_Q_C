#!/usr/bin/env python3
"""Hostile standard-library tests for the v2 source-closure boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from n18_common import (
    ContractError,
    ExpertGeometry,
    MatrixGeometry,
    N,
    PAYLOAD_BITS,
    RESERVOIR_BYTES,
    ROLES,
    SOURCE_PLAN_SCHEMA,
    bits_to_payload,
    canonical_bit_reencode,
    canonical_json,
    pack_reservoir,
    panel_ledger,
    parse_reservoir,
    seed_pair,
    validate_fixed_ledger,
)
from runtime_contract import (
    authenticate_dependencies,
    validate_environment_lock,
    validate_review_receipt,
    validate_telemetry_receipt,
)
from secure_io import CompletionLastPublisher, HeldRegularFile
from source_adapter import EvaluationPhase, SourceFirstProtocol, parse_source_plan


PACKAGE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("TACTIC_REPO_ROOT", str(PACKAGE.parents[1]))).resolve()


def source_plan(expert_shapes: list[tuple[int, int]]) -> bytes:
    experts = []
    for ordinal, (intermediate, hidden) in enumerate(expert_shapes):
        matrices = []
        for role in ("gate", "up", "down"):
            shape = [hidden, intermediate] if role == "down" else [intermediate, hidden]
            values = intermediate * hidden
            matrices.append(
                {
                    "role": role,
                    "stored_shape": shape,
                    "absolute_path": f"/immutable/input/expert_{ordinal}_{role}.bf16",
                    "bytes": 2 * values,
                    "sha256": hashlib.sha256(f"{ordinal}:{role}".encode()).hexdigest(),
                }
            )
        experts.append(
            {
                "expert_ordinal": ordinal,
                "intermediate": intermediate,
                "hidden": hidden,
                "matrices": matrices,
            }
        )
    return canonical_json(
        {
            "schema": SOURCE_PLAN_SCHEMA,
            "status": "AUTHENTICATED_INPUT_BINDINGS_NO_DECODER_IDENTITY",
            "experts": experts,
            "control_protocol": "SOURCE_ABSOLUTE_DH384_PILOT_BEFORE_ANY_GAUSSIAN_OR_STRUCTURE_CONTROL",
            "claim_boundary": "input binding only",
        }
    )


def review_receipt(manifest: str, actions: list[str]) -> bytes:
    value = {
        "schema": "tactic_actual_coarse_n18_independent_review_v2",
        "status": "PASS_INDEPENDENT_SOURCE_REVIEW",
        "source_manifest_sha256": manifest,
        "allowed_actions": actions,
        "findings_sha256": hashlib.sha256(b"findings").hexdigest(),
    }
    value["receipt_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return canonical_json(value)


class LedgerTests(unittest.TestCase):
    def test_qwen_exact_307_over_128_and_one_x(self) -> None:
        value = validate_fixed_ledger()
        self.assertEqual(value["physical_bpw"], 307 / 128)
        self.assertEqual(value["reservoir_bytes"], 8_487_936)
        for row in value["owners"]:
            self.assertEqual(row["cold_amplification_numerator"], row["cold_amplification_denominator"])

    def test_arbitrary_shape_tail_is_charged(self) -> None:
        expert = ExpertGeometry(769, 2051)
        self.assertEqual(expert.values, 3 * 769 * 2051)
        for matrix in expert.matrices:
            self.assertEqual(matrix.streams, math.ceil(matrix.values / N))
            self.assertLess(matrix.valid_values(matrix.streams - 1), N)
        self.assertGreater(8 * expert.reservoir_bytes / expert.values, 307 / 128)

    def test_owner_aware_unequal_geometry(self) -> None:
        experts = [ExpertGeometry(1, 1), ExpertGeometry(768, 2048)]
        value = panel_ledger(experts)
        small = value["owners"][0]
        large = value["owners"][1]
        self.assertGreater(small["cold_amplification"], 1.0)
        self.assertLess(large["cold_amplification"], 1.0)
        self.assertEqual(sum(row["reservoir_bytes"] for row in value["owners"]), value["reservoir_bytes"])

    def test_bounds_reject_before_stream_scale(self) -> None:
        with self.assertRaises(ContractError):
            ExpertGeometry(1 << 30, 1 << 30)
        with self.assertRaises(ContractError):
            panel_ledger([])


class PacketTests(unittest.TestCase):
    def packet(self, length: int = 19) -> tuple[bytes, list[int]]:
        bits = [((index * 13 + 5) >> 1) & 1 for index in range(length)]
        payload, logical = bits_to_payload(bits)
        geometry = MatrixGeometry("gate", 769, 2051)
        return pack_reservoir(payload, logical, 0.03125, geometry, 4), bits

    def test_hard_eof_and_canonical_roundtrip(self) -> None:
        for length in (0, 1, 7, 8, 9, 19, 4095):
            packet, bits = self.packet(length)
            parsed = parse_reservoir(packet)
            reader = parsed["reader"]
            observed = [reader.read_bit() for _ in range(length)]
            self.assertEqual(observed, bits)
            reader.finish()
            with self.assertRaises(ContractError):
                reader.read_bit()
            self.assertEqual(canonical_bit_reencode(packet, bits), packet)

    def test_payload_capacity_and_terminal_pad(self) -> None:
        payload, logical = bits_to_payload((index & 1 for index in range(PAYLOAD_BITS)))
        geometry = MatrixGeometry("up", 768, 2048)
        packet = pack_reservoir(payload, logical, 1.0, geometry, 0)
        self.assertEqual(len(packet), RESERVOIR_BYTES)
        with self.assertRaises(ContractError):
            bits_to_payload((index & 1 for index in range(PAYLOAD_BITS + 1)))
        payload, logical = bits_to_payload([1])
        corrupted = bytes([payload[0] | 1])
        with self.assertRaises(ContractError):
            pack_reservoir(corrupted, logical, 1.0, geometry, 0)

    def test_header_payload_fill_and_reserved_tamper(self) -> None:
        packet, _bits = self.packet()
        for index in (0, 104, len(packet) - 1):
            mutated = bytearray(packet)
            mutated[index] ^= 1
            with self.assertRaises(ContractError):
                parse_reservoir(bytes(mutated))

    def test_terminal_pad_tamper_even_with_digest_and_crc_repaired(self) -> None:
        packet, _bits = self.packet(1)
        mutated = bytearray(packet)
        mutated[128] |= 1
        mutated[56:88] = hashlib.sha256(bytes(mutated[128:129])).digest()
        struct.pack_into("<I", mutated, 124, zlib.crc32(mutated[:124]) & 0xFFFFFFFF)
        with self.assertRaises(ContractError):
            parse_reservoir(bytes(mutated))

    def test_canonical_decision_mismatch(self) -> None:
        packet, bits = self.packet()
        wrong = list(bits)
        wrong[0] ^= 1
        with self.assertRaises(ContractError):
            canonical_bit_reencode(packet, wrong)

    def test_seed_is_shape_role_coordinate_not_expert(self) -> None:
        base = seed_pair(0, 768, 2048, 0)
        self.assertEqual(base, seed_pair(0, 768, 2048, 0))
        self.assertNotEqual(base, seed_pair(1, 768, 2048, 0))
        self.assertNotEqual(base, seed_pair(0, 769, 2048, 0))
        self.assertNotEqual(base, seed_pair(0, 768, 2048, 1))


class AdapterTests(unittest.TestCase):
    def test_universal_different_shapes_and_down_transpose(self) -> None:
        plan = parse_source_plan(source_plan([(7, 11), (769, 2051)]))
        self.assertEqual(len(plan.experts), 2)
        self.assertEqual([row.canonical_role for row in plan.experts[0].matrices], list(ROLES))
        self.assertTrue(plan.experts[0].matrices[2].transpose_on_ingest)
        self.assertEqual(plan.experts[1].geometry, ExpertGeometry(769, 2051))

    def test_identity_and_role_order_reject(self) -> None:
        value = json.loads(source_plan([(7, 11)]))
        value["model"] = "qwen"
        with self.assertRaises(ContractError):
            parse_source_plan(canonical_json(value))
        value = json.loads(source_plan([(7, 11)]))
        value["experts"][0]["matrices"].reverse()
        with self.assertRaises(ContractError):
            parse_source_plan(canonical_json(value))

    def test_huge_geometry_rejects(self) -> None:
        value = json.loads(source_plan([(7, 11)]))
        value["experts"][0]["intermediate"] = 1 << 30
        with self.assertRaises(ContractError):
            parse_source_plan(canonical_json(value))

    def test_source_first_and_dh384_nonconverse(self) -> None:
        state = SourceFirstProtocol()
        with self.assertRaises(ContractError):
            state.authorize_controls()
        state.record_source_gate({role: False for role in ROLES})
        self.assertIs(state.phase, EvaluationPhase.STOP_DH384_ONLY)
        self.assertFalse(state.cage_is_killed)
        with self.assertRaises(ContractError):
            state.authorize_controls()
        survivor = SourceFirstProtocol()
        survivor.record_source_gate({"gate": True, "up": False, "down_transposed": False})
        survivor.authorize_controls()


class AuthorityTests(unittest.TestCase):
    def test_checked_in_environment_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContractError, "intentionally unfrozen"):
            validate_environment_lock((PACKAGE / "runtime_environment_lock.json").read_bytes())

    def test_review_receipt_is_manifest_and_action_bound(self) -> None:
        manifest = "a" * 64
        value = review_receipt(manifest, ["pilot"])
        validate_review_receipt(value, manifest, "pilot")
        with self.assertRaises(ContractError):
            validate_review_receipt(value, "b" * 64, "pilot")
        with self.assertRaises(ContractError):
            validate_review_receipt(value, manifest, "full")

    def telemetry(self) -> dict[str, object]:
        value: dict[str, object] = {
            "cuda_visible_devices": "0",
            "cuda_logical_index": 0,
            "cuda_uuid": "GPU-x",
            "cuda_pci_bus_id": "0000:16:00.0",
            "nvml_physical_index": 0,
            "nvml_uuid": "GPU-x",
            "nvml_pci_bus_id": "0000:16:00.0",
            "device_name": "NVIDIA GeForce RTX 5090",
            "compute_capability": "12.0",
            "driver_version": "x",
            "cuda_runtime_version": "x",
            "cupy_version": "x",
            "numpy_version": "x",
            "scipy_version": "x",
            "pynvml_version": "x",
            "logical_h2d_bytes": 1,
            "logical_d2h_bytes": 2,
            "model_h2d_bytes": 0,
            "cuda_event_h2d_ms": 0.1,
            "cuda_event_kernel_ms": 0.2,
            "cuda_event_d2h_ms": 0.1,
            "wall_seconds": 1.0,
            "telemetry_sampling_interval_ms": 2,
            "transfer_definition": "exact logical nbytes of explicitly enumerated host/device arrays; not claimed physical PCIe traffic",
        }
        for prefix in ("host_rss", "nvml_process", "nvml_device", "cupy_pool"):
            value[f"{prefix}_baseline_bytes"] = 10
            value[f"{prefix}_peak_bytes"] = 12
            value[f"{prefix}_delta_bytes"] = 2
        return value

    def test_telemetry_uuid_pci_and_delta(self) -> None:
        value = self.telemetry()
        validate_telemetry_receipt(value)
        value["nvml_uuid"] = "GPU-y"
        with self.assertRaises(ContractError):
            validate_telemetry_receipt(value)

    @unittest.skipUnless(os.name == "posix", "held-FD contract is POSIX-only")
    def test_dependency_sources_held_and_import_graph_exact(self) -> None:
        lock = (PACKAGE / "dependency_graph.json").read_bytes()
        with authenticate_dependencies(lock, str(REPO)) as dependencies:
            self.assertEqual(len(dependencies.rows), 2)
            dependencies.held.verify_stable()


@unittest.skipUnless(os.name == "posix", "no-follow/publication contract is POSIX-only")
class PosixIOTests(unittest.TestCase):
    def test_held_leaf_and_ancestor_symlinks_reject(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            real = Path(root) / "real"
            real.mkdir()
            target = real / "value.bin"
            target.write_bytes(b"abc")
            leaf = real / "leaf.bin"
            leaf.symlink_to(target)
            with self.assertRaises(Exception):
                HeldRegularFile(str(leaf), maximum_bytes=3)
            parent_link = Path(root) / "parent_link"
            parent_link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(Exception):
                HeldRegularFile(str(parent_link / "value.bin"), maximum_bytes=3)
            with HeldRegularFile(
                str(target),
                maximum_bytes=3,
                expected_bytes=3,
                expected_sha256=hashlib.sha256(b"abc").hexdigest(),
            ) as held:
                self.assertEqual(held.read(), b"abc")

    def test_completion_last_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = str(Path(root) / "result")
            with CompletionLastPublisher(output, "a" * 64) as publisher:
                publisher.write("payload.bin", b"payload")
                publisher.complete({"status": "TEST"})
            self.assertTrue((Path(output) / "COMPLETE.json").is_file())
            self.assertEqual(
                sorted(path.name for path in Path(output).iterdir()),
                ["ARTIFACTS.json", "COMPLETE.json", "payload.bin"],
            )
            with self.assertRaises(FileExistsError):
                CompletionLastPublisher(output, "a" * 64)


class ClosureTests(unittest.TestCase):
    def test_manifest_and_complete_verifier(self) -> None:
        from verify_source import verify_package

        receipt = verify_package(PACKAGE, REPO)
        self.assertEqual(receipt["status"], "PASS_SOURCE_CLOSURE_RUNTIME_INTENTIONALLY_BLOCKED")
        self.assertEqual(receipt["payloads_opened"], 0)

    def test_manifest_tamper_and_extra_member_reject(self) -> None:
        from verify_source import VerificationError, verify_package

        with tempfile.TemporaryDirectory() as root:
            clone = Path(root) / "package"
            shutil.copytree(PACKAGE, clone)
            with (clone / "README.md").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(VerificationError):
                verify_package(clone, REPO)

    @unittest.skipUnless(os.name == "posix", "preflight authority order is POSIX-only")
    def test_preflight_unfrozen_environment_blocks_before_source_plan(self) -> None:
        manifest_raw = (PACKAGE / "SOURCE_MANIFEST.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            review = Path(root) / "review.json"
            review.write_bytes(review_receipt(manifest_sha, ["pilot"]))
            output = Path(root) / "output"
            missing_plan = Path(root) / "must_not_open_source_plan.json"
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in ("PYTHONPATH", "PYTHONHOME")
            }
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(PACKAGE / "preflight_gate.py"),
                    "--action",
                    "pilot",
                    "--authorization",
                    "OPEN_AUTHENTICATED_UNIPOLAR_N18_307_PILOT_V2",
                    "--expected-source-manifest-sha256",
                    manifest_sha,
                    "--review-receipt",
                    str(review),
                    "--source-plan",
                    str(missing_plan),
                    "--repo-root",
                    str(REPO),
                    "--output",
                    str(output),
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("runtime environment intentionally unfrozen", completed.stderr)
            self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as root:
            clone = Path(root) / "package"
            shutil.copytree(PACKAGE, clone)
            (clone / "extra").write_bytes(b"x")
            with self.assertRaises(VerificationError):
                verify_package(clone, REPO)


if __name__ == "__main__":
    unittest.main(verbosity=2)
