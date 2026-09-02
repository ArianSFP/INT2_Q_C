#!/usr/bin/env python3
"""Independent hostile audit for the source-only UNIPOLAR-N18-307 v2 tree.

The producer package is authenticated against the exact pre-review inventory
below before any producer sibling is imported.  This audit deliberately does
not create a producer SOURCE_MANIFEST, runtime lock, payload, result, or CUDA
claim.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock


PACKAGE_NAME = "tactic_actual_coarse_n18_v2"
EXPECTED_INVENTORY = (
    ("dependency_graph.json", 1745, "0751a0a64ab18e9195508bfdfcf05edd11d43e4120e37036dfeb6ec4ded2d355"),
    ("design_lock.json", 4482, "be8fffe0eee387d07227ef6361fe78abc8e1ef0f901c8ba40b3184804f1a133c"),
    ("n18_common.py", 16189, "61c2719d7c58a968bee8bdcf456d9b6aa9402d134032ec7204c2015580f4dc0c"),
    ("POSTIMPLEMENTATION_REVIEW.md", 6710, "17ed841e6db0da8950b5688340eb460053b8e96a73b8af7768656636645a71b6"),
    ("preflight_gate.py", 11291, "36673ef80e0035e440dc36e275320b9e96f3636bc54f0c307354b2a5ff942dcf"),
    ("README.md", 8011, "2920fdf0dd29e475a0e95776d954071bcc3ae1a3a2d22cbdcd8a989452dd01c9"),
    ("runtime_contract.py", 12156, "1ef3c10a37de8366bcf72281dbf2e4aa40184e78ab74cc61dcf322089a755f1c"),
    ("runtime_environment_lock.json", 512, "79e72bba553ff09eb5a7e1a29d1082a1beafb25eba96fd7092ad701454855be0"),
    ("secure_io.py", 14211, "8dc7da600badce77769db529d9e53df5e5667c16ddf4828fc074a6c505beb8cd"),
    ("source_adapter.py", 8356, "48c672419c2b20dd36651055c01c3d94c2c3c5417c4f5edf5736c3d0812691eb"),
    ("test_source_only.py", 16779, "967dc7c8c3615a16e3fb1de69b4e86005e8ccfa261e1bc702837b527a8b6b73a"),
    ("verify_source.py", 11732, "e1b808fe5c75391ff7fcacaa390caa62a8a21d93aa7062be154a5a0f63e02680"),
)
EXPECTED_PRE_REVIEW_ROOT = "5719a483eef05571e93ab53eca80563a1a90a2c30d72644498fe0355735be917"
PRE_REVIEW_ROOT_PREFIX = b"TACTIC-N18-V2-PRE-REVIEW-ROOT-v1\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for packet in iter(lambda: stream.read(1 << 20), b""):
            digest.update(packet)
    return digest.hexdigest()


def _find_repo() -> Path:
    configured = os.environ.get("TACTIC_AUDIT_REPO_ROOT")
    if configured:
        return Path(configured).resolve(strict=True)
    here = Path(__file__).resolve(strict=True)
    for parent in here.parents:
        if (parent / "research" / PACKAGE_NAME).is_dir() and (
            parent / "docs" / "UNIVERSAL_SWIGLU_MOE_CODEC_CONTRACT.md"
        ).is_file():
            return parent
    raise RuntimeError("repository root not found")


def authenticate_before_import(repo: Path) -> tuple[Path, str]:
    package = repo / "research" / PACKAGE_NAME
    if package.is_symlink() or package.resolve(strict=True) != package:
        raise RuntimeError("producer package path is not canonical/no-follow")
    observed_names = sorted(path.name for path in package.iterdir())
    expected_names = sorted(row[0] for row in EXPECTED_INVENTORY)
    if observed_names != expected_names:
        raise RuntimeError("pre-review producer inventory drift")
    rows_by_name = {name: (expected_bytes, expected_sha256) for name, expected_bytes, expected_sha256 in EXPECTED_INVENTORY}
    lines: list[str] = []
    for name in sorted(rows_by_name, key=str.lower):
        expected_bytes, expected_sha256 = rows_by_name[name]
        path = package / name
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise RuntimeError(f"producer member is not a no-follow regular file: {name}")
        observed_sha256 = _sha256_file(path)
        if metadata.st_size != expected_bytes or observed_sha256 != expected_sha256:
            raise RuntimeError(f"producer member drift: {name}")
        lines.append(f"{name}\t{expected_bytes}\t{expected_sha256}\n")
    root = hashlib.sha256(PRE_REVIEW_ROOT_PREFIX + "".join(lines).encode("utf-8")).hexdigest()
    if root != EXPECTED_PRE_REVIEW_ROOT:
        raise RuntimeError("pre-review inventory content root mismatch")
    return package, root


REPO = _find_repo()
PACKAGE, AUTHENTICATED_ROOT = authenticate_before_import(REPO)
sys.path.insert(0, str(PACKAGE))

# Producer imports begin only after the exact tree authentication above.
import n18_common  # noqa: E402
import preflight_gate  # noqa: E402
import runtime_contract  # noqa: E402
import secure_io  # noqa: E402
import source_adapter  # noqa: E402


class PacketAndGeometryAudit(unittest.TestCase):
    def _packet(self, length: int, role: str = "gate") -> tuple[bytes, list[int]]:
        bits = [((index * 19 + 7) >> 2) & 1 for index in range(length)]
        payload, logical_bits = n18_common.bits_to_payload(bits)
        geometry = n18_common.MatrixGeometry(role, 769, 2051)
        return (
            n18_common.pack_reservoir(payload, logical_bits, 0.03125, geometry, 4),
            bits,
        )

    def test_exact_packet_hard_eof_and_canonical_bits(self) -> None:
        for length in (0, 1, 7, 8, 9, 19, 4095, n18_common.PAYLOAD_BITS):
            packet, bits = self._packet(length)
            self.assertEqual(len(packet), n18_common.RESERVOIR_BYTES)
            parsed = n18_common.parse_reservoir(packet)
            reader = parsed["reader"]
            self.assertEqual([reader.read_bit() for _ in range(length)], bits)
            reader.finish()
            with self.assertRaises(n18_common.ContractError):
                reader.read_bit()
            self.assertEqual(n18_common.canonical_bit_reencode(packet, iter(bits)), packet)
            with self.assertRaises(n18_common.ContractError):
                n18_common.canonical_bit_reencode(packet, iter(bits + [0]))

    def test_repaired_header_cannot_hide_noncanonical_terminal_or_fill_bits(self) -> None:
        packet, _ = self._packet(1)
        terminal = bytearray(packet)
        terminal[128] |= 1
        terminal[56:88] = hashlib.sha256(bytes(terminal[128:129])).digest()
        struct.pack_into("<I", terminal, 124, zlib.crc32(terminal[:124]) & 0xFFFFFFFF)
        with self.assertRaises(n18_common.ContractError):
            n18_common.parse_reservoir(bytes(terminal))

        fill = bytearray(packet)
        fill[-1] = 1
        with self.assertRaises(n18_common.ContractError):
            n18_common.parse_reservoir(bytes(fill))

    def test_arbitrary_shapes_tails_and_bounds(self) -> None:
        for intermediate, hidden in ((1, 1), (7, 11), (769, 2051), (1 << 24, 1024)):
            expert = n18_common.ExpertGeometry(intermediate, hidden)
            for geometry in expert.matrices:
                self.assertLessEqual(geometry.streams, n18_common.MAX_STREAMS_PER_MATRIX)
                self.assertGreaterEqual(geometry.valid_values(geometry.streams - 1), 1)
        with self.assertRaises(n18_common.ContractError):
            n18_common.ExpertGeometry((1 << 24) + 1, 1)
        with self.assertRaises(n18_common.ContractError):
            n18_common.ExpertGeometry(1 << 24, 1025)

    def test_owner_aware_bytes_but_missing_page_union(self) -> None:
        qwen = n18_common.ExpertGeometry(768, 2048)
        panel = n18_common.panel_ledger([qwen] * 6)
        self.assertTrue(
            all(
                row["cold_amplification_numerator"]
                == row["cold_amplification_denominator"]
                for row in panel["owners"]
            )
        )
        size = qwen.reservoir_bytes
        page = 4096
        page_amplifications = []
        for owner in range(6):
            start = owner * size
            end = start + size
            pages = (end - 1) // page - start // page + 1
            page_amplifications.append(pages * page / size)
        self.assertGreater(max(page_amplifications), 1.0)
        self.assertAlmostEqual(max(page_amplifications), 1_421_312 / 1_414_656)

    def test_legal_universal_shapes_can_breach_rate_and_cold_read_caps(self) -> None:
        portability_fixture = n18_common.ExpertGeometry(769, 2051)
        coarse_bpw = 8 * portability_fixture.reservoir_bytes / portability_fixture.values
        self.assertGreater(coarse_bpw, 2.5)
        unequal = n18_common.panel_ledger(
            [n18_common.ExpertGeometry(1, 1), n18_common.ExpertGeometry(768, 2048)]
        )
        self.assertGreater(unequal["owners"][0]["cold_amplification"], 2.0)


class StaticClosureAudit(unittest.TestCase):
    def test_all_source_modules_have_no_numeric_import(self) -> None:
        forbidden = {"cupy", "numpy", "scipy", "pynvml", "torch", "tensorflow"}
        for path in sorted(PACKAGE.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            roots: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    roots.add((node.module or "").split(".")[0])
            self.assertFalse(roots & forbidden, path.name)

    def test_authentication_fds_are_closed_before_live_path_import(self) -> None:
        source = (PACKAGE / "preflight_gate.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        bootstrap = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_bootstrap_source"
        )
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        bootstrap_source = ast.get_source_segment(source, bootstrap) or ""
        main_source = ast.get_source_segment(source, main) or ""
        self.assertIn("os.close(directory)", bootstrap_source)
        self.assertIn("return observed, packets", bootstrap_source)
        self.assertLess(
            main_source.index("_bootstrap_source(arguments.expected_source_manifest_sha256)"),
            main_source.index("from n18_common import"),
        )
        self.assertNotIn("TemporaryDirectory", source)
        self.assertNotIn("private source snapshot", source.lower())

    def test_bootstrap_has_no_manifest_row_or_total_materialization_cap(self) -> None:
        source = (PACKAGE / "preflight_gate.py").read_text(encoding="utf-8")
        bootstrap_start = source.index("def _bootstrap_source")
        authorization_start = source.index("def _authorization")
        bootstrap = source[bootstrap_start:authorization_start]
        self.assertNotIn("len(rows)", bootstrap)
        self.assertNotIn("EXPECTED_FILES", bootstrap)
        self.assertIn("packets[name] = value", bootstrap)

    def test_dependency_id_traversal_is_not_rejected_by_parser(self) -> None:
        source = (PACKAGE / "runtime_contract.py").read_text(encoding="utf-8")
        function_start = source.index("def authenticate_dependencies")
        telemetry_start = source.index("def validate_telemetry_receipt")
        function = source[function_start:telemetry_start]
        self.assertIn('f"{row[\'id\']}.py"', function)
        self.assertNotIn("dependency id", function)

    def test_fabricated_runtime_lock_passes_without_hashing_runtime(self) -> None:
        lock = {
            "schema": n18_common.ENVIRONMENT_SCHEMA,
            "status": "FROZEN_AUTHENTICATED_RUNTIME_READY",
            "interpreter": {
                "absolute_path": sys.executable,
                "bytes": -1,
                "sha256": "0" * 64,
                "python_version": "",
            },
            "distributions": [
                {
                    "name": name,
                    "version": "",
                    "record_sha256": "0" * 64,
                    "files_root_sha256": "0" * 64,
                }
                for name in sorted(("numpy", "cupy-cuda12x", "scipy", "nvidia-ml-py"))
            ],
        }
        lock["lock_sha256"] = hashlib.sha256(n18_common.canonical_json(lock)).hexdigest()
        accepted = runtime_contract.validate_environment_lock(n18_common.canonical_json(lock))
        self.assertEqual(accepted["interpreter"]["bytes"], -1)
        self.assertEqual(accepted["interpreter"]["sha256"], "0" * 64)

    def test_telemetry_schema_accepts_empty_identity_and_impossible_totals(self) -> None:
        value: dict[str, object] = {
            "cuda_visible_devices": "0",
            "cuda_logical_index": 0,
            "cuda_uuid": "",
            "cuda_pci_bus_id": "",
            "nvml_physical_index": "not-an-index",
            "nvml_uuid": "",
            "nvml_pci_bus_id": "",
            "device_name": "NVIDIA GeForce RTX 5090",
            "compute_capability": "",
            "driver_version": "",
            "cuda_runtime_version": "",
            "cupy_version": "",
            "numpy_version": "",
            "scipy_version": "",
            "pynvml_version": "",
            "logical_h2d_bytes": 1,
            "logical_d2h_bytes": 0,
            "model_h2d_bytes": 2,
            "cuda_event_h2d_ms": 0,
            "cuda_event_kernel_ms": 0,
            "cuda_event_d2h_ms": 0,
            "wall_seconds": 0,
            "telemetry_sampling_interval_ms": 0,
            "transfer_definition": "exact logical nbytes of explicitly enumerated host/device arrays; not claimed physical PCIe traffic",
        }
        for prefix in ("host_rss", "nvml_process", "nvml_device", "cupy_pool"):
            value[f"{prefix}_baseline_bytes"] = 0
            value[f"{prefix}_peak_bytes"] = 0
            value[f"{prefix}_delta_bytes"] = 0
        runtime_contract.validate_telemetry_receipt(value)


@unittest.skipUnless(os.name == "posix", "POSIX hostile cases require RunPod/Linux")
class PosixHostileAudit(unittest.TestCase):
    def test_held_input_rejects_leaf_and_ancestor_symlinks_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            real = base / "real"
            real.mkdir()
            target = real / "value.bin"
            target.write_bytes(b"abc")
            leaf = real / "leaf.bin"
            leaf.symlink_to(target)
            with self.assertRaises(Exception):
                secure_io.HeldRegularFile(str(leaf), maximum_bytes=3)
            ancestor = base / "ancestor"
            ancestor.symlink_to(real, target_is_directory=True)
            with self.assertRaises(Exception):
                secure_io.HeldRegularFile(str(ancestor / "value.bin"), maximum_bytes=3)
            with secure_io.HeldRegularFile(str(target), maximum_bytes=3) as held:
                with target.open("r+b") as stream:
                    stream.seek(0)
                    stream.write(b"abd")
                    stream.flush()
                    os.fsync(stream.fileno())
                with self.assertRaises(n18_common.ContractError):
                    held.verify_stable()

    def test_dependency_sources_and_ast_graph_authenticate(self) -> None:
        lock = (PACKAGE / "dependency_graph.json").read_bytes()
        with runtime_contract.authenticate_dependencies(lock, str(REPO)) as dependencies:
            self.assertEqual(len(dependencies.rows), 2)
            dependencies.held.verify_stable()

    def test_constructor_fault_leaves_private_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = str(Path(root) / "result")
            original_open = secure_io.os.open

            def fail_staging_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
                if isinstance(path, str) and ".result.staging." in path and flags & os.O_DIRECTORY:
                    raise OSError("injected staging directory open failure")
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(secure_io.os, "open", side_effect=fail_staging_open):
                with self.assertRaisesRegex(OSError, "injected"):
                    secure_io.CompletionLastPublisher(output, "a" * 64)
            leftovers = [path.name for path in Path(root).iterdir()]
            self.assertEqual(len(leftovers), 1)
            self.assertTrue(leftovers[0].startswith(".result.staging."))

    def test_postrename_rehash_fault_leaves_public_complete_tree(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result"
            self.assertFalse(output.exists())
            publisher = secure_io.CompletionLastPublisher(str(output), "a" * 64)
            publisher.write("payload.bin", b"payload")
            original_hash_all = secure_io._hash_all
            injected = False

            def corrupt_after_rename(descriptor: int, expected_bytes: int) -> str:
                nonlocal injected
                if not injected:
                    injected = True
                    # complete() has already renamed staging to the public leaf
                    # before its first _hash_all call.  Corrupt the same-length
                    # public payload so the real public rehash rejects it.
                    with (output / "payload.bin").open("wb") as stream:
                        stream.write(b"corrupt")
                        stream.flush()
                        os.fsync(stream.fileno())
                return original_hash_all(descriptor, expected_bytes)

            try:
                with mock.patch.object(
                    secure_io,
                    "_hash_all",
                    side_effect=corrupt_after_rename,
                ):
                    with self.assertRaisesRegex(n18_common.ContractError, "published file SHA-256"):
                        publisher.complete({"status": "TEST"})
                self.assertTrue(output.is_dir())
                self.assertTrue((output / "COMPLETE.json").is_file())
                self.assertEqual((output / "payload.bin").read_bytes(), b"corrupt")
                self.assertFalse(publisher.finished)
                with self.assertRaises(FileNotFoundError):
                    publisher.abort()
            finally:
                for attribute in ("staging_fd", "parent_fd"):
                    descriptor = getattr(publisher, attribute, -1)
                    if isinstance(descriptor, int) and descriptor >= 0:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                publisher.finished = True

    def test_preflight_is_inert_without_manifest_and_runtime(self) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONHOME")
        }
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            review = base / "missing_review.json"
            plan = base / "must_not_open_plan.json"
            output = base / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(PACKAGE / "preflight_gate.py"),
                    "--action",
                    "pilot",
                    "--authorization",
                    n18_common.PILOT_AUTHORIZATION,
                    "--expected-source-manifest-sha256",
                    "a" * 64,
                    "--review-receipt",
                    str(review),
                    "--source-plan",
                    str(plan),
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
            self.assertIn("SOURCE_MANIFEST.json", completed.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(review.exists())
            self.assertFalse(plan.exists())


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "schema": "tactic_actual_coarse_n18_v2_independent_audit_bootstrap_v1",
                "authenticated_pre_review_root": AUTHENTICATED_ROOT,
                "producer_imports_after_authentication": True,
                "manifest_created": False,
                "payload_opened": False,
                "cuda_launched": False,
            },
            sort_keys=True,
        )
    )
    unittest.main(verbosity=2)
