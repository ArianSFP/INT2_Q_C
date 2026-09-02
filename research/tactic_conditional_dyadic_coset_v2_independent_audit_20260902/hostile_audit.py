#!/usr/bin/env python3
"""Independent, source-only hostile audit of TACTIC-DH384 v2.

This program authenticates every producer byte before it executes any producer
module.  Producer modules are compiled directly from the authenticated byte
strings; the producer directory is never added to ``sys.path`` and bytecode is
disabled.  It accepts no model, coarse-artifact, CUDA, network, or result input.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT_PREFIX = b"TACTIC-DH384-V2-INDEPENDENT-SOURCE-ROOT-v1\n"
EXPECTED_ROOT = "cd03644f0e1c36f1c568208d863c10bfd52959fb3dd5e47b6d5c41132dafb61d"
EXPECTED = {
    "README.md": (12502, "638d3688e638fb1dd82d68b387acc7c97eb25235073c1abc5f85f5af929a8c08"),
    "SOURCE_MANIFEST.json": (1291, "f8de593784638cf7719d08ddda7061f4912166021214fb7a2894862a53050662"),
    "cupy_preflight.py": (4764, "db358f080a8d77d27204f7d098fe1caac65e172edaf17ab6ca205900b5883553"),
    "design_lock.json": (8045, "6549003de0a33797baab9131eb02c720389541272aa44902d65b950ae29f1c9a"),
    "stage0_gate.py": (21006, "3bae2633ac38cef3db12c3327f967ae1d3bc2b7caab0639df92bec5a009288a3"),
    "tactic_v2_common.py": (14619, "1d007e47f075d7b4c746d53e5bffb999ebde4cc1a2b85835cf44e951c18c87ba"),
    "test_source_only.py": (5647, "367e47b62b6c5c3474626b026f6c995de50183a9fc37711fc3de853c48675177"),
    "verify_source.py": (17144, "34c43105fec9acfa4bf834b4c8c7f7ccb668111d50c5f8f76070caaccbfe9087"),
}


def authenticate(producer: Path) -> tuple[dict[str, bytes], list[dict[str, Any]], str]:
    producer = producer.resolve(strict=True)
    if not producer.is_dir() or producer.is_symlink():
        raise RuntimeError("producer must be a real directory")
    entries = list(producer.iterdir())
    if any(not item.is_file() or item.is_symlink() for item in entries):
        raise RuntimeError("producer has a non-regular or symlink member")
    names = {item.name for item in entries}
    if names != set(EXPECTED):
        raise RuntimeError(f"producer closure mismatch: {sorted(names)}")
    packets: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    preimage = bytearray(ROOT_PREFIX)
    for name in sorted(EXPECTED, key=str.casefold):
        payload = (producer / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected_bytes, expected_digest = EXPECTED[name]
        if (len(payload), digest) != (expected_bytes, expected_digest):
            raise RuntimeError(f"producer member mismatch: {name}")
        packets[name] = payload
        inventory.append({"name": name, "bytes": len(payload), "sha256": digest})
        preimage.extend(f"{name}\t{len(payload)}\t{digest}\n".encode("utf-8"))
    root = hashlib.sha256(preimage).hexdigest()
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"independent source root mismatch: {root}")
    return packets, inventory, root


def execute_authenticated_modules(packets: dict[str, bytes]) -> tuple[types.ModuleType, types.ModuleType]:
    """Execute only already-authenticated bytes, without import-path lookup."""
    sys.dont_write_bytecode = True
    common = types.ModuleType("tactic_v2_common")
    common.__file__ = "<authenticated:tactic_v2_common.py>"
    exec(compile(packets["tactic_v2_common.py"], common.__file__, "exec"), common.__dict__)
    previous = sys.modules.get("tactic_v2_common")
    sys.modules["tactic_v2_common"] = common
    try:
        stage = types.ModuleType("stage0_gate")
        stage.__file__ = "<authenticated:stage0_gate.py>"
        exec(compile(packets["stage0_gate.py"], stage.__file__, "exec"), stage.__dict__)
    finally:
        if previous is None:
            del sys.modules["tactic_v2_common"]
        else:
            sys.modules["tactic_v2_common"] = previous
    return common, stage


def independent_selector_packet() -> bytes:
    mask = (1 << 64) - 1
    state = (0x5441435449434448 ^ (18 * 0xD1B54A32D192ED03)) & mask
    table = bytearray()
    for _ in range(12 * 256):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        word = (z ^ (z >> 31)) & mask
        table.append(word & 7)
    header = json.dumps({
        "format": "TACTIC-DH384-UNIVERSAL-SELECTOR-v2",
        "generator": "SplitMix64",
        "generator_domain_u64_hex": "5441435449434448",
        "universal_ordinal": 17,
        "stages": 12,
        "states": 256,
        "ops": "swap/sign0/sign1",
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
    return header + table + bytes(16_384 - len(header) - len(table))


def fake_descriptor(relpath: str, dtype: str, byte_count: int) -> dict[str, Any]:
    return {
        "relpath": relpath,
        "bytes": byte_count,
        "sha256": "0" * 64,
        "dtype": dtype,
        "shape": [768, 2048],
    }


def duplicate_path_lock(common: types.ModuleType) -> dict[str, Any]:
    panels = []
    roles = ("gate", "up", "down_transposed")
    for panel_ordinal, kind in enumerate(("source", "decoded_gaussian", "structure_destroyed")):
        records = []
        for ordinal in range(18):
            records.append({
                "matrix_ordinal": ordinal,
                "expert_ordinal": ordinal // 3,
                "role": roles[ordinal % 3],
                "source": fake_descriptor("same-source.bin", "<bf16", 2 * 768 * 2048),
                "reconstruction": fake_descriptor("same-reconstruction.bin", "<f4", 4 * 768 * 2048),
                "symbols": fake_descriptor("same-symbols.bin", "<i2", 2 * 768 * 2048),
            })
        receipt = {
            "relpath": "same-receipt.json",
            "bytes": 1,
            "sha256": "0" * 64,
            "dtype": "strict-json-receipt",
        }
        panels.append({
            "id": f"panel-{panel_ordinal}",
            "kind": kind,
            "rate": {
                "actual_bpw": 307 / 128,
                "streams": 108,
                "bytes_per_stream": 78_592,
                "coarse_container_bytes": 6 * 18 * 78_592,
                "decode_reencode_verified": True,
                "all_stream_reservoirs_within_capacity": True,
                "roundtrip_receipt": receipt,
            },
            "reservoirs": [
                {
                    "stream_ordinal": index,
                    "relpath": "same-stream.bin",
                    "bytes": 78_592,
                    "sha256": "0" * 64,
                    "dtype": "opaque-coarse-stream",
                }
                for index in range(108)
            ],
            "records": records,
        })
    lock = {
        "schema": "tactic_dh384_actual_coarse_lock_v2",
        "root": "/tmp",
        "panels": panels,
    }
    lock["lock_sha256"] = hashlib.sha256(common.canonical_json(lock)).hexdigest()
    return lock


class AuditTests(unittest.TestCase):
    packets: dict[str, bytes]
    common: types.ModuleType
    stage: types.ModuleType

    def test_01_manifest_matches_members_but_is_not_an_external_execution_pin(self) -> None:
        manifest = json.loads(self.packets["SOURCE_MANIFEST.json"])
        rows = {row["name"]: (row["bytes"], row["sha256"]) for row in manifest["files"]}
        self.assertEqual(rows, {name: value for name, value in EXPECTED.items() if name != "SOURCE_MANIFEST.json"})
        verify_tree = ast.parse(self.packets["verify_source.py"])
        imports = [node for node in verify_tree.body if isinstance(node, ast.ImportFrom)]
        self.assertTrue(any(node.module == "tactic_v2_common" for node in imports))
        stage_tree = ast.parse(self.packets["stage0_gate.py"])
        stage_imports = [node for node in stage_tree.body if isinstance(node, ast.ImportFrom)]
        self.assertTrue(any(node.module == "tactic_v2_common" for node in stage_imports))
        text = self.packets["stage0_gate.py"].decode()
        self.assertNotIn("expected_source_root", text)
        self.assertNotIn("expected_manifest_sha256", text)

    def test_02_exact_qwen_sized_rate_and_page_ledger(self) -> None:
        values_matrix = 768 * 2048
        values_expert = 3 * values_matrix
        total_values = 6 * values_expert
        coarse_expert = 18 * 78_592
        fine_expert = 1152 * 48
        global_bytes = 24_576
        frame = 512 + coarse_expert + fine_expert
        container = global_bytes + 6 * frame
        self.assertEqual(Fraction(8 * coarse_expert, values_expert), Fraction(307, 128))
        self.assertEqual(Fraction(8 * fine_expert, values_expert), Fraction(12, 128))
        self.assertEqual(Fraction(8 * (global_bytes + 6 * 512), total_values), Fraction(1, 128))
        self.assertEqual(Fraction(8 * container, total_values), Fraction(5, 2))
        self.assertEqual((global_bytes // 4096, frame // 4096), (6, 359))
        self.assertEqual(Fraction(365, 360), Fraction(73, 72))

    def test_03_literal_384_bits_per_full_block_has_zero_rate_slack(self) -> None:
        self.assertEqual(48 * 8, 384)
        self.assertEqual(1152 * 384, 442_368)
        self.assertEqual(442_368 // 8, 55_296)
        self.assertEqual(8 * 8_847_360, 28_311_552 * 5 // 2)

    def test_04_selector_is_independently_canonical(self) -> None:
        packet = independent_selector_packet()
        self.assertEqual(len(packet), 16_384)
        self.assertEqual(hashlib.sha256(packet).hexdigest(),
                         "0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad")
        self.assertEqual(packet, self.common.selector_packet(self.common.universal_selector_table()))

    def test_05_continuous_span_projection_identity_is_sound(self) -> None:
        table = self.common.universal_selector_table()
        symbols = [((index * 101 + 7) % 1021) - 510 for index in range(64)]
        error = [math.sin(index * 0.37) + math.cos(index * 0.11) for index in range(64)]
        result = self.common.cpu_projection(symbols, error, role=2, table=table, rank=6)
        self.assertTrue(math.isclose(result["energy"], result["transformed_energy"],
                                     rel_tol=2e-12, abs_tol=2e-12))
        self.assertTrue(math.isclose(result["residual_energy"],
                                     result["energy"] - result["projected_energy"],
                                     rel_tol=0.0, abs_tol=1e-14))

    def test_06_finite_qc_trellis_and_output_scale_are_not_defined(self) -> None:
        trees = {
            name: ast.parse(payload)
            for name, payload in self.packets.items()
            if name.endswith(".py")
        }
        definitions = {
            node.name.casefold()
            for tree in trees.values()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        self.assertFalse(any("trellis" in name or "coset_decode" in name or "q12_decode" in name
                             for name in definitions))
        stage_text = self.packets["stage0_gate.py"].decode().casefold()
        self.assertNotIn("global_qc_tables_bytes", stage_text)
        self.assertNotIn("public rational output scale", stage_text)
        self.assertNotIn("q12_abs_max", stage_text)

    def test_07_unequal_shapes_and_tails_are_rejected_not_supported(self) -> None:
        text = self.packets["stage0_gate.py"].decode()
        self.assertIn('require(row["shape"] == [ROWS, COLS]', text)
        self.assertIn("ROWS = 768", text)
        self.assertIn("COLS = 2048", text)
        self.assertNotIn("valid_values", text)
        self.assertNotIn("tail_values", text)
        self.assertNotIn("pad_block", text)
        with self.assertRaises(self.common.ContractError):
            row = fake_descriptor("x.bin", "<bf16", 2 * 769 * 2051)
            row["shape"] = [769, 2051]
            self.stage._validate_file_descriptor(row, "<bf16", 2 * 768 * 2048)

    def test_08_lock_accepts_duplicate_files_and_self_asserted_roundtrip(self) -> None:
        lock = duplicate_path_lock(self.common)
        accepted = self.stage.validate_coarse_lock(lock)
        self.assertIs(accepted, lock)
        source_paths = [record["source"]["relpath"] for record in lock["panels"][0]["records"]]
        stream_paths = [row["relpath"] for row in lock["panels"][0]["reservoirs"]]
        self.assertEqual(len(set(source_paths)), 1)
        self.assertEqual(len(set(stream_paths)), 1)
        text = self.packets["stage0_gate.py"].decode()
        self.assertNotIn("decoded_source_sha256", text)
        self.assertNotIn("decoded_reconstruction_sha256", text)
        self.assertNotIn("canonical_reencode_sha256", text)

    @unittest.skipUnless(os.name == "posix", "HeldOutput is deliberately POSIX-only")
    def test_09_output_fault_leaves_public_partial_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "public-result"
            output = self.common.HeldOutput(final)
            try:
                output.write_new("universal_selector_packet.bin", b"partial")
            finally:
                output.close()
            self.assertTrue(final.is_dir())
            self.assertEqual((final / "universal_selector_packet.bin").read_bytes(), b"partial")
            self.assertFalse((final / "COMPLETE.json").exists())
        stage_text = self.packets["stage0_gate.py"].decode()
        self.assertNotIn("staging", stage_text)
        self.assertNotIn("COMPLETE.json", stage_text)
        self.assertNotIn("rename", stage_text)

    def test_10_source_panel_precedes_control_payload_opens(self) -> None:
        text = self.packets["stage0_gate.py"].decode()
        source_open = text.index('source_records, source_receipt_bytes = _open_panel(root, lock["panels"][0])')
        source_eval = text.index("source_result = evaluate_panel", source_open)
        decision = text.index('if source_result["decision"] != "HARD_REJECT_CONTINUOUS_SPAN_FAR_SHORT"', source_eval)
        controls = text.index('for panel in lock["panels"][1:]', decision)
        self.assertLess(source_open, source_eval)
        self.assertLess(source_eval, decision)
        self.assertLess(decision, controls)

    def test_11_runtime_and_review_authority_are_not_authenticated(self) -> None:
        stage = self.packets["stage0_gate.py"].decode()
        preflight = self.packets["cupy_preflight.py"].decode()
        for text in (stage, preflight):
            self.assertNotIn("runtime_environment_lock", text)
            self.assertNotIn("interpreter_sha256", text)
            self.assertNotIn("distribution_record_sha256", text)
            self.assertNotIn("review_receipt", text)
        self.assertIn("OPEN_AUTHENTICATED_ACTUAL_LOWER_RATE_TACTIC_DH384_V2", self.packets["tactic_v2_common.py"].decode())
        self.assertIn("SYNTHETIC_ONLY_TACTIC_DH384_V2", self.packets["tactic_v2_common.py"].decode())
        self.assertLess(stage.index("source_records, source_receipt_bytes"), stage.index("import cupy as cp"))

    def test_12_read_claim_is_declarative_not_an_inference_trace(self) -> None:
        stage = self.packets["stage0_gate.py"].decode()
        self.assertIn('"cold_read_amplification": 73.0 / 72.0', stage)
        self.assertNotIn("compressed_frame_reads", stage)
        self.assertNotIn("unique_pages", stage)
        self.assertNotIn("page_trace", stage)
        self.assertNotIn("decode_coarse_stream", stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    args = parser.parse_args()
    packets, inventory, root = authenticate(args.producer)
    common, stage = execute_authenticated_modules(packets)
    AuditTests.packets = packets
    AuditTests.common = common
    AuditTests.stage = stage
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = {
        "schema": "tactic_dh384_v2_independent_hostile_source_audit_v1",
        "status": "PASS_AUDIT_HARNESS_BLOCK_PRODUCER",
        "producer_root": root,
        "inventory": inventory,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "platform": {"python": sys.version, "os_name": os.name, "platform": sys.platform},
        "test_log": stream.getvalue().splitlines(),
        "claim_boundary": "Source-only hostile audit; no model, coarse payload, CUDA, network, or numerical compression result.",
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
