#!/usr/bin/env python3
"""Independent source-only review tests for Ramanujan-384 authority v1."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRODUCER = REPO / "research" / "tactic_ramanujan384_adapter_v1_authority"
V0 = REPO / "research" / "tactic_ramanujan384_adapter_v0"
EXPECTED_MANIFEST = "f4ba72b9371d77ad4347d5a4fe377677473844dd696032e662acc6cd3bde22b4"
EXPECTED_ROOT = "6840b6a0eb4f2856f84c610ba11888382ecca257d88ebda7f5b49c0de9f3b3c5"
STATUS = "SOURCE_REPAIRS_SUBSTANTIALLY_CLOSE_V0__HOLD_PAYLOAD_FOR_RUNTIME_SCALABILITY_AND_COARSE_DECODER_AUDIT"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def load(name: str, root: Path, filename: str):
    spec = importlib.util.spec_from_file_location(name, root / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


manifest_module = load("review_v1_manifest", PRODUCER, "manifest.py")
contract = load("review_v1_contract", PRODUCER, "contract.py")
controls = load("review_v1_controls", PRODUCER, "stable_controls.py")
codec = load("review_v1_codec", PRODUCER, "codec_authority.py")
trace = load("review_v1_trace", PRODUCER, "read_trace.py")
fixture = load("review_v1_fixture", PRODUCER, "run_source_free_fixture.py")
adapter = load("review_v1_adapter", PRODUCER, "adapter.py")


def parse_packet(payload: bytes) -> dict[str, object]:
    if len(payload) != 48 or zlib.crc32(payload[:44]) & 0xFFFFFFFF != struct.unpack("<I", payload[44:])[0]:
        raise ValueError("packet")
    value = int.from_bytes(payload[:44], "little")
    cursor = 0

    def take(width: int) -> int:
        nonlocal cursor
        result = (value >> cursor) & ((1 << width) - 1)
        cursor += width
        return result

    if take(16) != 0x4652 or take(4) != 1:
        raise ValueError("identity")
    role = take(2)
    rank = take(4)
    scale_bits = take(16)
    support, coefficients = [], []
    for _ in range(rank):
        support.append(take(9))
        field = take(11)
        coefficients.append(field - 2048 if field & 1024 else field)
    if role >= 3 or rank > 14 or support != sorted(set(support)) or value >> cursor:
        raise ValueError("canonical")
    scale = float(struct.unpack("<e", struct.pack("<H", scale_bits))[0])
    return {"role": role, "rank": rank, "scale": scale,
            "support": tuple(support), "coefficients": tuple(coefficients)}


def splitmix_reference(seed: int, count: int) -> np.ndarray:
    mask = (1 << 64) - 1
    state = seed
    result = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & mask
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        result.append((value ^ (value >> 31)) & mask)
    return np.asarray(result, dtype="<u8")


class ClosureReview(unittest.TestCase):
    def test_v1_manifest_members_and_canonical_root(self):
        payload = (PRODUCER / "SOURCE_MANIFEST.json").read_bytes()
        self.assertEqual(digest(payload), EXPECTED_MANIFEST)
        document = json.loads(payload)
        rows = []
        for row in document["members"]:
            member = (PRODUCER / row["name"]).read_bytes()
            self.assertEqual((len(member), digest(member)), (row["bytes"], row["sha256"]))
            rows.append({"name": row["name"], "bytes": len(member), "sha256": digest(member)})
        self.assertEqual(digest(canonical(sorted(rows, key=lambda row: row["name"]))), EXPECTED_ROOT)
        self.assertEqual(manifest_module.source_root(reversed(rows)), EXPECTED_ROOT)
        self.assertEqual(set(path.name for path in PRODUCER.iterdir()),
                         {row["name"] for row in rows} | {"SOURCE_MANIFEST.json"})

    def test_v0_root_defect_remains_reproducible_and_pinned(self):
        document = json.loads((V0 / "SOURCE_MANIFEST.json").read_text())
        rows = []
        for row in document["members"]:
            member = (V0 / row["name"]).read_bytes()
            rows.append({"name": row["name"], "bytes": len(member), "sha256": digest(member)})
        sorted_key = digest(canonical(rows))
        insertion = digest(json.dumps(rows, sort_keys=False, separators=(",", ":"),
                                      allow_nan=False).encode("ascii"))
        self.assertEqual(sorted_key, "64669f3eeb9dd4f34a9fa36c9c6db592dcf5e37bdeb5ce149b3dbd51e2e24733")
        self.assertEqual(insertion, "2a66a5d745fc0a31e311cf6ab5f44836726ae341db977bca8eac314df61124ad")


class GeometryReview(unittest.TestCase):
    def test_qwen_ledger_exact_and_tail_score_domain(self):
        shape = contract.define_shape(768, 2048)
        ledger = contract.physical_ledger(shape)
        self.assertEqual(shape.total_values, 4718592)
        self.assertEqual(shape.coarse_bytes, 1414656)
        self.assertEqual(ledger["fine_bytes"], 55296)
        self.assertEqual(ledger["physical_bytes"], 1470464)
        self.assertEqual(Fraction(8 * ledger["physical_bytes"], shape.total_values), Fraction(359, 144))
        tail = contract.define_shape(32, 160)
        self.assertEqual((tail.last_block_valid_values, tail.tail_values_per_role), (1024, 3072))
        coordinate = np.arange(tail.role_values, dtype=np.float64)
        source = 0.01 * ((coordinate % 7) - 3)
        encoded = codec.encode_role(np, source, np.zeros_like(source), tail, "gate")
        self.assertFalse(encoded["tail_padding_scored"])
        self.assertEqual(encoded["input_sse"], float(np.sum(source * source, dtype=np.float64)))

    def test_declared_extreme_shape_can_exceed_container_uint32_block_field(self):
        shape = contract.define_shape(1 << 22, 1 << 22)
        self.assertEqual(shape.blocks_per_role, 1 << 32)
        self.assertGreater(shape.blocks_per_role, (1 << 32) - 1)
        container_source = (V0 / "container.py").read_text(encoding="utf-8")
        self.assertIn('PREFIX = struct.Struct("<8sIIIIIIQQ32s32s32s")', container_source)
        self.assertIn("math.ceil(unpadded / page_bytes)",
                      (PRODUCER / "contract.py").read_text(encoding="utf-8"))


class CandidateAndReplayReview(unittest.TestCase):
    def test_every_representable_candidate_is_packet_decoded_before_selection(self):
        shape = contract.define_shape(32, 160)
        values = np.arange(shape.role_values, dtype=np.float64)
        source = 0.01 * ((values % 7) - 3) + 0.002 * ((values % 11) - 5)
        prepared = codec.prepare_basis(np)
        encoded = codec.encode_role(np, source, np.zeros_like(source), shape, "gate", prepared)
        self.assertTrue(encoded["all_defined_candidates_packet_replayed_before_selection"])
        self.assertTrue(encoded["winner_stream_replayed_after_selection"])
        self.assertTrue(all(count >= 1 for count in encoded["candidate_packet_replays_per_block"]))
        packets = [encoded["stream"][offset:offset + 48]
                   for offset in range(0, len(encoded["stream"]), 48)]
        self.assertTrue(all(parse_packet(row)["role"] == 0 for row in packets))

    def test_literal_file_composite_has_independent_weight_and_score_replay(self):
        auth = load("review_v1_auth", PRODUCER, "authenticated_io.py")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            role_inputs, coarse_decoder = fixture.build_authenticated_fixture(directory)
            output = directory / "authority.trm384"
            result = adapter.run_authenticated_expert(
                np, role_inputs=role_inputs, coarse_decoder=coarse_decoder,
                composite_output_path=output,
            )
            self.assertTrue(output.is_file())
            authenticated = [auth.authenticate_role(**row) for row in role_inputs]
            prepared = codec.prepare_basis(np)
            decoded = codec.decode_literal_composite(
                np, composite=output.read_bytes(),
                shape=contract.define_shape(64, 64), coarse_decoder=coarse_decoder,
                expected_coarse_f32_sha256={
                    row["role"]: row["coarse_reconstruction_sha256"] for row in authenticated
                }, prepared=prepared,
            )
            independent = codec.independent_score(
                np, {row["role"]: row["source"] for row in authenticated},
                decoded["reconstructions"], result["physical_rate_bpw"],
            )
            self.assertEqual(independent["remaining_sse"], result["remaining_sse"])
            self.assertEqual(independent["relative_mse"], result["relative_mse"])
            self.assertEqual(independent["F"], result["F"])
            self.assertTrue(result["literal_composite_reconstructed_to_weights"])
            self.assertTrue(result["independent_source_domain_fp64_rescore"])


class ControlsAuthenticationAndReadReview(unittest.TestCase):
    def test_splitmix_reference_and_control_bytes_are_repeatable(self):
        for seed in (0, 1, 99, (1 << 64) - 1):
            self.assertTrue(np.array_equal(controls.splitmix64_words(seed, 32),
                                           splitmix_reference(seed, 32)))
        reference = np.arange(8192, dtype=np.float64).reshape(2, 4096)
        valid = (4096, 1024)
        first = controls.moment_matched_blocks(np, reference, 10619863, valid)
        second = controls.moment_matched_blocks(np, reference, 10619863, valid)
        self.assertEqual(controls.host_bytes(np, first), controls.host_bytes(np, second))
        source = (PRODUCER / "stable_controls.py").read_text(encoding="utf-8")
        self.assertNotIn("np.random", source)
        self.assertNotIn("xp.random", source)

    def test_actual_manifest_paths_are_opened_and_hash_checked(self):
        source = (PRODUCER / "authenticated_io.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(row for row in tree.body
                        if isinstance(row, ast.FunctionDef) and row.name == "authenticate_role")
        names = {row.arg for row in function.args.kwonlyargs}
        self.assertIn("input_manifest_path", names)
        self.assertIn("auditor_source_manifest_path", names)
        for fragment in ("actual input manifest SHA256", "actual auditor source manifest SHA256",
                         'v0.strict_json(input_payload', 'v0.strict_json(auditor_payload'):
            self.assertIn(fragment, source)

    def test_read_trace_is_one_data_event_but_denies_physical_telemetry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object"
            path.write_bytes(bytes(8192))
            payload, record = trace.read_once(path, 8192)
            self.assertEqual(len(payload), 8192)
            self.assertEqual(record["data_read_calls"], 1)
            self.assertEqual(record["eof_probe_calls"], 1)
            self.assertEqual(record["instrumented_file_read_amplification"], 1.0)
            self.assertTrue(record["instrumented_trace_is_not_physical_storage_or_hbm_telemetry"])
            self.assertFalse(record["accelerator_hbm_measured"])


class CuPyArchitectureReview(unittest.TestCase):
    def test_basis_and_gram_are_shared_but_candidate_scratch_is_not(self):
        source = (PRODUCER / "codec_authority.py").read_text(encoding="utf-8")
        self.assertIn("prepared = prepare_basis(xp) if prepared is None else prepared", source)
        self.assertIn('gram = prepared["gram"]', source)
        self.assertIn("decode_packets_to_correction(xp, (candidate,)", source)
        self.assertIn("float(xp.sum(error * error, dtype=xp.float64).item())", source)
        v0_source = (V0 / "ramanujan_codec.py").read_text(encoding="utf-8")
        decode = v0_source[v0_source.index("def decode_packets_to_correction"):]
        self.assertIn("result = xp.zeros", decode)
        self.assertIn("packet = _load_packet_module()", decode)

    def test_bundled_cupy_fixture_rate_gate_precedes_control_panel(self):
        shape = contract.define_shape(64, 64)
        self.assertGreater(contract.physical_ledger(shape)["physical_rate_bpw"], 2.5)
        source = (PRODUCER / "adapter.py").read_text(encoding="utf-8")
        rate_gate = source.index("if not (2.15 <= physical_rate <= 2.5):")
        controls_call = source.index("panel = _control_panel", rate_gate)
        self.assertLess(rate_gate, controls_call)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "tactic-ramanujan384-authority-v1-independent-source-review-v0",
        "status": STATUS,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "payload_authorized": False,
        "qwen_payload_accessed": False,
        "coarse_model_payload_accessed": False,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)

