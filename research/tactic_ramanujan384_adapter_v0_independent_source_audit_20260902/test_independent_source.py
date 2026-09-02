#!/usr/bin/env python3
"""Independent hostile source audit for the frozen Ramanujan-384 adapter."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
import unittest
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np


AUDIT_ROOT = Path(__file__).resolve().parent
REPO = AUDIT_ROOT.parents[1]
PRODUCER = REPO / "research" / "tactic_ramanujan384_adapter_v0"
EXPECTED_MANIFEST_SHA256 = "287b8ad4c377956c9bb264d9d8731893a83e45180f75472f9b42968e3f20acde"
EXPECTED_ROOT_SHA256 = "2a66a5d745fc0a31e311cf6ab5f44836726ae341db977bca8eac314df61124ad"
OWN_VERIFIER_ROOT_SHA256 = "64669f3eeb9dd4f34a9fa36c9c6db592dcf5e37bdeb5ce149b3dbd51e2e24733"
HOLD = "HOLD_PAYLOAD_AUTHORITY_PENDING_LITERAL_WEIGHT_REPLAY_AND_BACKEND_STABLE_CONTROLS"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(name: str, filename: str):
    path = PRODUCER / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
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


packet = load("independent_ramanujan_packet", "packet.py")
codec = load("independent_ramanujan_codec", "ramanujan_codec.py")
container = load("independent_ramanujan_container", "container.py")
adapter = load("independent_ramanujan_adapter", "adapter.py")


def independent_totient(value: int) -> int:
    return sum(math.gcd(value, item) == 1 for item in range(1, value + 1))


def modular_rank(matrix: list[list[int]], prime: int = 1_000_000_007) -> int:
    rows = [list(value % prime for value in row) for row in matrix]
    if not rows:
        return 0
    height = len(rows)
    width = len(rows[0])
    rank = 0
    for column in range(width):
        pivot = next((row for row in range(rank, height) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], prime - 2, prime)
        rows[rank] = [(value * inverse) % prime for value in rows[rank]]
        pivot_row = rows[rank]
        for row in range(height):
            if row == rank or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                (left - factor * right) % prime
                for left, right in zip(rows[row], pivot_row, strict=True)
            ]
        rank += 1
        if rank == height:
            break
    return rank


def proper_divisors(value: int) -> tuple[int, ...]:
    return tuple(divisor for divisor in range(1, value) if value % divisor == 0)


def independent_packet_parse(payload: bytes) -> dict[str, object]:
    if len(payload) != 48:
        raise ValueError("size")
    body = payload[:44]
    if zlib.crc32(body) & 0xFFFFFFFF != struct.unpack("<I", payload[44:])[0]:
        raise ValueError("crc")
    value = int.from_bytes(body, "little")
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
    if role >= 3 or rank > 14:
        raise ValueError("header")
    support = []
    coefficients = []
    for _ in range(rank):
        atom = take(9)
        field = take(11)
        coefficient = field - 2048 if field & 1024 else field
        if atom >= 384 or coefficient == 0 or coefficient < -1023 or coefficient > 1023:
            raise ValueError("entry")
        support.append(atom)
        coefficients.append(coefficient)
    if support != sorted(set(support)) or len(support) != rank:
        raise ValueError("support")
    scale = float(struct.unpack("<e", struct.pack("<H", scale_bits))[0])
    if rank == 0:
        if scale_bits != 0:
            raise ValueError("zero scale")
    elif not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale")
    if value >> cursor:
        raise ValueError("padding")
    return {
        "role": role,
        "rank": rank,
        "scale": scale,
        "support": tuple(support),
        "coefficients": tuple(coefficients),
        "padding_bits": 352 - cursor,
    }


class SourceClosure(unittest.TestCase):
    def test_exact_producer_manifest_and_root(self):
        payload = (PRODUCER / "SOURCE_MANIFEST.json").read_bytes()
        self.assertEqual(digest(payload), EXPECTED_MANIFEST_SHA256)
        manifest = json.loads(payload)
        self.assertEqual(manifest["source_root_sha256"], EXPECTED_ROOT_SHA256)
        self.assertEqual([row["name"] for row in manifest["members"]],
                         sorted(row["name"] for row in manifest["members"]))
        observed = []
        for row in manifest["members"]:
            member = (PRODUCER / row["name"]).read_bytes()
            self.assertEqual(len(member), row["bytes"])
            self.assertEqual(digest(member), row["sha256"])
            observed.append({"name": row["name"], "bytes": len(member), "sha256": digest(member)})
        own_verifier_root = digest(json.dumps(
            observed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii"))
        insertion_order_root = digest(json.dumps(
            observed, sort_keys=False, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii"))
        self.assertEqual(own_verifier_root, OWN_VERIFIER_ROOT_SHA256)
        self.assertEqual(insertion_order_root, EXPECTED_ROOT_SHA256)
        self.assertNotEqual(own_verifier_root, manifest["source_root_sha256"])


class DictionaryAudit(unittest.TestCase):
    def test_complete_period_coverage_unique_ids_and_exact_shift_dimensions(self):
        periods = tuple(value for value in range(3, 128) if value & (value - 1))
        self.assertEqual(periods, codec.non_dyadic_periods())
        self.assertEqual(len(periods), 120)
        labels = codec.period_bank_labels()
        self.assertEqual(len(labels), 384)
        self.assertEqual(len(set(labels)), 384)
        self.assertEqual({period for period, _ in labels}, set(periods))
        for period in periods:
            phi = independent_totient(period)
            self.assertEqual(codec.totient(period), phi)
            sequence = [codec.ramanujan_sum(period, coordinate) for coordinate in range(period)]
            self.assertFalse(any(
                all(sequence[index] == sequence[index % divisor] for index in range(period))
                for divisor in proper_divisors(period)
            ), f"atom lacks exact period {period}")
            first_phi_shifts = [
                [codec.ramanujan_sum(period, coordinate - shift) for shift in range(phi)]
                for coordinate in range(period)
            ]
            self.assertEqual(modular_rank(first_phi_shifts), phi,
                             f"first phi shifts fail exact modular rank for {period}")
            selected = [shift for selected_period, shift in labels if selected_period == period]
            self.assertEqual(selected, list(range(len(selected))))
            self.assertLessEqual(len(selected), phi)

    def test_literal_dictionary_integer_shape_and_atom_uniqueness(self):
        basis = codec.build_public_dictionary(np)
        dictionary = basis["dictionary"]
        self.assertEqual(dictionary.shape, (4096, 384))
        self.assertTrue(np.array_equal(dictionary, np.rint(dictionary)))
        identities = [digest(dictionary[:, column].astype("<f8", copy=False).tobytes())
                      for column in range(dictionary.shape[1])]
        self.assertEqual(len(set(identities)), 384)
        self.assertTrue(np.all(basis["norms"] > 0))


class PacketAudit(unittest.TestCase):
    def test_independent_parser_roundtrips_all_roles_and_ranks(self):
        for role_index, role in enumerate(packet.ROLES):
            for rank in range(15):
                support = tuple(range(rank))
                coefficients = tuple(index + 1 if index & 1 else -index - 1 for index in range(rank))
                payload = packet.encode_packet(role, support, coefficients, 0.0 if rank == 0 else 2 ** -10)
                self.assertEqual(len(payload), 48)
                independent = independent_packet_parse(payload)
                producer = packet.decode_packet(payload)
                self.assertEqual(independent["role"], role_index)
                self.assertEqual(independent["rank"], rank)
                self.assertEqual(independent["support"], producer["support"])
                self.assertEqual(independent["coefficients"], producer["coefficients"])
                self.assertEqual(independent["padding_bits"], 352 - (42 + 20 * rank))

    def test_coefficient_extremes_and_crc_padding_aliases(self):
        payload = packet.encode_packet("gate", (0, 383), (-1023, 1023), 65504.0)
        row = independent_packet_parse(payload)
        self.assertEqual(row["coefficients"], (-1023, 1023))
        hostile = bytearray(payload)
        hostile[20] ^= 1
        with self.assertRaises(ValueError):
            independent_packet_parse(bytes(hostile))


class LedgerAndGeometryAudit(unittest.TestCase):
    def test_qwen_exact_header_page_rate_and_read_layout(self):
        per_role = 768 * 2048
        weights = 3 * per_role
        coarse = 307 * weights // 1024
        fine = 48 * 3 * (per_role // 4096)
        unpadded = 512 + coarse + fine
        physical = ((unpadded + 4095) // 4096) * 4096
        self.assertEqual((weights, coarse, fine, unpadded, physical),
                         (4718592, 1414656, 55296, 1470464, 1470464))
        self.assertEqual(physical // 4096, 359)
        self.assertEqual(Fraction(8 * physical, weights), Fraction(359, 144))
        self.assertEqual(Fraction(8 * (coarse + fine), weights), Fraction(319, 128))
        ledger = codec.expert_read_ledger(
            role_weights=(per_role,) * 3,
            coarse_artifact_bytes=coarse,
        )
        self.assertEqual(ledger["physical_bytes"], physical)
        self.assertEqual(ledger["physical_rate_bpw"], float(Fraction(359, 144)))
        self.assertEqual(ledger["external_storage_passes"], 1)
        self.assertEqual(ledger["external_read_amplification"], 1.0)
        self.assertFalse(ledger["accelerator_hbm_measured"])

    def test_variable_geometry_is_conditional_and_tails_are_charged(self):
        with self.assertRaisesRegex(container.ContainerError, "no integral"):
            container.expected_coarse_bytes(1, 1)
        blocks = container.expected_blocks(65, 65)
        self.assertEqual(blocks, (2, 2, 2))
        ledger = codec.expert_read_ledger(
            role_weights=(4097, 4097, 4097),
            coarse_artifact_bytes=math.ceil(307 * 3 * 4097 / 1024),
        )
        self.assertFalse(ledger["tail_free"])
        self.assertFalse(ledger["target_rate_eligible"])
        padded = adapter._blocks(np, np.ones(4097), 4096)
        self.assertEqual(padded.shape, (2, 4096))
        self.assertTrue(np.all(padded.reshape(-1)[4097:] == 0.0))


class GateControlAndAuthorityAudit(unittest.TestCase):
    def test_absolute_gate_precedes_all_control_generation_and_eight_seeds_are_frozen(self):
        source = (PRODUCER / "ramanujan_codec.py").read_text(encoding="utf-8")
        gate = source.index('if relative_mse > TARGET_D + 1e-15:')
        phase = source.index("phase_destroyed_blocks", gate)
        gaussian = source.index("moment_matched_gaussian_blocks", gate)
        self.assertLess(gate, phase)
        self.assertLess(gate, gaussian)
        self.assertEqual(len(codec.GAUSSIAN_SEEDS), 8)
        self.assertEqual(len(set(codec.GAUSSIAN_SEEDS)), 8)

    def test_adapter_does_not_reconstruct_or_rescore_decoded_container(self):
        source = (PRODUCER / "adapter.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_authenticated_expert")
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("container.decode_composite", calls)
        self.assertNotIn("codec.decode_packets_to_correction", calls)
        self.assertNotIn("container.decode_coarse", calls)
        decoded_uses = [
            node.slice.value
            for node in ast.walk(function)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "decoded_container"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ]
        self.assertEqual(set(decoded_uses), {"page_padding_bytes", "external_read_amplification"})

    def test_candidate_winner_is_selected_before_literal_packet_replay(self):
        source = (PRODUCER / "ramanujan_codec.py").read_text(encoding="utf-8")
        function = source[source.index("def encode_residual_blocks"):source.index("def decode_packets_to_correction")]
        select = function.index("best_sse = xp.where")
        packet_emit = function.index("packets = []")
        replay = function.index("decoded = decode_packets_to_correction")
        self.assertLess(select, packet_emit)
        self.assertLess(packet_emit, replay)
        self.assertNotIn("best_sse =", function[replay:])

    def test_control_rng_is_backend_owned_not_portable(self):
        parent = REPO / "research" / "mosaic_secondary_oracles_v0" / "residual_oracles.py"
        source = parent.read_text(encoding="utf-8")
        function = source[source.index("def moment_matched_gaussian_blocks"):]
        self.assertIn("generator = xp.random.RandomState(seed)", function)
        self.assertNotIn("counter", function.lower())

    def test_authentication_pins_content_but_does_not_open_manifest_files(self):
        source = (PRODUCER / "authenticated_io.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "authenticate_role")
        arguments = {argument.arg for argument in function.args.kwonlyargs}
        self.assertEqual(arguments, {
            "binding_path", "expected_binding_sha256", "audit_receipt_path",
            "coarse_artifact_path", "source_bf16_path", "coarse_reconstruction_f32_path",
        })
        self.assertNotIn("input_manifest_path", arguments)
        self.assertNotIn("auditor_source_manifest_path", arguments)
        for fragment in (
            "binding SHA256", "independent audit receipt SHA256",
            "literal coarse artifact hash", "literal source role hash",
            "literal coarse reconstruction hash", "O_NOFOLLOW", "st_nlink == 1",
        ):
            self.assertIn(fragment, source)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "tactic-ramanujan384-independent-source-test-receipt-v0",
        "status": HOLD,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "source_mechanics_passed": result.wasSuccessful(),
        "payload_authorized": False,
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
