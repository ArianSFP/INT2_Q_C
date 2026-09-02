#!/usr/bin/env python3
"""Independent hostile source-only audit for STRATA-RM6 v0.

This program is deliberately given only the frozen source package, the frozen
STRATA independent auditor, and (optionally) a source-free CuPy smoke receipt.
It has no Qwen/coarse/control payload argument or discovery path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "c8d56e045159e3af613f02c4d5d97c70e8f8b4383b3fbf282d384b08f74b7300"
)
EXPECTED_SOURCE_ROOT_SHA256 = (
    "d17718615dedebca08ead66c0555e9d649768a353f3a55d169a9bf400f11bd32"
)
EXPECTED_AUDITOR_SHA256 = (
    "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
)
EXPECTED_AUDITOR_BYTES = 116_835
AUDIT_HEADER = struct.Struct("<4sBBBBIHHHBB6BIQH")
AUDIT_HEADER_BYTES = AUDIT_HEADER.size


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def independent_bit_reverse(n: int) -> np.ndarray:
    depth = n.bit_length() - 1
    if n < 2 or (1 << depth) != n:
        raise ValueError("power-of-two length required")
    return np.asarray([
        int(f"{index:0{depth}b}"[::-1], 2) for index in range(n)
    ], dtype=np.int64)


def independent_polar(bits: Any) -> np.ndarray:
    out = np.asarray(bits, dtype=np.uint8).copy()
    width = 1
    while width < out.size:
        for start in range(0, out.size, 2 * width):
            out[start:start + width] ^= out[start + width:start + 2 * width]
        width *= 2
    return out


def independent_rm_dimension(order: int, variables: int) -> int:
    return sum(math.comb(variables, degree) for degree in range(order + 1))


def independent_information_positions(variables: int, order: int) -> np.ndarray:
    threshold = variables - order
    return np.asarray([phase for phase in range(1 << variables)
                       if phase.bit_count() >= threshold], dtype=np.int64)


def independent_plane(info: Any, variables: int, order: int,
                      frozen_external: Any) -> np.ndarray:
    n = 1 << variables
    reverse = independent_bit_reverse(n)
    positions = independent_information_positions(variables, order)
    source_info = np.asarray(info, dtype=np.uint8)
    frozen = np.asarray(frozen_external, dtype=np.uint8)
    if source_info.shape != (positions.size,) or frozen.shape != (n,):
        raise ValueError("independent plane geometry")
    internal = frozen[reverse].copy()
    internal[positions] = source_info
    return independent_polar(internal[reverse])


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Audit:
    def __init__(self) -> None:
        self.tests: list[dict[str, Any]] = []
        self.findings: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, **evidence: Any) -> None:
        self.tests.append({"name": name, "passed": bool(condition), **evidence})

    def finding(self, identifier: str, severity: str, title: str,
                disposition: str, **evidence: Any) -> None:
        self.findings.append({"id": identifier, "severity": severity,
                              "title": title, "disposition": disposition,
                              **evidence})


def rejected(call: Callable[[], Any]) -> tuple[bool, str | None]:
    try:
        call()
    except Exception as error:  # hostile test: exact class is not trusted
        return True, f"{type(error).__name__}: {error}"
    return False, None


def independently_verify_source(package: Path, audit: Audit) -> dict[str, Any]:
    manifest_raw = (package / "SOURCE_MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)
    observed = []
    rows_ok = True
    for row in manifest.get("members", []):
        path = package / str(row["name"])
        raw = path.read_bytes()
        row_ok = len(raw) == int(row["bytes"]) and sha(raw) == row["sha256"]
        rows_ok &= row_ok
        observed.append({"name": row["name"], "bytes": len(raw),
                         "sha256": sha(raw)})
    actual_names = {path.name for path in package.iterdir()}
    listed_names = {row["name"] for row in observed} | {"SOURCE_MANIFEST.json"}
    root = sha(canonical(observed))
    audit.check("source_manifest_external_pin",
                sha(manifest_raw) == EXPECTED_SOURCE_MANIFEST_SHA256,
                observed_sha256=sha(manifest_raw),
                expected_sha256=EXPECTED_SOURCE_MANIFEST_SHA256)
    audit.check("source_manifest_member_bytes_and_hashes", rows_ok,
                members=len(observed))
    audit.check("source_manifest_sorted_member_order",
                [row["name"] for row in observed] ==
                sorted((row["name"] for row in observed), key=lambda x: x.encode("utf-8")))
    audit.check("source_manifest_root", root == EXPECTED_SOURCE_ROOT_SHA256 ==
                manifest.get("source_root_sha256"), observed_root_sha256=root)
    audit.check("source_manifest_directory_closure", actual_names == listed_names,
                unexpected=sorted(actual_names - listed_names),
                missing=sorted(listed_names - actual_names))
    return {"manifest_sha256": sha(manifest_raw), "source_root_sha256": root,
            "members": len(observed), "directory_closed": actual_names == listed_names}


def repaired_crc_packet(packet: bytes, fields: Sequence[Any] | None = None,
                        payload_mutator: Callable[[bytearray, int, int], None] | None = None,
                        alignment_mutator: Callable[[bytearray, int], None] | None = None) -> bytes:
    raw = bytearray(packet)
    original = list(AUDIT_HEADER.unpack(raw[:AUDIT_HEADER_BYTES]))
    use = original if fields is None else list(fields)
    raw[:AUDIT_HEADER_BYTES] = AUDIT_HEADER.pack(*use)
    logical_bits = int(use[6])
    payload_bytes = (logical_bits + 7) // 8
    crc_offset = AUDIT_HEADER_BYTES + payload_bytes
    if payload_mutator is not None:
        payload_mutator(raw, AUDIT_HEADER_BYTES, payload_bytes)
    raw[crc_offset:crc_offset + 4] = struct.pack(
        "<I", zlib.crc32(raw[:crc_offset]) & 0xFFFFFFFF)
    if alignment_mutator is not None:
        alignment_mutator(raw, crc_offset + 4)
    return bytes(raw)


def independent_exact_oracle(costs: np.ndarray, variables: int,
                             orders: Sequence[int], sc_seed: int,
                             coset_mode: str) -> tuple[int, np.ndarray, float]:
    n = 1 << variables
    dims = [independent_rm_dimension(order, variables) for order in orders]
    total = sum(dims)
    frozen = []
    for level0 in range(6):
        if coset_mode == "zero":
            frozen.append(np.zeros(n, dtype=np.uint8))
        else:
            rng = np.random.default_rng(sc_seed + 1_000_003 * (level0 + 1))
            frozen.append(rng.integers(0, 2, size=n, dtype=np.uint8))
    best_message, best_indices, best = -1, None, float("inf")
    for message in range(1 << total):
        cursor, indices = 0, np.zeros(n, dtype=np.uint8)
        for level0, (order, dim) in enumerate(zip(orders, dims, strict=True)):
            info = np.asarray([(message >> (cursor + bit)) & 1
                               for bit in range(dim)], dtype=np.uint8)
            cursor += dim
            plane = independent_plane(info, variables, order, frozen[level0])
            indices |= plane << np.uint8(level0)
        value = float(np.sum(costs[np.arange(n), indices], dtype=np.float64))
        if value < best:
            best_message, best_indices, best = message, indices.copy(), value
    assert best_indices is not None
    return best_message, best_indices, best


def audit_package(package: Path, auditor_path: Path,
                  cupy_receipt_path: Path | None) -> dict[str, Any]:
    audit = Audit()
    package = package.resolve(strict=True)
    auditor_path = auditor_path.resolve(strict=True)
    source_pin = independently_verify_source(package, audit)

    # Import only after the independent source pin/closure has passed.
    sys.path.insert(0, str(package))
    core = import_module(package / "rm6_core.py", "audit_target_rm6_core")
    # Target modules import rm6_core by its canonical name.
    sys.modules["rm6_core"] = core
    sc = import_module(package / "strata_rm_sc.py", "audit_target_strata_rm_sc")
    sys.modules["strata_rm_sc"] = sc
    packet = import_module(package / "packet_codec.py", "audit_target_packet_codec")
    oracle = import_module(package / "exact_oracle.py", "audit_target_exact_oracle")
    verifier = import_module(package / "verify_source.py", "audit_target_verify_source")

    auditor_raw = auditor_path.read_bytes()
    audit.check("frozen_strata_auditor_external_pin",
                len(auditor_raw) == EXPECTED_AUDITOR_BYTES and
                sha(auditor_raw) == EXPECTED_AUDITOR_SHA256,
                observed_bytes=len(auditor_raw), observed_sha256=sha(auditor_raw))
    frozen = import_module(auditor_path, "audit_frozen_strata_auditor")

    # Manifest verifier hostile clones; the source itself is never changed.
    own = verifier.verify(package)
    audit.check("source_self_verifier_clean_pass", own.get("status") == "PASS" and
                own.get("source_root_sha256") == EXPECTED_SOURCE_ROOT_SHA256)
    with tempfile.TemporaryDirectory(prefix="srm6-audit-") as tmp:
        clone = Path(tmp) / "package"
        shutil.copytree(package, clone)
        (clone / "README.md").write_bytes((clone / "README.md").read_bytes() + b"\n")
        was_rejected, error = rejected(lambda: verifier.verify(clone))
        audit.check("source_self_verifier_rejects_member_mutation", was_rejected,
                    rejection=error)
    with tempfile.TemporaryDirectory(prefix="srm6-audit-") as tmp:
        clone = Path(tmp) / "package"
        shutil.copytree(package, clone)
        (clone / "UNLISTED").write_bytes(b"audit")
        was_rejected, error = rejected(lambda: verifier.verify(clone))
        audit.check("source_self_verifier_rejects_extra_member", was_rejected,
                    rejection=error)

    # Frozen-auditor semantic and transform equivalence.
    rng = np.random.default_rng(0x51A7A)
    transform_match, reverse_match, layers_match = True, True, True
    for variables in range(1, 9):
        n = 1 << variables
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        transform_match &= np.array_equal(core.polar_transform(bits),
                                          frozen.polar_transform(bits))
        transform_match &= np.array_equal(core.polar_transform(bits),
                                          independent_polar(bits))
        reverse_match &= np.array_equal(core.bit_reverse_indices(n),
                                        frozen.bit_reverse_indices(n))
        reverse_match &= np.array_equal(core.bit_reverse_indices(n),
                                        independent_bit_reverse(n))
        layers_match &= np.array_equal(sc.sc_layers(n), frozen.sc_layers(n))
    audit.check("polar_transform_matches_frozen_auditor_and_independent_code",
                transform_match)
    audit.check("bit_reverse_matches_frozen_auditor_and_independent_code", reverse_match)
    audit.check("sc_layers_match_frozen_auditor", layers_match)

    rm_geometry_ok = True
    exact_set_ok = True
    for variables in range(1, 9):
        for order in range(variables + 1):
            expected_dim = independent_rm_dimension(order, variables)
            observed = core.rm_information_positions(variables, order)
            rm_geometry_ok &= core.rm_dimension(order, variables) == expected_dim
            rm_geometry_ok &= observed.size == expected_dim
            exact_set_ok &= np.array_equal(observed,
                                           independent_information_positions(variables, order))
            generator = core.generator_matrix(variables, order)
            expected_weights = np.asarray([1 << int(i).bit_count() for i in observed])
            exact_set_ok &= np.array_equal(generator.sum(axis=1), expected_weights)
    audit.check("rm_dimensions_independent_combinatorial_check", rm_geometry_ok)
    audit.check("rm_phase_orientation_and_generator_row_weights", exact_set_ok,
                selection="popcount(internal_phase)>=m-r")

    # Exhaustive six-plane LSB-first 0..63 ABI and reconstruction alphabet.
    singleton_planes = [np.asarray([(index >> level0) & 1 for index in range(64)],
                                   dtype=np.uint8) for level0 in range(6)]
    assembled = core.assemble_indices(singleton_planes)
    levels = core.reconstruction_levels(0x3C00)
    audit.check("six_plane_lsb_first_exhaustive_0_through_63",
                np.array_equal(assembled, np.arange(64, dtype=np.uint8)))
    audit.check("reconstruction_alphabet_matches_frozen_strata",
                np.array_equal(levels, frozen.ETA * np.arange(-31, 33,
                                                              dtype=np.float64)),
                eta=float(core.ETA), minimum=float(levels[0]), maximum=float(levels[-1]))
    four_rejected, four_error = rejected(lambda: core.assemble_indices(
        [np.zeros(8, dtype=np.uint8) for _ in range(4)]))
    audit.check("four_plane_abstraction_fails_closed", four_rejected,
                rejection=four_error)

    # Plane encoding independently checked for both affine cosets.
    plane_ok = True
    affine_differences = []
    variables, order, seed = 6, 3, 0x1234
    n = 1 << variables
    zero = np.zeros(n, dtype=np.uint8)
    random_external = core.frozen_external_from_seed(n, seed, 1)
    for _ in range(4):
        info = rng.integers(0, 2, size=core.rm_dimension(order, variables),
                            dtype=np.uint8)
        source_zero = core.plane_from_information(info, variables, order, zero)
        source_random = core.plane_from_information(info, variables, order,
                                                     random_external)
        plane_ok &= np.array_equal(source_zero,
                                   independent_plane(info, variables, order, zero))
        plane_ok &= np.array_equal(source_random,
                                   independent_plane(info, variables, order,
                                                     random_external))
        affine_differences.append(source_zero ^ source_random)
    affine_invariant = all(np.array_equal(affine_differences[0], item)
                           for item in affine_differences[1:])
    audit.check("zero_and_random_coset_planes_match_independent_encoder", plane_ok)
    audit.check("random_coset_is_fixed_affine_offset_not_polynomial_evidence",
                affine_invariant and bool(np.any(affine_differences[0])),
                disagreement_fraction=float(np.mean(affine_differences[0])))

    # Compare one full SC level with the frozen independent STRATA decoder.
    variables = 7
    n = 1 << variables
    leaf_lr = np.exp(rng.normal(0.0, 1.0, size=n)).astype(np.float64)
    freeze = core.rm_freeze_flag(variables, 3)
    frozen_external = rng.integers(0, 2, size=n, dtype=np.uint8)
    prescribed_bits = rng.integers(0, 2, size=int(np.count_nonzero(freeze == 0)),
                                   dtype=np.uint8)
    prescribed = sc.PrescribedBits(prescribed_bits)
    target_sc = sc.run_sc_level(leaf_lr, freeze, frozen_external, prescribed)
    target_payload, target_logical = sc.arithmetic_encode_binary(
        target_sc["selected"], target_sc["frequencies"])
    frozen_payload, frozen_logical = frozen.arithmetic_encode_binary(
        target_sc["selected"], target_sc["frequencies"])
    arithmetic = frozen.ArithmeticBinaryDecoder(target_payload, 0, target_logical)
    frozen_output, frozen_freq, frozen_selected = frozen.decode_sc_level(
        leaf_lr, freeze, frozen_external, frozen.bit_reverse_indices(n),
        frozen.sc_layers(n), arithmetic)
    audit.check("arithmetic_encoder_byte_exact_against_frozen_auditor",
                target_logical == frozen_logical and target_payload == frozen_payload,
                logical_bits=target_logical, selected=int(target_sc["selected"].size))
    audit.check("sc_level_output_frequency_and_decisions_match_frozen_auditor",
                np.array_equal(target_sc["output"], frozen_output) and
                np.array_equal(target_sc["frequencies"], frozen_freq) and
                np.array_equal(target_sc["selected"], frozen_selected))

    # Standalone arithmetic roundtrips, including extreme legal frequencies.
    arithmetic_ok = True
    arithmetic_cases = []
    for length in (1, 2, 31, 32, 33, 257, 2049):
        bits = rng.integers(0, 2, size=length, dtype=np.uint8)
        frequencies = rng.integers(1, 65_536, size=length, dtype=np.uint16)
        if length >= 2:
            frequencies[0], frequencies[1] = 1, 65_535
        encoded, logical = sc.arithmetic_encode_binary(bits, frequencies)
        encoded2, logical2 = frozen.arithmetic_encode_binary(bits, frequencies)
        decoder = sc.ArithmeticBinaryDecoder(encoded, logical)
        decoded = np.asarray([decoder(int(freq), phase)
                              for phase, freq in enumerate(frequencies)], dtype=np.uint8)
        passed = np.array_equal(decoded, bits) and encoded == encoded2 and logical == logical2
        arithmetic_ok &= passed
        arithmetic_cases.append({"symbols": length, "logical_bits": logical,
                                 "bytes": len(encoded), "passed": bool(passed)})
    audit.check("canonical_arithmetic_roundtrip_and_frozen_byte_equivalence",
                arithmetic_ok, cases=arithmetic_cases)

    # Bank dimensions are independent of, and must not be confused with, emitted bits.
    bank_rows = []
    banks_ok = True
    previous = None
    for bank_id, orders in sorted(core.ORDER_BANK.items()):
        dimensions = tuple(independent_rm_dimension(order, 12) for order in orders)
        row = core.dimension_ledger(bank_id)
        passed = (tuple(row["dimensions"]) == dimensions and
                  row["information_bits"] == sum(dimensions) and
                  row["packet_bytes"] <= 1280 and
                  row["dimension_screen_not_emitted_arithmetic_bits"] is True)
        if previous is not None:
            passed &= row["information_bits"] <= previous
        previous = row["information_bits"]
        banks_ok &= passed
        bank_rows.append({"bank_id": bank_id, "orders": list(orders),
                          "dimensions": list(dimensions),
                          "information_bits": sum(dimensions),
                          "dimension_packet_bytes": row["packet_bytes"],
                          "dimension_physical_bpw": row["physical_bpw"],
                          "target_eligible": row["dimension_screen_target_rate_eligible"]})
    audit.check("all_bank_dimensions_and_dimension_ledgers", banks_ok,
                banks=bank_rows)

    # Literal packet controls for both charged coset selectors.
    packets: dict[str, bytes] = {}
    packet_rows = []
    packet_controls_ok = True
    for mode in ("zero", "current_random"):
        greedy = sc.replay_six_greedy(0, 96, 0x13579BDF, mode)
        literal, encoded = packet.encode_packet(
            greedy["decisions"], bank_id=0, scale_fp16_bits=0x3C00,
            profile_q=96, coset_mode=mode, sc_seed=0x13579BDF,
            rht_seed=0x0123456789ABCDEF)
        decoded = packet.decode_packet(literal)
        packets[mode] = literal
        passed = (decoded["canonical_reencode_match"] and
                  np.array_equal(encoded["indices"], decoded["indices"]) and
                  decoded["information_bits"] == 9516 and
                  decoded["logical_bits"] ==
                  decoded["ledger"]["emitted_arithmetic_bits"] and
                  len(literal) % 128 == 0 and len(literal) <= 1280)
        packet_controls_ok &= passed
        packet_rows.append({"coset_mode": mode, "bytes": len(literal),
                            "physical_bpw": len(literal) * 8.0 / 4096,
                            "logical_bits": decoded["logical_bits"],
                            "information_bits": decoded["information_bits"],
                            "target_rate_eligible":
                                decoded["ledger"]["actual_target_rate_eligible"],
                            "promotion_status": decoded["ledger"]["promotion_status"],
                            "packet_sha256": sha(literal), "passed": bool(passed)})
    audit.check("literal_packet_roundtrip_both_cosets", packet_controls_ok,
                packets=packet_rows)
    subminimum_ok = all(row["physical_bpw"] < 2.15 and
                        row["target_rate_eligible"] is False and
                        row["promotion_status"] == "MECHANISM_FIXTURE_BELOW_2_15_BPW"
                        for row in packet_rows)
    audit.check("sub_2p15_controls_are_target_ineligible", subminimum_ok,
                packets=packet_rows)

    # Hostile packet mutation with attacker-repaired CRC.
    base = packets["zero"]
    base_fields = list(AUDIT_HEADER.unpack(base[:AUDIT_HEADER_BYTES]))
    mutation_rows = []

    def header_attack(name: str, index: int, value: Any) -> None:
        fields = base_fields.copy()
        fields[index] = value
        crafted = repaired_crc_packet(base, fields)
        denied, error = rejected(lambda: packet.decode_packet(crafted))
        mutation_rows.append({"attack": name, "rejected": denied, "error": error})

    header_attack("plane_count_four", 3, 4)
    header_attack("information_count_mismatch", 7, int(base_fields[7]) - 1)
    header_attack("nonfinite_fp16_scale", 8, 0x7E00)
    header_attack("unknown_coset_selector", 10, 2)
    header_attack("order_bank_mismatch", 11, max(0, int(base_fields[11]) - 1))
    header_attack("reserved_nonzero", 19, 1)

    crc_tamper = bytearray(base)
    crc_tamper[AUDIT_HEADER_BYTES] ^= 0x80
    denied, error = rejected(lambda: packet.decode_packet(bytes(crc_tamper)))
    mutation_rows.append({"attack": "payload_crc_tamper", "rejected": denied,
                          "error": error})

    logical_bits = int(base_fields[6])
    if logical_bits & 7:
        def set_terminal_padding_bit(raw: bytearray, start: int, length: int) -> None:
            raw[start + length - 1] |= 1

        crafted = repaired_crc_packet(
            base, payload_mutator=set_terminal_padding_bit)
        denied, error = rejected(lambda: packet.decode_packet(crafted))
        mutation_rows.append({"attack": "nonzero_terminal_bit_repaired_crc",
                              "rejected": denied, "error": error})
    else:
        mutation_rows.append({"attack": "nonzero_terminal_bit_repaired_crc",
                              "rejected": True,
                              "error": "not applicable: observed stream byte aligned"})

    def set_alignment_byte(raw: bytearray, begin: int) -> None:
        del begin
        raw[-1] = 1

    crafted = repaired_crc_packet(base, alignment_mutator=set_alignment_byte)
    denied, error = rejected(lambda: packet.decode_packet(crafted))
    mutation_rows.append({"attack": "nonzero_alignment_padding", "rejected": denied,
                          "error": error})
    denied, error = rejected(lambda: packet.decode_packet(base[:-128]))
    mutation_rows.append({"attack": "truncate_one_page", "rejected": denied,
                          "error": error})
    denied, error = rejected(lambda: packet.decode_packet(base + bytes(128)))
    mutation_rows.append({"attack": "append_one_page", "rejected": denied,
                          "error": error})
    audit.check("crc_header_terminal_and_alignment_attacks_fail_closed",
                all(row["rejected"] for row in mutation_rows), attacks=mutation_rows)

    # Exact physical cap boundary: 9,888 logical bits fit, 9,889 force 1,408 bytes.
    cap_header = packet._header(0, 9888, 0x3C00, 96, "zero", 0x13579BDF,
                                0x0123456789ABCDEF)
    boundary_packet = packet._build_packet(cap_header, bytes(9888 // 8), 9888)
    over_header = packet._header(0, 9889, 0x3C00, 96, "zero", 0x13579BDF,
                                 0x0123456789ABCDEF)
    over_rejected, over_error = rejected(lambda: packet._build_packet(
        over_header, bytes((9889 + 7) // 8), 9889))
    audit.check("exact_2p5_boundary_and_first_overcap_bit_fail_closed",
                len(boundary_packet) == 1280 and over_rejected,
                maximum_fitting_logical_bits=9888,
                first_overcap_logical_bits=9889, rejection=over_error)

    # Public ledger diagnostic has a bounded status-label defect for impossible packets.
    over_ledger = packet.packet_ledger(0, 9889)
    ledger_mislabels = (over_ledger["actual_passes_2_5_bpw"] is False and
                        over_ledger["actual_physical_bpw"] > 2.5 and
                        over_ledger["promotion_status"] ==
                        "MECHANISM_FIXTURE_BELOW_2_15_BPW")
    audit.check("oversize_packet_ledger_status_defect_reproduced", ledger_mislabels,
                ledger=over_ledger)
    if ledger_mislabels:
        audit.finding(
            "SRM6-AUDIT-001", "medium",
            "packet_ledger labels a >2.5-bpw hypothetical as below 2.15 bpw",
            "FIX_LEDGER_STATUS_BEFORE_ANY_AUTOMATED_PROMOTION; literal packet build/decode still fail closed",
            actual_physical_bpw=over_ledger["actual_physical_bpw"],
            actual_passes_2_5_bpw=over_ledger["actual_passes_2_5_bpw"],
            reported_promotion_status=over_ledger["promotion_status"])

    # Bounded exact oracle versus a separately written exhaustive enumerator.
    oracle_rows = []
    exact_ok = True
    variables, orders, scale_bits = 3, (1, 1, 0, 0, 0, 0), 0x3C00
    coordinate = np.arange(1 << variables)
    target = core.reconstruction_levels(scale_bits)[
        ((11 * coordinate + 3 * (coordinate >> 1)) & 63).astype(np.uint8)
    ] + 0.011 * np.sin(coordinate + 0.25)
    costs = core.exact_distortion_costs(target, scale_bits)
    for mode in ("zero", "current_random"):
        observed = oracle.exact_joint_oracle(costs, variables, orders,
                                             sc_seed=17, coset_mode=mode)
        message, indices, distortion = independent_exact_oracle(
            costs, variables, orders, 17, mode)
        passed = (observed["best_message"] == message and
                  np.array_equal(observed["indices"], indices) and
                  observed["distortion"] == distortion and
                  observed["legal_joint_rm_codeword"] is True)
        exact_ok &= passed
        oracle_rows.append({"coset_mode": mode, "candidate_messages":
                            observed["candidate_messages"], "best_message": message,
                            "distortion": distortion, "passed": bool(passed)})
    audit.check("small_n_exact_oracle_matches_independent_exhaustive_search",
                exact_ok, cases=oracle_rows)
    cap_rejected, cap_error = rejected(lambda: oracle.exact_joint_oracle(
        np.zeros((16, 64), dtype=np.float64), 4, (2, 2, 2, 2, 2, 2),
        sc_seed=1, coset_mode="zero", maximum_information_bits=18))
    audit.check("small_n_oracle_enumeration_cap_fails_closed", cap_rejected,
                rejection=cap_error)

    fractional_accepted = True
    fractional_error = None
    try:
        fractional = oracle.exact_joint_oracle(costs, variables,
                                               (1.5, 1, 0, 0, 0, 0),
                                               sc_seed=17, coset_mode="zero")
        fractional_accepted = fractional["orders"][0] == 1.5
    except Exception as error:
        fractional_accepted = False
        fractional_error = f"{type(error).__name__}: {error}"
    audit.check("fractional_order_input_acceptance_defect_reproduced",
                fractional_accepted, error=fractional_error)
    if fractional_accepted:
        audit.finding(
            "SRM6-AUDIT-002", "low",
            "small-N oracle silently truncates fractional RM orders with int()",
            "TYPE-CHECK ORACLE ORDERS BEFORE USING IT AS A DOMINANT RESULT; production bank packet is unaffected")

    # Global candidate is a row-order proposal only, not a completed packet codec.
    global_exact_ok = True
    for variables in range(1, 10):
        n = 1 << variables
        for order in range(variables + 1):
            k = independent_rm_dimension(order, variables)
            selected = sc.rm_ordered_positions(n, k)
            global_exact_ok &= np.array_equal(np.sort(selected),
                                              independent_information_positions(
                                                  variables, order))
            global_exact_ok &= sc.classify_selected_dimension(
                variables, k)["exact_rm"] is True
    arbitrary = sc.classify_selected_dimension(21, 700_000)
    global_exact_ok &= (arbitrary["exact_rm"] is False and
                        arbitrary["name"] == "RM-ordered truncated polar set")
    audit.check("global_exact_rm_versus_truncated_polar_distinction",
                global_exact_ok, arbitrary_example=arbitrary)

    global_text = (package / "strata_rm_sc.py").read_text(encoding="utf-8")
    packet_text = (package / "packet_codec.py").read_text(encoding="utf-8")
    run_text = (package / "run_gate.py").read_text(encoding="utf-8")
    global_not_implemented = (
        "HOLD_NO_PAYLOAD_ARITHMETIC_LENGTH" in run_text and
        "RM-ordered truncated polar set" in global_text and
        "global_n20_n21" not in packet_text
    )
    audit.check("global_rm_ordered_swap_has_no_physical_packet_implementation",
                global_not_implemented)
    audit.finding(
        "SRM6-AUDIT-003", "blocking",
        "global RM-ordered candidate has no current-K arithmetic payload measurement",
        "HOLD_GLOBAL_CANDIDATE_AND_ALL_QWEN_CLAIMS",
        classification="RM-ordered truncated polar unless K is a complete RM dimension")

    # The local packet stops at indices; outer expert decode/read topology is absent.
    header_fields = ["magic", "version", "log2_n", "planes", "bank_id",
                     "block_values", "logical_bits", "information_bits",
                     "scale_fp16", "profile_q", "coset_id", "six_orders",
                     "sc_seed", "rht_seed", "reserved"]
    absent_outer_fields = ["expert_id", "tensor_role", "matrix_shape",
                           "subblock_ordinal", "subblock_count", "KLT coefficients",
                           "outer source hash", "expert directory", "read schedule"]
    decoder_stops_at_indices = ("inverse_signed_rht" not in packet_text and
                                "inverse_klt" not in packet_text and
                                "indices" in packet_text)
    audit.check("local_header_and_decoder_outer_scope_audited",
                AUDIT_HEADER_BYTES == 40 and decoder_stops_at_indices,
                literal_header_fields=header_fields,
                absent_outer_fields=absent_outer_fields,
                decoder_output_scope="six-plane indices; no inverse RHT/KLT/expert materialization")
    audit.finding(
        "SRM6-AUDIT-004", "blocking",
        "no outer expert container, inverse-transform integration, or routed-read measurement",
        "HOLD_PRODUCTION_PHYSICAL_RATE_AND_READ_AMPLIFICATION",
        absent_outer_fields=absent_outer_fields,
        outer_overhead_reserved_inside_2p5_cap=False,
        local_packet_rht_seed_consumed_by_decoder=False)

    # CuPy smoke: actual execution can pass mechanics while its receipt is underbound.
    cupy_text = (package / "cupy_soft_search_smoke.py").read_text(encoding="utf-8")
    receipt_binding_keys = {
        "source_manifest_sha256": "source_manifest_sha256" in cupy_text,
        "source_root_sha256": "source_root_sha256" in cupy_text,
        "auditor_sha256": "auditor_sha256" in cupy_text,
        "python_executable_sha256": "python_executable_sha256" in cupy_text,
        "driver_version": "driverGetVersion" in cupy_text,
        "device_id": '"device_id"' in cupy_text,
        "receipt_sha256": "receipt_sha256" in cupy_text,
    }
    underbound = not any(receipt_binding_keys.values())
    audit.check("cupy_receipt_binding_scope_defect_reproduced", underbound,
                binding_keys_present=receipt_binding_keys)
    cupy_execution: dict[str, Any]
    if cupy_receipt_path is None:
        cupy_execution = {"executed": False,
                          "status": "NOT_EXECUTED_NO_RECEIPT_SUPPLIED"}
    else:
        cupy_raw = cupy_receipt_path.resolve(strict=True).read_bytes()
        cupy = json.loads(cupy_raw)
        cupy_pass = (
            cupy.get("schema") == "strata-rm6-label-flexible-cupy-smoke-v0" and
            cupy.get("status") ==
            "PASS_LOCAL_GREEDY_MECHANISM_HOLD_PRODUCTION_SEARCH" and
            cupy.get("monotone_nonincreasing") is True and
            cupy.get("cpu_selected_distortion_match") is True and
            cupy.get("qwen_payload_accessed") is False and
            cupy.get("coarse_payload_accessed") is False and
            cupy.get("control_payload_accessed") is False)
        audit.check("source_free_cupy_smoke_observed_mechanical_checks", cupy_pass,
                    receipt_sha256=sha(cupy_raw), device=cupy.get("device"),
                    steps_taken=cupy.get("steps_taken"), packet=cupy.get("packet"))
        cupy_execution = {"executed": True, "receipt_sha256": sha(cupy_raw),
                          "mechanical_checks_passed": bool(cupy_pass),
                          "device": cupy.get("device"),
                          "steps_requested": cupy.get("steps_requested"),
                          "steps_taken": cupy.get("steps_taken"),
                          "packet_fits_2_5_bpw": cupy.get("packet_fits_2_5_bpw"),
                          "packet_target_rate_eligible":
                              cupy.get("packet_target_rate_eligible")}
    audit.finding(
        "SRM6-AUDIT-005", "high",
        "CuPy smoke receipt is not cryptographically bound to frozen source/auditor/runtime",
        "HOLD_CUPY_RESULT_AS_MECHANISM_ONLY; freeze source, interpreter, packages, driver, device and receipt hash before payload",
        binding_keys_present=receipt_binding_keys,
        pass_status_does_not_require_packet_fit=True)

    # No payload path is accepted by this auditor; only source/auditor/receipt opened.
    all_tests_pass = all(row["passed"] for row in audit.tests)
    return {
        "schema": "strata-rm6-label-flexible-gate-v0-independent-source-audit-20260902",
        "status": (
            "PASS_LOCAL_SOURCE_MECHANISM__HOLD_PRODUCTION_GLOBAL_PAYLOAD_AND_READS"
            if all_tests_pass else "AUDIT_TEST_FAILURE"
        ),
        "all_executed_tests_passed": all_tests_pass,
        "source_pin": source_pin,
        "frozen_strata_auditor": {"bytes": len(auditor_raw),
                                    "sha256": sha(auditor_raw)},
        "tests": audit.tests,
        "findings": audit.findings,
        "cupy_execution": cupy_execution,
        "verdicts": {
            "six_plane_0_63_semantics": "PASS",
            "rm_orientation_against_frozen_strata": "PASS",
            "canonical_arithmetic_and_literal_local_packet": "PASS",
            "crc_padding_and_over_2p5_fail_close": "PASS",
            "under_2p15_target_eligibility": "PASS_REJECTED_AS_TARGET",
            "zero_vs_current_random_cosets": "PASS_DISTINCT_CHARGED_AFFINE_CONTROLS",
            "small_n_exact_oracle": "PASS_MECHANISM_ONLY__FIX_FRACTIONAL_ORDER_TYPE_CHECK",
            "cupy_search": ("PASS_SOURCE_FREE_MECHANICS__HOLD_RUNTIME_BINDING"
                            if cupy_execution["executed"] else
                            "NOT_EXECUTED__STATIC_RECEIPT_SCOPE_AUDITED"),
            "global_rm_ordered_swap": "HOLD_NO_PACKET_OR_CURRENT_K_ARITHMETIC_RATE",
            "outer_expert_container_and_inverse_transforms": "HOLD_NOT_IMPLEMENTED",
            "routed_read_amplification": "HOLD_NOT_MEASURED",
            "qwen_payload": "HOLD_NOT_OPENED",
            "coarse_payload": "HOLD_NOT_OPENED",
            "matched_control_payload": "HOLD_NOT_OPENED",
        },
        "claim_boundary": (
            "The local RM frozen-set, six-plane SC, arithmetic packet, and tiny exact "
            "oracle mechanisms are auditable. There is no production encoder, global "
            "current-K packet, outer expert decoder, physical expert ledger, read result, "
            "or source performance result."
        ),
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "control_payload_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--auditor", required=True)
    parser.add_argument("--cupy-receipt")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit_package(Path(args.package), Path(args.auditor),
                           None if args.cupy_receipt is None else
                           Path(args.cupy_receipt))
    result["audit_result_sha256"] = sha(canonical(result))
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"],
                      "all_executed_tests_passed":
                          result["all_executed_tests_passed"],
                      "tests": len(result["tests"]),
                      "findings": len(result["findings"]),
                      "audit_result_sha256": result["audit_result_sha256"],
                      "output": args.output}, sort_keys=True))
    if not result["all_executed_tests_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
