#!/usr/bin/env python3
"""Hostile, payload-free audit of MOSAIC secondary-oracle source v0.

This program deliberately accepts only the frozen source package.  It has no
Qwen, coarse-artifact, result, or matched-control path argument.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import stat
import struct
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence


UPSTREAM_MANIFEST_SHA256 = "4259e8e8dc87b4c25301ca89ade7dbd63c1e0c9e3415fdaa4d7881d7d10ccc06"
UPSTREAM_ROOT_SHA256 = "60bf8cb7575c165c1e8e648360b9d81f39c092070a9489684904bcf06d0bd820"
UPSTREAM_STATUS = "SEALED_SOURCE_ONLY_NO_QWEN_OR_COARSE_PAYLOAD_AUTHORITY"
UPSTREAM_EXPECTED = {
    "README.md",
    "cupy_backend.py",
    "design_lock.json",
    "gate_contract.py",
    "gf2_recurrence.py",
    "residual_oracles.py",
    "run_source_free_cupy_smoke.py",
    "run_source_free_fixture.py",
    "test_source_only.py",
    "verify_source.py",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"{label} duplicate key")
            output[key] = value
        return output

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AuditError(f"{label} nonfinite {token}")
        ),
    )
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, maximum: int = 4 * (1 << 20)) -> bytes:
    path = path.resolve(strict=True)
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and 0 < before.st_size <= maximum,
            f"regular single-link file {path.name}",
        )
        output = bytearray()
        while len(output) < before.st_size:
            chunk = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(chunk), f"short read {path.name}")
            output.extend(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
            f"identity drift {path.name}",
        )
        return bytes(output)
    finally:
        os.close(descriptor)


def authenticate_upstream(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    resolved = root.resolve(strict=True)
    require(resolved.is_dir() and not resolved.is_symlink(), "real upstream directory")
    entries = list(os.scandir(resolved))
    require(
        all(entry.is_file(follow_symlinks=False) and not entry.is_symlink() for entry in entries),
        "upstream regular files only",
    )
    require(
        {entry.name for entry in entries} == UPSTREAM_EXPECTED | {"SOURCE_MANIFEST.json"},
        "upstream exact member set",
    )
    manifest_payload = read_regular(resolved / "SOURCE_MANIFEST.json")
    require(digest(manifest_payload) == UPSTREAM_MANIFEST_SHA256, "upstream manifest pin")
    manifest = strict_json(manifest_payload, "upstream manifest")
    require(
        manifest.get("status") == UPSTREAM_STATUS
        and manifest.get("source_root_sha256") == UPSTREAM_ROOT_SHA256,
        "upstream identity",
    )
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == len(UPSTREAM_EXPECTED), "upstream rows")
    payloads: dict[str, bytes] = {}
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "upstream row")
        name = row["name"]
        require(name in UPSTREAM_EXPECTED, "upstream member name")
        payload = read_regular(resolved / name)
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"], f"upstream {name}")
        payloads[name] = payload
        observed.append({"name": name, "bytes": len(payload), "sha256": digest(payload)})
    require(digest(canonical_json(observed)) == UPSTREAM_ROOT_SHA256, "upstream root recomputation")
    access = manifest.get("access_attestation", {})
    require(
        access.get("qwen_model_checkpoint_payload_accessed") is False
        and access.get("coarse_result_or_COARSE_bin_accessed") is False
        and access.get("matched_control_payload_accessed") is False
        and access.get("production_payload_adapter_present") is False,
        "upstream source-only boundary",
    )
    return payloads, manifest


def load_authenticated(name: str, payload: bytes) -> types.ModuleType:
    require(name not in sys.modules, "module collision")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest(payload)}>"
    module.__package__ = ""
    module.__authenticated_sha256__ = digest(payload)
    sys.modules[name] = module
    exec(compile(payload, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def independent_recurrence_holds(sequence: Sequence[int], order: int, mask: int) -> bool:
    if order == 0:
        return all(value == 0 for value in sequence)
    for index in range(order, len(sequence)):
        predicted = 0
        for lag in range(1, order + 1):
            predicted ^= ((mask >> (lag - 1)) & 1) & int(sequence[index - lag])
        if predicted != int(sequence[index]):
            return False
    return True


def independent_minimal_complexity(sequence: Sequence[int]) -> int:
    for order in range(len(sequence) + 1):
        if any(independent_recurrence_holds(sequence, order, mask) for mask in range(1 << order)):
            return order
    raise AuditError("finite sequence recurrence existence")


def pack_msb(bits: Iterable[int]) -> bytes:
    values = tuple(int(value) for value in bits)
    require(all(value in (0, 1) for value in values), "pack bits")
    output = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        output[index >> 3] |= value << (7 - (index & 7))
    return bytes(output)


def make_block_packet(recurrence: Any, length: int, plans: Sequence[tuple[int, int, Sequence[int]]]) -> bytes:
    require(len(plans) == 2, "two plans")
    directory = bytearray()
    bits = []
    for mode, complexity, payload in plans:
        directory.extend(recurrence.PLANE.pack(mode, complexity, 0))
        bits.extend(int(value) for value in payload)
    body = recurrence.HEADER.pack(recurrence.MAGIC, length, 2, recurrence.VERSION)
    body += bytes(directory) + pack_msb(bits)
    return body + recurrence.CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def reseal_block(recurrence: Any, packet: bytes, mutate) -> bytes:
    body = bytearray(packet[:-recurrence.CRC.size])
    mutate(body)
    sealed = bytes(body)
    return sealed + recurrence.CRC.pack(zlib.crc32(sealed) & 0xFFFFFFFF)


def expect_recurrence_error(recurrence: Any, operation, needle: str | None = None) -> str:
    try:
        operation()
    except recurrence.RecurrenceError as error:
        message = str(error)
        if needle is not None:
            require(needle in message, f"expected {needle!r}, got {message!r}")
        return message
    raise AuditError("hostile mutation accepted")


def recurrence_audit(recurrence: Any, contract: Any) -> dict[str, Any]:
    exhaustive_sequences = 0
    for length in range(1, 11):
        for word in range(1 << length):
            sequence = tuple((word >> (length - 1 - index)) & 1 for index in range(length))
            expected = independent_minimal_complexity(sequence)
            observed, connection = recurrence.berlekamp_massey_gf2(sequence)
            require(observed == expected, f"BM minimality n={length} word={word}")
            coefficients = connection >> 1
            require(independent_recurrence_holds(sequence, observed, coefficients), "BM recurrence")
            require(
                recurrence.generate_lfsr(sequence[:observed], connection, length) == sequence,
                "BM replay",
            )
            exhaustive_sequences += 1

    length = 31
    first = recurrence.generate_lfsr((1, 0, 1), 1 | (1 << 1) | (1 << 3), length)
    second = recurrence.generate_lfsr((1, 1, 0, 1), 1 | (1 << 1) | (1 << 4), length)
    labels = recurrence.labels_from_gray(first, second)
    canonical, row = recurrence.encode_block(labels)
    require(recurrence.decode_block(canonical) == labels, "canonical block replay")

    raw_alias = make_block_packet(
        recurrence,
        length,
        ((recurrence.MODE_RAW, 0, first), (recurrence.MODE_RAW, 0, second)),
    )
    raw_alias_error = expect_recurrence_error(
        recurrence, lambda: recurrence.decode_block(raw_alias), "canonical encoding"
    )
    nonminimal_alias = make_block_packet(
        recurrence,
        length,
        (
            (recurrence.MODE_LFSR, length, (0,) * length + first),
            (recurrence.MODE_LFSR, length, (0,) * length + second),
        ),
    )
    nonminimal_error = expect_recurrence_error(
        recurrence, lambda: recurrence.decode_block(nonminimal_alias), "canonical encoding"
    )
    reserved = reseal_block(
        recurrence,
        canonical,
        lambda body: body.__setitem__(recurrence.HEADER.size + recurrence.PLANE.size - 1, 1),
    )
    reserved_error = expect_recurrence_error(
        recurrence, lambda: recurrence.decode_block(reserved), "plane record"
    )

    padding_labels = None
    padding_packet = None
    padding_bits = None
    for candidate in itertools.product(range(4), repeat=5):
        packet, _ = recurrence.encode_block(candidate)
        body = packet[:-recurrence.CRC.size]
        cursor = recurrence.HEADER.size
        total_bits = 0
        for _ in range(2):
            mode, complexity, _reserved = recurrence.PLANE.unpack_from(body, cursor)
            cursor += recurrence.PLANE.size
            total_bits += 5 if mode == recurrence.MODE_RAW else 2 * complexity
        if total_bits & 7:
            padding_labels, padding_packet, padding_bits = candidate, packet, total_bits
            break
    require(padding_packet is not None and padding_labels is not None and padding_bits is not None, "padding fixture")
    padded = reseal_block(
        recurrence,
        padding_packet,
        lambda body: body.__setitem__(len(body) - 1, body[-1] | 1),
    )
    padding_error = expect_recurrence_error(
        recurrence, lambda: recurrence.decode_block(padded), "terminal padding"
    )

    blocks = (labels, labels[::-1], labels)
    valid_scales = (b"\x00\x3c",) * len(blocks)
    components = tuple(
        recurrence.encode_component(role, blocks, valid_scales)
        for role in ("gate", "up", "down_transposed")
    )
    weights = sum(len(block) for block in blocks) * 3
    expert = recurrence.encode_expert(components, weights=weights)
    decoded = recurrence.decode_expert(expert)
    derived_weights = sum(
        len(block)
        for component in decoded["components"]
        for block in component["label_blocks"]
    )
    require(decoded["weights"] == derived_weights == weights, "decoded expert weight derivation")
    independent_rate = 8.0 * len(expert) / derived_weights
    require(decoded["physical_rate_bpw"] == independent_rate, "decoded physical rate")
    ledger = contract.physical_expert_ledger(
        weights=derived_weights,
        role_component_bytes=tuple(len(component) for component in components),
    )
    require(ledger["physical_bytes"] == len(expert), "literal packet versus abstract ledger")

    # Independently exercise the declared finite-rate geometry at a size where
    # page alignment does not dominate.  This is synthetic mechanism evidence,
    # not model evidence.
    fixture_length = 256
    fixture_first = recurrence.generate_lfsr(
        (1, 0, 0, 1, 1), 1 | (1 << 1) | (1 << 5), fixture_length
    )
    fixture_second = recurrence.generate_lfsr(
        (1, 1, 0, 0, 1, 0, 1), 1 | (1 << 2) | (1 << 7), fixture_length
    )
    fixture_labels = recurrence.labels_from_gray(fixture_first, fixture_second)
    fixture_blocks = tuple(
        fixture_labels[ordinal:] + fixture_labels[:ordinal]
        for ordinal in range(fixture_length)
    )
    fixture_scales = (b"\x00\x3c",) * len(fixture_blocks)
    fixture_components = tuple(
        recurrence.encode_component(role, fixture_blocks, fixture_scales)
        for role in ("gate", "up", "down_transposed")
    )
    fixture_weights = 3 * fixture_length * fixture_length
    fixture_expert = recurrence.encode_expert(fixture_components, weights=fixture_weights)
    fixture_decoded = recurrence.decode_expert(fixture_expert)
    fixture_ledger = contract.physical_expert_ledger(
        weights=fixture_weights,
        role_component_bytes=tuple(len(component) for component in fixture_components),
    )
    require(
        len(fixture_expert) == fixture_ledger["physical_bytes"] == 61_440
        and fixture_decoded["physical_rate_bpw"] == 2.5
        and fixture_ledger["passes_rate_interval"] is True,
        "large synthetic finite-rate geometry",
    )

    # The serializer accepts every two-byte scale pattern.  These are NaN bit
    # patterns under two common 16-bit floating encodings, demonstrating that
    # finite/positive scale semantics are an adapter responsibility.
    half_nan = b"\x00\x7e"
    bfloat_nan = b"\xc0\x7f"
    accepted_scale_patterns = []
    for value in (half_nan, bfloat_nan):
        packet = recurrence.encode_component("gate", (labels,), (value,))
        require(recurrence.decode_component(packet)["scale_f16le"] == (value,), "opaque scale replay")
        accepted_scale_patterns.append(value.hex())

    wrong_weight_fields = list(recurrence.EXPERT_HEADER.unpack_from(expert, 0))
    wrong_weight_fields[3] += 1
    body = expert[recurrence.EXPERT_HEADER.size:]
    wrong_weight_fields[9] = 0
    zero_header = recurrence.EXPERT_HEADER.pack(*wrong_weight_fields)
    wrong_weight_fields[9] = zlib.crc32(zero_header + body) & 0xFFFFFFFF
    wrong_weights = recurrence.EXPERT_HEADER.pack(*wrong_weight_fields) + body
    wrong_weight_error = expect_recurrence_error(
        recurrence, lambda: recurrence.decode_expert(wrong_weights), None
    )

    return {
        "exhaustive_binary_sequences_n1_through_n10": exhaustive_sequences,
        "bm_minimality_and_exact_replay": True,
        "canonical_packet_sha256": digest(canonical),
        "canonical_packet_physical_bytes": len(canonical),
        "canonical_plane_complexities": [item["linear_complexity"] for item in row["planes"]],
        "raw_alias_rejected": raw_alias_error,
        "nonminimal_lfsr_alias_rejected": nonminimal_error,
        "resealed_reserved_field_rejected": reserved_error,
        "resealed_nonzero_terminal_padding_rejected": padding_error,
        "resealed_wrong_expert_weight_count_rejected": wrong_weight_error,
        "decoded_weight_count_derived_from_literal_blocks": derived_weights,
        "independent_physical_rate_bpw": independent_rate,
        "abstract_ledger_matches_literal_packet_for_zero_shared_model": True,
        "large_synthetic_expert_weights": fixture_weights,
        "large_synthetic_expert_physical_bytes": len(fixture_expert),
        "large_synthetic_expert_physical_rate_bpw": fixture_decoded["physical_rate_bpw"],
        "large_synthetic_expert_sha256": digest(fixture_expert),
        "large_synthetic_expert_is_model_evidence": False,
        "opaque_two_byte_scale_patterns_accepted": accepted_scale_patterns,
        "scale_semantics_validated_by_serializer": False,
    }


def gate_and_traffic_audit(contract: Any) -> dict[str, Any]:
    forged_ledger = {
        "physical_rate_bpw": {"float": 2.15},
        "passes_rate_interval": True,
        "passes_strict_cold_read_below_2x": True,
    }
    forged = contract.recurrence_codec_gate(
        relative_mse=0.01,
        ledger=forged_ledger,
        literal4_physical_rate_bpw=3.0,
        matched_control_saving_bpw=0.0,
    )
    require(forged["status"] == "ELIGIBLE_FOR_PORTABILITY_AND_LITERAL_NESTING", "gate forgery witness")

    residual_forged = contract.residual_source_gate(
        input_sse=100.0,
        source_energy=100.0,
        source_remaining_sse=1.0,
        descriptor_bits_per_block=0,
        controls={"permutation": 99.0, "gaussian": 99.0},
    )
    require(residual_forged["status"] == "ELIGIBLE_FOR_ONE_NESTED_FINITE_BUILD", "residual forgery witness")

    ledgers = [
        contract.physical_expert_ledger(weights=4096 * factor, role_component_bytes=(64, 128, 192))
        for factor in (1, 17, 257)
    ]
    require(
        all(
            row["cold_storage_bytes"] == row["physical_bytes"]
            and row["cold_read_amplification"] == 1.0
            and row["external_storage_reads"] == 1
            and row["external_storage_refetches"] == 0
            for row in ledgers
        ),
        "cold-read identity witness",
    )
    shared = contract.physical_expert_ledger(
        weights=4096 * 257,
        role_component_bytes=(64, 128, 192),
        shared_model_bytes=12345,
    )
    require(shared["shared_model_bytes"] == 12345, "abstract shared bytes")
    return {
        "recurrence_gate_accepts_caller_supplied_ledger_and_mse": True,
        "forged_recurrence_gate_status": forged["status"],
        "residual_gate_accepts_caller_supplied_sse_and_energy": True,
        "forged_residual_gate_status": residual_forged["status"],
        "gate_recomputes_packet_bytes": False,
        "gate_recomputes_weight_count_from_packet": False,
        "gate_recomputes_source_sse_from_reconstruction": False,
        "gate_recomputes_matched_control": False,
        "physical_ledger_is_an_abstract_constructor_not_packet_parser": True,
        "shared_model_bytes_have_no_matching_field_in_LRE0_v1": shared["shared_model_bytes"],
        "cold_read_amplification_is_physical_bytes_divided_by_itself": True,
        "cold_read_claim_is_observed_runtime_IO": False,
        "cold_read_rows_tested": len(ledgers),
    }


def inverse_matrix_gain_independent(coefficients: Sequence[float], length: int) -> float:
    # Construct each column of the finite lower-triangular inverse directly.
    squared = 0.0
    order = len(coefficients)
    for source in range(length):
        column = [0.0] * length
        column[source] = 1.0
        for index in range(source + 1, length):
            value = 0.0
            for lag in range(1, min(order, index - source) + 1):
                value -= float(coefficients[lag - 1]) * column[index - lag]
            column[index] = value
        squared += sum(value * value for value in column)
    return squared / length


def residual_oracle_audit(oracles: Any, contract: Any) -> dict[str, Any]:
    import numpy as np

    basis = oracles.build_ramanujan_basis(
        np,
        length=256,
        periods=contract.NON_DYADIC_PERIODS,
        maximum_columns=64,
    )
    q = np.asarray(basis["basis"], dtype=np.float64)
    orthogonality = float(np.max(np.abs(q.T @ q - np.eye(q.shape[1]))))
    require(orthogonality <= 5e-9, "independent Ramanujan orthogonality")
    rng = np.random.default_rng(20260902)
    residual = rng.standard_normal((7, 256)).astype(np.float64)
    source_energy = float(np.sum(residual * residual, dtype=np.float64)) / 0.04
    ramanujan = oracles.ramanujan_panel_metrics(
        np, residual, basis, source_energy=source_energy, fine_bits_per_block=384
    )
    require(
        ramanujan["fixed_fp16_descriptor_bits_per_block"] == 384
        and ramanujan["source_selected_literal_bits_per_block"] <= 384,
        "Ramanujan per-block budgets",
    )
    require(ramanujan["ideal_waterfill_has_finite_backend"] is False, "waterfill scope")

    coefficients = (0.5, -0.25, 0.125)
    observed_gain = oracles.inverse_noise_gain(coefficients, 64)
    expected_gain = inverse_matrix_gain_independent(coefficients, 64)
    require(abs(observed_gain - expected_gain) <= 2e-13 * max(1.0, expected_gain), "inverse trace identity")

    ar = oracles.ar_hankel_panel_metrics(
        np,
        residual,
        source_energy=source_energy,
        orders=(1, 2, 4, 8, 12),
        fine_bits_per_block=384,
    )
    for row in ar["orders"]:
        require(
            row["descriptor_bits_per_block"]
            == row["order_selector_bits_per_block"] + row["coefficient_bits_per_block"],
            "AR descriptor identity",
        )
        require(
            row["descriptor_bits_per_block"] + row["innovation_bits_per_block"] == 384,
            "AR bit conservation",
        )
        require(row["finite_innovation_codec_executed"] is False, "AR finite scope")

    cpu_basis_sha = digest(q.astype("<f8", copy=False).tobytes(order="C"))
    cpu_gaussian = oracles.moment_matched_gaussian_blocks(np, residual, 10619863)
    cpu_gaussian_sha = digest(np.asarray(cpu_gaussian).astype("<f8", copy=False).tobytes(order="C"))
    return {
        "numpy_version": np.__version__,
        "ramanujan_basis_shape": list(q.shape),
        "ramanujan_orthogonality_max_abs_error": orthogonality,
        "ramanujan_basis_f64_sha256_numpy": cpu_basis_sha,
        "fixed_fp16_bits_per_block": ramanujan["fixed_fp16_descriptor_bits_per_block"],
        "source_selected_literal_bits_per_block": ramanujan["source_selected_literal_bits_per_block"],
        "literal_total_bits_for_seven_blocks": 7 * ramanujan["source_selected_literal_bits_per_block"],
        "waterfill_is_dominant_oracle_without_finite_backend": True,
        "inverse_noise_gain_observed": observed_gain,
        "inverse_noise_gain_independent_trace": expected_gain,
        "inverse_noise_gain_trace_identity": True,
        "ar_descriptor_and_innovation_bits_conserved": True,
        "ar_is_ideal_iid_gaussian_innovation_diagnostic_only": True,
        "cpu_gaussian_control_f64_sha256": cpu_gaussian_sha,
        "basis_has_bit_exact_cross_backend_specification": False,
        "gaussian_rng_has_bit_exact_cross_backend_specification": False,
    }


def source_binding_audit(payloads: dict[str, bytes]) -> dict[str, Any]:
    design = strict_json(payloads["design_lock.json"], "design")
    recurrence_text = payloads["gf2_recurrence.py"].decode("utf-8")
    require("mapping = ((0, 0), (0, 1), (1, 1), (1, 0))" in recurrence_text, "four-label mapping")
    require(design["source_access"]["production_payload_adapter_present"] is False, "adapter absent")
    return {
        "packet_alphabet": "exactly four labels mapped to two Gray bitplanes",
        "current_STRATA_reconstruction_alphabet": "64 indices assembled from six complete level-major polar passes",
        "direct_alias_is_valid": False,
        "required_adapter_paths": [
            "new direct four-level codec: derive legal reconstruction levels/scales, jointly select labels, emit LRC0/LRE0, independently decode, and score original-source MSE",
            "current STRATA recoding: generalize from two Gray planes to six reconstructed index planes or model selected SC events while preserving all six polar levels and canonical replay",
        ],
        "required_scale_binding": "freeze exact FP16/BF16 interpretation, byte order, finite-positive legality, block ownership, and current STRATA/POLARIS scale provenance",
        "required_result_scorer": "parse literal packet, derive labels/weights/rate, reconstruct through the frozen transform, and recompute FP64 source SSE and all controls",
        "required_traffic_evidence": "instrument or independently derive routed file/page reads against a frozen denominator; do not infer runtime IO from physical_bytes/physical_bytes",
        "qwen_pilot_would_not_establish_universal_swiglu_moe_claim": True,
    }


def run(upstream: Path) -> dict[str, Any]:
    payloads, _manifest = authenticate_upstream(upstream)
    recurrence = load_authenticated("mosaic_secondary_audit_recurrence", payloads["gf2_recurrence.py"])
    contract = load_authenticated("mosaic_secondary_audit_contract", payloads["gate_contract.py"])
    oracles = load_authenticated("mosaic_secondary_audit_oracles", payloads["residual_oracles.py"])
    recurrence_result = recurrence_audit(recurrence, contract)
    gate_result = gate_and_traffic_audit(contract)
    oracle_result = residual_oracle_audit(oracles, contract)
    binding_result = source_binding_audit(payloads)
    return {
        "schema": "mosaic-secondary-oracles-independent-source-audit-v1",
        "status": "MECHANISMS_VALID__HOLD_PRODUCTION_ADAPTER_SCORER_BACKEND_AND_IO_BINDING",
        "upstream_manifest_sha256": UPSTREAM_MANIFEST_SHA256,
        "upstream_source_root_sha256": UPSTREAM_ROOT_SHA256,
        "recurrence": recurrence_result,
        "gate_and_traffic": gate_result,
        "residual_oracles": oracle_result,
        "production_binding": binding_result,
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "network_accessed_by_this_program": False,
        "production_launch_authorized": False,
        "negative_qwen_evidence": False,
        "claim_boundary": "source mechanics only; no Qwen result, finite target result, runtime read result, or universal SwiGLU-MoE result",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--upstream-source", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args().upstream_source), sort_keys=True, separators=(",", ":"), allow_nan=False))
