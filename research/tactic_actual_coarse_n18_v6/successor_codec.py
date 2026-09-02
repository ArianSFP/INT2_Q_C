#!/usr/bin/env python3
"""Receipt-hardened v6 facade over the exact authenticated v4 packet codec."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping


class CodecError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CodecError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_rate_record(frame_bytes: int, geometry: Any) -> dict[str, Any]:
    numerator = 8 * frame_bytes
    denominator = int(geometry.values)
    divisor = math.gcd(numerator, denominator)
    reduced_numerator = numerator // divisor
    reduced_denominator = denominator // divisor
    target_identity = numerator * 128 == denominator * 307
    require(target_identity == bool(geometry.target_eligible),
            "target eligibility/exact aggregate-rate identity")
    return {
        "numerator": reduced_numerator,
        "denominator": reduced_denominator,
        "exact": f"{reduced_numerator}/{reduced_denominator}",
        "float": numerator / denominator,
        "equals_307_over_128": target_identity,
    }


@dataclass(frozen=True)
class DecodedTileV6:
    parsed: Any
    canonical_symbols_i32: Any
    reconstruction_f32: Any
    canonical_packet: bytes
    report: Mapping[str, Any]


def encode_tile_v6(
    raw_role_bf16: bytes,
    geometry: Any,
    role_ordinal: int,
    tile_ordinal: int,
    runtime: Any,
) -> tuple[bytes, dict[str, Any]]:
    packet, inherited = runtime.numeric_encoder.encode_tile(
        raw_role_bf16,
        geometry,
        role_ordinal,
        tile_ordinal,
        runtime.encoder_runtime,
    )
    report = dict(inherited)
    checks = (
        "arithmetic_roundtrip_bits_match",
        "causal_decoder_frequencies_match",
        "reconstruction_indices_match",
    )
    if report["status"] == "PASS_FINITE_TILE_WITHIN_RESERVOIR":
        for name in checks:
            require(report.get(name) is True, f"required encoder self-check: {name}")
        report["all_encoder_self_checks_required_and_passed"] = True
        report["encoder_self_checks_applicable"] = True
    else:
        require(report["status"] == "PASS_EXACT_ZERO_TILE", "inherited encoder status")
        require(not any(name in report for name in checks), "zero tile has fake encoder checks")
        report["all_encoder_self_checks_required_and_passed"] = True
        report["encoder_self_checks_applicable"] = False
    require(sha256(packet) == report["packet_sha256"], "encoder packet receipt binding")
    report["status"] = "PASS_V6_FINITE_TILE_ALL_ENCODER_CHECKS"
    report["schema"] = "tactic-actual-coarse-n18-v6-tile-encode-receipt-v1"
    return packet, report


def encode_expert_frame_from_bf16_v6(
    role_bf16: Mapping[str, bytes],
    geometry: Any,
    runtime: Any,
) -> tuple[bytes, dict[str, Any]]:
    packet_module = runtime.packet
    require(set(role_bf16) == set(packet_module.ROLES), "exact role bytes")
    packets: list[bytes] = []
    reports: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for role_ordinal, role in enumerate(packet_module.ROLES):
        raw = role_bf16[role]
        require(type(raw) is bytes, "BF16 role bytes type")
        require(len(raw) == 2 * geometry.role_values, "BF16 role bytes length")
        source_hashes[role] = sha256(raw)
        for tile_ordinal in range(geometry.streams_per_role):
            packet, report = encode_tile_v6(
                raw, geometry, role_ordinal, tile_ordinal, runtime
            )
            packets.append(packet)
            reports.append(report)
    frame = b"".join(packets)
    parsed = packet_module.parse_expert_frame(frame)
    require(parsed.geometry == geometry, "encoded frame geometry")
    require(len(frame) == geometry.frame_bytes, "encoded frame bytes")
    require(
        all(row["all_encoder_self_checks_required_and_passed"] is True for row in reports),
        "aggregate required encoder checks",
    )
    rate = exact_rate_record(len(frame), geometry)
    return frame, {
        "schema": "tactic-actual-coarse-n18-v6-frame-encode-receipt-v1",
        "status": "PASS_V6_FRAME_ALL_ENCODER_CHECKS_AWAITING_INDEPENDENT_DECODE",
        "geometry": {
            "intermediate": geometry.intermediate,
            "hidden": geometry.hidden,
            "weights": geometry.values,
        },
        "target_eligible_exact_307_over_128": geometry.target_eligible,
        "records": geometry.records,
        "frame_bytes": len(frame),
        "physical_bpw": rate["float"],
        "physical_bpw_exact": rate,
        "frame_sha256": sha256(frame),
        "source_bf16_sha256": source_hashes,
        "all_encoder_self_checks_required_and_passed": True,
        "tiles": reports,
        "runtime_closure": runtime.receipt,
    }


def retain_canonical_symbols_i32(value: Any, np: Any) -> Any:
    """Refuse post-hoc widening: the authenticated inverse must already be I32."""
    array = np.asarray(value)
    require(array.dtype.str == "<i4" and array.ndim == 1 and
            array.flags.c_contiguous,
            "canonical inverse symbols must arrive as contiguous little-endian I32")
    retained = array.astype("<i4", copy=False)
    require(retained is array or bool(np.shares_memory(retained, array)),
            "canonical I32 lifetime unexpectedly copied/downcast")
    return retained


def decode_tile_v6(packet: bytes, runtime: Any) -> DecodedTileV6:
    inherited = runtime.independent_decoder.decode_reservoir(packet, runtime.decoder_runtime)
    np = runtime.numpy
    inherited_symbols = np.asarray(inherited.canonical_symbols_i16)
    if inherited.parsed.zero_tile:
        # The inherited v4 zero fast path creates an all-zero I16 array without
        # entering the inverse. It is safe and explicitly not I32 evidence.
        require(inherited_symbols.dtype.str == "<i2" and
                not bool(np.any(inherited_symbols)),
                "zero-tile inherited symbol contract")
        symbols = inherited_symbols.astype("<i4")
        inverse_i32_lifetime_applicable = False
    else:
        # This check occurs before any facade cast. It fails if the installed
        # inverse override did not survive through decode_reservoir.
        symbols = retain_canonical_symbols_i32(inherited_symbols, np)
        inverse_i32_lifetime_applicable = True
    require(symbols.shape == (runtime.packet.N,), "I32 symbol geometry")
    require(inherited.canonical_packet == packet, "literal tile reencode equality")
    report = dict(inherited.report)
    report.pop("canonical_symbols_i16_sha256", None)
    report.update({
        "schema": "tactic-actual-coarse-n18-v6-tile-decode-receipt-v1",
        "status": "PASS_V6_CAUSAL_DECODE_LITERAL_REENCODE_I32",
        "canonical_reencode_matches": True,
        "inverse_transient_dtype": "<i4",
        "inverse_i32_dtype_verified_before_facade_cast":
            inverse_i32_lifetime_applicable,
        "canonical_symbols_i32_sha256": sha256(symbols.tobytes()),
        "canonical_symbol_abs_max": int(np.max(np.abs(symbols.astype(np.int64)))),
    })
    return DecodedTileV6(
        inherited.parsed,
        symbols,
        inherited.reconstruction_f32,
        inherited.canonical_packet,
        report,
    )


def _bf16_f64(np: Any, raw: bytes, expected_values: int) -> Any:
    require(type(raw) is bytes and len(raw) == 2 * expected_values, "canonical BF16 bytes")
    words = np.frombuffer(raw, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    require(values.size == expected_values and bool(np.all(np.isfinite(values))), "finite BF16 source")
    return values


def decode_expert_frame_bytes_v6(
    frame: bytes,
    runtime: Any,
    *,
    source_role_bf16: Mapping[str, bytes] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    packet_module = runtime.packet
    require(type(frame) is bytes and len(frame) >=
            len(packet_module.ROLES) * packet_module.RESERVOIR_BYTES,
            "frame byte lower bound")
    decoded_rows: list[DecodedTileV6] = []
    first = decode_tile_v6(frame[:packet_module.RESERVOIR_BYTES], runtime)
    require((first.parsed.role_ordinal, first.parsed.tile_ordinal) == (0, 0),
            "canonical first decoded record")
    geometry = first.parsed.geometry
    require(len(frame) == geometry.frame_bytes, "frame byte geometry")
    decoded_rows.append(first)
    for ordinal in range(1, geometry.records):
        begin = ordinal * packet_module.RESERVOIR_BYTES
        packet = frame[begin : begin + packet_module.RESERVOIR_BYTES]
        row = decode_tile_v6(packet, runtime)
        expected_role, expected_tile = divmod(ordinal, geometry.streams_per_role)
        require(
            row.parsed.geometry == geometry and
            (row.parsed.role_ordinal, row.parsed.tile_ordinal)
            == (expected_role, expected_tile),
            "canonical decoded frame geometry/role/tile order",
        )
        decoded_rows.append(row)

    # R1: this is literal equality, not truthiness of nonempty byte strings.
    aggregate_reencode = b"".join(row.canonical_packet for row in decoded_rows)
    require(aggregate_reencode == frame, "literal aggregate frame reencode equality")
    reconstructions: dict[str, Any] = {}
    symbol_hashes: dict[str, str] = {}
    for role_ordinal, role in enumerate(packet_module.ROLES):
        rows = [row for row in decoded_rows if row.parsed.role_ordinal == role_ordinal]
        reconstruction = runtime.numpy.concatenate([row.reconstruction_f32 for row in rows])
        require(reconstruction.size == geometry.role_values, "role reconstruction geometry")
        reconstructions[role] = reconstruction
        symbol_hashes[role] = sha256(
            b"".join(row.canonical_symbols_i32.astype("<i4", copy=False).tobytes() for row in rows)
        )

    score = None
    residuals = None
    if source_role_bf16 is not None:
        require(set(source_role_bf16) == set(packet_module.ROLES), "exact scorer role bytes")
        residuals = {}
        rows = []
        total_sse = 0.0
        total_energy = 0.0
        for role in packet_module.ROLES:
            source = _bf16_f64(runtime.numpy, source_role_bf16[role], geometry.role_values)
            reconstruction = reconstructions[role].astype(runtime.numpy.float64)
            residual = source - reconstruction
            sse = float(runtime.numpy.dot(residual, residual))
            energy = float(runtime.numpy.dot(source, source))
            require(math.isfinite(sse) and math.isfinite(energy) and
                    sse >= 0.0 and energy >= 0.0, "score finite nonnegative")
            if energy == 0.0:
                require(sse == 0.0,
                        "zero-energy source must reconstruct exactly")
                relative_mse = None
            else:
                relative_mse = sse / energy
            residuals[role] = residual
            total_sse += sse
            total_energy += energy
            rows.append({
                "role": role,
                "source_bf16_sha256": sha256(source_role_bf16[role]),
                "reconstruction_f32_sha256": sha256(reconstructions[role].astype("<f4", copy=False).tobytes()),
                "residual_f64_sha256": sha256(residual.astype("<f8", copy=False).tobytes()),
                "sse_fp64": sse,
                "source_energy_fp64": energy,
                "relative_mse": relative_mse,
                "zero_source_energy": energy == 0.0,
            })
        score = {
            "domain": "original canonical BF16 Gate/Up/DownT source coordinates",
            "roles": rows,
            "pooled_sse_fp64": total_sse,
            "pooled_source_energy_fp64": total_energy,
            "pooled_relative_mse": (
                total_sse / total_energy if total_energy > 0.0 else None),
            "zero_pooled_source_energy": total_energy == 0.0,
            "scorer_uses_exact_encoder_input_bytes": True,
        }

    traffic = frame_ledger_v6(
        geometry, external_compressed_read_passes=0,
        external_read_mode="prebuffered_encoder_output",
    )
    rate = exact_rate_record(len(frame), geometry)
    receipt = {
        "schema": "tactic-actual-coarse-n18-v6-frame-decode-receipt-v1",
        "status": "PASS_V6_PREBUFFERED_LITERAL_AGGREGATE_REENCODE_I32",
        "geometry": {
            "intermediate": geometry.intermediate,
            "hidden": geometry.hidden,
            "weights": geometry.values,
        },
        "target_eligible_exact_307_over_128": geometry.target_eligible,
        "frame_bytes": len(frame),
        "physical_bpw": rate["float"],
        "physical_bpw_exact": rate,
        "frame_sha256": sha256(frame),
        "aggregate_reencoded_frame_sha256": sha256(aggregate_reencode),
        "literal_aggregate_reencode_matches": True,
        "external_compressed_read_ledger": traffic["external_compressed_read"],
        "host_memory_parse_and_integrity_ledger":
            traffic["host_memory_parse_and_integrity"],
        "scratch_lower_bound_ledger": traffic["scratch_lower_bound"],
        "accelerator_hbm_ledger": traffic["accelerator_hbm"],
        "decoded_coarse_state_buffered": True,
        "inverse_transient_dtype": "<i4",
        "canonical_symbols_i32_sha256": symbol_hashes,
        "records": [row.report for row in decoded_rows],
        "original_domain_score": score,
        "runtime_closure": runtime.receipt,
    }
    return reconstructions, receipt, residuals


def frame_ledger_v6(
    geometry: Any,
    *,
    start_offset_mod_page: int = 0,
    external_compressed_read_passes: int = 0,
    external_read_mode: str = "prebuffered_encoder_output",
) -> dict[str, Any]:
    require(type(start_offset_mod_page) is int and 0 <= start_offset_mod_page < 4096, "page offset")
    require(type(external_compressed_read_passes) is int and
            0 <= external_compressed_read_passes <= 16, "external pass count")
    require(external_read_mode in {
        "prebuffered_encoder_output", "one_pass_external_file",
        "modeled_external_file_reread",
    }, "external read mode")
    require((external_compressed_read_passes == 0) ==
            (external_read_mode == "prebuffered_encoder_output"),
            "external mode/pass consistency")
    require(external_read_mode != "one_pass_external_file" or
            external_compressed_read_passes == 1,
            "one-pass external mode consistency")
    require(external_read_mode != "modeled_external_file_reread" or
            external_compressed_read_passes >= 2,
            "reread mode requires at least two passes")
    frame_bytes = geometry.frame_bytes
    unique_pages = (start_offset_mod_page + frame_bytes + 4095) // 4096
    first_pass = frame_bytes if external_compressed_read_passes else 0
    total_read = external_compressed_read_passes * frame_bytes
    reread = max(0, external_compressed_read_passes - 1) * frame_bytes
    require(total_read == first_pass + reread, "external read-byte identity")
    # Each record retains one complete N-vector, including an implicit tail.
    canonical_symbol_bytes = geometry.records * (1 << 18) * 4
    reconstruction_bytes = geometry.values * 4
    return {
        "schema": "tactic-actual-coarse-n18-v6-frame-ledger-v1",
        "frame_bytes": frame_bytes,
        "external_compressed_read": {
            "mode": external_read_mode,
            "passes": external_compressed_read_passes,
            "first_pass_bytes": first_pass,
            "total_read_bytes": total_read,
            "reread_bytes": reread,
            "total_read_amplification": total_read / frame_bytes,
            "reread_amplification": reread / frame_bytes,
            "one_external_pass": external_compressed_read_passes == 1 and
                                 reread == 0,
            "start_offset_mod_page": start_offset_mod_page,
            "unique_page_bytes_if_externally_read":
                unique_pages * 4096 if external_compressed_read_passes else 0,
        },
        "host_memory_parse_and_integrity": {
            "causal_packet_decode_input_frame_passes": 1,
            "causal_packet_decode_input_frame_bytes": frame_bytes,
            "aggregate_equality_input_frame_passes": 1,
            "aggregate_equality_input_frame_bytes": frame_bytes,
            "frame_sha256_input_frame_passes": 1,
            "frame_sha256_input_frame_bytes": frame_bytes,
            "minimum_input_frame_full_scan_equivalents": 3,
            "minimum_input_frame_bytes_touched": 3 * frame_bytes,
            "arithmetic_and_array_internal_traffic_measured": False,
            "claim_boundary": "host lower bound only; not CPU cache, DRAM, accelerator, or HBM traffic",
        },
        "scratch_lower_bound": {
            "canonical_packet_buffers_bytes": frame_bytes,
            "aggregate_reencode_buffer_bytes": frame_bytes,
            "canonical_symbols_i32_bytes": canonical_symbol_bytes,
            "reconstruction_f32_bytes": reconstruction_bytes,
            "minimum_numeric_and_packet_scratch_bytes":
                2 * frame_bytes + canonical_symbol_bytes + reconstruction_bytes,
            "python_object_and_decoder_internal_scratch_measured": False,
        },
        "accelerator_hbm": {
            "measured": False,
            "read_bytes": None,
            "read_amplification": None,
            "below_2x_claim_authority": False,
        },
        "target_eligible_exact_307_over_128": geometry.target_eligible,
    }
