#!/usr/bin/env python3
"""Independent finite decoder; does not import or call the encoder."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping


class DecodeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _schedule(cp: Any, symbols: Any, role_ordinal: int, spec: Any) -> list[Any]:
    symbol = cp.asarray(symbols, dtype=cp.int64).reshape(-1, spec.BLOCK_VALUES)
    table = cp.asarray(
        bytearray(spec.universal_selector_table()),
        dtype=cp.uint8).reshape(spec.STAGES, 256)
    mean_abs = cp.sum(cp.abs(symbol), axis=1, dtype=cp.int64) // cp.int64(
        spec.BLOCK_VALUES)
    shadow = symbol
    schedules: list[Any] = []
    for stage in range(spec.STAGES):
        stride = 1 << stage
        paired = shadow.reshape(shadow.shape[0], -1, 2, stride)
        left, right = paired[:, :, 0, :], paired[:, :, 1, :]
        absolute_left, absolute_right = cp.abs(left), cp.abs(right)
        threshold = mean_abs[:, None, None]
        feature = (
            cp.int64(role_ordinal << 6)
            | ((left < 0).astype(cp.int64) << cp.int64(5))
            | ((right < 0).astype(cp.int64) << cp.int64(4))
            | ((absolute_left > absolute_right).astype(cp.int64) << cp.int64(3))
            | (((absolute_left + absolute_right) > 2 * threshold).astype(cp.int64)
               << cp.int64(2))
            | ((absolute_left > threshold).astype(cp.int64) << cp.int64(1))
            | (absolute_right > threshold).astype(cp.int64)
        )
        operation = table[stage, feature]
        swap = (operation & cp.uint8(1)) != 0
        a = cp.where(swap, right, left)
        b = cp.where(swap, left, right)
        a = cp.where((operation & cp.uint8(2)) != 0, -a, a)
        b = cp.where((operation & cp.uint8(4)) != 0, -b, b)
        shadow = cp.stack((a + b, a - b), axis=2).reshape(shadow.shape)
        schedules.append(operation)
    return schedules


def _synthesis(cp: Any, coefficients: Any, schedules: list[Any],
               spec: Any) -> Any:
    values = coefficients
    for stage in range(spec.STAGES):
        stride = 1 << stage
        paired = values.reshape(values.shape[0], -1, 2, stride)
        left, right = paired[:, :, 0, :], paired[:, :, 1, :]
        operation = schedules[stage]
        swap = (operation & cp.uint8(1)) != 0
        a = cp.where(swap, right, left)
        b = cp.where(swap, left, right)
        a = cp.where((operation & cp.uint8(2)) != 0, -a, a)
        b = cp.where((operation & cp.uint8(4)) != 0, -b, b)
        values = cp.stack((a + b, a - b), axis=2).reshape(values.shape)
    return values / cp.float64(64.0)


def _analysis(cp: Any, values: Any, schedules: list[Any], spec: Any) -> Any:
    transformed = values
    for stage in reversed(range(spec.STAGES)):
        stride = 1 << stage
        paired = transformed.reshape(transformed.shape[0], -1, 2, stride)
        left, right = paired[:, :, 0, :], paired[:, :, 1, :]
        x0, x1 = left + right, left - right
        operation = schedules[stage]
        x0 = cp.where((operation & cp.uint8(2)) != 0, -x0, x0)
        x1 = cp.where((operation & cp.uint8(4)) != 0, -x1, x1)
        swap = (operation & cp.uint8(1)) != 0
        u = cp.where(swap, x1, x0)
        v = cp.where(swap, x0, x1)
        transformed = cp.stack((u, v), axis=2).reshape(transformed.shape)
    return transformed / cp.float64(64.0)


def decode_fine_tile(cp: Any, np: Any, fine_tile_bytes: bytes,
                     symbols: Any, reconstruction: Any,
                     role_ordinal: int, spec: Any) -> tuple[Any, dict[str, Any]]:
    blocks = spec.COARSE_TILE_VALUES // spec.BLOCK_VALUES
    require(type(fine_tile_bytes) is bytes and
            len(fine_tile_bytes) == blocks * spec.FINE_RECORD_BYTES,
            "independent fine-tile bytes")
    scale_codes: list[int] = []
    signs: list[tuple[bool, ...]] = []
    reencoded = bytearray()
    for begin in range(0, len(fine_tile_bytes), spec.FINE_RECORD_BYTES):
        record = fine_tile_bytes[begin:begin + spec.FINE_RECORD_BYTES]
        scale, positive = spec.unpack_record(record)
        canonical = spec.pack_record(scale, positive)
        require(canonical == record, "independent fine record reencode")
        scale_codes.append(scale)
        signs.append(positive)
        reencoded.extend(canonical)
    require(bytes(reencoded) == fine_tile_bytes,
            "independent fine tile aggregate reencode")
    coarse = cp.asarray(reconstruction, dtype=cp.float64).reshape(
        blocks, spec.BLOCK_VALUES)
    coarse_max = cp.max(cp.abs(coarse), axis=1)
    scale_gpu = cp.asarray(scale_codes, dtype=cp.float64)
    alpha = (coarse_max * scale_gpu * scale_gpu /
             cp.float64(spec.SCALE_DENOMINATOR))
    sign_gpu = cp.asarray(np.asarray(signs, dtype=np.bool_))
    require(tuple(sign_gpu.shape) == (blocks, spec.ACTIVE_RANK),
            "independent sign geometry")
    coefficients = cp.zeros(
        (blocks, spec.BLOCK_VALUES), dtype=cp.float64)
    coefficients[:, :spec.ACTIVE_RANK] = cp.where(
        sign_gpu, alpha[:, None], -alpha[:, None])
    coefficients[:, :spec.ACTIVE_RANK] = cp.where(
        scale_gpu[:, None] == cp.float64(0.0), cp.float64(0.0),
        coefficients[:, :spec.ACTIVE_RANK])
    require(not bool(cp.any(
        coefficients[:, spec.ACTIVE_RANK:] != cp.float64(0.0)).item()),
        "independent coefficient tail zero")
    schedules = _schedule(cp, symbols, role_ordinal, spec)
    correction = _synthesis(cp, coefficients, schedules, spec)
    recovered = _analysis(cp, correction, schedules, spec)
    scale = max(1.0, float(cp.max(cp.abs(coefficients)).item()))
    tail_max = float(cp.max(cp.abs(
        recovered[:, spec.ACTIVE_RANK:])).item())
    active_max_error = float(cp.max(cp.abs(
        recovered[:, :spec.ACTIVE_RANK] -
        coefficients[:, :spec.ACTIVE_RANK])).item())
    require(tail_max <= 2e-11 * scale and
            active_max_error <= 2e-11 * scale,
            "independent correction dyadic-span containment")
    return correction.reshape(-1), {
        "blocks": blocks,
        "fine_bytes": len(fine_tile_bytes),
        "fine_sha256": sha256(fine_tile_bytes),
        "all_records_independently_reencode": True,
        "aggregate_fine_tile_reencode_matches": True,
        "active_rank": spec.ACTIVE_RANK,
        "audited_parent_rank": spec.AUDITED_PARENT_RANK,
        "maximum_transformed_tail_abs_fp64": tail_max,
        "maximum_active_coefficient_roundtrip_error_fp64": active_max_error,
        "correction_in_active_rank376_subset_parent_rank384": True,
    }


def decode_composite(cp: Any, np: Any, composite: bytes,
                     source_role_bf16: Mapping[str, bytes],
                     runtime: Any, v6_codec: Any,
                     spec: Any) -> dict[str, Any]:
    header, coarse, fine = spec.split_composite(composite)
    geometry = runtime.packet.ExpertGeometry(spec.INTERMEDIATE, spec.HIDDEN)
    require(geometry.target_eligible and geometry.frame_bytes == len(coarse),
            "independent exact v6 coarse geometry")
    require(set(source_role_bf16) == set(spec.ROLES),
            "independent exact role source")
    fine_records = spec.split_fine_stream(fine)
    role_reconstructions: dict[str, Any] = {}
    tile_receipts: list[dict[str, Any]] = []
    aggregate_reencode = bytearray()
    for role_ordinal, role in enumerate(spec.ROLES):
        reconstructed_tiles = []
        for tile_ordinal in range(spec.COARSE_TILES_PER_ROLE):
            coarse_ordinal = role_ordinal * spec.COARSE_TILES_PER_ROLE + tile_ordinal
            coarse_begin = coarse_ordinal * spec.COARSE_RECORD_BYTES
            packet = coarse[coarse_begin:coarse_begin + spec.COARSE_RECORD_BYTES]
            decoded = v6_codec.decode_tile_v6(packet, runtime)
            require(decoded.canonical_packet == packet and
                    decoded.report["canonical_reencode_matches"] is True,
                    "independent exact v6 tile reencode")
            block_begin = (
                role_ordinal * spec.FINE_BLOCKS_PER_ROLE
                + tile_ordinal * (spec.COARSE_TILE_VALUES // spec.BLOCK_VALUES))
            block_end = block_begin + spec.COARSE_TILE_VALUES // spec.BLOCK_VALUES
            tile_fine = b"".join(fine_records[block_begin:block_end])
            correction, receipt = decode_fine_tile(
                cp, np, tile_fine, decoded.canonical_symbols_i32,
                decoded.reconstruction_f32, role_ordinal, spec)
            reconstructed = (
                cp.asarray(decoded.reconstruction_f32, dtype=cp.float64)
                + correction)
            reconstructed_tiles.append(reconstructed)
            aggregate_reencode.extend(tile_fine)
            tile_receipts.append({
                "role": role,
                "role_ordinal": role_ordinal,
                "tile_ordinal": tile_ordinal,
                "coarse_packet_sha256": sha256(packet),
                **receipt,
            })
        role_reconstructions[role] = cp.concatenate(reconstructed_tiles)
    require(bytes(aggregate_reencode) == fine,
            "independent full fine-stream reencode equality")

    rows: list[dict[str, Any]] = []
    pooled_sse = 0.0
    pooled_energy = 0.0
    reconstruction_hashes: dict[str, str] = {}
    for role in spec.ROLES:
        raw = source_role_bf16[role]
        words = np.frombuffer(raw, dtype="<u2")
        source_host = ((words.astype(np.uint32) << np.uint32(16))
                       .view(np.float32).astype(np.float64))
        source = cp.asarray(source_host, dtype=cp.float64)
        reconstruction = role_reconstructions[role]
        require(source.shape == reconstruction.shape == (spec.ROLE_VALUES,),
                "independent role score geometry")
        error = source - reconstruction
        sse = float(cp.sum(error * error, dtype=cp.float64).item())
        energy = float(cp.sum(source * source, dtype=cp.float64).item())
        require(math.isfinite(sse) and math.isfinite(energy) and
                sse >= 0.0 and energy >= 0.0,
                "independent finite FP64 score")
        reconstruction_host = cp.asnumpy(reconstruction).astype("<f8", copy=False)
        reconstruction_hashes[role] = sha256(reconstruction_host.tobytes())
        pooled_sse += sse
        pooled_energy += energy
        rows.append({
            "role": role,
            "source_bf16_sha256": sha256(raw),
            "fine_reconstruction_f64_sha256": reconstruction_hashes[role],
            "sse_fp64": sse,
            "source_energy_fp64": energy,
            "relative_mse": sse / energy if energy else None,
        })
    require(pooled_energy > 0.0, "independent positive pooled energy")
    relative_mse = pooled_sse / pooled_energy
    return {
        "schema": "tactic-dh384-finite-v3-independent-decode-receipt-v1",
        "status": "PASS_LITERAL_COMPOSITE_EXACT_V6_AND_FINE_REENCODE",
        "header_sha256": sha256(composite[:spec.PILOT_HEADER_BYTES]),
        "coarse_sha256": sha256(coarse),
        "fine_sha256": sha256(fine),
        "composite_sha256": sha256(composite),
        "composite_bytes": len(composite),
        "literal_composite_bpw_exact": "320/128",
        "exact_v6_coarse_reencode_matches_all_records": True,
        "fine_records": spec.FINE_RECORDS,
        "fine_record_bytes": spec.FINE_RECORD_BYTES,
        "fine_records_independently_decode_reencode": True,
        "fine_full_stream_reencode_matches": True,
        "all_corrections_in_active_rank376_subset_parent_rank384": True,
        "original_domain_score": {
            "domain": "exact encoder-input BF16 Gate/Up/DownT coordinates",
            "summation_dtype": "IEEE-754 binary64 CuPy reduction",
            "rows": rows,
            "pooled_sse_fp64": pooled_sse,
            "pooled_source_energy_fp64": pooled_energy,
            "pooled_relative_mse": relative_mse,
            "F_at_literal_2p5_bpw": relative_mse * 32.0,
        },
        "tile_receipts": tile_receipts,
        "traffic": spec.single_expert_traffic(
            start_offset_mod_page=0, external_passes=1),
        "six_expert_global_packet_emitted_or_parsed": False,
        "seventy_three_over_seventy_two_claim": False,
        "universal_tail_claim": False,
        "non_qwen_portability_claim": False,
        "header_canonical_reencode_matches":
            spec.make_header(header["bindings"]) ==
            composite[:spec.PILOT_HEADER_BYTES],
    }
