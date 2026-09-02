#!/usr/bin/env python3
"""Independent causal decoder and original-domain scorer for v4.

This module imports no encoder implementation.  It regenerates the profile,
Q31 BEC construction, frozen bits, causal frequencies, arithmetic symbols,
and inverse signed RHT from the packet plus universal constants.  The file
entry point reads each compressed reservoir once and buffers decoded state;
it never performs a second compressed-frame pass.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from packet_format import (
    ETA,
    N,
    PROFILE_Q,
    RESERVOIR_BYTES,
    ROLES,
    SQRT_N,
    ContractError,
    ExpertGeometry,
    ParsedReservoir,
    pack_reservoir,
    parse_reservoir,
    require,
)


DECODER_RELATIVE = Path("strata_v2_klt_mixed_independent_auditor_v1.py")
DECODER_BYTES = 116_835
DECODER_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"


@dataclass(frozen=True)
class DecoderRuntime:
    decoder: Any
    numpy: Any
    source_receipt: dict[str, Any]


@dataclass(frozen=True)
class DecodedTile:
    parsed: ParsedReservoir
    canonical_symbols_i16: Any
    reconstruction_f32: Any
    canonical_packet: bytes
    report: dict[str, Any]


def _read_pinned_decoder(path: Path) -> bytes:
    require(path.is_absolute(), "decoder source path must be absolute")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        require(not cursor.is_symlink(), f"decoder source symlink: {cursor}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot open independent decoder source: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == DECODER_BYTES,
            "independent decoder source identity",
        )
        chunks: list[bytes] = []
        remaining = DECODER_BYTES
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), "independent decoder source premature EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", "independent decoder source trailing bytes")
        packet = b"".join(chunks)
        require(hashlib.sha256(packet).hexdigest() == DECODER_SHA256, "independent decoder source hash")
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "independent decoder source drift",
        )
        return packet
    finally:
        os.close(descriptor)


def load_decoder_runtime(repo_root: Path) -> DecoderRuntime:
    root = repo_root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "repository root")
    path = (root / DECODER_RELATIVE).resolve(strict=True)
    require(path == root / DECODER_RELATIVE, "independent decoder canonical path")
    packet = _read_pinned_decoder(path)
    name = "tacn18_v4_independent_decoder_core"
    require(name not in sys.modules, "independent decoder module already loaded")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{DECODER_SHA256}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = DECODER_SHA256
    sys.modules[name] = module
    try:
        exec(compile(packet, module.__file__, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    import numpy as np

    require(module.np is np, "independent decoder NumPy identity")
    return DecoderRuntime(
        module,
        np,
        {
            "id": name,
            "relative_path": DECODER_RELATIVE.as_posix(),
            "bytes": len(packet),
            "sha256": hashlib.sha256(packet).hexdigest(),
        },
    )


def _integer_inverse_symbols(np: Any, indices: Any, rht_seed: int) -> Any:
    values = indices.astype(np.int32) - np.int32(31)
    width = 1
    while width < N:
        view = values.reshape(-1, 2, width)
        left = view[:, 0, :].copy()
        right = view[:, 1, :].copy()
        view[:, 0, :] = left + right
        view[:, 1, :] = left - right
        width *= 2
    with np.errstate(over="ignore"):
        z = np.arange(N, dtype=np.uint64) + np.uint64(rht_seed)
        z += np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    signs = np.where((z & np.uint64(1)) == 0, 1, -1).astype(np.int32)
    values *= signs
    maximum = int(np.max(np.abs(values.astype(np.int64))))
    require(maximum <= 32_767, f"canonical I16 symbol overflow: {maximum}")
    return values.astype("<i2")


def decode_reservoir(packet: bytes, runtime: DecoderRuntime) -> DecodedTile:
    parsed = parse_reservoir(packet)
    np = runtime.numpy
    if parsed.zero_tile:
        symbols = np.zeros(N, dtype="<i2")
        reconstruction = np.zeros(parsed.valid_values, dtype="<f4")
        canonical = pack_reservoir(
            b"",
            0,
            1.0,
            parsed.geometry,
            parsed.role_ordinal,
            parsed.tile_ordinal,
            zero_tile=True,
        )
        require(canonical == packet, "zero-tile canonical re-encode")
        return DecodedTile(
            parsed,
            symbols,
            reconstruction,
            canonical,
            {
                "status": "PASS_EXACT_ZERO_TILE_DECODE",
                "logical_bits": 0,
                "canonical_reencode_matches": True,
                "valid_values": parsed.valid_values,
                "padded_values": N - parsed.valid_values,
                "canonical_symbols_i16_sha256": hashlib.sha256(symbols.tobytes()).hexdigest(),
                "reconstruction_f32_sha256": hashlib.sha256(reconstruction.tobytes()).hexdigest(),
            },
        )

    decoder = runtime.decoder
    profile = decoder.profile_parameters(PROFILE_Q, ETA)
    require(float(profile["rate_bpw"]) == 1.75 + PROFILE_Q / 256.0, "profile rate")
    reverse = decoder.bit_reverse_indices(N)
    layers = decoder.sc_layers(N)
    flags = decoder.bec_freeze_flags(N, profile["capacities"], reverse)
    require(len(flags) == 6 and all(row.shape == (N,) for row in flags), "freeze flags")
    arithmetic = decoder.ArithmeticBinaryDecoder(parsed.payload, 0, parsed.logical_bits)
    alphabet = ETA * np.arange(-31, 33, dtype=np.float64)
    weights = np.exp(-0.5 * (alphabet / float(profile["sigma_reconstruction"])) ** 2)
    previous = np.zeros(N, dtype=np.int16)
    selected_rows = []
    frequency_rows = []
    level_rows = []
    for level_index, flag in enumerate(flags):
        level = level_index + 1
        frozen_rng = np.random.default_rng(parsed.sc_seed_u32 + 1_000_003 * level)
        frozen_external = frozen_rng.integers(0, 2, size=N, dtype=np.uint8)
        prior = decoder.leaf_prior_ratios(weights, previous, level)
        x_bit, frequencies, selected = decoder.decode_sc_level(
            prior, flag, frozen_external, reverse, layers, arithmetic
        )
        previous += (1 << level_index) * x_bit.astype(np.int16)
        selected_rows.append(selected)
        frequency_rows.append(frequencies)
        level_rows.append(
            {
                "level": level,
                "selected": int(frequencies.size),
                "capacity": float(profile["capacities"][level_index]),
            }
        )
    selected_all = np.concatenate(selected_rows)
    frequency_all = np.concatenate(frequency_rows)
    canonical_payload, canonical_bits = decoder.arithmetic_encode_binary(
        selected_all, frequency_all
    )
    require(canonical_bits == parsed.logical_bits, "canonical arithmetic length")
    require(canonical_payload == parsed.payload, "canonical arithmetic bytes")
    canonical = pack_reservoir(
        canonical_payload,
        canonical_bits,
        parsed.decoder_scale_fp32,
        parsed.geometry,
        parsed.role_ordinal,
        parsed.tile_ordinal,
    )
    require(canonical == packet, "canonical packet re-encode")

    symbols = _integer_inverse_symbols(np, previous, parsed.rht_seed_u64)
    reconstruction64_full = symbols.astype(np.float64) * (
        ETA * parsed.decoder_scale_fp32 / SQRT_N
    )
    transformed = alphabet[previous] * parsed.decoder_scale_fp32
    floating_reference = decoder.inverse_signed_rht(
        transformed, parsed.rht_seed_u64, "numpy"
    )
    inverse_difference = float(np.max(np.abs(reconstruction64_full - floating_reference)))
    require(inverse_difference <= 2e-14, f"integer/float inverse RHT parity: {inverse_difference}")
    # Only the shape-bound prefix is a source reconstruction.  The padded
    # suffix is internal decoder state and never becomes a model weight.
    reconstruction = reconstruction64_full[: parsed.valid_values].astype("<f4")
    return DecodedTile(
        parsed,
        symbols,
        reconstruction,
        canonical,
        {
            "status": "PASS_CAUSAL_INDEPENDENT_DECODE_REENCODE",
            "logical_bits": parsed.logical_bits,
            "selected_polar_bits": int(selected_all.size),
            "levels": level_rows,
            "valid_values": parsed.valid_values,
            "padded_values": N - parsed.valid_values,
            "canonical_reencode_matches": True,
            "integer_float_inverse_max_abs": inverse_difference,
            "canonical_symbol_abs_max": int(np.max(np.abs(symbols.astype(np.int32)))),
            "canonical_symbols_i16_sha256": hashlib.sha256(symbols.tobytes()).hexdigest(),
            "reconstruction_f32_sha256": hashlib.sha256(reconstruction.tobytes()).hexdigest(),
            "arithmetic_physical_bits": parsed.logical_bits,
            "arithmetic_virtual_zero_extension_rule": (
                "virtual zeros after hard logical EOF are decoder state, not physical or logical bits"
            ),
        },
    )


def _read_one_record(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    remaining = RESERVOIR_BYTES
    while remaining:
        chunk = os.read(descriptor, remaining)
        require(bool(chunk), "compressed frame truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_source_bf16(path: Path, expected_values: int) -> bytes:
    require(path.is_absolute(), "source score path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot open source score file: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        expected_bytes = 2 * expected_values
        require(stat.S_ISREG(before.st_mode) and before.st_size == expected_bytes, "source score geometry")
        chunks: list[bytes] = []
        remaining = expected_bytes
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), "source score premature EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", "source score trailing bytes")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "source score identity drift",
        )
        return raw
    finally:
        os.close(descriptor)


def _score_roles(
    reconstructions: Mapping[str, Any],
    source_role_paths: Mapping[str, Path],
    geometry: ExpertGeometry,
    runtime: DecoderRuntime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(set(source_role_paths) == set(ROLES), "exact source score role mapping")
    np = runtime.numpy
    residuals: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    total_sse = 0.0
    total_energy = 0.0
    for role in ROLES:
        raw = _read_source_bf16(source_role_paths[role], geometry.role_values)
        words = np.frombuffer(raw, dtype="<u2")
        source = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
        require(bool(np.all(np.isfinite(source))), "non-finite BF16 scoring source")
        reconstruction = np.asarray(reconstructions[role], dtype=np.float64)
        require(reconstruction.shape == source.shape, "source/reconstruction score geometry")
        residual = source - reconstruction
        sse = float(np.dot(residual, residual))
        energy = float(np.dot(source, source))
        require(math.isfinite(sse) and math.isfinite(energy) and energy > 0.0, "finite positive source score")
        residuals[role] = residual
        total_sse += sse
        total_energy += energy
        rows.append(
            {
                "role": role,
                "weights": geometry.role_values,
                "sse_fp64": sse,
                "source_energy_fp64": energy,
                "relative_mse": sse / energy,
                "source_bf16_sha256": hashlib.sha256(raw).hexdigest(),
                "residual_f64_sha256": hashlib.sha256(
                    residual.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
            }
        )
    return residuals, {
        "domain": "original canonical BF16 source coordinates after inverse signed RHT",
        "roles": rows,
        "pooled_sse_fp64": total_sse,
        "pooled_source_energy_fp64": total_energy,
        "pooled_relative_mse": total_sse / total_energy,
    }


def decode_expert_frame_file(
    frame_path: Path,
    runtime: DecoderRuntime,
    *,
    source_role_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Decode one frame in exactly one sequential compressed-byte pass.

    Returns ``(reconstructions, receipt, residuals_or_none)``.  Coarse symbols
    and reconstructions are buffered in host memory for all downstream graph
    or fine-stage work, preventing the invalid second compressed-frame read.
    """

    require(frame_path.is_absolute(), "expert frame path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(frame_path, flags)
    except OSError as exc:
        raise ContractError(f"cannot open expert frame: {exc}") from exc
    digest = hashlib.sha256()
    decoded_rows: list[DecodedTile] = []
    total_read = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size >= 3 * RESERVOIR_BYTES, "expert frame file")
        first_packet = _read_one_record(descriptor)
        total_read += len(first_packet)
        digest.update(first_packet)
        first = decode_reservoir(first_packet, runtime)
        require(first.parsed.role_ordinal == 0 and first.parsed.tile_ordinal == 0, "canonical first record")
        geometry = first.parsed.geometry
        require(before.st_size == geometry.frame_bytes, "shape-bound expert frame byte count")
        decoded_rows.append(first)
        for record_ordinal in range(1, geometry.records):
            packet = _read_one_record(descriptor)
            total_read += len(packet)
            digest.update(packet)
            row = decode_reservoir(packet, runtime)
            expected_role, expected_tile = divmod(record_ordinal, geometry.streams_per_role)
            require(row.parsed.geometry == geometry, "frame geometry drift")
            require(
                (row.parsed.role_ordinal, row.parsed.tile_ordinal)
                == (expected_role, expected_tile),
                "noncanonical role/tile order",
            )
            decoded_rows.append(row)
        require(os.read(descriptor, 1) == b"", "expert frame trailing bytes")
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "expert frame identity drift",
        )
    finally:
        os.close(descriptor)

    require(total_read == geometry.frame_bytes, "one-pass compressed byte count")
    reconstructions: dict[str, Any] = {}
    symbols: dict[str, Any] = {}
    for role_ordinal, role in enumerate(ROLES):
        role_rows = [row for row in decoded_rows if row.parsed.role_ordinal == role_ordinal]
        reconstruction = runtime.numpy.concatenate([row.reconstruction_f32 for row in role_rows])
        require(reconstruction.size == geometry.role_values, "role reconstruction geometry")
        reconstructions[role] = reconstruction
        symbols[role] = tuple(row.canonical_symbols_i16 for row in role_rows)

    score = None
    residuals = None
    if source_role_paths is not None:
        residuals, score = _score_roles(reconstructions, source_role_paths, geometry, runtime)

    receipt = {
        "schema": "tactic_actual_coarse_n18_v4_independent_decode_receipt",
        "status": "PASS_ONE_PASS_INDEPENDENT_DECODE_REENCODE",
        "geometry": {
            "intermediate": geometry.intermediate,
            "hidden": geometry.hidden,
            "weights": geometry.values,
        },
        "target_eligible_exact_307_over_128": geometry.target_eligible,
        "frame_bytes": geometry.frame_bytes,
        "physical_bpw": 8.0 * geometry.frame_bytes / geometry.values,
        "frame_sha256": digest.hexdigest(),
        "compressed_frame_passes": 1,
        "compressed_frame_bytes_read": total_read,
        "compressed_frame_reread_bytes": 0,
        "decoded_coarse_state_buffered": True,
        "canonical_reencode_all_match": all(row.canonical_packet for row in decoded_rows),
        "records": [row.report for row in decoded_rows],
        "reconstruction_f32_sha256": {
            role: hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()
            for role, value in reconstructions.items()
        },
        "canonical_symbols_i16_sha256": {
            role: hashlib.sha256(
                b"".join(value.astype("<i2", copy=False).tobytes() for value in rows)
            ).hexdigest()
            for role, rows in symbols.items()
        },
        "original_domain_score": score,
        "decoder_source_receipt": runtime.source_receipt,
    }
    return reconstructions, receipt, residuals
