#!/usr/bin/env python3
"""CuPy numerical producer for the v4 finite packet.

Importing this module is payload-free and does not import NumPy or CuPy.
``load_encoder_runtime`` authenticates the two pinned numerical source files
before executing their exact bytes.  A separate runtime/environment freeze is
still required before a production result can be promoted.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import struct
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from packet_format import (
    ETA,
    N,
    PAYLOAD_BITS,
    PROFILE_Q,
    RESERVOIR_BYTES,
    ROLES,
    ContractError,
    ExpertGeometry,
    pack_reservoir,
    parse_expert_frame,
    require,
    seed_pair,
)


ENCODER_RELATIVE = Path("src/polaris_sc_v2_rht_encoder.py")
ENCODER_BYTES = 29_633
ENCODER_SHA256 = "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0"
DECODER_RELATIVE = Path("strata_v2_klt_mixed_independent_auditor_v1.py")
DECODER_BYTES = 116_835
DECODER_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"


@dataclass(frozen=True)
class EncoderRuntime:
    encoder: Any
    construction: Any
    numpy: Any
    cupy: Any
    source_receipts: tuple[dict[str, Any], ...]


def _read_exact_pinned(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    require(path.is_absolute(), "pinned source path must be absolute")
    # Reject symlinks in every existing path component.  O_NOFOLLOW protects
    # the leaf on POSIX; the explicit walk gives the same source-only rule on
    # platforms where O_NOFOLLOW is absent.
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        require(not cursor.is_symlink(), f"pinned source symlink: {cursor}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot open pinned source: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "pinned source regular file")
        require(before.st_size == expected_bytes, "pinned source byte count")
        chunks: list[bytes] = []
        remaining = expected_bytes
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), "pinned source premature EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", "pinned source trailing bytes")
        packet = b"".join(chunks)
        require(hashlib.sha256(packet).hexdigest() == expected_sha256, "pinned source digest")
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "pinned source identity drift",
        )
        return packet
    finally:
        os.close(descriptor)


def _module_from_authenticated_bytes(name: str, packet: bytes, digest: str) -> Any:
    require(name not in sys.modules, f"authenticated module name already loaded: {name}")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = digest
    sys.modules[name] = module
    try:
        exec(compile(packet, module.__file__, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_encoder_runtime(repo_root: Path) -> EncoderRuntime:
    """Load the exact CuPy producer and procedural independent construction."""

    root = repo_root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "repository root")
    sources = (
        ("tacn18_v4_encoder_core", ENCODER_RELATIVE, ENCODER_BYTES, ENCODER_SHA256),
        ("tacn18_v4_construction_core", DECODER_RELATIVE, DECODER_BYTES, DECODER_SHA256),
    )
    packets: list[tuple[str, bytes, str, Path]] = []
    receipts: list[dict[str, Any]] = []
    for name, relative, expected_bytes, digest in sources:
        path = (root / relative).resolve(strict=True)
        require(path == root / relative, "pinned source canonical path")
        packet = _read_exact_pinned(path, expected_bytes, digest)
        packets.append((name, packet, digest, relative))
        receipts.append(
            {
                "id": name,
                "relative_path": relative.as_posix(),
                "bytes": len(packet),
                "sha256": hashlib.sha256(packet).hexdigest(),
            }
        )
    encoder = _module_from_authenticated_bytes(packets[0][0], packets[0][1], packets[0][2])
    construction = _module_from_authenticated_bytes(packets[1][0], packets[1][1], packets[1][2])
    # These imports occur only after the producer sources were authenticated.
    import cupy as cp
    import numpy as np

    require(encoder.cp is cp and encoder.np is np, "encoder numerical module identity")
    require(construction.np is np, "construction NumPy identity")
    return EncoderRuntime(encoder, construction, np, cp, tuple(receipts))


def _read_canonical_bf16(path: Path, expected_values: int) -> bytes:
    require(path.is_absolute(), "canonical BF16 path must be absolute")
    require(expected_values > 0, "canonical BF16 value count")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot open canonical BF16 source: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        expected_bytes = 2 * expected_values
        require(stat.S_ISREG(before.st_mode) and before.st_size == expected_bytes, "BF16 source geometry")
        chunks: list[bytes] = []
        remaining = expected_bytes
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), "BF16 premature EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", "BF16 trailing bytes")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "BF16 source identity drift",
        )
    finally:
        os.close(descriptor)
    # Reject infinities and NaNs before a numerical import sees the payload.
    for offset in range(0, len(raw), 2):
        word = raw[offset] | (raw[offset + 1] << 8)
        require(word & 0x7F80 != 0x7F80, "non-finite BF16 source word")
    return raw


def canonical_padded_tile(raw_role_bf16: bytes, geometry: ExpertGeometry, tile_ordinal: int) -> tuple[bytes, int, bool]:
    """Extract one canonical tile and append implicit BF16 +0 tail values."""

    require(type(raw_role_bf16) is bytes, "raw role BF16 bytes")
    require(len(raw_role_bf16) == 2 * geometry.role_values, "raw role BF16 geometry")
    valid_values = geometry.valid_values(tile_ordinal)
    begin = 2 * tile_ordinal * N
    end = begin + 2 * valid_values
    tile = raw_role_bf16[begin:end] + bytes(2 * (N - valid_values))
    require(len(tile) == 2 * N, "padded tile byte count")
    zero = True
    for offset in range(0, end - begin, 2):
        word = tile[offset] | (tile[offset + 1] << 8)
        if word & 0x7FFF:
            zero = False
            break
    return tile, valid_values, zero


def _flags(runtime: EncoderRuntime) -> tuple[dict[str, Any], list[Any]]:
    profile = runtime.construction.profile_parameters(PROFILE_Q, ETA)
    require(float(profile["rate_bpw"]) == 1.75 + PROFILE_Q / 256.0, "profile rate")
    reverse = runtime.construction.bit_reverse_indices(N)
    flags = runtime.construction.bec_freeze_flags(N, profile["capacities"], reverse)
    require(len(flags) == 6 and all(row.shape == (N,) for row in flags), "freeze flags")
    return profile, flags


def encode_tile(
    raw_role_bf16: bytes,
    geometry: ExpertGeometry,
    role_ordinal: int,
    tile_ordinal: int,
    runtime: EncoderRuntime,
) -> tuple[bytes, dict[str, Any]]:
    """Encode one source tile; overflow is a terminal architecture failure."""

    require(0 <= role_ordinal < len(ROLES), "role ordinal")
    tile, valid_values, zero_tile = canonical_padded_tile(raw_role_bf16, geometry, tile_ordinal)
    sc_seed, rht_seed = seed_pair(
        role_ordinal, geometry.intermediate, geometry.hidden, tile_ordinal
    )
    if zero_tile:
        packet = pack_reservoir(
            b"", 0, 1.0, geometry, role_ordinal, tile_ordinal, zero_tile=True
        )
        return packet, {
            "status": "PASS_EXACT_ZERO_TILE",
            "role": ROLES[role_ordinal],
            "tile_ordinal": tile_ordinal,
            "valid_values": valid_values,
            "logical_bits": 0,
            "capacity_margin_bits": PAYLOAD_BITS,
            "packet_sha256": hashlib.sha256(packet).hexdigest(),
        }

    profile, flags = _flags(runtime)
    with tempfile.TemporaryDirectory(prefix="tacn18_v4_tile_") as temporary:
        staging = Path(temporary) / "canonical_padded_tile.bf16"
        with staging.open("xb") as stream:
            stream.write(tile)
            stream.flush()
            os.fsync(stream.fileno())
        arguments = SimpleNamespace(
            block_length=N,
            seed=sc_seed,
            input_bf16=staging,
            input_block_start=0,
            canonical_source_id=(
                f"tacn18-v4:role:{role_ordinal}:shape:"
                f"{geometry.intermediate}x{geometry.hidden}:tile:{tile_ordinal}"
            ),
            canonical_block_index=tile_ordinal,
            apply_rht=True,
            rht_seed=rht_seed,
            sigma_source=1.0,
            test_distortion=float(profile["test_channel_distortion"]),
            eta=ETA,
            alphabet_size=64,
            decision="map",
            emit_container_hex=True,
        )
        row = runtime.encoder.run_trial(arguments, 0, profile["capacities"], flags)
    container_hex = row.pop("_container_hex")
    require(type(container_hex) is str, "encoder literal container")
    legacy = bytes.fromhex(container_hex)
    require(len(legacy) >= 8, "encoder container truncated")
    logical_bits, scale = struct.unpack_from("<If", legacy, 0)
    payload = legacy[8:]
    require(logical_bits == int(row["arithmetic_logical_bits"]), "encoder length disagreement")
    require(len(payload) == (logical_bits + 7) // 8, "encoder payload length")
    require(hashlib.sha256(payload).hexdigest() == row["arithmetic_payload_sha256"], "encoder payload hash")
    require(logical_bits <= PAYLOAD_BITS, "fixed-reservoir overflow; retry forbidden")
    packet = pack_reservoir(
        payload,
        logical_bits,
        float(scale),
        geometry,
        role_ordinal,
        tile_ordinal,
    )
    require(len(packet) == RESERVOIR_BYTES, "encoded packet bytes")
    return packet, {
        "status": "PASS_FINITE_TILE_WITHIN_RESERVOIR",
        "role": ROLES[role_ordinal],
        "tile_ordinal": tile_ordinal,
        "valid_values": valid_values,
        "padded_values": N - valid_values,
        "logical_bits": logical_bits,
        "capacity_margin_bits": PAYLOAD_BITS - logical_bits,
        "decoder_scale_fp32": float(scale),
        "normalized_rht_domain_relative_mse": float(row["relative_mse"]),
        "arithmetic_roundtrip_bits_match": row["arithmetic_roundtrip_bits_match"] is True,
        "causal_decoder_frequencies_match": row["causal_decoder_frequencies_match"] is True,
        "reconstruction_indices_match": row["reconstruction_indices_match"] is True,
        "source_padded_tile_sha256": hashlib.sha256(tile).hexdigest(),
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
    }


def encode_expert_frame(
    role_paths: Mapping[str, Path],
    geometry: ExpertGeometry,
    runtime: EncoderRuntime,
) -> tuple[bytes, dict[str, Any]]:
    """Emit one canonical contiguous expert frame.

    The mapping must contain exactly Gate, Up, and already-transposed Down.
    No expert, layer, checkpoint, or model identity enters the codec.
    """

    require(set(role_paths) == set(ROLES), "exact canonical role-path mapping")
    packets: list[bytes] = []
    reports: list[dict[str, Any]] = []
    for role_ordinal, role in enumerate(ROLES):
        path = role_paths[role]
        require(type(path) is Path, "role path type")
        raw = _read_canonical_bf16(path, geometry.role_values)
        for tile_ordinal in range(geometry.streams_per_role):
            packet, report = encode_tile(raw, geometry, role_ordinal, tile_ordinal, runtime)
            packets.append(packet)
            reports.append(report)
    frame = b"".join(packets)
    parsed = parse_expert_frame(frame)
    require(parsed.geometry == geometry and len(frame) == geometry.frame_bytes, "encoded frame closure")
    return frame, {
        "schema": "tactic_actual_coarse_n18_v4_encode_receipt",
        "status": "PASS_ENCODED_AWAITING_INDEPENDENT_NUMERICAL_DECODE",
        "geometry": {
            "intermediate": geometry.intermediate,
            "hidden": geometry.hidden,
            "weights": geometry.values,
        },
        "target_eligible_exact_307_over_128": geometry.target_eligible,
        "frame_bytes": len(frame),
        "physical_bpw": 8.0 * len(frame) / geometry.values,
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "tiles": reports,
        "runtime_source_receipts": list(runtime.source_receipts),
        "claim_boundary": "encoder receipt only; independent decode/re-encode and source-domain score required",
    }
