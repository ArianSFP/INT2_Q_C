#!/usr/bin/env python3
"""Packet-replayed Ramanujan search and literal composite reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


V0_MANIFEST_SHA256 = "287b8ad4c377956c9bb264d9d8731893a83e45180f75472f9b42968e3f20acde"
V0_FILE_SHA256 = {
    "packet.py": "51024bfa32d7063f877136f1d41937bc986862fa0bbbb74e5139e781082c7c85",
    "ramanujan_codec.py": "915b913b468243ec57800e1a717606e6c017921733977b09b737ee2d1fc1cc62",
    "container.py": "833987b27f05e61c9563c1c28c4880fdf86fce52e1979e0d61ede4dd246e6e24",
}
ROLE_ORDER = ("gate", "up", "down_transposed")
TARGET_D = 0.025
MIN_CONTROL_EXCESS_BPW = 0.03
GAUSSIAN_SEEDS = (
    10619863, 10619881, 10619909, 10619927,
    10619953, 10619971, 10619999, 10620017,
)


class CodecAuthorityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CodecAuthorityError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load(name: str, filename: str) -> Any:
    root = Path(__file__).resolve().parents[1] / "tactic_ramanujan384_adapter_v0"
    manifest = root / "SOURCE_MANIFEST.json"
    path = root / filename
    require(_sha256(manifest.read_bytes()) == V0_MANIFEST_SHA256, "pinned v0 manifest drift")
    require(_sha256(path.read_bytes()) == V0_FILE_SHA256[filename], f"pinned v0 {filename} drift")
    qualified = f"tactic_ramanujan384_authority_v1_{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    require(spec is not None and spec.loader is not None, f"{filename} loader")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(qualified)
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(qualified, None)
        else:
            sys.modules[qualified] = previous
    return module


def modules() -> tuple[Any, Any, Any]:
    return (_load("packet", "packet.py"), _load("codec", "ramanujan_codec.py"),
            _load("container", "container.py"))


def _host(xp: Any, value: Any) -> np.ndarray:
    return np.asarray(xp.asnumpy(value) if hasattr(xp, "asnumpy") else value)


def _pad_role(xp: Any, values: Any, shape: Any) -> Any:
    flat = xp.asarray(values, dtype=xp.float64).reshape(-1)
    require(int(flat.size) == shape.role_values, "exact role value count")
    output = xp.zeros(shape.blocks_per_role * 4096, dtype=xp.float64)
    output[:shape.role_values] = flat
    return output.reshape(shape.blocks_per_role, 4096)


def _candidate_packet(xp: Any, packet: Any, dictionary: Any, gram: Any,
                      correlations: Any, support_order: Any, block: int,
                      rank: int, role: str) -> bytes | None:
    if rank == 0:
        return packet.encode_packet(role, (), (), 0.0)
    selected = support_order[block, :rank]
    matrix = gram[selected[:, None], selected[None, :]]
    rhs = correlations[block, selected]
    diagonal_mean = xp.trace(matrix) / rank
    ridge = xp.maximum(diagonal_mean, 1.0) * (2.0 ** -40)
    matrix = matrix + ridge * xp.eye(rank, dtype=xp.float64)
    coefficients = xp.linalg.solve(matrix, rhs)
    maximum = xp.max(xp.abs(coefficients))
    scale = maximum / packet.COEFFICIENT_MAX
    scale = xp.asarray(scale, dtype=xp.float16).astype(xp.float64)
    scale_value = float(scale.item())
    if not (math.isfinite(scale_value) and scale_value > 0.0):
        return None
    quantized = xp.rint(coefficients / scale)
    quantized = xp.clip(quantized, packet.COEFFICIENT_MIN,
                        packet.COEFFICIENT_MAX).astype(xp.int64)
    quantized_host = _host(xp, quantized).astype(np.int64).tolist()
    selected_host = _host(xp, selected).astype(np.int64).tolist()
    if any(value == 0 for value in quantized_host):
        return None
    pairs = sorted(zip(selected_host, quantized_host, strict=True))
    return packet.encode_packet(
        role,
        tuple(int(pair[0]) for pair in pairs),
        tuple(int(pair[1]) for pair in pairs),
        scale_value,
    )


def prepare_basis(xp: Any) -> dict[str, Any]:
    """Build one public dictionary/Gram object shared by all roles and controls."""

    packet, v0_codec, container = modules()
    basis = v0_codec.build_public_dictionary(xp)
    dictionary = xp.asarray(basis["dictionary"], dtype=xp.float64)
    norms = xp.asarray(basis["norms"], dtype=xp.float64)
    gram = dictionary.T @ dictionary
    return {
        "packet": packet,
        "v0_codec": v0_codec,
        "container": container,
        "basis": basis,
        "dictionary": dictionary,
        "norms": norms,
        "gram": gram,
    }


def encode_role(xp: Any, source_values: Any, coarse_values: Any,
                shape: Any, role: str, prepared: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Search ranks 0..14; literally encode/decode before scoring every candidate."""

    prepared = prepare_basis(xp) if prepared is None else prepared
    packet = prepared["packet"]
    v0_codec = prepared["v0_codec"]
    require(role in ROLE_ORDER, "role")
    source = _pad_role(xp, source_values, shape)
    coarse = _pad_role(xp, coarse_values, shape)
    residual = source - coarse
    basis = prepared["basis"]
    dictionary = prepared["dictionary"]
    norms = prepared["norms"]
    correlations, support_order = v0_codec._candidate_support(xp, residual, dictionary, norms)
    gram = prepared["gram"]
    packets = []
    corrections = xp.zeros_like(residual)
    selected_sse = []
    candidate_replays = []
    ranks = []
    for block in range(shape.blocks_per_role):
        valid = shape.valid_values_for_block(block)
        best_key = None
        best_packet = None
        best_correction = None
        replayed = 0
        for rank in range(packet.MAX_RANK + 1):
            candidate = _candidate_packet(
                xp, packet, dictionary, gram, correlations, support_order, block, rank, role
            )
            if candidate is None:
                continue
            decoded = v0_codec.decode_packets_to_correction(xp, (candidate,), basis, role)[0]
            error = residual[block, :valid] - decoded[:valid]
            sse = float(xp.sum(error * error, dtype=xp.float64).item())
            require(math.isfinite(sse) and sse >= 0.0, "finite packet-replayed candidate SSE")
            key = (sse, candidate)
            replayed += 1
            if best_key is None or key < best_key:
                best_key = key
                best_packet = candidate
                best_correction = decoded
        require(best_packet is not None and best_correction is not None, "rank-zero candidate exists")
        decoded_winner = packet.decode_packet(best_packet)
        packets.append(best_packet)
        corrections[block] = best_correction
        selected_sse.append(float(best_key[0]))
        candidate_replays.append(replayed)
        ranks.append(int(decoded_winner["rank"]))
    stream = b"".join(packets)
    # Independent post-selection stream replay.  This must be byte-for-byte
    # the same reconstruction used below, not retained encoder corrections.
    replay_packets = packet.split_packets(stream)
    replay = v0_codec.decode_packets_to_correction(xp, replay_packets, basis, role)
    require(bool(xp.all(replay == corrections).item()), "winner replay equals candidate replay")
    input_sse = 0.0
    remaining_sse = 0.0
    for block in range(shape.blocks_per_role):
        valid = shape.valid_values_for_block(block)
        row = residual[block, :valid]
        error = row - replay[block, :valid]
        input_sse += float(xp.sum(row * row, dtype=xp.float64).item())
        remaining_sse += float(xp.sum(error * error, dtype=xp.float64).item())
    return {
        "stream": stream,
        "stream_sha256": _sha256(stream),
        "stream_bytes": len(stream),
        "packets": replay_packets,
        "correction": replay,
        "ranks": tuple(ranks),
        "candidate_packet_replays_per_block": tuple(candidate_replays),
        "all_defined_candidates_packet_replayed_before_selection": True,
        "winner_stream_replayed_after_selection": True,
        "tail_padding_scored": False,
        "input_sse": input_sse,
        "remaining_sse": remaining_sse,
    }


def split_fine_by_role(fine_payload: bytes, shape: Any) -> tuple[bytes, bytes, bytes]:
    span = shape.blocks_per_role * 48
    require(len(fine_payload) == 3 * span, "literal fine payload length")
    return tuple(fine_payload[index * span:(index + 1) * span] for index in range(3))  # type: ignore[return-value]


def decode_literal_composite(
    xp: Any,
    *,
    composite: bytes,
    shape: Any,
    coarse_decoder: Any,
    expected_coarse_f32_sha256: Mapping[str, str],
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode coarse bytes and fine bytes into literal weight arrays."""

    prepared = prepare_basis(xp) if prepared is None else prepared
    packet = prepared["packet"]
    v0_codec = prepared["v0_codec"]
    container = prepared["container"]
    decoded = container.decode_composite(composite)
    header = decoded["header"]
    require((header["intermediate"], header["hidden"])
            == (shape.intermediate, shape.hidden), "decoded shape contract")
    require(tuple(header["block_counts"]) == (shape.blocks_per_role,) * 3,
            "decoded block-count contract")
    require(len(decoded["coarse_payload"]) == shape.coarse_bytes,
            "decoded coarse integrality contract")
    require(callable(coarse_decoder), "explicit pinned coarse decoder capability")
    coarse_rows = coarse_decoder(
        decoded["coarse_payload"], shape.intermediate, shape.hidden, ROLE_ORDER
    )
    require(isinstance(coarse_rows, Mapping) and set(coarse_rows) == set(ROLE_ORDER),
            "coarse decoder role mapping")
    fine_streams = split_fine_by_role(decoded["fine_payload"], shape)
    basis = prepared["basis"]
    reconstructions = {}
    coarse_hashes = {}
    for role, fine in zip(ROLE_ORDER, fine_streams, strict=True):
        host_coarse = np.ascontiguousarray(coarse_rows[role], dtype="<f4").reshape(-1)
        require(host_coarse.size == shape.role_values, "coarse decoder exact role geometry")
        digest = _sha256(host_coarse.tobytes(order="C"))
        require(digest == expected_coarse_f32_sha256[role],
                "literal coarse decode independent reconstruction hash")
        coarse_hashes[role] = digest
        coarse = xp.asarray(host_coarse.astype(np.float64)).reshape(-1)
        packets = packet.split_packets(fine)
        correction = v0_codec.decode_packets_to_correction(xp, packets, basis, role).reshape(-1)
        reconstruction = coarse + correction[:shape.role_values]
        reconstructions[role] = reconstruction.reshape(shape.intermediate, shape.hidden)
    return {
        "reconstructions": reconstructions,
        "coarse_f32_sha256": coarse_hashes,
        "literal_coarse_payload_decoded": True,
        "literal_fine_payload_decoded": True,
        "tail_padding_materialized_as_weights": False,
        "decoded_container": decoded,
    }


def independent_score(xp: Any, source_rows: Mapping[str, Any],
                      reconstruction_rows: Mapping[str, Any], physical_rate: float) -> dict[str, float]:
    source_energy = 0.0
    remaining_sse = 0.0
    for role in ROLE_ORDER:
        source = xp.asarray(source_rows[role], dtype=xp.float64)
        reconstruction = xp.asarray(reconstruction_rows[role], dtype=xp.float64)
        require(source.shape == reconstruction.shape, "independent replay score geometry")
        source_energy += float(xp.sum(source * source, dtype=xp.float64).item())
        error = source - reconstruction
        remaining_sse += float(xp.sum(error * error, dtype=xp.float64).item())
    require(source_energy > 0.0 and remaining_sse > 0.0, "independent replay score energies")
    relative_mse = remaining_sse / source_energy
    return {
        "source_energy": source_energy,
        "remaining_sse": remaining_sse,
        "relative_mse": relative_mse,
        "F": relative_mse * 2.0 ** (2.0 * physical_rate),
    }
