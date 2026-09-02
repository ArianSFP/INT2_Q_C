#!/usr/bin/env python3
"""Finite CuPy/NumPy Ramanujan-polyphase refinement mechanics."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


BLOCK_VALUES = 4096
DICTIONARY_COLUMNS = 384
MAX_RANK = 14
TARGET_D = 0.025
COARSE_RELATIVE_MSE = 0.036975150060595235
MIN_CONTROL_EXCESS_BPW = 0.03
PHASE_SEED = 10619863
GAUSSIAN_SEEDS = (
    10619863, 10619881, 10619909, 10619927,
    10619953, 10619971, 10619999, 10620017,
)
DEPENDENCY_MANIFEST_SHA256 = "4259e8e8dc87b4c25301ca89ade7dbd63c1e0c9e3415fdaa4d7881d7d10ccc06"
DEPENDENCY_ORACLES_SHA256 = "f990aedf8eba0e9058bd9c77caaa05df98226b2855486bace0eaee15cfac806f"


class CodecError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CodecError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_audited_parent() -> Any:
    root = Path(__file__).resolve().parents[1] / "mosaic_secondary_oracles_v0"
    manifest_path = root / "SOURCE_MANIFEST.json"
    oracle_path = root / "residual_oracles.py"
    require(manifest_path.is_file() and oracle_path.is_file(), "audited parent dependency missing")
    require(_sha256(manifest_path.read_bytes()) == DEPENDENCY_MANIFEST_SHA256,
            "audited parent manifest drift")
    payload = oracle_path.read_bytes()
    require(_sha256(payload) == DEPENDENCY_ORACLES_SHA256, "audited parent oracle drift")
    spec = importlib.util.spec_from_file_location("tactic_ramanujan384_audited_parent", oracle_path)
    require(spec is not None and spec.loader is not None, "audited parent import")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def non_dyadic_periods() -> tuple[int, ...]:
    result = tuple(period for period in range(3, 128) if period & (period - 1))
    require(len(result) == 120 and result[0] == 3 and result[-1] == 127, "period bank")
    return result


def totient(value: int) -> int:
    require(type(value) is int and value >= 1, "totient input")
    result = value
    remainder = value
    factor = 2
    while factor * factor <= remainder:
        if remainder % factor == 0:
            while remainder % factor == 0:
                remainder //= factor
            result -= result // factor
        factor += 1
    if remainder > 1:
        result -= result // remainder
    return result


def mobius(value: int) -> int:
    require(type(value) is int and value >= 1, "mobius input")
    remainder = value
    factors = 0
    factor = 2
    while factor * factor <= remainder:
        if remainder % factor == 0:
            remainder //= factor
            factors += 1
            if remainder % factor == 0:
                return 0
            while remainder % factor == 0:
                remainder //= factor
        factor += 1
    if remainder > 1:
        factors += 1
    return -1 if factors & 1 else 1


def ramanujan_sum(period: int, coordinate: int) -> int:
    require(type(period) is int and period >= 1 and type(coordinate) is int, "Ramanujan sum input")
    quotient = period // math.gcd(period, coordinate)
    numerator = mobius(quotient) * totient(period)
    denominator = totient(quotient)
    require(numerator % denominator == 0, "integer Ramanujan sum")
    return numerator // denominator


def period_bank_labels(columns: int = DICTIONARY_COLUMNS) -> tuple[tuple[int, int], ...]:
    require(type(columns) is int and 120 <= columns <= 512, "dictionary columns")
    periods = non_dyadic_periods()
    parent = load_audited_parent()
    dimensions = {}
    for period in periods:
        rows = parent.primitive_real_frequencies(period)
        dimensions[period] = len(rows)
        require(len(rows) == totient(period), "audited exact-period dimension")
    labels = []
    shift = 0
    while len(labels) < columns:
        before = len(labels)
        for period in periods:
            if shift < dimensions[period]:
                labels.append((period, shift))
                if len(labels) == columns:
                    break
        require(len(labels) > before, "period bank exhaustion")
        shift += 1
    require({period for period, _ in labels} == set(periods), "every period represented")
    return tuple(labels)


def build_public_dictionary(xp: Any, length: int = BLOCK_VALUES, columns: int = DICTIONARY_COLUMNS) -> dict[str, Any]:
    require(type(length) is int and length >= 256, "dictionary length")
    labels = period_bank_labels(columns)
    coordinate = xp.arange(length, dtype=xp.int64)
    atoms = []
    for period, shift in labels:
        lookup = xp.asarray([ramanujan_sum(period, index) for index in range(period)], dtype=xp.float64)
        atoms.append(lookup[(coordinate - shift) % period])
    dictionary = xp.stack(atoms, axis=1).astype(xp.float64, copy=False)
    norms = xp.sum(dictionary * dictionary, axis=0, dtype=xp.float64)
    require(bool(xp.all(norms > 0.0).item()), "nonzero public atoms")
    return {
        "dictionary": dictionary,
        "norms": norms,
        "labels": labels,
        "length": length,
        "columns": columns,
        "every_period_represented": True,
        "source_independent": True,
        "integer_decoder_atoms": True,
    }


def _as_host(xp: Any, value: Any) -> Any:
    return xp.asnumpy(value) if hasattr(xp, "asnumpy") else value


def _load_packet_module() -> Any:
    path = Path(__file__).resolve().parent / "packet.py"
    spec = importlib.util.spec_from_file_location("tactic_ramanujan384_packet_runtime", path)
    require(spec is not None and spec.loader is not None, "packet import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_support(xp: Any, residual: Any, dictionary: Any, norms: Any) -> tuple[Any, Any]:
    correlations = residual @ dictionary
    scores = xp.abs(correlations) / xp.sqrt(norms)[None, :]
    indices = xp.broadcast_to(xp.arange(dictionary.shape[1], dtype=xp.int64)[None, :], scores.shape)
    try:
        order = xp.lexsort((indices, -scores), axis=1)
    except (TypeError, NotImplementedError):
        # A backend without axis-aware lexsort takes this small ordering-only
        # matrix to NumPy; all correlations, solves and source scoring remain
        # on the selected CuPy device.  NumPy lexsort gives an exact index tie
        # break without perturbing a physical correlation value.
        import numpy as np
        host_scores = np.asarray(_as_host(xp, scores), dtype=np.float64)
        host_indices = np.broadcast_to(np.arange(dictionary.shape[1], dtype=np.int64), host_scores.shape)
        order = xp.asarray(np.lexsort((host_indices, -host_scores), axis=1), dtype=xp.int64)
    return correlations, order[:, :MAX_RANK]


def encode_residual_blocks(xp: Any, residual_blocks: Any, basis: Mapping[str, Any], role: str) -> dict[str, Any]:
    packet = _load_packet_module()
    require(role in packet.ROLES, "normalized SwiGLU role")
    residual = xp.asarray(residual_blocks, dtype=xp.float64)
    dictionary = xp.asarray(basis["dictionary"], dtype=xp.float64)
    norms = xp.asarray(basis["norms"], dtype=xp.float64)
    require(residual.ndim == 2 and residual.shape[1] == dictionary.shape[0], "residual block geometry")
    blocks = int(residual.shape[0])
    require(blocks > 0 and dictionary.shape[1] == DICTIONARY_COLUMNS, "finite panel geometry")
    correlations, support_order = _candidate_support(xp, residual, dictionary, norms)
    gram = dictionary.T @ dictionary
    best_sse = xp.sum(residual * residual, axis=1, dtype=xp.float64)
    best_rank = xp.zeros(blocks, dtype=xp.int64)
    best_support = xp.zeros((blocks, MAX_RANK), dtype=xp.int64)
    best_q = xp.zeros((blocks, MAX_RANK), dtype=xp.int64)
    best_scale = xp.zeros(blocks, dtype=xp.float64)
    for rank in range(1, MAX_RANK + 1):
        selected = support_order[:, :rank]
        matrices = gram[selected[:, :, None], selected[:, None, :]]
        rhs = xp.take_along_axis(correlations, selected, axis=1)
        diagonal_mean = xp.trace(matrices, axis1=1, axis2=2) / rank
        ridge = xp.maximum(diagonal_mean, 1.0) * (2.0 ** -40)
        matrices = matrices + ridge[:, None, None] * xp.eye(rank, dtype=xp.float64)[None, :, :]
        coefficients = xp.linalg.solve(matrices, rhs[..., None])[..., 0]
        max_abs = xp.max(xp.abs(coefficients), axis=1)
        scale = (max_abs / packet.COEFFICIENT_MAX).astype(xp.float16).astype(xp.float64)
        safe_scale = xp.where(scale > 0.0, scale, 1.0)
        quantized = xp.rint(coefficients / safe_scale[:, None])
        quantized = xp.clip(quantized, packet.COEFFICIENT_MIN, packet.COEFFICIENT_MAX).astype(xp.int64)
        valid = (scale > 0.0) & xp.all(quantized != 0, axis=1)
        dequantized = quantized.astype(xp.float64) * safe_scale[:, None]
        atoms = xp.transpose(xp.take(dictionary, selected, axis=1), (1, 0, 2))
        correction = xp.sum(atoms * dequantized[:, None, :], axis=2, dtype=xp.float64)
        remaining = xp.sum((residual - correction) ** 2, axis=1, dtype=xp.float64)
        better = valid & (remaining < best_sse)
        best_sse = xp.where(better, remaining, best_sse)
        best_rank = xp.where(better, rank, best_rank)
        best_scale = xp.where(better, scale, best_scale)
        prefix_support = xp.zeros_like(best_support)
        prefix_q = xp.zeros_like(best_q)
        prefix_support[:, :rank] = selected
        prefix_q[:, :rank] = quantized
        best_support = xp.where(better[:, None], prefix_support, best_support)
        best_q = xp.where(better[:, None], prefix_q, best_q)

    ranks = _as_host(xp, best_rank).tolist()
    supports = _as_host(xp, best_support).tolist()
    quantized_rows = _as_host(xp, best_q).tolist()
    scales = _as_host(xp, best_scale).tolist()
    packets = []
    for rank, support_row, q_row, scale in zip(ranks, supports, quantized_rows, scales, strict=True):
        pairs = sorted((int(support_row[index]), int(q_row[index])) for index in range(int(rank)))
        packets.append(packet.encode_packet(
            role,
            tuple(item[0] for item in pairs),
            tuple(item[1] for item in pairs),
            0.0 if int(rank) == 0 else float(scale),
        ))
    decoded = decode_packets_to_correction(xp, packets, basis, role)
    exact_remaining = xp.sum((residual - decoded) ** 2, axis=1, dtype=xp.float64)
    require(bool(xp.all(xp.isfinite(exact_remaining)).item()), "finite exact remaining SSE")
    stream = b"".join(packets)
    return {
        "packets": tuple(packets),
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "stream_bytes": len(stream),
        "ranks": tuple(int(value) for value in ranks),
        "input_sse": float(xp.sum(residual * residual, dtype=xp.float64).item()),
        "remaining_sse": float(xp.sum(exact_remaining, dtype=xp.float64).item()),
        "correction": decoded,
        "exact_original_domain_scoring": True,
        "literal_bits_per_block": packet.PACKET_BYTES * 8,
    }


def decode_packets_to_correction(xp: Any, packets: Sequence[bytes], basis: Mapping[str, Any], role: str) -> Any:
    packet = _load_packet_module()
    require(role in packet.ROLES and bool(packets), "decode role and packets")
    dictionary = xp.asarray(basis["dictionary"], dtype=xp.float64)
    result = xp.zeros((len(packets), dictionary.shape[0]), dtype=xp.float64)
    for block, payload in enumerate(packets):
        row = packet.decode_packet(payload)
        require(row["role"] == role, "packet stream role")
        if row["rank"]:
            support = xp.asarray(row["support"], dtype=xp.int64)
            coefficients = xp.asarray(row["coefficients"], dtype=xp.float64) * float(row["scale"])
            result[block] = dictionary[:, support] @ coefficients
    return result


def _gain(input_sse: float, remaining_sse: float) -> float:
    require(math.isfinite(input_sse) and input_sse > 0.0 and math.isfinite(remaining_sse) and remaining_sse > 0.0,
            "gain energies")
    return -0.5 * math.log2(remaining_sse / input_sse)


def run_finite_panel(
    xp: Any,
    source_blocks: Any,
    coarse_blocks: Any,
    *,
    role: str,
    source_energy: float,
) -> dict[str, Any]:
    parent = load_audited_parent()
    source = xp.asarray(source_blocks, dtype=xp.float64)
    coarse = xp.asarray(coarse_blocks, dtype=xp.float64)
    require(source.shape == coarse.shape and source.ndim == 2 and source.shape[1] == BLOCK_VALUES,
            "source/coarse blocks")
    require(math.isfinite(source_energy) and source_energy > 0.0, "source energy")
    residual = source - coarse
    basis = build_public_dictionary(xp)
    encoded = encode_residual_blocks(xp, residual, basis, role)
    reconstruction = coarse + encoded["correction"]
    source_remaining = float(xp.sum((source - reconstruction) ** 2, dtype=xp.float64).item())
    require(abs(source_remaining - encoded["remaining_sse"]) <= 2e-12 * max(1.0, source_remaining),
            "original-domain residual identity")
    relative_mse = source_remaining / source_energy
    result = {
        "schema": "tactic-ramanujan384-finite-panel-v0",
        "role": role,
        "blocks": int(source.shape[0]),
        "values": int(source.size),
        "input_sse": encoded["input_sse"],
        "source_energy": source_energy,
        "source_remaining_sse": source_remaining,
        "relative_mse": relative_mse,
        "source_capture": 1.0 - source_remaining / encoded["input_sse"],
        "source_gain_bpw": _gain(encoded["input_sse"], source_remaining),
        "fine_stream_bytes": encoded["stream_bytes"],
        "fine_stream_sha256": encoded["stream_sha256"],
        "rank_histogram": {str(rank): encoded["ranks"].count(rank) for rank in range(MAX_RANK + 1)},
        "controls_rerun": False,
        "exact_original_domain_fp64_score": True,
        "packets": encoded["packets"],
    }
    if relative_mse > TARGET_D + 1e-15:
        result["status"] = "HARD_KILL_ABSOLUTE_SOURCE_MISSES_D_0P025"
        result["controls_permitted"] = False
        return result

    permutation = parent.phase_destroyed_blocks(xp, residual, PHASE_SEED)
    phase_result = encode_residual_blocks(xp, permutation, basis, role)
    gaussian_rows = []
    for seed in GAUSSIAN_SEEDS:
        control = parent.moment_matched_gaussian_blocks(xp, residual, seed)
        row = encode_residual_blocks(xp, control, basis, role)
        gaussian_rows.append({
            "seed": seed,
            "input_sse": row["input_sse"],
            "remaining_sse": row["remaining_sse"],
            "gain_bpw": _gain(row["input_sse"], row["remaining_sse"]),
            "complete_finite_search_rerun": True,
        })
    phase_gain = _gain(phase_result["input_sse"], phase_result["remaining_sse"])
    strongest = max([phase_gain] + [row["gain_bpw"] for row in gaussian_rows])
    excess = result["source_gain_bpw"] - strongest
    result.update({
        "controls_permitted": True,
        "controls_rerun": True,
        "phase_control": {
            "seed": PHASE_SEED,
            "input_sse": phase_result["input_sse"],
            "remaining_sse": phase_result["remaining_sse"],
            "gain_bpw": phase_gain,
            "complete_finite_search_rerun": True,
        },
        "gaussian_controls": gaussian_rows,
        "source_minus_strongest_control_bpw": excess,
        "status": (
            "ELIGIBLE_FOR_INDEPENDENT_PAYLOAD_PILOT_AUDIT"
            if excess + 1e-15 >= MIN_CONTROL_EXCESS_BPW
            else "HARD_KILL_SOURCE_NOT_SPECIFIC_0P03_BPW"
        ),
    })
    return result


def expert_read_ledger(
    *,
    role_weights: Sequence[int],
    coarse_artifact_bytes: int,
    container_header_bytes: int = 512,
    page_bytes: int = 4096,
) -> dict[str, Any]:
    require(len(role_weights) == 3 and all(type(value) is int and value > 0 for value in role_weights),
            "three universal role sizes")
    require(role_weights[0] == role_weights[1] == role_weights[2],
            "normalized SwiGLU roles must share intermediate-by-hidden geometry")
    require(type(coarse_artifact_bytes) is int and coarse_artifact_bytes > 0, "coarse bytes")
    require(container_header_bytes == 512, "canonical container header bytes")
    weights = sum(role_weights)
    blocks = sum((value + BLOCK_VALUES - 1) // BLOCK_VALUES for value in role_weights)
    fine_bytes = 48 * blocks
    unpadded = container_header_bytes + coarse_artifact_bytes + fine_bytes
    physical = ((unpadded + page_bytes - 1) // page_bytes) * page_bytes
    expected_coarse = 307 * weights / 1024.0
    rate = 8.0 * physical / weights
    tail_free = all(value % BLOCK_VALUES == 0 for value in role_weights)
    exact_coarse = float(coarse_artifact_bytes) == expected_coarse
    return {
        "weights": weights,
        "role_weights": list(role_weights),
        "blocks": blocks,
        "container_header_bytes": container_header_bytes,
        "coarse_artifact_bytes": coarse_artifact_bytes,
        "fine_bytes": fine_bytes,
        "coarse_plus_fine_rate_bpw": 8.0 * (coarse_artifact_bytes + fine_bytes) / weights,
        "page_padding_bytes": physical - unpadded,
        "physical_bytes": physical,
        "physical_rate_bpw": rate,
        "tail_free": tail_free,
        "coarse_is_exact_307_over_128_bpw": exact_coarse,
        "target_rate_eligible": tail_free and exact_coarse and 2.15 <= rate <= 2.5,
        "expert_local_contiguous_packet": True,
        "external_storage_passes": 1,
        "external_storage_refetches": 0,
        "external_storage_read_bytes": physical,
        "external_read_amplification": 1.0,
        "passes_strict_below_2x": True,
        "host_scratch_bytes": None,
        "accelerator_hbm_measured": False,
    }
