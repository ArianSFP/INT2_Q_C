#!/usr/bin/env python3
"""Atomic-snapshot adapter preserving scalable-v2 search and controls."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tactic_ramanujan384_v3_atomic_worker as coarse_worker_api


ROLE_ORDER = ("gate", "up", "down_transposed")


class AtomicAdapterError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AtomicAdapterError(message)


def _host(xp: Any, value: Any) -> np.ndarray:
    return np.asarray(xp.asnumpy(value) if hasattr(xp, "asnumpy") else value)


def _f64_bytes(xp: Any, value: Any) -> bytes:
    return np.ascontiguousarray(_host(xp, value), dtype="<f8").tobytes(order="C")


def _write_new(path: Path, payload: bytes) -> None:
    parent = path.parent.resolve(strict=True)
    require(parent == path.parent.absolute(), "canonical output parent")
    descriptor = os.open(
        os.fspath(path.absolute()),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600,
    )
    try:
        cursor = 0
        while cursor < len(payload):
            written = os.write(descriptor, payload[cursor:])
            require(written > 0, "short composite write")
            cursor += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _gain(input_sse: float, remaining_sse: float) -> float:
    require(math.isfinite(input_sse) and input_sse > 0.0
            and math.isfinite(remaining_sse) and remaining_sse > 0.0,
            "gain energies")
    return -0.5 * math.log2(remaining_sse / input_sse)


def _blocks(xp: Any, values: Any, shape: Any) -> Any:
    flat = xp.asarray(values, dtype=xp.float64).reshape(-1)
    require(int(flat.size) == shape.role_values, "role values")
    output = xp.zeros(shape.blocks_per_role * 4096, dtype=xp.float64)
    output[:shape.role_values] = flat
    return output.reshape(shape.blocks_per_role, 4096)


def _score(xp: Any, source_rows: Mapping[str, Any],
           reconstruction_rows: Mapping[str, Any], physical_rate: float) -> dict[str, float]:
    energies = []
    errors = []
    for role in ROLE_ORDER:
        source = xp.asarray(source_rows[role], dtype=xp.float64)
        reconstruction = xp.asarray(reconstruction_rows[role], dtype=xp.float64)
        require(source.shape == reconstruction.shape, "score geometry")
        energies.append(xp.sum(source * source, dtype=xp.float64))
        delta = source - reconstruction
        errors.append(xp.sum(delta * delta, dtype=xp.float64))
    summary = _host(xp, xp.stack((xp.sum(xp.stack(energies)),
                                  xp.sum(xp.stack(errors)))))
    source_energy, remaining = float(summary[0]), float(summary[1])
    require(source_energy > 0.0 and remaining > 0.0, "score energies")
    relative = remaining / source_energy
    return {"source_energy": source_energy, "remaining_sse": remaining,
            "relative_mse": relative, "F": relative * 2.0 ** (2.0 * physical_rate)}


def _phase_control_host(reference: np.ndarray, seed: int,
                        valid_counts: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(reference)
    for block, valid in enumerate(valid_counts):
        prefix = seed.to_bytes(8, "little") + block.to_bytes(8, "little")
        keys = [(hashlib.sha256(prefix + coordinate.to_bytes(4, "little")).digest(),
                 coordinate) for coordinate in range(valid)]
        keys.sort()
        output[block, :valid] = reference[block, [coordinate for _, coordinate in keys]]
    return output


def _run_controls(xp: Any, core: Any, authenticated: Sequence[Mapping[str, Any]],
                  shape: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    valid_counts = tuple(shape.valid_values_for_block(block)
                         for block in range(shape.blocks_per_role))
    residuals = [_blocks(xp, row["source"] - row["coarse"], shape)
                 for row in authenticated]
    phase_input = 0.0
    phase_remaining = 0.0
    for role_index, (role, residual) in enumerate(zip(ROLE_ORDER, residuals, strict=True)):
        host = np.ascontiguousarray(_host(xp, residual), dtype="<f8")
        controlled = _phase_control_host(host, 0xA17C9E35 + role_index, valid_counts)
        row = core.encode_role_batched(
            xp, controlled.reshape(-1)[:shape.role_values],
            xp.zeros(shape.role_values, dtype=xp.float64), shape, role, prepared,
        )
        phase_input += row["input_sse"]
        phase_remaining += row["remaining_sse"]
    phase_gain = _gain(phase_input, phase_remaining)
    gaussian_rows = []
    for seed in core.GAUSSIAN_SEEDS:
        input_sse = 0.0
        remaining_sse = 0.0
        hashes = []
        streams = []
        for role, residual in zip(ROLE_ORDER, residuals, strict=True):
            controlled, record = core.moment_matched_gaussian(
                xp, residual, seed, valid_counts
            )
            hashes.append(record["f64_sha256"])
            encoded = core.encode_role_batched(
                xp, controlled.reshape(-1)[:shape.role_values],
                xp.zeros(shape.role_values, dtype=xp.float64), shape, role, prepared,
            )
            input_sse += encoded["input_sse"]
            remaining_sse += encoded["remaining_sse"]
            streams.append(encoded["stream_sha256"])
        gaussian_rows.append({
            "seed": seed, "input_sse": input_sse, "remaining_sse": remaining_sse,
            "gain_bpw": _gain(input_sse, remaining_sse),
            "canonical_gaussian_f64_sha256_by_role": hashes,
            "fine_stream_sha256_by_role": streams,
            "backend_random_or_transcendental_calls": 0,
            "batched_all_rank_search": True,
        })
    return {
        "phase_control": {"gain_bpw": phase_gain, "input_sse": phase_input,
                          "remaining_sse": phase_remaining},
        "gaussian_controls": gaussian_rows,
        "strongest_control_gain_bpw": max(
            [phase_gain] + [row["gain_bpw"] for row in gaussian_rows]
        ),
    }


def run_authenticated_expert(
    xp: Any, *, core: Any, io: Any,
    role_inputs: Sequence[Mapping[str, Any]],
    coarse_worker_arguments: Mapping[str, Any], composite_output_path: Path,
) -> dict[str, Any]:
    """Execute a literal expert; no decoder object or callback is accepted."""
    require(len(role_inputs) == 3, "three role inputs")
    require("decoder" not in coarse_worker_arguments
            and "decode_literal" not in coarse_worker_arguments,
            "no live decoder capability")
    authenticated = [io.authenticate_role(**dict(row)) for row in role_inputs]
    require(tuple(row["role"] for row in authenticated) == ROLE_ORDER,
            "canonical role order")
    require(all(row["shape"] == authenticated[0]["shape"] for row in authenticated),
            "one shape")
    shape = core.define_shape(*authenticated[0]["shape"])
    coarse = authenticated[0]["coarse_artifact_payload"]
    require(len(coarse) == shape.coarse_bytes
            and all(row["coarse_artifact_sha256"]
                    == authenticated[0]["coarse_artifact_sha256"]
                    for row in authenticated), "one exact coarse payload")
    prepared = core.prepare_dictionary(xp)
    encoded_rows = []
    encoder_reconstructions = {}
    fine_streams = []
    input_sse = 0.0
    for role, source_row in zip(ROLE_ORDER, authenticated, strict=True):
        encoded = core.encode_role_batched(
            xp, source_row["source"], source_row["coarse"], shape, role, prepared
        )
        encoded_rows.append(encoded)
        fine_streams.append(encoded["stream"])
        input_sse += encoded["input_sse"]
        correction = encoded["correction"].reshape(-1)[:shape.role_values]
        encoder_reconstructions[role] = (
            xp.asarray(source_row["coarse"], dtype=xp.float64).reshape(-1) + correction
        ).reshape(shape.intermediate, shape.hidden)
    binding = hashlib.sha256(b"".join(
        bytes.fromhex(row["binding_sha256"]) for row in authenticated
    )).hexdigest()
    composite = core.encode_composite(shape, coarse, tuple(fine_streams), binding)
    require(not composite_output_path.exists(), "new composite output")
    _write_new(composite_output_path, composite)
    replay_bytes, read_trace = core.read_composite_once(composite_output_path, len(composite))
    decoded = core.decode_composite(replay_bytes)
    worker_arguments = dict(coarse_worker_arguments)
    worker_arguments.update({
        "coarse_payload": decoded["coarse"], "intermediate": shape.intermediate,
        "hidden": shape.hidden, "role_order": ROLE_ORDER,
    })
    worker = coarse_worker_api.authenticate_and_decode(**worker_arguments)
    require(worker["mutable_decoder_object_used"] is False
            and worker["zero_import_no_path_byte_worker"] is True,
            "isolated coarse byte worker")
    decoded_coarse = worker["coarse_f32_bytes"]
    require(isinstance(decoded_coarse, Mapping) and set(decoded_coarse) == set(ROLE_ORDER),
            "coarse worker role map")
    span = shape.blocks_per_role * core.PACKET_BYTES
    reconstruction_rows = {}
    for role_index, (role, source_row) in enumerate(zip(ROLE_ORDER, authenticated, strict=True)):
        payload = decoded_coarse[role]
        require(type(payload) is bytes
                and hashlib.sha256(payload).hexdigest()
                == source_row["coarse_reconstruction_sha256"],
                "independently recorded literal coarse decode")
        coarse_f32 = np.frombuffer(payload, dtype="<f4")
        require(coarse_f32.size == shape.role_values, "coarse worker geometry")
        fine = decoded["fine"][role_index * span:(role_index + 1) * span]
        correction = core.decode_fine_role(xp, fine, shape, role, prepared)
        reconstruction = (
            xp.asarray(coarse_f32.astype(np.float64), dtype=xp.float64)
            + correction.reshape(-1)[:shape.role_values]
        ).reshape(shape.intermediate, shape.hidden)
        require(_f64_bytes(xp, reconstruction)
                == _f64_bytes(xp, encoder_reconstructions[role]),
                "literal replay equals batched encoder state")
        reconstruction_rows[role] = reconstruction
    physical_rate = 8.0 * len(composite) / shape.total_values
    source_rows = {row["role"]: row["source"] for row in authenticated}
    score = _score(xp, source_rows, reconstruction_rows, physical_rate)
    ledger = core.physical_ledger(shape)
    require(ledger["physical_bytes"] == len(composite), "physical byte ledger")
    source_gain = _gain(input_sse, score["remaining_sse"])
    result = {
        "schema": "tactic-ramanujan384-atomic-authority-result-v3",
        "status": None,
        "shape": {"intermediate": shape.intermediate, "hidden": shape.hidden,
                  "blocks_per_role": shape.blocks_per_role,
                  "tail_values_per_role": shape.tail_values_per_role},
        "weights": shape.total_values, "physical_bytes": len(composite),
        "physical_rate_bpw": physical_rate, "relative_mse": score["relative_mse"],
        "F": score["F"], "remaining_sse": score["remaining_sse"],
        "source_energy": score["source_energy"], "source_gain_bpw": source_gain,
        "composite_sha256": hashlib.sha256(composite).hexdigest(),
        "literal_coarse_and_fine_weight_replay": True,
        "independent_fp64_rescore": True,
        "zero_import_no_path_coarse_worker": True,
        "mutable_decoder_object_used": False,
        "coarse_worker_capability_sha256": worker["capability_sha256"],
        "all_rank_search_batched": all(row["candidate_ranks_batched"] == 15
                                       for row in encoded_rows),
        "per_candidate_host_scalar_syncs": max(row["per_candidate_host_scalar_syncs"]
                                                for row in encoded_rows),
        "per_candidate_solves": max(row["per_candidate_solves"] for row in encoded_rows),
        "per_candidate_matmuls": max(row["per_candidate_matmuls"] for row in encoded_rows),
        "read_trace": read_trace, "controls_rerun": False,
        "qwen_payload_accessed": False,
    }
    if not 2.15 <= physical_rate <= 2.5:
        result.update({"status": "HARD_KILL_PHYSICAL_RATE", "controls_permitted": False})
        return result
    if score["relative_mse"] > core.TARGET_D + 1e-15:
        result.update({"status": "HARD_KILL_ABSOLUTE_D", "controls_permitted": False})
        return result
    controls = _run_controls(xp, core, authenticated, shape, prepared)
    excess = source_gain - controls["strongest_control_gain_bpw"]
    result.update(controls)
    result.update({
        "controls_permitted": True, "controls_rerun": True,
        "source_minus_strongest_control_bpw": excess,
        "status": ("ELIGIBLE_FOR_INDEPENDENT_PAYLOAD_PILOT_AUDIT"
                   if excess + 1e-15 >= core.MIN_CONTROL_EXCESS_BPW
                   else "HARD_KILL_SOURCE_NOT_SPECIFIC"),
    })
    return result
