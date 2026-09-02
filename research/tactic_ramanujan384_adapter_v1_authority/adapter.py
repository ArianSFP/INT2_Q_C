#!/usr/bin/env python3
"""Source-frozen authority adapter; no model-payload CLI exists in v1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
ROLE_ORDER = ("gate", "up", "down_transposed")


class AdapterError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _load(name: str, filename: str) -> Any:
    path = ROOT / filename
    qualified = f"tactic_ramanujan384_authority_adapter_{name}"
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


def _host_f64_bytes(xp: Any, values: Any) -> bytes:
    host = xp.asnumpy(values) if hasattr(xp, "asnumpy") else values
    return np.ascontiguousarray(host, dtype="<f8").tobytes(order="C")


def _write_new(path: Path, payload: bytes) -> None:
    parent = path.parent.resolve(strict=True)
    require(parent == path.parent.absolute(), "output parent canonical and contains no symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(os.fspath(path.absolute()), flags, 0o600)
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
            "finite gain energies")
    return -0.5 * math.log2(remaining_sse / input_sse)


def _blockify(xp: Any, flat: Any, shape: Any) -> Any:
    values = xp.asarray(flat, dtype=xp.float64).reshape(-1)
    require(int(values.size) == shape.role_values, "control role values")
    padded = xp.zeros(shape.blocks_per_role * 4096, dtype=xp.float64)
    padded[:shape.role_values] = values
    return padded.reshape(shape.blocks_per_role, 4096)


def _control_panel(xp: Any, authenticated: Sequence[Mapping[str, Any]], shape: Any,
                   codec: Any, controls: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    residual_blocks = [
        _blockify(xp, row["source"] - row["coarse"], shape) for row in authenticated
    ]
    valid_counts = tuple(shape.valid_values_for_block(block)
                         for block in range(shape.blocks_per_role))

    # The phase destroyer is a fixed coordinate permutation generated only
    # from public block length.  Sorting SplitMix keys with index tie-break is
    # specified on host and then copied to the selected backend.
    phase_input = 0.0
    phase_remaining = 0.0
    for role_index, (role, blocks) in enumerate(zip(ROLE_ORDER, residual_blocks, strict=True)):
        host = np.asarray(xp.asnumpy(blocks) if hasattr(xp, "asnumpy") else blocks, dtype="<f8")
        phase = np.zeros_like(host)
        for block in range(shape.blocks_per_role):
            valid = valid_counts[block]
            keys = controls.splitmix64_words(0xA17C9E35 + 131 * role_index + block, valid)
            order = np.lexsort((np.arange(valid, dtype=np.int64), keys))
            phase[block, :valid] = host[block, order]
        encoded = codec.encode_role(
            xp, phase.reshape(-1)[:shape.role_values],
            xp.zeros(shape.role_values, dtype=xp.float64), shape, role, prepared
        )
        phase_input += encoded["input_sse"]
        phase_remaining += encoded["remaining_sse"]
    phase_gain = _gain(phase_input, phase_remaining)

    gaussian_rows = []
    for seed in codec.GAUSSIAN_SEEDS:
        control_input = 0.0
        control_remaining = 0.0
        stream_hashes = []
        control_hashes = []
        for role, blocks in zip(ROLE_ORDER, residual_blocks, strict=True):
            generated = controls.moment_matched_blocks(
                xp, blocks, seed, valid_counts=valid_counts
            )
            control_hashes.append(hashlib.sha256(controls.host_bytes(xp, generated)).hexdigest())
            encoded = codec.encode_role(
                xp, generated.reshape(-1)[:shape.role_values],
                xp.zeros(shape.role_values, dtype=xp.float64), shape, role, prepared
            )
            control_input += encoded["input_sse"]
            control_remaining += encoded["remaining_sse"]
            stream_hashes.append(encoded["stream_sha256"])
        gaussian_rows.append({
            "seed": seed,
            "generator": "SplitMix64 plus fixed 12xuint16 Irwin-Hall, host FP64 moment match",
            "input_sse": control_input,
            "remaining_sse": control_remaining,
            "gain_bpw": _gain(control_input, control_remaining),
            "control_f64_sha256_by_role": control_hashes,
            "finite_stream_sha256_by_role": stream_hashes,
            "backend_random_api_used": False,
            "complete_packet_replayed_search": True,
        })
    return {
        "phase_control": {
            "input_sse": phase_input,
            "remaining_sse": phase_remaining,
            "gain_bpw": phase_gain,
            "fixed_integer_prng_permutation": True,
        },
        "gaussian_like_controls": gaussian_rows,
        "strongest_control_gain_bpw": max(
            [phase_gain] + [row["gain_bpw"] for row in gaussian_rows]
        ),
    }


def run_authenticated_expert(
    xp: Any,
    *,
    role_inputs: Sequence[Mapping[str, Any]],
    coarse_decoder: Any,
    composite_output_path: Path,
) -> dict[str, Any]:
    """Emit, file-replay, literally decode, and independently score one object."""

    require(len(role_inputs) == 3, "three role inputs")
    auth = _load("auth", "authenticated_io.py")
    contract = _load("contract", "contract.py")
    codec = _load("codec", "codec_authority.py")
    trace = _load("trace", "read_trace.py")
    controls = _load("controls", "stable_controls.py")
    authenticated = [auth.authenticate_role(**dict(arguments)) for arguments in role_inputs]
    require(tuple(row["role"] for row in authenticated) == ROLE_ORDER,
            "canonical authenticated role order")
    require(all(row["shape"] == authenticated[0]["shape"] for row in authenticated),
            "one universal shape")
    shape = contract.define_shape(*authenticated[0]["shape"])
    coarse_payload = authenticated[0]["coarse_artifact_payload"]
    coarse_digest = hashlib.sha256(coarse_payload).hexdigest()
    require(len(coarse_payload) == shape.coarse_bytes, "exact coarse byte-integrality contract")
    require(all(hashlib.sha256(row["coarse_artifact_payload"]).hexdigest() == coarse_digest
                for row in authenticated), "one shared literal coarse object")

    prepared = codec.prepare_basis(xp)

    encoded_rows = []
    encoder_reconstruction = {}
    input_sse = 0.0
    fine_streams = []
    for role, row in zip(ROLE_ORDER, authenticated, strict=True):
        encoded = codec.encode_role(xp, row["source"], row["coarse"], shape, role, prepared)
        encoded_rows.append(encoded)
        fine_streams.append(encoded["stream"])
        input_sse += encoded["input_sse"]
        coarse_flat = xp.asarray(row["coarse"], dtype=xp.float64).reshape(-1)
        correction_flat = encoded["correction"].reshape(-1)[:shape.role_values]
        encoder_reconstruction[role] = (coarse_flat + correction_flat).reshape(
            shape.intermediate, shape.hidden
        )

    container = prepared["container"]
    binding_sha = hashlib.sha256(b"".join(
        bytes.fromhex(row["binding_sha256"]) for row in authenticated
    )).hexdigest()
    composite = container.encode_composite(
        intermediate=shape.intermediate,
        hidden=shape.hidden,
        coarse_payload=coarse_payload,
        role_fine_streams=tuple(fine_streams),
        source_binding_sha256=binding_sha,
    )
    require(not composite_output_path.exists(), "composite output must not pre-exist")
    _write_new(composite_output_path, composite)
    replay_bytes, read_trace = trace.read_once(composite_output_path, len(composite))
    require(replay_bytes == composite, "instrumented file replay equals emitted object")

    replay = codec.decode_literal_composite(
        xp,
        composite=replay_bytes,
        shape=shape,
        coarse_decoder=coarse_decoder,
        expected_coarse_f32_sha256={
            row["role"]: row["coarse_reconstruction_sha256"] for row in authenticated
        },
        prepared=prepared,
    )
    for role in ROLE_ORDER:
        require(_host_f64_bytes(xp, replay["reconstructions"][role])
                == _host_f64_bytes(xp, encoder_reconstruction[role]),
                "literal composite reconstruction equals packet-replayed encoder reconstruction")
    physical_rate = 8.0 * len(replay_bytes) / shape.total_values
    source_rows = {row["role"]: row["source"] for row in authenticated}
    score = codec.independent_score(
        xp, source_rows, replay["reconstructions"], physical_rate
    )
    ledger = contract.physical_ledger(shape)
    require(ledger["physical_bytes"] == len(composite)
            and abs(float(ledger["physical_rate_bpw"]) - physical_rate) <= 1e-15,
            "literal physical ledger")
    source_gain = _gain(input_sse, score["remaining_sse"])
    result = {
        "schema": "tactic-ramanujan384-authority-result-v1",
        "status": None,
        "shape": {
            "intermediate": shape.intermediate,
            "hidden": shape.hidden,
            "roles": list(ROLE_ORDER),
            "tail_values_per_role": shape.tail_values_per_role,
            "last_block_valid_values": shape.last_block_valid_values,
        },
        "weights": shape.total_values,
        "physical_rate_bpw": physical_rate,
        "physical_bytes": len(composite),
        "composite_sha256": hashlib.sha256(composite).hexdigest(),
        "source_energy": score["source_energy"],
        "input_sse": input_sse,
        "remaining_sse": score["remaining_sse"],
        "relative_mse": score["relative_mse"],
        "F": score["F"],
        "source_gain_bpw": source_gain,
        "literal_composite_reconstructed_to_weights": True,
        "independent_source_domain_fp64_rescore": True,
        "every_defined_candidate_packet_replayed_before_selection": all(
            row["all_defined_candidates_packet_replayed_before_selection"] for row in encoded_rows
        ),
        "winner_stream_replayed_after_selection": True,
        "tail_padding_scored": False,
        "actual_input_manifests_opened": all(
            row["actual_input_manifest_opened"] for row in authenticated
        ),
        "actual_auditor_manifests_opened": all(
            row["actual_auditor_source_manifest_opened"] for row in authenticated
        ),
        "read_trace": read_trace,
        "layout": {
            "contiguous_minimal_page_aligned_object": True,
            "layout_read_amplification_upper_bound": 1.0,
            "layout_bound_is_not_a_measurement": True,
        },
        "accelerator_hbm_measured": False,
        "controls_rerun": False,
        "qwen_or_model_identity_used": False,
    }
    if not (2.15 <= physical_rate <= 2.5):
        result["status"] = "HARD_KILL_PHYSICAL_RATE_OUTSIDE_2P15_TO_2P5"
        result["controls_permitted"] = False
        return result
    if score["relative_mse"] > codec.TARGET_D + 1e-15:
        result["status"] = "HARD_KILL_ABSOLUTE_SOURCE_MISSES_D_0P025"
        result["controls_permitted"] = False
        return result

    panel = _control_panel(xp, authenticated, shape, codec, controls, prepared)
    excess = source_gain - panel["strongest_control_gain_bpw"]
    result.update(panel)
    result.update({
        "controls_permitted": True,
        "controls_rerun": True,
        "source_minus_strongest_control_bpw": excess,
        "status": (
            "ELIGIBLE_FOR_INDEPENDENT_PAYLOAD_PILOT_AUDIT"
            if excess + 1e-15 >= codec.MIN_CONTROL_EXCESS_BPW
            else "HARD_KILL_SOURCE_NOT_SPECIFIC_0P03_BPW"
        ),
    })
    return result
