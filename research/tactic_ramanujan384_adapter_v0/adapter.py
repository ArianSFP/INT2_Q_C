#!/usr/bin/env python3
"""Authenticated whole-expert adapter; source-frozen and not payload-authorized."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
ROLE_ORDER = ("gate", "up", "down_transposed")


class AdapterError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def load(name: str, filename: str) -> Any:
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module loader {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blocks(xp: Any, values: Any, block_values: int) -> Any:
    flat = xp.asarray(values, dtype=xp.float64).reshape(-1)
    blocks = (int(flat.size) + block_values - 1) // block_values
    padded = xp.zeros(blocks * block_values, dtype=xp.float64)
    padded[:flat.size] = flat
    return padded.reshape(blocks, block_values)


def _gain(input_sse: float, remaining_sse: float) -> float:
    require(input_sse > 0.0 and remaining_sse > 0.0
            and math.isfinite(input_sse) and math.isfinite(remaining_sse), "finite gain energies")
    return -0.5 * math.log2(remaining_sse / input_sse)


def run_authenticated_expert(
    xp: Any,
    *,
    role_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate three roles, run one nested score, and emit one literal object.

    Every mapping is passed directly to ``authenticated_io.authenticate_role``.
    There is deliberately no path-discovery or model-identity fallback.
    """

    require(len(role_inputs) == 3, "three authenticated role inputs")
    auth = load("tactic_ramanujan384_adapter_auth", "authenticated_io.py")
    codec = load("tactic_ramanujan384_adapter_codec", "ramanujan_codec.py")
    container = load("tactic_ramanujan384_adapter_container", "container.py")
    authenticated = [auth.authenticate_role(**dict(arguments)) for arguments in role_inputs]
    require(tuple(row["role"] for row in authenticated) == ROLE_ORDER, "canonical authenticated role order")
    shape = authenticated[0]["shape"]
    require(all(row["shape"] == shape for row in authenticated), "shared universal role shape")
    coarse_sha = hashlib.sha256(authenticated[0]["coarse_artifact_payload"]).hexdigest()
    require(all(hashlib.sha256(row["coarse_artifact_payload"]).hexdigest() == coarse_sha
                for row in authenticated), "one shared authenticated coarse artifact")
    require(all(row["coarse_artifact_bytes"] == authenticated[0]["coarse_artifact_bytes"]
                for row in authenticated), "shared coarse artifact length")

    basis = codec.build_public_dictionary(xp)
    source_rows = []
    residual_rows = []
    source_energy = 0.0
    input_sse = 0.0
    remaining_sse = 0.0
    fine_streams = []
    rank_histogram = {str(rank): 0 for rank in range(codec.MAX_RANK + 1)}
    for role, row in zip(ROLE_ORDER, authenticated, strict=True):
        source = _blocks(xp, row["source"], codec.BLOCK_VALUES)
        coarse = _blocks(xp, row["coarse"], codec.BLOCK_VALUES)
        residual = source - coarse
        encoded = codec.encode_residual_blocks(xp, residual, basis, role)
        source_rows.append(source)
        residual_rows.append(residual)
        source_energy += float(xp.sum(source * source, dtype=xp.float64).item())
        input_sse += encoded["input_sse"]
        remaining_sse += encoded["remaining_sse"]
        fine_streams.append(b"".join(encoded["packets"]))
        for rank in encoded["ranks"]:
            rank_histogram[str(rank)] += 1
    require(source_energy > 0.0 and input_sse > 0.0 and remaining_sse > 0.0, "nonzero expert score")
    relative_mse = remaining_sse / source_energy
    binding_sha = hashlib.sha256(b"".join(
        bytes.fromhex(row["binding_sha256"]) for row in authenticated
    )).hexdigest()
    composite = container.encode_composite(
        intermediate=int(shape[0]),
        hidden=int(shape[1]),
        coarse_payload=authenticated[0]["coarse_artifact_payload"],
        role_fine_streams=tuple(fine_streams),
        source_binding_sha256=binding_sha,
    )
    decoded_container = container.decode_composite(composite)
    physical_rate = 8.0 * len(composite) / (3 * shape[0] * shape[1])
    result = {
        "schema": "tactic-ramanujan384-authenticated-expert-result-v0",
        "status": None,
        "shape": {"intermediate": shape[0], "hidden": shape[1], "roles": list(ROLE_ORDER)},
        "weights": 3 * shape[0] * shape[1],
        "input_sse": input_sse,
        "source_energy": source_energy,
        "remaining_sse": remaining_sse,
        "relative_mse": relative_mse,
        "physical_rate_bpw": physical_rate,
        "F": relative_mse * 2.0 ** (2.0 * physical_rate),
        "source_capture": 1.0 - remaining_sse / input_sse,
        "source_gain_bpw": _gain(input_sse, remaining_sse),
        "rank_histogram": rank_histogram,
        "composite": composite,
        "composite_bytes": len(composite),
        "composite_sha256": hashlib.sha256(composite).hexdigest(),
        "container_page_padding_bytes": decoded_container["page_padding_bytes"],
        "external_read_amplification": decoded_container["external_read_amplification"],
        "controls_rerun": False,
        "exact_original_domain_fp64_score": True,
        "qwen_or_model_identity_used": False,
    }
    if not (2.15 <= physical_rate <= 2.5):
        result["status"] = "HARD_KILL_PHYSICAL_RATE_OUTSIDE_2P15_TO_2P5"
        result["controls_permitted"] = False
        return result
    if relative_mse > codec.TARGET_D + 1e-15:
        result["status"] = "HARD_KILL_ABSOLUTE_SOURCE_MISSES_D_0P025"
        result["controls_permitted"] = False
        return result

    parent = codec.load_audited_parent()
    phase_input = 0.0
    phase_remaining = 0.0
    for role, residual in zip(ROLE_ORDER, residual_rows, strict=True):
        control = parent.phase_destroyed_blocks(xp, residual, codec.PHASE_SEED)
        encoded = codec.encode_residual_blocks(xp, control, basis, role)
        phase_input += encoded["input_sse"]
        phase_remaining += encoded["remaining_sse"]
    phase_gain = _gain(phase_input, phase_remaining)
    gaussian_rows = []
    for seed in codec.GAUSSIAN_SEEDS:
        control_input = 0.0
        control_remaining = 0.0
        for role, residual in zip(ROLE_ORDER, residual_rows, strict=True):
            control = parent.moment_matched_gaussian_blocks(xp, residual, seed)
            encoded = codec.encode_residual_blocks(xp, control, basis, role)
            control_input += encoded["input_sse"]
            control_remaining += encoded["remaining_sse"]
        gaussian_rows.append({
            "seed": seed,
            "input_sse": control_input,
            "remaining_sse": control_remaining,
            "gain_bpw": _gain(control_input, control_remaining),
            "complete_three_role_finite_search_rerun": True,
        })
    strongest = max([phase_gain] + [row["gain_bpw"] for row in gaussian_rows])
    excess = result["source_gain_bpw"] - strongest
    result.update({
        "controls_permitted": True,
        "controls_rerun": True,
        "phase_control": {
            "seed": codec.PHASE_SEED,
            "input_sse": phase_input,
            "remaining_sse": phase_remaining,
            "gain_bpw": phase_gain,
            "complete_three_role_finite_search_rerun": True,
        },
        "gaussian_controls": gaussian_rows,
        "source_minus_strongest_control_bpw": excess,
        "status": (
            "ELIGIBLE_FOR_INDEPENDENT_PAYLOAD_PILOT_AUDIT"
            if excess + 1e-15 >= codec.MIN_CONTROL_EXCESS_BPW
            else "HARD_KILL_SOURCE_NOT_SPECIFIC_0P03_BPW"
        ),
    })
    return result
