#!/usr/bin/env python3
"""Authenticated source-first Qwen pilot runner.

The frozen v0 package cannot pass its first authorization check because its
compile-time capability digest is None.  The remaining code is the payload
path to be enabled only by a later independently audited deployment sibling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import capability


AUTHORIZATION = "RUN_PINNED_TACTIC_RAMANUJAN384_QWEN_PILOT_V0"


class PilotError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_BINARY", 0), 0o600)
    try:
        cursor = 0
        while cursor < len(payload):
            written = os.write(descriptor, payload[cursor:])
            require(written > 0, "short output write")
            cursor += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    write_new(path, canonical_json(dict(value)) + b"\n")


def load_snapshot_module(payload: bytes, name: str, origin: str) -> Any:
    require(type(payload) is bytes and payload, "snapshot module bytes")
    module = types.ModuleType(name)
    module.__file__ = origin
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(payload, origin, "exec", dont_inherit=True), module.__dict__)
    return module


def verify_cupy(cp: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    module_path = Path(cp.__file__).resolve(strict=True)
    module_payload = module_path.read_bytes()
    device = int(expected["device_ordinal"])
    cp.cuda.Device(device).use()
    properties = cp.cuda.runtime.getDeviceProperties(device)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    observed = {
        "version": str(cp.__version__),
        "module_file_sha256": hashlib.sha256(module_payload).hexdigest(),
        "device_ordinal": device,
        "device_name": str(name),
        "compute_capability": [int(properties["major"]), int(properties["minor"])],
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
    }
    require(observed == dict(expected), "CuPy/CUDA runtime capability mismatch")
    probe = cp.arange(4096, dtype=cp.int64)
    probe = (probe * cp.int64(17) + cp.int64(3)) % cp.int64(65521)
    observed_sum = int(cp.asnumpy(cp.sum(probe, dtype=cp.int64)))
    expected_sum = sum((index * 17 + 3) % 65521 for index in range(4096))
    require(observed_sum == expected_sum, "CuPy compiled-kernel probe")
    cp.cuda.Stream.null.synchronize()
    observed["compiled_kernel_probe"] = expected_sum
    return observed


def one_pass_page_trace(path: Path, expected: bytes) -> tuple[bytes, dict[str, Any]]:
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    output = bytearray()
    events = []
    try:
        before = os.fstat(descriptor)
        require(before.st_size == len(expected), "one-pass object size")
        page_index = 0
        while len(output) < len(expected):
            row = os.read(descriptor, capability.PAGE_BYTES)
            require(len(row) == capability.PAGE_BYTES, "complete physical page read")
            events.append({"sequence": page_index, "page_index": page_index,
                           "file_offset": page_index * capability.PAGE_BYTES,
                           "bytes_read": len(row),
                           "page_sha256": hashlib.sha256(row).hexdigest()})
            output.extend(row)
            page_index += 1
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                "one-pass object identity")
    finally:
        os.close(descriptor)
    payload = bytes(output)
    require(payload == expected, "one-pass read byte replay")
    return payload, {
        "schema": "tactic-ramanujan384-qwen-one-pass-page-trace-v0",
        "page_bytes": capability.PAGE_BYTES,
        "object_bytes": len(expected),
        "events": events,
        "physical_bytes_read": len(payload),
        "read_amplification": 1.0,
        "single_pass": True,
        "layout_projection": False,
        "accelerator_hbm_measured": False,
    }


def independent_decoded_score(np: Any, source: Mapping[str, Any],
                              reconstruction: Mapping[str, Any],
                              physical_bytes: int) -> dict[str, Any]:
    roles = []
    total_sse = 0.0
    total_energy = 0.0
    total_values = 0
    for role in capability.ROLE_ORDER:
        original = np.asarray(source[role], dtype=np.float64).reshape(-1)
        decoded = np.asarray(reconstruction[role], dtype=np.float64).reshape(-1)
        require(original.shape == decoded.shape and np.all(np.isfinite(decoded)),
                "independent decoded score arrays")
        error = original - decoded
        sse = float(np.sum(error * error, dtype=np.float64))
        energy = float(np.sum(original * original, dtype=np.float64))
        require(math.isfinite(sse) and math.isfinite(energy) and energy > 0.0,
                "independent decoded score energy")
        roles.append({"role": role, "weights": int(original.size),
                      "sse_fp64": sse, "source_energy_fp64": energy,
                      "reconstruction_f64_sha256": hashlib.sha256(
                          np.ascontiguousarray(decoded, dtype="<f8").tobytes()).hexdigest()})
        total_sse += sse
        total_energy += energy
        total_values += int(original.size)
    relative = total_sse / total_energy
    rate = 8.0 * physical_bytes / total_values
    return {"roles": roles, "pooled_sse_fp64": total_sse,
            "pooled_source_energy_fp64": total_energy,
            "weights": total_values, "physical_bytes": physical_bytes,
            "relative_mse": relative, "physical_rate_bpw": rate,
            "F": relative * 2.0 ** (2.0 * rate),
            "decoded_bytes_rescored_in_fp64": True}


def _gain(input_sse: float, remaining_sse: float) -> float:
    require(input_sse > 0.0 and remaining_sse > 0.0 and
            math.isfinite(input_sse) and math.isfinite(remaining_sse),
            "control gain energies")
    return -0.5 * math.log2(remaining_sse / input_sse)


def _phase_control(np: Any, residual: Any, seed: int) -> Any:
    rows = np.asarray(residual, dtype=np.float64).reshape(384, 4096)
    output = np.empty_like(rows)
    for block in range(384):
        prefix = seed.to_bytes(8, "little") + block.to_bytes(8, "little")
        keys = [hashlib.sha256(prefix + coordinate.to_bytes(4, "little")).digest()
                for coordinate in range(4096)]
        order = sorted(range(4096), key=lambda index: (keys[index], index))
        output[block] = rows[block, np.asarray(order, dtype=np.int64)]
    return output.reshape(-1)


def run_controls(cp: Any, np: Any, core: Any, prepared: Mapping[str, Any],
                 source: Mapping[str, Any], coarse: Mapping[str, Any],
                 shape: Any) -> dict[str, Any]:
    residual = {role: np.asarray(source[role] - coarse[role], dtype=np.float64)
                for role in capability.ROLE_ORDER}
    phase_input = phase_remaining = 0.0
    phase_streams = []
    for ordinal, role in enumerate(capability.ROLE_ORDER):
        controlled = _phase_control(np, residual[role], 0xA17C9E35 + ordinal)
        encoded = core.encode_role_batched(
            cp, cp.asarray(controlled), cp.zeros(shape.role_values, dtype=cp.float64),
            shape, role, prepared)
        phase_input += encoded["input_sse"]
        phase_remaining += encoded["remaining_sse"]
        phase_streams.append(encoded["stream_sha256"])
    phase = {"kind": "phase", "input_sse": phase_input,
             "remaining_sse": phase_remaining,
             "gain_bpw": _gain(phase_input, phase_remaining),
             "fine_stream_sha256_by_role": phase_streams}
    gaussians = []
    valid_counts = (4096,) * 384
    for seed in core.GAUSSIAN_SEEDS:
        input_sse = remaining_sse = 0.0
        streams = []
        controls = []
        for role in capability.ROLE_ORDER:
            reference = residual[role].reshape(384, 4096)
            controlled, receipt = core.moment_matched_gaussian(
                cp, cp.asarray(reference), int(seed), valid_counts)
            encoded = core.encode_role_batched(
                cp, controlled.reshape(-1),
                cp.zeros(shape.role_values, dtype=cp.float64), shape, role, prepared)
            input_sse += encoded["input_sse"]
            remaining_sse += encoded["remaining_sse"]
            streams.append(encoded["stream_sha256"])
            controls.append(receipt["f64_sha256"])
        gaussians.append({"kind": "gaussian", "seed": int(seed),
                          "input_sse": input_sse, "remaining_sse": remaining_sse,
                          "gain_bpw": _gain(input_sse, remaining_sse),
                          "control_f64_sha256_by_role": controls,
                          "fine_stream_sha256_by_role": streams})
    cp.cuda.Stream.null.synchronize()
    return {"phase_control": phase, "gaussian_controls": gaussians,
            "phase_count": 1, "gaussian_count": 8,
            "strongest_control_gain_bpw": max(
                [phase["gain_bpw"]] + [row["gain_bpw"] for row in gaussians])}


def execute(capability_path: Path, output_parent: Path, authorization: str) -> dict[str, Any]:
    require(authorization == AUTHORIZATION, "explicit pilot authorization")
    # This call fails before resolving capability_path while the compiled pin is None.
    authenticated = capability.authorize_production(capability_path)

    # Heavy/runtime imports occur only after the complete precommitted capability
    # and payload bindings have passed.
    np = importlib.import_module("numpy")
    cp = importlib.import_module("cupy")
    from aperture import (bf16_le_to_f64, bootstrap_capture_gate,
                          coarse_f32_le_to_f64, fixed_sample_blocks,
                          score_literal_rank_packets)
    backend = verify_cupy(cp, authenticated["document"]["cupy_runtime"])
    v2 = authenticated["closures"]["v2_scalable"]
    core = load_snapshot_module(v2["members"]["scalable_core.py"],
                                "tactic_r384_qwen_pilot_core",
                                "<authenticated-v2>/scalable_core.py")
    require(core.MAX_RANK == 14 and core.PACKET_BYTES == 48 and
            core.TARGET_D == capability.TARGET_D, "pinned core constants")
    values = 768 * 2048
    source = {}
    coarse = {}
    for role in capability.ROLE_ORDER:
        payloads = authenticated["role_payloads"][role]
        source[role] = bf16_le_to_f64(payloads["source_bf16"], values)
        coarse[role] = coarse_f32_le_to_f64(
            payloads["coarse_reconstruction_f32"], values)

    prepared = core.prepare_dictionary(cp)
    sample_rows = {}
    aperture_receipt = {"roles": {}, "sample_blocks": {
        role: list(capability.SAMPLE_BLOCKS[role]) for role in capability.ROLE_ORDER}}
    for role in capability.ROLE_ORDER:
        sampled_source = fixed_sample_blocks(source[role], capability.SAMPLE_BLOCKS[role])
        sampled_coarse = fixed_sample_blocks(coarse[role], capability.SAMPLE_BLOCKS[role])
        row = score_literal_rank_packets(
            cp, core, cp.asarray(sampled_source), cp.asarray(sampled_coarse),
            role, prepared)
        sample_rows[role] = row
        aperture_receipt["roles"][role] = {
            key: value for key, value in row.items()
            if key not in {"candidate_sse", "input_sse_by_block",
                           "remaining_sse_by_block", "winner_rank_by_block"}
        }
        aperture_receipt["roles"][role].update({
            "candidate_sse": row["candidate_sse"].tolist(),
            "input_sse_by_block": row["input_sse_by_block"].tolist(),
            "remaining_sse_by_block": row["remaining_sse_by_block"].tolist(),
            "winner_rank_by_block": row["winner_rank_by_block"].tolist(),
        })
    cp.cuda.Stream.null.synchronize()
    gate = bootstrap_capture_gate(sample_rows)
    aperture_receipt.update(gate)
    aperture_receipt.update({"backend": backend,
                             "source_blocks_opened": 48,
                             "controls_executed": 0,
                             "full_expert_search_executed": False})

    output = Path(authenticated["document"]["output_parent"]).resolve(strict=True)
    require(output == output_parent.resolve(strict=True), "capability output parent")
    result_directory = output / ("tactic-r384-qwen-pilot-" +
                                 authenticated["capability_sha256"][:16])
    os.mkdir(result_directory)
    write_json_new(result_directory / "EARLY_APERTURE.json", aperture_receipt)
    if not gate["survives_to_full_expert"]:
        terminal = {"schema": "tactic-ramanujan384-qwen-pilot-result-v0",
                    "status": "HARD_KILL_SOURCE_FIRST_APERTURE",
                    "capability_sha256": authenticated["capability_sha256"],
                    "full_expert_executed": False, "controls_executed": False,
                    "projected_transfer_used": False,
                    "early_aperture_sha256": hashlib.sha256(
                        (result_directory / "EARLY_APERTURE.json").read_bytes()).hexdigest()}
        write_json_new(result_directory / "COMPLETE.json", terminal)
        return terminal

    shape = core.define_shape(768, 2048)
    encoded = {}
    fine_streams = []
    for role in capability.ROLE_ORDER:
        row = core.encode_role_batched(cp, cp.asarray(source[role]),
                                       cp.asarray(coarse[role]), shape, role, prepared)
        encoded[role] = row
        fine_streams.append(row["stream"])
    binding = hashlib.sha256(b"".join(bytes.fromhex(
        authenticated["role_payloads"][role]["source_sha256"])
        for role in capability.ROLE_ORDER)).hexdigest()
    composite = core.encode_composite(shape, authenticated["coarse_bytes"],
                                      tuple(fine_streams), binding)
    require(len(composite) == capability.EXPECTED_PHYSICAL_BYTES and
            8 * len(composite) * capability.EXPECTED_RATE_DENOMINATOR ==
            capability.EXPECTED_RATE_NUMERATOR * capability.EXPECTED_WEIGHTS,
            "literal 359/144 = 2.4930556-bpw container")
    composite_path = result_directory / "COMPOSITE.bin"
    write_new(composite_path, composite)
    replay, read_trace = one_pass_page_trace(composite_path, composite)
    decoded = core.decode_composite(replay)
    require(decoded["coarse"] == authenticated["coarse_bytes"],
            "decoded composite coarse segment equals audited coarse artifact")
    span = shape.blocks_per_role * core.PACKET_BYTES
    reconstruction = {}
    for ordinal, role in enumerate(capability.ROLE_ORDER):
        fine = decoded["fine"][ordinal * span:(ordinal + 1) * span]
        correction = core.decode_fine_role(cp, fine, shape, role, prepared)
        reconstructed = cp.asarray(coarse[role], dtype=cp.float64) + correction.reshape(-1)
        reconstruction[role] = np.ascontiguousarray(cp.asnumpy(reconstructed), dtype="<f8")
    cp.cuda.Stream.null.synchronize()
    score = independent_decoded_score(np, source, reconstruction, len(composite))
    full_result = {"schema": "tactic-ramanujan384-qwen-full-result-v0",
                   "score": score, "composite_sha256": hashlib.sha256(composite).hexdigest(),
                   "coarse_residual_input_sse": sum(row["input_sse"]
                                                     for row in encoded.values()),
                   "remaining_sse": score["pooled_sse_fp64"],
                   "decoded_byte_fp64_rescore": True,
                   "projected_transfer_used": False}
    write_json_new(result_directory / "FULL_RESULT.json", full_result)
    write_json_new(result_directory / "READ_TRACE.json", read_trace)
    if score["relative_mse"] > capability.TARGET_D + 1e-15:
        terminal = {"schema": "tactic-ramanujan384-qwen-pilot-result-v0",
                    "status": "HARD_KILL_FULL_EXPERT_D_GT_0_025",
                    "capability_sha256": authenticated["capability_sha256"],
                    "full_expert_executed": True, "controls_executed": False,
                    "relative_mse": score["relative_mse"], "F": score["F"],
                    "projected_transfer_used": False}
        write_json_new(result_directory / "COMPLETE.json", terminal)
        return terminal

    controls = run_controls(cp, np, core, prepared, source, coarse, shape)
    source_input = full_result["coarse_residual_input_sse"]
    source_gain = _gain(source_input, score["pooled_sse_fp64"])
    excess = source_gain - controls["strongest_control_gain_bpw"]
    controls.update({"source_gain_bpw": source_gain,
                     "source_minus_strongest_control_bpw": excess})
    write_json_new(result_directory / "CONTROLS.json", controls)
    terminal = {
        "schema": "tactic-ramanujan384-qwen-pilot-result-v0",
        "status": ("PASS_ELIGIBLE_FOR_INDEPENDENT_RESULT_AUDIT"
                   if excess + 1e-15 >= core.MIN_CONTROL_EXCESS_BPW
                   else "HARD_KILL_SOURCE_NOT_SPECIFIC"),
        "capability_sha256": authenticated["capability_sha256"],
        "full_expert_executed": True, "controls_executed": True,
        "phase_controls": 1, "gaussian_controls": 8,
        "relative_mse": score["relative_mse"], "F": score["F"],
        "physical_rate_bpw": score["physical_rate_bpw"],
        "one_pass_read_amplification": read_trace["read_amplification"],
        "source_minus_strongest_control_bpw": excess,
        "projected_transfer_used": False,
    }
    write_json_new(result_directory / "COMPLETE.json", terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    result = execute(args.capability, args.output_parent, args.authorization)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
