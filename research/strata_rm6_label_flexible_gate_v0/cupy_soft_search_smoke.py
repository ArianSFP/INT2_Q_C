#!/usr/bin/env python3
"""CuPy source-free exact-cost local-search smoke for local RM6."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

import numpy as np

from packet_codec import decode_packet, encode_packet
from rm6_core import (LOCAL_LOG2, LOCAL_N, ORDER_BANK, RM6Error,
                      exact_distortion_costs, generator_matrix,
                      reconstruction_levels, require, selected_distortion)
from strata_rm_sc import replay_six_greedy


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_target(scale_bits: int) -> np.ndarray:
    coordinate = np.arange(LOCAL_N, dtype=np.int64)
    desired = ((17 * coordinate + 11 * (coordinate >> 3) +
                7 * (coordinate >> 7)) & 63).astype(np.uint8)
    levels = reconstruction_levels(scale_bits)
    return levels[desired] + 0.017 * np.sin(coordinate * (2.0 * np.pi / 37.0))


def run(steps: int) -> dict[str, Any]:
    import cupy as cp

    require(1 <= steps <= 32, "bounded smoke steps")
    bank_id, profile_q, sc_seed = 0, 96, 0x13579BDF
    rht_seed, scale_bits = 0x0123456789ABCDEF, 0x3C00  # FP16 1.0
    zero = replay_six_greedy(bank_id, profile_q, sc_seed, "zero")
    random_coset = replay_six_greedy(bank_id, profile_q, sc_seed, "current_random")
    target = source_target(scale_bits)
    costs = exact_distortion_costs(target, scale_bits)
    generator_cpu = generator_matrix(LOCAL_LOG2, 5)
    information = np.stack([np.asarray(bits, dtype=np.uint8)
                            for bits in zero["decisions"]])
    require(information.shape == (6, 1586), "uniform RM(5,12) message")

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    started = time.perf_counter()
    generator = cp.asarray(generator_cpu, dtype=cp.float64)
    table = cp.asarray(costs, dtype=cp.float64)
    indices = cp.asarray(zero["indices"], dtype=cp.uint8)
    coordinate = cp.arange(LOCAL_N, dtype=cp.int64)
    initial = float(cp.sum(table[coordinate, indices], dtype=cp.float64).get())
    trajectory = [initial]
    choices = []
    for iteration in range(steps):
        best_delta, best_plane, best_coefficient = float("inf"), -1, -1
        for plane in range(6):
            alternate = indices ^ np.uint8(1 << plane)
            delta_coordinate = table[coordinate, alternate] - table[coordinate, indices]
            delta = generator @ delta_coordinate
            coefficient = int(cp.argmin(delta).get())
            value = float(delta[coefficient].get())
            if value < best_delta:
                best_delta, best_plane, best_coefficient = value, plane, coefficient
        if best_delta >= -1e-12:
            break
        mask = generator[best_coefficient] != 0.0
        before = trajectory[-1]
        indices[mask] ^= np.uint8(1 << best_plane)
        information[best_plane, best_coefficient] ^= np.uint8(1)
        after = float(cp.sum(table[coordinate, indices], dtype=cp.float64).get())
        require(after <= before + 1e-9 and abs((after - before) - best_delta) <=
                1e-7 * max(1.0, abs(best_delta)), "GPU exact-cost delta replay")
        trajectory.append(after)
        choices.append({"iteration": iteration, "plane": best_plane,
                        "coefficient": best_coefficient, "predicted_delta": best_delta,
                        "measured_delta": after - before})
    final_indices = cp.asnumpy(indices)
    packet_fits, packet_row, packet_error = False, None, None
    try:
        packet, encoded = encode_packet(
            [information[level].tolist() for level in range(6)], bank_id=bank_id,
            scale_fp16_bits=scale_bits, profile_q=profile_q, coset_mode="zero",
            sc_seed=sc_seed, rht_seed=rht_seed)
        decoded = decode_packet(packet)
        require(np.array_equal(encoded["indices"], final_indices) and
                np.array_equal(decoded["indices"], final_indices),
                "GPU choices canonical packet replay")
        packet_fits = True
        packet_row = {"bytes": len(packet), "sha256": sha(packet),
                      "logical_bits": decoded["logical_bits"],
                      "physical_bpw": len(packet) * 8.0 / LOCAL_N,
                      "selected_information_bits": decoded["information_bits"],
                      "emitted_arithmetic_bits":
                          decoded["ledger"]["emitted_arithmetic_bits"],
                      "target_rate_eligible":
                          decoded["ledger"]["actual_target_rate_eligible"],
                      "promotion_status": decoded["ledger"]["promotion_status"]}
    except RM6Error as error:
        packet_error = str(error)
    cp.cuda.runtime.deviceSynchronize()
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "schema": "strata-rm6-label-flexible-cupy-smoke-v0",
        "status": "PASS_LOCAL_GREEDY_MECHANISM_HOLD_PRODUCTION_SEARCH",
        "device": {"name": str(name), "cupy_version": cp.__version__,
                   "runtime_version": int(cp.cuda.runtime.runtimeGetVersion())},
        "block_values": LOCAL_N, "bank_id": bank_id,
        "orders": list(ORDER_BANK[bank_id]), "dimensions": [1586] * 6,
        "information_bits": 9516, "steps_requested": steps,
        "steps_taken": len(choices), "trajectory": trajectory, "choices": choices,
        "monotone_nonincreasing": all(right <= left + 1e-9
                                       for left, right in zip(trajectory, trajectory[1:])),
        "cpu_selected_distortion_match": abs(
            selected_distortion(costs, final_indices) - trajectory[-1]) <= 1e-8,
        "packet_fits_2_5_bpw": packet_fits, "packet": packet_row,
        "packet_target_rate_eligible": bool(
            packet_row is not None and packet_row["target_rate_eligible"]),
        "subminimum_packet_is_mechanism_fixture_only": True,
        "packet_error": packet_error,
        "coset_control": {
            "zero_frozen_logical_bits": int(zero["logical_bits"]),
            "current_random_frozen_logical_bits": int(random_coset["logical_bits"]),
            "index_disagreement_fraction": float(np.mean(
                zero["indices"] != random_coset["indices"])),
            "random_coset_is_not_low_degree_coordinate_function_evidence": True,
        },
        "seconds": time.perf_counter() - started,
        "production_joint_rm5_12_search_implemented": False,
        "global_n20_n21_strata_swap_implemented": False,
        "qwen_payload_accessed": False, "coarse_payload_accessed": False,
        "control_payload_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output")
    args = parser.parse_args()
    receipt = run(args.steps)
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
