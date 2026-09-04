#!/usr/bin/env python3
"""Bounded Qwen STRATA-RM6 pilot on the pinned local RTX 3060."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
WORKSPACE = REPO.parent
SOURCE = REPO / "research/strata_rm6_label_flexible_gate_v0"
DESIGN_PATH = PACKAGE / "DESIGN_LOCK.json"
SITE = WORKSPACE / ".venv-cupy/Lib/site-packages"
CACHE = WORKSPACE / "tmp/strata_rm6_qwen_local3060_pilot_v0_cache"

_DLL_HANDLES: list[Any] = []
for _directory in [
    WORKSPACE / ".tools/cuda_dlls_3060",
    *[SITE / "nvidia" / name / "bin" for name in
      ("cublas", "cuda_nvrtc", "cuda_runtime", "cufft", "curand",
       "cusolver", "cusparse", "nvjitlink")],
]:
    if not _directory.is_dir():
        raise RuntimeError(f"missing pinned DLL directory: {_directory}")
    _DLL_HANDLES.append(os.add_dll_directory(str(_directory)))
os.environ["CUDA_PATH"] = str(SITE / "nvidia/cuda_runtime")
os.environ["CUPY_CACHE_DIR"] = str(CACHE)
CACHE.mkdir(parents=True, exist_ok=True)

import cupy as cp
import numpy as np

sys.path.insert(0, str(SOURCE))
import packet_codec
import rm6_core
import strata_rm_sc


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def verify_manifest(package: Path, expected_manifest_sha: str,
                    expected_root: str) -> dict[str, Any]:
    path = package / "SOURCE_MANIFEST.json"
    raw = path.read_bytes()
    if sha_bytes(raw) != expected_manifest_sha:
        raise RuntimeError(f"source manifest hash mismatch: {package}")
    data = json.loads(raw)
    digest = hashlib.sha256()
    for row in data["members"]:
        member = package / row["name"]
        payload = member.read_bytes()
        if len(payload) != int(row["bytes"]) or sha_bytes(payload) != row["sha256"]:
            raise RuntimeError(f"source member mismatch: {member}")
        digest.update(row["name"].encode("ascii") + b"\0" +
                      bytes.fromhex(row["sha256"]))
    if data["source_root_sha256"] != expected_root:
        raise RuntimeError("source root declaration mismatch")
    return {"manifest_sha256": expected_manifest_sha,
            "source_root_sha256": expected_root, "members": len(data["members"]),
            "member_hashes_verified": True}


def verify_static_audit(design: dict[str, Any]) -> dict[str, Any]:
    row = design["immutable_source"]
    package = REPO / row["static_independent_audit_package"]
    path = package / "AUDIT_SOURCE_MANIFEST.json"
    raw = path.read_bytes()
    if sha_bytes(raw) != row["static_independent_audit_manifest_sha256"]:
        raise RuntimeError("static audit manifest hash mismatch")
    manifest = json.loads(raw)
    for member in manifest["members"]:
        payload = (package / member["name"]).read_bytes()
        if len(payload) != int(member["bytes"]) or sha_bytes(payload) != member["sha256"]:
            raise RuntimeError(f"static audit member mismatch: {member['name']}")
    return {"package": row["static_independent_audit_package"],
            "manifest_sha256": sha_bytes(raw),
            "audit_source_root_sha256": manifest["audit_source_root_sha256"],
            "members": len(manifest["members"]),
            "disposition": row["static_audit_disposition"],
            "execution_scope": "static independent source review; payload remained held"}


def verify_panel(design: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    qwen = design["qwen"]
    panel_path = REPO / qwen["panel_lock"]
    raw = panel_path.read_bytes()
    if sha_bytes(raw) != qwen["panel_lock_sha256"]:
        raise RuntimeError("panel lock hash mismatch")
    panel = json.loads(raw)
    if (panel["model"] != qwen["model"] or panel["revision"] != qwen["revision"] or
            int(panel["layer"]) != int(qwen["layer"])):
        raise RuntimeError("panel identity mismatch")
    match = next((row for row in panel["files"]
                  if int(row["expert"]) == int(qwen["expert"]) and
                  row["role"] == qwen["role"]), None)
    if match is None or match["relative_path"] != qwen["payload_relative_path"]:
        raise RuntimeError("selected payload absent from panel")
    payload_path = WORKSPACE / match["relative_path"]
    payload = payload_path.read_bytes()
    if (len(payload) != int(qwen["payload_bytes"]) or
            sha_bytes(payload) != qwen["payload_sha256"] or
            len(payload) != int(match["bytes"]) or sha_bytes(payload) != match["sha256"]):
        raise RuntimeError("Qwen payload byte identity mismatch")
    return payload_path, {"panel_lock_sha256": sha_bytes(raw),
                          "payload_path": str(payload_path),
                          "payload_bytes": len(payload),
                          "payload_sha256": sha_bytes(payload),
                          "panel_entry_exact": True}


def bf16_values(raw: bytes) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)


def splitmix64_signs(n: int, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.arange(n, dtype=np.uint64) + np.uint64(seed)
        z += np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return np.where((z & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float64)


def signed_rht(values: np.ndarray, seed: int) -> np.ndarray:
    out = cp.asarray(values, dtype=cp.float64)
    out *= cp.asarray(splitmix64_signs(values.size, seed))
    width = 1
    while width < values.size:
        view = out.reshape(-1, 2, width)
        left, right = view[:, 0, :].copy(), view[:, 1, :].copy()
        view[:, 0, :], view[:, 1, :] = left + right, left - right
        width *= 2
    out *= 1.0 / math.sqrt(values.size)
    result = cp.asnumpy(out)
    return result


def fp16_bits(value: float) -> tuple[int, float]:
    half = np.asarray([value], dtype="<f2")
    bits = int(half.view("<u2")[0])
    exact = float(half[0])
    if not math.isfinite(exact) or exact <= 0.0:
        raise RuntimeError("invalid FP16 scale")
    return bits, exact


def posterior_leaf_lr(y: np.ndarray, weights: np.ndarray, distortion: float,
                      previous: np.ndarray, level: int) -> np.ndarray:
    alphabet = rm6_core.ETA * np.arange(-31, 33, dtype=np.float64)
    yg, ag, wg = cp.asarray(y), cp.asarray(alphabet), cp.asarray(weights)
    density = cp.exp(-0.5 * ((yg[:, None] - ag[None, :]) ** 2) / distortion)
    density *= wg[None, :]
    lower_mod, bit = 1 << (level - 1), 1 << (level - 1)
    out = cp.empty(y.size, dtype=cp.float64)
    index = cp.arange(64, dtype=cp.int16)
    previous_gpu = cp.asarray(previous)
    for context in range(lower_mod):
        positions = cp.where(previous_gpu == context)[0]
        candidates = index[(index % lower_mod) == context]
        zero = candidates[(candidates & bit) == 0]
        one = candidates[(candidates & bit) != 0]
        p0 = density[positions[:, None], zero[None, :]].sum(axis=1)
        p1 = density[positions[:, None], one[None, :]].sum(axis=1)
        out[positions] = cp.clip(p0 / cp.maximum(p1, 1e-300), 1e-30, 1e30)
    return cp.asnumpy(out)


def rm_sc_initial(y: np.ndarray, bank_id: int, profile_q: int, sc_seed: int,
                  coset_mode: str, test_distortion: float) -> dict[str, Any]:
    weights = strata_rm_sc.profile_weights(profile_q)
    previous = np.zeros(y.size, dtype=np.int16)
    decisions: list[np.ndarray] = []
    for level0, order in enumerate(rm6_core.ORDER_BANK[bank_id]):
        lr = posterior_leaf_lr(y, weights, test_distortion, previous, level0 + 1)
        if coset_mode == "zero":
            frozen = np.zeros(y.size, dtype=np.uint8)
        else:
            frozen = rm6_core.frozen_external_from_seed(y.size, sc_seed, level0 + 1)
        policy = strata_rm_sc.GreedyProbabilityBits()
        row = strata_rm_sc.run_sc_level(
            lr, rm6_core.rm_freeze_flag(12, order), frozen, policy)
        previous += (1 << level0) * row["output"].astype(np.int16)
        decisions.append(row["selected"].copy())
    replay = strata_rm_sc.replay_six_prescribed(
        bank_id, profile_q, sc_seed, coset_mode, decisions)
    if not np.array_equal(replay["indices"], previous.astype(np.uint8)):
        raise RuntimeError("source-conditioned RM-SC replay mismatch")
    return {"indices": replay["indices"], "decisions": decisions}


def packet_checkpoint(decisions: list[np.ndarray], *, bank_id: int,
                      scale_bits: int, profile_q: int, coset_mode: str,
                      sc_seed: int, rht_seed: int, costs: np.ndarray,
                      source_energy: float, flip: int, output_dir: Path,
                      prefix: str) -> dict[str, Any]:
    replay = strata_rm_sc.replay_six_prescribed(
        bank_id, profile_q, sc_seed, coset_mode, decisions)
    ledger = packet_codec.packet_ledger(bank_id, int(replay["logical_bits"]))
    packet = None
    decoded_indices = replay["indices"]
    canonical = False
    packet_file = None
    packet_sha = None
    try:
        packet, encoded = packet_codec.encode_packet(
            decisions, bank_id=bank_id, scale_fp16_bits=scale_bits,
            profile_q=profile_q, coset_mode=coset_mode, sc_seed=sc_seed,
            rht_seed=rht_seed)
        decoded = packet_codec.decode_packet(packet)
        if not (decoded["canonical_reencode_match"] and
                np.array_equal(decoded["indices"], encoded["indices"]) and
                np.array_equal(decoded["indices"], replay["indices"])):
            raise RuntimeError("literal packet independent replay mismatch")
        decoded_indices = decoded["indices"]
        canonical = True
        packet_file = f"{prefix}_flip{flip:03d}.srm6.bin"
        (output_dir / packet_file).write_bytes(packet)
        packet_sha = sha_bytes(packet)
    except rm6_core.RM6Error as error:
        if str(error) != "actual arithmetic packet exceeds 2.5 bpw":
            raise
    sse_normalized = rm6_core.selected_distortion(costs, decoded_indices)
    scale = rm6_core.fp16_from_bits(scale_bits)
    relative_mse = sse_normalized * scale * scale / source_energy
    rate = float(ledger["actual_physical_bpw"])
    f_value = relative_mse * math.exp2(2.0 * rate)
    return {"flip": flip, "packet_file": packet_file,
            "packet_bytes": int(ledger["actual_packet_bytes"]),
            "literal_packet_emitted": packet is not None,
            "packet_sha256": packet_sha, "logical_bits": int(replay["logical_bits"]),
            "physical_bpw": rate, "passes_2_5_bpw":
                bool(ledger["actual_passes_2_5_bpw"]),
            "target_rate_eligible": bool(ledger["actual_target_rate_eligible"]),
            "normalized_sse": sse_normalized, "relative_mse": relative_mse,
            "F_at_literal_rate": f_value, "indices_sha256":
                sha_bytes(decoded_indices.tobytes()),
            "canonical_reencode_match": canonical,
            "physical_failure": None if packet is not None else
                "actual arithmetic packet exceeds 2.5 bpw"}


def optimize_one(raw_values: np.ndarray, design: dict[str, Any], coset_mode: str,
                 output_dir: Path, prefix: str) -> dict[str, Any]:
    codec, search = design["codec"], design["search"]
    rht_seed = int(codec["rht_seed_u64"])
    transformed = signed_rht(raw_values, rht_seed)
    rms = float(np.sqrt(np.mean(transformed * transformed, dtype=np.float64)))
    scale_bits, scale = fp16_bits(rms)
    y = transformed / scale
    costs = rm6_core.exact_distortion_costs(y, 0x3C00)
    source_energy = float(np.sum(raw_values * raw_values, dtype=np.float64))
    initial = rm_sc_initial(y, int(codec["bank_id"]), int(codec["profile_q"]),
                            int(codec["sc_seed_u32"]), coset_mode,
                            float(codec["test_channel_distortion_normalized"]))
    decisions = [row.copy() for row in initial["decisions"]]
    indices = cp.asarray(initial["indices"], dtype=cp.uint8)
    table = cp.asarray(costs, dtype=cp.float64)
    coordinate = cp.arange(y.size, dtype=cp.int64)
    generators = [cp.asarray(rm6_core.generator_matrix(12, order), dtype=cp.float64)
                  for order in rm6_core.ORDER_BANK[int(codec["bank_id"])]]
    checkpoints = set(int(value) for value in search["checkpoint_flips"])
    snapshots: dict[int, list[np.ndarray]] = {0: [row.copy() for row in decisions]}
    trajectory = [rm6_core.selected_distortion(costs, initial["indices"])]
    choices: list[dict[str, Any]] = []
    started = time.perf_counter()
    for iteration in range(int(search["maximum_flips"])):
        delta_columns = []
        for plane in range(6):
            alternate = indices ^ cp.uint8(1 << plane)
            delta_columns.append(table[coordinate, alternate] - table[coordinate, indices])
        delta_matrix = cp.stack(delta_columns, axis=1)
        candidates = [generators[plane] @ delta_matrix[:, plane]
                      for plane in range(6)]
        minima = [float(cp.min(row).get()) for row in candidates]
        plane = min(range(6), key=lambda value: (minima[value], value))
        coefficient = int(cp.argmin(candidates[plane]).get())
        predicted = minima[plane]
        if predicted >= -1e-12:
            break
        before = trajectory[-1]
        mask = generators[plane][coefficient] != 0.0
        indices[mask] ^= cp.uint8(1 << plane)
        decisions[plane][coefficient] ^= np.uint8(1)
        current_cpu = cp.asnumpy(indices)
        after = rm6_core.selected_distortion(costs, current_cpu)
        if after > before + 1e-9:
            raise RuntimeError("non-monotone exact CPU flip")
        trajectory.append(after)
        choices.append({"flip": iteration + 1, "plane": plane,
                        "coefficient": coefficient, "gpu_predicted_delta": predicted,
                        "cpu_measured_delta": after - before})
        count = iteration + 1
        if count in checkpoints:
            snapshots[count] = [row.copy() for row in decisions]
    taken = len(choices)
    if taken not in snapshots:
        snapshots[taken] = [row.copy() for row in decisions]
    packet_rows = []
    for flip, snapshot in sorted(snapshots.items()):
        packet_rows.append(packet_checkpoint(
            snapshot, bank_id=int(codec["bank_id"]), scale_bits=scale_bits,
            profile_q=int(codec["profile_q"]), coset_mode=coset_mode,
            sc_seed=int(codec["sc_seed_u32"]), rht_seed=rht_seed, costs=costs,
            source_energy=source_energy, flip=flip, output_dir=output_dir,
            prefix=f"{prefix}_{coset_mode}"))
    eligible = [row for row in packet_rows if row["target_rate_eligible"]]
    selection_pool = eligible if eligible else packet_rows
    best = min(selection_pool, key=lambda row: (row["F_at_literal_rate"], row["flip"]))
    baseline = packet_rows[0]
    reduction = 1.0 - best["relative_mse"] / baseline["relative_mse"]
    equivalent = -0.5 * math.log2(best["F_at_literal_rate"] /
                                  baseline["F_at_literal_rate"])
    return {
        "coset_mode": coset_mode, "raw_values": raw_values.size,
        "source_energy": source_energy, "post_rht_rms_fp64": rms,
        "scale_fp16_bits": scale_bits, "scale_fp16_exact": scale,
        "exact_64way_cost_table_shape": list(costs.shape),
        "unconstrained_64way_relative_mse": float(np.min(costs, axis=1).sum() *
                                                   scale * scale / source_energy),
        "sc_initialization": baseline, "checkpoints": packet_rows,
        "selected_checkpoint": best,
        "selected_mse_reduction_fraction_vs_rm_sc": reduction,
        "selected_F_equivalent_gain_bpw_vs_rm_sc": equivalent,
        "flips_taken": taken, "terminated_at_local_single_flip_minimum":
            taken < int(search["maximum_flips"]),
        "trajectory_initial_normalized_sse": trajectory[0],
        "trajectory_final_normalized_sse": trajectory[-1],
        "trajectory_monotone_cpu_exact": all(b <= a + 1e-9
                                               for a, b in zip(trajectory, trajectory[1:])),
        "choices_sha256": sha_bytes(canonical(choices)),
        "last_choices": choices[-8:], "seconds": time.perf_counter() - started,
        "search_is_coordinate_descent_not_global_nearest_rm6": True,
    }


def runtime_identity() -> dict[str, Any]:
    output = subprocess.run(
        [r"C:\Windows\System32\nvidia-smi.exe", "--query-gpu=name,uuid,driver_version",
         "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    name, uuid, driver = [value.strip() for value in output.split(",")]
    props = cp.cuda.runtime.getDeviceProperties(0)
    cp_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    if name != "NVIDIA GeForce RTX 3060" or uuid != "GPU-458a424a-76e3-65e5-0470-803e0ed131ca":
        raise RuntimeError("local RTX 3060 identity mismatch")
    return {"nvidia_smi_name": name, "cupy_name": cp_name, "uuid": uuid,
            "driver_text": driver, "cupy_version": cp.__version__,
            "cuda_runtime_api": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_api": int(cp.cuda.runtime.driverGetVersion()),
            "python": sys.executable, "network_accessed": False,
            "runpod_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    design_raw = DESIGN_PATH.read_bytes()
    design = json.loads(design_raw)
    source = verify_manifest(
        SOURCE, design["immutable_source"]["source_manifest_sha256"],
        design["immutable_source"]["source_root_sha256"])
    audit = verify_static_audit(design)
    payload_path, panel = verify_panel(design)
    runtime = runtime_identity()
    qwen = design["qwen"]
    all_values = bf16_values(payload_path.read_bytes())
    begin = int(qwen["block_ordinal"]) * int(qwen["block_values"])
    raw_qwen = all_values[begin:begin + int(qwen["block_values"])].copy()
    if raw_qwen.size != int(qwen["block_values"]):
        raise RuntimeError("selected Qwen block outside payload")
    rng = np.random.default_rng(int(design["control"]["seed"]))
    gaussian = rng.normal(float(np.mean(raw_qwen)), float(np.std(raw_qwen)),
                          size=raw_qwen.size).astype(np.float32)
    gaussian_words = (gaussian.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    raw_control = (gaussian_words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    results = {"qwen": [], "matched_gaussian": []}
    for coset in design["codec"]["coset_modes"]:
        results["qwen"].append(optimize_one(raw_qwen, design, coset,
                                             output_path.parent, "qwen"))
        results["matched_gaussian"].append(optimize_one(
            raw_control, design, coset, output_path.parent, "gaussian"))
    qwen_best = max(results["qwen"], key=lambda row:
                    row["selected_mse_reduction_fraction_vs_rm_sc"])
    control_same = next(row for row in results["matched_gaussian"]
                        if row["coset_mode"] == qwen_best["coset_mode"])
    q_gain = qwen_best["selected_mse_reduction_fraction_vs_rm_sc"]
    c_gain = control_same["selected_mse_reduction_fraction_vs_rm_sc"]
    excess = q_gain - c_gain
    decision = design["decision"]
    target_eligible = qwen_best["selected_checkpoint"]["target_rate_eligible"]
    any_qwen_literal = any(checkpoint["literal_packet_emitted"]
                            for row in results["qwen"]
                            for checkpoint in row["checkpoints"])
    if not any_qwen_literal:
        verdict = "HARD_KILL_BANK0_QWEN_PHYSICAL_OVERFLOW_AT_ALL_CHECKPOINTS"
    elif (q_gain >= float(decision["promote_panel_at_qwen_mse_reduction_fraction"]) and
            excess >= float(decision["promote_panel_minimum_qwen_minus_control_percentage_points"]) and
            target_eligible):
        verdict = "PROMOTE_TO_FROZEN_MULTI_BLOCK_PANEL"
    elif q_gain < float(decision["hard_kill_pilot_below_qwen_mse_reduction_fraction"]):
        verdict = "HARD_KILL_THIS_BOUNDED_RM6_SINGLE_FLIP_PILOT"
    else:
        verdict = "HOLD_AMBIGUOUS_SINGLE_BLOCK_OR_CONTROL_CORRECTED_RESULT"
    receipt = {
        "schema": "strata-rm6-qwen-local3060-pilot-v0-result",
        "status": verdict, "claim_boundary": design["claim_boundary"],
        "design_lock_sha256": sha_bytes(design_raw), "source": source,
        "static_independent_audit": audit, "panel": panel, "runtime": runtime,
        "selection": {"block_ordinal": qwen["block_ordinal"],
                      "value_offset": begin, "values": raw_qwen.size,
                      "qwen_block_sha256": sha_bytes(raw_qwen.tobytes()),
                      "control_block_sha256": sha_bytes(raw_control.tobytes()),
                      "qwen_raw_mean": float(np.mean(raw_qwen)),
                      "qwen_raw_std": float(np.std(raw_qwen)),
                      "control_raw_mean": float(np.mean(raw_control)),
                      "control_raw_std": float(np.std(raw_control))},
        "results": results,
        "decision_metrics": {"best_qwen_coset": qwen_best["coset_mode"],
                             "qwen_mse_reduction_fraction": q_gain,
                             "matched_control_mse_reduction_fraction": c_gain,
                             "qwen_minus_control_percentage_points": excess,
                             "selected_literal_packet_target_rate_eligible": target_eligible,
                             "thresholds": decision},
        "scientific_limits": {
            "one_qwen_block_only": True, "up_role_only": True,
            "coordinate_descent_not_global_rm6_optimum": True,
            "current_global_strata_not_executed": True,
            "outer_expert_container_not_implemented": True,
            "cold_read_production_amplification_unestablished": True,
            "local_packet_is_contiguous_one_read_of_its_own_bytes": True,
            "no_whole_expert_or_target_F_claim": True,
        },
        "qwen_payload_accessed": True, "network_accessed": False,
        "runpod_accessed": False,
    }
    receipt["result_sha256_excluding_self"] = sha_bytes(canonical(receipt))
    output_path.write_bytes(canonical(receipt) + b"\n")
    print(json.dumps({"status": verdict, "output": str(output_path),
                      "result_sha256": sha_bytes(output_path.read_bytes()),
                      "qwen_gain": q_gain, "control_gain": c_gain,
                      "qwen_minus_control": excess}, sort_keys=True))


if __name__ == "__main__":
    main()
