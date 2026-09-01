#!/usr/bin/env python3
"""CuPy PMG1 tetrad screen on frozen, disjoint auxiliary stage-1 coordinates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time


MANIFEST_SHA256 = "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782"
PLAN_SHA256 = "adcf1d8153c2a8a5048153edfa90f8f12d959d1d09e1cf7524359a532da950d1"
SEEDS = (3306464084, 235286348, 2174751347, 256779041)
SELECTION_EXPERTS = (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)
PLANNING_CAPTURE = 0.1457530997916614
T = 261120
KEY_RE = re.compile(r"^e(\d{3})\|(up|down)\|r(\d{3})\|c(\d{4})$")


CUDA_SOURCE = r'''
#include <curand_kernel.h>
#include <cuda_bf16.h>

extern "C" __global__ void pmg_anchor_coords(
    const unsigned long long* bases,
    const unsigned long long* addends,
    const unsigned long long* sequences,
    const unsigned long long* offset_quads,
    const unsigned long long* normal4_indices,
    const unsigned char* lanes,
    const unsigned char* role_codes,
    unsigned long long coordinate_count,
    unsigned long long seed_count,
    float* output) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long total = coordinate_count * seed_count;
  if (linear >= total) return;
  const unsigned long long seed_index = linear / coordinate_count;
  const unsigned long long coordinate = linear - seed_index * coordinate_count;
  const unsigned long long seed64 = bases[seed_index] + addends[coordinate];
  const unsigned long long offset_base = offset_quads[coordinate];
  const unsigned long long counter_low = offset_base + normal4_indices[coordinate];
  const unsigned long long carry = counter_low < offset_base ? 1ULL : 0ULL;
  const unsigned long long counter_high = sequences[coordinate] + carry;
  const uint4 counter = make_uint4(
      (unsigned int)counter_low, (unsigned int)(counter_low >> 32),
      (unsigned int)counter_high, (unsigned int)(counter_high >> 32));
  const uint2 key = make_uint2(
      (unsigned int)seed64, (unsigned int)(seed64 >> 32));
  const uint4 raw = curand_Philox4x32_10(counter, key);
  const float2 pair0 = _curand_box_muller(raw.x, raw.y);
  const float2 pair1 = _curand_box_muller(raw.z, raw.w);
  const float values[4] = {pair0.x, pair0.y, pair1.x, pair1.y};
  const float scale = role_codes[coordinate] == 0
      ? __uint_as_float(0x3c03126fU) : __uint_as_float(0x3a560a28U);
  output[linear] = __bfloat162float(
      __float2bfloat16_rn(values[(int)lanes[coordinate]] * scale));
}
'''


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def load_plan(path: Path):
    observed = sha256_file(path)
    if observed != PLAN_SHA256:
        raise RuntimeError(f"PMG plan module hash mismatch: {observed}")
    spec = importlib.util.spec_from_file_location("pmg1_frozen_plan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen PMG plan")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage1_keys(plan):
    identities, global_by_split, _lines, _subsets, _bundles, _attempts = plan.enumerate_stage0()
    fit = set(global_by_split["fit"])
    score = set(global_by_split["score"])
    plan.fill_plan("stage1", "fit", fit, score, identities, 2048)
    plan.fill_plan("stage1", "score", score, fit, identities, 2048)
    if len(fit) != 2048 or len(score) != 2048 or fit & score:
        raise RuntimeError("frozen stage-1 coordinate closure failed")
    return tuple(sorted(fit)), tuple(sorted(score))


def parse_key(key: str) -> tuple[int, str, int, int]:
    match = KEY_RE.fullmatch(key)
    if match is None:
        raise RuntimeError(f"malformed canonical coordinate: {key}")
    expert, role, row, column = match.groups()
    return int(expert), role, int(row), int(column)


def bf16_matrix(np, path: Path, expected_hash: str, role: str):
    if path.stat().st_size != 3_145_728:
        raise RuntimeError(f"source byte count mismatch: {path}")
    observed = sha256_file(path)
    if observed != expected_hash:
        raise RuntimeError(f"source hash mismatch: {path}: {observed}")
    words = np.fromfile(path, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    matrix = values.reshape(768, 2048) if role == "up" else values.reshape(2048, 768).T
    if matrix.shape != (768, 2048) or not bool(np.isfinite(matrix).all()):
        raise RuntimeError(f"source shape/finite mismatch: {path}")
    return matrix


def load_targets(np, workspace: Path, manifest: dict, keys_by_split: dict[str, tuple[str, ...]]):
    tensor_rows = manifest.get("tensors")
    if not isinstance(tensor_rows, list):
        raise RuntimeError("source manifest tensor rows missing")
    required = sorted({parse_key(key)[:2] for keys in keys_by_split.values() for key in keys})
    expected_required = sorted(
        (expert, role)
        for expert in SELECTION_EXPERTS
        for role in ("up", "down")
        if not (expert == 0 and role == "up")
    )
    if required != expected_required:
        raise RuntimeError("stage-1 identity set mismatch")
    cache = {}
    receipts = []
    for expert, role in required:
        rows = [row for row in tensor_rows if int(row.get("expert", -1)) == expert and row.get("role") == role]
        if len(rows) != 1:
            raise RuntimeError(f"manifest identity cardinality mismatch: {expert}/{role}")
        row = rows[0]
        path = workspace / row["local_path"]
        cache[(expert, role)] = bf16_matrix(np, path, row["sha256"], role)
        receipts.append({
            "expert": expert,
            "role": role,
            "relative_path": row["local_path"],
            "bytes": path.stat().st_size,
            "sha256": row["sha256"],
        })
    targets = {}
    descriptors = {}
    for split, keys in keys_by_split.items():
        parsed = [parse_key(key) for key in keys]
        values = np.asarray([cache[(e, role)][r, c] for e, role, r, c in parsed], dtype=np.float32)
        if values.shape != (2048,) or not bool(np.isfinite(values).all()):
            raise RuntimeError(f"target vector invariant failed: {split}")
        targets[split] = values
        descriptors[split] = parsed
    return targets, descriptors, receipts


def coordinate_arrays(np, descriptors):
    addends, sequences, offsets, normal4, lanes, roles = [], [], [], [], [], []
    for expert, role, row, column in descriptors:
        if role == "up":
            native = (row + 768) * 2048 + column
            offset_values = 11520 + 16 * (expert % 32)
            role_code = 0
        else:
            native = column * 768 + row
            offset_values = 12032 + 8 * (expert % 32)
            role_code = 1
        sequence = native % T
        quotient = native // T
        lane = quotient & 3
        index = quotient >> 2
        if sequence + T * (4 * index + lane) != native:
            raise RuntimeError("native coordinate inversion failed")
        addends.append(1024 + 100 * (expert // 32))
        sequences.append(sequence)
        offsets.append(offset_values // 4)
        normal4.append(index)
        lanes.append(lane)
        roles.append(role_code)
    return (
        np.asarray(addends, dtype=np.uint64),
        np.asarray(sequences, dtype=np.uint64),
        np.asarray(offsets, dtype=np.uint64),
        np.asarray(normal4, dtype=np.uint64),
        np.asarray(lanes, dtype=np.uint8),
        np.asarray(roles, dtype=np.uint8),
    )


def generate_anchors(cp, np, kernel, descriptors):
    arrays = coordinate_arrays(np, descriptors)
    device_arrays = [cp.asarray(value) for value in arrays]
    bases = cp.asarray(np.asarray(SEEDS, dtype=np.uint64))
    count = len(descriptors)
    output = cp.empty((len(SEEDS), count), dtype=cp.float32)
    total = output.size
    block = 256
    kernel(
        ((total + block - 1) // block,),
        (block,),
        (
            bases,
            *device_arrays,
            np.uint64(count),
            np.uint64(len(SEEDS)),
            output,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    if not bool(cp.isfinite(output).all().item()):
        raise RuntimeError("nonfinite generated anchor")
    return cp.asnumpy(output.T).astype(np.float64, copy=False)


def explicit_fit(np, x_fit, y_fit):
    x_mean = np.mean(x_fit, axis=0, dtype=np.float64)
    y_mean = float(np.mean(y_fit, dtype=np.float64))
    xc = x_fit - x_mean
    yc = y_fit - y_mean
    gram = xc.T @ xc
    rhs = xc.T @ yc
    ridge = math.ldexp(float(np.trace(gram)), -20) / len(SEEDS)
    system = gram + ridge * np.eye(len(SEEDS), dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(system)
    if not bool(np.isfinite(eigenvalues).all()) or float(eigenvalues[0]) <= 0.0:
        raise RuntimeError("nonpositive tetrad fit system")
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if condition > 2**20:
        raise RuntimeError(f"tetrad fit condition exceeds limit: {condition}")
    lower = np.linalg.cholesky(system)
    beta = np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))
    mu = y_mean - float(beta @ x_mean)
    words = np.asarray([*beta, mu], dtype=np.float16).view(np.uint16)
    if bool(np.any(words == np.uint16(0x8000))):
        raise RuntimeError("negative-zero FP16 coefficient")
    decoded = words.view(np.float16).astype(np.float64)
    if not bool(np.isfinite(decoded).all()):
        raise RuntimeError("nonfinite decoded FP16 coefficient")
    return decoded[:4], float(decoded[4]), words, condition, ridge


def evaluate(np, targets, descriptors, anchors, permutations=None):
    by_identity = sorted({row[:2] for rows in descriptors.values() for row in rows})
    split_indices = {
        split: {
            identity: np.asarray([i for i, row in enumerate(rows) if row[:2] == identity], dtype=np.int64)
            for identity in by_identity
        }
        for split, rows in descriptors.items()
    }
    records = []
    total_sse = total_energy = total_centered = 0.0
    for identity in by_identity:
        fit_index = split_indices["fit"][identity]
        score_index = split_indices["score"][identity]
        if min(fit_index.size, score_index.size) < 5:
            raise RuntimeError(f"insufficient stage-1 coordinates for {identity}")
        x_fit = anchors["fit"][fit_index]
        x_score = anchors["score"][score_index]
        if permutations is not None:
            x_fit = x_fit[permutations[(identity, "fit")]]
            x_score = x_score[permutations[(identity, "score")]]
        y_fit = targets["fit"][fit_index].astype(np.float64)
        y_score = targets["score"][score_index].astype(np.float64)
        beta, mu, words, condition, ridge = explicit_fit(np, x_fit, y_fit)
        reconstruction = mu + x_score @ beta
        error = y_score - reconstruction
        sse = float(np.sum(error * error, dtype=np.float64))
        energy = float(np.sum(y_score * y_score, dtype=np.float64))
        centered = float(np.sum((y_score - float(np.mean(y_fit))) ** 2, dtype=np.float64))
        if min(sse, energy, centered) < 0.0 or not all(math.isfinite(x) for x in (sse, energy, centered)):
            raise RuntimeError("invalid score accumulator")
        total_sse += sse
        total_energy += energy
        total_centered += centered
        records.append({
            "expert": identity[0],
            "role": identity[1],
            "fit_coordinates": int(fit_index.size),
            "score_coordinates": int(score_index.size),
            "fp16_words_hex": [format(int(word), "04x") for word in words],
            "condition": condition,
            "ridge": ridge,
            "sse": sse,
            "source_energy": energy,
            "centered_baseline_sse": centered,
            "raw_source_capture": 1.0 - sse / energy,
            "centered_capture": 1.0 - sse / centered,
        })
    return {
        "raw_source_capture": 1.0 - total_sse / total_energy,
        "centered_capture": 1.0 - total_sse / total_centered,
        "total_sse": total_sse,
        "total_source_energy": total_energy,
        "total_centered_baseline_sse": total_centered,
        "rows": records,
    }


def aggregate_role(records, role: str):
    rows = [row for row in records if row["role"] == role]
    sse = math.fsum(row["sse"] for row in rows)
    energy = math.fsum(row["source_energy"] for row in rows)
    return {"matrices": len(rows), "sse": sse, "source_energy": energy, "raw_source_capture": 1.0 - sse / energy}


def jackknife(np, records):
    total_sse = math.fsum(row["sse"] for row in records)
    total_energy = math.fsum(row["source_energy"] for row in records)
    values = np.asarray([
        1.0 - (total_sse - row["sse"]) / (total_energy - row["source_energy"])
        for row in records
    ], dtype=np.float64)
    mean = float(np.mean(values, dtype=np.float64))
    se = math.sqrt((len(values) - 1) / len(values) * float(np.sum((values - mean) ** 2, dtype=np.float64)))
    return {"delete_one_values": values.tolist(), "mean": mean, "standard_error": se, "three_se_upper": float(1.0 - total_sse / total_energy) + 3.0 * se}


def scramble_controls(np, targets, descriptors, anchors, replicates=16):
    identities = sorted({row[:2] for rows in descriptors.values() for row in rows})
    rows = []
    for replicate in range(replicates):
        random = np.random.Generator(np.random.PCG64(26090100 + replicate))
        permutations = {}
        for identity in identities:
            for split in ("fit", "score"):
                count = sum(row[:2] == identity for row in descriptors[split])
                permutations[(identity, split)] = random.permutation(count)
        result = evaluate(np, targets, descriptors, anchors, permutations)
        rows.append({
            "replicate": replicate,
            "seed": 26090100 + replicate,
            "raw_source_capture": result["raw_source_capture"],
            "centered_capture": result["centered_capture"],
        })
    values = np.asarray([row["raw_source_capture"] for row in rows], dtype=np.float64)
    return {
        "construction": "independent within-expert-role anchor-row permutations for fit and score",
        "rows": rows,
        "mean_raw_source_capture": float(np.mean(values, dtype=np.float64)),
        "mc_standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("output directory must be absent")
    workspace = Path(args.workspace).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest_raw = manifest_path.read_bytes()
    if sha256_bytes(manifest_raw) != MANIFEST_SHA256:
        raise RuntimeError("source manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    plan_path = Path(__file__).resolve().parents[1] / "fuseed_pmg1_direct_source_calibration_v0" / "plan.py"
    plan = load_plan(plan_path)
    fit_keys, score_keys = stage1_keys(plan)
    keys = {"fit": fit_keys, "score": score_keys}

    import cupy as cp
    import numpy as np

    started = time.perf_counter()
    targets, descriptors, source_receipts = load_targets(np, workspace, manifest, keys)
    module = cp.RawModule(
        code=CUDA_SOURCE,
        options=("--std=c++17", "-I/usr/local/cuda/include"),
        backend="nvrtc",
        name_expressions=("pmg_anchor_coords",),
    )
    kernel = module.get_function("pmg_anchor_coords")
    anchors = {
        split: generate_anchors(cp, np, kernel, descriptors[split])
        for split in ("fit", "score")
    }
    primary = evaluate(np, targets, descriptors, anchors)
    controls = scramble_controls(np, targets, descriptors, anchors)
    uncertainty = jackknife(np, primary["rows"])
    roles = {role: aggregate_role(primary["rows"], role) for role in ("up", "down")}
    raw_capture = primary["raw_source_capture"]
    promoted = raw_capture >= PLANNING_CAPTURE and all(row["raw_source_capture"] > 0.0 for row in roles.values())
    decision = "PROMOTE_TO_STAGE2_AUXILIARY_ONLY" if promoted else "POLICY_REJECT_INCONCLUSIVE_FAR_SHORT_STOP_BEFORE_STAGE2"

    result = {
        "schema": "fuseed_pmg1_tetrad_aux_stage1_v0_result",
        "status": decision,
        "claim_boundary": (
            "Disjoint-coordinate development-panel falsification only; not Gate evidence, fresh validation, "
            "a rebuilt residual-codec score, a finite compression result, or target achievement."
        ),
        "fixed_hypothesis": {
            "seeds_u32": list(SEEDS),
            "fit_coordinates": len(fit_keys),
            "score_coordinates": len(score_keys),
            "decoded_fp16_coefficients": True,
            "planning_capture_not_converse": PLANNING_CAPTURE,
        },
        "primary": primary,
        "role_aggregates": roles,
        "delete_one_matrix_uncertainty": uncertainty,
        "scramble_controls": controls,
        "diagnostics": {
            "raw_minus_mean_scramble_capture": raw_capture - controls["mean_raw_source_capture"],
            "raw_fraction_of_planning_capture": raw_capture / PLANNING_CAPTURE,
            "three_se_upper_fraction_of_planning_capture": uncertainty["three_se_upper"] / PLANNING_CAPTURE,
        },
        "bindings": {
            "manifest_sha256": MANIFEST_SHA256,
            "plan_module_sha256": PLAN_SHA256,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "cuda_source_sha256": sha256_bytes(CUDA_SOURCE.encode("utf-8")),
            "fit_key_sha256": sha256_bytes(("\n".join(fit_keys) + "\n").encode("ascii")),
            "score_key_sha256": sha256_bytes(("\n".join(score_keys) + "\n").encode("ascii")),
            "source_receipts": source_receipts,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "compute_capability": cp.cuda.Device().compute_capability,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "access": {
            "selection_up_down_files_opened": len(source_receipts),
            "gate_files_opened": 0,
            "old_validation_files_opened": 0,
            "fresh_validation_files_opened": 0,
            "pinned_panel_files_opened": 0,
            "network_operations": 0,
        },
    }
    output.mkdir(parents=False, exist_ok=False)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": decision,
        "raw_source_capture": raw_capture,
        "centered_capture": primary["centered_capture"],
        "raw_minus_mean_scramble_capture": result["diagnostics"]["raw_minus_mean_scramble_capture"],
        "three_se_upper": uncertainty["three_se_upper"],
        "planning_capture": PLANNING_CAPTURE,
        "result_sha256": sha256_file(result_path),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

