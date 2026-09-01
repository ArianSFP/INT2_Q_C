#!/usr/bin/env python3
"""CuPy source-panel stage-0 oracle for QSB-PTQ-v0.

This is not a channel-simulation encoder and emits no compressed artifact.
It grants ideal KL communication, an unquantized decoder fitted on the scored
matrix itself, and the best of three frozen shared-randomness seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
from pathlib import Path


PLAN_SHA256 = "8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868"
PLAN_LOCK_SHA256 = "99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d"
EXPERTS, ROLES, ROWS, COLS = 6, 3, 768, 2048
MATRIX_VALUES, EXPERT_VALUES = ROWS * COLS, ROLES * ROWS * COLS
PANEL_VALUES = EXPERTS * EXPERT_VALUES
BLOCK, BLOCKS_PER_MATRIX, BLOCKS_PER_EXPERT = 64, MATRIX_VALUES // 64, EXPERT_VALUES // 64
LATENTS, RAW_FEATURES, INTERACTIONS, FEATURES = 160, 160, 96, 256
FIT_EXPERTS, CALIBRATION_EXPERTS, CLOSED_SCORE_EXPERTS = (0, 2, 4), (1,), (3, 5)
PROJECTION_SEED = 14592251004518932763
PAIR_SEED = 10058181636442808937
COMMON_SEEDS = (16443857425729824865, 6983438078262162903, 11299122902407625677)
CONTROL_SEEDS = tuple(range(5850734194750267521, 5850734194750267529))
BATCH = 2048
KL_FILL = 0.97
ROLE_ORDER = ("gate", "up", "down")
RATE_CELLS = (
    {"cell": "QSB215", "pages": 309, "container_bytes": 7618560,
     "payload_bytes_per_expert": 1261568, "physical_bpw": 155.0 / 72.0,
     "payload_bpw": 77.0 / 36.0, "metadata_bpw": 1.0 / 72.0,
     "cold_read_amplification": 63.0 / 62.0},
    {"cell": "QSB230", "pages": 331, "container_bytes": 8159232,
     "payload_bytes_per_expert": 1351680, "physical_bpw": 83.0 / 36.0,
     "payload_bpw": 55.0 / 24.0, "metadata_bpw": 1.0 / 72.0,
     "cold_read_amplification": 337.0 / 332.0},
    {"cell": "QSB250", "pages": 359, "container_bytes": 8847360,
     "payload_bytes_per_expert": 1466368, "physical_bpw": 2.5,
     "payload_bpw": 179.0 / 72.0, "metadata_bpw": 1.0 / 72.0,
     "cold_read_amplification": 73.0 / 72.0},
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def held_regular_read(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("input is not a regular non-link file: " + str(path))
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise ValueError("input identity changed before held read: " + str(path))
        chunks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block: break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("input changed during held read: " + str(path))
        raw = b"".join(chunks)
        if len(raw) != opened.st_size:
            raise ValueError("short held read: " + str(path))
        return raw
    finally:
        os.close(descriptor)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def strict_json(raw: bytes):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("duplicate JSON key " + key)
            out[key] = value
        return out
    def finite(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError("nonfinite JSON")
        return value
    def constant(text): raise ValueError("nonfinite JSON " + text)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=constant)


def validate_plan(plan, bindings):
    clean = dict(plan)
    claimed = clean.pop("lock_sha256", None)
    if claimed != PLAN_LOCK_SHA256 or sha256_bytes(canonical(clean)) != claimed:
        raise ValueError("plan internal seal")
    if plan.get("schema") != bindings["plan"]["expected_schema"]:
        raise ValueError("plan schema")
    if len(plan.get("sources", [])) != 18 or len(bindings.get("sources", [])) != 18:
        raise ValueError("source count")
    for ordinal, (source, bound) in enumerate(zip(plan["sources"], bindings["sources"])):
        role = ROLE_ORDER[ordinal % 3]
        expected_shape = [2048, 768] if role == "down" else [768, 2048]
        if source["matrix_ordinal"] != ordinal or bound["matrix_ordinal"] != ordinal:
            raise ValueError("matrix ordinal")
        if source["role"] != role or bound["role"] != role:
            raise ValueError("role order")
        if source["shape"] != expected_shape or bound["shape"] != expected_shape:
            raise ValueError("source shape")
        if source["source_relpath"] != bound["source_relpath"]:
            raise ValueError("source relative path")
        if source["source_bf16_sha256"] != bound["sha256"] or bound["bytes"] != 3145728:
            raise ValueError("source binding")
        relative = Path(bound["source_relpath"])
        if relative.is_absolute() or ".." in relative.parts or "validation" in str(relative).lower():
            raise ValueError("unsafe or validation source path")


def load_matrix(np, path: Path, role: str, expected_sha: str):
    raw_bytes = held_regular_read(path)
    if len(raw_bytes) != 3145728 or sha256_bytes(raw_bytes) != expected_sha:
        raise ValueError("source bytes/hash: " + str(path))
    words = np.frombuffer(raw_bytes, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    raw = values.reshape((2048, 768) if role == "down" else (768, 2048))
    matrix = raw.T.copy() if role == "down" else raw.copy()
    if matrix.shape != (768, 2048) or not bool(np.isfinite(matrix).all()):
        raise ValueError("source finite geometry")
    return matrix.reshape(BLOCKS_PER_MATRIX, BLOCK)


def load_panel(np, plan, bindings, source_root: Path):
    matrices, receipts = [], []
    for source, bound in zip(plan["sources"], bindings["sources"]):
        path = source_root / bound["source_relpath"]
        blocks = load_matrix(np, path, bound["role"], bound["sha256"])
        matrices.append(blocks)
        receipts.append({"matrix_ordinal": bound["matrix_ordinal"], "relative_path": bound["source_relpath"],
                         "bytes": bound["bytes"], "sha256": bound["sha256"]})
    return matrices, receipts


def frozen_maps(np):
    random = np.random.Generator(np.random.PCG64(PROJECTION_SEED))
    projection = random.integers(0, 2, size=(LATENTS, BLOCK), dtype=np.int8)
    projection = (2 * projection.astype(np.float64) - 1.0) / math.sqrt(BLOCK)
    pair_random = np.random.Generator(np.random.PCG64(PAIR_SEED))
    pairs = []
    while len(pairs) < INTERACTIONS:
        left, right = map(int, pair_random.integers(0, LATENTS, size=2))
        if left != right and (left, right) not in pairs:
            pairs.append((left, right))
    return projection, np.asarray(pairs, dtype=np.int64)


def normalized_projection(cp, blocks, projection):
    x = cp.asarray(blocks, dtype=cp.float64)
    centered = x - cp.mean(x, axis=1, keepdims=True, dtype=cp.float64)
    scale = cp.sqrt(cp.mean(centered * centered, axis=1, keepdims=True,
                            dtype=cp.float64) + math.ldexp(1.0, -24))
    return cp.matmul(centered / scale, projection.T)


def q_probabilities(cp, blocks, projection, alpha):
    logits = alpha * normalized_projection(cp, blocks, projection)
    return 1.0 / (1.0 + cp.exp(-cp.clip(logits, -60.0, 60.0)))


def iter_fit_matrices(matrices):
    for expert in FIT_EXPERTS:
        for role in range(3):
            yield matrices[3 * expert + role]


def fit_alpha(cp, matrices, projection, target_bits_per_block):
    fit_logits = cp.concatenate([
        normalized_projection(cp, blocks[begin:begin + BATCH], projection)
        for blocks in iter_fit_matrices(matrices)
        for begin in range(0, len(blocks), BATCH)
    ], axis=0)
    floor = math.ldexp(1.0, -24)
    log2 = math.log(2.0)
    def evaluate(alpha):
        q = cp.clip(1.0 / (1.0 + cp.exp(-cp.clip(alpha * fit_logits, -60.0, 60.0))),
                    floor, 1.0 - floor)
        prior = cp.clip(cp.mean(q, axis=0, dtype=cp.float64), floor, 1.0 - floor)
        kl = q * cp.log(q / prior) + (1.0 - q) * cp.log((1.0 - q) / (1.0 - prior))
        return float((cp.sum(kl, dtype=cp.float64) / (len(q) * log2)).item()), prior
    low, high = 0.0, 64.0
    high_kl, _ = evaluate(high)
    if high_kl < target_bits_per_block:
        raise RuntimeError("frozen encoder cannot reach ideal KL target")
    for _ in range(48):
        middle = (low + high) * 0.5
        value, _ = evaluate(middle)
        if value < target_bits_per_block: low = middle
        else: high = middle
    value, prior = evaluate((low + high) * 0.5)
    del fit_logits
    return (low + high) * 0.5, value, prior


def expert_kl(cp, matrices, projection, alpha, prior):
    values = []
    log2 = math.log(2.0)
    for expert in range(EXPERTS):
        total = cp.float64(0.0)
        for role in range(3):
            blocks = matrices[3 * expert + role]
            for begin in range(0, len(blocks), BATCH):
                q = cp.clip(q_probabilities(cp, blocks[begin:begin + BATCH], projection, alpha),
                            math.ldexp(1.0, -24), 1.0 - math.ldexp(1.0, -24))
                kl = q * cp.log(q / prior) + (1.0 - q) * cp.log((1.0 - q) / (1.0 - prior))
                total += cp.sum(kl, dtype=cp.float64) / log2
        values.append(float(total.item()))
    return values


def oracle_matrix(cp, np, blocks, projection, pairs, alpha, common_seed, matrix_ordinal):
    gram = cp.zeros((FEATURES + 1, FEATURES + 1), dtype=cp.float64)
    cross = cp.zeros((FEATURES + 1, BLOCK), dtype=cp.float64)
    energy = 0.0
    random = np.random.Generator(np.random.PCG64(common_seed ^ (matrix_ordinal * 0x9E3779B97F4A7C15)))
    pair_left, pair_right = cp.asarray(pairs[:, 0]), cp.asarray(pairs[:, 1])
    for begin in range(0, len(blocks), BATCH):
        host = blocks[begin:begin + BATCH]
        q = q_probabilities(cp, host, projection, alpha)
        uniform = cp.asarray(random.random(q.shape, dtype=np.float64))
        bipolar = cp.where(uniform < q, 1.0, -1.0)
        interactions = bipolar[:, pair_left] * bipolar[:, pair_right]
        features = cp.concatenate((bipolar, interactions), axis=1)
        augmented = cp.concatenate((cp.ones((len(host), 1), dtype=cp.float64), features), axis=1)
        target = cp.asarray(host, dtype=cp.float64)
        gram += augmented.T @ augmented
        cross += augmented.T @ target
        energy += float(cp.sum(target * target, dtype=cp.float64).item())
    eigenvalues, eigenvectors = cp.linalg.eigh(gram)
    cutoff = float(eigenvalues[-1].item()) * math.ldexp(1.0, -40)
    inverse = cp.where(eigenvalues > cutoff, 1.0 / eigenvalues, 0.0)
    weights = (eigenvectors * inverse[None, :]) @ (eigenvectors.T @ cross)
    fitted = float(cp.sum(weights * cross, dtype=cp.float64).item())
    sse = max(0.0, energy - fitted)
    rank = int(cp.count_nonzero(eigenvalues > cutoff).item())
    return {"sse": sse, "source_energy": energy, "capture": 1.0 - sse / energy,
            "feature_rank": rank, "common_seed_u64": common_seed}


def jackknife(expert_sse, expert_energy):
    total_sse, total_energy = math.fsum(expert_sse), math.fsum(expert_energy)
    estimate = 1.0 - total_sse / total_energy
    deletes = [1.0 - (total_sse - expert_sse[i]) / (total_energy - expert_energy[i])
               for i in range(EXPERTS)]
    center = math.fsum(deletes) / EXPERTS
    se = math.sqrt((EXPERTS - 1) / EXPERTS * math.fsum((x - center) ** 2 for x in deletes))
    return {"estimate": estimate, "delete_one_expert": deletes, "jackknife_center": center,
            "jackknife_se": se, "lower_three_se": estimate - 3.0 * se,
            "upper_three_se": estimate + 3.0 * se}


def score_oracle(cp, np, matrices, projection, pairs, alpha, cell, prior):
    payload_bits = cell["payload_bytes_per_expert"] * 8
    kls = expert_kl(cp, matrices, projection, alpha, prior)
    if any(value > KL_FILL * payload_bits + 2e-7 for value in kls):
        return {"status": "HARD_KILL_IDEAL_KL_EXCEEDS_FROZEN_RESERVOIR",
                "ideal_kl_bits_by_expert": kls, "reservoir_bits_by_expert": payload_bits}
    matrix_rows, expert_sse, expert_energy = [], [0.0] * 6, [0.0] * 6
    role_sse, role_energy = [0.0] * 3, [0.0] * 3
    for ordinal, blocks in enumerate(matrices):
        candidates = [oracle_matrix(cp, np, blocks, projection, pairs, alpha, seed, ordinal)
                      for seed in COMMON_SEEDS]
        best = min(candidates, key=lambda row: (row["sse"], row["common_seed_u64"]))
        expert, role = ordinal // 3, ordinal % 3
        expert_sse[expert] += best["sse"]; expert_energy[expert] += best["source_energy"]
        role_sse[role] += best["sse"]; role_energy[role] += best["source_energy"]
        matrix_rows.append({"matrix_ordinal": ordinal, "expert_ordinal": expert,
                            "role": ROLE_ORDER[role], **best})
        print(f"{cell['cell']} matrix {ordinal:02d}/17 oracle_capture={best['capture']:.9f}", flush=True)
    uncertainty = jackknife(expert_sse, expert_energy)
    required = 1.0 - 0.8 * 2.0 ** (-2.0 * cell["physical_bpw"])
    result = {"status": "ORACLE_COMPLETE", "matrix_rows": matrix_rows,
              "ideal_kl_bits_by_expert": kls, "reservoir_bits_by_expert": payload_bits,
              "source_energy": math.fsum(expert_energy), "sse": math.fsum(expert_sse),
              "capture": uncertainty["estimate"], "uncertainty": uncertainty,
              "required_capture": required,
              "experts": [{"expert_ordinal": i, "sse": expert_sse[i],
                           "source_energy": expert_energy[i],
                           "capture": 1.0 - expert_sse[i] / expert_energy[i]} for i in range(6)],
              "roles": [{"role": ROLE_ORDER[i], "sse": role_sse[i],
                         "source_energy": role_energy[i],
                         "capture": 1.0 - role_sse[i] / role_energy[i]} for i in range(3)]}
    if uncertainty["upper_three_se"] < required:
        result["status"] = "HARD_KILL_FAVOURABLE_ORACLE_UCB_BELOW_EXACT_REQUIREMENT"
    else:
        result["status"] = "ORACLE_SURVIVOR_REQUIRES_MATCHED_CONTROLS"
    return result


def matched_gaussian_panel(np, matrices, seed):
    controls = []
    for ordinal, source in enumerate(matrices):
        random = np.random.Generator(np.random.PCG64(seed ^ (ordinal * 0xD1B54A32D192ED03)))
        values = random.standard_normal(source.size, dtype=np.float64)
        values -= float(np.mean(values, dtype=np.float64))
        source64 = source.astype(np.float64, copy=False)
        source_mean = float(np.mean(source64, dtype=np.float64))
        target_centered = float(np.sum((source64 - source_mean) ** 2, dtype=np.float64))
        values *= math.sqrt(target_centered / float(np.sum(values * values, dtype=np.float64)))
        values += source_mean
        observed_mean = float(np.mean(values, dtype=np.float64))
        observed_centered = float(np.sum((values - observed_mean) ** 2, dtype=np.float64))
        rms = math.sqrt(target_centered / source.size)
        if abs(observed_mean - source_mean) > max(math.ldexp(1.0, -45), rms * math.ldexp(1.0, -40)):
            raise RuntimeError("matched Gaussian mean tolerance")
        if abs(observed_centered - target_centered) > target_centered * math.ldexp(1.0, -40):
            raise RuntimeError("matched Gaussian centered-energy tolerance")
        # Keep FP64: the construction then matches the source mean and centered
        # energy to FP64 arithmetic instead of perturbing both by an FP32 cast.
        controls.append(values.reshape(source.shape))
    return controls


def run_controls(cp, np, matrices, projection, pairs, cell):
    rows = []
    target = KL_FILL * cell["payload_bytes_per_expert"] * 8 / BLOCKS_PER_EXPERT
    for replicate, seed in enumerate(CONTROL_SEEDS):
        controls = matched_gaussian_panel(np, matrices, seed)
        alpha, fit_kl, prior = fit_alpha(cp, controls, projection, target)
        scored = score_oracle(cp, np, controls, projection, pairs, alpha, cell, prior)
        rows.append({"replicate": replicate, "seed_u64": seed, "alpha": alpha,
                     "fit_mean_ideal_kl_bits_per_block": fit_kl, "oracle": scored})
    return rows


def apply_control_gate(qwen, controls):
    if any("uncertainty" not in row["oracle"] for row in controls):
        return {"status": "POLICY_REJECT_CONTROL_RATE_INCOMPARABLE",
                "reason": "at least one matched control failed before a comparable decoder oracle"}
    qwen_lower = qwen["uncertainty"]["lower_three_se"]
    strongest_upper = max(row["oracle"]["uncertainty"]["upper_three_se"] for row in controls)
    expert_gains = [
        qwen["experts"][i]["capture"] - max(row["oracle"]["experts"][i]["capture"]
                                                   for row in controls)
        for i in range(EXPERTS)
    ]
    role_gains = [
        qwen["roles"][i]["capture"] - max(row["oracle"]["roles"][i]["capture"]
                                                for row in controls)
        for i in range(ROLES)
    ]
    aggregate_margin = qwen_lower - strongest_upper
    promoted = aggregate_margin > 0.0 and min(expert_gains) > 0.0 and min(role_gains) > 0.0
    return {"status": ("POLICY_HOLD_FOR_OPERATIONAL_IMPLEMENTATION" if promoted else
                       "POLICY_REJECT_SOURCE_NOT_ABOVE_MATCHED_CONTROLS"),
            "qwen_lower_three_se": qwen_lower,
            "strongest_control_upper_three_se": strongest_upper,
            "aggregate_margin": aggregate_margin,
            "expert_capture_gains_over_strongest_same_fold_control": expert_gains,
            "role_capture_gains_over_strongest_same_fold_control": role_gains}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0": raise RuntimeError("CUDA_VISIBLE_DEVICES must be 0")
    output = args.output.resolve()
    if output.exists(): raise FileExistsError("output must be absent")
    plan_path, source_root = args.plan.resolve(strict=True), args.source_root.resolve(strict=True)
    binding_path = Path(__file__).resolve().parent / "panel_bindings.json"
    design_path = Path(__file__).resolve().parent / "design_lock.json"
    script_raw = held_regular_read(Path(__file__).resolve())
    plan_raw = held_regular_read(plan_path)
    binding_raw = held_regular_read(binding_path)
    design_raw = held_regular_read(design_path)
    if sha256_bytes(plan_raw) != PLAN_SHA256: raise ValueError("plan hash")
    plan, bindings, design = strict_json(plan_raw), strict_json(binding_raw), strict_json(design_raw)
    validate_plan(plan, bindings)
    import cupy as cp
    import numpy as np
    started = time.time()
    projection_host, pairs = frozen_maps(np)
    projection = cp.asarray(projection_host)
    matrices, source_receipts = load_panel(np, plan, bindings, source_root)
    cells = []
    for cell in RATE_CELLS:
        target = KL_FILL * cell["payload_bytes_per_expert"] * 8 / BLOCKS_PER_EXPERT
        alpha, fit_kl, prior = fit_alpha(cp, matrices, projection, target)
        oracle = score_oracle(cp, np, matrices, projection, pairs, alpha, cell, prior)
        controls = []
        control_gate = {"status": "NOT_RUN_ORACLE_DID_NOT_SURVIVE"}
        if oracle["status"] == "ORACLE_SURVIVOR_REQUIRES_MATCHED_CONTROLS":
            controls = run_controls(cp, np, matrices, projection, pairs, cell)
            control_gate = apply_control_gate(oracle, controls)
        cells.append({"cell": cell, "alpha": alpha,
                      "fit_mean_ideal_kl_bits_per_block": fit_kl,
                      "prior_sha256_f64le": sha256_bytes(cp.asnumpy(prior).astype("<f8").tobytes()),
                      "qwen_oracle": oracle, "matched_gaussian_controls": controls,
                      "control_gate": control_gate})
    survivors = [row for row in cells if row["control_gate"]["status"] ==
                 "POLICY_HOLD_FOR_OPERATIONAL_IMPLEMENTATION"]
    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device_name, bytes): device_name = device_name.decode("utf-8")
    result = {
        "schema": "qwen_stochastic_binary_channel_ptq_stage0_result_v0",
        "status": ("POLICY_REJECT_ALL_RATE_CELLS" if not survivors else
                   "POLICY_HOLD_FOR_SEPARATE_OPERATIONAL_IMPLEMENTATION"),
        "cells": cells,
        "bindings": {"plan_sha256": PLAN_SHA256, "plan_internal_lock_sha256": PLAN_LOCK_SHA256,
                     "design_lock_sha256": sha256_bytes(design_raw),
                     "panel_bindings_sha256": sha256_bytes(binding_raw),
                     "stage0_script_sha256": sha256_bytes(script_raw),
                     "sources": source_receipts},
        "fixed_splits": {"fit_experts": list(FIT_EXPERTS),
                         "calibration_experts": list(CALIBRATION_EXPERTS),
                         "closed_score_experts": list(CLOSED_SCORE_EXPERTS)},
        "claim_boundary": "Optimistic source-leaking stage-0 topology bound only; not a serialized channel simulator, operational Gaussian/TCQ result, Shannon-limit result, compressed artifact, or validation result.",
        "execution": {"python": sys.version.split()[0], "numpy": np.__version__, "cupy": cp.__version__,
                      "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                      "device": str(device_name),
                      "elapsed_seconds": time.time() - started},
        "access": {"authenticated_panel_sources_opened": 18, "fresh_validation_files_opened": 0,
                   "compressed_outputs_created": 0, "network_operations": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(output, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": result["status"], "result_sha256": sha256_bytes(encoded),
                      "elapsed_seconds": result["execution"]["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
