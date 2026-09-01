#!/usr/bin/env python3
"""Bounded nonlinear conditional affine-flow gate.

A real two-hidden-layer MLP predicts the mean and log variance of a weight from
32 continuous long-range neighbours.  It is trained only on sampled layer-15
auxiliary experts, validated on four untouched auxiliary experts, and is
permitted to open the pinned six-expert panel only if validation gains at least
0.05 bpw.  An identically trained moment-matched iid-Gaussian control measures
finite-sample/training optimism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path

import numpy as np


ROWS = 768
COLS = 2048
N = ROWS * COLS
STRATA = 8
INPUT = 32
HIDDEN1 = 32
HIDDEN2 = 16
VALIDATION_GATE_BPW = 0.05
REQUIRED_S_BPW = -0.5 * math.log2(0.8)
AUX_RE = re.compile(r"l15e(?P<expert>\d+)_(?P<role>up|down)\.bf16\.bin$")
TARGET_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<role>gate|up|down)_proj\.weight\.bf16\.bin$"
)
HEADER = struct.Struct("<8sIIIIII32s")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def load_bf16(path: Path, role: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != N:
        raise ValueError(f"{path}: {raw.size}, expected {N}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if role == "down":
        return np.asarray(values.reshape(COLS, ROWS).T, dtype=np.float32)
    return np.asarray(values.reshape(ROWS, COLS), dtype=np.float32)


def stratum_standardize(matrix: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    energy = np.sum(matrix.astype(np.float64) ** 2, axis=1)
    order = np.lexsort((np.arange(ROWS), energy))
    rank = np.empty_like(order)
    rank[order] = np.arange(ROWS)
    labels = np.minimum(STRATA - 1, rank * STRATA // ROWS)
    result = matrix.copy()
    cells = []
    for stratum in range(STRATA):
        select = labels == stratum
        values = result[select]
        mean = float(np.mean(values, dtype=np.float64))
        std = float(np.std(values, dtype=np.float64))
        result[select] = (values - mean) / std
        cells.append({"stratum": stratum, "rows": int(select.sum()), "mean": mean, "std": std})
    return result, cells


def samples(path: Path, role: str, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    matrix, cells = stratum_standardize(load_bf16(path, role))
    rng = np.random.default_rng(seed)
    flat = rng.choice(256 * 512, size=count, replace=False)
    rr = 256 + flat // 512
    cc = 1024 + flat % 512
    columns = []
    for offset in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        columns.append(matrix[rr, cc - offset])
    for offset in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        columns.append(matrix[rr - offset, cc])
    for offset in (1, 8, 64, 512):
        columns.append(matrix[rr, cc + offset])
    for offset in (1, 8, 64, 256):
        columns.append(matrix[rr + offset, cc])
    for dr, dc in ((1, 1), (1, -1), (8, 8), (8, -8)):
        columns.append(matrix[rr - dr, cc + dc])
    x = np.stack(columns, axis=1).astype(np.float32)
    if x.shape[1] != INPUT:
        raise AssertionError(x.shape)
    return x, matrix[rr, cc].astype(np.float32), cells


def init_model(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    def weight(a: int, b: int) -> np.ndarray:
        return rng.normal(0.0, math.sqrt(2.0 / (a + b)), size=(a, b)).astype(np.float64)
    return {
        "w1": weight(INPUT, HIDDEN1), "b1": np.zeros(HIDDEN1),
        "w2": weight(HIDDEN1, HIDDEN2), "b2": np.zeros(HIDDEN2),
        "w3": weight(HIDDEN2, 2), "b3": np.zeros(2),
    }


def forward(model: dict[str, np.ndarray], x: np.ndarray) -> tuple:
    h1 = np.tanh(x @ model["w1"] + model["b1"])
    h2 = np.tanh(h1 @ model["w2"] + model["b2"])
    output = h2 @ model["w3"] + model["b3"]
    mean = output[:, 0]
    scaled = output[:, 1] / 1.5
    tanh_scale = np.tanh(scaled)
    logvar = 1.5 * tanh_scale
    return h1, h2, mean, logvar, tanh_scale


def gain_bpw(model: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, chunk: int = 8192) -> float:
    total = 0.0
    for start in range(0, len(y), chunk):
        xx = x[start : start + chunk].astype(np.float64)
        yy = y[start : start + chunk].astype(np.float64)
        _, _, mean, logvar, _ = forward(model, xx)
        candidate = -0.5 * (math.log(2.0 * math.pi) + logvar + (yy - mean) ** 2 * np.exp(-logvar))
        baseline = -0.5 * (math.log(2.0 * math.pi) + yy * yy)
        total += float(np.sum((candidate - baseline) / math.log(2.0)))
    return total / len(y)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    max_steps: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict], int, np.ndarray, np.ndarray]:
    feature_mean = np.mean(x_train, axis=0, dtype=np.float64)
    feature_std = np.maximum(np.std(x_train, axis=0, dtype=np.float64), 1e-4)
    train_x = ((x_train.astype(np.float64) - feature_mean) / feature_std).astype(np.float64)
    valid_x = ((x_valid.astype(np.float64) - feature_mean) / feature_std).astype(np.float64)
    model = init_model(seed)
    moments = {name: np.zeros_like(value) for name, value in model.items()}
    velocities = {name: np.zeros_like(value) for name, value in model.items()}
    rng = np.random.default_rng(seed + 1)
    history = []
    best_gain = -math.inf
    best = {name: value.copy() for name, value in model.items()}
    stopped = max_steps
    for step in range(1, max_steps + 1):
        index = rng.integers(0, len(y_train), size=batch_size)
        xb = train_x[index]
        yb = y_train[index].astype(np.float64)
        h1, h2, mean, logvar, tanh_scale = forward(model, xb)
        invvar = np.exp(-logvar)
        invn = 1.0 / batch_size
        grad_output = np.empty((batch_size, 2), dtype=np.float64)
        grad_output[:, 0] = (mean - yb) * invvar * invn
        grad_logvar = 0.5 * (1.0 - (yb - mean) ** 2 * invvar) * invn
        grad_output[:, 1] = grad_logvar * (1.0 - tanh_scale * tanh_scale)
        grads = {}
        grads["w3"] = h2.T @ grad_output
        grads["b3"] = np.sum(grad_output, axis=0)
        gh2 = (grad_output @ model["w3"].T) * (1.0 - h2 * h2)
        grads["w2"] = h1.T @ gh2
        grads["b2"] = np.sum(gh2, axis=0)
        gh1 = (gh2 @ model["w2"].T) * (1.0 - h1 * h1)
        grads["w1"] = xb.T @ gh1
        grads["b1"] = np.sum(gh1, axis=0)
        # Adam, fully deterministic.
        learning_rate = 2e-3
        for name in model:
            moments[name] = 0.9 * moments[name] + 0.1 * grads[name]
            velocities[name] = 0.999 * velocities[name] + 0.001 * grads[name] * grads[name]
            mhat = moments[name] / (1.0 - 0.9**step)
            vhat = velocities[name] / (1.0 - 0.999**step)
            model[name] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
        if step % 20 == 0 or step == 1:
            validation_gain = gain_bpw(model, valid_x, y_valid)
            training_gain = gain_bpw(model, train_x[: min(32768, len(train_x))], y_train[: min(32768, len(y_train))])
            history.append({"step": step, "training_gain_bpw": training_gain, "validation_gain_bpw": validation_gain})
            if validation_gain > best_gain:
                best_gain = validation_gain
                best = {name: value.copy() for name, value in model.items()}
            # Bounded early kill.  A branch below 0.05 after 80 updates is not
            # allowed to consume the remaining fit budget.
            if step >= 80 and best_gain < VALIDATION_GATE_BPW:
                stopped = step
                break
    return best, history, stopped, feature_mean, feature_std


def serialize_model(path: Path, model: dict[str, np.ndarray], feature_mean: np.ndarray, feature_std: np.ndarray) -> dict:
    arrays = [feature_mean, feature_std, model["w1"], model["b1"], model["w2"], model["b2"], model["w3"], model["b3"]]
    values = np.concatenate([np.asarray(array).reshape(-1) for array in arrays]).astype("<f2")
    header = HEADER.pack(b"NLAFV1\0\0", 1, INPUT, HIDDEN1, HIDDEN2, len(values), 16, b"\0" * 32)
    if len(header) != 64:
        raise AssertionError(len(header))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(values.tobytes())
    return {"bytes": path.stat().st_size, "fp16_values": len(values), "sha256": sha256_file(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux-dir", type=Path, required=True)
    ap.add_argument("--target-dir", type=Path, required=True)
    ap.add_argument("--target-lock", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--samples-per-matrix", type=int, default=4096)
    ap.add_argument("--target-samples-per-matrix", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=260901)
    args = ap.parse_args()
    started = time.time()
    lock = json.loads(args.target_lock.read_text(encoding="utf-8"))
    target_hashes = {
        f"L{row['layer']}_E{row['expert']}_{row['role']}": row["source_bf16_sha256"]
        for row in lock["matrices"]
    }

    files: dict[int, dict[str, Path]] = {}
    for path in sorted(args.aux_dir.glob("*.bf16.bin")):
        match = AUX_RE.fullmatch(path.name)
        if match:
            files.setdefault(int(match.group("expert")), {})[match.group("role")] = path
    experts = sorted(files)
    if len(experts) != 16 or any(set(files[e]) != {"up", "down"} for e in experts):
        raise RuntimeError({expert: sorted(roles) for expert, roles in files.items()})
    valid_experts = experts[3::4]
    train_experts = [expert for expert in experts if expert not in valid_experts]

    train_x, train_y, valid_x, valid_y = [], [], [], []
    source_hashes = {}
    normalization = []
    for expert in experts:
        for role_index, role in enumerate(("up", "down")):
            path = files[expert][role]
            x, y, cells = samples(path, role, args.samples_per_matrix, args.seed + expert * 101 + role_index)
            target_x, target_y = (valid_x, valid_y) if expert in valid_experts else (train_x, train_y)
            target_x.append(x)
            target_y.append(y)
            source_hashes[path.name] = sha256_file(path)
            normalization.append({"expert": expert, "role": role, "split": "validation" if expert in valid_experts else "train", "cells": cells})
    train_xa, train_ya = np.concatenate(train_x), np.concatenate(train_y)
    valid_xa, valid_ya = np.concatenate(valid_x), np.concatenate(valid_y)

    real_model, real_history, real_stop, feature_mean, feature_std = train_model(
        train_xa, train_ya, valid_xa, valid_ya, args.seed, args.max_steps, args.batch_size
    )
    real_gain = gain_bpw(real_model, (valid_xa - feature_mean) / feature_std, valid_ya)

    # Identical moment-matched Gaussian training/control, including shapes,
    # optimizer, initialization, updates and stopping rule.
    control_rng = np.random.default_rng(args.seed + 777)
    control_train_x = control_rng.standard_normal(train_xa.shape).astype(np.float32)
    control_train_y = control_rng.standard_normal(train_ya.shape).astype(np.float32)
    control_valid_x = control_rng.standard_normal(valid_xa.shape).astype(np.float32)
    control_valid_y = control_rng.standard_normal(valid_ya.shape).astype(np.float32)
    for array in (control_train_x, control_valid_x):
        array -= np.mean(array, axis=0, keepdims=True)
        array /= np.std(array, axis=0, keepdims=True)
    for array in (control_train_y, control_valid_y):
        array -= np.mean(array)
        array /= np.std(array)
    control_model, control_history, control_stop, control_mean, control_std = train_model(
        control_train_x, control_train_y, control_valid_x, control_valid_y,
        args.seed, args.max_steps, args.batch_size,
    )
    control_gain = gain_bpw(control_model, (control_valid_x - control_mean) / control_std, control_valid_y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_info = serialize_model(args.output_dir / "nonlinear_affine_flow.fp16.bin", real_model, feature_mean, feature_std)
    passes = real_gain >= VALIDATION_GATE_BPW
    target_evaluation = []
    opened = False
    if passes:
        opened = True
        for path in sorted(args.target_dir.glob("*.bf16.bin")):
            match = TARGET_RE.fullmatch(path.name)
            if match is None:
                continue
            role = match.group("role")
            x, y, _ = samples(path, role, args.target_samples_per_matrix, args.seed + int(match.group("layer")) * 101 + int(match.group("expert")))
            gain = gain_bpw(real_model, (x - feature_mean) / feature_std, y)
            target_evaluation.append(
                {
                    "file": path.name,
                    "layer": int(match.group("layer")),
                    "expert": int(match.group("expert")),
                    "role": role,
                    "samples": len(y),
                    "gain_bpw": gain,
                    "sha256": sha256_file(path),
                }
            )
        if len(target_evaluation) != 18:
            raise RuntimeError(f"expected 18 pinned matrices, found {len(target_evaluation)}")

    expert_payload_bytes = 3 * N * 2.5 / 8.0
    local_moment_bytes = 3 * STRATA * 2 * 2
    attributed_bytes = expert_payload_bytes + local_moment_bytes + model_info["bytes"] / 128.0
    cold_bytes = expert_payload_bytes + local_moment_bytes + model_info["bytes"]
    target_panel_weights = 6 * 3 * N
    charge_target_panel = (model_info["bytes"] + 6 * local_moment_bytes) * 8.0 / target_panel_weights
    result = {
        "decision": "OPEN_AND_EVALUATE_PINNED" if passes else "HARD_KILL_NONLINEAR_FLOW_BEFORE_PINNED",
        "claim_boundary": "bounded two-layer conditional affine-flow source-density test; failure excludes this 32-neighbour architecture, not arbitrary nonlinear manifolds",
        "protocol": {
            "strict_ptq": True,
            "auxiliary_layer": 15,
            "auxiliary_experts": experts,
            "train_experts": train_experts,
            "untouched_validation_experts": valid_experts,
            "training_matrices": len(train_x),
            "validation_matrices": len(valid_x),
            "samples_per_matrix": args.samples_per_matrix,
            "architecture": f"32 -> tanh({HIDDEN1}) -> tanh({HIDDEN2}) -> conditional mean, bounded log-variance",
            "longest_context_columns": 1024,
            "longest_context_rows": 256,
            "validation_promotion_gate_bpw": VALIDATION_GATE_BPW,
            "early_stop_rule": "after >=80 Adam updates, stop if best validation gain <0.05 bpw",
            "moment_matched_gaussian_control": "identical dimensions, optimizer, initialization seed, update budget and stopping rule",
            "required_final_s_bpw": REQUIRED_S_BPW,
            "identities": {
                "F": "D * 2^(2R)",
                "s": "heldout Gaussian NLL - heldout affine-flow NLL, bits/weight",
                "F_multiplier": "2^(-2s)",
            },
        },
        "validation": {
            "real_best_gain_bpw": real_gain,
            "real_F_multiplier": 2.0 ** (-2.0 * real_gain),
            "gaussian_control_best_gain_bpw": control_gain,
            "control_adjusted_gain_bpw": real_gain - control_gain,
            "fraction_of_required_final_s": real_gain / REQUIRED_S_BPW,
            "real_stopped_step": real_stop,
            "control_stopped_step": control_stop,
            "real_history": real_history,
            "control_history": control_history,
        },
        "pinned": {
            "opened": opened,
            "reason_not_opened": None if opened else "auxiliary validation failed the predeclared 0.05-bpw gate",
            "source_hashes_from_lock": target_hashes,
            "evaluation": target_evaluation,
        },
        "physical_ledger": {
            "decoder": model_info,
            "decoder_parameter_count_including_feature_moments": model_info["fp16_values"],
            "local_stratum_moment_bytes_per_expert": local_moment_bytes,
            "model_charge_bpw_over_six_expert_target_panel": model_info["bytes"] * 8.0 / target_panel_weights,
            "total_incremental_charge_bpw_over_target_panel": charge_target_panel,
            "s_after_target_panel_charge_bpw": real_gain - charge_target_panel,
            "cold_read_amplification_at_2p5_bpw_amortized_over_128_experts": cold_bytes / attributed_bytes,
            "hot_cached_read_amplification": (expert_payload_bytes + local_moment_bytes) / attributed_bytes,
            "basis_or_decoder_cache_assumption": "cold reads one shared decoder; hot keeps it resident; local moments are read with the expert",
        },
        "binding": {
            "script_sha256": sha256_file(Path(__file__)),
            "target_lock_sha256": sha256_file(args.target_lock),
            "auxiliary_source_sha256": source_hashes,
        },
        "normalization": normalization,
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "nonlinear_mlp_affine_flow_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": result["decision"], "validation": result["validation"], "physical_ledger": result["physical_ledger"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
