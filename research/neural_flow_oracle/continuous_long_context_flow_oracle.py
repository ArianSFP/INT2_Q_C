#!/usr/bin/env python3
"""Leakage-controlled continuous long-context flow oracle for Qwen weights.

The core candidate is a conditional affine normalizing flow whose mean and
log-scale depend on continuous contexts extending 1,024 columns and 256 rows,
plus an optimistic bidirectional/cross-role context.  Its base density is a
cross-fitted univariate Gaussian mixture.  All target matrices in a fold have
both their layer IDs and expert IDs removed from training.

The screen is intentionally favorable to the candidate:

* exact per-pair two-role KLT and role/stratum means/scales are free side info;
* the strongest variant sees future weights and the other KLT role exactly;
* decoder/model bytes are initially uncharged;
* a small ridge/GMM grid is reported by its best cross-fitted score.

It is therefore an early-kill oracle, not an operational codec.  Promotion
requires a free-side gain of -0.5*log2(0.8) bpw before serialization work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROWS = 768
COLS = 2048
N = ROWS * COLS
STRATA = 8
FOLDS = 12
REQUIRED_S = -0.5 * math.log2(0.8)
TARGET_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<role>gate|up|down)_proj\.weight"
)
DEV_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<role>up|down)_proj\.weight\.bf16\.bin$"
)


@dataclass(frozen=True)
class Pair:
    layer: int
    expert: int
    up: Path
    down: Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def stable_fold(layer: int, expert: int) -> int:
    digest = hashlib.sha256(f"flow-fold-v1:{layer}:{expert}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % FOLDS


def load_bf16(path: Path, role: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != N:
        raise ValueError(f"{path}: {raw.size} values, expected {N}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if role == "down":
        return np.asarray(values.reshape(COLS, ROWS).T, dtype=np.float64)
    return np.asarray(values.reshape(ROWS, COLS), dtype=np.float64)


def discover(dev_dir: Path, target_lock: Path) -> tuple[list[Pair], dict]:
    lock = json.loads(target_lock.read_text(encoding="utf-8"))
    target_layers = {int(row["layer"]) for row in lock["matrices"]}
    target_experts = {int(row["expert"]) for row in lock["matrices"]}
    found: dict[tuple[int, int], dict[str, Path]] = {}
    for path in sorted(dev_dir.glob("*.bf16.bin")):
        match = DEV_RE.fullmatch(path.name)
        if match is None:
            continue
        key = (int(match.group("layer")), int(match.group("expert")))
        found.setdefault(key, {})[match.group("role")] = path
    pairs = []
    excluded = []
    for (layer, expert), roles in sorted(found.items()):
        if set(roles) != {"up", "down"}:
            raise RuntimeError(f"incomplete pair L{layer} E{expert}: {sorted(roles)}")
        if layer in target_layers or expert in target_experts:
            excluded.append({"layer": layer, "expert": expert})
        else:
            pairs.append(Pair(layer, expert, roles["up"], roles["down"]))
    target_hashes = {
        f"L{int(row['layer'])}_E{int(row['expert'])}_{row['role']}": row["source_bf16_sha256"]
        for row in lock["matrices"]
    }
    return pairs, {
        "target_layers": sorted(target_layers),
        "target_experts": sorted(target_experts),
        "target_matrix_sha256": target_hashes,
        "target_lock_sha256": sha256_file(target_lock),
        "excluded_auxiliary_pairs": excluded,
    }


def standardized_klt(pair: Pair) -> tuple[np.ndarray, dict]:
    up = load_bf16(pair.up, "up")
    down = load_bf16(pair.down, "down")
    a = float(np.sum(up * up, dtype=np.float64))
    b = float(np.sum(down * down, dtype=np.float64))
    c = float(np.sum(up * down, dtype=np.float64))
    theta = 0.5 * math.atan2(2.0 * c, a - b)
    co, si = math.cos(theta), math.sin(theta)
    roles = np.stack((co * up + si * down, -si * up + co * down))

    row_energy = np.sum(roles * roles, axis=2, dtype=np.float64).reshape(-1)
    order = np.lexsort((np.arange(row_energy.size, dtype=np.int64), row_energy))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=np.int64)
    labels = np.minimum(STRATA - 1, rank * STRATA // rank.size).reshape(2, ROWS)
    cells = []
    for role in range(2):
        for stratum in range(STRATA):
            select = labels[role] == stratum
            if np.any(select):
                values = roles[role, select]
                mean = float(np.mean(values, dtype=np.float64))
                std = float(np.std(values, dtype=np.float64))
                roles[role, select] = (values - mean) / std
            else:
                # Exact two-role KLT can put every row of a weak component
                # outside an extreme pooled stratum.  The cell is unused, so
                # a finite identity backoff is canonical and score-neutral.
                mean, std = 0.0, 1.0
            cells.append(
                {
                    "role": role,
                    "stratum": stratum,
                    "rows": int(select.sum()),
                    "mean": mean,
                    "std": std,
                }
            )
    return roles.astype(np.float32), {"theta": theta, "cells": cells}


def add_feature(columns: list[np.ndarray], names: list[str], value: np.ndarray, name: str) -> None:
    columns.append(np.asarray(value, dtype=np.float32))
    names.append(name)


def pair_samples(pair: Pair, samples_per_role: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    roles, normalization = standardized_klt(pair)
    columns: list[np.ndarray] = []
    names: list[str] = []
    targets = []
    role_ids = []
    row_ids = []
    col_ids = []
    # Central rectangle makes every listed predecessor and future context
    # boundary-free, so there is no accidental wraparound dependence.
    for role in range(2):
        rng = np.random.default_rng(seed + 1009 * pair.layer + 9176 * pair.expert + role)
        flat = rng.choice(256 * 512, size=samples_per_role, replace=False)
        rr = 256 + flat // 512
        cc = 1024 + flat % 512
        targets.append(roles[role, rr, cc])
        role_ids.append(np.full(samples_per_role, role, dtype=np.int16))
        row_ids.append(rr)
        col_ids.append(cc)
    y = np.concatenate(targets).astype(np.float32)
    role_id = np.concatenate(role_ids)
    rr = np.concatenate(row_ids)
    cc = np.concatenate(col_ids)
    own = roles[role_id, rr, cc]  # target; used only to assert exact identity
    if not np.array_equal(own, y):
        raise AssertionError("target indexing mismatch")

    h_offsets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    v_offsets = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    causal_names = []
    for offset in h_offsets:
        name = f"self_h_minus_{offset}"
        add_feature(columns, names, roles[role_id, rr, cc - offset], name)
        causal_names.append(name)
    for offset in v_offsets:
        name = f"self_v_minus_{offset}"
        add_feature(columns, names, roles[role_id, rr - offset, cc], name)
        causal_names.append(name)
    for dr, dc in ((1, 1), (1, -1), (1, 8), (1, -8), (8, 1), (8, -1)):
        name = f"self_diag_minus_r{dr}_dc{dc}"
        add_feature(columns, names, roles[role_id, rr - dr, cc + dc], name)
        causal_names.append(name)

    future_names = []
    for offset in (1, 8, 64, 512):
        name = f"self_h_plus_{offset}"
        add_feature(columns, names, roles[role_id, rr, cc + offset], name)
        future_names.append(name)
    for offset in (1, 8, 64, 256):
        name = f"self_v_plus_{offset}"
        add_feature(columns, names, roles[role_id, rr + offset, cc], name)
        future_names.append(name)

    other = 1 - role_id
    partner_names = []
    name = "partner_same"
    add_feature(columns, names, roles[other, rr, cc], name)
    partner_names.append(name)
    for offset in (1, 8, 64, 512):
        for sign, label in ((-1, "minus"), (1, "plus")):
            name = f"partner_h_{label}_{offset}"
            add_feature(columns, names, roles[other, rr, cc + sign * offset], name)
            partner_names.append(name)
    for offset in (1, 8, 64):
        for sign, label in ((-1, "minus"), (1, "plus")):
            name = f"partner_v_{label}_{offset}"
            add_feature(columns, names, roles[other, rr + sign * offset, cc], name)
            partner_names.append(name)

    # Nonlinear multiscale summaries of exact continuous context values.
    base = {name: columns[index] for index, name in enumerate(names)}
    groups = {
        "h_past": [base[f"self_h_minus_{o}"] for o in h_offsets],
        "v_past": [base[f"self_v_minus_{o}"] for o in v_offsets],
        "future": [base[n] for n in future_names],
        "partner": [base[n] for n in partner_names],
    }
    summary_names = []
    for group_name, values in groups.items():
        stack = np.stack(values)
        for stat_name, stat in (
            ("mean", np.mean(stack, axis=0)),
            ("mean_square_minus_1", np.mean(stack * stack, axis=0) - 1.0),
            ("mean_abs_centered", np.mean(np.abs(stack), axis=0) - math.sqrt(2.0 / math.pi)),
        ):
            name = f"summary_{group_name}_{stat_name}"
            add_feature(columns, names, stat, name)
            summary_names.append(name)

    # Fixed nonlinear lifts; no fitted neural weights or heldout choices.
    primary = [
        "self_h_minus_1", "self_h_minus_2", "self_h_minus_8", "self_h_minus_64",
        "self_v_minus_1", "self_v_minus_8", "self_v_minus_64", "partner_same",
    ]
    nonlinear_names = []
    for source_name in primary:
        value = base[source_name]
        for suffix, transformed in (("square_minus_1", value * value - 1.0), ("abs_centered", np.abs(value) - math.sqrt(2.0 / math.pi))):
            name = f"nonlinear_{source_name}_{suffix}"
            add_feature(columns, names, transformed, name)
            nonlinear_names.append(name)
    for left, right in (
        ("self_h_minus_1", "self_h_minus_2"),
        ("self_h_minus_1", "self_v_minus_1"),
        ("self_v_minus_1", "self_v_minus_2"),
        ("partner_same", "self_h_minus_1"),
        ("partner_same", "self_v_minus_1"),
        ("summary_h_past_mean", "summary_v_past_mean"),
    ):
        left_value = columns[names.index(left)]
        right_value = columns[names.index(right)]
        name = f"interaction_{left}_x_{right}"
        add_feature(columns, names, left_value * right_value, name)
        nonlinear_names.append(name)

    # Decoder-known position/role coordinates.
    coordinate_names = []
    for period in (16, 64, 256):
        for axis, value in (("row", rr), ("col", cc)):
            for trig, fn in (("sin", np.sin), ("cos", np.cos)):
                name = f"position_{axis}_{trig}_{period}"
                add_feature(columns, names, fn(2.0 * math.pi * value / period), name)
                coordinate_names.append(name)
    add_feature(columns, names, 2.0 * role_id.astype(np.float32) - 1.0, "role_sign")
    coordinate_names.append("role_sign")
    x = np.stack(columns, axis=1).astype(np.float32)

    causal_set = set(causal_names + [n for n in summary_names if n.startswith("summary_h_past") or n.startswith("summary_v_past")] + [n for n in nonlinear_names if "partner" not in n and "future" not in n] + coordinate_names)
    nonpartner_set = set(causal_set | set(future_names) | {n for n in summary_names if n.startswith("summary_future")})
    masks = {
        "causal_self": np.asarray([name in causal_set for name in names]),
        "bidirectional_self_free": np.asarray([name in nonpartner_set for name in names]),
        "bidirectional_cross_role_free": np.ones(len(names), dtype=bool),
    }
    return x, y, role_id, names, {"normalization": normalization, "masks": {k: np.flatnonzero(v).tolist() for k, v in masks.items()}}


def normal_logpdf(y: np.ndarray, mean: np.ndarray | float, variance: np.ndarray | float) -> np.ndarray:
    return -0.5 * (np.log(2.0 * math.pi * variance) + (y - mean) ** 2 / variance)


def fit_gmm(z: np.ndarray, components: int, iterations: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if components == 1:
        return np.ones(1), np.asarray([float(np.mean(z))]), np.asarray([float(np.var(z))])
    quantiles = np.linspace(0.1, 0.9, components)
    means = np.quantile(z, quantiles)
    variances = np.full(components, max(float(np.var(z)) * 0.5, 0.05))
    weights = np.full(components, 1.0 / components)
    for _ in range(iterations):
        logp = np.stack(
            [np.log(weights[k] + 1e-300) + normal_logpdf(z, means[k], variances[k]) for k in range(components)],
            axis=1,
        )
        maximum = np.max(logp, axis=1, keepdims=True)
        responsibilities = np.exp(logp - maximum)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        counts = np.sum(responsibilities, axis=0) + 1e-9
        weights = counts / counts.sum()
        means = (responsibilities.T @ z) / counts
        for k in range(components):
            variances[k] = max(float(np.dot(responsibilities[:, k], (z - means[k]) ** 2) / counts[k]), 0.01)
    return weights, means, variances


def mixture_logpdf(z: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    logp = np.stack(
        [np.log(weights[k] + 1e-300) + normal_logpdf(z, means[k], variances[k]) for k in range(len(weights))],
        axis=1,
    )
    maximum = np.max(logp, axis=1)
    return maximum + np.log(np.sum(np.exp(logp - maximum[:, None]), axis=1))


def fit_flow(x_train: np.ndarray, y_train: np.ndarray, ridge: float, components: int) -> dict:
    feature_mean = np.mean(x_train, axis=0, dtype=np.float64)
    feature_std = np.std(x_train, axis=0, dtype=np.float64)
    feature_std = np.maximum(feature_std, 1e-3)
    z = (x_train.astype(np.float64) - feature_mean) / feature_std
    design = np.column_stack((np.ones(len(z)), z))
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * (ridge * len(z))
    penalty[0, 0] = 0.0
    beta_mean = np.linalg.solve(gram + penalty, design.T @ y_train)
    residual = y_train.astype(np.float64) - design @ beta_mean

    # Conditional log-scale is another affine flow driven by the same long
    # context.  A floor makes the log-square target stable; the multiplicative
    # calibration enforces unit average normalized residual energy.
    floor = 0.05 * float(np.mean(residual * residual))
    target_log = np.log(residual * residual + floor)
    beta_logvar = np.linalg.solve(gram + 10.0 * penalty + np.eye(gram.shape[0]) * 1e-9, design.T @ target_log)
    predicted_logvar = np.clip(design @ beta_logvar, -1.0, 1.0)
    calibration = float(np.mean(residual * residual / np.exp(predicted_logvar)))
    predicted_logvar = np.clip(predicted_logvar + math.log(calibration), -1.0, 1.0)
    scale = np.exp(0.5 * predicted_logvar)
    base = residual / scale
    weights, means, variances = fit_gmm(base, components)
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "beta_mean": beta_mean,
        "beta_logvar": beta_logvar,
        "logvar_calibration": math.log(calibration),
        "weights": weights,
        "means": means,
        "variances": variances,
    }


def eval_flow(model: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (x.astype(np.float64) - model["feature_mean"]) / model["feature_std"]
    design = np.column_stack((np.ones(len(z)), z))
    mean = design @ model["beta_mean"]
    logvar = np.clip(design @ model["beta_logvar"] + model["logvar_calibration"], -1.0, 1.0)
    scale = np.exp(0.5 * logvar)
    base = (y.astype(np.float64) - mean) / scale
    return mixture_logpdf(base, model["weights"], model["means"], model["variances"]) - np.log(scale)


def run_crossfit(x: np.ndarray, y: np.ndarray, metadata: list[dict], feature_names: list[str], masks: dict[str, np.ndarray]) -> dict:
    layers = np.asarray([row["layer"] for row in metadata], dtype=np.int16)
    experts = np.asarray([row["expert"] for row in metadata], dtype=np.int16)
    folds = np.asarray([row["fold"] for row in metadata], dtype=np.int16)
    pair_ids = np.asarray([row["pair_index"] for row in metadata], dtype=np.int16)
    role_ids = np.asarray([row["role"] for row in metadata], dtype=np.int8)
    variants = {}
    grid = [(1e-4, 4), (1e-3, 4), (1e-3, 8)]

    # Context-free continuous mixture establishes the marginal contribution.
    marginal_rows = []
    marginal_sample_gain = np.empty(len(y), dtype=np.float64)
    marginal_sample_gain.fill(np.nan)
    for fold in range(FOLDS):
        test = folds == fold
        if not np.any(test):
            continue
        test_layers = set(layers[test].tolist())
        test_experts = set(experts[test].tolist())
        train = (~test) & (~np.isin(layers, list(test_layers))) & (~np.isin(experts, list(test_experts)))
        weights, means, variances = fit_gmm(y[train].astype(np.float64), 8)
        logp = mixture_logpdf(y[test].astype(np.float64), weights, means, variances)
        base = normal_logpdf(y[test].astype(np.float64), 0.0, 1.0)
        gain = (logp - base) / math.log(2.0)
        marginal_sample_gain[test] = gain
        marginal_rows.append(
            {
                "fold": fold,
                "train_samples": int(train.sum()),
                "test_samples": int(test.sum()),
                "gain_bits": float(np.sum(gain)),
            }
        )
    marginal_gain = sum(r["gain_bits"] for r in marginal_rows) / sum(r["test_samples"] for r in marginal_rows)
    marginal_groups = []
    for pair_id in np.unique(pair_ids):
        for role in (0, 1):
            select = np.isfinite(marginal_sample_gain) & (pair_ids == pair_id) & (role_ids == role)
            if np.any(select):
                marginal_groups.append(float(np.mean(marginal_sample_gain[select])))
    marginal_se = float(np.std(marginal_groups, ddof=1) / math.sqrt(len(marginal_groups)))
    variants["marginal_gmm8"] = {
        "gain_bpw": marginal_gain,
        "matrix_role_groups": len(marginal_groups),
        "group_standard_error_bpw": marginal_se,
        "optimistic_two_se_upper_bpw": marginal_gain + 2.0 * marginal_se,
        "F_multiplier": 2.0 ** (-2.0 * marginal_gain),
        "folds": marginal_rows,
    }

    for variant, mask in masks.items():
        xv = x[:, mask]
        candidates = []
        for ridge, components in grid:
            fold_rows = []
            sample_gain = np.empty(len(y), dtype=np.float64)
            sample_gain.fill(np.nan)
            for fold in range(FOLDS):
                test = folds == fold
                if not np.any(test):
                    continue
                test_layers = set(layers[test].tolist())
                test_experts = set(experts[test].tolist())
                train = (~test) & (~np.isin(layers, list(test_layers))) & (~np.isin(experts, list(test_experts)))
                model = fit_flow(xv[train], y[train], ridge, components)
                logp = eval_flow(model, xv[test], y[test])
                baseline = normal_logpdf(y[test].astype(np.float64), 0.0, 1.0)
                gain = (logp - baseline) / math.log(2.0)
                sample_gain[test] = gain
                fold_rows.append(
                    {
                        "fold": fold,
                        "test_pairs": sorted({int(v) for v in pair_ids[test]}),
                        "excluded_layers": sorted(test_layers),
                        "excluded_experts": sorted(test_experts),
                        "train_samples": int(train.sum()),
                        "test_samples": int(test.sum()),
                        "gain_bpw": float(np.mean(gain)),
                    }
                )
            valid = np.isfinite(sample_gain)
            group_gains = []
            for pair_id in np.unique(pair_ids):
                for role in (0, 1):
                    select = valid & (pair_ids == pair_id) & (role_ids == role)
                    if np.any(select):
                        group_gains.append(float(np.mean(sample_gain[select])))
            group_gains_array = np.asarray(group_gains)
            mean = float(np.mean(sample_gain[valid]))
            se = float(np.std(group_gains_array, ddof=1) / math.sqrt(len(group_gains_array)))
            candidates.append(
                {
                    "ridge_per_sample": ridge,
                    "mixture_components": components,
                    "features": int(mask.sum()),
                    "gain_bpw": mean,
                    "matrix_role_groups": len(group_gains),
                    "group_standard_error_bpw": se,
                    "optimistic_two_se_upper_bpw": mean + 2.0 * se,
                    "F_multiplier": 2.0 ** (-2.0 * mean),
                    "folds": fold_rows,
                }
            )
        best = max(candidates, key=lambda row: row["gain_bpw"])
        variants[variant] = {"best": best, "grid": candidates}
    return variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", type=Path, required=True)
    ap.add_argument("--target-lock", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples-per-role", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=260901)
    args = ap.parse_args()
    started = time.time()
    pairs, target_binding = discover(args.dev_dir, args.target_lock)
    if len(pairs) != 57:
        raise RuntimeError(f"expected 57 leakage-clean auxiliary pairs, got {len(pairs)}")

    xs, ys = [], []
    metadata: list[dict] = []
    feature_names = None
    masks = None
    normalizations = []
    auxiliary_hashes = {}
    for pair_index, pair in enumerate(pairs):
        x, y, roles, names, details = pair_samples(pair, args.samples_per_role, args.seed)
        if feature_names is None:
            feature_names = names
            masks = {name: np.asarray([i in indices for i in range(len(names))]) for name, indices in details["masks"].items()}
        elif names != feature_names:
            raise AssertionError("feature schema changed")
        xs.append(x)
        ys.append(y)
        fold = stable_fold(pair.layer, pair.expert)
        for role in (0, 1):
            metadata.extend(
                {
                    "pair_index": pair_index,
                    "layer": pair.layer,
                    "expert": pair.expert,
                    "fold": fold,
                    "role": role,
                }
                for _ in range(args.samples_per_role)
            )
        normalizations.append(
            {
                "pair_index": pair_index,
                "layer": pair.layer,
                "expert": pair.expert,
                **details["normalization"],
            }
        )
        auxiliary_hashes[pair.up.name] = sha256_file(pair.up)
        auxiliary_hashes[pair.down.name] = sha256_file(pair.down)
        print(f"features {pair_index + 1}/{len(pairs)} L{pair.layer} E{pair.expert}", flush=True)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if len(metadata) != len(y):
        raise AssertionError((len(metadata), len(y)))
    variants = run_crossfit(x, y, metadata, feature_names, masks)
    scored = [
        (name, row["gain_bpw"] if name == "marginal_gmm8" else row["best"]["gain_bpw"])
        for name, row in variants.items()
    ]
    best_name, best_gain = max(scored, key=lambda item: item[1])
    best_two_se = best_gain
    if best_name == "marginal_gmm8":
        best_two_se = variants[best_name]["optimistic_two_se_upper_bpw"]
    else:
        best_two_se = variants[best_name]["best"]["optimistic_two_se_upper_bpw"]

    result = {
        "decision": "PROMOTE_TO_SERIALIZATION" if best_gain >= REQUIRED_S else "HARD_KILL_CONTINUOUS_LONG_CONTEXT_FLOW",
        "claim_boundary": "cross-fitted source-density early-kill for the specified affine-flow/Gaussian-mixture family; not a universal RD converse",
        "protocol": {
            "strict_ptq": True,
            "pinned_sources_opened": False,
            "reason_pinned_sources_not_opened": "predeclared auxiliary free-side promotion gate failed" if best_gain < REQUIRED_S else None,
            "eligible_auxiliary_pairs": len(pairs),
            "samples_per_role_per_pair": args.samples_per_role,
            "sampled_weights": int(len(y)),
            "folds": FOLDS,
            "heldout_rule": "for every fold, exclude all auxiliary pairs sharing any test layer OR any test expert",
            "representation": "exact two-role KLT; exact eight-stratum role-cell mean/std normalization supplied free",
            "strongest_oracle_context": "bidirectional continuous context to +/-512 columns and +/-256 rows, causal predecessor to 1024 columns, and exact paired KLT-role values",
            "model": "ridge conditional-mean + ridge conditional-logscale affine normalizing flow with cross-fitted Gaussian-mixture base",
            "all_model_and_side_bytes_free": True,
            "required_s_bpw": REQUIRED_S,
            "identities": {
                "F": "D * 2^(2R)",
                "effective_rate_advantage": "s = heldout Gaussian NLL - heldout flow NLL (bits/weight)",
                "F_multiplier": "2^(-2s)",
                "twenty_percent_target": "s >= -0.5*log2(0.8) = 0.160964047443681",
            },
            "seed": args.seed,
        },
        "binding": {
            **target_binding,
            "auxiliary_source_sha256": auxiliary_hashes,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "feature_names": feature_names,
        "normalizations": normalizations,
        "variants": variants,
        "gate": {
            "best_variant": best_name,
            "best_free_side_gain_bpw": best_gain,
            "best_optimistic_two_se_upper_bpw": best_two_se,
            "fraction_of_required_s": best_gain / REQUIRED_S,
            "shortfall_bpw": REQUIRED_S - best_gain,
            "best_F_multiplier": 2.0 ** (-2.0 * best_gain),
            "required_F_multiplier": 0.8,
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gate": result["gate"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
