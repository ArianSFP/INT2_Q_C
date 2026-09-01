#!/usr/bin/env python3
"""CPU-only, leakage-safe shape/partial-Hadamard information oracle.

This is an early-kill diagnostic, not a reconstruction-MSE claim.  It asks
whether scale-invariant 2048-weight group shape can predict a useful choice
between identity and nested signed partial Walsh--Hadamard transforms.

Every outer fold excludes the held-out layer and expert.  The comparator is a
strong moment-matched Gaussian for each role x energy-stratum x shape class;
therefore the reported likelihood gain is non-Gaussian *shape* gain rather
than an RMS-only hyperprior gain.  Class streams and a conservative physical
decoder-model image are charged before the 0.160964 bpw promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr


ROWS = 768
COLS = 2048
ROLES = 3
STRATA = 8
CONTEXTS = ROLES * STRATA
FEATURES = 8
DEPTHS = tuple(range(12))  # 0=identity; d>0 is signed FWHT width 2**d.
CLASS_COUNTS = (1, 2, 4, 8)
BIN_LOW = -8.0
BIN_HIGH = 8.0
BIN_STEP = 0.125
FINITE_EDGES = np.arange(BIN_LOW, BIN_HIGH + BIN_STEP / 2, BIN_STEP)
EDGES = np.concatenate(([-np.inf], FINITE_EDGES, [np.inf])).astype(np.float64)
NBINS = EDGES.size - 1
FREQ_TOTAL = 65535
TARGET_GAIN_BPW = -0.5 * math.log2(0.8)
TARGET_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.memmap(path, dtype="<u2", mode="r", shape=shape)
    return ((np.asarray(words, dtype=np.uint32) << np.uint32(16)).view(np.float32))


def exact_klt(up: np.ndarray, down_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    up64 = np.asarray(up, dtype=np.float64)
    down64 = np.asarray(down_t, dtype=np.float64)
    a = float(np.sum(up64 * up64, dtype=np.float64))
    b = float(np.sum(down64 * down64, dtype=np.float64))
    c = float(np.sum(up64 * down64, dtype=np.float64))
    theta = 0.5 * math.atan2(2.0 * c, a - b)
    co, si = math.cos(theta), math.sin(theta)
    # FP32 storage keeps the diagnostic practical while the KLT fit and every
    # reduction used for scoring remain FP64.
    k0 = (co * up64 + si * down64).astype(np.float32)
    k1 = (-si * up64 + co * down64).astype(np.float32)
    return k0, k1, theta


def energy_contexts(roles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.sum(
        np.asarray(roles, dtype=np.float64) ** 2, axis=2, dtype=np.float64
    )
    flat = energy.reshape(-1)
    order = np.lexsort((np.arange(flat.size, dtype=np.int64), flat))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=np.int64)
    labels = np.minimum(STRATA - 1, rank * STRATA // flat.size).reshape(ROLES, ROWS)
    scales = np.empty((ROLES, STRATA), dtype=np.float16)
    for role in range(ROLES):
        for stratum in range(STRATA):
            selected = labels[role] == stratum
            if np.any(selected):
                value = math.sqrt(float(np.sum(energy[role, selected])) / (int(selected.sum()) * COLS))
            else:
                pooled = labels == stratum
                value = math.sqrt(float(np.sum(energy[pooled])) / (int(pooled.sum()) * COLS))
            scales[role, stratum] = np.float16(value)
    contexts = (
        np.arange(ROLES, dtype=np.int16)[:, None] * STRATA + labels.astype(np.int16)
    )
    return contexts, scales, energy


def shape_features(roles: np.ndarray) -> np.ndarray:
    x = np.asarray(roles, dtype=np.float64)
    mean = np.mean(x, axis=2, dtype=np.float64)
    centered = x - mean[:, :, None]
    variance = np.mean(centered * centered, axis=2, dtype=np.float64)
    std = np.sqrt(np.maximum(variance, np.finfo(np.float64).tiny))
    z = centered / std[:, :, None]
    skew = np.mean(z * z * z, axis=2, dtype=np.float64)
    kurt = np.mean((z * z) ** 2, axis=2, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x, axis=2, dtype=np.float64))
    safe_rms = np.maximum(rms, np.finfo(np.float64).tiny)
    absolute = np.abs(x) / safe_rms[:, :, None]
    result = np.stack(
        (
            np.clip(mean / safe_rms, -1.0, 1.0),
            np.clip(skew, -12.0, 12.0),
            np.log(np.clip(kurt, 1.0, 512.0)),
            np.mean(absolute > 2.0, axis=2),
            np.mean(absolute > 3.0, axis=2),
            np.mean(absolute > 4.0, axis=2),
            np.mean(absolute, axis=2),
            np.log(np.maximum(np.max(absolute, axis=2), 1.0)),
        ),
        axis=2,
    )
    return result.astype(np.float32)


def quantized_frequencies(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size > FREQ_TOTAL:
        raise ValueError("bad frequency geometry")
    smooth = counts + 0.5
    probability = smooth / float(np.sum(smooth))
    available = FREQ_TOTAL - counts.size
    scaled = probability * available
    extra = np.floor(scaled).astype(np.int64)
    frequencies = extra + 1
    remaining = FREQ_TOTAL - int(np.sum(frequencies))
    if remaining:
        order = np.argsort(-(scaled - extra), kind="stable")
        frequencies[order[:remaining]] += 1
    if int(np.sum(frequencies)) != FREQ_TOTAL or np.any(frequencies <= 0):
        raise AssertionError("frequency normalization failed")
    return frequencies.astype(np.uint16)


def bin_values(values: np.ndarray) -> np.ndarray:
    # Search only the finite internal boundaries.  Result 0 and NBINS-1 are
    # the two tail symbols.
    return np.searchsorted(FINITE_EDGES, values, side="right").astype(np.uint8)


def gaussian_bin_probability(mean: float, variance: float) -> np.ndarray:
    std = math.sqrt(max(variance, 1e-12))
    probabilities = np.diff(ndtr((EDGES - mean) / std))
    probabilities = np.maximum(probabilities, np.finfo(np.float64).tiny)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def fit_kmeans(features: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    z = np.clip((values - mean) / std, -8.0, 8.0)
    if classes == 1:
        centers = np.mean(z, axis=0, keepdims=True)
        return mean, std, centers
    centers = [np.mean(z, axis=0)]
    nearest = np.sum((z - centers[0]) ** 2, axis=1)
    for _ in range(1, classes):
        index = int(np.argmax(nearest))
        centers.append(z[index].copy())
        distance = np.sum((z - centers[-1]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
    center_array = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(z.shape[0], dtype=np.int16)
    for _ in range(32):
        distance = np.sum((z[:, None, :] - center_array[None, :, :]) ** 2, axis=2)
        updated_labels = np.argmin(distance, axis=1).astype(np.int16)
        updated = center_array.copy()
        for label in range(classes):
            selected = updated_labels == label
            if np.any(selected):
                updated[label] = np.mean(z[selected], axis=0, dtype=np.float64)
        if np.array_equal(labels, updated_labels) and np.allclose(updated, center_array):
            center_array = updated
            break
        labels, center_array = updated_labels, updated
    return mean, std, center_array


def assign_kmeans(features: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std, centers = model
    z = np.clip((np.asarray(features, dtype=np.float64) - mean) / std, -8.0, 8.0)
    distance = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distance, axis=1).astype(np.uint8)


@dataclass
class ExpertData:
    layer: int
    expert: int
    theta: float
    contexts: np.ndarray  # 3 x 768
    scales: np.ndarray  # 3 x 8, FP16
    energy: np.ndarray  # 3 x 768
    features: np.ndarray  # 3 x 768 x FEATURES
    symbols: np.ndarray  # depths x 3 x 768 x 2048, uint8
    group_sums: np.ndarray  # depths x 3 x 768, normalized by context scale
    norm_energy: np.ndarray  # 3 x 768
    source_rows: list[dict]


def source_triplets(lock_path: Path, source_root: Path) -> list[tuple[int, int, dict[str, tuple[Path, dict]]]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, int], dict[str, tuple[Path, dict]]] = {}
    for row in lock["matrices"]:
        match = TARGET_RE.fullmatch(str(row["tensor"]))
        if not match:
            raise ValueError(f"unexpected tensor {row['tensor']}")
        layer, expert, role = int(match.group(1)), int(match.group(2)), match.group(3)
        relative = row.get("output_relpath") or row.get("source_relpath")
        grouped.setdefault((layer, expert), {})[role] = (source_root / relative, row)
    result = []
    for (layer, expert), roles in sorted(grouped.items()):
        if set(roles) != {"gate", "up", "down"}:
            raise ValueError(f"incomplete triplet L{layer} E{expert}")
        result.append((layer, expert, roles))
    if len(result) != 6:
        raise ValueError("expected six target experts")
    return result


def load_expert(layer: int, expert: int, paths: dict[str, tuple[Path, dict]]) -> ExpertData:
    arrays: dict[str, np.ndarray] = {}
    source_rows = []
    for role in ("gate", "up", "down"):
        path, row = paths[role]
        digest = sha256_file(path)
        if digest != row["source_bf16_sha256"]:
            raise ValueError(f"source hash mismatch {path}")
        shape = tuple(int(value) for value in row["shape"])
        values = bf16(path, shape)
        arrays[role] = values if role != "down" else values.T.copy()
        source_rows.append(
            {"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}
        )
    k0, k1, theta = exact_klt(arrays["up"], arrays["down"])
    roles = np.stack((arrays["gate"], k0, k1)).astype(np.float32, copy=False)
    contexts, scales, energy = energy_contexts(roles)
    features = shape_features(roles)
    scale_for_group = np.empty((ROLES, ROWS), dtype=np.float32)
    for role in range(ROLES):
        scale_for_group[role] = scales[role, contexts[role] % STRATA].astype(np.float32)
    norm_energy = energy / np.square(scale_for_group.astype(np.float64))

    symbol_rows = []
    sum_rows = []
    # True identity is included without a random sign diagonal.
    normalized = roles / scale_for_group[:, :, None]
    symbol_rows.append(bin_values(normalized))
    sum_rows.append(np.sum(normalized, axis=2, dtype=np.float64))

    # One deterministic sign diagonal is nested through every partial depth.
    # It is procedural decoder state, not a serialized model or side stream.
    seed_material = hashlib.sha256(
        f"shape-partial-fwht-v1:{layer}:{expert}".encode()
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_material, "little"))
    signs = rng.integers(0, 2, size=roles.shape, dtype=np.int8).astype(np.float32)
    signs = signs * 2.0 - 1.0
    transformed = normalized * signs
    width = 1
    inverse_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    for depth in range(1, 12):
        view = transformed.reshape(ROLES, ROWS, -1, 2 * width)
        left = view[:, :, :, :width].copy()
        right = view[:, :, :, width:].copy()
        view[:, :, :, :width] = (left + right) * inverse_sqrt2
        view[:, :, :, width:] = (left - right) * inverse_sqrt2
        symbol_rows.append(bin_values(transformed))
        sum_rows.append(np.sum(transformed, axis=2, dtype=np.float64))
        width *= 2
    symbols = np.stack(symbol_rows, axis=0)
    group_sums = np.stack(sum_rows, axis=0)
    del arrays, roles, normalized, transformed, signs
    return ExpertData(
        layer,
        expert,
        theta,
        contexts,
        scales,
        energy,
        features,
        symbols,
        group_sums,
        norm_energy,
        source_rows,
    )


def model_bytes(classes: int) -> dict[str, int]:
    # Conservative complete image: 16-bit frequency per source symbol, class
    # label frequencies, depth map, FP16 Gaussian moments, plus encoder-side
    # feature normalization/centroids even though inference does not need them.
    header = 256
    density = CONTEXTS * classes * NBINS * 2
    class_frequency = CONTEXTS * classes * 2
    depth_map = CONTEXTS * classes
    gaussian_moments = CONTEXTS * classes * 2 * 2
    feature_model = CONTEXTS * (2 * FEATURES + classes * FEATURES) * 2
    integrity = 32
    total = header + density + class_frequency + depth_map + gaussian_moments + feature_model + integrity
    return {
        "header": header,
        "density_frequencies": density,
        "class_frequencies": class_frequency,
        "depth_map": depth_map,
        "gaussian_moments": gaussian_moments,
        "feature_model": feature_model,
        "integrity": integrity,
        "total": total,
    }


def group_indices(row: ExpertData, context: int) -> tuple[int, np.ndarray]:
    role = context // STRATA
    return role, np.flatnonzero(row.contexts[role] == context)


def counts_for(row: ExpertData, depth: int, role: int, groups: np.ndarray) -> np.ndarray:
    if groups.size == 0:
        return np.zeros(NBINS, dtype=np.int64)
    return np.bincount(
        row.symbols[depth, role, groups].reshape(-1), minlength=NBINS
    ).astype(np.int64)


def evaluate_classes(rows: list[ExpertData], classes: int) -> dict:
    total_weights = len(rows) * ROLES * ROWS * COLS
    fold_reports = []
    gross_gain = 0.0
    class_bits_total = 0.0
    local_bytes_total = 0
    transform_weight = np.zeros(len(DEPTHS), dtype=np.int64)
    for heldout_index, heldout in enumerate(rows):
        train = [row for i, row in enumerate(rows) if i != heldout_index]
        if any(row.layer == heldout.layer or row.expert == heldout.expert for row in train):
            raise AssertionError("outer fold failed layer/expert exclusion")
        fold_gain = 0.0
        fold_label_bits = 0.0
        fold_transform_weight = np.zeros(len(DEPTHS), dtype=np.int64)
        context_reports = []
        for context in range(CONTEXTS):
            train_feature_parts = []
            train_group_refs: list[tuple[ExpertData, int, np.ndarray]] = []
            for row in train:
                role, groups = group_indices(row, context)
                if groups.size:
                    train_feature_parts.append(row.features[role, groups])
                    train_group_refs.append((row, role, groups))
            role, test_groups = group_indices(heldout, context)
            if not train_feature_parts or test_groups.size == 0:
                continue
            kmodel = fit_kmeans(np.concatenate(train_feature_parts, axis=0), classes)
            train_assignments = [
                assign_kmeans(row.features[train_role, groups], kmodel)
                for row, train_role, groups in train_group_refs
            ]
            test_assignment = assign_kmeans(heldout.features[role, test_groups], kmodel)
            train_class_counts = sum(
                (np.bincount(labels, minlength=classes) for labels in train_assignments),
                np.zeros(classes, dtype=np.int64),
            )
            class_freq = quantized_frequencies(train_class_counts)
            fold_label_bits += float(
                np.sum(
                    np.bincount(test_assignment, minlength=classes)
                    * -np.log2(class_freq.astype(np.float64) / FREQ_TOTAL),
                    dtype=np.float64,
                )
            )
            class_reports = []
            for label in range(classes):
                test_selected = test_groups[test_assignment == label]
                if test_selected.size == 0:
                    continue
                depth_training_gain = []
                depth_models = []
                for depth in DEPTHS:
                    counts = np.zeros(NBINS, dtype=np.int64)
                    total_sum = 0.0
                    total_sum2 = 0.0
                    values_count = 0
                    for (row, train_role, groups), assignments in zip(
                        train_group_refs, train_assignments, strict=True
                    ):
                        selected = groups[assignments == label]
                        counts += counts_for(row, depth, train_role, selected)
                        total_sum += float(np.sum(row.group_sums[depth, train_role, selected], dtype=np.float64))
                        total_sum2 += float(np.sum(row.norm_energy[train_role, selected], dtype=np.float64))
                        values_count += int(selected.size) * COLS
                    if values_count == 0:
                        depth_training_gain.append(-math.inf)
                        depth_models.append(None)
                        continue
                    mean = total_sum / values_count
                    variance = max(total_sum2 / values_count - mean * mean, 1e-12)
                    frequencies = quantized_frequencies(counts)
                    model_p = frequencies.astype(np.float64) / FREQ_TOTAL
                    gaussian_p = gaussian_bin_probability(mean, variance)
                    train_gain = float(np.sum(counts * np.log2(model_p / gaussian_p), dtype=np.float64))
                    depth_training_gain.append(train_gain / values_count)
                    depth_models.append((model_p, gaussian_p, mean, variance))
                chosen = int(np.argmax(depth_training_gain))
                model_p, gaussian_p, mean, variance = depth_models[chosen]
                test_counts = counts_for(heldout, chosen, role, test_selected)
                gain = float(np.sum(test_counts * np.log2(model_p / gaussian_p), dtype=np.float64))
                fold_gain += gain
                coefficients = int(test_selected.size) * COLS
                transform_weight[chosen] += coefficients
                fold_transform_weight[chosen] += coefficients
                class_reports.append(
                    {
                        "class": label,
                        "test_groups": int(test_selected.size),
                        "chosen_depth": chosen,
                        "chosen_width": 1 if chosen == 0 else 1 << chosen,
                        "training_shape_gain_bpw": depth_training_gain[chosen],
                        "heldout_shape_gain_bpw": gain / coefficients,
                        "training_gaussian_mean": mean,
                        "training_gaussian_variance": variance,
                    }
                )
            context_reports.append(
                {
                    "context": context,
                    "role": ("gate", "klt0", "klt1")[context // STRATA],
                    "stratum": context % STRATA,
                    "test_groups": int(test_groups.size),
                    "classes_present": len(class_reports),
                    "class_rows": class_reports,
                }
            )
        # A concrete local framing charge is rounded once per expert.  The 96
        # bytes cover magic/version, lengths, model binding, and integrity.
        label_bytes = math.ceil(fold_label_bits / 8.0) if classes > 1 else 0
        local_bytes = 96 + label_bytes
        fold_weights = ROLES * ROWS * COLS
        fold_net_before_model = (fold_gain - local_bytes * 8) / fold_weights
        gross_gain += fold_gain
        class_bits_total += fold_label_bits
        local_bytes_total += local_bytes
        fold_reports.append(
            {
                "heldout_layer": heldout.layer,
                "heldout_expert": heldout.expert,
                "training_layers": sorted(row.layer for row in train),
                "training_experts": sorted(row.expert for row in train),
                "gross_shape_gain_bpw": fold_gain / fold_weights,
                "ideal_class_label_bits": fold_label_bits,
                "physical_local_bytes": local_bytes,
                "net_before_shared_model_bpw": fold_net_before_model,
                "transform_weight_fraction": {
                    str(depth): float(fold_transform_weight[depth] / fold_weights)
                    for depth in DEPTHS
                    if fold_transform_weight[depth]
                },
                "contexts": context_reports,
            }
        )
    ledger = model_bytes(classes)
    all_side_bits = local_bytes_total * 8 + ledger["total"] * 8
    net = (gross_gain - all_side_bits) / total_weights
    conservative_energy_label_bits = len(rows) * ROLES * ROWS * 3
    conservative_energy_scale_bytes = len(rows) * CONTEXTS * 2
    net_if_all_context_side_recharged = (
        gross_gain
        - all_side_bits
        - conservative_energy_label_bits
        - conservative_energy_scale_bytes * 8
    ) / total_weights
    return {
        "classes": classes,
        "weights": total_weights,
        "gross_crossfit_shape_gain_bpw": gross_gain / total_weights,
        "ideal_class_label_bpw_before_byte_rounding": class_bits_total / total_weights,
        "expert_local_side_bytes": local_bytes_total,
        "expert_local_side_bpw": local_bytes_total * 8 / total_weights,
        "shared_model_ledger_bytes": ledger,
        "shared_model_bpw_charged_over_six_experts": ledger["total"] * 8 / total_weights,
        "net_incremental_shape_gain_bpw": net,
        "net_gain_if_existing_energy_context_side_recharged_bpw": net_if_all_context_side_recharged,
        "optimistic_F_multiplier": 2.0 ** (-2.0 * net),
        "passes_0p8_information_gate": net >= TARGET_GAIN_BPW,
        "transform_weight_fraction": {
            str(depth): float(transform_weight[depth] / total_weights)
            for depth in DEPTHS
            if transform_weight[depth]
        },
        "fold_min_net_before_model_bpw": min(row["net_before_shared_model_bpw"] for row in fold_reports),
        "fold_max_net_before_model_bpw": max(row["net_before_shared_model_bpw"] for row in fold_reports),
        "folds": fold_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    lock = args.target_lock.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    rows = []
    for ordinal, (layer, expert, paths) in enumerate(source_triplets(lock, source_root)):
        row = load_expert(layer, expert, paths)
        rows.append(row)
        print(
            f"loaded {ordinal + 1}/6 L{layer} E{expert}; symbols={row.symbols.nbytes / 2**20:.1f} MiB",
            flush=True,
        )
    results = []
    for classes in CLASS_COUNTS:
        score = evaluate_classes(rows, classes)
        results.append(score)
        print(
            f"K={classes} gross={score['gross_crossfit_shape_gain_bpw']:.9f} "
            f"net={score['net_incremental_shape_gain_bpw']:.9f} "
            f"F={score['optimistic_F_multiplier']:.9f}",
            flush=True,
        )
    winner = max(results, key=lambda row: row["net_incremental_shape_gain_bpw"])
    report = {
        "schema": "expert_local_shape_partial_hadamard_information_oracle_v1",
        "status": "complete_cpu_only_crossfit_information_screen",
        "strict_ptq": True,
        "claim_boundary": (
            "cross-fitted discretized marginal entropy/Shannon-lower-bound oracle; "
            "not an operational quantizer and not reconstruction MSE"
        ),
        "target": {
            "lock_path": str(lock),
            "lock_sha256": sha256_file(lock),
            "source_root": str(source_root),
            "experts": len(rows),
            "weights": len(rows) * ROLES * ROWS * COLS,
        },
        "protocol": {
            "natural_group_values": COLS,
            "roles_after_exact_expert_local_klt": ["gate", "klt0", "klt1"],
            "energy_contexts": "8 equipopulous expert-local strata over all 2304 natural groups",
            "shape_features": [
                "mean_over_rms",
                "standardized_skew",
                "log_pearson_kurtosis",
                "tail_fraction_gt_2rms",
                "tail_fraction_gt_3rms",
                "tail_fraction_gt_4rms",
                "l1_over_rms",
                "log_max_over_rms",
            ],
            "shape_features_are_scale_invariant": True,
            "outer_evaluation": "leave held-out layer AND expert out; five experts train, sixth scores",
            "depth_selection": "training-only per role x stratum x shape class",
            "depths": {"0": "true identity", **{str(d): f"procedurally signed FWHT width {1 << d}" for d in range(1, 12)}},
            "density_bins": {"low": BIN_LOW, "high": BIN_HIGH, "step": BIN_STEP, "symbols": NBINS},
            "probability_serialization": "each table normalized to exactly 65535 positive integer frequencies",
            "gaussian_comparator": (
                "training-moment-matched separately for every role x energy stratum x shape class x depth; "
                "context RMS advantage excluded"
            ),
            "all_shape_class_and_model_bytes_charged": True,
            "existing_energy_context_reported_both_incrementally and conservatively_recharged": True,
        },
        "required_gain_bpw": TARGET_GAIN_BPW,
        "class_candidates": results,
        "winner_classes": winner["classes"],
        "winner_net_gain_bpw": winner["net_incremental_shape_gain_bpw"],
        "winner_optimistic_F_multiplier": winner["optimistic_F_multiplier"],
        "decision": (
            "SURVIVES_INFORMATION_GATE_DESIGN_CUPY_FINITE_RATE_PILOT"
            if winner["passes_0p8_information_gate"]
            else "KILL_SHAPE_PARTIAL_HADAMARD_BRANCH"
        ),
        "source_bindings": [
            {"layer": row.layer, "expert": row.expert, "klt_theta_fp64": row.theta, "files": row.source_rows}
            for row in rows
        ],
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "winner_classes": winner["classes"],
                "winner_net_gain_bpw": winner["net_incremental_shape_gain_bpw"],
                "winner_F": winner["optimistic_F_multiplier"],
                "decision": report["decision"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
