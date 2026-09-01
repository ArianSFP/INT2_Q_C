#!/usr/bin/env python3
"""CPU-only PrismQuant-style covariance oracle for Qwen MoE natural groups.

The unit is one 2048-weight natural row/column after the same expert-local
Gate/(Up,Down)-KLT staging used by STRATA.  Scale-invariant group features
select a Gaussian-mixture component.  A component then chooses, using only
the five training experts in an outer fold, one of:

* identity,
* a procedural signed partial FWHT of width 2..2048, or
* a component-trained repeated 32x32/64x64 KLT.

The held-out layer and expert are absent from component fitting, means,
covariance spectra, transform selection, and label probabilities.  Evaluation
uses a single global reverse-waterfill level and the *training* eigenvalue
model.  Held-out second moments determine actual distortion, so a covariance
pattern that does not transfer is penalized.  This remains an optimistic
Gaussian RD oracle, not an operational quantizer-MSE result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROWS = 768
COLS = 2048
ROLES = 3
STRATA = 8
CONTEXTS = ROLES * STRATA
FEATURES = 8
COMPONENTS = (1, 4)
RATES = (2.15, 2.30, 2.50)
FREQ_TOTAL = 65535
TARGET_GAIN_BPW = -0.5 * math.log2(0.8)
MODES = ("identity",) + tuple(f"h{1 << depth}" for depth in range(1, 12)) + ("klt32", "klt64")
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
    return (np.asarray(words, dtype=np.uint32) << np.uint32(16)).view(np.float32)


def exact_klt(up: np.ndarray, down_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    a = float(np.sum(np.asarray(up, dtype=np.float64) ** 2, dtype=np.float64))
    b = float(np.sum(np.asarray(down_t, dtype=np.float64) ** 2, dtype=np.float64))
    c = float(np.sum(np.asarray(up, dtype=np.float64) * down_t, dtype=np.float64))
    theta = 0.5 * math.atan2(2.0 * c, a - b)
    co, si = np.float32(math.cos(theta)), np.float32(math.sin(theta))
    return co * up + si * down_t, -si * up + co * down_t, theta


def contexts_and_scales(roles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.sum(np.asarray(roles, dtype=np.float64) ** 2, axis=2, dtype=np.float64)
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
                numerator = float(np.sum(energy[role, selected], dtype=np.float64))
                denominator = int(selected.sum()) * COLS
            else:
                pooled = labels == stratum
                numerator = float(np.sum(energy[pooled], dtype=np.float64))
                denominator = int(pooled.sum()) * COLS
            scales[role, stratum] = np.float16(math.sqrt(numerator / denominator))
    contexts = np.arange(ROLES, dtype=np.int16)[:, None] * STRATA + labels.astype(np.int16)
    group_scale = np.empty((ROLES, ROWS), dtype=np.float32)
    for role in range(ROLES):
        group_scale[role] = scales[role, labels[role]].astype(np.float32)
    return contexts.reshape(-1), group_scale.reshape(-1), energy.reshape(-1)


def shape_features(groups: np.ndarray) -> np.ndarray:
    x = np.asarray(groups, dtype=np.float64)
    mean = np.mean(x, axis=1, dtype=np.float64)
    centered = x - mean[:, None]
    var = np.mean(centered * centered, axis=1, dtype=np.float64)
    std = np.sqrt(np.maximum(var, np.finfo(float).tiny))
    z = centered / std[:, None]
    skew = np.mean(z * z * z, axis=1, dtype=np.float64)
    kurt = np.mean((z * z) ** 2, axis=1, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x, axis=1, dtype=np.float64))
    absolute = np.abs(x) / np.maximum(rms[:, None], np.finfo(float).tiny)
    return np.stack(
        (
            np.clip(mean / np.maximum(rms, np.finfo(float).tiny), -1.0, 1.0),
            np.clip(skew, -12.0, 12.0),
            np.log(np.clip(kurt, 1.0, 512.0)),
            np.mean(absolute > 2.0, axis=1),
            np.mean(absolute > 3.0, axis=1),
            np.mean(absolute > 4.0, axis=1),
            np.mean(absolute, axis=1),
            np.log(np.maximum(np.max(absolute, axis=1), 1.0)),
        ),
        axis=1,
    ).astype(np.float32)


def fit_kmeans(features: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.maximum(np.std(values, axis=0, dtype=np.float64), 1e-6)
    z = np.clip((values - mean) / std, -8.0, 8.0)
    if components == 1:
        return mean, std, np.mean(z, axis=0, keepdims=True)
    centers = [np.mean(z, axis=0)]
    nearest = np.sum((z - centers[0]) ** 2, axis=1)
    for _ in range(1, components):
        centers.append(z[int(np.argmax(nearest))].copy())
        nearest = np.minimum(nearest, np.sum((z - centers[-1]) ** 2, axis=1))
    centers = np.asarray(centers, dtype=np.float64)
    old = None
    for _ in range(40):
        labels = np.argmin(np.sum((z[:, None] - centers[None]) ** 2, axis=2), axis=1)
        updated = centers.copy()
        for component in range(components):
            selected = labels == component
            if np.any(selected):
                updated[component] = np.mean(z[selected], axis=0, dtype=np.float64)
        if old is not None and np.array_equal(labels, old) and np.allclose(updated, centers):
            centers = updated
            break
        old, centers = labels, updated
    return mean, std, centers


def assign(features: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std, centers = model
    z = np.clip((np.asarray(features, dtype=np.float64) - mean) / std, -8.0, 8.0)
    return np.argmin(np.sum((z[:, None] - centers[None]) ** 2, axis=2), axis=1).astype(np.uint8)


def frequencies(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    p = (counts + 0.5) / float(np.sum(counts + 0.5))
    available = FREQ_TOTAL - counts.size
    scaled = p * available
    extra = np.floor(scaled).astype(np.int64)
    result = extra + 1
    remainder = FREQ_TOTAL - int(np.sum(result))
    if remainder:
        order = np.argsort(-(scaled - extra), kind="stable")
        result[order[:remainder]] += 1
    return result.astype(np.uint16)


def seed_signs(components: int, component: int) -> np.ndarray:
    seed = int.from_bytes(
        hashlib.sha256(f"prism-group-cov-v1:{components}:{component}".encode()).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=COLS, dtype=np.int8).astype(np.float32) * 2.0 - 1.0)


def fwht_to_width(values: np.ndarray, width_target: int, signs: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32) * signs[None, :]
    width = 1
    factor = np.float32(1.0 / math.sqrt(2.0))
    while width < width_target:
        view = result.reshape(result.shape[0], -1, 2 * width)
        left = view[:, :, :width].copy()
        right = view[:, :, width:].copy()
        view[:, :, :width] = (left + right) * factor
        view[:, :, width:] = (left - right) * factor
        width *= 2
    return result


def repeated_klt_fit(values: np.ndarray, width: int) -> np.ndarray:
    chunks = np.asarray(values, dtype=np.float64).reshape(-1, width)
    covariance = (chunks.T @ chunks) / chunks.shape[0]
    _, vectors = np.linalg.eigh(covariance)
    # The physically stored FP16 matrix is used in both train and held-out
    # scoring; any loss of orthogonality is therefore charged in distortion.
    return vectors.astype(np.float16).astype(np.float32)


def transform(values: np.ndarray, mode: str, signs: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
    if mode == "identity":
        return np.asarray(values, dtype=np.float32)
    if mode.startswith("h"):
        return fwht_to_width(values, int(mode[1:]), signs)
    width = int(mode[3:])
    if basis is None or basis.shape != (width, width):
        raise ValueError("missing repeated KLT basis")
    chunks = np.asarray(values, dtype=np.float32).reshape(-1, width)
    return (chunks @ basis).reshape(values.shape).astype(np.float32)


def reverse_waterfill(variance: np.ndarray, rate: float) -> tuple[float, float]:
    values = np.maximum(np.asarray(variance, dtype=np.float64), 1e-30)
    lo = math.log(float(np.min(values))) - 80.0
    hi = math.log(float(np.max(values))) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        water = math.exp(mid)
        actual = float(np.mean(np.maximum(0.5 * np.log2(values / water), 0.0)))
        if actual > rate:
            lo = mid
        else:
            hi = mid
    water = math.exp(0.5 * (lo + hi))
    distortion = float(np.mean(np.minimum(values, water)))
    return distortion, water


@dataclass
class Expert:
    layer: int
    expert: int
    groups: np.ndarray
    features: np.ndarray
    contexts: np.ndarray
    scales: np.ndarray
    energy: np.ndarray
    theta: float
    files: list[dict]


def triplets(lock: Path, source_root: Path):
    document = json.loads(lock.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, int], dict[str, tuple[Path, dict]]] = {}
    for row in document["matrices"]:
        match = TARGET_RE.fullmatch(str(row["tensor"]))
        if not match:
            raise ValueError(row["tensor"])
        layer, expert, role = int(match.group(1)), int(match.group(2)), match.group(3)
        relative = row.get("output_relpath") or row.get("source_relpath")
        grouped.setdefault((layer, expert), {})[role] = (source_root / relative, row)
    return [(layer, expert, rows) for (layer, expert), rows in sorted(grouped.items())]


def load_expert(layer: int, expert: int, rows: dict[str, tuple[Path, dict]]) -> Expert:
    arrays = {}
    files = []
    for role in ("gate", "up", "down"):
        path, metadata = rows[role]
        digest = sha256_file(path)
        if digest != metadata["source_bf16_sha256"]:
            raise ValueError(f"hash mismatch {path}")
        array = bf16(path, tuple(int(v) for v in metadata["shape"]))
        arrays[role] = array if role != "down" else array.T.copy()
        files.append({"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest})
    k0, k1, theta = exact_klt(arrays["up"], arrays["down"])
    staged = np.stack((arrays["gate"], k0, k1)).astype(np.float32, copy=False)
    contexts, scales, energy = contexts_and_scales(staged)
    groups = staged.reshape(-1, COLS) / scales[:, None]
    return Expert(layer, expert, groups.astype(np.float32), shape_features(staged.reshape(-1, COLS)), contexts, scales, energy, theta, files)


def choose_component_model(
    values: np.ndarray, components: int, component: int
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None, dict]:
    # Mean storage is part of the model and its FP16 rounding is used here.
    mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float16).astype(np.float32)
    residual = np.asarray(values, dtype=np.float32) - mean[None, :]
    signs = seed_signs(components, component)
    candidates = []
    chosen_basis = None
    for mode in MODES:
        basis = None
        if mode.startswith("klt"):
            basis = repeated_klt_fit(residual, int(mode[3:]))
        transformed = transform(residual, mode, signs, basis)
        variance = np.mean(np.asarray(transformed, dtype=np.float64) ** 2, axis=0, dtype=np.float64)
        distortion, _ = reverse_waterfill(variance, 2.5)
        arithmetic = float(np.mean(variance))
        geometric = float(np.exp(np.mean(np.log(np.maximum(variance, 1e-30)))))
        candidates.append(
            {
                "mode": mode,
                "training_relative_rd_at_2p5": distortion / arithmetic,
                "training_F_at_2p5": distortion / arithmetic * 32.0,
                "training_diagonal_logvolume_gain_bpw": 0.5 * math.log2(arithmetic / geometric),
                "basis": basis,
                "variance": variance,
            }
        )
    winner = min(candidates, key=lambda row: (row["training_relative_rd_at_2p5"], row["mode"]))
    # Serialized FP16 eigenvalue model is exactly what drives rate allocation.
    variance = np.maximum(winner["variance"].astype(np.float16).astype(np.float64), 2.0 ** -24)
    chosen_basis = winner["basis"]
    compact = [
        {key: value for key, value in row.items() if key not in {"basis", "variance"}}
        for row in candidates
    ]
    return mean, variance, str(winner["mode"]), chosen_basis, {
        "chosen_mode": winner["mode"],
        "training_F_at_2p5": winner["training_F_at_2p5"],
        "training_diagonal_logvolume_gain_bpw": winner["training_diagonal_logvolume_gain_bpw"],
        "candidates": compact,
    }


def shared_model_bytes(components: int, modes: list[str]) -> dict[str, int]:
    header = 256
    means = components * COLS * 2
    spectra = components * COLS * 2
    bases = sum((int(mode[3:]) ** 2 * 2 if mode.startswith("klt") else 0) for mode in modes)
    component_probabilities = components * 2
    mode_map = components
    feature_model = (2 * FEATURES + components * FEATURES) * 2
    integrity = 32
    total = header + means + spectra + bases + component_probabilities + mode_map + feature_model + integrity
    return {
        "header": header,
        "component_means_fp16": means,
        "component_spectra_fp16": spectra,
        "component_klt_bases_fp16": bases,
        "component_probabilities_u16": component_probabilities,
        "mode_map": mode_map,
        "encoder_feature_model_fp16_conservatively_charged": feature_model,
        "integrity": integrity,
        "total": total,
    }


def solve_global_waterlevel(cells: list[dict], payload_rate: float) -> float:
    maximum = max(float(np.max(cell["train_actual_variance"])) for cell in cells)
    minimum = min(float(np.min(cell["train_actual_variance"])) for cell in cells)
    total_coefficients = sum(int(cell["groups"]) * COLS for cell in cells)
    lo, hi = math.log(max(minimum, 1e-30)) - 80.0, math.log(maximum) + 1.0
    for _ in range(110):
        middle = 0.5 * (lo + hi)
        water = math.exp(middle)
        total_rate = 0.0
        for cell in cells:
            total_rate += int(cell["groups"]) * float(
                np.sum(np.maximum(0.5 * np.log2(cell["train_actual_variance"] / water), 0.0))
            )
        actual = total_rate / total_coefficients
        if actual > payload_rate:
            lo = middle
        else:
            hi = middle
    return math.exp(0.5 * (lo + hi))


def heldout_rd(cells: list[dict], source_energy: float, payload_rate: float) -> dict:
    water = solve_global_waterlevel(cells, payload_rate)
    error = 0.0
    rate_sum = 0.0
    coefficients = 0
    for cell in cells:
        train = cell["train_actual_variance"]
        test = cell["test_actual_second_moment"]
        rates = np.maximum(0.5 * np.log2(train / water), 0.0)
        multiplier = np.where(rates > 0.0, np.exp2(-2.0 * rates), 1.0)
        count = int(cell["groups"])
        error += count * float(np.sum(test * multiplier, dtype=np.float64))
        rate_sum += count * float(np.sum(rates, dtype=np.float64))
        coefficients += count * COLS
    return {
        "payload_rate_realized_bpw": rate_sum / coefficients,
        "global_waterlevel": water,
        "relative_mse_oracle": error / source_energy,
    }


def evaluate(rows: list[Expert], components: int) -> dict:
    total_panel_weights = len(rows) * ROLES * ROWS * COLS
    fold_models = []
    # First build every outer-fold model and its held-out covariance cells.
    for heldout_index, heldout in enumerate(rows):
        train = [row for i, row in enumerate(rows) if i != heldout_index]
        if any(row.layer == heldout.layer or row.expert == heldout.expert for row in train):
            raise AssertionError("leave-layer/expert-out failure")
        training_groups = np.concatenate([row.groups for row in train], axis=0)
        training_features = np.concatenate([row.features for row in train], axis=0)
        classifier = fit_kmeans(training_features, components)
        train_labels = assign(training_features, classifier)
        test_labels = assign(heldout.features, classifier)
        counts = np.bincount(train_labels, minlength=components)
        label_frequencies = frequencies(counts)
        label_bits = float(
            np.sum(
                np.bincount(test_labels, minlength=components)
                * -np.log2(label_frequencies.astype(np.float64) / FREQ_TOTAL),
                dtype=np.float64,
            )
        ) if components > 1 else 0.0
        component_models = []
        cells = []
        heldout_logvolume_weighted = 0.0
        heldout_logvolume_groups = 0
        for component in range(components):
            train_selected = training_groups[train_labels == component]
            test_indices = np.flatnonzero(test_labels == component)
            if train_selected.shape[0] == 0 or test_indices.size == 0:
                continue
            mean, variance, mode, basis, selection = choose_component_model(
                train_selected, components, component
            )
            signs = seed_signs(components, component)
            test_residual = heldout.groups[test_indices] - mean[None, :]
            test_transformed = transform(test_residual, mode, signs, basis)
            # Held-out log-volume is descriptive only.  The operational oracle
            # below allocates from the serialized training spectrum.
            test_global_m2 = np.mean(
                np.asarray(test_transformed, dtype=np.float64) ** 2,
                axis=0,
                dtype=np.float64,
            )
            test_arithmetic = float(np.mean(test_global_m2))
            test_geometric = float(np.exp(np.mean(np.log(np.maximum(test_global_m2, 1e-30)))))
            heldout_gain = 0.5 * math.log2(test_arithmetic / test_geometric)
            heldout_logvolume_weighted += test_indices.size * heldout_gain
            heldout_logvolume_groups += int(test_indices.size)
            for context in range(CONTEXTS):
                mask = heldout.contexts[test_indices] == context
                if not np.any(mask):
                    continue
                selected_values = np.asarray(test_transformed[mask], dtype=np.float64)
                test_m2 = np.mean(selected_values * selected_values, axis=0, dtype=np.float64)
                scale2 = float(np.float64(heldout.scales[test_indices[mask][0]]) ** 2)
                cells.append(
                    {
                        "component": component,
                        "context": context,
                        "groups": int(np.sum(mask)),
                        "train_actual_variance": variance * scale2,
                        "test_actual_second_moment": test_m2 * scale2,
                    }
                )
            component_models.append(
                {
                    "component": component,
                    "training_groups": int(train_selected.shape[0]),
                    "heldout_groups": int(test_indices.size),
                    "mode": mode,
                    "basis_bytes_fp16": int(int(mode[3:]) ** 2 * 2) if mode.startswith("klt") else 0,
                    "selection": selection,
                    "heldout_diagonal_logvolume_gain_bpw": heldout_gain,
                }
            )
            del test_residual, test_transformed
        modes = [row["mode"] for row in component_models]
        ledger = shared_model_bytes(components, modes)
        # Complete expert-local side image: original uint3 energy labels,
        # role x stratum scales, component arithmetic stream, water level,
        # framing/model binding, and integrity.
        component_bytes = math.ceil(label_bits / 8.0) if components > 1 else 0
        local_ledger = {
            "energy_labels_uint3": ROLES * ROWS * 3 // 8,
            "context_scales_fp16": CONTEXTS * 2,
            "component_label_stream": component_bytes,
            "framing_model_binding_integrity": 96,
            "waterlevel_fp16": 2,
        }
        local_ledger["total"] = sum(local_ledger.values())
        model_bpw = ledger["total"] * 8 / total_panel_weights
        local_bpw = local_ledger["total"] * 8 / (ROLES * ROWS * COLS)
        rate_rows = []
        for rate in RATES:
            payload = rate - model_bpw - local_bpw
            rd = heldout_rd(cells, float(np.sum(heldout.energy, dtype=np.float64)), payload)
            rd.update(
                {
                    "physical_rate_bpw": rate,
                    "shared_model_bpw": model_bpw,
                    "expert_local_side_bpw": local_bpw,
                    "payload_rate_target_bpw": payload,
                    "gaussian_limit": 2.0 ** (-2.0 * rate),
                    "F": rd["relative_mse_oracle"] * 2.0 ** (2.0 * rate),
                }
            )
            rate_rows.append(rd)
        fold_models.append(
            {
                "heldout_layer": heldout.layer,
                "heldout_expert": heldout.expert,
                "training_layers": sorted(row.layer for row in train),
                "training_experts": sorted(row.expert for row in train),
                "component_label_bits": label_bits,
                "component_label_bpw_before_rounding": label_bits / (ROLES * ROWS * COLS),
                "component_histogram": np.bincount(test_labels, minlength=components).astype(int).tolist(),
                "shared_model_ledger_bytes": ledger,
                "expert_local_ledger_bytes": local_ledger,
                "components": component_models,
                "heldout_conditional_diagonal_logvolume_gain_bpw": (
                    heldout_logvolume_weighted / heldout_logvolume_groups
                ),
                "rates": rate_rows,
            }
        )
        print(
            f"K={components} fold {heldout_index + 1}/6 L{heldout.layer} E{heldout.expert} "
            f"label={label_bits/(ROLES*ROWS*COLS):.6f} "
            f"logvol={fold_models[-1]['heldout_conditional_diagonal_logvolume_gain_bpw']:.6f} "
            f"F2.5={rate_rows[-1]['F']:.6f} modes={modes}",
            flush=True,
        )
    aggregate_rates = []
    for rate_index, rate in enumerate(RATES):
        # All experts carry exactly the same number of coefficients and the
        # reported relative MSEs share an energy normalization close enough to
        # use the equal-expert mean for this diagnostic.  We also report the
        # worst fold, which is the safer deployment gate.
        values = [fold["rates"][rate_index]["relative_mse_oracle"] for fold in fold_models]
        aggregate = float(np.mean(values))
        aggregate_rates.append(
            {
                "physical_rate_bpw": rate,
                "mean_crossfit_relative_mse_oracle": aggregate,
                "mean_crossfit_F": aggregate * 2.0 ** (2.0 * rate),
                "worst_fold_F": max(fold["rates"][rate_index]["F"] for fold in fold_models),
                "best_fold_F": min(fold["rates"][rate_index]["F"] for fold in fold_models),
            }
        )
    label_bpw = float(np.mean([row["component_label_bpw_before_rounding"] for row in fold_models]))
    logvolume = float(np.mean([row["heldout_conditional_diagonal_logvolume_gain_bpw"] for row in fold_models]))
    best = min(aggregate_rates, key=lambda row: row["mean_crossfit_F"])
    return {
        "components": components,
        "mean_component_label_bpw_before_byte_rounding": label_bpw,
        "mean_heldout_conditional_diagonal_logvolume_gain_bpw": logvolume,
        "net_logvolume_gain_after_ideal_label_only_bpw": logvolume - label_bpw,
        "rates": aggregate_rates,
        "best_crossfit_F_over_allowed_rates": best["mean_crossfit_F"],
        "passes_F_0p8": best["mean_crossfit_F"] <= 0.8,
        "folds": fold_models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = args.target_lock.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    rows = []
    for ordinal, (layer, expert, triplet) in enumerate(triplets(lock, source_root)):
        rows.append(load_expert(layer, expert, triplet))
        print(f"loaded {ordinal + 1}/6 L{layer} E{expert}", flush=True)
    candidates = []
    for components in COMPONENTS:
        candidates.append(evaluate(rows, components))
    winner = min(candidates, key=lambda row: row["best_crossfit_F_over_allowed_rates"])
    dense_basis_bytes = COLS * COLS * 2
    report = {
        "schema": "qwen_expert_local_prism_group_covariance_oracle_v1",
        "status": "complete_cpu_only_leave_layer_expert_out_screen",
        "strict_ptq": True,
        "claim_boundary": (
            "optimistic conditional-Gaussian reverse-waterfill bound using fitted covariance spectra; "
            "not an operational scalar/vector quantizer reconstruction"
        ),
        "target": {
            "lock": str(lock),
            "lock_sha256": sha256_file(lock),
            "source_root": str(source_root),
            "experts": len(rows),
            "weights": len(rows) * ROLES * ROWS * COLS,
        },
        "protocol": {
            "group_values": COLS,
            "mixture_label_unit": "one component label per 2048-weight natural group",
            "component_classifier": "scale-invariant eight-feature deterministic k-means",
            "outer_fold": "held-out layer AND expert excluded from every fitted object",
            "candidate_transforms": list(MODES),
            "transform_selection": "training-only minimum ideal Gaussian RD at 2.5 payload bpw per component",
            "allocation": "single global reverse-waterfill level driven by serialized training FP16 spectra",
            "evaluation": "held-out residual second moments in the fixed training transform",
            "physical_rate_includes": [
                "uint3 expert-local energy labels",
                "FP16 role x energy-stratum scales",
                "component label stream",
                "FP16 component means and covariance spectra",
                "any selected repeated-KLT bases",
                "framing/model binding/integrity",
            ],
        },
        "required_gain_bpw": TARGET_GAIN_BPW,
        "candidates": candidates,
        "winner_components": winner["components"],
        "winner_best_crossfit_F": winner["best_crossfit_F_over_allowed_rates"],
        "dense_2048_klt_storage_lower_bound": {
            "bytes_per_component_fp16": dense_basis_bytes,
            "bpw_per_component_if_amortized_only_over_six_experts": dense_basis_bytes * 8 / (len(rows) * ROLES * ROWS * COLS),
            "note": "one dense FP16 component KLT already consumes almost the entire allowed payload budget; it was not promoted",
        },
        "decision": (
            "SURVIVES_PRISMQUANT_INFORMATION_GATE"
            if winner["passes_F_0p8"]
            else "KILL_EXPERT_LOCAL_PRISM_GROUP_COVARIANCE_BRANCH"
        ),
        "source_bindings": [
            {"layer": row.layer, "expert": row.expert, "klt_theta_fp64": row.theta, "files": row.files}
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
                "winner_components": winner["components"],
                "winner_F": winner["best_crossfit_F_over_allowed_rates"],
                "decision": report["decision"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
