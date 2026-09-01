#!/usr/bin/env python3
"""CPU-only oracle gate for expert-local semantic nonlinear/VQ codecs.

The source files are opened read-only.  No CuPy/CUDA import is performed.

Two deliberately optimistic gates are implemented:

``predictor``
    For each aligned (gate, up, down.T) coefficient, reveal the other two
    roles losslessly and predict the third with a cross-fitted 2-D conditional
    mean table.  Repeating this independently in all three directions is not
    a realizable codec (it gives every direction free side information), so a
    failure to find 0.16 bit/weight of conditional variance advantage is a
    strong early-stop result.

``rvq``
    Cross-fit a source-domain residual-additive VQ on vectors containing eight
    aligned semantic triples (24 scalars).  Every stage has 16 FP16 centroids
    and a fixed four-bit index.  The rate ledger charges the complete private
    expert codebook, indices, and a 128-byte header.  This is a statistical
    pilot, not an emitted production bitstream.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


LAYERS_EXPERTS = ((5, 18), (12, 7), (18, 20), (28, 83), (36, 76), (45, 41))
ROWS = 768
COLS = 2048
ROLE_COUNT = 3
TRIPLES_PER_VECTOR = 8
DIMENSION = ROLE_COUNT * TRIPLES_PER_VECTOR
FULL_EXPERT_SCALARS = ROLE_COUNT * ROWS * COLS


def read_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Decode a raw little-endian BF16 file without ever opening it writable."""
    words = np.fromfile(path, dtype="<u2")
    expected = math.prod(shape)
    if words.size != expected:
        raise ValueError(f"{path}: got {words.size} BF16 words, expected {expected}")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite value")
    return values.reshape(shape)


def load_triplet(source_dir: Path, layer: int, expert: int) -> np.ndarray:
    stem = f"model.layers.{layer}.mlp.experts.{expert}"
    gate = read_bf16(source_dir / f"{stem}.gate_proj.weight.bf16.bin", (ROWS, COLS))
    up = read_bf16(source_dir / f"{stem}.up_proj.weight.bf16.bin", (ROWS, COLS))
    down = read_bf16(source_dir / f"{stem}.down_proj.weight.bf16.bin", (COLS, ROWS)).T
    # [coefficient, role], preserving the trained intermediate-neuron alignment.
    return np.stack((gate.reshape(-1), up.reshape(-1), down.reshape(-1)), axis=1)


def deterministic_fold(n: int, seed: int) -> np.ndarray:
    # SplitMix-style integer hash.  It avoids making a spatial checkerboard the
    # train/test boundary while remaining reproducible and source-independent.
    x = np.arange(n, dtype=np.uint64) + np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x & np.uint64(1)).astype(bool)


def strict_edges(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    # searchsorted tolerates duplicate BF16 quantiles; the declared table still
    # pays for all bins, which is conservative for the rate ledger.
    return edges.astype(np.float32)


def conditional_table_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, int]:
    edge0 = strict_edges(train_x[:, 0], bins)
    edge1 = strict_edges(train_x[:, 1], bins)
    a = np.searchsorted(edge0, train_x[:, 0], side="right")
    b = np.searchsorted(edge1, train_x[:, 1], side="right")
    cell = a.astype(np.int64) * bins + b
    count = np.bincount(cell, minlength=bins * bins)
    total = np.bincount(cell, weights=train_y, minlength=bins * bins)
    global_mean = float(np.mean(train_y, dtype=np.float64))
    # A tiny empirical-Bayes shrink avoids empty-cell pathologies.  Trying
    # several resolutions later is deliberately favorable to the oracle.
    prior = 2.0
    means = (total + prior * global_mean) / (count + prior)
    ta = np.searchsorted(edge0, test_x[:, 0], side="right")
    tb = np.searchsorted(edge1, test_x[:, 1], side="right")
    prediction = means[ta.astype(np.int64) * bins + tb].astype(np.float32)
    return prediction, int(np.count_nonzero(count))


def predictor_gate(source_dir: Path, bins_grid: list[int]) -> dict[str, object]:
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    aggregate = {
        bins: {"sse": np.zeros(ROLE_COUNT), "energy": np.zeros(ROLE_COUNT)}
        for bins in bins_grid
    }
    for expert_ordinal, (layer, expert) in enumerate(LAYERS_EXPERTS):
        values = load_triplet(source_dir, layer, expert)
        fold = deterministic_fold(len(values), 0x5EED + expert_ordinal * 7919)
        energy = np.sum(values.astype(np.float64) ** 2, axis=0)
        expert_record: dict[str, object] = {
            "layer": layer,
            "expert": expert,
            "coefficients_per_role": len(values),
            "role_energy": energy.tolist(),
            "resolutions": [],
        }
        for bins in bins_grid:
            sse = np.zeros(ROLE_COUNT, dtype=np.float64)
            occupied: list[int] = []
            for target in range(ROLE_COUNT):
                predictors = [r for r in range(ROLE_COUNT) if r != target]
                for heldout in (False, True):
                    train = fold != heldout
                    test = fold == heldout
                    pred, used = conditional_table_predict(
                        values[train][:, predictors], values[train, target],
                        values[test][:, predictors], bins,
                    )
                    error = values[test, target].astype(np.float64) - pred.astype(np.float64)
                    sse[target] += np.dot(error, error)
                    occupied.append(used)
            aggregate[bins]["sse"] += sse
            aggregate[bins]["energy"] += energy
            predictor_bits = (
                3 * bins * bins * 16
                + 3 * 2 * (bins - 1) * 16
                + 128 * 8
            )
            expert_record["resolutions"].append({
                "bins_per_predictor_axis": bins,
                "impossible_free_sideinfo_relative_mse": float(sse.sum() / energy.sum()),
                "directional_residual_variance_ratios": (sse / energy).tolist(),
                "occupied_cells_across_six_fits": occupied,
                "charged_predictor_bits": predictor_bits,
                "predictor_bpw_if_private_to_this_expert": predictor_bits / FULL_EXPERT_SCALARS,
            })
        rows.append(expert_record)

    aggregate_rows = []
    for bins in bins_grid:
        sse = aggregate[bins]["sse"]
        energy = aggregate[bins]["energy"]
        ratios = sse / energy
        # This additionally impossible ledger conditions all three roles on the
        # other two at once.  It is more favorable than any sequential codec.
        advantage = float(np.sum(0.5 * np.log2(1.0 / ratios)) / ROLE_COUNT)
        predictor_bits = (
            3 * bins * bins * 16              # three conditional-mean tables
            + 3 * 2 * (bins - 1) * 16         # two FP16 edge arrays per direction
            + 128 * 8                         # private format/header allowance
        )
        aggregate_rows.append({
            "bins_per_predictor_axis": bins,
            "impossible_free_sideinfo_relative_mse": float(sse.sum() / energy.sum()),
            "energy_improvement_percent": float(100.0 * (1.0 - sse.sum() / energy.sum())),
            "directional_residual_variance_ratios": ratios.tolist(),
            "impossible_conditional_advantage_bpw": advantage,
            "passes_0p16_bpw_gate": bool(advantage >= 0.16),
            "passes_19p4_energy_gate": bool(sse.sum() / energy.sum() <= 0.806),
            "charged_predictor_bits_per_expert": predictor_bits,
            "private_predictor_bpw": predictor_bits / FULL_EXPERT_SCALARS,
            "net_advantage_after_private_table_charge_bpw": advantage
            - predictor_bits / FULL_EXPERT_SCALARS,
        })
    best = max(aggregate_rows, key=lambda row: row["impossible_conditional_advantage_bpw"])
    return {
        "gate": "cross-fitted same-coordinate nonlinear semantic-role predictor",
        "optimism": (
            "Each direction receives the other two BF16 roles losslessly, all three "
            "directions are counted simultaneously, and prediction tables are first "
            "tested without rate charge. A realizable codec cannot obtain this oracle."
        ),
        "roles": ["gate", "up", "down_transposed"],
        "experts": rows,
        "aggregate": aggregate_rows,
        "best_optimistic_resolution": best,
        "decision": {
            "required_conditional_advantage_bpw": 0.16,
            "required_energy_improvement_percent": 19.4,
            "passes": bool(
                best["impossible_conditional_advantage_bpw"] >= 0.16
                and best["energy_improvement_percent"] >= 19.4
            ),
        },
        "seconds": time.perf_counter() - started,
    }


def nearest_indices(x: np.ndarray, centers: np.ndarray, chunk: int = 8192) -> np.ndarray:
    result = np.empty(len(x), dtype=np.uint8)
    c2 = np.sum(centers.astype(np.float64) ** 2, axis=1)
    for start in range(0, len(x), chunk):
        block = x[start : start + chunk]
        # Keep the large product FP32, but choose using an FP64-stable distance.
        d2 = (
            np.sum(block.astype(np.float64) ** 2, axis=1)[:, None]
            + c2[None, :]
            - 2.0 * block.astype(np.float64) @ centers.astype(np.float64).T
        )
        result[start : start + len(block)] = np.argmin(d2, axis=1).astype(np.uint8)
    return result


def fit_k16(
    residual: np.ndarray,
    seed: int,
    sample_count: int,
    iterations: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(residual) > sample_count:
        sample = residual[rng.choice(len(residual), sample_count, replace=False)].copy()
    else:
        sample = residual.copy()
    # Farthest-point initialization is deterministic after the first draw and
    # materially stronger than random seeds for the small K=16 dictionary.
    centers = np.empty((16, residual.shape[1]), dtype=np.float32)
    centers[0] = sample[rng.integers(len(sample))]
    nearest = np.sum((sample - centers[0]) ** 2, axis=1)
    for k in range(1, 16):
        # Squared-distance sampling (kmeans++) without constructing a huge CDF.
        total = float(np.sum(nearest, dtype=np.float64))
        if not total > 0:
            centers[k] = sample[rng.integers(len(sample))]
        else:
            pick = float(rng.random()) * total
            idx = int(np.searchsorted(np.cumsum(nearest, dtype=np.float64), pick))
            centers[k] = sample[min(idx, len(sample) - 1)]
        nearest = np.minimum(nearest, np.sum((sample - centers[k]) ** 2, axis=1))
    for _ in range(iterations):
        assignment = nearest_indices(sample, centers)
        updated = centers.copy()
        for k in range(16):
            members = assignment == k
            if np.any(members):
                updated[k] = np.mean(sample[members], axis=0, dtype=np.float64)
        # The production proposal stores the complete private codebook as FP16.
        centers = updated.astype(np.float16).astype(np.float32)
    return centers


def semantic_vectors(values: np.ndarray, seed: int) -> np.ndarray:
    if len(values) % TRIPLES_PER_VECTOR:
        raise AssertionError(len(values))
    # A source-independent permutation prevents contiguous memory layout from
    # being privileged.  The seed is format metadata, not per-expert payload.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    return values[order].reshape(-1, DIMENSION).astype(np.float32)


def rvq_gate(
    source_dir: Path,
    stages: int,
    sample_count: int,
    iterations: int,
) -> dict[str, object]:
    started = time.perf_counter()
    experts: list[dict[str, object]] = []
    total_error = np.zeros(stages, dtype=np.float64)
    total_energy = 0.0
    for ordinal, (layer, expert) in enumerate(LAYERS_EXPERTS):
        values = load_triplet(source_dir, layer, expert)
        vectors = semantic_vectors(values, 0xADD171E + ordinal * 104729)
        fold = deterministic_fold(len(vectors), 0xC0DEC + ordinal * 8191)
        train = vectors[~fold].copy()
        test = vectors[fold].copy()
        train_residual = train.copy()
        test_residual = test.copy()
        test_energy = float(np.sum(test.astype(np.float64) ** 2))
        total_energy += test_energy
        history: list[dict[str, object]] = []
        for stage in range(stages):
            centers = fit_k16(
                train_residual,
                seed=0x51A6E + ordinal * 1009 + stage,
                sample_count=sample_count,
                iterations=iterations,
            )
            train_idx = nearest_indices(train_residual, centers)
            test_idx = nearest_indices(test_residual, centers)
            train_residual -= centers[train_idx]
            test_residual -= centers[test_idx]
            test_sse = float(np.sum(test_residual.astype(np.float64) ** 2))
            total_error[stage] += test_sse
            history.append({
                "stage_count": stage + 1,
                "cross_fitted_relative_mse": test_sse / test_energy,
                "occupied_train_centroids": int(np.unique(train_idx).size),
                "occupied_test_centroids": int(np.unique(test_idx).size),
            })
        experts.append({
            "layer": layer,
            "expert": expert,
            "train_vectors": len(train),
            "heldout_vectors": len(test),
            "vector_dimension": DIMENSION,
            "history": history,
        })

    ledger: list[dict[str, object]] = []
    for stage in range(stages):
        count = stage + 1
        index_bits = count * 4 * (ROWS * COLS // TRIPLES_PER_VECTOR)
        codebook_bits = count * 16 * DIMENSION * 16
        header_bits = 128 * 8
        physical_bits = index_bits + codebook_bits + header_bits
        rate = physical_bits / FULL_EXPERT_SCALARS
        mse = total_error[stage] / total_energy
        gaussian = 2.0 ** (-2.0 * rate)
        target = 0.8 * gaussian
        ledger.append({
            "stage_count": count,
            "fixed_index_bits_per_expert": index_bits,
            "fp16_codebook_bits_per_expert": codebook_bits,
            "header_bits_per_expert": header_bits,
            "exact_physical_bpw": rate,
            "cross_fitted_relative_mse": mse,
            "gaussian_assumed_limit": gaussian,
            "target_20_percent_below_gaussian": target,
            "normalized_factor_mse_times_2pow2r": mse / gaussian,
            "passes_rate_window": 2.15 <= rate <= 2.5,
            "passes_rate_relative_target": 2.15 <= rate <= 2.5 and mse <= target,
            "expert_cold_read_bytes": physical_bits // 8,
            "expert_read_amplification": 1.0,
        })
    feasible = [row for row in ledger if row["passes_rate_window"]]
    best = min(feasible, key=lambda row: row["normalized_factor_mse_times_2pow2r"]) if feasible else None
    return {
        "gate": "cross-fitted source-domain semantic residual-additive VQ",
        "architecture": {
            "vector": "8 aligned (gate, up, down.T) triples",
            "dimension": DIMENSION,
            "stages": stages,
            "centroids_per_stage": 16,
            "index_bits_per_vector_per_stage": 4,
            "centroid_storage": "FP16, private to one expert",
            "assignment": "greedy residual nearest-centroid",
            "source_domain": True,
            "strict_ptq": True,
        },
        "experts": experts,
        "aggregate_ledger": ledger,
        "best_in_rate_window": best,
        "decision": {
            "required_normalized_factor": 0.8,
            "passes": bool(best and best["passes_rate_relative_target"]),
        },
        "read_implication": (
            "Each expert owns one contiguous index stream plus its private FP16 "
            "codebooks and header; no other expert is touched, so logical and cold "
            "artifact read amplification are exactly 1.0x (page rounding excluded)."
        ),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("predictor", "rvq", "both"), default="predictor")
    parser.add_argument("--bins", default="16,32,64,128")
    parser.add_argument("--stages", type=int, default=14)
    parser.add_argument("--sample-count", type=int, default=32768)
    parser.add_argument("--lloyd-iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, object] = {
        "protocol": {
            "cpu_only": True,
            "cupy_imported": False,
            "source_files_opened_read_only": True,
            "panel": [{"layer": l, "expert": e} for l, e in LAYERS_EXPERTS],
        }
    }
    if args.mode in ("predictor", "both"):
        result["predictor"] = predictor_gate(
            args.source_dir, [int(item) for item in args.bins.split(",")]
        )
    if args.mode in ("rvq", "both"):
        result["rvq"] = rvq_gate(
            args.source_dir, args.stages, args.sample_count, args.lloyd_iterations
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
