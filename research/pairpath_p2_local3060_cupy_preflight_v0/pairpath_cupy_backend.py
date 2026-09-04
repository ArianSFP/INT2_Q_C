"""Source-free CuPy backend for the PAIRPATH-P2 optimistic oracle.

This module has no payload locator, model name, network client, authority, or
production entry point.  It mirrors the source-only PAIRPATH-P2 oracle
semantics over caller-supplied finite arrays:

* two experts and two decoder-visible roles (Up and Down);
* four reconstruction labels per weight;
* role-conditioned nearest-label mutual information;
* one global Up/Down rate-distortion multiplier;
* the same deterministic symmetric multistart bank for independent and joint
  models; and
* deterministic lowest-index label selection on exact ties.

The expensive label-update step and exact integer counts execute in CuPy.
Candidate scoring is deliberately performed in canonical NumPy/FP64 order so
that GPU reduction association cannot change the selected multistart.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np


ALPHABET = 4
ROLES = ("up", "down")
BLOCK_VALUES = 2048
MAX_ALTERNATIONS = 8
ORACLE_EARLY_KILL_BPW = 0.045
REQUIRED_UPDOWN_GAIN_BPW = 0.22933495044437174
ORACLE_ENGINEERING_MARGIN_BPW = 0.27
FIXED_ASSIGNMENT_MI_REQUIRED_BITS_PER_PAIR = 0.4586699008887435
LEVELS_RMS = np.asarray(
    (-1.510417608, -0.452780039, 0.452780039, 1.510417608), dtype=np.float64
)
LAMBDA_GRID = tuple(Fraction(n, 4096) for n in
                    (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))


class BackendError(ValueError):
    """Fail-closed input or semantic error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendError(message)


def validate_values(values: np.ndarray) -> np.ndarray:
    """Return canonical `[expert, role, coordinate]` FP64 source values."""
    x = np.asarray(values)
    require(x.dtype == np.float64 and x.ndim == 3 and x.shape[0] == 2 and
            x.shape[1] == len(ROLES) and x.shape[2] > 0 and
            bool(np.all(np.isfinite(x))),
            "canonical finite FP64 [2,2,N] Up/Down values")
    return np.ascontiguousarray(x.astype("<f8", copy=False))


def validate_role(values: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values)
    reconstruction = np.asarray(levels)
    require(x.dtype == np.float64 and x.ndim == 2 and x.shape[0] == 2 and
            x.shape[1] > 0 and bool(np.all(np.isfinite(x))),
            "finite FP64 [2,N] role values")
    require(reconstruction.dtype == np.float64 and
            reconstruction.shape == x.shape + (ALPHABET,) and
            bool(np.all(np.isfinite(reconstruction))),
            "finite FP64 [2,N,4] reconstruction levels")
    return np.ascontiguousarray(x), np.ascontiguousarray(reconstruction)


def estimate_scale_bits(values: np.ndarray) -> np.ndarray:
    """Binary16 block-RMS scales, matching PAIRPATH-P2 source semantics."""
    x = validate_values(values)
    require(x.shape[2] % BLOCK_VALUES == 0, "coordinate count must be block aligned")
    blocks = x.shape[2] // BLOCK_VALUES
    result = np.empty((2, len(ROLES), blocks), dtype=np.uint16)
    smallest = np.float16(np.finfo(np.float16).tiny)
    for expert in range(2):
        for role in range(len(ROLES)):
            for block in range(blocks):
                row = x[expert, role, block * BLOCK_VALUES:(block + 1) * BLOCK_VALUES]
                rms = math.sqrt(float(np.dot(row, row)) / BLOCK_VALUES)
                quantized = np.float16(rms) if rms > 0 else smallest
                if not np.isfinite(quantized) or quantized <= 0:
                    quantized = np.float16(np.finfo(np.float16).max)
                result[expert, role, block] = np.asarray(
                    quantized, dtype=np.float16).view(np.uint16)
    return result


def levels_from_scales(scale_bits: np.ndarray, coordinates: int) -> np.ndarray:
    bits = np.asarray(scale_bits)
    require(bits.dtype == np.uint16 and bits.shape ==
            (2, len(ROLES), coordinates // BLOCK_VALUES) and
            coordinates % BLOCK_VALUES == 0, "scale geometry")
    decoded = bits.view(np.float16).astype(np.float64)
    require(bool(np.all(np.isfinite(decoded))) and bool(np.all(decoded > 0)),
            "decoded scales")
    per_coordinate = np.repeat(decoded, BLOCK_VALUES, axis=2)
    return np.ascontiguousarray(per_coordinate[:, :, :, None] *
                                LEVELS_RMS[None, None, None, :])


def prepare_levels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = validate_values(values)
    scales = estimate_scale_bits(x)
    return scales, levels_from_scales(scales, x.shape[2])


def entropy_bits(counts: Iterable[int]) -> float:
    c = np.asarray(tuple(counts), dtype=np.float64)
    total = float(np.sum(c, dtype=np.float64))
    require(c.ndim == 1 and total > 0 and bool(np.all(c >= 0)), "entropy counts")
    probability = c[c > 0] / total
    return float(-np.sum(probability * np.log2(probability), dtype=np.float64))


def nearest_labels(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    x, reconstruction = validate_role(values, levels)
    # NumPy argmin and CuPy argmin both choose the first (lowest) index on ties.
    return np.argmin((x[:, :, None] - reconstruction) ** 2, axis=2).astype(np.uint8)


def symmetric_initializations(values: np.ndarray, levels: np.ndarray) -> list[np.ndarray]:
    """Exact ordered PAIRPATH multistart bank, deduplicated by label bytes."""
    x, reconstruction = validate_role(values, levels)
    distortion = (x[:, :, None] - reconstruction) ** 2
    nearest = np.argmin(distortion, axis=2).astype(np.uint8)
    equal = np.argmin(distortion[0] + distortion[1], axis=1).astype(np.uint8)
    starts = [nearest, np.stack((equal, equal)).astype(np.uint8)]
    for label0 in range(ALPHABET):
        for label1 in range(ALPHABET):
            starts.append(np.stack((
                np.full(x.shape[1], label0, dtype=np.uint8),
                np.full(x.shape[1], label1, dtype=np.uint8),
            )))
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for start in starts:
        key = start.tobytes(order="C")
        if key not in seen:
            seen.add(key)
            unique.append(start)
    return unique


def label_hash(labels: np.ndarray) -> str:
    q = np.ascontiguousarray(np.asarray(labels, dtype=np.uint8))
    return hashlib.sha256(q.tobytes(order="C")).hexdigest()


def _counts_cpu(labels: np.ndarray, joint: bool) -> np.ndarray:
    q = np.asarray(labels, dtype=np.uint8)
    require(q.ndim == 2 and q.shape[0] == 2 and bool(np.all(q < ALPHABET)),
            "label geometry/range")
    if joint:
        index = q[0].astype(np.int16) * ALPHABET + q[1]
        return np.bincount(index, minlength=ALPHABET * ALPHABET).astype(np.int64)
    return np.stack([np.bincount(q[e], minlength=ALPHABET)
                     for e in range(2)]).astype(np.int64)


def score_labels(values: np.ndarray, levels: np.ndarray, labels: np.ndarray,
                 bit_weight: float, joint: bool) -> dict[str, Any]:
    """Canonical NumPy score and exact empirical entropy."""
    x, reconstruction = validate_role(values, levels)
    q = np.asarray(labels, dtype=np.uint8)
    require(q.shape == x.shape and bool(np.all(q < ALPHABET)), "score labels")
    decoded = np.take_along_axis(reconstruction, q[:, :, None], axis=2)[:, :, 0]
    sse = float(np.sum((x - decoded) ** 2, dtype=np.float64))
    counts = _counts_cpu(q, joint)
    if joint:
        rate = entropy_bits(counts) / 2.0
    else:
        rate = sum(entropy_bits(row) for row in counts) / 2.0
    total_bits = rate * x.size
    return {"objective": sse + float(bit_weight) * total_bits,
            "sse": sse, "rate_bpw": rate, "counts": counts}


def _lengths_from_counts(counts: np.ndarray, symbols: int) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    total = int(np.sum(c, dtype=np.float64))
    require(c.shape[-1] == symbols and total > 0 and bool(np.all(c >= 0)),
            "smoothed length counts")
    # For independent counts this is called once per expert, so total=N.
    return -np.log2((c + 0.5) / (total + 0.5 * symbols))


def update_labels_cpu(values: np.ndarray, levels: np.ndarray, labels: np.ndarray,
                      bit_weight: float, joint: bool) -> tuple[np.ndarray, np.ndarray]:
    """Independent reference update with deterministic low-index ties."""
    x, reconstruction = validate_role(values, levels)
    counts = _counts_cpu(labels, joint)
    if joint:
        lengths = _lengths_from_counts(counts, ALPHABET * ALPHABET)
        costs = ((x[0, :, None, None] - reconstruction[0, :, :, None]) ** 2 +
                 (x[1, :, None, None] - reconstruction[1, :, None, :]) ** 2 +
                 float(bit_weight) * lengths.reshape(1, ALPHABET, ALPHABET))
        selected = np.argmin(costs.reshape(x.shape[1], -1), axis=1)
        result = np.stack((selected // ALPHABET, selected % ALPHABET)).astype(np.uint8)
    else:
        result = np.empty_like(labels, dtype=np.uint8)
        for expert in range(2):
            lengths = _lengths_from_counts(counts[expert], ALPHABET)
            costs = ((x[expert, :, None] - reconstruction[expert]) ** 2 +
                     float(bit_weight) * lengths[None, :])
            result[expert] = np.argmin(costs, axis=1).astype(np.uint8)
    return result, counts


def _cupy_counts(cp: Any, labels: Any, joint: bool) -> tuple[np.ndarray, Any]:
    if joint:
        index = labels[0].astype(cp.int16) * ALPHABET + labels[1]
        counts_gpu = cp.bincount(index, minlength=ALPHABET * ALPHABET)
    else:
        counts_gpu = cp.stack(tuple(cp.bincount(labels[e], minlength=ALPHABET)
                                    for e in range(2)))
    return cp.asnumpy(counts_gpu).astype(np.int64, copy=False), counts_gpu


def update_labels_cupy_from_distortion(cp: Any, distortion_gpu: Any, labels_gpu: Any,
                                       bit_weight: float, joint: bool,
                                       chunk_coordinates: int = 32768) -> tuple[Any, np.ndarray]:
    """Chunked CuPy label update and exact GPU integer counts.

    `distortion_gpu` is `[2,N,4]` FP64 and remains resident across iterations.
    At most `[chunk,16]` FP64 joint costs are materialized.
    """
    require(isinstance(chunk_coordinates, int) and chunk_coordinates > 0,
            "positive chunk size")
    require(tuple(distortion_gpu.shape[:1]) == (2,) and
            int(distortion_gpu.shape[2]) == ALPHABET and
            tuple(labels_gpu.shape) == tuple(distortion_gpu.shape[:2]),
            "CuPy distortion/label geometry")
    coordinates = int(distortion_gpu.shape[1])
    counts, _ = _cupy_counts(cp, labels_gpu, joint)
    result = cp.empty((2, coordinates), dtype=cp.uint8)
    if joint:
        lengths = cp.asarray(_lengths_from_counts(counts, ALPHABET * ALPHABET),
                             dtype=cp.float64).reshape(1, ALPHABET, ALPHABET)
        for start in range(0, coordinates, chunk_coordinates):
            stop = min(coordinates, start + chunk_coordinates)
            costs = (distortion_gpu[0, start:stop, :, None] +
                     distortion_gpu[1, start:stop, None, :] +
                     float(bit_weight) * lengths)
            selected = cp.argmin(costs.reshape(stop - start, ALPHABET * ALPHABET), axis=1)
            result[0, start:stop] = (selected // ALPHABET).astype(cp.uint8)
            result[1, start:stop] = (selected % ALPHABET).astype(cp.uint8)
    else:
        for expert in range(2):
            lengths = cp.asarray(_lengths_from_counts(counts[expert], ALPHABET),
                                 dtype=cp.float64).reshape(1, ALPHABET)
            for start in range(0, coordinates, chunk_coordinates):
                stop = min(coordinates, start + chunk_coordinates)
                costs = distortion_gpu[expert, start:stop] + float(bit_weight) * lengths
                result[expert, start:stop] = cp.argmin(costs, axis=1).astype(cp.uint8)
    return result, counts


def solve_role_cpu(values: np.ndarray, levels: np.ndarray, bit_weight: float,
                   joint: bool) -> dict[str, Any]:
    x, reconstruction = validate_role(values, levels)
    require(math.isfinite(bit_weight) and bit_weight >= 0, "nonnegative bit weight")
    best: tuple | None = None
    starts = symmetric_initializations(x, reconstruction)
    for start_index, initial in enumerate(starts):
        labels = initial.copy()
        visited: set[bytes] = set()
        for iteration in range(MAX_ALTERNATIONS + 1):
            score = score_labels(x, reconstruction, labels, bit_weight, joint)
            key = (score["objective"], score["sse"], score["rate_bpw"],
                   start_index, iteration)
            if best is None or key < best[0]:
                best = (key, labels.copy(), score, start_index, iteration)
            packed = labels.tobytes(order="C")
            if iteration == MAX_ALTERNATIONS or packed in visited:
                break
            visited.add(packed)
            updated, _ = update_labels_cpu(x, reconstruction, labels, bit_weight, joint)
            if np.array_equal(labels, updated):
                break
            labels = updated
    require(best is not None, "CPU solver result")
    return {"labels": best[1], "sse": best[2]["sse"],
            "rate_bpw": best[2]["rate_bpw"], "counts": best[2]["counts"],
            "objective": best[2]["objective"], "start_index": best[3],
            "iteration": best[4], "start_count": len(starts),
            "label_sha256": label_hash(best[1])}


def solve_role_cupy(cp: Any, values: np.ndarray, levels: np.ndarray, bit_weight: float,
                    joint: bool, chunk_coordinates: int = 32768) -> dict[str, Any]:
    """CuPy assignment/count solver with canonical CPU scoring and selection."""
    x, reconstruction = validate_role(values, levels)
    require(math.isfinite(bit_weight) and bit_weight >= 0, "nonnegative bit weight")
    # Distortion is the only large resident tensor needed by all starts/updates.
    x_gpu = cp.asarray(x, dtype=cp.float64)
    levels_gpu = cp.asarray(reconstruction, dtype=cp.float64)
    distortion_gpu = (x_gpu[:, :, None] - levels_gpu) ** 2
    del x_gpu, levels_gpu
    best: tuple | None = None
    starts = symmetric_initializations(x, reconstruction)
    for start_index, initial in enumerate(starts):
        labels_gpu = cp.asarray(initial, dtype=cp.uint8)
        labels = initial.copy()
        visited: set[bytes] = set()
        for iteration in range(MAX_ALTERNATIONS + 1):
            # Canonical CPU association is an intentional semantic fence.
            score = score_labels(x, reconstruction, labels, bit_weight, joint)
            key = (score["objective"], score["sse"], score["rate_bpw"],
                   start_index, iteration)
            if best is None or key < best[0]:
                best = (key, labels.copy(), score, start_index, iteration)
            packed = labels.tobytes(order="C")
            if iteration == MAX_ALTERNATIONS or packed in visited:
                break
            visited.add(packed)
            updated_gpu, gpu_counts = update_labels_cupy_from_distortion(
                cp, distortion_gpu, labels_gpu, bit_weight, joint, chunk_coordinates)
            # Prove that counts were formed from the same current labels.
            require(np.array_equal(gpu_counts, _counts_cpu(labels, joint)),
                    "CuPy/CPU current-label count mismatch")
            updated = cp.asnumpy(updated_gpu).astype(np.uint8, copy=False)
            if np.array_equal(labels, updated):
                break
            labels_gpu, labels = updated_gpu, np.ascontiguousarray(updated)
    require(best is not None, "CuPy solver result")
    return {"labels": best[1], "sse": best[2]["sse"],
            "rate_bpw": best[2]["rate_bpw"], "counts": best[2]["counts"],
            "objective": best[2]["objective"], "start_index": best[3],
            "iteration": best[4], "start_count": len(starts),
            "label_sha256": label_hash(best[1])}


def fixed_assignment_mi(values: np.ndarray, levels: np.ndarray) -> dict[str, Any]:
    """Nearest-label MI conditioned on decoder-visible role, per UD weight."""
    x = validate_values(values)
    reconstruction = np.asarray(levels)
    require(reconstruction.dtype == np.float64 and
            reconstruction.shape == x.shape + (ALPHABET,), "MI level geometry")
    rows = []
    for role, role_name in enumerate(ROLES):
        labels = nearest_labels(x[:, role], reconstruction[:, role])
        joint = labels[0].astype(np.int16) * ALPHABET + labels[1]
        counts_joint = np.bincount(joint, minlength=ALPHABET * ALPHABET)
        counts0 = np.bincount(labels[0], minlength=ALPHABET)
        counts1 = np.bincount(labels[1], minlength=ALPHABET)
        mutual_information = (entropy_bits(counts0) + entropy_bits(counts1) -
                              entropy_bits(counts_joint))
        rows.append({"role": role_name, "coordinates": int(x.shape[2]),
                     "mutual_information_bits_per_coordinate_pair": mutual_information,
                     "joint_counts": counts_joint.tolist()})
    total_coordinates = sum(row["coordinates"] for row in rows)
    aggregate = sum(row["coordinates"] *
                    row["mutual_information_bits_per_coordinate_pair"]
                    for row in rows) / total_coordinates
    return {"conditioning": "decoder-visible role",
            "role_rows": rows,
            "mutual_information_bits_per_coordinate_pair": aggregate,
            "fixed_assignment_ceiling_bpw": aggregate / 2.0,
            "required_mutual_information_bits_per_pair":
                FIXED_ASSIGNMENT_MI_REQUIRED_BITS_PER_PAIR,
            "passes_fixed_assignment_standalone_necessary_screen":
                aggregate >= FIXED_ASSIGNMENT_MI_REQUIRED_BITS_PER_PAIR}


def _pareto_rd(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    best: dict[float, float] = {}
    for rate, distortion in points:
        require(math.isfinite(rate) and rate >= 0 and math.isfinite(distortion) and
                distortion > 0, "finite positive RD point")
        best[rate] = min(best.get(rate, math.inf), distortion)
    ordered: list[tuple[float, float]] = []
    running = math.inf
    for rate, distortion in sorted(best.items()):
        if distortion < running:
            ordered.append((rate, distortion))
            running = distortion
    hull: list[tuple[float, float]] = []
    for point in ordered:
        hull.append(point)
        while len(hull) >= 3:
            a, b, c = hull[-3:]
            slope_ab = (b[1] - a[1]) / (b[0] - a[0])
            slope_bc = (c[1] - b[1]) / (c[0] - b[0])
            if slope_bc <= slope_ab:
                hull.pop(-2)
            else:
                break
    return hull


def _interp_distortion(hull: Sequence[tuple[float, float]], rate: float) -> float:
    require(bool(hull) and hull[0][0] <= rate <= hull[-1][0], "rate interpolation")
    for left, right in zip(hull, hull[1:]):
        if left[0] <= rate <= right[0]:
            alpha = (rate - left[0]) / (right[0] - left[0])
            return left[1] + alpha * (right[1] - left[1])
    return hull[-1][1]


def _interp_rate(hull: Sequence[tuple[float, float]], distortion: float) -> float:
    require(bool(hull) and hull[-1][1] <= distortion <= hull[0][1],
            "distortion interpolation")
    for left, right in zip(hull, hull[1:]):
        if right[1] <= distortion <= left[1]:
            alpha = (left[1] - distortion) / (left[1] - right[1])
            return left[0] + alpha * (right[0] - left[0])
    return hull[-1][0]


def oracle_rd_points(values: np.ndarray, levels: np.ndarray,
                     lambdas: Sequence[Fraction], *, cp: Any | None = None,
                     chunk_coordinates: int = 32768) -> dict[str, Any]:
    """Run equal-flexibility independent/joint points under one global weight."""
    x = validate_values(values)
    reconstruction = np.asarray(levels)
    require(reconstruction.dtype == np.float64 and
            reconstruction.shape == x.shape + (ALPHABET,), "oracle levels")
    grid = tuple(lambdas)
    require(bool(grid) and all(isinstance(v, Fraction) and v >= 0 for v in grid),
            "exact nonnegative lambda grid")
    energy = float(np.sum(x * x, dtype=np.float64))
    require(energy > 0 and math.isfinite(energy), "positive source energy")
    bit_weight_divisor = x.size  # both experts, both roles, all coordinates
    solve = solve_role_cpu if cp is None else None
    independent_points: list[tuple[float, float]] = []
    joint_points: list[tuple[float, float]] = []
    rows = []
    for lagrange in grid:
        bit_weight = float(lagrange) * energy / bit_weight_divisor
        independent_sse = joint_sse = 0.0
        independent_rate = joint_rate = 0.0
        role_rows = []
        for role, role_name in enumerate(ROLES):
            if cp is None:
                ind = solve(x[:, role], reconstruction[:, role], bit_weight, False)
                joined = solve(x[:, role], reconstruction[:, role], bit_weight, True)
            else:
                ind = solve_role_cupy(cp, x[:, role], reconstruction[:, role], bit_weight,
                                      False, chunk_coordinates)
                joined = solve_role_cupy(cp, x[:, role], reconstruction[:, role], bit_weight,
                                         True, chunk_coordinates)
            independent_sse += ind["sse"]
            joint_sse += joined["sse"]
            independent_rate += ind["rate_bpw"]
            joint_rate += joined["rate_bpw"]
            role_rows.append({"role": role_name,
                              "source_energy": float(np.sum(x[:, role] ** 2,
                                                            dtype=np.float64)),
                              "bit_weight": bit_weight,
                              "independent": {k: v for k, v in ind.items()
                                              if k not in ("labels", "counts")},
                              "joint": {k: v for k, v in joined.items()
                                        if k not in ("labels", "counts")}})
        ind_point = (independent_rate / len(ROLES), independent_sse / energy)
        joint_point = (joint_rate / len(ROLES), joint_sse / energy)
        independent_points.append(ind_point)
        joint_points.append(joint_point)
        rows.append({"lambda": str(lagrange), "bit_weight": bit_weight,
                     "independent": {"rate_bpw": ind_point[0],
                                     "relative_distortion": ind_point[1]},
                     "joint": {"rate_bpw": joint_point[0],
                               "relative_distortion": joint_point[1]},
                     "roles": role_rows})
    return {"source_energy": energy, "source_weight_count": int(x.size),
            "global_bit_weight_formula": "float(lambda)*sum(UD^2)/(4*N)",
            "rows": rows, "independent_points": independent_points,
            "joint_points": joint_points,
            "fixed_assignment_mi": fixed_assignment_mi(x, reconstruction)}


def convexified_gate(rd: dict[str, Any]) -> dict[str, Any]:
    """PAIRPATH equal-rate/equal-MSE comparison with free time sharing."""
    independent_hull = _pareto_rd(rd["independent_points"])
    joint_hull = _pareto_rd(rd["joint_points"])
    lo = max(independent_hull[0][0], joint_hull[0][0])
    hi = min(independent_hull[-1][0], joint_hull[-1][0])
    equal_rate = []
    if lo <= hi:
        rates = sorted({lo, hi} |
                       {rate for rate, _ in independent_hull if lo <= rate <= hi} |
                       {rate for rate, _ in joint_hull if lo <= rate <= hi})
        for rate in rates:
            distortion_ind = _interp_distortion(independent_hull, rate)
            distortion_joint = _interp_distortion(joint_hull, rate)
            equal_rate.append({"rate_bpw": rate, "D_ind": distortion_ind,
                               "D_pair": distortion_joint,
                               "G_eq_bpw": 0.5 * math.log2(
                                   distortion_ind / distortion_joint)})
    distortion_lo = max(independent_hull[-1][1], joint_hull[-1][1])
    distortion_hi = min(independent_hull[0][1], joint_hull[0][1])
    equal_mse = []
    if distortion_lo <= distortion_hi:
        distortions = sorted({distortion_lo, distortion_hi} |
                             {d for _, d in independent_hull
                              if distortion_lo <= d <= distortion_hi} |
                             {d for _, d in joint_hull
                              if distortion_lo <= d <= distortion_hi}, reverse=True)
        for distortion in distortions:
            rate_ind = _interp_rate(independent_hull, distortion)
            rate_joint = _interp_rate(joint_hull, distortion)
            equal_mse.append({"relative_D": distortion, "R_ind_bpw": rate_ind,
                              "R_pair_bpw": rate_joint,
                              "G_eq_bpw": rate_ind - rate_joint})
    gains = [row["G_eq_bpw"] for row in equal_rate + equal_mse]
    best_gain = max(gains) if gains else -math.inf
    if best_gain < ORACLE_EARLY_KILL_BPW:
        status = "HARD_KILL_OPTIMISTIC_JOINT_GATE_BELOW_0P045"
    elif best_gain >= ORACLE_ENGINEERING_MARGIN_BPW:
        status = "SURVIVE_OPTIMISTIC_GATE_WITH_PHYSICAL_MARGIN"
    elif best_gain >= REQUIRED_UPDOWN_GAIN_BPW:
        status = "SURVIVE_OPTIMISTIC_GATE_STANDALONE_THRESHOLD"
    else:
        status = "INTERESTING_BUT_INSUFFICIENT_STANDALONE"
    return {"status": status, "claim": "kill-only; tables and time sharing are free",
            "independent_hull": independent_hull, "pair_hull": joint_hull,
            "equal_rate": equal_rate, "equal_mse": equal_mse,
            "best_G_eq_UD_bpw": best_gain,
            "early_kill_bpw": ORACLE_EARLY_KILL_BPW,
            "standalone_required_bpw": REQUIRED_UPDOWN_GAIN_BPW,
            "physical_engineering_margin_bpw": ORACLE_ENGINEERING_MARGIN_BPW}


def theoretical_memory_bytes(coordinates: int, chunk_coordinates: int = 32768) -> dict[str, int]:
    """Conservative explicit-array memory ledger for one role solver."""
    require(coordinates > 0 and chunk_coordinates > 0, "memory geometry")
    chunk = min(coordinates, chunk_coordinates)
    resident_distortion = 2 * coordinates * ALPHABET * 8
    resident_labels = 2 * coordinates * 1 * 2  # current and next
    joint_cost_chunk = chunk * ALPHABET * ALPHABET * 8
    independent_cost_chunk = chunk * ALPHABET * 8
    small_models = (ALPHABET * ALPHABET * 8 * 3) + 4096
    return {"coordinates_per_role": coordinates,
            "chunk_coordinates": chunk,
            "resident_distortion_bytes": resident_distortion,
            "resident_current_plus_next_label_bytes": resident_labels,
            "joint_cost_chunk_bytes": joint_cost_chunk,
            "independent_cost_chunk_bytes": independent_cost_chunk,
            "conservative_joint_peak_explicit_bytes":
                resident_distortion + resident_labels + joint_cost_chunk + small_models,
            "note_excludes_cupy_allocator_workspace_bytes": 0}
