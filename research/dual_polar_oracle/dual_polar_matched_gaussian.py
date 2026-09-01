#!/usr/bin/env python3
"""Frozen matched-Gaussian and Marchenko--Pastur red-team for dual polar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
ROLES = ("gate", "up", "down")
STACK_ROWS = ROWS * len(ROLES)
VALUES_PER_EXPERT = STACK_ROWS * COLS
EXPERTS = 6
PANEL_VALUES = EXPERTS * VALUES_PER_EXPERT
SYMMETRIC_DOF = COLS * (COLS + 1) // 2
STIEFEL_DOF = VALUES_PER_EXPERT - SYMMETRIC_DOF
RATES = (2.15, 2.30, 2.50)
GLOBAL_HEADER_BITS = 4096
EXPERT_HEADER_BITS = 128
RANK_BITS = 12
REQUIRED_S = -0.5 * math.log2(0.8)
SE_MULTIPLIER = 3.0
BASE_SEED = 26090131
IDENTITY_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def deterministic_seed(replica: int, layer: int, expert: int, role: str) -> int:
    text = f"dual-polar-control-v1:{BASE_SEED}:{replica}:{layer}:{expert}:{role}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def load_bf16_numpy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != shape[0] * shape[1]:
        raise RuntimeError(f"source size mismatch: {path}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite source: {path}")
    return values


def source_panel(lock_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    rows = lock.get("matrices")
    if not isinstance(rows, list) or len(rows) != 18:
        raise RuntimeError("expected pinned 18-matrix source lock")
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    receipts = []
    for row in rows:
        role = str(row["role"])
        if role not in ROLES:
            raise RuntimeError(role)
        path = lock_path.parent / row["output_relpath"]
        observed = sha256_file(path)
        if observed != row["source_bf16_sha256"]:
            raise RuntimeError(f"source hash mismatch: {path}")
        match = IDENTITY_RE.search(str(row["tensor"]))
        if match is None:
            raise RuntimeError(row["tensor"])
        key = (int(match.group(1)), int(match.group(2)))
        grouped.setdefault(key, {})[role] = {**row, "path": path}
        receipts.append({"tensor": row["tensor"], "role": role, "sha256": observed, "bytes": path.stat().st_size})
    if len(grouped) != EXPERTS or any(set(value) != set(ROLES) for value in grouped.values()):
        raise RuntimeError("not six complete experts")
    experts = [{"layer": key[0], "expert": key[1], "roles": grouped[key]} for key in sorted(grouped)]
    return experts, receipts


def role_moments(experts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float]]:
    output = []
    expert_energies = []
    for expert in experts:
        roles = []
        total = 0.0
        for role in ROLES:
            row = expert["roles"][role]
            shape = tuple(int(value) for value in row["shape"])
            matrix = load_bf16_numpy(row["path"], shape)
            values = matrix.astype(np.float64)
            mean = float(np.mean(values))
            centered_energy = float(np.sum((values - mean) ** 2, dtype=np.float64))
            energy = float(np.sum(values * values, dtype=np.float64))
            total += energy
            roles.append(
                {
                    "role": role,
                    "shape": list(shape),
                    "mean": mean,
                    "centered_energy": centered_energy,
                    "energy": energy,
                    "source_sha256": sha256_file(row["path"]),
                }
            )
        output.append({"layer": expert["layer"], "expert": expert["expert"], "roles": roles, "energy": total})
        expert_energies.append(total)
    return output, expert_energies


def moment_matched_gaussian(shape: tuple[int, int], mean: float, centered_energy: float, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(shape, dtype=np.float32).astype(np.float64)
    z -= float(np.mean(z))
    z *= math.sqrt(centered_energy / float(np.sum(z * z, dtype=np.float64)))
    output = (z + mean).astype(np.float32)
    check = output.astype(np.float64)
    achieved_mean = float(np.mean(check))
    achieved_centered = float(np.sum((check - achieved_mean) ** 2, dtype=np.float64))
    return output, {
        "requested_mean": mean,
        "achieved_mean": achieved_mean,
        "absolute_mean_error": abs(achieved_mean - mean),
        "requested_centered_energy": centered_energy,
        "achieved_centered_energy": achieved_centered,
        "relative_centered_energy_error": abs(achieved_centered - centered_energy) / centered_energy,
    }


def rank_curve(singular_values: np.ndarray, *, cols: int = COLS, values: int = VALUES_PER_EXPERT, stiefel: int = STIEFEL_DOF) -> list[dict[str, Any]]:
    singular = np.sort(np.asarray(singular_values, dtype=np.float64))
    if singular.shape != (cols,):
        raise RuntimeError(singular.shape)
    prefix = np.concatenate(([0.0], np.cumsum(singular)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(singular * singular)))
    curve = []
    for rank in range(cols - 1):
        unmodelled = cols - rank
        sums = prefix[unmodelled:] - prefix[:-unmodelled]
        sums_sq = prefix_sq[unmodelled:] - prefix_sq[:-unmodelled]
        errors = np.maximum(0.0, sums_sq - sums * sums / unmodelled)
        start = int(np.argmin(errors))
        rank_dof = cols * rank - rank * (rank - 1) // 2
        model_dof = stiefel + 1 + rank_dof
        if model_dof >= values:
            break
        curve.append(
            {
                "rank": rank,
                "model_dof": int(model_dof),
                "normal_dof": int(values - model_dof),
                "window_start": start,
                "window_stop": start + unmodelled,
                "common_scale": float(sums[start] / unmodelled),
                "residual_energy": float(errors[start]),
            }
        )
    return curve


def reverse_waterfill(records: list[dict[str, Any]], ranks: list[int], physical_rate: float) -> dict[str, Any]:
    side_bits = GLOBAL_HEADER_BITS + EXPERTS * (EXPERT_HEADER_BITS + RANK_BITS)
    payload_bits = physical_rate * PANEL_VALUES - side_bits
    total_energy = sum(float(record["energy"]) for record in records)
    energies = []
    dimensions = []
    for record, rank in zip(records, ranks, strict=True):
        cell = record["curve"][rank]
        residual = min(float(cell["residual_energy"]), float(record["energy"]) * (1.0 - 1e-15))
        energies.extend((float(record["energy"]) - residual, residual))
        dimensions.extend((int(cell["model_dof"]), int(cell["normal_dof"])))
    if sum(dimensions) != PANEL_VALUES:
        raise AssertionError("dimension closure")
    variances = [energy / dimension for energy, dimension in zip(energies, dimensions, strict=True)]
    lo = min(variances) * 2.0**-80
    hi = max(variances)
    for _ in range(150):
        theta = math.sqrt(lo * hi)
        used = 0.5 * sum(dimension * max(math.log2(variance / theta), 0.0) for variance, dimension in zip(variances, dimensions, strict=True))
        if used > payload_bits:
            lo = theta
        else:
            hi = theta
    distortion = sum(dimension * min(variance, hi) for variance, dimension in zip(variances, dimensions, strict=True)) / total_energy
    f_value = distortion * 2.0 ** (2.0 * physical_rate)
    return {
        "physical_rate_bpw": physical_rate,
        "payload_rate_bpw": payload_bits / PANEL_VALUES,
        "side_bits": side_bits,
        "relative_mse": distortion,
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "water_level": hi,
        "ranks": list(ranks),
    }


def optimize(records: list[dict[str, Any]], rate: float) -> dict[str, Any]:
    rank_count = min(len(record["curve"]) for record in records)
    common = [reverse_waterfill(records, [rank] * EXPERTS, rate) for rank in range(rank_count)]
    best_common = min(common, key=lambda row: (row["F"], row["ranks"]))
    ranks = list(best_common["ranks"])
    best = reverse_waterfill(records, ranks, rate)
    for _ in range(8):
        changed = False
        for expert in range(EXPERTS):
            choices = []
            for rank in range(len(records[expert]["curve"])):
                trial = list(ranks)
                trial[expert] = rank
                choices.append(reverse_waterfill(records, trial, rate))
            winner = min(choices, key=lambda row: (row["F"], row["ranks"]))
            if winner["ranks"][expert] != ranks[expert]:
                ranks = list(winner["ranks"])
                best = winner
                changed = True
        if not changed:
            break
    best["common_rank_result"] = best_common
    best["selected"] = [
        {
            "layer": record["layer"],
            "expert": record["expert"],
            **record["curve"][rank],
            "source_energy": record["energy"],
            "residual_energy_ratio": record["curve"][rank]["residual_energy"] / record["energy"],
        }
        for record, rank in zip(records, ranks, strict=True)
    ]
    return best


def array_backend(name: str):
    if name == "numpy":
        return np
    if name != "cupy":
        raise ValueError(name)
    import cupy as cp

    return cp


def to_numpy(value: Any, backend: str) -> np.ndarray:
    if backend == "numpy":
        return np.asarray(value)
    import cupy as cp

    return cp.asnumpy(value)


def free_backend(backend: str) -> None:
    if backend == "cupy":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()


def finite_controls(
    experts: list[dict[str, Any]],
    moments: list[dict[str, Any]],
    *,
    replicas: int,
    backend: str,
) -> list[dict[str, Any]]:
    xp = array_backend(backend)
    output = []
    for replica in range(replicas):
        records = []
        serialized = []
        for expert, expert_moments in zip(experts, moments, strict=True):
            stack = xp.empty((STACK_ROWS, COLS), dtype=xp.float32)
            control_moments = []
            for ordinal, (role, source_moments) in enumerate(zip(ROLES, expert_moments["roles"], strict=True)):
                shape = tuple(int(value) for value in source_moments["shape"])
                matrix, check = moment_matched_gaussian(
                    shape,
                    float(source_moments["mean"]),
                    float(source_moments["centered_energy"]),
                    deterministic_seed(replica, expert["layer"], expert["expert"], role),
                )
                canonical = matrix.T if role == "down" else matrix
                if canonical.shape != (ROWS, COLS):
                    raise RuntimeError(canonical.shape)
                stack[ordinal * ROWS : (ordinal + 1) * ROWS] = xp.asarray(canonical)
                check["role"] = role
                control_moments.append(check)
            energy = float(to_numpy(xp.sum(stack.astype(xp.float64) ** 2), backend))
            singular = xp.linalg.svd(stack, compute_uv=False)
            spectrum = to_numpy(singular, backend).astype(np.float64)
            spectral_energy = float(np.sum(spectrum * spectrum, dtype=np.float64))
            closure = abs(spectral_energy - energy) / energy
            if closure > 2e-5:
                raise RuntimeError(f"SVD closure {closure}")
            curve = rank_curve(spectrum)
            records.append({"layer": expert["layer"], "expert": expert["expert"], "energy": energy, "curve": curve})
            serialized.append(
                {
                    "layer": expert["layer"],
                    "expert": expert["expert"],
                    "energy": energy,
                    "spectral_energy": spectral_energy,
                    "relative_svd_energy_error": closure,
                    "control_moments": control_moments,
                    "singular_values_ascending": np.sort(spectrum).tolist(),
                }
            )
            del stack, singular
            free_backend(backend)
            print(f"control replica={replica} L{expert['layer']} E{expert['expert']} closure={closure:.3e}", flush=True)
        scores = [optimize(records, rate) for rate in RATES]
        output.append({"replica": replica, "records": serialized, "scores": scores, "best": min(scores, key=lambda row: (row["F"], row["physical_rate_bpw"]))})
    return output


def marchenko_pastur_control(expert_energies: list[float], grid_points: int) -> dict[str, Any]:
    aspect = COLS / STACK_ROWS
    root = math.sqrt(aspect)
    lower = (1.0 - root) ** 2
    upper = (1.0 + root) ** 2
    grid = np.linspace(lower, upper, grid_points, dtype=np.float64)
    density = np.sqrt(np.maximum(0.0, (upper - grid) * (grid - lower))) / (2.0 * math.pi * aspect * np.maximum(grid, 1e-300))
    step = grid[1] - grid[0]
    cdf = np.concatenate(([0.0], np.cumsum((density[:-1] + density[1:]) * (0.5 * step))))
    cdf /= cdf[-1]
    probabilities = (np.arange(COLS, dtype=np.float64) + 0.5) / COLS
    eigenvalues = np.interp(probabilities, cdf, grid)
    base = np.sqrt(eigenvalues)
    records = []
    spectra = []
    for index, energy in enumerate(expert_energies):
        spectrum = base * math.sqrt(energy / float(np.sum(base * base)))
        spectra.append(spectrum.tolist())
        records.append({"layer": index, "expert": index, "energy": energy, "curve": rank_curve(spectrum)})
    scores = [optimize(records, rate) for rate in RATES]
    return {
        "aspect": aspect,
        "eigenvalue_support": [lower, upper],
        "singular_support_normalized": [math.sqrt(lower), math.sqrt(upper)],
        "grid_points": grid_points,
        "midpoint_quantile_spectra": spectra,
        "scores": scores,
        "best": min(scores, key=lambda row: (row["F"], row["physical_rate_bpw"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--composite-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--mp-grid-points", type=int, default=1_000_001)
    args = parser.parse_args()
    if args.replicas != 3:
        raise RuntimeError("frozen protocol requires exactly three replicas")
    started = time.time()

    source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
    source_score = source_result["best"]
    if source_result["schema"] != "qwen_pinned_dual_polar_stage1_v1":
        raise RuntimeError(source_result["schema"])
    experts, receipts = source_panel(args.source_lock.resolve())
    moments, expert_energies = role_moments(experts)
    for observed, expected in zip(expert_energies, [float(row["energy"]) for row in source_result["records"]], strict=True):
        if not math.isclose(observed, expected, rel_tol=3e-8, abs_tol=1e-8):
            raise RuntimeError((observed, expected))

    controls = finite_controls(experts, moments, replicas=args.replicas, backend=args.backend)
    mp = marchenko_pastur_control(expert_energies, args.mp_grid_points)
    source_s = float(source_score["s_bpw"])
    control_s = np.asarray([float(replica["scores"][0]["s_bpw"]) for replica in controls], dtype=np.float64)
    control_mean = float(np.mean(control_s))
    control_std = float(np.std(control_s, ddof=1))
    control_se = control_std / math.sqrt(len(control_s))
    control_lower = control_mean - SE_MULTIPLIER * control_se
    mp_s = float(mp["scores"][0]["s_bpw"])
    control_floor = min(mp_s, control_lower)
    excess_mean = source_s - control_mean
    excess_upper = max(0.0, source_s - control_floor)

    composite = json.loads(args.composite_result.read_text(encoding="utf-8"))
    horizontal = composite["variants"]["role_gauge+polar"]["rates"]["2.50"]
    horizontal_s = float(horizontal["s_bpw"])
    raw_additive = source_s + horizontal_s
    union_upper = horizontal_s + excess_upper
    verdict = "PROMOTE_INTRINSIC_JOINT_NESTING" if union_upper >= REQUIRED_S else "HARD_KILL_DUAL_POLAR_AND_NAIVE_NESTING"

    result = {
        "schema": "qwen_dual_polar_matched_gaussian_redteam_v1",
        "decision": verdict,
        "claim_boundary": "matched-Gaussian and Marchenko--Pastur diagnostic of the ideal dual-polar oracle; not an emitted codec or a universal nonlinear RD converse",
        "protocol": {
            "replicas": args.replicas,
            "base_seed": BASE_SEED,
            "backend": args.backend,
            "moment_match": "each of 18 roles independently: exact source mean and centered energy before final FP32 rounding",
            "same_search_as_source": "all ranks, all contiguous unmodelled spectrum windows, adaptive six-expert coordinate descent, identical side bits and reverse waterfill",
            "confidence_se_multiplier": SE_MULTIPLIER,
        },
        "binding": {
            "source_lock_sha256": sha256_file(args.source_lock),
            "source_result_sha256": sha256_file(args.source_result),
            "source_script_sha256": sha256_file(args.source_script),
            "protocol_sha256": sha256_file(args.protocol),
            "composite_result_sha256": sha256_file(args.composite_result),
            "executing_script_sha256": sha256_file(Path(__file__)),
            "source_receipts": receipts,
        },
        "source": {
            "F": float(source_score["F"]),
            "s_bpw": source_s,
            "relative_mse": float(source_score["relative_mse"]),
            "rate_bpw": float(source_score["physical_rate_bpw"]),
            "ranks": source_score["ranks"],
            "moments": moments,
        },
        "matched_gaussian": {
            "replicas": controls,
            "s_at_2p15": control_s.tolist(),
            "mean_s_bpw": control_mean,
            "sample_std_s_bpw": control_std,
            "standard_error_s_bpw": control_se,
            "three_se_lower_s_bpw": control_lower,
        },
        "marchenko_pastur": mp,
        "diagnostic": {
            "generic_fraction_of_source_s": control_mean / source_s,
            "source_minus_control_mean_s_bpw": excess_mean,
            "favourable_control_floor_s_bpw": control_floor,
            "source_specific_excess_upper_s_bpw": excess_upper,
            "interpretation": "F<1 for the iid Marchenko--Pastur null is impossible as an actual improvement over the exact iid Gaussian RD function; it identifies missing nonlinear coordinate metric/Jacobian cost in the polar component waterfill",
        },
        "nesting": {
            "required_s_bpw": REQUIRED_S,
            "dual_raw_s_bpw": source_s,
            "dual_raw_shortfall_bpw": REQUIRED_S - source_s,
            "role_horizontal_polar_s_bpw": horizontal_s,
            "raw_additive_s_bpw_invalid": raw_additive,
            "raw_additive_shortfall_bpw": REQUIRED_S - raw_additive,
            "source_specific_dual_excess_upper_s_bpw": excess_upper,
            "favourable_zero_overlap_union_upper_s_bpw": union_upper,
            "favourable_zero_overlap_union_shortfall_bpw": REQUIRED_S - union_upper,
            "non_double_counting_rule": "a valid follow-up must decompose actual joint components in an intrinsic two-sided quotient, charge the induced metric/Jacobian and shared coordinates once, run one waterfill, and pass the identical Gaussian null; scalar s addition is forbidden",
        },
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "backend": args.backend,
        },
    }
    result["result_content_sha256"] = canonical_hash(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": verdict,
                "source_s": source_s,
                "control_s": control_s.tolist(),
                "control_mean_s": control_mean,
                "mp_s": mp_s,
                "excess_upper_s": excess_upper,
                "union_upper_s": union_upper,
                "required_s": REQUIRED_S,
                "seconds": result["runtime"]["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
