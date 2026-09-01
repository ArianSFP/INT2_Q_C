#!/usr/bin/env python3
"""Source/arithmetic verifier for the polar-normal predictor oracle.

This verifier does not import the experiment.  It rehashes all source, router,
and parent artifacts; rebuilds every source and matched-Gaussian polar normal;
checks normal identities; independently recomputes the decisive identity-band,
router, and tiny radial predictor projections; and repeats every serialized
candidate's side ledger and reverse-waterfill arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
ROLES = 3
EXPERTS = 6
MATRICES = 18
VALUES = ROWS * COLS
PANEL_VALUES = MATRICES * VALUES
RATE = 2.5
TARGET_F = 0.8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def close(a: float, b: float, *, rtol: float = 8e-9, atol: float = 8e-9) -> None:
    if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
        raise AssertionError((a, b))


def read_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size != math.prod(shape):
        raise AssertionError((path, words.size, shape))
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)


def best_window(singular: np.ndarray, rank: int) -> tuple[int, int, float, float]:
    width = len(singular) - rank
    prefix = np.concatenate(([0.0], np.cumsum(singular, dtype=np.float64)))
    prefix2 = np.concatenate(([0.0], np.cumsum(np.square(singular), dtype=np.float64)))
    sums = prefix[width:] - prefix[:-width]
    sums2 = prefix2[width:] - prefix2[:-width]
    errors = np.maximum(0.0, sums2 - sums * sums / width)
    start = int(np.argmin(errors))
    return start, start + width, float(sums[start] / width), float(errors[start])


def polar_normal(matrix: np.ndarray, rank: int) -> dict[str, Any]:
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    u = np.ascontiguousarray(u[:, ::-1])
    singular = np.ascontiguousarray(singular[::-1], dtype=np.float64)
    vt = np.ascontiguousarray(vt[::-1])
    start, stop, common, residual = best_window(singular, rank)
    delta = np.zeros_like(singular)
    delta[start:stop] = singular[start:stop] - common
    normal = (u * delta[None, :]) @ u.T
    error = (u * delta[None, :]) @ vt
    source_energy = float(np.sum(np.square(matrix), dtype=np.float64))
    normal_energy = float(np.sum(np.square(normal), dtype=np.float64))
    close(normal_energy, residual, rtol=4e-9, atol=4e-8)
    close(np.sum(np.square(error)), residual, rtol=4e-9, atol=4e-8)
    stiefel = ROWS * COLS - ROWS * (ROWS + 1) // 2
    model_dof = stiefel + 1 + ROWS * rank - rank * (rank - 1) // 2
    return {
        "source_energy": source_energy,
        "model_energy": source_energy - residual,
        "normal_energy": residual,
        "model_dof": model_dof,
        "normal_dof": VALUES - model_dof,
        "window_start": start,
        "window_stop": stop,
        "common_scale": common,
        "normal": normal,
        "error": error,
        "normal_sha256_f64": hashlib.sha256(
            np.ascontiguousarray(normal, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def waterfill(d: np.ndarray, e: np.ndarray, rate: float) -> dict[str, Any]:
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    keep = (d > 0.0) & (e > 0.0)
    d = d[keep]
    e = e[keep]
    logv = np.log2(e / d)
    order = np.argsort(logv)[::-1]
    lv = logv[order]
    ds = d[order]
    cd = np.cumsum(ds)
    cdlv = np.cumsum(ds * lv)
    levels = (cdlv - 2.0 * rate) / cd
    active_count = len(d)
    for k in range(1, len(d) + 1):
        level = levels[k - 1]
        if level <= lv[k - 1] + 2e-14 and (k == len(d) or level >= lv[k] - 2e-14):
            active_count = k
            break
    log_level = float(levels[active_count - 1])
    level = 2.0**log_level
    active = logv > log_level
    used = float(np.sum(d * 0.5 * np.maximum(0.0, logv - log_level)))
    close(used, rate, rtol=3e-10, atol=3e-10)
    distortion = float(np.sum(np.where(active, d * level, e), dtype=np.float64))
    return {
        "distortion": distortion,
        "active_components": int(np.count_nonzero(active)),
        "dimension_sum": float(np.sum(d)),
        "energy_sum": float(np.sum(e)),
    }


def candidate_score(
    records: list[dict[str, Any]],
    residuals: list[float],
    coefficient_counts: list[int],
    side_bpw: float,
) -> dict[str, Any]:
    total_energy = float(sum(row["source_energy"] for row in records))
    d: list[float] = []
    e: list[float] = []
    for row, residual, count in zip(records, residuals, coefficient_counts, strict=True):
        d.append(row["model_dof"] / PANEL_VALUES)
        e.append(row["model_energy"] / total_energy)
        if residual > 0.0:
            d.append(max(1, row["normal_dof"] - count) / PANEL_VALUES)
            e.append(residual / total_energy)
    wf = waterfill(np.asarray(d), np.asarray(e), RATE - side_bpw)
    f_value = wf["distortion"] * 2.0 ** (2.0 * RATE)
    return {
        "F": f_value,
        "ideal_relative_mse": wf["distortion"],
        "side_bpw": side_bpw,
        "payload_rate_bpw": RATE - side_bpw,
        "passes_F_le_0p8": f_value <= TARGET_F,
        **wf,
    }


def radial_template(training: list[np.ndarray]) -> np.ndarray:
    values = np.zeros(ROWS, dtype=np.float64)
    counts = np.zeros(ROWS, dtype=np.float64)
    for matrix in training:
        for offset in range(ROWS):
            diagonal = np.diagonal(matrix, offset=offset)
            multiplier = 1.0 if offset == 0 else 2.0
            values[offset] += multiplier * float(np.sum(diagonal, dtype=np.float64))
            counts[offset] += multiplier * len(diagonal)
    values /= counts
    index = np.abs(np.arange(ROWS)[:, None] - np.arange(ROWS)[None, :])
    return values[index]


def scalar_projection_residual(target: np.ndarray, template: np.ndarray) -> float:
    denominator = float(np.sum(np.square(template), dtype=np.float64))
    energy = float(np.sum(np.square(target), dtype=np.float64))
    if denominator <= 0.0:
        return energy
    numerator = float(np.sum(target * template, dtype=np.float64))
    return max(0.0, energy - numerator * numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    result_bytes = args.result.read_bytes()
    result = json.loads(result_bytes)
    claimed = result.get("result_lock_sha256")
    clean = dict(result)
    clean.pop("result_lock_sha256", None)
    if hashlib.sha256(canonical(clean)).hexdigest() != claimed:
        raise AssertionError("result internal seal")
    if result["schema"] != "qwen_polar_normal_predictor_oracle_v1":
        raise AssertionError("schema")
    if sha256_file(args.base_result) != result["audit"]["base_composite_result_sha256"]:
        raise AssertionError("base result hash")
    base = json.loads(args.base_result.read_text(encoding="utf-8"))
    base_clean = dict(base)
    base_claimed = base_clean.pop("result_lock_sha256")
    if hashlib.sha256(canonical(base_clean)).hexdigest() != base_claimed:
        raise AssertionError("base result seal")
    selections = base["variants"]["polar"]["rates"]["2.50"]["selections"]

    lock_path = args.root.resolve() / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    if sha256_file(lock_path) != result["audit"]["source_lock_file_sha256"]:
        raise AssertionError("source lock hash")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matrices: list[np.ndarray] = []
    for ordinal, row in enumerate(lock["matrices"]):
        path = lock_path.parent / row["output_relpath"]
        if sha256_file(path) != row["source_bf16_sha256"]:
            raise AssertionError((ordinal, "source hash"))
        matrix = read_bf16(path, tuple(int(x) for x in row["shape"]))
        if row["role"] == "down":
            matrix = matrix.T
        matrices.append(np.ascontiguousarray(matrix, dtype=np.float64))

    source_records: list[dict[str, Any]] = []
    for ordinal, (matrix, selected, recorded) in enumerate(
        zip(matrices, selections, result["source_normal_records"], strict=True)
    ):
        rebuilt = polar_normal(matrix, int(selected["rank"]))
        for key in (
            "source_energy",
            "model_energy",
            "normal_energy",
            "common_scale",
        ):
            close(rebuilt[key], recorded[key], rtol=5e-9, atol=5e-8)
        for key in ("model_dof", "normal_dof", "window_start", "window_stop"):
            if int(rebuilt[key]) != int(recorded[key]):
                raise AssertionError((ordinal, key))
        if rebuilt["normal_sha256_f64"] != recorded["normal_sha256_f64"]:
            # Scalar/geometry checks above remain authoritative across LAPACK builds.
            rebuilt["hash_portable_match"] = False
        else:
            rebuilt["hash_portable_match"] = True
        source_records.append(rebuilt)
    print("rebuilt 18 source polar normals", flush=True)

    rng = np.random.default_rng(int(result["audit"]["gaussian_seed"]))
    controls: list[dict[str, Any]] = []
    for ordinal, (source, selected, recorded) in enumerate(
        zip(matrices, selections, result["matched_gaussian_normal_records"], strict=True)
    ):
        mean = float(np.mean(source, dtype=np.float64))
        centered_energy = float(np.sum(np.square(source - mean), dtype=np.float64))
        gaussian = rng.standard_normal(source.shape, dtype=np.float64)
        gaussian -= float(np.mean(gaussian, dtype=np.float64))
        gaussian *= math.sqrt(centered_energy / float(np.sum(np.square(gaussian))))
        gaussian += mean
        rebuilt = polar_normal(gaussian, int(selected["rank"]))
        for key in ("source_energy", "model_energy", "normal_energy", "common_scale"):
            close(rebuilt[key], recorded[key], rtol=5e-9, atol=5e-8)
        controls.append(rebuilt)
    print("rebuilt 18 matched-Gaussian polar normals", flush=True)

    candidates = {row["name"]: row for row in result["candidates"]}
    base_side = float(result["scope"]["base_polar_explicit_side_bpw"])
    for row in candidates.values():
        residuals = [float(x) for x in row["residual_normal_energy_per_matrix"]]
        counts = [int(x) for x in row["private_coefficient_counts_per_matrix"]]
        if sum(counts) != int(row["private_coefficient_count"]):
            raise AssertionError((row["name"], "coefficient sum"))
        one_bpw = sum(counts) / PANEL_VALUES
        fp16_bpw = 16.0 * one_bpw
        close(one_bpw, row["private_field_bpw_one_bit_lower_bound"], atol=2e-15)
        close(fp16_bpw, row["private_field_bpw_fp16"], atol=2e-15)
        captured = 1.0 - sum(residuals) / sum(x["normal_energy"] for x in source_records)
        close(captured, row["captured_normal_energy_fraction"])
        shared = float(row["shared_table_full_model_amortized_bpw"])
        checks = (
            ("free_exact_coefficient_score", base_side),
            ("one_bit_coefficient_lower_bound_score", base_side + one_bpw + shared),
            ("fp16_exact_coefficient_optimistic_score", base_side + fp16_bpw + shared),
        )
        for key, side in checks:
            rebuilt = candidate_score(source_records, residuals, counts, side)
            close(rebuilt["F"], row[key]["F"])
            close(rebuilt["ideal_relative_mse"], row[key]["ideal_relative_mse"])
    print(f"verified arithmetic for {len(candidates)} candidates", flush=True)

    # Decisive analytic identity bands.
    for bandwidth in (12, 128):
        row = candidates[f"identity_band_{bandwidth}"]
        mask_row, mask_col = np.indices((ROWS, ROWS))
        mask = np.abs(mask_row - mask_col) <= bandwidth
        residuals = [
            max(
                0.0,
                record["normal_energy"]
                - float(np.sum(np.square(record["normal"][mask]), dtype=np.float64)),
            )
            for record in source_records
        ]
        if not np.allclose(residuals, row["residual_normal_energy_per_matrix"], rtol=6e-9, atol=6e-8):
            raise AssertionError((row["name"], "projection"))

    # Decoder-visible router ranks 8 and 128 and routed row.
    router_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    router_receipts = {int(row["layer"]): row for row in result["audit"]["router_receipts"]}
    for record_meta in result["source_normal_records"]:
        layer = int(record_meta["layer"])
        if layer in router_cache:
            continue
        path = args.router_dir / router_receipts[layer]["relative_path"]
        if sha256_file(path) != router_receipts[layer]["sha256"]:
            raise AssertionError((layer, "router hash"))
        router = np.ascontiguousarray(read_bf16(path, (128, COLS)), dtype=np.float64)
        _, _, vt = np.linalg.svd(router, full_matrices=False)
        router_cache[layer] = (router, vt.T)
    for rank in (8, 128):
        residuals = []
        for meta, record in zip(result["source_normal_records"], source_records, strict=True):
            basis = router_cache[int(meta["layer"])][1][:, :rank]
            captured = float(np.sum(np.square(record["error"] @ basis), dtype=np.float64))
            residuals.append(max(0.0, record["normal_energy"] - captured))
        expected = candidates[f"router_pca_right_rank_{rank}"][
            "residual_normal_energy_per_matrix"
        ]
        if not np.allclose(residuals, expected, rtol=8e-9, atol=8e-8):
            raise AssertionError((rank, "router projection"))
    routed_residuals = []
    for meta, record in zip(result["source_normal_records"], source_records, strict=True):
        router = router_cache[int(meta["layer"])][0]
        direction = router[int(meta["expert"])]
        direction /= np.linalg.norm(direction)
        captured = float(np.sum(np.square(record["error"] @ direction), dtype=np.float64))
        routed_residuals.append(max(0.0, record["normal_energy"] - captured))
    if not np.allclose(
        routed_residuals,
        candidates["routed_expert_row_right_rank_1"]["residual_normal_energy_per_matrix"],
        rtol=8e-9,
        atol=8e-8,
    ):
        raise AssertionError("routed row projection")

    # Tiny held-out coordinate-conditioned implicit winner under the FP16 ledger.
    radial_residuals = []
    for target, record in enumerate(source_records):
        training = [
            source_records[i]["normal"]
            for i in range(MATRICES)
            if i // ROLES != target // ROLES and i % ROLES == target % ROLES
        ]
        template = radial_template(training)
        radial_residuals.append(scalar_projection_residual(record["normal"], template))
    if not np.allclose(
        radial_residuals,
        candidates["heldout_role_radial_implicit"]["residual_normal_energy_per_matrix"],
        rtol=8e-9,
        atol=8e-8,
    ):
        raise AssertionError("radial implicit projection")

    decision = result["decision"]
    if decision["status"] != "KILL_POLAR_NORMAL_PREDICTOR_BRANCH":
        raise AssertionError("decision status")
    if decision["best_free_predictor"]["name"] != "router_pca_right_rank_128":
        raise AssertionError("free winner")
    if (
        decision["best_free_predictor_within_strict_fp16_field_gate"]["name"]
        != "identity_band_12"
    ):
        raise AssertionError("budgeted free winner")
    if decision["best_one_bit_coefficient_lower_bound"]["name"] != "identity_band_128":
        raise AssertionError("one-bit winner")
    if decision["best_fp16_predictor_within_strict_field_gate"]["name"] != "heldout_role_radial_implicit":
        raise AssertionError("FP16 winner")
    if decision["gpu_followup_warranted"]:
        raise AssertionError("GPU gate")

    receipt = {
        "schema": "qwen_polar_normal_predictor_verification_v1",
        "passed": True,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "result_internal_lock_sha256": claimed,
        "all_18_source_hashes_rechecked": True,
        "all_6_router_hashes_rechecked": True,
        "all_36_source_and_control_normals_rebuilt": True,
        "all_candidate_ledgers_and_waterfills_recomputed": True,
        "decisive_identity_router_and_implicit_projections_recomputed": True,
        "decision": decision["status"],
        "portable_exact_normal_hash_matches": sum(
            bool(row["hash_portable_match"]) for row in source_records
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"PASS: wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
