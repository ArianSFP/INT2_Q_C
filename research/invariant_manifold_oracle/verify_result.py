#!/usr/bin/env python3
"""Independent arithmetic, binding, and byte-ledger verifier for result.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CHANNELS = 768
WIDTH = 2048
EXPERTS = 6
VALUES_PER_EXPERT = 3 * CHANNELS * WIDTH
PANEL_VALUES = EXPERTS * VALUES_PER_EXPERT
SYMMETRIC_DOF = CHANNELS * (CHANNELS + 1) // 2
STIEFEL_DOF = VALUES_PER_EXPERT - SYMMETRIC_DOF
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 2e-11) -> None:
    if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError((left, right))


def reverse_waterfill(components: list[tuple[float, float]], rate: float) -> tuple[float, float, float]:
    close(sum(x[0] for x in components), 1.0)
    close(sum(x[1] for x in components), 1.0)
    variances = [energy / dimension for dimension, energy in components]

    def used(log_level: float) -> float:
        return 0.5 * sum(
            dimension * max(0.0, math.log2(variance) - log_level)
            for (dimension, _), variance in zip(components, variances)
        )

    lo = min(math.log2(x) for x in variances) - 2.0 * rate / min(x[0] for x in components) - 16.0
    hi = max(math.log2(x) for x in variances)
    for _ in range(180):
        mid = 0.5 * (lo + hi)
        if used(mid) > rate:
            lo = mid
        else:
            hi = mid
    water = 2.0**hi
    mse = sum(dimension * min(variance, water) for (dimension, _), variance in zip(components, variances))
    f_value = mse * 2.0 ** (2.0 * rate)
    return mse, f_value, -0.5 * math.log2(f_value)


def check_source_score(result: dict[str, Any], rate_key: str) -> None:
    rate = float(rate_key)
    analyses = result["source_analyses"]
    total_energy = sum(float(x["source_energy"]) for x in analyses)
    for policy in ("best_common", "adaptive"):
        score = result["source_scores"][rate_key][policy]
        ranks = [int(x) for x in score["ranks"]]
        if len(ranks) != EXPERTS:
            raise AssertionError(ranks)
        if policy == "best_common" and len(set(ranks)) != 1:
            raise AssertionError("common-rank policy is not common")
        components: list[tuple[float, float]] = []
        for analysis, rank, selected in zip(analyses, ranks, score["selected"]):
            curve = analysis["curve"]
            if len(curve) != CHANNELS - 1 or int(curve[rank]["rank"]) != rank:
                raise AssertionError((analysis["identity"], rank))
            row = curve[rank]
            expected_h = 1 + CHANNELS * rank - rank * (rank - 1) // 2
            expected_model = STIEFEL_DOF + expected_h
            expected_normal = VALUES_PER_EXPERT - expected_model
            if int(row["model_dof"]) != expected_model or int(row["normal_dof"]) != expected_normal:
                raise AssertionError((rank, row["model_dof"], row["normal_dof"]))
            residual = float(row["source_residual_energy"])
            energy = float(analysis["source_energy"])
            close(float(row["source_residual_ratio"]), residual / energy)
            close(float(selected["source_residual_energy"]), residual)
            components.extend(
                [
                    (expected_model / PANEL_VALUES, (energy - residual) / total_energy),
                    (expected_normal / PANEL_VALUES, residual / total_energy),
                ]
            )
        mse, f_value, s_value = reverse_waterfill(components, rate)
        close(score["ideal_relative_mse"], mse)
        close(score["F"], f_value)
        close(score["s_bpw"], s_value)
        if bool(score["passes_target"]) != (f_value <= TARGET_F):
            raise AssertionError("target flag mismatch")


def check_control_arithmetic(result: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for rows in result["gaussian_controls"].values():
        for replicate in rows:
            for rate_key, policies in replicate["scores"].items():
                rate = float(rate_key)
                for policy in policies.values():
                    mse = float(policy["ideal_relative_mse"])
                    f_value = mse * 2.0 ** (2.0 * rate)
                    close(policy["F"], f_value)
                    close(policy["s_bpw"], -0.5 * math.log2(f_value))
                values.append(float(policies["adaptive"]["s_bpw"]))
    return values


def permutation_bits() -> int:
    return int(math.ceil(math.lgamma(CHANNELS + 1) / math.log(2.0)))


def check_ledgers(result: dict[str, Any]) -> None:
    for rate_key, ledger in result["side_and_read_ledger"].items():
        rate = float(rate_key)
        ideal = rate * PANEL_VALUES / 8.0
        if math.isclose(rate, 2.15):
            container = math.ceil(ideal)
        elif math.isclose(rate, 2.5):
            container = math.floor(ideal)
        else:
            container = round(ideal)
        global_bytes = 4092 + ((container - 4092) % EXPERTS)
        frame = (container - global_bytes) // EXPERTS
        objects = {
            "frame_header": 64,
            "gauge_fp32_768": 4 * CHANNELS,
            "canonical_permutation_enumerative": math.ceil(permutation_bits() / 8),
            "rank_label_u16": 2,
            "payload_directory": 32,
            "crc32": 4,
        }
        if ledger["per_expert_side_objects_bytes"] != objects:
            raise AssertionError((ledger["per_expert_side_objects_bytes"], objects))
        side = sum(objects.values())
        share = container / EXPERTS
        cold = global_bytes + frame
        if int(ledger["container_bytes"]) != container or int(ledger["equal_expert_frame_bytes"]) != frame:
            raise AssertionError("container/frame mismatch")
        if int(ledger["global_manifest_directory_bytes"]) != global_bytes:
            raise AssertionError("global byte mismatch")
        if int(ledger["per_expert_side_bytes"]) != side:
            raise AssertionError("side byte mismatch")
        if int(ledger["per_expert_payload_bytes"]) != frame - side:
            raise AssertionError("payload byte mismatch")
        close(ledger["actual_byte_derived_rate_bpw"], container * 8.0 / PANEL_VALUES)
        close(ledger["cold_read_amplification"], cold / share)
        if not (2.15 <= float(ledger["actual_byte_derived_rate_bpw"]) <= 2.5):
            raise AssertionError("rate outside target interval")
        if not cold / share < 2.0:
            raise AssertionError("cold read ceiling failed")


def load_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size != math.prod(shape):
        raise AssertionError((path, words.size, shape))
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(shape).astype(np.float64)


def directly_recompute_selected_residuals(result: dict[str, Any], root: Path) -> None:
    """Independently form source-space residual matrices for the chosen 2.5-bpw ranks."""
    source_root = root.resolve() / "blind_protocol_v2/unblinded"
    grouped: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for receipt in result["source_binding"]["receipts"]:
        path = source_root / receipt["relative_path"]
        if sha256_file(path) != receipt["declared_sha256"]:
            raise AssertionError(f"source hash mismatch: {path}")
        shape = tuple(int(x) for x in receipt["shape"])
        matrix = load_bf16(path, shape)
        role = str(receipt["role"])
        if role == "down":
            matrix = matrix.T
        if matrix.shape != (CHANNELS, WIDTH):
            raise AssertionError((path, matrix.shape))
        key = (int(receipt["layer"]), int(receipt["expert"]))
        grouped.setdefault(key, {})[role] = np.ascontiguousarray(matrix)

    selected_rows = result["source_scores"]["2.5"]["adaptive"]["selected"]
    analyses = result["source_analyses"]
    for analysis, selected in zip(analyses, selected_rows):
        layer_expert = analysis["identity"].replace("L", "").replace("E", "").split(":")
        key = (int(layer_expert[0]), int(layer_expert[1]))
        matrices = grouped[key]
        gate, up, down = (matrices[role] for role in ("gate", "up", "down"))
        up_norm = np.sqrt(np.sum(up * up, axis=1, dtype=np.float64))
        down_norm = np.sqrt(np.sum(down * down, axis=1, dtype=np.float64))
        gauge = np.sqrt(down_norm / up_norm)
        inv_gauge = 1.0 / gauge
        up_c = gauge[:, None] * up
        down_c = inv_gauge[:, None] * down
        joint = gate @ gate.T + up_c @ up_c.T + down_c @ down_c.T
        eigenvalues, vectors = np.linalg.eigh(joint)
        singular = np.sqrt(np.maximum(eigenvalues, np.finfo(np.float64).tiny))
        start = int(selected["unmodelled_window_start"])
        stop = int(selected["unmodelled_window_stop"])
        common = float(selected["common_singular_value"])
        alpha = np.zeros(CHANNELS, dtype=np.float64)
        alpha[start:stop] = 1.0 - common / singular[start:stop]

        direct = 0.0
        for canonical, source_scale in (
            (gate, np.ones(CHANNELS, dtype=np.float64)),
            (up_c, inv_gauge),
            (down_c, gauge),
        ):
            residual = vectors @ (alpha[:, None] * (vectors.T @ canonical))
            residual *= source_scale[:, None]
            direct += float(np.sum(residual * residual, dtype=np.float64))
        serialized = float(selected["source_residual_energy"])
        if not math.isclose(direct, serialized, rel_tol=5e-10, abs_tol=2e-10):
            raise AssertionError((analysis["identity"], direct, serialized))
        source_energy = sum(float(np.sum(x * x, dtype=np.float64)) for x in (gate, up, down))
        if not math.isclose(source_energy, float(analysis["source_energy"]), rel_tol=5e-12, abs_tol=2e-10):
            raise AssertionError((analysis["identity"], source_energy, analysis["source_energy"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="optionally rehash all eighteen pinned BF16 sources")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result["schema"] != "qwen_swiglu_gauge_coupled_polar_oracle_v1":
        raise AssertionError(result["schema"])
    for rate_key in result["source_scores"]:
        check_source_score(result, rate_key)
    controls = check_control_arithmetic(result)
    check_ledgers(result)

    source_max = max(float(x["adaptive"]["s_bpw"]) for x in result["source_scores"].values())
    close(result["calibration_summary"]["maximum_source_s_over_rate_grid_bpw"], source_max)
    close(result["calibration_summary"]["matched_control_mean_s_over_families_replicates_rates_bpw"], sum(controls) / len(controls))
    expected_decision = "SURVIVE_INFORMATION_GATE" if source_max >= TARGET_S else "KILL_INVARIANT_MANIFOLD_BRANCH"
    if result["decision"] != expected_decision:
        raise AssertionError((result["decision"], expected_decision))

    rehashed = False
    direct_residuals = False
    if args.root is not None:
        directly_recompute_selected_residuals(result, args.root)
        rehashed = True
        direct_residuals = True
    audit = {
        "schema": "qwen_swiglu_gauge_coupled_polar_verification_v1",
        "status": "PASS",
        "result_path": str(args.result.resolve()),
        "result_sha256": sha256_file(args.result),
        "source_files_rehashed": rehashed,
        "selected_source_residuals_directly_recomputed": direct_residuals,
        "checked": [
            "all serialized source rank curves have exact manifold dimensions",
            "common and adaptive source scores independently reverse-waterfill",
            "selected 2.5-bpw inverse-gauge residuals directly reconstructed in source space",
            "Gaussian-control F/s arithmetic",
            "all byte-derived rates and cold-read amplification ledgers",
            "early-kill decision",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
