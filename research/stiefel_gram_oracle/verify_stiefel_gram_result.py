#!/usr/bin/env python3
"""Independent arithmetic/source verifier for ``result.json``.

The verifier does not import the experiment program.  It re-derives the
reverse-waterfill scores, aggregate residuals, tight-frame spectral errors,
decision minimum, and all declared source hashes from the serialized record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


VALUES = 768 * 2048
RATE = 2.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def close(a: float, b: float, tolerance: float = 2e-11) -> None:
    if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError((a, b))


def independent_components_score(
    dimensions: list[float], energies: list[float], side: float
) -> tuple[float, float, float]:
    close(sum(dimensions), 1.0)
    close(sum(energies), 1.0)
    variances = [e / d for d, e in zip(dimensions, energies)]
    payload = RATE - side
    logs = [math.log2(x) for x in variances]
    low = min(logs) - 2.0 * payload / min(dimensions) - 32.0
    high = max(logs)
    for _ in range(200):
        middle = (low + high) / 2.0
        used = 0.5 * sum(d * max(0.0, lv - middle) for d, lv in zip(dimensions, logs))
        if used > payload:
            low = middle
        else:
            high = middle
    level = 2.0**high
    distortion = sum(d * min(v, level) for d, v in zip(dimensions, variances))
    f_value = distortion * 2.0 ** (2.0 * RATE)
    return distortion, f_value, -0.5 * math.log2(f_value)


def independent_score(residual: float, model_dof: float, side: float) -> tuple[float, float, float]:
    dimensions = [model_dof / VALUES, 1.0 - model_dof / VALUES]
    return independent_components_score(dimensions, [1.0 - residual, residual], side)


def check_score(score: dict[str, Any], residual: float, model_dof: int, side: float) -> None:
    distortion, f_value, s_value = independent_score(residual, model_dof, side)
    close(score["ideal_relative_mse"], distortion)
    close(score["F"], f_value)
    close(score["rate_equivalent_s_bpw"], s_value)
    close(score["shared_side_bpw"], side)


def check_serialized_component_score(score: dict[str, Any]) -> None:
    dimensions = [float(x["dimension_fraction"]) for x in score["allocations"]]
    energies = [float(x["energy_fraction"]) for x in score["allocations"]]
    distortion, f_value, s_value = independent_components_score(
        dimensions, energies, float(score["shared_side_bpw"])
    )
    close(score["ideal_relative_mse"], distortion)
    close(score["F"], f_value)
    close(score["rate_equivalent_s_bpw"], s_value)


def symmetric_rank_curve(singular: list[float]) -> list[dict[str, Any]]:
    rows = len(singular)
    prefix = [0.0]
    prefix_sq = [0.0]
    for value in singular:
        prefix.append(prefix[-1] + value)
        prefix_sq.append(prefix_sq[-1] + value * value)
    stiefel_dof = rows * 2048 - rows * (rows + 1) // 2
    curve = []
    for rank in range(rows - 1):
        length = rows - rank
        best = None
        for start in range(rank + 1):
            total = prefix[start + length] - prefix[start]
            total_sq = prefix_sq[start + length] - prefix_sq[start]
            error = max(0.0, total_sq - total * total / length)
            candidate = (error, start, total / length)
            if best is None or candidate < best:
                best = candidate
        dof = stiefel_dof + 1 + rows * rank - rank * (rank - 1) // 2
        if dof >= VALUES:
            break
        curve.append(
            {
                "rank": rank,
                "model_dof": dof,
                "normal_dof": VALUES - dof,
                "residual_energy": best[0],
                "start": best[1],
                "stop": best[1] + length,
                "scale": best[2],
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result_path = args.result.resolve()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if data["schema"] != "qwen-stiefel-gram-oracle-v1":
        raise AssertionError(data["schema"])
    recorded_lock = Path(data["audit"]["source_lock_path"])
    lock_path = recorded_lock if recorded_lock.is_file() else root / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    if data["audit"]["source_lock_file_sha256"] != sha256(lock_path):
        raise AssertionError("source lock hash")
    recorded_script = Path(data["audit"]["script_path"])
    colocated_script = result_path.parent / "stiefel_gram_oracle.py"
    script_path = recorded_script if recorded_script.is_file() else colocated_script
    if data["audit"]["script_sha256"] != sha256(script_path):
        raise AssertionError("experiment script hash")
    for receipt in data["audit"]["source_receipts"]:
        path = root / "blind_protocol_v2/unblinded" / receipt["path_relative_to_source_lock_parent"]
        observed = sha256(path)
        if observed != receipt["declared_sha256"] or observed != receipt["observed_sha256"]:
            raise AssertionError((path, observed))

    public = data["matrices"]
    if len(public) != 18:
        raise AssertionError(len(public))
    total_energy = sum(float(x["energy"]) for x in public)
    for matrix in public:
        singular = [float(x) for x in matrix["singular_values_ascending"]]
        energy = sum(x * x for x in singular)
        close(energy, matrix["energy"], 3e-11)
        mean = sum(singular) / len(singular)
        residual = sum((x - mean) ** 2 for x in singular) / energy
        close(residual, matrix["tight_frame"]["residual_energy_ratio"], 3e-11)
        history = matrix["left_diagonal_scaled_frame"]["monotone_history"]
        if any(float(b) > float(a) + 2e-11 for a, b in zip(history, history[1:])):
            raise AssertionError("non-monotone recorded DQ history")
        close(min(history), matrix["left_diagonal_scaled_frame"]["optimized_residual_ratio"])

    direct_fields = {
        "nearest_scaled_tight_frame": lambda x: x["tight_frame"]["residual_energy_ratio"],
        "alternating_nearest_left_diagonal_scaled_frame": lambda x: x["left_diagonal_scaled_frame"]["optimized_residual_ratio"],
        "polar_fixed_right_diagonal_scaled_frame": lambda x: x["polar_fixed_right_diagonal_scaled_frame"]["residual_energy_ratio"],
    }
    candidates: list[tuple[str, float, float]] = []
    for name, getter in direct_fields.items():
        item = data["direct_frame_models"][name]
        aggregate = sum(float(x["energy"]) * float(getter(x)) for x in public) / total_energy
        close(item["residual_energy_ratio"], aggregate)
        check_score(item["score"], aggregate, int(item["model_dof"]), 0.0)
        check_serialized_component_score(item["panel_component_score"])
        candidates.append(
            (
                f"direct.{name}",
                item["panel_component_score"]["F"],
                item["panel_component_score"]["rate_equivalent_s_bpw"],
            )
        )

    heldout = data["heldout_shared_models"]
    heldout_total = sum(float(x["energy"]) for x in heldout["matrices"])
    for name, item in heldout["aggregates"].items():
        aggregate = sum(
            float(x["energy"]) * float(x[name]["residual_energy_ratio"])
            for x in heldout["matrices"]
        ) / heldout_total
        close(item["residual_energy_ratio"], aggregate)
        dof = int(item["model_dof"])
        check_score(item["optimistic_shared_table_free_score"], aggregate, dof, 0.0)
        check_serialized_component_score(item["optimistic_panel_component_free_score"])
        candidates.append(
            (
                f"heldout.{name}",
                item["optimistic_panel_component_free_score"]["F"],
                item["optimistic_panel_component_free_score"]["rate_equivalent_s_bpw"],
            )
        )
        for score_name in ("panel_local_fp16_table_score", "full_model_amortized_fp16_table_score"):
            if score_name in item:
                score = item[score_name]
                check_score(score, aggregate, dof, float(score["shared_side_bpw"]))
        if "panel_component_full_model_amortized_fp16_table_score" in item:
            check_serialized_component_score(item["panel_component_full_model_amortized_fp16_table_score"])

    check_serialized_component_score(data["panel_energy_only_gaussian_allocation_baseline"])

    # Independently rebuild the exact cI + symmetric-rank-k polar-normal curve
    # from the serialized spectra.  No curve values from the experiment are
    # trusted for this reconstruction.
    rank_data = data["structured_gram_rank_models"]
    rebuilt_curves = [symmetric_rank_curve([float(v) for v in x["singular_values_ascending"]]) for x in public]

    def panel_rank_score(picks: list[int], side: float = 0.0) -> tuple[float, float, float]:
        dimensions: list[float] = []
        energies: list[float] = []
        for matrix, curve, rank in zip(public, rebuilt_curves, picks):
            row = curve[rank]
            dof = float(row["model_dof"])
            residual_energy = float(row["residual_energy"])
            dimensions.extend([dof / (18 * VALUES), (VALUES - dof) / (18 * VALUES)])
            energies.extend(
                [
                    (float(matrix["energy"]) - residual_energy) / total_energy,
                    residual_energy / total_energy,
                ]
            )
        return independent_components_score(dimensions, energies, side)

    serialized_curve = rank_data["common_rank_curve"]
    if len(serialized_curve) != min(len(x) for x in rebuilt_curves):
        raise AssertionError("common rank curve length")
    rebuilt_common = []
    for rank, serialized in enumerate(serialized_curve):
        residual = sum(x[rank]["residual_energy"] for x in rebuilt_curves) / total_energy
        dof = rebuilt_curves[0][rank]["model_dof"]
        distortion, f_value, s_value = panel_rank_score([rank] * 18)
        structural = independent_score(residual, dof, 0.0)
        if int(serialized["rank"]) != rank or int(serialized["model_dof"]) != dof:
            raise AssertionError((rank, serialized["rank"], dof, serialized["model_dof"]))
        close(serialized["residual_energy_ratio"], residual)
        close(serialized["panel_component_F"], f_value)
        close(serialized["panel_component_s_bpw"], s_value)
        close(serialized["structural_only_F"], structural[1])
        close(serialized["structural_only_s_bpw"], structural[2])
        rebuilt_common.append((f_value, s_value, rank, residual, dof, distortion))
    best_common = min(rebuilt_common, key=lambda x: x[0])
    recorded_common = rank_data["best_common_rank"]
    if int(recorded_common["rank"]) != best_common[2]:
        raise AssertionError((recorded_common["rank"], best_common[2]))
    check_serialized_component_score(recorded_common["panel_component_score"])
    close(recorded_common["panel_component_score"]["F"], best_common[0])
    close(recorded_common["panel_component_score"]["rate_equivalent_s_bpw"], best_common[1])
    check_score(
        recorded_common["structural_only_score"],
        best_common[3],
        best_common[4],
        0.0,
    )
    candidates.append(
        (
            "structured_gram.best_common_rank",
            recorded_common["panel_component_score"]["F"],
            recorded_common["panel_component_score"]["rate_equivalent_s_bpw"],
        )
    )

    # The adaptive result must independently minimize each all-active log-F
    # contribution and must then agree with the general reverse-waterfiller.
    expected_picks = []
    for matrix, curve in zip(public, rebuilt_curves):
        options = []
        for row in curve:
            dof = float(row["model_dof"])
            residual_energy = float(row["residual_energy"])
            dm = dof / (18 * VALUES)
            dn = (VALUES - dof) / (18 * VALUES)
            em = (float(matrix["energy"]) - residual_energy) / total_energy
            en = residual_energy / total_energy
            options.append(dm * math.log2(em / dm) + dn * math.log2(en / dn))
        expected_picks.append(min(range(len(options)), key=options.__getitem__))
    adaptive = rank_data["adaptive_rank"]
    recorded_picks = [int(x["rank"]) for x in adaptive["selections"]]
    if recorded_picks != expected_picks:
        raise AssertionError((recorded_picks, expected_picks))
    for selection, curve, rank in zip(adaptive["selections"], rebuilt_curves, expected_picks):
        rebuilt = curve[rank]
        close(selection["residual_energy"], rebuilt["residual_energy"])
        if int(selection["model_dof"]) != rebuilt["model_dof"]:
            raise AssertionError("adaptive dof")
        if int(selection["unmodeled_window_start"]) != rebuilt["start"]:
            raise AssertionError("adaptive window")
    check_serialized_component_score(adaptive["panel_component_score_rank_labels_free"])
    check_serialized_component_score(adaptive["panel_component_score_rank_labels_charged"])
    adaptive_free = panel_rank_score(expected_picks, 0.0)
    adaptive_charged = panel_rank_score(expected_picks, float(adaptive["rank_label_side_bpw"]))
    close(adaptive["panel_component_score_rank_labels_free"]["F"], adaptive_free[1])
    close(adaptive["panel_component_score_rank_labels_free"]["rate_equivalent_s_bpw"], adaptive_free[2])
    close(adaptive["panel_component_score_rank_labels_charged"]["F"], adaptive_charged[1])
    close(adaptive["panel_component_score_rank_labels_charged"]["rate_equivalent_s_bpw"], adaptive_charged[2])
    adaptive_residual = sum(float(x["residual_energy"]) for x in adaptive["selections"]) / total_energy
    adaptive_average_dof = sum(int(x["model_dof"]) for x in adaptive["selections"]) / 18
    check_score(adaptive["structural_only_score"], adaptive_residual, adaptive_average_dof, 0.0)
    candidates.append(
        (
            "structured_gram.adaptive_rank",
            adaptive["panel_component_score_rank_labels_free"]["F"],
            adaptive["panel_component_score_rank_labels_free"]["rate_equivalent_s_bpw"],
        )
    )

    best = min(candidates, key=lambda x: x[1])
    decision = data["decision"]
    if best[0] != decision["best_optimistic_free_table_candidate"]:
        raise AssertionError((best[0], decision["best_optimistic_free_table_candidate"]))
    close(best[1], decision["best_optimistic_F_at_2p5"])
    close(best[2], decision["best_optimistic_rate_equivalent_s_bpw"])
    expected_status = "hard_kill" if best[2] < float(decision["required_rate_equivalent_s_bpw"]) else "survives"
    if decision["status"] != expected_status:
        raise AssertionError((decision["status"], expected_status))

    verifier_path = Path(__file__).resolve()
    audit = {
        "schema": "qwen-stiefel-gram-oracle-independent-audit-v1",
        "status": "pass",
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "experiment_script_sha256": data["audit"]["script_sha256"],
        "verifier_path": str(verifier_path),
        "verifier_sha256": sha256(verifier_path),
        "source_hashes_recomputed": len(data["audit"]["source_receipts"]),
        "source_hashes_all_match": True,
        "matrix_spectra_recomputed_from_serialized_values": len(public),
        "all_scores_independently_recomputed": True,
        "decision": decision,
    }
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
