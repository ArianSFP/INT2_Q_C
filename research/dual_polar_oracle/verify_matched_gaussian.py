#!/usr/bin/env python3
"""Independent binding/arithmetic verifier for the dual-polar red-team."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
STACK_ROWS = 2304
VALUES = STACK_ROWS * COLS
EXPERTS = 6
PANEL = EXPERTS * VALUES
STIEFEL = VALUES - COLS * (COLS + 1) // 2
RATES = (2.15, 2.30, 2.50)
SIDE_BITS = 4096 + EXPERTS * (128 + 12)
REQUIRED_S = -0.5 * math.log2(0.8)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def close(left: float, right: float, *, rel: float = 3e-12, absolute: float = 3e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=rel, abs_tol=absolute):
        raise AssertionError((left, right))


def curve(spectrum: np.ndarray) -> list[tuple[int, int, int, int, float, float]]:
    singular = np.sort(np.asarray(spectrum, dtype=np.float64))
    prefix = np.concatenate(([0.0], np.cumsum(singular)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(singular * singular)))
    rows = []
    for rank in range(COLS - 1):
        unmodelled = COLS - rank
        sums = prefix[unmodelled:] - prefix[:-unmodelled]
        squares = prefix_sq[unmodelled:] - prefix_sq[:-unmodelled]
        errors = np.maximum(0.0, squares - sums * sums / unmodelled)
        start = int(np.argmin(errors))
        model = STIEFEL + 1 + COLS * rank - rank * (rank - 1) // 2
        if model >= VALUES:
            break
        rows.append((model, VALUES - model, start, start + unmodelled, float(sums[start] / unmodelled), float(errors[start])))
    return rows


def waterfill(records: list[dict[str, Any]], ranks: list[int], rate: float) -> dict[str, float]:
    payload = rate * PANEL - SIDE_BITS
    total_energy = sum(float(record["energy"]) for record in records)
    energies = []
    dimensions = []
    for record, rank in zip(records, ranks, strict=True):
        model, normal, _, _, _, residual = record["curve"][rank]
        residual = min(residual, float(record["energy"]) * (1.0 - 1e-15))
        energies.extend((float(record["energy"]) - residual, residual))
        dimensions.extend((model, normal))
    variances = np.asarray(energies) / np.asarray(dimensions)
    lo = float(np.min(variances)) * 2.0**-80
    hi = float(np.max(variances))
    for _ in range(150):
        theta = math.sqrt(lo * hi)
        used = 0.5 * sum(d * max(math.log2(v / theta), 0.0) for v, d in zip(variances, dimensions, strict=True))
        if used > payload:
            lo = theta
        else:
            hi = theta
    distortion = sum(d * min(v, hi) for v, d in zip(variances, dimensions, strict=True)) / total_energy
    f_value = distortion * 2.0 ** (2.0 * rate)
    return {"relative_mse": distortion, "F": f_value, "s_bpw": -0.5 * math.log2(f_value), "water_level": hi}


def validate_score(stored: dict[str, Any], records: list[dict[str, Any]]) -> None:
    rate = float(stored["physical_rate_bpw"])
    ranks = [int(value) for value in stored["ranks"]]
    rebuilt = waterfill(records, ranks, rate)
    for key, value in rebuilt.items():
        close(stored[key], value)
    close(stored["payload_rate_bpw"], (rate * PANEL - SIDE_BITS) / PANEL)
    if int(stored["side_bits"]) != SIDE_BITS:
        raise AssertionError("side bits")

    common = stored["common_rank_result"]
    common_rank = int(common["ranks"][0])
    if common["ranks"] != [common_rank] * EXPERTS:
        raise AssertionError("common rank schema")
    common_values = [waterfill(records, [rank] * EXPERTS, rate)["F"] for rank in range(len(records[0]["curve"]))]
    if common_rank != int(np.argmin(common_values)):
        raise AssertionError("common rank not global optimum")
    validate = waterfill(records, [common_rank] * EXPERTS, rate)
    for key, value in validate.items():
        close(common[key], value)

    for expert in range(EXPERTS):
        values = []
        for rank in range(len(records[expert]["curve"])):
            trial = list(ranks)
            trial[expert] = rank
            values.append(waterfill(records, trial, rate)["F"])
        if ranks[expert] != int(np.argmin(values)):
            raise AssertionError(("coordinate not locally optimal", expert, ranks[expert], int(np.argmin(values))))

    if len(stored["selected"]) != EXPERTS:
        raise AssertionError("selected rows")
    for selected, record, rank in zip(stored["selected"], records, ranks, strict=True):
        model, normal, start, stop, scale, residual = record["curve"][rank]
        expected = {
            "rank": rank,
            "model_dof": model,
            "normal_dof": normal,
            "window_start": start,
            "window_stop": stop,
            "common_scale": scale,
            "residual_energy": residual,
            "source_energy": record["energy"],
            "residual_energy_ratio": residual / record["energy"],
        }
        for key, value in expected.items():
            if isinstance(value, int):
                if int(selected[key]) != value:
                    raise AssertionError((key, selected[key], value))
            else:
                close(selected[key], value)


def records_from_serialized(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        spectrum = np.asarray(row["singular_values_ascending"], dtype=np.float64)
        if spectrum.shape != (COLS,) or np.any(np.diff(spectrum) < 0.0):
            raise AssertionError("spectrum schema")
        spectral_energy = float(np.sum(spectrum * spectrum, dtype=np.float64))
        close(row["spectral_energy"], spectral_energy, rel=2e-14, absolute=2e-10)
        close(row["relative_svd_energy_error"], abs(spectral_energy - float(row["energy"])) / float(row["energy"]), rel=2e-11)
        if float(row["relative_svd_energy_error"]) > 2e-5:
            raise AssertionError("SVD closure")
        for moments in row.get("control_moments", []):
            if float(moments["relative_centered_energy_error"]) > 3e-9:
                raise AssertionError("moment match")
        records.append({"energy": float(row["energy"]), "curve": curve(spectrum)})
    return records


def rebuild_mp_spectra(grid_points: int, energies: list[float]) -> list[np.ndarray]:
    aspect = COLS / STACK_ROWS
    root = math.sqrt(aspect)
    lower = (1.0 - root) ** 2
    upper = (1.0 + root) ** 2
    grid = np.linspace(lower, upper, grid_points, dtype=np.float64)
    density = np.sqrt(np.maximum(0.0, (upper - grid) * (grid - lower))) / (2.0 * math.pi * aspect * np.maximum(grid, 1e-300))
    step = grid[1] - grid[0]
    cdf = np.concatenate(([0.0], np.cumsum((density[:-1] + density[1:]) * (0.5 * step))))
    cdf /= cdf[-1]
    probability = (np.arange(COLS) + 0.5) / COLS
    base = np.sqrt(np.interp(probability, cdf, grid))
    return [base * math.sqrt(energy / float(np.sum(base * base))) for energy in energies]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--control-script", type=Path, required=True)
    parser.add_argument("--composite-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    checks = []
    if result["schema"] != "qwen_dual_polar_matched_gaussian_redteam_v1":
        raise AssertionError(result["schema"])
    content = dict(result)
    expected_hash = content.pop("result_content_sha256")
    if canonical_hash(content) != expected_hash:
        raise AssertionError("content hash")
    checks.extend(("schema", "canonical_content_hash"))

    bindings = (
        (args.source_lock, "source_lock_sha256"),
        (args.source_result, "source_result_sha256"),
        (args.source_script, "source_script_sha256"),
        (args.protocol, "protocol_sha256"),
        (args.composite_result, "composite_result_sha256"),
        (args.control_script, "executing_script_sha256"),
    )
    for path, key in bindings:
        if sha256_file(path) != result["binding"][key]:
            raise AssertionError((path, key))
    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    hashes = {row["source_bf16_sha256"] for row in lock["matrices"]}
    if {row["sha256"] for row in result["binding"]["source_receipts"]} != hashes or len(hashes) != 18:
        raise AssertionError("source receipts")
    for row in lock["matrices"]:
        path = args.source_lock.parent / row["output_relpath"]
        if sha256_file(path) != row["source_bf16_sha256"]:
            raise AssertionError(path)
    checks.extend(("all_parent_hashes", "all_18_source_hashes"))

    source_parent = json.loads(args.source_result.read_text(encoding="utf-8"))
    for key in ("F", "s_bpw", "relative_mse"):
        close(result["source"][key], source_parent["best"][key])
    if result["source"]["ranks"] != source_parent["best"]["ranks"]:
        raise AssertionError("source ranks")
    checks.append("source_score_binding")

    control_scores = []
    for replica in result["matched_gaussian"]["replicas"]:
        records = records_from_serialized(replica["records"])
        for stored, rate in zip(replica["scores"], RATES, strict=True):
            close(stored["physical_rate_bpw"], rate)
            validate_score(stored, records)
        best = min(replica["scores"], key=lambda row: (row["F"], row["physical_rate_bpw"]))
        for key in ("F", "s_bpw", "relative_mse"):
            close(replica["best"][key], best[key])
        control_scores.append(float(replica["scores"][0]["s_bpw"]))
    checks.extend(("all_control_spectra_and_moments", "all_control_rank_searches", "all_control_waterfills"))

    stored_control = result["matched_gaussian"]
    np.testing.assert_allclose(stored_control["s_at_2p15"], control_scores, rtol=2e-14, atol=2e-14)
    x = np.asarray(control_scores)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / math.sqrt(len(x))
    close(stored_control["mean_s_bpw"], mean)
    close(stored_control["sample_std_s_bpw"], std)
    close(stored_control["standard_error_s_bpw"], se)
    close(stored_control["three_se_lower_s_bpw"], mean - 3.0 * se)
    checks.append("control_statistics")

    mp = result["marchenko_pastur"]
    energies = [float(row["energy"]) for row in result["source"]["moments"]]
    rebuilt_spectra = rebuild_mp_spectra(int(mp["grid_points"]), energies)
    for stored, rebuilt in zip(mp["midpoint_quantile_spectra"], rebuilt_spectra, strict=True):
        np.testing.assert_allclose(stored, rebuilt, rtol=3e-14, atol=3e-14)
    mp_records = [{"energy": energy, "curve": curve(spectrum)} for energy, spectrum in zip(energies, rebuilt_spectra, strict=True)]
    for stored, rate in zip(mp["scores"], RATES, strict=True):
        close(stored["physical_rate_bpw"], rate)
        validate_score(stored, mp_records)
    checks.extend(("marchenko_pastur_quantiles", "marchenko_pastur_rank_search_and_waterfill"))

    source_s = float(result["source"]["s_bpw"])
    mp_s = float(mp["scores"][0]["s_bpw"])
    control_floor = min(mp_s, mean - 3.0 * se)
    excess = max(0.0, source_s - control_floor)
    diagnostic = result["diagnostic"]
    close(diagnostic["generic_fraction_of_source_s"], mean / source_s)
    close(diagnostic["source_minus_control_mean_s_bpw"], source_s - mean)
    close(diagnostic["favourable_control_floor_s_bpw"], control_floor)
    close(diagnostic["source_specific_excess_upper_s_bpw"], excess)
    if not (mp["scores"][0]["F"] < 1.0 and mean > 0.08):
        raise AssertionError("generic polar null not reproduced")
    checks.extend(("control_subtraction", "iid_gaussian_Rd_contradiction"))

    composite = json.loads(args.composite_result.read_text(encoding="utf-8"))
    horizontal = float(composite["variants"]["role_gauge+polar"]["rates"]["2.50"]["s_bpw"])
    nesting = result["nesting"]
    close(nesting["required_s_bpw"], REQUIRED_S)
    close(nesting["dual_raw_shortfall_bpw"], REQUIRED_S - source_s)
    close(nesting["role_horizontal_polar_s_bpw"], horizontal)
    close(nesting["raw_additive_s_bpw_invalid"], source_s + horizontal)
    close(nesting["raw_additive_shortfall_bpw"], REQUIRED_S - source_s - horizontal)
    close(nesting["source_specific_dual_excess_upper_s_bpw"], excess)
    close(nesting["favourable_zero_overlap_union_upper_s_bpw"], horizontal + excess)
    close(nesting["favourable_zero_overlap_union_shortfall_bpw"], REQUIRED_S - horizontal - excess)
    expected_decision = "PROMOTE_INTRINSIC_JOINT_NESTING" if horizontal + excess >= REQUIRED_S else "HARD_KILL_DUAL_POLAR_AND_NAIVE_NESTING"
    if result["decision"] != expected_decision:
        raise AssertionError("decision")
    checks.extend(("non_double_counting_union_arithmetic", "decision_reselection"))

    receipt = {
        "schema": "qwen_dual_polar_matched_gaussian_verification_v1",
        "passed": True,
        "check_count": len(checks),
        "checks": checks,
        "result_sha256": sha256_file(args.result),
        "result_content_sha256": expected_hash,
        "decision": expected_decision,
        "source_s_bpw": source_s,
        "matched_control_mean_s_bpw": mean,
        "marchenko_pastur_s_bpw": mp_s,
        "source_specific_excess_upper_s_bpw": excess,
        "favourable_zero_overlap_union_upper_s_bpw": horizontal + excess,
        "verification_boundary": "independently rebuilds all serialized spectra curves, selected ranks, local/global rank optimality, waterfills, MP quantiles, controls, and nesting arithmetic; regenerating Gaussian matrices and their SVDs is delegated to the source-bound deterministic experiment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
