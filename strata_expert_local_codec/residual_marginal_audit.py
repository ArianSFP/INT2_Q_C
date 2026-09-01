#!/usr/bin/env python3
"""Hash-bound marginal/scale/correlation audit of frozen STRATA v2 residuals.

This audit exists to avoid conflating an AR(1) prediction-energy ratio with an
entropy-power estimate.  It compares each authenticated pre-RHT staging block
with the corresponding independently decoded post-inverse-RHT reconstruction,
then reports moments, a moment-matched generalized-Gaussian entropy-power
ratio, group variance dispersion, and selected lag correlations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_expert_local_codec import common


LAGS = (1, 2, 4, 8, 16, 32)
REQUIRED_STRUCTURE_FACTOR = 0.8145432262848827
EARLY_KILL_EPR = 0.90
EARLY_KILL_CORRELATION = 0.01
HISTOGRAM_BINS = 2_048
HISTOGRAM_SIGMA_RANGE = 6.0
HISTOGRAM_PSEUDOCOUNT = 0.5


def decoded_path(root: Path, ordinal: int) -> Path:
    return (
        root
        / "independent_audit"
        / "decoded_sorted_blocks"
        / f"block_{ordinal:02d}_sorted_post_inverse_rht.f64.bin"
    )


def bf16_values(path: Path) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    return common.bf16_to_fp32(words).astype(np.float64)


def central_moments(sums: np.ndarray, count: int) -> tuple[float, float, float, float]:
    raw1, raw2, raw3, raw4 = (float(value) / count for value in sums)
    variance = raw2 - raw1 * raw1
    third = raw3 - 3.0 * raw1 * raw2 + 2.0 * raw1**3
    fourth = raw4 - 4.0 * raw1 * raw3 + 6.0 * raw1 * raw1 * raw2 - 3.0 * raw1**4
    return raw1, variance, third, fourth


def ggd_fit(variance: float, excess_kurtosis: float) -> dict[str, float]:
    kurtosis = excess_kurtosis + 3.0

    def residual(beta: float) -> float:
        log_ratio = (
            gammaln(5.0 / beta)
            + gammaln(1.0 / beta)
            - 2.0 * gammaln(3.0 / beta)
        )
        return math.exp(float(log_ratio)) - kurtosis

    beta = float(brentq(residual, 0.15, 100.0))
    log_alpha = 0.5 * (
        math.log(variance)
        + float(gammaln(1.0 / beta))
        - float(gammaln(3.0 / beta))
    )
    entropy_nats = (
        1.0 / beta
        + math.log(2.0 / beta)
        + log_alpha
        + float(gammaln(1.0 / beta))
    )
    entropy_power = math.exp(2.0 * entropy_nats) / (2.0 * math.pi * math.e)
    return {
        "shape_beta": beta,
        "scale_alpha": math.exp(log_alpha),
        "differential_entropy_nats": entropy_nats,
        "entropy_power": entropy_power,
        "entropy_power_over_variance": entropy_power / variance,
    }


def block_audit(
    root: Path, block: dict[str, Any], ordinal: int, histogram_edges: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    staging = root / str(block["staging_relpath"])
    decoded = decoded_path(root, ordinal)
    if common.sha256_file(staging) != block["staging_sha256"]:
        raise ValueError(f"staging hash mismatch block {ordinal}")
    source = bf16_values(staging)
    reconstruction = np.fromfile(decoded, dtype="<f8")
    if source.size != int(block["values"]) or reconstruction.size != source.size:
        raise ValueError(f"block geometry mismatch {ordinal}")
    residual = source - reconstruction
    count = residual.size
    sums = np.asarray(
        [
            np.sum(residual, dtype=np.float64),
            np.sum(residual**2, dtype=np.float64),
            np.sum(residual**3, dtype=np.float64),
            np.sum(residual**4, dtype=np.float64),
        ],
        dtype=np.float64,
    )
    mean, variance, third, fourth = central_moments(sums, count)
    centered = residual - mean
    skewness = third / variance**1.5
    excess = fourth / (variance * variance) - 3.0
    correlations = {
        str(lag): float(
            np.dot(centered[:-lag], centered[lag:]) / ((count - lag) * variance)
        )
        for lag in LAGS
    }
    groups = centered.reshape(-1, common.GROUP_VALUES)
    group_variances = np.mean(groups * groups, axis=1, dtype=np.float64)
    histogram, _ = np.histogram(
        centered / math.sqrt(variance), bins=histogram_edges
    )
    return (
        {
            "block_ordinal": ordinal,
            "values": count,
            "staging_relpath": block["staging_relpath"],
            "staging_sha256": block["staging_sha256"],
            "decoded_relpath": str(decoded.relative_to(root)).replace("\\", "/"),
            "decoded_sha256": common.sha256_file(decoded),
            "mean": mean,
            "variance": variance,
            "skewness": skewness,
            "excess_kurtosis": excess,
            "mean_absolute_over_sigma": float(
                np.mean(np.abs(centered), dtype=np.float64) / math.sqrt(variance)
            ),
            "max_absolute_over_sigma": float(
                np.max(np.abs(centered)) / math.sqrt(variance)
            ),
            "lag_correlations": correlations,
            "raw_power_sums": sums.tolist(),
            "standardized_histogram_counts": histogram.astype(int).tolist(),
        },
        group_variances,
        histogram,
    )


def crossfit_histogram(counts: list[np.ndarray], edges: np.ndarray) -> dict[str, Any]:
    gaussian_probability = np.diff(ndtr(edges))
    if not np.isclose(float(gaussian_probability.sum()), 1.0, rtol=0.0, atol=1e-14):
        raise AssertionError("Gaussian histogram probability does not sum to one")
    folds = []
    total_test = 0
    weighted_gain = 0.0
    for heldout_parity in (0, 1):
        train = np.sum(
            [row for ordinal, row in enumerate(counts) if ordinal % 2 != heldout_parity],
            axis=0,
            dtype=np.int64,
        )
        test = np.sum(
            [row for ordinal, row in enumerate(counts) if ordinal % 2 == heldout_parity],
            axis=0,
            dtype=np.int64,
        )
        model = (train.astype(np.float64) + HISTOGRAM_PSEUDOCOUNT) / (
            int(train.sum()) + HISTOGRAM_PSEUDOCOUNT * HISTOGRAM_BINS
        )
        count = int(test.sum())
        gaussian_nll = float(-np.dot(test, np.log2(gaussian_probability)) / count)
        histogram_nll = float(-np.dot(test, np.log2(model)) / count)
        gain = gaussian_nll - histogram_nll
        folds.append(
            {
                "heldout_block_parity": heldout_parity,
                "train_values": int(train.sum()),
                "test_values": count,
                "gaussian_discrete_nll_bpw": gaussian_nll,
                "crossfit_histogram_nll_bpw": histogram_nll,
                "gross_gain_bpw": gain,
            }
        )
        total_test += count
        weighted_gain += count * gain
    gross = weighted_gain / total_test
    table_bits = HISTOGRAM_BINS * 16
    table_bpw = table_bits / total_test
    return {
        "bins": HISTOGRAM_BINS,
        "sigma_range": HISTOGRAM_SIGMA_RANGE,
        "pseudocount": HISTOGRAM_PSEUDOCOUNT,
        "folds": folds,
        "gross_gain_bpw": gross,
        "one_uint16_probability_table_bits": table_bits,
        "one_table_cost_over_panel_bpw": table_bpw,
        "net_gain_after_one_table_bpw": gross - table_bpw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.v2_run.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    manifest_path = root / "preencoding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 14:
        raise ValueError("frozen v2 block manifest mismatch")

    rows: list[dict[str, Any]] = []
    group_variance_parts: list[np.ndarray] = []
    histogram_parts: list[np.ndarray] = []
    aggregate_sums = np.zeros(4, dtype=np.float64)
    count = 0
    histogram_edges = np.concatenate(
        (
            np.asarray([-np.inf]),
            np.linspace(
                -HISTOGRAM_SIGMA_RANGE,
                HISTOGRAM_SIGMA_RANGE,
                HISTOGRAM_BINS - 1,
                dtype=np.float64,
            ),
            np.asarray([np.inf]),
        )
    )
    for ordinal, block in enumerate(blocks):
        row, group_variances, histogram = block_audit(
            root, block, ordinal, histogram_edges
        )
        rows.append(row)
        group_variance_parts.append(group_variances)
        histogram_parts.append(histogram)
        aggregate_sums += np.asarray(row["raw_power_sums"], dtype=np.float64)
        count += int(row["values"])
    if count != common.WEIGHTS:
        raise AssertionError("residual audit does not cover the panel")

    mean, variance, third, fourth = central_moments(aggregate_sums, count)
    skewness = third / variance**1.5
    excess = fourth / (variance * variance) - 3.0
    fit = ggd_fit(variance, excess)
    group_variances = np.concatenate(group_variance_parts)
    gm_am = float(
        math.exp(float(np.mean(np.log(group_variances), dtype=np.float64)))
        / float(np.mean(group_variances, dtype=np.float64))
    )
    aggregate_lags = {
        str(lag): float(
            sum(
                (row["values"] - lag)
                * row["variance"]
                * row["lag_correlations"][str(lag)]
                for row in rows
            )
            / sum((row["values"] - lag) * row["variance"] for row in rows)
        )
        for lag in LAGS
    }
    max_abs_correlation = max(abs(value) for value in aggregate_lags.values())
    histogram_audit = crossfit_histogram(histogram_parts, histogram_edges)
    required_equivalent_gain = -0.5 * math.log2(REQUIRED_STRUCTURE_FACTOR)
    early_kill = (
        fit["entropy_power_over_variance"] > EARLY_KILL_EPR
        and max_abs_correlation < EARLY_KILL_CORRELATION
        and histogram_audit["net_gain_after_one_table_bpw"]
        < 0.1 * required_equivalent_gain
    )
    report = {
        "schema": "strata_v2_residual_marginal_audit_v1",
        "status": "passed",
        "purpose": (
            "distinguish residual marginal entropy-power evidence from AR(1) "
            "prediction-energy evidence"
        ),
        "bindings": {
            "v2_preencoding_manifest_sha256": common.sha256_file(manifest_path),
            "v2_container_sha256": common.sha256_file(root / "strata_xklt_sc_v2.bin"),
            "independent_decode_audit_sha256": common.sha256_file(
                root / "independent_audit" / "independent_decode_audit.json"
            ),
        },
        "aggregate": {
            "values": count,
            "mean": mean,
            "variance": variance,
            "skewness": skewness,
            "excess_kurtosis": excess,
            "ggd_moment_fit": fit,
            "group_variance_geometric_over_arithmetic": gm_am,
            "lag_correlations": aggregate_lags,
            "max_absolute_tested_lag_correlation": max_abs_correlation,
            "crossfit_standardized_histogram": histogram_audit,
        },
        "gate": {
            "required_structure_factor": REQUIRED_STRUCTURE_FACTOR,
            "required_equivalent_gain_bpw": required_equivalent_gain,
            "early_kill_entropy_power_threshold": EARLY_KILL_EPR,
            "early_kill_absolute_correlation_threshold": EARLY_KILL_CORRELATION,
            "residual_refinement_family_killed_before_gpu": early_kill,
        },
        "blocks": rows,
    }
    common.write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "entropy_power_over_variance": fit[
                    "entropy_power_over_variance"
                ],
                "excess_kurtosis": excess,
                "group_variance_geometric_over_arithmetic": gm_am,
                "max_absolute_tested_lag_correlation": max_abs_correlation,
                "crossfit_histogram_net_gain_bpw": histogram_audit[
                    "net_gain_after_one_table_bpw"
                ],
                "residual_refinement_family_killed_before_gpu": early_kill,
            },
            indent=2,
        )
    )
    if not early_kill:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
