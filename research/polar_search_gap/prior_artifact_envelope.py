#!/usr/bin/env python3
"""Audit the observed STRATA-v2 MAP-SC finite-code gap without source access."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--projected-mse", type=float, default=0.03090139432980219)
    parser.add_argument("--target-mse", type=float, default=0.025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        Path(value)
        for value in sorted(
            glob.glob(
                str(
                    args.repo_root
                    / "strata_v2_blind_one_shot_v2"
                    / "encoded"
                    / "block_*.json"
                )
            )
        )
    ]
    if [path.stem for path in paths] != [f"block_{index:02d}" for index in range(14)]:
        raise RuntimeError("the frozen fourteen-block evidence panel is incomplete")

    rows: list[dict[str, object]] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        parameters = document["parameters"]
        trial = document["trials"][0]
        if parameters["decision"] != "map" or document["schema"] != "strata_xklt_sc_v2_single_block_encoder_v1":
            raise RuntimeError(f"unexpected encoder mode/schema in {path}")
        n = int(parameters["block_length"])
        nominal = float(parameters["test_channel_distortion"])
        mse = float(trial["relative_mse"])
        gaussian = float(trial["gaussian_limit_mse_at_screen_rate"])
        rows.append(
            {
                "block": path.stem,
                "metadata_sha256": sha256_file(path),
                "n": n,
                "nominal_test_channel_distortion": nominal,
                "measured_relative_mse": mse,
                "gaussian_mse_at_exact_screen_rate": gaussian,
                "mse_over_nominal_distortion": mse / nominal,
                "mse_over_gaussian_at_screen_rate": mse / gaussian,
                "screen_bpw": float(trial["screen_bpw"]),
                "arithmetic_logical_bpw": float(trial["arithmetic_logical_bits"]) / n,
            }
        )

    total_values = sum(int(row["n"]) for row in rows)
    if total_values != 28_311_552:
        raise RuntimeError(f"unexpected panel size {total_values}")

    def weighted(field: str) -> float:
        return sum(int(row["n"]) * float(row[field]) for row in rows) / total_values

    measured = weighted("measured_relative_mse")
    nominal = weighted("nominal_test_channel_distortion")
    gaussian = weighted("gaussian_mse_at_exact_screen_rate")
    nominal_factor = measured / nominal
    gaussian_factor = measured / gaussian
    block_nominal_factors = [float(row["mse_over_nominal_distortion"]) for row in rows]
    block_gaussian_factors = [float(row["mse_over_gaussian_at_screen_rate"]) for row in rows]
    required_factor = args.projected_mse / args.target_mse
    required_bits = 0.5 * math.log2(required_factor)
    nominal_closure_bits = 0.5 * math.log2(nominal_factor)
    gaussian_closure_bits = 0.5 * math.log2(gaussian_factor)
    result = {
        "schema": "strata_polar_map_sc_prior_artifact_envelope_v1",
        "scope": "source-free audit of frozen, independently decoded STRATA-v2 metadata",
        "claim_boundary": (
            "closing the measured MAP-SC excess down to nominal test-channel D is a generous "
            "empirical opportunity envelope, not a fixed-source information-theoretic proof"
        ),
        "current_goal": {
            "projected_current_mse": args.projected_mse,
            "target_mse_at_2p5_bpw": args.target_mse,
            "required_relative_mse_reduction": 1.0 - args.target_mse / args.projected_mse,
            "required_multiplicative_factor": required_factor,
            "required_equivalent_gain_bpw_from_current": required_bits,
            "required_equivalent_gain_bpw_from_gaussian": 0.5 * math.log2(1.0 / 0.8),
        },
        "frozen_panel": {
            "values": total_values,
            "blocks": len(rows),
            "weighted_measured_relative_mse": measured,
            "weighted_nominal_test_channel_distortion": nominal,
            "weighted_gaussian_mse_at_exact_screen_rate": gaussian,
            "weighted_mse_over_nominal_distortion": nominal_factor,
            "weighted_mse_over_gaussian_at_screen_rate": gaussian_factor,
            "block_mse_over_nominal_min": min(block_nominal_factors),
            "block_mse_over_nominal_max": max(block_nominal_factors),
            "block_mse_over_gaussian_min": min(block_gaussian_factors),
            "block_mse_over_gaussian_max": max(block_gaussian_factors),
        },
        "generous_full_gap_closure": {
            "nominal_D_relative_mse_reduction": 1.0 - 1.0 / nominal_factor,
            "nominal_D_equivalent_gain_bpw": nominal_closure_bits,
            "gaussian_screen_relative_mse_reduction": 1.0 - 1.0 / gaussian_factor,
            "gaussian_screen_equivalent_gain_bpw": gaussian_closure_bits,
            "projected_mse_after_nominal_D_factor_closure": args.projected_mse / nominal_factor,
            "projected_mse_after_gaussian_factor_closure": args.projected_mse / gaussian_factor,
            "required_gain_over_nominal_closure_ratio": required_bits / nominal_closure_bits,
            "required_gain_over_gaussian_closure_ratio": required_bits / gaussian_closure_bits,
            "still_misses_target_after_full_observed_gap_closure": bool(
                args.projected_mse / gaussian_factor > args.target_mse
            ),
        },
        "decision": (
            "Do not spend a full Qwen block GPU run on SCL/coordinate search unless a "
            "source-free diagnostic reveals a qualitatively new >0.15 bpw search gap."
        ),
        "bindings": {
            "script_sha256_before_result_write": sha256_file(Path(__file__)),
            "release_manifest_sha256": sha256_file(
                args.repo_root / "release" / "strata_v2_release_manifest.json"
            ),
            "independent_decode_audit_sha256": sha256_file(
                args.repo_root
                / "strata_v2_blind_one_shot_v2"
                / "independent_audit"
                / "independent_decode_audit.json"
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
