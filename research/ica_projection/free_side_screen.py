#!/usr/bin/env python3
"""Run only the deliberately leaky free-side ICA screen, CPU-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ica_projection_oracle import (
    REQUIRED_GAIN_BPW,
    all_values,
    canonical_json,
    load_sample_bank,
    screen_candidates,
    select_fit_values,
    sha256_file,
    stable_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimensions", default="8,16,32,64")
    parser.add_argument("--representations", default="raw,xklt")
    parser.add_argument("--vectors-per-matrix", type=int, default=2048)
    parser.add_argument("--fit-vectors-per-matrix", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--histogram-bins", type=int, default=128)
    parser.add_argument("--histogram-edge", type=float, default=8.0)
    args = parser.parse_args()
    started = time.time()
    dimensions = [int(item) for item in args.dimensions.split(",") if item]
    representations = [item for item in args.representations.split(",") if item]
    bank, provenance = load_sample_bank(args.plan, dimensions, args.vectors_per_matrix)
    rows = []
    for representation in representations:
        for dimension in dimensions:
            fit = select_fit_values(
                bank, representation, dimension, list(range(6)), args.fit_vectors_per_matrix
            )
            evaluation = all_values(bank, representation, dimension, list(range(6)))
            rng = np.random.default_rng(
                stable_seed("QWEN-ICA-GAUSS-SCREEN-v1", representation, dimension)
            )
            gaussian = rng.standard_normal(evaluation.shape)
            _, selected, candidates = screen_candidates(
                fit,
                evaluation,
                gaussian,
                args.iterations,
                args.tolerance,
                args.histogram_bins,
                args.histogram_edge,
            )
            row = {
                "representation": representation,
                "dimension": dimension,
                "selected": selected,
                "candidates": candidates,
            }
            rows.append(row)
            print(
                representation,
                dimension,
                selected["contrast"],
                selected["optimistic_shape_plus_variance_bpw"],
                flush=True,
            )
    best = max(rows, key=lambda row: row["selected"]["optimistic_shape_plus_variance_bpw"])
    output = {
        "schema": "qwen-hidden-ica-free-side-screen/v1",
        "warning": "Leaky in-panel selection; all transform/model bits free. This is intentionally optimistic and is only an early-kill screen.",
        "required_gain_bpw": REQUIRED_GAIN_BPW,
        "configuration": vars(args) | {
            "plan": str(args.plan.resolve()),
            "output": str(args.output.resolve()),
            "dimensions": dimensions,
            "representations": representations,
            "gpu_used": False,
        },
        "provenance": provenance,
        "results": rows,
        "summary": {
            "best_representation": best["representation"],
            "best_dimension": best["dimension"],
            "best_contrast": best["selected"]["contrast"],
            "best_optimistic_gain_bpw": best["selected"]["optimistic_shape_plus_variance_bpw"],
            "fraction_of_required_gain": best["selected"]["optimistic_shape_plus_variance_bpw"] / REQUIRED_GAIN_BPW,
            "shortfall_bpw": REQUIRED_GAIN_BPW - best["selected"]["optimistic_shape_plus_variance_bpw"],
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "engine_script_sha256": sha256_file(Path(__file__).resolve().parent / "ica_projection_oracle.py"),
            "numpy": np.__version__,
        },
    }
    output["result_seal_sha256"] = hashlib.sha256(canonical_json(output)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
