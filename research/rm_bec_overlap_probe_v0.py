#!/usr/bin/env python3
"""Source-free CuPy census of BEC-polar versus RM row selections.

This probe reads only published encoder metadata.  It does not open weight,
reconstruction, arithmetic-payload, or Gaussian-control files.  Set overlap is
not a rate/distortion result; it only measures how radical the proposed row-set
swap would be under the exact Q31 BEC construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FULL_Q31 = 1 << 31


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def capacity_q31(capacity: float) -> int:
    require(math.isfinite(capacity) and 0.0 <= capacity <= 1.0, "capacity")
    return min(FULL_Q31, max(0, int(round(float(capacity) * FULL_Q31))))


def _popcount_u32(values: Any, cp: Any) -> Any:
    """Portable vectorized popcount, avoiding backend-version intrinsics."""

    x = values.astype(cp.uint32, copy=True)
    x = x - ((x >> cp.uint32(1)) & cp.uint32(0x55555555))
    x = (x & cp.uint32(0x33333333)) + ((x >> cp.uint32(2)) & cp.uint32(0x33333333))
    x = (x + (x >> cp.uint32(4))) & cp.uint32(0x0F0F0F0F)
    x = x + (x >> cp.uint32(8))
    x = x + (x >> cp.uint32(16))
    return x & cp.uint32(0x3F)


def bec_scores_q31(capacity: float, n: int, cp: Any) -> Any:
    require(n > 0 and n & (n - 1) == 0, "power-of-two n")
    z = cp.full(n, FULL_Q31 - capacity_q31(capacity), dtype=cp.uint64)
    step = 1
    while step < n:
        view = z.reshape(-1, 2 * step)
        left = view[:, :step].copy()
        right = view[:, step:].copy()
        product = (left * right + cp.uint64(1 << 30)) >> cp.uint64(31)
        view[:, :step] = left + right - product
        view[:, step:] = product
        step *= 2
    return z


def one_capacity(capacity: float, n: int, cp: Any) -> dict[str, Any]:
    keep = min(n, max(0, int(math.ceil(n * float(capacity)))))
    index = cp.arange(n, dtype=cp.int64)
    scores = bec_scores_q31(capacity, n, cp)
    # NumPy's production construction uses lexsort((index, score)).
    bec_order = cp.lexsort(cp.stack((index, scores.astype(cp.int64)), axis=0))
    popcount = _popcount_u32(index.astype(cp.uint32), cp).astype(cp.int16)
    rm_order = cp.lexsort(cp.stack((index, -popcount.astype(cp.int64)), axis=0))
    bec_mask = cp.zeros(n, dtype=cp.bool_)
    rm_mask = cp.zeros(n, dtype=cp.bool_)
    bec_mask[bec_order[:keep]] = True
    rm_mask[rm_order[:keep]] = True
    intersection = int(cp.count_nonzero(bec_mask & rm_mask).item())
    symmetric_difference = int(cp.count_nonzero(bec_mask ^ rm_mask).item())
    by_weight = []
    variables = int(math.log2(n))
    for weight in range(variables + 1):
        class_mask = popcount == weight
        total = int(cp.count_nonzero(class_mask).item())
        bec_count = int(cp.count_nonzero(class_mask & bec_mask).item())
        rm_count = int(cp.count_nonzero(class_mask & rm_mask).item())
        if total:
            by_weight.append({"row_index_popcount": weight, "total": total,
                              "bec_selected": bec_count, "rm_selected": rm_count})
    return {
        "capacity": float(capacity),
        "capacity_q31": capacity_q31(capacity),
        "selected": keep,
        "intersection": intersection,
        "intersection_over_selected": (intersection / keep if keep else 1.0),
        "symmetric_difference": symmetric_difference,
        "symmetric_difference_over_n": symmetric_difference / n,
        "replacement_fraction_of_selected": ((keep - intersection) / keep if keep else 0.0),
        "bec_selected_by_row_index_popcount": by_weight,
    }


def metadata_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        raw = path.read_bytes()
        document = json.loads(raw)
        parameters = document["parameters"]
        capacities = parameters["capacity_schedule"]
        n = int(parameters["block_length"])
        require(len(capacities) == 6, "six-level capacity schedule")
        rows.append({
            "path": path.as_posix(),
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "n": n,
            "capacities": [float(value) for value in capacities],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-by-weight", action="store_true")
    args = parser.parse_args()
    import cupy as cp

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    inputs = metadata_rows(args.metadata)
    results = []
    for source in inputs:
        levels = [one_capacity(capacity, source["n"], cp)
                  for capacity in source["capacities"]]
        if not args.include_by_weight:
            for level in levels:
                del level["bec_selected_by_row_index_popcount"]
        results.append({**source, "levels": levels})
    aggregates = []
    for level_index in range(6):
        rows = [result["levels"][level_index] for result in results]
        selected = sum(row["selected"] for row in rows)
        intersection = sum(row["intersection"] for row in rows)
        symmetric_difference = sum(row["symmetric_difference"] for row in rows)
        total_n = sum(result["n"] for result in results)
        replacements = [row["replacement_fraction_of_selected"] for row in rows]
        aggregates.append({
            "level": level_index + 1,
            "selected": selected,
            "intersection": intersection,
            "weighted_replacement_fraction_of_selected":
                ((selected - intersection) / selected if selected else 0.0),
            "min_block_replacement_fraction_of_selected": min(replacements),
            "max_block_replacement_fraction_of_selected": max(replacements),
            "symmetric_difference": symmetric_difference,
            "symmetric_difference_over_total_n": symmetric_difference / total_n,
        })
    all_selected = sum(row["selected"] for result in results for row in result["levels"])
    all_intersection = sum(row["intersection"] for result in results for row in result["levels"])
    output = {
        "schema": "strata-rm-bec-source-free-overlap-v0",
        "claim_boundary": "row-set overlap only; no Qwen payload or RD claim",
        "backend": {
            "cupy_version": cp.__version__,
            "device_id": int(device.id),
            "device_name": properties["name"].decode()
            if isinstance(properties["name"], bytes) else str(properties["name"]),
        },
        "aggregate": {
            "blocks": len(results),
            "levels": aggregates,
            "all_levels_selected": all_selected,
            "all_levels_intersection": all_intersection,
            "all_levels_weighted_replacement_fraction_of_selected":
                ((all_selected - all_intersection) / all_selected if all_selected else 0.0),
        },
        "results": results,
    }
    encoded = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode()
    if args.output:
        args.output.write_bytes(encoded)
    else:
        print(encoded.decode(), end="")


if __name__ == "__main__":
    main()
