#!/usr/bin/env python3
"""Favourable blockwise-coefficient extension of the operator aperture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_probe as op


WIDTHS = (768, 128, 64, 32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    post_path = args.post.resolve(strict=True)
    output = args.output.resolve()
    op.require(not output.exists(), "output must not exist")
    op.require(op.sha256(post_path) == op.EXPECTED_POST_SHA, "decoded reconstruction hash")
    props = op.cp.cuda.runtime.getDeviceProperties(0)
    device = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    uuid_hex = bytes(props["uuid"][:16]).hex()
    op.require(device == op.EXPECTED_DEVICE and uuid_hex == op.EXPECTED_UUID_HEX,
               "pinned local RTX 3060")
    started = time.perf_counter()
    plan = json.loads((release / "plan.lock.json").read_text(encoding="utf-8"))
    header = (release / "assets/header.bin").read_bytes()
    coefficients = struct.unpack_from("<12f", header, 32)
    post = np.memmap(post_path, dtype="<f8", mode="r",
                     shape=(op.common.GROUPS, op.common.GROUP_VALUES))
    aggregates: dict[str, dict[str, float | int]] = {}
    matrix_rows = []
    baseline_total = 0.0
    energy_total = 0.0

    for expert in range(op.common.EXPERTS):
        base = expert * op.common.GROUPS_PER_EXPERT
        gate = np.asarray(post[base:base + op.ROWS], dtype=np.float32)
        z0 = np.asarray(post[base + op.ROWS:base + 2 * op.ROWS], dtype=np.float32)
        z1 = np.asarray(post[base + 2 * op.ROWS:base + 3 * op.ROWS], dtype=np.float32)
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        recon_np = [gate, (cosine * z0 - sine * z1) / norm2,
                     (sine * z0 + cosine * z1) / norm2]
        source_np = []
        for target, row in enumerate(plan["sources"][3 * expert:3 * expert + 3]):
            path = source_root / row["source_relpath"]
            op.require(path.is_file() and op.sha256(path) == row["source_bf16_sha256"],
                       f"source binding {path}")
            source_np.append(op.bf16(path, tuple(row["shape"]), target == 2))
        matrices = [op.cp.asarray(value, dtype=op.cp.float32) for value in recon_np]
        grams, cubics = op.operator_basis(matrices)
        for target, role in enumerate(op.ROLES):
            source = op.cp.asarray(source_np[target], dtype=op.cp.float32)
            residual = source - matrices[target]
            baseline_sse = float(op.cp.sum(residual.astype(op.cp.float64) ** 2,
                                            dtype=op.cp.float64).get())
            source_energy = float(op.cp.sum(source.astype(op.cp.float64) ** 2,
                                             dtype=op.cp.float64).get())
            baseline_total += baseline_sse
            energy_total += source_energy
            names, stack, _ = op.operator_features(matrices, target, grams, cubics)
            banks = op.bank_names(names, target)
            row_record = {"expert_ordinal": expert, "role": role,
                          "baseline_sse": baseline_sse, "widths": {}}
            flat_residual = residual.reshape(-1)
            for width in WIDTHS:
                op.require(op.ROWS % width == 0, "row-block width")
                width_record = {}
                for bank, selected in banks.items():
                    index = np.asarray([names.index(name) for name in selected], dtype=np.int64)
                    sse = 0.0
                    maximum_condition = 0.0
                    for begin in range(0, op.ROWS, width):
                        lo = begin * op.COLS
                        hi = (begin + width) * op.COLS
                        local_stack = stack[:, lo:hi]
                        gram = op.cp.asnumpy(local_stack[index] @ local_stack[index].T).astype(np.float64)
                        cross = op.cp.asnumpy(local_stack[index] @ flat_residual[lo:hi]).astype(np.float64)
                        fit = op.fit_and_rescore(local_stack, flat_residual[lo:hi], index,
                                                gram, cross)
                        sse += float(fit["sse"])
                        maximum_condition = max(maximum_condition,
                                                float(fit["retained_condition_number"]))
                    key = f"{bank}_rows{width}"
                    coefficient_count = (op.ROWS // width) * len(selected)
                    width_record[bank] = {"sse": sse,
                                          "capture_fraction": 1.0 - sse / baseline_sse,
                                          "coefficient_count": coefficient_count,
                                          "maximum_condition_number": maximum_condition}
                    aggregate = aggregates.setdefault(key, {
                        "sse": 0.0, "coefficient_count": 0,
                        "bank": bank, "row_width": width})
                    aggregate["sse"] = float(aggregate["sse"]) + sse
                    aggregate["coefficient_count"] = int(aggregate["coefficient_count"]) + coefficient_count
                row_record["widths"][str(width)] = width_record
            matrix_rows.append(row_record)
            del source, residual, stack
            op.cp.get_default_memory_pool().free_all_blocks()
        del matrices, grams, cubics
        op.cp.get_default_memory_pool().free_all_blocks()

    op.require(abs(baseline_total - op.EXPECTED_BASELINE_SSE) < 3e-5, "baseline SSE")
    for row in aggregates.values():
        sse = float(row["sse"])
        count = int(row["coefficient_count"])
        side = 16.0 * count / op.common.WEIGHTS
        row["capture_fraction"] = 1.0 - sse / baseline_total
        row["nominal_private_fp16_coefficient_bpw"] = side
        row["favourable_transfer_F"] = (op.BASELINE_F * sse / baseline_total *
                                          2.0 ** (2.0 * side))
    strongest = min(aggregates, key=lambda key: float(aggregates[key]["sse"]))
    best = aggregates[strongest]
    status = ("SURVIVES_BLOCKWISE_10_PERCENT_APERTURE_REQUIRES_CONTROLS"
              if float(best["capture_fraction"]) >= 0.10 else
              "HARD_KILL_BLOCKWISE_OPERATOR_SPAN_BELOW_10_PERCENT_CAPTURE")
    report = {
        "schema": "operator-innovation-blockwise-qwen-local3060-v0",
        "status": status,
        "claim_boundary": (
            "Exact source-fitted coefficients independently granted to every row block "
            "over the fixed compact operator bank. This favourable aperture is not a "
            "finite codec, held-out model, or general nonlinear-operator converse."),
        "baseline": {"sse": baseline_total, "source_energy": energy_total,
                     "relative_mse": baseline_total / energy_total,
                     "F": op.BASELINE_F},
        "gate": {"capture_threshold": 0.10, "strongest_cell": strongest},
        "aggregate": aggregates,
        "matrices": matrix_rows,
        "bindings": {"container_sha256": op.EXPECTED_CONTAINER_SHA,
                     "decoded_post_sha256": op.sha256(post_path)},
        "runtime": {"device": device, "device_uuid_hex": uuid_hex,
                    "elapsed_seconds": time.perf_counter() - started},
    }
    report["result_sha256_excluding_self"] = hashlib.sha256(op.canonical(report)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "strongest": strongest, "best": best,
                      "output": str(output), "sha256": op.sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
