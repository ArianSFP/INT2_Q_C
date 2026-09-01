#!/usr/bin/env python3
"""Matched-Qwen/Gaussian screen for overcomplete binary cut factorization.

This is deliberately more favorable than NanoQuant in one important respect:
every binary outer product receives an uncharged, optimally refit FP32
coefficient.  The physical-rate axis nevertheless uses NanoQuant's actual
payload formula, r(n+m)+16(n+m), so the result is an opportunity screen and
not a codec claim.

The algorithm and all random seeds are identical for the Qwen source and its
moment-matched Gaussian control.  No learned table, target identity, or
activation data enters the search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import cupy as cp
import numpy as np


NAME_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<role>down_proj|gate_proj|up_proj)\.weight\.bf16\.bin"
)
ROWS = 768
COLS = 2048
WEIGHTS = ROWS * COLS
SCALE_BITS = 16 * (ROWS + COLS)
CHECKPOINTS = (64, 128, 256, 512, 768, 1024, 1185, 1380)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bf16(path: Path, role: str) -> np.ndarray:
    words = np.memmap(path, dtype="<u2", mode="r")
    if words.size != WEIGHTS:
        raise ValueError(f"unexpected weight count for {path}: {words.size}")
    values = (np.asarray(words).astype(np.uint32) << 16).view(np.float32)
    if role == "down_proj":
        return values.reshape(COLS, ROWS).T.copy()
    return values.reshape(ROWS, COLS).copy()


def sign_no_zero(values: cp.ndarray) -> cp.ndarray:
    return cp.where(values >= 0, cp.float32(1.0), cp.float32(-1.0))


def greedy_cut_curve(
    source: cp.ndarray,
    *,
    max_rank: int,
    starts: int,
    alternating_steps: int,
    seed: int,
) -> dict:
    """Greedy free-coefficient sign outer products on a full matrix."""

    residual = source.astype(cp.float32, copy=True)
    initial_sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
    if not math.isfinite(initial_sse) or initial_sse <= 0:
        raise ValueError("non-positive source energy")

    generator = cp.random.RandomState(seed)
    rows, cols = residual.shape
    norm2 = float(rows * cols)
    checkpoints = set(k for k in CHECKPOINTS if k <= max_rank)
    curve: list[dict] = []
    coefficient_sq_sum = 0.0
    started = time.perf_counter()

    for stage in range(1, max_rank + 1):
        # Multiple deterministic restarts are batched into thin GEMMs.  The
        # source and Gaussian control receive the same restart signs.
        vectors = generator.randint(0, 2, size=(cols, starts), dtype=cp.int8)
        vectors = vectors.astype(cp.float32) * cp.float32(2.0) - cp.float32(1.0)
        for _ in range(alternating_steps):
            left = sign_no_zero(residual @ vectors)
            vectors = sign_no_zero(residual.T @ left)
        projected = residual @ vectors
        left = sign_no_zero(projected)
        scores = cp.sum(left * projected, axis=0, dtype=cp.float64)
        winner = int(cp.argmax(scores).get())
        score = float(scores[winner].get())
        coefficient = score / norm2
        u = left[:, winner]
        v = vectors[:, winner]
        residual -= cp.float32(coefficient) * u[:, None] * v[None, :]
        coefficient_sq_sum += coefficient * coefficient

        if stage in checkpoints:
            sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
            relative_mse = sse / initial_sse
            physical_bits = stage * (rows + cols) + SCALE_BITS
            physical_bpw = physical_bits / (rows * cols)
            curve.append(
                {
                    "rank": stage,
                    "relative_mse": relative_mse,
                    "physical_bits_nanoquant_formula": physical_bits,
                    "physical_bpw_nanoquant_formula": physical_bpw,
                    "free_coefficient_bits_omitted": 32 * stage,
                    "coefficient_sq_sum": coefficient_sq_sum,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    return {
        "initial_sse": initial_sse,
        "curve": curve,
        "elapsed_seconds": time.perf_counter() - started,
    }


def matched_gaussian(matrix: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    gaussian = generator.standard_normal(matrix.shape, dtype=np.float32)
    source_mean = float(np.mean(matrix, dtype=np.float64))
    centered = matrix.astype(np.float64) - source_mean
    source_std = float(np.sqrt(np.mean(centered * centered)))
    gaussian = gaussian.astype(np.float64)
    gaussian -= float(np.mean(gaussian))
    gaussian *= source_std / float(np.sqrt(np.mean(gaussian * gaussian)))
    gaussian += source_mean
    return gaussian.astype(np.float32)


def aggregate(records: list[dict]) -> list[dict]:
    by_rank: dict[int, dict[str, float]] = {}
    for record in records:
        for domain in ("qwen", "gaussian"):
            energy = float(record[domain]["initial_sse"])
            for point in record[domain]["curve"]:
                rank = int(point["rank"])
                row = by_rank.setdefault(
                    rank,
                    {
                        "qwen_energy": 0.0,
                        "qwen_error": 0.0,
                        "gaussian_energy": 0.0,
                        "gaussian_error": 0.0,
                        "physical_bpw": float(point["physical_bpw_nanoquant_formula"]),
                    },
                )
                row[f"{domain}_energy"] += energy
                row[f"{domain}_error"] += energy * float(point["relative_mse"])

    output = []
    for rank in sorted(by_rank):
        row = by_rank[rank]
        qwen_d = row["qwen_error"] / row["qwen_energy"]
        gaussian_d = row["gaussian_error"] / row["gaussian_energy"]
        source_advantage = -0.5 * math.log2(qwen_d / gaussian_d)
        rate = row["physical_bpw"]
        output.append(
            {
                "rank": rank,
                "physical_bpw_nanoquant_formula": rate,
                "qwen_pooled_relative_mse": qwen_d,
                "gaussian_pooled_relative_mse": gaussian_d,
                "qwen_vs_gaussian_advantage_s_bpw": source_advantage,
                "qwen_absolute_F_vs_gaussian_limit": qwen_d * (2.0 ** (2.0 * rate)),
                "gaussian_operational_F": gaussian_d * (2.0 ** (2.0 * rate)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identities", nargs="+", required=True)
    parser.add_argument("--max-rank", type=int, default=1380)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--alternating-steps", type=int, default=3)
    args = parser.parse_args()

    if args.max_rank not in (1185, 1380):
        raise ValueError("frozen screen permits max-rank 1185 or 1380")
    if args.starts != 2 or args.alternating_steps != 3:
        raise ValueError("restart/alternation settings are protocol-frozen")

    wanted = set(args.identities)
    paths = []
    for path in sorted(args.source_dir.glob("*.bf16.bin")):
        match = NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        identity = f"layer{match.group('layer')}_expert{match.group('expert')}"
        if identity in wanted:
            paths.append((path, match.group("role"), identity))
    if len(paths) != 3 * len(wanted):
        raise ValueError(f"expected {3 * len(wanted)} matrices, found {len(paths)}")

    records = []
    for ordinal, (path, role, identity) in enumerate(paths):
        matrix = read_bf16(path, role)
        gaussian_seed = 0x51A7C000 + ordinal
        search_seed = 0xC07FAC70 + ordinal
        gaussian = matched_gaussian(matrix, gaussian_seed)
        qwen_result = greedy_cut_curve(
            cp.asarray(matrix),
            max_rank=args.max_rank,
            starts=args.starts,
            alternating_steps=args.alternating_steps,
            seed=search_seed,
        )
        gaussian_result = greedy_cut_curve(
            cp.asarray(gaussian),
            max_rank=args.max_rank,
            starts=args.starts,
            alternating_steps=args.alternating_steps,
            seed=search_seed,
        )
        records.append(
            {
                "tensor": path.name,
                "identity": identity,
                "role": role,
                "source_sha256": sha256_file(path),
                "source_mean": float(np.mean(matrix, dtype=np.float64)),
                "source_rms": float(np.sqrt(np.mean(matrix.astype(np.float64) ** 2))),
                "gaussian_seed": gaussian_seed,
                "search_seed": search_seed,
                "qwen": qwen_result,
                "gaussian": gaussian_result,
            }
        )
        del matrix, gaussian, qwen_result, gaussian_result
        cp.get_default_memory_pool().free_all_blocks()

    result = {
        "schema": "qwen-nanoquant-cut-factor-opportunity-v1",
        "claim_boundary": (
            "Greedy sign outer products with one uncharged FP32 coefficient per factor; "
            "physical rate is NanoQuant's factor+FP16-scale formula. This is a favorable "
            "source-opportunity screen, not a serialized codec or mathematical converse."
        ),
        "configuration": {
            "identities": args.identities,
            "rows": ROWS,
            "cols": COLS,
            "max_rank": args.max_rank,
            "starts": args.starts,
            "alternating_steps": args.alternating_steps,
            "checkpoints": [k for k in CHECKPOINTS if k <= args.max_rank],
            "required_total_source_advantage_s_bpw": 0.160964047443681,
            "required_increment_over_checkpoint_s_bpw": 0.15287192093,
        },
        "records": records,
        "aggregate_curve": aggregate(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
