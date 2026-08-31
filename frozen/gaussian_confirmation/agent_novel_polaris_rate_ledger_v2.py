#!/usr/bin/env python3
"""Exact whole-checkpoint budget for POLARIS-SC-v2's global reservoir."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BLOCK_LENGTH = 1 << 18
LEVELS = 6
PAYLOAD_POOL_BITS_PER_BLOCK = 563_464
LENGTH_BITS_PER_BLOCK = 32
SCALE_BITS_PER_BLOCK = 16
MATRIX_HEADER_BITS = 64
RANK1_STORAGE_BITS = 16
RANK1_HEADER_BITS = 64
SET_TAG_BITS_PER_LEVEL_POSITION = 2
FROZEN_BITS_PER_LEVEL_POSITION = 1
GLOBAL_FORMAT_HEADER_BITS = 4096
PHYSICAL_RESERVOIR_HEADER_BITS = 96 * 8
RATE_CAP_BPW = 2.15


def load_checkpoint_shapes(header_dir: Path) -> list[tuple[str, tuple[int, ...]]]:
    tensors: dict[str, tuple[int, ...]] = {}
    paths = sorted(header_dir.glob("*.safetensors.header.json"))
    if not paths:
        raise FileNotFoundError(f"no safetensors header JSON files in {header_dir}")
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        header = document.get("header", document)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            shape = tuple(int(value) for value in metadata["shape"])
            if name in tensors and tensors[name] != shape:
                raise ValueError(f"conflicting shape for {name}")
            tensors[name] = shape
    return sorted(tensors.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header-dir", type=Path, default=Path("qwen_weight_cache/headers"))
    parser.add_argument(
        "--output", type=Path, default=Path("agent_novel_polaris_rate_ledger_v2.json")
    )
    args = parser.parse_args()

    tensors = load_checkpoint_shapes(args.header_dir)
    rank2 = [(name, shape) for name, shape in tensors if len(shape) == 2]
    rank1 = [(name, shape) for name, shape in tensors if len(shape) == 1]
    unsupported = [(name, shape) for name, shape in tensors if len(shape) not in (1, 2)]
    if unsupported:
        raise ValueError(f"unsupported tensor ranks: {unsupported[:5]}")
    rank2_parameters = sum(math.prod(shape) for _, shape in rank2)
    rank1_parameters = sum(math.prod(shape) for _, shape in rank1)
    total_parameters = rank2_parameters + rank1_parameters
    nondivisible = [
        (name, shape, math.prod(shape) % BLOCK_LENGTH)
        for name, shape in rank2
        if math.prod(shape) % BLOCK_LENGTH
    ]
    if nondivisible:
        raise ValueError(f"rank-2 tensors are not block-exact: {nondivisible[:5]}")
    blocks = rank2_parameters // BLOCK_LENGTH

    payload_pool_bits = blocks * PAYLOAD_POOL_BITS_PER_BLOCK
    length_directory_bits = blocks * LENGTH_BITS_PER_BLOCK
    scale_directory_bits = blocks * SCALE_BITS_PER_BLOCK
    matrix_header_bits = len(rank2) * MATRIX_HEADER_BITS
    rank1_payload_bits = rank1_parameters * RANK1_STORAGE_BITS
    rank1_header_bits = len(rank1) * RANK1_HEADER_BITS
    set_tag_bits = LEVELS * BLOCK_LENGTH * SET_TAG_BITS_PER_LEVEL_POSITION
    frozen_bits = LEVELS * BLOCK_LENGTH * FROZEN_BITS_PER_LEVEL_POSITION
    total_bits = sum(
        (
            payload_pool_bits,
            length_directory_bits,
            scale_directory_bits,
            matrix_header_bits,
            rank1_payload_bits,
            rank1_header_bits,
            set_tag_bits,
            frozen_bits,
            GLOBAL_FORMAT_HEADER_BITS,
        )
    )
    cap_bits = math.floor(RATE_CAP_BPW * total_parameters)
    if total_bits > cap_bits:
        raise ValueError((total_bits, cap_bits))
    if total_bits % 8:
        raise ValueError("budget envelope is not byte aligned")
    if PHYSICAL_RESERVOIR_HEADER_BITS > GLOBAL_FORMAT_HEADER_BITS:
        raise ValueError("physical v2 header exceeds the global header reservation")

    gaussian_limit = 2.0 ** (-2.0 * RATE_CAP_BPW)
    result = {
        "architecture": (
            "POLARIS-SC-v2: six-level polar-lattice PTQ with checkpoint-global "
            "arithmetic overflow pooling"
        ),
        "status": "exact fail-closed serialized budget envelope; v2 confirmation pending",
        "checkpoint": {
            "parameters": total_parameters,
            "tensor_count": len(tensors),
            "rank2_tensors": len(rank2),
            "rank2_parameters": rank2_parameters,
            "rank1_tensors": len(rank1),
            "rank1_parameters": rank1_parameters,
        },
        "block_geometry": {
            "block_length": BLOCK_LENGTH,
            "levels": LEVELS,
            "rank2_blocks": blocks,
            "all_rank2_tensors_exact_multiples": True,
            "padding_parameters": 0,
        },
        "reservoir_framing": {
            "logical_payload_pool_bits_per_block": PAYLOAD_POOL_BITS_PER_BLOCK,
            "logical_payload_pool_bpw_rank2": PAYLOAD_POOL_BITS_PER_BLOCK / BLOCK_LENGTH,
            "u32_length_bits_per_block": LENGTH_BITS_PER_BLOCK,
            "fp16_scale_bits_per_block": SCALE_BITS_PER_BLOCK,
            "total_rank2_budget_bits_per_block": (
                PAYLOAD_POOL_BITS_PER_BLOCK
                + LENGTH_BITS_PER_BLOCK
                + SCALE_BITS_PER_BLOCK
            ),
            "local_payload_max": "u32; local excursions are allowed",
            "global_failure_rule": (
                "sum of logical arithmetic lengths must not exceed "
                "563464 * 116470 bits"
            ),
            "payload_layout": "all logical messages concatenated MSB-first with one global tail",
            "physical_reservoir_header_bits": PHYSICAL_RESERVOIR_HEADER_BITS,
            "global_header_reservation_bits": GLOBAL_FORMAT_HEADER_BITS,
            "matrix_header_bits_each": MATRIX_HEADER_BITS,
            "rank1_storage_bits_each": RANK1_STORAGE_BITS,
            "rank1_header_bits_each": RANK1_HEADER_BITS,
            "decoder_set_map": "2-bit F/I/S tag reserve per position per level",
            "frozen_coset_map": "one-bit reserve per position per level",
        },
        "exact_budget_bits": {
            "rank2_logical_payload_pool": payload_pool_bits,
            "rank2_u32_length_directory": length_directory_bits,
            "rank2_fp16_scale_directory": scale_directory_bits,
            "rank2_total": payload_pool_bits + length_directory_bits + scale_directory_bits,
            "rank2_matrix_headers": matrix_header_bits,
            "rank1_bf16_payload": rank1_payload_bits,
            "rank1_headers": rank1_header_bits,
            "six_level_set_tags": set_tag_bits,
            "six_level_frozen_coset_bits": frozen_bits,
            "global_format_header": GLOBAL_FORMAT_HEADER_BITS,
            "total": total_bits,
        },
        "whole_checkpoint_rate": {
            "total_bits": total_bits,
            "total_bytes": total_bits // 8,
            "bpw": total_bits / total_parameters,
            "cap_bpw": RATE_CAP_BPW,
            "cap_bits_floor": cap_bits,
            "headroom_bits": cap_bits - total_bits,
            "headroom_bytes_floor": (cap_bits - total_bits) // 8,
            "fits": total_bits <= cap_bits,
        },
        "confirmation_gates": {
            "mean_logical_payload_u32_bits_max": PAYLOAD_POOL_BITS_PER_BLOCK,
            "rate_rule": (
                "both the realized aggregate mean and its one-sided 99% Student-t UCB "
                "must not exceed 563464 bits per block"
            ),
            "gaussian_limit_at_2p15_bpw": gaussian_limit,
            "five_percent_mse_target": 1.05 * gaussian_limit,
            "distortion_rule": (
                "both absolute-MSE and sample-relative-MSE one-sided 99% Student-t "
                "UCBs must not exceed the five-percent target"
            ),
        },
        "reference": {
            "paper": "https://arxiv.org/html/1501.05683v5",
            "official_repository": "https://github.com/graceBaoXP/PolarLatticeQuantization",
            "repository_commit": "458187b9b03db1768a4b72d617e591f7862f6fca",
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
