#!/usr/bin/env python3
"""Explicit universal SwiGLU shape, tail, and exact-coarse-rate contract."""

from __future__ import annotations

import math
from dataclasses import dataclass


BLOCK_VALUES = 4096
COARSE_RATE_NUMERATOR = 307
COARSE_RATE_DENOMINATOR = 128
ROLE_ORDER = ("gate", "up", "down_transposed")
MAX_U32 = (1 << 32) - 1
MAX_WEIGHTS = (1 << 63) - 1


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


@dataclass(frozen=True)
class ShapeContract:
    intermediate: int
    hidden: int
    role_values: int
    total_values: int
    blocks_per_role: int
    last_block_valid_values: int
    tail_values_per_role: int
    coarse_bytes: int

    @property
    def tail_free(self) -> bool:
        return self.tail_values_per_role == 0

    def valid_values_for_block(self, block: int) -> int:
        require(type(block) is int and 0 <= block < self.blocks_per_role,
                "block index")
        return BLOCK_VALUES if block + 1 < self.blocks_per_role else self.last_block_valid_values


def define_shape(intermediate: int, hidden: int) -> ShapeContract:
    """Return the exact supported-shape contract or reject it explicitly.

    Universality here means no checkpoint/layer/expert identity and arbitrary
    positive SwiGLU dimensions *within this physical packet format*.  The
    exact 307/128-bpw coarse stream must have an integral byte length; shapes
    outside that arithmetic domain are rejected rather than silently rounded.
    """

    require(type(intermediate) is int and 0 < intermediate <= MAX_U32,
            "positive uint32 intermediate")
    require(type(hidden) is int and 0 < hidden <= MAX_U32,
            "positive uint32 hidden")
    role_values = intermediate * hidden
    total_values = 3 * role_values
    require(total_values <= MAX_WEIGHTS, "bounded total weight count")
    coarse_bits_numerator = COARSE_RATE_NUMERATOR * total_values
    require(coarse_bits_numerator % COARSE_RATE_DENOMINATOR == 0,
            "shape has nonintegral 307/128-bpw coarse bit length")
    coarse_bits = coarse_bits_numerator // COARSE_RATE_DENOMINATOR
    require(coarse_bits % 8 == 0,
            "shape has nonintegral 307/128-bpw coarse byte length")
    blocks = (role_values + BLOCK_VALUES - 1) // BLOCK_VALUES
    last_valid = role_values - (blocks - 1) * BLOCK_VALUES
    require(1 <= last_valid <= BLOCK_VALUES, "last-block valid values")
    tail = 0 if last_valid == BLOCK_VALUES else BLOCK_VALUES - last_valid
    return ShapeContract(
        intermediate=intermediate,
        hidden=hidden,
        role_values=role_values,
        total_values=total_values,
        blocks_per_role=blocks,
        last_block_valid_values=last_valid,
        tail_values_per_role=tail,
        coarse_bytes=coarse_bits // 8,
    )


def physical_ledger(shape: ShapeContract, *, header_bytes: int = 512,
                    fine_packet_bytes: int = 48, page_bytes: int = 4096) -> dict[str, object]:
    require(all(type(value) is int and value > 0
                for value in (header_bytes, fine_packet_bytes, page_bytes)), "ledger constants")
    fine_bytes = 3 * shape.blocks_per_role * fine_packet_bytes
    unpadded = header_bytes + shape.coarse_bytes + fine_bytes
    physical = math.ceil(unpadded / page_bytes) * page_bytes
    rate = 8.0 * physical / shape.total_values
    return {
        "coarse_bytes": shape.coarse_bytes,
        "fine_bytes": fine_bytes,
        "header_bytes": header_bytes,
        "page_padding_bytes": physical - unpadded,
        "physical_bytes": physical,
        "physical_rate_bpw": rate,
        "tail_free": shape.tail_free,
        "tail_values_per_role": shape.tail_values_per_role,
        "last_block_valid_values": shape.last_block_valid_values,
        "target_rate_eligible": 2.15 <= rate <= 2.5,
    }
