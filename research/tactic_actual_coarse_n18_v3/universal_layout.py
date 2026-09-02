#!/usr/bin/env python3
"""Universal rate-bounded expert partition and exact 4 KiB page ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from v3_common import (
    COARSE_BYTES_PER_MICRO,
    FINE_BYTES_PER_MICRO,
    MAX_DIMENSION,
    MAX_EXPERTS,
    MAX_MATRIX_VALUES,
    METADATA_BYTES_PER_MICRO,
    MICRO,
    MICROS_PER_N18,
    MIN_EXPLICIT_FRAME_BYTES,
    N18_COARSE_RESERVOIR_BYTES,
    PAGE_BYTES,
    checked_product,
    exact_positive_int,
    require,
)


ROLES = ("gate", "up", "down_transposed")


@dataclass(frozen=True)
class ExpertGeometry:
    intermediate: int
    hidden: int

    def __post_init__(self) -> None:
        exact_positive_int(self.intermediate, MAX_DIMENSION, "intermediate dimension")
        exact_positive_int(self.hidden, MAX_DIMENSION, "hidden dimension")
        checked_product(self.intermediate, self.hidden, MAX_MATRIX_VALUES, "role matrix values")

    @property
    def role_values(self) -> int:
        return self.intermediate * self.hidden

    @property
    def weights(self) -> int:
        return 3 * self.role_values


@dataclass(frozen=True)
class RolePartition:
    role: str
    values: int
    full_microblocks: int
    n18_groups: int
    residual_microblocks: int
    tail_values: int


@dataclass(frozen=True)
class ExpertPartition:
    geometry: ExpertGeometry
    roles: tuple[RolePartition, ...]
    full_microblocks: int
    tail_values: int
    full_coarse_bytes: int
    tail_coarse_bytes: int
    fine_bytes: int
    metadata_bytes: int
    tail_extra_bytes: int
    coded_budget_bytes: int
    explicit: bool
    physical_frame_bytes: int
    fallback: str

    @property
    def coarse_bytes(self) -> int:
        return self.full_coarse_bytes + self.tail_coarse_bytes

    @property
    def final_bytes(self) -> int:
        return (
            self.coarse_bytes
            + self.fine_bytes
            + self.metadata_bytes
            + self.tail_extra_bytes
        )


@dataclass(frozen=True)
class OwnerFrame:
    expert_ordinal: int
    partition: ExpertPartition
    byte_offset: int
    byte_length: int
    first_page: int | None
    last_page: int | None
    unique_pages: int
    unique_page_bytes: int
    read_amplification_numerator: int
    read_amplification_denominator: int

    @property
    def read_amplification(self) -> float:
        return self.read_amplification_numerator / self.read_amplification_denominator


@dataclass(frozen=True)
class PanelLayout:
    owners: tuple[OwnerFrame, ...]
    weights: int
    physical_bytes: int

    @property
    def physical_bpw(self) -> float:
        return 8 * self.physical_bytes / self.weights

    @property
    def maximum_read_amplification(self) -> float:
        return max(owner.read_amplification for owner in self.owners)


def partition_expert(geometry: ExpertGeometry) -> ExpertPartition:
    roles: list[RolePartition] = []
    total_full = 0
    total_tail = 0
    for role in ROLES:
        full, tail = divmod(geometry.role_values, MICRO)
        n18_groups, residual = divmod(full, MICROS_PER_N18)
        roles.append(RolePartition(role, geometry.role_values, full, n18_groups, residual, tail))
        total_full += full
        total_tail += tail

    full_coarse = total_full * COARSE_BYTES_PER_MICRO
    tail_coarse = (307 * total_tail) // 1024
    fine = total_full * FINE_BYTES_PER_MICRO
    metadata = total_full * METADATA_BYTES_PER_MICRO
    tail_total = (5 * total_tail) // 16
    require(tail_total >= tail_coarse, "tail total contains tail coarse stream")
    tail_extra = tail_total - tail_coarse
    budget = (5 * geometry.weights) // 16
    final = full_coarse + tail_coarse + fine + metadata + tail_extra
    require(final == budget, "exact expert 2.5-bpw byte budget decomposition")

    explicit = budget >= MIN_EXPLICIT_FRAME_BYTES
    physical = budget if explicit else 0
    fallback = (
        "EXPLICIT_N18_MICRO_DH384_AND_AGGREGATE_TAIL_V1"
        if explicit
        else "IMPLICIT_ZERO_EXPERT_NO_BYTES_NO_READ_V1"
    )
    return ExpertPartition(
        geometry,
        tuple(roles),
        total_full,
        total_tail,
        full_coarse,
        tail_coarse,
        fine,
        metadata,
        tail_extra,
        budget,
        explicit,
        physical,
        fallback,
    )


def _owner(ordinal: int, partition: ExpertPartition, offset: int) -> OwnerFrame:
    length = partition.physical_frame_bytes
    if length == 0:
        return OwnerFrame(ordinal, partition, offset, 0, None, None, 0, 0, 0, 1)
    first_page = offset // PAGE_BYTES
    last_page = (offset + length - 1) // PAGE_BYTES
    pages = last_page - first_page + 1
    page_bytes = pages * PAGE_BYTES
    require(page_bytes < 2 * length, "strict per-owner unique-page read cap")
    return OwnerFrame(
        ordinal,
        partition,
        offset,
        length,
        first_page,
        last_page,
        pages,
        page_bytes,
        page_bytes,
        length,
    )


def layout_panel(experts: Iterable[ExpertGeometry]) -> PanelLayout:
    geometries = tuple(experts)
    require(1 <= len(geometries) <= MAX_EXPERTS, "panel expert count")
    owners: list[OwnerFrame] = []
    offset = 0
    total_weights = 0
    for ordinal, geometry in enumerate(geometries):
        require(isinstance(geometry, ExpertGeometry), "panel expert geometry")
        partition = partition_expert(geometry)
        owner = _owner(ordinal, partition, offset)
        owners.append(owner)
        offset += owner.byte_length
        total_weights += geometry.weights
    require(16 * offset <= 5 * total_weights, "panel physical rate <=2.5 bpw")
    require(all(owner.read_amplification < 2.0 for owner in owners), "panel strict unique-page read")
    return PanelLayout(tuple(owners), total_weights, offset)


def qwen_evaluation_ledger() -> dict[str, Any]:
    expert = ExpertGeometry(768, 2048)
    partition = partition_expert(expert)
    panel = layout_panel([expert] * 6)
    require(partition.full_microblocks == 1152 and partition.tail_values == 0, "Qwen micro geometry")
    require(sum(role.n18_groups for role in partition.roles) == 18, "Qwen N18 streams")
    require(partition.full_coarse_bytes == 1_414_656, "Qwen coarse expert bytes")
    require(partition.fine_bytes == 55_296 and partition.metadata_bytes == 4_608, "Qwen DH384 handoff bytes")
    require(partition.final_bytes == 1_474_560, "Qwen final expert bytes")
    require(panel.physical_bytes == 8_847_360 and panel.physical_bpw == 2.5, "Qwen panel physical ledger")
    require(panel.maximum_read_amplification == 1.0, "Qwen exact page-aligned read")
    return {
        "coarse_bpw": 307 / 128,
        "final_bpw": panel.physical_bpw,
        "coarse_bytes_per_expert": partition.coarse_bytes,
        "final_bytes_per_expert": partition.final_bytes,
        "panel_bytes": panel.physical_bytes,
        "maximum_unique_page_read_amplification": panel.maximum_read_amplification,
        "n18_streams_per_expert": 18,
    }


qwen_evaluation_ledger()
