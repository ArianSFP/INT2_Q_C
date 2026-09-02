#!/usr/bin/env python3
"""Universal shape/role source-plan validation and source-first control protocol."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from n18_common import (
    MAX_EXPERTS,
    MAX_MATRIX_VALUES,
    ExpertGeometry,
    MatrixGeometry,
    ROLES,
    SOURCE_PLAN_SCHEMA,
    checked_nonnegative_int,
    is_sha256,
    panel_ledger,
    require,
    strict_json_loads,
)


STORED_ROLES = ("gate", "up", "down")
FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "model",
        "model_id",
        "vendor",
        "checkpoint",
        "checkpoint_id",
        "layer",
        "layer_id",
        "expert_id",
        "provenance",
        "ancestor",
        "base_model",
        "router",
        "activation",
        "url",
    }
)


@dataclass(frozen=True)
class MatrixInput:
    expert_ordinal: int
    stored_role: str
    canonical_geometry: MatrixGeometry
    stored_rows: int
    stored_columns: int
    absolute_path: str
    bytes: int
    sha256: str

    @property
    def canonical_role(self) -> str:
        return self.canonical_geometry.role

    @property
    def transpose_on_ingest(self) -> bool:
        return self.stored_role == "down"


@dataclass(frozen=True)
class ExpertInput:
    ordinal: int
    geometry: ExpertGeometry
    matrices: tuple[MatrixInput, ...]


@dataclass(frozen=True)
class SourcePlan:
    experts: tuple[ExpertInput, ...]

    @property
    def matrices(self) -> tuple[MatrixInput, ...]:
        return tuple(matrix for expert in self.experts for matrix in expert.matrices)

    @property
    def ledger(self) -> dict[str, Any]:
        return panel_ledger(expert.geometry for expert in self.experts)


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} exact keys")


def _forbidden_walk(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(str(key).lower() not in FORBIDDEN_IDENTITY_KEYS, f"forbidden identity field: {key}")
            _forbidden_walk(child)
    elif isinstance(value, list):
        for child in value:
            _forbidden_walk(child)


def parse_source_plan(raw: bytes) -> SourcePlan:
    value = strict_json_loads(raw)
    require(isinstance(value, dict), "source plan object")
    _forbidden_walk(value)
    _exact_keys(
        value,
        {"schema", "status", "experts", "control_protocol", "claim_boundary"},
        "source plan",
    )
    require(value["schema"] == SOURCE_PLAN_SCHEMA, "source plan schema")
    require(value["status"] == "AUTHENTICATED_INPUT_BINDINGS_NO_DECODER_IDENTITY", "source plan status")
    require(
        value["control_protocol"]
        == "SOURCE_ABSOLUTE_DH384_PILOT_BEFORE_ANY_GAUSSIAN_OR_STRUCTURE_CONTROL",
        "source/control order lock",
    )
    rows = value["experts"]
    require(isinstance(rows, list) and 1 <= len(rows) <= MAX_EXPERTS, "source plan expert count")
    experts: list[ExpertInput] = []
    for ordinal, row in enumerate(rows):
        require(isinstance(row, dict), "expert row object")
        _exact_keys(row, {"expert_ordinal", "intermediate", "hidden", "matrices"}, "expert row")
        require(type(row["expert_ordinal"]) is int and row["expert_ordinal"] == ordinal, "expert coordinate order")
        geometry = ExpertGeometry(row["intermediate"], row["hidden"])
        matrices = row["matrices"]
        require(isinstance(matrices, list) and len(matrices) == 3, "one Gate/Up/Down triplet")
        parsed: list[MatrixInput] = []
        for role_ordinal, matrix in enumerate(matrices):
            require(isinstance(matrix, dict), "matrix row object")
            _exact_keys(
                matrix,
                {"role", "stored_shape", "absolute_path", "bytes", "sha256"},
                "matrix row",
            )
            stored_role = STORED_ROLES[role_ordinal]
            require(matrix["role"] == stored_role, "Gate/Up/Down canonical source order")
            expected_stored = (
                [geometry.hidden, geometry.intermediate]
                if stored_role == "down"
                else [geometry.intermediate, geometry.hidden]
            )
            require(
                isinstance(matrix["stored_shape"], list)
                and len(matrix["stored_shape"]) == 2
                and all(type(dimension) is int for dimension in matrix["stored_shape"])
                and matrix["stored_shape"] == expected_stored,
                "stored role shape",
            )
            absolute_path = matrix["absolute_path"]
            require(isinstance(absolute_path, str) and absolute_path.startswith("/"), "absolute source path")
            require("/../" not in absolute_path and not absolute_path.endswith("/.."), "source path traversal")
            expected_bytes = 2 * geometry.intermediate * geometry.hidden
            require(type(matrix["bytes"]) is int and matrix["bytes"] == expected_bytes, "BF16 source bytes")
            require(is_sha256(matrix["sha256"]), "source SHA-256")
            canonical_role = ROLES[role_ordinal]
            parsed.append(
                MatrixInput(
                    ordinal,
                    stored_role,
                    MatrixGeometry(canonical_role, geometry.intermediate, geometry.hidden),
                    expected_stored[0],
                    expected_stored[1],
                    absolute_path,
                    expected_bytes,
                    matrix["sha256"],
                )
            )
        experts.append(ExpertInput(ordinal, geometry, tuple(parsed)))
    plan = SourcePlan(tuple(experts))
    require(len(plan.matrices) == 3 * len(plan.experts), "source matrix conservation")
    require(sum(matrix.bytes // 2 for matrix in plan.matrices) == sum(expert.geometry.values for expert in plan.experts), "source weight conservation")
    return plan


class EvaluationPhase(enum.Enum):
    SOURCE_DH384_PILOT = "source_dh384_pilot"
    STOP_DH384_ONLY = "stop_dh384_only"
    CONTROLS_ALLOWED = "controls_allowed"
    COMPLETE = "complete"


class SourceFirstProtocol:
    """State machine preventing controls from influencing the absolute source gate."""

    def __init__(self) -> None:
        self.phase = EvaluationPhase.SOURCE_DH384_PILOT
        self.absolute_gate: dict[str, bool] | None = None

    def record_source_gate(self, role_survival: dict[str, bool]) -> None:
        require(self.phase is EvaluationPhase.SOURCE_DH384_PILOT, "source gate phase")
        require(set(role_survival) == set(ROLES), "all three source role gates")
        require(all(type(value) is bool for value in role_survival.values()), "source gate booleans")
        self.absolute_gate = dict(role_survival)
        self.phase = (
            EvaluationPhase.CONTROLS_ALLOWED
            if any(role_survival.values())
            else EvaluationPhase.STOP_DH384_ONLY
        )

    def authorize_controls(self) -> None:
        require(self.phase is EvaluationPhase.CONTROLS_ALLOWED, "controls require source DH384 survival")

    def finish_controls(self) -> None:
        self.authorize_controls()
        self.phase = EvaluationPhase.COMPLETE

    @property
    def cage_is_killed(self) -> bool:
        # The best-of-64 span oracle dominates only frozen DH384. It does not
        # dominate graph lifting, posterior centroids, adaptive trees, or
        # syndrome refinements proposed under the broader CAGE architecture.
        return False


def control_seed(kind: str, geometry: MatrixGeometry, tile_ordinal: int, replicate: int) -> int:
    """Shape/role/coordinate-only control seed; implementation lives downstream."""
    import hashlib
    import struct

    require(kind in ("decoded_gaussian", "structure_destroyed"), "control kind")
    checked_nonnegative_int(tile_ordinal, geometry.streams - 1, "control tile")
    checked_nonnegative_int(replicate, 255, "control replicate")
    packet = (
        b"UNIPOLAR-N18-307-CONTROL-v2\0"
        + kind.encode("ascii")
        + struct.pack(
            "<BIIIB",
            ROLES.index(geometry.role),
            geometry.rows,
            geometry.columns,
            tile_ordinal,
            replicate,
        )
    )
    return int.from_bytes(hashlib.sha256(packet).digest()[:8], "little")
