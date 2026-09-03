"""Cheap group-size-2 survivor gate for the r3 source-free fixture."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import clustered_ib_core as core
from source_free_fixture import (
    BLOCKS_PER_ROLE,
    COORDINATES,
    EXPERT_COUNT,
    FOLD_COUNT,
    ROLES,
    SCALE_BYTES_PER_VALUE,
    SCALE_BYTES_PER_EXPERT,
    SUPERBLOCK_VALUES,
    make_production_geometry_survivor_fixture,
)


def main() -> int:
    expected_scale_bytes = (
        ROLES * (COORDINATES // SUPERBLOCK_VALUES) * SCALE_BYTES_PER_VALUE
    )
    if BLOCKS_PER_ROLE != COORDINATES // SUPERBLOCK_VALUES:
        raise RuntimeError("fixture block geometry is inconsistent")
    if SCALE_BYTES_PER_EXPERT != expected_scale_bytes:
        raise RuntimeError("fixture scale bytes do not match exact geometry")
    labels = make_production_geometry_survivor_fixture()
    score = core.crossfit_group_size(
        labels, 2, fold_count=FOLD_COUNT,
        superblock_values=SUPERBLOCK_VALUES,
    )
    requirements = core.packet_requirements(score, SCALE_BYTES_PER_EXPERT)
    envelopes = {
        str(rate): core.physical_read_envelope(
            expert_count=EXPERT_COUNT,
            weights_per_expert=2 * COORDINATES,
            requested_rate=rate,
            **requirements,
        )
        for rate in core.RATE_ENDPOINTS
    }
    feasible = [
        rate for rate, envelope in envelopes.items()
        if envelope.get("status") == "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC"
        and envelope.get("capacity_ok") is True
        and envelope.get("strictly_below_2x") is True
    ]
    gain = float(score["favorable_gross_gain_bpw"])
    if not math.isfinite(gain) or gain < core.TARGET_GAIN_BPW:
        raise RuntimeError("balanced fixture misses favorable source threshold")
    if not feasible:
        raise RuntimeError("balanced fixture has no exact strict-read endpoint")
    result = {
        "schema": "same-layer-clustered-ib-r3-fixture-survivor-regression-v0",
        "status": "PASS_TARGETED_GROUP2_SOURCE_AND_STRICT_READ_SURVIVOR",
        "payload_or_qwen_accessed": False,
        "experts": EXPERT_COUNT,
        "roles": ROLES,
        "coordinates_per_role": COORDINATES,
        "fold_count": FOLD_COUNT,
        "superblock_values": SUPERBLOCK_VALUES,
        "scale_bytes_per_expert": SCALE_BYTES_PER_EXPERT,
        "fixture_labels_sha256": hashlib.sha256(
            labels.tobytes(order="C")
        ).hexdigest(),
        "group_size": 2,
        "target_gain_bpw": core.TARGET_GAIN_BPW,
        "favorable_gross_gain_bpw": gain,
        "charged_gain_bpw": float(score["charged_gain_bpw"]),
        "feasible_rate_endpoints": feasible,
        "read_envelopes": envelopes,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
