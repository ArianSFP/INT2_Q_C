#!/usr/bin/env python3
"""Run the frozen N=4096 signal and matched-Gaussian mechanism fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

from codec import (Geometry, active_features, decode_packet, descriptor_formula,
                   indices_to_planes, planes_to_indices, sha256)
from search import search_bank


def synthetic_source(geometry: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, features = active_features(geometry, 0)
    z = features
    planes = np.empty((6, geometry.count), dtype=np.uint8)
    planes[0] = z[:, 0] ^ z[:, 3] ^ z[:, 7]
    planes[1] = z[:, 1] ^ (z[:, 4] & z[:, 8])
    planes[2] = z[:, 2] ^ z[:, 5] ^ (z[:, 9] & z[:, 10])
    planes[3] = z[:, 6] ^ z[:, 11] ^ (z[:, 0] & z[:, 4])
    planes[4] = z[:, 1] ^ z[:, 8] ^ (z[:, 2] & z[:, 7])
    planes[5] = z[:, 3] ^ z[:, 9] ^ (z[:, 5] & z[:, 10])
    indices = planes_to_indices(planes)
    eta = np.float64(0.0625)
    levels = eta * (-31.0 + np.arange(64, dtype=np.float64))
    phase = np.arange(geometry.count, dtype=np.float64)
    source = levels[indices] + eta * 0.14 * np.sin(phase * 0.6180339887498949)
    distortion = (source[:, None] - levels[None, :]) ** 2
    return source, levels, distortion


def summarize(run: dict, distortion: np.ndarray) -> dict:
    winner = run["winner"]
    decoded = decode_packet(winner["packet"])
    nearest = np.argmin(distortion, axis=1).astype(np.uint8)
    return {
        "winner_family": winner["family"],
        "winner_order_id": winner["order_id"],
        "winner_packet_sha256": decoded["packet_sha256"],
        "packet_bytes": decoded["packet_bytes"],
        "physical_rate_bpw": decoded["physical_rate_bpw"],
        "sse": winner["sse"],
        "nearest_sse": float(distortion[np.arange(nearest.size), nearest].sum()),
        "objective": winner["objective"],
        "exception_count": len(decoded["exceptions"]),
        "completed_plane_sha256": decoded["plane_sha256"],
        "index_sha256": decoded["index_sha256"],
        "all_64_indices_legal": int(decoded["indices"].min()) >= 0 and
                                int(decoded["indices"].max()) < 64,
        "six_completed_planes": list(decoded["completed_planes"].shape) ==
                                [6, 4096],
        "plane_assembly_roundtrip": bool(np.array_equal(
            decoded["completed_planes"], indices_to_planes(decoded["indices"]))),
        "canonical_reencode": True,
        "descriptor": descriptor_formula(decoded),
        "search_evaluations": run["search_evaluations"],
        "candidate_count": len(run["candidates"]),
    }


def main() -> None:
    # The tile is in the third 256-row radix sector and uses role trit 2.
    geometry = Geometry(rows=768, cols=2048, role=2, row_start=512,
                        row_count=16, col_start=512, col_count=256)
    source, levels, distortion = synthetic_source(geometry)
    lambda_bit = 2.5e-5
    structured = search_bank(distortion, geometry, lambda_bit)

    rng = np.random.default_rng(0x5A17A6)
    gaussian = rng.standard_normal(geometry.count).astype(np.float64)
    gaussian = (gaussian - gaussian.mean()) / gaussian.std()
    control_source = (gaussian * source.std() + source.mean()).astype(np.float64)
    if not (abs(float(control_source.mean() - source.mean())) <= 1e-14 and
            abs(float(control_source.std() - source.std())) <= 1e-14):
        raise RuntimeError("matched Gaussian moment contract")
    control_distortion = (control_source[:, None] - levels[None, :]) ** 2
    control = search_bank(control_distortion, geometry, lambda_bit)
    role_receipts = {}
    for role, row_start in enumerate((0, 256, 512)):
        role_geometry = Geometry(768, 2048, role, row_start, 16, 512, 256)
        names, bits = active_features(role_geometry, 0)
        role_receipts[str(role)] = {
            "row_trit": row_start // 256,
            "active_feature_names": names,
            "feature_matrix_sha256": sha256(np.ascontiguousarray(bits).tobytes()),
        }
    result = {
        "schema": "strata-bmp-obdd-qtt6-source-free-fixture-v0",
        "status": "PASS_SOURCE_FREE_MECHANISM_FIXTURE__HOLD_PAYLOAD",
        "geometry": geometry.__dict__,
        "weights": geometry.count,
        "distortion_table_shape": list(distortion.shape),
        "distortion_table_exact_64_way": distortion.shape == (4096, 64),
        "source_sha256": sha256(np.ascontiguousarray(source).tobytes()),
        "levels_sha256": sha256(np.ascontiguousarray(levels).tobytes()),
        "lambda_per_physical_bit": lambda_bit,
        "structured": summarize(structured, distortion),
        "matched_gaussian_control": summarize(control, control_distortion),
        "identical_search_bank_for_source_and_control": True,
        "mixed_radix_role_receipts": role_receipts,
        "caps": structured["caps"],
        "model_or_qwen_payload_accessed": False,
        "current_strata_or_coarse_artifact_accessed": False,
        "network_accessed": False,
        "claim_boundary": (
            "Mechanism fixture only. Neither synthetic/control winner is Qwen "
            "evidence, an F<=0.8 result, or a current STRATA packet recoding."),
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
