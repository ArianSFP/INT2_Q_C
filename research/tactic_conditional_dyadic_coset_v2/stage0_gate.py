#!/usr/bin/env python3
"""Future authenticated CuPy span gate for TACTIC-DH384 v2.

The source package gives this file no launch authority.  The exact literal
authorization, a separately audited actual-coarse lock, an absent held output,
and CUDA_VISIBLE_DEVICES=0 are all mandatory before any record payload opens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from tactic_v2_common import (
    COARSE_LOCK_SCHEMA,
    PAYLOAD_AUTHORIZATION,
    ContractError,
    HeldAbsolute,
    HeldOutput,
    HeldRoot,
    canonical_json,
    read_bounded_stable_fd,
    read_stable_fd,
    require,
    selector_packet,
    sha256_bytes,
    strict_json_loads,
    universal_selector_table,
)


EXPERTS = 6
ROLES = ("gate", "up", "down_transposed")
ROWS = 768
COLS = 2048
VALUES_MATRIX = ROWS * COLS
VALUES_EXPERT = 3 * VALUES_MATRIX
VALUES_TOTAL = EXPERTS * VALUES_EXPERT
DECISION_EXPERTS = tuple(range(EXPERTS))
BLOCK = 4096
RANK = 384
STAGES = 12
BASE_BPW = 307.0 / 128.0
TOTAL_BPW = 2.5
TARGET_D = 0.025
EXPECTED_COARSE_BYTES = 6 * 18 * 78_592
AUTHORIZATION = PAYLOAD_AUTHORIZATION


def _internal_lock(value: dict[str, Any]) -> str:
    clone = dict(value)
    declared = clone.pop("lock_sha256", None)
    require(isinstance(declared, str) and len(declared) == 64, "coarse lock internal seal missing")
    observed = hashlib.sha256(canonical_json(clone)).hexdigest()
    require(observed == declared, "coarse lock internal seal mismatch")
    return observed


def _validate_file_descriptor(row: Any, dtype: str, expected_bytes: int) -> dict[str, Any]:
    require(isinstance(row, dict), "record file descriptor is not an object")
    require(set(row) == {"relpath", "bytes", "sha256", "dtype", "shape"},
            "record file descriptor schema")
    require(row["dtype"] == dtype, "record dtype mismatch")
    require(row["shape"] == [ROWS, COLS], "record canonical shape mismatch")
    require(int(row["bytes"]) == expected_bytes, "record byte count mismatch")
    require(isinstance(row["relpath"], str) and row["relpath"], "record relpath")
    digest = row["sha256"]
    require(isinstance(digest, str) and len(digest) == 64 and digest == digest.lower(),
            "record SHA-256 syntax")
    return row


def _validate_receipt_descriptor(row: Any) -> dict[str, Any]:
    require(isinstance(row, dict) and set(row) == {"relpath", "bytes", "sha256", "dtype"},
            "round-trip receipt descriptor schema")
    require(row["dtype"] == "strict-json-receipt", "round-trip receipt dtype")
    require(isinstance(row["relpath"], str) and row["relpath"], "round-trip receipt relpath")
    require(isinstance(row["bytes"], int) and row["bytes"] > 0, "round-trip receipt bytes")
    digest = row["sha256"]
    require(isinstance(digest, str) and len(digest) == 64 and digest == digest.lower(),
            "round-trip receipt SHA-256 syntax")
    return row


def _validate_reservoir_descriptor(row: Any, ordinal: int) -> dict[str, Any]:
    require(isinstance(row, dict) and set(row) == {
        "stream_ordinal", "relpath", "bytes", "sha256", "dtype"
    }, "reservoir descriptor schema")
    require(int(row["stream_ordinal"]) == ordinal, "reservoir canonical order")
    require(row["dtype"] == "opaque-coarse-stream", "reservoir dtype")
    require(int(row["bytes"]) == 78_592, "reservoir fixed byte count")
    require(isinstance(row["relpath"], str) and row["relpath"], "reservoir relpath")
    digest = row["sha256"]
    require(isinstance(digest, str) and len(digest) == 64 and digest == digest.lower(),
            "reservoir SHA-256 syntax")
    return row


def _validate_rate(rate: Any) -> dict[str, Any]:
    require(isinstance(rate, dict) and set(rate) == {
        "actual_bpw", "streams", "bytes_per_stream", "coarse_container_bytes",
        "decode_reencode_verified", "all_stream_reservoirs_within_capacity",
        "roundtrip_receipt"
    }, "panel rate receipt schema")
    require(float(rate["actual_bpw"]) == BASE_BPW, "coarse actual rate")
    require(int(rate["streams"]) == 108, "coarse stream count")
    require(int(rate["bytes_per_stream"]) == 78_592, "coarse reservoir bytes")
    require(int(rate["coarse_container_bytes"]) == EXPECTED_COARSE_BYTES,
            "coarse physical bytes")
    require(rate["decode_reencode_verified"] is True, "coarse round-trip absent")
    require(rate["all_stream_reservoirs_within_capacity"] is True, "coarse overflow")
    _validate_receipt_descriptor(rate["roundtrip_receipt"])
    return rate


def validate_coarse_lock(lock: dict[str, Any]) -> dict[str, Any]:
    require(lock.get("schema") == COARSE_LOCK_SCHEMA, "coarse lock schema")
    require(set(lock) == {"schema", "root", "panels", "lock_sha256"},
            "coarse lock top-level schema")
    _internal_lock(lock)
    require(isinstance(lock.get("root"), str) and os.path.isabs(lock["root"]), "coarse root")
    panels = lock.get("panels")
    require(isinstance(panels, list) and len(panels) == 3, "three coarse panels required")
    require(all(isinstance(panel, dict) for panel in panels), "coarse panel object")
    require([panel.get("kind") for panel in panels] == [
        "source", "decoded_gaussian", "structure_destroyed"
    ], "source/control panel order")
    panel_ids: set[str] = set()
    for panel in panels:
        require(isinstance(panel, dict) and set(panel) == {
            "id", "kind", "rate", "reservoirs", "records"
        },
                "panel schema")
        panel_id = panel["id"]
        require(isinstance(panel_id, str) and panel_id not in panel_ids, "panel id")
        panel_ids.add(panel_id)
        require(panel["kind"] in ("source", "decoded_gaussian", "structure_destroyed"),
                "panel kind")
        _validate_rate(panel["rate"])
        reservoirs = panel["reservoirs"]
        require(isinstance(reservoirs, list) and len(reservoirs) == 108,
                "panel reservoir count")
        for stream_ordinal, descriptor in enumerate(reservoirs):
            _validate_reservoir_descriptor(descriptor, stream_ordinal)
        records = panel["records"]
        require(isinstance(records, list) and len(records) == 18, "panel record count")
        identities: set[tuple[int, str]] = set()
        for ordinal, record in enumerate(records):
            require(isinstance(record, dict) and set(record) == {
                "matrix_ordinal", "expert_ordinal", "role", "source", "reconstruction", "symbols"
            }, "record schema")
            expert = int(record["expert_ordinal"])
            role = record["role"]
            require(int(record["matrix_ordinal"]) == ordinal, "matrix ordinal")
            require(0 <= expert < EXPERTS and role in ROLES, "record identity")
            require((expert, role) not in identities, "duplicate record identity")
            identities.add((expert, role))
            require(ordinal == 3 * expert + ROLES.index(role), "record canonical order")
            _validate_file_descriptor(record["source"], "<bf16", 2 * VALUES_MATRIX)
            _validate_file_descriptor(record["reconstruction"], "<f4", 4 * VALUES_MATRIX)
            _validate_file_descriptor(record["symbols"], "<i2", 2 * VALUES_MATRIX)
        require(len(identities) == 18, "incomplete panel identities")
    return lock


def _open_panel(root: HeldRoot, panel: dict[str, Any]) -> tuple[list[dict[str, Any]], bytes]:
    receipt_descriptor = panel["rate"]["roundtrip_receipt"]
    receipt_fd = root.open_relative(receipt_descriptor["relpath"])
    receipt_bytes = read_stable_fd(
        receipt_fd, int(receipt_descriptor["bytes"]), receipt_descriptor["sha256"]
    )
    receipt = strict_json_loads(receipt_bytes)
    require(receipt.get("status") == "PASS_ACTUAL_COARSE_DECODE_REENCODE",
            "coarse round-trip receipt status")
    require(receipt.get("actual_bpw") == BASE_BPW, "coarse receipt rate")
    require(receipt.get("coarse_container_bytes") == EXPECTED_COARSE_BYTES,
            "coarse receipt bytes")
    for descriptor in panel["reservoirs"]:
        fd = root.open_relative(descriptor["relpath"])
        read_stable_fd(fd, int(descriptor["bytes"]), descriptor["sha256"])
    opened: list[dict[str, Any]] = []
    for record in panel["records"]:
        item = {
            "matrix_ordinal": int(record["matrix_ordinal"]),
            "expert_ordinal": int(record["expert_ordinal"]),
            "role": record["role"],
        }
        for field in ("source", "reconstruction", "symbols"):
            descriptor = record[field]
            fd = root.open_relative(descriptor["relpath"])
            item[field] = read_stable_fd(fd, int(descriptor["bytes"]), descriptor["sha256"])
        opened.append(item)
    return opened, receipt_bytes


def _decode_arrays(np: Any, record: dict[str, Any]) -> tuple[Any, Any, Any]:
    words = np.frombuffer(record["source"], dtype="<u2")
    source = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    reconstruction = np.frombuffer(record["reconstruction"], dtype="<f4").astype(np.float64)
    symbols = np.frombuffer(record["symbols"], dtype="<i2").astype(np.int64)
    require(source.size == reconstruction.size == symbols.size == VALUES_MATRIX,
            "decoded record geometry")
    require(np.isfinite(source).all() and np.isfinite(reconstruction).all(),
            "non-finite decoded record")
    return source, reconstruction, symbols


def gpu_projection(cp: Any, symbols_host: Any, error_host: Any, role: int, table: bytes) -> tuple[float, float]:
    """Return (error energy, exact continuous rank-384 projected energy)."""
    symbol = cp.asarray(symbols_host, dtype=cp.int64).reshape(-1, BLOCK)
    error = cp.asarray(error_host, dtype=cp.float64).reshape(-1, BLOCK)
    table_gpu = cp.asarray(bytearray(table), dtype=cp.uint8).reshape(STAGES, 256)
    mean_abs = cp.sum(cp.abs(symbol), axis=1, dtype=cp.int64) // cp.int64(BLOCK)
    shadow = symbol
    schedules = []
    for stage in range(STAGES):
        stride = 1 << stage
        paired = shadow.reshape(shadow.shape[0], -1, 2, stride)
        u, v = paired[:, :, 0, :], paired[:, :, 1, :]
        au, av = cp.abs(u), cp.abs(v)
        threshold = mean_abs[:, None, None]
        feature = (
            cp.int64(role << 6)
            | ((u < 0).astype(cp.int64) << cp.int64(5))
            | ((v < 0).astype(cp.int64) << cp.int64(4))
            | ((au > av).astype(cp.int64) << cp.int64(3))
            | (((au + av) > 2 * threshold).astype(cp.int64) << cp.int64(2))
            | ((au > threshold).astype(cp.int64) << cp.int64(1))
            | (av > threshold).astype(cp.int64)
        )
        op = table_gpu[stage, feature]
        swap = (op & cp.uint8(1)) != 0
        a = cp.where(swap, v, u)
        b = cp.where(swap, u, v)
        a = cp.where((op & cp.uint8(2)) != 0, -a, a)
        b = cp.where((op & cp.uint8(4)) != 0, -b, b)
        shadow = cp.stack((a + b, a - b), axis=2).reshape(shadow.shape)
        schedules.append(op)

    transformed = error
    for stage in reversed(range(STAGES)):
        stride = 1 << stage
        paired = transformed.reshape(transformed.shape[0], -1, 2, stride)
        a, b = paired[:, :, 0, :], paired[:, :, 1, :]
        x0, x1 = a + b, a - b
        op = schedules[stage]
        x0 = cp.where((op & cp.uint8(2)) != 0, -x0, x0)
        x1 = cp.where((op & cp.uint8(4)) != 0, -x1, x1)
        swap = (op & cp.uint8(1)) != 0
        u = cp.where(swap, x1, x0)
        v = cp.where(swap, x0, x1)
        transformed = cp.stack((u, v), axis=2).reshape(transformed.shape)
    transformed = transformed / cp.float64(64.0)
    energy = float(cp.sum(error * error, dtype=cp.float64).item())
    transformed_energy = float(cp.sum(transformed * transformed, dtype=cp.float64).item())
    require(math.isclose(energy, transformed_energy, rel_tol=2e-11, abs_tol=2e-9),
            "CuPy conditional transform norm identity")
    projected = float(cp.sum(transformed[:, :RANK] ** 2, dtype=cp.float64).item())
    require(0.0 <= projected <= energy * (1.0 + 2e-12), "projected energy bounds")
    return energy, min(projected, energy)


def _jackknife_capture(expert_energy: dict[int, float], expert_projected: dict[int, float]) -> tuple[float, float]:
    experts = sorted(expert_energy)
    require(len(experts) >= 2, "jackknife needs at least two experts")
    total_e = math.fsum(expert_energy.values())
    total_p = math.fsum(expert_projected.values())
    capture = total_p / total_e
    leave = []
    for expert in experts:
        leave.append((total_p - expert_projected[expert]) / (total_e - expert_energy[expert]))
    mean = math.fsum(leave) / len(leave)
    se = math.sqrt((len(leave) - 1.0) / len(leave) * math.fsum((x - mean) ** 2 for x in leave))
    return capture, se


def evaluate_panel(cp: Any, np: Any, records: list[dict[str, Any]], panel_id: str) -> dict[str, Any]:
    decoded = []
    for record in records:
        source, reconstruction, symbols = _decode_arrays(np, record)
        error = source - reconstruction
        decoded.append((record, source, reconstruction, symbols, error))

    # This table is frozen source-independently.  No panel, residual, model,
    # layer, expert identity, or provenance value selects any frame parameter.
    universal_table = universal_selector_table()

    source_energy = 0.0
    coarse_sse = 0.0
    expert_energy: dict[int, float] = {expert: 0.0 for expert in DECISION_EXPERTS}
    expert_projected: dict[int, float] = {expert: 0.0 for expert in DECISION_EXPERTS}
    role_rows: dict[str, dict[str, float]] = {role: {"energy": 0.0, "projected": 0.0} for role in ROLES}
    rows = []
    for record, source, _reconstruction, symbols, error in decoded:
        expert = int(record["expert_ordinal"])
        if expert not in DECISION_EXPERTS:
            continue
        role = record["role"]
        value_e, value_p = gpu_projection(cp, symbols, error, ROLES.index(role), universal_table)
        value_source = float(np.sum(source * source, dtype=np.float64))
        source_energy += value_source
        coarse_sse += value_e
        expert_energy[expert] += value_e
        expert_projected[expert] += value_p
        role_rows[role]["energy"] += value_e
        role_rows[role]["projected"] += value_p
        rows.append({
            "matrix_ordinal": int(record["matrix_ordinal"]),
            "expert_ordinal": expert,
            "role": role,
            "source_energy": value_source,
            "coarse_sse": value_e,
            "projected_sse": value_p,
            "oracle_sse": value_e - value_p,
            "capture": value_p / value_e,
        })
    require(source_energy > 0.0 and coarse_sse > 0.0, "zero decision energy")
    capture, se = _jackknife_capture(expert_energy, expert_projected)
    d0 = coarse_sse / source_energy
    c_required = 1.0 - TARGET_D / d0
    upper, lower = capture + 3.0 * se, capture - 3.0 * se
    expert_captures = {str(e): expert_projected[e] / expert_energy[e] for e in DECISION_EXPERTS}
    role_captures = {role: row["projected"] / row["energy"] for role, row in role_rows.items()}
    folds_positive = all(value > 0.0 for value in expert_captures.values()) and all(
        value > 0.0 for value in role_captures.values()
    )
    oracle_sse = coarse_sse * (1.0 - capture)
    oracle_f = oracle_sse / source_energy * 2.0 ** (2.0 * TOTAL_BPW)
    if upper < c_required:
        decision = "HARD_REJECT_CONTINUOUS_SPAN_FAR_SHORT"
    elif lower >= c_required and folds_positive:
        decision = "PROMOTE_TO_LITERAL_FINITE_COSET_ONLY"
    else:
        decision = "HOLD_INCONCLUSIVE_NO_CELL_CHANGE"
    result = {
        "panel_id": panel_id,
        "universal_selector_ordinal": 17,
        "selector_searches": 0,
        "architecture_parameters_selected_from_panel": False,
        "decision_source_energy": source_energy,
        "decision_coarse_sse": coarse_sse,
        "measured_actual_lower_rate_D0": d0,
        "runtime_c_required": c_required,
        "capture": capture,
        "whole_expert_SE": se,
        "capture_upper_3SE": upper,
        "capture_lower_3SE": lower,
        "expert_captures": expert_captures,
        "role_captures": role_captures,
        "continuous_oracle_sse": oracle_sse,
        "continuous_oracle_F_at_2p5": oracle_f,
        "decision": decision,
        "matrix_rows": rows,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.authorization != AUTHORIZATION:
        raise SystemExit("authorization mismatch; no coarse lock or payload opened")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 0; no coarse lock or payload opened")
    if not args.coarse_lock.is_absolute() or not args.output.is_absolute():
        raise SystemExit("coarse lock and output must be absolute; no payload opened")
    if os.path.lexists(args.output):
        raise SystemExit("output must be absent; no payload opened")

    # Standard-library source closure is verified before the externally supplied lock.
    from verify_source import verify_package
    source_receipt = verify_package(Path(__file__).resolve().parent)

    started = time.monotonic()
    with HeldAbsolute(args.coarse_lock, want_directory=False) as held_lock:
        lock_bytes = read_bounded_stable_fd(held_lock.fd, 16 * 1024 * 1024)
        lock = validate_coarse_lock(strict_json_loads(lock_bytes))
        with HeldOutput(args.output) as output, HeldRoot(Path(lock["root"])) as root:
            source_records, source_receipt_bytes = _open_panel(root, lock["panels"][0])
            import cupy as cp
            import numpy as np
            source_result = evaluate_panel(cp, np, source_records, lock["panels"][0]["id"])
            panel_results = [source_result]
            receipt_hashes = {
                lock["panels"][0]["id"]: sha256_bytes(source_receipt_bytes)
            }
            if source_result["decision"] != "HARD_REJECT_CONTINUOUS_SPAN_FAR_SHORT":
                for panel in lock["panels"][1:]:
                    control_records, control_receipt_bytes = _open_panel(root, panel)
                    control_result = evaluate_panel(cp, np, control_records, panel["id"])
                    panel_results.append(control_result)
                    receipt_hashes[panel["id"]] = sha256_bytes(control_receipt_bytes)

            universal_packet = selector_packet(universal_selector_table())

            result = {
                "schema": "tactic_dh384_stage0_result_v2",
                "status": source_result["decision"],
                "claim_boundary": "Continuous rank-384 span gate only; arbitrary real coefficients are not emitted and no finite target achievement is claimed.",
                "coarse_lock_sha256": sha256_bytes(lock_bytes),
                "coarse_lock_internal_sha256": lock["lock_sha256"],
                "actual_coarse_bpw": BASE_BPW,
                "physical_candidate_bpw": TOTAL_BPW,
                "container_bytes": 8_847_360,
                "cold_read_amplification": 73.0 / 72.0,
                "panels": panel_results,
                "controls_opened": len(panel_results) - 1,
                "runtime": {
                    "elapsed_seconds": time.monotonic() - started,
                    "python": sys.version,
                    "numpy": np.__version__,
                    "cupy": cp.__version__,
                    "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
                },
                "bindings": {
                    "source_manifest_sha256": source_receipt["manifest_sha256"],
                    "stage0_gate_sha256": source_receipt["file_hashes"]["stage0_gate.py"],
                    "universal_selector_packet_sha256": sha256_bytes(universal_packet),
                    "panel_roundtrip_receipt_sha256": receipt_hashes,
                },
            }
            result["result_lock_sha256"] = sha256_bytes(canonical_json(result))
            output.write_new("universal_selector_packet.bin", universal_packet)
            output.write_new(
                "result.json",
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n",
            )
            print(json.dumps({
                "status": result["status"],
                "source_oracle_F": source_result["continuous_oracle_F_at_2p5"],
                "result_lock_sha256": result["result_lock_sha256"],
            }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        raise SystemExit(f"TACTIC-DH384 contract failure: {exc}") from exc
