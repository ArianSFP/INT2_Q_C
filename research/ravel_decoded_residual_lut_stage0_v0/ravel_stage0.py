#!/usr/bin/env python3
"""CuPy stage-0 gate for the RAVEL decoded nonlinear residual lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any


AUTHORIZATION = "OPEN_AUTHENTICATED_DECODED_PANEL_FOR_RAVEL_STAGE0_V0"
PLAN_SHA256 = "8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868"
PLAN_INTERNAL_SHA256 = "99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d"
HEADER_SHA256 = "3c16bcf308c0cfce2071be24bf612d202360510084540aa0b358938d8399a538"
DECODED_SHA256 = "af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0"
EXPERTS = 6
ROLES = 3
ROWS = 768
COLS = 2048
GROUPS_PER_EXPERT = 2304
GROUPS = EXPERTS * GROUPS_PER_EXPERT
PANEL_VALUES = GROUPS * COLS
VALUES_PER_EXPERT = GROUPS_PER_EXPERT * COLS
FIT_EXPERTS = (0, 2, 3, 5)
HOLDOUT_EXPERTS = (1, 4)
ROLE_NAMES = ("gate", "up", "down")
TABLE_ENTRIES = 3 * 4 * 32 * 4 * 4
PACKET_BYTES = 16_384
BASELINE_SSE = 500.39553685426534
BASELINE_ENERGY = 16192.89450885593
BASELINE_F = 0.9888693569009007
TARGET_F = 0.8
SIDE_BPW = 8.0 * PACKET_BYTES / PANEL_VALUES
PUBLISHED_COLD_AMP = 1.1694444444444445
CONSERVATIVE_COLD_AMP = PUBLISHED_COLD_AMP + PACKET_BYTES / (VALUES_PER_EXPERT * 2.5 / 8.0)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def bf16_to_f64(np: Any, path: Path, shape: tuple[int, int]) -> Any:
    raw = path.read_bytes()
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != shape[0] * shape[1]:
        raise RuntimeError(f"source geometry mismatch: {path}")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    result = values.astype(np.float64).reshape(shape)
    if not np.isfinite(result).all():
        raise RuntimeError(f"nonfinite source: {path}")
    return result


def matrix_pairs(np: Any, plan_dir: Path) -> tuple[list[tuple[int, int, str, Any, Any]], dict[str, str]]:
    plan_path = plan_dir / "plan.lock.json"
    header_path = plan_dir / "header.bin"
    decoded_path = plan_dir / "independent_audit" / "post_klt_canonical_groups.f64.bin"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise RuntimeError("plan file identity changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "strata_expert_affine_n20n21_plan_v1" or plan.get("lock_sha256") != PLAN_INTERNAL_SHA256:
        raise RuntimeError("plan schema or internal identity changed")
    if len(plan.get("sources", [])) != 18:
        raise RuntimeError("expected 18 source rows")
    if int(plan.get("experts", EXPERTS)) != EXPERTS:
        raise RuntimeError("expert count changed")
    if sha256_file(header_path) != HEADER_SHA256 or sha256_file(decoded_path) != DECODED_SHA256:
        raise RuntimeError("decoded baseline identity changed")
    header = header_path.read_bytes()
    if len(header) != 128:
        raise RuntimeError("header length changed")
    coefficients = struct.unpack_from("<12f", header, 32)
    decoded = np.memmap(decoded_path, dtype="<f8", mode="r", shape=(GROUPS, COLS))
    source_root = Path(plan["source_root"]).resolve(strict=True)
    result: list[tuple[int, int, str, Any, Any]] = []
    for expert in range(EXPERTS):
        base = expert * GROUPS_PER_EXPERT
        gate_hat = np.asarray(decoded[base:base + ROWS], dtype=np.float64)
        z0 = np.asarray(decoded[base + ROWS:base + 2 * ROWS], dtype=np.float64)
        z1 = np.asarray(decoded[base + 2 * ROWS:base + 3 * ROWS], dtype=np.float64)
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        reconstructions = (gate_hat, (cosine * z0 - sine * z1) / norm2, (sine * z0 + cosine * z1) / norm2)
        for role_index, reconstruction in enumerate(reconstructions):
            ordinal = 3 * expert + role_index
            row = plan["sources"][ordinal]
            role = str(row["role"])
            if int(row["matrix_ordinal"]) != ordinal or role != ROLE_NAMES[role_index]:
                raise RuntimeError(f"source order changed at {ordinal}")
            source_path = (source_root / str(row["source_relpath"])).resolve(strict=True)
            source_path.relative_to(source_root)
            if source_path.is_symlink() or sha256_file(source_path) != str(row["source_bf16_sha256"]):
                raise RuntimeError(f"source identity changed at {ordinal}")
            source = bf16_to_f64(np, source_path, tuple(int(x) for x in row["shape"]))
            natural = source.T.copy() if role == "down" else source.copy()
            if natural.shape != (ROWS, COLS) or reconstruction.shape != (ROWS, COLS):
                raise RuntimeError(f"natural geometry changed at {ordinal}")
            result.append((expert, role_index, role, reconstruction, natural))
    return result, {
        "plan_sha256": sha256_file(plan_path),
        "plan_internal_sha256": str(plan["lock_sha256"]),
        "header_sha256": sha256_file(header_path),
        "decoded_sha256": sha256_file(decoded_path),
    }


def features(cp: Any, reconstruction: Any, role_index: int) -> tuple[Any, Any]:
    row_rms = cp.sqrt(cp.mean(reconstruction * reconstruction, axis=1, keepdims=True, dtype=cp.float64))
    safe = cp.maximum(row_rms, cp.float64(1.0e-30))
    normalized = reconstruction / safe
    amplitude = cp.floor(cp.clip((normalized + 4.0) * 4.0, 0.0, 31.0)).astype(cp.int32)
    left = cp.roll(reconstruction, 1, axis=1)
    right = cp.roll(reconstruction, -1, axis=1)
    left_state = ((left >= 0.0).astype(cp.int32) << 1) | (cp.abs(left) > cp.abs(reconstruction)).astype(cp.int32)
    right_state = ((right >= 0.0).astype(cp.int32) << 1) | (cp.abs(right) > cp.abs(reconstruction)).astype(cp.int32)
    matrix_rms = cp.sqrt(cp.mean(reconstruction * reconstruction, dtype=cp.float64))
    log_ratio = cp.log2(safe / cp.maximum(matrix_rms, cp.float64(1.0e-30)))
    row_class = (log_ratio > -0.25).astype(cp.int32) + (log_ratio > 0.0).astype(cp.int32) + (log_ratio > 0.25).astype(cp.int32)
    row_class = cp.broadcast_to(row_class, reconstruction.shape)
    index = (((role_index * 4 + row_class) * 32 + amplitude) * 4 + left_state) * 4 + right_state
    return index.reshape(-1), cp.broadcast_to(safe, reconstruction.shape).reshape(-1)


def accumulate(cp: Any, index: Any, normalized_error: Any, sums: Any, counts: Any) -> None:
    sums += cp.bincount(index, weights=normalized_error, minlength=TABLE_ENTRIES).astype(cp.float64)
    counts += cp.bincount(index, minlength=TABLE_ENTRIES).astype(cp.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.authorization != AUTHORIZATION:
        raise SystemExit("authorization mismatch; no payload opened")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 0; no payload opened")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("output must be absent; no payload opened")
    plan_dir = args.plan_dir.resolve(strict=True)

    import cupy as cp
    import numpy as np

    started = time.monotonic()
    pairs, bindings = matrix_pairs(np, plan_dir)
    fit_sum = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    fit_count = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    oracle_sum = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    oracle_count = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    baseline_sse = 0.0
    baseline_energy = 0.0
    holdout_sse = 0.0
    holdout_energy = 0.0
    cached_holdout: list[tuple[int, int, str, Any, Any, Any]] = []

    for expert, role_index, role, reconstruction_host, source_host in pairs:
        x = cp.asarray(reconstruction_host, dtype=cp.float64)
        y = cp.asarray(source_host, dtype=cp.float64)
        residual = y - x
        raw_sse = float(cp.sum(residual * residual, dtype=cp.float64).item())
        energy = float(cp.sum(y * y, dtype=cp.float64).item())
        baseline_sse += raw_sse
        baseline_energy += energy
        index, scale = features(cp, x, role_index)
        normalized_error = residual.reshape(-1) / scale
        if expert in FIT_EXPERTS:
            accumulate(cp, index, normalized_error, fit_sum, fit_count)
        else:
            holdout_sse += raw_sse
            holdout_energy += energy
            accumulate(cp, index, normalized_error, oracle_sum, oracle_count)
            cached_holdout.append((expert, role_index, role, index, scale, residual.reshape(-1)))

    if not math.isclose(baseline_sse, BASELINE_SSE, rel_tol=0.0, abs_tol=2.0e-9):
        raise RuntimeError(f"baseline SSE mismatch: {baseline_sse}")
    if not math.isclose(baseline_energy, BASELINE_ENERGY, rel_tol=0.0, abs_tol=2.0e-9):
        raise RuntimeError(f"baseline energy mismatch: {baseline_energy}")

    fit_table = cp.where(fit_count > 0.0, fit_sum / cp.maximum(fit_count, 1.0), 0.0)
    fit_table_fp16_host = cp.asnumpy(fit_table).astype("<f2")
    fit_table_fp16 = cp.asarray(fit_table_fp16_host.astype(np.float64))
    oracle_table = cp.where(oracle_count > 0.0, oracle_sum / cp.maximum(oracle_count, 1.0), 0.0)
    finite_sse = 0.0
    oracle_sse = 0.0
    matrix_rows = []
    for expert, role_index, role, index, scale, residual in cached_holdout:
        finite_error = residual - fit_table_fp16[index] * scale
        oracle_error = residual - oracle_table[index] * scale
        finite_value = float(cp.sum(finite_error * finite_error, dtype=cp.float64).item())
        oracle_value = float(cp.sum(oracle_error * oracle_error, dtype=cp.float64).item())
        raw_value = float(cp.sum(residual * residual, dtype=cp.float64).item())
        finite_sse += finite_value
        oracle_sse += oracle_value
        matrix_rows.append({
            "expert_ordinal": expert,
            "role": role,
            "baseline_sse": raw_value,
            "fit_fp16_table_sse": finite_value,
            "holdout_self_fit_fp64_oracle_sse": oracle_value,
        })

    packet_header = canonical_json({
        "format": "RAVEL6144-v0",
        "entries": TABLE_ENTRIES,
        "dtype": "<f2",
        "features": [3, 4, 32, 4, 4],
    }) + b"\n"
    table_bytes = fit_table_fp16_host.tobytes(order="C")
    if len(packet_header) + len(table_bytes) > PACKET_BYTES:
        raise RuntimeError("table packet overflow")
    packet = packet_header + table_bytes + bytes(PACKET_BYTES - len(packet_header) - len(table_bytes))

    holdout_f0 = holdout_sse / holdout_energy * 32.0
    finite_ratio = finite_sse / holdout_sse
    oracle_ratio = oracle_sse / holdout_sse
    finite_f = holdout_f0 * finite_ratio * 2.0 ** (2.0 * SIDE_BPW)
    oracle_f = holdout_f0 * oracle_ratio * 2.0 ** (2.0 * SIDE_BPW)
    status = "HARD_KILL_RAVEL6144" if oracle_f > TARGET_F else "PROMOTE_TO_CONTROLS_AND_REDUCED_RATE_REENCODE_ONLY"
    result = {
        "schema": "ravel-decoded-residual-lut-stage0-result-v0",
        "status": status,
        "claim_boundary": "Frozen 6,144-cell decoder-visible lookup only; favorable transfer and source-leaking oracle are not a finite reduced-rate codec.",
        "split": {"fit_experts": list(FIT_EXPERTS), "holdout_experts": list(HOLDOUT_EXPERTS)},
        "geometry": {"table_entries": TABLE_ENTRIES, "packet_bytes": PACKET_BYTES, "features": [3, 4, 32, 4, 4]},
        "baseline": {"panel_sse": baseline_sse, "panel_energy": baseline_energy, "holdout_sse": holdout_sse, "holdout_energy": holdout_energy, "holdout_F_at_2p5": holdout_f0},
        "finite_fit_table": {"sse": finite_sse, "fraction_of_baseline_sse": finite_ratio, "favorable_transfer_F": finite_f},
        "source_leaking_oracle": {"sse": oracle_sse, "fraction_of_baseline_sse": oracle_ratio, "capture": 1.0 - oracle_ratio, "favorable_transfer_F": oracle_f, "emitted": False},
        "rate_read": {"side_bpw": SIDE_BPW, "base_payload_cap_bpw": 2.5 - SIDE_BPW, "published_cold_amp": PUBLISHED_COLD_AMP, "conservative_cold_amp": CONSERVATIVE_COLD_AMP, "below_2x": CONSERVATIVE_COLD_AMP < 2.0},
        "matrix_rows": matrix_rows,
        "bindings": {**bindings, "script_sha256": sha256_file(Path(__file__).resolve()), "packet_sha256": hashlib.sha256(packet).hexdigest()},
        "runtime": {"elapsed_seconds": time.monotonic() - started, "python": sys.version, "numpy": np.__version__, "cupy": cp.__version__, "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"])},
    }
    result["result_lock_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "fit_table_packet.bin", packet)
    write_new(output / "result.json", json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    print(json.dumps({"status": status, "oracle_F": oracle_f, "finite_F": finite_f, "result": str(output / "result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
