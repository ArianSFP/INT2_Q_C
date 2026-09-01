#!/usr/bin/env python3
"""Authenticated CuPy stage-0 gate for the repaired RAVEL-6144-v1 lookup."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


AUTHORIZATION = "OPEN_AUTHENTICATED_DECODED_PANEL_FOR_RAVEL_STAGE0_V1"
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
DECODED_BYTES = PANEL_VALUES * 8
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
SOURCE_FILES = {
    "MANIFEST.sha256", "README.md", "SOURCE_RECEIPT.json", "design_lock.json",
    "packet_codec.py", "ravel_stage0.py", "test_source_only.py", "verify_result.py",
    "verify_source.py",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def strict_json(raw: bytes) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in rows:
            if key in out:
                raise RuntimeError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def finite(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise RuntimeError("nonfinite JSON")
        return result

    def bad_constant(value: str) -> None:
        raise RuntimeError(f"nonfinite JSON constant: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=bad_constant)


def is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def reject_link_components(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        info = os.lstat(current)
        if is_link_or_reparse(info):
            raise RuntimeError(f"link/reparse path component rejected: {current}")
    return absolute


def read_regular_snapshot(path: Path, expected_sha256: str | None = None,
                          expected_bytes: int | None = None) -> bytes:
    path = reject_link_components(path)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or is_link_or_reparse(before):
        raise RuntimeError(f"not a regular input: {path}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("native O_NOFOLLOW support is required")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise RuntimeError(f"input identity changed before open: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"input changed while held: {path}")
        if len(raw) != opened.st_size:
            raise RuntimeError(f"short input read: {path}")
    finally:
        os.close(descriptor)
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise RuntimeError(f"input byte length changed: {path}")
    if expected_sha256 is not None and sha256(raw) != expected_sha256:
        raise RuntimeError(f"input hash changed: {path}")
    return raw


def parse_manifest(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n"):
        raise RuntimeError("source manifest lacks final LF")
    rows: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        pieces = line.split("  ")
        if (len(pieces) != 2 or len(pieces[0]) != 64 or
                any(char not in "0123456789abcdef" for char in pieces[0])):
            raise RuntimeError("malformed source manifest")
        name = pieces[1]
        if PurePosixPath(name).name != name or name in rows or name == "MANIFEST.sha256":
            raise RuntimeError("unsafe source manifest path")
        rows[name] = pieces[0]
    return rows


def self_authenticate() -> dict[str, Any]:
    package = reject_link_components(Path(__file__).parent)
    observed = {path.name for path in package.iterdir()}
    if observed != SOURCE_FILES:
        raise RuntimeError("source package closure changed")
    manifest_raw = read_regular_snapshot(package / "MANIFEST.sha256")
    manifest = parse_manifest(manifest_raw)
    if set(manifest) != SOURCE_FILES - {"MANIFEST.sha256"}:
        raise RuntimeError("source manifest member set changed")
    members: dict[str, str] = {}
    for name, expected in sorted(manifest.items()):
        raw = read_regular_snapshot(package / name, expected)
        members[name] = sha256(raw)
    return {"manifest_sha256": sha256(manifest_raw), "members": members}


def load_packet_codec() -> Any:
    path = Path(__file__).resolve().with_name("packet_codec.py")
    spec = importlib.util.spec_from_file_location("ravel_v1_packet_codec_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load authenticated packet codec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def bf16_snapshot_to_f64(np: Any, raw: bytes, shape: tuple[int, int]) -> Any:
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != shape[0] * shape[1]:
        raise RuntimeError("source geometry mismatch")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    result = values.astype(np.float64).reshape(shape)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite source snapshot")
    return result


def safe_source_path(source_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise RuntimeError(f"unsafe source relative path: {relative}")
    candidate = source_root.joinpath(*pure.parts)
    candidate = reject_link_components(candidate)
    candidate.relative_to(source_root)
    return candidate


def matrix_pairs(np: Any, plan_dir_arg: Path) -> tuple[list[tuple[int, int, str, Any, Any]], dict[str, Any]]:
    plan_dir = reject_link_components(plan_dir_arg)
    plan_raw = read_regular_snapshot(plan_dir / "plan.lock.json", PLAN_SHA256)
    plan = strict_json(plan_raw)
    if plan.get("schema") != "strata_expert_affine_n20n21_plan_v1" or plan.get("lock_sha256") != PLAN_INTERNAL_SHA256:
        raise RuntimeError("plan schema/internal identity changed")
    if len(plan.get("sources", [])) != 18 or int(plan.get("experts", EXPERTS)) != EXPERTS:
        raise RuntimeError("plan panel geometry changed")
    header_raw = read_regular_snapshot(plan_dir / "header.bin", HEADER_SHA256, 128)
    decoded_raw = read_regular_snapshot(
        plan_dir / "independent_audit" / "post_klt_canonical_groups.f64.bin",
        DECODED_SHA256, DECODED_BYTES,
    )
    coefficients = struct.unpack_from("<12f", header_raw, 32)
    decoded = np.frombuffer(decoded_raw, dtype="<f8").reshape((GROUPS, COLS))
    if not np.isfinite(decoded).all():
        raise RuntimeError("nonfinite decoded reconstruction snapshot")
    source_root_raw = Path(str(plan["source_root"]))
    if not source_root_raw.is_absolute():
        raise RuntimeError("source root must be absolute")
    source_root = reject_link_components(source_root_raw)
    result: list[tuple[int, int, str, Any, Any]] = []
    receipts: list[dict[str, Any]] = []
    for expert in range(EXPERTS):
        base = expert * GROUPS_PER_EXPERT
        gate_hat = np.array(decoded[base:base + ROWS], dtype=np.float64, copy=True)
        z0 = np.array(decoded[base + ROWS:base + 2 * ROWS], dtype=np.float64, copy=True)
        z1 = np.array(decoded[base + 2 * ROWS:base + 3 * ROWS], dtype=np.float64, copy=True)
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        if not math.isfinite(norm2) or norm2 <= 0.0:
            raise RuntimeError(f"invalid inverse-rotation norm for expert {expert}")
        reconstructions = (gate_hat, (cosine * z0 - sine * z1) / norm2,
                           (sine * z0 + cosine * z1) / norm2)
        for role_index, reconstruction in enumerate(reconstructions):
            ordinal = 3 * expert + role_index
            row = plan["sources"][ordinal]
            role = str(row["role"])
            if int(row["matrix_ordinal"]) != ordinal or role != ROLE_NAMES[role_index]:
                raise RuntimeError(f"source order changed at {ordinal}")
            source_path = safe_source_path(source_root, str(row["source_relpath"]))
            source_raw = read_regular_snapshot(source_path, str(row["source_bf16_sha256"]))
            shape = tuple(int(value) for value in row["shape"])
            source = bf16_snapshot_to_f64(np, source_raw, shape)
            natural = source.T.copy() if role == "down" else source.copy()
            if natural.shape != (ROWS, COLS) or reconstruction.shape != (ROWS, COLS):
                raise RuntimeError(f"matrix geometry changed at {ordinal}")
            result.append((expert, role_index, role, reconstruction, natural))
            receipts.append({"matrix_ordinal": ordinal, "expert_ordinal": expert, "role": role,
                             "source_relpath": str(row["source_relpath"]),
                             "bytes": len(source_raw), "sha256": sha256(source_raw)})
    return result, {
        "plan_sha256": sha256(plan_raw), "plan_internal_sha256": str(plan["lock_sha256"]),
        "header_sha256": sha256(header_raw), "decoded_sha256": sha256(decoded_raw),
        "decoded_bytes": len(decoded_raw), "sources": receipts,
    }


def features(cp: Any, reconstruction: Any, role_index: int) -> tuple[Any, Any]:
    row_scale = cp.maximum(
        cp.sqrt(cp.mean(reconstruction * reconstruction, axis=1, keepdims=True, dtype=cp.float64)),
        cp.float64(1.0e-30),
    )
    matrix_scale = cp.maximum(
        cp.sqrt(cp.mean(reconstruction * reconstruction, dtype=cp.float64)), cp.float64(1.0e-30)
    )
    normalized = reconstruction / row_scale
    amplitude = cp.floor(cp.clip((normalized + 4.0) * 4.0, 0.0, 31.0)).astype(cp.int32)
    left = cp.empty_like(reconstruction)
    right = cp.empty_like(reconstruction)
    left[:, 0] = reconstruction[:, 0]
    left[:, 1:] = reconstruction[:, :-1]
    right[:, -1] = reconstruction[:, -1]
    right[:, :-1] = reconstruction[:, 1:]
    left_state = ((left >= 0.0).astype(cp.int32) << 1) | (cp.abs(left) > cp.abs(reconstruction)).astype(cp.int32)
    right_state = ((right >= 0.0).astype(cp.int32) << 1) | (cp.abs(right) > cp.abs(reconstruction)).astype(cp.int32)
    log_ratio = cp.log2(row_scale / matrix_scale)
    row_class = ((log_ratio > -0.25).astype(cp.int32) + (log_ratio > 0.0).astype(cp.int32) +
                 (log_ratio > 0.25).astype(cp.int32))
    row_class = cp.broadcast_to(row_class, reconstruction.shape)
    index = ((((role_index * 4 + row_class) * 32 + amplitude) * 4 + left_state) * 4 + right_state)
    return index.reshape(-1), cp.broadcast_to(row_scale, reconstruction.shape).reshape(-1)


def accumulate_raw_ls(cp: Any, index: Any, scale: Any, residual: Any,
                      numerator: Any, denominator: Any) -> None:
    numerator += cp.bincount(index, weights=scale * residual, minlength=TABLE_ENTRIES).astype(cp.float64)
    denominator += cp.bincount(index, weights=scale * scale, minlength=TABLE_ENTRIES).astype(cp.float64)


def publish_result(output: Path, stage: Path, packet: bytes, result_raw: bytes,
                   completion: dict[str, Any]) -> None:
    write_new(stage / "fit_table_packet.bin", packet)
    write_new(stage / "result.json", result_raw)
    completion_raw = json.dumps(completion, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    write_new(stage / "COMPLETE.json", completion_raw)
    fsync_directory(stage)
    os.replace(stage / "fit_table_packet.bin", output / "fit_table_packet.bin")
    os.replace(stage / "result.json", output / "result.json")
    fsync_directory(output)
    os.replace(stage / "COMPLETE.json", output / "COMPLETE.json")
    stage.rmdir()
    fsync_directory(output)


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
    output = Path(os.path.abspath(args.output))
    reject_link_components(output.parent)
    if os.path.lexists(output):
        raise SystemExit("output must be absent; no payload opened")
    package_bindings = self_authenticate()
    os.mkdir(output, 0o700)
    stage = output / ".staging"
    os.mkdir(stage, 0o700)
    plan_dir = Path(os.path.abspath(args.plan_dir))
    reject_link_components(plan_dir)

    packet_codec = load_packet_codec()
    import cupy as cp
    import numpy as np

    started = time.monotonic()
    pairs, bindings = matrix_pairs(np, plan_dir)
    fit_num = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    fit_den = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    oracle_num = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    oracle_den = cp.zeros(TABLE_ENTRIES, dtype=cp.float64)
    panel_sse = 0.0
    panel_energy = 0.0
    holdout_sse = 0.0
    holdout_energy = 0.0
    matrix_rows: list[dict[str, Any]] = []
    cached_holdout: list[tuple[Any, Any, Any, dict[str, Any]]] = []

    for expert, role_index, role, reconstruction_host, source_host in pairs:
        reconstruction = cp.asarray(reconstruction_host, dtype=cp.float64)
        source = cp.asarray(source_host, dtype=cp.float64)
        residual = source - reconstruction
        raw_sse = float(cp.sum(residual * residual, dtype=cp.float64).item())
        energy = float(cp.sum(source * source, dtype=cp.float64).item())
        panel_sse += raw_sse
        panel_energy += energy
        index, scale = features(cp, reconstruction, role_index)
        flat_residual = residual.reshape(-1)
        row = {"expert_ordinal": expert, "role": role,
               "split": "fit" if expert in FIT_EXPERTS else "holdout",
               "baseline_sse": raw_sse, "source_energy": energy,
               "fit_fp16_table_sse": None, "holdout_self_fit_fp64_oracle_sse": None}
        matrix_rows.append(row)
        if expert in FIT_EXPERTS:
            accumulate_raw_ls(cp, index, scale, flat_residual, fit_num, fit_den)
        elif expert in HOLDOUT_EXPERTS:
            holdout_sse += raw_sse
            holdout_energy += energy
            accumulate_raw_ls(cp, index, scale, flat_residual, oracle_num, oracle_den)
            cached_holdout.append((index, scale, flat_residual, row))
        else:
            raise RuntimeError(f"expert outside frozen split: {expert}")

    if not math.isclose(panel_sse, BASELINE_SSE, rel_tol=0.0, abs_tol=2.0e-9):
        raise RuntimeError(f"baseline SSE mismatch: {panel_sse}")
    if not math.isclose(panel_energy, BASELINE_ENERGY, rel_tol=0.0, abs_tol=2.0e-9):
        raise RuntimeError(f"baseline energy mismatch: {panel_energy}")
    if not math.isclose(panel_sse / panel_energy * 32.0, BASELINE_F, rel_tol=0.0, abs_tol=2.0e-12):
        raise RuntimeError("baseline F mismatch")

    fit_table = cp.where(fit_den > 0.0, fit_num / cp.maximum(fit_den, cp.float64(1.0e-300)), 0.0)
    oracle_table = cp.where(oracle_den > 0.0, oracle_num / cp.maximum(oracle_den, cp.float64(1.0e-300)), 0.0)
    if not bool(cp.isfinite(fit_table).all().item()) or not bool(cp.isfinite(oracle_table).all().item()):
        raise RuntimeError("nonfinite FP64 table")
    fit_table_host = cp.asnumpy(fit_table).astype(np.float64)
    packet = packet_codec.build_packet(fit_table_host.tolist())
    parsed = packet_codec.parse_packet(packet)
    packet_table = cp.asarray(np.asarray(parsed["values"], dtype=np.float64))
    if not bool(cp.isfinite(packet_table).all().item()):
        raise RuntimeError("nonfinite parsed FP16 table")

    finite_sse = 0.0
    oracle_sse = 0.0
    for index, scale, residual, row in cached_holdout:
        finite_error = residual - packet_table[index] * scale
        oracle_error = residual - oracle_table[index] * scale
        finite_value = float(cp.sum(finite_error * finite_error, dtype=cp.float64).item())
        oracle_value = float(cp.sum(oracle_error * oracle_error, dtype=cp.float64).item())
        finite_sse += finite_value
        oracle_sse += oracle_value
        row["fit_fp16_table_sse"] = finite_value
        row["holdout_self_fit_fp64_oracle_sse"] = oracle_value

    dominance_tolerance = 1.0e-10 * max(1.0, holdout_sse, finite_sse)
    if oracle_sse > holdout_sse + dominance_tolerance:
        raise RuntimeError("oracle loses to legal zero-correction table")
    if oracle_sse > finite_sse + dominance_tolerance:
        raise RuntimeError("oracle loses to compared legal fit FP16 table")
    holdout_f0 = holdout_sse / holdout_energy * 32.0
    finite_ratio = finite_sse / holdout_sse
    oracle_ratio = oracle_sse / holdout_sse
    finite_f = holdout_f0 * finite_ratio * 2.0 ** (2.0 * SIDE_BPW)
    oracle_f = holdout_f0 * oracle_ratio * 2.0 ** (2.0 * SIDE_BPW)
    status = "HARD_KILL_RAVEL6144_V1" if oracle_f > TARGET_F else "PROMOTE_TO_CONTROLS_AND_REDUCED_RATE_REENCODE_ONLY"
    result = {
        "schema": "ravel-decoded-residual-lut-stage0-result-v1",
        "status": status,
        "claim_boundary": "Frozen one-table RAVEL-6144-v1 source-leaking favorable bound only; not a finite reduced-rate codec or universal converse.",
        "split": {"fit_experts": list(FIT_EXPERTS), "holdout_experts": list(HOLDOUT_EXPERTS)},
        "geometry": {"table_entries": TABLE_ENTRIES, "packet_bytes": PACKET_BYTES,
                     "features": [3, 4, 32, 4, 4], "shared_table_count": 1},
        "baseline": {"panel_sse": panel_sse, "panel_energy": panel_energy,
                     "panel_F_at_2p5": panel_sse / panel_energy * 32.0,
                     "holdout_sse": holdout_sse, "holdout_energy": holdout_energy,
                     "holdout_F_at_2p5": holdout_f0},
        "finite_fit_table": {"sse": finite_sse, "fraction_of_baseline_sse": finite_ratio,
                             "favorable_transfer_F": finite_f, "packet_dtype": "<f2-finite"},
        "source_leaking_oracle": {"method": "numerical FP64 raw-SSE weighted least squares",
                                  "sse": oracle_sse, "fraction_of_baseline_sse": oracle_ratio,
                                  "capture": 1.0 - oracle_ratio, "favorable_transfer_F": oracle_f,
                                  "emitted": False, "dominance_tolerance": dominance_tolerance,
                                  "dominates_zero_correction": oracle_sse <= holdout_sse + dominance_tolerance,
                                  "dominates_fit_fp16_table": oracle_sse <= finite_sse + dominance_tolerance},
        "rate_read": {"shared_table_count": 1, "side_bpw": SIDE_BPW,
                      "base_payload_cap_bpw": 2.5 - SIDE_BPW,
                      "published_cold_amp": PUBLISHED_COLD_AMP,
                      "conservative_cold_amp": CONSERVATIVE_COLD_AMP,
                      "below_2x": CONSERVATIVE_COLD_AMP < 2.0},
        "controls": {"matched_controls_run": False,
                     "reason": "stage-0 source-leaking oracle gate decides promotion before controls"},
        "matrix_rows": matrix_rows,
        "bindings": {**bindings, "source_package_manifest_sha256": package_bindings["manifest_sha256"],
                     "source_package_members": package_bindings["members"],
                     "script_sha256": package_bindings["members"]["ravel_stage0.py"],
                     "packet_sha256": sha256(packet)},
        "runtime": {"elapsed_seconds": time.monotonic() - started, "python": sys.version,
                    "numpy": np.__version__, "cupy": cp.__version__,
                    "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"])},
    }
    result["result_lock_sha256"] = sha256(canonical_json(result))
    result_raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    completion = {
        "schema": "ravel-decoded-residual-lut-stage0-completion-v1",
        "status": "COMPLETE",
        "members": {
            "fit_table_packet.bin": {"bytes": len(packet), "sha256": sha256(packet)},
            "result.json": {"bytes": len(result_raw), "sha256": sha256(result_raw)},
        },
        "result_lock_sha256": result["result_lock_sha256"],
        "source_package_manifest_sha256": package_bindings["manifest_sha256"],
    }
    completion["completion_lock_sha256"] = sha256(canonical_json(completion))
    publish_result(output, stage, packet, result_raw, completion)
    print(json.dumps({"status": status, "oracle_F": oracle_f, "finite_F": finite_f,
                      "result": str(output / "result.json"), "complete": str(output / "COMPLETE.json")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
