#!/usr/bin/env python3
"""Independent CuPy replay of the MALT64 rank-3 tangent projection."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import mmap
import os
import stat
import struct
import sys
import time
from pathlib import Path


sys.dont_write_bytecode = True
EXPERTS, ROLES, ROWS, COLS = 6, 3, 768, 2048
GROUPS_PER_EXPERT, GROUP = 2304, 2048
BLOCK_SIDE, BLOCKS_PER_MATRIX, RANK = 64, 384, 3
PANEL_VALUES = EXPERTS * ROLES * ROWS * COLS
PLAN_SHA = "8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868"
HEADER_SHA = "3c16bcf308c0cfce2071be24bf612d202360510084540aa0b358938d8399a538"
POST_SHA = "af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0"


def sha(raw): return hashlib.sha256(raw).hexdigest()


@contextlib.contextmanager
def held_bytes(path, size=None, digest=None):
    path = Path(path)
    named = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(named.st_mode):
        raise ValueError("non-regular evidence: " + str(path))
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                         getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if size is not None and before.st_size != size: raise ValueError("byte count: " + str(path))
        blocks, remaining = [], before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block: raise ValueError("short read: " + str(path))
            blocks.append(block); remaining -= len(block)
        if os.read(descriptor, 1): raise ValueError("late growth: " + str(path))
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("changed while held: " + str(path))
        raw = b"".join(blocks)
        if digest is not None and sha(raw) != digest: raise ValueError("SHA-256: " + str(path))
        yield raw
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def held_mmap(path, size, digest):
    path = Path(path)
    named = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(named.st_mode):
        raise ValueError("non-regular evidence: " + str(path))
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                         getattr(os, "O_NOFOLLOW", 0))
    mapping = None
    try:
        before = os.fstat(descriptor)
        if before.st_size != size: raise ValueError("byte count: " + str(path))
        digest_object = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(8 << 20, remaining))
            if not block: raise ValueError("short read: " + str(path))
            digest_object.update(block); remaining -= len(block)
        if os.read(descriptor, 1): raise ValueError("late growth: " + str(path))
        if digest_object.hexdigest() != digest: raise ValueError("SHA-256: " + str(path))
        after_hash = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after_hash.st_dev, after_hash.st_ino, after_hash.st_size, after_hash.st_mtime_ns):
            raise ValueError("changed while hashed: " + str(path))
        mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        yield mapping
        after_use = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after_use.st_dev, after_use.st_ino, after_use.st_size, after_use.st_mtime_ns):
            raise ValueError("changed while mapped: " + str(path))
    finally:
        if mapping is not None: mapping.close()
        os.close(descriptor)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("duplicate JSON key " + key)
            out[key] = value
        return out
    def finite(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError("nonfinite JSON")
        return value
    def constant(text): raise ValueError("nonfinite JSON " + text)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def matrix_score(cp, np, source, decoded, batch_blocks):
    source = np.ascontiguousarray(source).reshape(BLOCKS_PER_MATRIX, BLOCK_SIDE, BLOCK_SIDE)
    decoded = np.ascontiguousarray(decoded).reshape(BLOCKS_PER_MATRIX, BLOCK_SIDE, BLOCK_SIDE)
    error_total = capture_total = 0.0
    for begin in range(0, BLOCKS_PER_MATRIX, batch_blocks):
        stop = min(BLOCKS_PER_MATRIX, begin + batch_blocks)
        x = cp.asarray(source[begin:stop], dtype=cp.float64)
        y = cp.asarray(decoded[begin:stop], dtype=cp.float64)
        error = x - y
        u, _singular, vh = cp.linalg.svd(y, full_matrices=False)
        u = u[:, :, :RANK]
        v = vh[:, :RANK, :]
        # Alternative orthogonal decomposition of the tangent projection:
        # ||P_U E + E P_V - P_U E P_V||^2
        #   = ||U^T E||^2 + ||E V||^2 - ||U^T E V||^2.
        ute = cp.matmul(cp.swapaxes(u, 1, 2), error)
        ev = cp.matmul(error, cp.swapaxes(v, 1, 2))
        utev = cp.matmul(ute, cp.swapaxes(v, 1, 2))
        error_energy = float(cp.sum(error * error, dtype=cp.float64).item())
        capture = float((cp.sum(ute * ute, dtype=cp.float64) +
                         cp.sum(ev * ev, dtype=cp.float64) -
                         cp.sum(utev * utev, dtype=cp.float64)).item())
        if capture < -1e-12 or capture > error_energy * (1.0 + 3e-12):
            raise FloatingPointError("invalid independent tangent energy")
        error_total += error_energy; capture_total += capture
        del x, y, error, u, vh, ute, ev, utev
    return error_total, capture_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-blocks", type=int, default=32)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0": raise ValueError("CUDA_VISIBLE_DEVICES")
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    if not 1 <= args.batch_blocks <= BLOCKS_PER_MATRIX: raise ValueError("batch blocks")
    plan_dir = args.plan_dir.resolve(strict=True)
    started = time.time()
    import cupy as cp
    import numpy as np

    plan_path, header_path = plan_dir / "plan.lock.json", plan_dir / "header.bin"
    post_path = plan_dir / "independent_audit" / "post_klt_canonical_groups.f64.bin"
    with held_bytes(plan_path, 24790, PLAN_SHA) as plan_raw:
        plan = strict_json(plan_raw)
    claimed = plan["lock_sha256"]
    clean = dict(plan); del clean["lock_sha256"]
    if sha(canonical(clean)) != claimed: raise ValueError("plan internal seal")
    if len(plan["sources"]) != 18: raise ValueError("source count")
    with held_bytes(header_path, 128, HEADER_SHA) as header:
        coefficients = struct.unpack_from("<12f", header, 32)
    source_root = Path(plan["source_root"])
    expert_error = np.zeros(EXPERTS, dtype=np.float64)
    expert_capture = np.zeros(EXPERTS, dtype=np.float64)
    expert_energy = np.zeros(EXPERTS, dtype=np.float64)
    role_error = np.zeros(ROLES, dtype=np.float64)
    role_capture = np.zeros(ROLES, dtype=np.float64)
    rows, source_receipts = [], []
    with held_mmap(post_path, PANEL_VALUES * 8, POST_SHA) as post_mapping:
        post = np.ndarray((EXPERTS * GROUPS_PER_EXPERT, GROUP), dtype="<f8", buffer=post_mapping)
        for expert in range(EXPERTS):
            base = expert * GROUPS_PER_EXPERT
            gate = np.asarray(post[base:base + ROWS], dtype=np.float64)
            z0 = np.asarray(post[base + ROWS:base + 2 * ROWS], dtype=np.float64)
            z1 = np.asarray(post[base + 2 * ROWS:base + 3 * ROWS], dtype=np.float64)
            cosine, sine = coefficients[2 * expert:2 * expert + 2]
            norm2 = cosine * cosine + sine * sine
            decoded_roles = (gate, (cosine * z0 - sine * z1) / norm2,
                              (sine * z0 + cosine * z1) / norm2)
            for role_ordinal, decoded in enumerate(decoded_roles):
                ordinal = 3 * expert + role_ordinal
                row = plan["sources"][ordinal]
                role = ("gate", "up", "down")[role_ordinal]
                if row["matrix_ordinal"] != ordinal or row["role"] != role:
                    raise ValueError("plan source order")
                relative = Path(row["source_relpath"])
                if relative.is_absolute() or ".." in relative.parts: raise ValueError("source path")
                source_path = source_root / relative
                with held_bytes(source_path, 3145728, row["source_bf16_sha256"]) as raw:
                    words = np.frombuffer(raw, dtype="<u2")
                    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
                    shape = (COLS, ROWS) if role == "down" else (ROWS, COLS)
                    source_raw = values.astype(np.float64).reshape(shape)
                    source = source_raw.T.copy() if role == "down" else source_raw
                    if not bool(np.isfinite(source).all()): raise ValueError("nonfinite source")
                    energy = float(np.sum(source * source, dtype=np.float64))
                    error, capture = matrix_score(cp, np, source, decoded, args.batch_blocks)
                expert_error[expert] += error; expert_capture[expert] += capture
                expert_energy[expert] += energy
                role_error[role_ordinal] += error; role_capture[role_ordinal] += capture
                rows.append({"matrix_ordinal": ordinal, "expert_ordinal": expert, "role": role,
                             "source_energy_fp64": energy, "coarse_error_sse_fp64": error,
                             "tangent_projection_energy_fp64": capture,
                             "capture_fraction": capture / error, "blocks": BLOCKS_PER_MATRIX})
                source_receipts.append({"matrix_ordinal": ordinal, "sha256": row["source_bf16_sha256"],
                                        "bytes": 3145728})
                print(f"independent matrix {ordinal:02d}/17 {role:4s} capture={capture/error:.9f}", flush=True)
        # Release every ndarray view before closing the held mmap.
        del post, gate, z0, z1, decoded_roles, decoded
    total_error = float(np.sum(expert_error, dtype=np.float64))
    total_capture = float(np.sum(expert_capture, dtype=np.float64))
    total_energy = float(np.sum(expert_energy, dtype=np.float64))
    estimate = total_capture / total_error
    deletes = [(total_capture - float(expert_capture[i])) / (total_error - float(expert_error[i]))
               for i in range(EXPERTS)]
    center = float(np.mean(np.asarray(deletes), dtype=np.float64))
    se = math.sqrt(5.0 / 6.0 * float(np.sum((np.asarray(deletes) - center) ** 2,
                                            dtype=np.float64)))
    result = {
        "schema": "malt64_independent_gpu_recompute_v1", "status": "PASS",
        "method": "alternative tangent energy identity using UTE, EV, and UTEV norms",
        "matrices": rows,
        "experts": [{"expert_ordinal": i, "coarse_error_sse_fp64": float(expert_error[i]),
                     "tangent_projection_energy_fp64": float(expert_capture[i])} for i in range(6)],
        "roles": [{"role": ("gate", "up", "down")[i],
                   "coarse_error_sse_fp64": float(role_error[i]),
                   "tangent_projection_energy_fp64": float(role_capture[i])} for i in range(3)],
        "aggregate": {"source_energy_fp64": total_energy, "coarse_error_sse_fp64": total_error,
                      "tangent_projection_energy_fp64": total_capture, "capture_fraction": estimate,
                      "delete_one_expert": deletes, "jackknife_center": center,
                      "jackknife_se": se, "upper_three_se": estimate + 3.0 * se},
        "bindings": {"plan_sha256": PLAN_SHA, "plan_internal_lock_sha256": claimed,
                     "header_sha256": HEADER_SHA, "post_klt_sha256": POST_SHA,
                     "sources": source_receipts},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "cupy": cp.__version__, "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                    "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
                    "elapsed_seconds": time.time() - started},
        "access": {"plan_header_post_files_opened": 3, "pinned_source_files_opened": 18,
                   "fresh_validation_files_opened": 0, "network_operations": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "capture_fraction": estimate,
                      "upper_three_se": estimate + 3.0 * se,
                      "output_sha256": sha(output.read_bytes())}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
