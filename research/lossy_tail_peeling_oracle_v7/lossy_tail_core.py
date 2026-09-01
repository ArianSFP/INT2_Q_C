#!/usr/bin/env python3
"""NumPy/CuPy scientific core for lossy-tail peeling v7.

Every mask, scalar symbol, finite FP16 codebook, descriptor, directory and
frame bit is accounted.  The bulk remains an intentionally ideal Gaussian RD
channel, so this program is a promotion/early-kill gate rather than a codec.
Qwen and four moment-matched Gaussian panels execute the identical search.

This file is not an entrypoint.  The stdlib-only authenticated bootstrap or
runtime calibrator descriptor-executes these exact bytes with a bounded core
context.  Direct execution/import fails before NumPy is imported.
"""

from __future__ import annotations

import hashlib as _v7_firewall_hashlib
import os as _v7_firewall_os
import sys as _v7_firewall_sys


_V7_CORE_CONTEXT = globals().get("__V7_CORE_CONTEXT__")
if (
    not isinstance(_V7_CORE_CONTEXT, dict)
    or _V7_CORE_CONTEXT.get("schema") != "lossy-tail-v7-core-context-v1"
    or _V7_CORE_CONTEXT.get("mode") not in {"production_child", "runtime_calibration", "source_cpu_test"}
    or __name__ == "__main__"
):
    raise RuntimeError("V7_CORE_FIREWALL_REJECT: authenticated descriptor context required before NumPy import")


def _v7_preimport_production_firewall(context: dict) -> None:
    """Reprove the live preflight parent before any third-party import.

    A dictionary injected by an arbitrary wrapper is not authority.  The
    production context is accepted only in the still-live child whose parent
    has the exact authenticated-preflight command line and executable inode.
    """
    if context.get("mode") != "production_child":
        return
    expected_context_keys = {
        "schema", "mode", "parent_pid", "child_pid", "capability_sha256",
        "preflight_cmdline_sha256", "launch_manifest_sha256",
        "authorization_file_sha256", "authorization_internal_sha256",
    }
    if set(context) != expected_context_keys:
        raise RuntimeError("V7_CORE_FIREWALL_REJECT: production context key drift before NumPy import")
    if context.get("child_pid") != _v7_firewall_os.getpid() or context.get("parent_pid") != _v7_firewall_os.getppid():
        raise RuntimeError("V7_CORE_FIREWALL_REJECT: production process identity mismatch before NumPy import")
    flags = [
        "--bindings", "--protocol", "--repair-lock", "--runtime-contract",
        "--authorization-contract", "--launch-manifest", "--launch-manifest-sha256",
        "--authorization", "--authorization-sha256", "--control-replicates",
        "--maximum-coordinate-passes",
    ]
    raw = _v7_firewall_sys.argv[1:]
    if len(raw) != 2 * len(flags) or raw[::2] != flags or raw[19] != "4" or raw[21] != "4":
        raise RuntimeError("V7_CORE_FIREWALL_REJECT: production grammar mismatch before NumPy import")
    bootstrap = _v7_firewall_sys.argv[0]
    core_file = globals().get("__file__")
    if (
        not isinstance(bootstrap, str) or not isinstance(core_file, str)
        or not _v7_firewall_os.path.isabs(bootstrap) or not _v7_firewall_os.path.isabs(core_file)
        or bootstrap != _v7_firewall_os.path.normpath(bootstrap)
        or core_file != _v7_firewall_os.path.normpath(core_file)
        or bootstrap != _v7_firewall_os.path.realpath(bootstrap)
        or core_file != _v7_firewall_os.path.realpath(core_file)
        or _v7_firewall_os.path.basename(bootstrap) != "lossy_tail_oracle.py"
        or _v7_firewall_os.path.basename(core_file) != "lossy_tail_core.py"
        or _v7_firewall_os.path.dirname(bootstrap) != _v7_firewall_os.path.dirname(core_file)
    ):
        raise RuntimeError("V7_CORE_FIREWALL_REJECT: raw bootstrap/core stage mismatch before NumPy import")
    stage = _v7_firewall_os.path.dirname(core_file)
    exact_stage_values = {
        1: "source_bindings.json", 3: "protocol_lock.json", 5: "repair_lock.json",
        7: "runtime_contract.json", 9: "authorization_contract.json", 11: "launch_manifest.json",
    }
    if any(raw[index] != _v7_firewall_os.path.join(stage, name) for index, name in exact_stage_values.items()):
        raise RuntimeError("V7_CORE_FIREWALL_REJECT: stage argument mismatch before NumPy import")
    parent_fields = [
        _v7_firewall_sys.executable, "-B", "-I",
        _v7_firewall_os.path.join(stage, "preflight_launch.py"),
        "--manifest", _v7_firewall_os.path.join(stage, "launch_manifest.json"),
        "--manifest-sha256", raw[13],
        "--authorization", raw[15],
        "--authorization-sha256", raw[17],
    ]
    expected_parent_cmdline = b"\x00".join(field.encode("utf-8") for field in parent_fields) + b"\x00"
    try:
        with open(f"/proc/{context['parent_pid']}/cmdline", "rb") as stream:
            live_parent_cmdline = stream.read()
        parent_executable = _v7_firewall_os.stat(f"/proc/{context['parent_pid']}/exe")
        expected_executable = _v7_firewall_os.stat(_v7_firewall_sys.executable)
    except (OSError, KeyError, TypeError) as exc:
        raise RuntimeError("V7_CORE_FIREWALL_REJECT: cannot authenticate live preflight parent before NumPy import") from exc
    parent_hash = _v7_firewall_hashlib.sha256(live_parent_cmdline).hexdigest()
    if (
        live_parent_cmdline != expected_parent_cmdline
        or context.get("preflight_cmdline_sha256") != parent_hash
        or context.get("launch_manifest_sha256") != raw[13]
        or context.get("authorization_file_sha256") != raw[17]
        or (parent_executable.st_dev, parent_executable.st_ino)
        != (expected_executable.st_dev, expected_executable.st_ino)
    ):
        raise RuntimeError("V7_CORE_FIREWALL_REJECT: live preflight parent mismatch before NumPy import")


_v7_preimport_production_firewall(_V7_CORE_CONTEXT)
_V7_PREIMPORT_PRODUCTION_AUTHENTICATED = _V7_CORE_CONTEXT.get("mode") == "production_child"

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROWS = 768
COLS = 2048
N = ROWS * COLS
EXPERTS = 6
ROLES = 2
MATRICES = EXPERTS * ROLES
PANEL_N = MATRICES * N
RATES = (2.15, 2.30, 2.50)
FRACTIONS = (0.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0)
FAMILIES = (("coordinate", 1), ("block16", 16), ("block64", 64))
LEVELS = (1, 2, 4, 8, 16)
MODES = ("free_lloyd", "finite_fp16", "zero_tail_error")
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
KILL_GUARD_S = 0.02
KILL_THRESHOLD_S = TARGET_S - KILL_GUARD_S
NUMERIC_BOUNDARY_GUARD_S = 1.0e-4
DECISION_CONSISTENCY_EPSILON_S = 1.0e-12
MEAN_TOLERANCE_ULPS = 64
VARIANCE_TOLERANCE_ULPS = 256
FLOAT32_EPSILON = 2.0 ** -23

GLOBAL_HEADER_BITS = 4096 * 8
ROUTE_TABLE_BITS = 144 * 8
COMMON_BITS = GLOBAL_HEADER_BITS + ROUTE_TABLE_BITS
EXPERT_HEADER_BITS = 256
MATRIX_DESCRIPTOR_BITS = 128
RESIDUAL_DIRECTORY_BITS = 64
ANGLE_BITS = 16
END_PAD_RESERVE_BITS = 7 * EXPERTS
PAGE_BYTES = 4096
STAGE_MEMBERS_V7 = {
    "authorization_contract.json",
    "audit_lock_entrypoint.py",
    "launch_manifest.json", "lossy_tail_core.py", "lossy_tail_oracle.py", "preflight_launch.py",
    "protocol_lock.json", "repair_lock.json", "runtime_calibrate.py",
    "runtime_contract.json", "source_bindings.json",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_object)
    except Exception as exc:
        raise RuntimeError(f"invalid strict JSON for {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RuntimeError(
            f"{label} key drift: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def require_canonical_original_path(path: str | os.PathLike[str], label: str, *, allow_missing_tail: bool) -> Path:
    """Reject alias-bearing original spellings before returning a path.

    No normalization is performed before the raw spelling, dot components,
    and every existing prefix have been checked.  This closes the
    ``LINK/../stage`` mismatch between lexical abspath and kernel traversal.
    """
    raw = os.fspath(path)
    if not raw or "\x00" in raw or not os.path.isabs(raw):
        raise RuntimeError(f"{label} must use an absolute canonical spelling")
    if raw != os.path.normpath(raw):
        raise RuntimeError(f"{label} contains dot, parent, duplicate-separator, or trailing components")
    candidate = Path(raw)
    parts = candidate.parts
    probe = Path(parts[0])
    missing = False
    for component in parts[1:]:
        probe = probe / component
        if missing:
            continue
        try:
            metadata = os.lstat(probe)
        except FileNotFoundError:
            if not allow_missing_tail:
                raise RuntimeError(f"{label} component is missing: {probe}")
            missing = True
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} contains a symlink component: {probe}")
    if os.path.realpath(raw) != raw:
        raise RuntimeError(f"{label} does not name its actual canonical target")
    return candidate


def require_raw_entrypoint(expected_basename: str) -> Path:
    raw = sys.argv[0]
    entrypoint = require_canonical_original_path(raw, "raw argv0", allow_missing_tail=False)
    if entrypoint.name != expected_basename:
        raise RuntimeError(f"raw argv0 basename must be {expected_basename}")
    module_raw = os.fspath(Path(__file__))
    if raw != module_raw:
        raise RuntimeError("raw argv0 must exactly equal executing __file__")
    raw_stat = os.stat(entrypoint, follow_symlinks=False)
    module_stat = os.stat(module_raw, follow_symlinks=False)
    if (raw_stat.st_dev, raw_stat.st_ino) != (module_stat.st_dev, module_stat.st_ino):
        raise RuntimeError("raw argv0 and executing module identity differ")
    return entrypoint


def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def require_pairwise_disjoint(paths: Sequence[tuple[str, Path]]) -> None:
    for ordinal, (left_label, left) in enumerate(paths):
        for right_label, right in paths[ordinal + 1 :]:
            if path_contains(left, right) or path_contains(right, left):
                raise RuntimeError(f"protected paths overlap: {left_label}={left} {right_label}={right}")


def decode_mount_field(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def mount_snapshot() -> tuple[bytes, list[dict[str, Any]]]:
    payload = read_regular_descriptor(Path("/proc/self/mountinfo"), "mountinfo")
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        fields = line.split()
        if "-" not in fields or len(fields) < 7:
            raise RuntimeError("malformed /proc/self/mountinfo")
        rows.append({
            "mount_id": int(fields[0]),
            "root": decode_mount_field(fields[3]),
            "mount_point": decode_mount_field(fields[4]),
            "major_minor": fields[2],
        })
    return payload, rows


def mount_row_for(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matches = []
    for row in rows:
        mount_point = Path(row["mount_point"])
        if path_contains(mount_point, path):
            matches.append((len(mount_point.parts), row))
    if not matches:
        raise RuntimeError(f"no mountinfo row covers {path}")
    return max(matches, key=lambda item: item[0])[1]


def require_no_nested_mounts(root: Path, rows: Sequence[dict[str, Any]], label: str) -> None:
    covering = mount_row_for(root, rows)
    for row in rows:
        point = Path(row["mount_point"])
        if row["mount_id"] != covering["mount_id"] and path_contains(root, point):
            raise RuntimeError(f"nested mount transition below {label}: {point}")


def read_regular_descriptor(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(path), flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        payload = b"".join(chunks)
        if metadata.st_size != 0 and len(payload) != metadata.st_size:
            raise RuntimeError(f"{label} changed during descriptor read")
        return payload
    finally:
        os.close(descriptor)


def read_regular_child_descriptor(directory: Path, name: str, label: str) -> bytes:
    if not name or name in (".", "..") or os.path.basename(name) != name or "/" in name or "\\" in name:
        raise RuntimeError(f"invalid child name for {label}")
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(os.fspath(directory), directory_flags)
    try:
        directory_stat = os.fstat(directory_fd)
        live_stat = os.stat(directory, follow_symlinks=False)
        if (directory_stat.st_dev, directory_stat.st_ino) != (live_stat.st_dev, live_stat.st_ino):
            raise RuntimeError(f"directory identity changed for {label}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"{label} is not a regular file")
            blocks: list[bytes] = []
            while block := os.read(descriptor, 1 << 20):
                blocks.append(block)
            payload = b"".join(blocks)
            if len(payload) != metadata.st_size:
                raise RuntimeError(f"{label} changed during descriptor read")
            return payload
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)


def validate_stage_manifest(stage: Path, manifest_path: Path, external_sha256: str) -> dict[str, Any]:
    if len(external_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in external_sha256):
        raise RuntimeError("invalid external launch-manifest SHA-256")
    manifest_bytes = read_regular_descriptor(manifest_path, "launch manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != external_sha256:
        raise RuntimeError("external launch-manifest SHA-256 mismatch")
    manifest = strict_json_bytes(manifest_bytes, "launch manifest")
    require_exact_keys(manifest, {"schema", "status", "allowed_members", "members", "source_audit_invocation", "runtime_calibration_invocation_after_independent_source_pass_only", "production_invocation_after_independent_runtime_receipt_audit_and_separate_authorization_only", "production_child_grammar", "authorization"}, "launch manifest")
    if manifest.get("schema") != "lossy-tail-v7-launch-manifest-v1":
        raise RuntimeError("launch-manifest schema mismatch")
    if manifest.get("status") != "FROZEN_V7_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        raise RuntimeError("launch-manifest status mismatch")
    allowed_rows = manifest.get("allowed_members", [])
    if not isinstance(allowed_rows, list) or len(allowed_rows) != len(STAGE_MEMBERS_V7):
        raise RuntimeError("launch-manifest allowed-member cardinality drift")
    if len(set(allowed_rows)) != len(allowed_rows) or set(allowed_rows) != STAGE_MEMBERS_V7:
        raise RuntimeError("launch-manifest allowed-member drift")
    observed: set[str] = set()
    with os.scandir(stage) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise RuntimeError(f"forbidden stage member: {entry.name}")
            observed.add(entry.name)
    if observed != STAGE_MEMBERS_V7:
        raise RuntimeError(f"stage membership mismatch: {sorted(observed)}")
    rows = manifest.get("members", [])
    expected_rows = STAGE_MEMBERS_V7 - {"launch_manifest.json"}
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise RuntimeError("launch-manifest member-row cardinality drift")
    row_paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(row_paths) != len(rows) or len(set(row_paths)) != len(row_paths) or set(row_paths) != expected_rows:
        raise RuntimeError("launch-manifest member rows do not close")
    for row in rows:
        require_exact_keys(row, {"path", "bytes", "sha256"}, f"manifest row {row.get('path')}")
        member = stage / row["path"]
        member_bytes = read_regular_descriptor(member, f"manifested member {row['path']}")
        if len(member_bytes) != int(row["bytes"]) or hashlib.sha256(member_bytes).hexdigest() != row["sha256"]:
            raise RuntimeError(f"manifested member identity mismatch: {row['path']}")
    return manifest


def stable_seed(*parts: object) -> int:
    blob = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def write_sealed_json(path: Path, report: dict[str, Any]) -> None:
    clean = dict(report)
    clean.pop("result_lock_sha256", None)
    clean["result_lock_sha256"] = hashlib.sha256(canonical_json_bytes(clean)).hexdigest()
    payload = (json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    os.mkdir(path.parent, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(words, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


with np.errstate(invalid="ignore"):
    BF16_TABLE = bf16_words_to_float32(np.arange(1 << 16, dtype=np.uint16)).astype(np.float64)


@functools.lru_cache(maxsize=None)
def ceil_log2_binomial(n: int, k: int) -> int:
    if not 0 <= k <= n:
        raise ValueError((n, k))
    count = math.comb(n, min(k, n - k))
    return 0 if count <= 1 else (count - 1).bit_length()


def pad8(bits: int) -> int:
    return 8 * ((int(bits) + 7) // 8)


def make_profiles() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "id": 0, "family": "coordinate", "unit": 1,
        "fraction": 0.0, "levels": 1,
    }]
    for family, unit in FAMILIES:
        for fraction in FRACTIONS[1:]:
            for levels in LEVELS:
                rows.append({
                    "id": len(rows), "family": family, "unit": unit,
                    "fraction": fraction, "levels": levels,
                })
    return rows


PROFILES = make_profiles()


def weighted_lloyd(counts: np.ndarray, value_table: np.ndarray, levels: int) -> dict[str, Any]:
    """Deterministic weighted 1-D Lloyd fit over the represented alphabet."""
    counts = np.asarray(counts, dtype=np.int64)
    live = (counts > 0) & np.isfinite(value_table)
    x = np.asarray(value_table[live], dtype=np.float64)
    w = np.asarray(counts[live], dtype=np.float64)
    if not len(x):
        return {"centroids": [], "free_sse": 0.0, "fp16_centroids": [], "fp16_sse": 0.0, "iterations": 0}
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    levels = int(levels)
    cumulative = np.cumsum(w)
    targets = (np.arange(levels, dtype=np.float64) + 0.5) * cumulative[-1] / levels
    centers = x[np.minimum(np.searchsorted(cumulative, targets, side="left"), len(x) - 1)]
    iterations = 0
    for iterations in range(1, 65):
        bounds = 0.5 * (centers[:-1] + centers[1:])
        labels = np.searchsorted(bounds, x, side="right")
        next_centers = centers.copy()
        for j in range(levels):
            hit = labels == j
            if np.any(hit):
                next_centers[j] = float(np.sum(w[hit] * x[hit], dtype=np.float64) / np.sum(w[hit], dtype=np.float64))
        next_centers.sort()
        if np.array_equal(next_centers, centers) or np.max(np.abs(next_centers - centers)) <= 1e-14 * max(1.0, float(np.max(np.abs(centers)))):
            centers = next_centers
            break
        centers = next_centers
    free_dist = np.min((x[:, None] - centers[None, :]) ** 2, axis=1)
    free_sse = float(np.sum(w * free_dist, dtype=np.float64))
    fp16_centers = centers.astype(np.float16).astype(np.float64)
    finite_dist = np.min((x[:, None] - fp16_centers[None, :]) ** 2, axis=1)
    finite_sse = float(np.sum(w * finite_dist, dtype=np.float64))
    return {
        "centroids": centers.tolist(),
        "free_sse": free_sse,
        "fp16_centroids": fp16_centers.tolist(),
        "fp16_sse": finite_sse,
        "iterations": iterations,
    }


@dataclass(frozen=True)
class Component:
    name: str
    owner: int
    dimension: int
    energy: float


def continuous_waterfill(dimensions: np.ndarray, energies: np.ndarray, budget: int) -> np.ndarray:
    if budget < 0 or np.any(dimensions <= 0) or np.any(energies <= 0):
        raise ValueError("invalid waterfill")
    if budget == 0:
        return np.zeros_like(energies)
    variance = energies / dimensions
    logv = np.log2(variance)
    order = np.argsort(-logv, kind="stable")
    active_d = 0.0
    active_dlogv = 0.0
    logtheta = float("nan")
    for rank, idx in enumerate(order):
        active_d += float(dimensions[idx])
        active_dlogv += float(dimensions[idx] * logv[idx])
        logtheta = (active_dlogv - 2.0 * budget) / active_d
        if rank + 1 == len(order) or float(logv[order[rank + 1]]) <= logtheta:
            break
    bits = 0.5 * dimensions * np.maximum(0.0, logv - logtheta)
    # Remove harmless aggregate roundoff before integer closure.
    total = float(np.sum(bits, dtype=np.float64))
    if total > 0.0:
        bits *= budget / total
    return bits


def integer_waterfill(components: Sequence[Component], budget: int) -> dict[str, Any]:
    if not components:
        raise ValueError("no residual components")
    dims = np.asarray([c.dimension for c in components], dtype=np.float64)
    energies = np.asarray([c.energy for c in components], dtype=np.float64)
    real = continuous_waterfill(dims, energies, int(budget))
    bits = np.floor(real).astype(np.int64)
    remaining = int(budget - int(np.sum(bits, dtype=np.int64)))
    if remaining < 0:
        raise RuntimeError("negative waterfill remainder")
    # The continuous floors leave fewer than component_count bits.  Assign each
    # by its exact one-bit marginal reduction, recomputing only if necessary.
    for _ in range(remaining):
        current = energies * np.exp(-2.0 * bits / dims * math.log(2.0))
        marginal = current * (1.0 - np.exp(-2.0 / dims * math.log(2.0)))
        winner = int(np.argmax(marginal))
        bits[winner] += 1
    if int(np.sum(bits, dtype=np.int64)) != int(budget):
        raise RuntimeError("payload does not close")
    distortion = energies * np.exp(-2.0 * bits / dims * math.log(2.0))
    return {
        "payload_bits": int(budget),
        "distortion_sse": float(np.sum(distortion, dtype=np.float64)),
        "allocations": [
            {
                "name": c.name, "owner_expert": c.owner,
                "dimension": c.dimension, "energy": c.energy,
                "payload_bits": int(b), "distortion_sse": float(d),
            }
            for c, b, d in zip(components, bits, distortion, strict=True)
        ],
    }


def support_key(profile: dict[str, Any]) -> str:
    return f"{profile['family']}:{profile['fraction']:.8f}"


def candidate_tail_sse(candidate: dict[str, Any], mode: str) -> float:
    if mode == "free_lloyd":
        return float(candidate["free_lloyd_sse"])
    if mode == "finite_fp16":
        return float(candidate["fp16_sse"])
    if mode == "zero_tail_error":
        return 0.0
    raise ValueError(mode)


def candidate_side_bits(candidate: dict[str, Any], mode: str) -> int:
    bits = int(candidate["support_stream_bits"] + candidate["symbol_stream_bits"])
    if mode == "finite_fp16" and candidate["selected_scalars"]:
        bits += int(candidate["codebook_bits"])
    return bits


def stable_descending(metric: Any, cp: Any) -> Any:
    # CuPy argsort is stable for kind=None.  Canonical ordinal is therefore
    # the exact secondary key for equal metrics without a mixed-dtype stack.
    return cp.argsort(-metric, kind="stable" if cp is np else None)


def gpu_sum(value: Any, cp: Any) -> float:
    return float(cp.asnumpy(cp.sum(value, dtype=cp.float64)))


def build_matrix_candidates(x: Any, base_words: Any, value_table: np.ndarray, cp: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if int(x.size) != N or int(base_words.size) != N:
        raise ValueError("matrix length mismatch")
    x = x.reshape(-1)
    base_words = base_words.reshape(-1)
    total_energy = gpu_sum(x.astype(cp.float64) * x.astype(cp.float64), cp)
    mean = float(cp.asnumpy(cp.mean(x, dtype=cp.float64)))
    variance = float(cp.asnumpy(cp.mean((x.astype(cp.float64) - mean) ** 2, dtype=cp.float64)))
    support_masks: dict[str, Any] = {}
    support_meta: dict[str, dict[str, Any]] = {}
    zero = cp.zeros(N, dtype=cp.bool_)
    support_masks["coordinate:0.00000000"] = zero
    support_meta["coordinate:0.00000000"] = {
        "family": "coordinate", "unit": 1, "fraction": 0.0,
        "units": N, "selected_units": 0, "selected_scalars": 0,
        "support_bits": 0, "support_stream_bits": 0,
    }
    abs_order = stable_descending(cp.abs(x), cp)
    for family, unit in FAMILIES:
        units = N // unit
        if unit == 1:
            order = abs_order
        else:
            block_energy = cp.sum(x.reshape(units, unit).astype(cp.float64) ** 2, axis=1)
            order = stable_descending(block_energy, cp)
        for fraction in FRACTIONS[1:]:
            selected_units = int(round(units * fraction))
            unit_mask = cp.zeros(units, dtype=cp.bool_)
            unit_mask[order[:selected_units]] = True
            mask = unit_mask if unit == 1 else cp.repeat(unit_mask, unit)
            key = f"{family}:{fraction:.8f}"
            support_masks[key] = mask
            support_bits = ceil_log2_binomial(units, selected_units)
            support_meta[key] = {
                "family": family, "unit": unit, "fraction": fraction,
                "units": units, "selected_units": selected_units,
                "selected_scalars": selected_units * unit,
                "support_bits": support_bits,
                "support_stream_bits": pad8(support_bits),
            }
    del abs_order

    by_profile: list[dict[str, Any]] = []
    quant_cache: dict[tuple[str, int], dict[str, Any]] = {}
    energy_cache: dict[str, float] = {}
    hist_cache: dict[str, np.ndarray] = {}
    for profile in PROFILES:
        key = support_key(profile)
        meta = support_meta[key]
        k = int(meta["selected_scalars"])
        if key not in energy_cache:
            mask = support_masks[key]
            if k:
                selected_x = x[mask].astype(cp.float64)
                tail_energy = gpu_sum(selected_x * selected_x, cp)
                counts = cp.bincount(base_words[mask].astype(cp.int32), minlength=1 << 16)
                hist_cache[key] = np.asarray(cp.asnumpy(counts), dtype=np.int64)
            else:
                tail_energy = 0.0
                hist_cache[key] = np.zeros(1 << 16, dtype=np.int64)
            energy_cache[key] = tail_energy
        levels = int(profile["levels"])
        cache_key = (key, levels)
        if cache_key not in quant_cache:
            quant_cache[cache_key] = weighted_lloyd(hist_cache[key], value_table, levels)
        quant = quant_cache[cache_key]
        log_levels = int(round(math.log2(levels)))
        symbol_bits = k * log_levels
        candidate = {
            **profile,
            **meta,
            "tail_energy": energy_cache[key],
            "bulk_energy": max(0.0, total_energy - energy_cache[key]),
            "bulk_dimension": N - k,
            "symbol_bits": symbol_bits,
            "symbol_stream_bits": pad8(symbol_bits),
            "codebook_bits": 16 * levels if k else 0,
            "free_lloyd_sse": quant["free_sse"],
            "fp16_sse": quant["fp16_sse"],
            "centroids": quant["centroids"],
            "fp16_centroids": quant["fp16_centroids"],
            "lloyd_iterations": quant["iterations"],
        }
        by_profile.append(candidate)
    if [row["id"] for row in by_profile] != list(range(len(PROFILES))):
        raise RuntimeError("profile order drift")
    return by_profile, support_masks, {"energy": total_energy, "mean": mean, "variance": variance}


def pair_xklt_components(x0: Any, x1: Any, masks0: dict[str, Any], masks1: dict[str, Any], owner: int, cp: Any) -> dict[str, list[Component]]:
    output: dict[str, list[Component]] = {}
    keys = sorted(set(masks0) & set(masks1))
    x0d, x1d = x0.astype(cp.float64), x1.astype(cp.float64)
    for key in keys:
        active0, active1 = ~masks0[key], ~masks1[key]
        both = active0 & active1
        only0 = active0 & ~active1
        only1 = active1 & ~active0
        rows: list[Component] = []
        count_both = int(cp.asnumpy(cp.count_nonzero(both)))
        if count_both:
            a, b = x0d[both], x1d[both]
            e0 = gpu_sum(a * a, cp)
            e1 = gpu_sum(b * b, cp)
            cross = gpu_sum(a * b, cp)
            eig = np.linalg.eigvalsh(np.asarray([[e0, cross], [cross, e1]], dtype=np.float64))
            for axis, energy in enumerate(eig):
                if energy > 0.0:
                    rows.append(Component(f"expert_{owner:02d}.{key}.both_axis_{axis}", owner, count_both, float(energy)))
        for label, active, values in (("only_up", only0, x0d), ("only_down", only1, x1d)):
            count = int(cp.asnumpy(cp.count_nonzero(active)))
            if count:
                energy = gpu_sum(values[active] * values[active], cp)
                if energy > 0.0:
                    rows.append(Component(f"expert_{owner:02d}.{key}.{label}", owner, count, energy))
        output[key] = rows
    return output


def load_qwen_pair(source_dir: Path, entries: Sequence[dict[str, Any]], cp: Any) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    values, words, receipts = [], [], []
    for entry in entries:
        path = source_dir / entry["name"]
        payload = read_regular_child_descriptor(source_dir, entry["name"], f"source {entry['name']}")
        size = len(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if size != 2 * N or digest != entry["sha256"]:
            raise RuntimeError(f"source identity mismatch: {path}")
        host_words = np.frombuffer(payload, dtype="<u2").copy()
        host_values = bf16_words_to_float32(host_words)
        if not np.all(np.isfinite(host_values)):
            raise RuntimeError(f"non-finite source: {path}")
        words.append(cp.asarray(host_words))
        values.append(cp.asarray(host_values))
        receipts.append({**entry, "path": str(path), "bytes": size, "observed_sha256": digest})
    return values, words, receipts


def control_moment_tolerances(target_mean: float, target_variance: float) -> tuple[float, float]:
    if not math.isfinite(target_mean) or not math.isfinite(target_variance) or target_variance <= 0.0:
        raise RuntimeError("control target moments must be finite with positive variance")
    target_mean_square = target_mean * target_mean
    if not math.isfinite(target_mean_square):
        raise RuntimeError("control target mean square is not finite")
    mean_scale = max(math.sqrt(target_variance), abs(target_mean), 2.0 ** -126)
    variance_scale = max(target_variance, target_mean_square, 2.0 ** -252)
    tolerances = (
        MEAN_TOLERANCE_ULPS * FLOAT32_EPSILON * mean_scale,
        VARIANCE_TOLERANCE_ULPS * FLOAT32_EPSILON * variance_scale,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in tolerances):
        raise RuntimeError("control moment tolerance is not finite and positive")
    return tolerances


def validate_control_moments(
    *, target_mean: float, target_variance: float,
    observed_mean: float, observed_variance: float,
    scale: float, offset: float,
) -> dict[str, float]:
    values = (target_mean, target_variance, observed_mean, observed_variance, scale, offset)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite matched-control affine or moment")
    if target_variance <= 0.0 or observed_variance <= 0.0 or scale <= 0.0:
        raise RuntimeError("matched-control variance and scale must be positive")
    mean_tolerance, variance_tolerance = control_moment_tolerances(target_mean, target_variance)
    mean_error = abs(observed_mean - target_mean)
    variance_error = abs(observed_variance - target_variance)
    mean_normalized = mean_error / mean_tolerance
    variance_normalized = variance_error / variance_tolerance
    if mean_error > mean_tolerance or variance_error > variance_tolerance:
        raise RuntimeError(
            "post-FP32 matched-control moment mismatch: "
            f"mean={mean_normalized:.17g} variance={variance_normalized:.17g} normalized"
        )
    return {
        "mean_absolute_error": mean_error,
        "mean_absolute_tolerance": mean_tolerance,
        "mean_normalized_mismatch": mean_normalized,
        "variance_absolute_error": variance_error,
        "variance_absolute_tolerance": variance_tolerance,
        "variance_normalized_mismatch": variance_normalized,
    }


def make_control_matrix(source_moment: dict[str, float], replica: int, ordinal: int, cp: Any) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    seed = stable_seed("lossy-tail-peeling-gaussian-v1", replica, ordinal)
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal(N, dtype=cp.float32)
    bits = z.view(cp.uint32)
    rounding = cp.uint32(0x7FFF) + ((bits >> cp.uint32(16)) & cp.uint32(1))
    words = ((bits + rounding) >> cp.uint32(16)).astype(cp.uint16)
    zbf = (words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
    zmean = float(cp.asnumpy(cp.mean(zbf, dtype=cp.float64)))
    zvar = float(cp.asnumpy(cp.mean((zbf.astype(cp.float64) - zmean) ** 2, dtype=cp.float64)))
    if not math.isfinite(zmean) or not math.isfinite(zvar) or zvar <= 0.0:
        raise RuntimeError("invalid pre-affine BF16 control moments")
    scale = math.sqrt(float(source_moment["variance"]) / zvar)
    offset = float(source_moment["mean"]) - scale * zmean
    # Build one GPU lookup table and gather through it.  This makes every
    # control scalar bit-identical to the numeric alphabet used by the CPU
    # histogram scorer (and avoids a possible FMA/table mismatch).
    table_words = cp.arange(1 << 16, dtype=cp.uint16)
    table_gpu = (table_words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
    table_gpu = table_gpu * cp.float32(scale) + cp.float32(offset)
    x = table_gpu[words]
    table = np.asarray(cp.asnumpy(table_gpu), dtype=np.float64)
    observed_mean = float(cp.asnumpy(cp.mean(x, dtype=cp.float64)))
    observed_variance = float(cp.asnumpy(cp.mean((x.astype(cp.float64) - observed_mean) ** 2, dtype=cp.float64)))
    mismatch = validate_control_moments(
        target_mean=float(source_moment["mean"]),
        target_variance=float(source_moment["variance"]),
        observed_mean=observed_mean,
        observed_variance=observed_variance,
        scale=scale,
        offset=offset,
    )
    return x, words, table, {
        "seed_u64": seed, "pre_affine_bf16_mean": zmean,
        "pre_affine_bf16_variance": zvar, "scale": scale, "offset": offset,
        "observed_mean": observed_mean, "observed_variance": observed_variance,
        "target_mean": float(source_moment["mean"]),
        "target_variance": float(source_moment["variance"]),
        **mismatch,
    }


def build_panel(label: str, cp: Any, *, bindings: dict[str, Any] | None = None, source_dir: Path | None = None, source_moments: Sequence[dict[str, float]] | None = None, replica: int | None = None) -> dict[str, Any]:
    matrices: list[list[dict[str, Any]]] = []
    moments: list[dict[str, Any]] = []
    uniform_components: dict[str, list[Component]] = {}
    receipts: list[dict[str, Any]] = []
    for expert in range(EXPERTS):
        if bindings is not None:
            pair_entries = bindings["files"][2 * expert : 2 * expert + 2]
            pair_x, pair_words, pair_receipts = load_qwen_pair(source_dir, pair_entries, cp)  # type: ignore[arg-type]
            pair_tables = [BF16_TABLE, BF16_TABLE]
            receipts.extend(pair_receipts)
            control_meta = [None, None]
        else:
            pair_x, pair_words, pair_tables, control_meta = [], [], [], []
            for role in range(ROLES):
                ordinal = 2 * expert + role
                x, words, table, meta = make_control_matrix(source_moments[ordinal], int(replica), ordinal, cp)  # type: ignore[index]
                pair_x.append(x); pair_words.append(words); pair_tables.append(table); control_meta.append(meta)
        pair_masks: list[dict[str, Any]] = []
        for role in range(ROLES):
            candidates, masks, moment = build_matrix_candidates(pair_x[role], pair_words[role], pair_tables[role], cp)
            matrices.append(candidates)
            moment["expert_ordinal"] = expert
            moment["role"] = ("up", "down_transposed")[role]
            if control_meta[role] is not None:
                moment["control_affine"] = control_meta[role]
            moments.append(moment)
            pair_masks.append(masks)
        pair_components = pair_xklt_components(pair_x[0], pair_x[1], pair_masks[0], pair_masks[1], expert, cp)
        for key, components in pair_components.items():
            uniform_components.setdefault(key, []).extend(components)
        del pair_x, pair_words, pair_masks
        cp.get_default_memory_pool().free_all_blocks()
    total_energy = float(sum(row["energy"] for row in moments))
    candidate_digest_rows = []
    for ordinal, candidates in enumerate(matrices):
        for c in candidates:
            candidate_digest_rows.append({
                "matrix": ordinal, "id": c["id"], "family": c["family"],
                "fraction": c["fraction"], "levels": c["levels"],
                "selected_scalars": c["selected_scalars"],
                "support_bits": c["support_bits"], "symbol_bits": c["symbol_bits"],
                "tail_energy": c["tail_energy"], "bulk_energy": c["bulk_energy"],
                "free_lloyd_sse": c["free_lloyd_sse"], "fp16_sse": c["fp16_sse"],
            })
    return {
        "label": label, "matrices": matrices, "moments": moments,
        "uniform_components": uniform_components, "total_energy": total_energy,
        "source_receipts": receipts,
        "candidate_ledger_sha256": hashlib.sha256(canonical_json_bytes(candidate_digest_rows)).hexdigest(),
        "candidate_count": len(candidate_digest_rows),
    }


def raw_components(panel: dict[str, Any], choices: Sequence[int]) -> list[Component]:
    rows = []
    for ordinal, choice in enumerate(choices):
        c = panel["matrices"][ordinal][choice]
        if c["bulk_dimension"] > 0 and c["bulk_energy"] > 0.0:
            rows.append(Component(f"matrix_{ordinal:02d}.bulk", ordinal // ROLES, int(c["bulk_dimension"]), float(c["bulk_energy"])))
    return rows


def score(panel: dict[str, Any], choices: Sequence[int], rate: float, mode: str, geometry: str, *, include_allocations: bool = False) -> dict[str, Any]:
    choices = tuple(int(x) for x in choices)
    selected = [panel["matrices"][i][choice] for i, choice in enumerate(choices)]
    if geometry == "raw":
        components = raw_components(panel, choices)
        angle_by_expert = [0] * EXPERTS
    elif geometry == "support_xklt_uniform":
        if len(set(choices)) != 1:
            raise ValueError("support XKLT grid is uniform")
        key = support_key(selected[0])
        components = panel["uniform_components"][key]
        angle_by_expert = [0] * EXPERTS
        for component in components:
            if ".both_axis_" in component.name:
                angle_by_expert[component.owner] = 1
    else:
        raise ValueError(geometry)
    if not components:
        return {"valid": False, "reason": "no residual components"}
    component_counts = [0] * EXPERTS
    for component in components:
        component_counts[component.owner] += 1
    side_by_expert: list[int] = []
    for expert in range(EXPERTS):
        bits = EXPERT_HEADER_BITS + ROLES * MATRIX_DESCRIPTOR_BITS
        bits += component_counts[expert] * RESIDUAL_DIRECTORY_BITS
        bits += angle_by_expert[expert] * ANGLE_BITS
        for role in range(ROLES):
            bits += candidate_side_bits(selected[2 * expert + role], mode)
        side_by_expert.append(bits)
    capacity_bytes = int(math.floor(rate * PANEL_N / 8.0))
    physical_bits = capacity_bytes * 8
    side_bits = COMMON_BITS + sum(side_by_expert)
    payload_bits = physical_bits - side_bits - END_PAD_RESERVE_BITS
    if payload_bits <= 0:
        return {"valid": False, "reason": "side exceeds capacity"}
    waterfill = integer_waterfill(components, payload_bits)
    payload_by_expert = [0] * EXPERTS
    for allocation in waterfill["allocations"]:
        payload_by_expert[int(allocation["owner_expert"])] += int(allocation["payload_bits"])
    frame_bits, end_pad_bits = [], []
    for expert in range(EXPERTS):
        unpadded = side_by_expert[expert] + payload_by_expert[expert]
        pad = (-unpadded) % 8
        frame_bits.append(unpadded + pad)
        end_pad_bits.append(pad)
    closure = COMMON_BITS + sum(frame_bits)
    trailer_bits = physical_bits - closure
    if trailer_bits < 0:
        raise RuntimeError("container overflow")
    tail_sse = float(sum(candidate_tail_sse(c, mode) for c in selected))
    bulk_sse = float(waterfill["distortion_sse"])
    source_energy = float(panel["total_energy"])
    for label, value in (
        ("tail distortion", tail_sse), ("bulk distortion", bulk_sse),
        ("source energy", source_energy),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"non-finite or negative {label}")
    if source_energy <= 0.0:
        raise RuntimeError("source energy must be finite and positive")
    total_sse = tail_sse + bulk_sse
    if not math.isfinite(total_sse) or total_sse <= 0.0:
        raise RuntimeError("total distortion must be finite and positive")
    mse = total_sse / source_energy
    actual_rate = physical_bits / PANEL_N
    f_value = mse * 2.0 ** (2.0 * actual_rate)
    s_value = -0.5 * math.log2(f_value)
    if not all(math.isfinite(value) for value in (mse, actual_rate, f_value, s_value)) or mse <= 0.0 or f_value <= 0.0:
        raise RuntimeError("non-finite score aggregate")

    common_bytes = COMMON_BITS // 8
    common_pages = set(range((common_bytes + PAGE_BYTES - 1) // PAGE_BYTES))
    reference_bytes = capacity_bytes / EXPERTS
    offset = common_bytes
    expert_reads = []
    for expert, bits in enumerate(frame_bits):
        size = bits // 8
        start, end = offset, offset + size
        pages = set(range(start // PAGE_BYTES, (end - 1) // PAGE_BYTES + 1)) if size else set()
        page_bytes = len(common_pages | pages) * PAGE_BYTES
        logical_bytes = common_bytes + size
        logical_amplification = logical_bytes / reference_bytes
        page_amplification = page_bytes / reference_bytes
        if not math.isfinite(logical_amplification) or not math.isfinite(page_amplification):
            raise RuntimeError("non-finite read amplification")
        expert_reads.append({
            "expert_ordinal": expert, "frame_offset_bytes": start,
            "frame_bytes": size, "frame_end_pad_bits": end_pad_bits[expert],
            "residual_payload_bits": payload_by_expert[expert],
            "cold_logical_bytes": logical_bytes,
            "cold_logical_amplification": logical_amplification,
            "cold_page_bytes": page_bytes,
            "cold_page_amplification": page_amplification,
        })
        offset = end
    row: dict[str, Any] = {
        "valid": True, "panel": panel["label"], "mode": mode, "geometry": geometry,
        "requested_rate_bpw": rate, "physical_rate_bpw": actual_rate,
        "capacity_bytes": capacity_bytes, "physical_bits": physical_bits,
        "choices": list(choices),
        "profiles": [{k: c[k] for k in ("id", "family", "unit", "fraction", "levels", "selected_units", "selected_scalars", "support_bits", "support_stream_bits", "symbol_bits", "symbol_stream_bits", "codebook_bits", "tail_energy", "bulk_energy", "free_lloyd_sse", "fp16_sse", "centroids", "fp16_centroids")} for c in selected],
        "peeled_scalars": int(sum(c["selected_scalars"] for c in selected)),
        "peeled_fraction": float(sum(c["selected_scalars"] for c in selected) / PANEL_N),
        "peeled_energy_fraction": float(sum(c["tail_energy"] for c in selected) / panel["total_energy"]),
        "tail_distortion_sse": tail_sse,
        "bulk_ideal_distortion_sse": float(waterfill["distortion_sse"]),
        "total_distortion_sse": total_sse,
        "source_energy": source_energy,
        "ideal_relative_mse": mse, "F": f_value,
        "s_bpw": s_value,
        "gaussian_limit_mse": 2.0 ** (-2.0 * actual_rate),
        "target_mse": TARGET_F * 2.0 ** (-2.0 * actual_rate),
        "passes_absolute_F": bool(f_value <= TARGET_F),
        "side_ledger": {
            "common_bits": COMMON_BITS, "expert_side_bits": side_by_expert,
            "tail_and_codebook_bits": int(sum(candidate_side_bits(c, mode) for c in selected)),
            "residual_directory_bits": len(components) * RESIDUAL_DIRECTORY_BITS,
            "xklt_angle_bits": sum(angle_by_expert) * ANGLE_BITS,
            "end_pad_reserve_bits": END_PAD_RESERVE_BITS,
            "actual_end_pad_bits": sum(end_pad_bits), "trailer_bits": trailer_bits,
            "payload_bits": payload_bits, "bit_closure": closure + trailer_bits,
        },
        "component_count": len(components),
        "read_ledger": {
            "reference_one_sixth_container_bytes": reference_bytes,
            "common_prefix_bytes": common_bytes, "experts": expert_reads,
            "maximum_cold_logical_amplification": max(r["cold_logical_amplification"] for r in expert_reads),
            "maximum_cold_page_amplification": max(r["cold_page_amplification"] for r in expert_reads),
            "below_2x": all(r["cold_logical_amplification"] < 2.0 and r["cold_page_amplification"] < 2.0 for r in expert_reads),
        },
    }
    if include_allocations:
        row["allocations"] = waterfill["allocations"]
    return row


def finite_scalar(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"invalid numeric scalar for {label}") from exc
    if not math.isfinite(converted):
        raise RuntimeError(f"non-finite numeric scalar for {label}")
    return converted


def validate_score_row(row: dict[str, Any], label: str) -> None:
    if row.get("valid") is not True:
        raise RuntimeError(f"{label} is not a valid scored row")
    for key in (
        "requested_rate_bpw", "physical_rate_bpw", "ideal_relative_mse", "F",
        "s_bpw", "source_energy", "tail_distortion_sse",
        "bulk_ideal_distortion_sse", "total_distortion_sse",
    ):
        finite_scalar(row.get(key), f"{label}.{key}")
    read = row.get("read_ledger")
    if not isinstance(read, dict) or not isinstance(read.get("below_2x"), bool):
        raise RuntimeError(f"invalid read ledger for {label}")
    maximum_logical = finite_scalar(
        read.get("maximum_cold_logical_amplification"),
        f"{label}.maximum_cold_logical_amplification",
    )
    maximum_page = finite_scalar(
        read.get("maximum_cold_page_amplification"),
        f"{label}.maximum_cold_page_amplification",
    )
    experts = read.get("experts")
    if not isinstance(experts, list) or len(experts) != EXPERTS:
        raise RuntimeError(f"invalid expert read ledger for {label}")
    for ordinal, expert in enumerate(experts):
        finite_scalar(expert.get("cold_logical_amplification"), f"{label}.expert[{ordinal}].logical")
        finite_scalar(expert.get("cold_page_amplification"), f"{label}.expert[{ordinal}].page")
    recomputed = maximum_logical < 2.0 and maximum_page < 2.0 and all(
        finite_scalar(expert["cold_logical_amplification"], "expert logical") < 2.0
        and finite_scalar(expert["cold_page_amplification"], "expert page") < 2.0
        for expert in experts
    )
    if read["below_2x"] != recomputed:
        raise RuntimeError(f"contradictory read-valid flag for {label}")


def score_order(row: dict[str, Any]) -> tuple[Any, ...]:
    validate_score_row(row, "selection row")
    side_bits = row.get("side_ledger", {}).get("tail_and_codebook_bits")
    if not isinstance(side_bits, int) or isinstance(side_bits, bool) or side_bits < 0:
        raise RuntimeError("invalid selection side-bit scalar")
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) != MATRICES or not all(
        isinstance(choice, int) and not isinstance(choice, bool) for choice in choices
    ):
        raise RuntimeError("invalid selection choices")
    return (finite_scalar(row["F"], "selection F"), side_bits, tuple(choices))


def best_scored(rows: Sequence[dict[str, Any]], label: str, *, require_read_valid: bool) -> dict[str, Any]:
    candidates = []
    for ordinal, row in enumerate(rows):
        if row.get("valid") is not True:
            continue
        validate_score_row(row, f"{label}[{ordinal}]")
        if not require_read_valid or row["read_ledger"]["below_2x"]:
            candidates.append(row)
    if not candidates:
        suffix = " read-valid" if require_read_valid else ""
        raise RuntimeError(f"no{suffix} candidate for {label}")
    return min(candidates, key=score_order)


def eligible_profiles(mode: str) -> list[int]:
    if mode == "zero_tail_error":
        return [i for i, p in enumerate(PROFILES) if p["levels"] == 1]
    return list(range(len(PROFILES)))


def search_panel(panel: dict[str, Any], maximum_passes: int = 4) -> dict[str, Any]:
    if not isinstance(maximum_passes, int) or isinstance(maximum_passes, bool) or maximum_passes != 4:
        raise RuntimeError("v7 search requires exactly four maximum coordinate passes")
    output: dict[str, Any] = {}
    for rate in RATES:
        rate_rows: dict[str, Any] = {}
        for mode in MODES:
            eligible = eligible_profiles(mode)
            uniform = [score(panel, [idx] * MATRICES, rate, mode, "raw") for idx in eligible]
            if len(uniform) != len(eligible):
                raise RuntimeError("uniform candidate coverage mismatch")
            best_uniform_global = best_scored(uniform, "raw uniform global", require_read_valid=False)
            best_uniform = best_scored(uniform, "raw uniform read-valid", require_read_valid=True)
            zero_rows = [row for idx, row in zip(eligible, uniform, strict=True) if idx == 0 and row.get("valid") is True and row.get("read_ledger", {}).get("below_2x") is True]
            seeds = [tuple(best_uniform["choices"])]
            if zero_rows and tuple(zero_rows[0]["choices"]) not in seeds:
                seeds.append(tuple(zero_rows[0]["choices"]))
            best = best_uniform
            passes_used = 0
            coordinate_trial_rows_evaluated = 0
            for seed in seeds:
                choices = list(seed)
                current = score(panel, choices, rate, mode, "raw")
                validate_score_row(current, "coordinate seed")
                if not current["read_ledger"]["below_2x"]:
                    raise RuntimeError("coordinate seed is not read-valid")
                for pass_index in range(maximum_passes):
                    changed = False
                    for ordinal in range(MATRICES):
                        trials: list[dict[str, Any]] = []
                        covered: list[int] = []
                        for idx in eligible:
                            trial = choices.copy(); trial[ordinal] = idx
                            trials.append(score(panel, trial, rate, mode, "raw"))
                            covered.append(idx)
                        coordinate_trial_rows_evaluated += len(trials)
                        if covered != eligible or len(trials) != len(eligible):
                            raise RuntimeError("coordinate candidate coverage mismatch")
                        winner = best_scored(
                            trials,
                            f"raw coordinate read-valid pass={pass_index} ordinal={ordinal}",
                            require_read_valid=True,
                        )
                        if winner["choices"][ordinal] != choices[ordinal]:
                            changed = True
                        choices = list(winner["choices"])
                        current = winner
                    passes_used = max(passes_used, pass_index + 1)
                    if not changed:
                        break
                if score_order(current) < score_order(best):
                    best = current
            best = score(panel, best["choices"], rate, mode, "raw", include_allocations=True)
            validate_score_row(best, "final raw read-valid winner")
            if not best["read_ledger"]["below_2x"]:
                raise RuntimeError("final raw winner is not read-valid")
            uniform_xklt_rows = [score(panel, [idx] * MATRICES, rate, mode, "support_xklt_uniform") for idx in eligible]
            if len(uniform_xklt_rows) != len(eligible):
                raise RuntimeError("support-XKLT candidate coverage mismatch")
            best_xklt_global = best_scored(uniform_xklt_rows, "support-XKLT global", require_read_valid=False)
            best_xklt = best_scored(uniform_xklt_rows, "support-XKLT read-valid", require_read_valid=True)
            best_xklt = score(panel, best_xklt["choices"], rate, mode, "support_xklt_uniform", include_allocations=True)
            validate_score_row(best_xklt, "final support-XKLT read-valid winner")
            if not best_xklt["read_ledger"]["below_2x"]:
                raise RuntimeError("final support-XKLT winner is not read-valid")
            rate_rows[mode] = {
                "raw_adaptive": best,
                "raw_uniform_best": best_uniform,
                "support_xklt_uniform": best_xklt,
                "raw_uniform_global_diagnostic": best_uniform_global,
                "support_xklt_uniform_global_diagnostic": best_xklt_global,
                "eligible_profile_count": len(eligible),
                "coordinate_passes_max_used": passes_used,
                "read_valid_selection_ledger": {
                    "uniform_rows_evaluated": len(uniform),
                    "support_xklt_rows_evaluated": len(uniform_xklt_rows),
                    "coordinate_trial_rows_evaluated": coordinate_trial_rows_evaluated,
                    "seed_count": len(seeds),
                    "every_uniform_profile_evaluated": len(uniform) == len(eligible),
                    "every_support_xklt_profile_evaluated": len(uniform_xklt_rows) == len(eligible),
                    "every_coordinate_trial_profile_evaluated": coordinate_trial_rows_evaluated >= len(seeds) * MATRICES * len(eligible),
                    "retained_rows_below_2x": True,
                },
            }
        output[f"{rate:.2f}"] = rate_rows
    return output


def all_best_rows(search: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for rate_rows in search.values():
        for mode_rows in rate_rows.values():
            yield mode_rows["raw_adaptive"]
            yield mode_rows["support_xklt_uniform"]


def calibrate(qwen: dict[str, Any], controls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(controls) != 4:
        raise RuntimeError("calibration requires exactly four controls")
    rows = []
    for rate in RATES:
        rate_key = f"{rate:.2f}"
        for mode in MODES:
            for geometry_key in ("raw_adaptive", "support_xklt_uniform"):
                source = qwen[rate_key][mode][geometry_key]
                validate_score_row(source, f"Qwen calibration {rate_key}/{mode}/{geometry_key}")
                if not source["read_ledger"]["below_2x"]:
                    raise RuntimeError("calibration source row is not read-valid")
                control_rows = [c[rate_key][mode][geometry_key] for c in controls]
                for ordinal, control in enumerate(control_rows):
                    validate_score_row(control, f"control[{ordinal}] {rate_key}/{mode}/{geometry_key}")
                    if not control["read_ledger"]["below_2x"]:
                        raise RuntimeError("calibration control row is not read-valid")
                source_s = finite_scalar(source["s_bpw"], "Qwen calibration s")
                source_f = finite_scalar(source["F"], "Qwen calibration F")
                control_s = [finite_scalar(c["s_bpw"], f"control[{ordinal}] s") for ordinal, c in enumerate(control_rows)]
                control_f = [finite_scalar(c["F"], f"control[{ordinal}] F") for ordinal, c in enumerate(control_rows)]
                control_mean = finite_scalar(np.mean(control_s), "control mean s")
                control_std = finite_scalar(np.std(control_s, ddof=1), "control sample std s")
                excess = finite_scalar(source_s - control_mean, "Qwen excess s")
                calibrated_f = 2.0 ** (-2.0 * excess)
                calibrated_f = finite_scalar(calibrated_f, "calibrated F")
                fraction_required = finite_scalar(excess / TARGET_S, "fraction of required s")
                rows.append({
                    "rate": rate, "mode": mode, "geometry": geometry_key,
                    "qwen_F": source_f, "qwen_s_bpw": source_s,
                    "control_F": control_f,
                    "control_s_bpw": control_s,
                    "control_mean_s_bpw": control_mean,
                    "control_sample_std_s_bpw": control_std,
                    "qwen_excess_s_bpw": excess, "calibrated_F": calibrated_f,
                    "fraction_of_required_s": fraction_required,
                    "passes_calibrated_F": bool(calibrated_f <= TARGET_F),
                    "passes_absolute_F": bool(source_f <= TARGET_F),
                    "below_2x": True,
                })
    return rows


def require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        finite_scalar(value, label)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for ordinal, item in enumerate(value):
            require_finite_tree(item, f"{label}[{ordinal}]")
        return
    raise RuntimeError(f"unexpected decision value type for {label}")


def decision_from_calibrated(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError("decision row must be a dictionary")
        require_finite_tree(row, f"decision row[{ordinal}]")
        if not isinstance(row.get("below_2x"), bool):
            raise RuntimeError("decision below_2x flag must be boolean")
        if row.get("mode") not in MODES:
            raise RuntimeError("decision mode drift")
    read_valid = [row for row in rows if row["below_2x"]]
    optimistic = [row for row in read_valid if row["mode"] in ("free_lloyd", "zero_tail_error")]
    finite = [row for row in read_valid if row["mode"] == "finite_fp16"]
    if not optimistic or not finite:
        raise RuntimeError("decision requires read-valid optimistic and finite rows")

    def joint(row: dict[str, Any]) -> float:
        absolute = finite_scalar(row["qwen_s_bpw"], "absolute decision score")
        calibrated = finite_scalar(row["qwen_excess_s_bpw"], "calibrated decision score")
        return finite_scalar(min(absolute, calibrated), "joint decision score")

    def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (joint(row), -finite_scalar(row["rate"], "decision rate"), str(row["mode"]), str(row["geometry"]))

    best_optimistic = max(optimistic, key=selection_key)
    best_finite = max(finite, key=selection_key)
    optimistic_m = joint(best_optimistic)
    finite_joint = joint(best_finite)
    finite_absolute = finite_scalar(best_finite["qwen_s_bpw"], "best finite absolute s")
    finite_calibrated = finite_scalar(best_finite["qwen_excess_s_bpw"], "best finite calibrated s")
    if finite_joint > optimistic_m + DECISION_CONSISTENCY_EPSILON_S:
        raise RuntimeError("finite joint score exceeds optimistic envelope")

    boundary_values = {
        "optimistic_m": optimistic_m,
        "finite_best_absolute_s_bpw": finite_absolute,
        "finite_best_calibrated_s_bpw": finite_calibrated,
    }
    boundary_distances = {
        label: {
            "to_kill_threshold_s_bpw": abs(value - KILL_THRESHOLD_S),
            "to_target_s_bpw": abs(value - TARGET_S),
        }
        for label, value in boundary_values.items()
    }
    require_finite_tree(boundary_values, "decision boundary values")
    require_finite_tree(boundary_distances, "decision boundary distances")
    numeric_boundary = any(
        distance <= NUMERIC_BOUNDARY_GUARD_S
        for distances in boundary_distances.values()
        for distance in distances.values()
    )
    would_promote = finite_absolute >= TARGET_S and finite_calibrated >= TARGET_S
    if would_promote and optimistic_m < TARGET_S - DECISION_CONSISTENCY_EPSILON_S:
        raise RuntimeError("finite promotion contradicts optimistic envelope")

    if numeric_boundary:
        status = "HOLD_NUMERIC_BOUNDARY"
        finite_warranted = False
        early_kill = False
    elif would_promote:
        status = "FINITE_CODEC_WARRANTED"
        finite_warranted = True
        early_kill = False
    elif optimistic_m < KILL_THRESHOLD_S:
        status = "EARLY_KILL_FAR_SHORT"
        finite_warranted = False
        early_kill = True
    elif optimistic_m < TARGET_S:
        status = "HOLD_OPTIMISTIC_NEAR_BOUNDARY"
        finite_warranted = False
        early_kill = False
    else:
        status = "OPTIMISTIC_SURVIVOR"
        finite_warranted = False
        early_kill = False

    return {
        "status": status,
        "target_F": TARGET_F,
        "required_s_bpw": TARGET_S,
        "optimistic_kill_guard_s_bpw": KILL_GUARD_S,
        "optimistic_kill_threshold_s_bpw": KILL_THRESHOLD_S,
        "numeric_boundary_guard_s_bpw": NUMERIC_BOUNDARY_GUARD_S,
        "optimistic_m_s_bpw": optimistic_m,
        "best_optimistic_envelope": best_optimistic,
        "finite_best_joint_s_bpw": finite_joint,
        "finite_best_row": best_finite,
        "boundary_values_s_bpw": boundary_values,
        "boundary_distances_s_bpw": boundary_distances,
        "finite_residual_codec_warranted": finite_warranted,
        "early_kill": early_kill,
    }


def exact_runtime_tuple(cp: Any) -> dict[str, Any]:
    device = cp.cuda.runtime.getDeviceProperties(0)

    def decode(value: Any) -> Any:
        return value.decode() if isinstance(value, bytes) else value

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__,
        "cuda_runtime_integer": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_integer": int(cp.cuda.runtime.driverGetVersion()),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "device_ordinal": 0,
        "device_name": decode(device["name"]),
        "device_total_global_mem": int(device["totalGlobalMem"]),
        "device_compute_capability": [int(device["major"]), int(device["minor"])],
        "device_multiprocessor_count": int(device["multiProcessorCount"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def runtime_probe(cp: Any) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("runtime probe requires explicit CUDA_VISIBLE_DEVICES=0")
    runtime = exact_runtime_tuple(cp)
    if runtime["device_count"] != 1 or runtime["device_ordinal"] != 0:
        raise RuntimeError("runtime probe requires exactly visible device ordinal zero")
    affine_sentinels = (
        (float.fromhex("0x1.4000000000000p-2"), float.fromhex("-0x1.8000000000000p-7")),
        (float.fromhex("0x1.8000000000000p+0"), float.fromhex("0x1.0000000000000p-8")),
    )
    cells: list[dict[str, Any]] = []
    for replica in range(4):
        for ordinal in range(MATRICES):
            seed = stable_seed("lossy-tail-peeling-gaussian-v1", replica, ordinal)
            rng = cp.random.default_rng(seed)
            raw = rng.standard_normal(N, dtype=cp.float32)
            raw_host = np.ascontiguousarray(cp.asnumpy(raw.view(cp.uint32)), dtype="<u4")
            bits = raw.view(cp.uint32)
            rounding = cp.uint32(0x7FFF) + ((bits >> cp.uint32(16)) & cp.uint32(1))
            words = ((bits + rounding) >> cp.uint32(16)).astype(cp.uint16)
            words_host = np.ascontiguousarray(cp.asnumpy(words), dtype="<u2")
            zbf = (words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
            zmean = float(cp.asnumpy(cp.mean(zbf, dtype=cp.float64)))
            zvar = float(cp.asnumpy(cp.mean((zbf.astype(cp.float64) - zmean) ** 2, dtype=cp.float64)))
            table_words = cp.arange(1 << 16, dtype=cp.uint16)
            table = (table_words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
            affine_rows = []
            for scale, offset in affine_sentinels:
                affine_table = table * cp.float32(scale) + cp.float32(offset)
                gathered = affine_table[words]
                gathered_host = np.ascontiguousarray(cp.asnumpy(gathered.view(cp.uint32)), dtype="<u4")
                mean = float(cp.asnumpy(cp.mean(gathered, dtype=cp.float64)))
                variance = float(cp.asnumpy(cp.mean((gathered.astype(cp.float64) - mean) ** 2, dtype=cp.float64)))
                affine_rows.append({
                    "scale_float32_hex": float(np.float32(scale)).hex(),
                    "offset_float32_hex": float(np.float32(offset)).hex(),
                    "gathered_float32_u32_little_endian_sha256": hashlib.sha256(gathered_host.tobytes()).hexdigest(),
                    "float64_mean_hex": mean.hex(),
                    "float64_population_variance_hex": variance.hex(),
                })
            cells.append({
                "replica": replica,
                "ordinal": ordinal,
                "seed_u64": seed,
                "raw_float32_u32_little_endian_sha256": hashlib.sha256(raw_host.tobytes()).hexdigest(),
                "rne_bf16_u16_little_endian_sha256": hashlib.sha256(words_host.tobytes()).hexdigest(),
                "float64_mean_of_bf16_values_hex": zmean.hex(),
                "float64_population_variance_hex": zvar.hex(),
                "affine_sentinels": affine_rows,
            })
            # Every consumer is reduced/hashed before this deletion barrier.
            # No per-cell GPU array may remain live when the pool is freed.
            del (
                raw, bits, rounding, words, zbf, table_words, table,
                affine_table, gathered, rng,
            )
            del raw_host, words_host, gathered_host
            cp.cuda.get_current_stream().synchronize()
            pool = cp.get_default_memory_pool()
            used_before_free = int(pool.used_bytes())
            total_before_free = int(pool.total_bytes())
            if used_before_free != 0:
                raise RuntimeError(
                    f"runtime probe leaked live pool bytes before free at replica={replica} "
                    f"ordinal={ordinal}: {used_before_free}"
                )
            pool.free_all_blocks()
            used_after_free = int(pool.used_bytes())
            total_after_free = int(pool.total_bytes())
            if used_after_free != 0 or total_after_free != 0:
                raise RuntimeError(
                    f"runtime probe pool did not close at replica={replica} ordinal={ordinal}: "
                    f"used={used_after_free} total={total_after_free}"
                )
            cells[-1]["memory_release"] = {
                "stream_synchronized": True,
                "used_bytes_before_free": used_before_free,
                "total_bytes_before_free": total_before_free,
                "used_bytes_after_free": used_after_free,
                "total_bytes_after_free": total_after_free,
                "all_per_cell_gpu_arrays_deleted_before_free": True,
            }
            del pool

    adversaries = [
        np.asarray([3, 3, 2, 3, 2, 3], dtype=np.float32),
        np.asarray([0.0, -0.0, 0.0, -0.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([7] * 257, dtype=np.int64),
        np.asarray([2, 1, 2, 1] * 257, dtype=np.float64),
        np.repeat(np.asarray([5, 4, 3, 2, 1], dtype=np.float32), [513, 257, 129, 65, 33]),
    ]
    order_rows = []
    for ordinal, host in enumerate(adversaries):
        expected = np.lexsort((np.arange(host.size), -host))
        device_host = cp.asarray(host)
        device_order = stable_descending(device_host, cp)
        observed = cp.asnumpy(device_order)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"runtime stable-order mismatch at adversary {ordinal}")
        order_row = {
            "ordinal": ordinal,
            "dtype": str(host.dtype),
            "count": int(host.size),
            "input_sha256": hashlib.sha256(np.ascontiguousarray(host).tobytes()).hexdigest(),
            "order_sha256": hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest(),
        }
        del device_host, device_order
        cp.cuda.get_current_stream().synchronize()
        pool = cp.get_default_memory_pool()
        used_before_free = int(pool.used_bytes())
        if used_before_free != 0:
            raise RuntimeError(f"stable-order probe leaked live pool bytes at adversary {ordinal}")
        pool.free_all_blocks()
        if int(pool.used_bytes()) != 0 or int(pool.total_bytes()) != 0:
            raise RuntimeError(f"stable-order probe pool did not close at adversary {ordinal}")
        order_row["memory_release"] = {
            "stream_synchronized": True,
            "used_bytes_before_free": used_before_free,
            "used_bytes_after_free": int(pool.used_bytes()),
            "total_bytes_after_free": int(pool.total_bytes()),
        }
        order_rows.append(order_row)
        del pool
    core = {"runtime_tuple": runtime, "cells": cells, "stable_order": order_rows}
    core["probe_aggregate_sha256"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    return core


def verify_internal_seal(value: dict[str, Any], field: str, expected: str, label: str) -> None:
    if value.get(field) != expected:
        raise RuntimeError(f"{label} internal seal field mismatch")
    copy = dict(value)
    copy.pop(field)
    if hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != expected:
        raise RuntimeError(f"{label} internal seal recomputation mismatch")


def read_external_json(path_value: str, expected_file_sha256: str, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    path = require_canonical_original_path(path_value, label, allow_missing_tail=False)
    payload = read_regular_descriptor(path, label)
    if hashlib.sha256(payload).hexdigest() != expected_file_sha256:
        raise RuntimeError(f"{label} file SHA-256 mismatch")
    return path, payload, strict_json_bytes(payload, label)


def validate_external_audit(section: dict[str, Any], label: str) -> tuple[Path, Path]:
    require_exact_keys(section, {
        "manifest_path", "manifest_file_sha256", "receipt_path", "receipt_file_sha256",
        "receipt_internal_field", "receipt_internal_sha256", "required_status",
    }, label)
    manifest = require_canonical_original_path(section["manifest_path"], f"{label} manifest", allow_missing_tail=False)
    manifest_payload = read_regular_descriptor(manifest, f"{label} manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != section["manifest_file_sha256"]:
        raise RuntimeError(f"{label} manifest SHA-256 mismatch")
    receipt, _, value = read_external_json(section["receipt_path"], section["receipt_file_sha256"], f"{label} receipt")
    if value.get("status") != section["required_status"]:
        raise RuntimeError(f"{label} required status mismatch")
    verify_internal_seal(value, section["receipt_internal_field"], section["receipt_internal_sha256"], f"{label} receipt")
    return manifest, receipt


def validate_runtime_receipt(section: dict[str, Any], runtime_contract_sha256: str) -> tuple[Path, dict[str, Any]]:
    require_exact_keys(section, {
        "path", "file_sha256", "internal_sha256", "required_status", "runtime_contract_file_sha256",
    }, "runtime receipt authorization")
    if section["runtime_contract_file_sha256"] != runtime_contract_sha256:
        raise RuntimeError("authorized runtime-contract identity mismatch")
    path, _, receipt = read_external_json(section["path"], section["file_sha256"], "runtime receipt")
    if receipt.get("schema") != "lossy-tail-v7-source-free-runtime-receipt-v1":
        raise RuntimeError("runtime receipt schema mismatch")
    if receipt.get("status") != section["required_status"]:
        raise RuntimeError("runtime receipt status mismatch")
    verify_internal_seal(receipt, "runtime_receipt_sha256", section["internal_sha256"], "runtime receipt")
    if receipt.get("runtime_contract", {}).get("sha256") != runtime_contract_sha256:
        raise RuntimeError("runtime receipt does not bind runtime contract")
    access = receipt.get("access_ledger", {})
    required_zero = (
        "model_or_qwen_paths_supplied", "model_or_qwen_paths_opened",
        "payload_files_opened", "production_results_opened",
    )
    if any(access.get(key) != 0 for key in required_zero):
        raise RuntimeError("runtime receipt is not source-free")
    return path, receipt


def validate_production_authorization(
    authorization_path_value: str,
    authorization_file_sha256: str,
    *, stage: Path,
    launcher: Path,
    manifest_sha256: str,
    runtime_contract_sha256: str,
    bindings_sha256: str,
    repair_internal_sha256: str,
) -> dict[str, Any]:
    authorization_path, _, authorization = read_external_json(
        authorization_path_value, authorization_file_sha256, "production authorization"
    )
    require_exact_keys(authorization, {
        "schema", "status", "authorization_path", "authorization_nonce", "action",
        "stage", "source", "output", "source_audit", "runtime_receipt",
        "runtime_audit", "execution", "filesystem", "fixed_scientific_arguments",
        "authorization_sha256",
    }, "production authorization")
    if authorization["schema"] != "lossy-tail-v7-one-shot-production-authorization-v1":
        raise RuntimeError("production authorization schema mismatch")
    if authorization["status"] != "AUTHORIZED_ONCE_AFTER_INDEPENDENT_SOURCE_AND_RUNTIME_AUDITS":
        raise RuntimeError("production authorization status mismatch")
    if authorization["action"] != "CREATE_NEW_RUN_ROOT_AND_RESULT_JSON":
        raise RuntimeError("production authorization action mismatch")
    if not isinstance(authorization["authorization_nonce"], str) or not authorization["authorization_nonce"]:
        raise RuntimeError("production authorization nonce missing")
    if authorization["authorization_path"] != os.fspath(authorization_path):
        raise RuntimeError("production authorization copied or path mismatch")
    verify_internal_seal(
        authorization, "authorization_sha256", authorization["authorization_sha256"], "production authorization"
    )

    stage_row = authorization["stage"]
    require_exact_keys(stage_row, {"path", "launch_manifest_file_sha256", "launch_manifest_internal_stage_member_count"}, "authorized stage")
    if stage_row != {
        "path": os.fspath(stage),
        "launch_manifest_file_sha256": manifest_sha256,
        "launch_manifest_internal_stage_member_count": len(STAGE_MEMBERS_V7),
    }:
        raise RuntimeError("authorized stage mismatch")
    source_row = authorization["source"]
    require_exact_keys(source_row, {"path", "bindings_file_sha256"}, "authorized source")
    if source_row["bindings_file_sha256"] != bindings_sha256:
        raise RuntimeError("authorized source-binding mismatch")
    source = require_canonical_original_path(source_row["path"], "authorized source", allow_missing_tail=False)
    if not source.is_dir():
        raise RuntimeError("authorized source is not a directory")

    output_row = authorization["output"]
    require_exact_keys(output_row, {"run_root", "result_path"}, "authorized output")
    run_root = require_canonical_original_path(output_row["run_root"], "authorized run root", allow_missing_tail=True)
    result = require_canonical_original_path(output_row["result_path"], "authorized result", allow_missing_tail=True)
    if result != run_root / "result.json":
        raise RuntimeError("authorized result must be exactly run_root/result.json")
    if run_root.exists() or result.exists() or not run_root.parent.is_dir():
        raise RuntimeError("authorized run root/result must be absent below an existing parent")

    source_audit_manifest, source_audit_receipt = validate_external_audit(authorization["source_audit"], "source audit")
    source_audit_value = strict_json_bytes(read_regular_descriptor(source_audit_receipt, "source audit receipt replay"), "source audit receipt replay")
    audited_target = source_audit_value.get("audited_target", {})
    if audited_target.get("launch_manifest_sha256") != manifest_sha256:
        raise RuntimeError("source audit does not bind authorized launch manifest")
    if audited_target.get("repair_lock_internal_sha256") != repair_internal_sha256:
        raise RuntimeError("source audit does not bind authorized repair lock")
    source_access = source_audit_value.get("access_ledger", {})
    if any(source_access.get(key) != 0 for key in ("model_payload_files_opened", "cupy_imports", "cuda_initializations", "gpu_jobs")):
        raise RuntimeError("source audit is not source-only")
    runtime_receipt_path, runtime_receipt = validate_runtime_receipt(authorization["runtime_receipt"], runtime_contract_sha256)
    runtime_audit_manifest, runtime_audit_receipt = validate_external_audit(authorization["runtime_audit"], "runtime audit")
    runtime_audit_value = strict_json_bytes(read_regular_descriptor(runtime_audit_receipt, "runtime audit receipt replay"), "runtime audit receipt replay")
    if runtime_audit_value.get("audited_runtime_receipt", {}).get("file_sha256") != authorization["runtime_receipt"]["file_sha256"]:
        raise RuntimeError("runtime audit does not bind authorized runtime receipt file")
    if runtime_audit_value.get("audited_runtime_receipt", {}).get("internal_sha256") != authorization["runtime_receipt"]["internal_sha256"]:
        raise RuntimeError("runtime audit does not bind authorized runtime receipt internal seal")
    runtime_audit_access = runtime_audit_value.get("access_ledger", {})
    if any(runtime_audit_access.get(key) != 0 for key in ("model_payload_files_opened", "production_result_files_opened", "gpu_jobs")):
        raise RuntimeError("runtime audit exceeded source-free audit scope")

    execution = authorization["execution"]
    require_exact_keys(execution, {
        "python_executable", "raw_launcher_path", "cuda_visible_devices", "runtime_tuple",
    }, "authorized execution")
    if execution["python_executable"] != sys.executable:
        raise RuntimeError("authorized Python executable mismatch")
    if execution["raw_launcher_path"] != os.fspath(launcher):
        raise RuntimeError("authorized raw launcher mismatch")
    if execution["cuda_visible_devices"] != "0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("production requires explicitly authorized CUDA_VISIBLE_DEVICES=0")
    if execution["runtime_tuple"] != runtime_receipt.get("runtime_probe", {}).get("runtime_tuple"):
        raise RuntimeError("authorization runtime tuple does not equal audited receipt")
    fixed = authorization["fixed_scientific_arguments"]
    if fixed != {"control_replicates": 4, "maximum_coordinate_passes": 4}:
        raise RuntimeError("fixed scientific arguments mismatch")

    protected_roots = [("stage", stage), ("source", source), ("output_existing_parent", run_root.parent)]
    require_pairwise_disjoint(protected_roots)
    evidence = [
        ("authorization", authorization_path),
        ("source_audit_manifest", source_audit_manifest),
        ("source_audit_receipt", source_audit_receipt),
        ("runtime_receipt", runtime_receipt_path),
        ("runtime_audit_manifest", runtime_audit_manifest),
        ("runtime_audit_receipt", runtime_audit_receipt),
    ]
    for protected_label, protected_path in protected_roots:
        for evidence_label, evidence_path in evidence:
            if (
                path_contains(protected_path, evidence_path)
                or path_contains(evidence_path, protected_path)
                or path_contains(protected_path, evidence_path.parent)
                or path_contains(evidence_path.parent, protected_path)
            ):
                raise RuntimeError(f"{evidence_label} overlaps protected {protected_label}")

    filesystem = authorization["filesystem"]
    require_exact_keys(filesystem, {"mountinfo_path", "mountinfo_file_sha256", "identities"}, "authorized filesystem")
    if filesystem["mountinfo_path"] != "/proc/self/mountinfo":
        raise RuntimeError("mountinfo path mismatch")
    mount_payload, mount_rows = mount_snapshot()
    if hashlib.sha256(mount_payload).hexdigest() != filesystem["mountinfo_file_sha256"]:
        raise RuntimeError("live mountinfo differs from authorization")
    live_paths = {
        "stage": stage,
        "source": source,
        "output_existing_parent": run_root.parent,
        "authorization_parent": authorization_path.parent,
        "source_audit_manifest": source_audit_manifest,
        "source_audit_receipt": source_audit_receipt,
        "runtime_receipt": runtime_receipt_path,
        "runtime_audit_manifest": runtime_audit_manifest,
        "runtime_audit_receipt": runtime_audit_receipt,
    }
    identity_rows = filesystem["identities"]
    if not isinstance(identity_rows, list) or len(identity_rows) != len(live_paths):
        raise RuntimeError("filesystem identity cardinality mismatch")
    labels = [row.get("label") for row in identity_rows if isinstance(row, dict)]
    if len(labels) != len(identity_rows) or len(set(labels)) != len(labels) or set(labels) != set(live_paths):
        raise RuntimeError("filesystem identity labels mismatch")
    seen_inode: dict[tuple[int, int], str] = {}
    for row in identity_rows:
        require_exact_keys(row, {"label", "path", "st_dev", "st_ino", "mount_id"}, f"filesystem identity {row.get('label')}")
        live = live_paths[row["label"]]
        if row["path"] != os.fspath(live):
            raise RuntimeError(f"filesystem path mismatch for {row['label']}")
        metadata = os.stat(live, follow_symlinks=False)
        mount = mount_row_for(live, mount_rows)
        if (row["st_dev"], row["st_ino"], row["mount_id"]) != (metadata.st_dev, metadata.st_ino, mount["mount_id"]):
            raise RuntimeError(f"filesystem identity mismatch for {row['label']}")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen_inode and seen_inode[identity] != row["label"]:
            raise RuntimeError(f"filesystem alias identity: {seen_inode[identity]} and {row['label']}")
        seen_inode[identity] = row["label"]
    require_no_nested_mounts(stage, mount_rows, "stage")
    require_no_nested_mounts(source, mount_rows, "source")
    require_no_nested_mounts(run_root, mount_rows, "run root")
    return {
        "authorization": authorization,
        "authorization_path": authorization_path,
        "source": source,
        "run_root": run_root,
        "result": result,
        "runtime_receipt": runtime_receipt,
        "runtime_receipt_path": runtime_receipt_path,
        "mountinfo_sha256": filesystem["mountinfo_file_sha256"],
    }


def self_test() -> None:
    def require(condition: bool, label: str) -> None:
        if not condition:
            raise RuntimeError(f"v7 self-test failed: {label}")

    require(ceil_log2_binomial(5, 2) == 4, "binomial bit length")
    require(pad8(0) == 0 and pad8(1) == 8 and pad8(8) == 8, "byte padding")
    counts = np.zeros(1 << 16, dtype=np.int64)
    table = BF16_TABLE.copy()
    # Use known finite BF16 words for -1 and +1.
    neg = int(np.asarray([-1.0], dtype=np.float32).view(np.uint32)[0] >> 16)
    pos = int(np.asarray([1.0], dtype=np.float32).view(np.uint32)[0] >> 16)
    counts[neg] = 5; counts[pos] = 7
    q = weighted_lloyd(counts, table, 2)
    require(q["free_sse"] == 0.0 and q["fp16_sse"] == 0.0, "two-point Lloyd exactness")
    comps = [Component("a", 0, 10, 10.0), Component("b", 1, 10, 10.0)]
    wf = integer_waterfill(comps, 21)
    require(sum(x["payload_bits"] for x in wf["allocations"]) == 21, "waterfill closure")
    require(len(PROFILES) == 61, "profile closure")
    blank = {
        **PROFILES[0], "selected_units": 0, "selected_scalars": 0,
        "support_bits": 0, "support_stream_bits": 0,
        "symbol_bits": 0, "symbol_stream_bits": 0, "codebook_bits": 0,
        "tail_energy": 0.0, "bulk_energy": 100.0, "bulk_dimension": N,
        "free_lloyd_sse": 0.0, "fp16_sse": 0.0,
        "centroids": [], "fp16_centroids": [],
    }
    panel = {
        "label": "synthetic", "total_energy": 1200.0,
        "matrices": [[dict(blank) for _ in PROFILES] for _ in range(MATRICES)],
        "uniform_components": {
            "coordinate:0.00000000": [
                Component(f"m{i}", i // ROLES, N, 100.0) for i in range(MATRICES)
            ]
        },
    }
    row = score(panel, [0] * MATRICES, 2.15, "finite_fp16", "raw")
    require(row["side_ledger"]["bit_closure"] == row["physical_bits"], "side-bit closure")
    require(row["read_ledger"]["below_2x"], "read-amplification closure")
    print("lossy_tail_oracle self-test passed")


def main() -> None:
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize != 0:
        raise RuntimeError("v7 production core requires python -B -I without optimization")
    context = _V7_CORE_CONTEXT
    if not _V7_PREIMPORT_PRODUCTION_AUTHENTICATED:
        raise RuntimeError("v7 production main was not authenticated before NumPy import")
    require_exact_keys(context, {
        "schema", "mode", "parent_pid", "child_pid", "capability_sha256",
        "preflight_cmdline_sha256", "launch_manifest_sha256",
        "authorization_file_sha256", "authorization_internal_sha256",
    }, "v7 production child context")
    if context["mode"] != "production_child":
        raise RuntimeError("v7 production main requires a production-child context")
    if context["child_pid"] != os.getpid() or context["parent_pid"] != os.getppid():
        raise RuntimeError("v7 production child process identity mismatch")
    _v7_preimport_production_firewall(context)
    for label in (
        "capability_sha256", "preflight_cmdline_sha256", "launch_manifest_sha256",
        "authorization_file_sha256", "authorization_internal_sha256",
    ):
        value = context[label]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise RuntimeError(f"invalid v7 production child context hash: {label}")
    production_flags = [
        "--bindings", "--protocol", "--repair-lock", "--runtime-contract",
        "--authorization-contract", "--launch-manifest", "--launch-manifest-sha256",
        "--authorization", "--authorization-sha256", "--control-replicates",
        "--maximum-coordinate-passes",
    ]
    raw = sys.argv[1:]
    if len(raw) != 2 * len(production_flags) or raw[::2] != production_flags:
        raise RuntimeError("invalid frozen v7 production-core grammar")
    if raw[19] != "4" or raw[21] != "4":
        raise RuntimeError("frozen v7 production requires controls=4 and passes=4")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in production_flags[:-2]:
        parser.add_argument(name, required=True)
    parser.add_argument("--control-replicates", type=int, required=True)
    parser.add_argument("--maximum-coordinate-passes", type=int, required=True)
    args = parser.parse_args(raw)
    if args.control_replicates != 4 or args.maximum_coordinate_passes != 4:
        raise RuntimeError("frozen v7 production requires controls=4 and passes=4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("production requires explicit CUDA_VISIBLE_DEVICES=0")

    bootstrap_path = require_canonical_original_path(sys.argv[0], "authenticated child bootstrap", allow_missing_tail=False)
    if bootstrap_path.name != "lossy_tail_oracle.py":
        raise RuntimeError("production core argv0 must remain the authenticated child bootstrap")
    core_path = require_canonical_original_path(os.fspath(Path(__file__)), "scientific core", allow_missing_tail=False)
    if core_path.name != "lossy_tail_core.py" or core_path.parent != bootstrap_path.parent:
        raise RuntimeError("production core/bootstrap stage mismatch")
    stage = require_canonical_original_path(core_path.parent, "stage", allow_missing_tail=False)
    lock_names = (
        (args.bindings, "source_bindings.json"),
        (args.protocol, "protocol_lock.json"),
        (args.repair_lock, "repair_lock.json"),
        (args.runtime_contract, "runtime_contract.json"),
        (args.authorization_contract, "authorization_contract.json"),
        (args.launch_manifest, "launch_manifest.json"),
    )
    for supplied, basename in lock_names:
        checked = require_canonical_original_path(supplied, basename, allow_missing_tail=False)
        if checked != stage / basename:
            raise RuntimeError(f"{basename} must use exact canonical immediate-stage spelling")
    validate_stage_manifest(stage, stage / "launch_manifest.json", args.launch_manifest_sha256.lower())
    protocol_bytes = read_regular_descriptor(stage / "protocol_lock.json", "scientific protocol")
    bindings_bytes = read_regular_descriptor(stage / "source_bindings.json", "source bindings")
    repair_bytes = read_regular_descriptor(stage / "repair_lock.json", "repair lock")
    runtime_contract_bytes = read_regular_descriptor(stage / "runtime_contract.json", "runtime contract")
    authorization_contract_bytes = read_regular_descriptor(stage / "authorization_contract.json", "authorization contract")
    protocol = strict_json_bytes(protocol_bytes, "scientific protocol")
    bindings = strict_json_bytes(bindings_bytes, "source bindings")
    repair = strict_json_bytes(repair_bytes, "repair lock")
    runtime_contract = strict_json_bytes(runtime_contract_bytes, "runtime contract")
    authorization_contract = strict_json_bytes(authorization_contract_bytes, "authorization contract")
    if protocol.get("status") != "FROZEN_V7_BEFORE_ANY_RUNTIME_CALIBRATION_PAYLOAD_OR_GPU_EXECUTION":
        raise RuntimeError("v7 protocol is not frozen")
    if runtime_contract.get("status") != "FROZEN_SOURCE_FREE_BEFORE_RUNTIME_CALIBRATION":
        raise RuntimeError("runtime contract is not frozen")
    if authorization_contract.get("status") != "FROZEN_TEMPLATE_ONLY_NO_AUTHORIZATION_EXISTS":
        raise RuntimeError("authorization contract status mismatch")
    if repair.get("schema") != "lossy-tail-release-repair-lock-v7" or repair.get("status") != "FROZEN_V7_SOURCE_PACKAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        raise RuntimeError("v7 repair lock is not frozen")
    verify_internal_seal(repair, "repair_lock_sha256", repair.get("repair_lock_sha256"), "repair lock")
    identities = repair["authenticated_identities"]
    live_identities = {
        "scientific_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_bindings_sha256": hashlib.sha256(bindings_bytes).hexdigest(),
        "runtime_contract_sha256": hashlib.sha256(runtime_contract_bytes).hexdigest(),
        "authorization_contract_sha256": hashlib.sha256(authorization_contract_bytes).hexdigest(),
        "oracle_bootstrap_sha256": sha256_file(bootstrap_path),
        "scientific_core_sha256": sha256_file(core_path),
        "preflight_sha256": sha256_file(stage / "preflight_launch.py"),
        "audit_entrypoint_sha256": sha256_file(stage / "audit_lock_entrypoint.py"),
        "runtime_calibrate_sha256": sha256_file(stage / "runtime_calibrate.py"),
    }
    if identities != live_identities:
        raise RuntimeError("repair-lock authenticated identities mismatch")
    if [x["expert"] for x in bindings["files"][::2]] != [24, 56, 80, 88, 96, 120]:
        raise RuntimeError("binding cohort drift")
    binding_source = bindings.get("source_directory_at_execution")
    runtime_contract_sha256 = live_identities["runtime_contract_sha256"]
    authorization_state = validate_production_authorization(
        args.authorization,
        args.authorization_sha256.lower(),
        stage=stage,
        launcher=stage / "preflight_launch.py",
        manifest_sha256=args.launch_manifest_sha256.lower(),
        runtime_contract_sha256=runtime_contract_sha256,
        bindings_sha256=live_identities["source_bindings_sha256"],
        repair_internal_sha256=repair["repair_lock_sha256"],
    )
    if context["launch_manifest_sha256"] != args.launch_manifest_sha256.lower():
        raise RuntimeError("child capability launch-manifest identity mismatch")
    if context["authorization_file_sha256"] != args.authorization_sha256.lower():
        raise RuntimeError("child capability authorization-file identity mismatch")
    if context["authorization_internal_sha256"] != authorization_state["authorization"]["authorization_sha256"]:
        raise RuntimeError("child capability authorization-internal identity mismatch")
    source_dir = authorization_state["source"]
    output = authorization_state["result"]
    if os.fspath(source_dir) != binding_source:
        raise RuntimeError("authorized source path differs from frozen source binding")

    import cupy as cp  # type: ignore
    expected_runtime = runtime_contract["expected_runtime_tuple"]
    live_runtime = exact_runtime_tuple(cp)
    for key, expected in expected_runtime.items():
        if live_runtime.get(key) != expected:
            raise RuntimeError(f"live runtime tuple mismatch for {key}")
    print("[0/7] exact source-free runtime replay", flush=True)
    live_probe = runtime_probe(cp)
    sealed_probe = authorization_state["runtime_receipt"].get("runtime_probe")
    if live_probe != sealed_probe:
        raise RuntimeError("live runtime probe differs from independently audited receipt")

    start_wall = time.time(); start_mono = time.monotonic()
    runtime = {
        "argv": list(sys.argv), "cwd": os.getcwd(), "pid": os.getpid(),
        "python": sys.version, "platform": platform.platform(),
        "runtime_tuple": live_runtime,
        "runtime_probe_aggregate_sha256": live_probe["probe_aggregate_sha256"],
        "started_utc_epoch": start_wall,
    }
    print("[1/7] authenticated auxiliary Qwen panel", flush=True)
    qwen_panel = build_panel("qwen_aux_l15_up_down", cp, bindings=bindings, source_dir=source_dir)
    source_moments = [{k: float(m[k]) for k in ("energy", "mean", "variance")} for m in qwen_panel["moments"]]
    print("[2/7] Qwen joint searches", flush=True)
    qwen_search = search_panel(qwen_panel, args.maximum_coordinate_passes)
    control_searches, control_meta, mismatch_rows = [], [], []
    for replica in range(args.control_replicates):
        print(f"[{3 + replica}/7] matched Gaussian replica {replica}", flush=True)
        panel = build_panel(f"gaussian_control_{replica}", cp, source_moments=source_moments, replica=replica)
        for ordinal, moment in enumerate(panel["moments"]):
            affine = moment["control_affine"]
            mismatch_rows.append({
                "replica": replica, "ordinal": ordinal,
                **{key: affine[key] for key in (
                    "mean_absolute_error", "mean_absolute_tolerance", "mean_normalized_mismatch",
                    "variance_absolute_error", "variance_absolute_tolerance", "variance_normalized_mismatch",
                )},
            })
        control_searches.append(search_panel(panel, args.maximum_coordinate_passes))
        control_meta.append({
            "replica": replica, "moments": panel["moments"],
            "total_energy": panel["total_energy"],
            "candidate_ledger_sha256": panel["candidate_ledger_sha256"],
            "candidate_count": panel["candidate_count"],
        })
        del panel
        cp.get_default_memory_pool().free_all_blocks()
    calibrated = calibrate(qwen_search, control_searches)
    decision = decision_from_calibrated(calibrated)
    decision["claim_boundary"] = protocol["claim_boundary"]
    runtime["ended_utc_epoch"] = time.time()
    runtime["elapsed_seconds"] = time.monotonic() - start_mono
    control_mismatch_summary = {
        "cell_count": len(mismatch_rows),
        "cells": mismatch_rows,
        "maximum_mean_normalized_mismatch": max(row["mean_normalized_mismatch"] for row in mismatch_rows),
        "maximum_variance_normalized_mismatch": max(row["variance_normalized_mismatch"] for row in mismatch_rows),
        "all_cells_within_tolerance": all(
            row["mean_normalized_mismatch"] <= 1.0 and row["variance_normalized_mismatch"] <= 1.0
            for row in mismatch_rows
        ),
    }
    if not control_mismatch_summary["all_cells_within_tolerance"]:
        raise RuntimeError("control mismatch summary contradiction")
    report = {
        "schema": "qwen_lossy_tail_peeling_oracle_result_v7",
        "authorization": {
            "path": os.fspath(authorization_state["authorization_path"]),
            "file_sha256": args.authorization_sha256.lower(),
            "internal_sha256": authorization_state["authorization"]["authorization_sha256"],
            "nonce": authorization_state["authorization"]["authorization_nonce"],
            "action": authorization_state["authorization"]["action"],
        },
        "launch_manifest": {"path": os.fspath(stage / "launch_manifest.json"), "sha256": args.launch_manifest_sha256.lower()},
        "protocol": {"path": os.fspath(stage / "protocol_lock.json"), "sha256": live_identities["scientific_protocol_sha256"]},
        "repair_lock": {"path": os.fspath(stage / "repair_lock.json"), "sha256": sha256_file(stage / "repair_lock.json")},
        "runtime_contract": {"path": os.fspath(stage / "runtime_contract.json"), "sha256": runtime_contract_sha256},
        "runtime_receipt": {
            "path": os.fspath(authorization_state["runtime_receipt_path"]),
            "file_sha256": authorization_state["authorization"]["runtime_receipt"]["file_sha256"],
            "internal_sha256": authorization_state["authorization"]["runtime_receipt"]["internal_sha256"],
        },
        "bindings": {"path": os.fspath(stage / "source_bindings.json"), "sha256": live_identities["source_bindings_sha256"]},
        "oracle_bootstrap_sha256": live_identities["oracle_bootstrap_sha256"],
        "scientific_core_sha256": live_identities["scientific_core_sha256"],
        "child_capability": {
            "sha256": context["capability_sha256"],
            "parent_pid": context["parent_pid"],
            "child_pid": context["child_pid"],
            "preflight_cmdline_sha256": context["preflight_cmdline_sha256"],
        },
        "mountinfo_sha256": authorization_state["mountinfo_sha256"],
        "runtime": runtime,
        "panel": {
            "experts": EXPERTS, "roles": ROLES, "matrices": MATRICES,
            "values_per_matrix": N, "panel_values": PANEL_N,
            "total_energy": qwen_panel["total_energy"],
            "moments": qwen_panel["moments"],
            "source_receipts": qwen_panel["source_receipts"],
            "candidate_count": qwen_panel["candidate_count"],
            "candidate_ledger_sha256": qwen_panel["candidate_ledger_sha256"],
        },
        "grid": {"profiles": PROFILES, "profile_count": len(PROFILES), "rates": RATES, "modes": MODES},
        "qwen_search": qwen_search,
        "matched_gaussian_controls": {
            "replicates": args.control_replicates, "panels": control_meta,
            "searches": control_searches, "post_fp32_moment_mismatch": control_mismatch_summary,
        },
        "calibrated_rows": calibrated,
        "decision": decision,
    }
    write_sealed_json(output, report)
    print(json.dumps(report["decision"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {output}; elapsed={runtime['elapsed_seconds']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
