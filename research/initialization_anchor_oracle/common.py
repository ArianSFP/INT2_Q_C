"""Frozen protocol helpers for the Qwen3 initialization-anchor gate.

This module is deliberately CUDA-free.  Importing it must not import torch or
CuPy, which lets the protocol, firewall, sampler, ledgers, and result algebra
be checked before any CUDA context exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
CANDIDATE_LOCK_PATH = PACKAGE_DIR / "candidate_lock.json"
CANDIDATE_LOCK_FILE_SHA256 = "8f5ad9cb3bff21893e9fc2daf287942a43610f0af288e5070f173c93f05fb6ca"
CANDIDATE_LOCK_INTERNAL_SHA256 = "b7f100835e366c7ca68c189206dd953fbba7b737f00a19cacf74af277b131dbd"
LOCK_PLACEHOLDER = "TO_BE_FILLED_AFTER_CANONICAL_FREEZE"
EXCLUSION_INTERSECTION_PATH = PACKAGE_DIR / "exclusion_intersection_lock.json"
EXPECTED_EXCLUSION_INTERSECTION_SHA256 = "6521f4613183147a1602044c7f427fa38f27b82757e6b25543bb3d5194df88c7"

SCHEMA = "qwen3_initialization_anchor_result_v1"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
EXPERTS = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120)
TRAIN_EXPERTS = (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)
VALIDATION_EXPERTS = (24, 56, 88, 120)
ROLES = ("up", "down")
ROWS = 768
COLUMNS = 2048
WEIGHTS_PER_MATRIX = ROWS * COLUMNS
BYTES_PER_TENSOR = WEIGHTS_PER_MATRIX * 2
TOTAL_COORDINATES = 65_536
FIT_COORDINATES = 32_768
SCORE_COORDINATES = 32_768

EXPECTED_SOURCE_MANIFEST_SHA256 = "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782"
EXPECTED_SOURCE_FREEZE_SHA256 = "37743eaf6cb70c2bc68704dcf4d60e013552b76c11daf1ab4855f64ad4417193"
EXPECTED_EXCLUSION_MANIFEST_SHA256 = "3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55"
EXPECTED_EXCLUDED_TENSORS = ("model.layers.15.mlp.experts.0.up_proj.weight",)

TARGET_F = 0.8
CURRENT_F = 0.9888693569009007
COMPOSITE_F = 0.936397621
CURRENT_WORST_READ_AMP = 1.169444
METADATA_BYTES_PER_MATRIX = 23
TARGET_MATRIX_COUNT = 18
TARGET_WEIGHTS = TARGET_MATRIX_COUNT * WEIGHTS_PER_MATRIX
WEIGHTS_PER_EXPERT = 3 * WEIGHTS_PER_MATRIX


class ProtocolError(RuntimeError):
    """Raised on any fail-closed protocol or firewall violation."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_lock_internal_seal(raw: bytes, lock: Mapping[str, Any]) -> None:
    seal = str(lock.get("lock_sha256", ""))
    if seal != CANDIDATE_LOCK_INTERNAL_SHA256:
        raise ProtocolError("candidate lock internal seal literal mismatch")
    pattern = rb'("lock_sha256"\s*:\s*")([0-9a-f]{64})(")'
    matches = list(re.finditer(pattern, raw))
    if len(matches) != 1:
        raise ProtocolError("candidate lock must contain exactly one internal seal")
    match = matches[0]
    normalized = raw[: match.start(2)] + LOCK_PLACEHOLDER.encode("ascii") + raw[match.end(2) :]
    if sha256_bytes(normalized) != CANDIDATE_LOCK_INTERNAL_SHA256:
        raise ProtocolError("candidate lock internal placeholder-normalized seal mismatch")


def load_candidate_lock(path: Path = CANDIDATE_LOCK_PATH) -> dict[str, Any]:
    path = path.resolve()
    raw = path.read_bytes()
    if sha256_bytes(raw) != CANDIDATE_LOCK_FILE_SHA256:
        raise ProtocolError("candidate_lock.json file SHA-256 mismatch")
    lock = json.loads(raw.decode("utf-8"))
    _validate_lock_internal_seal(raw, lock)
    if lock.get("schema") != "qwen3_initialization_anchor_candidate_lock_v1":
        raise ProtocolError("candidate lock schema mismatch")
    if lock.get("status") != "FROZEN_BEFORE_AUXILIARY_PAYLOAD_ACCESS":
        raise ProtocolError("candidate lock is not frozen")
    if lock["provenance"]["revision"] != REVISION:
        raise ProtocolError("candidate lock revision mismatch")
    return lock


def load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"{label} must be a regular non-symlink file")
    if sha256_file(path) != expected_sha256:
        raise ProtocolError(f"{label} SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must contain one JSON object")
    return value


def default_workspace_root() -> Path:
    # .../INT2_Q_C/research/initialization_anchor_oracle -> workspace root
    return PACKAGE_DIR.parents[2]


def tensor_name(expert: int, role: str) -> str:
    projection = "up_proj" if role == "up" else "down_proj"
    return f"model.layers.15.mlp.experts.{expert}.{projection}.weight"


def tensor_basename(expert: int, role: str) -> str:
    return f"l15e{expert}_{role}.bf16.bin"


@dataclass(frozen=True)
class SourceRow:
    ordinal: int
    expert: int
    role: str
    tensor_name: str
    basename: str
    sha256: str
    bytes: int
    raw_shape: tuple[int, int]
    canonical_shape: tuple[int, int]
    excluded: bool
    split: str


def build_source_rows(
    source_manifest: Mapping[str, Any], exclusion_manifest: Mapping[str, Any]
) -> tuple[SourceRow, ...]:
    if source_manifest.get("revision") != REVISION:
        raise ProtocolError("source manifest revision mismatch")
    tensors = source_manifest.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != 32:
        raise ProtocolError("source manifest must bind exactly 32 tensors")
    blocks = exclusion_manifest.get("blocks")
    if exclusion_manifest.get("revision") != REVISION or not isinstance(blocks, list):
        raise ProtocolError("exclusion manifest revision/schema mismatch")
    excluded_names = {str(row.get("tensor")) for row in blocks}

    expected_order = [(expert, role) for expert in EXPERTS for role in ROLES]
    rows: list[SourceRow] = []
    seen: set[tuple[int, str]] = set()
    for ordinal, item in enumerate(tensors):
        expert = int(item["expert"])
        role = str(item["role"])
        key = (expert, role)
        if key in seen:
            raise ProtocolError(f"duplicate source tensor {key}")
        seen.add(key)
        if ordinal >= len(expected_order) or key != expected_order[ordinal]:
            raise ProtocolError("source-manifest tensor order differs from frozen expert/role order")
        expected_name = tensor_name(expert, role)
        expected_basename = tensor_basename(expert, role)
        if str(item["tensor_name"]) != expected_name:
            raise ProtocolError(f"unexpected tensor identity for {key}")
        if Path(str(item["local_path"])).name != expected_basename:
            raise ProtocolError(f"unexpected source basename for {key}")
        if int(item["bytes"]) != BYTES_PER_TENSOR:
            raise ProtocolError(f"wrong frozen byte count for {key}")
        raw_shape = tuple(int(x) for x in item["raw_shape"])
        expected_raw = (ROWS, COLUMNS) if role == "up" else (COLUMNS, ROWS)
        if raw_shape != expected_raw:
            raise ProtocolError(f"wrong native shape for {key}")
        canonical_shape = tuple(int(x) for x in item["canonical_shape"])
        if canonical_shape != (ROWS, COLUMNS):
            raise ProtocolError(f"wrong canonical shape for {key}")
        is_excluded = expected_name in excluded_names
        split = "validation" if expert in VALIDATION_EXPERTS else "candidate_selection"
        rows.append(
            SourceRow(
                ordinal=ordinal,
                expert=expert,
                role=role,
                tensor_name=expected_name,
                basename=expected_basename,
                sha256=str(item["sha256"]),
                bytes=BYTES_PER_TENSOR,
                raw_shape=expected_raw,
                canonical_shape=(ROWS, COLUMNS),
                excluded=is_excluded,
                split=split,
            )
        )
    if tuple(sorted(name for name in excluded_names if name in {r.tensor_name for r in rows})) != tuple(
        sorted(EXPECTED_EXCLUDED_TENSORS)
    ):
        raise ProtocolError("held-out intersection differs from the one frozen exclusion")
    eligible = [row for row in rows if not row.excluded]
    if len(eligible) != 31:
        raise ProtocolError("eligible source count must be exactly 31")
    if sum(row.split == "candidate_selection" for row in eligible) != 23:
        raise ProtocolError("candidate-selection source count must be exactly 23")
    if sum(row.split == "validation" for row in eligible) != 8:
        raise ProtocolError("validation source count must be exactly 8")
    return tuple(rows)


def load_exclusion_binding(workspace_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = (workspace_root or default_workspace_root()).resolve()
    intersection = load_bound_json(
        EXCLUSION_INTERSECTION_PATH,
        EXPECTED_EXCLUSION_INTERSECTION_SHA256,
        "packaged exclusion intersection",
    )
    strict_keys(
        intersection,
        (
            "schema", "status", "source_exclusion_manifest_sha256",
            "source_exclusion_manifest_revision", "source_panel_tensor_count",
            "intersection_count", "excluded_tensor_identities", "derivation",
        ),
        "packaged exclusion intersection",
    )
    if intersection["schema"] != "qwen3_initialization_anchor_exclusion_intersection_v1":
        raise ProtocolError("packaged exclusion-intersection schema mismatch")
    if intersection["status"] != "DERIVED_AND_FROZEN_BEFORE_AUXILIARY_PAYLOAD_ACCESS":
        raise ProtocolError("packaged exclusion intersection is not frozen")
    if intersection["source_exclusion_manifest_sha256"] != EXPECTED_EXCLUSION_MANIFEST_SHA256:
        raise ProtocolError("packaged intersection binds the wrong exclusion manifest")
    if intersection["source_exclusion_manifest_revision"] != REVISION:
        raise ProtocolError("packaged exclusion-intersection revision mismatch")
    identities = tuple(str(value) for value in intersection["excluded_tensor_identities"])
    if int(intersection["source_panel_tensor_count"]) != 32 or int(intersection["intersection_count"]) != 1:
        raise ProtocolError("packaged exclusion-intersection count mismatch")
    if identities != EXPECTED_EXCLUDED_TENSORS:
        raise ProtocolError("packaged exclusion tensor identities differ from frozen expectation")

    full_path = root / "qwen_polaris_heldout32_manifest.json"
    revalidated = False
    if full_path.exists() or full_path.is_symlink():
        full = load_bound_json(full_path, EXPECTED_EXCLUSION_MANIFEST_SHA256, "exclusion manifest")
        blocks = full.get("blocks")
        if full.get("revision") != REVISION or not isinstance(blocks, list):
            raise ProtocolError("full exclusion manifest revision/schema mismatch")
        source_names = {tensor_name(expert, role) for expert in EXPERTS for role in ROLES}
        observed = tuple(sorted(str(row.get("tensor")) for row in blocks if str(row.get("tensor")) in source_names))
        if observed != tuple(sorted(identities)):
            raise ProtocolError("full exclusion-manifest intersection differs from packaged lock")
        revalidated = True

    synthetic = {"revision": REVISION, "blocks": [{"tensor": value} for value in identities]}
    status = {
        "packaged_intersection_lock_sha256": EXPECTED_EXCLUSION_INTERSECTION_SHA256,
        "source_exclusion_manifest_sha256": EXPECTED_EXCLUSION_MANIFEST_SHA256,
        "full_external_manifest_revalidated_at_runtime": revalidated,
        "excluded_tensor_identities": list(identities),
    }
    return synthetic, status


def load_frozen_source_rows(workspace_root: Path | None = None) -> tuple[SourceRow, ...]:
    root = (workspace_root or default_workspace_root()).resolve()
    source = load_bound_json(
        root / "agent_rd_structure_diag_cross_expert_sources.json",
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "source manifest",
    )
    load_bound_json(
        root / "agent_rd_structure_diag_cross_expert_freeze.json",
        EXPECTED_SOURCE_FREEZE_SHA256,
        "source freeze",
    )
    exclusion, _ = load_exclusion_binding(root)
    return build_source_rows(source, exclusion)


def validate_aux_directory(aux_dir: Path, rows: Sequence[SourceRow]) -> dict[str, Path]:
    aux_dir = aux_dir.resolve()
    if not aux_dir.is_dir() or aux_dir.is_symlink():
        raise ProtocolError("auxiliary path must be a regular directory, not a symlink")
    forbidden = {"blind_protocol", "blind_protocol_v2", "qwen_polaris_heldout32", "heldout", "quarantine"}
    if any(part.lower() in forbidden for part in aux_dir.parts):
        raise ProtocolError("auxiliary path enters a forbidden held-out/quarantine component")
    expected = {row.basename for row in rows}
    observed = {path.name for path in aux_dir.glob("*.bf16.bin")}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProtocolError(f"auxiliary BF16 file set mismatch; missing={missing}, extra={extra}")
    result: dict[str, Path] = {}
    for row in rows:
        path = aux_dir / row.basename
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(f"source must be a regular non-symlink file: {row.basename}")
        if path.resolve().parent != aux_dir:
            raise ProtocolError(f"source escapes the resolved auxiliary directory: {row.basename}")
        if path.stat().st_size != row.bytes:
            raise ProtocolError(f"source byte count mismatch: {row.basename}")
        result[row.tensor_name] = path
    return result


@dataclass(frozen=True)
class CoordinateRow:
    source: SourceRow
    fit: tuple[int, ...]
    score: tuple[int, ...]
    fit_counter_stop: int
    score_counter_stop: int
    fit_modulo_rejections: int
    score_modulo_rejections: int
    fit_duplicate_rejections: int
    score_duplicate_rejections: int
    fit_sha256_le_u32: str
    score_sha256_le_u32: str


def _sample_unique_coordinates(
    tensor_identity: str,
    split: str,
    count: int,
    used: set[int],
    population: int = WEIGHTS_PER_MATRIX,
) -> tuple[tuple[int, ...], dict[str, int]]:
    domain = b"QWEN3-INIT-ANCHOR-COORD-v1\0"
    name = tensor_identity.encode("utf-8")
    split_bytes = split.encode("ascii")
    prefix = (
        domain
        + bytes.fromhex(REVISION)
        + len(name).to_bytes(2, "little")
        + name
        + len(split_bytes).to_bytes(1, "little")
        + split_bytes
    )
    limit = (1 << 64) - ((1 << 64) % population)
    accepted: list[int] = []
    modulo_rejections = 0
    duplicate_rejections = 0
    counter = 0
    while len(accepted) < count:
        digest = hashlib.sha256(prefix + counter.to_bytes(8, "little")).digest()
        counter += 1
        for lane in range(4):
            value = int.from_bytes(digest[8 * lane : 8 * (lane + 1)], "little")
            if value >= limit:
                modulo_rejections += 1
                continue
            coordinate = value % population
            if coordinate in used:
                duplicate_rejections += 1
                continue
            used.add(coordinate)
            accepted.append(coordinate)
            if len(accepted) == count:
                break
    accepted.sort()
    return tuple(accepted), {
        "counter_stop": counter,
        "modulo_rejections": modulo_rejections,
        "duplicate_rejections": duplicate_rejections,
    }


def _quotient_remainder_counts(total: int, items: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, items)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(items))


def make_coordinate_plan(rows: Sequence[SourceRow]) -> tuple[CoordinateRow, ...]:
    eligible = [row for row in rows if not row.excluded]
    if len(eligible) != 31:
        raise ProtocolError("coordinate plan requires the exact 31 eligible tensors")
    fit_counts = _quotient_remainder_counts(FIT_COORDINATES, len(eligible))
    score_counts = _quotient_remainder_counts(SCORE_COORDINATES, len(eligible))
    plan: list[CoordinateRow] = []
    for row, fit_count, score_count in zip(eligible, fit_counts, score_counts):
        used: set[int] = set()
        fit, fit_meta = _sample_unique_coordinates(row.tensor_name, "fit", fit_count, used)
        score, score_meta = _sample_unique_coordinates(row.tensor_name, "score", score_count, used)
        fit_bytes = b"".join(struct.pack("<I", value) for value in fit)
        score_bytes = b"".join(struct.pack("<I", value) for value in score)
        plan.append(
            CoordinateRow(
                source=row,
                fit=fit,
                score=score,
                fit_counter_stop=fit_meta["counter_stop"],
                score_counter_stop=score_meta["counter_stop"],
                fit_modulo_rejections=fit_meta["modulo_rejections"],
                score_modulo_rejections=score_meta["modulo_rejections"],
                fit_duplicate_rejections=fit_meta["duplicate_rejections"],
                score_duplicate_rejections=score_meta["duplicate_rejections"],
                fit_sha256_le_u32=sha256_bytes(fit_bytes),
                score_sha256_le_u32=sha256_bytes(score_bytes),
            )
        )
    if sum(len(row.fit) for row in plan) != FIT_COORDINATES:
        raise ProtocolError("fit-coordinate accounting mismatch")
    if sum(len(row.score) for row in plan) != SCORE_COORDINATES:
        raise ProtocolError("score-coordinate accounting mismatch")
    return tuple(plan)


def coordinate_plan_sha256(plan: Sequence[CoordinateRow]) -> str:
    digest = hashlib.sha256()
    for row in plan:
        name = row.source.tensor_name.encode("utf-8")
        digest.update(len(name).to_bytes(2, "little"))
        digest.update(name)
        for label, values in ((b"fit", row.fit), (b"score", row.score)):
            digest.update(label)
            digest.update(len(values).to_bytes(4, "little"))
            for value in values:
                digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def coordinate_plan_json(plan: Sequence[CoordinateRow]) -> list[dict[str, Any]]:
    result = []
    for row in plan:
        result.append(
            {
                "tensor_name": row.source.tensor_name,
                "expert": row.source.expert,
                "role": row.source.role,
                "split": row.source.split,
                "fit_count": len(row.fit),
                "score_count": len(row.score),
                "fit_counter_stop": row.fit_counter_stop,
                "score_counter_stop": row.score_counter_stop,
                "fit_modulo_rejections": row.fit_modulo_rejections,
                "score_modulo_rejections": row.score_modulo_rejections,
                "fit_duplicate_rejections": row.fit_duplicate_rejections,
                "score_duplicate_rejections": row.score_duplicate_rejections,
                "fit_sha256_le_u32": row.fit_sha256_le_u32,
                "score_sha256_le_u32": row.score_sha256_le_u32,
            }
        )
    return result


def canonical_to_native_flat(role: str, canonical: np.ndarray) -> np.ndarray:
    canonical = np.asarray(canonical, dtype=np.int64)
    rows = canonical // COLUMNS
    columns = canonical % COLUMNS
    if role == "up":
        native = rows * COLUMNS + columns
    elif role == "down":
        native = columns * ROWS + rows
    else:
        raise ProtocolError(f"unsupported auxiliary role: {role}")
    return native.astype(np.int64, copy=False)


def decode_bfloat16_words(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words, dtype="<u2")
    widened = words.astype("<u4", copy=False) << np.uint32(16)
    return widened.view("<f4")


def read_source_coordinates(
    path: Path,
    row: SourceRow,
    coordinates: Sequence[int],
    *,
    verify_hash: bool = True,
) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) != row.bytes:
        raise ProtocolError(f"source changed size during read: {row.basename}")
    if verify_hash and sha256_bytes(payload) != row.sha256:
        raise ProtocolError(f"source SHA-256 mismatch: {row.basename}")
    words = np.frombuffer(payload, dtype="<u2")
    native = canonical_to_native_flat(row.role, np.asarray(coordinates, dtype=np.int64))
    result = decode_bfloat16_words(words[native]).astype(np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ProtocolError(f"non-finite source value: {row.basename}")
    return result


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    family: str
    seed: int
    dtype_path: str
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    expert_assignment: str = "none"
    expert_projection_packing: str = "none"

    @property
    def id(self) -> str:
        if self.family.startswith("hf451_"):
            suffix = f"{self.family}|seed={self.seed}|dtype={self.dtype_path}"
        else:
            suffix = (
                f"{self.family}|seed={self.seed}|dtype={self.dtype_path}"
                f"|pp={self.pipeline_parallel_size}|ep={self.expert_parallel_size}"
                f"|etp={self.expert_tensor_parallel_size}|assign={self.expert_assignment}"
                f"|pack={self.expert_projection_packing}"
            )
        return f"{self.ordinal:04d}|{suffix}"

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["id"] = self.id
        return value


def enumerate_candidates(lock: Mapping[str, Any]) -> tuple[Candidate, ...]:
    seeds = tuple(int(seed) for seed in lock["seeds_in_order"])
    dtypes = tuple(str(dtype) for dtype in lock["dtype_paths_in_order"])
    families = lock["families_in_order"]
    result: list[Candidate] = []
    for family in families:
        family_id = str(family["id"])
        if family_id != "mcore_expert_parallel_stream":
            for seed in seeds:
                for dtype in dtypes:
                    result.append(Candidate(len(result), family_id, seed, dtype))
            continue
        for seed in seeds:
            if seed <= 0:
                continue
            for dtype in dtypes:
                for pp in family["pipeline_parallel_sizes_in_order"]:
                    for ep in family["expert_parallel_sizes_in_order"]:
                        for etp in family["expert_tensor_parallel_sizes_in_order"]:
                            for assignment in family["expert_assignment_in_order"]:
                                for packing in family["expert_projection_packing_in_order"]:
                                    result.append(
                                        Candidate(
                                            ordinal=len(result),
                                            family=family_id,
                                            seed=seed,
                                            dtype_path=dtype,
                                            pipeline_parallel_size=int(pp),
                                            expert_parallel_size=int(ep),
                                            expert_tensor_parallel_size=int(etp),
                                            expert_assignment=str(assignment),
                                            expert_projection_packing=str(packing),
                                        )
                                    )
    expected = int(lock["candidate_enumeration"]["expected_total_candidates"])
    if len(result) != expected:
        raise ProtocolError(f"candidate enumeration count {len(result)} != frozen {expected}")
    ids = [candidate.id for candidate in result]
    if len(set(ids)) != len(ids):
        raise ProtocolError("candidate IDs are not unique")
    return tuple(result)


def stateless_standard_normals(tensor_identity: str, split: str, coordinates: Sequence[int]) -> np.ndarray:
    """Platform-independent SHA256/Box-Muller Gaussian control."""
    prefix = b"QWEN3-INIT-ANCHOR-GAUSSIAN-v1\0" + bytes.fromhex(REVISION)
    name = tensor_identity.encode("utf-8")
    split_bytes = split.encode("ascii")
    prefix += len(name).to_bytes(2, "little") + name + len(split_bytes).to_bytes(1, "little") + split_bytes
    result = np.empty(len(coordinates), dtype=np.float32)
    scale = 1.0 / float(1 << 64)
    for index, coordinate in enumerate(coordinates):
        digest = hashlib.sha256(prefix + int(coordinate).to_bytes(8, "little")).digest()
        u1 = (int.from_bytes(digest[0:8], "little") + 0.5) * scale
        u2 = (int.from_bytes(digest[8:16], "little") + 0.5) * scale
        value = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        result[index] = np.float32(value)
    return result


def deterministic_permutation(tensor_identity: str, split: str, count: int) -> np.ndarray:
    prefix = b"QWEN3-INIT-ANCHOR-PERMUTE-v1\0" + bytes.fromhex(REVISION)
    name = tensor_identity.encode("utf-8")
    split_bytes = split.encode("ascii")
    prefix += len(name).to_bytes(2, "little") + name + len(split_bytes).to_bytes(1, "little") + split_bytes
    keyed = []
    for index in range(count):
        key = hashlib.sha256(prefix + index.to_bytes(4, "little")).digest()
        keyed.append((key, index))
    keyed.sort()
    return np.asarray([index for _, index in keyed], dtype=np.int64)


def sample_mean_rms(values: Sequence[float]) -> tuple[float, float]:
    values64 = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values64))
    centered = values64 - mean
    rms = math.sqrt(float(np.mean(centered * centered)))
    return mean, rms


def matched_gaussian_values(
    tensor_identity: str,
    fit_coordinates: Sequence[int],
    score_coordinates: Sequence[int],
    source_fit: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean, rms = sample_mean_rms(source_fit)
    fit = mean + rms * stateless_standard_normals(tensor_identity, "fit", fit_coordinates)
    score = mean + rms * stateless_standard_normals(tensor_identity, "score", score_coordinates)
    return fit.astype(np.float32), score.astype(np.float32), {"fit_only_mean": mean, "fit_only_centered_rms": rms}


def fit_affine_moments(w: Sequence[float], g: Sequence[float]) -> dict[str, float | int]:
    w64 = np.asarray(w, dtype=np.float64)
    g64 = np.asarray(g, dtype=np.float64)
    if w64.shape != g64.shape or w64.ndim != 1 or w64.size == 0:
        raise ProtocolError("affine fit requires equal nonempty vectors")
    n = int(w64.size)
    sum_w = float(np.sum(w64, dtype=np.float64))
    sum_g = float(np.sum(g64, dtype=np.float64))
    sum_w2 = float(np.dot(w64, w64))
    sum_g2 = float(np.dot(g64, g64))
    sum_wg = float(np.dot(w64, g64))
    centered_gg = sum_g2 - sum_g * sum_g / n
    centered_wg = sum_wg - sum_w * sum_g / n
    alpha = centered_wg / centered_gg if centered_gg > 0.0 else 0.0
    mean_w = sum_w / n
    mean_g = sum_g / n
    mu = mean_w - alpha * mean_g
    return {
        "n": n,
        "sum_w": sum_w,
        "sum_g": sum_g,
        "sum_w2": sum_w2,
        "sum_g2": sum_g2,
        "sum_wg": sum_wg,
        "alpha": alpha,
        "mu": mu,
        "fit_mean_w": mean_w,
    }


def score_affine_moments(
    w: Sequence[float], g: Sequence[float], alpha: float, mu: float, fit_mean_w: float
) -> dict[str, float | int]:
    w64 = np.asarray(w, dtype=np.float64)
    g64 = np.asarray(g, dtype=np.float64)
    if w64.shape != g64.shape or w64.ndim != 1 or w64.size == 0:
        raise ProtocolError("affine score requires equal nonempty vectors")
    prediction = float(mu) + float(alpha) * g64
    residual = w64 - prediction
    baseline = w64 - float(fit_mean_w)
    n = int(w64.size)
    sum_w = float(np.sum(w64, dtype=np.float64))
    sum_g = float(np.sum(g64, dtype=np.float64))
    sum_w2 = float(np.dot(w64, w64))
    sum_g2 = float(np.dot(g64, g64))
    sum_wg = float(np.dot(w64, g64))
    centered_ww = sum_w2 - sum_w * sum_w / n
    centered_gg = sum_g2 - sum_g * sum_g / n
    centered_wg = sum_wg - sum_w * sum_g / n
    rho = centered_wg / math.sqrt(centered_ww * centered_gg) if centered_ww > 0 and centered_gg > 0 else 0.0
    return {
        "n": n,
        "sum_w": sum_w,
        "sum_g": sum_g,
        "sum_w2": sum_w2,
        "sum_g2": sum_g2,
        "sum_wg": sum_wg,
        "sse": float(np.dot(residual, residual)),
        "baseline_sse": float(np.dot(baseline, baseline)),
        "rho": rho,
    }


def metric_from_sse(sse: float, baseline_sse: float) -> dict[str, float]:
    if baseline_sse <= 0.0 or sse < 0.0:
        raise ProtocolError("invalid SSE pair")
    q = sse / baseline_sse
    return {"q": q, "capture": 1.0 - q}


def aggregate_matrix_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ProtocolError("cannot aggregate an empty statistic set")
    sse = math.fsum(float(row["score"]["sse"]) for row in rows)
    baseline = math.fsum(float(row["score"]["baseline_sse"]) for row in rows)
    metric = metric_from_sse(sse, baseline)
    return {
        "matrix_count": len(rows),
        "score_coordinates": sum(int(row["score"]["n"]) for row in rows),
        "sse": sse,
        "baseline_sse": baseline,
        **metric,
    }


def fold_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    experts = sorted({int(row["expert"]) for row in rows})
    roles = sorted({str(row["role"]) for row in rows})
    per_expert = []
    for expert in experts:
        subset = [row for row in rows if int(row["expert"]) == expert]
        metric = aggregate_matrix_statistics(subset)
        metric.update({"expert": expert, "roles": sorted(str(row["role"]) for row in subset)})
        per_expert.append(metric)
    per_role = []
    for role in roles:
        subset = [row for row in rows if str(row["role"]) == role]
        metric = aggregate_matrix_statistics(subset)
        metric.update({"role": role, "experts": sorted(int(row["expert"]) for row in subset)})
        per_role.append(metric)
    captures = [float(row["capture"]) for row in per_expert]
    if len(captures) > 1:
        mean = math.fsum(captures) / len(captures)
        variance = math.fsum((value - mean) ** 2 for value in captures) / (len(captures) - 1)
        standard_error = math.sqrt(variance / len(captures))
    else:
        standard_error = 0.0
    return {
        "pooled": aggregate_matrix_statistics(rows),
        "whole_experts": per_expert,
        "roles": per_role,
        "whole_expert_capture_standard_error": standard_error,
    }


def physical_ledger() -> dict[str, Any]:
    metadata_bytes = TARGET_MATRIX_COUNT * METADATA_BYTES_PER_MATRIX
    side_bpw = metadata_bytes * 8.0 / TARGET_WEIGHTS
    descriptor_bytes = 4096
    self_contained_side_bpw = (metadata_bytes + descriptor_bytes) * 8.0 / TARGET_WEIGHTS
    current_q_max = TARGET_F / CURRENT_F * 2.0 ** (-2.0 * side_bpw)
    composite_q_max = TARGET_F / COMPOSITE_F * 2.0 ** (-2.0 * side_bpw)
    payload_bytes_per_expert_at_cap = WEIGHTS_PER_EXPERT * 2.5 / 8.0
    conservative_appended_read_amp = CURRENT_WORST_READ_AMP + (
        3 * METADATA_BYTES_PER_MATRIX / payload_bytes_per_expert_at_cap
    )
    return {
        "target_matrix_count": TARGET_MATRIX_COUNT,
        "weights_per_matrix": WEIGHTS_PER_MATRIX,
        "target_weights": TARGET_WEIGHTS,
        "metadata_bytes_per_matrix": METADATA_BYTES_PER_MATRIX,
        "metadata_bytes_total": metadata_bytes,
        "model_specific_side_bpw": side_bpw,
        "self_contained_descriptor_sensitivity_bytes": descriptor_bytes,
        "self_contained_descriptor_sensitivity_bpw": self_contained_side_bpw,
        "metadata_adjusted_current_q_max": current_q_max,
        "metadata_adjusted_current_required_capture": 1.0 - current_q_max,
        "metadata_adjusted_current_required_abs_rho": math.sqrt(max(0.0, 1.0 - current_q_max)),
        "metadata_adjusted_composite_q_max": composite_q_max,
        "metadata_adjusted_composite_required_capture": 1.0 - composite_q_max,
        "metadata_adjusted_composite_required_abs_rho": math.sqrt(max(0.0, 1.0 - composite_q_max)),
        "generator_external_read_bytes": 0,
        "generator_learned_table_bytes": 0,
        "current_worst_cold_read_amplification": CURRENT_WORST_READ_AMP,
        "conservative_metadata_appended_worst_cold_read_amplification": conservative_appended_read_amp,
        "strict_cold_read_amplification_max_exclusive": 2.0,
        "passes_read_gate": conservative_appended_read_amp < 2.0,
    }


def make_decision(
    source_validation_folds: Mapping[str, Any],
    gaussian_validation_capture: float,
    permuted_validation_capture: float,
) -> dict[str, Any]:
    ledger = physical_ledger()
    raw_capture = float(source_validation_folds["pooled"]["capture"])
    null_capture = max(0.0, float(gaussian_validation_capture), float(permuted_validation_capture))
    corrected = raw_capture - null_capture
    se = float(source_validation_folds["whole_expert_capture_standard_error"])
    upper = corrected + 2.0 * se
    lower = corrected - 2.0 * se
    composite_required = float(ledger["metadata_adjusted_composite_required_capture"])
    current_required = float(ledger["metadata_adjusted_current_required_capture"])
    folds_positive = all(float(row["capture"]) > 0.0 for row in source_validation_folds["whole_experts"])
    if upper < composite_required:
        state = "HARD_KILL_BOUNDED_INITIALIZER_SET"
    elif lower >= current_required and folds_positive and ledger["passes_read_gate"]:
        state = "AUXILIARY_STANDALONE_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL"
    else:
        state = "COMPOSITE_ONLY_OR_INCONCLUSIVE"
    return {
        "state": state,
        "raw_validation_capture": raw_capture,
        "matched_null_capture": null_capture,
        "bias_corrected_validation_capture": corrected,
        "whole_expert_capture_standard_error": se,
        "bias_corrected_upper_2se": upper,
        "bias_corrected_lower_2se": lower,
        "metadata_adjusted_composite_required_capture": composite_required,
        "metadata_adjusted_current_required_capture": current_required,
        "all_whole_expert_folds_positive": folds_positive,
        "read_gate_passes": bool(ledger["passes_read_gate"]),
    }


def strict_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise ProtocolError(
            f"{label} keys mismatch; missing={sorted(expected_set-observed)}, extra={sorted(observed-expected_set)}"
        )


def source_rows_json(rows: Sequence[SourceRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def environment_has_cuda_imports() -> bool:
    import sys

    return "torch" in sys.modules or "cupy" in sys.modules
