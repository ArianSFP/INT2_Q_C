"""CUDA-free frozen helpers for the Tier-B procedural-anchor search."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
TIER_A_DIR = PACKAGE_DIR.parent / "initialization_anchor_oracle"
TIER_A_COMMON_PATH = TIER_A_DIR / "common.py"
EXPECTED_TIER_A_COMMON_SHA256 = "4e92cd001963f232721b2cf37d90b53f3992c162e1bad2f5f28a3471555c0d11"
CANDIDATE_LOCK_PATH = PACKAGE_DIR / "candidate_lock.json"
CANDIDATE_LOCK_FILE_SHA256 = "bd1376d9bf4b13620d4a7c6c48a24cecd82d0054fa6899f4b343bad1ace23f23"
CANDIDATE_LOCK_INTERNAL_SHA256 = "bbfab21a0b5ec2c2ec2e40e40e64f31e44de1a574d313d74813a7d76d9340734"
LOCK_PLACEHOLDER = "TO_BE_FILLED_AFTER_CANONICAL_FREEZE"
SCHEMA = "qwen3_initialization_anchor_tier_b_result_v1"
QWEN_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
MCORE_REVISION = "1cb3264479f28b8526db3d335faa9c5ef2183989"

PP_SIZES = (1, 2, 4, 8)
EP_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
ETP_SIZES = (1, 2, 4)
ASSIGNMENTS = ("contiguous", "round_robin")
PACKINGS = ("separate_gate_up_down", "fused_gate_up_then_down", "fused_up_gate_then_down")
LAYOUTS_PER_SEED = 576
LOGICAL_CANDIDATES = 37_748_736
EFFECTIVE_LAYOUTS_PER_SEED = 432
EFFECTIVE_CANDIDATES = 28_311_552
SEED_SHARD_SIZE = 256
STAGE0_TOP_K = 2048
FULL_FIT = 32_768
FULL_SCORE = 32_768
STAGE0_FIT = 256
STAGE0_SCORE = 256

DOMAIN_IDS = ("source",) + tuple(f"gaussian_{index:02d}" for index in range(16)) + tuple(
    f"scramble_{index:02d}" for index in range(16)
)
NULL_DOMAIN_IDS = DOMAIN_IDS[1:]

TARGET_F = 0.8
CURRENT_F = 0.9888693569009007
COMPOSITE_F = 0.936397621
TARGET_WEIGHTS = 28_311_552
METADATA_BYTES = 80
CURRENT_WORST_READ_AMP = 1.169444
WEIGHTS_PER_EXPERT = 4_718_592


class ProtocolError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def strict_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise ProtocolError(
            f"{label} keys mismatch; missing={sorted(expected_set-observed)}, extra={sorted(observed-expected_set)}"
        )


def _load_tier_a_common():
    if sha256_file(TIER_A_COMMON_PATH) != EXPECTED_TIER_A_COMMON_SHA256:
        raise ProtocolError("frozen Tier-A common dependency SHA-256 mismatch")
    name = "_qwen_frozen_tier_a_common_for_tier_b"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TIER_A_COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot load frozen Tier-A source/firewall helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TIER_A = _load_tier_a_common()
SourceRow = TIER_A.SourceRow
ROWS = TIER_A.ROWS
COLUMNS = TIER_A.COLUMNS
WEIGHTS_PER_MATRIX = TIER_A.WEIGHTS_PER_MATRIX


def load_candidate_lock(path: Path = CANDIDATE_LOCK_PATH) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if sha256_bytes(raw) != CANDIDATE_LOCK_FILE_SHA256:
        raise ProtocolError("Tier-B candidate lock file SHA-256 mismatch")
    lock = json.loads(raw.decode())
    seal = str(lock.get("lock_sha256", ""))
    if seal != CANDIDATE_LOCK_INTERNAL_SHA256:
        raise ProtocolError("Tier-B internal lock literal mismatch")
    pattern = rb'("lock_sha256"\s*:\s*")([0-9a-f]{64})(")'
    matches = list(re.finditer(pattern, raw))
    if len(matches) != 1:
        raise ProtocolError("Tier-B lock must contain exactly one internal seal")
    match = matches[0]
    normalized = raw[: match.start(2)] + LOCK_PLACEHOLDER.encode() + raw[match.end(2) :]
    if sha256_bytes(normalized) != CANDIDATE_LOCK_INTERNAL_SHA256:
        raise ProtocolError("Tier-B placeholder-normalized seal mismatch")
    if lock.get("schema") != "qwen3_initialization_anchor_tier_b_candidate_lock_v1":
        raise ProtocolError("Tier-B lock schema mismatch")
    if lock.get("status") != "FROZEN_BEFORE_TIER_B_PAYLOAD_ACCESS":
        raise ProtocolError("Tier-B lock status mismatch")
    if lock["logical_key_space"]["logical_candidate_count"] != LOGICAL_CANDIDATES:
        raise ProtocolError("logical candidate count mismatch")
    if lock["search_cascade"]["search_domain_count"] != len(DOMAIN_IDS):
        raise ProtocolError("search-domain count mismatch")
    if lock["provenance"]["tier_a_common_dependency_sha256"] != EXPECTED_TIER_A_COMMON_SHA256:
        raise ProtocolError("Tier-A dependency binding mismatch")
    if lock["equivalence_deduplication"]["equivalence_map_sha256"] != equivalence_map_sha256():
        raise ProtocolError("equivalence-map hash mismatch")
    return lock


def default_workspace_root() -> Path:
    return PACKAGE_DIR.parents[2]


def load_source_rows(workspace_root: Path | None = None) -> tuple[SourceRow, ...]:
    return TIER_A.load_frozen_source_rows(workspace_root)


def exclusion_binding(workspace_root: Path | None = None) -> dict[str, Any]:
    _, status = TIER_A.load_exclusion_binding(workspace_root)
    return status


def validate_aux_directory(aux_dir: Path, rows: Sequence[SourceRow]) -> dict[str, Path]:
    return TIER_A.validate_aux_directory(aux_dir, rows)


def canonical_to_native_flat(role: str, canonical: np.ndarray) -> np.ndarray:
    return TIER_A.canonical_to_native_flat(role, canonical)


def decode_bfloat16_words(words: np.ndarray) -> np.ndarray:
    return TIER_A.decode_bfloat16_words(words)


def logical_ordinal(
    base_seed: int,
    pp_index: int,
    ep_index: int,
    etp_index: int,
    assignment_index: int,
    packing_index: int,
) -> int:
    if not 0 <= base_seed < 65_536:
        raise ProtocolError("base seed outside frozen u16 range")
    ordinal = base_seed
    for index, size in (
        (pp_index, 4), (ep_index, 8), (etp_index, 3), (assignment_index, 2), (packing_index, 3)
    ):
        if not 0 <= index < size:
            raise ProtocolError("candidate axis index out of range")
        ordinal = ordinal * size + index
    return ordinal


@dataclass(frozen=True)
class CandidateKey:
    ordinal: int
    base_seed: int
    pp_index: int
    ep_index: int
    etp_index: int
    assignment_index: int
    packing_index: int

    @property
    def pipeline_parallel_size(self) -> int:
        return PP_SIZES[self.pp_index]

    @property
    def expert_parallel_size(self) -> int:
        return EP_SIZES[self.ep_index]

    @property
    def expert_tensor_parallel_size(self) -> int:
        return ETP_SIZES[self.etp_index]

    @property
    def expert_assignment(self) -> str:
        return ASSIGNMENTS[self.assignment_index]

    @property
    def projection_packing(self) -> str:
        return PACKINGS[self.packing_index]

    @property
    def id(self) -> str:
        return (
            f"{self.ordinal:08d}|seed={self.base_seed}|pp={self.pipeline_parallel_size}"
            f"|ep={self.expert_parallel_size}|etp={self.expert_tensor_parallel_size}"
            f"|assign={self.expert_assignment}|pack={self.projection_packing}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "id": self.id,
            "base_seed": self.base_seed,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
            "expert_assignment": self.expert_assignment,
            "projection_packing": self.projection_packing,
        }


def decode_ordinal(ordinal: int) -> CandidateKey:
    if not 0 <= int(ordinal) < LOGICAL_CANDIDATES:
        raise ProtocolError("logical candidate ordinal out of range")
    value = int(ordinal)
    packing = value % 3
    value //= 3
    assignment = value % 2
    value //= 2
    etp = value % 3
    value //= 3
    ep = value % 8
    value //= 8
    pp = value % 4
    seed = value // 4
    candidate = CandidateKey(int(ordinal), seed, pp, ep, etp, assignment, packing)
    if logical_ordinal(seed, pp, ep, etp, assignment, packing) != int(ordinal):
        raise ProtocolError("candidate ordinal roundtrip failure")
    return candidate


def representative_ordinals(seed_start: int, seed_stop: int) -> np.ndarray:
    if not (0 <= seed_start <= seed_stop <= 65_536):
        raise ProtocolError("invalid seed shard")
    values = np.empty((seed_stop - seed_start) * EFFECTIVE_LAYOUTS_PER_SEED, dtype=np.uint64)
    cursor = 0
    for seed in range(seed_start, seed_stop):
        for pp_index in (0, 2, 3):
            for ep_index in range(8):
                for etp_index in range(3):
                    for assignment_index in range(2):
                        for packing_index in range(3):
                            values[cursor] = logical_ordinal(
                                seed, pp_index, ep_index, etp_index, assignment_index, packing_index
                            )
                            cursor += 1
    if cursor != len(values):
        raise ProtocolError("representative enumeration accounting mismatch")
    return values


def equivalence_map_object() -> dict[str, Any]:
    return {
        "schema": "tier_b_pp_equivalence_v1",
        "end_to_end_seed_formula": "base+100*pp_rank+1024+100*ep_rank+etp_rank",
        "layer": 15,
        "classes": [
            {"representative_pp_index": 0, "representative_pp": 1, "member_pp_indices": [0, 1], "members": [1, 2], "pipeline_rank": 0, "local_layer": 15},
            {"representative_pp_index": 2, "representative_pp": 4, "member_pp_indices": [2], "members": [4], "pipeline_rank": 1, "local_layer": 3},
            {"representative_pp_index": 3, "representative_pp": 8, "member_pp_indices": [3], "members": [8], "pipeline_rank": 2, "local_layer": 3},
        ],
        "all_other_axes_preserved": True,
        "tie_rule": "representative has the smallest logical ordinal",
    }


def equivalence_map_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(equivalence_map_object()))


@dataclass(frozen=True)
class PlanRow:
    source: SourceRow
    fit: tuple[int, ...]
    score: tuple[int, ...]
    fit_sha256_le_u32: str
    score_sha256_le_u32: str
    fit_counter_stop: int
    score_counter_stop: int
    fit_duplicate_rejections: int
    score_duplicate_rejections: int


def _counts(total: int, items: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, items)
    return tuple(quotient + (index < remainder) for index in range(items))


def _sample(
    row: SourceRow, split: str, count: int, used: set[int], domain: bytes
) -> tuple[tuple[int, ...], int, int]:
    name = row.tensor_name.encode()
    prefix = domain + b"\0" + bytes.fromhex(QWEN_REVISION) + len(name).to_bytes(2, "little") + name + split.encode()
    population = WEIGHTS_PER_MATRIX
    limit = (1 << 64) - ((1 << 64) % population)
    values: list[int] = []
    counter = 0
    duplicates = 0
    while len(values) < count:
        digest = hashlib.sha256(prefix + counter.to_bytes(8, "little")).digest()
        counter += 1
        for lane in range(4):
            raw = int.from_bytes(digest[lane * 8 : (lane + 1) * 8], "little")
            if raw >= limit:
                continue
            coordinate = raw % population
            if coordinate in used:
                duplicates += 1
                continue
            used.add(coordinate)
            values.append(coordinate)
            if len(values) == count:
                break
    values.sort()
    return tuple(values), counter, duplicates


def make_plan(
    rows: Sequence[SourceRow], *, stage0: bool
) -> tuple[PlanRow, ...]:
    eligible = [row for row in rows if not row.excluded]
    if stage0:
        eligible = [row for row in eligible if row.split == "candidate_selection"]
        fit_total, score_total = STAGE0_FIT, STAGE0_SCORE
        domain = b"QWEN3-INIT-ANCHOR-TIERB-S0-COORD-v1"
    else:
        fit_total, score_total = FULL_FIT, FULL_SCORE
        domain = b"QWEN3-INIT-ANCHOR-TIERB-FULL-COORD-v1"
    fit_counts = _counts(fit_total, len(eligible))
    score_counts = _counts(score_total, len(eligible))
    result = []
    for row, fit_count, score_count in zip(eligible, fit_counts, score_counts):
        used: set[int] = set()
        fit, fit_stop, fit_dupes = _sample(row, "fit", fit_count, used, domain)
        score, score_stop, score_dupes = _sample(row, "score", score_count, used, domain)
        fit_bytes = b"".join(struct.pack("<I", value) for value in fit)
        score_bytes = b"".join(struct.pack("<I", value) for value in score)
        result.append(
            PlanRow(
                row,
                fit,
                score,
                fit_sha256_le_u32=sha256_bytes(fit_bytes),
                score_sha256_le_u32=sha256_bytes(score_bytes),
                fit_counter_stop=fit_stop,
                score_counter_stop=score_stop,
                fit_duplicate_rejections=fit_dupes,
                score_duplicate_rejections=score_dupes,
            )
        )
    if sum(len(row.fit) for row in result) != fit_total or sum(len(row.score) for row in result) != score_total:
        raise ProtocolError("coordinate-plan accounting mismatch")
    return tuple(result)


def plan_sha256(plan: Sequence[PlanRow]) -> str:
    digest = hashlib.sha256()
    for row in plan:
        name = row.source.tensor_name.encode()
        digest.update(len(name).to_bytes(2, "little"))
        digest.update(name)
        for label, values in ((b"fit", row.fit), (b"score", row.score)):
            digest.update(label)
            digest.update(len(values).to_bytes(4, "little"))
            for value in values:
                digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def plan_json(plan: Sequence[PlanRow]) -> list[dict[str, Any]]:
    return [
        {
            "tensor_name": row.source.tensor_name,
            "expert": row.source.expert,
            "role": row.source.role,
            "split": row.source.split,
            "fit_count": len(row.fit),
            "score_count": len(row.score),
            "fit_sha256_le_u32": row.fit_sha256_le_u32,
            "score_sha256_le_u32": row.score_sha256_le_u32,
            "fit_counter_stop": row.fit_counter_stop,
            "score_counter_stop": row.score_counter_stop,
            "fit_duplicate_rejections": row.fit_duplicate_rejections,
            "score_duplicate_rejections": row.score_duplicate_rejections,
        }
        for row in plan
    ]


def _hash_uniforms(prefix: bytes, count: int) -> np.ndarray:
    values = np.empty(count, dtype=np.float64)
    scale = 1.0 / float(1 << 64)
    for index in range(count):
        digest = hashlib.sha256(prefix + index.to_bytes(8, "little")).digest()
        values[index] = (int.from_bytes(digest[:8], "little") + 0.5) * scale
    return values


def stateless_normals(domain_id: str, tensor: str, split: str, coordinates: Sequence[int]) -> np.ndarray:
    result = np.empty(len(coordinates), dtype=np.float32)
    prefix = b"QWEN3-INIT-ANCHOR-TIERB-NULL-GAUSS-v1\0" + bytes.fromhex(QWEN_REVISION)
    prefix += domain_id.encode() + b"\0" + tensor.encode() + b"\0" + split.encode() + b"\0"
    scale = 1.0 / float(1 << 64)
    for index, coordinate in enumerate(coordinates):
        digest = hashlib.sha256(prefix + int(coordinate).to_bytes(8, "little")).digest()
        u1 = (int.from_bytes(digest[:8], "little") + 0.5) * scale
        u2 = (int.from_bytes(digest[8:16], "little") + 0.5) * scale
        result[index] = np.float32(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))
    return result


def permutation_and_sign(domain_id: str, tensor: str, split: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = b"QWEN3-INIT-ANCHOR-TIERB-NULL-SCRAMBLE-v1\0" + bytes.fromhex(QWEN_REVISION)
    prefix += domain_id.encode() + b"\0" + tensor.encode() + b"\0" + split.encode() + b"\0"
    keyed = []
    signs = np.empty(count, dtype=np.float32)
    for index in range(count):
        digest = hashlib.sha256(prefix + index.to_bytes(4, "little")).digest()
        keyed.append((digest[:16], index))
        signs[index] = 1.0 if (digest[16] & 1) == 0 else -1.0
    keyed.sort()
    return np.asarray([index for _, index in keyed], dtype=np.int64), signs


def fit_affine_moments(w: Sequence[float], g: Sequence[float]) -> dict[str, Any]:
    return TIER_A.fit_affine_moments(w, g)


def score_affine_moments(
    w: Sequence[float], g: Sequence[float], alpha: float, mu: float, fit_mean_w: float
) -> dict[str, Any]:
    return TIER_A.score_affine_moments(w, g, alpha, mu, fit_mean_w)


def fold_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return TIER_A.fold_statistics(rows)


def metric_from_sse(sse: float, baseline: float) -> dict[str, float]:
    return TIER_A.metric_from_sse(sse, baseline)


def physical_ledger() -> dict[str, Any]:
    side_bpw = METADATA_BYTES * 8.0 / TARGET_WEIGHTS
    current_required = 1.0 - (TARGET_F / CURRENT_F) * 2.0 ** (-2.0 * side_bpw)
    composite_required = 1.0 - (TARGET_F / COMPOSITE_F) * 2.0 ** (-2.0 * side_bpw)
    payload_bytes = WEIGHTS_PER_EXPERT * 2.5 / 8.0
    read_amp = CURRENT_WORST_READ_AMP + 20.0 / payload_bytes
    return {
        "target_weights": TARGET_WEIGHTS,
        "metadata_bytes_total": METADATA_BYTES,
        "side_bpw": side_bpw,
        "metadata_adjusted_composite_required_capture": composite_required,
        "metadata_adjusted_standalone_required_capture": current_required,
        "global_lineage_descriptor_bytes": 8,
        "per_matrix_affine_bytes": 4,
        "learned_generator_table_bytes": 0,
        "external_generator_read_bytes": 0,
        "metadata_read_bytes_per_expert": 20,
        "current_worst_cold_read_amplification": CURRENT_WORST_READ_AMP,
        "conservative_appended_cold_read_amplification": read_amp,
        "strict_read_amplification_max_exclusive": 2.0,
        "passes_read_gate": read_amp < 2.0,
    }


def make_decision(source_folds: Mapping[str, Any], null_captures: Mapping[str, float]) -> dict[str, Any]:
    if set(null_captures) != set(NULL_DOMAIN_IDS):
        raise ProtocolError("decision requires exactly the 32 frozen null captures")
    ledger = physical_ledger()
    raw = float(source_folds["pooled"]["capture"])
    max_null_id, max_null = max(null_captures.items(), key=lambda item: (float(item[1]), item[0]))
    correction = max(0.0, float(max_null))
    corrected = raw - correction
    se = float(source_folds["whole_expert_capture_standard_error"])
    lower = corrected - 3.0 * se
    upper = corrected + 3.0 * se
    all_experts = all(float(row["capture"]) > 0.0 for row in source_folds["whole_experts"])
    all_roles = all(float(row["capture"]) > 0.0 for row in source_folds["roles"])
    beats_all_nulls = all(raw > float(value) for value in null_captures.values())
    composite = float(ledger["metadata_adjusted_composite_required_capture"])
    standalone = float(ledger["metadata_adjusted_standalone_required_capture"])
    gates = beats_all_nulls and all_experts and all_roles and bool(ledger["passes_read_gate"])
    if upper < composite:
        state = "HARD_KILL_BOUNDED_TIER_B_PROCEDURAL_SET"
    elif lower >= standalone and gates:
        state = "STANDALONE_PROCEDURAL_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL"
    elif lower >= composite and gates:
        state = "COMPOSITE_PROCEDURAL_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL"
    else:
        state = "INCONCLUSIVE_OR_COMPOSITE_ONLY"
    return {
        "state": state,
        "raw_source_validation_capture": raw,
        "maximum_null_validation_capture": float(max_null),
        "maximum_null_domain": max_null_id,
        "applied_null_correction": correction,
        "bias_corrected_capture": corrected,
        "whole_expert_capture_standard_error": se,
        "bias_corrected_lower_3se": lower,
        "bias_corrected_upper_3se": upper,
        "metadata_adjusted_composite_required_capture": composite,
        "metadata_adjusted_standalone_required_capture": standalone,
        "source_beats_all_32_nulls": beats_all_nulls,
        "all_four_whole_expert_folds_positive": all_experts,
        "both_role_folds_positive": all_roles,
        "read_gate_passes": bool(ledger["passes_read_gate"]),
        "empirical_randomization_p_upper": (1 + sum(float(value) >= raw for value in null_captures.values())) / 33.0,
    }


def environment_has_cuda_imports() -> bool:
    return "torch" in sys.modules or "cupy" in sys.modules
