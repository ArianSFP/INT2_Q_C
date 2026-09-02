#!/usr/bin/env python3
"""Source-independent contracts for full-PTQ matched-Gaussian controls.

The module deliberately has no repository-relative imports.  NumPy is
injected or imported only by the explicit matrix generator; importing the
contract itself cannot open a model, a STRATA artifact, or a CUDA context.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA = "uwfa-sc-v9-matched-gaussian-ptq-producer-v0"
MOMENT_CONTRACT_SCHEMA = "uwfa-sc-v9-bf16-matrix-moment-contract-v1"
MOMENT_RECEIPT_SCHEMA = "uwfa-sc-v9-bf16-moment-replay-receipt-v1"
CONTROL_BINDING_SCHEMA = "uwfa-matched-gaussian-control-binding-v9"
SCORE_SCHEMA = "uwfa-bound-baseline-score-v8"
CONTROL_SEEDS = (
    10619863,
    10619881,
    10619909,
    10619927,
    10619953,
    10619971,
    10619999,
    10620017,
)
ROLES = ("gate", "up", "down")
CURRENT_EXPERTS = 6
CURRENT_HIDDEN = 2048
CURRENT_INTERMEDIATE = 768
VALUES_PER_MATRIX = CURRENT_HIDDEN * CURRENT_INTERMEDIATE
CURRENT_WEIGHTS = 3 * CURRENT_EXPERTS * VALUES_PER_MATRIX
GENERATOR_DOMAIN = b"UWFA-SC-V9-FULL-PTQ-MATCHED-GAUSSIAN-PCG64-v1\x00"
MEAN_RMS_TOLERANCE = 2.0 ** -17
CENTERED_RMS_RELATIVE_TOLERANCE = 2.0 ** -15


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def digest(value: Any, label: str) -> str:
    require(is_sha256(value), f"{label} must be SHA-256 hex")
    return str(value)


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    require(field not in result, f"seal field already present: {field}")
    result[field] = sha256(canonical_json(result))
    return result


def validate_seal(value: Mapping[str, Any], field: str) -> None:
    claimed = digest(value.get(field), field)
    clean = dict(value)
    clean.pop(field)
    require(sha256(canonical_json(clean)) == claimed, f"{field} integrity")


def validate_generator_capsule(payload: bytes) -> dict[str, Any]:
    """Validate a self-contained source capsule used by independent replay."""
    require(isinstance(payload, bytes) and bool(payload), "generator capsule bytes")
    try:
        record = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("generator capsule JSON") from exc
    require(isinstance(record, dict), "generator capsule object")
    require(
        set(record) == {
            "schema",
            "status",
            "members",
            "runtime_distribution_closure_sha256",
            "capsule_sha256",
        },
        "generator capsule fields",
    )
    require(record["schema"] == "uwfa-sc-v9-gaussian-generator-source-capsule-v1", "generator capsule schema")
    require(record["status"] == "AUTHENTICATED_SOURCE_BYTES_FOR_INDEPENDENT_REPLAY", "generator capsule status")
    digest(record["runtime_distribution_closure_sha256"], "runtime distribution closure")
    rows = record["members"]
    require(isinstance(rows, list) and bool(rows), "generator capsule members")
    names = []
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row) == {"name", "bytes", "sha256", "source_base64"},
            "generator capsule member fields",
        )
        name = row["name"]
        require(isinstance(name, str) and name and "/" not in name and "\\" not in name, "generator member name")
        require(name not in names, "duplicate generator member")
        names.append(name)
        try:
            source = base64.b64decode(row["source_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("generator member base64") from exc
        require(type(row["bytes"]) is int and row["bytes"] == len(source), "generator member bytes")
        require(digest(row["sha256"], "generator member digest") == sha256(source), "generator member hash")
    require("producer_contract.py" in names and "full_ptq_producer.py" in names, "generator core source coverage")
    validate_seal(record, "capsule_sha256")
    return record


def f64_hex(value: float) -> str:
    require(math.isfinite(value), "finite binary64")
    return struct.pack("<d", float(value)).hex()


def from_f64_hex(value: Any, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, str) and len(value) == 16, f"{label} binary64 bits")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} binary64 encoding") from exc
    observed, = struct.unpack("<d", raw)
    require(math.isfinite(observed), f"{label} finite")
    if positive:
        require(observed > 0.0, f"{label} positive")
    return observed


def role_shape(role: str) -> tuple[int, int]:
    require(role in ROLES, "SwiGLU role")
    return (
        (CURRENT_HIDDEN, CURRENT_INTERMEDIATE)
        if role == "down"
        else (CURRENT_INTERMEDIATE, CURRENT_HIDDEN)
    )


def matrix_key(slot: int, role: str) -> str:
    require(type(slot) is int and 0 <= slot < CURRENT_EXPERTS, "slot")
    require(role in ROLES, "role")
    return f"slot/{slot}/role/{role}"


def expected_matrix_order() -> tuple[tuple[int, str], ...]:
    return tuple((slot, role) for slot in range(CURRENT_EXPERTS) for role in ROLES)


@dataclass(frozen=True)
class MatrixMoment:
    ordinal: int
    slot: int
    role: str
    shape: tuple[int, int]
    values: int
    mean: float
    centered_sse: float
    energy: float
    source_matrix_bf16_sha256: str

    @property
    def rms(self) -> float:
        return math.sqrt(self.energy / self.values)

    def public_generator_key(self) -> dict[str, Any]:
        """The generator key intentionally omits model/layer/expert identity."""
        return {
            "slot": self.slot,
            "role": self.role,
            "shape": list(self.shape),
            "values": self.values,
            "mean_f64_hex": f64_hex(self.mean),
            "centered_sse_f64_hex": f64_hex(self.centered_sse),
        }


_CONTRACT_CLOSURE_FIELDS = (
    "source_artifact_sha256",
    "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "source_pipeline_sha256",
    "source_score_receipt_sha256",
    "source_moment_auditor_sha256",
)

SYMMETRIC_CODEC_CLOSURE_FIELDS = (
    "runtime_snapshot_root_sha256",
    "numpy_runtime_closure_sha256",
    "cupy_runtime_closure_sha256",
    "polar_repository_tree_sha256",
    "strata_v2_emitter_sha256",
    "strata_v2_common_sha256",
    "strata_polar_wrapper_sha256",
    "polaris_base_encoder_sha256",
    "bec_encoder_sha256",
    "expert_common_sha256",
    "run_and_pack_sha256",
    "independent_auditor_sha256",
    "frozen_auditor_sha256",
    "v8_adapter_sha256",
    "v8_protocol_sha256",
    "v8_common_sha256",
)


def validate_moment_contract(record: Any) -> tuple[dict[str, Any], tuple[MatrixMoment, ...]]:
    require(isinstance(record, dict), "moment contract object")
    required = {
        "schema",
        "status",
        "moment_semantics",
        "panel",
        "source_closure",
        "matrices",
        "moment_contract_sha256",
    }
    require(set(record) == required, "moment contract fields")
    require(record["schema"] == MOMENT_CONTRACT_SCHEMA, "moment contract schema")
    require(record["status"] == "AUTHENTICATED_EXTERNAL_SOURCE_MOMENTS", "moment contract status")
    require(
        record["moment_semantics"]
        == "ORIGINAL_BF16_MATRIX_STORAGE_ORIENTATION_BINARY64_MEAN_AND_CENTERED_SSE",
        "moment semantics",
    )
    panel = record["panel"]
    require(
        panel
        == {
            "experts": CURRENT_EXPERTS,
            "hidden": CURRENT_HIDDEN,
            "intermediate": CURRENT_INTERMEDIATE,
            "roles": list(ROLES),
            "weights": CURRENT_WEIGHTS,
            "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
        },
        "current-format panel geometry",
    )
    closure = record["source_closure"]
    require(isinstance(closure, dict) and set(closure) == set(_CONTRACT_CLOSURE_FIELDS), "source closure fields")
    for field in _CONTRACT_CLOSURE_FIELDS:
        digest(closure[field], f"source closure {field}")
    raw_rows = record["matrices"]
    require(isinstance(raw_rows, list) and len(raw_rows) == 3 * CURRENT_EXPERTS, "eighteen moment rows")
    moments: list[MatrixMoment] = []
    row_fields = {
        "matrix_ordinal",
        "slot",
        "role",
        "shape",
        "values",
        "mean_f64_hex",
        "centered_sse_f64_hex",
        "energy_f64_hex",
        "source_matrix_bf16_sha256",
    }
    for ordinal, (raw, expected) in enumerate(zip(raw_rows, expected_matrix_order(), strict=True)):
        require(isinstance(raw, dict) and set(raw) == row_fields, f"moment row fields {ordinal}")
        slot, role = expected
        shape = role_shape(role)
        require(raw["matrix_ordinal"] == ordinal, f"moment ordinal {ordinal}")
        require(raw["slot"] == slot and raw["role"] == role, f"moment slot/role {ordinal}")
        require(raw["shape"] == list(shape), f"moment shape {ordinal}")
        require(raw["values"] == VALUES_PER_MATRIX, f"moment values {ordinal}")
        mean = from_f64_hex(raw["mean_f64_hex"], f"mean {ordinal}")
        centered_sse = from_f64_hex(raw["centered_sse_f64_hex"], f"centered SSE {ordinal}", positive=True)
        energy = from_f64_hex(raw["energy_f64_hex"], f"energy {ordinal}", positive=True)
        expected_energy = centered_sse + VALUES_PER_MATRIX * mean * mean
        # Source auditors may obtain energy by a separate FP64 sum of squares
        # while centered SSE is accumulated after a mean pass.  Bind both,
        # but allow only their ordinary summation-order discrepancy.
        tolerance = max(4096.0 * math.ulp(expected_energy), 2.0 ** -44 * expected_energy)
        require(abs(expected_energy - energy) <= tolerance, f"moment energy identity {ordinal}")
        moments.append(
            MatrixMoment(
                ordinal=ordinal,
                slot=slot,
                role=role,
                shape=shape,
                values=VALUES_PER_MATRIX,
                mean=mean,
                centered_sse=centered_sse,
                energy=energy,
                source_matrix_bf16_sha256=digest(
                    raw["source_matrix_bf16_sha256"], f"source matrix {ordinal}"
                ),
            )
        )
    validate_seal(record, "moment_contract_sha256")
    return dict(record), tuple(moments)


def build_moment_contract_from_authenticated_bf16(
    np: Any,
    matrix_inputs: Sequence[Mapping[str, Any]],
    source_closure: Mapping[str, str],
) -> dict[str, Any]:
    """Build the external moment contract from already-authenticated bytes.

    Path opening and source-to-slot authorization remain the responsibility of
    an independent dispatcher.  This function accepts bytes only and uses no
    model/tensor identity.
    """
    require(len(matrix_inputs) == 18, "moment source matrix coverage")
    require(set(source_closure) == set(_CONTRACT_CLOSURE_FIELDS), "moment source closure")
    for name in _CONTRACT_CLOSURE_FIELDS:
        digest(source_closure[name], f"moment source closure {name}")
    rows = []
    for ordinal, (item, expected) in enumerate(
        zip(matrix_inputs, expected_matrix_order(), strict=True)
    ):
        require(
            isinstance(item, Mapping)
            and set(item) == {"matrix_ordinal", "slot", "role", "shape", "bf16_bytes"},
            f"moment source fields {ordinal}",
        )
        slot, role = expected
        shape = role_shape(role)
        require(item["matrix_ordinal"] == ordinal, f"moment source ordinal {ordinal}")
        require(item["slot"] == slot and item["role"] == role, f"moment source slot/role {ordinal}")
        require(item["shape"] == list(shape), f"moment source shape {ordinal}")
        payload = item["bf16_bytes"]
        require(isinstance(payload, bytes) and len(payload) == 2 * VALUES_PER_MATRIX, f"moment source bytes {ordinal}")
        words = np.frombuffer(payload, dtype="<u2").reshape(shape)
        mean, centered_sse, energy = measured_moments(np, words)
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "slot": slot,
                "role": role,
                "shape": list(shape),
                "values": VALUES_PER_MATRIX,
                "mean_f64_hex": f64_hex(mean),
                "centered_sse_f64_hex": f64_hex(centered_sse),
                "energy_f64_hex": f64_hex(energy),
                "source_matrix_bf16_sha256": sha256(payload),
            }
        )
    clean = {
        "schema": MOMENT_CONTRACT_SCHEMA,
        "status": "AUTHENTICATED_EXTERNAL_SOURCE_MOMENTS",
        "moment_semantics": "ORIGINAL_BF16_MATRIX_STORAGE_ORIENTATION_BINARY64_MEAN_AND_CENTERED_SSE",
        "panel": {
            "experts": CURRENT_EXPERTS,
            "hidden": CURRENT_HIDDEN,
            "intermediate": CURRENT_INTERMEDIATE,
            "roles": list(ROLES),
            "weights": CURRENT_WEIGHTS,
            "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
        },
        "source_closure": dict(source_closure),
        "matrices": rows,
    }
    result = sealed(clean, "moment_contract_sha256")
    validate_moment_contract(result)
    return result


def derive_matrix_seed(global_seed: int, moment: MatrixMoment) -> int:
    require(global_seed in CONTROL_SEEDS, "frozen control seed")
    key = canonical_json(moment.public_generator_key())
    payload = (
        GENERATOR_DOMAIN
        + struct.pack("<Q", global_seed)
        + struct.pack("<I", moment.ordinal)
        + struct.pack("<Q", len(key))
        + key
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def fp32_to_bf16_rne(np: Any, values: Any) -> Any:
    fp32 = np.ascontiguousarray(values, dtype=np.float32)
    raw = fp32.view(np.uint32)
    rounded = raw + np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype("<u2"))


def bf16_to_f64(np: Any, words: Any) -> Any:
    source = np.ascontiguousarray(words, dtype="<u2")
    return (source.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)


def measured_moments(np: Any, words: Any) -> tuple[float, float, float]:
    values = bf16_to_f64(np, words).reshape(-1)
    require(bool(np.isfinite(values).all()), "generated BF16 finite")
    mean = float(np.mean(values, dtype=np.float64))
    centered = values - mean
    centered_sse = float(np.sum(centered * centered, dtype=np.float64))
    energy = float(np.sum(values * values, dtype=np.float64))
    require(centered_sse > 0.0 and energy > 0.0, "generated BF16 nondegenerate")
    return mean, centered_sse, energy


def generate_matrix_bf16(np: Any, moment: MatrixMoment, global_seed: int) -> tuple[Any, dict[str, Any]]:
    """Generate one deterministic BF16 Gaussian matrix and close its moments.

    PCG64 and the affine/rounding loop are part of the source law.  The
    generator chooses no seed, profile, transform, or codec candidate from the
    source.  Six fixed affine iterations correct the small BF16-RNE bias.
    """
    seed128 = derive_matrix_seed(global_seed, moment)
    rng = np.random.Generator(np.random.PCG64(seed128))
    normal = rng.standard_normal(moment.values, dtype=np.float64)
    normal -= np.mean(normal, dtype=np.float64)
    normal_sse = float(np.sum(normal * normal, dtype=np.float64))
    require(normal_sse > 0.0 and math.isfinite(normal_sse), "Gaussian draw nondegenerate")
    normal *= math.sqrt(moment.centered_sse / normal_sse)
    scale = 1.0
    offset = moment.mean
    best: tuple[float, Any, tuple[float, float, float], int] | None = None
    for iteration in range(6):
        words = fp32_to_bf16_rne(np, offset + scale * normal)
        observed = measured_moments(np, words)
        observed_mean, observed_sse, _ = observed
        mean_normalized = abs(observed_mean - moment.mean) / max(moment.rms, np.finfo(np.float64).tiny)
        rms_relative = abs(math.sqrt(observed_sse / moment.centered_sse) - 1.0)
        objective = max(
            mean_normalized / MEAN_RMS_TOLERANCE,
            rms_relative / CENTERED_RMS_RELATIVE_TOLERANCE,
        )
        if best is None or objective < best[0]:
            best = (objective, words.copy(), observed, iteration)
        offset += moment.mean - observed_mean
        scale *= math.sqrt(moment.centered_sse / observed_sse)
    assert best is not None
    _, words, observed, iteration = best
    achieved_mean, achieved_sse, achieved_energy = observed
    mean_normalized = abs(achieved_mean - moment.mean) / max(moment.rms, np.finfo(np.float64).tiny)
    rms_relative = abs(math.sqrt(achieved_sse / moment.centered_sse) - 1.0)
    require(mean_normalized <= MEAN_RMS_TOLERANCE, "BF16 control mean tolerance")
    require(rms_relative <= CENTERED_RMS_RELATIVE_TOLERANCE, "BF16 control centered RMS tolerance")
    payload = np.ascontiguousarray(words.reshape(moment.shape), dtype="<u2")
    receipt = {
        "matrix_ordinal": moment.ordinal,
        "slot": moment.slot,
        "role": moment.role,
        "shape": list(moment.shape),
        "values": moment.values,
        "global_seed": global_seed,
        "derived_pcg64_seed_u128_hex": seed128.to_bytes(16, "little").hex(),
        "selected_affine_iteration": iteration,
        "requested_mean_f64_hex": f64_hex(moment.mean),
        "achieved_mean_f64_hex": f64_hex(achieved_mean),
        "requested_centered_sse_f64_hex": f64_hex(moment.centered_sse),
        "achieved_centered_sse_f64_hex": f64_hex(achieved_sse),
        "achieved_energy_f64_hex": f64_hex(achieved_energy),
        "mean_error_over_source_rms": mean_normalized,
        "centered_rms_relative_error": rms_relative,
        "mean_tolerance": MEAN_RMS_TOLERANCE,
        "centered_rms_relative_tolerance": CENTERED_RMS_RELATIVE_TOLERANCE,
        "control_bf16_sha256": sha256(payload.tobytes(order="C")),
    }
    return payload, receipt


def build_universal_route() -> bytes:
    """Current-format numeric route using only canonical panel slots."""
    rows = []
    for slot in range(CURRENT_EXPERTS):
        for role_id, role in enumerate(ROLES):
            axis_id = 1 if role == "down" else 0
            rows.append(struct.pack(">HHBBH", 0, slot, role_id, axis_id, CURRENT_INTERMEDIATE))
    payload = b"".join(rows)
    require(len(payload) == 144, "current route bytes")
    return payload


def universal_format_geometry() -> dict[str, Any]:
    """Source-independent format geometry used by the v9 control bridge.

    Profiles, symbol counts, labels, role contribution counts and arithmetic
    lengths are intentionally absent: an independently re-encoded Gaussian
    source is allowed to derive each of them through the same PTQ algorithm.
    """
    block_logn = [21, 21] * 6 + [20, 20, 20]
    owners = [[slot] for slot in range(6) for _ in range(2)] + [[0, 1], [2, 3], [4, 5]]
    return {
        "schema": "uwfa-sc-v9-universal-strata15-format-geometry-v1",
        "experts": CURRENT_EXPERTS,
        "weights": CURRENT_WEIGHTS,
        "expert_shapes": [
            {"slot": slot, "hidden": CURRENT_HIDDEN, "intermediate": CURRENT_INTERMEDIATE}
            for slot in range(CURRENT_EXPERTS)
        ],
        "roles": list(ROLES),
        "blocks": [
            {"ordinal": ordinal, "logn": logn, "owner_slots": owner}
            for ordinal, (logn, owner) in enumerate(zip(block_logn, owners, strict=True))
        ],
        "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
    }


def universal_format_geometry_sha256() -> str:
    return sha256(canonical_json(universal_format_geometry()))


def build_moment_receipt(
    *,
    seed: int,
    moment_contract_sha256: str,
    generator_capsule_sha256: str,
    matrix_receipts: Sequence[Mapping[str, Any]],
    source_panel_manifest_sha256: str,
) -> dict[str, Any]:
    require(seed in CONTROL_SEEDS, "receipt seed")
    require(len(matrix_receipts) == 18, "moment receipt matrix coverage")
    clean = {
        "schema": MOMENT_RECEIPT_SCHEMA,
        "status": "PASS_RECOMPUTED_BF16_MATRIX_MOMENT_MATCH",
        "seed": seed,
        "moment_contract_sha256": digest(moment_contract_sha256, "moment contract"),
        "generator_capsule_sha256": digest(generator_capsule_sha256, "generator capsule"),
        "universal_format_geometry_sha256": universal_format_geometry_sha256(),
        "source_panel_manifest_sha256": digest(source_panel_manifest_sha256, "source panel manifest"),
        "matrices": [dict(row) for row in matrix_receipts],
        "all_eighteen_within_frozen_tolerances": True,
    }
    return sealed(clean, "moment_match_receipt_sha256")


def build_score_receipt(
    *,
    artifact_sha256: str,
    artifact_bytes: int,
    reconstruction_sha256: str,
    control_full_geometry_sha256: str,
    independent_decoder_source_sha256: str,
    sse: float,
    energy: float,
) -> dict[str, Any]:
    require(artifact_bytes > 0, "artifact bytes")
    require(math.isfinite(sse) and sse > 0.0, "score SSE")
    require(math.isfinite(energy) and energy > 0.0, "score energy")
    clean = {
        "schema": SCORE_SCHEMA,
        "status": "PASS_INDEPENDENT_BASELINE_SCORE",
        "artifact_sha256": digest(artifact_sha256, "artifact"),
        "artifact_bytes": artifact_bytes,
        "weights": CURRENT_WEIGHTS,
        "relative_mse": sse / energy,
        "sse_fp64": sse,
        "source_energy_fp64": energy,
        "normalization": "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
        "reconstruction_f64_sha256": digest(reconstruction_sha256, "reconstruction"),
        "original_source_panel_sha256": digest(control_full_geometry_sha256, "control full geometry"),
        "independent_decoder_source_sha256": digest(independent_decoder_source_sha256, "decoder"),
    }
    return sealed(clean, "score_receipt_sha256")


def build_control_binding_v9(
    *,
    seed: int,
    source_closure: Mapping[str, str],
    generator_capsule_sha256: str,
    moment_match_receipt_sha256: str,
    source_panel_manifest_sha256: str,
    control_artifact_sha256: str,
    control_full_geometry_sha256: str,
    control_structural_geometry_sha256: str,
    symmetric_codec_closure: Mapping[str, str],
) -> dict[str, Any]:
    require(seed in CONTROL_SEEDS, "binding seed")
    for name in _CONTRACT_CLOSURE_FIELDS:
        digest(source_closure.get(name), f"source closure {name}")
    require(
        set(symmetric_codec_closure) == set(SYMMETRIC_CODEC_CLOSURE_FIELDS),
        "symmetric codec closure fields",
    )
    for name in SYMMETRIC_CODEC_CLOSURE_FIELDS:
        digest(symmetric_codec_closure[name], f"symmetric closure {name}")
    clean = {
        "schema": CONTROL_BINDING_SCHEMA,
        "seed": seed,
        "source_artifact_sha256": source_closure["source_artifact_sha256"],
        "source_full_geometry_sha256": source_closure["source_full_geometry_sha256"],
        "source_structural_geometry_sha256": source_closure["source_structural_geometry_sha256"],
        "pipeline_sha256": source_closure["source_pipeline_sha256"],
        "source_score_receipt_sha256": source_closure["source_score_receipt_sha256"],
        "source_moment_auditor_sha256": source_closure["source_moment_auditor_sha256"],
        "universal_format_geometry_sha256": universal_format_geometry_sha256(),
        "generator_capsule_sha256": digest(generator_capsule_sha256, "generator capsule"),
        "moment_match_receipt_sha256": digest(moment_match_receipt_sha256, "moment receipt"),
        "source_panel_manifest_sha256": digest(source_panel_manifest_sha256, "source panel"),
        "control_artifact_sha256": digest(control_artifact_sha256, "control artifact"),
        "control_full_geometry_sha256": digest(control_full_geometry_sha256, "control full geometry"),
        "control_structural_geometry_sha256": digest(control_structural_geometry_sha256, "control structural geometry"),
        "symmetric_codec_closure": dict(sorted(symmetric_codec_closure.items())),
    }
    return sealed(clean, "binding_sha256")


def direct_main() -> int:
    print(
        "BLOCK_DIRECT_EXECUTION_REQUIRES_AUTHENTICATED_RUNTIME_DISPATCHER",
        flush=True,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(direct_main())
