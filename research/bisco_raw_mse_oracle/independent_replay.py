#!/usr/bin/env python3
"""Independent, state-backed replay of the frozen BiSCo auxiliary gate.

This auditor deliberately does not import :mod:`bisco_raw_mse_oracle`.  It
parses the serialized FP32 model state from a closed schema, proves that the
published FP16 decoder files are literal roundings of that state, regenerates
the held-out Qwen and matched-Gaussian matrices from the frozen auxiliary
files, and reruns the update-512 codec with a separately implemented CuPy
evaluator.  Reconstruction errors are accumulated in FP64.

The resulting receipt is sealed by a SHA-256 of canonical JSON with the seal
field omitted.  The seal detects receipt edits; the receipt's input hashes bind
the result, states, decoders, launch protocol, and all 32 auxiliary sources.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROWS = 768
COLS = 2048
VALUES_PER_MATRIX = ROWS * COLS
CHUNK_DIMENSION = 16
HIDDEN = 64
BITS_PER_STAGE = 18
ROLES = ("up", "down")
DOMAINS = ("qwen", "gaussian")
EXPERTS = tuple(range(0, 121, 8))
VALIDATION_EXPERTS = (24, 56, 88, 120)
TRAIN_EXPERTS = tuple(expert for expert in EXPERTS if expert not in VALIDATION_EXPERTS)
STATE_PARAMETER_SHAPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("s1_ew1", (16, 64)),
    ("s1_eb1", (64,)),
    ("s1_ew2", (64, 18)),
    ("s1_eb2", (18,)),
    ("s1_dw1", (18, 64)),
    ("s1_db1", (64,)),
    ("s1_dw2", (64, 16)),
    ("s1_db2", (16,)),
    ("s2_ew1", (16, 64)),
    ("s2_eb1", (64,)),
    ("s2_ew2", (64, 18)),
    ("s2_eb2", (18,)),
    ("s2_dw1", (18, 64)),
    ("s2_db1", (64,)),
    ("s2_dw2", (64, 16)),
    ("s2_db2", (16,)),
)
DECODER_NAMES = tuple(name for name, _ in STATE_PARAMETER_SHAPES if "_d" in name)
STATE_VALUES_PER_ROLE = sum(math.prod(shape) for _, shape in STATE_PARAMETER_SHAPES)
STATE_VALUES = len(ROLES) * STATE_VALUES_PER_ROLE
DECODER_VALUES = len(ROLES) * sum(
    math.prod(shape) for name, shape in STATE_PARAMETER_SHAPES if name in DECODER_NAMES
)
EXPECTED_STATE_BYTES = STATE_VALUES * 4
EXPECTED_DECODER_BYTES = DECODER_VALUES * 2
GAUSSIAN_SEED_BASE = 260_901 + 104_729
EVALUATION_BATCH_CHUNKS = 16_384
BITFLIP_SWEEPS = 1
EXPECTED_LAUNCH_SHA256 = "0d79a1b8e3cacbc345bdea464986279b0935c4cf2e20290dea75507f7fbfcd4c"
EXPECTED_RESULT_SHA256 = "5904e3887e69cf47ee4a882aeaacceb27823504c1e23eeff6adb4b3360874d92"
EXPECTED_ARTIFACT_SHA256 = {
    "qwen_training_state.fp32.bin": "86c97eb25644e9535dcd8f7b47be8a58f6e9c8b6e47ba0d89524aac6fe764881",
    "qwen_aux_up_down_decoder.fp16.bin": "c1c0837b1658681f8282050afd8bf4d16e115708ac425991e11224d07d37e685",
    "gaussian_training_state.fp32.bin": "7fc63009f57fe2803e55780984c66207334f02a2003a4d58b2f85807730c57b9",
    "gaussian_aux_up_down_decoder.fp16.bin": "8167d5003b3365b0772a0e2193fe0381fea841305ac3bfd9088642bcb27f60ba",
}

# The published evaluator squares FP32 residuals and performs a GPU FP32 sum.
# The replay squares and sums the same residuals in FP64.  A tolerance of
# gamma_128 = 128*u/(1-128*u), u=2^-24, conservatively allows one square plus
# a hierarchical GPU reduction with at most 127 dependent FP32 additions.
FP32_UNIT_ROUNDOFF = 2.0 ** -24
FP64_VS_PUBLISHED_RTOL = (128.0 * FP32_UNIT_ROUNDOFF) / (
    1.0 - 128.0 * FP32_UNIT_ROUNDOFF
)
FP64_VS_PUBLISHED_ATOL = 1.0e-10
# The separate evaluator also calculates the original FP32 reduction.  On the
# same frozen CuPy/device path that number should be much tighter than the
# cross-precision bound, while allowing harmless reduction scheduling jitter.
FP32_EMULATION_RTOL = 5.0e-7
FP32_EMULATION_ATOL = 1.0e-10
# Energy is recomputed on CPU from original FP32 values using an independent
# blocked FP64 reduction.  This permits 64 FP64 ulps per matrix-scale sum.
ENERGY_RTOL = 64.0 * (2.0 ** -52)
ENERGY_ATOL = 1.0e-12


class AuditFailure(AssertionError):
    """A fail-closed replay or binding error."""


def require(condition: bool, message: str, **details: Any) -> None:
    if not condition:
        raise AuditFailure({"message": message, **details})


def require_keys(value: Mapping[str, Any], expected: Iterable[str], where: str) -> None:
    actual = set(value)
    wanted = set(expected)
    require(actual == wanted, "field set mismatch", where=where, actual=sorted(actual), expected=sorted(wanted))


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def seal_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    require("receipt_seal" not in receipt, "receipt is already sealed")
    digest = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    sealed = copy.deepcopy(receipt)
    sealed["receipt_seal"] = {
        "algorithm": "sha256",
        "canonicalization": "sorted-key compact ASCII JSON, receipt_seal omitted",
        "sha256": digest,
    }
    return sealed


def verify_receipt_seal(receipt: Mapping[str, Any]) -> str:
    require("receipt_seal" in receipt, "receipt seal is missing")
    seal = receipt["receipt_seal"]
    require_keys(seal, ("algorithm", "canonicalization", "sha256"), "receipt_seal")
    require(seal["algorithm"] == "sha256", "unsupported receipt seal")
    unsigned = dict(receipt)
    unsigned.pop("receipt_seal")
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    require(actual == seal["sha256"], "receipt seal mismatch", actual=actual, expected=seal["sha256"])
    return actual


def expected_schema(names: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(names)
    rows: list[dict[str, Any]] = []
    offset = 0
    for role in ROLES:
        for name, shape in STATE_PARAMETER_SHAPES:
            if name not in wanted:
                continue
            count = math.prod(shape)
            rows.append(
                {
                    "role": role,
                    "parameter": name,
                    "shape": list(shape),
                    "offset_values": offset,
                    "values": count,
                }
            )
            offset += count
    return rows


EXPECTED_STATE_SCHEMA = expected_schema(name for name, _ in STATE_PARAMETER_SHAPES)
EXPECTED_DECODER_SCHEMA = expected_schema(DECODER_NAMES)


def artifact_path(result_dir: Path, filename: str) -> Path:
    require(Path(filename).name == filename, "artifact filename is not a basename", filename=filename)
    path = (result_dir / filename).resolve()
    require(path.parent == result_dir.resolve(), "artifact escapes result directory", filename=filename)
    return path


def validate_artifact_descriptor(
    descriptor: Mapping[str, Any],
    *,
    result_dir: Path,
    filename: str,
    byte_count: int,
    schema: list[dict[str, Any]],
    decoder: bool,
) -> Path:
    expected_keys = {"file", "bytes", "sha256", "schema"}
    if decoder:
        expected_keys.add("not_the_deployment_ledger_decoder")
    require_keys(descriptor, expected_keys, f"artifact:{filename}")
    require(descriptor["file"] == filename, "wrong artifact filename", actual=descriptor["file"], expected=filename)
    require(int(descriptor["bytes"]) == byte_count, "wrong artifact byte count", filename=filename)
    require(descriptor["schema"] == schema, "artifact schema is not the frozen schema", filename=filename)
    if decoder:
        require(descriptor["not_the_deployment_ledger_decoder"] is True, "decoder ledger caveat is missing")
    path = artifact_path(result_dir, filename)
    require(path.is_file(), "artifact is missing", path=str(path))
    require(path.stat().st_size == byte_count, "artifact file size mismatch", path=str(path))
    actual_hash = sha256_file(path)
    require(actual_hash == descriptor["sha256"], "artifact descriptor hash mismatch", filename=filename)
    require(actual_hash == EXPECTED_ARTIFACT_SHA256[filename], "artifact differs from frozen run_1", filename=filename)
    return path


def parse_state_file(path: Path, schema: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Parse little-endian FP32 state solely from the closed independent schema."""

    require(schema == EXPECTED_STATE_SCHEMA, "state schema is not frozen")
    raw = np.fromfile(path, dtype="<f4")
    require(raw.size == STATE_VALUES, "state value count mismatch", actual=int(raw.size), expected=STATE_VALUES)
    require(bool(np.all(np.isfinite(raw))), "state contains nonfinite values", path=str(path))
    parsed: dict[str, dict[str, np.ndarray]] = {role: {} for role in ROLES}
    for row in schema:
        offset = int(row["offset_values"])
        count = int(row["values"])
        shape = tuple(int(value) for value in row["shape"])
        parsed[str(row["role"])][str(row["parameter"])] = np.ascontiguousarray(
            raw[offset : offset + count].reshape(shape), dtype=np.float32
        )
    require(
        all(tuple(parsed[role]) == tuple(name for name, _ in STATE_PARAMETER_SHAPES) for role in ROLES),
        "parsed state parameter order mismatch",
    )
    return parsed


def decoder_bytes_from_state(state: Mapping[str, Mapping[str, np.ndarray]]) -> bytes:
    blocks = []
    for role in ROLES:
        for name in DECODER_NAMES:
            blocks.append(np.asarray(state[role][name], dtype="<f4", order="C").astype("<f2").tobytes(order="C"))
    return b"".join(blocks)


def parse_and_bind_models(result: Mapping[str, Any], result_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require_keys(result["artifacts"], DOMAINS, "artifacts")
    models: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_artifacts = result["artifacts"][domain]
        require_keys(domain_artifacts, ("training_state", "auxiliary_two_role_decoder"), f"artifacts.{domain}")
        state_name = f"{domain}_training_state.fp32.bin"
        decoder_name = f"{domain}_aux_up_down_decoder.fp16.bin"
        state_path = validate_artifact_descriptor(
            domain_artifacts["training_state"],
            result_dir=result_dir,
            filename=state_name,
            byte_count=EXPECTED_STATE_BYTES,
            schema=EXPECTED_STATE_SCHEMA,
            decoder=False,
        )
        decoder_path = validate_artifact_descriptor(
            domain_artifacts["auxiliary_two_role_decoder"],
            result_dir=result_dir,
            filename=decoder_name,
            byte_count=EXPECTED_DECODER_BYTES,
            schema=EXPECTED_DECODER_SCHEMA,
            decoder=True,
        )
        state = parse_state_file(state_path, EXPECTED_STATE_SCHEMA)
        expected_decoder = decoder_bytes_from_state(state)
        actual_decoder = decoder_path.read_bytes()
        require(
            actual_decoder == expected_decoder,
            "FP16 decoder is not the literal round-to-FP16 projection of FP32 state",
            domain=domain,
        )
        models[domain] = state
        evidence[domain] = {
            "state_file": state_name,
            "state_bytes": EXPECTED_STATE_BYTES,
            "state_sha256": sha256_file(state_path),
            "state_schema_sha256": hashlib.sha256(canonical_json_bytes(EXPECTED_STATE_SCHEMA)).hexdigest(),
            "decoder_file": decoder_name,
            "decoder_bytes": EXPECTED_DECODER_BYTES,
            "decoder_sha256": sha256_file(decoder_path),
            "decoder_schema_sha256": hashlib.sha256(canonical_json_bytes(EXPECTED_DECODER_SCHEMA)).hexdigest(),
            "decoder_equals_state_rounded_fp16": True,
            "decoder_expected_bytes_sha256": hashlib.sha256(expected_decoder).hexdigest(),
        }
    return models, evidence


def exact_float(actual: Any, expected: float, where: str) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), "expected numeric field", where=where)
    require(math.isfinite(float(actual)), "nonfinite numeric field", where=where)
    require(float(actual) == expected, "derived float is not exact", where=where, actual=actual, expected=expected)


def ledger(experts: int) -> dict[str, float | int]:
    weights = 4_718_592
    decoder_parameters = 13_536
    decoder_bytes = 2 * decoder_parameters
    header_bytes = 256
    local_scale_bytes = 12
    code_bits = (weights // CHUNK_DIMENSION) * (2 * BITS_PER_STAGE)
    code_bytes = code_bits // 8
    attributed = code_bytes + (decoder_bytes + header_bytes) / experts + local_scale_bytes
    cold = code_bytes + decoder_bytes + header_bytes + local_scale_bytes
    physical_r = 8.0 * attributed / weights
    side_r = physical_r - 2 * BITS_PER_STAGE / CHUNK_DIMENSION
    target_s = -0.5 * math.log2(0.8)
    return {
        "d": CHUNK_DIMENSION,
        "hidden": HIDDEN,
        "b1": BITS_PER_STAGE,
        "b2": BITS_PER_STAGE,
        "experts_amortized": experts,
        "decoder_parameters": decoder_parameters,
        "decoder_bytes": decoder_bytes,
        "header_bytes": header_bytes,
        "local_scale_bytes_per_expert": local_scale_bytes,
        "code_bits_per_expert": code_bits,
        "code_bytes_per_expert": code_bytes,
        "attributed_physical_bytes_per_expert": attributed,
        "cold_bytes_per_expert": cold,
        "physical_bpw": physical_r,
        "side_bpw": side_r,
        "cold_read_amplification": cold / attributed,
        "minimum_matched_s_if_gaussian_code_is_ideal": target_s + side_r,
        "target_relative_mse": 0.8 * 2.0 ** (-2.0 * physical_r),
    }


EVALUATION_KEYS = (
    "D_Qwen",
    "D_Gaussian",
    "s_match",
    "fold_s_match",
    "whole_expert_standard_error",
    "upper_s_match_2se",
    "all_whole_expert_folds_positive",
    "Gaussian_operational_gap",
    "absolute",
    "per_expert",
    "per_matrix",
)


def enforce_evaluation(evaluation: Mapping[str, Any], where: str) -> dict[str, float]:
    """Fail closed on field sets and exactly recompute every derived field."""

    require_keys(evaluation, EVALUATION_KEYS, where)
    matrices = evaluation["per_matrix"]
    require(isinstance(matrices, list) and len(matrices) == 8, "wrong per-matrix row count", where=where)
    expected_identities = [(expert, role) for role in ROLES for expert in VALIDATION_EXPERTS]
    identities: list[tuple[int, str]] = []
    sums = {
        expert: {"qwen_sse": 0.0, "qwen_energy": 0.0, "gaussian_sse": 0.0, "gaussian_energy": 0.0}
        for expert in VALIDATION_EXPERTS
    }
    for index, row in enumerate(matrices):
        require_keys(row, ("expert", "role", "qwen_sse", "qwen_energy", "gaussian_sse", "gaussian_energy"), f"{where}.per_matrix[{index}]")
        identity = (int(row["expert"]), str(row["role"]))
        identities.append(identity)
        target = sums[identity[0]]
        for field in target:
            value = float(row[field])
            require(math.isfinite(value) and value > 0.0, "invalid base matrix statistic", where=where, field=field)
            target[field] += value
    require(identities == expected_identities, "wrong per-matrix identities/order", where=where)

    experts = evaluation["per_expert"]
    require(isinstance(experts, list) and len(experts) == 4, "wrong per-expert row count", where=where)
    fold_s: list[float] = []
    for index, row in enumerate(experts):
        require_keys(
            row,
            ("expert", "qwen_sse", "qwen_energy", "gaussian_sse", "gaussian_energy", "D_Qwen", "D_Gaussian", "s_match"),
            f"{where}.per_expert[{index}]",
        )
        expert = int(row["expert"])
        require(expert == VALIDATION_EXPERTS[index], "wrong per-expert identity/order", where=where)
        for field, expected in sums[expert].items():
            exact_float(row[field], expected, f"{where}.per_expert[{index}].{field}")
        d_qwen = sums[expert]["qwen_sse"] / sums[expert]["qwen_energy"]
        d_gaussian = sums[expert]["gaussian_sse"] / sums[expert]["gaussian_energy"]
        s_value = -0.5 * math.log2(d_qwen / d_gaussian)
        exact_float(row["D_Qwen"], d_qwen, f"{where}.per_expert[{index}].D_Qwen")
        exact_float(row["D_Gaussian"], d_gaussian, f"{where}.per_expert[{index}].D_Gaussian")
        exact_float(row["s_match"], s_value, f"{where}.per_expert[{index}].s_match")
        fold_s.append(s_value)

    q_sse = sum(float(row["qwen_sse"]) for row in experts)
    q_energy = sum(float(row["qwen_energy"]) for row in experts)
    g_sse = sum(float(row["gaussian_sse"]) for row in experts)
    g_energy = sum(float(row["gaussian_energy"]) for row in experts)
    d_qwen = q_sse / q_energy
    d_gaussian = g_sse / g_energy
    s_match = -0.5 * math.log2(d_qwen / d_gaussian)
    standard_error = statistics.stdev(fold_s) / math.sqrt(len(fold_s))
    upper = s_match + 2.0 * standard_error
    exact_float(evaluation["D_Qwen"], d_qwen, f"{where}.D_Qwen")
    exact_float(evaluation["D_Gaussian"], d_gaussian, f"{where}.D_Gaussian")
    exact_float(evaluation["s_match"], s_match, f"{where}.s_match")
    require(evaluation["fold_s_match"] == fold_s, "fold_s_match is not exact", where=where)
    exact_float(evaluation["whole_expert_standard_error"], standard_error, f"{where}.whole_expert_standard_error")
    exact_float(evaluation["upper_s_match_2se"], upper, f"{where}.upper_s_match_2se")
    require(
        evaluation["all_whole_expert_folds_positive"] is all(value > 0.0 for value in fold_s),
        "fold positivity decision mismatch",
        where=where,
    )

    gap = evaluation["Gaussian_operational_gap"]
    require_keys(gap, ("code_rate_bpw", "F_gaussian", "s_gaussian", "distortion_ratio_to_ideal_gaussian"), f"{where}.Gaussian_operational_gap")
    code_rate = 2 * BITS_PER_STAGE / CHUNK_DIMENSION
    gaussian_f = d_gaussian * 2.0 ** (2.0 * code_rate)
    exact_float(gap["code_rate_bpw"], code_rate, f"{where}.Gaussian_operational_gap.code_rate_bpw")
    exact_float(gap["F_gaussian"], gaussian_f, f"{where}.Gaussian_operational_gap.F_gaussian")
    exact_float(gap["s_gaussian"], -0.5 * math.log2(gaussian_f), f"{where}.Gaussian_operational_gap.s_gaussian")
    exact_float(gap["distortion_ratio_to_ideal_gaussian"], gaussian_f, f"{where}.Gaussian_operational_gap.distortion_ratio_to_ideal_gaussian")

    absolute = evaluation["absolute"]
    require_keys(absolute, ("production_128", "self_contained_panel_6"), f"{where}.absolute")
    for name, experts_count in (("production_128", 128), ("self_contained_panel_6", 6)):
        row = absolute[name]
        require_keys(row, ("physical_R", "F", "s_absolute", "target_D", "passes_F_0p8"), f"{where}.absolute.{name}")
        physical_r = float(ledger(experts_count)["physical_bpw"])
        f_value = d_qwen * 2.0 ** (2.0 * physical_r)
        target_d = float(ledger(experts_count)["target_relative_mse"])
        exact_float(row["physical_R"], physical_r, f"{where}.absolute.{name}.physical_R")
        exact_float(row["F"], f_value, f"{where}.absolute.{name}.F")
        exact_float(row["s_absolute"], -0.5 * math.log2(f_value), f"{where}.absolute.{name}.s_absolute")
        exact_float(row["target_D"], target_d, f"{where}.absolute.{name}.target_D")
        require(row["passes_F_0p8"] is (f_value <= 0.8), "absolute gate mismatch", where=where, ledger=name)
    return {"D_Qwen": d_qwen, "D_Gaussian": d_gaussian, "s_match": s_match, "upper_s_match_2se": upper}


def enforce_history_and_decision(result: Mapping[str, Any]) -> dict[str, Any]:
    training = result["training"]
    require_keys(training, ("stopped_update", "max_updates", "history", "early_kill"), "training")
    require(training["stopped_update"] == 512, "run did not stop at frozen early-kill update")
    require(training["max_updates"] == 2048, "wrong maximum update budget")
    history = training["history"]
    require(isinstance(history, list) and len(history) == 2, "history must contain exactly updates 256 and 512")
    summaries: dict[int, dict[str, float]] = {}
    for index, expected_update in enumerate((256, 512)):
        entry = history[index]
        require_keys(entry, ("update", "temperature", "evaluation"), f"training.history[{index}]")
        require(entry["update"] == expected_update, "unexpected history update", index=index)
        fraction = (expected_update - 1) / (2048 - 1)
        expected_temperature = 1.0 * (0.25 / 1.0) ** fraction
        exact_float(entry["temperature"], expected_temperature, f"training.history[{index}].temperature")
        summaries[expected_update] = enforce_evaluation(entry["evaluation"], f"training.history[{index}].evaluation")

    require(result["final_evaluation"] == history[-1]["evaluation"], "final_evaluation is not the exact final history object")
    enforce_evaluation(result["final_evaluation"], "final_evaluation")
    half_upper = summaries[256]["upper_s_match_2se"]
    quarter_upper = summaries[512]["upper_s_match_2se"]
    improvement = quarter_upper - half_upper
    projected = quarter_upper + 6.0 * max(0.0, improvement)
    expected_early = {
        "half_checkpoint_upper_s_match_2se": half_upper,
        "quarter_checkpoint_upper_s_match_2se": quarter_upper,
        "late_improvement": improvement,
        "constant_recent_slope_projection_to_full_budget": projected,
        "boundary_projection_max": 0.14,
        "kill_if_upper_below": 0.08,
        "and_improvement_below": 0.01,
        "kill": quarter_upper < 0.08 and improvement < 0.01,
        "interpretation": "preregistered empirical trend kill; not a mathematical converse",
    }
    require(training["early_kill"] == expected_early, "early-kill record is not exact", actual=training["early_kill"], expected=expected_early)
    require(expected_early["kill"] is True, "frozen early-kill condition did not fire")
    require(result["decision"] == "HARD_KILL_D16_SHALLOW_BEFORE_PINNED", "top-level decision mismatch")
    require(result["strict_ptq"] is True, "strict_ptq flag is not true")
    require(result["pinned_panel"]["opened"] is False, "pinned panel was marked opened")
    require(result["pinned_panel"]["path_argument_supported"] is False, "pinned-panel path was supported")
    return {
        "history_updates_exact": [256, 512],
        "temperatures_exact": [history[0]["temperature"], history[1]["temperature"]],
        "final_equals_history_512_exactly": True,
        "early_kill_exact": expected_early,
        "decision": result["decision"],
    }


def load_bf16_canonical(path: Path, role: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    require(raw.size == VALUES_PER_MATRIX, "wrong BF16 source size", path=str(path), values=int(raw.size))
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if role == "down":
        return np.ascontiguousarray(values.reshape(COLS, ROWS).T)
    require(role == "up", "unknown role", role=role)
    return np.ascontiguousarray(values.reshape(ROWS, COLS))


def blocked_energy(values: np.ndarray, block_values: int = 1 << 18) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    total = 0.0
    for start in range(0, flat.size, block_values):
        block64 = flat[start : start + block_values].astype(np.float64)
        total += float(np.sum(block64 * block64, dtype=np.float64))
    return total


def normalize_matrix(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    exact_mean = float(np.mean(flat, dtype=np.float64))
    centered64 = flat.astype(np.float64) - exact_mean
    exact_rms = math.sqrt(float(np.mean(centered64 * centered64, dtype=np.float64)))
    stored_mean = float(np.float16(exact_mean))
    stored_rms = float(np.float16(exact_rms))
    require(math.isfinite(stored_rms) and stored_rms > 0.0, "nonpositive stored RMS")
    normalized = ((flat - np.float32(stored_mean)) / np.float32(stored_rms)).astype(np.float32)
    return np.ascontiguousarray(normalized.reshape(-1, CHUNK_DIMENSION)), {
        "exact_mean": exact_mean,
        "exact_centered_rms": exact_rms,
        "stored_fp16_mean": stored_mean,
        "stored_fp16_centered_rms": stored_rms,
        "source_energy": blocked_energy(flat),
    }


def gaussian_seed(filename: str) -> int:
    digest = hashlib.sha256(f"BISCO-GAUSSIAN-v1|{GAUSSIAN_SEED_BASE}|{filename}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def matched_gaussian(count: int, mean: float, rms: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    standard = generator.standard_normal(count, dtype=np.float32)
    return (standard * np.float32(rms) + np.float32(mean)).astype(np.float32)


def discover_and_hash_sources(aux_dir: Path, expected_hashes: Mapping[str, str]) -> dict[str, Path]:
    resolved = aux_dir.resolve()
    require("blind_protocol" not in {part.lower() for part in resolved.parts}, "forbidden target-protocol path")
    expected_names = {f"l15e{expert}_{role}.bf16.bin" for role in ROLES for expert in EXPERTS}
    require(set(expected_hashes) == expected_names, "result source hash keyset is not frozen")
    actual_paths = sorted(resolved.glob("*.bf16.bin"))
    actual_names = {path.name for path in actual_paths}
    require(actual_names == expected_names, "auxiliary BF16 file set mismatch", missing=sorted(expected_names - actual_names), unexpected=sorted(actual_names - expected_names))
    result: dict[str, Path] = {}
    for path in actual_paths:
        require(path.stat().st_size == VALUES_PER_MATRIX * 2, "wrong auxiliary source bytes", path=path.name)
        actual_hash = sha256_file(path)
        require(actual_hash == expected_hashes[path.name], "auxiliary source hash mismatch", path=path.name)
        result[path.name] = path
    return result


def training_moments(paths: Mapping[str, Path], role: str) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    count = 0
    for expert in TRAIN_EXPERTS:
        name = f"l15e{expert}_{role}.bf16.bin"
        flat = load_bf16_canonical(paths[name], role).reshape(-1)
        values64 = flat.astype(np.float64)
        total += float(np.sum(values64, dtype=np.float64))
        total_sq += float(np.sum(values64 * values64, dtype=np.float64))
        count += flat.size
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    return mean, math.sqrt(variance)


def close(actual: float, expected: float, *, rtol: float, atol: float, where: str) -> dict[str, float]:
    absolute = abs(actual - expected)
    denominator = max(abs(expected), np.finfo(np.float64).tiny)
    relative = absolute / denominator
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), "numeric replay mismatch", where=where, actual=actual, expected=expected, absolute_error=absolute, relative_error=relative, rtol=rtol, atol=atol)
    return {"absolute_error": absolute, "relative_error": relative, "rtol": rtol, "atol": atol}


def normalization_records(result: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    records = result["data_firewall"]["normalization"]
    require(isinstance(records, list) and len(records) == len(ROLES) * len(EXPERTS), "wrong normalization record count")
    expected_identities = [(role, expert) for role in ROLES for expert in EXPERTS]
    actual_identities = [(str(row["role"]), int(row["expert"])) for row in records]
    require(actual_identities == expected_identities, "normalization records have wrong identities/order")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in records:
        require_keys(row, ("expert", "role", "split", "source", "gaussian"), "normalization record")
        expected_split = "validation" if int(row["expert"]) in VALIDATION_EXPERTS else "training"
        require(row["split"] == expected_split, "normalization split mismatch")
        for domain in DOMAINS:
            key = "source" if domain == "qwen" else "gaussian"
            require_keys(row[key], ("exact_mean", "exact_centered_rms", "stored_fp16_mean", "stored_fp16_centered_rms", "source_energy"), f"normalization.{key}")
        indexed[(str(row["role"]), int(row["expert"]))] = row
    return indexed


def compare_moments(actual: Mapping[str, float], expected: Mapping[str, Any], where: str) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for field in ("exact_mean", "exact_centered_rms", "source_energy"):
        comparisons[field] = close(
            float(actual[field]),
            float(expected[field]),
            rtol=ENERGY_RTOL,
            atol=ENERGY_ATOL,
            where=f"{where}.{field}",
        )
    for field in ("stored_fp16_mean", "stored_fp16_centered_rms"):
        require(float(actual[field]) == float(expected[field]), "stored FP16 normalization field mismatch", where=f"{where}.{field}")
        comparisons[field] = {"exact": True, "value": float(actual[field])}
    return comparisons


def backend(name: str) -> Any:
    if name == "numpy":
        return np
    require(name == "cupy", "unknown backend", backend=name)
    try:
        import cupy as cp  # type: ignore
    except ImportError as error:
        raise AuditFailure("CuPy is required for production replay") from error
    require(int(cp.cuda.runtime.getDeviceCount()) >= 1, "no CUDA device is visible")
    return cp


def host_array(value: Any, xp: Any) -> np.ndarray:
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)


def silu(value: Any, xp: Any) -> Any:
    clipped = xp.clip(value, -20.0, 20.0)
    sigmoid = 1.0 / (1.0 + xp.exp(-clipped))
    return value * sigmoid


def decode(params: Mapping[str, Any], prefix: str, code: Any, xp: Any) -> Any:
    hidden = silu(code @ params[f"{prefix}_dw1"] + params[f"{prefix}_db1"], xp)
    return hidden @ params[f"{prefix}_dw2"] + params[f"{prefix}_db2"]


def initial_stage(params: Mapping[str, Any], prefix: str, source: Any, xp: Any) -> tuple[Any, Any]:
    hidden = silu(source @ params[f"{prefix}_ew1"] + params[f"{prefix}_eb1"], xp)
    logits = hidden @ params[f"{prefix}_ew2"] + params[f"{prefix}_eb2"]
    code = xp.where(logits >= 0.0, 1.0, -1.0).astype(xp.float32) / math.sqrt(BITS_PER_STAGE)
    return code, decode(params, prefix, code, xp)


def independent_reconstruct(params: Mapping[str, Any], source: Any, xp: Any) -> tuple[Any, Any, Any]:
    """Independent update-512 inference and fixed-order greedy code search."""

    q1, y1 = initial_stage(params, "s1", source, xp)
    q2, y2 = initial_stage(params, "s2", source - y1, xp)
    best = y1 + y2
    best_error = xp.sum((best - source) ** 2, axis=1)
    for _ in range(BITFLIP_SWEEPS):
        for prefix, code in (("s1", q1), ("s2", q2)):
            for bit in range(BITS_PER_STAGE):
                code[:, bit] *= -1.0
                candidate_stage = decode(params, prefix, code, xp)
                candidate = candidate_stage + (y2 if prefix == "s1" else y1)
                candidate_error = xp.sum((candidate - source) ** 2, axis=1)
                accept = candidate_error < best_error
                code[:, bit] = xp.where(accept, code[:, bit], -code[:, bit])
                if prefix == "s1":
                    y1 = xp.where(accept[:, None], candidate_stage, y1)
                else:
                    y2 = xp.where(accept[:, None], candidate_stage, y2)
                best = xp.where(accept[:, None], candidate, best)
                best_error = xp.where(accept, candidate_error, best_error)
    return best, q1, q2


def device_model(state: Mapping[str, np.ndarray], xp: Any) -> dict[str, Any]:
    model: dict[str, Any] = {}
    for name, _ in STATE_PARAMETER_SHAPES:
        array = state[name]
        if name in DECODER_NAMES:
            array = array.astype(np.float16).astype(np.float32)
        model[name] = xp.asarray(array, dtype=xp.float32)
    return model


def replay_matrix(params: Mapping[str, Any], chunks: np.ndarray, stored_scale: float, xp: Any) -> dict[str, Any]:
    fp64_sse = 0.0
    fp32_emulation_sse = 0.0
    code_digest = hashlib.sha256()
    reconstruction_digest = hashlib.sha256()
    chunks_count = int(chunks.shape[0])
    for start in range(0, chunks_count, EVALUATION_BATCH_CHUNKS):
        host_batch = np.ascontiguousarray(chunks[start : start + EVALUATION_BATCH_CHUNKS], dtype=np.float32)
        batch = xp.asarray(host_batch, dtype=xp.float32)
        reconstruction, q1, q2 = independent_reconstruct(params, batch, xp)
        error32 = reconstruction - batch
        fp32_emulation_sse += float(host_array(xp.sum(error32 * error32), xp)) * (stored_scale ** 2)
        scaled64 = error32.astype(xp.float64) * float(stored_scale)
        fp64_sse += float(host_array(xp.sum(scaled64 * scaled64, dtype=xp.float64), xp))
        q1_host = host_array(q1 > 0.0, xp)
        q2_host = host_array(q2 > 0.0, xp)
        bits = np.concatenate((q1_host, q2_host), axis=1)
        code_digest.update(np.packbits(bits, axis=None, bitorder="little").tobytes())
        reconstruction_host = np.ascontiguousarray(host_array(reconstruction, xp), dtype="<f4")
        reconstruction_digest.update(reconstruction_host.tobytes(order="C"))
    return {
        "chunks": chunks_count,
        "code_bits": chunks_count * 2 * BITS_PER_STAGE,
        "code_sha256_little_bitorder_stage1_then_stage2": code_digest.hexdigest(),
        "normalized_reconstruction_fp32_sha256": reconstruction_digest.hexdigest(),
        "source_domain_sse_fp64": fp64_sse,
        "source_domain_sse_published_fp32_emulation": fp32_emulation_sse,
    }


def aggregate_replay(per_matrix: list[dict[str, Any]]) -> dict[str, Any]:
    per_expert: list[dict[str, Any]] = []
    fold_s: list[float] = []
    for expert in VALIDATION_EXPERTS:
        rows = [row for row in per_matrix if int(row["expert"]) == expert]
        require(len(rows) == 2, "replay matrix rows missing for expert", expert=expert)
        q_sse = sum(float(row["qwen"]["source_domain_sse_fp64"]) for row in rows)
        q_energy = sum(float(row["qwen"]["source_energy_fp64"]) for row in rows)
        g_sse = sum(float(row["gaussian"]["source_domain_sse_fp64"]) for row in rows)
        g_energy = sum(float(row["gaussian"]["source_energy_fp64"]) for row in rows)
        d_qwen = q_sse / q_energy
        d_gaussian = g_sse / g_energy
        s_match = -0.5 * math.log2(d_qwen / d_gaussian)
        fold_s.append(s_match)
        per_expert.append(
            {
                "expert": expert,
                "qwen_sse_fp64": q_sse,
                "qwen_energy_fp64": q_energy,
                "gaussian_sse_fp64": g_sse,
                "gaussian_energy_fp64": g_energy,
                "D_Qwen_fp64": d_qwen,
                "D_Gaussian_fp64": d_gaussian,
                "s_match_fp64": s_match,
            }
        )
    q_sse = sum(float(row["qwen_sse_fp64"]) for row in per_expert)
    q_energy = sum(float(row["qwen_energy_fp64"]) for row in per_expert)
    g_sse = sum(float(row["gaussian_sse_fp64"]) for row in per_expert)
    g_energy = sum(float(row["gaussian_energy_fp64"]) for row in per_expert)
    d_qwen = q_sse / q_energy
    d_gaussian = g_sse / g_energy
    s_match = -0.5 * math.log2(d_qwen / d_gaussian)
    standard_error = statistics.stdev(fold_s) / math.sqrt(len(fold_s))
    return {
        "per_expert": per_expert,
        "pooled": {
            "qwen_sse_fp64": q_sse,
            "qwen_energy_fp64": q_energy,
            "gaussian_sse_fp64": g_sse,
            "gaussian_energy_fp64": g_energy,
            "D_Qwen_fp64": d_qwen,
            "D_Gaussian_fp64": d_gaussian,
            "s_match_fp64": s_match,
            "fold_s_match_fp64": fold_s,
            "whole_expert_standard_error_fp64": standard_error,
            "upper_s_match_2se_fp64": s_match + 2.0 * standard_error,
        },
    }


def replay(
    result_path: Path,
    aux_dir: Path,
    output_path: Path,
    backend_name: str = "cupy",
) -> dict[str, Any]:
    started = time.time()
    result_path = result_path.resolve()
    result_dir = result_path.parent
    require(result_path.is_file(), "result file is missing", path=str(result_path))
    result_hash = sha256_file(result_path)
    require(result_hash == EXPECTED_RESULT_SHA256, "result is not frozen run_1", actual=result_hash, expected=EXPECTED_RESULT_SHA256)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result["protocol"] == "bisco_raw_mse_aux_d16_launch_v1", "wrong result protocol")

    launch_path = Path(__file__).resolve().parent / "launch_protocol.json"
    launch_hash = sha256_file(launch_path)
    require(launch_hash == EXPECTED_LAUNCH_SHA256, "launch protocol binding mismatch", actual=launch_hash)
    history_evidence = enforce_history_and_decision(result)
    models, artifact_evidence = parse_and_bind_models(result, result_dir)

    source_hashes = result["data_firewall"]["source_sha256"]
    paths = discover_and_hash_sources(aux_dir, source_hashes)
    records = normalization_records(result)
    computed_training_moments: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        mean, rms = training_moments(paths, role)
        published = result["data_firewall"]["gaussian_role_training_moments"][role]
        require(np.float32(mean).tobytes() == np.float32(published["mean"]).tobytes(), "training mean changes after FP32 rounding", role=role)
        require(np.float32(rms).tobytes() == np.float32(published["centered_rms"]).tobytes(), "training RMS changes after FP32 rounding", role=role)
        mean_comparison = close(mean, float(published["mean"]), rtol=ENERGY_RTOL, atol=ENERGY_ATOL, where=f"training_moments.{role}.mean")
        rms_comparison = close(rms, float(published["centered_rms"]), rtol=ENERGY_RTOL, atol=ENERGY_ATOL, where=f"training_moments.{role}.centered_rms")
        computed_training_moments[role] = {
            "independent_mean_fp64": mean,
            "independent_centered_rms_fp64": rms,
            "published_mean": float(published["mean"]),
            "published_centered_rms": float(published["centered_rms"]),
            "same_fp32_mean": True,
            "same_fp32_centered_rms": True,
            "mean_comparison": mean_comparison,
            "centered_rms_comparison": rms_comparison,
        }

    xp = backend(backend_name)
    require(backend_name == "cupy", "sealed production replay requires CuPy")
    require(result["backend"]["name"] == "cupy", "original run was not CuPy")
    device_models = {
        domain: {role: device_model(models[domain][role], xp) for role in ROLES}
        for domain in DOMAINS
    }
    per_matrix: list[dict[str, Any]] = []
    max_fp64_relative_error = 0.0
    max_fp32_emulation_relative_error = 0.0
    max_energy_relative_error = 0.0
    published_rows = {
        (str(row["role"]), int(row["expert"])): row
        for row in result["final_evaluation"]["per_matrix"]
    }
    for role in ROLES:
        mean = computed_training_moments[role]["independent_mean_fp64"]
        rms = computed_training_moments[role]["independent_centered_rms_fp64"]
        for expert in VALIDATION_EXPERTS:
            filename = f"l15e{expert}_{role}.bf16.bin"
            source = load_bf16_canonical(paths[filename], role).reshape(-1)
            qwen_chunks, qwen_moments = normalize_matrix(source)
            seed = gaussian_seed(filename)
            gaussian_source = matched_gaussian(VALUES_PER_MATRIX, float(mean), float(rms), seed)
            gaussian_chunks, gaussian_moments = normalize_matrix(gaussian_source)
            published_normalization = records[(role, expert)]
            qwen_moment_comparison = compare_moments(qwen_moments, published_normalization["source"], f"{filename}.source")
            gaussian_moment_comparison = compare_moments(gaussian_moments, published_normalization["gaussian"], f"{filename}.gaussian")

            qwen_replay = replay_matrix(
                device_models["qwen"][role], qwen_chunks, qwen_moments["stored_fp16_centered_rms"], xp
            )
            gaussian_replay = replay_matrix(
                device_models["gaussian"][role], gaussian_chunks, gaussian_moments["stored_fp16_centered_rms"], xp
            )
            published = published_rows[(role, expert)]
            for domain, values, moments, sse_field, energy_field in (
                ("qwen", qwen_replay, qwen_moments, "qwen_sse", "qwen_energy"),
                ("gaussian", gaussian_replay, gaussian_moments, "gaussian_sse", "gaussian_energy"),
            ):
                values["source_energy_fp64"] = float(moments["source_energy"])
                values["published_source_domain_sse"] = float(published[sse_field])
                values["published_source_energy"] = float(published[energy_field])
                values["fp64_sse_comparison"] = close(
                    float(values["source_domain_sse_fp64"]),
                    float(published[sse_field]),
                    rtol=FP64_VS_PUBLISHED_RTOL,
                    atol=FP64_VS_PUBLISHED_ATOL,
                    where=f"{filename}.{domain}.source_domain_sse_fp64",
                )
                values["fp32_emulation_sse_comparison"] = close(
                    float(values["source_domain_sse_published_fp32_emulation"]),
                    float(published[sse_field]),
                    rtol=FP32_EMULATION_RTOL,
                    atol=FP32_EMULATION_ATOL,
                    where=f"{filename}.{domain}.source_domain_sse_published_fp32_emulation",
                )
                values["energy_comparison"] = close(
                    float(moments["source_energy"]),
                    float(published[energy_field]),
                    rtol=ENERGY_RTOL,
                    atol=ENERGY_ATOL,
                    where=f"{filename}.{domain}.source_energy",
                )
                max_fp64_relative_error = max(max_fp64_relative_error, values["fp64_sse_comparison"]["relative_error"])
                max_fp32_emulation_relative_error = max(max_fp32_emulation_relative_error, values["fp32_emulation_sse_comparison"]["relative_error"])
                max_energy_relative_error = max(max_energy_relative_error, values["energy_comparison"]["relative_error"])
            per_matrix.append(
                {
                    "expert": expert,
                    "role": role,
                    "source_file": filename,
                    "source_sha256": source_hashes[filename],
                    "gaussian_seed_uint64": seed,
                    "qwen_normalization_comparison": qwen_moment_comparison,
                    "gaussian_normalization_comparison": gaussian_moment_comparison,
                    "qwen": qwen_replay,
                    "gaussian": gaussian_replay,
                }
            )
            del source, qwen_chunks, gaussian_source, gaussian_chunks

    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    aggregate = aggregate_replay(per_matrix)
    pooled = aggregate["pooled"]
    final_published = result["final_evaluation"]
    pooled_comparisons = {
        "D_Qwen": close(
            float(pooled["D_Qwen_fp64"]),
            float(final_published["D_Qwen"]),
            rtol=FP64_VS_PUBLISHED_RTOL,
            atol=FP64_VS_PUBLISHED_ATOL,
            where="pooled.D_Qwen",
        ),
        "D_Gaussian": close(
            float(pooled["D_Gaussian_fp64"]),
            float(final_published["D_Gaussian"]),
            rtol=FP64_VS_PUBLISHED_RTOL,
            atol=FP64_VS_PUBLISHED_ATOL,
            where="pooled.D_Gaussian",
        ),
        "s_match": close(
            float(pooled["s_match_fp64"]),
            float(final_published["s_match"]),
            rtol=FP64_VS_PUBLISHED_RTOL,
            atol=FP64_VS_PUBLISHED_ATOL,
            where="pooled.s_match",
        ),
    }
    source_root = hashlib.sha256()
    for filename in sorted(source_hashes):
        source_root.update(filename.encode("ascii") + b"\0" + source_hashes[filename].encode("ascii") + b"\n")
    backend_evidence = {
        "name": backend_name,
        "numpy_version": np.__version__,
        "cupy_version": xp.__version__,
        "device_id": int(xp.cuda.Device().id),
        "device_name": xp.cuda.runtime.getDeviceProperties(int(xp.cuda.Device().id))["name"].decode("utf-8"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }
    receipt_unsigned = {
        "protocol": "bisco_raw_mse_independent_state_replay_v1",
        "verified": True,
        "claim": "Independent replay verifies the frozen update-512 negative auxiliary result; it does not test or reject arbitrary nonlinear codecs.",
        "pinned_panel_opened": False,
        "inputs": {
            "result_file": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": result_hash,
            "launch_protocol_sha256": launch_hash,
            "independent_replay_script_sha256": sha256_file(Path(__file__).resolve()),
            "artifacts": artifact_evidence,
            "auxiliary_source_files": len(source_hashes),
            "auxiliary_source_hash_root_sha256": source_root.hexdigest(),
        },
        "structural_audit": {
            "independent_state_schema_values": STATE_VALUES,
            "independent_state_schema_bytes": EXPECTED_STATE_BYTES,
            "independent_decoder_schema_values": DECODER_VALUES,
            "independent_decoder_schema_bytes": EXPECTED_DECODER_BYTES,
            "history_and_decision": history_evidence,
        },
        "data_reconstruction": {
            "source_files_rehashed": len(source_hashes),
            "source_file_rule_exact": True,
            "training_experts_per_role": list(TRAIN_EXPERTS),
            "validation_experts": list(VALIDATION_EXPERTS),
            "roles": list(ROLES),
            "matched_gaussian_seed_base": GAUSSIAN_SEED_BASE,
            "training_role_moments": computed_training_moments,
            "validation_qwen_and_gaussian_matrices_reconstructed": len(per_matrix) * 2,
        },
        "evaluator": {
            "implementation": "independent_replay.py; no import or call into bisco_raw_mse_oracle.evaluate_models",
            "state_update": 512,
            "decoder_parameters": "FP32 state fields rounded through IEEE binary16 then evaluated as FP32",
            "matmul_and_code_search_dtype": "FP32",
            "sse_accumulation_dtype": "FP64 after multiplying normalized FP32 residual by exact stored-FP16 RMS",
            "bitflip_sweeps": BITFLIP_SWEEPS,
            "batch_chunks": EVALUATION_BATCH_CHUNKS,
            "backend": backend_evidence,
        },
        "tolerances": {
            "fp64_sse_vs_published": {
                "rtol": FP64_VS_PUBLISHED_RTOL,
                "atol": FP64_VS_PUBLISHED_ATOL,
                "derivation": "gamma_128=128*2^-24/(1-128*2^-24), conservatively covering FP32 square plus hierarchical FP32 reduction; replay accumulation is FP64",
            },
            "same_backend_fp32_emulation_vs_published": {
                "rtol": FP32_EMULATION_RTOL,
                "atol": FP32_EMULATION_ATOL,
                "derivation": "same frozen CuPy/device arithmetic but separately written evaluator; allows reduction scheduling jitter",
            },
            "independent_fp64_energy_vs_published": {
                "rtol": ENERGY_RTOL,
                "atol": ENERGY_ATOL,
                "derivation": "64 binary64 ulps plus 1e-12 absolute for independently blocked source-energy and training-moment reductions",
            },
        },
        "replay": {
            "per_matrix": per_matrix,
            **aggregate,
            "published_final": {
                "D_Qwen": final_published["D_Qwen"],
                "D_Gaussian": final_published["D_Gaussian"],
                "s_match": final_published["s_match"],
                "decision": result["decision"],
            },
            "pooled_comparisons": pooled_comparisons,
            "observed_max_relative_error": {
                "fp64_sse_vs_published": max_fp64_relative_error,
                "same_backend_fp32_emulation_vs_published": max_fp32_emulation_relative_error,
                "fp64_energy_vs_published": max_energy_relative_error,
            },
        },
        "verdict": {
            "status": "PASS",
            "state_backed_update_512_replay": True,
            "fp16_decoders_linked_to_fp32_states": True,
            "history_and_hard_kill_exact": True,
            "decision": result["decision"],
        },
        "runtime_seconds": time.time() - started,
    }
    sealed = seal_receipt(receipt_unsigned)
    verify_receipt_seal(sealed)
    output_path = output_path.resolve()
    require(output_path != result_path, "receipt would overwrite result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sealed, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return sealed


def verify_receipt_file(receipt_path: Path) -> dict[str, Any]:
    """Verify the seal, local input bindings, and receipt-internal replay math.

    This is intentionally GPU-free: it authenticates the already sealed replay
    against the frozen local state/result package.  A fresh source/model replay
    still requires :func:`replay` and the 32 auxiliary BF16 files.
    """

    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    seal = verify_receipt_seal(receipt)
    require(receipt["protocol"] == "bisco_raw_mse_independent_state_replay_v1", "wrong receipt protocol")
    require(receipt["verified"] is True, "receipt is not verified")
    require(receipt["verdict"]["status"] == "PASS", "receipt verdict is not PASS")
    require(receipt["pinned_panel_opened"] is False, "receipt claims the pinned panel was opened")
    inputs = receipt["inputs"]
    require_keys(
        inputs,
        (
            "result_file",
            "result_bytes",
            "result_sha256",
            "launch_protocol_sha256",
            "independent_replay_script_sha256",
            "artifacts",
            "auxiliary_source_files",
            "auxiliary_source_hash_root_sha256",
        ),
        "receipt.inputs",
    )
    script_hash = sha256_file(Path(__file__).resolve())
    require(script_hash == inputs["independent_replay_script_sha256"], "receipt is bound to a different replay script", actual=script_hash)
    launch_hash = sha256_file(Path(__file__).resolve().parent / "launch_protocol.json")
    require(launch_hash == inputs["launch_protocol_sha256"] == EXPECTED_LAUNCH_SHA256, "receipt launch binding mismatch")
    result_path = artifact_path(receipt_path.parent, str(inputs["result_file"]))
    require(result_path.stat().st_size == int(inputs["result_bytes"]), "receipt result byte count mismatch")
    result_hash = sha256_file(result_path)
    require(result_hash == inputs["result_sha256"] == EXPECTED_RESULT_SHA256, "receipt result hash mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = enforce_history_and_decision(result)
    _, artifacts = parse_and_bind_models(result, receipt_path.parent)
    require(artifacts == inputs["artifacts"], "receipt artifact evidence differs from local artifacts")
    require(
        history == receipt["structural_audit"]["history_and_decision"],
        "receipt history evidence differs from frozen result",
    )

    per_matrix = receipt["replay"]["per_matrix"]
    identities = [(int(row["expert"]), str(row["role"])) for row in per_matrix]
    expected_identities = [(expert, role) for role in ROLES for expert in VALIDATION_EXPERTS]
    require(identities == expected_identities, "receipt replay matrix identities/order mismatch")
    max_fp64 = 0.0
    max_fp32 = 0.0
    max_energy = 0.0
    published_rows = {
        (int(row["expert"]), str(row["role"])): row
        for row in result["final_evaluation"]["per_matrix"]
    }
    for row in per_matrix:
        published = published_rows[(int(row["expert"]), str(row["role"]))]
        for domain, sse_field, energy_field in (
            ("qwen", "qwen_sse", "qwen_energy"),
            ("gaussian", "gaussian_sse", "gaussian_energy"),
        ):
            values = row[domain]
            require(float(values["published_source_domain_sse"]) == float(published[sse_field]), "receipt published SSE unlink", domain=domain)
            require(float(values["published_source_energy"]) == float(published[energy_field]), "receipt published energy unlink", domain=domain)
            fp64_comparison = close(
                float(values["source_domain_sse_fp64"]),
                float(published[sse_field]),
                rtol=FP64_VS_PUBLISHED_RTOL,
                atol=FP64_VS_PUBLISHED_ATOL,
                where=f"sealed.{row['source_file']}.{domain}.fp64_sse",
            )
            fp32_comparison = close(
                float(values["source_domain_sse_published_fp32_emulation"]),
                float(published[sse_field]),
                rtol=FP32_EMULATION_RTOL,
                atol=FP32_EMULATION_ATOL,
                where=f"sealed.{row['source_file']}.{domain}.fp32_sse",
            )
            energy_comparison = close(
                float(values["source_energy_fp64"]),
                float(published[energy_field]),
                rtol=ENERGY_RTOL,
                atol=ENERGY_ATOL,
                where=f"sealed.{row['source_file']}.{domain}.energy",
            )
            require(values["fp64_sse_comparison"] == fp64_comparison, "stored FP64 comparison record mismatch")
            require(values["fp32_emulation_sse_comparison"] == fp32_comparison, "stored FP32 comparison record mismatch")
            require(values["energy_comparison"] == energy_comparison, "stored energy comparison record mismatch")
            max_fp64 = max(max_fp64, fp64_comparison["relative_error"])
            max_fp32 = max(max_fp32, fp32_comparison["relative_error"])
            max_energy = max(max_energy, energy_comparison["relative_error"])
    aggregate = aggregate_replay(per_matrix)
    require(aggregate["per_expert"] == receipt["replay"]["per_expert"], "sealed per-expert replay aggregate mismatch")
    require(aggregate["pooled"] == receipt["replay"]["pooled"], "sealed pooled replay aggregate mismatch")
    expected_max = {
        "fp64_sse_vs_published": max_fp64,
        "same_backend_fp32_emulation_vs_published": max_fp32,
        "fp64_energy_vs_published": max_energy,
    }
    require(receipt["replay"]["observed_max_relative_error"] == expected_max, "sealed maximum-error record mismatch")
    final = result["final_evaluation"]
    require(
        receipt["replay"]["published_final"]
        == {
            "D_Qwen": final["D_Qwen"],
            "D_Gaussian": final["D_Gaussian"],
            "s_match": final["s_match"],
            "decision": result["decision"],
        },
        "sealed published-final binding mismatch",
    )
    pooled = aggregate["pooled"]
    for key, replay_key in (
        ("D_Qwen", "D_Qwen_fp64"),
        ("D_Gaussian", "D_Gaussian_fp64"),
        ("s_match", "s_match_fp64"),
    ):
        recomputed = close(
            float(pooled[replay_key]),
            float(final[key]),
            rtol=FP64_VS_PUBLISHED_RTOL,
            atol=FP64_VS_PUBLISHED_ATOL,
            where=f"sealed.pooled.{key}",
        )
        require(receipt["replay"]["pooled_comparisons"][key] == recomputed, "sealed pooled comparison mismatch", field=key)
    return {
        "verified": True,
        "receipt": str(receipt_path),
        "canonical_unsigned_sha256": seal,
        "result_sha256": result_hash,
        "script_sha256": script_hash,
        "decision": result["decision"],
        "D_Qwen_fp64": pooled["D_Qwen_fp64"],
        "D_Gaussian_fp64": pooled["D_Gaussian_fp64"],
        "s_match_fp64": pooled["s_match_fp64"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--aux-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument("--verify-receipt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_receipt is not None:
        require(args.result is None and args.aux_dir is None and args.output is None, "receipt verification cannot be mixed with replay")
        print(json.dumps(verify_receipt_file(args.verify_receipt), indent=2, allow_nan=False))
        return
    require(args.result is not None and args.aux_dir is not None and args.output is not None, "--result, --aux-dir, and --output are required for replay")
    receipt = replay(args.result, args.aux_dir, args.output, args.backend)
    print(
        json.dumps(
            {
                "verified": receipt["verified"],
                "decision": receipt["verdict"]["decision"],
                "D_Qwen_fp64": receipt["replay"]["pooled"]["D_Qwen_fp64"],
                "D_Gaussian_fp64": receipt["replay"]["pooled"]["D_Gaussian_fp64"],
                "s_match_fp64": receipt["replay"]["pooled"]["s_match_fp64"],
                "observed_max_relative_error": receipt["replay"]["observed_max_relative_error"],
                "receipt": str(args.output),
                "receipt_seal": receipt["receipt_seal"]["sha256"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
