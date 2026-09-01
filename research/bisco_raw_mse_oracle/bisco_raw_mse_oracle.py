#!/usr/bin/env python3
"""Preregistered auxiliary-only raw-MSE gate for shallow two-stage BiSCo.

The production path uses CuPy and never accepts a target-panel path.  NumPy is
available only so that the exact same mathematical kernels can be exercised by
small CPU unit tests.  Model selection is restricted to the frozen layer-15
auxiliary experts; four whole experts remain untouched until evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROWS = 768
COLS = 2048
N = ROWS * COLS
D = 16
HIDDEN = 64
BITS = 18
STAGES = 2
ROLES = ("up", "down")
DEPLOYMENT_ROLES = ("gate", "up", "down")
EXPERTS = tuple(range(0, 121, 8))
VALIDATION_EXPERTS = (24, 56, 88, 120)
TRAIN_EXPERTS = tuple(expert for expert in EXPERTS if expert not in VALIDATION_EXPERTS)
WEIGHTS_PER_EXPERT = 4_718_592
HEADER_BYTES = 256
LOCAL_SCALE_BYTES = 12
TARGET_S = -0.5 * math.log2(0.8)
PARENT_PROTOCOL_SHA256 = "28c2bd6656f31ce7315601d0048d0b43759a7f2859142f745465e8fa0fe83164"
ASSESSMENT_SHA256 = "859ba01b285ad497fbcca63c9ef47c6e4c079c7e549ba33ca22ac24fab54f581"
LEDGER_SCRIPT_SHA256 = "0c8be46df79b42e15d1435a4d4edd60a511201786f21dcd5920e97b5e0d70cc0"
LAUNCH_PROTOCOL_SHA256 = "0d79a1b8e3cacbc345bdea464986279b0935c4cf2e20290dea75507f7fbfcd4c"
AUX_RE = re.compile(r"l15e(?P<expert>\d+)_(?P<role>up|down)\.bf16\.bin$")
PARAMETER_NAMES = (
    "s1_ew1", "s1_eb1", "s1_ew2", "s1_eb2",
    "s1_dw1", "s1_db1", "s1_dw2", "s1_db2",
    "s2_ew1", "s2_eb1", "s2_ew2", "s2_eb2",
    "s2_dw1", "s2_db1", "s2_dw2", "s2_db2",
)
DECODER_PARAMETER_NAMES = tuple(name for name in PARAMETER_NAMES if "_d" in name)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        host = np.asarray(array, dtype="<f4", order="C")
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def decoder_parameters(d: int = D, hidden: int = HIDDEN, total_bits: int = 2 * BITS) -> int:
    """Six deployment decoders, including every bias."""
    return len(DEPLOYMENT_ROLES) * hidden * (total_bits + 2 * d + 2) + 2 * len(DEPLOYMENT_ROLES) * d


def physical_ledger(experts: int) -> dict[str, float | int]:
    params = decoder_parameters()
    decoder_bytes = 2 * params
    global_bytes = decoder_bytes + HEADER_BYTES
    chunks_per_expert = WEIGHTS_PER_EXPERT // D
    code_bits = chunks_per_expert * (2 * BITS)
    if code_bits % 8:
        raise AssertionError("frozen dense stream is not byte aligned")
    code_bytes = code_bits // 8
    attributed = code_bytes + global_bytes / experts + LOCAL_SCALE_BYTES
    cold = code_bytes + global_bytes + LOCAL_SCALE_BYTES
    physical_r = 8.0 * attributed / WEIGHTS_PER_EXPERT
    side_r = physical_r - (2 * BITS / D)
    return {
        "d": D,
        "hidden": HIDDEN,
        "b1": BITS,
        "b2": BITS,
        "experts_amortized": experts,
        "decoder_parameters": params,
        "decoder_bytes": decoder_bytes,
        "header_bytes": HEADER_BYTES,
        "local_scale_bytes_per_expert": LOCAL_SCALE_BYTES,
        "code_bits_per_expert": code_bits,
        "code_bytes_per_expert": code_bytes,
        "attributed_physical_bytes_per_expert": attributed,
        "cold_bytes_per_expert": cold,
        "physical_bpw": physical_r,
        "side_bpw": side_r,
        "cold_read_amplification": cold / attributed,
        "minimum_matched_s_if_gaussian_code_is_ideal": TARGET_S + side_r,
        "target_relative_mse": 0.8 * 2.0 ** (-2.0 * physical_r),
    }


def validate_ledgers() -> dict[str, dict[str, float | int]]:
    ledgers = {"production_128": physical_ledger(128), "self_contained_panel_6": physical_ledger(6)}
    for row in ledgers.values():
        if not (2.15 <= float(row["physical_bpw"]) <= 2.5):
            raise RuntimeError(f"physical-rate gate failed: {row}")
        if not float(row["cold_read_amplification"]) < 2.0:
            raise RuntimeError(f"cold-read gate failed: {row}")
    return ledgers


def package_paths() -> dict[str, Path]:
    here = Path(__file__).resolve().parent
    redteam = here.parent / "breakthrough_redteam"
    return {
        "launch": here / "launch_protocol.json",
        "parent": redteam / "bisco_protocol_freeze.json",
        "assessment": redteam / "BISCO_BSQ_ASSESSMENT.md",
        "ledger": redteam / "bisco_bsq_ledger.py",
    }


def validate_protocol_bindings(paths: dict[str, Path] | None = None) -> dict[str, Any]:
    paths = package_paths() if paths is None else paths
    expected = {
        "launch": LAUNCH_PROTOCOL_SHA256,
        "parent": PARENT_PROTOCOL_SHA256,
        "assessment": ASSESSMENT_SHA256,
        "ledger": LEDGER_SCRIPT_SHA256,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != expected:
        raise RuntimeError({"protocol_binding_mismatch": {"expected": expected, "actual": actual}})
    launch = json.loads(paths["launch"].read_text(encoding="utf-8"))
    parent = json.loads(paths["parent"].read_text(encoding="utf-8"))
    if launch["scope"]["pinned_panel_access"] != "forbidden":
        raise RuntimeError("launch protocol no longer forbids pinned-panel access")
    if tuple(launch["data"]["validation_experts"]) != VALIDATION_EXPERTS:
        raise RuntimeError("validation split differs from executable constants")
    if tuple(parent["data_firewall"]["auxiliary_validation_experts"]) != VALIDATION_EXPERTS:
        raise RuntimeError("parent validation split differs from executable constants")
    if tuple(parent["data_firewall"]["auxiliary_experts"]) != EXPERTS:
        raise RuntimeError("parent auxiliary expert set differs from executable constants")
    return {"hashes": actual, "launch": launch, "parent": parent}


def discover_auxiliary(aux_dir: Path) -> dict[str, dict[int, Path]]:
    resolved = aux_dir.resolve()
    if "blind_protocol" in {part.lower() for part in resolved.parts}:
        raise RuntimeError("target/pinned protocol paths are categorically forbidden")
    bf16_files = sorted(resolved.glob("*.bf16.bin"))
    parsed: dict[str, dict[int, Path]] = {role: {} for role in ROLES}
    invalid = []
    for path in bf16_files:
        match = AUX_RE.fullmatch(path.name)
        if match is None:
            invalid.append(path.name)
            continue
        role = match.group("role")
        expert = int(match.group("expert"))
        if expert in parsed[role]:
            raise RuntimeError(f"duplicate auxiliary identity: {path.name}")
        parsed[role][expert] = path
    expected_names = {f"l15e{expert}_{role}.bf16.bin" for expert in EXPERTS for role in ROLES}
    actual_names = {path.name for path in bf16_files}
    if invalid or actual_names != expected_names:
        raise RuntimeError(
            {
                "auxiliary_firewall_failure": True,
                "invalid": invalid,
                "missing": sorted(expected_names - actual_names),
                "unexpected": sorted(actual_names - expected_names),
            }
        )
    for role in ROLES:
        if tuple(sorted(parsed[role])) != EXPERTS:
            raise AssertionError(parsed[role])
    return parsed


def load_bf16_canonical(path: Path, role: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != N:
        raise ValueError(f"{path}: found {raw.size} BF16 values, expected {N}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if role == "down":
        return np.ascontiguousarray(values.reshape(COLS, ROWS).T)
    return np.ascontiguousarray(values.reshape(ROWS, COLS))


def stored_normalize(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    exact_mean = float(np.mean(vector, dtype=np.float64))
    centered = vector.astype(np.float64) - exact_mean
    exact_rms = float(math.sqrt(float(np.mean(centered * centered))))
    stored_mean = float(np.float16(exact_mean))
    stored_rms = float(np.float16(exact_rms))
    if not math.isfinite(stored_rms) or stored_rms <= 0.0:
        raise RuntimeError("nonpositive FP16 stored RMS")
    normalized = ((vector - np.float32(stored_mean)) / np.float32(stored_rms)).astype(np.float32)
    return normalized.reshape(-1, D), {
        "exact_mean": exact_mean,
        "exact_centered_rms": exact_rms,
        "stored_fp16_mean": stored_mean,
        "stored_fp16_centered_rms": stored_rms,
        "source_energy": float(np.dot(vector.astype(np.float64), vector.astype(np.float64))),
    }


def role_training_moments(files: dict[int, Path], role: str) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    count = 0
    for expert in TRAIN_EXPERTS:
        vector = load_bf16_canonical(files[expert], role).reshape(-1).astype(np.float64)
        total += float(np.sum(vector))
        total_sq += float(np.dot(vector, vector))
        count += vector.size
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    return mean, math.sqrt(variance)


def gaussian_seed(base_seed: int, filename: str) -> int:
    digest = hashlib.sha256(f"BISCO-GAUSSIAN-v1|{base_seed}|{filename}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def matched_gaussian(count: int, mean: float, rms: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    standard = rng.standard_normal(count, dtype=np.float32)
    return (standard * np.float32(rms) + np.float32(mean)).astype(np.float32)


def get_backend(name: str) -> Any:
    if name == "numpy":
        return np
    if name != "cupy":
        raise ValueError(name)
    try:
        import cupy as cp  # type: ignore
    except ImportError as error:
        raise RuntimeError("production launch requires CuPy") from error
    count = int(cp.cuda.runtime.getDeviceCount())
    if count < 1:
        raise RuntimeError("CuPy imported but no CUDA device is visible")
    return cp


def to_host(array: Any, xp: Any) -> np.ndarray:
    if xp is np:
        return np.asarray(array)
    return xp.asnumpy(array)


@dataclass
class ValidationMatrix:
    expert: int
    role: str
    source: Any
    gaussian: Any
    source_scale: float
    gaussian_scale: float
    source_energy: float
    gaussian_energy: float
    source_moments: dict[str, float]
    gaussian_moments: dict[str, float]


@dataclass
class RoleData:
    source_train: Any
    gaussian_train: Any
    validation: list[ValidationMatrix]
    matched_mean: float
    matched_rms: float


def prepare_data(
    aux_dir: Path,
    xp: Any,
    gaussian_seed_base: int,
) -> tuple[dict[str, RoleData], dict[str, str], list[dict[str, Any]]]:
    files = discover_auxiliary(aux_dir)
    source_hashes = {
        path.name: sha256_file(path)
        for role in ROLES for path in (files[role][expert] for expert in EXPERTS)
    }
    role_data: dict[str, RoleData] = {}
    normalization_records: list[dict[str, Any]] = []
    for role in ROLES:
        matched_mean, matched_rms = role_training_moments(files[role], role)
        source_train: list[np.ndarray] = []
        gaussian_train: list[np.ndarray] = []
        validation: list[ValidationMatrix] = []
        for expert in EXPERTS:
            path = files[role][expert]
            raw_source = load_bf16_canonical(path, role).reshape(-1)
            source_chunks, source_meta = stored_normalize(raw_source)
            raw_gaussian = matched_gaussian(
                N,
                matched_mean,
                matched_rms,
                gaussian_seed(gaussian_seed_base, path.name),
            )
            gaussian_chunks, gaussian_meta = stored_normalize(raw_gaussian)
            normalization_records.append(
                {
                    "expert": expert,
                    "role": role,
                    "split": "validation" if expert in VALIDATION_EXPERTS else "training",
                    "source": source_meta,
                    "gaussian": gaussian_meta,
                }
            )
            if expert in VALIDATION_EXPERTS:
                validation.append(
                    ValidationMatrix(
                        expert=expert,
                        role=role,
                        source=xp.asarray(source_chunks),
                        gaussian=xp.asarray(gaussian_chunks),
                        source_scale=source_meta["stored_fp16_centered_rms"],
                        gaussian_scale=gaussian_meta["stored_fp16_centered_rms"],
                        source_energy=source_meta["source_energy"],
                        gaussian_energy=gaussian_meta["source_energy"],
                        source_moments=source_meta,
                        gaussian_moments=gaussian_meta,
                    )
                )
            else:
                source_train.append(source_chunks)
                gaussian_train.append(gaussian_chunks)
        src_host = np.concatenate(source_train, axis=0)
        gauss_host = np.concatenate(gaussian_train, axis=0)
        if src_host.shape != gauss_host.shape:
            raise AssertionError((src_host.shape, gauss_host.shape))
        role_data[role] = RoleData(
            source_train=xp.asarray(src_host),
            gaussian_train=xp.asarray(gauss_host),
            validation=validation,
            matched_mean=matched_mean,
            matched_rms=matched_rms,
        )
    return role_data, source_hashes, normalization_records


def _weight(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    scale = math.sqrt(2.0 / (rows + cols))
    return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)


def initialize_codec(seed: int, xp: Any) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for stage in ("s1", "s2"):
        params[f"{stage}_ew1"] = _weight(rng, D, HIDDEN)
        params[f"{stage}_eb1"] = np.zeros(HIDDEN, dtype=np.float32)
        params[f"{stage}_ew2"] = _weight(rng, HIDDEN, BITS)
        params[f"{stage}_eb2"] = np.zeros(BITS, dtype=np.float32)
        params[f"{stage}_dw1"] = _weight(rng, BITS, HIDDEN)
        params[f"{stage}_db1"] = np.zeros(HIDDEN, dtype=np.float32)
        params[f"{stage}_dw2"] = _weight(rng, HIDDEN, D)
        params[f"{stage}_db2"] = np.zeros(D, dtype=np.float32)
    if tuple(params) != PARAMETER_NAMES:
        raise AssertionError(tuple(params))
    return {name: xp.asarray(value) for name, value in params.items()}


def copy_codec(params: dict[str, Any]) -> dict[str, Any]:
    return {name: value.copy() for name, value in params.items()}


def sigmoid(value: Any, xp: Any) -> Any:
    clipped = xp.clip(value, -20.0, 20.0)
    return 1.0 / (1.0 + xp.exp(-clipped))


def silu(value: Any, xp: Any) -> Any:
    return value * sigmoid(value, xp)


def silu_derivative(value: Any, xp: Any) -> Any:
    sig = sigmoid(value, xp)
    return sig + value * sig * (1.0 - sig)


def stage_forward(params: dict[str, Any], prefix: str, x: Any, temperature: float, xp: Any) -> tuple[Any, dict[str, Any]]:
    ea = x @ params[f"{prefix}_ew1"] + params[f"{prefix}_eb1"]
    eh = silu(ea, xp)
    logits = eh @ params[f"{prefix}_ew2"] + params[f"{prefix}_eb2"]
    soft = xp.tanh(logits / temperature)
    q = xp.where(logits >= 0.0, 1.0, -1.0).astype(xp.float32) / math.sqrt(BITS)
    da = q @ params[f"{prefix}_dw1"] + params[f"{prefix}_db1"]
    dh = silu(da, xp)
    y = dh @ params[f"{prefix}_dw2"] + params[f"{prefix}_db2"]
    return y, {"x": x, "ea": ea, "eh": eh, "logits": logits, "soft": soft, "q": q, "da": da, "dh": dh}


def stage_backward(
    params: dict[str, Any],
    prefix: str,
    cache: dict[str, Any],
    grad_y: Any,
    temperature: float,
    balance_weight: float,
    xp: Any,
) -> tuple[Any, dict[str, Any]]:
    grads: dict[str, Any] = {}
    grads[f"{prefix}_dw2"] = cache["dh"].T @ grad_y
    grads[f"{prefix}_db2"] = xp.sum(grad_y, axis=0)
    grad_dh = grad_y @ params[f"{prefix}_dw2"].T
    grad_da = grad_dh * silu_derivative(cache["da"], xp)
    grads[f"{prefix}_dw1"] = cache["q"].T @ grad_da
    grads[f"{prefix}_db1"] = xp.sum(grad_da, axis=0)
    grad_q = grad_da @ params[f"{prefix}_dw1"].T
    ste = (1.0 - cache["soft"] * cache["soft"]) / (temperature * math.sqrt(BITS))
    grad_logits = grad_q * ste
    if balance_weight:
        bit_mean = xp.mean(cache["soft"], axis=0)
        grad_soft = (2.0 * balance_weight / (BITS * cache["soft"].shape[0])) * bit_mean[None, :]
        grad_logits = grad_logits + grad_soft * (1.0 - cache["soft"] * cache["soft"]) / temperature
    grads[f"{prefix}_ew2"] = cache["eh"].T @ grad_logits
    grads[f"{prefix}_eb2"] = xp.sum(grad_logits, axis=0)
    grad_eh = grad_logits @ params[f"{prefix}_ew2"].T
    grad_ea = grad_eh * silu_derivative(cache["ea"], xp)
    grads[f"{prefix}_ew1"] = cache["x"].T @ grad_ea
    grads[f"{prefix}_eb1"] = xp.sum(grad_ea, axis=0)
    grad_x = grad_ea @ params[f"{prefix}_ew1"].T
    return grad_x, grads


def codec_loss_and_grads(
    params: dict[str, Any],
    x: Any,
    temperature: float,
    balance_weight: float,
    xp: Any,
) -> tuple[Any, dict[str, Any]]:
    y1, cache1 = stage_forward(params, "s1", x, temperature, xp)
    residual = x - y1
    y2, cache2 = stage_forward(params, "s2", residual, temperature, xp)
    reconstruction = y1 + y2
    error = reconstruction - x
    loss = xp.mean(error * error)
    grad_reconstruction = (2.0 / error.size) * error
    grad_residual, grads2 = stage_backward(
        params, "s2", cache2, grad_reconstruction, temperature, balance_weight, xp
    )
    grad_y1 = grad_reconstruction - grad_residual
    _, grads1 = stage_backward(params, "s1", cache1, grad_y1, temperature, balance_weight, xp)
    grads = {**grads1, **grads2}
    if tuple(grads) != PARAMETER_NAMES:
        grads = {name: grads[name] for name in PARAMETER_NAMES}
    return loss, grads


class Adam:
    def __init__(self, params: dict[str, Any], beta1: float, beta2: float, epsilon: float, xp: Any):
        self.m = {name: xp.zeros_like(value) for name, value in params.items()}
        self.v = {name: xp.zeros_like(value) for name, value in params.items()}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_number = 0
        self.xp = xp

    def update(self, params: dict[str, Any], grads: dict[str, Any], learning_rate: float, clip: float) -> None:
        self.step_number += 1
        norm_sq = self.xp.asarray(0.0, dtype=self.xp.float32)
        for grad in grads.values():
            norm_sq = norm_sq + self.xp.sum(grad * grad)
        norm = self.xp.sqrt(norm_sq)
        scale = self.xp.minimum(self.xp.asarray(1.0, dtype=self.xp.float32), clip / (norm + 1e-12))
        correction1 = 1.0 - self.beta1 ** self.step_number
        correction2 = 1.0 - self.beta2 ** self.step_number
        for name in PARAMETER_NAMES:
            grad = grads[name] * scale
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * grad * grad
            mhat = self.m[name] / correction1
            vhat = self.v[name] / correction2
            params[name] -= learning_rate * mhat / (self.xp.sqrt(vhat) + self.epsilon)


def quantized_decoder_codec(params: dict[str, Any], xp: Any) -> dict[str, Any]:
    result = copy_codec(params)
    for name in DECODER_PARAMETER_NAMES:
        host = to_host(params[name], xp).astype(np.float16).astype(np.float32)
        result[name] = xp.asarray(host)
    return result


def stage_infer(params: dict[str, Any], prefix: str, x: Any, xp: Any) -> tuple[Any, Any]:
    eh = silu(x @ params[f"{prefix}_ew1"] + params[f"{prefix}_eb1"], xp)
    logits = eh @ params[f"{prefix}_ew2"] + params[f"{prefix}_eb2"]
    q = xp.where(logits >= 0.0, 1.0, -1.0).astype(xp.float32) / math.sqrt(BITS)
    return q, decode_stage(params, prefix, q, xp)


def decode_stage(params: dict[str, Any], prefix: str, q: Any, xp: Any) -> Any:
    hidden = silu(q @ params[f"{prefix}_dw1"] + params[f"{prefix}_db1"], xp)
    return hidden @ params[f"{prefix}_dw2"] + params[f"{prefix}_db2"]


def reconstruct_batch(params: dict[str, Any], x: Any, bitflip_sweeps: int, xp: Any) -> Any:
    q1, y1 = stage_infer(params, "s1", x, xp)
    q2, y2 = stage_infer(params, "s2", x - y1, xp)
    best = y1 + y2
    best_error = xp.sum((best - x) ** 2, axis=1)
    for _ in range(bitflip_sweeps):
        for prefix, q in (("s1", q1), ("s2", q2)):
            for bit in range(BITS):
                q[:, bit] *= -1.0
                candidate_stage = decode_stage(params, prefix, q, xp)
                candidate = candidate_stage + (y2 if prefix == "s1" else y1)
                candidate_error = xp.sum((candidate - x) ** 2, axis=1)
                accept = candidate_error < best_error
                q[:, bit] = xp.where(accept, q[:, bit], -q[:, bit])
                if prefix == "s1":
                    y1 = xp.where(accept[:, None], candidate_stage, y1)
                else:
                    y2 = xp.where(accept[:, None], candidate_stage, y2)
                best = xp.where(accept[:, None], candidate, best)
                best_error = xp.where(accept, candidate_error, best_error)
    return best


def normalized_sse(
    params: dict[str, Any],
    chunks: Any,
    stored_scale: float,
    bitflip_sweeps: int,
    batch_chunks: int,
    xp: Any,
) -> float:
    total = 0.0
    for start in range(0, chunks.shape[0], batch_chunks):
        batch = chunks[start : start + batch_chunks]
        reconstruction = reconstruct_batch(params, batch, bitflip_sweeps, xp)
        error = reconstruction - batch
        total += float(to_host(xp.sum(error * error), xp)) * (stored_scale ** 2)
    return total


def aggregate_evaluation(per_expert: list[dict[str, float | int]]) -> dict[str, Any]:
    q_sse = sum(float(row["qwen_sse"]) for row in per_expert)
    q_energy = sum(float(row["qwen_energy"]) for row in per_expert)
    g_sse = sum(float(row["gaussian_sse"]) for row in per_expert)
    g_energy = sum(float(row["gaussian_energy"]) for row in per_expert)
    d_qwen = q_sse / q_energy
    d_gaussian = g_sse / g_energy
    fold_s = []
    for row in per_expert:
        d_q = float(row["qwen_sse"]) / float(row["qwen_energy"])
        d_g = float(row["gaussian_sse"]) / float(row["gaussian_energy"])
        value = -0.5 * math.log2(d_q / d_g)
        row["D_Qwen"] = d_q
        row["D_Gaussian"] = d_g
        row["s_match"] = value
        fold_s.append(value)
    s_match = -0.5 * math.log2(d_qwen / d_gaussian)
    se = statistics.stdev(fold_s) / math.sqrt(len(fold_s)) if len(fold_s) > 1 else 0.0
    code_rate = 2 * BITS / D
    gaussian_f = d_gaussian * 2.0 ** (2.0 * code_rate)
    gaussian_operational_s = -0.5 * math.log2(gaussian_f)
    ledgers = validate_ledgers()
    absolute = {}
    for name, ledger in ledgers.items():
        rate = float(ledger["physical_bpw"])
        f_value = d_qwen * 2.0 ** (2.0 * rate)
        absolute[name] = {
            "physical_R": rate,
            "F": f_value,
            "s_absolute": -0.5 * math.log2(f_value),
            "target_D": float(ledger["target_relative_mse"]),
            "passes_F_0p8": f_value <= 0.8,
        }
    return {
        "D_Qwen": d_qwen,
        "D_Gaussian": d_gaussian,
        "s_match": s_match,
        "fold_s_match": fold_s,
        "whole_expert_standard_error": se,
        "upper_s_match_2se": s_match + 2.0 * se,
        "all_whole_expert_folds_positive": all(value > 0.0 for value in fold_s),
        "Gaussian_operational_gap": {
            "code_rate_bpw": code_rate,
            "F_gaussian": gaussian_f,
            "s_gaussian": gaussian_operational_s,
            "distortion_ratio_to_ideal_gaussian": gaussian_f,
        },
        "absolute": absolute,
        "per_expert": per_expert,
    }


def evaluate_models(
    models: dict[str, dict[str, dict[str, Any]]],
    data: dict[str, RoleData],
    bitflip_sweeps: int,
    batch_chunks: int,
    xp: Any,
) -> dict[str, Any]:
    accum = {
        expert: {"expert": expert, "qwen_sse": 0.0, "qwen_energy": 0.0, "gaussian_sse": 0.0, "gaussian_energy": 0.0}
        for expert in VALIDATION_EXPERTS
    }
    matrix_rows = []
    for role in ROLES:
        qwen_codec = quantized_decoder_codec(models[role]["qwen"], xp)
        gaussian_codec = quantized_decoder_codec(models[role]["gaussian"], xp)
        for matrix in data[role].validation:
            q_sse = normalized_sse(qwen_codec, matrix.source, matrix.source_scale, bitflip_sweeps, batch_chunks, xp)
            g_sse = normalized_sse(gaussian_codec, matrix.gaussian, matrix.gaussian_scale, bitflip_sweeps, batch_chunks, xp)
            row = accum[matrix.expert]
            row["qwen_sse"] += q_sse
            row["qwen_energy"] += matrix.source_energy
            row["gaussian_sse"] += g_sse
            row["gaussian_energy"] += matrix.gaussian_energy
            matrix_rows.append(
                {
                    "expert": matrix.expert,
                    "role": role,
                    "qwen_sse": q_sse,
                    "qwen_energy": matrix.source_energy,
                    "gaussian_sse": g_sse,
                    "gaussian_energy": matrix.gaussian_energy,
                }
            )
    result = aggregate_evaluation([accum[expert] for expert in VALIDATION_EXPERTS])
    result["per_matrix"] = matrix_rows
    return result


def early_kill_decision(half: dict[str, Any], quarter: dict[str, Any]) -> dict[str, Any]:
    half_upper = float(half["upper_s_match_2se"])
    quarter_upper = float(quarter["upper_s_match_2se"])
    improvement = quarter_upper - half_upper
    projected = quarter_upper + 6.0 * max(0.0, improvement)
    killed = quarter_upper < 0.08 and improvement < 0.01
    return {
        "half_checkpoint_upper_s_match_2se": half_upper,
        "quarter_checkpoint_upper_s_match_2se": quarter_upper,
        "late_improvement": improvement,
        "constant_recent_slope_projection_to_full_budget": projected,
        "boundary_projection_max": 0.14,
        "kill_if_upper_below": 0.08,
        "and_improvement_below": 0.01,
        "kill": killed,
        "interpretation": "preregistered empirical trend kill; not a mathematical converse",
    }


def flatten_model_arrays(models: dict[str, dict[str, dict[str, Any]]], domain: str, names: tuple[str, ...], xp: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays = []
    schema = []
    offset = 0
    for role in ROLES:
        for name in names:
            host = to_host(models[role][domain][name], xp).astype(np.float32, copy=False)
            flat = host.reshape(-1)
            arrays.append(flat)
            schema.append({"role": role, "parameter": name, "shape": list(host.shape), "offset_values": offset, "values": flat.size})
            offset += flat.size
    return np.concatenate(arrays), schema


def write_model_artifacts(output_dir: Path, models: dict[str, dict[str, dict[str, Any]]], xp: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in ("qwen", "gaussian"):
        state, schema = flatten_model_arrays(models, domain, PARAMETER_NAMES, xp)
        state_path = output_dir / f"{domain}_training_state.fp32.bin"
        state.astype("<f4").tofile(state_path)
        decoder, decoder_schema = flatten_model_arrays(models, domain, DECODER_PARAMETER_NAMES, xp)
        decoder_path = output_dir / f"{domain}_aux_up_down_decoder.fp16.bin"
        decoder.astype("<f2").tofile(decoder_path)
        result[domain] = {
            "training_state": {"file": state_path.name, "bytes": state_path.stat().st_size, "sha256": sha256_file(state_path), "schema": schema},
            "auxiliary_two_role_decoder": {
                "file": decoder_path.name,
                "bytes": decoder_path.stat().st_size,
                "sha256": sha256_file(decoder_path),
                "schema": decoder_schema,
                "not_the_deployment_ledger_decoder": True,
            },
        }
    return result


def model_initialization_hashes(models: dict[str, dict[str, dict[str, Any]]], xp: Any) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for role in ROLES:
        result[role] = {}
        for domain in ("qwen", "gaussian"):
            result[role][domain] = sha256_arrays(to_host(models[role][domain][name], xp) for name in PARAMETER_NAMES)
        if result[role]["qwen"] != result[role]["gaussian"]:
            raise AssertionError("paired initialization diverged")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    bindings = validate_protocol_bindings()
    launch = bindings["launch"]
    if args.backend != "cupy" and not args.allow_cpu_test_backend:
        raise RuntimeError("non-CuPy execution is allowed only with --allow-cpu-test-backend")
    xp = get_backend(args.backend)
    data, source_hashes, normalization = prepare_data(
        args.aux_dir,
        xp,
        int(launch["optimization"]["seed"]) + int(launch["optimization"]["gaussian_seed_offset"]),
    )
    seed = int(launch["optimization"]["seed"])
    models: dict[str, dict[str, dict[str, Any]]] = {}
    optimizers: dict[str, dict[str, Adam]] = {}
    for role_index, role in enumerate(ROLES):
        initial = initialize_codec(seed + 1000 * role_index, xp)
        models[role] = {"qwen": copy_codec(initial), "gaussian": copy_codec(initial)}
        optimizers[role] = {
            domain: Adam(
                models[role][domain],
                float(launch["optimization"]["adam_beta1"]),
                float(launch["optimization"]["adam_beta2"]),
                float(launch["optimization"]["adam_epsilon"]),
                xp,
            )
            for domain in ("qwen", "gaussian")
        }
    initialization_hashes = model_initialization_hashes(models, xp)
    rng = np.random.default_rng(seed + 17)
    max_updates = int(launch["optimization"]["max_updates"])
    batch_size = int(launch["optimization"]["batch_chunks_per_role"])
    eval_steps = set(int(value) for value in launch["evaluation"]["steps"])
    history: list[dict[str, Any]] = []
    early_decision = None
    stopped_update = max_updates
    for update in range(1, max_updates + 1):
        fraction = (update - 1) / max(1, max_updates - 1)
        temperature = float(launch["optimization"]["temperature_start"]) * (
            float(launch["optimization"]["temperature_end"]) / float(launch["optimization"]["temperature_start"])
        ) ** fraction
        for role in ROLES:
            role_data = data[role]
            indices_host = rng.integers(0, role_data.source_train.shape[0], size=batch_size, dtype=np.int64)
            indices = xp.asarray(indices_host)
            batches = {
                "qwen": role_data.source_train[indices],
                "gaussian": role_data.gaussian_train[indices],
            }
            for domain in ("qwen", "gaussian"):
                _, grads = codec_loss_and_grads(
                    models[role][domain],
                    batches[domain],
                    temperature,
                    float(launch["optimization"]["bit_balance_weight"]),
                    xp,
                )
                optimizers[role][domain].update(
                    models[role][domain],
                    grads,
                    float(launch["optimization"]["learning_rate"]),
                    float(launch["optimization"]["global_gradient_norm_clip"]),
                )
        if update in eval_steps:
            if xp is not np:
                xp.cuda.Stream.null.synchronize()
            evaluation = evaluate_models(
                models,
                data,
                int(launch["evaluation"]["greedy_bitflip_sweeps"]),
                int(launch["evaluation"]["batch_chunks"]),
                xp,
            )
            history.append({"update": update, "temperature": temperature, "evaluation": evaluation})
            if update == int(launch["early_kill"]["quarter_checkpoint_update"]):
                prior = next(row["evaluation"] for row in history if row["update"] == int(launch["early_kill"]["half_checkpoint_update"]))
                early_decision = early_kill_decision(prior, evaluation)
                if early_decision["kill"]:
                    stopped_update = update
                    break
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = write_model_artifacts(args.output_dir, models, xp)
    final_evaluation = history[-1]["evaluation"]
    if early_decision is None:
        early_decision = {"kill": False, "reason": "quarter checkpoint not reached"}
    if early_decision["kill"]:
        decision = "HARD_KILL_D16_SHALLOW_BEFORE_PINNED"
    else:
        full = stopped_update == max_updates and history[-1]["update"] == max_updates
        production_pass = bool(final_evaluation["absolute"]["production_128"]["passes_F_0p8"])
        panel_pass = bool(final_evaluation["absolute"]["self_contained_panel_6"]["passes_F_0p8"])
        fold_pass = bool(final_evaluation["all_whole_expert_folds_positive"])
        decision = "AUXILIARY_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL" if full and production_pass and panel_pass and fold_pass else "NO_PROMOTION_FROM_AUXILIARY_D16"
    result = {
        "protocol": "bisco_raw_mse_aux_d16_launch_v1",
        "decision": decision,
        "pinned_panel": {
            "opened": False,
            "path_argument_supported": False,
            "authorization": "forbidden by launch protocol",
            "survival_caveat": "even a survival cannot open the pinned panel until a separate gate-role auxiliary protocol is frozen",
        },
        "strict_ptq": True,
        "backend": {
            "name": args.backend,
            "cupy_version": None if xp is np else xp.__version__,
            "device": None if xp is np else int(xp.cuda.Device().id),
        },
        "data_firewall": {
            "auxiliary_directory": str(args.aux_dir.resolve()),
            "layer": 15,
            "training_experts": list(TRAIN_EXPERTS),
            "untouched_validation_experts": list(VALIDATION_EXPERTS),
            "roles": list(ROLES),
            "source_sha256": source_hashes,
            "normalization": normalization,
            "gaussian_role_training_moments": {
                role: {"mean": data[role].matched_mean, "centered_rms": data[role].matched_rms}
                for role in ROLES
            },
        },
        "paired_control": {
            "initialization_sha256": initialization_hashes,
            "identical_batch_indices": True,
            "identical_optimizer_and_updates": True,
            "identical_bitflip_search": True,
        },
        "training": {
            "stopped_update": stopped_update,
            "max_updates": max_updates,
            "history": history,
            "early_kill": early_decision,
        },
        "final_evaluation": final_evaluation,
        "physical_ledger": validate_ledgers(),
        "artifacts": artifacts,
        "bindings": {
            **bindings["hashes"],
            "executed_script_sha256": sha256_file(Path(__file__)),
        },
        "claim_boundary": launch["scope"]["claim_boundary"],
        "runtime_seconds": time.time() - started,
    }
    result_path = args.output_dir / "bisco_raw_mse_result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "stopped_update": stopped_update,
                "D_Qwen": final_evaluation["D_Qwen"],
                "D_Gaussian": final_evaluation["D_Gaussian"],
                "s_match": final_evaluation["s_match"],
                "upper_s_match_2se": final_evaluation["upper_s_match_2se"],
                "physical_ledger": result["physical_ledger"],
                "result": str(result_path),
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument(
        "--allow-cpu-test-backend",
        action="store_true",
        help="unit/synthetic testing only; a result produced with this flag is not a production gate",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
