#!/usr/bin/env python3
"""Independent, standard-library-only verifier for the lossy-tail-v8 result.

This verifier never imports the producer module (whose authenticated-context
firewall is intentionally not bypassed), CuPy, NumPy, torch, or any model
loader.  It treats ``result.json`` and ``authorization.json`` as hostile input,
binds them to exact external hashes, and independently derives the arithmetic
and decisions exposed by the frozen result format.
"""

from __future__ import annotations

import hashlib
import functools
import json
import math
import os
import posixpath
import re
import statistics
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
PRODUCER = RESEARCH / "lossy_tail_peeling_oracle_v8"
SOURCE_AUDIT = RESEARCH / "lossy_tail_peeling_oracle_v8_independent_source_audit_20260901"
RUNTIME_AUDIT = RESEARCH / "lossy_tail_peeling_oracle_v8_runtime_audit_20260901"

RESULT_PATH = HERE / "result.json"
AUTHORIZATION_PATH = HERE / "authorization.json"
AUDIT_RECEIPT_PATH = HERE / "audit_receipt.json"
AUDIT_MANIFEST_PATH = HERE / "audit_manifest.json"

RESULT_BYTES = 3_800_771
RESULT_FILE_SHA256 = "2f3ebe509fa3c78c2caf6084510bb14e9e2a2fef9cabbdb4b99c9b396a4bfdf9"
RESULT_INTERNAL_SHA256 = "83ef57d23fddc1ecc5443ecc5bd4da5a35497c8f5e7e5f9d6584095458943d1c"
AUTHORIZATION_BYTES = 5_960
AUTHORIZATION_FILE_SHA256 = "16bf378c1c6baa23eaff7054ca4c1b82fa06ec45bd89e152956d6a51c752d1ef"
AUTHORIZATION_INTERNAL_SHA256 = "5dd327a2a4b15be84c57c8dc9821d9d4ade707e254083730ca6449d4a90207a9"

KNOWN_FILE_HASHES = {
    PRODUCER / "launch_manifest.json": "6c5f5cd05973dbc0bf16cd9ea39951e690b15e15e13e969d2a33823117c2aa94",
    PRODUCER / "protocol_lock.json": "7cc2ce69587f6426e108a3970e3e61a5f467f0c984622b98989af251ce758e9b",
    PRODUCER / "repair_lock.json": "7bcaf3bca835f34bbbf3df39c08c2ebebf70890a4f2e9b5f9128ed71c0fea442",
    PRODUCER / "runtime_contract.json": "c83f065122a05f05e5b4788ffc717b517e8c0310cb771dfc1324fc1115745f0a",
    PRODUCER / "source_bindings.json": "79edb94c8227e89debaf7eac5f4e181924d97300c9f7bc961a29151292e62a22",
    PRODUCER / "lossy_tail_oracle.py": "b8574858ab90c528769cbdd3cd6d0cdf5cd62f58dfa2bf6e46ec8f465a2059b2",
    PRODUCER / "lossy_tail_core.py": "be51a7f895af2cf1f2863b491affa4e9dbdcc2eb59798e239560bd1f85172c66",
    PRODUCER / "preflight_launch.py": "f32618bcb33ffb3a3a319b0dbfb2c7e2df18e437cfeec478edec80bff3c2bca8",
    SOURCE_AUDIT / "audit_manifest.json": "045eac28701decf60837be335cfdf316b3ab1650125bc2f3744f097c0e75bb87",
    SOURCE_AUDIT / "audit_receipt.json": "6e2bc929904c7c05274a3c2bfc634707336d8646f85db2776ef024032988dde0",
    SOURCE_AUDIT / "replay_receipt.json": "c6f2eb7405d8f24ad707e88216bc0a278da44b8ec50ac8cc13e993c086a71fd0",
    RUNTIME_AUDIT / "audit_manifest.json": "7d0fbd622fd061641a3c0b96f00d1c6d4bf9c4b5f785d664fd6c8268df2a4134",
    RUNTIME_AUDIT / "audit_receipt.json": "4c1f72a11beff243ac67460dacdc18b9eac7b69445bfebc15f38a0593b135321",
    RUNTIME_AUDIT / "runtime_receipt.json": "45862549f34530964c4f8f7a4134228ccf036a8de3534f23e63e07acde7985b3",
}

ROWS = 768
COLS = 2048
N = ROWS * COLS
EXPERTS = 6
ROLES = 2
MATRICES = EXPERTS * ROLES
PANEL_N = N * MATRICES
RATES = (2.15, 2.30, 2.50)
MODES = ("free_lloyd", "finite_fp16", "zero_tail_error")
GEOMETRIES = ("raw_adaptive", "support_xklt_uniform")
FAMILIES = (("coordinate", 1), ("block16", 16), ("block64", 64))
FRACTIONS = (0.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0)
LEVELS = (1, 2, 4, 8, 16)
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
KILL_GUARD_S = 0.02
KILL_THRESHOLD_S = TARGET_S - KILL_GUARD_S
NUMERIC_BOUNDARY_GUARD_S = 1.0e-4
DECISION_EPSILON_S = 1.0e-12
FLOAT32_EPSILON = 2.0 ** -23
COMMON_BITS = 4096 * 8 + 144 * 8
EXPERT_HEADER_BITS = 256
MATRIX_DESCRIPTOR_BITS = 128
RESIDUAL_DIRECTORY_BITS = 64
ANGLE_BITS = 16
END_PAD_RESERVE_BITS = 7 * EXPERTS
PAGE_BYTES = 4096
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

RESULT_ROOT_KEYS = {
    "schema", "authorization", "launch_manifest", "protocol", "repair_lock",
    "runtime_contract", "runtime_receipt", "bindings", "oracle_bootstrap_sha256",
    "scientific_core_sha256", "child_capability", "mountinfo_sha256", "runtime",
    "panel", "grid", "qwen_search", "matched_gaussian_controls",
    "calibrated_rows", "decision", "result_lock_sha256",
}
SCORE_KEYS = {
    "valid", "panel", "mode", "geometry", "requested_rate_bpw",
    "physical_rate_bpw", "capacity_bytes", "physical_bits", "choices", "profiles",
    "peeled_scalars", "peeled_fraction", "peeled_energy_fraction",
    "tail_distortion_sse", "bulk_ideal_distortion_sse", "total_distortion_sse",
    "source_energy", "ideal_relative_mse", "F", "s_bpw", "gaussian_limit_mse",
    "target_mse", "passes_absolute_F", "side_ledger", "component_count", "read_ledger",
}
PROFILE_KEYS = {
    "id", "family", "unit", "fraction", "levels", "selected_units",
    "selected_scalars", "support_bits", "support_stream_bits", "symbol_bits",
    "symbol_stream_bits", "codebook_bits", "tail_energy", "bulk_energy",
    "free_lloyd_sse", "fp16_sse", "centroids", "fp16_centroids",
}
CALIBRATED_KEYS = {
    "rate", "mode", "geometry", "qwen_F", "qwen_s_bpw", "control_F",
    "control_s_bpw", "control_mean_s_bpw", "control_sample_std_s_bpw",
    "qwen_excess_s_bpw", "calibrated_F", "fraction_of_required_s",
    "passes_calibrated_F", "passes_absolute_F", "below_2x",
}


class AuditFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def true(self, condition: bool, label: str) -> None:
        self.count += 1
        if not condition:
            raise AuditFailure(label)

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        self.count += 1
        if observed != expected:
            raise AuditFailure(f"{label}: observed={observed!r} expected={expected!r}")

    def close(self, observed: Any, expected: Any, label: str, *, rel: float = 3e-13, absolute: float = 2e-15) -> None:
        self.count += 1
        try:
            left, right = float(observed), float(expected)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AuditFailure(f"{label}: nonnumeric value") from exc
        if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left, right, rel_tol=rel, abs_tol=absolute):
            raise AuditFailure(f"{label}: observed={left!r} expected={right!r}")


C = Checks()


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise AuditFailure(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def reject_constant(value: str) -> Any:
    raise AuditFailure(f"nonfinite JSON token {value}")


def load_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise AuditFailure(f"{path.name}: not UTF-8") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"{path.name}: root is not an object")
    return payload, value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_internal(value: dict[str, Any], field: str, expected: str, label: str) -> None:
    C.equal(value.get(field), expected, f"{label} internal field")
    copy = dict(value)
    copy.pop(field, None)
    C.equal(digest(canonical_bytes(copy)), expected, f"{label} recomputed internal seal")


def exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    C.true(isinstance(value, dict), f"{label} is an object")
    C.equal(set(value), keys, f"{label} exact keys")
    return value


def finite(value: Any, label: str) -> float:
    C.true(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} numeric type")
    converted = float(value)
    C.true(math.isfinite(converted), f"{label} finite")
    return converted


def finite_tree(value: Any, label: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        C.true(math.isfinite(float(value)), f"{label} finite")
        return
    if isinstance(value, list):
        for ordinal, child in enumerate(value):
            finite_tree(child, f"{label}[{ordinal}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{label}.{key}")
        return
    raise AuditFailure(f"{label}: unsupported tree type {type(value).__name__}")


def sha(value: Any, label: str) -> str:
    C.true(isinstance(value, str) and SHA_RE.fullmatch(value) is not None, f"{label} lowercase SHA-256")
    return value


def compare_numeric_tree(observed: Any, expected: Any, label: str) -> None:
    """Compare independently derived trees while allowing last-bit reduction drift."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        C.equal(observed, expected, label)
    elif isinstance(expected, int):
        C.equal(observed, expected, label)
    elif isinstance(expected, float):
        C.close(observed, expected, label, rel=2e-8, absolute=2e-16)
    elif isinstance(expected, list):
        C.true(isinstance(observed, list) and len(observed) == len(expected), f"{label} list shape")
        for ordinal, (left, right) in enumerate(zip(observed, expected)):
            compare_numeric_tree(left, right, f"{label}[{ordinal}]")
    elif isinstance(expected, dict):
        C.true(isinstance(observed, dict), f"{label} object")
        C.equal(set(observed), set(expected), f"{label} keys")
        for key, right in expected.items():
            compare_numeric_tree(observed[key], right, f"{label}.{key}")
    else:
        raise AuditFailure(f"{label}: unsupported comparison type {type(expected).__name__}")


def verify_file(path: Path, expected: str) -> None:
    C.true(path.is_file(), f"required file exists: {path}")
    C.equal(digest(path.read_bytes()), expected, f"file SHA-256: {path.name}")


@functools.lru_cache(maxsize=None)
def ceil_log2_binomial(n: int, k: int) -> int:
    count = math.comb(n, min(k, n - k))
    return 0 if count <= 1 else (count - 1).bit_length()


def pad8(bits: int) -> int:
    return 8 * ((bits + 7) // 8)


def make_profiles() -> list[dict[str, Any]]:
    rows = [{"id": 0, "family": "coordinate", "unit": 1, "fraction": 0.0, "levels": 1}]
    for family, unit in FAMILIES:
        for fraction in FRACTIONS[1:]:
            for levels in LEVELS:
                rows.append({
                    "id": len(rows), "family": family, "unit": unit,
                    "fraction": fraction, "levels": levels,
                })
    return rows


PROFILES = make_profiles()


def stable_seed(*parts: object) -> int:
    blob = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def verify_memory(rows: Any, label: str) -> None:
    C.true(isinstance(rows, list) and len(rows) == EXPERTS, f"{label} six memory rows")
    expected_keys = {
        "expert_ordinal", "stream_synchronized", "used_bytes_before_free",
        "total_bytes_before_free", "used_bytes_after_free", "total_bytes_after_free",
        "all_per_expert_gpu_arrays_deleted_before_free",
    }
    for ordinal, row in enumerate(rows):
        exact_keys(row, expected_keys, f"{label}[{ordinal}]")
        C.equal(row["expert_ordinal"], ordinal, f"{label}[{ordinal}] ordinal")
        C.equal(row["stream_synchronized"], True, f"{label}[{ordinal}] synchronized")
        C.equal(row["used_bytes_before_free"], 0, f"{label}[{ordinal}] used before free")
        C.true(isinstance(row["total_bytes_before_free"], int) and row["total_bytes_before_free"] >= 0, f"{label}[{ordinal}] cached bytes")
        C.equal(row["used_bytes_after_free"], 0, f"{label}[{ordinal}] used after free")
        C.equal(row["total_bytes_after_free"], 0, f"{label}[{ordinal}] total after free")
        C.equal(row["all_per_expert_gpu_arrays_deleted_before_free"], True, f"{label}[{ordinal}] arrays deleted")


def verify_moments(rows: Any, label: str, *, control: bool, qwen_moments: list[dict[str, Any]] | None = None, replica: int | None = None) -> list[dict[str, Any]]:
    C.true(isinstance(rows, list) and len(rows) == MATRICES, f"{label} 12 moments")
    mismatch_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        keys = {"energy", "mean", "variance", "expert_ordinal", "role"}
        if control:
            keys.add("control_affine")
        exact_keys(row, keys, f"{label}[{ordinal}]")
        energy = finite(row["energy"], f"{label}[{ordinal}].energy")
        mean = finite(row["mean"], f"{label}[{ordinal}].mean")
        variance = finite(row["variance"], f"{label}[{ordinal}].variance")
        C.true(energy > 0.0 and variance > 0.0, f"{label}[{ordinal}] positive energy/variance")
        C.equal(row["expert_ordinal"], ordinal // ROLES, f"{label}[{ordinal}] expert")
        C.equal(row["role"], ("up", "down_transposed")[ordinal % ROLES], f"{label}[{ordinal}] role")
        C.close(energy, N * (variance + mean * mean), f"{label}[{ordinal}] moment-energy identity", rel=2e-14, absolute=1e-12)
        if not control:
            continue
        assert qwen_moments is not None and replica is not None
        affine = exact_keys(row["control_affine"], {
            "seed_u64", "pre_affine_bf16_mean", "pre_affine_bf16_variance",
            "scale", "offset", "observed_mean", "observed_variance", "target_mean",
            "target_variance", "mean_absolute_error", "mean_absolute_tolerance",
            "mean_normalized_mismatch", "variance_absolute_error",
            "variance_absolute_tolerance", "variance_normalized_mismatch",
        }, f"{label}[{ordinal}].control_affine")
        C.equal(affine["seed_u64"], stable_seed("lossy-tail-peeling-gaussian-v1", replica, ordinal), f"{label}[{ordinal}] seed")
        pre_mean = finite(affine["pre_affine_bf16_mean"], f"{label}[{ordinal}] pre mean")
        pre_variance = finite(affine["pre_affine_bf16_variance"], f"{label}[{ordinal}] pre variance")
        scale = finite(affine["scale"], f"{label}[{ordinal}] scale")
        offset = finite(affine["offset"], f"{label}[{ordinal}] offset")
        observed_mean = finite(affine["observed_mean"], f"{label}[{ordinal}] observed mean")
        observed_variance = finite(affine["observed_variance"], f"{label}[{ordinal}] observed variance")
        target_mean = finite(affine["target_mean"], f"{label}[{ordinal}] target mean")
        target_variance = finite(affine["target_variance"], f"{label}[{ordinal}] target variance")
        C.true(pre_variance > 0.0 and scale > 0.0 and observed_variance > 0.0 and target_variance > 0.0, f"{label}[{ordinal}] positive affine values")
        C.equal(target_mean, qwen_moments[ordinal]["mean"], f"{label}[{ordinal}] target mean binds Qwen")
        C.equal(target_variance, qwen_moments[ordinal]["variance"], f"{label}[{ordinal}] target variance binds Qwen")
        C.close(scale, math.sqrt(target_variance / pre_variance), f"{label}[{ordinal}] fitted scale")
        C.close(offset, target_mean - scale * pre_mean, f"{label}[{ordinal}] fitted offset")
        mean_tol = 64 * FLOAT32_EPSILON * max(math.sqrt(target_variance), abs(target_mean), 2.0 ** -126)
        variance_tol = 256 * FLOAT32_EPSILON * max(target_variance, target_mean * target_mean, 2.0 ** -252)
        mean_error = abs(observed_mean - target_mean)
        variance_error = abs(observed_variance - target_variance)
        C.close(affine["mean_absolute_tolerance"], mean_tol, f"{label}[{ordinal}] mean tolerance")
        C.close(affine["variance_absolute_tolerance"], variance_tol, f"{label}[{ordinal}] variance tolerance")
        C.close(affine["mean_absolute_error"], mean_error, f"{label}[{ordinal}] mean error")
        C.close(affine["variance_absolute_error"], variance_error, f"{label}[{ordinal}] variance error")
        C.close(affine["mean_normalized_mismatch"], mean_error / mean_tol, f"{label}[{ordinal}] normalized mean")
        C.close(affine["variance_normalized_mismatch"], variance_error / variance_tol, f"{label}[{ordinal}] normalized variance")
        C.true(mean_error <= mean_tol and variance_error <= variance_tol, f"{label}[{ordinal}] moment tolerance pass")
        C.close(mean, observed_mean, f"{label}[{ordinal}] recorded mean")
        C.close(variance, observed_variance, f"{label}[{ordinal}] recorded variance")
        mismatch_rows.append({
            "replica": replica, "ordinal": ordinal,
            "mean_absolute_error": mean_error, "mean_absolute_tolerance": mean_tol,
            "mean_normalized_mismatch": mean_error / mean_tol,
            "variance_absolute_error": variance_error, "variance_absolute_tolerance": variance_tol,
            "variance_normalized_mismatch": variance_error / variance_tol,
        })
    return mismatch_rows


def fp16_round(value: float) -> float:
    return float(struct.unpack("<e", struct.pack("<e", value))[0])


def verify_profile(profile: Any, static: dict[str, Any], moment: dict[str, Any], label: str) -> None:
    exact_keys(profile, PROFILE_KEYS, label)
    for key in ("id", "family", "unit", "fraction", "levels"):
        C.equal(profile[key], static[key], f"{label}.{key}")
    units = N // static["unit"]
    selected_units = int(round(units * static["fraction"]))
    selected_scalars = selected_units * static["unit"]
    support_bits = ceil_log2_binomial(units, selected_units)
    symbol_bits = selected_scalars * int(round(math.log2(static["levels"])))
    C.equal(profile["selected_units"], selected_units, f"{label}.selected_units")
    C.equal(profile["selected_scalars"], selected_scalars, f"{label}.selected_scalars")
    C.equal(profile["support_bits"], support_bits, f"{label}.support_bits")
    C.equal(profile["support_stream_bits"], pad8(support_bits), f"{label}.support_stream_bits")
    C.equal(profile["symbol_bits"], symbol_bits, f"{label}.symbol_bits")
    C.equal(profile["symbol_stream_bits"], pad8(symbol_bits), f"{label}.symbol_stream_bits")
    C.equal(profile["codebook_bits"], 16 * static["levels"] if selected_scalars else 0, f"{label}.codebook_bits")
    tail = finite(profile["tail_energy"], f"{label}.tail_energy")
    bulk = finite(profile["bulk_energy"], f"{label}.bulk_energy")
    free_sse = finite(profile["free_lloyd_sse"], f"{label}.free_lloyd_sse")
    fp16_sse = finite(profile["fp16_sse"], f"{label}.fp16_sse")
    C.true(tail >= 0 and bulk >= 0 and free_sse >= 0 and fp16_sse >= 0, f"{label} nonnegative energies")
    C.close(tail + bulk, moment["energy"], f"{label} energy partition", rel=3e-14, absolute=1e-12)
    C.true(free_sse <= fp16_sse + max(1e-15, abs(fp16_sse) * 2e-12), f"{label} free Lloyd is optimistic")
    centroids, rounded = profile["centroids"], profile["fp16_centroids"]
    expected_length = static["levels"] if selected_scalars else 0
    C.true(isinstance(centroids, list) and len(centroids) == expected_length, f"{label} centroid count")
    C.true(isinstance(rounded, list) and len(rounded) == expected_length, f"{label} fp16 centroid count")
    previous = -math.inf
    for ordinal, (centroid, fp16) in enumerate(zip(centroids, rounded)):
        centroid = finite(centroid, f"{label}.centroids[{ordinal}]")
        fp16 = finite(fp16, f"{label}.fp16_centroids[{ordinal}]")
        C.true(centroid >= previous, f"{label} sorted centroids")
        previous = centroid
        C.equal(fp16, fp16_round(centroid), f"{label} literal FP16 centroid")


def expected_tail_side(profile: dict[str, Any], mode: str) -> int:
    bits = profile["support_stream_bits"] + profile["symbol_stream_bits"]
    if mode == "finite_fp16" and profile["selected_scalars"]:
        bits += profile["codebook_bits"]
    return bits


def verify_score(row: Any, *, rate: float, mode: str, geometry_key: str, expected_panel: str,
                 moments: list[dict[str, Any]], total_energy: float, retained: bool, label: str) -> None:
    expected = SCORE_KEYS | ({"allocations"} if retained else set())
    exact_keys(row, expected, label)
    C.equal(row["valid"], True, f"{label}.valid")
    C.equal(row["panel"], expected_panel, f"{label}.panel")
    C.equal(row["mode"], mode, f"{label}.mode")
    raw_geometry = "raw" if geometry_key.startswith("raw_") else "support_xklt_uniform"
    C.equal(row["geometry"], raw_geometry, f"{label}.geometry")
    C.equal(row["requested_rate_bpw"], rate, f"{label}.requested_rate")
    capacity = int(math.floor(rate * PANEL_N / 8.0))
    physical_bits = capacity * 8
    physical_rate = physical_bits / PANEL_N
    C.equal(row["capacity_bytes"], capacity, f"{label}.capacity")
    C.equal(row["physical_bits"], physical_bits, f"{label}.physical_bits")
    C.close(row["physical_rate_bpw"], physical_rate, f"{label}.physical_rate", rel=0.0, absolute=0.0)
    choices = row["choices"]
    profiles = row["profiles"]
    C.true(isinstance(choices, list) and len(choices) == MATRICES, f"{label}.choices")
    C.true(isinstance(profiles, list) and len(profiles) == MATRICES, f"{label}.profiles")
    if geometry_key in {"raw_uniform_best", "raw_uniform_global_diagnostic", "support_xklt_uniform", "support_xklt_uniform_global_diagnostic"}:
        C.equal(len(set(choices)), 1, f"{label} uniform choices")
    for ordinal, (choice, profile) in enumerate(zip(choices, profiles)):
        C.true(isinstance(choice, int) and not isinstance(choice, bool) and 0 <= choice < len(PROFILES), f"{label}.choice[{ordinal}]")
        verify_profile(profile, PROFILES[choice], moments[ordinal], f"{label}.profile[{ordinal}]")
    peeled = sum(p["selected_scalars"] for p in profiles)
    tail_energy = sum(p["tail_energy"] for p in profiles)
    C.equal(row["peeled_scalars"], peeled, f"{label}.peeled_scalars")
    C.close(row["peeled_fraction"], peeled / PANEL_N, f"{label}.peeled_fraction")
    C.close(row["peeled_energy_fraction"], tail_energy / total_energy, f"{label}.peeled_energy_fraction")
    if mode == "free_lloyd":
        tail_sse = sum(p["free_lloyd_sse"] for p in profiles)
    elif mode == "finite_fp16":
        tail_sse = sum(p["fp16_sse"] for p in profiles)
    else:
        tail_sse = 0.0
    C.close(row["tail_distortion_sse"], tail_sse, f"{label}.tail_sse", rel=5e-13, absolute=2e-13)
    C.close(row["source_energy"], total_energy, f"{label}.source_energy")

    side = exact_keys(row["side_ledger"], {
        "common_bits", "expert_side_bits", "tail_and_codebook_bits",
        "residual_directory_bits", "xklt_angle_bits", "end_pad_reserve_bits",
        "actual_end_pad_bits", "trailer_bits", "payload_bits", "bit_closure",
    }, f"{label}.side_ledger")
    component_count = row["component_count"]
    C.true(isinstance(component_count, int) and not isinstance(component_count, bool) and component_count > 0, f"{label}.component_count")
    C.equal(side["common_bits"], COMMON_BITS, f"{label}.common_bits")
    C.equal(side["tail_and_codebook_bits"], sum(expected_tail_side(p, mode) for p in profiles), f"{label}.tail bits")
    C.equal(side["residual_directory_bits"], component_count * RESIDUAL_DIRECTORY_BITS, f"{label}.directory bits")
    C.equal(side["end_pad_reserve_bits"], END_PAD_RESERVE_BITS, f"{label}.end pad reserve")
    C.true(isinstance(side["expert_side_bits"], list) and len(side["expert_side_bits"]) == EXPERTS, f"{label}.expert side rows")
    C.true(all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in side["expert_side_bits"]), f"{label}.expert side integers")
    expected_payload = physical_bits - COMMON_BITS - sum(side["expert_side_bits"]) - END_PAD_RESERVE_BITS
    C.equal(side["payload_bits"], expected_payload, f"{label}.payload budget")
    C.true(expected_payload > 0, f"{label}.positive payload")

    allocation_payload = [0] * EXPERTS
    component_by_expert = [0] * EXPERTS
    angle_by_expert = [0] * EXPERTS
    if retained:
        allocations = row["allocations"]
        C.true(isinstance(allocations, list) and len(allocations) == component_count, f"{label}.allocation count")
        C.equal(len({a.get("name") for a in allocations}), len(allocations), f"{label}.unique allocation names")
        allocation_distortion = 0.0
        for ordinal, allocation in enumerate(allocations):
            exact_keys(allocation, {"name", "owner_expert", "dimension", "energy", "payload_bits", "distortion_sse"}, f"{label}.allocation[{ordinal}]")
            name = allocation["name"]
            owner = allocation["owner_expert"]
            dimension = allocation["dimension"]
            bits = allocation["payload_bits"]
            energy = finite(allocation["energy"], f"{label}.allocation[{ordinal}].energy")
            distortion = finite(allocation["distortion_sse"], f"{label}.allocation[{ordinal}].distortion")
            C.true(isinstance(name, str) and name, f"{label}.allocation[{ordinal}].name")
            C.true(isinstance(owner, int) and not isinstance(owner, bool) and 0 <= owner < EXPERTS, f"{label}.allocation[{ordinal}].owner")
            C.true(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0, f"{label}.allocation[{ordinal}].dimension")
            C.true(isinstance(bits, int) and not isinstance(bits, bool) and bits >= 0, f"{label}.allocation[{ordinal}].bits")
            C.true(energy > 0.0 and distortion > 0.0, f"{label}.allocation[{ordinal}] positive")
            expected_distortion = energy * math.exp(-2.0 * bits / dimension * math.log(2.0))
            C.close(distortion, expected_distortion, f"{label}.allocation[{ordinal}] RD equation", rel=2e-13, absolute=1e-15)
            allocation_payload[owner] += bits
            component_by_expert[owner] += 1
            if ".both_axis_" in name:
                angle_by_expert[owner] = 1
            allocation_distortion += distortion
            if raw_geometry == "raw":
                match = re.fullmatch(r"matrix_(\d{2})\.bulk", name)
                C.true(match is not None, f"{label}.allocation[{ordinal}] raw name")
                matrix_ordinal = int(match.group(1))
                C.equal(owner, matrix_ordinal // ROLES, f"{label}.allocation[{ordinal}] raw owner")
                C.equal(dimension, N - profiles[matrix_ordinal]["selected_scalars"], f"{label}.allocation[{ordinal}] raw dimension")
                C.close(energy, profiles[matrix_ordinal]["bulk_energy"], f"{label}.allocation[{ordinal}] raw energy")
            else:
                C.true(re.fullmatch(r"expert_\d{2}\..+\.(both_axis_[01]|only_up|only_down)", name) is not None, f"{label}.allocation[{ordinal}] XKLT name")
        C.equal(sum(allocation_payload), side["payload_bits"], f"{label}.allocation payload closure")
        C.close(allocation_distortion, row["bulk_ideal_distortion_sse"], f"{label}.allocation distortion closure", rel=5e-13, absolute=1e-13)
        C.equal(side["xklt_angle_bits"], sum(angle_by_expert) * ANGLE_BITS, f"{label}.angle charge")
        for expert in range(EXPERTS):
            expected_expert_side = EXPERT_HEADER_BITS + ROLES * MATRIX_DESCRIPTOR_BITS
            expected_expert_side += component_by_expert[expert] * RESIDUAL_DIRECTORY_BITS
            expected_expert_side += angle_by_expert[expert] * ANGLE_BITS
            expected_expert_side += expected_tail_side(profiles[2 * expert], mode)
            expected_expert_side += expected_tail_side(profiles[2 * expert + 1], mode)
            C.equal(side["expert_side_bits"][expert], expected_expert_side, f"{label}.expert[{expert}] side derivation")
    else:
        if raw_geometry == "raw":
            C.equal(side["xklt_angle_bits"], 0, f"{label}.raw angle bits")
        else:
            C.true(isinstance(side["xklt_angle_bits"], int) and 0 <= side["xklt_angle_bits"] <= EXPERTS * ANGLE_BITS and side["xklt_angle_bits"] % ANGLE_BITS == 0, f"{label}.diagnostic angle range")

    bulk_sse = finite(row["bulk_ideal_distortion_sse"], f"{label}.bulk_sse")
    total_sse = finite(row["total_distortion_sse"], f"{label}.total_sse")
    C.true(bulk_sse > 0.0 and total_sse > 0.0, f"{label} positive distortion")
    C.close(total_sse, tail_sse + bulk_sse, f"{label}.distortion sum", rel=3e-13, absolute=2e-13)
    relative_mse = total_sse / total_energy
    f_value = relative_mse * 2.0 ** (2.0 * physical_rate)
    s_value = -0.5 * math.log2(f_value)
    C.close(row["ideal_relative_mse"], relative_mse, f"{label}.relative MSE")
    C.close(row["F"], f_value, f"{label}.F")
    C.close(row["s_bpw"], s_value, f"{label}.s")
    C.close(row["gaussian_limit_mse"], 2.0 ** (-2.0 * physical_rate), f"{label}.Gaussian limit")
    C.close(row["target_mse"], TARGET_F * 2.0 ** (-2.0 * physical_rate), f"{label}.target MSE")
    C.equal(row["passes_absolute_F"], f_value <= TARGET_F, f"{label}.absolute pass")

    read = exact_keys(row["read_ledger"], {
        "reference_one_sixth_container_bytes", "common_prefix_bytes", "experts",
        "maximum_cold_logical_amplification", "maximum_cold_page_amplification", "below_2x",
    }, f"{label}.read_ledger")
    reference_bytes = capacity / EXPERTS
    common_bytes = COMMON_BITS // 8
    C.close(read["reference_one_sixth_container_bytes"], reference_bytes, f"{label}.read reference", rel=0.0, absolute=0.0)
    C.equal(read["common_prefix_bytes"], common_bytes, f"{label}.common bytes")
    experts = read["experts"]
    C.true(isinstance(experts, list) and len(experts) == EXPERTS, f"{label}.expert read rows")
    offset = common_bytes
    logical_values, page_values, derived_sides = [], [], []
    common_pages = set(range((common_bytes + PAGE_BYTES - 1) // PAGE_BYTES))
    for expert, expert_read in enumerate(experts):
        exact_keys(expert_read, {
            "expert_ordinal", "frame_offset_bytes", "frame_bytes", "frame_end_pad_bits",
            "residual_payload_bits", "cold_logical_bytes", "cold_logical_amplification",
            "cold_page_bytes", "cold_page_amplification",
        }, f"{label}.read.expert[{expert}]")
        C.equal(expert_read["expert_ordinal"], expert, f"{label}.read.expert[{expert}].ordinal")
        C.equal(expert_read["frame_offset_bytes"], offset, f"{label}.read.expert[{expert}].offset")
        frame_bytes = expert_read["frame_bytes"]
        pad = expert_read["frame_end_pad_bits"]
        payload = expert_read["residual_payload_bits"]
        C.true(all(isinstance(x, int) and not isinstance(x, bool) for x in (frame_bytes, pad, payload)), f"{label}.read.expert[{expert}] integer fields")
        C.true(frame_bytes > 0 and 0 <= pad <= 7 and payload >= 0, f"{label}.read.expert[{expert}] field ranges")
        if retained:
            C.equal(payload, allocation_payload[expert], f"{label}.read.expert[{expert}] allocation bits")
        unpadded = side["expert_side_bits"][expert] + payload
        C.equal(pad, (-unpadded) % 8, f"{label}.read.expert[{expert}] pad")
        C.equal(frame_bytes * 8, unpadded + pad, f"{label}.read.expert[{expert}] frame closure")
        derived_sides.append(frame_bytes * 8 - pad - payload)
        logical = common_bytes + frame_bytes
        start, end = offset, offset + frame_bytes
        pages = set(range(start // PAGE_BYTES, (end - 1) // PAGE_BYTES + 1))
        page_bytes = len(common_pages | pages) * PAGE_BYTES
        C.equal(expert_read["cold_logical_bytes"], logical, f"{label}.read.expert[{expert}] logical bytes")
        C.close(expert_read["cold_logical_amplification"], logical / reference_bytes, f"{label}.read.expert[{expert}] logical amplification", rel=0.0, absolute=0.0)
        C.equal(expert_read["cold_page_bytes"], page_bytes, f"{label}.read.expert[{expert}] page bytes")
        C.close(expert_read["cold_page_amplification"], page_bytes / reference_bytes, f"{label}.read.expert[{expert}] page amplification", rel=0.0, absolute=0.0)
        logical_values.append(logical / reference_bytes)
        page_values.append(page_bytes / reference_bytes)
        offset = end
    C.equal(derived_sides, side["expert_side_bits"], f"{label}.expert side reconstructed from frames")
    C.equal(sum(e["residual_payload_bits"] for e in experts), side["payload_bits"], f"{label}.read payload closure")
    C.equal(sum(e["frame_end_pad_bits"] for e in experts), side["actual_end_pad_bits"], f"{label}.actual pad sum")
    closure = COMMON_BITS + sum(e["frame_bytes"] * 8 for e in experts)
    C.equal(side["trailer_bits"], physical_bits - closure, f"{label}.trailer")
    C.equal(side["bit_closure"], physical_bits, f"{label}.bit closure")
    C.equal(side["bit_closure"], closure + side["trailer_bits"], f"{label}.container closure")
    C.close(read["maximum_cold_logical_amplification"], max(logical_values), f"{label}.maximum logical", rel=0.0, absolute=0.0)
    C.close(read["maximum_cold_page_amplification"], max(page_values), f"{label}.maximum page", rel=0.0, absolute=0.0)
    below = all(x < 2.0 for x in logical_values) and all(x < 2.0 for x in page_values)
    C.equal(read["below_2x"], below, f"{label}.strict read flag")
    if geometry_key in GEOMETRIES:
        C.equal(below, True, f"{label}.retained row strictly below 2x")


def iter_retained(search: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for rate in RATES:
        for mode in MODES:
            for geometry in GEOMETRIES:
                yield search[f"{rate:.2f}"][mode][geometry]


def iter_all_scored(search: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for rate in RATES:
        for mode in MODES:
            node = search[f"{rate:.2f}"][mode]
            for geometry in (
                "raw_adaptive", "raw_uniform_best", "support_xklt_uniform",
                "raw_uniform_global_diagnostic", "support_xklt_uniform_global_diagnostic",
            ):
                yield node[geometry]


def verify_search(search: Any, *, panel_label: str, moments: list[dict[str, Any]], total_energy: float, label: str) -> None:
    C.true(isinstance(search, dict), f"{label} object")
    # The sealed writer sorts object keys; semantic loop ordering is captured
    # by the calibrated-row array, not by JSON object insertion order.
    C.equal(set(search), {f"{r:.2f}" for r in RATES}, f"{label} exact rates")
    for rate in RATES:
        rate_key = f"{rate:.2f}"
        rate_rows = search[rate_key]
        C.equal(set(rate_rows), set(MODES), f"{label}.{rate_key} exact modes")
        for mode in MODES:
            node = exact_keys(rate_rows[mode], {
                "raw_adaptive", "raw_uniform_best", "support_xklt_uniform",
                "raw_uniform_global_diagnostic", "support_xklt_uniform_global_diagnostic",
                "eligible_profile_count", "coordinate_passes_max_used", "read_valid_selection_ledger",
            }, f"{label}.{rate_key}.{mode}")
            eligible = len(PROFILES) if mode != "zero_tail_error" else sum(p["levels"] == 1 for p in PROFILES)
            C.equal(node["eligible_profile_count"], eligible, f"{label}.{rate_key}.{mode}.eligible count")
            passes = node["coordinate_passes_max_used"]
            C.true(isinstance(passes, int) and not isinstance(passes, bool) and 1 <= passes <= 4, f"{label}.{rate_key}.{mode}.pass count")
            ledger = exact_keys(node["read_valid_selection_ledger"], {
                "uniform_rows_evaluated", "support_xklt_rows_evaluated",
                "coordinate_trial_rows_evaluated", "seed_count",
                "every_uniform_profile_evaluated", "every_support_xklt_profile_evaluated",
                "every_coordinate_trial_profile_evaluated", "retained_rows_below_2x",
            }, f"{label}.{rate_key}.{mode}.selection ledger")
            C.equal(ledger["uniform_rows_evaluated"], eligible, f"{label}.{rate_key}.{mode} uniform coverage")
            C.equal(ledger["support_xklt_rows_evaluated"], eligible, f"{label}.{rate_key}.{mode} XKLT coverage")
            seeds = ledger["seed_count"]
            trials = ledger["coordinate_trial_rows_evaluated"]
            C.true(isinstance(seeds, int) and not isinstance(seeds, bool) and seeds in (1, 2), f"{label}.{rate_key}.{mode} seed count")
            C.true(isinstance(trials, int) and not isinstance(trials, bool), f"{label}.{rate_key}.{mode} trial count type")
            C.true(trials % (MATRICES * eligible) == 0, f"{label}.{rate_key}.{mode} complete coordinate sweeps")
            C.true(seeds * MATRICES * eligible <= trials <= seeds * 4 * MATRICES * eligible, f"{label}.{rate_key}.{mode} trial bounds")
            for key in ("every_uniform_profile_evaluated", "every_support_xklt_profile_evaluated", "every_coordinate_trial_profile_evaluated", "retained_rows_below_2x"):
                C.equal(ledger[key], True, f"{label}.{rate_key}.{mode}.{key}")
            for geometry_key in (
                "raw_adaptive", "raw_uniform_best", "support_xklt_uniform",
                "raw_uniform_global_diagnostic", "support_xklt_uniform_global_diagnostic",
            ):
                verify_score(
                    node[geometry_key], rate=rate, mode=mode, geometry_key=geometry_key,
                    expected_panel=panel_label, moments=moments, total_energy=total_energy,
                    retained=geometry_key in GEOMETRIES,
                    label=f"{label}.{rate_key}.{mode}.{geometry_key}",
                )


def verify_calibration(result: dict[str, Any], control_searches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = result["calibrated_rows"]
    C.true(isinstance(rows, list) and len(rows) == len(RATES) * len(MODES) * len(GEOMETRIES), "18 calibrated rows")
    expected_rows: list[dict[str, Any]] = []
    index = 0
    for rate in RATES:
        rate_key = f"{rate:.2f}"
        for mode in MODES:
            for geometry in GEOMETRIES:
                row = exact_keys(rows[index], CALIBRATED_KEYS, f"calibrated[{index}]")
                source = result["qwen_search"][rate_key][mode][geometry]
                controls = [search[rate_key][mode][geometry] for search in control_searches]
                qwen_f = source["F"]
                qwen_s = source["s_bpw"]
                control_f = [x["F"] for x in controls]
                control_s = [x["s_bpw"] for x in controls]
                control_mean = sum(control_s) / 4.0
                control_std = statistics.stdev(control_s)
                excess = qwen_s - control_mean
                calibrated_f = 2.0 ** (-2.0 * excess)
                fraction = excess / TARGET_S
                C.equal(row["rate"], rate, f"calibrated[{index}].rate")
                C.equal(row["mode"], mode, f"calibrated[{index}].mode")
                C.equal(row["geometry"], geometry, f"calibrated[{index}].geometry")
                C.close(row["qwen_F"], qwen_f, f"calibrated[{index}].qwen_F", rel=0.0, absolute=0.0)
                C.close(row["qwen_s_bpw"], qwen_s, f"calibrated[{index}].qwen_s", rel=0.0, absolute=0.0)
                C.equal(row["control_F"], control_f, f"calibrated[{index}].control_F")
                C.equal(row["control_s_bpw"], control_s, f"calibrated[{index}].control_s")
                C.close(row["control_mean_s_bpw"], control_mean, f"calibrated[{index}].control mean", rel=2e-14, absolute=2e-16)
                C.close(row["control_sample_std_s_bpw"], control_std, f"calibrated[{index}].control std", rel=2e-8, absolute=2e-16)
                C.close(row["qwen_excess_s_bpw"], excess, f"calibrated[{index}].excess", rel=2e-8, absolute=2e-16)
                C.close(row["calibrated_F"], calibrated_f, f"calibrated[{index}].calibrated F", rel=2e-13, absolute=2e-15)
                C.close(row["fraction_of_required_s"], fraction, f"calibrated[{index}].fraction required", rel=2e-8, absolute=2e-16)
                C.equal(row["passes_calibrated_F"], calibrated_f <= TARGET_F, f"calibrated[{index}].calibrated pass")
                C.equal(row["passes_absolute_F"], qwen_f <= TARGET_F, f"calibrated[{index}].absolute pass")
                C.equal(row["below_2x"], True, f"calibrated[{index}].strict read validity")
                C.true(source["read_ledger"]["below_2x"] and all(x["read_ledger"]["below_2x"] for x in controls), f"calibrated[{index}] source/control read gate")
                expected_rows.append({
                    **row,
                    "control_mean_s_bpw": control_mean,
                    "control_sample_std_s_bpw": control_std,
                    "qwen_excess_s_bpw": excess,
                    "calibrated_F": calibrated_f,
                    "fraction_of_required_s": fraction,
                })
                index += 1
    return expected_rows


def independent_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def joint(row: dict[str, Any]) -> float:
        return min(float(row["qwen_s_bpw"]), float(row["qwen_excess_s_bpw"]))

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (joint(row), -float(row["rate"]), str(row["mode"]), str(row["geometry"]))

    read_valid = [row for row in rows if row["below_2x"]]
    optimistic = [row for row in read_valid if row["mode"] in ("free_lloyd", "zero_tail_error")]
    finite_rows = [row for row in read_valid if row["mode"] == "finite_fp16"]
    best_optimistic = max(optimistic, key=key)
    best_finite = max(finite_rows, key=key)
    optimistic_m = joint(best_optimistic)
    finite_joint = joint(best_finite)
    finite_absolute = float(best_finite["qwen_s_bpw"])
    finite_calibrated = float(best_finite["qwen_excess_s_bpw"])
    C.true(finite_joint <= optimistic_m + DECISION_EPSILON_S, "finite score inside optimistic envelope")
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
    numeric_boundary = any(
        distance <= NUMERIC_BOUNDARY_GUARD_S
        for pair in boundary_distances.values() for distance in pair.values()
    )
    would_promote = finite_absolute >= TARGET_S and finite_calibrated >= TARGET_S
    C.true(not would_promote or optimistic_m >= TARGET_S - DECISION_EPSILON_S, "promotion/optimistic consistency")
    if numeric_boundary:
        status, warranted, early = "HOLD_NUMERIC_BOUNDARY", False, False
    elif would_promote:
        status, warranted, early = "FINITE_CODEC_WARRANTED", True, False
    elif optimistic_m < KILL_THRESHOLD_S:
        status, warranted, early = "EARLY_KILL_FAR_SHORT", False, True
    elif optimistic_m < TARGET_S:
        status, warranted, early = "HOLD_OPTIMISTIC_NEAR_BOUNDARY", False, False
    else:
        status, warranted, early = "OPTIMISTIC_SURVIVOR", False, False
    return {
        "status": status, "target_F": TARGET_F, "required_s_bpw": TARGET_S,
        "optimistic_kill_guard_s_bpw": KILL_GUARD_S,
        "optimistic_kill_threshold_s_bpw": KILL_THRESHOLD_S,
        "numeric_boundary_guard_s_bpw": NUMERIC_BOUNDARY_GUARD_S,
        "optimistic_m_s_bpw": optimistic_m,
        "best_optimistic_envelope": best_optimistic,
        "finite_best_joint_s_bpw": finite_joint,
        "finite_best_row": best_finite,
        "boundary_values_s_bpw": boundary_values,
        "boundary_distances_s_bpw": boundary_distances,
        "finite_residual_codec_warranted": warranted,
        "early_kill": early,
    }


def verify_external_evidence(auth: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected in KNOWN_FILE_HASHES.items():
        verify_file(path, expected)
    _, protocol = load_json(PRODUCER / "protocol_lock.json")
    _, bindings = load_json(PRODUCER / "source_bindings.json")
    _, source_manifest = load_json(SOURCE_AUDIT / "audit_manifest.json")
    _, source_receipt = load_json(SOURCE_AUDIT / "audit_receipt.json")
    _, source_replay = load_json(SOURCE_AUDIT / "replay_receipt.json")
    _, runtime_manifest = load_json(RUNTIME_AUDIT / "audit_manifest.json")
    _, runtime_audit_receipt = load_json(RUNTIME_AUDIT / "audit_receipt.json")
    _, runtime_receipt = load_json(RUNTIME_AUDIT / "runtime_receipt.json")
    verify_internal(source_manifest, "audit_manifest_sha256", "afe6c6f43a0939861973e4362a5ffa8389f0ac77941dfe8b8e9a5b98a6e1d483", "source audit manifest")
    verify_internal(source_receipt, "audit_receipt_sha256", "b06d584ec0e5c8fd69d8f6b609f72631b69dc1feee0af467018e76607384f38b", "source audit receipt")
    verify_internal(source_replay, "replay_receipt_sha256", "fa0fdeba885539f2fb476850ce94c62ec911af8df480f127f7d3bbf90c2574d3", "source replay receipt")
    verify_internal(runtime_manifest, "audit_manifest_sha256", "f420b221d73dbe87c1ba8517fadd575997c0379146c55a3baaecc4ec42b76420", "runtime audit manifest")
    verify_internal(runtime_audit_receipt, "audit_receipt_sha256", "f19c47de87e1626b9bdf36e6d90e2a31e9a5006134d79095a113043a4e4e3ea2", "runtime audit receipt")
    verify_internal(runtime_receipt, "runtime_receipt_sha256", "de096442f734689ed981c0b3870a335255c9914d9d818b56aca1ac956add4d50", "runtime receipt")
    C.equal(source_manifest["schema"], "lossy-tail-v8-independent-source-audit-manifest-v1", "source manifest schema")
    C.equal(source_manifest["status"], "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET", "source manifest status")
    C.equal(source_receipt["schema"], "lossy-tail-v8-independent-source-audit-receipt-v1", "source receipt schema")
    C.equal(source_receipt["status"], "PASS_V8_INDEPENDENT_SOURCE_AUDIT", "source receipt status")
    C.equal(runtime_manifest["schema"], "lossy-tail-v8-independent-runtime-audit-manifest-v1", "runtime manifest schema")
    C.equal(runtime_manifest["status"], "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET", "runtime manifest status")
    C.equal(runtime_audit_receipt["schema"], "lossy-tail-v8-independent-runtime-audit-receipt-v1", "runtime receipt-audit schema")
    C.equal(runtime_audit_receipt["status"], "PASS_V8_INDEPENDENT_RUNTIME_AUDIT", "runtime receipt-audit status")
    C.equal(runtime_receipt["schema"], "lossy-tail-v8-source-free-runtime-receipt-v1", "runtime evidence schema")
    C.equal(runtime_receipt["status"], "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT", "runtime evidence pre-audit status")
    for label, ledger, keys in (
        ("source audit", source_receipt["access_ledger"], ("model_payload_files_opened", "cupy_imports", "cuda_initializations", "gpu_jobs", "production_runs")),
        ("runtime audit", runtime_audit_receipt["access_ledger"], ("model_payload_files_opened", "production_result_files_opened", "gpu_jobs", "production_runs")),
        ("runtime calibration", runtime_receipt["access_ledger"], ("model_or_qwen_paths_supplied", "model_or_qwen_paths_opened", "payload_files_opened", "production_results_opened")),
    ):
        for key in keys:
            C.equal(ledger.get(key), 0, f"{label} zero-access {key}")
    C.equal(auth["source_audit"]["manifest_file_sha256"], KNOWN_FILE_HASHES[SOURCE_AUDIT / "audit_manifest.json"], "authorization source manifest")
    C.equal(auth["source_audit"]["receipt_file_sha256"], KNOWN_FILE_HASHES[SOURCE_AUDIT / "audit_receipt.json"], "authorization source receipt")
    C.equal(auth["source_audit"]["receipt_internal_sha256"], source_receipt["audit_receipt_sha256"], "authorization source internal")
    C.equal(auth["runtime_receipt"]["file_sha256"], KNOWN_FILE_HASHES[RUNTIME_AUDIT / "runtime_receipt.json"], "authorization runtime receipt")
    C.equal(auth["runtime_receipt"]["internal_sha256"], runtime_receipt["runtime_receipt_sha256"], "authorization runtime internal")
    C.equal(auth["runtime_audit"]["manifest_file_sha256"], KNOWN_FILE_HASHES[RUNTIME_AUDIT / "audit_manifest.json"], "authorization runtime-audit manifest")
    C.equal(auth["runtime_audit"]["receipt_file_sha256"], KNOWN_FILE_HASHES[RUNTIME_AUDIT / "audit_receipt.json"], "authorization runtime-audit receipt")
    C.equal(auth["runtime_audit"]["receipt_internal_sha256"], runtime_audit_receipt["audit_receipt_sha256"], "authorization runtime-audit internal")
    C.equal(runtime_audit_receipt["audited_runtime_receipt"]["file_sha256"], auth["runtime_receipt"]["file_sha256"], "runtime audit targets receipt file")
    C.equal(runtime_audit_receipt["audited_runtime_receipt"]["internal_sha256"], auth["runtime_receipt"]["internal_sha256"], "runtime audit targets receipt internal")
    C.equal(result["runtime"]["runtime_probe_aggregate_sha256"], runtime_receipt["runtime_probe"]["probe_aggregate_sha256"], "production replay aggregate")
    C.equal(result["runtime"]["runtime_tuple"], runtime_receipt["runtime_probe"]["runtime_tuple"], "production runtime tuple equals audited receipt")
    C.equal(auth["execution"]["runtime_tuple"], runtime_receipt["runtime_probe"]["runtime_tuple"], "authorization runtime tuple equals audited receipt")
    C.equal(bindings["source_directory_at_execution"], auth["source"]["path"], "source path binds source metadata")
    C.equal(protocol["claim_boundary"], result["decision"]["claim_boundary"], "claim boundary binds protocol")
    return protocol, bindings, runtime_receipt


def verify_authorization(auth: dict[str, Any], result: dict[str, Any]) -> None:
    exact_keys(auth, {
        "schema", "status", "authorization_path", "authorization_nonce", "action",
        "stage", "source", "output", "source_audit", "runtime_receipt",
        "runtime_audit", "execution", "filesystem", "fixed_scientific_arguments",
        "authorization_sha256",
    }, "authorization")
    C.equal(auth["schema"], "lossy-tail-v8-one-shot-production-authorization-v1", "authorization schema")
    C.equal(auth["status"], "AUTHORIZED_ONCE_AFTER_INDEPENDENT_SOURCE_AND_RUNTIME_AUDITS", "authorization status")
    C.equal(auth["action"], "CREATE_NEW_RUN_ROOT_AND_RESULT_JSON", "authorization action")
    C.true(isinstance(auth["authorization_nonce"], str) and re.fullmatch(r"[0-9a-f]{64}", auth["authorization_nonce"]) is not None, "authorization nonce")
    verify_internal(auth, "authorization_sha256", AUTHORIZATION_INTERNAL_SHA256, "authorization")
    C.equal(auth["fixed_scientific_arguments"], {"control_replicates": 4, "maximum_coordinate_passes": 4}, "fixed scientific arguments")
    C.equal(auth["stage"], {
        "path": auth["stage"]["path"],
        "launch_manifest_file_sha256": KNOWN_FILE_HASHES[PRODUCER / "launch_manifest.json"],
        "launch_manifest_internal_stage_member_count": 11,
    }, "authorized stage")
    C.equal(auth["source"]["bindings_file_sha256"], KNOWN_FILE_HASHES[PRODUCER / "source_bindings.json"], "authorized bindings")
    C.equal(auth["runtime_receipt"]["runtime_contract_file_sha256"], KNOWN_FILE_HASHES[PRODUCER / "runtime_contract.json"], "authorized runtime contract")
    C.equal(auth["execution"]["cuda_visible_devices"], "0", "authorized CUDA visibility")
    C.equal(auth["execution"]["python_executable"], "/workspace/int2-cupy-venv/bin/python", "authorized interpreter")
    C.equal(auth["execution"]["raw_launcher_path"], posixpath.join(auth["stage"]["path"], "preflight_launch.py"), "authorized raw launcher")
    C.equal(auth["output"]["result_path"], posixpath.join(auth["output"]["run_root"], "result.json"), "authorized result path")
    C.equal(auth["filesystem"]["mountinfo_path"], "/proc/self/mountinfo", "mountinfo path")
    sha(auth["filesystem"]["mountinfo_file_sha256"], "authorized mountinfo hash")
    identity_rows = auth["filesystem"]["identities"]
    labels = [
        "stage", "source", "output_existing_parent", "authorization_parent",
        "source_audit_manifest", "source_audit_receipt", "runtime_receipt",
        "runtime_audit_manifest", "runtime_audit_receipt",
    ]
    C.true(isinstance(identity_rows, list) and len(identity_rows) == len(labels), "nine filesystem identities")
    C.equal([row.get("label") for row in identity_rows], labels, "filesystem identity order")
    C.equal(len({(row.get("st_dev"), row.get("st_ino")) for row in identity_rows}), len(identity_rows), "filesystem identities unique")
    expected_paths = {
        "stage": auth["stage"]["path"], "source": auth["source"]["path"],
        "output_existing_parent": posixpath.dirname(auth["output"]["run_root"]),
        "authorization_parent": posixpath.dirname(auth["authorization_path"]),
        "source_audit_manifest": auth["source_audit"]["manifest_path"],
        "source_audit_receipt": auth["source_audit"]["receipt_path"],
        "runtime_receipt": auth["runtime_receipt"]["path"],
        "runtime_audit_manifest": auth["runtime_audit"]["manifest_path"],
        "runtime_audit_receipt": auth["runtime_audit"]["receipt_path"],
    }
    for ordinal, row in enumerate(identity_rows):
        exact_keys(row, {"label", "path", "st_dev", "st_ino", "mount_id"}, f"filesystem identity[{ordinal}]")
        C.equal(row["path"], expected_paths[row["label"]], f"filesystem identity[{ordinal}] path")
        C.true(all(isinstance(row[key], int) and not isinstance(row[key], bool) and row[key] > 0 for key in ("st_dev", "st_ino", "mount_id")), f"filesystem identity[{ordinal}] positive IDs")

    protected = [PurePosixPath(auth["stage"]["path"]), PurePosixPath(auth["source"]["path"]), PurePosixPath(expected_paths["output_existing_parent"])]
    for i, left in enumerate(protected):
        for right in protected[i + 1:]:
            C.true(left not in right.parents and right not in left.parents and left != right, "protected roots pairwise disjoint")
    # The identity table binds the authorization *parent* directory, whereas
    # the disjointness contract applies to the authorization file and its
    # immediate parent.  Do not accidentally promote that row to its own
    # parent (for this run, broad ``/var/tmp``), which is not the frozen rule.
    evidence = [
        PurePosixPath(auth["authorization_path"]),
        PurePosixPath(auth["source_audit"]["manifest_path"]),
        PurePosixPath(auth["source_audit"]["receipt_path"]),
        PurePosixPath(auth["runtime_receipt"]["path"]),
        PurePosixPath(auth["runtime_audit"]["manifest_path"]),
        PurePosixPath(auth["runtime_audit"]["receipt_path"]),
    ]
    for path in evidence:
        parent = path.parent
        for root in protected:
            C.true(path != root and path not in root.parents and root not in path.parents, "evidence path disjoint")
            C.true(parent != root and parent not in root.parents and root not in parent.parents, "evidence parent disjoint")

    result_auth = exact_keys(result["authorization"], {"path", "file_sha256", "internal_sha256", "nonce", "action"}, "result authorization binding")
    C.equal(result_auth, {
        "path": auth["authorization_path"], "file_sha256": AUTHORIZATION_FILE_SHA256,
        "internal_sha256": AUTHORIZATION_INTERNAL_SHA256,
        "nonce": auth["authorization_nonce"], "action": auth["action"],
    }, "result binds exact authorization")
    C.equal(result["launch_manifest"], {"path": posixpath.join(auth["stage"]["path"], "launch_manifest.json"), "sha256": KNOWN_FILE_HASHES[PRODUCER / "launch_manifest.json"]}, "result launch manifest")
    C.equal(result["protocol"], {"path": posixpath.join(auth["stage"]["path"], "protocol_lock.json"), "sha256": KNOWN_FILE_HASHES[PRODUCER / "protocol_lock.json"]}, "result protocol")
    C.equal(result["repair_lock"], {"path": posixpath.join(auth["stage"]["path"], "repair_lock.json"), "sha256": KNOWN_FILE_HASHES[PRODUCER / "repair_lock.json"]}, "result repair lock")
    C.equal(result["runtime_contract"], {"path": posixpath.join(auth["stage"]["path"], "runtime_contract.json"), "sha256": KNOWN_FILE_HASHES[PRODUCER / "runtime_contract.json"]}, "result runtime contract")
    C.equal(result["runtime_receipt"], {"path": auth["runtime_receipt"]["path"], "file_sha256": auth["runtime_receipt"]["file_sha256"], "internal_sha256": auth["runtime_receipt"]["internal_sha256"]}, "result runtime receipt")
    C.equal(result["bindings"], {"path": posixpath.join(auth["stage"]["path"], "source_bindings.json"), "sha256": KNOWN_FILE_HASHES[PRODUCER / "source_bindings.json"]}, "result bindings")
    C.equal(result["oracle_bootstrap_sha256"], KNOWN_FILE_HASHES[PRODUCER / "lossy_tail_oracle.py"], "result bootstrap")
    C.equal(result["scientific_core_sha256"], KNOWN_FILE_HASHES[PRODUCER / "lossy_tail_core.py"], "result core")
    C.equal(result["mountinfo_sha256"], auth["filesystem"]["mountinfo_file_sha256"], "result mountinfo")

    cap = exact_keys(result["child_capability"], {
        "sha256", "parent_pid", "child_pid", "preflight_memfd_sha256",
        "preflight_memfd_st_dev", "preflight_memfd_st_ino", "preflight_memfd_seals",
        "output_parent_st_dev", "output_parent_st_ino",
    }, "child capability")
    sha(cap["sha256"], "child capability hash")
    C.true(isinstance(cap["parent_pid"], int) and cap["parent_pid"] > 0, "parent PID")
    C.true(isinstance(cap["child_pid"], int) and cap["child_pid"] > 0 and cap["child_pid"] != cap["parent_pid"], "child PID")
    C.equal(cap["preflight_memfd_sha256"], KNOWN_FILE_HASHES[PRODUCER / "preflight_launch.py"], "sealed preflight bytes")
    C.equal(cap["preflight_memfd_seals"], 15, "complete Linux memfd seal mask")
    output_identity = identity_rows[labels.index("output_existing_parent")]
    C.equal(cap["output_parent_st_dev"], output_identity["st_dev"], "capability output dev")
    C.equal(cap["output_parent_st_ino"], output_identity["st_ino"], "capability output inode")
    C.true(cap["preflight_memfd_st_dev"] > 0 and cap["preflight_memfd_st_ino"] > 0, "capability memfd identity")


def verify_runtime(result: dict[str, Any], auth: dict[str, Any]) -> None:
    runtime = exact_keys(result["runtime"], {
        "argv", "cwd", "pid", "python", "platform", "runtime_tuple",
        "runtime_probe_aggregate_sha256", "started_utc_epoch", "ended_utc_epoch", "elapsed_seconds",
    }, "production runtime")
    C.equal(runtime["pid"], result["child_capability"]["child_pid"], "runtime PID equals capability child")
    C.equal(runtime["runtime_tuple"], auth["execution"]["runtime_tuple"], "runtime tuple authorization")
    sha(runtime["runtime_probe_aggregate_sha256"], "runtime probe aggregate")
    started = finite(runtime["started_utc_epoch"], "runtime start")
    ended = finite(runtime["ended_utc_epoch"], "runtime end")
    elapsed = finite(runtime["elapsed_seconds"], "runtime elapsed")
    C.true(ended >= started and elapsed > 0.0, "runtime chronological")
    C.true(isinstance(runtime["cwd"], str) and runtime["cwd"], "runtime cwd")
    C.true(isinstance(runtime["python"], str) and runtime["python"], "runtime Python string")
    C.true(isinstance(runtime["platform"], str) and runtime["platform"], "runtime platform string")
    argv = runtime["argv"]
    C.true(isinstance(argv, list) and len(argv) == 23, "scientific argv cardinality")
    flags = [
        "--bindings", "--protocol", "--repair-lock", "--runtime-contract",
        "--authorization-contract", "--launch-manifest", "--launch-manifest-sha256",
        "--authorization", "--authorization-sha256", "--control-replicates",
        "--maximum-coordinate-passes",
    ]
    C.equal(argv[1::2], flags, "scientific argv flag grammar")
    C.equal(argv[0], posixpath.join(auth["stage"]["path"], "lossy_tail_oracle.py"), "scientific argv bootstrap")
    C.equal(argv[2], posixpath.join(auth["stage"]["path"], "source_bindings.json"), "scientific argv bindings")
    C.equal(argv[14], KNOWN_FILE_HASHES[PRODUCER / "launch_manifest.json"], "scientific argv manifest hash")
    C.equal(argv[16], auth["authorization_path"], "scientific argv authorization path")
    C.equal(argv[18], AUTHORIZATION_FILE_SHA256, "scientific argv authorization hash")
    C.equal(argv[20], "4", "scientific argv controls")
    C.equal(argv[22], "4", "scientific argv passes")


def verify_panel_and_controls(result: dict[str, Any], bindings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panel = exact_keys(result["panel"], {
        "experts", "roles", "matrices", "values_per_matrix", "panel_values",
        "total_energy", "moments", "source_receipts", "candidate_count",
        "candidate_ledger_sha256", "memory_release",
    }, "Qwen panel")
    C.equal((panel["experts"], panel["roles"], panel["matrices"]), (EXPERTS, ROLES, MATRICES), "panel dimensions")
    C.equal((panel["values_per_matrix"], panel["panel_values"]), (N, PANEL_N), "panel value counts")
    qwen_moments = panel["moments"]
    verify_moments(qwen_moments, "Qwen moments", control=False)
    total_energy = sum(row["energy"] for row in qwen_moments)
    C.close(panel["total_energy"], total_energy, "Qwen panel energy")
    C.equal(panel["candidate_count"], MATRICES * len(PROFILES), "Qwen candidate count")
    sha(panel["candidate_ledger_sha256"], "Qwen candidate ledger")
    verify_memory(panel["memory_release"], "Qwen memory release")
    receipts = panel["source_receipts"]
    C.true(isinstance(receipts, list) and len(receipts) == MATRICES, "12 source receipts")
    C.equal(len(bindings["files"]), MATRICES, "12 frozen bindings")
    for ordinal, (receipt, bound) in enumerate(zip(receipts, bindings["files"])):
        exact_keys(receipt, {"expert", "role", "name", "sha256", "path", "bytes", "observed_sha256"}, f"source receipt[{ordinal}]")
        for key in ("expert", "role", "name", "sha256"):
            C.equal(receipt[key], bound[key], f"source receipt[{ordinal}].{key}")
        C.equal(receipt["bytes"], 2 * N, f"source receipt[{ordinal}].bytes")
        C.equal(receipt["observed_sha256"], bound["sha256"], f"source receipt[{ordinal}].observed hash")
        C.equal(receipt["path"], posixpath.join(bindings["source_directory_at_execution"], bound["name"]), f"source receipt[{ordinal}].path")

    verify_search(result["qwen_search"], panel_label="qwen_aux_l15_up_down", moments=qwen_moments, total_energy=total_energy, label="Qwen search")

    controls = exact_keys(result["matched_gaussian_controls"], {"replicates", "panels", "searches", "post_fp32_moment_mismatch"}, "matched controls")
    C.equal(controls["replicates"], 4, "control replicate count")
    C.true(isinstance(controls["panels"], list) and len(controls["panels"]) == 4, "four control panels")
    C.true(isinstance(controls["searches"], list) and len(controls["searches"]) == 4, "four control searches")
    independently_derived_mismatches: list[dict[str, Any]] = []
    for replica, (control_panel, search) in enumerate(zip(controls["panels"], controls["searches"])):
        exact_keys(control_panel, {"replica", "moments", "total_energy", "candidate_ledger_sha256", "candidate_count", "memory_release"}, f"control panel[{replica}]")
        C.equal(control_panel["replica"], replica, f"control panel[{replica}].replica")
        control_moments = control_panel["moments"]
        independently_derived_mismatches.extend(verify_moments(control_moments, f"control[{replica}] moments", control=True, qwen_moments=qwen_moments, replica=replica))
        control_energy = sum(row["energy"] for row in control_moments)
        C.close(control_panel["total_energy"], control_energy, f"control panel[{replica}] energy")
        C.equal(control_panel["candidate_count"], MATRICES * len(PROFILES), f"control panel[{replica}] candidate count")
        sha(control_panel["candidate_ledger_sha256"], f"control panel[{replica}] candidate ledger")
        verify_memory(control_panel["memory_release"], f"control[{replica}] memory release")
        verify_search(search, panel_label=f"gaussian_control_{replica}", moments=control_moments, total_energy=control_energy, label=f"control[{replica}] search")
    mismatch = exact_keys(controls["post_fp32_moment_mismatch"], {
        "cell_count", "cells", "maximum_mean_normalized_mismatch",
        "maximum_variance_normalized_mismatch", "all_cells_within_tolerance",
    }, "control mismatch summary")
    C.equal(mismatch["cell_count"], 48, "control mismatch cell count")
    C.true(isinstance(mismatch["cells"], list) and len(mismatch["cells"]) == 48, "control mismatch rows")
    for ordinal, (reported, derived) in enumerate(zip(mismatch["cells"], independently_derived_mismatches)):
        C.equal(set(reported), set(derived), f"control mismatch[{ordinal}] keys")
        for key, value in derived.items():
            if isinstance(value, float):
                C.close(reported[key], value, f"control mismatch[{ordinal}].{key}")
            else:
                C.equal(reported[key], value, f"control mismatch[{ordinal}].{key}")
    max_mean = max(row["mean_normalized_mismatch"] for row in independently_derived_mismatches)
    max_variance = max(row["variance_normalized_mismatch"] for row in independently_derived_mismatches)
    C.close(mismatch["maximum_mean_normalized_mismatch"], max_mean, "maximum mean mismatch")
    C.close(mismatch["maximum_variance_normalized_mismatch"], max_variance, "maximum variance mismatch")
    C.equal(mismatch["all_cells_within_tolerance"], max_mean <= 1.0 and max_variance <= 1.0, "all controls within tolerance")
    return qwen_moments, controls["searches"]


def verify_grid(result: dict[str, Any]) -> None:
    grid = exact_keys(result["grid"], {"profiles", "profile_count", "rates", "modes"}, "grid")
    C.equal(grid["profiles"], PROFILES, "independently reconstructed 61-profile grid")
    C.equal(grid["profile_count"], len(PROFILES), "profile count")
    C.equal(grid["rates"], list(RATES), "rates")
    C.equal(grid["modes"], list(MODES), "modes")


def verify_audit_seals_if_present() -> None:
    if not AUDIT_RECEIPT_PATH.exists() and not AUDIT_MANIFEST_PATH.exists():
        return
    C.true(AUDIT_RECEIPT_PATH.is_file() and AUDIT_MANIFEST_PATH.is_file(), "audit receipt and manifest are both present")
    _, receipt = load_json(AUDIT_RECEIPT_PATH)
    _, manifest = load_json(AUDIT_MANIFEST_PATH)
    C.equal(receipt["schema"], "lossy-tail-v8-independent-result-audit-receipt-v1", "audit receipt schema")
    C.equal(receipt["status"], "PASS_V8_INDEPENDENT_RESULT_AUDIT", "audit receipt status")
    verify_internal(receipt, "audit_receipt_sha256", receipt["audit_receipt_sha256"], "result audit receipt")
    C.equal(manifest["schema"], "lossy-tail-v8-independent-result-audit-manifest-v1", "audit manifest schema")
    C.equal(manifest["status"], "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET", "audit manifest status")
    verify_internal(manifest, "audit_manifest_sha256", manifest["audit_manifest_sha256"], "result audit manifest")
    members = manifest["audit_artifacts"]
    C.true(isinstance(members, list), "audit artifact rows")
    C.equal(len({row["path"] for row in members}), len(members), "unique audit artifact paths")
    for row in members:
        path = HERE / row["path"]
        C.true(path.is_file(), f"audit artifact exists: {row['path']}")
        payload = path.read_bytes()
        C.equal(len(payload), row["bytes"], f"audit artifact bytes: {row['path']}")
        C.equal(digest(payload), row["sha256"], f"audit artifact hash: {row['path']}")
    C.equal(receipt["audited_result"]["file_sha256"], RESULT_FILE_SHA256, "receipt result file binding")
    C.equal(receipt["audited_result"]["internal_sha256"], RESULT_INTERNAL_SHA256, "receipt result internal binding")
    C.equal(receipt["audited_authorization"]["file_sha256"], AUTHORIZATION_FILE_SHA256, "receipt authorization file binding")
    C.equal(receipt["audited_authorization"]["internal_sha256"], AUTHORIZATION_INTERNAL_SHA256, "receipt authorization internal binding")


def main() -> None:
    if not RESULT_PATH.exists() or not AUTHORIZATION_PATH.exists():
        print(json.dumps({
            "schema": "lossy-tail-v8-independent-result-audit-preflight-v1",
            "status": "AWAITING_RESULT",
            "required_files": ["result.json", "authorization.json"],
            "result_present": RESULT_PATH.is_file(),
            "authorization_present": AUTHORIZATION_PATH.is_file(),
            "model_payload_files_opened": 0,
            "gpu_jobs": 0,
        }, sort_keys=True))
        return

    result_payload, result = load_json(RESULT_PATH)
    auth_payload, auth = load_json(AUTHORIZATION_PATH)
    C.equal(len(result_payload), RESULT_BYTES, "exact result byte count")
    C.equal(digest(result_payload), RESULT_FILE_SHA256, "exact result file SHA-256")
    C.equal(len(auth_payload), AUTHORIZATION_BYTES, "exact authorization byte count")
    C.equal(digest(auth_payload), AUTHORIZATION_FILE_SHA256, "exact authorization file SHA-256")
    exact_keys(result, RESULT_ROOT_KEYS, "result root")
    C.equal(result["schema"], "qwen_lossy_tail_peeling_oracle_result_v8", "result schema")
    verify_internal(result, "result_lock_sha256", RESULT_INTERNAL_SHA256, "production result")
    finite_tree(result, "result")
    finite_tree(auth, "authorization")
    verify_authorization(auth, result)
    _, bindings, _ = verify_external_evidence(auth, result)
    verify_runtime(result, auth)
    verify_grid(result)
    _, control_searches = verify_panel_and_controls(result, bindings)
    derived_rows = verify_calibration(result, control_searches)
    derived_decision = independent_decision(derived_rows)
    reported_decision = result["decision"]
    claim_boundary = reported_decision.get("claim_boundary")
    reported_without_claim = dict(reported_decision)
    reported_without_claim.pop("claim_boundary", None)
    C.equal(set(reported_without_claim), set(derived_decision), "decision exact fields")
    for key, expected in derived_decision.items():
        compare_numeric_tree(reported_without_claim[key], expected, f"decision.{key}")
    C.true(isinstance(claim_boundary, str) and claim_boundary, "decision claim boundary")
    retained = list(iter_retained(result["qwen_search"]))
    controls = [row for search in control_searches for row in iter_retained(search)]
    all_scored = list(iter_all_scored(result["qwen_search"])) + [
        row for search in control_searches for row in iter_all_scored(search)
    ]
    maximum_qwen_logical = max(row["read_ledger"]["maximum_cold_logical_amplification"] for row in retained)
    maximum_qwen_page = max(row["read_ledger"]["maximum_cold_page_amplification"] for row in retained)
    maximum_control_logical = max(row["read_ledger"]["maximum_cold_logical_amplification"] for row in controls)
    maximum_control_page = max(row["read_ledger"]["maximum_cold_page_amplification"] for row in controls)
    C.true(max(maximum_qwen_logical, maximum_qwen_page, maximum_control_logical, maximum_control_page) < 2.0, "all 90 retained source/control rows strictly below 2x")
    maximum_all_logical = max(row["read_ledger"]["maximum_cold_logical_amplification"] for row in all_scored)
    maximum_all_page = max(row["read_ledger"]["maximum_cold_page_amplification"] for row in all_scored)
    C.equal(len(all_scored), 225, "225 scored source/control read ledgers")
    C.true(max(maximum_all_logical, maximum_all_page) < 2.0, "all 225 scored source/control rows strictly below 2x")
    verify_audit_seals_if_present()
    print(json.dumps({
        "schema": "lossy-tail-v8-independent-result-verification-summary-v1",
        "status": "PASS_V8_INDEPENDENT_RESULT_VERIFICATION",
        "checks": C.count,
        "result_file_sha256": RESULT_FILE_SHA256,
        "result_internal_sha256": RESULT_INTERNAL_SHA256,
        "authorization_file_sha256": AUTHORIZATION_FILE_SHA256,
        "authorization_internal_sha256": AUTHORIZATION_INTERNAL_SHA256,
        "recomputed_decision_status": derived_decision["status"],
        "recomputed_optimistic_m_s_bpw": derived_decision["optimistic_m_s_bpw"],
        "recomputed_finite_best_joint_s_bpw": derived_decision["finite_best_joint_s_bpw"],
        "finite_residual_codec_warranted": derived_decision["finite_residual_codec_warranted"],
        "maximum_retained_qwen_logical_read_amplification": maximum_qwen_logical,
        "maximum_retained_qwen_page_read_amplification": maximum_qwen_page,
        "maximum_retained_control_logical_read_amplification": maximum_control_logical,
        "maximum_retained_control_page_read_amplification": maximum_control_page,
        "maximum_all_225_logical_read_amplification": maximum_all_logical,
        "maximum_all_225_page_read_amplification": maximum_all_page,
        "retained_qwen_rows_verified": len(retained),
        "retained_control_rows_verified": len(controls),
        "score_rows_verified": 5 * len(RATES) * len(MODES) * 5,
        "calibrated_rows_verified": len(derived_rows),
        "source_receipts_verified": MATRICES,
        "control_moment_cells_verified": 4 * MATRICES,
        "model_payload_files_opened_by_audit": 0,
        "cupy_imports_by_audit": 0,
        "gpu_jobs_by_audit": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT_AUDIT_FAIL: {exc}", file=sys.stderr)
        raise
