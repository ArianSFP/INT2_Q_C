#!/usr/bin/env python3
"""Independent standard-library verifier for the sealed CCQ stage-0 result."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Callable


AUDIT_RELPATH = Path("research/ccq_raw_mse_stage0_v0_independent_result_audit_20260901")
SOURCE_RELPATH = Path("research/ccq_raw_mse_stage0_v0")
RESULT_RELPATH = Path("research/ccq_raw_mse_stage0_v0_runpod_result_20260901")

AUDIT_FILES = {
    "AUDIT_RECEIPT.json",
    "MANIFEST.sha256",
    "README.md",
    "VERDICT.json",
    "test_audit.py",
    "verify_audit.py",
}
AUDIT_MANIFEST_MEMBERS = AUDIT_FILES - {"MANIFEST.sha256"}

SOURCE_HASHES = {
    "ccq_stage0.py": "7288be471925fe9596c76b6f39a814e3a2589fc3857b10f7387ca9cd2271f474",
    "design_lock.json": "89e1be67601a5d19161e29104e6205afd3850bc546e137390b95bd5392276aef",
    "MANIFEST.sha256": "e8beee384b1d4de37010b5b97ed4f412f147d826547d8dc6b2934a2bd795c78a",
    "PRIMARY_SOURCES.json": "8d40c0a68387cdd8ae45f26eef7d7181b1e6c4e0995850162b55b50fd1c7cdd6",
    "README.md": "717062424db8649c1adc8e01d23c8dc5df83f5481638c53ddce802b4ba038fbb",
    "RESEARCH_FINDING.md": "790d10b1581936db2f96d842f7df70b60f16f0fd1fa6f3006adeb614fe84c245",
    "SOURCE_RECEIPT.json": "7c683cddfe3ab80d6cb5f71af96c9dc70c8fce021407a9dee54037abd7179ea6",
    "test_source_only.py": "15fbdc80f82263e213b9c93fe4fecacc114d6c6dd7fc1ebc088cae68ed900e76",
    "verify_source.py": "f8c2e3c1396295109461bdafa8552bede3e4619abc87217ada9bedbee428f570",
}
SOURCE_MANIFEST_MEMBERS = set(SOURCE_HASHES) - {"MANIFEST.sha256"}

RESULT_HASHES = {
    "gaussian_prefix.bin": "cffc586d515c70ee98320d394730032e97d689d490a6b45ce6dcff7ee8286ec3",
    "result.json": "f48a7462aa18fcf973c8b1bdb76544d851439a2036cd79fc58d29016dbcc93c4",
    "source_prefix.bin": "125c5bb318b93ab49dd5ba0d42ee0f2068648547dc151d29b72010f81de3ab1b",
}
RESULT_SIZES = {
    "gaussian_prefix.bin": 7_518_592,
    "result.json": 49_169,
    "source_prefix.bin": 7_518_592,
}
RESULT_LOCK_SHA256 = "d458ba0f642f02b515aee2e5726dfd00b541ef782710089014a515b87387f63a"

ROWS = 768
COLS = 2048
EXPERTS = 6
MATRICES = 18
VALUES_PER_MATRIX = ROWS * COLS
VALUES_PER_EXPERT = 3 * VALUES_PER_MATRIX
EXPECTED_SLOTS = ((5, 18), (12, 7), (18, 20), (28, 83), (36, 76), (45, 41))
ROLES = ("gate", "up", "down")
FIT_SLOTS = (0, 2, 3, 5)
HOLDOUT_SLOTS = (1, 4)
OUTPUT_CHANNELS = (768, 768, 2048)
RATES = (2.15, 2.30, 2.50)

GLOBAL_HEADER_BYTES = 4096
EXPERT_HEADER_BYTES = 64
INDEX_BYTES_PER_MATRIX = VALUES_PER_MATRIX // 4
LOCAL_SCALE_BYTES_PER_MATRIX = VALUES_PER_MATRIX // 128
INDEX_BYTES_PER_EXPERT = 3 * INDEX_BYTES_PER_MATRIX
LOCAL_SCALE_BYTES_PER_EXPERT = 3 * LOCAL_SCALE_BYTES_PER_MATRIX
PARAMETER_BYTES_PER_EXPERT = sum(OUTPUT_CHANNELS) * 10
EXPERT_FIXED_BYTES = (
    EXPERT_HEADER_BYTES
    + INDEX_BYTES_PER_EXPERT
    + LOCAL_SCALE_BYTES_PER_EXPERT
    + PARAMETER_BYTES_PER_EXPERT
)
PANEL_FIXED_BYTES = GLOBAL_HEADER_BYTES + EXPERTS * EXPERT_FIXED_BYTES

TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
PROMOTION_S = TARGET_S + 0.02


class Failure(RuntimeError):
    """A closed-world audit invariant failed."""


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, label: str) -> None:
        self.count += 1
        if not condition:
            raise Failure(label)

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        self.count += 1
        if observed != expected:
            raise Failure(f"{label}: observed {observed!r}, expected {expected!r}")

    def close(self, observed: Any, expected: Any, label: str, tolerance: float = 1.0e-12) -> None:
        self.count += 1
        try:
            left = float(observed)
            right = float(expected)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Failure(f"{label}: non-numeric value") from exc
        if not math.isfinite(left) or not math.isfinite(right):
            raise Failure(f"{label}: non-finite value")
        scale = max(1.0, abs(left), abs(right))
        if abs(left - right) > tolerance * scale:
            raise Failure(f"{label}: observed {left!r}, expected {right!r}")


def must(condition: bool, label: str) -> None:
    if not condition:
        raise Failure(label)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _finite_float(token: str) -> float:
    try:
        value = float(token)
    except (ValueError, OverflowError) as exc:
        raise Failure("invalid JSON float") from exc
    if not math.isfinite(value):
        raise Failure("non-finite JSON float")
    return value


def _reject_constant(token: str) -> Any:
    raise Failure(f"non-standard JSON constant {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise Failure(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def parse_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Failure(f"{label}: invalid UTF-8") from exc
    try:
        return json.loads(
            text,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise Failure(f"{label}: invalid JSON") from exc


def parse_json_file(path: Path) -> Any:
    return parse_json_bytes(path.read_bytes(), path.name)


def parse_manifest(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    must(raw.endswith(b"\n") and not raw.endswith(b"\n\n"), "manifest final LF")
    must(b"\r" not in raw, "manifest CR rejected")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Failure("manifest is not ASCII") from exc
    rows: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ")
        must(len(parts) == 2, "malformed manifest row")
        digest, name = parts
        must(len(digest) == 64 and digest == digest.lower(), "malformed manifest digest")
        must(all(character in "0123456789abcdef" for character in digest), "non-hex manifest digest")
        must(name == Path(name).name and "/" not in name and "\\" not in name, "unsafe manifest name")
        must(name not in rows, "duplicate manifest name")
        rows[name] = digest
    return rows


def check_exact_closure(path: Path, expected: set[str], checks: Checks, label: str) -> None:
    checks.require(path.is_dir() and not path.is_symlink(), f"{label} directory")
    observed: set[str] = set()
    with os.scandir(path) as iterator:
        for entry in iterator:
            checks.require(not entry.is_symlink(), f"{label} symlink rejected: {entry.name}")
            checks.require(entry.is_file(follow_symlinks=False), f"{label} non-file rejected: {entry.name}")
            observed.add(entry.name)
    checks.equal(observed, expected, f"{label} exact closure")


def contained(root: Path, relative: Path) -> Path:
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise Failure(f"path escaped root: {relative}") from exc
    return result


def check_source(source: Path, checks: Checks) -> None:
    check_exact_closure(source, set(SOURCE_HASHES), checks, "source")
    for name, expected in sorted(SOURCE_HASHES.items()):
        checks.equal(sha256_file(source / name), expected, f"source hash {name}")

    manifest = parse_manifest(source / "MANIFEST.sha256")
    checks.equal(set(manifest), SOURCE_MANIFEST_MEMBERS, "source manifest member set")
    for name in sorted(SOURCE_MANIFEST_MEMBERS):
        checks.equal(manifest[name], SOURCE_HASHES[name], f"source manifest row {name}")

    verifier_path = source / "verify_source.py"
    verifier_text = verifier_path.read_text(encoding="utf-8")
    ast.parse(verifier_text, filename=str(verifier_path))
    compile(verifier_text, str(verifier_path), "exec", dont_inherit=True)
    checks.require(True, "source verifier parses and compiles")
    payload_call = "source, source_receipts = load_sources(cp, lock_path, lock)"
    checks.equal(verifier_text.count(payload_call), 1, "source verifier unique payload call predicate")
    checks.require(
        verifier_text.index('runner.index("args.authorization != AUTHORIZATION")')
        < verifier_text.index("runner.index(payload_call)"),
        "source verifier repaired predicate order",
    )
    checks.require('runner.index("load_sources(cp")' not in verifier_text, "source verifier stale predicate absent")

    runner_path = source / "ccq_stage0.py"
    runner_text = runner_path.read_text(encoding="utf-8")
    ast.parse(runner_text, filename=str(runner_path))
    compile(runner_text, str(runner_path), "exec", dont_inherit=True)
    checks.require(True, "runner parses and compiles")
    checks.require(
        runner_text.index("args.authorization != AUTHORIZATION") < runner_text.index(payload_call),
        "runner authorization before actual payload call",
    )

    receipt = parse_json_file(source / "SOURCE_RECEIPT.json")
    checks.equal(receipt.get("schema"), "ccq-raw-mse-stage0-source-receipt-v0", "source receipt schema")
    checks.equal(receipt.get("status"), "READY_SOURCE_ONLY_NOT_EXECUTED", "source receipt sealing status")
    checks.equal(receipt.get("gpu_execution"), "not run", "source receipt pre-execution scope")
    checks.equal(receipt.get("model_payload_access"), "none", "source receipt payload scope")
    checks.equal(receipt.get("runner_sha256"), SOURCE_HASHES["ccq_stage0.py"], "source receipt runner")


def _validate_float32(raw: bytes, positive: bool, label: str) -> None:
    for (value,) in struct.iter_unpack("<f", raw):
        must(math.isfinite(value), f"{label}: non-finite float32")
        if positive:
            must(value > 0.0, f"{label}: non-positive float32")


def _validate_float16(raw: bytes, label: str) -> None:
    for (value,) in struct.iter_unpack("<e", raw):
        must(math.isfinite(value) and value > 0.0, f"{label}: invalid float16")


def parse_packet(payload: bytes, label: str, checks: Checks | None = None) -> dict[str, int]:
    own = checks if checks is not None else Checks()
    own.equal(len(payload), PANEL_FIXED_BYTES, f"{label} packet bytes")
    metadata = canonical_json_bytes(
        {"format": "CCQ-RMSE-S0-v0", "label": label, "experts": EXPERTS, "matrices": MATRICES}
    )
    global_prefix = struct.pack(
        "<8sIIII", b"CCQRM0\0\0", 0, EXPERTS, MATRICES, EXPERT_FIXED_BYTES
    ) + metadata
    expected_global = global_prefix + bytes(GLOBAL_HEADER_BYTES - len(global_prefix))
    own.require(payload[:GLOBAL_HEADER_BYTES] == expected_global, f"{label} canonical global header")

    rebuilt = bytearray(expected_global)
    offset = GLOBAL_HEADER_BYTES
    float32_values = 0
    float16_values = 0
    for slot, (layer, expert) in enumerate(EXPECTED_SLOTS):
        local_prefix = struct.pack(
            "<8sIIIIII", b"CCQEXP0\0", 0, slot, layer, expert, 3, EXPERT_FIXED_BYTES
        )
        expected_local = local_prefix + bytes(EXPERT_HEADER_BYTES - len(local_prefix))
        own.require(
            payload[offset : offset + EXPERT_HEADER_BYTES] == expected_local,
            f"{label} canonical expert header {slot}",
        )
        rebuilt.extend(expected_local)
        offset += EXPERT_HEADER_BYTES

        for role_index, channels in enumerate(OUTPUT_CHANNELS):
            field_label = f"{label} slot {slot} role {ROLES[role_index]}"
            index_raw = payload[offset : offset + INDEX_BYTES_PER_MATRIX]
            own.equal(len(index_raw), INDEX_BYTES_PER_MATRIX, f"{field_label} index bytes")
            rebuilt.extend(index_raw)
            offset += INDEX_BYTES_PER_MATRIX

            local_raw = payload[offset : offset + LOCAL_SCALE_BYTES_PER_MATRIX]
            own.equal(len(local_raw), LOCAL_SCALE_BYTES_PER_MATRIX, f"{field_label} local-scale bytes")
            repacked = bytes((value & 15) | ((value >> 4) << 4) for value in local_raw)
            own.require(repacked == local_raw, f"{field_label} canonical uint4 packing")
            rebuilt.extend(local_raw)
            offset += LOCAL_SCALE_BYTES_PER_MATRIX

            code_scale_raw = payload[offset : offset + 4 * channels]
            _validate_float32(code_scale_raw, True, f"{field_label} code scale")
            own.require(True, f"{field_label} finite positive code scales")
            rebuilt.extend(code_scale_raw)
            offset += 4 * channels

            code_zp_raw = payload[offset : offset + 4 * channels]
            _validate_float32(code_zp_raw, False, f"{field_label} code zero point")
            own.require(True, f"{field_label} finite code zero points")
            rebuilt.extend(code_zp_raw)
            offset += 4 * channels

            super_raw = payload[offset : offset + 2 * channels]
            _validate_float16(super_raw, f"{field_label} super scale")
            own.require(True, f"{field_label} canonical finite positive FP16 super scales")
            rebuilt.extend(super_raw)
            offset += 2 * channels
            float32_values += 2 * channels
            float16_values += channels

    own.equal(offset, len(payload), f"{label} no trailing member")
    own.require(bytes(rebuilt) == payload, f"{label} canonical byte roundtrip")
    return {
        "bytes": len(payload),
        "float32_values": float32_values,
        "float16_values": float16_values,
        "checks": own.count,
    }


def rate_ledger(expert_count: int) -> tuple[list[dict[str, Any]], float, float]:
    values = expert_count * VALUES_PER_EXPERT
    fixed = GLOBAL_HEADER_BYTES + expert_count * EXPERT_FIXED_BYTES
    prefix = fixed * 8.0 / values
    required_q = TARGET_F / 2.0 ** (2.0 * prefix)
    rows: list[dict[str, Any]] = []
    for requested in RATES:
        physical = math.ceil(requested * values / 8.0)
        local_total = physical - GLOBAL_HEADER_BYTES
        local_min, remainder = divmod(local_total, expert_count)
        local_max = local_min + int(remainder != 0)
        cold = GLOBAL_HEADER_BYTES + 4096 * math.ceil(local_max / 4096.0)
        rows.append(
            {
                "requested_bpw": requested,
                "actual_bpw": physical * 8.0 / values,
                "physical_bytes": physical,
                "fixed_prefix_bytes": fixed,
                "ideal_residual_bytes": physical - fixed,
                "local_frame_min_bytes": local_min,
                "local_frame_max_bytes": local_max,
                "large_frame_count": remainder,
                "cold_expert_bytes_4k": cold,
                "cold_read_amplification": cold / (physical / expert_count),
            }
        )
    return rows, prefix, required_q


def compare_rate_row(observed: dict[str, Any], expected: dict[str, Any], checks: Checks, label: str) -> None:
    integer_fields = (
        "physical_bytes",
        "fixed_prefix_bytes",
        "ideal_residual_bytes",
        "local_frame_min_bytes",
        "local_frame_max_bytes",
        "large_frame_count",
        "cold_expert_bytes_4k",
    )
    float_fields = ("requested_bpw", "actual_bpw", "cold_read_amplification")
    for name in integer_fields:
        checks.equal(observed.get(name), expected[name], f"{label} {name}")
    for name in float_fields:
        checks.close(observed.get(name), expected[name], f"{label} {name}")
    checks.require(float(observed["cold_read_amplification"]) < 2.0, f"{label} cold read below 2x")
    checks.require(int(observed["ideal_residual_bytes"]) >= 0, f"{label} nonnegative residual budget")


def check_rates(result: dict[str, Any], checks: Checks) -> None:
    rows6, prefix6, required6 = rate_ledger(6)
    checks.close(result.get("six_expert_fixed_prefix_bpw"), prefix6, "six-expert prefix bpw")
    checks.close(result.get("six_expert_required_first_stage_q"), required6, "six-expert required q")
    observed6 = result.get("six_expert_rate_ledger")
    checks.require(isinstance(observed6, list) and len(observed6) == 3, "six-expert rate rows")
    for index, expected in enumerate(rows6):
        compare_rate_row(observed6[index], expected, checks, f"six-expert rate {index}")

    projection = result.get("hypothetical_128_expert_ledger")
    checks.require(isinstance(projection, dict), "128-expert ledger object")
    checks.equal(
        projection.get("claim_boundary"),
        "Arithmetic projection only; no 128-expert generalization evidence.",
        "128-expert claim boundary",
    )
    rows128, prefix128, required128 = rate_ledger(128)
    checks.close(projection.get("fixed_prefix_bpw"), prefix128, "128-expert prefix bpw")
    checks.close(projection.get("required_first_stage_q"), required128, "128-expert required q")
    observed128 = projection.get("rates")
    checks.require(isinstance(observed128, list) and len(observed128) == 3, "128-expert rate rows")
    for index, expected in enumerate(rows128):
        compare_rate_row(observed128[index], expected, checks, f"128-expert rate {index}")


def finite_number(value: Any, label: str, positive: bool = False) -> float:
    must(not isinstance(value, bool), f"{label}: boolean is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Failure(f"{label}: non-numeric") from exc
    must(math.isfinite(result), f"{label}: non-finite")
    if positive:
        must(result > 0.0, f"{label}: non-positive")
    return result


def check_score(score: dict[str, Any], name: str, prefix: float, checks: Checks) -> dict[str, float]:
    rows = score.get("matrices")
    checks.require(isinstance(rows, list) and len(rows) == MATRICES, f"{name} matrix rows")
    expert_sse = [0.0] * EXPERTS
    expert_energy = [0.0] * EXPERTS
    fit_sse = 0.0
    fit_energy = 0.0
    held_sse = 0.0
    held_energy = 0.0
    for ordinal, row in enumerate(rows):
        checks.equal(row.get("matrix_ordinal"), ordinal, f"{name} matrix ordinal {ordinal}")
        slot = ordinal // 3
        role = ROLES[ordinal % 3]
        split = "holdout" if slot in HOLDOUT_SLOTS else "fit"
        checks.equal(row.get("slot"), slot, f"{name} slot {ordinal}")
        checks.equal(row.get("role"), role, f"{name} role {ordinal}")
        checks.equal(row.get("split"), split, f"{name} split {ordinal}")
        sse = finite_number(row.get("sse"), f"{name} SSE {ordinal}", positive=True)
        energy = finite_number(row.get("source_energy"), f"{name} energy {ordinal}", positive=True)
        checks.close(row.get("relative_residual_energy"), sse / energy, f"{name} q {ordinal}")
        expert_sse[slot] += sse
        expert_energy[slot] += energy
        if split == "fit":
            fit_sse += sse
            fit_energy += energy
            checks.require("accumulated_no_recovery_F_lower_bound" not in row, f"{name} no fit lower bound {ordinal}")
        else:
            held_sse += sse
            held_energy += energy

    factor = 2.0 ** (2.0 * prefix)
    checks.close(score.get("fit_relative_residual_energy"), fit_sse / fit_energy, f"{name} fit q")
    held_q = held_sse / held_energy
    held_f = held_q * factor
    held_s = -0.5 * math.log2(held_f)
    checks.close(score.get("holdout_relative_residual_energy"), held_q, f"{name} held q")
    checks.close(score.get("holdout_F_oracle"), held_f, f"{name} held F")
    checks.close(score.get("holdout_s_oracle"), held_s, f"{name} held s")

    experts = score.get("heldout_experts")
    checks.require(isinstance(experts, list) and len(experts) == len(HOLDOUT_SLOTS), f"{name} expert rows")
    for expert_row, slot in zip(experts, HOLDOUT_SLOTS):
        layer, expert = EXPECTED_SLOTS[slot]
        checks.equal(expert_row.get("slot"), slot, f"{name} expert slot {slot}")
        checks.equal(expert_row.get("layer"), layer, f"{name} expert layer {slot}")
        checks.equal(expert_row.get("expert"), expert, f"{name} expert identity {slot}")
        q_value = expert_sse[slot] / expert_energy[slot]
        f_value = q_value * factor
        checks.close(expert_row.get("sse"), expert_sse[slot], f"{name} expert SSE {slot}")
        checks.close(expert_row.get("source_energy"), expert_energy[slot], f"{name} expert energy {slot}")
        checks.close(expert_row.get("relative_residual_energy"), q_value, f"{name} expert q {slot}")
        checks.close(expert_row.get("F_oracle"), f_value, f"{name} expert F {slot}")
        checks.close(expert_row.get("s_oracle"), -0.5 * math.log2(f_value), f"{name} expert s {slot}")

    accumulated = 0.0
    first_crossing: tuple[int, float] | None = None
    for ordinal, row in enumerate(rows):
        if ordinal // 3 not in HOLDOUT_SLOTS:
            continue
        accumulated += float(row["sse"])
        lower_f = accumulated / held_energy * factor
        checks.close(
            row.get("accumulated_no_recovery_F_lower_bound"),
            lower_f,
            f"{name} accumulated lower F {ordinal}",
        )
        if first_crossing is None and lower_f > TARGET_F:
            first_crossing = (ordinal, lower_f)
    checks.require(first_crossing is not None, f"{name} early crossing exists")
    certificate = score.get("strict_early_kill_certificate")
    checks.require(isinstance(certificate, dict), f"{name} early certificate object")
    checks.equal(certificate.get("after_matrix_ordinal"), first_crossing[0], f"{name} earliest crossing ordinal")
    checks.close(certificate.get("F_lower_bound"), first_crossing[1], f"{name} early lower F")
    checks.equal(certificate.get("why"), "remaining SSE is nonnegative", f"{name} early proof")
    return {"q": held_q, "F": held_f, "s": held_s, "early_F": first_crossing[1]}


def check_receipts_and_controls(result: dict[str, Any], checks: Checks) -> None:
    lock = result.get("source_lock")
    checks.require(isinstance(lock, dict), "result source-lock object")
    checks.equal(lock.get("bytes"), 46_013, "result source-lock bytes")
    checks.equal(
        lock.get("file_sha256"),
        "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23",
        "result source-lock file hash",
    )
    checks.equal(
        lock.get("internal_sha256"),
        "5a82dac742110d4f48bbd73ae82081e1622b10b660b7850dadfe613ff475cc5b",
        "result source-lock internal hash",
    )
    checks.equal(
        lock.get("path"),
        "/workspace/INT2__compression/blind_protocol_v2/unblinded/source_hashes.lock.json",
        "result source-lock remote path",
    )

    receipts = result.get("source_receipts")
    checks.require(isinstance(receipts, list) and len(receipts) == MATRICES, "source receipt count")
    for ordinal, receipt in enumerate(receipts):
        slot = ordinal // 3
        layer, expert = EXPECTED_SLOTS[slot]
        checks.equal(receipt.get("matrix_ordinal"), ordinal, f"source receipt ordinal {ordinal}")
        checks.equal(receipt.get("layer"), layer, f"source receipt layer {ordinal}")
        checks.equal(receipt.get("expert"), expert, f"source receipt expert {ordinal}")
        checks.equal(receipt.get("role"), ROLES[ordinal % 3], f"source receipt role {ordinal}")
        checks.equal(receipt.get("bytes"), VALUES_PER_MATRIX * 2, f"source receipt bytes {ordinal}")
        declared = receipt.get("declared_sha256")
        observed = receipt.get("observed_sha256")
        checks.require(
            isinstance(declared, str)
            and len(declared) == 64
            and all(character in "0123456789abcdef" for character in declared),
            f"source receipt hash syntax {ordinal}",
        )
        checks.equal(observed, declared, f"source receipt observed identity {ordinal}")

    matches = result.get("gaussian_moment_match")
    checks.require(isinstance(matches, list) and len(matches) == MATRICES, "Gaussian match row count")
    maximum_mean = 0.0
    maximum_rms = 0.0
    for ordinal, row in enumerate(matches):
        checks.equal(row.get("matrix_ordinal"), ordinal, f"Gaussian match ordinal {ordinal}")
        mean_error = finite_number(row.get("max_abs_mean_error"), f"Gaussian mean error {ordinal}")
        rms_error = finite_number(row.get("max_abs_centered_rms_error"), f"Gaussian RMS error {ordinal}")
        checks.require(mean_error >= 0.0 and rms_error >= 0.0, f"Gaussian nonnegative error {ordinal}")
        maximum_mean = max(maximum_mean, mean_error)
        maximum_rms = max(maximum_rms, rms_error)
    checks.require(maximum_mean <= 2.0e-9, "Gaussian row-mean match within 2e-9")
    checks.require(maximum_rms <= 2.0e-9, "Gaussian row-RMS match within 2e-9")


def check_traces(result: dict[str, Any], checks: Checks) -> None:
    stages = ["continuous_1", "continuous_2", "continuous_3", "cluster_refine_1", "cluster_refine_2"]
    for name in ("source_trace", "gaussian_trace"):
        traces = result.get(name)
        checks.require(isinstance(traces, list) and len(traces) == MATRICES, f"{name} row count")
        for ordinal, row in enumerate(traces):
            channels = OUTPUT_CHANNELS[ordinal % 3]
            checks.equal(row.get("matrix_ordinal"), ordinal, f"{name} ordinal {ordinal}")
            checks.equal(row.get("N"), channels, f"{name} N {ordinal}")
            checks.equal(row.get("K"), VALUES_PER_MATRIX // channels, f"{name} K {ordinal}")
            trace = row.get("trace")
            checks.require(isinstance(trace, list) and len(trace) == len(stages), f"{name} stages {ordinal}")
            checks.equal([item.get("stage") for item in trace], stages, f"{name} stage names {ordinal}")
            checks.require(
                all(finite_number(item.get("relative_residual_energy"), f"{name} trace q") > 0.0 for item in trace),
                f"{name} finite positive trace {ordinal}",
            )


def check_result(result_dir: Path, checks: Checks) -> dict[str, Any]:
    check_exact_closure(result_dir, set(RESULT_HASHES), checks, "result")
    for name, expected in sorted(RESULT_HASHES.items()):
        path = result_dir / name
        checks.equal(path.stat().st_size, RESULT_SIZES[name], f"result bytes {name}")
        checks.equal(sha256_file(path), expected, f"result hash {name}")

    result_path = result_dir / "result.json"
    raw_result = result_path.read_bytes()
    result = parse_json_bytes(raw_result, "result.json")
    checks.require(isinstance(result, dict), "result JSON object")
    pretty = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    checks.require(raw_result == pretty, "canonical pretty result JSON")
    locked = dict(result)
    declared_lock = locked.pop("result_lock_sha256", None)
    checks.equal(declared_lock, RESULT_LOCK_SHA256, "declared result lock")
    checks.equal(
        hashlib.sha256(canonical_json_bytes(locked)).hexdigest(),
        RESULT_LOCK_SHA256,
        "recomputed result lock",
    )

    checks.equal(result.get("schema"), "ccq-raw-mse-stage0-result-v0", "result schema")
    checks.equal(result.get("status"), "KILL", "result decision")
    checks.equal(
        result.get("decision_reasons"),
        ["pooled held-out canonical CCQ first stage fails the ideal-residual oracle"],
        "result decision reason",
    )
    checks.equal(
        result.get("claim_boundary"),
        "Frozen paper-derived CCQ Code-Cluster cell only. Not an official encoder reproduction, finite residual codec, fresh-validation result, model-wide result, or target achievement.",
        "result claim boundary",
    )
    checks.equal(result.get("split"), {"fit_slots": list(FIT_SLOTS), "holdout_slots": list(HOLDOUT_SLOTS)}, "fixed split")
    checks.equal(result.get("target"), {"F": TARGET_F, "s": TARGET_S, "promotion_s": PROMOTION_S}, "target constants")

    for label in ("source", "gaussian"):
        declaration = result.get(f"{label}_prefix")
        checks.require(isinstance(declaration, dict), f"{label} packet declaration")
        checks.equal(declaration.get("bytes"), RESULT_SIZES[f"{label}_prefix.bin"], f"{label} declared bytes")
        checks.equal(declaration.get("sha256"), RESULT_HASHES[f"{label}_prefix.bin"], f"{label} declared hash")
        payload = (result_dir / f"{label}_prefix.bin").read_bytes()
        stats = parse_packet(payload, label, checks)
        checks.equal(stats["float32_values"], 43_008, f"{label} packet float32 count")
        checks.equal(stats["float16_values"], 21_504, f"{label} packet float16 count")

    check_rates(result, checks)
    check_receipts_and_controls(result, checks)
    check_traces(result, checks)
    prefix = float(result["six_expert_fixed_prefix_bpw"])
    source_summary = check_score(result["source_score"], "source", prefix, checks)
    gaussian_summary = check_score(result["gaussian_score"], "gaussian", prefix, checks)
    checks.close(
        result.get("matched_source_minus_gaussian_s"),
        source_summary["s"] - gaussian_summary["s"],
        "matched source-minus-Gaussian s",
    )
    checks.require(float(result["matched_source_minus_gaussian_s"]) < 0.0, "source does not beat Gaussian control")
    checks.require(source_summary["F"] > TARGET_F, "source pooled ideal-residual oracle fails")
    checks.require(source_summary["early_F"] > TARGET_F, "source strict early kill")
    checks.require(all(float(row["F_oracle"]) > TARGET_F for row in result["source_score"]["heldout_experts"]), "both source heldout experts fail")

    # Re-derive the containing-oracle identity at every requested final rate.
    q_value = source_summary["q"]
    oracle_f = source_summary["F"]
    for requested in RATES:
        ideal_final_q = q_value * 2.0 ** (-2.0 * (requested - prefix))
        normalized_f = ideal_final_q / 2.0 ** (-2.0 * requested)
        checks.close(normalized_f, oracle_f, f"oracle composition identity R={requested}")

    runtime = result.get("runtime")
    checks.require(isinstance(runtime, dict), "runtime object")
    checks.equal(
        runtime.get("environment"),
        {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUPY_ACCELERATORS": "", "NVIDIA_TF32_OVERRIDE": "0"},
        "runtime deterministic environment",
    )
    checks.require(finite_number(runtime.get("elapsed_seconds"), "elapsed seconds", positive=True) > 0.0, "positive runtime")
    return result


def check_audit_metadata(package: Path, checks: Checks) -> None:
    receipt = parse_json_file(package / "AUDIT_RECEIPT.json")
    verdict = parse_json_file(package / "VERDICT.json")
    checks.equal(receipt.get("schema"), "ccq-stage0-independent-result-audit-receipt-v0", "audit receipt schema")
    checks.equal(receipt.get("status"), "PASS_KILL_CONFIRMED", "audit receipt status")
    checks.equal(receipt.get("verdict"), "PASS", "audit receipt verdict")
    checks.equal(receipt.get("source_manifest_sha256"), SOURCE_HASHES["MANIFEST.sha256"], "audit source binding")
    checks.equal(receipt.get("result_lock_sha256"), RESULT_LOCK_SHA256, "audit result-lock binding")
    checks.equal(receipt.get("result_file_sha256"), RESULT_HASHES, "audit result-file binding")
    checks.equal(verdict.get("schema"), "ccq-stage0-independent-result-audit-verdict-v0", "verdict schema")
    checks.equal(verdict.get("verdict"), "PASS", "verdict value")
    checks.equal(verdict.get("confirmed_result"), "KILL", "confirmed result")
    checks.equal(verdict.get("claim_scope"), "frozen_paper_derived_code_cluster_cell_on_authenticated_six_expert_panel", "verdict scope")


def verify(root: Path, audit_override: Path | None = None, verify_audit_closure: bool = True) -> int:
    checks = Checks()
    root = root.resolve()
    source = contained(root, SOURCE_RELPATH)
    result_dir = contained(root, RESULT_RELPATH)
    package = audit_override.resolve() if audit_override is not None else contained(root, AUDIT_RELPATH)
    try:
        package.relative_to(root)
    except ValueError as exc:
        raise Failure("audit package escaped root") from exc

    if verify_audit_closure:
        check_exact_closure(package, AUDIT_FILES, checks, "audit")
        manifest = parse_manifest(package / "MANIFEST.sha256")
        checks.equal(set(manifest), AUDIT_MANIFEST_MEMBERS, "audit manifest member set")
        for name in sorted(AUDIT_MANIFEST_MEMBERS):
            checks.equal(sha256_file(package / name), manifest[name], f"audit manifest hash {name}")

    check_source(source, checks)
    check_result(result_dir, checks)
    check_audit_metadata(package, checks)
    return checks.count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = verify(args.root)
    except (Failure, OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: KILL_CONFIRMED checks={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
