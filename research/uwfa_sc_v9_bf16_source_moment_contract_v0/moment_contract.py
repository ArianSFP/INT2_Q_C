#!/usr/bin/env python3
"""Universal, fail-closed BF16 source-moment contract.

Importing this module is inert.  NumPy and source bytes are supplied only by a
reviewed dispatcher after it has authenticated this package, the exact CPU
runtime, and an out-of-band authorization digest.  The public record uses only
canonical slot, SwiGLU role, shape, order, and cryptographic closures.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AUTHORIZATION_SCHEMA = "uwfa-sc-v9-bf16-source-authorization-v1"
AUTHORIZATION_STATUS = "EXTERNALLY_AUTHORIZED_EXACT_BF16_SOURCE_SET"
MOMENT_CONTRACT_SCHEMA = "uwfa-sc-v9-bf16-matrix-moment-contract-v1"
MOMENT_CONTRACT_STATUS = "AUTHENTICATED_EXTERNAL_SOURCE_MOMENTS"
MOMENT_SEMANTICS = (
    "ORIGINAL_BF16_MATRIX_STORAGE_ORIENTATION_BINARY64_MEAN_AND_CENTERED_SSE"
)
PUBLICATION_SCHEMA = "uwfa-sc-v9-bf16-source-moment-publication-v1"
COMPLETE_SCHEMA = "uwfa-sc-v9-bf16-source-moment-complete-v1"
PACKAGE_MANIFEST_SCHEMA = "uwfa-sc-v9-bf16-source-moment-source-manifest-v0"
RUNTIME_PINS_SCHEMA = "uwfa-sc-v9-bf16-source-moment-runtime-pins-v0"

ROLES = ("gate", "up", "down")
EXPERTS = 6
HIDDEN = 2048
INTERMEDIATE = 768
VALUES_PER_MATRIX = HIDDEN * INTERMEDIATE
BYTES_PER_MATRIX = 2 * VALUES_PER_MATRIX
MATRIX_COUNT = EXPERTS * len(ROLES)
TOTAL_VALUES = MATRIX_COUNT * VALUES_PER_MATRIX
TOTAL_SOURCE_BYTES = MATRIX_COUNT * BYTES_PER_MATRIX
MAX_AUTHORIZATION_BYTES = 128 * 1024

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
GENERATOR_DOMAIN = b"UWFA-SC-V9-FULL-PTQ-MATCHED-GAUSSIAN-PCG64-v1\x00"
SOURCE_SET_DOMAIN = b"UWFA-SC-V9-EXTERNAL-BF16-SOURCE-SET-v1\x00"
PUBLICATION_MEMBERS_DOMAIN = b"UWFA-SC-V9-SOURCE-MOMENT-PUBLICATION-v1\x00"
MEAN_RMS_TOLERANCE = 2.0 ** -17
CENTERED_RMS_RELATIVE_TOLERANCE = 2.0 ** -15
AFFINE_ITERATIONS = 6

SOURCE_CLOSURE_FIELDS = (
    "source_artifact_sha256",
    "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "source_pipeline_sha256",
    "source_score_receipt_sha256",
    "source_moment_auditor_sha256",
)


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def digest(value: Any, label: str) -> str:
    require(is_sha256(value), f"{label} must be lowercase SHA-256 hex")
    return str(value)


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    require(field not in result, f"seal field already present: {field}")
    result[field] = sha256_bytes(canonical_json(result))
    return result


def validate_seal(value: Mapping[str, Any], field: str) -> None:
    claimed = digest(value.get(field), field)
    clean = dict(value)
    clean.pop(field, None)
    require(sha256_bytes(canonical_json(clean)) == claimed, f"{field} integrity")


def f64_hex(value: float) -> str:
    require(math.isfinite(value), "finite binary64")
    return struct.pack("<d", float(value)).hex()


def from_f64_hex(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, str) and len(value) == 16 and value == value.lower(),
        f"{label} little-endian binary64 bits",
    )
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
    return (HIDDEN, INTERMEDIATE) if role == "down" else (INTERMEDIATE, HIDDEN)


def expected_matrix_order() -> tuple[tuple[int, str], ...]:
    return tuple((slot, role) for slot in range(EXPERTS) for role in ROLES)


def canonical_source_relpath(ordinal: int, slot: int, role: str) -> str:
    require(type(ordinal) is int and 0 <= ordinal < MATRIX_COUNT, "matrix ordinal")
    require(type(slot) is int and 0 <= slot < EXPERTS, "canonical slot")
    require(role in ROLES, "SwiGLU role")
    return f"matrix_{ordinal:02d}_slot_{slot:02d}_{role}.bf16"


def panel_record() -> dict[str, Any]:
    return {
        "experts": EXPERTS,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "roles": list(ROLES),
        "weights": TOTAL_VALUES,
        "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
    }


def source_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(SOURCE_SET_DOMAIN + canonical_json(list(rows)))


def validate_authorization_record(record: Any) -> dict[str, Any]:
    """Validate semantics and the internal seal of an external authorization.

    The internal seal provides integrity only.  A caller must separately pin
    the exact authorization *file* digest with ``parse_external_authorization``.
    """
    require(isinstance(record, dict), "source authorization object")
    required = {
        "schema",
        "status",
        "panel",
        "source_closure",
        "matrices",
        "source_set_sha256",
        "authorization_sha256",
    }
    require(set(record) == required, "source authorization fields")
    require(record["schema"] == AUTHORIZATION_SCHEMA, "source authorization schema")
    require(record["status"] == AUTHORIZATION_STATUS, "source authorization status")
    require(record["panel"] == panel_record(), "source authorization panel")

    closure = record["source_closure"]
    require(
        isinstance(closure, dict) and set(closure) == set(SOURCE_CLOSURE_FIELDS),
        "source closure fields",
    )
    for field in SOURCE_CLOSURE_FIELDS:
        digest(closure[field], f"source closure {field}")

    raw_rows = record["matrices"]
    require(isinstance(raw_rows, list) and len(raw_rows) == MATRIX_COUNT, "eighteen source rows")
    row_fields = {
        "matrix_ordinal",
        "slot",
        "role",
        "shape",
        "values",
        "bytes",
        "source_relpath",
        "source_matrix_bf16_sha256",
    }
    for ordinal, (row, expected) in enumerate(
        zip(raw_rows, expected_matrix_order(), strict=True)
    ):
        require(isinstance(row, dict) and set(row) == row_fields, f"source row fields {ordinal}")
        slot, role = expected
        shape = role_shape(role)
        require(type(row["matrix_ordinal"]) is int, f"source ordinal type {ordinal}")
        require(row["matrix_ordinal"] == ordinal, f"source ordinal {ordinal}")
        require(type(row["slot"]) is int, f"source slot type {ordinal}")
        require(row["slot"] == slot and row["role"] == role, f"source slot/role {ordinal}")
        require(row["shape"] == list(shape), f"source shape {ordinal}")
        require(type(row["values"]) is int and row["values"] == VALUES_PER_MATRIX, f"source values {ordinal}")
        require(type(row["bytes"]) is int and row["bytes"] == BYTES_PER_MATRIX, f"source bytes {ordinal}")
        require(
            row["source_relpath"] == canonical_source_relpath(ordinal, slot, role),
            f"canonical source path {ordinal}",
        )
        digest(row["source_matrix_bf16_sha256"], f"source matrix {ordinal}")

    observed_root = source_set_sha256(raw_rows)
    require(
        digest(record["source_set_sha256"], "source set") == observed_root,
        "source set closure",
    )
    validate_seal(record, "authorization_sha256")
    return dict(record)


def parse_external_authorization(
    payload: bytes, expected_authorization_file_sha256: str
) -> dict[str, Any]:
    """Authenticate exact external bytes before trusting a self-sealed record."""
    require(isinstance(payload, bytes) and bool(payload), "authorization file bytes")
    require(len(payload) <= MAX_AUTHORIZATION_BYTES, "authorization file size bound")
    expected = digest(expected_authorization_file_sha256, "expected authorization file")
    require(sha256_bytes(payload) == expected, "out-of-band authorization file digest")
    try:
        record = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authorization canonical JSON") from exc
    require(pretty_json(record) == payload, "authorization must use canonical pretty JSON serialization")
    return validate_authorization_record(record)


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
        return {
            "slot": self.slot,
            "role": self.role,
            "shape": list(self.shape),
            "values": self.values,
            "mean_f64_hex": f64_hex(self.mean),
            "centered_sse_f64_hex": f64_hex(self.centered_sse),
        }


def bf16_to_f64(np: Any, words: Any) -> Any:
    """Bit-expand little-endian BF16 to binary32, then exactly widen to FP64."""
    source = np.ascontiguousarray(words, dtype="<u2")
    expanded = source.astype(np.uint32) << np.uint32(16)
    return expanded.view(np.float32).astype(np.float64)


def measured_moments(np: Any, words: Any) -> tuple[float, float, float]:
    """Frozen two-pass NumPy FP64 population-moment computation.

    Storage order is C-order.  The mean is ``np.mean(..., dtype=np.float64)``;
    centered SSE is a second FP64 pass over ``(x - mean) * (x - mean)``;
    energy is a separate FP64 sum of ``x * x``.  There is no ddof correction.
    """
    values = bf16_to_f64(np, words).reshape(-1, order="C")
    require(bool(np.isfinite(values).all()), "source BF16 values finite")
    mean = float(np.mean(values, dtype=np.float64))
    centered = values - mean
    centered_sse = float(np.sum(centered * centered, dtype=np.float64))
    energy = float(np.sum(values * values, dtype=np.float64))
    require(centered_sse > 0.0 and energy > 0.0, "source BF16 matrix nondegenerate")
    require(
        math.isfinite(mean) and math.isfinite(centered_sse) and math.isfinite(energy),
        "source moments finite",
    )
    return mean, centered_sse, energy


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_bound_regular_file(root: Path, row: Mapping[str, Any]) -> bytes:
    """Read one canonical file once, rejecting links, races, and digest drift."""
    relpath = str(row["source_relpath"])
    require(Path(relpath).name == relpath, "source relpath must be one plain filename")
    candidate = root / relpath
    require(not candidate.is_symlink(), f"source symlink forbidden: {relpath}")
    resolved = candidate.resolve(strict=True)
    require(resolved == candidate, f"source indirection forbidden: {relpath}")
    require(resolved.parent == root, f"source path escapes root: {relpath}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"source not regular: {relpath}")
        expected_bytes = int(row["bytes"])
        require(before.st_size == expected_bytes, f"source byte length: {relpath}")
        blocks: list[bytes] = []
        observed_bytes = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, expected_bytes + 1 - observed_bytes))
            if not block:
                break
            blocks.append(block)
            observed_bytes += len(block)
            require(observed_bytes <= expected_bytes, f"source grew while reading: {relpath}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(_stat_identity(before) == _stat_identity(after), f"source changed while reading: {relpath}")
    require(observed_bytes == expected_bytes, f"source truncated while reading: {relpath}")
    payload = b"".join(blocks)
    require(
        sha256_bytes(payload) == row["source_matrix_bf16_sha256"],
        f"source digest: {relpath}",
    )
    return payload


def load_authenticated_payloads(
    source_root: Path, authorization: Mapping[str, Any]
) -> tuple[bytes, ...]:
    """Authenticate the complete set before any moment is computed."""
    root_input = Path(source_root)
    require(not root_input.is_symlink(), "source root symlink forbidden")
    root = root_input.resolve(strict=True)
    require(root.is_dir(), "source root directory")
    validated = validate_authorization_record(authorization)
    payloads = tuple(_read_bound_regular_file(root, row) for row in validated["matrices"])
    require(len(payloads) == MATRIX_COUNT, "authenticated payload count")
    require(sum(len(payload) for payload in payloads) == TOTAL_SOURCE_BYTES, "authenticated payload bytes")
    return payloads


def build_moment_contract(
    np: Any,
    authorization: Mapping[str, Any],
    authenticated_payloads: Sequence[bytes],
) -> dict[str, Any]:
    """Compute all source moments after the entire byte set is authenticated."""
    auth = validate_authorization_record(authorization)
    require(len(authenticated_payloads) == MATRIX_COUNT, "moment payload coverage")
    rows: list[dict[str, Any]] = []
    for ordinal, (source_row, payload) in enumerate(
        zip(auth["matrices"], authenticated_payloads, strict=True)
    ):
        require(isinstance(payload, bytes), f"moment payload type {ordinal}")
        require(len(payload) == source_row["bytes"], f"moment payload bytes {ordinal}")
        require(
            sha256_bytes(payload) == source_row["source_matrix_bf16_sha256"],
            f"moment payload digest {ordinal}",
        )
        shape = tuple(source_row["shape"])
        words = np.frombuffer(payload, dtype="<u2").reshape(shape, order="C")
        mean, centered_sse, energy = measured_moments(np, words)
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "slot": source_row["slot"],
                "role": source_row["role"],
                "shape": list(shape),
                "values": source_row["values"],
                "mean_f64_hex": f64_hex(mean),
                "centered_sse_f64_hex": f64_hex(centered_sse),
                "energy_f64_hex": f64_hex(energy),
                "source_matrix_bf16_sha256": source_row["source_matrix_bf16_sha256"],
            }
        )
    clean = {
        "schema": MOMENT_CONTRACT_SCHEMA,
        "status": MOMENT_CONTRACT_STATUS,
        "moment_semantics": MOMENT_SEMANTICS,
        "panel": panel_record(),
        "source_closure": dict(auth["source_closure"]),
        "matrices": rows,
    }
    result = sealed(clean, "moment_contract_sha256")
    validate_moment_contract_record(result)
    return result


def validate_moment_contract_record(record: Any) -> tuple[dict[str, Any], tuple[MatrixMoment, ...]]:
    """Validate the exact ABI consumed by the frozen matched-Gaussian producer."""
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
    require(record["status"] == MOMENT_CONTRACT_STATUS, "moment contract status")
    require(record["moment_semantics"] == MOMENT_SEMANTICS, "moment contract semantics")
    require(record["panel"] == panel_record(), "moment contract panel")
    closure = record["source_closure"]
    require(
        isinstance(closure, dict) and set(closure) == set(SOURCE_CLOSURE_FIELDS),
        "moment source closure fields",
    )
    for field in SOURCE_CLOSURE_FIELDS:
        digest(closure[field], f"moment source closure {field}")

    rows = record["matrices"]
    require(isinstance(rows, list) and len(rows) == MATRIX_COUNT, "eighteen moment rows")
    fields = {
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
    moments: list[MatrixMoment] = []
    for ordinal, (row, expected) in enumerate(zip(rows, expected_matrix_order(), strict=True)):
        require(isinstance(row, dict) and set(row) == fields, f"moment row fields {ordinal}")
        slot, role = expected
        shape = role_shape(role)
        require(type(row["matrix_ordinal"]) is int and row["matrix_ordinal"] == ordinal, f"moment ordinal {ordinal}")
        require(type(row["slot"]) is int and row["slot"] == slot and row["role"] == role, f"moment slot/role {ordinal}")
        require(row["shape"] == list(shape), f"moment shape {ordinal}")
        require(type(row["values"]) is int and row["values"] == VALUES_PER_MATRIX, f"moment values {ordinal}")
        mean = from_f64_hex(row["mean_f64_hex"], f"mean {ordinal}")
        centered_sse = from_f64_hex(row["centered_sse_f64_hex"], f"centered SSE {ordinal}", positive=True)
        energy = from_f64_hex(row["energy_f64_hex"], f"energy {ordinal}", positive=True)
        expected_energy = centered_sse + VALUES_PER_MATRIX * mean * mean
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
                    row["source_matrix_bf16_sha256"], f"moment source matrix {ordinal}"
                ),
            )
        )
    validate_seal(record, "moment_contract_sha256")
    return dict(record), tuple(moments)


def derive_matrix_seed(global_seed: int, moment: MatrixMoment) -> int:
    require(type(global_seed) is int and global_seed in CONTROL_SEEDS, "frozen control seed")
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
    """Exact binary32-to-BF16 round-to-nearest, ties-to-even bit rule."""
    fp32 = np.ascontiguousarray(values, dtype=np.float32)
    raw = fp32.view(np.uint32)
    rounded = raw + np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype("<u2"))


def regenerate_gaussian_bf16(
    np: Any, moment: MatrixMoment, global_seed: int
) -> tuple[Any, dict[str, Any]]:
    """Independent reference for the frozen BF16 Gaussian regeneration law.

    A fresh NumPy PCG64 receives the domain-derived 128-bit seed and emits one
    binary64 standard-normal vector.  It is centered and scaled in FP64.  Six
    fixed affine/RNE iterations run; the lowest normalized objective wins,
    with earliest iteration breaking ties.  No search over seed or shape is
    permitted.
    """
    require(type(moment.values) is int and moment.values > 1, "moment values")
    require(moment.shape[0] * moment.shape[1] == moment.values, "moment shape/values")
    require(moment.centered_sse > 0.0 and moment.energy > 0.0, "moment nondegenerate")
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
    for iteration in range(AFFINE_ITERATIONS):
        words = fp32_to_bf16_rne(np, offset + scale * normal)
        observed = measured_moments(np, words)
        observed_mean, observed_sse, _ = observed
        mean_normalized = abs(observed_mean - moment.mean) / max(
            moment.rms, np.finfo(np.float64).tiny
        )
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
    mean_normalized = abs(achieved_mean - moment.mean) / max(
        moment.rms, np.finfo(np.float64).tiny
    )
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
        "control_bf16_sha256": sha256_bytes(payload.tobytes(order="C")),
    }
    return payload, receipt


def load_runtime_pins(package_root: Path) -> tuple[dict[str, Any], bytes]:
    path = package_root / "runtime_pins.json"
    payload = path.read_bytes()
    try:
        record = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime pins JSON") from exc
    require(pretty_json(record) == payload, "runtime pins canonical pretty JSON")
    require(isinstance(record, dict), "runtime pins object")
    require(record.get("schema") == RUNTIME_PINS_SCHEMA, "runtime pins schema")
    require(record.get("status") == "FROZEN_BEFORE_ANY_SOURCE_ACCESS", "runtime pins status")
    return record, payload


def validate_exact_runtime(np: Any, pins: Mapping[str, Any]) -> None:
    """Fail before source access unless the frozen CPU numerical tuple matches."""
    runtime = pins["moment_runtime"]
    require(sys.implementation.name == runtime["python_implementation"], "Python implementation pin")
    require(platform.python_version() == runtime["python_version"], "Python version pin")
    require(sys.byteorder == runtime["byteorder"], "runtime byte order pin")
    require(platform.system() == runtime["platform_system"], "runtime system pin")
    require(platform.machine() == runtime["platform_machine"], "runtime machine pin")
    require(sys.flags.isolated == 1, "Python must use isolated mode (-I)")
    require(sys.dont_write_bytecode, "Python must disable bytecode writes (-B)")

    executable = Path(sys.executable).resolve(strict=True)
    expected_executable = Path(runtime["python_executable_resolved"])
    require(executable == expected_executable, "Python executable path pin")
    require(executable.stat().st_size == runtime["python_executable_bytes"], "Python executable bytes pin")
    require(sha256_file(executable) == runtime["python_executable_sha256"], "Python executable digest pin")

    numpy_pin = runtime["numpy"]
    require(str(np.__version__) == numpy_pin["version"], "NumPy version pin")
    origin = Path(np.__file__).resolve(strict=True)
    require(origin == Path(numpy_pin["origin_path"]), "NumPy origin path pin")
    require(origin.stat().st_size == numpy_pin["origin_bytes"], "NumPy origin bytes pin")
    require(sha256_file(origin) == numpy_pin["origin_sha256"], "NumPy origin digest pin")
    record = Path(numpy_pin["record_path"]).resolve(strict=True)
    require(record.stat().st_size == numpy_pin["record_bytes"], "NumPy RECORD bytes pin")
    require(sha256_file(record) == numpy_pin["record_sha256"], "NumPy RECORD digest pin")
    require(np.dtype("<u2").itemsize == 2, "uint16 ABI")
    require(np.dtype("<f4").itemsize == 4 and np.dtype("<f8").itemsize == 8, "IEEE float ABI")


def verify_source_package(package_root: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Verify the externally pinned source closure before source access."""
    root = Path(package_root).resolve(strict=True)
    manifest_path = root / "SOURCE_MANIFEST.json"
    expected = digest(expected_manifest_sha256, "expected package manifest")
    require(sha256_file(manifest_path) == expected, "out-of-band package manifest digest")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    require(manifest.get("schema") == PACKAGE_MANIFEST_SCHEMA, "package manifest schema")
    require(manifest.get("status") == "SOURCE_ONLY_FROZEN_NONPROMOTING", "package manifest status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and bool(rows), "package manifest members")
    names = [row.get("name") for row in rows]
    require(names == sorted(names) and len(names) == len(set(names)), "package manifest member order")
    observed = sorted(
        path.name for path in root.iterdir() if path.is_file() and path.name != "SOURCE_MANIFEST.json"
    )
    require(names == observed, "package member set")
    for row in rows:
        require(set(row) == {"name", "bytes", "sha256"}, f"manifest row fields {row.get('name')}")
        member = root / row["name"]
        require(type(row["bytes"]) is int and member.stat().st_size == row["bytes"], f"member bytes {row['name']}")
        require(sha256_file(member) == digest(row["sha256"], f"member hash {row['name']}"), f"member digest {row['name']}")
    return manifest


def _members_root(rows: Sequence[Mapping[str, Any]]) -> str:
    value = hashlib.sha256(PUBLICATION_MEMBERS_DOMAIN)
    for row in rows:
        name = str(row["name"]).encode("ascii")
        value.update(struct.pack("<Q", len(name)))
        value.update(name)
        value.update(struct.pack("<Q", int(row["bytes"])))
        value.update(bytes.fromhex(str(row["sha256"])))
    return value.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            require(count > 0, f"write made no progress: {path.name}")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_records(output_root: Path, records: Mapping[str, bytes]) -> dict[str, Any]:
    """Publish an absent directory atomically; failures never create final."""
    requested = Path(output_root)
    parent = requested.parent.resolve(strict=True)
    require(requested.name not in {"", ".", ".."}, "publication directory name")
    output = parent / requested.name
    require(not output.exists() and not output.is_symlink(), "publication target must not exist")
    require(
        set(records) == {
            "MOMENT_CONTRACT.json",
            "PUBLICATION.json",
            "RUNTIME_PINS.json",
            "SOURCE_AUTHORIZATION.json",
        },
        "publication record set",
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=parent))
    _write_new(stage / "INCOMPLETE", b"UWFA-SC-v9 source moments incomplete\n")
    member_rows: list[dict[str, Any]] = []
    for name in sorted(records):
        payload = records[name]
        require(isinstance(payload, bytes), f"publication bytes {name}")
        _write_new(stage / name, payload)
        member_rows.append({"name": name, "bytes": len(payload), "sha256": sha256_bytes(payload)})
    complete = sealed(
        {
            "schema": COMPLETE_SCHEMA,
            "status": "COMPLETE_AUTHENTICATED_SOURCE_MOMENTS_NONPROMOTING",
            "members": member_rows,
            "members_root_sha256": _members_root(member_rows),
            "payload_files_in_publication": 0,
            "positive_claim_authority": False,
        },
        "complete_sha256",
    )
    _write_new(stage / "COMPLETE.json", pretty_json(complete))
    (stage / "INCOMPLETE").unlink()
    os.rename(stage, output)
    return complete


def publish_authenticated_contract(
    *,
    package_root: Path,
    expected_source_manifest_sha256: str,
    np: Any,
    source_root: Path,
    authorization_bytes: bytes,
    expected_authorization_file_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Reviewed-dispatcher API.  There is intentionally no executable CLI.

    Order is fail-closed: authenticate package, external authorization, and
    exact runtime; authenticate *all* source bytes; compute all moments; build
    and revalidate the producer ABI; only then atomically publish an absent
    directory.  No partial final directory and no payload copy is produced.
    """
    package = Path(package_root).resolve(strict=True)
    verify_source_package(package, expected_source_manifest_sha256)
    authorization = parse_external_authorization(
        authorization_bytes, expected_authorization_file_sha256
    )
    runtime_pins, runtime_pins_bytes = load_runtime_pins(package)
    validate_exact_runtime(np, runtime_pins)
    payloads = load_authenticated_payloads(source_root, authorization)
    moment_contract = build_moment_contract(np, authorization, payloads)
    moment_bytes = pretty_json(moment_contract)
    publication = sealed(
        {
            "schema": PUBLICATION_SCHEMA,
            "status": "READY_FOR_ATOMIC_NONPROMOTING_PUBLICATION",
            "source_manifest_sha256": expected_source_manifest_sha256,
            "runtime_pins_sha256": sha256_bytes(runtime_pins_bytes),
            "authorization_file_sha256": expected_authorization_file_sha256,
            "authorization_sha256": authorization["authorization_sha256"],
            "source_set_sha256": authorization["source_set_sha256"],
            "moment_contract_file_sha256": sha256_bytes(moment_bytes),
            "moment_contract_sha256": moment_contract["moment_contract_sha256"],
            "matrices": MATRIX_COUNT,
            "source_values": TOTAL_VALUES,
            "source_bytes_authenticated": TOTAL_SOURCE_BYTES,
            "payload_files_copied": 0,
            "runtime_identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
            "positive_claim_authority": False,
        },
        "publication_sha256",
    )
    publication_bytes = pretty_json(publication)
    complete = _atomic_publish_records(
        output_root,
        {
            "MOMENT_CONTRACT.json": moment_bytes,
            "PUBLICATION.json": publication_bytes,
            "RUNTIME_PINS.json": runtime_pins_bytes,
            "SOURCE_AUTHORIZATION.json": authorization_bytes,
        },
    )
    return {
        "output_root": str(Path(output_root).resolve(strict=True)),
        "moment_contract_sha256": moment_contract["moment_contract_sha256"],
        "publication_sha256": publication["publication_sha256"],
        "complete_sha256": complete["complete_sha256"],
    }


DIRECT_STATUS = (
    "BLOCKED_SOURCE_ONLY: import publish_authenticated_contract from an "
    "externally pinned reviewed dispatcher; direct execution has no payload authority"
)


def direct_main() -> int:
    print(DIRECT_STATUS, file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(direct_main())
