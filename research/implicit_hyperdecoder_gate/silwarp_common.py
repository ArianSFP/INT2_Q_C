"""Source-free reference primitives for the frozen SILWARP auxiliary gate.

This module deliberately performs no filesystem inventory or CUDA import at
module import time.  The production runner imports CuPy only after its explicit
payload command has passed the closed protocol and path firewall.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol_lock.json"
SOURCE_LOCK_PATH = HERE / "source_lock.json"
SCHEMA = "silwarp_auxiliary_protocol_v1"
SOURCE_LOCK_SCHEMA = "silwarp_auxiliary_source_lock_v1"
SENTINEL_SCHEMA = "silwarp_launch_sentinel_v1"
AUTHORIZATION_PHRASE = "AUTHORIZE_SILWARP_AUXILIARY_IDEAL_CHANNEL_RUN_V1"
MODEL_MAGIC = "SILWARP_MODEL_V1"
EXPERT_MAGIC = b"SILWEX1\0"
MODEL_HEADER_BYTES = 4096
EXPERT_HEADER_BYTES = 64
MOMENT_BYTES = 12
ROLE_ORDER = ("gate", "up", "down")
ROLE_SCALAR = {"up": -1.0, "gate": 0.0, "down": 1.0}
ROLE_INDEX = {name: index for index, name in enumerate(ROLE_ORDER)}
MATRIX_ROWS = 768
MATRIX_COLS = 2048
TILE_SIDE = 16
TILE_VALUES = TILE_SIDE * TILE_SIDE
TILE_ROWS = MATRIX_ROWS // TILE_SIDE
TILE_COLS = MATRIX_COLS // TILE_SIDE
WEIGHTS_PER_ROLE = MATRIX_ROWS * MATRIX_COLS
WEIGHTS_PER_EXPERT = 3 * WEIGHTS_PER_ROLE
CORE_RATE_BPW = 2.15
CHANNEL_D = 2.0 ** (-2.0 * CORE_RATE_BPW)
_CHANNEL_A_EXACT = 1.0 - CHANNEL_D
CHANNEL_A_FP32 = np.float32(_CHANNEL_A_EXACT)
if float(CHANNEL_A_FP32) > _CHANNEL_A_EXACT:
    CHANNEL_A_FP32 = np.nextafter(
        CHANNEL_A_FP32, np.float32(-np.inf), dtype=np.float32
    )
CHANNEL_SIGMA_NEAREST_FP32 = np.float32(math.sqrt(CHANNEL_D * (1.0 - CHANNEL_D)))
# One upward ULP makes the information bound conservative after FP32 rounding.
CHANNEL_SIGMA_FP32 = np.nextafter(
    CHANNEL_SIGMA_NEAREST_FP32, np.float32(np.inf), dtype=np.float32
)
TARGET_F = 0.8
PRODUCTION_EXPERTS = 128
PARAMETER_ORDER = (
    "Wy",
    "Wc",
    "b0",
    "A",
    "ba",
    "C",
    "bc",
    "B",
    "bb",
    "Wo",
    "bo",
    "role_gain",
)
PARAMETER_SHAPES = {
    "Wy": (256, 256),
    "Wc": (21, 256),
    "b0": (256,),
    "A": (256, 128),
    "ba": (128,),
    "C": (256, 128),
    "bc": (128,),
    "B": (128, 256),
    "bb": (256,),
    "Wo": (256, 256),
    "bo": (256,),
    "role_gain": (3,),
}
PARAMETER_COUNT = sum(math.prod(shape) for shape in PARAMETER_SHAPES.values())
MODEL_PARAMETER_BYTES = 2 * PARAMETER_COUNT
MODEL_TOTAL_BYTES = MODEL_HEADER_BYTES + MODEL_PARAMETER_BYTES
ROLE_PAYLOAD_BYTES = math.ceil(WEIGHTS_PER_ROLE * CORE_RATE_BPW / 8.0)
ROLE_PAYLOAD_BPW = 8.0 * ROLE_PAYLOAD_BYTES / WEIGHTS_PER_ROLE
EXPERT_PAYLOAD_BYTES = 3 * ROLE_PAYLOAD_BYTES
EXPERT_LOCAL_BYTES = EXPERT_HEADER_BYTES + MOMENT_BYTES
TRAINING_SEEDS = (26090131, 26090179)
CONTROL_NAMES = ("null_a", "null_b")
CHECKPOINTS = (256, 512, 1536)

TENSOR_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.weight\.bf16\.bin$"
)


@dataclass(frozen=True)
class SmallSpec:
    """Shape-generic test specification; production uses the frozen constants."""

    values: int
    features: int
    hidden: int
    bottleneck: int
    steps: int
    roles: int = 3


PRODUCTION_SPEC = SmallSpec(256, 21, 256, 128, 6, 3)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_regular_file_once(path: Path, maximum_bytes: int = 16 << 20) -> bytes:
    """Read one regular non-symlink descriptor, with identity/size rechecks."""

    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"file must be regular and non-symlink: {path}")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise ValueError(f"file exceeds closed read limit: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"opened file is not regular: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"file identity changed during open: {path}")
        if opened.st_size != before.st_size or opened.st_size > maximum_bytes:
            raise ValueError(f"file size changed during open: {path}")
        remaining = opened.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"file truncated during read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"file grew during read: {path}")
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
        ):
            raise ValueError(f"file changed during read: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise AssertionError("single-descriptor read length mismatch")
    return payload


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = json.loads(read_regular_file_once(path).decode("utf-8"))
    if protocol.get("schema") != SCHEMA:
        raise ValueError("SILWARP protocol schema mismatch")
    if protocol.get("status") != "FROZEN_BEFORE_PAYLOAD_OR_CUDA":
        raise ValueError("SILWARP protocol is not frozen")
    return protocol


def protocol_sha256(path: Path = PROTOCOL_PATH) -> str:
    return sha256_bytes(read_regular_file_once(path))


def load_source_lock(
    path: Path = SOURCE_LOCK_PATH,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = load_protocol() if protocol is None else protocol
    payload = read_regular_file_once(path)
    expected_hash = protocol["source_authentication"]["source_lock_sha256"]
    if sha256_bytes(payload) != expected_hash:
        raise ValueError("source lock file hash mismatch")
    lock = json.loads(payload.decode("utf-8"))
    if lock.get("schema") != SOURCE_LOCK_SCHEMA:
        raise ValueError("source lock schema mismatch")
    seal = lock.get("internal_seal_sha256")
    unsigned = dict(lock)
    unsigned.pop("internal_seal_sha256", None)
    if seal != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("source lock internal seal mismatch")
    if seal != protocol["source_authentication"]["source_lock_internal_seal_sha256"]:
        raise ValueError("source lock seal/protocol mismatch")
    checkpoint = lock.get("checkpoint", {})
    if checkpoint.get("revision") != protocol["source_authentication"]["required_revision"]:
        raise ValueError("source lock revision mismatch")
    if checkpoint.get("repo") != protocol["source"]["checkpoint"]:
        raise ValueError("source lock checkpoint identity mismatch")
    if lock.get("dtype") != protocol["source"]["dtype"]:
        raise ValueError("source lock dtype mismatch")
    if int(lock.get("canonical_file_bytes", -1)) != 2 * WEIGHTS_PER_ROLE:
        raise ValueError("source lock canonical byte count mismatch")
    rows = lock.get("files")
    if not isinstance(rows, list) or len(rows) != 116:
        raise ValueError("source lock file-count mismatch")
    if int(lock.get("file_count", -1)) != len(rows):
        raise ValueError("source lock declared file-count mismatch")
    names = [row.get("filename") for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("duplicate source-lock filename")
    return lock


def source_lock_sha256(path: Path = SOURCE_LOCK_PATH) -> str:
    return sha256_bytes(read_regular_file_once(path))


def require_finite_scalar(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite scalar: {name}")
    return result


def require_all_finite(name: str, values: Any, xp: Any = np) -> None:
    array = xp.asarray(values)
    if not bool(xp.all(xp.isfinite(array)).item()):
        raise FloatingPointError(f"nonfinite array: {name}")


def require_json_finite(name: str, value: Any) -> None:
    """Reject non-finite numbers anywhere in a JSON-like result tree."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            require_json_finite(f"{name}.{key}", child)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_json_finite(f"{name}[{index}]", child)
    elif isinstance(value, (float, np.floating)):
        require_finite_scalar(name, value)


def spec_shapes(spec: SmallSpec) -> dict[str, tuple[int, ...]]:
    return {
        "Wy": (spec.values, spec.hidden),
        "Wc": (spec.features, spec.hidden),
        "b0": (spec.hidden,),
        "A": (spec.hidden, spec.bottleneck),
        "ba": (spec.bottleneck,),
        "C": (spec.hidden, spec.bottleneck),
        "bc": (spec.bottleneck,),
        "B": (spec.bottleneck, spec.hidden),
        "bb": (spec.hidden,),
        "Wo": (spec.hidden, spec.values),
        "bo": (spec.values,),
        "role_gain": (spec.roles,),
    }


def validate_frozen_constants(protocol: Mapping[str, Any] | None = None) -> None:
    protocol = load_protocol() if protocol is None else protocol
    arch = protocol["architecture"]
    expected_shapes = {key: list(value) for key, value in PARAMETER_SHAPES.items()}
    if arch["parameter_order"] != list(PARAMETER_ORDER):
        raise AssertionError("frozen parameter order mismatch")
    if arch["parameter_shapes"] != expected_shapes:
        raise AssertionError("frozen parameter shapes mismatch")
    if arch["parameter_count"] != PARAMETER_COUNT or PARAMETER_COUNT != 235779:
        raise AssertionError("frozen parameter count mismatch")
    ledger = production_ledger(PRODUCTION_EXPERTS)
    frozen = protocol["ledger"]
    checks = {
        "weights_per_role": WEIGHTS_PER_ROLE,
        "weights_per_expert": WEIGHTS_PER_EXPERT,
        "role_payload_bytes": ROLE_PAYLOAD_BYTES,
        "expert_payload_bytes": EXPERT_PAYLOAD_BYTES,
        "expert_header_bytes": EXPERT_HEADER_BYTES,
        "expert_moment_bytes": MOMENT_BYTES,
        "decoder_parameter_bytes": MODEL_PARAMETER_BYTES,
        "decoder_header_bytes": MODEL_HEADER_BYTES,
        "decoder_total_bytes": MODEL_TOTAL_BYTES,
    }
    for name, expected in checks.items():
        if frozen[name] != expected:
            raise AssertionError(f"frozen ledger mismatch: {name}")
    floating_checks = {
        "role_payload_bpw": ROLE_PAYLOAD_BPW,
        "production_physical_bpw": ledger["production_physical_bpw"],
        "production_cold_read_amplification": ledger[
            "production_cold_read_amplification"
        ],
        "represented_identity_error_variance": channel_second_moments()[
            "error_variance"
        ],
        "represented_information_upper_bound_bpw": channel_second_moments()[
            "information_upper_bound_bpw"
        ],
    }
    for name, expected in floating_checks.items():
        if not math.isclose(frozen[name], expected, rel_tol=0.0, abs_tol=5e-15):
            raise AssertionError(f"frozen floating ledger mismatch: {name}")
    information = protocol["information_channel"]
    channel = channel_second_moments()
    information_checks = {
        "represented_coefficient_fp32": channel["coefficient"],
        "represented_noise_sigma_fp32": channel["noise_sigma"],
        "represented_noise_variance": channel["noise_variance"],
        "represented_identity_error_variance": channel["error_variance"],
        "information_upper_bound_bpw": channel["information_upper_bound_bpw"],
        "physical_role_payload_bpw": ROLE_PAYLOAD_BPW,
    }
    for name, expected in information_checks.items():
        if not math.isclose(
            information[name], expected, rel_tol=0.0, abs_tol=5e-15
        ):
            raise AssertionError(f"frozen information-channel mismatch: {name}")
    if channel["information_upper_bound_bpw"] > CORE_RATE_BPW:
        raise AssertionError("implemented channel exceeds core information rate")
    if not math.isclose(
        protocol["objective"]["required_absolute_s"],
        ledger["required_absolute_s"],
        rel_tol=0.0,
        abs_tol=5e-15,
    ):
        raise AssertionError("frozen objective threshold mismatch")


def production_ledger(amortization_experts: int = PRODUCTION_EXPERTS) -> dict[str, Any]:
    if amortization_experts <= 0:
        raise ValueError("amortization expert count must be positive")
    attributed = (
        EXPERT_PAYLOAD_BYTES
        + EXPERT_LOCAL_BYTES
        + MODEL_TOTAL_BYTES / amortization_experts
    )
    cold = EXPERT_PAYLOAD_BYTES + EXPERT_LOCAL_BYTES + MODEL_TOTAL_BYTES
    physical = 8.0 * attributed / WEIGHTS_PER_EXPERT
    read_amp = cold / attributed
    identity_error = channel_second_moments()["error_variance"]
    identity_f = identity_error * 2.0 ** (2.0 * physical)
    required_s = 0.5 * math.log2(identity_f / TARGET_F)
    target_mse = TARGET_F * 2.0 ** (-2.0 * physical)
    return {
        "amortization_experts": amortization_experts,
        "weights_per_expert": WEIGHTS_PER_EXPERT,
        "role_payload_bytes": ROLE_PAYLOAD_BYTES,
        "role_payload_bpw": ROLE_PAYLOAD_BPW,
        "expert_payload_bytes": EXPERT_PAYLOAD_BYTES,
        "expert_local_bytes": EXPERT_LOCAL_BYTES,
        "decoder_total_bytes": MODEL_TOTAL_BYTES,
        "attributed_bytes_per_expert": attributed,
        "production_physical_bpw": physical,
        "production_cold_expert_bytes": cold,
        "production_cold_read_amplification": read_amp,
        "required_absolute_s": required_s,
        "target_mse": target_mse,
        "identity_F": identity_f,
    }


def information_upper_bound_bpw(distortion: float) -> float:
    if not 0.0 < distortion < 1.0:
        raise ValueError("distortion must lie in (0,1)")
    return 0.5 * math.log2(1.0 / distortion)


def implemented_channel_constants(distortion: float = CHANNEL_D) -> tuple[float, float]:
    distortion = require_finite_scalar("channel distortion", distortion)
    if not 0.0 < distortion < 1.0:
        raise ValueError("distortion must lie in (0,1)")
    if distortion == CHANNEL_D:
        return float(CHANNEL_A_FP32), float(CHANNEL_SIGMA_FP32)
    exact_coefficient = 1.0 - distortion
    coefficient = np.float32(exact_coefficient)
    if float(coefficient) > exact_coefficient:
        coefficient = np.nextafter(
            coefficient, np.float32(-np.inf), dtype=np.float32
        )
    sigma_nearest = np.float32(math.sqrt(distortion * (1.0 - distortion)))
    sigma = np.nextafter(sigma_nearest, np.float32(np.inf), dtype=np.float32)
    return float(coefficient), float(sigma)


def channel_second_moments(distortion: float = CHANNEL_D) -> dict[str, float]:
    a, sigma = implemented_channel_constants(distortion)
    noise_variance = sigma * sigma
    y_variance = a * a + noise_variance
    error_variance = (1.0 - a) ** 2 + noise_variance
    implemented_bound = 0.5 * math.log2(y_variance / noise_variance)
    return {
        "coefficient": a,
        "noise_sigma": sigma,
        "noise_variance": noise_variance,
        "y_variance": y_variance,
        "error_variance": error_variance,
        "analytic_design_information_bpw": information_upper_bound_bpw(distortion),
        "information_upper_bound_bpw": implemented_bound,
        "physical_role_payload_bpw": ROLE_PAYLOAD_BPW,
    }


def upward_fp16_moments(values: Any) -> dict[str, Any]:
    """Serialize a mean and an RMS that cannot understate second moment.

    The mean is rounded first.  The centered RMS is then measured about that
    serialized mean and rounded toward positive infinity in the FP16 value
    set.  This ordering is part of the information-validity proof.
    """

    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("cannot serialize moments of an empty array")
    source = array.astype(np.float64, copy=False).reshape(-1)
    if not np.all(np.isfinite(source)):
        raise ValueError("nonfinite source moment input")
    exact_mean = float(np.mean(source, dtype=np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        mean16 = np.float16(exact_mean)
    if not np.isfinite(mean16):
        raise OverflowError("FP16 mean overflow")
    centered = source - float(mean16)
    rms64 = math.sqrt(float(np.mean(centered * centered, dtype=np.float64)))
    if not math.isfinite(rms64):
        raise ValueError("nonfinite centered RMS")
    minimum_positive = np.nextafter(
        np.float16(0.0), np.float16(np.inf), dtype=np.float16
    )
    if rms64 == 0.0:
        rms16 = minimum_positive
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            rms16 = np.float16(rms64)
        if not np.isfinite(rms16):
            raise OverflowError("FP16 RMS overflow")
        if float(rms16) < rms64:
            with np.errstate(over="ignore", invalid="ignore"):
                rms16 = np.nextafter(
                    rms16, np.float16(np.inf), dtype=np.float16
                )
        if not np.isfinite(rms16) or float(rms16) < rms64:
            raise OverflowError("no finite upward FP16 RMS")
        if rms16 <= 0:
            rms16 = minimum_positive
    precast_second_moment = float(
        np.mean((centered / float(rms16)) ** 2, dtype=np.float64)
    )
    # The decoder consumes FP32 normalized values.  FP32 rounding can increase
    # the empirical second moment even when the pre-cast FP64 quotient is <=1,
    # so advance through the FP16 lattice until the actual array is safe.
    while True:
        normalized_fp32 = (centered / float(rms16)).astype(np.float32)
        normalized_second_moment = float(
            np.mean(normalized_fp32.astype(np.float64) ** 2, dtype=np.float64)
        )
        if normalized_second_moment <= 1.0:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            rms16 = np.nextafter(rms16, np.float16(np.inf), dtype=np.float16)
        if not np.isfinite(rms16):
            raise OverflowError("no finite FP16 RMS satisfying FP32 unit moment")
    return {
        "exact_mean_fp64": exact_mean,
        "serialized_mean_fp16": mean16,
        "centered_rms_fp64": rms64,
        "serialized_rms_fp16": rms16,
        "precast_normalized_second_moment_fp64": precast_second_moment,
        "normalized_second_moment_fp64": normalized_second_moment,
    }


def normalize_with_serialized_moments(
    values: Any, moments: Mapping[str, Any]
) -> np.ndarray:
    mean = float(np.float16(moments["serialized_mean_fp16"]))
    rms = float(np.float16(moments["serialized_rms_fp16"]))
    if not math.isfinite(mean) or not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("invalid serialized normalization moments")
    normalized = (
        (np.asarray(values, dtype=np.float64) - mean) / rms
    ).astype(np.float32)
    second_moment = float(
        np.mean(normalized.reshape(-1).astype(np.float64) ** 2, dtype=np.float64)
    )
    if second_moment > 1.0:
        raise AssertionError("serialized normalization exceeds unit second moment")
    return normalized


def split_sets(protocol: Mapping[str, Any] | None = None) -> dict[str, set[tuple[int, int]]]:
    protocol = load_protocol() if protocol is None else protocol
    split = protocol["split"]
    return {
        "fit": {tuple(map(int, row)) for row in split["fit_pairs"]},
        "calibration": {
            tuple(map(int, row)) for row in split["calibration_pairs"]
        },
        "confirmation": {
            tuple(map(int, row)) for row in split["confirmation_pairs"]
        },
    }


def validate_split(protocol: Mapping[str, Any] | None = None) -> None:
    protocol = load_protocol() if protocol is None else protocol
    sets = split_sets(protocol)
    names = tuple(sets)
    for i, left_name in enumerate(names):
        left = sets[left_name]
        if len(left) != len(protocol["split"][f"{left_name}_pairs"]):
            raise AssertionError(f"duplicate pair in {left_name}")
        left_layers = {layer for layer, _ in left}
        left_experts = {expert for _, expert in left}
        for right_name in names[i + 1 :]:
            right = sets[right_name]
            if left & right:
                raise AssertionError(f"pair overlap: {left_name}/{right_name}")
            if left_layers & {layer for layer, _ in right}:
                raise AssertionError(f"layer overlap: {left_name}/{right_name}")
            if left_experts & {expert for _, expert in right}:
                raise AssertionError(f"expert overlap: {left_name}/{right_name}")
    target_layers = set(protocol["pinned_panel"]["target_layers_excluded_before_split"])
    target_experts = set(protocol["pinned_panel"]["target_experts_excluded_before_split"])
    for name, pairs in sets.items():
        for layer, expert in pairs:
            if layer in target_layers or expert in target_experts:
                raise AssertionError(f"target identity leaked into {name}")


def reject_forbidden_path(path: Path, protocol: Mapping[str, Any] | None = None) -> Path:
    protocol = load_protocol() if protocol is None else protocol
    resolved = path.expanduser().resolve()
    lowered = {part.lower() for part in resolved.parts}
    forbidden = {
        str(part).lower()
        for part in protocol["pinned_panel"]["forbidden_path_components"]
    }
    overlap = lowered & forbidden
    if overlap:
        raise ValueError(f"forbidden source path component(s): {sorted(overlap)}")
    return resolved


def parse_tensor_name(path: Path) -> tuple[int, int, str]:
    match = TENSOR_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected tensor filename: {path.name}")
    layer, expert, role = match.groups()
    return int(layer), int(expert), role


def expected_split_keys(protocol: Mapping[str, Any] | None = None) -> dict[str, set[tuple[int, int, str]]]:
    protocol = load_protocol() if protocol is None else protocol
    pairs = split_sets(protocol)
    result = {
        name: {(layer, expert, role) for layer, expert in values for role in ("up", "down")}
        for name, values in pairs.items()
    }
    result["confirmation"] |= {
        (int(layer), int(expert), "gate")
        for layer, expert in protocol["split"]["confirmation_gate_pairs"]
    }
    return result


def validate_source_lock(
    protocol: Mapping[str, Any] | None = None,
    lock: Mapping[str, Any] | None = None,
) -> None:
    protocol = load_protocol() if protocol is None else protocol
    lock = load_source_lock(protocol=protocol) if lock is None else lock
    expected = expected_split_keys(protocol)
    expected_by_name = {}
    for split, keys in expected.items():
        for layer, expert, role in keys:
            name = (
                f"model.layers.{layer}.mlp.experts.{expert}."
                f"{role}_proj.weight.bf16.bin"
            )
            expected_by_name[name] = split
    rows = lock["files"]
    actual_by_name = {str(row["filename"]): str(row["split"]) for row in rows}
    if actual_by_name != expected_by_name:
        raise ValueError("source lock identities/splits differ from frozen protocol")
    for row in rows:
        if int(row.get("nbytes", -1)) != 2 * WEIGHTS_PER_ROLE:
            raise ValueError("source lock byte length mismatch")
        digest = str(row.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError("source lock SHA-256 shape mismatch")
        bytes.fromhex(digest)
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("fit", "calibration", "confirmation")
    }
    if counts != lock["split_counts"]:
        raise ValueError("source lock split counts mismatch")


def source_lock_rows(
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    lock = load_source_lock(protocol=protocol)
    validate_source_lock(protocol, lock)
    return {str(row["filename"]): dict(row) for row in lock["files"]}


def read_authenticated_locked_file(
    path: Path, row: Mapping[str, Any]
) -> tuple[bytes, str]:
    """Authenticate and return one descriptor's bytes, closing hash/decode TOCTOU."""

    if path.name != row["filename"]:
        raise ValueError("source basename/lock mismatch")
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"opened source is not regular: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"source identity changed during open: {path}")
        expected_size = int(row["nbytes"])
        if opened.st_size != expected_size:
            raise ValueError(f"source byte length mismatch: {path}")
        hasher = hashlib.sha256()
        chunks = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"source truncated during authenticated read: {path}")
            chunks.append(chunk)
            hasher.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"source grew during authenticated read: {path}")
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
        ):
            raise ValueError(f"source changed during authenticated read: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != int(row["nbytes"]):
        raise AssertionError("authenticated source assembly length mismatch")
    digest = hasher.hexdigest()
    if digest != row["sha256"]:
        raise ValueError(f"source SHA-256 mismatch: {path}")
    return payload, digest


def authenticate_locked_file(path: Path, row: Mapping[str, Any]) -> str:
    """Compatibility wrapper for authentication-only callers."""

    _, digest = read_authenticated_locked_file(path, row)
    return digest


def inventory_auxiliary(
    directory: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, dict[tuple[int, int, str], Path]]:
    protocol = load_protocol() if protocol is None else protocol
    expanded = directory.expanduser()
    if expanded.is_symlink():
        raise ValueError("auxiliary directory may not be a symlink")
    directory = reject_forbidden_path(expanded, protocol)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    expected = expected_split_keys(protocol)
    locked_rows = source_lock_rows(protocol)
    wanted = set().union(*expected.values())
    found: dict[tuple[int, int, str], Path] = {}
    for path in sorted(directory.glob("*.bf16.bin")):
        try:
            key = parse_tensor_name(path)
        except ValueError:
            continue
        if key in wanted:
            if key in found:
                raise ValueError(f"duplicate tensor identity: {key}")
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source is not a regular non-symlink file: {path}")
            row = locked_rows[path.name]
            if path.stat().st_size != int(row["nbytes"]):
                raise ValueError(f"wrong tensor byte length: {path}")
            found[key] = path
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"missing exact frozen auxiliary identities: {sorted(missing)}")
    return {
        split: {key: found[key] for key in sorted(keys)}
        for split, keys in expected.items()
    }


def coordinate_features(
    layer: int,
    expert: int,
    role: str,
    tile_row: Any,
    tile_col: Any,
    serialized_rms: float,
    log_rms_center: float,
    log_rms_scale: float,
    xp: Any = np,
) -> Any:
    """Build only decoder-visible features; no source tile is an argument."""

    if role not in ROLE_SCALAR:
        raise ValueError(f"unknown role: {role}")
    serialized_rms = require_finite_scalar("serialized RMS feature", serialized_rms)
    log_rms_center = require_finite_scalar("log-RMS center feature", log_rms_center)
    log_rms_scale = require_finite_scalar("log-RMS scale feature", log_rms_scale)
    if not serialized_rms > 0.0 or not log_rms_scale > 0.0:
        raise ValueError("invalid serialized RMS normalization")
    row = xp.asarray(tile_row, dtype=xp.float32)
    col = xp.asarray(tile_col, dtype=xp.float32)
    row, col = xp.broadcast_arrays(row, col)
    rn = (row + xp.float32(0.5)) / xp.float32(TILE_ROWS)
    cn = (col + xp.float32(0.5)) / xp.float32(TILE_COLS)
    layer_n = xp.float32((float(layer) - 23.5) / 23.5)
    expert_n = xp.float32((float(expert) - 63.5) / 63.5)
    log_rms_n = xp.float32(
        (math.log(float(serialized_rms)) - float(log_rms_center))
        / float(log_rms_scale)
    )
    columns = [
        xp.full_like(rn, xp.float32(ROLE_SCALAR[role])),
        xp.full_like(rn, layer_n),
        xp.full_like(rn, expert_n),
        xp.full_like(rn, log_rms_n),
        xp.float32(2.0) * rn - xp.float32(1.0),
        xp.float32(2.0) * cn - xp.float32(1.0),
    ]
    for frequency in (1.0, 2.0, 4.0):
        angle_r = xp.float32(2.0 * math.pi * frequency) * rn
        angle_c = xp.float32(2.0 * math.pi * frequency) * cn
        columns.extend(
            [xp.sin(angle_r), xp.cos(angle_r), xp.sin(angle_c), xp.cos(angle_c)]
        )
    layer_angle = xp.float32(math.pi) * layer_n
    columns.extend(
        [
            xp.full_like(rn, xp.sin(layer_angle)),
            xp.full_like(rn, xp.cos(layer_angle)),
            xp.full_like(rn, layer_n * layer_n - xp.float32(1.0 / 3.0)),
        ]
    )
    result = xp.stack(columns, axis=-1).astype(xp.float32, copy=False)
    if result.shape[-1] != 21:
        raise AssertionError("coordinate feature count mismatch")
    require_all_finite("coordinate features", result, xp)
    return result


def derive_seed(domain: str, *parts: Any) -> int:
    text = "|".join(["SILWARP-v1", domain, *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _counter_words(count: int, domain: str, *parts: Any, xp: Any = np) -> Any:
    """SplitMix64 PRF words indexed explicitly by counter, with no RNG state."""

    if count < 0:
        raise ValueError("negative counter length")
    seed = derive_seed(domain, *parts)
    with np.errstate(over="ignore"):
        value = xp.arange(count, dtype=xp.uint64) + xp.uint64(seed)
        value = value + xp.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> 30)) * xp.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> 27)) * xp.uint64(0x94D049BB133111EB)
        value = value ^ (value >> 31)
    return value


def counter_indices(
    high: int,
    size: int,
    domain: str,
    *parts: Any,
    xp: Any = np,
) -> Any:
    """Deterministic counter-indexed samples in ``[0, high)``."""

    if high <= 0 or size < 0:
        raise ValueError("invalid counter-index request")
    words = _counter_words(size, domain, *parts, xp=xp)
    return (words % xp.uint64(high)).astype(xp.int64, copy=False)


def counter_standard_normal(
    shape: tuple[int, ...],
    domain: str,
    *parts: Any,
    xp: Any = np,
    float64: bool = False,
) -> Any:
    """Box-Muller normals from explicit PRF counters, never mutable RNG state."""

    count = math.prod(shape)
    if count < 0:
        raise ValueError("invalid normal shape")
    pairs = (count + 1) // 2
    words_a = _counter_words(pairs, f"{domain}:u1", *parts, xp=xp)
    words_b = _counter_words(pairs, f"{domain}:u2", *parts, xp=xp)
    dtype = xp.float64 if float64 else xp.float32
    if float64:
        scale = dtype(1.0 / float(1 << 53))
        u1 = ((words_a >> 11).astype(dtype) + dtype(0.5)) * scale
        u2 = ((words_b >> 11).astype(dtype) + dtype(0.5)) * scale
    else:
        scale = dtype(1.0 / float(1 << 24))
        u1 = ((words_a >> 40).astype(dtype) + dtype(0.5)) * scale
        u2 = ((words_b >> 40).astype(dtype) + dtype(0.5)) * scale
    radius = xp.sqrt(dtype(-2.0) * xp.log(u1))
    angle = dtype(2.0 * math.pi) * u2
    result = xp.empty(2 * pairs, dtype=dtype)
    result[0::2] = radius * xp.cos(angle)
    result[1::2] = radius * xp.sin(angle)
    result = result[:count].reshape(shape)
    require_all_finite(f"counter normal {domain}", result, xp)
    return result


def build_launch_sentinel(
    runner_path: Path,
    common_path: Path | None = None,
) -> dict[str, Any]:
    common_path = Path(__file__).resolve() if common_path is None else common_path
    sentinel = {
        "schema": SENTINEL_SCHEMA,
        "authorization_phrase": AUTHORIZATION_PHRASE,
        "protocol_sha256": protocol_sha256(),
        "source_lock_sha256": source_lock_sha256(),
        "runner_sha256": sha256_file(runner_path.resolve()),
        "common_sha256": sha256_file(common_path.resolve()),
        "pinned_panel_authorized": False,
        "confirmation_numeric_access_before_promotion": False,
    }
    sentinel["internal_seal_sha256"] = sha256_bytes(canonical_json_bytes(sentinel))
    return sentinel


def validate_launch_sentinel(
    path: Path,
    runner_path: Path,
    common_path: Path | None = None,
) -> dict[str, Any]:
    sentinel = json.loads(read_regular_file_once(path).decode("utf-8"))
    seal = sentinel.get("internal_seal_sha256")
    unsigned = dict(sentinel)
    unsigned.pop("internal_seal_sha256", None)
    if seal != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("launch sentinel internal seal mismatch")
    expected = build_launch_sentinel(runner_path, common_path)
    if sentinel != expected:
        raise ValueError("launch sentinel bindings mismatch")
    return sentinel


def gaussian_rdf_channel(x: Any, noise: Any, xp: Any = np, distortion: float = CHANNEL_D) -> Any:
    """Training channel; counter-PRF FP32 noise is not used for MI claims."""

    x = xp.asarray(x, dtype=xp.float32)
    noise = xp.asarray(noise, dtype=xp.float32)
    if x.shape != noise.shape:
        raise ValueError("source/noise shape mismatch")
    require_all_finite("channel source", x, xp)
    require_all_finite("channel noise", noise, xp)
    distortion = require_finite_scalar("channel distortion", distortion)
    if not 0.0 < distortion < 1.0:
        raise ValueError("channel distortion must lie strictly between zero and one")
    coefficient_value, sigma_value = implemented_channel_constants(distortion)
    coefficient = xp.float32(coefficient_value)
    sigma = xp.float32(sigma_value)
    result = coefficient * x + sigma * noise
    require_all_finite("channel output", result, xp)
    return result


def ideal_awgn_mc_channel(
    x: Any, standard_normal_fp64: Any, distortion: float = CHANNEL_D
) -> np.ndarray:
    """Proof-aligned MC draw: conceptual FP64 AWGN followed by one Q32 cast.

    The supplied normal array is a reproducible Monte Carlo approximation to
    iid N(0,1); the mutual-information statement is about the mathematical
    ideal normal law.  The final cast is deterministic data processing.
    """

    source = np.asarray(x, dtype=np.float32)
    noise = np.asarray(standard_normal_fp64, dtype=np.float64)
    if source.shape != noise.shape:
        raise ValueError("source/noise shape mismatch")
    require_all_finite("ideal MC source", source)
    require_all_finite("ideal MC noise", noise)
    coefficient, sigma = implemented_channel_constants(distortion)
    y64 = coefficient * source.astype(np.float64) + sigma * noise
    require_all_finite("ideal MC prequantized output", y64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = y64.astype(np.float32)
    require_all_finite("ideal MC Q32 output", result)
    return result


def silu(x: Any, xp: Any) -> Any:
    return x / (xp.float32(1.0) + xp.exp(-x))


def silu_derivative(x: Any, xp: Any) -> Any:
    sigmoid = xp.float32(1.0) / (xp.float32(1.0) + xp.exp(-x))
    return sigmoid * (xp.float32(1.0) + x * (xp.float32(1.0) - sigmoid))


def sigmoid(x: Any, xp: Any) -> Any:
    return xp.float32(1.0) / (xp.float32(1.0) + xp.exp(-x))


def initialize_parameters(
    seed: int,
    spec: SmallSpec = PRODUCTION_SPEC,
    xp: Any = np,
) -> dict[str, Any]:
    shapes = spec_shapes(spec)

    def normal(name: str, scale: float) -> Any:
        array = counter_standard_normal(
            shapes[name], "parameter-initialization", seed, name, xp=xp
        )
        return (array * xp.float32(scale)).astype(xp.float32, copy=False)

    params = {
        "Wy": normal("Wy", 1.0 / math.sqrt(spec.values)),
        "Wc": normal("Wc", 0.05 / math.sqrt(spec.features)),
        "b0": xp.zeros(shapes["b0"], dtype=xp.float32),
        "A": normal("A", 0.5 / math.sqrt(spec.hidden)),
        "ba": xp.zeros(shapes["ba"], dtype=xp.float32),
        "C": normal("C", 0.25 / math.sqrt(spec.hidden)),
        "bc": xp.zeros(shapes["bc"], dtype=xp.float32),
        "B": normal("B", 0.25 / math.sqrt(spec.bottleneck)),
        "bb": xp.zeros(shapes["bb"], dtype=xp.float32),
        # Exact identity and a live first gradient: Wo is zero, role gain is one.
        "Wo": xp.zeros(shapes["Wo"], dtype=xp.float32),
        "bo": xp.zeros(shapes["bo"], dtype=xp.float32),
        "role_gain": xp.ones(shapes["role_gain"], dtype=xp.float32),
    }
    return params


def forward(
    params: Mapping[str, Any],
    y: Any,
    features: Any,
    role_indices: Any,
    spec: SmallSpec = PRODUCTION_SPEC,
    xp: Any = np,
    return_cache: bool = False,
) -> Any:
    y = xp.asarray(y, dtype=xp.float32)
    features = xp.asarray(features, dtype=xp.float32)
    roles = xp.asarray(role_indices, dtype=xp.int64)
    if y.ndim != 2 or y.shape[1] != spec.values:
        raise ValueError("invalid SILWARP tile array")
    if features.shape != (y.shape[0], spec.features):
        raise ValueError("invalid SILWARP feature array")
    if roles.shape != (y.shape[0],):
        raise ValueError("invalid SILWARP role array")
    require_all_finite("decoder input Y", y, xp)
    require_all_finite("decoder features", features, xp)
    for name, value in params.items():
        require_all_finite(f"decoder parameter {name}", value, xp)
    pre0 = y @ params["Wy"] + features @ params["Wc"] + params["b0"]
    h = silu(pre0, xp)
    recurrent = []
    for _ in range(spec.steps):
        h_before = h
        pre_u = h_before @ params["A"] + params["ba"]
        pre_g = h_before @ params["C"] + params["bc"]
        u = silu(pre_u, xp)
        g = sigmoid(pre_g, xp)
        product = u * g
        delta = product @ params["B"] + params["bb"]
        h = h_before + delta
        if return_cache:
            recurrent.append((h_before, pre_u, pre_g, u, g, product))
    pre_out = h @ params["Wo"] + params["bo"]
    residual = xp.tanh(pre_out)
    gains = params["role_gain"][roles]
    decoded = y + gains[:, None] * residual
    require_all_finite("decoder output", decoded, xp)
    if not return_cache:
        return decoded
    cache = {
        "y": y,
        "features": features,
        "roles": roles,
        "pre0": pre0,
        "recurrent": recurrent,
        "h_final": h,
        "residual": residual,
        "gains": gains,
    }
    return decoded, cache


def mse_loss_and_gradients(
    params: Mapping[str, Any],
    y: Any,
    target: Any,
    features: Any,
    role_indices: Any,
    sample_weights: Any | None = None,
    spec: SmallSpec = PRODUCTION_SPEC,
    xp: Any = np,
) -> tuple[float, dict[str, Any]]:
    target = xp.asarray(target, dtype=xp.float32)
    decoded, cache = forward(
        params, y, features, role_indices, spec=spec, xp=xp, return_cache=True
    )
    if target.shape != decoded.shape:
        raise ValueError("target shape mismatch")
    error = decoded - target
    require_all_finite("training target", target, xp)
    require_all_finite("training error", error, xp)
    if sample_weights is None:
        weights = xp.ones(error.shape[0], dtype=xp.float32)
    else:
        weights = xp.asarray(sample_weights, dtype=xp.float32)
        if weights.shape != (error.shape[0],):
            raise ValueError("training sample-weight shape mismatch")
    require_all_finite("training sample weights", weights, xp)
    if bool(xp.any(weights < xp.float32(0.0)).item()):
        raise ValueError("training sample weights must be nonnegative")
    weight_sum = require_finite_scalar(
        "training sample-weight sum", xp.sum(weights.astype(xp.float64)).item()
    )
    if weight_sum <= 0.0:
        raise ValueError("training sample weights must have positive sum")
    denominator = weight_sum * error.shape[1]
    loss = require_finite_scalar(
        "training loss",
        (
            xp.sum(
                error.astype(xp.float64) ** 2 * weights.astype(xp.float64)[:, None]
            ).item()
            / denominator
        ),
    )
    ddecoded = (
        xp.float32(2.0 / denominator) * weights[:, None] * error
    )
    grads = {name: xp.zeros_like(value) for name, value in params.items()}
    residual = cache["residual"]
    roles = cache["roles"]
    for role in range(spec.roles):
        mask = roles == role
        if bool(xp.any(mask).item()):
            grads["role_gain"][role] = xp.sum(ddecoded[mask] * residual[mask])
    dresidual = ddecoded * cache["gains"][:, None]
    dpre_out = dresidual * (xp.float32(1.0) - residual * residual)
    grads["Wo"] = cache["h_final"].T @ dpre_out
    grads["bo"] = xp.sum(dpre_out, axis=0)
    dh = dpre_out @ params["Wo"].T
    for h_before, pre_u, pre_g, u, g, product in reversed(cache["recurrent"]):
        ddelta = dh
        grads["B"] += product.T @ ddelta
        grads["bb"] += xp.sum(ddelta, axis=0)
        dproduct = ddelta @ params["B"].T
        du = dproduct * g
        dg = dproduct * u
        dpre_u = du * silu_derivative(pre_u, xp)
        gate = sigmoid(pre_g, xp)
        dpre_g = dg * gate * (xp.float32(1.0) - gate)
        grads["A"] += h_before.T @ dpre_u
        grads["ba"] += xp.sum(dpre_u, axis=0)
        grads["C"] += h_before.T @ dpre_g
        grads["bc"] += xp.sum(dpre_g, axis=0)
        dh = dh + dpre_u @ params["A"].T + dpre_g @ params["C"].T
    dpre0 = dh * silu_derivative(cache["pre0"], xp)
    grads["Wy"] = cache["y"].T @ dpre0
    grads["Wc"] = cache["features"].T @ dpre0
    grads["b0"] = xp.sum(dpre0, axis=0)
    for name, value in grads.items():
        require_all_finite(f"gradient {name}", value, xp)
    return loss, grads


class Adam:
    def __init__(
        self,
        params: Mapping[str, Any],
        xp: Any = np,
        learning_rate: float = 5e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.xp = xp
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step = 0
        self.m = {name: xp.zeros_like(value) for name, value in params.items()}
        self.v = {name: xp.zeros_like(value) for name, value in params.items()}

    def update(self, params: Mapping[str, Any], grads: Mapping[str, Any]) -> None:
        self.step += 1
        one = self.xp.float32(1.0)
        b1 = self.xp.float32(self.beta1)
        b2 = self.xp.float32(self.beta2)
        lr = self.xp.float32(self.learning_rate)
        eps = self.xp.float32(self.epsilon)
        correction1 = one - self.xp.float32(self.beta1**self.step)
        correction2 = one - self.xp.float32(self.beta2**self.step)
        for name in PARAMETER_ORDER:
            if name not in params:
                continue
            grad = grads[name].astype(self.xp.float32, copy=False)
            self.m[name] *= b1
            self.m[name] += (one - b1) * grad
            self.v[name] *= b2
            self.v[name] += (one - b2) * grad * grad
            mhat = self.m[name] / correction1
            vhat = self.v[name] / correction2
            params[name] -= lr * mhat / (self.xp.sqrt(vhat) + eps)
            require_all_finite(f"Adam parameter {name}", params[name], self.xp)
            require_all_finite(f"Adam first moment {name}", self.m[name], self.xp)
            require_all_finite(f"Adam second moment {name}", self.v[name], self.xp)


def _parameter_payload(params: Mapping[str, Any]) -> bytes:
    chunks = []
    for name in PARAMETER_ORDER:
        array = np.asarray(params[name])
        expected = PARAMETER_SHAPES[name]
        if array.shape != expected:
            raise ValueError(f"parameter shape mismatch: {name}")
        require_all_finite(f"parameter {name}", array)
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = array.astype("<f2", copy=False)
        require_all_finite(f"FP16 parameter {name}", rounded)
        chunks.append(rounded.tobytes(order="C"))
    payload = b"".join(chunks)
    if len(payload) != MODEL_PARAMETER_BYTES:
        raise AssertionError("serialized parameter length mismatch")
    return payload


def serialize_model_bytes(
    params: Mapping[str, Any],
    training_seed: int,
    log_rms_center: float,
    log_rms_scale: float,
    protocol_hash: str | None = None,
) -> bytes:
    training_seed = int(training_seed)
    log_rms_center = require_finite_scalar("model log-RMS center", log_rms_center)
    log_rms_scale = require_finite_scalar("model log-RMS scale", log_rms_scale)
    if log_rms_scale <= 0.0:
        raise ValueError("model log-RMS scale must be positive")
    center16 = np.float16(log_rms_center)
    scale16 = np.float16(log_rms_scale)
    require_all_finite("serialized model log-RMS metadata", [center16, scale16])
    if float(scale16) <= 0.0:
        raise ValueError("serialized model log-RMS scale must be positive")
    payload = _parameter_payload(params)
    header = {
        "magic": MODEL_MAGIC,
        "schema": SCHEMA,
        "protocol_sha256": protocol_sha256() if protocol_hash is None else protocol_hash,
        "source_lock_sha256": source_lock_sha256(),
        "training_seed": training_seed,
        "dtype": "float16-le",
        "parameter_order": list(PARAMETER_ORDER),
        "parameter_shapes": {name: list(PARAMETER_SHAPES[name]) for name in PARAMETER_ORDER},
        "parameter_count": PARAMETER_COUNT,
        "parameter_payload_bytes": len(payload),
        "parameter_payload_sha256": sha256_bytes(payload),
        "log_rms_center_fp16": float(center16),
        "log_rms_scale_fp16": float(scale16),
        "identity_bypass": "three role bits in expert header",
    }
    encoded = canonical_json_bytes(header) + b"\n"
    if len(encoded) > MODEL_HEADER_BYTES:
        raise ValueError("model header exceeds frozen allocation")
    result = encoded + bytes(MODEL_HEADER_BYTES - len(encoded)) + payload
    if len(result) != MODEL_TOTAL_BYTES:
        raise AssertionError("model object length mismatch")
    return result


def deserialize_model_bytes(
    blob: bytes, expected_protocol_hash: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(blob) != MODEL_TOTAL_BYTES:
        raise ValueError("model byte length mismatch")
    header_region = blob[:MODEL_HEADER_BYTES]
    json_region, separator, padding = header_region.partition(b"\n")
    if not separator or any(padding):
        raise ValueError("noncanonical model header padding")
    header = json.loads(json_region.decode("ascii"))
    if header.get("magic") != MODEL_MAGIC or header.get("schema") != SCHEMA:
        raise ValueError("model magic/schema mismatch")
    expected_protocol_hash = (
        protocol_sha256() if expected_protocol_hash is None else expected_protocol_hash
    )
    if header.get("protocol_sha256") != expected_protocol_hash:
        raise ValueError("model protocol binding mismatch")
    if header.get("source_lock_sha256") != source_lock_sha256():
        raise ValueError("model source-lock binding mismatch")
    require_finite_scalar("decoded model log-RMS center", header.get("log_rms_center_fp16"))
    decoded_scale = require_finite_scalar(
        "decoded model log-RMS scale", header.get("log_rms_scale_fp16")
    )
    if decoded_scale <= 0.0:
        raise ValueError("decoded model log-RMS scale must be positive")
    if header.get("parameter_order") != list(PARAMETER_ORDER):
        raise ValueError("model parameter order mismatch")
    if header.get("parameter_shapes") != {
        name: list(PARAMETER_SHAPES[name]) for name in PARAMETER_ORDER
    }:
        raise ValueError("model parameter schema mismatch")
    payload = blob[MODEL_HEADER_BYTES:]
    if header.get("parameter_payload_bytes") != len(payload):
        raise ValueError("model parameter byte count mismatch")
    if header.get("parameter_payload_sha256") != sha256_bytes(payload):
        raise ValueError("model parameter hash mismatch")
    params: dict[str, np.ndarray] = {}
    offset = 0
    for name in PARAMETER_ORDER:
        count = math.prod(PARAMETER_SHAPES[name])
        nbytes = 2 * count
        array = np.frombuffer(payload[offset : offset + nbytes], dtype="<f2")
        params[name] = array.astype(np.float32).reshape(PARAMETER_SHAPES[name])
        require_all_finite(f"decoded model parameter {name}", params[name])
        offset += nbytes
    if offset != len(payload):
        raise AssertionError("model payload parse mismatch")
    return params, header


def pack_expert_header(
    layer: int,
    expert: int,
    bypass_roles: Iterable[str],
    model_sha256: str,
) -> bytes:
    flags = 0
    for role in bypass_roles:
        if role not in ROLE_INDEX:
            raise ValueError(f"unknown bypass role: {role}")
        flags |= 1 << ROLE_INDEX[role]
    if len(model_sha256) != 64:
        raise ValueError("invalid model SHA-256")
    model_hash_bytes = bytes.fromhex(model_sha256)
    blob = struct.pack(
        "<8sHHHBBIIII32s",
        EXPERT_MAGIC,
        1,
        int(layer),
        int(expert),
        flags,
        0,
        ROLE_PAYLOAD_BYTES,
        ROLE_PAYLOAD_BYTES,
        ROLE_PAYLOAD_BYTES,
        EXPERT_PAYLOAD_BYTES,
        model_hash_bytes,
    )
    if len(blob) != EXPERT_HEADER_BYTES:
        raise AssertionError("expert header length mismatch")
    return blob


def parse_expert_header(blob: bytes, expected_model_sha256: str | None = None) -> dict[str, Any]:
    if len(blob) != EXPERT_HEADER_BYTES:
        raise ValueError("expert header byte length mismatch")
    values = struct.unpack("<8sHHHBBIIII32s", blob)
    magic, version, layer, expert, flags, reserved, gate_n, up_n, down_n, total, model_hash = values
    if magic != EXPERT_MAGIC or version != 1 or reserved != 0:
        raise ValueError("expert header identity mismatch")
    if (gate_n, up_n, down_n) != (ROLE_PAYLOAD_BYTES,) * 3:
        raise ValueError("expert role length mismatch")
    if total != EXPERT_PAYLOAD_BYTES:
        raise ValueError("expert total payload mismatch")
    model_sha = model_hash.hex()
    if expected_model_sha256 is not None and model_sha != expected_model_sha256:
        raise ValueError("expert/model binding mismatch")
    return {
        "layer": layer,
        "expert": expert,
        "bypass_roles": [
            role for role in ROLE_ORDER if flags & (1 << ROLE_INDEX[role])
        ],
        "role_payload_bytes": [gate_n, up_n, down_n],
        "expert_payload_bytes": total,
        "model_sha256": model_sha,
    }


def pack_moments(means: Mapping[str, float], rms: Mapping[str, float]) -> bytes:
    values = [means[role] for role in ROLE_ORDER] + [rms[role] for role in ROLE_ORDER]
    require_all_finite("expert moment metadata", values)
    if any(float(rms[role]) <= 0.0 for role in ROLE_ORDER):
        raise ValueError("nonpositive RMS")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.asarray(values, dtype="<f2")
    require_all_finite("serialized expert moment metadata", rounded)
    if any(float(rounded[index + len(ROLE_ORDER)]) <= 0.0 for index in range(len(ROLE_ORDER))):
        raise ValueError("serialized nonpositive RMS")
    blob = rounded.tobytes()
    if len(blob) != MOMENT_BYTES:
        raise AssertionError("moment byte length mismatch")
    return blob


def unpack_moments(blob: bytes) -> dict[str, dict[str, float]]:
    if len(blob) != MOMENT_BYTES:
        raise ValueError("moment byte length mismatch")
    values = np.frombuffer(blob, dtype="<f2").astype(np.float32)
    result = {
        "mean": {role: float(values[index]) for index, role in enumerate(ROLE_ORDER)},
        "rms": {
            role: float(values[index + len(ROLE_ORDER)])
            for index, role in enumerate(ROLE_ORDER)
        },
    }
    require_all_finite("decoded expert moment metadata", values)
    if any(value <= 0.0 for value in result["rms"].values()):
        raise ValueError("decoded nonpositive RMS")
    return result


def relative_metrics(
    source_energy: float,
    identity_sse: float,
    decoded_sse: float,
    physical_rate: float,
) -> dict[str, float]:
    source_energy = require_finite_scalar("source energy", source_energy)
    identity_sse = require_finite_scalar("identity SSE", identity_sse)
    decoded_sse = require_finite_scalar("decoded SSE", decoded_sse)
    physical_rate = require_finite_scalar("physical rate", physical_rate)
    if source_energy <= 0.0 or identity_sse <= 0.0 or decoded_sse <= 0.0:
        raise ValueError("metric sufficient statistics must be positive")
    mse = decoded_sse / source_energy
    q = decoded_sse / identity_sse
    s = -0.5 * math.log2(q)
    result = {
        "source_relative_mse": mse,
        "relative_to_identity_q": q,
        "s_absolute_from_identity": s,
        "F_at_physical_rate": mse * 2.0 ** (2.0 * physical_rate),
    }
    for name, value in result.items():
        require_finite_scalar(name, value)
    return result


def delete_one_jackknife_se(values: Sequence[tuple[float, float]]) -> float:
    """SE for a pooled SSE ratio from (decoded_sse, identity_sse) clusters."""

    if len(values) < 2:
        return math.inf
    decoded_total = sum(row[0] for row in values)
    identity_total = sum(row[1] for row in values)
    estimates = []
    for decoded, identity in values:
        q = (decoded_total - decoded) / (identity_total - identity)
        estimates.append(-0.5 * math.log2(q))
    mean = sum(estimates) / len(estimates)
    variance = (len(estimates) - 1.0) / len(estimates) * sum(
        (value - mean) ** 2 for value in estimates
    )
    return require_finite_scalar("jackknife SE", math.sqrt(max(variance, 0.0)))


def hard_kill_at_512(history_by_seed: Mapping[int, Mapping[int, Mapping[str, float]]]) -> bool:
    """Exact preregistered update-512 decision; both fixed seeds must qualify."""

    if set(history_by_seed) != set(TRAINING_SEEDS):
        raise ValueError("early-stop history must contain both frozen seeds")
    for seed in TRAINING_SEEDS:
        history = history_by_seed[seed]
        if 256 not in history or 512 not in history:
            raise ValueError("early-stop history missing frozen checkpoint")
        u256 = float(history[256]["s_match_worst"]) + 2.0 * float(
            history[256]["cluster_se"]
        )
        u512 = float(history[512]["s_match_worst"]) + 2.0 * float(
            history[512]["cluster_se"]
        )
        require_finite_scalar("U256", u256)
        require_finite_scalar("U512", u512)
        if not (u512 < 0.10 and u512 - u256 < 0.012):
            return False
    return True


validate_frozen_constants()
validate_split()
validate_source_lock()
