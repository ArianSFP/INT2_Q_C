#!/usr/bin/env python3
"""Standard-library contracts for the tied MPS entropy census v0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Sequence


DESIGN_SCHEMA = "tied-mps-entropy-census-design-v0"
RESULT_SCHEMA = "tied-mps-entropy-census-result-v0"
REVIEW_SCHEMA = "tied-mps-entropy-census-independent-source-review-v0"
STREAM_LOCK_SCHEMA = "tied-mps-canonical-stream-lock-v0"
EXTRACTION_SCHEMA = "tied-mps-independent-stream-extraction-v0"
AUTHORIZATION = "OPEN_AUTHENTICATED_TIED_MPS_ENTROPY_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V0"
TARGET_F = 0.8
RATE_MIN = 2.15
RATE_MAX = 2.5
CURRENT_FINITE_F = 0.9888693569009007
CURRENT_FINITE_S_BPW = 0.008074080480766676
TOTAL_REQUIRED_S_BPW = -0.5 * math.log2(TARGET_F)
STANDALONE_REQUIRED_SAVING_BPW = TOTAL_REQUIRED_S_BPW - CURRENT_FINITE_S_BPW
SPECULATIVE_COMPOSITE_GAP_BPW = 0.11356063457
PAGE_BYTES = 4096
EXPERT_HEADER_BYTES = 512
GLOBAL_HEADER_BYTES = 256
DIRECTORY_BYTES_PER_STREAM = 64
RESET_SYMBOLS = 4096
LEVELS = 6
PRIOR_BINS = 16
PERIODS = (1, 2, 4)
SUFFIX_DEPTHS = tuple(range(9))
HIDDEN_DIMENSIONS = (4, 8, 16, 32, 64)
FIT_SEEDS = (314159, 271828, 161803)
CONTROL_SEEDS = (10619863, 10619881, 10619909, 10619927, 10619953, 10619971, 10619999, 10620017)
EM_ITERATIONS = 12
Q16_TOTAL = 65536
ROW_TOTAL = 65535


class ContractError(RuntimeError):
    """A frozen source, input, arithmetic, or lifecycle contract failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _reject_constant(text: str) -> None:
    raise ContractError(f"non-finite JSON constant: {text}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ContractError(f"non-finite JSON number: {text}")
    return value


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(data: bytes | str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"invalid JSON: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-canonical JSON value: {exc}") from exc


def pretty_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-finite result: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def seal_record(record: dict[str, Any], field: str) -> dict[str, Any]:
    clean = dict(record)
    clean.pop(field, None)
    clean[field] = sha256_bytes(canonical_json(clean))
    return clean


def verify_internal_seal(record: dict[str, Any], field: str) -> None:
    claimed = record.get(field)
    require(isinstance(claimed, str) and len(claimed) == 64, f"missing {field}")
    clean = dict(record)
    clean.pop(field, None)
    require(sha256_bytes(canonical_json(clean)) == claimed, f"{field} mismatch")


def align_up(value: int, alignment: int) -> int:
    require(value >= 0 and alignment > 0, "alignment arguments")
    return (value + alignment - 1) // alignment * alignment


def prior_bin(freq1: int) -> int:
    require(1 <= int(freq1) <= 65535, "Q0.16 frequency")
    return min(15, int(freq1) * PRIOR_BINS // Q16_TOTAL)


def public_context(level: int, freq1: int, position: int, period: int) -> int:
    require(0 <= int(level) < LEVELS, "polar level")
    require(period in PERIODS and position >= 0, "small public phase")
    return ((int(level) * PRIOR_BINS + prior_bin(freq1)) * period) + (position % period)


def context_count(period: int) -> int:
    require(period in PERIODS, "frozen period")
    return LEVELS * PRIOR_BINS * period


def suffix_state(previous_bits: int, depth: int) -> int:
    require(depth in SUFFIX_DEPTHS, "suffix depth")
    if depth == 0:
        return 0
    return int(previous_bits) & ((1 << depth) - 1)


def suffix_model_ledger(depth: int, period: int) -> dict[str, int]:
    require(depth in SUFFIX_DEPTHS and period in PERIODS, "suffix model cell")
    states = 1 << depth
    probabilities = context_count(period) * states
    tensor_bytes = 2 * probabilities
    physical = GLOBAL_HEADER_BYTES + tensor_bytes
    return {
        "depth": depth,
        "states": states,
        "period": period,
        "contexts": context_count(period),
        "header_bytes": GLOBAL_HEADER_BYTES,
        "probability_u16_values": probabilities,
        "tensor_bytes": tensor_bytes,
        "physical_model_bytes": physical,
        "cold_model_bytes": align_up(physical, PAGE_BYTES),
    }


def hmm_model_ledger(chi: int, period: int) -> dict[str, int]:
    require(chi in HIDDEN_DIMENSIONS and period in PERIODS, "hidden HMM cell")
    contexts = context_count(period)
    initial = chi
    transitions = chi * chi
    emissions = contexts * chi
    values = initial + transitions + emissions
    tensor_bytes = 2 * values
    physical = GLOBAL_HEADER_BYTES + tensor_bytes
    return {
        "chi": chi,
        "period": period,
        "contexts": contexts,
        "header_bytes": GLOBAL_HEADER_BYTES,
        "initial_u16_values": initial,
        "transition_u16_values": transitions,
        "emission_u16_values": emissions,
        "tensor_u16_values": values,
        "tensor_bytes": tensor_bytes,
        "physical_model_bytes": physical,
        "cold_model_bytes": align_up(physical, PAGE_BYTES),
    }


def quantize_probability(value: float) -> int:
    require(math.isfinite(value), "finite probability")
    return min(65534, max(1, int(math.floor(value * Q16_TOTAL + 0.5))))


def quantize_simplex(values: Sequence[float], total: int = ROW_TOTAL) -> tuple[int, ...]:
    """Positive largest-remainder quantization with exact integer sum."""
    rows = [float(value) for value in values]
    require(rows and all(math.isfinite(value) and value >= 0.0 for value in rows), "simplex values")
    require(total >= len(rows), "simplex total")
    norm = math.fsum(rows)
    require(norm > 0.0, "positive simplex norm")
    remaining = total - len(rows)
    scaled = [value / norm * remaining for value in rows]
    floors = [int(math.floor(value)) for value in scaled]
    result = [1 + value for value in floors]
    deficit = total - sum(result)
    order = sorted(range(len(rows)), key=lambda index: (-(scaled[index] - floors[index]), index))
    for index in order[:deficit]:
        result[index] += 1
    require(sum(result) == total and all(value >= 1 for value in result), "quantized simplex closure")
    return tuple(result)


def arithmetic_encode_binary(bits: Iterable[int], freq1: Iterable[int]) -> tuple[bytes, int]:
    """Exact 32-bit binary arithmetic encoder used by the current codec."""
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    low = 0
    high = full - 1
    pending = 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        if pending:
            output.extend([1 - bit] * pending)
            pending = 0

    sentinel = object()
    bit_iterator = iter(bits)
    frequency_iterator = iter(freq1)
    count = 0
    while True:
        bit_value = next(bit_iterator, sentinel)
        frequency_value = next(frequency_iterator, sentinel)
        require((bit_value is sentinel) == (frequency_value is sentinel), "arithmetic geometry")
        if bit_value is sentinel:
            break
        bit = int(bit_value)
        f1 = int(frequency_value)
        require(bit in (0, 1) and 1 <= f1 <= 65535, "arithmetic symbol")
        f0 = Q16_TOTAL - f1
        width = high - low + 1
        split = low + (width * f0 // Q16_TOTAL) - 1
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
        count += 1
    require(count > 0, "nonempty arithmetic stream")
    pending += 1
    emit(0 if low < quarter else 1)
    logical_bits = len(output)
    packed = bytearray((logical_bits + 7) // 8)
    for index, bit in enumerate(output):
        packed[index >> 3] |= bit << (7 - (index & 7))
    return bytes(packed), logical_bits


def packet_ledger(
    *,
    weights: int,
    current_object_bytes: int,
    immutable_global_bytes: int,
    immutable_local_bytes: Sequence[int],
    model_bytes: int,
    stream_payload_bytes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Exact storage and worst-unaligned cold page ledger for one candidate."""
    experts = len(immutable_local_bytes)
    require(weights > 0 and experts > 0 and weights % experts == 0, "packet geometry")
    require(len(stream_payload_bytes) == experts, "payload expert geometry")
    streams = sum(len(rows) for rows in stream_payload_bytes)
    raw_global = (
        GLOBAL_HEADER_BYTES
        + int(immutable_global_bytes)
        + int(model_bytes)
        + DIRECTORY_BYTES_PER_STREAM * streams
    )
    global_bytes = align_up(raw_global, PAGE_BYTES)
    local_bytes = []
    for immutable, payloads in zip(immutable_local_bytes, stream_payload_bytes, strict=True):
        require(immutable >= 0 and all(int(value) >= 1 for value in payloads), "local packet bytes")
        local_bytes.append(EXPERT_HEADER_BYTES + int(immutable) + sum(int(value) for value in payloads))
    total = global_bytes + sum(local_bytes)
    minimum_bytes = math.ceil(weights * RATE_MIN / 8.0)
    padding = max(0, minimum_bytes - total)
    padding_each, padding_remainder = divmod(padding, experts)
    for index in range(experts):
        local_bytes[index] += padding_each + (1 if index < padding_remainder else 0)
    total += padding
    rate = 8.0 * total / weights
    require(total <= current_object_bytes, "candidate does not save physical bytes")
    require(RATE_MIN <= rate <= RATE_MAX, "candidate physical rate interval")
    equal_share = total / experts
    global_cold = align_up(global_bytes, PAGE_BYTES)
    cold_rows = []
    for ordinal, frame in enumerate(local_bytes):
        worst_local_pages = (frame + 2 * PAGE_BYTES - 2) // PAGE_BYTES
        cold = global_cold + worst_local_pages * PAGE_BYTES
        cold_rows.append(
            {
                "expert_ordinal": ordinal,
                "frame_bytes": frame,
                "worst_unaligned_local_pages": worst_local_pages,
                "global_cold_bytes": global_cold,
                "cold_bytes": cold,
                "cold_read_amplification": cold / equal_share,
            }
        )
    maximum = max(row["cold_read_amplification"] for row in cold_rows)
    require(maximum < 2.0, "cold read amplification")
    saving_bpw = 8.0 * (current_object_bytes - total) / weights
    f_value = CURRENT_FINITE_F * 2.0 ** (-2.0 * saving_bpw)
    return {
        "weights": weights,
        "experts": experts,
        "current_object_bytes": current_object_bytes,
        "raw_global_bytes": raw_global,
        "global_bytes_after_alignment": global_bytes,
        "model_bytes": model_bytes,
        "directory_bytes": DIRECTORY_BYTES_PER_STREAM * streams,
        "local_frame_bytes": local_bytes,
        "minimum_rate_padding_bytes": padding,
        "total_bytes": total,
        "physical_rate_bpw": rate,
        "net_physical_saving_bpw": saving_bpw,
        "required_standalone_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW,
        "F_from_unchanged_current_reconstruction": f_value,
        "passes_F_le_0p8": f_value <= TARGET_F,
        "cold_rows": cold_rows,
        "maximum_cold_read_amplification": maximum,
    }


class HeldRegularFile:
    """Open one regular file without following links and retain its descriptor."""

    def __init__(self, path: Path, expected_size: int | None = None, expected_sha256: str | None = None):
        self.path = path
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.fd: int | None = None
        self.identity: tuple[int, int, int, int] | None = None
        self.sha256: str | None = None

    def open(self) -> "HeldRegularFile":
        require(self.path.is_absolute(), f"path must be absolute: {self.path}")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(str(self.path), flags)
        except OSError as exc:
            raise ContractError(f"cannot open held file {self.path}: {exc}") from exc
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise ContractError(f"held object is not regular: {self.path}")
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if self.expected_size is not None and info.st_size != self.expected_size:
            os.close(fd)
            raise ContractError(f"held size mismatch: {self.path}")
        self.fd = fd
        self.identity = identity
        digest = hashlib.sha256()
        os.lseek(fd, 0, os.SEEK_SET)
        while chunk := os.read(fd, 1 << 20):
            digest.update(chunk)
        self.sha256 = digest.hexdigest()
        if self.expected_sha256 is not None and self.sha256 != self.expected_sha256:
            self.close()
            raise ContractError(f"held hash mismatch: {self.path}")
        os.lseek(fd, 0, os.SEEK_SET)
        return self

    @property
    def size(self) -> int:
        require(self.identity is not None, "held file not open")
        return int(self.identity[2])

    def read_all(self) -> bytes:
        require(self.fd is not None, "held file not open")
        os.lseek(self.fd, 0, os.SEEK_SET)
        remaining = self.size
        chunks = []
        while remaining:
            chunk = os.read(self.fd, min(1 << 20, remaining))
            require(bool(chunk), f"short held read: {self.path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def verify_stable(self) -> None:
        require(self.fd is not None and self.identity is not None, "held file not open")
        info = os.fstat(self.fd)
        observed = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        require(observed == self.identity, f"held file changed: {self.path}")

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "HeldRegularFile":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class HeldFileSet:
    def __init__(self) -> None:
        self.files: list[HeldRegularFile] = []

    def add(self, item: HeldRegularFile) -> HeldRegularFile:
        item.open()
        self.files.append(item)
        return item

    def verify_stable(self) -> None:
        for item in self.files:
            item.verify_stable()

    def close(self) -> None:
        for item in reversed(self.files):
            item.close()
        self.files.clear()

    def __enter__(self) -> "HeldFileSet":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class CompletionLastOutput:
    """Reserve an absent directory and publish COMPLETE.json exclusively last."""

    def __init__(self, path: Path):
        self.path = path
        self.completed = False

    def __enter__(self) -> "CompletionLastOutput":
        require(self.path.is_absolute(), "output path must be absolute")
        require(not os.path.lexists(self.path), "output path already exists")
        os.mkdir(self.path, 0o700)
        self.write_new(
            "RUN_STATE.json",
            pretty_json({"schema": "tied-mps-entropy-census-run-state-v0", "complete": False}),
        )
        return self

    def write_new(self, name: str, data: bytes) -> dict[str, Any]:
        require(name not in {"", ".", ".."} and "/" not in name and "\\" not in name, "output name")
        target = self.path / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(target), flags, 0o600)
        try:
            view = memoryview(data)
            written = 0
            while written < len(view):
                count = os.write(fd, view[written:])
                require(count > 0, f"short output write: {name}")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
        return {"name": name, "bytes": len(data), "sha256": sha256_bytes(data)}

    def complete(self, members: list[dict[str, Any]], source_manifest_sha256: str) -> dict[str, Any]:
        require(not self.completed, "output already completed")
        record = seal_record(
            {
                "schema": "tied-mps-entropy-census-completion-v0",
                "status": "COMPLETE_LAST",
                "source_manifest_sha256": source_manifest_sha256,
                "members": members,
            },
            "completion_sha256",
        )
        metadata = self.write_new("COMPLETE.json", pretty_json(record))
        self.completed = True
        return metadata

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        # Incomplete directories remain fail-closed and cannot be resumed.
        pass
