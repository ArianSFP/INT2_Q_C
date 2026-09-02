#!/usr/bin/env python3
"""Standard-library contracts and exact CPU reference for UWFA census v4.

This module deliberately imports neither NumPy nor CuPy.  The production GPU
backend is loaded only after the source manifest and an external independent
review receipt have been authenticated by ``stage0_census.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


DESIGN_SCHEMA = "unifilar-wfa-entropy-census-design-v4"
RESULT_SCHEMA = "unifilar-wfa-entropy-census-result-v4"
REVIEW_SCHEMA = "unifilar-wfa-entropy-census-independent-source-review-v4"
STREAM_LOCK_SCHEMA = "unifilar-wfa-canonical-stream-lock-v4"
CONTROL_LOCK_SCHEMA = "unifilar-wfa-gaussian-control-panel-lock-v4"
AUTHORIZATION = "OPEN_AUTHENTICATED_UNIFILAR_WFA_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V4"

TARGET_F = 0.8
RATE_MIN = 2.15
RATE_MAX = 2.5
CURRENT_FINITE_F = 0.9888693569009007
CURRENT_FINITE_S_BPW = 0.008074080480766676
TOTAL_REQUIRED_S_BPW = -0.5 * math.log2(TARGET_F)
STANDALONE_REQUIRED_SAVING_BPW = TOTAL_REQUIRED_S_BPW - CURRENT_FINITE_S_BPW

Q16_TOTAL = 65536
LEVELS = 6
PRIOR_BINS = 16
PHASES = 4
CONTEXTS = LEVELS * PRIOR_BINS * PHASES
STATE_SIZES = (2, 4, 8, 16, 32, 64)
RESET_LENGTHS = (32, 128, 512, 2048, 4096)
TOPOLOGIES = (
    "suffix",
    "xor_sketch",
    "modular_ones",
    "rolling_affine",
    "signed_saturating",
)
TOPOLOGY_IDS = {name: ordinal for ordinal, name in enumerate(TOPOLOGIES)}
FIT_SMOOTHING_HALF_COUNTS = 1
MODEL_HEADER_BYTES = 64
GLOBAL_HEADER_BYTES = 256
EXPERT_HEADER_BYTES = 512
DIRECTORY_BYTES_PER_STREAM = 64
PAGE_BYTES = 4096
SELECTOR_BYTES = 2
INNER_VALIDATION_MODULUS = 5
INNER_VALIDATION_RESIDUE = 0
SPLIT_HASH_SEED = b"UWFA-V4-NESTED-SPLIT-2026-09-02"
CONTROL_SEEDS = (10619863, 10619881, 10619909, 10619927, 10619953, 10619971, 10619999, 10620017)


class ContractError(RuntimeError):
    """A source, input, arithmetic, or lifecycle contract failed."""


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
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(data: bytes | str) -> Any:
    if not isinstance(data, (bytes, str)):
        raise ContractError("JSON input must be bytes or text")
    if len(data) > 16 * (1 << 20):
        raise ContractError("JSON input exceeds 16 MiB frozen bound")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"invalid JSON: {exc}") from exc
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 1_000_000:
            raise ContractError("JSON node count exceeds frozen bound")
        if depth > 64:
            raise ContractError("JSON nesting exceeds frozen bound")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str) and len(item) > 1 << 20:
            raise ContractError("JSON string exceeds frozen bound")
    return value


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
        raise ContractError(f"non-finite JSON result: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_decision_triplet_sha256(bits: bytes, levels: bytes, base_u16le: bytes) -> str:
    """Canonical commitment to selected bits and both regenerated SC contexts."""
    if not all(isinstance(value, bytes) for value in (bits, levels, base_u16le)):
        raise ContractError("decision triplet byte types")
    if not bits or len(levels) != len(bits) or len(base_u16le) != 2 * len(bits):
        raise ContractError("decision triplet geometry")
    digest = hashlib.sha256()
    for item in (bits, levels, base_u16le):
        digest.update(struct.pack("<Q", len(item)))
        digest.update(item)
    return digest.hexdigest()


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
    require(1 <= int(freq1) <= 65535, "base Q0.16 frequency")
    return min(PRIOR_BINS - 1, int(freq1) * PRIOR_BINS // Q16_TOTAL)


def public_context(level: int, base_freq1: int, position_in_reset: int) -> int:
    """Decoder-visible emission context. Position zero is the first symbol."""
    require(0 <= int(level) < LEVELS, "polar level")
    require(position_in_reset >= 0, "nonnegative reset position")
    return ((int(level) * PRIOR_BINS + prior_bin(base_freq1)) * PHASES) + (position_in_reset & 3)


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


@dataclass(frozen=True, order=True)
class Candidate:
    topology: str
    states: int
    reset_length: int

    def __post_init__(self) -> None:
        require(self.topology in TOPOLOGIES, "frozen topology")
        require(self.states in STATE_SIZES, "frozen state size")
        require(self.reset_length in RESET_LENGTHS, "frozen reset length")

    @property
    def topology_id(self) -> int:
        return TOPOLOGY_IDS[self.topology]

    @property
    def selector_ordinal(self) -> int:
        return (
            (self.topology_id * len(STATE_SIZES) + STATE_SIZES.index(self.states))
            * len(RESET_LENGTHS)
            + RESET_LENGTHS.index(self.reset_length)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology,
            "topology_id": self.topology_id,
            "states": self.states,
            "reset_length": self.reset_length,
            "selector_ordinal": self.selector_ordinal,
        }


def candidate_bank() -> tuple[Candidate, ...]:
    rows = tuple(
        Candidate(topology, states, reset)
        for topology in TOPOLOGIES
        for states in STATE_SIZES
        for reset in RESET_LENGTHS
    )
    require(len(rows) == 150 and len({row.selector_ordinal for row in rows}) == 150, "candidate bank closure")
    require(max(row.selector_ordinal for row in rows) < (1 << (8 * SELECTOR_BYTES)), "selector capacity")
    return rows


def transition(candidate: Candidate, state: int, bit: int, context: int, position_in_reset: int) -> int:
    """Exact O(1) unifilar update after observing the current symbol.

    At t=0 the caller first resets state to zero, obtains the emission frequency
    from (state=0, public_context(..., t=0)), codes the bit, then calls this
    function.  Reset is repeated before each t where t % reset_length == 0.
    """
    require(0 <= state < candidate.states and bit in (0, 1), "transition input")
    require(0 <= context < CONTEXTS and 0 <= position_in_reset < candidate.reset_length, "transition context")
    mask = candidate.states - 1
    topology = candidate.topology
    if topology == "suffix":
        return ((state << 1) | bit) & mask
    if topology == "xor_sketch":
        if bit == 0:
            return state
        sketch = _mix32(0xA511E9B3 ^ context ^ (position_in_reset * 0x9E3779B1)) & mask
        if sketch == 0:
            sketch = 1
        return state ^ sketch
    if topology == "modular_ones":
        weight = (_mix32(0x63D83595 ^ context ^ ((position_in_reset & 3) << 20)) & mask) | 1
        return (state + weight * bit) & mask
    if topology == "rolling_affine":
        multiplier = (5 if candidate.states >= 8 else 1) & mask
        addend = _mix32(0xB5297A4D ^ context ^ ((position_in_reset & 3) << 24)) & mask
        return (multiplier * state + addend + bit) & mask
    if topology == "signed_saturating":
        return min(mask, state + 1) if bit else max(0, state - 1)
    raise ContractError("unreachable topology")


def model_frequency_count(candidate: Candidate) -> int:
    return candidate.states * CONTEXTS


def model_ledger(candidate: Candidate) -> dict[str, int]:
    tensor_values = model_frequency_count(candidate)
    tensor_bytes = 2 * tensor_values
    physical = MODEL_HEADER_BYTES + SELECTOR_BYTES + tensor_bytes
    return {
        "states": candidate.states,
        "contexts": CONTEXTS,
        "frequency_u16_values": tensor_values,
        "tensor_bytes": tensor_bytes,
        "model_header_bytes": MODEL_HEADER_BYTES,
        "selector_bytes": SELECTOR_BYTES,
        "physical_model_bytes": physical,
        "cold_model_bytes": align_up(physical, PAGE_BYTES),
    }


def zero_counts(candidate: Candidate) -> list[int]:
    return [0] * (model_frequency_count(candidate) * 2)


def count_stream_cpu(
    bits: Sequence[int], levels: Sequence[int], base_freq1: Sequence[int], candidate: Candidate, counts: list[int] | None = None
) -> list[int]:
    require(len(bits) == len(levels) == len(base_freq1) and len(bits) > 0, "stream geometry")
    if counts is None:
        counts = zero_counts(candidate)
    require(len(counts) == model_frequency_count(candidate) * 2, "count geometry")
    state = 0
    for position, (raw_bit, raw_level, raw_frequency) in enumerate(zip(bits, levels, base_freq1, strict=True)):
        within = position % candidate.reset_length
        if within == 0:
            state = 0
        bit = int(raw_bit)
        context = public_context(int(raw_level), int(raw_frequency), within)
        index = (state * CONTEXTS + context) * 2 + bit
        counts[index] += 1
        state = transition(candidate, state, bit, context, within)
    return counts


def merge_counts(rows: Iterable[Sequence[int]]) -> list[int]:
    iterator = iter(rows)
    first = next(iterator, None)
    require(first is not None, "nonempty count collection")
    result = [int(value) for value in first]
    for row in iterator:
        require(len(row) == len(result), "count merge geometry")
        for index, value in enumerate(row):
            result[index] += int(value)
    return result


def q16_frequencies_from_counts(counts: Sequence[int]) -> list[int]:
    require(len(counts) > 0 and len(counts) % 2 == 0, "binary count geometry")
    result: list[int] = []
    for index in range(0, len(counts), 2):
        c0 = int(counts[index])
        c1 = int(counts[index + 1])
        require(c0 >= 0 and c1 >= 0, "nonnegative counts")
        # Jeffreys 1/2 smoothing evaluated and rounded with integers only:
        # p1=(2*c1+1)/(2*(c0+c1+1)).
        numerator = Q16_TOTAL * (2 * c1 + FIT_SMOOTHING_HALF_COUNTS)
        denominator = 2 * (c0 + c1 + FIT_SMOOTHING_HALF_COUNTS)
        value = (numerator + denominator // 2) // denominator
        result.append(min(65535, max(1, value)))
    return result


def stream_frequencies_cpu(
    bits: Sequence[int], levels: Sequence[int], base_freq1: Sequence[int], candidate: Candidate, frequencies: Sequence[int]
) -> list[int]:
    require(len(frequencies) == model_frequency_count(candidate), "frequency geometry")
    require(len(bits) == len(levels) == len(base_freq1) and len(bits) > 0, "stream geometry")
    output: list[int] = []
    state = 0
    for position, (raw_bit, raw_level, raw_frequency) in enumerate(zip(bits, levels, base_freq1, strict=True)):
        within = position % candidate.reset_length
        if within == 0:
            state = 0
        bit = int(raw_bit)
        context = public_context(int(raw_level), int(raw_frequency), within)
        output.append(int(frequencies[state * CONTEXTS + context]))
        state = transition(candidate, state, bit, context, within)
    return output


def serialize_model(candidate: Candidate, frequencies: Sequence[int]) -> bytes:
    require(len(frequencies) == model_frequency_count(candidate), "frequency geometry")
    require(all(1 <= int(value) <= 65535 for value in frequencies), "Q0.16 model")
    magic = b"UWFAV4\x00\x00"
    tensor = struct.pack(f"<{len(frequencies)}H", *(int(value) for value in frequencies))
    header = struct.pack(
        "<8sHHHHII32s8s",
        magic,
        3,
        candidate.topology_id,
        candidate.states,
        SELECTOR_BYTES,
        candidate.reset_length,
        CONTEXTS,
        hashlib.sha256(canonical_json(candidate.as_dict())).digest(),
        hashlib.sha256(tensor).digest()[:8],
    )
    require(len(header) == MODEL_HEADER_BYTES, "model header geometry")
    selector = struct.pack("<H", candidate.selector_ordinal)
    packet = header + selector + tensor
    require(len(packet) == model_ledger(candidate)["physical_model_bytes"], "model serialization ledger")
    return packet


def deserialize_model(packet: bytes) -> tuple[Candidate, list[int]]:
    require(len(packet) >= MODEL_HEADER_BYTES + SELECTOR_BYTES, "short model packet")
    fields = struct.unpack("<8sHHHHII32s8s", packet[:MODEL_HEADER_BYTES])
    magic, version, topology_id, states, selector_bytes, reset_length, contexts, candidate_hash, reserved = fields
    require(magic == b"UWFAV4\x00\x00" and version == 3, "model magic/version")
    require(selector_bytes == SELECTOR_BYTES and contexts == CONTEXTS, "model header")
    require(0 <= topology_id < len(TOPOLOGIES), "model topology")
    candidate = Candidate(TOPOLOGIES[topology_id], states, reset_length)
    require(candidate_hash == hashlib.sha256(canonical_json(candidate.as_dict())).digest(), "candidate header hash")
    selector = struct.unpack("<H", packet[MODEL_HEADER_BYTES:MODEL_HEADER_BYTES + 2])[0]
    require(selector == candidate.selector_ordinal, "selector mismatch")
    count = model_frequency_count(candidate)
    expected = model_ledger(candidate)["physical_model_bytes"]
    require(len(packet) == expected, "model packet length")
    tensor = packet[MODEL_HEADER_BYTES + 2:]
    require(reserved == hashlib.sha256(tensor).digest()[:8], "model tensor checksum")
    frequencies = list(struct.unpack(f"<{count}H", tensor))
    require(all(1 <= value <= 65535 for value in frequencies), "model frequencies")
    return candidate, frequencies


def _pack_output_bits(output: Sequence[int]) -> bytes:
    packed = bytearray((len(output) + 7) // 8)
    for index, bit in enumerate(output):
        packed[index >> 3] |= int(bit) << (7 - (index & 7))
    return bytes(packed)


def arithmetic_encode_binary(bits: Iterable[int], freq1: Iterable[int]) -> tuple[bytes, int]:
    """Canonical 32-bit binary arithmetic encoder, including termination."""
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
        require(low <= split < high, "arithmetic split")
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
    return _pack_output_bits(output), len(output)


def _bit_reader(payload: bytes, logical_bits: int) -> Callable[[], int]:
    require(0 < logical_bits <= len(payload) * 8, "logical payload geometry")
    # Canonical final-byte padding is zero.
    if logical_bits & 7:
        require(payload[-1] & ((1 << (8 - (logical_bits & 7))) - 1) == 0, "nonzero arithmetic padding")
    position = 0

    def read() -> int:
        nonlocal position
        if position >= logical_bits:
            return 0
        value = (payload[position >> 3] >> (7 - (position & 7))) & 1
        position += 1
        return value

    return read


def arithmetic_decode_binary(
    payload: bytes,
    logical_bits: int,
    symbol_count: int,
    frequency_before_symbol: Callable[[int], int],
    observe_symbol: Callable[[int, int], None] | None = None,
) -> list[int]:
    """Decode with causal frequencies and verify canonicality by re-encoding."""
    require(symbol_count > 0, "positive symbol count")
    read = _bit_reader(payload, logical_bits)
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    low = 0
    high = full - 1
    code = 0
    for _ in range(32):
        code = ((code << 1) & (full - 1)) | read()
    output: list[int] = []
    used_frequencies: list[int] = []
    for index in range(symbol_count):
        f1 = int(frequency_before_symbol(index))
        require(1 <= f1 <= 65535, "decoder Q0.16 frequency")
        f0 = Q16_TOTAL - f1
        width = high - low + 1
        split = low + (width * f0 // Q16_TOTAL) - 1
        require(low <= split < high, "decoder arithmetic split")
        bit = 0 if code <= split else 1
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarters:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
            code = ((code << 1) & (full - 1)) | read()
        output.append(bit)
        used_frequencies.append(f1)
        if observe_symbol is not None:
            observe_symbol(index, bit)
    replay, replay_bits = arithmetic_encode_binary(output, used_frequencies)
    require(replay_bits == logical_bits and replay == payload, "noncanonical arithmetic packet")
    return output


def encode_unifilar_stream(
    bits: Sequence[int], levels: Sequence[int], base_freq1: Sequence[int], candidate: Candidate, frequencies: Sequence[int]
) -> tuple[bytes, int]:
    model_freq = stream_frequencies_cpu(bits, levels, base_freq1, candidate, frequencies)
    return arithmetic_encode_binary(bits, model_freq)


def decode_unifilar_stream(
    payload: bytes,
    logical_bits: int,
    levels: Sequence[int],
    base_freq1: Sequence[int],
    candidate: Candidate,
    frequencies: Sequence[int],
) -> list[int]:
    require(len(levels) == len(base_freq1) and len(levels) > 0, "decode geometry")
    state_box = [0]

    def before(index: int) -> int:
        within = index % candidate.reset_length
        if within == 0:
            state_box[0] = 0
        context = public_context(int(levels[index]), int(base_freq1[index]), within)
        return int(frequencies[state_box[0] * CONTEXTS + context])

    def observe(index: int, bit: int) -> None:
        within = index % candidate.reset_length
        context = public_context(int(levels[index]), int(base_freq1[index]), within)
        state_box[0] = transition(candidate, state_box[0], bit, context, within)

    return arithmetic_decode_binary(payload, logical_bits, len(levels), before, observe)


def exact_stream_length_cpu(
    bits: Sequence[int], levels: Sequence[int], base_freq1: Sequence[int], candidate: Candidate, frequencies: Sequence[int]
) -> int:
    return encode_unifilar_stream(bits, levels, base_freq1, candidate, frequencies)[1]


def nested_split_bucket(layer_group: str, expert_group: str, stream_key: str) -> int:
    """Partition-only deterministic bucket; never a probability-model key."""
    payload = SPLIT_HASH_SEED + b"\x00" + layer_group.encode("utf-8") + b"\x00" + expert_group.encode("utf-8") + b"\x00" + stream_key.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % INNER_VALIDATION_MODULUS


def nested_split_digest(layer_group: str, expert_group: str, stream_key: str) -> bytes:
    """Full deterministic partition digest used for nonempty ranked 20% split."""
    payload = SPLIT_HASH_SEED + b"\x00" + layer_group.encode("utf-8") + b"\x00" + expert_group.encode("utf-8") + b"\x00" + stream_key.encode("utf-8")
    return hashlib.sha256(payload).digest()


def _raw_absolute_parts(path: Path) -> tuple[str, tuple[str, ...]]:
    """Return an absolute path anchor and lexical components without resolving.

    ``resolve()`` is deliberately forbidden here: resolving first would erase
    evidence that an ancestor was a symlink.  Dot and dot-dot components are
    rejected rather than normalized.
    """
    text = os.fspath(path)
    require(os.path.isabs(text), f"path must be absolute: {path}")
    drive, tail = os.path.splitdrive(text)
    separator_normalized = tail.replace("\\", "/")
    raw = tuple(part for part in separator_normalized.split("/") if part != "")
    require(all(part not in {".", ".."} for part in raw), f"noncanonical path component: {path}")
    anchor = drive + os.path.sep if drive else os.path.sep
    return anchor, raw


def _open_pinned_regular(path: Path) -> tuple[int, list[tuple[int, tuple[int, int]]]]:
    """Open one regular leaf through no-follow, retained ancestor descriptors.

    Linux/Unix uses descriptor-relative ``openat`` for every component.  The
    Windows fallback lexically lstat-checks every component before the leaf
    open; it exists only for source-free portability tests.  Numeric evidence
    records and requires the Unix descriptor-relative path.
    """
    anchor, parts = _raw_absolute_parts(path)
    require(bool(parts), f"regular leaf required: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    binary = getattr(os, "O_BINARY", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    retained: list[tuple[int, tuple[int, int]]] = []
    if os.name != "nt" and os.supports_dir_fd and os.open in os.supports_dir_fd:
        parent = os.open(anchor, os.O_RDONLY | directory | nofollow)
        info = os.fstat(parent)
        retained.append((parent, (info.st_dev, info.st_ino)))
        try:
            for component in parts[:-1]:
                child = os.open(component, os.O_RDONLY | directory | nofollow, dir_fd=parent)
                child_info = os.fstat(child)
                require(stat.S_ISDIR(child_info.st_mode), f"non-directory ancestor: {path}")
                retained.append((child, (child_info.st_dev, child_info.st_ino)))
                parent = child
            leaf_fd = os.open(parts[-1], os.O_RDONLY | binary | nofollow, dir_fd=parent)
            leaf_info = os.fstat(leaf_fd)
            require(stat.S_ISREG(leaf_info.st_mode), f"held object is not regular: {path}")
            return leaf_fd, retained
        except Exception:
            for fd, _identity in reversed(retained):
                os.close(fd)
            raise
    # Portable hostile-test fallback.  Every lexical component is checked
    # before the leaf is opened; source manifests state this is not the numeric
    # launch mechanism.
    cursor = Path(anchor)
    for component in parts:
        cursor = cursor / component
        require(os.path.lexists(cursor), f"path component absent: {cursor}")
        info = os.lstat(cursor)
        require(not stat.S_ISLNK(info.st_mode), f"symlink path component forbidden: {cursor}")
    leaf_fd = os.open(str(path), os.O_RDONLY | binary | nofollow)
    leaf_info = os.fstat(leaf_fd)
    require(stat.S_ISREG(leaf_info.st_mode), f"held object is not regular: {path}")
    return leaf_fd, retained


class HeldRegularFile:
    """Hold one regular descriptor and all Unix ancestor descriptors."""

    def __init__(self, path: Path, expected_size: int | None = None, expected_sha256: str | None = None):
        self.path = path
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.fd: int | None = None
        self.identity: tuple[int, int, int, int] | None = None
        self.sha256: str | None = None
        self.ancestor_fds: list[tuple[int, tuple[int, int]]] = []

    def open(self) -> "HeldRegularFile":
        try:
            fd, ancestors = _open_pinned_regular(self.path)
        except OSError as exc:
            raise ContractError(f"cannot open held file {self.path}: {exc}") from exc
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise ContractError(f"held object is not regular: {self.path}")
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if self.expected_size is not None and info.st_size != self.expected_size:
            os.close(fd)
            for ancestor_fd, _ancestor_identity in reversed(ancestors):
                os.close(ancestor_fd)
            raise ContractError(f"held size mismatch: {self.path}")
        self.fd = fd
        self.ancestor_fds = ancestors
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
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(self.fd, min(1 << 20, remaining))
            require(bool(chunk), f"short held read: {self.path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def verify_stable(self) -> None:
        require(self.fd is not None and self.identity is not None, "held file not open")
        info = os.fstat(self.fd)
        require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == self.identity, f"held file changed: {self.path}")
        for fd, identity in self.ancestor_fds:
            ancestor = os.fstat(fd)
            require((ancestor.st_dev, ancestor.st_ino) == identity and stat.S_ISDIR(ancestor.st_mode), f"held ancestor changed: {self.path}")

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        for fd, _identity in reversed(self.ancestor_fds):
            os.close(fd)
        self.ancestor_fds.clear()

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


class RetainedOutputParent:
    """Authenticated directory identity retained for all output operations."""

    def __init__(self, fd: int, expected_identity: tuple[int, int], authority_sha256: str):
        require(type(fd) is int and fd >= 0, "output parent descriptor")
        require(
            isinstance(expected_identity, tuple) and len(expected_identity) == 2
            and all(type(value) is int and value >= 0 for value in expected_identity),
            "output parent expected identity",
        )
        require(isinstance(authority_sha256, str) and len(authority_sha256) == 64, "output parent authority digest")
        bytes.fromhex(authority_sha256)
        self.fd = os.dup(fd)
        info = os.fstat(self.fd)
        require(stat.S_ISDIR(info.st_mode), "output parent is not a directory")
        self.identity = (int(info.st_dev), int(info.st_ino))
        require(self.identity == expected_identity, "output parent identity mismatch")
        self.authority_sha256 = authority_sha256.lower()
        self.closed = False

    @classmethod
    def open_path_source_only(cls, path: Path, authority_sha256: str) -> "RetainedOutputParent":
        """Source-only secure walk; production receives a bootstrap-held fd."""
        anchor, parts = _raw_absolute_parts(path)
        require(os.name == "posix" and os.open in os.supports_dir_fd, "descriptor-relative output requires POSIX dir_fd")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        current = os.open(anchor, os.O_RDONLY | directory | nofollow)
        try:
            for component in parts:
                child = os.open(component, os.O_RDONLY | directory | nofollow, dir_fd=current)
                require(stat.S_ISDIR(os.fstat(child).st_mode), "output ancestor is not a directory")
                os.close(current)
                current = child
            info = os.fstat(current)
            return cls(current, (int(info.st_dev), int(info.st_ino)), authority_sha256)
        finally:
            os.close(current)

    def verify_stable(self) -> None:
        require(not self.closed, "closed output parent")
        info = os.fstat(self.fd)
        require(stat.S_ISDIR(info.st_mode), "output parent ceased to be directory")
        require((int(info.st_dev), int(info.st_ino)) == self.identity, "output parent identity changed")

    def close(self) -> None:
        if not self.closed:
            self.verify_stable()
            os.close(self.fd)
            self.closed = True


def _rename_directory_noreplace(parent_fd: int, source: str, destination: str) -> None:
    """Linux renameat2(RENAME_NOREPLACE); fail closed when unavailable."""
    if os.name != "posix":
        raise ContractError("atomic no-replace publication requires POSIX")
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ContractError("renameat2 unavailable; publication refused")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


class CompletionLastOutput:
    """Durable staging transaction atomically published under a retained parent."""

    def __init__(self, parent: RetainedOutputParent, final_name: str, transaction_id: str):
        require(isinstance(parent, RetainedOutputParent), "retained output parent required")
        require(final_name == Path(final_name).name and final_name not in {"", ".", ".."} and "/" not in final_name and "\\" not in final_name, "output final name")
        require(isinstance(transaction_id, str) and len(transaction_id) == 32, "output transaction id")
        try:
            bytes.fromhex(transaction_id)
        except ValueError as exc:
            raise ContractError("output transaction id must be 128-bit hex") from exc
        self.parent = parent
        self.final_name = final_name
        self.transaction_id = transaction_id.lower()
        self.staging_name = f".uwfa-{final_name}.{self.transaction_id}.incomplete"
        self.completed = False
        self.published = False
        self.dir_fd: int | None = None
        self.members: list[dict[str, Any]] = []

    def __enter__(self) -> "CompletionLastOutput":
        self.parent.verify_stable()
        require(os.name == "posix" and os.mkdir in os.supports_dir_fd and os.open in os.supports_dir_fd, "descriptor-relative output platform")
        for name in (self.final_name, self.staging_name):
            try:
                os.stat(name, dir_fd=self.parent.fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ContractError(f"output member already exists: {name}")
        os.mkdir(self.staging_name, 0o700, dir_fd=self.parent.fd)
        os.fsync(self.parent.fd)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        self.dir_fd = os.open(self.staging_name, os.O_RDONLY | directory | nofollow, dir_fd=self.parent.fd)
        require(stat.S_ISDIR(os.fstat(self.dir_fd).st_mode), "output staging descriptor is not a directory")
        self.write_new("RUN_STATE.json", pretty_json({
            "schema": "unifilar-wfa-run-state-v4",
            "complete": False,
            "transaction_id": self.transaction_id,
            "output_parent_authority_sha256": self.parent.authority_sha256,
        }))
        return self

    def write_new(self, name: str, data: bytes) -> dict[str, Any]:
        require(not self.completed, "writes disabled after completion")
        require(self.dir_fd is not None, "output descriptor absent")
        require(name not in {"", ".", ".."} and "/" not in name and "\\" not in name, "output name")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if os.name != "nt" and os.open in os.supports_dir_fd:
            fd = os.open(name, flags, 0o600, dir_fd=self.dir_fd)
        else:
            raise ContractError("descriptor-relative output write unavailable")
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
        os.fsync(self.dir_fd)
        metadata = {"name": name, "bytes": len(data), "sha256": sha256_bytes(data)}
        self.members.append(metadata)
        return metadata

    def complete(self, members: list[dict[str, Any]], source_manifest_sha256: str) -> dict[str, Any]:
        require(not self.completed, "output already completed")
        require(members == self.members, "completion must cover every prior output member in exact order")
        record = seal_record(
            {
                "schema": "unifilar-wfa-completion-v4",
                "status": "COMPLETE_LAST",
                "source_manifest_sha256": source_manifest_sha256,
                "members": members,
            },
            "completion_sha256",
        )
        metadata = self.write_new("COMPLETE.json", pretty_json(record))
        self.completed = True
        require(self.dir_fd is not None, "output descriptor absent")
        os.fsync(self.dir_fd)
        self.parent.verify_stable()
        _rename_directory_noreplace(self.parent.fd, self.staging_name, self.final_name)
        # Publication is irrevocable before any later operation can fail.
        self.published = True
        os.fsync(self.parent.fd)
        return metadata

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        # Incomplete staging directories remain fail-closed and are never
        # resumed. Published final members are never deleted by cleanup, even
        # when a post-rename parent fsync or caller operation fails.
        if self.dir_fd is not None:
            os.close(self.dir_fd)
            self.dir_fd = None
