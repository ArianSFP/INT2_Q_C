#!/usr/bin/env python3
"""Standard-library contracts and exact CPU reference for UWFA census v1.

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


DESIGN_SCHEMA = "unifilar-wfa-entropy-census-design-v1"
RESULT_SCHEMA = "unifilar-wfa-entropy-census-result-v1"
REVIEW_SCHEMA = "unifilar-wfa-entropy-census-independent-source-review-v1"
STREAM_LOCK_SCHEMA = "unifilar-wfa-canonical-stream-lock-v1"
CONTROL_LOCK_SCHEMA = "unifilar-wfa-gaussian-control-panel-lock-v1"
AUTHORIZATION = "OPEN_AUTHENTICATED_UNIFILAR_WFA_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V1"

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
SPLIT_HASH_SEED = b"UWFA-V1-NESTED-SPLIT-2026-09-01"
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
        raise ContractError(f"non-finite JSON result: {exc}") from exc


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
    magic = b"UWFAV1\x00\x00"
    header = struct.pack(
        "<8sHHHHII32s8s",
        magic,
        1,
        candidate.topology_id,
        candidate.states,
        SELECTOR_BYTES,
        candidate.reset_length,
        CONTEXTS,
        hashlib.sha256(canonical_json(candidate.as_dict())).digest(),
        b"\x00" * 8,
    )
    require(len(header) == MODEL_HEADER_BYTES, "model header geometry")
    selector = struct.pack("<H", candidate.selector_ordinal)
    tensor = struct.pack(f"<{len(frequencies)}H", *(int(value) for value in frequencies))
    packet = header + selector + tensor
    require(len(packet) == model_ledger(candidate)["physical_model_bytes"], "model serialization ledger")
    return packet


def deserialize_model(packet: bytes) -> tuple[Candidate, list[int]]:
    require(len(packet) >= MODEL_HEADER_BYTES + SELECTOR_BYTES, "short model packet")
    fields = struct.unpack("<8sHHHHII32s8s", packet[:MODEL_HEADER_BYTES])
    magic, version, topology_id, states, selector_bytes, reset_length, contexts, candidate_hash, reserved = fields
    require(magic == b"UWFAV1\x00\x00" and version == 1, "model magic/version")
    require(selector_bytes == SELECTOR_BYTES and contexts == CONTEXTS and reserved == b"\x00" * 8, "model header")
    require(0 <= topology_id < len(TOPOLOGIES), "model topology")
    candidate = Candidate(TOPOLOGIES[topology_id], states, reset_length)
    require(candidate_hash == hashlib.sha256(canonical_json(candidate.as_dict())).digest(), "candidate header hash")
    selector = struct.unpack("<H", packet[MODEL_HEADER_BYTES:MODEL_HEADER_BYTES + 2])[0]
    require(selector == candidate.selector_ordinal, "selector mismatch")
    count = model_frequency_count(candidate)
    expected = model_ledger(candidate)["physical_model_bytes"]
    require(len(packet) == expected, "model packet length")
    frequencies = list(struct.unpack(f"<{count}H", packet[MODEL_HEADER_BYTES + 2:]))
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


def packet_ledger(
    *,
    weights: int,
    current_object_bytes: int,
    immutable_global_bytes: int,
    immutable_local_bytes: Sequence[int],
    model_packet_bytes: int,
    stream_payload_bytes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Literal final whole-panel two-part packet and cold-read ledger."""
    experts = len(immutable_local_bytes)
    require(weights > 0 and experts > 0 and weights % experts == 0, "packet geometry")
    require(len(stream_payload_bytes) == experts, "payload expert geometry")
    streams = sum(len(rows) for rows in stream_payload_bytes)
    raw_global = (
        GLOBAL_HEADER_BYTES
        + int(immutable_global_bytes)
        + int(model_packet_bytes)
        + DIRECTORY_BYTES_PER_STREAM * streams
    )
    global_bytes = align_up(raw_global, PAGE_BYTES)
    local_bytes: list[int] = []
    for immutable, payloads in zip(immutable_local_bytes, stream_payload_bytes, strict=True):
        require(immutable >= 0 and all(int(value) >= 1 for value in payloads), "local packet bytes")
        local_bytes.append(EXPERT_HEADER_BYTES + int(immutable) + sum(int(value) for value in payloads))
    total = global_bytes + sum(local_bytes)
    minimum_bytes = math.ceil(weights * RATE_MIN / 8.0)
    padding = max(0, minimum_bytes - total)
    quotient, remainder = divmod(padding, experts)
    for ordinal in range(experts):
        local_bytes[ordinal] += quotient + (1 if ordinal < remainder else 0)
    total += padding
    rate = 8.0 * total / weights
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
    saving_bpw = 8.0 * (current_object_bytes - total) / weights
    f_value = CURRENT_FINITE_F * 2.0 ** (-2.0 * saving_bpw)
    rate_valid = RATE_MIN <= rate <= RATE_MAX
    cold_valid = maximum < 2.0
    return {
        "weights": weights,
        "experts": experts,
        "current_object_bytes": current_object_bytes,
        "raw_global_bytes": raw_global,
        "global_bytes_after_alignment": global_bytes,
        "model_packet_bytes": model_packet_bytes,
        "directory_bytes": DIRECTORY_BYTES_PER_STREAM * streams,
        "local_frame_bytes": local_bytes,
        "minimum_rate_padding_bytes": padding,
        "total_bytes": total,
        "physical_rate_bpw": rate,
        "passes_rate_interval": rate_valid,
        "net_physical_saving_bpw": saving_bpw,
        "required_standalone_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW,
        "F_from_unchanged_current_reconstruction": f_value,
        "passes_absolute_physical_target": f_value <= TARGET_F and rate_valid,
        "cold_rows": cold_rows,
        "maximum_cold_read_amplification": maximum,
        "passes_cold_read": cold_valid,
    }


class HeldRegularFile:
    """Reject a symlink leaf before resolution, then hold one regular descriptor."""

    def __init__(self, path: Path, expected_size: int | None = None, expected_sha256: str | None = None):
        self.path = path
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.fd: int | None = None
        self.identity: tuple[int, int, int, int] | None = None
        self.sha256: str | None = None

    def open(self) -> "HeldRegularFile":
        require(self.path.is_absolute(), f"path must be absolute: {self.path}")
        # lexists/lstat is intentionally applied to the unresolved leaf.  This
        # closes the common Path.resolve()-then-O_NOFOLLOW symlink mistake.
        require(os.path.lexists(self.path), f"held path absent: {self.path}")
        leaf = os.lstat(self.path)
        require(not stat.S_ISLNK(leaf.st_mode), f"symlink leaf forbidden: {self.path}")
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
    """Reserve an absent directory; COMPLETE.json is created exclusively last."""

    def __init__(self, path: Path):
        self.path = path
        self.completed = False

    def __enter__(self) -> "CompletionLastOutput":
        require(self.path.is_absolute(), "output path must be absolute")
        require(not os.path.lexists(self.path), "output path already exists")
        os.mkdir(self.path, 0o700)
        self.write_new("RUN_STATE.json", pretty_json({"schema": "unifilar-wfa-run-state-v1", "complete": False}))
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
                "schema": "unifilar-wfa-completion-v1",
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
        # Incomplete directories remain fail-closed and are never resumed.
        pass
