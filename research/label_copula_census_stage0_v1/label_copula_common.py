#!/usr/bin/env python3
"""Source-only reference for the universal SwiGLU label-copula census.

This module deliberately has no checkpoint, current-codec, network, NumPy,
CuPy, or CUDA input path.  It specifies the exact integer decoder, candidate
topologies, finite arithmetic coder, nested split, and byte/read ledgers used
by the future authenticated payload run.  Raw floating-point Lloyd-label
extraction is an upstream diagnostic and is intentionally not a decoder step.
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
from typing import Any, Iterable, Sequence


SCHEMA = "label-copula-census-stage0-v1"
DESIGN_SCHEMA = "label-copula-census-design-lock-v1"
RESULT_SCHEMA = "label-copula-census-result-v1"
REVIEW_SCHEMA = "label-copula-census-independent-source-review-v1"
INPUT_LOCK_SCHEMA = "label-copula-raw-swiglu-input-lock-v1"
CONTROL_PIPELINE_SCHEMA = "label-copula-matched-gaussian-control-pipeline-v1"
AUTHORIZATION = "OPEN_AUTHENTICATED_LABEL_COPULA_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V1"

TARGET_F = 0.8
CURRENT_FINITE_S_BPW = 0.008074080480766676
STANDALONE_REQUIRED_SAVING_BPW = -0.5 * math.log2(TARGET_F) - CURRENT_FINITE_S_BPW

Q_TOTAL = 1 << 16
MODEL_HEADER_BYTES = 256
CONTAINER_HEADER_BYTES = 4096
FRAME_HEADER_BYTES = 256
DIRECTORY_ENTRY_BYTES = 64
PAGE_BYTES = 4096
FRAME_ALIGNMENT = 64
ROLE_COUNT = 3
PLANE_COUNT = 2
PHASE_COUNT = 8
BOUNDARY_BUCKETS = 9
CONTEXT_COUNT = ROLE_COUNT * PLANE_COUNT * PHASE_COUNT * BOUNDARY_BUCKETS
ROLE_NAMES = ("gate", "up", "down")
TOPOLOGIES = ("suffix", "parity_sketch", "modular", "rolling", "regime")
STATE_SIZES = (2, 4, 8, 16, 32, 64)
RESET_SYMBOLS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
CONTROL_SEEDS = (10619863, 10619881, 10619909, 10619927, 10619953, 10619971, 10619999, 10620017)
BOOTSTRAP_SEED = 600613
BOOTSTRAP_REPLICATES = 4096
MIN_TOTAL_LAYERS = 10
MIN_TEST_LAYERS = 5
GAUSSIAN_LLOYD4_THRESHOLD = 0.981598821873
RAW_NORMALIZATION_WEIGHTS = 2048
MAGIC = b"LCWFA1\0\0"


class ContractError(RuntimeError):
    """A frozen source, arithmetic, split, or lifecycle invariant failed."""


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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_reparse(row: os.stat_result) -> bool:
    """Recognize POSIX symlinks and Windows reparse-point links."""
    if stat.S_ISLNK(row.st_mode):
        return True
    attributes = int(getattr(row, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_flag)


def reject_symlink_path_and_ancestors(path: Path, *, allow_missing_leaf: bool = False) -> Path:
    """Return a lexical absolute path only after checking every component.

    ``Path.resolve`` is intentionally forbidden here: resolving first erases
    the evidence needed to reject a symlinked package or a symlinked ancestor.
    The caller may open the returned path only after this function succeeds.
    """
    absolute = Path(os.path.abspath(os.fspath(path)))
    chain = tuple(reversed((absolute,) + tuple(absolute.parents)))
    for index, component in enumerate(chain):
        try:
            row = os.lstat(component)
        except FileNotFoundError:
            is_leaf = index == len(chain) - 1
            require(allow_missing_leaf and is_leaf, f"missing path component before open: {component}")
            continue
        require(not _is_link_or_reparse(row), f"symlink/reparse path component forbidden: {component}")
    return absolute


def align_up(value: int, alignment: int) -> int:
    require(value >= 0 and alignment > 0, "alignment arguments")
    return (value + alignment - 1) // alignment * alignment


def page_ceil(value: int) -> int:
    return align_up(value, PAGE_BYTES)


def _stable_order_key(namespace: str, value: str) -> tuple[str, str]:
    raw = f"{SCHEMA}|{namespace}|{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), value


def gray_bits(label: int) -> tuple[int, int]:
    """Gray map for ordered Lloyd labels: 0,1,2,3 -> 00,01,11,10."""
    require(label in (0, 1, 2, 3), "Lloyd-4 label")
    code = (0, 1, 3, 2)[label]
    return (code >> 1) & 1, code & 1


def lloyd4_label(value: float, rms: float) -> int:
    """Frozen Gaussian Lloyd-4 decision after public-block RMS normalization.

    This float operation is encoder-side diagnostic label extraction.  The
    lossless label decoder below is integer-only and never calls this function.
    """
    require(math.isfinite(value) and math.isfinite(rms) and rms >= 0.0, "finite Lloyd input")
    threshold = GAUSSIAN_LLOYD4_THRESHOLD * rms
    if value < -threshold:
        return 0
    if value < 0.0:
        return 1
    if value <= threshold:
        return 2
    return 3


def canonical_swiglu_values(
    gate: Sequence[Sequence[float]],
    up: Sequence[Sequence[float]],
    down: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Canonical micro-neuron order for arbitrary valid SwiGLU expert shapes.

    Gate and Up are semantically `[d_ff, d_model]`; Down is semantically
    `[d_model, d_ff]` and is therefore read as `Down.T`.  For each `(j,k)` the
    public order is Gate[j,k], Up[j,k], Down[k,j].  Checkpoint tensor names and
    storage orientation belong to an authenticated adapter, never to the
    probability key.
    """
    d_ff = len(gate)
    require(d_ff > 0 and len(up) == d_ff, "Gate/Up d_ff")
    d_model = len(gate[0])
    require(d_model > 0 and len(down) == d_model, "Down d_model")
    require(all(len(row) == d_model for row in gate), "Gate rectangular")
    require(all(len(row) == d_model for row in up), "Up rectangular")
    require(all(len(row) == d_ff for row in down), "Down semantic transpose")
    values: list[float] = []
    roles: list[int] = []
    for neuron in range(d_ff):
        for coordinate in range(d_model):
            triple = (
                float(gate[neuron][coordinate]),
                float(up[neuron][coordinate]),
                float(down[coordinate][neuron]),
            )
            require(all(math.isfinite(value) for value in triple), "finite raw weights")
            values.extend(triple)
            roles.extend((0, 1, 2))
    return tuple(values), tuple(roles)


def _labels_from_canonical_values(
    values: Sequence[float],
    weight_roles: Sequence[int],
    *,
    layer_group: str,
    expert_group: str,
) -> SymbolStream:
    require(len(values) > 0 and len(values) == len(weight_roles), "canonical raw geometry")
    symbols: list[int] = []
    roles: list[int] = []
    planes: list[int] = []
    for start in range(0, len(values), RAW_NORMALIZATION_WEIGHTS):
        block = tuple(float(value) for value in values[start : start + RAW_NORMALIZATION_WEIGHTS])
        require(all(math.isfinite(value) for value in block), "finite normalization block")
        rms = math.sqrt(math.fsum(value * value for value in block) / len(block))
        for offset, value in enumerate(block):
            role = int(weight_roles[start + offset])
            require(0 <= role < ROLE_COUNT, "canonical role")
            first, second = gray_bits(lloyd4_label(value, rms))
            symbols.extend((first, second))
            roles.extend((role, role))
            planes.extend((0, 1))
    stream = SymbolStream(
        layer_group=str(layer_group),
        expert_group=str(expert_group),
        symbols=tuple(symbols),
        roles=tuple(roles),
        planes=tuple(planes),
        source_weights=len(values),
    )
    stream.validate()
    return stream


def canonical_raw_lloyd4_stream(
    gate: Sequence[Sequence[float]],
    up: Sequence[Sequence[float]],
    down: Sequence[Sequence[float]],
    *,
    layer_group: str,
    expert_group: str,
) -> SymbolStream:
    values, roles = canonical_swiglu_values(gate, up, down)
    return _labels_from_canonical_values(
        values,
        roles,
        layer_group=layer_group,
        expert_group=expert_group,
    )


def matched_gaussian_raw_control(
    gate: Sequence[Sequence[float]],
    up: Sequence[Sequence[float]],
    down: Sequence[Sequence[float]],
    *,
    layer_group: str,
    expert_group: str,
    seed: int,
) -> SymbolStream:
    """Independently sample and relabel a block-moment-matched Gaussian control."""
    import random

    require(seed in CONTROL_SEEDS, "frozen Gaussian control seed")
    values, roles = canonical_swiglu_values(gate, up, down)
    rng = random.Random(seed ^ int(hashlib.sha256(
        f"{layer_group}|{expert_group}".encode("utf-8")
    ).hexdigest()[:16], 16))
    control: list[float] = []
    for start in range(0, len(values), RAW_NORMALIZATION_WEIGHTS):
        block = tuple(values[start : start + RAW_NORMALIZATION_WEIGHTS])
        mean = math.fsum(block) / len(block)
        variance = math.fsum((value - mean) ** 2 for value in block) / len(block)
        deviation = math.sqrt(max(0.0, variance))
        if deviation == 0.0:
            control.extend([mean] * len(block))
        else:
            control.extend(rng.gauss(mean, deviation) for _ in block)
    return _labels_from_canonical_values(
        control,
        roles,
        layer_group=layer_group,
        expert_group=expert_group,
    )


def public_context(role: int, plane: int, position: int, reset: int) -> int:
    """Decoder-visible context; it contains no layer/expert/model identity."""
    require(0 <= role < ROLE_COUNT and 0 <= plane < PLANE_COUNT, "role/plane")
    require(reset in RESET_SYMBOLS and position >= 0, "reset/position")
    within = position % reset
    remaining = reset - 1 - within
    boundary = remaining if remaining < BOUNDARY_BUCKETS - 1 else BOUNDARY_BUCKETS - 1
    phase = within & (PHASE_COUNT - 1)
    context = (((role * PLANE_COUNT + plane) * PHASE_COUNT + phase) * BOUNDARY_BUCKETS) + boundary
    require(0 <= context < CONTEXT_COUNT, "public context range")
    return context


@dataclass(frozen=True, order=True)
class Candidate:
    topology: str
    chi: int
    reset: int

    def validate(self, allow_factorized: bool = False) -> None:
        allowed = TOPOLOGIES + (("factorized",) if allow_factorized else ())
        require(self.topology in allowed, "candidate topology")
        if self.topology == "factorized":
            require(self.chi == 1, "factorized chi")
        else:
            require(self.chi in STATE_SIZES, "candidate chi")
        require(self.reset in RESET_SYMBOLS, "candidate reset")

    @property
    def code(self) -> int:
        if self.topology == "factorized":
            return 0
        return TOPOLOGIES.index(self.topology) + 1


def candidate_bank() -> tuple[Candidate, ...]:
    rows = tuple(Candidate(name, chi, reset) for name in TOPOLOGIES for chi in STATE_SIZES for reset in RESET_SYMBOLS)
    require(len(rows) == 240 and len(set(rows)) == 240, "candidate-bank closure")
    return rows


def factorized_bank() -> tuple[Candidate, ...]:
    return tuple(Candidate("factorized", 1, reset) for reset in RESET_SYMBOLS)


def candidate_bank_sha256(bank: Sequence[Candidate]) -> str:
    rows = []
    for candidate in bank:
        candidate.validate(allow_factorized=False)
        rows.append({
            "topology": candidate.topology,
            "chi": candidate.chi,
            "reset_symbols": candidate.reset,
        })
    require(bool(rows) and len(rows) == len(set(bank)), "candidate-bank binding")
    return sha256_bytes(canonical_json(rows))


def _parity_bits(chi: int) -> int:
    require(chi in STATE_SIZES, "parity chi")
    return chi.bit_length() - 1


def next_state(candidate: Candidate, state: int, symbol: int, role: int, plane: int, position: int) -> int:
    """O(1), exact-integer, unifilar state recurrence."""
    candidate.validate(allow_factorized=True)
    require(0 <= state < candidate.chi and symbol in (0, 1), "state/symbol")
    if candidate.topology == "factorized":
        return 0
    chi = candidate.chi
    mask = chi - 1
    within = position % candidate.reset
    if candidate.topology == "suffix":
        return ((state << 1) | symbol) & mask
    if candidate.topology == "parity_sketch":
        bits = _parity_bits(chi)
        # Reserve the final k sites as public check positions.  Earlier one-bits
        # toggle one of k deterministic sketches; check symbols do not erase it.
        if symbol and within < candidate.reset - bits:
            sketch = (within + 3 * role + 5 * plane) % bits
            return state ^ (1 << sketch)
        return state
    if candidate.topology == "modular":
        half = max(1, chi // 2)
        step = (2 * ((within + 3 * role + 5 * plane) % half) + 1) & mask
        return (state + symbol * step) & mask
    if candidate.topology == "rolling":
        multiplier = 5 if chi >= 8 else 3
        return (state * multiplier + symbol) & mask
    if candidate.topology == "regime":
        half = chi // 2
        last = state // half
        age = state % half
        if symbol == last:
            age = min(half - 1, age + 1)
        else:
            last = symbol
            age = 0
        return last * half + age
    raise AssertionError("unreachable topology")


@dataclass(frozen=True)
class SymbolStream:
    layer_group: str
    expert_group: str
    symbols: tuple[int, ...]
    roles: tuple[int, ...]
    planes: tuple[int, ...]
    source_weights: int

    def validate(self) -> None:
        size = len(self.symbols)
        require(size > 0 and len(self.roles) == size and len(self.planes) == size, "stream geometry")
        require(self.source_weights > 0, "source weights")
        require(size == 2 * self.source_weights, "two Gray decisions per raw weight")
        require(all(value in (0, 1) for value in self.symbols), "binary symbols")
        require(all(0 <= value < ROLE_COUNT for value in self.roles), "role range")
        require(all(0 <= value < PLANE_COUNT for value in self.planes), "plane range")
        require(bool(self.layer_group) and bool(self.expert_group), "split metadata")


@dataclass(frozen=True)
class RawSwiGLUExpert:
    """Immutable source-side reference record for one canonical expert.

    This pure-Python record is only the source-only semantic reference.  A
    future payload adapter must implement the same hashes and label pipeline in
    CuPy and receive its own independent review before opening a checkpoint.
    """

    layer_group: str
    expert_group: str
    gate: tuple[tuple[float, ...], ...]
    up: tuple[tuple[float, ...], ...]
    down: tuple[tuple[float, ...], ...]

    def validate(self) -> None:
        require(bool(self.layer_group) and bool(self.expert_group), "raw split metadata")
        require(
            isinstance(self.gate, tuple)
            and isinstance(self.up, tuple)
            and isinstance(self.down, tuple)
            and all(isinstance(row, tuple) for row in self.gate + self.up + self.down),
            "raw panel must use immutable tuples",
        )
        canonical_swiglu_values(self.gate, self.up, self.down)


def _ordered_streams_and_slot_universe(
    streams: Sequence[SymbolStream],
) -> tuple[tuple[SymbolStream, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate one reusable expert-slot universe and canonicalize row order."""
    require(bool(streams), "partition streams")
    rows: dict[tuple[str, str], SymbolStream] = {}
    for stream in streams:
        stream.validate()
        key = (stream.layer_group, stream.expert_group)
        require(key not in rows, "exactly one stream per layer/expert slot")
        rows[key] = stream
    layers = tuple(sorted({key[0] for key in rows}, key=lambda value: _stable_order_key("layer", value)))
    slots = tuple(sorted({key[1] for key in rows}, key=lambda value: _stable_order_key("expert-slot", value)))
    require(len(slots) >= 3, "nested split needs at least three reusable expert slots")
    expected = set(slots)
    for layer in layers:
        observed = {slot for candidate_layer, slot in rows if candidate_layer == layer}
        require(observed == expected, "every layer must instantiate the identical reusable expert-slot universe")
    require(len(rows) == len(layers) * len(slots), "rectangular layer/expert-slot panel")
    ordered = tuple(rows[(layer, slot)] for layer in layers for slot in slots)
    return ordered, layers, slots


def _ordered_raw_panel(
    raw_panel: Sequence[RawSwiGLUExpert],
) -> tuple[RawSwiGLUExpert, ...]:
    require(bool(raw_panel), "raw panel")
    for expert in raw_panel:
        expert.validate()
    placeholder = tuple(
        SymbolStream(
            expert.layer_group,
            expert.expert_group,
            (0, 0),
            (0, 0),
            (0, 1),
            1,
        )
        for expert in raw_panel
    )
    ordered_placeholder, _, _ = _ordered_streams_and_slot_universe(placeholder)
    by_key = {(row.layer_group, row.expert_group): row for row in raw_panel}
    return tuple(by_key[(row.layer_group, row.expert_group)] for row in ordered_placeholder)


def _hash_text(digest: Any, value: str) -> None:
    raw = value.encode("utf-8")
    digest.update(struct.pack("<I", len(raw)))
    digest.update(raw)


def _hash_float64(digest: Any, value: float) -> None:
    require(math.isfinite(value), "finite provenance float")
    digest.update(struct.pack("<d", float(value)))


def _symbol_panel_sha256(streams: Sequence[SymbolStream]) -> str:
    ordered, _, _ = _ordered_streams_and_slot_universe(streams)
    digest = hashlib.sha256(b"label-copula-symbol-panel-v1\0")
    for stream in ordered:
        _hash_text(digest, stream.layer_group)
        _hash_text(digest, stream.expert_group)
        digest.update(struct.pack("<QQ", stream.source_weights, len(stream.symbols)))
        digest.update(bytes(stream.symbols))
        digest.update(bytes(stream.roles))
        digest.update(bytes(stream.planes))
    return digest.hexdigest()


def _raw_panel_records(
    raw_panel: Sequence[RawSwiGLUExpert],
) -> tuple[tuple[RawSwiGLUExpert, tuple[float, ...], tuple[int, ...]], ...]:
    return tuple((expert, *canonical_swiglu_values(expert.gate, expert.up, expert.down)) for expert in _ordered_raw_panel(raw_panel))


def raw_panel_provenance(raw_panel: Sequence[RawSwiGLUExpert]) -> dict[str, Any]:
    """Bind source tensor bits, geometry, and every matched block moment."""
    records = _raw_panel_records(raw_panel)
    source_digest = hashlib.sha256(b"label-copula-raw-source-panel-v1\0")
    geometry_digest = hashlib.sha256(b"label-copula-raw-source-geometry-v1\0")
    moment_digest = hashlib.sha256(b"label-copula-raw-source-block-moments-v1\0")
    blocks = 0
    weights = 0
    for expert, values, roles in records:
        d_ff = len(expert.gate)
        d_model = len(expert.gate[0])
        for digest in (source_digest, geometry_digest, moment_digest):
            _hash_text(digest, expert.layer_group)
            _hash_text(digest, expert.expert_group)
        geometry_digest.update(struct.pack("<QQQ", d_ff, d_model, len(values)))
        source_digest.update(struct.pack("<QQQ", d_ff, d_model, len(values)))
        source_digest.update(bytes(roles))
        for value in values:
            _hash_float64(source_digest, value)
        for start in range(0, len(values), RAW_NORMALIZATION_WEIGHTS):
            block = values[start : start + RAW_NORMALIZATION_WEIGHTS]
            mean = math.fsum(block) / len(block)
            variance = math.fsum((value - mean) ** 2 for value in block) / len(block)
            second_moment = math.fsum(value * value for value in block) / len(block)
            moment_digest.update(struct.pack("<QQ", start, len(block)))
            for value in (mean, variance, second_moment):
                _hash_float64(moment_digest, value)
            blocks += 1
        weights += len(values)
    layers = {row[0].layer_group for row in records}
    slots = {row[0].expert_group for row in records}
    return {
        "schema": "label-copula-raw-source-provenance-v1",
        "canonical_orientation": "Gate[j,k],Up[j,k],Down[k,j]",
        "normalization_block_source_weights": RAW_NORMALIZATION_WEIGHTS,
        "experts": len(records),
        "layers": len(layers),
        "reusable_expert_slots": len(slots),
        "source_weights": weights,
        "source_blocks": blocks,
        "source_tensor_sha256": source_digest.hexdigest(),
        "source_geometry_sha256": geometry_digest.hexdigest(),
        "source_block_moments_sha256": moment_digest.hexdigest(),
    }


def build_matched_gaussian_control_panel(
    raw_panel: Sequence[RawSwiGLUExpert], seed: int
) -> tuple[tuple[SymbolStream, ...], dict[str, Any]]:
    """Generate one frozen-seed control and bind every pre-label float.

    This is the only accepted control-panel constructor.  The scientific
    evaluator below accepts raw experts rather than caller-supplied symbol
    streams, so an arbitrary prebuilt panel cannot be relabeled as a frozen
    Gaussian control.
    """
    import random

    require(seed in CONTROL_SEEDS, "frozen Gaussian control seed")
    records = _raw_panel_records(raw_panel)
    source = raw_panel_provenance(raw_panel)
    generated_digest = hashlib.sha256(b"label-copula-generated-gaussian-tensor-v1\0")
    streams: list[SymbolStream] = []
    generated_weights = 0
    for expert, values, roles in records:
        derived_seed = seed ^ int(
            hashlib.sha256(
                f"{CONTROL_PIPELINE_SCHEMA}|{expert.layer_group}|{expert.expert_group}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        rng = random.Random(derived_seed)
        control: list[float] = []
        for start in range(0, len(values), RAW_NORMALIZATION_WEIGHTS):
            block = values[start : start + RAW_NORMALIZATION_WEIGHTS]
            mean = math.fsum(block) / len(block)
            variance = math.fsum((value - mean) ** 2 for value in block) / len(block)
            deviation = math.sqrt(max(0.0, variance))
            if deviation == 0.0:
                generated = [mean] * len(block)
            else:
                generated = [rng.gauss(mean, deviation) for _ in block]
            require(all(math.isfinite(value) for value in generated), "finite generated Gaussian control")
            control.extend(generated)
        _hash_text(generated_digest, expert.layer_group)
        _hash_text(generated_digest, expert.expert_group)
        generated_digest.update(struct.pack("<QQQ", len(expert.gate), len(expert.gate[0]), len(control)))
        for value in control:
            _hash_float64(generated_digest, value)
        streams.append(
            _labels_from_canonical_values(
                control,
                roles,
                layer_group=expert.layer_group,
                expert_group=expert.expert_group,
            )
        )
        generated_weights += len(control)
    ordered_streams, layers, slots = _ordered_streams_and_slot_universe(streams)
    evidence = {
        "schema": "label-copula-generated-control-provenance-v1",
        "control_pipeline_schema": CONTROL_PIPELINE_SCHEMA,
        "seed": seed,
        "seed_index": CONTROL_SEEDS.index(seed),
        "source_tensor_sha256": source["source_tensor_sha256"],
        "source_geometry_sha256": source["source_geometry_sha256"],
        "source_block_moments_sha256": source["source_block_moments_sha256"],
        "generated_tensor_sha256": generated_digest.hexdigest(),
        "generated_label_panel_sha256": _symbol_panel_sha256(ordered_streams),
        "generated_source_weights": generated_weights,
        "layers": len(layers),
        "reusable_expert_slots": len(slots),
        "pipeline_prefix": [
            "frozen seed and layer/slot derivation",
            "source 2048-weight mean/variance matching",
            "independent Gaussian generation",
            "canonical Gate/Up/Down geometry",
            "RMS normalization",
            "fixed Gaussian Lloyd-4 labeling",
            "ordered Gray serialization",
        ],
    }
    evidence["generation_binding_sha256"] = sha256_bytes(canonical_json(evidence))
    return ordered_streams, evidence


@dataclass(frozen=True)
class QuantizedModel:
    candidate: Candidate
    freq1: tuple[int, ...]

    def validate(self) -> None:
        self.candidate.validate(allow_factorized=True)
        expected = CONTEXT_COUNT * self.candidate.chi
        require(len(self.freq1) == expected, "frequency-table geometry")
        require(all(1 <= int(value) < Q_TOTAL for value in self.freq1), "positive q16 frequencies")

    def probability(self, context: int, state: int) -> int:
        require(0 <= context < CONTEXT_COUNT and 0 <= state < self.candidate.chi, "probability key")
        return int(self.freq1[context * self.candidate.chi + state])

    def serialize(self) -> bytes:
        self.validate()
        header = struct.pack(
            "<8sHHHHIIII",
            MAGIC,
            0,
            self.candidate.code,
            self.candidate.chi,
            0,
            self.candidate.reset,
            CONTEXT_COUNT,
            Q_TOTAL,
            len(self.freq1),
        )
        require(len(header) <= MODEL_HEADER_BYTES, "model header fit")
        table = struct.pack(f"<{len(self.freq1)}H", *self.freq1)
        return header + bytes(MODEL_HEADER_BYTES - len(header)) + table

    @classmethod
    def deserialize(cls, packet: bytes) -> "QuantizedModel":
        require(len(packet) >= MODEL_HEADER_BYTES, "model packet header")
        magic, version, code, chi, reserved, reset, contexts, total, values = struct.unpack(
            "<8sHHHHIIII", packet[:32]
        )
        require(magic == MAGIC and version == 0 and reserved == 0, "model magic/version")
        require(contexts == CONTEXT_COUNT and total == Q_TOTAL, "model constants")
        topology = "factorized" if code == 0 else TOPOLOGIES[code - 1] if 1 <= code <= len(TOPOLOGIES) else ""
        candidate = Candidate(topology, int(chi), int(reset))
        candidate.validate(allow_factorized=True)
        require(values == CONTEXT_COUNT * chi, "model value count")
        require(len(packet) == MODEL_HEADER_BYTES + 2 * values, "model packet length")
        freq1 = struct.unpack(f"<{values}H", packet[MODEL_HEADER_BYTES:])
        result = cls(candidate, tuple(int(value) for value in freq1))
        result.validate()
        return result


def model_ledger(candidate: Candidate) -> dict[str, int | str]:
    candidate.validate(allow_factorized=True)
    values = CONTEXT_COUNT * candidate.chi
    packet = MODEL_HEADER_BYTES + 2 * values
    return {
        "topology": candidate.topology,
        "chi": candidate.chi,
        "reset_symbols": candidate.reset,
        "contexts": CONTEXT_COUNT,
        "frequency_u16_values": values,
        "header_bytes": MODEL_HEADER_BYTES,
        "physical_model_bytes": packet,
        "cold_model_page_bytes": page_ceil(packet),
    }


def _quantized_jeffreys(count0: int, count1: int) -> int:
    require(count0 >= 0 and count1 >= 0, "counts")
    # Round Q*(n1+1/2)/(n0+n1+1) using integers only.
    numerator = Q_TOTAL * (2 * count1 + 1)
    denominator = 2 * (count0 + count1 + 1)
    value = (numerator + denominator // 2) // denominator
    return min(Q_TOTAL - 1, max(1, int(value)))


def fit_model(streams: Sequence[SymbolStream], candidate: Candidate) -> QuantizedModel:
    candidate.validate(allow_factorized=True)
    require(bool(streams), "nonempty training streams")
    counts0 = [0] * (CONTEXT_COUNT * candidate.chi)
    counts1 = [0] * (CONTEXT_COUNT * candidate.chi)
    for stream in streams:
        stream.validate()
        state = 0
        for position, (symbol, role, plane) in enumerate(zip(stream.symbols, stream.roles, stream.planes, strict=True)):
            if position % candidate.reset == 0:
                state = 0
            context = public_context(role, plane, position, candidate.reset)
            key = context * candidate.chi + state
            (counts1 if symbol else counts0)[key] += 1
            state = next_state(candidate, state, symbol, role, plane, position)
    freq1 = tuple(_quantized_jeffreys(n0, n1) for n0, n1 in zip(counts0, counts1, strict=True))
    model = QuantizedModel(candidate, freq1)
    model.validate()
    return model


class ArithmeticEncoder:
    """Finite 32-bit binary arithmetic encoder with literal termination bits."""

    FULL = 1 << 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE_QUARTERS = 3 << 30

    def __init__(self) -> None:
        self.low = 0
        self.high = self.FULL - 1
        self.pending = 0
        self.output: list[int] = []

    def _emit(self, bit: int) -> None:
        self.output.append(bit)
        if self.pending:
            self.output.extend([1 - bit] * self.pending)
            self.pending = 0

    def write(self, symbol: int, freq1: int) -> None:
        require(symbol in (0, 1) and 1 <= freq1 < Q_TOTAL, "arithmetic event")
        f0 = Q_TOTAL - freq1
        width = self.high - self.low + 1
        split = self.low + (width * f0 // Q_TOTAL) - 1
        require(self.low <= split < self.high, "noncollapsed arithmetic split")
        if symbol == 0:
            self.high = split
        else:
            self.low = split + 1
        while True:
            if self.high < self.HALF:
                self._emit(0)
            elif self.low >= self.HALF:
                self._emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTERS:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & (self.FULL - 1)
            self.high = ((self.high << 1) & (self.FULL - 1)) | 1

    def finish(self) -> tuple[bytes, int]:
        self.pending += 1
        self._emit(0 if self.low < self.QUARTER else 1)
        meaningful = len(self.output)
        packed = bytearray((meaningful + 7) // 8)
        for index, bit in enumerate(self.output):
            packed[index >> 3] |= bit << (7 - (index & 7))
        return bytes(packed), meaningful


class _BitReader:
    def __init__(self, payload: bytes, meaningful_bits: int) -> None:
        require(0 < meaningful_bits <= 8 * len(payload), "meaningful arithmetic bits")
        self.payload = payload
        self.meaningful_bits = meaningful_bits
        self.position = 0

    def read(self) -> int:
        if self.position >= self.meaningful_bits:
            self.position += 1
            return 0
        index = self.position
        self.position += 1
        return (self.payload[index >> 3] >> (7 - (index & 7))) & 1


class ArithmeticDecoder:
    FULL = 1 << 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE_QUARTERS = 3 << 30

    def __init__(self, payload: bytes, meaningful_bits: int) -> None:
        self.reader = _BitReader(payload, meaningful_bits)
        self.low = 0
        self.high = self.FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) | self.reader.read()) & (self.FULL - 1)

    def read(self, freq1: int) -> int:
        require(1 <= freq1 < Q_TOTAL, "decoder frequency")
        f0 = Q_TOTAL - freq1
        width = self.high - self.low + 1
        split = self.low + (width * f0 // Q_TOTAL) - 1
        require(self.low <= self.code <= self.high and self.low <= split < self.high, "decoder interval")
        if self.code <= split:
            symbol = 0
            self.high = split
        else:
            symbol = 1
            self.low = split + 1
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.code -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTERS:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.code -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & (self.FULL - 1)
            self.high = ((self.high << 1) & (self.FULL - 1)) | 1
            self.code = ((self.code << 1) & (self.FULL - 1)) | self.reader.read()
        return symbol


def encode_stream(model: QuantizedModel, stream: SymbolStream) -> tuple[bytes, int]:
    model.validate()
    stream.validate()
    encoder = ArithmeticEncoder()
    state = 0
    candidate = model.candidate
    for position, (symbol, role, plane) in enumerate(zip(stream.symbols, stream.roles, stream.planes, strict=True)):
        if position % candidate.reset == 0:
            state = 0
        context = public_context(role, plane, position, candidate.reset)
        encoder.write(symbol, model.probability(context, state))
        state = next_state(candidate, state, symbol, role, plane, position)
    return encoder.finish()


def decode_stream(
    model: QuantizedModel,
    roles: Sequence[int],
    planes: Sequence[int],
    payload: bytes,
    meaningful_bits: int,
) -> tuple[int, ...]:
    model.validate()
    require(len(roles) > 0 and len(roles) == len(planes), "decode schedule")
    decoder = ArithmeticDecoder(payload, meaningful_bits)
    state = 0
    output: list[int] = []
    candidate = model.candidate
    for position, (role, plane) in enumerate(zip(roles, planes, strict=True)):
        if position % candidate.reset == 0:
            state = 0
        context = public_context(role, plane, position, candidate.reset)
        symbol = decoder.read(model.probability(context, state))
        output.append(symbol)
        state = next_state(candidate, state, symbol, role, plane, position)
    return tuple(output)


def nested_partition(streams: Sequence[SymbolStream]) -> dict[str, tuple[SymbolStream, ...]]:
    """Outer whole-layer test and inner reusable whole-slot validation split."""
    ordered, layers, experts = _ordered_streams_and_slot_universe(streams)
    require(
        len(layers) >= MIN_TOTAL_LAYERS,
        f"confidence gate needs at least {MIN_TOTAL_LAYERS} whole layers",
    )
    test_count = max(MIN_TEST_LAYERS, math.ceil(len(layers) / 5))
    require(
        len(layers) - test_count >= 2,
        "at least two non-test layers are required after the whole-layer holdout",
    )
    test_layers = set(layers[:test_count])
    validation_count = max(1, math.ceil(len(experts) / 5))
    validation_experts = set(experts[:validation_count])
    train = tuple(row for row in ordered if row.layer_group not in test_layers and row.expert_group not in validation_experts)
    validation = tuple(row for row in ordered if row.layer_group not in test_layers and row.expert_group in validation_experts)
    test = tuple(row for row in ordered if row.layer_group in test_layers)
    require(bool(train) and bool(validation) and bool(test), "nonempty nested folds")
    require(len({row.layer_group for row in test}) >= MIN_TEST_LAYERS, "nondegenerate whole-layer confidence clusters")
    require(not ({row.layer_group for row in train} & {row.layer_group for row in test}), "whole-layer test isolation")
    require(not ({row.expert_group for row in train} & {row.expert_group for row in validation}), "whole-expert validation isolation")
    return {"train": train, "validation": validation, "test": test}


def container_ledger(model_bytes: int, encoded: Sequence[tuple[SymbolStream, bytes, int]]) -> dict[str, Any]:
    """Literal model/directory/frame/alignment storage and routed page reads."""
    require(model_bytes > 0 and bool(encoded), "container geometry")
    model_stored = page_ceil(model_bytes)
    directory_stored = page_ceil(DIRECTORY_ENTRY_BYTES * len(encoded))
    shared_stored = CONTAINER_HEADER_BYTES + model_stored + directory_stored
    offset = shared_stored
    rows: list[dict[str, Any]] = []
    total_weights = 0
    for ordinal, (stream, payload, meaningful) in enumerate(encoded):
        stream.validate()
        require(len(payload) == (meaningful + 7) // 8, "payload byte padding")
        frame_bytes = FRAME_HEADER_BYTES + len(payload)
        stored = align_up(frame_bytes, FRAME_ALIGNMENT)
        start = offset
        end = start + frame_bytes
        first_page = start // PAGE_BYTES
        final_page = (end - 1) // PAGE_BYTES
        local_read = (final_page - first_page + 1) * PAGE_BYTES
        rows.append({
            "ordinal": ordinal,
            "layer_group": stream.layer_group,
            "expert_group": stream.expert_group,
            "source_weights": stream.source_weights,
            "symbols": len(stream.symbols),
            "meaningful_arithmetic_bits": meaningful,
            "arithmetic_payload_bytes": len(payload),
            "frame_header_bytes": FRAME_HEADER_BYTES,
            "frame_bytes": frame_bytes,
            "stored_frame_bytes": stored,
            "frame_offset": start,
            "local_page_read_bytes": local_read,
        })
        offset += stored
        total_weights += stream.source_weights
    total_bytes = page_ceil(offset)
    final_padding = total_bytes - offset
    shared_allocation = shared_stored + final_padding
    header_pages = set(range(0, CONTAINER_HEADER_BYTES // PAGE_BYTES))
    model_start = CONTAINER_HEADER_BYTES
    model_pages = set(range(model_start // PAGE_BYTES, (model_start + model_stored) // PAGE_BYTES))
    directory_start = CONTAINER_HEADER_BYTES + model_stored
    for row in rows:
        directory_entry_start = directory_start + int(row["ordinal"]) * DIRECTORY_ENTRY_BYTES
        directory_pages = {directory_entry_start // PAGE_BYTES}
        frame_start = int(row["frame_offset"])
        frame_end = frame_start + int(row["frame_bytes"])
        frame_pages = set(range(frame_start // PAGE_BYTES, (frame_end - 1) // PAGE_BYTES + 1))
        require(len(frame_pages) * PAGE_BYTES == int(row["local_page_read_bytes"]), "literal frame-page closure")
        shared_pages = header_pages | model_pages | directory_pages
        cold_pages = shared_pages | frame_pages
        denominator = shared_allocation / len(encoded) + int(row["stored_frame_bytes"])
        row["header_page_indices"] = sorted(header_pages)
        row["model_page_indices"] = sorted(model_pages)
        row["addressed_directory_page_indices"] = sorted(directory_pages)
        row["frame_page_indices"] = sorted(frame_pages)
        row["cold_page_indices"] = sorted(cold_pages)
        row["shared_cold_bytes"] = len(shared_pages) * PAGE_BYTES
        row["cold_read_bytes"] = len(cold_pages) * PAGE_BYTES
        row["shared_denominator_bytes_per_expert"] = shared_allocation / len(encoded)
        row["private_denominator_bytes"] = int(row["stored_frame_bytes"])
        row["physical_denominator_bytes"] = denominator
        row["cold_read_amplification"] = row["cold_read_bytes"] / denominator
    denominator_total = math.fsum(float(row["physical_denominator_bytes"]) for row in rows)
    require(math.isclose(denominator_total, float(total_bytes), rel_tol=0.0, abs_tol=1e-9), "per-expert denominator closure")
    return {
        "source_weights": total_weights,
        "experts": len(encoded),
        "container_header_bytes": CONTAINER_HEADER_BYTES,
        "raw_model_bytes": model_bytes,
        "stored_model_page_bytes": model_stored,
        "raw_directory_bytes": DIRECTORY_ENTRY_BYTES * len(encoded),
        "stored_directory_page_bytes": directory_stored,
        "shared_stored_bytes": shared_stored,
        "shared_allocation_bytes_including_final_padding": shared_allocation,
        "frame_rows": rows,
        "final_container_padding_bytes": final_padding,
        "total_physical_bytes": total_bytes,
        "physical_bpw_per_source_weight": 8.0 * total_bytes / total_weights,
        "maximum_cold_read_amplification": max(row["cold_read_amplification"] for row in rows),
        "cold_read_denominator": "B_shared/E + B_private_i; B_shared includes final page padding",
        "cold_read_page_rule": "literal union of header, all model, addressed directory, and intersecting frame page indices",
    }


def encode_panel(model: QuantizedModel, streams: Sequence[SymbolStream]) -> tuple[dict[str, Any], tuple[tuple[SymbolStream, bytes, int], ...]]:
    encoded = tuple((stream, *encode_stream(model, stream)) for stream in streams)
    ledger = container_ledger(len(model.serialize()), encoded)
    return ledger, encoded


def select_on_validation(
    train: Sequence[SymbolStream], validation: Sequence[SymbolStream], bank: Sequence[Candidate]
) -> tuple[Candidate, tuple[dict[str, Any], ...]]:
    """Validation-only selection using actual finite bytes and full model charge."""
    require(bool(train) and bool(validation) and bool(bank), "selection inputs")
    rows: list[dict[str, Any]] = []
    for candidate in bank:
        model = fit_model(train, candidate)
        ledger, _ = encode_panel(model, validation)
        rows.append({
            "topology": candidate.topology,
            "chi": candidate.chi,
            "reset_symbols": candidate.reset,
            "validation_physical_bytes": ledger["total_physical_bytes"],
            "validation_physical_bpw": ledger["physical_bpw_per_source_weight"],
            "model_bytes": len(model.serialize()),
        })
    best = min(rows, key=lambda row: (
        row["validation_physical_bytes"],
        row["model_bytes"],
        row["topology"],
        row["chi"],
        row["reset_symbols"],
    ))
    selected = Candidate(str(best["topology"]), int(best["chi"]), int(best["reset_symbols"]))
    return selected, tuple(rows)


def evaluate_nested(streams: Sequence[SymbolStream], bank: Sequence[Candidate] | None = None) -> dict[str, Any]:
    """Run selection without touching the outer test, then charge final packets."""
    folds = nested_partition(streams)
    nonlocal_bank = tuple(bank) if bank is not None else candidate_bank()
    selected, selection = select_on_validation(folds["train"], folds["validation"], nonlocal_bank)
    baseline, baseline_selection = select_on_validation(folds["train"], folds["validation"], factorized_bank())
    refit = folds["train"] + folds["validation"]
    selected_model = fit_model(refit, selected)
    baseline_model = fit_model(refit, baseline)
    selected_ledger, selected_encoded = encode_panel(selected_model, folds["test"])
    baseline_ledger, baseline_encoded = encode_panel(baseline_model, folds["test"])
    for (_, payload, meaningful), stream in zip(selected_encoded, folds["test"], strict=True):
        decoded = decode_stream(selected_model, stream.roles, stream.planes, payload, meaningful)
        require(decoded == stream.symbols, "selected arithmetic round trip")
    for (_, payload, meaningful), stream in zip(baseline_encoded, folds["test"], strict=True):
        decoded = decode_stream(baseline_model, stream.roles, stream.planes, payload, meaningful)
        require(decoded == stream.symbols, "baseline arithmetic round trip")
    weights = int(selected_ledger["source_weights"])
    require(weights == int(baseline_ledger["source_weights"]), "paired panel weights")
    gain = 8.0 * (int(baseline_ledger["total_physical_bytes"]) - int(selected_ledger["total_physical_bytes"])) / weights
    uncertainty = paired_whole_layer_bootstrap(selected_ledger, baseline_ledger)
    require(math.isclose(gain, float(uncertainty["point_saving_bpw"]), rel_tol=0.0, abs_tol=2e-15), "paired point closure")
    source_absolute_survival = float(uncertainty["lower_95_saving_bpw"]) >= STANDALONE_REQUIRED_SAVING_BPW
    cold_below_2x = float(selected_ledger["maximum_cold_read_amplification"]) < 2.0
    return {
        "schema": RESULT_SCHEMA,
        "stream_view": "A_raw_normalized_gaussian_lloyd4_gray_labels",
        "claim_boundary": "Diagnostic label source census; not a complete 2.15-2.5 bpw weight codec.",
        "nonlocal_candidate_cells": len(nonlocal_bank),
        "nonlocal_candidate_bank_sha256": candidate_bank_sha256(nonlocal_bank),
        "split": {
            "train_streams": len(folds["train"]),
            "validation_streams": len(folds["validation"]),
            "test_streams": len(folds["test"]),
            "test_layers": sorted({row.layer_group for row in folds["test"]}),
            "validation_experts": sorted({row.expert_group for row in folds["validation"]}),
        },
        "selected_candidate": selected.__dict__,
        "baseline_candidate": baseline.__dict__,
        "selection_rows": selection,
        "baseline_selection_rows": baseline_selection,
        "selected_test_ledger": selected_ledger,
        "factorized_test_ledger": baseline_ledger,
        "net_nonlocal_physical_saving_bpw": gain,
        "paired_whole_layer_uncertainty": uncertainty,
        "standalone_required_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW,
        "absolute_source_survival_before_controls": source_absolute_survival,
        "controls_may_be_generated": source_absolute_survival,
        "cold_read_below_2x": cold_below_2x,
        "deployment_survival_before_controls": source_absolute_survival and cold_below_2x,
    }


def paired_whole_layer_bootstrap(selected: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Deterministic paired cluster bootstrap over untouched whole test layers.

    Shared/container bytes have no intrinsic layer identity.  Their exact total
    difference is allocated to layers in proportion to source weights solely
    for uncertainty accounting; allocated cluster numerators sum exactly to the
    literal container-byte difference.
    """
    import random

    selected_rows = selected["frame_rows"]
    baseline_rows = baseline["frame_rows"]
    require(len(selected_rows) == len(baseline_rows) and len(selected_rows) > 0, "paired ledger rows")
    local_delta: dict[str, int] = {}
    layer_weights: dict[str, int] = {}
    local_total = 0
    for left, right in zip(selected_rows, baseline_rows, strict=True):
        require(
            left["layer_group"] == right["layer_group"]
            and left["expert_group"] == right["expert_group"]
            and left["source_weights"] == right["source_weights"],
            "paired frame identity",
        )
        layer = str(left["layer_group"])
        delta = int(right["stored_frame_bytes"]) - int(left["stored_frame_bytes"])
        local_delta[layer] = local_delta.get(layer, 0) + delta
        layer_weights[layer] = layer_weights.get(layer, 0) + int(left["source_weights"])
        local_total += delta
    total_weights = sum(layer_weights.values())
    total_delta = int(baseline["total_physical_bytes"]) - int(selected["total_physical_bytes"])
    shared_delta = total_delta - local_total
    clusters = sorted(layer_weights, key=lambda value: _stable_order_key("bootstrap-layer", value))
    require(
        len(clusters) >= MIN_TEST_LAYERS,
        f"a confidence claim requires at least {MIN_TEST_LAYERS} independent whole-test-layer clusters",
    )
    allocated = {
        layer: float(local_delta[layer]) + shared_delta * layer_weights[layer] / total_weights
        for layer in clusters
    }
    require(math.isclose(math.fsum(allocated.values()), float(total_delta), rel_tol=0.0, abs_tol=1e-9), "cluster allocation closure")
    rng = random.Random(BOOTSTRAP_SEED)
    replicas: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        numerator_bytes = math.fsum(allocated[layer] for layer in sample)
        denominator_weights = sum(layer_weights[layer] for layer in sample)
        replicas.append(8.0 * numerator_bytes / denominator_weights)
    replicas.sort()
    lower_index = max(0, math.ceil(0.025 * BOOTSTRAP_REPLICATES) - 1)
    upper_index = min(BOOTSTRAP_REPLICATES - 1, math.ceil(0.975 * BOOTSTRAP_REPLICATES) - 1)
    return {
        "method": "paired whole-test-layer percentile cluster bootstrap; lower endpoint of a two-sided 95% interval",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "layers": len(clusters),
        "minimum_layers_required": MIN_TEST_LAYERS,
        "resampling_unit": "paired untouched whole layer; selected and factorized byte deltas resampled together",
        "independence_assumption": "whole test layers are the independent sampling clusters; no within-layer expert is resampled independently",
        "shared_byte_delta_allocation": "proportional to layer source weights; exact total closure",
        "point_saving_bpw": 8.0 * total_delta / total_weights,
        "lower_95_saving_bpw": replicas[lower_index],
        "upper_95_saving_bpw": replicas[upper_index],
    }


def synthetic_parity_streams(
    *, layers: int, experts: int, blocks_per_stream: int, seed: int, constrained: bool
) -> tuple[SymbolStream, ...]:
    """Source-free long-memory fixture with exactly matched bit marginals."""
    import random

    require(layers >= 3 and experts >= 3 and blocks_per_stream > 0, "fixture geometry")
    rng = random.Random(seed)
    reset = 32
    checks = 6
    rows: list[SymbolStream] = []
    for layer in range(layers):
        for expert in range(experts):
            symbols: list[int] = []
            roles: list[int] = []
            planes: list[int] = []
            for _ in range(blocks_per_stream):
                body = [rng.getrandbits(1) for _ in range(reset - checks)]
                state = 0
                fixture_candidate = Candidate("parity_sketch", 64, reset)
                for position, symbol in enumerate(body):
                    state = next_state(fixture_candidate, state, symbol, 0, 0, position)
                if constrained:
                    tail = [((state >> bit) & 1) for bit in range(checks - 1, -1, -1)]
                else:
                    tail = [rng.getrandbits(1) for _ in range(checks)]
                block = body + tail
                symbols.extend(block)
                roles.extend([0] * reset)
                planes.extend([0] * reset)
            # The fixture interprets two binary decisions as one diagnostic source
            # weight, matching the real Gray-label denominator.
            rows.append(SymbolStream(
                layer_group=f"layer-{layer}",
                expert_group=f"expert-{expert}",
                symbols=tuple(symbols),
                roles=tuple(roles),
                planes=tuple(planes),
                source_weights=len(symbols) // 2,
            ))
    return tuple(rows)


def matched_control_gate(source_result: dict[str, Any]) -> bool:
    """Controls cannot rescue an absolute source miss."""
    uncertainty = source_result.get("paired_whole_layer_uncertainty", {})
    lower = float(uncertainty.get("lower_95_saving_bpw", -math.inf)) if isinstance(uncertainty, dict) else -math.inf
    survival = bool(source_result.get("absolute_source_survival_before_controls", False))
    return survival and lower >= STANDALONE_REQUIRED_SAVING_BPW


def evaluate_raw_source_panel(
    raw_panel: Sequence[RawSwiGLUExpert],
    bank: Sequence[Candidate] | None = None,
) -> dict[str, Any]:
    """Run the raw source path while binding tensors, labels, and pipeline."""
    frozen_bank = tuple(bank) if bank is not None else candidate_bank()
    ordered = _ordered_raw_panel(raw_panel)
    streams = tuple(
        canonical_raw_lloyd4_stream(
            expert.gate,
            expert.up,
            expert.down,
            layer_group=expert.layer_group,
            expert_group=expert.expert_group,
        )
        for expert in ordered
    )
    base = evaluate_nested(streams, frozen_bank)
    source = raw_panel_provenance(ordered)
    require(
        int(base["selected_test_ledger"]["source_weights"])
        <= int(source["source_weights"]),
        "test weights cannot exceed bound source panel",
    )
    nested_sha256 = sha256_bytes(canonical_json(base))
    pipeline = {
        "schema": "label-copula-raw-source-complete-pipeline-v1",
        "source_provenance": source,
        "source_label_panel_sha256": _symbol_panel_sha256(streams),
        "candidate_cells": len(frozen_bank),
        "candidate_bank_sha256": candidate_bank_sha256(frozen_bank),
        "nested_selection_fit_pack_result_sha256": nested_sha256,
        "stages": [
            "canonical arbitrary-shape SwiGLU traversal",
            "2048-weight RMS normalization",
            "fixed Gaussian Lloyd-4 and Gray labeling",
            "rectangular reusable-slot validation",
            "whole-layer/whole-slot nested split",
            "validation-only cell selection",
            "train-plus-validation refit",
            "finite arithmetic packing and independent roundtrip",
            "model/frame/page/cold ledgers",
            "paired whole-layer cluster bootstrap",
        ],
    }
    result = dict(base)
    result["raw_source_provenance"] = source
    result["source_label_panel_sha256"] = pipeline["source_label_panel_sha256"]
    result["nested_result_before_provenance_sha256"] = nested_sha256
    result["complete_source_pipeline_sha256"] = sha256_bytes(canonical_json(pipeline))
    return result


def evaluate_independent_matched_controls(
    source_result: dict[str, Any],
    raw_panel: Sequence[RawSwiGLUExpert],
) -> dict[str, Any]:
    """Regenerate and rerun all eight controls through the full 240-cell path."""
    full_bank = candidate_bank()
    replayed_source = evaluate_raw_source_panel(raw_panel, full_bank)
    require(
        canonical_json(source_result) == canonical_json(replayed_source),
        "source result must exactly replay from the same raw tensor panel and complete 240-cell bank",
    )
    require(matched_control_gate(source_result), "source absolute lower-bound gate failed; controls forbidden")
    rows: list[dict[str, Any]] = []
    for expected_index, seed in enumerate(CONTROL_SEEDS):
        panel, generation = build_matched_gaussian_control_panel(raw_panel, seed)
        require(generation["seed_index"] == expected_index and generation["seed"] == seed, "frozen control order")
        result = evaluate_nested(panel, full_bank)
        require(result["nonlocal_candidate_cells"] == 240, "control complete 240-cell selection")
        require(result["nonlocal_candidate_bank_sha256"] == candidate_bank_sha256(full_bank), "control bank binding")
        pipeline = {
            "schema": CONTROL_PIPELINE_SCHEMA,
            "seed": seed,
            "seed_index": expected_index,
            "generation_provenance": generation,
            "candidate_cells": 240,
            "candidate_bank_sha256": candidate_bank_sha256(full_bank),
            "nested_selection_fit_pack_result_sha256": sha256_bytes(canonical_json(result)),
            "complete_stages": [
                "matched Gaussian generation from bound source block moments",
                "canonicalization/RMS/Lloyd-4/Gray pipeline",
                "rectangular reusable-slot and nested fold validation",
                "all 240 nonlocal cells fit on training and selected on validation",
                "all eight factorized reset cells selected on validation",
                "winner refit on train plus validation",
                "untouched whole-layer finite arithmetic packing and exact decode",
                "literal model/frame/alignment/page/cold ledgers",
                "paired whole-layer cluster bootstrap",
            ],
        }
        rows.append({
            "seed": seed,
            "seed_index": expected_index,
            "generation_provenance": generation,
            "result": result,
            "complete_pipeline_sha256": sha256_bytes(canonical_json(pipeline)),
        })
    gains = tuple(float(row["result"]["net_nonlocal_physical_saving_bpw"]) for row in rows)
    source_gain = float(source_result["net_nonlocal_physical_saving_bpw"])
    return {
        "schema": "label-copula-independent-matched-controls-v1",
        "control_seeds": CONTROL_SEEDS,
        "controls": tuple(rows),
        "each_control_candidate_cells": 240,
        "candidate_bank_sha256": candidate_bank_sha256(full_bank),
        "source_complete_pipeline_replayed_before_control_generation": True,
        "mean_control_nonlocal_saving_bpw": math.fsum(gains) / len(gains),
        "source_specific_excess_bpw": source_gain - math.fsum(gains) / len(gains),
        "source_pass_was_established_before_controls": True,
        "control_subtraction_used_for_source_pass": False,
    }


def make_source_free_control(source: Sequence[SymbolStream], seed: int) -> tuple[SymbolStream, ...]:
    """Independent matched-Bernoulli stand-in used only by the hostile fixture."""
    import random

    require(seed in CONTROL_SEEDS, "frozen control seed")
    rng = random.Random(seed)
    rows = []
    for stream in source:
        symbols = tuple(rng.getrandbits(1) for _ in stream.symbols)
        rows.append(SymbolStream(
            stream.layer_group,
            stream.expert_group,
            symbols,
            stream.roles,
            stream.planes,
            stream.source_weights,
        ))
    return tuple(rows)


class HeldRegularFile:
    """Hold a regular input descriptor and reject symlink/TOCTOU substitution."""

    def __init__(self, path: Path, expected_bytes: int | None = None, expected_sha256: str | None = None):
        self.path = path
        self.expected_bytes = expected_bytes
        self.expected_sha256 = expected_sha256
        self.fd: int | None = None
        self.identity: tuple[int, int, int, int] | None = None

    def __enter__(self) -> "HeldRegularFile":
        self.path = reject_symlink_path_and_ancestors(self.path)
        before = os.lstat(self.path)
        require(not _is_link_or_reparse(before), "held input symlink/reparse")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.fd = os.open(str(self.path), flags)
        row = os.fstat(self.fd)
        require(stat.S_ISREG(row.st_mode), "held input regular file")
        self.identity = (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if self.expected_bytes is not None:
            require(row.st_size == self.expected_bytes, "held input bytes")
        if self.expected_sha256 is not None:
            require(sha256_bytes(self.read_all()) == self.expected_sha256, "held input hash")
        return self

    def read_all(self) -> bytes:
        require(self.fd is not None, "held input open")
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(self.fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)

    def verify_stable(self) -> None:
        require(self.fd is not None and self.identity is not None, "held input open")
        row = os.fstat(self.fd)
        require((row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns) == self.identity, "held input changed")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class CompletionLastOutput:
    """Reserve an absent directory; COMPLETE.json is created exclusively last."""

    def __init__(self, output: Path):
        self.output = output
        self.members: list[dict[str, Any]] = []
        self._entered = False
        self._completed = False

    def __enter__(self) -> "CompletionLastOutput":
        require(not self._entered, "output writer cannot be entered twice")
        self.output = reject_symlink_path_and_ancestors(self.output, allow_missing_leaf=True)
        self.output.mkdir(parents=False, exist_ok=False)
        self._entered = True
        self.write_new("RUN_STATE.json", pretty_json({"schema": SCHEMA, "state": "INCOMPLETE"}))
        return self

    def write_new(self, name: str, payload: bytes) -> dict[str, Any]:
        require(self._entered and not self._completed, "output is not writable after completion")
        require("/" not in name and "\\" not in name and name not in (".", "..", "COMPLETE.json"), "output member name")
        path = self.output / name
        with path.open("xb") as stream:
            stream.write(payload)
        row = {"name": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        self.members.append(row)
        return row

    def complete(self, source_manifest_sha256: str) -> None:
        require(self._entered and not self._completed, "completion is irrevocable and may occur once")
        require(len(source_manifest_sha256) == 64, "source manifest hash")
        payload = pretty_json({
            "schema": f"{SCHEMA}-completion",
            "source_manifest_sha256": source_manifest_sha256,
            "members": self.members,
            "status": "COMPLETE",
        })
        path = self.output / "COMPLETE.json"
        with path.open("xb") as stream:
            stream.write(payload)
        self._completed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None
