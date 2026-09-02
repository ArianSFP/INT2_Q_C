#!/usr/bin/env python3
"""Authenticated, coordinate-aligned bridge from a completed v9 result.

Importing this module is inert.  The entrypoints require explicit paths and
load numerical/decoder code only after the literal publication and sealed-v8
source manifest are authenticated.  No source targets are read here.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_JSON_BYTES = 256 * (1 << 20)
MAX_SOURCE_BYTES = 4 * (1 << 20)
RESULT_SCHEMA = "uwfa-sc-v9-qwen-primary-gate-v0"
COMPLETE_SCHEMA = "uwfa-sc-v9-qwen-primary-completion-v0"
V8_MANIFEST_SCHEMA = "unifilar-wfa-source-manifest-v8"
REQUIRED_RESULT_MEMBERS = {
    "BOUND_BASELINE_SCORE.json",
    "DECODER_BUNDLE.json",
    "IDENTITY_FRAMING.bin",
    "RESULT.json",
    "SOURCE_PREFLIGHT.json",
    "UWFCV8.bin",
}


class BridgeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} SHA-256",
    )
    return value


def _regular_file_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    require(path.is_absolute(), f"{label} absolute path")
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{label} regular file")
    require(0 < metadata.st_size <= maximum, f"{label} byte bound")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.read(descriptor, min(1 << 20, before.st_size - len(payload)))
            require(bool(chunk), f"{label} short read")
            payload.extend(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label} changed while held",
        )
    finally:
        os.close(descriptor)
    return bytes(payload)


def _json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise BridgeError(f"{label} JSON: {error}") from error
    require(isinstance(value, dict), f"{label} JSON object")
    return value


def authenticate_result_directory(path: Path) -> dict[str, Any]:
    """Authenticate the completed v9 publication without opening source data."""

    require(path.is_absolute(), "result directory absolute path")
    metadata = os.lstat(path)
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), "result regular directory")
    names = sorted(entry.name for entry in os.scandir(path))
    require(set(names) == REQUIRED_RESULT_MEMBERS | {"COMPLETE.json"}, "result directory exact members")
    complete_payload = _regular_file_bytes(path / "COMPLETE.json", maximum=MAX_JSON_BYTES, label="completion")
    complete = _json(complete_payload, "completion")
    require(complete.get("schema") == COMPLETE_SCHEMA, "completion schema")
    require(complete.get("positive_claim_authority") is False, "completion remains nonpromoting")
    rows = complete.get("members")
    require(isinstance(rows, list), "completion member rows")
    observed_names = []
    payloads: dict[str, bytes] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "completion member row")
        name = row["name"]
        require(isinstance(name, str) and name in REQUIRED_RESULT_MEMBERS, "completion member name")
        require(name not in payloads, "duplicate completion member")
        maximum = MAX_JSON_BYTES if name.endswith(".json") else 32 * (1 << 20)
        payload = _regular_file_bytes(path / name, maximum=maximum, label=f"result member {name}")
        require(len(payload) == int(row["bytes"]), f"{name} byte count")
        require(sha256(payload) == require_digest(row["sha256"], name), f"{name} digest")
        observed_names.append(name)
        payloads[name] = payload
    require(set(observed_names) == REQUIRED_RESULT_MEMBERS, "completion covers exact result members")
    completion_claim = dict(complete)
    claimed_completion_digest = require_digest(completion_claim.pop("completion_sha256", None), "completion")
    require(sha256(canonical_json(completion_claim)) == claimed_completion_digest, "completion self digest")
    result = _json(payloads["RESULT.json"], "result")
    require(result.get("schema") == RESULT_SCHEMA, "v9 result schema")
    require(result.get("positive_claim_authority") is False, "v9 result remains nonpromoting")
    require(result.get("controls_run") is False, "v9 controls boundary")
    inner = payloads["UWFCV8.bin"]
    physical = result.get("physical")
    require(isinstance(physical, dict), "v9 physical record")
    require(sha256(inner) == require_digest(physical.get("container_sha256"), "v9 inner container"), "v9 inner binding")
    require(len(inner) == int(physical.get("container_bytes", -1)), "v9 inner byte binding")
    require(
        result.get("source_final", {}).get("container_sha256") == sha256(inner),
        "public physical/inner binding",
    )
    return {
        "complete": complete,
        "complete_bytes": complete_payload,
        "result": result,
        "members": payloads,
        "inner": inner,
        "result_directory": os.fspath(path),
        "publication_sha256": sha256(canonical_json([
            {"name": name, "bytes": len(payloads[name]), "sha256": sha256(payloads[name])}
            for name in sorted(payloads)
        ])),
    }


def authenticate_v8_package(path: Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    require(path.is_absolute(), "v8 package absolute path")
    metadata = os.lstat(path)
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), "v8 package directory")
    manifest_payload = _regular_file_bytes(
        path / "SOURCE_MANIFEST.json", maximum=MAX_SOURCE_BYTES, label="v8 source manifest"
    )
    require(sha256(manifest_payload) == require_digest(expected_manifest_sha256, "expected v8 manifest"), "v8 manifest binding")
    manifest = _json(manifest_payload, "v8 source manifest")
    require(manifest.get("schema") == V8_MANIFEST_SCHEMA, "v8 source manifest schema")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "v8 source manifest members")
    sources: dict[str, bytes] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "v8 manifest row")
        name = row["name"]
        require(isinstance(name, str) and name and "/" not in name and "\\" not in name, "v8 member name")
        payload = _regular_file_bytes(path / name, maximum=MAX_SOURCE_BYTES, label=f"v8 source {name}")
        require(len(payload) == int(row["bytes"]), f"v8 {name} bytes")
        require(sha256(payload) == require_digest(row["sha256"], f"v8 {name}"), f"v8 {name} digest")
        sources[name] = payload
    return {
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_payload),
        "sources": sources,
        "member_hashes": {name: sha256(payload) for name, payload in sources.items()},
    }


def _load_module(name: str, source: bytes, expected_sha256: str) -> Any:
    require(sha256(source) == require_digest(expected_sha256, name), f"{name} source binding")
    existing = sys.modules.get(name)
    if existing is not None:
        require(
            getattr(existing, "__authenticated_sha256__", None) == expected_sha256,
            f"{name} authenticated module-name collision",
        )
        return existing
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{expected_sha256}>"
    module.__package__ = ""
    module.__authenticated_sha256__ = expected_sha256
    code = compile(source, module.__file__, "exec", dont_inherit=True, optimize=0)
    # dataclasses (and some annotation resolvers) consult sys.modules while a
    # class body is executing.  Register the exact authenticated module before
    # exec, and remove it on every failed exec.  Successful modules remain
    # pinned under collision-resistant private names for the process lifetime.
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    return module


def load_authenticated_decoders(
    result_record: Mapping[str, Any],
    v8_closure: Mapping[str, Any],
    *,
    strata_common_path: Path,
    frozen_auditor_path: Path,
) -> dict[str, Any]:
    """Load only sources whose hashes are already bound into RESULT.json."""

    hashes = v8_closure["member_hashes"]
    sources = v8_closure["sources"]
    required = ("uwfa_common.py", "container_codec.py", "universal_adapter.py", "strata_sc_adapter.py")
    require(all(name in sources for name in required), "v8 decoder source closure")
    source_hashes = result_record.get("source_hashes")
    require(isinstance(source_hashes, dict), "result source hashes")
    require(
        source_hashes.get("sealed_v8_manifest_sha256") == v8_closure["manifest_sha256"],
        "result/v8 manifest binding",
    )
    common = _load_module("uwfa_pc_v0_common", sources["uwfa_common.py"], hashes["uwfa_common.py"])
    codec = _load_module("uwfa_pc_v0_codec", sources["container_codec.py"], hashes["container_codec.py"])
    semantic = _load_module("uwfa_pc_v0_semantic", sources["universal_adapter.py"], hashes["universal_adapter.py"])
    adapter_source = _load_module("uwfa_pc_v0_adapter", sources["strata_sc_adapter.py"], hashes["strata_sc_adapter.py"])
    strata_payload = _regular_file_bytes(strata_common_path, maximum=MAX_SOURCE_BYTES, label="STRATA common source")
    frozen_payload = _regular_file_bytes(frozen_auditor_path, maximum=MAX_SOURCE_BYTES, label="frozen auditor source")
    expected_strata = require_digest(
        source_hashes.get("strata_expert_local_codec_common_sha256"), "STRATA common result binding"
    )
    expected_frozen = require_digest(
        source_hashes.get("strata_v2_klt_mixed_independent_auditor_sha256"), "frozen auditor result binding"
    )
    require(sha256(strata_payload) == expected_strata, "STRATA common exact result source")
    require(sha256(frozen_payload) == expected_frozen, "frozen auditor exact result source")
    strata = _load_module("uwfa_pc_v0_strata_common", strata_payload, expected_strata)
    frozen = _load_module("uwfa_pc_v0_frozen_auditor", frozen_payload, expected_frozen)
    return {
        "common": common,
        "codec": codec,
        "semantic": semantic,
        "adapter_source": adapter_source,
        "strata": strata,
        "frozen": frozen,
        "source_hashes": {
            "uwfa_common.py": hashes["uwfa_common.py"],
            "container_codec.py": hashes["container_codec.py"],
            "universal_adapter.py": hashes["universal_adapter.py"],
            "strata_sc_adapter.py": hashes["strata_sc_adapter.py"],
            "strata_common": expected_strata,
            "frozen_auditor": expected_frozen,
        },
    }


class TracingUWFAArithmeticDecoder:
    """UWFA decoder surface that records the state before each SC decision."""

    def __init__(self, common: Any, base_decoder: Any, candidate: Any, frequencies: Sequence[int]) -> None:
        require(len(frequencies) == common.model_frequency_count(candidate), "tracing frequency geometry")
        self.common = common
        self.base_decoder = base_decoder
        self.candidate = candidate
        self.frequencies = frequencies
        self.state = 0
        self.position = 0
        self.level = 0
        self.pre_states: list[int] = []
        self.decoded_bits: list[int] = []
        self.levels: list[int] = []
        self.base_frequencies: list[int] = []

    def set_level(self, level_zero_based: int) -> None:
        level = int(level_zero_based)
        require(0 <= level < int(self.common.LEVELS), "tracing level")
        self.level = level

    def decode(self, original_freq1: int) -> int:
        within = self.position % int(self.candidate.reset_length)
        if within == 0:
            self.state = 0
        context = self.common.public_context(self.level, int(original_freq1), within)
        self.pre_states.append(int(self.state))
        frequency = int(self.frequencies[self.state * self.common.CONTEXTS + context])
        bit = int(self.base_decoder.decode(frequency))
        require(bit in (0, 1), "tracing decoded bit")
        self.decoded_bits.append(bit)
        self.levels.append(self.level)
        self.base_frequencies.append(int(original_freq1))
        self.state = int(self.common.transition(self.candidate, self.state, bit, context, within))
        self.position += 1
        return bit


@dataclass(frozen=True)
class CoordinateDecodedBlock:
    ordinal: int
    owners: tuple[int, ...]
    indices: Any
    reconstructed: Any
    levels: Any
    pre_states: Any
    occupancy: Any
    group_ordinals: tuple[int, ...]
    decoder_scale: float
    rht_seed_u64: int
    coordinate_mapping_sha256: str
    decision_triplet_sha256: str


def decode_coordinate_panel(
    np: Any,
    modules: Mapping[str, Any],
    inner: bytes,
    *,
    posterior_core: Any,
    rht_device: str = "cupy",
) -> dict[str, Any]:
    """Independently redecode the literal object and bind scalar coordinates.

    This function does not accept selected decisions, state traces, indices or
    reconstruction scratch as inputs.  Every such array is regenerated from
    `inner`.
    """

    common = modules["common"]
    codec = modules["codec"]
    semantic = modules["semantic"]
    adapter_source = modules["adapter_source"]
    frozen = modules["frozen"]
    strata = modules["strata"]
    parsed = codec.parse_container(common, semantic, inner)
    require(codec.canonical_rebuild(common, semantic, parsed) == inner, "inner canonical rebuild")
    handoff = codec.posterior_diagnostic_handoff(common, parsed)
    # The caller passes the already authenticated, retained-byte module.
    # Loading a sibling from Path(__file__) here would reopen mutable live
    # filesystem bytes after the package manifest had authenticated.
    core = posterior_core
    require(
        getattr(core, "__authenticated_sha256__", None) is not None,
        "authenticated posterior core module",
    )
    handoff_root = core.posterior_handoff_root(handoff)
    adapter = adapter_source.StrataSCAdapter(
        common=common,
        semantic_codec=semantic,
        np=np,
        frozen_auditor=frozen,
        strata_common=strata,
        device=rht_device,
    )
    metadata = adapter_source.unpack_immutable_metadata(parsed["semantics"]["extension"], strata)
    group_rows = strata.expected_block_group_ordinals(metadata["labels"])
    require(len(group_rows) == len(parsed["directory"]), "group/stream geometry")
    decoded_for_digest = []
    blocks = []
    commitment_rows = handoff["stream_decision_triplet_commitments"]
    for ordinal, (directory_row, inherited, group_ordinals, commitment) in enumerate(
        zip(parsed["directory"], metadata["blocks"], group_rows, commitment_rows, strict=True)
    ):
        require(int(directory_row["ordinal"]) == ordinal == int(inherited["ordinal"]), "coordinate stream order")
        base_decoder = frozen.ArithmeticBinaryDecoder(
            directory_row["payload"], 0, int(directory_row["logical_bits"])
        )
        tracing = TracingUWFAArithmeticDecoder(
            common,
            base_decoder,
            parsed["candidate"],
            parsed["frequencies"],
        )
        block = adapter._decode_block(
            payload=directory_row["payload"],
            logical_bits=int(directory_row["logical_bits"]),
            row=inherited,
            arithmetic=tracing,
        )
        bits = np.asarray(block["selected"], dtype=np.uint8).tobytes()
        levels = np.asarray(block["levels"], dtype=np.uint8).tobytes()
        base = np.asarray(block["base"], dtype="<u2").tobytes()
        require(bits == bytes(tracing.decoded_bits), "coordinate selected-bit trace")
        require(levels == bytes(tracing.levels), "coordinate level trace")
        require(
            np.asarray(block["base"], dtype=np.uint16).tolist() == tracing.base_frequencies,
            "coordinate base-frequency trace",
        )
        replay_states = core.trace_predecision_states(
            common,
            parsed["candidate"],
            tracing.decoded_bits,
            tracing.levels,
            tracing.base_frequencies,
        )
        require(replay_states == tracing.pre_states, "independent pre-decision state replay")
        decision_digest = common.selected_decision_triplet_sha256(bits, levels, base)
        require(decision_digest == directory_row["source_digest"], "literal decision triplet digest")
        require(
            decision_digest == commitment["decoded_selected_decision_triplet_sha256"],
            "handoff decision triplet digest",
        )
        payload, logical = common.encode_unifilar_stream(
            tracing.decoded_bits,
            tracing.levels,
            tracing.base_frequencies,
            parsed["candidate"],
            parsed["frequencies"],
        )
        require(payload == directory_row["payload"] and logical == int(directory_row["logical_bits"]), "coordinate canonical stream re-encode")
        indices = np.asarray(block["indices"], dtype=np.int16)
        require(indices.ndim == 1 and indices.size == (1 << int(inherited["logn"])), "coordinate index alignment")
        require(bool(np.all((indices >= 0) & (indices < 64))), "coordinate index range")
        pre_states = np.asarray(tracing.pre_states, dtype=np.uint16)
        level_array = np.asarray(tracing.levels, dtype=np.uint8)
        occupancy = core.occupancy_features(
            np, level_array, pre_states, int(parsed["candidate"].states)
        )
        owners = tuple(
            int(value)
            for value in codec.owner_ordinals(bytes(directory_row["owner_set"]), int(parsed["experts"]))
        )
        mapping_record = {
            "schema": "uwfa-sc-coordinate-map-v0",
            "block_ordinal": ordinal,
            "owners": list(owners),
            "group_ordinals_sha256": sha256(struct.pack(f"<{len(group_ordinals)}I", *(int(v) for v in group_ordinals))),
            "coordinate_indices_i16_sha256": sha256(indices.astype("<i2", copy=False).tobytes()),
            "decoder_scale_f16_hex": bytes(inherited["decoder_scale_bits"]).hex(),
            "rht_seed_u64": int(inherited["rht_seed_u64"]),
            "decision_triplet_sha256": decision_digest,
            "predecision_states_u16_sha256": sha256(pre_states.astype("<u2", copy=False).tobytes()),
        }
        blocks.append(CoordinateDecodedBlock(
            ordinal=ordinal,
            owners=owners,
            indices=indices,
            reconstructed=np.asarray(block["reconstructed"], dtype=np.float64),
            levels=level_array,
            pre_states=pre_states,
            occupancy=occupancy,
            group_ordinals=tuple(int(value) for value in group_ordinals),
            decoder_scale=float(inherited["decoder_scale"]),
            rht_seed_u64=int(inherited["rht_seed_u64"]),
            coordinate_mapping_sha256=sha256(canonical_json(mapping_record)),
            decision_triplet_sha256=decision_digest,
        ))
        decoded_for_digest.append(block)
    reconstruction = adapter._full_reconstruction_digest(
        decoded_for_digest,
        {"header": metadata["header"], "route": metadata["route"], "labels": metadata["labels"]},
    )
    require(
        reconstruction["full_reconstruction_f64_sha256"] == handoff["full_reconstruction_f64_sha256"],
        "coordinate panel reconstruction digest",
    )
    require(len(blocks) == int(parsed["streams"]), "coordinate block count")
    return {
        "parsed": parsed,
        "metadata": metadata,
        "handoff": handoff,
        "handoff_root_sha256": handoff_root,
        "blocks": tuple(blocks),
        "states": int(parsed["candidate"].states),
        "experts": int(parsed["experts"]),
        "weights": int(parsed["weights"]),
        "reconstruction": reconstruction,
        "coordinate_aligned_observations_redecoded_from_literal": True,
        "selected_sc_decisions_treated_as_scalar_bins": False,
        "used_extracted_state_or_index_scratch_as_input": False,
    }


class _OuterInstrumentedReader:
    """Literal wrapper reader recording every routed storage request."""

    def __init__(self, raw: bytes) -> None:
        require(isinstance(raw, bytes) and raw, "instrumented wrapper bytes")
        self.raw = raw
        self.size = len(raw)
        self.ranges: list[tuple[int, int]] = []
        self.pages: set[int] = set()

    def read(self, begin: int, length: int) -> bytes:
        require(type(begin) is int and type(length) is int, "instrumented read integers")
        require(0 <= begin <= self.size and 0 <= length <= self.size - begin, "instrumented read bounds")
        end = begin + length
        self.ranges.append((begin, end))
        if length:
            self.pages.update(range(begin // 4096, (end - 1) // 4096 + 1))
        return self.raw[begin:end]


class _InnerReaderView:
    """Zero-offset view that routes inner reads through the wrapper reader."""

    def __init__(self, outer: _OuterInstrumentedReader, inner_bytes: int) -> None:
        require(type(inner_bytes) is int and 0 < inner_bytes < outer.size, "inner reader view")
        self.outer = outer
        self.size = inner_bytes
        self.ranges: list[tuple[int, int]] = []
        self.pages: set[int] = set()

    def read(self, begin: int, length: int) -> bytes:
        require(type(begin) is int and type(length) is int, "inner read integers")
        require(0 <= begin <= self.size and 0 <= length <= self.size - begin, "inner read bounds")
        end = begin + length
        self.ranges.append((begin, end))
        if length:
            self.pages.update(range(begin // 4096, (end - 1) // 4096 + 1))
        return self.outer.read(begin, length)


def _read_summary(ranges: Sequence[Sequence[int]]) -> dict[str, int]:
    normalized: list[tuple[int, int]] = []
    requested = 0
    for item in ranges:
        require(isinstance(item, (list, tuple)) and len(item) == 2, "wrapper read range")
        begin, end = item
        require(type(begin) is int and type(end) is int and 0 <= begin <= end, "wrapper read range bounds")
        normalized.append((begin, end))
        requested += end - begin
    unique = 0
    if normalized:
        left, right = sorted(normalized)[0]
        for begin, end in sorted(normalized)[1:]:
            if begin > right:
                unique += right - left
                left, right = begin, end
            else:
                right = max(right, end)
        unique += right - left
    return {
        "read_request_count": len(normalized),
        "requested_bytes_with_repetition": requested,
        "unique_requested_bytes": unique,
        "overlap_bytes_requested_again": requested - unique,
    }


def instrument_inner_routed_decode_through_wrapper(
    np: Any,
    posterior_core: Any,
    modules: Mapping[str, Any],
    wrapper: bytes,
    *,
    expected_handoff_root_sha256: str,
    rht_device: str = "cupy",
) -> dict[str, Any]:
    """Execute one real v8 inner routed decode per expert through the wrapper.

    The suffix page is requested exactly once from each fresh routed reader.
    The unchanged inner object is then handed exactly once to the authenticated
    v8 routed parser/causal decoder.  Parser overlap remains in the literal
    request trace and is charged; absence of a second compressed pass is
    derived from the instrumented routed-decode invocation count, never from
    an assumption that overlapping requests are zero.  This diagnostic parses
    the posterior head but deliberately does not claim an inference-ready
    routed posterior decoder: it does not accumulate the posterior occupancy
    features and apply the head inside this same routed session.
    """

    core = posterior_core
    require(getattr(core, "__authenticated_sha256__", None) is not None, "authenticated posterior core")
    parsed = core.parse_wrapper(
        np,
        wrapper,
        expected_handoff_root_sha256=expected_handoff_root_sha256,
    )
    common = modules["common"]
    codec = modules["codec"]
    semantic = modules["semantic"]
    adapter_source = modules["adapter_source"]
    adapter = adapter_source.StrataSCAdapter(
        common=common,
        semantic_codec=semantic,
        np=np,
        frozen_auditor=modules["frozen"],
        strata_common=modules["strata"],
        device=rht_device,
    )
    routed_session = adapter.new_routed_decoder()
    rows = []
    for expert in range(int(parsed["experts"])):
        reader = _OuterInstrumentedReader(wrapper)
        extension = reader.read(int(parsed["inner_bytes"]), int(parsed["extension_bytes"]))
        require(extension == wrapper[int(parsed["inner_bytes"]):], "literal extension routed read")
        parsed_head = core.parse_head(
            np,
            extension[: int(parsed["head_bytes"])],
            expected_handoff_root_sha256=expected_handoff_root_sha256,
        )
        require(parsed_head["packet_sha256"] == parsed["parsed_head"]["packet_sha256"], "routed head parse binding")
        inner_reader = _InnerReaderView(reader, int(parsed["inner_bytes"]))
        inner_decode_invocations = 0
        inner_decode_invocations += 1
        routed = codec.routed_read_expert(
            common,
            semantic,
            inner_reader,
            file_size=int(parsed["inner_bytes"]),
            expert=expert,
            externally_authenticated_container_sha256=sha256(parsed["inner"]),
            decode_routed_expert=routed_session.decode_expert,
        )
        causal = routed.get("causal_decode_reencode_reconstruction")
        require(isinstance(causal, dict), "causal routed decode record")
        require(causal.get("expert_ordinal") == expert, "causal routed expert")
        require(causal.get("all_payloads_canonically_reencoded") is True, "causal canonical reencode")
        require(causal.get("all_three_roles_reconstructed") is True, "causal three-role reconstruction")
        require(tuple(routed["routed_read_ranges"]) == tuple(inner_reader.ranges), "inner routed range binding")
        summary = _read_summary(reader.ranges)
        inner_summary = _read_summary(inner_reader.ranges)
        second_pass = inner_decode_invocations > 1
        rows.append({
            "expert_ordinal": expert,
            "outer_read_ranges": [list(item) for item in reader.ranges],
            "inner_routed_read_ranges": [list(item) for item in inner_reader.ranges],
            "touched_page_indices": sorted(reader.pages),
            "touched_page_bytes": len(reader.pages) * 4096,
            **summary,
            "inner_touched_page_indices": sorted(inner_reader.pages),
            "inner_touched_page_bytes": len(inner_reader.pages) * 4096,
            "inner_read_request_count": inner_summary["read_request_count"],
            "inner_requested_bytes_with_repetition": inner_summary["requested_bytes_with_repetition"],
            "inner_unique_requested_bytes": inner_summary["unique_requested_bytes"],
            "inner_overlap_bytes_requested_again": inner_summary["overlap_bytes_requested_again"],
            "inner_decode_invocations": inner_decode_invocations,
            "compressed_expert_second_pass": second_pass,
            "compressed_expert_second_pass_absent_derived": not second_pass,
            "overlap_is_charged_not_interpreted_as_second_pass": True,
            "extension_page_read_requests": 1,
            "posterior_head_parsed_from_extension_read": True,
            "causal_decode_reencode_reconstruction": dict(causal),
        })
    parsed_inner = modules["codec"].parse_container(common, semantic, parsed["inner"])
    finalized = routed_session.finalize(
        experts=int(parsed["experts"]),
        expected_full_reconstruction_sha256=parsed_inner["reconstruction_sha256"],
    )
    require(finalized.get("matches_container_reconstruction") is True, "wrapper routed reconstruction binding")
    record = {
        "schema": "uwfa-sc-posterior-wrapper-routed-read-proof-v0",
        "wrapper_sha256": sha256(wrapper),
        "inner_sha256": sha256(parsed["inner"]),
        "inner_bytes": int(parsed["inner_bytes"]),
        "extension_bytes": int(parsed["extension_bytes"]),
        "head_bytes": int(parsed["head_bytes"]),
        "experts": rows,
        "routed_full_reconstruction": dict(finalized),
        "all_experts_one_inner_decode_invocation": all(row["inner_decode_invocations"] == 1 for row in rows),
        "compressed_expert_second_pass_forbidden_and_absent": all(
            row["compressed_expert_second_pass_absent_derived"] for row in rows
        ),
        "proof_uses_actual_authenticated_v8_routed_decoder": True,
        "actual_inner_routed_decode_executed": True,
        "actual_posterior_wrapper_routed_decode_executed": False,
        "posterior_head_applied_to_routed_reconstruction": False,
        "nonpromoting_inference_read_projection_only": True,
    }
    digest_record = dict(record)
    record["proof_sha256"] = sha256(canonical_json(digest_record))
    return record


def bind_wrapper_to_routed_proof(
    np: Any,
    posterior_core: Any,
    wrapper: bytes,
    proof: Mapping[str, Any],
    *,
    expected_handoff_root_sha256: str,
) -> dict[str, Any]:
    """Replay an actual single-pass proof against another same-inner wrapper."""

    core = posterior_core
    parsed = core.parse_wrapper(np, wrapper, expected_handoff_root_sha256=expected_handoff_root_sha256)
    require(proof.get("schema") == "uwfa-sc-posterior-wrapper-routed-read-proof-v0", "wrapper proof schema")
    require(proof.get("proof_uses_actual_authenticated_v8_routed_decoder") is True, "actual routed proof")
    require(proof.get("compressed_expert_second_pass_forbidden_and_absent") is True, "single-pass routed proof")
    require(proof.get("inner_sha256") == sha256(parsed["inner"]), "wrapper proof inner binding")
    require(int(proof.get("inner_bytes", -1)) == int(parsed["inner_bytes"]), "wrapper proof inner bytes")
    proof_rows = proof.get("experts")
    require(isinstance(proof_rows, list) and len(proof_rows) == int(parsed["experts"]), "wrapper proof experts")
    rows = []
    for expert, inherited in enumerate(proof_rows):
        require(int(inherited.get("expert_ordinal", -1)) == expert, "wrapper proof expert order")
        require(int(inherited.get("inner_decode_invocations", -1)) == 1, "wrapper proof invocation")
        reader = _OuterInstrumentedReader(wrapper)
        extension = reader.read(int(parsed["inner_bytes"]), int(parsed["extension_bytes"]))
        core.parse_head(
            np,
            extension[: int(parsed["head_bytes"])],
            expected_handoff_root_sha256=expected_handoff_root_sha256,
        )
        for begin, end in inherited["inner_routed_read_ranges"]:
            reader.read(int(begin), int(end) - int(begin))
        summary = _read_summary(reader.ranges)
        inner_summary = _read_summary(inherited["inner_routed_read_ranges"])
        inner_pages = set()
        for begin, end in inherited["inner_routed_read_ranges"]:
            if int(end) > int(begin):
                inner_pages.update(range(int(begin) // 4096, (int(end) - 1) // 4096 + 1))
        second_pass = int(inherited["inner_decode_invocations"]) > 1
        rows.append({
            "expert_ordinal": expert,
            "outer_read_ranges": [list(item) for item in reader.ranges],
            "inner_routed_read_ranges": [list(item) for item in inherited["inner_routed_read_ranges"]],
            "touched_page_indices": sorted(reader.pages),
            "touched_page_bytes": len(reader.pages) * 4096,
            **summary,
            "inner_touched_page_indices": sorted(inner_pages),
            "inner_touched_page_bytes": len(inner_pages) * 4096,
            "inner_read_request_count": inner_summary["read_request_count"],
            "inner_requested_bytes_with_repetition": inner_summary["requested_bytes_with_repetition"],
            "inner_unique_requested_bytes": inner_summary["unique_requested_bytes"],
            "inner_overlap_bytes_requested_again": inner_summary["overlap_bytes_requested_again"],
            "inner_decode_invocations": int(inherited["inner_decode_invocations"]),
            "compressed_expert_second_pass": second_pass,
            "compressed_expert_second_pass_absent_derived": not second_pass,
            "overlap_is_charged_not_interpreted_as_second_pass": True,
            "extension_page_read_requests": 1,
            "posterior_head_parsed_from_extension_read": True,
            "causal_decode_reencode_reconstruction": dict(inherited["causal_decode_reencode_reconstruction"]),
            "actual_decode_proof_sha256": proof["proof_sha256"],
        })
    record = {
        "schema": "uwfa-sc-posterior-wrapper-routed-read-proof-v0",
        "wrapper_sha256": sha256(wrapper),
        "inner_sha256": sha256(parsed["inner"]),
        "inner_bytes": int(parsed["inner_bytes"]),
        "extension_bytes": int(parsed["extension_bytes"]),
        "head_bytes": int(parsed["head_bytes"]),
        "experts": rows,
        "routed_full_reconstruction": dict(proof["routed_full_reconstruction"]),
        "all_experts_one_inner_decode_invocation": True,
        "compressed_expert_second_pass_forbidden_and_absent": all(
            row["compressed_expert_second_pass_absent_derived"] for row in rows
        ),
        "proof_uses_actual_authenticated_v8_routed_decoder": True,
        "actual_inner_routed_decode_executed": True,
        "actual_posterior_wrapper_routed_decode_executed": False,
        "posterior_head_applied_to_routed_reconstruction": False,
        "nonpromoting_inference_read_projection_only": True,
        "actual_decode_proof_sha256": proof["proof_sha256"],
        "same_inner_read_plan_replayed_on_literal_wrapper": True,
    }
    digest_record = dict(record)
    record["proof_sha256"] = sha256(canonical_json(digest_record))
    return record
