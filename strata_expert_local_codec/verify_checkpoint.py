#!/usr/bin/env python3
"""Dependency-free verification for a published expert-affine checkpoint.

This verifier deliberately does not claim to repeat the expensive causal
decode.  It authenticates every published byte through the release manifest,
reparses the physical container without importing the encoder, recomputes the
expert read ranges, and checks that the independent-audit claims are
arithmetically consistent with the bytes and the declared scientific gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any


MAGIC = b"PLRLOC3\0"
MANIFEST_SCHEMA = "strata_expert_affine_checkpoint_manifest_v1"
PLAN_SCHEMA = "strata_expert_affine_n20n21_plan_v1"
SUMMARY_SCHEMA = "strata_expert_affine_n20n21_summary_v1"
AUDIT_SCHEMA = "strata_expert_affine_independent_audit_v1"
WEIGHTS = 28_311_552
EXPERTS = 6
BLOCKS = 15
PRIVATE_BLOCKS = 12
HEADER_BYTES = 128
ROUTE_BYTES = 144
LABEL_BYTES = 5_184
DIRECTORY_RECORD = struct.Struct("<BeI")
DIRECTORY_BYTES = BLOCKS * DIRECTORY_RECORD.size
PHYSICAL_BYTES = 8_847_360
PHYSICAL_BITS = PHYSICAL_BYTES * 8
PHYSICAL_BPW = PHYSICAL_BITS / WEIGHTS
PREFIX_BYTES = HEADER_BYTES + ROUTE_BYTES + LABEL_BYTES + DIRECTORY_BYTES
CURRENT_MSE = 0.04985939119332436
TARGET_FRACTION = 0.20
GROUP_VALUES = 2_048
GROUPS = 13_824
GROUPS_PER_MATRIX = 768
# Immutable identity anchors for the precommitted Qwen panel.  The release
# manifest authenticates bytes, while these constants prevent a comprehensively
# resealed bundle from silently substituting a different route, label partition,
# or set of original BF16 sources.
EXPECTED_ROUTE_SHA256 = "94feb3564fe0c3eddfc745703f1f6001b5ae316e7146209e6b45323cdf81697c"
EXPECTED_LABELS_SHA256 = "4bb444bd14248bc72dd521b7f581700cc95e5f6d5a9e6cbea21c2119efae89e9"
EXPECTED_SOURCES_CANONICAL_SHA256 = (
    "768573dffbb7605a0993a0fd4485e4eb5fc5201529797a89d63d3c9fb18b51d6"
)
BLOCK_LOG2 = (21, 21) * EXPERTS + (20,) * 3
BLOCK_GROUPS = tuple((1 << value) // GROUP_VALUES for value in BLOCK_LOG2)
SEED_DOMAIN = b"POLARIS-STRATA-EXPERT-AFFINE-N20N21-v1\0"
CORE_ROLES = {"plan", "summary", "container", "independent_audit"}
ASSET_ROLES = {
    "asset_header_bin",
    "asset_route_bin",
    "asset_labels_3bit_bin",
    "asset_profiles_bin",
}
ENCODER_ROLES = {f"encoder_block_{ordinal:02d}" for ordinal in range(BLOCKS)}
REQUIRED_ROLES = CORE_ROLES | ASSET_ROLES | ENCODER_ROLES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def same_float(left: Any, right: float, *, atol: float = 1e-14) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def same_fp64_sum(left: Any, right: float) -> bool:
    """Compare independently ordered FP64 sums at a roundoff-sized bound."""
    try:
        return math.isclose(
            float(left),
            right,
            rel_tol=32.0 * math.ulp(1.0),
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return False


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_owner_experts(ordinal: int) -> list[int]:
    if ordinal < PRIVATE_BLOCKS:
        return [ordinal // 2]
    pair = ordinal - PRIVATE_BLOCKS
    return [2 * pair, 2 * pair + 1]


def derive_seeds(
    header: bytes, route: bytes, labels: bytes, profiles: bytes, ordinal: int
) -> tuple[int, int, str]:
    digest = hashlib.sha256(
        SEED_DOMAIN + header + route + labels + profiles + bytes((ordinal,))
    ).digest()
    return (
        int.from_bytes(digest[:4], "big") or 1,
        int.from_bytes(digest[4:12], "big"),
        digest.hex(),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_json(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{description} must be a JSON object")
    return value


def verify_seal(value: dict[str, Any], description: str) -> str:
    expected = value.get("lock_sha256")
    require(is_sha256(expected), f"{description} lock must be lowercase SHA-256")
    clean = dict(value)
    clean.pop("lock_sha256", None)
    require(
        sha256_bytes(canonical_json_bytes(clean)) == expected,
        f"{description} internal seal mismatch",
    )
    return str(expected)


def safe_relative(value: Any, description: str) -> Path:
    require(isinstance(value, str) and value, f"{description} must be a path")
    path = Path(value)
    require(not path.is_absolute() and not path.drive, f"{description} must be relative")
    require(".." not in path.parts, f"{description} escapes the release root")
    return path


def resolve_under(root: Path, value: Any, description: str) -> Path:
    path = (root / safe_relative(value, description)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise AssertionError(f"{description} escapes the release root") from error
    return path


def verify_manifest(
    release_root: Path, manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, Path]:
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    rows = manifest.get("files")
    require(isinstance(rows, list) and rows, "manifest files must be nonempty")
    roles: dict[str, Path] = {}
    seen: set[Path] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"manifest row {index} must be an object")
        role = row.get("role")
        require(isinstance(role, str) and role, f"manifest row {index} role invalid")
        require(role not in roles, f"duplicate manifest role: {role}")
        path = resolve_under(release_root, row.get("path"), f"manifest row {index} path")
        require(path not in seen, f"duplicate manifest path: {path}")
        require(path != manifest_path.resolve(), "manifest must not authenticate itself")
        require(path.is_file(), f"manifest file missing: {path}")
        size = row.get("bytes")
        digest = row.get("sha256")
        require(isinstance(size, int) and not isinstance(size, bool), "file size invalid")
        require(is_sha256(digest), "file digest invalid")
        require(path.stat().st_size == size, f"file size mismatch: {path.name}")
        require(sha256_file(path) == digest, f"file hash mismatch: {path.name}")
        roles[role] = path
        seen.add(path)
    require(REQUIRED_ROLES <= roles.keys(), "manifest lacks a required evidence role")
    return roles


def page_union_bytes(ranges: list[tuple[int, int]], page_bytes: int = 4096) -> int:
    pages: set[int] = set()
    for begin, end in ranges:
        require(0 <= begin <= end <= PHYSICAL_BYTES, "read range outside container")
        if begin != end:
            pages.update(range(begin // page_bytes, (end - 1) // page_bytes + 1))
    return len(pages) * page_bytes


def parse_route(payload: bytes) -> list[dict[str, Any]]:
    require(len(payload) == ROUTE_BYTES, "route byte length mismatch")
    roles = ("gate", "up", "down")
    rows: list[dict[str, Any]] = []
    for ordinal in range(18):
        layer, expert, role_id, axis_id, groups = struct.unpack_from(
            ">HHBBH", payload, 8 * ordinal
        )
        expected_role = ordinal % 3
        expected_axis = 1 if expected_role == 2 else 0
        require(role_id == expected_role, f"route role mismatch record {ordinal}")
        require(axis_id == expected_axis, f"route axis mismatch record {ordinal}")
        require(groups == GROUPS_PER_MATRIX, f"route group count mismatch record {ordinal}")
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "layer": layer,
                "expert": expert,
                "role": roles[role_id],
                "axis": "column" if axis_id else "row",
                "groups": groups,
            }
        )
    identities: set[tuple[int, int]] = set()
    for expert_ordinal in range(EXPERTS):
        triplet = rows[3 * expert_ordinal : 3 * expert_ordinal + 3]
        identity = (int(triplet[0]["layer"]), int(triplet[0]["expert"]))
        require(
            all((int(row["layer"]), int(row["expert"])) == identity for row in triplet),
            f"route triplet identity mismatch expert {expert_ordinal}",
        )
        require(identity not in identities, f"duplicate route expert identity: {identity}")
        identities.add(identity)
    return rows


def label_histogram(payload: bytes) -> list[int]:
    require(len(payload) == LABEL_BYTES, "label byte length mismatch")
    histogram = [0] * 8
    for ordinal in range(GROUPS):
        bit = 3 * ordinal
        byte = bit // 8
        offset = bit % 8
        word = payload[byte] << 8
        if byte + 1 < len(payload):
            word |= payload[byte + 1]
        value = (word >> (13 - offset)) & 7
        histogram[value] += 1
    require(histogram == [1728] * 8, "label histogram is not equipopulous")
    return histogram


def validate_klt_header(header: bytes) -> None:
    coefficients = struct.unpack_from("<12f", header, 32)
    codes = struct.unpack_from("<6h", header, 80)
    for expert_ordinal, code in enumerate(codes):
        require(-16384 <= code <= 16384, "KLT code outside Q15-over-pi range")
        theta = code * math.pi / 32768.0
        regenerated = (math.cos(theta), math.sin(theta))
        for component, value in enumerate(regenerated):
            actual = struct.pack("<f", coefficients[2 * expert_ordinal + component])
            expected = struct.pack("<f", value)
            require(actual == expected, "KLT coefficient is not regenerated by its code")


def parse_container(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(len(raw) == PHYSICAL_BYTES, "container physical byte count mismatch")
    header = raw[:HEADER_BYTES]
    fields = struct.unpack_from("<8sHHIIHHBBBBf", header, 0)
    require(
        fields
        == (
            MAGIC,
            1,
            HEADER_BYTES,
            0x000001FF,
            WEIGHTS,
            2048,
            13_824,
            BLOCKS,
            PRIVATE_BLOCKS,
            21,
            20,
            0.25,
        ),
        "container header constants mismatch",
    )
    validate_klt_header(header)
    route_begin = HEADER_BYTES
    labels_begin = route_begin + ROUTE_BYTES
    directory_begin = labels_begin + LABEL_BYTES
    reservoir_begin = directory_begin + DIRECTORY_BYTES
    route = raw[route_begin:labels_begin]
    labels = raw[labels_begin:directory_begin]
    require(header[92:124] == hashlib.sha256(route + labels).digest(), "asset binding mismatch")
    crc, = struct.unpack_from("<I", header, 124)
    require(crc == zlib.crc32(header[:124]) & 0xFFFFFFFF, "header CRC mismatch")
    route_rows = parse_route(route)
    histogram = label_histogram(labels)
    require(
        sha256_bytes(route) == EXPECTED_ROUTE_SHA256,
        "pinned route SHA-256 mismatch",
    )
    require(
        sha256_bytes(labels) == EXPECTED_LABELS_SHA256,
        "pinned label SHA-256 mismatch",
    )

    cursor = reservoir_begin
    directory: list[dict[str, int]] = []
    profiles = bytearray()
    logical_total = 0
    for ordinal in range(BLOCKS):
        q, scale, logical_bits = DIRECTORY_RECORD.unpack_from(
            raw, directory_begin + ordinal * DIRECTORY_RECORD.size
        )
        require(math.isfinite(scale) and scale > 0.0, f"invalid decoder scale block {ordinal}")
        profiles.append(q)
        payload_bytes = (logical_bits + 7) // 8
        end = cursor + payload_bytes
        require(end <= PHYSICAL_BYTES, f"block {ordinal} exceeds reservoir")
        padding = payload_bytes * 8 - logical_bits
        if padding:
            require(raw[end - 1] & ((1 << padding) - 1) == 0, "nonzero payload padding")
        directory.append(
            {
                "block_ordinal": ordinal,
                "profile_q": q,
                "decoder_scale": float(scale),
                "scale_fp16_hex": raw[
                    directory_begin + ordinal * DIRECTORY_RECORD.size + 1 :
                    directory_begin + ordinal * DIRECTORY_RECORD.size + 3
                ].hex(),
                "logical_bits": logical_bits,
                "payload_bytes": payload_bytes,
                "file_byte_begin": cursor,
                "file_byte_end_exclusive": end,
                "payload_sha256": sha256_bytes(raw[cursor:end]),
            }
        )
        cursor = end
        logical_total += logical_bits
    require(not any(raw[cursor:]), "terminal reservoir fill is nonzero")

    equal_share = PHYSICAL_BYTES / EXPERTS
    experts: list[dict[str, Any]] = []
    for expert in range(EXPERTS):
        required = (2 * expert, 2 * expert + 1, PRIVATE_BLOCKS + expert // 2)
        selected = [directory[index] for index in required]
        payload_bytes = sum(row["payload_bytes"] for row in selected)
        ranges = [(0, PREFIX_BYTES)] + [
            (row["file_byte_begin"], row["file_byte_end_exclusive"])
            for row in selected
        ]
        cold_bytes = PREFIX_BYTES + payload_bytes
        page_bytes = page_union_bytes(ranges)
        experts.append(
            {
                "expert_ordinal": expert,
                "required_blocks": list(required),
                "payload_bytes": payload_bytes,
                "cold_bytes": cold_bytes,
                "cold_amplification": cold_bytes / equal_share,
                "page_4k_union_bytes": page_bytes,
                "page_4k_amplification": page_bytes / equal_share,
            }
        )
    return {
        "sha256": sha256_bytes(raw),
        "physical_bytes": len(raw),
        "physical_bpw": len(raw) * 8 / WEIGHTS,
        "logical_payload_bits": logical_total,
        "used_payload_bytes": cursor - reservoir_begin,
        "zero_tail_bytes": len(raw) - cursor,
        "directory": directory,
        "asset_payloads": {
            "header.bin": header,
            "route.bin": route,
            "labels_3bit.bin": labels,
            "profiles.bin": bytes(profiles),
        },
        "route_rows": route_rows,
        "label_histogram": histogram,
        "experts": experts,
        "max_cold": max(row["cold_amplification"] for row in experts),
        "max_4k": max(row["page_4k_amplification"] for row in experts),
        "_raw": raw,
    }


def verify_read_ledger(
    read: Any, parsed: dict[str, Any], description: str
) -> None:
    require(isinstance(read, dict), f"{description} read ledger missing")
    equal_share = PHYSICAL_BYTES / EXPERTS
    require(
        same_float(read.get("equal_physical_share_bytes"), equal_share),
        f"{description} equal-share ledger mismatch",
    )
    require(same_float(read.get("max_cold"), parsed["max_cold"]), f"{description} cold read amp mismatch")
    require(same_float(read.get("max_4k"), parsed["max_4k"]), f"{description} 4-KiB read amp mismatch")
    require(
        read.get("passes_below_2x") is True and parsed["max_4k"] < 2.0,
        f"{description} read gate failed",
    )
    observed_rows = read.get("experts")
    require(
        isinstance(observed_rows, list) and len(observed_rows) == EXPERTS,
        f"{description} read rows missing",
    )
    for expected, observed in zip(parsed["experts"], observed_rows, strict=True):
        require(observed.get("expert_ordinal") == expected["expert_ordinal"], f"{description} expert ordinal mismatch")
        require(observed.get("required_blocks") == expected["required_blocks"], f"{description} read block map mismatch")
        require(observed.get("payload_bytes") == expected["payload_bytes"], f"{description} payload byte ledger mismatch")
        require(observed.get("cold_bytes") == expected["cold_bytes"], f"{description} cold byte ledger mismatch")
        require(observed.get("page_4k_union_bytes") == expected["page_4k_union_bytes"], f"{description} page ledger mismatch")
        require(
            same_float(
                observed.get("cold_amplification_vs_equal_physical_share"),
                expected["cold_amplification"],
            ),
            f"{description} expert cold amplification mismatch",
        )
        require(
            same_float(
                observed.get("page_4k_amplification_vs_equal_physical_share"),
                expected["page_4k_amplification"],
            ),
            f"{description} expert page amplification mismatch",
        )


def verify_encoder_metadata(
    roles: dict[str, Path],
    parsed: dict[str, Any],
    plan: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    encoded_rows = summary.get("encoded_blocks")
    require(
        isinstance(encoded_rows, list) and len(encoded_rows) == BLOCKS,
        "summary encoded-block coverage mismatch",
    )
    raw = parsed["_raw"]
    for ordinal in range(BLOCKS):
        sealed = plan["blocks"][ordinal]
        directory = parsed["directory"][ordinal]
        execution = encoded_rows[ordinal]
        require(isinstance(execution, dict), f"summary encoded row invalid block {ordinal}")
        require(execution.get("block_ordinal") == ordinal, f"summary encoded ordinal mismatch block {ordinal}")
        metadata_path = roles[f"encoder_block_{ordinal:02d}"]
        require(
            execution.get("metadata_sha256") == sha256_file(metadata_path),
            f"summary/encoder metadata hash mismatch block {ordinal}",
        )
        metadata = load_json(metadata_path, f"encoder metadata block {ordinal}")
        require(
            metadata.get("schema") == "strata_xklt_sc_v2_single_block_encoder_v1",
            f"encoder metadata schema mismatch block {ordinal}",
        )
        parameters = metadata.get("parameters")
        trials = metadata.get("trials")
        require(isinstance(parameters, dict), f"encoder parameters missing block {ordinal}")
        require(isinstance(trials, list) and len(trials) == 1, f"encoder trial coverage mismatch block {ordinal}")
        trial = trials[0]
        require(isinstance(trial, dict), f"encoder trial invalid block {ordinal}")
        source = trial.get("source")
        require(isinstance(source, dict), f"encoder source row missing block {ordinal}")
        require(parameters.get("block_length") == 1 << BLOCK_LOG2[ordinal], f"encoder length mismatch block {ordinal}")
        require(parameters.get("trials") == 1, f"encoder trial count mismatch block {ordinal}")
        require(parameters.get("alphabet_size") == 64, f"encoder alphabet mismatch block {ordinal}")
        require(parameters.get("decision") == "map", f"encoder decision mismatch block {ordinal}")
        require(same_float(parameters.get("eta"), 0.25), f"encoder eta mismatch block {ordinal}")
        require(same_float(parameters.get("sigma_source"), 1.0), f"encoder sigma mismatch block {ordinal}")
        require(
            same_float(parameters.get("test_channel_distortion"), float(sealed["test_distortion"])),
            f"encoder profile distortion mismatch block {ordinal}",
        )
        require(parameters.get("seed") == sealed.get("sc_seed_u32"), f"encoder SC seed mismatch block {ordinal}")
        require(source.get("block_bf16_sha256") == sealed.get("staging_sha256"), f"encoder source hash mismatch block {ordinal}")
        require(source.get("values") == 1 << BLOCK_LOG2[ordinal], f"encoder source length mismatch block {ordinal}")
        rht = source.get("rht")
        require(isinstance(rht, dict) and rht.get("enabled") is True, f"encoder RHT record missing block {ordinal}")
        require(rht.get("seed_u64") == sealed.get("rht_seed_u64"), f"encoder RHT seed mismatch block {ordinal}")

        logical_bits = int(directory["logical_bits"])
        payload_begin = int(directory["file_byte_begin"])
        payload_end = int(directory["file_byte_end_exclusive"])
        payload = raw[payload_begin:payload_end]
        require(trial.get("arithmetic_logical_bits") == logical_bits, f"encoder logical length mismatch block {ordinal}")
        require(trial.get("arithmetic_payload_bytes") == len(payload), f"encoder payload size mismatch block {ordinal}")
        require(
            trial.get("arithmetic_payload_sha256") == sha256_bytes(payload),
            f"encoder payload hash mismatch block {ordinal}",
        )
        require(trial.get("literal_container_bytes") == len(payload) + 8, f"encoder literal-container size mismatch block {ordinal}")
        decoder_scale = float(source.get("decoder_scale_fp32"))
        require(math.isfinite(decoder_scale) and decoder_scale > 0.0, f"encoder FP32 scale invalid block {ordinal}")
        legacy = struct.pack("<If", logical_bits, decoder_scale) + payload
        literal_hash = sha256_bytes(legacy)
        require(trial.get("literal_container_sha256") == literal_hash, f"encoder literal-container hash mismatch block {ordinal}")
        require(execution.get("container_sha256") == literal_hash, f"summary legacy-container hash mismatch block {ordinal}")
        expected_rms = math.sqrt(float(sealed["source_energy_fp64"]) / (1 << BLOCK_LOG2[ordinal]))
        require(same_float(source.get("block_rms_fp64"), expected_rms, atol=1e-12), f"encoder RMS mismatch block {ordinal}")
        require(
            struct.pack("<f", decoder_scale) == struct.pack("<f", expected_rms),
            f"encoder FP32 scale/plan-energy mismatch block {ordinal}",
        )
        require(execution.get("logical_bits") == logical_bits, f"summary logical length mismatch block {ordinal}")
        require(same_float(execution.get("block_rms_fp64"), expected_rms, atol=1e-12), f"summary RMS mismatch block {ordinal}")
        require(same_float(execution.get("normalized_relative_mse"), float(trial["relative_mse"])), f"summary encoder MSE mismatch block {ordinal}")
        for field in (
            "arithmetic_roundtrip_bits_match",
            "causal_decoder_frequencies_match",
            "reconstruction_indices_match",
        ):
            require(trial.get(field) is True, f"encoder {field} failed block {ordinal}")
        checks = execution.get("checks")
        require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()), f"summary encoder checks failed block {ordinal}")


def verify_evidence(
    roles: dict[str, Path], parsed: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    plan = load_json(roles["plan"], "plan")
    summary = load_json(roles["summary"], "summary")
    audit = load_json(roles["independent_audit"], "independent audit")
    require(plan.get("schema") == PLAN_SCHEMA, "plan schema mismatch")
    require(plan.get("status") == "sealed_before_arithmetic_encoding", "plan status mismatch")
    plan_lock = verify_seal(plan, "plan")
    require(summary.get("schema") == SUMMARY_SCHEMA, "summary schema mismatch")
    require(audit.get("schema") == AUDIT_SCHEMA, "audit schema mismatch")
    require(summary.get("plan_lock_sha256") == plan_lock, "summary/plan binding mismatch")
    bindings = audit.get("bindings")
    require(isinstance(bindings, dict), "audit bindings missing")
    require(bindings.get("plan_lock_sha256") == plan_lock, "audit/plan binding mismatch")
    sources = plan.get("sources")
    require(isinstance(sources, list) and len(sources) == 18, "plan source coverage mismatch")
    sources_digest = sha256_bytes(canonical_json_bytes(sources))
    require(
        sources_digest == EXPECTED_SOURCES_CANONICAL_SHA256,
        "pinned source digest mismatch",
    )
    require(
        bindings.get("sources_canonical_sha256") == sources_digest,
        "audit/source binding mismatch",
    )

    physical = plan.get("physical_ledger")
    require(isinstance(physical, dict), "plan physical ledger missing")
    expected_physical = {
        "header_bytes": HEADER_BYTES,
        "route_bytes": ROUTE_BYTES,
        "label_bytes": LABEL_BYTES,
        "directory_bytes": DIRECTORY_BYTES,
        "reservoir_bytes": PHYSICAL_BYTES - PREFIX_BYTES,
        "physical_bytes": PHYSICAL_BYTES,
        "physical_bits": PHYSICAL_BITS,
        "reserve_bits": 65_536,
    }
    for key, expected in expected_physical.items():
        require(physical.get(key) == expected, f"plan physical ledger mismatch: {key}")
    require(same_float(physical.get("physical_bpw"), PHYSICAL_BPW), "plan physical rate mismatch")

    coverage = plan.get("coverage")
    require(isinstance(coverage, dict), "plan coverage missing")
    for key, expected in {
        "experts": EXPERTS,
        "matrices": 18,
        "groups": GROUPS,
        "weights": WEIGHTS,
        "blocks": BLOCKS,
    }.items():
        require(coverage.get(key) == expected, f"plan coverage mismatch: {key}")
    require(coverage.get("every_group_once") is True, "plan group coverage failed")

    blocks = plan.get("blocks")
    require(isinstance(blocks, list) and len(blocks) == BLOCKS, "plan block rows missing")
    profiles = parsed["asset_payloads"]["profiles.bin"]
    for ordinal, (sealed_block, parsed_block) in enumerate(zip(blocks, parsed["directory"], strict=True)):
        require(sealed_block.get("block_ordinal") == ordinal, "plan block ordinal mismatch")
        logn = BLOCK_LOG2[ordinal]
        require(sealed_block.get("block_log2") == logn, f"plan block length class mismatch block {ordinal}")
        require(sealed_block.get("values") == 1 << logn, f"plan value count mismatch block {ordinal}")
        require(sealed_block.get("groups") == BLOCK_GROUPS[ordinal], f"plan group count mismatch block {ordinal}")
        require(sealed_block.get("owner_experts") == expected_owner_experts(ordinal), f"plan owner mismatch block {ordinal}")
        expected_segment = "private" if ordinal < PRIVATE_BLOCKS else "paired_tail"
        require(sealed_block.get("segment") == expected_segment, f"plan segment mismatch block {ordinal}")
        q = sealed_block.get("profile_id")
        require(isinstance(q, int) and not isinstance(q, bool) and 0 <= q <= 255, f"plan profile invalid block {ordinal}")
        require(q == parsed_block["profile_q"] == profiles[ordinal], f"plan profile mismatch block {ordinal}")
        nominal_rate = 1.75 + q / 256.0
        require(same_float(sealed_block.get("nominal_rate_bpw"), nominal_rate), f"plan nominal rate mismatch block {ordinal}")
        require(
            same_float(sealed_block.get("test_distortion"), math.exp2(-2.0 * nominal_rate)),
            f"plan test distortion mismatch block {ordinal}",
        )
        energy = float(sealed_block.get("source_energy_fp64"))
        require(math.isfinite(energy) and energy > 0.0, f"plan energy invalid block {ordinal}")
        require(is_sha256(sealed_block.get("staging_sha256")), f"plan staging hash invalid block {ordinal}")
        require(is_sha256(sealed_block.get("selected_group_ordinals_sha256")), f"plan group-map hash invalid block {ordinal}")
        expected_scale = struct.pack("<e", math.sqrt(energy / (1 << logn))).hex()
        require(parsed_block["scale_fp16_hex"] == expected_scale, f"directory scale/plan-energy mismatch block {ordinal}")
        sc_seed, rht_seed, seed_digest = derive_seeds(
            parsed["asset_payloads"]["header.bin"],
            parsed["asset_payloads"]["route.bin"],
            parsed["asset_payloads"]["labels_3bit.bin"],
            profiles,
            ordinal,
        )
        require(sealed_block.get("sc_seed_u32") == sc_seed, f"plan SC seed mismatch block {ordinal}")
        require(sealed_block.get("rht_seed_u64") == rht_seed, f"plan RHT seed mismatch block {ordinal}")
        require(sealed_block.get("seed_digest_sha256") == seed_digest, f"plan seed digest mismatch block {ordinal}")

    allocation = plan.get("allocation")
    require(isinstance(allocation, dict), "plan allocation missing")
    require(allocation.get("profile_ids") == list(profiles), "plan allocation/profile bytes mismatch")
    block_energy_sum = sum(float(row["source_energy_fp64"]) for row in blocks)
    require(
        same_float(coverage.get("cupy_energy_sum_fp64"), block_energy_sum, atol=1e-10),
        "plan block/coverage energy mismatch",
    )

    assets = plan.get("assets")
    require(isinstance(assets, dict), "plan assets missing")
    for name, payload in parsed["asset_payloads"].items():
        row = assets.get(name)
        require(isinstance(row, dict), f"plan asset missing: {name}")
        require(row.get("bytes") == len(payload), f"plan asset size mismatch: {name}")
        require(row.get("sha256") == sha256_bytes(payload), f"plan asset hash mismatch: {name}")
        role = f"asset_{name.replace('.', '_')}"
        require(roles[role].read_bytes() == payload, f"published asset/container mismatch: {name}")

    route_rows = parsed["route_rows"]
    for ordinal, (source, route) in enumerate(zip(sources, route_rows, strict=True)):
        require(isinstance(source, dict), f"plan source row invalid matrix {ordinal}")
        require(source.get("matrix_ordinal") == ordinal, f"plan source ordinal mismatch matrix {ordinal}")
        role = str(route["role"])
        shape = [2048, 768] if role == "down" else [768, 2048]
        tensor = (
            f"model.layers.{route['layer']}.mlp.experts.{route['expert']}."
            f"{role}_proj.weight"
        )
        require(source.get("role") == role, f"plan source role mismatch matrix {ordinal}")
        require(source.get("axis") == route["axis"], f"plan source axis mismatch matrix {ordinal}")
        require(source.get("shape") == shape, f"plan source shape mismatch matrix {ordinal}")
        require(source.get("tensor") == tensor, f"plan source tensor mismatch matrix {ordinal}")
        require(source.get("bytes") == 2 * GROUPS_PER_MATRIX * GROUP_VALUES, f"plan source size mismatch matrix {ordinal}")
        require(is_sha256(source.get("source_bf16_sha256")), f"plan source hash invalid matrix {ordinal}")
        safe_relative(source.get("source_relpath"), f"plan source path matrix {ordinal}")

    artifact = summary.get("artifact")
    require(isinstance(artifact, dict), "summary artifact missing")
    require(artifact.get("sha256") == parsed["sha256"], "summary/container hash mismatch")
    expected_artifact = {
        "physical_bytes": parsed["physical_bytes"],
        "physical_bits": PHYSICAL_BITS,
        "physical_bpw": parsed["physical_bpw"],
        "logical_payload_bits": parsed["logical_payload_bits"],
        "payload_bytes": parsed["used_payload_bytes"],
        "zero_reservoir_tail_bytes": parsed["zero_tail_bytes"],
    }
    for key, expected in expected_artifact.items():
        require(same_float(artifact.get(key), float(expected)), f"summary container field mismatch: {key}")
    require(summary.get("status") == "encoded_once_and_packed", "summary status mismatch")

    summary_directory = summary.get("directory")
    require(isinstance(summary_directory, list) and len(summary_directory) == BLOCKS, "summary directory coverage mismatch")
    for ordinal, (observed, expected) in enumerate(zip(summary_directory, parsed["directory"], strict=True)):
        require(observed.get("block_ordinal") == ordinal, f"summary directory ordinal mismatch block {ordinal}")
        require(observed.get("owner_experts") == expected_owner_experts(ordinal), f"summary directory owner mismatch block {ordinal}")
        for key in ("logical_bits", "payload_bytes", "file_byte_begin", "file_byte_end_exclusive", "scale_fp16_hex"):
            require(observed.get(key) == expected[key], f"summary directory mismatch {key} block {ordinal}")

    verify_read_ledger(summary.get("read_amplification"), parsed, "summary")
    for ordinal, observed in enumerate(summary["read_amplification"]["experts"]):
        route = route_rows[3 * ordinal]
        require(observed.get("layer") == route["layer"], f"summary read layer mismatch expert {ordinal}")
        require(observed.get("expert") == route["expert"], f"summary read identity mismatch expert {ordinal}")

    verify_encoder_metadata(roles, parsed, plan, summary)
    encoded_sse = sum(
        float(block["source_energy_fp64"])
        * float(execution["normalized_relative_mse"])
        for block, execution in zip(blocks, summary["encoded_blocks"], strict=True)
    )
    encoded_mse = encoded_sse / block_energy_sum
    require(
        same_float(summary.get("encoder_side_staging_mse"), encoded_mse, atol=1e-12),
        "summary encoder-side MSE mismatch",
    )
    require(
        same_float(
            summary.get("encoder_side_gaussian_gain_at_physical_rate"),
            1.0 - encoded_mse / math.exp2(-2.0 * PHYSICAL_BPW),
            atol=1e-12,
        ),
        "summary encoder-side Gaussian gain mismatch",
    )

    audit_container = audit.get("container")
    require(isinstance(audit_container, dict), "audit container record missing")
    require(audit_container.get("sha256") == parsed["sha256"], "audit/container hash mismatch")
    require(audit_container.get("physical_bytes") == PHYSICAL_BYTES, "audit physical size mismatch")
    require(same_float(audit_container.get("physical_bpw"), PHYSICAL_BPW), "audit rate mismatch")
    require(audit_container.get("logical_payload_bits") == parsed["logical_payload_bits"], "audit logical length mismatch")
    require(audit_container.get("used_payload_bytes") == parsed["used_payload_bytes"], "audit payload size mismatch")
    require(audit_container.get("zero_tail_bytes") == parsed["zero_tail_bytes"], "audit zero tail mismatch")
    decode = audit.get("decode")
    require(isinstance(decode, dict), "audit decode record missing")
    require(decode.get("decoded_blocks") == BLOCKS, "audit did not decode all blocks")
    require(decode.get("canonical_reencode_all_match") is True, "canonical re-encode failed")
    require(decode.get("every_group_once") is True, "decoded coverage failed")
    decoded_blocks = decode.get("blocks")
    require(isinstance(decoded_blocks, list) and len(decoded_blocks) == BLOCKS, "audit decoded block rows missing")
    for ordinal, (decoded, directory) in enumerate(zip(decoded_blocks, parsed["directory"], strict=True)):
        require(decoded.get("block_ordinal") == ordinal, f"audit decoded ordinal mismatch block {ordinal}")
        require(decoded.get("values") == 1 << BLOCK_LOG2[ordinal], f"audit decoded length mismatch block {ordinal}")
        require(decoded.get("profile_q") == directory["profile_q"], f"audit decoded profile mismatch block {ordinal}")
        require(decoded.get("logical_bits") == directory["logical_bits"], f"audit decoded logical length mismatch block {ordinal}")
        require(decoded.get("payload_sha256") == directory["payload_sha256"], f"audit decoded payload hash mismatch block {ordinal}")
        require(decoded.get("canonical_reencode_matches") is True, f"audit canonical re-encode failed block {ordinal}")
    require(audit.get("status") == "passed", "independent audit status failed")
    verify_read_ledger(audit.get("read_amplification"), parsed, "audit")

    source_score = audit.get("source_score")
    require(isinstance(source_score, dict), "source score missing")
    mse = float(source_score.get("energy_weighted_relative_mse"))
    sse = float(source_score.get("sse_sum_fp64"))
    energy = float(source_score.get("source_energy_sum_fp64"))
    require(math.isfinite(sse) and sse > 0.0, "source SSE invalid")
    require(math.isfinite(energy) and energy > 0.0, "source energy invalid")
    require(same_float(mse, sse / energy), "source score quotient mismatch")
    matrices = source_score.get("matrices")
    experts = source_score.get("experts")
    require(isinstance(matrices, list) and len(matrices) == 18, "matrix score coverage mismatch")
    require(isinstance(experts, list) and len(experts) == EXPERTS, "expert score coverage mismatch")
    require(
        same_fp64_sum(sum(float(row["sse_fp64"]) for row in matrices), sse),
        "matrix SSE sum mismatch",
    )
    require(
        same_fp64_sum(
            sum(float(row["source_energy_fp64"]) for row in matrices), energy
        ),
        "matrix energy sum mismatch",
    )
    for ordinal, (row, source, route) in enumerate(zip(matrices, sources, route_rows, strict=True)):
        require(row.get("matrix_ordinal") == ordinal, f"audit matrix ordinal mismatch {ordinal}")
        for key in ("tensor", "role", "axis", "shape", "source_relpath", "source_bf16_sha256"):
            require(row.get(key) == source.get(key), f"audit/source row mismatch {key} matrix {ordinal}")
        require(row.get("role") == route["role"], f"audit/route role mismatch matrix {ordinal}")
    for ordinal, row in enumerate(experts):
        require(row.get("expert_ordinal") == ordinal, f"audit expert ordinal mismatch {ordinal}")
        require(row.get("layer") == route_rows[3 * ordinal]["layer"], f"audit expert layer mismatch {ordinal}")
        require(row.get("expert") == route_rows[3 * ordinal]["expert"], f"audit expert identity mismatch {ordinal}")
        triplet = matrices[3 * ordinal : 3 * ordinal + 3]
        require(same_float(row.get("sse_fp64"), sum(float(item["sse_fp64"]) for item in triplet), atol=1e-10), f"audit expert SSE mismatch {ordinal}")
        require(same_float(row.get("source_energy_fp64"), sum(float(item["source_energy_fp64"]) for item in triplet), atol=1e-10), f"audit expert energy mismatch {ordinal}")
    for row in matrices + experts:
        require(
            same_float(
                row.get("relative_mse"),
                float(row["sse_fp64"]) / float(row["source_energy_fp64"]),
            ),
            "component MSE quotient mismatch",
        )
    require(math.isfinite(mse) and 0.0 < mse <= CURRENT_MSE, "checkpoint MSE gate failed")
    rate_relative = audit.get("rate_relative")
    require(isinstance(rate_relative, dict), "rate-relative record missing")
    gaussian = math.exp2(-2.0 * PHYSICAL_BPW)
    gain = 1.0 - mse / gaussian
    target = (1.0 - TARGET_FRACTION) * gaussian
    require(same_float(rate_relative.get("gaussian_assumed_mse"), gaussian), "Gaussian reference mismatch")
    require(same_float(rate_relative.get("physical_bpw"), PHYSICAL_BPW), "rate-relative physical rate mismatch")
    require(same_float(rate_relative.get("target_fraction"), TARGET_FRACTION), "rate-relative target fraction mismatch")
    require(same_float(rate_relative.get("mse_below_gaussian_fraction"), gain), "Gaussian gain mismatch")
    require(same_float(rate_relative.get("target_mse_at_same_rate"), target), "same-rate target mismatch")
    final_pass = mse <= target
    require(
        rate_relative.get("passes_20_percent_below_same_rate_gaussian") is final_pass,
        "same-rate final gate boolean mismatch",
    )
    milestone = audit.get("milestone_gate")
    require(isinstance(milestone, dict) and milestone.get("passed") is True, "milestone gate failed")
    require(same_float(milestone.get("current_mse_ceiling"), CURRENT_MSE), "milestone MSE ceiling mismatch")
    require(milestone.get("source_mse_passed") is (mse <= CURRENT_MSE), "milestone MSE boolean mismatch")
    require(same_float(milestone.get("max_4k_read_amplification"), parsed["max_4k"]), "milestone read value mismatch")
    require(milestone.get("read_below_2x_passed") is (parsed["max_4k"] < 2.0), "milestone read boolean mismatch")
    require(milestone.get("rate_at_or_below_2p5_passed") is True, "milestone rate boolean mismatch")

    manifest_artifact = manifest.get("artifact")
    require(isinstance(manifest_artifact, dict), "manifest artifact record missing")
    require(manifest_artifact.get("format_magic") == "PLRLOC3\\0", "manifest format magic mismatch")
    require(manifest_artifact.get("model") == "Qwen/Qwen3-30B-A3B", "manifest model mismatch")
    require(manifest_artifact.get("weights") == WEIGHTS, "manifest weight count mismatch")
    require(manifest_artifact.get("physical_bytes") == PHYSICAL_BYTES, "manifest physical bytes mismatch")
    require(manifest_artifact.get("physical_bits") == PHYSICAL_BITS, "manifest physical bits mismatch")
    require(same_float(manifest_artifact.get("physical_bpw"), PHYSICAL_BPW), "manifest artifact rate mismatch")
    require(manifest_artifact.get("container_sha256") == parsed["sha256"], "manifest artifact/container hash mismatch")

    claim = manifest.get("claim")
    require(isinstance(claim, dict), "manifest claim missing")
    require(claim.get("checkpoint_passed") is True, "manifest checkpoint claim failed")
    require(claim.get("final_rate_relative_gate_passed") is final_pass, "manifest final claim mismatch")
    require(same_float(claim.get("energy_weighted_relative_mse"), mse), "manifest MSE mismatch")
    require(same_float(claim.get("physical_bpw"), PHYSICAL_BPW), "manifest rate mismatch")
    require(same_float(claim.get("max_4k_read_amplification"), parsed["max_4k"]), "manifest read amp mismatch")
    require(same_float(claim.get("gaussian_assumed_mse"), gaussian), "manifest Gaussian reference mismatch")
    require(same_float(claim.get("target_mse_at_same_rate"), target), "manifest target MSE mismatch")

    if "tamper_report" in roles:
        tamper = load_json(roles["tamper_report"], "tamper report")
        require(tamper.get("schema") == "strata_expert_affine_tamper_tests_v1", "tamper report schema mismatch")
        require(tamper.get("status") == "passed", "tamper report status failed")
        require(tamper.get("container_sha256") == parsed["sha256"], "tamper report/container binding mismatch")
        cases_passed = tamper.get("cases_passed")
        cases_total = tamper.get("cases_total")
        require(
            isinstance(cases_passed, int)
            and isinstance(cases_total, int)
            and cases_total > 0
            and cases_passed == cases_total,
            "tamper report case coverage failed",
        )
    return {
        "checkpoint_passed": True,
        "final_rate_relative_gate_passed": final_pass,
        "physical_bpw": PHYSICAL_BPW,
        "mse": mse,
        "gaussian_assumed_mse": gaussian,
        "target_mse": target,
        "mse_below_gaussian_fraction": gain,
        "max_4k_read_amplification": parsed["max_4k"],
        "container_sha256": parsed["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    release_root = args.release_dir.resolve(strict=True)
    manifest_path = (
        args.manifest.resolve(strict=True)
        if args.manifest is not None
        else (release_root / "checkpoint_manifest.json").resolve(strict=True)
    )
    manifest = load_json(manifest_path, "release manifest")
    roles = verify_manifest(release_root, manifest_path, manifest)
    parsed = parse_container(roles["container"])
    result = verify_evidence(roles, parsed, manifest)
    print(json.dumps({"status": "passed", **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
