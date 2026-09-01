#!/usr/bin/env python3
"""Source-only hostile reproducer for the sealed UWFA-SC v2 producer.

This script opens only the producer source package. It authenticates the exact
manifest and every declared source byte before compiling buffered snapshots.
It never accesses Qwen/model/current-artifact/extracted/control payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import types
import zlib


PINNED_MANIFEST_SHA256 = "223a96585444a0b3e4344c470e243dbd4b84662fddfda881185e879a4caee693"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path: Path) -> bytes:
    if not path.is_absolute():
        raise RuntimeError("absolute producer path required")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        if part in {".", ".."}:
            raise RuntimeError("noncanonical path")
        cursor /= part
        info = os.lstat(cursor)
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"symlink component: {cursor}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("source member is not regular")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def snapshot_module(name: str, source: bytes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<audited-snapshot:{name}>"
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


def authenticate(package: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    manifest_bytes = regular_bytes(package / "SOURCE_MANIFEST.json")
    if sha(manifest_bytes) != PINNED_MANIFEST_SHA256:
        raise RuntimeError("producer manifest pin mismatch")
    manifest = json.loads(manifest_bytes)
    rows = manifest["members"]
    declared = {"SOURCE_MANIFEST.json"}
    snapshots: dict[str, bytes] = {}
    for row in rows:
        if set(row) != {"name", "bytes", "sha256"}:
            raise RuntimeError("manifest member schema")
        name = row["name"]
        if not isinstance(name, str) or name != Path(name).name or name in declared:
            raise RuntimeError("manifest member name")
        data = regular_bytes(package / name)
        if len(data) != row["bytes"] or sha(data) != row["sha256"]:
            raise RuntimeError(f"source member mismatch: {name}")
        snapshots[name] = data
        declared.add(name)
    actual = {entry.name for entry in os.scandir(package)}
    if actual != declared:
        raise RuntimeError("undeclared or missing source member")
    return snapshots, manifest


def binding_hashes() -> dict[str, str]:
    return {
        "baseline_plan_sha256": "41" * 32,
        "baseline_score_sha256": "42" * 32,
        "universal_decoder_sha256": "43" * 32,
        "producer_manifest_sha256": PINNED_MANIFEST_SHA256,
        "audit_bootstrap_sha256": "45" * 32,
        "source_panel_sha256": "46" * 32,
        "extraction_program_sha256": "47" * 32,
    }


def build_minimal(common: object, codec: object, experts: int, owner_masks: list[int]) -> bytes:
    candidate = common.Candidate("suffix", 2, 32)
    frequencies = [32768] * common.model_frequency_count(candidate)
    model = common.serialize_model(candidate, frequencies)
    specs = []
    for ordinal, owner_mask in enumerate(owner_masks):
        stream = codec.StreamSpec(
            ordinal=ordinal,
            symbols=1,
            logical_bits=2,
            payload=b"@",
            source_digest=(f"{ordinal + 1:064x}"),
            profile_q=0,
            decoder_scale=1.0,
        )
        specs.append(codec.RegionSpec(owner_mask, (stream,)))
    container, _metrics = codec.build_container(
        common,
        model_packet=model,
        immutable_state=b"source-only",
        regions=specs,
        weights=1_000_000,
        experts=experts,
        baseline_object_bytes=1_000_000,
        audited_relative_mse=0.025,
        baseline_artifact_sha256="11" * 32,
        reconstruction_sha256="22" * 32,
        audit_binding_sha256="33" * 32,
        binding_hashes=binding_hashes(),
    )
    return container


def reseal_experts(codec: object, container: bytes, experts: int) -> bytes:
    header = bytearray(container[: codec.HEADER_BYTES])
    struct.pack_into("<I", header, 28, experts)
    header[codec._HEADER_SEAL_BEGIN : codec._HEADER_SEAL_END] = bytes(32)
    struct.pack_into("<I", header, codec._HEADER_CRC_OFFSET, 0)
    header[codec._HEADER_SEAL_BEGIN : codec._HEADER_SEAL_END] = hashlib.sha256(header).digest()
    struct.pack_into("<I", header, codec._HEADER_CRC_OFFSET, zlib.crc32(header) & 0xFFFFFFFF)
    return bytes(header) + container[codec.HEADER_BYTES :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("producer", type=Path)
    args = parser.parse_args()
    package = args.producer
    snapshots, _manifest = authenticate(package)
    common = snapshot_module("uwfa_v2_audit_common", snapshots["uwfa_common.py"])
    protocol = snapshot_module("uwfa_v2_audit_protocol", snapshots["protocol.py"])
    codec = snapshot_module("uwfa_v2_audit_codec", snapshots["container_codec.py"])

    six = build_minimal(common, codec, 6, [1 << expert for expert in range(6)])
    parsed_six = codec.parse_container(common, six)

    above = reseal_experts(codec, six, protocol.MAX_EXPERTS + 1)
    parsed_above = codec.parse_container(common, above)

    high_owner_error = "NO_ERROR"
    try:
        build_minimal(common, codec, 128, [1 << 127])
    except Exception as exc:  # exact hostile observation
        high_owner_error = f"{type(exc).__name__}: {exc}"

    degenerate = build_minimal(common, codec, 128, [1])
    parsed_degenerate = codec.parse_container(common, degenerate)
    owned = {
        expert
        for expert in range(128)
        if any(int(region["owner_mask"]) & (1 << expert) for region in parsed_degenerate["regions"])
    }

    result = {
        "schema": "uwfa-sc-v2-source-only-blocker-reproduction-v1",
        "producer_manifest_sha256": PINNED_MANIFEST_SHA256,
        "six_expert_parse": {
            "accepted_experts": int(parsed_six["experts"]),
            "container_bytes": len(six),
        },
        "unbounded_parser": {
            "protocol_MAX_EXPERTS": int(protocol.MAX_EXPERTS),
            "resealed_experts": int(parsed_above["experts"]),
            "accepted": int(parsed_above["experts"]) == int(protocol.MAX_EXPERTS) + 1,
        },
        "expert_128_high_owner": {
            "owner_mask": "1<<127",
            "build_error": high_owner_error,
            "container_region_frame_directory_owner_width_bits": 32,
            "inherited_metadata_owner_width_bits": 16,
        },
        "expert_128_degenerate_low_owner": {
            "accepted_experts": int(parsed_degenerate["experts"]),
            "owned_experts": sorted(owned),
            "empty_expert_count": 128 - len(owned),
        },
        "payload_access": False,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    result["reproduction_sha256"] = hashlib.sha256(encoded).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if high_owner_error == "NO_ERROR" or not result["unbounded_parser"]["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
