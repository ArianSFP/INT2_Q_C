#!/usr/bin/env python3
"""Execute the sealed expert-affine plan once and pack an exact 2.5-bpw file."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_expert_local_codec import common


def verify_plan(plan_dir: Path) -> dict[str, Any]:
    path = plan_dir / "plan.lock.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not common.verify_internal_seal(plan):
        raise ValueError("plan lock seal mismatch")
    if plan.get("status") != "sealed_before_arithmetic_encoding":
        raise ValueError("plan is not in the pre-encoding state")
    blocks = plan.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != common.BLOCKS:
        raise ValueError("plan block coverage mismatch")
    for name, row in plan["assets"].items():
        asset = plan_dir / name
        if asset.stat().st_size != row["bytes"] or common.sha256_file(asset) != row["sha256"]:
            raise ValueError(f"sealed asset changed: {name}")
    for ordinal, row in enumerate(blocks):
        if int(row["block_ordinal"]) != ordinal:
            raise ValueError("plan block ordinal mismatch")
        path = plan_dir / row["staging_relpath"]
        if path.stat().st_size != row["staging_bytes"] or common.sha256_file(path) != row["staging_sha256"]:
            raise ValueError(f"sealed staging changed block {ordinal}")
    profiles = (plan_dir / "profiles.bin").read_bytes()
    if profiles != bytes(int(row["profile_id"]) for row in blocks):
        raise ValueError("profile bytes differ from plan")
    common.validate_header(
        (plan_dir / "header.bin").read_bytes(),
        (plan_dir / "route.bin").read_bytes(),
        (plan_dir / "labels_3bit.bin").read_bytes(),
    )
    return plan


def run_block(
    workspace: Path,
    python: Path,
    encoder: Path,
    polar_repo: Path,
    plan_dir: Path,
    block: dict[str, Any],
) -> dict[str, Any]:
    ordinal = int(block["block_ordinal"])
    encoded_dir = plan_dir / "encoded"
    log_dir = plan_dir / "encode_logs"
    encoded_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    output = encoded_dir / f"block_{ordinal:02d}.json"
    container = output.with_suffix(".polar.bin")
    if output.exists() or container.exists():
        raise FileExistsError(f"one-shot output already exists block {ordinal}")
    source = plan_dir / block["staging_relpath"]
    command = [
        str(python),
        str(encoder),
        "--polar-repo", str(polar_repo),
        "--block-length", str(int(block["values"])),
        "--trials", "1",
        "--sigma-source", "1.0",
        "--test-distortion", repr(float(block["test_distortion"])),
        "--eta", "0.25",
        "--alphabet-size", "64",
        "--decision", "map",
        "--seed", str(int(block["sc_seed_u32"])),
        "--input-bf16", str(source),
        "--input-block-start", "0",
        "--canonical-source-id", f"strata-expert-affine:block:{ordinal}",
        "--canonical-block-index", "0",
        "--apply-rht",
        "--rht-seed", str(int(block["rht_seed_u64"])),
        "--emit-container-hex",
        "--output", str(output),
    ]
    process = subprocess.run(
        command,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (log_dir / f"block_{ordinal:02d}.log").write_text(
        process.stdout, encoding="utf-8", newline="\n"
    )
    if process.returncode:
        raise RuntimeError(f"encoder failed block {ordinal}; no retry")
    metadata = json.loads(output.read_text(encoding="utf-8"))
    trial = metadata["trials"][0]
    source_row = trial["source"]
    checks = {
        "schema": metadata.get("schema") == "strata_xklt_sc_v2_single_block_encoder_v1",
        "block_length": int(metadata["parameters"]["block_length"]) == int(block["values"]),
        "profile_distortion": float(metadata["parameters"]["test_channel_distortion"])
        == float(block["test_distortion"]),
        "sc_seed": int(metadata["parameters"]["seed"]) == int(block["sc_seed_u32"]),
        "rht_seed": int(source_row["rht"]["seed_u64"]) == int(block["rht_seed_u64"]),
        "source_hash": source_row["block_bf16_sha256"] == block["staging_sha256"],
        "arithmetic_roundtrip": trial["arithmetic_roundtrip_bits_match"] is True,
        "causal_frequencies": trial["causal_decoder_frequencies_match"] is True,
        "reconstruction_indices": trial["reconstruction_indices_match"] is True,
    }
    if not all(checks.values()):
        raise AssertionError(f"encoder contract failed block {ordinal}: {checks}")
    if not output.is_file() or not container.is_file():
        raise FileNotFoundError(f"encoder outputs missing block {ordinal}")
    if trial["literal_container_sha256"] != common.sha256_file(container):
        raise AssertionError(f"literal container hash mismatch block {ordinal}")
    return {
        "block_ordinal": ordinal,
        "metadata_relpath": str(output.relative_to(plan_dir)).replace("\\", "/"),
        "metadata_sha256": common.sha256_file(output),
        "container_relpath": str(container.relative_to(plan_dir)).replace("\\", "/"),
        "container_sha256": common.sha256_file(container),
        "logical_bits": int(trial["arithmetic_logical_bits"]),
        "normalized_relative_mse": float(trial["relative_mse"]),
        "block_rms_fp64": float(source_row["block_rms_fp64"]),
        "seconds": float(metadata["seconds"]),
        "checks": checks,
    }


def legacy_parts(path: Path, metadata: dict[str, Any]) -> tuple[int, float, bytes]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated encoder container: {path}")
    logical_bits, scale = struct.unpack("<If", raw[:8])
    payload = raw[8:]
    if len(payload) != (logical_bits + 7) // 8:
        raise ValueError("noncanonical byte-padded payload")
    padding = len(payload) * 8 - logical_bits
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ValueError("nonzero low padding bits")
    trial = metadata["trials"][0]
    if logical_bits != int(trial["arithmetic_logical_bits"]):
        raise ValueError("metadata/container logical length mismatch")
    if hashlib.sha256(payload).hexdigest() != trial["arithmetic_payload_sha256"]:
        raise ValueError("metadata/container payload hash mismatch")
    return int(logical_bits), float(scale), payload


def page_union_bytes(ranges: list[tuple[int, int]], page_bytes: int = 4096) -> int:
    pages: set[int] = set()
    for begin, end in ranges:
        if not 0 <= begin <= end:
            raise ValueError("invalid byte range")
        if begin == end:
            continue
        pages.update(range(begin // page_bytes, (end - 1) // page_bytes + 1))
    return len(pages) * page_bytes


def pack(plan_dir: Path, plan: dict[str, Any], encoded: list[dict[str, Any]]) -> dict[str, Any]:
    encoded = sorted(encoded, key=lambda row: int(row["block_ordinal"]))
    if [int(row["block_ordinal"]) for row in encoded] != list(range(common.BLOCKS)):
        raise ValueError("encoded block coverage mismatch")
    directory = bytearray()
    streams = bytearray()
    payload_rows: list[dict[str, Any]] = []
    energy_sse = 0.0
    energy_total = 0.0
    for block, row in zip(plan["blocks"], encoded, strict=True):
        metadata_path = plan_dir / row["metadata_relpath"]
        container_path = plan_dir / row["container_relpath"]
        if common.sha256_file(metadata_path) != row["metadata_sha256"]:
            raise ValueError("encoded metadata changed before pack")
        if common.sha256_file(container_path) != row["container_sha256"]:
            raise ValueError("encoded container changed before pack")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        logical_bits, _, payload = legacy_parts(container_path, metadata)
        rms = float(metadata["trials"][0]["source"]["block_rms_fp64"])
        scale_fp16 = np.float16(rms)
        if not np.isfinite(scale_fp16) or scale_fp16 <= 0:
            raise ValueError("invalid FP16 decoder scale")
        directory.extend(
            struct.pack("<BeI", int(block["profile_id"]), float(scale_fp16), logical_bits)
        )
        file_begin = common.HEADER_BYTES + common.ROUTE_BYTES + common.LABEL_BYTES + common.DIRECTORY_BYTES + len(streams)
        streams.extend(payload)
        payload_rows.append(
            {
                "block_ordinal": int(block["block_ordinal"]),
                "owner_experts": block["owner_experts"],
                "logical_bits": logical_bits,
                "payload_bytes": len(payload),
                "file_byte_begin": file_begin,
                "file_byte_end_exclusive": file_begin + len(payload),
                "scale_fp16_hex": struct.pack("<e", float(scale_fp16)).hex(),
            }
        )
        energy = float(block["source_energy_fp64"])
        energy_total += energy
        energy_sse += energy * float(row["normalized_relative_mse"])
    if len(directory) != common.DIRECTORY_BYTES:
        raise AssertionError("directory size mismatch")
    if len(streams) > common.RESERVOIR_BYTES:
        common.write_json(
            plan_dir / "RATE_FAILURE.json",
            {
                "status": "sealed plan overflow; no retry",
                "payload_bytes": len(streams),
                "reservoir_bytes": common.RESERVOIR_BYTES,
            },
        )
        raise RuntimeError("sealed arithmetic payloads overflow physical reservoir")
    reservoir = bytes(streams) + bytes(common.RESERVOIR_BYTES - len(streams))
    header = (plan_dir / "header.bin").read_bytes()
    route = (plan_dir / "route.bin").read_bytes()
    labels = (plan_dir / "labels_3bit.bin").read_bytes()
    common.validate_header(header, route, labels)
    artifact = header + route + labels + bytes(directory) + reservoir
    if len(artifact) != common.PHYSICAL_BYTES:
        raise AssertionError("packed physical byte count mismatch")
    artifact_path = plan_dir / "strata_expert_affine_n20n21.bin"
    artifact_path.write_bytes(artifact)

    equal_share = common.PHYSICAL_BYTES / common.EXPERTS
    prefix = common.HEADER_BYTES + common.ROUTE_BYTES + common.LABEL_BYTES + common.DIRECTORY_BYTES
    read_rows = []
    for expert_ordinal in range(common.EXPERTS):
        required = common.expert_required_blocks(expert_ordinal)
        selected = [payload_rows[index] for index in required]
        payload_bytes = sum(int(row["payload_bytes"]) for row in selected)
        ranges = [(0, prefix)] + [
            (int(row["file_byte_begin"]), int(row["file_byte_end_exclusive"]))
            for row in selected
        ]
        cold_bytes = prefix + payload_bytes
        page_bytes = page_union_bytes(ranges)
        read_rows.append(
            {
                "expert_ordinal": expert_ordinal,
                "layer": int(common.parse_route(route)[3 * expert_ordinal]["layer"]),
                "expert": int(common.parse_route(route)[3 * expert_ordinal]["expert"]),
                "required_blocks": list(required),
                "payload_bytes": payload_bytes,
                "cold_bytes": cold_bytes,
                "cold_amplification_vs_equal_physical_share": cold_bytes / equal_share,
                "page_4k_union_bytes": page_bytes,
                "page_4k_amplification_vs_equal_physical_share": page_bytes / equal_share,
            }
        )
    summary = {
        "schema": "strata_expert_affine_n20n21_summary_v1",
        "status": "encoded_once_and_packed",
        "plan_lock_sha256": plan["lock_sha256"],
        "artifact": {
            "relpath": artifact_path.name,
            "sha256": common.sha256_file(artifact_path),
            "physical_bytes": len(artifact),
            "physical_bits": len(artifact) * 8,
            "physical_bpw": len(artifact) * 8 / common.WEIGHTS,
            "logical_payload_bits": sum(int(row["logical_bits"]) for row in payload_rows),
            "payload_bytes": len(streams),
            "zero_reservoir_tail_bytes": common.RESERVOIR_BYTES - len(streams),
        },
        "encoded_blocks": encoded,
        "directory": payload_rows,
        "read_amplification": {
            "definition": "cold bytes fetched divided by one-sixth of physical container bytes",
            "equal_physical_share_bytes": equal_share,
            "experts": read_rows,
            "max_cold": max(row["cold_amplification_vs_equal_physical_share"] for row in read_rows),
            "max_4k": max(row["page_4k_amplification_vs_equal_physical_share"] for row in read_rows),
            "passes_below_2x": max(row["page_4k_amplification_vs_equal_physical_share"] for row in read_rows) < 2.0,
        },
        "encoder_side_staging_mse": energy_sse / energy_total,
        "encoder_side_gaussian_gain_at_physical_rate": common.gaussian_gain(
            energy_sse / energy_total, 2.5
        ),
        "claim_boundary": "independent causal decode and original-BF16 source-domain audit required",
    }
    common.write_json(plan_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--polar-repo", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    plan_dir = args.plan_dir.resolve(strict=True)
    python = args.python.absolute()
    encoder = args.encoder.resolve(strict=True)
    polar_repo = args.polar_repo.resolve(strict=True)
    plan = verify_plan(plan_dir)
    encoded: list[dict[str, Any]] = []
    for block in plan["blocks"]:
        encoded.append(
            run_block(workspace, python, encoder, polar_repo, plan_dir, block)
        )
    summary = pack(plan_dir, plan, encoded)
    print(json.dumps({
        "status": summary["status"],
        "artifact": summary["artifact"],
        "max_4k_read_amplification": summary["read_amplification"]["max_4k"],
        "encoder_side_staging_mse": summary["encoder_side_staging_mse"],
    }, indent=2))


if __name__ == "__main__":
    main()
