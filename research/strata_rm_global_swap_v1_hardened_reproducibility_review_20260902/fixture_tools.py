#!/usr/bin/env python3
"""Independent synthetic fixture construction for byte-authority review."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Callable

from independent_auth import EXTERNAL_PINS


MAGIC = b"SRMGF1\0\0"
PREFIX = struct.Struct("<8sII")
TRAILER = struct.Struct("<I")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def bf16(values) -> bytes:
    words = []
    for value in values:
        bits, = struct.unpack("<I", struct.pack("<f", float(value)))
        words.append((bits >> 16) & 0xFFFF)
    return struct.pack("<" + "H" * len(words), *words)


def fixture_packet(reconstructions) -> bytes:
    chunks = [struct.pack("<" + "d" * len(row), *row)
              for row in reconstructions]
    header = canonical({
        "schema": "strata-rm-v1-authority-fixture-packet",
        "reconstruction_f64_bytes": [len(chunk) for chunk in chunks],
    })
    payload = b"".join(chunks)
    prefix = PREFIX.pack(MAGIC, len(header), len(payload))
    packet = prefix + header + payload
    return packet + TRAILER.pack(zlib.crc32(packet) & 0xFFFFFFFF)


def make_fixture(root: Path, *, producer: Path, external_root: Path,
                 mutate_packet: Callable[[bytes], bytes] | None = None,
                 mutate_commitment: Callable[[dict], None] | None = None):
    """Create only tiny synthetic source/packet bytes under ``root``."""
    evidence = root / "evidence"
    evidence.mkdir()
    values = (
        [1.0, -2.0, 3.0, -4.0],
        [0.5, 1.5, -2.5, 3.5],
        [2.0, 2.0, -1.0, -1.0],
    )
    sources = []
    for ordinal, (role, row) in enumerate(
            zip(("gate", "up", "down"), values, strict=True)):
        payload = bf16(row)
        name = f"source-{ordinal}.bf16"
        (evidence / name).write_bytes(payload)
        sources.append({
            "ordinal": ordinal, "role": role, "layer": 0, "expert": 0,
            "shape": [2, 2], "source_relative_path": name,
            "source_bytes": len(payload),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
        })
    packet = fixture_packet(values)
    if mutate_packet is not None:
        packet = mutate_packet(packet)
    (evidence / "packet.bin").write_bytes(packet)
    worker = producer / "fixture_decoder_worker.py"
    worker_relative = worker.resolve(strict=True).relative_to(
        external_root.resolve(strict=True)).as_posix()
    commitment = {
        "schema": "strata-rm-global-swap-v1-physical-commitment",
        "mode": "synthetic_authority_fixture",
        "v0_source_root_sha256":
            "4f856e268d37ee1d6f32b4a2d1b8cd6879c235639ad75809ffd75fc7c4372d6c",
        "v0_audit_source_root_sha256":
            "7eabe4580908d4a79eceb2f7fdaf838d535028c06263c2f4841032664db11ad0",
        "external_pins": dict(EXTERNAL_PINS),
        "decoder_worker": {
            "relative_path": worker_relative,
            "sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "protocol": "strata-rm-v1-decoder-worker-protocol",
            "independent_from_encoder": True,
            "independent_audit": None,
        },
        "cases": [{
            "case_id": "synthetic-review-fixture", "kind": "synthetic_fixture",
            "architecture_family": "synthetic", "pipeline_id": "fixture-v1",
            "matched_case_id": None,
            "packet": {"relative_path": "packet.bin", "bytes": len(packet),
                       "sha256": hashlib.sha256(packet).hexdigest()},
            "sources": sources, "charged_shared_bytes": 0,
        }],
        "universal_contract": {
            "roles": ["gate", "up", "down"], "shape_parameterized": True,
            "qwen_specific_tables": False, "model_family_agnostic": True,
            "architecture_families": ["synthetic"],
        },
        "shared_model_bytes": 0, "selection_frozen_before_test": True,
        "test_bytes_opened_during_selection": False,
    }
    if mutate_commitment is not None:
        mutate_commitment(commitment)
    commitment_payload = canonical(commitment) + b"\n"
    commitment_path = evidence / "commitment.json"
    commitment_path.write_bytes(commitment_payload)
    return {
        "evidence": evidence, "commitment_path": commitment_path,
        "commitment_sha256": hashlib.sha256(commitment_payload).hexdigest(),
        "commitment": commitment,
    }
