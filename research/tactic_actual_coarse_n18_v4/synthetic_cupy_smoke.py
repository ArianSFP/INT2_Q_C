#!/usr/bin/env python3
"""Explicitly authorized source-free CuPy encode/independent-decode smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import independent_decoder
import numeric_encoder
from packet_format import N, ExpertGeometry, require


AUTHORIZATION = "SYNTHETIC_ONLY_TACN18_V4_CUPY_SMOKE"


def _fp32_to_bf16(value: float) -> int:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    upper = word >> 16
    lower = word & 0xFFFF
    increment = lower > 0x8000 or (lower == 0x8000 and upper & 1)
    return (upper + int(increment)) & 0xFFFF


def fixture() -> bytes:
    rng = random.Random(0x5441434E31385634)
    output = bytearray(2 * N)
    for index in range(N):
        struct.pack_into("<H", output, 2 * index, _fp32_to_bf16(rng.gauss(0.0, 1.0)))
    return bytes(output)


def run(repo_root: Path) -> dict[str, object]:
    geometry = ExpertGeometry(N, 1)
    source = fixture()
    encoder_runtime = numeric_encoder.load_encoder_runtime(repo_root)
    packet, encode = numeric_encoder.encode_tile(source, geometry, 0, 0, encoder_runtime)
    decoder_runtime = independent_decoder.load_decoder_runtime(repo_root)
    decoded = independent_decoder.decode_reservoir(packet, decoder_runtime)
    require(decoded.canonical_packet == packet, "synthetic canonical packet equality")
    np = decoder_runtime.numpy
    words = np.frombuffer(source, dtype="<u2")
    original = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    reconstruction = decoded.reconstruction_f32.astype(np.float64)
    residual = original - reconstruction
    sse = float(np.dot(residual, residual))
    energy = float(np.dot(original, original))
    require(energy > 0.0, "synthetic source energy")
    cp = encoder_runtime.cupy
    device = cp.cuda.Device()
    return {
        "schema": "tactic_actual_coarse_n18_v4_synthetic_cupy_smoke",
        "status": "PASS_SOURCE_FREE_NUMERICAL_ENCODE_INDEPENDENT_DECODE_REENCODE",
        "source_kind": "deterministic Python-gaussian BF16 fixture; no model payload",
        "source_bf16_sha256": hashlib.sha256(source).hexdigest(),
        "packet_bytes": len(packet),
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "physical_bpw": 8.0 * len(packet) / N,
        "logical_bits": encode["logical_bits"],
        "capacity_margin_bits": encode["capacity_margin_bits"],
        "relative_mse_original_coordinates": sse / energy,
        "canonical_reencode_matches": True,
        "encoder": encode,
        "decoder": decoded.report,
        "cuda_device_id": int(device.id),
        "cuda_compute_capability": list(device.compute_capability),
        "claim_boundary": "source-free mechanics only; no Qwen/Tactic/F result or runtime freeze",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    arguments = parser.parse_args()
    if arguments.authorization != AUTHORIZATION:
        raise SystemExit("authorization mismatch; no numerical import")
    result = run(arguments.repo_root.resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
