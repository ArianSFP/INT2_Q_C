#!/usr/bin/env python3
"""Fresh-process independent decoder for one v2 reservoir staging record.

The staging record is not an extra serialized artifact.  It is the exact block
slice recovered from the checkpoint-global reservoir: u32 logical length,
raw FP16 scale, then the MSB-first arithmetic payload bytes.  Polar decoding is
provided by the frozen independent audit core, which imports no encoder code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import cupy as cp
import numpy as np

import agent_polaris_independent_decoder_v1 as core


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_record(path: Path) -> tuple[int, float, bytes, bytes]:
    container = path.read_bytes()
    if len(container) < 6:
        raise ValueError("variable-u32-fp16 record is truncated")
    logical_bits = struct.unpack("<I", container[:4])[0]
    scale = float(np.frombuffer(container[4:6], dtype="<f2")[0])
    payload = container[6:]
    expected = (logical_bits + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"logical payload needs {expected} bytes; found {len(payload)}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid FP16 scale: {scale}")
    if logical_bits % 8:
        unused = 8 - logical_bits % 8
        if payload[-1] & ((1 << unused) - 1):
            raise ValueError("nonzero block-local staging tail")
    return logical_bits, scale, payload, container


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    parameters = metadata["parameters"]
    trial_row = metadata["trials"][0]
    n = int(parameters["block_length"])
    trial = int(trial_row["trial"])
    seed = int(parameters["seed"])
    levels = int(math.log2(int(parameters["alphabet_size"])))
    sigma_source = float(parameters["sigma_source"])
    distortion = float(parameters["test_channel_distortion"])
    eta = float(parameters["eta"])

    logical_bits, scale, payload, container = read_record(args.container)
    if logical_bits != int(trial_row["arithmetic_logical_bits"]):
        raise AssertionError("reservoir length disagrees with encoder metadata")
    if hashlib.sha256(payload).hexdigest() != trial_row["arithmetic_payload_sha256"]:
        raise AssertionError("reservoir payload hash disagrees with encoder metadata")

    flags = core.frozen_map_flags(args.map, n, levels)
    reverse = core.bit_reverse_indices(n)
    layers = core.sc_layers(n)
    sigma_recon = math.sqrt(sigma_source**2 - distortion)
    alphabet_size = int(parameters["alphabet_size"])
    alphabet = eta * np.arange(
        -alphabet_size // 2 + 1, alphabet_size // 2 + 1, dtype=np.float64
    )
    weights = np.exp(-0.5 * (alphabet / sigma_recon) ** 2)
    arithmetic = core.ArithmeticBinaryDecoder(payload, logical_bits)
    previous = np.zeros(n, dtype=np.int16)
    frequency_hash = hashlib.sha256()
    selected_count = 0
    for level_index in range(levels):
        level = level_index + 1
        frozen_rng = np.random.default_rng(seed + 104_729 * trial + 1_000_003 * level)
        frozen_external = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior_lr = core.leaf_prior_ratios(weights, previous, level)
        decoded_x, frequencies = core.decode_sc_level(
            prior_lr,
            flags[level_index],
            frozen_external,
            reverse,
            layers,
            arithmetic,
        )
        previous += (1 << level_index) * decoded_x.astype(np.int16)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected_count += int(frequencies.size)

    reconstruction = (alphabet[previous] * scale).astype(np.float64)
    source_seed = seed + 104_729 * trial
    source = cp.asnumpy(
        cp.random.RandomState(source_seed).normal(
            0.0, sigma_source, size=n, dtype=cp.float64
        )
    )
    squared = np.square(source - reconstruction)
    absolute_mse = float(squared.mean(dtype=np.float64))
    relative_mse = float(squared.sum(dtype=np.float64) / np.square(source).sum(dtype=np.float64))
    encoder_absolute = float(trial_row["absolute_mse"])
    encoder_relative = float(trial_row["relative_mse"])
    tolerance = 1e-12
    absolute_match = abs(absolute_mse - encoder_absolute) <= tolerance
    relative_match = abs(relative_mse - encoder_relative) <= tolerance
    result = {
        "status": "independently decoded variable-u32-fp16 reservoir slice",
        "record_format": "variable-u32-fp16 staging slice",
        "seed": seed,
        "trial": trial,
        "synthetic_source_seed": source_seed,
        "logical_arithmetic_bits": logical_bits,
        "staging_payload_bytes": len(payload),
        "fp16_scale": scale,
        "selected_symbols_decoded": selected_count,
        "container_sha256": sha256_bytes(container),
        "payload_sha256": sha256_bytes(payload),
        "reconstruction_indices_sha256": sha256_bytes(
            previous.astype("<i2", copy=False).tobytes()
        ),
        "reconstruction_fp64_sha256": sha256_bytes(
            reconstruction.astype("<f8", copy=False).tobytes()
        ),
        "causal_frequency_u16_sha256": frequency_hash.hexdigest(),
        "decoded_absolute_mse": absolute_mse,
        "decoded_sample_relative_mse": relative_mse,
        "encoder_absolute_mse": encoder_absolute,
        "encoder_sample_relative_mse": encoder_relative,
        "absolute_mse_match_at_1e_12": absolute_match,
        "relative_mse_match_at_1e_12": relative_match,
        "passed": bool(absolute_match and relative_match),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
