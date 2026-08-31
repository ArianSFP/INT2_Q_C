#!/usr/bin/env python3
"""Validate the independent v2 decoder core on the known opened N20 KLT block.

This consumes only the already-opened development block/result.  It does not
import an encoder and does not read any blind-v2 artifact.  Equality of the
decoded normalized MSE to the encoder's recorded metric validates the full
procedural-BEC construction, causal arithmetic decode, reconstruction indices,
and inverse signed RHT together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

import strata_v2_klt_mixed_independent_auditor_v1 as decoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--source-bf16", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--inverse-device", choices=("cupy", "numpy"), default="cupy")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if fixture.get("schema") != "strata_v2_known_n20_opened_fixture_v1":
        raise AssertionError("known-N20 fixture schema mismatch")
    if decoder.sha256_file(args.result) != fixture.get("encoder_result_sha256"):
        raise AssertionError("fixture/encoder-result hash mismatch")
    if decoder.sha256_file(args.container) != fixture.get("literal_container_sha256"):
        raise AssertionError("fixture/literal-container hash mismatch")
    source_payload = args.source_bf16.read_bytes()
    if hashlib.sha256(source_payload).hexdigest() != fixture.get("source_bf16_sha256"):
        raise AssertionError("fixture/source hash mismatch")
    metadata = json.loads(args.result.read_text(encoding="utf-8"))
    parameters = metadata["parameters"]
    expected = metadata["trials"][0]
    n = int(parameters["block_length"])
    if n != 1 << 20:
        raise ValueError(n)
    rate = -0.5 * math.log2(float(parameters["test_channel_distortion"]))
    profile_q = int(round((rate - 1.75) * 256.0))
    if not math.isclose(rate, 1.75 + profile_q / 256.0, rel_tol=0.0, abs_tol=1e-14):
        raise AssertionError((rate, profile_q))
    profile = decoder.profile_parameters(profile_q, float(parameters["eta"]))
    if not np.allclose(profile["capacities"], parameters["capacity_schedule"], rtol=0.0, atol=2e-15):
        raise AssertionError("independent capacity schedule mismatch")

    literal = args.container.read_bytes()
    if hashlib.sha256(literal).hexdigest() != fixture["literal_container_sha256"]:
        raise AssertionError("literal container changed after binding check")
    logical_bits, scale = struct.unpack("<If", literal[:8])
    payload = literal[8:]
    if logical_bits != int(expected["arithmetic_logical_bits"]):
        raise AssertionError("logical length mismatch")
    if hashlib.sha256(payload).hexdigest() != expected["arithmetic_payload_sha256"]:
        raise AssertionError("payload hash mismatch")
    source_row = expected["source"]
    if hashlib.sha256(source_payload).hexdigest() != source_row["block_bf16_sha256"]:
        raise AssertionError("metadata/source retained-byte hash mismatch")
    if float(scale) != float(source_row["decoder_scale_fp32"]):
        raise AssertionError("serialized/metadata decoder-scale mismatch")
    reverse = decoder.bit_reverse_indices(n)
    layers = decoder.sc_layers(n)
    flags = decoder.bec_freeze_flags(n, profile["capacities"], reverse)
    arithmetic = decoder.ArithmeticBinaryDecoder(payload, 0, logical_bits)
    alphabet = float(parameters["eta"]) * np.arange(-31, 33, dtype=np.float64)
    weights = np.exp(-0.5 * (alphabet / float(profile["sigma_reconstruction"])) ** 2)
    previous = np.zeros(n, dtype=np.int16)
    frequency_hash = hashlib.sha256()
    selected = 0
    selected_chunks = []
    frequency_chunks = []
    seed = int(parameters["seed"])
    for level_index, flag in enumerate(flags):
        level = level_index + 1
        frozen = np.random.default_rng(seed + 1_000_003 * level).integers(
            0, 2, size=n, dtype=np.uint8
        )
        prior = decoder.leaf_prior_ratios(weights, previous, level)
        x_bit, frequencies, selected_values = decoder.decode_sc_level(
            prior, flag, frozen, reverse, layers, arithmetic
        )
        previous += (1 << level_index) * x_bit.astype(np.int16)
        selected += int(frequencies.size)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected_chunks.append(selected_values)
        frequency_chunks.append(frequencies)

    canonical_payload, canonical_logical_bits = decoder.arithmetic_encode_binary(
        np.concatenate(selected_chunks), np.concatenate(frequency_chunks)
    )
    if canonical_logical_bits != logical_bits:
        raise AssertionError("canonical arithmetic logical length mismatch")
    if canonical_payload != payload:
        raise AssertionError("canonical arithmetic payload-byte mismatch")

    normalized_reconstruction = decoder.inverse_signed_rht(
        alphabet[previous], int(expected["source"]["rht"]["seed_u64"]), args.inverse_device
    )
    words = np.frombuffer(source_payload, dtype="<u2")
    source = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    if source.size != n:
        raise ValueError(source.size)
    block_rms = float(expected["source"]["block_rms_fp64"])
    normalized_source = source / block_rms
    squared = (normalized_source - normalized_reconstruction) ** 2
    relative_mse = float(np.sum(squared) / np.sum(normalized_source**2))
    expected_mse = float(expected["relative_mse"])
    physical_reconstruction = normalized_reconstruction * float(scale)
    physical_squared = (source - physical_reconstruction) ** 2
    serialized_scale_relative_mse = float(
        np.sum(physical_squared, dtype=np.float64)
        / np.sum(source * source, dtype=np.float64)
    )
    result = {
        "schema": "strata_v2_known_n20_independent_decoder_validation_v1",
        "passed": abs(relative_mse - expected_mse) <= 2e-12,
        "claim_boundary": "opened v1-derived KLT development block only; no blind-v2 artifact read",
        "independence": {
            "encoder_imported": False,
            "encoder_probabilities_read": False,
            "encoder_decisions_read": False,
            "external_reliability_tables_read": False,
        },
        "bindings": {
            "fixture": str(args.fixture),
            "fixture_sha256": decoder.sha256_file(args.fixture),
            "result": str(args.result),
            "result_sha256": fixture["encoder_result_sha256"],
            "container": str(args.container),
            "container_sha256": hashlib.sha256(literal).hexdigest(),
            "source_bf16": str(args.source_bf16),
            "source_bf16_sha256": hashlib.sha256(source_payload).hexdigest(),
        },
        "profile_q": profile_q,
        "rate_bpw": rate,
        "capacity_schedule": profile["capacities"],
        "logical_bits": logical_bits,
        "canonical_reencode_logical_length_match": True,
        "canonical_reencode_payload_bytes_match": True,
        "canonical_payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        "payload_terminal_padding_bits": len(payload) * 8 - logical_bits,
        "payload_terminal_padding_all_zero": not decoder.bit_range_has_one(
            payload, logical_bits, len(payload) * 8
        ),
        "serialized_scale_fp32": float(scale),
        "serialized_scale_matches_metadata": True,
        "serialized_scale_source_domain_relative_mse": serialized_scale_relative_mse,
        "selected_polar_bits": selected,
        "expected_selected_polar_bits": int(expected["selected_polar_bits"]),
        "selected_count_match": selected == int(expected["selected_polar_bits"]),
        "decoded_reconstruction_indices_i16_sha256": hashlib.sha256(
            previous.astype("<i2", copy=False).tobytes()
        ).hexdigest(),
        "causal_frequency_u16_sha256": frequency_hash.hexdigest(),
        "decoded_normalized_relative_mse": relative_mse,
        "recorded_encoder_relative_mse": expected_mse,
        "absolute_mse_difference": abs(relative_mse - expected_mse),
        "inverse_device": args.inverse_device,
        "runtime_seconds": time.perf_counter() - started,
    }
    if not result["passed"] or not result["selected_count_match"]:
        raise AssertionError(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
