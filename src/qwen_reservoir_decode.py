#!/usr/bin/env python3
"""Fresh-process audit of one real-BF16 POLARIS v2 reservoir record.

The decoder imports only the frozen independent decoder core, never the polar
encoder.  It verifies the source BF16 block, the RMS normalization contract,
the exact FP16 scale bytes chosen by the reservoir packer, causal arithmetic
decoding, and the MSE change caused by FP32-to-FP16 scale serialization.

This is intentionally separate from the frozen Gaussian confirmation tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

import agent_polaris_independent_decoder_v1 as core


RHT_FORMULA = "H_N diag(splitmix64(seed,index) parity) / sqrt(N)"
RHT_INVERSE = "diag(signs) H_N / sqrt(N)"
RHT_DEFAULT_SEED = 20_260_831
RHT_MODE = "hadamard_rademacher_splitmix64"
RHT_NORMALIZATION = "orthonormal"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_record(path: Path) -> tuple[int, bytes, float, bytes, bytes]:
    container = path.read_bytes()
    if len(container) < 6:
        raise ValueError("variable-u32-fp16 record is truncated")
    logical_bits = struct.unpack("<I", container[:4])[0]
    scale_bytes = container[4:6]
    scale = float(struct.unpack("<e", scale_bytes)[0])
    payload = container[6:]
    expected = (logical_bits + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"logical payload needs {expected} bytes; found {len(payload)}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid serialized FP16 scale {scale}")
    if logical_bits % 8:
        unused = 8 - logical_bits % 8
        if payload[-1] & ((1 << unused) - 1):
            raise ValueError("nonzero record-local tail padding")
    return logical_bits, scale_bytes, scale, payload, container


def bf16_block(path: Path, index: int, n: int) -> tuple[np.ndarray, bytes]:
    raw = np.memmap(path, dtype="<u2", mode="r")
    begin = index * n
    end = begin + n
    if end > raw.size:
        raise ValueError(f"block {index} ends at {end}, beyond {raw.size} BF16 values")
    raw_block = np.asarray(raw[begin:end]).copy()
    values = (raw_block.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return values.astype(np.float64), raw_block.tobytes()


def splitmix_signs(n: int, seed: int) -> np.ndarray:
    """Portable stateless diagonal used by the existing polar FHT codecs."""
    with np.errstate(over="ignore"):
        values = np.arange(n, dtype=np.uint64) + np.uint64(seed)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        values ^= values >> np.uint64(31)
    return np.where((values & np.uint64(1)) == 0, 1.0, -1.0)


def orthonormal_fht(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    width = 1
    while width < result.size:
        view = result.reshape(-1, 2 * width)
        left = view[:, :width].copy()
        right = view[:, width:].copy()
        view[:, :width] = left + right
        view[:, width:] = left - right
        width *= 2
    result /= math.sqrt(float(result.size))
    return result


def signed_rht(values: np.ndarray, seed: int) -> np.ndarray:
    """Forward H*D and inverse D*H are distinct despite H being involutory."""
    return orthonormal_fht(np.asarray(values, dtype=np.float64) * splitmix_signs(values.size, seed))


def inverse_signed_rht(values: np.ndarray, seed: int) -> np.ndarray:
    return orthonormal_fht(values) * splitmix_signs(values.size, seed)


def rht_metadata(
    metadata: dict[str, object],
    parameters: dict[str, object],
    trial_row: dict[str, object],
    source_row: dict[str, object],
) -> tuple[bool, int, dict[str, object] | None]:
    """Recognize one explicit, zero-side-bit deterministic RHT contract."""
    rht = metadata.get("rht")
    if rht is None:
        rht = parameters.get("rht")
    if rht is None:
        rht = trial_row.get("rht")
    if rht is None:
        rht = source_row.get("rht")
    if rht is not None:
        if not isinstance(rht, dict):
            raise TypeError("rht metadata must be an object")
        mode = str(rht.get("mode"))
        normalization = str(rht.get("normalization"))
        seed = int(rht["seed_u64"])
        if mode != RHT_MODE or normalization != RHT_NORMALIZATION:
            raise ValueError((mode, normalization))
        if not 0 <= seed < (1 << 64):
            raise ValueError(f"rht.seed_u64 is outside u64: {seed}")
        normalized_rht = {
            "mode": mode,
            "seed_u64": seed,
            "normalization": normalization,
            "forward": "y = H(signs * x) / sqrt(N)",
            "inverse": "xhat = signs * H(yhat) / sqrt(N)",
            "sign_rule": "+1 iff splitmix64(seed_u64 + i) low bit is zero; else -1",
            "side_bits": 0,
        }
        return True, seed, normalized_rht

    # Backward-compatible recognition of the pre-existing development schema.
    # Newly frozen Qwen runs must use the explicit `rht` object above.
    candidate = metadata.get("preconditioner")
    if candidate is None:
        candidate = parameters.get("preconditioner")
    if candidate is None:
        candidate = trial_row.get("preconditioner")
    if candidate is None:
        candidate = source_row.get("preconditioner")
    legacy = source_row.get("zero_bit_preconditioner")
    if candidate is None and legacy is None:
        return False, RHT_DEFAULT_SEED, None
    if candidate is None:
        expected = f"signed_fht_seed_{RHT_DEFAULT_SEED}"
        if legacy != expected:
            raise ValueError(f"unsupported zero-bit preconditioner {legacy!r}")
        candidate = {
            "formula": RHT_FORMULA,
            "seed": RHT_DEFAULT_SEED,
            "block_length": int(parameters["block_length"]),
            "inverse": RHT_INVERSE,
            "side_bits": 0,
        }
    if not isinstance(candidate, dict):
        raise TypeError("preconditioner metadata must be an object")
    formula = str(candidate.get("formula"))
    inverse = str(candidate.get("inverse"))
    seed = int(candidate.get("seed", RHT_DEFAULT_SEED))
    block_length = int(candidate.get("block_length", parameters["block_length"]))
    side_bits = int(candidate.get("side_bits", 0))
    if formula != RHT_FORMULA or inverse != RHT_INVERSE:
        raise ValueError((formula, inverse))
    if block_length != int(parameters["block_length"]) or side_bits != 0:
        raise ValueError((block_length, side_bits))
    normalized = {
        "mode": RHT_MODE,
        "seed_u64": seed,
        "normalization": RHT_NORMALIZATION,
        "forward": "y = H(signs * x) / sqrt(N)",
        "inverse": "xhat = signs * H(yhat) / sqrt(N)",
        "sign_rule": "+1 iff splitmix64(seed_u64 + i) low bit is zero; else -1",
        "side_bits": side_bits,
        "legacy_metadata_adapter": True,
    }
    return True, seed, normalized


def decode(
    record_path: Path,
    metadata_path: Path,
    map_path: Path,
    source_path: Path,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = metadata["parameters"]
    trials = metadata["trials"]
    if len(trials) != 1:
        raise ValueError("audit metadata must contain exactly one trial")
    trial_row = trials[0]
    source_row = trial_row["source"]
    if source_row["kind"] != "frozen_bf16_weight_block":
        raise ValueError("metadata does not describe a real BF16 weight block")
    if int(trial_row.get("tail_escape_count", 0)) != 0:
        raise ValueError("v2 base reservoir audit forbids unrepresented tail escapes")

    n = int(parameters["block_length"])
    trial = int(trial_row["trial"])
    seed = int(parameters["seed"])
    sigma_source = float(parameters["sigma_source"])
    distortion = float(parameters["test_channel_distortion"])
    eta = float(parameters["eta"])
    alphabet_size = int(parameters["alphabet_size"])
    levels = int(math.log2(alphabet_size))
    if n != 1 << 18 or levels != 6:
        raise ValueError((n, levels))

    logical_bits, scale_bytes, scale_fp16, payload, container = read_record(record_path)
    if logical_bits != int(trial_row["arithmetic_logical_bits"]):
        raise AssertionError("reservoir logical length disagrees with encoder metadata")
    if sha256_bytes(payload) != trial_row["arithmetic_payload_sha256"]:
        raise AssertionError("reservoir payload differs from the encoder arithmetic payload")

    block_index = int(source_row["block_index"])
    source, source_bf16_bytes = bf16_block(source_path, block_index, n)
    use_rht, rht_seed, preconditioner = rht_metadata(
        metadata,
        parameters,
        trial_row,
        source_row,
    )
    codec_source = signed_rht(source, rht_seed) if use_rht else source
    source_hash = sha256_bytes(source_bf16_bytes)
    recorded_block_hash = source_row.get("block_bf16_sha256")
    legacy_source_hash = source_row.get("source_bf16_sha256")
    if (
        recorded_block_hash is not None
        and legacy_source_hash is not None
        and recorded_block_hash != legacy_source_hash
    ):
        raise AssertionError(
            "block_bf16_sha256 and legacy source_bf16_sha256 disagree"
        )
    recorded_source_hash = (
        recorded_block_hash
        if recorded_block_hash is not None
        else legacy_source_hash
    )
    source_hash_metadata_field = (
        "block_bf16_sha256"
        if recorded_block_hash is not None
        else (
            "source_bf16_sha256"
            if legacy_source_hash is not None
            else None
        )
    )
    if recorded_source_hash is not None and source_hash != recorded_source_hash:
        raise AssertionError("source BF16 block hash differs from encoder metadata")
    rms_fp64 = float(np.sqrt(np.mean(np.square(codec_source), dtype=np.float64)))
    recorded_rms = float(source_row["block_rms_fp64"])
    rms_tolerance = max(1e-15, 2e-12 * abs(recorded_rms))
    if abs(rms_fp64 - recorded_rms) > rms_tolerance:
        raise AssertionError((rms_fp64, recorded_rms))
    scale_fp32 = float(np.float32(recorded_rms / sigma_source))
    recorded_scale_fp32 = float(source_row["decoder_scale_fp32"])
    if scale_fp32 != recorded_scale_fp32:
        raise AssertionError((scale_fp32, recorded_scale_fp32))
    expected_scale_bytes = np.asarray([scale_fp32], dtype="<f2").tobytes()
    if scale_bytes != expected_scale_bytes:
        raise AssertionError(
            f"reservoir FP16 scale bytes {scale_bytes.hex()} != expected "
            f"{expected_scale_bytes.hex()}"
        )

    flags = core.frozen_map_flags(map_path, n, levels)
    reverse = core.bit_reverse_indices(n)
    layers = core.sc_layers(n)
    sigma_recon = math.sqrt(sigma_source * sigma_source - distortion)
    alphabet = eta * np.arange(
        -alphabet_size // 2 + 1,
        alphabet_size // 2 + 1,
        dtype=np.float64,
    )
    weights = np.exp(-0.5 * np.square(alphabet / sigma_recon))
    arithmetic = core.ArithmeticBinaryDecoder(payload, logical_bits)
    previous = np.zeros(n, dtype=np.int16)
    frequency_hash = hashlib.sha256()
    selected = 0
    for level_index, flag in enumerate(flags):
        level = level_index + 1
        frozen_rng = np.random.default_rng(
            seed + 104_729 * trial + 1_000_003 * level
        )
        frozen_external = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior_lr = core.leaf_prior_ratios(weights, previous, level)
        decoded_x, frequencies = core.decode_sc_level(
            prior_lr,
            flag,
            frozen_external,
            reverse,
            layers,
            arithmetic,
        )
        previous += (1 << level_index) * decoded_x.astype(np.int16)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected += int(frequencies.size)

    normalized_source = codec_source * (sigma_source / recorded_rms)
    normalized_reconstruction = alphabet[previous]
    normalized_error = np.square(normalized_source - normalized_reconstruction)
    normalized_relative_mse = float(
        normalized_error.sum(dtype=np.float64)
        / np.square(normalized_source).sum(dtype=np.float64)
    )
    # The immutable frozen encoder calls this simply ``relative_mse`` because
    # it measures in the normalized domain.  Newer development variants expose
    # the longer explicit key; accepting both does not alter codec semantics.
    encoder_normalized_mse = float(
        trial_row.get(
            "normalized_relative_mse_before_fp32_scale",
            trial_row["relative_mse"],
        )
    )
    normalized_match = abs(normalized_relative_mse - encoder_normalized_mse) <= 1e-12

    codec_reconstruction_fp32 = normalized_reconstruction * scale_fp32
    codec_reconstruction_fp16 = normalized_reconstruction * scale_fp16
    reconstruction_fp32 = (
        inverse_signed_rht(codec_reconstruction_fp32, rht_seed)
        if use_rht
        else codec_reconstruction_fp32
    )
    reconstruction_fp16 = (
        inverse_signed_rht(codec_reconstruction_fp16, rht_seed)
        if use_rht
        else codec_reconstruction_fp16
    )
    source_energy = float(np.square(source).sum(dtype=np.float64))
    fp32_squared = np.square(source - reconstruction_fp32)
    fp16_squared = np.square(source - reconstruction_fp16)
    fp32_sse = float(fp32_squared.sum(dtype=np.float64))
    fp16_sse = float(fp16_squared.sum(dtype=np.float64))
    fp32_relative_mse = fp32_sse / source_energy
    fp16_relative_mse = fp16_sse / source_energy
    encoder_literal_value = trial_row.get("literal_decoded_relative_mse")
    encoder_literal_mse = (
        float(encoder_literal_value) if encoder_literal_value is not None else None
    )
    # With no stored literal-domain metric, exact normalized-index agreement is
    # the frozen encoder audit.  FP32 and FP16 literal metrics are then fresh
    # deployment measurements rather than falsely claimed metadata matches.
    fp32_match = (
        abs(fp32_relative_mse - encoder_literal_mse) <= 1e-12
        if encoder_literal_mse is not None
        else True
    )

    result: dict[str, object] = {
        "status": "fresh-process real-BF16 variable-u32-fp16 decode",
        "source": {
            "path": str(source_path.resolve()),
            "block_index": block_index,
            "bf16_sha256": source_hash,
            "rms_fp64": rms_fp64,
        },
        "normalization": {
            "formula": (
                "normalized = RHT(BF16_weight) * sigma_source / RMS_FP64"
                if use_rht
                else "normalized = BF16_weight * sigma_source / RMS_FP64"
            ),
            "sigma_source": sigma_source,
            "normalized_rms_fp64": float(
                np.sqrt(np.mean(np.square(normalized_source), dtype=np.float64))
            ),
            "scale_fp32_legacy": scale_fp32,
            "scale_fp16_serialized": scale_fp16,
            "scale_fp16_raw_le_hex": scale_bytes.hex(),
            "relative_scale_error": scale_fp16 / scale_fp32 - 1.0,
        },
        "preconditioner": preconditioner,
        "aggregation": {
            "source_values": int(source.size),
            "source_energy_sum_fp64": source_energy,
            "fp32_sse_sum_fp64": fp32_sse,
            "fp16_sse_sum_fp64": fp16_sse,
            "final_reconstruction_fp64_sha256": sha256_bytes(
                reconstruction_fp16.astype("<f8", copy=False).tobytes()
            ),
            "final_reconstruction_definition": (
                "inverse-RHT(normalized_reconstruction * serialized_FP16_scale)"
                if use_rht
                else "normalized_reconstruction * serialized_FP16_scale"
            ),
        },
        "codec": {
            "logical_arithmetic_bits": logical_bits,
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "record_bytes": len(container),
            "record_sha256": sha256_bytes(container),
            "selected_symbols": selected,
            "causal_frequency_u16_sha256": frequency_hash.hexdigest(),
            "reconstruction_indices_sha256": sha256_bytes(
                previous.astype("<i2", copy=False).tobytes()
            ),
        },
        "distortion": {
            "encoder_normalized_relative_mse": encoder_normalized_mse,
            "fresh_normalized_relative_mse": normalized_relative_mse,
            "normalized_mse_match_at_1e_12": normalized_match,
            "encoder_fp32_scale_relative_mse": encoder_literal_mse,
            "fresh_fp32_scale_relative_mse": fp32_relative_mse,
            "fp32_scale_mse_match_at_1e_12": (
                fp32_match if encoder_literal_mse is not None else None
            ),
            "fresh_fp16_scale_relative_mse": fp16_relative_mse,
            "fp16_minus_fp32_relative_mse": fp16_relative_mse - fp32_relative_mse,
            "fp16_scale_relative_mse_ratio": fp16_relative_mse / fp32_relative_mse,
            "fresh_fp16_scale_absolute_mse": float(
                fp16_squared.mean(dtype=np.float64)
            ),
        },
        "audits": {
            "no_tail_escapes": True,
            "source_hash_match": (
                True if recorded_source_hash is not None else "not recorded by frozen encoder"
            ),
            "source_hash_metadata_field": source_hash_metadata_field,
            "rms_match": True,
            "fp32_scale_match": True,
            "fp16_scale_bytes_match": True,
            "arithmetic_payload_match": True,
            "normalized_decoder_match": normalized_match,
            "fp32_decoder_match": (
                fp32_match if encoder_literal_mse is not None else "fresh metric"
            ),
        },
        "passed": bool(normalized_match and fp32_match),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--source-bf16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decode(args.record, args.metadata, args.map, args.source_bf16)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="", flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
