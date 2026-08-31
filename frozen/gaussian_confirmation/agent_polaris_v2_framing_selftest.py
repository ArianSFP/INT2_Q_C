#!/usr/bin/env python3
"""Synthetic, seed-free framing tests for the POLARIS-SC-v2 reservoir."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path

import numpy as np

import agent_polaris_reservoir_pack_v2 as packer
import agent_polaris_reservoir_unpack_v2 as unpacker


def logical_bits(payload: bytes, count: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[
        :count
    ]


def legacy_bytes(count: int, scale: float, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=count, dtype=np.uint8)
    payload = np.packbits(bits, bitorder="big").tobytes()
    return struct.pack("<If", count, scale) + payload


def raw_reservoir(
    length: int,
    scale_bytes: bytes,
    payload: bytes,
    *,
    fixed_capacity: bool = False,
) -> bytes:
    if fixed_capacity:
        capacity_bytes = packer.PAYLOAD_BUDGET_BITS_PER_BLOCK // 8
        if len(payload) > capacity_bytes:
            raise ValueError("test payload exceeds one fixed reservoir")
        payload = payload + bytes(capacity_bytes - len(payload))
    directory = struct.pack(">I", length) + scale_bytes
    directory_hash = hashlib.sha256(directory).digest()
    payload_hash = hashlib.sha256(payload).digest()
    header = packer.HEADER.pack(
        packer.MAGIC,
        packer.VERSION,
        packer.FLAGS,
        1,
        48,
        length,
        directory_hash,
        payload_hash,
    )
    return header + directory + payload


def main() -> None:
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33]
    result: dict[str, object] = {"logical_lengths": lengths}
    with tempfile.TemporaryDirectory(prefix="polaris-v2-framing-") as raw:
        root = Path(raw)
        inputs: list[Path] = []
        originals: list[bytes] = []
        for index, count in enumerate(lengths):
            value = legacy_bytes(count, 1.0 + index / 32.0, 1000 + index)
            path = root / f"input_{index:02d}.legacy.bin"
            path.write_bytes(value)
            inputs.append(path)
            originals.append(value)

        reservoir_path = root / "test.plrsv2"
        pack_audit = packer.pack(inputs, reservoir_path)
        ordinary_size_immediately_after_pack = reservoir_path.stat().st_size
        ordinary_hash_immediately_after_pack = hashlib.sha256(
            reservoir_path.read_bytes()
        ).hexdigest()
        validated = unpacker.validate_reservoir(reservoir_path)
        extracted_dir = root / "extracted"
        blocks = unpacker.extract_variable_records(validated, extracted_dir)
        payloads_match = True
        scales_match = True
        for index, block in enumerate(blocks):
            original = originals[index]
            original_count, original_scale = struct.unpack("<If", original[:8])
            recovered = (extracted_dir / str(block["relative_path"])).read_bytes()
            recovered_count = struct.unpack("<I", recovered[:4])[0]
            recovered_scale = float(np.frombuffer(recovered[4:6], dtype="<f2")[0])
            payloads_match &= original_count == recovered_count
            payloads_match &= bool(
                np.array_equal(
                    logical_bits(original[8:], original_count),
                    logical_bits(recovered[6:], recovered_count),
                )
            )
            expected_scale = float(np.float16(original_scale))
            scales_match &= recovered_scale == expected_scale

        result.update(
            {
                "ordinary_roundtrip_payloads_match": bool(payloads_match),
                "ordinary_roundtrip_fp16_scales_match": bool(scales_match),
                "ordinary_reservoir_bytes": reservoir_path.stat().st_size,
                "ordinary_reservoir_bytes_immediately_after_pack": (
                    ordinary_size_immediately_after_pack
                ),
                "ordinary_pack_audit_physical_bytes": (
                    int(pack_audit["rate"]["physical_file_bits"]) // 8
                ),
                "ordinary_hash_unchanged_after_extract": (
                    ordinary_hash_immediately_after_pack
                    == hashlib.sha256(reservoir_path.read_bytes()).hexdigest()
                ),
                "ordinary_expected_bytes": (
                    packer.HEADER_BYTES
                    + 6 * len(lengths)
                    + (
                        packer.PAYLOAD_BUDGET_BITS_PER_BLOCK
                        * len(lengths)
                        // 8
                    )
                ),
                "ordinary_pack_passed": bool(pack_audit["passed"]),
                "ordinary_physical_size_matches": (
                    reservoir_path.stat().st_size
                    == packer.HEADER_BYTES
                    + 6 * len(lengths)
                    + (
                        packer.PAYLOAD_BUDGET_BITS_PER_BLOCK
                        * len(lengths)
                        // 8
                    )
                ),
                "ordinary_all_bit_residues_exercised": (
                    {value % 8 for value in lengths} == set(range(8))
                ),
            }
        )

        bad_tail = root / "bad_tail.legacy.bin"
        bad_tail.write_bytes(struct.pack("<If", 1, 1.0) + b"\x81")
        try:
            packer.pack([bad_tail], root / "bad_tail.plrsv2")
        except ValueError:
            result["packer_rejects_nonzero_legacy_tail"] = True
        else:
            result["packer_rejects_nonzero_legacy_tail"] = False

        overflow_length = packer.PAYLOAD_BUDGET_BITS_PER_BLOCK + 1
        overflow = root / "overflow.legacy.bin"
        overflow.write_bytes(
            struct.pack("<If", overflow_length, 1.0)
            + bytes((overflow_length + 7) // 8)
        )
        try:
            packer.pack([overflow], root / "overflow.plrsv2")
        except OverflowError:
            result["packer_rejects_global_overflow"] = True
        else:
            result["packer_rejects_global_overflow"] = False

        # Bypass the packer with a fully self-consistent over-budget container.
        over_payload = bytes((overflow_length + 7) // 8)
        raw_over_path = root / "raw_overbudget.plrsv2"
        raw_over_path.write_bytes(
            raw_reservoir(
                overflow_length,
                np.asarray([1.0], dtype="<f2").tobytes(),
                over_payload,
            )
        )
        try:
            unpacker.validate_reservoir(raw_over_path)
        except Exception as exc:  # audit records behavior, not exception type
            result["unpacker_accepts_self_consistent_overbudget"] = False
            result["overbudget_rejection"] = type(exc).__name__
        else:
            result["unpacker_accepts_self_consistent_overbudget"] = True

        nan_path = root / "nan_scale.plrsv2"
        nan_scale = np.asarray([np.nan], dtype="<f2").tobytes()
        nan_path.write_bytes(
            raw_reservoir(8, nan_scale, b"\0", fixed_capacity=True)
        )
        try:
            unpacker.validate_reservoir(nan_path)
        except Exception as exc:
            result["unpacker_accepts_nan_scale"] = False
            result["nan_scale_rejection"] = type(exc).__name__
        else:
            result["unpacker_accepts_nan_scale"] = True

    required_true = (
        "ordinary_roundtrip_payloads_match",
        "ordinary_roundtrip_fp16_scales_match",
        "ordinary_pack_passed",
        "ordinary_physical_size_matches",
        "ordinary_hash_unchanged_after_extract",
        "ordinary_all_bit_residues_exercised",
        "packer_rejects_nonzero_legacy_tail",
        "packer_rejects_global_overflow",
    )
    result["core_roundtrip_passed"] = (
        all(bool(result[key]) for key in required_true)
        and result.get("unpacker_accepts_self_consistent_overbudget") is False
        and result.get("unpacker_accepts_nan_scale") is False
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
