#!/usr/bin/env python3
"""Synthetic format and corruption tests for the standalone v2 unpacker."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

import agent_polaris_reservoir_unpack_v2 as unpack


def make_container(
    path: Path,
    lengths: list[int],
    patterns: list[str],
    scales: list[float],
    corrupt_suffix: bool = False,
) -> None:
    directory = b"".join(
        struct.pack(">I", length) + struct.pack("<e", scale)
        for length, scale in zip(lengths, scales, strict=True)
    )
    logical = "".join(patterns)
    payload = bytearray(
        unpack.PAYLOAD_CAPACITY_BYTES_PER_BLOCK * len(lengths)
    )
    for bit_index, bit in enumerate(logical):
        if bit == "1":
            payload[bit_index // 8] |= 1 << (7 - bit_index % 8)
    if corrupt_suffix:
        suffix_index = len(logical)
        payload[suffix_index // 8] |= 1 << (7 - suffix_index % 8)
    header = unpack.HEADER_STRUCT.pack(
        unpack.MAGIC,
        unpack.VERSION,
        unpack.FLAGS,
        len(lengths),
        unpack.DIRECTORY_ENTRY_BITS * len(lengths),
        len(logical),
        hashlib.sha256(directory).digest(),
        hashlib.sha256(payload).digest(),
    )
    path.write_bytes(header + directory + payload)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="polaris-v2-unpack-selftest-"))
    try:
        lengths = [1, 7, 8, 9, 17]
        patterns = [("10" * ((length + 1) // 2))[:length] for length in lengths]
        scales = [0.5, 1.0, 1.5, 2.0, 3.25]
        source = root / "valid.plrsv2"
        make_container(source, lengths, patterns, scales)
        reservoir = unpack.validate_reservoir(source)
        records = unpack.extract_variable_records(reservoir, root / "records")
        assert len(records) == len(lengths)

        for index, (length, pattern, scale) in enumerate(
            zip(lengths, patterns, scales, strict=True)
        ):
            record_path = (
                root
                / "records"
                / f"block_{index:06d}.variable-u32-fp16.bin"
            )
            raw = record_path.read_bytes()
            assert struct.unpack("<I", raw[:4])[0] == length
            assert raw[4:6] == struct.pack("<e", scale)
            bit_text = "".join(f"{byte:08b}" for byte in raw[6:])
            assert bit_text[:length] == pattern
            assert set(bit_text[length:]) <= {"0"}

        corrupt = root / "nonzero-suffix.plrsv2"
        make_container(corrupt, lengths, patterns, scales, corrupt_suffix=True)
        try:
            unpack.validate_reservoir(corrupt)
        except unpack.ReservoirFormatError as exc:
            if "after payload_logical_bits" not in str(exc) and "unused" not in str(exc):
                raise
        else:
            raise AssertionError("nonzero fixed-reservoir suffix was accepted")

        bad_hash = bytearray(source.read_bytes())
        payload_offset = unpack.HEADER_BYTES + len(lengths) * unpack.DIRECTORY_ENTRY_BYTES
        bad_hash[payload_offset] ^= 0x80
        bad_hash_path = root / "bad-hash.plrsv2"
        bad_hash_path.write_bytes(bad_hash)
        try:
            unpack.validate_reservoir(bad_hash_path)
        except unpack.ReservoirFormatError:
            pass
        else:
            raise AssertionError("payload hash corruption was accepted")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "blocks": len(records),
                    "logical_bits": sum(lengths),
                    "record_layout": "variable-u32-fp16",
                    "reservoir_sha256": reservoir.reservoir_sha256,
                },
                sort_keys=True,
            )
        )
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
