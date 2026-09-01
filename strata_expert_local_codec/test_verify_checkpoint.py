from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import struct
import tempfile
import unittest
from unittest import mock
import zlib
from pathlib import Path

from strata_expert_local_codec import verify_checkpoint as verify


PUBLISHED_RELEASE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "qwen"
    / "strata_expert_affine_checkpoint"
)


def packed_labels() -> bytes:
    payload = bytearray(verify.LABEL_BYTES)
    for ordinal in range(verify.GROUPS):
        value = ordinal // 1728
        for component in range(3):
            if value & (1 << (2 - component)):
                bit = 3 * ordinal + component
                payload[bit // 8] |= 1 << (7 - bit % 8)
    return bytes(payload)


def valid_route() -> bytes:
    payload = bytearray()
    for expert_ordinal in range(verify.EXPERTS):
        for role, axis in ((0, 0), (1, 0), (2, 1)):
            payload.extend(
                struct.pack(
                    ">HHBBH", 5 + expert_ordinal, 18 + expert_ordinal, role, axis, 768
                )
            )
    return bytes(payload)


@contextmanager
def synthetic_identity() -> Iterator[None]:
    """Temporarily anchor the parser to this module's synthetic assets."""
    with (
        mock.patch.object(
            verify,
            "EXPECTED_ROUTE_SHA256",
            hashlib.sha256(valid_route()).hexdigest(),
        ),
        mock.patch.object(
            verify,
            "EXPECTED_LABELS_SHA256",
            hashlib.sha256(packed_labels()).hexdigest(),
        ),
    ):
        yield


def synthetic_container(*, first_logical_bits: int = 0, first_payload: int = 0) -> bytes:
    header = bytearray(verify.HEADER_BYTES)
    struct.pack_into(
        "<8sHHIIHHBBBBf",
        header,
        0,
        verify.MAGIC,
        1,
        verify.HEADER_BYTES,
        0x000001FF,
        verify.WEIGHTS,
        2048,
        13_824,
        verify.BLOCKS,
        verify.PRIVATE_BLOCKS,
        21,
        20,
        0.25,
    )
    struct.pack_into("<12f", header, 32, *((1.0, 0.0) * verify.EXPERTS))
    struct.pack_into("<6h", header, 80, *(0,) * verify.EXPERTS)
    route = valid_route()
    labels = packed_labels()
    header[92:124] = hashlib.sha256(route + labels).digest()
    struct.pack_into("<I", header, 124, zlib.crc32(header[:124]) & 0xFFFFFFFF)
    directory = bytearray(verify.DIRECTORY_BYTES)
    for ordinal in range(verify.BLOCKS):
        struct.pack_into("<BeI", directory, ordinal * verify.DIRECTORY_RECORD.size, 0, 1.0, 0)
    struct.pack_into("<BeI", directory, 0, 0, 1.0, first_logical_bits)
    reservoir_bytes = verify.PHYSICAL_BYTES - verify.PREFIX_BYTES
    reservoir = bytearray(reservoir_bytes)
    if first_logical_bits:
        reservoir[0] = first_payload
    return bytes(header) + route + labels + bytes(directory) + bytes(reservoir)


class CheckpointContainerTests(unittest.TestCase):
    def write(self, root: Path, payload: bytes) -> Path:
        path = root / "checkpoint.bin"
        path.write_bytes(payload)
        return path

    def test_fp64_sum_comparison_accepts_only_roundoff_scale(self) -> None:
        self.assertTrue(
            verify.same_fp64_sum(500.3955368542653, 500.39553685426534)
        )
        self.assertTrue(
            verify.same_fp64_sum(16192.894508855932, 16192.89450885593)
        )
        self.assertFalse(verify.same_fp64_sum(500.3955368642653, 500.39553685426534))

    def test_empty_stream_ledger_and_expert_map(self) -> None:
        with synthetic_identity(), tempfile.TemporaryDirectory() as text:
            parsed = verify.parse_container(
                self.write(Path(text), synthetic_container())
            )
        self.assertEqual(parsed["physical_bytes"], verify.PHYSICAL_BYTES)
        self.assertEqual(parsed["logical_payload_bits"], 0)
        self.assertEqual(
            parsed["zero_tail_bytes"], verify.PHYSICAL_BYTES - verify.PREFIX_BYTES
        )
        self.assertEqual(parsed["experts"][5]["required_blocks"], [10, 11, 14])
        self.assertLess(parsed["max_4k"], 2.0)

    def test_route_tamper_is_rejected(self) -> None:
        payload = bytearray(synthetic_container())
        payload[verify.HEADER_BYTES] ^= 1
        with synthetic_identity(), tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "asset binding"):
                verify.parse_container(self.write(Path(text), bytes(payload)))

    def test_crc_tamper_is_rejected(self) -> None:
        payload = bytearray(synthetic_container())
        payload[124] ^= 1
        with synthetic_identity(), tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "CRC"):
                verify.parse_container(self.write(Path(text), bytes(payload)))

    def test_klt_coefficient_code_mismatch_is_rejected(self) -> None:
        payload = bytearray(synthetic_container())
        payload[32] ^= 1
        struct.pack_into("<I", payload, 124, zlib.crc32(payload[:124]) & 0xFFFFFFFF)
        with synthetic_identity(), tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "KLT coefficient"):
                verify.parse_container(self.write(Path(text), bytes(payload)))

    def test_rebound_non_equipopulous_labels_are_rejected(self) -> None:
        payload = bytearray(synthetic_container())
        labels_begin = verify.HEADER_BYTES + verify.ROUTE_BYTES
        payload[labels_begin] ^= 0x80
        route = bytes(payload[verify.HEADER_BYTES:labels_begin])
        labels = bytes(payload[labels_begin:labels_begin + verify.LABEL_BYTES])
        payload[92:124] = hashlib.sha256(route + labels).digest()
        struct.pack_into("<I", payload, 124, zlib.crc32(payload[:124]) & 0xFFFFFFFF)
        with synthetic_identity(), tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "label histogram"):
                verify.parse_container(self.write(Path(text), bytes(payload)))

    def test_nonzero_terminal_fill_is_rejected(self) -> None:
        payload = bytearray(synthetic_container())
        payload[-1] = 1
        with synthetic_identity(), tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "terminal reservoir"):
                verify.parse_container(self.write(Path(text), bytes(payload)))

    def test_nonzero_low_payload_padding_is_rejected(self) -> None:
        payload = synthetic_container(first_logical_bits=1, first_payload=0x01)
        with synthetic_identity(), tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "payload padding"):
                verify.parse_container(self.write(Path(text), payload))

    def test_pinned_identity_anchors_match_published_release(self) -> None:
        route = (PUBLISHED_RELEASE / "assets" / "route.bin").read_bytes()
        labels = (PUBLISHED_RELEASE / "assets" / "labels_3bit.bin").read_bytes()
        plan = json.loads(
            (PUBLISHED_RELEASE / "plan.lock.json").read_text(encoding="utf-8")
        )
        self.assertEqual(hashlib.sha256(route).hexdigest(), verify.EXPECTED_ROUTE_SHA256)
        self.assertEqual(hashlib.sha256(labels).hexdigest(), verify.EXPECTED_LABELS_SHA256)
        self.assertEqual(
            hashlib.sha256(verify.canonical_json_bytes(plan["sources"])).hexdigest(),
            verify.EXPECTED_SOURCES_CANONICAL_SHA256,
        )

    def test_comprehensively_rebound_valid_route_is_rejected(self) -> None:
        payload = bytearray(
            (PUBLISHED_RELEASE / "strata_expert_affine_n20n21.bin").read_bytes()
        )
        for record in range(3):
            offset = verify.HEADER_BYTES + 8 * record
            layer = struct.unpack_from(">H", payload, offset)[0]
            struct.pack_into(">H", payload, offset, layer + 1)
        labels_begin = verify.HEADER_BYTES + verify.ROUTE_BYTES
        route = bytes(payload[verify.HEADER_BYTES:labels_begin])
        labels = bytes(payload[labels_begin:labels_begin + verify.LABEL_BYTES])
        payload[92:124] = hashlib.sha256(route + labels).digest()
        struct.pack_into("<I", payload, 124, zlib.crc32(payload[:124]) & 0xFFFFFFFF)
        with tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "pinned route SHA-256"):
                verify.parse_container(self.write(Path(text), bytes(payload)))

    def test_rebound_equipopulous_labels_are_rejected_by_pinned_hash(self) -> None:
        payload = bytearray(
            (PUBLISHED_RELEASE / "strata_expert_affine_n20n21.bin").read_bytes()
        )
        labels_begin = verify.HEADER_BYTES + verify.ROUTE_BYTES

        def label(ordinal: int) -> int:
            value = 0
            for component in range(3):
                bit = 3 * ordinal + component
                byte = labels_begin + bit // 8
                value = (value << 1) | ((payload[byte] >> (7 - bit % 8)) & 1)
            return value

        def set_label(ordinal: int, value: int) -> None:
            for component in range(3):
                bit = 3 * ordinal + component
                byte = labels_begin + bit // 8
                mask = 1 << (7 - bit % 8)
                if value & (1 << (2 - component)):
                    payload[byte] |= mask
                else:
                    payload[byte] &= ~mask

        first = 0
        second = next(
            ordinal for ordinal in range(1, verify.GROUPS) if label(ordinal) != label(first)
        )
        first_value, second_value = label(first), label(second)
        set_label(first, second_value)
        set_label(second, first_value)

        labels_end = labels_begin + verify.LABEL_BYTES
        route = bytes(payload[verify.HEADER_BYTES:labels_begin])
        labels = bytes(payload[labels_begin:labels_end])
        self.assertEqual(verify.label_histogram(labels), [1728] * 8)
        payload[92:124] = hashlib.sha256(route + labels).digest()
        struct.pack_into("<I", payload, 124, zlib.crc32(payload[:124]) & 0xFFFFFFFF)
        with tempfile.TemporaryDirectory() as text:
            with self.assertRaisesRegex(AssertionError, "pinned label SHA-256"):
                verify.parse_container(self.write(Path(text), bytes(payload)))

    def test_manifest_requires_every_encoder_and_asset_role(self) -> None:
        missing = "encoder_block_14"
        with tempfile.TemporaryDirectory() as text:
            root = Path(text)
            manifest_path = root / "checkpoint_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            rows = []
            for ordinal, role in enumerate(sorted(verify.REQUIRED_ROLES - {missing})):
                path = root / f"evidence_{ordinal:02d}.bin"
                payload = role.encode("ascii")
                path.write_bytes(payload)
                rows.append(
                    {
                        "path": path.name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "role": role,
                    }
                )
            value = {"schema": verify.MANIFEST_SCHEMA, "files": rows}
            with self.assertRaisesRegex(AssertionError, "lacks a required evidence role"):
                verify.verify_manifest(root, manifest_path, value)


if __name__ == "__main__":
    unittest.main()
