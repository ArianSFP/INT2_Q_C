from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import uwfa_bridge as bridge
from synthetic_fixture import (
    build_synthetic_container,
    make_synthetic_model,
    verify_synthetic_container,
)


class ArithmeticTests(unittest.TestCase):
    def test_binary_arithmetic_exact_roundtrip(self) -> None:
        bits = tuple(((index * 73 + index // 7) ^ (index >> 2)) & 1 for index in range(10000))
        frequencies = tuple(1 + ((index * 7919 + 17) % 65535) for index in range(len(bits)))
        encoder = bridge.BinaryArithmeticEncoder()
        for bit, frequency in zip(bits, frequencies):
            encoder.encode(bit, frequency)
        payload, logical_bits = encoder.finish()
        decoder = bridge.BinaryArithmeticDecoder(payload, logical_bits)
        decoded = tuple(decoder.decode(frequency) for frequency in frequencies)
        self.assertEqual(decoded, bits)

        canonical = bridge.BinaryArithmeticEncoder()
        for bit, frequency in zip(decoded, frequencies):
            canonical.encode(bit, frequency)
        self.assertEqual(canonical.finish(), (payload, logical_bits))

    def test_model_rows_are_explicit_complete_and_canonical(self) -> None:
        model = make_synthetic_model()
        encoded = bridge.serialize_model(model)
        self.assertEqual(bridge.deserialize_model(encoded), model)

        # Swap two complete records and repair the CRC.  The parser must reject
        # the now nonlexicographic keys, rather than silently indexing by row.
        hostile = bytearray(encoded)
        first = bridge.MODEL_HEADER_SIZE
        second = first + bridge.MODEL_ROW_SIZE
        row0 = bytes(hostile[first:second])
        hostile[first:second] = hostile[second : second + bridge.MODEL_ROW_SIZE]
        hostile[second : second + bridge.MODEL_ROW_SIZE] = row0
        struct.pack_into("<I", hostile, 40, 0)
        struct.pack_into("<I", hostile, 40, bridge.crc32(bytes(hostile)))
        with self.assertRaises(bridge.FormatError):
            bridge.deserialize_model(bytes(hostile))

    def test_zero_byte_extension_is_not_an_alternate_arithmetic_code(self) -> None:
        encoder = bridge.BinaryArithmeticEncoder()
        encoder.encode(1, 32768)
        payload, logical_bits = encoder.finish()
        with self.assertRaisesRegex(bridge.FormatError, "zero-byte extension"):
            bridge.BinaryArithmeticDecoder(payload + b"\x00", logical_bits)


class ContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 983,040 causal decisions is still a mini fixture relative to a real
        # expert, while making page accounting nondegenerate.
        cls.raw = build_synthetic_container(log2n=16)
        cls.parsed = bridge.parse_container(cls.raw)

    @staticmethod
    def _fix_header_crc(image: bytearray) -> None:
        struct.pack_into("<I", image, bridge.HEADER_CRC_OFFSET, 0)
        struct.pack_into(
            "<I",
            image,
            bridge.HEADER_CRC_OFFSET,
            bridge.crc32(bytes(image[: bridge.GLOBAL_HEADER_SIZE])),
        )

    @staticmethod
    def _reseal_directory(image: bytearray, parsed: bridge.ParsedContainer) -> None:
        header = parsed.header
        directory = bytes(
            image[
                header.directory_offset : header.directory_offset + header.directory_actual
            ]
        )
        image[528:560] = bridge.sha256(directory)
        # The root excludes page/frame padding, but binds the changed literal
        # directory and its hash in the normalized header.
        image[bridge.ROOT_HASH_OFFSET : bridge.ROOT_HASH_END] = bytes(32)
        struct.pack_into("<I", image, bridge.HEADER_CRC_OFFSET, 0)
        root = bridge._root_digest(
            bytes(image[: bridge.GLOBAL_HEADER_SIZE]),
            parsed.metadata,
            parsed.model_bytes,
            directory,
            parsed.frames,
        )
        image[bridge.ROOT_HASH_OFFSET : bridge.ROOT_HASH_END] = root
        ContainerTests._fix_header_crc(image)

    def test_literal_parse_decode_regenerate_and_reencode(self) -> None:
        verification = verify_synthetic_container(self.raw)
        self.assertEqual(len(verification.context_hashes), bridge.BLOCK_COUNT)
        self.assertEqual(
            verification.decoded_hash,
            verification.parsed.header.bindings.decoded_reconstruction_hash,
        )
        self.assertTrue(verification.score.rate_in_range)
        self.assertTrue(verification.score.f_pass)

    def test_actual_rate_f_and_owner_page_ledger(self) -> None:
        verification = verify_synthetic_container(self.raw)
        expected_rate = Fraction(8 * len(self.raw), self.parsed.header.source_weights)
        self.assertEqual(verification.score.rate, expected_rate)
        self.assertAlmostEqual(
            verification.score.f_actual,
            0.03 * 2.0 ** (2.0 * float(expected_rate)),
            places=15,
        )
        self.assertEqual(verification.ledger.storage_conservation, Fraction(len(self.raw), 1))
        self.assertEqual(
            sum((item.storage_share for item in verification.ledger.experts), Fraction()),
            Fraction(len(self.raw), 1),
        )
        # This fixture is deliberately large enough to exercise, and pass,
        # the strict routed-read gate with physical page unions.
        self.assertLess(verification.ledger.maximum_amplification, 2.0)
        for item in verification.ledger.experts:
            self.assertEqual(item.touched_bytes, len(set(item.touched_pages)) * bridge.PAGE_SIZE)

    def test_header_crc_tamper_rejected(self) -> None:
        hostile = bytearray(self.raw)
        hostile[208] ^= 1
        with self.assertRaises(bridge.FormatError):
            bridge.parse_container(bytes(hostile))

    def test_reserved_header_byte_rejected_even_with_valid_crc(self) -> None:
        hostile = bytearray(self.raw)
        hostile[700] = 1
        self._fix_header_crc(hostile)
        with self.assertRaises(bridge.FormatError):
            bridge.parse_container(bytes(hostile))

    def test_metadata_page_padding_tamper_rejected(self) -> None:
        hostile = bytearray(self.raw)
        offset = self.parsed.header.metadata_offset + self.parsed.header.metadata_actual
        hostile[offset] = 1
        with self.assertRaises(bridge.FormatError):
            bridge.parse_container(bytes(hostile))

    def test_extra_zero_page_is_noncanonical(self) -> None:
        hostile = bytearray(self.raw + bytes(bridge.PAGE_SIZE))
        new_length = len(hostile)
        struct.pack_into("<Q", hostile, 72, new_length)
        struct.pack_into(
            "<Q",
            hostile,
            176,
            self.parsed.header.final_padding_length + bridge.PAGE_SIZE,
        )
        self._fix_header_crc(hostile)
        with self.assertRaises(bridge.FormatError):
            bridge.parse_container(bytes(hostile))

    def test_directory_overlap_rejected_after_full_reseal(self) -> None:
        hostile = bytearray(self.raw)
        first_record = self.parsed.header.directory_offset
        original_offset = struct.unpack_from("<Q", hostile, first_record + 24)[0]
        struct.pack_into("<Q", hostile, first_record + 24, original_offset + 64)
        self._reseal_directory(hostile, self.parsed)
        with self.assertRaisesRegex(bridge.FormatError, "offsets overlap"):
            bridge.parse_container(bytes(hostile))

    def test_owner_topology_tamper_rejected_after_full_reseal(self) -> None:
        hostile = bytearray(self.raw)
        first_record = self.parsed.header.directory_offset
        # Keep the mask in-range and singleton but violate exact ownership.
        struct.pack_into("<Q", hostile, first_record + 12, 2)
        self._reseal_directory(hostile, self.parsed)
        with self.assertRaisesRegex(bridge.FormatError, "two private"):
            bridge.parse_container(bytes(hostile))

    def test_frame_padding_rejected_after_hash_reseal(self) -> None:
        hostile = bytearray(self.raw)
        record = self.parsed.records[0]
        frame = self.parsed.frames[0]
        tail = record.payload_offset + bridge.FRAME_HEADER_SIZE + len(frame.encoded_bytes)
        self.assertLess(tail, record.payload_offset + record.physical_length)
        hostile[tail] = 1
        record_start = self.parsed.header.directory_offset
        physical = bytes(hostile[record.payload_offset : record.payload_offset + record.physical_length])
        hostile[record_start + 64 : record_start + 96] = bridge.sha256(physical)
        self._reseal_directory(hostile, self.parsed)
        with self.assertRaisesRegex(bridge.FormatError, "frame tail"):
            bridge.parse_container(bytes(hostile))

    def test_payload_tamper_rejected(self) -> None:
        hostile = bytearray(self.raw)
        hostile[self.parsed.records[0].payload_offset + bridge.FRAME_HEADER_SIZE] ^= 1
        with self.assertRaises(bridge.FormatError):
            bridge.parse_container(bytes(hostile))

    def test_semantic_decision_hash_tamper_rejected_after_structural_reseal(self) -> None:
        hostile = bytearray(self.raw)
        record_start = self.parsed.header.directory_offset
        hostile[record_start + 96] ^= 1
        self._reseal_directory(hostile, self.parsed)
        # This is structurally well formed and must be rejected by the
        # independent semantic decode, not accidentally by byte framing.
        bridge.parse_container(bytes(hostile))
        with self.assertRaisesRegex(ValueError, "decoded decision hash"):
            verify_synthetic_container(bytes(hostile))

    def test_serialized_model_is_the_only_probability_source(self) -> None:
        # Parse makes a fresh model strictly from literal container bytes.  A
        # separate incompatible Python model cannot affect verification.
        external = make_synthetic_model(state_count=1, reset_length=1)
        self.assertNotEqual(external, self.parsed.model)
        self.assertEqual(
            verify_synthetic_container(self.raw).parsed.model,
            bridge.deserialize_model(self.parsed.model_bytes),
        )


class CompletionLastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = Path(tempfile.mkdtemp(prefix="uwfa-completion-last-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.parent, ignore_errors=True)

    def test_complete_is_exclusive_and_disables_every_api_write(self) -> None:
        target = self.parent / "fresh"
        capsule = bridge.CompletionLastCapsule(target)
        artifact = capsule.write_bytes("container.bin", b"literal")
        complete = capsule.complete({"artifact_sha256": bridge.sha256(artifact.read_bytes()).hex()})
        self.assertEqual(set(path.name for path in target.iterdir()), {"container.bin", "COMPLETE.json"})
        self.assertEqual(json.loads(complete.read_text("utf-8"))["artifact_sha256"], bridge.sha256(b"literal").hex())
        with self.assertRaises(RuntimeError):
            capsule.write_bytes("late.bin", b"forbidden")
        with self.assertRaises(RuntimeError):
            capsule.complete({})

    def test_existing_target_and_path_traversal_rejected(self) -> None:
        existing = self.parent / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            bridge.CompletionLastCapsule(existing)
        capsule = bridge.CompletionLastCapsule(self.parent / "fresh")
        for hostile_name in ("../escape", "sub/leaf", "COMPLETE.json", ".."):
            with self.subTest(hostile_name=hostile_name):
                with self.assertRaises(ValueError):
                    capsule.write_bytes(hostile_name, b"x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
