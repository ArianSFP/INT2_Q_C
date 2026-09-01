#!/usr/bin/env python3
"""Independent, payload-free audit for the frozen UWFA bridge reference."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import struct
import sys
import tempfile
import types
import unittest
import zlib


PINNED_MANIFEST_SHA256 = (
    "51f158c7f82fad81bd2b15d30e6581a2847e0e436d98f085055b8d818bf43f31"
)
EXPECTED_HASHES = {
    "README.md": "bc95180b98688999e3dd75797cce4b8f26fa0499e5ee444725760e3142944625",
    "synthetic_fixture.py": "f495614561ff5ab9b263b3279115ebe95bbecca1e30443589dd7d1f0f3e795c7",
    "test_uwfa_bridge.py": "3d276087a7907e734e5443acdba1316acc11ca983b4662d99c368fa7ad90a2e9",
    "uwfa_bridge.py": "225b3a4148ff57e4568e4f1bf3ccb002ab49a9f27eeea6566a95bf00ea236ffd",
}
EXPECTED_INVENTORY = set(EXPECTED_HASHES) | {"SOURCE_MANIFEST.json"}

PAGE = 4096
HEADER = 4096
RECORD = 256
BLOCKS = 15
CRC_OFFSET = 16
ROOT_START = 400
ROOT_END = 432


class AuditFailure(RuntimeError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    chain = [absolute]
    chain.extend(absolute.parents)
    for item in reversed(chain):
        try:
            mode = os.lstat(item).st_mode
        except FileNotFoundError as exc:
            raise AuditFailure(f"missing source ancestor: {item}") from exc
        if stat.S_ISLNK(mode):
            raise AuditFailure(f"symlink source ancestor: {item}")


def _read_regular_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuditFailure(f"source is not a regular file: {path.name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AuditFailure(f"source identity changed while held: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def authenticate_source(root: Path) -> dict[str, bytes]:
    root = root.absolute()
    _reject_symlink_ancestors(root)
    entries = {entry.name for entry in os.scandir(root)}
    if entries != EXPECTED_INVENTORY:
        raise AuditFailure(
            f"producer inventory mismatch: expected={sorted(EXPECTED_INVENTORY)!r} "
            f"actual={sorted(entries)!r}"
        )
    source: dict[str, bytes] = {}
    for name in sorted(entries):
        path = root / name
        _reject_symlink_ancestors(path)
        source[name] = _read_regular_nofollow(path)
    if digest(source["SOURCE_MANIFEST.json"]) != PINNED_MANIFEST_SHA256:
        raise AuditFailure("pinned producer manifest digest mismatch")
    manifest = json.loads(source["SOURCE_MANIFEST.json"])
    listed = {row["path"]: row["sha256"] for row in manifest["files"]}
    if listed != EXPECTED_HASHES:
        raise AuditFailure("producer manifest file inventory/hash table mismatch")
    for name, expected in EXPECTED_HASHES.items():
        if digest(source[name]) != expected:
            raise AuditFailure(f"producer file digest mismatch: {name}")
    return source


def load_authenticated_modules(source: dict[str, bytes]):
    """Compile and execute only the already authenticated in-memory bytes."""

    sys.dont_write_bytecode = True
    bridge = types.ModuleType("uwfa_bridge")
    bridge.__file__ = "<authenticated:uwfa_bridge.py>"
    sys.modules["uwfa_bridge"] = bridge
    exec(compile(source["uwfa_bridge.py"], bridge.__file__, "exec"), bridge.__dict__)

    fixture = types.ModuleType("synthetic_fixture")
    fixture.__file__ = "<authenticated:synthetic_fixture.py>"
    sys.modules["synthetic_fixture"] = fixture
    exec(compile(source["synthetic_fixture.py"], fixture.__file__, "exec"), fixture.__dict__)

    producer_tests = types.ModuleType("test_uwfa_bridge")
    producer_tests.__file__ = "<authenticated:test_uwfa_bridge.py>"
    sys.modules["test_uwfa_bridge"] = producer_tests
    exec(
        compile(source["test_uwfa_bridge.py"], producer_tests.__file__, "exec"),
        producer_tests.__dict__,
    )
    return bridge, fixture, producer_tests


def _ranges(image: bytes | bytearray) -> tuple[int, ...]:
    return struct.unpack_from("<" + "Q" * 14, image, 72)


def _fix_header_crc(image: bytearray) -> None:
    struct.pack_into("<I", image, CRC_OFFSET, 0)
    struct.pack_into("<I", image, CRC_OFFSET, zlib.crc32(image[:HEADER]) & 0xFFFFFFFF)


def _fix_frame_crcs(image: bytearray, frame_offset: int, encoded_length: int) -> None:
    stream = bytes(image[frame_offset + 64 : frame_offset + 64 + encoded_length])
    struct.pack_into("<I", image, frame_offset + 40, zlib.crc32(stream) & 0xFFFFFFFF)
    struct.pack_into("<I", image, frame_offset + 44, 0)
    struct.pack_into(
        "<I",
        image,
        frame_offset + 44,
        zlib.crc32(image[frame_offset : frame_offset + 64]) & 0xFFFFFFFF,
    )


def independent_reseal(image: bytearray, refresh_payload_hashes: bool = True) -> None:
    """Reseal using an independent implementation of the documented root ABI."""

    (
        container_bytes,
        metadata_offset,
        metadata_actual,
        _metadata_region,
        model_offset,
        model_actual,
        _model_region,
        directory_offset,
        directory_actual,
        _directory_region,
        _frames_offset,
        _frames_end,
        _final_padding_offset,
        _final_padding_length,
    ) = _ranges(image)
    if container_bytes != len(image):
        raise AuditFailure("counterexample reseal length mismatch")
    metadata = bytes(image[metadata_offset : metadata_offset + metadata_actual])
    model = bytes(image[model_offset : model_offset + model_actual])
    if refresh_payload_hashes:
        for ordinal in range(BLOCKS):
            record = directory_offset + ordinal * RECORD
            payload_offset, physical_length = struct.unpack_from("<QQ", image, record + 24)
            payload = bytes(image[payload_offset : payload_offset + physical_length])
            image[record + 64 : record + 96] = hashlib.sha256(payload).digest()
    directory = bytes(image[directory_offset : directory_offset + directory_actual])
    image[464:496] = hashlib.sha256(metadata).digest()
    image[496:528] = hashlib.sha256(model).digest()
    image[528:560] = hashlib.sha256(directory).digest()

    normalized = bytearray(image[:HEADER])
    struct.pack_into("<I", normalized, CRC_OFFSET, 0)
    normalized[ROOT_START:ROOT_END] = bytes(ROOT_END - ROOT_START)
    hasher = hashlib.sha256()
    hasher.update(normalized)
    hasher.update(metadata)
    hasher.update(model)
    hasher.update(directory)
    for ordinal in range(BLOCKS):
        record = directory_offset + ordinal * RECORD
        payload_offset = struct.unpack_from("<Q", image, record + 24)[0]
        encoded_length = struct.unpack_from("<Q", image, record + 56)[0]
        hasher.update(image[payload_offset : payload_offset + 64 + encoded_length])
    image[ROOT_START:ROOT_END] = hasher.digest()
    _fix_header_crc(image)


def independently_expected_ledger(parsed) -> tuple[list[tuple[set[int], Fraction]], Fraction]:
    raw = parsed.raw
    records = parsed.records
    global_bytes = len(raw) - sum(record.physical_length for record in records)

    def pages(offset: int, length: int) -> set[int]:
        if not length:
            return set()
        return set(range(offset // PAGE, (offset + length - 1) // PAGE + 1))

    rows: list[tuple[set[int], Fraction]] = []
    for expert in range(parsed.header.expert_count):
        touched = pages(0, HEADER)
        touched |= pages(parsed.header.metadata_offset, parsed.header.metadata_actual)
        touched |= pages(parsed.header.model_offset, parsed.header.model_actual)
        share = Fraction(global_bytes, parsed.header.expert_count)
        for record in records:
            if record.owner_mask & (1 << expert):
                touched |= pages(
                    parsed.header.directory_offset + record.ordinal * RECORD, RECORD
                )
                touched |= pages(record.payload_offset, record.physical_length)
                share += Fraction(record.physical_length, record.owner_mask.bit_count())
        rows.append((touched, share))
    return rows, sum((share for _, share in rows), Fraction())


def make_adversarial_suite(bridge, fixture, pristine: bytes):
    class IndependentAdversarialTests(unittest.TestCase):
        raw = pristine
        parsed = bridge.parse_container(pristine)

        def assert_structural_reject(self, image: bytes | bytearray) -> None:
            with self.assertRaises((bridge.FormatError, ValueError, OverflowError)):
                bridge.parse_container(bytes(image))

        def test_arithmetic_extremes_and_canonical_reencode(self) -> None:
            rng = random.Random(0xA55C2202)
            for length in (0, 1, 2, 3, 7, 8, 31, 32, 33, 255, 1024):
                bits = [rng.randrange(2) for _ in range(length)]
                freqs = [rng.choice((1, 2, 32767, 32768, 65534, 65535)) for _ in bits]
                encoder = bridge.BinaryArithmeticEncoder()
                for bit, freq in zip(bits, freqs):
                    encoder.encode(bit, freq)
                payload, logical = encoder.finish()
                decoder = bridge.BinaryArithmeticDecoder(payload, logical)
                restored = [decoder.decode(freq) for freq in freqs]
                self.assertEqual(restored, bits)
                second = bridge.BinaryArithmeticEncoder()
                for bit, freq in zip(restored, freqs):
                    second.encode(bit, freq)
                self.assertEqual(second.finish(), (payload, logical))

        def test_all_header_ranges_reject_u64_overflow_or_noncanonicality(self) -> None:
            # Every QWORD in the canonical range table is independently hostile.
            for offset in range(72, 184, 8):
                hostile = bytearray(self.raw)
                struct.pack_into("<Q", hostile, offset, 0xFFFFFFFFFFFFFFFF)
                _fix_header_crc(hostile)
                with self.subTest(offset=offset):
                    self.assert_structural_reject(hostile)

        def test_directory_gap_and_overlap_rejected_after_reseal(self) -> None:
            for delta in (-64, 64):
                hostile = bytearray(self.raw)
                directory_offset = self.parsed.header.directory_offset
                original = struct.unpack_from("<Q", hostile, directory_offset + 24)[0]
                struct.pack_into("<Q", hostile, directory_offset + 24, original + delta)
                independent_reseal(hostile)
                with self.subTest(delta=delta):
                    self.assert_structural_reject(hostile)

        def test_root_normalization_and_crc_are_both_checked(self) -> None:
            hostile = bytearray(self.raw)
            hostile[ROOT_START] ^= 1
            _fix_header_crc(hostile)
            self.assert_structural_reject(hostile)
            hostile = bytearray(self.raw)
            hostile[CRC_OFFSET] ^= 1
            self.assert_structural_reject(hostile)

        def test_canonical_padding_and_extra_page_rejected(self) -> None:
            hostile = bytearray(self.raw)
            hostile[-1] ^= 1
            self.assert_structural_reject(hostile)

            hostile = bytearray(self.raw + bytes(PAGE))
            struct.pack_into("<Q", hostile, 72, len(hostile))
            old_final = struct.unpack_from("<Q", hostile, 176)[0]
            struct.pack_into("<Q", hostile, 176, old_final + PAGE)
            independent_reseal(hostile)
            self.assert_structural_reject(hostile)

        def test_frame_tail_rejected_even_after_full_hash_reseal(self) -> None:
            hostile = bytearray(self.raw)
            record = self.parsed.records[0]
            frame = self.parsed.frames[0]
            tail = record.payload_offset + 64 + len(frame.encoded_bytes)
            self.assertLess(tail, record.payload_offset + record.physical_length)
            hostile[tail] = 1
            independent_reseal(hostile)
            self.assert_structural_reject(hostile)

        def test_fifteen_block_owner_topology_and_conservation(self) -> None:
            masks = [record.owner_mask for record in self.parsed.records]
            self.assertEqual(len(masks), 15)
            self.assertTrue(all(mask.bit_count() == 1 for mask in masks[:12]))
            self.assertTrue(all(mask.bit_count() == 2 for mask in masks[12:]))
            for expert in range(6):
                self.assertEqual(sum(mask == 1 << expert for mask in masks[:12]), 2)
                self.assertEqual(sum(bool(mask & 1 << expert) for mask in masks[12:]), 1)
            rows, total = independently_expected_ledger(self.parsed)
            ledger = bridge.routed_read_ledger(self.parsed)
            self.assertEqual(total, Fraction(len(self.raw)))
            self.assertEqual(ledger.storage_conservation, total)
            for expected, actual in zip(rows, ledger.experts):
                pages, share = expected
                self.assertEqual(actual.touched_pages, tuple(sorted(pages)))
                self.assertEqual(actual.touched_bytes, len(pages) * PAGE)
                self.assertEqual(actual.storage_share, share)
                self.assertEqual(actual.amplification, float(Fraction(len(pages) * PAGE, 1) / share))

        def test_actual_rate_and_f_formula(self) -> None:
            verification = fixture.verify_synthetic_container(self.raw)
            rate = Fraction(8 * len(self.raw), self.parsed.header.source_weights)
            self.assertEqual(verification.score.rate, rate)
            expected_f = 0.03 * 2.0 ** (2.0 * float(rate))
            self.assertEqual(verification.score.f_actual, expected_f)

        def test_serialized_model_only_and_causal_metadata_seed(self) -> None:
            original_constructor = fixture.make_synthetic_model
            fixture.make_synthetic_model = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("external model constructor was consulted")
            )
            try:
                fixture.verify_synthetic_container(self.raw)
            finally:
                fixture.make_synthetic_model = original_constructor

            hostile = bytearray(self.raw)
            seed_offset = self.parsed.header.metadata_offset + 32
            struct.pack_into("<I", hostile, seed_offset, struct.unpack_from("<I", hostile, seed_offset)[0] ^ 1)
            independent_reseal(hostile)
            bridge.parse_container(bytes(hostile))
            with self.assertRaisesRegex(ValueError, "decision hash|re-encode"):
                fixture.verify_synthetic_container(bytes(hostile))

        def test_serialized_model_change_is_semantically_observed(self) -> None:
            hostile = bytearray(self.raw)
            model_offset = self.parsed.header.model_offset
            row_count = struct.unpack_from("<Q", hostile, model_offset + 32)[0]
            for index in range(row_count):
                struct.pack_into("<H", hostile, model_offset + 64 + index * 8 + 6, 1)
            struct.pack_into("<I", hostile, model_offset + 40, 0)
            actual = self.parsed.header.model_actual
            crc = zlib.crc32(hostile[model_offset : model_offset + actual]) & 0xFFFFFFFF
            struct.pack_into("<I", hostile, model_offset + 40, crc)
            independent_reseal(hostile)
            bridge.parse_container(bytes(hostile))
            with self.assertRaisesRegex(ValueError, "decision hash|re-encode"):
                fixture.verify_synthetic_container(bytes(hostile))

        def test_payload_tamper_survives_structure_but_fails_semantics(self) -> None:
            hostile = bytearray(self.raw)
            record = self.parsed.records[0]
            hostile[record.payload_offset + 64] ^= 1
            _fix_frame_crcs(hostile, record.payload_offset, record.encoded_length)
            independent_reseal(hostile)
            bridge.parse_container(bytes(hostile))
            with self.assertRaisesRegex(ValueError, "decision hash|re-encode"):
                fixture.verify_synthetic_container(bytes(hostile))

        def test_logical_length_alternate_fails_required_reencode(self) -> None:
            candidate = next(
                record
                for record in self.parsed.records
                if record.logical_bits < record.encoded_length * 8
            )
            hostile = bytearray(self.raw)
            new_logical = candidate.logical_bits + 1
            struct.pack_into("<Q", hostile, candidate.payload_offset + 32, new_logical)
            _fix_frame_crcs(hostile, candidate.payload_offset, candidate.encoded_length)
            record_offset = self.parsed.header.directory_offset + candidate.ordinal * RECORD
            struct.pack_into("<Q", hostile, record_offset + 40, new_logical)
            independent_reseal(hostile)
            bridge.parse_container(bytes(hostile))
            with self.assertRaisesRegex(ValueError, "re-encode"):
                fixture.verify_synthetic_container(bytes(hostile))

        def test_random_single_byte_tampering_is_rejected(self) -> None:
            rng = random.Random(0x51F158C7)
            offsets = {rng.randrange(len(self.raw)) for _ in range(96)}
            for offset in offsets:
                hostile = bytearray(self.raw)
                hostile[offset] ^= 1
                with self.subTest(offset=offset):
                    with self.assertRaises((bridge.FormatError, ValueError)):
                        fixture.verify_synthetic_container(bytes(hostile))

    return IndependentAdversarialTests


def build_accepted_counterexamples(bridge, fixture, pristine: bytes) -> dict[str, object]:
    parsed = bridge.parse_container(pristine)
    accepted: dict[str, object] = {}

    # 1. Coordinated scale semantics are ignored by the decision-hash stand-in.
    scale = bytearray(pristine)
    metadata_scale = parsed.header.metadata_offset + 128 + 144 + 5184 + 15
    old_scale = struct.unpack_from("<H", scale, metadata_scale)[0]
    new_scale = old_scale ^ 1
    struct.pack_into("<H", scale, metadata_scale, new_scale)
    struct.pack_into("<H", scale, parsed.header.directory_offset + 22, new_scale)
    independent_reseal(scale)
    scale_verification = fixture.verify_synthetic_container(bytes(scale))
    accepted["semantic_scale_mutation"] = {
        "accepted": True,
        "old_binary16_bits": old_scale,
        "new_binary16_bits": new_scale,
        "decoded_hash_unchanged": (
            scale_verification.decoded_hash
            == parsed.header.bindings.decoded_reconstruction_hash
        ),
        "meaning": "synthetic callback hashes decisions, not reconstructed values",
    }

    # 2. MSE and evidence bindings are assertions unless compared externally.
    asserted = bytearray(pristine)
    struct.pack_into("<d", asserted, 192, 0.0)
    asserted[208:240] = hashlib.sha256(b"untrusted replacement baseline").digest()
    asserted[336:368] = hashlib.sha256(b"untrusted replacement source manifest").digest()
    asserted[368:400] = hashlib.sha256(b"untrusted replacement bootstrap").digest()
    independent_reseal(asserted)
    assertion_verification = fixture.verify_synthetic_container(bytes(asserted))
    accepted["untrusted_evidence_and_zero_mse"] = {
        "accepted": True,
        "reported_relative_mse": assertion_verification.score.relative_mse,
        "reported_f": assertion_verification.score.f_actual,
        "meaning": "bound hashes and score are not authenticated/recomputed by this API",
    }

    # 3. Source weight count need not equal decoded block geometry.
    geometry = bytearray(pristine)
    original_weights = parsed.header.source_weights
    replacement_weights = None
    for delta in range(1, 20000):
        trial = original_weights + delta
        floor = (215 * trial + 799) // 800
        expected = max(parsed.header.frames_end, floor)
        expected = (expected + PAGE - 1) & -PAGE
        if expected == len(pristine):
            replacement_weights = trial
            break
    if replacement_weights is None:
        raise AuditFailure("could not construct same-layout geometry counterexample")
    struct.pack_into("<Q", geometry, 32, replacement_weights)
    struct.pack_into("<Q", geometry, parsed.header.metadata_offset + 16, replacement_weights)
    independent_reseal(geometry)
    geometry_verification = fixture.verify_synthetic_container(bytes(geometry))
    decoded_decisions = sum(record.decision_count for record in geometry_verification.parsed.records)
    accepted["source_weight_geometry_mismatch"] = {
        "accepted": True,
        "header_source_weights": replacement_weights,
        "sum_decision_counts": decoded_decisions,
        "rate": float(geometry_verification.score.rate),
        "meaning": "source_weights is not tied to decoded block geometry",
    }

    # 4. A canonical >2.5-bpw packet parses/verifies; flags are not gates.
    overweight = bytearray(pristine)
    minimal_length = (parsed.header.frames_end + PAGE - 1) & -PAGE
    del overweight[minimal_length:]
    replacement_weights = max(1, int(minimal_length * 3.1))
    struct.pack_into("<Q", overweight, 72, len(overweight))
    struct.pack_into("<Q", overweight, 32, replacement_weights)
    struct.pack_into("<Q", overweight, parsed.header.metadata_offset + 16, replacement_weights)
    struct.pack_into("<Q", overweight, 176, len(overweight) - parsed.header.frames_end)
    independent_reseal(overweight)
    overweight_verification = fixture.verify_synthetic_container(bytes(overweight))
    if overweight_verification.score.rate_in_range:
        raise AuditFailure("overweight counterexample did not exceed the rate cap")
    accepted["rate_and_f_flags_not_enforced"] = {
        "accepted": True,
        "rate": float(overweight_verification.score.rate),
        "rate_in_range": overweight_verification.score.rate_in_range,
        "f": overweight_verification.score.f_actual,
        "f_pass": overweight_verification.score.f_pass,
        "maximum_cold_amplification": overweight_verification.ledger.maximum_amplification,
        "meaning": "verify returns gate flags/ledger but does not reject gate misses",
    }

    # 5. COMPLETE is API-last, not immutable or an authenticated bootstrap.
    temp = Path(tempfile.mkdtemp(prefix="uwfa-audit-completion-"))
    try:
        target = temp / "capsule"
        capsule = bridge.CompletionLastCapsule(target)
        artifact = capsule.write_bytes("container.bin", b"before")
        capsule.complete({"sha256": hashlib.sha256(b"before").hexdigest()})
        artifact.write_bytes(b"after")
        accepted["post_complete_external_mutation"] = {
            "accepted_by_filesystem": artifact.read_bytes() == b"after",
            "complete_still_present": (target / "COMPLETE.json").exists(),
            "meaning": "capsule disables only its own write API",
        }

        symlink_accepted: bool | None = None
        if hasattr(os, "symlink"):
            real_parent = temp / "real-parent"
            real_parent.mkdir()
            link_parent = temp / "linked-parent"
            try:
                os.symlink(real_parent, link_parent, target_is_directory=True)
                bridge.CompletionLastCapsule(link_parent / "through-link")
                symlink_accepted = True
            except (OSError, NotImplementedError, PermissionError):
                symlink_accepted = None
        accepted["symlink_ancestor"] = {
            "accepted": symlink_accepted,
            "meaning": "reference capsule is deliberately not the external bootstrap",
        }
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    return accepted


def run_suite(module_or_case) -> tuple[unittest.TestResult, str]:
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module_or_case) if isinstance(
        module_or_case, types.ModuleType
    ) else unittest.defaultTestLoader.loadTestsFromTestCase(module_or_case)
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    return result, stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("producer", type=Path)
    args = parser.parse_args()

    source = authenticate_source(args.producer)
    bridge, fixture, producer_tests = load_authenticated_modules(source)
    producer_result, producer_log = run_suite(producer_tests)
    print(producer_log, end="")
    if not producer_result.wasSuccessful() or producer_result.testsRun != 17:
        raise AuditFailure("producer test suite did not pass exactly 17 tests")

    pristine = fixture.build_synthetic_container(log2n=16)
    adversarial_case = make_adversarial_suite(bridge, fixture, pristine)
    adversarial_result, adversarial_log = run_suite(adversarial_case)
    print(adversarial_log, end="")
    if not adversarial_result.wasSuccessful():
        raise AuditFailure("independent adversarial suite failed")

    verification = fixture.verify_synthetic_container(pristine)
    counterexamples = build_accepted_counterexamples(bridge, fixture, pristine)
    summary = {
        "format": "uwfa-sc-v2-integrated-bridge-reference-independent-audit-v1",
        "producer_manifest_sha256": PINNED_MANIFEST_SHA256,
        "claim": "PASS_REFERENCE_MECHANISMS_WITH_BLOCKING_SPEC_DIVERGENCES",
        "payload_authority": False,
        "qwen_evidence": False,
        "producer_tests": {
            "run": producer_result.testsRun,
            "failures": len(producer_result.failures),
            "errors": len(producer_result.errors),
        },
        "independent_tests": {
            "run": adversarial_result.testsRun,
            "failures": len(adversarial_result.failures),
            "errors": len(adversarial_result.errors),
        },
        "fixture": {
            "container_bytes": len(pristine),
            "source_decisions": verification.parsed.header.source_weights,
            "physical_bpw_numerator": verification.score.rate.numerator,
            "physical_bpw_denominator": verification.score.rate.denominator,
            "physical_bpw": float(verification.score.rate),
            "relative_mse_header_fixture": verification.score.relative_mse,
            "f_actual_from_header_fixture": verification.score.f_actual,
            "maximum_cold_read_amplification": verification.ledger.maximum_amplification,
            "storage_conservation_bytes": verification.ledger.storage_conservation.numerator,
            "semantic_decoded_hash": verification.decoded_hash.hex(),
        },
        "accepted_counterexamples": counterexamples,
        "blocking_spec_divergences": [
            "No real inherited-STRATA parser or inverse RHT/group/XKLT reconstruction callback; synthetic decoded hash covers decisions only.",
            "No independent original-BF16 FP64 SSE/energy recomputation; F consumes baseline_relative_mse asserted in the header.",
            "Evidence hashes are structurally bound but not compared against caller-supplied trusted expected hashes.",
            "source_weights is not constrained to the block geometry or total restored decisions.",
            "parse/verify expose rate/F/cold results but do not reject rate, F, or cold-read gate failures.",
            "CompletionLastCapsule is not the normative pinned launcher: it does not reject symlink ancestors, retain authenticated input descriptors, execute an immutable snapshot, or detect direct post-COMPLETE mutations.",
            "The synthetic fixture binds dummy source/bootstrap hashes rather than an externally pinned launcher/source identity.",
        ],
        "positive_mechanisms": [
            "Exact sealed source inventory and authenticated in-memory execution",
            "Canonical explicit dense UWFA model rows",
            "Serialized-model-only causal decode(original_freq1) adapter",
            "Canonical arithmetic byte and logical-length re-encode",
            "Strict canonical range/order/non-overlap/padding checks",
            "Fifteen-frame six-expert owner topology",
            "Exact rational physical rate and independently reproduced page/share ledger",
            "Header CRC plus normalized semantic root and per-frame hashes/CRCs",
        ],
    }
    print("AUDIT_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as exc:
        print(f"AUDIT_FAILURE={exc}", file=sys.stderr)
        raise SystemExit(1)

