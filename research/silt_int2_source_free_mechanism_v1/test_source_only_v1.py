#!/usr/bin/env python3
"""Hostile, source-only acceptance suite for SILT v1."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zlib

import numpy as np

import independent_decoder_v1 as independent
import silt_v1 as sm
from cupy_backend_v1 import search_metadata_cupy
from safe_publish import PublicationError, SafePublisher


GPU_TELEMETRY_RECEIPTS: list[dict[str, object]] = []


def make_model(alphabet: int = 2) -> sm.Q16TreeModel:
    roots, details = sm.generate_transformed_source(alphabet, 2048, 17, 8191 + alphabet, True)
    return sm.fit_model(alphabet, roots, details)


def make_expert(alphabet: int, lanes: int, vectors: int, seed: int) -> sm.ExpertInput:
    permutation = sm.deterministic_permutation(lanes, seed)
    selectors = sm.deterministic_selectors(lanes, alphabet, seed ^ 0x5A17)
    leaves = sm.synthesize_leaves(alphabet, vectors, lanes, seed + 100_000, True, permutation, selectors)
    return sm.ExpertInput.create(leaves, permutation, selectors)


def make_container(experts: int, alphabet: int = 2, large_others: bool = False) -> tuple[bytes, list[sm.ExpertInput]]:
    model = make_model(alphabet)
    sources: list[sm.ExpertInput] = []
    for index in range(experts):
        if large_others:
            lanes, vectors = 17, 128 if index == 0 else 3000
        else:
            lanes, vectors = 1 + index % 7, 1 + index % 5
        sources.append(make_expert(alphabet, lanes, vectors, 1000 + index))
    return sm.build_container(model, sources), sources


def replace_first_frame(packet: bytes, transform) -> bytes:
    parsed = sm.parse_container(packet)
    entry = parsed.entries[0]
    frame = bytearray(packet[entry.offset : entry.offset + entry.padded_bytes])
    fields = list(sm.FRAME_STRUCT.unpack(frame[: sm.FRAME_STRUCT.size]))
    body = bytearray(frame[sm.FRAME_HEADER_BYTES : fields[11]])
    fields, body = transform(fields, body)
    fields[14] = __import__("hashlib").sha256(body).digest()
    fields[15] = 0
    raw = sm.FRAME_STRUCT.pack(*fields)
    zero_header = raw + bytes(sm.FRAME_HEADER_BYTES - len(raw))
    fields[15] = zlib.crc32(zero_header) & 0xFFFFFFFF
    raw = sm.FRAME_STRUCT.pack(*fields)
    header = raw + bytes(sm.FRAME_HEADER_BYTES - len(raw))
    logical = int(fields[11])
    padded = int(fields[12])
    rebuilt = header + bytes(body) + bytes(padded - logical)
    output = bytearray(packet)
    output[entry.offset : entry.offset + entry.padded_bytes] = rebuilt
    return bytes(output)


class AlgebraTests(unittest.TestCase):
    def test_canonical_map_sets_are_distinct_bijections(self) -> None:
        for alphabet, ids in ((2, range(6)), (4, range(8))):
            maps = []
            for code in ids:
                output = []
                for left in range(alphabet):
                    for right in range(alphabet):
                        leaves = np.asarray([[left, right]], dtype=np.uint8)
                        lifted = sm.lift_forward(leaves, alphabet, [0, 1], [code])
                        decoded = sm.lift_inverse(lifted, 2, alphabet, [0, 1], [code])
                        self.assertTrue(np.array_equal(decoded, leaves))
                        output.append((int(lifted.roots[0]), int(lifted.detail_levels[0][0, 0])))
                self.assertEqual(len(set(output)), alphabet * alphabet)
                maps.append(tuple(output))
            self.assertEqual(len(set(maps)), len(ids))
        with self.assertRaises(sm.FormatError):
            sm.pack_selectors([6], 2)
        with self.assertRaises(sm.FormatError):
            sm.pack_selectors([7], 2)

    def test_arbitrary_positive_lanes_and_all_legal_selectors(self) -> None:
        rng = np.random.default_rng(10019)
        for alphabet, ids in ((2, range(6)), (4, range(8))):
            for lanes in (1, 2, 3, 5, 17, 97, 257):
                leaves = rng.integers(0, alphabet, size=(7, lanes), dtype=np.uint8)
                for code in ids:
                    permutation = sm.deterministic_permutation(lanes, lanes + code)
                    selectors = [code] * (lanes - 1)
                    lifted = sm.lift_forward(leaves, alphabet, permutation, selectors)
                    rebuilt = sm.lift_inverse(lifted, lanes, alphabet, permutation, selectors)
                    self.assertTrue(np.array_equal(rebuilt, leaves))


class ArithmeticTests(unittest.TestCase):
    @staticmethod
    def roundtrip_sequence(sequence: tuple[int, ...], row: tuple[int, ...]) -> None:
        encoder = sm.ArithmeticEncoder()
        for symbol in sequence:
            encoder.write(symbol, row)
        packet, meaningful = encoder.finish()
        decoder = sm.ArithmeticDecoder(packet, meaningful)
        decoded = tuple(decoder.read(row) for _ in sequence)
        if decoded != sequence:
            raise AssertionError((decoded, sequence))
        if decoder.reader.position != meaningful:
            raise AssertionError((decoder.reader.position, meaningful))

    def test_exhaustive_short_sequences_and_skew_rows(self) -> None:
        for length in range(1, 9):
            for sequence in itertools.product(range(2), repeat=length):
                self.roundtrip_sequence(sequence, (32768, 32768))
                self.roundtrip_sequence(sequence, (1, 65535))
        rng = np.random.default_rng(8821)
        for alphabet in (2, 4):
            row = (Q := sm.Q16_TOTAL // alphabet,) * alphabet
            for length in (1, 2, 31, 32, 33, 257):
                sequence = tuple(int(value) for value in rng.integers(0, alphabet, size=length))
                self.roundtrip_sequence(sequence, row)

    def test_meaningful_bit_truncation_extension_and_tails_reject_ordinary_decode(self) -> None:
        packet, _ = make_container(1, 2)
        parsed = sm.parse_container(packet)
        frame = bytes(parsed.frame_view(0))
        info = sm.parse_frame_header(frame)
        original_meaningful = info.meaningful_bits
        candidates = sorted(set([0, 1, 31, original_meaningful - 1, original_meaningful + 1] + list(range(original_meaningful))))
        for new_meaningful in candidates:
            if new_meaningful == original_meaningful or new_meaningful < 0:
                continue

            def mutate(fields, body, value=new_meaningful):
                fields[10] = value
                permutation_bytes = int(fields[7])
                selector_bytes = int(fields[8])
                payload_start = permutation_bytes + selector_bytes
                for bit in range(max(0, value), int(fields[9]) * 8):
                    byte_index = payload_start + bit // 8
                    if byte_index < len(body):
                        body[byte_index] &= ~(1 << (7 - (bit & 7)))
                return fields, body

            forged = replace_first_frame(packet, mutate)
            with self.assertRaises(sm.FormatError):
                sm.decode_container(forged)
            with self.assertRaises(independent.IndependentFormatError):
                independent.verify_decode_reencode(forged)

    def test_nonzero_guard_and_gf2_alias_with_valid_hashes_reject(self) -> None:
        packet, _ = make_container(1, 2)

        def guard(fields, body):
            payload_start = int(fields[7]) + int(fields[8])
            bit = int(fields[10]) - 1
            body[payload_start + bit // 8] |= 1 << (7 - (bit & 7))
            return fields, body

        with self.assertRaises(sm.FormatError):
            sm.decode_container(replace_first_frame(packet, guard))

        def alias(fields, body):
            selector_start = int(fields[7])
            body[selector_start] = (body[selector_start] & 0x1F) | (6 << 5)
            return fields, body

        with self.assertRaises(sm.FormatError):
            sm.decode_container(replace_first_frame(packet, alias))

    def test_appended_and_deleted_payload_byte_reject(self) -> None:
        packet, _ = make_container(1, 2)
        parsed = sm.parse_container(packet)
        frame = bytes(parsed.frame_view(0))
        fields = list(sm.FRAME_STRUCT.unpack(frame[: sm.FRAME_STRUCT.size]))
        body = bytearray(frame[sm.FRAME_HEADER_BYTES : fields[11]])
        for delta in (-1, 1):
            altered_fields = fields.copy()
            altered_body = body[:-1] if delta < 0 else body + b"\0"
            altered_fields[9] = int(altered_fields[9]) + delta
            altered_fields[11] = int(altered_fields[11]) + delta
            altered_fields[14] = __import__("hashlib").sha256(altered_body).digest()
            altered_fields[15] = 0
            raw = sm.FRAME_STRUCT.pack(*altered_fields)
            zero = raw + bytes(sm.FRAME_HEADER_BYTES - len(raw))
            altered_fields[15] = zlib.crc32(zero) & 0xFFFFFFFF
            raw = sm.FRAME_STRUCT.pack(*altered_fields)
            forged = raw + bytes(sm.FRAME_HEADER_BYTES - len(raw)) + altered_body + bytes(int(altered_fields[12]) - int(altered_fields[11]))
            with self.assertRaises(sm.FormatError):
                sm.decode_frame(parsed.model, forged)


class BoundsAndDirectoryTests(unittest.TestCase):
    def test_runtime_expert_counts_and_unequal_geometries(self) -> None:
        for count in (1, 128, 249, 250, 256):
            packet, sources = make_container(count, 2)
            parsed = sm.parse_container(packet)
            self.assertEqual(parsed.expert_count, count)
            self.assertEqual(len(parsed.entries), count)
            decoded = sm.decode_container(packet)
            self.assertTrue(all(np.array_equal(values, source.leaves) for values, source in zip(decoded, sources, strict=True)))
            receipt, independent_values, rebuilt = independent.verify_decode_reencode(
                packet, [sm.leaf_digest(source.leaves) for source in sources]
            )
            self.assertEqual(rebuilt, packet)
            self.assertEqual(receipt["status"], "PASS_INDEPENDENT_V1_CANONICAL_DECODE_REENCODE")
            self.assertTrue(all(np.array_equal(values, source.leaves) for values, source in zip(independent_values, sources, strict=True)))

    def test_expert_count_and_geometry_caps_reject_before_work(self) -> None:
        packet, _ = make_container(1, 2)
        parsed = sm.parse_container(packet)
        for experts in (0, 257, (1 << 32) - 1):
            fields = list(sm.GLOBAL_STRUCT.unpack(packet[: sm.GLOBAL_STRUCT.size]))
            fields[3] = experts
            fields[-1] = 0
            raw = sm.GLOBAL_STRUCT.pack(*fields)
            zero = raw + bytes(sm.GLOBAL_HEADER_BYTES - len(raw))
            fields[-1] = zlib.crc32(zero) & 0xFFFFFFFF
            raw = sm.GLOBAL_STRUCT.pack(*fields)
            forged = raw + bytes(sm.GLOBAL_HEADER_BYTES - len(raw)) + packet[sm.GLOBAL_HEADER_BYTES :]
            started = time.perf_counter()
            with self.assertRaises(sm.FormatError):
                sm.parse_container(forged)
            self.assertLess(time.perf_counter() - started, 0.25)
        for field_index in (5, 6):  # lanes, vectors
            def extreme(fields, body, index=field_index):
                fields[index] = (1 << 32) - 1
                fields[13] = (1 << 64) - 1
                return fields, body

            forged = replace_first_frame(packet, extreme)
            started = time.perf_counter()
            with self.assertRaises(sm.FormatError):
                sm.parse_container(forged)
            self.assertLess(time.perf_counter() - started, 0.25)
        with self.assertRaises(sm.FormatError):
            sm.permutation_byte_count(sm.MAX_LANES + 1)


class ColdLedgerTests(unittest.TestCase):
    def test_exact_audit_counterexample_and_strict_boundary(self) -> None:
        ledger = sm.audit_unequal_frame_counterexample()
        row = ledger["cold"][0]
        self.assertEqual((row["cold_amplification_numerator"], row["cold_amplification_denominator"]), (12, 5))
        self.assertFalse(row["cold_below_two_by_integer_cross_multiplication"])
        boundary = sm.layout_cold_ledger(8192, [4096] * 4)
        for boundary_row in boundary["cold"]:
            self.assertEqual(boundary_row["cold_amplification_float"], 2.0)
            self.assertFalse(boundary_row["cold_below_two_by_integer_cross_multiplication"])

    def test_unrelated_padding_cannot_change_owner_denominator(self) -> None:
        first = sm.layout_cold_ledger(8192, [4096] + [8192] * 7)["cold"][0]
        second = sm.layout_cold_ledger(8192, [4096] + [16384] * 7)["cold"][0]
        keys = (
            "owner_share_numerator",
            "owner_share_denominator",
            "cold_amplification_numerator",
            "cold_amplification_denominator",
        )
        self.assertEqual(tuple(first[key] for key in keys), tuple(second[key] for key in keys))

    def test_real_unequal_frames_and_instrumented_page_union(self) -> None:
        packet, _ = make_container(8, 2, large_others=True)
        parsed = sm.parse_container(packet)
        frame_pages = [entry.padded_bytes for entry in parsed.entries]
        self.assertEqual(frame_pages, [4096] + [8192] * 7)
        ledger = sm.physical_ledger(packet)
        self.assertTrue(ledger["layout"]["owner_sum_equals_container"])
        self.assertFalse(ledger["cold_below_two"])
        for row, trace in zip(ledger["layout"]["cold"], ledger["instrumented_page_unions"], strict=True):
            self.assertEqual(row["cold_bytes"], trace["union_bytes"])
            self.assertTrue(trace["union_matches_expected"])


class PublicationAndRootTests(unittest.TestCase):
    def test_safe_exclusive_publication_and_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "result")
            with SafePublisher(output, "0" * 64) as publisher:
                publisher.write("a.bin", b"abc")
                receipt = publisher.finish()
            self.assertTrue(os.path.isfile(os.path.join(output, "COMPLETE")))
            self.assertEqual(receipt.output_path, output)
            with self.assertRaises(PublicationError):
                with SafePublisher(output, "0" * 64) as publisher:
                    publisher.write("b.bin", b"def")
                    publisher.finish()

    def test_symlink_parent_fault_and_concurrent_no_replace(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            real = os.path.join(parent, "real")
            os.mkdir(real)
            link = os.path.join(parent, "link")
            os.symlink(real, link)
            with self.assertRaises(Exception):
                SafePublisher(os.path.join(link, "bad"), "0" * 64)
            for stage in (
                "artifact:a.bin",
                "artifact_index_fsynced",
                "completion_linked_and_directory_fsynced",
            ):
                fault_output = os.path.join(parent, f"fault-{stage.replace(':', '-')}")
                with self.assertRaises(PublicationError):
                    with SafePublisher(fault_output, "0" * 64, fault_after=stage) as publisher:
                        publisher.write("a.bin", b"abc")
                        publisher.finish()
                self.assertFalse(os.path.exists(fault_output))

            symlink_target = os.path.join(parent, "attacker-target")
            os.mkdir(symlink_target)
            symlink_output = os.path.join(parent, "attacker-link")
            os.symlink(symlink_target, symlink_output)
            with self.assertRaises(PublicationError):
                with SafePublisher(symlink_output, "0" * 64) as publisher:
                    publisher.write("a.bin", b"abc")
                    publisher.finish()

            target = os.path.join(parent, "race")
            publishers = [SafePublisher(target, "0" * 64) for _ in range(2)]
            for index, publisher in enumerate(publishers):
                publisher.write(f"p{index}.bin", bytes([index]))
            barrier = threading.Barrier(2)
            outcomes: list[str] = []

            def finish(publisher):
                barrier.wait()
                try:
                    publisher.finish()
                    outcomes.append("pass")
                except PublicationError:
                    outcomes.append("fail")
                    publisher.abort()

            threads = [threading.Thread(target=finish, args=(publisher,)) for publisher in publishers]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sorted(outcomes), ["fail", "pass"])
            self.assertTrue(os.path.isfile(os.path.join(target, "COMPLETE")))

    def test_content_root_rejects_extra_missing_symlink_and_poisoned_cwd(self) -> None:
        import source_bootstrap

        source_dir = os.path.dirname(os.path.abspath(__file__))
        observed, _ = source_bootstrap.read_source_tree(source_dir)
        self.assertEqual(len(observed), 64)
        with tempfile.TemporaryDirectory() as copied:
            for name in source_bootstrap.ROOT_FILES:
                shutil.copy2(os.path.join(source_dir, name), os.path.join(copied, name))
            with open(os.path.join(copied, "extra.py"), "w", encoding="utf-8") as handle:
                handle.write("raise RuntimeError('poison')\n")
            with self.assertRaises(source_bootstrap.BootstrapError):
                source_bootstrap.read_source_tree(copied)
            os.unlink(os.path.join(copied, "extra.py"))
            os.unlink(os.path.join(copied, "README.md"))
            with self.assertRaises(source_bootstrap.BootstrapError):
                source_bootstrap.read_source_tree(copied)
            os.symlink(os.path.join(source_dir, "README.md"), os.path.join(copied, "README.md"))
            with self.assertRaises(source_bootstrap.BootstrapError):
                source_bootstrap.read_source_tree(copied)

        with tempfile.TemporaryDirectory() as parent:
            linked = os.path.join(parent, "linked_source")
            os.symlink(source_dir, linked)
            with self.assertRaises(Exception):
                source_bootstrap.read_source_tree(linked)

        with tempfile.TemporaryDirectory() as poison:
            marker = os.path.join(poison, "POISON_EXECUTED")
            with open(os.path.join(poison, "sitecustomize.py"), "w", encoding="utf-8") as handle:
                handle.write(f"open({marker!r}, 'w').write('bad')\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    os.path.join(source_dir, "source_bootstrap.py"),
                    "--source-dir",
                    source_dir,
                    "--print-observed-root",
                ],
                cwd=poison,
                env={**os.environ, "PYTHONPATH": poison},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), observed)
            self.assertFalse(os.path.exists(marker))


class MandatoryGpuTests(unittest.TestCase):
    def test_canonical_transfer_accounting_and_exact_cpu_cupy(self) -> None:
        alphabet = 2
        lanes = 97
        hidden = 0x6C31
        permutation = sm.deterministic_permutation(lanes, hidden)
        selectors = sm.deterministic_selectors(lanes, alphabet, hidden ^ 0x5A17)
        train = sm.synthesize_leaves(alphabet, 2048, lanes, 871, True, permutation, selectors)
        validation = sm.synthesize_leaves(alphabet, 1024, lanes, 872, True, permutation, selectors)
        candidates = [0x0B51, 0x193D, 0x2E71, hidden, 0x79A3, 0x8849, 0xA117, 0xD20B]
        result = search_metadata_cupy(train, validation, alphabet, candidates, require_rtx_5090=True)
        telemetry = result.telemetry
        self.assertEqual(telemetry["device_name"], "NVIDIA GeForce RTX 5090")
        self.assertTrue(telemetry["gpu_mapping_asserted_by_pci_bus_id"])
        self.assertTrue(telemetry["cpu_cupy_selected_coefficients_equal"])
        self.assertTrue(telemetry["cupy_inverse_roundtrip_equal"])
        self.assertEqual(telemetry["h2d_bytes"], 305832)
        self.assertEqual(telemetry["d2h_bytes"], 2483200)
        self.assertEqual(telemetry["model_h2d_bytes"], 0)
        for key in (
            "gpu_uuid",
            "cuda_logical_index",
            "nvml_physical_index",
            "cuda_pci_bus_id",
            "host_rss_baseline_bytes",
            "host_rss_peak_bytes",
            "host_rss_delta_bytes",
            "vram_process_baseline_used_bytes",
            "vram_process_peak_used_bytes",
            "vram_process_delta_bytes",
            "vram_device_baseline_used_bytes",
            "vram_device_peak_used_bytes",
            "vram_device_delta_bytes",
        ):
            self.assertIn(key, telemetry)
        self.assertGreater(telemetry["resource_sample_count"], 0)
        self.assertEqual(
            telemetry["h2d_bytes"],
            sum(row["logical_array_bytes"] for row in telemetry["array_transfers"] if row["direction"] == "H2D"),
        )
        self.assertEqual(
            telemetry["d2h_bytes"],
            sum(row["logical_array_bytes"] for row in telemetry["array_transfers"] if row["direction"] == "D2H"),
        )
        GPU_TELEMETRY_RECEIPTS.append(telemetry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
