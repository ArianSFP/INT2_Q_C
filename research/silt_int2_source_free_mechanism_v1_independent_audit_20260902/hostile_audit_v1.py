#!/usr/bin/env python3
"""Independent hostile checks for the SILT-v1 postimplementation boundary.

This audit never writes inside the producer directory and accepts no model or
weight payload.  It deliberately confirms both successful invariants and
security/read-path counterexamples that are absent from the producer suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import struct
import sys
import tempfile
import time
import zlib
from fractions import Fraction
from pathlib import Path


ROOT_FILES = (
    "POSTIMPLEMENTATION_REVIEW.md",
    "README.md",
    "cupy_backend_v1.py",
    "design_lock.json",
    "independent_decoder_v1.py",
    "run_synthetic_v1.py",
    "safe_publish.py",
    "silt_v1.py",
    "source_bootstrap.py",
    "test_source_only_v1.py",
    "verify_source_v1.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def source_root(directory: Path) -> tuple[str, dict[str, str]]:
    require(directory.is_absolute(), "absolute producer directory")
    observed = {entry.name for entry in os.scandir(directory)}
    require(observed == set(ROOT_FILES), "exact source member set")
    packets: dict[str, bytes] = {}
    member_hashes: dict[str, str] = {}
    for name in sorted(ROOT_FILES):
        path = directory / name
        metadata = os.lstat(path)
        require(stat.S_ISREG(metadata.st_mode), f"regular source member: {name}")
        with path.open("rb") as handle:
            packet = handle.read()
        require(len(packet) == metadata.st_size, f"stable source member: {name}")
        packets[name] = packet
        member_hashes[name] = hashlib.sha256(packet).hexdigest()
    hasher = hashlib.sha256()
    hasher.update(b"SILT-V1-SOURCE-ROOT\0")
    for name in sorted(ROOT_FILES):
        encoded = name.encode("utf-8")
        packet = packets[name]
        hasher.update(len(encoded).to_bytes(4, "big"))
        hasher.update(encoded)
        hasher.update(len(packet).to_bytes(8, "big"))
        hasher.update(hashlib.sha256(packet).digest())
    return hasher.hexdigest(), member_hashes


def frame_with_recomputed_crc(sm: object, packet: bytes, expert: int, mutate: object) -> bytes:
    """Return a container whose chosen frame was mutated and locally rehashed."""

    parsed = sm.parse_container(packet)
    entry = parsed.entries[expert]
    frame = bytearray(packet[entry.offset : entry.offset + entry.padded_bytes])
    fields = list(sm.FRAME_STRUCT.unpack(frame[: sm.FRAME_STRUCT.size]))
    body = bytearray(frame[sm.FRAME_HEADER_BYTES : int(fields[11])])
    mutate(fields, body)
    fields[14] = hashlib.sha256(body).digest()
    fields[15] = 0
    raw = sm.FRAME_STRUCT.pack(*fields)
    zero = raw + bytes(sm.FRAME_HEADER_BYTES - len(raw))
    fields[15] = zlib.crc32(zero) & 0xFFFFFFFF
    raw = sm.FRAME_STRUCT.pack(*fields)
    logical = int(fields[11])
    padded = int(fields[12])
    rebuilt = raw + bytes(sm.FRAME_HEADER_BYTES - len(raw)) + body + bytes(padded - logical)
    require(len(rebuilt) == entry.padded_bytes, "mutated frame length")
    output = bytearray(packet)
    output[entry.offset : entry.offset + entry.padded_bytes] = rebuilt
    return bytes(output)


def arithmetic_fuzz(sm: object, independent: object, np: object) -> dict[str, object]:
    rng = np.random.default_rng(0x51A7)
    cases = 0
    symbols = 0
    for alphabet in (2, 4):
        for case in range(160):
            raw = rng.integers(0, 1 << 20, size=alphabet, dtype=np.int64)
            row_array = sm._q16_row(raw)
            row = tuple(int(value) for value in row_array)
            length = 1 + (case * 37) % 521
            sequence = tuple(int(value) for value in rng.integers(0, alphabet, size=length, dtype=np.uint8))
            first = sm.ArithmeticEncoder()
            second = independent.Encoder()
            for value in sequence:
                first.write(value, row)
                second.symbol(value, row)
            first_packet, first_bits = first.finish()
            second_packet, second_bits = second.finish()
            require((first_packet, first_bits) == (second_packet, second_bits), "independent encoder equality")
            first_decoder = sm.ArithmeticDecoder(first_packet, first_bits)
            second_decoder = independent.Decoder(second_packet, second_bits)
            first_values = tuple(first_decoder.read(row) for _ in sequence)
            second_values = tuple(second_decoder.symbol(row) for _ in sequence)
            require(first_values == second_values == sequence, "arithmetic fuzz roundtrip")
            require(first_decoder.reader.position == second_decoder.reader.position == first_bits, "exact meaningful exhaustion")
            cases += 1
            symbols += length
    return {"cases": cases, "symbols": symbols, "status": "PASS"}


def selector_exhaustion(sm: object, np: object) -> dict[str, object]:
    rows: dict[str, object] = {}
    for alphabet, selector_ids in ((2, range(6)), (4, range(8))):
        maps: list[tuple[tuple[int, int], ...]] = []
        for selector in selector_ids:
            mapping: list[tuple[int, int]] = []
            for left in range(alphabet):
                for right in range(alphabet):
                    leaves = np.asarray([[left, right]], dtype=np.uint8)
                    lifted = sm.lift_forward(leaves, alphabet, [0, 1], [selector])
                    rebuilt = sm.lift_inverse(lifted, 2, alphabet, [0, 1], [selector])
                    require(np.array_equal(leaves, rebuilt), "selector inverse")
                    mapping.append((int(lifted.roots[0]), int(lifted.detail_levels[0][0, 0])))
            require(len(set(mapping)) == alphabet * alphabet, "selector bijection")
            maps.append(tuple(mapping))
        require(len(set(maps)) == len(maps), "selector IDs distinct")
        rows[str(alphabet)] = {"canonical_ids": list(selector_ids), "distinct_maps": len(set(maps))}
    for alias in (6, 7):
        try:
            sm.pack_selectors([alias], 2)
        except sm.FormatError:
            pass
        else:
            raise AssertionError("GF2 alias accepted")
    return {"status": "PASS", "alphabets": rows}


def bounds_probe(sm: object) -> dict[str, object]:
    timings: list[dict[str, object]] = []
    for value in (0, 257, (1 << 32) - 1):
        started = time.perf_counter()
        try:
            sm.validate_expert_count(value)
        except sm.FormatError:
            elapsed = time.perf_counter() - started
            require(elapsed < 0.05, "expert cap prompt rejection")
            timings.append({"value": value, "seconds": elapsed})
        else:
            raise AssertionError("expert cap accepted")
    started = time.perf_counter()
    try:
        sm.permutation_byte_count(sm.MAX_LANES + 1)
    except sm.FormatError:
        lane_seconds = time.perf_counter() - started
        require(lane_seconds < 0.05, "lane cap before factorial")
    else:
        raise AssertionError("lane cap accepted")
    return {"status": "PASS", "expert_rejections": timings, "lane_rejection_seconds": lane_seconds}


def owner_ledger_checks(sm: object) -> dict[str, object]:
    unequal = sm.layout_cold_ledger(8192, [4096] + [8192] * 7)
    row = unequal["cold"][0]
    require(
        (row["cold_amplification_numerator"], row["cold_amplification_denominator"]) == (12, 5),
        "owner-aware unequal-frame counterexample",
    )
    require(not row["cold_below_two_by_integer_cross_multiplication"], "12/5 must fail")
    boundary = sm.layout_cold_ledger(8192, [4096] * 4)
    require(
        all(
            item["cold_amplification_numerator"] == 2
            and item["cold_amplification_denominator"] == 1
            and not item["cold_below_two_by_integer_cross_multiplication"]
            for item in boundary["cold"]
        ),
        "strict two boundary",
    )
    return {
        "status": "PASS",
        "unequal_expert_zero": row,
        "strict_two_rejected_for_all_four": True,
        "unequal_owner_sum_equals_container": unequal["owner_sum_equals_container"],
    }


def canonicality_mutations(sm: object, independent: object) -> dict[str, object]:
    roots, details = sm.generate_transformed_source(2, 256, 17, 0xCA11, True)
    model = sm.fit_model(2, roots, details)
    permutation = sm.deterministic_permutation(17, 0xCA12)
    selectors = sm.deterministic_selectors(17, 2, 0xCA13)
    leaves = sm.synthesize_leaves(2, 128, 17, 0xCA14, True, permutation, selectors)
    packet = sm.build_container(model, [sm.ExpertInput.create(leaves, permutation, selectors)])
    parsed = sm.parse_container(packet)
    info = sm.parse_frame_header(parsed.frame_view(0))

    def shortened(fields: list[object], body: bytearray) -> None:
        value = int(fields[10]) - 1
        fields[10] = value
        payload_start = int(fields[7]) + int(fields[8])
        for bit in range(value, int(fields[9]) * 8):
            body[payload_start + bit // 8] &= ~(1 << (7 - (bit & 7)))

    def nonzero_guard(fields: list[object], body: bytearray) -> None:
        payload_start = int(fields[7]) + int(fields[8])
        bit = int(fields[10]) - 1
        body[payload_start + bit // 8] |= 1 << (7 - (bit & 7))

    def gf2_alias(fields: list[object], body: bytearray) -> None:
        selector_start = int(fields[7])
        body[selector_start] = (body[selector_start] & 0x1F) | (6 << 5)

    rows: dict[str, object] = {}
    for name, mutation in (
        ("meaningful_bits_minus_one", shortened),
        ("nonzero_guard", nonzero_guard),
        ("gf2_alias_6", gf2_alias),
    ):
        forged = frame_with_recomputed_crc(sm, packet, 0, mutation)
        failures: dict[str, str] = {}
        for decoder_name, decoder in (
            ("producer", lambda: sm.decode_container(forged)),
            ("independent", lambda: independent.verify_decode_reencode(forged)),
        ):
            try:
                decoder()
            except (sm.FormatError, independent.IndependentFormatError) as exc:
                failures[decoder_name] = str(exc)
            else:
                raise AssertionError(f"{decoder_name} accepted {name}")
        rows[name] = failures
    return {
        "status": "PASS_REJECTED_ALL",
        "original_meaningful_bits": info.meaningful_bits,
        "mutations": rows,
    }


def root_hostility(source_bootstrap: object, producer: Path) -> dict[str, object]:
    rows: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="silt-v1-independent-root-") as temporary:
        copied = Path(temporary) / "source"
        copied.mkdir()
        for name in ROOT_FILES:
            shutil.copy2(producer / name, copied / name)
        (copied / "extra.py").write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
        try:
            source_bootstrap.read_source_tree(str(copied))
        except source_bootstrap.BootstrapError as exc:
            rows["extra_member"] = str(exc)
        else:
            raise AssertionError("extra source member accepted")
        (copied / "extra.py").unlink()
        (copied / "README.md").unlink()
        os.symlink(producer / "README.md", copied / "README.md")
        try:
            source_bootstrap.read_source_tree(str(copied))
        except source_bootstrap.BootstrapError as exc:
            rows["symlink_member"] = str(exc)
        else:
            raise AssertionError("symlink source member accepted")
    return {"status": "PASS_REJECTED_ALL", "cases": rows}


def make_large_equal_container(sm: object, np: object) -> tuple[bytes, list[object]]:
    alphabet = 2
    roots, details = sm.generate_transformed_source(alphabet, 2048, 17, 9001, True)
    model = sm.fit_model(alphabet, roots, details)
    sources: list[object] = []
    for expert in range(3):
        lanes = 17
        vectors = 3000
        permutation = sm.deterministic_permutation(lanes, 12000 + expert)
        selectors = sm.deterministic_selectors(lanes, alphabet, 17000 + expert)
        leaves = sm.synthesize_leaves(
            alphabet,
            vectors,
            lanes,
            22000 + expert,
            True,
            permutation,
            selectors,
        )
        sources.append(sm.ExpertInput.create(leaves, permutation, selectors))
    return sm.build_container(model, sources), sources


def cold_path_counterexample(sm: object, np: object) -> dict[str, object]:
    packet, sources = make_large_equal_container(sm, np)
    parsed = sm.parse_container(packet)
    require(parsed.expert_count == 3, "counterexample expert count")
    frame_bytes = [entry.padded_bytes for entry in parsed.entries]
    require(len(set(frame_bytes)) == 1, "counterexample equal frames")
    layout = sm.layout_cold_ledger(parsed.frames_offset, frame_bytes)
    require(layout["all_cold_below_two"], "intended page ledger must pass")
    selected = 0
    selected_row = layout["cold"][selected]
    owner = Fraction(
        int(selected_row["owner_share_numerator"]),
        int(selected_row["owner_share_denominator"]),
    )
    actual_materialized_amplification = Fraction(len(packet), 1) / owner
    require(actual_materialized_amplification == 3, "ordinary bytes API materializes the full equal-frame container")

    unrelated = parsed.entries[-1]
    mutation_offset = unrelated.offset + unrelated.padded_bytes - 1
    selected_ranges_end = parsed.entries[selected].offset + parsed.entries[selected].padded_bytes
    require(mutation_offset >= selected_ranges_end, "mutation outside selected intended page union")
    forged = bytearray(packet)
    forged[mutation_offset] ^= 1
    forged_packet = bytes(forged)
    require(
        forged_packet[: parsed.frames_offset] == packet[: parsed.frames_offset]
        and forged_packet[
            parsed.entries[selected].offset : parsed.entries[selected].offset + parsed.entries[selected].padded_bytes
        ]
        == packet[
            parsed.entries[selected].offset : parsed.entries[selected].offset + parsed.entries[selected].padded_bytes
        ],
        "selected claimed pages unchanged",
    )
    failures: dict[str, str] = {}
    for name, operation in (
        ("decode_expert", lambda: sm.decode_expert(forged_packet, selected)),
        ("trace_expert_cold_pages", lambda: sm.trace_expert_cold_pages(forged_packet, selected)),
    ):
        try:
            operation()
        except sm.FormatError as exc:
            failures[name] = str(exc)
        else:
            raise AssertionError(f"{name} did not depend on the unselected frame")
    require(np.array_equal(sm.decode_expert(packet, selected), sources[selected].leaves), "unmodified selected decode")
    return {
        "status": "BLOCK_CONFIRMED",
        "container_bytes": len(packet),
        "frames_offset": parsed.frames_offset,
        "frame_bytes": frame_bytes,
        "container_pages": len(packet) // sm.PAGE_BYTES,
        "claimed_selected_page_indices": list(
            range(parsed.frames_offset // sm.PAGE_BYTES)
        )
        + list(
            range(
                parsed.entries[selected].offset // sm.PAGE_BYTES,
                (parsed.entries[selected].offset + parsed.entries[selected].padded_bytes) // sm.PAGE_BYTES,
            )
        ),
        "actual_ordinary_parse_page_indices": list(range(len(packet) // sm.PAGE_BYTES)),
        "owner_share_bytes": {
            "numerator": owner.numerator,
            "denominator": owner.denominator,
            "float": float(owner),
        },
        "claimed_cold_bytes": int(selected_row["cold_bytes"]),
        "actual_ordinary_parse_bytes": len(packet),
        "claimed_cold_amplification": {
            "numerator": selected_row["cold_amplification_numerator"],
            "denominator": selected_row["cold_amplification_denominator"],
            "float": selected_row["cold_amplification_float"],
        },
        "actual_full_bytes_api_amplification": {
            "numerator": actual_materialized_amplification.numerator,
            "denominator": actual_materialized_amplification.denominator,
            "float": float(actual_materialized_amplification),
        },
        "unselected_mutation_offset": mutation_offset,
        "selected_claimed_pages_unchanged": True,
        "ordinary_paths_rejected_unselected_mutation": failures,
        "finding": "the instrumented range list is not the ordinary routed decoder access path",
    }


def publication_counterexamples(safe_publish: object, root: str) -> dict[str, object]:
    rows: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="silt-v1-independent-publication-") as parent:
        output = os.path.join(parent, "postrename")
        snapshots: list[dict[str, object]] = []

        class CapturingPublisher(safe_publish.SafePublisher):
            def _checkpoint(self, stage: str) -> None:
                if stage == "published_and_parent_fsynced":
                    final_stat = os.stat(output, follow_symlinks=False)
                    staging_stat = os.fstat(self.staging_fd)
                    members = sorted(os.listdir(output))
                    snapshots.append(
                        {
                            "stage": stage,
                            "parent_members": sorted(os.listdir(parent)),
                            "final_members": members,
                            "complete_present": "COMPLETE" in members,
                            "final_inode": int(final_stat.st_ino),
                            "staging_fd_inode": int(staging_stat.st_ino),
                            "same_inode_after_rename": final_stat.st_ino == staging_stat.st_ino,
                        }
                    )
                super()._checkpoint(stage)

        exception = ""
        try:
            with CapturingPublisher(output, root, fault_after="published_and_parent_fsynced") as publisher:
                publisher.write("a.bin", b"independent-audit")
                publisher.finish()
        except Exception as exc:  # the cleanup bug may mask the injected error
            exception = f"{type(exc).__name__}: {exc}"
        require(os.path.isdir(output), "post-rename final directory is visible")
        final_names = sorted(os.listdir(output))
        require(final_names == [], "current abort emptied the visible final tree")
        rows["postrename_fault"] = {
            "status": "BLOCK_CONFIRMED",
            "exception": exception,
            "final_path_visible": True,
            "final_members": final_names,
            "complete_present": os.path.exists(os.path.join(output, "COMPLETE")),
            "durable_stage_snapshot": snapshots,
            "finding": "cleanup follows staging_fd through rename and deletes the published tree",
        }

    with tempfile.TemporaryDirectory(prefix="silt-v1-independent-constructor-") as parent:
        output = os.path.join(parent, "constructor")
        exception = ""
        try:
            safe_publish.SafePublisher(output, root, fault_after="staging_created")
        except Exception as exc:
            exception = f"{type(exc).__name__}: {exc}"
        else:
            raise AssertionError("constructor fault did not trigger")
        members = sorted(os.listdir(parent))
        staging = [name for name in members if name.startswith(".constructor.staging.")]
        require(len(staging) == 1 and not os.path.exists(output), "constructor orphan staging counterexample")
        rows["constructor_fault"] = {
            "status": "FAIL_CLOSED_BUT_ORPHANED",
            "exception": exception,
            "final_path_visible": False,
            "orphan_staging_members": staging,
        }
    return rows


def portability_limits(sm: object) -> dict[str, object]:
    qwen_shape_symbols_per_expert = 3 * 768 * 2048
    qwen_shape_symbols_128_experts = 128 * qwen_shape_symbols_per_expert
    return {
        "format_has_role_or_triplet_metadata": False,
        "float_to_label_quantizer_present": False,
        "raw_weight_mse_scorer_present": False,
        "qwen_geometry_symbols_per_expert": qwen_shape_symbols_per_expert,
        "qwen_geometry_symbols_for_128_experts": qwen_shape_symbols_128_experts,
        "max_total_symbols": sm.MAX_TOTAL_SYMBOLS,
        "single_container_accepts_that_128_expert_geometry": qwen_shape_symbols_128_experts <= sm.MAX_TOTAL_SYMBOLS,
        "source_gain_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-dir", required=True)
    parser.add_argument("--expected-root", required=True)
    arguments = parser.parse_args()
    producer = Path(arguments.producer_dir).resolve(strict=True)
    observed_root, member_hashes = source_root(producer)
    require(observed_root == arguments.expected_root.lower(), "external source root mismatch")
    sys.path.insert(0, str(producer))

    # Third-party and producer imports occur only after the external root check.
    import numpy as np

    import independent_decoder_v1 as independent
    import safe_publish
    import silt_v1 as sm
    import source_bootstrap

    result = {
        "schema": "silt-v1-independent-hostile-audit-receipt",
        "status": "BLOCK",
        "authenticated_source_root": observed_root,
        "source_member_sha256": member_hashes,
        "positive_checks": {
            "arithmetic_fuzz": arithmetic_fuzz(sm, independent, np),
            "selector_exhaustion": selector_exhaustion(sm, np),
            "bounds_probe": bounds_probe(sm),
            "owner_ledger": owner_ledger_checks(sm),
            "canonicality_mutations": canonicality_mutations(sm, independent),
            "root_hostility": root_hostility(source_bootstrap, producer),
        },
        "blocking_counterexamples": {
            "cold_path": cold_path_counterexample(sm, np),
            "publication": publication_counterexamples(safe_publish, observed_root),
        },
        "portability_limits": portability_limits(sm),
        "payload_accessed": False,
        "qwen_or_model_accessed": False,
        "manifest_created": False,
        "result_frozen": False,
        "source_gain_claim": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
