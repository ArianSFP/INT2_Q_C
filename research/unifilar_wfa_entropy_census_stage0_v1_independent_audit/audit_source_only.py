#!/usr/bin/env python3
"""Independent hostile, source-only audit for sealed UWFA census v1.

This audit never discovers or opens a model, Qwen checkpoint, extracted stream,
current artifact, or Gaussian-control payload.  Every dynamic input is generated
inside a temporary directory.  The producer package is authenticated before it
is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any


EXPECTED_MANIFEST_SHA256 = "1dbea65550d879c3cc6ca81974223d251d669c15f5af17fa9681800cf03cf9ff"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def regular_leaf(path: Path) -> bytes:
    require(path.is_absolute(), f"absolute path required: {path}")
    info = os.lstat(path)
    require(not stat.S_ISLNK(info.st_mode), f"symlink leaf: {path}")
    require(stat.S_ISREG(info.st_mode), f"not regular: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        observed = os.fstat(fd)
        require(stat.S_ISREG(observed.st_mode), f"descriptor not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def authenticate_package(package: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    package = package.absolute()
    directory = os.lstat(package)
    require(stat.S_ISDIR(directory.st_mode) and not stat.S_ISLNK(directory.st_mode), "producer directory must be real")
    manifest_bytes = regular_leaf(package / "SOURCE_MANIFEST.json")
    require(sha256(manifest_bytes) == EXPECTED_MANIFEST_SHA256, "reviewed manifest hash")
    manifest = json.loads(manifest_bytes)
    require(manifest["schema"] == "unifilar-wfa-source-manifest-v1", "manifest schema")
    expected = {"SOURCE_MANIFEST.json"}
    verified: list[dict[str, Any]] = []
    for row in manifest["members"]:
        name = row["name"]
        require(name == Path(name).name and name not in expected, f"member name: {name}")
        data = regular_leaf(package / name)
        require(len(data) == row["bytes"], f"member bytes: {name}")
        require(sha256(data) == row["sha256"], f"member hash: {name}")
        expected.add(name)
        verified.append({"name": name, "bytes": len(data), "sha256": sha256(data)})
    actual = {entry.name for entry in os.scandir(package) if entry.name != "__pycache__"}
    require(actual == expected, "manifest closure")
    return manifest, verified


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def independent_context(level: int, base: int, within: int) -> int:
    return ((level * 16 + min(15, base * 16 // 65536)) * 4) + (within & 3)


def independent_transition(candidate: Any, state: int, bit: int, context: int, within: int) -> int:
    mask = candidate.states - 1
    if candidate.topology == "suffix":
        return ((state << 1) | bit) & mask
    if candidate.topology == "xor_sketch":
        if bit == 0:
            return state
        sketch = mix32(0xA511E9B3 ^ context ^ ((within * 0x9E3779B1) & 0xFFFFFFFF)) & mask
        return state ^ (sketch or 1)
    if candidate.topology == "modular_ones":
        weight = (mix32(0x63D83595 ^ context ^ ((within & 3) << 20)) & mask) | 1
        return (state + weight * bit) & mask
    if candidate.topology == "rolling_affine":
        multiplier = (5 if candidate.states >= 8 else 1) & mask
        addend = mix32(0xB5297A4D ^ context ^ ((within & 3) << 24)) & mask
        return (multiplier * state + addend + bit) & mask
    if candidate.topology == "signed_saturating":
        return min(mask, state + 1) if bit else max(0, state - 1)
    raise AssertionError("unknown topology")


def pack_bits(rows: list[int]) -> bytes:
    output = bytearray((len(rows) + 7) // 8)
    for index, bit in enumerate(rows):
        output[index >> 3] |= int(bit) << (7 - (index & 7))
    return bytes(output)


def independent_arithmetic_encode(bits: list[int], frequencies: list[int]) -> tuple[bytes, int]:
    require(len(bits) == len(frequencies) and bits, "arithmetic geometry")
    low, high = 0, (1 << 32) - 1
    half, quarter, three_quarters = 1 << 31, 1 << 30, 3 << 30
    pending = 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        output.extend([1 - bit] * pending)
        pending = 0

    for bit, f1 in zip(bits, frequencies, strict=True):
        f0 = 65536 - f1
        width = high - low + 1
        split = low + width * f0 // 65536 - 1
        require(low <= split < high, "independent split")
        if bit:
            low = split + 1
        else:
            high = split
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & 0xFFFFFFFF
            high = ((high << 1) & 0xFFFFFFFF) | 1
    pending += 1
    emit(0 if low < quarter else 1)
    return pack_bits(output), len(output)


def pseudo_stream(seed: int, length: int) -> tuple[list[int], list[int], list[int]]:
    value = seed & 0xFFFFFFFF or 1
    bits: list[int] = []
    levels: list[int] = []
    base: list[int] = []
    parity = 0
    for index in range(length):
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        value &= 0xFFFFFFFF
        threshold = 21000 if parity == 0 else 44500
        bit = int((value & 0xFFFF) < threshold)
        bits.append(bit)
        levels.append((index * 5 + (seed & 7)) % 6)
        base.append(1 + ((index * 7919 + seed * 17) % 65535))
        parity ^= bit
    return bits, levels, base


def u16_bytes(values: list[int]) -> bytes:
    packed = array("H", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


class HostResult:
    def __init__(self, values: list[int]):
        self.values = values

    def get(self) -> "HostResult":
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class CpuBackend:
    def __init__(self, common: Any):
        self.common = common

    def pack_streams(self, streams: list[tuple[bytes, bytes, bytes]]) -> list[tuple[list[int], list[int], list[int]]]:
        output = []
        for bits, levels, base_raw in streams:
            base = array("H")
            base.frombytes(base_raw)
            if sys.byteorder != "little":
                base.byteswap()
            output.append((list(bits), list(levels), [int(value) for value in base]))
        return output

    def fit_counts(self, packed: Any, topology: int, states: int, reset: int) -> HostResult:
        candidate = self.common.Candidate(self.common.TOPOLOGIES[topology], states, reset)
        counts = self.common.zero_counts(candidate)
        for bits, levels, base in packed:
            self.common.count_stream_cpu(bits, levels, base, candidate, counts)
        return HostResult(counts)

    def exact_lengths(self, packed: Any, topology: int, states: int, reset: int, frequencies: list[int]) -> HostResult:
        candidate = self.common.Candidate(self.common.TOPOLOGIES[topology], states, reset)
        return HostResult([
            self.common.exact_stream_length_cpu(bits, levels, base, candidate, frequencies)
            for bits, levels, base in packed
        ])


def source_math_tests(common: Any) -> dict[str, Any]:
    bank = common.candidate_bank()
    require(len(bank) == 150 and [row.selector_ordinal for row in bank] == list(range(150)), "candidate bank")
    transition_cases = 0
    for candidate in bank:
        positions = sorted({0, 1, 3, candidate.reset_length - 1})
        for state in range(candidate.states):
            for bit in (0, 1):
                for within in positions:
                    for level, base in ((0, 1), (2, 32768), (5, 65535)):
                        context = independent_context(level, base, within)
                        require(common.public_context(level, base, within) == context, "context law")
                        require(common.transition(candidate, state, bit, context, within) == independent_transition(candidate, state, bit, context, within), "transition law")
                        transition_cases += 1

    q16_cases = 0
    counts: list[int] = []
    for c0, c1 in ((0, 0), (1, 0), (0, 1), (7, 11), (10**6, 17), (17, 10**6)):
        counts.extend([c0, c1])
    observed = common.q16_frequencies_from_counts(counts)
    for index, (c0, c1) in enumerate(zip(counts[::2], counts[1::2], strict=True)):
        numerator = 65536 * (2 * c1 + 1)
        denominator = 2 * (c0 + c1 + 1)
        expected = min(65535, max(1, (numerator + denominator // 2) // denominator))
        require(observed[index] == expected, "Q0.16 Jeffreys")
        q16_cases += 1

    arithmetic_cases = 0
    for length in range(1, 11):
        for word in range(1 << length):
            bits = [(word >> (length - 1 - index)) & 1 for index in range(length)]
            frequencies = [1 + ((index * 7919 + length * 997) % 65535) for index in range(length)]
            expected_payload, expected_bits = independent_arithmetic_encode(bits, frequencies)
            payload, logical = common.arithmetic_encode_binary(bits, frequencies)
            require((payload, logical) == (expected_payload, expected_bits), "independent arithmetic encode")
            decoded = common.arithmetic_decode_binary(payload, logical, length, lambda index, f=frequencies: f[index])
            require(decoded == bits, "arithmetic decode")
            arithmetic_cases += 1

    candidate = common.Candidate("xor_sketch", 16, 128)
    bits, levels, base = pseudo_stream(0x51515151, 513)
    counts = common.count_stream_cpu(bits, levels, base, candidate)
    frequencies = common.q16_frequencies_from_counts(counts)
    packet = common.serialize_model(candidate, frequencies)
    header = struct.unpack("<8sHHHHII32s8s", packet[:64])
    require(header[:7] == (b"UWFAV1\x00\x00", 1, candidate.topology_id, 16, 2, 128, 384), "independent model header")
    require(struct.unpack("<H", packet[64:66])[0] == candidate.selector_ordinal, "selector")
    require(list(struct.unpack(f"<{len(frequencies)}H", packet[66:])) == frequencies, "literal Q0.16 tensor")
    recovered_candidate, recovered_frequencies = common.deserialize_model(packet)
    require(recovered_candidate == candidate and recovered_frequencies == frequencies, "model deserialize")
    payload, logical = common.encode_unifilar_stream(bits, levels, base, candidate, frequencies)
    require(common.decode_unifilar_stream(payload, logical, levels, base, candidate, frequencies) == bits, "unifilar roundtrip")

    # Exact reset-before-emission check at all reset boundaries.
    marker = [1 + (index % 65535) for index in range(common.model_frequency_count(candidate))]
    stream_bits = [1] * 257
    stream_levels = [0] * 257
    stream_base = [32768] * 257
    used = common.stream_frequencies_cpu(stream_bits, stream_levels, stream_base, candidate, marker)
    context0 = common.public_context(0, 32768, 0)
    require(used[0] == marker[context0] and used[128] == marker[context0] and used[256] == marker[context0], "t=0 reset emission")
    return {
        "candidate_cells": len(bank),
        "transition_cases": transition_cases,
        "q16_cases": q16_cases,
        "exhaustive_arithmetic_cases": arithmetic_cases,
        "actual_model_and_arithmetic_roundtrip": True,
        "reset_before_t0_emission": True,
    }


def gpu_tests(common: Any, cupy_module: Any) -> dict[str, Any]:
    import cupy as cp  # type: ignore

    backend = cupy_module.build_backend(cp)
    streams = [pseudo_stream(0xA5000000 ^ index * 0x9E3779B1, length) for index, length in enumerate((97, 257, 513, 4097))]
    packed_rows = [(bytes(bits), bytes(levels), u16_bytes(base)) for bits, levels, base in streams]
    packed = backend.pack_streams(packed_rows)
    compared = 0
    for candidate in common.candidate_bank():
        expected_counts = common.zero_counts(candidate)
        for bits, levels, base in streams:
            common.count_stream_cpu(bits, levels, base, candidate, expected_counts)
        observed_counts = [int(value) for value in backend.fit_counts(packed, candidate.topology_id, candidate.states, candidate.reset_length).get().tolist()]
        require(observed_counts == expected_counts, f"CPU/CuPy counts: {candidate}")
        frequencies = common.q16_frequencies_from_counts(expected_counts)
        expected_lengths = [common.exact_stream_length_cpu(bits, levels, base, candidate, frequencies) for bits, levels, base in streams]
        observed_lengths = [int(value) for value in backend.exact_lengths(packed, candidate.topology_id, candidate.states, candidate.reset_length, frequencies).get().tolist()]
        require(observed_lengths == expected_lengths, f"CPU/CuPy arithmetic lengths: {candidate}")
        compared += 1

    candidate = common.Candidate("xor_sketch", 64, 4096)
    timings = []
    for total in (131072, 262144, 524288):
        per = total // 16
        rows = []
        for index in range(16):
            bits, levels, base = pseudo_stream(0x76000000 ^ index, per)
            rows.append((bytes(bits), bytes(levels), u16_bytes(base)))
        item = backend.pack_streams(rows)
        backend.fit_counts(item, candidate.topology_id, candidate.states, candidate.reset_length).get()
        started = time.perf_counter()
        backend.fit_counts(item, candidate.topology_id, candidate.states, candidate.reset_length).get()
        elapsed = time.perf_counter() - started
        timings.append({"symbols": total, "seconds": elapsed})
    return {
        "status": "PASS_ALL_150_CPU_CUPY_EXACT",
        "cells_compared": compared,
        "streams": len(streams),
        "symbols_per_cell": sum(len(row[0]) for row in streams),
        "measured_count_kernel_scaling": timings,
        "core_complexity": "one sequential state update per symbol per cell: O(N) for fixed cell, O(150*N) for fixed bank",
        "full_protocol_complexity": "O(outer_folds*150*N); not globally O(N) when fold count grows",
    }


def nested_and_identity_tests(common: Any, stage: Any) -> dict[str, Any]:
    streams = []
    ordinal = 0
    for layer in range(3):
        for expert in range(3):
            bits, levels, base = pseudo_stream(0x11000000 ^ layer * 101 ^ expert * 1009, 192)
            baseline, logical = common.arithmetic_encode_binary(bits, base)
            streams.append({
                "stream_key": f"stream-{ordinal}",
                "layer_group": f"layer-{layer}",
                "expert_group": f"expert-{expert}",
                "expert_ordinal": expert,
                "weight_charge": len(bits),
                "symbols": len(bits),
                "bits": bits,
                "levels": levels,
                "base": base,
                "bits_bytes": bytes(bits),
                "levels_bytes": bytes(levels),
                "base_bytes": u16_bytes(base),
                "baseline_payload_bytes": len(baseline),
                "baseline_logical_bits": logical,
            })
            ordinal += 1
    panel = {
        "weights": sum(row["weight_charge"] for row in streams),
        "current_object_bytes": 100000,
        "immutable_global_bytes": 0,
        "experts": {0: 0, 1: 0, 2: 0},
        "streams": streams,
    }
    backend = CpuBackend(common)
    first = stage.nested_holdout(common, backend, panel)
    second = stage.nested_holdout(common, backend, panel)
    require(first == second, "nested determinism")
    require(len(first["folds"]) == 9, "outer folds")
    require(sum(row["test_weight_charge"] for row in first["folds"]) == panel["weights"], "fold partition")
    for fold in first["folds"]:
        require(fold["development_stream_count"] == 4, "layer-and-expert exclusion")
        require(fold["inner_train_stream_count"] == 3 and fold["inner_validation_stream_count"] == 1, "ranked inner split")

    original = stage.packed_rows(streams)
    relabelled = []
    for index, row in enumerate(streams):
        clone = dict(row)
        clone["layer_group"] = f"forbidden-layer-key-{index}"
        clone["expert_group"] = f"forbidden-expert-key-{index}"
        clone["stream_key"] = f"forbidden-stream-key-{index}"
        relabelled.append(clone)
    require(stage.packed_rows(relabelled) == original, "identity metadata entered probability input")
    return {
        "nested_folds": len(first["folds"]),
        "deterministic_replay": True,
        "development_excludes_same_layer_or_expert": True,
        "identity_relabel_probability_input_invariant": True,
        "selection_only_uses_inner_validation": True,
    }


def make_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.absolute()), "bytes": len(data), "sha256": sha256(data)}


def synthetic_lock_test(common: Any, stage: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).absolute()
        artifact = root / "artifact.bin"
        extraction = root / "extraction.json"
        artifact.write_bytes(b"thirteen-byte")
        extraction.write_bytes(b"{}")
        bits, levels, base = pseudo_stream(0x12345678, 64)
        payload, logical = common.arithmetic_encode_binary(bits, base)
        paths = {}
        for name, data in {
            "bits.bin": bytes(bits),
            "levels.bin": bytes(levels),
            "base.bin": u16_bytes(base),
            "payload.bin": payload,
        }.items():
            path = root / name
            path.write_bytes(data)
            paths[name] = path
        row = {
            "stream_key": "synthetic-only",
            "layer_group": "layer-synthetic",
            "expert_group": "expert-synthetic",
            "expert_ordinal": 0,
            "weight_charge": 64,
            "symbols": 64,
            "original_logical_bits": logical,
            "selected_bits_u8": make_ref(paths["bits.bin"]),
            "polar_level_u8": make_ref(paths["levels.bin"]),
            "regenerated_base_freq1_u16le": make_ref(paths["base.bin"]),
            "original_arithmetic_payload": make_ref(paths["payload.bin"]),
        }
        # Deliberately unrelated to the 13-byte artifact.  The loader accepts it.
        lock = common.seal_record({
            "schema": common.STREAM_LOCK_SCHEMA,
            "weights": 64,
            "current_object_bytes": 10_000_000,
            "immutable_global_bytes": 0,
            "current_artifact": make_ref(artifact),
            "extraction_receipt": make_ref(extraction),
            "experts": [{"expert_ordinal": 0, "immutable_local_bytes": 0}],
            "streams": [row],
        }, "lock_sha256")
        lock_path = root / "stream_lock.json"
        lock_path.write_bytes(common.pretty_json(lock))
        panel = stage.load_panel(common, lock_path)
        try:
            require(panel["baseline_replayed_before_candidate"], "baseline replay")
            require(panel["current_object_bytes"] == 10_000_000, "inflated current object not accepted")
            accepted_mismatch = panel["current_object_bytes"] != artifact.stat().st_size
        finally:
            panel["held"].close()
        require(accepted_mismatch, "expected unbound current_object_bytes defect")

        # Self-consistent hashes cannot hide an invalid arithmetic baseline.
        bad_payload = bytearray(payload)
        bad_payload[0] ^= 0x80
        paths["payload.bin"].write_bytes(bytes(bad_payload))
        bad_row = dict(row)
        bad_row["original_arithmetic_payload"] = make_ref(paths["payload.bin"])
        bad_lock = common.seal_record({
            "schema": common.STREAM_LOCK_SCHEMA,
            "weights": 64,
            "current_object_bytes": 100,
            "immutable_global_bytes": 0,
            "current_artifact": make_ref(artifact),
            "extraction_receipt": make_ref(extraction),
            "experts": [{"expert_ordinal": 0, "immutable_local_bytes": 0}],
            "streams": [bad_row],
        }, "lock_sha256")
        bad_lock_path = root / "bad_lock.json"
        bad_lock_path.write_bytes(common.pretty_json(bad_lock))
        rejected = False
        try:
            stage.load_panel(common, bad_lock_path)
        except Exception:
            rejected = True
        require(rejected, "invalid current baseline replay accepted")
    return {
        "valid_synthetic_baseline_exactly_replayed": True,
        "tampered_synthetic_baseline_rejected": True,
        "blocking_unbound_current_object_bytes_accepted": True,
    }


def final_packet_binding_test(common: Any, stage: Any) -> dict[str, Any]:
    bits, levels, base = pseudo_stream(0xCAFEBABE, 257)
    baseline, baseline_bits = common.arithmetic_encode_binary(bits, base)
    stream = {
        "stream_key": "synthetic-final",
        "layer_group": "layer",
        "expert_group": "expert",
        "expert_ordinal": 0,
        "weight_charge": 257,
        "symbols": 257,
        "bits": bits,
        "levels": levels,
        "base": base,
        "bits_bytes": bytes(bits),
        "levels_bytes": bytes(levels),
        "base_bytes": u16_bytes(base),
        "baseline_payload_bytes": len(baseline),
        "baseline_logical_bits": baseline_bits,
    }
    panel = {
        "weights": 257,
        "current_object_bytes": 100000,
        "immutable_global_bytes": 0,
        "experts": {0: 0},
        "streams": [stream],
    }
    candidate = common.Candidate("xor_sketch", 2, 128)
    backend = CpuBackend(common)
    genuine = stage.final_packet(common, backend, panel, candidate, emit_payload=True)
    model = genuine["_model_packet"]
    recovered_candidate, recovered_frequencies = common.deserialize_model(model)
    payload = genuine["_payload_packet"]
    logical = genuine["streams"][0]["logical_bits"]
    decoded = common.decode_unifilar_stream(payload, logical, levels, base, recovered_candidate, recovered_frequencies)
    require(decoded == bits, "serialized model/payload independent roundtrip")

    # Prove the producer's acceptance path does not consume the serialized model.
    original_serializer = common.serialize_model
    expected_size = common.model_ledger(candidate)["physical_model_bytes"]
    common.serialize_model = lambda _candidate, _frequencies: b"\xFF" * expected_size
    try:
        corrupted = stage.final_packet(common, backend, panel, candidate, emit_payload=True)
    finally:
        common.serialize_model = original_serializer
    require(corrupted["_model_packet"] == b"\xFF" * expected_size, "corruption injection")
    require(corrupted["streams"][0]["logical_bits"] > 0, "corrupt serialized model should not be accepted")
    emitted_keys = {"final_model.bin", "final_arithmetic_payloads.bin", "result.json"}
    return {
        "genuine_serialized_roundtrip": True,
        "blocking_acceptance_ignores_corrupted_serialized_model": True,
        "blocking_no_literal_full_container": True,
        "producer_emitted_members_before_completion": sorted(emitted_keys),
        "ledger_only_components": ["global header", "expert headers", "directory", "immutable state", "rate-floor padding", "page placement"],
    }


def lifecycle_tests(package: Path, common: Any, stage: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).absolute()
        output = root / "wrong-token-output"
        absent = root / "absent.json"
        command = [
            sys.executable, "-B", "-I", str(package / "stage0_census.py"),
            "--authorization", "WRONG", "--review-receipt", str(absent),
            "--stream-lock", str(absent), "--gaussian-control-lock", str(absent),
            "--output", str(output),
        ]
        run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        require(run.returncode == 2 and not output.exists(), "wrong-token lifecycle")

        real = root / "real"
        real.mkdir()
        target = real / "file.bin"
        target.write_bytes(b"synthetic")
        alias = root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        ancestor_accepted_bootstrap = stage.read_regular_leaf(alias / "file.bin") == b"synthetic"
        with common.HeldRegularFile((alias / "file.bin").absolute()) as held:
            ancestor_accepted_held = held.read_all() == b"synthetic"
        require(ancestor_accepted_bootstrap and ancestor_accepted_held, "expected ancestor symlink gap")

        clone = root / "clone"
        shutil.copytree(package, clone)
        clone_stage = load_module("uwfa_audit_clone_stage", clone / "stage0_census.py")
        manifest_sha = sha256((clone / "SOURCE_MANIFEST.json").read_bytes())
        review = {
            "schema": clone_stage.REVIEW_SCHEMA_BOOTSTRAP,
            "status": "PASS_INDEPENDENT_SOURCE_REVIEW",
            "payload_authority_granted": True,
            "authorization_token": clone_stage.AUTHORIZATION_BOOTSTRAP,
            "reviewed_source_manifest_sha256": manifest_sha,
        }
        review["review_sha256"] = sha256(canonical_json(review))
        review_path = root / "synthetic-review.json"
        review_path.write_bytes(canonical_json(review))
        returned_package, _, _, _ = clone_stage.bootstrap_source(review_path)
        require(returned_package == clone.absolute(), "clone bootstrap")
        (clone / "uwfa_common.py").write_text("TOCTOU_MARKER = 'executed-after-authentication'\n", encoding="utf-8")
        tampered = clone_stage.load_module("uwfa_audit_toctou_marker", clone / "uwfa_common.py")
        source_toctou_executed = getattr(tampered, "TOCTOU_MARKER", None) == "executed-after-authentication"
        require(source_toctou_executed, "expected source TOCTOU gap")
    return {
        "wrong_token_no_output": True,
        "leaf_symlink_rejection_present": True,
        "blocking_symlink_ancestor_accepted": True,
        "blocking_authenticated_source_reopened_by_path_and_replaceable": True,
    }


def static_control_and_status_tests(stage_path: Path) -> dict[str, Any]:
    source = stage_path.read_text(encoding="utf-8")
    load_loop = source.index("for seed, lock_path, ref in control_rows:")
    fit_loop = source.index("for seed, panel in zip(common.CONTROL_SEEDS, control_panels, strict=True):")
    require(load_loop < fit_loop, "control load-before-fit ordering")
    require(source.index("panel = load_panel(common, lock_path)", load_loop) < fit_loop, "baseline replay before control fit")
    required_status = [
        "HARD_KILL_EXACT_PHYSICAL_SOURCE_PACKET",
        "NO_PROMOTION_NESTED_HELDOUT_THRESHOLD_FAIL",
        "NO_PROMOTION_GAUSSIAN_SPECIFICITY_FAIL",
        "PROMOTE_FINITE_UNIVERSAL_CELL",
    ]
    require(all(value in source for value in required_status), "status lattice")
    # There is no source/control geometry equality assertion in the control path.
    control_region = source[load_loop:source.index("source_saving =", fit_loop)]
    geometry_terms = ("panel[\"weights\"] == source_panel", "panel[\"experts\"] == source_panel", "panel[\"streams\"] == source_panel")
    geometry_missing = not any(term in control_region for term in geometry_terms)
    require(geometry_missing, "expected missing matched-control enforcement")
    return {
        "all_eight_control_panels_loaded_and_baselines_replayed_before_first_fit_by_control_flow": True,
        "status_precedence": required_status,
        "positive_status_requires_physical_then_heldout_then_specificity": True,
        "blocking_matched_control_geometry_not_enforced": True,
    }


def accounting_tests(common: Any) -> dict[str, Any]:
    candidate = common.Candidate("xor_sketch", 64, 4096)
    model = common.model_ledger(candidate)
    require(model["physical_model_bytes"] == 64 + 2 + 2 * 64 * 384, "model bytes")
    require(model["cold_model_bytes"] == math.ceil(model["physical_model_bytes"] / 4096) * 4096, "model pages")
    payloads = [[1000, 2000], [3000], [4000, 5000, 6000]]
    immutable = [111, 222, 333]
    weights = 3 * 100000
    current = 100000
    ledger = common.packet_ledger(
        weights=weights,
        current_object_bytes=current,
        immutable_global_bytes=777,
        immutable_local_bytes=immutable,
        model_packet_bytes=model["physical_model_bytes"],
        stream_payload_bytes=payloads,
    )
    raw_global = 256 + 777 + model["physical_model_bytes"] + 64 * sum(len(row) for row in payloads)
    require(ledger["raw_global_bytes"] == raw_global, "global ledger")
    require(ledger["global_bytes_after_alignment"] == math.ceil(raw_global / 4096) * 4096, "global alignment")
    prepad_local = [512 + imm + sum(row) for imm, row in zip(immutable, payloads, strict=True)]
    require(sum(ledger["local_frame_bytes"]) >= sum(prepad_local), "local ledger")
    for frame, row in zip(ledger["local_frame_bytes"], ledger["cold_rows"], strict=True):
        brute_worst = max(math.ceil((offset + frame) / 4096) for offset in range(4096))
        require(row["worst_unaligned_local_pages"] == brute_worst, "worst-page formula")
    expected_saving = 8 * (current - ledger["total_bytes"]) / weights
    require(abs(ledger["net_physical_saving_bpw"] - expected_saving) < 1e-15, "saving bpw")
    expected_f = common.CURRENT_FINITE_F * 2 ** (-2 * expected_saving)
    require(abs(ledger["F_from_unchanged_current_reconstruction"] - expected_f) < 1e-15, "F conversion")
    return {
        "model_bytes_exact": True,
        "q016_tensor_bytes_charged": model["tensor_bytes"],
        "global_alignment_and_worst_unaligned_local_pages_exact": True,
        "rate_floor_and_F_conversion_exact": True,
        "blocking_ledger_not_bound_to_emitted_full_container": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--with-cupy", action="store_true")
    args = parser.parse_args()
    package = Path(args.package).absolute()

    manifest, verified = authenticate_package(package)
    # Project imports happen only after the independent closure above.
    common = load_module("uwfa_independent_audit_common", package / "uwfa_common.py")
    stage = load_module("uwfa_independent_audit_stage", package / "stage0_census.py")
    fixture = load_module("uwfa_independent_audit_fixture", package / "fixture_long_memory.py")
    cupy_module = load_module("uwfa_independent_audit_cupy_source", package / "cupy_backend.py")

    results: dict[str, Any] = {
        "schema": "unifilar-wfa-entropy-census-independent-hostile-tests-v1",
        "reviewed_source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_closure": {"status": "PASS", "members": verified},
        "source_math": source_math_tests(common),
        "nested_holdout_and_identity": nested_and_identity_tests(common, stage),
        "synthetic_lock_and_baseline": synthetic_lock_test(common, stage),
        "final_packet_binding": final_packet_binding_test(common, stage),
        "lifecycle": lifecycle_tests(package, common, stage),
        "controls_and_status": static_control_and_status_tests(package / "stage0_census.py"),
        "accounting": accounting_tests(common),
    }
    long_memory = fixture.run_fixture()
    require(long_memory["status"] == "PASS_LONG_MEMORY_SEPARATION", "long-memory fixture")
    results["long_memory_fixture"] = long_memory
    if args.with_cupy:
        results["cpu_cupy"] = gpu_tests(common, cupy_module)
    else:
        results["cpu_cupy"] = {"status": "NOT_RUN"}

    blockers = [
        "AUTHENTICATED_SOURCE_TOCTOU",
        "SYMLINK_ANCESTOR_NOT_PINNED",
        "CURRENT_OBJECT_BYTES_NOT_BOUND_TO_ARTIFACT",
        "SERIALIZED_MODEL_NOT_USED_BY_FINAL_ACCEPTANCE_DECODE",
        "NO_LITERAL_FULL_PHYSICAL_CONTAINER",
        "MATCHED_GAUSSIAN_CONTROL_GEOMETRY_NOT_ENFORCED",
    ]
    results["status"] = "BLOCK_SOURCE_REVIEW"
    results["blocking_findings"] = blockers
    results["payload_authority_granted"] = False
    results["no_payload_access_attestation"] = {
        "model_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "current_finite_artifact_or_selected_stream_opened_statted_hashed_or_enumerated": False,
        "extracted_selected_stream_opened_statted_hashed_or_enumerated": False,
        "gaussian_control_payload_opened_statted_hashed_or_enumerated": False,
        "all_dynamic_inputs": "generated in temporary directories or deterministic memory",
    }
    print(json.dumps(results, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
