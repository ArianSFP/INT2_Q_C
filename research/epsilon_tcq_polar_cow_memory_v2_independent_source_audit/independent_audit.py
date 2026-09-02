#!/usr/bin/env python3
"""Independent no-payload audit of epsilon-TCQ compact polar memory v2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SOURCE_MANIFEST_SHA256 = "cef51a7a62619927503749ebf3a390241aa9297842480023b3ffcc5abd4cf277"
SOURCE_ROOT_SHA256 = "92c7969cddbebf255c19f1aa10869d704c68727a8562f8f47bd27dd4c3593ff4"
AUDITOR_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
AUDITOR_BYTES = 116835
CUPY_RECEIPT_SHA256 = "083979b2531066e0a81f4bec3a9afa5dd027d4cd934b6fb9ce240491fa099c14"
N = 1 << 21
D = 21
LEVELS = 6
CAP = 4 * (1 << 30)


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def aligned(value: int) -> int:
    return (value + 255) // 256 * 256


def independent_memory(beam: int) -> dict[str, int | bool]:
    require(beam in (4, 8, 16, 32), "independent beam")
    events = LEVELS * N
    symbol_bits = int(math.log2(beam)) + 1
    payload_bits = (5 * N + 1) // 2 + 64
    rows = [
        beam * N * 8,                         # explicit leaf LR
        beam * (N - 1) * 8,                   # ragged LR
        beam * (N - 1),                       # ragged mu
        beam * N,                             # lower index
        beam * N,                             # level internal/plane scratch
        (events * beam * symbol_bits + 7) // 8,
        beam * 8, beam * 4, beam * 4, beam * 8, beam * 8,
        beam * 2, beam, beam,                 # causal, active, lower handle
        1, 4, 8, 1, (LEVELS + 1) * 8,        # controller SoA
        beam * D * 4, beam * D * 2,           # coupled layer handles/refcounts
        N * 8, LEVELS * N, N, LEVELS * 64 * 8,
        2 * beam * 8, 2 * beam, 2 * beam, 2 * beam, beam, beam,
        64 * 2 * 4,
        (events + 7) // 8,
        (payload_bits + 7) // 8,
    ]
    logical = sum(rows)
    allocated = sum(aligned(value) for value in rows)
    return {"logical_peak_bytes": logical, "aligned_peak_bytes": allocated,
            "passes_4gib_cap": allocated < CAP, "buffer_rows": len(rows)}


def independent_work(beam: int) -> dict[str, int | bool]:
    require(beam in (4, 8, 16, 32), "independent work beam")
    events = LEVELS * N
    startup = int(math.log2(beam))
    candidates = 2 * (beam - 1) + 2 * beam * (events - startup)
    tape_written = 2 * beam - 2 + beam * (events - startup)
    width, log_width = 2 * beam, int(math.log2(2 * beam))
    comparators_per_round = width * log_width * (log_width + 1) // 4
    return {
        "likelihood_node_updates": LEVELS * beam * N * D,
        "partial_sum_state_writes": LEVELS * beam * (N * D // 2 + 1),
        "partial_sum_xors": LEVELS * beam * (N * (D - 2) // 2 + 1),
        "level_end_polar_xors": LEVELS * beam * N * D // 2,
        "lower_index_adds": LEVELS * beam * N,
        "branch_candidates_scored": candidates,
        "survivor_tape_symbols_written": tape_written,
        "fixed_tape_symbol_capacity": beam * events,
        "stable_bitonic_comparators": events * comparators_per_round,
        "winner_backtrace_events": events,
        "winner_replay_likelihood_node_updates": LEVELS * N * D,
        "winner_replay_partial_sum_state_writes": LEVELS * (N * D // 2 + 1),
        "winner_replay_partial_sum_xors": LEVELS * (N * (D - 2) // 2 + 1),
        "winner_replay_polar_xors": LEVELS * N * D // 2,
        "winner_replay_lower_index_adds": LEVELS * N,
        "winner_arithmetic_replay_events": events,
    }


def authenticate_source(source: Path) -> dict[str, Any]:
    manifest_path = source / "SOURCE_MANIFEST.json"
    raw = manifest_path.read_bytes()
    require(sha(raw) == SOURCE_MANIFEST_SHA256, "source manifest pin")
    manifest = json.loads(raw)
    require(manifest["source_root_sha256"] == SOURCE_ROOT_SHA256, "source root pin")
    observed = []
    for row in manifest["members"]:
        member = (source / row["name"]).read_bytes()
        require(len(member) == row["bytes"] and sha(member) == row["sha256"],
                f"source member: {row['name']}")
        observed.append({"name": row["name"], "bytes": len(member),
                         "sha256": sha(member)})
    require(sha(canonical(observed)) == SOURCE_ROOT_SHA256, "independent source root")
    require({path.name for path in source.iterdir()} ==
            {row["name"] for row in observed} | {"SOURCE_MANIFEST.json"},
            "source closure")
    return {"manifest_sha256": sha(raw), "source_root_sha256": SOURCE_ROOT_SHA256,
            "members": len(observed)}


def authenticate_decoder(auditor: Path) -> dict[str, Any]:
    raw = auditor.read_bytes()
    require(len(raw) == AUDITOR_BYTES and sha(raw) == AUDITOR_SHA256, "decoder pin")
    text = raw.decode("utf-8")
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    require({"decode_sc_level", "decode_one_block", "leaf_prior_ratios"} <= functions.keys(),
            "decoder functions")
    sc_text = ast.get_source_segment(text, functions["decode_sc_level"]) or ""
    block_text = ast.get_source_segment(text, functions["decode_one_block"]) or ""
    require("lr_reg = np.ones((n // 2, depth), dtype=np.float64)" in sc_text,
            "dense LR schedule anchor")
    require("mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)" in sc_text,
            "dense mu schedule anchor")
    require("for i0 in range(n):" in sc_text, "whole-level phase traversal")
    require("for level_index, flag in enumerate(flags):" in block_text,
            "six level-major traversal")
    require("previous += (1 << level_index) * x_bit.astype(np.int16)" in block_text,
            "six-plane 64-index accumulation")
    return {"bytes": len(raw), "sha256": sha(raw),
            "six_level_major_passes_authenticated": True,
            "coordinate_local_six_event_abi": False}


def run_source_gate(source: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(source / "run_gate.py")], cwd=source,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", env=environment)
    return json.loads(completed.stdout)


def run_source_tests(source: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(source / "test_source_only.py")], cwd=source,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", env=environment)
    transcript = completed.stdout + completed.stderr
    require("Ran 14 tests" in transcript and "OK" in transcript, "hostile source tests")
    return {"status": "PASS", "tests": 14,
            "transcript_sha256": sha(transcript.encode("utf-8"))}


def authenticate_cupy_receipt(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(sha(raw) == CUPY_RECEIPT_SHA256, "CuPy receipt pin")
    receipt = json.loads(raw)
    require(receipt["status"] ==
            "PASS_GO_MEMORY_CAPACITY_ONLY_HOLD_COMPUTE_AND_DEVICE_COW",
            "CuPy split verdict")
    require(not receipt["qwen_payload_accessed"] and
            not receipt["current_codec_payload_accessed"], "CuPy no payload")
    expected = {4: 211958272, 8: 402799616,
                16: 797065216, 32: 1610762240}
    for row in receipt["beams"]:
        beam = int(row["beam_width"])
        require(int(row["cupy_pool_total_delta_bytes"]) == expected[beam],
                f"CuPy pool total B{beam}")
        require(bool(row["passes_4gib_actual_pool"]), f"CuPy capacity B{beam}")
        require(int(row["literal_live_array_bytes"]) ==
                independent_memory(beam)["aligned_peak_bytes"], f"CuPy literal B{beam}")
    primitive = receipt["primitive_kernel"]
    require(not primitive["production_persistent_kernel_demonstrated"] and
            not primitive["q0_16_frequency_rounding_equivalence_demonstrated"],
            "primitive smoke must not promote compute")
    return {"sha256": sha(raw), "device": receipt["device"],
            "beams": sorted(expected), "maximum_pool_bytes": expected[32],
            "scope": "allocation/primitive only; not persistent SC or device COW"}


def audit(source: Path, auditor: Path, cupy_receipt: Path) -> dict[str, Any]:
    source_auth = authenticate_source(source.resolve(strict=True))
    decoder_auth = authenticate_decoder(auditor.resolve(strict=True))
    gate = run_source_gate(source)
    tests = run_source_tests(source)
    memory_rows, work_rows = [], []
    for beam in (4, 8, 16, 32):
        memory = independent_memory(beam)
        work = independent_work(beam)
        observed = next(row for row in gate["beam_table"] if row["beam_width"] == beam)
        require(observed["logical_peak_bytes"] == memory["logical_peak_bytes"],
                f"logical memory B{beam}")
        require(observed["aligned_peak_bytes"] == memory["aligned_peak_bytes"],
                f"aligned memory B{beam}")
        require(observed["likelihood_node_updates"] == work["likelihood_node_updates"],
                f"LR work B{beam}")
        require(observed["partial_sum_state_writes"] == work["partial_sum_state_writes"],
                f"mu write work B{beam}")
        require(observed["partial_sum_xors"] == work["partial_sum_xors"],
                f"mu xor work B{beam}")
        require(observed["branch_candidates_scored"] == work["branch_candidates_scored"],
                f"branch work B{beam}")
        require(observed["survivor_tape_symbols_written"] ==
                work["survivor_tape_symbols_written"], f"tape work B{beam}")
        memory_rows.append({"beam_width": beam, **memory})
        work_rows.append({"beam_width": beam, **work})
    maximum = gate["maximum_beam_work"]
    maximum_key = {
        "partial_sum_state_writes": "partial_sum_state_writes_worst_active_upper_bound",
        "partial_sum_xors": "partial_sum_xors_worst_active_upper_bound",
        "lower_index_adds": "lower_index_adds_worst_active_upper_bound",
    }
    for key, value in independent_work(32).items():
        observed_key = maximum_key.get(key, key)
        require(maximum[observed_key] == value, f"maximum work: {key}")
    require(gate["verdicts"] == {"memory": "GO_MEMORY_CAPACITY",
                                 "compute": "HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION",
                                 "payload": "HOLD_PAYLOAD"}, "split verdict")
    cupy = authenticate_cupy_receipt(cupy_receipt.resolve(strict=True))
    return {
        "schema": "epsilon-tcq-polar-cow-memory-v2-independent-source-audit",
        "status": "PASS_GO_MEMORY_CAPACITY_HOLD_COMPUTE_AND_PAYLOAD",
        "source": source_auth, "decoder": decoder_auth, "source_tests": tests,
        "memory": memory_rows, "work": work_rows, "cupy": cupy,
        "verdicts": gate["verdicts"],
        "scientific_limits": {
            "memory_is_frozen_representation_capacity_not_production_decoder": True,
            "work_counts_are_worst_active_upper_bounds_where_labelled": True,
            "device_cow_implementation_demonstrated": False,
            "persistent_full_six_level_kernel_demonstrated": False,
            "cupy_q0_16_boundary_equivalence_demonstrated": False,
        },
        "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--auditor", required=True)
    parser.add_argument("--cupy-receipt", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    receipt = audit(Path(args.source), Path(args.auditor), Path(args.cupy_receipt))
    encoded = canonical(receipt).decode("ascii")
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="ascii", newline="\n")
    print(encoded)


if __name__ == "__main__":
    main()
