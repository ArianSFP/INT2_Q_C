#!/usr/bin/env python3
"""Authenticated synthetic-only v1 replay with safe unsealed publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys


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


def authenticate(directory: str) -> str:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    packets: dict[str, bytes] = {}
    try:
        if set(os.listdir(descriptor)) != set(ROOT_FILES):
            raise RuntimeError("authenticated file set")
        for name in sorted(ROOT_FILES):
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("authenticated regular source members")
            file_descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                pieces: list[bytes] = []
                while True:
                    chunk = os.read(file_descriptor, 1 << 20)
                    if not chunk:
                        break
                    pieces.append(chunk)
                packets[name] = b"".join(pieces)
                if len(packets[name]) != metadata.st_size:
                    raise RuntimeError("source changed during authentication")
            finally:
                os.close(file_descriptor)
    finally:
        os.close(descriptor)
    hasher = hashlib.sha256()
    hasher.update(b"SILT-V1-SOURCE-ROOT\0")
    for name in sorted(ROOT_FILES):
        encoded = name.encode()
        packet = packets[name]
        hasher.update(len(encoded).to_bytes(4, "big"))
        hasher.update(encoded)
        hasher.update(len(packet).to_bytes(8, "big"))
        hasher.update(hashlib.sha256(packet).digest())
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authenticated-root", required=True)
    parser.add_argument("--output-path", required=True)
    arguments = parser.parse_args()
    source_dir = os.path.dirname(os.path.abspath(__file__))
    observed_root = authenticate(source_dir)
    if observed_root != arguments.authenticated_root.lower():
        raise RuntimeError("source root changed before imports")

    sys.path.insert(0, source_dir)
    import numpy as np

    from cupy_backend_v1 import search_metadata_cupy
    from independent_decoder_v1 import verify_decode_reencode
    from safe_publish import SafePublisher
    from silt_v1 import (
        ExpertInput,
        build_container,
        deterministic_permutation,
        deterministic_selectors,
        fit_model,
        flatten_details,
        leaf_digest,
        lift_forward,
        physical_ledger,
        sha256_bytes,
        synthesize_leaves,
    )

    hidden = 0x6C31
    candidates = (0x0B51, 0x193D, 0x2E71, hidden, 0x79A3, 0x8849, 0xA117, 0xD20B)
    lanes = 97
    rows: dict[str, object] = {}
    packets: dict[str, bytes] = {}
    for alphabet in (2, 4):
        hidden_permutation = deterministic_permutation(lanes, hidden)
        hidden_selectors = deterministic_selectors(lanes, alphabet, hidden ^ 0x5A17)
        for structured in (True, False):
            tag = f"a{alphabet}_{'structured' if structured else 'control'}"
            base = alphabet * 10_000_000 + int(structured) * 1_000_000
            search_train = synthesize_leaves(alphabet, 2048, lanes, base + 101, structured, hidden_permutation, hidden_selectors)
            search_validation = synthesize_leaves(alphabet, 1024, lanes, base + 202, structured, hidden_permutation, hidden_selectors)
            search = search_metadata_cupy(search_train, search_validation, alphabet, candidates, require_rtx_5090=True)
            permutation = list(search.selected_permutation)
            selectors = list(search.selected_selectors)
            model_leaves = synthesize_leaves(alphabet, 8192, lanes, base + 303, structured, hidden_permutation, hidden_selectors)
            lifted = lift_forward(model_leaves, alphabet, permutation, selectors)
            model = fit_model(alphabet, lifted.roots, flatten_details(lifted))
            leaves = [
                synthesize_leaves(alphabet, 8192, lanes, base + 400 + index, structured, hidden_permutation, hidden_selectors)
                for index in range(8)
            ]
            packet = build_container(
                model,
                [ExpertInput.create(values, permutation, selectors) for values in leaves],
            )
            receipt, independent, rebuilt = verify_decode_reencode(packet, [leaf_digest(values) for values in leaves])
            if rebuilt != packet or not all(np.array_equal(a, b) for a, b in zip(independent, leaves, strict=True)):
                raise RuntimeError("independent synthetic replay")
            ledger = physical_ledger(packet)
            rows[tag] = {
                "search": {
                    "selected_seed": search.selected_seed,
                    "candidate_rows": list(search.candidate_rows),
                    "telemetry": search.telemetry,
                },
                "physical_ledger": ledger,
                "independent_receipt": receipt,
                "source_gain_claim": False,
            }
            packets[tag] = packet
    comparisons: dict[str, object] = {}
    pass_all = True
    for alphabet in (2, 4):
        structured_rate = float(rows[f"a{alphabet}_structured"]["physical_ledger"]["physical_bits_per_leaf_symbol"])
        control_rate = float(rows[f"a{alphabet}_control"]["physical_ledger"]["physical_bits_per_leaf_symbol"])
        gap = control_rate - structured_rate
        conditions = {
            "finite_gap_gt_0_15": gap > 0.15,
            "structured_cold_below_two": bool(rows[f"a{alphabet}_structured"]["physical_ledger"]["cold_below_two"]),
            "control_cold_below_two": bool(rows[f"a{alphabet}_control"]["physical_ledger"]["cold_below_two"]),
        }
        passed = all(conditions.values())
        pass_all = pass_all and passed
        comparisons[f"a{alphabet}"] = {
            "structured_rate": structured_rate,
            "control_rate": control_rate,
            "gap": gap,
            "conditions": conditions,
            "status": "PASS_SYNTHETIC_MECHANISM" if passed else "EARLY_KILL",
        }
    result = {
        "schema": "silt-v1-unsealed-synthetic-only-result",
        "status": "UNSEALED_SYNTHETIC_PASS" if pass_all else "UNSEALED_EARLY_KILL",
        "authenticated_source_root": observed_root,
        "rows": rows,
        "comparisons": comparisons,
        "source_gain_claim": False,
        "payload_authority": False,
        "result_frozen": False,
    }
    with SafePublisher(arguments.output_path, observed_root) as publisher:
        for name, packet in packets.items():
            publisher.write(f"{name}.silt", packet)
        publisher.write("UNSEALED_RESULT.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        publication = publisher.finish()
    print(json.dumps({"status": result["status"], "output_path": publication.output_path, "artifact_root": publication.artifact_root_sha256}))
    return 0 if pass_all else 2


if __name__ == "__main__":
    raise SystemExit(main())

