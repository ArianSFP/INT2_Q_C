#!/usr/bin/env python3
"""Authority/input preflight only; never imports numeric packages or launches CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


def _open_absolute_directory_no_symlinks(path: str) -> int:
    normalized = os.path.normpath(path)
    if os.name != "posix" or not os.path.isabs(path) or normalized != path:
        raise RuntimeError("canonical absolute POSIX package path required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in [value for value in normalized.split("/") if value]:
            if component in (".", ".."):
                raise RuntimeError("unsafe package path component")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _bootstrap_source(expected_manifest_sha256: str) -> tuple[str, dict[str, bytes]]:
    """Authenticate the flat package before importing any sibling module."""
    if os.name != "posix" or len(expected_manifest_sha256) != 64:
        raise RuntimeError("POSIX and an external source-manifest SHA-256 are required")
    package = os.path.dirname(os.path.abspath(__file__))
    directory = _open_absolute_directory_no_symlinks(package)
    try:
        descriptor = os.open(
            "SOURCE_MANIFEST.json",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or not (0 < metadata.st_size <= 1 << 20):
                raise RuntimeError("source manifest type/size")
            raw_manifest = b""
            while len(raw_manifest) < metadata.st_size:
                packet = os.read(descriptor, metadata.st_size - len(raw_manifest))
                if not packet:
                    raise RuntimeError("source manifest early EOF")
                raw_manifest += packet
            if os.read(descriptor, 1):
                raise RuntimeError("source manifest trailing bytes")
        finally:
            os.close(descriptor)
        observed = hashlib.sha256(raw_manifest).hexdigest()
        if observed != expected_manifest_sha256.lower():
            raise RuntimeError("external source-manifest digest mismatch")
        manifest = json.loads(raw_manifest)
        rows = manifest.get("files")
        if not isinstance(rows, list):
            raise RuntimeError("source manifest rows")
        expected_names = {"SOURCE_MANIFEST.json"}
        packets: dict[str, bytes] = {}
        for row in rows:
            name = row.get("name")
            if (
                not isinstance(name, str)
                or name in expected_names
                or name in ("", ".", "..")
                or "/" in name
                or "\\" in name
            ):
                raise RuntimeError("source manifest member name")
            if type(row.get("bytes")) is not int or not (0 <= row["bytes"] <= 1 << 20):
                raise RuntimeError("source manifest member byte cap")
            expected_names.add(name)
            member = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
            try:
                member_stat = os.fstat(member)
                if not stat.S_ISREG(member_stat.st_mode) or member_stat.st_size != row.get("bytes"):
                    raise RuntimeError(f"source member type/bytes: {name}")
                value = b""
                while len(value) < member_stat.st_size:
                    chunk = os.read(member, min(1 << 20, member_stat.st_size - len(value)))
                    if not chunk:
                        raise RuntimeError(f"source member early EOF: {name}")
                    value += chunk
                if hashlib.sha256(value).hexdigest() != row.get("sha256"):
                    raise RuntimeError(f"source member digest: {name}")
                packets[name] = value
            finally:
                os.close(member)
        if set(os.listdir(directory)) != expected_names:
            raise RuntimeError("source package closure")
        return observed, packets
    finally:
        os.close(directory)


def _authorization(action: str) -> str:
    return {
        "synthetic": "SYNTHETIC_ONLY_UNIPOLAR_N18_307_V2",
        "pilot": "OPEN_AUTHENTICATED_UNIPOLAR_N18_307_PILOT_V2",
        "full": "OPEN_AUTHENTICATED_UNIPOLAR_N18_307_FULL_V2",
    }[action]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("synthetic", "pilot", "full"), required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--review-receipt", required=True)
    parser.add_argument("--source-plan")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.authorization != _authorization(arguments.action):
        raise SystemExit("authorization mismatch; output/review/environment/payload/CUDA not opened")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 0; output/review/environment/payload/CUDA not opened")
    for label, value in (
        ("review receipt", arguments.review_receipt),
        ("repository root", arguments.repo_root),
        ("output", arguments.output),
    ):
        if not os.path.isabs(value) or os.path.normpath(value) != value:
            raise SystemExit(f"{label} must be a canonical absolute path")
    if arguments.action in ("pilot", "full"):
        if arguments.source_plan is None or not os.path.isabs(arguments.source_plan) or os.path.normpath(arguments.source_plan) != arguments.source_plan:
            raise SystemExit("pilot/full source plan must be a canonical absolute path")
    elif arguments.source_plan is not None:
        raise SystemExit("synthetic preflight rejects a source plan")

    manifest_sha256, _packets = _bootstrap_source(arguments.expected_source_manifest_sha256)

    # Sibling imports occur only after the complete source package is held and hashed.
    package = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, package)
    from n18_common import (
        MAX_ENVIRONMENT_LOCK_BYTES,
        MAX_PLAN_BYTES,
        MAX_REVIEW_BYTES,
        canonical_json,
        sha256_bytes,
    )
    from runtime_contract import (
        authenticate_dependencies,
        validate_environment_lock,
        validate_review_receipt,
    )
    from secure_io import CompletionLastPublisher, HeldFileSet, HeldRegularFile
    from source_adapter import parse_source_plan

    # Reserve a private staging output before reading review, environment or payload.
    with CompletionLastPublisher(arguments.output, manifest_sha256) as output:
        with HeldFileSet() as held:
            review = held.add(
                HeldRegularFile(arguments.review_receipt, maximum_bytes=MAX_REVIEW_BYTES)
            )
            validate_review_receipt(review.read(), manifest_sha256, arguments.action)

            environment_path = os.path.join(package, "runtime_environment_lock.json")
            environment = held.add(
                HeldRegularFile(
                    environment_path,
                    maximum_bytes=MAX_ENVIRONMENT_LOCK_BYTES,
                    expected_sha256=hashlib.sha256(
                        _packets["runtime_environment_lock.json"]
                    ).hexdigest(),
                )
            )
            # The checked-in v2 lock intentionally fails here. Thus no source plan,
            # matrix payload, numeric module, CuPy runtime, or CUDA context is opened.
            validate_environment_lock(environment.read())

            plan = None
            source_rows: list[dict[str, Any]] = []
            if arguments.source_plan is not None:
                source_plan = held.add(
                    HeldRegularFile(arguments.source_plan, maximum_bytes=MAX_PLAN_BYTES)
                )
                plan = parse_source_plan(source_plan.read())
                # All counts, products, paths and byte/hash descriptors were bounded
                # before this loop opens or allocates anything proportional to source.
                selected_matrices = (
                    plan.experts[0].matrices
                    if arguments.action == "pilot"
                    else plan.matrices
                )
                for row in selected_matrices:
                    source = held.add(
                        HeldRegularFile(
                            row.absolute_path,
                            maximum_bytes=row.bytes,
                            expected_bytes=row.bytes,
                            expected_sha256=row.sha256,
                        )
                    )
                    source_rows.append(
                        {
                            "expert_ordinal": row.expert_ordinal,
                            "role": row.canonical_role,
                            "bytes": source.bytes,
                            "sha256": source.sha256,
                        }
                    )

            dependency_path = os.path.join(package, "dependency_graph.json")
            dependency_lock = held.add(
                HeldRegularFile(
                    dependency_path,
                    maximum_bytes=1 << 20,
                    expected_sha256=hashlib.sha256(_packets["dependency_graph.json"]).hexdigest(),
                )
            )
            with authenticate_dependencies(dependency_lock.read(), arguments.repo_root) as dependencies:
                held.verify_stable()
                receipt = {
                    "schema": "tactic_actual_coarse_n18_preflight_v2",
                    "status": "PASS_AUTHORITY_AND_HELD_INPUT_PREFLIGHT_NO_NUMERIC_IMPORT_NO_CUDA",
                    "action": arguments.action,
                    "source_manifest_sha256": manifest_sha256,
                    "source_plan": None if plan is None else {
                        "experts": len(plan.experts),
                        "matrices": len(plan.matrices),
                        "ledger": plan.ledger,
                    },
                    "held_sources": source_rows,
                    "dependencies": list(dependencies.rows),
                    "numeric_imports": 0,
                    "cuda_contexts": 0,
                    "payload_result_claim": False,
                }
                packet = canonical_json(receipt) + b"\n"
                output.write("preflight.json", packet)
                output.complete(
                    {
                        "status": receipt["status"],
                        "preflight_sha256": sha256_bytes(packet),
                        "producer_implemented": False,
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
