#!/usr/bin/env python3
"""Independent standard-library closure verifier for the v1 review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


MEMBERS = {
    "README.md", "review_contract.json", "run_cupy_reproducibility.py",
    "test_independent_source.py", "verify_review.py",
}
STATUS = "SOURCE_REPAIRS_SUBSTANTIALLY_CLOSE_V0__HOLD_PAYLOAD_FOR_RUNTIME_SCALABILITY_AND_COARSE_DECODER_AUDIT"
PRODUCER_MANIFEST = "f4ba72b9371d77ad4347d5a4fe377677473844dd696032e662acc6cd3bde22b4"
PRODUCER_ROOT = "6840b6a0eb4f2856f84c610ba11888382ecca257d88ebda7f5b49c0de9f3b3c5"


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, object]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    result = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                        parse_constant=lambda token: (_ for _ in ()).throw(ReviewError(token)))
    require(isinstance(result, dict), f"{label} object")
    return result


def read_regular(path: Path, maximum: int = 4 << 20) -> bytes:
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and 0 < before.st_size <= maximum, "regular single-link source")
        output = bytearray()
        while len(output) < before.st_size:
            row = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(row), "short source read")
            output.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
                "source identity drift")
        return bytes(output)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest: str | None) -> dict[str, object]:
    root = package.resolve(strict=True)
    entries = list(os.scandir(root))
    require(root.is_dir() and not root.is_symlink(), "real review directory")
    require(all(entry.is_file(follow_symlinks=False) and not entry.is_symlink() for entry in entries),
            "review files only")
    require({entry.name for entry in entries} == MEMBERS | {"SOURCE_MANIFEST.json"},
            "exact review member set")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json")
    manifest_sha = digest(manifest_payload)
    if expected_manifest is not None:
        require(manifest_sha == expected_manifest, "expected review manifest")
    document = strict_json(manifest_payload, "review manifest")
    require(document.get("schema") == "tactic-ramanujan384-authority-v1-independent-review-manifest-v0"
            and document.get("status") == STATUS, "review identity")
    require(document.get("producer") == {
        "source_manifest_sha256": PRODUCER_MANIFEST,
        "source_root_sha256": PRODUCER_ROOT,
    }, "producer pins")
    rows = document.get("members")
    require(isinstance(rows, list) and [row.get("name") for row in rows] == sorted(MEMBERS),
            "review member rows")
    observed = []
    for row in rows:
        payload = read_regular(root / row["name"])
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"],
                f"review member closure {row['name']}")
        observed.append({"name": row["name"], "bytes": len(payload), "sha256": digest(payload)})
    require(digest(canonical(observed)) == document.get("source_root_sha256"), "review root")
    producer = root.parent / "tactic_ramanujan384_adapter_v1_authority"
    producer_payload = read_regular(producer / "SOURCE_MANIFEST.json")
    require(digest(producer_payload) == PRODUCER_MANIFEST, "producer manifest")
    producer_document = strict_json(producer_payload, "producer manifest")
    producer_rows = []
    for row in producer_document["members"]:
        payload = read_regular(producer / row["name"])
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"],
                f"producer member {row['name']}")
        producer_rows.append({"name": row["name"], "bytes": len(payload), "sha256": digest(payload)})
    require(digest(canonical(producer_rows)) == PRODUCER_ROOT, "producer canonical root")
    contract = strict_json(read_regular(root / "review_contract.json"), "review contract")
    require(contract.get("status") == STATUS and contract.get("payload_execution_authorized") is False,
            "review disposition")
    require(document.get("access_attestation") == {
        "qwen_payload_accessed": False,
        "coarse_model_payload_accessed": False,
        "matched_model_control_payload_accessed": False,
        "network_accessed": False,
    }, "review source-only access")
    return {
        "schema": "tactic-ramanujan384-authority-v1-independent-review-verifier-receipt-v0",
        "status": STATUS,
        "review_manifest_sha256": manifest_sha,
        "review_source_root_sha256": document["source_root_sha256"],
        "producer_manifest_sha256": PRODUCER_MANIFEST,
        "producer_source_root_sha256": PRODUCER_ROOT,
        "payload_authorized": False,
        "qwen_payload_accessed": False,
        "network_accessed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.package, arguments.manifest_sha256),
                     sort_keys=True, separators=(",", ":")))

