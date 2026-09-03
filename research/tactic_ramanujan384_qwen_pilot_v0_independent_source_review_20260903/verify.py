#!/usr/bin/env python3
"""Stdlib-only verifier for the independent review and pinned producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


PRODUCER_MANIFEST = "340ef7f532ab02e03bf04257f3ff07dbc4736bd9e5e96203169603df918e3a8a"
PRODUCER_ROOT = "611bf1b9c822cb90f32a2956e52d8332ef75374186e4acedc958ec3a6c5468ec"
PRODUCER_MEMBERS = {
    "README.md", "SOURCE_ONLY_TEST_RESULT.json", "aperture.py", "capability.py",
    "design_lock.json", "pilot_runner.py", "test_source_only.py", "verify_source.py",
}
REVIEW_SCHEMA = "tactic-ramanujan384-qwen-pilot-v0-independent-source-review-manifest-v1"
REVIEW_MEMBERS = {
    "README.md", "STATIC_REVIEW_RECEIPT.json", "THREAT_MODEL.md",
    "review_static.ps1", "verify.py",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, f"{label}: duplicate key")
            output[key] = value
        return output
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               VerifyError(f"{label}: nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label}: regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
             after.st_mode), f"{label}: identity drift")
    return payload


def verify_flat(root: Path, manifest_digest: str, source_root: str,
                expected_names: set[str]) -> dict[str, Any]:
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json", "manifest")
    require(hashlib.sha256(manifest_payload).hexdigest() == manifest_digest,
            "manifest external pin")
    manifest = strict_json(manifest_payload, "manifest")
    require(manifest.get("source_root_sha256") == source_root, "source root pin")
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == len(expected_names), "members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member row")
        name = row["name"]
        require(isinstance(name, str) and name == Path(name).name and
                name in expected_names and name not in names, "member name")
        payload = read_regular(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}
        require(item == row, f"member pin {name}")
        observed.append(item)
        names.append(name)
    require(names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical order")
    require(hashlib.sha256(canonical_json(observed)).hexdigest() == source_root,
            "canonical root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == expected_names | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact flat closure")
    return {"manifest_sha256": manifest_digest,
            "source_root_sha256": source_root, "members": len(rows)}


def verify(review: Path, producer: Path, expected_review_manifest: str) -> dict[str, Any]:
    require(HEX64.fullmatch(expected_review_manifest) is not None,
            "review manifest SHA-256")
    producer_result = verify_flat(producer.resolve(strict=True), PRODUCER_MANIFEST,
                                  PRODUCER_ROOT, PRODUCER_MEMBERS)
    review_root = review.resolve(strict=True)
    manifest_payload = read_regular(review_root / "SOURCE_MANIFEST.json", "review manifest")
    require(hashlib.sha256(manifest_payload).hexdigest() == expected_review_manifest,
            "review manifest external pin")
    manifest = strict_json(manifest_payload, "review manifest")
    require(manifest.get("schema") == REVIEW_SCHEMA and
            manifest.get("producer_manifest_sha256") == PRODUCER_MANIFEST and
            manifest.get("producer_source_root_sha256") == PRODUCER_ROOT,
            "review semantic pins")
    review_result = verify_flat(review_root, expected_review_manifest,
                                manifest["source_root_sha256"], REVIEW_MEMBERS)
    receipt = strict_json(read_regular(
        review_root / "STATIC_REVIEW_RECEIPT.json", "receipt"), "receipt")
    require(receipt.get("powershell_static_review_executed") is True and
            receipt.get("final_frozen_python_tests_executed") is False and
            receipt.get("runtime_executed") is False and
            receipt.get("producer_manifest_sha256") == PRODUCER_MANIFEST and
            receipt.get("producer_source_root_sha256") == PRODUCER_ROOT,
            "honest static receipt")
    return {
        "schema": "tactic-ramanujan384-qwen-pilot-v0-independent-review-verification-v1",
        "status": "PASS_EXACT_REVIEW_AND_PRODUCER_CLOSURES_RUNTIME_HELD",
        "producer": producer_result,
        "review": review_result,
        "python_cupy_payload_runtime_executed": False,
        "disposition": receipt["disposition"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--expected-review-manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.review, args.producer,
                            args.expected_review_manifest_sha256),
                     sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
