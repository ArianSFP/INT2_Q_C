#!/usr/bin/env python3
"""External atomic bootstrap; executes no package code before snapshot seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import types
from pathlib import Path
from types import MappingProxyType
from typing import Any


VERIFY_AUTHORIZATION = "VERIFY_SOURCE_ONLY_TACTIC_RAMANUJAN384_V3_ATOMIC"
V3_SCHEMA = "tactic-ramanujan384-atomic-source-manifest-v3"
V2_MANIFEST_SHA256 = "1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209"
V2_ROOT_SHA256 = "bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495"
REVIEW_MANIFEST_SHA256 = "4ed8c0fe24db072e22aef84791a01ccf637cb337376a389d47119248fd257281"
REVIEW_ROOT_SHA256 = "16ea8dfde5cf7a48552dc7b5a74b209488934b8764e890bf51bb5cd02985cd39"


def fail(message: str) -> None:
    raise SystemExit(f"atomic bootstrap: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"{label} duplicate key")
            output[key] = value
        return output
    try:
        value = json.loads(
            payload.decode("ascii"), object_pairs_hook=pairs,
            parse_constant=lambda token: fail(f"{label} nonfinite {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"{label} JSON: {exc}")
    require(isinstance(value, dict), f"{label} object")
    return value


def safe_read(path: Path, maximum: int) -> bytes:
    absolute = path.resolve(strict=True)
    require(absolute == path.absolute(), "canonical nonsymlink file path")
    descriptor = os.open(
        os.fspath(absolute), os.O_RDONLY | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "regular single-link file")
        require(0 < before.st_size <= maximum, "bounded nonempty file")
        output = bytearray()
        while len(output) < before.st_size:
            row = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(row), "short descriptor read")
            output.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_nlink), "descriptor identity drift")
        return bytes(output)
    finally:
        os.close(descriptor)


def directory_entries(root: Path) -> list[str]:
    rows = []
    for entry in os.scandir(root):
        require(not entry.is_symlink(), "closure symlink")
        require(entry.is_file(follow_symlinks=False), "flat closure only")
        rows.append(entry.name)
    return sorted(rows)


def authenticate_closure(*, root_argument: Path, manifest_name: str,
                         expected_manifest_sha256: str,
                         expected_root_sha256: str,
                         expected_schema: str | None) -> tuple[dict[str, bytes], dict[str, Any]]:
    root = root_argument.resolve(strict=True)
    require(root == root_argument.absolute() and root.is_dir(), "canonical package directory")
    before_entries = directory_entries(root)
    manifest_payload = safe_read(root / manifest_name, 16 << 20)
    require(sha256(manifest_payload) == expected_manifest_sha256, "manifest external pin")
    document = strict_json(manifest_payload, "source manifest")
    if expected_schema is not None:
        require(document.get("schema") == expected_schema, "source manifest schema")
    require(document.get("source_root_sha256") == expected_root_sha256,
            "source manifest root pin")
    rows = document.get("members")
    require(isinstance(rows, list) and bool(rows), "source manifest members")
    names = []
    canonical_rows = []
    verified = {manifest_name: manifest_payload}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "source member exact schema")
        name, size, digest = row["name"], row["bytes"], row["sha256"]
        require(isinstance(name, str) and name and Path(name).name == name
                and name not in names, "flat unique source member")
        require(type(size) is int and size > 0 and isinstance(digest, str)
                and len(digest) == 64, "source member fields")
        payload = safe_read(root / name, 64 << 20)
        require(len(payload) == size and sha256(payload) == digest,
                f"source member drift {name}")
        names.append(name)
        verified[name] = payload
        canonical_rows.append({"name": name, "bytes": size, "sha256": digest})
    canonical_rows.sort(key=lambda row: row["name"])
    require(rows == canonical_rows, "canonical source member rows")
    require(sha256(canonical_json(canonical_rows)) == expected_root_sha256,
            "independent source root")
    expected_entries = sorted(names + [manifest_name])
    after_entries = directory_entries(root)
    require(before_entries == expected_entries == after_entries,
            "exact stable closure before and after descriptor reads")
    return verified, document


def write_snapshot(root: Path, verified: dict[str, bytes]) -> None:
    for relative in sorted(verified):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_BINARY", 0), 0o400,
        )
        try:
            payload = verified[relative]
            cursor = 0
            while cursor < len(payload):
                written = os.write(descriptor, payload[cursor:])
                require(written > 0, "short snapshot write")
                cursor += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, stat.S_IRUSR)
    directories = sorted({root} | {path.parent for path in root.rglob("*")},
                         key=lambda value: len(value.parts), reverse=True)
    for directory in directories:
        os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR)


def verify_snapshot(root: Path, expected: dict[str, bytes]) -> MappingProxyType:
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    require(actual == sorted(expected), "snapshot exact closure")
    output = {}
    for relative in sorted(expected):
        payload = safe_read(root / relative, 64 << 20)
        require(payload == expected[relative], f"snapshot byte identity {relative}")
        output[relative] = payload
    return MappingProxyType(output)


def execute_runner(snapshot: MappingProxyType, mode: str, authorization: str,
                   receipt: dict[str, Any]) -> int:
    payload = snapshot["v3/snapshot_runner.py"]
    module = types.ModuleType("tactic_ramanujan384_v3_atomic_snapshot_runner")
    module.__file__ = "<immutable-atomic-snapshot>/v3/snapshot_runner.py"
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(payload, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return int(module.snapshot_main(snapshot_bytes=snapshot, mode=mode,
                                    authorization=authorization,
                                    snapshot_receipt=MappingProxyType(receipt)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v2-review", type=Path, required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--mode", choices=("verify-only", "source-free-cpu",
                                            "source-free-cupy"), required=True)
    parser.add_argument("--authorization", required=True)
    arguments = parser.parse_args()
    own_payload = safe_read(Path(__file__), 4 << 20)
    require(sha256(own_payload) == arguments.bootstrap_sha256,
            "external bootstrap SHA256")
    if arguments.mode == "verify-only":
        require(arguments.authorization == VERIFY_AUTHORIZATION,
                "verify-only authorization")
    v3, v3_manifest = authenticate_closure(
        root_argument=arguments.package, manifest_name="SOURCE_MANIFEST.json",
        expected_manifest_sha256=arguments.manifest_sha256,
        expected_root_sha256=strict_json(
            safe_read(arguments.package / "SOURCE_MANIFEST.json", 16 << 20),
            "v3 pin manifest",
        ).get("source_root_sha256"), expected_schema=V3_SCHEMA,
    )
    dependencies = v3_manifest.get("dependency_pins", {})
    require(dependencies.get("v2", {}).get("source_manifest_sha256") == V2_MANIFEST_SHA256
            and dependencies.get("v2", {}).get("source_root_sha256") == V2_ROOT_SHA256,
            "v2 dependency pins")
    require(dependencies.get("v2_review", {}).get("source_manifest_sha256")
            == REVIEW_MANIFEST_SHA256
            and dependencies.get("v2_review", {}).get("source_root_sha256")
            == REVIEW_ROOT_SHA256, "v2 review dependency pins")
    require(dependencies.get("external_bootstrap", {}).get("sha256")
            == arguments.bootstrap_sha256, "bootstrap dependency pin")
    v2, _ = authenticate_closure(
        root_argument=arguments.v2, manifest_name="SOURCE_MANIFEST.json",
        expected_manifest_sha256=V2_MANIFEST_SHA256,
        expected_root_sha256=V2_ROOT_SHA256,
        expected_schema="tactic-ramanujan384-scalable-source-manifest-v2",
    )
    review, _ = authenticate_closure(
        root_argument=arguments.v2_review, manifest_name="source_manifest.json",
        expected_manifest_sha256=REVIEW_MANIFEST_SHA256,
        expected_root_sha256=REVIEW_ROOT_SHA256,
        expected_schema="tactic-ramanujan384-v2-scalable-independent-source-review-manifest",
    )
    combined = {}
    for prefix, closure in (("v3", v3), ("v2", v2), ("review", review)):
        for name, payload in closure.items():
            combined[f"{prefix}/{name}"] = payload
    sys.dont_write_bytecode = True
    with tempfile.TemporaryDirectory(prefix="tactic-r384-v3-atomic-") as temporary:
        snapshot_root = Path(temporary) / "snapshot"
        snapshot_root.mkdir()
        write_snapshot(snapshot_root, combined)
        immutable = verify_snapshot(snapshot_root, combined)
        receipt = {
            "schema": "tactic-ramanujan384-v3-atomic-snapshot-receipt",
            "v3_manifest_sha256": arguments.manifest_sha256,
            "v3_source_root_sha256": v3_manifest["source_root_sha256"],
            "v2_manifest_sha256": V2_MANIFEST_SHA256,
            "v2_source_root_sha256": V2_ROOT_SHA256,
            "v2_review_manifest_sha256": REVIEW_MANIFEST_SHA256,
            "v2_review_source_root_sha256": REVIEW_ROOT_SHA256,
            "snapshot_files": len(immutable),
            "immutable_verified_byte_snapshot": True,
            "package_imports_before_snapshot": 0,
            "mutable_producer_tree_imports": 0,
            "network_accessed": False,
        }
        try:
            if arguments.mode == "verify-only":
                print(json.dumps({**receipt, "status": "PASS_ATOMIC_SOURCE_SNAPSHOT_ONLY"},
                                 sort_keys=True, separators=(",", ":")))
                return 0
            return execute_runner(immutable, arguments.mode, arguments.authorization, receipt)
        finally:
            # Restore owner write permission solely so TemporaryDirectory can
            # remove its private tree after execution; the runner has already
            # consumed only immutable in-memory bytes.
            for path in sorted(snapshot_root.rglob("*"),
                               key=lambda value: len(value.parts), reverse=True):
                os.chmod(path, (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                         if path.is_dir() else (stat.S_IRUSR | stat.S_IWUSR))
            os.chmod(snapshot_root, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


if __name__ == "__main__":
    raise SystemExit(main())
