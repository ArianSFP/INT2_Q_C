#!/usr/bin/env python3
"""Retained, standard-library verification of the frozen N18 v6 sources."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MANIFEST_SCHEMA = "tactic-actual-coarse-n18-v6-source-manifest-v1"


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            require(key not in output, f"{label}: duplicate JSON key")
            output[key] = value
        return output

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                VerifyError(f"{label}: nonfinite {item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerifyError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode),
                f"{label}: symlink component {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def read_nofollow(
    path: Path, *, expected_bytes: int | None = None,
    expected_sha256: str | None = None, maximum_bytes: int = 4 * (1 << 20),
    label: str,
) -> bytes:
    require(path.is_absolute(), f"{label}: absolute path")
    reject_symlink_chain(path, label)
    descriptor = os.open(
        os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) |
        getattr(os, "O_BINARY", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= maximum_bytes,
                f"{label}: sole-link regular byte bound")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes,
                    f"{label}: exact bytes")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label}: trailing bytes")
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_mode, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns, before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_mode, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns, after.st_nlink),
                f"{label}: identity drift")
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256,
                    f"{label}: SHA-256")
        return payload
    finally:
        os.close(descriptor)


def load_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, f"authenticated module collision: {name}")
    digest = sha256(source)
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True,
                     optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def v4_root(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"TACTIC-N18-V4-SOURCE-ROOT-v1\0")
    for row in sorted(rows, key=lambda item: item["name"].encode("utf-8")):
        name = row["name"].encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(int(row["bytes"]).to_bytes(8, "big"))
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def verify() -> dict[str, Any]:
    require(sys.dont_write_bytecode, "invoke source verifier with CPython -B")
    manifest_payload = read_nofollow(
        ROOT / "SOURCE_MANIFEST.json", maximum_bytes=1 << 20,
        label="source manifest bootstrap")
    bootstrap = strict_json(manifest_payload, "source manifest bootstrap")
    require(bootstrap.get("schema") == MANIFEST_SCHEMA,
            "source manifest bootstrap schema")
    rows = bootstrap.get("members")
    require(isinstance(rows, list), "source manifest bootstrap rows")
    source_auth_rows = [row for row in rows if isinstance(row, dict) and
                        row.get("name") == "source_auth.py"]
    require(len(source_auth_rows) == 1, "source_auth bootstrap row")
    row = source_auth_rows[0]
    source_auth_bytes = read_nofollow(
        ROOT / "source_auth.py", expected_bytes=row["bytes"],
        expected_sha256=row["sha256"], label="source_auth bootstrap")
    auth = load_module("tacn18_v6_verify_source_auth", source_auth_bytes)
    with auth.HeldSourcePackage(
        ROOT, sha256(manifest_payload),
        executing_path=Path(__file__).resolve(strict=True),
    ) as package:
        sources = package.sources
        predecessor = strict_json(
            sources["PREDECESSOR_LOCK.json"], "predecessor lock")
        require(predecessor.get("schema") ==
                "tactic-actual-coarse-n18-v6-predecessor-lock-v1",
                "predecessor schema")
        require(predecessor.get("source_root_sha256") ==
                "1f9f2c92df3796f5f23b7e3a6b0826d6d8a2ea53bc70014fb75e61e7bc8a9fbf",
                "v4 root pin")
        v4_dir = REPO / predecessor["relative_directory"]
        reject_symlink_chain(v4_dir, "v4 package")
        v4_rows = []
        for member in predecessor["members"]:
            require(isinstance(member, dict) and
                    set(member) == {"name", "bytes", "sha256"},
                    "v4 member row")
            payload = read_nofollow(
                v4_dir / member["name"], expected_bytes=member["bytes"],
                expected_sha256=member["sha256"], label=f"v4 {member['name']}")
            v4_rows.append({"name": member["name"], "bytes": len(payload),
                            "sha256": sha256(payload)})
        actual_v4 = list(os.scandir(v4_dir))
        require({entry.name for entry in actual_v4} ==
                {row["name"] for row in v4_rows} and
                all(entry.is_file(follow_symlinks=False) for entry in actual_v4),
                "exact v4 regular-file closure")
        require(v4_root(v4_rows) == predecessor["source_root_sha256"],
                "v4 framed root")
        for dependency in predecessor["numerical_dependencies"]:
            read_nofollow(
                REPO / dependency["relative_path"],
                expected_bytes=dependency["bytes"],
                expected_sha256=dependency["sha256"],
                maximum_bytes=1 << 20,
                label=f"numerical dependency {dependency['relative_path']}")

        runtime = strict_json(sources["RUNTIME_LOCK.json"], "runtime lock")
        require(runtime.get("schema") ==
                "tactic-actual-coarse-n18-v6-runtime-lock-v1",
                "runtime schema")
        require(runtime.get("python") == "3.12.3" and
                runtime.get("numpy") == "2.5.2" and
                runtime.get("cupy") == "14.2.0" and
                runtime.get("device_name") == "NVIDIA GeForce RTX 5090" and
                runtime.get("compute_capability") == "120",
                "exact runtime lock")
        design = strict_json(sources["design_lock.json"], "design lock")
        require(design.get("schema") ==
                "tactic-actual-coarse-n18-source-design-v6" and
                design["predecessor"]["v4_modified"] is False and
                design["predecessor"]["v5_modified"] is False and
                design["predecessor"]["packet_grammar_changed"] is False,
                "immutable predecessor design")
        boundary = design["claim_boundary"]
        require(boundary["source_free_smoke_executed_in_this_source_freeze"]
                is False and boundary["payload_accessed_during_build"] is False
                and boundary["positive_claim_authority"] is False,
                "source-only claim boundary")
        package.verify_final()
        receipt = package.receipt()
        return {
            "schema": "tactic-actual-coarse-n18-v6-source-verification-v1",
            "status": "PASS_RETAINED_SOURCE_CLOSURE_AWAITING_EXTERNAL_CUPY_SMOKE",
            "manifest_sha256": package.manifest_sha256,
            "source_root_sha256": package.source_root_sha256,
            "source_members": len(package.members),
            "source_bytes": sum(len(value) for value in sources.values()),
            "executing_entry_name": receipt["executing_entry_name"],
            "retained_no_follow_descriptors": True,
            "v4_source_root_sha256": predecessor["source_root_sha256"],
            "runtime_lock_sha256": sha256(sources["RUNTIME_LOCK.json"]),
            "source_free_smoke_run": False,
            "payload_accessed": False,
            "cuda_initialized": False,
            "positive_claim_authority": False,
        }


def main() -> int:
    print(json.dumps(verify(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
