#!/usr/bin/env python3
"""Fail-closed source-only entry point for epsilon-TCQ v0.

V0 intentionally accepts no payload path. It authenticates its exact source
closure and returns the typed missing-adapter HOLD.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


AUTHORIZATION = "REPORT_EPSILON_TCQ_V0_SOURCE_ONLY_HOLD"
SCHEMA = "epsilon-tcq-wfa-early-gate-v0-source-manifest"
STATUS = "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY"


class RunError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"duplicate key {key!r}")
            output[key] = value
        return output
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda item: (_ for _ in ()).throw(
                           RunError(f"nonfinite {item}")))
    require(isinstance(value, dict), "JSON object")
    return value


def read_regular(path: Path, expected_sha256: str | None = None) -> bytes:
    require(path.is_absolute(), "absolute source path")
    descriptor = os.open(os.fspath(path), os.O_RDONLY |
                         getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= 4 * (1 << 20), "source regular file")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(chunk, "source short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns, before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns, after.st_nlink), "source identity drift")
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256, "source SHA-256")
        return payload
    finally:
        os.close(descriptor)


def authenticate(package: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    root = package.resolve(strict=True)
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json",
                                    expected_manifest_sha256)
    manifest = strict_json(manifest_payload)
    require(manifest.get("schema") == SCHEMA and
            manifest.get("status") == STATUS, "manifest schema/status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "manifest rows")
    observed = []
    sources = {}
    names = []
    for row in rows:
        require(set(row) == {"name", "bytes", "sha256"}, "member row")
        name = row["name"]
        require(name not in names and "/" not in name and "\\" not in name,
                "member name")
        payload = read_regular(root / name, row["sha256"])
        require(len(payload) == row["bytes"], "member bytes")
        names.append(name)
        sources[name] = payload
        observed.append({"name": name, "bytes": len(payload),
                         "sha256": sha256(payload)})
    require(names == sorted(names, key=lambda value: value.encode("utf-8")) and
            manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "manifest root/order")
    require({entry.name for entry in os.scandir(root)} ==
            set(names) | {"SOURCE_MANIFEST.json"}, "exact package closure")
    return {"manifest_sha256": sha256(manifest_payload),
            "source_root_sha256": manifest["source_root_sha256"],
            "sources": sources}


def load_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, "module collision")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{sha256(source)}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True),
         module.__dict__)
    return module


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION and
            sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "authorization and CPython -I -B")
    package = Path(__file__).resolve().parent
    closure = authenticate(package, arguments.package_manifest_sha256)
    gate = load_module("epsilon_tcq_v0_authenticated_gate_contract",
                       closure["sources"]["gate_contract.py"])
    return {
        **gate.missing_strata_adapter_hold(),
        "source_manifest_sha256": closure["manifest_sha256"],
        "source_root_sha256": closure["source_root_sha256"],
        "source_package_authenticated": True,
        "payload_argument_accepted_by_v0": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--authorization", required=True)
    value.add_argument("--package-manifest-sha256", required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True,
                     separators=(",", ":")))

