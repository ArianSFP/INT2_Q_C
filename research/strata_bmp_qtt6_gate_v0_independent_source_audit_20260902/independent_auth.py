#!/usr/bin/env python3
"""Independent authentication of the frozen STRATA-BMP/OBDD/QTT6 source."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "a7778080a00d5d2967636ac8d60dd31698401c4dcf8da160c9451c92dc5f6b18"
)
EXPECTED_SOURCE_ROOT_SHA256 = (
    "6b7baf9706349d10108121d4dcb03661b2378dc436303bbfe1bbccd38a0c8914"
)
EXPECTED_MEMBERS = {
    "README.md": (8640, "97eef3376c24ab1b7609d2e8a24f3b240bd40470db4b38f9e3395980dc0e551b"),
    "THREAT_MODEL.md": (5328, "78f84ebfd29d2677669792a47c3ac6ab139e4974a3155dc65eebc00d1be99ecf"),
    "codec.py": (23144, "feef4ae809ad9c6d132730f9271c089590620b2f9eda2659fc2a4ac7535637ca"),
    "cupy_backend.py": (2167, "3fd8fc40771226acb34f1accb0bf77be077484bfa77bdb679b74392822ea36ea"),
    "design_lock.json": (2561, "41bbbfc2fe13233c9a42b0a9cb223c22056fd8191820a14807f3bb1bfba25976"),
    "run_cupy_smoke.py": (2746, "8259b7f304b62482fec695969879ecf0e6f20ec363eb134c4b6ef53bc232fc70"),
    "run_source_free_fixture.py": (5400, "eccce4e832435d50f5cd1f54f23b4e9f99188d1e5eadff4c35d0d50f70e6fa49"),
    "search.py": (15450, "3b977b1d5a03828860257a3d2615bf014ddb631e32b75eb376633faa67952df5"),
    "test_source_only.py": (7890, "867e781bb95421eec8bd4b5e2bb402d13ff0a092987a1a491dff405c2d3b6b41"),
    "verify_source.py": (9639, "b0ea0742de7d1a133677a4cce5780661e128e17149fcc99a1bfa852b66721510"),
}


class AuthError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuthError(f"{label} nonfinite {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthError(f"{label} strict JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label} regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise AuthError(f"{label} read") from exc
    require(
        (before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
        (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
        f"{label} changed during read",
    )
    return payload


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def authenticate_source(source: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    require(source.is_dir() and not source.is_symlink(),
            "source real directory")
    manifest_payload = read_regular(source / "SOURCE_MANIFEST.json", "manifest")
    require(sha256(manifest_payload) == EXPECTED_SOURCE_MANIFEST_SHA256,
            "externally pinned source manifest")
    manifest = strict_json(manifest_payload, "manifest")
    require(manifest.get("source_root_sha256") == EXPECTED_SOURCE_ROOT_SHA256,
            "declared source root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_MEMBERS),
            "manifest member count")
    observed = []
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"}, "member schema")
        name = row["name"]
        require(isinstance(name, str) and name == Path(name).name and
                name in EXPECTED_MEMBERS and name not in seen, "member name")
        payload = read_regular(source / name, f"member {name}")
        actual = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require((actual["bytes"], actual["sha256"]) == EXPECTED_MEMBERS[name],
                f"external member pin {name}")
        require(actual == row, f"manifest member pin {name}")
        observed.append(actual)
        seen.add(name)
    require(seen == set(EXPECTED_MEMBERS), "complete member set")
    require(sha256(canonical_json(observed)) == EXPECTED_SOURCE_ROOT_SHA256,
            "independent source root")
    entries = list(os.scandir(source))
    require({entry.name for entry in entries} ==
            set(EXPECTED_MEMBERS) | {"SOURCE_MANIFEST.json"},
            "exact source closure")
    require(all(entry.is_file(follow_symlinks=False) for entry in entries),
            "source entries regular")
    return {
        "status": "PASS_EXACT_EXTERNALLY_PINNED_SOURCE",
        "source": str(source),
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "source_root_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "member_count": len(observed),
        "strict_regular_closure": True,
    }
