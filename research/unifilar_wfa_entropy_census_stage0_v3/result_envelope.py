#!/usr/bin/env python3
"""Independent completion-last envelope verifier (standard library only)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def verify_completed_directory(common: Any, path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("result path must be absolute")
    names = sorted(entry.name for entry in os.scandir(path))
    if "COMPLETE.json" not in names:
        raise ValueError("incomplete output: COMPLETE.json absent")
    complete_file = common.HeldRegularFile(path / "COMPLETE.json").open()
    held = [complete_file]
    try:
        record = common.strict_json_loads(complete_file.read_all())
        if not isinstance(record, dict) or set(record) != {
            "schema", "status", "source_manifest_sha256", "members", "completion_sha256"
        }:
            raise ValueError("completion schema fields")
        if record["schema"] != "unifilar-wfa-completion-v3" or record["status"] != "COMPLETE_LAST":
            raise ValueError("completion schema/status")
        common.verify_internal_seal(record, "completion_sha256")
        members = record["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("completion member list")
        expected_names = []
        seen = set()
        for row in members:
            if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
                raise ValueError("completion member record")
            name = row["name"]
            if not isinstance(name, str) or name != Path(name).name or name in seen or name == "COMPLETE.json":
                raise ValueError("completion member name")
            if type(row["bytes"]) is not int or row["bytes"] < 0:
                raise ValueError("completion member bytes")
            if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
                raise ValueError("completion member digest")
            item = common.HeldRegularFile(path / name, row["bytes"], row["sha256"]).open()
            held.append(item)
            seen.add(name)
            expected_names.append(name)
        if sorted(expected_names + ["COMPLETE.json"]) != names:
            raise ValueError("completion has unlisted or missing members")
        for item in held:
            item.verify_stable()
        return {
            "status": "PASS_COMPLETE_LAST_ENVELOPE",
            "source_manifest_sha256": record["source_manifest_sha256"],
            "members": members,
            "completion_sha256": record["completion_sha256"],
        }
    finally:
        for item in reversed(held):
            item.close()
