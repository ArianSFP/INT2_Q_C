#!/usr/bin/env python3
"""Held review, v6-result, and identity-free BF16 input contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from pathlib import Path
from typing import Any


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REVIEW_SCHEMA = "tactic-dh384-finite-v3-launch-review-v1"
REVIEW_STATUS = "AUTHORIZE_ONE_BOUND_QWEN_GEOMETRY_EXPERT_FINITE_PILOT"
V6_COMPLETE_SCHEMA = "tactic-actual-coarse-n18-v6-completion-v1"
V6_RESULT_STATUS = (
    "PASS_V6_BOUND_TARGET_ELIGIBLE_FRAME_NONPROMOTING_"
    "INDEPENDENT_RESULT_AUDIT_REQUIRED"
)
INPUT_SCHEMA = "tactic-actual-coarse-n18-v6-input-manifest-v1"
MAX_JSON = 4 * (1 << 20)
MAX_ROLE_BYTES = 1 << 34


class ExternalError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ExternalError(f"{label}: nonfinite {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: object")
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


def identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def pread_exact(descriptor: int, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1 << 20, size - offset), offset)
        require(bool(chunk), f"{label}: premature EOF")
        chunks.append(chunk)
        offset += len(chunk)
    require(os.pread(descriptor, 1, size) == b"", f"{label}: trailing bytes")
    return b"".join(chunks)


def read_held_regular(path: Path, *, expected_bytes: int | None = None,
                      expected_sha256: str | None = None,
                      maximum_bytes: int, label: str) -> bytes:
    require(path.is_absolute(), f"{label}: absolute path")
    reject_symlink_chain(path, label)
    descriptor = os.open(
        os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0) |
        getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= maximum_bytes,
                f"{label}: regular sole-link byte bound")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes, f"{label}: exact bytes")
        payload = pread_exact(descriptor, before.st_size, label)
        if expected_sha256 is not None:
            require(HEX64.fullmatch(expected_sha256) is not None and
                    sha256(payload) == expected_sha256,
                    f"{label}: digest")
        after = os.fstat(descriptor)
        require(identity(after) == identity(before), f"{label}: identity drift")
        return payload
    finally:
        os.close(descriptor)


def validate_launch_review(
    payload: bytes, *,
    package_manifest_sha256: str,
    package_source_root_sha256: str,
    v6_source_manifest_sha256: str,
    v6_source_root_sha256: str,
) -> dict[str, Any]:
    record = strict_json(payload, "launch review")
    require(set(record) == {
        "schema", "status", "package_manifest_sha256",
        "package_source_root_sha256", "v6_source_manifest_sha256",
        "v6_source_root_sha256", "v6_complete_sha256",
        "input_manifest_sha256", "allowed_scope", "independent_audit",
        "review_claim_sha256",
    }, "launch review exact schema")
    require(record["schema"] == REVIEW_SCHEMA and
            record["status"] == REVIEW_STATUS,
            "launch review schema/status")
    require(record["package_manifest_sha256"] == package_manifest_sha256 and
            record["package_source_root_sha256"] == package_source_root_sha256,
            "launch review finite-source binding")
    require(record["v6_source_manifest_sha256"] ==
            v6_source_manifest_sha256 and
            record["v6_source_root_sha256"] == v6_source_root_sha256,
            "launch review v6-source binding")
    require(record["allowed_scope"] == {
        "experts": 1,
        "geometry": [768, 2048],
        "qwen_or_model_identity_available_to_codec": False,
        "universal_tail_claim": False,
    }, "launch review exact scope")
    require(record["independent_audit"] == {
        "finite_source_reviewed": True,
        "v6_completed_result_reviewed": True,
        "payload_launch_explicitly_authorized": True,
    }, "launch review independent audit assertions")
    for field in (
        "v6_complete_sha256", "input_manifest_sha256",
        "review_claim_sha256",
    ):
        require(isinstance(record[field], str) and
                HEX64.fullmatch(record[field]) is not None,
                f"launch review {field}")
    clone = dict(record)
    claimed = clone.pop("review_claim_sha256")
    require(sha256(canonical_json(clone)) == claimed,
            "launch review internal seal")
    return record


class HeldCompletedV6Result:
    def __init__(self, result_dir: Path, *, expected_complete_sha256: str,
                 expected_v6_source_root_sha256: str,
                 expected_input_manifest_sha256: str) -> None:
        require(result_dir.is_absolute(), "v6 result absolute path")
        reject_symlink_chain(result_dir, "v6 result")
        self.path = result_dir
        self.directory_fd = os.open(
            os.fspath(result_dir), os.O_RDONLY |
            getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        self.directory_identity = identity(os.fstat(self.directory_fd))
        self.descriptors: dict[str, int] = {}
        self.identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}
        self.payloads: dict[str, bytes] = {}
        try:
            complete_fd = os.open(
                "COMPLETE.json", os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                getattr(os, "O_NOFOLLOW", 0), dir_fd=self.directory_fd)
            metadata = os.fstat(complete_fd)
            require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
                    and 0 < metadata.st_size <= MAX_JSON,
                    "v6 COMPLETE identity")
            complete_payload = pread_exact(
                complete_fd, metadata.st_size, "v6 COMPLETE")
            require(sha256(complete_payload) == expected_complete_sha256,
                    "v6 externally pinned COMPLETE digest")
            self.descriptors["COMPLETE.json"] = complete_fd
            self.identities["COMPLETE.json"] = identity(metadata)
            self.payloads["COMPLETE.json"] = complete_payload
            complete = strict_json(complete_payload, "v6 COMPLETE")
            require(complete.get("schema") == V6_COMPLETE_SCHEMA and
                    complete.get("status") == V6_RESULT_STATUS and
                    complete.get("positive_claim_authority") is False,
                    "v6 completed result status")
            require(complete.get("source_root_sha256") ==
                    expected_v6_source_root_sha256,
                    "v6 completed source root")
            rows = complete.get("members")
            require(isinstance(rows, list) and rows,
                    "v6 COMPLETE member rows")
            require(complete.get("members_root_sha256") ==
                    sha256(canonical_json(rows)), "v6 member-root seal")
            clone = dict(complete)
            claimed = clone.pop("completion_claim_sha256", None)
            require(isinstance(claimed, str) and
                    sha256(canonical_json(clone)) == claimed,
                    "v6 completion claim seal")
            names: list[str] = []
            inodes = {(metadata.st_dev, metadata.st_ino)}
            for row in rows:
                require(isinstance(row, dict) and
                        set(row) == {"name", "bytes", "sha256"},
                        "v6 result member row")
                name = row["name"]
                require(isinstance(name, str) and name and
                        name not in {".", "..", "COMPLETE.json"} and
                        "/" not in name and "\\" not in name and
                        name not in names, "v6 result unique safe member")
                descriptor = os.open(
                    name, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                    getattr(os, "O_NOFOLLOW", 0), dir_fd=self.directory_fd)
                item = os.fstat(descriptor)
                require(stat.S_ISREG(item.st_mode) and item.st_nlink == 1 and
                        item.st_size == row["bytes"],
                        f"v6 result member identity: {name}")
                inode = (item.st_dev, item.st_ino)
                require(inode not in inodes, f"v6 result inode alias: {name}")
                inodes.add(inode)
                payload = pread_exact(descriptor, item.st_size, name)
                require(sha256(payload) == row["sha256"],
                        f"v6 result member digest: {name}")
                self.descriptors[name] = descriptor
                self.identities[name] = identity(item)
                self.payloads[name] = payload
                names.append(name)
            require(names == sorted(names, key=lambda item: item.encode("utf-8")),
                    "v6 result canonical member order")
            entries = list(os.scandir(self.directory_fd))
            require({entry.name for entry in entries} ==
                    set(names) | {"COMPLETE.json"} and
                    all(entry.is_file(follow_symlinks=False)
                        for entry in entries),
                    "v6 completed exact member closure")
            require(set(names) == {
                "COARSE.bin", "DECODER_RECEIPT.json", "ENCODER_RECEIPT.json",
                "INPUT_BINDING.json", "RESULT.json", "RUNTIME_RECEIPT.json",
                "SMOKE_BINDING.json",
            }, "v6 expected result member grammar")
            require(len(self.payloads["COARSE.bin"]) == 1_414_656 and
                    sha256(self.payloads["COARSE.bin"]) ==
                    complete["frame_sha256"], "v6 coarse frame binding")
            input_binding = strict_json(
                self.payloads["INPUT_BINDING.json"], "v6 input binding")
            require(input_binding.get("manifest_sha256") ==
                    expected_input_manifest_sha256,
                    "v6 completed input-manifest binding")
            require(input_binding.get("geometry") ==
                    {"intermediate": 768, "hidden": 2048},
                    "v6 completed Qwen pilot geometry")
            roles = input_binding.get("roles")
            require(isinstance(roles, list) and
                    [row.get("role") for row in roles] ==
                    ["gate", "up", "down_transposed"],
                    "v6 completed role order")
            result = strict_json(self.payloads["RESULT.json"], "v6 result")
            require(result.get("status") == V6_RESULT_STATUS and
                    result.get("frame_sha256") == complete["frame_sha256"] and
                    result.get("input_manifest_sha256") ==
                    expected_input_manifest_sha256 and
                    result.get("source_closure", {}).get(
                        "source_root_sha256") == expected_v6_source_root_sha256,
                    "v6 RESULT binding")
            require(result.get("literal_aggregate_reencode_matches") is True and
                    result.get("target_eligible_exact_307_over_128") is True,
                    "v6 result exact decode/rate gate")
            self.complete = complete
            self.input_binding = input_binding
            self.result = result
            self.complete_sha256 = expected_complete_sha256
        except BaseException:
            self.close()
            raise

    @property
    def coarse(self) -> bytes:
        return self.payloads["COARSE.bin"]

    def receipt(self) -> dict[str, Any]:
        return {
            "v6_complete_sha256": self.complete_sha256,
            "v6_coarse_sha256": sha256(self.coarse),
            "v6_coarse_bytes": len(self.coarse),
            "v6_result_status": self.result["status"],
            "v6_input_manifest_sha256":
                self.input_binding["manifest_sha256"],
            "retained_completed_result_descriptors": True,
        }

    def verify_final(self) -> None:
        require(identity(os.fstat(self.directory_fd)) ==
                self.directory_identity, "v6 result directory changed")
        require(identity(os.stat(self.path, follow_symlinks=False)) ==
                self.directory_identity, "v6 result path rebound")
        for name, descriptor in self.descriptors.items():
            require(identity(os.fstat(descriptor)) == self.identities[name],
                    f"v6 result changed: {name}")
            require(identity(os.stat(name, dir_fd=self.directory_fd,
                                     follow_symlinks=False)) ==
                    self.identities[name], f"v6 result rebound: {name}")
            require(pread_exact(descriptor, len(self.payloads[name]), name) ==
                    self.payloads[name], f"v6 result bytes changed: {name}")

    def close(self) -> None:
        for descriptor in list(getattr(self, "descriptors", {}).values()):
            try:
                os.close(descriptor)
            except OSError:
                pass
        getattr(self, "descriptors", {}).clear()
        descriptor = getattr(self, "directory_fd", None)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self.directory_fd = None

    def __enter__(self) -> "HeldCompletedV6Result":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def require_finite_bf16(payload: bytes, label: str) -> None:
    require(len(payload) % 2 == 0 and
            all((word & 0x7F80) != 0x7F80
                for (word,) in struct.iter_unpack("<H", payload)),
            f"{label}: finite canonical BF16 words")


def authenticate_inputs(
    manifest_path: Path, *, expected_manifest_sha256: str,
    expected_v6_binding: dict[str, Any],
) -> dict[str, Any]:
    payload = read_held_regular(
        manifest_path, expected_sha256=expected_manifest_sha256,
        maximum_bytes=1 << 20, label="expert input manifest")
    record = strict_json(payload, "expert input manifest")
    require(set(record) == {
        "schema", "geometry", "roles", "output_directory_name"},
        "input manifest exact schema")
    require(record["schema"] == INPUT_SCHEMA and
            record["geometry"] == {"intermediate": 768, "hidden": 2048},
            "input manifest Qwen pilot geometry")
    rows = record["roles"]
    require(isinstance(rows, list) and len(rows) == 3,
            "input role rows")
    role_bytes: dict[str, bytes] = {}
    bindings: list[dict[str, Any]] = []
    root = manifest_path.parent
    expected_bytes = 2 * 768 * 2048
    for ordinal, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == {
            "role", "relative_path", "bytes", "sha256"},
            "input role schema")
        role = ("gate", "up", "down_transposed")[ordinal]
        require(row["role"] == role, "canonical input role order")
        relative = row["relative_path"]
        require(isinstance(relative, str) and relative and
                not Path(relative).is_absolute() and
                ".." not in Path(relative).parts,
                "input relative path")
        require(row["bytes"] == expected_bytes and
                HEX64.fullmatch(row["sha256"]) is not None,
                "input role bytes/digest")
        member = read_held_regular(
            root / relative, expected_bytes=expected_bytes,
            expected_sha256=row["sha256"], maximum_bytes=MAX_ROLE_BYTES,
            label=f"input role {role}")
        require_finite_bf16(member, f"input role {role}")
        role_bytes[role] = member
        bindings.append({
            "role": role, "bytes": len(member), "sha256": sha256(member)})
    require(bindings == expected_v6_binding.get("roles"),
            "input bytes equal authenticated v6 input binding")
    return {
        "manifest_sha256": sha256(payload),
        "role_bytes": role_bytes,
        "bindings": bindings,
        "geometry": record["geometry"],
        "v6_output_directory_name": record["output_directory_name"],
        "identity_fields_available_to_codec": False,
    }
