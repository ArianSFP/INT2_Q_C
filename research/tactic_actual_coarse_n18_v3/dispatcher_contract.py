#!/usr/bin/env python3
"""Opaque independent-dispatch assertion ABI; this package issues no authority."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from typing import Any

from v3_common import (
    DISPATCH_SCHEMA,
    exact_keys,
    require,
    strict_json_loads,
    valid_sha256,
)


MAX_DISPATCH_BYTES = 64 << 10
ACTIONS = ("synthetic", "pilot", "full")


@dataclass(frozen=True)
class DispatcherAssertion:
    action: str
    source_root: str
    runtime_lock_sha256: str
    audit_evidence_sha256: str
    dispatcher_nonce: str
    authority_kind: str
    held_fd_identity: tuple[int, int, int, int, int, int]


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def consume_dispatcher_assertion(
    inherited_fd: int,
    *,
    expected_action: str,
    expected_source_root: str,
    expected_runtime_lock_sha256: str,
) -> DispatcherAssertion:
    """Consume a held assertion already authenticated by an external dispatcher.

    There is intentionally no producer-generated token, self-hash, review seal,
    reviewer key or PASS constructor here. The external dispatcher owns the FD,
    expected roots and decision; this package merely checks their binding.
    """

    require(os.name == "posix", "dispatcher assertion requires POSIX")
    require(type(inherited_fd) is int and inherited_fd >= 3, "inherited dispatcher FD")
    require(expected_action in ACTIONS, "dispatcher action")
    require(valid_sha256(expected_source_root, nonzero=True), "dispatcher expected source root")
    require(valid_sha256(expected_runtime_lock_sha256, nonzero=True), "dispatcher expected runtime lock")
    before = os.fstat(inherited_fd)
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_DISPATCH_BYTES, "dispatcher FD type/size")
    raw = os.pread(inherited_fd, before.st_size + 1, 0)
    require(len(raw) == before.st_size, "dispatcher FD exact bytes")
    after = os.fstat(inherited_fd)
    require(_identity(before) == _identity(after), "dispatcher FD identity drift")
    value = exact_keys(
        strict_json_loads(raw),
        {
            "schema",
            "status",
            "action",
            "source_root",
            "runtime_lock_sha256",
            "audit_evidence_sha256",
            "dispatcher_nonce",
            "authority_kind",
        },
        "dispatcher assertion",
    )
    require(value["schema"] == DISPATCH_SCHEMA, "dispatcher assertion schema")
    require(value["status"] == "EXTERNALLY_AUTHENTICATED_DISPATCH_ASSERTION", "dispatcher assertion status")
    require(value["action"] == expected_action, "dispatcher action binding")
    require(value["source_root"] == expected_source_root, "dispatcher source-root binding")
    require(value["runtime_lock_sha256"] == expected_runtime_lock_sha256, "dispatcher runtime binding")
    require(valid_sha256(value["audit_evidence_sha256"], nonzero=True), "dispatcher audit evidence")
    require(
        isinstance(value["dispatcher_nonce"], str)
        and 16 <= len(value["dispatcher_nonce"].encode("utf-8")) <= 256,
        "dispatcher nonce",
    )
    require(
        value["authority_kind"]
        == "opaque-held-fd-assertion-authenticated-outside-producer-package",
        "external authority kind",
    )
    return DispatcherAssertion(
        value["action"],
        value["source_root"],
        value["runtime_lock_sha256"],
        value["audit_evidence_sha256"],
        value["dispatcher_nonce"],
        value["authority_kind"],
        _identity(before),
    )


def no_standalone_authority() -> None:
    raise RuntimeError(
        "v3 cannot self-authorize review or payload execution; an independent dispatcher must supply a held assertion FD"
    )


if __name__ == "__main__":
    no_standalone_authority()
