#!/usr/bin/env python3
"""Stdlib-only authenticated v8 production-child bootstrap.

This entrypoint cannot run production directly.  It requires a one-record
Unix-domain capability inherited from the exact authenticated preflight
parent, proves the peer/parent/child/process-command identities, consumes the
record to EOF, acknowledges it once, and only then descriptor-executes the
authenticated NumPy/CuPy scientific core.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import struct
import sys
import types
from pathlib import Path
from typing import Any

try:
    import fcntl  # type: ignore
except ImportError:  # Windows source inspection only; production is Linux-only.
    fcntl = None  # type: ignore


STAGE_MEMBERS = {
    "authorization_contract.json",
    "audit_lock_entrypoint.py",
    "launch_manifest.json",
    "lossy_tail_core.py",
    "lossy_tail_oracle.py",
    "preflight_launch.py",
    "protocol_lock.json",
    "repair_lock.json",
    "runtime_calibrate.py",
    "runtime_contract.json",
    "source_bindings.json",
}
CORE_FLAGS = [
    "--bindings", "--protocol", "--repair-lock", "--runtime-contract",
    "--authorization-contract", "--launch-manifest", "--launch-manifest-sha256",
    "--authorization", "--authorization-sha256", "--control-replicates",
    "--maximum-coordinate-passes",
]


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(f"V8_CHILD_BOOTSTRAP_REJECT: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_pairs)
    except Exception as exc:
        fail(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        fail(f"{label} key drift")


def checked_path(raw: str, label: str) -> Path:
    if not raw or "\x00" in raw or not os.path.isabs(raw) or raw != os.path.normpath(raw):
        fail(f"{label} must be absolute and lexically canonical")
    path = Path(raw)
    probe = Path(path.parts[0])
    for component in path.parts[1:]:
        probe = probe / component
        try:
            metadata = os.lstat(probe)
        except FileNotFoundError:
            fail(f"{label} component missing: {probe}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains a symlink component: {probe}")
    if os.path.realpath(raw) != raw:
        fail(f"{label} does not name its actual target")
    return path


def read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"{label} is not regular")
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        payload = b"".join(chunks)
        if metadata.st_size != 0 and len(payload) != metadata.st_size:
            fail(f"{label} changed during read")
        return payload
    finally:
        os.close(descriptor)


def valid_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        fail(f"invalid SHA-256 for {label}")
    return value


def validate_stage(stage: Path, manifest_path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest_payload = read_regular(manifest_path, "launch manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != expected_sha:
        fail("launch-manifest file SHA-256 mismatch")
    manifest = strict_json(manifest_payload, "launch manifest")
    exact_keys(manifest, {
        "schema", "status", "allowed_members", "members", "source_audit_invocation",
        "runtime_calibration_invocation_after_independent_source_pass_only",
        "production_invocation_after_independent_runtime_receipt_audit_and_separate_authorization_only",
        "production_child_grammar", "authorization",
    }, "launch manifest")
    if manifest.get("schema") != "lossy-tail-v8-launch-manifest-v1":
        fail("launch-manifest schema mismatch")
    if manifest.get("status") != "FROZEN_V8_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        fail("launch-manifest status mismatch")
    allowed = manifest.get("allowed_members")
    if not isinstance(allowed, list) or len(allowed) != len(STAGE_MEMBERS) or len(set(allowed)) != len(allowed) or set(allowed) != STAGE_MEMBERS:
        fail("allowed-member closure mismatch")
    observed: set[str] = set()
    with os.scandir(stage) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                fail(f"forbidden stage entry: {entry.name}")
            observed.add(entry.name)
    if observed != STAGE_MEMBERS:
        fail("stage closure mismatch")
    rows = manifest.get("members")
    expected = STAGE_MEMBERS - {"launch_manifest.json"}
    if not isinstance(rows, list) or len(rows) != len(expected):
        fail("manifest member-row cardinality mismatch")
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(paths) != len(rows) or len(set(paths)) != len(paths) or set(paths) != expected:
        fail("manifest member-row closure mismatch")
    payloads = {"launch_manifest.json": manifest_payload}
    for row in rows:
        exact_keys(row, {"path", "bytes", "sha256"}, f"manifest row {row.get('path')}")
        member = read_regular(stage / row["path"], f"stage member {row['path']}")
        if len(member) != row["bytes"] or hashlib.sha256(member).hexdigest() != row["sha256"]:
            fail(f"stage member identity mismatch: {row['path']}")
        payloads[row["path"]] = member
    return manifest, payloads


def descriptor_payload(descriptor: int, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1 << 20, size - offset), offset)
        if not block:
            fail(f"short descriptor read for {label}")
        chunks.append(block)
        offset += len(block)
    return b"".join(chunks)


def required_memfd_seals() -> int:
    if fcntl is None:
        fail("Linux fcntl seals are required")
    return sum(int(getattr(fcntl, name)) for name in ("F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE"))


def verify_inherited_preflight(
    descriptor: int,
    peer_pid: int,
    expected_payload: bytes,
    capability: dict[str, Any],
) -> None:
    if fcntl is None:
        fail("Linux fcntl seals are required")
    os.set_inheritable(descriptor, False)
    local = os.fstat(descriptor)
    parent = os.stat(f"/proc/{peer_pid}/fd/{descriptor}")
    seals = int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
    payload = descriptor_payload(descriptor, local.st_size, "inherited preflight memfd")
    expected = {
        "preflight_memfd_fd": descriptor,
        "preflight_memfd_st_dev": local.st_dev,
        "preflight_memfd_st_ino": local.st_ino,
        "preflight_memfd_bytes": local.st_size,
        "preflight_memfd_seals": seals,
        "preflight_memfd_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if (
        not stat.S_ISREG(local.st_mode)
        or (parent.st_dev, parent.st_ino) != (local.st_dev, local.st_ino)
        or seals & required_memfd_seals() != required_memfd_seals()
        or payload != expected_payload
        or any(capability.get(key) != value for key, value in expected.items())
    ):
        fail("immutable authenticated preflight descriptor mismatch")


def verify_inherited_output_parent(
    descriptor: int,
    peer_pid: int,
    capability: dict[str, Any],
) -> None:
    os.set_inheritable(descriptor, False)
    local = os.fstat(descriptor)
    parent = os.stat(f"/proc/{peer_pid}/fd/{descriptor}")
    if (
        not stat.S_ISDIR(local.st_mode)
        or (parent.st_dev, parent.st_ino) != (local.st_dev, local.st_ino)
        or capability.get("output_parent_fd") != descriptor
        or capability.get("output_parent_st_dev") != local.st_dev
        or capability.get("output_parent_st_ino") != local.st_ino
    ):
        fail("held output-parent descriptor mismatch")


def consume_capability(
    descriptor: int,
    *,
    preflight_descriptor: int,
    output_parent_descriptor: int,
    stage: Path,
    payloads: dict[str, bytes],
    manifest_sha: str,
    authorization_path: str,
    authorization_sha: str,
) -> dict[str, Any]:
    try:
        os.set_inheritable(descriptor, False)
        channel = socket.socket(fileno=descriptor)
    except Exception as exc:
        fail(f"capability fd is not an inherited socket: {exc}")
    try:
        if channel.family != socket.AF_UNIX or channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET:
            fail("capability must be an inherited Unix SOCK_SEQPACKET channel")
        if not hasattr(socket, "SO_PEERCRED"):
            fail("SO_PEERCRED is required")
        peer_payload = channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        peer_pid, peer_uid, peer_gid = struct.unpack("3i", peer_payload)
        if peer_pid != os.getppid() or peer_uid != os.getuid() or peer_gid != os.getgid():
            fail("capability peer credentials do not equal the live parent")
        channel.settimeout(15.0)
        message = channel.recv(65536)
        if not message:
            fail("capability record missing")
        if channel.recv(1) != b"":
            fail("capability channel contains more than one record")
        capability = strict_json(message, "child capability")
        exact_keys(capability, {
            "schema", "status", "parent_pid", "child_pid", "nonce_hex",
            "preflight_memfd_fd", "preflight_memfd_st_dev", "preflight_memfd_st_ino",
            "preflight_memfd_bytes", "preflight_memfd_seals", "preflight_memfd_sha256",
            "output_parent_fd", "output_parent_st_dev", "output_parent_st_ino",
            "supplemental_parent_executable_st_dev", "supplemental_parent_executable_st_ino",
            "launch_manifest_sha256",
            "authorization_file_sha256", "authorization_internal_sha256",
            "bootstrap_sha256", "scientific_core_sha256",
        }, "child capability")
        if capability["schema"] != "lossy-tail-v8-one-use-child-capability-v1" or capability["status"] != "ISSUED_ONCE_BY_AUTHENTICATED_PREFLIGHT":
            fail("capability schema/status mismatch")
        if capability["parent_pid"] != peer_pid or capability["child_pid"] != os.getpid():
            fail("capability process identity mismatch")
        nonce = capability["nonce_hex"]
        if not isinstance(nonce, str) or len(nonce) != 64 or any(ch not in "0123456789abcdef" for ch in nonce):
            fail("capability nonce malformed")
        for label in (
            "preflight_memfd_sha256", "launch_manifest_sha256", "authorization_file_sha256",
            "authorization_internal_sha256", "bootstrap_sha256", "scientific_core_sha256",
        ):
            valid_hash(capability[label], label)
        verify_inherited_preflight(
            preflight_descriptor, peer_pid, payloads["preflight_launch.py"], capability,
        )
        verify_inherited_output_parent(output_parent_descriptor, peer_pid, capability)
        supplemental_parent_executable = os.stat(f"/proc/{peer_pid}/exe")
        expected_executable = os.stat(sys.executable)
        if (
            (supplemental_parent_executable.st_dev, supplemental_parent_executable.st_ino)
            != (expected_executable.st_dev, expected_executable.st_ino)
            or capability["supplemental_parent_executable_st_dev"] != supplemental_parent_executable.st_dev
            or capability["supplemental_parent_executable_st_ino"] != supplemental_parent_executable.st_ino
        ):
            fail("supplemental parent executable identity mismatch")
        if capability["launch_manifest_sha256"] != manifest_sha:
            fail("capability launch-manifest identity mismatch")
        if capability["authorization_file_sha256"] != authorization_sha:
            fail("capability authorization-file identity mismatch")
        if capability["bootstrap_sha256"] != hashlib.sha256(payloads["lossy_tail_oracle.py"]).hexdigest():
            fail("capability bootstrap identity mismatch")
        if capability["scientific_core_sha256"] != hashlib.sha256(payloads["lossy_tail_core.py"]).hexdigest():
            fail("capability scientific-core identity mismatch")
        capability_sha = hashlib.sha256(canonical(capability)).hexdigest()
        acknowledgement = {
            "schema": "lossy-tail-v8-child-capability-ack-v1",
            "status": "CONSUMED_ONCE_BEFORE_THIRD_PARTY_IMPORT",
            "child_pid": os.getpid(),
            "capability_sha256": capability_sha,
        }
        channel.send(canonical(acknowledgement))
        capability["capability_sha256"] = capability_sha
        return capability
    finally:
        channel.close()


def main() -> None:
    raw = sys.argv[1:]
    expected_flags = [*CORE_FLAGS, "--preflight-memfd-fd", "--output-parent-fd", "--capability-fd"]
    if len(raw) != 2 * len(expected_flags) or raw[::2] != expected_flags:
        fail("invalid frozen v8 child grammar")
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize != 0:
        fail("requires python -B -I without optimization")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        fail("requires explicit CUDA_VISIBLE_DEVICES=0 before scientific-core import")
    launcher = checked_path(sys.argv[0], "raw argv0")
    module_file = os.fspath(Path(__file__))
    if launcher.name != "lossy_tail_oracle.py" or sys.argv[0] != module_file:
        fail("raw argv0 must exactly equal child-bootstrap __file__")
    left = os.stat(launcher, follow_symlinks=False)
    right = os.stat(module_file, follow_symlinks=False)
    if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
        fail("raw argv0 and child-bootstrap identities differ")
    stage = checked_path(os.fspath(launcher.parent), "stage")

    values = {raw[index]: raw[index + 1] for index in range(0, len(raw), 2)}
    manifest_path = checked_path(values["--launch-manifest"], "launch manifest")
    if manifest_path != stage / "launch_manifest.json":
        fail("launch manifest must use exact immediate-stage spelling")
    manifest_sha = valid_hash(values["--launch-manifest-sha256"].lower(), "launch manifest")
    authorization_path = os.fspath(checked_path(values["--authorization"], "authorization"))
    authorization_sha = valid_hash(values["--authorization-sha256"].lower(), "authorization")
    if values["--control-replicates"] != "4" or values["--maximum-coordinate-passes"] != "4":
        fail("v8 child requires controls=4 and passes=4")
    descriptors: dict[str, int] = {}
    for flag in ("--preflight-memfd-fd", "--output-parent-fd", "--capability-fd"):
        try:
            descriptors[flag] = int(values[flag], 10)
        except ValueError:
            fail(f"{flag} must be a canonical decimal integer")
        if descriptors[flag] < 3 or str(descriptors[flag]) != values[flag]:
            fail(f"{flag} must be a canonical inherited descriptor >=3")
    if len(set(descriptors.values())) != len(descriptors):
        fail("inherited child descriptors must be pairwise distinct")

    _, payloads = validate_stage(stage, manifest_path, manifest_sha)
    capability = consume_capability(
        descriptors["--capability-fd"],
        preflight_descriptor=descriptors["--preflight-memfd-fd"],
        output_parent_descriptor=descriptors["--output-parent-fd"],
        stage=stage,
        payloads=payloads,
        manifest_sha=manifest_sha,
        authorization_path=authorization_path,
        authorization_sha=authorization_sha,
    )

    core_path = stage / "lossy_tail_core.py"
    core_module = types.ModuleType("lossy_tail_v8_authenticated_production_core")
    core_module.__file__ = os.fspath(core_path)
    core_module.__package__ = None
    core_module.__dict__["__V8_CORE_CONTEXT__"] = {
        "schema": "lossy-tail-v8-core-context-v1",
        "mode": "production_child",
        "parent_pid": capability["parent_pid"],
        "child_pid": capability["child_pid"],
        "capability_sha256": capability["capability_sha256"],
        "preflight_memfd_fd": capability["preflight_memfd_fd"],
        "preflight_memfd_st_dev": capability["preflight_memfd_st_dev"],
        "preflight_memfd_st_ino": capability["preflight_memfd_st_ino"],
        "preflight_memfd_bytes": capability["preflight_memfd_bytes"],
        "preflight_memfd_seals": capability["preflight_memfd_seals"],
        "preflight_memfd_sha256": capability["preflight_memfd_sha256"],
        "output_parent_fd": capability["output_parent_fd"],
        "output_parent_st_dev": capability["output_parent_st_dev"],
        "output_parent_st_ino": capability["output_parent_st_ino"],
        "launch_manifest_sha256": capability["launch_manifest_sha256"],
        "authorization_file_sha256": capability["authorization_file_sha256"],
        "authorization_internal_sha256": capability["authorization_internal_sha256"],
    }
    sys.modules[core_module.__name__] = core_module
    sys.argv = [os.fspath(launcher), *raw[: 2 * len(CORE_FLAGS)]]
    exec(compile(payloads["lossy_tail_core.py"], os.fspath(core_path), "exec"), core_module.__dict__)
    core_module.main()


if __name__ == "__main__":
    main()
