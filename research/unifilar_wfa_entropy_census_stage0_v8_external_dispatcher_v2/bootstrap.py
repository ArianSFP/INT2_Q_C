#!/usr/bin/env python3
"""Independent, fail-closed UWFA-SC v8 production dispatcher.

This file is deliberately outside the producer tree.  The checked-in v2 pins
are unresolved, so direct execution stops before any request or payload path
access.  Source-only tests exercise the underlying authority and accounting
primitives with synthetic files; they cannot override production pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import marshal
import os
import platform
import re
import stat
import subprocess
import sys
import types
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


class DispatchError(RuntimeError):
    """A fail-closed dispatcher contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchError(message)


def sha256(data: bytes) -> str:
    require(isinstance(data, bytes), "SHA-256 input must be bytes")
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes, *, label: str = "JSON") -> dict[str, Any]:
    require(isinstance(data, bytes), f"{label} bytes")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise DispatchError(f"{label} nonfinite constant: {value}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject_constant)
    except DispatchError:
        raise
    except Exception as exc:
        raise DispatchError(f"{label} invalid: {exc}") from exc
    require(isinstance(value, dict), f"{label} root must be object")
    return value


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except Exception as exc:
        raise DispatchError(f"canonical JSON failure: {exc}") from exc


def exact_fields(
    value: Any,
    names: Iterable[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    expected = set(names)
    require(isinstance(value, dict), f"{label} object")
    require(set(value) == expected, f"{label} fields: {sorted(set(value) ^ expected)}")
    return value


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")
_TXID = re.compile(r"[0-9a-f]{32}\Z")
_PLACEHOLDER_PREFIX = "__UNRESOLVED_"


def digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and _HEX64.fullmatch(value) is not None, f"{label} digest")
    return value


def public_commit(value: Any, label: str = "public commit") -> str:
    require(isinstance(value, str) and _HEX40.fullmatch(value) is not None, f"{label} geometry")
    return value


def safe_name(value: Any, label: str) -> str:
    require(isinstance(value, str) and _SAFE_NAME.fullmatch(value) is not None, f"{label} safe name")
    require(value not in {".", ".."} and "/" not in value and "\\" not in value, f"{label} leaf")
    return value


def length_prefixed_root(domain: bytes, rows: Sequence[bytes | str | int]) -> str:
    require(isinstance(domain, bytes) and domain, "root domain")
    h = hashlib.sha256()
    h.update(len(domain).to_bytes(8, "little"))
    h.update(domain)
    for row in rows:
        if isinstance(row, str):
            raw = row.encode("utf-8")
        elif isinstance(row, int) and type(row) is int and row >= 0:
            width = max(1, (row.bit_length() + 7) // 8)
            raw = row.to_bytes(width, "little")
        else:
            require(isinstance(row, bytes), "root row type")
            raw = bytes(row)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


# These values are production constants, never CLI/environment settings.  V0
# intentionally cannot launch until a separately reviewed lifecycle transition
# replaces the remaining runtime/decoder placeholders and seals a new
# dispatcher manifest.  This file never self-pins that manifest or its audit;
# a separately reviewed out-of-tree launcher supplies those trust anchors.
PINNED_PRODUCER_MANIFEST_SHA256 = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"
PINNED_PRODUCER_REVIEW_SHA256 = "57e19b93f9f771381945a42060e9b15b71962f8af4b7800fd71be0c1949c2cce"
PINNED_PUBLIC_GIT_COMMIT = "d563c4ac1e78a6b6e7f0722291211d1209f775af"
PINNED_RUNTIME_LOCK_SHA256 = "__UNRESOLVED_RUNTIME_LOCK_SHA256__"
PINNED_DECODER_BUNDLE_SHA256 = "__UNRESOLVED_DECODER_BUNDLE_SHA256__"

# A later controls dispatcher may reach this producer-side terminal status,
# but only an independent result audit may turn it into a scientific claim.
MAXIMUM_PRE_AUDIT_PRODUCER_STATUS = "PASS_MATCHED_NULL_SPECIFICITY_AWAITING_EXTERNAL_RESULT_AUDIT"


@dataclass(frozen=True)
class ProductionPins:
    producer_manifest_sha256: str
    producer_review_sha256: str
    public_git_commit: str
    runtime_lock_sha256: str
    decoder_bundle_sha256: str

    @classmethod
    def embedded(cls) -> "ProductionPins":
        values = (
            PINNED_PRODUCER_MANIFEST_SHA256,
            PINNED_PRODUCER_REVIEW_SHA256,
            PINNED_PUBLIC_GIT_COMMIT,
            PINNED_RUNTIME_LOCK_SHA256,
            PINNED_DECODER_BUNDLE_SHA256,
        )
        require(not any(value.startswith(_PLACEHOLDER_PREFIX) for value in values), "BLOCK_UNRESOLVED_EMBEDDED_PRODUCTION_PINS")
        return cls(
            digest(values[0], "producer manifest pin"),
            digest(values[1], "producer review pin"),
            public_commit(values[2]),
            digest(values[3], "runtime lock pin"),
            digest(values[4], "decoder bundle pin"),
        )


@dataclass(frozen=True)
class ExternalLaunchAuthority:
    """Trust anchors supplied by a separately pinned out-of-tree launcher.

    A dispatcher cannot honestly embed the digest of an audit that reviews its
    own bytes: doing so changes those bytes and creates an infinite pin cycle.
    The human/deployment command must authenticate the tiny launcher (or pass
    its held fd from another trust domain); that launcher then constructs this
    typed value from its reviewed constants.
    """

    dispatcher_manifest_sha256: str
    dispatcher_audit_sha256: str
    dispatcher_public_git_commit: str
    launcher_source_sha256: str
    launcher_review_sha256: str
    request_sha256: str
    baseline_plan_sha256: str
    legacy_independent_audit_sha256: str
    original_source_binding_sha256: str
    native_audit_event_fd: int
    native_audit_session_nonce: str

    def __post_init__(self) -> None:
        digest(self.dispatcher_manifest_sha256, "external dispatcher manifest pin")
        digest(self.dispatcher_audit_sha256, "external dispatcher audit pin")
        public_commit(self.dispatcher_public_git_commit, "external dispatcher commit")
        digest(self.launcher_source_sha256, "external launcher source pin")
        digest(self.launcher_review_sha256, "external launcher review pin")
        digest(self.request_sha256, "external exact request pin")
        digest(self.baseline_plan_sha256, "external baseline plan pin")
        digest(self.legacy_independent_audit_sha256, "external independently reviewed legacy audit pin")
        digest(self.original_source_binding_sha256, "external independently reviewed original-source binding pin")
        require(type(self.native_audit_event_fd) is int and self.native_audit_event_fd >= 3, "external native-audit event fd")
        digest(self.native_audit_session_nonce, "external native-audit session nonce")


EXPECTED_PRODUCER_MEMBERS = frozenset({
    "INDEPENDENT_BOOTSTRAP_ABI.md",
    "README.md",
    "container_codec.py",
    "cupy_backend.py",
    "design_lock.json",
    "dispatcher_contract.py",
    "fixture_long_memory.py",
    "fixture_portability.py",
    "protocol.py",
    "result_envelope.py",
    "run_source_free_gpu_dev.py",
    "stage0_census.py",
    "strata_sc_adapter.py",
    "test_source_only.py",
    "universal_adapter.py",
    "uwfa_common.py",
    "verify_source.py",
})

SNAPSHOT_DEPENDENCY_ORDER = (
    "uwfa_common.py",
    "protocol.py",
    "universal_adapter.py",
    "container_codec.py",
    "stage0_census.py",
    "cupy_backend.py",
    "result_envelope.py",
    "strata_sc_adapter.py",
)

SNAPSHOT_MODULE_NAMES = {
    "uwfa_common.py": "uwfa_v8_dispatch_common",
    "protocol.py": "uwfa_v8_dispatch_protocol",
    "universal_adapter.py": "uwfa_v8_dispatch_semantic",
    "container_codec.py": "uwfa_v8_dispatch_container",
    "stage0_census.py": "uwfa_v8_dispatch_stage",
    "cupy_backend.py": "uwfa_v8_dispatch_cupy_backend",
    "result_envelope.py": "uwfa_v8_dispatch_result_envelope",
    "strata_sc_adapter.py": "uwfa_v8_dispatch_strata_adapter",
}

PRELOADED_PRODUCER_ALIASES = frozenset(
    set(SNAPSHOT_MODULE_NAMES.values())
    | {Path(name).stem for name in EXPECTED_PRODUCER_MEMBERS if name.endswith(".py")}
)

# The decoder bundle is not allowed to choose which producer snapshot satisfies
# a logical role.  This injective map is dispatcher source and manifest bound.
EXPECTED_LOGICAL_TO_PRODUCER_MEMBER = {
    "fixed_strata_sc_adapter": "strata_sc_adapter.py",
    "universal_semantic_adapter": "universal_adapter.py",
}


def validate_logical_to_producer_member_map(value: Any) -> dict[str, str]:
    require(isinstance(value, dict), "decoder logical-to-producer member map")
    require(value == EXPECTED_LOGICAL_TO_PRODUCER_MEMBER, "decoder exact logical-to-producer member map")
    require(all(type(key) is str and type(member) is str for key, member in value.items()), "decoder logical map strings")
    require(len(set(value.values())) == len(value), "decoder logical-to-producer map must be injective")
    return dict(value)


def require_isolated_cpython() -> None:
    require(platform.python_implementation() == "CPython", "dispatcher requires CPython")
    require(getattr(sys.flags, "isolated", 0) == 1, "dispatcher requires CPython -I")
    require(bool(sys.dont_write_bytecode), "dispatcher requires CPython -B")
    require(getattr(sys.flags, "safe_path", True), "dispatcher requires safe_path")


def _absolute_components(path: str, *, label: str) -> tuple[str, ...]:
    require(isinstance(path, str) and path.startswith("/"), f"{label} must be absolute POSIX path")
    require("\x00" not in path and "\\" not in path, f"{label} invalid characters")
    require(path == "/" or not path.endswith("/"), f"{label} trailing separator")
    parts = tuple(path.split("/")[1:])
    require(parts and all(part and part not in {".", ".."} for part in parts), f"{label} noncanonical")
    require("/" + "/".join(parts) == path, f"{label} normalized spelling")
    return parts


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
        int(info.st_nlink),
    )


def _inode(identity: Sequence[int]) -> tuple[int, int]:
    require(len(identity) >= 2, "inode identity tuple")
    return int(identity[0]), int(identity[1])


def reject_authority_request_output_inode_aliasing(
    *,
    authority: Mapping[str, Sequence[int]],
    request: Mapping[str, Sequence[int]],
    output: Mapping[str, Sequence[int]],
) -> None:
    """Reject hard-link/inode reuse both within and across trust domains."""

    categories = {"authority": authority, "request": request, "output": output}
    owned: dict[tuple[int, int], tuple[str, str]] = {}
    for category, rows in categories.items():
        require(isinstance(rows, Mapping), f"{category} inode map")
        for label, identity in rows.items():
            require(isinstance(label, str) and label, f"{category} inode label")
            key = _inode(identity)
            previous = owned.get(key)
            require(previous is None, f"inode aliasing rejected: {previous} aliases {(category, label)}")
            owned[key] = (category, label)


def _pread_exact(fd: int, size: int, *, cap: int, label: str) -> bytes:
    require(type(size) is int and 0 <= size <= cap, f"{label} bounded size")
    require(hasattr(os, "pread"), "production dispatcher requires os.pread")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1 << 20, size - offset), offset)
        require(bool(chunk), f"{label} short read")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


@dataclass
class HeldRegular:
    parent_fd: int
    name: str
    fd: int
    identity: tuple[int, int, int, int, int, int, int]
    data: bytes
    sha256: str
    closed: bool = False

    def verify_stable(self) -> None:
        require(not self.closed, f"closed held file: {self.name}")
        observed = os.fstat(self.fd)
        require(_identity(observed) == self.identity, f"held descriptor changed: {self.name}")
        try:
            named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except Exception as exc:
            raise DispatchError(f"held name unavailable: {self.name}: {exc}") from exc
        require(_identity(named) == self.identity, f"held name substituted: {self.name}")
        require(_pread_exact(self.fd, self.identity[3], cap=max(self.identity[3], 1), label=self.name) == self.data, f"held bytes changed: {self.name}")

    def close(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


@dataclass
class HeldDirectory:
    path: str
    fds: list[int]
    identity: tuple[int, int, int, int, int, int, int]
    members: list[HeldRegular] = field(default_factory=list)
    closed: bool = False

    @property
    def fd(self) -> int:
        require(not self.closed and bool(self.fds), "closed held directory")
        return self.fds[-1]

    @classmethod
    def open_absolute(cls, path: str, *, label: str) -> "HeldDirectory":
        require(os.name == "posix" and os.open in os.supports_dir_fd, "production descriptor traversal requires POSIX dir_fd")
        parts = _absolute_components(path, label=label)
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        cloexec = getattr(os, "O_CLOEXEC", 0)
        fds: list[int] = []
        try:
            current = os.open("/", os.O_RDONLY | directory_flag | nofollow | cloexec)
            fds.append(current)
            for part in parts:
                current = os.open(part, os.O_RDONLY | directory_flag | nofollow | cloexec, dir_fd=current)
                info = os.fstat(current)
                require(stat.S_ISDIR(info.st_mode), f"{label} component not directory")
                fds.append(current)
            return cls(path, fds, _identity(os.fstat(fds[-1])))
        except Exception:
            for fd in reversed(fds):
                os.close(fd)
            raise

    def enumerate_names(self) -> set[str]:
        require(not self.closed, "closed held directory")
        try:
            with os.scandir(self.fd) as rows:
                return {row.name for row in rows}
        except Exception as exc:
            raise DispatchError(f"held directory enumeration failed: {exc}") from exc

    def open_member(self, name: str, *, cap: int, label: str) -> HeldRegular:
        safe_name(name, label)
        require(not self.closed, "closed held directory")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        cloexec = getattr(os, "O_CLOEXEC", 0)
        fd = os.open(name, os.O_RDONLY | nofollow | cloexec, dir_fd=self.fd)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode), f"{label} not regular")
            require(info.st_nlink >= 1, f"{label} unlinked")
            data = _pread_exact(fd, int(info.st_size), cap=cap, label=label)
            held = HeldRegular(self.fd, name, fd, _identity(info), data, sha256(data))
            held.verify_stable()
            self.members.append(held)
            return held
        except Exception:
            os.close(fd)
            raise

    def verify_stable(self) -> None:
        require(not self.closed, "closed held directory")
        require(_identity(os.fstat(self.fd)) == self.identity, f"held directory changed: {self.path}")
        for member in self.members:
            member.verify_stable()

    def verify_same_directory_identity(self) -> None:
        """Permit intentional child creation while retaining the directory inode."""
        require(not self.closed, "closed held directory")
        observed = os.fstat(self.fd)
        require(
            (int(observed.st_dev), int(observed.st_ino)) == (self.identity[0], self.identity[1])
            and stat.S_ISDIR(observed.st_mode),
            f"held directory identity changed: {self.path}",
        )

    def close(self) -> None:
        if not self.closed:
            for member in reversed(self.members):
                member.close()
            self.members.clear()
            for fd in reversed(self.fds):
                os.close(fd)
            self.fds.clear()
            self.closed = True


@dataclass
class HeldAbsoluteFile:
    directory: HeldDirectory
    member: HeldRegular

    @classmethod
    def open(cls, path: str, *, cap: int, label: str) -> "HeldAbsoluteFile":
        parts = _absolute_components(path, label=label)
        parent = "/" if len(parts) == 1 else "/" + "/".join(parts[:-1])
        directory = HeldDirectory.open_absolute(parent, label=f"{label} parent")
        try:
            member = directory.open_member(parts[-1], cap=cap, label=label)
            return cls(directory, member)
        except Exception:
            directory.close()
            raise

    def verify_stable(self) -> None:
        self.directory.verify_stable()

    def close(self) -> None:
        self.directory.close()


def _manifest_rows(value: Any, *, expected_names: set[str] | frozenset[str], label: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    require(isinstance(value, list) and value, f"{label} rows")
    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        row = dict(exact_fields(raw, {"name", "bytes", "sha256"}, label=f"{label}[{index}]"))
        name = safe_name(row["name"], f"{label}[{index}].name")
        require(name not in by_name, f"{label} duplicate member")
        require(type(row["bytes"]) is int and 0 <= row["bytes"] <= (1 << 30), f"{label} member bytes")
        digest(row["sha256"], f"{label} member SHA-256")
        rows.append(row)
        by_name[name] = row
    require(set(by_name) == set(expected_names), f"{label} exact member set")
    require([row["name"] for row in rows] == sorted(expected_names, key=lambda value: value.encode("utf-8")), f"{label} canonical order")
    return rows, by_name


@dataclass
class ProducerClosure:
    package: HeldDirectory
    manifest: HeldRegular
    review: HeldAbsoluteFile
    members: dict[str, HeldRegular]
    manifest_record: dict[str, Any]
    manifest_rows: list[dict[str, Any]]
    source_snapshot_root_sha256: str

    def verify_stable(self) -> None:
        self.package.verify_stable()
        self.review.verify_stable()

    def close(self) -> None:
        self.review.close()
        self.package.close()


def authenticate_producer(
    *,
    package_path: str,
    review_path: str,
    manifest_pin: str,
    review_pin: str,
    expected_public_commit: str | None = None,
) -> ProducerClosure:
    manifest_pin = digest(manifest_pin, "producer manifest pin")
    review_pin = digest(review_pin, "producer review pin")
    package = HeldDirectory.open_absolute(package_path, label="producer package")
    review: HeldAbsoluteFile | None = None
    try:
        manifest = package.open_member("SOURCE_MANIFEST.json", cap=4 << 20, label="producer manifest")
        require(manifest.sha256 == manifest_pin, "producer manifest external pin mismatch")
        record = strict_json(manifest.data, label="producer manifest")
        exact_fields(record, {"schema", "status", "members", "access_attestation", "post_freeze_requirements"}, label="producer manifest")
        require(record["schema"] == "unifilar-wfa-source-manifest-v8", "producer manifest schema")
        require(record["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "producer manifest status")
        rows, by_name = _manifest_rows(record["members"], expected_names=EXPECTED_PRODUCER_MEMBERS, label="producer manifest")
        actual = package.enumerate_names()
        require(actual == set(EXPECTED_PRODUCER_MEMBERS) | {"SOURCE_MANIFEST.json"}, "producer undeclared/missing members")
        members: dict[str, HeldRegular] = {}
        for row in rows:
            name = row["name"]
            item = package.open_member(name, cap=256 << 20, label=f"producer member {name}")
            require(len(item.data) == row["bytes"] and item.sha256 == row["sha256"], f"producer member binding: {name}")
            if name.endswith(".py"):
                compile(item.data, f"<authenticated-producer-syntax:{name}>", "exec", dont_inherit=True, optimize=0)
            members[name] = item
        review = HeldAbsoluteFile.open(review_path, cap=16 << 20, label="producer final review")
        require(review.member.sha256 == review_pin, "producer final review external pin mismatch")
        review_record = strict_json(review.member.data, label="producer final review")
        require(review_record.get("schema") == "unifilar-wfa-entropy-census-independent-source-review-v8", "producer final review schema")
        require(review_record.get("status") == "PASS_INDEPENDENT_SOURCE_REVIEW", "producer final review status")
        require(review_record.get("reviewed_source_manifest_sha256") == manifest.sha256, "producer review/manifest binding")
        if expected_public_commit is not None:
            require(review_record.get("reviewed_public_commit") == public_commit(expected_public_commit), "producer review/public commit binding")
        source_root = sha256(canonical_json(rows))
        package.verify_stable()
        review.verify_stable()
        return ProducerClosure(package, manifest, review, members, record, rows, source_root)
    except Exception:
        if review is not None:
            review.close()
        package.close()
        raise


def reject_preloaded_snapshot_modules(
    extra_names: Iterable[str] = (),
    *,
    include_producer_aliases: bool = True,
) -> None:
    forbidden = (set(PRELOADED_PRODUCER_ALIASES) if include_producer_aliases else set()) | set(extra_names)
    present = sorted(name for name in forbidden if name in sys.modules)
    require(not present, f"preloaded authenticated snapshot module(s): {present}")


def exec_snapshot_module(
    name: str,
    data: bytes,
    expected_sha256: str,
    *,
    provenance_ledger: "AppendOnlyImportNativeLedger | None" = None,
    held_member: HeldRegular | None = None,
    member_path: str | None = None,
) -> types.ModuleType:
    safe_name(name, "snapshot module name")
    require(name not in sys.modules, f"preloaded snapshot module: {name}")
    require(sha256(data) == digest(expected_sha256, "snapshot digest"), f"snapshot digest mismatch: {name}")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated-snapshot:{name}>"
    module.__package__ = ""
    code = compile(data, module.__file__, "exec", dont_inherit=True, optimize=0)
    sys.modules[name] = module
    if provenance_ledger is not None:
        provenance_ledger._append(
            "PYTHON_SNAPSHOT_EXECUTE_START",
            module_name=name,
            member_path=member_path,
            member_sha256=expected_sha256,
        )
    try:
        exec(code, module.__dict__)
    except Exception:
        if provenance_ledger is not None:
            provenance_ledger._append(
                "PYTHON_SNAPSHOT_EXECUTE_FAILED_REMOVAL",
                module_name=name,
                member_path=member_path,
                member_sha256=expected_sha256,
            )
        sys.modules.pop(name, None)
        raise
    if provenance_ledger is not None:
        require(held_member is not None and member_path is not None, "snapshot held provenance required")
        provenance_ledger.bind_authenticated_module(
            name,
            module,
            held_member,
            member_path=member_path,
            execution_kind="authenticated_snapshot",
        )
    return module


@dataclass
class SnapshotModules:
    by_member: dict[str, types.ModuleType]
    installed_names: list[str]
    provenance_ledger: "AppendOnlyImportNativeLedger | None" = None

    def remove(self) -> None:
        for name in reversed(self.installed_names):
            module = sys.modules.get(name)
            if self.provenance_ledger is not None and module is not None and self.provenance_ledger.active:
                self.provenance_ledger.retire_authenticated_module(name, module)
            else:
                sys.modules.pop(name, None)
        self.installed_names.clear()


def compile_producer_snapshots(
    closure: ProducerClosure,
    provenance_ledger: "AppendOnlyImportNativeLedger",
) -> SnapshotModules:
    require(provenance_ledger.active, "authenticated provenance enforcement must precede producer compilation")
    reject_preloaded_snapshot_modules()
    loaded: dict[str, types.ModuleType] = {}
    installed: list[str] = []
    try:
        by_manifest = {row["name"]: row for row in closure.manifest_rows}
        for member_name in SNAPSHOT_DEPENDENCY_ORDER:
            module_name = SNAPSHOT_MODULE_NAMES[member_name]
            module = exec_snapshot_module(
                module_name,
                closure.members[member_name].data,
                by_manifest[member_name]["sha256"],
                provenance_ledger=provenance_ledger,
                held_member=closure.members[member_name],
                member_path=closure.package.path + "/" + member_name,
            )
            loaded[member_name] = module
            installed.append(module_name)
        require(callable(getattr(loaded["stage0_census.py"], "source_phase", None)), "stage source_phase ABI")
        require(callable(getattr(loaded["stage0_census.py"], "validate_source_preflight", None)), "stage preflight ABI")
        require(callable(getattr(loaded["cupy_backend.py"], "build_backend", None)), "CuPy backend ABI")
        require(hasattr(loaded["strata_sc_adapter.py"], "StrataSCAdapter"), "STRATA adapter ABI")
        return SnapshotModules(loaded, installed, provenance_ledger)
    except Exception:
        for name in reversed(installed):
            module = sys.modules.get(name)
            if module is not None and provenance_ledger.active:
                provenance_ledger.retire_authenticated_module(name, module)
            else:
                sys.modules.pop(name, None)
        raise


EXPECTED_DISPATCHER_MEMBERS = frozenset({
    "README.md",
    "bootstrap.py",
    "decoder_bundle.json",
    "design_lock.json",
    "runtime_lock.json",
    "strata_ordinal_bridge.py",
    "test_source_only.py",
    "verify_output.py",
    "verify_source.py",
})


def _absolute_file_from_module(value: Any, *, label: str) -> str:
    raw = str(value)
    require(raw.startswith("/"), f"{label} absolute file")
    _absolute_components(raw, label=label)
    return raw


def _loaded_python_module_paths() -> set[str]:
    observed: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if raw is None or str(raw).startswith("<"):
            continue
        observed.add(_absolute_file_from_module(raw, label=f"loaded Python module {name}"))
    return observed


def _loaded_native_image_paths() -> set[str]:
    require(os.name == "posix" and os.path.exists("/proc/self/maps"), "runtime image closure requires Linux /proc/self/maps")
    observed: set[str] = set()
    with open("/proc/self/maps", "rb", buffering=0) as handle:
        data = handle.read(64 << 20)
    require(len(data) < (64 << 20), "runtime process map is bounded")
    for line in data.decode("utf-8", "strict").splitlines():
        fields = line.split(None, 5)
        require(len(fields) >= 5, "runtime process map row")
        if len(fields) != 6 or "x" not in fields[1] or not fields[5].startswith("/"):
            continue
        path = fields[5]
        require(not path.endswith(" (deleted)"), "loaded native image was deleted")
        observed.add(_absolute_file_from_module(path, label="loaded native image"))
    return observed


def validate_loaded_image_path_set(*, observed: set[str], held_manifest_paths: set[str]) -> None:
    require(observed <= held_manifest_paths, f"unmanifested imported Python/native image(s): {sorted(observed - held_manifest_paths)}")


_NUMERIC_RUNTIME_PREFIXES = ("numpy", "cupy", "scipy", "safetensors", "cuda.pathfinder")


def _is_numeric_runtime_module(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _NUMERIC_RUNTIME_PREFIXES)


def reject_preloaded_numeric_modules() -> None:
    present = sorted(name for name in sys.modules if _is_numeric_runtime_module(name))
    require(not present, f"preloaded numeric module(s) rejected: {present}")


class _LockedImportList(list[Any]):
    """A list-compatible import setting which rejects in-place hook changes."""

    def _blocked(self, *_args: Any, **_kwargs: Any) -> None:
        raise DispatchError("authenticated import boundary is immutable")

    __setitem__ = _blocked
    __delitem__ = _blocked
    __iadd__ = _blocked
    __imul__ = _blocked
    append = _blocked
    clear = _blocked
    extend = _blocked
    insert = _blocked
    pop = _blocked
    remove = _blocked
    reverse = _blocked
    sort = _blocked


def reject_ambient_import_hooks(
    *,
    meta_path: Sequence[Any] | None = None,
    path_hooks: Sequence[Any] | None = None,
) -> None:
    """Accept only pristine CPython importers; arbitrary ambient hooks reject."""

    frozen = sys.modules.get("_frozen_importlib")
    external = sys.modules.get("_frozen_importlib_external")
    require(frozen is not None and external is not None, "CPython frozen import machinery unavailable")
    observed_meta = tuple(sys.meta_path if meta_path is None else meta_path)
    allowed_meta = (
        frozen.BuiltinImporter,
        frozen.FrozenImporter,
        external.PathFinder,
    )
    require(observed_meta == allowed_meta, "ambient sys.meta_path hook rejected")
    observed_hooks = tuple(sys.path_hooks if path_hooks is None else path_hooks)
    for hook in observed_hooks:
        module = getattr(hook, "__module__", "")
        qualname = getattr(hook, "__qualname__", getattr(hook, "__name__", ""))
        is_zip = module == "zipimport" and qualname == "zipimporter"
        is_file = module == "_frozen_importlib_external" and qualname.endswith("path_hook_for_FileFinder")
        require(is_zip or is_file, "ambient sys.path_hooks hook rejected")


@dataclass(frozen=True)
class ModuleProvenance:
    module_name: str
    member_path: str
    member_sha256: str
    member_identity: tuple[int, int, int, int, int, int, int]
    execution_kind: str


@dataclass(frozen=True)
class _RuntimeModuleCandidate:
    module_name: str
    held: HeldAbsoluteFile
    path: str
    is_package: bool
    execution_kind: str


@dataclass(frozen=True)
class _RuntimeNamespaceCandidate:
    module_name: str
    held: HeldAbsoluteFile
    path: str
    package_search_path: str


class AppendOnlyImportNativeLedger:
    """Hash-chained Python/import and native-loader history for one run.

    Module authority lives in this registry and the held manifest member, not
    in mutable presentation attributes such as ``module.__file__``.  The
    ledger has no deletion API; snapshots returned to callers are copies.
    """

    _DOMAIN = b"UWFA-SC-V8-IMPORT-NATIVE-EVENT-LEDGER-V2\0"

    def __init__(self, held_by_path: Mapping[str, HeldAbsoluteFile]) -> None:
        self._held_by_path = dict(held_by_path)
        self._events: list[dict[str, Any]] = []
        self._chain = sha256(self._DOMAIN)
        self._module_provenance: dict[str, tuple[types.ModuleType, ModuleProvenance, HeldRegular]] = {}
        self._retired_module_names: set[str] = set()
        self._native_active: dict[str, tuple[int, int]] = {}
        self._native_seen_loads: set[str] = set()
        self._native_feed_fd: int | None = None
        self._native_feed_nonce: str | None = None
        self._native_feed_buffer = b""
        self._native_feed_sequence = 0
        self._native_feed_ready = False
        self._native_baseline_complete = False
        self._native_auditor_path: str | None = None
        self._finder: AuthenticatedManifestFinder | None = None
        self._locked_meta_path: _LockedImportList | None = None
        self._locked_path_hooks: _LockedImportList | None = None
        self._locked_sys_path: _LockedImportList | None = None
        self._active = False
        self._finalized = False

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._events)

    @property
    def chain_sha256(self) -> str:
        return self._chain

    @property
    def active(self) -> bool:
        return self._active and not self._finalized

    def _append(self, action: str, **fields: Any) -> None:
        require(not self._finalized, "event after finalized provenance ledger")
        row = {"sequence": len(self._events), "action": action, **fields}
        prior = self._chain
        row["prior_chain_sha256"] = prior
        self._chain = length_prefixed_root(self._DOMAIN, (prior, canonical_json(row)))
        row["chain_sha256"] = self._chain
        self._events.append(row)

    def bind_authenticated_module(
        self,
        name: str,
        module: types.ModuleType,
        held: HeldRegular,
        *,
        member_path: str,
        execution_kind: str,
    ) -> None:
        require(name not in self._module_provenance, f"duplicate module provenance: {name}")
        require(held.sha256 == sha256(held.data), f"held module digest changed: {name}")
        held.verify_stable()
        provenance = ModuleProvenance(name, member_path, held.sha256, held.identity, execution_kind)
        self._module_provenance[name] = (module, provenance, held)
        module.__authenticated_manifest_provenance__ = {
            "module_name": name,
            "member_path": member_path,
            "member_sha256": held.sha256,
            "member_identity": list(held.identity),
            "execution_kind": execution_kind,
        }
        self._append(
            "PYTHON_MODULE_BOUND",
            module_name=name,
            member_path=member_path,
            member_sha256=held.sha256,
            member_identity=list(held.identity),
            execution_kind=execution_kind,
        )

    def note_python_resolution(self, name: str, candidate: _RuntimeModuleCandidate) -> None:
        self._append(
            "PYTHON_IMPORT_RESOLVED",
            module_name=name,
            member_path=candidate.path,
            member_sha256=candidate.held.member.sha256,
            execution_kind=candidate.execution_kind,
        )

    def retire_authenticated_module(self, name: str, module: types.ModuleType) -> None:
        bound = self._module_provenance.get(name)
        require(bound is not None and bound[0] is module, f"unbound module removal rejected: {name}")
        require(sys.modules.get(name) is module, f"module removal object mismatch: {name}")
        del self._module_provenance[name]
        self._retired_module_names.add(name)
        sys.modules.pop(name)
        self._append(
            "PYTHON_MODULE_AUTHENTICATED_REMOVAL",
            module_name=name,
            member_path=bound[1].member_path,
            member_sha256=bound[1].member_sha256,
        )

    def note_native_extension_request(self, candidate: _RuntimeModuleCandidate) -> None:
        candidate.held.verify_stable()
        self._append(
            "NATIVE_EXTENSION_LOAD_REQUEST",
            module_name=candidate.module_name,
            member_path=candidate.path,
            member_sha256=candidate.held.member.sha256,
            member_identity=list(candidate.held.member.identity),
        )

    def _assert_import_boundary(self) -> None:
        require(self._active and not self._finalized, "authenticated import boundary inactive")
        require(sys.meta_path is self._locked_meta_path, "sys.meta_path object replaced")
        require(sys.path_hooks is self._locked_path_hooks, "sys.path_hooks object replaced")
        require(sys.path is self._locked_sys_path, "sys.path object replaced")
        frozen = sys.modules["_frozen_importlib"]
        require(
            tuple(sys.meta_path) == (frozen.BuiltinImporter, frozen.FrozenImporter, self._finder),
            "sys.meta_path authenticated finder order changed",
        )
        require(tuple(sys.path_hooks) == () and tuple(sys.path) == (), "ambient path import surface re-enabled")

    def _audit_hook(self, event: str, args: tuple[Any, ...]) -> None:
        if not self._active:
            return
        self._assert_import_boundary()
        if event == "import":
            name = str(args[0]) if args else ""
            filename = None if len(args) < 2 or args[1] is None else str(args[1])
            self._append("PYTHON_IMPORT_AUDIT", module_name=name, requested_path=filename)
            if _is_numeric_runtime_module(name):
                require(
                    self._finder is not None and self._finder.has_module(name),
                    f"unmanifested numeric import rejected: {name}",
                )
        elif event == "ctypes.dlopen":
            path = None if not args or args[0] is None else str(args[0])
            self._append("NATIVE_CTYPES_DLOPEN_AUDIT", requested_path=path)
            if path is not None:
                require(path.startswith("/"), "relative ctypes.dlopen rejected")
                require(path in self._held_by_path, f"unmanifested ctypes.dlopen rejected: {path}")

    def activate(
        self,
        *,
        finder: "AuthenticatedManifestFinder",
        native_audit_event_fd: int,
        native_audit_session_nonce: str,
        native_auditor_path: str,
        trusted_preloaded: Mapping[str, tuple[types.ModuleType, HeldRegular, str]],
    ) -> None:
        require(not self._active and not self._finalized, "provenance ledger activation order")
        reject_ambient_import_hooks()
        reject_preloaded_numeric_modules()
        require(type(native_audit_event_fd) is int and native_audit_event_fd >= 3, "native audit event fd")
        require(os.get_blocking(native_audit_event_fd) is False, "native audit event fd must be nonblocking")
        self._native_feed_fd = native_audit_event_fd
        self._native_feed_nonce = digest(native_audit_session_nonce, "native audit session nonce")
        auditor = self._held_by_path.get(native_auditor_path)
        require(auditor is not None, "native loader auditor is not a held manifest member")
        auditor.verify_stable()
        require(os.environ.get("LD_AUDIT") == native_auditor_path, "LD_AUDIT is not exactly the held native auditor")
        self._native_auditor_path = native_auditor_path
        self._finder = finder
        finder.attach_ledger(self)
        frozen = sys.modules["_frozen_importlib"]
        self._locked_meta_path = _LockedImportList((frozen.BuiltinImporter, frozen.FrozenImporter, finder))
        self._locked_path_hooks = _LockedImportList()
        self._locked_sys_path = _LockedImportList()
        sys.meta_path = self._locked_meta_path
        sys.path_hooks = self._locked_path_hooks
        sys.path_importer_cache.clear()
        sys.path = self._locked_sys_path
        self._active = True
        sys.addaudithook(self._audit_hook)
        self._append("IMPORT_NATIVE_ENFORCEMENT_ACTIVE")
        for name, (module, held, path) in sorted(trusted_preloaded.items()):
            require(sys.modules.get(name) is module, f"trusted preloaded module object mismatch: {name}")
            self.bind_authenticated_module(name, module, held, member_path=path, execution_kind="authenticated_snapshot")
        self._bind_preexisting_interpreter_modules(set(trusted_preloaded))
        self._drain_native_feed(require_baseline=True)
        self._assert_native_current_set()
        self._append("IMPORT_NATIVE_BASELINE_CLOSED")

    def _bind_preexisting_interpreter_modules(self, exempt: set[str]) -> None:
        interpreter_rows = [item for path, item in self._held_by_path.items() if path == sys.executable]
        require(len(interpreter_rows) == 1, "authenticated interpreter member unavailable")
        interpreter = interpreter_rows[0].member
        for name, module in sorted(tuple(sys.modules.items())):
            if name in exempt or module is None:
                continue
            raw = getattr(module, "__file__", None)
            if raw is not None and not str(raw).startswith("<"):
                path = os.path.abspath(str(raw))
                held = self._held_by_path.get(path)
                require(held is not None, f"preloaded Python module is unmanifested: {name}: {path}")
                self.bind_authenticated_module(name, module, held.member, member_path=path, execution_kind="preloaded_held_python")
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            require(origin in {"built-in", "frozen", None}, f"unbound preloaded module provenance: {name}")
            provenance = ModuleProvenance(name, sys.executable, interpreter.sha256, interpreter.identity, f"interpreter_{origin or 'intrinsic'}")
            self._module_provenance[name] = (module, provenance, interpreter)
            self._append(
                "PYTHON_INTERPRETER_MODULE_BOUND",
                module_name=name,
                member_path=sys.executable,
                member_sha256=interpreter.sha256,
                execution_kind=provenance.execution_kind,
            )

    def _drain_native_feed(self, *, require_baseline: bool = False) -> None:
        require(self._native_feed_fd is not None, "native audit feed unavailable")
        while True:
            try:
                chunk = os.read(self._native_feed_fd, 1 << 20)
            except BlockingIOError:
                break
            require(bool(chunk), "native audit feed closed before final closure")
            self._native_feed_buffer += chunk
            require(len(self._native_feed_buffer) <= (8 << 20), "native audit feed buffer bound")
        parts = self._native_feed_buffer.split(b"\n")
        self._native_feed_buffer = parts.pop()
        for raw in parts:
            require(bool(raw), "empty native audit event")
            record = strict_json(raw, label="native audit event")
            require(canonical_json(record) == raw, "native audit event must be canonical JSON")
            self.ingest_native_event(record)
        require(not self._native_feed_buffer, "partial native audit event at checkpoint")
        if require_baseline:
            require(self._native_feed_ready and self._native_baseline_complete, "native audit feed not active from process start")

    def ingest_native_event(self, record: Mapping[str, Any]) -> None:
        require(isinstance(record, Mapping), "native audit event object")
        action = str(record.get("action", ""))
        sequence = record.get("sequence")
        require(type(sequence) is int and sequence == self._native_feed_sequence, "native audit event sequence gap/replay")
        self._native_feed_sequence += 1
        self._append("NATIVE_LOADER_EVENT", native_event=dict(record))
        if action == "READY":
            require(
                set(record) == {"schema", "sequence", "action", "nonce", "auditor_path", "auditor_sha256", "device", "inode"},
                "native READY fields",
            )
            require(not self._native_feed_ready and sequence == 0, "native READY order")
            require(record["schema"] == "uwfa-native-loader-event-v1", "native event schema")
            require(record["nonce"] == self._native_feed_nonce, "native audit session binding")
            require(record["auditor_path"] == self._native_auditor_path, "native auditor path binding")
            auditor = self._held_by_path.get(str(record["auditor_path"]))
            require(auditor is not None, "native auditor held member")
            require(record["auditor_sha256"] == auditor.member.sha256, "native auditor digest binding")
            require((record["device"], record["inode"]) == _inode(auditor.member.identity), "native auditor identity binding")
            self._native_feed_ready = True
            return
        require(self._native_feed_ready, "native event before READY")
        require(record.get("schema") == "uwfa-native-loader-event-v1", "native event schema")
        if action == "BASELINE_END":
            require(set(record) == {"schema", "sequence", "action"}, "native BASELINE_END fields")
            require(not self._native_baseline_complete, "duplicate native baseline end")
            self._native_baseline_complete = True
            return
        require(action in {"LOAD", "UNLOAD"}, "native event action")
        require(set(record) == {"schema", "sequence", "action", "path", "device", "inode"}, "native load event fields")
        path = str(record["path"])
        require(path.startswith("/"), "native event absolute path")
        require(type(record["device"]) is int and type(record["inode"]) is int, "native event identity")
        identity = (record["device"], record["inode"])
        if action == "LOAD":
            held = self._held_by_path.get(path)
            require(held is not None, f"unmanifested transient native load: {path}")
            require(_inode(held.member.identity) == identity, f"native load identity mismatch: {path}")
            held.verify_stable()
            require(path not in self._native_active, f"duplicate active native load: {path}")
            self._native_active[path] = identity
            self._native_seen_loads.add(path)
        else:
            require(self._native_active.get(path) == identity, f"native unload without matching load: {path}")
            del self._native_active[path]

    def _assert_native_current_set(self) -> None:
        observed = _loaded_native_image_paths()
        validate_loaded_image_path_set(observed=observed, held_manifest_paths=set(self._held_by_path))
        require(observed == set(self._native_active), "native loader event/current image closure mismatch")

    def checkpoint(self, label: str) -> None:
        safe_name(label, "provenance checkpoint")
        self._assert_import_boundary()
        self._drain_native_feed()
        self._assert_native_current_set()
        self._verify_module_closure()
        self._append("PROVENANCE_CHECKPOINT", label=label)

    def _verify_module_closure(self) -> None:
        for name, (module, provenance, held) in sorted(self._module_provenance.items()):
            if sys.modules.get(name) is not module:
                self._append("PYTHON_MODULE_REMOVED_OR_REPLACED", module_name=name)
                raise DispatchError(f"authenticated module removed/replaced in sys.modules: {name}")
            require(held.identity == provenance.member_identity and held.sha256 == provenance.member_sha256, f"module provenance registry changed: {name}")
            held.verify_stable()
        unbound = []
        for name, module in tuple(sys.modules.items()):
            if module is None or name in self._module_provenance:
                continue
            spec = getattr(module, "__spec__", None)
            if getattr(spec, "origin", None) in {"built-in", "frozen"}:
                continue
            unbound.append(name)
        require(not (self._retired_module_names & set(sys.modules)), "retired authenticated module reappeared")
        require(not unbound, f"loaded Python module(s) lack authenticated object provenance: {sorted(unbound)}")

    def finalize(self) -> None:
        self.checkpoint("FINAL")
        self._append("IMPORT_NATIVE_EVENT_CLOSURE_FINAL")
        self._finalized = True


class AuthenticatedManifestFinder:
    """Meta-path finder/loader executing only bytes retained by held members."""

    def __init__(self, *, import_roots: Sequence[str], held_by_path: Mapping[str, HeldAbsoluteFile]) -> None:
        require(bool(import_roots), "authenticated import roots")
        self.import_roots = tuple(import_roots)
        self._ledger: AppendOnlyImportNativeLedger | None = None
        self._candidates: dict[str, _RuntimeModuleCandidate] = {}
        self._namespaces: set[str] = set()
        self._namespace_anchors: dict[str, _RuntimeNamespaceCandidate] = {}
        for path, held in sorted(held_by_path.items()):
            matching = [root for root in self.import_roots if path.startswith(root + "/")]
            if not matching:
                continue
            root = max(matching, key=len)
            relative = path[len(root) + 1:]
            candidate = self._candidate_from_relative(relative, held, path)
            if candidate is None:
                continue
            previous = self._candidates.get(candidate.module_name)
            priority = {"source": 3, "bytecode": 2, "extension": 1}
            require(previous is None or priority[candidate.execution_kind] != priority[previous.execution_kind], f"ambiguous authenticated module member: {candidate.module_name}")
            if previous is None or priority[candidate.execution_kind] > priority[previous.execution_kind]:
                self._candidates[candidate.module_name] = candidate
            parts = candidate.module_name.split(".")
            for index in range(1, len(parts)):
                namespace = ".".join(parts[:index])
                self._namespaces.add(namespace)
                anchor = self._namespace_anchors.get(namespace)
                if anchor is None or path.encode("utf-8") < anchor.path.encode("utf-8"):
                    package_search_path = root + "/" + "/".join(parts[:index])
                    self._namespace_anchors[namespace] = _RuntimeNamespaceCandidate(namespace, held, path, package_search_path)

    def attach_ledger(self, ledger: AppendOnlyImportNativeLedger) -> None:
        require(self._ledger is None, "authenticated finder already attached")
        self._ledger = ledger

    def has_module(self, fullname: str) -> bool:
        return fullname in self._candidates or fullname in self._namespaces

    @staticmethod
    def _candidate_from_relative(relative: str, held: HeldAbsoluteFile, path: str) -> _RuntimeModuleCandidate | None:
        normalized = relative.replace("\\", "/")
        if ".dist-info/" in normalized or ".data/" in normalized:
            return None
        is_package = False
        kind: str
        if normalized.endswith("/__init__.py"):
            stem = normalized[:-len("/__init__.py")]
            is_package = True
            kind = "source"
        elif normalized.endswith(".py"):
            stem = normalized[:-3]
            kind = "source"
        elif normalized.endswith(".pyc"):
            stem = normalized[:-4]
            if "/__pycache__/" in stem:
                parent, leaf = stem.rsplit("/__pycache__/", 1)
                leaf = leaf.split(".", 1)[0]
                if leaf == "__init__":
                    stem = parent
                    is_package = True
                else:
                    stem = parent + "/" + leaf
            elif stem.endswith("/__init__"):
                stem = stem[:-len("/__init__")]
                is_package = True
            kind = "bytecode"
        elif normalized.endswith(".so"):
            parent, slash, leaf = normalized.rpartition("/")
            stem = (parent + slash if slash else "") + leaf.split(".", 1)[0]
            kind = "extension"
        else:
            return None
        require(stem and all(safe_name(part, "runtime module component") for part in stem.split("/")), "runtime module path")
        module_name = stem.replace("/", ".")
        return _RuntimeModuleCandidate(module_name, held, path, is_package, kind)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        del path, target
        require(self._ledger is not None and self._ledger.active, "authenticated finder used before enforcement")
        candidate = self._candidates.get(fullname)
        frozen = sys.modules["_frozen_importlib"]
        if candidate is not None:
            candidate.held.verify_stable()
            self._ledger.note_python_resolution(fullname, candidate)
            spec = frozen.ModuleSpec(fullname, self, is_package=candidate.is_package)
            spec.loader_state = candidate
            spec.origin = f"<authenticated-manifest:{candidate.held.member.sha256}:{fullname}>"
            if candidate.is_package:
                spec.submodule_search_locations = [f"<authenticated-package:{candidate.held.member.sha256}:{fullname}>"]
            return spec
        if fullname in self._namespaces:
            anchor = self._namespace_anchors[fullname]
            anchor.held.verify_stable()
            self._ledger._append("PYTHON_NAMESPACE_RESOLVED", module_name=fullname, anchor_path=anchor.path)
            spec = frozen.ModuleSpec(fullname, loader=self, is_package=True)
            spec.loader_state = anchor
            spec.submodule_search_locations = [f"<authenticated-namespace:{fullname}>"]
            return spec
        raise DispatchError(f"unmanifested Python import rejected: {fullname}")

    def create_module(self, spec: Any) -> Any:
        candidate = spec.loader_state
        if isinstance(candidate, _RuntimeNamespaceCandidate):
            return None
        if candidate.execution_kind != "extension":
            return None
        require(self._ledger is not None, "extension provenance ledger")
        candidate.held.verify_stable()
        self._ledger.note_native_extension_request(candidate)
        fd_path = f"/proc/self/fd/{candidate.held.member.fd}"
        external = sys.modules["_frozen_importlib_external"]
        delegate = external.ExtensionFileLoader(spec.name, fd_path)
        delegate_spec = sys.modules["_frozen_importlib"].ModuleSpec(spec.name, delegate, origin=fd_path)
        module = delegate.create_module(delegate_spec)
        spec.loader_state = (candidate, delegate)
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        require(self._ledger is not None, "module provenance ledger")
        state = module.__spec__.loader_state
        if isinstance(state, _RuntimeNamespaceCandidate):
            state.held.verify_stable()
            module.__file__ = None
            module.__path__ = [state.package_search_path]
            self._ledger.bind_authenticated_module(
                state.module_name,
                module,
                state.held.member,
                member_path=state.path,
                execution_kind="namespace_no_code_anchor",
            )
            return
        if isinstance(state, tuple):
            candidate, delegate = state
        else:
            candidate, delegate = state, None
        candidate.held.verify_stable()
        label = f"<authenticated-manifest:{candidate.held.member.sha256}:{candidate.module_name}>"
        module.__file__ = candidate.path
        module.__cached__ = None
        if candidate.is_package:
            module.__path__ = [candidate.path.rsplit("/", 1)[0]]
        if candidate.execution_kind == "source":
            code = compile(candidate.held.member.data, label, "exec", dont_inherit=True, optimize=0)
            exec(code, module.__dict__, module.__dict__)
        elif candidate.execution_kind == "bytecode":
            data = candidate.held.member.data
            require(len(data) >= 16, f"authenticated bytecode header: {candidate.module_name}")
            marshal = sys.modules.get("marshal")
            require(marshal is not None, "marshal builtin must be preloaded before enforcement")
            code = marshal.loads(data[16:])
            require(isinstance(code, types.CodeType), f"authenticated bytecode object: {candidate.module_name}")
            exec(code, module.__dict__, module.__dict__)
        else:
            require(delegate is not None, "extension loader delegate")
            delegate.exec_module(module)
            module.__file__ = candidate.path
        candidate.held.verify_stable()
        self._ledger.bind_authenticated_module(
            candidate.module_name,
            module,
            candidate.held.member,
            member_path=candidate.path,
            execution_kind=candidate.execution_kind,
        )


def _enumerate_regular_tree_paths(root: "HeldDirectory") -> set[str]:
    """Enumerate a held tree descriptor-relatively and reject special links."""

    result: set[str] = set()
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)

    def visit(fd: int, prefix: str) -> None:
        entries = sorted(list(os.scandir(fd)), key=lambda row: row.name.encode("utf-8"))
        for entry in entries:
            safe_name(entry.name, "runtime tree leaf")
            info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
            path = prefix + "/" + entry.name
            if stat.S_ISDIR(info.st_mode):
                child = os.open(entry.name, os.O_RDONLY | directory_flag | nofollow | cloexec, dir_fd=fd)
                try:
                    require(_identity(os.fstat(child)) == _identity(info), "runtime tree directory substitution")
                    visit(child, path)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                result.add(path)
            else:
                raise DispatchError(f"runtime tree contains symlink/special member: {path}")

    visit(root.fd, root.path)
    return result


@dataclass
class DispatcherClosure:
    package: HeldDirectory
    manifest: HeldRegular
    audit: HeldAbsoluteFile
    members: dict[str, HeldRegular]
    manifest_rows: list[dict[str, Any]]

    def verify_stable(self) -> None:
        self.package.verify_stable()
        self.audit.verify_stable()

    def close(self) -> None:
        self.audit.close()
        self.package.close()


def authenticate_dispatcher(
    *,
    package_path: str,
    audit_path: str,
    manifest_pin: str,
    audit_pin: str,
    expected_public_commit: str | None = None,
) -> DispatcherClosure:
    manifest_pin = digest(manifest_pin, "dispatcher manifest pin")
    audit_pin = digest(audit_pin, "dispatcher audit pin")
    package = HeldDirectory.open_absolute(package_path, label="dispatcher package")
    audit: HeldAbsoluteFile | None = None
    try:
        manifest = package.open_member("SOURCE_MANIFEST.json", cap=4 << 20, label="dispatcher manifest")
        require(manifest.sha256 == manifest_pin, "dispatcher manifest external pin mismatch")
        record = strict_json(manifest.data, label="dispatcher manifest")
        exact_fields(record, {"schema", "status", "members", "access_attestation", "remaining_authority_gates"}, label="dispatcher manifest")
        require(record["schema"] == "uwfa-sc-v8-external-dispatcher-source-manifest-v2", "dispatcher manifest schema")
        require(record["status"] == "SEALED_EXTERNAL_DISPATCHER_SOURCE_NO_PAYLOAD_AUTHORITY", "dispatcher manifest production status")
        rows, _by_name = _manifest_rows(record["members"], expected_names=EXPECTED_DISPATCHER_MEMBERS, label="dispatcher manifest")
        actual = package.enumerate_names()
        require(actual == set(EXPECTED_DISPATCHER_MEMBERS) | {"SOURCE_MANIFEST.json"}, "dispatcher undeclared/missing members")
        members: dict[str, HeldRegular] = {}
        for row in rows:
            name = row["name"]
            member = package.open_member(name, cap=256 << 20, label=f"dispatcher member {name}")
            require(len(member.data) == row["bytes"] and member.sha256 == row["sha256"], f"dispatcher member binding: {name}")
            if name.endswith(".py"):
                compile(member.data, f"<authenticated-dispatcher-syntax:{name}>", "exec", dont_inherit=True, optimize=0)
            members[name] = member
        audit = HeldAbsoluteFile.open(audit_path, cap=16 << 20, label="dispatcher independent audit")
        require(audit.member.sha256 == audit_pin, "dispatcher audit external pin mismatch")
        audit_record = strict_json(audit.member.data, label="dispatcher independent audit")
        require(audit_record.get("schema") == "uwfa-sc-v8-external-dispatcher-independent-review-v2", "dispatcher audit schema")
        require(audit_record.get("status") == "PASS_INDEPENDENT_DISPATCHER_REVIEW", "dispatcher audit status")
        require(audit_record.get("reviewed_dispatcher_source_manifest_sha256") == manifest.sha256, "dispatcher audit/manifest binding")
        if expected_public_commit is not None:
            require(audit_record.get("reviewed_dispatcher_public_commit") == public_commit(expected_public_commit, "dispatcher reviewed commit"), "dispatcher audit/public commit binding")
        package.verify_stable()
        audit.verify_stable()
        return DispatcherClosure(package, manifest, audit, members, rows)
    except Exception:
        if audit is not None:
            audit.close()
        package.close()
        raise


@dataclass
class RuntimeClosure:
    lock_record: dict[str, Any]
    lock_bytes: bytes
    tree_manifest: HeldAbsoluteFile
    held_runtime_files: list[HeldAbsoluteFile]
    by_role: dict[str, list[HeldAbsoluteFile]]
    site_packages: HeldDirectory
    python_import_roots: list[HeldDirectory]
    held_by_path: dict[str, HeldAbsoluteFile]
    tree_inventory_paths: frozenset[str]
    imported_modules: dict[str, Any] = field(default_factory=dict)
    settled_image_paths: frozenset[str] | None = None
    additional_authenticated_image_paths: frozenset[str] = frozenset()
    provenance_ledger: AppendOnlyImportNativeLedger | None = None

    @property
    def tree_manifest_sha256(self) -> str:
        return self.tree_manifest.member.sha256

    def activate_import_native_enforcement(
        self,
        *,
        all_held_by_path: Mapping[str, HeldAbsoluteFile],
        native_audit_event_fd: int,
        native_audit_session_nonce: str,
        trusted_preloaded: Mapping[str, tuple[types.ModuleType, HeldRegular, str]],
    ) -> AppendOnlyImportNativeLedger:
        require(self.provenance_ledger is None, "runtime provenance already active")
        require(set(self.held_by_path) <= set(all_held_by_path), "runtime provenance held-path coverage")
        finder = AuthenticatedManifestFinder(
            import_roots=self.lock_record["python_import_roots"],
            held_by_path=self.held_by_path,
        )
        ledger = AppendOnlyImportNativeLedger(all_held_by_path)
        auditor_rows = self.by_role.get("native_loader_auditor", [])
        require(len(auditor_rows) == 1, "runtime native loader auditor role cardinality")
        native_auditor_path = auditor_rows[0].directory.path + "/" + auditor_rows[0].member.name
        ledger.activate(
            finder=finder,
            native_audit_event_fd=native_audit_event_fd,
            native_audit_session_nonce=native_audit_session_nonce,
            native_auditor_path=native_auditor_path,
            trusted_preloaded=trusted_preloaded,
        )
        self.provenance_ledger = ledger
        return ledger

    def _authenticated_distribution_versions(self) -> dict[str, str]:
        site = self.lock_record["site_packages"] + "/"
        versions: dict[str, str] = {}
        for path, held in sorted(self.held_by_path.items()):
            relative = path[len(site):] if path.startswith(site) else ""
            if not relative.endswith(".dist-info/METADATA"):
                continue
            name: str | None = None
            version: str | None = None
            for raw in held.member.data.decode("utf-8", "strict").splitlines():
                if raw.startswith("Name: ") and name is None:
                    name = raw[6:].strip().lower().replace("_", "-").replace(".", "-")
                elif raw.startswith("Version: ") and version is None:
                    version = raw[9:].strip()
                if name is not None and version is not None:
                    break
            require(name is not None and version is not None, f"distribution metadata identity: {path}")
            require(name not in versions, f"duplicate authenticated distribution metadata: {name}")
            versions[name] = version
        return versions

    def enable_python_runtime(self) -> tuple[Any, Any]:
        require(self.provenance_ledger is not None and self.provenance_ledger.active, "runtime import provenance enforcement inactive")
        for name in _NUMERIC_RUNTIME_PREFIXES:
            require(name not in sys.modules, f"runtime module preloaded before authentication: {name}")
        observed_versions = self._authenticated_distribution_versions()
        required = self.lock_record["required_distributions"]
        for distribution, version in required.items():
            normalized = distribution.lower().replace("_", "-").replace(".", "-")
            require(observed_versions.get(normalized) == version, f"runtime distribution version: {distribution}")
        np = __import__("numpy")
        cp = __import__("cupy")
        scipy = __import__("scipy")
        cuda_pathfinder = __import__("cuda.pathfinder", fromlist=["pathfinder"])
        self.imported_modules = {"numpy": np, "cupy": cp, "scipy": scipy, "cuda.pathfinder": cuda_pathfinder}
        for label, module in self.imported_modules.items():
            bound = self.provenance_ledger._module_provenance.get(label)
            require(bound is not None and bound[0] is module, f"{label} module object lacks held-manifest provenance")
        self.provenance_ledger.checkpoint("NUMERIC_IMPORTS_COMPLETE")
        return np, cp

    def seal_imported_image_closure(self, *, additional_authenticated_paths: Iterable[str] = ()) -> frozenset[str]:
        require(self.imported_modules, "runtime imports must settle before image closure")
        additional = frozenset(str(path) for path in additional_authenticated_paths)
        for path in additional:
            _absolute_components(path, label="additional authenticated image path")
        self.additional_authenticated_image_paths = additional
        require(self.provenance_ledger is not None and self.provenance_ledger.active, "runtime provenance closure inactive")
        require(additional <= set(self.provenance_ledger._held_by_path), "late unauthenticated image-path expansion rejected")
        self.provenance_ledger.checkpoint("IMPORTED_IMAGE_CLOSURE_SEALED")
        self.settled_image_paths = frozenset(_loaded_native_image_paths())
        return self.settled_image_paths

    def nvidia_smi_file(self) -> HeldRegular:
        rows = self.by_role.get("nvidia_smi", [])
        require(len(rows) == 1, "runtime manifest requires exactly one nvidia_smi")
        return rows[0].member

    def verify_stable(self) -> None:
        self.tree_manifest.verify_stable()
        self.site_packages.verify_stable()
        for root in self.python_import_roots:
            root.verify_stable()
        for item in self.held_runtime_files:
            item.verify_stable()
        require(_enumerate_regular_tree_paths(self.site_packages) == set(self.tree_inventory_paths), "runtime tree inventory changed")
        if self.settled_image_paths is not None:
            require(self.provenance_ledger is not None and self.provenance_ledger.active, "runtime provenance ledger inactive")
            self.provenance_ledger.checkpoint("RUNTIME_VERIFY_STABLE")

    def close(self) -> None:
        for item in reversed(self.held_runtime_files):
            item.close()
        self.held_runtime_files.clear()
        for root in reversed(self.python_import_roots):
            root.close()
        self.python_import_roots.clear()
        self.tree_manifest.close()


def authenticate_runtime(lock_bytes: bytes, *, expected_lock_sha256: str) -> RuntimeClosure:
    require(sha256(lock_bytes) == digest(expected_lock_sha256, "runtime lock pin"), "runtime lock external pin mismatch")
    lock = strict_json(lock_bytes, label="runtime lock")
    exact_fields(lock, {"schema", "status", "python", "site_packages", "python_import_roots", "runtime_tree_manifest", "required_distributions", "cuda", "gpu", "required_runtime_roles", "image_closure_policy", "native_load_enforcement"}, label="runtime lock")
    require(lock["schema"] == "uwfa-sc-v8-external-runtime-lock-v2", "runtime lock schema")
    require(lock["status"] == "SEALED_EXHAUSTIVE_RUNTIME_TREE", "runtime lock status")
    require(
        lock["image_closure_policy"]
        == "HELD_BYTES_META_PATH_PLUS_PROCESS_START_RTLD_AUDIT_LOAD_UNLOAD_LEDGER_AND_FINAL_DESCRIPTOR_NAME_EVENT_CLOSURE",
        "runtime image closure policy",
    )
    python = exact_fields(lock["python"], {"implementation", "executable", "executable_bytes", "executable_sha256", "version"}, label="runtime Python")
    require(python["implementation"] == "CPython", "runtime implementation")
    require(sys.executable == python["executable"], "runtime interpreter path")
    require(platform.python_version() == python["version"], "runtime interpreter version")
    require(type(python["executable_bytes"]) is int and python["executable_bytes"] > 0, "runtime interpreter bytes")
    digest(python["executable_sha256"], "runtime interpreter SHA-256")
    site_path = str(lock["site_packages"])
    import_root_paths = lock["python_import_roots"]
    require(isinstance(import_root_paths, list) and import_root_paths and import_root_paths[0] == site_path, "runtime import root order/site binding")
    require(import_root_paths == list(dict.fromkeys(import_root_paths)), "runtime import roots unique")
    for root_path in import_root_paths:
        _absolute_components(str(root_path), label="runtime import root")
    native_policy = exact_fields(lock["native_load_enforcement"], {"mechanism", "event_schema", "auditor_role", "active_from_process_start", "records_load_and_unload", "ready_binds_held_auditor_identity_hash"}, label="native load enforcement")
    require(
        native_policy
        == {
            "mechanism": "LINUX_RTLD_AUDIT_APPEND_ONLY_NONBLOCKING_PIPE_V1",
            "event_schema": "uwfa-native-loader-event-v1",
            "auditor_role": "native_loader_auditor",
            "active_from_process_start": True,
            "records_load_and_unload": True,
            "ready_binds_held_auditor_identity_hash": True,
        },
        "native load enforcement policy",
    )
    import_roots: list[HeldDirectory] = []
    try:
        for index, root_path in enumerate(import_root_paths):
            import_roots.append(HeldDirectory.open_absolute(str(root_path), label=f"runtime import root[{index}]"))
    except Exception:
        for root in reversed(import_roots):
            root.close()
        raise
    site = import_roots[0]
    tree: HeldAbsoluteFile | None = None
    held: list[HeldAbsoluteFile] = []
    try:
        tree_spec = exact_fields(lock["runtime_tree_manifest"], {"bytes", "path", "sha256"}, label="runtime tree manifest spec")
        require(type(tree_spec["bytes"]) is int and tree_spec["bytes"] > 0, "runtime tree manifest bytes")
        tree = HeldAbsoluteFile.open(str(tree_spec["path"]), cap=128 << 20, label="runtime tree manifest")
        require(len(tree.member.data) == tree_spec["bytes"], "runtime tree manifest length")
        require(tree.member.sha256 == digest(tree_spec["sha256"], "runtime tree manifest digest"), "runtime tree manifest digest mismatch")
        manifest = strict_json(tree.member.data, label="runtime tree manifest")
        exact_fields(manifest, {"schema", "status", "roots", "members", "coverage"}, label="runtime tree manifest")
        require(manifest["schema"] == "uwfa-sc-v8-runtime-tree-manifest-v2", "runtime tree schema")
        require(manifest["status"] == "SEALED_EXHAUSTIVE_RUNTIME_TREE", "runtime tree status")
        require(manifest["roots"] == import_root_paths, "runtime tree/import roots binding")
        require(
            manifest["coverage"]
            == {
                "all_regular_files_under_roots": True,
                "all_imported_python_module_files": True,
                "all_executable_proc_map_images": True,
                "held_descriptors_final_name_inode_byte_rebinding": True,
                "python_execution_from_held_bytes_or_interpreter_only": True,
                "native_load_unload_events_from_process_start": True,
            },
            "runtime exhaustive coverage declaration",
        )
        rows = manifest["members"]
        require(isinstance(rows, list) and rows, "runtime tree member rows")
        logical_names: set[str] = set()
        paths: set[str] = set()
        by_role: dict[str, list[HeldAbsoluteFile]] = {}
        held_by_path: dict[str, HeldAbsoluteFile] = {}
        prior_path: bytes | None = None
        for index, raw in enumerate(rows):
            row = exact_fields(raw, {"logical_name", "path", "bytes", "sha256", "role"}, label=f"runtime member[{index}]")
            logical = safe_name(row["logical_name"], f"runtime member[{index}] logical name")
            path = str(row["path"])
            _absolute_components(path, label=f"runtime member[{index}] path")
            require(logical not in logical_names and path not in paths, "runtime member duplicate")
            encoded_path = path.encode("utf-8")
            require(prior_path is None or prior_path < encoded_path, "runtime member canonical path order")
            prior_path = encoded_path
            logical_names.add(logical)
            paths.add(path)
            require(type(row["bytes"]) is int and row["bytes"] >= 0, "runtime member bytes")
            digest(row["sha256"], "runtime member SHA-256")
            member = HeldAbsoluteFile.open(path, cap=2 << 30, label=f"runtime member {logical}")
            require(len(member.member.data) == row["bytes"] and member.member.sha256 == row["sha256"], f"runtime member binding: {logical}")
            held.append(member)
            held_by_path[path] = member
            role = str(row["role"])
            by_role.setdefault(role, []).append(member)
        required_roles = lock["required_runtime_roles"]
        require(isinstance(required_roles, list) and required_roles == list(dict.fromkeys(required_roles)), "runtime required roles")
        require(set(required_roles) <= set(by_role), "runtime tree missing required role")
        interpreter_rows = by_role.get("python_interpreter", [])
        require(len(interpreter_rows) == 1, "runtime interpreter member cardinality")
        interpreter_absolute = interpreter_rows[0]
        interpreter = interpreter_absolute.member
        require(interpreter_absolute.directory.path + "/" + interpreter.name == python["executable"], "runtime interpreter member path")
        require(len(interpreter.data) == python["executable_bytes"] and interpreter.sha256 == python["executable_sha256"], "runtime interpreter member binding")
        tree_inventory = frozenset(path for path in paths if any(path.startswith(str(root) + "/") for root in import_root_paths))
        enumerated: set[str] = set()
        for root in import_roots:
            enumerated.update(_enumerate_regular_tree_paths(root))
        require(enumerated == set(tree_inventory), "runtime Python import roots are not exhaustively manifested")
        require(len(by_role.get("native_loader_auditor", [])) == 1, "runtime native loader auditor role cardinality")
        result = RuntimeClosure(lock, lock_bytes, tree, held, by_role, site, import_roots, held_by_path, tree_inventory)
        result.verify_stable()
        return result
    except Exception:
        for member in reversed(held):
            member.close()
        if tree is not None:
            tree.close()
        for root in reversed(import_roots):
            root.close()
        raise


@dataclass
class DecoderClosure:
    bundle_record: dict[str, Any]
    bundle_bytes: bytes
    external_members: dict[str, HeldAbsoluteFile]
    producer_rows: dict[str, dict[str, Any]]
    dispatcher_rows: dict[str, dict[str, Any]]
    dispatcher_members: dict[str, HeldRegular]
    logical_to_producer_member_map_sha256: str
    ordinal_bridge_sha256: str
    universal_decoder_sha256: str
    external_modules: dict[str, types.ModuleType] = field(default_factory=dict)
    installed_names: list[str] = field(default_factory=list)
    provenance_ledger: AppendOnlyImportNativeLedger | None = None

    def compile_external(self, provenance_ledger: AppendOnlyImportNativeLedger) -> dict[str, types.ModuleType]:
        require(provenance_ledger.active, "authenticated provenance enforcement must precede decoder compilation")
        self.provenance_ledger = provenance_ledger
        names = [
            row["module_name"]
            for row in self.bundle_record["members"]
            if row["kind"] in {"external_python_snapshot", "dispatcher_python_snapshot"}
        ]
        reject_preloaded_snapshot_modules(names, include_producer_aliases=False)
        try:
            for row in self.bundle_record["members"]:
                if row["kind"] not in {"external_python_snapshot", "dispatcher_python_snapshot"}:
                    continue
                logical = row["logical_name"]
                held = (
                    self.external_members[logical].member
                    if row["kind"] == "external_python_snapshot"
                    else self.dispatcher_members[logical]
                )
                member_path = (
                    self.external_members[logical].directory.path + "/" + held.name
                    if row["kind"] == "external_python_snapshot"
                    else "<dispatcher-held>/" + held.name
                )
                if row["kind"] == "dispatcher_python_snapshot":
                    matches = [path for path, item in provenance_ledger._held_by_path.items() if item.member is held]
                    require(len(matches) == 1, "dispatcher snapshot held-path provenance")
                    member_path = matches[0]
                module = exec_snapshot_module(
                    row["module_name"],
                    held.data,
                    row["sha256"],
                    provenance_ledger=provenance_ledger,
                    held_member=held,
                    member_path=member_path,
                )
                if row["kind"] == "dispatcher_python_snapshot":
                    require(getattr(module, "BRIDGE_ABI_SHA256", None) == row["bridge_abi_sha256"], "STRATA ordinal bridge ABI binding")
                    require(callable(getattr(module, "wrap_strata_common", None)), "STRATA ordinal bridge wrapper ABI")
                self.external_modules[logical] = module
                self.installed_names.append(row["module_name"])
            return dict(self.external_modules)
        except Exception:
            self.remove_modules()
            raise

    def verify_stable(self) -> None:
        for member in self.external_members.values():
            member.verify_stable()

    def remove_modules(self) -> None:
        for name in reversed(self.installed_names):
            module = sys.modules.get(name)
            if self.provenance_ledger is not None and module is not None and self.provenance_ledger.active:
                self.provenance_ledger.retire_authenticated_module(name, module)
            else:
                sys.modules.pop(name, None)
        self.installed_names.clear()
        self.external_modules.clear()

    def close(self) -> None:
        self.remove_modules()
        for member in reversed(list(self.external_members.values())):
            member.close()
        self.external_members.clear()


def authenticate_decoder_bundle(
    bundle_bytes: bytes,
    *,
    expected_bundle_sha256: str,
    producer: ProducerClosure | None = None,
    dispatcher: DispatcherClosure | None = None,
) -> DecoderClosure:
    require(sha256(bundle_bytes) == digest(expected_bundle_sha256, "decoder bundle pin"), "decoder bundle external pin mismatch")
    record = strict_json(bundle_bytes, label="decoder bundle")
    exact_fields(record, {"schema", "status", "members", "logical_to_producer_member", "forbidden_side_channels", "universal_decoder_sha256"}, label="decoder bundle")
    require(record["schema"] == "uwfa-sc-v8-external-decoder-bundle-v2", "decoder bundle schema")
    require(record["status"] == "SEALED_AUTHENTICATED_DECODER_CLOSURE", "decoder bundle status")
    rows = record["members"]
    require(isinstance(rows, list) and len(rows) == 5, "decoder bundle member count")
    logical_map = validate_logical_to_producer_member_map(record["logical_to_producer_member"])
    logical_map_sha = sha256(canonical_json(logical_map))
    logical_names: set[str] = set()
    module_names: set[str] = set()
    external: dict[str, HeldAbsoluteFile] = {}
    producer_rows: dict[str, dict[str, Any]] = {}
    dispatcher_rows: dict[str, dict[str, Any]] = {}
    dispatcher_members: dict[str, HeldRegular] = {}
    root_rows: list[bytes] = [canonical_json({"logical_to_producer_member": logical_map})]
    require(producer is not None and dispatcher is not None, "decoder authentication closures required")
    manifest_by_name = {row["name"]: row for row in producer.manifest_rows}
    try:
        for index, raw in enumerate(rows):
            require(isinstance(raw, dict), f"decoder member[{index}] object")
            kind = raw.get("kind")
            require(kind in {"external_python_snapshot", "producer_python_snapshot", "dispatcher_python_snapshot"}, "decoder member kind")
            logical = safe_name(raw.get("logical_name"), f"decoder member[{index}] logical name")
            require(logical not in logical_names, "decoder duplicate logical name")
            logical_names.add(logical)
            if kind == "external_python_snapshot":
                row = exact_fields(raw, {"kind", "logical_name", "module_name", "path", "bytes", "sha256"}, label=f"decoder member[{index}]")
                module_name = safe_name(row["module_name"], f"decoder member[{index}] module name")
                require(module_name not in module_names, "decoder duplicate module name")
                module_names.add(module_name)
                require(type(row["bytes"]) is int and row["bytes"] > 0, "decoder external bytes")
                digest(row["sha256"], "decoder external digest")
                member = HeldAbsoluteFile.open(str(row["path"]), cap=256 << 20, label=f"decoder external {logical}")
                require(len(member.member.data) == row["bytes"] and member.member.sha256 == row["sha256"], f"decoder external binding: {logical}")
                compile(member.member.data, f"<authenticated-decoder-syntax:{logical}>", "exec", dont_inherit=True, optimize=0)
                external[logical] = member
                root_rows.append(canonical_json({key: row[key] for key in ("kind", "logical_name", "module_name", "bytes", "sha256")}))
            elif kind == "producer_python_snapshot":
                row = exact_fields(raw, {"kind", "logical_name", "member_name", "sha256"}, label=f"decoder member[{index}]")
                member_name = str(row["member_name"])
                require(member_name in {"strata_sc_adapter.py", "universal_adapter.py"}, "decoder producer member")
                require(manifest_by_name[member_name]["sha256"] == row["sha256"], "decoder producer member digest")
                producer_rows[logical] = dict(row)
                root_rows.append(canonical_json(dict(row)))
            else:
                row = exact_fields(
                    raw,
                    {"kind", "logical_name", "module_name", "member_name", "sha256", "bridge_abi_sha256"},
                    label=f"decoder member[{index}]",
                )
                require(logical == "numpy_strata_ordinal_bridge", "decoder dispatcher logical member")
                require(row["member_name"] == "strata_ordinal_bridge.py", "decoder dispatcher member")
                module_name = safe_name(row["module_name"], f"decoder member[{index}] module name")
                require(module_name not in module_names, "decoder duplicate module name")
                module_names.add(module_name)
                member = dispatcher.members["strata_ordinal_bridge.py"]
                require(member.sha256 == digest(row["sha256"], "ordinal bridge source digest"), "ordinal bridge dispatcher member digest")
                digest(row["bridge_abi_sha256"], "ordinal bridge ABI digest")
                dispatcher_rows[logical] = dict(row)
                dispatcher_members[logical] = member
                root_rows.append(canonical_json(dict(row)))
        require(
            logical_names
            == {
                "strata_format_common",
                "independent_strata_decoder",
                "fixed_strata_sc_adapter",
                "universal_semantic_adapter",
                "numpy_strata_ordinal_bridge",
            },
            "decoder logical member set",
        )
        require(
            {logical: row["member_name"] for logical, row in producer_rows.items()}
            == EXPECTED_LOGICAL_TO_PRODUCER_MEMBER,
            "decoder producer rows must realize exact logical map",
        )
        root = length_prefixed_root(b"UWFA-SC-V8-UNIVERSAL-DECODER-BUNDLE-V2\0", root_rows)
        require(root == digest(record["universal_decoder_sha256"], "decoder bundle root"), "decoder bundle root mismatch")
        ordinal_bridge_sha = dispatcher_rows["numpy_strata_ordinal_bridge"]["sha256"]
        closure = DecoderClosure(
            record,
            bundle_bytes,
            external,
            producer_rows,
            dispatcher_rows,
            dispatcher_members,
            logical_map_sha,
            ordinal_bridge_sha,
            root,
        )
        closure.verify_stable()
        return closure
    except Exception:
        for member in reversed(list(external.values())):
            member.close()
        raise


@dataclass
class AccessJournal:
    events: list[str] = field(default_factory=list)
    authority_complete: bool = False
    preflight_complete: bool = False

    def authority_passed(self) -> None:
        require(not self.authority_complete and not self.preflight_complete, "authority journal order")
        self.authority_complete = True
        self.events.append("AUTHORITY_CLOSURES_AUTHENTICATED")

    def preflight_started(self) -> None:
        require(self.authority_complete and not self.preflight_complete, "preflight before authority")
        self.events.append("FRESH_TYPED_SOURCE_FREE_GPU_PREFLIGHT_STARTED")

    def preflight_passed(self) -> None:
        require(self.events and self.events[-1] == "FRESH_TYPED_SOURCE_FREE_GPU_PREFLIGHT_STARTED", "preflight completion order")
        self.preflight_complete = True
        self.events.append("FRESH_TYPED_SOURCE_FREE_GPU_PREFLIGHT_PASSED")

    def before_payload_path_access(self, label: str) -> None:
        require(self.authority_complete and self.preflight_complete, f"payload path access before authority/preflight: {label}")
        self.events.append(f"POST_PREFLIGHT_OPEN:{label}")


@dataclass
class PreflightBundle:
    all150: dict[str, Any]
    representative: dict[str, Any]
    independent_gpu_identity: dict[str, Any]
    record: dict[str, Any]
    receipt_sha256: str
    typed_evidence: Any
    backend: Any


def _independent_gpu_identity(runtime: RuntimeClosure, protocol: Any, backend: Any) -> dict[str, Any]:
    held = runtime.nvidia_smi_file()
    require(os.path.isdir("/proc/self/fd"), "descriptor-addressed nvidia-smi requires /proc/self/fd")
    executable = f"/proc/self/fd/{held.fd}"
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        close_fds=True,
        pass_fds=(held.fd,),
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    require(len(rows) == 1, "independent GPU identity row count")
    columns = [value.strip() for value in rows[0].split(",")]
    require(len(columns) == 3, "independent GPU identity columns")
    name, uuid, pci = columns
    record = {
        "schema": "uwfa-sc-v8-independent-gpu-identity",
        "status": "PASS_INDEPENDENT_GPU_IDENTITY",
        "device_uuid": protocol.canonical_gpu_uuid(uuid, "independent GPU UUID"),
        "pci_bus_id": protocol.canonical_pci_bus_id(pci, "independent GPU PCI"),
        "device_name": name,
        "provider": "held-descriptor-nvidia-smi",
        "provider_sha256": held.sha256,
    }
    expected = runtime.lock_record["gpu"]
    require(
        (record["device_name"], record["device_uuid"], record["pci_bus_id"])
        == (expected["device_name"], expected["device_uuid"], expected["pci_bus_id"]),
        "runtime-lock/independent-GPU mismatch",
    )
    clean = dict(record)
    record["identity_receipt_sha256"] = sha256(canonical_json(clean))
    return record


def run_fresh_typed_source_free_preflight(
    *,
    modules: SnapshotModules,
    runtime: RuntimeClosure,
    producer: ProducerClosure,
    cp: Any,
    journal: AccessJournal,
) -> PreflightBundle:
    require(runtime.provenance_ledger is not None and runtime.provenance_ledger.active, "import/native enforcement must be active before preflight")
    runtime.provenance_ledger.checkpoint("BEFORE_AUTHORITATIVE_PREFLIGHT")
    journal.preflight_started()
    common = modules.by_member["uwfa_common.py"]
    protocol = modules.by_member["protocol.py"]
    semantic = modules.by_member["universal_adapter.py"]
    codec = modules.by_member["container_codec.py"]
    stage = modules.by_member["stage0_census.py"]
    backend_module = modules.by_member["cupy_backend.py"]
    backend = backend_module.build_backend(cp)
    identity = _independent_gpu_identity(runtime, protocol, backend)
    all150 = stage.gpu_preflight_all_150(common, backend, producer.source_snapshot_root_sha256)
    representative = stage.representative_outer_fold_benchmark(
        common,
        protocol,
        codec,
        semantic,
        backend,
        producer.source_snapshot_root_sha256,
    )
    record = {
        "schema": "uwfa-sc-v8-bound-source-preflight",
        "source_snapshot_root_sha256": producer.source_snapshot_root_sha256,
        "all150": all150,
        "representative": representative,
        "independent_gpu_identity": identity,
    }
    receipt_sha = sha256(canonical_json(record))
    filler = "00" * 32
    binding = stage.BoundEvidence(
        baseline_plan_sha256=filler,
        baseline_score_sha256=filler,
        universal_decoder_sha256=filler,
        producer_manifest_sha256=producer.manifest.sha256,
        audit_bootstrap_sha256=filler,
        source_full_geometry_sha256=filler,
        source_structural_geometry_sha256=filler,
        extraction_program_sha256=filler,
        universal_adapter_sha256=filler,
        pipeline_sha256=filler,
        source_snapshot_root_sha256=producer.source_snapshot_root_sha256,
        source_preflight_receipt_sha256=receipt_sha,
    )
    typed = stage.SourcePreflightEvidence(all150, representative, identity, receipt_sha)
    validated = stage.validate_source_preflight(common, protocol, typed, binding)
    require(validated["receipt_sha256"] == receipt_sha, "fresh typed preflight validator digest")
    runtime.verify_stable()
    producer.verify_stable()
    runtime.provenance_ledger.checkpoint("AFTER_AUTHORITATIVE_PREFLIGHT")
    journal.preflight_passed()
    return PreflightBundle(all150, representative, identity, record, receipt_sha, typed, backend)


@dataclass(frozen=True)
class InputSpec:
    path: str
    bytes: int
    sha256: str


def validate_baseline_plan(
    data: bytes,
    *,
    public_git_commit: str,
    artifact: InputSpec,
    legacy_independent_audit_sha256: str,
    original_source_binding_sha256: str,
) -> dict[str, Any]:
    """Parse the held plan and bind every score/source dependency it names."""

    record = strict_json(data, label="baseline plan")
    require(canonical_json(record) == data, "baseline plan must use exact canonical JSON encoding")
    exact_fields(
        record,
        {
            "schema",
            "status",
            "producer_public_git_commit",
            "artifact",
            "weights",
            "matrix_ordinals",
            "matrix_roles",
            "score_normalization",
            "legacy_independent_audit_sha256",
            "original_source_binding_sha256",
            "plan_sha256",
        },
        label="baseline plan",
    )
    require(record["schema"] == "uwfa-sc-v8-authenticated-baseline-plan-v1", "baseline plan schema")
    require(record["status"] == "FROZEN_INDEPENDENTLY_REVIEWED_BASELINE_PLAN", "baseline plan status")
    require(record["producer_public_git_commit"] == public_commit(public_git_commit), "baseline plan/public commit binding")
    artifact_row = exact_fields(record["artifact"], {"bytes", "sha256"}, label="baseline plan artifact")
    require(artifact_row == {"bytes": artifact.bytes, "sha256": artifact.sha256}, "baseline plan/artifact binding")
    require(type(record["weights"]) is int and record["weights"] == 18 * 768 * 2048, "baseline plan weights")
    require(record["matrix_ordinals"] == list(range(18)), "baseline plan exact matrix order")
    require(record["matrix_roles"] == [("gate", "up", "down")[index % 3] for index in range(18)], "baseline plan exact role order")
    require(record["score_normalization"] == "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM", "baseline plan score normalization")
    require(
        record["legacy_independent_audit_sha256"] == digest(legacy_independent_audit_sha256, "baseline plan legacy audit digest"),
        "baseline plan/legacy audit binding",
    )
    require(
        record["original_source_binding_sha256"] == digest(original_source_binding_sha256, "baseline plan source binding digest"),
        "baseline plan/original-source binding",
    )
    claimed = digest(record["plan_sha256"], "baseline plan internal seal")
    clean = dict(record)
    clean.pop("plan_sha256")
    require(sha256(canonical_json(clean)) == claimed, "baseline plan internal seal")
    return record


@dataclass
class SourceInputs:
    request: HeldAbsoluteFile
    request_record: dict[str, Any]
    inputs: dict[str, HeldAbsoluteFile]
    specs: dict[str, InputSpec]
    baseline_plan_record: dict[str, Any]

    def verify_stable(self) -> None:
        self.request.verify_stable()
        for member in self.inputs.values():
            member.verify_stable()

    def close(self) -> None:
        for member in reversed(list(self.inputs.values())):
            member.close()
        self.inputs.clear()
        self.request.close()


SOURCE_INPUT_NAMES = frozenset({
    "artifact",
    "baseline_plan",
    "legacy_independent_audit",
    "original_source_binding",
})


def _input_spec(value: Any, *, label: str) -> InputSpec:
    row = exact_fields(value, {"path", "bytes", "sha256"}, label=label)
    _absolute_components(str(row["path"]), label=f"{label}.path")
    require(type(row["bytes"]) is int and 0 < row["bytes"] <= (16 << 30), f"{label}.bytes")
    return InputSpec(str(row["path"]), int(row["bytes"]), digest(row["sha256"], f"{label}.sha256"))


def open_source_inputs_after_preflight(
    *,
    request_path: str,
    request_sha256: str,
    public_commit_pin: str,
    external_authority: ExternalLaunchAuthority,
    journal: AccessJournal,
) -> SourceInputs:
    require(isinstance(external_authority, ExternalLaunchAuthority), "typed request authority required")
    require(request_sha256 == external_authority.request_sha256, "CLI request digest/external authority mismatch")
    journal.before_payload_path_access("source-request")
    request = HeldAbsoluteFile.open(request_path, cap=4 << 20, label="source request")
    opened: dict[str, HeldAbsoluteFile] = {}
    try:
        require(request.member.sha256 == digest(request_sha256, "source request CLI digest"), "source request digest mismatch")
        require(request.member.sha256 == external_authority.request_sha256, "source request external authority pin mismatch")
        record = strict_json(request.member.data, label="source request")
        require(canonical_json(record) == request.member.data, "source request must use exact canonical JSON encoding")
        exact_fields(record, {"schema", "status", "producer_public_commit", "transaction_id", "output_parent", "final_name", "inputs"}, label="source request")
        require(record["schema"] == "uwfa-sc-v8-external-source-phase-request-v1", "source request schema")
        require(record["status"] == "AUTHORIZED_SOURCE_ONLY_NO_CONTROLS", "source request status")
        require(record["producer_public_commit"] == public_commit(public_commit_pin), "source request/public commit binding")
        require(isinstance(record["transaction_id"], str) and _TXID.fullmatch(record["transaction_id"]) is not None, "source request transaction id")
        safe_name(record["final_name"], "source request final name")
        _absolute_components(str(record["output_parent"]), label="source request output parent")
        input_rows = record["inputs"]
        require(isinstance(input_rows, dict) and set(input_rows) == set(SOURCE_INPUT_NAMES), "source request input member set")
        specs = {name: _input_spec(input_rows[name], label=f"source input {name}") for name in sorted(SOURCE_INPUT_NAMES)}
        require(specs["baseline_plan"].sha256 == external_authority.baseline_plan_sha256, "baseline plan external authority pin mismatch")
        require(specs["legacy_independent_audit"].sha256 == external_authority.legacy_independent_audit_sha256, "legacy audit external authority pin mismatch")
        require(specs["original_source_binding"].sha256 == external_authority.original_source_binding_sha256, "original-source binding external authority pin mismatch")
        for name in sorted(SOURCE_INPUT_NAMES):
            journal.before_payload_path_access(name)
            member = HeldAbsoluteFile.open(specs[name].path, cap=max(specs[name].bytes, 1), label=f"source input {name}")
            require(len(member.member.data) == specs[name].bytes and member.member.sha256 == specs[name].sha256, f"source input binding: {name}")
            opened[name] = member
        reject_authority_request_output_inode_aliasing(
            authority={},
            request={
                "source-request": request.member.identity,
                **{name: member.member.identity for name, member in opened.items()},
            },
            output={},
        )
        baseline_plan_record = validate_baseline_plan(
            opened["baseline_plan"].member.data,
            public_git_commit=public_commit_pin,
            artifact=specs["artifact"],
            legacy_independent_audit_sha256=specs["legacy_independent_audit"].sha256,
            original_source_binding_sha256=specs["original_source_binding"].sha256,
        )
        result = SourceInputs(request, record, opened, specs, baseline_plan_record)
        result.verify_stable()
        return result
    except Exception:
        for member in reversed(list(opened.values())):
            member.close()
        request.close()
        raise


def _finite_positive(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} numeric")
    result = float(value)
    require(result > 0.0 and result < float("inf"), f"{label} finite positive")
    return result


def _fp64_close(left: Any, right: Any, *, label: str, abs_tol: float = 1e-12) -> None:
    import math

    a = _finite_positive(left, f"{label} left")
    b = _finite_positive(right, f"{label} right")
    require(math.isclose(a, b, rel_tol=32.0 * math.ulp(1.0), abs_tol=abs_tol), f"{label} mismatch")


def validate_original_source_binding(data: bytes, *, legacy_audit_sha256: str, weights: int) -> dict[str, Any]:
    record = strict_json(data, label="original source binding")
    exact_fields(
        record,
        {"schema", "status", "weights", "matrices", "sources_canonical_sha256", "source_energy_fp64", "legacy_independent_audit_sha256", "source_binding_sha256"},
        label="original source binding",
    )
    require(record["schema"] == "uwfa-qwen-original-source-panel-binding-v8", "original source binding schema")
    require(record["status"] == "PASS_INDEPENDENT_ORIGINAL_SOURCE_BINDING", "original source binding status")
    require(type(record["weights"]) is int and record["weights"] == weights, "original source binding weights")
    require(record["legacy_independent_audit_sha256"] == digest(legacy_audit_sha256, "legacy audit digest"), "source binding/legacy audit digest")
    digest(record["sources_canonical_sha256"], "source canonical digest")
    energy = _finite_positive(record["source_energy_fp64"], "source binding energy")
    matrices = record["matrices"]
    require(isinstance(matrices, list) and len(matrices) == 18, "source binding matrix count")
    seen: set[int] = set()
    energy_sum = 0.0
    roles: dict[int, str] = {}
    for index, raw in enumerate(matrices):
        row = exact_fields(raw, {"matrix_ordinal", "tensor", "role", "shape", "source_bf16_sha256", "source_energy_fp64"}, label=f"source binding matrix[{index}]")
        ordinal = row["matrix_ordinal"]
        require(type(ordinal) is int and 0 <= ordinal < 18 and ordinal not in seen, "source binding matrix ordinal")
        seen.add(ordinal)
        role = row["role"]
        require(role in {"gate", "up", "down"}, "source binding matrix role")
        shape = row["shape"]
        expected_shape = [2048, 768] if role == "down" else [768, 2048]
        require(shape == expected_shape, "source binding matrix shape")
        require(isinstance(row["tensor"], str) and row["tensor"], "source binding tensor")
        digest(row["source_bf16_sha256"], "source BF16 digest")
        energy_sum += _finite_positive(row["source_energy_fp64"], "source matrix energy")
        roles[ordinal] = role
    require(seen == set(range(18)), "source binding complete matrix ordinals")
    require(all(roles[index] == ("gate", "up", "down")[index % 3] for index in range(18)), "source binding role order")
    _fp64_close(energy_sum, energy, label="source binding energy sum", abs_tol=5e-12)
    claimed = digest(record["source_binding_sha256"], "source binding internal seal")
    clean = dict(record)
    clean.pop("source_binding_sha256")
    require(sha256(canonical_json(clean)) == claimed, "original source binding internal seal")
    return record


def construct_bound_baseline_score(
    *,
    artifact_bytes: bytes,
    panel: Mapping[str, Any],
    source_full_geometry_sha256: str,
    universal_decoder_sha256: str,
    legacy_audit_bytes: bytes,
    original_source_binding_bytes: bytes,
) -> tuple[dict[str, Any], bytes]:
    artifact_sha = sha256(artifact_bytes)
    weights = int(panel["weights"])
    legacy_sha = sha256(legacy_audit_bytes)
    legacy = strict_json(legacy_audit_bytes, label="legacy independent audit")
    require(legacy.get("schema") == "strata_expert_affine_independent_audit_v1" and legacy.get("status") == "passed", "legacy audit schema/status")
    container = legacy.get("container")
    require(isinstance(container, dict), "legacy audit container")
    require(container.get("sha256") == artifact_sha and container.get("physical_bytes") == len(artifact_bytes), "legacy audit artifact binding")
    decode = legacy.get("decode")
    require(isinstance(decode, dict) and decode.get("canonical_reencode_all_match") is True and decode.get("every_group_once") is True, "legacy audit decode integrity")
    score = legacy.get("source_score")
    require(isinstance(score, dict), "legacy audit source score")
    sse = _finite_positive(score.get("sse_sum_fp64"), "legacy SSE")
    energy = _finite_positive(score.get("source_energy_sum_fp64"), "legacy source energy")
    mse = _finite_positive(score.get("energy_weighted_relative_mse"), "legacy relative MSE")
    import math

    require(abs(sse / energy - mse) <= 4.0 * math.ulp(sse / energy), "legacy relative MSE arithmetic")
    binding = validate_original_source_binding(original_source_binding_bytes, legacy_audit_sha256=legacy_sha, weights=weights)
    _fp64_close(binding["source_energy_fp64"], energy, label="legacy/source-binding energy")
    legacy_bindings = legacy.get("bindings")
    require(isinstance(legacy_bindings, dict), "legacy audit bindings")
    require(legacy_bindings.get("sources_canonical_sha256") == binding["sources_canonical_sha256"], "legacy/source canonical binding")
    matrices = score.get("matrices")
    require(isinstance(matrices, list) and len(matrices) == 18, "legacy score matrix panel")
    source_by_ordinal = {row["matrix_ordinal"]: row for row in binding["matrices"]}
    for ordinal, row in enumerate(matrices):
        require(isinstance(row, dict), "legacy matrix score row")
        source = source_by_ordinal[ordinal]
        require(row.get("matrix_ordinal") == ordinal, "legacy matrix ordinal")
        require(row.get("role") == source["role"] and row.get("shape") == source["shape"], "legacy/source matrix geometry")
        require(row.get("source_bf16_sha256") == source["source_bf16_sha256"], "legacy/source matrix digest")
        _fp64_close(row.get("source_energy_fp64"), source["source_energy_fp64"], label=f"legacy/source matrix energy {ordinal}")
    reconstruction = panel.get("reconstruction")
    require(isinstance(reconstruction, dict), "adapter reconstruction record")
    reconstruction_sha = digest(reconstruction.get("full_reconstruction_f64_sha256"), "adapter matrix-order reconstruction")
    record: dict[str, Any] = {
        "schema": "uwfa-bound-baseline-score-v8",
        "status": "PASS_INDEPENDENT_BASELINE_SCORE",
        "artifact_sha256": artifact_sha,
        "artifact_bytes": len(artifact_bytes),
        "weights": weights,
        "relative_mse": mse,
        "sse_fp64": sse,
        "source_energy_fp64": energy,
        "normalization": "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
        "reconstruction_f64_sha256": reconstruction_sha,
        "original_source_panel_sha256": digest(source_full_geometry_sha256, "source full geometry"),
        "independent_decoder_source_sha256": digest(universal_decoder_sha256, "universal decoder root"),
    }
    record["score_receipt_sha256"] = sha256(canonical_json(record))
    encoded = canonical_json(record)
    return record, encoded


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "float": float(value),
    }


def _parse_fraction_record(value: Any, *, label: str, positive: bool = True) -> Fraction:
    row = exact_fields(value, {"numerator", "denominator", "exact", "float"}, label=label)
    require(type(row["numerator"]) is int and type(row["denominator"]) is int and row["denominator"] > 0, f"{label} integer fraction")
    result = Fraction(row["numerator"], row["denominator"])
    require(not positive or result > 0, f"{label} positive")
    require(row["exact"] == f"{result.numerator}/{result.denominator}", f"{label} exact spelling")
    require(type(row["float"]) in (int, float) and float(row["float"]) == float(result), f"{label} float projection")
    return result


def _request_range_summary(ranges: Any, *, total_bytes: int) -> tuple[int, int, int, int, set[int]]:
    require(isinstance(ranges, list), "routed request ranges")
    normalized: list[tuple[int, int]] = []
    repeated = 0
    pages: set[int] = set()
    for raw in ranges:
        require(isinstance(raw, list) and len(raw) == 2, "routed request range row")
        begin, end = raw
        require(type(begin) is int and type(end) is int and 0 <= begin <= end <= total_bytes, "routed request range bounds")
        normalized.append((begin, end))
        repeated += end - begin
        if end > begin:
            pages.update(range(begin // 4096, (end - 1) // 4096 + 1))
    unique = 0
    if normalized:
        left, right = sorted(normalized)[0]
        for begin, end in sorted(normalized)[1:]:
            if begin > right:
                unique += right - left
                left, right = begin, end
            else:
                right = max(right, end)
        unique += right - left
    return len(normalized), repeated, unique, repeated - unique, pages


@dataclass(frozen=True)
class AuthenticatedContainerFraming:
    literal_container_sha256: str
    literal_container_bytes: int
    attributable_total: tuple[Fraction, ...]
    attributable_nonpadding: tuple[Fraction, ...]
    universal_decoder_sha256: str
    framing_root_sha256: str


def derive_authenticated_container_framing(
    literal_container: bytes,
    *,
    common: Any,
    semantic_codec: Any,
    container_codec: Any,
    universal_decoder_sha256: str,
) -> AuthenticatedContainerFraming:
    """Parse the literal bytes with held codec images and derive denominators."""

    require(isinstance(literal_container, bytes) and literal_container, "literal container bytes")
    require(callable(getattr(container_codec, "parse_container", None)), "authenticated container parser ABI")
    require(callable(getattr(container_codec, "owner_ordinals", None)), "authenticated owner framing ABI")
    parsed = container_codec.parse_container(common, semantic_codec, literal_container)
    require(isinstance(parsed, dict) and bytes(parsed.get("raw", b"")) == literal_container, "authenticated parser literal-byte rebinding")
    experts = parsed.get("experts")
    require(type(experts) is int and experts > 0, "authenticated framing expert count")
    ledger = parsed.get("byte_ledger")
    require(isinstance(ledger, (tuple, list)) and ledger, "authenticated framing byte ledger")
    total = [Fraction(0, 1) for _ in range(experts)]
    nonpadding = [Fraction(0, 1) for _ in range(experts)]
    allocation = Fraction(0, 1)
    ledger_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(ledger):
        require(isinstance(raw, Mapping), f"authenticated framing ledger[{index}]")
        size = raw.get("bytes")
        padding = raw.get("padding")
        owner_set = raw.get("owner_set")
        require(type(size) is int and size >= 0 and type(padding) is bool, "authenticated framing ledger value")
        require(isinstance(owner_set, bytes), "authenticated framing owner set")
        owners = tuple(container_codec.owner_ordinals(owner_set, experts))
        require(owners and all(type(owner) is int for owner in owners), "authenticated framing built-in owner ordinals")
        require(tuple(sorted(set(owners))) == owners and 0 <= owners[0] <= owners[-1] < experts, "authenticated framing owner order")
        share = Fraction(size, len(owners))
        for owner in owners:
            total[owner] += share
            if not padding:
                nonpadding[owner] += share
            allocation += share
        ledger_rows.append({"bytes": size, "padding": padding, "owners": list(owners)})
    require(allocation == len(literal_container), "authenticated framing allocation/container length")
    require(all(value > 0 for value in total) and all(value > 0 for value in nonpadding), "authenticated framing positive denominators")
    decoder_sha = digest(universal_decoder_sha256, "authenticated framing decoder root")
    framing_body = {
        "literal_container_sha256": sha256(literal_container),
        "literal_container_bytes": len(literal_container),
        "experts": experts,
        "attributable_total": [_fraction_record(value) for value in total],
        "attributable_nonpadding": [_fraction_record(value) for value in nonpadding],
        "ledger": ledger_rows,
        "universal_decoder_sha256": decoder_sha,
    }
    framing_root = length_prefixed_root(
        b"UWFA-SC-V8-AUTHENTICATED-CONTAINER-FRAMING-V1\0",
        (canonical_json(framing_body),),
    )
    return AuthenticatedContainerFraming(
        sha256(literal_container),
        len(literal_container),
        tuple(total),
        tuple(nonpadding),
        decoder_sha,
        framing_root,
    )


def validate_external_bandwidth_gate(
    literal_container: bytes,
    authenticated_framing: AuthenticatedContainerFraming,
    metrics: Any,
) -> dict[str, Any]:
    require(isinstance(literal_container, bytes) and literal_container, "bandwidth gate literal container")
    require(isinstance(authenticated_framing, AuthenticatedContainerFraming), "bandwidth gate authenticated framing type")
    require(authenticated_framing.literal_container_sha256 == sha256(literal_container), "bandwidth gate framing/container digest")
    require(authenticated_framing.literal_container_bytes == len(literal_container), "bandwidth gate framing/container length")
    require(isinstance(metrics, dict), "physical metrics object")
    require(metrics.get("routed_io_authoritative_descriptor_backed") is True, "bandwidth gate requires authoritative descriptor routing")
    require(metrics.get("installation_authentication_reported_separately") is not None, "bandwidth gate installation authentication record")
    total_bytes = len(literal_container)
    require(metrics.get("actual_container_bytes") == total_bytes, "bandwidth gate reported/literal container bytes")
    rows = metrics.get("experts")
    require(
        isinstance(rows, list)
        and len(rows) == len(authenticated_framing.attributable_total)
        and rows,
        "bandwidth gate exact framed expert set",
    )
    repeated_sum = unique_sum = overlap_sum = request_count_sum = touched_sum = 0
    maximum_repeated = Fraction(0, 1)
    maximum_coalesced = Fraction(0, 1)
    maximum_page = Fraction(0, 1)
    output_rows = []
    for expected_ordinal, raw in enumerate(rows):
        require(isinstance(raw, dict), "bandwidth expert row")
        require(raw.get("expert_ordinal") == expected_ordinal, "bandwidth expert order")
        count, repeated, coalesced, overlap, pages = _request_range_summary(raw.get("instrumented_routed_read_ranges"), total_bytes=total_bytes)
        require(raw.get("instrumented_routed_read_request_count") == count, "bandwidth request count recomputation")
        require(raw.get("instrumented_routed_requested_bytes_with_repetition") == repeated, "bandwidth repeated-byte recomputation")
        require(raw.get("instrumented_routed_unique_requested_bytes") == coalesced, "bandwidth coalesced-byte recomputation")
        require(raw.get("instrumented_routed_overlap_bytes_requested_again") == overlap, "bandwidth overlap-byte recomputation")
        touched = len(pages) * 4096
        require(raw.get("touched_page_indices") == sorted(pages), "bandwidth touched page set recomputation")
        require(raw.get("touched_page_bytes") == touched, "bandwidth touched page bytes recomputation")
        attributable_total = _parse_fraction_record(raw.get("attributable_total_physical_bytes"), label="attributable total")
        attributable_nonpadding = _parse_fraction_record(raw.get("attributable_nonpadding_decodable_bytes"), label="attributable nonpadding")
        require(attributable_total == authenticated_framing.attributable_total[expected_ordinal], "bandwidth total denominator/framing mismatch")
        require(attributable_nonpadding == authenticated_framing.attributable_nonpadding[expected_ordinal], "bandwidth nonpadding denominator/framing mismatch")
        repeated_ratio = max(Fraction(repeated, 1) / attributable_total, Fraction(repeated, 1) / attributable_nonpadding)
        coalesced_ratio = max(Fraction(coalesced, 1) / attributable_total, Fraction(coalesced, 1) / attributable_nonpadding)
        page_ratio = max(Fraction(touched, 1) / attributable_total, Fraction(touched, 1) / attributable_nonpadding)
        producer_ratio = _parse_fraction_record(raw.get("strict_cold_amplification"), label="producer strict cold ratio")
        require(page_ratio == producer_ratio, "producer/external unique-page ratio mismatch")
        maximum_repeated = max(maximum_repeated, repeated_ratio)
        maximum_coalesced = max(maximum_coalesced, coalesced_ratio)
        maximum_page = max(maximum_page, page_ratio)
        repeated_sum += repeated
        unique_sum += coalesced
        overlap_sum += overlap
        request_count_sum += count
        touched_sum += touched
        output_rows.append({
            "expert_ordinal": expected_ordinal,
            "read_request_count": count,
            "requested_bytes_with_repetition": repeated,
            "coalesced_unique_requested_bytes": coalesced,
            "overlap_bytes_requested_again": overlap,
            "unique_touched_page_bytes": touched,
            "strict_repeated_request_amplification": _fraction_record(repeated_ratio),
            "strict_coalesced_request_amplification": _fraction_record(coalesced_ratio),
            "strict_unique_page_amplification": _fraction_record(page_ratio),
            "passes_repeated_request_below_2x": repeated_ratio < 2,
            "passes_coalesced_request_below_2x": coalesced_ratio < 2,
            "passes_unique_page_below_2x": page_ratio < 2,
        })
    aggregate = metrics.get("routed_read_request_aggregates")
    require(isinstance(aggregate, dict), "producer routed request aggregate")
    require(aggregate.get("read_request_count_sum_across_experts") == request_count_sum, "aggregate request count")
    require(aggregate.get("requested_bytes_with_repetition_sum_across_experts") == repeated_sum, "aggregate repeated bytes")
    require(aggregate.get("unique_requested_bytes_sum_across_experts") == unique_sum, "aggregate coalesced bytes")
    require(aggregate.get("overlap_bytes_requested_again_sum_across_experts") == overlap_sum, "aggregate overlap bytes")
    require(aggregate.get("unique_touched_page_bytes_sum_across_experts") == touched_sum, "aggregate touched pages")
    require(aggregate.get("frozen_cold_gate_uses_unique_touched_page_bytes_only") is True, "producer cold-gate declaration")
    producer_max = _parse_fraction_record(metrics.get("maximum_strict_cold_read_amplification"), label="producer maximum cold ratio")
    require(producer_max == maximum_page, "producer maximum cold ratio recomputation")
    producer_pass = metrics.get("passes_cold_read_below_2x") is True
    repeated_pass = maximum_repeated < 2
    coalesced_pass = maximum_coalesced < 2
    page_pass = maximum_page < 2
    return {
        "schema": "uwfa-sc-v8-external-repeated-coalesced-bandwidth-gate-v1",
        "status": "PASS_STRICT_ALL_BANDWIDTH_GATES" if producer_pass and repeated_pass and coalesced_pass and page_pass else "FAIL_STRICT_BANDWIDTH_GATE",
        "gate_definition": "conjunction of descriptor-backed unique-page, literal repeated-request, and coalesced-range amplification; each uses the worse exact owner-local total/nonpadding denominator and is strictly below 2",
        "experts": output_rows,
        "maximum_strict_repeated_request_amplification": _fraction_record(maximum_repeated),
        "maximum_strict_coalesced_request_amplification": _fraction_record(maximum_coalesced),
        "maximum_strict_unique_page_amplification": _fraction_record(maximum_page),
        "producer_unique_page_gate_passed": producer_pass,
        "repeated_request_gate_passed": repeated_pass,
        "coalesced_request_gate_passed": coalesced_pass,
        "independently_recomputed_unique_page_gate_passed": page_pass,
        "passes_all_bandwidth_gates": producer_pass and repeated_pass and coalesced_pass and page_pass,
        "installation_authentication_excluded_and_reported_separately": True,
        "literal_container_sha256": authenticated_framing.literal_container_sha256,
        "literal_container_bytes": authenticated_framing.literal_container_bytes,
        "authenticated_framing_root_sha256": authenticated_framing.framing_root_sha256,
        "universal_decoder_sha256": authenticated_framing.universal_decoder_sha256,
    }


def _closure_roots(
    *,
    producer: ProducerClosure,
    dispatcher: DispatcherClosure,
    runtime: RuntimeClosure,
    decoder: DecoderClosure,
    public_git_commit: str,
    inputs: SourceInputs,
) -> dict[str, str]:
    producer_manifest_by_name = {row["name"]: row for row in producer.manifest_rows}
    bootstrap_sha = dispatcher.members["bootstrap.py"].sha256
    universal_adapter_sha = producer_manifest_by_name["universal_adapter.py"]["sha256"]
    decoder_rows = decoder.bundle_record["members"]
    extraction_rows = [
        canonical_json({
            "kind": row["kind"],
            "logical_name": row["logical_name"],
            "sha256": row["sha256"],
        })
        for row in decoder_rows
    ]
    extraction_root = length_prefixed_root(
        b"UWFA-SC-V8-EXTRACTION-PROGRAM-V1\0",
        extraction_rows
        + [
            canonical_json({
                "logical_to_producer_member_map_sha256": decoder.logical_to_producer_member_map_sha256,
                "numpy_strata_ordinal_bridge_sha256": decoder.ordinal_bridge_sha256,
            })
        ],
    )
    pipeline_root = length_prefixed_root(
        b"UWFA-SC-V8-EXTERNAL-SOURCE-PIPELINE-V1\0",
        (
            producer.manifest.sha256,
            producer.source_snapshot_root_sha256,
            dispatcher.manifest.sha256,
            dispatcher.audit.member.sha256,
            public_git_commit,
            bootstrap_sha,
            sha256(runtime.lock_bytes),
            runtime.tree_manifest_sha256,
            decoder.universal_decoder_sha256,
            decoder.logical_to_producer_member_map_sha256,
            decoder.ordinal_bridge_sha256,
            extraction_root,
            inputs.request.member.sha256,
            inputs.inputs["baseline_plan"].member.sha256,
            inputs.inputs["legacy_independent_audit"].member.sha256,
            inputs.inputs["original_source_binding"].member.sha256,
        ),
    )
    return {
        "audit_bootstrap_sha256": bootstrap_sha,
        "universal_adapter_sha256": universal_adapter_sha,
        "universal_decoder_sha256": decoder.universal_decoder_sha256,
        "logical_to_producer_member_map_sha256": decoder.logical_to_producer_member_map_sha256,
        "numpy_strata_ordinal_bridge_sha256": decoder.ordinal_bridge_sha256,
        "extraction_program_sha256": extraction_root,
        "pipeline_sha256": pipeline_root,
    }


def _memfd_descriptor_source_builder(container_codec: Any) -> Callable[[bytes], Any]:
    def build(raw: bytes) -> Any:
        require(isinstance(raw, bytes), "memfd container bytes")
        require(os.name == "posix" and hasattr(os, "memfd_create"), "sealed memfd routed source required")
        allow_sealing = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
        cloexec = getattr(os, "MFD_CLOEXEC", 0x0001)
        fd = os.memfd_create("uwfa-v8-routed-container", flags=allow_sealing | cloexec)
        try:
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                amount = os.write(fd, view[offset:])
                require(amount > 0, "memfd short write")
                offset += amount
            os.fsync(fd)
            import fcntl

            seals = (
                getattr(fcntl, "F_SEAL_SEAL", 0x0001)
                | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
                | getattr(fcntl, "F_SEAL_GROW", 0x0004)
                | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
            )
            fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), seals)
            source = container_codec.AuthenticatedDescriptorSource(fd, sha256(raw))
            return source
        finally:
            os.close(fd)

    return build


@dataclass
class SourceComputation:
    result: dict[str, Any]
    public_result: dict[str, Any]
    score_record: dict[str, Any]
    score_bytes: bytes
    bindings: Any
    bandwidth_gate: dict[str, Any]
    closure_roots: dict[str, str]


def expected_publication_member_names(has_literal_container: bool) -> frozenset[str]:
    require(type(has_literal_container) is bool, "publication container presence flag")
    names = {
        "RUN_STATE.json",
        "LAUNCH_RECEIPT.json",
        "RUNTIME_LOCK.authenticated.json",
        "SOURCE_PREFLIGHT.json",
        "BOUND_BASELINE_SCORE.json",
        "SOURCE_PHASE.json",
        "BANDWIDTH_GATE.json",
        "IMPORT_NATIVE_EVENT_LEDGER.json",
        "COMPLETE.json",
    }
    if has_literal_container:
        names.update({"UWFCV8.bin", "IDENTITY_FRAMING.bin", "POSTERIOR_HANDOFF.json"})
    return frozenset(names)


def run_source_computation(
    *,
    producer: ProducerClosure,
    dispatcher: DispatcherClosure,
    runtime: RuntimeClosure,
    decoder: DecoderClosure,
    modules: SnapshotModules,
    np: Any,
    preflight: PreflightBundle,
    inputs: SourceInputs,
    public_git_commit: str,
) -> SourceComputation:
    common = modules.by_member["uwfa_common.py"]
    protocol = modules.by_member["protocol.py"]
    semantic = modules.by_member["universal_adapter.py"]
    codec = modules.by_member["container_codec.py"]
    stage = modules.by_member["stage0_census.py"]
    adapter_module = modules.by_member["strata_sc_adapter.py"]
    external = decoder.external_modules
    require(
        set(external) == {"strata_format_common", "independent_strata_decoder", "numpy_strata_ordinal_bridge"},
        "compiled external decoder modules",
    )
    bridged_strata_common = external["numpy_strata_ordinal_bridge"].wrap_strata_common(
        external["strata_format_common"], np
    )
    adapter = adapter_module.StrataSCAdapter(
        common=common,
        semantic_codec=semantic,
        np=np,
        frozen_auditor=external["independent_strata_decoder"],
        strata_common=bridged_strata_common,
        device="cupy",
    )
    artifact = inputs.inputs["artifact"].member.data
    panel = stage.prepare_panel(protocol, adapter, artifact)
    require(int(panel["weights"]) == inputs.baseline_plan_record["weights"], "baseline plan/panel weight binding")
    source_full_geometry = protocol.geometry_sha256(common, panel)
    source_structural_geometry = protocol.structural_geometry_sha256(common, panel)
    roots = _closure_roots(
        producer=producer,
        dispatcher=dispatcher,
        runtime=runtime,
        decoder=decoder,
        public_git_commit=public_git_commit,
        inputs=inputs,
    )
    score_record, score_bytes = construct_bound_baseline_score(
        artifact_bytes=artifact,
        panel=panel,
        source_full_geometry_sha256=source_full_geometry,
        universal_decoder_sha256=roots["universal_decoder_sha256"],
        legacy_audit_bytes=inputs.inputs["legacy_independent_audit"].member.data,
        original_source_binding_bytes=inputs.inputs["original_source_binding"].member.data,
    )
    bindings = stage.BoundEvidence(
        baseline_plan_sha256=inputs.inputs["baseline_plan"].member.sha256,
        baseline_score_sha256=sha256(score_bytes),
        universal_decoder_sha256=roots["universal_decoder_sha256"],
        producer_manifest_sha256=producer.manifest.sha256,
        audit_bootstrap_sha256=roots["audit_bootstrap_sha256"],
        source_full_geometry_sha256=source_full_geometry,
        source_structural_geometry_sha256=source_structural_geometry,
        extraction_program_sha256=roots["extraction_program_sha256"],
        universal_adapter_sha256=roots["universal_adapter_sha256"],
        pipeline_sha256=roots["pipeline_sha256"],
        source_snapshot_root_sha256=producer.source_snapshot_root_sha256,
        source_preflight_receipt_sha256=preflight.receipt_sha256,
    )
    # The producer validates this exact derived receipt again inside
    # source_phase; no caller-supplied geometry or score hash enters.
    protocol.validate_score_receipt(
        score_record,
        artifact_sha256=sha256(artifact),
        artifact_bytes=len(artifact),
        weights=int(panel["weights"]),
        reconstruction_sha256=str(panel["reconstruction"]["full_reconstruction_f64_sha256"]),
        original_source_panel_sha256=source_full_geometry,
        independent_decoder_source_sha256=roots["universal_decoder_sha256"],
    )
    result = stage.source_phase(
        common=common,
        protocol=protocol,
        container_codec=codec,
        semantic_codec=semantic,
        adapter=adapter,
        backend=preflight.backend,
        artifact_bytes=artifact,
        score_receipt_bytes=score_bytes,
        bindings=bindings,
        source_preflight=preflight.typed_evidence,
        authenticated_descriptor_source_builder=_memfd_descriptor_source_builder(codec),
    )
    require(isinstance(result, dict) and result.get("schema") == "uwfa-sc-v8-source-phase-result", "producer source result schema")
    public_result = {key: value for key, value in result.items() if not key.startswith("_")}
    require(not any(key.startswith("_") for key in public_result), "private source result escaped")
    container = result.get("_container")
    if container is None:
        bandwidth = {
            "schema": "uwfa-sc-v8-external-repeated-coalesced-bandwidth-gate-v1",
            "status": "NOT_APPLICABLE_NO_LITERAL_SOURCE_CONTAINER",
            "passes_all_bandwidth_gates": False,
            "producer_source_status": result.get("status"),
        }
    else:
        require(isinstance(container, bytes), "source container type")
        framing = derive_authenticated_container_framing(
            container,
            common=common,
            semantic_codec=semantic,
            container_codec=codec,
            universal_decoder_sha256=decoder.universal_decoder_sha256,
        )
        metrics = public_result.get("source_final", {}).get("parsed_metrics")
        bandwidth = validate_external_bandwidth_gate(container, framing, metrics)
        if not bandwidth["passes_all_bandwidth_gates"]:
            public_result["external_dispatcher_status_override"] = "FAIL_EXTERNAL_REPEATED_OR_COALESCED_BANDWIDTH_GATE"
            public_result["controls_may_be_opened"] = False
    inputs.verify_stable()
    producer.verify_stable()
    dispatcher.verify_stable()
    runtime.verify_stable()
    decoder.verify_stable()
    return SourceComputation(result, public_result, score_record, score_bytes, bindings, bandwidth, roots)


def _publication_source_root(
    *,
    producer: ProducerClosure,
    dispatcher: DispatcherClosure,
    runtime: RuntimeClosure,
    decoder: DecoderClosure,
    public_git_commit: str,
    external_authority: ExternalLaunchAuthority,
) -> str:
    return length_prefixed_root(
        b"UWFA-SC-V8-EXTERNAL-PUBLICATION-SOURCE-ROOT-V2\0",
        (
            producer.manifest.sha256,
            producer.review.member.sha256,
            producer.source_snapshot_root_sha256,
            dispatcher.manifest.sha256,
            dispatcher.audit.member.sha256,
            sha256(runtime.lock_bytes),
            runtime.tree_manifest_sha256,
            decoder.universal_decoder_sha256,
            public_git_commit,
            external_authority.dispatcher_public_git_commit,
            external_authority.launcher_source_sha256,
            external_authority.launcher_review_sha256,
            external_authority.request_sha256,
            external_authority.baseline_plan_sha256,
            external_authority.legacy_independent_audit_sha256,
            external_authority.original_source_binding_sha256,
        ),
    )


def publish_source_result(
    *,
    producer: ProducerClosure,
    dispatcher: DispatcherClosure,
    runtime: RuntimeClosure,
    decoder: DecoderClosure,
    modules: SnapshotModules,
    preflight: PreflightBundle,
    inputs: SourceInputs,
    computation: SourceComputation,
    public_git_commit: str,
    external_authority: ExternalLaunchAuthority,
    journal: AccessJournal,
) -> dict[str, Any]:
    common = modules.by_member["uwfa_common.py"]
    envelope = modules.by_member["result_envelope.py"]
    request = inputs.request_record
    journal.before_payload_path_access("output-parent")
    held_parent = HeldDirectory.open_absolute(str(request["output_parent"]), label="output parent")
    authority_inodes: dict[str, Sequence[int]] = {
        "producer-package": producer.package.identity,
        "producer-review": producer.review.member.identity,
        "dispatcher-package": dispatcher.package.identity,
        "dispatcher-audit": dispatcher.audit.member.identity,
        "runtime-tree-manifest": runtime.tree_manifest.member.identity,
        "runtime-site-packages": runtime.site_packages.identity,
        **{f"producer-member:{name}": member.identity for name, member in producer.members.items()},
        **{f"dispatcher-member:{name}": member.identity for name, member in dispatcher.members.items()},
        **{f"runtime-member:{path}": member.member.identity for path, member in runtime.held_by_path.items()},
        **{f"decoder-external:{name}": member.member.identity for name, member in decoder.external_members.items()},
    }
    request_inodes: dict[str, Sequence[int]] = {
        "source-request": inputs.request.member.identity,
        **{f"source-input:{name}": member.member.identity for name, member in inputs.inputs.items()},
    }
    reject_authority_request_output_inode_aliasing(
        authority=authority_inodes,
        request=request_inodes,
        output={"output-parent": held_parent.identity},
    )
    authority = _publication_source_root(
        producer=producer,
        dispatcher=dispatcher,
        runtime=runtime,
        decoder=decoder,
        public_git_commit=public_git_commit,
        external_authority=external_authority,
    )
    retained_parent = common.RetainedOutputParent(
        held_parent.fd,
        (held_parent.identity[0], held_parent.identity[1]),
        authority,
    )
    try:
        container = computation.result.get("_container")
        identity = computation.result.get("_identity_framing_container")
        conditional = container is not None
        require((identity is not None) == conditional, "source container pair presence")
        launch_status = "SOURCE_PHASE_COMPLETE_AWAITING_EXTERNAL_RESULT_AUDIT"
        if conditional and computation.bandwidth_gate.get("passes_all_bandwidth_gates") is not True:
            launch_status = "FAIL_EXTERNAL_BANDWIDTH_GATE_AWAITING_EXTERNAL_RESULT_AUDIT"
        launch = {
            "schema": "uwfa-sc-v8-external-launch-receipt-v2",
            "status": launch_status,
            "producer_public_git_commit": public_git_commit,
            "producer_manifest_sha256": producer.manifest.sha256,
            "producer_final_review_sha256": producer.review.member.sha256,
            "producer_source_snapshot_root_sha256": producer.source_snapshot_root_sha256,
            "dispatcher_manifest_sha256": dispatcher.manifest.sha256,
            "dispatcher_audit_sha256": dispatcher.audit.member.sha256,
            "dispatcher_public_git_commit": external_authority.dispatcher_public_git_commit,
            "external_launcher_source_sha256": external_authority.launcher_source_sha256,
            "external_launcher_review_sha256": external_authority.launcher_review_sha256,
            "external_exact_request_sha256": external_authority.request_sha256,
            "external_baseline_plan_sha256": external_authority.baseline_plan_sha256,
            "external_legacy_independent_audit_sha256": external_authority.legacy_independent_audit_sha256,
            "external_original_source_binding_sha256": external_authority.original_source_binding_sha256,
            "runtime_lock_sha256": sha256(runtime.lock_bytes),
            "runtime_tree_manifest_sha256": runtime.tree_manifest_sha256,
            "universal_decoder_sha256": decoder.universal_decoder_sha256,
            "source_request_sha256": inputs.request.member.sha256,
            "authenticated_input_descriptors": {
                name: {"bytes": len(member.member.data), "sha256": member.member.sha256}
                for name, member in sorted(inputs.inputs.items())
            },
            "descriptor_inode_domains": {
                "authority": [
                    {"label": label, "device": int(identity[0]), "inode": int(identity[1])}
                    for label, identity in sorted(authority_inodes.items())
                ],
                "request": [
                    {"label": label, "device": int(identity[0]), "inode": int(identity[1])}
                    for label, identity in sorted(request_inodes.items())
                ],
                "output_parent": {
                    "device": int(held_parent.identity[0]),
                    "inode": int(held_parent.identity[1]),
                },
                "all_domains_pairwise_inode_disjoint": True,
            },
            "bound_evidence": computation.bindings.container_hashes(),
            "source_preflight_receipt_sha256": preflight.receipt_sha256,
            "source_phase_status": computation.public_result.get("status"),
            "external_bandwidth_status": computation.bandwidth_gate.get("status"),
            "access_journal": list(journal.events),
            "request_path_access_started_only_after_fresh_preflight": journal.events.index("FRESH_TYPED_SOURCE_FREE_GPU_PREFLIGHT_PASSED") < journal.events.index("POST_PREFLIGHT_OPEN:source-request"),
            "controls_opened": False,
            "final_performance_claim_authority": False,
            "requires_fresh_process_independent_result_audit": True,
        }
        source_publication = {
            "schema": "uwfa-sc-v8-external-source-phase-publication-v2",
            "status": launch_status,
            "producer_result": computation.public_result,
            "bound_evidence": computation.bindings.container_hashes(),
            "derived_baseline_score_sha256": sha256(computation.score_bytes),
            "bandwidth_gate_sha256": sha256(canonical_json(computation.bandwidth_gate)),
            "controls_opened": False,
            "matched_null_specificity_established": False,
            "external_result_audit_complete": False,
        }
        members: list[tuple[str, bytes]] = [
            ("LAUNCH_RECEIPT.json", canonical_json(launch)),
            ("RUNTIME_LOCK.authenticated.json", runtime.lock_bytes),
            ("SOURCE_PREFLIGHT.json", canonical_json(preflight.record)),
            ("BOUND_BASELINE_SCORE.json", computation.score_bytes),
            ("SOURCE_PHASE.json", canonical_json(source_publication)),
            ("BANDWIDTH_GATE.json", canonical_json(computation.bandwidth_gate)),
        ]
        if conditional:
            handoff = computation.public_result["source_final"]["posterior_diagnostic_handoff"]
            members.extend([
                ("UWFCV8.bin", bytes(container)),
                ("IDENTITY_FRAMING.bin", bytes(identity)),
                ("POSTERIOR_HANDOFF.json", canonical_json(handoff)),
            ])
        with common.CompletionLastOutput(retained_parent, request["final_name"], request["transaction_id"]) as transaction:
            for name, data in members:
                transaction.write_new(name, data)
            inputs.verify_stable()
            producer.verify_stable()
            dispatcher.verify_stable()
            runtime.verify_stable()
            decoder.verify_stable()
            held_parent.verify_same_directory_identity()
            decoder.remove_modules()
            modules.remove()
            provenance = runtime.provenance_ledger
            require(provenance is not None and provenance.active, "publication provenance ledger inactive")
            provenance.finalize()
            ledger_record = {
                "schema": "uwfa-sc-v8-import-native-event-ledger-v2",
                "status": "PASS_FINAL_DESCRIPTOR_NAME_MODULE_AND_NATIVE_EVENT_CLOSURE",
                "events": list(provenance.events),
                "event_count": len(provenance.events),
                "final_chain_sha256": provenance.chain_sha256,
            }
            ledger_bytes = canonical_json(ledger_record)
            transaction.write_new("IMPORT_NATIVE_EVENT_LEDGER.json", ledger_bytes)
            members.append(("IMPORT_NATIVE_EVENT_LEDGER.json", ledger_bytes))
            transaction.complete(list(transaction.members), authority)
        with envelope.verify_completed_under_parent(
            common,
            retained_parent,
            request["final_name"],
            expected_source_manifest_sha256=authority,
        ) as verified:
            for name, expected in members:
                require(verified.read_member_bytes(name) == expected, f"published member differs: {name}")
            metadata = dict(verified.metadata)
        held_parent.verify_same_directory_identity()
        return {
            "schema": "uwfa-sc-v8-external-publication-summary-v2",
            "status": launch_status,
            "final_name": request["final_name"],
            "parent_commit_sha256": metadata["parent_commit_sha256"],
            "directory_root_sha256": metadata["directory_root_sha256"],
            "source_authority_root_sha256": authority,
            "payload_authority_final": False,
        }
    finally:
        retained_parent.close()
        held_parent.close()


def dispatch_production(
    arguments: argparse.Namespace,
    external_authority: ExternalLaunchAuthority,
) -> dict[str, Any]:
    require_isolated_cpython()
    require(isinstance(external_authority, ExternalLaunchAuthority), "typed out-of-tree launch authority required")
    # Production pins are not a caller capability.  Unresolved checked-in pins
    # therefore stop the dispatcher before any authority/request pathname use.
    pins = ProductionPins.embedded()
    require(arguments.request_sha256 == external_authority.request_sha256, "exact request pin/argument mismatch")
    journal = AccessJournal()
    dispatcher: DispatcherClosure | None = None
    producer: ProducerClosure | None = None
    runtime: RuntimeClosure | None = None
    decoder: DecoderClosure | None = None
    modules: SnapshotModules | None = None
    provenance: AppendOnlyImportNativeLedger | None = None
    inputs: SourceInputs | None = None
    try:
        package_path = os.path.dirname(__file__)
        _absolute_components(package_path, label="dispatcher __file__ package")
        dispatcher = authenticate_dispatcher(
            package_path=package_path,
            audit_path=arguments.dispatcher_audit,
            manifest_pin=external_authority.dispatcher_manifest_sha256,
            audit_pin=external_authority.dispatcher_audit_sha256,
            expected_public_commit=external_authority.dispatcher_public_git_commit,
        )
        producer = authenticate_producer(
            package_path=arguments.producer_package,
            review_path=arguments.producer_review,
            manifest_pin=pins.producer_manifest_sha256,
            review_pin=pins.producer_review_sha256,
            expected_public_commit=pins.public_git_commit,
        )
        runtime = authenticate_runtime(
            dispatcher.members["runtime_lock.json"].data,
            expected_lock_sha256=pins.runtime_lock_sha256,
        )
        decoder = authenticate_decoder_bundle(
            dispatcher.members["decoder_bundle.json"].data,
            expected_bundle_sha256=pins.decoder_bundle_sha256,
            producer=producer,
            dispatcher=dispatcher,
        )
        all_held_by_path = dict(runtime.held_by_path)
        all_held_by_path.update({
            producer.package.path + "/" + name: HeldAbsoluteFile(producer.package, member)
            for name, member in producer.members.items()
        })
        all_held_by_path.update({
            dispatcher.package.path + "/" + name: HeldAbsoluteFile(dispatcher.package, member)
            for name, member in dispatcher.members.items()
        })
        all_held_by_path.update({
            member.directory.path + "/" + member.member.name: member
            for member in decoder.external_members.values()
        })
        bootstrap_module = sys.modules.get(__name__)
        require(isinstance(bootstrap_module, types.ModuleType), "dispatcher bootstrap module object unavailable")
        bootstrap_path = dispatcher.package.path + "/bootstrap.py"
        provenance = runtime.activate_import_native_enforcement(
            all_held_by_path=all_held_by_path,
            native_audit_event_fd=external_authority.native_audit_event_fd,
            native_audit_session_nonce=external_authority.native_audit_session_nonce,
            trusted_preloaded={
                __name__: (bootstrap_module, dispatcher.members["bootstrap.py"], bootstrap_path),
            },
        )
        modules = compile_producer_snapshots(producer, provenance)
        np, cp = runtime.enable_python_runtime()
        decoder.compile_external(provenance)
        journal.authority_passed()
        preflight = run_fresh_typed_source_free_preflight(
            modules=modules,
            runtime=runtime,
            producer=producer,
            cp=cp,
            journal=journal,
        )
        additional_runtime_paths = {
            dispatcher.package.path + "/" + name for name in dispatcher.members
        } | {
            member.directory.path + "/" + member.member.name
            for member in decoder.external_members.values()
        }
        runtime.seal_imported_image_closure(additional_authenticated_paths=additional_runtime_paths)
        inputs = open_source_inputs_after_preflight(
            request_path=arguments.request,
            request_sha256=arguments.request_sha256,
            public_commit_pin=pins.public_git_commit,
            external_authority=external_authority,
            journal=journal,
        )
        computation = run_source_computation(
            producer=producer,
            dispatcher=dispatcher,
            runtime=runtime,
            decoder=decoder,
            modules=modules,
            np=np,
            preflight=preflight,
            inputs=inputs,
            public_git_commit=pins.public_git_commit,
        )
        result = publish_source_result(
            producer=producer,
            dispatcher=dispatcher,
            runtime=runtime,
            decoder=decoder,
            modules=modules,
            preflight=preflight,
            inputs=inputs,
            computation=computation,
            public_git_commit=pins.public_git_commit,
            external_authority=external_authority,
            journal=journal,
        )
        require(provenance._finalized, "publication must finalize import/native provenance")
        result["import_native_event_ledger_sha256"] = provenance.chain_sha256
        result["import_native_event_count"] = len(provenance.events)
        return result
    finally:
        if inputs is not None:
            inputs.close()
        if decoder is not None:
            decoder.close()
        if runtime is not None:
            runtime.close()
        if modules is not None:
            modules.remove()
        if producer is not None:
            producer.close()
        if dispatcher is not None:
            dispatcher.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed UWFA-SC v8 external source dispatcher")
    parser.add_argument("--producer-package", required=True)
    parser.add_argument("--producer-review", required=True)
    parser.add_argument("--dispatcher-audit", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    return parser


def main() -> int:
    require_isolated_cpython()
    raise DispatchError(
        "BLOCK_DIRECT_PRODUCTION_LAUNCH_REQUIRES_OUT_OF_TREE_PINNED_AUTHORITY; "
        "authenticate this bootstrap as held bytes and call dispatch_production "
        "with typed ExternalLaunchAuthority"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCK_EXTERNAL_DISPATCHER: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
