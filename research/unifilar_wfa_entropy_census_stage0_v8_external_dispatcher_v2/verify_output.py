#!/usr/bin/env python3
"""Dependency-free structural verifier for a dispatcher publication.

This verifies the completion-last directory and parent marker without importing
producer code.  It does not recompute Qwen MSE, entropy, decode, or telemetry;
the fresh-process independent numeric result audit remains mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")
BASE_MEMBERS = {
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
CONDITIONAL_MEMBERS = {"UWFCV8.bin", "IDENTITY_FRAMING.bin", "POSTERIOR_HANDOFF.json"}


class OutputVerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OutputVerificationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def length_prefixed_root(domain: bytes, rows: list[bytes | str]) -> str:
    h = hashlib.sha256()
    h.update(len(domain).to_bytes(8, "little"))
    h.update(domain)
    for row in rows:
        raw = row.encode("utf-8") if isinstance(row, str) else row
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def strict_json(data: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise OutputVerificationError(f"{label} nonfinite {value}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), f"{label} object")
    return value


def safe(value: str, label: str) -> str:
    require(isinstance(value, str) and SAFE.fullmatch(value) is not None and value not in {".", ".."}, f"{label} safe name")
    return value


def marker_name(final_name: str) -> str:
    final_name = safe(final_name, "final name")
    digest = hashlib.sha256(b"UWFA-V8-COMMIT-NAME\x00" + final_name.encode()).hexdigest()
    return f".uwfa-publish-v8-{digest}.json"


def seal(record: dict[str, Any], field: str) -> str:
    claimed = record.get(field)
    require(isinstance(claimed, str) and HEX64.fullmatch(claimed) is not None, f"{field} digest")
    clean = dict(record)
    clean.pop(field)
    require(sha256(canonical(clean)) == claimed, f"{field} internal seal")
    return claimed


def directory_root(source_root: str, members: list[dict[str, Any]]) -> str:
    require(isinstance(source_root, str) and HEX64.fullmatch(source_root) is not None, "source root")
    rows = sorted((dict(row) for row in members), key=lambda row: row["name"].encode("utf-8"))
    body = canonical({"source_manifest_sha256": source_root, "members": rows})
    return sha256(b"UWFA-V8-HELD-DIRECTORY-ROOT\x00" + body)


def identity(row: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (row.st_dev, row.st_ino, row.st_mode, row.st_size, row.st_mtime_ns, row.st_ctime_ns, row.st_nlink)


def pread_exact(fd: int, size: int, cap: int, label: str) -> bytes:
    require(0 <= size <= cap, f"bounded member: {label}")
    chunks = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1 << 20, size - offset), offset)
        require(bool(chunk), f"short member: {label}")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


@dataclass
class HeldRead:
    parent_fd: int
    name: str
    fd: int
    before: tuple[int, int, int, int, int, int, int]
    data: bytes
    cap: int

    def verify_final(self) -> None:
        require(identity(os.fstat(self.fd)) == self.before, f"held descriptor changed: {self.name}")
        named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        require(identity(named) == self.before, f"held name/inode substituted: {self.name}")
        require(pread_exact(self.fd, self.before[3], self.cap, self.name) == self.data, f"held bytes changed: {self.name}")

    def close(self) -> None:
        os.close(self.fd)


def read_regular_at(directory_fd: int, name: str, cap: int) -> HeldRead:
    safe(name, "member")
    fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        before_info = os.fstat(fd)
        require(stat.S_ISREG(before_info.st_mode) and 0 <= before_info.st_size <= cap, f"regular bounded member: {name}")
        data = pread_exact(fd, before_info.st_size, cap, name)
        held = HeldRead(directory_fd, name, fd, identity(before_info), data, cap)
        held.verify_final()
        return held
    except Exception:
        os.close(fd)
        raise


def open_absolute_directory(path: str) -> tuple[list[int], os.stat_result]:
    require(os.name == "posix" and path.startswith("/") and not path.endswith("/") and "//" not in path, "absolute canonical POSIX parent")
    parts = path.split("/")[1:]
    require(parts and all(part and part not in {".", ".."} for part in parts), "parent components")
    fds = []
    current = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    fds.append(current)
    try:
        for part in parts:
            current = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
            require(stat.S_ISDIR(os.fstat(current).st_mode), "parent component directory")
            fds.append(current)
        return fds, os.fstat(fds[-1])
    except Exception:
        for fd in reversed(fds):
            os.close(fd)
        raise


def verify(output_parent: str, final_name: str, expected_authority_root: str) -> dict[str, Any]:
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "verify_output requires CPython -I -B")
    require(HEX64.fullmatch(expected_authority_root) is not None, "expected authority root")
    parent_fds, parent_info = open_absolute_directory(output_parent)
    parent_fd = parent_fds[-1]
    final_fd = -1
    held_reads: list[HeldRead] = []
    try:
        final_name = safe(final_name, "final name")
        marker_held = read_regular_at(parent_fd, marker_name(final_name), 16 << 20)
        held_reads.append(marker_held)
        require(marker_held.before[6] == 1, "parent marker sole link")
        marker = strict_json(marker_held.data, "parent marker")
        require(marker.get("schema") == "unifilar-wfa-parent-commit-v8" and marker.get("status") == "PARENT_MARKER_COMMITTED", "parent marker schema/status")
        seal(marker, "parent_commit_sha256")
        require(marker.get("final_name") == final_name, "parent marker final name")
        require(marker.get("output_parent_authority_sha256") == expected_authority_root, "parent marker authority")
        require(marker.get("source_manifest_sha256") == expected_authority_root, "parent marker source root")
        require((marker.get("parent_device"), marker.get("parent_inode")) == (parent_info.st_dev, parent_info.st_ino), "parent marker parent identity")
        require((marker.get("commit_marker_device"), marker.get("commit_marker_inode")) == marker_held.before[:2], "parent marker inode binding")
        final_fd = os.open(final_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        final_info = os.fstat(final_fd)
        require((marker.get("final_directory_device"), marker.get("final_directory_inode")) == (final_info.st_dev, final_info.st_ino), "parent marker final-directory binding")
        actual = {entry.name for entry in os.scandir(final_fd)}
        has_container = "UWFCV8.bin" in actual
        expected = BASE_MEMBERS | (CONDITIONAL_MEMBERS if has_container else set())
        require(actual == expected, f"publication member set: {sorted(actual ^ expected)}")
        rows = marker.get("members")
        require(isinstance(rows, list) and [row.get("name") for row in rows] == sorted(expected, key=lambda name: name.encode("utf-8")), "parent marker canonical members")
        observed: dict[str, bytes] = {}
        observed_rows = []
        output_inodes: dict[tuple[int, int], str] = {marker_held.before[:2]: "parent marker"}
        for row in rows:
            require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "parent marker member row")
            held = read_regular_at(final_fd, row["name"], 1 << 34)
            held_reads.append(held)
            data = held.data
            require(row["bytes"] == len(data) and row["sha256"] == sha256(data), f"parent marker member binding: {row['name']}")
            key = held.before[:2]
            require(key not in output_inodes, f"output member inode alias: {row['name']} / {output_inodes.get(key)}")
            output_inodes[key] = row["name"]
            observed[row["name"]] = data
            observed_rows.append(dict(row))
        require(marker.get("directory_root_sha256") == directory_root(expected_authority_root, observed_rows), "directory root")
        complete = strict_json(observed["COMPLETE.json"], "completion")
        require(complete.get("schema") == "unifilar-wfa-completion-v8" and complete.get("status") == "COMPLETE_LAST", "completion schema/status")
        completion_sha = seal(complete, "completion_sha256")
        require(marker.get("completion_sha256") == completion_sha, "marker/completion seal")
        require(complete.get("source_manifest_sha256") == expected_authority_root, "completion source root")
        complete_rows = complete.get("members")
        expected_insertion_order = [
            "RUN_STATE.json",
            "LAUNCH_RECEIPT.json",
            "RUNTIME_LOCK.authenticated.json",
            "SOURCE_PREFLIGHT.json",
            "BOUND_BASELINE_SCORE.json",
            "SOURCE_PHASE.json",
            "BANDWIDTH_GATE.json",
        ] + (["UWFCV8.bin", "IDENTITY_FRAMING.bin", "POSTERIOR_HANDOFF.json"] if has_container else []) + ["IMPORT_NATIVE_EVENT_LEDGER.json"]
        require(isinstance(complete_rows, list) and [row.get("name") for row in complete_rows] == expected_insertion_order, "completion insertion-order members")
        marker_without_complete = {row["name"]: row for row in rows if row["name"] != "COMPLETE.json"}
        require({row["name"]: row for row in complete_rows} == marker_without_complete, "completion/marker member equality")
        for name in ("LAUNCH_RECEIPT.json", "SOURCE_PREFLIGHT.json", "BOUND_BASELINE_SCORE.json", "SOURCE_PHASE.json", "BANDWIDTH_GATE.json", "IMPORT_NATIVE_EVENT_LEDGER.json"):
            record = strict_json(observed[name], name)
            require(observed[name] == canonical(record), f"noncanonical generated JSON: {name}")
        launch = strict_json(observed["LAUNCH_RECEIPT.json"], "launch")
        source = strict_json(observed["SOURCE_PHASE.json"], "source phase")
        bandwidth = strict_json(observed["BANDWIDTH_GATE.json"], "bandwidth")
        event_ledger = strict_json(observed["IMPORT_NATIVE_EVENT_LEDGER.json"], "import/native event ledger")
        require(launch.get("schema") == "uwfa-sc-v8-external-launch-receipt-v2", "launch schema")
        require(source.get("schema") == "uwfa-sc-v8-external-source-phase-publication-v2", "source publication schema")
        require(bandwidth.get("schema") == "uwfa-sc-v8-external-repeated-coalesced-bandwidth-gate-v1", "bandwidth schema")
        require(
            set(event_ledger) == {"schema", "status", "events", "event_count", "final_chain_sha256"}
            and event_ledger["schema"] == "uwfa-sc-v8-import-native-event-ledger-v2"
            and event_ledger["status"] == "PASS_FINAL_DESCRIPTOR_NAME_MODULE_AND_NATIVE_EVENT_CLOSURE",
            "import/native event ledger schema/status",
        )
        events = event_ledger["events"]
        require(isinstance(events, list) and events and event_ledger["event_count"] == len(events), "import/native event count")
        chain = sha256(b"UWFA-SC-V8-IMPORT-NATIVE-EVENT-LEDGER-V2\0")
        for index, event in enumerate(events):
            require(isinstance(event, dict) and event.get("sequence") == index, "import/native event sequence")
            require(event.get("prior_chain_sha256") == chain, "import/native prior chain")
            claimed_chain = event.get("chain_sha256")
            require(isinstance(claimed_chain, str) and HEX64.fullmatch(claimed_chain) is not None, "import/native event chain digest")
            clean = {key: value for key, value in event.items() if key != "chain_sha256"}
            chain = length_prefixed_root(b"UWFA-SC-V8-IMPORT-NATIVE-EVENT-LEDGER-V2\0", [chain, canonical(clean)])
            require(claimed_chain == chain, "import/native event hash chain")
        require(events[-1].get("action") == "IMPORT_NATIVE_EVENT_CLOSURE_FINAL", "import/native final closure event")
        require(event_ledger["final_chain_sha256"] == chain, "import/native final chain")
        require(launch.get("controls_opened") is False and launch.get("final_performance_claim_authority") is False, "launch claim boundary")
        require(source.get("controls_opened") is False and source.get("external_result_audit_complete") is False, "source claim boundary")
        require(source.get("bandwidth_gate_sha256") == sha256(observed["BANDWIDTH_GATE.json"]), "source/bandwidth binding")
        domains = launch.get("descriptor_inode_domains")
        require(isinstance(domains, dict) and set(domains) == {"authority", "request", "output_parent", "all_domains_pairwise_inode_disjoint"}, "descriptor inode domains")
        require(domains["all_domains_pairwise_inode_disjoint"] is True, "descriptor inode disjoint declaration")
        claimed: dict[tuple[int, int], str] = {}
        for domain in ("authority", "request"):
            domain_rows = domains[domain]
            require(isinstance(domain_rows, list) and domain_rows, f"{domain} inode rows")
            for row in domain_rows:
                require(isinstance(row, dict) and set(row) == {"label", "device", "inode"}, f"{domain} inode row")
                require(isinstance(row["label"], str) and row["label"] and type(row["device"]) is int and type(row["inode"]) is int, f"{domain} inode value")
                key = (row["device"], row["inode"])
                require(key not in claimed, f"authority/request inode alias: {row['label']} / {claimed.get(key)}")
                claimed[key] = f"{domain}:{row['label']}"
        claimed_parent = domains["output_parent"]
        require(isinstance(claimed_parent, dict) and set(claimed_parent) == {"device", "inode"}, "claimed output parent inode")
        parent_key = (parent_info.st_dev, parent_info.st_ino)
        require((claimed_parent["device"], claimed_parent["inode"]) == parent_key, "launch/current output parent inode binding")
        require(parent_key not in claimed, "authority/request/output-parent inode alias")
        require((final_info.st_dev, final_info.st_ino) not in claimed, "authority/request/final-directory inode alias")
        require(not (set(output_inodes) & set(claimed)), "authority/request/output-member inode alias")
        # Final no-follow rebind: every held byte descriptor, both directory
        # names, and the parent inode are checked only after all parsing.
        for held in held_reads:
            held.verify_final()
        require({entry.name for entry in os.scandir(final_fd)} == expected, "final publication member set changed during verification")
        require(identity(os.fstat(parent_fd)) == identity(parent_info), "output parent changed during verification")
        require(identity(os.fstat(final_fd)) == identity(final_info), "final directory descriptor changed during verification")
        final_named = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        require(identity(final_named) == identity(final_info), "final directory name/inode substituted")
        return {
            "schema": "uwfa-sc-v8-external-publication-structural-verification-v2",
            "status": "PASS_STRUCTURE_ONLY_AWAITING_INDEPENDENT_NUMERIC_RESULT_AUDIT",
            "final_name": final_name,
            "parent_commit_sha256": marker["parent_commit_sha256"],
            "directory_root_sha256": marker["directory_root_sha256"],
            "source_authority_root_sha256": expected_authority_root,
            "literal_container_present": has_container,
            "members": observed_rows,
            "numeric_result_claim_authority": False,
        }
    finally:
        for held in reversed(held_reads):
            held.close()
        if final_fd >= 0:
            os.close(final_fd)
        for fd in reversed(parent_fds):
            os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--final-name", required=True)
    parser.add_argument("--expected-authority-root", required=True)
    args = parser.parse_args()
    result = verify(args.output_parent, args.final_name, args.expected_authority_root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_EXTERNAL_PUBLICATION_VERIFICATION: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
