#!/usr/bin/env python3
"""Independent source-only hostile audit for the stopped UWFA-SC v6 tree.

This script hard-codes the reviewed 17-member inventory.  It authenticates
every byte before compiling the two producer modules needed by the tests.  It
does not know a model path and has no payload entry point.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
import tempfile
import types
from pathlib import Path
from typing import Any


DOMAIN = b"UWFA-SC-V6-INDEPENDENT-INVENTORY-v1\0"
EXPECTED_ROOT = "b14bf19aa8965f0ab22ec26db43cddd63e0c5f3c4d996edeed45e512e516cca2"
EXPECTED = {
    "INDEPENDENT_BOOTSTRAP_ABI.md": (10301, "309efb9bd678ea14a35826090406196810d40a8920000ed686687518272c0004"),
    "README.md": (14453, "8649be924b9cfa0bc3908e383477ca7b3e9a06d85242dd5830aee798e6d05fb7"),
    "container_codec.py": (86661, "eb60851eac72d599c9e1054c948eb1b5a3fda44988637e4f18999d7751276db3"),
    "cupy_backend.py": (40964, "7904a5e122686487d89fb684b70052507089bfe3bbfe4f1f02520df6ce3fb1ba"),
    "design_lock.json": (10922, "14ff71d39193af4a2fdbe58be34b237951de3f27da4bd647d87d5b179a6ff435"),
    "dispatcher_contract.py": (9205, "bfec9a32752f351856fc9f07b0e4657b6365980cf13f11b0ab991cd4b38db4e4"),
    "fixture_long_memory.py": (4307, "5fb59317c0e8d9f468ee215fba0cd248dfc12523f3e79bd02a533094cd01d4ce"),
    "fixture_portability.py": (16350, "961246b2f281cf7cc5269804a1fe1501f45ee5554c07bea649bd8dd33868108e"),
    "protocol.py": (20312, "0362294df49f6a497e56cd580f9061fd516b07b93f049c078a2a119521b85080"),
    "result_envelope.py": (12565, "b2479c2fdbaff768883df2be0207d21f995fa99b2bdad39bdb4a9d18bbc2461f"),
    "run_source_free_gpu_dev.py": (7940, "e91431c03af7d8ea03c7aedc5a640b489b12b7fce725da03c095b3c7c1ea4492"),
    "stage0_census.py": (107299, "dafed50a096a1ceae1143f032b161375437a09ed6a85bb82dd704f790c47ba4b"),
    "strata_sc_adapter.py": (36184, "3d3875bf485dc5a60709735621aafd5e61e1c8f24a13ea5aab05e0c40bdd76e7"),
    "test_source_only.py": (113369, "995b5a4447238196805de5f3c6c6828f70312ecb0a75844671be4d796581c716"),
    "universal_adapter.py": (11577, "308879f4e58bec86f2aaf47ac1e7fa5c36da634f145ed5928a4ee6851c1ed23f"),
    "uwfa_common.py": (55807, "e1b8d18179f7b34a4afb66e7d21301fa0d709d22450ae5c4259164eaecd32ce4"),
    "verify_source.py": (14330, "521bd6ae271001dd7a6ed08f532699492c40dd18d44167484dfe2785aac84455"),
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise RuntimeError(f"nonfinite JSON constant: {value}")

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(result, dict), "JSON root")
    return result


def verify_seal(record: dict[str, Any], field: str) -> None:
    require(isinstance(record.get(field), str) and len(record[field]) == 64, f"seal field {field}")
    clean = dict(record)
    observed = clean.pop(field)
    require(observed == sha(canonical(clean)), f"seal mismatch {field}")


def read_regular_at(directory_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"nonregular member {name}")
        chunks = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            require(bool(chunk), f"short member {name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(fd, 1) == b"", f"growing member {name}")
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        require(identity(before) == identity(after), f"changing member {name}")
        return b"".join(chunks), before
    finally:
        os.close(fd)


def authenticate(package: Path) -> tuple[dict[str, bytes], list[dict[str, Any]], str]:
    package = package.resolve(strict=True)
    require(package.is_absolute(), "absolute package")
    cursor = Path(package.anchor)
    for part in package.parts[1:]:
        cursor /= part
        info = os.lstat(cursor)
        require(not stat.S_ISLNK(info.st_mode), f"symlink package ancestor {cursor}")
    directory_fd = os.open(
        package, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        actual = {entry.name for entry in os.scandir(directory_fd)}
        require(actual == set(EXPECTED), f"inventory mismatch {sorted(actual ^ set(EXPECTED))}")
        blobs: dict[str, bytes] = {}
        rows = []
        root_preimage = bytearray(DOMAIN)
        for name in sorted(EXPECTED, key=lambda item: item.encode("utf-8")):
            data, info = read_regular_at(directory_fd, name)
            expected_bytes, expected_sha = EXPECTED[name]
            require(len(data) == expected_bytes == int(info.st_size), f"byte mismatch {name}")
            require(sha(data) == expected_sha, f"digest mismatch {name}")
            blobs[name] = data
            rows.append({"name": name, "bytes": len(data), "sha256": expected_sha})
            root_preimage.extend(name.encode("utf-8") + b"\0")
            root_preimage.extend(str(len(data)).encode("ascii") + b"\0")
            root_preimage.extend(expected_sha.encode("ascii") + b"\n")
        root = sha(bytes(root_preimage))
        require(root == EXPECTED_ROOT, "inventory root mismatch")
        return blobs, rows, root
    finally:
        os.close(directory_fd)


def load_snapshot(name: str, source: bytes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<independent-authenticated:{name}>"
    module.__package__ = ""
    import sys
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def marker_name(final_name: str) -> str:
    digest = sha(b"UWFA-V6-COMMIT-NAME\0" + final_name.encode("utf-8"))
    return f".uwfa-publish-v6-{digest}.json"


def independent_verify(parent_path: Path, final_name: str, expected_source: str) -> dict[str, Any]:
    parent_fd = os.open(
        parent_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    marker_fd = -1
    final_fd = -1
    try:
        parent_info = os.fstat(parent_fd)
        marker_bytes, marker_info = read_regular_at(parent_fd, marker_name(final_name))
        marker = strict_json(marker_bytes)
        verify_seal(marker, "parent_commit_sha256")
        require(marker["schema"] == "unifilar-wfa-parent-commit-v6", "marker schema")
        require(marker["status"] == "PARENT_MARKER_COMMITTED", "marker status")
        require(marker["final_name"] == final_name, "marker final")
        require(marker["source_manifest_sha256"] == expected_source, "marker source")
        require((marker["parent_device"], marker["parent_inode"]) == (parent_info.st_dev, parent_info.st_ino), "marker parent inode")
        require((marker["commit_marker_device"], marker["commit_marker_inode"]) == (marker_info.st_dev, marker_info.st_ino), "marker inode")
        members = marker["members"]
        require(isinstance(members, list) and members, "marker members")
        require(members == sorted(members, key=lambda row: row["name"].encode("utf-8")), "member order")
        names = [row["name"] for row in members]
        require(len(names) == len(set(names)) and "COMPLETE.json" in names, "member names")
        final_fd = os.open(
            final_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        final_info = os.fstat(final_fd)
        require((marker["final_directory_device"], marker["final_directory_inode"]) == (final_info.st_dev, final_info.st_ino), "final-directory inode mismatch")
        actual = sorted(entry.name for entry in os.scandir(final_fd))
        require(actual == sorted(names), "directory membership mismatch")
        complete_bytes = b""
        for row in members:
            require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "member row")
            data, _info = read_regular_at(final_fd, row["name"])
            require(len(data) == row["bytes"] and sha(data) == row["sha256"], f"member mismatch {row['name']}")
            if row["name"] == "COMPLETE.json":
                complete_bytes = data
        expected_root = sha(b"UWFA-V6-HELD-DIRECTORY-ROOT\0" + canonical({
            "source_manifest_sha256": expected_source,
            "members": members,
        }))
        require(marker["directory_root_sha256"] == expected_root, "directory root mismatch")
        complete = strict_json(complete_bytes)
        verify_seal(complete, "completion_sha256")
        require(complete["schema"] == "unifilar-wfa-completion-v6" and complete["status"] == "COMPLETE_LAST", "complete schema")
        require(complete["source_manifest_sha256"] == expected_source, "complete source")
        require(complete["completion_sha256"] == marker["completion_sha256"], "complete seal binding")
        require(sorted(complete["members"], key=lambda row: row["name"].encode("utf-8")) == [row for row in members if row["name"] != "COMPLETE.json"], "complete members")
        end_marker = os.stat(marker_name(final_name), dir_fd=parent_fd, follow_symlinks=False)
        end_final = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        require((end_marker.st_dev, end_marker.st_ino) == (marker_info.st_dev, marker_info.st_ino), "marker name substitution")
        require((end_final.st_dev, end_final.st_ino) == (final_info.st_dev, final_info.st_ino), "final name substitution")
        return {
            "parent_commit_sha256": marker["parent_commit_sha256"],
            "directory_root_sha256": expected_root,
            "marker_sha256": sha(marker_bytes),
            "final_directory_device": final_info.st_dev,
            "final_directory_inode": final_info.st_ino,
            "marker_device": marker_info.st_dev,
            "marker_inode": marker_info.st_ino,
        }
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        if marker_fd >= 0:
            os.close(marker_fd)
        os.close(parent_fd)


def expect_failure(call: Any, contains: str | None = None) -> str:
    try:
        call()
    except BaseException as exc:
        text = f"{type(exc).__name__}: {exc}"
        if contains is not None:
            require(contains in text, f"wrong failure: {text}")
        return text
    raise RuntimeError("expected failure did not occur")


def publication_tests(common: Any, envelope: Any) -> dict[str, Any]:
    require(os.name == "posix", "POSIX publication audit required")
    source = "ab" * 32
    results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as raw:
        parent_path = Path(raw)
        parent = common.RetainedOutputParent.open_path_source_only(parent_path, "91" * 32)
        try:
            with common.CompletionLastOutput(parent, "success", "31" * 16) as tx:
                held = os.fstat(tx.dir_fd)
                tx.write_new("RESULT.bin", b"independent-success")
                tx.complete(list(tx.members), source)
            named = os.stat(parent_path / "success", follow_symlinks=False)
            require((held.st_dev, held.st_ino) == (named.st_dev, named.st_ino), "retained staging/final inode")
            producer = envelope.verify_completed_under_parent(common, parent, "success", expected_source_manifest_sha256=source)
            independent = independent_verify(parent_path, "success", source)
            require(producer["directory_root_sha256"] == independent["directory_root_sha256"], "producer/independent root")
            results["success"] = independent
        finally:
            parent.close()

    with tempfile.TemporaryDirectory() as raw:
        parent_path = Path(raw)
        parent = common.RetainedOutputParent.open_path_source_only(parent_path, "92" * 32)
        real = common._rename_directory_noreplace
        final = "before-move"
        tid = "32" * 16
        def substitute(parent_fd: int, source_name: str, destination: str) -> None:
            os.rename(source_name, "held-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.mkdir(source_name, 0o700, dir_fd=parent_fd)
            real(parent_fd, source_name, destination)
        common._rename_directory_noreplace = substitute
        try:
            failure = expect_failure(lambda: _publish(common, parent, final, tid, source), "retained staging directory")
            require(not (parent_path / marker_name(final)).exists(), "before-move marker")
            results["before_named_move"] = failure
        finally:
            common._rename_directory_noreplace = real
            parent.close()

    with tempfile.TemporaryDirectory() as raw:
        parent_path = Path(raw)
        parent = common.RetainedOutputParent.open_path_source_only(parent_path, "93" * 32)
        real = common._open_held_commit_authority_file
        final = "before-marker"
        def substitute(parent_fd: int, anchor: str) -> tuple[int, str | None]:
            os.rename(final, "held-final-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.mkdir(final, 0o700, dir_fd=parent_fd)
            return real(parent_fd, anchor)
        common._open_held_commit_authority_file = substitute
        try:
            failure = expect_failure(lambda: _publish(common, parent, final, "33" * 16, source), "retained staging directory")
            require(not (parent_path / marker_name(final)).exists(), "before-marker marker")
            results["after_move_before_marker"] = failure
        finally:
            common._open_held_commit_authority_file = real
            parent.close()

    with tempfile.TemporaryDirectory() as raw:
        parent_path = Path(raw)
        parent = common.RetainedOutputParent.open_path_source_only(parent_path, "94" * 32)
        real = common._link_held_unnamed_file_noreplace
        final = "after-marker"
        def substitute(source_fd: int, parent_fd: int, destination: str) -> None:
            real(source_fd, parent_fd, destination)
            os.rename(final, "committed-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.mkdir(final, 0o700, dir_fd=parent_fd)
        common._link_held_unnamed_file_noreplace = substitute
        try:
            failure = expect_failure(lambda: _publish(common, parent, final, "34" * 16, source), "retained staging directory")
            require((parent_path / marker_name(final)).is_file(), "after-marker marker absent")
            consumer = expect_failure(lambda: independent_verify(parent_path, final, source), "final-directory inode mismatch")
            results["after_marker_link"] = {"producer": failure, "consumer": consumer}
        finally:
            common._link_held_unnamed_file_noreplace = real
            parent.close()

    with tempfile.TemporaryDirectory() as raw:
        parent_path = Path(raw)
        markerless = parent_path / "markerless"
        markerless.mkdir()
        (markerless / "COMPLETE.json").write_bytes(b"{}")
        results["complete_without_marker"] = expect_failure(
            lambda: independent_verify(parent_path, "markerless", source)
        )

    with tempfile.TemporaryDirectory() as raw:
        parent_path = Path(raw)
        parent = common.RetainedOutputParent.open_path_source_only(parent_path, "95" * 32)
        try:
            _publish(common, parent, "mutate", "35" * 16, source)
        finally:
            parent.close()
        (parent_path / "mutate" / "RESULT.bin").write_bytes(b"changed-after-marker")
        results["root_member_mismatch"] = expect_failure(
            lambda: independent_verify(parent_path, "mutate", source), "member mismatch"
        )

    with tempfile.TemporaryDirectory() as raw:
        parent_path = Path(raw)
        parent = common.RetainedOutputParent.open_path_source_only(parent_path, "96" * 32)
        real = common._link_held_unnamed_file_noreplace
        final = "marker-substitute"
        def substitute_marker(source_fd: int, parent_fd: int, destination: str) -> None:
            real(source_fd, parent_fd, destination)
            os.rename(destination, "real-marker-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
            os.write(fd, b"attacker")
            os.close(fd)
        common._link_held_unnamed_file_noreplace = substitute_marker
        try:
            failure = expect_failure(lambda: _publish(common, parent, final, "36" * 16, source), "held unnamed inode")
            consumer = expect_failure(lambda: independent_verify(parent_path, final, source))
            results["marker_name_substitution"] = {"producer": failure, "consumer": consumer}
        finally:
            common._link_held_unnamed_file_noreplace = real
            parent.close()

    results["proc_self_fd"] = proc_fd_identity_test(common)
    return results


def _publish(common: Any, parent: Any, final: str, transaction: str, source: str) -> None:
    with common.CompletionLastOutput(parent, final, transaction) as tx:
        tx.write_new("RESULT.bin", b"independent-hostile")
        tx.complete(list(tx.members), source)


def proc_fd_identity_test(common: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as raw:
        parent = Path(raw)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        source_fd = -1
        try:
            source_fd, anchor = common._open_held_commit_authority_file(parent_fd, "anchor")
            os.write(source_fd, b"held-marker")
            os.fsync(source_fd)
            held = os.fstat(source_fd)
            proc_path = f"/proc/self/fd/{source_fd}"
            require(os.path.exists(proc_path), "proc fd path unavailable")
            if anchor is not None:
                os.rename(anchor, "held-anchor-aside", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                attacker = os.open(anchor, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
                os.write(attacker, b"attacker-anchor")
                os.close(attacker)
            libc = ctypes.CDLL(None, use_errno=True)
            linkat = libc.linkat
            linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            linkat.restype = ctypes.c_int
            result = linkat(-100, os.fsencode(proc_path), parent_fd, b"linked-marker", 0x400)
            require(result == 0, f"proc-fd linkat failed errno={ctypes.get_errno()}")
            linked = os.stat("linked-marker", dir_fd=parent_fd, follow_symlinks=False)
            require((linked.st_dev, linked.st_ino) == (held.st_dev, held.st_ino), "proc fd linked wrong inode")
            if anchor is not None:
                attacker_info = os.stat(anchor, dir_fd=parent_fd, follow_symlinks=False)
                require((linked.st_dev, linked.st_ino) != (attacker_info.st_dev, attacker_info.st_ino), "proc fd followed mutable anchor")
            return {
                "creation_mode": "O_TMPFILE" if anchor is None else "named-held-anchor",
                "proc_self_fd_available": True,
                "held_device": held.st_dev,
                "held_inode": held.st_ino,
                "linked_device": linked.st_dev,
                "linked_inode": linked.st_ino,
                "mutable_anchor_not_authority": anchor is not None,
            }
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--gpu-output-parent")
    parser.add_argument("--gpu-final-name", default="gpu-receipt")
    args = parser.parse_args()
    blobs, inventory, root = authenticate(Path(args.package))
    common = load_snapshot("uwfa_v6_independent_common", blobs["uwfa_common.py"])
    envelope = load_snapshot("uwfa_v6_independent_envelope", blobs["result_envelope.py"])
    result: dict[str, Any] = {
        "schema": "uwfa-sc-v6-independent-hostile-audit-v1",
        "status": "PASS_INDEPENDENT_SOURCE_ONLY_HOSTILE_AUDIT",
        "inventory": inventory,
        "inventory_root_sha256": root,
        "publication": publication_tests(common, envelope),
        "gpu_output": None,
        "payload_opened": False,
        "qwen_opened": False,
    }
    if args.gpu_output_parent:
        development_root = sha(canonical(inventory))
        result["gpu_output"] = independent_verify(
            Path(args.gpu_output_parent), args.gpu_final_name, development_root
        )
        result["gpu_output"]["development_source_root_sha256"] = development_root
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
