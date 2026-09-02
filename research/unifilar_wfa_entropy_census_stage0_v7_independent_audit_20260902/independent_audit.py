#!/usr/bin/env python3
"""Independent source/receipt authenticator for stopped UWFA-SC v7.

This file imports no producer module and opens no model/Qwen payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


INVENTORY_DOMAIN = b"UWFA-SC-V7-INDEPENDENT-INVENTORY-v1\0"
EXPECTED_ROOT = "2cd5e4cc7f53ec0e9ab91f8a9f9a505b8e82e316ab7fe0c1e69e8dcac28bc97f"
EXPECTED: dict[str, tuple[int, str]] = {
    "INDEPENDENT_BOOTSTRAP_ABI.md": (10724, "2d6c15fa5be3773e3da8d44a7cee452c18918fe192a7b886c22812889ddb2763"),
    "README.md": (15700, "bbe15a3e30d2a7c455c33f1d0d2188e35ce255058335fe35eaf76773d0f70062"),
    "container_codec.py": (93072, "ed620383cbc518b6d7c969697649a4d7cd547e644a50f7cb216ed4fbf55c02c7"),
    "cupy_backend.py": (40964, "7904a5e122686487d89fb684b70052507089bfe3bbfe4f1f02520df6ce3fb1ba"),
    "design_lock.json": (11301, "c87d1580650c5c7af20db9afd0383985ec2e92e9c56e385a9a9096121cbb4148"),
    "dispatcher_contract.py": (9205, "7cb7fbcc72502ffbe6d30173b23a44959a43dc9cc4a6628d7a3ffa9b4a40ba10"),
    "fixture_long_memory.py": (4307, "1d7d12a00bdde113157c34b30db66970b721a600614017927edf9ceff730ef52"),
    "fixture_portability.py": (16350, "acf33657ec956249810d8c3a2ef8827314439662e5706514810d4c27a936f70b"),
    "protocol.py": (21051, "98678af78170445fcfbef7e13b47fc0be5ef9c5f27ce051a4f8b57fee582e22c"),
    "result_envelope.py": (19002, "21aeefe04b0601aff1e7c4bd70356545c3407de4d39cba08ef28e83bb7602dee"),
    "run_source_free_gpu_dev.py": (8263, "87b6f1c3eaee8a526dcd188cab80dab486e444ff96323c77c59c984de020662c"),
    "stage0_census.py": (122429, "182b2baee33969ae7aab39ecf2becb68115847b08599919073ac6901c268590e"),
    "strata_sc_adapter.py": (36184, "d2079c60d34518bd46d4c2b1f698aa8be6c20eafd50e27fe238b3038cdf7e045"),
    "test_source_only.py": (131057, "1418591b2e7230659ba46ae95d06a5df7f66c4d1ff9499c391eefd958caaac29"),
    "universal_adapter.py": (11577, "2241f7bfcd02851383c826a27c206ee8e0a6ad85d83f3c8c5ac65252b7a08b42"),
    "uwfa_common.py": (58875, "956f47ea49a137d027c1f154fe1867659137cd6081edda56e0b5e3a6848f42b1"),
    "verify_source.py": (15539, "5be25ae5e02498d4d5706efc2b1c35d624488389f64c169023a6176533021201"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(data: bytes) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs)


def verify_seal(row: dict[str, Any], field: str) -> None:
    claimed = row.get(field)
    require(isinstance(claimed, str) and re.fullmatch(r"[0-9a-f]{64}", claimed) is not None, f"{field} syntax")
    clean = dict(row)
    clean.pop(field)
    require(sha(canonical(clean)) == claimed, f"{field} mismatch")


def identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode), int(info.st_size),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def read_regular_at(directory_fd: int, name: str, maximum: int = 1 << 30) -> tuple[bytes, os.stat_result]:
    require("/" not in name and name not in {"", ".", ".."}, "unsafe member name")
    fd = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"nonregular member {name}")
        require(0 <= before.st_size <= maximum, f"oversized member {name}")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            require(bool(chunk), f"short member {name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(fd, 1) == b"", f"growing member {name}")
        after = os.fstat(fd)
        require(identity(before) == identity(after), f"changing member {name}")
        return b"".join(chunks), before
    finally:
        os.close(fd)


def authenticate_source(package: Path) -> tuple[list[dict[str, Any]], str]:
    package = package.resolve(strict=True)
    cursor = Path(package.anchor)
    for part in package.parts[1:]:
        cursor /= part
        require(not stat.S_ISLNK(os.lstat(cursor).st_mode), f"symlink ancestor {cursor}")
    directory_fd = os.open(
        package,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        actual = {entry.name for entry in os.scandir(directory_fd)}
        require(actual == set(EXPECTED), f"source inventory mismatch {sorted(actual ^ set(EXPECTED))}")
        preimage = bytearray(INVENTORY_DOMAIN)
        rows: list[dict[str, Any]] = []
        for name in sorted(EXPECTED, key=lambda value: value.encode("utf-8")):
            data, info = read_regular_at(directory_fd, name)
            size, expected_sha = EXPECTED[name]
            require(len(data) == size == info.st_size, f"source size {name}")
            require(sha(data) == expected_sha, f"source digest {name}")
            rows.append({"name": name, "bytes": size, "sha256": expected_sha})
            preimage.extend(name.encode("utf-8") + b"\0")
            preimage.extend(str(size).encode("ascii") + b"\0")
            preimage.extend(expected_sha.encode("ascii") + b"\n")
        root = sha(bytes(preimage))
        require(root == EXPECTED_ROOT, "independent inventory root")
        return rows, root
    finally:
        os.close(directory_fd)


def marker_name(final_name: str) -> str:
    digest = sha(b"UWFA-V7-COMMIT-NAME\0" + final_name.encode("utf-8"))
    return f".uwfa-publish-v7-{digest}.json"


def verify_gpu_receipt(parent_path: Path, final_name: str, expected_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    parent_path = parent_path.resolve(strict=True)
    parent_fd = os.open(
        parent_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    directory_fd = -1
    try:
        parent_info = os.fstat(parent_fd)
        marker_bytes, marker_info = read_regular_at(parent_fd, marker_name(final_name), 1 << 20)
        marker = strict_json(marker_bytes)
        require(isinstance(marker, dict), "marker object")
        verify_seal(marker, "parent_commit_sha256")
        require(marker.get("schema") == "unifilar-wfa-parent-commit-v7", "marker schema")
        require(marker.get("status") == "PARENT_MARKER_COMMITTED", "marker status")
        require(marker.get("final_name") == final_name, "marker final name")
        require((marker["parent_device"], marker["parent_inode"]) == (parent_info.st_dev, parent_info.st_ino), "marker parent inode")
        require((marker["commit_marker_device"], marker["commit_marker_inode"]) == (marker_info.st_dev, marker_info.st_ino), "marker inode")
        require(marker_info.st_nlink == 1, "marker is not sole hardlink")
        members = marker.get("members")
        require(isinstance(members, list) and members, "marker members")
        require(members == sorted(members, key=lambda row: row["name"].encode("utf-8")), "member order")
        require(len({row["name"] for row in members}) == len(members), "duplicate member")

        directory_fd = os.open(
            final_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        directory_info = os.fstat(directory_fd)
        require((marker["final_directory_device"], marker["final_directory_inode"]) == (directory_info.st_dev, directory_info.st_ino), "final inode")
        require(sorted(entry.name for entry in os.scandir(directory_fd)) == sorted(row["name"] for row in members), "exact result membership")
        blobs: dict[str, bytes] = {}
        for row in members:
            require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "member row")
            data, info = read_regular_at(directory_fd, row["name"])
            require(len(data) == row["bytes"] == info.st_size, f"member size {row['name']}")
            require(sha(data) == row["sha256"], f"member digest {row['name']}")
            blobs[row["name"]] = data
        source_root = marker["source_manifest_sha256"]
        directory_root = sha(b"UWFA-V7-HELD-DIRECTORY-ROOT\0" + canonical({
            "source_manifest_sha256": source_root,
            "members": members,
        }))
        require(directory_root == marker["directory_root_sha256"], "directory root")
        complete = strict_json(blobs["COMPLETE.json"])
        verify_seal(complete, "completion_sha256")
        require(complete.get("schema") == "unifilar-wfa-completion-v7" and complete.get("status") == "COMPLETE_LAST", "complete schema/status")
        require(complete["completion_sha256"] == marker["completion_sha256"], "complete binding")
        require(complete["source_manifest_sha256"] == source_root, "complete source")
        require(sorted(complete["members"], key=lambda row: row["name"].encode("utf-8")) == [row for row in members if row["name"] != "COMPLETE.json"], "complete members")

        receipt_bytes = blobs["GPU_DEV_RECEIPT.json"]
        receipt = strict_json(receipt_bytes)
        require(receipt.get("status") == "PASS_SOURCE_FREE_DEVELOPMENT_REPLAY_NO_CLAIM_AUTHORITY", "receipt status")
        require(receipt.get("payload_authority_granted") is False and receipt.get("public_commit_evidence") is False, "receipt nonclaim flags")
        require(receipt.get("development_source_root_sha256") == source_root, "receipt/marker source")
        receipt_inventory = receipt.get("development_source_inventory")
        require(receipt_inventory == expected_inventory, "receipt exact source inventory")
        producer_root = sha(canonical(receipt_inventory))
        require(producer_root == source_root, "producer source root")
        cells = receipt["all150"]["cells"]
        selectors = [cell["selector_ordinal"] for cell in cells]
        require(len(cells) == 150 and selectors == list(range(150)), "receipt exact all150 selectors")
        gpu = receipt["independent_gpu_identity"]
        require(re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu["device_uuid"]) is not None, "GPU UUID")
        require(re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", gpu["pci_bus_id"]) is not None, "PCI bus")
        end_marker = os.stat(marker_name(final_name), dir_fd=parent_fd, follow_symlinks=False)
        end_directory = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        require((end_marker.st_dev, end_marker.st_ino) == (marker_info.st_dev, marker_info.st_ino), "marker name changed")
        require((end_directory.st_dev, end_directory.st_ino) == (directory_info.st_dev, directory_info.st_ino), "directory name changed")
        return {
            "receipt_sha256": sha(receipt_bytes),
            "receipt_bytes": len(receipt_bytes),
            "development_source_root_sha256": source_root,
            "bound_source_preflight_receipt_sha256": receipt["bound_source_preflight_receipt_sha256"],
            "directory_root_sha256": directory_root,
            "parent_commit_sha256": marker["parent_commit_sha256"],
            "parent_marker_sha256": sha(marker_bytes),
            "completion_sha256": complete["completion_sha256"],
            "final_directory_device": directory_info.st_dev,
            "final_directory_inode": directory_info.st_ino,
            "marker_device": marker_info.st_dev,
            "marker_inode": marker_info.st_ino,
            "marker_link_count": marker_info.st_nlink,
            "all150_selectors_exact_0_149": True,
            "gpu_name": gpu["device_name"],
            "gpu_uuid": gpu["device_uuid"],
            "pci_bus_id": gpu["pci_bus_id"],
            "payload_opened": False,
            "qwen_opened": False,
        }
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--gpu-output-parent", required=True)
    parser.add_argument("--gpu-final-name", required=True)
    args = parser.parse_args()
    inventory, root = authenticate_source(Path(args.package))
    receipt = verify_gpu_receipt(Path(args.gpu_output_parent), args.gpu_final_name, inventory)
    print(json.dumps({
        "schema": "uwfa-sc-v7-independent-source-only-audit-v1",
        "status": "PASS_INDEPENDENT_SOURCE_AND_GPU_RECEIPT_AUTHENTICATION",
        "inventory": inventory,
        "inventory_bytes": sum(row["bytes"] for row in inventory),
        "inventory_root_sha256": root,
        "gpu_receipt": receipt,
        "payload_opened": False,
        "qwen_opened": False,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
