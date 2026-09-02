#!/usr/bin/env python3
"""Independent sealed-source authenticator for UWFA-SC v8.

This verifier intentionally imports no producer module.  It authenticates the
literal public Git commit, the transition from the independently reviewed
pre-freeze parent, the manifest and every declared source member.  It never
looks for or opens any model, Qwen, current-codec, extracted-stream, or
Gaussian-control payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "unifilar-wfa-entropy-census-independent-source-review-v8"
EXPECTED_COMMIT = "d563c4ac1e78a6b6e7f0722291211d1209f775af"
EXPECTED_PARENT = "2315551e504b0c7c1e357793aa259b745ff4d717"
PACKAGE_REL = "research/unifilar_wfa_entropy_census_stage0_v8"
EXPECTED_MANIFEST_SHA256 = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"

EXPECTED_ROWS = [
    ("INDEPENDENT_BOOTSTRAP_ABI.md", 11025, "b46b2703121d2e50460025bc0c5ff53ca28fffb94a1a2b23e58a52ce41bd2160"),
    ("README.md", 16213, "253b89f19c041118fb4148d8cbf76ebc71301b701e20f8cfefa421d77df68d0c"),
    ("container_codec.py", 93379, "645debb547a76818a880bfc346a2dd6230af97b07dc832afb3548a83d6920fed"),
    ("cupy_backend.py", 40964, "7904a5e122686487d89fb684b70052507089bfe3bbfe4f1f02520df6ce3fb1ba"),
    ("design_lock.json", 11554, "da0514b2e1fa0f033b113912bbe05e7ae640c3a606fa5386ee202d45dcc71805"),
    ("dispatcher_contract.py", 9205, "747db5747b75074c1191e17055d615df3cddc54da00e29ba03edfd99ddb2a243"),
    ("fixture_long_memory.py", 4307, "d72e7c109920f7d2c6a64bcbf9de0c6463ae80b40cbdb3e772af44c30b3a8c38"),
    ("fixture_portability.py", 16350, "b8e9c8d0741f5c7de44ad9ae2bedf8ea6b0fba3ec6fa58df80d8d08fb5a8a1db"),
    ("protocol.py", 21051, "9e18675a1e646eb10c0900aa3767bff96666943309dbd8db3953c745888d2cc1"),
    ("result_envelope.py", 19002, "ad568758b318a9a6f298da2dc17edcd7f7639e2f772511ae680798f301bc4601"),
    ("run_source_free_gpu_dev.py", 8263, "888c5420353951d164a76015e6563154df119f1481da29621154a01347791838"),
    ("stage0_census.py", 123776, "7b7c2e0fcb6593805e6b2c8234ae59cb42d90fbb7dcf945a35aa5dfe331ae618"),
    ("strata_sc_adapter.py", 36184, "08fc8808ac168f6930ee9482e160f25f2bd087829fca4630553aea3510d722c6"),
    ("test_source_only.py", 135687, "5dc3730b629dc3c05a1353d036c6a9049013b6c163540c31f2cb8275d5a68383"),
    ("universal_adapter.py", 11577, "a5ab2e1919af98c2aa9b3032faa0ba5552efe05cca250bd6844fd48c76aabbc8"),
    ("uwfa_common.py", 58875, "db53567ab6d71d5150cc92ef4a78fa9ce5cca01f5474fa2ca32edc8711cc4325"),
    ("verify_source.py", 15907, "c9ccbcd0b68681400dab97636bad7e4d445a83f2446d032b53863a8ab77b7714"),
]

EXPECTED_ATTESTATION = {
    "model_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
    "current_finite_artifact_or_selected_stream_opened_statted_hashed_or_enumerated": False,
    "gaussian_control_opened_statted_hashed_or_enumerated": False,
    "numpy_directly_imported_by_builder": False,
    "source_free_cupy_development_run_launched_by_builder": True,
    "source_free_cuda_development_run_launched_by_builder": True,
    "development_run_is_claim_evidence": False,
}

EXPECTED_POST_FREEZE = [
    "EXTERNAL_PINNED_V8_SOURCE_AUDIT",
    "EXTERNAL_PINNED_DISPATCHER_AUDIT",
    "PUBLIC_GITHUB_COMMIT_FRESH_RTX5090_ALL150_AND_REPRESENTATIVE_REPLAY",
    "FRESH_PROCESS_INDEPENDENT_RESULT_AUDIT",
    "NO_PAYLOAD_BEFORE_EXTERNAL_AUTHORITY",
]


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ReviewError(f"nonfinite JSON token: {value}")

    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    except ReviewError:
        raise
    except Exception as exc:
        raise ReviewError(f"invalid UTF-8 JSON: {exc}") from exc
    require(isinstance(parsed, dict), "JSON root is not an object")
    return parsed


def no_symlink_components(path: Path) -> Path:
    resolved = path.absolute()
    cursor = Path(resolved.anchor)
    for component in resolved.parts[1:]:
        cursor = cursor / component
        require(os.path.lexists(cursor), f"missing path component: {cursor}")
        require(not stat.S_ISLNK(os.lstat(cursor).st_mode), f"symlink path component: {cursor}")
    return resolved


def held_regular_bytes(path: Path, cap: int = 1 << 30) -> bytes:
    path = no_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"not a regular file: {path.name}")
        require(0 <= before.st_size <= cap, f"file outside size cap: {path.name}")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            require(bool(chunk), f"short read: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(fd, 1) == b"", f"growing file: {path.name}")
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        require(identity_before == identity_after, f"file changed during held read: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def git(repo: Path, *args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ReviewError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="strict").strip()


def git_blob(repo: Path, revision: str, relative: str) -> bytes:
    value = git(repo, "show", f"{revision}:{relative}", binary=True)
    require(isinstance(value, bytes), "internal binary Git result")
    return value


def review(repo: Path) -> dict[str, Any]:
    repo = no_symlink_components(repo)
    require((repo / ".git").is_dir(), "not a Git checkout")
    package = no_symlink_components(repo / PACKAGE_REL)

    head = git(repo, "rev-parse", "HEAD")
    require(head == EXPECTED_COMMIT, f"wrong checked-out commit: {head}")
    parents = git(repo, "rev-list", "--parents", "-n", "1", EXPECTED_COMMIT).split()
    require(parents == [EXPECTED_COMMIT, EXPECTED_PARENT], f"unexpected commit parent list: {parents}")
    require(git(repo, "status", "--porcelain=v1", "--untracked-files=no") == "", "tracked checkout is dirty")

    changed_text = git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        EXPECTED_PARENT,
        EXPECTED_COMMIT,
        "--",
        PACKAGE_REL,
    )
    changed = [line for line in changed_text.splitlines() if line]
    expected_changed = [
        f"M\t{PACKAGE_REL}/README.md",
        f"A\t{PACKAGE_REL}/SOURCE_MANIFEST.json",
        f"M\t{PACKAGE_REL}/design_lock.json",
    ]
    require(changed == expected_changed, f"unexpected freeze transition: {changed}")

    expected_map = {name: (size, digest) for name, size, digest in EXPECTED_ROWS}
    protected = sorted(set(expected_map) - {"README.md", "design_lock.json"}, key=lambda item: item.encode("utf-8"))
    unchanged_sha256: list[dict[str, Any]] = []
    for name in protected:
        relative = f"{PACKAGE_REL}/{name}"
        before = git_blob(repo, EXPECTED_PARENT, relative)
        after = git_blob(repo, EXPECTED_COMMIT, relative)
        require(before == after, f"non-lifecycle source changed during freeze: {name}")
        size, digest = expected_map[name]
        require(len(after) == size and sha256(after) == digest, f"protected Git blob mismatch: {name}")
        unchanged_sha256.append({"name": name, "bytes": size, "sha256": digest})

    before_design = strict_json(git_blob(repo, EXPECTED_PARENT, f"{PACKAGE_REL}/design_lock.json"))
    after_design = strict_json(git_blob(repo, EXPECTED_COMMIT, f"{PACKAGE_REL}/design_lock.json"))
    require(before_design.get("status") == "PRE_REVIEW_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "pre-freeze design status")
    require(after_design.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "sealed design status")
    normalized_before = dict(before_design)
    normalized_before["status"] = after_design["status"]
    require(normalized_before == after_design, "design_lock changed beyond the lifecycle status")

    before_readme = git_blob(repo, EXPECTED_PARENT, f"{PACKAGE_REL}/README.md")
    after_readme = git_blob(repo, EXPECTED_COMMIT, f"{PACKAGE_REL}/README.md")
    require(len(before_readme) == 16067 and sha256(before_readme) == "ba6d7aa17494dd8a1ef34cf28fa0aea5e3b187443ae6dd3278f61eda3b3c43f2", "pre-freeze README binding")
    require(len(after_readme) == 16213 and sha256(after_readme) == expected_map["README.md"][1], "sealed README binding")

    actual_names: set[str] = set()
    with os.scandir(package) as entries:
        for entry in entries:
            require(not entry.is_symlink(), f"symlink package member: {entry.name}")
            require(entry.is_file(follow_symlinks=False), f"non-regular package member: {entry.name}")
            actual_names.add(entry.name)
    expected_actual = set(expected_map) | {"SOURCE_MANIFEST.json"}
    require(actual_names == expected_actual, f"undeclared/missing package members: {sorted(actual_names ^ expected_actual)}")

    manifest_bytes = held_regular_bytes(package / "SOURCE_MANIFEST.json", 1 << 20)
    require(len(manifest_bytes) == 3518, "literal manifest byte length")
    require(sha256(manifest_bytes) == EXPECTED_MANIFEST_SHA256, "literal manifest SHA-256")
    require(git_blob(repo, EXPECTED_COMMIT, f"{PACKAGE_REL}/SOURCE_MANIFEST.json") == manifest_bytes, "manifest differs from committed blob")
    manifest = strict_json(manifest_bytes)
    require(set(manifest) == {"schema", "status", "members", "access_attestation", "post_freeze_requirements"}, "manifest fields")
    require(manifest["schema"] == "unifilar-wfa-source-manifest-v8", "manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "manifest status")
    require(manifest["access_attestation"] == EXPECTED_ATTESTATION, "manifest access attestation")
    require(manifest["post_freeze_requirements"] == EXPECTED_POST_FREEZE, "manifest post-freeze requirements")
    expected_manifest_rows = [
        {"name": name, "bytes": size, "sha256": digest}
        for name, size, digest in EXPECTED_ROWS
    ]
    require(manifest["members"] == expected_manifest_rows, "manifest member rows/order")

    observed: list[dict[str, Any]] = []
    for name, expected_size, expected_digest in EXPECTED_ROWS:
        data = held_regular_bytes(package / name)
        require(len(data) == expected_size, f"member byte length: {name}")
        require(sha256(data) == expected_digest, f"member SHA-256: {name}")
        require(git_blob(repo, EXPECTED_COMMIT, f"{PACKAGE_REL}/{name}") == data, f"member differs from committed blob: {name}")
        if name.endswith(".py"):
            compile(data, f"<independent-syntax:{name}>", "exec", dont_inherit=True, optimize=0)
        observed.append({"name": name, "bytes": expected_size, "sha256": expected_digest})

    return {
        "schema": SCHEMA,
        "status": "PASS_INDEPENDENT_SOURCE_REVIEW",
        "reviewed_public_commit": EXPECTED_COMMIT,
        "reviewed_public_commit_parent": EXPECTED_PARENT,
        "reviewed_source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "reviewed_manifest_bytes": len(manifest_bytes),
        "reviewed_member_count_excluding_manifest": len(observed),
        "reviewed_member_bytes_excluding_manifest": sum(row["bytes"] for row in observed),
        "reviewed_package_bytes_including_manifest": len(manifest_bytes) + sum(row["bytes"] for row in observed),
        "freeze_transition_changed_paths": changed,
        "freeze_transition_only_lifecycle_readme_design_and_manifest": True,
        "non_lifecycle_members_byte_identical_to_prefreeze_parent": True,
        "python_members_byte_identical_to_prefreeze_parent": True,
        "design_lock_only_status_changed": True,
        "manifest_members": observed,
        "unchanged_prefreeze_members": unchanged_sha256,
        "producer_modules_imported_by_reviewer": False,
        "payloads_opened_statted_hashed_or_enumerated": False,
        "qwen_opened_statted_hashed_or_enumerated": False,
        "current_artifact_opened_statted_hashed_or_enumerated": False,
        "gaussian_control_opened_statted_hashed_or_enumerated": False,
        "payload_authority_granted": False,
        "remaining_external_gates": EXPECTED_POST_FREEZE,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = review(Path(args.repo))
    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_INDEPENDENT_SOURCE_REVIEW: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
