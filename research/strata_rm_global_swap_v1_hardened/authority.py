#!/usr/bin/env python3
"""Fail-closed authority primitives for the global STRATA RM-order swap.

This module is deliberately source- and payload-agnostic.  It authenticates
the immutable v0 mechanism, its independent audit, and the exact current
encoder lineage.  It never accepts a caller-supplied reliability hook.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


V0_SOURCE_ROOT_SHA256 = (
    "4f856e268d37ee1d6f32b4a2d1b8cd6879c235639ad75809ffd75fc7c4372d6c")
V0_MANIFEST_SHA256 = (
    "939b57518a4afe11c56c59a20f109e423c8eab5815c947cb3e5a91d559704b3c")
V0_AUDIT_SOURCE_ROOT_SHA256 = (
    "7eabe4580908d4a79eceb2f7fdaf838d535028c06263c2f4841032664db11ad0")
V0_AUDIT_MANIFEST_SHA256 = (
    "228d8377396ea8d599054e111b2862c02acf0f765a681038decb0be8f8644a39")
EXTERNAL_PINS = {
    "agent_polaris_qwen_rht_encoder.py":
        "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "bg_codec_bec_encoder.py":
        "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267",
    "strata_v2_klt_mixed_independent_auditor_v1.py":
        "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e",
}
TARGET_N = (1 << 20, 1 << 21)
V1_MANIFEST_SCHEMA = "strata-rm-global-swap-v1-hardened-source-manifest"
HEX = frozenset("0123456789abcdef")


class AuthorityError(RuntimeError):
    """Authentication or authority boundary failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AuthorityError("noncanonical JSON value") from exc


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs_hook(pairs):
        row = {}
        for key, value in pairs:
            require(key not in row, f"{label}: duplicate JSON key")
            row[key] = value
        return row

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuthorityError(f"{label}: nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    """Read one immutable regular file and reject links/races."""
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label}: regular non-symlink file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: read") from exc
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
             after.st_mode), f"{label}: changed during read")
    return payload


def _manifest_root(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["name"]):
        digest.update(row["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def authenticate_flat_package(package: Path, *, manifest_name: str,
                              expected_manifest_sha256: str,
                              expected_source_root_sha256: str,
                              expected_schema: str) -> dict[str, Any]:
    """Authenticate an exact flat closure, including entry kinds.

    Unlike v0, every top-level entry is enumerated.  Directories, FIFOs,
    sockets, device nodes, junctions and symlinks are all rejected.
    """
    original = Path(package)
    try:
        original_stat = original.lstat()
        require(stat.S_ISDIR(original_stat.st_mode) and not original.is_symlink(),
                "dependency package must not be a link")
        root = original.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError("dependency package resolution") from exc
    require(root.is_dir() and not root.is_symlink(), "dependency real directory")
    manifest_payload = regular_bytes(root / manifest_name, "dependency manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "dependency manifest external pin")
    manifest = strict_json(manifest_payload, "dependency manifest")
    require(manifest.get("schema") == expected_schema, "dependency manifest schema")
    require(manifest.get("source_root_sha256") == expected_source_root_sha256,
            "dependency declared source root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency member list")
    names: list[str] = []
    observed: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member schema")
        name = row["name"]
        require(isinstance(name, str) and name and Path(name).name == name and
                name not in names and name != manifest_name,
                "dependency flat unique member")
        require(isinstance(row["bytes"], int) and row["bytes"] >= 0 and
                is_sha256(row["sha256"]), "dependency member metadata")
        payload = regular_bytes(root / name, f"dependency member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin: {name}")
        names.append(name)
        observed.append(item)
    require(_manifest_root(observed) == expected_source_root_sha256,
            "dependency independently recomputed root")
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise AuthorityError("dependency closure scan") from exc
    require({entry.name for entry in entries} == set(names) | {manifest_name},
            "dependency exact top-level closure")
    require(all(entry.is_file(follow_symlinks=False) and
                not entry.is_dir(follow_symlinks=False) for entry in entries),
            "dependency closure contains only regular files")
    return {
        "path": str(root), "manifest_sha256": expected_manifest_sha256,
        "source_root_sha256": expected_source_root_sha256,
        "members": len(observed), "exact_entry_closure": True,
        "symlinks_and_directories_rejected": True,
    }


def authenticate_dependencies(v0_package: Path, audit_package: Path) -> dict[str, Any]:
    v0 = authenticate_flat_package(
        v0_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V0_MANIFEST_SHA256,
        expected_source_root_sha256=V0_SOURCE_ROOT_SHA256,
        expected_schema="strata-rm-global-swap-v0-source-manifest")
    audit = authenticate_flat_package(
        audit_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V0_AUDIT_MANIFEST_SHA256,
        expected_source_root_sha256=V0_AUDIT_SOURCE_ROOT_SHA256,
        expected_schema="strata-rm-global-swap-v0-independent-audit-manifest")
    audit_manifest = strict_json(
        regular_bytes(audit_package.resolve() / "source_manifest.json",
                      "v0 audit manifest"), "v0 audit manifest")
    require(audit_manifest.get("producer_source_root_sha256") ==
            V0_SOURCE_ROOT_SHA256, "v0 audit producer binding")
    return {"v0": v0, "v0_independent_audit": audit,
            "status": "PASS_PINNED_V0_AND_AUDIT_CLOSURE"}


def authenticate_v1_package(package: Path,
                            expected_manifest_sha256: str) -> dict[str, Any]:
    """Authenticate this package from one out-of-band manifest hash."""
    require(is_sha256(expected_manifest_sha256), "v1 external manifest SHA-256")
    original = Path(package)
    try:
        original_stat = original.lstat()
        require(stat.S_ISDIR(original_stat.st_mode) and not original.is_symlink(),
                "v1 package real directory")
        root = original.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError("v1 package resolution") from exc
    manifest_payload = regular_bytes(root / "source_manifest.json", "v1 manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "v1 manifest external pin")
    manifest = strict_json(manifest_payload, "v1 manifest")
    require(manifest.get("schema") == V1_MANIFEST_SCHEMA, "v1 manifest schema")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "v1 manifest members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "v1 member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and
                name not in names and name != "source_manifest.json",
                "v1 flat unique member")
        payload = regular_bytes(root / name, f"v1 member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"v1 member pin {name}")
        observed.append(item)
        names.append(name)
    require(sha256(canonical_json(observed)) == manifest.get("source_root_sha256"),
            "v1 source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {"source_manifest.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "v1 exact regular closure")
    return {"path": str(root), "manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": manifest["source_root_sha256"],
            "member_rows": observed, "status": "PASS_V1_SOURCE_CLOSURE"}


def _function_text(path: Path, name: str) -> str:
    text = regular_bytes(path, f"external source {path.name}").decode("utf-8")
    tree = ast.parse(text)
    rows = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
            node.name == name]
    require(len(rows) == 1, f"external function {name}")
    return ast.get_source_segment(text, rows[0]) or ""


def authenticate_current_external_root(external_root: Path) -> dict[str, Any]:
    original = Path(external_root)
    try:
        original_stat = original.lstat()
        require(stat.S_ISDIR(original_stat.st_mode) and not original.is_symlink(),
                "external root must not be a link")
        root = original.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError("external root resolution") from exc
    require(root.is_dir(), "external root directory")
    for name, expected in EXTERNAL_PINS.items():
        payload = regular_bytes(root / name, f"external pin {name}")
        require(sha256(payload) == expected, f"external source pin: {name}")
    base = root / "agent_polaris_qwen_rht_encoder.py"
    bg = root / "bg_codec_bec_encoder.py"
    decoder = root / "strata_v2_klt_mixed_independent_auditor_v1.py"
    sc = _function_text(base, "sc_encode_ratio").replace(" ", "")
    trial = _function_text(base, "run_trial")
    bec = _function_text(bg, "bec_flags").replace(" ", "")
    decoded = _function_text(decoder, "decode_one_block")
    require("external_u=u[reverse].copy()" in sc, "current SC orientation")
    require("x_bit = polar_transform(chosen.external_u)" in trial,
            "current encoder reconstruction orientation")
    require("external[order[:keep]]=0" in bec and
            "external[reverse].copy()" in bec, "current BEC hook semantics")
    require("sc_seed + 1_000_003 * level" in decoded and
            "canonical_payload == payload_packed" in decoded,
            "independent decoder causal replay")
    return {
        "external_root": str(root), "external_pins": dict(EXTERNAL_PINS),
        "base_module": "agent_polaris_qwen_rht_encoder",
        "reference_module": "bg_codec_bec_encoder",
        "reference_hook": "bec_flags", "installed_hook": "reliability_freeze_flags",
        "independent_decoder": "strata_v2_klt_mixed_independent_auditor_v1.py",
        "status": "PASS_CURRENT_SOURCE_LINEAGE__NO_MODULES_IMPORTED",
    }


def sanitized_worker_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an allowlisted environment; never inherit Python import knobs."""
    environment = dict(os.environ if source is None else source)
    allowed = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
               "CUDA_PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
    result = {name: environment[name] for name in allowed if name in environment}
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONHASHSEED"] = "0"
    return result


def isolated_worker_command(package: Path, *, worker_name: str,
                            external_root: Path, output: Path) -> list[str]:
    """Build the only permitted integration-worker command."""
    worker = package.resolve(strict=True) / worker_name
    require(worker_name in {"current_integration_worker.py", "real_cupy_worker.py"},
            "pinned integration worker name")
    regular_bytes(worker, "integration worker")
    return [sys.executable, "-I", "-B", str(worker), "--external-root",
            str(external_root.resolve(strict=True)), "--output", str(output)]


def run_isolated_worker(package: Path, *, expected_manifest_sha256: str,
                        worker_name: str, external_root: Path,
                        timeout_seconds: int = 900) -> dict[str, Any]:
    """Launch a fresh worker after source-only authentication.

    The worker path is package-owned, the interpreter is ``sys.executable``,
    and no caller-controlled module/hook/backend object crosses the boundary.
    """
    source_auth = authenticate_v1_package(package, expected_manifest_sha256)
    authenticate_current_external_root(external_root)
    with tempfile.TemporaryDirectory(prefix="strata-rm-v1-worker-") as directory:
        root = Path(directory).resolve(strict=True)
        snapshot = root / "source"
        snapshot.mkdir()
        original = Path(source_auth["path"])
        names = [row["name"] for row in source_auth["member_rows"]]
        for name in names + ["source_manifest.json"]:
            payload = regular_bytes(original / name, f"v1 snapshot source {name}")
            if name != "source_manifest.json":
                expected = next(row for row in source_auth["member_rows"]
                                if row["name"] == name)
                require(len(payload) == expected["bytes"] and
                        sha256(payload) == expected["sha256"],
                        f"v1 snapshot pin {name}")
            else:
                require(sha256(payload) == expected_manifest_sha256,
                        "v1 snapshot manifest pin")
            (snapshot / name).write_bytes(payload)
        authenticate_v1_package(snapshot, expected_manifest_sha256)
        output = root / "receipt.json"
        command = isolated_worker_command(
            snapshot, worker_name=worker_name, external_root=external_root,
            output=output)
        completed = subprocess.run(
            command, cwd=root, env=sanitized_worker_environment(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
        require(completed.returncode == 0,
                f"isolated worker failed: {completed.stderr.decode('utf-8', errors='replace')[-2000:]}")
        receipt = strict_json(regular_bytes(output, "isolated worker receipt"),
                              "isolated worker receipt")
    require(receipt.get("external_pins") == EXTERNAL_PINS,
            "isolated worker external pins")
    require(receipt.get("fresh_interpreter") is True and
            receipt.get("python_isolated_flag") is True and
            receipt.get("pythonpath_inherited") is False and
            receipt.get("payloads_opened") == 0,
            "isolated worker authority fields")
    return receipt


def module_origin_outside_controlled_roots(module_name: str,
                                           controlled_roots: list[Path]) -> Path:
    spec = importlib.util.find_spec(module_name)
    require(spec is not None and spec.origin not in (None, "built-in", "frozen"),
            f"{module_name} import origin")
    origin = Path(str(spec.origin)).resolve(strict=True)
    for controlled in controlled_roots:
        root = controlled.resolve(strict=True)
        require(origin != root and root not in origin.parents,
                f"{module_name} resolves inside controlled root")
    return origin
