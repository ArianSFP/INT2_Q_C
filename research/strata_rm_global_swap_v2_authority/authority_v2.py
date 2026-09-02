#!/usr/bin/env python3
"""Fail-closed authority for the global STRATA RM-order swap v2.

This module contains no model discovery and no encoder.  It authenticates the
frozen v1 producer and its independent review, snapshots every executable
source before use, consumes only auditor-owned scientific capabilities, and
derives rate/distortion/read claims from literal bytes.

The physical entry point is intentionally unusable without four out-of-band
pins: this package, the experiment commitment, the successful decoder-audit
closure/receipt, and the scientific-provenance capability.  None of those
pins may be taken from the experiment commitment itself.
"""

from __future__ import annotations

import array
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


V1_SOURCE_ROOT_SHA256 = (
    "980a5f1d272ca5ffc7b4d35e7c234a86994d135fcacaf0d47a8b3e00fc3d4f14")
V1_MANIFEST_SHA256 = (
    "4c2c5b371b1b9661d371de607e6a650f8c43fe0128726476854c2eb2ca560c85")
V1_REVIEW_SOURCE_ROOT_SHA256 = (
    "1dfa55969b87543adbee785d72933f9ccb6f754eaade9e4e340a022c96c1afa8")
V1_REVIEW_MANIFEST_SHA256 = (
    "19f0051e901b5c824f761d2f74309a908bfb9303df3c8bfbfad69bd326958802")
V2_MANIFEST_SCHEMA = "strata-rm-global-swap-v2-authority-source-manifest"
EXTERNAL_PINS = {
    "agent_polaris_qwen_rht_encoder.py":
        "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "bg_codec_bec_encoder.py":
        "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267",
    "strata_v2_klt_mixed_independent_auditor_v1.py":
        "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e",
}
TARGET_N = (1 << 20, 1 << 21)
RATE_MIN = 2.15
RATE_MAX = 2.5
TARGET_F = 0.8
MAX_READ_AMPLIFICATION = 2.0
MIN_SOURCE_SPECIFIC_BPW = 0.03
ROLE_ORDER = ("gate", "up", "down")
PRODUCTION_AUTHORIZATION = "AUDIT_LITERAL_GLOBAL_RM_SWAP_RESULT_V2"
FIXTURE_AUTHORIZATION = "SOURCE_ONLY_SYNTHETIC_AUTHORITY_FIXTURE_V2"
HEX = frozenset("0123456789abcdef")


class AuthorityError(RuntimeError):
    """The authority boundary failed closed."""


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
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate JSON key")
            result[key] = value
        return result

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
    """Read one regular non-link file while detecting simple replacement races."""
    candidate = Path(path)
    try:
        before = candidate.lstat()
        require(stat.S_ISREG(before.st_mode) and not candidate.is_symlink(),
                f"{label}: regular non-symlink file")
        payload = candidate.read_bytes()
        after = candidate.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: read") from exc
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_mode)
    require(identity(before) == identity(after), f"{label}: changed during read")
    return payload


def real_directory(path: Path, label: str) -> Path:
    """Reject a linked package root *before* resolving it (v1 review gap 1)."""
    original = Path(path)
    try:
        before = original.lstat()
        require(stat.S_ISDIR(before.st_mode) and not original.is_symlink(),
                f"{label}: real non-link directory")
        resolved = original.resolve(strict=True)
        after = original.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: directory resolution") from exc
    require((before.st_dev, before.st_ino, before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_mode),
            f"{label}: root changed during resolution")
    require(resolved.is_dir() and not resolved.is_symlink(),
            f"{label}: resolved real directory")
    return resolved


def _safe_relative(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}: relative path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts and
            "." not in pure.parts and "\\" not in value,
            f"{label}: safe POSIX relative path")
    return Path(*pure.parts)


def resolve_member(root: Path, relative: Any, label: str) -> Path:
    resolved_root = real_directory(root, f"{label} root")
    rel = _safe_relative(relative, label)
    current = resolved_root
    try:
        for part in rel.parts:
            current = current / part
            before = current.lstat()
            require(not stat.S_ISLNK(before.st_mode),
                    f"{label}: symlink component")
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError(f"{label}: resolution") from exc
    require(resolved_root in resolved.parents and resolved != resolved_root,
            f"{label}: containment")
    return resolved


def _manifest_root(rows: list[dict[str, Any]]) -> str:
    return sha256(canonical_json(rows))


def authenticate_flat_package(package: Path, *, manifest_name: str,
                              expected_manifest_sha256: str,
                              expected_source_root_sha256: str,
                              expected_schema: str) -> dict[str, Any]:
    require(is_sha256(expected_manifest_sha256) and
            is_sha256(expected_source_root_sha256), "external package pins")
    root = real_directory(package, "dependency package")
    manifest_payload = regular_bytes(root / manifest_name, "dependency manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "dependency manifest external pin")
    manifest = strict_json(manifest_payload, "dependency manifest")
    require(manifest.get("schema") == expected_schema and
            manifest.get("source_root_sha256") == expected_source_root_sha256,
            "dependency schema/root binding")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency members")
    names: list[str] = []
    observed: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name != manifest_name and name not in names,
                "dependency member name")
        payload = regular_bytes(root / name, f"dependency member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin {name}")
        observed.append(item)
        names.append(name)
    require(_manifest_root(observed) == expected_source_root_sha256,
            "dependency recomputed source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {manifest_name} and
            all(entry.is_file(follow_symlinks=False) and
                not entry.is_dir(follow_symlinks=False) for entry in entries),
            "dependency exact regular closure")
    return {"path": str(root), "manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "members": len(rows)}


def authenticate_v1_and_review(v1_package: Path,
                               review_package: Path) -> dict[str, Any]:
    v1 = authenticate_flat_package(
        v1_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V1_MANIFEST_SHA256,
        expected_source_root_sha256=V1_SOURCE_ROOT_SHA256,
        expected_schema="strata-rm-global-swap-v1-hardened-source-manifest")
    review = authenticate_flat_package(
        review_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V1_REVIEW_MANIFEST_SHA256,
        expected_source_root_sha256=V1_REVIEW_SOURCE_ROOT_SHA256,
        expected_schema=(
            "strata-rm-global-swap-v1-hardened-reproducibility-review-manifest"))
    review_manifest = strict_json(
        regular_bytes(Path(review["path"]) / "source_manifest.json",
                      "v1 review manifest"), "v1 review manifest")
    require(review_manifest.get("producer_source_root_sha256") ==
            V1_SOURCE_ROOT_SHA256 and
            review_manifest.get("producer_manifest_sha256") ==
            V1_MANIFEST_SHA256, "review-to-v1 binding")
    return {"v1": v1, "review": review,
            "status": "PASS_PINNED_V1_AND_NINE_GAP_REVIEW"}


def authenticate_v2_package(package: Path,
                            expected_manifest_sha256: str) -> dict[str, Any]:
    require(is_sha256(expected_manifest_sha256), "v2 manifest external pin")
    root = real_directory(package, "v2 package")
    payload = regular_bytes(root / "source_manifest.json", "v2 manifest")
    require(sha256(payload) == expected_manifest_sha256,
            "v2 manifest external pin mismatch")
    manifest = strict_json(payload, "v2 manifest")
    require(manifest.get("schema") == V2_MANIFEST_SCHEMA and
            manifest.get("v1_source_root_sha256") == V1_SOURCE_ROOT_SHA256 and
            manifest.get("v1_review_source_root_sha256") ==
            V1_REVIEW_SOURCE_ROOT_SHA256, "v2 manifest lineage")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "v2 members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "v2 member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name != "source_manifest.json",
                "v2 member name")
        member = regular_bytes(root / name, f"v2 member {name}")
        item = {"name": name, "bytes": len(member), "sha256": sha256(member)}
        require(item == row, f"v2 member pin {name}")
        observed.append(item)
        names.append(name)
    require(_manifest_root(observed) == manifest.get("source_root_sha256"),
            "v2 recomputed source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {"source_manifest.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "v2 exact regular closure")
    return {"path": str(root), "manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": manifest["source_root_sha256"],
            "member_rows": observed}


def _write_immutable(path: Path, payload: bytes, label: str) -> None:
    require(not path.exists(), f"{label}: destination pre-exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    require(regular_bytes(path, label) == payload, f"{label}: snapshot byte parity")


def snapshot_pinned_files(source_root: Path, pins: Mapping[str, str],
                          destination: Path) -> dict[str, Any]:
    """Copy exact current files into a private read-only snapshot before use."""
    source = real_directory(source_root, "snapshot source")
    dest = Path(destination)
    require(not dest.exists(), "snapshot destination must be fresh")
    dest.mkdir(parents=True)
    rows = []
    for name, expected in sorted(pins.items()):
        require(Path(name).name == name and is_sha256(expected), "snapshot pin row")
        payload = regular_bytes(source / name, f"snapshot source {name}")
        require(sha256(payload) == expected, f"snapshot source pin {name}")
        _write_immutable(dest / name, payload, f"snapshot copy {name}")
        rows.append({"name": name, "bytes": len(payload), "sha256": expected})
    require({entry.name for entry in os.scandir(dest)} == set(pins),
            "snapshot exact closure")
    return {"path": str(dest.resolve(strict=True)), "rows": rows,
            "snapshot_root_sha256": _manifest_root(rows), "immutable": True}


def sanitized_worker_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    allowed = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
               "CUDA_PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
    result = {name: environment[name] for name in allowed if name in environment}
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONHASHSEED"] = "0"
    return result


def module_origin_outside_controlled_roots(module_name: str,
                                           controlled_roots: Iterable[Path]) -> Path:
    spec = importlib.util.find_spec(module_name)
    require(spec is not None and spec.origin not in (None, "built-in", "frozen"),
            f"{module_name} origin")
    origin = Path(str(spec.origin)).resolve(strict=True)
    for controlled in controlled_roots:
        root = Path(controlled).resolve(strict=True)
        require(origin != root and root not in origin.parents,
                f"{module_name} inside controlled root")
    return origin


def run_snapshot_worker(*, package: Path, expected_manifest_sha256: str,
                        worker_name: str, worker_args: list[str],
                        timeout_seconds: int = 1800) -> dict[str, Any]:
    """Execute a package worker only from a complete immutable v2 snapshot."""
    auth = authenticate_v2_package(package, expected_manifest_sha256)
    require(worker_name in {"current_snapshot_worker.py", "parity_worker.py"},
            "approved worker name")
    with tempfile.TemporaryDirectory(prefix="strata-rm-v2-worker-") as directory:
        root = Path(directory).resolve(strict=True)
        source = root / "source"
        pins = {row["name"]: row["sha256"] for row in auth["member_rows"]}
        pins["source_manifest.json"] = expected_manifest_sha256
        snapshot_pinned_files(Path(auth["path"]), pins, source)
        output = root / "receipt.json"
        command = [sys.executable, "-I", "-B", str(source / worker_name),
                   *worker_args, "--output", str(output)]
        completed = subprocess.run(
            command, cwd=root, env=sanitized_worker_environment(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
        require(completed.returncode == 0,
                "snapshot worker failed: " +
                completed.stderr.decode("utf-8", errors="replace")[-3000:])
        receipt = strict_json(regular_bytes(output, "worker receipt"),
                              "worker receipt")
    require(receipt.get("fresh_interpreter") is True and
            receipt.get("python_isolated_flag") is True and
            receipt.get("payloads_opened") == 0,
            "snapshot worker authority semantics")
    return receipt


def run_current_integration_snapshot(*, package: Path,
                                     expected_manifest_sha256: str,
                                     external_root: Path,
                                     timeout_seconds: int = 1800) -> dict[str, Any]:
    """Snapshot current external modules before the integration worker imports."""
    with tempfile.TemporaryDirectory(prefix="strata-rm-v2-current-") as directory:
        root = Path(directory).resolve(strict=True)
        external_snapshot = root / "external"
        snapshot_pinned_files(external_root, EXTERNAL_PINS, external_snapshot)
        receipt = run_snapshot_worker(
            package=package, expected_manifest_sha256=expected_manifest_sha256,
            worker_name="current_snapshot_worker.py",
            worker_args=["--external-snapshot", str(external_snapshot)],
            timeout_seconds=timeout_seconds)
        for name, expected in EXTERNAL_PINS.items():
            require(sha256(regular_bytes(external_snapshot / name,
                                         f"post-worker snapshot {name}")) == expected,
                    f"post-worker immutable external pin {name}")
    return receipt


def authenticate_decoder_audit_capability(
        audit_root: Path, *, expected_manifest_sha256: str,
        expected_source_root_sha256: str,
        expected_receipt_sha256: str,
        expected_decoder_worker_sha256: str,
        expected_launcher_sha256: str) -> dict[str, Any]:
    """Authenticate a separately pinned *successful* independent decoder audit."""
    require(all(is_sha256(value) for value in
                (expected_manifest_sha256, expected_source_root_sha256,
                 expected_receipt_sha256, expected_decoder_worker_sha256,
                 expected_launcher_sha256)), "decoder-audit out-of-band pins")
    root = real_directory(audit_root, "decoder audit capability")
    manifest_payload = regular_bytes(root / "source_manifest.json",
                                     "decoder audit manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "decoder audit external manifest pin")
    manifest = strict_json(manifest_payload, "decoder audit manifest")
    require(canonical_json(manifest) + b"\n" == manifest_payload,
            "decoder audit manifest canonical bytes")
    require(set(manifest) == {"schema", "producer_worker_sha256",
                              "instrumented_launcher_sha256",
                              "source_root_sha256", "receipt_name", "members"} and
            manifest["schema"] ==
            "strata-rm-global-swap-v2-decoder-independent-audit-manifest" and
            manifest["producer_worker_sha256"] == expected_decoder_worker_sha256 and
            manifest["instrumented_launcher_sha256"] == expected_launcher_sha256 and
            manifest["source_root_sha256"] == expected_source_root_sha256 and
            manifest["receipt_name"] == "AUDIT_RECEIPT.json",
            "decoder audit manifest binding")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows and
            _manifest_root(rows) == expected_source_root_sha256,
            "decoder audit source root")
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "decoder audit member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name not in
                {"source_manifest.json", "AUDIT_RECEIPT.json"},
                "decoder audit member name")
        payload = regular_bytes(root / name, f"decoder audit member {name}")
        require({"name": name, "bytes": len(payload), "sha256": sha256(payload)} == row,
                f"decoder audit member pin {name}")
        names.append(name)
    require({entry.name for entry in os.scandir(root)} ==
            set(names) | {"source_manifest.json", "AUDIT_RECEIPT.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in os.scandir(root)),
            "decoder audit exact closure")
    receipt_payload = regular_bytes(root / "AUDIT_RECEIPT.json",
                                    "decoder audit receipt")
    require(sha256(receipt_payload) == expected_receipt_sha256,
            "decoder audit out-of-band receipt pin")
    receipt = strict_json(receipt_payload, "decoder audit receipt")
    require(canonical_json(receipt) + b"\n" == receipt_payload,
            "decoder audit receipt canonical bytes")
    required = {"schema", "executed", "status", "producer_worker_sha256",
                "instrumented_launcher_sha256", "audit_source_root_sha256",
                "protocol", "filesystem_bypass_absent",
                "source_payload_access_absent",
                "packet_only_read_instrumentation_verified",
                "canonical_replay_verified", "fixed_universal_decoder",
                "qwen_specific_tables_absent", "hostile_tests", "payloads_opened"}
    require(set(receipt) == required and receipt["schema"] ==
            "strata-rm-global-swap-v2-decoder-independent-audit-receipt" and
            receipt["executed"] is True and receipt["status"] ==
            "PASS_INDEPENDENT_DECODER_AUDIT_V2" and
            receipt["producer_worker_sha256"] == expected_decoder_worker_sha256 and
            receipt["instrumented_launcher_sha256"] == expected_launcher_sha256 and
            receipt["audit_source_root_sha256"] == expected_source_root_sha256 and
            receipt["protocol"] == "strata-rm-v2-decoder-worker-protocol" and
            all(receipt[name] is True for name in
                ("filesystem_bypass_absent", "source_payload_access_absent",
                 "packet_only_read_instrumentation_verified",
                 "canonical_replay_verified", "fixed_universal_decoder",
                 "qwen_specific_tables_absent")) and
            isinstance(receipt["hostile_tests"], int) and
            receipt["hostile_tests"] >= 10 and receipt["payloads_opened"] == 0,
            "successful independent decoder audit receipt")
    return {"manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "receipt_sha256": expected_receipt_sha256,
            "worker_sha256": expected_decoder_worker_sha256,
            "status": "PASS_SEPARATELY_PINNED_SUCCESSFUL_DECODER_AUDIT"}


def _validate_source_rows(rows: Any, label: str) -> None:
    require(isinstance(rows, list) and rows, f"{label}: source rows")
    ordinals = []
    groups: dict[tuple[int, int], dict[str, list[int]]] = {}
    for row in rows:
        required = {"ordinal", "role", "layer", "expert", "shape",
                    "relative_path", "bytes", "sha256"}
        require(isinstance(row, dict) and set(row) == required,
                f"{label}: source schema")
        shape = row["shape"]
        require(isinstance(row["ordinal"], int) and row["ordinal"] >= 0 and
                row["role"] in ROLE_ORDER and
                isinstance(row["layer"], int) and row["layer"] >= 0 and
                isinstance(row["expert"], int) and row["expert"] >= 0 and
                isinstance(shape, list) and len(shape) == 2 and
                all(isinstance(value, int) and value > 0 for value in shape) and
                row["bytes"] == 2 * shape[0] * shape[1] and
                is_sha256(row["sha256"]), f"{label}: source metadata")
        _safe_relative(row["relative_path"], f"{label}: source path")
        ordinals.append(row["ordinal"])
        key = (row["layer"], row["expert"])
        require(row["role"] not in groups.setdefault(key, {}),
                f"{label}: duplicate role")
        groups[key][row["role"]] = shape
    require(ordinals == list(range(len(rows))), f"{label}: contiguous ordinals")
    for roles in groups.values():
        require(set(roles) == set(ROLE_ORDER) and
                roles["gate"] == roles["up"] and
                roles["down"] == [roles["gate"][1], roles["gate"][0]],
                f"{label}: complete SwiGLU geometry")


def authenticate_scientific_capability(path: Path,
                                       expected_sha256: str) -> dict[str, Any]:
    """Authenticate auditor-owned source/checkpoint/control/family provenance."""
    require(is_sha256(expected_sha256), "scientific capability external pin")
    payload = regular_bytes(path, "scientific capability")
    require(sha256(payload) == expected_sha256,
            "scientific capability out-of-band pin")
    record = strict_json(payload, "scientific capability")
    require(canonical_json(record) + b"\n" == payload,
            "scientific capability canonical bytes")
    required = {"schema", "owner", "audit_execution", "selection",
                "architecture_families", "cases", "status"}
    require(set(record) == required and record["schema"] ==
            "strata-rm-global-swap-v2-scientific-capability" and
            record["owner"] == "independent_auditor" and
            record["status"] == "PASS_AUDITOR_OWNED_PROVENANCE_CAPABILITIES",
            "scientific capability authority")
    execution = record["audit_execution"]
    require(isinstance(execution, dict) and set(execution) ==
            {"receipt_sha256", "auditor_source_root_sha256", "executed", "status"} and
            is_sha256(execution["receipt_sha256"]) and
            is_sha256(execution["auditor_source_root_sha256"]) and
            execution["executed"] is True and execution["status"] ==
            "PASS_INDEPENDENT_PROVENANCE_AUDIT", "scientific audit execution")
    selection = record["selection"]
    require(isinstance(selection, dict) and set(selection) ==
            {"frozen_before_test", "test_bytes_opened", "search_replayed_on_controls",
             "pipeline_sha256"} and selection["frozen_before_test"] is True and
            selection["test_bytes_opened"] == 0 and
            selection["search_replayed_on_controls"] is True and
            is_sha256(selection["pipeline_sha256"]), "sealed selection capability")
    families = record["architecture_families"]
    require(isinstance(families, list) and len(families) >= 2 and
            len(families) == len(set(families)) and
            all(isinstance(value, str) and value for value in families),
            "audited architecture families")
    cases = record["cases"]
    require(isinstance(cases, list) and cases, "scientific cases")
    by_id: dict[str, dict[str, Any]] = {}
    for row in cases:
        required_case = {
            "capability_id", "kind", "architecture_family", "pipeline_sha256",
            "checkpoint_manifest_sha256", "tensor_manifest_sha256",
            "control_family", "paired_model_capability_id",
            "generator_sha256", "seed_commitment_sha256", "moments_sha256",
            "required_control_capability_ids", "sources"}
        require(isinstance(row, dict) and set(row) == required_case,
                "scientific case schema")
        capability_id = row["capability_id"]
        require(isinstance(capability_id, str) and capability_id and
                capability_id not in by_id and
                row["pipeline_sha256"] == selection["pipeline_sha256"],
                "scientific case identity/pipeline")
        require(row["kind"] in {"qwen_bf16", "swiglu_moe_bf16",
                                "matched_gaussian_bf16"}, "scientific case kind")
        require(row["architecture_family"] in families,
                "case family is auditor-owned")
        _validate_source_rows(row["sources"], capability_id)
        if row["kind"] in {"qwen_bf16", "swiglu_moe_bf16"}:
            require(is_sha256(row["checkpoint_manifest_sha256"]) and
                    is_sha256(row["tensor_manifest_sha256"]) and
                    row["control_family"] is None and
                    row["paired_model_capability_id"] is None and
                    row["generator_sha256"] is None and
                    row["seed_commitment_sha256"] is None and
                    row["moments_sha256"] is None and
                    isinstance(row["required_control_capability_ids"], list) and
                    row["required_control_capability_ids"],
                    "auditor-owned model/checkpoint capability")
        else:
            require(row["checkpoint_manifest_sha256"] is None and
                    row["tensor_manifest_sha256"] is None and
                    isinstance(row["control_family"], str) and
                    row["control_family"] and
                    isinstance(row["paired_model_capability_id"], str) and
                    is_sha256(row["generator_sha256"]) and
                    is_sha256(row["seed_commitment_sha256"]) and
                    is_sha256(row["moments_sha256"]) and
                    row["required_control_capability_ids"] == [],
                    "auditor-owned matched-control capability")
        by_id[capability_id] = row
    model_rows = [row for row in cases if row["kind"] != "matched_gaussian_bf16"]
    control_rows = [row for row in cases if row["kind"] == "matched_gaussian_bf16"]
    require({row["architecture_family"] for row in model_rows} == set(families),
            "every claimed family has auditor-owned model evidence")
    referenced_controls = []
    for model in model_rows:
        controls = model["required_control_capability_ids"]
        require(len(controls) == len(set(controls)), "unique required controls")
        for control_id in controls:
            require(control_id in by_id and
                    by_id[control_id]["kind"] == "matched_gaussian_bf16" and
                    by_id[control_id]["paired_model_capability_id"] ==
                    model["capability_id"] and
                    by_id[control_id]["architecture_family"] ==
                    model["architecture_family"] and
                    [source["shape"] for source in by_id[control_id]["sources"]] ==
                    [source["shape"] for source in model["sources"]],
                    "exact auditor-owned matched control")
            referenced_controls.append(control_id)
    require(len(referenced_controls) == len(set(referenced_controls)) and
            set(referenced_controls) ==
            {row["capability_id"] for row in control_rows},
            "exact control closure; no unpaired or multiply-used controls")
    return {"sha256": expected_sha256, "record": record, "cases": by_id,
            "status": "PASS_AUDITOR_OWNED_SCIENTIFIC_CAPABILITIES"}


def _strict_commitment(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(is_sha256(expected_sha256), "commitment external pin")
    payload = regular_bytes(path, "experiment commitment")
    require(sha256(payload) == expected_sha256,
            "experiment commitment out-of-band pin")
    record = strict_json(payload, "experiment commitment")
    require(canonical_json(record) + b"\n" == payload,
            "experiment commitment canonical bytes")
    required = {"schema", "mode", "v1_source_root_sha256",
                "v1_review_source_root_sha256", "decoder_worker", "cases"}
    require(set(record) == required and record["schema"] ==
            "strata-rm-global-swap-v2-physical-commitment" and
            record["v1_source_root_sha256"] == V1_SOURCE_ROOT_SHA256 and
            record["v1_review_source_root_sha256"] ==
            V1_REVIEW_SOURCE_ROOT_SHA256, "experiment commitment lineage/schema")
    worker = record["decoder_worker"]
    require(isinstance(worker, dict) and set(worker) ==
            {"relative_path", "bytes", "sha256", "protocol"} and
            isinstance(worker["bytes"], int) and worker["bytes"] > 0 and
            is_sha256(worker["sha256"]) and worker["protocol"] ==
            "strata-rm-v2-decoder-worker-protocol", "decoder commitment")
    _safe_relative(worker["relative_path"], "decoder worker")
    cases = record["cases"]
    require(isinstance(cases, list) and cases, "committed cases")
    ids = set()
    capability_ids = set()
    for row in cases:
        require(isinstance(row, dict) and set(row) ==
                {"case_id", "capability_id", "packet"}, "commitment case schema")
        require(isinstance(row["case_id"], str) and row["case_id"] and
                row["case_id"] not in ids and
                isinstance(row["capability_id"], str) and row["capability_id"] and
                row["capability_id"] not in capability_ids,
                "unique commitment case/capability")
        packet = row["packet"]
        require(isinstance(packet, dict) and set(packet) ==
                {"relative_path", "bytes", "sha256"} and
                isinstance(packet["bytes"], int) and packet["bytes"] > 0 and
                is_sha256(packet["sha256"]), "literal packet commitment")
        _safe_relative(packet["relative_path"], "packet")
        ids.add(row["case_id"])
        capability_ids.add(row["capability_id"])
    return record


def _bf16_values(payload: bytes) -> array.array:
    require(payload and len(payload) % 2 == 0, "nonempty BF16 source")
    words = array.array("H")
    words.frombytes(payload)
    if sys.byteorder != "little":
        words.byteswap()
    wide = array.array("I", (int(value) << 16 for value in words))
    values = array.array("f")
    values.frombytes(wide.tobytes())
    return values


def _f64_values(payload: bytes) -> array.array:
    require(payload and len(payload) % 8 == 0, "nonempty FP64 reconstruction")
    values = array.array("d")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def exact_bf16_f64_score(source_bf16: bytes,
                         reconstruction_f64: bytes) -> dict[str, Any]:
    source = _bf16_values(source_bf16)
    reconstruction = _f64_values(reconstruction_f64)
    require(len(source) == len(reconstruction), "source/reconstruction count")
    require(all(math.isfinite(value) for value in source) and
            all(math.isfinite(value) for value in reconstruction),
            "finite source/reconstruction")
    energy = math.fsum(float(value) ** 2 for value in source)
    sse = math.fsum((float(left) - float(right)) ** 2
                    for left, right in zip(source, reconstruction, strict=True))
    require(energy > 0.0 and math.isfinite(energy) and math.isfinite(sse),
            "finite scoring domain")
    return {"weights": len(source), "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex(), "relative_mse": sse / energy}


def _read_pinned(root: Path, row: Mapping[str, Any], label: str) -> bytes:
    path = resolve_member(root, row["relative_path"], label)
    payload = regular_bytes(path, label)
    require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"],
            f"{label}: literal byte pin")
    return payload


def _validate_instrumentation(record: Mapping[str, Any], packet: bytes,
                              case_id: str) -> dict[str, Any]:
    required = {"schema", "case_id", "packet_sha256", "packet_bytes",
                "packet_open_count", "packet_read_operations", "denied_read_paths",
                "denied_os_open", "denied_process_escape", "source_paths_supplied",
                "total_packet_bytes_read", "unique_packet_bytes_read", "operations",
                "status"}
    require(set(record) == required and record["schema"] ==
            "strata-rm-v2-instrumented-decoder-io-receipt" and
            record["case_id"] == case_id and
            record["packet_sha256"] == sha256(packet) and
            record["packet_bytes"] == len(packet) and
            record["packet_open_count"] >= 1 and
            record["packet_read_operations"] >= 1 and
            record["denied_read_paths"] == 0 and
            record["denied_os_open"] is True and
            record["denied_process_escape"] is True and
            record["source_paths_supplied"] == 0 and
            record["status"] == "PASS_INSTRUMENTED_LITERAL_PACKET_IO",
            "instrumented decoder receipt")
    operations = record["operations"]
    require(isinstance(operations, list) and operations, "instrumented operations")
    intervals = []
    total = 0
    for row in operations:
        require(isinstance(row, dict) and set(row) == {"offset", "length"} and
                isinstance(row["offset"], int) and row["offset"] >= 0 and
                isinstance(row["length"], int) and row["length"] > 0 and
                row["offset"] + row["length"] <= len(packet),
                "instrumented packet operation")
        total += row["length"]
        intervals.append((row["offset"], row["offset"] + row["length"]))
    intervals.sort()
    covered = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    covered += end - start
    require(total == record["total_packet_bytes_read"] and
            covered == record["unique_packet_bytes_read"] == len(packet),
            "instrumentation reconciles literal packet bytes")
    return {"literal_read_bytes": total, "unique_packet_bytes": covered,
            "read_amplification": total / len(packet),
            "instrumented_not_decoder_reported": True}


def _run_case(*, committed: Mapping[str, Any], capability: Mapping[str, Any],
              evidence_root: Path, decoder_payload: bytes,
              launcher_payload: bytes, timeout_seconds: int) -> dict[str, Any]:
    packet = _read_pinned(evidence_root, committed["packet"],
                          f"packet {committed['case_id']}")
    sources = [_read_pinned(evidence_root, row,
                            f"source {committed['case_id']}:{row['ordinal']}")
               for row in capability["sources"]]
    request_sources = [{key: row[key] for key in
                        ("ordinal", "role", "layer", "expert", "shape")}
                       for row in capability["sources"]]
    request = {"schema": "strata-rm-v2-decoder-request",
               "case_id": committed["case_id"],
               "packet_sha256": sha256(packet), "packet_bytes": len(packet),
               "sources": request_sources}
    with tempfile.TemporaryDirectory(prefix="strata-rm-v2-decode-") as directory:
        root = Path(directory).resolve(strict=True)
        decoder = root / "decoder.py"
        launcher = root / "instrumented_decoder_worker.py"
        packet_path = root / "packet.bin"
        request_path = root / "request.json"
        output_dir = root / "output"
        output_dir.mkdir()
        instrumentation_path = root / "instrumentation.json"
        _write_immutable(decoder, decoder_payload, "decoder snapshot")
        _write_immutable(launcher, launcher_payload, "launcher snapshot")
        _write_immutable(packet_path, packet, "packet snapshot")
        _write_immutable(request_path, canonical_json(request) + b"\n",
                         "request snapshot")
        command = [sys.executable, "-I", "-B", str(launcher),
                   "--decoder", str(decoder), "--request", str(request_path),
                   "--packet", str(packet_path), "--output-dir", str(output_dir),
                   "--instrumentation-output", str(instrumentation_path),
                   "--case-id", committed["case_id"]]
        completed = subprocess.run(
            command, cwd=root, env=sanitized_worker_environment(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
        require(completed.returncode == 0,
                "instrumented independent decoder failed: " +
                completed.stderr.decode("utf-8", errors="replace")[-3000:])
        instrumentation = strict_json(
            regular_bytes(instrumentation_path, "instrumentation receipt"),
            "instrumentation receipt")
        io_result = _validate_instrumentation(
            instrumentation, packet, committed["case_id"])
        decoder_receipt = strict_json(
            regular_bytes(output_dir / "receipt.json", "decoder receipt"),
            "decoder receipt")
        canonical = regular_bytes(output_dir / "canonical_packet.bin",
                                  "canonical packet")
        reconstructions = [regular_bytes(
            output_dir / f"reconstruction-{row['ordinal']:04d}.f64",
            f"reconstruction {row['ordinal']}") for row in capability["sources"]]
        require(sha256(regular_bytes(decoder, "post-run decoder snapshot")) ==
                sha256(decoder_payload) and
                sha256(regular_bytes(launcher, "post-run launcher snapshot")) ==
                sha256(launcher_payload), "immutable executable snapshots")
    required_receipt = {"schema", "case_id", "packet_sha256", "packet_bytes",
                        "canonical_packet_sha256", "canonical_packet_bytes",
                        "independent_decode_complete", "canonical_reencode_complete",
                        "causal_probabilities_regenerated", "packet_consumed_exactly",
                        "encoder_decisions_read", "encoder_probabilities_read",
                        "source_payloads_opened", "reconstruction_files", "status"}
    require(set(decoder_receipt) == required_receipt and
            decoder_receipt["schema"] ==
            "strata-rm-v2-independent-decoder-receipt" and
            decoder_receipt["case_id"] == committed["case_id"] and
            decoder_receipt["packet_sha256"] == sha256(packet) and
            decoder_receipt["packet_bytes"] == len(packet) and
            decoder_receipt["canonical_packet_sha256"] == sha256(canonical) and
            decoder_receipt["canonical_packet_bytes"] == len(canonical) and
            decoder_receipt["independent_decode_complete"] is True and
            decoder_receipt["canonical_reencode_complete"] is True and
            decoder_receipt["causal_probabilities_regenerated"] is True and
            decoder_receipt["packet_consumed_exactly"] is True and
            decoder_receipt["encoder_decisions_read"] is False and
            decoder_receipt["encoder_probabilities_read"] is False and
            decoder_receipt["source_payloads_opened"] is False and
            decoder_receipt["reconstruction_files"] ==
            [f"reconstruction-{row['ordinal']:04d}.f64"
             for row in capability["sources"]] and
            decoder_receipt["status"] == "PASS_INDEPENDENT_DECODE_V2" and
            canonical == packet, "independent decoder/canonical replay")
    score_rows = [exact_bf16_f64_score(source, reconstruction)
                  for source, reconstruction in zip(sources, reconstructions,
                                                    strict=True)]
    weights = sum(row["weights"] for row in score_rows)
    sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in score_rows)
    energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in score_rows)
    rate = 8.0 * len(packet) / weights
    relative = sse / energy
    factor = relative * 2.0 ** (2.0 * rate)
    return {"case_id": committed["case_id"],
            "capability_id": committed["capability_id"],
            "kind": capability["kind"],
            "architecture_family": capability["architecture_family"],
            "control_family": capability["control_family"],
            "paired_model_capability_id": capability["paired_model_capability_id"],
            "weights": weights, "literal_packet_bytes": len(packet),
            "literal_packet_sha256": sha256(packet),
            "physical_rate_bpw": rate, "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex(), "relative_mse": relative,
            "F": factor, "saving_bpw": -0.5 * math.log2(factor),
            "read": io_result, "matrix_rows": score_rows,
            "canonical_reencode_byte_identical": True}


def _pool(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    require(rows, "nonempty pooling rows")
    weights = sum(row["weights"] for row in rows)
    bits = 8 * sum(row["literal_packet_bytes"] for row in rows)
    sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in rows)
    energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in rows)
    rate = bits / weights
    relative = sse / energy
    factor = relative * 2.0 ** (2.0 * rate)
    return {"weights": weights, "physical_bits": bits,
            "physical_rate_bpw": rate, "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex(), "relative_mse": relative,
            "F": factor, "saving_bpw": -0.5 * math.log2(factor),
            "maximum_read_amplification": max(
                row["read"]["read_amplification"] for row in rows)}


def evaluate_family_acceptance(results: list[Mapping[str, Any]],
                               scientific: Mapping[str, Any],
                               *, enforce: bool = True) -> dict[str, Any]:
    """Apply target and strongest-control gates to *every* claimed family."""
    capability_cases = {row["capability_id"]: row
                        for row in scientific["cases"]}
    result_by_capability = {row["capability_id"]: row for row in results}
    require(set(result_by_capability) == set(capability_cases),
            "literal results cover exact scientific capability")
    family_rows = []
    for family in scientific["architecture_families"]:
        models = [row for row in results
                  if row["architecture_family"] == family and
                  row["kind"] != "matched_gaussian_bf16"]
        require(models, f"family {family}: model results")
        model_pool = _pool(models)
        control_groups: dict[str, list[Mapping[str, Any]]] = {}
        for model in models:
            capability = capability_cases[model["capability_id"]]
            for control_id in capability["required_control_capability_ids"]:
                control = result_by_capability[control_id]
                control_groups.setdefault(control["control_family"], []).append(control)
        require(control_groups and all(len(rows) == len(models)
                                       for rows in control_groups.values()),
                f"family {family}: complete strongest-control panel")
        control_pools = {name: _pool(rows)
                         for name, rows in sorted(control_groups.items())}
        strongest_name, strongest = max(
            control_pools.items(), key=lambda item: item[1]["saving_bpw"])
        advantage = model_pool["saving_bpw"] - strongest["saving_bpw"]
        passed = (RATE_MIN <= model_pool["physical_rate_bpw"] <= RATE_MAX and
                  model_pool["F"] <= TARGET_F and
                  model_pool["maximum_read_amplification"] <
                  MAX_READ_AMPLIFICATION and
                  all(RATE_MIN <= pool["physical_rate_bpw"] <= RATE_MAX and
                      pool["maximum_read_amplification"] <
                      MAX_READ_AMPLIFICATION for pool in control_pools.values()) and
                  advantage >= MIN_SOURCE_SPECIFIC_BPW)
        if enforce:
            require(passed, f"family {family}: target/control/read acceptance")
        family_rows.append({"architecture_family": family, "model": model_pool,
                            "controls": control_pools,
                            "strongest_control": strongest_name,
                            "source_specific_advantage_bpw": advantage,
                            "minimum_required_source_specific_bpw":
                            MIN_SOURCE_SPECIFIC_BPW, "passed": passed})
    qwen = [row for row in results if row["kind"] == "qwen_bf16"]
    require(qwen, "absolute Qwen result required")
    qwen_pool = _pool(qwen)
    if enforce:
        require(qwen_pool["F"] <= TARGET_F,
                "absolute pooled Qwen F <= 0.8")
    return {"families": family_rows, "qwen": qwen_pool,
            "all_families_passed": all(row["passed"] for row in family_rows),
            "strongest_control_subtracted": True,
            "per_family_target_enforced": True}


def validate_physical_bundle(
        *, v2_package: Path, expected_v2_manifest_sha256: str,
        evidence_root: Path, commitment_path: Path,
        expected_commitment_sha256: str,
        scientific_capability_path: Path,
        expected_scientific_capability_sha256: str,
        decoder_audit_root: Path,
        expected_decoder_audit_manifest_sha256: str,
        expected_decoder_audit_source_root_sha256: str,
        expected_decoder_audit_receipt_sha256: str,
        authorization: str, timeout_seconds: int = 3600) -> dict[str, Any]:
    """Derive one physical result using no encoder-owned provenance labels."""
    mode = ("production_global_rm_swap" if authorization == PRODUCTION_AUTHORIZATION
            else "synthetic_authority_fixture" if authorization ==
            FIXTURE_AUTHORIZATION else None)
    require(mode is not None, "explicit v2 authority token")
    package_auth = authenticate_v2_package(v2_package,
                                           expected_v2_manifest_sha256)
    evidence = real_directory(evidence_root, "evidence root")
    try:
        commitment_relative = str(
            Path(commitment_path).resolve(strict=True).relative_to(evidence)
        ).replace(os.sep, "/")
    except (OSError, ValueError) as exc:
        raise AuthorityError("commitment must be inside evidence root") from exc
    commitment_resolved = resolve_member(
        evidence, commitment_relative, "experiment commitment")
    commitment = _strict_commitment(commitment_resolved,
                                    expected_commitment_sha256)
    require(commitment["mode"] == mode, "commitment mode")
    scientific = authenticate_scientific_capability(
        scientific_capability_path, expected_scientific_capability_sha256)
    committed_capabilities = {row["capability_id"] for row in commitment["cases"]}
    require(committed_capabilities == set(scientific["cases"]),
            "commitment uses exact auditor capability case set")
    worker_row = commitment["decoder_worker"]
    worker = resolve_member(evidence, worker_row["relative_path"], "decoder worker")
    worker_payload = regular_bytes(worker, "decoder worker")
    require(len(worker_payload) == worker_row["bytes"] and
            sha256(worker_payload) == worker_row["sha256"],
            "decoder worker literal pin")
    launcher_path = Path(package_auth["path"]) / "instrumented_decoder_worker.py"
    launcher_payload = regular_bytes(launcher_path, "instrumented launcher")
    decoder_audit = authenticate_decoder_audit_capability(
        decoder_audit_root,
        expected_manifest_sha256=expected_decoder_audit_manifest_sha256,
        expected_source_root_sha256=expected_decoder_audit_source_root_sha256,
        expected_receipt_sha256=expected_decoder_audit_receipt_sha256,
        expected_decoder_worker_sha256=worker_row["sha256"],
        expected_launcher_sha256=sha256(launcher_payload))
    require(mode == "production_global_rm_swap",
            "fixture physical path deliberately omitted from frozen v2")
    results = [_run_case(
        committed=row, capability=scientific["cases"][row["capability_id"]],
        evidence_root=evidence, decoder_payload=worker_payload,
        launcher_payload=launcher_payload, timeout_seconds=timeout_seconds)
        for row in commitment["cases"]]
    acceptance = evaluate_family_acceptance(
        results, scientific["record"], enforce=True)
    return {"schema": "strata-rm-global-swap-v2-derived-physical-result",
            "mode": mode, "commitment_sha256": expected_commitment_sha256,
            "scientific_capability_sha256":
            expected_scientific_capability_sha256,
            "decoder_independent_audit": decoder_audit,
            "cases": results, "acceptance": acceptance,
            "source_provenance_owned_by_auditor": True,
            "checkpoint_provenance_owned_by_auditor": True,
            "control_provenance_owned_by_auditor": True,
            "family_provenance_owned_by_auditor": True,
            "literal_packet_reads_instrumented": True,
            "decoder_executed_from_immutable_snapshot": True,
            "caller_supplied_metrics_accepted": False,
            "status": "PASS_LITERAL_QWEN_TARGET_ALL_FAMILIES_AND_STRONGEST_CONTROLS"}
