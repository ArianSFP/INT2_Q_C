#!/usr/bin/env python3
"""Create an exact one-shot FOSP-v2 auxiliary-run authorization.

The builder is intentionally unable to waive either independent audit.  It
binds their exact bytes, the audited source-free runtime tuple, the immutable
package closure, the fixed source root, and one create-new result path.  The
resulting JSON is still an external controller artifact: this source package
alone grants no model access.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


SOURCE_AUDIT_SCHEMA = "free-order-swiglu-path-v2-independent-source-audit-receipt-v1"
SOURCE_AUDIT_STATUS = "PASS_V2_INDEPENDENT_SOURCE_AUDIT"
RUNTIME_AUDIT_SCHEMA = "free-order-swiglu-path-v2-independent-runtime-audit-receipt-v1"
RUNTIME_AUDIT_STATUS = "PASS_V2_INDEPENDENT_RUNTIME_AUDIT"
SCOPE_LITERAL = "FOSP_V2_AUXILIARY_DISCOVERY_ONLY_NO_PINNED_PANEL"


def _load_oracle(package: Path) -> Any:
    path = package / "free_order_oracle_v2.py"
    spec = importlib.util.spec_from_file_location("fosp_v2_authorization_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"file grew while open: {path}")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"file identity changed while open: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    raw = _regular_bytes(path)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _pairwise_disjoint(paths: Sequence[Path]) -> bool:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if _is_within(left, right) or _is_within(right, left):
                return False
    return True


def _require_zero(mapping: dict[str, Any], keys: Sequence[str], label: str) -> None:
    for key in keys:
        if int(mapping.get(key, -1)) != 0:
            raise RuntimeError(f"{label} does not prove zero {key}")


def _canonical_seal(value: dict[str, Any], oracle: Any, label: str) -> str:
    unsigned = dict(value)
    observed = str(unsigned.pop("canonical_unsigned_sha256", ""))
    if observed != oracle.canonical_sha256(unsigned):
        raise RuntimeError(f"{label} canonical seal mismatch")
    return observed


def _manifest_binds(raw: bytes, receipt_sha256: str, label: str) -> None:
    rows: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        pieces = line.split("  ")
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise RuntimeError(f"malformed {label} manifest")
        digest, name = pieces
        if name in rows or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"malformed {label} manifest row")
        rows[name] = digest
    if rows.get("audit_receipt.json") != receipt_sha256 or "verify_audit.py" not in rows:
        raise RuntimeError(f"{label} manifest does not bind receipt and verifier")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-manifest", type=Path, required=True)
    parser.add_argument("--source-audit-receipt", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--runtime-audit-manifest", type=Path, required=True)
    parser.add_argument("--runtime-audit-receipt", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--authorization-output", type=Path, required=True)
    parser.add_argument("--scope-literal", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.scope_literal != SCOPE_LITERAL:
        raise RuntimeError("explicit discovery-only scope acknowledgement missing")
    package = Path(__file__).absolute().parent.resolve(strict=True)
    oracle = _load_oracle(package)
    artifact_rows, artifact_raw = oracle._artifact_rows(package)
    artifact_sha = hashlib.sha256(artifact_raw).hexdigest()

    source_audit_manifest_path = args.source_audit_manifest.resolve(strict=True)
    source_audit_receipt_path = args.source_audit_receipt.resolve(strict=True)
    runtime_receipt_path = args.runtime_receipt.resolve(strict=True)
    runtime_audit_manifest_path = args.runtime_audit_manifest.resolve(strict=True)
    runtime_audit_receipt_path = args.runtime_audit_receipt.resolve(strict=True)
    if source_audit_manifest_path.name != "AUDIT_SHA256SUMS.txt":
        raise RuntimeError("source audit manifest filename drift")
    if source_audit_receipt_path.name != "audit_receipt.json":
        raise RuntimeError("source audit receipt filename drift")
    if runtime_receipt_path.name != "runtime_receipt.json":
        raise RuntimeError("runtime receipt filename drift")
    if runtime_audit_manifest_path.name != "AUDIT_SHA256SUMS.txt":
        raise RuntimeError("runtime audit manifest filename drift")
    if runtime_audit_receipt_path.name != "audit_receipt.json":
        raise RuntimeError("runtime audit receipt filename drift")
    if source_audit_manifest_path.parent != source_audit_receipt_path.parent:
        raise RuntimeError("source audit closure directory mismatch")
    if runtime_audit_manifest_path.parent != runtime_audit_receipt_path.parent:
        raise RuntimeError("runtime audit closure directory mismatch")

    source_audit, source_audit_sha = _json_with_hash(source_audit_receipt_path)
    runtime_receipt, runtime_receipt_sha = _json_with_hash(runtime_receipt_path)
    runtime_audit, runtime_audit_sha = _json_with_hash(runtime_audit_receipt_path)
    source_audit_manifest_raw = _regular_bytes(source_audit_manifest_path)
    runtime_audit_manifest_raw = _regular_bytes(runtime_audit_manifest_path)
    source_audit_manifest_sha = hashlib.sha256(source_audit_manifest_raw).hexdigest()
    runtime_audit_manifest_sha = hashlib.sha256(runtime_audit_manifest_raw).hexdigest()
    _manifest_binds(source_audit_manifest_raw, source_audit_sha, "source audit")
    _manifest_binds(runtime_audit_manifest_raw, runtime_audit_sha, "runtime audit")

    if source_audit.get("schema") != SOURCE_AUDIT_SCHEMA or source_audit.get("status") != SOURCE_AUDIT_STATUS:
        raise RuntimeError("independent source audit is not an exact PASS receipt")
    if source_audit.get("artifact_set_status") != "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET":
        raise RuntimeError("source audit artifact set is not immutable PASS")
    source_audit_internal = _canonical_seal(source_audit, oracle, "source audit")
    expected_package = {
        "artifact_manifest_sha256": artifact_sha,
        "source_only_receipt_sha256": artifact_rows["source_only_receipt.json"],
        "runner_sha256": artifact_rows["free_order_oracle_v2.py"],
        "runtime_calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
        "source_bindings_sha256": oracle.BINDINGS_SHA256,
    }
    source_package = source_audit.get("audited_package", {})
    if source_package != expected_package:
        raise RuntimeError("source audit did not bind this exact package")
    if source_audit.get("v1_counterexample_replay", {}).get("status") != "PASS_COUNTEREXAMPLE_REACHES_DIRECT_STAGE":
        raise RuntimeError("source audit did not replay the v1 counterexample")
    _require_zero(
        source_audit.get("zero_access_ledger", {}),
        (
            "qwen_or_model_payload_files_opened",
            "qwen_or_model_payload_bytes_read",
            "pinned_panel_files_opened",
            "validation_files_opened",
            "cupy_imports",
            "cuda_api_calls",
            "gpu_device_calls",
            "external_data_fetches",
        ),
        "source audit",
    )

    if runtime_receipt.get("schema") != "free_order_swiglu_path_runtime_calibration_v2":
        raise RuntimeError("wrong runtime calibration schema")
    if runtime_receipt.get("status") != "PASS_SOURCE_FREE_FULL_GEOMETRY_RUNTIME_CALIBRATION":
        raise RuntimeError("runtime calibration did not pass")
    internal_sha = _canonical_seal(runtime_receipt, oracle, "runtime receipt")
    runtime_package = runtime_receipt.get("artifact_binding", {})
    if runtime_package != {
        "artifact_manifest_sha256": artifact_sha,
        "runner_sha256": artifact_rows["free_order_oracle_v2.py"],
        "calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
    }:
        raise RuntimeError("runtime receipt package binding mismatch")
    _require_zero(
        runtime_receipt.get("zero_access_ledger", {}),
        (
            "workspace_or_source_arguments_supported",
            "source_bindings_loaded",
            "qwen_or_model_payload_files_opened",
            "qwen_or_model_payload_bytes_read",
            "pinned_panel_files_opened",
            "validation_files_opened",
            "external_data_fetches",
            "production_result_files_opened",
            "production_gpu_jobs",
        ),
        "runtime receipt",
    )
    if int(runtime_receipt.get("zero_access_ledger", {}).get("synthetic_gpu_jobs", -1)) != 1:
        raise RuntimeError("runtime receipt did not execute exactly one synthetic GPU job")

    if runtime_audit.get("schema") != RUNTIME_AUDIT_SCHEMA or runtime_audit.get("status") != RUNTIME_AUDIT_STATUS:
        raise RuntimeError("independent runtime audit is not an exact PASS receipt")
    if runtime_audit.get("artifact_set_status") != "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET":
        raise RuntimeError("runtime audit artifact set is not immutable PASS")
    runtime_audit_internal = _canonical_seal(runtime_audit, oracle, "runtime audit")
    audited_runtime = runtime_audit.get("audited_runtime_receipt", {})
    if audited_runtime.get("file_sha256") != runtime_receipt_sha or audited_runtime.get("internal_sha256") != internal_sha:
        raise RuntimeError("runtime audit did not bind this exact runtime receipt")
    if runtime_audit.get("audited_package") != expected_package:
        raise RuntimeError("runtime audit package binding mismatch")
    _require_zero(
        runtime_audit.get("zero_access_ledger", {}),
        (
            "qwen_or_model_payload_files_opened",
            "qwen_or_model_payload_bytes_read",
            "pinned_panel_files_opened",
            "validation_files_opened",
            "production_result_files_opened",
            "production_gpu_jobs",
            "cupy_imports",
            "cuda_api_calls",
            "gpu_device_calls",
        ),
        "runtime audit",
    )

    workspace_argument = args.workspace_root.absolute()
    if workspace_argument.is_symlink():
        raise RuntimeError("workspace root symlink forbidden")
    workspace = workspace_argument.resolve(strict=True)
    if not workspace.is_dir():
        raise RuntimeError("workspace root is not a directory")
    output_argument = args.output.absolute()
    if output_argument.exists() or output_argument.is_symlink():
        raise RuntimeError("production output already exists")
    output_parent = output_argument.parent.resolve(strict=True)
    output = output_parent / output_argument.name
    authorization_argument = args.authorization_output.absolute()
    if authorization_argument.exists() or authorization_argument.is_symlink():
        raise RuntimeError("authorization output already exists")
    authorization_parent = authorization_argument.parent.resolve(strict=True)
    authorization_output = authorization_parent / authorization_argument.name
    protected = (
        package,
        workspace,
        output_parent,
        authorization_parent,
        source_audit_manifest_path.parent,
        runtime_receipt_path.parent,
        runtime_audit_manifest_path.parent,
    )
    if not _pairwise_disjoint(protected):
        raise RuntimeError("package, sources, output, authorization, and evidence roots must be pairwise disjoint")

    python_executable = args.python_executable.resolve(strict=True)
    runtime_backend = runtime_receipt.get("backend", {})
    if runtime_backend.get("python_executable_resolved") != os.fspath(python_executable):
        raise RuntimeError("requested Python executable differs from calibrated executable")
    if runtime_backend.get("cuda_visible_devices") != "0":
        raise RuntimeError("calibrated CUDA visibility is not exactly device zero")
    for key in (
        "python_version",
        "numpy_version",
        "cupy_version",
        "scipy_version",
        "device_name",
        "cuda_runtime",
    ):
        if key not in runtime_backend:
            raise RuntimeError(f"runtime tuple missing {key}")

    audit_binding = {
        "source_audit_status": SOURCE_AUDIT_STATUS,
        "source_audit_manifest_sha256": source_audit_manifest_sha,
        "source_audit_receipt_sha256": source_audit_sha,
        "source_audit_receipt_internal_sha256": source_audit_internal,
        "runtime_receipt_sha256": runtime_receipt_sha,
        "runtime_receipt_internal_sha256": internal_sha,
        "runtime_audit_status": RUNTIME_AUDIT_STATUS,
        "runtime_audit_manifest_sha256": runtime_audit_manifest_sha,
        "runtime_audit_receipt_sha256": runtime_audit_sha,
        "runtime_audit_receipt_internal_sha256": runtime_audit_internal,
    }
    artifact_binding = {
        "artifact_manifest_sha256": artifact_sha,
        "source_only_receipt_sha256": artifact_rows["source_only_receipt.json"],
        "runner_sha256": artifact_rows["free_order_oracle_v2.py"],
        "source_bindings_sha256": oracle.BINDINGS_SHA256,
        "runtime_calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
    }
    path_binding = {
        "workspace_root": os.fspath(workspace),
        "output": os.fspath(output),
        "authorization_parent": os.fspath(authorization_parent),
    }
    runtime_binding = dict(runtime_backend)
    audit_paths = {
        "source_audit_manifest": os.fspath(source_audit_manifest_path),
        "source_audit_receipt": os.fspath(source_audit_receipt_path),
        "runtime_receipt": os.fspath(runtime_receipt_path),
        "runtime_audit_manifest": os.fspath(runtime_audit_manifest_path),
        "runtime_audit_receipt": os.fspath(runtime_audit_receipt_path),
    }
    run_material = {
        "artifact_binding": artifact_binding,
        "audit_binding": audit_binding,
        "audit_paths": audit_paths,
        "path_binding": path_binding,
        "runtime_binding": runtime_binding,
    }
    authorization: dict[str, Any] = {
        "schema": oracle.AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED_ONE_SHOT_AUXILIARY_SOURCE_RUN",
        "one_shot": True,
        "scope_literal": SCOPE_LITERAL,
        "pinned_panel_authorized": False,
        "run_id": oracle.canonical_sha256(run_material),
        **run_material,
        "builder": {
            "script_sha256": artifact_rows["create_authorization.py"],
            "python_executable_resolved": os.fspath(Path(sys.executable).resolve(strict=True)),
            "python_version": platform.python_version(),
        },
        "claim_boundary": "Authorizes exactly one discovery-only auxiliary source-oracle result path; no pinned or finite-codec run.",
    }
    authorization["canonical_unsigned_sha256"] = oracle.canonical_sha256(authorization)
    oracle._write_create_new(authorization_output, authorization)
    print(
        json.dumps(
            {
                "authorization": os.fspath(authorization_output),
                "authorization_sha256": hashlib.sha256(_regular_bytes(authorization_output)).hexdigest(),
                "run_id": authorization["run_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
