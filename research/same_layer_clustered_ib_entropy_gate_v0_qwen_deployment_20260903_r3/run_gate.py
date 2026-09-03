"""One-use, hard-pinned RunPod launcher for the CBIB-1 Qwen aperture."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import sys
from datetime import datetime, timezone


PAYLOAD_EXECUTION_ENABLED = True
AUTHORIZATION_PHRASE = "EXECUTE_AUTHENTICATED_QWEN_L15_CBIB1_V0_R3_ONCE"
PAYLOAD_ROOT = "/workspace/INT2__compression"
OUTPUT_PARENT = "/tmp/codex_cbib1_qwen_l15_oneuse_20260903_r3"
OUTPUT_NAME = "result.json"
CLAIM_NAME = "ONE_USE_CLAIM.json"
EXPECTED_HOSTNAME = "5d4226946659"
PYTHON_EXECUTABLE = "/workspace/int2-cupy-venv/bin/python"
PYTHON_REALPATH = "/usr/bin/python3.12"
PYTHON_SHA256 = "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5"
PYTHON_VERSION = (3, 12, 3)
CUPY_VERSION = "14.2.0"
CUPY_FILE = "/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy/__init__.py"
CUPY_FILE_SHA256 = "8c4724758587dea5f1c1d7c217c74a9fa0e4ed7f9d76a2b86fa001117cf3c718"
NUMPY_VERSION = "2.5.2"
NUMPY_FILE = "/workspace/int2-cupy-venv/lib/python3.12/site-packages/numpy/__init__.py"
NUMPY_FILE_SHA256 = "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11"
NUMPY_SITE_ROOT = "/workspace/int2-cupy-venv/lib/python3.12/site-packages"
NUMPY_RECORD = (
    "/workspace/int2-cupy-venv/lib/python3.12/site-packages/"
    "numpy-2.5.2.dist-info/RECORD"
)
NUMPY_RECORD_SHA256 = "662e57f69a042c5b9efa7a46e8c2901f5d733e21b255a34d7f061e466bceab0d"
CUDA_RUNTIME_VERSION = 12090
CUDA_DRIVER_VERSION = 13000
DEVICE_NAME = "NVIDIA GeForce RTX 5090"
PANEL_LOCK_SHA256 = "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5"
CORE_SHA256 = "25e84b9d5e598a72984e48cb5593c41725d096e36082b20b3d47a78f2100e340"
WORKER_SHA256 = "a34ca17dd8f76afa0331bb56d5b5dec26dcde693d05755ea2ca342a76a6badfc"
PARENT_SOURCE_MANIFEST_SHA256 = "1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a"
PARENT_SOURCE_ROOT_SHA256 = "18a4043e99b17cfa535f4a6c2930f2c1ac42eff092f4e5d61b9408b1986f457e"
PARENT_AUDIT_MANIFEST_SHA256 = "5c07e720928f2642867524b201d0abef5a17ea57b4cae68f5c0df59010e3f051"
PARENT_AUDIT_ROOT_SHA256 = "2d0b25666b2dc20feef8dfa56fd62c377b7ba7e1c66e34c3844fb5d1b02b45ca"


def _hold(reason: str) -> int:
    print(json.dumps({
        "schema": "same-layer-clustered-ib-deployment-hold-v0-r3",
        "status": "HOLD_NO_PAYLOAD_ACCESS", "reason": reason,
        "payload_execution_enabled": PAYLOAD_EXECUTION_ENABLED,
    }, sort_keys=True, separators=(",", ":")))
    return 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest(package: Path, expected_manifest_sha256: str) -> dict:
    manifest_path = package / "SOURCE_MANIFEST.json"
    if _sha(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("external deployment-manifest digest mismatch")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != canonical:
        raise RuntimeError("noncanonical deployment manifest")
    if manifest.get("schema") != "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3":
        raise RuntimeError("deployment manifest schema")
    if manifest.get("parent_source_manifest_sha256") != PARENT_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("parent source manifest pin")
    if manifest.get("parent_source_root_sha256") != PARENT_SOURCE_ROOT_SHA256:
        raise RuntimeError("parent source root pin")
    if manifest.get("parent_audit_manifest_sha256") != PARENT_AUDIT_MANIFEST_SHA256:
        raise RuntimeError("parent audit manifest pin")
    if manifest.get("parent_audit_root_sha256") != PARENT_AUDIT_ROOT_SHA256:
        raise RuntimeError("parent audit root pin")
    rows = manifest.get("files")
    names = [row.get("name") for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("deployment manifest member order")
    actual = sorted(path.name for path in package.iterdir())
    if actual != sorted(names + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("deployment package closure")
    normalized = []
    for row in rows:
        path = package / row["name"]
        if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
            raise RuntimeError("nonregular deployment member")
        if path.stat().st_size != int(row["bytes"]) or _sha(path) != row["sha256"]:
            raise RuntimeError(f"deployment member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    root = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if root != manifest.get("source_root_sha256"):
        raise RuntimeError("deployment source-root mismatch")
    return manifest


def _load_verified_module(name: str, path: Path, expected_sha256: str):
    if _sha(path) != expected_sha256:
        raise RuntimeError(f"source hash mismatch: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_numpy_record_closure(np) -> dict:
    """Authenticate every wheel-recorded NumPy source/native member.

    The pinned RECORD transitively binds the NumPy Python modules, extension
    modules, and bundled numpy.libs/OpenBLAS objects used by dot/log2 and dtype
    operations.  This is performed before the one-use claim.
    """
    numpy_path = Path(np.__file__).resolve(strict=True)
    if np.__version__ != NUMPY_VERSION or str(numpy_path) != NUMPY_FILE:
        raise RuntimeError("NumPy identity mismatch")
    if _sha(numpy_path) != NUMPY_FILE_SHA256:
        raise RuntimeError("NumPy entry-file hash mismatch")
    site_root = Path(NUMPY_SITE_ROOT).resolve(strict=True)
    record_path = Path(NUMPY_RECORD).resolve(strict=True)
    if not stat.S_ISREG(record_path.lstat().st_mode) or record_path.is_symlink():
        raise RuntimeError("NumPy RECORD must be a regular non-link")
    record_bytes = record_path.read_bytes()
    if hashlib.sha256(record_bytes).hexdigest() != NUMPY_RECORD_SHA256:
        raise RuntimeError("NumPy RECORD digest mismatch")
    rows_checked = 0
    native_checked = 0
    unhashed = []
    for row in csv.reader(record_bytes.decode("utf-8").splitlines()):
        if len(row) != 3:
            raise RuntimeError("malformed NumPy RECORD row")
        relative, encoded_digest, encoded_size = row
        if not relative.startswith(("numpy/", "numpy.libs/", "numpy-2.5.2.dist-info/")):
            continue
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("unsafe NumPy RECORD path")
        candidate = site_root.joinpath(rel)
        resolved = candidate.resolve(strict=True)
        if site_root not in resolved.parents:
            raise RuntimeError("NumPy RECORD path escapes site root")
        if not stat.S_ISREG(candidate.lstat().st_mode) or candidate.is_symlink():
            raise RuntimeError("NumPy closure member must be a regular non-link")
        if not encoded_digest:
            unhashed.append(relative)
            continue
        algorithm, payload = encoded_digest.split("=", 1)
        if algorithm != "sha256" or not encoded_size:
            raise RuntimeError("NumPy RECORD digest/size algorithm")
        data = candidate.read_bytes()
        if len(data) != int(encoded_size):
            raise RuntimeError(f"NumPy closure size mismatch: {relative}")
        expected = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        if hashlib.sha256(data).digest() != expected:
            raise RuntimeError(f"NumPy closure digest mismatch: {relative}")
        rows_checked += 1
        if candidate.suffix in {".so", ".a"} or ".so." in candidate.name:
            native_checked += 1
    if unhashed != ["numpy-2.5.2.dist-info/RECORD"]:
        raise RuntimeError("unexpected unhashed NumPy closure member")
    if rows_checked < 500 or native_checked < 10:
        raise RuntimeError("insufficient NumPy source/native closure")
    return {
        "version": NUMPY_VERSION,
        "entry_file": NUMPY_FILE,
        "entry_file_sha256": NUMPY_FILE_SHA256,
        "record_file": NUMPY_RECORD,
        "record_sha256": NUMPY_RECORD_SHA256,
        "record_hashed_members_verified": rows_checked,
        "native_members_verified": native_checked,
        "unhashed_record_self_only": True,
        "verified_before_one_use_claim": True,
    }


def _validate_runtime():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if platform.node() != EXPECTED_HOSTNAME:
        raise RuntimeError("RunPod hostname mismatch")
    if sys.executable != PYTHON_EXECUTABLE:
        raise RuntimeError("Python executable invocation path mismatch")
    executable = Path(sys.executable).resolve(strict=True)
    if str(executable) != PYTHON_REALPATH or _sha(executable) != PYTHON_SHA256:
        raise RuntimeError("Python executable identity mismatch")
    if tuple(sys.version_info[:3]) != PYTHON_VERSION:
        raise RuntimeError("Python version mismatch")
    import numpy as np
    numpy_receipt = _verify_numpy_record_closure(np)
    import cupy as cp
    cupy_path = Path(cp.__file__).resolve(strict=True)
    if cp.__version__ != CUPY_VERSION or str(cupy_path) != CUPY_FILE:
        raise RuntimeError("CuPy identity mismatch")
    if _sha(cupy_path) != CUPY_FILE_SHA256:
        raise RuntimeError("CuPy source hash mismatch")
    if int(cp.cuda.runtime.runtimeGetVersion()) != CUDA_RUNTIME_VERSION:
        raise RuntimeError("CUDA runtime mismatch")
    if int(cp.cuda.runtime.driverGetVersion()) != CUDA_DRIVER_VERSION:
        raise RuntimeError("CUDA driver mismatch")
    if int(cp.cuda.runtime.getDevice()) != 0:
        raise RuntimeError("CUDA device ordinal mismatch")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="strict")
    if str(name) != DEVICE_NAME:
        raise RuntimeError("CUDA device name mismatch")
    return cp, numpy_receipt


def _claim_once(claim_path: Path, deployment_manifest_sha256: str) -> None:
    payload = (json.dumps({
        "schema": "same-layer-clustered-ib-one-use-claim-v0-r3",
        "status": "CONSUMED_BEFORE_PAYLOAD_ACCESS",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "parent_source_manifest_sha256": PARENT_SOURCE_MANIFEST_SHA256,
        "parent_audit_manifest_sha256": PARENT_AUDIT_MANIFEST_SHA256,
    }, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(str(claim_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A created claim is deliberately never removed: any invocation attempt
        # consumes the authority even when later payload processing fails.
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", default="HOLD")
    parser.add_argument("--deployment-manifest-sha256", default="")
    args = parser.parse_args(argv)
    # Wrong authorization returns before any Path construction, runtime import,
    # package read, output mutation, or payload access.
    if not PAYLOAD_EXECUTION_ENABLED:
        return _hold("compile_time_payload_switch_is_false")
    if args.authorization != AUTHORIZATION_PHRASE:
        return _hold("authorization_phrase_mismatch")
    if len(args.deployment_manifest_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.deployment_manifest_sha256
    ):
        raise RuntimeError("external lowercase deployment-manifest SHA-256 required")

    package = Path(__file__).resolve().parent
    _verify_manifest(package, args.deployment_manifest_sha256)
    if _sha(package / "clustered_ib_core.py") != CORE_SHA256:
        raise RuntimeError("frozen CBIB core mismatch")
    if _sha(package / "cupy_worker.py") != WORKER_SHA256:
        raise RuntimeError("CuPy worker mismatch")
    panel_path = package / "panel_lock.json"
    if _sha(panel_path) != PANEL_LOCK_SHA256:
        raise RuntimeError("panel-lock mismatch")
    _cp, numpy_receipt = _validate_runtime()

    payload_root = Path(PAYLOAD_ROOT)
    output_parent = Path(OUTPUT_PARENT)
    if not payload_root.is_absolute() or not payload_root.is_dir() or payload_root.is_symlink():
        raise RuntimeError("hard-pinned payload root unavailable or unsafe")
    if (not output_parent.is_absolute() or not output_parent.is_dir()
            or output_parent.is_symlink()):
        raise RuntimeError("hard-pinned output parent unavailable or unsafe")
    output = output_parent / OUTPUT_NAME
    claim = output_parent / CLAIM_NAME
    if output.exists() or output.is_symlink():
        raise RuntimeError("hard-pinned output must be absent")
    if claim.exists() or claim.is_symlink():
        raise RuntimeError("one-use authority already consumed")
    _claim_once(claim, args.deployment_manifest_sha256)

    _load_verified_module("clustered_ib_core", package / "clustered_ib_core.py", CORE_SHA256)
    worker = _load_verified_module(
        "same_layer_clustered_ib_qwen_cupy_worker_v0_r3",
        package / "cupy_worker.py", WORKER_SHA256,
    )
    result = worker.run_authorized_panel(panel_path, payload_root)
    result["runtime_numpy_closure"] = numpy_receipt
    serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    print(json.dumps({
        "schema": "same-layer-clustered-ib-one-use-launch-receipt-v0-r3",
        "status": result["status"], "output": str(output),
        "result_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "claim": str(claim),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
