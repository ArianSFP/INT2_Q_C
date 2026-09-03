"""Windows/RTX3060 runtime bridge for the frozen CBIB-1 r3 science.

Only the deployment/runtime boundary is adapted.  The bridge authenticates and
loads the original r3 ``run_gate.py``, replaces its Linux runtime validator with
the exact local Windows validator below, and leaves the frozen core, CuPy
worker, panel, thresholds, controls, and result construction untouched.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys


AUTHORIZATION = "EXECUTE_CBIB1_R3_LOCAL3060_BRIDGE_ONCE_458A424A"
DEPLOYMENT_ROOT = Path(
    r"C:\INT2__compression\INT2_Q_C\research\same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3"
)
DEPLOYMENT_MANIFEST = "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
DEPLOYMENT_SOURCE_ROOT = "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"
RUN_GATE_SHA256 = "531aa3c07710231311a86d460c88e2f2112e2d286a8f49da21eab2bab09fd624"
CORE_SHA256 = "25e84b9d5e598a72984e48cb5593c41725d096e36082b20b3d47a78f2100e340"
WORKER_SHA256 = "a34ca17dd8f76afa0331bb56d5b5dec26dcde693d05755ea2ca342a76a6badfc"
PANEL_SHA256 = "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5"
PAYLOAD_ROOT = Path(r"C:\INT2__compression")
RUN_ROOT = Path(r"C:\INT2__compression\tmp\cbib1_r3_local3060_qwen_once_20260903_458a424a")
ATTEMPT_CLAIM = Path(
    r"C:\INT2__compression\tmp\CBIB1_R3_LOCAL3060_AUTHORITY_ATTEMPT_20260903_458A424A.json"
)
RUNTIME_LOCK = Path(__file__).resolve().parent / "LOCAL_RUNTIME_LOCK.json"
PYTHON = Path(r"C:\INT2__compression\.venv-cupy\Scripts\python.exe")
SITE_ROOT = Path(r"C:\INT2__compression\.venv-cupy\Lib\site-packages")
CUPY_INIT = SITE_ROOT / "cupy" / "__init__.py"
NUMPY_INIT = SITE_ROOT / "numpy" / "__init__.py"
CUDA_PATH = SITE_ROOT / "nvidia" / "cuda_runtime"
CACHE_ROOT = Path(r"C:\INT2__compression\.cupy_cache\cbib1_r3_local3060_20260903_458a424a")
NVIDIA_SMI = Path(r"C:\Windows\System32\nvidia-smi.exe")
EXPECTED_HOSTNAME = "DESKTOP-4UMQUSL"
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 3060"
EXPECTED_DEVICE_UUID = "GPU-458a424a-76e3-65e5-0470-803e0ed131ca"
EXPECTED_DRIVER_TEXT = "560.94"
EXPECTED_RUNTIME_API = 12090
EXPECTED_DRIVER_API = 12060
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
_DLL_HANDLES: list[object] = []


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def real_file(path: Path) -> None:
    need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
         f"regular nonsymlink required: {path}")


def real_directory(path: Path) -> None:
    need(path.is_dir() and not path.is_symlink(), f"real directory required: {path}")
    need(path.resolve(strict=True) == path.absolute(), f"directory indirection forbidden: {path}")


def load_runtime_lock() -> dict:
    real_file(RUNTIME_LOCK)
    raw = RUNTIME_LOCK.read_bytes()
    lock = json.loads(raw)
    need(raw == (json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "noncanonical runtime lock")
    need(lock.get("schema") == "cbib1-r3-local-rtx3060-runtime-lock-v0-r1",
         "runtime lock schema")
    return lock


def authenticate_deployment() -> dict:
    real_directory(DEPLOYMENT_ROOT)
    manifest_path = DEPLOYMENT_ROOT / "SOURCE_MANIFEST.json"
    real_file(manifest_path)
    need(sha256_file(manifest_path) == DEPLOYMENT_MANIFEST, "deployment manifest digest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "noncanonical deployment manifest")
    need(manifest.get("schema") == "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3",
         "deployment schema")
    rows = manifest.get("files")
    need(isinstance(rows, list) and len(rows) == 15, "deployment member count")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "deployment member order")
    need(sorted(path.name for path in DEPLOYMENT_ROOT.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "deployment closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "deployment manifest row")
        member = DEPLOYMENT_ROOT / row["name"]
        real_file(member)
        need(member.stat().st_size == int(row["bytes"]) and
             sha256_file(member) == row["sha256"], f"deployment member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(observed == DEPLOYMENT_SOURCE_ROOT == manifest.get("source_root_sha256"),
         "deployment source root")
    need(sha256_file(DEPLOYMENT_ROOT / "run_gate.py") == RUN_GATE_SHA256 and
         sha256_file(DEPLOYMENT_ROOT / "clustered_ib_core.py") == CORE_SHA256 and
         sha256_file(DEPLOYMENT_ROOT / "cupy_worker.py") == WORKER_SHA256 and
         sha256_file(DEPLOYMENT_ROOT / "panel_lock.json") == PANEL_SHA256,
         "frozen science hashes")
    return manifest


def static_runtime_preflight(lock: dict) -> None:
    """Verify pinned runtime identity without importing NumPy/CuPy or touching a GPU."""
    need(platform.node() == EXPECTED_HOSTNAME, "hostname mismatch")
    real_file(PYTHON)
    need(sha256_file(PYTHON) == lock["python"]["sha256"], "Python executable digest")
    real_file(NUMPY_INIT)
    need(sha256_file(NUMPY_INIT) == lock["numpy"]["init_sha256"], "NumPy init digest")
    real_file(CUPY_INIT)
    need(sha256_file(CUPY_INIT) == lock["cupy"]["init_sha256"], "CuPy init digest")
    real_file(NVIDIA_SMI)
    need(sha256_file(NVIDIA_SMI) == lock["nvidia_smi"]["sha256"], "nvidia-smi digest")
    compatibility = Path(lock["process_local_dll"]["compatibility_file"])
    real_file(compatibility)
    need(compatibility.stat().st_size == lock["process_local_dll"]["bytes"] and
         sha256_file(compatibility) == lock["process_local_dll"]["sha256"],
         "process-local MSVCP140 identity")
    for row in lock["wheel_records"]:
        record = Path(row["path"])
        real_file(record)
        need(record.stat().st_size == row["bytes"] and sha256_file(record) == row["sha256"],
             f"wheel RECORD identity: {row['distribution']}")
    expected_dlls = [Path(value) for value in lock["process_local_dll"]["directories"]]
    need(expected_dlls[0] == compatibility.parent, "compatibility DLL directory ordering")
    for directory in expected_dlls:
        real_directory(directory)
    need(Path(lock["cuda_path"]) == CUDA_PATH and Path(lock["cupy_cache_dir"]) == CACHE_ROOT,
         "fixed CUDA/cache paths")


def verify_wheel_record(row: dict) -> dict:
    record = Path(row["path"])
    raw = record.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == row["sha256"] and len(raw) == row["bytes"],
         f"wheel RECORD changed: {row['distribution']}")
    prefixes = tuple(row["member_prefixes"])
    checked = 0
    native = 0
    unhashed: list[str] = []
    for values in csv.reader(raw.decode("utf-8").splitlines()):
        need(len(values) == 3, f"malformed RECORD row: {row['distribution']}")
        relative, encoded_digest, encoded_size = values
        if not relative.startswith(prefixes):
            continue
        posix = PurePosixPath(relative)
        need(not posix.is_absolute() and ".." not in posix.parts, "unsafe RECORD member")
        candidate = SITE_ROOT.joinpath(*posix.parts)
        real_file(candidate)
        need(SITE_ROOT.resolve(strict=True) in candidate.resolve(strict=True).parents,
             "RECORD member escapes site root")
        if not encoded_digest:
            unhashed.append(relative)
            continue
        algorithm, payload = encoded_digest.split("=", 1)
        need(algorithm == "sha256" and encoded_size != "", "RECORD digest syntax")
        need(candidate.stat().st_size == int(encoded_size), "RECORD member size")
        expected = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).hex()
        need(sha256_file(candidate) == expected, f"RECORD member digest: {relative}")
        checked += 1
        if candidate.suffix.lower() in {".pyd", ".dll"}:
            native += 1
    need(unhashed == [row["record_relative_path"]], "unexpected unhashed RECORD members")
    need(checked >= int(row["minimum_hashed_members"]) and
         native >= int(row["minimum_native_members"]), "insufficient runtime closure")
    return {"distribution": row["distribution"], "hashed_members_verified": checked,
            "native_members_verified": native, "record_sha256": row["sha256"]}


def canonical_uuid(raw) -> str:
    if isinstance(raw, str):
        text = raw.removeprefix("GPU-").replace("-", "").lower()
        need(re.fullmatch(r"[0-9a-f]{32}", text) is not None, "CUDA UUID syntax")
    else:
        packed = bytes(raw)
        need(len(packed) == 16, "CUDA UUID bytes")
        text = packed.hex()
    return "GPU-" + "-".join((text[:8], text[8:12], text[12:16], text[16:20], text[20:]))


def validate_local_runtime():
    """Exact replacement for the Linux-only r3 runtime validator."""
    lock = load_runtime_lock()
    static_runtime_preflight(lock)
    need(sys.executable == str(PYTHON) and tuple(sys.version_info[:3]) == (3, 12, 14),
         "Python invocation/version mismatch")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES mismatch")
    need(os.environ.get("CUDA_PATH") == str(CUDA_PATH), "CUDA_PATH mismatch")
    need(os.environ.get("CUPY_CACHE_DIR") == str(CACHE_ROOT), "CUPY_CACHE_DIR mismatch")
    real_directory(CACHE_ROOT)
    need(not any(CACHE_ROOT.iterdir()), "fresh CuPy cache must initially be empty")

    closures = [verify_wheel_record(row) for row in lock["wheel_records"]]
    global _DLL_HANDLES
    for value in lock["process_local_dll"]["directories"]:
        _DLL_HANDLES.append(os.add_dll_directory(value))

    smi = subprocess.run(
        [str(NVIDIA_SMI), "--query-gpu=index,name,uuid,driver_version",
         "--format=csv,noheader,nounits"], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, check=False,
        text=True, encoding="utf-8",
    )
    need(smi.returncode == 0 and smi.stderr == "", "nvidia-smi identity query")
    rows = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    need(rows == [f"0, {EXPECTED_DEVICE_NAME}, {EXPECTED_DEVICE_UUID}, {EXPECTED_DRIVER_TEXT}"],
         "nvidia-smi GPU identity")

    import numpy as np
    import cupy as cp
    need(np.__version__ == "2.5.2" and Path(np.__file__).resolve(strict=True) == NUMPY_INIT and
         sha256_file(NUMPY_INIT) == lock["numpy"]["init_sha256"], "NumPy runtime identity")
    need(cp.__version__ == "14.2.0" and Path(cp.__file__).resolve(strict=True) == CUPY_INIT and
         sha256_file(CUPY_INIT) == lock["cupy"]["init_sha256"], "CuPy runtime identity")
    runtime = cp.cuda.runtime
    need(int(runtime.runtimeGetVersion()) == EXPECTED_RUNTIME_API and
         int(runtime.driverGetVersion()) == EXPECTED_DRIVER_API and
         int(runtime.getDevice()) == 0, "CUDA API identity")
    props = runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="strict")
    raw_uuid = runtime.deviceGetUuid(0) if callable(getattr(runtime, "deviceGetUuid", None)) \
        else props.get("uuid", props.get(b"uuid"))
    need(str(name) == EXPECTED_DEVICE_NAME and canonical_uuid(raw_uuid) == EXPECTED_DEVICE_UUID and
         (int(props["major"]), int(props["minor"])) == EXPECTED_COMPUTE_CAPABILITY,
         "exact RTX3060 identity")
    numpy_row = next(item for item in closures if item["distribution"] == "numpy-2.5.2")
    return cp, {
        "version": "2.5.2",
        "entry_file": str(NUMPY_INIT),
        "entry_file_sha256": lock["numpy"]["init_sha256"],
        "record_file": next(row["path"] for row in lock["wheel_records"]
                            if row["distribution"] == "numpy-2.5.2"),
        "record_sha256": numpy_row["record_sha256"],
        "record_hashed_members_verified": numpy_row["hashed_members_verified"],
        "native_members_verified": numpy_row["native_members_verified"],
        "unhashed_record_self_only": True,
        "verified_before_one_use_claim": True,
        "local_runtime_all_wheel_closures": closures,
        "process_local_dll_directories": lock["process_local_dll"]["directories"],
        "fresh_cupy_cache": str(CACHE_ROOT),
        "device_uuid": EXPECTED_DEVICE_UUID,
        "driver_text": EXPECTED_DRIVER_TEXT,
    }


def load_frozen_run_gate():
    authenticate_deployment()
    path = DEPLOYMENT_ROOT / "run_gate.py"
    spec = importlib.util.spec_from_file_location("cbib1_r3_frozen_run_gate_local_bridge", path)
    need(spec is not None and spec.loader is not None, "cannot load frozen run_gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PAYLOAD_ROOT = str(PAYLOAD_ROOT)
    module.OUTPUT_PARENT = str(RUN_ROOT)
    module.EXPECTED_HOSTNAME = EXPECTED_HOSTNAME
    module.PYTHON_EXECUTABLE = str(PYTHON)
    module.PYTHON_REALPATH = str(PYTHON)
    module.PYTHON_SHA256 = sha256_file(PYTHON)
    module.PYTHON_VERSION = (3, 12, 14)
    module.NUMPY_FILE = str(NUMPY_INIT)
    module.NUMPY_FILE_SHA256 = sha256_file(NUMPY_INIT)
    module.CUPY_FILE = str(CUPY_INIT)
    module.CUPY_FILE_SHA256 = sha256_file(CUPY_INIT)
    module.CUDA_RUNTIME_VERSION = EXPECTED_RUNTIME_API
    module.CUDA_DRIVER_VERSION = EXPECTED_DRIVER_API
    module.DEVICE_NAME = EXPECTED_DEVICE_NAME
    module._validate_runtime = validate_local_runtime
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", default="HOLD")
    args = parser.parse_args(argv)
    if args.authorization != AUTHORIZATION:
        print(json.dumps({"schema": "cbib1-r3-local3060-bridge-hold-v0-r1",
                          "status": "HOLD_NO_PATH_RUNTIME_OR_PAYLOAD_ACCESS"},
                         sort_keys=True, separators=(",", ":")))
        return 2
    real_file(ATTEMPT_CLAIM)
    claim = json.loads(ATTEMPT_CLAIM.read_bytes())
    need(claim.get("schema") == "cbib1-r3-local3060-authority-attempt-claim-v0-r1" and
         claim.get("status") == "ATTEMPT_CONSUMED_BEFORE_VALIDATION_OR_PAYLOAD_ACCESS" and
         claim.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST,
         "outer one-use claim")
    need(Path.cwd() == RUN_ROOT and RUN_ROOT.is_dir() and not RUN_ROOT.is_symlink(),
         "fixed local run root")
    frozen = load_frozen_run_gate()
    return int(frozen.main([
        "--authorization", "EXECUTE_AUTHENTICATED_QWEN_L15_CBIB1_V0_R3_ONCE",
        "--deployment-manifest-sha256", DEPLOYMENT_MANIFEST,
    ]))


if __name__ == "__main__":
    raise SystemExit(main())
