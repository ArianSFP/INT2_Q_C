"""One-use, shell-free capture for the sole authorized r3 source-free preflight."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


FRESH_PARENT = Path(
    "/tmp/codex_cbib1_r3_source_free_rtx5090_preflight_20260903_5bac3594"
)
PACKAGE_ROOT = FRESH_PARENT / "package"
REVIEW_ROOT = FRESH_PARENT / "review"
CLAIM_PATH = FRESH_PARENT / "ONE_USE_PREFLIGHT_CLAIM.json"
RECEIPT_PATH = FRESH_PARENT / "receipt.json"
CHILD_STDERR_PATH = FRESH_PARENT / "child_stderr.txt"
WRAPPER_STATUS_PATH = FRESH_PARENT / "wrapper_status.json"
PYTHON_EXECUTABLE = "/workspace/int2-cupy-venv/bin/python"
PYTHON_REALPATH = "/usr/bin/python3.12"
PYTHON_SHA256 = "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5"
PYTHON_VERSION = (3, 12, 3)
DEPLOYMENT_MANIFEST_SHA256 = (
    "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
)
DEPLOYMENT_ROOT_SHA256 = (
    "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"
)
RUNNER_SHA256 = "111a176bbba4165f82bc5380feae8532b6fee069a25770d8a054f2d0b4391ca6"
AUTHORIZATION = "RUN_SOURCE_FREE_CBIB1_QWEN_DEPLOYMENT_PARITY_V0_R3"


def _need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_write(path: Path, data: bytes) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _authenticate_package() -> None:
    _need(PACKAGE_ROOT.is_dir() and not PACKAGE_ROOT.is_symlink(), "package root")
    manifest_path = PACKAGE_ROOT / "SOURCE_MANIFEST.json"
    _need(_sha(manifest_path) == DEPLOYMENT_MANIFEST_SHA256, "deployment manifest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    _need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
          "canonical deployment manifest")
    _need(manifest.get("schema") ==
          "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3", "schema")
    rows = manifest.get("files")
    _need(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    _need(names == sorted(names) and len(names) == len(set(names)), "member order")
    _need(sorted(path.name for path in PACKAGE_ROOT.iterdir()) ==
          sorted(names + ["SOURCE_MANIFEST.json"]), "package closure")
    normalized = []
    for row in rows:
        _need(set(row) == {"bytes", "name", "sha256"}, "manifest row")
        member = PACKAGE_ROOT / row["name"]
        _need(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink(),
              "regular nonsymlink member")
        _need(member.stat().st_size == int(row["bytes"]) and
              _sha(member) == row["sha256"], f"member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    root = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _need(root == manifest.get("source_root_sha256") == DEPLOYMENT_ROOT_SHA256,
          "deployment source root")
    _need(_sha(PACKAGE_ROOT / "run_source_free_cupy.py") == RUNNER_SHA256,
          "source-free runner")


def _validate_receipt(raw: bytes) -> dict:
    lines = raw.splitlines()
    _need(len(lines) == 1 and bool(lines[0]), "exactly one JSON receipt line")
    result = json.loads(lines[0])
    _need(result.get("schema") ==
          "same-layer-clustered-ib-qwen-deployment-source-free-cupy-v0-r3" and
          result.get("status") == "PASS_PRODUCTION_GEOMETRY_FULL_CPU_CUPY_PARITY" and
          result.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST_SHA256 and
          result.get("payload_or_qwen_accessed") is False and
          result.get("device_name") == "NVIDIA GeForce RTX 5090" and
          result.get("cupy_version") == "14.2.0" and
          result.get("numpy_version") == "2.5.2" and
          result.get("cuda_runtime") == 12090 and
          result.get("cuda_driver") == 13000 and
          result.get("all_controls_executed") is True and
          result.get("control_count") == 8 and
          2 in result.get("source_read_survivor_group_sizes", []) and
          "5/2" in result.get("source_read_survivor_endpoints", {}).get("2", []),
          "preflight receipt acceptance")
    return result


def main() -> int:
    _need(Path.cwd() == FRESH_PARENT, "fixed fresh-parent cwd")
    _need(Path(__file__).resolve(strict=True).parent == REVIEW_ROOT, "fixed review root")
    _need(FRESH_PARENT.is_dir() and not FRESH_PARENT.is_symlink(), "fresh parent")

    # This is the first mutation and permanently consumes the sole attempt.
    claim = (json.dumps({
        "authorization_id": "CBIB1_R3_SOURCE_FREE_RTX5090_PREFLIGHT_ONCE_20260903",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
        "schema": "same-layer-clustered-ib-r3-source-free-preflight-claim-v0",
        "status": "ATTEMPT_CONSUMED_BEFORE_VALIDATION_AND_CHILD_SPAWN",
    }, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _exclusive_write(CLAIM_PATH, claim)

    try:
        _need(sys.executable == PYTHON_EXECUTABLE, "Python invocation path")
        executable = Path(sys.executable).resolve(strict=True)
        _need(str(executable) == PYTHON_REALPATH and _sha(executable) == PYTHON_SHA256,
              "Python identity")
        _need(tuple(sys.version_info[:3]) == PYTHON_VERSION, "Python version")
        _authenticate_package()
        argv = [
            PYTHON_EXECUTABLE, "-I", "-B",
            str(PACKAGE_ROOT / "run_source_free_cupy.py"),
            "--authorization", AUTHORIZATION,
            "--deployment-manifest-sha256", DEPLOYMENT_MANIFEST_SHA256,
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        receipt_fd = os.open(
            str(RECEIPT_PATH), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        stderr_fd = os.open(
            str(CHILD_STDERR_PATH), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(receipt_fd, "wb", closefd=True) as receipt_handle, \
                os.fdopen(stderr_fd, "wb", closefd=True) as stderr_handle:
            completed = subprocess.run(
                argv, cwd=PACKAGE_ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=receipt_handle, stderr=stderr_handle, shell=False, check=False,
            )
            receipt_handle.flush()
            os.fsync(receipt_handle.fileno())
            stderr_handle.flush()
            os.fsync(stderr_handle.fileno())
        _need(completed.returncode == 0, f"preflight exit {completed.returncode}")
        result = _validate_receipt(RECEIPT_PATH.read_bytes())
        status = {
            "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
            "receipt_sha256": _sha(RECEIPT_PATH),
            "schema": "same-layer-clustered-ib-r3-source-free-preflight-wrapper-v0",
            "status": "PASS_SINGLE_ATTEMPT_CAPTURED_AND_VALIDATED",
            "underlying_status": result["status"],
        }
    except BaseException as exc:
        status = {
            "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
            "error_type": type(exc).__name__,
            "schema": "same-layer-clustered-ib-r3-source-free-preflight-wrapper-v0",
            "status": "FAIL_ATTEMPT_CONSUMED_NO_RETRY_AUTHORIZED",
        }
        _exclusive_write(
            WRAPPER_STATUS_PATH,
            (json.dumps(status, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        raise
    _exclusive_write(
        WRAPPER_STATUS_PATH,
        (json.dumps(status, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    print(json.dumps(status, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
