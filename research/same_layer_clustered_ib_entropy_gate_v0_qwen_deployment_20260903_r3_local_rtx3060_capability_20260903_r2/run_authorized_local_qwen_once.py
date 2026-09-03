"""One-use shell-free local RTX3060 wrapper for frozen CBIB-1 r3 science."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess


AUTHORIZATION = "EXECUTE_FROZEN_CBIB1_R3_LOCAL3060_QWEN_ONCE_R2_09F4C6D1"
AUTHORITY_ROOT = Path(
    r"C:\INT2__compression\INT2_Q_C\research\same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3_local_rtx3060_capability_20260903_r2"
)
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
ATTEMPT_CLAIM = Path(
    r"C:\INT2__compression\tmp\CBIB1_R3_LOCAL3060_AUTHORITY_ATTEMPT_20260903_R2_09F4C6D1.json"
)
AUTHORITY_STATUS = Path(
    r"C:\INT2__compression\tmp\CBIB1_R3_LOCAL3060_AUTHORITY_STATUS_20260903_R2_09F4C6D1.json"
)
RUN_ROOT = Path(r"C:\INT2__compression\tmp\cbib1_r3_local3060_qwen_once_20260903_r2_09f4c6d1")
CACHE_ROOT = Path(r"C:\INT2__compression\tmp\cbib1_r3_local3060_cupy_cache_20260903_r2_09f4c6d1")
CHILD_STDOUT = RUN_ROOT / "child_stdout.jsonl"
CHILD_STDERR = RUN_ROOT / "child_stderr.txt"
CHILD_CLAIM = RUN_ROOT / "ONE_USE_CLAIM.json"
RESULT_PATH = RUN_ROOT / "result.json"
PYTHON = Path(r"C:\INT2__compression\.venv-cupy\Scripts\python.exe")
CUDA_PATH = Path(
    r"C:\INT2__compression\.venv-cupy\Lib\site-packages\nvidia\cuda_runtime"
)
BRIDGE = AUTHORITY_ROOT / "local_runtime_bridge.py"
RUNTIME_LOCK = AUTHORITY_ROOT / "LOCAL_RUNTIME_LOCK.json"


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


def exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def authenticate_package(root: Path, manifest_sha: str, source_root: str | None,
                         schema: str) -> dict:
    need(root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root.absolute(),
         "real fixed package root")
    manifest_path = root / "SOURCE_MANIFEST.json"
    real_file(manifest_path)
    need(sha256_file(manifest_path) == manifest_sha, "external manifest digest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "canonical manifest")
    need(manifest.get("schema") == schema, "manifest schema")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "manifest members")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "manifest member order")
    need(sorted(path.name for path in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "package closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "manifest row")
        member = root / row["name"]
        real_file(member)
        need(member.stat().st_size == int(row["bytes"]) and
             sha256_file(member) == row["sha256"], f"member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(observed == manifest.get("source_root_sha256") and
         (source_root is None or observed == source_root), "source-root digest")
    return manifest


def validate_prerequisites(capability_manifest_sha: str) -> None:
    capability = authenticate_package(
        AUTHORITY_ROOT, capability_manifest_sha, None,
        "same-layer-clustered-ib-r3-local-rtx3060-capability-manifest-v0-r2",
    )
    need(capability.get("status") == "AUTHORIZED_NOT_EXECUTED" and
         capability.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST and
         capability.get("panel_lock_sha256") == PANEL_SHA256,
         "capability semantics")
    deployment = authenticate_package(
        DEPLOYMENT_ROOT, DEPLOYMENT_MANIFEST, DEPLOYMENT_SOURCE_ROOT,
        "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3",
    )
    need(len(deployment["files"]) == 15 and
         sha256_file(DEPLOYMENT_ROOT / "run_gate.py") == RUN_GATE_SHA256 and
         sha256_file(DEPLOYMENT_ROOT / "clustered_ib_core.py") == CORE_SHA256 and
         sha256_file(DEPLOYMENT_ROOT / "cupy_worker.py") == WORKER_SHA256 and
         sha256_file(DEPLOYMENT_ROOT / "panel_lock.json") == PANEL_SHA256,
         "frozen r3 science")
    real_file(BRIDGE)
    spec = importlib.util.spec_from_file_location("cbib1_local3060_bridge_preflight", BRIDGE)
    need(spec is not None and spec.loader is not None, "bridge loader")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    lock = bridge.load_runtime_lock()
    bridge.static_runtime_preflight(lock)
    need(PAYLOAD_ROOT.is_dir() and not PAYLOAD_ROOT.is_symlink() and
         PAYLOAD_ROOT.resolve(strict=True) == PAYLOAD_ROOT, "fixed payload root")


def validate_result() -> tuple[dict, str]:
    real_file(CHILD_STDOUT)
    real_file(CHILD_STDERR)
    need(CHILD_STDERR.read_bytes() == b"", "child stderr")
    raw = CHILD_STDOUT.read_bytes()
    need(raw.endswith(b"\n") and len(raw.splitlines()) == 1, "sole child receipt")
    launch = json.loads(raw)
    need(set(launch) == {"claim", "output", "result_sha256", "schema", "status"} and
         launch["schema"] == "same-layer-clustered-ib-one-use-launch-receipt-v0-r3" and
         launch["claim"] == str(CHILD_CLAIM) and launch["output"] == str(RESULT_PATH),
         "child launch receipt")
    real_file(CHILD_CLAIM)
    child_claim = json.loads(CHILD_CLAIM.read_bytes())
    need(child_claim.get("status") == "CONSUMED_BEFORE_PAYLOAD_ACCESS" and
         child_claim.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST,
         "child one-use claim")
    real_file(RESULT_PATH)
    result_sha = sha256_file(RESULT_PATH)
    need(result_sha == launch["result_sha256"], "result digest binding")
    result = json.loads(RESULT_PATH.read_bytes())
    need(result.get("schema") == "same_layer_clustered_ib_entropy_gate_result_v0" and
         result.get("status") == launch["status"] and
         result.get("panel_lock_sha256") == PANEL_SHA256 and
         result.get("claim_boundary") ==
         "IDEAL_LABEL_ENTROPY_CENSUS_ONLY_NOT_A_FINITE_CODEC_OR_MSE_RESULT",
         "result identity")
    io = result.get("input_read_ledger", {})
    need(io.get("source_files_read_once") == 32 and io.get("source_bytes_read_once") == 100663296 and
         io.get("source_logical_host_scan_amplification") == 1.0 and
         len(io.get("authenticated_inputs", [])) == 32, "authenticated input ledger")
    cuda = result.get("cuda", {})
    need(cuda.get("cupy_version") == "14.2.0" and cuda.get("device_id") == 0 and
         cuda.get("device_name") == "NVIDIA GeForce RTX 3060" and
         cuda.get("runtime_version") == 12090 and cuda.get("driver_version") == 12060,
         "local CUDA result identity")
    numpy = result.get("runtime_numpy_closure", {})
    need(numpy.get("version") == "2.5.2" and numpy.get("verified_before_one_use_claim") is True and
         numpy.get("unhashed_record_self_only") is True and
         numpy.get("device_uuid") == "GPU-458a424a-76e3-65e5-0470-803e0ed131ca" and
         len(numpy.get("local_runtime_all_wheel_closures", [])) == 10,
         "local runtime closure result")
    return launch, result_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", default="HOLD")
    parser.add_argument("--capability-manifest-sha256", default="")
    args = parser.parse_args(argv)
    if args.authorization != AUTHORIZATION:
        print(json.dumps({"schema": "cbib1-r3-local3060-authority-hold-v0-r2",
                          "status": "HOLD_NO_PATH_RUNTIME_GPU_OR_PAYLOAD_ACCESS"},
                         sort_keys=True, separators=(",", ":")))
        return 2
    need(Path(__file__).resolve(strict=True).parent == AUTHORITY_ROOT,
         "fixed authority package path")
    need(Path.cwd() == AUTHORITY_ROOT, "fixed authority cwd")
    need(len(args.capability_manifest_sha256) == 64 and all(
        char in "0123456789abcdef" for char in args.capability_manifest_sha256
    ), "lowercase capability manifest SHA-256 required")
    need(ATTEMPT_CLAIM.parent.is_dir() and not ATTEMPT_CLAIM.parent.is_symlink(),
         "fixed claim parent")
    claim_raw = (json.dumps({
        "authorization_id": "CBIB1_R3_LOCAL3060_QWEN_ONCE_20260903_R2_09F4C6D1",
        "capability_manifest_sha256": args.capability_manifest_sha256,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST,
        "schema": "cbib1-r3-local3060-authority-attempt-claim-v0-r2",
        "status": "ATTEMPT_CONSUMED_BEFORE_VALIDATION_OR_PAYLOAD_ACCESS",
    }, sort_keys=True, separators=(",", ":")) + "\n").encode()
    exclusive_write(ATTEMPT_CLAIM, claim_raw)

    try:
        validate_prerequisites(args.capability_manifest_sha256)
        need(not RUN_ROOT.exists() and not RUN_ROOT.is_symlink(), "fresh run root must be absent")
        need(not CACHE_ROOT.exists() and not CACHE_ROOT.is_symlink(), "fresh cache must be absent")
        os.mkdir(RUN_ROOT)
        os.mkdir(CACHE_ROOT)
        stdout_fd = os.open(str(CHILD_STDOUT), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o600)
        stderr_fd = os.open(str(CHILD_STDERR), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o600)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["CUDA_PATH"] = str(CUDA_PATH)
        environment["CUPY_CACHE_DIR"] = str(CACHE_ROOT)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        child_argv = [str(PYTHON), "-I", "-B", str(BRIDGE), "--authorization",
                      "EXECUTE_CBIB1_R3_LOCAL3060_BRIDGE_ONCE_R2_09F4C6D1"]
        with os.fdopen(stdout_fd, "wb", closefd=True) as out, \
                os.fdopen(stderr_fd, "wb", closefd=True) as err:
            completed = subprocess.run(
                child_argv, cwd=RUN_ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, shell=False, check=False,
            )
            out.flush(); os.fsync(out.fileno())
            err.flush(); os.fsync(err.fileno())
        need(completed.returncode == 0, f"local bridge exit {completed.returncode}")
        launch, result_sha = validate_result()
        status = {
            "capability_manifest_sha256": args.capability_manifest_sha256,
            "deployment_manifest_sha256": DEPLOYMENT_MANIFEST,
            "result_sha256": result_sha,
            "result_status": launch["status"],
            "schema": "cbib1-r3-local3060-authority-wrapper-v0-r2",
            "status": "PASS_SINGLE_LOCAL_QWEN_ATTEMPT_CAPTURED_AND_VALIDATED",
        }
    except BaseException as exc:
        status = {
            "capability_manifest_sha256": args.capability_manifest_sha256,
            "deployment_manifest_sha256": DEPLOYMENT_MANIFEST,
            "error_type": type(exc).__name__,
            "schema": "cbib1-r3-local3060-authority-wrapper-v0-r2",
            "status": "FAIL_ATTEMPT_CONSUMED_NO_RETRY_AUTHORIZED",
        }
        exclusive_write(AUTHORITY_STATUS,
                        (json.dumps(status, sort_keys=True, separators=(",", ":")) + "\n").encode())
        raise
    exclusive_write(AUTHORITY_STATUS,
                    (json.dumps(status, sort_keys=True, separators=(",", ":")) + "\n").encode())
    print(json.dumps(status, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
