"""Stdlib-only verifier for the sealed local RTX3060 capability."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat


DEPLOYMENT_MANIFEST = "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
DEPLOYMENT_SOURCE_ROOT = "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate(root: Path, manifest_sha: str, source_root: str | None, schema: str) -> dict:
    root = root.resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(), "real package root")
    manifest_path = root / "SOURCE_MANIFEST.json"
    need(stat.S_ISREG(manifest_path.lstat().st_mode) and not manifest_path.is_symlink(),
         "regular manifest")
    need(sha(manifest_path) == manifest_sha, "external manifest digest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "canonical manifest")
    need(manifest.get("schema") == schema, "manifest schema")
    rows = manifest.get("files")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "manifest ordering")
    need(sorted(path.name for path in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "package closure")
    normalized = []
    for row in rows:
        member = root / row["name"]
        need(set(row) == {"bytes", "name", "sha256"} and
             stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink() and
             member.stat().st_size == int(row["bytes"]) and sha(member) == row["sha256"],
             f"member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(observed == manifest.get("source_root_sha256") and
         (source_root is None or observed == source_root), "source-root digest")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--deployment", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    manifest = authenticate(
        package, args.manifest_sha256, None,
        "same-layer-clustered-ib-r3-local-rtx3060-capability-manifest-v0-r2",
    )
    deployment = authenticate(
        args.deployment, DEPLOYMENT_MANIFEST, DEPLOYMENT_SOURCE_ROOT,
        "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3",
    )
    need(len(deployment["files"]) == 15 and manifest.get("status") == "AUTHORIZED_NOT_EXECUTED",
         "sealed dependency semantics")
    authority = json.loads((package / "AUTHORIZED_LOCAL_QWEN_ONCE.json").read_bytes())
    lock_raw = (package / "LOCAL_RUNTIME_LOCK.json").read_bytes()
    lock = json.loads(lock_raw)
    need(lock_raw == (json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "canonical runtime lock")
    need(authority.get("schema") ==
         "same-layer-clustered-ib-r3-local-rtx3060-production-authority-v0-r2" and
         authority.get("authorization_id") ==
         "CBIB1_R3_LOCAL3060_QWEN_ONCE_20260903_R2_09F4C6D1" and
         authority.get("status") == "AUTHORIZED_NOT_EXECUTED" and
         authority.get("permitted_attempts") == 1 and authority["command"]["shell"] is False and
         authority["frozen_science"]["panel_lock_sha256"] ==
         "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5" and
         authority["runtime"]["device_uuid"] ==
         "GPU-458a424a-76e3-65e5-0470-803e0ed131ca" and
         authority["runtime"]["local_runtime_lock_sha256"] == sha(package / "LOCAL_RUNTIME_LOCK.json") and
         authority["runtime"]["bridge_sha256"] == sha(package / "local_runtime_bridge.py") and
         authority["wrapper_sha256"] == sha(package / "run_authorized_local_qwen_once.py") and
         authority["paths"]["cache_root"] ==
         r"C:\INT2__compression\tmp\cbib1_r3_local3060_cupy_cache_20260903_r2_09f4c6d1" and
         authority["paths"]["run_root"] ==
         r"C:\INT2__compression\tmp\cbib1_r3_local3060_qwen_once_20260903_r2_09f4c6d1",
         "authority pins")
    need(lock["status"] == "PINNED_SOURCE_ONLY_BEFORE_LOCAL_QWEN_AUTHORITY" and
         lock["cuda"]["device_name"] == "NVIDIA GeForce RTX 3060" and
         lock["cuda"]["driver_api_version"] == 12060 and
         lock["cuda"]["runtime_api_version"] == 12090 and
         len(lock["wheel_records"]) == 10 and len(lock["process_local_dll"]["directories"]) == 9,
         "runtime lock pins")
    wrapper_text = (package / "run_authorized_local_qwen_once.py").read_text(encoding="utf-8")
    bridge_text = (package / "local_runtime_bridge.py").read_text(encoding="utf-8")
    ast.parse(wrapper_text); ast.parse(bridge_text)
    need("exclusive_write(ATTEMPT_CLAIM, claim_raw)" in wrapper_text and
         "shell=False" in wrapper_text and "os.add_dll_directory" in bridge_text and
         "module._validate_runtime = validate_local_runtime" in bridge_text and
         not any(token in wrapper_text + bridge_text for token in (".unlink(", "rmtree(")),
         "one-use process-local source boundary")
    receipt = json.loads((package / "SOURCE_ONLY_TEST_RECEIPT.json").read_bytes())
    need(receipt.get("schema") == "cbib1-r3-local3060-capability-source-only-test-v0-r2" and
         receipt.get("status") == "PASS_SOURCE_ONLY_NOT_EXECUTED" and
         receipt.get("payload_accessed") is False and receipt.get("gpu_initialized") is False and
         receipt.get("production_executed") is False and
         receipt.get("parent_writeability_passed") is True and
         receipt.get("parent_write_probe_cleaned") is True,
         "source-only receipt")
    print(json.dumps({
        "authorized_attempts": 1,
        "capability_manifest_sha256": args.manifest_sha256,
        "capability_source_root_sha256": manifest["source_root_sha256"],
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST,
        "gpu_initialized": False,
        "payload_accessed": False,
        "production_executed": False,
        "schema": "cbib1-r3-local3060-capability-verification-v0-r2",
        "status": "PASS_SEALED_SOURCE_ONLY_AUTHORIZED_NOT_EXECUTED",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
