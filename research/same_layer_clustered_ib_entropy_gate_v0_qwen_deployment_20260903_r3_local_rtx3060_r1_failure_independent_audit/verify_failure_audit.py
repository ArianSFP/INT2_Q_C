"""Read-only verifier for the consumed CBIB-1 local RTX3060 r1 failure."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CAPABILITY = REPO / "research" / (
    "same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3_"
    "local_rtx3060_capability_20260903_r1"
)
CAPABILITY_MANIFEST_SHA256 = (
    "1589197b2ad053efacd7287b667294e6e8b4d736bff78196a1c29d991613fb5d"
)
CAPABILITY_SOURCE_ROOT_SHA256 = (
    "28b639186be82851e409c3ec9058ddca01acc816be54ff58473628e325c0f166"
)
WRAPPER_SHA256 = "c388a7c8e4377f009c368e1a6857286aff525eca74d15c13d711071b05045ceb"
BRIDGE_SHA256 = "d14bc02e00aed0c1bd109c46247342ff4402f28a23bdd97b9940d2fb5bd0420f"
CLAIM = Path(
    r"C:\INT2__compression\tmp\CBIB1_R3_LOCAL3060_AUTHORITY_ATTEMPT_20260903_458A424A.json"
)
STATUS = Path(
    r"C:\INT2__compression\tmp\CBIB1_R3_LOCAL3060_AUTHORITY_STATUS_20260903_458A424A.json"
)
RUN_ROOT = Path(r"C:\INT2__compression\tmp\cbib1_r3_local3060_qwen_once_20260903_458a424a")
CACHE_ROOT = Path(r"C:\INT2__compression\.cupy_cache\cbib1_r3_local3060_20260903_458a424a")
CLAIM_SHA256 = "e5c9cd1fd09a91dbe403178c65cea6bb9a1d87d212de08d05f19373a003c195f"
STATUS_SHA256 = "eafcf3e422e87da229a4bba3f24b41cde435516b8a17869ed33b1e9be0c97d2f"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def real_file(path: Path) -> None:
    need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
         f"regular nonsymlink required: {path}")


def real_directory(path: Path) -> None:
    need(path.is_dir() and not path.is_symlink(), f"real directory required: {path}")
    need(path.resolve(strict=True) == path.absolute(), f"directory indirection forbidden: {path}")


def canonical_json(path: Path) -> tuple[bytes, dict]:
    real_file(path)
    raw = path.read_bytes()
    obj = json.loads(raw)
    need(raw == (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         f"canonical JSON required: {path}")
    return raw, obj


def authenticate_package(root: Path, expected_manifest_sha: str,
                         expected_source_root: str, expected_schema: str) -> dict:
    real_directory(root)
    manifest_path = root / "SOURCE_MANIFEST.json"
    raw, manifest = canonical_json(manifest_path)
    need(sha256_bytes(raw) == expected_manifest_sha, "external manifest SHA-256")
    need(manifest.get("schema") == expected_schema, "external manifest schema")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "external manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "manifest ordering")
    need(sorted(path.name for path in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "package closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "manifest row schema")
        member = root / row["name"]
        real_file(member)
        need(member.stat().st_size == int(row["bytes"]) and
             sha256_file(member) == row["sha256"], f"member identity: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed = sha256_bytes(json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode())
    need(observed == expected_source_root == manifest.get("source_root_sha256"),
         "source-root digest")
    return manifest


def call_at(tree: ast.AST, line: int, dotted_name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node, "lineno", None) != line:
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            name = f"{func.value.id}.{func.attr}"
        else:
            continue
        if name == dotted_name:
            return True
    return False


def verify_source_boundary() -> None:
    wrapper = CAPABILITY / "run_authorized_local_qwen_once.py"
    bridge = CAPABILITY / "local_runtime_bridge.py"
    need(sha256_file(wrapper) == WRAPPER_SHA256, "wrapper identity")
    need(sha256_file(bridge) == BRIDGE_SHA256, "bridge identity")
    wrapper_tree = ast.parse(wrapper.read_text(encoding="utf-8"), filename=str(wrapper))
    need(call_at(wrapper_tree, 213, "exclusive_write"), "outer claim boundary")
    need(call_at(wrapper_tree, 216, "validate_prerequisites"), "prerequisite boundary")
    need(call_at(wrapper_tree, 219, "os.mkdir"), "run-root creation boundary")
    need(call_at(wrapper_tree, 220, "os.mkdir"), "cache creation boundary")
    need(call_at(wrapper_tree, 221, "os.open") and call_at(wrapper_tree, 222, "os.open"),
         "child log creation boundary")
    need(call_at(wrapper_tree, 232, "subprocess.run"), "child launch boundary")
    bridge_tree = ast.parse(bridge.read_text(encoding="utf-8"), filename=str(bridge))
    imports = set()
    for node in bridge_tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    need("cupy" not in imports and "numpy" not in imports, "bridge static import boundary")
    text = wrapper.read_text(encoding="utf-8")
    need("bridge.static_runtime_preflight(lock)" in text and
         "PAYLOAD_ROOT.resolve(strict=True) == PAYLOAD_ROOT" in text,
         "parent-only static prerequisite boundary")


def verify_attempt_state() -> None:
    claim_raw, claim = canonical_json(CLAIM)
    status_raw, status = canonical_json(STATUS)
    need(sha256_bytes(claim_raw) == CLAIM_SHA256, "outer claim snapshot identity")
    need(sha256_bytes(status_raw) == STATUS_SHA256, "authority status snapshot identity")
    need(claim == json.loads((ROOT / "OUTER_CLAIM_SNAPSHOT.json").read_bytes()),
         "outer claim snapshot binding")
    need(status == json.loads((ROOT / "AUTHORITY_STATUS_SNAPSHOT.json").read_bytes()),
         "authority status snapshot binding")
    need(claim.get("status") == "ATTEMPT_CONSUMED_BEFORE_VALIDATION_OR_PAYLOAD_ACCESS" and
         claim.get("capability_manifest_sha256") == CAPABILITY_MANIFEST_SHA256,
         "outer claim semantics")
    need(status.get("status") == "FAIL_ATTEMPT_CONSUMED_NO_RETRY_AUTHORIZED" and
         status.get("error_type") == "PermissionError" and
         status.get("capability_manifest_sha256") == CAPABILITY_MANIFEST_SHA256,
         "authority failure semantics")
    real_directory(RUN_ROOT)
    need(list(RUN_ROOT.iterdir()) == [], "fixed run root must remain empty")
    need(not os.path.lexists(CACHE_ROOT), "fixed cache root must be absent")
    for child in ("child_stdout.jsonl", "child_stderr.txt", "ONE_USE_CLAIM.json", "result.json"):
        need(not os.path.lexists(RUN_ROOT / child), f"pre-child artifact must be absent: {child}")


def main() -> int:
    manifest_raw, manifest = canonical_json(ROOT / "SOURCE_MANIFEST.json")
    need(manifest.get("schema") ==
         "cbib1-r3-local3060-r1-consumed-failure-audit-manifest-v0", "audit manifest schema")
    expected_manifest_sha = manifest.get("self_sha256_unbound")
    need(expected_manifest_sha == "SELF_NOT_HASHED_BY_DESIGN", "manifest self convention")
    names = [row["name"] for row in manifest["files"]]
    need(names == sorted(names) and len(names) == len(set(names)), "audit manifest order")
    need(sorted(path.name for path in ROOT.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "audit package closure")
    normalized = []
    for row in manifest["files"]:
        member = ROOT / row["name"]
        real_file(member)
        need(member.stat().st_size == row["bytes"] and sha256_file(member) == row["sha256"],
             f"audit member identity: {row['name']}")
        normalized.append({"bytes": row["bytes"], "name": row["name"],
                           "sha256": row["sha256"]})
    need(sha256_bytes(json.dumps(normalized, sort_keys=True,
                                 separators=(",", ":")).encode()) ==
         manifest.get("source_root_sha256"), "audit source-root digest")
    authenticate_package(
        CAPABILITY, CAPABILITY_MANIFEST_SHA256, CAPABILITY_SOURCE_ROOT_SHA256,
        "same-layer-clustered-ib-r3-local-rtx3060-capability-manifest-v0-r1",
    )
    verify_source_boundary()
    verify_attempt_state()
    _, receipt = canonical_json(ROOT / "FAILURE_AUDIT_RECEIPT.json")
    need(receipt.get("status") == "PASS_R1_CONSUMED_PRE_CHILD_NO_GPU_QWEN_OR_RESULT",
         "receipt verdict")
    conclusions = receipt.get("conclusions", {})
    need(conclusions == {
        "attempt_consumed": True,
        "child_process_launched": False,
        "gpu_initialized_by_attempt": False,
        "qwen_payload_enumerated_or_opened_by_attempt": False,
        "result_produced": False,
        "retry_authorized": False,
        "scientific_result": False,
    }, "receipt conclusions")
    output = {
        "attempts_remaining": 0,
        "cache_root_absent": True,
        "child_artifacts_present": 0,
        "gpu_initialized_by_attempt": False,
        "manifest_sha256": sha256_bytes(manifest_raw),
        "network_accessed": False,
        "payload_accessed": False,
        "result_produced": False,
        "schema": "cbib1-r3-local3060-r1-failure-audit-verification-v0",
        "status": "PASS_R1_CONSUMED_PRE_CHILD_NO_GPU_QWEN_OR_RESULT",
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
