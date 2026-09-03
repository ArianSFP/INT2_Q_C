"""Read-only verifier for the consumed CBIB-1 local RTX3060 r2 failure."""
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
    "local_rtx3060_capability_20260903_r2"
)
DEPLOYMENT = REPO / "research" / (
    "same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3"
)
CAPABILITY_MANIFEST_SHA256 = (
    "fda1ce828a0bd6cb847971a9ec6575e3d7395e2a163fcb2d43f4d6f1240f4601"
)
CAPABILITY_SOURCE_ROOT_SHA256 = (
    "c359201c6f84c8153a7f5bb33230369216645b0da4a5adf6435fa0fb9e1fddc8"
)
DEPLOYMENT_MANIFEST_SHA256 = (
    "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
)
DEPLOYMENT_SOURCE_ROOT_SHA256 = (
    "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"
)
WRAPPER_SHA256 = "10afd965328338da8b7903ec52b7da935ffba0d6e5ca2b7a34f9d64d81f88bfa"
BRIDGE_SHA256 = "dcc4784c691e29d3131ffd65997d1de3216123db7ca8c02cfb753bc2fae3fa09"
RUN_GATE_SHA256 = "531aa3c07710231311a86d460c88e2f2112e2d286a8f49da21eab2bab09fd624"
CLAIM = Path(
    r"C:\INT2__compression\tmp\CBIB1_R3_LOCAL3060_AUTHORITY_ATTEMPT_20260903_R2_09F4C6D1.json"
)
STATUS = Path(
    r"C:\INT2__compression\tmp\CBIB1_R3_LOCAL3060_AUTHORITY_STATUS_20260903_R2_09F4C6D1.json"
)
RUN_ROOT = Path(
    r"C:\INT2__compression\tmp\cbib1_r3_local3060_qwen_once_20260903_r2_09f4c6d1"
)
CACHE_ROOT = Path(
    r"C:\INT2__compression\tmp\cbib1_r3_local3060_cupy_cache_20260903_r2_09f4c6d1"
)
CHILD_STDOUT = RUN_ROOT / "child_stdout.jsonl"
CHILD_STDERR = RUN_ROOT / "child_stderr.txt"
INNER_CLAIM = RUN_ROOT / "ONE_USE_CLAIM.json"
RESULT = RUN_ROOT / "result.json"
CLAIM_SHA256 = "84ccb60b915ab067f58510a16091e7eb8f8d441bc0b1f36eaa64ce2bd1f42fbc"
STATUS_SHA256 = "5a59637c922304c9e1e3c7b717ba6906445bb6cf667df92c6502f2b942b053b1"
STDERR_SHA256 = "467331a3c0f47a56ba1daf81a89a231df250943efd976d892933b634da17a720"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


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
    expected = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()
    need(raw == expected, f"canonical JSON required: {path}")
    return raw, obj


def authenticate_package(root: Path, expected_manifest_sha: str,
                         expected_source_root: str, expected_schema: str) -> dict:
    real_directory(root)
    raw, manifest = canonical_json(root / "SOURCE_MANIFEST.json")
    need(sha256_bytes(raw) == expected_manifest_sha, "external manifest digest")
    need(manifest.get("schema") == expected_schema, "external manifest schema")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "external manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "manifest ordering")
    need(sorted(path.name for path in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "external package closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "external manifest row")
        member = root / row["name"]
        real_file(member)
        need(member.stat().st_size == int(row["bytes"]) and
             sha256_file(member) == row["sha256"], f"external member: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed = sha256_bytes(json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode())
    need(observed == expected_source_root == manifest.get("source_root_sha256"),
         "external source-root digest")
    return manifest


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    need(len(matches) == 1, f"single function required: {name}")
    return matches[0]


def line_of_call(fn: ast.FunctionDef, name: str) -> int:
    matches = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            observed = target.id
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            observed = f"{target.value.id}.{target.attr}"
        else:
            continue
        if observed == name:
            matches.append(node.lineno)
    need(matches, f"call required: {fn.name}:{name}")
    return min(matches)


def verify_source_boundary() -> None:
    wrapper = CAPABILITY / "run_authorized_local_qwen_once.py"
    bridge = CAPABILITY / "local_runtime_bridge.py"
    run_gate = DEPLOYMENT / "run_gate.py"
    need(sha256_file(wrapper) == WRAPPER_SHA256, "wrapper identity")
    need(sha256_file(bridge) == BRIDGE_SHA256, "bridge identity")
    need(sha256_file(run_gate) == RUN_GATE_SHA256, "run_gate identity")

    wrapper_main = function(ast.parse(wrapper.read_text(encoding="utf-8")), "main")
    wrapper_order = [
        line_of_call(wrapper_main, "exclusive_write"),
        line_of_call(wrapper_main, "validate_prerequisites"),
        line_of_call(wrapper_main, "os.mkdir"),
        line_of_call(wrapper_main, "subprocess.run"),
    ]
    need(wrapper_order == sorted(wrapper_order) and len(set(wrapper_order)) == 4,
         "outer claim/prerequisite/directory/child launch order")

    bridge_tree = ast.parse(bridge.read_text(encoding="utf-8"))
    runtime_fn = function(bridge_tree, "validate_local_runtime")
    runtime_text = ast.get_source_segment(bridge.read_text(encoding="utf-8"), runtime_fn) or ""
    required_runtime_tokens = [
        "import cupy as cp", "runtime.runtimeGetVersion()", "runtime.driverGetVersion()",
        "runtime.getDevice()", "runtime.getDeviceProperties(0)", "runtime.deviceGetUuid(0)",
        "canonical_uuid(raw_uuid)", 'need(len(packed) == 16, "CUDA UUID bytes")',
    ]
    # canonical_uuid is a separate function, so bind its validation from full source.
    bridge_full = bridge.read_text(encoding="utf-8")
    for token in required_runtime_tokens[:-1]:
        need(token in runtime_text, f"runtime-boundary token: {token}")
    need(required_runtime_tokens[-1] in bridge_full, "UUID byte-length validation")
    ordered = [bridge_full.index(token) for token in required_runtime_tokens]
    # The byte-length check is defined earlier, while the call site is reached last.
    need(ordered[0] < ordered[1] < ordered[2] < ordered[3] < ordered[4] < ordered[5] < ordered[6],
         "CuPy/CUDA query order before canonical UUID call")

    gate_text = run_gate.read_text(encoding="utf-8")
    gate_order = [
        gate_text.index("_cp, numpy_receipt = _validate_runtime()"),
        gate_text.index("payload_root = Path(PAYLOAD_ROOT)"),
        gate_text.index("\n    _claim_once(claim, args.deployment_manifest_sha256)"),
        gate_text.index("worker.run_authorized_panel"),
    ]
    need(gate_order == sorted(gate_order) and len(set(gate_order)) == 4,
         "runtime before payload Path, inner claim, and worker payload access")


def verify_attempt_state() -> None:
    claim_raw, claim = canonical_json(CLAIM)
    status_raw, status = canonical_json(STATUS)
    need(sha256_bytes(claim_raw) == CLAIM_SHA256, "live outer claim identity")
    need(sha256_bytes(status_raw) == STATUS_SHA256, "live status identity")
    need(claim_raw == (ROOT / "OUTER_CLAIM_SNAPSHOT.json").read_bytes(),
         "outer claim snapshot binding")
    need(status_raw == (ROOT / "AUTHORITY_STATUS_SNAPSHOT.json").read_bytes(),
         "status snapshot binding")
    need(claim.get("status") == "ATTEMPT_CONSUMED_BEFORE_VALIDATION_OR_PAYLOAD_ACCESS" and
         claim.get("capability_manifest_sha256") == CAPABILITY_MANIFEST_SHA256,
         "outer claim semantics")
    need(status.get("status") == "FAIL_ATTEMPT_CONSUMED_NO_RETRY_AUTHORIZED" and
         status.get("error_type") == "RuntimeError" and
         status.get("capability_manifest_sha256") == CAPABILITY_MANIFEST_SHA256,
         "authority failure semantics")

    real_directory(RUN_ROOT)
    real_directory(CACHE_ROOT)
    need(list(CACHE_ROOT.iterdir()) == [], "fresh attempt cache must remain empty")
    real_file(CHILD_STDOUT)
    real_file(CHILD_STDERR)
    need(CHILD_STDOUT.stat().st_size == 0 and sha256_file(CHILD_STDOUT) == EMPTY_SHA256,
         "child stdout must remain empty")
    need(CHILD_STDERR.stat().st_size == 1896 and sha256_file(CHILD_STDERR) == STDERR_SHA256,
         "child stderr identity")
    live_stderr_text = CHILD_STDERR.read_text(encoding="utf-8").replace("\r\n", "\n")
    sealed_stderr_text = (ROOT / "CHILD_STDERR_SNAPSHOT.txt").read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n")
    need(live_stderr_text == sealed_stderr_text, "normalized child stderr snapshot binding")
    need(not os.path.lexists(INNER_CLAIM), "inner run_gate claim must be absent")
    need(not os.path.lexists(RESULT), "result must be absent")
    need(sorted(path.name for path in RUN_ROOT.iterdir()) ==
         ["child_stderr.txt", "child_stdout.jsonl"], "run-root exact closure")

    stderr = CHILD_STDERR.read_text(encoding="utf-8")
    required_trace = [
        "run_gate.py\", line 272, in main",
        "_cp, numpy_receipt = _validate_runtime()",
        "local_runtime_bridge.py\", line 256, in validate_local_runtime",
        "canonical_uuid(raw_uuid)",
        "local_runtime_bridge.py\", line 207, in canonical_uuid",
        'need(len(packed) == 16, "CUDA UUID bytes")',
        "RuntimeError: CUDA UUID bytes",
    ]
    positions = [stderr.index(token) for token in required_trace]
    need(positions == sorted(positions), "exact runtime failure traceback order")


def verify_audit_package() -> tuple[bytes, dict]:
    raw, manifest = canonical_json(ROOT / "SOURCE_MANIFEST.json")
    need(manifest.get("schema") ==
         "cbib1-r3-local3060-r2-consumed-failure-audit-manifest-v0", "audit schema")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "audit manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "audit manifest order")
    need(sorted(path.name for path in ROOT.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "audit package closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "audit manifest row")
        member = ROOT / row["name"]
        real_file(member)
        need(member.stat().st_size == int(row["bytes"]) and
             sha256_file(member) == row["sha256"], f"audit member: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    need(sha256_bytes(json.dumps(normalized, sort_keys=True,
                                 separators=(",", ":")).encode()) ==
         manifest.get("source_root_sha256"), "audit source-root digest")
    return raw, manifest


def main() -> int:
    manifest_raw, _ = verify_audit_package()
    authenticate_package(
        CAPABILITY, CAPABILITY_MANIFEST_SHA256, CAPABILITY_SOURCE_ROOT_SHA256,
        "same-layer-clustered-ib-r3-local-rtx3060-capability-manifest-v0-r2",
    )
    authenticate_package(
        DEPLOYMENT, DEPLOYMENT_MANIFEST_SHA256, DEPLOYMENT_SOURCE_ROOT_SHA256,
        "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3",
    )
    verify_source_boundary()
    verify_attempt_state()
    _, receipt = canonical_json(ROOT / "FAILURE_AUDIT_RECEIPT.json")
    need(receipt.get("status") ==
         "PASS_R2_CONSUMED_AFTER_CUPY_CUDA_INIT_BEFORE_INNER_CLAIM_QWEN_OR_RESULT",
         "receipt verdict")
    need(receipt.get("conclusions") == {
        "attempt_consumed": True,
        "cupy_imported_by_attempt": True,
        "cuda_runtime_initialized_by_attempt": True,
        "inner_run_gate_claim_created": False,
        "qwen_payload_enumerated_or_opened_by_attempt": False,
        "result_produced": False,
        "retry_authorized": False,
        "scientific_result": False,
    }, "receipt conclusions")
    print(json.dumps({
        "attempts_remaining": 0,
        "cuda_runtime_initialized_by_attempt": True,
        "inner_claim_present": False,
        "manifest_sha256": sha256_bytes(manifest_raw),
        "network_accessed_by_audit": False,
        "payload_accessed_by_attempt": False,
        "result_produced": False,
        "schema": "cbib1-r3-local3060-r2-failure-audit-verification-v0",
        "status": "PASS_R2_CONSUMED_AFTER_CUPY_CUDA_INIT_BEFORE_INNER_CLAIM_QWEN_OR_RESULT",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
