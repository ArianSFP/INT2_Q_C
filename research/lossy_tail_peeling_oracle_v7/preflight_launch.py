#!/usr/bin/env python3
"""V7 stdlib release preflight and one-use child-capability issuer."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


STAGE_MEMBERS = {
    "authorization_contract.json", "audit_lock_entrypoint.py",
    "launch_manifest.json", "lossy_tail_core.py", "lossy_tail_oracle.py",
    "preflight_launch.py", "protocol_lock.json", "repair_lock.json",
    "runtime_calibrate.py", "runtime_contract.json", "source_bindings.json",
}


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(f"V7_PREFLIGHT_REJECT: {message}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_object)
    except Exception as exc:
        fail(f"invalid strict JSON for {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        fail(f"{label} key drift")


def canonical_original(raw: str, label: str, *, allow_missing_tail: bool) -> Path:
    if not raw or "\x00" in raw or not os.path.isabs(raw) or raw != os.path.normpath(raw):
        fail(f"{label} must be absolute and lexically canonical")
    path = Path(raw)
    probe = Path(path.parts[0])
    missing = False
    for component in path.parts[1:]:
        probe = probe / component
        if missing:
            continue
        try:
            metadata = os.lstat(probe)
        except FileNotFoundError:
            if not allow_missing_tail:
                fail(f"{label} component missing: {probe}")
            missing = True
            continue
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains symlink component: {probe}")
    if os.path.realpath(raw) != raw:
        fail(f"{label} does not name its actual target")
    return path


def raw_entrypoint(expected_name: str) -> Path:
    launcher = canonical_original(sys.argv[0], "raw argv0", allow_missing_tail=False)
    module_raw = os.fspath(Path(__file__))
    if launcher.name != expected_name or sys.argv[0] != module_raw:
        fail("raw argv0 must exactly equal executing __file__")
    left = os.stat(launcher, follow_symlinks=False)
    right = os.stat(module_raw, follow_symlinks=False)
    if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
        fail("raw argv0 and executing module identities differ")
    return launcher


def read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"{label} is not regular")
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        payload = b"".join(chunks)
        if metadata.st_size != 0 and len(payload) != metadata.st_size:
            fail(f"{label} changed during descriptor read")
        return payload
    finally:
        os.close(descriptor)


def contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def disjoint(rows: Sequence[tuple[str, Path]]) -> None:
    for index, (left_label, left) in enumerate(rows):
        for right_label, right in rows[index + 1:]:
            if contains(left, right) or contains(right, left):
                fail(f"protected overlap: {left_label} and {right_label}")


def decode_mount(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def mount_snapshot() -> tuple[bytes, list[dict[str, Any]]]:
    payload = read_regular(Path("/proc/self/mountinfo"), "mountinfo")
    rows = []
    for line in payload.decode("utf-8").splitlines():
        fields = line.split()
        if "-" not in fields or len(fields) < 7:
            fail("malformed mountinfo")
        rows.append({"mount_id": int(fields[0]), "root": decode_mount(fields[3]), "mount_point": decode_mount(fields[4])})
    return payload, rows


def mount_for(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matches = [(len(Path(row["mount_point"]).parts), row) for row in rows if contains(Path(row["mount_point"]), path)]
    if not matches:
        fail(f"no mount row for {path}")
    return max(matches, key=lambda item: item[0])[1]


def no_nested_mounts(root: Path, rows: Sequence[dict[str, Any]], label: str) -> None:
    covering = mount_for(root, rows)
    for row in rows:
        point = Path(row["mount_point"])
        if row["mount_id"] != covering["mount_id"] and contains(root, point):
            fail(f"nested mount below {label}: {point}")


def verify_seal(value: dict[str, Any], field: str, expected: str, label: str) -> None:
    if value.get(field) != expected:
        fail(f"{label} internal field mismatch")
    copy = dict(value)
    copy.pop(field)
    if hashlib.sha256(canonical_bytes(copy)).hexdigest() != expected:
        fail(f"{label} internal recomputation mismatch")


def read_external(path_value: str, file_sha256: str, label: str) -> tuple[Path, dict[str, Any]]:
    path = canonical_original(path_value, label, allow_missing_tail=False)
    payload = read_regular(path, label)
    if hashlib.sha256(payload).hexdigest() != file_sha256:
        fail(f"{label} file SHA-256 mismatch")
    return path, strict_json(payload, label)


def validate_audit(section: dict[str, Any], label: str) -> tuple[Path, Path]:
    exact_keys(section, {
        "manifest_path", "manifest_file_sha256", "receipt_path", "receipt_file_sha256",
        "receipt_internal_field", "receipt_internal_sha256", "required_status",
    }, label)
    manifest = canonical_original(section["manifest_path"], f"{label} manifest", allow_missing_tail=False)
    if hashlib.sha256(read_regular(manifest, f"{label} manifest")).hexdigest() != section["manifest_file_sha256"]:
        fail(f"{label} manifest mismatch")
    receipt, value = read_external(section["receipt_path"], section["receipt_file_sha256"], f"{label} receipt")
    if value.get("status") != section["required_status"]:
        fail(f"{label} status mismatch")
    verify_seal(value, section["receipt_internal_field"], section["receipt_internal_sha256"], f"{label} receipt")
    return manifest, receipt


def validate_stage(stage: Path, manifest_path: Path, manifest_sha256: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    payload = read_regular(manifest_path, "launch manifest")
    if hashlib.sha256(payload).hexdigest() != manifest_sha256:
        fail("external launch-manifest SHA-256 mismatch")
    manifest = strict_json(payload, "launch manifest")
    exact_keys(manifest, {"schema", "status", "allowed_members", "members", "source_audit_invocation", "runtime_calibration_invocation_after_independent_source_pass_only", "production_invocation_after_independent_runtime_receipt_audit_and_separate_authorization_only", "production_child_grammar", "authorization"}, "launch manifest")
    if manifest.get("schema") != "lossy-tail-v7-launch-manifest-v1":
        fail("launch-manifest schema mismatch")
    if manifest.get("status") != "FROZEN_V7_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        fail("launch-manifest status mismatch")
    allowed = manifest.get("allowed_members")
    if not isinstance(allowed, list) or len(allowed) != len(STAGE_MEMBERS) or len(set(allowed)) != len(allowed) or set(allowed) != STAGE_MEMBERS:
        fail("allowed-member cardinality/set mismatch")
    observed = set()
    with os.scandir(stage) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                fail(f"forbidden stage entry: {entry.name}")
            observed.add(entry.name)
    if observed != STAGE_MEMBERS:
        fail("stage closure mismatch")
    rows = manifest.get("members")
    expected = STAGE_MEMBERS - {"launch_manifest.json"}
    if not isinstance(rows, list) or len(rows) != len(expected):
        fail("manifest member-row cardinality mismatch")
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(paths) != len(rows) or len(set(paths)) != len(paths) or set(paths) != expected:
        fail("manifest member-row closure mismatch")
    payloads = {"launch_manifest.json": payload}
    for row in rows:
        exact_keys(row, {"path", "bytes", "sha256"}, f"manifest row {row.get('path')}")
        member = read_regular(stage / row["path"], f"stage member {row['path']}")
        if len(member) != row["bytes"] or hashlib.sha256(member).hexdigest() != row["sha256"]:
            fail(f"stage member identity mismatch: {row['path']}")
        payloads[row["path"]] = member
    return manifest, payloads


def main() -> None:
    raw = sys.argv[1:]
    flags = ["--manifest", "--manifest-sha256", "--authorization", "--authorization-sha256"]
    if len(raw) != 8 or raw[::2] != flags:
        fail("exact grammar is --manifest PATH --manifest-sha256 HEX --authorization PATH --authorization-sha256 HEX")
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize != 0:
        fail("requires python -B -I without optimization")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        fail("requires explicit CUDA_VISIBLE_DEVICES=0 before CuPy import")
    launcher = raw_entrypoint("preflight_launch.py")
    stage = canonical_original(os.fspath(launcher.parent), "stage", allow_missing_tail=False)
    manifest_path = canonical_original(raw[1], "manifest", allow_missing_tail=False)
    if manifest_path != stage / "launch_manifest.json":
        fail("manifest must use exact immediate-stage spelling")
    manifest_sha256 = raw[3].lower()
    authorization_sha256 = raw[7].lower()
    for value, label in ((manifest_sha256, "manifest"), (authorization_sha256, "authorization")):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            fail(f"invalid {label} SHA-256")
    _, payloads = validate_stage(stage, manifest_path, manifest_sha256)
    repair = strict_json(payloads["repair_lock.json"], "repair lock")
    if repair.get("schema") != "lossy-tail-release-repair-lock-v7" or repair.get("status") != "FROZEN_V7_SOURCE_PACKAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION":
        fail("repair-lock status mismatch")
    verify_seal(repair, "repair_lock_sha256", repair.get("repair_lock_sha256"), "repair lock")
    identities = repair.get("authenticated_identities")
    live = {
        "scientific_protocol_sha256": hashlib.sha256(payloads["protocol_lock.json"]).hexdigest(),
        "source_bindings_sha256": hashlib.sha256(payloads["source_bindings.json"]).hexdigest(),
        "runtime_contract_sha256": hashlib.sha256(payloads["runtime_contract.json"]).hexdigest(),
        "authorization_contract_sha256": hashlib.sha256(payloads["authorization_contract.json"]).hexdigest(),
        "oracle_bootstrap_sha256": hashlib.sha256(payloads["lossy_tail_oracle.py"]).hexdigest(),
        "scientific_core_sha256": hashlib.sha256(payloads["lossy_tail_core.py"]).hexdigest(),
        "preflight_sha256": hashlib.sha256(payloads["preflight_launch.py"]).hexdigest(),
        "audit_entrypoint_sha256": hashlib.sha256(payloads["audit_lock_entrypoint.py"]).hexdigest(),
        "runtime_calibrate_sha256": hashlib.sha256(payloads["runtime_calibrate.py"]).hexdigest(),
    }
    if identities != live:
        fail("repair-lock authenticated identities mismatch")
    protocol = strict_json(payloads["protocol_lock.json"], "protocol")
    runtime_contract = strict_json(payloads["runtime_contract.json"], "runtime contract")
    authorization_contract = strict_json(payloads["authorization_contract.json"], "authorization contract")
    bindings = strict_json(payloads["source_bindings.json"], "bindings")
    if protocol.get("status") != "FROZEN_V7_BEFORE_ANY_RUNTIME_CALIBRATION_PAYLOAD_OR_GPU_EXECUTION":
        fail("protocol status mismatch")
    if runtime_contract.get("status") != "FROZEN_SOURCE_FREE_BEFORE_RUNTIME_CALIBRATION":
        fail("runtime contract status mismatch")
    if authorization_contract.get("status") != "FROZEN_TEMPLATE_ONLY_NO_AUTHORIZATION_EXISTS":
        fail("authorization contract status mismatch")

    authorization_path, authorization = read_external(raw[5], authorization_sha256, "production authorization")
    exact_keys(authorization, {
        "schema", "status", "authorization_path", "authorization_nonce", "action", "stage", "source",
        "output", "source_audit", "runtime_receipt", "runtime_audit", "execution", "filesystem",
        "fixed_scientific_arguments", "authorization_sha256",
    }, "production authorization")
    if authorization.get("schema") != "lossy-tail-v7-one-shot-production-authorization-v1" or authorization.get("status") != "AUTHORIZED_ONCE_AFTER_INDEPENDENT_SOURCE_AND_RUNTIME_AUDITS":
        fail("production authorization status mismatch")
    if authorization.get("action") != "CREATE_NEW_RUN_ROOT_AND_RESULT_JSON" or not authorization.get("authorization_nonce"):
        fail("production authorization action/nonce mismatch")
    if authorization.get("authorization_path") != os.fspath(authorization_path):
        fail("production authorization path mismatch")
    verify_seal(authorization, "authorization_sha256", authorization["authorization_sha256"], "production authorization")
    stage_row = authorization["stage"]
    exact_keys(stage_row, {"path", "launch_manifest_file_sha256", "launch_manifest_internal_stage_member_count"}, "authorized stage")
    if stage_row != {"path": os.fspath(stage), "launch_manifest_file_sha256": manifest_sha256, "launch_manifest_internal_stage_member_count": len(STAGE_MEMBERS)}:
        fail("authorized stage mismatch")
    source_row = authorization["source"]
    exact_keys(source_row, {"path", "bindings_file_sha256"}, "authorized source")
    if source_row["bindings_file_sha256"] != live["source_bindings_sha256"] or source_row["path"] != bindings.get("source_directory_at_execution"):
        fail("authorized source/bindings mismatch")
    source = canonical_original(source_row["path"], "source", allow_missing_tail=False)
    if not source.is_dir():
        fail("source is not a directory")
    output_row = authorization["output"]
    exact_keys(output_row, {"run_root", "result_path"}, "authorized output")
    run_root = canonical_original(output_row["run_root"], "run root", allow_missing_tail=True)
    result = canonical_original(output_row["result_path"], "result", allow_missing_tail=True)
    if result != run_root / "result.json" or run_root.exists() or result.exists() or not run_root.parent.is_dir():
        fail("authorized create-new output mismatch")
    protected_roots = (("stage", stage), ("source", source), ("output_existing_parent", run_root.parent))
    disjoint(protected_roots)

    source_audit_manifest, source_audit_receipt = validate_audit(authorization["source_audit"], "source audit")
    source_audit = strict_json(read_regular(source_audit_receipt, "source audit replay"), "source audit replay")
    audited_target = source_audit.get("audited_target", {})
    if audited_target.get("launch_manifest_sha256") != manifest_sha256 or audited_target.get("repair_lock_internal_sha256") != repair.get("repair_lock_sha256"):
        fail("source audit does not bind current manifest/repair lock")
    source_access = source_audit.get("access_ledger", {})
    if any(source_access.get(key) != 0 for key in ("model_payload_files_opened", "cupy_imports", "cuda_initializations", "gpu_jobs")):
        fail("source audit is not source-only")
    runtime_section = authorization["runtime_receipt"]
    exact_keys(runtime_section, {"path", "file_sha256", "internal_sha256", "required_status", "runtime_contract_file_sha256"}, "runtime receipt")
    if runtime_section["runtime_contract_file_sha256"] != live["runtime_contract_sha256"]:
        fail("runtime receipt contract mismatch")
    runtime_receipt_path, runtime_receipt = read_external(runtime_section["path"], runtime_section["file_sha256"], "runtime receipt")
    if runtime_receipt.get("schema") != "lossy-tail-v7-source-free-runtime-receipt-v1" or runtime_receipt.get("status") != runtime_section["required_status"]:
        fail("runtime receipt schema/status mismatch")
    verify_seal(runtime_receipt, "runtime_receipt_sha256", runtime_section["internal_sha256"], "runtime receipt")
    if runtime_receipt.get("runtime_contract", {}).get("sha256") != live["runtime_contract_sha256"]:
        fail("runtime receipt contract identity mismatch")
    access = runtime_receipt.get("access_ledger", {})
    if any(access.get(key) != 0 for key in ("model_or_qwen_paths_supplied", "model_or_qwen_paths_opened", "payload_files_opened", "production_results_opened")):
        fail("runtime receipt is not source-free")
    runtime_audit_manifest, runtime_audit_receipt = validate_audit(authorization["runtime_audit"], "runtime audit")
    runtime_audit = strict_json(read_regular(runtime_audit_receipt, "runtime audit replay"), "runtime audit replay")
    audited = runtime_audit.get("audited_runtime_receipt", {})
    if audited.get("file_sha256") != runtime_section["file_sha256"] or audited.get("internal_sha256") != runtime_section["internal_sha256"]:
        fail("runtime audit does not bind runtime receipt")
    runtime_audit_access = runtime_audit.get("access_ledger", {})
    if any(runtime_audit_access.get(key) != 0 for key in ("model_payload_files_opened", "production_result_files_opened", "gpu_jobs")):
        fail("runtime audit exceeded source-free audit scope")
    execution = authorization["execution"]
    exact_keys(execution, {"python_executable", "raw_launcher_path", "cuda_visible_devices", "runtime_tuple"}, "authorized execution")
    if execution["python_executable"] != sys.executable or execution["raw_launcher_path"] != os.fspath(launcher) or execution["cuda_visible_devices"] != "0":
        fail("authorized execution mismatch")
    if execution["runtime_tuple"] != runtime_receipt.get("runtime_probe", {}).get("runtime_tuple"):
        fail("authorized runtime tuple mismatch")
    if authorization["fixed_scientific_arguments"] != {"control_replicates": 4, "maximum_coordinate_passes": 4}:
        fail("fixed scientific arguments mismatch")

    evidence = (
        ("authorization", authorization_path), ("source_audit_manifest", source_audit_manifest),
        ("source_audit_receipt", source_audit_receipt), ("runtime_receipt", runtime_receipt_path),
        ("runtime_audit_manifest", runtime_audit_manifest), ("runtime_audit_receipt", runtime_audit_receipt),
    )
    for protected_label, protected in protected_roots:
        for evidence_label, evidence_path in evidence:
            if (
                contains(protected, evidence_path) or contains(evidence_path, protected)
                or contains(protected, evidence_path.parent) or contains(evidence_path.parent, protected)
            ):
                fail(f"{evidence_label} overlaps {protected_label}")
    filesystem = authorization["filesystem"]
    exact_keys(filesystem, {"mountinfo_path", "mountinfo_file_sha256", "identities"}, "filesystem")
    if filesystem["mountinfo_path"] != "/proc/self/mountinfo":
        fail("mountinfo path mismatch")
    mount_payload, mount_rows = mount_snapshot()
    if hashlib.sha256(mount_payload).hexdigest() != filesystem["mountinfo_file_sha256"]:
        fail("mountinfo hash mismatch")
    paths = {
        "stage": stage, "source": source, "output_existing_parent": run_root.parent,
        "authorization_parent": authorization_path.parent,
        "source_audit_manifest": source_audit_manifest, "source_audit_receipt": source_audit_receipt,
        "runtime_receipt": runtime_receipt_path, "runtime_audit_manifest": runtime_audit_manifest,
        "runtime_audit_receipt": runtime_audit_receipt,
    }
    rows = filesystem["identities"]
    if not isinstance(rows, list) or len(rows) != len(paths):
        fail("filesystem identity cardinality mismatch")
    labels = [row.get("label") for row in rows if isinstance(row, dict)]
    if len(labels) != len(rows) or len(set(labels)) != len(labels) or set(labels) != set(paths):
        fail("filesystem identity labels mismatch")
    seen: dict[tuple[int, int], str] = {}
    for row in rows:
        exact_keys(row, {"label", "path", "st_dev", "st_ino", "mount_id"}, f"identity {row.get('label')}")
        path = paths[row["label"]]
        metadata = os.stat(path, follow_symlinks=False)
        mount = mount_for(path, mount_rows)
        if row["path"] != os.fspath(path) or (row["st_dev"], row["st_ino"], row["mount_id"]) != (metadata.st_dev, metadata.st_ino, mount["mount_id"]):
            fail(f"filesystem identity mismatch: {row['label']}")
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in seen and seen[inode] != row["label"]:
            fail(f"filesystem alias: {seen[inode]} and {row['label']}")
        seen[inode] = row["label"]
    no_nested_mounts(stage, mount_rows, "stage")
    no_nested_mounts(source, mount_rows, "source")
    no_nested_mounts(run_root, mount_rows, "run root")
    print(json.dumps({
        "v7_production_preflight": "PASS_AUTHORIZATION_BOUND_BEFORE_CHILD_CAPABILITY",
        "stage": os.fspath(stage), "manifest_sha256": manifest_sha256,
        "authorization_file_sha256": authorization_sha256,
        "authorization_internal_sha256": authorization["authorization_sha256"],
        "runtime_receipt_file_sha256": runtime_section["file_sha256"],
        "payload_files_opened": 0, "cupy_imported": False,
    }, sort_keys=True), flush=True)

    child_flags = [
        "--bindings", os.fspath(stage / "source_bindings.json"),
        "--protocol", os.fspath(stage / "protocol_lock.json"),
        "--repair-lock", os.fspath(stage / "repair_lock.json"),
        "--runtime-contract", os.fspath(stage / "runtime_contract.json"),
        "--authorization-contract", os.fspath(stage / "authorization_contract.json"),
        "--launch-manifest", os.fspath(manifest_path),
        "--launch-manifest-sha256", manifest_sha256,
        "--authorization", os.fspath(authorization_path),
        "--authorization-sha256", authorization_sha256,
        "--control-replicates", "4",
        "--maximum-coordinate-passes", "4",
    ]
    oracle_path = stage / "lossy_tail_oracle.py"
    live_parent_cmdline = read_regular(Path("/proc/self/cmdline"), "live preflight cmdline")
    expected_parent_fields = [sys.executable, "-B", "-I", *sys.argv]
    expected_parent_cmdline = b"\x00".join(field.encode("utf-8") for field in expected_parent_fields) + b"\x00"
    if live_parent_cmdline != expected_parent_cmdline:
        fail("live preflight command line differs from frozen interpreter/-B/-I/argv grammar")
    parent_channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    process: subprocess.Popen[bytes] | None = None
    try:
        child_descriptor = child_channel.fileno()
        command = [
            sys.executable, "-B", "-I", os.fspath(oracle_path),
            *child_flags, "--capability-fd", str(child_descriptor),
        ]
        process = subprocess.Popen(
            command,
            close_fds=True,
            pass_fds=(child_descriptor,),
            env=dict(os.environ),
        )
        child_channel.close()
        capability = {
            "schema": "lossy-tail-v7-one-use-child-capability-v1",
            "status": "ISSUED_ONCE_BY_AUTHENTICATED_PREFLIGHT",
            "parent_pid": os.getpid(),
            "child_pid": process.pid,
            "nonce_hex": secrets.token_hex(32),
            "preflight_cmdline_sha256": hashlib.sha256(live_parent_cmdline).hexdigest(),
            "launch_manifest_sha256": manifest_sha256,
            "authorization_file_sha256": authorization_sha256,
            "authorization_internal_sha256": authorization["authorization_sha256"],
            "bootstrap_sha256": hashlib.sha256(payloads["lossy_tail_oracle.py"]).hexdigest(),
            "scientific_core_sha256": hashlib.sha256(payloads["lossy_tail_core.py"]).hexdigest(),
        }
        capability_sha256 = hashlib.sha256(canonical_bytes(capability)).hexdigest()
        parent_channel.send(canonical_bytes(capability))
        parent_channel.shutdown(socket.SHUT_WR)
        parent_channel.settimeout(30.0)
        acknowledgement_payload = parent_channel.recv(65536)
        if not acknowledgement_payload:
            fail("child closed capability channel without acknowledgement")
        acknowledgement = strict_json(acknowledgement_payload, "child capability acknowledgement")
        exact_keys(acknowledgement, {
            "schema", "status", "child_pid", "capability_sha256",
        }, "child capability acknowledgement")
        if acknowledgement != {
            "schema": "lossy-tail-v7-child-capability-ack-v1",
            "status": "CONSUMED_ONCE_BEFORE_THIRD_PARTY_IMPORT",
            "child_pid": process.pid,
            "capability_sha256": capability_sha256,
        }:
            fail("child capability acknowledgement mismatch")
        if parent_channel.recv(1) != b"":
            fail("child capability channel contains more than one acknowledgement record")
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise
    finally:
        parent_channel.close()
        child_channel.close()
    if process is None:
        fail("child process was not created")
    child_exit = process.wait()
    if child_exit != 0:
        fail(f"authenticated production child exited with code {child_exit}")
    print(json.dumps({
        "v7_production_child": "EXITED_SUCCESS_AFTER_ONE_USE_CAPABILITY",
        "child_pid": process.pid,
        "capability_sha256": capability_sha256,
        "payload_files_opened_by_preflight": 0,
        "cupy_imported_by_preflight": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
