#!/usr/bin/env python3
"""Immutable input/runtime/output dispatcher for N18 v6.

The entry module imports no numerical package or sibling source at import
time.  It first authenticates its own complete source manifest, then executes
the exact authenticated runtime/codec bytes, binds BF16 inputs through held
file descriptors, and publishes through a private completed staging tree plus
atomic directory rename.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import struct
import sys
import types
from pathlib import Path
from typing import Any, Mapping


AUTHORIZATION = "RUN_TACTIC_ACTUAL_COARSE_N18_V6_BOUND_EXPERT"
PACKAGE_SCHEMA = "tactic-actual-coarse-n18-v6-source-manifest-v1"
INPUT_SCHEMA = "tactic-actual-coarse-n18-v6-input-manifest-v1"
COMPLETE_SCHEMA = "tactic-actual-coarse-n18-v6-completion-v1"
MAX_SOURCE_BYTES = 4 * (1 << 20)
MAX_INPUT_MANIFEST_BYTES = 1 << 20
MAX_ROLE_BYTES = 1 << 34


class DispatchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(DispatchError(f"{label} nonfinite {item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DispatchError(f"{label} JSON: {error}") from error
    require(isinstance(value, dict), f"{label} JSON object")
    return value


def _reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode), f"{label} symlink chain: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def read_held_regular(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    maximum_bytes: int,
    label: str,
) -> bytes:
    require(path.is_absolute(), f"{label} absolute path")
    _reject_symlink_chain(path, label)
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} regular file")
        require(0 < before.st_size <= maximum_bytes, f"{label} byte bound")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes, f"{label} exact bytes")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label} short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label} trailing read")
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns),
            f"{label} identity drift",
        )
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256, f"{label} SHA-256")
        return payload
    finally:
        os.close(descriptor)


def bootstrap_source_auth(package_dir: Path, expected_manifest_sha256: str) -> Any:
    manifest_payload = read_held_regular(
        package_dir / "SOURCE_MANIFEST.json",
        expected_sha256=expected_manifest_sha256,
        maximum_bytes=MAX_SOURCE_BYTES,
        label="v6 source manifest",
    )
    manifest = _strict_json(manifest_payload, "v6 source manifest")
    require(manifest.get("schema") == PACKAGE_SCHEMA, "v6 source manifest schema")
    rows = manifest.get("members")
    require(isinstance(rows, list), "v6 source members")
    matches = [row for row in rows if isinstance(row, dict) and
               row.get("name") == "source_auth.py"]
    require(len(matches) == 1, "v6 source_auth manifest row")
    row = matches[0]
    source = read_held_regular(
        package_dir / "source_auth.py",
        expected_bytes=int(row["bytes"]),
        expected_sha256=str(row["sha256"]),
        maximum_bytes=MAX_SOURCE_BYTES,
        label="v6 source_auth",
    )
    return _load_authenticated_module(
        "tacn18_v6_dispatch_source_auth", source)


def authenticate_package(
    package_dir: Path, expected_manifest_sha256: str,
) -> Any:
    auth = bootstrap_source_auth(package_dir, expected_manifest_sha256)
    return auth.HeldSourcePackage(
        package_dir, expected_manifest_sha256,
        executing_path=Path(__file__).resolve(strict=True),
    )


def _load_authenticated_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, f"authenticated module collision: {name}")
    digest = sha256(source)
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = digest
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _require_finite_bf16(payload: bytes, label: str) -> None:
    require(len(payload) % 2 == 0, f"{label} BF16 alignment")
    require(all((word & 0x7F80) != 0x7F80
                for (word,) in struct.iter_unpack("<H", payload)),
            f"{label} finite canonical BF16 words")


def require_output_outside_package(
    output_dir: Path, package_dir: Path,
) -> None:
    require(output_dir.is_absolute(), "output directory absolute")
    resolved_output_parent = output_dir.parent.resolve(strict=True)
    require(resolved_output_parent != package_dir and
            package_dir not in resolved_output_parent.parents,
            "output namespace must remain outside immutable source package")


def authenticate_inputs(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    payload = read_held_regular(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        maximum_bytes=MAX_INPUT_MANIFEST_BYTES,
        label="expert input manifest",
    )
    record = _strict_json(payload, "expert input manifest")
    require(
        set(record) == {"schema", "geometry", "roles", "output_directory_name"},
        "identity-free input manifest schema",
    )
    require(record["schema"] == INPUT_SCHEMA, "input manifest schema")
    geometry = record["geometry"]
    require(
        isinstance(geometry, dict) and set(geometry) == {"intermediate", "hidden"},
        "input geometry",
    )
    intermediate = geometry["intermediate"]
    hidden = geometry["hidden"]
    require(type(intermediate) is int and type(hidden) is int and intermediate > 0 and hidden > 0, "input dimensions")
    expected_bytes = 2 * intermediate * hidden
    rows = record["roles"]
    require(isinstance(rows, list) and len(rows) == 3, "input role rows")
    role_bytes: dict[str, bytes] = {}
    bindings = []
    root = manifest_path.parent
    for row in rows:
        require(
            isinstance(row, dict) and set(row) == {"role", "relative_path", "bytes", "sha256"},
            "input role schema",
        )
        role = row["role"]
        require(role in ("gate", "up", "down_transposed") and role not in role_bytes, "input role")
        relative = row["relative_path"]
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "input relative path",
        )
        require(type(row["bytes"]) is int and row["bytes"] == expected_bytes, "input role bytes")
        digest = row["sha256"]
        require(isinstance(digest, str) and len(digest) == 64, "input role SHA-256 syntax")
        member = read_held_regular(
            root / relative,
            expected_bytes=expected_bytes,
            expected_sha256=digest,
            maximum_bytes=MAX_ROLE_BYTES,
            label=f"input role {role}",
        )
        _require_finite_bf16(member, f"input role {role}")
        role_bytes[role] = member
        bindings.append({"role": role, "bytes": len(member), "sha256": sha256(member)})
    require(set(role_bytes) == {"gate", "up", "down_transposed"},
            "complete inherited role ABI")
    output_name = record["output_directory_name"]
    require(
        isinstance(output_name, str)
        and output_name
        and output_name not in (".", "..")
        and "/" not in output_name
        and "\\" not in output_name,
        "output directory name",
    )
    return {
        "manifest_sha256": sha256(payload),
        "geometry": {"intermediate": intermediate, "hidden": hidden},
        "role_bytes": role_bytes,
        "bindings": bindings,
        "output_directory_name": output_name,
        "identity_fields_available_to_codec": False,
    }


def _write_member(directory_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    require(name and "/" not in name and "\\" not in name, "output member name")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        cursor = 0
        while cursor < len(payload):
            written = os.write(descriptor, payload[cursor:])
            require(written > 0, "output short write")
            cursor += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"name": name, "bytes": len(payload), "sha256": sha256(payload)}


def _rename_noreplace(
    parent_fd: int, source_name: str, destination_name: str,
) -> None:
    """Linux renameat2(RENAME_NOREPLACE), with no racy check/rename gap."""
    require(sys.platform.startswith("linux"),
            "race-free publication requires frozen Linux runtime")
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    require(renameat2 is not None, "libc renameat2 unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd, os.fsencode(source_name), parent_fd,
        os.fsencode(destination_name), 1,
    )
    if result != 0:
        observed = ctypes.get_errno()
        if observed == errno.EEXIST:
            raise DispatchError("output namespace already exists")
        raise DispatchError(
            f"renameat2(RENAME_NOREPLACE) failed: errno {observed}")


def _rehash_member_at(
    directory_fd: int, row: Mapping[str, Any], label: str,
) -> None:
    descriptor = os.open(
        row["name"], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) |
        getattr(os, "O_BINARY", 0), dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns, before.st_nlink,
        )
        require(stat.S_ISREG(before.st_mode) and
                before.st_size == row["bytes"] and before.st_nlink == 1,
                f"{label}: final member identity")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: final short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"" and
                sha256(b"".join(chunks)) == row["sha256"],
                f"{label}: final member bytes")
        after = os.fstat(descriptor)
        named = os.stat(row["name"], dir_fd=directory_fd,
                        follow_symlinks=False)
        require((after.st_dev, after.st_ino, after.st_mode, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns, after.st_nlink) ==
                identity,
                f"{label}: member changed while hashing")
        require((named.st_dev, named.st_ino, named.st_mode, named.st_size,
                 named.st_mtime_ns, named.st_ctime_ns, named.st_nlink) ==
                identity,
                f"{label}: final name/inode binding")
    finally:
        os.close(descriptor)


def publish_atomic(output_dir: Path, members: Mapping[str, bytes], completion: Mapping[str, Any]) -> dict[str, Any]:
    require(output_dir.is_absolute(), "output absolute path")
    parent = output_dir.parent
    _reject_symlink_chain(parent, "output parent")
    require(stat.S_ISDIR(os.lstat(parent).st_mode), "output parent directory")
    require("COMPLETE.json" not in members, "completion is terminal")
    parent_before = os.lstat(parent)
    parent_fd = -1
    directory_fd = -1
    final_fd = -1
    staging_created = False
    renamed = False
    staging_name = (
        f".{output_dir.name}.partial.{os.getpid()}.{secrets.token_hex(8)}")
    rows = []
    complete_row: dict[str, Any] | None = None
    try:
        parent_fd = os.open(
            os.fspath(parent), os.O_RDONLY |
            getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
        )
        parent_after = os.fstat(parent_fd)
        require((parent_before.st_dev, parent_before.st_ino,
                 stat.S_IFMT(parent_before.st_mode)) ==
                (parent_after.st_dev, parent_after.st_ino,
                 stat.S_IFMT(parent_after.st_mode)),
                "output parent inode binding")
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        staging_created = True
        try:
            directory_fd = os.open(
                staging_name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd,
            )
        except BaseException:
            # The just-created name must still be an empty directory for this
            # narrow cleanup to succeed; no recursive or foreign deletion.
            os.rmdir(staging_name, dir_fd=parent_fd)
            staging_created = False
            raise
        staging_metadata = os.fstat(directory_fd)
        staging_identity = (
            staging_metadata.st_dev, staging_metadata.st_ino,
            stat.S_IFMT(staging_metadata.st_mode),
        )
        require(len(members) == len(set(members)), "unique output members")
        for name in sorted(members, key=lambda item: item.encode("utf-8")):
            rows.append(_write_member(directory_fd, name, members[name]))
        completion_record = dict(completion)
        completion_record["members"] = rows
        completion_record["members_root_sha256"] = sha256(canonical_json(rows))
        completion_record["completion_claim_sha256"] = sha256(canonical_json(completion_record))
        completion_payload = pretty_json(completion_record)
        os.fsync(directory_fd)
        pending_complete_name = ".COMPLETE.pending"
        pending_complete_row = _write_member(
            directory_fd, pending_complete_name, completion_payload)
        os.fsync(directory_fd)
        require({entry.name for entry in os.scandir(directory_fd)} ==
                {row["name"] for row in rows} | {pending_complete_name},
                "staging exact precompletion member set")
        _rename_noreplace(parent_fd, staging_name, output_dir.name)
        renamed = True
        os.fsync(parent_fd)
        # Keep the staging FD alive through every post-rename check, then
        # independently reopen the published name and prove both FDs name the
        # same inode.
        final_fd = os.open(
            output_dir.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd,
        )
        final_metadata = os.fstat(final_fd)
        final_identity = (
            final_metadata.st_dev, final_metadata.st_ino,
            stat.S_IFMT(final_metadata.st_mode),
        )
        require(final_identity == staging_identity,
                "published directory inode rebind")
        expected_names = {row["name"] for row in rows} | {
            pending_complete_name}
        require({entry.name for entry in os.scandir(final_fd)} == expected_names,
                "published exact precompletion member set")
        for row in rows:
            _rehash_member_at(final_fd, row, f"published {row['name']}")
        _rehash_member_at(
            final_fd, pending_complete_row, "published pending completion")
        named = os.stat(output_dir.name, dir_fd=parent_fd,
                        follow_symlinks=False)
        require((named.st_dev, named.st_ino, stat.S_IFMT(named.st_mode)) ==
                staging_identity,
                "published final name/inode binding")
        absolute_named = os.stat(output_dir, follow_symlinks=False)
        require((absolute_named.st_dev, absolute_named.st_ino,
                 stat.S_IFMT(absolute_named.st_mode)) == staging_identity,
                "published absolute path/inode binding")
        # COMPLETE becomes visible only after the post-rename directory and
        # ordinary-member audit. Consumers must ignore a namespace without it.
        _rename_noreplace(
            final_fd, pending_complete_name, "COMPLETE.json")
        os.fsync(final_fd)
        complete_row = {
            **pending_complete_row, "name": "COMPLETE.json"}
        require({entry.name for entry in os.scandir(final_fd)} ==
                {row["name"] for row in rows} | {"COMPLETE.json"},
                "published terminal completed member set")
        _rehash_member_at(final_fd, complete_row, "published COMPLETE")
        named_after_complete = os.stat(
            output_dir.name, dir_fd=parent_fd, follow_symlinks=False)
        require((named_after_complete.st_dev, named_after_complete.st_ino,
                 stat.S_IFMT(named_after_complete.st_mode)) ==
                staging_identity,
                "completed output directory name/inode binding")
        os.fsync(parent_fd)
        return {
            "output_directory": os.fspath(output_dir),
            "members": rows,
            "complete": complete_row,
            "completion_published_only_after_post_rename_audit": True,
            "atomic_directory_rename": True,
            "rename_noreplace": True,
            "completion_rename_noreplace": True,
            "staging_descriptor_retained_through_final_audit": True,
            "final_name_inode_rebound": True,
            "final_members_rehashed_and_name_bound": True,
        }
    except BaseException as original:
        if staging_created and not renamed and directory_fd >= 0 and parent_fd >= 0:
            try:
                current = os.fstat(directory_fd)
                named = os.stat(staging_name, dir_fd=parent_fd,
                                follow_symlinks=False)
                require((current.st_dev, current.st_ino,
                         stat.S_IFMT(current.st_mode)) ==
                        (named.st_dev, named.st_ino,
                         stat.S_IFMT(named.st_mode)),
                        "failed staging inode ownership")
                for entry in list(os.scandir(directory_fd)):
                    require(entry.is_file(follow_symlinks=False),
                            "failed staging contains only owned regular files")
                    metadata = entry.stat(follow_symlinks=False)
                    require(metadata.st_nlink == 1,
                            "failed staging member sole link")
                    os.unlink(entry.name, dir_fd=directory_fd)
                os.fsync(directory_fd)
                os.rmdir(staging_name, dir_fd=parent_fd)
                staging_created = False
                os.fsync(parent_fd)
            except BaseException as cleanup_error:
                raise DispatchError(
                    f"publication failed and narrow staging cleanup failed: "
                    f"{cleanup_error}") from original
        raise
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION, "explicit dispatcher authorization")
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "invoke bound dispatcher with CPython -I -B")
    package_dir = Path(__file__).resolve().parent
    with authenticate_package(
        package_dir, arguments.package_manifest_sha256,
    ) as package:
        output_dir = Path(arguments.output_dir)
        require_output_outside_package(output_dir, package_dir)
        sources = package.sources
        smoke_contract = _load_authenticated_module(
            "tacn18_v6_dispatch_smoke_contract",
            sources["smoke_contract.py"],
        )
        # The source-free receipt is authenticated before CuPy initialization
        # and before the input manifest or any BF16 role path is opened.
        smoke_payload = read_held_regular(
            Path(arguments.smoke_receipt),
            expected_sha256=arguments.smoke_receipt_sha256,
            maximum_bytes=1 << 20,
            label="source-free smoke receipt",
        )
        smoke_record = _strict_json(smoke_payload, "source-free smoke receipt")
        smoke_binding = smoke_contract.validate_smoke_receipt(
            smoke_record,
            source_manifest_sha256=package.manifest_sha256,
            source_root_sha256=package.source_root_sha256,
            predecessor_lock_sha256=arguments.predecessor_lock_sha256,
            runtime_lock_sha256=arguments.runtime_lock_sha256,
            source_member_hashes=package.receipt()["member_hashes"],
        )
        runtime_module = _load_authenticated_module(
            "tacn18_v6_runtime_closure_authenticated",
            sources["runtime_closure.py"],
        )
        codec = _load_authenticated_module(
            "tacn18_v6_successor_codec_authenticated",
            sources["successor_codec.py"],
        )
        repo_root = package_dir.parents[1]
        runtime = runtime_module.load_runtime(
            repo_root,
            package_dir,
            expected_predecessor_lock_sha256=arguments.predecessor_lock_sha256,
            expected_runtime_lock_sha256=arguments.runtime_lock_sha256,
        )
        inputs = authenticate_inputs(
            Path(arguments.input_manifest),
            expected_manifest_sha256=arguments.input_manifest_sha256,
        )
        require(output_dir.name == inputs["output_directory_name"],
                "input/output namespace binding")
        geometry = runtime.packet.ExpertGeometry(
            inputs["geometry"]["intermediate"],
            inputs["geometry"]["hidden"],
        )
        frame, encoder_receipt = codec.encode_expert_frame_from_bf16_v6(
            inputs["role_bytes"], geometry, runtime)
        _reconstructions, decoder_receipt, residuals = (
            codec.decode_expert_frame_bytes_v6(
                frame, runtime, source_role_bf16=inputs["role_bytes"])
        )
        require(decoder_receipt["literal_aggregate_reencode_matches"] is True,
                "aggregate reencode gate")
        external_actual = decoder_receipt[
            "external_compressed_read_ledger"]
        require(external_actual["mode"] == "prebuffered_encoder_output" and
                external_actual["first_pass_bytes"] == 0 and
                external_actual["total_read_bytes"] == 0 and
                external_actual["reread_bytes"] == 0,
                "prebuffered decode external-read honesty")
        require(decoder_receipt["accelerator_hbm_ledger"][
            "below_2x_claim_authority"] is False,
            "no inference-HBM laundering")
        require(encoder_receipt[
            "all_encoder_self_checks_required_and_passed"] is True,
            "encoder checks gate")
        require(residuals is not None, "original-domain residuals")
        exact_rate = encoder_receipt["physical_bpw_exact"]
        require(exact_rate == decoder_receipt["physical_bpw_exact"],
                "encoder/decoder exact aggregate rate")
        if geometry.target_eligible:
            require(exact_rate["exact"] == "307/128" and
                    exact_rate["equals_307_over_128"] is True,
                    "target-eligible exact 307/128 aggregate grammar")
            status = (
                "PASS_V6_BOUND_TARGET_ELIGIBLE_FRAME_NONPROMOTING_"
                "INDEPENDENT_RESULT_AUDIT_REQUIRED")
        else:
            require(exact_rate["equals_307_over_128"] is False and
                    exact_rate["float"] > 307 / 128,
                    "tail compatibility rate is explicitly above target")
            status = (
                "PASS_V6_BOUND_COMPATIBILITY_TAIL_ABOVE_TARGET_"
                "NONPROMOTING")
        one_external_read_plan = codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=1,
            external_read_mode="one_pass_external_file",
        )
        qwen_pilot_geometry = (
            geometry.intermediate == 768 and geometry.hidden == 2048)
        result = {
            "schema": "tactic-actual-coarse-n18-v6-bound-result-v1",
            "status": status,
            "positive_claim_authority": False,
            "source_closure": package.receipt(),
            "source_free_smoke_binding": smoke_binding,
            "source_free_smoke_file_sha256": sha256(smoke_payload),
            "input_manifest_sha256": inputs["manifest_sha256"],
            "input_bindings": inputs["bindings"],
            "identity_fields_available_to_codec": False,
            "runtime_closure": runtime.receipt,
            "frame_sha256": sha256(frame),
            "frame_bytes": len(frame),
            "physical_bpw_exact": exact_rate,
            "target_eligible_exact_307_over_128": geometry.target_eligible,
            "matches_qwen_pilot_shape_only": qwen_pilot_geometry,
            "qwen_or_model_identity_used_by_codec": False,
            "universal_arbitrary_shape_below_2_5_bpw_claim": False,
            "encoder_all_self_checks_pass": True,
            "literal_aggregate_reencode_matches": True,
            "actual_prebuffered_decode_traffic": {
                "external_compressed_read": external_actual,
                "host_memory_parse_and_integrity": decoder_receipt[
                    "host_memory_parse_and_integrity_ledger"],
                "scratch_lower_bound": decoder_receipt[
                    "scratch_lower_bound_ledger"],
                "accelerator_hbm": decoder_receipt[
                    "accelerator_hbm_ledger"],
            },
            "modeled_one_external_file_read_not_executed_here":
                one_external_read_plan["external_compressed_read"],
            "original_domain_score": decoder_receipt[
                "original_domain_score"],
            "claim_boundary":
                "one bound expert coarse pilot only; no Qwen-wide, TACTIC, F, fine-code, universal-tail, below-2x inference-HBM, or compression claim without a separate sealed result audit",
        }
        members = {
            "COARSE.bin": frame,
            "ENCODER_RECEIPT.json": pretty_json(encoder_receipt),
            "DECODER_RECEIPT.json": pretty_json(decoder_receipt),
            "INPUT_BINDING.json": pretty_json({
                "schema": "tactic-actual-coarse-n18-v6-input-binding-v1",
                "manifest_sha256": inputs["manifest_sha256"],
                "geometry": inputs["geometry"],
                "roles": inputs["bindings"],
                "identity_fields_available_to_codec": False,
            }),
            "RUNTIME_RECEIPT.json": pretty_json(runtime.receipt),
            "SMOKE_BINDING.json": pretty_json({
                **smoke_binding,
                "receipt_file_sha256": sha256(smoke_payload),
            }),
            "RESULT.json": pretty_json(result),
        }
        completion = {
            "schema": COMPLETE_SCHEMA,
            "status": result["status"],
            "positive_claim_authority": False,
            "source_root_sha256": package.source_root_sha256,
            "source_free_smoke_file_sha256": sha256(smoke_payload),
            "frame_sha256": sha256(frame),
        }
        package.verify_final()
        publication = publish_atomic(output_dir, members, completion)
        package.verify_final()
        return {"result": result, "publication": publication}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--package-manifest-sha256", required=True)
    result.add_argument("--predecessor-lock-sha256", required=True)
    result.add_argument("--runtime-lock-sha256", required=True)
    result.add_argument("--smoke-receipt", type=Path, required=True)
    result.add_argument("--smoke-receipt-sha256", required=True)
    result.add_argument("--input-manifest", type=Path, required=True)
    result.add_argument("--input-manifest-sha256", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    record = run(parser().parse_args())
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
