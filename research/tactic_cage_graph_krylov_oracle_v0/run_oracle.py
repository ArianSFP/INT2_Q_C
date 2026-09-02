#!/usr/bin/env python3
"""Authenticated runner for the TACTIC-CAGE graph/Krylov ideal oracle v0."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import types
from pathlib import Path
from typing import Any, Mapping


AUTHORIZATION = "RUN_TACTIC_CAGE_GRAPH_KRYLOV_ORACLE_V0_QWEN_PILOT"
PACKAGE_SCHEMA = "tactic-cage-graph-krylov-oracle-v0-source-manifest"
PACKAGE_STATUS = "SEALED_SOURCE_ONLY_AWAITING_EXPLICIT_QWEN_PILOT"
V6_SCHEMA = "tactic-actual-coarse-n18-v6-source-manifest-v1"
V6_STATUS = "SEALED_SOURCE_ONLY_AWAITING_SOURCE_FREE_SMOKE"
V6_COMPLETE_SCHEMA = "tactic-actual-coarse-n18-v6-completion-v1"
V6_INPUT_SCHEMA = "tactic-actual-coarse-n18-v6-input-manifest-v1"
OUTPUT_SCHEMA = "tactic-cage-graph-krylov-oracle-v0-result"
COMPLETE_SCHEMA = "tactic-cage-graph-krylov-oracle-v0-completion"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class RunError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            require(key not in output, f"{label}: duplicate key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RunError(f"{label}: nonfinite {item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def _reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path.absolute()
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode),
                f"{label}: symlink component {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def read_regular_once(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    maximum_bytes: int = 256 * (1 << 20),
    label: str,
) -> bytes:
    require(path.is_absolute(), f"{label}: absolute path")
    _reject_symlink_chain(path, label)
    descriptor = os.open(
        os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                f"{label}: regular sole-link file")
        require(0 < before.st_size <= maximum_bytes, f"{label}: byte bound")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes, f"{label}: exact bytes")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: premature EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label}: trailing read")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_mode, row.st_size,
            row.st_mtime_ns, row.st_ctime_ns, row.st_nlink,
        )
        require(identity(before) == identity(after), f"{label}: identity drift")
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(HEX64.fullmatch(expected_sha256) is not None,
                    f"{label}: expected SHA syntax")
            require(sha256(payload) == expected_sha256, f"{label}: SHA-256")
        return payload
    finally:
        os.close(descriptor)


def authenticate_source_package(
    package_dir: Path,
    expected_manifest_sha256: str,
    *,
    schema: str,
    status: str,
    expected_entry: str | None,
) -> dict[str, Any]:
    package = package_dir.resolve(strict=True)
    require(package.is_dir(), "source package directory")
    _reject_symlink_chain(package, "source package")
    manifest_payload = read_regular_once(
        package / "SOURCE_MANIFEST.json",
        expected_sha256=expected_manifest_sha256,
        maximum_bytes=1 << 20,
        label="source manifest",
    )
    manifest = strict_json(manifest_payload, "source manifest")
    require(manifest.get("schema") == schema and manifest.get("status") == status,
            "source manifest schema/status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "source manifest members")
    names: list[str] = []
    observed: list[dict[str, Any]] = []
    sources: dict[str, bytes] = {}
    for row in rows:
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"},
                "source manifest member row")
        name = row["name"]
        require(isinstance(name, str) and name and
                name != "SOURCE_MANIFEST.json" and
                "/" not in name and "\\" not in name and name not in names,
                "safe unique source member")
        payload = read_regular_once(
            package / name,
            expected_bytes=row["bytes"], expected_sha256=row["sha256"],
            maximum_bytes=4 * (1 << 20), label=f"source member {name}",
        )
        names.append(name)
        sources[name] = payload
        observed.append({"name": name, "bytes": len(payload),
                         "sha256": sha256(payload)})
    require(names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical source member order")
    require(manifest.get("source_root_sha256") == sha256(canonical_json(observed)),
            "source snapshot root")
    actual = list(os.scandir(package))
    require({entry.name for entry in actual} == set(names) | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in actual),
            "exact source package closure")
    if expected_entry is not None:
        require(expected_entry in sources, "executing entry in source closure")
        entry_meta = os.lstat(Path(__file__).resolve())
        named_meta = os.lstat(package / expected_entry)
        require((entry_meta.st_dev, entry_meta.st_ino) ==
                (named_meta.st_dev, named_meta.st_ino),
                "executing entry inode binding")
    return {
        "path": package,
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_payload),
        "source_root_sha256": manifest["source_root_sha256"],
        "sources": sources,
        "members": observed,
    }


def load_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, f"module collision: {name}")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{sha256(source)}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = sha256(source)
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0),
             module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def authenticate_v6_result(
    result_dir: Path, expected_complete_sha256: str,
) -> dict[str, Any]:
    root = result_dir.resolve(strict=True)
    require(root.is_dir(), "v6 result directory")
    _reject_symlink_chain(root, "v6 result")
    complete_payload = read_regular_once(
        root / "COMPLETE.json", expected_sha256=expected_complete_sha256,
        maximum_bytes=1 << 20, label="v6 COMPLETE",
    )
    complete = strict_json(complete_payload, "v6 COMPLETE")
    required = {
        "schema", "status", "positive_claim_authority", "source_root_sha256",
        "source_free_smoke_file_sha256", "frame_sha256", "members",
        "members_root_sha256", "completion_claim_sha256",
    }
    require(set(complete) == required, "v6 COMPLETE exact schema")
    require(complete["schema"] == V6_COMPLETE_SCHEMA and
            complete["positive_claim_authority"] is False,
            "v6 COMPLETE schema/authority")
    claim = dict(complete)
    claimed_digest = claim.pop("completion_claim_sha256")
    require(claimed_digest == sha256(canonical_json(claim)),
            "v6 completion claim")
    rows = complete["members"]
    require(isinstance(rows, list) and rows, "v6 result members")
    require(complete["members_root_sha256"] == sha256(canonical_json(rows)),
            "v6 members root")
    required_names = {
        "COARSE.bin", "ENCODER_RECEIPT.json", "DECODER_RECEIPT.json",
        "INPUT_BINDING.json", "RUNTIME_RECEIPT.json", "SMOKE_BINDING.json",
        "RESULT.json",
    }
    names: list[str] = []
    payloads: dict[str, bytes] = {}
    observed: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"},
                "v6 result member row")
        name = row["name"]
        require(isinstance(name, str) and name in required_names and name not in names,
                "v6 result safe unique member")
        # COARSE.bin is opened exactly here and remains buffered. There is no
        # subsequent compressed expert-frame file access.
        payload = read_regular_once(
            root / name, expected_bytes=row["bytes"],
            expected_sha256=row["sha256"], maximum_bytes=32 * (1 << 20),
            label=f"v6 result {name}",
        )
        names.append(name)
        payloads[name] = payload
        observed.append({"name": name, "bytes": len(payload),
                         "sha256": sha256(payload)})
    require(set(names) == required_names, "v6 exact required result members")
    require(observed == rows, "v6 member order/content binding")
    actual = list(os.scandir(root))
    require({entry.name for entry in actual} == required_names | {"COMPLETE.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in actual),
            "v6 exact terminal result closure")
    result = strict_json(payloads["RESULT.json"], "v6 RESULT")
    decoder = strict_json(payloads["DECODER_RECEIPT.json"], "v6 decoder receipt")
    input_binding = strict_json(payloads["INPUT_BINDING.json"], "v6 input binding")
    runtime_receipt = strict_json(payloads["RUNTIME_RECEIPT.json"], "v6 runtime receipt")
    frame = payloads["COARSE.bin"]
    require(result.get("schema") == "tactic-actual-coarse-n18-v6-bound-result-v1",
            "v6 result schema")
    require(result.get("positive_claim_authority") is False,
            "v6 result nonpromoting authority")
    require(result.get("frame_sha256") == sha256(frame) == complete["frame_sha256"],
            "v6 result frame binding")
    require(result.get("physical_bpw_exact", {}).get("exact") == "307/128" and
            result.get("target_eligible_exact_307_over_128") is True,
            "v6 exact target-eligible coarse rate")
    require(result.get("matches_qwen_pilot_shape_only") is True and
            result.get("qwen_or_model_identity_used_by_codec") is False,
            "v6 shape-only universal source contract")
    require(decoder.get("frame_sha256") == sha256(frame) and
            decoder.get("literal_aggregate_reencode_matches") is True,
            "v6 decoder/frame binding")
    require(input_binding.get("schema") ==
            "tactic-actual-coarse-n18-v6-input-binding-v1",
            "v6 input binding schema")
    return {
        "path": root,
        "complete": complete,
        "complete_sha256": sha256(complete_payload),
        "members": observed,
        "payloads": payloads,
        "result": result,
        "decoder_receipt": decoder,
        "input_binding": input_binding,
        "runtime_receipt": runtime_receipt,
        "frame": frame,
        "compressed_frame_file_read_count": 1,
    }


def _finite_bf16(raw: bytes, label: str) -> None:
    require(len(raw) % 2 == 0, f"{label}: BF16 alignment")
    for offset in range(0, len(raw), 2):
        word = raw[offset] | (raw[offset + 1] << 8)
        require((word & 0x7F80) != 0x7F80, f"{label}: finite BF16")


def authenticate_inputs(
    manifest_path: Path, expected_manifest_sha256: str,
    v6_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = manifest_path.resolve(strict=True)
    payload = read_regular_once(
        manifest, expected_sha256=expected_manifest_sha256,
        maximum_bytes=1 << 20, label="v6 input manifest",
    )
    record = strict_json(payload, "v6 input manifest")
    require(set(record) == {"schema", "geometry", "roles", "output_directory_name"}
            and record["schema"] == V6_INPUT_SCHEMA,
            "v6 input exact identity-free schema")
    geometry = record["geometry"]
    require(isinstance(geometry, dict) and set(geometry) == {"intermediate", "hidden"}
            and geometry == {"intermediate": 768, "hidden": 2048},
            "frozen Qwen pilot geometry only")
    expected_bytes = 2 * 768 * 2048
    rows = record["roles"]
    require(isinstance(rows, list) and len(rows) == 3, "input role rows")
    roles: dict[str, bytes] = {}
    bindings: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict) and
                set(row) == {"role", "relative_path", "bytes", "sha256"},
                "input role row")
        role = row["role"]
        require(role in {"gate", "up", "down_transposed"} and role not in roles,
                "input role ABI")
        relative = row["relative_path"]
        require(isinstance(relative, str) and relative and
                not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "input safe relative path")
        require(row["bytes"] == expected_bytes and
                HEX64.fullmatch(str(row["sha256"])) is not None,
                "input role bytes/hash")
        raw = read_regular_once(
            manifest.parent / relative, expected_bytes=expected_bytes,
            expected_sha256=row["sha256"], maximum_bytes=8 * (1 << 20),
            label=f"input BF16 {role}",
        )
        _finite_bf16(raw, role)
        roles[role] = raw
        bindings.append({"role": role, "bytes": len(raw), "sha256": sha256(raw)})
    require(set(roles) == {"gate", "up", "down_transposed"},
            "complete role ABI")
    require(v6_input_binding.get("manifest_sha256") == sha256(payload) and
            v6_input_binding.get("geometry") == geometry and
            v6_input_binding.get("roles") == bindings and
            v6_input_binding.get("identity_fields_available_to_codec") is False,
            "BF16 triplet equals v6 authenticated input binding")
    return {
        "manifest_sha256": sha256(payload), "geometry": geometry,
        "roles": roles, "bindings": bindings,
    }


def bf16_f64(raw: bytes, np: Any) -> Any:
    words = np.frombuffer(raw, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    require(bool(np.all(np.isfinite(values))), "finite BF16 conversion")
    return values


def replay_v6_once(
    frame: bytes, role_bytes: Mapping[str, bytes], runtime: Any, codec: Any,
    published_decoder: Mapping[str, Any], np: Any,
) -> dict[str, Any]:
    packet = runtime.packet
    require(len(frame) == 18 * packet.RESERVOIR_BYTES,
            "Qwen pilot exact 18-record frame")
    decoded = []
    for ordinal in range(18):
        begin = ordinal * packet.RESERVOIR_BYTES
        # Every packet slice comes from the one buffered frame. No path is
        # passed to the decoder and no compressed file can be refetched.
        row = codec.decode_tile_v6(
            frame[begin: begin + packet.RESERVOIR_BYTES], runtime)
        expected_role, expected_tile = divmod(ordinal, 6)
        require((row.parsed.role_ordinal, row.parsed.tile_ordinal) ==
                (expected_role, expected_tile), "canonical role/tile order")
        require(row.parsed.valid_values == packet.N and not row.parsed.zero_tile,
                "Qwen pilot full nonzero N18 records")
        decoded.append(row)
    reencoded = b"".join(row.canonical_packet for row in decoded)
    require(reencoded == frame, "literal aggregate reencode from buffered frame")
    role_names = tuple(packet.ROLES)
    require(role_names == ("gate", "up", "down_transposed"), "v6 role ABI")
    residual_records: list[dict[str, Any]] = []
    source_energy = 0.0
    source_sse = 0.0
    symbol_hashes: dict[str, str] = {}
    reconstruction_hashes: dict[str, str] = {}
    residual_hashes: dict[str, str] = {}
    for role_ordinal, role in enumerate(role_names):
        source = bf16_f64(role_bytes[role], np)
        rows = decoded[role_ordinal * 6: (role_ordinal + 1) * 6]
        reconstruction = np.concatenate([row.reconstruction_f32 for row in rows])
        symbols = np.concatenate([row.canonical_symbols_i32 for row in rows])
        require(reconstruction.size == source.size == 768 * 2048,
                "role reconstruction/source geometry")
        require(symbols.size == source.size and symbols.dtype.str == "<i4",
                "role canonical symbol geometry/dtype")
        residual = source - reconstruction.astype(np.float64)
        sse = float(np.dot(residual, residual))
        energy = float(np.dot(source, source))
        require(math.isfinite(sse) and math.isfinite(energy) and energy > 0.0,
                "finite positive source score")
        source_sse += sse
        source_energy += energy
        symbol_hashes[role] = sha256(symbols.astype("<i4", copy=False).tobytes())
        reconstruction_hashes[role] = sha256(
            reconstruction.astype("<f4", copy=False).tobytes())
        residual_hashes[role] = sha256(residual.astype("<f8", copy=False).tobytes())
        for tile in range(6):
            begin = tile * packet.N
            end = begin + packet.N
            residual_records.append({
                "ordinal": role_ordinal * 6 + tile,
                "role": role,
                "role_ordinal": role_ordinal,
                "tile_ordinal": tile,
                "residual_f64": residual[begin:end].copy(),
                "symbols_i32": symbols[begin:end].copy(),
                "sse_fp64": float(np.dot(residual[begin:end], residual[begin:end])),
            })
    require(symbol_hashes == published_decoder.get("canonical_symbols_i32_sha256"),
            "published canonical symbol digests")
    published_score = published_decoder.get("original_domain_score")
    require(isinstance(published_score, dict), "published original score")
    require(float(published_score["pooled_sse_fp64"]) == source_sse and
            float(published_score["pooled_source_energy_fp64"]) == source_energy,
            "exact published FP64 pooled score")
    for row in published_score["roles"]:
        role = row["role"]
        require(row["reconstruction_f32_sha256"] == reconstruction_hashes[role] and
                row["residual_f64_sha256"] == residual_hashes[role],
                "published role reconstruction/residual hashes")
    return {
        "decoded_rows": decoded,
        "residual_records": residual_records,
        "pooled_sse_fp64": source_sse,
        "pooled_source_energy_fp64": source_energy,
        "pooled_relative_mse": source_sse / source_energy,
        "canonical_symbols_i32_sha256": symbol_hashes,
        "reconstruction_f32_sha256": reconstruction_hashes,
        "residual_f64_sha256": residual_hashes,
        "literal_aggregate_reencode_matches": True,
        "compressed_frame_storage_reads": 1,
        "decoded_coarse_state_buffered_once": True,
    }


def _score_records(
    records: list[dict[str, Any]], core: Any, cp: Any, dct: Any,
    *, control: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    accumulator = core.new_accumulator()
    per_record: list[dict[str, Any]] = []
    control_receipts: list[dict[str, Any]] = []
    for record in records:
        residual = cp.asarray(
            record["residual_f64"].reshape(-1, core.BLOCK_VALUES),
            dtype=cp.float64)
        symbols = cp.asarray(
            record["symbols_i32"].reshape(-1, core.BLOCK_VALUES),
            dtype=cp.int32)
        global_base = int(record["ordinal"]) * (len(record["residual_f64"]) // core.BLOCK_VALUES)
        if control == "permutation":
            residual, receipt = core.affine_permutation_control(
                residual, global_base, cp)
            control_receipts.append(receipt)
        elif control == "gaussian":
            residual, receipt = core.gaussian_moment_control(
                residual, global_base, cp)
            control_receipts.append(receipt)
        elif control is not None:
            raise RunError("unknown control")
        family_rows = []
        for family in core.ALL_FAMILIES:
            score = core.score_family(residual, symbols, family, cp, dct)
            core.add_score(accumulator, score)
            family_rows.append({
                "family": family,
                "input_sse_fp64": score["input_sse_fp64"],
                "fixed_remaining_sse_fp64": score[
                    "fixed_first384_exact_amplitudes_free_remaining_sse_fp64"],
                "top384_remaining_sse_fp64": score[
                    "free_support_top384_exact_amplitudes_free_remaining_sse_fp64"],
                "waterfill_remaining_sse_fp64": score[
                    "ideal_384bit_gaussian_waterfill_remaining_sse_fp64"],
                "parseval_max_abs_error": score["graph_receipt"][
                    "maximum_fp64_parseval_abs_error"],
            })
        per_record.append({
            "ordinal": record["ordinal"], "role": record["role"],
            "tile_ordinal": record["tile_ordinal"],
            "blocks": len(record["residual_f64"]) // core.BLOCK_VALUES,
            "families": family_rows,
        })
        del residual, symbols
        cp.get_default_memory_pool().free_all_blocks()
    return accumulator, per_record, control_receipts


def _ensure_disjoint_output(output: Path, forbidden: list[Path]) -> None:
    require(output.is_absolute() and not output.exists(),
            "output absolute and absent")
    parent = output.parent.resolve(strict=True)
    target = parent / output.name
    require(output.name not in {"", ".", ".."} and "/" not in output.name
            and "\\" not in output.name, "safe output name")
    for path in forbidden:
        resolved = path.resolve(strict=True)
        require(target != resolved and resolved not in target.parents and
                target not in resolved.parents,
                "output disjoint from immutable input/source")


def _write_exclusive(directory_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    require(name and "/" not in name and "\\" not in name, "safe output member")
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            require(written > 0, "output short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"name": name, "bytes": len(payload), "sha256": sha256(payload)}


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    require(sys.platform.startswith("linux"), "renameat2 Linux runtime")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    # x86_64 Linux, pinned by the v6 runtime lock.
    result = syscall(
        ctypes.c_long(316), ctypes.c_int(parent_fd), os.fsencode(source),
        ctypes.c_int(parent_fd), os.fsencode(destination), ctypes.c_uint(1))
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def publish_atomic(output: Path, members: Mapping[str, bytes], completion: Mapping[str, Any]) -> dict[str, Any]:
    parent = output.parent.resolve(strict=True)
    parent_fd = os.open(os.fspath(parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    stage = f".{output.name}.partial.{os.getpid()}.{secrets.token_hex(8)}"
    stage_fd = -1
    renamed = False
    try:
        os.mkdir(stage, 0o700, dir_fd=parent_fd)
        stage_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                           | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        rows = []
        for name in sorted(members, key=lambda value: value.encode("utf-8")):
            rows.append(_write_exclusive(stage_fd, name, members[name]))
        complete = dict(completion)
        complete["members"] = rows
        complete["members_root_sha256"] = sha256(canonical_json(rows))
        complete["completion_claim_sha256"] = sha256(canonical_json(complete))
        complete_payload = pretty_json(complete)
        complete_row = _write_exclusive(stage_fd, "COMPLETE.json", complete_payload)
        os.fsync(stage_fd)
        require({entry.name for entry in os.scandir(stage_fd)} ==
                set(members) | {"COMPLETE.json"}, "staged exact output closure")
        _rename_noreplace(parent_fd, stage, output.name)
        renamed = True
        os.fsync(parent_fd)
        return {
            "output_directory": os.fspath(output), "members": rows,
            "complete": complete_row, "atomic_directory_rename": True,
            "rename_noreplace": True,
        }
    finally:
        if not renamed:
            try:
                require(stage_fd >= 0, "owned staging descriptor")
                entries = list(os.scandir(stage_fd))
                for entry in entries:
                    metadata = entry.stat(follow_symlinks=False)
                    require(entry.is_file(follow_symlinks=False) and
                            metadata.st_nlink == 1,
                            "owned staging contains sole-link regular files")
                    os.unlink(entry.name, dir_fd=stage_fd)
                os.fsync(stage_fd)
            except BaseException:
                pass
        if stage_fd >= 0:
            os.close(stage_fd)
        if not renamed:
            try:
                # Only remove an empty directory whose random private name was
                # created in this call. Material files remain for forensics if
                # narrow cleanup cannot prove emptiness.
                os.rmdir(stage, dir_fd=parent_fd)
            except BaseException:
                pass
        os.close(parent_fd)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION, "explicit authorization")
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "invoke with CPython -I -B")
    package_dir = Path(__file__).resolve().parent
    own = authenticate_source_package(
        package_dir, arguments.package_manifest_sha256,
        schema=PACKAGE_SCHEMA, status=PACKAGE_STATUS,
        expected_entry="run_oracle.py")
    core = load_module("tactic_cage_graph_krylov_oracle_v0_core",
                       own["sources"]["oracle_core.py"])
    secondary = load_module("tactic_cage_graph_krylov_oracle_v0_secondary",
                            own["sources"]["secondary_hooks.py"])
    require(core.exact_budget_record() == {
        "coarse_bpw_exact": "307/128", "cap_bpw_exact": "5/2",
        "remaining_bpw_exact": "13/128", "remaining_bits_per_4096": 416,
        "oracle_fine_bits_per_4096": 384,
        "reserved_noncoefficient_bits_per_4096": 32,
    }, "frozen exact budget")

    v6_package = authenticate_source_package(
        Path(arguments.v6_package), arguments.v6_package_manifest_sha256,
        schema=V6_SCHEMA, status=V6_STATUS, expected_entry=None)
    v6_result = authenticate_v6_result(
        Path(arguments.v6_result_dir), arguments.v6_complete_sha256)
    require(v6_result["complete"]["source_root_sha256"] ==
            v6_package["source_root_sha256"] ==
            v6_result["result"]["source_closure"]["source_root_sha256"],
            "v6 result/source package root binding")
    inputs = authenticate_inputs(
        Path(arguments.input_manifest), arguments.input_manifest_sha256,
        v6_result["input_binding"])

    runtime_module = load_module(
        "tactic_cage_authenticated_v6_runtime",
        v6_package["sources"]["runtime_closure.py"])
    codec = load_module(
        "tactic_cage_authenticated_v6_codec",
        v6_package["sources"]["successor_codec.py"])
    repo_root = v6_package["path"].parents[1]
    runtime = runtime_module.load_runtime(
        repo_root, v6_package["path"],
        expected_predecessor_lock_sha256=arguments.v6_predecessor_lock_sha256,
        expected_runtime_lock_sha256=arguments.v6_runtime_lock_sha256)
    np = runtime.numpy
    cp = runtime.cupy
    from cupyx.scipy.fft import dct

    replay = replay_v6_once(
        v6_result["frame"], inputs["roles"], runtime, codec,
        v6_result["decoder_receipt"], np)
    require(v6_result["compressed_frame_file_read_count"] == 1 and
            replay["compressed_frame_storage_reads"] == 1,
            "one compressed expert-frame storage read")

    source_accumulator, source_per_record, _ = _score_records(
        replay["residual_records"], core, cp, dct, control=None)
    source_metrics = core.finalize_accumulator(
        source_accumulator, replay["pooled_source_energy_fp64"])
    source_decision = core.source_gate(source_metrics)

    controls_opened = bool(source_decision["controls_may_open"])
    control_metrics: dict[str, Any] = {}
    control_records: dict[str, Any] = {}
    control_receipts: dict[str, Any] = {}
    final_decision: dict[str, Any]
    if controls_opened:
        for control in ("permutation", "gaussian"):
            accumulator, rows, receipts = _score_records(
                replay["residual_records"], core, cp, dct, control=control)
            control_metrics[control] = core.finalize_accumulator(
                accumulator, replay["pooled_source_energy_fp64"])
            control_records[control] = rows
            control_receipts[control] = receipts
        final_decision = core.controls_gate(
            source_metrics, control_metrics,
            source_decision["best_waterfill_family"])
    else:
        final_decision = {
            "status": source_decision["status"],
            "controls_not_opened_due_to_source_hard_kill": True,
            "composite_gap_eligible": False,
        }

    cp.cuda.Stream.null.synchronize()
    memory_pool_used = int(cp.get_default_memory_pool().used_bytes())
    frame_bytes = len(v6_result["frame"])
    values = 3 * 768 * 2048
    symbol_bytes = 18 * (1 << 18) * 4
    reconstruction_bytes = values * 4
    residual_bytes = values * 8
    page_bytes = ((frame_bytes + 4095) // 4096) * 4096
    traffic = {
        "schema": "tactic-cage-graph-krylov-oracle-v0-traffic-ledger",
        "external_compressed_expert_storage": {
            "coarse_frame_bytes": frame_bytes,
            "file_open_read_count": 1,
            "first_pass_bytes": frame_bytes,
            "reread_bytes": 0,
            "total_read_bytes": frame_bytes,
            "byte_read_amplification": 1.0,
            "page_rounded_first_pass_bytes": page_bytes,
            "page_rounded_amplification": page_bytes / frame_bytes,
            "compressed_frame_refetch": False,
        },
        "host_parse_and_buffer_lower_bound": {
            "buffered_coarse_frame_bytes": frame_bytes,
            "causal_packet_decode_input_bytes": frame_bytes,
            "aggregate_reencode_compare_bytes": frame_bytes,
            "canonical_symbols_i32_bytes": symbol_bytes,
            "reconstruction_f32_bytes": reconstruction_bytes,
            "residual_f64_bytes": residual_bytes,
            "decoded_coarse_state_buffered_once": True,
            "compressed_storage_read_equivalent": 1,
            "python_object_and_decoder_internal_scratch_measured": False,
        },
        "accelerator_hbm": {
            "cupy_required_for_heavy_graph_work": True,
            "memory_pool_used_bytes_after_synchronization": memory_pool_used,
            "kernel_read_write_bytes_measured": False,
            "candidate_graph_passes": len(core.ALL_FAMILIES),
            "control_graph_passes_if_opened":
                2 * len(core.ALL_FAMILIES) if controls_opened else 0,
            "below_2x_inference_hbm_claim_authority": False,
        },
        "inference_read_claim_authority": False,
        "oracle_execution_is_offline_analysis": True,
    }

    coarse_relative = replay["pooled_relative_mse"]
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": final_decision["status"],
        "positive_claim_authority": False,
        "finite_codec_executed": False,
        "continuous_ideal_containment_oracle": True,
        "graph_basis_free_only_because_regenerated_from_literal_coarse_word": True,
        "source_or_residual_used_to_build_graph": False,
        "source_or_residual_used_by_oracle_projection_and_model_selection": True,
        "budget": core.exact_budget_record(),
        "target_relative_mse": core.TARGET_RELATIVE_MSE,
        "coarse_pooled_relative_mse": coarse_relative,
        "required_coarse_error_capture_fraction": core.required_capture(coarse_relative),
        "source_metrics": source_metrics,
        "source_per_record": source_per_record,
        "source_gate": source_decision,
        "controls_opened_after_source_survival": controls_opened,
        "control_metrics": control_metrics,
        "control_per_record": control_records,
        "control_receipts": control_receipts,
        "final_gate": final_decision,
        "one_pass_and_memory_traffic": traffic,
        "secondary_screen_routing_not_executed":
            secondary.routing_record(final_decision["status"]),
        "claim_boundary": (
            "One continuous/ideal containment oracle on one authenticated "
            "shape-bound expert. No finite coefficient/support packet, physical "
            "2.5-bpw result, F<=0.8 result, routed inference-HBM result, "
            "Qwen-wide transfer, or universal SwiGLU-MoE claim."
        ),
    }
    provenance = {
        "schema": "tactic-cage-graph-krylov-oracle-v0-provenance",
        "oracle_source_manifest_sha256": own["manifest_sha256"],
        "oracle_source_root_sha256": own["source_root_sha256"],
        "v6_source_manifest_sha256": v6_package["manifest_sha256"],
        "v6_source_root_sha256": v6_package["source_root_sha256"],
        "v6_predecessor_lock_sha256": arguments.v6_predecessor_lock_sha256,
        "v6_runtime_lock_sha256": arguments.v6_runtime_lock_sha256,
        "v6_result_complete_sha256": v6_result["complete_sha256"],
        "v6_result_members_root_sha256":
            v6_result["complete"]["members_root_sha256"],
        "v6_result_members": v6_result["members"],
        "v6_frame_sha256": sha256(v6_result["frame"]),
        "input_manifest_sha256": inputs["manifest_sha256"],
        "input_role_bindings": inputs["bindings"],
        "runtime_closure": runtime.receipt,
        "literal_aggregate_reencode_matches":
            replay["literal_aggregate_reencode_matches"],
        "canonical_symbols_i32_sha256":
            replay["canonical_symbols_i32_sha256"],
        "reconstruction_f32_sha256": replay["reconstruction_f32_sha256"],
        "residual_f64_sha256": replay["residual_f64_sha256"],
        "compressed_expert_frame_file_read_count": 1,
        "compressed_expert_frame_refetch": False,
    }
    output = Path(arguments.output_dir)
    _ensure_disjoint_output(output, [package_dir, v6_package["path"], v6_result["path"]])
    # Re-authenticate immutable source closures before publishing; this does
    # not touch the buffered compressed expert frame or any payload file.
    own_final = authenticate_source_package(
        package_dir, arguments.package_manifest_sha256,
        schema=PACKAGE_SCHEMA, status=PACKAGE_STATUS,
        expected_entry="run_oracle.py")
    require(own_final["source_root_sha256"] == own["source_root_sha256"],
            "oracle source stable through run")
    publication = publish_atomic(output, {
        "PROVENANCE.json": pretty_json(provenance),
        "RESULT.json": pretty_json(result),
    }, {
        "schema": COMPLETE_SCHEMA,
        "status": result["status"],
        "positive_claim_authority": False,
        "finite_codec_executed": False,
        "oracle_source_root_sha256": own["source_root_sha256"],
        "v6_frame_sha256": sha256(v6_result["frame"]),
        "input_manifest_sha256": inputs["manifest_sha256"],
    })
    return {"result": result, "provenance": provenance,
            "publication": publication}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--authorization", required=True)
    value.add_argument("--package-manifest-sha256", required=True)
    value.add_argument("--v6-package", type=Path, required=True)
    value.add_argument("--v6-package-manifest-sha256", required=True)
    value.add_argument("--v6-predecessor-lock-sha256", required=True)
    value.add_argument("--v6-runtime-lock-sha256", required=True)
    value.add_argument("--v6-result-dir", type=Path, required=True)
    value.add_argument("--v6-complete-sha256", required=True)
    value.add_argument("--input-manifest", type=Path, required=True)
    value.add_argument("--input-manifest-sha256", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
