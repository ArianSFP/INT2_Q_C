#!/usr/bin/env python3
"""Manifest-bound, source-free RTX smoke for every numerical v6 repair.

Import is standard-library-only and inert. The package manifest and retained
source descriptors are authenticated before NumPy, CuPy, or predecessor code
is loaded. The receipt is written outside the immutable source package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import struct
import sys
import types
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            require(key not in output, f"{label}: duplicate key")
            output[key] = value
        return output

    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            RuntimeError(f"{label}: nonfinite {item}")
        ),
    )
    require(isinstance(value, dict), f"{label}: object")
    return value


def read_nofollow(path: Path, maximum: int, label: str) -> bytes:
    require(path.is_absolute(), f"{label}: absolute path")
    descriptor = os.open(
        os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) |
        getattr(os, "O_BINARY", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and
                0 < before.st_size <= maximum,
                f"{label}: regular byte bound")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label}: trailing bytes")
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_mode,
                 before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_mode,
                 after.st_size, after.st_mtime_ns),
                f"{label}: identity drift")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, f"module collision: {name}")
    digest = sha256(source)
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = digest
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True,
                     optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def bootstrap_source_auth(package_dir: Path, manifest_sha256: str) -> Any:
    """Authenticate source_auth.py before executing its retained closure."""
    manifest_payload = read_nofollow(
        package_dir / "SOURCE_MANIFEST.json", 1 << 20, "source manifest")
    require(sha256(manifest_payload) == manifest_sha256,
            "external source-manifest digest")
    manifest = strict_json(manifest_payload, "source manifest")
    require(manifest.get("schema") ==
            "tactic-actual-coarse-n18-v6-source-manifest-v1",
            "source manifest schema")
    rows = manifest.get("members")
    require(isinstance(rows, list), "source manifest rows")
    matches = [row for row in rows if isinstance(row, dict) and
               row.get("name") == "source_auth.py"]
    require(len(matches) == 1, "source_auth manifest row")
    row = matches[0]
    source = read_nofollow(
        package_dir / "source_auth.py", 4 * (1 << 20), "source_auth")
    require(len(source) == row.get("bytes") and
            sha256(source) == row.get("sha256"),
            "source_auth manifest binding")
    return load_module("tacn18_v6_smoke_source_auth", source)


def fp32_to_bf16(value: float) -> int:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    upper = word >> 16
    lower = word & 0xFFFF
    increment = lower > 0x8000 or (lower == 0x8000 and upper & 1)
    return (upper + int(increment)) & 0xFFFF


def fixture(values: int) -> bytes:
    rng = random.Random(0x5441434E31385636)
    output = bytearray(2 * values)
    for index in range(values):
        struct.pack_into(
            "<H", output, 2 * index, fp32_to_bf16(rng.gauss(0.0, 1.0)))
    return bytes(output)


def run(
    repo_root: Path,
    package_dir: Path,
    package_manifest_sha: str,
    predecessor_sha: str,
    runtime_sha: str,
    receipt_output: Path,
) -> tuple[dict[str, object], dict[str, Any], bytes]:
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "invoke source-free smoke with CPython -I -B")
    package_dir = package_dir.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    auth = bootstrap_source_auth(package_dir, package_manifest_sha)
    with auth.HeldSourcePackage(
        package_dir, package_manifest_sha,
        executing_path=Path(__file__).resolve(strict=True),
    ) as package:
        sources = package.sources
        runtime_module = load_module(
            "tacn18_v6_smoke_runtime_closure", sources["runtime_closure.py"])
        codec = load_module(
            "tacn18_v6_smoke_successor_codec", sources["successor_codec.py"])
        smoke_contract = load_module(
            "tacn18_v6_smoke_contract", sources["smoke_contract.py"])
        runtime = runtime_module.load_runtime(
            repo_root,
            package_dir,
            expected_predecessor_lock_sha256=predecessor_sha,
            expected_runtime_lock_sha256=runtime_sha,
        )
        geometry = runtime.packet.ExpertGeometry(runtime.packet.N, 1)
        source = fixture(runtime.packet.N)
        packet, encode = codec.encode_tile_v6(
            source, geometry, 0, 0, runtime)
        decoded = codec.decode_tile_v6(packet, runtime)
        require(decoded.canonical_packet == packet,
                "numeric tile canonical equality")
        require(encode["all_encoder_self_checks_required_and_passed"] is True,
                "numeric encoder checks")
        require(decoded.report[
            "inverse_i32_dtype_verified_before_facade_cast"] is True,
            "normal decode I32 lifetime before facade")
        original = runtime.numpy.frombuffer(
            source, dtype="<u2").astype(runtime.numpy.uint32)
        original = ((original << runtime.numpy.uint32(16))
                    .view(runtime.numpy.float32).astype(runtime.numpy.float64))
        residual = original - decoded.reconstruction_f32.astype(
            runtime.numpy.float64)
        relative_mse = float(runtime.numpy.dot(residual, residual) /
                             runtime.numpy.dot(original, original))

        # A direct worst-case Hadamard input forces the exact mathematical
        # 8,388,608 maximum. This exceeds I16 before the facade sees it.
        stress_indices = runtime.numpy.full(
            runtime.packet.N, 63, dtype=runtime.numpy.int16)
        require(runtime.independent_decoder._integer_inverse_symbols is
                runtime.inverse_symbols_i32,
                "stress uses the override installed in inherited decoder")
        inverse_output = runtime.independent_decoder._integer_inverse_symbols(
            runtime.numpy, stress_indices, 0)
        require(inverse_output.dtype.str == "<i4",
                "stress inverse output is native I32")
        stress_max = int(runtime.numpy.max(runtime.numpy.abs(
            inverse_output.astype(runtime.numpy.int64))))
        require(stress_max == 32 * runtime.packet.N and stress_max > 32_767,
                "stress exceeds I16 with exact bound")
        retained = codec.retain_canonical_symbols_i32(
            inverse_output, runtime.numpy)
        require(retained is inverse_output or bool(runtime.numpy.shares_memory(
            retained, inverse_output)), "stress I32 no-copy lifetime")
        downstream = retained.astype(runtime.numpy.float64) * (
            runtime.packet.ETA / runtime.packet.SQRT_N)
        downstream_max = float(runtime.numpy.max(runtime.numpy.abs(downstream)))
        expected_downstream_max = float(stress_max) * (
            runtime.packet.ETA / runtime.packet.SQRT_N)
        require(downstream_max == expected_downstream_max,
                "stress inherited reconstruction-F64 lifetime")

        # Exact inherited ABI, including the transposed Down spelling.
        zero = bytes(2 * runtime.packet.N)
        zero_frame, zero_encode = codec.encode_expert_frame_from_bf16_v6(
            {"gate": zero, "up": zero, "down_transposed": zero},
            geometry, runtime,
        )
        _recon, zero_decode, _residual = codec.decode_expert_frame_bytes_v6(
            zero_frame, runtime)
        require(zero_decode["literal_aggregate_reencode_matches"] is True,
                "aggregate frame equality")
        prebuffered = codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=0,
            external_read_mode="prebuffered_encoder_output",
        )
        require(
            zero_decode["external_compressed_read_ledger"] ==
            prebuffered["external_compressed_read"] and
            zero_decode["host_memory_parse_and_integrity_ledger"] ==
            prebuffered["host_memory_parse_and_integrity"] and
            zero_decode["scratch_lower_bound_ledger"] ==
            prebuffered["scratch_lower_bound"] and
            zero_decode["accelerator_hbm_ledger"] ==
            prebuffered["accelerator_hbm"],
            "decoder receipt binds complete frame ledger",
        )
        one = codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=1,
            external_read_mode="one_pass_external_file",
        )
        two = codec.frame_ledger_v6(
            geometry, external_compressed_read_passes=2,
            external_read_mode="modeled_external_file_reread",
        )
        require(prebuffered["external_compressed_read"]["first_pass_bytes"] == 0 and
                prebuffered["external_compressed_read"]["total_read_bytes"] == 0 and
                prebuffered["external_compressed_read"]["reread_bytes"] == 0,
                "prebuffered external-read honesty")
        require(one["external_compressed_read"]["first_pass_bytes"] == len(zero_frame) and
                one["external_compressed_read"]["total_read_bytes"] == len(zero_frame) and
                one["external_compressed_read"]["reread_bytes"] == 0,
                "one external pass ledger")
        require(two["external_compressed_read"]["total_read_bytes"] == 2 * len(zero_frame) and
                two["external_compressed_read"]["reread_bytes"] == len(zero_frame),
                "external reread ledger")

        record: dict[str, Any] = {
            "schema": smoke_contract.SCHEMA,
            "status": smoke_contract.STATUS,
            "source_closure": package.receipt(),
            "runtime_closure": runtime.receipt,
            "numeric_tile": {
                "source_bf16_sha256": sha256(source),
                "packet_bytes": len(packet),
                "packet_sha256": sha256(packet),
                "logical_bits": encode["logical_bits"],
                "capacity_margin_bits": encode["capacity_margin_bits"],
                "all_encoder_self_checks_required_and_passed": True,
                "canonical_reencode_matches": True,
                "inverse_transient_dtype": decoded.canonical_symbols_i32.dtype.str,
                "inverse_i32_dtype_verified_before_facade_cast": True,
                "canonical_symbols_i32_sha256":
                    sha256(decoded.canonical_symbols_i32.tobytes()),
                "canonical_symbol_abs_max":
                    decoded.report["canonical_symbol_abs_max"],
                "integer_float_inverse_max_abs":
                    decoded.report["integer_float_inverse_max_abs"],
                "relative_mse_original_coordinates": relative_mse,
            },
            "i32_stress_lifetime": {
                "input_index": 63,
                "expected_abs_max": 8_388_608,
                "observed_abs_max": stress_max,
                "installed_in_inherited_decoder_before_call": True,
                "inverse_output_dtype_before_facade": inverse_output.dtype.str,
                "facade_retained_dtype": retained.dtype.str,
                "no_copy_or_downcast": True,
                "downstream_reconstruction_float64_abs_max": downstream_max,
                "downstream_reconstruction_expected_abs_max":
                    expected_downstream_max,
            },
            "aggregate_zero_frame": {
                "roles": list(runtime.packet.ROLES),
                "exact_inherited_role_abi":
                    tuple(runtime.packet.ROLES) ==
                    ("gate", "up", "down_transposed"),
                "frame_bytes": len(zero_frame),
                "frame_sha256": sha256(zero_frame),
                "encoder_all_checks_pass": zero_encode[
                    "all_encoder_self_checks_required_and_passed"],
                "literal_aggregate_reencode_matches": True,
            },
            "traffic_ledgers": {
                "prebuffered_decode": prebuffered,
                "modeled_one_external_pass": one,
                "modeled_two_external_passes": two,
            },
            "payload_accessed": False,
            "model_or_qwen_path_discovered_or_enumerated": False,
            "claim_boundary":
                "source-free mechanics/runtime only; authorizes a separately bound Qwen pilot but is not a Qwen, MSE, universal-tail, fine-code, or inference-HBM result",
        }
        record["receipt_sha256"] = sha256(canonical_json(record))
        smoke_contract.validate_smoke_receipt(
            record,
            source_manifest_sha256=package.manifest_sha256,
            source_root_sha256=package.source_root_sha256,
            predecessor_lock_sha256=predecessor_sha,
            runtime_lock_sha256=runtime_sha,
            source_member_hashes=package.receipt()["member_hashes"],
        )
        package.verify_final()
        payload = pretty_json(record)
        publication = write_receipt_exclusive(
            receipt_output, payload, package_dir)
        # The source descriptors remain retained across receipt creation; a
        # receipt path race can therefore never silently mutate this package.
        package.verify_final()
        return record, publication, payload


def write_receipt_exclusive(
    path: Path, payload: bytes, package_dir: Path,
) -> dict[str, Any]:
    require(path.is_absolute(), "receipt path absolute")
    parent = path.parent.resolve(strict=True)
    package = package_dir.resolve(strict=True)
    require(parent != package and package not in parent.parents,
            "receipt must remain outside immutable source package")
    require(path.name not in {"", ".", ".."} and
            "/" not in path.name and "\\" not in path.name,
            "safe receipt basename")
    parent_before = os.lstat(parent)
    require(stat.S_ISDIR(parent_before.st_mode), "receipt parent directory")
    parent_fd = -1
    descriptor = -1
    verify_fd = -1
    created = None
    success = False
    try:
        parent_fd = os.open(
            os.fspath(parent), os.O_RDONLY |
            getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0))
        parent_after = os.fstat(parent_fd)
        require((parent_before.st_dev, parent_before.st_ino,
                 stat.S_IFMT(parent_before.st_mode)) ==
                (parent_after.st_dev, parent_after.st_ino,
                 stat.S_IFMT(parent_after.st_mode)),
                "receipt parent inode binding")
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL |
            getattr(os, "O_BINARY", 0), 0o600, dir_fd=parent_fd)
        created = os.fstat(descriptor)
        require(stat.S_ISREG(created.st_mode) and created.st_nlink == 1,
                "receipt sole-link regular file")
        cursor = 0
        while cursor < len(payload):
            written = os.write(descriptor, payload[cursor:])
            require(written > 0, "receipt short write")
            cursor += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        verify_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) |
            getattr(os, "O_BINARY", 0), dir_fd=parent_fd)
        verified = os.fstat(verify_fd)
        require((verified.st_dev, verified.st_ino, verified.st_mode,
                 verified.st_size, verified.st_nlink) ==
                (created.st_dev, created.st_ino, created.st_mode,
                 len(payload), 1),
                "receipt reopened inode binding")
        chunks = []
        remaining = len(payload)
        while remaining:
            chunk = os.read(verify_fd, min(1 << 20, remaining))
            require(bool(chunk), "receipt verify short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(verify_fd, 1) == b"" and
                b"".join(chunks) == payload,
                "receipt exact reopened bytes")
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        require((named.st_dev, named.st_ino, named.st_mode,
                 named.st_size, named.st_nlink) ==
                (verified.st_dev, verified.st_ino, verified.st_mode,
                 verified.st_size, verified.st_nlink),
                "receipt final name/inode binding")
        os.fsync(parent_fd)
        success = True
        return {
            "bytes": len(payload),
            "sha256": sha256(payload),
            "exclusive_create": True,
            "final_name_inode_rebound": True,
            "reopened_bytes_verified": True,
        }
    finally:
        if verify_fd >= 0:
            os.close(verify_fd)
        if descriptor >= 0:
            os.close(descriptor)
        if not success and created is not None and parent_fd >= 0:
            try:
                named = os.stat(
                    path.name, dir_fd=parent_fd, follow_symlinks=False)
                if (stat.S_ISREG(named.st_mode) and
                        (named.st_dev, named.st_ino) ==
                        (created.st_dev, created.st_ino)):
                    os.unlink(path.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        if parent_fd >= 0:
            os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--predecessor-lock-sha256", required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt, publication, payload = run(
        arguments.repo_root,
        arguments.package_dir,
        arguments.package_manifest_sha256,
        arguments.predecessor_lock_sha256,
        arguments.runtime_lock_sha256,
        arguments.receipt_output,
    )
    print(json.dumps({
        "status": receipt["status"],
        "receipt_output": os.fspath(arguments.receipt_output),
        "receipt_bytes": len(payload),
        "receipt_file_sha256": sha256(payload),
        "internal_receipt_sha256": receipt["receipt_sha256"],
        "receipt_publication": publication,
        "positive_claim_authority": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
