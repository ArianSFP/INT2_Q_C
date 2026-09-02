#!/usr/bin/env python3
"""Independent standard-library verifier for TACTIC Ramanujan-384 source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "tactic-ramanujan384-source-manifest-v0"
STATUS = "FROZEN_SOURCE_ONLY_RUNPOD_EXECUTION_PENDING_NO_PAYLOAD_AUTHORITY"
EXPECTED = {
    "README.md",
    "adapter.py",
    "authenticated_io.py",
    "container.py",
    "cupy_backend.py",
    "dependency_lock.json",
    "design_lock.json",
    "packet.py",
    "ramanujan_codec.py",
    "run_source_free_cupy_smoke.py",
    "run_source_free_fixture.py",
    "test_source_only.py",
    "verify_source.py",
}
DEPENDENCY = {
    "SOURCE_MANIFEST.json": "4259e8e8dc87b4c25301ca89ade7dbd63c1e0c9e3415fdaa4d7881d7d10ccc06",
    "residual_oracles.py": "f990aedf8eba0e9058bd9c77caaa05df98226b2855486bace0eaee15cfac806f",
    "gate_contract.py": "0556915344dfd996ccfdd5005cab437c0c5ad154a98efafe59a07f32f30bb0e2",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(VerifyError(f"{label} nonfinite {token}")),
    )
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, maximum: int = 4 << 20) -> bytes:
    require(path.is_absolute(), "absolute source path")
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and 0 < before.st_size <= maximum, "regular single-link source")
        output = bytearray()
        while len(output) < before.st_size:
            row = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(row), "short source read")
            output.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
                "source identity drift")
        return bytes(output)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "real package directory")
    entries = list(os.scandir(root))
    require(all(entry.is_file(follow_symlinks=False) and not entry.is_symlink() for entry in entries),
            "package files only")
    require({entry.name for entry in entries} == EXPECTED | {"SOURCE_MANIFEST.json"}, "exact member set")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json")
    manifest_sha = digest(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, "expected manifest digest")
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {
        "schema", "status", "source_root_sha256", "members", "dependency_pins",
        "access_attestation", "execution_attestation", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS, "manifest identity")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(EXPECTED), "manifest rows")
    require([row.get("name") for row in rows] == sorted(EXPECTED), "canonical member order")
    observed = []
    payloads = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "member row")
        name = row["name"]
        require(name in EXPECTED and type(row["bytes"]) is int and row["bytes"] > 0, "member metadata")
        payload = read_regular(root / name)
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"], f"member closure {name}")
        observed.append({"name": name, "bytes": len(payload), "sha256": digest(payload)})
        payloads[name] = payload
    root_sha = digest(canonical_json(observed))
    require(root_sha == manifest["source_root_sha256"], "source root")
    require(manifest["dependency_pins"] == DEPENDENCY, "dependency pins")

    parent = root.parent / "mosaic_secondary_oracles_v0"
    for name, expected in DEPENDENCY.items():
        require(digest(read_regular(parent / name)) == expected, f"audited dependency {name}")

    access = manifest["access_attestation"]
    require(access == {
        "qwen_model_checkpoint_payload_accessed": False,
        "coarse_result_or_COARSE_bin_accessed": False,
        "matched_control_payload_accessed": False,
        "network_used_by_tests": False,
        "production_adapter_source_present": True,
    }, "source-only access attestation")
    require(manifest["execution_attestation"] == {
        "source_only_cpu_tests_run": False,
        "source_free_cupy_smoke_run": False,
        "runpod_execution_pending": True,
    }, "pending execution attestation")
    require(manifest["claim_boundary"] ==
            "source mechanics only; no Qwen result, target pass, portability result, universal performance claim, or HBM measurement",
            "claim boundary")

    design = strict_json(payloads["design_lock.json"], "design")
    require(design["dictionary"]["period_count"] == 120
            and design["dictionary"]["every_period_represented"] is True, "period coverage")
    require(design["packet"]["bits_per_block"] == 384
            and design["packet"]["crc_bits"] == 32, "literal packet budget")
    require(design["container"]["header_bytes"] == 512
            and design["container"]["qwen_768x2048x3_physical_bytes"] == 1470464
            and design["container"]["qwen_768x2048x3_physical_rate_bpw"] < 2.5,
            "literal container budget")
    require(design["controls"]["controls_forbidden_before_absolute_D_le_0p025"] is True
            and len(design["controls"]["gaussian_seeds"]) == 8, "full control contract")
    require(design["traffic"]["tail_free_external_read_amplification"] == 1.0
            and design["traffic"]["accelerator_HBM_measured"] is False, "read ledger boundary")
    require(design["authentication"]["expected_binding_sha256_required"] is True
            and design["authentication"]["independent_coarse_audit_receipt_sha256_required"] is True,
            "authentication closure")

    packet_source = payloads["packet.py"].decode("utf-8")
    for fragment in (
        "PACKET_BYTES = 48", "MAX_RANK = 14", "ATOM_COUNT = 384", "zlib.crc32(body)",
        "canonical zero padding", "canonical packet reencode",
    ):
        require(fragment in packet_source, f"packet mechanism {fragment}")
    container_source = payloads["container.py"].decode("utf-8")
    for fragment in (
        "HEADER_BYTES = 512", "def encode_composite", "def decode_composite",
        "def expected_coarse_bytes", "literal 307/128-bpw coarse payload length",
        "minimal page padding", "canonical fine role order", "canonical header reencode",
        "external_read_amplification\": 1.0",
    ):
        require(fragment in container_source, f"container mechanism {fragment}")
    codec_source = payloads["ramanujan_codec.py"].decode("utf-8")
    for fragment in (
        "def ramanujan_sum", "def period_bank_labels", "every period represented",
        "xp.linalg.solve", "astype(xp.float16).astype(xp.float64)",
        "source - reconstruction", "phase_destroyed_blocks", "moment_matched_gaussian_blocks",
        "external_read_amplification\": 1.0",
    ):
        require(fragment in codec_source, f"codec mechanism {fragment}")
    auth_source = payloads["authenticated_io.py"].decode("utf-8")
    for fragment in (
        "expected binding SHA256 required", "independent audit receipt SHA256",
        "coarse artifact independent publication pin", "source role independent pin",
        "coarse reconstruction independent pin", "O_NOFOLLOW", "duplicate key",
        "duplicate independent publication member", "duplicate independent input role",
    ):
        require(fragment in auth_source, f"authentication mechanism {fragment}")
    adapter_source = payloads["adapter.py"].decode("utf-8")
    for fragment in (
        "def run_authenticated_expert", "canonical authenticated role order",
        "HARD_KILL_PHYSICAL_RATE_OUTSIDE_2P15_TO_2P5", "source_minus_strongest_control_bpw",
        "complete_three_role_finite_search_rerun", "container.encode_composite",
    ):
        require(fragment in adapter_source, f"whole-expert adapter mechanism {fragment}")
    backend = payloads["cupy_backend.py"].decode("utf-8")
    require("import cupy" not in backend.split("def load_cupy", 1)[0], "lazy CuPy import")
    runner = payloads["run_source_free_cupy_smoke.py"].decode("utf-8")
    for forbidden in ("--payload", "--qwen", "--coarse", "COARSE.bin"):
        require(forbidden not in runner, f"source-free runner aperture {forbidden}")
    return {
        "schema": "tactic-ramanujan384-source-verifier-receipt-v0",
        "status": "PASS_FROZEN_SOURCE_ONLY_EXECUTION_PENDING",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": root_sha,
        "members": len(rows),
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "runpod_execution_pending": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", type=Path, required=True)
    result.add_argument("--manifest-sha256")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.package, arguments.manifest_sha256), sort_keys=True, separators=(",", ":")))
