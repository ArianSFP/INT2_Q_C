#!/usr/bin/env python3
"""Independent standard-library verifier for MOSAIC secondary-oracle source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "mosaic-secondary-oracles-source-manifest-v0"
STATUS = "SEALED_SOURCE_ONLY_NO_QWEN_OR_COARSE_PAYLOAD_AUTHORITY"
EXPECTED = {
    "README.md",
    "cupy_backend.py",
    "design_lock.json",
    "gate_contract.py",
    "gf2_recurrence.py",
    "residual_oracles.py",
    "run_source_free_cupy_smoke.py",
    "run_source_free_fixture.py",
    "test_source_only.py",
    "verify_source.py",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"{label} duplicate key")
            output[key] = value
        return output

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(VerifyError(f"{label} nonfinite {token}")),
    )
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, maximum: int = 4 * (1 << 20)) -> bytes:
    require(path.is_absolute(), "absolute source path")
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and 0 < before.st_size <= maximum,
            "regular single-link source",
        )
        output = bytearray()
        while len(output) < before.st_size:
            chunk = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(chunk), "short source read")
            output.extend(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
            "source identity drift",
        )
        return bytes(output)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "real package directory")
    entries = list(os.scandir(root))
    require(
        all(entry.is_file(follow_symlinks=False) and not entry.is_symlink() for entry in entries),
        "package files only",
    )
    require({entry.name for entry in entries} == EXPECTED | {"SOURCE_MANIFEST.json"}, "exact member set")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json")
    manifest_sha = digest(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, "expected manifest digest")
    manifest = strict_json(manifest_payload, "manifest")
    require(
        set(manifest)
        == {
            "schema",
            "status",
            "source_root_sha256",
            "members",
            "access_attestation",
            "claim_boundary",
        },
        "manifest exact schema",
    )
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

    access = manifest["access_attestation"]
    require(
        set(access)
        == {
            "qwen_model_checkpoint_payload_accessed",
            "coarse_result_or_COARSE_bin_accessed",
            "matched_control_payload_accessed",
            "network_used_by_tests",
            "production_payload_adapter_present",
            "source_free_cpu_tests_run",
            "source_free_cupy_smoke_run",
        },
        "access attestation schema",
    )
    require(
        access["qwen_model_checkpoint_payload_accessed"] is False
        and access["coarse_result_or_COARSE_bin_accessed"] is False
        and access["matched_control_payload_accessed"] is False
        and access["network_used_by_tests"] is False
        and access["production_payload_adapter_present"] is False
        and access["source_free_cpu_tests_run"] is True
        and access["source_free_cupy_smoke_run"] is True,
        "source-only attestation",
    )
    require(
        manifest["claim_boundary"]
        == "source-frozen mechanics only; no Qwen/coarse result, finite target pass, converse, or universal SwiGLU-MoE claim",
        "claim boundary",
    )

    design = strict_json(payloads["design_lock.json"], "design")
    require(design["schema"] == "mosaic-secondary-oracles-design-v0", "design schema")
    require(design["repository_evidence"]["periods_1_2_4_are_not_retested"] is True, "no cyclo duplication")
    require(abs(design["objective"]["required_coarse_residual_capture"] - 0.32387022205373717) < 1e-15, "coarse gate")
    require(design["gf2_recurrence"]["exceptions_implemented"] is False, "recurrence claim scope")
    require(design["hankel_AR"]["finite_innovation_codec_executed"] is False, "Hankel oracle scope")
    require(design["BM3D"]["status"] == "DEFERRED_NOT_WARRANTED_BY_CURRENT_GRAPH_EVIDENCE", "BM3D defer")
    require(design["controls"]["minimum_source_minus_stronger_control_bpw"] == 0.03, "control threshold")

    recurrence_source = payloads["gf2_recurrence.py"].decode("utf-8")
    for fragment in (
        "def berlekamp_massey_gf2",
        "def encode_block",
        "def decode_block",
        "def encode_component",
        "def decode_component",
        "def encode_expert",
        "def decode_expert",
        "generate_lfsr(initial, connection, len(raw)) == raw",
        "encode_block_no_check(decoded) == packet",
        "recurrence canonical encoding",
        "component canonical padding",
        "expert canonical padding",
        "decoded[\"component_packets\"] == tuple(role_components)",
    ):
        require(fragment in recurrence_source, f"recurrence mechanism {fragment}")
    residual_source = payloads["residual_oracles.py"].decode("utf-8")
    for fragment in (
        "def build_ramanujan_basis",
        "ceil_log2_binomial(columns, rank)",
        "def inverse_noise_gain",
        "pullback_noise_amplification_charged",
        "def phase_destroyed_blocks",
        "def moment_matched_gaussian_blocks",
    ):
        require(fragment in residual_source, f"residual mechanism {fragment}")
    contract_source = payloads["gate_contract.py"].decode("utf-8")
    require("controls forbidden after absolute source miss" in contract_source, "source-first controls")
    gpu_source = payloads["cupy_backend.py"].decode("utf-8")
    require("import cupy" not in gpu_source.split("def load_cupy", 1)[0], "lazy CuPy")
    gpu_runner = payloads["run_source_free_cupy_smoke.py"].decode("utf-8")
    for forbidden in ("--payload", "--qwen", "--coarse", "COARSE.bin"):
        require(forbidden not in gpu_runner, f"GPU runner payload aperture {forbidden}")
    return {
        "schema": "mosaic-secondary-oracles-source-verifier-receipt-v0",
        "status": "PASS_SOURCE_ONLY",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": root_sha,
        "members": len(rows),
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "production_adapter_present": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", type=Path, required=True)
    result.add_argument("--manifest-sha256")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.package, arguments.manifest_sha256), sort_keys=True, separators=(",", ":")))
