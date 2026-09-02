#!/usr/bin/env python3
"""Retained standard-library verifier for finite TACTIC-DH384 v3."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MANIFEST_SCHEMA = "tactic-dh384-finite-v3-source-manifest-v1"


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                VerifyError(f"{label}: nonfinite {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerifyError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: object")
    return value


def reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode),
                f"{label}: symlink component {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def read_nofollow(path: Path, *, expected_bytes: int | None = None,
                  expected_sha256: str | None = None,
                  maximum_bytes: int = 4 * (1 << 20),
                  label: str) -> bytes:
    require(path.is_absolute(), f"{label}: absolute")
    reject_symlink_chain(path, label)
    descriptor = os.open(
        os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0) |
        getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= maximum_bytes,
                f"{label}: regular sole-link byte bound")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes, f"{label}: exact bytes")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label}: trailing bytes")
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256,
                    f"{label}: digest")
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_mode, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns, before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_mode, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns, after.st_nlink),
                f"{label}: identity drift")
        return payload
    finally:
        os.close(descriptor)


def load_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, f"module collision: {name}")
    digest = sha256(source)
    module = types.ModuleType(name)
    module.__file__ = f"<source-verify:{name}:{digest}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True,
                     optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def verify() -> dict[str, Any]:
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "invoke source verifier with CPython -I -B")
    manifest_payload = read_nofollow(
        ROOT / "SOURCE_MANIFEST.json", label="source manifest bootstrap")
    bootstrap = strict_json(manifest_payload, "source manifest bootstrap")
    require(bootstrap.get("schema") == MANIFEST_SCHEMA,
            "source manifest schema")
    rows = bootstrap.get("members")
    require(isinstance(rows, list), "source manifest rows")
    source_auth_rows = [row for row in rows if isinstance(row, dict) and
                        row.get("name") == "source_auth.py"]
    require(len(source_auth_rows) == 1, "source_auth row")
    auth_row = source_auth_rows[0]
    auth_source = read_nofollow(
        ROOT / "source_auth.py", expected_bytes=auth_row["bytes"],
        expected_sha256=auth_row["sha256"], label="source_auth bootstrap")
    auth = load_module("tactic_dh384_finite_v3_verify_auth", auth_source)
    with auth.HeldSourcePackage(
        ROOT, sha256(manifest_payload),
        executing_path=Path(__file__).resolve(strict=True)) as package:
        sources = package.sources
        for name, payload in sources.items():
            if name.endswith(".py"):
                compile(payload, f"<verified:{name}:{sha256(payload)}>",
                        "exec", dont_inherit=True, optimize=0)
        spec = load_module(
            "tactic_dh384_finite_v3_verify_spec", sources["format_spec.py"])
        require(spec.sha256(spec.universal_selector_packet()) ==
                "0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad",
                "audited selector identity")
        require(str(spec.COARSE_BPW) == "307/128" and
                str(spec.FINE_BPW) == "3/32" and
                str(spec.HEADER_BPW) == "1/128" and
                str(spec.COMPOSITE_BPW) == "5/2" and
                spec.COMPOSITE_BYTES == 1_474_560,
                "literal single-expert rate")
        require(spec.FINE_RECORD_BYTES == 48 and
                spec.ACTIVE_RANK == 376 and
                spec.AUDITED_PARENT_RANK == 384 and
                8 + spec.ACTIVE_RANK == 384,
                "finite record allocation")
        design = strict_json(sources["design_lock.json"], "design lock")
        require(design.get("schema") ==
                "tactic-dh384-finite-v3-design-lock-v1" and
                design["predecessor_boundary"][
                    "tactic_conditional_dyadic_coset_v2_modified"] is False and
                design["predecessor_boundary"][
                    "tactic_actual_coarse_n18_v6_modified"] is False,
                "immutable predecessor boundary")
        require(design["literal_single_expert_packet"] == {
            "aligned_4096_pages": 360,
            "coarse_bpw": "307/128",
            "coarse_bytes": 1414656,
            "fine_bpw": "12/128",
            "fine_bytes": 55296,
            "global_packet_bytes_emitted": 0,
            "header_bpw": "1/128",
            "header_bytes": 4608,
            "metadata_slack_bits": 0,
            "one_external_file_pass_amplification_over_literal_packet": 1.0,
            "physical_bpw": "320/128 = 2.5",
            "total_bytes": 1474560,
        }, "literal pilot ledger")
        require(design["six_expert_amortized_layout"]["emitted"] is False and
                design["six_expert_amortized_layout"][
                    "seventy_three_over_seventy_two_claim"] is False,
                "no inferred six-expert layout")
        require(design["encoder_objective"]["fine_labels_source_selected"] is
                True and
                design["encoder_objective"]["coarse_codeword_reoptimized"] is
                False, "fine-label search boundary")
        require(design["claim_boundary"][
                    "qwen_payload_accessed_during_source_build"] is False and
                design["claim_boundary"][
                    "v6_live_result_accessed_during_source_build"] is False and
                design["claim_boundary"]["positive_claim_authority"] is False,
                "source-only design boundary")

        v6_lock = strict_json(sources["V6_LOCK.json"], "v6 lock")
        require(v6_lock.get("schema") ==
                "tactic-dh384-finite-v3-v6-lock-v1" and
                v6_lock["source_manifest"]["sha256"] ==
                "31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d" and
                v6_lock["source_root_sha256"] ==
                "161ab23169af3427648ec1bbcb9402568a0fb8aefc4a794daf3ebd1c56cc83f2",
                "v6 source pins")
        bridge = load_module(
            "tactic_dh384_finite_v3_verify_bridge",
            sources["runtime_bridge.py"])
        with bridge.HeldV6Package(
            REPO, REPO / v6_lock["relative_directory"],
            sources["V6_LOCK.json"]) as v6:
            v6.verify_final()
            v6_receipt = v6.receipt()

        dispatcher = sources["dispatcher.py"].decode("utf-8")
        require(dispatcher.index("validate_launch_review") <
                dispatcher.index("v6_package.load_runtime") <
                dispatcher.index("HeldCompletedV6Result") <
                dispatcher.index("authenticate_inputs"),
                "review/runtime/result/input order")
        require("HARD_REJECT_PARENT_RANK384" in dispatcher and
                "HARD_REJECT_ACTIVE_RANK376" in dispatcher,
                "continuous hard-kill implementation")
        decoder = sources["independent_decoder.py"].decode("utf-8")
        require("finite_encoder" not in decoder and
                "correction dyadic-span containment" in decoder,
                "independent decoder boundary")
        publisher = sources["atomic_publish.py"].decode("utf-8")
        require("RENAME_NOREPLACE" in publisher and
                ".COMPLETE.pending" in publisher and
                "COMPLETE.json" in publisher,
                "atomic terminal publication")
        package.verify_final()
        return {
            "schema": "tactic-dh384-finite-v3-source-verification-v1",
            "status": "PASS_SOURCE_ONLY_FINITE_CODEC_AWAITING_INDEPENDENT_REVIEW",
            "manifest_sha256": package.manifest_sha256,
            "source_root_sha256": package.source_root_sha256,
            "source_members": len(package.members),
            "source_bytes_excluding_manifest":
                sum(len(payload) for payload in sources.values()),
            "v6_source_manifest_sha256":
                v6_receipt["v6_source_manifest_sha256"],
            "v6_source_root_sha256": v6_receipt["v6_source_root_sha256"],
            "finite_record_bits": 384,
            "charged_scale_bits": 8,
            "charged_sign_bits": 376,
            "literal_single_expert_bpw": "320/128",
            "qwen_payload_accessed": False,
            "v6_live_result_accessed": False,
            "cupy_or_cuda_initialized": False,
            "positive_claim_authority": False,
        }


def main() -> int:
    print(json.dumps(verify(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
