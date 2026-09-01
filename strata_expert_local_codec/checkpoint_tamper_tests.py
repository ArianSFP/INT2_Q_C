#!/usr/bin/env python3
"""Adversarial, resealed mutation tests for a published checkpoint bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_expert_local_codec import common
from strata_expert_local_codec import verify_checkpoint as verify


Mutation = Callable[[Path], None]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))


def role_path(root: Path, role: str) -> Path:
    value = manifest(root)
    rows = [row for row in value["files"] if row["role"] == role]
    if len(rows) != 1:
        raise AssertionError(f"role lookup failed: {role}")
    return root / rows[0]["path"]


def reseal_file_row(root: Path, role: str) -> None:
    path = root / "checkpoint_manifest.json"
    value = manifest(root)
    rows = [row for row in value["files"] if row["role"] == role]
    if len(rows) != 1:
        raise AssertionError(f"manifest role lookup failed: {role}")
    target = root / rows[0]["path"]
    rows[0]["bytes"] = target.stat().st_size
    rows[0]["sha256"] = common.sha256_file(target)
    write_json(path, value)


def mutate_binary(root: Path, role: str, change: Callable[[bytearray], None]) -> None:
    path = role_path(root, role)
    payload = bytearray(path.read_bytes())
    change(payload)
    path.write_bytes(payload)
    reseal_file_row(root, role)


def mutate_json(root: Path, role: str, change: Callable[[dict[str, Any]], None]) -> None:
    path = role_path(root, role)
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    write_json(path, value)
    reseal_file_row(root, role)


def run_verifier(root: Path) -> dict[str, Any]:
    manifest_path = root / "checkpoint_manifest.json"
    value = verify.load_json(manifest_path, "manifest")
    roles = verify.verify_manifest(root, manifest_path, value)
    parsed = verify.parse_container(roles["container"])
    return verify.verify_evidence(roles, parsed, value)


def wrong_magic(root: Path) -> None:
    mutate_binary(root, "container", lambda data: data.__setitem__(0, data[0] ^ 1))


def broken_route_binding(root: Path) -> None:
    mutate_binary(
        root,
        "container",
        lambda data: data.__setitem__(verify.HEADER_BYTES, data[verify.HEADER_BYTES] ^ 1),
    )


def broken_header_crc(root: Path) -> None:
    mutate_binary(root, "container", lambda data: data.__setitem__(124, data[124] ^ 1))


def changed_directory_profile(root: Path) -> None:
    offset = verify.HEADER_BYTES + verify.ROUTE_BYTES + verify.LABEL_BYTES
    mutate_binary(root, "container", lambda data: data.__setitem__(offset, data[offset] ^ 1))


def nonzero_zero_tail(root: Path) -> None:
    mutate_binary(root, "container", lambda data: data.__setitem__(-1, 1))


def truncated_container(root: Path) -> None:
    path = role_path(root, "container")
    payload = path.read_bytes()
    path.write_bytes(payload[:-1])
    reseal_file_row(root, "container")


def nonzero_payload_padding(root: Path) -> None:
    path = role_path(root, "container")
    payload = bytearray(path.read_bytes())
    directory_begin = verify.HEADER_BYTES + verify.ROUTE_BYTES + verify.LABEL_BYTES
    cursor = verify.PREFIX_BYTES
    changed = False
    for ordinal in range(verify.BLOCKS):
        _, _, logical_bits = struct.unpack_from(
            "<BeI", payload, directory_begin + ordinal * verify.DIRECTORY_RECORD.size
        )
        size = (logical_bits + 7) // 8
        padding = size * 8 - logical_bits
        if padding:
            payload[cursor + size - 1] |= 1
            changed = True
            break
        cursor += size
    if not changed:
        raise AssertionError("test fixture has no padded arithmetic stream")
    path.write_bytes(payload)
    reseal_file_row(root, "container")


def forged_summary_hash(root: Path) -> None:
    mutate_json(root, "summary", lambda value: value["artifact"].__setitem__("sha256", "0" * 64))


def forged_read_ledger(root: Path) -> None:
    mutate_json(
        root,
        "summary",
        lambda value: value["read_amplification"].__setitem__(
            "max_4k", float(value["read_amplification"]["max_4k"]) + 0.01
        ),
    )


def forged_mse_quotient(root: Path) -> None:
    audit_path = role_path(root, "independent_audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    forged = float(audit["source_score"]["energy_weighted_relative_mse"]) * 0.9
    audit["source_score"]["energy_weighted_relative_mse"] = forged
    write_json(audit_path, audit)
    reseal_file_row(root, "independent_audit")
    manifest_path = root / "checkpoint_manifest.json"
    value = manifest(root)
    value["claim"]["energy_weighted_relative_mse"] = forged
    write_json(manifest_path, value)


def broken_plan_seal(root: Path) -> None:
    mutate_json(
        root,
        "plan",
        lambda value: value["physical_ledger"].__setitem__(
            "physical_bytes", int(value["physical_ledger"]["physical_bytes"]) - 1
        ),
    )


def rebind_container_sha(root: Path) -> str:
    container = role_path(root, "container")
    digest = common.sha256_file(container)
    manifest_path = root / "checkpoint_manifest.json"
    outer = manifest(root)
    outer["artifact"]["container_sha256"] = digest
    write_json(manifest_path, outer)
    reseal_file_row(root, "container")
    summary_path = role_path(root, "summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifact"]["sha256"] = digest
    write_json(summary_path, summary)
    reseal_file_row(root, "summary")
    audit_path = role_path(root, "independent_audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["container"]["sha256"] = digest
    write_json(audit_path, audit)
    reseal_file_row(root, "independent_audit")
    return digest


def payload_rebound_but_metadata_stale(root: Path) -> None:
    container = role_path(root, "container")
    parsed = verify.parse_container(container)
    row = next(item for item in parsed["directory"] if item["payload_bytes"] > 0)
    payload = bytearray(container.read_bytes())
    payload[int(row["file_byte_begin"])] ^= 0x80
    container.write_bytes(payload)
    rebind_container_sha(root)


def audit_plan_unlinked(root: Path) -> None:
    mutate_json(
        root,
        "independent_audit",
        lambda value: value["bindings"].__setitem__("plan_lock_sha256", "0" * 64),
    )


def source_plan_comprehensively_rebound(root: Path) -> None:
    """Substitute one source identity and reseal every dependent evidence row.

    This mutation would satisfy the former dynamic plan/audit bindings.  It is
    rejected only because the verifier now anchors the canonical digest of the
    eighteen precommitted BF16 source records.
    """
    plan_path = role_path(root, "plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rebound_hash = "0" * 64
    plan["sources"][0]["source_bf16_sha256"] = rebound_hash
    plan = common.sealed(plan)
    write_json(plan_path, plan)
    reseal_file_row(root, "plan")

    summary_path = role_path(root, "summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["plan_lock_sha256"] = plan["lock_sha256"]
    write_json(summary_path, summary)
    reseal_file_row(root, "summary")

    audit_path = role_path(root, "independent_audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["bindings"]["plan_lock_sha256"] = plan["lock_sha256"]
    audit["bindings"]["sources_canonical_sha256"] = hashlib.sha256(
        verify.canonical_json_bytes(plan["sources"])
    ).hexdigest()
    audit["source_score"]["matrices"][0]["source_bf16_sha256"] = rebound_hash
    write_json(audit_path, audit)
    reseal_file_row(root, "independent_audit")


def required_encoder_evidence_removed(root: Path) -> None:
    path = root / "checkpoint_manifest.json"
    value = manifest(root)
    value["files"] = [
        row for row in value["files"] if row["role"] != "encoder_block_14"
    ]
    write_json(path, value)


def coefficient_code_mismatch_rebound(root: Path) -> None:
    container_path = role_path(root, "container")
    payload = bytearray(container_path.read_bytes())
    payload[32] ^= 1
    struct.pack_into("<I", payload, 124, zlib.crc32(payload[:124]) & 0xFFFFFFFF)
    container_path.write_bytes(payload)

    header_path = role_path(root, "asset_header_bin")
    header_path.write_bytes(payload[: verify.HEADER_BYTES])
    reseal_file_row(root, "asset_header_bin")

    plan_path = role_path(root, "plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["assets"]["header.bin"]["sha256"] = hashlib.sha256(
        payload[: verify.HEADER_BYTES]
    ).hexdigest()
    plan = common.sealed(plan)
    write_json(plan_path, plan)
    reseal_file_row(root, "plan")

    summary_path = role_path(root, "summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["plan_lock_sha256"] = plan["lock_sha256"]
    write_json(summary_path, summary)
    reseal_file_row(root, "summary")
    audit_path = role_path(root, "independent_audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["bindings"]["plan_lock_sha256"] = plan["lock_sha256"]
    write_json(audit_path, audit)
    reseal_file_row(root, "independent_audit")
    rebind_container_sha(root)


CASES: tuple[tuple[str, Mutation, str], ...] = (
    ("wrong_magic_resealed", wrong_magic, "header constants"),
    ("route_binding_resealed", broken_route_binding, "asset binding"),
    ("header_crc_resealed", broken_header_crc, "CRC"),
    ("directory_profile_resealed", changed_directory_profile, "plan profile"),
    ("zero_tail_resealed", nonzero_zero_tail, "terminal reservoir"),
    ("truncated_container_resealed", truncated_container, "physical byte count"),
    ("payload_padding_resealed", nonzero_payload_padding, "payload padding"),
    ("summary_hash_resealed", forged_summary_hash, "summary/container hash"),
    ("read_ledger_resealed", forged_read_ledger, "4-KiB read amp"),
    ("mse_quotient_resealed", forged_mse_quotient, "source score quotient"),
    ("plan_internal_seal_resealed", broken_plan_seal, "internal seal"),
    ("payload_rebound_metadata_stale", payload_rebound_but_metadata_stale, "encoder payload hash"),
    ("audit_plan_unlinked_resealed", audit_plan_unlinked, "audit/plan binding"),
    (
        "source_plan_comprehensive_rebind",
        source_plan_comprehensively_rebound,
        "pinned source digest",
    ),
    ("required_encoder_evidence_removed", required_encoder_evidence_removed, "lacks a required evidence role"),
    ("coefficient_code_mismatch_rebound", coefficient_code_mismatch_rebound, "KLT coefficient"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attach-to-manifest", action="store_true")
    args = parser.parse_args()
    release = args.release_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    pristine = run_verifier(release)
    results: list[dict[str, Any]] = []
    for name, mutation, expected in CASES:
        with tempfile.TemporaryDirectory(prefix="strata-local-tamper-") as text:
            trial = Path(text) / "release"
            shutil.copytree(release, trial)
            mutation(trial)
            try:
                run_verifier(trial)
            except Exception as error:  # the exact rejection is evidence
                message = f"{type(error).__name__}: {error}"
                passed = expected in message
            else:
                message = "mutation was incorrectly accepted"
                passed = False
            results.append(
                {
                    "name": name,
                    "expected_rejection_substring": expected,
                    "observed": message,
                    "passed": passed,
                }
            )
    report = {
        "schema": "strata_expert_affine_tamper_tests_v1",
        "status": "passed" if all(row["passed"] for row in results) else "failed",
        "pristine": pristine,
        "container_sha256": pristine["container_sha256"],
        "executing_verifier_sha256": common.sha256_file(Path(verify.__file__)),
        "executing_harness_sha256": common.sha256_file(Path(__file__)),
        "cases_passed": sum(bool(row["passed"]) for row in results),
        "cases_total": len(results),
        "cases": results,
    }
    common.write_json(output, report)
    if args.attach_to_manifest:
        try:
            relative = output.relative_to(release)
        except ValueError as error:
            raise ValueError("attached tamper report must be inside release-dir") from error
        manifest_path = release / "checkpoint_manifest.json"
        value = manifest(release)
        if any(row["role"] == "tamper_report" for row in value["files"]):
            raise ValueError("manifest already contains a tamper report")
        value["files"].append(
            {
                "path": relative.as_posix(),
                "bytes": output.stat().st_size,
                "sha256": common.sha256_file(output),
                "role": "tamper_report",
                "classification": "adversarial_verification_evidence",
            }
        )
        write_json(manifest_path, value)
        run_verifier(release)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
