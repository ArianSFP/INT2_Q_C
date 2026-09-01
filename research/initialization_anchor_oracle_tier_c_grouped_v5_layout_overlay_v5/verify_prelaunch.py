"""Isolated, bytecode-free verifier for grouped-v5 layout-overlay v5.

This file deliberately imports only the Python standard library until the
package directory has been proven closed and every manifested byte has been
authenticated.  Invoke it with ``python -B -I``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import stat
import sys
from pathlib import Path


EXPECTED_CANONICAL_PACKAGE = (
    "/workspace/INT2__compression/INT2_Q_C/research/"
    "initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5"
)
EXPECTED_CANONICAL_ENTRYPOINT = f"{EXPECTED_CANONICAL_PACKAGE}/verify_prelaunch.py"
EXPECTED_PACKAGE_MOUNT_POINT = "/workspace"
MANIFEST_BASENAME = "ARTIFACT_SHA256SUMS.txt"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _decode_mountinfo_field(value: str) -> str:
    return re.sub(
        r"\\(040|011|012|134)",
        lambda match: {"040": " ", "011": "\t", "012": "\n", "134": "\\"}[
            match.group(1)
        ],
        value,
    )


def _mount_identity_for_raw_absolute(path: str) -> dict[str, object]:
    try:
        raw = Path("/proc/self/mountinfo").read_bytes()
    except OSError as exc:
        raise RuntimeError("Linux mountinfo is mandatory for the v5 bootstrap") from exc
    candidates: list[dict[str, object]] = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            raise RuntimeError("malformed /proc/self/mountinfo row")
        fields = left.split()
        tail = right.split()
        if len(fields) < 6 or len(tail) < 3:
            raise RuntimeError("short /proc/self/mountinfo row")
        mount_point = _decode_mountinfo_field(fields[4])
        if path != mount_point and not path.startswith(mount_point.rstrip("/") + "/"):
            continue
        relative = posixpath.relpath(path, mount_point)
        filesystem_path = posixpath.normpath(
            posixpath.join(_decode_mountinfo_field(fields[3]), relative)
        )
        candidates.append(
            {
                "mount_id": int(fields[0]),
                "parent_mount_id": int(fields[1]),
                "major_minor": fields[2],
                "mount_root": _decode_mountinfo_field(fields[3]),
                "mount_point": mount_point,
                "filesystem_path": filesystem_path,
                "filesystem_type": tail[0],
                "mount_source": _decode_mountinfo_field(tail[1]),
                "mount_row_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
            }
        )
    if not candidates:
        raise RuntimeError("entrypoint has no mountinfo identity")
    return max(candidates, key=lambda row: len(str(row["mount_point"])))


def _capture_raw_entrypoint_identity() -> dict[str, object]:
    raw = sys.argv[0]
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise RuntimeError("raw argv0 is absent or malformed")
    if not os.path.isabs(raw):
        raise RuntimeError("raw argv0 must be absolute")
    if raw != os.path.normpath(raw) or "." in Path(raw).parts or ".." in Path(raw).parts:
        raise RuntimeError("raw argv0 must use one canonical lexical spelling")
    if raw != EXPECTED_CANONICAL_ENTRYPOINT:
        raise RuntimeError("raw argv0 differs from the frozen canonical entrypoint")
    if os.fspath(__file__) != raw:
        raise RuntimeError("raw argv0 and unnormalized __file__ spellings differ")

    component_rows: list[dict[str, object]] = []
    parts = Path(raw).parts
    cursor = Path(parts[0])
    for part in parts[1:]:
        cursor = cursor / part
        try:
            info = os.lstat(cursor)
        except OSError as exc:
            raise RuntimeError(f"raw argv0 component is missing: {cursor}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"raw argv0 contains a symlink component: {cursor}")
        component_rows.append(
            {
                "path": os.fspath(cursor),
                "device": int(info.st_dev),
                "inode": int(info.st_ino),
                "mode": int(info.st_mode),
            }
        )
    file_info = os.lstat(raw)
    if not stat.S_ISREG(file_info.st_mode):
        raise RuntimeError("raw argv0 target is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(raw, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (file_info.st_dev, file_info.st_ino, file_info.st_size)
        ):
            raise RuntimeError("raw argv0 changed before descriptor binding")
    finally:
        os.close(descriptor)
    package_info = os.lstat(EXPECTED_CANONICAL_PACKAGE)
    if not stat.S_ISDIR(package_info.st_mode) or stat.S_ISLNK(package_info.st_mode):
        raise RuntimeError("frozen canonical package is not a real directory")
    package_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    package_descriptor = os.open(EXPECTED_CANONICAL_PACKAGE, package_flags)
    try:
        opened_package = os.fstat(package_descriptor)
        if (
            not stat.S_ISDIR(opened_package.st_mode)
            or (opened_package.st_dev, opened_package.st_ino)
            != (package_info.st_dev, package_info.st_ino)
        ):
            raise RuntimeError("package changed before directory-descriptor binding")
    finally:
        os.close(package_descriptor)
    mount = _mount_identity_for_raw_absolute(raw)
    package_mount = _mount_identity_for_raw_absolute(EXPECTED_CANONICAL_PACKAGE)
    if (
        mount["mount_id"] != package_mount["mount_id"]
        or mount["mount_point"] != EXPECTED_PACKAGE_MOUNT_POINT
        or package_mount["mount_point"] != EXPECTED_PACKAGE_MOUNT_POINT
    ):
        raise RuntimeError("entrypoint/package mount identity differs from frozen boundary")
    return {
        "raw_argv0": raw,
        "raw_argv0_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "verifier_device": int(file_info.st_dev),
        "verifier_inode": int(file_info.st_ino),
        "verifier_bytes": int(file_info.st_size),
        "package_device": int(package_info.st_dev),
        "package_inode": int(package_info.st_ino),
        "component_identity_sha256": hashlib.sha256(
            json.dumps(component_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "component_count": len(component_rows),
        "mount": mount,
    }


# In the executable authentication path this capture deliberately occurs before
# any normalization of ``__file__``.  Library import exists only so the CPU test
# suite can exercise the closure parser against copied fixtures; it is never an
# authenticated dispatch path and cannot request the runtime firewall.
if __name__ == "__main__":
    RAW_ENTRYPOINT_IDENTITY: dict[str, object] | None = (
        _capture_raw_entrypoint_identity()
    )
    PACKAGE = Path(str(RAW_ENTRYPOINT_IDENTITY["raw_argv0"])).parent
else:
    RAW_ENTRYPOINT_IDENTITY = None
    PACKAGE = Path(EXPECTED_CANONICAL_PACKAGE)
EXPECTED_ARTIFACTS = frozenset(
    {
        "README.md",
        "candidate_lock.json",
        "common.py",
        "kernels.py",
        "overlay.py",
        "parity.py",
        "source_trace.py",
        "test_bootstrap.py",
        "test_common.py",
        "test_overlay.py",
        "test_parity.py",
        "test_source_trace.py",
        "test_tier_c_gate.py",
        "tier_c_gate.py",
        "verify_prelaunch.py",
        "verify_v4_reuse.py",
    }
)
EXPECTED_PACKAGE_MEMBERS = EXPECTED_ARTIFACTS | {MANIFEST_BASENAME}
FORBIDDEN_PREAUTH_MODULES = frozenset(
    {"numpy", "common", "kernels", "overlay", "parity", "source_trace", "tier_c_gate"}
)
EXPECTED_DEPENDENCIES = {
    "../initialization_anchor_layout_expansion_v1/ARTIFACT_SHA256SUMS.txt":
        "cda2a770e03110e3a8c9a31af2fc1b16fa0753836241e8af4c4398039e2e3244",
    "../initialization_anchor_layout_expansion_v1_design_audit/ARTIFACT_SHA256SUMS.txt":
        "8455d86d8143f8b729a37041aca6e37637e4c73d3e29cb970d7e00342ddbd174",
    "../tier_c_grouped_v5_layout_overlay_v1_source_audit/ARTIFACT_SHA256SUMS.txt":
        "16b41a79e663440cff1db6a4b53408160069d2270b565a4aee3da1c19f01af7b",
    "../tier_c_grouped_v5_layout_overlay_v2_source_audit/ARTIFACT_SHA256SUMS.txt":
        "23943f35887e321b285437a8ca517f59bc749a7637500ff1b6bb89af8b8f3705",
    "../initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v3/ARTIFACT_SHA256SUMS.txt":
        "0b0a69cc209037cd9130d6b5bba8b9e920c9b398f32ba2fb0862a1cd4b3a292d",
    "../initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v3_source_audit/ARTIFACT_SHA256SUMS.txt":
        "0c23f0afc98611de8ae36b32c4a9959fe1cb7c16142fcdf44fe131fb529351dd",
    "../initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v3_source_audit/audit_receipt.json":
        "64ec0ab258c916259b7d7b4ce73be6929385c8e68bbc86ad1c25ca0c8c131844",
    "../initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v4/ARTIFACT_SHA256SUMS.txt":
        "dbcc8ce2c7bc63c90fa36f01e6353a72f5c2572170a4a98ad607c11481445f97",
    "../initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v4_source_audit/ARTIFACT_SHA256SUMS.txt":
        "095d94ff55677a4c5542f3c3e711d49952a64df788eb9812fe216a82db0f0d87",
    "../initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v4_source_audit/audit_receipt.json":
        "42be2a15a8ab5ed1383a76c1f2f41a634052a26a0c525066c00f2a456f65e846",
    "../initialization_anchor_oracle_tier_c_grouped_v4/ARTIFACT_SHA256SUMS.txt":
        "19f5b729230f90413a6e1f8c1ef2b0421c94f14635abf57f9b2d7f000599f715",
    "../tier_c_grouped_v4_source_audit/ARTIFACT_SHA256SUMS.txt":
        "fc2f7ad554436e2136c224bd46a5b0c3beee62d69e51c352304b8943da0db028",
    "../tier_c_grouped_v4_calibration_audit/ARTIFACT_SHA256SUMS.txt":
        "0c7ca1241bad8bd3b99c807c50945b371248a6b77820bc72244b5d433604a621",
    "../tier_c_grouped_v4_result_audit/ARTIFACT_SHA256SUMS.txt":
        "ae4e1f38b5602e2c43355e4c68604bca65dbb974724b328ca78e10367e9b992e",
    "../legacy_packed_descriptor_execution_v1/CODE_MANIFEST.sha256":
        "2468baeb6d962b3e8a305b87791fdc4663b29c0f79beaa5957411653ade1c44f",
    "../legacy_packed_descriptor_execution_v1/result_2468baeb/ARTIFACT_MANIFEST.sha256":
        "b44052815d0f3be4e437d56e60cdca4fcdc941fd750bc40fa535363d37080be2",
}


class BootstrapError(RuntimeError):
    """Fail-closed package bootstrap error raised before scientific imports."""


def _revalidate_raw_entrypoint_identity() -> None:
    if RAW_ENTRYPOINT_IDENTITY is None:
        raise BootstrapError(
            "raw entrypoint authentication is available only in the executable verifier"
        )
    try:
        current = _capture_raw_entrypoint_identity()
    except RuntimeError as exc:
        raise BootstrapError(str(exc)) from exc
    for key in (
        "raw_argv0", "raw_argv0_sha256", "verifier_device", "verifier_inode",
        "verifier_bytes", "package_device", "package_inode",
        "component_identity_sha256", "component_count", "mount",
    ):
        if current[key] != RAW_ENTRYPOINT_IDENTITY[key]:
            raise BootstrapError(f"raw entrypoint identity changed during bootstrap: {key}")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular_once(path: Path, label: str) -> bytes:
    """Read one regular non-link file from one authenticated descriptor."""
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise BootstrapError(f"{label} is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise BootstrapError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BootstrapError(f"cannot open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size)
        identity_open = (opened.st_dev, opened.st_ino, opened.st_size)
        if not stat.S_ISREG(opened.st_mode) or identity_before != identity_open:
            raise BootstrapError(f"{label} changed before authenticated open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise BootstrapError(f"{label} was truncated during authenticated read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BootstrapError(f"{label} grew during authenticated read")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != identity_open:
            raise BootstrapError(f"{label} changed during authenticated read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _assert_runtime_firewall() -> None:
    if sys.flags.isolated != 1 or sys.flags.ignore_environment != 1 or sys.flags.no_user_site != 1:
        raise BootstrapError("verifier requires isolated Python (-I)")
    if sys.flags.dont_write_bytecode != 1:
        raise BootstrapError("verifier requires bytecode disabled (-B)")
    if getattr(sys, "pycache_prefix", None) is not None:
        raise BootstrapError("external Python bytecode cache prefixes are forbidden")
    loaded = sorted(name for name in FORBIDDEN_PREAUTH_MODULES if name in sys.modules)
    if loaded:
        raise BootstrapError(f"scientific/package modules loaded before authentication: {loaded}")


def _parse_manifest(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise BootstrapError("artifact manifest must be ASCII") from exc
    if not text.endswith("\n"):
        raise BootstrapError("artifact manifest must end with one LF-terminated row")
    rows: dict[str, str] = {}
    previous_name: str | None = None
    for line in text.splitlines():
        parts = line.split("  ")
        if len(parts) != 2:
            raise BootstrapError("malformed artifact manifest row")
        digest, name = parts
        if not HEX64.fullmatch(digest) or not name or name in rows:
            raise BootstrapError("malformed or duplicate artifact manifest row")
        if Path(name).name != name or "/" in name or "\\" in name:
            raise BootstrapError("artifact manifest members must be plain basenames")
        if previous_name is not None and name <= previous_name:
            raise BootstrapError("artifact manifest rows must be strictly sorted")
        previous_name = name
        rows[name] = digest
    if frozenset(rows) != EXPECTED_ARTIFACTS:
        raise BootstrapError("artifact manifest member set mismatch")
    return rows


def _assert_exact_directory_members(package: Path) -> None:
    try:
        package_stat = os.lstat(package)
    except OSError as exc:
        raise BootstrapError("package directory is missing") from exc
    if stat.S_ISLNK(package_stat.st_mode) or not stat.S_ISDIR(package_stat.st_mode):
        raise BootstrapError("package path must be a real non-symlink directory")
    with os.scandir(package) as iterator:
        entries = {entry.name: entry for entry in iterator}
    if frozenset(entries) != EXPECTED_PACKAGE_MEMBERS:
        extra = sorted(set(entries) - EXPECTED_PACKAGE_MEMBERS)
        missing = sorted(EXPECTED_PACKAGE_MEMBERS - set(entries))
        raise BootstrapError(f"package directory closure mismatch; extra={extra}, missing={missing}")
    for name, entry in entries.items():
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise BootstrapError(f"package member is not a regular non-symlink file: {name}")


def authenticate_package_before_imports(
    package: Path = PACKAGE, *, require_runtime_firewall: bool = True
) -> dict[str, object]:
    """Authenticate exact package closure without importing NumPy or package code."""
    if require_runtime_firewall:
        _revalidate_raw_entrypoint_identity()
        _assert_runtime_firewall()
    _assert_exact_directory_members(package)
    manifest_raw = _read_regular_once(package / MANIFEST_BASENAME, "artifact manifest")
    rows = _parse_manifest(manifest_raw)
    for name, expected in rows.items():
        raw = _read_regular_once(package / name, f"artifact {name}")
        if _sha256_bytes(raw) != expected:
            raise BootstrapError(f"artifact hash mismatch: {name}")
    if require_runtime_firewall:
        _revalidate_raw_entrypoint_identity()
    result: dict[str, object] = {
        "manifest_rows": rows,
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "artifact_count": len(rows),
    }
    if RAW_ENTRYPOINT_IDENTITY is not None:
        result["raw_entrypoint_identity"] = dict(RAW_ENTRYPOINT_IDENTITY)
    return result


def _old_representatives(common, np, seed_start: int, seed_stop: int):
    values = np.empty((seed_stop - seed_start) * 252, dtype=np.uint64)
    cursor = 0
    for seed in range(seed_start, seed_stop):
        for pp in (0, 2, 3):
            for ep in range(8):
                for etp in range(3):
                    for assignment in ((0,) if ep in (0, 7) else (0, 1)):
                        for half in range(2):
                            values[cursor] = common.v4_logical_ordinal(
                                seed, pp, ep, etp, assignment, half, 0
                            )
                            cursor += 1
    if cursor != len(values) or (len(values) > 1 and not np.all(values[1:] > values[:-1])):
        raise BootstrapError("verifier old-representative construction failed")
    return values


def _import_authenticated_modules():
    sys.path.insert(0, str(PACKAGE))
    import numpy as np
    import common
    import kernels
    import overlay
    import tier_c_gate

    modules = {
        "common.py": common,
        "kernels.py": kernels,
        "overlay.py": overlay,
        "tier_c_gate.py": tier_c_gate,
    }
    for basename, module in modules.items():
        if Path(os.path.abspath(module.__file__)) != PACKAGE / basename:
            raise BootstrapError(f"authenticated import resolved outside package: {basename}")
    return np, common, kernels, overlay, tier_c_gate


def verify(*, dependencies: bool) -> dict[str, object]:
    bootstrap = authenticate_package_before_imports()
    np, common, kernels, overlay, tier_c_gate = _import_authenticated_modules()
    # -B must make imports side-effect-free at the package-directory level.
    post_import = authenticate_package_before_imports(require_runtime_firewall=False)
    if post_import != bootstrap:
        raise BootstrapError("package identity changed across authenticated imports")
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("verifier imported a forbidden runtime")
    lock = common.load_candidate_lock()

    dependency_rows: dict[str, str] = {}
    if dependencies:
        for relative, digest in EXPECTED_DEPENDENCIES.items():
            # EXPECTED_DEPENDENCIES is code-frozen, not caller input.  Translate
            # its exactly-one-parent grammar structurally so the common path
            # boundary never receives a spelling containing ``..``.
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or len(relative_path.parts) < 2
                or relative_path.parts[0] != ".."
                or ".." in relative_path.parts[1:]
            ):
                raise common.ProtocolError(f"invalid frozen dependency path: {relative}")
            dependency_target = PACKAGE.parent.joinpath(*relative_path.parts[1:])
            path = common.require_regular_file_before_resolve(
                dependency_target, relative
            )
            if common.sha256_file(path) != digest:
                raise common.ProtocolError(f"dependency hash mismatch: {relative}")
            dependency_rows[relative] = digest

    new = common.representative_ordinals(0, 256)
    full = common.full_representative_ordinals(0, 256)
    old_raw = _old_representatives(common, np, 0, 256)
    old = np.asarray([common.translate_v4_ordinal(int(x)) for x in old_raw], dtype=np.uint64)
    if (len(new), len(old), len(full)) != (164_864, 64_512, 229_376):
        raise common.ProtocolError("seed-shard family accounting mismatch")
    if not np.all(new[1:] > new[:-1]) or not np.all(old[1:] > old[:-1]):
        raise common.ProtocolError("expanded ordinal total order mismatch")
    if np.intersect1d(new, old).size or not np.array_equal(np.union1d(new, old), full):
        raise common.ProtocolError("old/new partition does not equal full expanded union")
    final_old = _old_representatives(common, np, 65_535, 65_536)
    final_translated = np.asarray(
        [common.translate_v4_ordinal(int(x)) for x in final_old], dtype=np.uint64
    )
    if not np.all(final_translated[1:] > final_translated[:-1]):
        raise common.ProtocolError("terminal-seed translation order mismatch")

    old_o = np.tile(old[:2048], (33, 1))
    new_o = np.tile(new[:2048], (33, 1))
    old_q = np.tile(np.arange(2048, dtype=np.float64), (33, 1))
    new_q = old_q + 0.5
    merged = overlay.merge_topk(old_o, old_q, new_o, new_q)
    if merged.domain_ordinals.shape != (33, 4096) or len(merged.union_ordinals) > 135_168:
        raise common.ProtocolError("overlay merge accounting mismatch")

    if "value % 4ULL" not in kernels.CUDA_SOURCE or "value % 10ULL" not in kernels.CUDA_SOURCE:
        raise common.ProtocolError("CUDA expanded ordinal decoder is absent")
    if "{1, 2, 3, 4, 6, 8, 12, 16, 24, 48}" not in kernels.CUDA_SOURCE:
        raise common.ProtocolError("CUDA PP lookup table is absent")
    if common.equivalence_map_sha256() != lock["equivalence_deduplication"]["equivalence_map_sha256"]:
        raise common.ProtocolError("lock equivalence-map hash mismatch")
    audit = common.equivalence_audit()
    ledger = common.physical_ledger()
    if lock["search_cascade"]["stage0"]["maximum_generated_normal_values"] != 42_205_184 * 512:
        raise common.ProtocolError("stage-0 generated-value ledger mismatch")
    if lock["search_cascade"]["stage1"]["maximum_generated_normal_values"] != 135_168 * 48_624:
        raise common.ProtocolError("stage-1 generated-value ledger mismatch")
    if lock["search_cascade"]["end_to_end_maximum_generated_normal_values"] != 28_183_625_728:
        raise common.ProtocolError("end-to-end generated-value ledger mismatch")
    if lock["research_read_ledger"]["eligible_qwen_payload_bytes"] != 31 * 3_145_728:
        raise common.ProtocolError("Qwen research read ledger mismatch")
    if lock["research_read_ledger"]["bound_v4_result_audit_event_topk_bytes"] != 2_360_016 + 9_276 + 334 + 723_930:
        raise common.ProtocolError("v4 research read ledger mismatch")
    if not ledger["passes_read_gate_arithmetic_only"] or ledger["metadata_bytes_total"] != 80:
        raise common.ProtocolError("physical rate/read ledger mismatch")
    if lock["chance_search_control"]["nested_reference"] != ledger["metadata_adjusted_composite_required_capture"]:
        raise common.ProtocolError("composite-screen threshold is not physically derived")
    if lock["chance_search_control"]["standalone_reference"] != ledger["metadata_adjusted_standalone_required_capture"]:
        raise common.ProtocolError("standalone-screen threshold is not physically derived")
    if "primary_gate" in lock["chance_search_control"]:
        raise common.ProtocolError("dimensionally contradictory v1 primary gate survived")
    resume = lock["resume_protocol"]
    if not all(
        resume.get(key) is True
        for key in (
            "every_existing_stage0_shard_replays_all_164864_candidates_on_all_512_frozen_stage0_coordinates",
            "stage0_replay_recomputes_payload_derived_q_for_the_complete_shard_not_only_retained_ordinals",
            "stage0_replay_recomputes_exact_metric_then_ordinal_topk2048_and_compares_every_npz_array",
            "every_existing_stage1_batch_replays_all_48624_frozen_selection_coordinates",
            "recorded_stage1_batch_keys_must_be_the_exact_replayed_union_prefix_and_complete_before_winners",
            "stage1_global_winners_are_independently_reranked_from_replayed_batches",
            "resumed_stage1_winners_require_exact_members_dtypes_shapes_finiteness_canonicality_union_membership_and_replay_equality",
            "resumed_header_overlay_receipt_winner_firewall_result_and_journal_json_reject_duplicate_keys_and_nonfinite_constants",
            "resumed_overlay_state_requires_exact_members_dtypes_shapes_finiteness_order_family_membership_and_replay_equality",
            "journal_is_a_strict_prefix_of_header_stage0_merged_overlay_stage1_winners_firewall_result_grammar",
            "completed_result_event_must_be_last_and_after_winner_firewall",
            "completed_resume_replays_all_stage0_stage1_winner_firewall_selection_validation_and_decision_semantics",
            "completed_resume_requires_exact_full_canonical_result_byte_comparison_before_return",
            "completed_result_never_authorizes_early_return_or_missing_state_repair",
            "every_fresh_full_stage0_q_requires_exact_float64_shape_and_device_wide_finiteness_before_topk",
        )
    ):
        raise common.ProtocolError("v4 full scientific replay lock is incomplete")
    authorization = lock["authorization_protocol"]
    if (
        authorization.get("action") != "QWEN_AUX_33_DOMAIN_SINGLE_RUN"
        or authorization.get("existence_only_authorization_permitted") is not False
        or authorization.get("external_receipt_must_bind_current_package_manifest") is not True
        or authorization.get(
            "original_path_parent_traversal_rejected_lexically_before_normalization_or_io"
        ) is not True
        or authorization.get(
            "every_existing_component_lstat_checked_and_symlinks_rejected_before_resolve_or_open"
        ) is not True
        or authorization.get(
            "qwen_workspace_and_aux_component_io_deferred_until_content_authorization_runtime_parity_and_v4_authentication"
        ) is not True
        or authorization.get(
            "component_checked_absolute_paths_only_passed_to_filesystem_consumers"
        ) is not True
        or authorization.get(
            "candidate_lock_exact_file_and_internal_bytes_bound_by_package_authorization_and_both_audits"
        ) is not True
        or authorization.get("audit_manifest_basename") != "ARTIFACT_SHA256SUMS.txt"
        or authorization.get("audit_receipt_basename") != "audit_receipt.json"
        or authorization.get(
            "audit_package_must_be_exact_flat_regular_nonsymlink_manifest_closure"
        ) is not True
        or authorization.get(
            "audit_package_closure_and_member_bytes_reauthenticated_after_receipt_semantics"
        ) is not True
        or authorization.get(
            "audit_receipt_internal_hash_must_be_independently_recomputed"
        ) is not True
        or authorization.get(
            "audit_receipt_target_verification_access_authorization_and_bindings_exact_member_sets_required"
        ) is not True
        or len(authorization.get("forbidden_legacy_sentinel_paths", ())) != 4
        or "launch_sentinel" in lock["execution"]
    ):
        raise common.ProtocolError("v4 content-bound authorization lock is incomplete")
    runner_source = (PACKAGE / "tier_c_gate.py").read_text(encoding="utf-8")
    common_source = (PACKAGE / "common.py").read_text(encoding="utf-8")
    required_runner_tokens = (
        "replayed = _compute_stage0_shard_state(",
        "replayed_pair = _compare_stage0_shard_replay(",
        "winner_ordinals, winner_q = _run_stage1_strict(",
        "stage1 batch {key} differs from exact payload-derived replay",
        "stage1 winners differ from exact batch replay",
        "recorded stage1 batches are not an exact prefix of the replayed union",
        "legacy existence-only launch sentinel must be absent",
        "common.reject_parent_traversal(supplied_path, supplied_label)",
        "reject_symlink_components_before_normalization",
        '"v4_topk_authentication_receipt_sha256"',
        "members = _audit_manifest_closure(",
        "members_after = _audit_manifest_closure(",
        "expected_internal = common.sha256_bytes(common.canonical_json_bytes(normalized))",
        "_validate_full_stage0_q(access, q, len(candidate_ordinals))",
        "completed_result, result_events_before_final = _prepare_completed_result_replay(",
        "_verify_or_create_header_strict(journal, header)",
        "return _commit_or_verify_result(result_path, journal, result, completed_result)",
        "workspace_aux_closure = common.revalidate_workspace_aux_closure(",
        "production_boundary.revalidate(",
        "boundary_guard=production_boundary",
    )
    if not all(token in runner_source for token in required_runner_tokens):
        raise common.ProtocolError("v4 executable replay/authorization path is incomplete")
    required_common_path_tokens = (
        "original spelling contains forbidden parent traversal",
        "def reject_symlink_components_before_normalization(",
        "stat.S_ISLNK(info.st_mode)",
        "class BoundaryGuard:",
        '"existing_component_identities"',
        "path boundary mount-coordinate alias",
    )
    if not all(token in common_source for token in required_common_path_tokens):
        raise common.ProtocolError("v4 original-spelling path boundary is incomplete")
    if "winner_ordinals, winner_q = BASE._run_stage1(" in runner_source:
        raise common.ProtocolError("v4 production still delegates to inherited stage1 resume")
    source_trace_source = (PACKAGE / "source_trace.py").read_text(encoding="utf-8")
    if (
        'if __name__ == "__main__":' not in source_trace_source
        or "direct execution is forbidden" not in source_trace_source
        or "SOURCE_TRACE_CREATE_ONCE" not in source_trace_source
    ):
        raise common.ProtocolError("v5 source-trace authenticated dispatcher boundary is incomplete")
    if runner_source.index("workspace_aux_closure = common.revalidate_workspace_aux_closure(") > runner_source.index(
        "output_dir = common.ensure_output_directory("
    ):
        raise common.ProtocolError("v5 output root can be created before complete workspace/aux closure")
    preflight = tier_c_gate.cpu_preflight()
    if preflight["end_to_end_max_generated_values"] != 28_183_625_728:
        raise common.ProtocolError("CPU preflight ledger mismatch")
    return {
        "schema": "qwen3_tier_c_grouped_v5_layout_overlay_clean_source_only_verification_v5",
        "status": "PASS_CLEAN_PACKAGE_SOURCE_ONLY_NO_QWEN_OR_CUDA_ACCESS",
        "package_manifest_sha256": bootstrap["manifest_sha256"],
        "raw_entrypoint_identity": bootstrap["raw_entrypoint_identity"],
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "artifact_count": bootstrap["artifact_count"],
        "exact_package_directory_closure": True,
        "authenticated_before_numpy_or_package_imports": True,
        "isolated_python": True,
        "bytecode_read_write_disabled": True,
        "post_import_directory_closure_unchanged": True,
        "dependency_hashes_checked": dependency_rows,
        "logical_candidates": common.LOGICAL_CANDIDATES,
        "new_effective_candidates": common.NEW_EFFECTIVE_CANDIDATES,
        "full_union_effective_candidates": common.FULL_EFFECTIVE_CANDIDATES,
        "representatives_per_new_shard": len(new),
        "representatives_per_full_shard": len(full),
        "old_new_partition_exact": True,
        "translation_order_preserved": True,
        "overlay_merge_max": 135_168,
        "equivalence_audit": audit,
        "physical_ledger": ledger,
        "composite_screen_is_not_final_target_claim": True,
        "complete_stage0_and_stage1_resume_replay_frozen": True,
        "completed_result_full_semantic_replay_frozen": True,
        "audit_manifest_closure_and_internal_seal_recomputation_frozen": True,
        "full_stage0_q_dtype_shape_finiteness_before_topk_frozen": True,
        "content_bound_versioned_authorization_frozen": True,
        "cuda_or_qwen_access": False,
    }


def main() -> int:
    raw_args = list(sys.argv[1:])
    dispatch_args: list[str] | None = None
    dispatch_kind: str | None = None
    dispatch_flags = [
        flag for flag in ("--dispatch-tier-c", "--dispatch-source-trace")
        if flag in raw_args
    ]
    if len(dispatch_flags) > 1:
        raise BootstrapError("only one authenticated scientific dispatcher is permitted")
    if dispatch_flags:
        flag = dispatch_flags[0]
        dispatch_kind = flag
        split = raw_args.index(flag)
        dispatch_args = raw_args[split + 1 :]
        raw_args = raw_args[:split]
        if not dispatch_args:
            raise BootstrapError(f"{flag} requires a frozen runner command")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument("--auth-only", action="store_true")
    parser.add_argument("--v4-run-root", type=Path)
    parser.add_argument("--v4-result-audit", type=Path)
    args = parser.parse_args(raw_args)
    if (args.v4_run_root is None) != (args.v4_result_audit is None):
        raise BootstrapError("named v4 authentication requires both exact paths")
    if args.auth_only and args.v4_run_root is not None:
        raise BootstrapError("--auth-only cannot open named v4 metadata")
    if args.auth_only and dispatch_args is not None:
        raise BootstrapError("--auth-only cannot dispatch a scientific runner")
    if args.v4_run_root is not None and dispatch_args is not None:
        raise BootstrapError("named v4 audit mode and runner dispatch are distinct actions")
    if args.auth_only:
        bootstrap = authenticate_package_before_imports()
        result = {
            "schema": "qwen3_tier_c_grouped_v5_layout_overlay_bootstrap_auth_v5",
            "status": "PASS_EXACT_CLOSED_PACKAGE_AUTHENTICATED_BEFORE_SCIENTIFIC_IMPORTS",
            "package_manifest_sha256": bootstrap["manifest_sha256"],
            "artifact_count": bootstrap["artifact_count"],
            "raw_entrypoint_identity": bootstrap["raw_entrypoint_identity"],
            "isolated_python": True,
            "bytecode_read_write_disabled": True,
        }
    else:
        result = verify(dependencies=not args.skip_dependencies)
        if dispatch_kind == "--dispatch-tier-c":
            import tier_c_gate

            return tier_c_gate.main(dispatch_args)
        if dispatch_kind == "--dispatch-source-trace":
            import source_trace

            return source_trace.main(dispatch_args)
        if args.v4_run_root is not None:
            import common
            import overlay
            import verify_v4_reuse

            result["named_v4_state_verification"] = verify_v4_reuse.verify_named_state(
                common, overlay, args.v4_run_root, args.v4_result_audit
            )
            authenticate_package_before_imports(require_runtime_firewall=False)
    print(json.dumps(result, sort_keys=True, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
