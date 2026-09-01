"""CUDA-free helpers for the Tier-C grouped-v5 layout-overlay gate.

The executable overlay searches only the 42,205,184 anchors that are new to
the frozen grouped-v4 family.  It uses one expanded canonical ordinal space
for both the translated v4 shortlist and the new layouts.  Importing this
module does not import CuPy, PyTorch, Transformer Engine, or Megatron.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import posixpath
import re
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableSequence, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
TIER_B_DIR = PACKAGE_DIR.parent / "initialization_anchor_oracle_tier_b"
TIER_B_COMMON_PATH = TIER_B_DIR / "common.py"
EXPECTED_TIER_B_COMMON_SHA256 = "75d8bbd7af9271ea5d2f099e7d720c1560bcc72864ac88f458095647468e7da3"
CANDIDATE_LOCK_PATH = PACKAGE_DIR / "candidate_lock.json"
CANDIDATE_LOCK_FILE_SHA256 = "cec0b12927340d82c1c8c78cd02b7849c7371b1b16d7f2c1b1d24ae889ada58d"
CANDIDATE_LOCK_INTERNAL_SHA256 = "4b6b3a73e4fca1175adb26b492978a3e42bcf2ce8af4296d45366b104b568a6e"
LOCK_PLACEHOLDER = "TO_BE_FILLED_AFTER_CANONICAL_FREEZE"
CANONICAL_PACKAGE_PATH = Path(
    "/workspace/INT2__compression/INT2_Q_C/research/"
    "initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5"
)
CANONICAL_RESEARCH_ROOT = CANONICAL_PACKAGE_PATH.parent
EXPECTED_PACKAGE_MOUNT_POINT = "/workspace"

SCHEMA = "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_result_v5"
QWEN_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
MCORE_REVISION = "1cb3264479f28b8526db3d335faa9c5ef2183989"
TE_REVISION = "27486e03cfc1fa41f6932dcecdc47c71c47eac3e"
TE_SOURCE_VERSION = "2.18.0+27486e03"
TE_PYPI_VERSION = "2.18.0"

PP_SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)
EP_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
ETP_SIZES = (1, 2, 4, 8)
ASSIGNMENTS = ("contiguous", "round_robin")
HALF_ASSIGNMENTS = ("gate_then_up", "up_then_gate")
STORAGE_ABIS = ("numbered_per_gemm_parameters", "single_grouped_copy_pack")
LAYOUTS_PER_STORED_SEED = 2_560
STORED_SEED_COUNT = 65_536
LOGICAL_CANDIDATES = STORED_SEED_COUNT * LAYOUTS_PER_STORED_SEED
ASSIGNMENT_CLASSES_PER_PP_ETP_HALF = 14
LAYOUTS_PER_EFFECTIVE_PP_STREAM = ASSIGNMENT_CLASSES_PER_PP_ETP_HALF * 4 * 2
FULL_EFFECTIVE_PP_STREAM_COUNT = 8
FULL_REPRESENTATIVES_PER_SEED = FULL_EFFECTIVE_PP_STREAM_COUNT * LAYOUTS_PER_EFFECTIVE_PP_STREAM
FULL_EFFECTIVE_CANDIDATES = STORED_SEED_COUNT * FULL_REPRESENTATIVES_PER_SEED
V4_REPRESENTATIVES_PER_SEED = 252
V4_EFFECTIVE_CANDIDATES = STORED_SEED_COUNT * V4_REPRESENTATIVES_PER_SEED
NEW_REPRESENTATIVES_PER_SEED = 644
NEW_EFFECTIVE_CANDIDATES = STORED_SEED_COUNT * NEW_REPRESENTATIVES_PER_SEED
# The frozen Tier-B orchestration calls this name for the family searched in
# stage 0.  In the overlay package that family is deliberately new-only.
REPRESENTATIVES_PER_SEED = NEW_REPRESENTATIVES_PER_SEED
EFFECTIVE_CANDIDATES = NEW_EFFECTIVE_CANDIDATES
SEED_SHARD_SIZE = 256
SEED_SHARD_COUNT = STORED_SEED_COUNT // SEED_SHARD_SIZE
MAX_REPRESENTATIVES_PER_SHARD = SEED_SHARD_SIZE * NEW_REPRESENTATIVES_PER_SEED
STAGE0_TOP_K = 2048
FULL_FIT = 32_768
FULL_SCORE = 32_768
STAGE0_FIT = 256
STAGE0_SCORE = 256

DOMAIN_IDS = ("source",) + tuple(f"gaussian_{index:02d}" for index in range(16)) + tuple(
    f"scramble_{index:02d}" for index in range(16)
)
NULL_DOMAIN_IDS = DOMAIN_IDS[1:]

TARGET_F = 0.8
CURRENT_F = 0.9888693569009007
COMPOSITE_F = 0.936397621
TARGET_WEIGHTS = 28_311_552
TARGET_MATRIX_COUNT = 18
METADATA_BYTES = 80
CURRENT_WORST_READ_AMP = 1.169444
WEIGHTS_PER_EXPERT = 4_718_592
READ_LEDGER_REFERENCE_BPW = 2.5
STRICT_BPW_CAP = 2.15

SOURCE_MANIFEST_BASENAME = "agent_rd_structure_diag_cross_expert_sources.json"
SOURCE_FREEZE_BASENAME = "agent_rd_structure_diag_cross_expert_freeze.json"
EXPECTED_SOURCE_MANIFEST_SHA256 = "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782"
EXPECTED_SOURCE_FREEZE_SHA256 = "37743eaf6cb70c2bc68704dcf4d60e013552b76c11daf1ab4855f64ad4417193"
EXPECTED_EXCLUSION_MANIFEST_SHA256 = "3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55"
EXCLUDED_TENSOR = "model.layers.15.mlp.experts.0.up_proj.weight"
EXCLUDED_BASENAME = "l15e0_up.bf16.bin"
SELECTION_EXPERTS = (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)
VALIDATION_EXPERTS = (24, 56, 88, 120)
CALIBRATION_SELECTION_IDENTITIES = tuple(
    (expert, role)
    for expert in SELECTION_EXPERTS
    for role in ("up", "down")
    if not (expert == 0 and role == "up")
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol violation."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_unresolved(path: Path) -> Path:
    """Return an absolute spelling without following the target or its ancestors."""
    return Path(os.path.abspath(os.fspath(path)))


def reject_parent_traversal(path: Path, label: str) -> None:
    """Reject ``..`` in the original spelling before any normalization or I/O.

    ``abspath``/``normpath`` must not run first: on POSIX, ``LINK/../target``
    traverses ``LINK`` before applying ``..``.  Normalizing that spelling would
    erase the component whose link status has to be checked.  Production path
    arguments therefore use a deliberately smaller, canonical lexical class.
    """
    try:
        original = Path(os.fspath(path))
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{label} has an invalid path spelling") from error
    if ".." in original.parts:
        raise ProtocolError(
            f"{label} original spelling contains forbidden parent traversal"
        )


def lstat_or_none(path: Path):
    """Like lexists, but returns the lstat result needed for type checks."""
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def reject_symlink_components_before_normalization(
    path: Path,
    label: str,
    *,
    require_exists: bool,
) -> Path:
    """Check the original path chain before returning its absolute spelling.

    Relative spellings are anchored at the current directory without first
    collapsing components.  Every existing component is inspected with
    ``lstat`` and no link is followed.  Missing suffixes are permitted only for
    create-new destinations.  The returned spelling is safe to normalize
    because parent traversal was rejected first.
    """
    reject_parent_traversal(path, label)
    original = Path(os.fspath(path))
    anchored = original if original.is_absolute() else Path.cwd() / original
    parts = anchored.parts
    if not parts:
        raise ProtocolError(f"{label} has no path components")
    cursor = Path(parts[0])
    missing_seen = False
    for index, part in enumerate(parts[1:], start=1):
        cursor = cursor / part
        info = lstat_or_none(cursor)
        if info is None:
            missing_seen = True
            if require_exists:
                raise ProtocolError(f"{label} path component is missing: {cursor}")
            continue
        if missing_seen:
            raise ProtocolError(f"{label} path chain changed during component walk")
        if stat.S_ISLNK(info.st_mode):
            raise ProtocolError(f"{label} path contains a symlink component: {cursor}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ProtocolError(
                f"{label} path contains a non-directory component: {cursor}"
            )
    return absolute_unresolved(anchored)


def require_regular_file_before_resolve(path: Path, label: str) -> Path:
    """Reject an original symlink/non-file before resolving, then recheck."""
    unresolved = reject_symlink_components_before_normalization(
        path, label, require_exists=True
    )
    info = lstat_or_none(unresolved)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProtocolError(f"{label} must be an existing regular non-symlink file")
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise ProtocolError(f"{label} cannot be resolved after lstat") from error
    resolved_info = lstat_or_none(resolved)
    if (
        resolved_info is None
        or stat.S_ISLNK(resolved_info.st_mode)
        or not stat.S_ISREG(resolved_info.st_mode)
    ):
        raise ProtocolError(f"{label} resolved target is not a regular file")
    return resolved


def _check_or_create_directory_chain(directory: Path, *, create: bool, label: str) -> Path:
    """Walk an absolute directory chain without following any symlink component."""
    absolute = reject_symlink_components_before_normalization(
        directory, label, require_exists=False
    )
    parts = absolute.parts
    if not parts:
        raise ProtocolError(f"{label} has no absolute path components")
    cursor = Path(parts[0])
    for part in parts[1:]:
        cursor = cursor / part
        info = lstat_or_none(cursor)
        if info is None:
            if not create:
                continue
            try:
                os.mkdir(cursor)
            except FileExistsError as error:
                raise ProtocolError(f"{label} directory appeared during create-new") from error
            info = lstat_or_none(cursor)
        if info is not None and (
            stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
        ):
            raise ProtocolError(f"{label} contains a symlink or non-directory component: {cursor}")
    return absolute


def preflight_create_new_file(path: Path, label: str) -> Path:
    """Reject existing objects, including dangling symlinks, without resolving."""
    absolute = reject_symlink_components_before_normalization(
        path, label, require_exists=False
    )
    _check_or_create_directory_chain(absolute.parent, create=False, label=f"{label} parent")
    if lstat_or_none(absolute) is not None:
        raise ProtocolError(f"{label} already exists or is a symlink")
    return absolute


def preflight_output_directory(path: Path, *, allow_existing: bool, label: str) -> Path:
    """Validate an output-directory spelling without resolving or creating it."""
    absolute = reject_symlink_components_before_normalization(
        path, label, require_exists=False
    )
    _check_or_create_directory_chain(absolute.parent, create=False, label=f"{label} parent")
    info = lstat_or_none(absolute)
    if info is None:
        return absolute
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ProtocolError(f"{label} is an existing or dangling symlink/non-directory")
    if not allow_existing:
        raise ProtocolError(f"{label} already exists")
    return absolute


def ensure_output_directory(path: Path, *, allow_existing: bool, label: str) -> Path:
    """Safely create a missing directory tree, or validate an allowed resume tree."""
    absolute = preflight_output_directory(path, allow_existing=allow_existing, label=label)
    info = lstat_or_none(absolute)
    if info is not None:
        return absolute
    _check_or_create_directory_chain(absolute, create=True, label=label)
    info = lstat_or_none(absolute)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ProtocolError(f"{label} create-new directory verification failed")
    return absolute


@contextmanager
def open_create_new(path: Path, *, binary: bool, label: str):
    """Open a file with O_EXCL/O_NOFOLLOW after symlink-safe parent validation."""
    absolute = preflight_create_new_file(path, label)
    _check_or_create_directory_chain(absolute.parent, create=True, label=f"{label} parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if binary and hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(absolute, flags, 0o600)
    except (FileExistsError, OSError) as error:
        raise ProtocolError(f"{label} create-new open failed") from error
    if binary:
        handle = os.fdopen(descriptor, "wb")
    else:
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
    try:
        yield handle
    finally:
        handle.close()


def write_json_create_new(path: Path, value: Mapping[str, Any], label: str) -> Path:
    # Preserve the caller's spelling until the component walk has run.
    absolute = preflight_create_new_file(path, label)
    with open_create_new(absolute, binary=False, label=label) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return absolute


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def require_canonical_absolute_spelling(path: Path, label: str) -> Path:
    """Accept one absolute lexical spelling before any normalization or I/O."""
    try:
        raw = os.fspath(path)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{label} has an invalid path spelling") from error
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ProtocolError(f"{label} has an empty or malformed path spelling")
    if not os.path.isabs(raw):
        raise ProtocolError(f"{label} must be an absolute path")
    original = Path(raw)
    if ".." in original.parts:
        raise ProtocolError(f"{label} contains forbidden parent traversal")
    if "." in original.parts:
        raise ProtocolError(f"{label} contains a forbidden lexical component")
    if os.path.normpath(raw) != raw:
        raise ProtocolError(f"{label} must use one canonical lexical spelling")
    return original


def _decode_mountinfo_field(value: str) -> str:
    return re.sub(
        r"\\(040|011|012|134)",
        lambda match: {"040": " ", "011": "\t", "012": "\n", "134": "\\"}[
            match.group(1)
        ],
        value,
    )


def _mountinfo_identity(path: Path) -> dict[str, Any]:
    """Map an absolute path to its mount and underlying filesystem coordinate."""
    raw_path = os.fspath(path)
    try:
        mountinfo = Path("/proc/self/mountinfo").read_bytes()
    except OSError as error:
        raise ProtocolError("Linux mountinfo is mandatory for path-boundary checks") from error
    candidates: list[dict[str, Any]] = []
    try:
        lines = mountinfo.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ProtocolError("mountinfo is not UTF-8") from error
    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            raise ProtocolError("malformed mountinfo row")
        fields = left.split()
        tail = right.split()
        if len(fields) < 6 or len(tail) < 3:
            raise ProtocolError("short mountinfo row")
        mount_point = _decode_mountinfo_field(fields[4])
        if raw_path != mount_point and not raw_path.startswith(
            mount_point.rstrip("/") + "/"
        ):
            continue
        relative = posixpath.relpath(raw_path, mount_point)
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
                "mount_row_sha256": sha256_bytes(line.encode("utf-8")),
            }
        )
    if not candidates:
        raise ProtocolError(f"no mount identity covers path: {path}")
    return max(candidates, key=lambda row: len(str(row["mount_point"])))


def _path_boundary_descriptor(
    path: Path,
    label: str,
    *,
    require_exists: bool,
    expected_kind: str,
) -> dict[str, Any]:
    canonical = require_canonical_absolute_spelling(path, label)
    checked = reject_symlink_components_before_normalization(
        canonical, label, require_exists=require_exists
    )
    if checked != canonical:
        raise ProtocolError(f"{label} changed during lexical component walk")
    component_rows: list[dict[str, Any]] = []
    parts = checked.parts
    cursor = Path(parts[0])
    root_info = lstat_or_none(cursor)
    if root_info is None or stat.S_ISLNK(root_info.st_mode):
        raise ProtocolError(f"{label} filesystem root is missing or linked")
    component_rows.append({
        "path": os.fspath(cursor),
        "device": int(root_info.st_dev),
        "inode": int(root_info.st_ino),
        "mode": int(root_info.st_mode),
    })
    for part in parts[1:]:
        cursor = cursor / part
        component_info = lstat_or_none(cursor)
        if component_info is None:
            break
        if stat.S_ISLNK(component_info.st_mode):
            raise ProtocolError(f"{label} has a symlink component: {cursor}")
        component_rows.append({
            "path": os.fspath(cursor),
            "device": int(component_info.st_dev),
            "inode": int(component_info.st_ino),
            "mode": int(component_info.st_mode),
        })
    info = lstat_or_none(checked)
    if require_exists and info is None:
        raise ProtocolError(f"{label} is missing")
    if info is not None:
        if stat.S_ISLNK(info.st_mode):
            raise ProtocolError(f"{label} is a symlink")
        if expected_kind == "file" and not stat.S_ISREG(info.st_mode):
            raise ProtocolError(f"{label} must be a regular file")
        if expected_kind == "directory" and not stat.S_ISDIR(info.st_mode):
            raise ProtocolError(f"{label} must be a real directory")
    nearest = checked
    nearest_info = info
    if nearest_info is not None and not stat.S_ISDIR(nearest_info.st_mode):
        nearest = checked.parent
        nearest_info = lstat_or_none(nearest)
    while nearest_info is None:
        parent = nearest.parent
        if parent == nearest:
            raise ProtocolError(f"{label} has no existing ancestor")
        nearest = parent
        nearest_info = lstat_or_none(nearest)
    if stat.S_ISLNK(nearest_info.st_mode) or not stat.S_ISDIR(nearest_info.st_mode):
        raise ProtocolError(f"{label} nearest existing ancestor is not a real directory")
    mount = _mountinfo_identity(checked)
    return {
        "label": label,
        "path": os.fspath(checked),
        "expected_kind": expected_kind,
        "exists": info is not None,
        "device": None if info is None else int(info.st_dev),
        "inode": None if info is None else int(info.st_ino),
        "mode": None if info is None else int(info.st_mode),
        "bytes": None if info is None or not stat.S_ISREG(info.st_mode) else int(info.st_size),
        "nearest_existing_ancestor": os.fspath(nearest),
        "nearest_ancestor_device": int(nearest_info.st_dev),
        "nearest_ancestor_inode": int(nearest_info.st_ino),
        "existing_component_identities": component_rows,
        "mount": mount,
    }


def _path_is_ancestor_or_equal(left: str, right: str) -> bool:
    left_normal = posixpath.normpath(left)
    right_normal = posixpath.normpath(right)
    return right_normal == left_normal or right_normal.startswith(
        left_normal.rstrip("/") + "/"
    )


def _assert_boundary_pair_disjoint(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    if _path_is_ancestor_or_equal(str(left["path"]), str(right["path"])) or (
        _path_is_ancestor_or_equal(str(right["path"]), str(left["path"]))
    ):
        raise ProtocolError(
            f"path boundary ancestry overlap: {left['label']} vs {right['label']}"
        )
    if (
        left["exists"]
        and right["exists"]
        and (left["device"], left["inode"]) == (right["device"], right["inode"])
    ):
        raise ProtocolError(
            f"path boundary device/inode alias: {left['label']} vs {right['label']}"
        )
    left_mount = left["mount"]
    right_mount = right["mount"]
    if left_mount["major_minor"] == right_mount["major_minor"] and (
        _path_is_ancestor_or_equal(
            str(left_mount["filesystem_path"]), str(right_mount["filesystem_path"])
        )
        or _path_is_ancestor_or_equal(
            str(right_mount["filesystem_path"]), str(left_mount["filesystem_path"])
        )
    ):
        raise ProtocolError(
            f"path boundary mount-coordinate alias: {left['label']} vs {right['label']}"
        )


class BoundaryGuard:
    """Revalidating no-link, inode and mount-coordinate output boundary."""

    def __init__(
        self,
        action: str,
        *,
        outputs: Sequence[tuple[str, Path, str, bool]],
        inputs: Sequence[tuple[str, Path, str]],
        protected_roots: Sequence[tuple[str, Path]] | None = None,
    ):
        if not outputs:
            raise ProtocolError("path boundary requires at least one output")
        self.action = action
        self._output_specs = tuple(outputs)
        self._input_specs = tuple(inputs)
        roots = protected_roots or (("frozen research/package root", CANONICAL_RESEARCH_ROOT),)
        self._protected_specs = tuple(
            (label, path, "directory") for label, path in roots
        )
        self._inputs = tuple(
            _path_boundary_descriptor(path, label, require_exists=True, expected_kind=kind)
            for label, path, kind in (*self._input_specs, *self._protected_specs)
        )
        self._outputs = [
            _path_boundary_descriptor(
                path, label, require_exists=allow_existing, expected_kind=kind
            )
            for label, path, kind, allow_existing in self._output_specs
        ]
        for descriptor, spec in zip(self._outputs, self._output_specs):
            allow_existing = spec[3]
            if descriptor["exists"] and not allow_existing:
                raise ProtocolError(f"{descriptor['label']} already exists")
        self._bound_output_objects: dict[str, tuple[int, int, int]] = {}
        for descriptor in self._outputs:
            if descriptor["exists"]:
                self._bound_output_objects[str(descriptor["path"])] = (
                    int(descriptor["device"]),
                    int(descriptor["inode"]),
                    int(descriptor["mode"]),
                )
        self._assert_all_disjoint(self._outputs, self._inputs)

    @staticmethod
    def _assert_all_disjoint(
        outputs: Sequence[Mapping[str, Any]],
        inputs: Sequence[Mapping[str, Any]],
    ) -> None:
        for index, left in enumerate(outputs):
            for right in outputs[index + 1 :]:
                _assert_boundary_pair_disjoint(left, right)
            for right in inputs:
                _assert_boundary_pair_disjoint(left, right)

    def revalidate(self, phase: str) -> None:
        current_inputs = tuple(
            _path_boundary_descriptor(path, label, require_exists=True, expected_kind=kind)
            for label, path, kind in (*self._input_specs, *self._protected_specs)
        )
        for frozen, current in zip(self._inputs, current_inputs):
            for key in (
                "path", "exists", "device", "inode", "mode", "bytes",
                "existing_component_identities", "mount",
            ):
                if frozen[key] != current[key]:
                    raise ProtocolError(
                        f"{self.action} input/protected identity changed at {phase}: "
                        f"{current['label']} ({key})"
                    )
        current_outputs = [
            _path_boundary_descriptor(
                path, label, require_exists=False, expected_kind=kind
            )
            for label, path, kind, _allow_existing in self._output_specs
        ]
        for frozen, current in zip(self._outputs, current_outputs):
            if (
                frozen["path"] != current["path"]
                or frozen["mount"] != current["mount"]
            ):
                raise ProtocolError(
                    f"{self.action} output mount/path identity changed at {phase}"
                )
            frozen_components = frozen["existing_component_identities"]
            current_components = current["existing_component_identities"]
            if current_components[: len(frozen_components)] != frozen_components:
                raise ProtocolError(
                    f"{self.action} output component identity changed at {phase}"
                )
            if (
                (frozen["exists"] or str(current["path"]) in self._bound_output_objects)
                and not current["exists"]
            ):
                raise ProtocolError(
                    f"{self.action} bound output disappeared at {phase}"
                )
            if current["exists"]:
                identity = (
                    int(current["device"]), int(current["inode"]), int(current["mode"])
                )
                prior = self._bound_output_objects.setdefault(str(current["path"]), identity)
                if prior != identity:
                    raise ProtocolError(
                        f"{self.action} output object identity changed at {phase}"
                    )
        self._assert_all_disjoint(current_outputs, current_inputs)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "qwen3_tier_c_grouped_v5_path_boundary_v1",
            "action": self.action,
            "outputs": [dict(row) for row in self._outputs],
            "inputs_and_protected_roots": [dict(row) for row in self._inputs],
            "pairwise_lexical_inode_and_mount_disjoint": True,
            "revalidation_required_before_every_create_new": True,
        }

    def authorization_receipt(self) -> dict[str, Any]:
        """Stable content binding: input objects plus output mount coordinate."""
        output_rows = [
            {
                "label": row["label"],
                "path": row["path"],
                "expected_kind": row["expected_kind"],
                "mount": row["mount"],
            }
            for row in self._outputs
        ]
        return {
            "schema": "qwen3_tier_c_grouped_v5_path_boundary_authorization_v1",
            "action": self.action,
            "outputs": output_rows,
            "inputs_and_protected_roots": [
                dict(row)
                for row in self._inputs
                if row["label"] != "production authorization input"
            ],
            "pairwise_lexical_inode_and_mount_disjoint": True,
            "raw_absolute_canonical_spellings_required": True,
        }


def strict_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise ProtocolError(
            f"{label} keys mismatch; missing={sorted(expected_set-observed)}, "
            f"extra={sorted(observed-expected_set)}"
        )


def _load_tier_b_common():
    dependency_path = require_regular_file_before_resolve(
        TIER_B_COMMON_PATH, "byte-frozen Tier-B common dependency"
    )
    if sha256_file(dependency_path) != EXPECTED_TIER_B_COMMON_SHA256:
        raise ProtocolError("byte-frozen Tier-B common dependency SHA-256 mismatch")
    name = "_qwen_frozen_tier_b_common_for_tier_c_grouped_v5_layout_overlay"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, dependency_path)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot load byte-frozen Tier-B common dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TIER_B = _load_tier_b_common()
TIER_A = TIER_B.TIER_A
SourceRow = TIER_B.SourceRow
PlanRow = TIER_B.PlanRow
ROWS = TIER_B.ROWS
COLUMNS = TIER_B.COLUMNS
WEIGHTS_PER_MATRIX = TIER_B.WEIGHTS_PER_MATRIX
EXPECTED_TIER_A_COMMON_SHA256 = TIER_B.EXPECTED_TIER_A_COMMON_SHA256


def load_candidate_lock(path: Path = CANDIDATE_LOCK_PATH) -> dict[str, Any]:
    if PACKAGE_DIR != CANONICAL_PACKAGE_PATH:
        raise ProtocolError("v5 package was imported outside its frozen canonical path")
    package_descriptor = _path_boundary_descriptor(
        PACKAGE_DIR, "canonical v5 package", require_exists=True, expected_kind="directory"
    )
    if package_descriptor["mount"]["mount_point"] != EXPECTED_PACKAGE_MOUNT_POINT:
        raise ProtocolError("v5 package mount identity differs from the frozen boundary")
    path = require_regular_file_before_resolve(path, "Tier-C grouped-v5 overlay candidate lock")
    raw = path.read_bytes()
    if sha256_bytes(raw) != CANDIDATE_LOCK_FILE_SHA256:
        raise ProtocolError("Tier-C grouped-v5 candidate lock file SHA-256 mismatch")
    lock = json.loads(raw.decode("utf-8"))
    if lock.get("lock_sha256") != CANDIDATE_LOCK_INTERNAL_SHA256:
        raise ProtocolError("Tier-C grouped-v5 internal lock literal mismatch")
    matches = list(re.finditer(rb'("lock_sha256"\s*:\s*")([0-9a-f]{64})(")', raw))
    if len(matches) != 1:
        raise ProtocolError("Tier-C grouped-v5 lock must contain exactly one internal seal")
    match = matches[0]
    normalized = raw[: match.start(2)] + LOCK_PLACEHOLDER.encode() + raw[match.end(2) :]
    if sha256_bytes(normalized) != CANDIDATE_LOCK_INTERNAL_SHA256:
        raise ProtocolError("Tier-C grouped-v5 placeholder-normalized seal mismatch")
    if lock.get("schema") != "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_candidate_lock_v5":
        raise ProtocolError("Tier-C grouped-v5 lock schema mismatch")
    if lock.get("status") != "FROZEN_SOURCE_ONLY_V5_RAW_ENTRYPOINT_AND_OUTPUT_BOUNDARY_NOT_GPU_OR_PAYLOAD_AUTHORIZATION":
        raise ProtocolError("Tier-C grouped-v5 lock status mismatch")
    if lock.get("sealed") is not True:
        raise ProtocolError("Tier-C grouped-v5 lock is not sealed")
    if lock["logical_key_space"]["logical_candidate_count"] != LOGICAL_CANDIDATES:
        raise ProtocolError("logical candidate count mismatch")
    dedup = lock["equivalence_deduplication"]
    if dedup["full_union_distinct_anchor_count"] != FULL_EFFECTIVE_CANDIDATES:
        raise ProtocolError("full-union distinct-anchor count mismatch")
    if dedup["new_distinct_anchor_count"] != NEW_EFFECTIVE_CANDIDATES:
        raise ProtocolError("new distinct-anchor count mismatch")
    chance = lock.get("chance_search_control", {})
    if "primary_gate" in chance:
        raise ProtocolError("v4 lock must not retain the dimensionally contradictory v1 primary gate")
    if chance.get("nested_reference") != 0.1456888483858212:
        raise ProtocolError("v4 composite-screen threshold mismatch")
    if chance.get("standalone_reference") != 0.19102060916075425:
        raise ProtocolError("v4 standalone-screen threshold mismatch")
    if chance.get("deprecated_0_160964_design_value_is_not_a_capture_gate") is not True:
        raise ProtocolError("v4 capture-unit claim boundary is absent")
    if chance.get("final_20_percent_below_gaussian_claim_emitted_by_this_gate") is not False:
        raise ProtocolError("v4 auxiliary gate overclaims the final rate-distortion target")
    package_auth = lock.get("package_authentication", {})
    expected_predecessor_seals = {
        "successor_of_v4_artifact_manifest_sha256":
            "dbcc8ce2c7bc63c90fa36f01e6353a72f5c2572170a4a98ad607c11481445f97",
        "successor_of_v4_candidate_lock_file_sha256":
            "d87ba7fde52c105df89d5e8e2adcbc45486dee0c9d01b1b9a79e7f5282fd9af3",
        "successor_of_v4_block_audit_manifest_sha256":
            "095d94ff55677a4c5542f3c3e711d49952a64df788eb9812fe216a82db0f0d87",
        "successor_of_v4_block_audit_receipt_file_sha256":
            "42be2a15a8ab5ed1383a76c1f2f41a634052a26a0c525066c00f2a456f65e846",
        "successor_of_v4_block_audit_receipt_internal_sha256":
            "c501814e3a3c9cac4945c01629f01dbb41d0b273cf9974189e75567944501285",
    }
    if any(package_auth.get(key) != value for key, value in expected_predecessor_seals.items()):
        raise ProtocolError("v5 package authentication does not bind its exact v4 predecessor")
    if (
        package_auth.get("manifest_basename") != "ARTIFACT_SHA256SUMS.txt"
        or package_auth.get("allowed_members_are_exactly_manifest_plus_rows") is not True
        or package_auth.get("directories_symlinks_devices_and_unexpected_members_forbidden") is not True
        or package_auth.get("authentication_precedes_numpy_and_all_package_imports") is not True
        or package_auth.get("required_verifier_interpreter_flags") != ["-B", "-I"]
        or package_auth.get("external_python_environment_and_user_site_ignored") is not True
        or package_auth.get("package_bytecode_read_or_write_permitted") is not False
        or package_auth.get("post_import_directory_closure_reauthenticated") is not True
        or package_auth.get("canonical_package_path") != os.fspath(CANONICAL_PACKAGE_PATH)
        or package_auth.get("canonical_verifier_argv0")
        != os.fspath(CANONICAL_PACKAGE_PATH / "verify_prelaunch.py")
        or package_auth.get("canonical_package_mount_point") != EXPECTED_PACKAGE_MOUNT_POINT
        or package_auth.get("raw_argv0_captured_before_any_normalized___file___derivation") is not True
        or package_auth.get("raw_argv0_absolute_canonical_lexical_and_exact_frozen_path_required") is not True
        or package_auth.get("raw_argv0_all_components_lstat_and_verifier_single_descriptor_bound") is not True
        or package_auth.get("package_directory_single_descriptor_bound") is not True
        or package_auth.get("raw_argv0_verifier_and_package_device_inode_mount_identity_bound") is not True
        or package_auth.get("authorized_scientific_dispatchers")
        != ["--dispatch-tier-c", "--dispatch-source-trace"]
    ):
        raise ProtocolError("v5 package closure/bootstrap protocol is not frozen")
    provenance = lock.get("provenance", {})
    if provenance.get("v1_independent_source_audit_manifest_sha256") != (
        "16b41a79e663440cff1db6a4b53408160069d2270b565a4aee3da1c19f01af7b"
    ):
        raise ProtocolError("v4 does not bind the sealed v1 independent audit")
    if provenance.get("v1_independent_source_audit_receipt_sha256") != (
        "dd3135cdea071aad45157ac567d663e57b9cc4a58606ea2d00a8bd356fcac70b"
    ):
        raise ProtocolError("v4 does not bind the v1 audit receipt")
    if provenance.get("v2_independent_source_audit_manifest_sha256") != (
        "23943f35887e321b285437a8ca517f59bc749a7637500ff1b6bb89af8b8f3705"
    ):
        raise ProtocolError("v4 does not bind the sealed v2 independent audit")
    if provenance.get("v2_independent_source_audit_receipt_sha256") != (
        "22ec1e90d9b786dc804da86fb89faaa9eb1276434b24627c5db5216a73ceb8f0"
    ):
        raise ProtocolError("v4 does not bind the v2 audit receipt")
    if provenance.get("v3_independent_source_audit_manifest_sha256") != (
        "0c23f0afc98611de8ae36b32c4a9959fe1cb7c16142fcdf44fe131fb529351dd"
    ):
        raise ProtocolError("v4 does not bind the sealed v3 independent audit")
    if provenance.get("v3_independent_source_audit_receipt_sha256") != (
        "64ec0ab258c916259b7d7b4ce73be6929385c8e68bbc86ad1c25ca0c8c131844"
    ) or provenance.get("v3_independent_source_audit_internal_sha256") != (
        "079576831339fe0df793e74d751471e12b5e446c18837885547ec4e4ecd12eed"
    ):
        raise ProtocolError("v4 does not bind the v3 BLOCK receipt seals")
    resume = lock.get("resume_protocol", {})
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
        raise ProtocolError("v4 full scientific replay protocol mismatch")
    authorization = lock.get("authorization_protocol", {})
    expected_legacy_sentinels = [
        "/tmp/init_anchor_tier_c_grouped_v5_layout_overlay_release_v1",
        "/tmp/init_anchor_tier_c_grouped_v5_layout_overlay_release_v2",
        "/tmp/init_anchor_tier_c_grouped_v5_layout_overlay_release_v3.json",
        "/tmp/init_anchor_tier_c_grouped_v5_layout_overlay_release_v4.json",
    ]
    expected_audit_member_sets = {
        "source_top_level": ["schema", "status", "audited_target", "verification", "access_attestation", "authorization", "audit_receipt_sha256"],
        "calibration_top_level": ["schema", "status", "audited_target", "verification", "access_attestation", "bindings", "authorization", "audit_receipt_sha256"],
        "audited_target": ["artifact_manifest_sha256", "candidate_lock_file_sha256", "candidate_lock_internal_sha256", "runner_sha256", "common_sha256", "kernels_sha256", "overlay_sha256", "parity_sha256"],
        "verification": ["manifest_closure_verified", "receipt_internal_sha256_recomputed", "candidate_lock_internal_sha256_recomputed", "exact_schema_and_member_sets_verified", "access_all_zero"],
        "access_attestation": ["qwen_or_model_payload_manifest_or_directory_accessed", "torch_cupy_cuda_transformer_engine_or_megatron_imported", "gpu_used", "network_accessed", "producer_artifacts_modified"],
        "source_authorization": ["source_package_passed", "source_free_gpu_calibration_authorized", "qwen_payload_or_manifest_launch_authorized", "production_run_authorized"],
        "calibration_authorization": ["source_package_passed", "source_free_calibration_passed", "qwen_payload_or_manifest_launch_authorized", "production_run_authorized"],
        "calibration_bindings": ["calibration_receipt_sha256", "calibration_receipt_internal_sha256", "source_audit_manifest_sha256", "source_audit_receipt_sha256"],
    }
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
        or authorization.get("audit_receipt_exact_member_sets") != expected_audit_member_sets
        or authorization.get(
            "audit_manifest_rows_must_be_sorted_unique_lowercase_sha256_and_include_receipt"
        ) is not True
        or authorization.get("source_audit_receipt_schema")
        != "tier_c_grouped_v5_layout_overlay_v5_independent_source_audit_receipt_v1"
        or authorization.get("source_audit_required_status")
        != "PASS_SOURCE_PACKAGE_AUTHORIZED_FOR_SOURCE_FREE_GPU_CALIBRATION"
        or authorization.get("calibration_audit_receipt_schema")
        != "tier_c_grouped_v5_layout_overlay_v5_calibration_audit_receipt_v1"
        or authorization.get("calibration_audit_required_status")
        != "PASS_CALIBRATION_AUTHORIZED_FOR_SINGLE_QWEN_RUN"
        or authorization.get(
            "source_and_calibration_audit_manifest_and_receipt_bytes_hash_bound"
        ) is not True
        or authorization.get("forbidden_legacy_sentinel_paths")
        != expected_legacy_sentinels
        or authorization.get("any_existing_legacy_sentinel_fails_closed") is not True
        or authorization.get(
            "external_receipt_must_bind_path_boundary_device_inode_and_mount_receipt"
        ) is not True
        or lock.get("execution", {}).get("production_authorization_receipt")
        != authorization.get("receipt_path")
        or "launch_sentinel" in lock.get("execution", {})
    ):
        raise ProtocolError("v5 content-bound authorization protocol mismatch")
    boundary = lock.get("path_boundary_protocol", {})
    if boundary != {
        "schema": "qwen3_tier_c_grouped_v5_path_boundary_protocol_v1",
        "all_runtime_paths_require_raw_absolute_canonical_lexical_spelling": True,
        "all_existing_components_lstat_checked_without_following_links": True,
        "all_existing_component_device_inode_mode_sequences_bound_and_revalidated": True,
        "existing_objects_bind_device_inode_mode_and_mountinfo_identity": True,
        "prospective_outputs_bind_nearest_existing_ancestor_and_underlying_mount_coordinate": True,
        "bind_mount_aliases_rejected_by_major_minor_plus_mount_root_relative_coordinate": True,
        "outputs_rejected_if_equal_ancestor_or_descendant_of_any_input_or_frozen_research_root": True,
        "calibration_output_disjoint_from_source_trace_and_frozen_research_root": True,
        "source_trace_output_disjoint_from_mcore_te_and_frozen_research_root": True,
        "production_output_disjoint_from_workspace_aux_calibration_trace_grouped_v4_audits_authorization_and_frozen_research_root": True,
        "production_authorization_binds_stable_input_inode_mount_and_output_mount_coordinate_receipt": True,
        "boundary_revalidated_immediately_before_every_create_new_operation": True,
        "workspace_manifests_and_complete_relevant_aux_name_inode_closure_revalidated_before_output_or_journal": True,
        "excluded_payload_lstat_stat_or_open_during_closure_revalidation": False,
    }:
        raise ProtocolError("v5 path-boundary protocol mismatch")
    if lock["equivalence_deduplication"]["equivalence_map_sha256"] != equivalence_map_sha256():
        raise ProtocolError("equivalence-map SHA-256 mismatch")
    return lock


def default_workspace_root() -> Path:
    return PACKAGE_DIR.parents[2]


def _event(events: MutableSequence[dict[str, Any]], action: str, **fields: Any) -> None:
    events.append({"sequence": len(events), "action": action, **fields})


def _read_bound_json_with_events(
    path: Path,
    expected_sha256: str,
    label: str,
    events: MutableSequence[dict[str, Any]],
) -> dict[str, Any]:
    path = reject_symlink_components_before_normalization(
        path, label, require_exists=True
    )
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ProtocolError(f"{label} must be a regular non-symlink file")
    _event(events, "manifest_lstat_regular", manifest_class=label, bytes=info.st_size)
    payload = path.read_bytes()
    observed = sha256_bytes(payload)
    if observed != expected_sha256:
        raise ProtocolError(f"{label} SHA-256 mismatch")
    _event(
        events,
        "manifest_bytes_opened_read_and_hash_verified",
        manifest_class=label,
        bytes=len(payload),
        sha256=observed,
    )
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must contain one JSON object")
    return value


def load_source_rows(
    workspace_root: Path | None = None,
    access_log: MutableSequence[dict[str, Any]] | None = None,
) -> tuple[SourceRow, ...]:
    """Load only auxiliary manifests; never consult the external heldout manifest."""
    if access_log is None:
        raise ProtocolError("v5 source-manifest access requires an explicit event log")
    root_original = workspace_root or default_workspace_root()
    root_unresolved = reject_symlink_components_before_normalization(
        root_original, "workspace root", require_exists=True
    )
    root_info = lstat_or_none(root_unresolved)
    if root_info is None or not stat.S_ISDIR(root_info.st_mode):
        raise ProtocolError("workspace root must be an existing non-symlink directory")
    root = root_unresolved.resolve(strict=True)
    source = _read_bound_json_with_events(
        root / SOURCE_MANIFEST_BASENAME,
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "auxiliary_source_manifest",
        access_log,
    )
    _read_bound_json_with_events(
        root / SOURCE_FREEZE_BASENAME,
        EXPECTED_SOURCE_FREEZE_SHA256,
        "auxiliary_source_freeze",
        access_log,
    )
    synthetic_exclusion = {
        "revision": QWEN_REVISION,
        "blocks": [{"tensor": EXCLUDED_TENSOR}],
    }
    rows = TIER_A.build_source_rows(source, synthetic_exclusion)
    _event(
        access_log,
        "packaged_exclusion_identity_applied_without_external_manifest_path_resolution",
        excluded_tensor=EXCLUDED_TENSOR,
        external_manifest_expected_sha256=EXPECTED_EXCLUSION_MANIFEST_SHA256,
    )
    return rows


def exclusion_binding() -> dict[str, Any]:
    return {
        "mode": "packaged_identity_only",
        "excluded_tensor_identities": [EXCLUDED_TENSOR],
        "external_heldout_manifest_path_resolved": False,
        "external_heldout_manifest_existence_checked": False,
        "external_heldout_manifest_statted": False,
        "external_heldout_manifest_opened_or_read": False,
    }


def validate_aux_directory(
    aux_dir: Path,
    rows: Sequence[SourceRow],
    access_log: MutableSequence[dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Validate eligible files while never lstat/stat/opening the excluded payload."""
    requested_aux_dir = reject_symlink_components_before_normalization(
        aux_dir, "auxiliary path", require_exists=True
    )
    directory_info = requested_aux_dir.lstat()
    if not stat.S_ISDIR(directory_info.st_mode) or requested_aux_dir.is_symlink():
        raise ProtocolError("auxiliary path must be a regular non-symlink directory")
    aux_dir = requested_aux_dir.resolve(strict=True)
    forbidden = {"blind_protocol", "blind_protocol_v2", "qwen_polaris_heldout32", "heldout", "quarantine"}
    if any(part.lower() in forbidden for part in aux_dir.parts):
        raise ProtocolError("auxiliary path enters a forbidden heldout/quarantine component")
    _event(access_log, "auxiliary_directory_lstat_regular")
    observed_names = os.listdir(aux_dir)
    observed_bf16 = {name for name in observed_names if name.endswith(".bf16.bin")}
    eligible_rows = [row for row in rows if not row.excluded]
    eligible_names = {row.basename for row in eligible_rows}
    allowed_names = eligible_names | {EXCLUDED_BASENAME}
    if not eligible_names.issubset(observed_bf16) or not observed_bf16.issubset(allowed_names):
        raise ProtocolError(
            "auxiliary BF16 filename set mismatch; "
            f"missing={sorted(eligible_names-observed_bf16)}, "
            f"extra={sorted(observed_bf16-allowed_names)}"
        )
    excluded_observed = EXCLUDED_BASENAME in observed_bf16
    _event(
        access_log,
        "auxiliary_directory_names_enumerated",
        bf16_name_count=len(observed_bf16),
        eligible_name_count=len(eligible_names),
        excluded_basename_observed=excluded_observed,
        excluded_payload_lstat_or_stat_performed=False,
        excluded_payload_open_or_byte_read_performed=False,
    )
    paths: dict[str, Path] = {}
    eligible_identity: dict[str, dict[str, int]] = {}
    for row in eligible_rows:
        path = aux_dir / row.basename
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise ProtocolError(f"eligible source must be a regular non-symlink file: {row.basename}")
        if info.st_size != row.bytes:
            raise ProtocolError(f"eligible source byte count mismatch: {row.basename}")
        if path.resolve().parent != aux_dir:
            raise ProtocolError(f"eligible source escapes auxiliary directory: {row.basename}")
        _event(
            access_log,
            "eligible_payload_lstat_size_verified_without_byte_read",
            tensor_name=row.tensor_name,
            basename=row.basename,
            bytes=info.st_size,
        )
        paths[row.tensor_name] = path
        eligible_identity[row.tensor_name] = {
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "bytes": int(info.st_size),
            "mode": int(info.st_mode),
        }
    status = {
        "directory_names_enumerated": True,
        "eligible_payloads_lstat_or_statted": len(eligible_rows),
        "excluded_basename_observed_during_enumeration": excluded_observed,
        "excluded_payload_lstat_or_statted": False,
        "excluded_payload_opened_or_bytes_read": False,
        "all_directory_names_sha256": sha256_bytes(
            canonical_json_bytes(sorted(observed_names))
        ),
        "auxiliary_directory_device": int(directory_info.st_dev),
        "auxiliary_directory_inode": int(directory_info.st_ino),
        "eligible_identity": eligible_identity,
    }
    return paths, status


def revalidate_workspace_aux_closure(
    workspace_root: Path,
    aux_dir: Path,
    rows: Sequence[SourceRow],
    paths: Mapping[str, Path],
    frozen_status: Mapping[str, Any],
    access_log: MutableSequence[dict[str, Any]],
) -> dict[str, Any]:
    """Recheck all named workspace metadata and relevant auxiliary closure."""
    workspace = reject_symlink_components_before_normalization(
        require_canonical_absolute_spelling(workspace_root, "workspace closure root"),
        "workspace closure root",
        require_exists=True,
    )
    _read_bound_json_with_events(
        workspace / SOURCE_MANIFEST_BASENAME,
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "preoutput_auxiliary_source_manifest_revalidation",
        access_log,
    )
    _read_bound_json_with_events(
        workspace / SOURCE_FREEZE_BASENAME,
        EXPECTED_SOURCE_FREEZE_SHA256,
        "preoutput_auxiliary_source_freeze_revalidation",
        access_log,
    )
    auxiliary = reject_symlink_components_before_normalization(
        require_canonical_absolute_spelling(aux_dir, "auxiliary closure root"),
        "auxiliary closure root",
        require_exists=True,
    )
    aux_info = auxiliary.lstat()
    if (
        not stat.S_ISDIR(aux_info.st_mode)
        or stat.S_ISLNK(aux_info.st_mode)
        or int(aux_info.st_dev) != frozen_status["auxiliary_directory_device"]
        or int(aux_info.st_ino) != frozen_status["auxiliary_directory_inode"]
    ):
        raise ProtocolError("auxiliary directory identity changed before output creation")
    names = os.listdir(auxiliary)
    if sha256_bytes(canonical_json_bytes(sorted(names))) != frozen_status[
        "all_directory_names_sha256"
    ]:
        raise ProtocolError("auxiliary directory closure changed before output creation")
    eligible_rows = [row for row in rows if not row.excluded]
    expected_identities = frozen_status["eligible_identity"]
    if set(paths) != {row.tensor_name for row in eligible_rows} or set(expected_identities) != set(paths):
        raise ProtocolError("eligible auxiliary identity set changed before output creation")
    for row in eligible_rows:
        path = paths[row.tensor_name]
        info = path.lstat()
        observed = {
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "bytes": int(info.st_size),
            "mode": int(info.st_mode),
        }
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or observed != expected_identities[row.tensor_name]
        ):
            raise ProtocolError(
                f"eligible auxiliary identity changed before output: {row.basename}"
            )
    receipt = {
        "schema": "qwen3_tier_c_grouped_v5_workspace_aux_closure_v1",
        "workspace_manifest_and_freeze_rehashed": True,
        "auxiliary_directory_names_sha256": frozen_status["all_directory_names_sha256"],
        "eligible_identity_count": len(eligible_rows),
        "excluded_payload_lstat_or_stat_performed": False,
        "excluded_payload_opened_or_bytes_read": False,
        "closure_complete_before_output_or_journal_creation": True,
    }
    _event(
        access_log,
        "workspace_and_auxiliary_closure_revalidated_before_output_or_journal_creation",
        receipt_sha256=sha256_bytes(canonical_json_bytes(receipt)),
        eligible_identity_count=len(eligible_rows),
        excluded_payload_lstat_or_stat_performed=False,
        excluded_payload_open_or_byte_read_performed=False,
    )
    return receipt


def canonical_to_native_flat(role: str, canonical: np.ndarray) -> np.ndarray:
    return TIER_B.canonical_to_native_flat(role, canonical)


def decode_bfloat16_words(words: np.ndarray) -> np.ndarray:
    return TIER_B.decode_bfloat16_words(words)


def logical_ordinal(
    stored_seed_u16: int,
    pp_index: int,
    ep_index: int,
    etp_index: int,
    assignment_index: int,
    half_index: int,
    abi_index: int,
) -> int:
    if not 0 <= stored_seed_u16 < STORED_SEED_COUNT:
        raise ProtocolError("stored seed outside frozen u16 range")
    ordinal = stored_seed_u16
    for index, size in (
        (pp_index, 10),
        (ep_index, 8),
        (etp_index, 4),
        (assignment_index, 2),
        (half_index, 2),
        (abi_index, 2),
    ):
        if not 0 <= index < size:
            raise ProtocolError("candidate axis index out of range")
        ordinal = ordinal * size + index
    return ordinal


@dataclass(frozen=True)
class CandidateKey:
    ordinal: int
    stored_seed_u16: int
    pp_index: int
    ep_index: int
    etp_index: int
    assignment_index: int
    half_index: int
    abi_index: int

    @property
    def cli_base_seed(self) -> int:
        return self.stored_seed_u16 + 1

    @property
    def base_seed(self) -> int:
        """Compatibility alias: always the legal positive CLI seed."""
        return self.cli_base_seed

    @property
    def pipeline_parallel_size(self) -> int:
        return PP_SIZES[self.pp_index]

    @property
    def expert_parallel_size(self) -> int:
        return EP_SIZES[self.ep_index]

    @property
    def expert_tensor_parallel_size(self) -> int:
        return ETP_SIZES[self.etp_index]

    @property
    def expert_assignment(self) -> str:
        return ASSIGNMENTS[self.assignment_index]

    @property
    def canonical_fc1_half_assignment(self) -> str:
        return HALF_ASSIGNMENTS[self.half_index]

    @property
    def storage_abi(self) -> str:
        return STORAGE_ABIS[self.abi_index]

    @property
    def target_pipeline_rank_and_local_layer(self) -> tuple[int, int]:
        layers_per_stage = 48 // self.pipeline_parallel_size
        return divmod(15, layers_per_stage)

    @property
    def pipeline_stream_class(self) -> str:
        return ("A", "A", "A", "B", "C", "D", "E", "F", "G", "H")[self.pp_index]

    @property
    def id(self) -> str:
        return (
            f"{self.ordinal:08d}|stored_seed={self.stored_seed_u16}|cli_seed={self.cli_base_seed}"
            f"|pp={self.pipeline_parallel_size}|ep={self.expert_parallel_size}"
            f"|etp={self.expert_tensor_parallel_size}|assign={self.expert_assignment}"
            f"|half={self.canonical_fc1_half_assignment}|abi={self.storage_abi}"
        )

    def to_json(self) -> dict[str, Any]:
        pp_rank, local_layer = self.target_pipeline_rank_and_local_layer
        return {
            "ordinal": self.ordinal,
            "id": self.id,
            "stored_seed_u16": self.stored_seed_u16,
            "cli_base_seed": self.cli_base_seed,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
            "expert_assignment": self.expert_assignment,
            "canonical_fc1_half_assignment": self.canonical_fc1_half_assignment,
            "storage_abi": self.storage_abi,
            "target_pipeline_rank": pp_rank,
            "target_local_layer": local_layer,
            "pipeline_seed_delta": 100 * pp_rank,
            "pipeline_stream_class": self.pipeline_stream_class,
            "canonical_representative_ordinal": representative_ordinal(self.ordinal),
            "is_canonical_representative": representative_ordinal(self.ordinal) == self.ordinal,
        }


def decode_ordinal(ordinal: int) -> CandidateKey:
    if not 0 <= int(ordinal) < LOGICAL_CANDIDATES:
        raise ProtocolError("logical candidate ordinal out of range")
    value = int(ordinal)
    abi = value % 2
    value //= 2
    half = value % 2
    value //= 2
    assignment = value % 2
    value //= 2
    etp = value % 4
    value //= 4
    ep = value % 8
    value //= 8
    pp = value % 10
    stored_seed = value // 10
    candidate = CandidateKey(
        int(ordinal), stored_seed, pp, ep, etp, assignment, half, abi
    )
    if logical_ordinal(stored_seed, pp, ep, etp, assignment, half, abi) != int(ordinal):
        raise ProtocolError("candidate ordinal roundtrip failure")
    return candidate


def representative_ordinal(ordinal: int) -> int:
    candidate = decode_ordinal(ordinal)
    # PP1/PP2/PP3 are the one same-seed class-A stream.  Cross-seed
    # coincidences are intentionally not collapsed.
    pp = 0 if candidate.pp_index in (1, 2) else candidate.pp_index
    assignment = 0 if candidate.ep_index in (0, 7) else candidate.assignment_index
    return logical_ordinal(
        candidate.stored_seed_u16,
        pp,
        candidate.ep_index,
        candidate.etp_index,
        assignment,
        candidate.half_index,
        0,
    )


def _enumerate_representatives(
    seed_start: int, seed_stop: int, *, new_only: bool
) -> np.ndarray:
    if not (0 <= seed_start <= seed_stop <= STORED_SEED_COUNT):
        raise ProtocolError("invalid stored-seed shard")
    per_seed = NEW_REPRESENTATIVES_PER_SEED if new_only else FULL_REPRESENTATIVES_PER_SEED
    count = (seed_stop - seed_start) * per_seed
    values = np.empty(count, dtype=np.uint64)
    cursor = 0
    for stored_seed in range(seed_start, seed_stop):
        # Expanded representatives are PP1,4,6,8,12,16,24,48.  New-only is
        # ETP8 on old PP streams (PP1/4/8) plus every ETP on the five new PP
        # streams.  Loop order follows the expanded logical ordinal axes.
        for pp_index in (0, 3, 4, 5, 6, 7, 8, 9):
            for ep_index in range(8):
                assignment_indices = (0,) if ep_index in (0, 7) else (0, 1)
                if new_only and pp_index in (0, 3, 5):
                    etp_indices = (3,)
                else:
                    etp_indices = range(4)
                for etp_index in etp_indices:
                    for assignment_index in assignment_indices:
                        for half_index in range(2):
                            values[cursor] = logical_ordinal(
                                stored_seed,
                                pp_index,
                                ep_index,
                                etp_index,
                                assignment_index,
                                half_index,
                                0,
                            )
                            cursor += 1
    if cursor != count or len(np.unique(values)) != count:
        raise ProtocolError("layout-overlay representative enumeration accounting mismatch")
    if len(values) > 1 and not bool(np.all(values[1:] > values[:-1])):
        raise ProtocolError("representatives are not in strict expanded-ordinal order")
    return values


def representative_ordinals(seed_start: int, seed_stop: int) -> np.ndarray:
    """Return only the genuinely new representatives searched by stage 0."""
    return _enumerate_representatives(seed_start, seed_stop, new_only=True)


def full_representative_ordinals(seed_start: int, seed_stop: int) -> np.ndarray:
    """Return the full expanded-union representatives for source-only audits."""
    return _enumerate_representatives(seed_start, seed_stop, new_only=False)


def v4_logical_ordinal(
    stored_seed_u16: int,
    pp_index: int,
    ep_index: int,
    etp_index: int,
    assignment_index: int,
    half_index: int,
    abi_index: int,
) -> int:
    """Independent frozen-v4 ordinal encoder used only for translation tests."""
    if not 0 <= stored_seed_u16 < STORED_SEED_COUNT:
        raise ProtocolError("v4 stored seed outside frozen u16 range")
    ordinal = stored_seed_u16
    for index, size in (
        (pp_index, 4), (ep_index, 8), (etp_index, 3),
        (assignment_index, 2), (half_index, 2), (abi_index, 2),
    ):
        if not 0 <= index < size:
            raise ProtocolError("v4 candidate axis index out of range")
        ordinal = ordinal * size + index
    return ordinal


def translate_v4_ordinal(v4_ordinal: int) -> int:
    """Translate a canonical v4 representative into the expanded ordinal.

    The map is strictly increasing on the v4 representative set, so v4's
    metric/ordinal TopK tie order is unchanged after translation.
    """
    if not 0 <= int(v4_ordinal) < 50_331_648:
        raise ProtocolError("v4 ordinal outside frozen logical range")
    value = int(v4_ordinal)
    abi = value % 2; value //= 2
    half = value % 2; value //= 2
    assignment = value % 2; value //= 2
    etp = value % 3; value //= 3
    ep = value % 8; value //= 8
    old_pp = value % 4
    seed = value // 4
    old_rep = v4_logical_ordinal(
        seed, 0 if old_pp == 1 else old_pp, ep, etp,
        0 if ep in (0, 7) else assignment, half, 0,
    )
    if old_rep != int(v4_ordinal):
        raise ProtocolError("v4 shortlist contains a noncanonical ordinal")
    expanded_pp = {0: 0, 2: 3, 3: 5}.get(old_pp)
    if expanded_pp is None:
        raise ProtocolError("v4 canonical ordinal uses a collapsed PP descriptor")
    return logical_ordinal(seed, expanded_pp, ep, etp, assignment, half, 0)


def stream_signature(candidate: CandidateKey) -> tuple[int, int, int, int, int, int, int]:
    layers_per_stage = 48 // candidate.pipeline_parallel_size
    pp_rank, local_layer = divmod(15, layers_per_stage)
    effective_seed = candidate.cli_base_seed + 100 * pp_rank
    assignment = 0 if candidate.ep_index in (0, 7) else candidate.assignment_index
    return (
        effective_seed,
        local_layer,
        candidate.ep_index,
        candidate.etp_index,
        assignment,
        candidate.half_index,
        0,
    )


def equivalence_map_object() -> dict[str, Any]:
    return {
        "schema": "tier_c_grouped_v5_layout_overlay_equivalence_map_v1",
        "seed_encoding": {
            "stored": "u16 0..65535",
            "cli_seed": "stored_seed_u16 + 1",
            "legal_cli_range": [1, 65536],
        },
        "logical_axis_order": [
            "stored_seed_u16",
            "pipeline_parallel_size",
            "expert_parallel_size",
            "expert_tensor_parallel_size",
            "expert_assignment",
            "canonical_fc1_half_assignment",
            "storage_abi",
        ],
        "logical_layouts_per_stored_seed": LAYOUTS_PER_STORED_SEED,
        "logical_candidate_count": LOGICAL_CANDIDATES,
        "global_representative_rule": "smallest logical ordinal in every exact class",
        "mapping_rule_in_order": [
            "PP indices 1 and 2 (PP2 and PP3) map to PP index 0 at the same stored seed",
            "both storage ABI indices map to ABI index 0",
            "assignment index maps to 0 at EP indices 0 and 7",
            "all other axes are identity",
        ],
        "pp_stream_relations": {
            "pp1_equals_pp2_equals_pp3_same_seed": True,
            "other_pp_streams_remain_distinct_at_every_stored_seed": True,
            "cross_seed_pp_equivalence_used": False,
        },
        "assignment_relations": {
            "ep1_contiguous_equals_round_robin": True,
            "ep128_contiguous_equals_round_robin": True,
            "all_other_ep_assignments_distinct": True,
        },
        "abi_copy_pack_relation_pending_runtime_parity": True,
        "proof_dependencies": {
            "storage_abi_collapse_requires_numbered_copy_pack_value_and_terminal_rng_parity": True,
            "pp_relations_require_exact_seven_file_source_trace": True,
            "philox_mapping_requires_pytorch_cupy_runtime_parity": True,
            "source_only_package_does_not_claim_these_runtime_dependencies_have_passed": True,
        },
        "full_representatives_per_stored_seed": FULL_REPRESENTATIVES_PER_SEED,
        "full_union_distinct_anchor_count_conditional_on_proof_dependencies": FULL_EFFECTIVE_CANDIDATES,
        "full_union_distinct_anchor_count": FULL_EFFECTIVE_CANDIDATES,
        "retained_v4_representatives_per_stored_seed": V4_REPRESENTATIVES_PER_SEED,
        "retained_v4_distinct_anchor_count": V4_EFFECTIVE_CANDIDATES,
        "new_representatives_per_stored_seed": NEW_REPRESENTATIVES_PER_SEED,
        "new_distinct_anchor_count": NEW_EFFECTIVE_CANDIDATES,
        "new_increment_breakdown": {
            "five_new_pp_streams_crossed_with_four_etp_per_seed": 560,
            "etp8_on_three_old_pp_streams_per_seed": 84,
        },
        "old_v4_to_expanded_ordinal_translation_is_strictly_increasing": True,
        "same_map_for_source_and_all_32_matched_null_controls": True,
    }


def equivalence_map_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(equivalence_map_object()))


def equivalence_audit() -> dict[str, Any]:
    """Exhaust the seed/PP relations; other axes are finite separable products."""
    pp_rows = 0
    for stored_seed in range(STORED_SEED_COUNT):
        for pp_index in range(10):
            ordinal = logical_ordinal(stored_seed, pp_index, 0, 0, 0, 0, 0)
            representative = decode_ordinal(representative_ordinal(ordinal))
            if stream_signature(decode_ordinal(ordinal)) != stream_signature(representative):
                raise ProtocolError("same-seed PP representative changes the stream signature")
            if representative.ordinal > ordinal:
                raise ProtocolError("representative is not the smallest observed logical ordinal")
            pp_rows += 1
    expected = STORED_SEED_COUNT * FULL_REPRESENTATIVES_PER_SEED
    if expected != FULL_EFFECTIVE_CANDIDATES:
        raise ProtocolError("closed-form global representative count mismatch")
    return {
        "seed_pp_rows_exhausted": pp_rows,
        "separable_ep_assignment_classes": ASSIGNMENT_CLASSES_PER_PP_ETP_HALF,
        "separable_etp_count": len(ETP_SIZES),
        "separable_half_count": len(HALF_ASSIGNMENTS),
        "separable_abi_classes": 1,
        "full_union_distinct_anchor_count": expected,
        "new_distinct_anchor_count": NEW_EFFECTIVE_CANDIDATES,
        "retained_v4_distinct_anchor_count": V4_EFFECTIVE_CANDIDATES,
    }


make_plan = TIER_B.make_plan
plan_sha256 = TIER_B.plan_sha256
plan_json = TIER_B.plan_json
stateless_normals = TIER_B.stateless_normals
permutation_and_sign = TIER_B.permutation_and_sign
fold_statistics = TIER_B.fold_statistics
metric_from_sse = TIER_B.metric_from_sse


def quantize_affine_f16le(alpha: float, mu: float) -> tuple[bytes, float, float]:
    values = np.asarray([float(alpha), float(mu)], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ProtocolError("non-finite affine before FP16 serialization")
    with np.errstate(over="ignore", invalid="ignore"):
        encoded_values = values.astype("<f2")
    if not np.all(np.isfinite(encoded_values.astype(np.float64))):
        raise ProtocolError("affine is not representable as finite IEEE binary16")
    payload = encoded_values.tobytes(order="C")
    if len(payload) != 4:
        raise ProtocolError("affine codec did not produce exactly four bytes")
    decoded = np.frombuffer(payload, dtype="<f2").astype(np.float64)
    return payload, float(decoded[0]), float(decoded[1])


def fit_affine_moments(w: Sequence[float], g: Sequence[float]) -> dict[str, Any]:
    raw = dict(TIER_B.fit_affine_moments(w, g))
    payload, alpha, mu = quantize_affine_f16le(float(raw["alpha"]), float(raw["mu"]))
    raw["unquantized_alpha"] = float(raw["alpha"])
    raw["unquantized_mu"] = float(raw["mu"])
    raw["alpha"] = alpha
    raw["mu"] = mu
    raw["alpha_mu_f16le_hex"] = payload.hex()
    raw["affine_storage_bytes"] = 4
    raw["score_uses_decoded_fp16"] = True
    return raw


def score_affine_moments(
    w: Sequence[float], g: Sequence[float], alpha: float, mu: float, fit_mean_w: float
) -> dict[str, Any]:
    payload, decoded_alpha, decoded_mu = quantize_affine_f16le(alpha, mu)
    if decoded_alpha != float(alpha) or decoded_mu != float(mu):
        raise ProtocolError("score received an affine that was not decoded from frozen FP16")
    result = dict(
        TIER_B.score_affine_moments(w, g, decoded_alpha, decoded_mu, fit_mean_w)
    )
    result["alpha_mu_f16le_hex"] = payload.hex()
    result["decoded_alpha"] = decoded_alpha
    result["decoded_mu"] = decoded_mu
    result["score_uses_decoded_fp16"] = True
    return result


def physical_ledger() -> dict[str, Any]:
    side_bpw = METADATA_BYTES * 8.0 / TARGET_WEIGHTS
    standalone = 1.0 - (TARGET_F / CURRENT_F) * 2.0 ** (-2.0 * side_bpw)
    composite = 1.0 - (TARGET_F / COMPOSITE_F) * 2.0 ** (-2.0 * side_bpw)
    payload_bytes = WEIGHTS_PER_EXPERT * READ_LEDGER_REFERENCE_BPW / 8.0
    read_amp = CURRENT_WORST_READ_AMP + 20.0 / payload_bytes
    strict_payload_bytes = WEIGHTS_PER_EXPERT * STRICT_BPW_CAP / 8.0
    strict_read_amp = CURRENT_WORST_READ_AMP + 20.0 / strict_payload_bytes
    return {
        "target_matrix_count": TARGET_MATRIX_COUNT,
        "target_weights": TARGET_WEIGHTS,
        "metadata_bytes_total": METADATA_BYTES,
        "side_bpw": side_bpw,
        "metadata_adjusted_composite_required_capture": composite,
        "metadata_adjusted_standalone_required_capture": standalone,
        "global_lineage_descriptor_bytes": 8,
        "per_matrix_affine_bytes": 4,
        "per_matrix_affine_codec": "alpha IEEE-754 binary16 little-endian then mu binary16 little-endian",
        "scientific_scores_use_decoded_fp16_affines": True,
        "learned_generator_table_bytes": 0,
        "external_generator_read_bytes": 0,
        "metadata_read_bytes_per_expert": 20,
        "metadata_read_amplification_denominator_bpw": READ_LEDGER_REFERENCE_BPW,
        "metadata_read_amplification_denominator_bytes_per_expert": payload_bytes,
        "current_worst_cold_read_amplification": CURRENT_WORST_READ_AMP,
        "upstream_read_baseline_requires_later_composition_receipt": True,
        "conservative_appended_cold_read_amplification": read_amp,
        "bytes_per_expert_at_strict_2_15_bpw": strict_payload_bytes,
        "conservative_appended_cold_read_amplification_at_2_15_bpw": strict_read_amp,
        "strict_read_amplification_max_exclusive": 2.0,
        "strict_bpw_cap": STRICT_BPW_CAP,
        "maximum_compatible_base_codec_bpw_after_side_metadata": STRICT_BPW_CAP - side_bpw,
        "passes_read_gate_arithmetic_only": strict_read_amp < 2.0,
    }


def make_decision(
    source_folds: Mapping[str, Any], null_captures: Mapping[str, float]
) -> dict[str, Any]:
    if set(null_captures) != set(NULL_DOMAIN_IDS):
        raise ProtocolError("decision requires exactly the 32 frozen matched-null captures")
    ledger = physical_ledger()
    raw = float(source_folds["pooled"]["capture"])
    max_null_id, max_null = max(
        null_captures.items(), key=lambda item: (float(item[1]), item[0])
    )
    correction = max(0.0, float(max_null))
    corrected = raw - correction
    se = float(source_folds["whole_expert_capture_standard_error"])
    lower = corrected - 3.0 * se
    upper = corrected + 3.0 * se
    all_experts = all(float(row["capture"]) > 0.0 for row in source_folds["whole_experts"])
    all_roles = all(float(row["capture"]) > 0.0 for row in source_folds["roles"])
    beats_all_nulls = all(raw > float(value) for value in null_captures.values())
    composite = float(ledger["metadata_adjusted_composite_required_capture"])
    standalone = float(ledger["metadata_adjusted_standalone_required_capture"])
    gates = (
        beats_all_nulls
        and all_experts
        and all_roles
        and bool(ledger["passes_read_gate_arithmetic_only"])
    )
    if upper < composite:
        state = "HARD_KILL_BOUNDED_TIER_C_GROUPED_V5_LAYOUT_OVERLAY_SET"
    elif lower >= standalone and gates:
        state = "STANDALONE_ENERGY_SCREEN_SURVIVOR_REQUIRES_SEPARATELY_FROZEN_CODEC_COMPOSITION"
    elif lower >= composite and gates:
        state = "COMPOSITE_SCREEN_ONLY_REQUIRES_FINITE_AUDITED_COMPOSITION_NOT_FINAL_TARGET_CLAIM"
    else:
        state = "INCONCLUSIVE_NO_COMPOSITE_OR_FINAL_TARGET_CLAIM"
    control_rank = 1 + sum(float(value) >= raw for value in null_captures.values())
    return {
        "state": state,
        "raw_source_validation_capture": raw,
        "maximum_matched_null_validation_capture": float(max_null),
        "maximum_matched_null_domain": max_null_id,
        "descriptive_max_null_correction": correction,
        "bias_corrected_capture": corrected,
        "whole_expert_capture_standard_error": se,
        "bias_corrected_lower_3se": lower,
        "bias_corrected_upper_3se": upper,
        "metadata_adjusted_composite_required_capture": composite,
        "metadata_adjusted_standalone_required_capture": standalone,
        "capture_unit": "fraction_of_heldout_serialized_mean_sse_removed",
        "composite_screen_is_final_20_percent_below_gaussian_claim": False,
        "standalone_screen_is_final_20_percent_below_gaussian_claim": False,
        "final_rate_distortion_claim_requires_separately_finite_audited_composition": True,
        "source_beats_all_32_matched_null_controls": beats_all_nulls,
        "all_four_whole_expert_folds_positive": all_experts,
        "both_role_folds_positive": all_roles,
        "read_gate_passes_arithmetic_only": bool(ledger["passes_read_gate_arithmetic_only"]),
        "empirical_control_rank_numerator": control_rank,
        "empirical_control_rank_denominator": 33,
        "control_rank_interpretation": "descriptive_rank_only_controls_not_exchangeable",
        "randomization_p_value_claimed": False,
        "matched_null_controls_declared_exchangeable_with_source": False,
    }


def environment_has_forbidden_runtime_imports() -> bool:
    prefixes = ("torch", "cupy", "transformer_engine", "transformer_engine_torch", "megatron")
    return any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in prefixes
    )


def environment_has_cuda_imports() -> bool:
    return environment_has_forbidden_runtime_imports()
