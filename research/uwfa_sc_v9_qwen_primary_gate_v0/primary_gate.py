#!/usr/bin/env python3
"""Nonpromoting real-Qwen primary-only runtime envelope for sealed UWFA-SC v8.

UWFA-SC v8 itself is preserved byte-for-byte.  This sibling authenticates the
sealed v8 package and the repaired exploratory Qwen bridge, runs and validates
the complete source-free CuPy review, and only then opens the pinned Qwen
artifact.  It executes exactly the disjoint-owner-component nested holdout and
the final literal physical container.  Shuffles, coordinate diagnostics and
controls are intentionally unreachable here and require separate authority.

Importing this file performs no path access, numerical-library import, CUDA
initialization, subprocess launch, payload discovery or output mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "uwfa-sc-v9-qwen-primary-gate-v0"
AUTHORIZATION = "RUN_EXACT_QWEN_PRIMARY_ONLY_NONPROMOTING_V0"
SEALED_V8_MANIFEST_SHA256 = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"
PINNED_SUPPORT_SHA256 = "399cb25260d34ec299cc91a17f129da9be5ba5b799c961e43f0c1b0637ee0174"
CURRENT_ARTIFACT_SHA256 = "4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b"
CURRENT_ARTIFACT_BYTES = 8_847_360
SOURCE_WEIGHTS = 28_311_552
MAX_MEMBER_BYTES = 2 * (1 << 20)
MAX_RESULT_JSON_BYTES = 256 * (1 << 20)

# These are evaluation-runner admission pins for the already authenticated
# Qwen panel.  They are not decoder inputs and are never serialized into the
# UWFCV8 decoder state.
PINNED_PRIMARY_CELL_SYMBOL_UPDATES = 38_621_316_130
PINNED_DEFERRED_MAXIMUM_UPDATES = 286_625_070_746
PINNED_DEFERRED_COORDINATE_UPDATES = 93_518_490_096
PINNED_PANEL_SYMBOLS = 126_627_266
PINNED_PANEL_STREAMS = 15
PINNED_FOLD_UPDATES = (
    (0, (0, 1), 12_865_688_966),
    (1, (2, 3), 12_794_875_916),
    (2, (4, 5), 12_707_496_716),
)

PRIMARY_KERNEL_BUDGET_SECONDS = 21_600.0
CONSERVATIVE_THROUGHPUT_MIN = 1_800_000.0
CONSERVATIVE_THROUGHPUT_MAX = 4_500_000.0
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090"

V9_REQUIRED_MEMBERS = (
    "README.md",
    "design_lock.json",
    "primary_gate.py",
    "test_source_only.py",
)


class PrimaryGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrimaryGateError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            require(key not in output, f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def reject(value: str) -> None:
        raise PrimaryGateError(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    except PrimaryGateError:
        raise
    except Exception as exc:
        raise PrimaryGateError(f"invalid JSON: {exc}") from exc
    require(isinstance(value, dict), "JSON root must be object")
    return value


def require_digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64, f"{label} digest geometry")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise PrimaryGateError(f"{label} digest encoding") from exc
    require(len(raw) == 32, f"{label} digest width")
    return value.lower()


def require_absolute_lexical(path: Path, label: str) -> Path:
    text = os.fspath(path)
    require(os.path.isabs(text), f"{label} path must be absolute")
    _drive, tail = os.path.splitdrive(text)
    parts = tuple(part for part in tail.replace("\\", "/").split("/") if part)
    require(parts and all(part not in {".", ".."} for part in parts), f"{label} path noncanonical")
    return Path(text)


def reject_symlink_chain(path: Path, label: str) -> None:
    path = require_absolute_lexical(path, label)
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor = cursor / component
        require(os.path.lexists(cursor), f"{label} component absent: {cursor}")
        info = os.lstat(cursor)
        require(not stat.S_ISLNK(info.st_mode), f"{label} symlink component: {cursor}")


class HeldRegularInput:
    """Bounded, no-follow source input retained while its bytes are consumed."""

    def __init__(
        self,
        path: Path,
        *,
        label: str,
        maximum_bytes: int,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> None:
        self.path = require_absolute_lexical(path, label)
        reject_symlink_chain(self.path, label)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(os.fspath(self.path), flags)
        try:
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode), f"{label} is not regular")
            require(0 <= info.st_size <= maximum_bytes, f"{label} byte bound")
            if expected_bytes is not None:
                require(info.st_size == expected_bytes, f"{label} exact bytes")
            self.identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            chunks: list[bytes] = []
            while chunk := os.read(self.fd, 1 << 20):
                chunks.append(chunk)
            self.data = b"".join(chunks)
            require(len(self.data) == info.st_size, f"{label} short read")
            self.sha256 = sha256(self.data)
            if expected_sha256 is not None:
                require(self.sha256 == require_digest(expected_sha256, label), f"{label} digest")
            self.verify_stable()
        except Exception:
            os.close(self.fd)
            self.fd = -1
            raise

    def verify_stable(self) -> None:
        require(self.fd >= 0, "closed held input")
        info = os.fstat(self.fd)
        require(
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == self.identity,
            f"held input changed: {self.path}",
        )

    def close(self) -> None:
        if self.fd >= 0:
            self.verify_stable()
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "HeldRegularInput":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        self.close()


def authenticate_v9_package(package: Path) -> dict[str, Any]:
    package = require_absolute_lexical(package, "v9 package")
    reject_symlink_chain(package, "v9 package")
    require(stat.S_ISDIR(os.lstat(package).st_mode), "v9 package must be directory")
    with HeldRegularInput(
        package / "SOURCE_MANIFEST.json",
        label="v9 source manifest",
        maximum_bytes=1 << 20,
    ) as held_manifest:
        manifest_bytes = held_manifest.data
    manifest = strict_json(manifest_bytes)
    require(
        set(manifest) == {"schema", "status", "members", "access_attestation", "claim_boundary"},
        "v9 manifest fields",
    )
    require(manifest["schema"] == "uwfa-sc-v9-primary-source-manifest-v0", "v9 manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "v9 manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(V9_REQUIRED_MEMBERS), "v9 member rows")
    names = [row.get("name") if isinstance(row, dict) else None for row in rows]
    require(names == list(V9_REQUIRED_MEMBERS), "v9 member order")
    actual = {entry.name for entry in os.scandir(package)}
    require(actual == set(V9_REQUIRED_MEMBERS) | {"SOURCE_MANIFEST.json"}, "v9 undeclared/missing members")
    snapshots: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "v9 manifest row")
        name = str(row["name"])
        require(type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_MEMBER_BYTES, f"v9 bytes: {name}")
        with HeldRegularInput(
            package / name,
            label=f"v9 member {name}",
            maximum_bytes=MAX_MEMBER_BYTES,
            expected_bytes=int(row["bytes"]),
            expected_sha256=require_digest(row["sha256"], f"v9 member {name}"),
        ) as held:
            snapshots[name] = held.data
            hashes[name] = held.sha256
    return {
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_bytes),
        "member_hashes": hashes,
        "snapshots": snapshots,
        "source_snapshot_root_sha256": sha256(canonical_json(rows)),
    }


def load_authenticated_support(path: Path) -> tuple[types.ModuleType, dict[str, Any]]:
    with HeldRegularInput(
        path,
        label="pinned repaired v8 Qwen support",
        maximum_bytes=MAX_MEMBER_BYTES,
        expected_sha256=PINNED_SUPPORT_SHA256,
    ) as held:
        source = held.data
    name = "uwfa_sc_v9_pinned_qwen_support"
    require(name not in sys.modules, "pinned support already loaded")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated-v9-support:{PINNED_SUPPORT_SHA256}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(source, module.__file__, "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    require(module.SEALED_V8_MANIFEST_SHA256 == SEALED_V8_MANIFEST_SHA256, "support v8 pin")
    require(module.CURRENT_ARTIFACT_SHA256 == CURRENT_ARTIFACT_SHA256, "support artifact pin")
    require(module.CURRENT_ARTIFACT_BYTES == CURRENT_ARTIFACT_BYTES, "support artifact bytes")
    require(module.SOURCE_WEIGHTS == SOURCE_WEIGHTS, "support source weights")
    return module, {
        "sha256": sha256(source),
        "semantic_bridge_abi": "list[list[int]]",
        "tuple_rows_forbidden": True,
        "positive_claim_authority": False,
    }


@dataclass(frozen=True)
class SourceFreeReview:
    status: str
    v9_source_snapshot_root_sha256: str
    v8_source_snapshot_root_sha256: str
    preflight_receipt_sha256: str
    support_sha256: str
    measured_updates_per_second: float
    conservative_updates_per_second: float
    device_name: str
    device_uuid: str
    pci_bus_id: str
    receipt_sha256: str

    def clean_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "v9_source_snapshot_root_sha256": self.v9_source_snapshot_root_sha256,
            "v8_source_snapshot_root_sha256": self.v8_source_snapshot_root_sha256,
            "preflight_receipt_sha256": self.preflight_receipt_sha256,
            "support_sha256": self.support_sha256,
            "measured_updates_per_second": self.measured_updates_per_second,
            "conservative_updates_per_second": self.conservative_updates_per_second,
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "pci_bus_id": self.pci_bus_id,
        }


def verify_source_free_review(review: SourceFreeReview) -> dict[str, Any]:
    require(type(review) is SourceFreeReview, "exact SourceFreeReview capability required")
    clean = review.clean_record()
    require(clean["status"] == "PASS_AUTHENTICATED_SOURCE_FREE_REVIEW", "source-free review status")
    for name in (
        "v9_source_snapshot_root_sha256",
        "v8_source_snapshot_root_sha256",
        "preflight_receipt_sha256",
        "support_sha256",
        "device_uuid",
    ):
        require_digest(clean[name], name) if name != "device_uuid" else None
    require(clean["support_sha256"] == PINNED_SUPPORT_SHA256, "review support pin")
    measured = float(clean["measured_updates_per_second"])
    conservative = float(clean["conservative_updates_per_second"])
    require(math.isfinite(measured) and math.isfinite(conservative), "finite review throughput")
    require(
        abs(conservative - 0.5 * measured) <= 8.0 * math.ulp(0.5 * measured),
        "review conservative throughput derivation",
    )
    require(
        CONSERVATIVE_THROUGHPUT_MIN <= conservative <= CONSERVATIVE_THROUGHPUT_MAX,
        "review throughput tamper bounds",
    )
    require(clean["device_name"] == EXPECTED_DEVICE_NAME, "review device name")
    require(sha256(canonical_json(clean)) == review.receipt_sha256, "review capability seal")
    return clean


def _provisional_preflight_bindings(stage: Any, source_root: str, receipt: str) -> Any:
    filler = sha256(b"UWFA-SC-V9-PREFLIGHT-ONLY-NONDECODER-BINDING")
    return stage.BoundEvidence(
        baseline_plan_sha256=filler,
        baseline_score_sha256=filler,
        universal_decoder_sha256=filler,
        producer_manifest_sha256=SEALED_V8_MANIFEST_SHA256,
        audit_bootstrap_sha256=filler,
        source_full_geometry_sha256=filler,
        source_structural_geometry_sha256=filler,
        extraction_program_sha256=filler,
        universal_adapter_sha256=filler,
        pipeline_sha256=filler,
        source_snapshot_root_sha256=source_root,
        source_preflight_receipt_sha256=receipt,
    )


def run_source_free_review(
    support: Any,
    modules: Mapping[str, Any],
    backend: Any,
    *,
    v9_source_root: str,
    v8_source_root: str,
) -> tuple[SourceFreeReview, Any, dict[str, Any]]:
    """Run and fully validate source-free evidence before any Qwen leaf access."""

    typed, record = support.run_source_free_preflight(modules, backend, v8_source_root)
    provisional = _provisional_preflight_bindings(
        modules["stage"], v8_source_root, str(record["receipt_sha256"])
    )
    validated = modules["stage"].validate_source_preflight(
        modules["common"], modules["protocol"], typed, provisional
    )
    require(validated["receipt_sha256"] == record["receipt_sha256"], "validated preflight receipt")
    runtime = validated["representative"]["runtime_projection"]
    measured = float(runtime["measured_updates_per_second"])
    conservative = float(runtime["conservative_updates_per_second"])
    identity = validated["independent_gpu_identity"]
    clean = {
        "status": "PASS_AUTHENTICATED_SOURCE_FREE_REVIEW",
        "v9_source_snapshot_root_sha256": require_digest(v9_source_root, "v9 source root"),
        "v8_source_snapshot_root_sha256": require_digest(v8_source_root, "v8 source root"),
        "preflight_receipt_sha256": require_digest(record["receipt_sha256"], "preflight receipt"),
        "support_sha256": PINNED_SUPPORT_SHA256,
        "measured_updates_per_second": measured,
        "conservative_updates_per_second": conservative,
        "device_name": str(identity["device_name"]),
        "device_uuid": str(identity["device_uuid"]),
        "pci_bus_id": str(identity["pci_bus_id"]),
    }
    review = SourceFreeReview(**clean, receipt_sha256=sha256(canonical_json(clean)))
    verify_source_free_review(review)
    return review, typed, record


def held_qwen_artifact_after_review(support: Any, review: SourceFreeReview, path: Path) -> Any:
    """The only Qwen artifact opener; review verification precedes delegation."""

    verify_source_free_review(review)
    return support.HeldRegularInput(
        path,
        label="authenticated current Qwen artifact",
        expected_sha256=CURRENT_ARTIFACT_SHA256,
        expected_bytes=CURRENT_ARTIFACT_BYTES,
        maximum_bytes=CURRENT_ARTIFACT_BYTES,
    )


def primary_runtime_admission(
    stage: Any,
    common: Any,
    protocol: Any,
    panel: Mapping[str, Any],
    review: SourceFreeReview,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Admit only exact primary work; disclose but never charge deferred work."""

    verify_source_free_review(review)
    projection = stage.projected_updates(common, protocol, panel)
    require(projection["primary_fold_policy"] == "disjoint_stream_owner_dependence_components", "primary fold policy")
    require(projection["primary_exact_identity_estimable"] is True, "primary estimability")
    require(projection["disjoint_dependence_component_count"] == 3, "primary component count")
    require(projection["exact_cell_symbol_updates"] == PINNED_PRIMARY_CELL_SYMBOL_UPDATES, "primary update pin")
    require(
        projection["maximum_source_survivor_updates_including_four_shuffles"] == PINNED_DEFERRED_MAXIMUM_UPDATES,
        "deferred maximum update pin",
    )
    require(
        projection["coordinate_disjoint_diagnostic_cell_symbol_updates"] == PINNED_DEFERRED_COORDINATE_UPDATES,
        "deferred coordinate update pin",
    )
    require(projection["coordinate_disjoint_diagnostic_estimable_folds"] == 6, "deferred coordinate fold pin")
    require(projection["passes_pre_fit_resource_budget"] is True, "resource admission")
    require(projection["passes_pre_fit_runtime_budget"] is False, "known v8 full-survivor runtime disposition")
    static = projection["static_resource_admission"]
    require(static["streams"] == PINNED_PANEL_STREAMS, "panel stream pin")
    require(static["symbols"] == PINNED_PANEL_SYMBOLS, "panel symbol pin")
    require(static["passes"] is True, "static resource pass")
    observed_folds = tuple(
        (
            int(row["component_ordinal"]),
            tuple(int(value) for value in row["identity_indices"]),
            int(row["cell_symbol_updates"]),
        )
        for row in projection["folds"]
    )
    require(observed_folds == PINNED_FOLD_UPDATES, "exact fold workload pins")
    bank = common.candidate_bank()
    require(len(bank) == 150, "complete candidate bank")
    require([candidate.selector_ordinal for candidate in bank] == list(range(150)), "canonical candidate selectors")
    conservative = float(review.conservative_updates_per_second)
    kernel_seconds = PINNED_PRIMARY_CELL_SYMBOL_UPDATES / conservative
    admission = {
        "schema": "uwfa-sc-v9-primary-runtime-admission-v0",
        "status": "PASS_PRIMARY_KERNEL_WORKLOAD_ADMITTED",
        "exact_primary_cell_symbol_updates": PINNED_PRIMARY_CELL_SYMBOL_UPDATES,
        "authenticated_live_measured_updates_per_second": review.measured_updates_per_second,
        "authenticated_live_conservative_updates_per_second": conservative,
        "conservative_fraction_of_measured": 0.5,
        "throughput_tamper_bounds": [CONSERVATIVE_THROUGHPUT_MIN, CONSERVATIVE_THROUGHPUT_MAX],
        "projected_primary_gpu_kernel_work_seconds": kernel_seconds,
        "primary_gpu_kernel_budget_seconds": PRIMARY_KERNEL_BUDGET_SECONDS,
        "passes": kernel_seconds <= PRIMARY_KERNEL_BUDGET_SECONDS,
        "is_total_launch_wall_time_projection": False,
        "unmodeled_wall_components": [
            "one authenticated STRATA panel decode before primary fitting",
            "host-side final arithmetic encode/decode and canonical rebuild",
            "routed and standalone final causal decode plus physical metrics",
            "filesystem publication and synchronization",
        ],
        "evaluation_runner_pins_are_decoder_identity_inputs": False,
        "deferred_not_counted_or_executed": {
            "four_survivor_shuffles_and_coordinate_diagnostic_maximum_updates": PINNED_DEFERRED_MAXIMUM_UPDATES - PINNED_PRIMARY_CELL_SYMBOL_UPDATES,
            "coordinate_diagnostic_updates": PINNED_DEFERRED_COORDINATE_UPDATES,
            "matched_controls": "separate authorization; no control path exists in this runner",
        },
    }
    require(admission["passes"] is True, "primary kernel runtime budget")
    admission["admission_sha256"] = sha256(canonical_json(admission))
    return projection, admission


def execute_exact_primary(
    modules: Mapping[str, Any],
    backend: Any,
    adapter: Any,
    panel: Mapping[str, Any],
    score: Mapping[str, Any],
    bindings: Any,
    descriptor_builder: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The sole scientific path: exact primary holdout, then final container."""

    stage = modules["stage"]
    cache = stage.prepare_backend_cache(backend, panel)
    scientific = stage.nested_holdout(
        modules["common"],
        modules["protocol"],
        modules["codec"],
        backend,
        cache,
        panel,
        policy="exact_identity",
        diagnostic_only=False,
    )
    require(scientific.get("estimable") is True, "primary holdout became unestimable")
    require(scientific.get("primary_policy") == "exact_identity", "primary result policy")
    selected_row = scientific["final_topology_selected_from_nested_fold_votes"]
    selected = modules["common"].candidate_bank()[int(selected_row["selector_ordinal"])]
    require(selected.as_dict() == selected_row, "selected candidate bank identity")
    physical = stage.final_container(
        modules["common"],
        modules["codec"],
        modules["semantic"],
        adapter,
        backend,
        cache,
        panel,
        selected,
        score,
        bindings,
        descriptor_builder,
    )
    return scientific, physical


def primary_status(scientific: Mapping[str, Any], physical: Mapping[str, Any]) -> str:
    metrics = physical["parsed_metrics"]
    integrity = bool(
        physical["standalone_decode"]["all_payloads_canonically_reencoded"]
        and physical["identical_reconstruction_proved_by_full_f64_digest"]
        and physical["all_adapted_values_deserialized_from_transmitted_model"]
    )
    if not integrity:
        return "FAIL_EVIDENCE_INTEGRITY_PRIMARY_CONTAINER"
    if not bool(metrics["passes_rate_interval"] and metrics["passes_F_target"]):
        return "HARD_KILL_PRIMARY_PHYSICAL_RATE_OR_F"
    if not bool(metrics["passes_cold_read_below_2x"]):
        return "FAIL_PRIMARY_STRICT_COLD_READ"
    if not bool(scientific["passes_heldout_gate"]):
        return "NO_PROMOTION_PRIMARY_NESTED_HELDOUT"
    return "PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED"


def _write_exclusive(dir_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    require(name and name not in {".", ".."} and "/" not in name and "\\" not in name, "output name")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
    try:
        cursor = 0
        while cursor < len(payload):
            written = os.write(fd, payload[cursor:])
            require(written > 0, "short output write")
            cursor += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"name": name, "bytes": len(payload), "sha256": sha256(payload)}


def publish(output_dir: Path, members: Mapping[str, bytes], *, status: str, source_root: str) -> dict[str, Any]:
    output_dir = require_absolute_lexical(output_dir, "output directory")
    parent = output_dir.parent
    reject_symlink_chain(parent, "output parent")
    require(stat.S_ISDIR(os.lstat(parent).st_mode), "output parent must be directory")
    os.mkdir(output_dir, 0o700)
    dir_fd = os.open(
        os.fspath(output_dir),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        rows = []
        for name in sorted(members, key=lambda value: value.encode("utf-8")):
            rows.append(_write_exclusive(dir_fd, name, members[name]))
        complete = {
            "schema": "uwfa-sc-v9-qwen-primary-completion-v0",
            "status": status,
            "positive_claim_authority": False,
            "controls_run": False,
            "shuffles_run": False,
            "coordinate_diagnostic_run": False,
            "v9_source_snapshot_root_sha256": source_root,
            "members": rows,
        }
        complete["completion_sha256"] = sha256(canonical_json(complete))
        completion_row = _write_exclusive(dir_fd, "COMPLETE.json", pretty_json(complete))
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return {"output_dir": os.fspath(output_dir), "members": rows + [completion_row], "completion": complete}


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    launched = time.perf_counter()
    require(arguments.authorization == AUTHORIZATION, "explicit primary-only authorization token")
    # Lexical validation does not stat or open either leaf.
    artifact_path = require_absolute_lexical(Path(arguments.artifact), "artifact")
    output_dir = require_absolute_lexical(Path(arguments.output_dir), "output directory")

    v9_closure = authenticate_v9_package(Path(arguments.v9_package))
    support, support_record = load_authenticated_support(Path(arguments.pinned_support))
    v8_closure = support.authenticate_v8_package(Path(arguments.v8_package))
    require(v8_closure["manifest_sha256"] == SEALED_V8_MANIFEST_SHA256, "sealed v8 manifest pin")
    modules = support.load_v8_modules(v8_closure)

    # Numerical imports and CUDA occur only after both source packages close.
    # No Qwen/control leaf has been statted, opened or enumerated.
    import numpy as np
    import cupy as cp

    external = support.load_external_decoder_sources(
        Path(arguments.strata_common), Path(arguments.frozen_auditor)
    )
    require(support_record["semantic_bridge_abi"] == "list[list[int]]", "list group-ordinal ABI")
    require(support_record["tuple_rows_forbidden"] is True, "tuple group-ordinal rows forbidden")
    backend = modules["backend_source"].build_backend(cp)
    review, typed_preflight, preflight_record = run_source_free_review(
        support,
        modules,
        backend,
        v9_source_root=v9_closure["source_snapshot_root_sha256"],
        v8_source_root=v8_closure["source_snapshot_root_sha256"],
    )

    # Sole first access to the Qwen artifact, guarded by the sealed review.
    with held_qwen_artifact_after_review(support, review, artifact_path) as held_artifact:
        artifact_bytes = held_artifact.data
        artifact_identity = {
            "st_dev": held_artifact.identity[0],
            "st_ino": held_artifact.identity[1],
            "bytes": held_artifact.identity[2],
            "mtime_ns": held_artifact.identity[3],
            "sha256": held_artifact.sha256,
        }

    raw_adapter = modules["adapter_source"].StrataSCAdapter(
        common=modules["common"],
        semantic_codec=modules["semantic"],
        np=np,
        frozen_auditor=external["frozen_auditor"],
        strata_common=external["strata_common"],
        device="cupy",
    )
    adapter = support.SingleArtifactPanelCache(raw_adapter)
    panel = modules["stage"].prepare_panel(modules["protocol"], adapter, artifact_bytes)
    require(panel["artifact"]["raw_sha256"] == artifact_identity["sha256"], "panel/artifact SHA-256 binding")
    require(panel["artifact"]["raw_bytes"] == artifact_identity["bytes"], "panel/artifact byte binding")
    require(int(panel["weights"]) == SOURCE_WEIGHTS, "panel source-weight binding")
    full_geometry = modules["protocol"].geometry_sha256(modules["common"], panel)
    structural_geometry = modules["protocol"].structural_geometry_sha256(modules["common"], panel)
    reconstruction_sha = str(panel["reconstruction"]["full_reconstruction_f64_sha256"])

    decoder_bundle = {
        "schema": "uwfa-sc-v9-primary-decoder-bundle-v0",
        "members": [
            {"name": "strata_expert_local_codec/common.py", "sha256": support.STRATA_COMMON_SHA256},
            {"name": "strata_v2_klt_mixed_independent_auditor_v1.py", "sha256": support.FROZEN_AUDITOR_SHA256},
            {"name": "strata_sc_adapter.py", "sha256": v8_closure["member_hashes"]["strata_sc_adapter.py"]},
            {"name": "universal_adapter.py", "sha256": v8_closure["member_hashes"]["universal_adapter.py"]},
            {"name": "container_codec.py", "sha256": v8_closure["member_hashes"]["container_codec.py"]},
        ],
        "exploratory_semantic_bridge": external["group_ordinal_abi"],
        "semantic_bridge_container_abi": "list[list[int]]",
        "tuple_rows_forbidden_by_numpy_advanced_index_semantics": True,
        "single_artifact_panel_cache_required": True,
    }
    decoder_bundle_sha = sha256(canonical_json(decoder_bundle))
    score, score_bytes = support.build_score_receipt(
        modules["common"],
        artifact_sha256=artifact_identity["sha256"],
        reconstruction_sha256=reconstruction_sha,
        full_geometry_sha256=full_geometry,
        decoder_bundle_sha256=decoder_bundle_sha,
    )
    modules["protocol"].validate_score_receipt(
        score,
        artifact_sha256=artifact_identity["sha256"],
        artifact_bytes=CURRENT_ARTIFACT_BYTES,
        weights=SOURCE_WEIGHTS,
        reconstruction_sha256=reconstruction_sha,
        original_source_panel_sha256=full_geometry,
        independent_decoder_source_sha256=decoder_bundle_sha,
    )

    runner_sha = v9_closure["member_hashes"]["primary_gate.py"]
    pipeline_record = {
        "schema": "uwfa-sc-v9-primary-pipeline-v0",
        "v9_source_snapshot_root_sha256": v9_closure["source_snapshot_root_sha256"],
        "v8_source_snapshot_root_sha256": v8_closure["source_snapshot_root_sha256"],
        "sealed_v8_manifest_sha256": v8_closure["manifest_sha256"],
        "pinned_support_sha256": support_record["sha256"],
        "runner_sha256": runner_sha,
        "decoder_bundle_sha256": decoder_bundle_sha,
        "baseline_plan_sha256": support.BASELINE_PLAN_SHA256,
        "scope": "exact primary nested holdout plus final physical container only",
    }
    pipeline_sha = sha256(canonical_json(pipeline_record))
    bindings = modules["stage"].BoundEvidence(
        baseline_plan_sha256=support.BASELINE_PLAN_SHA256,
        baseline_score_sha256=sha256(score_bytes),
        universal_decoder_sha256=decoder_bundle_sha,
        producer_manifest_sha256=v8_closure["manifest_sha256"],
        audit_bootstrap_sha256=runner_sha,
        source_full_geometry_sha256=full_geometry,
        source_structural_geometry_sha256=structural_geometry,
        extraction_program_sha256=v8_closure["member_hashes"]["strata_sc_adapter.py"],
        universal_adapter_sha256=v8_closure["member_hashes"]["universal_adapter.py"],
        pipeline_sha256=pipeline_sha,
        source_snapshot_root_sha256=v8_closure["source_snapshot_root_sha256"],
        source_preflight_receipt_sha256=preflight_record["receipt_sha256"],
    )
    validated_again = modules["stage"].validate_source_preflight(
        modules["common"], modules["protocol"], typed_preflight, bindings
    )
    require(validated_again["receipt_sha256"] == review.preflight_receipt_sha256, "post-source preflight revalidation")

    # A second identical extraction request is satisfied from the retained
    # exact panel object, proving a single underlying causal artifact decode.
    primary_panel = modules["stage"].prepare_panel(modules["protocol"], adapter, artifact_bytes)
    require(primary_panel is panel, "single decoded panel object was not reused")
    cache_receipt = adapter.receipt()
    require(cache_receipt["same_panel_object_reused"] is True, "panel-cache receipt")

    projection, admission = primary_runtime_admission(
        modules["stage"], modules["common"], modules["protocol"], primary_panel, review
    )
    scientific, physical = execute_exact_primary(
        modules,
        backend,
        adapter,
        primary_panel,
        score,
        bindings,
        support.descriptor_source_builder(modules["codec"]),
    )
    status = primary_status(scientific, physical)
    public_physical = {
        key: value
        for key, value in physical.items()
        if key not in {"container", "identity_framing_container"}
    }
    metrics = physical["parsed_metrics"]
    compact_physical = {
        "container_bytes": metrics["actual_container_bytes"],
        "physical_rate_bpw": metrics["actual_physical_rate_bpw"],
        "physical_rate_rational": metrics["actual_physical_rate_rational"],
        "relative_mse": metrics["audited_identical_reconstruction_relative_mse"],
        "F": metrics["F_from_actual_bytes_and_identical_reconstruction"],
        "net_physical_saving_bpw": metrics["net_physical_saving_bpw"],
        "passes_rate_interval": metrics["passes_rate_interval"],
        "passes_F_target": metrics["passes_F_target"],
        "passes_cold_read_below_2x": metrics["passes_cold_read_below_2x"],
        "container_sha256": physical["container_sha256"],
        "identity_framing_container_sha256": physical["identity_framing_container_sha256"],
        "model_packet_sha256": physical["model_packet_sha256"],
    }
    total_elapsed = time.perf_counter() - launched
    result = {
        "schema": SCHEMA,
        "status": status,
        "positive_claim_authority": False,
        "positive_claim_even_if_all_primary_gates_pass": False,
        "controls_run": False,
        "controls_may_be_opened_or_inferred_from_this_result": False,
        "shuffles_run": False,
        "coordinate_disjoint_diagnostic_run": False,
        "deferred_stages": {
            "survivor_shuffles": "NOT_RUN_REQUIRES_SEPARATE_REVIEW_AND_AUTHORIZATION",
            "coordinate_disjoint_diagnostic": "NOT_RUN_REQUIRES_SEPARATE_REVIEW_AND_AUTHORIZATION",
            "matched_gaussian_controls": "NOT_RUN_REQUIRES_SEPARATE_REVIEW_AND_AUTHORIZATION",
            "independent_result_audit": "REQUIRED_BEFORE_ANY_CLAIM",
        },
        "claim_boundary": "runtime repair for exact sealed-v8 primary Qwen estimand only; never a compression claim or universal SwiGLU-MoE result",
        "artifact_identity": artifact_identity,
        "baseline_score": score,
        "source_full_geometry_sha256": full_geometry,
        "source_structural_geometry_sha256": structural_geometry,
        "recomputed_panel_reconstruction_f64_sha256": reconstruction_sha,
        "source_free_review": {**review.clean_record(), "receipt_sha256": review.receipt_sha256},
        "runtime_admission": admission,
        "original_v8_full_survivor_projection": projection,
        "scientific_primary_nested_holdout": scientific,
        "source_final": public_physical,
        "physical": compact_physical,
        "exploratory_panel_cache": cache_receipt,
        "decoder_bundle": decoder_bundle,
        "decoder_bundle_sha256": decoder_bundle_sha,
        "pipeline_record": pipeline_record,
        "pipeline_sha256": pipeline_sha,
        "telemetry": backend.environment_receipt(),
        "total_observed_launch_wall_seconds": total_elapsed,
        "runtime_projection_was_not_total_wall_time": True,
        "evaluation_workload_pins_are_decoder_identity_inputs": False,
        "source_hashes": {
            "v9_source_manifest_sha256": v9_closure["manifest_sha256"],
            "v9_source_snapshot_root_sha256": v9_closure["source_snapshot_root_sha256"],
            "v9_members": v9_closure["member_hashes"],
            "sealed_v8_manifest_sha256": v8_closure["manifest_sha256"],
            "sealed_v8_source_snapshot_root_sha256": v8_closure["source_snapshot_root_sha256"],
            "pinned_support_sha256": support_record["sha256"],
            **external["source_hashes"],
        },
    }
    result_bytes = pretty_json(result)
    require(len(result_bytes) <= MAX_RESULT_JSON_BYTES, "result JSON bound")
    members = {
        "RESULT.json": result_bytes,
        "BOUND_BASELINE_SCORE.json": score_bytes,
        "SOURCE_PREFLIGHT.json": pretty_json(preflight_record),
        "DECODER_BUNDLE.json": pretty_json(decoder_bundle),
        "UWFCV8.bin": bytes(physical["container"]),
        "IDENTITY_FRAMING.bin": bytes(physical["identity_framing_container"]),
    }
    publication = publish(
        output_dir,
        members,
        status=status,
        source_root=v9_closure["source_snapshot_root_sha256"],
    )
    return {
        "schema": "uwfa-sc-v9-qwen-primary-launch-summary-v0",
        "status": status,
        "positive_claim_authority": False,
        "controls_run": False,
        "shuffles_run": False,
        "coordinate_disjoint_diagnostic_run": False,
        "output_dir": publication["output_dir"],
        "result_sha256": sha256(result_bytes),
        "container_sha256": physical["container_sha256"],
        "physical": compact_physical,
        "winner": scientific["final_topology_selected_from_nested_fold_votes"],
        "pooled_exact_heldout_saving_bpw": scientific["pooled_exact_heldout_saving_bpw"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--v9-package", required=True)
    result.add_argument("--pinned-support", required=True)
    result.add_argument("--v8-package", required=True)
    result.add_argument("--strata-common", required=True)
    result.add_argument("--frozen-auditor", required=True)
    result.add_argument("--artifact", required=True)
    result.add_argument("--output-dir", required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    summary = run(arguments)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(f"FAIL_UWFA_SC_V9_QWEN_PRIMARY_GATE: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
