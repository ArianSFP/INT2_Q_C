#!/usr/bin/env python3
"""Run the bounded UWFA-SC continuous posterior-centroid v1 diagnostic.

This runner is inert on import.  Execution requires an explicit authorization,
an independently audited completed v9 result, a separately sealed launch
review, the exact result-bound decoder sources, and a generic hash-authorized
SwiGLU BF16 score manifest.  It never edits v0, v8, or v9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
import time
import types
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


AUTHORIZATION = "RUN_UWFA_SC_POSTERIOR_CENTROID_V1_DISCOVERY"
SOURCE_MANIFEST_SCHEMA = "swiglu-bf16-score-panel-v0"
PACKAGE_MANIFEST_SCHEMA = "uwfa-sc-posterior-centroid-source-manifest-v1"
RESULT_SCHEMA = "uwfa-sc-posterior-centroid-result-v1"
PREDECESSOR_AUDIT_SCHEMA = "uwfa-sc-v9-primary-independent-result-audit-v0"
PREDECESSOR_AUDIT_STATUS = "PASS_FAIL_CLOSED_NONPROMOTING_PRIMARY_RESULT_AUDIT"
LAUNCH_REVIEW_SCHEMA = "uwfa-sc-posterior-centroid-v1-launch-review-v0"
LAUNCH_REVIEW_STATUS = "APPROVED_FOR_ONE_NONPROMOTING_DISCOVERY_RUN"
AUDITED_PUBLICATION_MEMBERS = {
    "BOUND_BASELINE_SCORE.json",
    "COMPLETE.json",
    "DECODER_BUNDLE.json",
    "IDENTITY_FRAMING.bin",
    "RESULT.json",
    "SOURCE_PREFLIGHT.json",
    "UWFCV8.bin",
}
MAX_SOURCE_MANIFEST_BYTES = 1 << 20
MAX_SOURCE_MATRIX_BYTES = 1 << 30


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


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


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} SHA-256",
    )
    return value


def _regular_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    require(path.is_absolute(), f"{label} absolute path")
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{label} regular file")
    require(0 < metadata.st_size <= maximum, f"{label} byte bound")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        output = bytearray()
        while len(output) < before.st_size:
            chunk = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(chunk), f"{label} short read")
            output.extend(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label} changed while held",
        )
    finally:
        os.close(descriptor)
    return bytes(output)


def _fraction_from_record(row: Mapping[str, Any], label: str) -> Fraction:
    require(isinstance(row, Mapping), f"{label} fraction record")
    numerator = row.get("numerator")
    denominator = row.get("denominator")
    require(type(numerator) is int and type(denominator) is int and denominator > 0, f"{label} fraction")
    return Fraction(numerator, denominator)


def authenticate_own_package(
    package_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate this source closure before any predecessor/source input."""

    manifest_payload = _regular_bytes(
        package_dir / "SOURCE_MANIFEST.json",
        maximum=MAX_SOURCE_MANIFEST_BYTES,
        label="posterior package manifest",
    )
    require(
        sha256(manifest_payload)
        == require_digest(expected_manifest_sha256, "expected posterior package manifest"),
        "posterior package manifest authorization",
    )
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except Exception as error:
        raise DiagnosticError(f"posterior package manifest JSON: {error}") from error
    require(
        isinstance(manifest, dict) and manifest.get("schema") == PACKAGE_MANIFEST_SCHEMA,
        "posterior package manifest schema",
    )
    require(
        manifest.get("status") == "SEALED_SOURCE_ONLY_NONPROMOTING_NO_PAYLOAD_AUTHORITY",
        "posterior package manifest status",
    )
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "posterior package members")
    observed = []
    sources: dict[str, bytes] = {}
    for row in rows:
        require(
            isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
            "posterior package member row",
        )
        name = row["name"]
        require(
            isinstance(name, str)
            and name
            and "/" not in name
            and "\\" not in name
            and name != "SOURCE_MANIFEST.json",
            "posterior package member name",
        )
        payload = _regular_bytes(
            package_dir / name,
            maximum=4 * (1 << 20),
            label=f"posterior package member {name}",
        )
        require(len(payload) == int(row["bytes"]), f"posterior package {name} bytes")
        member_digest = require_digest(row["sha256"], f"posterior package {name}")
        require(sha256(payload) == member_digest, f"posterior package {name} digest")
        observed.append({"name": name, "bytes": len(payload), "sha256": member_digest})
        sources[name] = payload
    expected_names = {
        "README.md",
        "design_lock.json",
        "diagnostic.py",
        "posterior_core.py",
        "result_bridge.py",
        "test_source_only.py",
        "verify_source.py",
    }
    require({row["name"] for row in observed} == expected_names, "posterior package exact source closure")
    observed.sort(key=lambda row: row["name"].encode("utf-8"))
    root = sha256(canonical_json(observed))
    require(manifest.get("source_snapshot_root_sha256") == root, "posterior source snapshot root")
    return {
        "manifest_sha256": sha256(manifest_payload),
        "source_snapshot_root_sha256": root,
        "members": observed,
        "sources": sources,
    }


def authenticate_launch_preconditions(
    arguments: argparse.Namespace,
    package_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate audit/review authority before touching the Qwen result.

    Neither record contains model payload.  Both paths and expected digests
    are explicit CLI inputs.  The launch review binds this exact source
    closure, score panel, result directory string, output namespace, RHT
    backend, and predecessor audit receipt.
    """

    audit_payload = _regular_bytes(
        Path(arguments.predecessor_audit_receipt),
        maximum=MAX_SOURCE_MANIFEST_BYTES,
        label="predecessor independent result-audit receipt",
    )
    audit_sha = sha256(audit_payload)
    require(
        audit_sha
        == require_digest(
            arguments.predecessor_audit_receipt_sha256,
            "expected predecessor audit receipt",
        ),
        "predecessor audit receipt authorization",
    )
    try:
        audit = json.loads(audit_payload.decode("utf-8"))
    except Exception as error:
        raise DiagnosticError(f"predecessor audit receipt JSON: {error}") from error
    require(isinstance(audit, dict), "predecessor audit receipt object")
    require(audit.get("schema") == PREDECESSOR_AUDIT_SCHEMA, "predecessor audit schema")
    require(audit.get("status") == PREDECESSOR_AUDIT_STATUS, "predecessor audit pass status")
    require(audit.get("positive_claim_authority") is False, "predecessor audit nonpromoting")
    rows = audit.get("publication_members")
    require(
        isinstance(rows, dict) and set(rows) == AUDITED_PUBLICATION_MEMBERS,
        "audited publication exact members",
    )
    for name, row in rows.items():
        require(
            isinstance(row, dict)
            and set(row) == {"bytes", "sha256"}
            and type(row["bytes"]) is int
            and row["bytes"] > 0,
            f"audited publication member row {name}",
        )
        require_digest(row["sha256"], f"audited publication member {name}")

    review_payload = _regular_bytes(
        Path(arguments.launch_review),
        maximum=MAX_SOURCE_MANIFEST_BYTES,
        label="posterior v1 launch review",
    )
    review_sha = sha256(review_payload)
    require(
        review_sha
        == require_digest(arguments.launch_review_sha256, "expected launch review"),
        "launch review authorization",
    )
    try:
        review = json.loads(review_payload.decode("utf-8"))
    except Exception as error:
        raise DiagnosticError(f"launch review JSON: {error}") from error
    require(isinstance(review, dict), "launch review object")
    expected_review_keys = {
        "schema",
        "status",
        "positive_claim_authority",
        "authorization",
        "predecessor_audit_receipt_sha256",
        "posterior_source_manifest_sha256",
        "score_manifest_sha256",
        "v9_result_dir",
        "source_manifest",
        "output_dir",
        "rht_device",
        "output_directory_must_not_exist",
    }
    require(set(review) == expected_review_keys, "launch review exact schema")
    require(review["schema"] == LAUNCH_REVIEW_SCHEMA, "launch review schema")
    require(review["status"] == LAUNCH_REVIEW_STATUS, "launch review status")
    require(review["positive_claim_authority"] is False, "launch review nonpromoting")
    require(review["authorization"] == AUTHORIZATION, "launch review authorization token")
    require(review["predecessor_audit_receipt_sha256"] == audit_sha, "launch review audit binding")
    require(
        review["posterior_source_manifest_sha256"] == package_closure["manifest_sha256"],
        "launch review posterior source binding",
    )
    require(
        review["score_manifest_sha256"]
        == require_digest(arguments.source_manifest_sha256, "score manifest launch argument"),
        "launch review score-manifest binding",
    )
    require(review["v9_result_dir"] == os.fspath(arguments.v9_result_dir), "launch review result path")
    require(review["source_manifest"] == os.fspath(arguments.source_manifest), "launch review score path")
    require(review["output_dir"] == os.fspath(arguments.output_dir), "launch review output path")
    require(review["rht_device"] == arguments.rht_device, "launch review RHT device")
    require(review["output_directory_must_not_exist"] is True, "launch review exclusive output")
    require(
        not os.path.lexists(arguments.output_dir),
        "launch-reviewed output directory already exists",
    )
    return {
        "predecessor_audit_receipt_sha256": audit_sha,
        "launch_review_sha256": review_sha,
        "audit": audit,
    }


def bind_publication_to_audit(
    publication: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> None:
    """Bind every completed predecessor member to the passing audit receipt."""

    actual = {
        name: {"bytes": len(payload), "sha256": sha256(payload)}
        for name, payload in publication["members"].items()
    }
    actual["COMPLETE.json"] = {
        "bytes": len(publication["complete_bytes"]),
        "sha256": sha256(publication["complete_bytes"]),
    }
    require(
        actual == launch["audit"]["publication_members"],
        "publication differs from passing audit receipt",
    )


def _load_authenticated_owned_module(
    closure: Mapping[str, Any],
    *,
    member_name: str,
    private_name: str,
) -> Any:
    """Compile retained manifest-authenticated bytes, never a live sibling."""

    sources = closure.get("sources")
    require(isinstance(sources, Mapping) and member_name in sources, "owned module retained bytes")
    source = sources[member_name]
    digest_by_name = {row["name"]: row["sha256"] for row in closure["members"]}
    expected = require_digest(digest_by_name.get(member_name), f"owned module {member_name}")
    require(sha256(source) == expected, "owned retained module binding")
    existing = sys.modules.get(private_name)
    if existing is not None:
        require(
            getattr(existing, "__authenticated_sha256__", None) == expected,
            "owned authenticated module-name collision",
        )
        return existing
    module = types.ModuleType(private_name)
    module.__file__ = f"<authenticated-owned:{member_name}:{expected}>"
    module.__package__ = ""
    module.__authenticated_sha256__ = expected
    code = compile(source, module.__file__, "exec", dont_inherit=True, optimize=0)
    sys.modules[private_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(private_name) is module:
            sys.modules.pop(private_name, None)
        raise
    return module


def authenticate_source_panel(
    np: Any,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_artifact_sha256: str,
    experts: int,
    intermediate: int,
    hidden: int,
    selected_experts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Open only selected experts from a generic identity-free manifest.

    The manifest authenticates the complete panel, but unselected BF16 leaves
    are not statted, hashed, opened, or numerically materialized by this call.
    """

    payload = _regular_bytes(manifest_path, maximum=MAX_SOURCE_MANIFEST_BYTES, label="source manifest")
    require(sha256(payload) == require_digest(expected_manifest_sha256, "expected source manifest"), "source manifest authorization")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise DiagnosticError(f"source manifest JSON: {error}") from error
    require(isinstance(manifest, dict) and manifest.get("schema") == SOURCE_MANIFEST_SCHEMA, "source manifest schema")
    require(
        manifest.get("bound_artifact_sha256") == require_digest(expected_artifact_sha256, "expected artifact"),
        "source panel/artifact binding",
    )
    require(int(manifest.get("experts", -1)) == experts, "source manifest experts")
    selected = (
        tuple(range(experts))
        if selected_experts is None
        else tuple(sorted(int(value) for value in selected_experts))
    )
    require(selected and tuple(sorted(set(selected))) == selected, "selected source experts")
    require(0 <= selected[0] and selected[-1] < experts, "selected source expert bounds")
    selected_set = set(selected)
    rows = manifest.get("matrices")
    require(isinstance(rows, list) and len(rows) == 3 * experts, "source manifest matrix count")
    root = manifest_path.parent
    matrices: dict[tuple[int, str], Any] = {}
    seen_keys: set[tuple[int, str]] = set()
    clean_rows = []
    source_energy = 0.0
    for ordinal, row in enumerate(rows):
        require(isinstance(row, dict), "source matrix row")
        require(
            set(row) == {"expert_ordinal", "role", "shape", "relative_path", "bytes", "sha256"},
            "identity-free source matrix schema",
        )
        expert = row["expert_ordinal"]
        role = row["role"]
        require(type(expert) is int and 0 <= expert < experts, "source expert ordinal")
        require(role in {"gate", "up", "down"}, "source role")
        key = (expert, role)
        require(key not in seen_keys, "duplicate source matrix")
        seen_keys.add(key)
        expected_shape = [hidden, intermediate] if role == "down" else [intermediate, hidden]
        require(row["shape"] == expected_shape, "source matrix shape")
        expected_bytes = 2 * intermediate * hidden
        require(type(row["bytes"]) is int and row["bytes"] == expected_bytes, "source matrix bytes")
        relative = row["relative_path"]
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "source relative path",
        )
        digest = require_digest(row["sha256"], "source matrix")
        clean_rows.append({
            "expert_ordinal": expert,
            "role": role,
            "shape": expected_shape,
            "bytes": expected_bytes,
            "sha256": digest,
        })
        if expert in selected_set:
            path = (root / relative).resolve(strict=False)
            require(path.parent == root.resolve() or root.resolve() in path.parents, "source path containment")
            raw = _regular_bytes(path, maximum=MAX_SOURCE_MATRIX_BYTES, label=f"source matrix {ordinal}")
            require(len(raw) == expected_bytes, "source matrix exact bytes")
            require(sha256(raw) == digest, "source matrix digest")
            words = np.frombuffer(raw, dtype="<u2")
            values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
            require(values.size == intermediate * hidden and bool(np.all(np.isfinite(values))), "source matrix values")
            matrix = values.reshape(tuple(expected_shape))
            matrices[key] = matrix
            source_energy += float(np.sum(matrix * matrix, dtype=np.float64))
    require(
        seen_keys
        == {(expert, role) for expert in range(experts) for role in ("gate", "up", "down")},
        "complete source manifest role grid",
    )
    require(
        set(matrices)
        == {(expert, role) for expert in selected for role in ("gate", "up", "down")},
        "selected source role grid",
    )
    clean_rows.sort(key=lambda row: (row["expert_ordinal"], ("gate", "up", "down").index(row["role"])))
    record_root = sha256(canonical_json(clean_rows))
    require(
        manifest.get("source_record_set_sha256") == record_root,
        "source record-set commitment",
    )
    return {
        "manifest": manifest,
        "manifest_sha256": sha256(payload),
        "source_record_set_sha256": record_root,
        "matrices": matrices,
        "source_energy_fp64": source_energy,
        "materialized_experts": list(selected),
        "unselected_BF16_leaves_opened_statted_hashed_or_enumerated": False,
        "identity_fields_available_to_decoder": False,
    }


def source_post_coordinates(
    np: Any,
    source: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    experts: int,
    intermediate: int,
    hidden: int,
) -> Any:
    """Map source matrices into the exact decoded Gate/z0/z1 group domain."""

    coefficients = struct.unpack_from(f"<{2 * experts}f", bytes(metadata["header"]), 32)
    post = np.empty((experts * 3 * intermediate, hidden), dtype=np.float64)
    matrices = source["matrices"]
    for expert in source["materialized_experts"]:
        base = expert * 3 * intermediate
        gate = np.asarray(matrices[(expert, "gate")], dtype=np.float64)
        up = np.asarray(matrices[(expert, "up")], dtype=np.float64)
        down = np.asarray(matrices[(expert, "down")], dtype=np.float64).T
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        post[base : base + intermediate] = gate
        post[base + intermediate : base + 2 * intermediate] = cosine * up + sine * down
        post[base + 2 * intermediate : base + 3 * intermediate] = -sine * up + cosine * down
    return post


def candidate_source_matrices(
    np: Any,
    post: Any,
    metadata: Mapping[str, Any],
    *,
    experts: int,
    intermediate: int,
) -> dict[tuple[int, str], Any]:
    coefficients = struct.unpack_from(f"<{2 * experts}f", bytes(metadata["header"]), 32)
    output = {}
    for expert in range(experts):
        base = expert * 3 * intermediate
        gate = np.asarray(post[base : base + intermediate], dtype=np.float64)
        z0 = np.asarray(post[base + intermediate : base + 2 * intermediate], dtype=np.float64)
        z1 = np.asarray(post[base + 2 * intermediate : base + 3 * intermediate], dtype=np.float64)
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        require(norm2 > 0.0 and math.isfinite(norm2), "inverse role transform")
        up = (cosine * z0 - sine * z1) / norm2
        down = ((sine * z0 + cosine * z1) / norm2).T
        output[(expert, "gate")] = gate
        output[(expert, "up")] = up
        output[(expert, "down")] = down
    return output


def candidate_source_matrices_subset(
    np: Any,
    post: Any,
    metadata: Mapping[str, Any],
    selected_experts: Sequence[int],
    *,
    experts: int,
    intermediate: int,
) -> dict[tuple[int, str], Any]:
    coefficients = struct.unpack_from(f"<{2 * experts}f", bytes(metadata["header"]), 32)
    output = {}
    for expert in sorted(int(value) for value in selected_experts):
        require(0 <= expert < experts, "selected reconstruction expert")
        base = expert * 3 * intermediate
        gate = np.asarray(post[base : base + intermediate], dtype=np.float64)
        z0 = np.asarray(post[base + intermediate : base + 2 * intermediate], dtype=np.float64)
        z1 = np.asarray(post[base + 2 * intermediate : base + 3 * intermediate], dtype=np.float64)
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        require(norm2 > 0.0 and math.isfinite(norm2), "selected inverse role transform")
        output[(expert, "gate")] = gate
        output[(expert, "up")] = (cosine * z0 - sine * z1) / norm2
        output[(expert, "down")] = ((sine * z0 + cosine * z1) / norm2).T
    return output


def assemble_post(np: Any, decoded_blocks: Sequence[Any], values_by_ordinal: Mapping[int, Any], *, total_groups: int, hidden: int) -> Any:
    post = np.empty((total_groups, hidden), dtype=np.float64)
    covered: set[int] = set()
    for block in sorted(decoded_blocks, key=lambda item: item.ordinal):
        values = np.asarray(values_by_ordinal[block.ordinal], dtype=np.float64)
        require(values.ndim == 1 and values.size == len(block.group_ordinals) * hidden, "block reconstruction geometry")
        rows = values.reshape(len(block.group_ordinals), hidden)
        for local, group in enumerate(block.group_ordinals):
            require(group not in covered and 0 <= group < total_groups, "block group coverage")
            post[group] = rows[local]
            covered.add(group)
    require(covered == set(range(total_groups)), "complete post-coordinate coverage")
    return post


def score_matrices(
    np: Any,
    source: Mapping[str, Any],
    candidate: Mapping[tuple[int, str], Any],
    experts_to_score: Sequence[int],
) -> dict[str, Any]:
    source_matrices = source["matrices"]
    sse = 0.0
    energy = 0.0
    rows = []
    for expert in sorted(int(value) for value in experts_to_score):
        for role in ("gate", "up", "down"):
            target = np.asarray(source_matrices[(expert, role)], dtype=np.float64)
            reconstruction = np.asarray(candidate[(expert, role)], dtype=np.float64)
            require(target.shape == reconstruction.shape, "scored matrix shape")
            residual = target - reconstruction
            matrix_sse = float(np.sum(residual * residual, dtype=np.float64))
            matrix_energy = float(np.sum(target * target, dtype=np.float64))
            require(math.isfinite(matrix_sse) and math.isfinite(matrix_energy) and matrix_energy > 0.0, "matrix score")
            sse += matrix_sse
            energy += matrix_energy
            rows.append({
                "expert_ordinal": expert,
                "role": role,
                "sse_fp64": matrix_sse,
                "source_energy_fp64": matrix_energy,
                "relative_mse": matrix_sse / matrix_energy,
            })
    require(energy > 0.0, "score energy")
    return {
        "experts": list(sorted(int(value) for value in experts_to_score)),
        "matrices": rows,
        "sse_fp64": sse,
        "source_energy_fp64": energy,
        "relative_mse": sse / energy,
    }


def build_observations(
    np: Any,
    core: Any,
    coordinate: Mapping[str, Any],
    source_post: Any,
    frozen: Any,
    *,
    rht_device: str,
    selected_experts: Sequence[int] | None = None,
) -> tuple[Any, ...]:
    """Bind each decoded scalar coordinate to its continuous source target."""

    observations = []
    selected = (
        set(range(int(coordinate["experts"])))
        if selected_experts is None
        else {int(value) for value in selected_experts}
    )
    for block in coordinate["blocks"]:
        owners = set(block.owners)
        require(owners <= selected or owners.isdisjoint(selected), "source selection cuts a connected stream")
        if owners.isdisjoint(selected):
            continue
        target_source = np.asarray(source_post[list(block.group_ordinals)], dtype=np.float64).reshape(-1)
        require(target_source.size == np.asarray(block.indices).size, "hard fail: missing coordinate-aligned continuous observations")
        transformed, _rms = frozen.forward_signed_rht_and_rms(
            target_source,
            int(block.rht_seed_u64),
            rht_device,
        )
        scale = float(block.decoder_scale)
        require(math.isfinite(scale) and scale > 0.0, "decoder scale")
        target_normalized = np.asarray(transformed, dtype=np.float64) / scale
        require(target_normalized.shape == np.asarray(block.indices).shape, "hard fail: target/index alignment")
        observations.append(core.BlockObservation(
            ordinal=block.ordinal,
            owners=block.owners,
            indices=np.asarray(block.indices, dtype=np.int16),
            target_normalized=target_normalized,
            occupancy=np.asarray(block.occupancy, dtype=np.float64),
            coordinate_mapping_sha256=block.coordinate_mapping_sha256,
        ))
    require(observations, "selected decoded blocks have source targets")
    return tuple(observations)


def decoder_feature_observations(core: Any, coordinate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Build source-free feature records for posterior application only."""

    return tuple(
        core.BlockObservation(
            ordinal=block.ordinal,
            owners=block.owners,
            indices=block.indices,
            target_normalized=None,
            occupancy=block.occupancy,
            coordinate_mapping_sha256=block.coordinate_mapping_sha256,
        )
        for block in coordinate["blocks"]
    )


def reconstruct_from_parameters(
    np: Any,
    core: Any,
    coordinate: Mapping[str, Any],
    observations: Sequence[Any],
    parameters: Any,
    *,
    law: int,
    frozen: Any,
    strata: Any,
    metadata: Mapping[str, Any],
    rht_device: str,
    experts_to_reconstruct: Sequence[int] | None = None,
) -> dict[tuple[int, str], Any]:
    states = int(coordinate["states"])
    observation_by_ordinal = {block.ordinal: block for block in observations}
    selected_experts = tuple(range(int(coordinate["experts"]))) if experts_to_reconstruct is None else tuple(sorted(int(value) for value in experts_to_reconstruct))
    selected_set = set(selected_experts)
    values = {}
    selected_blocks = []
    for decoded in coordinate["blocks"]:
        owners = set(decoded.owners)
        require(owners <= selected_set or owners.isdisjoint(selected_set), "selected experts cut a connected stream")
        if owners.isdisjoint(selected_set):
            continue
        observation = observation_by_ordinal[decoded.ordinal]
        normalized = core.predict_normalized(
            np,
            observation,
            parameters,
            law=law,
            states=states,
        )
        transformed = normalized * float(decoded.decoder_scale)
        reconstructed = frozen.inverse_signed_rht(
            transformed,
            int(decoded.rht_seed_u64),
            rht_device,
        )
        values[decoded.ordinal] = np.asarray(reconstructed, dtype=np.float64)
        selected_blocks.append(decoded)
    hidden = int(strata.GROUP_VALUES)
    intermediate = int(strata.GROUPS_PER_MATRIX)
    experts = int(coordinate["experts"])
    post = np.empty((experts * 3 * intermediate, hidden), dtype=np.float64)
    covered: set[int] = set()
    for decoded in selected_blocks:
        reconstructed = np.asarray(values[decoded.ordinal], dtype=np.float64).reshape(len(decoded.group_ordinals), hidden)
        for local, group in enumerate(decoded.group_ordinals):
            require(group not in covered, "selected block duplicate group")
            post[group] = reconstructed[local]
            covered.add(group)
    expected_groups = {
        group
        for expert in selected_experts
        for group in range(expert * 3 * intermediate, (expert + 1) * 3 * intermediate)
    }
    require(covered == expected_groups, "selected expert group coverage")
    return candidate_source_matrices_subset(
        np,
        post,
        metadata,
        selected_experts,
        experts=experts,
        intermediate=intermediate,
    )


def identity_reconstruction(
    np: Any,
    coordinate: Mapping[str, Any],
    metadata: Mapping[str, Any],
    strata: Any,
) -> dict[tuple[int, str], Any]:
    values = {
        block.ordinal: np.asarray(block.reconstructed, dtype=np.float64)
        for block in coordinate["blocks"]
    }
    hidden = int(strata.GROUP_VALUES)
    intermediate = int(strata.GROUPS_PER_MATRIX)
    experts = int(coordinate["experts"])
    post = assemble_post(
        np,
        coordinate["blocks"],
        values,
        total_groups=experts * 3 * intermediate,
        hidden=hidden,
    )
    return candidate_source_matrices(
        np,
        post,
        metadata,
        experts=experts,
        intermediate=intermediate,
    )


def _inner_physical_vectors(result: Mapping[str, Any], experts: int) -> dict[str, Any]:
    source_final = result.get("source_final")
    require(isinstance(source_final, dict), "source-final record")
    metrics = source_final.get("parsed_metrics")
    require(isinstance(metrics, dict), "source-final physical metrics")
    rows = metrics.get("experts")
    require(isinstance(rows, list) and len(rows) == experts, "source-final expert ledger")
    rows = sorted(rows, key=lambda row: int(row["expert_ordinal"]))
    require([int(row["expert_ordinal"]) for row in rows] == list(range(experts)), "source-final expert order")
    return {
        "inner_bytes": int(metrics["actual_container_bytes"]),
        "attributed_total": tuple(_fraction_from_record(row["attributable_total_physical_bytes"], "inner total") for row in rows),
        "attributed_nonpadding": tuple(_fraction_from_record(row["attributable_nonpadding_decodable_bytes"], "inner nonpadding") for row in rows),
        "touched": tuple(int(row["touched_page_bytes"]) for row in rows),
        "requested": tuple(int(row["instrumented_routed_requested_bytes_with_repetition"]) for row in rows),
        "unique_requested": tuple(int(row["instrumented_routed_unique_requested_bytes"]) for row in rows),
        "requests": tuple(int(row["instrumented_routed_read_request_count"]) for row in rows),
        "ranges": tuple(
            tuple((int(item[0]), int(item[1])) for item in row["instrumented_routed_read_ranges"])
            for row in rows
        ),
        "causal": tuple(dict(row["causal_decode_reencode_reconstruction"]) for row in rows),
        "metrics": metrics,
    }


def _validate_actual_wrapper_read_proof(
    proof: Mapping[str, Any],
    inner_physical: Mapping[str, Any],
) -> None:
    require(proof.get("proof_uses_actual_authenticated_v8_routed_decoder") is True, "actual wrapper inner proof")
    require(proof.get("actual_inner_routed_decode_executed") is True, "inner routed decode executed")
    require(proof.get("actual_posterior_wrapper_routed_decode_executed") is False, "posterior routed claim boundary")
    require(proof.get("posterior_head_applied_to_routed_reconstruction") is False, "posterior routed application boundary")
    require(proof.get("compressed_expert_second_pass_forbidden_and_absent") is True, "proof second-pass absence")
    rows = proof.get("experts")
    require(isinstance(rows, list) and len(rows) == len(inner_physical["ranges"]), "proof expert rows")
    for expert, row in enumerate(rows):
        require(tuple(tuple(item) for item in row["inner_routed_read_ranges"]) == inner_physical["ranges"][expert], "proof/v9 routed ranges")
        require(int(row["inner_touched_page_bytes"]) == int(inner_physical["touched"][expert]), "proof/v9 touched pages")
        require(int(row["inner_requested_bytes_with_repetition"]) == int(inner_physical["requested"][expert]), "proof/v9 repeated requests")
        require(int(row["inner_unique_requested_bytes"]) == int(inner_physical["unique_requested"][expert]), "proof/v9 unique requests")
        require(int(row["inner_read_request_count"]) == int(inner_physical["requests"][expert]), "proof/v9 request count")
        require(dict(row["causal_decode_reencode_reconstruction"]) == inner_physical["causal"][expert], "proof/v9 causal reconstruction")


def _allocated_component_rate(
    attributed: Sequence[Fraction],
    weights_by_expert: Sequence[int],
    component: Sequence[int],
) -> float:
    bytes_value = sum((Fraction(attributed[expert]) for expert in component), Fraction(0, 1))
    weights = sum(int(weights_by_expert[expert]) for expert in component)
    require(weights > 0, "component weights")
    return float(Fraction(8, weights) * bytes_value)


def _roundtrip_parameters(np: Any, values: Any) -> Any:
    result = np.asarray(values, dtype=np.float64).astype("<f2").astype(np.float64)
    require(bool(np.all(np.isfinite(result))), "binary16 parameter roundtrip")
    return result


def _reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode), f"{label} symlink chain")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent


def _write_exclusive(
    output_dir: Path,
    members: Mapping[str, bytes],
    *,
    completion_payload: bytes,
) -> dict[str, Any]:
    require(output_dir.is_absolute(), "output directory absolute path")
    parent = output_dir.parent
    _reject_symlink_chain(parent, "output parent")
    require(stat.S_ISDIR(os.lstat(parent).st_mode), "output parent directory")
    require(not os.path.lexists(output_dir), "output directory must not pre-exist")
    require("COMPLETE.json" not in members, "completion must be published separately")
    require(isinstance(completion_payload, bytes) and completion_payload, "completion payload")
    os.mkdir(output_dir, 0o700)
    descriptor = os.open(
        os.fspath(output_dir),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    rows = []
    try:
        def write_member(name: str, payload: bytes) -> None:
            require(name and "/" not in name and "\\" not in name, "output member name")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            fd = os.open(name, flags, 0o600, dir_fd=descriptor)
            try:
                cursor = 0
                while cursor < len(payload):
                    written = os.write(fd, payload[cursor:])
                    require(written > 0, "short output write")
                    cursor += written
                os.fsync(fd)
            finally:
                os.close(fd)
            rows.append({"name": name, "bytes": len(payload), "sha256": sha256(payload)})
        for name in sorted(members):
            write_member(name, members[name])
        # Content is durable before the terminal completion seal appears.
        os.fsync(descriptor)
        write_member("COMPLETE.json", completion_payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(rows[-1]["name"] == "COMPLETE.json", "completion written last")
    return {
        "output_dir": os.fspath(output_dir),
        "members": rows,
        "write_order": [row["name"] for row in rows],
        "completion_written_last": True,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION, "explicit posterior diagnostic authorization")
    started = time.perf_counter()
    package_dir = Path(__file__).resolve().parent
    package_closure = authenticate_own_package(
        package_dir,
        expected_manifest_sha256=arguments.package_manifest_sha256,
    )
    require(
        package_closure["sources"]["diagnostic.py"]
        == _regular_bytes(package_dir / "diagnostic.py", maximum=4 * (1 << 20), label="running diagnostic source"),
        "running diagnostic is manifest-authenticated byte snapshot",
    )
    core = _load_authenticated_owned_module(
        package_closure,
        member_name="posterior_core.py",
        private_name="uwfa_pc_v1_authenticated_core",
    )
    bridge = _load_authenticated_owned_module(
        package_closure,
        member_name="result_bridge.py",
        private_name="uwfa_pc_v1_authenticated_bridge",
    )

    # The passing independent audit and separate launch review are both
    # authenticated before even statting the Qwen publication.  The literal
    # publication and decoder closure are then authenticated before importing
    # NumPy/CuPy or opening any original BF16 source matrix.
    launch = authenticate_launch_preconditions(arguments, package_closure)
    publication = bridge.authenticate_result_directory(Path(arguments.v9_result_dir))
    bind_publication_to_audit(publication, launch)
    result = publication["result"]
    expected_v8_manifest = require_digest(
        result["source_hashes"]["sealed_v8_manifest_sha256"], "result-bound v8 manifest"
    )
    v8 = bridge.authenticate_v8_package(
        Path(arguments.v8_package), expected_manifest_sha256=expected_v8_manifest
    )

    import numpy as np

    modules = bridge.load_authenticated_decoders(
        result,
        v8,
        strata_common_path=Path(arguments.strata_common),
        frozen_auditor_path=Path(arguments.frozen_auditor),
    )
    coordinate = bridge.decode_coordinate_panel(
        np,
        modules,
        publication["inner"],
        posterior_core=core,
        rht_device=arguments.rht_device,
    )
    require(coordinate["coordinate_aligned_observations_redecoded_from_literal"] is True, "coordinate-aligned decoder handoff")
    require(coordinate["selected_sc_decisions_treated_as_scalar_bins"] is False, "SC decision semantic boundary")

    strata = modules["strata"]
    hidden = int(strata.GROUP_VALUES)
    intermediate = int(strata.GROUPS_PER_MATRIX)
    experts = int(coordinate["experts"])
    metadata = coordinate["metadata"]
    decoder_observations = decoder_feature_observations(core, coordinate)
    identity = identity_reconstruction(np, coordinate, metadata, strata)
    all_experts = tuple(range(experts))
    components = core.owner_components(experts, [block.owners for block in coordinate["blocks"]])
    require(len(components) == 3, "expected three stream-owner connected components")
    weights_by_expert = tuple(3 * intermediate * hidden for _ in range(experts))

    def open_source(selected_experts: Sequence[int]) -> dict[str, Any]:
        return authenticate_source_panel(
            np,
            Path(arguments.source_manifest),
            expected_manifest_sha256=arguments.source_manifest_sha256,
            expected_artifact_sha256=result["artifact_identity"]["sha256"],
            experts=experts,
            intermediate=intermediate,
            hidden=hidden,
            selected_experts=selected_experts,
        )

    def source_observations(source_subset: Mapping[str, Any]) -> tuple[Any, ...]:
        post_target = source_post_coordinates(
            np,
            source_subset,
            metadata,
            experts=experts,
            intermediate=intermediate,
            hidden=hidden,
        )
        return build_observations(
            np,
            core,
            coordinate,
            post_target,
            modules["frozen"],
            rht_device=arguments.rht_device,
            selected_experts=source_subset["materialized_experts"],
        )

    def score_candidate_against(
        parameters: Any,
        law: int,
        component_ordinal: int,
        source_subset: Mapping[str, Any],
    ) -> dict[str, Any]:
        rounded = _roundtrip_parameters(np, parameters)
        reconstruction = reconstruct_from_parameters(
            np,
            core,
            coordinate,
            decoder_observations,
            rounded,
            law=law,
            frozen=modules["frozen"],
            strata=strata,
            metadata=metadata,
            rht_device=arguments.rht_device,
            experts_to_reconstruct=components[component_ordinal],
        )
        return score_matrices(
            np,
            source_subset,
            reconstruction,
            components[component_ordinal],
        )

    inner_physical = _inner_physical_vectors(result, experts)
    require(inner_physical["inner_bytes"] == len(publication["inner"]), "inner ledger/container bytes")
    baseline_rates = {
        ordinal: _allocated_component_rate(
            inner_physical["attributed_total"], weights_by_expert, component
        )
        for ordinal, component in enumerate(components)
    }
    fold_rows = []
    output_members: dict[str, bytes] = {}
    pooled = {
        core.LAW_LOCAL: {"sse": 0.0, "energy": 0.0},
        core.LAW_STATE: {"sse": 0.0, "energy": 0.0},
        core.LAW_STATE_PERMUTED: {"sse": 0.0, "energy": 0.0},
    }
    baseline_components: dict[int, dict[str, Any]] = {}
    source_manifest_sha256: str | None = None
    source_record_set_sha256: str | None = None
    states = int(coordinate["states"])
    # One source-free zero-head wrapper supplies the actual storage/read proof.
    # It performs one authenticated v8 routed inner decode per expert through
    # the literal wrapper reader.  The posterior head is parsed but is not
    # applied in that routed session; all read conclusions remain explicitly
    # nonpromoting projections until a routed posterior decoder exists.
    read_probe_head = core.serialize_head(
        np,
        np.zeros(core.parameter_count(core.LAW_STATE, states), dtype=np.float64),
        law=core.LAW_STATE,
        states=states,
        ridge_exponent=0,
        handoff_root_sha256=coordinate["handoff_root_sha256"],
    )
    read_probe_wrapper = core.build_wrapper(
        publication["inner"],
        read_probe_head,
        weights=int(coordinate["weights"]),
        experts=experts,
        fold_ordinal=-1,
        handoff_root_sha256=coordinate["handoff_root_sha256"],
    )
    actual_read_proof = bridge.instrument_inner_routed_decode_through_wrapper(
        np,
        core,
        modules,
        read_probe_wrapper,
        expected_handoff_root_sha256=coordinate["handoff_root_sha256"],
        rht_device=arguments.rht_device,
    )
    _validate_actual_wrapper_read_proof(actual_read_proof, inner_physical)
    for outer in range(3):
        development_ordinals = tuple(value for value in range(3) if value != outer)
        development_experts = tuple(
            sorted(
                expert
                for component_ordinal in development_ordinals
                for expert in components[component_ordinal]
            )
        )
        # Only development BF16 leaves are materialized before every head for
        # this outer fold has been selected, refit and serialized.
        development_source = open_source(development_experts)
        development_observations = source_observations(development_source)
        source_manifest_sha256 = development_source["manifest_sha256"]
        source_record_set_sha256 = development_source["source_record_set_sha256"]
        pending_laws: dict[int, dict[str, Any]] = {}
        for law in (core.LAW_LOCAL, core.LAW_STATE, core.LAW_STATE_PERMUTED):
            selected = core.select_ridge_for_outer(
                np,
                development_observations,
                components,
                outer_component=outer,
                law=law,
                states=states,
                score_sse=lambda parameters, selected_law, component_ordinal: score_candidate_against(
                    parameters,
                    selected_law,
                    component_ordinal,
                    development_source,
                )["sse_fp64"],
            )
            head = core.serialize_head(
                np,
                selected["refit_parameters"],
                law=law,
                states=states,
                ridge_exponent=selected["selected_ridge_exponent"],
                handoff_root_sha256=coordinate["handoff_root_sha256"],
            )
            parsed_head = core.parse_head(
                np,
                head,
                expected_handoff_root_sha256=coordinate["handoff_root_sha256"],
            )
            wrapper = core.build_wrapper(
                publication["inner"],
                head,
                weights=int(coordinate["weights"]),
                experts=experts,
                fold_ordinal=outer,
                handoff_root_sha256=coordinate["handoff_root_sha256"],
            )
            parsed_wrapper = core.parse_wrapper(
                np,
                wrapper,
                expected_handoff_root_sha256=coordinate["handoff_root_sha256"],
            )
            routed_wrapper_trace = bridge.bind_wrapper_to_routed_proof(
                np,
                core,
                wrapper,
                actual_read_proof,
                expected_handoff_root_sha256=coordinate["handoff_root_sha256"],
            )
            ledger = core.wrapper_read_ledger(
                routed_wrapper_trace=routed_wrapper_trace,
                weights_by_expert=weights_by_expert,
                inner_attributed_total=inner_physical["attributed_total"],
                inner_attributed_nonpadding=inner_physical["attributed_nonpadding"],
                head_bytes=len(head),
            )
            pending_laws[law] = {
                "selected": selected,
                "head": head,
                "parsed_head": parsed_head,
                "wrapper": wrapper,
                "parsed_wrapper": parsed_wrapper,
                "ledger": ledger,
            }

        # This is the numerical aperture: no heldout BF16 array exists until
        # all three operational binary16 heads are immutable above.
        del development_observations
        del development_source
        heldout_source = open_source(components[outer])
        require(
            heldout_source["materialized_experts"] == list(components[outer]),
            "outer heldout source aperture",
        )
        baseline_score = score_matrices(
            np,
            heldout_source,
            identity,
            components[outer],
        )
        baseline_components[outer] = baseline_score
        laws = {}
        for law in (core.LAW_LOCAL, core.LAW_STATE, core.LAW_STATE_PERMUTED):
            pending = pending_laws[law]
            selected = pending["selected"]
            parsed_head = pending["parsed_head"]
            score = score_candidate_against(
                parsed_head["parameters"], law, outer, heldout_source
            )
            ledger = pending["ledger"]
            outer_attributed = tuple(
                _fraction_from_record(row["attributable_total_physical_bytes"], "outer total")
                for row in ledger["experts"]
            )
            rate = _allocated_component_rate(
                outer_attributed, weights_by_expert, components[outer]
            )
            ds = core.delta_s(
                baseline_rate=baseline_rates[outer],
                candidate_rate=rate,
                baseline_distortion=baseline_score["relative_mse"],
                candidate_distortion=score["relative_mse"],
            )
            f_value = score["relative_mse"] * math.pow(2.0, 2.0 * rate)
            name = f"FOLD{outer}_{core.LAW_NAMES[law].upper().replace('-', '_')}.cagepst1"
            output_members[name] = pending["wrapper"]
            selected_public = {key: value for key, value in selected.items() if key != "refit_parameters"}
            laws[core.LAW_NAMES[law]] = {
                "nested_selection": selected_public,
                "head": {key: value for key, value in parsed_head.items() if key != "parameters"},
                "wrapper_bytes": len(pending["wrapper"]),
                "wrapper_sha256": pending["parsed_wrapper"]["wrapper_sha256"],
                "owner_allocated_rate_bpw": rate,
                "score": score,
                "F_from_owner_allocated_rate": f_value,
                "Delta_s_from_owner_ledger": ds,
                "physical_ledger": ledger,
                "owner_ledger_is_literal_heldout_packet": False,
            }
            pooled[law]["sse"] += score["sse_fp64"]
            pooled[law]["energy"] += score["source_energy_fp64"]
        del heldout_source
        state_ds = laws["state-aware"]["Delta_s_from_owner_ledger"]
        comparator = max(
            laws["local-only"]["Delta_s_from_owner_ledger"],
            laws["state-permuted"]["Delta_s_from_owner_ledger"],
        )
        g_state = state_ds - comparator
        state_row = laws["state-aware"]
        fold_gate = core.state_fold_gate(
            delta_s_value=state_ds,
            g_state_value=g_state,
            candidate_rate_bpw=float(state_row["owner_allocated_rate_bpw"]),
            candidate_f=float(state_row["F_from_owner_allocated_rate"]),
            cold_read_below_2x=bool(
                state_row["physical_ledger"]["passes_strict_cold_read_below_2x"]
            ),
        )
        fold_rows.append({
            "outer_component": outer,
            "heldout_experts": list(components[outer]),
            "baseline": {
                "owner_allocated_rate_bpw": baseline_rates[outer],
                "score": baseline_score,
            },
            "laws": laws,
            "G_state_bpw": g_state,
            **fold_gate,
            "passes_positive_state_specific_gate": fold_gate["passes_all_fold_gates"],
            "source_access_order": {
                "development_experts_materialized_before_fit": list(development_experts),
                "all_three_heads_serialized_before_heldout_open": True,
                "heldout_experts_materialized_after_head_serialization": list(components[outer]),
                "unselected_source_leaves_not_opened_by_each_aperture": True,
            },
        })

    crossfit_pass = all(row["passes_positive_state_specific_gate"] for row in fold_rows)
    pooled_rows = {}
    for law in pooled:
        pooled_rows[core.LAW_NAMES[law]] = {
            "heldout_sse_sum_fp64": pooled[law]["sse"],
            "heldout_energy_sum_fp64": pooled[law]["energy"],
            "pooled_relative_mse": pooled[law]["sse"] / pooled[law]["energy"],
        }
    baseline_sse = sum(row["sse_fp64"] for row in baseline_components.values())
    baseline_energy = sum(row["source_energy_fp64"] for row in baseline_components.values())
    baseline_full = {
        "experts": list(all_experts),
        "matrices": [
            matrix
            for component_ordinal in range(3)
            for matrix in baseline_components[component_ordinal]["matrices"]
        ],
        "sse_fp64": baseline_sse,
        "source_energy_fp64": baseline_energy,
        "relative_mse": baseline_sse / baseline_energy,
        "assembled_from_three_disjoint_heldout_apertures": True,
    }
    require(
        math.isclose(
            baseline_energy,
            float(result["baseline_score"]["source_energy_fp64"]),
            rel_tol=2.0 ** -45,
            abs_tol=1e-10,
        ),
        "original-source energy/result binding",
    )
    require(
        math.isclose(
            baseline_full["relative_mse"],
            float(result["physical"]["relative_mse"]),
            rel_tol=2.0 ** -43,
            abs_tol=1e-13,
        ),
        "identity original-MSE/result binding",
    )

    final_record = None
    if crossfit_pass:
        # Hyperparameter selection for the final packet uses only the already
        # emitted outer-fold selections.  Median exponent is deterministic and
        # avoids picking the best post-hoc full-panel score.
        exponents = sorted(
            int(row["laws"]["state-aware"]["nested_selection"]["selected_ridge_exponent"])
            for row in fold_rows
        )
        final_exponent = exponents[1]
        final_source = open_source(all_experts)
        final_observations = source_observations(final_source)
        final_parameters = core.fit_head(
            np,
            final_observations,
            law=core.LAW_STATE,
            states=states,
            ridge_exponent=final_exponent,
        )
        final_head = core.serialize_head(
            np,
            final_parameters,
            law=core.LAW_STATE,
            states=states,
            ridge_exponent=final_exponent,
            handoff_root_sha256=coordinate["handoff_root_sha256"],
        )
        parsed_final_head = core.parse_head(
            np,
            final_head,
            expected_handoff_root_sha256=coordinate["handoff_root_sha256"],
        )
        final_reconstruction = reconstruct_from_parameters(
            np,
            core,
            coordinate,
            decoder_observations,
            parsed_final_head["parameters"],
            law=core.LAW_STATE,
            frozen=modules["frozen"],
            strata=strata,
            metadata=metadata,
            rht_device=arguments.rht_device,
        )
        final_score = score_matrices(
            np, final_source, final_reconstruction, all_experts
        )
        final_wrapper = core.build_wrapper(
            publication["inner"],
            final_head,
            weights=int(coordinate["weights"]),
            experts=experts,
            fold_ordinal=-1,
            handoff_root_sha256=coordinate["handoff_root_sha256"],
        )
        parsed_final = core.parse_wrapper(
            np,
            final_wrapper,
            expected_handoff_root_sha256=coordinate["handoff_root_sha256"],
        )
        final_routed_wrapper_trace = bridge.bind_wrapper_to_routed_proof(
            np,
            core,
            final_wrapper,
            actual_read_proof,
            expected_handoff_root_sha256=coordinate["handoff_root_sha256"],
        )
        final_ledger = core.wrapper_read_ledger(
            routed_wrapper_trace=final_routed_wrapper_trace,
            weights_by_expert=weights_by_expert,
            inner_attributed_total=inner_physical["attributed_total"],
            inner_attributed_nonpadding=inner_physical["attributed_nonpadding"],
            head_bytes=len(final_head),
        )
        final_rate = float(final_ledger["physical_rate_bpw"]["float"])
        final_f = final_score["relative_mse"] * math.pow(2.0, 2.0 * final_rate)
        final_record = {
            "selected_ridge_exponent_by_outer_median": final_exponent,
            "head": {key: value for key, value in parsed_final_head.items() if key != "parameters"},
            "literal_wrapper_bytes": len(final_wrapper),
            "literal_wrapper_sha256": parsed_final["wrapper_sha256"],
            "physical_rate_bpw": final_rate,
            "score": final_score,
            "F": final_f,
            "physical_ledger": final_ledger,
            "passes_rate_interval": 2.15 <= final_rate <= 2.5,
            "passes_F_target": final_f <= 0.8,
            "passes_cold_read_below_2x": final_ledger["passes_strict_cold_read_below_2x"],
            "training_panel_result_not_portability_evidence": True,
        }
        output_members["FINAL_STATE_AWARE.cagepst1"] = final_wrapper
        del final_observations
        del final_source

    final_physical_pass = bool(
        final_record is not None
        and final_record["passes_rate_interval"]
        and final_record["passes_F_target"]
        and final_record["passes_cold_read_below_2x"]
    )
    overall_survivor = crossfit_pass and final_physical_pass
    require(source_manifest_sha256 is not None, "source manifest fold binding")
    require(source_record_set_sha256 is not None, "source record-set fold binding")
    result_record = {
        "schema": RESULT_SCHEMA,
        "status": (
            "CROSS_FIT_SOURCE_SURVIVOR_NONPROMOTING_CONTROLS_AND_PORTABILITY_REQUIRED"
            if overall_survivor
            else (
                "HARD_KILL_FINAL_LITERAL_RATE_F_OR_COLD_READ"
                if crossfit_pass
                else "HARD_KILL_POSTERIOR_STATE_SPECIFIC_CROSS_FIT"
            )
        ),
        "positive_claim_authority": False,
        "selected_sc_decisions_treated_as_scalar_bins": False,
        "coordinate_aligned_lattice_indices_redecoded_from_literal": True,
        "predecessor_publication_sha256": publication["publication_sha256"],
        "predecessor_inner_sha256": sha256(publication["inner"]),
        "posterior_handoff_root_sha256": coordinate["handoff_root_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_record_set_sha256": source_record_set_sha256,
        "source_identity_fields_available_to_decoder": False,
        "decoder_source_hashes": modules["source_hashes"],
        "ordinal_bridge": modules["ordinal_bridge"],
        "launch_authority": {
            "predecessor_audit_receipt_sha256": launch["predecessor_audit_receipt_sha256"],
            "launch_review_sha256": launch["launch_review_sha256"],
            "audit_and_review_authenticated_before_qwen_publication_access": True,
        },
        "routed_read_proof": {
            "proof_sha256": actual_read_proof["proof_sha256"],
            "probe_wrapper_sha256": actual_read_proof["wrapper_sha256"],
            "actual_inner_routed_decode_executed": actual_read_proof["actual_inner_routed_decode_executed"],
            "actual_posterior_wrapper_routed_decode_executed": actual_read_proof["actual_posterior_wrapper_routed_decode_executed"],
            "posterior_head_applied_to_routed_reconstruction": actual_read_proof["posterior_head_applied_to_routed_reconstruction"],
            "compressed_expert_second_pass_forbidden_and_absent": actual_read_proof["compressed_expert_second_pass_forbidden_and_absent"],
            "nonpromoting_inference_read_projection_only": True,
        },
        "posterior_package_closure": {
            key: value for key, value in package_closure.items() if key != "sources"
        },
        "rht_device": arguments.rht_device,
        "components": [list(component) for component in components],
        "baseline_full": baseline_full,
        "folds": fold_rows,
        "pooled_crossfit": pooled_rows,
        "pooled_crossfit_is_one_literal_packet": False,
        "crossfit_passes_state_specific_gate_on_every_component": crossfit_pass,
        "final_literal_passes_rate_F_and_cold_read": final_physical_pass,
        "overall_nonpromoting_source_survivor": overall_survivor,
        "final_all_component_candidate": final_record,
        "controls_run": False,
        "matched_gaussian_controls_run": False,
        "structure_destroying_controls_run": False,
        "portability_family_run": False,
        "claim_boundary": "Qwen-panel discovery diagnostic only; no universal SwiGLU-MoE performance claim",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_members["RESULT.json"] = pretty_json(result_record)
    completion = {
        "schema": "uwfa-sc-posterior-centroid-completion-v1",
        "status": result_record["status"],
        "positive_claim_authority": False,
        "members": [
            {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
            for name, payload in sorted(output_members.items())
        ],
    }
    completion["completion_sha256"] = sha256(canonical_json(completion))
    completion_payload = pretty_json(completion)
    publication_record = _write_exclusive(
        Path(arguments.output_dir),
        output_members,
        completion_payload=completion_payload,
    )
    return {
        "schema": "uwfa-sc-posterior-centroid-launch-summary-v1",
        "status": result_record["status"],
        "positive_claim_authority": False,
        "output_dir": publication_record["output_dir"],
        "result_sha256": sha256(output_members["RESULT.json"]),
        "crossfit_pass": crossfit_pass,
        "overall_survivor": overall_survivor,
        "final_F": None if final_record is None else final_record["F"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--package-manifest-sha256", required=True)
    result.add_argument("--predecessor-audit-receipt", type=Path, required=True)
    result.add_argument("--predecessor-audit-receipt-sha256", required=True)
    result.add_argument("--launch-review", type=Path, required=True)
    result.add_argument("--launch-review-sha256", required=True)
    result.add_argument("--v9-result-dir", type=Path, required=True)
    result.add_argument("--v8-package", type=Path, required=True)
    result.add_argument("--strata-common", type=Path, required=True)
    result.add_argument("--frozen-auditor", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--rht-device", choices=("cupy", "numpy"), default="cupy")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    try:
        summary = run(parser().parse_args())
        print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Exception as error:
        print(f"posterior-centroid-v1 failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
