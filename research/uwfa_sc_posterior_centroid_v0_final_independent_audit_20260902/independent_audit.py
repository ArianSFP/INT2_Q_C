#!/usr/bin/env python3
"""Clean-room, source-only audit of the frozen posterior-centroid v0 package.

This audit never opens a model payload, a completed v9 publication, a BF16
score source, a Gaussian control, or CUDA.  It authenticates the exact frozen
source closure, compiles only authenticated retained bytes, and uses synthetic
objects for grammar, cross-fit, read-ledger, and publication checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import types
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_MANIFEST_SHA256 = "0ef30253d4d31504fbd8f88b8203cf35bce6c14952e570aace44b7bc089cb713"
EXPECTED_SOURCE_ROOT_SHA256 = "ea3ad9cf9b723cdf7501eeff004bd7f2821af4d37ff186b72f2972482a05e11c"
EXPECTED_MEMBERS = {
    "README.md",
    "design_lock.json",
    "diagnostic.py",
    "posterior_core.py",
    "result_bridge.py",
    "test_source_only.py",
}

AUDIT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = AUDIT_DIR.parent
PACKAGE = (RESEARCH_DIR / "uwfa_sc_posterior_centroid_v0").resolve()
REPOSITORY = RESEARCH_DIR.parent


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def regular_bytes(path: Path, *, maximum: int = 1 << 24) -> bytes:
    metadata = os.lstat(path)
    check(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"regular file: {path}")
    check(0 < metadata.st_size <= maximum, f"bounded file: {path}")
    payload = path.read_bytes()
    check(len(payload) == metadata.st_size, f"stable length: {path}")
    return payload


def authenticate_package() -> dict[str, Any]:
    manifest_payload = regular_bytes(PACKAGE / "SOURCE_MANIFEST.json")
    check(digest(manifest_payload) == EXPECTED_MANIFEST_SHA256, "manifest SHA-256")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    check(manifest.get("schema") == "uwfa-sc-posterior-centroid-source-manifest-v0", "manifest schema")
    check(manifest.get("status") == "SEALED_SOURCE_ONLY_NONPROMOTING_NO_PAYLOAD_AUTHORITY", "manifest status")
    rows = manifest.get("members")
    check(isinstance(rows, list) and len(rows) == len(EXPECTED_MEMBERS), "manifest member count")
    check({row.get("name") for row in rows} == EXPECTED_MEMBERS, "manifest exact member names")
    observed = []
    sources: dict[str, bytes] = {}
    for row in rows:
        check(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest member grammar")
        name = row["name"]
        payload = regular_bytes(PACKAGE / name)
        check(len(payload) == row["bytes"], f"manifest bytes: {name}")
        check(digest(payload) == row["sha256"], f"manifest digest: {name}")
        observed.append({"name": name, "bytes": len(payload), "sha256": digest(payload)})
        sources[name] = payload
    observed.sort(key=lambda row: row["name"].encode("utf-8"))
    root = digest(canonical_json(observed))
    check(root == EXPECTED_SOURCE_ROOT_SHA256 == manifest.get("source_snapshot_root_sha256"), "source snapshot root")
    return {
        "manifest": manifest,
        "manifest_payload": manifest_payload,
        "members": observed,
        "sources": sources,
        "source_root": root,
    }


def load_retained(name: str, payload: bytes, expected_digest: str) -> Any:
    check(digest(payload) == expected_digest, f"retained source binding: {name}")
    check(name not in sys.modules, f"fresh audit module name: {name}")
    module = types.ModuleType(name)
    module.__file__ = f"<audit-retained:{name}:{expected_digest}>"
    module.__package__ = ""
    module.__authenticated_sha256__ = expected_digest
    code = compile(payload, module.__file__, "exec", dont_inherit=True, optimize=0)
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class DirectoryFDCompatibilityProxy:
    """Exercise POSIX directory-FD control flow on a Windows audit host.

    The producer remains untouched.  Only the audit module's `os` global is
    temporarily replaced.  Ordinary member files still use real exclusive
    descriptors, writes, fsyncs, and closes.  The directory descriptor itself
    is represented by a sentinel because Windows refuses O_RDONLY directory
    opens; directory fsync is recorded but cannot be executed on this host.
    """

    TOKEN = 0x6F735044

    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def open(self, path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is None and Path(os.fspath(path)) == self.output:
            self.events.append(("open-directory", os.fspath(path)))
            return self.TOKEN
        if dir_fd == self.TOKEN:
            member = str(path)
            self.events.append(("open-member", member))
            return os.open(os.fspath(self.output / member), flags, mode)
        return os.open(path, flags, mode, dir_fd=dir_fd)

    def fsync(self, descriptor: int) -> None:
        if descriptor == self.TOKEN:
            self.events.append(("fsync-directory", "."))
            return
        self.events.append(("fsync-file", str(descriptor)))
        os.fsync(descriptor)

    def close(self, descriptor: int) -> None:
        if descriptor == self.TOKEN:
            self.events.append(("close-directory", "."))
            return
        os.close(descriptor)


def publication_order_check(diagnostic: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary).resolve() / "result"
        proxy = DirectoryFDCompatibilityProxy(output)
        original = diagnostic.os
        diagnostic.os = proxy
        try:
            record = diagnostic._write_exclusive(
                output,
                {"RESULT.json": b"result\n", "Z.bin": b"z"},
                completion_payload=b"complete\n",
            )
            rejected_existing = False
            try:
                diagnostic._write_exclusive(
                    output,
                    {"RESULT.json": b"again\n"},
                    completion_payload=b"again-complete\n",
                )
            except diagnostic.DiagnosticError:
                rejected_existing = True
        finally:
            diagnostic.os = original
        member_opens = [value for event, value in proxy.events if event == "open-member"]
        directory_sync_positions = [index for index, event in enumerate(proxy.events) if event[0] == "fsync-directory"]
        completion_position = proxy.events.index(("open-member", "COMPLETE.json"))
        check(record["write_order"][-1] == "COMPLETE.json", "completion write record last")
        check(member_opens[-1] == "COMPLETE.json", "completion descriptor opened last")
        check(len(directory_sync_positions) == 2, "two directory durability barriers")
        check(directory_sync_positions[0] < completion_position < directory_sync_positions[1], "completion between durability barriers")
        check((output / "COMPLETE.json").read_bytes() == b"complete\n", "completion bytes")
        check(rejected_existing, "existing publication rejection")
        return {
            "compatibility_harness_only": os.name == "nt",
            "write_order": record["write_order"],
            "directory_fsync_count": len(directory_sync_positions),
            "completion_between_directory_fsyncs": True,
            "existing_directory_rejected": True,
        }


def synthetic_killed_publication_check(bridge: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        inner = bytes((index * 19) & 255 for index in range(8192))
        members = {
            "BOUND_BASELINE_SCORE.json": b"{}\n",
            "DECODER_BUNDLE.json": b"{}\n",
            "IDENTITY_FRAMING.bin": b"identity",
            "SOURCE_PREFLIGHT.json": b"{}\n",
            "UWFCV8.bin": inner,
        }
        result = {
            "schema": bridge.RESULT_SCHEMA,
            "status": "HARD_KILL_SYNTHETIC_BUT_LITERAL_CONTAINER_COMPLETE",
            "positive_claim_authority": False,
            "controls_run": False,
            "physical": {"container_sha256": digest(inner), "container_bytes": len(inner)},
            "source_final": {"container_sha256": digest(inner)},
        }
        members["RESULT.json"] = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii")
        for name, payload in members.items():
            (root / name).write_bytes(payload)
        completion = {
            "schema": bridge.COMPLETE_SCHEMA,
            "status": result["status"],
            "positive_claim_authority": False,
            "members": [
                {"name": name, "bytes": len(payload), "sha256": digest(payload)}
                for name, payload in sorted(members.items())
            ],
        }
        completion["completion_sha256"] = digest(bridge.canonical_json(completion))
        (root / "COMPLETE.json").write_bytes((json.dumps(completion, sort_keys=True) + "\n").encode("ascii"))
        authenticated = bridge.authenticate_result_directory(root)
        check(authenticated["inner"] == inner, "killed completed container accepted")
        (root / "UWFCV8.bin").write_bytes(inner + b"x")
        tamper_rejected = False
        try:
            bridge.authenticate_result_directory(root)
        except bridge.BridgeError:
            tamper_rejected = True
        check(tamper_rejected, "completed publication tamper rejected")
        return {
            "completed_hard_kill_container_is_accepted": True,
            "standalone_v9_survival_is_not_required": True,
            "member_tamper_rejected": True,
        }


def wrapper_and_read_checks(core: Any) -> dict[str, Any]:
    states = 64
    parameters = np.linspace(-0.125, 0.125, core.parameter_count(core.LAW_STATE, states), dtype=np.float64)
    handoff = digest(b"audit-handoff")
    head = core.serialize_head(
        np,
        parameters,
        law=core.LAW_STATE,
        states=states,
        ridge_exponent=-20,
        handoff_root_sha256=handoff,
    )
    check(len(head) == 1636, "maximum head byte count")
    parsed_head = core.parse_head(np, head, expected_handoff_root_sha256=handoff)
    check(parsed_head["canonical_reencode_matches"] is True, "head canonical roundtrip")
    inner = bytes((index * 7) & 255 for index in range(8192))
    wrapper = core.build_wrapper(
        inner,
        head,
        weights=24576,
        experts=2,
        fold_ordinal=-1,
        handoff_root_sha256=handoff,
    )
    parsed_wrapper = core.parse_wrapper(np, wrapper, expected_handoff_root_sha256=handoff)
    check(parsed_wrapper["inner"] == inner and parsed_wrapper["total_bytes"] == len(inner) + 4096, "wrapper roundtrip")
    damaged = bytearray(wrapper)
    damaged[len(inner) + len(head) + 1] ^= 1
    tamper_rejected = False
    try:
        core.parse_wrapper(np, bytes(damaged), expected_handoff_root_sha256=handoff)
    except core.PosteriorContractError:
        tamper_rejected = True
    check(tamper_rejected, "wrapper padding tamper rejection")

    def trace(*, invocations: int = 1, requested: int = 7200) -> dict[str, Any]:
        rows = []
        for expert in range(2):
            rows.append({
                "expert_ordinal": expert,
                "extension_page_read_requests": 1,
                "inner_decode_invocations": invocations,
                "compressed_expert_second_pass": invocations > 1,
                "compressed_expert_second_pass_absent_derived": invocations == 1,
                "overlap_is_charged_not_interpreted_as_second_pass": True,
                "touched_page_bytes": 8192,
                "requested_bytes_with_repetition": requested,
                "unique_requested_bytes": min(7000, requested),
                "read_request_count": 5,
                "overlap_bytes_requested_again": max(0, requested - min(7000, requested)),
                "causal_decode_reencode_reconstruction": {
                    "all_payloads_canonically_reencoded": True,
                    "all_three_roles_reconstructed": True,
                },
            })
        return {
            "schema": "uwfa-sc-posterior-wrapper-routed-read-proof-v0",
            "proof_uses_actual_authenticated_v8_routed_decoder": True,
            "compressed_expert_second_pass_forbidden_and_absent": invocations == 1,
            "inner_bytes": 8192,
            "head_bytes": len(head),
            "proof_sha256": digest(b"audit-read-proof"),
            "experts": rows,
        }

    ledger = core.wrapper_read_ledger(
        routed_wrapper_trace=trace(),
        weights_by_expert=(12288, 12288),
        inner_attributed_total=(Fraction(4096), Fraction(4096)),
        inner_attributed_nonpadding=(Fraction(4000), Fraction(4000)),
        head_bytes=len(head),
    )
    check(ledger["actual_inner_routed_decode_executed"] is True, "inner decode boundary")
    check(ledger["actual_posterior_wrapper_routed_decode_executed"] is False, "posterior routed boundary")
    check(ledger["posterior_head_applied_to_routed_reconstruction"] is False, "posterior application boundary")
    check(ledger["read_claim_is_nonpromoting_projection_from_instrumented_inner_decode_plus_literal_suffix"] is True, "read projection boundary")
    second_pass_rejected = False
    try:
        core.wrapper_read_ledger(
            routed_wrapper_trace=trace(invocations=2, requested=16000),
            weights_by_expert=(12288, 12288),
            inner_attributed_total=(Fraction(4096), Fraction(4096)),
            inner_attributed_nonpadding=(Fraction(4000), Fraction(4000)),
            head_bytes=len(head),
        )
    except core.PosteriorContractError:
        second_pass_rejected = True
    check(second_pass_rejected, "second compressed pass rejected")
    excessive = core.wrapper_read_ledger(
        routed_wrapper_trace=trace(requested=20000),
        weights_by_expert=(12288, 12288),
        inner_attributed_total=(Fraction(4096), Fraction(4096)),
        inner_attributed_nonpadding=(Fraction(4000), Fraction(4000)),
        head_bytes=len(head),
    )
    check(excessive["passes_strict_cold_read_below_2x"] is False, "repeated request over-2x rejection")
    return {
        "maximum_head_bytes": len(head),
        "extension_page_bytes": ledger["extension_physical_bytes"],
        "head_roundtrip": True,
        "wrapper_roundtrip": True,
        "tamper_rejected": True,
        "second_pass_rejected": True,
        "over_2x_rejected": True,
        "actual_posterior_wrapper_routed_decode_executed": False,
        "read_claim_is_nonpromoting_projection": True,
    }


def crossfit_check(core: Any) -> dict[str, Any]:
    states = 2
    blocks = []
    for component in range(3):
        indices = np.tile(np.arange(64, dtype=np.int16), 2)
        occupancy = np.zeros((6, states), dtype=np.float64)
        occupancy[:, 0] = 0.25 - 0.05 * component
        occupancy[:, 1] = -occupancy[:, 0]
        q = 0.25 * (indices.astype(np.float64) - 31.0)
        target = q + (0.01 * (component + 1)) + 0.002 * q
        blocks.append(core.BlockObservation(
            ordinal=component,
            owners=(component,),
            indices=indices,
            target_normalized=target,
            occupancy=occupancy,
            coordinate_mapping_sha256=digest(f"map-{component}".encode("ascii")),
        ))
    components = ((0,), (1,), (2,))
    fit_owner_sets: list[tuple[int, ...]] = []
    original_fit = core.fit_head

    def logged_fit(np_module: Any, selected_blocks: Any, **kwargs: Any) -> Any:
        owners = tuple(sorted({owner for block in selected_blocks for owner in block.owners}))
        fit_owner_sets.append(owners)
        return original_fit(np_module, selected_blocks, **kwargs)

    def score_sse(parameters: Any, law: int, validation_component: int) -> float:
        check(validation_component in (1, 2), "outer component excluded from inner validation")
        block = blocks[validation_component]
        prediction = core.predict_normalized(np, block, parameters, law=law, states=states)
        residual = np.asarray(block.target_normalized) - prediction
        return float(np.sum(residual * residual, dtype=np.float64))

    core.fit_head = logged_fit
    try:
        selected = core.select_ridge_for_outer(
            np,
            tuple(blocks),
            components,
            outer_component=0,
            law=core.LAW_LOCAL,
            states=states,
            score_sse=score_sse,
        )
    finally:
        core.fit_head = original_fit
    check(all(0 not in owners for owners in fit_owner_sets), "outer component absent from every fit")
    check(fit_owner_sets[-1] == (1, 2), "refit uses exactly two development components")
    for row in selected["ridge_grid"]:
        directions = row["directions"]
        check(
            [(entry["train_component"], entry["validation_component"]) for entry in directions]
            == [(1, 2), (2, 1)],
            "bidirectional inner split",
        )
    return {
        "outer_component": 0,
        "outer_absent_from_all_fit_calls": True,
        "inner_directions": [[1, 2], [2, 1]],
        "final_refit_components": [1, 2],
        "ridge_candidates": len(selected["ridge_grid"]),
    }


def retained_loader_and_mutation_checks(closure: dict[str, Any], diagnostic: Any, bridge: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        copy = Path(temporary).resolve() / "package"
        copy.mkdir()
        for name in EXPECTED_MEMBERS | {"SOURCE_MANIFEST.json"}:
            shutil.copyfile(PACKAGE / name, copy / name)
        authenticated = diagnostic.authenticate_own_package(
            copy,
            expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        )
        (copy / "posterior_core.py").write_bytes((copy / "posterior_core.py").read_bytes() + b"\n# hostile mutation\n")
        mutation_rejected = False
        try:
            diagnostic.authenticate_own_package(
                copy,
                expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
            )
        except diagnostic.DiagnosticError:
            mutation_rejected = True
        check(mutation_rejected, "post-freeze source mutation rejection")
        retained = diagnostic._load_authenticated_owned_module(
            authenticated,
            member_name="posterior_core.py",
            private_name="uwfa_pc_final_audit_retained_core",
        )
        check(
            retained.__authenticated_sha256__
            == next(row["sha256"] for row in authenticated["members"] if row["name"] == "posterior_core.py"),
            "retained module digest",
        )

    v8_path = (RESEARCH_DIR / "unifilar_wfa_entropy_census_stage0_v8").resolve()
    v8_manifest_payload = regular_bytes(v8_path / "SOURCE_MANIFEST.json")
    v8 = bridge.authenticate_v8_package(v8_path, expected_manifest_sha256=digest(v8_manifest_payload))
    strata_path = (REPOSITORY / "strata_expert_local_codec" / "common.py").resolve()
    frozen_path = (REPOSITORY / "strata_v2_klt_mixed_independent_auditor_v1.py").resolve()
    result_record = {
        "source_hashes": {
            "sealed_v8_manifest_sha256": digest(v8_manifest_payload),
            "strata_expert_local_codec_common_sha256": digest(regular_bytes(strata_path)),
            "strata_v2_klt_mixed_independent_auditor_sha256": digest(regular_bytes(frozen_path)),
        },
    }
    loaded = bridge.load_authenticated_decoders(
        result_record,
        v8,
        strata_common_path=strata_path,
        frozen_auditor_path=frozen_path,
    )
    check(hasattr(loaded["common"], "Candidate"), "authenticated v8 dataclass source")
    check(hasattr(loaded["codec"], "OwnerContribution"), "authenticated codec dataclass source")
    check(sys.modules.get(loaded["common"].__name__) is loaded["common"], "dataclass module registered")
    return {
        "manifest_member_mutation_rejected": True,
        "owned_module_executed_from_retained_authenticated_bytes": True,
        "v8_dataclass_sources_registered_before_exec": True,
        "live_sibling_import_after_authentication": False,
    }


def declared_boundary_checks(closure: dict[str, Any]) -> dict[str, Any]:
    design = json.loads(closure["sources"]["design_lock.json"].decode("utf-8"))
    readme = closure["sources"]["README.md"].decode("utf-8")
    diagnostic_source = closure["sources"]["diagnostic.py"].decode("utf-8")
    check(design["semantic_correction"]["selected_sc_decisions_are_scalar_bins"] is False, "decision semantics")
    check(design["physical_accounting"]["actual_posterior_wrapper_routed_decode_executed"] is False, "routed posterior boundary")
    check(design["physical_accounting"]["read_result_is_nonpromoting_projection"] is True, "read projection design")
    check(design["physical_accounting"]["inference_read_promotion_blocked_until_routed_posterior_decoder"] is True, "inference promotion block")
    check(design["universality"]["portability_required_before_universal_performance_claim"] is True, "portability requirement")
    check("There is deliberately **no inference-ready routed posterior decoder in v0**" in readme, "README inference boundary")
    for forbidden in ("Qwen/Qwen3", "model.layers.5", "ad44e777", "fe4fd2b8438d"):
        check(forbidden not in diagnostic_source, f"no frozen identity in runner: {forbidden}")
    serialize_position = diagnostic_source.index("head = core.serialize_head(", diagnostic_source.index("for outer in range(3):"))
    heldout_open_position = diagnostic_source.index("heldout_source = open_source(components[outer])")
    check(serialize_position < heldout_open_position, "heads serialized before heldout aperture")
    check("actual_posterior_wrapper_routed_decode_executed\": actual_read_proof" in diagnostic_source, "result preserves routed boundary")
    check("\"positive_claim_authority\": False" in diagnostic_source, "result/completion nonpromoting")
    check("\"matched_gaussian_controls_run\": False" in diagnostic_source, "controls not claimed")
    check("\"portability_family_run\": False" in diagnostic_source, "portability not claimed")
    return {
        "heldout_BF16_open_occurs_after_all_three_fold_heads_are_serialized": True,
        "selected_SC_decisions_not_scalar_bins": True,
        "decoder_identity_features_forbidden": design["universality"]["forbidden_decoder_features"],
        "portability_required_before_universal_claim": True,
        "inference_ready_routed_posterior_decoder_present": False,
        "read_result_nonpromoting": True,
        "controls_and_portability_claimed": False,
    }


def exact_producer_self_test() -> dict[str, Any]:
    command = [sys.executable, "-I", "-B", os.fspath(PACKAGE / "test_source_only.py")]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    combined = completed.stdout + completed.stderr
    summary = None
    for line in completed.stdout.splitlines()[::-1]:
        if line.startswith("{"):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if candidate.get("schema") == "uwfa-sc-posterior-centroid-source-test-v0":
                summary = candidate
                break
    check(summary is not None and summary.get("tests_run") == 22, "producer test summary")
    windows_posix_limitation = (
        os.name == "nt"
        and completed.returncode == 1
        and summary.get("errors") == 1
        and summary.get("failures") == 0
        and "test_new_publication_writes_completion_last_and_rejects_existing" in combined
        and "PermissionError" in combined
        and "descriptor = os.open" in combined
    )
    check(completed.returncode == 0 or windows_posix_limitation, "producer source-only test outcome")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "summary": summary,
        "windows_directory_fd_limitation_only": windows_posix_limitation,
        "producer_suite_passed_unmodified": completed.returncode == 0,
        "completion_logic_independently_exercised_by_compatibility_harness": windows_posix_limitation,
    }


def main() -> int:
    closure = authenticate_package()
    member_hashes = {row["name"]: row["sha256"] for row in closure["members"]}
    core = load_retained(
        "uwfa_pc_final_audit_core",
        closure["sources"]["posterior_core.py"],
        member_hashes["posterior_core.py"],
    )
    bridge = load_retained(
        "uwfa_pc_final_audit_bridge",
        closure["sources"]["result_bridge.py"],
        member_hashes["result_bridge.py"],
    )
    diagnostic = load_retained(
        "uwfa_pc_final_audit_diagnostic",
        closure["sources"]["diagnostic.py"],
        member_hashes["diagnostic.py"],
    )
    checks = {
        "manifest": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_snapshot_root_sha256": closure["source_root"],
            "members": closure["members"],
        },
        "retained_loader": retained_loader_and_mutation_checks(closure, diagnostic, bridge),
        "serialization_and_read": wrapper_and_read_checks(core),
        "crossfit": crossfit_check(core),
        "publication": publication_order_check(diagnostic),
        "predecessor_scope": synthetic_killed_publication_check(bridge),
        "declared_boundaries": declared_boundary_checks(closure),
        "producer_self_test": exact_producer_self_test(),
    }
    result = {
        "schema": "uwfa-sc-posterior-centroid-v0-final-independent-audit-v0",
        "verdict": "PASS",
        "scope": "SOURCE_ONLY_NONPROMOTING_NO_QWEN_NO_PAYLOAD_NO_CUDA",
        "producer_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "producer_source_snapshot_root_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "checks": checks,
        "payload_or_Qwen_accessed": False,
        "completed_v9_result_accessed": False,
        "CUDA_initialized": False,
        "claim_boundary": "PASS authenticates source mechanics only; it is not Qwen evidence, an inference-read promotion, a Gaussian-control result, or universal SwiGLU-MoE performance evidence.",
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
