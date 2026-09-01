"""Independent fail-closed verifier for the sealed Tier-B procedural-anchor gate.

This process never imports PyTorch or CuPy.  The default verification binds the
published result, source-free calibration, recovery evidence, frozen protocol,
and package code byte-for-byte.  Supplying ``--output-dir`` additionally
rehashes the append-only journal, validates every saved cascade array, rebuilds
the stage-0 merge and stage-1 winners, and checks the preserved interrupted
orphan.  Supplying ``--aux-dir`` hashes the 31 eligible auxiliary payloads; the
excluded payload is deliberately never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import common


PUBLISHED_RESULT_SHA256 = "e450c10767b54c190f901df8460c6ac57fe86cfaca7719c3db48475d4196fb92"
PUBLISHED_CALIBRATION_SHA256 = "92710a3c73533512f38f05b6010f4f59f307d308bced053411c51c8f4fbd1b23"
PUBLISHED_RECOVERY_SHA256 = "075cdaa3feeb518f0a42336c946cbfe29cc2a4fee5cc84ca39b40395bb6f95b1"
RUNNER_SHA256 = "83eb7682c8185d8f27dbd4b7d39de96cb54dad1c887a4cf026ac4ea759159665"
COMMON_SHA256 = "75d8bbd7af9271ea5d2f099e7d720c1560bcc72864ac88f458095647468e7da3"
KERNELS_SHA256 = "b563b977251dd754f1d6ed7dfe08a486ae4ed6aab3ff60b1e9f9399be804a195"
LAST_EVENT_SHA256 = "2a1632c5735a1616af4fc6aab4997a1c46072af401c6422968d7385003dcf16a"
EXCLUSION_BINDING_KEYS = (
    "packaged_intersection_lock_sha256",
    "source_exclusion_manifest_sha256",
    "full_external_manifest_revalidated_at_runtime",
    "excluded_tensor_identities",
)
IMMUTABLE_EXCLUSION_BINDING_KEYS = (
    "packaged_intersection_lock_sha256",
    "source_exclusion_manifest_sha256",
    "excluded_tensor_identities",
)


def _reject_constant(value: str) -> None:
    raise common.ProtocolError(f"non-finite JSON numeric literal: {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise common.ProtocolError(f"{label} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise common.ProtocolError(f"{label} must contain one JSON object")
    return value


def _close(
    left: float,
    right: float,
    label: str,
    *,
    rtol: float = 2e-9,
    atol: float = 2e-11,
) -> None:
    if not (math.isfinite(float(left)) and math.isfinite(float(right))):
        raise common.ProtocolError(f"{label} is non-finite")
    if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
        raise common.ProtocolError(f"{label} mismatch: {left!r} != {right!r}")


def _equal_tree(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise common.ProtocolError(f"{label} is not an object")
        common.strict_keys(observed, expected.keys(), label)
        for key in expected:
            _equal_tree(observed[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise common.ProtocolError(f"{label} list shape mismatch")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _equal_tree(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise common.ProtocolError(f"{label} mismatch")
        return
    if isinstance(expected, int):
        if isinstance(observed, bool) or int(observed) != expected or float(observed) != expected:
            raise common.ProtocolError(f"{label} integer mismatch")
        return
    if isinstance(expected, float):
        _close(float(observed), expected, label)
        return
    raise common.ProtocolError(f"unsupported verifier tree type at {label}")


def _verify_exclusion_binding(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind immutable exclusion facts while reporting runtime revalidation separately.

    The production result records whether the *external* full exclusion manifest
    happened to be present on that machine.  Presence is environmental, not a
    scientific binding: a verifier with the same immutable packaged lock may
    legitimately observe the opposite value.  ``common.exclusion_binding`` has
    already fail-closed revalidated the external manifest whenever it is present.
    """

    common.strict_keys(recorded, EXCLUSION_BINDING_KEYS, "firewall.exclusion_binding")
    common.strict_keys(current, EXCLUSION_BINDING_KEYS, "current exclusion binding")
    for key in IMMUTABLE_EXCLUSION_BINDING_KEYS:
        _equal_tree(recorded[key], current[key], f"firewall.exclusion_binding.{key}")
    recorded_runtime = recorded["full_external_manifest_revalidated_at_runtime"]
    current_runtime = current["full_external_manifest_revalidated_at_runtime"]
    if not isinstance(recorded_runtime, bool) or not isinstance(current_runtime, bool):
        raise common.ProtocolError("exclusion-manifest runtime status must be boolean")
    return {
        "immutable_binding_verified": True,
        "recorded_full_external_manifest_revalidated_at_runtime": recorded_runtime,
        "current_full_external_manifest_revalidated_at_runtime": current_runtime,
        "current_external_manifest_if_present_revalidated_fail_closed": True,
    }


def _hex64(value: Any, label: str) -> str:
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise common.ProtocolError(f"{label} is not lowercase SHA-256")
    return value


def _canonical_event_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _policy_increment(numel: int, sm_count: int, max_threads_per_sm: int) -> int:
    grid = min((numel + 255) // 256, sm_count * (max_threads_per_sm // 256))
    if grid <= 0:
        raise common.ProtocolError("invalid parity launch geometry")
    return ((numel - 1) // (256 * grid * 4) + 1) * 4


def _verify_parity(parity: Mapping[str, Any], label: str) -> None:
    common.strict_keys(
        parity,
        (
            "all_required_checks_passed",
            "torch_version",
            "cupy_version",
            "device_name",
            "device_index",
            "multi_processor_count",
            "max_threads_per_multi_processor",
            "descriptor_checks",
            "candidate_coordinate_checks",
            "persistent_packing_checks",
            "dlpack_sha256_f32le",
        ),
        label,
    )
    if parity["all_required_checks_passed"] is not True or int(parity["device_index"]) != 0:
        raise common.ProtocolError(f"{label} did not pass on device zero")
    sm_count = int(parity["multi_processor_count"])
    max_threads = int(parity["max_threads_per_multi_processor"])
    if sm_count <= 0 or max_threads < 256 or max_threads % 256:
        raise common.ProtocolError(f"{label} has invalid device geometry")
    if not str(parity["torch_version"]) or not str(parity["cupy_version"]) or not str(parity["device_name"]):
        raise common.ProtocolError(f"{label} has empty implementation identity")

    shapes = (
        (768, 2048),
        (2048, 768),
        (1536, 2048),
        (384, 2048),
        (2048, 384),
        (768, 2048),
        (192, 2048),
        (2048, 192),
        (384, 2048),
    )
    offsets = (0, 4, 8192, 1_048_576, 4_294_967_300)
    descriptor_rows = parity["descriptor_checks"]
    if not isinstance(descriptor_rows, list) or len(descriptor_rows) != 45:
        raise common.ProtocolError(f"{label} descriptor-case count mismatch")
    cursor = 0
    for shape in shapes:
        numel = math.prod(shape)
        stride = 256 * min((numel + 255) // 256, sm_count * (max_threads // 256))
        native_count = len(
            {0, 1, min(numel - 1, stride - 1), min(numel - 1, stride), min(numel - 1, 4 * stride), numel - 1}
        )
        for offset in offsets:
            row = descriptor_rows[cursor]
            cursor += 1
            common.strict_keys(
                row,
                ("shape", "offset", "coordinate_count", "increment", "float32_sha256", "bf16_widened_sha256"),
                f"{label}.descriptor[{cursor-1}]",
            )
            if list(row["shape"]) != list(shape) or int(row["offset"]) != offset:
                raise common.ProtocolError(f"{label} descriptor order mismatch")
            if int(row["coordinate_count"]) != native_count:
                raise common.ProtocolError(f"{label} descriptor coordinate count mismatch")
            if int(row["increment"]) != _policy_increment(numel, sm_count, max_threads):
                raise common.ProtocolError(f"{label} descriptor increment mismatch")
            _hex64(row["float32_sha256"], f"{label}.descriptor.float32")
            _hex64(row["bf16_widened_sha256"], f"{label}.descriptor.bf16")

    expected_candidates = []
    for pp_index in (0, 2, 3):
        for ep_index, assignment_index in ((0, 0), (3, 0), (3, 1), (7, 0), (7, 1)):
            for etp_index in range(3):
                for packing_index in range(3):
                    ordinal = common.logical_ordinal(
                        3407, pp_index, ep_index, etp_index, assignment_index, packing_index
                    )
                    expected_candidates.append(common.decode_ordinal(ordinal).id)
    candidate_rows = parity["candidate_coordinate_checks"]
    if not isinstance(candidate_rows, list) or len(candidate_rows) != 810:
        raise common.ProtocolError(f"{label} candidate-coordinate count mismatch")
    cursor = 0
    for candidate_id in expected_candidates:
        for expert in (0, 57, 127):
            for role in ("up", "down"):
                row = candidate_rows[cursor]
                cursor += 1
                common.strict_keys(
                    row,
                    ("candidate", "expert", "role", "coordinate_count", "scaled_sha256"),
                    f"{label}.candidate_coordinate[{cursor-1}]",
                )
                if (
                    row["candidate"] != candidate_id
                    or int(row["expert"]) != expert
                    or row["role"] != role
                    or int(row["coordinate_count"]) != 9
                ):
                    raise common.ProtocolError(f"{label} candidate-coordinate order mismatch")
                _hex64(row["scaled_sha256"], f"{label}.candidate_coordinate.scaled")

    packing_rows = parity["persistent_packing_checks"]
    if not isinstance(packing_rows, list) or len(packing_rows) != 9:
        raise common.ProtocolError(f"{label} persistent packing count mismatch")
    cursor = 0
    for etp in common.ETP_SIZES:
        n = (common.ROWS // etp) * common.COLUMNS
        inc_n = _policy_increment(n, sm_count, max_threads)
        for packing in common.PACKINGS:
            row = packing_rows[cursor]
            cursor += 1
            common.strict_keys(row, ("packing", "etp", "offsets"), f"{label}.packing[{cursor-1}]")
            numels = (n, n, n) if packing == "separate_gate_up_down" else (2 * n, n)
            expected_offsets = [0]
            for numel in numels:
                expected_offsets.append(expected_offsets[-1] + _policy_increment(numel, sm_count, max_threads))
            if row["packing"] != packing or int(row["etp"]) != etp or list(row["offsets"]) != expected_offsets:
                raise common.ProtocolError(f"{label} persistent packing mismatch")
    expected_dlpack = hashlib.sha256(np.arange(257, dtype="<f4").tobytes()).hexdigest()
    if parity["dlpack_sha256_f32le"] != expected_dlpack:
        raise common.ProtocolError(f"{label} DLPack hash mismatch")


def _verify_calibration(calibration: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    common.strict_keys(
        calibration,
        (
            "schema",
            "status",
            "source_manifest_or_payload_opened",
            "candidate_lock_file_sha256",
            "candidate_lock_internal_sha256",
            "runner_sha256",
            "common_sha256",
            "kernels_sha256",
            "parity",
            "candidate_count",
            "coordinate_count",
            "values_per_repetition",
            "elapsed_seconds",
            "values_per_second",
            "median_values_per_second",
            "estimated_stage0_seconds_at_median_kernel_rate",
            "output_sentinel_sha256_f32le",
            "working_output_bytes",
            "logical_candidate_count",
            "effective_candidate_count",
            "equivalence_map_sha256",
        ),
        "calibration",
    )
    if (
        calibration["schema"] != "qwen3_initialization_anchor_tier_b_source_free_calibration_v1"
        or calibration["status"] != "PASS_SOURCE_FREE_CALIBRATION"
        or calibration["source_manifest_or_payload_opened"] is not False
    ):
        raise common.ProtocolError("calibration status/source-free boundary mismatch")
    expected_scalars = {
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "common_sha256": COMMON_SHA256,
        "kernels_sha256": KERNELS_SHA256,
        "candidate_count": int(lock["source_free_calibration"]["candidate_count"]),
        "coordinate_count": int(lock["source_free_calibration"]["coordinate_count"]),
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "equivalence_map_sha256": common.equivalence_map_sha256(),
    }
    for key, expected in expected_scalars.items():
        if calibration[key] != expected:
            raise common.ProtocolError(f"calibration binding mismatch: {key}")
    generated = int(calibration["candidate_count"]) * int(calibration["coordinate_count"])
    if int(calibration["values_per_repetition"]) != generated:
        raise common.ProtocolError("calibration generated-value accounting mismatch")
    if int(calibration["working_output_bytes"]) != generated * 4:
        raise common.ProtocolError("calibration working-memory accounting mismatch")
    elapsed = calibration["elapsed_seconds"]
    rates = calibration["values_per_second"]
    repetitions = int(lock["source_free_calibration"]["repetitions"])
    if not isinstance(elapsed, list) or not isinstance(rates, list) or len(elapsed) != repetitions or len(rates) != repetitions:
        raise common.ProtocolError("calibration repetition count mismatch")
    for index, (seconds, rate) in enumerate(zip(elapsed, rates)):
        if float(seconds) <= 0.0:
            raise common.ProtocolError("calibration elapsed time is non-positive")
        _close(rate, generated / float(seconds), f"calibration.rate[{index}]", rtol=2e-12, atol=1e-3)
    median_rate = statistics.median(float(value) for value in rates)
    _close(calibration["median_values_per_second"], median_rate, "calibration.median_rate", rtol=2e-12, atol=1e-3)
    _close(
        calibration["estimated_stage0_seconds_at_median_kernel_rate"],
        int(lock["search_cascade"]["stage0"]["maximum_generated_normal_values"]) / median_rate,
        "calibration.stage0_estimate",
    )
    _hex64(calibration["output_sentinel_sha256_f32le"], "calibration.output_sentinel")
    _verify_parity(calibration["parity"], "calibration.parity")


def _verify_detail_row(row: Mapping[str, Any], plan: common.PlanRow, label: str) -> None:
    common.strict_keys(row, ("tensor_name", "expert", "role", "fit", "score"), label)
    source = plan.source
    if (
        row["tensor_name"] != source.tensor_name
        or int(row["expert"]) != source.expert
        or row["role"] != source.role
    ):
        raise common.ProtocolError(f"{label} identity mismatch")
    fit = row["fit"]
    score = row["score"]
    fit_keys = ("n", "sum_w", "sum_g", "sum_w2", "sum_g2", "sum_wg", "alpha", "mu", "fit_mean_w")
    score_keys = ("n", "sum_w", "sum_g", "sum_w2", "sum_g2", "sum_wg", "sse", "baseline_sse", "rho")
    common.strict_keys(fit, fit_keys, f"{label}.fit")
    common.strict_keys(score, score_keys, f"{label}.score")
    n_fit = int(fit["n"])
    n_score = int(score["n"])
    if n_fit != len(plan.fit) or n_score != len(plan.score):
        raise common.ProtocolError(f"{label} coordinate counts mismatch")
    for section_name, section in (("fit", fit), ("score", score)):
        for key, value in section.items():
            if key != "n" and not math.isfinite(float(value)):
                raise common.ProtocolError(f"{label}.{section_name}.{key} is non-finite")

    sum_w = float(fit["sum_w"])
    sum_g = float(fit["sum_g"])
    cgg = float(fit["sum_g2"]) - sum_g * sum_g / n_fit
    cwg = float(fit["sum_wg"]) - sum_w * sum_g / n_fit
    alpha = cwg / cgg if cgg > 0.0 else 0.0
    mean_w = sum_w / n_fit
    mu = mean_w - alpha * sum_g / n_fit
    _close(fit["alpha"], alpha, f"{label}.fit.alpha", rtol=5e-10, atol=5e-12)
    _close(fit["fit_mean_w"], mean_w, f"{label}.fit.fit_mean_w")
    _close(fit["mu"], mu, f"{label}.fit.mu")

    sw = float(score["sum_w"])
    sg = float(score["sum_g"])
    sw2 = float(score["sum_w2"])
    sg2 = float(score["sum_g2"])
    swg = float(score["sum_wg"])
    expected_sse = (
        sw2
        + n_score * mu * mu
        + alpha * alpha * sg2
        + 2.0 * mu * alpha * sg
        - 2.0 * mu * sw
        - 2.0 * alpha * swg
    )
    expected_baseline = sw2 - 2.0 * mean_w * sw + n_score * mean_w * mean_w
    _close(score["sse"], expected_sse, f"{label}.score.sse", rtol=2e-8, atol=2e-10)
    _close(score["baseline_sse"], expected_baseline, f"{label}.score.baseline_sse", rtol=2e-8, atol=2e-10)
    cww = sw2 - sw * sw / n_score
    cgg_score = sg2 - sg * sg / n_score
    cwg_score = swg - sw * sg / n_score
    rho = cwg_score / math.sqrt(cww * cgg_score) if cww > 0.0 and cgg_score > 0.0 else 0.0
    _close(score["rho"], rho, f"{label}.score.rho", rtol=2e-8, atol=2e-10)


def _verify_domain_details_and_folds(
    details: Mapping[str, Any],
    folds: Mapping[str, Any],
    plans: Sequence[common.PlanRow],
    label: str,
) -> None:
    common.strict_keys(details, common.DOMAIN_IDS, f"{label}.details")
    common.strict_keys(folds, common.DOMAIN_IDS, f"{label}.folds")
    for domain_id in common.DOMAIN_IDS:
        rows = details[domain_id]
        if not isinstance(rows, list) or len(rows) != len(plans):
            raise common.ProtocolError(f"{label}.{domain_id} detail row count mismatch")
        for index, (row, plan) in enumerate(zip(rows, plans)):
            _verify_detail_row(row, plan, f"{label}.{domain_id}[{index}]")
        expected_folds = common.fold_statistics(rows)
        _equal_tree(folds[domain_id], expected_folds, f"{label}.folds.{domain_id}")


def _verify_embedded_events(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    resume = result["resume_state"]
    common.strict_keys(
        resume,
        ("run_header_sha256", "winner_freeze_sha256", "event_count_before_result", "events"),
        "resume_state",
    )
    events = resume["events"]
    if not isinstance(events, list) or len(events) != 392 or int(resume["event_count_before_result"]) != len(events):
        raise common.ProtocolError("resume event count mismatch")
    previous = "0" * 64
    for sequence, event in enumerate(events):
        common.strict_keys(
            event,
            (
                "sequence",
                "previous_event_sha256",
                "kind",
                "key",
                "relative_path",
                "file_sha256",
                "file_bytes",
                "created_unix_ns",
            ),
            f"resume.event[{sequence}]",
        )
        if int(event["sequence"]) != sequence or event["previous_event_sha256"] != previous:
            raise common.ProtocolError("embedded journal sequence/hash-chain mismatch")
        if int(event["file_bytes"]) <= 0 or int(event["created_unix_ns"]) <= 0:
            raise common.ProtocolError("embedded journal size/time is invalid")
        _hex64(event["file_sha256"], "embedded event target hash")
        if sequence == 0:
            expected_identity = ("run_header", "immutable", "files/run_header_immutable.json")
        elif sequence <= 256:
            shard = sequence - 1
            expected_identity = ("stage0", f"{shard:03d}", f"files/stage0_{shard:03d}.npz")
        elif sequence == 257:
            expected_identity = ("stage0_merged", "global", "files/stage0_merged_global.npz")
        elif sequence <= 389:
            batch = sequence - 258
            expected_identity = ("stage1", f"{batch:04d}", f"files/stage1_{batch:04d}.npz")
        elif sequence == 390:
            expected_identity = ("stage1_winners", "global", "files/stage1_winners_global.npz")
        else:
            expected_identity = (
                "validation_firewall",
                "winners_frozen",
                "files/validation_firewall_winners_frozen.json",
            )
        if (event["kind"], event["key"], event["relative_path"]) != expected_identity:
            raise common.ProtocolError(f"embedded journal event identity mismatch at {sequence}")
        previous = hashlib.sha256(_canonical_event_bytes(event)).hexdigest()
    if previous != LAST_EVENT_SHA256:
        raise common.ProtocolError("published final event hash mismatch")
    if resume["run_header_sha256"] != events[0]["file_sha256"]:
        raise common.ProtocolError("run-header target hash binding mismatch")
    if resume["winner_freeze_sha256"] != events[-1]["file_sha256"]:
        raise common.ProtocolError("winner-freeze target hash binding mismatch")
    return events, previous


def _verify_recovery_evidence(
    recovery: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    common.strict_keys(
        recovery,
        (
            "schema",
            "action",
            "old_absolute_path",
            "new_absolute_path",
            "file_bytes",
            "file_sha256",
            "zip_integrity",
            "state_reference_count",
            "last_journal_event",
            "post_move_journal_validation",
            "reason",
        ),
        "recovery",
    )
    if (
        recovery["schema"] != "qwen3_initialization_anchor_tier_b_interruption_recovery_v1"
        or recovery["action"] != "move_unjournaled_partial_without_content_change"
        or int(recovery["file_bytes"]) != 192_751
        or recovery["file_sha256"] != "7bb36fd08dc69647b50af44b801c2be1d4578da49c9e864cbf591c748016cd04"
        or int(recovery["state_reference_count"]) != 0
    ):
        raise common.ProtocolError("recovery identity/size/hash mismatch")
    _equal_tree(
        recovery["zip_integrity"],
        {"status": "BadZipFile", "message": "File is not a zip file", "end_of_central_directory_present": False},
        "recovery.zip_integrity",
    )
    last = events[-1]
    expected_last = {
        "absolute_path": "/workspace/INT2__compression/tier_b_initialization_anchor_run_v1/state/events/000136.json",
        "event_sha256": "16db813027de03d0b6f7f4d748e051cd2a2b4c9e9062dceb116fa3a919c4e8e9",
        "sequence": 136,
        "kind": "stage0",
        "key": "135",
        "target_relative_path": "files/stage0_135.npz",
        "target_file_bytes": 486_952,
        "target_file_sha256": "c67c2a1b71546ed3424466b83c544c8103220284b419b29d09740be17bb23a1d",
    }
    _equal_tree(recovery["last_journal_event"], expected_last, "recovery.last_journal_event")
    if events[136]["key"] != "135" or events[136]["file_sha256"] != expected_last["target_file_sha256"]:
        raise common.ProtocolError("recovery does not bind embedded event 136")
    _equal_tree(
        recovery["post_move_journal_validation"],
        {"event_count": 137, "hash_chain_and_all_referenced_targets_valid": True, "next_resumable_stage0_key": "136"},
        "recovery.post_move_journal_validation",
    )
    references = sum(event["relative_path"] == "files/stage0_136.npz" for event in events[:137])
    if references != 0 or not str(recovery["reason"]):
        raise common.ProtocolError("recovery reference/reason mismatch")


def _load_npz(path: Path, expected_keys: Iterable[str], label: str) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise common.ProtocolError(f"{label} is missing/non-regular")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_keys):
                raise common.ProtocolError(f"{label} array keys mismatch")
            return {key: archive[key] for key in archive.files}
    except common.ProtocolError:
        raise
    except Exception as error:
        raise common.ProtocolError(f"invalid {label}: {error}") from error


def _verify_state_arrays(
    output_dir: Path,
    result: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise common.ProtocolError("output directory must be a regular directory")
    output_result = output_dir / "tier_b_initialization_anchor_result.json"
    if common.sha256_file(output_result) != PUBLISHED_RESULT_SHA256:
        raise common.ProtocolError("output result differs from published result")
    state = output_dir / "state"
    event_dir = state / "events"
    file_dir = state / "files"
    if any(not path.is_dir() or path.is_symlink() for path in (state, event_dir, file_dir)):
        raise common.ProtocolError("state journal directories are invalid")

    expected_event_paths = set()
    expected_target_paths = set()
    previous = "0" * 64
    for sequence, embedded in enumerate(events):
        event_path = event_dir / f"{sequence:06d}.json"
        if not event_path.is_file() or event_path.is_symlink():
            raise common.ProtocolError(f"journal event missing/non-regular: {sequence}")
        raw = event_path.read_bytes()
        observed = json.loads(raw.decode(), parse_constant=_reject_constant)
        if observed != embedded or embedded["previous_event_sha256"] != previous:
            raise common.ProtocolError(f"journal event content/hash-chain mismatch: {sequence}")
        previous = hashlib.sha256(raw).hexdigest()
        target = state / str(embedded["relative_path"])
        if not target.is_file() or target.is_symlink():
            raise common.ProtocolError(f"journal target missing/non-regular: {sequence}")
        if target.stat().st_size != int(embedded["file_bytes"]) or common.sha256_file(target) != embedded["file_sha256"]:
            raise common.ProtocolError(f"journal target size/hash mismatch: {sequence}")
        expected_event_paths.add(event_path.resolve())
        expected_target_paths.add(target.resolve())
    if previous != LAST_EVENT_SHA256:
        raise common.ProtocolError("on-disk final event hash mismatch")
    observed_event_paths = {path.resolve() for path in event_dir.iterdir() if path.is_file()}
    observed_target_paths = {path.resolve() for path in file_dir.iterdir() if path.is_file()}
    if observed_event_paths != expected_event_paths or observed_target_paths != expected_target_paths:
        raise common.ProtocolError("unreferenced/missing journal files")
    if any(path.is_symlink() for path in event_dir.iterdir()) or any(path.is_symlink() for path in file_dir.iterdir()):
        raise common.ProtocolError("journal contains a symlink")

    shard_ordinals = np.empty((256, len(common.DOMAIN_IDS), common.STAGE0_TOP_K), dtype=np.uint64)
    shard_q = np.empty((256, len(common.DOMAIN_IDS), common.STAGE0_TOP_K), dtype=np.float64)
    identity = np.arange(common.STAGE0_TOP_K)
    for shard in range(256):
        path = state / events[shard + 1]["relative_path"]
        arrays = _load_npz(path, ("seed_start", "seed_stop", "top_ordinals", "top_q"), f"stage0[{shard}]")
        if (
            arrays["seed_start"].shape != (1,)
            or arrays["seed_start"].dtype != np.int32
            or arrays["seed_stop"].shape != (1,)
            or arrays["seed_stop"].dtype != np.int32
            or int(arrays["seed_start"][0]) != shard * common.SEED_SHARD_SIZE
            or int(arrays["seed_stop"][0]) != (shard + 1) * common.SEED_SHARD_SIZE
        ):
            raise common.ProtocolError(f"stage0 seed range mismatch: {shard}")
        ordinals = arrays["top_ordinals"]
        q = arrays["top_q"]
        if (
            ordinals.shape != (len(common.DOMAIN_IDS), common.STAGE0_TOP_K)
            or ordinals.dtype != np.uint64
            or q.shape != ordinals.shape
            or q.dtype != np.float64
            or not np.all(np.isfinite(q))
        ):
            raise common.ProtocolError(f"stage0 array shape/dtype/finite mismatch: {shard}")
        base_seeds = ordinals // common.LAYOUTS_PER_SEED
        pp_indices = (ordinals // 144) % 4
        if (
            np.any(ordinals >= common.LOGICAL_CANDIDATES)
            or np.any(base_seeds < shard * common.SEED_SHARD_SIZE)
            or np.any(base_seeds >= (shard + 1) * common.SEED_SHARD_SIZE)
            or not np.all(np.isin(pp_indices, (0, 2, 3)))
        ):
            raise common.ProtocolError(f"stage0 representative/range mismatch: {shard}")
        for domain_index in range(len(common.DOMAIN_IDS)):
            if len(np.unique(ordinals[domain_index])) != common.STAGE0_TOP_K:
                raise common.ProtocolError(f"stage0 duplicate ordinal: {shard}/{domain_index}")
            if not np.array_equal(np.lexsort((ordinals[domain_index], q[domain_index])), identity):
                raise common.ProtocolError(f"stage0 top-k tie/order mismatch: {shard}/{domain_index}")
        shard_ordinals[shard] = ordinals
        shard_q[shard] = q

    merged = _load_npz(
        state / events[257]["relative_path"],
        ("domain_top_ordinals", "domain_top_q", "union_ordinals"),
        "stage0_merged",
    )
    merged_ordinals = merged["domain_top_ordinals"]
    merged_q = merged["domain_top_q"]
    union = merged["union_ordinals"]
    if (
        merged_ordinals.shape != (len(common.DOMAIN_IDS), common.STAGE0_TOP_K)
        or merged_ordinals.dtype != np.uint64
        or merged_q.shape != merged_ordinals.shape
        or merged_q.dtype != np.float64
        or union.ndim != 1
        or union.dtype != np.uint64
        or not np.all(np.isfinite(merged_q))
    ):
        raise common.ProtocolError("stage0 merged shape/dtype/finite mismatch")
    for domain_index in range(len(common.DOMAIN_IDS)):
        ordinals = shard_ordinals[:, domain_index, :].reshape(-1)
        q = shard_q[:, domain_index, :].reshape(-1)
        order = np.lexsort((ordinals, q))[: common.STAGE0_TOP_K]
        if not np.array_equal(merged_ordinals[domain_index], ordinals[order]) or not np.array_equal(
            merged_q[domain_index], q[order]
        ):
            raise common.ProtocolError(f"stage0 merge mismatch: {domain_index}")
    expected_union = np.unique(merged_ordinals.reshape(-1))
    if not np.array_equal(union, expected_union) or len(union) != int(result["search"]["union_shortlist_count"]):
        raise common.ProtocolError("stage0 union mismatch")

    batch_count = (len(union) + 511) // 512
    if batch_count != 132:
        raise common.ProtocolError("stage1 batch count differs from frozen result")
    all_ordinals = []
    all_q = []
    for batch in range(batch_count):
        arrays = _load_npz(
            state / events[258 + batch]["relative_path"],
            ("ordinals", "q"),
            f"stage1[{batch}]",
        )
        start = batch * 512
        stop = min(len(union), start + 512)
        if (
            arrays["ordinals"].dtype != np.uint64
            or not np.array_equal(arrays["ordinals"], union[start:stop])
            or arrays["q"].shape != (stop - start, len(common.DOMAIN_IDS))
            or arrays["q"].dtype != np.float64
            or not np.all(np.isfinite(arrays["q"]))
        ):
            raise common.ProtocolError(f"stage1 array mismatch: {batch}")
        all_ordinals.append(arrays["ordinals"])
        all_q.append(arrays["q"])
    stage1_ordinals = np.concatenate(all_ordinals)
    stage1_q = np.concatenate(all_q, axis=0)
    winners = _load_npz(
        state / events[390]["relative_path"],
        ("winner_ordinals", "winner_q"),
        "stage1_winners",
    )
    winner_ordinals = winners["winner_ordinals"]
    winner_q = winners["winner_q"]
    if (
        winner_ordinals.shape != (len(common.DOMAIN_IDS),)
        or winner_ordinals.dtype != np.uint64
        or winner_q.shape != winner_ordinals.shape
        or winner_q.dtype != np.float64
        or not np.all(np.isfinite(winner_q))
    ):
        raise common.ProtocolError("stage1 winner array mismatch")
    for domain_index, domain_id in enumerate(common.DOMAIN_IDS):
        order = np.lexsort((stage1_ordinals, stage1_q[:, domain_index]))
        local = int(order[0])
        if int(winner_ordinals[domain_index]) != int(stage1_ordinals[local]) or float(winner_q[domain_index]) != float(
            stage1_q[local, domain_index]
        ):
            raise common.ProtocolError(f"stage1 exact winner mismatch: {domain_id}")
        record = result["search"]["stage1_winners"][domain_id]
        if int(record["candidate"]["ordinal"]) != int(winner_ordinals[domain_index]):
            raise common.ProtocolError(f"result/state winner ordinal mismatch: {domain_id}")
        _close(record["selection_q"], winner_q[domain_index], f"result/state winner q {domain_id}", rtol=0.0, atol=0.0)

    freeze_path = state / events[391]["relative_path"]
    freeze = _load_json(freeze_path, "winner freeze")
    expected_freeze = {
        "schema": "qwen3_initialization_anchor_tier_b_winner_freeze_v1",
        "domain_count": len(common.DOMAIN_IDS),
        "domain_ids": list(common.DOMAIN_IDS),
        "winners": result["search"]["stage1_winners"],
        "union_shortlist_count": len(union),
        "validation_payload_opened": False,
    }
    _equal_tree(freeze, expected_freeze, "winner_freeze")

    packaged_recovery = common.PACKAGE_DIR / "recovery_orphan_stage0_136.json"
    output_recovery = output_dir / "recovery_orphan_stage0_136.json"
    if common.sha256_file(output_recovery) != common.sha256_file(packaged_recovery):
        raise common.ProtocolError("output/package recovery evidence mismatch")
    resumed_path = output_dir / "state" / "files" / "stage0_136.npz"
    orphan_path = output_dir / "interrupted_orphans" / Path(str(recovery["new_absolute_path"])).name
    # The historical partial was moved out of state.  Resume then legitimately
    # recreated the same state target name as the complete, event-bound shard
    # 136; distinguish the two by the final journal binding and orphan hash.
    if (
        not resumed_path.is_file()
        or resumed_path.is_symlink()
        or resumed_path.resolve() != (state / str(events[137]["relative_path"])).resolve()
        or events[137]["kind"] != "stage0"
        or events[137]["key"] != "136"
        or not orphan_path.is_file()
        or orphan_path.is_symlink()
    ):
        raise common.ProtocolError("interrupted orphan placement mismatch")
    if orphan_path.stat().st_size != int(recovery["file_bytes"]) or common.sha256_file(orphan_path) != recovery["file_sha256"]:
        raise common.ProtocolError("interrupted orphan size/hash mismatch")
    try:
        with zipfile.ZipFile(orphan_path, "r") as archive:
            archive.testzip()
    except zipfile.BadZipFile as error:
        if str(error) != recovery["zip_integrity"]["message"]:
            raise common.ProtocolError("interrupted orphan BadZipFile message mismatch") from error
    else:
        raise common.ProtocolError("interrupted orphan unexpectedly became a valid ZIP")

    return {
        "performed": True,
        "event_count": len(events),
        "last_event_sha256": previous,
        "referenced_state_file_count": len(expected_target_paths),
        "stage0_shards": 256,
        "union_shortlist_count": len(union),
        "stage1_batches": batch_count,
        "winner_count": len(winner_ordinals),
        "orphan_preserved_and_bad_zip": True,
    }


def verify_result(
    result_path: Path,
    *,
    workspace_root: Path | None = None,
    calibration_path: Path | None = None,
    recovery_path: Path | None = None,
    output_dir: Path | None = None,
    aux_dir: Path | None = None,
    enforce_published_artifact_hashes: bool = True,
) -> dict[str, Any]:
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("verifier must run in a CUDA-library-free process")
    result_path = result_path.resolve()
    calibration_path = (
        common.PACKAGE_DIR / "tier_b_source_free_calibration.json"
        if calibration_path is None
        else calibration_path.resolve()
    )
    recovery_path = (
        common.PACKAGE_DIR / "recovery_orphan_stage0_136.json"
        if recovery_path is None
        else recovery_path.resolve()
    )
    if enforce_published_artifact_hashes:
        expected_hashes = (
            (result_path, PUBLISHED_RESULT_SHA256, "result"),
            (calibration_path, PUBLISHED_CALIBRATION_SHA256, "calibration"),
            (recovery_path, PUBLISHED_RECOVERY_SHA256, "recovery"),
        )
        for path, expected, label in expected_hashes:
            if common.sha256_file(path) != expected:
                raise common.ProtocolError(f"published {label} SHA-256 mismatch")

    lock = common.load_candidate_lock()
    result = _load_json(result_path, "result")
    calibration = _load_json(calibration_path, "calibration")
    recovery = _load_json(recovery_path, "recovery")
    _verify_calibration(calibration, lock)

    common.strict_keys(
        result,
        (
            "schema",
            "strict_ptq",
            "claim",
            "pinned_panel",
            "bindings",
            "backend",
            "data_firewall",
            "candidate_space",
            "coordinates",
            "resume_state",
            "search",
            "validation",
            "physical_ledger",
            "decision",
        ),
        "result",
    )
    if result["schema"] != common.SCHEMA or result["strict_ptq"] is not True:
        raise common.ProtocolError("result schema/strict-PTQ mismatch")
    _equal_tree(
        result["claim"],
        {
            "procedural_anchor_discovery_only": True,
            "qwen_training_lineage_claimed": False,
            "tier_a_artifacts_modified": False,
            "claim_boundary": lock["claim_boundary"],
        },
        "claim",
    )
    _equal_tree(result["pinned_panel"], {"opened": False, "access_permitted": False}, "pinned_panel")

    expected_bindings = {
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "common_sha256": COMMON_SHA256,
        "kernels_sha256": KERNELS_SHA256,
        "tier_a_common_sha256": common.EXPECTED_TIER_A_COMMON_SHA256,
        "calibration_sha256": common.sha256_file(calibration_path),
        "qwen_revision": common.QWEN_REVISION,
        "mcore_revision": common.MCORE_REVISION,
    }
    _equal_tree(result["bindings"], expected_bindings, "bindings")
    actual_package_hashes = {
        "runner": common.sha256_file(common.PACKAGE_DIR / "tier_b_gate.py"),
        "common": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "candidate_lock": common.sha256_file(common.PACKAGE_DIR / "candidate_lock.json"),
        "tier_a_common": common.sha256_file(common.TIER_A_COMMON_PATH),
    }
    expected_package_hashes = {
        "runner": RUNNER_SHA256,
        "common": COMMON_SHA256,
        "kernels": KERNELS_SHA256,
        "candidate_lock": common.CANDIDATE_LOCK_FILE_SHA256,
        "tier_a_common": common.EXPECTED_TIER_A_COMMON_SHA256,
    }
    if actual_package_hashes != expected_package_hashes:
        raise common.ProtocolError("sealed package code hash mismatch")

    common.strict_keys(result["backend"], ("production", "name", "parity", "source_free_calibration"), "backend")
    if (
        result["backend"]["production"] is not True
        or result["backend"]["name"] != "cupy_curand_philox_random_access_with_torch_cuda_parity"
    ):
        raise common.ProtocolError("result backend mismatch")
    _verify_parity(result["backend"]["parity"], "backend.parity")
    _equal_tree(result["backend"]["source_free_calibration"], calibration, "backend.source_free_calibration")
    _equal_tree(result["backend"]["parity"], calibration["parity"], "production/calibration parity")

    rows = common.load_source_rows(workspace_root)
    exclusion = common.exclusion_binding(workspace_root)
    eligible = [row for row in rows if not row.excluded]
    selection_rows = [row for row in eligible if row.split == "candidate_selection"]
    validation_rows = [row for row in eligible if row.split == "validation"]
    stage0_plan = common.make_plan(rows, stage0=True)
    full_plan = common.make_plan(rows, stage0=False)
    selection_plan = [row for row in full_plan if row.source.split == "candidate_selection"]
    validation_plan = [row for row in full_plan if row.source.split == "validation"]
    expected_candidate_space = {
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "equivalence_map": common.equivalence_map_object(),
        "equivalence_map_sha256": common.equivalence_map_sha256(),
        "domain_count": len(common.DOMAIN_IDS),
        "domain_ids": list(common.DOMAIN_IDS),
    }
    _equal_tree(result["candidate_space"], expected_candidate_space, "candidate_space")
    expected_coordinates = {
        "stage0_plan_sha256": common.plan_sha256(stage0_plan),
        "full_plan_sha256": common.plan_sha256(full_plan),
        "stage0": common.plan_json(stage0_plan),
        "full": common.plan_json(full_plan),
    }
    _equal_tree(result["coordinates"], expected_coordinates, "coordinates")

    events, last_event_hash = _verify_embedded_events(result)
    _verify_recovery_evidence(recovery, events)

    firewall = result["data_firewall"]
    common.strict_keys(
        firewall,
        (
            "auxiliary_directory",
            "exclusion_binding",
            "excluded",
            "eligible",
            "access_log",
            "all_winners_frozen_before_validation",
            "excluded_payloads_opened",
        ),
        "data_firewall",
    )
    expected_excluded = [
        {"tensor_name": row.tensor_name, "basename": row.basename, "payload_opened": False}
        for row in rows
        if row.excluded
    ]
    expected_eligible = [
        {
            "tensor_name": row.tensor_name,
            "basename": row.basename,
            "expert": row.expert,
            "role": row.role,
            "split": row.split,
            "sha256": row.sha256,
            "bytes": row.bytes,
        }
        for row in eligible
    ]
    exclusion_receipt = _verify_exclusion_binding(firewall["exclusion_binding"], exclusion)
    _equal_tree(firewall["excluded"], expected_excluded, "firewall.excluded")
    _equal_tree(firewall["eligible"], expected_eligible, "firewall.eligible")
    if firewall["all_winners_frozen_before_validation"] is not True or int(firewall["excluded_payloads_opened"]) != 0:
        raise common.ProtocolError("firewall freeze/exclusion status mismatch")
    expected_log: list[dict[str, Any]] = [
        {
            "sequence": 0,
            "event": "production_cuda_parity_passed_before_manifest_directory_or_payload_access",
        }
    ]
    for source in selection_rows:
        expected_log.append(
            {
                "sequence": len(expected_log),
                "event": "payload_opened_and_hash_verified",
                "tensor_name": source.tensor_name,
                "split": source.split,
                "sha256": source.sha256,
            }
        )
    expected_log.append(
        {
            "sequence": len(expected_log),
            "event": "all_33_global_winners_state_backed_before_validation_payload_access",
            "winner_freeze_sha256": result["resume_state"]["winner_freeze_sha256"],
        }
    )
    for source in validation_rows:
        expected_log.append(
            {
                "sequence": len(expected_log),
                "event": "payload_opened_and_hash_verified",
                "tensor_name": source.tensor_name,
                "split": source.split,
                "sha256": source.sha256,
            }
        )
    _equal_tree(firewall["access_log"], expected_log, "firewall.access_log")

    search = result["search"]
    common.strict_keys(
        search,
        (
            "stage0_top_k_per_domain",
            "stage0_shard_count",
            "union_shortlist_count",
            "stage1_winners",
            "selection_details",
            "selection_folds",
        ),
        "search",
    )
    if (
        int(search["stage0_top_k_per_domain"]) != common.STAGE0_TOP_K
        or int(search["stage0_shard_count"]) != 256
        or not 1 <= int(search["union_shortlist_count"]) <= len(common.DOMAIN_IDS) * common.STAGE0_TOP_K
    ):
        raise common.ProtocolError("search cascade count mismatch")
    common.strict_keys(search["stage1_winners"], common.DOMAIN_IDS, "stage1_winners")
    _verify_domain_details_and_folds(search["selection_details"], search["selection_folds"], selection_plan, "selection")
    for domain_id in common.DOMAIN_IDS:
        record = search["stage1_winners"][domain_id]
        common.strict_keys(record, ("candidate", "selection_q"), f"winner.{domain_id}")
        candidate_json = record["candidate"]
        common.strict_keys(
            candidate_json,
            (
                "ordinal",
                "id",
                "base_seed",
                "pipeline_parallel_size",
                "expert_parallel_size",
                "expert_tensor_parallel_size",
                "expert_assignment",
                "projection_packing",
            ),
            f"winner.{domain_id}.candidate",
        )
        candidate = common.decode_ordinal(int(candidate_json["ordinal"]))
        _equal_tree(candidate_json, candidate.to_json(), f"winner.{domain_id}.candidate")
        if candidate.pp_index not in (0, 2, 3):
            raise common.ProtocolError(f"winner is not a representative ordinal: {domain_id}")
        _close(
            record["selection_q"],
            search["selection_folds"][domain_id]["pooled"]["q"],
            f"winner/detail selection q {domain_id}",
            rtol=3e-7,
            atol=3e-9,
        )

    validation = result["validation"]
    common.strict_keys(validation, ("details", "folds", "null_captures"), "validation")
    _verify_domain_details_and_folds(validation["details"], validation["folds"], validation_plan, "validation")
    common.strict_keys(validation["null_captures"], common.NULL_DOMAIN_IDS, "validation.null_captures")
    for domain_id in common.NULL_DOMAIN_IDS:
        _close(
            validation["null_captures"][domain_id],
            validation["folds"][domain_id]["pooled"]["capture"],
            f"null capture {domain_id}",
        )
    ledger = common.physical_ledger()
    _equal_tree(result["physical_ledger"], ledger, "physical_ledger")
    decision = common.make_decision(validation["folds"]["source"], validation["null_captures"])
    _equal_tree(result["decision"], decision, "decision")

    if output_dir is None:
        journal_receipt = {
            "performed": False,
            "embedded_event_chain_verified": True,
            "event_count": len(events),
            "last_event_sha256": last_event_hash,
        }
    else:
        journal_receipt = _verify_state_arrays(output_dir, result, events, recovery)

    if aux_dir is None:
        payload_receipt: dict[str, Any] = {
            "rehash_performed": False,
            "eligible_hashed": 0,
            "excluded_payloads_opened": 0,
        }
    else:
        aux_dir = aux_dir.resolve()
        if aux_dir != Path(str(firewall["auxiliary_directory"])).resolve():
            raise common.ProtocolError("verification auxiliary directory differs from production record")
        paths = common.validate_aux_directory(aux_dir, rows)
        hashes = []
        for source in eligible:
            digest = common.sha256_file(paths[source.tensor_name])
            if digest != source.sha256:
                raise common.ProtocolError(f"eligible payload hash mismatch: {source.basename}")
            hashes.append({"tensor_name": source.tensor_name, "sha256": digest})
        payload_receipt = {
            "rehash_performed": True,
            "eligible_hashed": len(hashes),
            "eligible_hashes": hashes,
            "excluded_payloads_opened": 0,
        }

    return {
        "schema": "qwen3_initialization_anchor_tier_b_verification_receipt_v1",
        "status": "PASS",
        "result_path": str(result_path),
        "result_sha256": common.sha256_file(result_path),
        "calibration_sha256": common.sha256_file(calibration_path),
        "recovery_evidence_sha256": common.sha256_file(recovery_path),
        "verifier_sha256": common.sha256_file(Path(__file__).resolve()),
        "sealed_package_hashes": actual_package_hashes,
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "search_domain_count": len(common.DOMAIN_IDS),
        "union_shortlist_count": int(search["union_shortlist_count"]),
        "decision_state": decision["state"],
        "raw_source_validation_capture": decision["raw_source_validation_capture"],
        "bias_corrected_capture": decision["bias_corrected_capture"],
        "bias_corrected_upper_3se": decision["bias_corrected_upper_3se"],
        "journal_verification": journal_receipt,
        "payload_verification": payload_receipt,
        "exclusion_manifest_verification": exclusion_receipt,
        "source_firewall_verified": True,
        "pinned_panel_opened": False,
        "excluded_payloads_opened": 0,
        "cuda_libraries_imported": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--recovery", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aux-dir", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = verify_result(
        args.result,
        workspace_root=args.workspace_root,
        calibration_path=args.calibration,
        recovery_path=args.recovery,
        output_dir=args.output_dir,
        aux_dir=args.aux_dir,
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.receipt is not None:
        receipt_path = args.receipt.resolve()
        if receipt_path.exists():
            raise common.ProtocolError("verification receipt already exists")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.chmod(receipt_path, 0o444)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
