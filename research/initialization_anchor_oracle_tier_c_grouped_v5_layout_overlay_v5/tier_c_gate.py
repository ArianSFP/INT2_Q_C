"""Tier-C grouped-v5 new-layout overlay calibration and 33-domain gate."""

from __future__ import annotations

# A scientific action may only be dispatched after the clean v5 bootstrap has
# authenticated exact directory closure and every source byte.  This guard is
# intentionally before NumPy or any package import.
if __name__ == "__main__":
    raise SystemExit(
        "direct execution is forbidden; use `python -B -I "
        "/workspace/INT2__compression/INT2_Q_C/research/"
        "initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5/verify_prelaunch.py "
        "--dispatch-tier-c ...`"
    )

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common
import kernels
import overlay
import parity


EXPECTED_TIER_B_GATE_SHA256 = "83eb7682c8185d8f27dbd4b7d39de96cb54dad1c887a4cf026ac4ea759159665"
TIER_B_GATE_PATH = common.TIER_B_DIR / "tier_b_gate.py"


def _load_frozen_search_engine():
    dependency_path = common.require_regular_file_before_resolve(
        TIER_B_GATE_PATH, "byte-frozen Tier-B search engine"
    )
    if common.sha256_file(dependency_path) != EXPECTED_TIER_B_GATE_SHA256:
        raise common.ProtocolError("byte-frozen Tier-B search-engine SHA-256 mismatch")
    name = "_qwen_frozen_tier_b_search_engine_for_tier_c"
    if name in sys.modules:
        return sys.modules[name]
    # The frozen generic engine imports the already-loaded Tier-C common and
    # kernels modules.  Its candidate enumeration therefore cannot fall back
    # to the Tier-B family.
    if sys.modules.get("common") is not common or sys.modules.get("kernels") is not kernels:
        raise common.ProtocolError("Tier-C dependency injection identity mismatch")
    spec = importlib.util.spec_from_file_location(name, dependency_path)
    if spec is None or spec.loader is None:
        raise common.ProtocolError("cannot load byte-frozen search engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.common is not common or module.kernels is not kernels:
        raise common.ProtocolError("frozen search engine imported wrong candidate ABI")
    return module


BASE = _load_frozen_search_engine()
class StateJournal:
    """Symlink-hardened append-only state compatible with the frozen engine."""

    def __init__(self, root: Path, boundary_guard: common.BoundaryGuard | None = None):
        self.boundary_guard = boundary_guard
        self.assert_boundary("before state journal root create/open")
        self.root = common.ensure_output_directory(
            root, allow_existing=True, label="state journal root"
        )
        self.assert_boundary("before state files directory create/open")
        self.files = common.ensure_output_directory(
            self.root / "files", allow_existing=True, label="state files directory"
        )
        self.assert_boundary("before state events directory create/open")
        self.events = common.ensure_output_directory(
            self.root / "events", allow_existing=True, label="state events directory"
        )
        self._events = self._load_and_verify_events()

    def assert_boundary(self, phase: str) -> None:
        if self.boundary_guard is not None:
            self.boundary_guard.revalidate(phase)

    def _load_and_verify_events(self) -> list[dict[str, Any]]:
        directory_entries = sorted(self.events.iterdir())
        if any(path.suffix != ".json" for path in directory_entries):
            raise common.ProtocolError("unexpected state event directory entry")
        paths = directory_entries
        events: list[dict[str, Any]] = []
        expected_file_names: set[str] = set()
        previous = "0" * 64
        for expected_sequence, unresolved in enumerate(paths):
            if unresolved.name != f"{expected_sequence:06d}.json":
                raise common.ProtocolError("state journal sequence/name violation")
            path = common.require_regular_file_before_resolve(
                unresolved, "state journal event"
            )
            raw = path.read_bytes()
            value = _json_object(raw, "state journal event")
            common.strict_keys(
                value,
                (
                    "sequence", "previous_event_sha256", "kind", "key", "relative_path",
                    "file_sha256", "file_bytes", "created_unix_ns",
                ),
                "state event",
            )
            if (
                not isinstance(value["sequence"], int)
                or isinstance(value["sequence"], bool)
                or value["sequence"] != expected_sequence
                or value["previous_event_sha256"] != previous
            ):
                raise common.ProtocolError("state journal hash-chain violation")
            if (
                not isinstance(value["previous_event_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", value["previous_event_sha256"]) is None
                or not isinstance(value["kind"], str)
                or not isinstance(value["key"], str)
                or not isinstance(value["relative_path"], str)
                or not isinstance(value["file_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", value["file_sha256"]) is None
                or not isinstance(value["file_bytes"], int)
                or isinstance(value["file_bytes"], bool)
                or value["file_bytes"] < 0
                or not isinstance(value["created_unix_ns"], int)
                or isinstance(value["created_unix_ns"], bool)
                or value["created_unix_ns"] <= 0
            ):
                raise common.ProtocolError("state journal event type/value violation")
            relative = Path(str(value["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise common.ProtocolError("state event target escapes journal")
            target = common.require_regular_file_before_resolve(
                self.root / relative, "state event target"
            )
            try:
                target.relative_to(self.root)
            except ValueError as error:
                raise common.ProtocolError("state event target escapes journal") from error
            if (
                target.stat().st_size != int(value["file_bytes"])
                or common.sha256_file(target) != value["file_sha256"]
            ):
                raise common.ProtocolError("state event target hash/size mismatch")
            if target.parent != self.files:
                raise common.ProtocolError("state event target is outside the files directory")
            expected_file_names.add(target.name)
            previous = common.sha256_bytes(raw)
            events.append(value)
        actual_file_entries = list(self.files.iterdir())
        if {path.name for path in actual_file_entries} != expected_file_names:
            raise common.ProtocolError("orphan, missing, or unexpected state target")
        for path in actual_file_entries:
            common.require_regular_file_before_resolve(path, "state target directory entry")
        self._validate_event_grammar(events)
        return events

    @staticmethod
    def _validate_event_grammar(events: Sequence[Mapping[str, Any]]) -> None:
        """Require every journal to be a prefix of the one frozen run grammar."""
        phase = "header"
        stage0_next = 0
        stage1_next = 0
        for index, event in enumerate(events):
            pair = (event.get("kind"), event.get("key"))
            if phase == "header":
                if pair != ("run_header", "immutable") or index != 0:
                    raise common.ProtocolError("state journal must begin with run_header/immutable")
                phase = "stage0"
            elif phase == "stage0":
                if pair == ("stage0", f"{stage0_next:03d}") and stage0_next < common.SEED_SHARD_COUNT:
                    stage0_next += 1
                elif pair == ("stage0_merged", "global") and stage0_next == common.SEED_SHARD_COUNT:
                    phase = "overlay_state"
                else:
                    raise common.ProtocolError("state journal stage0 grammar violation")
            elif phase == "overlay_state":
                if pair != ("layout_overlay_merged", "global"):
                    raise common.ProtocolError("state journal overlay-state grammar violation")
                phase = "overlay_receipt"
            elif phase == "overlay_receipt":
                if pair != ("layout_overlay_receipt", "global"):
                    raise common.ProtocolError("state journal overlay-receipt grammar violation")
                phase = "stage1"
            elif phase == "stage1":
                if pair == ("stage1", f"{stage1_next:04d}"):
                    stage1_next += 1
                elif pair == ("stage1_winners", "global") and stage1_next > 0:
                    phase = "firewall"
                else:
                    raise common.ProtocolError("state journal stage1 grammar violation")
            elif phase == "firewall":
                if pair != ("validation_firewall", "winners_frozen"):
                    raise common.ProtocolError("state journal validation-firewall grammar violation")
                phase = "result"
            elif phase == "result":
                if pair != ("result", "final") or index != len(events) - 1:
                    raise common.ProtocolError("state journal final-result grammar violation")
                phase = "complete"
            else:
                raise common.ProtocolError("state journal contains an event after final result")

    def _assert_next_event(self, kind: str, key: str) -> None:
        if any(event["kind"] == "result" for event in self._events):
            raise common.ProtocolError("completed state journal is immutable")
        probe = list(self._events)
        probe.append({"kind": kind, "key": key})
        self._validate_event_grammar(probe)

    @property
    def events_list(self) -> list[dict[str, Any]]:
        return list(self._events)

    def lookup(self, kind: str, key: str) -> Path | None:
        matches = [
            event for event in self._events
            if event["kind"] == kind and event["key"] == key
        ]
        if len(matches) > 1:
            raise common.ProtocolError(f"duplicate state journal key: {kind}/{key}")
        if not matches:
            return None
        relative = Path(str(matches[0]["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise common.ProtocolError("state journal lookup escapes root")
        return common.require_regular_file_before_resolve(
            self.root / relative, "state journal lookup target"
        )

    def _record_existing_file(self, kind: str, key: str, target: Path) -> Path:
        if self.lookup(kind, key) is not None:
            raise common.ProtocolError(f"attempt to overwrite state key: {kind}/{key}")
        target = common.require_regular_file_before_resolve(target, "new state target")
        try:
            relative = target.relative_to(self.root)
        except ValueError as error:
            raise common.ProtocolError("new state target escapes journal") from error
        sequence = len(self._events)
        previous = (
            "0" * 64
            if not self._events
            else common.sha256_file(
                common.require_regular_file_before_resolve(
                    self.events / f"{sequence-1:06d}.json", "previous state event"
                )
            )
        )
        event = {
            "sequence": sequence,
            "previous_event_sha256": previous,
            "kind": kind,
            "key": key,
            "relative_path": str(relative).replace("\\", "/"),
            "file_sha256": common.sha256_file(target),
            "file_bytes": target.stat().st_size,
            "created_unix_ns": time.time_ns(),
        }
        event_path = self.events / f"{sequence:06d}.json"
        self.assert_boundary(f"immediately before state event create-new {kind}/{key}")
        with common.open_create_new(event_path, binary=False, label="state event") as handle:
            json.dump(event, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._events.append(event)
        return target

    def write_json(self, kind: str, key: str, value: Mapping[str, Any]) -> Path:
        self._assert_next_event(kind, key)
        self.assert_boundary(f"immediately before state JSON create-new {kind}/{key}")
        target = self.files / f"{kind}_{key}.json"
        common.preflight_create_new_file(target, "state JSON target")
        common.preflight_create_new_file(
            self.events / f"{len(self._events):06d}.json", "state event"
        )
        common.write_json_create_new(target, value, "state JSON target")
        return self._record_existing_file(kind, key, target)

    def write_npz(self, kind: str, key: str, **arrays: np.ndarray) -> Path:
        self._assert_next_event(kind, key)
        self.assert_boundary(f"immediately before state NPZ create-new {kind}/{key}")
        target = self.files / f"{kind}_{key}.npz"
        common.preflight_create_new_file(target, "state NPZ target")
        common.preflight_create_new_file(
            self.events / f"{len(self._events):06d}.json", "state event"
        )
        with common.open_create_new(target, binary=True, label="state NPZ target") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        return self._record_existing_file(kind, key, target)


def _affine_sse_from_moments_f16(cp, g_fit, g_score, w_fit, w_score):
    """Fit in float64, serialize alpha/mu as FP16, then score decoded FP16."""
    g_fit = cp.asarray(g_fit, dtype=cp.float64)
    g_score = cp.asarray(g_score, dtype=cp.float64)
    w_fit = cp.asarray(w_fit, dtype=cp.float64)
    w_score = cp.asarray(w_score, dtype=cp.float64)
    n_fit = int(g_fit.shape[1])
    n_score = int(g_score.shape[1])
    sum_g_fit = cp.sum(g_fit, axis=1, dtype=cp.float64)
    sum_g2_fit = cp.sum(g_fit * g_fit, axis=1, dtype=cp.float64)
    sum_w_fit = cp.sum(w_fit, axis=1, dtype=cp.float64)
    sum_wg_fit = g_fit @ w_fit.T
    centered_g2 = sum_g2_fit[:, None] - sum_g_fit[:, None] ** 2 / n_fit
    centered_wg = sum_wg_fit - sum_g_fit[:, None] * sum_w_fit[None, :] / n_fit
    alpha_unstored = cp.where(centered_g2 > 0.0, centered_wg / centered_g2, 0.0)
    mean_w = sum_w_fit / n_fit
    mu_unstored = mean_w[None, :] - alpha_unstored * (sum_g_fit[:, None] / n_fit)
    # This is the device equivalent of common.quantize_affine_f16le. Both
    # coefficients are rounded independently to IEEE binary16 before scoring.
    alpha = alpha_unstored.astype(cp.float16).astype(cp.float64)
    mu = mu_unstored.astype(cp.float16).astype(cp.float64)
    if bool(cp.any(~cp.isfinite(alpha))) or bool(cp.any(~cp.isfinite(mu))):
        raise common.ProtocolError("affine coefficient overflowed the charged FP16 codec")

    sum_g_score = cp.sum(g_score, axis=1, dtype=cp.float64)
    sum_g2_score = cp.sum(g_score * g_score, axis=1, dtype=cp.float64)
    sum_w_score = cp.sum(w_score, axis=1, dtype=cp.float64)
    sum_w2_score = cp.sum(w_score * w_score, axis=1, dtype=cp.float64)
    sum_wg_score = g_score @ w_score.T
    sse = (sum_w2_score[None, :] + n_score * mu * mu
           + alpha * alpha * sum_g2_score[:, None]
           + 2.0 * mu * alpha * sum_g_score[:, None]
           - 2.0 * mu * sum_w_score[None, :] - 2.0 * alpha * sum_wg_score)
    baseline = sum_w2_score - 2.0 * mean_w * sum_w_score + n_score * mean_w * mean_w
    return cp.maximum(sse, 0.0), baseline


# The byte-frozen Tier-B engine calls this global from stage 0, stage 1 and
# reporting. Rebinding it here makes every scientific score use the charged
# decoder precision without mutating Tier B.
BASE._affine_sse_from_moments = _affine_sse_from_moments_f16


def _float_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def _counts(total: int, items: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, items)
    return tuple(quotient + (index < remainder) for index in range(items))


def _synthetic_calibration_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Payload-free emulation of the exact production 23-matrix slice layout."""
    identities = common.CALIBRATION_SELECTION_IDENTITIES
    if len(identities) != 23:
        raise common.ProtocolError("calibration identity fixture must contain 23 matrices")
    fit_counts = _counts(common.STAGE0_FIT, len(identities))
    score_counts = _counts(common.STAGE0_SCORE, len(identities))
    experts: list[int] = []
    roles: list[int] = []
    coordinates: list[int] = []
    role_fit_anchor = {"up": [], "down": []}
    role_score_anchor = {"up": [], "down": []}
    role_fit_domain = {"up": [], "down": []}
    role_score_domain = {"up": [], "down": []}
    matrix_rows = []
    fit_cursor = 0
    score_cursor = 0
    anchor_cursor = 0
    for matrix_index, ((expert, role), fit_count, score_count) in enumerate(zip(identities, fit_counts, score_counts)):
        role_code = 0 if role == "up" else 1
        fit_anchor = list(range(anchor_cursor, anchor_cursor + fit_count))
        anchor_cursor += fit_count
        score_anchor = list(range(anchor_cursor, anchor_cursor + score_count))
        anchor_cursor += score_count
        role_fit_anchor[role].extend(fit_anchor)
        role_score_anchor[role].extend(score_anchor)
        role_fit_domain[role].extend(range(fit_cursor, fit_cursor + fit_count))
        role_score_domain[role].extend(range(score_cursor, score_cursor + score_count))
        fit_cursor += fit_count
        score_cursor += score_count
        for split, count in (("fit", fit_count), ("score", score_count)):
            for local in range(count):
                digest = hashlib.sha256(
                    b"TIERC-GROUPED-SOURCE-FREE-CALIBRATION-v3\0"
                    + matrix_index.to_bytes(2, "little") + split.encode("ascii")
                    + local.to_bytes(2, "little")
                ).digest()
                experts.append(expert)
                roles.append(role_code)
                coordinates.append(int.from_bytes(digest[:8], "little") % common.WEIGHTS_PER_MATRIX)
        matrix_rows.append({"matrix_index": matrix_index, "expert": expert, "role": role,
                            "fit_count": fit_count, "score_count": score_count,
                            "fit_anchor_start": fit_anchor[0], "score_anchor_start": score_anchor[0]})
    role_counts = {role: {"fit": len(role_fit_anchor[role]), "score": len(role_score_anchor[role])}
                   for role in ("up", "down")}
    if role_counts != {"up": {"fit": 122, "score": 122}, "down": {"fit": 134, "score": 134}}:
        raise common.ProtocolError("calibration role counts differ from production 122/134 split")
    layout = {"matrix_rows": matrix_rows, "role_counts": role_counts,
              "role_fit_anchor": role_fit_anchor, "role_score_anchor": role_score_anchor,
              "role_fit_domain": role_fit_domain, "role_score_domain": role_score_domain}
    return (np.asarray(experts, dtype=np.int32), np.asarray(roles, dtype=np.int32),
            np.asarray(coordinates, dtype=np.uint64), layout)


def _synthetic_domains() -> tuple[np.ndarray, np.ndarray]:
    # The 256 fit and 256 score values are indexed through the exact production
    # 122-up/134-down role map. They are timing fixtures, not scientific data.
    raw = np.arange(len(common.DOMAIN_IDS) * 512, dtype=np.uint64).reshape(
        len(common.DOMAIN_IDS), 512
    )
    values = (((raw * np.uint64(0x9E3779B1) + np.uint64(0x7F4A7C15)) & np.uint64(0xFFFFFF))
              .astype(np.float64) / float(1 << 23) - 1.0).astype(np.float32)
    return values[:, :256].copy(), values[:, 256:].copy()


def _synthetic_coordinate_sha256(
    experts: np.ndarray, roles: np.ndarray, coordinates: np.ndarray, layout: Mapping[str, Any]
) -> str:
    return common.sha256_bytes(
        np.asarray(experts, dtype="<i4").tobytes()
        + np.asarray(roles, dtype="<i4").tobytes()
        + np.asarray(coordinates, dtype="<u8").tobytes()
        + common.canonical_json_bytes(layout)
    )


def _synthetic_domain_sha256(fit: np.ndarray, score: np.ndarray) -> str:
    return common.sha256_bytes(
        np.asarray(fit, dtype="<f4").tobytes()
        + np.asarray(score, dtype="<f4").tobytes()
    )


def _synthetic_stage0_q(access: kernels.PhiloxRandomAccess, anchors, fit, score, layout):
    cp = access.cp
    total_sse = cp.zeros((anchors.shape[0], len(common.DOMAIN_IDS)), dtype=cp.float64)
    total_baseline = cp.zeros(len(common.DOMAIN_IDS), dtype=cp.float64)
    for role in ("up", "down"):
        fit_anchor = cp.asarray(layout["role_fit_anchor"][role], dtype=cp.int64)
        score_anchor = cp.asarray(layout["role_score_anchor"][role], dtype=cp.int64)
        fit_domain = np.asarray(layout["role_fit_domain"][role], dtype=np.int64)
        score_domain = np.asarray(layout["role_score_domain"][role], dtype=np.int64)
        sse, baseline = _affine_sse_from_moments_f16(
            cp, anchors[:, fit_anchor], anchors[:, score_anchor],
            fit[:, fit_domain], score[:, score_domain]
        )
        total_sse += sse
        total_baseline += baseline
    if bool(cp.any(total_baseline <= 0.0)):
        raise common.ProtocolError("synthetic calibration baseline is non-positive")
    return total_sse / total_baseline[None, :]


def run_calibration(output_path: Path, source_trace_path: Path) -> Path:
    boundary = common.BoundaryGuard(
        "SOURCE_FREE_CALIBRATION_CREATE_ONCE",
        outputs=(("source-free calibration output", output_path, "file", False),),
        inputs=(("source-trace input", source_trace_path, "file"),),
    )
    if os.environ.get(parity.SINGLE_PARAM_ENV) != "1":
        raise common.ProtocolError(
            f"set {parity.SINGLE_PARAM_ENV}=1 before calibration can initialize CUDA/TE"
        )
    lock = common.load_candidate_lock()
    source_trace = parity.load_source_trace(source_trace_path)
    access = kernels.PhiloxRandomAccess(0)
    parity_receipt = parity.run_parity(access, source_trace_path)
    ordinals = common.representative_ordinals(0, 256)
    expected_candidates = int(lock["source_free_calibration"]["candidate_count"])
    if len(ordinals) != expected_candidates or len(ordinals) != 164_864:
        raise common.ProtocolError("calibration candidate count differs from lock")
    experts, roles, coordinates, calibration_layout = _synthetic_calibration_coordinates()
    if len(coordinates) != int(lock["source_free_calibration"]["coordinate_count"]):
        raise common.ProtocolError("calibration coordinate count differs from lock")
    synthetic_fit, synthetic_score = _synthetic_domains()
    coordinate_fixture_sha256 = _synthetic_coordinate_sha256(experts, roles, coordinates, calibration_layout)
    domain_fixture_sha256 = _synthetic_domain_sha256(synthetic_fit, synthetic_score)
    matrix_layout_sha256 = common.sha256_bytes(
        common.canonical_json_bytes(calibration_layout["matrix_rows"])
    )
    if coordinate_fixture_sha256 != lock["source_free_calibration"]["coordinate_fixture_sha256"]:
        raise common.ProtocolError("calibration coordinate fixture differs from lock")
    if domain_fixture_sha256 != lock["source_free_calibration"]["domain_fixture_sha256"]:
        raise common.ProtocolError("calibration domain fixture differs from lock")
    if matrix_layout_sha256 != lock["source_free_calibration"]["matrix_layout_sha256"]:
        raise common.ProtocolError("calibration matrix layout differs from lock")
    if calibration_layout["role_counts"] != {
        role: {"fit": count, "score": count}
        for role, count in lock["source_free_calibration"]["role_counts_per_fit_and_score_split"].items()
    }:
        raise common.ProtocolError("calibration role counts differ from lock")
    cp = access.cp
    output = cp.empty((len(ordinals), len(coordinates)), dtype=cp.float32)

    # Warmup proves the exact full-shard allocation and end-to-end objective fit.
    access.generate(ordinals, experts, roles, coordinates, output=output)
    warm_q = _synthetic_stage0_q(access, output, synthetic_fit, synthetic_score, calibration_layout)
    _validate_full_stage0_q(access, warm_q, len(ordinals))
    BASE._exact_top_k(cp, warm_q, ordinals, common.STAGE0_TOP_K)
    cp.cuda.runtime.deviceSynchronize()
    del warm_q

    kernel_seconds = []
    end_to_end_seconds = []
    output_hashes = []
    top_hashes = []
    repetitions = int(lock["source_free_calibration"]["repetitions"])
    for _ in range(repetitions):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        access.generate(ordinals, experts, roles, coordinates, output=output)
        stop.record()
        stop.synchronize()
        kernel_seconds.append(float(cp.cuda.get_elapsed_time(start, stop)) / 1000.0)
        sentinel = output.reshape(-1)[:: max(1, output.size // 4096)][:4096]
        output_hashes.append(_float_sha(cp.asnumpy(sentinel)))

        wall_start = time.perf_counter()
        access.generate(ordinals, experts, roles, coordinates, output=output)
        q = _synthetic_stage0_q(access, output, synthetic_fit, synthetic_score, calibration_layout)
        _validate_full_stage0_q(access, q, len(ordinals))
        top_ordinals, top_q = BASE._exact_top_k(cp, q, ordinals, common.STAGE0_TOP_K)
        cp.cuda.runtime.deviceSynchronize()
        end_to_end_seconds.append(time.perf_counter() - wall_start)
        top_hashes.append(
            common.sha256_bytes(
                np.asarray(top_ordinals, dtype="<u8").tobytes()
                + np.asarray(top_q, dtype="<f8").tobytes()
            )
        )
        del q
    if len(set(output_hashes)) != 1 or len(set(top_hashes)) != 1:
        raise common.ProtocolError("calibration output/objective was nondeterministic")
    generated = len(ordinals) * len(coordinates)
    kernel_rates = [generated / seconds for seconds in kernel_seconds]
    shard_rates = [len(ordinals) / seconds for seconds in end_to_end_seconds]
    result = {
        "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_calibration_v5",
        "status": "PASS_SOURCE_FREE_PARITY_AND_CALIBRATION",
        "source_manifest_directory_or_payload_opened": False,
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "parity_sha256": common.sha256_file(common.PACKAGE_DIR / "parity.py"),
        "frozen_search_engine_sha256": EXPECTED_TIER_B_GATE_SHA256,
        "source_trace_file_sha256": common.sha256_file(
            common.require_regular_file_before_resolve(
                source_trace_path, "calibration source-trace receipt"
            )
        ),
        "source_trace_internal_sha256": source_trace["receipt_sha256"],
        "parity": parity_receipt,
        "candidate_count": len(ordinals),
        "coordinate_count": len(coordinates),
        "coordinate_fixture_sha256": coordinate_fixture_sha256,
        "production_matrix_identity_fixture": [list(row) for row in common.CALIBRATION_SELECTION_IDENTITIES],
        "production_matrix_role_counts": calibration_layout["role_counts"],
        "production_matrix_layout_sha256": matrix_layout_sha256,
        "domain_fixture_sha256": domain_fixture_sha256,
        "values_per_repetition": generated,
        "repetitions": repetitions,
        "kernel_elapsed_seconds": kernel_seconds,
        "kernel_values_per_second": kernel_rates,
        "median_kernel_values_per_second": statistics.median(kernel_rates),
        "end_to_end_shard_seconds": end_to_end_seconds,
        "end_to_end_candidates_per_second": shard_rates,
        "median_end_to_end_shard_seconds": statistics.median(end_to_end_seconds),
        "estimated_stage0_seconds_at_median_kernel_rate": (
            int(lock["search_cascade"]["stage0"]["maximum_generated_normal_values"])
            / statistics.median(kernel_rates)
        ),
        "estimated_stage0_seconds_at_median_end_to_end_rate": (
            int(lock["search_cascade"]["stage0"]["seed_shard_count"])
            * statistics.median(end_to_end_seconds)
        ),
        "output_sentinel_sha256_f32le": output_hashes[0],
        "objective_topk_sha256": top_hashes[0],
        "working_output_bytes": int(output.nbytes),
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "equivalence_map_sha256": common.equivalence_map_sha256(),
        "path_boundary": boundary.receipt(),
    }
    result["receipt_sha256"] = common.sha256_bytes(common.canonical_json_bytes(result))
    boundary.revalidate("immediately before calibration create-new")
    common.write_json_create_new(
        output_path, result, "source-free calibration output"
    )
    return output_path


def _load_calibration(path: Path, source_trace_path: Path) -> dict[str, Any]:
    path = common.require_regular_file_before_resolve(path, "calibration receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "status", "source_manifest_directory_or_payload_opened",
        "candidate_lock_file_sha256", "candidate_lock_internal_sha256", "runner_sha256",
        "common_sha256", "kernels_sha256", "parity_sha256", "frozen_search_engine_sha256",
        "source_trace_file_sha256", "source_trace_internal_sha256", "parity",
        "candidate_count", "coordinate_count", "coordinate_fixture_sha256",
        "production_matrix_identity_fixture", "production_matrix_role_counts",
        "production_matrix_layout_sha256",
        "domain_fixture_sha256", "values_per_repetition", "repetitions",
        "kernel_elapsed_seconds", "kernel_values_per_second", "median_kernel_values_per_second",
        "end_to_end_shard_seconds", "end_to_end_candidates_per_second",
        "median_end_to_end_shard_seconds", "estimated_stage0_seconds_at_median_kernel_rate",
        "estimated_stage0_seconds_at_median_end_to_end_rate", "output_sentinel_sha256_f32le",
        "objective_topk_sha256", "working_output_bytes", "logical_candidate_count",
        "effective_candidate_count", "equivalence_map_sha256",
        "path_boundary",
        "receipt_sha256",
    }
    common.strict_keys(value, required, "calibration")
    normalized = dict(value)
    observed_receipt_sha256 = normalized.pop("receipt_sha256")
    if observed_receipt_sha256 != common.sha256_bytes(common.canonical_json_bytes(normalized)):
        raise common.ProtocolError("calibration internal receipt SHA-256 mismatch")
    if value["schema"] != "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_calibration_v5":
        raise common.ProtocolError("calibration schema mismatch")
    if value["status"] != "PASS_SOURCE_FREE_PARITY_AND_CALIBRATION":
        raise common.ProtocolError("calibration did not pass")
    if value["source_manifest_directory_or_payload_opened"] is not False:
        raise common.ProtocolError("calibration source-free claim mismatch")
    path_boundary = value["path_boundary"]
    if (
        not isinstance(path_boundary, Mapping)
        or path_boundary.get("schema") != "qwen3_tier_c_grouped_v5_path_boundary_v1"
        or path_boundary.get("action") != "SOURCE_FREE_CALIBRATION_CREATE_ONCE"
        or path_boundary.get("pairwise_lexical_inode_and_mount_disjoint") is not True
        or path_boundary.get("revalidation_required_before_every_create_new") is not True
    ):
        raise common.ProtocolError("calibration path boundary is absent or malformed")
    source_trace = parity.load_source_trace(source_trace_path)
    lock = common.load_candidate_lock()
    fixture_experts, fixture_roles, fixture_coordinates, fixture_layout = _synthetic_calibration_coordinates()
    expected = {
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "parity_sha256": common.sha256_file(common.PACKAGE_DIR / "parity.py"),
        "frozen_search_engine_sha256": EXPECTED_TIER_B_GATE_SHA256,
        "source_trace_file_sha256": common.sha256_file(
            common.require_regular_file_before_resolve(
                source_trace_path, "calibration source-trace receipt"
            )
        ),
        "source_trace_internal_sha256": source_trace["receipt_sha256"],
        "candidate_count": 164_864,
        "coordinate_count": 512,
        "coordinate_fixture_sha256": _synthetic_coordinate_sha256(
            fixture_experts, fixture_roles, fixture_coordinates, fixture_layout
        ),
        "domain_fixture_sha256": _synthetic_domain_sha256(*_synthetic_domains()),
        "production_matrix_identity_fixture": [list(row) for row in common.CALIBRATION_SELECTION_IDENTITIES],
        "production_matrix_role_counts": fixture_layout["role_counts"],
        "production_matrix_layout_sha256": common.sha256_bytes(
            common.canonical_json_bytes(fixture_layout["matrix_rows"])
        ),
        "values_per_repetition": 84_410_368,
        "repetitions": 3,
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "equivalence_map_sha256": common.equivalence_map_sha256(),
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise common.ProtocolError(f"calibration binding mismatch: {key}")
    if value["coordinate_fixture_sha256"] != lock["source_free_calibration"]["coordinate_fixture_sha256"]:
        raise common.ProtocolError("calibration coordinate fixture is not lock-bound")
    if value["domain_fixture_sha256"] != lock["source_free_calibration"]["domain_fixture_sha256"]:
        raise common.ProtocolError("calibration domain fixture is not lock-bound")
    if value["production_matrix_layout_sha256"] != lock["source_free_calibration"]["matrix_layout_sha256"]:
        raise common.ProtocolError("calibration matrix layout is not lock-bound")
    locked_role_counts = lock["source_free_calibration"]["role_counts_per_fit_and_score_split"]
    if value["production_matrix_role_counts"] != {
        role: {"fit": count, "score": count} for role, count in locked_role_counts.items()
    }:
        raise common.ProtocolError("calibration role counts are not lock-bound")
    parity.validate_runtime_parity_receipt(value["parity"], source_trace_path)
    repetitions = int(value["repetitions"])
    lists = {
        "kernel_elapsed_seconds": value["kernel_elapsed_seconds"],
        "kernel_values_per_second": value["kernel_values_per_second"],
        "end_to_end_shard_seconds": value["end_to_end_shard_seconds"],
        "end_to_end_candidates_per_second": value["end_to_end_candidates_per_second"],
    }
    for label, rows in lists.items():
        if not isinstance(rows, list) or len(rows) != repetitions:
            raise common.ProtocolError(f"calibration timing count mismatch: {label}")
        if any(not math.isfinite(float(item)) or float(item) <= 0.0 for item in rows):
            raise common.ProtocolError(f"invalid calibration timing/rate: {label}")
    generated = int(value["values_per_repetition"])
    candidates = int(value["candidate_count"])
    for seconds, rate in zip(value["kernel_elapsed_seconds"], value["kernel_values_per_second"]):
        if not math.isclose(float(rate), generated / float(seconds), rel_tol=1e-12):
            raise common.ProtocolError("calibration kernel rate is inconsistent with elapsed time")
    for seconds, rate in zip(
        value["end_to_end_shard_seconds"], value["end_to_end_candidates_per_second"]
    ):
        if not math.isclose(float(rate), candidates / float(seconds), rel_tol=1e-12):
            raise common.ProtocolError("calibration shard rate is inconsistent with elapsed time")
    if value["median_kernel_values_per_second"] != statistics.median(value["kernel_values_per_second"]):
        raise common.ProtocolError("calibration median kernel rate mismatch")
    if value["median_end_to_end_shard_seconds"] != statistics.median(value["end_to_end_shard_seconds"]):
        raise common.ProtocolError("calibration median shard time mismatch")
    if value["working_output_bytes"] != 164_864 * 512 * 4:
        raise common.ProtocolError("calibration working-output bytes mismatch")
    for key in ("output_sentinel_sha256_f32le", "objective_topk_sha256", "receipt_sha256"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise common.ProtocolError(f"calibration malformed SHA-256: {key}")
    return value


def _read_regular_once(path: Path, label: str) -> bytes:
    """Single-descriptor read that rejects links in the complete path chain."""
    unresolved = common.reject_symlink_components_before_normalization(
        path, label, require_exists=True
    )
    before = common.lstat_or_none(unresolved)
    if before is None or not stat.S_ISREG(before.st_mode):
        raise common.ProtocolError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(unresolved, flags)
    except OSError as error:
        raise common.ProtocolError(f"cannot open {label}") from error
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev, opened.st_ino, opened.st_size
        ) != identity:
            raise common.ProtocolError(f"{label} changed before single-descriptor open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise common.ProtocolError(f"{label} was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise common.ProtocolError(f"{label} grew during read")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != identity:
            raise common.ProtocolError(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise common.ProtocolError(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    def reject_constant(token: str):
        raise common.ProtocolError(f"non-finite JSON constant in {label}: {token}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except common.ProtocolError:
        raise
    except Exception as error:
        raise common.ProtocolError(f"invalid UTF-8 JSON in {label}") from error
    if not isinstance(value, dict):
        raise common.ProtocolError(f"{label} must contain one JSON object")
    return value


_AUDIT_MANIFEST_BASENAME = "ARTIFACT_SHA256SUMS.txt"
_AUDIT_RECEIPT_BASENAME = "audit_receipt.json"
_AUDITED_TARGET_KEYS = (
    "artifact_manifest_sha256", "candidate_lock_file_sha256",
    "candidate_lock_internal_sha256", "runner_sha256", "common_sha256",
    "kernels_sha256", "overlay_sha256", "parity_sha256",
)
_AUDIT_VERIFICATION = {
    "manifest_closure_verified": True,
    "receipt_internal_sha256_recomputed": True,
    "candidate_lock_internal_sha256_recomputed": True,
    "exact_schema_and_member_sets_verified": True,
    "access_all_zero": True,
}
_AUDIT_ACCESS = {
    "qwen_or_model_payload_manifest_or_directory_accessed": False,
    "torch_cupy_cuda_transformer_engine_or_megatron_imported": False,
    "gpu_used": False,
    "network_accessed": False,
    "producer_artifacts_modified": False,
}
_SOURCE_AUDIT_AUTHORIZATION = {
    "source_package_passed": True,
    "source_free_gpu_calibration_authorized": True,
    "qwen_payload_or_manifest_launch_authorized": False,
    "production_run_authorized": False,
}
_CALIBRATION_AUDIT_AUTHORIZATION = {
    "source_package_passed": True,
    "source_free_calibration_passed": True,
    "qwen_payload_or_manifest_launch_authorized": True,
    "production_run_authorized": True,
}


def _audit_manifest_closure(
    manifest_path: Path,
    receipt_path: Path,
    manifest_raw: bytes,
    *,
    label: str,
) -> dict[str, bytes]:
    """Parse and authenticate one exact flat audit-package closure."""
    checked_manifest = common.reject_symlink_components_before_normalization(
        manifest_path, f"{label} artifact manifest", require_exists=True
    )
    checked_receipt = common.reject_symlink_components_before_normalization(
        receipt_path, f"{label} receipt", require_exists=True
    )
    if checked_manifest.name != _AUDIT_MANIFEST_BASENAME:
        raise common.ProtocolError(f"{label} artifact manifest basename mismatch")
    if checked_receipt.name != _AUDIT_RECEIPT_BASENAME:
        raise common.ProtocolError(f"{label} receipt basename mismatch")
    if checked_manifest.parent != checked_receipt.parent:
        raise common.ProtocolError(f"{label} manifest and receipt must share one directory")
    if b"\r" in manifest_raw or not manifest_raw.endswith(b"\n"):
        raise common.ProtocolError(f"{label} artifact manifest must be LF-terminated ASCII")
    try:
        lines = manifest_raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise common.ProtocolError(f"{label} artifact manifest is not ASCII") from error
    if not lines:
        raise common.ProtocolError(f"{label} artifact manifest is empty")
    rows: dict[str, str] = {}
    previous_name: str | None = None
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9_.-]*)", line)
        if match is None:
            raise common.ProtocolError(f"{label} artifact manifest row is malformed")
        digest, name = match.groups()
        if name == _AUDIT_MANIFEST_BASENAME or name in rows:
            raise common.ProtocolError(f"{label} artifact manifest duplicate/self member")
        if previous_name is not None and name <= previous_name:
            raise common.ProtocolError(f"{label} artifact manifest rows are not strictly sorted")
        previous_name = name
        rows[name] = digest
    if _AUDIT_RECEIPT_BASENAME not in rows:
        raise common.ProtocolError(f"{label} receipt is absent from artifact manifest")

    audit_root = checked_manifest.parent
    actual: dict[str, Path] = {}
    for entry in audit_root.iterdir():
        info = common.lstat_or_none(entry)
        if info is None or not stat.S_ISREG(info.st_mode):
            raise common.ProtocolError(f"{label} package contains a non-regular member")
        actual[entry.name] = entry
    expected_names = set(rows) | {_AUDIT_MANIFEST_BASENAME}
    if set(actual) != expected_names:
        raise common.ProtocolError(f"{label} artifact manifest does not close the package")

    member_bytes: dict[str, bytes] = {}
    for name, digest in rows.items():
        raw = _read_regular_once(actual[name], f"{label} package member {name}")
        if common.sha256_bytes(raw) != digest:
            raise common.ProtocolError(f"{label} package member hash mismatch: {name}")
        member_bytes[name] = raw
    if common.absolute_unresolved(actual[_AUDIT_RECEIPT_BASENAME]) != common.absolute_unresolved(
        checked_receipt
    ):
        raise common.ProtocolError(f"{label} receipt path is not the manifested receipt")
    return member_bytes


def _audit_target_binding(package_manifest_sha256: str) -> dict[str, str]:
    return {
        "artifact_manifest_sha256": package_manifest_sha256,
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "overlay_sha256": common.sha256_file(common.PACKAGE_DIR / "overlay.py"),
        "parity_sha256": common.sha256_file(common.PACKAGE_DIR / "parity.py"),
    }


def _verify_authorized_audit_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    expected_schema: str,
    expected_status: str,
    package_manifest_sha256: str,
    calibration_sha256: str | None,
    calibration_internal_sha256: str | None,
    source_audit_manifest_sha256: str | None,
    source_audit_receipt_sha256: str | None,
) -> dict[str, Any]:
    """Authenticate one exact independent-audit manifest and receipt pair."""
    if not isinstance(binding, Mapping):
        raise common.ProtocolError(f"{label} authorization binding must be an object")
    common.strict_keys(
        binding,
        (
            "manifest_path", "manifest_sha256", "receipt_path", "receipt_sha256",
            "receipt_internal_sha256",
        ),
        f"{label} authorization binding",
    )
    for key in ("manifest_sha256", "receipt_sha256", "receipt_internal_sha256"):
        if not isinstance(binding[key], str) or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None:
            raise common.ProtocolError(f"{label} authorization has malformed {key}")
    if not isinstance(binding["manifest_path"], str) or not isinstance(binding["receipt_path"], str):
        raise common.ProtocolError(f"{label} authorization paths must be strings")
    manifest_path = Path(binding["manifest_path"])
    receipt_path = Path(binding["receipt_path"])
    if not manifest_path.is_absolute() or not receipt_path.is_absolute():
        raise common.ProtocolError(f"{label} authorization paths must be absolute")
    manifest_raw = _read_regular_once(manifest_path, f"{label} artifact manifest")
    if common.sha256_bytes(manifest_raw) != binding["manifest_sha256"]:
        raise common.ProtocolError(f"{label} artifact-manifest hash mismatch")
    members = _audit_manifest_closure(
        manifest_path, receipt_path, manifest_raw, label=label
    )
    receipt_raw = members[_AUDIT_RECEIPT_BASENAME]
    if common.sha256_bytes(receipt_raw) != binding["receipt_sha256"]:
        raise common.ProtocolError(f"{label} receipt file hash mismatch")
    value = _json_object(receipt_raw, f"{label} receipt")
    top_keys = (
        "schema", "status", "audited_target", "verification", "access_attestation",
        "authorization", "audit_receipt_sha256",
    )
    if label == "calibration audit":
        top_keys = (*top_keys[:-2], "bindings", *top_keys[-2:])
    common.strict_keys(value, top_keys, f"{label} receipt")
    if value["schema"] != expected_schema or value["status"] != expected_status:
        raise common.ProtocolError(f"{label} receipt schema/status mismatch")
    normalized = dict(value)
    observed_internal = normalized.pop("audit_receipt_sha256")
    expected_internal = common.sha256_bytes(common.canonical_json_bytes(normalized))
    if observed_internal != expected_internal:
        raise common.ProtocolError(f"{label} receipt internal hash mismatch")
    if expected_internal != binding["receipt_internal_sha256"]:
        raise common.ProtocolError(f"{label} receipt internal binding mismatch")
    audited = value["audited_target"]
    if not isinstance(audited, Mapping):
        raise common.ProtocolError(f"{label} receipt audited_target must be an object")
    common.strict_keys(audited, _AUDITED_TARGET_KEYS, f"{label} audited_target")
    if dict(audited) != _audit_target_binding(package_manifest_sha256):
        raise common.ProtocolError(f"{label} receipt target binding mismatch")
    verification = value["verification"]
    if not isinstance(verification, Mapping):
        raise common.ProtocolError(f"{label} verification must be an object")
    common.strict_keys(verification, _AUDIT_VERIFICATION, f"{label} verification")
    if dict(verification) != _AUDIT_VERIFICATION:
        raise common.ProtocolError(f"{label} verification semantics mismatch")
    access_attestation = value["access_attestation"]
    if not isinstance(access_attestation, Mapping):
        raise common.ProtocolError(f"{label} access_attestation must be an object")
    common.strict_keys(access_attestation, _AUDIT_ACCESS, f"{label} access_attestation")
    if dict(access_attestation) != _AUDIT_ACCESS:
        raise common.ProtocolError(f"{label} access attestation is broader than permitted")
    authorization = value["authorization"]
    if not isinstance(authorization, Mapping):
        raise common.ProtocolError(f"{label} receipt authorization must be an object")
    if label == "source audit":
        common.strict_keys(
            authorization, _SOURCE_AUDIT_AUTHORIZATION, "source audit authorization"
        )
        if dict(authorization) != _SOURCE_AUDIT_AUTHORIZATION:
            raise common.ProtocolError("source audit authorization semantics mismatch")
    else:
        common.strict_keys(
            authorization, _CALIBRATION_AUDIT_AUTHORIZATION,
            "calibration audit authorization",
        )
        if dict(authorization) != _CALIBRATION_AUDIT_AUTHORIZATION:
            raise common.ProtocolError("calibration audit authorization semantics mismatch")
        bindings = value["bindings"]
        if not isinstance(bindings, Mapping):
            raise common.ProtocolError("calibration audit bindings must be an object")
        common.strict_keys(
            bindings,
            (
                "calibration_receipt_sha256", "calibration_receipt_internal_sha256",
                "source_audit_manifest_sha256", "source_audit_receipt_sha256",
            ),
            "calibration audit bindings",
        )
        expected_bindings = {
            "calibration_receipt_sha256": calibration_sha256,
            "calibration_receipt_internal_sha256": calibration_internal_sha256,
            "source_audit_manifest_sha256": source_audit_manifest_sha256,
            "source_audit_receipt_sha256": source_audit_receipt_sha256,
        }
        if None in expected_bindings.values() or dict(bindings) != expected_bindings:
            raise common.ProtocolError("calibration audit did not authorize the bound Qwen run")
    manifest_raw_after = _read_regular_once(
        manifest_path, f"{label} artifact manifest post-semantic reauthentication"
    )
    if manifest_raw_after != manifest_raw:
        raise common.ProtocolError(f"{label} artifact manifest changed during authentication")
    members_after = _audit_manifest_closure(
        manifest_path, receipt_path, manifest_raw_after, label=f"{label} post-semantic"
    )
    if members_after != members:
        raise common.ProtocolError(f"{label} package changed during authentication")
    return {
        "manifest_path": str(common.absolute_unresolved(manifest_path)),
        "manifest_sha256": str(binding["manifest_sha256"]),
        "receipt_path": str(common.absolute_unresolved(receipt_path)),
        "receipt_sha256": str(binding["receipt_sha256"]),
        "receipt_internal_sha256": str(binding["receipt_internal_sha256"]),
        "receipt_schema": expected_schema,
        "receipt_status": expected_status,
        "manifest_member_count": len(members),
        "manifest_closure_verified": True,
        "manifest_closure_reauthenticated_after_receipt_semantics": True,
        "receipt_internal_sha256_recomputed": True,
    }


def _load_production_authorization(
    authorization_path: Path,
    lock: Mapping[str, Any],
    *,
    workspace_root: Path,
    aux_dir: Path,
    output_dir: Path,
    calibration_path: Path,
    source_trace_path: Path,
    v4_run_root: Path,
    v4_result_audit_path: Path,
    calibration: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Verify the versioned, content-bound, one-shot production authorization."""
    # This check is deliberately lexical and performs no filesystem operation.
    # In particular it rejects LINK/../target before abspath can erase LINK.
    supplied_paths = {
        "production authorization": authorization_path,
        "workspace root": workspace_root,
        "auxiliary directory": aux_dir,
        "production output directory": output_dir,
        "calibration receipt": calibration_path,
        "source trace": source_trace_path,
        "v4 run root": v4_run_root,
        "v4 result audit": v4_result_audit_path,
    }
    for supplied_label, supplied_path in supplied_paths.items():
        common.reject_parent_traversal(supplied_path, supplied_label)
    protocol = lock["authorization_protocol"]
    if common.absolute_unresolved(authorization_path) != common.absolute_unresolved(
        Path(str(protocol["receipt_path"]))
    ):
        raise common.ProtocolError("production authorization path differs from frozen v5 path")
    for legacy in protocol["forbidden_legacy_sentinel_paths"]:
        legacy_path = common.absolute_unresolved(Path(str(legacy)))
        if common.lstat_or_none(legacy_path) is not None:
            raise common.ProtocolError(
                f"legacy existence-only launch sentinel must be absent: {legacy_path}"
            )
    raw = _read_regular_once(authorization_path, "v5 production authorization receipt")
    value = _json_object(raw, "v5 production authorization receipt")
    common.strict_keys(
        value,
        (
            "schema", "status", "authorization_receipt_sha256", "authorization_nonce",
            "created_unix_ns", "action", "one_shot", "package", "source_audit",
            "calibration", "calibration_audit", "run", "path_boundary",
            "gpu_runtime_authorized", "qwen_manifest_or_payload_access_authorized",
        ),
        "v5 production authorization receipt",
    )
    if value["schema"] != protocol["receipt_schema"] or value["status"] != protocol["receipt_status"]:
        raise common.ProtocolError("production authorization schema/status mismatch")
    if value["action"] != protocol["action"] or value["one_shot"] is not True:
        raise common.ProtocolError("production authorization action/one-shot mismatch")
    if (
        value["gpu_runtime_authorized"] is not True
        or value["qwen_manifest_or_payload_access_authorized"] is not True
    ):
        raise common.ProtocolError("production authorization does not permit the requested phase")
    if (
        not isinstance(value["authorization_nonce"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["authorization_nonce"]) is None
        or not isinstance(value["created_unix_ns"], int)
        or isinstance(value["created_unix_ns"], bool)
        or value["created_unix_ns"] <= 0
    ):
        raise common.ProtocolError("production authorization nonce/timestamp malformed")
    normalized = dict(value)
    observed_internal = normalized.pop("authorization_receipt_sha256")
    expected_internal = common.sha256_bytes(common.canonical_json_bytes(normalized))
    if observed_internal != expected_internal:
        raise common.ProtocolError("production authorization internal hash mismatch")

    package_manifest = common.PACKAGE_DIR / "ARTIFACT_SHA256SUMS.txt"
    package_manifest_raw = _read_regular_once(package_manifest, "v5 package artifact manifest")
    package_manifest_sha256 = common.sha256_bytes(package_manifest_raw)
    expected_package = {
        "artifact_manifest_sha256": package_manifest_sha256,
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "overlay_sha256": common.sha256_file(common.PACKAGE_DIR / "overlay.py"),
        "parity_sha256": common.sha256_file(common.PACKAGE_DIR / "parity.py"),
    }
    if value["package"] != expected_package:
        raise common.ProtocolError("production authorization package binding mismatch")

    calibration_file = common.require_regular_file_before_resolve(
        calibration_path, "authorization-bound calibration receipt"
    )
    source_trace_file = common.require_regular_file_before_resolve(
        source_trace_path, "authorization-bound source trace"
    )
    expected_calibration = {
        "receipt_path": str(common.absolute_unresolved(calibration_path)),
        "receipt_sha256": common.sha256_file(calibration_file),
        "receipt_internal_sha256": calibration["receipt_sha256"],
        "source_trace_path": str(common.absolute_unresolved(source_trace_path)),
        "source_trace_sha256": common.sha256_file(source_trace_file),
    }
    if value["calibration"] != expected_calibration:
        raise common.ProtocolError("production authorization calibration binding mismatch")
    expected_run = {
        "workspace_root": str(common.absolute_unresolved(workspace_root)),
        "aux_dir": str(common.absolute_unresolved(aux_dir)),
        "output_dir": str(common.absolute_unresolved(output_dir)),
        "v4_run_root": str(common.absolute_unresolved(v4_run_root)),
        "v4_result_audit": str(common.absolute_unresolved(v4_result_audit_path)),
        "v4_topk_authentication_receipt_sha256": lock["v4_reuse"][
            "expected_authentication_receipt_sha256"
        ],
        "resume_same_output_permitted": True,
    }
    if value["run"] != expected_run:
        raise common.ProtocolError("production authorization run-path binding mismatch")

    source_audit = _verify_authorized_audit_binding(
        value["source_audit"],
        label="source audit",
        expected_schema=protocol["source_audit_receipt_schema"],
        expected_status=protocol["source_audit_required_status"],
        package_manifest_sha256=package_manifest_sha256,
        calibration_sha256=None,
        calibration_internal_sha256=None,
        source_audit_manifest_sha256=None,
        source_audit_receipt_sha256=None,
    )
    calibration_audit = _verify_authorized_audit_binding(
        value["calibration_audit"],
        label="calibration audit",
        expected_schema=protocol["calibration_audit_receipt_schema"],
        expected_status=protocol["calibration_audit_required_status"],
        package_manifest_sha256=package_manifest_sha256,
        calibration_sha256=expected_calibration["receipt_sha256"],
        calibration_internal_sha256=expected_calibration["receipt_internal_sha256"],
        source_audit_manifest_sha256=source_audit["manifest_sha256"],
        source_audit_receipt_sha256=source_audit["receipt_sha256"],
    )
    boundary_guard = common.BoundaryGuard(
        "QWEN_AUX_33_DOMAIN_SINGLE_RUN",
        outputs=(("production output/run root", output_dir, "directory", resume),),
        inputs=(
            ("workspace input root", workspace_root, "directory"),
            ("auxiliary input root", aux_dir, "directory"),
            ("calibration receipt input", calibration_path, "file"),
            ("source-trace receipt input", source_trace_path, "file"),
            ("grouped-v4 run input root", v4_run_root, "directory"),
            ("grouped-v4 result audit input", v4_result_audit_path, "file"),
            ("production authorization input", authorization_path, "file"),
            ("source-audit input root", Path(source_audit["manifest_path"]).parent, "directory"),
            ("source-audit manifest input", Path(source_audit["manifest_path"]), "file"),
            ("source-audit receipt input", Path(source_audit["receipt_path"]), "file"),
            (
                "calibration-audit input root",
                Path(calibration_audit["manifest_path"]).parent,
                "directory",
            ),
            (
                "calibration-audit manifest input",
                Path(calibration_audit["manifest_path"]),
                "file",
            ),
            (
                "calibration-audit receipt input",
                Path(calibration_audit["receipt_path"]),
                "file",
            ),
        ),
    )
    if value["path_boundary"] != boundary_guard.authorization_receipt():
        raise common.ProtocolError("production authorization path-boundary binding mismatch")
    return {
        "path": str(common.absolute_unresolved(authorization_path)),
        "file_sha256": common.sha256_bytes(raw),
        "internal_sha256": expected_internal,
        "authorization_nonce": value["authorization_nonce"],
        "created_unix_ns": value["created_unix_ns"],
        "package_manifest_sha256": package_manifest_sha256,
        "source_audit": source_audit,
        "calibration_audit": calibration_audit,
        "run": expected_run,
        "calibration": expected_calibration,
        "boundary_guard": boundary_guard,
    }


def _expected_run_header(
    aux_dir: Path,
    calibration_path: Path,
    source_trace_path: Path,
    v4_run_root: Path,
    v4_result_audit_path: Path,
    v4_topk_receipt: Mapping[str, Any],
    calibration: Mapping[str, Any],
    authorization: Mapping[str, Any],
    path_boundary: Mapping[str, Any],
    workspace_aux_closure: Mapping[str, Any],
    stage0_plan,
    full_plan,
) -> dict[str, Any]:
    return {
        "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_run_header_v5",
        "status": "IMMUTABLE_STATE_HEADER",
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "parity_sha256": common.sha256_file(common.PACKAGE_DIR / "parity.py"),
        "overlay_sha256": common.sha256_file(common.PACKAGE_DIR / "overlay.py"),
        "frozen_search_engine_sha256": EXPECTED_TIER_B_GATE_SHA256,
        "tier_b_common_sha256": common.EXPECTED_TIER_B_COMMON_SHA256,
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": common.sha256_file(calibration_path.resolve()),
        "calibration_output_sentinel_sha256": calibration["output_sentinel_sha256_f32le"],
        "source_trace_path": str(source_trace_path.resolve()),
        "source_trace_sha256": common.sha256_file(source_trace_path.resolve()),
        "v4_run_root": str(v4_run_root.resolve()),
        "v4_result_audit_path": str(v4_result_audit_path.resolve()),
        "v4_topk_authentication_receipt_sha256": v4_topk_receipt["receipt_sha256"],
        "auxiliary_directory": str(aux_dir.resolve()),
        "stage0_plan_sha256": common.plan_sha256(stage0_plan),
        "full_plan_sha256": common.plan_sha256(full_plan),
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "new_effective_candidate_count": common.NEW_EFFECTIVE_CANDIDATES,
        "full_union_effective_candidate_count": common.FULL_EFFECTIVE_CANDIDATES,
        "domain_ids": list(common.DOMAIN_IDS),
        "equivalence_map_sha256": common.equivalence_map_sha256(),
        "production_authorization_path": authorization["path"],
        "production_authorization_file_sha256": authorization["file_sha256"],
        "production_authorization_internal_sha256": authorization["internal_sha256"],
        "source_audit_manifest_sha256": authorization["source_audit"]["manifest_sha256"],
        "calibration_audit_manifest_sha256": authorization["calibration_audit"]["manifest_sha256"],
        "path_boundary": dict(path_boundary),
        "workspace_aux_closure": dict(workspace_aux_closure),
    }


def _validate_domain_topk_arrays(
    ordinals: np.ndarray, metrics: np.ndarray, *, label: str
) -> None:
    """Validate the exact 33x2048 metric/ordinal total-order representation."""
    expected_shape = (len(common.DOMAIN_IDS), common.STAGE0_TOP_K)
    if not isinstance(ordinals, np.ndarray) or ordinals.dtype != np.dtype(np.uint64):
        raise common.ProtocolError(f"{label} ordinals dtype mismatch")
    if not isinstance(metrics, np.ndarray) or metrics.dtype != np.dtype(np.float64):
        raise common.ProtocolError(f"{label} metrics dtype mismatch")
    if ordinals.shape != expected_shape or metrics.shape != expected_shape:
        raise common.ProtocolError(f"{label} shape mismatch")
    if not np.all(np.isfinite(metrics)):
        raise common.ProtocolError(f"{label} contains non-finite metrics")
    wanted_order = np.arange(common.STAGE0_TOP_K)
    for domain_index in range(len(common.DOMAIN_IDS)):
        row_ordinals = ordinals[domain_index]
        row_metrics = metrics[domain_index]
        if len(np.unique(row_ordinals)) != common.STAGE0_TOP_K:
            raise common.ProtocolError(f"{label} contains duplicate ordinals")
        if not np.array_equal(np.lexsort((row_ordinals, row_metrics)), wanted_order):
            raise common.ProtocolError(f"{label} violates stable metric/ordinal order")


def _assert_new_family_membership(
    ordinals: np.ndarray, seed_start: int, seed_stop: int, *, label: str
) -> None:
    """Vectorized proof that every ordinal is canonical and in the new family."""
    if not (0 <= seed_start < seed_stop <= common.STORED_SEED_COUNT):
        raise common.ProtocolError(f"{label} has an invalid seed interval")
    if not isinstance(ordinals, np.ndarray) or ordinals.dtype != np.dtype(np.uint64):
        raise common.ProtocolError(f"{label} ordinals dtype mismatch")
    value = ordinals.reshape(-1).copy()
    if np.any(value >= np.uint64(common.LOGICAL_CANDIDATES)):
        raise common.ProtocolError(f"{label} contains an out-of-range ordinal")
    abi = value % np.uint64(2); value //= np.uint64(2)
    half = value % np.uint64(2); value //= np.uint64(2)
    assignment = value % np.uint64(2); value //= np.uint64(2)
    etp = value % np.uint64(4); value //= np.uint64(4)
    ep = value % np.uint64(8); value //= np.uint64(8)
    pp = value % np.uint64(10); seed = value // np.uint64(10)
    del half
    endpoint = (ep == 0) | (ep == 7)
    canonical = (abi == 0) & (~endpoint | (assignment == 0))
    new_pp = (pp == 4) | (pp == 6) | (pp == 7) | (pp == 8) | (pp == 9)
    new_etp = ((pp == 0) | (pp == 3) | (pp == 5)) & (etp == 3)
    in_new_family = canonical & (new_pp | new_etp)
    in_seed_interval = (seed >= np.uint64(seed_start)) & (seed < np.uint64(seed_stop))
    if not bool(np.all(in_new_family & in_seed_interval)):
        raise common.ProtocolError(
            f"{label} contains stale, retained-v4, noncanonical, or out-of-family ordinals"
        )


def _validate_stage0_shard_state(
    state: Mapping[str, np.ndarray], shard: int
) -> tuple[np.ndarray, np.ndarray]:
    label = f"stage0 shard {shard:03d}"
    if set(state) != {"seed_start", "seed_stop", "top_ordinals", "top_q"}:
        raise common.ProtocolError(f"{label} NPZ member set mismatch")
    seed_start = shard * common.SEED_SHARD_SIZE
    seed_stop = seed_start + common.SEED_SHARD_SIZE
    for key, wanted in (("seed_start", seed_start), ("seed_stop", seed_stop)):
        value = state[key]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(np.int32)
            or value.shape != (1,)
            or int(value[0]) != wanted
        ):
            raise common.ProtocolError(f"{label} {key} is stale or malformed")
    ordinals = state["top_ordinals"]
    metrics = state["top_q"]
    _validate_domain_topk_arrays(ordinals, metrics, label=label)
    _assert_new_family_membership(ordinals, seed_start, seed_stop, label=label)
    # Membership in the independently enumerated exact shard is stronger than
    # an axis predicate and binds the complete 42,205,184-anchor construction.
    expected = common.representative_ordinals(seed_start, seed_stop)
    flat = ordinals.reshape(-1)
    positions = np.searchsorted(expected, flat)
    if np.any(positions >= len(expected)) or not np.array_equal(expected[positions], flat):
        raise common.ProtocolError(f"{label} contains an ordinal outside exact enumeration")
    return ordinals, metrics


def _recompute_stage0_global(
    shards: Sequence[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(shards) != common.SEED_SHARD_COUNT:
        raise common.ProtocolError("stage0 exact recomputation requires all 256 shards")
    domain_ordinals = np.empty(
        (len(common.DOMAIN_IDS), common.STAGE0_TOP_K), dtype=np.uint64
    )
    domain_metrics = np.empty_like(domain_ordinals, dtype=np.float64)
    for domain_index in range(len(common.DOMAIN_IDS)):
        ordinals = np.concatenate([row[0][domain_index] for row in shards])
        metrics = np.concatenate([row[1][domain_index] for row in shards])
        order = np.lexsort((ordinals, metrics))[: common.STAGE0_TOP_K]
        domain_ordinals[domain_index] = ordinals[order]
        domain_metrics[domain_index] = metrics[order]
    union = np.unique(domain_ordinals.reshape(-1))
    if len(union) > len(common.DOMAIN_IDS) * common.STAGE0_TOP_K:
        raise common.ProtocolError("recomputed stage0 union exceeds frozen maximum")
    _validate_domain_topk_arrays(
        domain_ordinals, domain_metrics, label="recomputed global new-family stage0"
    )
    _assert_new_family_membership(
        domain_ordinals, 0, common.STORED_SEED_COUNT,
        label="recomputed global new-family stage0",
    )
    return domain_ordinals, domain_metrics, union


def _validate_stage0_global_state(
    state: Mapping[str, np.ndarray],
    expected_ordinals: np.ndarray,
    expected_metrics: np.ndarray,
    expected_union: np.ndarray,
) -> None:
    if set(state) != {"domain_top_ordinals", "domain_top_q", "union_ordinals"}:
        raise common.ProtocolError("global new-family stage0 NPZ member set mismatch")
    ordinals = state["domain_top_ordinals"]
    metrics = state["domain_top_q"]
    union = state["union_ordinals"]
    _validate_domain_topk_arrays(ordinals, metrics, label="global new-family stage0")
    _assert_new_family_membership(
        ordinals, 0, common.STORED_SEED_COUNT, label="global new-family stage0"
    )
    if not isinstance(union, np.ndarray) or union.dtype != np.dtype(np.uint64):
        raise common.ProtocolError("global new-family stage0 union dtype mismatch")
    if union.ndim != 1 or len(union) > len(common.DOMAIN_IDS) * common.STAGE0_TOP_K:
        raise common.ProtocolError("global new-family stage0 union shape mismatch")
    if not np.array_equal(union, np.unique(ordinals.reshape(-1))):
        raise common.ProtocolError("global new-family stage0 union is not exact")
    if (
        not np.array_equal(ordinals, expected_ordinals)
        or not np.array_equal(metrics, expected_metrics)
        or not np.array_equal(union, expected_union)
    ):
        raise common.ProtocolError("resumed global stage0 differs from exact shard recomputation")


def _compute_stage0_shard_state(
    access: kernels.PhiloxRandomAccess,
    matrices: Sequence[Any],
    experts: np.ndarray,
    roles: np.ndarray,
    coordinates: np.ndarray,
    slices: Sequence[Any],
    shard: int,
) -> dict[str, np.ndarray]:
    """Recompute one complete shard from the frozen payload-derived objective."""
    seed_start = shard * common.SEED_SHARD_SIZE
    seed_stop = seed_start + common.SEED_SHARD_SIZE
    candidate_ordinals = common.representative_ordinals(seed_start, seed_stop)
    anchors = access.generate(candidate_ordinals, experts, roles, coordinates)
    q = BASE._stage0_q(access, anchors, matrices, slices)
    _validate_full_stage0_q(access, q, len(candidate_ordinals))
    top_ordinals, top_q = BASE._exact_top_k(
        access.cp, q, candidate_ordinals, common.STAGE0_TOP_K
    )
    state = {
        "seed_start": np.asarray([seed_start], dtype=np.int32),
        "seed_stop": np.asarray([seed_stop], dtype=np.int32),
        "top_ordinals": top_ordinals,
        "top_q": top_q,
    }
    del anchors, q
    return state


def _validate_full_stage0_q(
    access: kernels.PhiloxRandomAccess, q: Any, candidate_count: int
) -> None:
    """Reject dtype/shape/nonfiniteness across every candidate before TopK."""
    cp = access.cp
    wanted_dtype = cp.dtype(cp.float64)
    if getattr(q, "dtype", None) != wanted_dtype:
        raise common.ProtocolError("complete stage0 q dtype mismatch before TopK")
    if getattr(q, "shape", None) != (candidate_count, len(common.DOMAIN_IDS)):
        raise common.ProtocolError("complete stage0 q shape mismatch before TopK")
    if not bool(cp.all(cp.isfinite(q))):
        raise common.ProtocolError("complete stage0 q contains non-finite values before TopK")


def _compare_stage0_shard_replay(
    observed: Mapping[str, np.ndarray],
    replayed: Mapping[str, np.ndarray],
    shard: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Require an existing shard to equal a fresh scientific replay exactly."""
    observed_pair = _validate_stage0_shard_state(observed, shard)
    replayed_pair = _validate_stage0_shard_state(replayed, shard)
    if set(observed) != set(replayed) or any(
        not np.array_equal(observed[key], replayed[key]) for key in replayed
    ):
        raise common.ProtocolError(
            f"stage0 shard {shard:03d} differs from exact payload-derived replay"
        )
    return replayed_pair


def _run_stage0_strict(
    access: kernels.PhiloxRandomAccess,
    journal: StateJournal,
    matrices: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Run stage0, exactly replaying every existing shard before it is trusted."""
    experts, roles, coordinates, slices = BASE._flatten_coordinate_metadata(matrices)
    if len(coordinates) != common.STAGE0_FIT + common.STAGE0_SCORE:
        raise common.ProtocolError("stage0 coordinate count mismatch")
    merged_path = journal.lookup("stage0_merged", "global")
    shards: list[tuple[np.ndarray, np.ndarray]] = []
    for shard in range(common.SEED_SHARD_COUNT):
        key = f"{shard:03d}"
        existing = journal.lookup("stage0", key)
        if merged_path is not None:
            if existing is None:
                raise common.ProtocolError("global stage0 exists before all exact shard states")
        replayed = _compute_stage0_shard_state(
            access, matrices, experts, roles, coordinates, slices, shard
        )
        replayed_pair = _validate_stage0_shard_state(replayed, shard)
        if existing is None:
            journal.write_npz("stage0", key, **replayed)
        else:
            observed = BASE._load_npz(existing)
            replayed_pair = _compare_stage0_shard_replay(observed, replayed, shard)
        shards.append(replayed_pair)
        print(
            f"grouped-v5 v5 exact stage0 replay seed shard {shard + 1}/256",
            flush=True,
        )

    expected_ordinals, expected_metrics, expected_union = _recompute_stage0_global(shards)
    if merged_path is None:
        merged_path = journal.write_npz(
            "stage0_merged",
            "global",
            domain_top_ordinals=expected_ordinals,
            domain_top_q=expected_metrics,
            union_ordinals=expected_union,
        )
    merged = BASE._load_npz(merged_path)
    _validate_stage0_global_state(
        merged, expected_ordinals, expected_metrics, expected_union
    )
    return expected_ordinals, expected_union


def _assert_full_family_membership(ordinals: np.ndarray, *, label: str) -> None:
    """Require canonical membership in the exact 58,720,256-anchor full family."""
    if not isinstance(ordinals, np.ndarray) or ordinals.dtype != np.dtype(np.uint64):
        raise common.ProtocolError(f"{label} ordinal dtype mismatch")
    value = ordinals.reshape(-1).copy()
    if np.any(value >= np.uint64(common.LOGICAL_CANDIDATES)):
        raise common.ProtocolError(f"{label} contains an out-of-range ordinal")
    abi = value % np.uint64(2); value //= np.uint64(2)
    value //= np.uint64(2)  # canonical fc1 half remains an independent axis
    assignment = value % np.uint64(2); value //= np.uint64(2)
    value //= np.uint64(4)  # all four ETP values are represented in the full family
    ep = value % np.uint64(8); value //= np.uint64(8)
    pp = value % np.uint64(10)
    endpoint = (ep == 0) | (ep == 7)
    effective_pp = (
        (pp == 0) | (pp == 3) | (pp == 4) | (pp == 5)
        | (pp == 6) | (pp == 7) | (pp == 8) | (pp == 9)
    )
    canonical = (abi == 0) & (~endpoint | (assignment == 0)) & effective_pp
    if not bool(np.all(canonical)):
        raise common.ProtocolError(f"{label} contains a noncanonical/out-of-family ordinal")


def _validate_stage1_batch_state(
    state: Mapping[str, np.ndarray], expected_ordinals: np.ndarray, *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate an exact stage-1 batch representation before comparison/use."""
    if set(state) != {"ordinals", "q"}:
        raise common.ProtocolError(f"{label} NPZ member set mismatch")
    ordinals = state["ordinals"]
    q = state["q"]
    if (
        not isinstance(ordinals, np.ndarray)
        or ordinals.dtype != np.dtype(np.uint64)
        or ordinals.shape != expected_ordinals.shape
        or not np.array_equal(ordinals, expected_ordinals)
    ):
        raise common.ProtocolError(f"{label} ordinal identity/dtype/shape mismatch")
    _assert_full_family_membership(ordinals, label=label)
    expected_q_shape = (len(expected_ordinals), len(common.DOMAIN_IDS))
    if (
        not isinstance(q, np.ndarray)
        or q.dtype != np.dtype(np.float64)
        or q.shape != expected_q_shape
        or not np.all(np.isfinite(q))
    ):
        raise common.ProtocolError(f"{label} q dtype/shape/finiteness mismatch")
    return ordinals, q


def _validate_stage1_winner_state(
    state: Mapping[str, np.ndarray],
    union_ordinals: np.ndarray,
    expected_ordinals: np.ndarray,
    expected_q: np.ndarray,
) -> None:
    """Validate and compare the global winner state to exact batch replay."""
    if set(state) != {"winner_ordinals", "winner_q"}:
        raise common.ProtocolError("stage1 winner NPZ member set mismatch")
    ordinals = state["winner_ordinals"]
    q = state["winner_q"]
    expected_shape = (len(common.DOMAIN_IDS),)
    if (
        not isinstance(ordinals, np.ndarray)
        or ordinals.dtype != np.dtype(np.uint64)
        or ordinals.shape != expected_shape
    ):
        raise common.ProtocolError("stage1 winner ordinal dtype/shape mismatch")
    if (
        not isinstance(q, np.ndarray)
        or q.dtype != np.dtype(np.float64)
        or q.shape != expected_shape
        or not np.all(np.isfinite(q))
    ):
        raise common.ProtocolError("stage1 winner q dtype/shape/finiteness mismatch")
    _assert_full_family_membership(ordinals, label="stage1 winners")
    positions = np.searchsorted(union_ordinals, ordinals)
    if (
        np.any(positions >= len(union_ordinals))
        or not np.array_equal(union_ordinals[positions], ordinals)
    ):
        raise common.ProtocolError("stage1 winner is absent from the exact overlay union")
    if not np.array_equal(ordinals, expected_ordinals) or not np.array_equal(q, expected_q):
        raise common.ProtocolError("stage1 winners differ from exact batch replay")


def _run_stage1_strict(
    access: kernels.PhiloxRandomAccess,
    journal: StateJournal,
    matrices: Sequence[Any],
    union_ordinals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly replay every stage-1 score batch and independently rerank winners."""
    if (
        not isinstance(union_ordinals, np.ndarray)
        or union_ordinals.dtype != np.dtype(np.uint64)
        or union_ordinals.ndim != 1
        or len(union_ordinals) == 0
        or len(union_ordinals) > len(common.DOMAIN_IDS) * 2 * common.STAGE0_TOP_K
        or (len(union_ordinals) > 1 and not np.all(union_ordinals[1:] > union_ordinals[:-1]))
    ):
        raise common.ProtocolError("stage1 overlay union dtype/shape/order mismatch")
    _assert_full_family_membership(union_ordinals, label="stage1 overlay union")
    experts, roles, coordinates, slices = BASE._flatten_coordinate_metadata(matrices)
    if len(coordinates) != 48_624:
        raise common.ProtocolError(
            f"selection full-coordinate count {len(coordinates)} != 48624"
        )
    batch_size = 512
    batch_count = (len(union_ordinals) + batch_size - 1) // batch_size
    recorded_stage1_keys = [
        event["key"] for event in journal.events_list if event["kind"] == "stage1"
    ]
    expected_recorded_prefix = [
        f"{index:04d}" for index in range(len(recorded_stage1_keys))
    ]
    if (
        len(recorded_stage1_keys) > batch_count
        or recorded_stage1_keys != expected_recorded_prefix
    ):
        raise common.ProtocolError(
            "recorded stage1 batches are not an exact prefix of the replayed union"
        )
    winner_ordinals = np.zeros(len(common.DOMAIN_IDS), dtype=np.uint64)
    winner_q = np.full(len(common.DOMAIN_IDS), np.inf, dtype=np.float64)
    winners_path = journal.lookup("stage1_winners", "global")
    if winners_path is not None and len(recorded_stage1_keys) != batch_count:
        raise common.ProtocolError(
            "stage1 winners exist without the exact complete replayed batch set"
        )
    for batch_index in range(batch_count):
        key = f"{batch_index:04d}"
        start = batch_index * batch_size
        stop = min(len(union_ordinals), start + batch_size)
        ordinals = union_ordinals[start:stop]
        existing = journal.lookup("stage1", key)
        if existing is None and winners_path is not None:
            raise common.ProtocolError("stage1 winners exist before every exact batch state")
        anchors = access.generate(ordinals, experts, roles, coordinates)
        q_device = BASE._stage1_q(access, anchors, matrices, slices)
        q = np.asarray(access.cp.asnumpy(q_device))
        replayed = {"ordinals": np.asarray(ordinals, dtype=np.uint64), "q": q}
        _, replayed_q = _validate_stage1_batch_state(
            replayed, ordinals, label=f"replayed stage1 batch {key}"
        )
        if existing is None:
            journal.write_npz("stage1", key, **replayed)
        else:
            observed = BASE._load_npz(existing)
            _, observed_q = _validate_stage1_batch_state(
                observed, ordinals, label=f"resumed stage1 batch {key}"
            )
            if not np.array_equal(observed_q, replayed_q):
                raise common.ProtocolError(
                    f"stage1 batch {key} differs from exact payload-derived replay"
                )
        for domain_index in range(len(common.DOMAIN_IDS)):
            column = replayed_q[:, domain_index]
            order = np.lexsort((ordinals, column))
            local = int(order[0])
            pair = (float(column[local]), int(ordinals[local]))
            best = (float(winner_q[domain_index]), int(winner_ordinals[domain_index]))
            if pair < best:
                winner_q[domain_index] = pair[0]
                winner_ordinals[domain_index] = pair[1]
        print(
            f"grouped-v5 v5 exact stage1 replay batch {batch_index + 1}/{batch_count}",
            flush=True,
        )
        del anchors, q_device
    if not np.all(np.isfinite(winner_q)):
        raise common.ProtocolError("exact stage1 replay did not produce finite winners")
    replayed_winners = {
        "winner_ordinals": winner_ordinals,
        "winner_q": winner_q,
    }
    if winners_path is None:
        journal.write_npz("stage1_winners", "global", **replayed_winners)
    else:
        observed_winners = BASE._load_npz(winners_path)
        _validate_stage1_winner_state(
            observed_winners, union_ordinals, winner_ordinals, winner_q
        )
    _validate_stage1_winner_state(
        replayed_winners, union_ordinals, winner_ordinals, winner_q
    )
    return winner_ordinals, winner_q


def _run_overlay_merge(
    journal: StateJournal,
    authenticated_v4: overlay.AuthenticatedV4TopK,
    new_domain_ordinals: np.ndarray,
    new_domain_metrics: np.ndarray,
) -> overlay.OverlayMerge:
    """State-back the exact 2K-old plus 2K-new lists and their stage-1 union."""
    _validate_domain_topk_arrays(
        new_domain_ordinals, new_domain_metrics, label="overlay input new-family stage0"
    )
    _assert_new_family_membership(
        new_domain_ordinals, 0, common.STORED_SEED_COUNT,
        label="overlay input new-family stage0",
    )
    expected = overlay.merge_topk(
        authenticated_v4.translated_ordinals,
        authenticated_v4.metrics,
        np.asarray(new_domain_ordinals, dtype=np.uint64),
        np.asarray(new_domain_metrics, dtype=np.float64),
    )
    state_path = journal.lookup("layout_overlay_merged", "global")
    receipt_path = journal.lookup("layout_overlay_receipt", "global")
    if state_path is None:
        if receipt_path is not None:
            raise common.ProtocolError("overlay receipt exists without merged state")
        state_path = journal.write_npz(
            "layout_overlay_merged",
            "global",
            domain_ordinals=expected.domain_ordinals,
            domain_metrics=expected.domain_metrics,
            union_ordinals=expected.union_ordinals,
        )
        receipt_value = {
            "schema": "qwen3_tier_c_grouped_v5_state_backed_overlay_receipt_v5",
            "v4_authentication": authenticated_v4.receipt,
            "merge": expected.receipt,
            "merged_state_sha256": common.sha256_file(state_path),
            "merged_state_bytes": state_path.stat().st_size,
        }
        receipt_value["receipt_sha256"] = common.sha256_bytes(
            common.canonical_json_bytes(receipt_value)
        )
        journal.write_json("layout_overlay_receipt", "global", receipt_value)
        return expected
    state = BASE._load_npz(state_path)
    if set(state) != {"domain_ordinals", "domain_metrics", "union_ordinals"}:
        raise common.ProtocolError("resumed overlay state member mismatch")
    _validate_domain_topk_arrays(
        state["domain_ordinals"], state["domain_metrics"],
        label="resumed old/new overlay",
    )
    resumed_union = state["union_ordinals"]
    if (
        not isinstance(resumed_union, np.ndarray)
        or resumed_union.dtype != np.dtype(np.uint64)
        or resumed_union.ndim != 1
        or len(resumed_union) == 0
        or len(resumed_union) > len(common.DOMAIN_IDS) * 2 * common.STAGE0_TOP_K
        or (
            len(resumed_union) > 1
            and not np.all(resumed_union[1:] > resumed_union[:-1])
        )
    ):
        raise common.ProtocolError("resumed overlay union dtype/shape/order mismatch")
    _assert_full_family_membership(
        state["domain_ordinals"], label="resumed overlay domain lists"
    )
    _assert_full_family_membership(resumed_union, label="resumed overlay union")
    if (
        not np.array_equal(state["domain_ordinals"], expected.domain_ordinals)
        or not np.array_equal(state["domain_metrics"], expected.domain_metrics)
        or not np.array_equal(state["union_ordinals"], expected.union_ordinals)
    ):
        raise common.ProtocolError("resumed overlay state differs from exact recomputation")
    if receipt_path is None:
        raise common.ProtocolError("resumed overlay state lacks its receipt")
    receipt_value = _json_object(
        _read_regular_once(receipt_path, "resumed overlay receipt"),
        "resumed overlay receipt",
    )
    common.strict_keys(
        receipt_value,
        (
            "schema", "v4_authentication", "merge", "merged_state_sha256",
            "merged_state_bytes", "receipt_sha256",
        ),
        "resumed overlay receipt",
    )
    normalized = dict(receipt_value)
    observed = normalized.pop("receipt_sha256", None)
    if observed != common.sha256_bytes(common.canonical_json_bytes(normalized)):
        raise common.ProtocolError("resumed overlay receipt internal hash mismatch")
    if (
        receipt_value.get("schema")
        != "qwen3_tier_c_grouped_v5_state_backed_overlay_receipt_v5"
        or receipt_value.get("v4_authentication") != authenticated_v4.receipt
        or receipt_value.get("merge") != expected.receipt
        or receipt_value.get("merged_state_sha256") != common.sha256_file(state_path)
        or receipt_value.get("merged_state_bytes") != state_path.stat().st_size
    ):
        raise common.ProtocolError("resumed overlay receipt differs from exact recomputation")
    return expected


def _verify_or_create_header_strict(
    journal: StateJournal, expected_header: Mapping[str, Any]
) -> Path:
    """Create the header once or compare strict JSON to every current binding."""
    existing = journal.lookup("run_header", "immutable")
    if existing is None:
        return journal.write_json("run_header", "immutable", expected_header)
    observed = _json_object(
        _read_regular_once(existing, "resumed immutable run header"),
        "resumed immutable run header",
    )
    if observed != expected_header:
        raise common.ProtocolError("resume run header differs from current frozen bindings")
    return existing


def _prepare_completed_result_replay(
    journal: StateJournal, result_path: Path, *, resume: bool
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Authenticate a completed target but never treat it as scientific state."""
    info = common.lstat_or_none(result_path)
    receipt_path = journal.lookup("result", "final")
    events = journal.events_list
    if info is None:
        if receipt_path is not None:
            raise common.ProtocolError("result journal entry exists without its final target")
        return None, events
    if result_path.is_symlink() or not result_path.is_file():
        raise common.ProtocolError("result output is an existing or dangling symlink/non-file")
    if not resume:
        raise common.ProtocolError("result target already exists")
    if receipt_path is None:
        raise common.ProtocolError("completed result lacks its journal receipt")
    if not events or (events[-1]["kind"], events[-1]["key"]) != ("result", "final"):
        raise common.ProtocolError("completed result receipt is not the final journal event")

    receipt = _json_object(
        _read_regular_once(receipt_path, "recorded final result receipt"),
        "recorded final result receipt",
    )
    common.strict_keys(
        receipt, ("output_basename", "sha256", "bytes"),
        "recorded final result receipt",
    )
    result_raw = _read_regular_once(result_path, "recorded final result")
    expected_receipt = {
        "output_basename": result_path.name,
        "sha256": common.sha256_bytes(result_raw),
        "bytes": len(result_raw),
    }
    if receipt != expected_receipt:
        raise common.ProtocolError("recorded final result receipt mismatch")
    value = _json_object(result_raw, "recorded final result")
    return {
        "raw": result_raw,
        "value": value,
        "receipt": receipt,
    }, events[:-1]


def _canonical_result_file_bytes(result: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise common.ProtocolError("recomputed result is not finite canonical JSON") from error


def _commit_or_verify_result(
    result_path: Path,
    journal: StateJournal,
    result: Mapping[str, Any],
    completed: Mapping[str, Any] | None,
) -> Path:
    """Create once, or compare every completed-result byte after full replay."""
    expected_raw = _canonical_result_file_bytes(result)
    if completed is not None:
        if completed.get("value") != result or completed.get("raw") != expected_raw:
            raise common.ProtocolError(
                "completed result differs from full payload-derived scientific replay"
            )
        return result_path
    journal.assert_boundary("immediately before production result create-new")
    common.write_json_create_new(result_path, result, "production result output")
    journal.write_json(
        "result",
        "final",
        {
            "output_basename": result_path.name,
            "sha256": common.sha256_file(result_path),
            "bytes": result_path.stat().st_size,
        },
    )
    return result_path


def run_gate(
    workspace_root: Path,
    aux_dir: Path,
    output_dir: Path,
    calibration_path: Path,
    source_trace_path: Path,
    v4_run_root: Path,
    v4_result_audit_path: Path,
    *,
    resume: bool,
) -> Path:
    workspace_root = common.require_canonical_absolute_spelling(
        workspace_root, "workspace root"
    )
    aux_dir = common.require_canonical_absolute_spelling(aux_dir, "auxiliary directory")
    output_dir = common.require_canonical_absolute_spelling(
        output_dir, "production output directory"
    )
    calibration_path = common.require_canonical_absolute_spelling(
        calibration_path, "production calibration receipt"
    )
    source_trace_path = common.require_canonical_absolute_spelling(
        source_trace_path, "production source trace"
    )
    v4_run_root = common.require_canonical_absolute_spelling(
        v4_run_root, "grouped-v4 run root"
    )
    v4_result_audit_path = common.require_canonical_absolute_spelling(
        v4_result_audit_path, "grouped-v4 result audit"
    )
    requested_output_dir = common.preflight_output_directory(
        output_dir, allow_existing=resume, label="production output directory"
    )
    if resume and common.lstat_or_none(requested_output_dir) is None:
        raise common.ProtocolError("--resume requires a pre-existing output directory")
    # These gates complete before output creation, CUDA initialization, or any
    # Qwen manifest/directory/payload access.
    lock = common.load_candidate_lock()
    result_path = requested_output_dir / str(lock["execution"]["output_json"])
    result_info = common.lstat_or_none(result_path)
    if result_info is not None and (
        result_path.is_symlink() or not result_path.is_file()
    ):
        raise common.ProtocolError("result output is an existing or dangling symlink/non-file")
    calibration_path = common.require_regular_file_before_resolve(
        calibration_path, "production calibration receipt"
    )
    source_trace_path = common.require_regular_file_before_resolve(
        source_trace_path, "production source trace"
    )
    calibration = _load_calibration(calibration_path, source_trace_path)
    authorization = _load_production_authorization(
        Path(str(lock["execution"]["production_authorization_receipt"])),
        lock,
        workspace_root=workspace_root,
        aux_dir=aux_dir,
        output_dir=requested_output_dir,
        calibration_path=calibration_path,
        source_trace_path=source_trace_path,
        v4_run_root=v4_run_root,
        v4_result_audit_path=v4_result_audit_path,
        calibration=calibration,
        resume=resume,
    )
    production_boundary = authorization["boundary_guard"]
    production_boundary.revalidate("before CUDA/runtime initialization")
    if os.environ.get(parity.SINGLE_PARAM_ENV) != "1":
        raise common.ProtocolError(
            f"set {parity.SINGLE_PARAM_ENV}=1 before production can initialize CUDA/TE"
        )
    access = kernels.PhiloxRandomAccess(0)
    parity_receipt = parity.run_parity(access, source_trace_path)
    if parity_receipt.get("all_required_checks_passed") is not True:
        raise common.ProtocolError("production parity did not pass")
    authenticated_v4 = overlay.authenticate_v4_topk(
        v4_run_root, v4_result_audit_path, lock
    )
    v4_run_root = common.reject_symlink_components_before_normalization(
        v4_run_root, "authenticated grouped-v4 run root", require_exists=True
    )
    v4_result_audit_path = common.require_regular_file_before_resolve(
        v4_result_audit_path, "authenticated grouped-v4 result audit"
    )
    if authenticated_v4.receipt.get("receipt_sha256") != authorization["run"][
        "v4_topk_authentication_receipt_sha256"
    ]:
        raise common.ProtocolError(
            "named v4 authentication receipt differs from production authorization"
        )
    access_log: list[dict[str, Any]] = [{
        "sequence": 0,
        "action": "content_bound_v5_production_authorization_authenticated_before_cuda_or_qwen_access",
        "authorization_file_sha256": authorization["file_sha256"],
        "source_audit_manifest_sha256": authorization["source_audit"]["manifest_sha256"],
        "calibration_audit_manifest_sha256": authorization["calibration_audit"]["manifest_sha256"],
        "manifest_lstat_or_stat_performed": False,
        "directory_enumeration_performed": False,
        "payload_lstat_or_stat_performed": False,
        "payload_open_or_byte_read_performed": False,
    }, {
        "sequence": 1,
        "action": "lexical_inode_and_mount_boundaries_passed_before_cuda_or_qwen_access",
        "path_boundary": production_boundary.receipt(),
        "manifest_lstat_or_stat_performed": False,
        "directory_enumeration_performed": False,
        "payload_lstat_or_stat_performed": False,
        "payload_open_or_byte_read_performed": False,
    }, {
        "sequence": 2,
        "action": "runtime_parity_passed_before_any_manifest_directory_or_payload_operation",
        "manifest_lstat_or_stat_performed": False,
        "directory_enumeration_performed": False,
        "payload_lstat_or_stat_performed": False,
        "payload_open_or_byte_read_performed": False,
    }, {
        "sequence": 3,
        "action": "authenticated_frozen_v4_result_audit_event_and_global_topk_before_qwen_manifest_or_payload_access",
        "v4_topk_authentication_receipt_sha256": authenticated_v4.receipt["receipt_sha256"],
        "new_qwen_manifest_lstat_or_stat_performed": False,
        "new_qwen_payload_open_or_byte_read_performed": False,
        "prior_v4_validation_statistics_used_for_candidate_selection": False,
    }]
    rows = common.load_source_rows(workspace_root, access_log)
    exclusion = common.exclusion_binding()
    paths, directory_status = common.validate_aux_directory(aux_dir, rows, access_log)
    if not paths:
        raise common.ProtocolError("authenticated auxiliary path set is empty")
    # Every payload path was checked and resolved inside validate_aux_directory;
    # retain only that canonical parent for all subsequent metadata consumers.
    aux_dir = next(iter(paths.values())).parent
    stage0_plan = common.make_plan(rows, stage0=True)
    full_plan = common.make_plan(rows, stage0=False)
    if common.plan_sha256(stage0_plan) != lock["coordinate_protocol"]["stage0_coordinate_plan_sha256"]:
        raise common.ProtocolError("stage0 coordinate plan differs from lock")
    if common.plan_sha256(full_plan) != lock["coordinate_protocol"]["full_coordinate_plan_sha256"]:
        raise common.ProtocolError("full coordinate plan differs from lock")

    workspace_aux_closure = common.revalidate_workspace_aux_closure(
        workspace_root, aux_dir, rows, paths, directory_status, access_log
    )
    production_boundary.revalidate(
        "after complete workspace/aux closure and immediately before output root create/open"
    )
    output_dir = common.ensure_output_directory(
        requested_output_dir,
        allow_existing=resume,
        label="production output directory",
    )
    production_boundary.revalidate("after output root create/open and before journal")
    journal = StateJournal(
        output_dir / str(lock["execution"]["state_directory"]),
        boundary_guard=production_boundary,
    )
    result_path = output_dir / str(lock["execution"]["output_json"])
    completed_result, result_events_before_final = _prepare_completed_result_replay(
        journal, result_path, resume=resume
    )

    header = _expected_run_header(
        aux_dir, calibration_path, source_trace_path, v4_run_root,
        v4_result_audit_path, authenticated_v4.receipt, calibration, authorization,
        production_boundary.receipt(), workspace_aux_closure,
        stage0_plan, full_plan
    )
    _verify_or_create_header_strict(journal, header)
    selection_full_plan = [row for row in full_plan if row.source.split == "candidate_selection"]
    validation_plan = [row for row in full_plan if row.source.split == "validation"]
    selection_full, selection_stage0 = BASE._load_selection_payloads_once(
        selection_full_plan, stage0_plan, paths, access_log
    )
    new_domain_ordinals, _ = _run_stage0_strict(access, journal, selection_stage0)
    new_merged_path = journal.lookup("stage0_merged", "global")
    if new_merged_path is None:
        raise common.ProtocolError("new-layout global stage-0 state is absent")
    new_merged_state = BASE._load_npz(new_merged_path)
    if set(new_merged_state) != {"domain_top_ordinals", "domain_top_q", "union_ordinals"}:
        raise common.ProtocolError("new-layout global stage-0 member set changed after validation")
    new_domain_metrics = new_merged_state["domain_top_q"]
    _validate_domain_topk_arrays(
        new_domain_ordinals, new_domain_metrics,
        label="post-recomputation new-family stage0",
    )
    overlay_merge = _run_overlay_merge(
        journal, authenticated_v4, new_domain_ordinals, new_domain_metrics
    )
    union_ordinals = overlay_merge.union_ordinals
    winner_ordinals, winner_q = _run_stage1_strict(
        access, journal, selection_full, union_ordinals
    )
    if len(winner_ordinals) != len(common.DOMAIN_IDS):
        raise common.ProtocolError("winner count differs from frozen domain count")
    winner_records = {
        domain_id: {
            "candidate": common.decode_ordinal(int(winner_ordinals[index])).to_json(),
            "selection_q": float(winner_q[index]),
        }
        for index, domain_id in enumerate(common.DOMAIN_IDS)
    }
    winner_freeze_value = {
        "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_winner_freeze_v5",
        "domain_count": len(common.DOMAIN_IDS),
        "domain_ids": list(common.DOMAIN_IDS),
        "winners": winner_records,
        "union_shortlist_count": len(union_ordinals),
        "overlay_merge_receipt_sha256": overlay_merge.receipt["receipt_sha256"],
        "validation_payload_opened": False,
    }
    frozen_winners = journal.lookup("validation_firewall", "winners_frozen")
    if frozen_winners is None:
        frozen_winners = journal.write_json(
            "validation_firewall", "winners_frozen", winner_freeze_value
        )
    else:
        observed_winner_freeze = _json_object(
            _read_regular_once(frozen_winners, "resumed winner firewall"),
            "resumed winner firewall",
        )
        if observed_winner_freeze != winner_freeze_value:
            raise common.ProtocolError("frozen winner state differs on resume")
    access_log.append(
        {
            "sequence": len(access_log),
            "action": "all_33_tier_c_winners_state_backed_before_validation_payload_access",
            "winner_freeze_sha256": common.sha256_file(frozen_winners),
        }
    )

    validation_data = BASE._load_plan_payloads(validation_plan, paths, access_log)
    selection_details = {}
    validation_details = {}
    selection_folds = {}
    validation_folds = {}
    for domain_index, domain_id in enumerate(common.DOMAIN_IDS):
        ordinal = int(winner_ordinals[domain_index])
        selection_details[domain_id] = BASE._candidate_details(
            access, ordinal, domain_index, selection_full
        )
        validation_details[domain_id] = BASE._candidate_details(
            access, ordinal, domain_index, validation_data
        )
        selection_folds[domain_id] = common.fold_statistics(selection_details[domain_id])
        validation_folds[domain_id] = common.fold_statistics(validation_details[domain_id])
    null_captures = {
        domain_id: float(validation_folds[domain_id]["pooled"]["capture"])
        for domain_id in common.NULL_DOMAIN_IDS
    }
    decision = common.make_decision(validation_folds["source"], null_captures)
    eligible_rows = [row for row in rows if not row.excluded]
    excluded_rows = [row for row in rows if row.excluded]
    if completed_result is None:
        result_events_before_final = journal.events_list
    result = {
        "schema": common.SCHEMA,
        "strict_ptq": True,
        "claim": {
            "procedural_anchor_discovery_only": True,
            "qwen_training_lineage_claimed": False,
            "tier_b_artifacts_modified": False,
            "grouped_v4_artifacts_modified": False,
            "legacy_descriptor_negative_used_as_science": False,
            "claim_boundary": lock["claim_boundary"],
        },
        "pinned_panel": {
            "external_heldout_manifest_path_resolved": False,
            "external_heldout_manifest_existence_checked": False,
            "external_heldout_manifest_lstat_or_statted": False,
            "external_heldout_manifest_opened_or_read": False,
            "access_permitted": False,
        },
        "bindings": {
            "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
            "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
            "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
            "parity_sha256": common.sha256_file(common.PACKAGE_DIR / "parity.py"),
            "overlay_sha256": common.sha256_file(common.PACKAGE_DIR / "overlay.py"),
            "frozen_search_engine_sha256": EXPECTED_TIER_B_GATE_SHA256,
            "tier_b_common_sha256": common.EXPECTED_TIER_B_COMMON_SHA256,
            "calibration_sha256": common.sha256_file(calibration_path.resolve()),
            "source_trace_sha256": common.sha256_file(source_trace_path.resolve()),
            "qwen_revision": common.QWEN_REVISION,
            "mcore_revision": common.MCORE_REVISION,
            "transformer_engine_revision": common.TE_REVISION,
            "layout_expansion_design_manifest_sha256": lock["provenance"]["layout_expansion_design_manifest_sha256"],
            "v4_result_sha256": lock["v4_reuse"]["result_sha256"],
            "v4_result_audit_sha256": lock["v4_reuse"]["result_audit"]["file_sha256"],
            "v4_merged_stage0_sha256": lock["v4_reuse"]["merged_event"]["required_fields"]["file_sha256"],
            "package_artifact_manifest_sha256": authorization["package_manifest_sha256"],
            "production_authorization_file_sha256": authorization["file_sha256"],
            "production_authorization_internal_sha256": authorization["internal_sha256"],
            "source_audit_manifest_sha256": authorization["source_audit"]["manifest_sha256"],
            "calibration_audit_manifest_sha256": authorization["calibration_audit"]["manifest_sha256"],
        },
        "backend": {
            "production": True,
            "name": "cupy_curand_projection_major_random_access_with_exact_te_parity",
            "parity": parity_receipt,
            "source_free_calibration": calibration,
        },
        "data_firewall": {
            "path_boundary": production_boundary.receipt(),
            "workspace_aux_closure": workspace_aux_closure,
            "auxiliary_directory": str(aux_dir.resolve()),
            "exclusion_binding": exclusion,
            "excluded": [
                {
                    "tensor_name": row.tensor_name,
                    "basename": row.basename,
                    "basename_observed_only_by_directory_enumeration": bool(
                        directory_status["excluded_basename_observed_during_enumeration"]
                    ),
                    "payload_lstat_or_statted": False,
                    "payload_opened_or_bytes_read": False,
                }
                for row in excluded_rows
            ],
            "eligible": [
                {
                    "tensor_name": row.tensor_name,
                    "basename": row.basename,
                    "expert": row.expert,
                    "role": row.role,
                    "split": row.split,
                    "sha256": row.sha256,
                    "bytes": row.bytes,
                }
                for row in eligible_rows
            ],
            "access_log": access_log,
            "directory_access_status": directory_status,
            "all_winners_frozen_before_validation": True,
            "prior_v4_result_and_stage0_state_were_prospectively_bound_inputs": True,
            "prior_v4_validation_statistics_used_for_new_candidate_scoring": False,
            "excluded_payloads_lstat_or_statted": 0,
            "excluded_payloads_opened_or_bytes_read": 0,
        },
        "candidate_space": {
            "logical_candidate_count": common.LOGICAL_CANDIDATES,
            "full_union_effective_candidate_count": common.FULL_EFFECTIVE_CANDIDATES,
            "retained_v4_effective_candidate_count": common.V4_EFFECTIVE_CANDIDATES,
            "new_effective_candidate_count": common.NEW_EFFECTIVE_CANDIDATES,
            "equivalence_map": common.equivalence_map_object(),
            "equivalence_map_sha256": common.equivalence_map_sha256(),
            "domain_count": len(common.DOMAIN_IDS),
            "domain_ids": list(common.DOMAIN_IDS),
        },
        "coordinates": {
            "stage0_plan_sha256": common.plan_sha256(stage0_plan),
            "full_plan_sha256": common.plan_sha256(full_plan),
            "stage0": common.plan_json(stage0_plan),
            "full": common.plan_json(full_plan),
        },
        "resume_state": {
            "run_header_sha256": common.sha256_file(journal.lookup("run_header", "immutable")),
            "winner_freeze_sha256": common.sha256_file(frozen_winners),
            "every_stage0_shard_fully_rescored_and_topk_replayed": True,
            "every_stage1_batch_fully_rescored_and_compared": True,
            "global_stage1_winners_independently_reranked_and_compared": True,
            "event_count_before_result": len(result_events_before_final),
            "events": result_events_before_final,
        },
        "search": {
            "stage0_top_k_per_domain": common.STAGE0_TOP_K,
            "stage0_shard_count": 256,
            "new_candidates_per_seed_shard": common.MAX_REPRESENTATIVES_PER_SHARD,
            "authenticated_v4_topk": authenticated_v4.receipt,
            "overlay_merge": overlay_merge.receipt,
            "merged_old_plus_new_per_domain": 4096,
            "union_shortlist_count": len(union_ordinals),
            "stage1_winners": winner_records,
            "selection_details": selection_details,
            "selection_folds": selection_folds,
        },
        "validation": {
            "details": validation_details,
            "folds": validation_folds,
            "null_captures": null_captures,
        },
        "physical_ledger": common.physical_ledger(),
        "generated_value_ledger": {
            "new_stage0": common.NEW_EFFECTIVE_CANDIDATES * 512,
            "stage1_actual": len(union_ordinals) * 48_624,
            "stage1_maximum": 135_168 * 48_624,
            "post_selection_reporting": len(common.DOMAIN_IDS) * 65_536,
            "end_to_end_actual": (
                common.NEW_EFFECTIVE_CANDIDATES * 512
                + len(union_ordinals) * 48_624
                + len(common.DOMAIN_IDS) * 65_536
            ),
            "end_to_end_maximum": 28_183_625_728,
            "learned_generator_table_bytes": 0,
            "external_generator_read_bytes": 0,
        },
        "research_read_ledger": {
            "qwen_eligible_payload_count": len(eligible_rows),
            "qwen_eligible_payload_bytes": sum(int(row.bytes) for row in eligible_rows),
            "each_qwen_eligible_payload_opened_exactly_once": True,
            "excluded_qwen_payload_reads": 0,
            "v4_bound_metadata_and_topk_bytes": (
                lock["v4_reuse"]["result_bytes"]
                + lock["v4_reuse"]["result_audit"]["file_bytes"]
                + lock["v4_reuse"]["merged_event"]["bytes"]
                + lock["v4_reuse"]["merged_event"]["required_fields"]["file_bytes"]
            ),
            "v4_bound_metadata_and_topk_files_each_read_once": True,
            "inference_read_ledger_is_physical_ledger_not_research_search_io": True,
        },
        "decision": decision,
    }
    return _commit_or_verify_result(result_path, journal, result, completed_result)


def cpu_preflight() -> dict[str, Any]:
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("CPU preflight must run without CUDA/TE imports")
    lock = common.load_candidate_lock()
    representatives = common.representative_ordinals(0, 256)
    if len(representatives) != 164_864:
        raise common.ProtocolError("new-layout seed-shard representative count mismatch")
    full_representatives = common.full_representative_ordinals(0, 256)
    if len(full_representatives) != 229_376:
        raise common.ProtocolError("full-union seed-shard representative count mismatch")
    fixture = _synthetic_calibration_coordinates()
    return {
        "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_cpu_preflight_v5",
        "status": "PASS_NO_QWEN_ACCESS_CUDA_NOT_IMPORTED_OR_TOUCHED",
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "logical_candidates": common.LOGICAL_CANDIDATES,
        "new_effective_candidates": common.NEW_EFFECTIVE_CANDIDATES,
        "full_union_effective_candidates": common.FULL_EFFECTIVE_CANDIDATES,
        "retained_v4_effective_candidates": common.V4_EFFECTIVE_CANDIDATES,
        "domain_ids": list(common.DOMAIN_IDS),
        "equivalence_map": common.equivalence_map_object(),
        "equivalence_map_sha256": common.equivalence_map_sha256(),
        "representatives_per_seed_shard": len(representatives),
        "full_union_representatives_per_seed_shard": len(full_representatives),
        "calibration_coordinate_fixture_sha256": _synthetic_coordinate_sha256(*fixture),
        "calibration_role_counts": fixture[3]["role_counts"],
        "calibration_domain_fixture_sha256": _synthetic_domain_sha256(*_synthetic_domains()),
        "stage0_max_generated_values": common.NEW_EFFECTIVE_CANDIDATES * 512,
        "stage1_union_max": len(common.DOMAIN_IDS) * 2 * common.STAGE0_TOP_K,
        "stage1_max_generated_values": len(common.DOMAIN_IDS) * 2 * common.STAGE0_TOP_K * 48_624,
        "post_selection_reporting_generated_values": len(common.DOMAIN_IDS) * 65_536,
        "end_to_end_max_generated_values": (
            common.NEW_EFFECTIVE_CANDIDATES * 512
            + len(common.DOMAIN_IDS) * 2 * common.STAGE0_TOP_K * 48_624
            + len(common.DOMAIN_IDS) * 65_536
        ),
        "all_33_domains_share_identical_candidates": True,
        "v4_reuse_bindings_present": bool(lock.get("v4_reuse")),
        "v4_result_or_state_opened_by_preflight": False,
        "physical_ledger": common.physical_ledger(),
        "cuda_modules_imported": common.environment_has_cuda_imports(),
        "qwen_manifest_directory_or_payload_accessed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--source-trace", type=Path, required=True)
    production = subparsers.add_parser("run")
    production.add_argument("--workspace-root", type=Path, required=True)
    production.add_argument("--aux-dir", type=Path, required=True)
    production.add_argument("--output-dir", type=Path, required=True)
    production.add_argument("--calibration", type=Path, required=True)
    production.add_argument("--source-trace", type=Path, required=True)
    production.add_argument("--v4-run-root", type=Path, required=True)
    production.add_argument("--v4-result-audit", type=Path, required=True)
    production.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(cpu_preflight(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "calibrate":
        print(run_calibration(args.output, args.source_trace))
        return 0
    if args.command == "run":
        print(
            run_gate(
                args.workspace_root,
                args.aux_dir,
                args.output_dir,
                args.calibration,
                args.source_trace,
                args.v4_run_root,
                args.v4_result_audit,
                resume=args.resume,
            )
        )
        return 0
    raise common.ProtocolError("unsupported command")
