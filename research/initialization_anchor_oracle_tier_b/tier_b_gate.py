"""Sealed Tier-B MCore/Philox procedural-anchor calibration and gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common
import kernels


@dataclass
class MatrixData:
    plan: common.PlanRow
    source_fit: np.ndarray
    source_score: np.ndarray
    domains_fit: np.ndarray  # [33,n_fit]
    domains_score: np.ndarray  # [33,n_score]


class StateJournal:
    """Append-only create-new state files plus a SHA-chained event journal."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files = self.root / "files"
        self.events = self.root / "events"
        self.files.mkdir(parents=True, exist_ok=True)
        self.events.mkdir(parents=True, exist_ok=True)
        self._events = self._load_and_verify_events()

    def _load_and_verify_events(self) -> list[dict[str, Any]]:
        paths = sorted(self.events.glob("*.json"))
        events = []
        previous = "0" * 64
        for expected_sequence, path in enumerate(paths):
            if path.name != f"{expected_sequence:06d}.json" or path.is_symlink():
                raise common.ProtocolError("state journal sequence/name violation")
            raw = path.read_bytes()
            value = json.loads(raw.decode())
            if int(value["sequence"]) != expected_sequence or value["previous_event_sha256"] != previous:
                raise common.ProtocolError("state journal hash-chain violation")
            common.strict_keys(
                value,
                (
                    "sequence", "previous_event_sha256", "kind", "key", "relative_path",
                    "file_sha256", "file_bytes", "created_unix_ns",
                ),
                "state event",
            )
            target = self.root / str(value["relative_path"])
            if not target.is_file() or target.is_symlink():
                raise common.ProtocolError(f"state event target missing/non-regular: {target}")
            if target.stat().st_size != int(value["file_bytes"]) or common.sha256_file(target) != value["file_sha256"]:
                raise common.ProtocolError(f"state event target hash/size mismatch: {target}")
            previous = common.sha256_bytes(raw)
            events.append(value)
        return events

    @property
    def events_list(self) -> list[dict[str, Any]]:
        return list(self._events)

    def lookup(self, kind: str, key: str) -> Path | None:
        matches = [event for event in self._events if event["kind"] == kind and event["key"] == key]
        if len(matches) > 1:
            raise common.ProtocolError(f"duplicate state journal key: {kind}/{key}")
        return None if not matches else self.root / str(matches[0]["relative_path"])

    def _record_existing_file(self, kind: str, key: str, target: Path) -> Path:
        if self.lookup(kind, key) is not None:
            raise common.ProtocolError(f"attempt to overwrite state key: {kind}/{key}")
        sequence = len(self._events)
        previous = "0" * 64 if not self._events else common.sha256_file(self.events / f"{sequence-1:06d}.json")
        event = {
            "sequence": sequence,
            "previous_event_sha256": previous,
            "kind": kind,
            "key": key,
            "relative_path": str(target.relative_to(self.root)).replace("\\", "/"),
            "file_sha256": common.sha256_file(target),
            "file_bytes": target.stat().st_size,
            "created_unix_ns": time.time_ns(),
        }
        event_path = self.events / f"{sequence:06d}.json"
        with event_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(event, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._events.append(event)
        return target

    def write_json(self, kind: str, key: str, value: Mapping[str, Any]) -> Path:
        target = self.files / f"{kind}_{key}.json"
        if target.exists():
            raise common.ProtocolError(f"state target already exists: {target}")
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self._record_existing_file(kind, key, target)

    def write_npz(self, kind: str, key: str, **arrays: np.ndarray) -> Path:
        target = self.files / f"{kind}_{key}.npz"
        if target.exists():
            raise common.ProtocolError(f"state target already exists: {target}")
        with target.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        return self._record_existing_file(kind, key, target)


def _domain_arrays(
    plan: common.PlanRow, source_fit: np.ndarray, source_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fit = np.empty((len(common.DOMAIN_IDS), len(source_fit)), dtype=np.float32)
    score = np.empty((len(common.DOMAIN_IDS), len(source_score)), dtype=np.float32)
    fit[0] = source_fit
    score[0] = source_score
    mean = float(np.mean(source_fit.astype(np.float64)))
    centered = source_fit.astype(np.float64) - mean
    rms = math.sqrt(float(np.mean(centered * centered)))
    for domain_index, domain_id in enumerate(common.DOMAIN_IDS[1:17], start=1):
        fit[domain_index] = mean + rms * common.stateless_normals(
            domain_id, plan.source.tensor_name, "fit", plan.fit
        )
        score[domain_index] = mean + rms * common.stateless_normals(
            domain_id, plan.source.tensor_name, "score", plan.score
        )
    for domain_index, domain_id in enumerate(common.DOMAIN_IDS[17:], start=17):
        fit_permutation, fit_sign = common.permutation_and_sign(
            domain_id, plan.source.tensor_name, "fit", len(source_fit)
        )
        score_permutation, score_sign = common.permutation_and_sign(
            domain_id, plan.source.tensor_name, "score", len(source_score)
        )
        fit[domain_index] = mean + fit_sign * (source_fit[fit_permutation] - mean)
        score[domain_index] = mean + score_sign * (source_score[score_permutation] - mean)
    if not np.all(np.isfinite(fit)) or not np.all(np.isfinite(score)):
        raise common.ProtocolError("non-finite source/null domain value")
    return fit, score


def _decode_coordinates(payload: bytes, source: common.SourceRow, coordinates: Sequence[int]) -> np.ndarray:
    native = common.canonical_to_native_flat(source.role, np.asarray(coordinates, dtype=np.int64))
    words = np.frombuffer(payload, dtype="<u2")
    return common.decode_bfloat16_words(words[native]).astype(np.float32, copy=True)


def _load_plan_payloads(
    plan: Sequence[common.PlanRow],
    paths: Mapping[str, Path],
    access_log: list[dict[str, Any]],
) -> list[MatrixData]:
    result = []
    for row in plan:
        source = row.source
        if source.excluded:
            raise common.ProtocolError("excluded source reached payload loader")
        payload = paths[source.tensor_name].read_bytes()
        if len(payload) != source.bytes or common.sha256_bytes(payload) != source.sha256:
            raise common.ProtocolError(f"source payload size/SHA mismatch: {source.basename}")
        source_fit = _decode_coordinates(payload, source, row.fit)
        source_score = _decode_coordinates(payload, source, row.score)
        domains_fit, domains_score = _domain_arrays(row, source_fit, source_score)
        access_log.append(
            {
                "sequence": len(access_log),
                "event": "payload_opened_and_hash_verified",
                "tensor_name": source.tensor_name,
                "split": source.split,
                "sha256": source.sha256,
            }
        )
        result.append(MatrixData(row, source_fit, source_score, domains_fit, domains_score))
    return result


def _load_selection_payloads_once(
    full_plan: Sequence[common.PlanRow],
    stage0_plan: Sequence[common.PlanRow],
    paths: Mapping[str, Path],
    access_log: list[dict[str, Any]],
) -> tuple[list[MatrixData], list[MatrixData]]:
    stage0_by_name = {row.source.tensor_name: row for row in stage0_plan}
    if len(stage0_by_name) != len(stage0_plan):
        raise common.ProtocolError("duplicate stage0 tensor identity")
    full_result = []
    stage0_result = []
    for full_row in full_plan:
        source = full_row.source
        if source.tensor_name not in stage0_by_name or source.split != "candidate_selection":
            raise common.ProtocolError("selection full/stage0 plan identity mismatch")
        stage0_row = stage0_by_name[source.tensor_name]
        payload = paths[source.tensor_name].read_bytes()
        if len(payload) != source.bytes or common.sha256_bytes(payload) != source.sha256:
            raise common.ProtocolError(f"source payload size/SHA mismatch: {source.basename}")
        full_fit = _decode_coordinates(payload, source, full_row.fit)
        full_score = _decode_coordinates(payload, source, full_row.score)
        full_domains_fit, full_domains_score = _domain_arrays(full_row, full_fit, full_score)
        full_result.append(
            MatrixData(full_row, full_fit, full_score, full_domains_fit, full_domains_score)
        )
        stage0_fit = _decode_coordinates(payload, source, stage0_row.fit)
        stage0_score = _decode_coordinates(payload, source, stage0_row.score)
        stage0_domains_fit, stage0_domains_score = _domain_arrays(
            stage0_row, stage0_fit, stage0_score
        )
        stage0_result.append(
            MatrixData(
                stage0_row,
                stage0_fit,
                stage0_score,
                stage0_domains_fit,
                stage0_domains_score,
            )
        )
        access_log.append(
            {
                "sequence": len(access_log),
                "event": "payload_opened_and_hash_verified",
                "tensor_name": source.tensor_name,
                "split": source.split,
                "sha256": source.sha256,
            }
        )
    return full_result, stage0_result


def _flatten_coordinate_metadata(matrices: Sequence[MatrixData]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[slice, slice]]]:
    experts: list[int] = []
    roles: list[int] = []
    coordinates: list[int] = []
    slices = []
    cursor = 0
    for matrix in matrices:
        fit_start = cursor
        values = matrix.plan.fit
        experts.extend([matrix.plan.source.expert] * len(values))
        roles.extend([0 if matrix.plan.source.role == "up" else 1] * len(values))
        coordinates.extend(values)
        cursor += len(values)
        fit_slice = slice(fit_start, cursor)
        score_start = cursor
        values = matrix.plan.score
        experts.extend([matrix.plan.source.expert] * len(values))
        roles.extend([0 if matrix.plan.source.role == "up" else 1] * len(values))
        coordinates.extend(values)
        cursor += len(values)
        slices.append((fit_slice, slice(score_start, cursor)))
    return (
        np.asarray(experts, dtype=np.int32),
        np.asarray(roles, dtype=np.int32),
        np.asarray(coordinates, dtype=np.uint64),
        slices,
    )


def _affine_sse_from_moments(cp, g_fit, g_score, w_fit, w_score):
    """All domains against a common anchor batch; float64 throughout."""
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
    alpha = cp.where(centered_g2 > 0.0, centered_wg / centered_g2, 0.0)
    mean_w = sum_w_fit / n_fit
    mu = mean_w[None, :] - alpha * (sum_g_fit[:, None] / n_fit)

    sum_g_score = cp.sum(g_score, axis=1, dtype=cp.float64)
    sum_g2_score = cp.sum(g_score * g_score, axis=1, dtype=cp.float64)
    sum_w_score = cp.sum(w_score, axis=1, dtype=cp.float64)
    sum_w2_score = cp.sum(w_score * w_score, axis=1, dtype=cp.float64)
    sum_wg_score = g_score @ w_score.T
    sse = (
        sum_w2_score[None, :]
        + n_score * mu * mu
        + alpha * alpha * sum_g2_score[:, None]
        + 2.0 * mu * alpha * sum_g_score[:, None]
        - 2.0 * mu * sum_w_score[None, :]
        - 2.0 * alpha * sum_wg_score
    )
    baseline = (
        sum_w2_score
        - 2.0 * mean_w * sum_w_score
        + n_score * mean_w * mean_w
    )
    return cp.maximum(sse, 0.0), baseline


def _stage0_q(access: kernels.PhiloxRandomAccess, anchors, matrices: Sequence[MatrixData], slices):
    cp = access.cp
    total_sse = cp.zeros((anchors.shape[0], len(common.DOMAIN_IDS)), dtype=cp.float64)
    total_baseline = cp.zeros(len(common.DOMAIN_IDS), dtype=cp.float64)
    for role in ("up", "down"):
        selected = [index for index, matrix in enumerate(matrices) if matrix.plan.source.role == role]
        fit_indices = np.concatenate(
            [np.arange(slices[index][0].start, slices[index][0].stop, dtype=np.int64) for index in selected]
        )
        score_indices = np.concatenate(
            [np.arange(slices[index][1].start, slices[index][1].stop, dtype=np.int64) for index in selected]
        )
        w_fit = np.concatenate([matrices[index].domains_fit for index in selected], axis=1)
        w_score = np.concatenate([matrices[index].domains_score for index in selected], axis=1)
        sse, baseline = _affine_sse_from_moments(
            cp,
            anchors[:, cp.asarray(fit_indices)],
            anchors[:, cp.asarray(score_indices)],
            w_fit,
            w_score,
        )
        total_sse += sse
        total_baseline += baseline
    if bool(cp.any(total_baseline <= 0.0)):
        raise common.ProtocolError("stage0 baseline is non-positive")
    return total_sse / total_baseline[None, :]


def _stage1_q(access: kernels.PhiloxRandomAccess, anchors, matrices: Sequence[MatrixData], slices):
    cp = access.cp
    total_sse = cp.zeros((anchors.shape[0], len(common.DOMAIN_IDS)), dtype=cp.float64)
    total_baseline = cp.zeros(len(common.DOMAIN_IDS), dtype=cp.float64)
    for matrix, (fit_slice, score_slice) in zip(matrices, slices):
        sse, baseline = _affine_sse_from_moments(
            cp,
            anchors[:, fit_slice],
            anchors[:, score_slice],
            matrix.domains_fit,
            matrix.domains_score,
        )
        total_sse += sse
        total_baseline += baseline
    if bool(cp.any(total_baseline <= 0.0)):
        raise common.ProtocolError("stage1 baseline is non-positive")
    return total_sse / total_baseline[None, :]


def _exact_top_k(cp, q, ordinals: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if q.ndim != 2 or q.shape[0] != len(ordinals) or q.shape[1] != len(common.DOMAIN_IDS):
        raise common.ProtocolError("top-k metric shape mismatch")
    ord_gpu = cp.asarray(ordinals, dtype=cp.uint64)
    top_ordinals = np.empty((len(common.DOMAIN_IDS), top_k), dtype=np.uint64)
    top_q = np.empty((len(common.DOMAIN_IDS), top_k), dtype=np.float64)
    for domain_index in range(len(common.DOMAIN_IDS)):
        column = q[:, domain_index]
        partition = cp.argpartition(column, top_k - 1)[:top_k]
        threshold = cp.max(column[partition])
        eligible = cp.where(column <= threshold)[0]
        eligible_ordinals = cp.asnumpy(ord_gpu[eligible])
        eligible_q = cp.asnumpy(column[eligible])
        order = np.lexsort((eligible_ordinals, eligible_q))[:top_k]
        top_ordinals[domain_index] = eligible_ordinals[order]
        top_q[domain_index] = eligible_q[order]
    return top_ordinals, top_q


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except Exception as error:
        raise common.ProtocolError(f"invalid state NPZ {path}: {error}") from error


def _run_stage0(
    access: kernels.PhiloxRandomAccess,
    journal: StateJournal,
    matrices: Sequence[MatrixData],
) -> tuple[np.ndarray, np.ndarray]:
    experts, roles, coordinates, slices = _flatten_coordinate_metadata(matrices)
    if len(coordinates) != common.STAGE0_FIT + common.STAGE0_SCORE:
        raise common.ProtocolError("stage0 coordinate count mismatch")
    for shard in range(256):
        key = f"{shard:03d}"
        existing = journal.lookup("stage0", key)
        if existing is not None:
            state = _load_npz(existing)
            if state["top_ordinals"].shape != (len(common.DOMAIN_IDS), common.STAGE0_TOP_K):
                raise common.ProtocolError(f"stage0 state shape mismatch: {key}")
            continue
        seed_start = shard * common.SEED_SHARD_SIZE
        seed_stop = seed_start + common.SEED_SHARD_SIZE
        ordinals = common.representative_ordinals(seed_start, seed_stop)
        anchors = access.generate(ordinals, experts, roles, coordinates)
        q = _stage0_q(access, anchors, matrices, slices)
        top_ordinals, top_q = _exact_top_k(
            access.cp, q, ordinals, common.STAGE0_TOP_K
        )
        journal.write_npz(
            "stage0",
            key,
            seed_start=np.asarray([seed_start], dtype=np.int32),
            seed_stop=np.asarray([seed_stop], dtype=np.int32),
            top_ordinals=top_ordinals,
            top_q=top_q,
        )
        print(f"tier-b stage0 seed shard {shard + 1}/256", flush=True)
        del anchors, q

    merged_path = journal.lookup("stage0_merged", "global")
    if merged_path is not None:
        merged = _load_npz(merged_path)
        return merged["domain_top_ordinals"], merged["union_ordinals"]
    all_ordinals = [[] for _ in common.DOMAIN_IDS]
    all_q = [[] for _ in common.DOMAIN_IDS]
    for shard in range(256):
        path = journal.lookup("stage0", f"{shard:03d}")
        if path is None:
            raise common.ProtocolError("stage0 merge encountered missing shard")
        state = _load_npz(path)
        for domain_index in range(len(common.DOMAIN_IDS)):
            all_ordinals[domain_index].append(state["top_ordinals"][domain_index])
            all_q[domain_index].append(state["top_q"][domain_index])
    domain_top_ordinals = np.empty(
        (len(common.DOMAIN_IDS), common.STAGE0_TOP_K), dtype=np.uint64
    )
    domain_top_q = np.empty(
        (len(common.DOMAIN_IDS), common.STAGE0_TOP_K), dtype=np.float64
    )
    for domain_index in range(len(common.DOMAIN_IDS)):
        ordinals = np.concatenate(all_ordinals[domain_index])
        q = np.concatenate(all_q[domain_index])
        order = np.lexsort((ordinals, q))[: common.STAGE0_TOP_K]
        domain_top_ordinals[domain_index] = ordinals[order]
        domain_top_q[domain_index] = q[order]
    union_ordinals = np.unique(domain_top_ordinals.reshape(-1))
    if len(union_ordinals) > 33 * common.STAGE0_TOP_K:
        raise common.ProtocolError("stage0 union exceeds frozen maximum")
    journal.write_npz(
        "stage0_merged",
        "global",
        domain_top_ordinals=domain_top_ordinals,
        domain_top_q=domain_top_q,
        union_ordinals=union_ordinals,
    )
    return domain_top_ordinals, union_ordinals


def _run_stage1(
    access: kernels.PhiloxRandomAccess,
    journal: StateJournal,
    matrices: Sequence[MatrixData],
    union_ordinals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    experts, roles, coordinates, slices = _flatten_coordinate_metadata(matrices)
    if len(coordinates) != 48_624:
        raise common.ProtocolError(f"selection full-coordinate count {len(coordinates)} != 48624")
    batch_size = 512
    batch_count = (len(union_ordinals) + batch_size - 1) // batch_size
    for batch_index in range(batch_count):
        key = f"{batch_index:04d}"
        existing = journal.lookup("stage1", key)
        start = batch_index * batch_size
        stop = min(len(union_ordinals), start + batch_size)
        ordinals = union_ordinals[start:stop]
        if existing is not None:
            state = _load_npz(existing)
            if not np.array_equal(state["ordinals"], ordinals) or state["q"].shape != (
                len(ordinals), len(common.DOMAIN_IDS)
            ):
                raise common.ProtocolError(f"stage1 state mismatch: {key}")
            continue
        anchors = access.generate(ordinals, experts, roles, coordinates)
        q = _stage1_q(access, anchors, matrices, slices)
        journal.write_npz(
            "stage1", key, ordinals=ordinals, q=access.cp.asnumpy(q)
        )
        print(f"tier-b stage1 candidate batch {batch_index + 1}/{batch_count}", flush=True)
        del anchors, q

    winners_path = journal.lookup("stage1_winners", "global")
    if winners_path is not None:
        state = _load_npz(winners_path)
        return state["winner_ordinals"], state["winner_q"]
    winner_ordinals = np.zeros(len(common.DOMAIN_IDS), dtype=np.uint64)
    winner_q = np.full(len(common.DOMAIN_IDS), np.inf, dtype=np.float64)
    for batch_index in range(batch_count):
        path = journal.lookup("stage1", f"{batch_index:04d}")
        if path is None:
            raise common.ProtocolError("stage1 merge encountered missing batch")
        state = _load_npz(path)
        for domain_index in range(len(common.DOMAIN_IDS)):
            q = state["q"][:, domain_index]
            ordinals = state["ordinals"]
            order = np.lexsort((ordinals, q))
            local = int(order[0])
            pair = (float(q[local]), int(ordinals[local]))
            best_pair = (float(winner_q[domain_index]), int(winner_ordinals[domain_index]))
            if pair < best_pair:
                winner_q[domain_index] = pair[0]
                winner_ordinals[domain_index] = pair[1]
    journal.write_npz(
        "stage1_winners", "global", winner_ordinals=winner_ordinals, winner_q=winner_q
    )
    return winner_ordinals, winner_q


def _candidate_details(
    access: kernels.PhiloxRandomAccess,
    ordinal: int,
    domain_index: int,
    matrices: Sequence[MatrixData],
) -> list[dict[str, Any]]:
    experts, roles, coordinates, slices = _flatten_coordinate_metadata(matrices)
    anchors = access.generate(np.asarray([ordinal], dtype=np.uint64), experts, roles, coordinates)
    anchor = access.cp.asnumpy(anchors[0]).astype(np.float32, copy=False)
    rows = []
    for matrix, (fit_slice, score_slice) in zip(matrices, slices):
        g_fit = anchor[fit_slice]
        g_score = anchor[score_slice]
        w_fit = matrix.domains_fit[domain_index]
        w_score = matrix.domains_score[domain_index]
        fit = common.fit_affine_moments(w_fit, g_fit)
        score = common.score_affine_moments(
            w_score, g_score, float(fit["alpha"]), float(fit["mu"]), float(fit["fit_mean_w"])
        )
        rows.append(
            {
                "tensor_name": matrix.plan.source.tensor_name,
                "expert": matrix.plan.source.expert,
                "role": matrix.plan.source.role,
                "fit": fit,
                "score": score,
            }
        )
    return rows


def _torch_generate(torch, device, seed: int, offset: int, shape: tuple[int, ...], dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    if not hasattr(generator, "set_offset") or not hasattr(generator, "get_offset"):
        raise common.ProtocolError("PyTorch CUDA Generator offset API unavailable")
    generator.set_offset(int(offset))
    tensor = torch.empty(shape, dtype=dtype, device=device)
    tensor.normal_(0.0, 0.02, generator=generator)
    return tensor, int(generator.get_offset())


def _float_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def parity_preflight(access: kernels.PhiloxRandomAccess) -> dict[str, Any]:
    """Bit-exact parity; called before any source manifest/payload operation."""
    try:
        import torch
    except Exception as error:  # pragma: no cover - production environment
        raise common.ProtocolError(f"PyTorch unavailable for mandatory parity: {error}") from error
    if not torch.cuda.is_available():
        raise common.ProtocolError("PyTorch CUDA unavailable for mandatory parity")
    cp = access.cp
    device = torch.device(f"cuda:{access.device_index}")
    torch.cuda.set_device(device)

    shapes = (
        (768, 2048), (2048, 768), (1536, 2048),
        (384, 2048), (2048, 384), (768, 2048),
        (192, 2048), (2048, 192), (384, 2048),
    )
    offsets = (0, 4, 8192, 1_048_576, 4_294_967_300)
    descriptor_rows = []
    for shape_index, shape in enumerate(shapes):
        numel = math.prod(shape)
        stride = 256 * min(
            (numel + 255) // 256,
            access.sm_count * (access.max_threads_per_sm // 256),
        )
        native = sorted({0, 1, min(numel - 1, stride - 1), min(numel - 1, stride), min(numel - 1, 4 * stride), numel - 1})
        for offset in offsets:
            seed = 12_345 + shape_index
            seeds = [seed] * len(native)
            numels = [numel] * len(native)
            probe_standard, probe_scaled, probe_bf16 = access.descriptor_probe(
                seeds, numels, [offset] * len(native), native
            )
            probe_scaled_np = cp.asnumpy(probe_scaled)
            probe_bf16_np = cp.asnumpy(probe_bf16)
            tensor_f32, observed_after = _torch_generate(torch, device, seed, offset, shape, torch.float32)
            expected_increment = kernels.policy_increment(numel, access.sm_count, access.max_threads_per_sm)
            if observed_after - offset != expected_increment:
                raise common.ProtocolError("float32 generator increment parity failed")
            torch_values = tensor_f32.reshape(-1).index_select(
                0, torch.as_tensor(native, dtype=torch.int64, device=device)
            ).cpu().numpy()
            if not np.array_equal(probe_scaled_np, torch_values):
                raise common.ProtocolError(
                    f"exact float32 normal transform parity failed shape={shape}, offset={offset}"
                )
            direct_bf16, observed_bf16_after = _torch_generate(
                torch, device, seed, offset, shape, torch.bfloat16
            )
            if observed_bf16_after - offset != expected_increment:
                raise common.ProtocolError("BF16 generator increment parity failed")
            direct_values = direct_bf16.reshape(-1).index_select(
                0, torch.as_tensor(native, dtype=torch.int64, device=device)
            ).float().cpu().numpy()
            cast_values = tensor_f32.to(torch.bfloat16).reshape(-1).index_select(
                0, torch.as_tensor(native, dtype=torch.int64, device=device)
            ).float().cpu().numpy()
            if not np.array_equal(probe_bf16_np, direct_values) or not np.array_equal(
                probe_bf16_np, cast_values
            ):
                raise common.ProtocolError(
                    f"BF16/direct/cast parity failed shape={shape}, offset={offset}"
                )
            descriptor_rows.append(
                {
                    "shape": list(shape),
                    "offset": offset,
                    "coordinate_count": len(native),
                    "increment": expected_increment,
                    "float32_sha256": _float_sha(torch_values),
                    "bf16_widened_sha256": _float_sha(direct_values),
                }
            )
            del tensor_f32, direct_bf16

    # Candidate-coordinate parity spans PP rank/local-layer classes, both EP
    # assignments, ETP 1/2/4 shard boundaries, and all three packings.
    candidates = []
    for pp_index in (0, 2, 3):
        for ep_index, assignment in ((0, 0), (3, 0), (3, 1), (7, 0), (7, 1)):
            for etp_index in range(3):
                for packing in range(3):
                    ordinal = common.logical_ordinal(3407, pp_index, ep_index, etp_index, assignment, packing)
                    candidates.append(common.decode_ordinal(ordinal))
    coordinates = (0, 1, 2047, 2048, 393_215, 393_216, 786_431, 786_432, common.WEIGHTS_PER_MATRIX - 1)
    candidate_rows = []
    for candidate in candidates:
        for expert in (0, 57, 127):
            for role in ("up", "down"):
                valid_coordinates = []
                descriptors = []
                for coordinate in coordinates:
                    if coordinate >= common.WEIGHTS_PER_MATRIX:
                        continue
                    descriptor = kernels.coordinate_descriptor(
                        candidate,
                        expert,
                        role,
                        coordinate,
                        access.sm_count,
                        access.max_threads_per_sm,
                    )
                    valid_coordinates.append(coordinate)
                    descriptors.append(descriptor)
                standard, scaled, bf16 = access.descriptor_probe(
                    [row.seed for row in descriptors],
                    [row.target_numel for row in descriptors],
                    [row.target_offset for row in descriptors],
                    [row.native_index for row in descriptors],
                )
                expected = np.empty(len(descriptors), dtype=np.float32)
                expected_bf16 = np.empty(len(descriptors), dtype=np.float32)
                grouped: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
                for index, descriptor in enumerate(descriptors):
                    grouped.setdefault(
                        (descriptor.seed, descriptor.target_numel, descriptor.target_offset), []
                    ).append((index, descriptor.native_index))
                for (seed, numel, offset), entries in grouped.items():
                    tensor, after = _torch_generate(torch, device, seed, offset, (numel,), torch.float32)
                    if after - offset != kernels.policy_increment(
                        numel, access.sm_count, access.max_threads_per_sm
                    ):
                        raise common.ProtocolError("candidate target increment parity failed")
                    indices = torch.as_tensor([native for _, native in entries], dtype=torch.int64, device=device)
                    values = tensor.index_select(0, indices)
                    values_np = values.cpu().numpy()
                    bf16_np = values.to(torch.bfloat16).float().cpu().numpy()
                    for local, (destination, _) in enumerate(entries):
                        expected[destination] = values_np[local]
                        expected_bf16[destination] = bf16_np[local]
                    del tensor, values
                if not np.array_equal(cp.asnumpy(scaled), expected):
                    raise common.ProtocolError(f"packing/TP/EP/PP float parity failed: {candidate.id}")
                if not np.array_equal(cp.asnumpy(bf16), expected_bf16):
                    raise common.ProtocolError(f"packing/TP/EP/PP BF16 parity failed: {candidate.id}")
                generated = access.generate(
                    np.asarray([candidate.ordinal], dtype=np.uint64),
                    np.asarray([expert] * len(valid_coordinates), dtype=np.int32),
                    np.asarray([0 if role == "up" else 1] * len(valid_coordinates), dtype=np.int32),
                    np.asarray(valid_coordinates, dtype=np.uint64),
                )
                if not np.array_equal(cp.asnumpy(generated[0]) * np.float32(0.02), expected):
                    raise common.ProtocolError(f"candidate anchor-kernel parity failed: {candidate.id}")
                candidate_rows.append(
                    {
                        "candidate": candidate.id,
                        "expert": expert,
                        "role": role,
                        "coordinate_count": len(valid_coordinates),
                        "scaled_sha256": _float_sha(expected),
                    }
                )

    # Explicit persistent-call offset parity for every packing.  This checks
    # the offset arithmetic against Generator.get_offset, not only values at a
    # supplied offset.
    packing_rows = []
    for etp in common.ETP_SIZES:
        n = (common.ROWS // etp) * common.COLUMNS
        inc_n = kernels.policy_increment(n, access.sm_count, access.max_threads_per_sm)
        for packing in common.PACKINGS:
            generator = torch.Generator(device=device)
            generator.manual_seed(2024)
            if packing == "separate_gate_up_down":
                call_numels = (n, n, n)
            else:
                call_numels = (2 * n, n)
            observed = [int(generator.get_offset())]
            for numel in call_numels:
                temporary = torch.empty((numel,), dtype=torch.float32, device=device)
                temporary.normal_(0.0, 0.02, generator=generator)
                observed.append(int(generator.get_offset()))
                del temporary
            expected = [0]
            for numel in call_numels:
                expected.append(expected[-1] + kernels.policy_increment(numel, access.sm_count, access.max_threads_per_sm))
            if observed != expected:
                raise common.ProtocolError(f"persistent packing offset parity failed: {packing}/ETP{etp}")
            packing_rows.append({"packing": packing, "etp": etp, "offsets": observed})

    # Same-device DLPack interop.
    torch_probe = torch.arange(257, dtype=torch.float32, device=device)
    cupy_probe = cp.from_dlpack(torch_probe)
    if not np.array_equal(torch_probe.cpu().numpy(), cp.asnumpy(cupy_probe)):
        raise common.ProtocolError("PyTorch/CuPy DLPack parity failed")
    torch.cuda.synchronize(device)
    cp.cuda.runtime.deviceSynchronize()
    return {
        "all_required_checks_passed": True,
        "torch_version": str(torch.__version__),
        "cupy_version": str(cp.__version__),
        "device_name": access.device_name,
        "device_index": access.device_index,
        "multi_processor_count": access.sm_count,
        "max_threads_per_multi_processor": access.max_threads_per_sm,
        "descriptor_checks": descriptor_rows,
        "candidate_coordinate_checks": candidate_rows,
        "persistent_packing_checks": packing_rows,
        "dlpack_sha256_f32le": _float_sha(cp.asnumpy(cupy_probe)),
    }


def _synthetic_calibration_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experts = np.empty(512, dtype=np.int32)
    roles = np.empty(512, dtype=np.int32)
    coordinates = np.empty(512, dtype=np.uint64)
    for index in range(512):
        digest = hashlib.sha256(b"TIERB-SOURCE-FREE-CALIBRATION-v1" + index.to_bytes(4, "little")).digest()
        experts[index] = int.from_bytes(digest[:2], "little") % 128
        roles[index] = digest[2] & 1
        coordinates[index] = int.from_bytes(digest[3:11], "little") % common.WEIGHTS_PER_MATRIX
    return experts, roles, coordinates


def run_calibration(output_path: Path) -> Path:
    if output_path.exists():
        raise common.ProtocolError("calibration output already exists")
    lock = common.load_candidate_lock()
    access = kernels.PhiloxRandomAccess(0)
    parity = parity_preflight(access)
    ordinals = common.representative_ordinals(0, 256)
    if len(ordinals) != int(lock["source_free_calibration"]["candidate_count"]):
        raise common.ProtocolError("calibration candidate count differs from lock")
    experts, roles, coordinates = _synthetic_calibration_coordinates()
    cp = access.cp
    output = cp.empty((len(ordinals), len(coordinates)), dtype=cp.float32)
    # One warmup is excluded.  It also proves the exact production shape fits.
    access.generate(ordinals, experts, roles, coordinates, output=output)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = []
    hashes = []
    for _ in range(int(lock["source_free_calibration"]["repetitions"])):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        access.generate(ordinals, experts, roles, coordinates, output=output)
        stop.record()
        stop.synchronize()
        seconds = float(cp.cuda.get_elapsed_time(start, stop)) / 1000.0
        elapsed.append(seconds)
        # Bind deterministic sparse sentinels without copying the 216 MiB output.
        sentinel = output.reshape(-1)[:: max(1, output.size // 4096)][:4096]
        hashes.append(_float_sha(cp.asnumpy(sentinel)))
    if len(set(hashes)) != 1:
        raise common.ProtocolError("calibration kernel output was nondeterministic")
    generated = len(ordinals) * len(coordinates)
    rates = [generated / seconds for seconds in elapsed]
    result = {
        "schema": "qwen3_initialization_anchor_tier_b_source_free_calibration_v1",
        "status": "PASS_SOURCE_FREE_CALIBRATION",
        "source_manifest_or_payload_opened": False,
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "parity": parity,
        "candidate_count": len(ordinals),
        "coordinate_count": len(coordinates),
        "values_per_repetition": generated,
        "elapsed_seconds": elapsed,
        "values_per_second": rates,
        "median_values_per_second": statistics.median(rates),
        "estimated_stage0_seconds_at_median_kernel_rate": int(lock["search_cascade"]["stage0"]["maximum_generated_normal_values"]) / statistics.median(rates),
        "output_sentinel_sha256_f32le": hashes[0],
        "working_output_bytes": int(output.nbytes),
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "equivalence_map_sha256": common.equivalence_map_sha256(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output_path


def _load_calibration(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise common.ProtocolError("calibration must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "status", "source_manifest_or_payload_opened",
        "candidate_lock_file_sha256", "candidate_lock_internal_sha256", "runner_sha256",
        "common_sha256", "kernels_sha256", "parity", "candidate_count", "coordinate_count",
        "values_per_repetition", "elapsed_seconds", "values_per_second",
        "median_values_per_second", "estimated_stage0_seconds_at_median_kernel_rate",
        "output_sentinel_sha256_f32le", "working_output_bytes", "logical_candidate_count",
        "effective_candidate_count", "equivalence_map_sha256",
    }
    common.strict_keys(value, required, "calibration")
    if value["schema"] != "qwen3_initialization_anchor_tier_b_source_free_calibration_v1":
        raise common.ProtocolError("calibration schema mismatch")
    if value["status"] != "PASS_SOURCE_FREE_CALIBRATION" or value["source_manifest_or_payload_opened"] is not False:
        raise common.ProtocolError("calibration status/source-free claim mismatch")
    expected = {
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "equivalence_map_sha256": common.equivalence_map_sha256(),
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise common.ProtocolError(f"calibration binding mismatch: {key}")
    if value["parity"].get("all_required_checks_passed") is not True:
        raise common.ProtocolError("calibration parity did not pass")
    return value


def _expected_run_header(
    aux_dir: Path,
    calibration_path: Path,
    calibration: Mapping[str, Any],
    stage0_plan,
    full_plan,
) -> dict[str, Any]:
    return {
        "schema": "qwen3_initialization_anchor_tier_b_run_header_v1",
        "status": "IMMUTABLE_STATE_HEADER",
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
        "tier_a_common_sha256": common.EXPECTED_TIER_A_COMMON_SHA256,
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": common.sha256_file(calibration_path.resolve()),
        "calibration_output_sentinel_sha256": calibration["output_sentinel_sha256_f32le"],
        "auxiliary_directory": str(aux_dir.resolve()),
        "stage0_plan_sha256": common.plan_sha256(stage0_plan),
        "full_plan_sha256": common.plan_sha256(full_plan),
        "logical_candidate_count": common.LOGICAL_CANDIDATES,
        "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
        "domain_ids": list(common.DOMAIN_IDS),
        "equivalence_map_sha256": common.equivalence_map_sha256(),
        "launch_sentinel": "/tmp/init_anchor_tier_b_release_v1",
    }


def _verify_or_create_header(
    journal: StateJournal, expected_header: Mapping[str, Any]
) -> Path:
    existing = journal.lookup("run_header", "immutable")
    if existing is None:
        return journal.write_json("run_header", "immutable", expected_header)
    observed = json.loads(existing.read_text(encoding="utf-8"))
    if observed != expected_header:
        raise common.ProtocolError("resume run header differs from current frozen bindings")
    return existing


def run_gate(
    workspace_root: Path,
    aux_dir: Path,
    output_dir: Path,
    calibration_path: Path,
    *,
    resume: bool,
) -> Path:
    # Lock, calibration and CUDA parity are deliberately completed before any
    # source manifest, source directory, or source payload access.
    lock = common.load_candidate_lock()
    calibration = _load_calibration(calibration_path)
    sentinel = Path(str(lock["execution"]["launch_sentinel"]))
    if not sentinel.is_file() or sentinel.is_symlink():
        raise common.ProtocolError("required launch sentinel is absent/non-regular")
    access = kernels.PhiloxRandomAccess(0)
    parity = parity_preflight(access)
    if not parity.get("all_required_checks_passed"):
        raise common.ProtocolError("production parity did not pass")

    rows = common.load_source_rows(workspace_root)
    exclusion = common.exclusion_binding(workspace_root)
    paths = common.validate_aux_directory(aux_dir, rows)
    stage0_plan = common.make_plan(rows, stage0=True)
    full_plan = common.make_plan(rows, stage0=False)
    if common.plan_sha256(stage0_plan) != lock["coordinate_protocol"]["stage0_coordinate_plan_sha256"]:
        raise common.ProtocolError("stage0 coordinate plan differs from lock")
    if common.plan_sha256(full_plan) != lock["coordinate_protocol"]["full_coordinate_plan_sha256"]:
        raise common.ProtocolError("full coordinate plan differs from lock")

    output_dir = output_dir.resolve()
    if output_dir.exists() and not resume:
        raise common.ProtocolError("output directory exists; explicit --resume required")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir() or output_dir.is_symlink():
        raise common.ProtocolError("resume output is not a regular directory")
    journal = StateJournal(output_dir / str(lock["execution"]["state_directory"]))
    header = _expected_run_header(aux_dir, calibration_path, calibration, stage0_plan, full_plan)
    _verify_or_create_header(journal, header)
    result_path = output_dir / str(lock["execution"]["output_json"])
    if result_path.exists():
        if resume and result_path.is_file() and not result_path.is_symlink():
            return result_path
        raise common.ProtocolError("result target already exists")

    access_log: list[dict[str, Any]] = [
        {
            "sequence": 0,
            "event": "production_cuda_parity_passed_before_manifest_directory_or_payload_access",
        }
    ]
    selection_full_plan = [row for row in full_plan if row.source.split == "candidate_selection"]
    validation_plan = [row for row in full_plan if row.source.split == "validation"]
    selection_full, selection_stage0 = _load_selection_payloads_once(
        selection_full_plan, stage0_plan, paths, access_log
    )

    _, union_ordinals = _run_stage0(access, journal, selection_stage0)
    winner_ordinals, winner_q = _run_stage1(
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
    frozen_winners = journal.lookup("validation_firewall", "winners_frozen")
    winner_freeze_value = {
        "schema": "qwen3_initialization_anchor_tier_b_winner_freeze_v1",
        "domain_count": len(common.DOMAIN_IDS),
        "domain_ids": list(common.DOMAIN_IDS),
        "winners": winner_records,
        "union_shortlist_count": len(union_ordinals),
        "validation_payload_opened": False,
    }
    if frozen_winners is None:
        frozen_winners = journal.write_json(
            "validation_firewall", "winners_frozen", winner_freeze_value
        )
    elif json.loads(frozen_winners.read_text(encoding="utf-8")) != winner_freeze_value:
        raise common.ProtocolError("frozen winner state differs on resume")
    access_log.append(
        {
            "sequence": len(access_log),
            "event": "all_33_global_winners_state_backed_before_validation_payload_access",
            "winner_freeze_sha256": common.sha256_file(frozen_winners),
        }
    )

    validation_data = _load_plan_payloads(validation_plan, paths, access_log)
    selection_details = {}
    validation_details = {}
    selection_folds = {}
    validation_folds = {}
    for domain_index, domain_id in enumerate(common.DOMAIN_IDS):
        ordinal = int(winner_ordinals[domain_index])
        selection_details[domain_id] = _candidate_details(
            access, ordinal, domain_index, selection_full
        )
        validation_details[domain_id] = _candidate_details(
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
    result = {
        "schema": common.SCHEMA,
        "strict_ptq": True,
        "claim": {
            "procedural_anchor_discovery_only": True,
            "qwen_training_lineage_claimed": False,
            "tier_a_artifacts_modified": False,
            "claim_boundary": lock["claim_boundary"],
        },
        "pinned_panel": {"opened": False, "access_permitted": False},
        "bindings": {
            "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
            "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
            "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
            "tier_a_common_sha256": common.EXPECTED_TIER_A_COMMON_SHA256,
            "calibration_sha256": common.sha256_file(calibration_path.resolve()),
            "qwen_revision": common.QWEN_REVISION,
            "mcore_revision": common.MCORE_REVISION,
        },
        "backend": {
            "production": True,
            "name": "cupy_curand_philox_random_access_with_torch_cuda_parity",
            "parity": parity,
            "source_free_calibration": calibration,
        },
        "data_firewall": {
            "auxiliary_directory": str(aux_dir.resolve()),
            "exclusion_binding": exclusion,
            "excluded": [
                {
                    "tensor_name": row.tensor_name,
                    "basename": row.basename,
                    "payload_opened": False,
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
            "all_winners_frozen_before_validation": True,
            "excluded_payloads_opened": 0,
        },
        "candidate_space": {
            "logical_candidate_count": common.LOGICAL_CANDIDATES,
            "effective_candidate_count": common.EFFECTIVE_CANDIDATES,
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
            "event_count_before_result": len(journal.events_list),
            "events": journal.events_list,
        },
        "search": {
            "stage0_top_k_per_domain": common.STAGE0_TOP_K,
            "stage0_shard_count": 256,
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
        "decision": decision,
    }
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result_path


def cpu_preflight(workspace_root: Path | None = None) -> dict[str, Any]:
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("CPU preflight must run without torch/CuPy imports")
    lock = common.load_candidate_lock()
    rows = common.load_source_rows(workspace_root)
    stage0 = common.make_plan(rows, stage0=True)
    full = common.make_plan(rows, stage0=False)
    return {
        "schema": "qwen3_initialization_anchor_tier_b_cpu_preflight_v1",
        "status": "PASS_CUDA_NOT_IMPORTED_OR_TOUCHED",
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "logical_candidates": common.LOGICAL_CANDIDATES,
        "effective_candidates": common.EFFECTIVE_CANDIDATES,
        "domain_ids": list(common.DOMAIN_IDS),
        "stage0_plan_sha256": common.plan_sha256(stage0),
        "full_plan_sha256": common.plan_sha256(full),
        "stage0_coordinates": sum(len(row.fit) + len(row.score) for row in stage0),
        "full_coordinates": sum(len(row.fit) + len(row.score) for row in full),
        "selection_full_coordinates": sum(
            len(row.fit) + len(row.score) for row in full if row.source.split == "candidate_selection"
        ),
        "validation_full_coordinates": sum(
            len(row.fit) + len(row.score) for row in full if row.source.split == "validation"
        ),
        "equivalence_map": common.equivalence_map_object(),
        "equivalence_map_sha256": common.equivalence_map_sha256(),
        "representatives_per_seed_shard": len(common.representative_ordinals(0, 256)),
        "physical_ledger": common.physical_ledger(),
        "cuda_modules_imported": common.environment_has_cuda_imports(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--workspace-root", type=Path, default=None)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--output", type=Path, required=True)
    production = subparsers.add_parser("run")
    production.add_argument("--workspace-root", type=Path, required=True)
    production.add_argument("--aux-dir", type=Path, required=True)
    production.add_argument("--output-dir", type=Path, required=True)
    production.add_argument("--calibration", type=Path, required=True)
    production.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(cpu_preflight(args.workspace_root), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "calibrate":
        print(run_calibration(args.output))
        return 0
    if args.command == "run":
        print(
            run_gate(
                args.workspace_root,
                args.aux_dir,
                args.output_dir,
                args.calibration,
                resume=args.resume,
            )
        )
        return 0
    raise common.ProtocolError("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
