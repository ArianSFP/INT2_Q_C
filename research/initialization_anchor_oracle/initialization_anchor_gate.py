"""Run the frozen auxiliary Qwen3 initialization-anchor gate.

Production generation is delegated to PyTorch's CUDA Philox-backed ``normal_``
kernel.  CuPy receives those tensors over DLPack and performs the batched
float64 affine sufficient-statistic evaluation.  Production execution aborts
before opening a source payload unless offset-jump and interop parity pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common


@dataclass
class MatrixData:
    coordinate_row: common.CoordinateRow
    source_fit: np.ndarray
    source_score: np.ndarray
    gaussian_fit: np.ndarray
    gaussian_score: np.ndarray
    gaussian_metadata: dict[str, float]


def philox_offset_increment(
    numel: int,
    multi_processor_count: int,
    max_threads_per_multi_processor: int,
    *,
    block_size: int = 256,
    unroll_factor: int = 4,
) -> int:
    """Mirror PyTorch CUDA DistributionTemplates ``calc_execution_policy``.

    The frozen candidate paths use contiguous float/BF16 normal kernels, whose
    curand return type is float4 and therefore has unroll factor four.
    """
    if numel <= 0:
        return 0
    blocks_per_sm = max_threads_per_multi_processor // block_size
    if multi_processor_count <= 0 or blocks_per_sm <= 0:
        raise common.ProtocolError("invalid CUDA execution-policy properties")
    grid = min((numel + block_size - 1) // block_size, multi_processor_count * blocks_per_sm)
    iterations = (numel - 1) // (block_size * grid * unroll_factor) + 1
    return int(iterations * 4)


class TorchCudaPhiloxProvider:
    """Exact locked anchor generator plus CuPy staging/evaluation backend."""

    def __init__(self, device_index: int = 0):
        try:
            import torch
            import cupy as cp
        except Exception as error:  # pragma: no cover - production environment only
            raise common.ProtocolError(f"required CUDA libraries unavailable: {error}") from error
        if not torch.cuda.is_available():
            raise common.ProtocolError("PyTorch CUDA is unavailable")
        try:
            cupy_devices = int(cp.cuda.runtime.getDeviceCount())
        except Exception as error:
            raise common.ProtocolError(f"CuPy CUDA is unavailable: {error}") from error
        if cupy_devices <= device_index or torch.cuda.device_count() <= device_index:
            raise common.ProtocolError("requested common PyTorch/CuPy CUDA device is unavailable")
        self.torch = torch
        self.cp = cp
        self.device_index = int(device_index)
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        cp.cuda.Device(device_index).use()
        properties = torch.cuda.get_device_properties(self.device)
        self.multi_processor_count = int(properties.multi_processor_count)
        self.max_threads_per_multi_processor = int(properties.max_threads_per_multi_processor)
        self.device_name = str(properties.name)
        self._hf_offset_cache: dict[tuple[str, str, int, str], int] = {}

    def increment(self, numel: int) -> int:
        return philox_offset_increment(
            int(numel), self.multi_processor_count, self.max_threads_per_multi_processor
        )

    def _torch_dtype(self, dtype_path: str):
        if dtype_path == "fp32_then_bfloat16":
            return self.torch.float32
        if dtype_path == "bfloat16_direct":
            return self.torch.bfloat16
        raise common.ProtocolError(f"unsupported locked dtype path: {dtype_path}")

    def _normal_tensor(
        self, shape: tuple[int, ...], seed: int, offset: int, dtype_path: str
    ):
        torch = self.torch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        if not hasattr(generator, "set_offset") or not hasattr(generator, "get_offset"):
            raise common.ProtocolError("CUDA Generator offset API unavailable")
        generator.set_offset(int(offset))
        tensor = torch.empty(shape, dtype=self._torch_dtype(dtype_path), device=self.device)
        tensor.normal_(mean=0.0, std=0.02, generator=generator)
        if dtype_path == "fp32_then_bfloat16":
            tensor = tensor.to(torch.bfloat16)
        return tensor

    def parity_preflight(self, lock: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        cp = self.cp
        parity = lock["cuda_parity"]
        seed = int(parity["parity_seed"])
        numels = tuple(int(value) for value in parity["parity_numels"])
        dtype_results = []
        for dtype_path in lock["dtype_paths_in_order"]:
            dtype = self._torch_dtype(dtype_path)
            increment_rows = []
            for numel in numels:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                before = int(generator.get_offset())
                probe = torch.empty((numel,), dtype=dtype, device=self.device)
                probe.normal_(0.0, 0.02, generator=generator)
                torch.cuda.synchronize(self.device)
                after = int(generator.get_offset())
                expected = self.increment(numel)
                observed = after - before
                increment_rows.append({"numel": numel, "expected": expected, "observed": observed})
                if observed != expected:
                    raise common.ProtocolError(
                        f"PyTorch Philox increment parity failed for {dtype_path}, numel={numel}: "
                        f"expected {expected}, observed {observed}"
                    )
                del probe

            prefix_numels = (257, 65_537)
            target_numel = common.WEIGHTS_PER_MATRIX
            sequential = torch.Generator(device=self.device)
            sequential.manual_seed(seed)
            for numel in prefix_numels:
                discard = torch.empty((numel,), dtype=dtype, device=self.device)
                discard.normal_(0.0, 0.02, generator=sequential)
                del discard
            target_offset = int(sequential.get_offset())
            target_a = torch.empty((target_numel,), dtype=dtype, device=self.device)
            target_a.normal_(0.0, 0.02, generator=sequential)
            jumped = torch.Generator(device=self.device)
            jumped.manual_seed(seed)
            jumped.set_offset(target_offset)
            target_b = torch.empty((target_numel,), dtype=dtype, device=self.device)
            target_b.normal_(0.0, 0.02, generator=jumped)
            if dtype_path == "fp32_then_bfloat16":
                target_a = target_a.to(torch.bfloat16)
                target_b = target_b.to(torch.bfloat16)
            torch.cuda.synchronize(self.device)
            if not bool(torch.equal(target_a, target_b)):
                raise common.ProtocolError(f"sequential/offset-jump parity failed for {dtype_path}")

            # CuPy 14 may not expose BF16 as a public dtype on every build, so
            # interop is tested on the exact generated values after float32
            # widening, which is also the scoring representation.
            torch_f32 = target_a.float().contiguous()
            cupy_view = cp.from_dlpack(torch_f32)
            if int(cupy_view.size) != target_numel:
                raise common.ProtocolError("DLPack size parity failed")
            torch_hash = hashlib.sha256(torch_f32.cpu().numpy().astype("<f4", copy=False).tobytes()).hexdigest()
            cupy_hash = hashlib.sha256(cp.asnumpy(cupy_view).astype("<f4", copy=False).tobytes()).hexdigest()
            if torch_hash != cupy_hash:
                raise common.ProtocolError(f"PyTorch/CuPy DLPack value parity failed for {dtype_path}")
            dtype_results.append(
                {
                    "dtype_path": dtype_path,
                    "increment_checks": increment_rows,
                    "jump_target_offset": target_offset,
                    "jump_value_sha256_f32le": torch_hash,
                    "dlpack_value_sha256_f32le": cupy_hash,
                    "passed": True,
                }
            )
            del target_a, target_b, torch_f32, cupy_view
        torch.cuda.synchronize(self.device)
        cp.cuda.runtime.deviceSynchronize()
        return {
            "all_required_checks_passed": True,
            "torch_version": str(torch.__version__),
            "cupy_version": str(cp.__version__),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device_index": self.device_index,
            "device_name": self.device_name,
            "multi_processor_count": self.multi_processor_count,
            "max_threads_per_multi_processor": self.max_threads_per_multi_processor,
            "dtype_paths": dtype_results,
        }

    def _attention_numels(self) -> tuple[int, ...]:
        hidden = 2048
        return (
            4096 * hidden,
            512 * hidden,
            512 * hidden,
            hidden * 4096,
        )

    def _expert_numel(self) -> int:
        return common.WEIGHTS_PER_MATRIX

    def _hf_layer_post_init_increment(self) -> int:
        total = sum(self.increment(value) for value in self._attention_numels())
        total += self.increment(128 * 2048)  # router
        total += 128 * 3 * self.increment(self._expert_numel())
        return total

    def _hf_constructor_prefix_increment(self) -> int:
        # nn.Embedding constructor; every Linear has bias=False in the pinned
        # config, so exactly one uniform kernel is consumed per weight.
        total = self.increment(151_936 * 2_048)
        attention = sum(self.increment(value) for value in self._attention_numels())
        dense_mlp = (
            self.increment(6_144 * 2_048)
            + self.increment(6_144 * 2_048)
            + self.increment(2_048 * 6_144)
        )
        sparse_mlp = self.increment(128 * 2_048) + 128 * 3 * self.increment(self._expert_numel())
        # Qwen3MoeDecoderLayer v4.51.0 constructs and overwrites one attention
        # and one dense MLP before constructing the reachable attention/sparse
        # MLP.  Constructor RNG is consumed by both versions.
        total += 48 * (attention + dense_mlp + attention + sparse_mlp)
        return total

    def _hf_within_layer_offset(self, expert: int, role: str) -> int:
        total = sum(self.increment(value) for value in self._attention_numels())
        total += self.increment(128 * 2048)
        total += int(expert) * 3 * self.increment(self._expert_numel())
        role_index = {"gate": 0, "up": 1, "down": 2}[role]
        total += role_index * self.increment(self._expert_numel())
        return total

    def _hf_offset(self, candidate: common.Candidate, row: common.SourceRow) -> int:
        cache_key = (candidate.family, candidate.dtype_path, row.expert, row.role)
        if cache_key in self._hf_offset_cache:
            return self._hf_offset_cache[cache_key]
        if candidate.family == "hf451_tensor_reset":
            offset = 0
        elif candidate.family == "hf451_layer_reset":
            offset = self._hf_within_layer_offset(row.expert, row.role)
        else:
            offset = self.increment(151_936 * 2_048)
            offset += 15 * self._hf_layer_post_init_increment()
            offset += self._hf_within_layer_offset(row.expert, row.role)
            if candidate.family == "hf451_constructor_then_global_post_init":
                offset += self._hf_constructor_prefix_increment()
            elif candidate.family != "hf451_global_post_init":
                raise common.ProtocolError(f"unsupported HF family: {candidate.family}")
        self._hf_offset_cache[cache_key] = offset
        return offset

    def _generate_hf(
        self, candidate: common.Candidate, row: common.SourceRow, canonical_coordinates: np.ndarray
    ):
        torch = self.torch
        offset = self._hf_offset(candidate, row)
        tensor = self._normal_tensor(row.raw_shape, candidate.seed, offset, candidate.dtype_path)
        native = common.canonical_to_native_flat(row.role, canonical_coordinates)
        index = torch.as_tensor(native, dtype=torch.int64, device=self.device)
        selected = tensor.reshape(-1).index_select(0, index).float().contiguous()
        result = self.cp.from_dlpack(selected)
        del tensor, index, selected
        return result

    @staticmethod
    def _mcore_parallel_position(candidate: common.Candidate, expert: int) -> tuple[int, int, int, int]:
        pp = candidate.pipeline_parallel_size
        layers_per_stage = 48 // pp
        pp_rank = 15 // layers_per_stage
        local_layer = 15 - pp_rank * layers_per_stage
        ep = candidate.expert_parallel_size
        local_experts = 128 // ep
        if candidate.expert_assignment == "contiguous":
            ep_rank = expert // local_experts
            local_expert = expert % local_experts
        elif candidate.expert_assignment == "round_robin":
            ep_rank = expert % ep
            local_expert = expert // ep
        else:
            raise common.ProtocolError("unsupported MCore expert assignment")
        return pp_rank, local_layer, ep_rank, local_expert

    def _mcore_target_offset(
        self,
        candidate: common.Candidate,
        row: common.SourceRow,
        local_layer: int,
        local_expert: int,
    ) -> tuple[int, int, str]:
        etp = candidate.expert_tensor_parallel_size
        local_rows = common.ROWS // etp
        separate_numel = local_rows * common.COLUMNS
        local_experts = 128 // candidate.expert_parallel_size
        prior_experts = local_layer * local_experts + local_expert
        packing = candidate.expert_projection_packing
        if packing == "separate_gate_up_down":
            per_expert = 3 * self.increment(separate_numel)
            offset = prior_experts * per_expert
            if row.role == "up":
                offset += self.increment(separate_numel)
            elif row.role == "down":
                offset += 2 * self.increment(separate_numel)
            return offset, separate_numel, "separate"
        if packing == "fused_gate_up_then_down":
            fused_numel = 2 * separate_numel
            per_expert = self.increment(fused_numel) + self.increment(separate_numel)
            offset = prior_experts * per_expert
            if row.role == "down":
                offset += self.increment(fused_numel)
                return offset, separate_numel, "down"
            return offset, fused_numel, "fused_up"
        raise common.ProtocolError("unsupported MCore expert projection packing")

    def _generate_mcore(
        self, candidate: common.Candidate, row: common.SourceRow, canonical_coordinates: np.ndarray
    ):
        torch = self.torch
        pp_rank, local_layer, ep_rank, local_expert = self._mcore_parallel_position(candidate, row.expert)
        etp = candidate.expert_tensor_parallel_size
        if common.ROWS % etp:
            raise common.ProtocolError("frozen ETP size does not divide expert intermediate width")
        local_rows = common.ROWS // etp
        canonical_coordinates = np.asarray(canonical_coordinates, dtype=np.int64)
        canonical_rows = canonical_coordinates // common.COLUMNS
        canonical_columns = canonical_coordinates % common.COLUMNS
        output = torch.empty((len(canonical_coordinates),), dtype=torch.float32, device=self.device)
        assigned = np.zeros(len(canonical_coordinates), dtype=bool)
        for etp_rank in range(etp):
            low = etp_rank * local_rows
            high = low + local_rows
            mask = (canonical_rows >= low) & (canonical_rows < high)
            if not bool(np.any(mask)):
                continue
            offset, target_numel, target_kind = self._mcore_target_offset(
                candidate, row, local_layer, local_expert
            )
            effective_seed = candidate.seed + 100 * pp_rank + 1024 + 100 * ep_rank + etp_rank
            if target_kind == "fused_up":
                shape = (2 * local_rows, common.COLUMNS)
            elif row.role == "up":
                shape = (local_rows, common.COLUMNS)
            else:
                shape = (common.COLUMNS, local_rows)
            if math.prod(shape) != target_numel:
                raise common.ProtocolError("internal MCore target-shape accounting mismatch")
            tensor = self._normal_tensor(shape, effective_seed, offset, candidate.dtype_path)
            local_canonical_rows = canonical_rows[mask] - low
            if target_kind == "fused_up":
                native = (local_canonical_rows + local_rows) * common.COLUMNS + canonical_columns[mask]
            elif row.role == "up":
                native = local_canonical_rows * common.COLUMNS + canonical_columns[mask]
            else:
                native = canonical_columns[mask] * local_rows + local_canonical_rows
            native_index = torch.as_tensor(native, dtype=torch.int64, device=self.device)
            output_index_np = np.flatnonzero(mask).astype(np.int64)
            output_index = torch.as_tensor(output_index_np, dtype=torch.int64, device=self.device)
            values = tensor.reshape(-1).index_select(0, native_index).float()
            output.index_copy_(0, output_index, values)
            assigned[mask] = True
            del tensor, native_index, output_index, values
        if not bool(np.all(assigned)):
            raise common.ProtocolError("MCore ETP reconstruction left coordinates unassigned")
        result = self.cp.from_dlpack(output.contiguous())
        del output
        return result

    def generate_coordinates(
        self, candidate: common.Candidate, row: common.SourceRow, canonical_coordinates: np.ndarray
    ):
        if candidate.family.startswith("hf451_"):
            return self._generate_hf(candidate, row, canonical_coordinates)
        if candidate.family == "mcore_expert_parallel_stream":
            return self._generate_mcore(candidate, row, canonical_coordinates)
        raise common.ProtocolError(f"unimplemented candidate family: {candidate.family}")


def _load_matrix_data(
    rows: Sequence[common.CoordinateRow],
    paths: Mapping[str, Path],
    access_log: list[dict[str, Any]],
) -> list[MatrixData]:
    result: list[MatrixData] = []
    for coordinate_row in rows:
        row = coordinate_row.source
        all_coordinates = np.asarray(coordinate_row.fit + coordinate_row.score, dtype=np.int64)
        values = common.read_source_coordinates(paths[row.tensor_name], row, all_coordinates, verify_hash=True)
        fit_count = len(coordinate_row.fit)
        source_fit = values[:fit_count]
        source_score = values[fit_count:]
        gaussian_fit, gaussian_score, gaussian_metadata = common.matched_gaussian_values(
            row.tensor_name,
            coordinate_row.fit,
            coordinate_row.score,
            source_fit,
        )
        access_log.append(
            {
                "sequence": len(access_log),
                "event": "payload_opened_and_hash_verified",
                "tensor_name": row.tensor_name,
                "split": row.split,
                "sha256": row.sha256,
            }
        )
        result.append(
            MatrixData(
                coordinate_row=coordinate_row,
                source_fit=source_fit,
                source_score=source_score,
                gaussian_fit=gaussian_fit,
                gaussian_score=gaussian_score,
                gaussian_metadata=gaussian_metadata,
            )
        )
    return result


def _batched_affine_sse(cp, w_fit, w_score, g_fit, g_score):
    w_fit64 = cp.asarray(w_fit, dtype=cp.float64)
    w_score64 = cp.asarray(w_score, dtype=cp.float64)
    g_fit64 = cp.asarray(g_fit, dtype=cp.float64)
    g_score64 = cp.asarray(g_score, dtype=cp.float64)
    n = int(w_fit64.size)
    sum_w = cp.sum(w_fit64, dtype=cp.float64)
    sum_g = cp.sum(g_fit64, axis=1, dtype=cp.float64)
    sum_gg = cp.sum(g_fit64 * g_fit64, axis=1, dtype=cp.float64)
    sum_wg = cp.sum(g_fit64 * w_fit64[None, :], axis=1, dtype=cp.float64)
    centered_gg = sum_gg - sum_g * sum_g / n
    centered_wg = sum_wg - sum_w * sum_g / n
    alpha = cp.where(centered_gg > 0.0, centered_wg / centered_gg, 0.0)
    mean_w = sum_w / n
    mu = mean_w - alpha * (sum_g / n)
    residual = w_score64[None, :] - (mu[:, None] + alpha[:, None] * g_score64)
    sse = cp.sum(residual * residual, axis=1, dtype=cp.float64)
    baseline = cp.sum((w_score64 - mean_w) ** 2, dtype=cp.float64)
    return sse, baseline


def evaluate_candidate_batch(
    provider: TorchCudaPhiloxProvider,
    candidates: Sequence[common.Candidate],
    matrices: Sequence[MatrixData],
) -> dict[str, np.ndarray]:
    cp = provider.cp
    batch = len(candidates)
    source_sse = cp.zeros(batch, dtype=cp.float64)
    source_baseline = cp.zeros(batch, dtype=cp.float64)
    gaussian_sse = cp.zeros(batch, dtype=cp.float64)
    gaussian_baseline = cp.zeros(batch, dtype=cp.float64)
    permuted_sse = cp.zeros(batch, dtype=cp.float64)
    permuted_baseline = cp.zeros(batch, dtype=cp.float64)
    for matrix in matrices:
        coordinate_row = matrix.coordinate_row
        coordinates = np.asarray(coordinate_row.fit + coordinate_row.score, dtype=np.int64)
        fit_count = len(coordinate_row.fit)
        anchors = cp.empty((batch, len(coordinates)), dtype=cp.float32)
        for index, candidate in enumerate(candidates):
            anchors[index] = provider.generate_coordinates(candidate, coordinate_row.source, coordinates)
        g_fit = anchors[:, :fit_count]
        g_score = anchors[:, fit_count:]
        sse, baseline = _batched_affine_sse(
            cp, matrix.source_fit, matrix.source_score, g_fit, g_score
        )
        source_sse += sse
        source_baseline += baseline
        sse, baseline = _batched_affine_sse(
            cp, matrix.gaussian_fit, matrix.gaussian_score, g_fit, g_score
        )
        gaussian_sse += sse
        gaussian_baseline += baseline
        fit_permutation = common.deterministic_permutation(
            coordinate_row.source.tensor_name, "fit", fit_count
        )
        score_permutation = common.deterministic_permutation(
            coordinate_row.source.tensor_name, "score", len(coordinate_row.score)
        )
        sse, baseline = _batched_affine_sse(
            cp,
            matrix.source_fit,
            matrix.source_score,
            g_fit[:, fit_permutation],
            g_score[:, score_permutation],
        )
        permuted_sse += sse
        permuted_baseline += baseline
        del anchors, g_fit, g_score, sse, baseline
    cp.cuda.runtime.deviceSynchronize()
    result = {
        "source_sse": cp.asnumpy(source_sse),
        "source_baseline": cp.asnumpy(source_baseline),
        "gaussian_sse": cp.asnumpy(gaussian_sse),
        "gaussian_baseline": cp.asnumpy(gaussian_baseline),
        "permuted_sse": cp.asnumpy(permuted_sse),
        "permuted_baseline": cp.asnumpy(permuted_baseline),
    }
    return result


def _candidate_search(
    provider: TorchCudaPhiloxProvider,
    candidates: Sequence[common.Candidate],
    matrices: Sequence[MatrixData],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    summaries: list[dict[str, Any]] = []
    best = {"source": 0, "gaussian": 0, "permuted": 0}
    best_q = {"source": math.inf, "gaussian": math.inf, "permuted": math.inf}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        values = evaluate_candidate_batch(provider, batch, matrices)
        for local, candidate in enumerate(batch):
            metrics = {}
            for domain in ("source", "gaussian", "permuted"):
                sse = float(values[f"{domain}_sse"][local])
                baseline = float(values[f"{domain}_baseline"][local])
                metric = common.metric_from_sse(sse, baseline)
                metrics[domain] = {"sse": sse, "baseline_sse": baseline, **metric}
                if metric["q"] < best_q[domain]:
                    best_q[domain] = metric["q"]
                    best[domain] = candidate.ordinal
            summaries.append({"ordinal": candidate.ordinal, "id": candidate.id, **metrics})
        print(
            f"candidate batch {min(start + len(batch), len(candidates))}/{len(candidates)}",
            flush=True,
        )
    return summaries, best


def _detailed_candidate_statistics(
    provider: TorchCudaPhiloxProvider,
    candidate: common.Candidate,
    matrices: Sequence[MatrixData],
    domain: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for matrix in matrices:
        coordinate_row = matrix.coordinate_row
        coordinates = np.asarray(coordinate_row.fit + coordinate_row.score, dtype=np.int64)
        fit_count = len(coordinate_row.fit)
        anchor = provider.generate_coordinates(candidate, coordinate_row.source, coordinates)
        anchor_np = provider.cp.asnumpy(anchor).astype(np.float32, copy=False)
        g_fit = anchor_np[:fit_count]
        g_score = anchor_np[fit_count:]
        if domain == "source":
            w_fit, w_score = matrix.source_fit, matrix.source_score
        elif domain == "gaussian":
            w_fit, w_score = matrix.gaussian_fit, matrix.gaussian_score
        elif domain == "permuted":
            w_fit, w_score = matrix.source_fit, matrix.source_score
            g_fit = g_fit[
                common.deterministic_permutation(coordinate_row.source.tensor_name, "fit", len(g_fit))
            ]
            g_score = g_score[
                common.deterministic_permutation(coordinate_row.source.tensor_name, "score", len(g_score))
            ]
        else:
            raise common.ProtocolError(f"unsupported detail domain: {domain}")
        fit = common.fit_affine_moments(w_fit, g_fit)
        score = common.score_affine_moments(
            w_score,
            g_score,
            float(fit["alpha"]),
            float(fit["mu"]),
            float(fit["fit_mean_w"]),
        )
        rows.append(
            {
                "tensor_name": coordinate_row.source.tensor_name,
                "expert": coordinate_row.source.expert,
                "role": coordinate_row.source.role,
                "fit": fit,
                "score": score,
            }
        )
    return rows


def cpu_preflight(workspace_root: Path | None = None) -> dict[str, Any]:
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("CPU preflight must run before importing torch or CuPy")
    lock = common.load_candidate_lock()
    rows = common.load_frozen_source_rows(workspace_root)
    plan = common.make_coordinate_plan(rows)
    candidates = common.enumerate_candidates(lock)
    return {
        "schema": "qwen3_initialization_anchor_cpu_preflight_v1",
        "status": "PASS_CUDA_NOT_IMPORTED_OR_TOUCHED",
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "source_rows": len(rows),
        "eligible_rows": sum(not row.excluded for row in rows),
        "candidate_selection_rows": sum((not row.excluded) and row.split == "candidate_selection" for row in rows),
        "validation_rows": sum((not row.excluded) and row.split == "validation" for row in rows),
        "excluded_tensors": [row.tensor_name for row in rows if row.excluded],
        "coordinate_plan_sha256": common.coordinate_plan_sha256(plan),
        "fit_coordinates": sum(len(row.fit) for row in plan),
        "score_coordinates": sum(len(row.score) for row in plan),
        "candidate_count": len(candidates),
        "first_candidate": candidates[0].id,
        "last_candidate": candidates[-1].id,
        "physical_ledger": common.physical_ledger(),
        "cuda_modules_imported": common.environment_has_cuda_imports(),
    }


def run_gate(aux_dir: Path, output_dir: Path, workspace_root: Path | None = None) -> Path:
    lock = common.load_candidate_lock()
    rows = common.load_frozen_source_rows(workspace_root)
    _, exclusion_binding = common.load_exclusion_binding(workspace_root)
    paths = common.validate_aux_directory(aux_dir, rows)
    plan = common.make_coordinate_plan(rows)
    candidates = common.enumerate_candidates(lock)

    # Mandatory source-independent parity.  No source payload has been opened.
    provider = TorchCudaPhiloxProvider(device_index=0)
    parity = provider.parity_preflight(lock)
    if not parity.get("all_required_checks_passed"):
        raise common.ProtocolError("CUDA parity did not pass")

    access_log: list[dict[str, Any]] = [
        {"sequence": 0, "event": "cuda_parity_passed_before_payload_access"}
    ]
    selection_plan = [row for row in plan if row.source.split == "candidate_selection"]
    validation_plan = [row for row in plan if row.source.split == "validation"]
    selection_data = _load_matrix_data(selection_plan, paths, access_log)
    summaries, best_ordinals = _candidate_search(
        provider,
        candidates,
        selection_data,
        int(lock["execution"]["candidate_batch_size"]),
    )
    winners = {domain: candidates[ordinal] for domain, ordinal in best_ordinals.items()}
    access_log.append(
        {
            "sequence": len(access_log),
            "event": "global_candidates_frozen_before_validation_payload_access",
            "winners": {domain: candidate.id for domain, candidate in winners.items()},
        }
    )

    # The validation files are opened only after all three global candidate
    # selections (source and the two matched-search controls) are immutable.
    validation_data = _load_matrix_data(validation_plan, paths, access_log)
    selection_details = {
        domain: _detailed_candidate_statistics(provider, candidate, selection_data, domain)
        for domain, candidate in winners.items()
    }
    validation_details = {
        domain: _detailed_candidate_statistics(provider, candidate, validation_data, domain)
        for domain, candidate in winners.items()
    }
    selection_folds = {
        domain: common.fold_statistics(detail) for domain, detail in selection_details.items()
    }
    validation_folds = {
        domain: common.fold_statistics(detail) for domain, detail in validation_details.items()
    }
    decision = common.make_decision(
        validation_folds["source"],
        validation_folds["gaussian"]["pooled"]["capture"],
        validation_folds["permuted"]["pooled"]["capture"],
    )

    excluded_rows = [row for row in rows if row.excluded]
    eligible_rows = [row for row in rows if not row.excluded]
    script_path = Path(__file__).resolve()
    result = {
        "schema": common.SCHEMA,
        "protocol": {
            "candidate_lock_status": lock["status"],
            "global_candidate_only": True,
            "scientific_cli_knobs": False,
        },
        "strict_ptq": True,
        "pinned_panel": {"opened": False, "access_permitted": False},
        "backend": {
            "production": True,
            "name": "cupy_with_pytorch_cuda_philox",
            "parity": parity,
        },
        "bindings": {
            "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
            "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
            "source_manifest_sha256": common.EXPECTED_SOURCE_MANIFEST_SHA256,
            "source_freeze_sha256": common.EXPECTED_SOURCE_FREEZE_SHA256,
            "exclusion_manifest_sha256": common.EXPECTED_EXCLUSION_MANIFEST_SHA256,
            "exclusion_intersection_lock_sha256": common.EXPECTED_EXCLUSION_INTERSECTION_SHA256,
            "runner_sha256": common.sha256_file(script_path),
            "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
            "revision": common.REVISION,
        },
        "data_firewall": {
            "auxiliary_directory": str(aux_dir.resolve()),
            "exact_file_count": len(rows),
            "eligible_tensor_count": len(eligible_rows),
            "candidate_selection_tensor_count": len(selection_plan),
            "validation_tensor_count": len(validation_plan),
            "excluded": [
                {
                    "tensor_name": row.tensor_name,
                    "basename": row.basename,
                    "reason": "tensor identity occurs in bound heldout32 exclusion manifest",
                    "payload_opened": False,
                }
                for row in excluded_rows
            ],
            "exclusion_binding": exclusion_binding,
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
            "candidate_selection_before_validation_payload_open": True,
        },
        "sampler": {
            "total_coordinates": common.TOTAL_COORDINATES,
            "fit_coordinates": common.FIT_COORDINATES,
            "score_coordinates": common.SCORE_COORDINATES,
            "coordinate_plan_sha256": common.coordinate_plan_sha256(plan),
            "per_tensor": common.coordinate_plan_json(plan),
        },
        "candidate_selection": {
            "candidate_count": len(candidates),
            "candidate_order_sha256": common.sha256_bytes(
                b"\n".join(candidate.id.encode("utf-8") for candidate in candidates)
            ),
            "winners": {domain: candidate.to_json() for domain, candidate in winners.items()},
            "summaries": summaries,
            "training_details": selection_details,
            "training_folds": selection_folds,
        },
        "controls": {
            "matched_gaussian": {
                "same_candidate_count": len(candidates),
                "winner": winners["gaussian"].id,
                "generation": "fit-only moments plus stateless SHA256 Box-Muller",
            },
            "permuted_anchor": {
                "same_candidate_count": len(candidates),
                "winner": winners["permuted"].id,
                "generation": "SHA256 fixed permutation independently within matrix and split",
            },
        },
        "validation": {
            "source_winner": winners["source"].id,
            "details": validation_details,
            "folds": validation_folds,
        },
        "physical_ledger": common.physical_ledger(),
        "decision": decision,
        "claim_boundary": lock["claim_boundary"],
    }
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise common.ProtocolError("output directory already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / str(lock["execution"]["output_json"])
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="run CUDA-free static preflight only")
    parser.add_argument("--workspace-root", type=Path, default=None, help="operational root; manifests remain hash-bound")
    parser.add_argument("--aux-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=("cupy",), default="cupy")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight:
        if args.aux_dir is not None or args.output_dir is not None:
            raise common.ProtocolError("CPU preflight accepts no source/output path")
        print(json.dumps(cpu_preflight(args.workspace_root), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.aux_dir is None or args.output_dir is None:
        raise common.ProtocolError("production run requires --aux-dir and --output-dir")
    result_path = run_gate(args.aux_dir, args.output_dir, args.workspace_root)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
