"""Authorized-GPU parity for the pinned Tier-C grouped initializer.

Importing this module is CUDA-free.  ``run_parity`` is intentionally the only
entry point that imports PyTorch, Transformer Engine, Megatron-Core, or CuPy.
It must run before any auxiliary manifest, directory, or payload operation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common
import kernels


SINGLE_PARAM_ENV = "NVTE_GROUPED_LINEAR_SINGLE_PARAM"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_RUNTIME_SOURCE_HASHES = {
    "mcore_pyproject": "1c1837d50833f18e33fbc02d012f7aafcc99d12349a2cbba040fd4ffc7079cb5",
    "mcore_experts": "80f889f30cf56eefc85bc1a8908a4cb92014787f5b3a1490ee19289f3d960620",
    "mcore_te_wrapper": "efc1b8e6cd9517862ec9cdd25eefb6aa2d8eceb34222b65c82bcbc3215932650",
    "te_grouped_linear": "84d27f52ecaee38de2e324b6aa5b5fe9625129d5835183fd529f9cdeac634143",
    "te_base": "67d4a7665150761a84f8f77123f5741807e80af72469aa19bad9a9ad91704e56",
    "mcore_initialize": "76eb1beb86c18c3b96dfc97142f94b6acadd413835e207bcfbe80a36c1dbd801",
    "mcore_rng": "4fe12e3feab6135ec273adde452605d85434374918c6261ba05274043c16a2f0",
}
RUNTIME_TRACE_CASES = (
    {"local_experts": 4, "etp": 1},
    {"local_experts": 2, "etp": 2},
    {"local_experts": 128, "etp": 4},
    {"local_experts": 16, "etp": 8},
)


def _float_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def _raw_sha(tensor, torch) -> str:
    contiguous = tensor.detach().contiguous()
    if contiguous.dtype == torch.bfloat16:
        payload = contiguous.view(torch.uint16).cpu().numpy().astype("<u2", copy=False).tobytes()
    elif contiguous.dtype == torch.float32:
        payload = contiguous.cpu().numpy().astype("<f4", copy=False).tobytes()
    else:
        raise common.ProtocolError(f"unsupported parity tensor dtype: {contiguous.dtype}")
    return common.sha256_bytes(payload)


def _rng_sha(torch, device) -> str:
    state = torch.cuda.get_rng_state(device)
    return common.sha256_bytes(state.cpu().numpy().astype(np.uint8, copy=False).tobytes())


def _default_generator(torch, device):
    generator = torch.cuda.default_generators[device.index]
    if not hasattr(generator, "get_offset") or not hasattr(generator, "set_offset"):
        raise common.ProtocolError("PyTorch default CUDA generator offset API unavailable")
    return generator


def _torch_generate(torch, device, seed: int, offset: int, shape: tuple[int, ...], dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    if not hasattr(generator, "set_offset") or not hasattr(generator, "get_offset"):
        raise common.ProtocolError("PyTorch CUDA Generator offset API unavailable")
    generator.set_offset(int(offset))
    tensor = torch.empty(shape, dtype=dtype, device=device)
    tensor.normal_(0.0, 0.02, generator=generator)
    return tensor, int(generator.get_offset())


def load_source_trace(path: Path) -> dict[str, Any]:
    path = common.require_regular_file_before_resolve(
        path, "source-trace receipt"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    common.strict_keys(
        value,
        {
            "schema", "status", "mcore_revision", "transformer_engine_revision",
            "transformer_engine_source_version", "transformer_engine_pypi_version_policy",
            "files", "semantic_edges", "procedural_geometry_trace", "claim_boundary",
            "access_attestation", "execution_boundary", "receipt_sha256",
        },
        "source-trace receipt",
    )
    if value.get("schema") != "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_source_trace_v2":
        raise common.ProtocolError("source-trace receipt schema mismatch")
    if value.get("status") != "PASS_EXACT_SEVEN_FILE_SOURCE_TRACE_RUNTIME_PARITY_STILL_REQUIRED":
        raise common.ProtocolError("source-trace receipt did not pass")
    access = value.get("access_attestation")
    if not isinstance(access, Mapping):
        raise common.ProtocolError("source-trace access attestation missing")
    common.strict_keys(
        access,
        {
            "qwen_payload_or_manifest_opened_statted_or_hashed",
            "cuda_or_forbidden_runtime_imported", "only_the_seven_explicit_source_files_opened",
        },
        "source-trace access attestation",
    )
    if access != {
        "qwen_payload_or_manifest_opened_statted_or_hashed": False,
        "cuda_or_forbidden_runtime_imported": False,
        "only_the_seven_explicit_source_files_opened": True,
    }:
        raise common.ProtocolError("source-trace receipt is not source-only")
    execution_boundary = value.get("execution_boundary")
    if (
        not isinstance(execution_boundary, Mapping)
        or execution_boundary.get("schema")
        != "qwen3_tier_c_grouped_v5_path_boundary_v1"
        or execution_boundary.get("action") != "SOURCE_TRACE_CREATE_ONCE"
        or execution_boundary.get("pairwise_lexical_inode_and_mount_disjoint") is not True
        or execution_boundary.get("revalidation_required_before_every_create_new") is not True
    ):
        raise common.ProtocolError("source-trace execution boundary is absent or malformed")
    clean = dict(value)
    observed = clean.pop("receipt_sha256", None)
    if observed != common.sha256_bytes(common.canonical_json_bytes(clean)):
        raise common.ProtocolError("source-trace receipt internal SHA-256 mismatch")
    files = value.get("files")
    if not isinstance(files, Mapping) or set(files) != set(EXPECTED_RUNTIME_SOURCE_HASHES):
        raise common.ProtocolError("source-trace seven-file set mismatch")
    for label, expected in EXPECTED_RUNTIME_SOURCE_HASHES.items():
        row = files[label]
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
            raise common.ProtocolError(f"source-trace malformed file row: {label}")
        if row.get("sha256") != expected:
            raise common.ProtocolError(f"source-trace binding mismatch: {label}")
    boundary = value.get("claim_boundary")
    required_true = {
        "source_proves_ordinary_bf16_fc1_all_then_fc2_all_constructor_callback_order",
        "source_proves_numbered_then_copy_pack_order",
    }
    required_false = {
        "copy_pack_bitwise_and_terminal_rng_parity_source_only",
        "pytorch_cupy_philox_parity_source_only", "direct_bf16_vs_cast_parity_source_only",
        "numeric_full_pre_layer_15_expert_rng_lifecycle_source_only",
        "pp_cross_seed_equivalence_used",
    }
    if not isinstance(boundary, Mapping) or set(boundary) != required_true | required_false:
        raise common.ProtocolError("source-trace claim boundary keys changed")
    if any(boundary[key] is not True for key in required_true) or any(boundary[key] is not False for key in required_false):
        raise common.ProtocolError("source-trace claim boundary values changed")
    return value


def _runtime_source_hashes() -> dict[str, dict[str, str]]:
    import transformer_engine.pytorch.module.base as te_base
    import transformer_engine.pytorch.module.grouped_linear as te_grouped
    import megatron.core.extensions.transformer_engine as mcore_wrapper
    import megatron.core.tensor_parallel.random as mcore_rng
    import megatron.core.transformer.moe.experts as mcore_experts
    import megatron.training.initialize as mcore_initialize

    modules = {
        "mcore_experts": mcore_experts,
        "mcore_te_wrapper": mcore_wrapper,
        "mcore_initialize": mcore_initialize,
        "mcore_rng": mcore_rng,
        "te_grouped_linear": te_grouped,
        "te_base": te_base,
    }
    original_paths: dict[str, Path] = {}
    for label, module in modules.items():
        source = inspect.getsourcefile(module)
        if source is None:
            raise common.ProtocolError(f"cannot locate runtime source: {label}")
        # Preserve the runtime's original spelling until the component-safe
        # regular-file check has run.
        original_paths[label] = Path(source)
    experts_path = common.require_regular_file_before_resolve(
        original_paths["mcore_experts"], "runtime source mcore_experts"
    )
    if len(experts_path.parents) < 5:
        raise common.ProtocolError("cannot derive MCore checkout root from experts source")
    original_paths["mcore_pyproject"] = experts_path.parents[4] / "pyproject.toml"
    result = {}
    for label, original in original_paths.items():
        path = common.require_regular_file_before_resolve(original, f"runtime source {label}")
        observed = common.sha256_file(path)
        if observed != EXPECTED_RUNTIME_SOURCE_HASHES[label]:
            raise common.ProtocolError(f"runtime source SHA-256 mismatch: {label}")
        result[label] = {"path": str(original), "sha256": observed}
    return result


def _te_version(transformer_engine, runtime_sources: Mapping[str, Mapping[str, str]]) -> str:
    candidates = []
    version = getattr(transformer_engine, "__version__", None)
    if version is not None:
        candidates.append(str(version))
    for distribution in ("transformer_engine", "transformer-engine"):
        try:
            candidates.append(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            pass
    candidates = list(dict.fromkeys(candidates))
    allowed = {common.TE_SOURCE_VERSION, common.TE_PYPI_VERSION}
    observed = [row for row in candidates if row in allowed]
    if len(observed) != 1:
        raise common.ProtocolError(
            f"exact Transformer Engine version must be one of {sorted(allowed)!r}; observed {candidates!r}"
        )
    if observed[0] == common.TE_PYPI_VERSION:
        if set(runtime_sources) != set(EXPECTED_RUNTIME_SOURCE_HASHES):
            raise common.ProtocolError("PyPI TE 2.18.0 requires all seven pinned runtime sources")
        for label, expected in EXPECTED_RUNTIME_SOURCE_HASHES.items():
            if runtime_sources[label].get("sha256") != expected:
                raise common.ProtocolError(f"PyPI TE source pin mismatch: {label}")
    return observed[0]


def _members(module, count: int, single: bool, torch):
    if not single:
        result = [getattr(module, f"weight{index}") for index in range(count)]
    else:
        if not bool(getattr(module, "single_grouped_weight", False)):
            raise common.ProtocolError("single_grouped_weight was silently forced false")
        weight = getattr(module, "weight", None)
        if weight is None:
            raise common.ProtocolError("single grouped module has no packed weight")
        quantized = getattr(weight, "quantized_tensors", None)
        if quantized is not None:
            result = list(quantized)
        elif weight.numel() % count == 0:
            result = list(weight.reshape(count, -1))
        else:
            raise common.ProtocolError("cannot split single grouped weight into members")
    if len(result) != count:
        raise common.ProtocolError("grouped weight member count mismatch")
    return result


def _te_init_callback(torch, generator, device, events, projection: str):
    """Initialize a TE leaf parameter without autograd tracking.

    ``torch.no_grad`` changes only autograd recording. The direct-BF16 mutation,
    generator, callback ordering, byte hashes, and RNG-state checks remain the
    exact operations already bound by runtime parity.
    """
    def initialize(tensor):
        before = int(generator.get_offset())
        pre_state = _rng_sha(torch, device)
        with torch.no_grad():
            tensor.normal_(0.0, 0.02)
        after = int(generator.get_offset())
        events.append(
            {
                "ordinal": len(events),
                "projection": projection,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "offset_before": before,
                "offset_after": after,
                "increment": after - before,
                "rng_state_before_sha256": pre_state,
                "rng_state_after_sha256": _rng_sha(torch, device),
                "weight_sha256": _raw_sha(tensor, torch),
            }
        )
    return initialize


def _run_te_storage_mode(torch, te, device, *, local_experts: int, etp: int, single: bool):
    if os.environ.get(SINGLE_PARAM_ENV) != "1":
        raise common.ProtocolError(f"{SINGLE_PARAM_ENV}=1 must be set before TE import")
    h = common.ROWS // etp
    generator = _default_generator(torch, device)
    torch.cuda.manual_seed(0x1A2B3C4D)
    generator.set_offset(0)
    events = []

    fc1 = te.pytorch.GroupedLinear(
        local_experts,
        common.COLUMNS,
        2 * h,
        init_method=_te_init_callback(torch, generator, device, events, "fc1"),
        bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=None,
        device=device,
        single_grouped_weight=single,
        name="tier_c_parity_fc1",
    )
    fc2 = te.pytorch.GroupedLinear(
        local_experts,
        h,
        common.COLUMNS,
        init_method=_te_init_callback(torch, generator, device, events, "fc2"),
        bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=None,
        device=device,
        single_grouped_weight=single,
        name="tier_c_parity_fc2",
    )
    torch.cuda.synchronize(device)
    expected_projections = ["fc1"] * local_experts + ["fc2"] * local_experts
    if [row["projection"] for row in events] != expected_projections:
        raise common.ProtocolError("TE init callback order is not FC1-all then FC2-all")
    expected_shapes = [[2 * h, common.COLUMNS]] * local_experts + [
        [common.COLUMNS, h]
    ] * local_experts
    if [row["shape"] for row in events] != expected_shapes:
        raise common.ProtocolError("TE init callback shapes differ from grouped Qwen geometry")
    if any(row["dtype"] != "torch.bfloat16" for row in events):
        raise common.ProtocolError("TE direct BF16 initialization dtype mismatch")
    if any(row["increment"] <= 0 for row in events):
        raise common.ProtocolError("TE initialization did not advance the RNG")
    fc1_members = _members(fc1, local_experts, single, torch)
    fc2_members = _members(fc2, local_experts, single, torch)
    member_hashes = [_raw_sha(row, torch) for row in fc1_members + fc2_members]
    callback_hashes = [row["weight_sha256"] for row in events]
    if member_hashes != callback_hashes:
        raise common.ProtocolError("single-grouped packing changed initialized member bytes")
    result = {
        "single_grouped_weight": single,
        "local_experts": local_experts,
        "etp": etp,
        "h": h,
        "events": events,
        "events_sha256": common.sha256_bytes(common.canonical_json_bytes(events)),
        "member_hashes": member_hashes,
        "final_rng_offset": int(generator.get_offset()),
        "final_rng_state_sha256": _rng_sha(torch, device),
        "fc1_single_grouped_weight_observed": bool(getattr(fc1, "single_grouped_weight", False)),
        "fc2_single_grouped_weight_observed": bool(getattr(fc2, "single_grouped_weight", False)),
    }
    del fc1, fc2, fc1_members, fc2_members
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


def _te_storage_parity(torch, te, device) -> list[dict[str, Any]]:
    rows = []
    for case in RUNTIME_TRACE_CASES:
        numbered = _run_te_storage_mode(torch, te, device, single=False, **case)
        grouped = _run_te_storage_mode(torch, te, device, single=True, **case)
        for key in ("events_sha256", "member_hashes", "final_rng_offset", "final_rng_state_sha256"):
            if numbered[key] != grouped[key]:
                raise common.ProtocolError(f"numbered/copy-packed TE parity failed: {key}/{case}")
        if grouped["fc1_single_grouped_weight_observed"] is not True or grouped["fc2_single_grouped_weight_observed"] is not True:
            raise common.ProtocolError("TE single grouped mode was not actually active")
        rows.append(
            {
                "case": dict(case),
                "callback_count": 2 * int(case["local_experts"]),
                "events_sha256": numbered["events_sha256"],
                "member_hashes_sha256": common.sha256_bytes(
                    common.canonical_json_bytes(numbered["member_hashes"])
                ),
                "final_rng_offset": numbered["final_rng_offset"],
                "final_rng_state_sha256": numbered["final_rng_state_sha256"],
                "numbered_equals_copy_packed": True,
            }
        )
    return rows


def _philox_parity(access: kernels.PhiloxRandomAccess, torch, device) -> dict[str, Any]:
    cp = access.cp
    shapes = tuple(
        shape
        for etp in common.ETP_SIZES
        for shape in (
            (2 * (common.ROWS // etp), common.COLUMNS),
            (common.COLUMNS, common.ROWS // etp),
        )
    )
    offsets = (0, 4, 8192, 1_048_576, 4_294_967_300)
    descriptor_rows = []
    for shape_index, shape in enumerate(shapes):
        numel = math.prod(shape)
        stride = 256 * min(
            (numel + 255) // 256,
            access.sm_count * (access.max_threads_per_sm // 256),
        )
        native = sorted(
            {
                0,
                1,
                min(numel - 1, stride - 1),
                min(numel - 1, stride),
                min(numel - 1, 4 * stride),
                numel - 1,
            }
        )
        for offset in offsets:
            seed = 12_345 + shape_index
            _, probe_scaled, probe_bf16 = access.descriptor_probe(
                [seed] * len(native),
                [numel] * len(native),
                [offset] * len(native),
                native,
            )
            expected_increment = kernels.policy_increment(
                numel, access.sm_count, access.max_threads_per_sm
            )
            tensor_f32, after = _torch_generate(torch, device, seed, offset, shape, torch.float32)
            if after - offset != expected_increment:
                raise common.ProtocolError("float32 generator increment parity failed")
            indices = torch.as_tensor(native, dtype=torch.int64, device=device)
            expected = tensor_f32.reshape(-1).index_select(0, indices).cpu().numpy()
            if not np.array_equal(cp.asnumpy(probe_scaled), expected):
                raise common.ProtocolError("exact float32 normal transform parity failed")
            direct_bf16, bf16_after = _torch_generate(
                torch, device, seed, offset, shape, torch.bfloat16
            )
            if bf16_after - offset != expected_increment:
                raise common.ProtocolError("direct BF16 generator increment parity failed")
            direct = direct_bf16.reshape(-1).index_select(0, indices).float().cpu().numpy()
            cast = tensor_f32.to(torch.bfloat16).reshape(-1).index_select(0, indices).float().cpu().numpy()
            if not np.array_equal(cp.asnumpy(probe_bf16), direct) or not np.array_equal(direct, cast):
                raise common.ProtocolError("BF16/direct/cast parity failed")
            descriptor_rows.append(
                {
                    "shape": list(shape),
                    "offset": offset,
                    "coordinate_count": len(native),
                    "increment": expected_increment,
                    "float32_sha256": _float_sha(expected),
                    "bf16_widened_sha256": _float_sha(direct),
                }
            )
            del tensor_f32, direct_bf16

    candidates = []
    for pp_index in (0, 3, 4, 5, 6, 7, 8, 9):
        for ep_index, assignment in ((0, 0), (3, 0), (3, 1), (7, 0), (7, 1)):
            for etp_index in range(4):
                for half in range(2):
                    ordinal = common.logical_ordinal(
                        3407, pp_index, ep_index, etp_index, assignment, half, 0
                    )
                    candidates.append(common.decode_ordinal(ordinal))
    coordinates = (
        0,
        1,
        2047,
        2048,
        393_215,
        393_216,
        786_431,
        786_432,
        common.WEIGHTS_PER_MATRIX - 1,
    )
    candidate_rows = []
    for candidate in candidates:
        for expert in (0, 57, 127):
            for role in ("up", "down"):
                descriptors = [
                    kernels.coordinate_descriptor(
                        candidate,
                        expert,
                        role,
                        coordinate,
                        access.sm_count,
                        access.max_threads_per_sm,
                    )
                    for coordinate in coordinates
                ]
                _, scaled, bf16 = access.descriptor_probe(
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
                    tensor, after = _torch_generate(
                        torch, device, seed, offset, (numel,), torch.float32
                    )
                    increment = kernels.policy_increment(
                        numel, access.sm_count, access.max_threads_per_sm
                    )
                    if after - offset != increment:
                        raise common.ProtocolError("candidate target increment parity failed")
                    indices = torch.as_tensor(
                        [native for _, native in entries], dtype=torch.int64, device=device
                    )
                    values = tensor.index_select(0, indices)
                    values_np = values.cpu().numpy()
                    bf16_np = values.to(torch.bfloat16).float().cpu().numpy()
                    for local, (destination, _) in enumerate(entries):
                        expected[destination] = values_np[local]
                        expected_bf16[destination] = bf16_np[local]
                    del tensor, values
                if not np.array_equal(cp.asnumpy(scaled), expected):
                    raise common.ProtocolError(f"grouped mapping float parity failed: {candidate.id}")
                if not np.array_equal(cp.asnumpy(bf16), expected_bf16):
                    raise common.ProtocolError(f"grouped mapping BF16 parity failed: {candidate.id}")
                generated = access.generate(
                    np.asarray([candidate.ordinal], dtype=np.uint64),
                    np.asarray([expert] * len(coordinates), dtype=np.int32),
                    np.asarray([0 if role == "up" else 1] * len(coordinates), dtype=np.int32),
                    np.asarray(coordinates, dtype=np.uint64),
                )
                if not np.array_equal(cp.asnumpy(generated[0]) * np.float32(0.02), expected):
                    raise common.ProtocolError(f"candidate anchor-kernel parity failed: {candidate.id}")
                candidate_rows.append(
                    {
                        "candidate": candidate.id,
                        "expert": expert,
                        "role": role,
                        "coordinate_count": len(coordinates),
                        "scaled_sha256": _float_sha(expected),
                    }
                )

    torch_probe = torch.arange(257, dtype=torch.float32, device=device)
    cupy_probe = cp.from_dlpack(torch_probe)
    if not np.array_equal(torch_probe.cpu().numpy(), cp.asnumpy(cupy_probe)):
        raise common.ProtocolError("PyTorch/CuPy same-device DLPack parity failed")
    return {
        "descriptor_checks": descriptor_rows,
        "candidate_coordinate_checks": candidate_rows,
        "dlpack_sha256_f32le": _float_sha(cp.asnumpy(cupy_probe)),
    }


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _content_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise common.ProtocolError(f"runtime parity malformed hash: {label}")
    if value in {_EMPTY_SHA256, "0" * 64}:
        raise common.ProtocolError(f"runtime parity empty/sentinel hash: {label}")
    return value


def _rehash_runtime_source(row: Mapping[str, Any], label: str, expected: str) -> None:
    common.strict_keys(row, {"path", "sha256"}, f"runtime source row {label}")
    if row.get("sha256") != expected or not isinstance(row.get("path"), str) or not row["path"]:
        raise common.ProtocolError(f"runtime parity source-file binding mismatch: {label}")
    original = Path(row["path"])
    common.reject_parent_traversal(original, f"runtime parity source path {label}")
    unresolved = common.reject_symlink_components_before_normalization(
        original, f"runtime parity source path {label}", require_exists=True
    )
    info = common.lstat_or_none(unresolved)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise common.ProtocolError(
            f"runtime parity original source path is missing/non-regular/symlink: {label}"
        )
    path = common.require_regular_file_before_resolve(
        unresolved, f"runtime parity original source path {label}"
    )
    if common.sha256_file(path) != expected:
        raise common.ProtocolError(f"runtime parity source rehash mismatch: {label}")


def _expected_descriptor_rows(sm_count: int, max_threads: int) -> list[dict[str, Any]]:
    shapes = tuple(shape for etp in common.ETP_SIZES for shape in (
        (2 * (common.ROWS // etp), common.COLUMNS),
        (common.COLUMNS, common.ROWS // etp),
    ))
    offsets = (0, 4, 8192, 1_048_576, 4_294_967_300)
    result: list[dict[str, Any]] = []
    for shape in shapes:
        numel = math.prod(shape)
        stride = 256 * min((numel + 255) // 256, sm_count * (max_threads // 256))
        native = sorted({0, 1, min(numel - 1, stride - 1), min(numel - 1, stride),
                         min(numel - 1, 4 * stride), numel - 1})
        increment = kernels.policy_increment(numel, sm_count, max_threads)
        for offset in offsets:
            result.append({"shape": list(shape), "offset": offset,
                           "coordinate_count": len(native), "increment": increment})
    return result


def _expected_candidate_rows() -> list[tuple[str, int, str]]:
    candidates = []
    for pp_index in (0, 3, 4, 5, 6, 7, 8, 9):
        for ep_index, assignment in ((0, 0), (3, 0), (3, 1), (7, 0), (7, 1)):
            for etp_index in range(4):
                for half in range(2):
                    ordinal = common.logical_ordinal(3407, pp_index, ep_index, etp_index, assignment, half, 0)
                    candidates.append(common.decode_ordinal(ordinal))
    return [(candidate.id, expert, role) for candidate in candidates
            for expert in (0, 57, 127) for role in ("up", "down")]


def validate_runtime_parity_receipt(
    value: Mapping[str, Any], source_trace_path: Path
) -> dict[str, Any]:
    """Fail closed on every row and rehash all runtime-bound source files."""
    required = {
        "schema", "all_required_checks_passed", "source_trace_sha256",
        "source_trace_internal_sha256", "transformer_engine_version",
        "transformer_engine_revision", "mcore_revision", "runtime_source_files",
        "single_param_environment", "te_storage_parity", "torch_version",
        "cupy_version", "device_name", "device_index", "multi_processor_count",
        "max_threads_per_multi_processor", "descriptor_checks",
        "candidate_coordinate_checks", "dlpack_sha256_f32le",
        "qwen_manifest_directory_or_payload_accessed", "version_acceptance",
    }
    common.strict_keys(value, required, "runtime parity receipt")
    trace = load_source_trace(source_trace_path)
    trace_path = common.require_regular_file_before_resolve(
        source_trace_path, "runtime parity source-trace receipt"
    )
    expected_scalars = {
        "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_runtime_parity_v1",
        "all_required_checks_passed": True,
        "source_trace_sha256": common.sha256_file(trace_path),
        "source_trace_internal_sha256": trace["receipt_sha256"],
        "transformer_engine_revision": common.TE_REVISION,
        "mcore_revision": common.MCORE_REVISION,
        "qwen_manifest_directory_or_payload_accessed": False,
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            raise common.ProtocolError(f"runtime parity receipt mismatch: {key}")
    version = value.get("transformer_engine_version")
    if version not in {common.TE_SOURCE_VERSION, common.TE_PYPI_VERSION}:
        raise common.ProtocolError("runtime parity Transformer Engine version mismatch")
    expected_acceptance = {
        "observed": version,
        "pypi_2_18_0_requires_exact_seven_file_source_rehash": True,
        "all_seven_source_hashes_matched": True,
    }
    if value.get("version_acceptance") != expected_acceptance:
        raise common.ProtocolError("runtime parity version-acceptance policy mismatch")
    if value.get("single_param_environment") != {"name": SINGLE_PARAM_ENV, "value": "1"}:
        raise common.ProtocolError("runtime parity single-parameter environment mismatch")
    for key in ("device_index", "multi_processor_count", "max_threads_per_multi_processor"):
        if not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < (0 if key == "device_index" else 1):
            raise common.ProtocolError(f"runtime parity invalid device field: {key}")
    for key in ("torch_version", "cupy_version", "device_name"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise common.ProtocolError(f"runtime parity invalid runtime field: {key}")
    sources = value.get("runtime_source_files")
    if not isinstance(sources, Mapping) or set(sources) != set(EXPECTED_RUNTIME_SOURCE_HASHES):
        raise common.ProtocolError("runtime parity source-file member mismatch")
    for label, expected in EXPECTED_RUNTIME_SOURCE_HASHES.items():
        row = sources[label]
        if not isinstance(row, Mapping):
            raise common.ProtocolError(f"runtime parity source-file row malformed: {label}")
        _rehash_runtime_source(row, label, expected)
    storage = value.get("te_storage_parity")
    expected_cases = [dict(row) for row in RUNTIME_TRACE_CASES]
    if (not isinstance(storage, list) or len(storage) != len(expected_cases)
            or any(not isinstance(row, Mapping) for row in storage)
            or [row.get("case") for row in storage] != expected_cases):
        raise common.ProtocolError("runtime parity TE trace-case mismatch")
    for row, case in zip(storage, RUNTIME_TRACE_CASES):
        if not isinstance(row, Mapping):
            raise common.ProtocolError("runtime parity storage row malformed")
        common.strict_keys(row, {"case", "callback_count", "events_sha256", "member_hashes_sha256",
                                 "final_rng_offset", "final_rng_state_sha256",
                                 "numbered_equals_copy_packed"}, "runtime storage row")
        if not isinstance(row["case"], Mapping) or row["case"] != case:
            raise common.ProtocolError("runtime parity storage case content mismatch")
        if row.get("callback_count") != 2 * int(case["local_experts"]):
            raise common.ProtocolError("runtime parity callback-count mismatch")
        if row.get("numbered_equals_copy_packed") is not True:
            raise common.ProtocolError("runtime parity copy-pack proof missing")
        for key in ("events_sha256", "member_hashes_sha256", "final_rng_state_sha256"):
            _content_hash(row.get(key), f"storage.{key}")
        if not isinstance(row.get("final_rng_offset"), int) or row["final_rng_offset"] <= 0:
            raise common.ProtocolError("runtime parity final RNG offset invalid")
    descriptors = value.get("descriptor_checks")
    if not isinstance(descriptors, list) or len(descriptors) != 40:
        raise common.ProtocolError("runtime parity descriptor-check count mismatch")
    expected_descriptors = _expected_descriptor_rows(value["multi_processor_count"], value["max_threads_per_multi_processor"])
    for index, (row, expected) in enumerate(zip(descriptors, expected_descriptors)):
        if not isinstance(row, Mapping):
            raise common.ProtocolError(f"runtime descriptor row {index} malformed")
        common.strict_keys(row, {"shape", "offset", "coordinate_count", "increment",
                                 "float32_sha256", "bf16_widened_sha256"},
                           f"runtime descriptor row {index}")
        for key, expected_value in expected.items():
            if row.get(key) != expected_value:
                raise common.ProtocolError(f"runtime descriptor row {index} mismatch: {key}")
        _content_hash(row.get("float32_sha256"), f"descriptor[{index}].float32")
        _content_hash(row.get("bf16_widened_sha256"), f"descriptor[{index}].bf16")
    candidates = value.get("candidate_coordinate_checks")
    if not isinstance(candidates, list) or len(candidates) != 1_920:
        raise common.ProtocolError("runtime parity candidate-coordinate-check count mismatch")
    for index, (row, expected) in enumerate(zip(candidates, _expected_candidate_rows())):
        if not isinstance(row, Mapping):
            raise common.ProtocolError(f"runtime candidate row {index} malformed")
        common.strict_keys(row, {"candidate", "expert", "role", "coordinate_count", "scaled_sha256"},
                           f"runtime candidate row {index}")
        candidate_id, expert, role = expected
        if (row.get("candidate"), row.get("expert"), row.get("role"), row.get("coordinate_count")) != (candidate_id, expert, role, 9):
            raise common.ProtocolError(f"runtime candidate row {index} content mismatch")
        _content_hash(row.get("scaled_sha256"), f"candidate[{index}].scaled")
    _content_hash(value.get("dlpack_sha256_f32le"), "dlpack")
    return dict(value)


def run_parity(
    access: kernels.PhiloxRandomAccess,
    source_trace_path: Path,
) -> dict[str, Any]:
    """Run all exact-version TE and Philox checks before any payload contact."""
    if os.environ.get(SINGLE_PARAM_ENV) != "1":
        raise common.ProtocolError(f"set {SINGLE_PARAM_ENV}=1 before importing/running parity")
    source_trace = load_source_trace(source_trace_path)
    try:
        import torch
        import transformer_engine as te
    except Exception as error:  # pragma: no cover - authorized GPU runtime only
        raise common.ProtocolError(f"mandatory TE/PyTorch import failed: {error}") from error
    if not torch.cuda.is_available():
        raise common.ProtocolError("PyTorch CUDA unavailable for mandatory parity")
    device = torch.device(f"cuda:{access.device_index}")
    torch.cuda.set_device(device)
    runtime_sources = _runtime_source_hashes()
    version = _te_version(te, runtime_sources)
    storage_rows = _te_storage_parity(torch, te, device)
    philox = _philox_parity(access, torch, device)
    torch.cuda.synchronize(device)
    access.cp.cuda.runtime.deviceSynchronize()
    result = {
        "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_runtime_parity_v1",
        "all_required_checks_passed": True,
        "source_trace_sha256": common.sha256_file(
            common.require_regular_file_before_resolve(
                source_trace_path, "runtime parity source-trace receipt"
            )
        ),
        "source_trace_internal_sha256": source_trace["receipt_sha256"],
        "transformer_engine_version": version,
        "version_acceptance": {
            "observed": version,
            "pypi_2_18_0_requires_exact_seven_file_source_rehash": True,
            "all_seven_source_hashes_matched": True,
        },
        "transformer_engine_revision": common.TE_REVISION,
        "mcore_revision": common.MCORE_REVISION,
        "runtime_source_files": runtime_sources,
        "single_param_environment": {"name": SINGLE_PARAM_ENV, "value": os.environ[SINGLE_PARAM_ENV]},
        "te_storage_parity": storage_rows,
        "torch_version": str(torch.__version__),
        "cupy_version": str(access.cp.__version__),
        "device_name": access.device_name,
        "device_index": access.device_index,
        "multi_processor_count": access.sm_count,
        "max_threads_per_multi_processor": access.max_threads_per_sm,
        **philox,
        "qwen_manifest_directory_or_payload_accessed": False,
    }
    validate_runtime_parity_receipt(result, source_trace_path)
    return result
