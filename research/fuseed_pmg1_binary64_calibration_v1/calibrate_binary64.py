#!/usr/bin/env python3
"""Design-conformant binary64 stage-0 calibration for FUSEED-PMG1-v2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time


EXPECTED_BASE_CALIBRATION_SHA256 = "0432aaf16d809587157644db1335dc0a3d05b9d2d05a9ba1fdaebec655ca5c96"
EXPECTED_DESIGN_MANIFEST_SHA256 = "ea7086c401cd6981d097ecc9b52196d3d01cda123d0cb8ab28c001cf008b27ff"
EXPECTED_DESIGN_BUNDLE_SHA256 = "16aacb6f5fa6a1ed12fe0c01506410ad69585894077a4a6af627674b6b90adda"
EXPECTED_DESIGN_COMPLETE_PLAN_SHA256 = "86639758eda1835b9ea9e883372bb55ec13ec3487705a91d892878972db74760"
COMPILE_OPTIONS = (
    "--std=c++17",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-I/usr/local/cuda/include",
)
COMPILE_ARCH = "120"
SHARD_SIZE = 1 << 24
SHARDS = 256
TOP_K = 8192
REPETITIONS = 3
STAGE0_MARGIN_GATE_SECONDS = 650.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, expected: str, name: str):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_exact(cp, source: str, filename: str):
    from cupy.cuda import compiler

    blob, mapping = compiler.compile_using_nvrtc(
        source, options=COMPILE_OPTIONS, arch=COMPILE_ARCH, filename=filename
    )
    if not isinstance(blob, bytes) or not blob.startswith(b"\x7fELF") or mapping is not None:
        raise RuntimeError("NVRTC did not emit the expected anonymous ELF cubin")
    module = cp.cuda.function.Module()
    module.load(blob)
    return module, {
        "filename": filename,
        "options": list(COMPILE_OPTIONS),
        "arch": COMPILE_ARCH,
        "cubin_bytes": len(blob),
        "cubin_sha256": hashlib.sha256(blob).hexdigest(),
        "cubin_magic_hex": blob[:8].hex(),
    }


def derive_binary64_capture_source(source: str) -> tuple[str, dict[str, int]]:
    counts = {}
    counts["metric_function_signature"] = source.count(
        "__device__ __forceinline__ float domain_q("
    )
    if counts["metric_function_signature"] != 1:
        raise RuntimeError("unexpected metric signature cardinality")
    source = source.replace(
        "__device__ __forceinline__ float domain_q(",
        "__device__ __forceinline__ double domain_capture(",
        1,
    )
    counts["metric_return"] = source.count("return (float)(sse / baseline);")
    if counts["metric_return"] != 1:
        raise RuntimeError("unexpected metric return cardinality")
    source = source.replace(
        "return (float)(sse / baseline);",
        "const double capture = 1.0 - (sse / baseline);\n"
        "  return capture == 0.0 ? 0.0 : capture;",
        1,
    )
    counts["metric_call_sites_plus_definition"] = source.count("domain_q(")
    if counts["metric_call_sites_plus_definition"] != 2:
        raise RuntimeError("unexpected metric call cardinality")
    source = source.replace("domain_q(", "domain_capture(")
    counts["output_pointer"] = source.count("float* q)")
    if counts["output_pointer"] != 1:
        raise RuntimeError("unexpected q pointer cardinality")
    source = source.replace("float* q)", "double* q)", 1)
    if "domain_q(" in source or "float* q)" in source:
        raise RuntimeError("binary32 metric path survived derivation")
    return source, counts


def design_bundle_digest(plan_result) -> str:
    counters: dict[tuple[int, str, str], int] = {}
    lines = []
    for index, bundle in enumerate(plan_result["wire"]):
        key = (bundle["expert"], bundle["role"], bundle["split"])
        accepted = counters.get(key, 0)
        counters[key] = accepted + 1
        category = f"{bundle['role']}_{bundle['split']}"
        scale = "3c03126f" if bundle["role"] == "up" else "3a560a28"
        line = (
            f"bundle|index={index:03d}|category={category}|e={bundle['expert']:03d}|"
            f"role={bundle['role']}|split={bundle['split']}|accepted={accepted:02d}|"
            f"seed_delta={bundle['seed_addend']}|offset={bundle['offset_values']}|"
            f"scale={scale}|seq={bundle['sequence']}|j={bundle['normal4_index']}"
        )
        for coordinate in bundle["coordinates"]:
            line += (
                f"|lane{coordinate['lane']}=r{coordinate['row']:03d},"
                f"c{coordinate['column']:04d},native{coordinate['native']}"
            )
        lines.append(line)
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    if digest != EXPECTED_DESIGN_BUNDLE_SHA256:
        raise RuntimeError(f"design bundle equivalence mismatch: {digest}")
    return digest


def canonical_topk_descending(cp, np, row, top_k: int):
    cutoff_index = int(row.size) - top_k
    provisional = cp.argpartition(row, cutoff_index)[cutoff_index:]
    threshold = float(cp.asnumpy(cp.min(row[provisional])))
    better = cp.asnumpy(cp.nonzero(row > threshold)[0]).astype(np.uint64)
    equal = cp.asnumpy(cp.nonzero(row == threshold)[0]).astype(np.uint64)
    need = top_k - better.size
    if need < 0 or equal.size < need:
        raise RuntimeError("binary64 Top-K threshold cardinality failed")
    equal.sort()
    seeds = np.concatenate((better, equal[:need]))
    values = cp.asnumpy(row[cp.asarray(seeds)]).astype(np.float64, copy=False)
    order = np.lexsort((seeds, -values))
    seeds = seeds[order]
    values = values[order]
    if seeds.size != top_k or not np.isfinite(values).all():
        raise RuntimeError("binary64 canonical Top-K invariant failed")
    return seeds, values, threshold, int(equal.size)


def write_journal64(path: Path, header: dict, seeds, captures) -> tuple[str, int, float]:
    started = time.perf_counter()
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    record_bytes = b"".join(
        struct.pack("<Id", int(seed), float(capture))
        for seed, capture in zip(seeds, captures, strict=True)
    )
    if len(record_bytes) != TOP_K * 12:
        raise RuntimeError("packed u32+binary64 journal record size mismatch")
    payload = struct.pack("<I", len(header_bytes)) + header_bytes + record_bytes
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest(), len(payload), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=SHARD_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if (args.candidates, args.top_k, args.repetitions) != (
        SHARD_SIZE, TOP_K, REPETITIONS
    ):
        raise RuntimeError("shape must be exactly 2^24/8192/3")

    root = Path(__file__).resolve().parent
    research = root.parent
    base_path = research / "fuseed_pmg1_direct_source_calibration_v0" / "calibrate.py"
    base = load_module(base_path, EXPECTED_BASE_CALIBRATION_SHA256, "fuseed_pmg1_base")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if os.environ.get("PYTHONPATH") != base.EXPECTED_PYTHONPATH:
        raise RuntimeError("PYTHONPATH binding mismatch")
    if Path(sys.executable).resolve() != Path("/usr/bin/python3.12"):
        raise RuntimeError("Python executable binding mismatch")
    runtime_files = base.bind_runtime_files()
    if base.EXPECTED_PYTHONPATH in sys.path:
        raise RuntimeError("isolated launch preloaded the system package root")
    sys.path.append(base.EXPECTED_PYTHONPATH)

    import cupy as cp
    import numpy as np
    import torch
    from cupy.cuda import nvrtc

    if (np.__version__, cp.__version__, torch.__version__) != (
        "2.5.2", "14.2.0", "2.8.0+cu128"
    ):
        raise RuntimeError("package version mismatch")
    if (
        torch.version.cuda != "12.8"
        or cp.cuda.runtime.runtimeGetVersion() != 12090
        or cp.cuda.runtime.driverGetVersion() != 13000
        or nvrtc.getVersion() != (12, 8)
    ):
        raise RuntimeError("CUDA/NVRTC version mismatch")

    design_manifest = research / "fuseed_pmg1_v2_design_draft" / "ARTIFACT_SHA256SUMS.txt"
    if sha256_file(design_manifest) != EXPECTED_DESIGN_MANIFEST_SHA256:
        raise RuntimeError("PMG1 design manifest binding mismatch")
    plan_path = research / "fuseed_pmg1_direct_source_calibration_v0" / "plan.py"
    direct_path = research / "fuseed_u32_direct_counter_calibration_v0" / "calibrate_direct.py"
    domain_path = research / "fuseed_u32_direct_domain_collapse_probe_v0" / "probe.py"
    plan = base.load_module(plan_path, base.EXPECTED_PLAN_SHA256, "fuseed_pmg1_plan64")
    direct = base.load_module(direct_path, base.EXPECTED_DIRECT_SHA256, "fuseed_direct64")
    domain = base.load_module(domain_path, base.EXPECTED_DOMAIN_PROBE_SHA256, "fuseed_domain64")
    plan_result = plan.reconstruct_plan()
    bundle_digest = design_bundle_digest(plan_result)
    shape_path, shape = direct.load_shape_module()
    header_hashes = direct.bind_headers()
    direct_source, direct_counts = direct.derive_direct_source(shape.CUDA_SOURCE)
    active_source, active_counts = domain.derive_active_domain_source(direct_source, 1)
    capture_source, capture_counts = derive_binary64_capture_source(active_source)
    if capture_source.count("curand_init(") or capture_source.count("curand_normal4("):
        raise RuntimeError("stateful generator survived performance derivation")

    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    if device_name != "NVIDIA GeForce RTX 5090" or cp.cuda.Device().compute_capability != "120":
        raise RuntimeError("GPU identity mismatch")
    performance_module, performance_compile = compile_exact(
        cp, capture_source, "fuseed_pmg1_binary64_stage0.cu"
    )
    parity_module, parity_compile = compile_exact(
        cp, base.PARITY_SOURCE, "fuseed_pmg1_binary64_parity.cu"
    )
    performance = performance_module.get_function("fuseed_33domain_scores")
    direct_replays = base.run_three_direct_replays(direct)
    sequential_replays = base.run_sequential_replays(cp, np, direct, parity_module)
    torch_replays = base.run_torch_replays(cp, np, torch, parity_module, props)

    loaded_libraries = base.loaded_cuda_libraries()
    expected_loaded = sorted(
        path for path in base.EXPECTED_RUNTIME_FILES
        if Path(path).name.startswith(("libcuda.so", "libcudart.so", "libnvrtc.so"))
    )
    if loaded_libraries != expected_loaded:
        raise RuntimeError("loaded CUDA library closure mismatch")

    targets_host, stats_host = shape.make_targets()
    targets = cp.asarray(targets_host)
    target_stats = cp.asarray(stats_host)
    addends, sequences, offset_quads, normal4_indices = [
        cp.asarray(value) for value in base.plan_arrays(np, plan_result)
    ]
    q = cp.empty(args.candidates, dtype=cp.float64)
    block, warps = 256, 8
    grid = min(65535, (args.candidates + warps - 1) // warps)

    def launch(count: int) -> None:
        performance(
            (min(grid, (count + warps - 1) // warps),), (block,),
            (
                np.uint64(0), np.uint64(count), addends, sequences,
                offset_quads, normal4_indices, targets, target_stats, q,
            ),
        )

    launch(1 << 14)
    base.synchronize(cp)
    args.output.parent.mkdir(parents=False, exist_ok=False)
    rows = []
    retained_seeds = retained_captures = None
    sample_indices = cp.linspace(0, args.candidates - 1, 32768, dtype=cp.int64)
    for repetition in range(args.repetitions):
        q.fill(cp.nan)
        base.synchronize(cp)
        started = time.perf_counter()
        launch(args.candidates)
        base.synchronize(cp)
        kernel_seconds = time.perf_counter() - started

        started = time.perf_counter()
        finite = bool(cp.asnumpy(cp.isfinite(q).all()))
        negative_zero_count = int(cp.asnumpy(cp.count_nonzero(cp.signbit(q) & (q == 0.0))))
        base.synchronize(cp)
        finite_seconds = time.perf_counter() - started
        if not finite or negative_zero_count != 0:
            raise RuntimeError("binary64 capture finite/canonical-zero invariant failed")

        started = time.perf_counter()
        seeds, captures, threshold, ties = canonical_topk_descending(
            cp, np, q, args.top_k
        )
        base.synchronize(cp)
        topk_seconds = time.perf_counter() - started
        sample = cp.asnumpy(q[sample_indices]).astype("<f8", copy=False)
        seed_hash = hashlib.sha256(seeds.astype("<u4", copy=False).tobytes()).hexdigest()
        capture_hash = hashlib.sha256(captures.astype("<f8", copy=False).tobytes()).hexdigest()
        q_hash = hashlib.sha256(sample.tobytes()).hexdigest()
        header = {
            "schema": "fuseed_pmg1_binary64_shard_journal_v1",
            "repetition": repetition,
            "shard_base_u32": 0,
            "candidate_count": args.candidates,
            "top_k": args.top_k,
            "record_wire": "packed little-endian u32 seed then binary64 capture",
            "metric_order": "capture descending then seed_u32 ascending",
            "performance_cubin_sha256": performance_compile["cubin_sha256"],
            "design_complete_plan_sha256": EXPECTED_DESIGN_COMPLETE_PLAN_SHA256,
            "design_bundle_sha256": bundle_digest,
            "seed_sha256": seed_hash,
            "capture_sha256": capture_hash,
        }
        journal_hash, journal_bytes, journal_seconds = write_journal64(
            args.output.parent / f"binary64_shard_replay_{repetition}.bin",
            header, seeds, captures,
        )
        rows.append({
            "repetition": repetition,
            "kernel_seconds": kernel_seconds,
            "finite_and_zero_validation_seconds": finite_seconds,
            "topk_seconds": topk_seconds,
            "journal_fsync_seconds": journal_seconds,
            "shard_end_to_end_seconds": kernel_seconds + finite_seconds + topk_seconds + journal_seconds,
            "threshold_capture": threshold,
            "boundary_tie_cardinality": ties,
            "negative_zero_count": negative_zero_count,
            "best_seed_u32": int(seeds[0]),
            "best_capture": float(captures[0]),
            "topk_seed_sha256": seed_hash,
            "topk_capture_sha256": capture_hash,
            "q_sentinel_sha256": q_hash,
            "journal_sha256": journal_hash,
            "journal_bytes": journal_bytes,
            "packed_record_bytes": TOP_K * 12,
        })
        retained_seeds, retained_captures = seeds, captures

    for field in ("topk_seed_sha256", "topk_capture_sha256", "q_sentinel_sha256"):
        if len({row[field] for row in rows}) != 1:
            raise RuntimeError(f"binary64 replay differs: {field}")
    if retained_seeds is None or retained_captures is None:
        raise RuntimeError("missing retained binary64 Top-K")

    started = time.perf_counter()
    merge_seeds = np.concatenate(
        [retained_seeds + np.uint64(shard * SHARD_SIZE) for shard in range(SHARDS)]
    )
    merge_captures = np.tile(retained_captures, SHARDS)
    merge_order = np.lexsort((merge_seeds, -merge_captures))[:TOP_K]
    global_seeds = merge_seeds[merge_order].astype("<u4", copy=False)
    global_captures = merge_captures[merge_order].astype("<f8", copy=False)
    global_merge_seconds = time.perf_counter() - started

    shard_median = statistics.median(row["shard_end_to_end_seconds"] for row in rows)
    cold_excess = max(0.0, rows[0]["shard_end_to_end_seconds"] - shard_median)
    stage0_projection = shard_median * SHARDS + cold_excess + global_merge_seconds
    result = {
        "schema": "fuseed_pmg1_binary64_source_free_stage0_calibration_v1",
        "status": (
            "BINARY64_STAGE0_MARGIN_PASS_PENDING_FULL_PIPELINE_AND_INDEPENDENT_AUDIT"
            if stage0_projection < STAGE0_MARGIN_GATE_SECONDS
            else "EARLY_KILL_BINARY64_STAGE0_NO_QWEN"
        ),
        "claim_boundary": (
            "Source-free binary64 stage0 calibration only. This closes the v0 metric/wire "
            "mismatch but does not time stage1/stage2/validation, test Qwen, or authorize payload."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "bindings": {
            "base_calibration": {"path": str(base_path), "sha256": EXPECTED_BASE_CALIBRATION_SHA256},
            "design_manifest": {"path": str(design_manifest), "sha256": EXPECTED_DESIGN_MANIFEST_SHA256},
            "design_complete_plan_sha256": EXPECTED_DESIGN_COMPLETE_PLAN_SHA256,
            "design_bundle_sha256": bundle_digest,
            "plan": {"path": str(plan_path), "sha256": base.EXPECTED_PLAN_SHA256},
            "direct": {"path": str(direct_path), "sha256": base.EXPECTED_DIRECT_SHA256},
            "domain_probe": {"path": str(domain_path), "sha256": base.EXPECTED_DOMAIN_PROBE_SHA256},
            "shape": {"path": str(shape_path), "sha256": direct.EXPECTED_SHAPE_SOURCE_SHA256},
            "cuda_headers": header_hashes,
        },
        "runtime": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cupy": cp.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_api": int(cp.cuda.runtime.driverGetVersion()),
            "nvrtc": list(nvrtc.getVersion()),
            "device": device_name,
            "compute_capability": cp.cuda.Device().compute_capability,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "pythonpath": os.environ["PYTHONPATH"],
            "file_bindings": runtime_files,
            "loaded_cuda_libraries": loaded_libraries,
        },
        "derivation": {
            "direct_replacement_counts": direct_counts,
            "active_domain_replacement_counts": active_counts,
            "binary64_capture_replacement_counts": capture_counts,
            "active_domains": 1,
            "abi_id": "CURRENT_PMG_GATE_UP_DIRECT_BF16",
            "metric_wire": "IEEE-754 binary64 capture, canonical positive zero",
            "metric_order": "capture descending then seed_u32 ascending",
            "performance_cuda_sha256": hashlib.sha256(capture_source.encode()).hexdigest(),
        },
        "compiled_kernels": {
            "performance": performance_compile,
            "parity": parity_compile,
            "parity_source_sha256": hashlib.sha256(base.PARITY_SOURCE.encode()).hexdigest(),
        },
        "parity": {
            "direct_shifted_reference_three_replays": direct_replays,
            "direct_shifted_and_original_offset_sequential_three_replays": sequential_replays,
            "torch_initial_terminal_and_bf16_three_replays": torch_replays,
        },
        "shape": {
            "candidates_per_shard": args.candidates,
            "shards": SHARDS,
            "complete_candidate_count": 1 << 32,
            "abi_count": 1,
            "active_domains": 1,
            "normal4_bundles_per_candidate": 256,
            "normal_values_per_candidate": 1024,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "q_dtype": "binary64",
            "q_bytes": int(q.nbytes),
            "journal_record_bytes": 12,
            "all_shard_topk_record_bytes": SHARDS * TOP_K * 12,
            "block_threads": block,
            "warps_per_block": warps,
            "grid_blocks": grid,
        },
        "rows": rows,
        "global_merge_shape_probe": {
            "input_records": SHARDS * TOP_K,
            "output_records": TOP_K,
            "seconds": global_merge_seconds,
            "seed_sha256": hashlib.sha256(global_seeds.tobytes()).hexdigest(),
            "capture_sha256": hashlib.sha256(global_captures.tobytes()).hexdigest(),
            "uses_synthetic_repeated_shard_metrics": True,
        },
        "aggregate": {
            "median_complete_stage0_shard_seconds": shard_median,
            "one_time_cold_excess_seconds": cold_excess,
            "projected_complete_u32_stage0_seconds_including_finite_topk_journal_and_global_merge": stage0_projection,
            "prospective_stage0_margin_gate_seconds": STAGE0_MARGIN_GATE_SECONDS,
            "stage0_projection_below_margin_gate": stage0_projection < STAGE0_MARGIN_GATE_SECONDS,
            "full_pipeline_wall_gate_seconds": 900.0,
            "reserved_seconds_for_unmeasured_stage1_stage2_validation_and_final_journal": 900.0 - STAGE0_MARGIN_GATE_SECONDS,
            "full_pipeline_projection_claimed": False,
            "replay_deterministic": True,
        },
        "access": {
            "model_or_qwen_path_arguments": 0,
            "payload_files_opened": 0,
            "network_operations": 0,
        },
    }
    with args.output.open("xb") as handle:
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "status": result["status"],
        "stage0_projection_seconds": stage0_projection,
        "performance_cubin_sha256": performance_compile["cubin_sha256"],
        "best_seed": rows[0]["best_seed_u32"],
        "best_capture": rows[0]["best_capture"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
