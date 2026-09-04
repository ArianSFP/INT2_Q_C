"""Authenticated, bounded fixed-label TETRAPATH census on the local RTX 3060."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SITE = WORKSPACE / ".venv-cupy/Lib/site-packages"
_DLL_HANDLES = []
for _directory in [WORKSPACE / ".tools/cuda_dlls_3060",
                   *sorted((SITE / "nvidia").glob("*/bin"))]:
    if not _directory.is_dir():
        raise RuntimeError(f"missing process-local CUDA DLL directory: {_directory}")
    _DLL_HANDLES.append(os.add_dll_directory(str(_directory)))
os.environ["CUDA_PATH"] = str(SITE / "nvidia/cuda_runtime")
os.environ["CUPY_CACHE_DIR"] = str(WORKSPACE / "tmp/tetrapath4_fixed_qwen_cupy_cache_v0")

import cupy as cp


EXPECTED_UUID = "GPU-458a424a-76e3-65e5-0470-803e0ed131ca"
EXPECTED_TETRA_SHA = "b303d9d87659d0ae36687fed9ab82b00e1eea8a6bd94ea4769453e42b5fb611a"
EXPECTED_PAIR_BACKEND_SHA = "e16e657604be8f5ddd2858c6b8c49a8d548072afdbcef866e3895d366a45251c"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bf16_to_fp64(path: Path, raw_shape: tuple[int, int], transpose: bool) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    require(raw.size == math.prod(raw_shape), f"BF16 element count: {path}")
    value = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(raw_shape)
    if transpose:
        value = value.T
    return np.ascontiguousarray(value, dtype=np.float64).reshape(-1)


def canonical_labels_gpu(values: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xg = cp.asarray(values)
    lg = cp.asarray(levels)
    labels = cp.argmin((xg[..., None] - lg) ** 2, axis=3).astype(cp.uint8)
    tetra = cp.stack((labels[0, 0], labels[0, 1], labels[1, 0], labels[1, 1]), axis=1)
    ids = (((tetra[:, 0].astype(cp.uint16) * 4 + tetra[:, 1]) * 4 + tetra[:, 2]) * 4 +
           tetra[:, 3])
    counts = cp.bincount(ids, minlength=256)
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(tetra), cp.asnumpy(counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source_root = args.source_root.resolve()

    tetra_path = repo / "research/tetrapath4_source_oracle_v0_20260904/tetrapath4_oracle.py"
    pair_path = repo / "research/pairpath_p2_local3060_cupy_preflight_v0/pairpath_cupy_backend.py"
    panel_path = repo / "research/same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3/panel_lock.json"
    require(sha256(tetra_path) == EXPECTED_TETRA_SHA, "TETRAPATH source hash")
    require(sha256(pair_path) == EXPECTED_PAIR_BACKEND_SHA, "PAIRPATH backend hash")
    tetra = load_module("tetrapath4_probe_core", tetra_path)
    pair = load_module("pairpath_probe_backend", pair_path)

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    uuid_hex = "".join(f"{int(x):02x}" for x in props["uuid"])
    require(name == "NVIDIA GeForce RTX 3060", "local RTX 3060 name")
    require(uuid_hex.endswith("560701"), "local RTX 3060 UUID bytes")

    panel_bytes = panel_path.read_bytes()
    panel = json.loads(panel_bytes)
    require(panel["model"] == "Qwen/Qwen3-30B-A3B" and panel["layer"] == 15,
            "pinned Qwen panel")
    by_expert: dict[int, dict[str, dict]] = {}
    authenticated = []
    for entry in panel["files"]:
        path = source_root / entry["relative_path"]
        require(path.is_file() and path.stat().st_size == entry["bytes"], f"file binding: {path}")
        actual = sha256(path)
        require(actual == entry["sha256"], f"SHA-256 binding: {path}")
        by_expert.setdefault(int(entry["expert"]), {})[entry["role"]] = entry
        authenticated.append({"expert": entry["expert"], "role": entry["role"],
                              "bytes": entry["bytes"], "sha256": actual})

    experts = list(map(int, panel["experts"]))
    require(len(experts) % 2 == 0, "even panel")
    expert_pairs = list(zip(experts[::2], experts[1::2]))
    pair_results = []
    started = time.perf_counter()
    for pair_index, (expert_e, expert_f) in enumerate(expert_pairs):
        rows = []
        for expert in (expert_e, expert_f):
            roles = []
            for role in ("up", "down"):
                entry = by_expert[expert][role]
                roles.append(bf16_to_fp64(source_root / entry["relative_path"],
                                          tuple(entry["raw_shape"]),
                                          bool(entry["down_transposed"])))
            rows.append(roles)
        values = np.ascontiguousarray(np.asarray(rows, dtype=np.float64))
        scales, levels = pair.prepare_levels(values)
        labels, gpu_counts = canonical_labels_gpu(values, levels)
        cpu_counts = np.bincount(tetra.tuple_ids(labels), minlength=256)
        require(np.array_equal(gpu_counts, cpu_counts), "GPU/CPU 256-tuple count parity")
        source_census = tetra.fixed_assignment_census(labels)

        control_labels = labels.copy()
        rng = np.random.Generator(np.random.PCG64(0x5445545241000000 + pair_index))
        for variable in range(4):
            control_labels[:, variable] = control_labels[rng.permutation(labels.shape[0]), variable]
        control_census = tetra.fixed_assignment_census(control_labels)
        source_gain = float(source_census["fourway_gain_over_best_factorized_bpw"])
        control_gain = float(control_census["fourway_gain_over_best_factorized_bpw"])
        connected = float(source_census["residual_connected_information_bpw"])
        connected_control = float(control_census["residual_connected_information_bpw"])
        pair_results.append({
            "experts": [expert_e, expert_f],
            "coordinates": int(labels.shape[0]),
            "scale_sha256": hashlib.sha256(scales.astype("<u2", copy=False).tobytes()).hexdigest(),
            "source": source_census,
            "independent_permutation_control": control_census,
            "control_corrected_fourway_gain_bpw": source_gain - control_gain,
            "control_corrected_connected_information_bpw": connected - connected_control,
        })

    corrected = np.asarray([x["control_corrected_fourway_gain_bpw"] for x in pair_results])
    connected = np.asarray([x["control_corrected_connected_information_bpw"] for x in pair_results])
    maximum_source_gain = max(float(x["source"]["fourway_gain_over_best_factorized_bpw"])
                              for x in pair_results)
    report = {
        "schema": "tetrapath4.fixed_label_qwen_local3060_probe.v0",
        "status": ("FIXED_LABEL_MEMORYLESS_FOURWAY_HARD_KILL_BELOW_0P045_BPW" if
                   maximum_source_gain < 0.045 else
                   "FIXED_LABEL_SIGNAL_ONLY_REQUIRES_LABEL_FLEXIBLE_ORACLE"),
        "scientific_boundary": (
            "Nearest four-level labels only. This does not kill label-flexible TETRAPATH, "
            "the actual six-plane STRATA code, multiscale structure, or a finite codec."),
        "threshold_bpw": 0.045,
        "panel": {"model": panel["model"], "revision": panel["revision"],
                  "layer": panel["layer"], "panel_lock_sha256": hashlib.sha256(panel_bytes).hexdigest(),
                  "authenticated_files": authenticated},
        "runtime": {"hostname": platform.node(), "python": list(sys.version_info[:3]),
                    "numpy": np.__version__, "cupy": cp.__version__, "device_name": name,
                    "device_uuid": EXPECTED_UUID, "device_uuid_hex": uuid_hex,
                    "elapsed_seconds": time.perf_counter() - started},
        "pairing": [list(x) for x in expert_pairs],
        "aggregate": {
            "maximum_source_fourway_gain_bpw": maximum_source_gain,
            "mean_control_corrected_fourway_gain_bpw": float(corrected.mean()),
            "maximum_control_corrected_fourway_gain_bpw": float(corrected.max()),
            "mean_control_corrected_connected_information_bpw": float(connected.mean()),
            "maximum_control_corrected_connected_information_bpw": float(connected.max()),
        },
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"],
                      "output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
