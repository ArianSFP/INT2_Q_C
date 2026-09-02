#!/usr/bin/env python3
"""Run the pinned fresh CuPy worker and independently validate its search."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_audit import authenticate_source, require


def independent_rank01_reference(source: Path) -> dict:
    sys.path.insert(0, str(source))
    for name in ("codec", "search"):
        sys.modules.pop(name, None)
    codec = importlib.import_module("codec")
    search = importlib.import_module("search")

    geometry = codec.Geometry(704, 2304, 1, 320, 16, 1024, 256)
    coordinate = np.arange(geometry.count, dtype=np.float64)
    signal = (0.71 * np.sin(coordinate * 0.017) +
              0.23 * np.cos(coordinate * 0.071))
    levels = np.linspace(-1.5, 1.5, 64, dtype=np.float64)
    table = (signal[:, None] - levels[None, :]) ** 2
    nearest = np.argmin(table, axis=1).astype(np.uint8)
    rows = np.arange(geometry.count)
    candidates = []
    for requested_rank in (0, 1):
        current = nearest.copy()
        factors = []
        planes = []
        for level in range(6):
            clear = current & np.uint8(63 ^ (1 << level))
            one = clear | np.uint8(1 << level)
            c0 = table[rows, clear.astype(np.int64)]
            c1 = table[rows, one.astype(np.int64)]
            if requested_rank == 0:
                plane = np.zeros(geometry.count, np.uint8)
            else:
                c0m = c0.reshape(geometry.row_count, geometry.col_count)
                c1m = c1.reshape(geometry.row_count, geometry.col_count)
                preferred = (c1m < c0m).astype(np.uint8)
                v = preferred[0].copy()
                u = np.zeros(geometry.row_count, np.uint8)
                for _ in range(4):
                    score0 = c0m.sum(axis=1, dtype=np.float64)
                    score1 = np.where(v[None, :], c1m, c0m).sum(
                        axis=1, dtype=np.float64
                    )
                    u = (score1 < score0).astype(np.uint8)
                    score0_col = c0m.sum(axis=0, dtype=np.float64)
                    score1_col = np.where(u[:, None], c1m, c0m).sum(
                        axis=0, dtype=np.float64
                    )
                    v = (score1_col < score0_col).astype(np.uint8)
                plane = (u[:, None] * v[None, :]).astype(np.uint8).reshape(-1)
            u_factor, v_factor = codec.canonical_gf2_factor(
                plane, geometry.row_count, geometry.col_count
            )
            factors.append((u_factor, v_factor))
            planes.append(plane)
            current = clear | (plane << level)
        model = {"ranks": [u.shape[1] for u, _ in factors],
                 "factors": factors}
        base = codec.planes_to_indices(np.stack(planes, axis=0))
        exceptions = search.add_joint_exceptions(table, base, 0.01)
        packet = codec.encode_packet(codec.FAMILY_BMP, 0, geometry,
                                     model, exceptions)
        decoded = codec.decode_packet(packet)
        sse = float(table[rows, decoded["indices"]].sum(dtype=np.float64))
        candidates.append({
            "family": "GF2_MATRIX_FACTOR",
            "requested_rank": requested_rank,
            "sse": sse,
            "physical_bits": len(packet) * 8,
            "objective": sse + 0.01 * len(packet) * 8,
            "packet_sha256": hashlib.sha256(packet).hexdigest(),
            "_packet": packet,
        })
    winner = min(candidates,
                 key=lambda row: (row["objective"], row["_packet"]))
    public_candidates = [
        {key: value for key, value in row.items() if key != "_packet"}
        for row in candidates
    ]
    public_winner = {key: value for key, value in winner.items()
                     if key != "_packet"}
    return {"winner": public_winner, "candidates": public_candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    producer = authenticate_source(args.source)
    source = args.source.resolve(strict=True)
    launcher = source / "run_cupy_smoke.py"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(launcher)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    envelope = json.loads(completed.stdout)
    require(envelope.get("schema") ==
            "strata-bmp-qtt6-v1-fresh-cupy-launch-receipt",
            "producer launcher schema")
    require(envelope.get("launcher_pid") != os.getpid(),
            "producer launcher distinct process")
    require(envelope.get("command_shape") ==
            ["PYTHON", "-I", "-B", "WORKER", "--nonce", "NONCE"],
            "producer command shape")
    worker = envelope.get("worker")
    require(isinstance(worker, dict), "producer worker receipt")
    require(worker.get("schema") ==
            "strata-bmp-qtt6-v1-fresh-cupy-worker-receipt",
            "producer worker schema")
    require(worker.get("pid") != envelope.get("launcher_pid") and
            isinstance(worker.get("nonce"), str) and len(worker["nonce"]) == 64,
            "worker nonce and distinct PID")
    require(worker.get("isolated_flag") is True and
            worker.get("dont_write_bytecode_flag") is True,
            "worker isolation flags")
    require(Path(worker.get("source_root", "")).resolve(strict=True) == source,
            "worker source root")
    require(worker.get("backend_scope") ==
            "actual_cupy_rank0_rank1_bmp_bounded_search",
            "worker backend scope")
    require(worker.get("candidate_count") == 2 and
            worker.get("held_families") ==
            ["ROBDD GPU search", "canonical QTT GPU search"],
            "worker candidate and hold scope")
    require(worker.get("payload_authority") is False and
            worker.get("model_or_qwen_payload_opened_statted_hashed_or_enumerated")
            is False, "worker payload boundary")

    identity = worker.get("runtime_identity")
    require(isinstance(identity, dict) and identity.get("module") == "cupy",
            "CuPy runtime identity")
    origin = Path(identity.get("module_origin", "")).resolve(strict=True)
    require(hashlib.sha256(origin.read_bytes()).hexdigest() ==
            identity.get("module_file_sha256"), "CuPy module file hash")
    distributions = identity.get("owning_distributions")
    require(isinstance(distributions, dict) and
            identity.get("module_version") in distributions.values(),
            "CuPy distribution ownership/version")
    require(identity.get("compiled_kernel_identity_probe") is True and
            isinstance(identity.get("active_device_id"), int) and
            identity.get("active_device_id") >= 0 and
            bool(identity.get("active_device_name")) and
            bool(identity.get("active_device_pci_bus_id")) and
            identity.get("cuda_visible_device_count", 0) >= 1,
            "active CUDA device receipt")

    workspace = worker.get("workspace")
    require(isinstance(workspace, dict) and
            workspace.get("allocator") == "fresh dedicated cupy MemoryPool",
            "dedicated CuPy pool")
    samples = workspace.get("samples")
    require(isinstance(samples, list) and len(samples) >= 40 and
            samples[0].get("label") == "fresh_pool",
            "CuPy workspace samples")
    reserved = [row.get("total_reserved_bytes") for row in samples]
    used = [row.get("used_bytes") for row in samples]
    require(all(isinstance(value, int) and value >= 0 for value in reserved + used),
            "CuPy workspace integer samples")
    require(all(a <= b for a, b in zip(reserved, reserved[1:])),
            "CuPy pool reservation monotonicity")
    require(max(reserved) == workspace.get("peak_total_reserved_bytes") and
            max(reserved) <= workspace.get("cap_bytes") == 128 * 1024 * 1024,
            "CuPy workspace cap closure")
    labels = [row["label"] for row in samples]
    for required_label in (
        "compiled_kernel_identity_probe", "distortion_and_nearest",
        "rank0_level0", "rank0_level5", "rank0_packet_score",
        "rank1_als_0", "rank1_als_3", "rank1_plane",
        "rank1_level0", "rank1_level5", "rank1_packet_score",
    ):
        require(required_label in labels, f"CuPy sample {required_label}")

    reference = independent_rank01_reference(source)
    winner = worker.get("winner")
    expected = reference["winner"]
    require(isinstance(winner, dict) and
            winner.get("family") == expected["family"] and
            winner.get("requested_rank") == expected["requested_rank"] and
            winner.get("physical_bits") == expected["physical_bits"] and
            winner.get("packet_sha256") == expected["packet_sha256"],
            "independent winner packet closure")
    require(abs(float(winner.get("sse")) - expected["sse"]) <=
            1e-10 * max(1.0, abs(expected["sse"])) and
            abs(float(winner.get("objective")) - expected["objective"]) <=
            1e-10 * max(1.0, abs(expected["objective"])),
            "independent winner metric closure")

    record = {
        "schema": "strata-bmp-qtt6-v1-independent-real-cupy-audit-v1",
        "status": (
            "PASS_FRESH_CUPY_RANK01_SEARCH__QTT_AND_ROBDD_GPU_HELD__"
            "HOLD_EXACT_CPU_WORKSPACE_LEDGER__HOLD_PAYLOAD"
        ),
        "producer": producer,
        "producer_stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "producer_stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
        "runtime_identity": identity,
        "workspace": workspace,
        "worker_winner": winner,
        "independent_reference": reference,
        "qwen_or_other_model_payload_accessed": False,
        "strata_or_coarse_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "network_accessed_by_audit": False,
        "payload_authority": False,
        "claim_boundary": (
            "Source-free rank-0/rank-1 CuPy mechanism only; no Qwen, full-family "
            "GPU, F<=0.8, physical-composite or routed-read result."
        ),
    }
    payload = json.dumps(record, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")


if __name__ == "__main__":
    main()
