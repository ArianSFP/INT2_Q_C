#!/usr/bin/env python3
"""Authenticated source-free RTX replay for the exact v3 snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


EXPECTED_ROOT = "a1fc85ffdfaa5e7fde25deea98b33d186c915868a8a546333a7fefb64fa9b035"
EXPECTED = {
    "INDEPENDENT_BOOTSTRAP_ABI.md": (7165, "5a43ff712395fb8c7d8354edd0a83960bba96940b99aaf2b2175c94672c77513"),
    "README.md": (9776, "57a114c12121d98fc52ba5de5d44790713c19d83800400ed1e87a6c2e900bae2"),
    "container_codec.py": (80409, "3c81ea3e67a7908a0e28ff05c2fd3f17d7404a24ca44710d0acfdd97d8f4d8d5"),
    "cupy_backend.py": (35251, "e717d602086457a5e3b5fb0746d7a67e9a3584090a22ad675b6e0206e4212424"),
    "design_lock.json": (8203, "f57cc432dc39ac72a83ae3781bff380cf6d8a96b55e26278fd9059e47799d630"),
    "dispatcher_contract.py": (9205, "2a231ef7e7b37f296387bc7825567e372736ced1769917ba7b97839924e83bbf"),
    "fixture_long_memory.py": (4307, "1d425b56ea0923e74996b488ea7c12ef0b70569df19c5468c36812648bb3f6ff"),
    "fixture_portability.py": (11265, "71a8c1eb2c5dad9f6b8e66f106547f09b0bfff449be1aa785b363d3055e2318d"),
    "protocol.py": (17596, "2b5ea430bb73a715c2eda08de359d874fa8a5a823d825f8256d2dff230f6b4f0"),
    "result_envelope.py": (2688, "9ada6c9b6a5fcb57fb8972e05e519e8aada68aeabb740ce3b67bd318cf2b7993"),
    "stage0_census.py": (59116, "fd71686644e13253293644d3793adaacdc1e9977b792771c75f56e08288d89c7"),
    "strata_sc_adapter.py": (36184, "cfdb1f887fc1473f67aa758cd45570d9fd58b33765443e6c87581a43f1435bc7"),
    "test_source_only.py": (59352, "e9025bf1ee1702c5778b6527657fdcaf4edc38229d2c8422284944819c1835ef"),
    "universal_adapter.py": (11577, "dae13363c23e3a59a071b16b36ad282fc71b5e08be158539690e82eefbcbc899"),
    "uwfa_common.py": (33639, "dea23efa7211715ff6fa654cbf98452dc08a318d54874869db2321649d511397"),
    "verify_source.py": (10277, "72926daba955af6bdb3a701bd0192a9fa8e5271e33ab3b9a85920e2103408edf"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def authenticate(package: Path) -> dict[str, bytes]:
    package = package.absolute()
    cursor = Path(package.anchor)
    for component in package.parts[1:]:
        cursor = cursor / component
        if stat.S_ISLNK(os.lstat(cursor).st_mode):
            raise RuntimeError(f"symlink package ancestor: {cursor}")
    if {entry.name for entry in os.scandir(package)} != set(EXPECTED):
        raise RuntimeError("exact inventory mismatch")
    snapshots = {}
    canonical = bytearray()
    for name in sorted(EXPECTED, key=str.lower):
        expected_bytes, expected_sha = EXPECTED[name]
        fd = os.open(package / name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"nonregular source: {name}")
            chunks = []
            while chunk := os.read(fd, 1 << 20):
                chunks.append(chunk)
            data = b"".join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"unstable source: {name}")
        if len(data) != expected_bytes or sha(data) != expected_sha:
            raise RuntimeError(f"source mismatch: {name}")
        snapshots[name] = data
        canonical.extend(name.encode() + b"\0" + str(len(data)).encode() + b"\0" + expected_sha.encode() + b"\n")
    if sha(bytes(canonical)) != EXPECTED_ROOT:
        raise RuntimeError("root mismatch")
    return snapshots


def load(name: str, source: bytes) -> Any:
    module = types.ModuleType(name)
    module.__file__ = f"<independently-authenticated:{name}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: gpu_replay.py ABSOLUTE_PACKAGE")
    snapshots = authenticate(Path(sys.argv[1]))
    common = load("uwfa_v3_gpu_common", snapshots["uwfa_common.py"])
    protocol = load("uwfa_v3_gpu_protocol", snapshots["protocol.py"])
    semantic = load("uwfa_v3_gpu_semantic", snapshots["universal_adapter.py"])
    codec = load("uwfa_v3_gpu_codec", snapshots["container_codec.py"])
    stage = load("uwfa_v3_gpu_stage", snapshots["stage0_census.py"])
    cuda = load("uwfa_v3_gpu_cuda", snapshots["cupy_backend.py"])
    import cupy as cp

    backend = cuda.build_backend(cp)
    all150 = stage.gpu_preflight_all_150(common, backend)
    representative = stage.representative_outer_fold_benchmark(
        common, protocol, codec, semantic, backend
    )
    environment = representative["telemetry"]
    stats = environment["statistics"]
    result = {
        "schema": "uwfa-sc-v3-independent-gpu-replay",
        "authenticated_root": EXPECTED_ROOT,
        "all150": {
            "status": all150["status"],
            "cell_count": all150["cell_count"],
            "streams": all150["streams"],
            "symbols_per_complete_bank": all150["symbols_per_complete_bank"],
            "elapsed_seconds": all150["elapsed_seconds"],
            "cells_canonical_sha256": sha(common.canonical_json(all150["cells"])),
        },
        "representative": {
            "status": representative["status"],
            "winner": representative["outer_fold"]["winner"],
            "container_sha256": representative["outer_fold"]["container_sha256"],
            "runtime_projection": representative["runtime_projection"],
            "measured_phase_statistics_delta": representative["measured_phase_statistics_delta"],
            "model_h2d_bytes_nonzero": representative["model_h2d_bytes_nonzero"],
            "d2h_bytes_nonzero": representative["d2h_bytes_nonzero"],
        },
        "environment": {
            "cupy_version": environment["cupy_version"],
            "cuda_runtime_version": environment["cuda_runtime_version"],
            "cuda_driver_version": environment["cuda_driver_version"],
            "device_id": environment["device_id"],
            "device_name": environment["device_name"],
            "compute_capability": environment["compute_capability"],
            "total_vram_bytes": environment["total_vram_bytes"],
            "receipt_keys": sorted(environment),
            "has_device_uuid": "device_uuid" in environment,
            "has_pci_bus_id": "pci_bus_id" in environment,
        },
        "cumulative_statistics": stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
