#!/usr/bin/env python3
"""Authenticated, mandatory-GPU verifier entry for SILT source-free v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time


ROOT_FILES = (
    "POSTIMPLEMENTATION_REVIEW.md",
    "README.md",
    "cupy_backend_v1.py",
    "design_lock.json",
    "independent_decoder_v1.py",
    "run_synthetic_v1.py",
    "safe_publish.py",
    "silt_v1.py",
    "source_bootstrap.py",
    "test_source_only_v1.py",
    "verify_source_v1.py",
)


def authenticated_root(directory: str) -> str:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    packets: dict[str, bytes] = {}
    try:
        if set(os.listdir(descriptor)) != set(ROOT_FILES):
            raise RuntimeError("authenticated snapshot file set")
        for name in sorted(ROOT_FILES):
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("authenticated snapshot regular files")
            file_descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_descriptor, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                packets[name] = b"".join(chunks)
                if len(packets[name]) != metadata.st_size:
                    raise RuntimeError("authenticated snapshot changed")
            finally:
                os.close(file_descriptor)
    finally:
        os.close(descriptor)
    hasher = hashlib.sha256()
    hasher.update(b"SILT-V1-SOURCE-ROOT\0")
    for name in sorted(ROOT_FILES):
        encoded = name.encode()
        packet = packets[name]
        hasher.update(len(encoded).to_bytes(4, "big"))
        hasher.update(encoded)
        hasher.update(len(packet).to_bytes(8, "big"))
        hasher.update(hashlib.sha256(packet).digest())
    return hasher.hexdigest()


def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authenticated-root", required=True)
    arguments = parser.parse_args()
    source_dir = os.path.dirname(os.path.abspath(__file__))
    observed = authenticated_root(source_dir)
    if observed != arguments.authenticated_root.lower():
        raise RuntimeError("authenticated root changed before import")

    # Only after root authentication may sibling and third-party imports occur.
    sys.path.insert(0, source_dir)
    import importlib.util
    import importlib.metadata
    import unittest

    import cupy
    import numpy
    import pynvml

    module_path = os.path.join(source_dir, "test_source_only_v1.py")
    specification = importlib.util.spec_from_file_location("silt_v1_authenticated_tests", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("exact authenticated test loader")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "schema": "silt-v1-authenticated-source-verifier-receipt",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "authenticated_source_root": observed,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "elapsed_seconds": time.perf_counter() - started,
        "mandatory_gpu": True,
        "mandatory_rtx_5090": True,
        "payload_input_accepted": False,
        "source_gain_claim": False,
        "result_frozen": False,
        "environment_closure": {
            "interpreter": sys.executable,
            "interpreter_sha256": file_sha256(sys.executable),
            "python_version": sys.version,
            "numpy_version": numpy.__version__,
            "cupy_version": cupy.__version__,
            "pynvml_version": importlib.metadata.version("nvidia-ml-py"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_isolated_flag": int(sys.flags.isolated),
            "no_user_site_flag": int(sys.flags.no_user_site),
            "source_member_sha256": {
                name: file_sha256(os.path.join(source_dir, name)) for name in sorted(ROOT_FILES)
            },
        },
        "gpu_telemetry_receipts": list(getattr(module, "GPU_TELEMETRY_RECEIPTS", [])),
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
