#!/usr/bin/env python3
"""Entrypoint compiled only from authenticated immutable snapshot bytes."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from types import MappingProxyType
from typing import Any, Mapping


CPU_AUTHORIZATION = "RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_V3_ATOMIC_CPU"
CUPY_AUTHORIZATION = "RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_V3_ATOMIC_CUPY"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"atomic snapshot runner: {message}")


def _load(snapshot: Mapping[str, bytes], key: str, module_name: str) -> Any:
    require(type(snapshot[key]) is bytes, f"immutable module bytes {key}")
    module = types.ModuleType(module_name)
    module.__file__ = f"<atomic-snapshot>/{key}"
    module.__package__ = ""
    sys.modules[module_name] = module
    code = compile(snapshot[key], module.__file__, "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def snapshot_main(*, snapshot_bytes: Mapping[str, bytes], mode: str,
                  authorization: str, snapshot_receipt: Mapping[str, Any]) -> int:
    require(isinstance(snapshot_bytes, MappingProxyType), "mapping-proxy snapshot")
    require(snapshot_receipt.get("immutable_verified_byte_snapshot") is True,
            "verified snapshot receipt")
    require(mode in ("source-free-cpu", "source-free-cupy"), "source-free mode")
    expected = CPU_AUTHORIZATION if mode == "source-free-cpu" else CUPY_AUTHORIZATION
    require(authorization == expected, "explicit source-free authorization")
    core = _load(snapshot_bytes, "v2/scalable_core.py", "tactic_r384_v3_pinned_core")
    io = _load(snapshot_bytes, "v2/authenticated_io.py", "tactic_r384_v3_pinned_io")
    worker = _load(snapshot_bytes, "v3/coarse_byte_worker.py",
                   "tactic_ramanujan384_v3_atomic_worker")
    adapter = _load(snapshot_bytes, "v3/adapter_atomic.py", "tactic_r384_v3_adapter")
    fixture = _load(snapshot_bytes, "v3/source_free_fixture_atomic.py",
                    "tactic_r384_v3_fixture")
    import numpy as np
    if mode == "source-free-cupy":
        import cupy as xp
        require(xp.cuda.runtime.getDeviceCount() > 0, "CUDA device")
        canonical = core.canonical_gaussian_f64((2, core.BLOCK_VALUES),
                                                core.GAUSSIAN_SEEDS[0])
        copied = xp.asnumpy(xp.asarray(canonical, dtype=xp.float64))
        require(canonical.tobytes(order="C")
                == np.ascontiguousarray(copied, dtype="<f8").tobytes(order="C"),
                "canonical Gaussian CPU/CuPy bytes")
        reference = np.arange(2 * core.BLOCK_VALUES, dtype=np.float64).reshape(
            2, core.BLOCK_VALUES
        )
        cpu_matched, cpu_record = core.moment_matched_gaussian(
            np, reference, core.GAUSSIAN_SEEDS[1],
            (core.BLOCK_VALUES, core.BLOCK_VALUES),
        )
        gpu_matched, gpu_record = core.moment_matched_gaussian(
            xp, xp.asarray(reference), core.GAUSSIAN_SEEDS[1],
            (core.BLOCK_VALUES, core.BLOCK_VALUES),
        )
        require(np.ascontiguousarray(cpu_matched, dtype="<f8").tobytes(order="C")
                == np.ascontiguousarray(xp.asnumpy(gpu_matched),
                                        dtype="<f8").tobytes(order="C")
                and cpu_record["f64_sha256"] == gpu_record["f64_sha256"],
                "moment-matched Gaussian CPU/CuPy bytes")
    else:
        xp = np
    result = fixture.run(xp, core=core, io=io, adapter=adapter)
    if mode == "source-free-cupy":
        xp.cuda.Stream.null.synchronize()
    require(result["physical_rate_bpw"] == 2.5 and result["controls_rerun"],
            "target-rate fixture reached all controls")
    require(result["per_candidate_host_scalar_syncs"] == 0,
            "zero per-candidate host scalar sync")
    require(result["mutable_decoder_object_used"] is False
            and result["zero_import_no_path_coarse_worker"] is True,
            "byte-worker boundary")
    result.update({
        "schema": "tactic-ramanujan384-v3-atomic-source-free-runtime-receipt",
        "runtime_backend": mode, "atomic_snapshot": dict(snapshot_receipt),
        "snapshot_member_sha256": {
            key: hashlib.sha256(snapshot_bytes[key]).hexdigest()
            for key in ("v2/scalable_core.py", "v2/authenticated_io.py",
                        "v3/coarse_byte_worker.py", "v3/adapter_atomic.py",
                        "v3/source_free_fixture_atomic.py")
        },
        "qwen_payload_accessed": False, "coarse_model_payload_accessed": False,
        "network_accessed": False, "synthetic_fixture_mechanism_only": True,
    })
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit("snapshot_runner.py may only be compiled by the external atomic bootstrap")
