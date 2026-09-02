#!/usr/bin/env python3
"""Fresh isolated CuPy encoder worker for LOGIC-Q v3 authority.

This executable is launched with ``python -I -B``.  It imports CuPy before any
repository module, validates the precommitted module/device policy, reads only
the explicitly named synthetic or future authorized source files, runs one
frozen four-level config, and emits literal packet bytes plus a receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def regular_bytes(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
            f"{label} changed during read")
    return payload


def strict_json(payload: bytes, label: str):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                       parse_constant=lambda token: fail(
                           f"{label} nonfinite {token}"))
    require(isinstance(value, dict), f"{label} object")
    return value


def load_authority(path: Path):
    spec = importlib.util.spec_from_file_location(
        "logicq_v3_fresh_worker_authority", path)
    require(spec is not None and spec.loader is not None, "authority import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--packet-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse()
    worker_path = Path(__file__).resolve(strict=True)
    worker_sha = sha256(regular_bytes(worker_path, "worker source"))
    request_path = args.request.resolve(strict=True)
    root = request_path.parent
    require(args.packet_output.resolve().parent == root and
            args.receipt_output.resolve().parent == root,
            "worker outputs stay in request directory")
    request = strict_json(regular_bytes(request_path, "worker request"),
                          "worker request")
    required = {"schema", "precommit_sha256", "backend_policy", "config_id",
                "layer", "slot", "rows", "cols", "source_dtype",
                "source_sha256_by_role", "source_files_by_role", "worker_sha256",
                "v2_source_root_sha256"}
    require(set(request) == required and request["schema"] ==
            "logic-q-v3-fresh-worker-request-v1" and
            request["worker_sha256"] == worker_sha,
            "worker request schema/source pin")

    # -I prevents PYTHONPATH/user-site injection.  Reject a pre-imported CuPy
    # module before the one canonical import performed by this worker.
    require("cupy" not in sys.modules, "CuPy imported before fresh worker gate")
    cp = importlib.import_module("cupy")

    package = worker_path.parent
    authority = load_authority(package / "authority.py")
    receipt_backend = authority.backend_receipt_from_cupy(
        cp, request["backend_policy"])
    research = package.parent
    binder, v1, core = authority.load_dependencies(
        research / "logic_q_label_flexible_algebraic_gate_v2_bound_adapter",
        research / "logic_q_label_flexible_algebraic_gate_v1_capped_adapter",
        research / "logic_q_label_flexible_algebraic_gate_v0")
    require(request["v2_source_root_sha256"] ==
            authority.V2_SOURCE_ROOT_SHA256, "worker v2 source pin")
    configs = [config for config in v1.FROZEN_CONFIGS
               if config.config_id == request["config_id"]]
    require(len(configs) == 1, "worker frozen config")
    config = configs[0]
    require(set(request["source_files_by_role"]) == set(authority.ROLE_ORDER) and
            set(request["source_sha256_by_role"]) == set(authority.ROLE_ORDER),
            "worker exact roles")
    roles = {}
    count = int(request["rows"]) * int(request["cols"])
    np = importlib.import_module("numpy")
    for role in authority.ROLE_ORDER:
        filename = request["source_files_by_role"][role]
        require(isinstance(filename, str) and filename and
                Path(filename).name == filename, "worker safe source filename")
        blob = regular_bytes(root / filename, f"worker source {role}")
        require(sha256(blob) == request["source_sha256_by_role"][role],
                "worker source hash")
        values = authority.decode_source(np, blob, request["source_dtype"], count)
        roles[role] = (cp.asarray(values, dtype=cp.float64),
                       cp.ones(count, dtype=cp.float64))
    encoded = v1.encode_expert(
        cp, core, roles, rows=int(request["rows"]), cols=int(request["cols"]),
        config=config, live=True)
    packet = bytes(encoded["packet"])
    binder.packet_geometry(np, v1, core, packet)
    args.packet_output.write_bytes(packet)
    receipt = {
        "schema": "logic-q-v3-fresh-worker-receipt-v1",
        "request_sha256": sha256(canonical_json(request)),
        "precommit_sha256": request["precommit_sha256"],
        "config_id": request["config_id"],
        "layer": request["layer"], "slot": request["slot"],
        "rows": request["rows"], "cols": request["cols"],
        "source_sha256_by_role": request["source_sha256_by_role"],
        "inner_packet_bytes": len(packet),
        "inner_packet_sha256": sha256(packet),
        "backend_receipt": receipt_backend,
        "worker_sha256": worker_sha,
        "v2_source_root_sha256": authority.V2_SOURCE_ROOT_SHA256,
    }
    receipt["receipt_sha256"] = sha256(canonical_json(receipt))
    args.receipt_output.write_bytes(canonical_json(receipt))


if __name__ == "__main__":
    main()
