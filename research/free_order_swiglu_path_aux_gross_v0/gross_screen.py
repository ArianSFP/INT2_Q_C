#!/usr/bin/env python3
"""CuPy gross-relaxed necessary-bound screen for frozen FOSP-v3/v4 science."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


ORACLE_SHA256 = "9ca6f4bdd4150c8c0c68c0a298c00eb45c088a4af287895ebfdf9bf1e661a070"
BINDINGS_SHA256 = "cd12742910503f23d0d9224e277a030b923f8fc917c75a13a1aff8e9bcde090a"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def strict_json(raw: bytes):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise RuntimeError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def nonfinite(token):
        raise RuntimeError(f"nonfinite JSON value: {token}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def load_oracle(path: Path):
    observed = sha256_file(path)
    if observed != ORACLE_SHA256:
        raise RuntimeError(f"scientific oracle hash mismatch: {observed}")
    spec = importlib.util.spec_from_file_location("fosp_frozen_science", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen scientific oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bf16_values(np, path: Path, expected_hash: str, shape):
    expected_bytes = math.prod(shape) * 2
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"BF16 byte-count mismatch: {path}")
    observed = sha256_file(path)
    if observed != expected_hash:
        raise RuntimeError(f"BF16 hash mismatch: {path}: {observed}")
    words = np.fromfile(path, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
    if not bool(np.isfinite(values).all()):
        raise RuntimeError(f"nonfinite BF16 source: {path}")
    return values


def load_experts(np, workspace: Path, bindings: dict):
    experts = []
    receipts = []
    for expert_row in bindings["experts"]:
        roles = {}
        for role_row in expert_row["roles"]:
            path = workspace / role_row["relative_path"]
            values = bf16_values(np, path, role_row["sha256"], tuple(role_row["shape"]))
            roles[role_row["role"]] = values
            receipts.append({
                "layer": int(expert_row["layer"]),
                "expert": int(expert_row["expert"]),
                "role": role_row["role"],
                "relative_path": role_row["relative_path"],
                "shape": role_row["shape"],
                "bytes": path.stat().st_size,
                "sha256": role_row["sha256"],
            })
        if set(roles) != {"gate", "up", "down"}:
            raise RuntimeError("expert role closure mismatch")
        canonical_down = roles["down"].T
        joined = np.stack((roles["gate"], roles["up"], canonical_down), axis=1)
        if joined.shape != (768, 3, 2048):
            raise RuntimeError("canonical expert geometry mismatch")
        experts.append({
            "ordinal": int(expert_row["ordinal"]),
            "layer": int(expert_row["layer"]),
            "expert": int(expert_row["expert"]),
            "values": joined,
        })
    if len(experts) != 2 or len(receipts) != 6:
        raise RuntimeError("auxiliary source closure mismatch")
    return experts, receipts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("output directory must be absent")
    root = Path(__file__).resolve().parents[1]
    oracle_path = root / "free_order_swiglu_path_oracle_v4" / "scientific_oracle_v3.py"
    bindings_path = root / "free_order_swiglu_path_oracle_v3" / "source_bindings.json"
    bindings_raw = bindings_path.read_bytes()
    if sha256_bytes(bindings_raw) != BINDINGS_SHA256:
        raise RuntimeError("source bindings hash mismatch")
    bindings = strict_json(bindings_raw)
    if bindings.get("schema") != "free_order_swiglu_path_auxiliary_bindings_v1":
        raise RuntimeError("source bindings schema mismatch")
    oracle = load_oracle(oracle_path)

    import cupy as cp
    import numpy as np

    started = time.perf_counter()
    experts, source_receipts = load_experts(np, Path(args.workspace).resolve(), bindings)
    rows = []
    total_energy = 0.0
    total_capture = 0.0
    for source in experts:
        device = cp.asarray(source["values"], dtype=cp.float64)
        energy = float(cp.sum(device * device, dtype=cp.float64).item())
        scores, cross, inverse = oracle._pair_scores(device, cp)
        diagonal = cp.arange(oracle.ROWS)
        if int(cp.count_nonzero(cp.isneginf(scores)).item()) != oracle.ROWS:
            raise RuntimeError("nonself score-mask cardinality mismatch")
        if not bool(cp.isneginf(scores[diagonal, diagonal]).all().item()):
            raise RuntimeError("self pair was not forbidden")
        best_scores = cp.max(scores, axis=1)
        predecessors = cp.argmax(scores, axis=1).astype(cp.uint16)
        if not bool(cp.isfinite(best_scores).all().item()):
            raise RuntimeError("nonfinite best nonself score")
        capture = float(cp.sum(best_scores, dtype=cp.float64).item())
        predecessor_host = cp.asnumpy(predecessors).astype("<u2", copy=False)
        if bool(np.any(predecessor_host == np.arange(oracle.ROWS, dtype=np.uint16))):
            raise RuntimeError("self predecessor survived")
        rows.append({
            "ordinal": source["ordinal"],
            "layer": source["layer"],
            "expert": source["expert"],
            "source_energy": energy,
            "gross_relaxed_capture": capture,
            "energy_reduction": capture / energy,
            "gross_s_bpw": -0.5 * math.log2(1.0 - capture / energy),
            "best_predecessor_sha256_u16le": sha256_bytes(predecessor_host.tobytes()),
            "ordered_nonself_pairs": oracle.ROWS * (oracle.ROWS - 1),
        })
        total_energy += energy
        total_capture += capture
        del scores, cross, inverse, best_scores, predecessors, device
        cp.get_default_memory_pool().free_all_blocks()

    residual_ratio = (total_energy - total_capture) / total_energy
    gross_s = -0.5 * math.log2(residual_ratio)
    if gross_s < oracle.REQUIRED_GROSS_S:
        decision = "HARD_KILL_GROSS_QWEN_RELAXED_NECESSARY_BOUND"
        early_stop = True
    else:
        decision = "SURVIVE_TO_INDEPENDENT_LEGAL_FP16_AUXILIARY_AUDIT"
        early_stop = False
    ledgers = {format(rate, ".2f"): oracle.frame_ledger(rate) for rate in oracle.RATES}
    result = {
        "schema": "free_order_swiglu_path_aux_gross_v0_result",
        "status": decision,
        "early_stop": early_stop,
        "claim_boundary": (
            "Exact frozen-family gross necessary-bound result on two already-open auxiliary experts only; "
            "not fresh validation, pinned-panel evidence, a legal path, finite residual codec, or target achievement."
        ),
        "science": {
            "oracle_sha256": ORACLE_SHA256,
            "source_bindings_sha256": BINDINGS_SHA256,
            "pair_model": "all ordered nonself full 3x3 predecessor-to-target regressions",
            "gross_relaxed_contains_every_legal_path": True,
            "controls_required_after_gross_survival_only": True,
            "required_net_s_bpw": oracle.REQUIRED_S,
            "side_bpw": oracle.SIDE_BPW,
            "required_gross_s_bpw": oracle.REQUIRED_GROSS_S,
        },
        "aggregate": {
            "experts": len(rows),
            "source_energy": total_energy,
            "gross_relaxed_capture": total_capture,
            "energy_reduction": total_capture / total_energy,
            "residual_ratio": residual_ratio,
            "gross_s_bpw": gross_s,
            "net_s_after_side_bpw": gross_s - oracle.SIDE_BPW,
            "fraction_of_required_gross_s": gross_s / oracle.REQUIRED_GROSS_S,
            "projected_optimistic_F_after_side": 2.0 ** (-2.0 * (gross_s - oracle.SIDE_BPW)),
        },
        "experts": rows,
        "physical_ledgers": ledgers,
        "bindings": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "source_receipts": source_receipts,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "compute_capability": cp.cuda.Device().compute_capability,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "access": {
            "auxiliary_qwen_files_opened": len(source_receipts),
            "fresh_validation_files_opened": 0,
            "pinned_panel_files_opened": 0,
            "network_operations": 0,
        },
    }
    output.mkdir(parents=False, exist_ok=False)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": decision,
        "gross_s_bpw": gross_s,
        "required_gross_s_bpw": oracle.REQUIRED_GROSS_S,
        "fraction_of_required_gross_s": gross_s / oracle.REQUIRED_GROSS_S,
        "energy_reduction": total_capture / total_energy,
        "net_s_after_side_bpw": gross_s - oracle.SIDE_BPW,
        "projected_optimistic_F_after_side": result["aggregate"]["projected_optimistic_F_after_side"],
        "result_sha256": sha256_file(result_path),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

