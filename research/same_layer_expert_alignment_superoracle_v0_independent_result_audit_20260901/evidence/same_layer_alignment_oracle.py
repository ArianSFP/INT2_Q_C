#!/usr/bin/env python3
"""CuPy illegal many-to-one same-layer expert-alignment super-oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


DESIGN = "design_lock.json"
MANIFEST_SHA256 = "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782"
EXPERTS = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120)
ROLES = ("up", "down")
ROWS, COLS = 768, 2048
REQUIRED_CAPTURE = 0.14566207552117194
CUSHION = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bf16(np, path: Path, role: str):
    raw = path.read_bytes()
    if len(raw) != ROWS * COLS * 2:
        raise ValueError(f"payload length: {path}")
    words = np.frombuffer(raw, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    shape = (COLS, ROWS) if role == "down" else (ROWS, COLS)
    matrix = values.reshape(shape)
    return (matrix.T.copy() if role == "down" else matrix.copy()).astype(np.float64)


def exact_selected_capture(cp, target, references, selected):
    target64 = cp.asarray(target, dtype=cp.float64)
    chosen64 = cp.asarray(references[selected], dtype=cp.float64)
    target_mean = cp.mean(target64, axis=1, keepdims=True, dtype=cp.float64)
    chosen_mean = cp.mean(chosen64, axis=1, keepdims=True, dtype=cp.float64)
    target_centered = target64 - target_mean
    chosen_centered = chosen64 - chosen_mean
    target_centered_energy = cp.sum(target_centered * target_centered, axis=1,
                                    dtype=cp.float64)
    chosen_energy = cp.sum(chosen_centered * chosen_centered, axis=1,
                           dtype=cp.float64)
    dot = cp.sum(target_centered * chosen_centered, axis=1, dtype=cp.float64)
    regression = cp.where(chosen_energy > 0.0, dot * dot / chosen_energy, 0.0)
    mean_capture = COLS * cp.square(target_mean[:, 0])
    captured = float(cp.sum(regression + mean_capture, dtype=cp.float64).item())
    energy = float(cp.sum(target64 * target64, dtype=cp.float64).item())
    return captured, energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    workspace = args.workspace.resolve(strict=True)
    manifest_path = workspace / "agent_rd_structure_diag_cross_expert_sources.json"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("source manifest hash")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experts_in_order") != list(EXPERTS) or manifest.get("roles_in_order") != list(ROLES):
        raise ValueError("source manifest order")
    rows = manifest.get("tensors")
    if not isinstance(rows, list) or len(rows) != len(EXPERTS) * len(ROLES):
        raise ValueError("source manifest tensor closure")

    import cupy as cp
    import numpy as np

    started = time.time()
    arrays = {}
    receipts = []
    for row in rows:
        expert, role = int(row["expert"]), str(row["role"])
        if expert not in EXPERTS or role not in ROLES or int(row["bytes"]) != ROWS * COLS * 2:
            raise ValueError("tensor declaration")
        path = (workspace / str(row["local_path"])).resolve(strict=True)
        try:
            path.relative_to(workspace)
        except ValueError as error:
            raise ValueError("tensor path escapes workspace") from error
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise ValueError(f"tensor hash: {path}")
        arrays[(expert, role)] = load_bf16(np, path, role)
        receipts.append({"expert": expert, "role": role, "bytes": path.stat().st_size,
                         "sha256": observed, "local_path": row["local_path"]})
    if set(arrays) != {(expert, role) for expert in EXPERTS for role in ROLES}:
        raise ValueError("tensor key closure")

    scored = []
    pooled_capture = 0.0
    pooled_energy = 0.0
    for target_expert in EXPERTS:
        for role in ROLES:
            target = arrays[(target_expert, role)]
            reference_experts = [expert for expert in EXPERTS if expert != target_expert]
            references = np.concatenate(
                [arrays[(expert, role)] for expert in reference_experts], axis=0)
            if references.shape != (ROWS * (len(EXPERTS) - 1), COLS):
                raise ValueError("reference geometry")
            target32 = cp.asarray(target, dtype=cp.float32)
            refs32 = cp.asarray(references, dtype=cp.float32)
            target32 -= cp.mean(target32, axis=1, keepdims=True, dtype=cp.float32)
            refs32 -= cp.mean(refs32, axis=1, keepdims=True, dtype=cp.float32)
            target_norm = cp.sqrt(cp.sum(target32 * target32, axis=1, keepdims=True,
                                         dtype=cp.float32))
            ref_norm = cp.sqrt(cp.sum(refs32 * refs32, axis=1, keepdims=True,
                                      dtype=cp.float32))
            target32 /= cp.maximum(target_norm, cp.float32(math.ldexp(1.0, -60)))
            refs32 /= cp.maximum(ref_norm, cp.float32(math.ldexp(1.0, -60)))
            correlations = target32 @ refs32.T
            selected = cp.asnumpy(cp.argmax(cp.square(correlations), axis=1)).astype(np.int64)
            captured, energy = exact_selected_capture(cp, target, references, selected)
            selected_expert = np.asarray([reference_experts[index // ROWS]
                                          for index in selected], dtype=np.int64)
            selected_row = selected % ROWS
            selection_hash = hashlib.sha256(
                np.stack((selected_expert, selected_row), axis=1).astype("<i8").tobytes()
            ).hexdigest()
            pooled_capture += captured
            pooled_energy += energy
            scored.append({"target_expert": target_expert, "role": role,
                           "captured_energy": captured, "source_energy": energy,
                           "capture": captured / energy, "selection_sha256": selection_hash})
            print(f"expert={target_expert:03d} role={role} capture={captured / energy:.9f}", flush=True)
            del target32, refs32, correlations

    capture = pooled_capture / pooled_energy
    favourable_capture = min(1.0, capture + CUSHION)
    status = ("HARD_KILL_SAME_LAYER_UP_DOWN_ANCESTRY" if favourable_capture < REQUIRED_CAPTURE
              else "PROMOTE_TO_MATCHED_CONTROL_AND_LEGAL_MAPPING")
    script_path = Path(__file__).resolve()
    design_path = script_path.parent / DESIGN
    result = {
        "schema": "same-layer-expert-alignment-superoracle-result-v0",
        "status": status,
        "claim_boundary": (
            "Illegal many-to-one, role-inconsistent, source-fitted Up/Down oracle on sixteen "
            "auxiliary layer-15 experts. A kill rejects this ancestry signal as the sole missing "
            "module after the existing composite; Gate and nonlinear/activation-aware codecs are not covered."
        ),
        "capture": capture,
        "absolute_capture_cushion": CUSHION,
        "favourable_capture": favourable_capture,
        "required_capture": REQUIRED_CAPTURE,
        "shortfall": REQUIRED_CAPTURE - favourable_capture,
        "scored": scored,
        "bindings": {"source_manifest_sha256": sha256_file(manifest_path),
                     "script_sha256": sha256_file(script_path),
                     "design_sha256": sha256_file(design_path), "sources": receipts},
        "runtime": {"elapsed_seconds": time.time() - started, "python": os.sys.version,
                    "numpy": np.__version__, "cupy": cp.__version__,
                    "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"])},
        "access": {"auxiliary_payloads_opened": len(receipts), "pinned_payloads_opened": 0,
                   "fresh_validation_payloads_opened": 0},
    }
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": status, "capture": capture,
                      "favourable_capture": favourable_capture,
                      "result_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
