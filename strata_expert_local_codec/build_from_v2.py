#!/usr/bin/env python3
"""Seal an expert-affine staging/allocation plan from the audited v2 staging.

The v2 staging files contain the exact KLT-staged BF16 words in global label
order.  This tool inverts that order, then repartitions the same words without
changing a single staged value.  It must finish and seal every profile and
seed before the arithmetic encoder is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_expert_local_codec import common
from strata_v2_codec import common as v2_common


def asset_row(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relpath": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def gpu_energy(words: np.ndarray) -> float:
    device_words = cp.asarray(words, dtype=cp.uint16)
    values = (device_words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
    values64 = values.astype(cp.float64)
    result = float(cp.sum(values64 * values64, dtype=cp.float64).get())
    del device_words, values, values64
    cp.get_default_memory_pool().free_all_blocks()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_run = args.v2_run.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.mkdir(parents=True)
    staging_dir = output / "staging"
    staging_dir.mkdir()

    old_manifest_path = source_run / "preencoding_manifest.json"
    old_manifest_bytes = old_manifest_path.read_bytes()
    old_manifest = json.loads(old_manifest_bytes.decode("utf-8"))
    route = (source_run / "route.bin").read_bytes()
    labels_packed = (source_run / "labels_3bit.bin").read_bytes()
    old_header = (source_run / "header.bin").read_bytes()
    v2_common.validate_header(old_header, route, labels_packed)
    route_rows = common.parse_route(route)
    labels = common.unpack_labels(labels_packed)

    old_blocks = old_manifest.get("blocks")
    if not isinstance(old_blocks, list) or len(old_blocks) != 14:
        raise ValueError("v2 manifest block coverage mismatch")
    sorted_parts: list[np.ndarray] = []
    old_staging_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(old_blocks):
        if int(row.get("block_ordinal", -1)) != ordinal:
            raise ValueError("v2 staging ordinal mismatch")
        path = source_run / str(row["staging_relpath"])
        if common.sha256_file(path) != row["staging_sha256"]:
            raise ValueError(f"v2 staging hash mismatch block {ordinal}")
        words = np.fromfile(path, dtype="<u2")
        if words.size != int(row["values"]):
            raise ValueError(f"v2 staging value count mismatch block {ordinal}")
        sorted_parts.append(words.reshape(-1, common.GROUP_VALUES))
        old_staging_rows.append(asset_row(path, source_run))
    globally_sorted = np.concatenate(sorted_parts, axis=0)
    global_order = np.lexsort((np.arange(common.GROUPS, dtype=np.int64), labels))
    canonical = np.empty_like(globally_sorted)
    canonical[global_order] = globally_sorted

    block_ordinals = common.expected_block_group_ordinals(labels)
    energies = np.empty(common.BLOCKS, dtype=np.float64)
    block_rows: list[dict[str, Any]] = []
    for ordinal, (logn, selected_ordinals) in enumerate(
        zip(common.BLOCK_LOG2, block_ordinals, strict=True)
    ):
        selected = np.ascontiguousarray(canonical[selected_ordinals], dtype="<u2")
        path = staging_dir / f"block_{ordinal:02d}_n{logn}.bf16.bin"
        selected.tofile(path)
        energies[ordinal] = gpu_energy(selected)
        block_rows.append(
            {
                "block_ordinal": ordinal,
                "block_log2": logn,
                "values": 1 << logn,
                "groups": len(selected_ordinals),
                "owner_experts": common.block_owner_experts(ordinal),
                "segment": "private" if ordinal < common.PRIVATE_BLOCKS else "paired_tail",
                "source_energy_fp64": float(energies[ordinal]),
                "selected_group_ordinals_sha256": hashlib.sha256(
                    selected_ordinals.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "staging_relpath": str(path.relative_to(output)).replace("\\", "/"),
                "staging_bytes": path.stat().st_size,
                "staging_sha256": common.sha256_file(path),
            }
        )

    if not np.isclose(
        float(energies.sum(dtype=np.float64)),
        gpu_energy(canonical),
        rtol=0.0,
        atol=1e-10,
    ):
        raise AssertionError("repartitioning changed staged energy")

    profiles, allocation = common.allocate_profiles(energies)
    coefficients_raw = struct.unpack_from("<12f", old_header, 32)
    coefficients = [
        (np.float32(coefficients_raw[2 * index]), np.float32(coefficients_raw[2 * index + 1]))
        for index in range(common.EXPERTS)
    ]
    angle_codes = list(struct.unpack_from("<6h", old_header, 80))
    header = common.build_header(coefficients, angle_codes, route, labels_packed)
    profile_bytes = profiles.tobytes()
    for row in block_rows:
        ordinal = int(row["block_ordinal"])
        sc_seed, rht_seed, digest = common.derive_seeds(
            header, route, labels_packed, profile_bytes, ordinal
        )
        q = int(profiles[ordinal])
        row.update(
            {
                "profile_id": q,
                "nominal_rate_bpw": common.PROFILE_BASE + q / 256.0,
                "test_distortion": 2.0 ** (-2.0 * (common.PROFILE_BASE + q / 256.0)),
                "sc_seed_u32": sc_seed,
                "rht_seed_u64": rht_seed,
                "seed_digest_sha256": digest,
            }
        )

    assets = {
        "header.bin": header,
        "route.bin": route,
        "labels_3bit.bin": labels_packed,
        "profiles.bin": profile_bytes,
    }
    asset_rows: dict[str, dict[str, Any]] = {}
    for name, payload in assets.items():
        path = output / name
        path.write_bytes(payload)
        asset_rows[name] = asset_row(path, output)

    source_rows = old_manifest.get("bindings", {}).get("sources")
    if not isinstance(source_rows, list) or len(source_rows) != common.MATRICES:
        raise ValueError("v2 manifest source binding coverage mismatch")
    authenticated_sources: list[dict[str, Any]] = []
    for ordinal, row in enumerate(source_rows):
        route_row = route_rows[ordinal]
        expected_role = str(route_row["role"])
        expected_axis = "column" if expected_role == "down" else "row"
        expected_shape = [2048, 768] if expected_role == "down" else [768, 2048]
        expected_tensor = (
            f"model.layers.{route_row['layer']}.mlp.experts.{route_row['expert']}."
            f"{expected_role}_proj.weight"
        )
        if (
            int(row.get("matrix_ordinal", -1)) != ordinal
            or row.get("role") != expected_role
            or row.get("axis") != expected_axis
            or list(row.get("shape", [])) != expected_shape
            or row.get("tensor") != expected_tensor
        ):
            raise ValueError(f"v2 source/route binding mismatch matrix {ordinal}")
        relpath = str(row["source_relpath"])
        path = source_root / relpath
        if not path.is_file() or path.stat().st_size != 2 * 768 * common.GROUP_VALUES:
            raise ValueError(f"source file geometry mismatch matrix {ordinal}: {path}")
        digest = common.sha256_file(path)
        if digest != row["source_bf16_sha256"]:
            raise ValueError(f"source file hash mismatch matrix {ordinal}")
        authenticated_sources.append(
            {
                "matrix_ordinal": ordinal,
                "tensor": row["tensor"],
                "role": expected_role,
                "axis": expected_axis,
                "shape": row["shape"],
                "source_relpath": relpath,
                "source_bf16_sha256": digest,
                "bytes": path.stat().st_size,
            }
        )

    plan = common.sealed(
        {
            "schema": "strata_expert_affine_n20n21_plan_v1",
            "status": "sealed_before_arithmetic_encoding",
            "architecture": (
                "expert-affine XKLT/STRATA with two private N21 blocks per expert "
                "and three paired N20 tails"
            ),
            "source_run": {
                "path": str(source_run),
                "manifest_sha256": hashlib.sha256(old_manifest_bytes).hexdigest(),
                "container_sha256": common.sha256_file(source_run / "strata_xklt_sc_v2.bin"),
                "staging": old_staging_rows,
            },
            "source_root": str(source_root),
            "sources": authenticated_sources,
            "assets": asset_rows,
            "blocks": block_rows,
            "allocation": allocation,
            "physical_ledger": {
                "header_bytes": common.HEADER_BYTES,
                "route_bytes": common.ROUTE_BYTES,
                "label_bytes": common.LABEL_BYTES,
                "directory_bytes": common.DIRECTORY_BYTES,
                "reservoir_bytes": common.RESERVOIR_BYTES,
                "physical_bytes": common.PHYSICAL_BYTES,
                "physical_bits": common.PHYSICAL_BITS,
                "physical_bpw": common.PHYSICAL_BITS / common.WEIGHTS,
                "reserve_bits": common.GLOBAL_RESERVE_BITS,
            },
            "coverage": {
                "experts": common.EXPERTS,
                "matrices": common.MATRICES,
                "groups": common.GROUPS,
                "weights": common.WEIGHTS,
                "blocks": common.BLOCKS,
                "every_group_once": True,
                "cupy_energy_sum_fp64": float(energies.sum(dtype=np.float64)),
            },
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "cupy": cp.__version__,
                "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            },
        }
    )
    common.write_json(output / "plan.lock.json", plan)
    print(json.dumps({
        "status": plan["status"],
        "output": str(output),
        "profiles": allocation["profile_ids"],
        "projected_relative_mse": allocation["projected_relative_mse"],
        "physical_bpw": 2.5,
        "lock_sha256": plan["lock_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
