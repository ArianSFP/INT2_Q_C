#!/usr/bin/env python3
"""Source-free full-geometry CuPy rehearsal of the STRATA-v2 emitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_v2_codec import common
from strata_v2_codec import emit_and_lock


def block_hashes(words: np.ndarray) -> list[dict]:
    rows = []
    n = 1 << 18
    for block in range(6):
        payload = words[block * n : (block + 1) * n].astype("<u2", copy=False).tobytes()
        rows.append(
            {
                "canonical_block_index": block,
                "nvalues": n,
                "nbytes": 2 * n,
                "source_bf16_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def create_fixture(root: Path, format_path: Path) -> dict:
    route_rows = []
    selection_rows = []
    source_rows = []
    source_dir = root / "sources"
    source_dir.mkdir(parents=True)
    value_count = 768 * 2048
    checkpoint = {"repo": "source-free/synthetic", "revision": "test-only"}
    flat = np.arange(value_count, dtype=np.uint32)
    for triplet in range(6):
        layer, expert = 2 + 3 * triplet, 17 + triplet
        for role_id, role in enumerate(("gate", "up", "down")):
            ordinal = 3 * triplet + role_id
            shape = [768, 2048] if role != "down" else [2048, 768]
            axis = 1 if role == "down" else 0
            tensor = f"model.layers.{layer}.mlp.experts.{expert}.{role}_proj.weight"
            relpath = f"sources/synthetic_{ordinal:02d}_{role}.bf16.bin"
            shard = "model-00001-of-00001.safetensors"
            range_begin = ordinal * (2 * value_count)
            range_end = range_begin + 2 * value_count - 1
            request_url = (
                f"https://huggingface.co/{checkpoint['repo']}/resolve/"
                f"{checkpoint['revision']}/{shard}"
            )
            # Deterministic, finite, signed BF16 values with nonuniform group
            # energy and pair correlation; no external/model source is read.
            mantissa = (flat * np.uint32(29 + 2 * ordinal) + np.uint32(11 * ordinal)) & np.uint32(0x7F)
            exponent = np.uint32(118 + (ordinal % 7)) << np.uint32(7)
            sign = (((flat >> np.uint32(6 + ordinal % 3)) + np.uint32(ordinal)) & 1) << np.uint32(15)
            words = (sign | exponent | mantissa).astype("<u2")
            path = root / relpath
            words.tofile(path)
            digest = common.sha256_file(path)
            route_rows.append(struct.pack(">HHBBH", layer, expert, role_id, axis, 768))
            selection_rows.append(
                {
                    "matrix_ordinal": ordinal,
                    "tensor": tensor,
                    "role": role,
                    "layer": layer,
                    "expert": expert,
                    "dtype": "BF16",
                    "shape": shape,
                    "nvalues": value_count,
                    "nbytes": 2 * value_count,
                    "shard": shard,
                    "absolute_http_byte_range_inclusive": [range_begin, range_end],
                    "future_output_relpath": relpath,
                    "source_bf16_sha256": None,
                    "blocks": [
                        {
                            "canonical_block_index": block,
                            "source_bf16_sha256": None,
                        }
                        for block in range(6)
                    ],
                }
            )
            source_rows.append(
                {
                    "matrix_ordinal": ordinal,
                    "tensor": tensor,
                    "role": role,
                    "layer": layer,
                    "expert": expert,
                    "dtype": "BF16",
                    "shape": shape,
                    "nvalues": value_count,
                    "nbytes": 2 * value_count,
                    "block_count": 6,
                    "shard": shard,
                    "http_range_inclusive": [range_begin, range_end],
                    "http_response": {
                        "status": 206,
                        "request_url": request_url,
                        "requested_range": f"bytes={range_begin}-{range_end}",
                        "content_range": f"bytes {range_begin}-{range_end}/{18 * 2 * value_count + 1}",
                        "content_length": 2 * value_count,
                        "content_encoding": "identity",
                        "body_bytes": 2 * value_count,
                        "body_sha256": digest,
                    },
                    "output_relpath": relpath,
                    "source_bf16_sha256": digest,
                    "blocks": block_hashes(words),
                }
            )
    route_path = root / "route.bin"
    route_path.write_bytes(b"".join(route_rows))
    selection = common.sealed(
        {
            "schema": "int2-qwen-blind-selection-proposal-v2",
            "status": "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
            "checkpoint": checkpoint,
            "matrices": selection_rows,
        }
    )
    selection_path = root / "selection.lock.json"
    common.write_json(selection_path, selection)
    emitter_path = Path(emit_and_lock.__file__).resolve()
    common_path = Path(common.__file__).resolve()
    frozen_artifacts = {
        "emitter": common.sha256_file(emitter_path),
        "common": common.sha256_file(common_path),
        "format": common.sha256_file(format_path),
        "freeze_validator": "2" * 64,
    }
    freeze = common.sealed(
        {
            "schema": "strata_xklt_sc_v2_codec_freeze_v1",
            "status": "frozen_before_blind_source_access",
            "selection_lock_sha256": selection["lock_sha256"],
            "selection_lock_file_sha256": common.sha256_file(selection_path),
            "route_file_sha256": common.sha256_file(route_path),
            "preaccess_state": {},
            "frozen_artifact_sha256s": frozen_artifacts,
        }
    )
    freeze_path = root / "codec_freeze.lock.json"
    common.write_json(freeze_path, freeze)
    validation = common.sealed(
        {
            "schema": "strata_xklt_sc_v2_codec_freeze_validation_v1",
            "status": "validated_before_blind_source_access",
            "passed": True,
            "freeze_path": "blind_protocol_v2/codec_freeze.lock.json",
            "freeze_file_sha256": common.sha256_file(freeze_path),
            "freeze_internal_lock_sha256": freeze["lock_sha256"],
            "executing_validator_sha256": frozen_artifacts["freeze_validator"],
            "frozen_artifact_count": len(frozen_artifacts),
            "development_pooled_relative_mse": 0.049,
            "gaussian_mse_reference": math.exp2(-4.3),
            "physical_bits": common.PHYSICAL_BITS,
            "physical_bpw": common.PHYSICAL_BITS / common.WEIGHTS,
            "preaccess_state": freeze["preaccess_state"],
        }
    )
    validation_path = root / "codec_freeze.validation.json"
    common.write_json(validation_path, validation)
    source_lock = common.sealed(
        {
            "schema": "int2-qwen-blind-source-finalization-v2",
            "status": "all_locked_sources_materialized_and_hash_finalized",
            "dtype": "BF16",
            "checkpoint": checkpoint,
            "selection_lock_sha256": selection["lock_sha256"],
            "codec_freeze": {
                "file_sha256": common.sha256_file(freeze_path),
                "internal_lock_sha256": freeze["lock_sha256"],
                "path_as_invoked": str(freeze_path),
            },
            "codec_freeze_validation": {
                "file_sha256": common.sha256_file(validation_path),
                "internal_lock_sha256": validation["lock_sha256"],
            },
            "source_root": ".",
            "matrix_count": 18,
            "block_count": 108,
            "source_values": common.WEIGHTS,
            "source_bytes": 2 * common.WEIGHTS,
            "matrices": source_rows,
        }
    )
    source_lock_path = root / "source_hashes.lock.json"
    common.write_json(source_lock_path, source_lock)
    return {
        "selection": selection_path,
        "route": route_path,
        "source_lock": source_lock_path,
        "codec_freeze": freeze_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--format",
        type=Path,
        default=Path(__file__).with_name("FORMAT.md"),
    )
    ap.add_argument("--keep", type=Path)
    args = ap.parse_args()
    if args.keep is None:
        context = tempfile.TemporaryDirectory(prefix="strata_v2_synthetic_")
        root = Path(context.name)
    else:
        args.keep.mkdir(parents=True, exist_ok=False)
        root = args.keep
        context = None
    fixture = create_fixture(root, args.format.resolve())
    receipts = []
    lock_hashes = []
    for repetition in range(2):
        output = root / f"emission_{repetition}"
        namespace = SimpleNamespace(
            selection_lock=fixture["selection"],
            route=fixture["route"],
            source_lock=fixture["source_lock"],
            source_root=root,
            protocol_mode="blind",
            codec_freeze=fixture["codec_freeze"],
            format=args.format.resolve(),
            output_dir=output,
        )
        receipts.append(emit_and_lock.emit_candidate(namespace))
        manifest = json.loads((output / "preencoding_manifest.json").read_text())
        lock = json.loads((output / "allocation.lock.json").read_text())
        if not common.verify_internal_seal(lock):
            raise AssertionError("synthetic allocation seal failed")
        if manifest["protocol_mode"] != "blind":
            raise AssertionError("synthetic emission did not remain blind mode")
        common.validate_header(
            (output / "header.bin").read_bytes(),
            (output / "route.bin").read_bytes(),
            (output / "labels_3bit.bin").read_bytes(),
        )
        if not all(
            row["cupy_cpu_bf16_staging_parity"]
            and row["cupy_cpu_q15_code_parity"]
            for row in manifest["klt"]["rows"]
        ):
            raise AssertionError("CuPy parity audit failed")
        lock_hashes.append(lock["lock_sha256"])
    if lock_hashes[0] != lock_hashes[1]:
        raise AssertionError("repeated source-free emissions are not byte-deterministic")
    first, second = root / "emission_0", root / "emission_1"
    compared = ["header.bin", "route.bin", "labels_3bit.bin", "profiles.bin"]
    compared += [f"staging/block_{block:02d}_n{logn}.bf16.bin" for block, logn in enumerate(common.BLOCK_LOG2)]
    for relpath in compared:
        if common.sha256_file(first / relpath) != common.sha256_file(second / relpath):
            raise AssertionError(f"repeated emission differs: {relpath}")
    result = {
        "schema": "strata_xklt_sc_v2_source_free_emitter_test_v1",
        "status": "passed",
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
        "allocation_lock_sha256": lock_hashes[0],
        "profiles": receipts[0]["profile_ids"],
        "projected_relative_mse": receipts[0]["projected_relative_mse"],
        "repeated_emission_assets_match": True,
        "source_payload": "deterministic synthetic BF16 only",
    }
    result_path = root / "test_result.json"
    common.write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if context is not None:
        context.cleanup()


if __name__ == "__main__":
    main()
