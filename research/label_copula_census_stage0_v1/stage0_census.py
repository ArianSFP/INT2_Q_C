#!/usr/bin/env python3
"""Fail-closed source-only preflight for a future raw-label census.

This sealed v1 package has no payload or claim authority.  It validates lifecycle order
and the independently reviewed source/input metadata, then performs only a
CuPy availability preflight.  The exact CPU reference experiment lives in
``label_copula_common.py``; a payload adapter must be added and separately
reviewed before any checkpoint file can be opened.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_COMMON_PATH = Path(os.path.abspath(__file__)).with_name("label_copula_common.py")
_COMMON_SPEC = importlib.util.spec_from_file_location("label_copula_common", _COMMON_PATH)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load same-directory label_copula_common.py")
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules["label_copula_common"] = _COMMON_MODULE
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)

from label_copula_common import (
    AUTHORIZATION,
    INPUT_LOCK_SCHEMA,
    REVIEW_SCHEMA,
    CompletionLastOutput,
    HeldRegularFile,
    pretty_json,
    require,
    sha256_bytes,
    strict_json_loads,
)


def _dynamic_verify_source() -> dict[str, object]:
    path = Path(os.path.abspath(__file__)).with_name("verify_source.py")
    spec = importlib.util.spec_from_file_location("label_copula_verify_source", path)
    require(spec is not None and spec.loader is not None, "source verifier import")
    module = importlib.util.module_from_spec(spec)
    sys.modules["label_copula_verify_source"] = module
    spec.loader.exec_module(module)
    return module.verify_package(Path(os.path.abspath(__file__)).parent)


def _review(path: Path, manifest_sha256: str, entrypoint_sha256: str) -> dict[str, object]:
    with HeldRegularFile(path) as held:
        record = strict_json_loads(held.read_all())
        held.verify_stable()
    require(isinstance(record, dict) and record.get("schema") == REVIEW_SCHEMA, "review schema")
    require(record.get("status") == "PASS_INDEPENDENT_SOURCE_REVIEW", "review status")
    require(record.get("source_manifest_sha256") == manifest_sha256, "review source binding")
    require(record.get("entrypoint_sha256") == entrypoint_sha256, "review entrypoint-byte binding")
    require(record.get("payloads_opened") == 0 and record.get("cuda_jobs") == 0, "source-only review")
    require(record.get("payload_authority") is False, "a source review cannot grant payload authority")
    return record


def _input_metadata(path: Path) -> dict[str, object]:
    with HeldRegularFile(path) as held:
        record = strict_json_loads(held.read_all())
        held.verify_stable()
    require(isinstance(record, dict) and record.get("schema") == INPUT_LOCK_SCHEMA, "input lock schema")
    require(record.get("stream_view") == "A_raw_normalized_gaussian_lloyd4", "v1 raw-label-only view")
    require(record.get("canonical_orientation") == "Gate[j,k],Up[j,k],Down[k,j]", "canonical orientation")
    require(record.get("payload_authority") is False, "metadata lock cannot grant payload authority")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # This branch is deliberately before output creation, input resolution,
    # source verification, CuPy import, and any possible CUDA initialization.
    if args.authorization != AUTHORIZATION:
        print("authorization mismatch: CuPy not imported; no input resolved; no output created", file=sys.stderr)
        return 2

    try:
        with CompletionLastOutput(args.output) as output:
            source = _dynamic_verify_source()
            manifest_sha256 = str(source["manifest_sha256"])
            with HeldRegularFile(Path(os.path.abspath(__file__))) as held_entrypoint:
                entrypoint_sha256 = sha256_bytes(held_entrypoint.read_all())
                held_entrypoint.verify_stable()
            review = _review(args.review_receipt, manifest_sha256, entrypoint_sha256)
            metadata = _input_metadata(args.input_lock)

            # CuPy is intentionally late.  Future GPU work is CuPy-only; this
            # v1 preflight does not open any checkpoint or initialize a payload
            # adapter.  Import success is not a scientific result.
            import cupy as cp

            preflight = {
                "schema": "label-copula-census-cupy-preflight-v1",
                "status": "PASS_CUPY_AVAILABLE_NO_PAYLOAD_OR_CLAIM_AUTHORITY",
                "cupy_version": str(cp.__version__),
                "source_manifest_sha256": manifest_sha256,
                "entrypoint_sha256": entrypoint_sha256,
                "review_sha256": str(review.get("receipt_sha256", "not-self-sealed")),
                "input_lock_schema": metadata["schema"],
                "payloads_opened": 0,
                "cuda_kernels_launched": 0,
                "payload_authority": False,
                "claim_authority": False,
                "public_token_or_self_seal_is_authority": False,
                "future_payload_launch_requirement": "An independently pinned bootstrap outside this producer package must authenticate and execute the reviewed entrypoint bytes.",
                "claim_boundary": "Lifecycle/CuPy preflight only; not a label-census result and incapable of opening a payload.",
            }
            output.write_new("preflight.json", pretty_json(preflight))
            output.complete(manifest_sha256)
        return 0
    except Exception as exc:
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
