#!/usr/bin/env python3
"""Emit, seal, encode once, and pack one STRATA-XKLT-SC v2 artifact.

There is no retry, resume, rate adjustment, or result-conditioned branch.
Worker count changes scheduling only.  The emitter and allocation lock finish
before the first of exactly fourteen encoder subprocesses is started.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_v2_codec import common
from strata_v2_codec import emit_and_lock


PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "cupy": "cupy-cuda12x",
    "scipy": "scipy",
    "cuda.pathfinder": "cuda-pathfinder",
}


def cuda_runtime_receipt() -> dict[str, Any]:
    import cupy as cp

    device = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "cupy_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "device_name": device["name"].decode(),
        "compute_capability": [int(device["major"]), int(device["minor"])],
    }


def run_process(command: list[str], cwd: Path, log: Path) -> int:
    process = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout, encoding="utf-8", newline="\n")
    return int(process.returncode)


def verify_blind_runtime_freeze(args: argparse.Namespace) -> dict[str, Any]:
    """Fail before emission unless the freeze pins every executing component."""
    freeze_path = args.codec_freeze.resolve()
    freeze_payload = freeze_path.read_bytes()
    freeze_file_sha256 = hashlib.sha256(freeze_payload).hexdigest()
    freeze = json.loads(freeze_payload.decode("utf-8"))
    if not common.verify_internal_seal(freeze):
        raise ValueError("codec freeze internal seal is invalid")
    expected_freeze_schema = (
        "strata_xklt_sc_v2_codec_freeze_v1"
        if args.protocol_mode == "blind"
        else "polaris_strata_blind_codec_freeze_v1"
    )
    if (freeze.get("schema"), freeze.get("status")) != (
        expected_freeze_schema,
        "frozen_before_blind_source_access",
    ):
        raise ValueError("codec freeze schema/status contract mismatch")
    base_encoder = args.workspace.resolve() / "agent_polaris_qwen_rht_encoder.py"
    bec_builder = args.workspace.resolve() / "bg_codec_bec_encoder.py"
    expected_encoder = args.workspace.resolve() / "strata_v2_codec" / "polar_encoder.py"
    if args.encoder.resolve() != expected_encoder:
        raise ValueError(
            "encoder must be the workspace strata_v2_codec/polar_encoder.py so its "
            "base/BEC import origin is frozen"
        )
    runner_python_invocation = Path(sys.executable).absolute()
    requested_python_invocation = args.python.absolute()
    if runner_python_invocation != requested_python_invocation:
        raise ValueError(
            "the runner/emitter interpreter differs from the requested encoder "
            "interpreter: "
            f"{runner_python_invocation} != {requested_python_invocation}"
        )
    if runner_python_invocation.resolve(strict=True) != requested_python_invocation.resolve(
        strict=True
    ):
        raise ValueError("runner and encoder Python interpreter targets differ")
    paths = {
        # Preserve the venv entry path when executing.  Resolving its symlink
        # would launch the system interpreter without the venv site-packages.
        "python_interpreter": runner_python_invocation,
        "runner": Path(__file__).resolve(),
        "polar_encoder": args.encoder.resolve(),
        "base_encoder": base_encoder,
        "procedural_bec_builder": bec_builder,
        "common": Path(common.__file__).resolve(),
        "emitter": Path(emit_and_lock.__file__).resolve(),
        "format": args.format.resolve(),
        "independent_auditor": args.independent_auditor.resolve(),
    }
    rows = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required blind runtime artifact is absent: {name}: {path}")
        rows[name] = {"path": str(path), "sha256": common.sha256_file(path)}
    package_rows = {
        name: common.distribution_tree_receipt(name, distribution)
        for name, distribution in PACKAGE_DISTRIBUTIONS.items()
    }
    cuda_row = cuda_runtime_receipt()
    if args.protocol_mode == "blind":
        runtime_environment = freeze.get("runtime_environment")
        if not isinstance(runtime_environment, dict):
            raise ValueError("blind codec freeze has no runtime_environment map")
        frozen_python = runtime_environment.get("python_interpreter")
        if not isinstance(frozen_python, dict):
            raise ValueError("blind codec freeze has no frozen Python interpreter receipt")
        invocation_path = Path(str(frozen_python.get("invocation_path", ""))).absolute()
        if args.python.absolute() != invocation_path:
            raise ValueError(
                "blind Python invocation path differs from the frozen venv entry path: "
                f"{args.python.absolute()} != {invocation_path}"
            )
        if runner_python_invocation != invocation_path:
            raise ValueError(
                "executing runner/emitter Python differs from the frozen invocation path"
            )
        if rows["python_interpreter"]["sha256"] != frozen_python.get("sha256"):
            raise ValueError("blind Python interpreter digest differs from the freeze")
        if args.python.resolve(strict=True) != Path(
            str(frozen_python.get("resolved_path", ""))
        ).resolve(strict=True):
            raise ValueError("blind Python interpreter target differs from the freeze")
        if sys.version.split()[0] != frozen_python.get("version"):
            raise ValueError("executing Python version differs from the freeze")
        if runtime_environment.get("packages") != package_rows:
            raise ValueError("blind NumPy/CuPy/SciPy distribution trees differ from the freeze")
        if runtime_environment.get("cuda") != cuda_row:
            raise ValueError("blind CUDA runtime/driver/device receipt differs from the freeze")
        frozen = freeze.get("frozen_artifact_sha256s")
        if not isinstance(frozen, dict):
            raise ValueError("blind codec freeze has no frozen_artifact_sha256s map")
        freeze_names = {
            "python_interpreter": "python_interpreter",
            "runner": "one_shot_runner",
            "polar_encoder": "polar_encoder",
            "base_encoder": "base_cupy_encoder",
            "procedural_bec_builder": "procedural_q31_bec",
            "common": "common",
            "emitter": "emitter",
            "format": "format",
            "independent_auditor": "independent_auditor",
        }
        mismatched = {
            runtime_name: {
                "actual": rows[runtime_name]["sha256"],
                "frozen": frozen.get(freeze_name),
            }
            for runtime_name, freeze_name in freeze_names.items()
            if rows[runtime_name]["sha256"] != frozen.get(freeze_name)
        }
        if mismatched:
            raise ValueError(
                "codec freeze named runtime bindings differ: " f"{mismatched}"
            )
    return {
        "codec_freeze": {
            "path": str(freeze_path),
            "file_sha256": freeze_file_sha256,
            "internal_lock_sha256": freeze["lock_sha256"],
        },
        "artifacts": rows,
        "packages": package_rows,
        "cuda": cuda_row,
    }


def verify_runtime_snapshot(runtime_freeze: dict[str, Any]) -> None:
    """Close runtime-code/interpreter TOCTOU windows around encoder subprocesses."""
    freeze = runtime_freeze["codec_freeze"]
    if common.sha256_file(Path(freeze["path"])) != freeze["file_sha256"]:
        raise RuntimeError("codec freeze changed after the initial runtime gate")
    for name, row in runtime_freeze["artifacts"].items():
        if common.sha256_file(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"frozen runtime artifact changed: {name}")
    current_packages = {
        name: common.distribution_tree_receipt(name, distribution)
        for name, distribution in PACKAGE_DISTRIBUTIONS.items()
    }
    if current_packages != runtime_freeze.get("packages"):
        raise RuntimeError("frozen package distribution tree changed")
    if cuda_runtime_receipt() != runtime_freeze.get("cuda"):
        raise RuntimeError("frozen CUDA runtime/driver/device receipt changed")


def verify_preencoding(output_dir: Path, expected_protocol_mode: str) -> tuple[dict, dict]:
    manifest_path = output_dir / "preencoding_manifest.json"
    lock_path = output_dir / "allocation.lock.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not common.verify_internal_seal(lock):
        raise AssertionError("allocation lock internal seal is invalid")
    if lock["manifest_sha256"] != common.sha256_file(manifest_path):
        raise AssertionError("allocation lock does not bind the manifest")
    if manifest["status"] != "complete_and_allocation_sealed_before_encoding":
        raise AssertionError("pre-encoding manifest has wrong state")
    if (
        manifest.get("protocol_mode") != expected_protocol_mode
        or lock.get("protocol_mode") != expected_protocol_mode
    ):
        raise AssertionError("pre-encoding manifest/lock protocol mode mismatch")
    if common.canonical_bytes(lock["blocks"]) != common.canonical_bytes(manifest["blocks"]):
        raise AssertionError("allocation lock and manifest block decisions differ")
    if common.canonical_bytes(lock["allocation"]) != common.canonical_bytes(manifest["allocation"]):
        raise AssertionError("allocation lock and manifest DP records differ")
    for filename, row in manifest["assets"].items():
        path = output_dir / filename
        if path.stat().st_size != int(row["bytes"]) or common.sha256_file(path) != row["sha256"]:
            raise AssertionError(f"sealed asset drift: {filename}")
    for block in manifest["blocks"]:
        path = output_dir / block["staging_relpath"]
        if path.stat().st_size != int(block["staging_bytes"]):
            raise AssertionError(f"staging size drift block {block['block_ordinal']}")
        if common.sha256_file(path) != block["staging_sha256"]:
            raise AssertionError(f"staging hash drift block {block['block_ordinal']}")
    profiles = (output_dir / "profiles.bin").read_bytes()
    sealed_profiles = bytes(int(block["profile_id"]) for block in manifest["blocks"])
    if profiles != sealed_profiles or profiles != bytes(manifest["allocation"]["profile_ids"]):
        raise AssertionError("profiles.bin, allocation lock, and block records differ")
    common.validate_header(
        (output_dir / "header.bin").read_bytes(),
        (output_dir / "route.bin").read_bytes(),
        (output_dir / "labels_3bit.bin").read_bytes(),
    )
    return manifest, lock


def encode_one(
    args: argparse.Namespace,
    manifest: dict,
    block: dict,
    runtime_freeze: dict[str, Any],
) -> dict[str, Any]:
    ordinal = int(block["block_ordinal"])
    n = int(block["values"])
    # Every worker closes its own executable/interpreter TOCTOU window.  The
    # outer checks remain as a whole-run guard around emission and packing.
    verify_runtime_snapshot(runtime_freeze)
    source = args.output_dir.resolve() / block["staging_relpath"]
    if common.sha256_file(source) != block["staging_sha256"]:
        raise AssertionError(f"source TOCTOU check failed immediately before block {ordinal}")
    encoded_dir = args.output_dir.resolve() / "encoded"
    output = encoded_dir / f"block_{ordinal:02d}.json"
    command = [
        str(args.python.absolute()),
        str(args.encoder.resolve()),
        "--polar-repo", str(args.polar_repo.resolve()),
        "--block-length", str(n),
        "--trials", "1",
        "--sigma-source", "1.0",
        "--test-distortion", repr(float(block["test_distortion"])),
        "--eta", "0.25",
        "--alphabet-size", "64",
        "--decision", "map",
        "--seed", str(int(block["sc_seed_u32"])),
        "--input-bf16", str(source),
        "--input-block-start", "0",
        "--canonical-source-id", (
            f"strata-xklt-sc-v2:{manifest['bindings']['route']['sha256']}:block:{ordinal}"
        ),
        "--canonical-block-index", "0",
        "--apply-rht",
        "--rht-seed", str(int(block["rht_seed_u64"])),
        "--emit-container-hex",
        "--output", str(output),
    ]
    returncode = run_process(
        command,
        args.workspace.resolve(),
        args.output_dir.resolve() / "encode_logs" / f"block_{ordinal:02d}.log",
    )
    if returncode:
        raise RuntimeError(f"one-shot encoder invocation failed block {ordinal}; no retry")
    verify_runtime_snapshot(runtime_freeze)
    metadata = json.loads(output.read_text(encoding="utf-8"))
    trial = metadata["trials"][0]
    source_row = trial["source"]
    checks = {
        "schema": metadata["schema"] == "strata_xklt_sc_v2_single_block_encoder_v1",
        "block_length": int(metadata["parameters"]["block_length"]) == n,
        "distortion": float(metadata["parameters"]["test_channel_distortion"])
        == float(block["test_distortion"]),
        "eta": float(metadata["parameters"]["eta"]) == 0.25,
        "sc_seed": int(metadata["parameters"]["seed"]) == int(block["sc_seed_u32"]),
        "rht_seed": int(source_row["rht"]["seed_u64"]) == int(block["rht_seed_u64"]),
        "source_hash": source_row["block_bf16_sha256"] == block["staging_sha256"],
        "roundtrip_bits": bool(trial["arithmetic_roundtrip_bits_match"]),
        "causal_frequencies": bool(trial["causal_decoder_frequencies_match"]),
        "reconstruction_indices": bool(trial["reconstruction_indices_match"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"one-shot encoder audit failed block {ordinal}: {checks}")
    container = output.with_suffix(".polar.bin")
    return {
        "block_ordinal": ordinal,
        "encoder_invocations": 1,
        "metadata_relpath": str(output.relative_to(args.output_dir.resolve())).replace("\\", "/"),
        "metadata_sha256": common.sha256_file(output),
        "container_relpath": str(container.relative_to(args.output_dir.resolve())).replace("\\", "/"),
        "container_sha256": common.sha256_file(container),
        "logical_bits": int(trial["arithmetic_logical_bits"]),
        "normalized_relative_mse": float(trial["relative_mse"]),
        "block_rms_fp64": float(source_row["block_rms_fp64"]),
        "checks": checks,
    }


def legacy_parts(
    container: Path, metadata: dict, expected_container_sha256: str
) -> tuple[int, float, bytes]:
    raw = container.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_container_sha256:
        raise AssertionError(f"encoded container changed before pack: {container}")
    if len(raw) < 8:
        raise ValueError(f"truncated legacy staging container: {container}")
    logical, scale = struct.unpack("<If", raw[:8])
    payload = raw[8:]
    if len(payload) != (logical + 7) // 8:
        raise ValueError(f"noncanonical payload length: {container}")
    trailing = len(payload) * 8 - logical
    if trailing:
        tail_bits = np.unpackbits(
            np.frombuffer(payload[-1:], dtype=np.uint8), bitorder="big"
        )
        if np.any(tail_bits[8 - trailing :]):
            raise ValueError(f"nonzero low arithmetic tail padding: {container}")
    trial = metadata["trials"][0]
    if logical != int(trial["arithmetic_logical_bits"]):
        raise AssertionError("container/metadata logical length mismatch")
    if hashlib.sha256(payload).hexdigest() != trial["arithmetic_payload_sha256"]:
        raise AssertionError("container/metadata payload hash mismatch")
    if float(scale) != float(trial["source"]["decoder_scale_fp32"]):
        raise AssertionError("container/metadata FP32 scale mismatch")
    return int(logical), float(scale), payload


def pack_artifact(args: argparse.Namespace, manifest: dict, rows: list[dict]) -> dict:
    output_dir = args.output_dir.resolve()
    rows = sorted(rows, key=lambda row: int(row["block_ordinal"]))
    if [row["block_ordinal"] for row in rows] != list(range(14)):
        raise AssertionError("encoded block coverage is not exactly 0..13")
    directory = bytearray()
    streams = bytearray()
    logical_total = 0
    for block, row in zip(manifest["blocks"], rows):
        metadata_path = output_dir / row["metadata_relpath"]
        container_path = output_dir / row["container_relpath"]
        metadata_payload = metadata_path.read_bytes()
        if hashlib.sha256(metadata_payload).hexdigest() != row["metadata_sha256"]:
            raise AssertionError(f"encoded metadata changed before pack: {metadata_path}")
        metadata = json.loads(metadata_payload.decode("utf-8"))
        if metadata["trials"][0].get("literal_container_sha256") != row["container_sha256"]:
            raise AssertionError(
                f"metadata literal-container hash mismatch: {container_path}"
            )
        logical, _, payload = legacy_parts(
            container_path, metadata, row["container_sha256"]
        )
        rms_fp64 = float(metadata["trials"][0]["source"]["block_rms_fp64"])
        scale_fp16 = np.float16(rms_fp64)
        if not np.isfinite(scale_fp16) or scale_fp16 <= 0:
            raise ValueError(f"invalid FP16 decoder scale block {block['block_ordinal']}")
        directory.extend(struct.pack("<BeI", int(block["profile_id"]), float(scale_fp16), logical))
        streams.extend(payload)
        logical_total += logical
    if len(directory) != common.DIRECTORY_BYTES:
        raise AssertionError("directory length mismatch")
    if len(streams) > common.RESERVOIR_BYTES:
        failure = {
            "schema": "strata_xklt_sc_v2_one_shot_failure_v1",
            "status": "rate gate failed; no retry permitted",
            "payload_bytes": len(streams),
            "reservoir_bytes": common.RESERVOIR_BYTES,
            "overflow_bytes": len(streams) - common.RESERVOIR_BYTES,
            "encoder_invocations": 14,
        }
        common.write_json(output_dir / "ONE_SHOT_FAILURE.json", failure)
        raise RuntimeError("one-shot streams overflow the sealed reservoir; no retry")
    reservoir = bytes(streams) + bytes(common.RESERVOIR_BYTES - len(streams))
    header = (output_dir / "header.bin").read_bytes()
    route = (output_dir / "route.bin").read_bytes()
    labels = (output_dir / "labels_3bit.bin").read_bytes()
    common.validate_header(header, route, labels)
    artifact = header + route + labels + bytes(directory) + reservoir
    if len(artifact) != common.PHYSICAL_BYTES:
        raise AssertionError("physical artifact byte length mismatch")
    artifact_path = output_dir / "strata_xklt_sc_v2.bin"
    artifact_path.write_bytes(artifact)
    nominal_bits = int(manifest["allocation"]["nominal_profile_bits"])
    offset = logical_total - nominal_bits
    pooled_energy = float(sum(float(block["source_energy_fp64"]) for block in manifest["blocks"]))
    pooled_sse = float(
        sum(
            float(block["source_energy_fp64"]) * float(row["normalized_relative_mse"])
            for block, row in zip(manifest["blocks"], rows)
        )
    )
    return {
        "artifact_relpath": artifact_path.name,
        "artifact_sha256": common.sha256_file(artifact_path),
        "physical_bytes": len(artifact),
        "physical_bits": 8 * len(artifact),
        "physical_bpw": 8 * len(artifact) / common.WEIGHTS,
        "integer_2p15_gate_passed": 8 * len(artifact) <= common.INTEGER_CAP_BITS,
        "directory_bytes": len(directory),
        "logical_payload_bits": logical_total,
        "nominal_profile_bits": nominal_bits,
        "observed_logical_minus_nominal_bits": offset,
        "payload_byte_count": len(streams),
        "zero_reservoir_tail_bytes": common.RESERVOIR_BYTES - len(streams),
        "reservoir_fit": len(streams) <= common.RESERVOIR_BYTES,
        "encoder_side_staging_energy_relative_mse": pooled_sse / pooled_energy,
        "claim_boundary": "encoder-side staged-KLT BF16 metric; independent inverse-KLT source audit required",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--selection-lock", type=Path, required=True)
    ap.add_argument("--route", type=Path, required=True)
    ap.add_argument("--source-lock", type=Path, required=True)
    ap.add_argument("--source-root", type=Path)
    ap.add_argument(
        "--protocol-mode", choices=("development", "blind"), required=True
    )
    ap.add_argument(
        "--allow-development-rehearsal",
        action="store_true",
        help="explicitly allow a historical-lineage development rehearsal",
    )
    ap.add_argument("--codec-freeze", type=Path, required=True)
    ap.add_argument("--format", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, default=Path(__file__).with_name("polar_encoder.py"))
    ap.add_argument("--independent-auditor", type=Path, required=True)
    ap.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.workers < 1 or args.workers > 14:
        raise ValueError("workers must be in [1,14]")
    if args.protocol_mode == "development" and not args.allow_development_rehearsal:
        raise ValueError(
            "development mode is rehearsal-only and requires --allow-development-rehearsal"
        )
    if args.protocol_mode == "blind" and args.allow_development_rehearsal:
        raise ValueError("--allow-development-rehearsal is forbidden in blind mode")
    args.workspace = args.workspace.resolve()
    args.output_dir = args.output_dir.resolve()
    args.encoder = args.encoder.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"one-shot output already exists: {args.output_dir}")

    # This gate reads only public/frozen control files and executable hashes.
    # It runs before the emitter opens any finalized tensor payload.
    runtime_freeze = verify_blind_runtime_freeze(args)

    # This call completes all source-dependent decisions and writes the
    # allocation lock.  No encoder subprocess exists before it returns.
    emit_and_lock.emit_candidate(args)
    manifest, lock = verify_preencoding(args.output_dir, args.protocol_mode)
    intent = {
        "schema": "strata_xklt_sc_v2_one_shot_intent_v1",
        "status": "sealed_before_first_encoder_invocation",
        "allocation_lock_file_sha256": common.sha256_file(args.output_dir / "allocation.lock.json"),
        "allocation_lock_internal_sha256": lock["lock_sha256"],
        "manifest_sha256": common.sha256_file(args.output_dir / "preencoding_manifest.json"),
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "encoder_sha256": common.sha256_file(args.encoder),
        "protocol_mode": args.protocol_mode,
        "runtime_freeze": runtime_freeze,
        "workers": args.workers,
        "encoder_invocations_planned": 14,
        "retry_resume_or_adaptive_rate_change_allowed": False,
    }
    common.write_json(args.output_dir / "ONE_SHOT_INTENT.json", intent)
    (args.output_dir / "encoded").mkdir()
    verify_runtime_snapshot(runtime_freeze)
    invocation_rows: list[dict] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    encode_one, args, manifest, block, runtime_freeze
                ): int(block["block_ordinal"])
                for block in manifest["blocks"]
            }
            for future in concurrent.futures.as_completed(futures):
                invocation_rows.append(future.result())
    except Exception as exc:
        failure = {
            "schema": "strata_xklt_sc_v2_one_shot_failure_v1",
            "status": "encoder phase failed; no retry or resume",
            "error": repr(exc),
            "completed_encoder_rows": sorted(invocation_rows, key=lambda row: row["block_ordinal"]),
        }
        common.write_json(args.output_dir / "ONE_SHOT_FAILURE.json", failure)
        raise
    if sum(int(row["encoder_invocations"]) for row in invocation_rows) != 14:
        raise AssertionError("one-shot invocation count is not exactly fourteen")
    # Close the encode-to-pack TOCTOU window: every allocation-locked source
    # and asset is rehashed after all encoders finish and before packing.
    verify_runtime_snapshot(runtime_freeze)
    manifest_after, lock_after = verify_preencoding(args.output_dir, args.protocol_mode)
    if lock_after["lock_sha256"] != lock["lock_sha256"]:
        raise AssertionError("allocation lock changed during encoding")
    packed = pack_artifact(args, manifest_after, invocation_rows)
    summary = {
        "schema": "strata_xklt_sc_v2_one_shot_summary_v1",
        "status": "one-shot physical artifact complete",
        "protocol_mode": args.protocol_mode,
        "allocation_lock_file_sha256": common.sha256_file(args.output_dir / "allocation.lock.json"),
        "allocation_lock_internal_sha256": lock["lock_sha256"],
        "intent_sha256": common.sha256_file(args.output_dir / "ONE_SHOT_INTENT.json"),
        "encoder_invocations": 14,
        "retries": 0,
        "resumes": 0,
        "postencoding_profile_changes": 0,
        "encoded_blocks": sorted(invocation_rows, key=lambda row: row["block_ordinal"]),
        "physical": packed,
    }
    common.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
