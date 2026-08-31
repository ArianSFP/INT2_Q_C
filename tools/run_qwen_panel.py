#!/usr/bin/env python3
"""Run, serialize, independently decode, and aggregate a Qwen POLARIS panel.

The manifest fixes all source identities and both deterministic seeds before
any encoder result is opened.  Two deliberately distinct variants are
supported: the byte-frozen POLARIS-SC-v2 encoder (``exact``), and the same
polar core behind a zero-side-bit randomized Hadamard adapter (``rht``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


N = 1 << 18
PAYLOAD_CAPACITY_BITS_PER_BLOCK = 563_464
DIRECTORY_BITS_PER_BLOCK = 48
GAUSSIAN_LIMIT_AT_2P15 = 2.0 ** (-2.0 * 2.15)
GAUSSIAN_5PCT_CEILING = 1.05 * GAUSSIAN_LIMIT_AT_2P15
EXPECTED_IMPLEMENTATION_SHA256 = {
    "exact_encoder": "95cfd32e5d026f07ceffe90daa7f88ca5e62f9f90546dfe74fc37cf06854d9b8",
    "rht_encoder": "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "packer": "c5fda34242153365dac07b5990bcd1fa19f0ac98d2512d47c3c8e1ec2a81dde8",
    "unpacker": "cf7113c3fbc6340f0870dadcf7608739aa651f5706befa163b5d13516dac7e07",
    "decoder": "2e1e484bf8ba98d493cfda55d4b23e275267e097e08907f5a9c606ae7350c797",
    "decoder_map": "a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_block(path: Path, local_block_index: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(local_block_index * N * 2)
        data = handle.read(N * 2)
        if len(data) != N * 2:
            raise ValueError(f"short BF16 block in {path}: {len(data)} bytes")
        digest.update(data)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def implementation_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "encoder": sha256_file(
            args.exact_encoder if args.variant == "exact" else args.rht_encoder
        ),
        "packer": sha256_file(args.packer),
        "unpacker": sha256_file(args.unpacker),
        "decoder": sha256_file(args.decoder),
        "decoder_map": sha256_file(args.decoder_map),
    }


def assert_pinned_implementations(args: argparse.Namespace) -> dict[str, str]:
    observed = implementation_hashes(args)
    expected = {
        "encoder": EXPECTED_IMPLEMENTATION_SHA256[
            "exact_encoder" if args.variant == "exact" else "rht_encoder"
        ],
        "packer": EXPECTED_IMPLEMENTATION_SHA256["packer"],
        "unpacker": EXPECTED_IMPLEMENTATION_SHA256["unpacker"],
        "decoder": EXPECTED_IMPLEMENTATION_SHA256["decoder"],
        "decoder_map": EXPECTED_IMPLEMENTATION_SHA256["decoder_map"],
    }
    if observed != expected:
        raise AssertionError(f"implementation hash mismatch: {observed} != {expected}")
    return observed


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        tail = process.stdout[-4000:]
        raise RuntimeError(
            f"command failed ({process.returncode}) after "
            f"{time.perf_counter() - started:.1f}s: {command!r}\n{tail}"
        )


def resolve_source(workspace: Path, block: dict[str, Any]) -> Path:
    source = Path(block["source_path"])
    return source if source.is_absolute() else workspace / source


def validate_manifest(workspace: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if int(manifest["block_length"]) != N:
        raise ValueError("manifest block length is not 2^18")
    blocks = list(manifest["blocks"])
    if not blocks:
        raise ValueError("empty panel")
    ordinals = [int(row["ordinal"]) for row in blocks]
    if ordinals not in (list(range(len(blocks))), list(range(1, len(blocks) + 1))):
        raise ValueError("manifest ordinals must be contiguous, zero- or one-based")
    if len({str(row["id"]) for row in blocks}) != len(blocks):
        raise ValueError("duplicate manifest id")
    canonical_ids = {
        (str(row["tensor"]), int(row["canonical_block_index"])) for row in blocks
    }
    if len(canonical_ids) != len(blocks):
        raise ValueError("duplicate canonical tensor/block identity")
    if len({int(row["sc_seed_u32"]) for row in blocks}) != len(blocks):
        raise ValueError("SC seed collision in manifest")
    for panel_index, row in enumerate(blocks):
        # Physical reservoir order is always the JSON array order.  Keep the
        # manifest's human-facing zero/one-based ordinal as provenance only.
        row["_panel_index"] = panel_index
        source = resolve_source(workspace, row)
        observed = sha256_block(source, int(row["source_local_block_index"]))
        expected = str(row["source_bf16_sha256"])
        if observed != expected:
            raise AssertionError(f"source hash mismatch for {row['id']}: {observed} != {expected}")
        sc_seed = int(row["sc_seed_u32"])
        rht_seed = int(row["rht_seed_u64"])
        if not 1 <= sc_seed < (1 << 32) or not 0 <= rht_seed < (1 << 64):
            raise ValueError(f"seed out of range for {row['id']}")
        seed_material = (
            f"{manifest['revision']}:{row['tensor']}:"
            f"{int(row['canonical_block_index'])}"
        ).encode("utf-8")
        seed_digest = hashlib.sha256(seed_material).digest()
        expected_sc = int.from_bytes(seed_digest[0:4], "big")
        expected_rht = int.from_bytes(seed_digest[4:12], "big")
        if sc_seed != expected_sc or rht_seed != expected_rht:
            raise AssertionError(
                f"seed derivation mismatch for {row['id']}: "
                f"{(sc_seed, rht_seed)} != {(expected_sc, expected_rht)}"
            )
    return blocks


def encoder_command(
    args: argparse.Namespace,
    block: dict[str, Any],
    output_json: Path,
) -> list[str]:
    if args.variant == "exact":
        encoder = args.exact_encoder
    else:
        encoder = args.rht_encoder
    command = [
        str(args.python),
        str(encoder),
        "--polar-repo", str(args.polar_repo),
        "--block-length", str(N),
        "--trials", "1",
        "--sigma-source", "1.0",
        "--test-distortion", "0.05110",
        "--eta", "0.25",
        "--alphabet-size", "64",
        "--decision", "random",
        "--seed", str(int(block["sc_seed_u32"])),
        "--input-bf16", str(resolve_source(args.workspace, block)),
        "--input-block-start", str(int(block["source_local_block_index"])),
        "--emit-container-hex",
        "--output", str(output_json),
    ]
    if args.variant == "rht":
        command.extend(
            [
                "--canonical-source-id", str(block["tensor"]),
                "--canonical-block-index", str(int(block["canonical_block_index"])),
                "--apply-rht",
                "--rht-seed", str(int(block["rht_seed_u64"])),
            ]
        )
    return command


def encode_one(
    args: argparse.Namespace,
    workdir: Path,
    block: dict[str, Any],
) -> dict[str, Any]:
    panel_index = int(block["_panel_index"])
    stem = f"block_{panel_index:03d}"
    output_json = workdir / "encoded" / f"{stem}.encoder.json"
    output_bin = output_json.with_suffix(".polar.bin")
    log = workdir / "logs" / f"{stem}.encode.log"
    source = resolve_source(args.workspace, block)
    source_hash_before = sha256_block(source, int(block["source_local_block_index"]))
    if source_hash_before != block["source_bf16_sha256"]:
        raise AssertionError(f"pre-encode source mutation for {block['id']}")
    started = time.perf_counter()
    run_logged(encoder_command(args, block, output_json), args.workspace, log)
    source_hash_after = sha256_block(source, int(block["source_local_block_index"]))
    if source_hash_after != source_hash_before:
        raise AssertionError(f"source changed while encoding {block['id']}")
    metadata = json.loads(output_json.read_text(encoding="utf-8"))
    trial = metadata["trials"][0]
    parameters = metadata["parameters"]
    expected_parameters = {
        "block_length": N,
        "trials": 1,
        "sigma_source": 1.0,
        "test_channel_distortion": 0.0511,
        "eta": 0.25,
        "alphabet_size": 64,
        "decision": "random",
        "seed": int(block["sc_seed_u32"]),
    }
    for key, expected in expected_parameters.items():
        if parameters[key] != expected:
            raise AssertionError(
                f"encoder parameter mismatch for {block['id']} {key}: "
                f"{parameters[key]!r} != {expected!r}"
            )
    source_metadata = trial["source"]
    if int(source_metadata["block_index"]) != int(block["source_local_block_index"]):
        raise AssertionError(f"encoder source index mismatch for {block['id']}")
    if args.variant == "rht":
        if source_metadata.get("block_bf16_sha256") != source_hash_before:
            raise AssertionError(f"RHT encoder source hash mismatch for {block['id']}")
        rht = source_metadata.get("rht", {})
        if not (
            rht.get("enabled") is True
            and rht.get("mode") == "hadamard_rademacher_splitmix64"
            and rht.get("normalization") == "orthonormal"
            and int(rht.get("seed_u64", -1)) == int(block["rht_seed_u64"])
        ):
            raise AssertionError(f"RHT metadata mismatch for {block['id']}")
    if not (
        trial["arithmetic_roundtrip_bits_match"]
        and trial["causal_decoder_frequencies_match"]
        and trial["reconstruction_indices_match"]
    ):
        raise AssertionError(f"encoder self-audit failed for {block['id']}")
    return {
        "panel_index": panel_index,
        "seconds": time.perf_counter() - started,
        "resumed": False,
        "source_hash_before_after": source_hash_after,
    }


def decode_one(
    args: argparse.Namespace,
    workdir: Path,
    block: dict[str, Any],
) -> dict[str, Any]:
    panel_index = int(block["_panel_index"])
    stem = f"block_{panel_index:03d}"
    output = workdir / "decoded" / f"{stem}.decode.json"
    log = workdir / "logs" / f"{stem}.decode.log"
    command = [
        str(args.python),
        str(args.decoder),
        "--record", str(workdir / "unpacked" / f"block_{panel_index:06d}.variable-u32-fp16.bin"),
        "--metadata", str(workdir / "encoded" / f"{stem}.encoder.json"),
        "--map", str(args.decoder_map),
        "--source-bf16", str(resolve_source(args.workspace, block)),
        "--output", str(output),
    ]
    started = time.perf_counter()
    run_logged(command, args.workspace, log)
    decoded = json.loads(output.read_text(encoding="utf-8"))
    if not decoded["passed"]:
        raise AssertionError(f"independent decoder failed for {block['id']}")
    return {"panel_index": panel_index, "seconds": time.perf_counter() - started}


def execute_parallel(function: Any, args: argparse.Namespace, workdir: Path,
                     blocks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, args, workdir, row): row for row in blocks}
        for future in concurrent.futures.as_completed(futures):
            block = futures[future]
            result = future.result()
            rows.append(result)
            print(
                f"{function.__name__}: {int(block['_panel_index']) + 1}/{len(blocks)} "
                f"{block['id']} ({result['seconds']:.1f}s)",
                flush=True,
            )
    return sorted(rows, key=lambda row: int(row["panel_index"]))


def aggregate(
    args: argparse.Namespace,
    workdir: Path,
    manifest: dict[str, Any],
    manifest_hash: str,
    blocks: list[dict[str, Any]],
    encode_timings: list[dict[str, Any]],
    decode_timings: list[dict[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {"blocks": 0.0, "logical_bits": 0.0, "energy": 0.0, "sse": 0.0}
    )
    total_logical = 0
    total_energy = 0.0
    total_sse = 0.0
    for block in blocks:
        panel_index = int(block["_panel_index"])
        metadata_path = workdir / "encoded" / f"block_{panel_index:03d}.encoder.json"
        decode_path = workdir / "decoded" / f"block_{panel_index:03d}.decode.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        decoded = json.loads(decode_path.read_text(encoding="utf-8"))
        trial = metadata["trials"][0]
        observed_source_hash = str(decoded["source"]["bf16_sha256"])
        current_source_hash = sha256_block(
            resolve_source(args.workspace, block),
            int(block["source_local_block_index"]),
        )
        if not (
            observed_source_hash == current_source_hash
            == str(block["source_bf16_sha256"])
        ):
            raise AssertionError(f"aggregate source binding failed for {block['id']}")
        source_metadata = trial["source"]
        preconditioner = decoded.get("preconditioner")
        if args.variant == "exact":
            if preconditioner is not None:
                raise AssertionError(f"exact result has an RHT for {block['id']}")
        else:
            if not (
                source_metadata.get("canonical_source_id") == block["tensor"]
                and int(source_metadata.get("canonical_block_index", -1))
                == int(block["canonical_block_index"])
            ):
                raise AssertionError(f"RHT canonical identity mismatch for {block['id']}")
            if not isinstance(preconditioner, dict) or not (
                preconditioner.get("mode") == "hadamard_rademacher_splitmix64"
                and preconditioner.get("normalization") == "orthonormal"
                and int(preconditioner.get("seed_u64", -1))
                == int(block["rht_seed_u64"])
                and int(preconditioner.get("side_bits", -1)) == 0
            ):
                raise AssertionError(f"decoded RHT metadata mismatch for {block['id']}")
        logical = int(trial["arithmetic_logical_bits"])
        totals = decoded.get("aggregation", {})
        if totals:
            energy = float(totals["source_energy_sum_fp64"])
            sse = float(totals["fp16_sse_sum_fp64"])
        else:
            distortion = decoded["distortion"]
            absolute = float(distortion["fresh_fp16_scale_absolute_mse"])
            relative = float(distortion["fresh_fp16_scale_relative_mse"])
            sse = absolute * N
            energy = sse / relative
        role = str(block["role"])
        groups[role]["blocks"] += 1
        groups[role]["logical_bits"] += logical
        groups[role]["energy"] += energy
        groups[role]["sse"] += sse
        total_logical += logical
        total_energy += energy
        total_sse += sse
        rows.append(
            {
                "panel_index": panel_index,
                "manifest_ordinal": int(block["ordinal"]),
                "id": block["id"],
                "tensor": block["tensor"],
                "canonical_block_index": int(block["canonical_block_index"]),
                "role": role,
                "logical_bits": logical,
                "source_energy_sum_fp64": energy,
                "fp16_sse_sum_fp64": sse,
                "fp16_relative_mse": sse / energy,
                "encoder_normalized_relative_mse": float(trial["relative_mse"]),
                "decoder_passed": bool(decoded["passed"]),
                "source_bf16_sha256": block["source_bf16_sha256"],
                "reconstruction_fp64_sha256": totals.get(
                    "final_reconstruction_fp64_sha256"
                ),
            }
        )
    role_rows = {
        role: {
            "blocks": int(value["blocks"]),
            "mean_logical_bits": value["logical_bits"] / value["blocks"],
            "energy_weighted_relative_mse": value["sse"] / value["energy"],
            "source_energy_sum_fp64": value["energy"],
            "fp16_sse_sum_fp64": value["sse"],
        }
        for role, value in sorted(groups.items())
    }
    reservoir = workdir / "panel.plrsv2.bin"
    pack_audit = json.loads((workdir / "pack.audit.json").read_text(encoding="utf-8"))
    unpack_audit = json.loads((workdir / "unpack.audit.json").read_text(encoding="utf-8"))
    if pack_audit.get("passed") is not True:
        raise AssertionError("packer audit did not pass")
    if unpack_audit.get("validation") != "passed":
        raise AssertionError("independent unpacker validation did not pass")
    capacity = PAYLOAD_CAPACITY_BITS_PER_BLOCK * len(blocks)
    relative_mse = total_sse / total_energy
    exact_hashes = implementation_hashes(args)
    exact_hashes["reservoir"] = sha256_file(reservoir)
    physical_bits = reservoir.stat().st_size * 8
    expected_physical_bits = 768 + len(blocks) * (
        PAYLOAD_CAPACITY_BITS_PER_BLOCK + DIRECTORY_BITS_PER_BLOCK
    )
    if physical_bits != expected_physical_bits:
        raise AssertionError(
            f"fixed reservoir size {physical_bits} != {expected_physical_bits} bits"
        )
    physical_rate_pass = physical_bits * 20 <= 43 * N * len(blocks)
    all_decodes_passed = all(row["decoder_passed"] for row in rows)
    return {
        "status": "complete",
        "variant": args.variant,
        "claim_scope": "deterministic preregistered Qwen coverage panel; not a full-checkpoint census",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "checkpoint": manifest.get("checkpoint"),
        "revision": manifest.get("revision"),
        "block_length": N,
        "blocks": len(blocks),
        "aggregate": {
            "logical_bits_sum": total_logical,
            "logical_bits_mean": total_logical / len(blocks),
            "payload_capacity_bits": capacity,
            "payload_headroom_bits": capacity - total_logical,
            "payload_fits_global_reservoir": total_logical <= capacity,
            "fixed_rank2_allocation_bpw": (
                PAYLOAD_CAPACITY_BITS_PER_BLOCK + DIRECTORY_BITS_PER_BLOCK
            ) / N,
            "emitted_sample_reservoir_bytes": reservoir.stat().st_size,
            "emitted_sample_reservoir_bits": physical_bits,
            "expected_fixed_reservoir_bits": expected_physical_bits,
            "fixed_reservoir_size_exact": physical_bits == expected_physical_bits,
            "emitted_sample_reservoir_bpw_including_96_byte_header": (
                physical_bits / (len(blocks) * N)
            ),
            "physical_rate_gate_fraction": "physical_bits * 20 <= 43 * source_values",
            "physical_rate_at_most_2p15": physical_rate_pass,
            "source_energy_sum_fp64": total_energy,
            "fp16_sse_sum_fp64": total_sse,
            "energy_weighted_relative_mse": relative_mse,
            "gaussian_limit_mse_at_2p15": GAUSSIAN_LIMIT_AT_2P15,
            "gaussian_5pct_ceiling": GAUSSIAN_5PCT_CEILING,
            "relative_excess_over_gaussian": relative_mse / GAUSSIAN_LIMIT_AT_2P15 - 1.0,
            "passes_5pct_gaussian_mse_gate": relative_mse <= GAUSSIAN_5PCT_CEILING,
            "passes_joint_rate_and_mse_gate": (
                total_logical <= capacity
                and physical_rate_pass
                and relative_mse <= GAUSSIAN_5PCT_CEILING
                and all_decodes_passed
            ),
            "all_independent_decodes_passed": all_decodes_passed,
        },
        "by_role": role_rows,
        "blocks_detail": rows,
        "serialization": {
            "pack_audit": pack_audit,
            "unpack_audit": unpack_audit,
        },
        "hashes": exact_hashes,
        "claim_boundary_by_variant": (
            "Frozen POLARIS-SC-v2, unpreconditioned."
            if args.variant == "exact"
            else (
                "POLARIS-SC-v2 plus a deterministic zero-side-bit RHT adapter; "
                "a same-manifest exact run is required for comparative attribution."
            )
        ),
        "execution": {
            "python": str(args.python),
            "workers": args.workers,
            "wall_seconds": wall_seconds,
            "encode_timings": encode_timings,
            "decode_timings": decode_timings,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--variant", choices=("exact", "rht"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    parser.add_argument("--exact-encoder", type=Path, default=Path("agent_root_polar_lattice_gate.py"))
    parser.add_argument("--rht-encoder", type=Path, default=Path("agent_polaris_qwen_rht_encoder.py"))
    parser.add_argument("--packer", type=Path, default=Path("agent_polaris_reservoir_pack_v2.py"))
    parser.add_argument("--unpacker", type=Path, default=Path("agent_polaris_reservoir_unpack_v2.py"))
    parser.add_argument("--decoder", type=Path, default=Path("agent_novel_qwen_reservoir_decode.py"))
    parser.add_argument("--decoder-map", type=Path, default=Path("agent_polaris_sc_v1_decoder_map.npz"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.manifest = args.manifest.resolve()
    args.workdir = args.workdir.resolve()
    if args.resume:
        raise ValueError(
            "--resume is disabled: an unbound partial artifact must never be relabelled"
        )
    for name in ("exact_encoder", "rht_encoder", "packer", "unpacker", "decoder", "decoder_map"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else args.workspace / path)
    pinned_hashes = assert_pinned_implementations(args)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    for folder in ("encoded", "decoded", "logs"):
        (args.workdir / folder).mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    blocks = validate_manifest(args.workspace, manifest)
    print(f"Frozen manifest {manifest_hash}; {len(blocks)} source blocks validated", flush=True)
    started = time.perf_counter()
    if args.aggregate_only:
        previous_summary_path = args.workdir / "summary.json"
        if not previous_summary_path.is_file():
            raise FileNotFoundError("aggregate-only requires the prior summary.json")
        previous = json.loads(previous_summary_path.read_text(encoding="utf-8"))
        lock_path = args.workdir / "execution_lock.json"
        if lock_path.is_file():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if not (
                lock.get("variant") == args.variant
                and lock.get("manifest_sha256") == manifest_hash
                and lock.get("implementation_hashes") == pinned_hashes
            ):
                raise AssertionError("execution lock does not bind aggregate-only inputs")
        else:
            prior_hashes = previous.get("hashes", {})
            prior_order = [
                (row.get("id"), row.get("tensor"), int(row.get("canonical_block_index", -1)))
                for row in previous.get("blocks_detail", [])
            ]
            expected_order = [
                (row["id"], row["tensor"], int(row["canonical_block_index"]))
                for row in blocks
            ]
            if not (
                previous.get("variant") == args.variant
                and previous.get("manifest_sha256") == manifest_hash
                and all(prior_hashes.get(key) == value for key, value in pinned_hashes.items())
                and prior_hashes.get("reservoir")
                == sha256_file(args.workdir / "panel.plrsv2.bin")
                and prior_order == expected_order
            ):
                raise AssertionError(
                    "legacy result lacks a matching lock and prior summary binding"
                )
        summary = aggregate(
            args,
            args.workdir,
            manifest,
            manifest_hash,
            blocks,
            previous.get("execution", {}).get("encode_timings", []),
            previous.get("execution", {}).get("decode_timings", []),
            float(previous.get("execution", {}).get("wall_seconds", 0.0)),
        )
        summary["execution"]["reaggregation_seconds"] = time.perf_counter() - started
        summary["execution"]["reaggregation_note"] = (
            "Post-run fail-closed source/rate revalidation; no encode, pack, or decode rerun."
        )
        write_json(args.workdir / "summary.json", summary)
        print(json.dumps(summary["aggregate"], indent=2, sort_keys=True), flush=True)
        raise SystemExit(0 if summary["aggregate"]["passes_joint_rate_and_mse_gate"] else 2)

    execution_lock = {
        "variant": args.variant,
        "manifest_sha256": manifest_hash,
        "manifest": str(args.manifest),
        "seed_derivation_revalidated": True,
        "implementation_hashes": pinned_hashes,
    }
    lock_path = args.workdir / "execution_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"refusing to reuse existing execution lock: {lock_path}")
    write_json(lock_path, execution_lock)
    encode_timings = execute_parallel(encode_one, args, args.workdir, blocks, args.workers)
    inputs = [
        str(args.workdir / "encoded" / f"block_{int(row['_panel_index']):03d}.encoder.polar.bin")
        for row in blocks
    ]
    run_logged(
        [
            str(args.python), str(args.packer), "--inputs", *inputs,
            "--output", str(args.workdir / "panel.plrsv2.bin"),
            "--audit", str(args.workdir / "pack.audit.json"),
        ],
        args.workspace,
        args.workdir / "logs" / "pack.log",
    )
    run_logged(
        [
            str(args.python), str(args.unpacker),
            "--input", str(args.workdir / "panel.plrsv2.bin"),
            "--output-dir", str(args.workdir / "unpacked"),
            "--audit", str(args.workdir / "unpack.audit.json"),
        ],
        args.workspace,
        args.workdir / "logs" / "unpack.log",
    )
    decode_timings = execute_parallel(decode_one, args, args.workdir, blocks, args.workers)
    summary = aggregate(
        args, args.workdir, manifest, manifest_hash, blocks,
        encode_timings, decode_timings, time.perf_counter() - started,
    )
    write_json(args.workdir / "summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True), flush=True)
    if not summary["aggregate"]["passes_joint_rate_and_mse_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
