#!/usr/bin/env python3
"""Independent causal decoder and original-BF16 scorer for the locality fork."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_expert_local_codec import common
import strata_v2_klt_mixed_independent_auditor_v1 as frozen_auditor


def parse_container(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) != common.PHYSICAL_BYTES:
        raise ValueError("container physical byte count mismatch")
    header = raw[: common.HEADER_BYTES]
    route_begin = common.HEADER_BYTES
    labels_begin = route_begin + common.ROUTE_BYTES
    directory_begin = labels_begin + common.LABEL_BYTES
    reservoir_begin = directory_begin + common.DIRECTORY_BYTES
    route = raw[route_begin:labels_begin]
    labels_packed = raw[labels_begin:directory_begin]
    common.validate_header(header, route, labels_packed)
    labels = common.unpack_labels(labels_packed)
    profiles = bytes(int(row["profile_id"]) for row in plan["blocks"])
    directory = []
    cursor = reservoir_begin
    logical_total = 0
    for ordinal in range(common.BLOCKS):
        q, scale, logical_bits = struct.unpack_from(
            "<BeI", raw, directory_begin + ordinal * common.DIRECTORY_RECORD_BYTES
        )
        if q != profiles[ordinal]:
            raise ValueError(f"directory profile mismatch block {ordinal}")
        payload_bytes = (logical_bits + 7) // 8
        end = cursor + payload_bytes
        if end > len(raw):
            raise ValueError("payload exceeds physical file")
        payload = raw[cursor:end]
        padding = payload_bytes * 8 - logical_bits
        if padding and payload[-1] & ((1 << padding) - 1):
            raise ValueError(f"nonzero low padding bits block {ordinal}")
        expected_scale = np.float16(
            math.sqrt(float(plan["blocks"][ordinal]["source_energy_fp64"]) / (1 << common.BLOCK_LOG2[ordinal]))
        )
        if struct.pack("<e", scale) != struct.pack("<e", float(expected_scale)):
            raise ValueError(f"directory scale mismatch block {ordinal}")
        sc_seed, rht_seed, digest = common.derive_seeds(
            header, route, labels_packed, profiles, ordinal
        )
        sealed = plan["blocks"][ordinal]
        if (
            sc_seed != int(sealed["sc_seed_u32"])
            or rht_seed != int(sealed["rht_seed_u64"])
            or digest != sealed["seed_digest_sha256"]
        ):
            raise ValueError(f"seed mismatch block {ordinal}")
        directory.append(
            {
                "block_ordinal": ordinal,
                "profile_q": q,
                "decoder_scale": float(scale),
                "logical_bits": logical_bits,
                "payload_bytes": payload_bytes,
                "file_byte_begin": cursor,
                "file_byte_end_exclusive": end,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "sc_seed_u32": sc_seed,
                "rht_seed_u64": rht_seed,
            }
        )
        logical_total += logical_bits
        cursor = end
    if any(raw[cursor:]):
        raise ValueError("nonzero terminal reservoir fill")
    if len(raw) * 8 / common.WEIGHTS != 2.5:
        raise ValueError("physical rate mismatch")
    return {
        "raw": raw,
        "header": header,
        "route": route,
        "labels": labels,
        "labels_packed": labels_packed,
        "directory": directory,
        "used_payload_bytes": cursor - reservoir_begin,
        "zero_tail_bytes": len(raw) - cursor,
        "logical_payload_bits": logical_total,
    }


def page_union_bytes(ranges: list[tuple[int, int]], page_bytes: int = 4096) -> int:
    pages: set[int] = set()
    for begin, end in ranges:
        if not 0 <= begin <= end <= common.PHYSICAL_BYTES:
            raise ValueError("read range outside physical container")
        if begin != end:
            pages.update(range(begin // page_bytes, (end - 1) // page_bytes + 1))
    return len(pages) * page_bytes


def recompute_read_amplification(
    directory: list[dict[str, Any]], route: bytes
) -> dict[str, Any]:
    equal_share = common.PHYSICAL_BYTES / common.EXPERTS
    prefix = (
        common.HEADER_BYTES
        + common.ROUTE_BYTES
        + common.LABEL_BYTES
        + common.DIRECTORY_BYTES
    )
    route_rows = common.parse_route(route)
    experts: list[dict[str, Any]] = []
    for expert_ordinal in range(common.EXPERTS):
        required = common.expert_required_blocks(expert_ordinal)
        selected = [directory[index] for index in required]
        payload_bytes = sum(int(row["payload_bytes"]) for row in selected)
        ranges = [(0, prefix)] + [
            (int(row["file_byte_begin"]), int(row["file_byte_end_exclusive"]))
            for row in selected
        ]
        cold_bytes = prefix + payload_bytes
        page_bytes = page_union_bytes(ranges)
        experts.append(
            {
                "expert_ordinal": expert_ordinal,
                "layer": int(route_rows[3 * expert_ordinal]["layer"]),
                "expert": int(route_rows[3 * expert_ordinal]["expert"]),
                "required_blocks": list(required),
                "payload_bytes": payload_bytes,
                "cold_bytes": cold_bytes,
                "cold_amplification_vs_equal_physical_share": cold_bytes / equal_share,
                "page_4k_union_bytes": page_bytes,
                "page_4k_amplification_vs_equal_physical_share": page_bytes / equal_share,
            }
        )
    return {
        "definition": "cold bytes fetched divided by one-sixth of physical container bytes",
        "equal_physical_share_bytes": equal_share,
        "experts": experts,
        "max_cold": max(row["cold_amplification_vs_equal_physical_share"] for row in experts),
        "max_4k": max(row["page_4k_amplification_vs_equal_physical_share"] for row in experts),
        "passes_below_2x": max(
            row["page_4k_amplification_vs_equal_physical_share"] for row in experts
        )
        < 2.0,
    }


def decode_block_worker(arguments: tuple[str, str, dict[str, Any], int]) -> dict[str, Any]:
    container_path_text, output_path_text, row, logn = arguments
    raw = Path(container_path_text).read_bytes()
    payload = raw[int(row["file_byte_begin"]) : int(row["file_byte_end_exclusive"])]
    n = 1 << logn
    profile = frozen_auditor.profile_parameters(int(row["profile_q"]), 0.25)
    reverse = frozen_auditor.bit_reverse_indices(n)
    layers = frozen_auditor.sc_layers(n)
    flags = frozen_auditor.bec_freeze_flags(n, profile["capacities"], reverse)
    arithmetic = frozen_auditor.ArithmeticBinaryDecoder(
        payload, 0, int(row["logical_bits"])
    )
    alphabet = 0.25 * np.arange(-31, 33, dtype=np.float64)
    weights = np.exp(-0.5 * (alphabet / float(profile["sigma_reconstruction"])) ** 2)
    previous = np.zeros(n, dtype=np.int16)
    selected_chunks: list[np.ndarray] = []
    frequency_chunks: list[np.ndarray] = []
    frequency_hash = hashlib.sha256()
    for level_index, flag in enumerate(flags):
        level = level_index + 1
        frozen_rng = np.random.default_rng(int(row["sc_seed_u32"]) + 1_000_003 * level)
        frozen_external = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior = frozen_auditor.leaf_prior_ratios(weights, previous, level)
        x_bit, frequencies, selected = frozen_auditor.decode_sc_level(
            prior, flag, frozen_external, reverse, layers, arithmetic
        )
        previous += (1 << level_index) * x_bit.astype(np.int16)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected_chunks.append(selected)
        frequency_chunks.append(frequencies)
    transformed = alphabet[previous] * float(row["decoder_scale"])
    reconstructed = frozen_auditor.inverse_signed_rht(
        transformed, int(row["rht_seed_u64"]), "cupy"
    )
    output_path = Path(output_path_text)
    np.asarray(reconstructed, dtype="<f8").tofile(output_path)
    all_selected = np.concatenate(selected_chunks)
    all_frequencies = np.concatenate(frequency_chunks)
    canonical_payload, canonical_bits = frozen_auditor.arithmetic_encode_binary(
        all_selected, all_frequencies
    )
    if canonical_bits != int(row["logical_bits"]) or canonical_payload != payload:
        raise AssertionError("canonical arithmetic re-encoding mismatch")
    return {
        "block_ordinal": int(row["block_ordinal"]),
        "values": n,
        "profile_q": int(row["profile_q"]),
        "logical_bits": int(row["logical_bits"]),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "frequency_u16_sha256": frequency_hash.hexdigest(),
        "reconstruction_indices_i16_sha256": hashlib.sha256(
            previous.astype("<i2", copy=False).tobytes()
        ).hexdigest(),
        "decoded_f64_sha256": common.sha256_file(output_path),
        "canonical_reencode_matches": True,
        "arithmetic_bits_read_including_zero_extension": arithmetic.cursor,
        "output_path": str(output_path),
    }


def bf16_matrix(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size != shape[0] * shape[1]:
        raise ValueError(f"source shape mismatch: {path}")
    return common.bf16_to_fp32(words).astype(np.float64).reshape(shape)


def score_sources(
    post: np.memmap,
    plan: dict[str, Any],
    header: bytes,
    route: bytes,
) -> dict[str, Any]:
    coefficients = struct.unpack_from("<12f", header, 32)
    source_root = Path(plan["source_root"])
    route_rows = common.parse_route(route)
    total_sse = 0.0
    total_energy = 0.0
    matrix_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    for expert_ordinal in range(common.EXPERTS):
        base = expert_ordinal * common.GROUPS_PER_EXPERT
        gate_hat = np.asarray(post[base : base + 768])
        z0 = np.asarray(post[base + 768 : base + 1536])
        z1 = np.asarray(post[base + 1536 : base + 2304])
        cosine = float(coefficients[2 * expert_ordinal])
        sine = float(coefficients[2 * expert_ordinal + 1])
        norm2 = cosine * cosine + sine * sine
        up_hat = (cosine * z0 - sine * z1) / norm2
        down_hat = (sine * z0 + cosine * z1) / norm2
        reconstructions = (gate_hat, up_hat, down_hat)
        expert_sse = 0.0
        expert_energy = 0.0
        for local_role, reconstruction in enumerate(reconstructions):
            matrix_ordinal = 3 * expert_ordinal + local_role
            source_row = plan["sources"][matrix_ordinal]
            if int(source_row.get("matrix_ordinal", -1)) != matrix_ordinal:
                raise ValueError(f"source ordinal mismatch during scoring matrix {matrix_ordinal}")
            path = source_root / source_row["source_relpath"]
            if common.sha256_file(path) != source_row["source_bf16_sha256"]:
                raise ValueError(f"source hash mismatch during scoring matrix {matrix_ordinal}")
            role = str(route_rows[matrix_ordinal]["role"])
            shape = (2048, 768) if role == "down" else (768, 2048)
            expected_axis = "column" if role == "down" else "row"
            expected_tensor = (
                f"model.layers.{route_rows[matrix_ordinal]['layer']}.mlp.experts."
                f"{route_rows[matrix_ordinal]['expert']}.{role}_proj.weight"
            )
            if (
                source_row.get("role") != role
                or source_row.get("axis") != expected_axis
                or list(source_row.get("shape", [])) != list(shape)
                or source_row.get("tensor") != expected_tensor
            ):
                raise ValueError(f"source/route binding mismatch matrix {matrix_ordinal}")
            source = bf16_matrix(path, shape)
            natural = source.T if role == "down" else source
            error = reconstruction - natural
            sse = float(np.sum(error * error, dtype=np.float64))
            energy = float(np.sum(natural * natural, dtype=np.float64))
            expert_sse += sse
            expert_energy += energy
            total_sse += sse
            total_energy += energy
            matrix_rows.append(
                {
                    "matrix_ordinal": matrix_ordinal,
                    "tensor": source_row["tensor"],
                    "role": role,
                    "axis": expected_axis,
                    "shape": list(shape),
                    "source_relpath": source_row["source_relpath"],
                    "source_bf16_sha256": source_row["source_bf16_sha256"],
                    "sse_fp64": sse,
                    "source_energy_fp64": energy,
                    "relative_mse": sse / energy,
                }
            )
        expert_rows.append(
            {
                "expert_ordinal": expert_ordinal,
                "layer": int(route_rows[3 * expert_ordinal]["layer"]),
                "expert": int(route_rows[3 * expert_ordinal]["expert"]),
                "sse_fp64": expert_sse,
                "source_energy_fp64": expert_energy,
                "relative_mse": expert_sse / expert_energy,
            }
        )
    return {
        "sse_sum_fp64": total_sse,
        "source_energy_sum_fp64": total_energy,
        "energy_weighted_relative_mse": total_sse / total_energy,
        "matrices": matrix_rows,
        "experts": expert_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan_dir = args.plan_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"audit output must not exist: {output_dir}")
    output_dir.mkdir(parents=True)
    decoded_dir = output_dir / "decoded"
    decoded_dir.mkdir()
    plan = json.loads((plan_dir / "plan.lock.json").read_text(encoding="utf-8"))
    if not common.verify_internal_seal(plan):
        raise ValueError("plan seal mismatch")
    summary = json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("plan_lock_sha256") != plan.get("lock_sha256"):
        raise ValueError("summary/plan lock binding mismatch")
    container_path = plan_dir / summary["artifact"]["relpath"]
    if common.sha256_file(container_path) != summary["artifact"]["sha256"]:
        raise ValueError("summary/container hash mismatch")
    parsed = parse_container(container_path, plan)
    read_amplification = recompute_read_amplification(
        parsed["directory"], parsed["route"]
    )
    summary_read = summary.get("read_amplification", {})
    if (
        summary_read.get("max_cold") != read_amplification["max_cold"]
        or summary_read.get("max_4k") != read_amplification["max_4k"]
        or summary_read.get("experts") != read_amplification["experts"]
    ):
        raise ValueError("summary read ledger differs from physical container")

    tasks = []
    for ordinal, row in enumerate(parsed["directory"]):
        output_path = decoded_dir / f"block_{ordinal:02d}.f64.bin"
        tasks.append((str(container_path), str(output_path), row, common.BLOCK_LOG2[ordinal]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        decoded = list(executor.map(decode_block_worker, tasks))
    decoded.sort(key=lambda row: int(row["block_ordinal"]))
    if len(decoded) != common.BLOCKS or not all(row["canonical_reencode_matches"] for row in decoded):
        raise AssertionError("independent block audit coverage mismatch")

    labels = parsed["labels"]
    block_ordinals = common.expected_block_group_ordinals(labels)
    post_path = output_dir / "post_klt_canonical_groups.f64.bin"
    post = np.memmap(
        post_path,
        dtype="<f8",
        mode="w+",
        shape=(common.GROUPS, common.GROUP_VALUES),
    )
    coverage = np.zeros(common.GROUPS, dtype=np.uint8)
    for row, ordinals in zip(decoded, block_ordinals, strict=True):
        values = np.memmap(
            row["output_path"], dtype="<f8", mode="r", shape=(int(row["values"]),)
        ).reshape(-1, common.GROUP_VALUES)
        post[ordinals] = values
        coverage[ordinals] += 1
    post.flush()
    if not np.all(coverage == 1):
        raise AssertionError("decoded reconstruction does not cover every group once")

    source_score = score_sources(post, plan, parsed["header"], parsed["route"])
    mse = float(source_score["energy_weighted_relative_mse"])
    rate = common.PHYSICAL_BITS / common.WEIGHTS
    gain = common.gaussian_gain(mse, rate)
    max_read = float(read_amplification["max_4k"])
    milestone = (
        rate <= 2.5
        and max_read < 2.0
        and mse <= common.CURRENT_SOURCE_MSE
        and all(row["canonical_reencode_matches"] for row in decoded)
    )
    final_gate = milestone and gain >= common.GAUSSIAN_GAIN_TARGET
    report = {
        "schema": "strata_expert_affine_independent_audit_v1",
        "status": "passed" if milestone else "failed",
        "bindings": {
            "plan_lock_sha256": plan["lock_sha256"],
            "sources_canonical_sha256": hashlib.sha256(
                common.canonical_bytes(plan["sources"])
            ).hexdigest(),
        },
        "container": {
            "path": str(container_path),
            "sha256": common.sha256_file(container_path),
            "physical_bytes": container_path.stat().st_size,
            "physical_bpw": rate,
            "logical_payload_bits": parsed["logical_payload_bits"],
            "used_payload_bytes": parsed["used_payload_bytes"],
            "zero_tail_bytes": parsed["zero_tail_bytes"],
        },
        "decode": {
            "blocks": decoded,
            "decoded_blocks": len(decoded),
            "canonical_reencode_all_match": True,
            "every_group_once": True,
            "post_klt_sha256": common.sha256_file(post_path),
        },
        "source_score": source_score,
        "rate_relative": {
            "physical_bpw": rate,
            "gaussian_assumed_mse": common.gaussian_limit(rate),
            "mse_below_gaussian_fraction": gain,
            "target_fraction": common.GAUSSIAN_GAIN_TARGET,
            "target_mse_at_same_rate": (1.0 - common.GAUSSIAN_GAIN_TARGET)
            * common.gaussian_limit(rate),
            "passes_20_percent_below_same_rate_gaussian": final_gate,
        },
        "read_amplification": read_amplification,
        "milestone_gate": {
            "current_mse_ceiling": common.CURRENT_SOURCE_MSE,
            "source_mse_passed": mse <= common.CURRENT_SOURCE_MSE,
            "max_4k_read_amplification": max_read,
            "read_below_2x_passed": max_read < 2.0,
            "rate_at_or_below_2p5_passed": rate <= 2.5,
            "passed": milestone,
        },
    }
    common.write_json(output_dir / "independent_audit.json", report)
    print(json.dumps({
        "status": report["status"],
        "mse": mse,
        "physical_bpw": rate,
        "gaussian_gain_fraction": gain,
        "max_4k_read_amplification": max_read,
        "milestone_passed": milestone,
        "final_rate_relative_gate_passed": final_gate,
    }, indent=2))
    if not milestone:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
