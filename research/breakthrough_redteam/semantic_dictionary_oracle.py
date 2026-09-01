#!/usr/bin/env python3
"""Blocked one-atom cosine oracle against existing Qwen weight dictionaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import cupy as cp
import numpy as np


NAME_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<role>down_proj|gate_proj|up_proj)\.weight\.bf16\.bin"
)
ROWS = 768
COLS = 2048
WEIGHTS = ROWS * COLS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bf16_words_to_float(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words).astype(np.uint32) << 16).view(np.float32)


def read_source(path: Path, role: str) -> np.ndarray:
    words = np.memmap(path, dtype="<u2", mode="r")
    if words.size != WEIGHTS:
        raise ValueError(f"bad source size: {path}")
    values = bf16_words_to_float(words)
    if role == "down_proj":
        return values.reshape(COLS, ROWS).T.copy()
    return values.reshape(ROWS, COLS).copy()


def select_rows(matrix: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(matrix.shape[0], size=count, replace=False))
    return matrix[indices].copy(), indices


def search_dictionary(
    dictionary_path: Path,
    targets: np.ndarray,
    *,
    block_rows: int,
    random_control: bool,
    random_seed: int,
) -> dict:
    words = np.memmap(dictionary_path, dtype="<u2", mode="r")
    if words.size % COLS:
        raise ValueError(f"dictionary is not row-major d={COLS}: {dictionary_path}")
    dictionary_rows = words.size // COLS
    target_norms = np.linalg.norm(targets.astype(np.float64), axis=1)
    normalized_targets = targets / np.maximum(target_norms[:, None], 1e-30)
    target_gpu = cp.asarray(normalized_targets.astype(np.float32))
    best = cp.zeros(targets.shape[0], dtype=cp.float32)
    best_index = cp.full(targets.shape[0], -1, dtype=cp.int64)
    generator = cp.random.RandomState(random_seed)
    started = time.perf_counter()

    for start in range(0, dictionary_rows, block_rows):
        stop = min(start + block_rows, dictionary_rows)
        if random_control:
            block = generator.standard_normal((stop - start, COLS), dtype=cp.float32)
        else:
            host_words = np.asarray(words[start * COLS : stop * COLS])
            block = cp.asarray(bf16_words_to_float(host_words).reshape(stop - start, COLS))
        norms = cp.sqrt(cp.sum(block * block, axis=1, dtype=cp.float64)).astype(cp.float32)
        block /= cp.maximum(norms[:, None], cp.float32(1e-30))
        cosine = cp.abs(block @ target_gpu.T)
        local_index = cp.argmax(cosine, axis=0)
        local_value = cosine[local_index, cp.arange(targets.shape[0])]
        improved = local_value > best
        best = cp.where(improved, local_value, best)
        best_index = cp.where(improved, local_index.astype(cp.int64) + start, best_index)
        del block, norms, cosine, local_index, local_value, improved

    best_host = cp.asnumpy(best).astype(np.float64)
    index_host = cp.asnumpy(best_index)
    energy_weights = target_norms * target_norms
    explained = float(np.sum(energy_weights * best_host * best_host) / np.sum(energy_weights))
    unique, counts = np.unique(index_host, return_counts=True)
    return {
        "dictionary_rows": int(dictionary_rows),
        "target_vectors": int(targets.shape[0]),
        "pooled_explained_energy": explained,
        "free_predictor_s_bpw": -0.5 * math.log2(max(1.0 - explained, 1e-300)),
        "cosine_mean": float(np.mean(best_host)),
        "cosine_median": float(np.median(best_host)),
        "cosine_p95": float(np.quantile(best_host, 0.95)),
        "cosine_p99": float(np.quantile(best_host, 0.99)),
        "cosine_max": float(np.max(best_host)),
        "unique_selected_atoms": int(unique.size),
        "maximum_atom_reuse": int(np.max(counts)),
        "selected_atom_indices_sha256": hashlib.sha256(index_host.astype("<i8").tobytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--rows-per-role", type=int, default=256)
    parser.add_argument("--block-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.rows_per_role != 256 or args.block_rows != 4096:
        raise ValueError("sampling and block sizes are protocol-frozen")

    selected = []
    bindings = []
    for path in sorted(args.source_dir.glob("*.bf16.bin")):
        match = NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        identity = f"layer{match.group('layer')}_expert{match.group('expert')}"
        if identity != args.identity:
            continue
        role = match.group("role")
        matrix = read_source(path, role)
        role_seed = 0x5EAA0000 + {"down_proj": 1, "gate_proj": 2, "up_proj": 3}[role]
        rows, indices = select_rows(matrix, args.rows_per_role, role_seed)
        selected.append(rows)
        bindings.append(
            {
                "tensor": path.name,
                "role": role,
                "source_sha256": sha256_file(path),
                "selected_rows": indices.tolist(),
            }
        )
    if len(selected) != 3:
        raise ValueError(f"expected three roles for {args.identity}")
    targets = np.concatenate(selected, axis=0)

    dictionaries = []
    for ordinal, (name, path) in enumerate(
        (("embedding", args.embedding), ("attention_q", args.attention))
    ):
        actual = search_dictionary(
            path,
            targets,
            block_rows=args.block_rows,
            random_control=False,
            random_seed=0xD1C70000 + ordinal,
        )
        random = search_dictionary(
            path,
            targets,
            block_rows=args.block_rows,
            random_control=True,
            random_seed=0xD1C70000 + ordinal,
        )
        dictionary_rows = int(actual["dictionary_rows"])
        index_bits = math.ceil(math.log2(dictionary_rows))
        side_bpw_fp16_coefficient = (index_bits + 16) / COLS
        required_increment = 0.15287192093
        required_energy = 1.0 - 2.0 ** (-2.0 * (required_increment + side_bpw_fp16_coefficient))
        random_extreme_energy = 2.0 * math.log(2.0 * dictionary_rows) / COLS
        dictionaries.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "index_bits_fixed": index_bits,
                "side_bpw_with_fp16_coefficient": side_bpw_fp16_coefficient,
                "required_explained_energy_to_close_increment": required_energy,
                "required_absolute_cosine_if_uniform": math.sqrt(required_energy),
                "iid_random_extreme_prediction_energy": random_extreme_energy,
                "iid_random_extreme_prediction_cosine": math.sqrt(random_extreme_energy),
                "actual": actual,
                "matched_random_dictionary": random,
            }
        )
        cp.get_default_memory_pool().free_all_blocks()

    result = {
        "schema": "qwen-existing-semantic-dictionary-cosine-oracle-v1",
        "claim_boundary": (
            "One atom per 2048-vector, target-aware nearest absolute cosine, and an "
            "uncharged exact real coefficient for distortion. Rate/read ledgers separately "
            "charge an index plus FP16 coefficient. This is a sampled opportunity screen."
        ),
        "configuration": {
            "identity": args.identity,
            "rows_per_role": args.rows_per_role,
            "target_vectors": int(targets.shape[0]),
            "vector_dimension": COLS,
            "block_rows": args.block_rows,
        },
        "bindings": bindings,
        "dictionaries": dictionaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
