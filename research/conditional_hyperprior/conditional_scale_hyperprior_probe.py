#!/usr/bin/env python3
"""Leakage-safe expert-local conditional scale-hyperprior probe.

The probe has two deliberately separated stages.

``train`` reads only auxiliary Qwen expert Up/Down pairs.  It excludes the
union of all layer IDs and expert IDs in the pinned target source lock, scores
a preregistered grid with leave-expert-AND-layer-out cross-validation, freezes
one candidate, and serializes a canonical Huffman model.

``evaluate`` checks the freeze and model hashes before opening any pinned
source.  It evaluates all six Gate/Up/Down triplets once, physically writes an
expert-local hyperlatent stream for each triplet, decodes every stream, and
reports Gaussian cross-entropy gain minus every serialized model/side byte.

This is an information/RD screen, not an operational quantizer MSE result.
The deliberately strong baseline gets exact per-expert role x STRATA FP16
scales for free in the NLL comparison; the candidate nevertheless serializes
those scales so its physical rate is complete.  Thus a failure is a useful
early kill, while a success still requires integration and source-domain MSE.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROWS = 768
COLS = 2048
GROUPS_PER_ROLE = ROWS
STRATA = 8
MODEL_MAGIC = b"CHYPV1\0\0"
SIDE_MAGIC = b"CHYPS1\0\0"
MODEL_HEADER = struct.Struct("<8sIHHfHH32s32s")
SIDE_HEADER = struct.Struct("<8sHHIQ32s32s")
TARGET_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)
DEV_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(up|down)_proj\.weight\.bf16\.bin$"
)
CHUNKS = (32, 64, 128, 256, 512, 1024, 2048)
CODE_BITS = (2, 3, 4, 5, 6)
LOG2_STEPS = (0.125, 0.25, 0.5)
TARGET_NET_GAIN_BPW = -0.5 * math.log2(0.8)
PROMOTION_MARGIN_BPW = 0.02


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    expected = int(np.prod(shape)) * 2
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes")
    words = np.memmap(path, dtype="<u2", mode="r", shape=shape)
    result = (np.asarray(words, dtype=np.uint32) << np.uint32(16)).view(np.float32)
    return np.asarray(result, dtype=np.float64)


def exact_klt(up: np.ndarray, down_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    a = float(np.sum(up * up, dtype=np.float64))
    b = float(np.sum(down_t * down_t, dtype=np.float64))
    c = float(np.sum(up * down_t, dtype=np.float64))
    theta = 0.5 * math.atan2(2.0 * c, a - b)
    co, si = math.cos(theta), math.sin(theta)
    return co * up + si * down_t, -si * up + co * down_t, theta


def strata_labels(roles: np.ndarray) -> np.ndarray:
    # roles: R x 768 x 2048.  Stable ordinal tie breaking matches STRATA.
    energy = np.sum(roles * roles, axis=2, dtype=np.float64).reshape(-1)
    order = np.lexsort((np.arange(energy.size, dtype=np.int64), energy))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=np.int64)
    labels = np.minimum(STRATA - 1, rank * STRATA // energy.size)
    return labels.reshape(roles.shape[0], ROWS).astype(np.uint8)


def fp16_context_scales(roles: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scales = np.empty((roles.shape[0], STRATA), dtype=np.float16)
    all_group_energy = np.sum(roles * roles, axis=2, dtype=np.float64)
    for role in range(roles.shape[0]):
        group_energy = all_group_energy[role]
        for stratum in range(STRATA):
            select = labels[role] == stratum
            if int(select.sum()):
                numerator = float(group_energy[select].sum())
                denominator = int(select.sum()) * COLS
            else:
                # A strongly separated KLT component can leave a role absent
                # from an extreme global stratum.  Use the pooled stratum as a
                # deterministic decoder-visible backoff for that unused cell.
                pooled = labels == stratum
                numerator = float(all_group_energy[pooled].sum())
                denominator = int(pooled.sum()) * COLS
            value = math.sqrt(numerator / denominator)
            rounded = np.float16(value)
            if not np.isfinite(rounded) or rounded <= 0:
                raise ValueError("invalid FP16 context scale")
            scales[role, stratum] = rounded
    return scales


@dataclass(frozen=True, order=True)
class Candidate:
    chunk: int
    bits: int
    step: float

    @property
    def symbols(self) -> int:
        return 1 << self.bits

    def as_dict(self) -> dict[str, int | float]:
        return {"chunk": self.chunk, "bits": self.bits, "log2_step": self.step}


@dataclass
class PairStats:
    layer: int
    expert: int
    weights: int
    signal_gain_bits: float
    histogram: np.ndarray  # 3 x 8 x symbols; role 0 reserved for Gate


def chunk_statistics(
    roles: np.ndarray, labels: np.ndarray, scales: np.ndarray, chunk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    role_count = roles.shape[0]
    per_group = COLS // chunk
    values = roles.reshape(role_count, ROWS, per_group, chunk)
    tile_energy = np.sum(values * values, axis=3, dtype=np.float64)
    base = np.asarray(scales, dtype=np.float64)
    base_for_tile = np.empty((role_count, ROWS, per_group), dtype=np.float64)
    contexts = np.empty((role_count, ROWS, per_group), dtype=np.uint8)
    for role in range(role_count):
        base_for_tile[role] = base[role, labels[role]][:, None]
        # Contexts 0..23 map Gate/K0/K1 x stratum.  Two-role development
        # arrays correspond to K0/K1 and therefore start at role index 1.
        output_role = role if role_count == 3 else role + 1
        contexts[role] = output_role * STRATA + labels[role, :, None]
    return tile_energy, base_for_tile, contexts


def candidate_stats_from_chunk(
    tile_energy: np.ndarray,
    base_for_tile: np.ndarray,
    contexts: np.ndarray,
    candidate: Candidate,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    role_count = tile_energy.shape[0]
    tile_rms = np.sqrt(tile_energy / candidate.chunk)
    log_ratio = np.log2(np.maximum(tile_rms / base_for_tile, np.finfo(float).tiny))
    q_lo = -(1 << (candidate.bits - 1))
    q_hi = (1 << (candidate.bits - 1)) - 1
    q_signed = np.clip(np.rint(log_ratio / candidate.step), q_lo, q_hi).astype(np.int16)
    symbols = (q_signed - q_lo).astype(np.uint8)
    conditional_scale = base_for_tile * np.exp2(q_signed.astype(np.float64) * candidate.step)
    base_var = base_for_tile * base_for_tile
    conditional_var = conditional_scale * conditional_scale
    n = candidate.chunk
    inv = 1.0 / (2.0 * math.log(2.0))
    # Constant 0.5*log2(2*pi) cancels.
    baseline_nll = 0.5 * n * np.log2(base_var) + tile_energy * inv / base_var
    conditional_nll = (
        0.5 * n * np.log2(conditional_var) + tile_energy * inv / conditional_var
    )
    signal_gain = float(np.sum(baseline_nll - conditional_nll, dtype=np.float64))
    hist = np.zeros((3, STRATA, candidate.symbols), dtype=np.int64)
    for output_role in range(3):
        for stratum in range(STRATA):
            selected = symbols[contexts == output_role * STRATA + stratum].reshape(-1)
            hist[output_role, stratum] = np.bincount(
                selected, minlength=candidate.symbols
            )
    return signal_gain, hist, contexts.reshape(-1), symbols.reshape(-1)


def candidate_stats(
    roles: np.ndarray, labels: np.ndarray, scales: np.ndarray, candidate: Candidate
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    return candidate_stats_from_chunk(
        *chunk_statistics(roles, labels, scales, candidate.chunk), candidate
    )


def huffman_lengths(counts: np.ndarray) -> np.ndarray:
    # Add-one smoothing makes every symbol encodable on held-out data.
    smooth = np.asarray(counts, dtype=np.int64) + 1
    size = int(smooth.size)
    if size == 1:
        return np.ones(1, dtype=np.uint8)
    heap: list[tuple[int, int, tuple]] = []
    serial = 0
    for symbol, count in enumerate(smooth.tolist()):
        heap.append((int(count), serial, (symbol,)))
        serial += 1
    heapq.heapify(heap)
    lengths = np.zeros(size, dtype=np.uint8)
    while len(heap) > 1:
        wa, _, a = heapq.heappop(heap)
        wb, _, b = heapq.heappop(heap)
        for symbol in a:
            lengths[symbol] += 1
        for symbol in b:
            lengths[symbol] += 1
        merged = a + b
        heapq.heappush(heap, (wa + wb, serial, merged))
        serial += 1
    if int(lengths.max()) > 63:
        raise ValueError("Huffman length exceeds format bound")
    return lengths


def model_lengths(histogram: np.ndarray) -> np.ndarray:
    result = np.empty_like(histogram, dtype=np.uint8)
    pooled = histogram[1] + histogram[2]
    for stratum in range(STRATA):
        result[0, stratum] = huffman_lengths(pooled[stratum])
        result[1, stratum] = huffman_lengths(histogram[1, stratum])
        result[2, stratum] = huffman_lengths(histogram[2, stratum])
    return result


def canonical_codes(lengths: np.ndarray) -> list[tuple[int, int]]:
    ordered = sorted((int(length), symbol) for symbol, length in enumerate(lengths))
    if not ordered or ordered[0][0] <= 0:
        raise ValueError("bad Huffman lengths")
    result: list[tuple[int, int]] = [(0, 0)] * len(ordered)
    code = 0
    previous = ordered[0][0]
    for length, symbol in ordered:
        code <<= length - previous
        if code >= (1 << length):
            raise ValueError("oversubscribed Huffman lengths")
        result[symbol] = (code, length)
        code += 1
        previous = length
    return result


def encode_huffman(symbols: np.ndarray, contexts: np.ndarray, lengths: np.ndarray) -> tuple[bytes, int]:
    tables = [canonical_codes(lengths.reshape(24, -1)[i]) for i in range(24)]
    output = bytearray()
    accumulator = 0
    pending = 0
    total = 0
    for symbol, context in zip(symbols.tolist(), contexts.tolist(), strict=True):
        code, width = tables[int(context)][int(symbol)]
        accumulator = (accumulator << width) | code
        pending += width
        total += width
        while pending >= 8:
            pending -= 8
            output.append((accumulator >> pending) & 0xFF)
            accumulator &= (1 << pending) - 1 if pending else 0
    if pending:
        output.append((accumulator << (8 - pending)) & 0xFF)
    return bytes(output), total


def decode_huffman(
    payload: bytes, bit_length: int, contexts: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    reverse: list[dict[tuple[int, int], int]] = []
    max_lengths: list[int] = []
    for row in lengths.reshape(24, -1):
        codes = canonical_codes(row)
        reverse.append({(width, code): symbol for symbol, (code, width) in enumerate(codes)})
        max_lengths.append(max(width for _, width in codes))
    output = np.empty(contexts.size, dtype=np.uint8)
    cursor = 0
    for i, context in enumerate(contexts.tolist()):
        code = 0
        table = reverse[int(context)]
        found = False
        for width in range(1, max_lengths[int(context)] + 1):
            if cursor >= bit_length:
                raise ValueError("truncated Huffman stream")
            bit = (payload[cursor >> 3] >> (7 - (cursor & 7))) & 1
            cursor += 1
            code = (code << 1) | bit
            symbol = table.get((width, code))
            if symbol is not None:
                output[i] = symbol
                found = True
                break
        if not found:
            raise ValueError("invalid Huffman code")
    if cursor != bit_length:
        raise ValueError(f"unused Huffman bits: {bit_length-cursor}")
    return output


def target_identity(lock_path: Path) -> tuple[set[int], set[int], list[dict]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matrices = lock["matrices"]
    layers = {int(row["layer"]) for row in matrices}
    experts = {int(row["expert"]) for row in matrices}
    if len(matrices) != 18 or len(layers) != 6 or len(experts) != 6:
        raise ValueError("expected six disjoint target triplets")
    return layers, experts, matrices


def dev_pairs(dev_dir: Path, excluded_layers: set[int], excluded_experts: set[int]):
    pairs: dict[tuple[int, int], dict[str, Path]] = {}
    for path in sorted(dev_dir.glob("*.bf16.bin")):
        match = DEV_RE.search(path.name)
        if not match:
            continue
        layer, expert, role = int(match.group(1)), int(match.group(2)), match.group(3)
        if layer in excluded_layers or expert in excluded_experts:
            continue
        pairs.setdefault((layer, expert), {})[role] = path
    return [(key, value) for key, value in sorted(pairs.items()) if set(value) == {"up", "down"}]


def candidate_grid() -> list[Candidate]:
    return [Candidate(chunk, bits, step) for chunk in CHUNKS for bits in CODE_BITS for step in LOG2_STEPS]


def load_dev_roles(paths: dict[str, Path]) -> tuple[np.ndarray, float]:
    up = bf16(paths["up"], (ROWS, COLS))
    down_t = bf16(paths["down"], (COLS, ROWS)).T.copy()
    k0, k1, theta = exact_klt(up, down_t)
    return np.stack((k0, k1)), theta


def crossfit_score(candidate: Candidate, rows: list[PairStats], target_weights: int) -> dict:
    total_hist = sum((row.histogram for row in rows), np.zeros_like(rows[0].histogram))
    signal = 0.0
    side_bits = 0
    fold_rows = []
    for heldout in rows:
        excluded = [
            row for row in rows if row.layer == heldout.layer or row.expert == heldout.expert
        ]
        training_hist = total_hist - sum(
            (row.histogram for row in excluded), np.zeros_like(total_hist)
        )
        lengths = model_lengths(training_hist)
        bits = int(np.sum(heldout.histogram * lengths, dtype=np.int64))
        physical = ((bits + 7) // 8) * 8 + SIDE_HEADER.size * 8 + 32 * 8
        signal += heldout.signal_gain_bits
        side_bits += physical
        fold_rows.append(
            {
                "layer": heldout.layer,
                "expert": heldout.expert,
                "signal_gain_bpw": heldout.signal_gain_bits / heldout.weights,
                "physical_hyperlatent_bpw": physical / heldout.weights,
                "net_gain_bpw": (heldout.signal_gain_bits - physical) / heldout.weights,
                "training_pairs_after_layer_or_expert_exclusion": len(rows) - len(excluded),
            }
        )
    final_lengths = model_lengths(total_hist)
    model_bytes = MODEL_HEADER.size + final_lengths.size + 32
    weights = sum(row.weights for row in rows)
    # Model is a single cached shared asset, but every byte is charged over the
    # six-triplet target panel when choosing the candidate.
    model_bpw_target = model_bytes * 8 / target_weights
    net = (signal - side_bits) / weights - model_bpw_target
    return {
        "candidate": candidate.as_dict(),
        "development_weights": weights,
        "crossfit_signal_gain_bpw": signal / weights,
        "crossfit_physical_hyperlatent_bpw": side_bits / weights,
        "shared_model_bytes": model_bytes,
        "shared_model_bpw_charged_over_target_panel": model_bpw_target,
        "crossfit_net_gain_bpw": net,
        "projected_F_multiplier": 2.0 ** (-2.0 * net),
        "fold_min_net_gain_bpw": min(row["net_gain_bpw"] for row in fold_rows),
        "fold_max_net_gain_bpw": max(row["net_gain_bpw"] for row in fold_rows),
        "folds": fold_rows,
        "lengths": final_lengths,
    }


def write_model(
    path: Path,
    candidate: Candidate,
    lengths: np.ndarray,
    target_lock_sha: str,
    dev_manifest_sha: str,
) -> str:
    prefix = MODEL_HEADER.pack(
        MODEL_MAGIC,
        1,
        candidate.chunk,
        candidate.bits,
        candidate.step,
        24,
        candidate.symbols,
        bytes.fromhex(target_lock_sha),
        bytes.fromhex(dev_manifest_sha),
    ) + lengths.astype(np.uint8).tobytes(order="C")
    raw = prefix + hashlib.sha256(prefix).digest()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def read_model(path: Path) -> tuple[Candidate, np.ndarray, dict]:
    raw = path.read_bytes()
    if len(raw) < MODEL_HEADER.size + 32:
        raise ValueError("short model")
    fields = MODEL_HEADER.unpack_from(raw)
    magic, version, chunk, bits, step, contexts, symbols, lock_sha, dev_sha = fields
    if magic != MODEL_MAGIC or version != 1 or contexts != 24 or symbols != 1 << bits:
        raise ValueError("model header mismatch")
    body_end = MODEL_HEADER.size + contexts * symbols
    if len(raw) != body_end + 32 or hashlib.sha256(raw[:body_end]).digest() != raw[body_end:]:
        raise ValueError("model integrity mismatch")
    lengths = np.frombuffer(raw[MODEL_HEADER.size:body_end], dtype=np.uint8).copy().reshape(3, 8, symbols)
    for row in lengths.reshape(24, symbols):
        canonical_codes(row)
    return Candidate(chunk, bits, float(step)), lengths, {
        "target_lock_sha256": lock_sha.hex(),
        "development_manifest_sha256": dev_sha.hex(),
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "model_bytes": len(raw),
    }


def train(args: argparse.Namespace) -> None:
    target_lock = args.target_lock.resolve()
    target_lock_sha = sha256_file(target_lock)
    layers, experts, matrices = target_identity(target_lock)
    target_weights = sum(int(row.get("nvalues", np.prod(row["shape"]))) for row in matrices)
    pairs = dev_pairs(args.dev_dir.resolve(), layers, experts)
    if len(pairs) < 20:
        raise ValueError(f"too few clean development pairs: {len(pairs)}")
    source_manifest = []
    grids = candidate_grid()
    stats_by_candidate: dict[Candidate, list[PairStats]] = {candidate: [] for candidate in grids}
    for ordinal, ((layer, expert), paths) in enumerate(pairs):
        roles, theta = load_dev_roles(paths)
        labels = strata_labels(roles)
        scales = fp16_context_scales(roles, labels)
        source_manifest.append(
            {
                "layer": layer,
                "expert": expert,
                "up": {"path": str(paths["up"].resolve()), "sha256": sha256_file(paths["up"])},
                "down": {"path": str(paths["down"].resolve()), "sha256": sha256_file(paths["down"])},
                "klt_theta_fp64": theta,
            }
        )
        for chunk in CHUNKS:
            precomputed = chunk_statistics(roles, labels, scales, chunk)
            for bits in CODE_BITS:
                for step in LOG2_STEPS:
                    candidate = Candidate(chunk, bits, step)
                    signal, hist, _, _ = candidate_stats_from_chunk(
                        *precomputed, candidate
                    )
                    stats_by_candidate[candidate].append(
                        PairStats(layer, expert, int(roles.size), signal, hist)
                    )
        del roles, labels, scales
        print(f"development pair {ordinal+1}/{len(pairs)} L{layer} E{expert}", flush=True)
    manifest_core = {
        "schema": "conditional_hyperprior_development_sources_v1",
        "target_lock_sha256": target_lock_sha,
        "excluded_layers": sorted(layers),
        "excluded_experts": sorted(experts),
        "source_pairs": source_manifest,
    }
    dev_manifest_sha = hashlib.sha256(canonical_json_bytes(manifest_core)).hexdigest()
    scores = []
    for index, candidate in enumerate(grids):
        score = crossfit_score(candidate, stats_by_candidate[candidate], target_weights)
        scores.append(score)
        print(
            f"candidate {index+1}/{len(grids)} {candidate} net={score['crossfit_net_gain_bpw']:.9f}",
            flush=True,
        )
    scores.sort(key=lambda row: (-row["crossfit_net_gain_bpw"], row["candidate"]["chunk"], row["candidate"]["bits"], row["candidate"]["log2_step"]))
    winner = scores[0]
    candidate = Candidate(
        int(winner["candidate"]["chunk"]),
        int(winner["candidate"]["bits"]),
        float(winner["candidate"]["log2_step"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "conditional_hyperprior_model.bin"
    model_sha = write_model(
        model_path, candidate, winner.pop("lengths"), target_lock_sha, dev_manifest_sha
    )
    for row in scores[1:]:
        row.pop("lengths", None)
        # Full per-pair rows are retained only for the winner to keep the
        # report compact; every candidate retains aggregate LEO+LLO metrics.
        row.pop("folds", None)
    script_sha = sha256_file(Path(__file__).resolve())
    freeze = {
        "schema": "conditional_scale_hyperprior_freeze_v1",
        "status": "frozen_before_pinned_source_evaluation",
        "strict_ptq": True,
        "target_lock": {"path": str(target_lock), "sha256": target_lock_sha},
        "target_panel_weights": target_weights,
        "target_layers_excluded_from_training": sorted(layers),
        "target_experts_excluded_from_training": sorted(experts),
        "development_pair_count": len(pairs),
        "development_manifest_sha256": dev_manifest_sha,
        "development_source_manifest": source_manifest,
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": script_sha},
        "grid": {
            "chunks": list(CHUNKS),
            "code_bits": list(CODE_BITS),
            "log2_steps": list(LOG2_STEPS),
            "candidates": len(grids),
        },
        "selection_protocol": "maximize leave-expert-AND-layer-out cross-fitted net gain; ties by smaller chunk,bits,step",
        "selected": winner,
        "all_candidate_aggregates": scores,
        "model": {"path": model_path.name, "sha256": model_sha, "bytes": model_path.stat().st_size},
        "gates": {
            "required_net_gain_bpw": TARGET_NET_GAIN_BPW,
            "promotion_net_gain_bpw_with_margin": TARGET_NET_GAIN_BPW + PROMOTION_MARGIN_BPW,
            "early_kill": "kill if pinned physically charged net gain < required_net_gain_bpw",
            "promotion": "integrate only if pinned net gain >= required+0.02 and every expert net gain is positive",
        },
    }
    freeze_path = args.output_dir / "conditional_hyperprior_freeze.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    summary = {
        "freeze": str(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "model": str(model_path),
        "model_sha256": model_sha,
        "selected": winner,
    }
    print(json.dumps(summary, indent=2), flush=True)


def matrix_path(source_root: Path, row: dict) -> Path:
    rel = row.get("output_relpath") or row.get("source_relpath")
    if not rel:
        raise ValueError(f"source row lacks relative path: {row.get('tensor')}")
    return source_root / rel


def target_triplets(lock_path: Path, source_root: Path) -> list[tuple[int, int, dict[str, tuple[Path, dict]]]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, int], dict[str, tuple[Path, dict]]] = {}
    for row in lock["matrices"]:
        key = (int(row["layer"]), int(row["expert"]))
        grouped.setdefault(key, {})[str(row["role"])] = (matrix_path(source_root, row), row)
    result = []
    for key, roles in sorted(grouped.items()):
        if set(roles) != {"gate", "up", "down"}:
            raise ValueError(f"incomplete target triplet {key}")
        result.append((key[0], key[1], roles))
    return result


def load_target_roles(roles: dict[str, tuple[Path, dict]]) -> tuple[np.ndarray, float, list[dict]]:
    checked = []
    arrays = {}
    for role in ("gate", "up", "down"):
        path, row = roles[role]
        expected_hash = row.get("source_bf16_sha256")
        actual_hash = sha256_file(path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"source hash mismatch: {path}")
        shape = tuple(int(value) for value in row["shape"])
        arrays[role] = bf16(path, shape)
        checked.append({"role": role, "path": str(path.resolve()), "sha256": actual_hash, "bytes": path.stat().st_size})
    k0, k1, theta = exact_klt(arrays["up"], arrays["down"].T.copy())
    return np.stack((arrays["gate"], k0, k1)), theta, checked


def write_side(
    path: Path,
    layer: int,
    expert: int,
    model_sha: str,
    scales: np.ndarray,
    contexts: np.ndarray,
    symbols: np.ndarray,
    lengths: np.ndarray,
) -> dict:
    payload, bit_length = encode_huffman(symbols, contexts, lengths)
    context_sha = hashlib.sha256(contexts.astype(np.uint8).tobytes()).digest()
    header = SIDE_HEADER.pack(
        SIDE_MAGIC,
        layer,
        expert,
        int(symbols.size),
        bit_length,
        bytes.fromhex(model_sha),
        context_sha,
    )
    scale_bytes = scales.astype("<f2").tobytes(order="C")
    prefix = header + scale_bytes + payload
    raw = prefix + hashlib.sha256(prefix).digest()
    path.write_bytes(raw)
    # Independent parse/decode round trip from serialized bytes.
    parsed = path.read_bytes()
    fields = SIDE_HEADER.unpack_from(parsed)
    if fields[0] != SIDE_MAGIC or fields[1] != layer or fields[2] != expert:
        raise ValueError("side header roundtrip mismatch")
    if fields[5].hex() != model_sha or fields[6] != context_sha:
        raise ValueError("side binding mismatch")
    scale_end = SIDE_HEADER.size + scales.size * 2
    payload_end = len(parsed) - 32
    if hashlib.sha256(parsed[:payload_end]).digest() != parsed[payload_end:]:
        raise ValueError("side integrity mismatch")
    decoded = decode_huffman(parsed[scale_end:payload_end], int(fields[4]), contexts, lengths)
    if not np.array_equal(decoded, symbols):
        raise ValueError("side symbol roundtrip mismatch")
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tile_count": int(symbols.size),
        "logical_huffman_bits": bit_length,
        "payload_bytes": len(payload),
        "fp16_context_scale_bytes": len(scale_bytes),
        "roundtrip": True,
    }


def evaluate(args: argparse.Namespace) -> None:
    freeze_path = args.freeze.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    model_path = args.model.resolve()
    model_sha = sha256_file(model_path)
    if model_sha != freeze["model"]["sha256"]:
        raise ValueError("freeze/model hash mismatch")
    lock_path = args.target_lock.resolve()
    lock_sha = sha256_file(lock_path)
    if lock_sha != freeze["target_lock"]["sha256"]:
        raise ValueError("freeze/target-lock hash mismatch")
    candidate, lengths, model_meta = read_model(model_path)
    if model_meta["target_lock_sha256"] != lock_sha:
        raise ValueError("model/target-lock hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experts = []
    total_signal = 0.0
    total_weights = 0
    total_side_bytes = model_meta["model_bytes"]
    for ordinal, (layer, expert, role_paths) in enumerate(
        target_triplets(lock_path, args.source_root.resolve())
    ):
        roles, theta, checked = load_target_roles(role_paths)
        labels = strata_labels(roles)
        scales = fp16_context_scales(roles, labels)
        signal, _, contexts, symbols = candidate_stats(roles, labels, scales, candidate)
        side_path = args.output_dir / f"L{layer:02d}_E{expert:03d}.chyps"
        side = write_side(
            side_path, layer, expert, model_sha, scales, contexts, symbols, lengths
        )
        weights = int(roles.size)
        net = (signal - side["bytes"] * 8) / weights
        experts.append(
            {
                "ordinal": ordinal,
                "layer": layer,
                "expert": expert,
                "weights": weights,
                "source_files": checked,
                "exact_klt_theta_fp64": theta,
                "signal_gain_bits": signal,
                "signal_gain_bpw": signal / weights,
                "expert_local_side": side,
                "expert_local_side_bpw": side["bytes"] * 8 / weights,
                "net_gain_before_shared_model_bpw": net,
            }
        )
        total_signal += signal
        total_weights += weights
        total_side_bytes += side["bytes"]
        print(
            f"pinned expert {ordinal+1}/6 L{layer} E{expert} gross={signal/weights:.9f} local_net={net:.9f}",
            flush=True,
        )
    net_gain = (total_signal - total_side_bytes * 8) / total_weights
    shared_model_bpw = model_meta["model_bytes"] * 8 / total_weights
    local_side_bytes = total_side_bytes - model_meta["model_bytes"]
    per_expert_cold_model_fraction = model_meta["model_bytes"] / 6
    result = {
        "schema": "conditional_scale_hyperprior_pinned_evaluation_v1",
        "status": "complete_single_pinned_panel_evaluation",
        "strict_ptq": True,
        "claim_boundary": "Gaussian conditional cross-entropy/RD information probe; not an operational quantizer or source-domain reconstruction-MSE result",
        "bindings": {
            "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
            "model": {"path": str(model_path), **model_meta},
            "target_lock": {"path": str(lock_path), "sha256": lock_sha},
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
        },
        "protocol": {
            "candidate": candidate.as_dict(),
            "development_pairs": freeze["development_pair_count"],
            "all_target_layers_excluded_from_development": freeze["target_layers_excluded_from_training"],
            "all_target_experts_excluded_from_development": freeze["target_experts_excluded_from_training"],
            "selection": freeze["selection_protocol"],
            "baseline": "free per-expert role x 8-STRATA FP16 Gaussian scale (deliberately stronger than current block scale baseline)",
            "conditional": "serialized residual log2-RMS code per fixed-size tile; frozen canonical Huffman table conditioned on role and STRATA label",
            "all_serialized_bytes_charged": True,
            "expert_payloads_independently_framed": True,
            "shared_model_cacheable": True,
        },
        "aggregate": {
            "weights": total_weights,
            "gross_conditional_signal_gain_bits": total_signal,
            "gross_conditional_signal_gain_bpw": total_signal / total_weights,
            "expert_local_side_bytes": local_side_bytes,
            "expert_local_side_bpw": local_side_bytes * 8 / total_weights,
            "shared_model_bytes": model_meta["model_bytes"],
            "shared_model_bpw": shared_model_bpw,
            "all_in_side_bytes": total_side_bytes,
            "all_in_side_bpw": total_side_bytes * 8 / total_weights,
            "physically_charged_net_information_gain_bpw": net_gain,
            "required_net_gain_bpw": TARGET_NET_GAIN_BPW,
            "shortfall_bpw": TARGET_NET_GAIN_BPW - net_gain,
            "F_multiplier_from_net_gain": 2.0 ** (-2.0 * net_gain),
            "passes_required_gain": net_gain >= TARGET_NET_GAIN_BPW,
            "passes_promotion_margin": net_gain >= TARGET_NET_GAIN_BPW + PROMOTION_MARGIN_BPW,
            "all_experts_positive_before_shared_model": all(row["net_gain_before_shared_model_bpw"] > 0 for row in experts),
        },
        "read_locality": {
            "expert_local_side_streams": 6,
            "side_streams_roundtrip_verified": all(row["expert_local_side"]["roundtrip"] for row in experts),
            "steady_state_read_amplification_if_co_located_with_each_expert_payload": 1.0,
            "shared_model_bytes_read_once_then_cached": model_meta["model_bytes"],
            "shared_model_bytes_amortized_per_expert_for_cold_accounting": per_expert_cold_model_fraction,
            "note": "the hyperprior introduces no cross-expert payload dependency",
        },
        "experts": experts,
        "decision": (
            "PROMOTE_TO_OPERATIONAL_CODEC_PROBE"
            if net_gain >= TARGET_NET_GAIN_BPW + PROMOTION_MARGIN_BPW
            and all(row["net_gain_before_shared_model_bpw"] > 0 for row in experts)
            else "KILL_CONDITIONAL_SCALE_HYPERPRIOR_BRANCH"
        ),
    }
    output = args.output_dir / "conditional_hyperprior_pinned_result.json"
    output.write_bytes(canonical_json_bytes(result))
    print(json.dumps({"output": str(output), "output_sha256": sha256_file(output), "aggregate": result["aggregate"], "decision": result["decision"]}, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dev-dir", type=Path, required=True)
    train_parser.add_argument("--target-lock", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.set_defaults(func=train)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--freeze", type=Path, required=True)
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--target-lock", type=Path, required=True)
    eval_parser.add_argument("--source-root", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.set_defaults(func=evaluate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
