#!/usr/bin/env python3
"""Strict-PTQ polar-lattice Gaussian source-coding gate.

This is an independent Python/CuPy reproduction scaffold for the public
PolarLatticeQuantization construction (Liu, Shi, Ling).  It deliberately
starts with a synthetic Gaussian source and reports both distortion and the
decoder-visible prior codelength of the polar decisions.  No model weights,
calibration data, retraining, QAT, or task loss are used.

The first gate reuses the published Tal--Vardy reliability order at D=0.20
while allowing the test-channel distortion to move.  It is therefore a
screen, not yet the final normative codec.  A passing screen must later be
reconstructed with reliability tables generated for the exact target and an
actual arithmetic-coded bitstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cupy as cp
import numpy as np
from scipy.io import loadmat


def bit_reverse_indices(n: int) -> np.ndarray:
    k = int(math.log2(n))
    if 1 << k != n:
        raise ValueError("block length must be a power of two")
    x = np.arange(n, dtype=np.uint32)
    out = np.zeros(n, dtype=np.uint32)
    for _ in range(k):
        out = (out << 1) | (x & 1)
        x >>= 1
    return out.astype(np.int64)


def sc_layers(n: int) -> np.ndarray:
    """Zero-based translation of PolarSCDecodePrepare.m."""
    k = int(math.log2(n))
    out = np.ones(n + 1, dtype=np.int32)
    out[0] = k
    for i_one in range(2, n + 1):
        end_layer = 1
        idx = i_one
        while idx % 2 == 1:
            end_layer += 1
            idx = (idx + 1) // 2
        out[i_one - 1] = end_layer
    return out


def polar_transform(bits: np.ndarray) -> np.ndarray:
    """Exact vectorized translation of Encoder4Polar.m."""
    x = np.asarray(bits, dtype=np.uint8).copy()
    n = x.size
    step = 1
    while step < n:
        view = x.reshape(-1, 2 * step)
        view[:, :step] ^= view[:, step:]
        step *= 2
    return x


@dataclass
class SCResult:
    external_u: np.ndarray
    internal_u: np.ndarray
    selected_nll_bits: float
    selected_count: int
    selected_bits: np.ndarray
    selected_freq1_u16: np.ndarray


def sc_encode_ratio(
    leaf_lr: np.ndarray,
    freeze_flag: np.ndarray,
    frozen_external: np.ndarray,
    reverse: np.ndarray,
    layers: np.ndarray,
    *,
    rng: np.random.Generator,
    decision: str,
    forced_internal: np.ndarray | None = None,
    score_selected: bool = False,
) -> SCResult:
    """Successive-cancellation encoder in likelihood-ratio form.

    The register updates are a direct zero-based port of the authors'
    PolarNewLossySCEncoder.m, LRCalc4PolarSC.m, and MiuCalc4PolarSC.m.
    If ``forced_internal`` is supplied, it evaluates the causal probability
    of an already chosen path, which is the decoder-visible entropy model.
    """
    lr_in = np.clip(np.asarray(leaf_lr, dtype=np.float64), 1e-30, 1e30)
    n = lr_in.size
    depth = int(math.log2(n))
    lr_reg = np.ones((n // 2, depth), dtype=np.float64)
    mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)
    u = np.zeros(n, dtype=np.uint8)
    nll = 0.0
    selected_count = 0
    selected_bits: list[int] = []
    selected_freq1: list[int] = []

    def adjust(a: np.ndarray | float) -> np.ndarray | float:
        return np.clip(a, 1e-30, 1e30)

    for i0 in range(n):
        i_one = i0 + 1
        if i_one == 1:
            end = int(layers[i0])
            col = end - 1
            a = lr_in[0::2]
            b = lr_in[1::2]
            lr_reg[:, col] = adjust((a * b + 1.0) / (a + b))
            for lev_one in range(end - 1, 0, -1):
                src_col = lev_one
                dst_col = lev_one - 1
                count = 1 << lev_one
                a = lr_reg[0:count:2, src_col]
                b = lr_reg[1:count:2, src_col]
                lr_reg[: count // 2, dst_col] = adjust((a * b + 1.0) / (a + b))
        elif i_one == n // 2 + 1:
            end = int(layers[i0])
            col = end - 1
            a = lr_in[0::2]
            b = lr_in[1::2]
            used = mu_reg[:, -1].astype(np.int8)
            lr_reg[:, col] = adjust(np.power(a, 1 - 2 * used) * b)
            for lev_one in range(end - 1, 0, -1):
                src_col = lev_one
                dst_col = lev_one - 1
                count = 1 << lev_one
                a = lr_reg[0:count:2, src_col]
                b = lr_reg[1:count:2, src_col]
                lr_reg[: count // 2, dst_col] = adjust((a * b + 1.0) / (a + b))
        elif i_one % 2 == 0:
            end = int(layers[i0])
            dst_col = end - 1
            src_col = end
            a = float(lr_reg[0, src_col])
            b = float(lr_reg[1, src_col])
            used = int(mu_reg[0, 0])
            lr_reg[0, dst_col] = adjust((a ** (1 - 2 * used)) * b)
        else:
            end = int(layers[i0])
            dst_col = end - 1
            src_col = end
            count = 1 << end
            a = lr_reg[0:count:2, src_col]
            b = lr_reg[1:count:2, src_col]
            used = mu_reg[: count // 2, dst_col].astype(np.int8)
            lr_reg[: count // 2, dst_col] = adjust(np.power(a, 1 - 2 * used) * b)
            for lev_one in range(end - 1, 0, -1):
                src_col2 = lev_one
                dst_col2 = lev_one - 1
                count2 = 1 << lev_one
                a2 = lr_reg[0:count2:2, src_col2]
                b2 = lr_reg[1:count2:2, src_col2]
                lr_reg[: count2 // 2, dst_col2] = adjust((a2 * b2 + 1.0) / (a2 + b2))

        root_lr = float(np.clip(lr_reg[0, 0], 1e-30, 1e30))
        p1 = 1.0 / (1.0 + root_lr)
        if forced_internal is not None:
            bit = int(forced_internal[i0])
        elif freeze_flag[i0]:
            bit = int(frozen_external[reverse[i0]])
        elif decision == "map":
            bit = int(root_lr < 1.0)
        elif decision == "random":
            bit = int(rng.random() < p1)
        else:
            raise ValueError(f"unknown decision {decision}")
        u[i0] = bit

        if score_selected and not freeze_flag[i0]:
            prob = p1 if bit else (1.0 - p1)
            nll -= math.log2(max(prob, 1e-300))
            selected_count += 1
            selected_bits.append(bit)
            selected_freq1.append(min(65535, max(1, int(math.floor(p1 * 65536.0 + 0.5)))))

        # Direct port of MiuCalc4PolarSC.m.
        if i_one % 2 == 1:
            mu_reg[0, 0] = bit
        else:
            end = int(layers[i_one])  # MATLAB indexes SCLayer(I+1).
            temp = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temp[0] = bit
            for j_one in range(1, end):
                length = 1 << (j_one - 1)
                left = mu_reg[:length, j_one - 1]
                right = temp[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2] = left ^ right
                merged[1::2] = right
                temp[: 2 * length] = merged
            mu_reg[: 1 << max(end - 1, 0), end - 1] = temp

    return SCResult(
        external_u=u[reverse].copy(),
        internal_u=u,
        selected_nll_bits=nll,
        selected_count=selected_count,
        selected_bits=np.asarray(selected_bits, dtype=np.uint8),
        selected_freq1_u16=np.asarray(selected_freq1, dtype=np.uint16),
    )


def arithmetic_encode_binary(bits: np.ndarray, freq1: np.ndarray) -> tuple[bytes, int]:
    """32-bit binary arithmetic coder with a fixed 16-bit frequency total."""
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    low = 0
    high = full - 1
    pending = 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        if pending:
            output.extend([1 - bit] * pending)
            pending = 0

    for bit_u8, f1_u16 in zip(bits, freq1, strict=True):
        bit = int(bit_u8)
        f1 = int(f1_u16)
        f0 = 65536 - f1
        width = high - low + 1
        split = low + (width * f0 // 65536) - 1
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1

    pending += 1
    emit(0 if low < quarter else 1)
    logical_bits = len(output)
    packed = np.packbits(np.asarray(output, dtype=np.uint8), bitorder="big").tobytes()
    return packed, logical_bits


def arithmetic_decode_binary(payload: bytes, logical_bits: int, freq1: np.ndarray) -> np.ndarray:
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    packed_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    cursor = 0

    def read_bit() -> int:
        nonlocal cursor
        if cursor >= logical_bits:
            return 0
        value = int(packed_bits[cursor])
        cursor += 1
        return value

    low = 0
    high = full - 1
    code = 0
    for _ in range(32):
        code = ((code << 1) & (full - 1)) | read_bit()
    decoded = np.empty(freq1.size, dtype=np.uint8)
    for index, f1_u16 in enumerate(freq1):
        f1 = int(f1_u16)
        f0 = 65536 - f1
        width = high - low + 1
        scaled = ((code - low + 1) * 65536 - 1) // width
        split = low + (width * f0 // 65536) - 1
        if scaled < f0:
            decoded[index] = 0
            high = split
        else:
            decoded[index] = 1
            low = split + 1
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarters:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
            code = ((code << 1) & (full - 1)) | read_bit()
    return decoded


def periodic_binary_capacity(sigma: float, grid: int = 1 << 17, neighbors: int = 16) -> float:
    """Capacity of uniform binary input through (X+N(0,sigma^2)) mod 2."""
    y = (np.arange(grid, dtype=np.float64) + 0.5) * (2.0 / grid)
    ks = np.arange(-neighbors, neighbors + 1, dtype=np.float64)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    p0 = np.exp(-0.5 * ((y[:, None] + 2.0 * ks[None, :]) / sigma) ** 2).sum(1) * norm
    p1 = np.exp(-0.5 * ((y[:, None] - 1.0 + 2.0 * ks[None, :]) / sigma) ** 2).sum(1) * norm
    mix = 0.5 * (p0 + p1)
    post = np.divide(p1, p0 + p1, out=np.full_like(p1, 0.5), where=(p0 + p1) > 0)
    h = -(post * np.log2(np.maximum(post, 1e-300)) + (1.0 - post) * np.log2(np.maximum(1.0 - post, 1e-300)))
    return float(1.0 - np.sum(mix * h) * (2.0 / grid))


def reliability_freeze_flags(repo: Path, n: int, capacities: Iterable[float]) -> list[np.ndarray]:
    reverse = bit_reverse_indices(n)
    logn = int(math.log2(n))
    flags: list[np.ndarray] = []
    for level, capacity in enumerate(capacities, start=1):
        keep = min(n, max(0, int(math.ceil(n * float(capacity)))))
        flag = np.zeros(n, dtype=np.uint8)
        if keep == n:
            flags.append(flag)
            continue
        if keep == 0:
            flag[:] = 1
            flags.append(flag)
            continue
        if level <= 3:
            path = repo / f"Pe_BIMod2AWGN_test_D_0.20_tSigma_0.4422_Lvl_{level}_n_{logn}.mat"
            pe = np.asarray(loadmat(path)["PeLast"]).ravel()
            zn = pe[reverse]
            ordered = np.argsort(zn, kind="stable")
            freeze_index = np.sort(ordered[keep:])
            freeze_resolved = reverse[freeze_index]
            flag[freeze_resolved] = 1
        else:
            # The public reference omits tables for nearly-perfect levels.
            # Freeze the earliest internal positions only as a conservative
            # screen; a promoted run must build exact target reliability sets.
            flag[: n - keep] = 1
        flags.append(flag)
    return flags


def leaf_likelihood_ratios_gpu(
    y: cp.ndarray,
    alphabet: cp.ndarray,
    weights: cp.ndarray,
    distortion: float,
    previous: cp.ndarray,
    level: int,
) -> np.ndarray:
    dens = cp.exp(-0.5 * ((y[:, None] - alphabet[None, :]) ** 2) / distortion) * weights[None, :]
    lower_mod = 1 << (level - 1)
    bit = 1 << (level - 1)
    out = cp.empty(y.size, dtype=cp.float64)
    for context in range(lower_mod):
        pos = cp.where(previous == context)[0]
        if pos.size == 0:
            continue
        idx0 = cp.asarray([j for j in range(alphabet.size) if (j % lower_mod == context and (j & bit) == 0)])
        idx1 = cp.asarray([j for j in range(alphabet.size) if (j % lower_mod == context and (j & bit) != 0)])
        p0 = dens[pos[:, None], idx0[None, :]].sum(axis=1)
        p1 = dens[pos[:, None], idx1[None, :]].sum(axis=1)
        out[pos] = cp.clip(p0 / cp.maximum(p1, 1e-300), 1e-30, 1e30)
    return cp.asnumpy(out)


def leaf_prior_ratios(weights: np.ndarray, previous: np.ndarray, level: int) -> np.ndarray:
    lower_mod = 1 << (level - 1)
    bit = 1 << (level - 1)
    values = np.empty(lower_mod, dtype=np.float64)
    for context in range(lower_mod):
        idx0 = [j for j in range(weights.size) if j % lower_mod == context and (j & bit) == 0]
        idx1 = [j for j in range(weights.size) if j % lower_mod == context and (j & bit) != 0]
        values[context] = weights[idx0].sum() / max(weights[idx1].sum(), 1e-300)
    return np.clip(values[previous], 1e-30, 1e30)


def run_trial(args: argparse.Namespace, trial: int, capacities: list[float], flags: list[np.ndarray]) -> dict:
    n = args.block_length
    reverse = bit_reverse_indices(n)
    layers = sc_layers(n)
    rng = np.random.default_rng(args.seed + 104729 * trial)
    cp_rng = cp.random.RandomState(args.seed + 104729 * trial)
    source_row: dict[str, object]
    if args.input_bf16 is None:
        y_gpu = cp_rng.normal(0.0, args.sigma_source, size=n, dtype=cp.float64)
        source_row = {"kind": "synthetic_gaussian", "trial_seed": args.seed + 104729 * trial}
    else:
        raw = np.fromfile(args.input_bf16, dtype="<u2")
        values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        block_index = args.input_block_start + trial
        begin = block_index * n
        end = begin + n
        if end > values.size:
            raise ValueError(
                f"input block {block_index} ends at {end}, beyond {values.size} values"
            )
        block = cp.asarray(values[begin:end], dtype=cp.float64)
        block_rms = float(cp.sqrt(cp.mean(block * block)).get())
        if not math.isfinite(block_rms) or block_rms <= 0:
            raise ValueError(f"invalid block RMS {block_rms}")
        y_gpu = block * (args.sigma_source / block_rms)
        source_row = {
            "kind": "frozen_bf16_weight_block",
            "path": str(args.input_bf16),
            "block_index": block_index,
            "values": n,
            "block_rms_fp64": block_rms,
            "decoder_scale_fp32": float(np.float32(block_rms / args.sigma_source)),
        }

    sigma_recon = math.sqrt(args.sigma_source**2 - args.test_distortion)
    alphabet_np = args.eta * np.arange(-args.alphabet_size // 2 + 1, args.alphabet_size // 2 + 1, dtype=np.float64)
    weights_np = np.exp(-0.5 * (alphabet_np / sigma_recon) ** 2)
    alphabet_gpu = cp.asarray(alphabet_np)
    weights_gpu = cp.asarray(weights_np)

    previous_gpu = cp.zeros(n, dtype=cp.int16)
    previous_np = np.zeros(n, dtype=np.int16)
    x_levels: list[np.ndarray] = []
    ideal_bits = 0.0
    selected = 0
    level_rows = []
    level_audit: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for level in range(1, int(math.log2(args.alphabet_size)) + 1):
        posterior_lr = leaf_likelihood_ratios_gpu(
            y_gpu, alphabet_gpu, weights_gpu, args.test_distortion, previous_gpu, level
        )
        frozen_rng = np.random.default_rng(args.seed + 104729 * trial + 1000003 * level)
        frozen_external = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        chosen = sc_encode_ratio(
            posterior_lr,
            flags[level - 1],
            frozen_external,
            reverse,
            layers,
            rng=rng,
            decision=args.decision,
        )
        x_bit = polar_transform(chosen.external_u)

        prior_lr = leaf_prior_ratios(weights_np, previous_np, level)
        scored = sc_encode_ratio(
            prior_lr,
            flags[level - 1],
            frozen_external,
            reverse,
            layers,
            rng=rng,
            decision="map",
            forced_internal=chosen.internal_u,
            score_selected=True,
        )
        ideal_bits += scored.selected_nll_bits
        selected += scored.selected_count
        level_audit.append((frozen_external, chosen.internal_u, scored.selected_freq1_u16))
        x_levels.append(x_bit)
        previous_np += (1 << (level - 1)) * x_bit.astype(np.int16)
        previous_gpu = cp.asarray(previous_np)
        level_rows.append(
            {
                "level": level,
                "capacity_schedule": capacities[level - 1],
                "selected_fraction": float((flags[level - 1] == 0).mean()),
                "selected_nll_bits": scored.selected_nll_bits,
                "selected_nll_bpw": scored.selected_nll_bits / n,
            }
        )

    reconstruct_gpu = alphabet_gpu[previous_gpu]
    squared = cp.square(y_gpu - reconstruct_gpu)
    source_energy = cp.square(y_gpu)
    distortion_abs = float(cp.mean(squared).get())
    relative_mse = float((cp.sum(squared) / cp.sum(source_energy)).get())

    all_selected_bits = np.concatenate(
        [
            chosen_internal[flags[level_index] == 0]
            for level_index, (_, chosen_internal, _) in enumerate(level_audit)
        ]
    )
    all_freq1 = np.concatenate([row[2] for row in level_audit])
    payload, arithmetic_logical_bits = arithmetic_encode_binary(all_selected_bits, all_freq1)
    decoded_selected = arithmetic_decode_binary(payload, arithmetic_logical_bits, all_freq1)
    arithmetic_bits_match = bool(np.array_equal(decoded_selected, all_selected_bits))

    # Rebuild every causal prior probability from only decoded bits, fixed
    # frozen bits, and already reconstructed lower levels. This catches a
    # hidden encoder-only probability schedule.
    decoded_cursor = 0
    decoded_previous = np.zeros(n, dtype=np.int16)
    regenerated_freqs: list[np.ndarray] = []
    decoded_x_levels: list[np.ndarray] = []
    for level_index in range(len(x_levels)):
        frozen_external, _, original_freqs = level_audit[level_index]
        flag = flags[level_index]
        decoded_internal = np.empty(n, dtype=np.uint8)
        count = int((flag == 0).sum())
        take = decoded_selected[decoded_cursor : decoded_cursor + count]
        decoded_cursor += count
        selected_cursor = 0
        for i0 in range(n):
            if flag[i0]:
                decoded_internal[i0] = frozen_external[reverse[i0]]
            else:
                decoded_internal[i0] = take[selected_cursor]
                selected_cursor += 1
        prior_lr = leaf_prior_ratios(weights_np, decoded_previous, level_index + 1)
        rescored = sc_encode_ratio(
            prior_lr,
            flag,
            frozen_external,
            reverse,
            layers,
            rng=rng,
            decision="map",
            forced_internal=decoded_internal,
            score_selected=True,
        )
        regenerated_freqs.append(rescored.selected_freq1_u16)
        decoded_x = polar_transform(rescored.external_u)
        decoded_x_levels.append(decoded_x)
        decoded_previous += (1 << level_index) * decoded_x.astype(np.int16)
        if not np.array_equal(rescored.selected_freq1_u16, original_freqs):
            raise AssertionError(f"causal frequency mismatch at level {level_index + 1}")

    causal_frequencies_match = bool(np.array_equal(np.concatenate(regenerated_freqs), all_freq1))
    reconstruction_indices_match = bool(np.array_equal(decoded_previous, previous_np))
    if not (arithmetic_bits_match and causal_frequencies_match and reconstruction_indices_match):
        raise AssertionError("arithmetic round-trip audit failed")

    # Literal per-block container: u32 logical payload length, FP32 decoder
    # scale, then byte-padded arithmetic payload. The fixed architecture and
    # reliability tables are model-global constants and ledgers charge them
    # separately.
    decoder_scale = float(source_row.get("decoder_scale_fp32", np.float32(1.0)))
    container = struct.pack("<If", arithmetic_logical_bits, decoder_scale) + payload
    framing_bits = 64
    total_bits = len(container) * 8
    rate = total_bits / n
    gaussian = 2.0 ** (-2.0 * rate)
    gap_db = 10.0 * math.log10(relative_mse / gaussian)
    threshold = (10.0 ** (0.10 / 10.0)) * gaussian
    return {
        "trial": trial,
        "source": source_row,
        "absolute_mse": distortion_abs,
        "relative_mse": relative_mse,
        "ideal_entropy_bits": ideal_bits,
        "arithmetic_logical_bits": arithmetic_logical_bits,
        "arithmetic_payload_bytes": len(payload),
        "arithmetic_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "arithmetic_roundtrip_bits_match": arithmetic_bits_match,
        "causal_decoder_frequencies_match": causal_frequencies_match,
        "reconstruction_indices_match": reconstruction_indices_match,
        "literal_container_bytes": len(container),
        "literal_container_sha256": hashlib.sha256(container).hexdigest(),
        "framing_bits": framing_bits,
        "total_screen_bits": total_bits,
        "screen_bpw": rate,
        "gaussian_limit_mse_at_screen_rate": gaussian,
        "threshold_mse_0p10db": threshold,
        "gap_db": gap_db,
        "passes_rate_lt_2p5": rate < 2.5,
        "passes_gap_lt_0p10db": gap_db < 0.10,
        "selected_polar_bits": selected,
        "levels": level_rows,
        "_container_hex": container.hex() if args.emit_container_hex else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    ap.add_argument("--block-length", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--sigma-source", type=float, default=3.0)
    ap.add_argument("--test-distortion", type=float, default=0.28)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--alphabet-size", type=int, default=64)
    ap.add_argument("--decision", choices=("map", "random"), default="random")
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--input-bf16", type=Path)
    ap.add_argument("--input-block-start", type=int, default=0)
    ap.add_argument("--emit-container-hex", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    n = args.block_length
    if n not in (1 << k for k in range(10, 19)):
        raise ValueError("public reliability tables cover powers 2^10 through 2^18")
    sigma_recon = math.sqrt(args.sigma_source**2 - args.test_distortion)
    tilde_sigma = sigma_recon * math.sqrt(args.test_distortion) / args.sigma_source
    levels = int(math.log2(args.alphabet_size))
    capacities = [
        periodic_binary_capacity(tilde_sigma / args.eta / (1 << level0))
        for level0 in range(levels)
    ]
    flags = reliability_freeze_flags(args.polar_repo, n, capacities)

    started = time.perf_counter()
    rows = [run_trial(args, trial, capacities, flags) for trial in range(args.trials)]
    elapsed = time.perf_counter() - started
    mean_rate = float(np.mean([r["screen_bpw"] for r in rows]))
    mean_mse = float(np.mean([r["relative_mse"] for r in rows]))
    aggregate_gap = 10.0 * math.log10(mean_mse / (2.0 ** (-2.0 * mean_rate)))
    result = {
        "architecture": "entropy-coded multilevel polar-lattice PTQ screen",
        "claim_boundary": (
            "published D=0.20 Tal-Vardy reliability order with ideal entropy length; "
            "not yet a normative arithmetic bitstream"
        ),
        "strict_ptq": True,
        "source_training_or_retraining": False,
        "parameters": {
            "block_length": n,
            "trials": args.trials,
            "sigma_source": args.sigma_source,
            "test_channel_distortion": args.test_distortion,
            "eta": args.eta,
            "alphabet_size": args.alphabet_size,
            "decision": args.decision,
            "tilde_sigma": tilde_sigma,
            "capacity_schedule": capacities,
            "seed": args.seed,
        },
        "aggregate": {
            "mean_relative_mse": mean_mse,
            "mean_screen_bpw": mean_rate,
            "gaussian_limit_mse": 2.0 ** (-2.0 * mean_rate),
            "threshold_mse_0p10db": (10.0 ** 0.01) * (2.0 ** (-2.0 * mean_rate)),
            "gap_db": aggregate_gap,
            "passes_rate_lt_2p5": mean_rate < 2.5,
            "passes_gap_lt_0p10db": aggregate_gap < 0.10,
        },
        "trials": rows,
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "seconds": elapsed,
    }
    containers: list[bytes] = []
    for row in rows:
        encoded = row.pop("_container_hex")
        if encoded is not None:
            containers.append(bytes.fromhex(encoded))
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.emit_container_hex and args.output:
        bitstream_path = args.output.with_suffix(".polar.bin")
        bitstream_path.write_bytes(b"".join(containers))


if __name__ == "__main__":
    main()
