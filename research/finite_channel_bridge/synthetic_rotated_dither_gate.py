#!/usr/bin/env python3
"""Source-free rotated subtractive-dither channel and real serializer gate.

The input is generated i.i.d. Gaussian data only.  The script never accepts a
source path.  With ``--backend cupy`` the N19 RHT/quantizer path runs on CUDA;
the canonical arithmetic serializer stays on the CPU so its bytes are stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


A32 = np.float32(0.9492341876029968)
SIGMA32 = np.float32(0.21951906383037567)
DELTA32 = np.float32(math.sqrt(12.0) * float(SIGMA32))
DESIGN_D = 2.0 ** (-4.3)
MODEL_BYTES_FP16 = 475_654
EXPERT_HEADER_BYTES = 128
EXPERTS = 128
BLOCKS_PER_EXPERT = 9
WEIGHTS_PER_EXPERT = 4_718_592
K = 31
ALPHABET = 2 * K + 3  # left escape, -K..K, right escape
FREQ_TOTAL = 1 << 16


def splitmix64_array(n: int, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.arange(n, dtype=np.uint64) + np.uint64(seed)
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def public_dither_and_signs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    words = splitmix64_array(n, seed)
    dither = (((words >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53 - 0.5).astype(
        np.float32
    )
    sign_words = splitmix64_array(n, seed ^ 0xD1B54A32D192ED03)
    signs = np.where((sign_words & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32)
    return dither, signs


def rht(values: Any, xp: Any) -> Any:
    n = int(values.size)
    if n <= 0 or n & (n - 1):
        raise ValueError("RHT length must be a positive power of two")
    out = values.reshape(-1).copy()
    width = 1
    while width < n:
        view = out.reshape(-1, 2 * width)
        left = view[:, :width].copy()
        right = view[:, width:].copy()
        view[:, :width] = left + right
        view[:, width:] = left - right
        width *= 2
    out *= xp.float32(1.0 / math.sqrt(n))
    return out


def forward_rht(values: Any, signs: Any, xp: Any) -> Any:
    return rht(values * signs, xp)


def inverse_rht(values: Any, signs: Any, xp: Any) -> Any:
    return rht(values, xp) * signs


def normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def symbol_probabilities(dither: float) -> np.ndarray:
    scale = float(DELTA32) / float(A32)
    p = np.empty(ALPHABET, dtype=np.float64)
    p[0] = normal_cdf(scale * (-K - 0.5 - dither))
    for offset, q in enumerate(range(-K, K + 1), start=1):
        lo = scale * (q - 0.5 - dither)
        hi = scale * (q + 0.5 - dither)
        p[offset] = normal_cdf(hi) - normal_cdf(lo)
    p[-1] = 1.0 - normal_cdf(scale * (K + 0.5 - dither))
    # The escape tails can underflow on this deliberately wide alphabet.  A
    # tiny positive floor keeps every arithmetic branch decodable.
    p = np.maximum(p, np.finfo(np.float64).tiny)
    p /= p.sum(dtype=np.float64)
    return p


def build_tree() -> tuple[list[tuple[int, int, int, int, int]], np.ndarray, np.ndarray, np.ndarray]:
    nodes: list[tuple[int, int, int, int, int]] = []

    def add(lo: int, hi: int) -> int:
        node = len(nodes)
        nodes.append((lo, hi, -1, -1, -1))
        if hi - lo == 1:
            return node
        mid = (lo + hi) // 2
        left = add(lo, mid)
        right = add(mid, hi)
        nodes[node] = (lo, hi, mid, left, right)
        return node

    add(0, ALPHABET)
    max_depth = int(math.ceil(math.log2(ALPHABET)))
    path_nodes = np.full((ALPHABET, max_depth), -1, dtype=np.int16)
    path_bits = np.zeros((ALPHABET, max_depth), dtype=np.uint8)
    path_lengths = np.zeros(ALPHABET, dtype=np.uint8)
    for symbol in range(ALPHABET):
        node = 0
        depth = 0
        while nodes[node][1] - nodes[node][0] > 1:
            lo, hi, mid, left, right = nodes[node]
            del lo, hi
            path_nodes[symbol, depth] = node
            bit = int(symbol >= mid)
            path_bits[symbol, depth] = bit
            node = right if bit else left
            depth += 1
        path_lengths[symbol] = depth
    return nodes, path_nodes, path_bits, path_lengths


TREE, PATH_NODES, PATH_BITS, PATH_LENGTHS = build_tree()
INTERNAL_NODES = sum(1 for lo, hi, *_ in TREE if hi - lo > 1)


def build_frequency_table(phase_bins: int, quadrature: int = 32) -> np.ndarray:
    pmf = np.zeros((phase_bins, ALPHABET), dtype=np.float64)
    for phase in range(phase_bins):
        for sub in range(quadrature):
            dither = -0.5 + (phase + (sub + 0.5) / quadrature) / phase_bins
            pmf[phase] += symbol_probabilities(dither)
        pmf[phase] /= quadrature
    table = np.zeros((phase_bins, len(TREE)), dtype=np.uint16)
    for phase in range(phase_bins):
        for node, (lo, hi, mid, _left, _right) in enumerate(TREE):
            if hi - lo == 1:
                continue
            # Direct slices retain subnormal tail mass.  Prefix subtraction can
            # erase it when a far-tail node follows probability mass near one.
            total = float(np.sum(pmf[phase, lo:hi], dtype=np.float64))
            right = float(np.sum(pmf[phase, mid:hi], dtype=np.float64))
            freq = int(round(FREQ_TOTAL * right / total))
            table[phase, node] = np.uint16(min(65535, max(1, freq)))
    return table


def phases_for_dither(dither: np.ndarray, phase_bins: int) -> np.ndarray:
    phase = np.floor((dither.astype(np.float64) + 0.5) * phase_bins).astype(np.int64)
    return np.clip(phase, 0, phase_bins - 1)


def q_to_symbols(q: np.ndarray) -> tuple[np.ndarray, int]:
    symbols = np.empty(q.size, dtype=np.int16)
    low = q < -K
    high = q > K
    middle = ~(low | high)
    symbols[low] = 0
    symbols[middle] = q[middle].astype(np.int64) + K + 1
    symbols[high] = ALPHABET - 1
    return symbols, int(np.count_nonzero(low | high))


def decisions_for_symbols(
    symbols: np.ndarray, phases: np.ndarray, frequency_table: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    max_depth = PATH_NODES.shape[1]
    n = symbols.size
    bits = np.zeros((n, max_depth), dtype=np.uint8)
    frequencies = np.ones((n, max_depth), dtype=np.uint16)
    valid = np.arange(max_depth)[None, :] < PATH_LENGTHS[symbols, None]
    bits[:] = PATH_BITS[symbols]
    for depth in range(max_depth):
        active = valid[:, depth]
        node = PATH_NODES[symbols[active], depth]
        frequencies[active, depth] = frequency_table[phases[active], node]
    return bits[valid], frequencies[valid]


def arithmetic_encode_binary(bits: np.ndarray, freq1: np.ndarray) -> tuple[bytes, int]:
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
        f0 = FREQ_TOTAL - f1
        width = high - low + 1
        split = low + width * f0 // FREQ_TOTAL - 1
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
    payload = np.packbits(np.asarray(output, dtype=np.uint8), bitorder="big").tobytes()
    return payload, logical_bits


class ArithmeticDecoder:
    def __init__(self, payload: bytes, logical_bits: int):
        self.full = 1 << 32
        self.half = 1 << 31
        self.quarter = 1 << 30
        self.three_quarters = 3 << 30
        self.bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
        self.logical_bits = logical_bits
        self.cursor = 0
        self.low = 0
        self.high = self.full - 1
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()

    def _read(self) -> int:
        if self.cursor >= self.logical_bits:
            return 0
        bit = int(self.bits[self.cursor])
        self.cursor += 1
        return bit

    def decode(self, f1: int) -> int:
        f1 = min(65535, max(1, int(f1)))
        f0 = FREQ_TOTAL - f1
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * FREQ_TOTAL - 1) // width
        split = self.low + width * f0 // FREQ_TOTAL - 1
        if scaled < f0:
            bit = 0
            self.high = split
        else:
            bit = 1
            self.low = split + 1
        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.code -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.low -= self.quarter
                self.high -= self.quarter
                self.code -= self.quarter
            else:
                break
            self.low = (self.low << 1) & (self.full - 1)
            self.high = ((self.high << 1) & (self.full - 1)) | 1
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()
        return bit


def decode_symbols(
    count: int, phases: np.ndarray, frequency_table: np.ndarray, payload: bytes, logical_bits: int
) -> np.ndarray:
    decoder = ArithmeticDecoder(payload, logical_bits)
    decoded = np.empty(count, dtype=np.int16)
    for index in range(count):
        node = 0
        phase = int(phases[index])
        while TREE[node][1] - TREE[node][0] > 1:
            bit = decoder.decode(int(frequency_table[phase, node]))
            node = TREE[node][4] if bit else TREE[node][3]
        decoded[index] = TREE[node][0]
    return decoded


def ks_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.sort(np.asarray(first, dtype=np.float64))
    second = np.sort(np.asarray(second, dtype=np.float64))
    grid = np.sort(np.concatenate((first, second)))
    cdf_a = np.searchsorted(first, grid, side="right") / first.size
    cdf_b = np.searchsorted(second, grid, side="right") / second.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def distribution_metrics(noise: np.ndarray, seed: int) -> dict[str, Any]:
    z = np.asarray(noise, dtype=np.float64)
    mean = float(np.mean(z))
    centered = z - mean
    variance = float(np.mean(centered * centered))
    skew = float(np.mean(centered**3) / variance**1.5)
    excess = float(np.mean(centered**4) / variance**2 - 3.0)
    tile_count = min(z.size // 256, 4096)
    tiles = z[: tile_count * 256].reshape(tile_count, 256)
    rng = np.random.default_rng(seed)
    gaussian_a = rng.standard_normal(tiles.shape)
    gaussian_b = rng.standard_normal(tiles.shape)
    directions = rng.standard_normal((256, 24))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    proj = tiles @ directions
    proj_a = gaussian_a @ directions
    proj_b = gaussian_b @ directions
    w2 = []
    w2_baseline = []
    ks = []
    ks_baseline = []
    for column in range(directions.shape[1]):
        p = np.sort(proj[:, column])
        a = np.sort(proj_a[:, column])
        b = np.sort(proj_b[:, column])
        w2.append(float(math.sqrt(np.mean((p - a) ** 2))))
        w2_baseline.append(float(math.sqrt(np.mean((a - b) ** 2))))
        ks.append(ks_distance(p, a))
        ks_baseline.append(ks_distance(a, b))
    energy = np.sum(tiles * tiles, axis=1, dtype=np.float64)
    covariance = np.cov(tiles[:, :16], rowvar=False, ddof=0)
    offdiag = covariance - np.diag(np.diag(covariance))
    return {
        "scalar": {
            "mean": mean,
            "variance": variance,
            "skew": skew,
            "excess_kurtosis": excess,
        },
        "tiles": {
            "tile_values": 256,
            "tile_count": tile_count,
            "energy_mean": float(np.mean(energy)),
            "energy_variance": float(np.var(energy)),
            "gaussian_energy_mean": 256.0,
            "gaussian_energy_variance": 512.0,
            "max_abs_offdiag_covariance_first16": float(np.max(np.abs(offdiag))),
            "sliced_w2_mean_vs_gaussian": float(np.mean(w2)),
            "sliced_w2_mean_gaussian_vs_gaussian": float(np.mean(w2_baseline)),
            "sliced_ks_max_vs_gaussian": float(np.max(ks)),
            "sliced_ks_max_gaussian_vs_gaussian": float(np.max(ks_baseline)),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    n = 1 << args.log2_n
    if args.phase_bins <= 0:
        raise ValueError("phase bins must be positive")
    rng = np.random.default_rng(args.source_seed)
    x_np = rng.standard_normal(n).astype(np.float32)
    dither_np, signs_np = public_dither_and_signs(n, args.public_seed)

    if args.backend == "cupy":
        import cupy as cp

        xp = cp
        x = cp.asarray(x_np)
        dither = cp.asarray(dither_np)
        signs = cp.asarray(signs_np)
        device = {
            "backend": "cupy",
            "cupy": cp.__version__,
            "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        }
    else:
        xp = np
        x = x_np
        dither = dither_np
        signs = signs_np
        device = {"backend": "numpy", "numpy": np.__version__, "platform": platform.platform()}

    started = time.perf_counter()
    transformed = forward_rht(x, signs, xp)
    q = xp.rint(A32 * transformed / DELTA32 + dither).astype(xp.int16)
    reconstructed_rotated = DELTA32 * (q.astype(xp.float32) - dither)
    y = inverse_rht(reconstructed_rotated, signs, xp)
    if args.backend == "cupy":
        xp.cuda.Stream.null.synchronize()
        transformed_np = xp.asnumpy(transformed)
        q_np = xp.asnumpy(q)
        y_np = xp.asnumpy(y)
    else:
        transformed_np = np.asarray(transformed)
        q_np = np.asarray(q)
        y_np = np.asarray(y)
    transform_seconds = time.perf_counter() - started

    normalized_error = (y_np.astype(np.float64) - float(A32) * x_np.astype(np.float64)) / float(
        SIGMA32
    )
    mse = float(np.mean((x_np.astype(np.float64) - y_np.astype(np.float64)) ** 2))
    phases = phases_for_dither(dither_np, args.phase_bins)
    table = build_frequency_table(args.phase_bins)
    symbols, escapes = q_to_symbols(q_np)
    if escapes:
        raise AssertionError(
            f"synthetic wide-alphabet gate unexpectedly produced {escapes} escapes; "
            "the production format would put their signed residuals in the charged escape stream"
        )
    bits, frequencies = decisions_for_symbols(symbols, phases, table)
    ideal_codelength = float(
        np.sum(
            np.where(
                bits == 1,
                -np.log2(frequencies.astype(np.float64) / FREQ_TOTAL),
                -np.log2(1.0 - frequencies.astype(np.float64) / FREQ_TOTAL),
            ),
            dtype=np.float64,
        )
    )
    serialize_started = time.perf_counter()
    payload, logical_bits = arithmetic_encode_binary(bits, frequencies)
    decoded_symbols = decode_symbols(n, phases, table, payload, logical_bits)
    serialize_seconds = time.perf_counter() - serialize_started
    if not np.array_equal(decoded_symbols, symbols):
        raise AssertionError("canonical arithmetic round trip failed")
    if any(payload[-1:]) and logical_bits % 8:
        terminal = payload[-1]
        unused_mask = (1 << (8 - logical_bits % 8)) - 1
        if terminal & unused_mask:
            raise AssertionError("nonzero arithmetic padding bits")

    table_bytes = args.phase_bins * INTERNAL_NODES * 2 + 256
    global_bytes = MODEL_BYTES_FP16 + table_bytes
    # For smoke sizes below N19, scale the observed byte-padded rate to an N19
    # stream before constructing the representative system ledger.  At N19
    # this is exactly the emitted payload length.
    n19 = 1 << 19
    projected_block_bytes = math.ceil(len(payload) * n19 / n)
    representative_local_bytes = EXPERT_HEADER_BYTES + BLOCKS_PER_EXPERT * projected_block_bytes
    representative_total_bytes = global_bytes + EXPERTS * representative_local_bytes
    physical_bpw = 8.0 * representative_total_bytes / (EXPERTS * WEIGHTS_PER_EXPERT)
    equal_share = representative_total_bytes / EXPERTS
    cold_bytes = representative_local_bytes + global_bytes
    target = 0.8 * 2.0 ** (-2.0 * physical_bpw)
    ratio_needed = target / mse

    parseval_before = float(np.sum(x_np.astype(np.float64) ** 2))
    parseval_after = float(np.sum(transformed_np.astype(np.float64) ** 2))
    return {
        "schema": "rotated_dither_source_free_synthetic_gate_v1",
        "claim_boundary": (
            "Gaussian synthetic input only; actual arithmetic bytes and decode round trip, "
            "but no Qwen result and no finite SILWARP success claim."
        ),
        "configuration": {
            "n": n,
            "log2_n": args.log2_n,
            "a32": float(A32),
            "sigma32": float(SIGMA32),
            "delta32": float(DELTA32),
            "phase_bins": args.phase_bins,
            "alphabet": ALPHABET,
            "public_seed": args.public_seed,
            "source_seed": args.source_seed,
            "device": device,
        },
        "transform": {
            "seconds": transform_seconds,
            "parseval_relative_error": abs(parseval_after - parseval_before) / parseval_before,
        },
        "distortion": {
            "empirical_mse": mse,
            "represented_awgn_identity_mse": 0.05076578709329227,
            "mathematical_design_mse": DESIGN_D,
            "empirical_over_represented": mse / 0.05076578709329227,
        },
        "noise_distribution": distribution_metrics(normalized_error, args.source_seed ^ 0xA551),
        "serialization": {
            "roundtrip_exact": True,
            "escape_symbols": escapes,
            "decision_count": int(bits.size),
            "ideal_quantized_frequency_bits": ideal_codelength,
            "logical_arithmetic_bits": logical_bits,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "logical_bpw": logical_bits / n,
            "byte_padded_bpw": 8.0 * len(payload) / n,
            "arithmetic_redundancy_bits": logical_bits - ideal_codelength,
            "encode_plus_decode_seconds": serialize_seconds,
            "stored_frequency_table_bytes": table_bytes,
        },
        "representative_128_expert_system": {
            "warning": "nine identical synthetic stream lengths are a planning extrapolation, not a Qwen byte measurement",
            "global_decoder_plus_frequency_bytes": global_bytes,
            "projected_n19_block_bytes_from_observed_rate": projected_block_bytes,
            "expert_local_bytes": representative_local_bytes,
            "physical_bpw": physical_bpw,
            "cold_bytes": cold_bytes,
            "cold_read_amplification": cold_bytes / equal_share,
            "same_rate_target_mse": target,
            "required_after_over_identity": ratio_needed,
            "required_fractional_mse_reduction": 1.0 - ratio_needed,
            "required_s_bpw": -0.5 * math.log2(ratio_needed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--log2-n", type=int, default=16)
    parser.add_argument("--phase-bins", type=int, default=32)
    parser.add_argument("--public-seed", type=lambda x: int(x, 0), default=0x53494C57415250)
    parser.add_argument("--source-seed", type=lambda x: int(x, 0), default=0x46494E495445)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 10 <= args.log2_n <= 20:
        raise SystemExit("log2-n must be in [10,20]")
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
