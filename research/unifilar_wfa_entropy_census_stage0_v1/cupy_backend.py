#!/usr/bin/env python3
"""CuPy RawKernel backend for the frozen sparse unifilar bank.

This file must only be imported after external source-manifest bootstrap and
independent-review authentication.  It intentionally supports CuPy only: no
Torch, JAX, NumPy fitting path, or dense chi-by-chi contraction exists here.
Each CUDA thread processes one canonical stream sequentially.  Across a fixed
150-cell topology bank, work is O(N) in the number of selected symbols.
"""

from __future__ import annotations

from typing import Any


CUDA_SOURCE = r'''
extern "C" {

__device__ __forceinline__ unsigned int mix32(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    return value ^ (value >> 16);
}

__device__ __forceinline__ unsigned int context_of(
    unsigned char level, unsigned short base_freq, unsigned long long within
) {
    unsigned int bucket = (((unsigned int)base_freq) * 16u) >> 16;
    if (bucket > 15u) bucket = 15u;
    return ((((unsigned int)level) * 16u + bucket) * 4u) + ((unsigned int)within & 3u);
}

__device__ __forceinline__ unsigned int next_state(
    int topology,
    unsigned int states,
    unsigned int state,
    unsigned int bit,
    unsigned int context,
    unsigned long long within
) {
    const unsigned int mask = states - 1u;
    if (topology == 0) {
        return ((state << 1) | bit) & mask;
    }
    if (topology == 1) {
        if (bit == 0u) return state;
        unsigned int sketch = mix32(0xA511E9B3u ^ context ^ ((unsigned int)within * 0x9E3779B1u)) & mask;
        if (sketch == 0u) sketch = 1u;
        return state ^ sketch;
    }
    if (topology == 2) {
        unsigned int weight = (mix32(0x63D83595u ^ context ^ (((unsigned int)within & 3u) << 20)) & mask) | 1u;
        return (state + weight * bit) & mask;
    }
    if (topology == 3) {
        unsigned int multiplier = (states >= 8u ? 5u : 1u) & mask;
        unsigned int addend = mix32(0xB5297A4Du ^ context ^ (((unsigned int)within & 3u) << 24)) & mask;
        return (multiplier * state + addend + bit) & mask;
    }
    if (bit) return state < mask ? state + 1u : mask;
    return state > 0u ? state - 1u : 0u;
}

__global__ void count_streams(
    const unsigned char* bits,
    const unsigned char* levels,
    const unsigned short* base_freq,
    const unsigned long long* offsets,
    const unsigned long long* lengths,
    unsigned int stream_count,
    int topology,
    unsigned int states,
    unsigned int reset_length,
    unsigned long long* counts
) {
    const unsigned int stream = blockDim.x * blockIdx.x + threadIdx.x;
    if (stream >= stream_count) return;
    const unsigned long long start = offsets[stream];
    const unsigned long long length = lengths[stream];
    unsigned int state = 0u;
    for (unsigned long long position = 0; position < length; ++position) {
        const unsigned long long within = position % (unsigned long long)reset_length;
        if (within == 0ull) state = 0u;
        const unsigned long long index = start + position;
        const unsigned int bit = (unsigned int)bits[index];
        const unsigned int context = context_of(levels[index], base_freq[index], within);
        const unsigned long long count_index = (((unsigned long long)state * 384ull + context) << 1) + bit;
        atomicAdd(&counts[count_index], 1ull);
        state = next_state(topology, states, state, bit, context, within);
    }
}

__global__ void exact_arithmetic_lengths(
    const unsigned char* bits,
    const unsigned char* levels,
    const unsigned short* base_freq,
    const unsigned long long* offsets,
    const unsigned long long* lengths,
    unsigned int stream_count,
    int topology,
    unsigned int states,
    unsigned int reset_length,
    const unsigned short* frequencies,
    unsigned long long* output_lengths
) {
    const unsigned int stream = blockDim.x * blockIdx.x + threadIdx.x;
    if (stream >= stream_count) return;
    const unsigned long long start = offsets[stream];
    const unsigned long long length = lengths[stream];
    unsigned int state = 0u;
    unsigned long long low = 0ull;
    unsigned long long high = 0xFFFFFFFFull;
    unsigned long long pending = 0ull;
    unsigned long long emitted = 0ull;
    for (unsigned long long position = 0; position < length; ++position) {
        const unsigned long long within = position % (unsigned long long)reset_length;
        if (within == 0ull) state = 0u;
        const unsigned long long index = start + position;
        const unsigned int bit = (unsigned int)bits[index];
        const unsigned int context = context_of(levels[index], base_freq[index], within);
        const unsigned int f1 = (unsigned int)frequencies[(unsigned long long)state * 384ull + context];
        const unsigned int f0 = 65536u - f1;
        const unsigned long long width = high - low + 1ull;
        const unsigned long long split = low + (width * (unsigned long long)f0 / 65536ull) - 1ull;
        if (bit == 0u) high = split; else low = split + 1ull;
        while (true) {
            if (high < 0x80000000ull) {
                emitted += 1ull + pending;
                pending = 0ull;
            } else if (low >= 0x80000000ull) {
                emitted += 1ull + pending;
                pending = 0ull;
                low -= 0x80000000ull;
                high -= 0x80000000ull;
            } else if (low >= 0x40000000ull && high < 0xC0000000ull) {
                pending += 1ull;
                low -= 0x40000000ull;
                high -= 0x40000000ull;
            } else {
                break;
            }
            low = (low << 1) & 0xFFFFFFFFull;
            high = ((high << 1) & 0xFFFFFFFFull) | 1ull;
        }
        state = next_state(topology, states, state, bit, context, within);
    }
    // The canonical encoder performs pending += 1 followed by one emit.
    output_lengths[stream] = emitted + pending + 2ull;
}

} // extern C
'''


def _kernels(cp: Any) -> tuple[Any, Any]:
    module = cp.RawModule(
        code=CUDA_SOURCE,
        options=("--std=c++11",),
        name_expressions=("count_streams", "exact_arithmetic_lengths"),
    )
    return module.get_function("count_streams"), module.get_function("exact_arithmetic_lengths")


class CuPyUnifilarBackend:
    """Exact integer fitting and arithmetic-length scoring on CUDA."""

    def __init__(self, cp: Any):
        self.cp = cp
        self.count_kernel, self.length_kernel = _kernels(cp)

    def pack_streams(self, streams: list[tuple[bytes, bytes, bytes]]) -> dict[str, Any]:
        """Pack (bits-u8, levels-u8, base-frequency-u16le) without NumPy."""
        cp = self.cp
        offsets: list[int] = []
        lengths: list[int] = []
        bit_parts: list[bytes] = []
        level_parts: list[bytes] = []
        frequency_parts: list[bytes] = []
        offset = 0
        for bits, levels, base_frequency in streams:
            if not bits or len(bits) != len(levels) or len(base_frequency) != 2 * len(bits):
                raise ValueError("packed stream geometry")
            offsets.append(offset)
            lengths.append(len(bits))
            offset += len(bits)
            bit_parts.append(bits)
            level_parts.append(levels)
            frequency_parts.append(base_frequency)
        if not streams:
            raise ValueError("empty stream pack")
        return {
            "bits": cp.frombuffer(b"".join(bit_parts), dtype=cp.uint8).copy(),
            "levels": cp.frombuffer(b"".join(level_parts), dtype=cp.uint8).copy(),
            "base_freq": cp.frombuffer(b"".join(frequency_parts), dtype=cp.uint16).copy(),
            "offsets": cp.asarray(offsets, dtype=cp.uint64),
            "lengths": cp.asarray(lengths, dtype=cp.uint64),
            "stream_count": len(streams),
            "symbol_count": offset,
        }

    def fit_counts(self, packed: dict[str, Any], topology_id: int, states: int, reset_length: int) -> Any:
        cp = self.cp
        counts = cp.zeros(states * 384 * 2, dtype=cp.uint64)
        stream_count = int(packed["stream_count"])
        threads = 128
        blocks = (stream_count + threads - 1) // threads
        self.count_kernel(
            (blocks,),
            (threads,),
            (
                packed["bits"], packed["levels"], packed["base_freq"],
                packed["offsets"], packed["lengths"], cp.uint32(stream_count),
                cp.int32(topology_id), cp.uint32(states), cp.uint32(reset_length), counts,
            ),
        )
        return counts

    def exact_lengths(
        self,
        packed: dict[str, Any],
        topology_id: int,
        states: int,
        reset_length: int,
        frequencies: Any,
    ) -> Any:
        cp = self.cp
        model = cp.asarray(frequencies, dtype=cp.uint16)
        if int(model.size) != states * 384:
            raise ValueError("GPU frequency geometry")
        stream_count = int(packed["stream_count"])
        result = cp.zeros(stream_count, dtype=cp.uint64)
        threads = 128
        blocks = (stream_count + threads - 1) // threads
        self.length_kernel(
            (blocks,),
            (threads,),
            (
                packed["bits"], packed["levels"], packed["base_freq"],
                packed["offsets"], packed["lengths"], cp.uint32(stream_count),
                cp.int32(topology_id), cp.uint32(states), cp.uint32(reset_length), model, result,
            ),
        )
        return result


def build_backend(cp: Any) -> CuPyUnifilarBackend:
    return CuPyUnifilarBackend(cp)
