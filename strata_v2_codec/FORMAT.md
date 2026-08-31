# STRATA-XKLT-SC v2 physical format (candidate freeze contract)

This document fixes the byte-level candidate format before the second Qwen
panel is opened.  Except for the explicitly big-endian literal legacy route
records (`>HHBBH`), all multibyte integers and floating-point fields are little
endian.  The complete artifact is exactly 7,608,729 bytes (60,869,832 bits)
for 28,311,552 weights, or 2.149999830457899 bits/weight.

## Top-level layout

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 128 | fixed header |
| 128 | 144 | literal `>HHBBH` route records from the sealed proposal |
| 272 | 5,184 | 13,824 canonical 3-bit stratum labels, MSB first |
| 5,456 | 98 | fourteen 7-byte block-directory records |
| 5,554 | 7,603,175 | concatenated arithmetic streams and zero tail |

The literal route, rather than a compressed representation of it, is charged
to the physical bundle.  Labels are likewise stored raw rather than with the
development-only enumerative compressor.

## Header (128 bytes)

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `8s` | magic `PLRKLT2\0` |
| 8 | `u16` | physical-format version, exactly 1 |
| 10 | `u16` | header bytes, exactly 128 |
| 12 | `u32` | flags, exactly `0x0000007f` |
| 16 | `u32` | weight count, 28,311,552 |
| 20 | `u16` | natural-group length, 2,048 |
| 22 | `u16` | natural-group count, 13,824 |
| 24 | `u8` | arithmetic-block count, 14 |
| 25 | `u8` | leading `N=2^21` block count, 13 |
| 26 | `u8` | leading block log2, 21 |
| 27 | `u8` | tail block log2, 20 |
| 28 | `f32` | lattice eta, exactly 0.25 |
| 32 | `12*f32` | `(cos,sin)` coefficient pairs in triplet order |
| 80 | `6*i16` | source-derived Q15-over-pi KLT angle codes |
| 92 | `32s` | SHA-256 of the literal route bytes followed by label bytes |
| 124 | `u32` | CRC32 of header bytes 0 through 123 |

The seven flag bits state procedural Q31-BEC construction, causal arithmetic
coding, signed RHT, triplet KLT, FP32-to-BF16-RNE KLT staging, raw 3-bit
labels, and one global reservoir.  No other flag bit may be set.  Constants
such as the 65,536-bit allocation reserve and `R(q)=1.75+q/256` belong to
format version 1 and are validated by the frozen packer rather than duplicated
inside the header.

The six coefficient pairs correspond to selection triplets in route order.
Each triplet is `gate_proj`, `up_proj`, `down_proj`; the transform pairs
`up_proj` with `down_proj.T`.  The decoder consumes the stored FP32
coefficients directly.  A conforming unpacker regenerates each coefficient
pair from its angle code and requires the little-endian FP32 bit patterns to
match exactly.  It also rejects codes outside `[-16384, 16384]`.  This
redundant integrity check prevents the encoder from hiding uncharged
high-precision transform metadata.

## KLT derivation and staging

For a BF16 triplet, let `u=up_proj` and `d=down_proj.T`, converted exactly to
FP32.  FP64 reductions compute

```text
A = sum(u*u), B = sum(d*d), C = sum(u*d)
theta = 0.5 * atan2(2*C, A-B)
code = clip(rint(theta/pi * 32768), -16384, 16384)
theta_decoded = code*pi/32768
cosine = FP32(cos(theta_decoded))
sine   = FP32(sin(theta_decoded))
```

`rint` is ties-to-even.  The materializer stores `code`, `cosine`, and
`sine` in the header.  It rounds each FP32 multiplication separately, then
rounds each FP32 addition; fused multiply-add contraction is forbidden.  Both
completed components are then rounded to BF16-RNE:

```text
z0 = cosine*u + sine*d
z1 = -sine*u + cosine*d
```

After block decoding and canonical unordering, the decoder/scorer converts
the reconstructed components to FP64 and applies the exact scaled-orthogonal
inverse using the stored FP32 coefficients converted exactly to FP64:

```text
norm2 = FP64(cosine)^2 + FP64(sine)^2
u_hat = (FP64(cosine)*z0_hat - FP64(sine)*z1_hat) / norm2
d_hat = (FP64(sine)*z0_hat + FP64(cosine)*z1_hat) / norm2
```

Omitting `norm2` is nonconforming even though its deviation from one is small.
The reconstructed down component is transposed back to its original matrix
orientation after this inverse.

Gate groups remain unchanged.  Canonical group order is matrix route order,
then natural row for gate/up or natural column for down.  Thus the `z0` groups
occupy the up slot and `z1` groups occupy the down slot.

Each transformed BF16 group's FP64 squared energy is ranked with a stable
canonical-ordinal tie break.  Its label is `floor(rank*8/13824)`, producing
exactly 1,728 groups per label.  Stable sorting by `(label, canonical ordinal)`
reconstructs the stream order without a permutation sidecar.  The first
thirteen blocks contain 1,024 groups each; the final block contains 512.

## Allocation frozen before encoding

Profile byte `q` means

```text
R(q) = 1.75 + q/256 bpw
D(q) = 2^(-2*R(q))
q in [0,255]
```

The procedural Q31 BEC construction regenerates all freeze sets from `D(q)`;
there is no stored reliability table.  A multiple-choice DP minimizes

```text
sum_b finite_factor(log2(N_b)) * block_energy_b * 2^(-2*R(q_b))
```

using the fixed development constants

```text
finite_factor(20) = 1.0124498003545317
finite_factor(21) = 1.0107341453912242
```

subject to at most 60,759,864 nominal profile bits, which is the 60,825,400-bit
reservoir less the sealed 65,536-bit blind reserve.  Strict-less DP updates,
ascending profile IDs, and the lowest used-bit terminal state resolve ties.
The resulting fourteen profiles and all staging hashes are sealed before any
arithmetic block is encoded.

The reserve is not an invitation to adapt after encoding.  Every blind block
is encoded exactly once.  If the concatenated byte-padded streams exceed the
reservoir, the rate gate fails; there is no retry, profile decrement, or
post-result architecture change.

## Directory and reservoir

Directory record `b` is `struct <BeI`:

```text
u8     profile q_b
f16    decoder block scale
u32    arithmetic logical length in bits
```

The decoder scale is the encoder's FP64 post-transform RMS rounded once to
IEEE binary16 ties-to-even.  Streams appear in block order.  Stream `b`
occupies `ceil(logical_bits_b/8)` bytes; the next stream begins immediately.
All unused reservoir bytes are zero and are physically present.  The decoder
regenerates causal frequencies, so no probability or decision sidecar exists.

SC frozen-bit and signed-RHT seeds are derived together from

```text
SHA256(
  b"POLARIS-STRATA-V2-KLT-MIXED-SEED-v1\0" ||
  exact_header || exact_route || exact_labels || profile_bytes ||
  u8(block_ordinal)
)
```

where `profile_bytes` is the fourteen profile IDs in block order.  Logical
lengths and scales are deliberately excluded, avoiding a seed cycle.
The SC seed is the first four digest bytes interpreted big endian, with zero
replaced by one.  The RHT seed is the next eight digest bytes interpreted big
endian.

## Exact rate ledger

```text
header                    1,024 bits
literal route             1,152 bits
raw labels               41,472 bits
14 directories              784 bits
global reservoir     60,825,400 bits
------------------------------------------------
physical total       60,869,832 bits
integer 2.15 cap     60,869,836 bits
headroom                       4 bits
```

The pass/fail metric is pooled source-domain FP64 SSE divided by pooled BF16
source energy, with every original value scored exactly once after inverse
RHT, stream unordering, and inverse KLT.
