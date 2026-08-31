# STRATA-XKLT-SC v2 physical format

> **Status:** format frozen before second-panel source access and instantiated
> by the passing blind artifact. The final container SHA-256 is
> `e89e2a97fa655cc4248de849ba1b1b84b46e8357044beb563318efb359be2be7`;
> its observed logical arithmetic payload is 60,768,579 bits.

The architecture is called STRATA-XKLT-SC v2. Its on-disk format is version
`1`, identified by magic `PLRKLT2\0`. Except for the literal route records,
all multibyte fields are little endian.

## Fixed geometry and top-level layout

For 28,311,552 source weights, a conforming artifact is exactly 7,608,729
bytes:

| Offset | Bytes | Bits | Section |
|---:|---:|---:|---|
| 0 | 128 | 1,024 | fixed header |
| 128 | 144 | 1,152 | 18 literal big-endian route records |
| 272 | 5,184 | 41,472 | 13,824 raw 3-bit labels, MSB first |
| 5,456 | 98 | 784 | fourteen 7-byte directories |
| 5,554 | 7,603,175 | 60,825,400 | arithmetic streams plus zero tail |
| **total** | **7,608,729** | **60,869,832** | |

The route, labels, directories, byte padding, and unused global reservoir are
all charged. There is no entropy-estimated rate in the primary rate claim.

## Header

The 128-byte header is:

| Offset | Type | Required meaning |
|---:|---|---|
| 0 | `8s` | magic `PLRKLT2\0` |
| 8 | `u16` | format version `1` |
| 10 | `u16` | header bytes `128` |
| 12 | `u32` | flags `0x0000007f` |
| 16 | `u32` | weight count `28,311,552` |
| 20 | `u16` | natural group length `2,048` |
| 22 | `u16` | natural group count `13,824` |
| 24 | `u8` | arithmetic block count `14` |
| 25 | `u8` | leading `2^21` block count `13` |
| 26 | `u8` | leading block log2 `21` |
| 27 | `u8` | tail block log2 `20` |
| 28 | `f32` | lattice spacing `0.25` |
| 32 | `12*f32` | six `(cosine,sine)` KLT pairs |
| 80 | `6*i16` | six Q15-over-pi angle codes |
| 92 | `32s` | SHA-256 of `route || labels` |
| 124 | `u32` | CRC32 of header bytes 0–123 |

The seven flags declare procedural Q31-BEC construction, causal arithmetic
coding, signed RHT, triplet KLT, FP32-to-BF16-RNE KLT staging, raw 3-bit
labels, and one global reservoir. Every other bit must be zero.

Each stored coefficient pair must have the exact IEEE binary32 bit pattern
obtained from its Q15 code:

```text
theta = code*pi/32768
expected = (FP32(cos(theta)), FP32(sin(theta))).
```

Codes outside `[-16384,16384]`, coefficient mismatches, route/label hash
mismatches, and CRC mismatches are hard format failures.

## Literal route

The route contains eighteen `>HHBBH` records, one for each matrix:

```text
u16be layer
u16be expert
u8    role: 0=gate, 1=up, 2=down
u8    natural axis: 0=row, 1=column
u16be group count, exactly 768
```

Records are six consecutive `(gate,up,down)` triplets. Gate and up use the
row axis; down uses the column axis. All three records in a triplet must share
the same layer and expert. The route is stored literally and costs 1,152 bits.

## Labels

There is one unsigned 3-bit label for every canonical natural group. Bits are
packed MSB first, with no padding because `13,824 * 3 = 41,472` is byte
aligned. A conforming label histogram is exactly

```text
[1728, 1728, 1728, 1728, 1728, 1728, 1728, 1728].
```

The decoder reconstructs stream order with a stable sort by
`(label, canonical_group_ordinal)`. It must reject a label set that is not
equipopulous or does not form the sealed source-derived ordering.

## Profile IDs and block geometry

Fourteen profile bytes are not stored in a separate top-level section; each
is the first byte of its directory record. Profile `q` means

```text
rate(q) = 1.75 + q/256 bpw
test_distortion(q) = 2^(-2*rate(q)).
```

Blocks 0–12 each have `N=2^21`; block 13 has `N=2^20`. The complete profile
byte string participates in every SC/RHT seed derivation.

## Directory

Directory record `b` is the seven-byte little-endian struct `<BeI`:

| Bytes | Type | Meaning |
|---:|---|---|
| 1 | `u8` | profile `q_b` |
| 2 | IEEE binary16 | decoder scale |
| 4 | `u32le` | logical arithmetic length in bits |

The scale is the block's FP64 post-RHT RMS rounded once to binary16
ties-to-even. The independent auditor recomputes the RHT from the exact
staging words and requires its expected two bytes to match the directory
literally.

## Arithmetic reservoir

The fourteen causal arithmetic payloads appear in block order. Stream `b`
occupies exactly `ceil(logical_bits_b/8)` bytes. If a logical stream ends
inside a byte, its unused low-order bits must be zero. The next stream starts
at the next byte; streams are not bit-concatenated across this boundary.

After the last padded stream, every remaining byte of the 7,603,175-byte
reservoir must be zero. A parser must enforce both the per-stream low-bit
padding and the global all-zero tail. Logical lengths may not cause any
window to overlap the next stream or leave the physical reservoir.

No decisions, causal probabilities, RHT signs, frozen-set tables, KLT
permutation, or reconstruction indices are stored. They are regenerated from
the header, route, labels, profiles, seeds, and procedural construction.

## Seed derivation

For block ordinal `b`, compute

```text
digest = SHA256(
    b"POLARIS-STRATA-V2-KLT-MIXED-SEED-v1\0" ||
    exact_128_byte_header ||
    exact_144_byte_route ||
    exact_5184_byte_labels ||
    exact_14_profile_bytes ||
    u8(b)
)
```

The SC seed is `digest[0:4]` interpreted big endian, replacing zero with one.
The RHT seed is `digest[4:12]` interpreted big endian. Logical lengths and
decoder scales are excluded to avoid a circular dependency. A decoder must
derive seeds independently; reading an encoder-side seed sidecar is not
conforming.

## Exact rate gate

The frozen ledger is:

| Section | Bits |
|---|---:|
| Header | 1,024 |
| Literal route | 1,152 |
| Raw labels | 41,472 |
| Fourteen directories | 784 |
| Global reservoir | 60,825,400 |
| **Physical total** | **60,869,832** |

Because `2.15 = 43/20`, the exact integer gate is

```text
physical_bits * 20 <= 43 * 28,311,552.
```

The right-hand integral floor is 60,869,836 bits, so the frozen format has
four bits of headroom and a rate of `2.149999830457899 bpw`. That arithmetic
does not establish that the encoded logical streams fit. Reservoir fit and
zero-tail validation remain separate mandatory gates in the final artifact.

## Independent parser obligations

A release parser should fail closed unless it independently verifies:

- total file size, magic, version, flags, constants, CRC, and section offsets;
- KLT coefficient regeneration from all six Q15 codes;
- route geometry, route/label hash, and exact label histogram;
- all fourteen profile values and independently derived seeds;
- directory logical bounds and raw FP16 scale bytes;
- zero low bits in every padded arithmetic byte;
- a zero-filled unused reservoir tail;
- exact physical integer rate; and
- causal decode followed by canonical arithmetic re-encode with the same
  logical length and every payload byte identical.

The final source-domain distortion is not a container-header field. It must be
recomputed from authenticated original BF16 sources after inverse RHT,
unordering, and inverse KLT.
