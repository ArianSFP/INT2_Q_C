# STRATA expert-affine N20/N21 checkpoint format

This format is a locality-oriented fork of the frozen STRATA-XKLT-SC v2
artifact. It leaves that historical format and its evidence unchanged. The
quantization payload primitive remains the same six-level, procedural-Q31-BEC,
MAP successive-cancellation polar-lattice codec with a deterministic signed
RHT and causal arithmetic coding.

The format changes the coding-block ownership to match MoE inference. Two
`N=2^21` blocks belong privately to each expert. The remaining 256 natural
groups of two adjacent experts are combined into one `N=2^20` tail block.

## Geometry

One expert contains three matrices and 4,718,592 values:

```text
gate  768 x 2048
up    768 x 2048
down  2048 x 768, grouped as down.T rows
```

The six-expert panel is divided into fifteen coding blocks:

```text
blocks 0..11:  N=2^21, two private blocks per expert
blocks 12..14: N=2^20, paired tails for experts (0,1), (2,3), (4,5)
```

The existing global three-bit energy labels remain literal and unchanged.
Inside expert `e`, canonical ordinals `[2304e,2304(e+1))` are stable-sorted by
`(label, canonical ordinal)`. The first 1,024 and next 1,024 groups form its
private blocks; the final 256 groups form its paired tail. Every group appears
exactly once.

## Physical layout

The artifact is exactly 8,847,360 bytes, or exactly 2.5 bits per weight:

| Offset | Bytes | Section |
|---:|---:|---|
| 0 | 128 | fixed header |
| 128 | 144 | eighteen literal route records |
| 272 | 5,184 | 13,824 raw three-bit labels |
| 5,456 | 105 | fifteen seven-byte directories |
| 5,561 | 8,841,799 | arithmetic streams and zero tail |

Each directory is `<BeI`:

- `u8` profile identifier `q`;
- IEEE binary16 decoder scale;
- `u32le` logical arithmetic length in bits.

The byte-padded streams occur in block order. Unused low bits in the terminal
payload byte and every byte after the last stream must be zero.

The header magic is `PLRLOC3\0`. It carries the panel geometry, six Q15-over-pi
KLT codes, the corresponding twelve regenerated FP32 coefficients, a SHA-256
binding of route and labels, and CRC32. A distinct format magic and seed domain
ensure that a frozen-v2 decoder fails closed on this fork.

The compact checkpoint additionally binds every physical arithmetic slice to
its one-shot encoder metadata by logical length and payload SHA-256. Rebuilding
the legacy `<u32 logical_bits, f32 scale, payload>` record must reproduce the
sealed per-block literal-container SHA-256. The independent audit is bound to
the pre-encoding plan lock and to a canonical digest of all source rows.

## Rate allocation

The exact physical cap is 70,778,880 bits. The fixed reservoir is charged in
full. A 65,536-bit no-retry reserve leaves a nominal profile budget of
70,668,856 bits. Profiles use

```text
R(q) = 1.75 + q/256, q in [0,255].
```

An exact multiple-choice dynamic program assigns one profile per block using
the frozen N20/N21 finite-length factors. Allocation, labels, block mapping,
staging hashes, profiles, and seeds are sealed before arithmetic encoding.

## Per-expert reads

Expert `e` requires blocks:

```text
2e, 2e+1, 12+floor(e/2).
```

The coefficient-volume amplification is therefore

```text
(2*2^21 + 2^20) / (2*2^21 + 2^19) = 10/9.
```

The release reports exact payload-byte, cold-prefix, and distinct-4-KiB-page
amplification from the emitted lengths. The mandatory checkpoint gate is a
maximum 4-KiB-rounded amplification below 2x.

## Scientific gates

The first checkpoint requires independently decoded, original-BF16-domain MSE
no worse than `0.04985939119332436`, exact physical rate at or below 2.5 bpw,
and maximum per-expert reads below 2x.

The later research objective is rate-relative. For the artifact's actual
physical rate `R`, success requires

```text
MSE <= 0.8 * 2^(-2R).
```

At exactly 2.5 bpw that target is `0.025`.
