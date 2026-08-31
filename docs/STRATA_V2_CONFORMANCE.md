# STRATA-XKLT-SC v2 conformance appendix

> **Status:** post-run explanatory appendix. This page makes conventions in
> the frozen Python implementations easier to audit. It was written after the
> blind run and is **not** part of the pre-access normative freeze, does not
> amend format version 1, and cannot be used to repair or reinterpret the
> result. If this prose and a freeze-bound implementation disagree, the exact
> frozen source and [`strata_v2_codec/FORMAT.md`](../strata_v2_codec/FORMAT.md)
> control the historical artifact.

The reference identities relevant to this appendix are:

| Frozen implementation | SHA-256 |
|---|---|
| `agent_polaris_qwen_rht_encoder.py` (withheld; upstream-derived) | `062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0` |
| [`bg_codec_bec_encoder.py`](../bg_codec_bec_encoder.py) | `456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267` |
| [`strata_v2_codec/common.py`](../strata_v2_codec/common.py) | `bb5ad9f91ed6c4ee51f337d70fbfb3b1f174001cf2585d48c84034622b6b4ab8` |
| [`strata_v2_klt_mixed_independent_auditor_v1.py`](../strata_v2_klt_mixed_independent_auditor_v1.py) | `85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e` |

The base encoder is named here because it was part of the historical frozen
execution. It is not redistributed: its comments identify a direct port of an
upstream MATLAB implementation that supplied no visible license grant. The
freeze, runtime intent, and release verifier retain its exact hash binding.

## Arithmetic stream bit convention

The causal binary arithmetic coder uses an inclusive unsigned 32-bit interval
and a fixed total frequency of 65,536. For each symbol, `f1` is in
`[1,65535]`, `f0 = 65536-f1`, and

```text
width = high-low+1
split = low + floor(width*f0/65536) - 1
symbol 0 -> [low, split]
symbol 1 -> [split+1, high].
```

Renormalization is the conventional E1/E2/E3 sequence at `2^31`, `2^30`,
and `3*2^30`. Finalization increments the pending-bit count once and emits
zero when `low < 2^30`, otherwise one, followed by the complementary pending
bits. This is the canonical encoding checked by the independent re-encoder;
an alternative bitstream that happens to decode to the same symbols is not a
match.

Logical bits are packed **most-significant bit first** within each byte. Bit
position zero is `(byte >> 7) & 1`; position seven is `byte & 1`. A decoder
initializes its 32-bit code register from the first 32 logical positions and
returns virtual zero bits after the declared logical end.

Each block nevertheless occupies `ceil(logical_bits/8)` physical bytes. If
the logical end is not byte aligned, the unused positions are the **low**, or
least-significant, bits of the final byte and must be literal zero. The next
block begins at the next byte, not immediately after the previous logical
bit. The reservoir after the fourteenth byte-padded stream must also be all
zero. The label stream uses the same within-byte MSB-first convention, three
bits per label from label bit 2 down to bit 0.

Selected arithmetic symbols are concatenated in this order:

1. polar level 1 through level 6;
2. within a level, internal SC index 0 through `N-1`; and
3. only indices whose internal freeze flag is zero.

The matching `f1` frequencies use exactly the same order.

## Unsigned-Q31 BEC ordering

For each level capacity `C`, the frozen construction computes

```text
FULL = 2^31
Cq   = clamp(round(C*FULL), 0, FULL)
z0   = FULL-Cq
```

in the freeze-bound Python/NumPy runtime. `round` is Python's ties-to-even
rounding. An unsigned-64 work array initially contains `z0` at every leaf.
For butterfly widths 1, 2, 4, ..., `N/2`, each `(left,right)` pair becomes

```text
product = (left*right + 2^30) >> 31
minus   = left + right - product
plus    = product.
```

The left half receives `minus`; the right half receives `plus`. Let
`keep = ceil(N*C)`, clamped to `[0,N]`. Synthesized channels are sorted
lexicographically by ascending `(z_score, canonical_external_index)`, so the
lowest score wins every reliability tie. The first `keep` external positions
receive flag zero (selected/information); all others receive flag one
(frozen). The external flag vector is then indexed by the exact bit-reversal
permutation to produce internal SC order.

No floating comparison orders the synthesized channels after `C` has been
converted to Q31: scores and tie indices are integers. The FP64-derived `C`
and `ceil(N*C)` remain reference-runtime-defined as described below.

## SC `f`/`g`, transform, and reconstruction order

Likelihood ratios are `LR = p0/p1` and are clipped to
`[1e-30,1e30]`. The reference SC recursion uses

```text
f(a,b)   = (a*b + 1)/(a+b)
g(a,b,u) = a^(1-2u) * b.
```

SC visits internal indices in ascending order. At a frozen index it uses
`frozen_external[bit_reverse(internal_index)]`. At a selected index it forms
`p1 = 1/(1+root_LR)`, quantizes

```text
f1 = clamp(floor(p1*65536 + 0.5), 1, 65535),
```

then consumes one arithmetic symbol. Partial sums are updated by interleaving
`left XOR right` and `right`, exactly as implemented in the linked auditor.
The six frozen external bit vectors come from NumPy `default_rng` seeded by
`sc_seed + 1,000,003*level`, with levels numbered 1 through 6.

After one level, internal bits are mapped to external order by bit reversal
and passed through the polar transform. The transform performs in-place XOR
butterflies at strides 1, 2, 4, ..., `N/2`, replacing each left half by
`left XOR right` and retaining the right half. There is no extra bit reversal
after this transform.

Reconstruction proceeds least-significant plane first:

```text
index = x_level_1
      + 2*x_level_2
      + 4*x_level_3
      + 8*x_level_4
      + 16*x_level_5
      + 32*x_level_6
value = eta * [-31, -30, ..., 31, 32][index].
```

Each completed lower-level reconstruction index is the causal context for
the next level. This level-major, internal-index-major order is also why the
auditor can regenerate every frequency without reading an encoder trace.

## SplitMix64 signs and RHT convention

For flat index `i`, all integer operations below wrap modulo `2^64`:

```text
z = seed + i + 0x9E3779B97F4A7C15
z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z XOR (z >> 27)) * 0x94D049BB133111EB
z =  z XOR (z >> 31)
sign[i] = +1 if (z & 1) == 0 else -1.
```

The unpermuted Walsh-Hadamard butterflies use widths 1, 2, 4, ..., `N/2`
and map each adjacent pair of halves to `(left+right, left-right)`. Forward
and inverse conventions are

```text
forward(x) = H * diag(sign) * x / sqrt(N)
inverse(y) = diag(sign) * H * y / sqrt(N).
```

Thus signs are applied before the forward butterflies and after the inverse
butterflies. Scaling is applied once after the butterflies. There is no
Walsh-order permutation or bit reversal in the RHT itself.

## Header flag bits

The 32-bit little-endian header word must equal `0x0000007f`. The ordered
feature description in the frozen format expands as follows for audit tools:

| Bit | Mask | Declared feature |
|---:|---:|---|
| 0 | `0x01` | procedural unsigned-Q31 BEC construction |
| 1 | `0x02` | causal binary arithmetic coding |
| 2 | `0x04` | signed deterministic RHT |
| 3 | `0x08` | triplet cross-projection KLT |
| 4 | `0x10` | separate-operation FP32 KLT followed by BF16-RNE staging |
| 5 | `0x20` | literal raw 3-bit stratum labels |
| 6 | `0x40` | one fixed global arithmetic reservoir |

These are not optional feature-negotiation switches in version 1. The frozen
packer and parser compare the complete word to `0x7f`; clearing one bit or
setting any higher bit is nonconforming. This per-bit table is an explanatory
expansion, not a post-run extension of the frozen format.

## Reference-runtime-defined numerical behavior

The byte layout and integer arithmetic above are explicit, but the historical
codec is not specified as a platform-independent correctly-rounded numerical
standard. Several source-derived decisions use the behavior and operation
order of the exact frozen Python stack:

- FP64 reductions and `atan2` for KLT angle derivation;
- Python/libm `pi`, `cos`, `sin`, `sqrt`, `exp2`, and `pow`;
- NumPy exponential, logarithm, summation, sorting, and RNG behavior used for
  periodic-channel capacities, likelihood ratios, and frozen bits;
- CuPy FP64 reductions and ordered Hadamard butterflies for RHT staging and
  RMS; and
- FP32 and BF16 ties-to-even conversions, with the explicitly separate KLT
  multiplications/additions and no FMA contraction.

The canonical environment was Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
CuPy 14.2.0 on CUDA runtime 12.9 / driver API 13.0 and an RTX 5090. The
[`codec freeze`](../blind_protocol_v2/codec_freeze.lock.json) binds the
interpreter and complete package trees, not only those version strings.

An independent implementation may be scientifically equivalent without
being bit-identical at every intermediate FP64 value. It must not claim a
bit-exact replay unless it reproduces the stored KLT coefficient words,
source-derived labels/profiles/scales, causal frequencies, logical lengths,
and canonical payload bytes. For the historical result, the linked frozen
Python and the hash-bound physical/audit artifacts are the executable
reference wherever this appendix leaves numerical behavior to FP64, libm,
NumPy, SciPy, CuPy, or CUDA.
