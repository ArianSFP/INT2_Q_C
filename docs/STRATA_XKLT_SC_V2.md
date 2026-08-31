# STRATA-XKLT-SC v2 architecture

> **Blind-result status:** passed. This document describes the codec frozen
> before second-panel source access. The completed one-shot result and its
> independent audit are reported in [STRATA_V2_RESULTS.md](STRATA_V2_RESULTS.md).

## Objective and claim boundary

STRATA-XKLT-SC v2 is a weight-only post-training codec for a precommitted
18-matrix panel from `Qwen/Qwen3-30B-A3B` at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. It sees weights, but no
calibration activations, training examples, gradients, task loss, or
downstream evaluation data.

The frozen scientific gate is deliberately stricter than the earlier
within-5%-of-Gaussian POLARIS result:

```text
physical_bits * 20 <= 43 * source_weights
pooled source-domain MSE = sum(error^2) / sum(source^2) < 2^-4.3
complete blind lineage and every independent audit check must pass
```

Here `2^-4.3 = 0.050765774772264724`. The candidate panel contains
28,311,552 BF16 weights. It is one precommitted expert-panel experiment, not
a full-checkpoint, probability-sample, perplexity, inference-speed, or
universal rate-distortion claim.

## Name

- **STRATA**: natural 2,048-value groups are ranked by source energy and split
  into eight exactly equipopulous strata.
- **XKLT**: each expert's aligned `up_proj` and `down_proj.T` matrices receive
  one source-derived 2-by-2 cross-projection Karhunen-Loève-style transform.
- **SC**: each long stream block is quantized by a six-level polar lattice
  with causal successive-cancellation decisions and arithmetic coding.
- **v2**: the architecture replaces the first blind STRATA codec. The physical
  container's internal format version is separately numbered `1`.

## End-to-end data path

```text
six frozen Qwen expert triplets (gate, up, down)
        |
        +-- authenticate all 18 BF16 matrices and 108 original 2^18 blocks
        |
        +-- derive six quantized 2x2 KLTs from aligned up/down.T pairs
        |      gate is unchanged; transformed up/down are staged as BF16-RNE
        |
        +-- form 13,824 canonical natural groups of 2,048 values
        |
        +-- stable energy rank -> eight labels, exactly 1,728 groups each
        |
        +-- stable (label, canonical ordinal) stream order
        |
        +-- 13 blocks of 2^21 values + one tail block of 2^20 values
        |
        +-- exact DP chooses one q in [0,255] for each of 14 blocks
        |
        +-- signed deterministic RHT + RMS normalization
        |
        +-- procedural-Q31-BEC six-level POLARIS, MAP SC, arithmetic coding
        |
        +-- one fixed global byte reservoir; no retry or rate fallback
        |
        +-- independent parse, causal decode/re-encode, inverse RHT,
               canonical unordering, inverse KLT, original-BF16 FP64 score
```

All source-derived decisions—KLT angle codes, labels, block ordering, block
energies, profile IDs, staging hashes, and seeds—are sealed before the first
arithmetic encoder process starts.

## Source geometry

Each selected expert contributes:

| Matrix | Original shape | Natural grouping | Groups |
|---|---:|---|---:|
| `gate_proj.weight` | 768 × 2,048 | rows | 768 |
| `up_proj.weight` | 768 × 2,048 | rows | 768 |
| `down_proj.weight` | 2,048 × 768 | columns (`down.T` rows) | 768 |

Six triplets therefore produce `18 * 768 = 13,824` groups, each containing
2,048 values. Their total is 28,311,552 values. Every original matrix also
contains six contiguous `2^18` BF16 blocks, giving 108 source blocks for
hashing and lineage purposes. Those 108 blocks are reorganized into fourteen
long coding blocks; they are not fourteen conveniently selected source
blocks.

## Quantized cross-projection KLT

For one triplet, let `u = up_proj` and `d = down_proj.T`, converted exactly
from BF16 to FP32. Deterministic FP64 reductions compute

```text
A = sum(u*u)
B = sum(d*d)
C = sum(u*d)
theta = 0.5 * atan2(2*C, A-B)
code = clip(rint(theta/pi * 32768), -16384, 16384)
```

`rint` is ties-to-even. The decoder-visible coefficients are regenerated only
from the signed Q15-over-pi code:

```text
theta_q = code*pi/32768
c = FP32(cos(theta_q))
s = FP32(sin(theta_q))
```

The forward transform is

```text
z0 = c*u + s*d
z1 = -s*u + c*d.
```

Every FP32 multiplication and addition is rounded separately; fused
multiply-add contraction is forbidden. Both outputs are then rounded to BF16
round-to-nearest-even. CPU and CuPy staging words must agree exactly. Gate
groups remain unchanged, `z0` occupies the canonical up slot, and `z1` the
canonical down slot.

The header carries both each Q15 code and its two FP32 coefficients. A decoder
must regenerate the coefficients and compare their raw bit patterns, so the
encoder cannot hide an uncharged high-precision transform in the float
fields.

Because rounded FP32 `c` and `s` do not necessarily satisfy `c^2+s^2 == 1`
in FP64, the normative inverse is

```text
norm2 = FP64(c)^2 + FP64(s)^2
u_hat = (FP64(c)*z0_hat - FP64(s)*z1_hat) / norm2
d_hat = (FP64(s)*z0_hat + FP64(c)*z1_hat) / norm2.
```

Omitting `norm2` is nonconforming. The reconstructed `d_hat` is transposed
back to the original down-projection orientation before scoring.

## Energy strata and mixed long blocks

The staged BF16 value of every canonical group is widened to FP64, and its
squared energy is reduced deterministically. Groups are stable-sorted by
`(energy, canonical_group_ordinal)`. If `rank` is zero based, the label is

```text
label = floor(rank * 8 / 13824).
```

This guarantees exactly 1,728 labels of each value `0..7`. The raw labels cost
three physical bits per canonical group. Groups are then ordered by
`(label, canonical_group_ordinal)`. No full permutation is transmitted: the
literal labels and canonical route reproduce it.

The ordered stream is partitioned without gaps:

- blocks 0–12: `N = 2^21` values, 1,024 groups each;
- block 13: `N = 2^20` values, 512 groups.

Long blocks reduce finite-length polar loss and let rate follow source energy
across the sorted stream rather than assigning the same rate to all natural
groups.

## Frozen rate allocator

One profile byte `q` defines

```text
R(q) = 1.75 + q/256 bpw
D(q) = 2^(-2*R(q)),       q in [0,255].
```

An exact multiple-choice dynamic program minimizes the frozen proxy

```text
sum_b finite_factor(log2(N_b)) * energy_b * D(q_b)
```

with development-frozen finite factors

```text
finite_factor(20) = 1.0124498003545317
finite_factor(21) = 1.0107341453912242.
```

The DP has a 60,759,864-bit nominal profile budget. That is the physical
60,825,400-bit arithmetic reservoir less a fixed 65,536-bit no-retry reserve.
Updates use strict `<`, profiles are visited in ascending `q`, and the lowest
used-bit terminal state breaks the final tie. The reserve absorbs operational
codelength variance; it does not permit observing a payload and lowering its
profile. If the fourteen byte-padded arithmetic streams overflow the fixed
reservoir, the blind run fails.

The second-panel profile IDs are source-derived pre-encoding data sealed in
the allocation lock and physically carried by the final container directories.

## Polar-lattice block codec

Each long block uses:

- six Construction-D levels and a 64-point alphabet;
- lattice spacing `eta = 0.25`;
- source sigma 1 after FP64 RMS normalization;
- profile test distortion `D(q)`;
- a signed, orthonormal randomized Hadamard transform;
- MAP successive-cancellation decisions;
- causal binary arithmetic coding; and
- one encoder invocation, with no retry or resume.

Unlike the older decoder-map path, v2 constructs reliability flags from a
capacity-matched binary-erasure-channel surrogate. Capacities are converted
to unsigned Q31, the BEC polarization recursion is integer deterministic, and
ties are resolved by index. There is no external reliability table or
probability sidecar.

SC frozen-bit and RHT seeds are derived together from the complete sealed
control state and the block ordinal. This prevents a post-result seed search
and makes all randomness decoder-visible without transmitting a sign vector.
See [STRATA_V2_FORMAT.md](STRATA_V2_FORMAT.md) for the exact seed domain and
byte layout.

## Normative score

The encoder's normalized staging-domain metric is diagnostic only. The
scientific metric is recomputed independently after:

1. strict physical parsing and causal arithmetic decode;
2. reconstruction of all six polar levels;
3. inverse signed RHT using the independently derived seed;
4. undoing the label order back to canonical natural groups;
5. scaled-orthogonal inverse KLT; and
6. comparison with every original BF16 source value exactly once.

For matrices `m`, the pooled score is

```text
MSE = sum_m ||W_m - What_m||_F^2 / sum_m ||W_m||_F^2,
```

with source energy and squared error reduced in FP64. It is not the mean of
matrix-relative MSEs, the DP projection, or the encoder's staged metric.

## What changed from historical STRATA v1

Historical blind v1 used sorted-scale STRATA/POLARIS and emitted a valid
route-inclusive `2.14990912543403 bpw` artifact, but independently scored
`0.05166003144302383`, missing the strict Gaussian reference by 1.7615%.
That result remains negative.

V2 was developed only after v1 was opened. Its principal changes are:

- quantized cross-projection KLT for correlated up/down pairs;
- mixed `2^21`/`2^20` blocks rather than the earlier shorter organization;
- an exact energy-aware profile DP on a 1/256-bpw grid;
- procedural Q31-BEC construction rather than a stored reliability map;
- explicit source-derived staging/scale/allocator revalidation in the
  independent auditor; and
- stronger runtime, lineage, and tamper bindings.

These improvements were selected using already-opened development evidence.
The disjoint second panel supplied the one-shot confirmatory result: pooled
source-domain relative MSE `0.04985939119332436` at
`2.149999830457899 bpw`, passing the frozen strict gate. See the result card
for the complete hashes and limitations.

## Frozen implementation identities

The blind codec freeze binds the following normative files by SHA-256:

| Component | SHA-256 |
|---|---|
| `strata_v2_codec/common.py` | `bb5ad9f91ed6c4ee51f337d70fbfb3b1f174001cf2585d48c84034622b6b4ab8` |
| `strata_v2_codec/emit_and_lock.py` | `8fe43becd70eaf3d2bbc009db424db8c152029aa3f0e4bf4ceed7108f342ed40` |
| `strata_v2_codec/polar_encoder.py` | `ab4f7f6ed6f55f7eaa0bba24436b22a26cc0bf3452e1ce0d2ed56be805020309` |
| `strata_v2_codec/run_one_shot.py` | `764bacb3f6690ea5308560dc850c410a1dd59337a358152985fdf20d3b3f69fe` |
| `strata_v2_codec/FORMAT.md` | `004f195055e2289ff647ebadc69181978acbf61c19192d49ed3c509d6843023c` |
| base CuPy/RHT encoder (withheld from redistribution) | `062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0` |
| procedural Q31-BEC builder | `456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267` |
| independent auditor | `85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e` |
| lineage tamper harness | `91d47e7f5531da6bcf900c0263a29af049bd155d9e89b167b0b5494f937cfe05` |

The complete freeze also binds the selection/route artifacts, tests, freeze
builder and validator, Python interpreter, installed NumPy/SciPy/CuPy package
trees, CUDA runtime/driver, and RTX 5090 device identity.

The historical base encoder is hash-bound but absent from the compact public
tree because its source identifies a direct port of upstream MATLAB code for
which no redistribution license was visible. This reduces exact public
re-encoding portability, but does not remove the emitted container,
independent decoder/auditor, score evidence, or retained runtime hash binding.
