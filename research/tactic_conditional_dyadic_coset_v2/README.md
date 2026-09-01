# TACTIC-DH384 v2: decoder-conditioned dyadic coset gate

Status: **source-only frozen candidate**.  This package does not authorize a
model read, a coarse encode, a CUDA launch, or a compression claim.  The only
permitted first actions are the standard-library source verification, an
independent source review, and then the synthetic-only CuPy preflight.

## The one frozen cell

TACTIC-DH384 is a decoder-visible refinement of a literal lower-rate finite
coarse stream.  It does not predict raw weights from another expert.  Each
expert first decodes eighteen fixed-capacity `N=2^18` streams at
`307/128 = 2.3984375` bits per weight.  Each stream is split into 64 contiguous
4,096-value blocks.  For a decoded block `y` and its canonical signed integer
coarse symbols, the decoder constructs one orthogonal matrix `Q_theta(y)` from
twelve stages of signed/swapped two-point Hadamards.  Every routing decision is
a deterministic function of those coarse symbols, the role, the stage, and a
charged 16-KiB selector packet.

The first 384 columns form

```text
B_theta(y) = Q_theta(y)[:, 0:384].
```

The expert frame contains exactly 48 target bytes per block.  A fixed QC/
trellis decoder maps those 384 bits to one bounded Q12 coefficient vector
`a`.  The emitted correction is `B_theta(y) a`.

This v2 deliberately narrows the earlier source-only TACTIC-v1 sketch.  V1
mentioned saturating lifting and intermediate rounding while also claiming a
continuous linear-span dominance bound.  Those statements are not generally
compatible: rounding or saturation can move a finite output outside the real
span of decoded unit columns.  V2 uses only exact integer sums, differences,
signs and swaps, proves the accumulator bound, and applies one public dyadic
scale after the network.  For fixed `y`, the map from `a` to the rational
correction is exactly linear.  The stage-0 continuous span therefore really
contains every finite v2 correction.

## Exact conditional butterfly

For each stage, 4,096 coordinates are partitioned into the standard radix-2
perfect matching.  A pair's eight-bit feature contains role, both signs,
relative magnitude, and three comparisons to the block's integer mean
absolute symbol.  The `12 x 256` byte selector table returns one of eight
signed/swap variants.  After the optional swap and two optional sign changes,
the pair map is

```text
(u, v) -> (u + v, u - v).
```

For fixed decoder-visible decisions, every stage divided by `sqrt(2)` is
orthogonal.  Twelve stages have the single exact normalization `1/64`, so
`Q_theta(y)^T Q_theta(y)=I`.  No decision depends on the target source,
residual, tangent bits, another expert, router activations, or an unserialized
value.  With Q12 coefficients bounded by 2,047, the unnormalized twelve-stage
accumulator is at most `2,047 * 4,096 = 8,384,512 < 2^24`; signed int32 is sufficient and
the format requires int64 accumulation.

There is no trainer and no selector search.  The one universal selector is
SplitMix64 domain `0x5441435449434448`, ordinal 17, frozen before any source.
The 16-KiB packet serializes that same table for every checkpoint and is still
fully charged; its frozen SHA-256 is
`0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad`.
The encoder may not choose a candidate, retry a seed, fit a
table, or alter topology, rank, rate split, or feature rules.

## Frozen universality contract

The format is universal across canonical SwiGLU-MoE expert triplets.  It does
not accept model, checkpoint, layer, expert, vendor, training-corpus, file-name
or provenance identity.  Gate rows, Up rows and transposed Down columns are
role-labelled and flattened in row-major order.  Literal 4,096-value blocks
are zero-padded only when the fixed matrix shape requires it; that padding is
fixed by shape metadata and physically charged.

The frame rule may read only (1) canonical integer coarse symbols from the
same block, (2) fixed role/shape/block/stage metadata, and (3) the physically
transmitted universal packet.  The current rule actually uses only same-block
symbols, role, stage and the universal selector.  It has no Qwen-specific table,
external reference, calibration corpus, other expert read, router
signal, or source-fitted parameter.  The six-expert `768 x 2048` object below
is a fixed evaluation/ledger geometry, not an input that selects the decoder.

## Exact physical and read ledger

There are 28,311,552 weights in the six-expert/18-matrix object.

| Field | Exact bytes |
|---|---:|
| Global schema, hashes, layout | 4,096 |
| Selector packet and zero padding | 16,384 |
| Q12/QC tables | 2,048 |
| Seeds, fixtures, reserved bytes | 2,048 |
| **Global packet** | **24,576** |
| Expert header/directory | 512 |
| `1,152 * 48` target-coset bytes | 55,296 |
| `18 * 78,592` coarse bytes | 1,414,656 |
| **One expert frame** | **1,470,464** |

Thus

```text
24,576 + 6 * 1,470,464 = 8,847,360 bytes
8 * 8,847,360 / 28,311,552 = 2.5 bpw exactly.
```

The decomposition is

```text
coarse payload    307/128 = 2.3984375 bpw
target bits        12/128 = 0.09375 bpw
all metadata        1/128 = 0.0078125 bpw
total              320/128 = 2.5 bpw.
```

The global packet is six 4-KiB pages and an expert frame is 359 pages.  One
equal physical share is 360 pages.  A completely cold expert read is therefore

```text
(6 + 359) / 360 = 73/72 = 1.0138888888888888x.
```

The complete global packet is charged and reread for every cold expert.  The
decoder reads one expert frame once; no warm cache or second compressed-byte
pass is assumed.

## Dominant CuPy gate

The gate must use the **actual** independently decoded `307/128` coarse
artifact.  Transferring the old 2.5-bpw factor is planning arithmetic only.
The future coarse lock must bind every source, reconstruction, integer-symbol
file, all 108 fixed reservoirs, byte count and causal decode/re-encode receipt
separately for the source and both controls.  All inputs are component-walked with no-follow opens,
hashed and read from held regular-file descriptors, and kept held until the
result is sealed.

For each calibration block let `e=x-y`.  Because `B^T B=I`, the impossible
continuous-coefficient reconstruction has exact residual

```text
e_perp = e - B B^T e
||e_perp||^2 = ||e||^2 - ||B^T e||^2.
```

Let `D0=sum||e||^2/sum||x||^2` and
`c=1-sum||e_perp||^2/sum||e||^2`.  At physical rate 2.5 the exact requirement
is

```text
c_req = 1 - 0.025 / D0.
```

Using the published finite factor only as a favourable planning transfer gives
`D0=0.035574242296714034` and
`c_req=0.2972443434920543`.  The runtime must replace both with its measured
lower-rate values.  A rank-384 isotropic frame captures only `384/4096 =
0.09375`, so the planning cell needs 3.1706063305819128 times the isotropic
rank share even before finite coefficient shaping loss.

Across the six predeclared whole-expert evaluation clusters (none is used to
fit or select the architecture):

- hard reject this frozen cell when `capture + 3*expert_cluster_SE < c_req`;
- promote to finite coset search only when
  `capture - 3*expert_cluster_SE >= c_req`, every expert/role fold is positive,
  and all authentication/round-trip checks pass;
- otherwise report `HOLD_INCONCLUSIVE` without changing the cell.

For the complete fixed decision object, there is also a deterministic stop:
as soon as accumulated oracle SSE exceeds `0.025 * total_source_energy`, unseen
SSE is nonnegative and cannot recover `F<=0.8`.

The continuous oracle grants arbitrary real coefficients and is never an
emitted codec.  A survivor must run literal 384-bit QC/trellis search, decode
the exact Q12 coefficients independently, and satisfy `F=MSE*2^5<=0.8`.

## Decoded controls

The source absolute oracle runs first.  If it is already below the architecture
gate, controls remain unopened.  A survivor requires one independently encoded
row-moment-matched Gaussian panel and one independently encoded
energy-preserving sign/permutation panel.  Each control repeats the exact
`307/128` finite coarse encode, the same one universal packet, projection,
finite search budget, and byte/read accounting.  A control is an interpretation
of source-specific signal, not a converse: the continuous-span ordering need
not be preserved by a finite coefficient code.  Final promotion requires the
finite source result to beat both identically configured controls under the frozen
expert-cluster rule as well as pass the absolute `F<=0.8` gate.

## Containment and overlap boundary

- **RAVEL is not this cell.** RAVEL applies a zero-target-bit scalar conditional
  mean.  TACTIC-DH384 reads 384 target bits and emits a dense 4,096-D vector;
  neither the 6,144-entry LUT nor its WLS oracle contains this codebook.
- **MALT64 does not contain it.** MALT grants real coefficients only in the
  rank-three `64x64` SVD tangent of the decoded block (`375/4096` dimensions).
  A conditional dyadic frame is generically outside that tangent.  MALT's
  observed `+3SE` capture near its isotropic dimension is adverse prior
  evidence, not a bound on this span.
- **SILWARP does not contain it.** SILWARP-v2 emits one deterministic correction
  for a decoded tile and has no target-specific refinement bits.  For one `y`,
  TACTIC-DH384 exposes up to `2^384` finite corrections.
- **CCQ, direct VQ and additive VQ do not contain it.** They code raw short
  vectors with fixed/channel-conditioned first-stage codebooks.  This cell
  conditions a 4,096-D residual codebook on a literal finite coarse codeword.
- **KBVQ-IDRE and initializer anchors do not contain it.** Those subtract a
  raw-weight low-rank or procedural component before residual coding.  This
  frame changes block by block with decoded quantization state.
- **The role/polar composite may overlap.** No `s` values are added and no `F`
  values are multiplied.  This v2 ledger is paired only with its literal
  lower-rate finite coarse decode.  A future role/polar nesting must rebuild
  the combined residual, selector, allocation, bytes and reads once; the old
  `F=0.936397621` cannot be transferred as evidence.

This is the widest bounded decoder-visible hole left by the completed gates,
but it has a low prior: MALT captured only its dimensional null and SILWARP
selected identity.  One dominant span test is justified; architecture growth
after a clear miss is not.

## Source-only and future commands

No authenticated actual `307/128` coarse artifact is included in or authorized
by this package.  Stage 0 is therefore not runnable as sealed.  Its minimum
upstream dependency is a separately reviewed, model-agnostic finite coarse
producer that emits, for each of the source, decoded-Gaussian and
structure-destroyed panels: 108 independently decodable 78,592-byte
reservoirs, 18 canonical BF16 source/F32 reconstruction/I16 symbol records,
and a held-file decode-reencode PASS receipt.  A strict JSON lock must bind all
paths, byte counts and SHA-256 values.  Producing that artifact is a distinct
source-only task and is not implied by these commands.

Source verification imports only the standard library:

```bash
/usr/bin/python3.12 -B verify_source.py --package .
/usr/bin/python3.12 -B test_source_only.py -v
```

Only after that passes and an independent source review authorizes synthetic
CUDA may the payload-free preflight run:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B \
  cupy_preflight.py \
  --output /workspace/tactic_dh384_v2_synthetic_preflight \
  --authorization SYNTHETIC_ONLY_TACTIC_DH384_V2
```

The preflight has no model/coarse/root/manifest argument.  It compares CuPy
against a standard-library dyadic fixture, checks norm/projection identities,
packet packing and the exact ledger, then writes a create-new receipt.

The future stage-0 entry point remains inert without the literal authorization
`OPEN_AUTHENTICATED_ACTUAL_LOWER_RATE_TACTIC_DH384_V2`.  Its coarse-lock and
output paths must be absolute, the output must be absent, and any identity,
rate, closure, no-follow, held-FD, finite-value or lower-rate round-trip failure
stops before numeric scoring.

## Claim boundary

This package freezes one rank-384 conditional dyadic cell and its favourable
continuous-span falsifier.  It is not an achieved codec, a result on Qwen, an
all-neural-decoder converse, a model-wide claim, or permission to access a
payload.  A negative closes only this exact feature/table/rank/rate cell.  A
positive only authorizes the separately audited finite-coset phase.
