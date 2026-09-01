# SILWARP auxiliary nonlinear-codec gate v2

Status: **frozen before any v2 numeric payload decode or v2 CUDA launch**.
This v2 freeze follows the fully disclosed v1 preflight/failure access below.
The two predeclared Gate-confirmation files were read only by the root agent
for SHA-256 authentication before training; their values remain undecoded and
unscored.  This access is disclosed in `source_lock.json`.

## v1 pre-training failure and exact v2 repair

The audited v1 candidate (`protocol c6b5dffd...`) passed its source-free CPU
and RTX 5090 GPU preflights, authenticated all frozen fit/calibration files,
and then failed closed before training while constructing `null_a` for fit
matrix `(layer=31, expert=31, role=down)`.  No update, checkpoint, score,
confirmation access, or result exists.  Its immutable run log SHA-256 is
`c4f11b0005e4cc4ee5725d70e1553bdef3aa45dad1f26eeec68d9be0878d3991`.
One post-failure CPU diagnostic reopened only this already-authenticated fit
record to recover the exact moments below.  It opened no new fit/calibration
identity, confirmation value, pinned value, or CUDA context.

The real source's exact FP64 mean was
`-4.649028828573876e-09`, which nearest-even FP16 represented as negative zero
(`0x8000`).  The mathematically zero `null_a` mean was
`2.238990423908867e-19`, represented as positive zero (`0x0000`).  These have
identical numerical value; source and control RMS and unit-moment checks also
matched.  V1 nevertheless rejected their different sign bits.

V2 makes exactly one mathematical no-op repair: immediately after nearest-even
FP16 conversion, either signed zero mean is canonicalized to the unique `+0`
encoding.  Real-valued centering, RMS, normalization, random streams,
architecture, hyperparameters, seeds, splits, stopping rules, thresholds,
ledgers, and bitwise source/control equality checks are otherwise unchanged.
Nonzero FP16 metadata mismatches still abort.  The v1 log and GPU receipt are
preserved in `research/implicit_hyperdecoder_gate_v1_failure/`.

For avoidance of doubt, the seed-derivation domain remains the literal
`SILWARP-v1`; changing the protocol/container schema to v2 does not regenerate
either Gaussian control or any training/evaluation counter.  "Unchanged"
above is numerical: a signed zero in a centered array may acquire the canonical
zero sign, but no real-valued normalization quantity, decoder value, SSE, or
decision metric changes because of that representation.

SILWARP is a source-conditioned implicit lattice warp.  A base rate-`R0`
reconstruction `Y` is retained in full and a compact shared nonlinear decoder
maps each canonical `16 x 16` tile to

```text
X_hat = Y + f_theta(Y, public_coordinate, serialized_matrix_moments).
```

The eventual PTQ encoder is asymmetric: it may perform arbitrary
analysis-by-synthesis/list search over base lattice or polar-SC codewords, but
the deployed decoder receives only the stored base code, public coordinates,
the charged matrix moments, and the frozen FP16 hyperdecoder.  It never sees a
source-derived per-tile feature.  A role-local bypass flag makes `X_hat=Y` an
exact selectable baseline.

This first experiment is an auxiliary, information-valid oracle.  It replaces
the finite base lattice with a conservatively represented Gaussian
rate-distortion test channel.  Let `D=2^(-4.3)`.  The exact executable
constants are

```text
a32     = 0.9492341876029968   (rounded downward)
sigma32 = 0.21951906383037567  (rounded upward)
Y*      = a32 X32 + sigma32 Z, Z ~ Normal(0,1), independent of X32
Y32     = Q32(Y*)
```

after charged FP16 whole-matrix centering and RMS normalization.  The mean is
rounded to nearest-even FP16 first, and either signed zero result is assigned
the unique `+0` encoding.  This canonicalization does not change its numerical
value.  The exact FP64 RMS is then computed about
that serialized mean, and the stored RMS is the **smallest positive finite
FP16 value greater than or equal to the exact RMS**.  The candidate is then
used to form the actual FP32 normalized array.  Its squared FP32 values are
re-accumulated in FP64, and the FP16 RMS advances by as many ULPs as needed
until this post-cast empirical second moment is at most one.  Exact zero maps
to the smallest positive FP16 subnormal; NaN, infinity, or FP16 overflow fail
closed.  Thus `E[X32^2] <= 1` for the exact values consumed by the decoder.
For every such source under the mathematical ideal normal law,

```text
Var(Y*) <= a32^2 + sigma32^2,
I(X32;Y*) <= 0.5 log2(1 + a32^2/sigma32^2)
            = 2.149999824926515 bpw
            <= 2.15 bpw
            <= 2.1500040690104165 emitted role-payload bpw.
```

The first inequality follows from post-cast RMS validation and the second from
Gaussian maximum entropy.  `Q32` and a deterministic SILWARP decoder are data
processing and cannot increase mutual information.  Evaluation generates a
reproducible FP64 Box-Muller Monte Carlo draw, forms `Y*` in FP64, and casts
once to FP32.  This finite counter-PRF draw estimates the mathematical AWGN
expectation; the finite deterministic PRF law is not itself the MI proof.
Training-only noise is counter-derived FP32 Box-Muller and carries no proof
claim.  Public coordinates add no source information, and the only
source-derived metadata (three FP16 means and three FP16 RMS values) is
explicitly charged.  Therefore a held-out improvement is a favorable
source-RD opportunity, not a finite-code claim.  Any survivor must still pass
an actual serialized auxiliary lattice/POLARIS experiment before a pinned-panel
protocol can be considered.

## Why this is a distinct architecture

- Shallow BiSCo synthesizes a 16-vector directly from two additive 18-bit
  binary latents.  SILWARP retains a full-rate base reconstruction, processes
  256 coordinates jointly, and applies a tied recurrent, coordinate-conditioned
  nonlinear codeword warp.
- Additive VQ reconstructs a sum of stored atoms.  SILWARP's reproduction is a
  nonlinear function of the complete base codeword, so its implicit codebook
  has one warped point for every base-lattice point without storing those
  points.
- The completed affine-flow gates predict scalar location/scale from bounded
  contexts.  SILWARP is a non-invertible 256-D posterior-mean map optimized
  directly for source MSE.
- Linear shared-subspace and template gates are strict linear projections.
  SILWARP contains multiplicative gates and six reused nonlinear residual
  steps.
- The old 64-continuous-latent hyperdecoder forced a 256-value tile onto a
  64-D manifold and consequently left about 75% of its energy.  SILWARP keeps
  all 256 noisy/base coordinates and predicts only their low-energy error.

The gate does not claim to reject all nonlinear decoders if it fails.

## Exact architecture

All canonical matrices are `768 x 2048`; Down is transposed before tiling.
Tiles are non-overlapping `16 x 16` row-major arrays.  Whole-matrix FP16 mean
and centered RMS are the only source-derived conditioning values.

The decoder has hidden width 256, gated bottleneck 128, and six tied recurrent
steps.  The 21 public/charged features are:

1. role scalar (`up=-1`, `gate=0`, `down=+1`);
2. normalized layer and expert indices;
3. standardized log of the serialized matrix RMS;
4. normalized tile-row and tile-column centers;
5. row/column sine and cosine at frequencies 1, 2, and 4;
6. sine and cosine of normalized layer depth; and
7. centered squared layer depth.

No feature may depend on an unencoded source tile.  Identity and learned
distortion are both scored only after applying the exact serialized mean and
post-cast-validated upward RMS path above.  With SiLU `phi`, the fixed
map is

```text
h0 = phi(Y Wy + c Wc + b0)
for t in 0..5:
    u = phi(h A + ba)
    g = sigmoid(h C + bc)
    h = h + (u * g) B + bb
r = tanh(h Wo + bo)
X_hat = Y + gain[role] * r
```

The parameter shapes and count are frozen in `protocol_lock.json`.  The output
path is initialized to exact zero, so identity is exact before training.  A
role bypass bit in the 64-byte expert header can select identity after
training.  Promotion metrics always serialize the model in the frozen order,
round every parameter to IEEE binary16, hash those exact bytes, reload them,
and evaluate the reloaded decoder.  The 4,096-byte model header also binds the
closed source-lock hash.

## Leakage firewall and exact split

The pinned source directory is forbidden.  Before inventory selection, every
auxiliary pair whose layer is in `{5,12,18,28,36,45}` **or** whose expert is in
`{7,18,20,41,76,83}` is rejected.  The remaining identities are divided by
the exact lists in `protocol_lock.json`:

- fit: 41 complete Up/Down layer-expert pairs (82 matrices);
- calibration: the eight pairs formed by layers `{10,22,34,46}` and experts
  `{15,87}` (16 matrices); and
- untouched confirmation: the eight pairs formed by layers
  `{3,13,27,39}` and experts `{57,121}` (16 Up/Down matrices), plus the two
  available Gate matrices `(3,57)` and `(3,121)` as a role-transfer check.

There is no shared layer and no shared expert between any two split classes.
The confirmation payload remains unopened until the complete calibration
promotion rule passes.  The pinned panel remains unopened for every possible
outcome of this executable.

`source_lock.json` binds revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, all 116 exact basenames,
3,145,728-byte lengths, split memberships, and SHA-256 values under its own
canonical internal seal.  Its file SHA-256 is
`b0dc982e42a22fa960a1436ed5ebdcab1233d0bc2e463972e16dab4315e14042`.
The 114 Up/Down identities inherit the sealed hash map in the prior
continuous-flow artifact; the two Gate hashes are the authentication-only
reads disclosed above.  The runner rejects symlinked directories, symlinked
inputs, nonregular files, wrong sizes, unexpected identities, or any hash
mismatch.  Each source is opened once with `O_NOFOLLOW` where available,
checked by `fstat`, read and hashed, then decoded from that same authenticated
byte buffer, closing the hash-then-reopen race.  Fit and calibration hashes are
verified before numeric decode; confirmation hashing and decode occur only
after calibration promotion.

## Paired controls and fixed training seeds

Two independent Monte Carlo Gaussian corpora, `null_a` and `null_b`, are
generated by domain-separated explicit SplitMix64 counters and FP64
Box-Muller transforms.  A control uses the interior midpoint of its Qwen
partner's serialized FP16 RMS cell, then passes the identical post-cast moment
algorithm.  This preserves the exact charged FP16 mean/RMS while leaving
unit-moment headroom even when Qwen needed an extra safety ULP.  Each matrix
therefore has the same shape and charged serialized whole-matrix moments as
its Qwen partner.  For each of two fixed
training seeds, Qwen and both null decoders use identical initial parameters,
tile indices, update counts, optimizer settings, and evaluation points.  The
two seeds are both reported; seed zero is predeclared as the only deployable
model and seed one is sensitivity evidence, never a retry or winner choice.

For source `S`, let `q_S = D_after / D_identity` and
`s_S=-0.5 log2(q_S)`.  The conservative matched statistic for one training
seed is

```text
s_match_worst = s_Qwen - max(s_null_a, s_null_b).
```

Cluster uncertainty is the maximum delete-one jackknife standard error over
whole layer, whole expert, and complete layer-expert-pair clusters.  All
absolute source-domain distortions, `F = MSE * 2^(2 Rphysical)`, both nulls,
both seeds, and all cluster folds are retained.

## Exact physical and cold-read ledger

One expert contains `3 * 768 * 2048 = 4,718,592` weights.  Each role has a
byte-padded base stream of 422,708 bytes, or exactly
`2.1500040690104165` emitted bpw.  The expert-local object is
three such streams plus a 64-byte header and 12 FP16 moment bytes.

The decoder has exactly 235,779 FP16 parameters (471,558 bytes) and a closed,
zero-padded 4,096-byte header, for exactly **475,654 bytes**.  The header binds
the architecture, feature construction, parameter order, protocol hash, and
decoder-payload SHA-256.

Conservatively attributing the global decoder over only 128 experts gives:

```text
payload bytes/expert             1,268,124
local header/moments bytes              76
attributed global bytes        475,654 / 128
physical R                    2.15643318494161 bpw
cold bytes/expert                    1,743,854
cold read amplification       1.37104489269124 x
```

The self-contained six-expert ledger is also reported (`R=2.28453855161314`,
still under 2.5 bpw).  The decoder is counted once in every cold read ledger,
even when it could be resident in practice.

At the conservative 128-expert rate, the exact target is

```text
MSE <= 0.8 * 2^(-2 Rphysical) = 0.0402520350667720
F <= 0.8
```

Using the exact represented-channel identity error
`0.05076578709329227`, rather than substituting the design `D`, gives

```text
identity F = 1.008958419301376
s_absolute >= 0.5 log2(identity F / 0.8)
           = 0.1673974074587855.
```

For observed Gaussian operational advantage `s_G`, the necessary matched
advantage is `s_match >= 0.1673974074587855 - s_G`.  The promotion rule below
uses the stricter worst-null statistic directly.

## Compute and preregistered stopping

The production cell uses CuPy on one RTX 5090, batch 512, at most 1,536 paired
Adam updates, two fixed training seeds, and two independent nulls per seed.
Tiles are sampled uniformly, but the training loss weights every tile by the
square of its charged serialized matrix RMS and divides by the sum of those
weights.  It is therefore aligned with pooled raw-domain SSE rather than
uniform normalized-matrix MSE; Qwen and its two matched controls receive
identical weights visible to the decoder.
Expected total work is about 10.2 trillion dense multiply-accumulates,
approximately 45--90 minutes including full validation; the update-512 gate
should finish materially earlier.  No alternate width, depth, tile size,
learning rate, or seed retry is allowed after a payload statistic exists.

A run additionally requires a regular, non-symlink launch sentinel containing
the fixed authorization phrase and exact protocol, source-lock, runner, and
common-module hashes under an internal seal.  Updates 256, 512, and 1,536
write new append-only checkpoint directories containing all six model and Adam
states, checkpoint history, predecessor hashes, and binding hashes.  One batch
index vector is counter-derived from `(training_seed, absolute_update)` and is
shared exactly by Qwen and both nulls; channel noise additionally includes the
corpus domain.  The explicit SplitMix64/Box-Muller construction has no opaque
RNG state.  Every checkpoint binds Python, NumPy, CuPy, CUDA runtime and driver,
device id/name/UUID, and compute capability; resume rejects a runtime mismatch.
Existing checkpoints are never overwritten, the entire predecessor chain is
verified, and resume rejects any incomplete, nonfinite, hash-inconsistent, or
binding-inconsistent state.

Before payload launch, the source-free CPU command is `python silwarp_gate.py
preflight`.  After audit authorization to create a CUDA context, `python
silwarp_gate.py gpu-preflight --output <new-receipt>` must also pass.  That
qualification opens no tensor payload: it keeps all six production model/Adam
cells resident, runs one warmup plus three timed batch-512 production-shape
forward/backward/Adam steps, and records median/projected time and observed
CuPy memory-pool peaks under the exact runtime identity.

At updates 256 and 512, for each training seed define

```text
U_t = s_match_worst(t) + 2 * SE_cluster(t).
```

Hard-kill this **one preregistered training cell** at update 512 only if, for
both training seeds,

```text
U_512 < 0.10
U_512 - U_256 < 0.012.
```

Four remaining 256-update intervals at the boundary slope project below
`0.10 + 4*0.012 = 0.148`, still below the required `0.1673974`.  This is an
aggressive trend gate, not a converse for SILWARP or nonlinear PTQ.

If the stop does not fire, both seeds train to update 1,536.  Calibration can
open confirmation only if the FP16-reloaded primary decoder has absolute
`F<=0.8`, worst-null `s_match-2SE>=0.1673974074587855`, every calibration layer
and layer-expert aggregate has positive `s_Qwen`, and the sensitivity seed
also has point-estimate absolute `F<=0.8` and positive worst-null matched
gain.  Confirmation uses no retraining or hyperparameter choice.  A positive
confirmation remains only `AUXILIARY_IDEAL_CHANNEL_SURVIVOR_REQUIRES_FINITE_CODE`.
