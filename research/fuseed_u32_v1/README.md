# FUSEED-U32 v1

Status: **source-only v1 runtime early-kill pending independent audit of the
architecture-equivalent direct Philox calibration; no Qwen/model access**.

FUSEED-U32 is a finite, exhaustive initializer-state search for three fixed
Qwen/MCore-compatible generator ABIs.  It searches every serialized base seed
from `0` through `2^32-1`, but it never writes a candidate anchor matrix and
never allocates a seed vector.  A direct Philox counter kernel generates each
prospectively fixed anchor coordinate once, uses warp lanes as the 32 matched
control domains, emits only one float32 metric per candidate/domain, and keeps
an exact tie-aware Top-K.

This package contains a design and a source-only verifier.  It contains no CUDA
implementation, model path, model manifest, source lock, cache, payload or
result.  It is not execution authorization.

## Outcome and claim boundary

The design reduced the family to `12,884,901,888` hypotheses from the
procedural draft's `3,266,322,626,690`, a 253.5-fold reduction, and the
direct-counter arithmetic remains promising.  A full-shard 33-domain surrogate
projected 2,992.413043 seconds for the 768 shards, but independent red-team found
that it used `curand_init` plus `curand_normal4` per bundle rather than v1's
direct Philox4x32-10 counter core.  It also used two repetitions instead of the
frozen three and charged cold Top-K incorrectly.  A 2.85x kernel speedup would
be enough to clear the 900-second gate, so this slower surrogate is not a lower
bound and cannot kill v1.

That surrogate was correctly demoted.  A distinct architecture-equivalent
calibration then ran the direct Philox4x32-10 core, pinned cuRAND Box-Muller and
BF16 path for three full-shard repetitions.  It projected 2,672.980880 seconds
warm end to end, 2.970x over the unchanged 900-second limit, before omitted
journal/global-merge and later-stage work.  Conditional on an independent
audit of that package, v1 is runtime-killed.  No Qwen/model payload, manifest,
cache or result was opened, and no initializer ABI or seed was statistically
tested.

The scientific limits are deliberately strict:

- A positive result is auxiliary procedural evidence, not proof of the public
  Qwen producer initializer.  The public release ABI is unknown.
- A negative result kills only the three listed ABI families and only if all
  retention, parity and completeness gates pass.
- `0.1456888483858212` capture is a **composition screen only**.  It is usable
  only when a separately finite, audited residual-codec parent is already
  bound.
- Without that parent, the finite standalone threshold is
  `0.19102060916075425`.
- The search and validation panel exposes Up and Down only.  Even a standalone
  Up/Down survivor needs a separately frozen, one-shot Gate-role confirmation
  using the exact winning ABI/seed, with no reselection, before an 18-matrix
  residual codec or 18 FP16 affine cells can be claimed.

## The three frozen ABIs

All candidates use Qwen layer 15 under the fixed public geometry PP2, EP4,
ETP1.  For global expert `e`:

```text
ep_rank      = e // 32
local_expert = e % 32
seed64       = base_u32 + 1024 + 100*ep_rank
```

The addition is mathematical u64 addition.  It must not wrap at u32.  The
serialized base is four bytes, but `seed64` can exceed `UINT32_MAX`; its carry
becomes the high Philox key word.

The ABIs are:

1. `PAI6BA_LGP_GATE_UP_DIRECT_BF16`: the legacy MCore giant-packed fallback,
   one FC1 call for all 32 local experts followed by one FC2 call.  At layer 15
   its call offsets are 8760 and 9148.  This exact fixed public topology is not
   covered by either the procedural MCore projection-major branch or the
   direct RNG-state envelope.
2. `CURRENT_PMG_GATE_UP_DIRECT_BF16`: the later pinned MCore/TE
   projection-major, per-expert calls, gate-then-up.
3. `CURRENT_PMG_UP_GATE_DIRECT_BF16`: the same exact later call geometry with
   the source-supported alternate fused-half order.

The mutable public PAI external-TE image is not an ABI: it does not pin the
necessary TE, PyTorch, CUDA and copy/init lifecycle.  Legacy ETP2/4 and other
topologies are also excluded.  Adding any of them would be a new family and a
new protocol.

Seed zero is included because the requested auxiliary family is the complete
u32 label space.  It has no positive-CLI provenance in the public PAI launcher,
and the design does not pretend otherwise.

## Breakthrough: direct counters, not state advancing

For the frozen RTX 5090 policy, block 256 and grid 1020 give a native stride of
261120.  For native index `i`:

```text
sequence      = i % 261120
q             = i // 261120
lane          = q & 3
normal4_index = q >> 2
C             = (sequence << 64) + (initializer_offset >> 2)
                + normal4_index                    (mod 2^128)
```

Every frozen initializer offset is divisible by four.  The desired value is
therefore obtainable with one direct Philox4x32-10 evaluation at `C` followed
by the exact cuRAND Box-Muller lane, rather than `curand_init` followed by up
to 97 sequential `curand_normal4` advances in the giant FC1 call.

The stage-0 plan goes further.  It prospectively chooses four semantic
coordinates that share one counter and consume lanes 0, 1, 2 and 3.  Thus 1,024
anchor values need exactly 256 Philox cores per candidate.

The direct counter is still conditional on exact runtime parity.  Integer
Philox alone does not certify cuRAND's `logf`, `sqrtf`, `sincosf`, scale
multiplication or BF16 rounding.  Calibration must cover effective seeds over
`UINT32_MAX`, generator offset `2^34` where the counter low word carries, every
native sequence/lane boundary and the final generator state.  Any mismatch is
a pre-payload hard kill.

The direct kernel must call the hash-bound `curand_normal.h` device-inline
`_curand_box_muller` on Philox word pairs `(x,y)` and `(z,w)` in that order.  An
algebraic reimplementation is not accepted as executable authority.  Up uses
the exact float32 scale bits `3c03126f`; Down uses `3a560a28`.  Multiplication
is binary32 round-to-nearest, followed by `__float2bfloat16_rn` and exact
widening.  Parity compares both raw normal float32 bits and scaled BF16 bits to
`curand_init(seed,sequence,O+4*j)` plus one `curand_normal4`, as well as the
original `O` plus `j+1` calls.

## Exact coordinate nest

The payload roles are Up and Down.  Selection uses the 23 predeclared matrices
from experts:

```text
0,8,16,32,40,48,64,72,80,96,104,112
```

with expert-0 Up excluded.  Validation uses both roles for experts
`24,56,88,120`.

For each ABI, stage 0 has 512 fit and 512 score coordinates.  Per split, six Up
and seven Down matrices receive 24 coordinates; the remainder receive 20.
This gives exact role counts Up=244 and Down=268, all divisible into normal4
bundles.  The high-count identities and all bundles are selected by the locked
SHA-256 grammar in `design_lock.json`, with no tensor statistic.

The verifier independently enumerates the native maps and proves every bundle:

- has one sequence and one normal4 index;
- consumes lanes 0..3 exactly;
- inverts to the requested expert and role;
- has four unique canonical `(expert,role,row,column)` coordinates;
- never crosses fit and score;
- fits the exact native call boundary.

The three ABI-specific stage-0 sets are nested into a common 2,048-fit /
2,048-score stage-1 plan.  That is nested into the 24,312-fit / 24,312-score
full selection plan.  Validation independently contains 8,456 fit and 8,456
score coordinates.  The complete deterministic plan digest is:

```text
f19492e5ed1cc93949f1c9ca8038576a7fe17fe7f519b473773d721d17d6f260
```

The stage-0 native-record digest is:

```text
97ac7933a1d3735960bed34977279d45212c553e1717d03bd7c87fe4ff7e9981
```

## Fused stage-0 kernel

One logical warp owns one base-seed candidate.  Generator work is distributed,
not serialized in lane 0:

1. For `t=0..7`, every lane `l` computes bundle `32*t+l` and retains its
   float4.
2. In fixed t-major, generating-lane-major, value-lane order, warp shuffles
   broadcast all 128 values for that `t`.
3. Lanes 0..31 accumulate the 32 control-domain sufficient statistics; lane 0
   additionally accumulates source.
4. The next `t` starts only after every domain has consumed the prior 128
   values.

Per candidate there are eight common float64 anchor moments
(`role × fit/score × sum/sumsq`).  Each domain owns four float64 cross moments
(`role × fit/score`).  Target sums, squares and counts are precomputed.  Alpha
and mu are fitted, rounded once to FP16, reloaded, and used to reconstruct held
out SSE from moments.  A nonfinite moment, affine, baseline, SSE or capture
aborts the entire cell.

The final finite capture is rounded once to float32, and `-0` is canonicalized
to `+0`.  This float32 value is the exact stage-0 metric.  The later stages use
float64 per-matrix moments and decoded FP16 affines.

No seed array, anchor array or composite per-candidate sort-key array is
allowed.  The only full shard output is `q[33,2^24]` float32, exactly
2,214,592,512 bytes (2.0625 GiB).  The exact radix selector derives the seed
from `shard_base+index` on demand.

## Complete-u32 sharding and exact Top-K

Each ABI partitions the u32 interval into 256 adjacent shards of `2^24` seeds;
there are 768 shards total.  Shard arithmetic is checked in u64.  Each shard
contains `2^34` generated values and `2^32` normal4 bundles, so 32-bit count
arithmetic is forbidden even though each seed label is u32.

The total order is:

```text
capture float32 descending,
abi_index ascending,
base_seed_u32 ascending
```

Stage 0 retains K=8192 separately for every `(ABI,domain)`.  Within one ABI the
tie is unsigned seed ascending.  Exact cutoff ties select the smallest seeds;
`argpartition` alone is not valid.  Each shard record is `(seed_u32,
metric_f32)`, eight bytes, with ABI/domain/shard supplied by its authenticated
container.  All 768 shard Top-K records occupy at most 1,660,944,384 bytes.

The rolling merge is exact because, under one total order,
`TopK(A union B)` is contained in `TopK(A) union TopK(B)`.  Replay under a
different legal merge tree must hash identically.  Cross-ABI hypotheses remain
distinct descriptors; they are not collapsed merely because a sparse screen
happens to tie.  The cross-ABI/domain descriptor union, at most 811,008, is not
truncated before stage 1.

The cascade ledger is:

| Stage | Candidates (maximum) | Coordinates | Generated values |
|---|---:|---:|---:|
| 0 | 3 × `2^32` | 1,024 | 13,194,139,533,312 |
| 1 | 811,008 | 4,096 | 3,321,888,768 |
| 2 | 8,448 | 48,624 | 410,775,552 |
| validation | 33 frozen winners | 16,912 | 558,096 |

Total: `13,197,872,755,728` generated normal values.

## Multiplicity without an exchangeability fiction

The 4.3-billion-way multiplicity per ABI is handled computationally: every seed
is evaluated, and exact Top-K retains the metric winners.  It is not replaced
by a random sample or a null-exchangeability assumption.

Source plus 16 Gaussian controls plus 16 scramble controls all execute the same
three-ABI complete-u32 family and the same cascade.  Their descriptive
correction is:

```text
source capture - max(0, maximum of all 32 control-winner captures)
```

Those controls are heterogeneous and are not asserted to form an exchangeable
orbit.  No randomization p-value or familywise p-value is claimed.  They expose
winner's-curse behavior under matched searches; they do not mathematically
erase it.

Stage-0-to-stage-1 recall is likewise not a theorem.  The familiar planning
approximation, with N=`2^32`, K=8192, 512 score coordinates and the composite
capture target, gives a modeled retention of about 0.99983.  This is only a
planning value.

The actual pre-payload gate freezes 256 stress cells over ABI, matrix
concentration, Up:Down ratio through 100:1 in both directions, coordinate
correlation through ±0.9, shared-candidate correlation, heteroscedasticity and
finite heavy tails.  Each cell has 1,010 planted trials.  With all successes,
the exact Bonferroni one-sided lower bound is `0.9900004818473975`, just above
the required 0.99 simultaneously across 256 cells.  Boundary and exact-tie
cases are a separate deterministic suite.  Any failed cell kills the search
before model-source authentication; it cannot support a negative payload
claim.

## Feasibility gates

The historical raw generator rate gives a 283.54-second floor for stage 0.  A
materialized older 33-domain path extrapolates to roughly 3.09 hours.  FUSEED's
direct counters, four-lane bundles, distributed warp generation and moment-only
objective are intended to close that gap, but only exact calibration decides.

Two hash-bound synthetic probes are encouraging:

- The 80-bundle single-score probe reported 1.88779 seconds kernel and 2.44351
  seconds warm end-to-end for a linear complete-u32 projection.
- The complete-scan probe traversed every u32 seed twice with exact tie-aware
  shard Top-K/global merge and reported 2.915525191 seconds median; both replay
  hashes matched.

They are feasibility evidence only.  Neither used the three ABIs, 256 bundles,
33 domains, float64/FP16 moments, the final journal, retention, or model data.

The prospective source-free calibration required a complete `2^24` shard with
the production shape, including finite validation and exact Top-K.  Its median
times 768 had to be at most 900 seconds.  Equivalently, stage 0 had to sustain
at least:

- 14,316,557.65 candidates/s;
- 14.660155 billion anchor values/s;
- 3.665039 billion normal4 bundles/s;
- 483.785116 billion domain cross-moment FMAs/s;
- 1.171875 seconds average per shard.

The non-authoritative source-free surrogate was:

| Measurement | Result |
|---|---:|
| registers/thread | 106 |
| local spill bytes | 0 |
| q buffer | 2,214,592,512 bytes |
| median full-shard kernel | 3.3351742995 s |
| median full-shard finite-check + exact Top-K | 3.89637115 s |
| projected 768-shard kernel | 2,561.413862 s |
| projected 768-shard end to end | 2,992.413043 s |

Both q sentinels and Top-K hashes matched over its two replays, but the run is
not architecture-equivalent and does not satisfy the three-replay lock.  It
also used synthetic sequence/offset/scale descriptors, emitted q rather than
the frozen capture wire, omitted exact runtime/compiler/header/kernel hashes,
and multiplied cold first-Top-K cost across all shards.  Its package is bound
only as a warning.

The required source-free calibration used the exact direct Philox counter core,
pinned cuRAND Box-Muller/BF16 path,
distributed 256-bundle kernel, three timed repetitions, three Top-K hashes and
the prospective cold-overhead rule.  That calibration has now completed:

| Direct-counter measurement | Result |
|---|---:|
| parity rows | 132 |
| raw float32 / scaled BF16 / terminal counter | bitwise PASS |
| repetitions and deterministic hashes | 3 / 3 |
| registers/thread; local spill | 108; 0 bytes |
| median full-shard kernel | 3.3335224930197 s |
| median warm selection | 0.1463984082220 s |
| one-time cold selection excess | 0.4016282198718 s |
| projected 768-shard kernel | 2,560.1452746391 s |
| projected warm end to end | 2,672.9808803735 s |

The direct performance kernel contains zero `curand_init` and zero
`curand_normal4` calls; each bundle uses one direct Philox core and two pinned
`_curand_box_muller` calls.  Omitted journal/global merge and stage 1/2 work can
only add time.  The only valid v1 next step is an independent source-only audit
of the calibration and bound arithmetic.  If it passes, finalize the runtime
kill; if it fails, revert to blocked.  No model access is allowed.

A conservative FP32 interval screen followed by exact FP64 refinement could be
a worthwhile **distinct v2**, as could a prospectively justified smaller
stage-0 plan.  Either changes the scientific screen and must receive a new
numeric containment proof, false-negative retention stress, exact tie audit and
runtime freeze.  It is not an adaptive v1 retry.

## Physical rate and read ledger

The conditional final descriptor is eight bytes:

```text
u32 base_seed, u8 abi_index, u8 family_version, u16 reserved_zero
```

Eighteen decoded FP16 `(alpha,mu)` pairs add 72 bytes, for 80 bytes total across
28,311,552 weights.  The side rate is `0.0000226056134259259` bpw, leaving a
maximum compatible base codec of `2.149977394386574` bpw under the 2.15 cap.
There is no learned generator table and no external generator read.  The
conditional cold metadata read is 20 bytes/expert; appended to the inherited
worst read ledger at 2.15 bpw it is `1.1694597713582042x`, below 2x.

This ledger is arithmetic only.  It becomes available only after the separate
Gate confirmation supplies all 18 affine cells and a finite upstream
codec/read receipt is bound.  An Up/Down screen alone has no right to book the
80-byte final codec.

## Relationship to prior families

- `procedural_anchor_expansion` searched a 3.27T broad Cartesian union.  FUSEED
  contains only its fixed CURRENT-PMG PP2/EP4/ETP1 u32 rows, excludes its other
  topology/offset/HF rows, and adds the legacy giant-packed ABI that the draft
  did not model.
- `direct_rng_state_envelope_v1` searched 125,862,912 later-PMG states over
  many prefixes/topologies but a bounded pipeline-seed interval.  FUSEED
  intersects its prefix-15, EP4-contiguous, ETP1 rows for seeds 1..70236 and
  both half orders.  Neither family contains the other; direct RNG state does
  not contain legacy giant packing.
- The public legacy descriptor at base 1234 is one exact member of FUSEED ABI0.
  Earlier collision evidence tested a different fixed cell and does not kill
  the coherent all-u32 family.
- The prior 100x cascade result rules out an ordinary broad early-stopping
  design.  It does not rule out this fixed-ABI, direct-counter, fully enumerated
  fused scan.

## Verification

Run with PowerShell 7 or newer:

```powershell
pwsh -NoLogo -NoProfile -File research/fuseed_u32_v1/verify_design.ps1
```

The verifier hashes every dependency, proves complete u32 shard coverage,
checks u64 seed carry and the Philox integer KAT, independently enumerates all
3,072 stage-0 coordinates and every nested plan coordinate, tests exact Top-K
union/tie behavior, recomputes all value/rate/read/threshold ledgers, checks the
finite retention-cell construction, authenticates both the demoted surrogate
and decisive direct-counter result fields, proves the unchanged runtime-gate
arithmetic, and rejects unexpected package members.

The receipt is producer self-verification, not an independent audit or launch
authorization.
