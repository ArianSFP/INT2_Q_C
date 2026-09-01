# STRATA expert-affine MoE checkpoint

## Status and claim boundary

This document describes the locality checkpoint that follows the frozen
STRATA-XKLT-SC v2 Qwen result. It is a format fork, not a modification of the
published v2 artifact. The fork keeps the same Q31-BEC/MAP-SC polar-lattice
payload primitive and changes block ownership so a routed MoE expert can be
read without fetching most of the other experts.

The checkpoint has two distinct gates:

1. exact physical rate at or below 2.5 bits per weight, maximum measured
   per-expert compressed reads below 2x, and original-BF16-domain MSE no worse
   than the frozen v2 result `0.04985939119332436`;
2. the later research target, evaluated at the artifact's own physical rate
   `R`: `MSE <= 0.8 * 2^(-2R)`.

Passing gate 1 does not imply gate 2. At exactly 2.5 bpw, gate 2 requires MSE
at or below `0.025`.

The reported read quantity is external compressed-container traffic. It is not
yet a benchmark of total HBM traffic in a fused decoder/GEMM kernel. A decoder
that materializes a full BF16 expert or repeatedly spills Hadamard scratch can
have much higher internal memory traffic despite reading the compressed object
only once.

## Why the frozen format over-reads

The panel contains six expert triplets, each with 4,718,592 weights:

```text
gate       768 x 2048
up         768 x 2048
down      2048 x 768, grouped in down.T order
```

Frozen v2 globally stable-sorts all 13,824 natural 2,048-weight groups by an
eight-way energy label, then cuts that order into fourteen dense RHT/arithmetic
blocks. One expert is consequently scattered across 9--11 blocks. Because the
inverse signed RHT is dense and the arithmetic stream is causal, touching one
group requires the entire containing stream.

Measured against one sixth of the physical v2 container:

- reading the complete container is exactly `6x`;
- selecting only touched block ranges is `4.479422x` on average and
  `4.906014x` in the worst expert, including the common prefix;
- indexing arithmetic offsets alone cannot remove this coupling.

The KLT between aligned Up and Down values was already expert-local. Global
energy sorting and block packing, not XKLT, caused the read amplification.

## Expert-affine architecture

One expert has an exact binary decomposition:

```text
4,718,592 = 2 * 2^21 + 2^19 weights.
```

The fork uses fifteen blocks:

```text
blocks 0..11    N=2^21, two private blocks per expert
block 12        N=2^20, tails of experts 0 and 1
block 13        N=2^20, tails of experts 2 and 3
block 14        N=2^20, tails of experts 4 and 5
```

For expert `e`, its 2,304 canonical groups are stable-sorted by
`(global_three_bit_label, canonical_group_ordinal)`. The first 1,024 groups and
next 1,024 groups become private blocks `2e` and `2e+1`. Its remaining 256
groups occupy half of block `12 + floor(e/2)`. This mapping is a permutation:
every source group occurs exactly once.

The geometric coefficient-volume amplification is therefore:

```text
(2 * 2^21 + 2^20) / (2 * 2^21 + 2^19) = 10/9 = 1.111111...
```

The release does not substitute that geometric estimate for a byte result. It
uses the fifteen emitted arithmetic lengths, includes the cold common prefix,
and separately reports the union of physical 4-KiB pages for each expert.

## What is unchanged

The fork retains:

- the literal eighteen-record Qwen route table;
- the literal 13,824 three-bit STRATA labels;
- six Q15-over-pi XKLT angle codes and regenerated FP32 coefficients;
- deterministic signed RHT preprocessing;
- six-level procedural Q31-BEC construction;
- MAP successive-cancellation decisions;
- canonical causal arithmetic coding;
- the frozen N20 and N21 finite-length allocation factors.

It deliberately uses a new magic, `PLRLOC3\0`, and a new seed domain. A frozen
v2 decoder must fail closed on this format.

## Exact physical ledger

The checkpoint container has a fixed 8,847,360-byte capacity:

| Section | Bytes |
|---|---:|
| Header | 128 |
| Route table | 144 |
| Raw three-bit labels | 5,184 |
| Fifteen directories | 105 |
| Arithmetic reservoir, including charged zero fill | 8,841,799 |
| **Total** | **8,847,360** |

The total is 70,778,880 bits over 28,311,552 weights, exactly 2.5 bpw. The
reservoir is charged in full, whether or not the arithmetic payload consumes
every byte.

Each seven-byte directory record is `<BeI`:

- one profile identifier `q`;
- one IEEE binary16 decoder scale;
- one little-endian unsigned 32-bit logical stream length.

Profiles use `R(q) = 1.75 + q/256`. A multiple-choice dynamic program assigns
the fifteen profiles under a predeclared 65,536-bit no-retry reserve. All
staging hashes, group mappings, profiles, decoder scales, and SC/RHT seeds are
sealed before arithmetic encoding.

## One-shot and independent audit protocol

The execution layer refuses pre-existing per-block outputs and invokes the
unchanged polar encoder once per sealed block. It verifies source hashes,
seeds, block geometry, causal frequencies, reconstruction indices, arithmetic
round trips, and literal payload hashes before packing.

The independent audit then:

1. reparses the fixed-capacity container;
2. validates the header CRC and route/label SHA-256 binding;
3. derives all profiles, scales, and domain-separated seeds;
4. causally decodes every arithmetic stream;
5. canonically re-encodes every decoded decision stream and requires literal
   equality;
6. applies the inverse signed RHT;
7. restores every group to its canonical ordinal exactly once;
8. applies the inverse XKLT;
9. hashes and opens all eighteen original BF16 sources; and
10. accumulates FP64 SSE and source energy in the original matrix domain.

The audit records the sealed plan lock and a canonical digest of all eighteen
source bindings. The compact verifier requires those bindings to match the
published plan. It also hashes each arithmetic slice in the physical container
against its corresponding one-shot encoder record and reconstructs the legacy
`<u32 logical_bits, f32 scale, payload>` container hash. Thus the retained
metadata is evidence for the actual packed streams, rather than merely fifteen
authenticated but unrelated JSON files.

The compact release omits raw BF16 sources, duplicate block payloads, and
decoded FP64 scratch. It retains the physical container, sealed plan, all
fifteen encoder metadata records, independent audit, tamper report, and a
manifest binding every published evidence byte.

## Audited result

The checkpoint gate passed. The stronger 20%-below-Gaussian research target
did not pass and is not claimed.

| Metric | Independently audited result |
|---|---:|
| Original BF16 source matrices / weights | `18 / 28,311,552` |
| Physical container bytes / bits | `8,847,360 / 70,778,880` |
| Exact physical rate | **`2.5 bpw`** |
| Logical arithmetic payload | `70,677,314 bits` |
| Used payload / charged zero tail | `8,834,670 / 7,129 bytes` |
| FP64 SSE / source energy | `500.39553685426534 / 16192.89450885593` |
| Energy-weighted original-source relative MSE | **`0.030902167403153148`** |
| Frozen-v2 MSE ceiling | `0.04985939119332436` — pass |
| Same-rate Gaussian reference | `0.03125` |
| MSE below Gaussian | **`1.113064309909928%`** |
| `F = MSE * 2^(2R)` | `0.9888693569009007` |
| `s = -0.5 log2(F)` | `0.008074080480766676 bpw` |
| Same-rate 20%-below target MSE | `0.025` — **fail** |
| Mean cold byte amplification | **`1.1231235080295139x`** |
| Mean cold 4-KiB-page amplification | `1.1300925925925926x` |
| Worst cold byte / 4-KiB amplification | **`1.161114501953125x / 1.1694444444444445x`** |

The expert-level physical read ledger is:

| Expert | Required blocks | Cold bytes | Cold amp | 4-KiB page bytes | Page amp |
|---|---|---:|---:|---:|---:|
| L5/E18 | `0,1,12` | `1,617,046` | `1.0966295030x` | `1,622,016` | `1.1000000000x` |
| L12/E7 | `2,3,12` | `1,622,292` | `1.1001871745x` | `1,634,304` | `1.1083333333x` |
| L18/E20 | `4,5,13` | `1,660,070` | `1.1258070204x` | `1,671,168` | `1.1333333333x` |
| L28/E83 | `6,7,13` | `1,650,833` | `1.1195427789x` | `1,662,976` | `1.1277777778x` |
| L36/E76 | `8,9,14` | `1,674,304` | `1.1354600694x` | `1,683,456` | `1.1416666667x` |
| L45/E41 | `10,11,14` | `1,712,133` | `1.1611145020x` | `1,724,416` | `1.1694444444x` |

The independent auditor causally decoded and canonically re-encoded all
`15/15` streams with literal payload equality, restored every one of the
13,824 groups exactly once, and rehashed/scored `18/18` original BF16 source
files. The compact verifier binds all fifteen encoder transcripts to their
literal physical payload slices and legacy-container hashes. All `16/16`
resealed tamper cases were rejected on their intended deep invariant.

Evidence identities:

```text
plan lock       99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d
container       4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b
audit JSON      df2ac4e204f78b30486b25146dfe12f83a8310679e104742f2d6811675f5379a
post-XKLT FP64  af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0
```

The result is a locality checkpoint: it reduces worst audited external
compressed reads from the frozen format's `4.906014x` range-reader cost (or
`6x` full-container read) to `1.169445x` with a much lower MSE. It remains
`23.608669612612587%` above the strict `0.025` final target at this rate.

## Reproduction

The build stage requires CuPy because it computes and verifies the exact BF16
staging-energy ledger on the GPU:

```bash
python strata_expert_local_codec/build_from_v2.py \
  --v2-run /path/to/strata_v2_blind_one_shot_v2 \
  --source-root /path/to/blind_protocol_v2/unblinded \
  --output-dir /path/to/new_expert_affine_run
```

Run the sealed encoder once:

```bash
python strata_expert_local_codec/run_and_pack.py \
  --workspace /path/to/INT2__compression \
  --plan-dir /path/to/new_expert_affine_run \
  --python /path/to/cupy-python \
  --encoder /path/to/strata_v2_codec/polar_encoder.py \
  --polar-repo /path/to/PolarLatticeQuantization
```

Audit from the physical bytes and original sources:

```bash
python strata_expert_local_codec/independent_audit.py \
  --plan-dir /path/to/new_expert_affine_run \
  --output-dir /path/to/new_expert_affine_run/independent_audit \
  --workers 2
```

Build and verify the compact bundle:

```bash
REPRO_RELEASE=/path/to/reproduced_expert_affine_release

python strata_expert_local_codec/publish_checkpoint.py \
  --plan-dir /path/to/new_expert_affine_run \
  --audit-report /path/to/new_expert_affine_run/independent_audit/independent_audit.json \
  --output-dir "$REPRO_RELEASE"

python strata_expert_local_codec/verify_checkpoint.py \
  --release-dir "$REPRO_RELEASE"

python strata_expert_local_codec/checkpoint_tamper_tests.py \
  --release-dir "$REPRO_RELEASE" \
  --output "$REPRO_RELEASE/tamper_report.json" \
  --attach-to-manifest

python strata_expert_local_codec/verify_checkpoint.py \
  --release-dir "$REPRO_RELEASE"
```

`REPRO_RELEASE` must name a path that does not already exist; the publisher
and tamper harness fail closed instead of overwriting prior evidence. The
checked-in release remains available for source-free verification at
`results/qwen/strata_expert_affine_checkpoint`.

The dependency-free verifier recomputes container size/rate, header bindings,
KLT-code regeneration, route semantics, label histogram, stream boundaries and
padding, decoder scales, seeds, zero fill, all fifteen encoder/payload bindings,
required expert blocks, cold bytes, 4-KiB page unions, source-plan bindings,
source-score quotients, and the rate-relative target. Immutable SHA-256 anchors
also require the exact precommitted route bytes, label bytes, and canonical
eighteen-source record set rather than merely a self-consistent replacement.
The tamper harness reseals
each modified file in the outer manifest before testing, so its cases exercise
deeper invariants rather than only file checksums. Its adversarial cases include
a rebound payload mutation, an audit detached from its plan, a comprehensively
rebound source-plan identity, required evidence removal, and a rebound KLT
coefficient/code mismatch.

## Inference integration boundary

An operational MoE kernel should decode the three required blocks once and
consume Gate, Up, and Down without refetching the paired tail. To turn
compressed-object locality into low HBM traffic, the inverse RHT and inverse
XKLT should be tiled or fused with GEMM consumption. Those kernels are future
engineering work and are not represented by the checkpoint's external-read
number.

An N19 extension can remove the paired tail and approach 1x reads, but it needs
a separately calibrated finite-length factor and a complete re-audit. The
checkpoint intentionally chooses the already validated N20/N21 kernels.
