# POLARIS-SC-v2 final report

Date: 2026-08-31

## Outcome

POLARIS-SC-v2 passes the refined objective: a strict post-training source
codec below 2.15 bits per weight whose matched-Gaussian mean-squared error is
within 5% of the Gaussian rate-distortion limit.

The all-in Qwen3-30B-A3B-shaped budget envelope is
`2.1499176041040124 bpw`. On 32 preregistered, independent, `N=2^18`
standard-normal blocks, the family-wise-99%-corrected upper confidence bounds
are:

| Gate | Mean | One-sided corrected 99% UCB | Threshold | Result |
|---|---:|---:|---:|---|
| Logical arithmetic bits/block | 562,736.8125 | 563,012.4867 | 563,464 | Pass |
| Absolute MSE | 0.0528997614 | 0.0529443693 | 0.0533040635 | Pass |
| Sample-relative MSE | 0.0528995599 | 0.0529844887 | 0.0533040635 | Pass |

The stricter sample-relative UCB is 4.37049% above the Gaussian limit
`0.050765774772264724`, leaving 0.62951 percentage point inside the requested
5% boundary. The absolute-MSE UCB is 4.29146% above the limit.

## Architecture

The quantizer is a six-level Construction-D polar lattice with randomized
successive-cancellation decisions and arithmetic coding. It uses:

- block length `262,144`;
- a 64-point lattice alphabet at spacing `eta=0.25`;
- Gaussian test-channel distortion `D=0.05110`;
- a frozen six-level reliability map derived from the public polar-lattice
  construction;
- deterministic per-block frozen cosets and no learned parameters;
- an FP16 reconstruction scale for each block.

The new systems contribution is a checkpoint-global fixed-capacity overflow
reservoir. Each block has a 32-bit logical-length entry and FP16 scale, while
only the logical arithmetic bits are concatenated, MSB first. Local messages
may exceed the old per-block allocation. Savings from other blocks absorb the
excursion. The only rate failure is global:

`sum_i logical_length_i > 563464 * block_count`.

The physical reservoir is zero-filled to the exact global capacity. This
makes rate a literal serialized-file property rather than an entropy estimate.

## Bottleneck and breakthrough

The first frozen design used a local 563,496-bit arithmetic slot. It failed
honestly: one holdout required 563,672 bits, an overflow of 176 bits. Yet the
same 16-block set was 10,307 bits below the aggregate old allocation. The
failure was therefore block-local framing, not rate-distortion quality.

V2 changed no quantizer parameter and no reconstruction decision. It replaced
independent local caps with deterministic global risk pooling, then used an
entirely new 32-seed confirmation set. A formerly overflowing block was also
packed, independently extracted, and decoded exactly before v2 was frozen.

## Exact rate ledger

For 30,532,122,624 parameters and 116,470 rank-2 blocks:

| Serialized allocation | Bits |
|---|---:|
| Rank-2 logical payload reservoir | 65,626,652,080 |
| Rank-2 u32 length directory | 3,727,040 |
| Rank-2 FP16 scale directory | 1,863,520 |
| Rank-2 matrix headers | 1,195,136 |
| Rank-1 BF16 payload | 3,375,104 |
| Rank-1 headers | 12,352 |
| Decoder set-map reservation | 3,145,728 |
| Frozen/coset-map reservation | 1,572,864 |
| Global format header | 4,096 |
| **Total** | **65,641,547,920** |

That is 8,205,193,490 bytes or `2.1499176041040124 bpw`, with 2,515,721
ledger bits below the mathematical 2.15 cap (2,515,720 usable bits at whole-byte
granularity). The canonical run also materialized the decoder-map reservation
as an exactly 589,824-byte hashed asset, including zero reserve, rather than
leaving it as an off-ledger assumption. That upstream-derived asset is not
redistributed here; the reproduction builder regenerates and hash-checks its
200,184-byte meaningful map locally.

The 32-block confirmation reservoir is exactly 2,254,144 bytes including its
96-byte header. It contains 18,007,578 logical payload bits inside an
18,030,848-bit fixed payload capacity, leaving 23,270 zero-reserve bits. Its
SHA-256 is
`ad0c35e72b5900ffa6ed353df1bf1b163d912b8bfb692fc8e5b318ea6f9eb3f5`.

## Confirmation integrity

The protocol was frozen before any v2 result existed. It used 32 public,
preregistered seeds, an irreversible workspace-global `O_EXCL` opened lock,
and no retry path. Three one-sided Student-t tests used a Bonferroni correction
for 99% family-wise confidence:

`q = 1 - 0.01/3 = 0.9966666666666667`, `df=31`, `t=2.9080702125010807`.

For every block, a standalone parser validated the full physical reservoir,
hashes, checked prefix arithmetic, and zero suffix. A fresh-process decoder that
imports no encoder implementation regenerated the causal probabilities and
source, decoded every decision, and reproduced both encoder MSE values within
`1e-12`. All 32 blocks passed. Frozen artifacts were rehashed after completion.

A separate post-result audit then performed 452 checks across all source
containers, extracted records, decoder JSON files, logs, hashes, timestamps,
statistics, and raw reservoir bits. The largest encoder/decoder discrepancy
was `1.388e-17` for absolute MSE and `2.776e-17` for relative MSE. It found no
substantive discrepancy.

Key hashes:

- Harness: `b9db626f716461d4932bffc52a85ba1c3e46ace70e8d9713eae5b6fcd37413b2`
- Manifest: `06967a4e852c9d39c97fe39b45d50df558471e0f35912d330b6dc1e7493df5e0`
- Protocol: `107c97381aebec49423270fae378a5dfaa6b75415671f026e8d60bd19f487004`
- Result summary: `f4988f8e92b99fa90a4fe2b6b153beb02a1cb123c49e21f64d90ce77f008e6b5`

## Additional breakthrough probe

A development-only proof carried one arithmetic-coder state across all 16
already-open v1 blocks. It regenerated and round-tripped 17,347,696 causal
decisions exactly, removed every block length, and saved a further 18 logical
bits versus separately finalized coders. This proves that production v2.1 can
use 16- or 32-block continuous superframes to reduce directory cost while
retaining parallelism. It was deliberately not substituted into the frozen v2
confirmation because the simpler reservoir was already independently audited.

## Claim boundary

This result establishes the requested matched-Gaussian source-code/PTQ core.
It does not claim that a complete Qwen checkpoint has been encoded and run for
perplexity or downstream accuracy. The checkpoint ledger uses the cached Qwen
tensor schema, but a final model deployment still needs the full RHT/block
transform, complete checkpoint serialization, whole-model decode, and model
evaluation. The 2.25 MB confirmation reservoir is physically emitted; the
8.205 GB full-checkpoint figure is a shape-verified fixed-capacity envelope,
not an emitted full checkpoint. The current Python implementation is an auditable reference; a
production encoder/decoder should move the serial arithmetic loop to compiled,
file-backed code while retaining the same bitstream tests.

The original request for 50% lower raw MSE cannot coexist with 2.15 bpw on a
matched Gaussian source: half of the prior approximately 0.0694 MSE would be
below the Gaussian rate-distortion lower bound. The later, physically possible
goal—within 5% of that lower bound—is the one proven here.

Primary construction: <https://arxiv.org/html/1501.05683v5>  
Pinned official code: <https://github.com/graceBaoXP/PolarLatticeQuantization/commit/458187b9b03db1768a4b72d617e591f7862f6fca>
