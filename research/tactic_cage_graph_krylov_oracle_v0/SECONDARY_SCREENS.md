# Decoder-legal secondary coarse-residual screens

These are frozen bounded follow-ons, not executed results. They reuse the one
buffered N18-v6 frame, decoded I32 symbols, F32 reconstruction, and FP64
residual produced by `run_oracle.py`. They may not reopen `COARSE.bin`.

Every screen follows the same discipline:

* graph/group/order/period candidates use only coarse symbols, role/shape and
  public coordinates;
* the residual is used only for a clearly labelled continuous containment or
  ideal rate-distortion score;
* a source survivor alone opens its permutation and block-moment Gaussian
  controls;
* promotion is based only on Qwen-minus-the-stronger-control excess gain;
* any finite coefficients, supports, references, ranks, selectors, seeds,
  headers and padding are charged inside the 384 fine bits per 4,096 values;
* all extra host/HBM passes are reported separately and confer no inference
  read authority.

## S1 — coarse-signature collaborative patch groups

Split a 4,096 block into 64 contiguous patches of 64 values. A patch's
signature is the frozen vector of coarse-symbol count sketch, signed sum,
magnitude sum, and eight Walsh sketch values. For patch `i`, search only the
32 preceding patches with the smallest coarse-signature distance; ties are
resolved by coordinate. Form at most eight-member causal groups. No residual
distance or model identity may select a neighbour.

Containment metrics:

1. exact source-domain SSE after an oracle separable 2-D Walsh/DCT group
   projection at every retained rank 1..384 per block;
2. ideal 384-bit Gaussian waterfill over the fixed group-transform modes;
3. Qwen-minus-permutation/Gaussian excess capture and rate gain;
4. comparison with the public contiguous-patch grouping at every rank.

This is related to a BM3D collaborative transform, but the matching is
decoder-legal and derived from coarse signatures. In a finite causal codec,
deterministic matches cost zero descriptor bits; any alternate match index,
gain, group size, or transform selector is charged. Referencing a prior fine
patch requires it to be decoded and buffered inside the same expert; no
second expert-frame read is permitted.

Hard kill: the free-support continuous envelope misses relative MSE 0.025, or
the ideal charged-bit score misses 0.025, or Qwen-minus-the-stronger matched
control is below 0.03 bpw. A finite build additionally requires descriptor
and reference bytes to fit the literal 384-bit block budget.

## S2 — coarse-seriated Hankel/displacement rank

Use the already frozen signed-symbol seriation. For each 4,096 residual block,
construct a 256-by-3,841 overlapping Hankel lift. Screen ranks
`{1,2,4,8,16,32}` with CuPy SVD or randomized range finding. The scored
reconstruction is not Hankel Frobenius error: anti-diagonals are averaged back
to 4,096 source coordinates and exact FP64 source SSE is reported. A public
native-order Hankel lift is the rank baseline.

The singular vectors and recurrence coefficients are source-derived and free
only in the continuous oracle. A finite annihilating-filter codec must charge
initial state, poles/coefficients, innovation bits, rank and numerical format.
The gate reports the minimum coefficient description necessary under binary16
and fixed dyadic coefficient variants before promotion.

Hard kill: no tested rank reaches the target even with free parameters; or
the best source-domain gain above both matched controls is below 0.03 bpw; or
the minimum finite coefficient/header charge leaves too few of the 384 bits
for innovations to retain the target. Hankel-lift energy alone can never pass.

## S3 — non-dyadic Ramanujan/phase energy

The frozen period bank is

```text
3,5,6,7,9,10,11,12,13,15,17,19,21,23,25,27,29,31,
33,35,37,41,43,47,53,59,61,63,65,67,71,73,79,83,
89,97,101,107,113,127
```

Construct orthonormalized Ramanujan-sum subspaces in native order and after
the coarse signed-symbol seriation. Periods and ordering are public. Report
source energy captured at every cumulative dimension 1..384, the isotropic
`k/4096` baseline, an ideal 384-bit waterfill, and phase-preserving versus
phase-destroying controls. Per-block residual-selected periods are a
source-leaking oracle and their selectors must be charged in a finite codec.

Hard kill: maximum Qwen-minus-control excess is below 0.03 bpw, ideal
rate-shaped relative MSE exceeds 0.025, or selector/phase bytes erase the gain.

## Routing and bispectral boundary

The three screens are independent branches and may run after the graph gate
whether it survives or hard-kills, but only from the already buffered state.
They execute in cost order: Ramanujan, collaborative patches, then Hankel.
Each screen independently enforces source-before-control opening. No gain is
added across branches until one nested reconstruction is rerun.

Bispectral/Volterra prediction remains deferred. It opens only if the
Ramanujan screen shows at least 0.03 bpw Qwen-minus-control excess and a cheap
fixed third-order phase statistic predicts at least 5% of the remaining
coarse SSE on a held-out block partition. Its first gate is limited to eight
public lag pairs and 32 dyadic coefficients; otherwise it is killed without a
dense quadratic implementation.
