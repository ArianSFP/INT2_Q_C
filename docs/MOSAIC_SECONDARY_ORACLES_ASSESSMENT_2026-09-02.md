# MOSAIC-Q secondary-oracle assessment

Date: 2026-09-02

Status: source-only implementation review. No Qwen checkpoint, decoded coarse
payload, matched-control payload, or completed-result payload was opened for
this work. No number below is a new model result.

## Decision

The next cheap sequence is:

1. exact GF(2) recurrence on authenticated legal four-level label planes;
2. non-dyadic Ramanujan periods on the real coarse residual;
3. capped low-order AR/annihilating filters with inverse-noise pullback; and
4. no BM3D build unless a separate coarse-signature neighbour pretest passes.

This sequence is deliberately narrower than the MOSAIC-Q proposal. It tests
three mathematical hiding places without reopening the graph basis, dyadic
cyclostationarity, another arithmetic coder, or an unbounded source-fitted
dictionary.

## Repository evidence that constrains the search

The audited lower-rate coarse starting point is

```text
R0 = 307/128 = 2.3984375 bpw
D0 = 0.036975150060595235
```

At a final 2.5 bpw, reaching `D <= 0.025` therefore requires removal of
`32.3870222053737%` of coarse-residual SSE. This is materially harder than the
`19.0995%` reduction measured from the separate finite 2.5-bpw baseline.

Two apparent neighbouring routes are already closed at their tested scope:

- CYCLO-FRI4 tested periods `{1,2,4}`. Its independent audit found best
  `F=0.9379899307967997` and stopped before controls with
  `HARD_KILL_ABSOLUTE_DOMINANT_ORACLE_NO_CONTROLS`.
- The coarse-programmed graph/Krylov oracle found a large optimistic
  waterfill gain, but the identical-geometry control reproduced it. The
  Qwen-specific excess was only `0.00015723251757482348 bpw`, versus the
  predeclared `0.03 bpw` floor, and the result hard-killed.

These facts lower the prior for any residual transform selected mainly by
isotropic high-dimensional projection. A new branch must expose source-specific
structure and not merely spend 384 ideal bits in a 4,096-dimensional space.

## Exact finite-field recurrence

The source package implements exact Berlekamp-Massey over GF(2) on the two
Gray bitplanes of each four-level label block. For a plane of length `n`, the
canonical choice is either `n` raw bits or `2L` bits for the shortest exact
recurrence: `L` connection bits plus `L` initial-state bits. Ties select raw.

This is the lowest-cost decisive test because random-looking planes normally
have `L` close to `n/2`; they therefore obtain no logical saving, and physical
headers make them lose. A material exact recurrence is immediately visible.
The implementation emits literal packets rather than an entropy estimate:

- `LRB0`: two canonical raw-or-LFSR planes and CRC32;
- `LRC0`: one role, block offsets, BF16 scales, canonical 64-byte padding and
  CRC32; and
- `LRE0`: fixed Gate/Up/Down-transposed order, complete weight coverage, a
  2.15-bpw lower-bound pad, canonical 4-KiB expert padding and CRC32.

Valid-CRC but nonminimal block modes and equivalent overpadding are rejected.
The literal expert packet is independently compared with the physical ledger,
uses one external packet read, and permits no refetch.

The result is still only a label serializer. A production adapter must first
authenticate what a "legal four-level label" means for the selected codec and
must retain every reconstruction scale, transform and state field. The current
coarse artifact cannot be silently reinterpreted as native INT2 if its actual
reconstruction alphabet or polar state is larger. A miss closes exact LFSR
recoding only; it does not close recurrences with exceptions, label-flexible
trellis search, Reed-Muller, QTT, or general finite-field models.

## Non-dyadic Ramanujan periods

The public period bank is

```text
3,5,6,7,9,10,11,12,13,15,17,19,21,23,25,27,29,31,
33,35,37,41,43,47,53,59,61,63,65,67,71,73,79,83,
89,97,101,107,113,127
```

It is genuinely distinct from the completed `{1,2,4}` CYCLO screen. The
implementation forms real primitive-frequency atoms and one source-independent
canonical QR basis. It reports four noninterchangeable quantities:

- fixed public-prefix projection with free continuous amplitudes;
- fixed public-prefix FP16 amplitudes literally inside 384 bits;
- source-selected support with combinatorial rank and FP16 amplitudes charged
  inside 384 bits; and
- ideal Gaussian waterfill, explicitly marked as having no finite backend.

The source must first reach `D <= 0.025`. Only then may the frozen odd-affine
phase-destruction control and all eight blockwise moment-matched Gaussian
pipelines run. Period, support and model selection repeat inside every control.
Promotion requires at least `0.03 bpw` above the stronger control; raw captured
energy is not a promotion statistic.

## Capped Hankel/annihilating-filter branch

Full source-fitted Hankel SVD is postponed. The cheap first gate uses AR orders
`{1,2,4,8,12}`. Every 4,096-value block pays four order bits plus one IEEE
binary16 coefficient per lag, displacing the same number of the 384 refinement
bits. Initial samples are retained in the innovation sequence rather than
treated as free state.

The important correction is source-metric pullback. Low innovation variance
does not imply low reconstructed MSE when an inverse filter is nearly unstable.
For every rounded filter, the gate computes the finite impulse response and
charges `trace(H H^T)/n`. The remaining innovation distortion is still an ideal
iid-Gaussian diagnostic, not a finite innovation codec. A miss closes only
this low-order binary16 AR family, not arbitrary Hankel rank or nonlinear
dynamics.

## Why BM3D is deferred

Within-expert collaborative grouping remains logically different from a global
graph basis, but it is much more expensive and can overfit through matching.
The graph/Krylov null result provides no current evidence that similarity in
decoded coarse signatures predicts similarity in the residual.

BM3D opens only if a causal, coarse-only nearest-neighbour map beats contiguous
grouping, a frozen within-block permutation, and all matched-Gaussian controls
by a whole-owner lower confidence bound of at least `0.03 bpw`. Residual-based
matching and a second expert-packet fetch are forbidden. Until that pretest
passes, collaborative transform engineering is not warranted.

## Frozen gate and claim boundary

The source-only package is `research/mosaic_secondary_oracles_v0`. It includes:

- exact finite packet replay and physical accounting for recurrence labels;
- capped Ramanujan and AR dominant-oracle mechanics;
- CuPy implementations of the heavy basis, projection, AR and matched-control
  paths;
- source-first control enforcement;
- hostile CRC, canonicality, role, coverage, padding and rate tests; and
- an independent standard-library source verifier.

There is intentionally no production payload adapter. A separate source audit
and one-run launch review must bind an authenticated universal SwiGLU-MoE
adapter before any Qwen pilot. Even a pilot pass would not establish
universality; that requires a frozen transfer to a disjoint SwiGLU-MoE family.
No separately fitted oracle gains may be added, and no dominant-oracle score is
a finite-code success.
