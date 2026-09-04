# HYPERPATH-RG research frontier

Date: 2026-09-04

Execution boundary: local RTX 3060 only.  No RunPod or remote execution.

## Current evidence

The deployable finite baseline remains:

```text
R = 2.5 bpw
D = 0.030902167403153148
F = 0.9888693569009007
```

It is 1.1131% below the Gaussian reference and needs a further 8.9870% MSE
reduction to reach `F=0.9`, or 19.0995% to reach `F=0.8`.

Fixed-label same-layer CBIB is closed on the authenticated Qwen layer-15 panel.
Its best pair grouping showed `0.4597951 bpw` gross private saving, but only
`0.00001073076 bpw` ideal net gain after the necessary common stream and
`-0.0035780649 bpw` after physical charges.  Its read envelope stayed below
`2x`; source advantage, not bandwidth, was the failure.

PAIRPATH remains logically distinct because it changes nearby quantizer labels.
Its r2 hostile audit blocked Qwen execution: the finite encoder used role-local
rather than global Up/Down multipliers, its alternating joint solver could
return an objective `0.07489358865` worse than an already legal independent
assignment, and its decoder accepted an invalid unreplayed tree descriptor.
The r3 repair closes the finite multiplier and descriptor defects and emits a
dominance certificate, but correctly has no hard-kill authority because global
optimality remains unproved.  It remains source-only.  A stronger stochastic
BA relaxation subsequently bounded the exact PAIRPATH topology
`(Up_e,Up_f)+(Down_e,Down_f)` at only `0.00316455 bpw` maximum and
`0.00283499 bpw` mean across the eight pair apertures, even with free
block-conditioned laws.  PAIRPATH payload engineering is therefore stopped.

## Assessment of the proposed branches

### 1. TETRAPATH-4 — memoryless branch now closed

For an expert pair, align the four variables

```text
(Up_e, Down_e, Up_f, Down_f)
```

and jointly consider their `4^4 = 256` legal label tuples.  This tests genuine
four-way synergy that can be invisible to every pairwise statistic.  The XOR
construction is the required positive control: four individually fair bits
with one four-way parity have zero pairwise mutual information and save one bit
per tetrad, or `0.25 bpw`.

The dominant oracle compares equal-flexibility RD frontiers for:

1. four independent unary models;
2. all three disjoint 2+2 pair factorizations;
3. the best label-tree factorization;
4. sparse sign/magnitude parity factors;
5. the full 256-state joint model.

All rates are normalized per source weight.  The decisive statistic is

```text
G4 = R_factorized - R_full + 0.5*log2(D_factorized/D_full).
```

Empirical tables, model selection, and convex time sharing may be free only for
this kill-only oracle.  If the best full-joint envelope is below `0.045 bpw`,
memoryless four-way coding is closed.  `0.22933495044437174 bpw` is the ideal
Up/Down standalone threshold; at least `0.27 bpw` is required before finite
engineering.

Four-way parity is also compatible with `<2x` reads in principle.  If one
bitplane satisfies `d=a xor b xor c`, store the common bit `z=a xor b`, one
private bit for expert 1, and one private bit for expert 2.  Expert 1 recovers
its second bit from `(z,a)` and expert 2 from `(z,c)`.  The three stored bits
retain the one-bit tetrad saving, while either routed expert reads one common
plus one private bit rather than the other expert's stream.  This constructive
Gray-Wyner example is why a four-way survivor must next be projected into
common functions of each expert's local Up/Down pair—not serialized as one
interleaved 256-symbol stream, whose cold read is at least `2x` before framing.

The general finite form is **TETRAPATH-FIBER**.  Let each expert-local
four-bit `(Up-label, Down-label)` word map through a fixed rank-`k` Boolean or
GF(4) function `f`, and jointly move labels until

```text
z = f(q_Up,e, q_Down,e) = f(q_Up,f, q_Down,f).
```

Encode `z` once and encode each local label pair by its index inside
`f^{-1}(z)`.  For a balanced rank-`k` map, the uncoded bit counts per tetrad are

```text
common = k
private per expert = 4-k
total = 8-k
logical per-route amplification = 4 / (4-k/2).
```

Thus `k=1,2,3` have ideal read amplification `1.143x`, `1.333x`, and `1.6x`,
respectively; `k=4` reaches exactly `2x` and is excluded.  A one-bit constraint
can save at most `0.25 bpw`, leaving little distortion margin above the
`0.22933495 bpw` target.  A two-bit syndrome can save up to `0.5 bpw` and is the
more credible finite target.  Before finite overhead, the one-bit map can
tolerate at most a `2.9062%` Up/Down distortion penalty; the ideal two-bit map
can tolerate up to `45.5314%`.  The first public map bank contains sign parity,
magnitude parity, their two-bit product, bitwise/GF(4) Up-Down sum, and small
rank-1 through rank-3 affine maps.  Selector bytes and page padding are charged
in the finite stage.

The independent hostile audit passed the XOR/fiber mechanism but rejected the
original alternating optimizer as either a hard-kill or promotion oracle.  An
exact two-coordinate counterexample improved its objective by
`0.0023643742362364`; family-specific smoothing also broke full-model
containment.

The replacement Qwen experiment used a CuPy Blahut--Arimoto relaxation.  It
gave each 2,048-coordinate block a free stochastic 256-state channel and free
time sharing, making the full four-way side strictly more favourable than a
finite deterministic entropy code.  Across all eight fixed layer-15 expert
pairs at a 1/64-block aperture:

```text
full versus independent, mean raw gain                 0.06235429927 bpw
full versus independent, mean control-corrected gain   0.02861814222 bpw
largest full-versus-independent corrected pair         0.04600639092 bpw
full versus best 2+2, largest raw gain                 0.03087324830 bpw
full versus best 2+2, best corrected gain             -0.00028168362 bpw
full versus best 2+2, mean corrected gain             -0.00187063042 bpw
```

Even the raw, control-free irreducible envelope stayed below the `0.045 bpw`
continuation threshold.  The apparent expert-32/40 survivor was therefore
pair-factorizable, not irreducible four-way structure.  A 240-iteration rerun kept the irreducible
control-corrected result negative (`-0.00181322834 bpw`).  Fixed nearest-label
four-way structure was even smaller: maximum raw `0.00003870032 bpw` and
maximum permutation-corrected `0.00001369844 bpw`.

Disposition: stop TETRAPATH-FIBER and any larger coordinate-memoryless joint
table.  This result does not close multiscale RENORM-Q, spatial COCHAIN-Q, the
pairwise flexible route, or legal six-plane STRATA-RM6.

The surviving 2+2 attribution was overwhelmingly `(Up_e,Down_e)` plus
`(Up_f,Down_f)`, not same-role coupling between experts.  Its
control-corrected advantage ranged from `0.0102643` to `0.0521481 bpw` and
averaged `0.0316639 bpw`.  This is useful evidence for an expert-local
role-joint flexible quantizer with essentially `1x` expert reads, but it is far
below the `0.22933495 Up/Down bpw` standalone requirement and does not justify
a packet by itself.  It also agrees with the earlier conclusion that
cross-expert common/private coding is not where the measured structure lives.

### 2. RENORM-Q — highest ceiling, conditional launch

The promising part of RENORM-Q is not the statistical-physics vocabulary.  It
is the search for a small collective variable that predicts distant labels
after a buffer while costing one or two bits per shared block.  This can turn a
global parity or phase into a local tree state instead of requiring a huge WFA
memory.

The first valid census must use small, shared maps only:

- Boolean parity and affine maps;
- thresholded counts;
- quaternary modular sums;
- role-coupled plaquette maps;
- tiny decision diagrams.

Maps are trained on auxiliary layers and evaluated on whole held-out layers.
Their bytes are charged.  Moment-matched Gaussian, block shuffle, role shuffle,
and buffer-destroying controls rerun map selection.  Promotion requires at
least `0.03 bpw` lower-confidence-bound source-specific gain; otherwise the
hierarchy is killed before label-search integration.

The v0 source kernel passed its XOR/IID fixtures and exact tree DP matched
exhaustive enumeration, but independent review blocked any Qwen capability.
It accepted non-Kraft NLL tables (including a zero-bit description of all 16
sequences), allowed caller maps to self-declare zero descriptor cost, included
three maps with unreachable declared states, and implemented the `0.03 bpw`
LCB gate incorrectly.  RENORM-Q remains a promising architecture, but v0 is a
mechanism fixture only until those source-contract defects are repaired.

### 3. COCHAIN-Q — bounded after TETRAPATH

Plaquette and cube differences can expose higher-order Boolean structure, but
an invertible difference transform does not reduce entropy.  The only valid
test jointly moves legal labels to make mixed derivatives sparse:

```text
min_q D(x,q) + lambda*(L(boundary)+L(delta q)+L(delta^2 q)+L(delta^3 q)).
```

The first oracle grants factor parameters free and uses exact small-cell
min-sum.  It must beat both an equally flexible unary baseline and a
phase-randomized control.  A result below `0.045 bpw` closes the memoryless
plaquette/cube family.

The v0 mechanism correctly proved that invertible differencing alone saves
zero bits and that a fixed public even-syndrome fiber can save at most
`0.25 bpw` per affected plaquette bitplane site (`0.125` for a cube) with
ideal expert-local `1x` reads.  Independent review nevertheless blocked Qwen:
the verifier accepted an unlisted executable and omitted a mandatory external
manifest pin, the encoder silently cast invalid floating labels to integers,
and the per-affected-plane rate was not normalized over all expert weights.
A future result must emit one legal six-plane reconstruction and pool original
source SSE; per-plane gains can never be summed.

### 4. HAMILTON-Q — implementation framework, not source evidence

A factor-graph Hamiltonian is a useful common optimizer for pair, tetrad,
parity, and hierarchy factors.  It cannot itself explain a below-Gaussian gain.
It should be built only around factors that survive their own dominant oracle.
The encoder must report convergence bounds or fail open to `HOLD`; a local
min-sum failure must never be used as a scientific hard kill.

### 5. STRATA-RM6 — real legal-label algebraic test

The stronger algebraic branch must operate on all 64 legal STRATA reproduction
states and six causally replayable planes.  Four-level abstractions cannot be
transferred.  The first stage is a source-free exact 64-way distortion KAT and
legal decode; Qwen execution follows only after a bounded Walsh/Gowers/RM
screen shows material structure.

The bounded local-RTX3060 pilot is now complete on one authenticated Qwen
layer-15 expert-0 Up block.  The best zero-coset legal search reduced its own
RM-SC initialization MSE by `5.23756%`, but the identical moment-matched
Gaussian path improved by `7.07671%` (`-1.83915` Qwen-specific percentage
points).  Its selected stream was `1,408` bytes or `2.75 bpw`; all nine Qwen
checkpoints exceeded `2.5 bpw` and were rejected by the immutable packet
codec.  The tested RM(5,12)^6 local-bank coordinate-descent route is therefore
stopped.  This does not transfer its very high local-code relative MSE to the
deployed global STRATA codec and is not a converse for every RM6 construction.

### 6. BAYES-384 and sheaf coupling — hold behind dominant oracles

BAYES-384 requires a validated conditional residual prior; otherwise adaptive
questions only repartition an effectively Gaussian residual.  A sheaf/global-
section model receives a free continuous-capture oracle first and dies below
the required capture.  Neither is allowed to become a large implementation
project without evidence.

## Flagship architecture if the gates survive

The coherent successor is HYPERPATH-RG:

```text
exact legal distortion fields
  -> four-way role/expert factors
  -> shared multiscale collective variables
  -> sparse plaquette/cube defects
  -> exact or bounded min-sum flexible-label search
  -> integer conditional probabilities
  -> expert-local common/private packet
  -> posterior centroids learned without source leakage
```

The architecture is universal only if its decoder depends on packet bytes,
shape, role, expert membership, and public coordinates—not Qwen identity,
provenance, an external checkpoint, or untransmitted source-derived state.

## Execution order

1. Retain PAIRPATH r3 as a corrected source mechanism, but do not build its
   payload after the stronger BA aperture hard kill.
2. Keep the intra-expert role-pair BA signal separate from irreducible
   higher-order claims;
   test whether it survives cross-layer/held-out modelling and a local packet.
3. Run the already source-audited true STRATA-RM6 local-block pilot on the local
   RTX 3060.
4. Launch bounded exact RENORM-Q and COCHAIN-Q mechanism gates; only build Qwen
   capabilities for source-closed survivors.
5. Emit a finite codec only when a dominant oracle exceeds the standalone
   threshold with enough margin for tables, headers, pages, and controls.

No gains from separately fitted branches are additive.  Final success still
requires one decoded physical object at `2.15--2.5 bpw`, `F<=0.8`, and maximum
routed cold-read amplification below `2x`.
