# MOSAIC-Q research assessment

Date: 2026-09-02

## Objective and accounting contract

The research target remains one literal post-training weight codec satisfying

```text
2.15 <= R_physical <= 2.5 bpw
F = D_relative * 2^(2 R_physical) <= 0.8
worst routed cold expert read < 2x
```

The intended endpoint is a universal SwiGLU-MoE codec. A Qwen pilot is an
experiment, not codec identity: shape, role, public coordinates, and bytes in
the selected expert packet may be decoder-visible, but Qwen identity, a
related checkpoint, or source-derived state is not free. A universal claim
requires a frozen codec and transfer to a disjoint SwiGLU-MoE family.

At the audited finite 2.5-bpw baseline,

```text
D = 0.030902167403153148
F = 0.9888693569009007
```

so the exact 2.5-bpw target is `D <= 0.025`: a 19.0995257% reduction from
that baseline, or a rate-equivalent source advantage of
`0.15288996696 bpw` at unchanged distortion.

The independently decoded lower-rate TACTIC coarse object has a different
starting point:

```text
R0 = 307/128 = 2.3984375 bpw
D0 = 0.036975150060595235
F0 = 1.0278108682335156
```

Consequently, a literal refinement to 2.5 bpw must remove

```text
1 - 0.025/D0 = 0.32387022205373717
```

or 32.3870222% of the coarse residual SSE. Reusing the 19.10% threshold for
this residual would be wrong.

## Main conclusion

The proposal's central discontinuity is valid:

> Do not only compress labels selected by the old nearest-neighbour codec;
> jointly select legal reconstruction labels and their structured physical
> description.

That is the most important open direction after the negative conventional
screens. It does not, however, create entropy. A label-flexible code can beat
the old finite quantizer or expose source structure hidden by its labels, but
it cannot beat the Gaussian rate-distortion curve on a truly iid Gaussian
source. Every experiment must therefore rerun the same label search and model
selection on complete matched-Gaussian PTQ controls.

Three proposed branches have been converted into source-frozen experiments:

1. **ε-TCQ / causal-state joint label selection.** Exact STRATA inspection
   proves that a coordinate's 0..63 reconstruction index is assembled from six
   complete level-major polar passes, not six coordinate-local arithmetic
   events. The audited v1 adapter retracts that invalid ABI. A dense beam-32
   state costs 7,147,102,208 bytes; the audited ragged/COW design lowers the
   complete beam-32 peak to 1,610,756,864 bytes, including explicit leaf
   buffers, but compute and exact device-COW semantics remain on hold.
2. **LOGIC-Q algebraic label selection.** A capped v1 now has executable
   mixed-family pre-search, joint two-scale/RM(1) scoring, canonical expert
   packets, and a real CuPy path. Independent audit found three remaining
   production bindings: publicly resealable selection receipts, trusted
   encoder-side score objects, and a name-only CuPy guard. No Qwen access is
   authorized until a narrow successor closes them.
3. **Coarse-programmed graph/Krylov refinement.** The decoder-legal optimistic
   gate has completed and hard-killed. Its best continuous Qwen waterfill gain
   was `0.30344543088152887 bpw`, but the strongest identical-geometry control
   achieved `0.30328819836395404 bpw`; the source-specific excess was only
   `0.00015723251757482348 bpw`.

## Evidence obtained in this iteration

### Real lower-rate coarse object

The v1 independent audit:

- parsed and causally decoded all 18 streams;
- verified every inverse state as little-endian I32;
- reproduced every canonical symbol and the complete `COARSE.bin`;
- rescored exact externally pinned BF16 Gate, Up, and transposed-Down bytes;
- recomputed `D0=0.036975150060595235` without using producer `RESULT.json`
  as a numerical input.

The coarse packet is 1,414,656 bytes and reads in one external pass during the
audit. This is file-I/O evidence, not a measured accelerator inference-HBM
claim.

### TACTIC-DH384 decisive hard kill

The frozen rank-384 continuous parent captured only
`0.09361861769116196` of Qwen coarse-residual SSE. The implemented rank-376
subset captured `0.0916533630916034`. Both are far below the required
`0.3238702220537387`.

The rank-384 result is essentially its isotropic dimension share
`384/4096 = 0.09375`. Thus the fixed dyadic basis found no material
source-aligned concentration. Since every finite v3 codeword lies in the
failed parent span, the run correctly emitted no composite.

### Fixed-label long-range WFA

The earlier χ=64 WFA saving of `0.1675415 bits/symbol` was a source-free
parity fixture, not Qwen evidence. The raw Qwen producer selected a two-state
suffix model but reported negative physical saving and `F>1`; its corrected
independent numerical replay remains the authority required before closing
the fixed-label branch. Even a confirmed negative result would not close
label-flexible ε-TCQ or posterior reconstruction.

### Posterior centroid repair

Posterior-centroid v0 failed before scoring because authenticated v8 code
strictly required built-in `int`, while NumPy iteration returned `np.int64`.
V1 wraps the exact authenticated module and normalizes only values accepted by
`operator.index`, then rechecks range, length, uniqueness, and complete
coverage. It rejects booleans, floats, duplicates, missing rows, and
out-of-range values. The exact v8 unpack route now passes 25 hostile tests.
The failed v0 namespace remains invalid and is not reused.

### ε-TCQ exact integration and memory gate

The source-frozen v1 adapter and independent audit pass 14 hostile tests and a
source-free CuPy top-k smoke. They establish that legal search must carry a
whole resumable polar state. The straightforward beam-32 representation at
`N=2^21` is 7,147,102,208 bytes, so it is not a viable production state.

A second ragged-state derivation uses the exact identity that active
likelihood cells total `N-1` per path rather than `(N/2) log2(N)`. Its audited
complete beam-32 peak is 1,610,756,864 bytes; the simultaneous RTX 5090 CuPy
pool measured 1,610,762,240 bytes. This makes memory capacity a GO. It is not a
throughput GO: the frozen worst-active upper ledger includes
8,455,716,864 likelihood updates, 4,227,858,624 partial-sum writes,
3,825,205,440 partial-sum XORs, 4,227,858,432 polar XORs, and a complete
one-path six-pass winner replay with another 264,241,152 likelihood updates.
Device COW/fork semantics, persistent six-pass execution, and exact Q0.16/FMA
boundary equivalence remain unimplemented, so payload access stays blocked.

### LOGIC-Q capped adapter audit

V1's exact closure, 33 hostile tests, and finite fixture pass. The independent
RunPod audit also exercised a real CuPy 14.2.0 RM path on an RTX 5090 and
independently decoded its 512 labels.

Three adversarial probes retain a production HOLD:

- a different frozen configuration can be inserted into a receipt and the
  public self-hash recomputed without `authorize_test` replaying selection;
- the pooled scorer authenticates packet syntax but trusts encoder-provided
  SSE, energy, and label-count objects; and
- the live-backend guard accepts an object whose public name is merely
  `cupy`.

These are orchestration defects, not negative evidence for RM, GF(2), QTT,
BDD, or algebraic label-flexible quantization.

## Assessment of the proposed branches

### 1. ε-TCQ and causal states — high scientific value, hard integration

Entropy-constrained trellis-coded quantization already establishes the broad
joint distortion/rate idea. CSSR and spectral Hankel methods provide ways to
infer decoder-replayable predictive states. The research novelty is narrower:
a learned, source-frozen causal state over legal low-bit weight-code decisions,
joint label selection, state-conditioned centroids, full physical packets,
and MoE-local decoding.

The essential correction is that the current STRATA polar stream is globally
coupled within a block. A valid successor must either:

- search block-level polar paths while carrying the complete resumable SC
  state, arithmetic state, WFA state, and reconstruction state; or
- define a genuinely new direct-INT2 codec and charge its complete coarse
  packet, rather than aliasing four labels to current STRATA indices.

The ragged-state result makes the first path memory-feasible but not yet
compute-feasible. Any promotion must preserve all six level transitions,
survivor ancestry, level-boundary state materialization, and the final complete
SC replay; a one-level dense/ragged parity fixture is insufficient.

Promotion requires literal packet rescoring, whole-owner cross-fitting,
state-aware centroids beating both label-only and occupancy-preserving
state-scrambled controls, eight full Gaussian PTQ reruns, and a strict
one-pass expert layout. Cross entropy alone is not enough.

### 2. LOGIC-Q — high novelty, corrected rate claims

The proposal's rank-680 calculation mixes an information count with a finite
format. For one `768 x 2048` role and two bitplanes:

```text
raw U,V factors: 2*r*(m+n) = 3,829,760 bits/role
raw operational rate at r=680 = 2.4348958333 bpw
```

This already exceeds 2 bits per label before headers, scales, exceptions, and
alignment. The smaller

```text
2*r*(m+n-r) ~= 2,904,960 bits/role
1.8469 bpw
```

is an asymptotic quotient/counting dimension. The exact finite count is
2,904,964 bits/role, but no canonical enumerative serializer or rank-680
weighted optimizer is implemented. Weighted low-rank approximation over
GF(2) is NP-hard even at rank one, so rank 680 cannot be treated as an easy
SVD analogue.

Likewise, a global RM(3,23) function has only 2,048 coefficients per
bitplane, but Qwen occupies only 56.25% of the naive `2^23` coordinate cube.
The code must define a valid mixed-radix or punctured domain and solve a
weighted soft-decision problem; cheap coefficients do not imply a close
codeword. Blockwise RM descriptors are honest but repeat: RM(3,11) over 2,048
sites costs 0.2265625 bpw before exceptions and framing.

The bounded source package therefore implements only what can be checked:
exact small RM(1), bounded lists, exact tiny/bounded rank<=12 GF(2), finite
exceptions, and bounded ROMDD. RM(2/3) RPA, QTT, canonical rank serialization,
and global rank-680 search remain future work, not results.

The capped v1 successor improves the live bank but remains intentionally
narrow: literal labels, capped RM(1)+exceptions, and depth-0/2/4/6 ROMDD.
GF(2) search is not scheduled. A production-bound v2 must recompute selection
from literal row receipts, bind the real CuPy module/device at launch, and use
an independent source scorer before any Qwen result can be authoritative.

#### Required current-codec bridge: STRATA-RM6

The capped four-level packet is not an adapter for the audited STRATA codec.
STRATA exposes a 64-way reconstruction index assembled by six complete
level-major polar passes. A valid algebraic successor must therefore optimize
one exact 64-entry distortion table per coordinate and constrain the six
completed index bitplanes, rather than replacing one four-level label after the
fact.

For a 4,096-coordinate block, an RM code has `m=12`. The particularly useful
rate point is

```text
dim RM(5,12) = sum_{j=0}^5 C(12,j) = 1,586 bits/plane
six planes = 9,516 bits = 2.3232421875 bpw
```

This leaves 724 bits in a literal 2.5-bpw block for scale, family/profile,
framing, CRC, and alignment. It is therefore a real physical candidate, not
only a coefficient-count argument. It is also a severe code constraint: a
matched iid source may pay large distortion. The first gate must use a frozen
bank of per-level orders whose total dimensions fit the literal packet, perform
soft weighted RM/sub-RM decoding from exact source-domain distortion
increments, reconstruct all six planes back into 0..63 indices, and repeat the
complete selection on matched-Gaussian PTQ controls. Alternating per-plane
updates may screen the idea, but only a jointly rescored index packet can
promote. No four-level LOGIC-Q result may be transferred to this bridge.

There are two different experiments and their rates must not be conflated.
The 4,096-site calculation above is a new locally blocked direct packet. The
existing expert codec instead uses `N=2^20` and `N=2^21` polar blocks; its six
selected-position fractions can sum to more than 4 positions/weight because
the selected SC decisions are subsequently arithmetic-coded. A cheap
current-codec experiment can exploit the fact that RM and polar codes use the
same Arikan transform: retain each current level's selected count and replace
the BEC reliability ordering by row-Hamming-weight/RM ordering. Unless the
count lands on a complete RM dimension this is an RM-ordered truncated polar
code, not an exact RM code, and its physical rate is the emitted arithmetic
packet rather than the selected-row count. It must compare zero/low-complexity
frozen cosets with the current procedural random coset; a random frozen coset
can erase the coordinate-function interpretation even when the row set is RM.

### 3. Coarse-programmed graph, BM3D, Hankel, and Ramanujan — bounded oracles

Using paid coarse bytes to define a graph or match map is legitimate and
descriptor-free. It does not make residual information free. The first gate
therefore compares Qwen capture with two controls under the identical graph:
a within-block odd-affine permutation and a moment-matched Gaussian residual.

That gate is now closed. The winning `coarse_signed_path_dct` fixed first-384
span captured `0.0939015892002368`, almost exactly the isotropic dimension
fraction. Its optimistic free-support oracle captured `0.422096957207252`, and
its continuous 384-bit waterfill captured `0.343389767350314`, nominally
reaching relative MSE `0.0242782618835445`. Controls reproduced essentially all
of that apparent gain: Qwen-minus-control was just
`0.00015723251757482348 bpw`, below the predeclared `0.03 bpw` kill floor.
Consequently no finite graph codec was emitted and no graph result may be
nested into another branch.

The secondary screens remain logically distinct hypotheses rather than graph
successors. If opened, they are:

- within-expert coarse-signature patch groups with a fixed Hadamard/DCT/Haar
  collaborative transform;
- coarse-seriated Hankel/Krylov innovation rank;
- non-dyadic Ramanujan periods, because the earlier cyclostationary screen
  covered only periods 1, 2, and 4.

All must operate from one buffered coarse-file pass. A second storage fetch is
not permitted; host scratch and accelerator HBM remain separate ledgers.

### 4. Bispectral Volterra lifting — conditional later gate

Sparse quadratic/cubic prediction is mathematically distinct from covariance
and can be embedded in an exactly invertible lifting transform. It is worth
opening only if a fixed auxiliary-data bicoherence bank predicts at least 5%
of held-out residual SSE and beats a phase-randomized surrogate preserving
the full power spectrum. Dense source-fitted Volterra coefficients would be
too expensive and too easy to overfit.

### 5. Sheaf, two-adic, redundant frames, and program synthesis — moonshots

These remain scientifically interesting but have lower expected value:

- a sheaf oracle should first give restriction maps away and demand at least
  20% continuous capture;
- finite-field and two-adic recurrence screens are cheap diagnostics, but a
  few significant recurrences cannot supply hundreds of bits per block;
- redundant frames can improve finite search or noise shaping, but cannot
  beat Gaussian RD without measured source structure;
- an MDL program language charges decoder programs but does not by itself
  prevent selection overfitting. Program selection must remain outside whole
  test layers and disjoint model families.

## Read-bandwidth implications

The target runtime form remains one expert-local packet read once into the
expert cache, followed by materialization of native packed INT2. Coarse-derived
graphs, match maps, or transform selectors are acceptable only when generated
from buffered bytes. The literal DH384 layout was exactly 2.5 bpw and one
external packet pass, but it failed MSE before emission. The v6 coarse audit
measured one file pass and approximately 1.002--1.005x page alignment, while
explicitly declining an inference-HBM claim.

An architecture is not below 2x merely because its stored fields sum to less
than two packets. A second expert-file fetch, repeated common-page request, or
unreported model/centroid page must be included. Host parsing, D2H/H2D, HBM,
and external storage traffic are reported separately.

## Promotion order

1. Finish the independent fixed-label WFA replay.
2. Preserve the completed graph/Krylov hard kill; do not spend finite-code
   engineering on a `0.00015723251757482348 bpw` source-specific excess.
3. Finish ε-TCQ's ragged/COW six-level semantics and compute gate; memory alone
   has passed, while payload authority remains held.
4. Repair LOGIC-Q's selector, source scorer, and actual-CuPy launch bindings;
   do not open Qwen under the audited v1 HOLD.
5. Run posterior-centroid v1 only after a passing WFA audit receipt and a
   separately bound launch review.
6. Run exact GF(2) recurrence, non-dyadic Ramanujan, and capped Hankel/AR as
   independent source-first gates; defer BM3D grouping and Volterra until a
   cheaper relational or higher-order diagnostic survives.
7. Promote only one literal nested packet. Never add gains from separately
   fitted decompositions.

## Prior-art boundary used for this assessment

- Entropy-constrained trellis quantization predates this work; the proposed
  novelty is the legal weight-code state and physical MoE integration.
- CSSR reconstructs causal-state models from discrete sequences:
  <https://bactra.org/CSSR/>.
- Recursive projection-aggregation provides practical soft decoding for RM
  codes: <https://arxiv.org/abs/1902.01470>.
- Binary low-rank approximation is NP-hard even at rank one:
  <https://arxiv.org/abs/1511.01699>.
- BM3D's relevant mechanism is grouping similar patches before a collaborative
  3D transform: <https://webpages.tuni.fi/foi/GCF-BM3D/index.html>.
- Ramanujan subspace pursuit targets exact integer-period components:
  <https://arxiv.org/abs/1512.08112>.

This is a research-novelty boundary, not a patentability opinion.
