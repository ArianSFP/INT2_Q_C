# TACTIC-CAGE research assessment

Date: 2026-09-02  
Scope: universal, weight-only PTQ for arbitrary SwiGLU-MoE expert triplets,
raw-weight MSE, `2.15 <= R <= 2.5` physical bpw, and routed cold reads below
`2x`.

## Decision

The proposal contains two worthwhile ideas that remain scientifically open:

1. use an already transmitted coarse codeword as decoder-visible conditioning
   for the geometry of its refinement; and
2. score a non-local source model by both its operational symbol rate and the
   MSE of its posterior centroid.

The second is the higher-value immediate addition.  The first is a legitimate
conditional successive-refinement architecture, but the coarse codeword does
not provide free information.  A graph, transform, selector, prior, or trellis
that is a deterministic function of the coarse packet carries no information
beyond that packet.  Its opportunity is to organize the remaining conditional
codebook efficiently.

The corrected order is therefore:

```text
finish and independently audit the UWFA-SC source-model census
    -> run one joint rate-and-posterior-centroid diagnostic
    -> independently audit both the N18-307 producer and frozen DH384 cell
    -> run the source-free smoke and three-stream actual-coarse pilots
    -> test separate frozen upper bounds for DH384 and broader CAGE branches
    -> build the full 307/128-bpw coarse artifact if any branch survives
    -> execute frozen DH384 only if its own branch survives
       (otherwise continue directly with the surviving CAGE branch)
    -> test a coarse-derived graph/Krylov dominant oracle
    -> test a bounded adaptive refinement tree
    -> consider syndrome or bits-back only after a measured conditional law
```

Building all 108 lower-rate streams before the existing optimistic
three-stream pilot would reverse the repository's efficient early-kill ladder.
There is presently no Qwen result for TACTIC-DH384 or TACTIC-CAGE.

## Exact target and two different gaps

The independently decoded finite baseline is

```text
N  = 28,311,552 weights
R0 = 2.5 bpw
D0 = 0.030902167403153148
F0 = D0 * 2^(2 R0) = 0.9888693569009007
s0 = -0.5 log2(F0) = 0.008074080480766676 bpw.
```

At `R=2.5`, the target `F<=0.8` requires `D<=0.025`, hence a direct
distortion reduction of

```text
g_required = 1 - 0.025/D0 = 0.19099525693951513.
```

Keeping the current reconstruction unchanged instead requires the ideal net
rate saving

```text
Delta R_ideal = 0.15288996696291446 bpw.
```

That ideal value is not the finite page-aligned pass line.  The current
artifact occupies `8,847,360` bytes.  The unaligned same-reconstruction ceiling
is `8,306,290.968...` bytes, while the largest passing 4-KiB-aligned object is
`8,302,592` bytes.  A literal recode must therefore save at least

```text
544,768 bytes = 0.15393518518518517 bpw.
```

At that page point `R=2.346064814814815` and
`F=0.798841655309746`.

The frozen TACTIC ledger is a different experiment:

```text
coarse N18-307 stream       307/128 = 2.3984375 bpw
384 fine bits / 4096        12/128 = 0.09375 bpw
global/private metadata       1/128 = 0.0078125 bpw
total                       320/128 = 2.5 bpw
cold read                              73/72 = 1.0138888889x.
```

The fine field is `384 bits per 4,096-weight block`, not 384 bits per expert.
There are 1,152 such blocks in one evaluation expert, so its fine field is
`442,368 bits = 55,296 bytes`.  Reallocating bits across blocks is a different
format with new framing, random-access, and cold-read semantics.

The ledger also has **zero metadata slack**.  The `24,576`-byte global packet
plus six `512`-byte expert headers already consume the complete `1/128` metadata
share (`27,648` bytes).  A WFA, graph kernel, posterior table, latent model, or
new framing field cannot be appended at 2.5 bpw: it must replace existing
selector/QC/schema capacity or displace coarse/fine payload under a newly
measured rate-distortion split.

Those equalities hold only for the divisible Qwen panel.  A fixed full coarse
reservoir for a partial 262,144-value tile and a fixed 48-byte fine field for a
partial 4,096-value block can make rate and owner-page amplification arbitrarily
large on small or unequal legal shapes.  Every CAGE packet needs a
shape-derived tail/fallback format and an owner-aware 4-KiB proof; padding the
Qwen ledger is not a universal-format proof.

Its actual lower-rate coarse distortion is unknown.  The favorable planning
transfer uses `D_coarse=0.035574242296714034`, which would require the fine
stage to capture

```text
1 - 0.025/D_coarse = 0.2972443434920543
```

of coarse residual SSE.  A random orthogonal rank-384 subspace captures only
`384/4096=0.09375` in expectation.  The frozen TACTIC cell therefore needs
about `3.17x` its isotropic dimensional share before finite coefficient loss.
This is a low prior, not a proof of failure.

The proposal's side-cost table is only an optimistic intuition when it adds
`b` bpw on top of a 2.5-bpw baseline.  The deployed cap forbids that.  At fixed
total rate, every side bit displaces a coarse or fine bit, and the resulting
coarse distortion must be remeasured.  The authoritative test is always the
literal final `D * 2^(2R)`.

## What the coarse packet can and cannot do

Let `C` be the transmitted coarse codeword, let `S=(shape,role)` be public
semantic metadata, let `theta` denote any charged shared model, and let
`G=Phi(C,S,theta)` be a graph, seriation, transform bank, syndrome matrix, or
trellis selector.  Because `G` is deterministic given those decoder inputs,

```text
H(G | C,S,theta) = 0
I(X ; G | C,S,theta) = 0.
```

This has four consequences.

1. **The graph costs no separate descriptor bytes, but contains no new
   residual information.**  It can be useful only when the conditional law of
   the residual given `C` has geometry that `Phi` exposes compactly.
2. **A hash selector is not a free side channel.**  If `h(C)` partitions the
   coarse codebook into `M` balanced classes and the encoder needs a specified
   class, it gives up roughly `log2(M)` bits of coarse-codeword freedom or pays
   the corresponding coarse-distortion penalty.  A joint search may still
   improve the final codebook, but the selected ordinary coarse word already
   pays for the choice.
3. **An adaptive 384-bit tree has exactly `2^384` leaves for each coarse
   word.**  Its advantage over a fixed frame is compact procedural geometry
   and tractable search, not a larger message alphabet than any other 384-bit
   fine code.
4. **A 384-bit syndrome names at most `2^384` bins.**  Each bin may contain an
   enormous coset, but a deterministic decoder produces only one posterior
   reconstruction per `(C,syndrome)`.  Syndrome coding can realize a good
   conditional code; it does not multiply the available rate.

This is classical successive refinement.  Gaussian sources under squared
error are successively refinable, so coarse-plus-fine structure alone cannot
move below the Gaussian rate-distortion curve.  The required advantage must
come from a non-Gaussian conditional source law, not from the topology of the
description.  See [Equitz and Cover](https://isl.stanford.edu/~cover/papers/transIT/0269equi.pdf).

## Posterior reconstruction is the strongest addition

Let `M` denote the entire decoded physical message: coarse and fine bytes,
headers, and any serialized source-fitted model or emission table.  The
MSE-optimal reconstruction is

```text
mu(M) = E[X | M].
```

For any current reconstruction `Y(M)`, the orthogonality identity gives

```text
E ||X-Y(M)||^2
  = E ||X-mu(M)||^2 + E ||mu(M)-Y(M)||^2.
```

Thus the exact opportunity is the energy of the conditional bias
`mu(M)-Y(M)`.  This is different from predicting the discrete label itself.
A process can offer a modest entropy gain but a useful within-cell centroid
shift, or vice versa.

There are also strict limitations:

- for one fixed checkpoint, an in-sample empirical conditional mean granted
  free is source leakage; a source-fitted table/model is legal only when it is
  serialized and fully charged;
- the posterior model must be fixed from disjoint data or serialized and
  charged;
- whole experts/layers, rather than coordinates from the same tensor, must be
  held out;
- model mismatch can make the estimated centroid worse than the nominal
  lattice point; and
- a quantizer already trained with its correct conditional centroids leaves no
  separate posterior gain.  The posterior stage is then simply part of the
  codebook design.

Cross-fitting establishes discovery/generalization evidence; it never waives
the bytes or routed pages of a model fitted for the emitted artifact.  Any
update derived from source residuals rather than decoded message state is an
adapted value and must likewise be serialized and charged.

The production diagnostic should report one joint improvement in rate-relative
score:

```text
Delta s_joint
  = (R0 - R_model)
    - 0.5 log2(D_model / D0).
```

For a centroid-only fractional MSE reduction `g`, the remaining ideal rate
saving is

```text
Delta R_remaining(g)
  = 0.15288996696291446 + 0.5 log2(1-g).
```

This gives `0.11588968`, `0.07688842`, and `0.03565734` bpw after respective
centroid reductions of 5%, 10%, and 15%.  These are diagnostics only.  A final
claim must directly score one jointly decoded packet rather than add a
rate-only experiment to a separately fitted centroid experiment.

A Gaussian scale mixture is a reasonable small continuous rung because its
hidden scale can model dependencies in coefficient magnitudes and supplies a
Bayesian least-squares reconstruction; this is the mechanism in
[Portilla et al.](https://www.cns.nyu.edu/pub/lcv/portilla03-reprint.pdf).
It is evidence from image statistics, not evidence that SwiGLU weights have
the same law.

A discrete UWFA probability table alone cannot produce `E[X|C]`.  The dual
diagnostic needs an explicit continuous emission/bin-moment head tied to the
same latent-state law, with exact cell boundaries and source-transform
semantics.  Every parameter and arithmetic rule in that head is additional
charged model state; sharing the law does not make the continuous head free.
Entropy probabilities must be causal, but posterior reconstruction need not
be: after the complete expert message has decoded, a forward-backward smoother
may condition centroids on all expert-local labels.  Such smoothing must
charge buffering, scratch, latency, and any repeated memory traffic.

For the present **unifilar** WFA, the state after a known reset and decoded
prefix is already deterministic.  Under a conditionally independent emission,
future labels cannot refine that state or its conditional moment, so ordinary
forward-backward smoothing is exactly redundant.  Future labels become useful
only after adding an explicit suffix/backward feature or a non-unifilar or
persistent continuous latent state; that is a separately frozen and charged
model.  The first posterior rung should consequently be a one-pass,
deterministic-state exact-cell or held-out-residual centroid.

There is a further STRATA-specific binding problem: the modeled sequence is a
concatenation of selected SC decisions across six polar levels, not one analog
emission per source weight.  A rate model can exploit structure in that
decision schedule without supplying a usable conditional moment for a
particular transformed coefficient.  The posterior diagnostic must therefore
replay all six levels, bind each coefficient to its exact collection of
decision-state features, scales, profile, role, and transform metadata, define
the exact conditional source cell (or use a cross-fitted empirical residual
target), and score after the exact inverse source transform.  A generic scalar
truncated-normal bin is not the block-coupled polar source cell.  The mapping
and any tied feature/centroid table are part of the charged model.  If this
binding cannot be specified compactly, the WFA may remain useful for rate while
being unsuitable as the posterior law; the two gains must not be assumed to
coincide.

## Component-by-component assessment

| Proposed component | Assessment under this repository's evidence | Action |
|---|---|---|
| Repaired CTW/WFA/MPS census | Directly tests the leading open source-rate hypothesis. CTW is a strong bounded-memory universal baseline; the source-free parity WFA proves only mechanism sensitivity, not Qwen structure. | Continue the audited UWFA-SC v3 path; add no new model family until it closes. |
| Posterior-centroid metric | Mathematically exact and not contained by a label-NLL result. Existing RAVEL/affine kills make small local corrections unlikely, but do not contain a non-local hidden-state posterior. | Promote immediately after the source ABI is frozen; use cross-fit and matched controls. |
| Real N18-307 coarse stream | Mandatory for any TACTIC claim; the 2.5-bpw artifact cannot be truncated or rate-transferred. | Audit producer, source-free smoke, then three-stream impossible pilot before the full panel. |
| Frozen TACTIC-DH384 | Exact, local, and low-read, but needs 29.7% planning capture from a 9.375%-rank frame. | Execute unchanged only after its own pilot survival; a miss does not kill CAGE's nonlinear graph, posterior, or adaptive-tree branches. |
| Coarse-derived graph/Krylov basis | Decoder-legal and descriptor-free. The auxiliary-Qwen free-predecessor result (`s=0.01446386` before side cost) is adverse evidence only for unary linear `3x3` prediction. It does not bound multineighbour, nonlinear, multiscale, or graph-spectral prediction. | Run a declared-family dominant actual-coarse oracle before building lifting. |
| Exact graph lifting | Perfect reconstruction is useful for implementation, but a bijection preserves joint entropy and a nonorthogonal inverse can amplify raw MSE. | Conditional on graph-oracle survival; use dyadic fixed-point and inverse-domain scoring. |
| Adaptive per-block 384-step refinement tree | A legitimate fixed-length tree-structured conditional VQ, not free exponential capacity. Branch probabilities do not reduce its physical 48-byte field. An entropy-coded/variable-depth tree is a different format with new reservoirs and framing. | Small-block/beam oracle after its branch-specific graph gate; do not infer failure from a DH384 miss. Redesign the rate split before claiming entropy savings. |
| `h(C)` transform selection | Legitimate joint codebook partition/index modulation. A random CRC has no expected source alignment, and forcing a class spends coarse freedom. | Test only inside a joint coarse-list/fine-path search with no transferred selector gain. |
| Syndrome/coset fine stage | A backend for a measured conditional law. The repository already killed one aligned-role syndrome cell whose favorable information was only `0.00370826 bpw`; coarse-residual side information is distinct but unproven. | Defer until the conditional entropy oracle passes. |
| Bits-back / Bit-Swap | Can approach the marginal latent-model codelength rather than paying every latent literally. It does not erase model bytes, posterior mismatch, seed reservoir, termination, or page cost. | Defer until explicit latent-state cost is the measured bottleneck. |
| Polyphase/Hankel/FRI | Not wholly unexplored. CYCLO-FRI4-NORMAL already ran on Qwen and its dominant rank-four/free-tail oracle hard-killed at `F=0.93798993`. Coarse-derived seriation is a distinct but lower-priority conditional family. | Do not repeat the frozen FRI4 cell; allow one actual-coarse oracle only if its geometry is demonstrably different. |
| Algebraic parity/type trees | High-order parity remains logically open and is what the source-free WFA fixture detects. Type transmission alone gives no entropy rebate: `H(Q)=H(type)+H(Q|type)`. | Keep in UWFA/SILT ladder; require hundreds of reproducible constraints, not statistical significance alone. |
| Two-dimensional TTN/HMT | May match row/column/role geometry better than a chain, but adds model state and coding complexity. | Only after a simpler causal WFA leaves a measured, non-control residual gap. |
| Internal block matching | Decoder-causal when references are already decoded, but identifiers/affine fields cost bits and current pairwise evidence is poor. | Low-priority source-leaking oracle, not implementation. |
| Joint Gate/Up/Down rounding | A real entropy-constrained coupling problem, though existing role dependence is adverse. | Bounded small-block oracle only. |
| Program-synthesized codec | MDL-correct if the program bytes are charged; search selection and interpreter semantics are substantial audit surfaces. | A future architecture-search tool, not a current codec stage. |
| Orbit/Gray-Wyner | Existing aligned same-layer capture is far below need and shared weight-valued pages hurt routed reads. | Keep paused unless a compact procedural common latent changes the oracle. |

The classical context-tree baseline is justified by
[Willems, Shtarkov and Tjalkens](https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/CTW.pdf).
Entropy-constrained trellis search is also established rather than a new
capacity mechanism; see the original
[ECTCQ paper](https://doi.org/10.1109/18.119697).

The source and backend roles must remain separate.  CTW/WFA, continuous
emissions, GSM/HMT, and a conditional graph residual law test whether usable
non-Gaussian structure exists.  Lifting, trellis search, syndrome/coset
coding, QIM, bits-back, and LDGM can realize or search a measured law, but do
not create the below-Gaussian advantage themselves.

## Refined architecture: one latent law, two inference modes

If the source census survives, the most coherent form of TACTIC-CAGE is not a
collection of separately fitted modules.  It is one latent-state law used in
causal filtering mode for branch probabilities and, after full expert decode,
optionally in smoothing mode for posterior reconstruction:

```text
coarse message C
    -> deterministic program Phi(C,shape,role)
    -> causal filtered state z_j from C and previous fine decisions
    -> p_theta(b_j | z_j,C) for exact entropy coding
    -> one fine message B
    -> optional full expert-local forward/backward state posterior
    -> posterior centroid/direction from the same latent law
    -> one reconstruction mu_theta(C,B).
```

The encoder minimizes

```text
||X - mu_theta(C,B)||^2
  + lambda * [ell(C) + ell_theta(B|C) + B(theta) + framing],
```

subject to a literal byte cap and the routed page union.  The decoder reads
one coarse/fine expert frame and a small shared model packet.  A learned model,
transform table, graph kernel, posterior table, padding, and termination state
are all physical.  Large learned constants may not be relabeled as a free
universal specification.

At the frozen 2.5-bpw TACTIC split, `B` is exactly 48 bytes per block and its
branch probabilities earn no rate reduction.  The objective above describes a
new entropy-coded or variable-depth CAGE format only after its coarse/fine/model
reservoirs, termination, and page alignment have been frozen.  The fixed-length
tree can still improve codebook geometry, but not physical rate.

This dual-use law avoids double-counting.  If it predicts labels but not
within-cell residuals, it earns only rate.  If it shifts centroids but does not
improve code probability, it earns only MSE.  The final packet measures both
together.  A smoother is not charged extra compressed bits when it uses only
decoded expert-local labels, but its continuous head, buffering, scratch,
latency, and memory traffic are not free.

## Decisive experimental sequence

### Gate 0: close current source-free work

Finish the ongoing UWFA-SC repair and hostile source review.  UWFA-SC v4
repairs v3's fold, resource, GPU-identity, structural-control, posterior-triplet
and symlink-parent defects, and two source-free RTX 5090 replays are exactly
reproducible.  Its independent audit is nevertheless a source-free **BLOCK**:
same-UID staging mutation/extra files can be published without a final rehash,
controls are not bound to the source artifact and authenticated decoder chain,
and the typed preflight accepts duplicate 150-cell and sparse representative
receipts.  The independent SILT-v1 audit is also a source-free **BLOCK**:
ordinary selected-expert decode touches
all expert frames (`3x` in its counterexample despite a claimed `5/3x`), a
post-rename crash can leave a visible empty final directory, and the declared
total-symbol cap cannot hold one Qwen-shaped 128-expert triplet container.
SILT requires a new repaired source boundary before any payload work.  Neither
mechanism result is Qwen evidence.

UWFA-SC v5 closes those three v4 defects.  V6 additionally makes a separate
parent marker, retained directory inode, and canonical directory root the
publication authority; its full source suite, independent directory-fault
tests, and source-free RTX 5090 replay pass.  The independent v6 audit still
blocks freeze for five narrower downstream reasons: verified result bytes are
closed before consumption, marker content is not rehashed after linking,
inner candidate/held-out selection omits literal 64-byte/4-KiB layout costs,
telemetry is not semantically joined to the claimed workload/canonical device
IDs, and Student-t uncertainty treats overlapping owner folds as iid.  V7 is a
narrow repair of those issues, not a new source model.

### Gate 1: joint entropy and posterior census on the current artifact

After UWFA-SC v3 is frozen, replay its authenticated causal decisions and add
a separately frozen continuous diagnostic.  For each candidate state model,
report:

- exact arithmetic-coded bytes and all model/framing pages;
- exact modeled-symbol count and its ratio to shape-derived source weights;
- a canonical six-level SC-decision-to-coefficient/state binding and exact
  inverse-transform reconstruction receipt;
- current reconstruction MSE;
- cross-fitted posterior-centroid MSE;
- `Delta s_joint` from the same state and split;
- whole-expert and whole-layer uncertainty;
- identical moment-matched Gaussian and structure-destroying searches; and
- model bytes and routed-page cost.

For the discovery diagnostic, the posterior estimator must be trained on
auxiliary whole experts/layers and evaluated on untouched ones.  A separately
declared source-fitted alternative is legal only when its complete table/model
is serialized and charged.  A Qwen pass is absolute; controls interpret source
specificity and may veto promotion, but are never subtracted to create a pass.

### Gate 2: N18-307 closure and efficient pilot

The current working-tree package `tactic_actual_coarse_n18_v1` is not yet an
auditable source closure: its README invokes `verify_source.py` and
`test_source_only.py`, while those files and `SOURCE_MANIFEST.json` are absent.
The subsequent v2 scaffold is also independently blocked: it authenticates
source bytes but later imports mutable live paths, accepts a fabricated runtime
lock, can expose a corrupt public `COMPLETE` tree after a post-rename failure,
and its fixed full-tail reservoirs violate rate/read caps on legal unequal
shapes.  The frozen DH384 v2 cell is now independently blocked as well: its
continuous rank-384 projection is sound, but its finite QC/trellis, rational
scale, and literal fine packet are undefined; its live-path source imports,
coarse reconstruction/symbol binding, publication, runtime/review authority,
tail support, and measured read trace also fail closure.  Repair both findings
in new source boundaries before any payload work.  Then run the source-free
CuPy smoke and encode only block zero of Gate, Up, and transposed Down in the
first triplet.

The attempted N18 v3 repair is also independently blocked before payload.  It
correctly repairs immutable source execution and verify-before-rename
publication, but `1,228 bytes per 4,096 weights` is only a byte-allocation
identity: it supplies no versioned header/reservoir, arithmetic capacity, hard
EOF, scale, tail language, canonical decode/re-encode, or binding to the frozen
N18 packet.  Its four metadata bytes per block merely reproduce the Qwen total
of the old global-plus-expert bytes without defining where the global selector
and QC packet live.  The zero-byte tiny fallback is outside the target cell,
and runtime-import and repeated-read closure remain blocked.

Freeze separate pilot upper bounds before opening those streams:

- for DH384, retain its already declared impossible best-of-64 selector-table
  advantage on every subblock; and
- for CAGE, declare branch-specific coarse-derived graph, posterior, and
  adaptive-tree oracles.  A DH384 upper bound does not dominate those broader
  nonlinear mechanisms.

Stop before the remaining 105 source streams only when every declared branch
fails its own optimistic threshold.  For DH384, this is when every role's
capture plus its frozen uncertainty margin is below both its measured
`1-0.025/D_coarse` and the planning `0.2972443434920543` threshold.  Build the
full source panel when any branch survives.  Pilot survival is permission to
continue, not evidence of target performance.

### Gate 3: execute frozen TACTIC unchanged

After its own pilot survives, build and independently decode all 108 source
reservoirs.  Run the frozen rank-384 continuous span first.  If the absolute
source oracle fails, stop with controls unopened as required by the frozen
protocol.  Only a source survivor permits independently generated Gaussian
and structure-destroyed panels, followed by literal fine search.  Do not
replace the selector, rank, or table after seeing the payload.

### Gate 4: coarse-programmed graph oracle

Only after a real coarse residual exists:

1. for the frozen TACTIC packet, construct one block-local graph solely from
   the same 4,096-weight decoded `C`, shapes, roles, and fixed procedural
   constants;
2. freeze canonical graph ties, eigenvalue ordering, eigenvector signs, and
   degenerate-subspace handling before source access;
3. grant exact spectral/Krylov components for free as a dominant oracle only
   for the declared linear graph/Krylov family;
4. allocate the real per-block 384-bit fine budget through a decoder-known
   codeword.  If residual-derived component choices or reverse-waterfill
   allocations are granted without spending those bits, label that result an
   impossible/source-leaking upper bound;
5. score after exact inverse transformation in source coordinates;
6. repeat the complete construction on matched controls; and
7. report compute, scratch, full-expert scan requirements, unique page union,
   and repeated external/compressed/HBM traffic.

The first gate keeps every correction inside its source block and preserves
exactly 384 fine bits per block.  On Qwen geometry a 4,096-value role block is
only two 2,048-wide neuron rows, so a negative closes only that block-local
family; it does not dominate the proposal's 768-neuron or cross-role graph.
An expert-wide graph or transform that mixes blocks is a separate format: it
receives the explicit expert total of 442,368 fine bits and must freeze a new
allocation, framing, random-access, and read ledger before payload access.

An expert-wide implementation should place the coarse records before fine
records and buffer the decoded coarse/state needed by graph construction.  A
second fetch of the frozen 359-page expert frame would read
`6 + 2*359 = 724` pages against a 360-page physical share, or about `2.011x`,
and fails the strict bandwidth gate even though the unique page union is
unchanged.  Report both unique 4-KiB pages and actual repeated external bytes;
a buffered pass may avoid external refetch but still incurs scratch, latency,
and HBM traffic.  Another expert's frame remains outside this expert-local
design.

### Gate 5: bounded adaptive tree

For source-free fixtures and then small actual-coarse blocks, compare at equal
message count:

- fixed rank-384 frame;
- nonadaptive graph-wavelet directions;
- directions conditioned on previous fine bits; and
- joint coarse-list/fine-path search.

Use exact enumeration where possible, otherwise predeclared beam widths and a
monotone upper bound.  A beam improvement is an algorithm result, not proof of
the full per-block `2^384` optimum.  The last row also needs a frozen generator
of multiple legal coarse candidates, each independently re-encodable within
the same fixed reservoir.  Searching fine paths under one already-fixed coarse
artifact does not test CAGE's central joint-choice mechanism; no separate list
index may be transmitted or hidden.

### Gate 6: finite posterior-coupled packet

Only a survivor receives the shared predictive-state model and posterior
centroid.  Emit the exact integer probabilities, model bytes, fine stream,
centroid arithmetic, and independent reconstruction.  Evaluate one literal
`F`, not a product or sum of earlier gains.

### Gate 7: syndrome and bits-back backends

Use a syndrome only when its measured conditional prior beats explicit branch
coding after finite loss.  Use BB-ANS/Bit-Swap only when explicit latent-state
bytes, rather than absent source structure, are the demonstrated blocker.
Primary references are
[Townsend, Bird and Barber](https://arxiv.org/abs/1901.04866) and
[Kingma, Abbeel and Ho](https://proceedings.mlr.press/v97/kingma19a.html).
Nested/coset constructions are genuine implementations of side-information
coding, not an extra information source; see
[Zamir, Shamai and Erez](https://web.mit.edu/6.454/www/www_fall_2004/latticedecoding/zamirErezShamai02.pdf).

### Gate 8: one sealed composite

Success requires all of the following in the same artifact:

```text
2.15 <= R_physical <= 2.5
D_original_BF16 * 2^(2 R_physical) <= 0.8
maximum owner-aware cold 4-KiB page union < 2x
independent causal decode and canonical re-encode
all model/graph/table/header/padding/tail bytes charged
whole-expert/layer Qwen records and matched controls
portability evidence on a disjoint SwiGLU-MoE family or sealed legal fixture.
```

## Prior-art and novelty boundary

Adaptive lifting without separate bookkeeping is established in signal
processing; see
[Heijmans and Piella](https://ir.cwi.nl/pub/4357).  Selecting among quantizers
through an index/coset is related to
[quantization-index modulation](https://sia.mit.edu/wp-content/uploads/2015/04/2001-chen-wornell-it.pdf).
Tree-structured refinement, entropy-constrained trellises, posterior means,
Gaussian scale mixtures, syndrome binning, and bits-back are likewise known.
Direct LLM precedents further narrow the claim:
[R2Q](https://arxiv.org/abs/2511.21736) uses sequential residual refinement,
[MG-PTQ](https://arxiv.org/abs/2501.18154) uses a graph network to model weight
dependencies and assign precision, and
[SAGE-PTQ](https://arxiv.org/abs/2606.05429) uses a sparse graph to choose
low-overhead scale groups.  None of those abstracts describes a coarse
codeword programming its own graph/refinement packet.

The defensible research novelty is narrower:

> an expert-local, universal SwiGLU PTQ packet in which the charged coarse
> codeword deterministically programs a conditional graph/refinement code,
> while one latent-state law supplies causal branch probabilities and an
> optional expert-local smoothed posterior centroid under a physical bpw and
> owner-aware cold-page ledger.

That is a novelty assessment, not a patentability opinion and not evidence of
a compression gain.

## Current status

- UWFA-SC v3 is independently blocked before freeze.  Its CPU/CuPy mechanics
  and a 250-expert unequal-shape fixture passed, but its scientific folds are
  undefined on legal single-layer panels; source/preflight/score bindings,
  symlink-safe publication, resource caps, UUID/PCI identity, control geometry,
  and one posterior-handoff digest require repair.  It has no Qwen result.
- UWFA-SC v4 repairs those v3 defects and passes its source-free 150-cell and
  representative CuPy workloads twice on the RTX 5090 with deterministic
  equality, but its independent audit remains a **BLOCK**: publication does
  not re-enumerate/rehash staging immediately before completion, controls can
  carry a foreign source/decoder closure, and preflight validation accepts
  duplicate selector rows and underspecified representative evidence.  No
  Qwen/current-codec payload was opened, so this is not a source-model result.
- UWFA-SC v6 passes the repaired publication substitution cases, exact
  source/control closure, canonical 150-cell preflight, and a fresh source-free
  RTX 5090 replay, but its independent audit is **BLOCK_SOURCE_FREEZE** on
  verified-byte lifetime, post-link marker content, literal inner aligned-rate
  selection, workload-bound telemetry/canonical IDs, and dependency-invalid
  owner-fold confidence.  V7 is repairing those five gates.  No Qwen or
  current-codec object has been opened by v4-v7.
- SILT v1 is independently blocked on routed-read authentication, durable
  publication, and universal-capacity issues.  It has no Qwen payload or
  source-gain result.
- The `0.1675415 bits/symbol` WFA result remains a synthetic parity fixture,
  not a Qwen measurement.  `bits/symbol` is not automatically `bits/weight`;
  the production STRATA adapter concatenates selected SC decisions across six
  levels, so its modeled-symbol/weight ratio depends on the block and profile.
  Only the production container's literal bytes divided by its shape-derived
  weight count can establish the required physical-bpw saving.
- TACTIC-DH384 remains unexecuted because no authenticated physical
  `307/128`-bpw coarse artifact exists.  Its independent v2 source audit is now
  a **BLOCK**: the continuous orthogonal rank-384 upper bound is sound, but the
  claimed finite 384-bit QC/trellis map and output scale are not implemented or
  frozen, coarse decoded records are not bound to reservoirs, faulted output
  can remain public, the runtime/review boundary is unauthenticated, and the
  executable accepts only the fixed Qwen geometry.  The audit opened no Qwen or
  CUDA payload.
- The present N18-307 producer directory is also incomplete as a claimed
  source freeze: its manifest, verifier, and hostile source tests are absent,
  so neither its synthetic smoke nor its payload entry points are authorized.
- N18-307 v2 closes many packet/shape/owner-ledger mechanics but is
  independently blocked on executed-source identity, runtime authentication,
  post-rename publication, universal tail rate/read caps, and review authority.
- N18-307 v3 repairs source-byte execution and the publication state machine,
  but is independently **BLOCKED** because its microblock partition is not an
  implemented coarse code, its zero-slack metadata topology is undefined, its
  tiny fallback has `R=0` and `F=1` for nonzero `1x1` weights, runtime hashes do
  not bind later imports, repeated reads are not bounded, and its mandatory
  POSIX suite exits nonzero on exception normalization.  No Qwen, numeric
  dependency, or CUDA payload was opened.
- TACTIC-CAGE is therefore a promoted research path with strict gates, not a
  result or checkpoint-achieving codec.
