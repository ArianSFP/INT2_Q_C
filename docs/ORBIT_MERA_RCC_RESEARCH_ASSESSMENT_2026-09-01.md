# ORBIT–MERA–RCC research assessment

Date: 2026-09-01
Scope: universal, weight-only, post-training compression of arbitrary SwiGLU
MoE expert triplets under the repository's raw-weight-MSE and cold-read rules.

## Decision

The proposal contains one high-value unclosed hypothesis:

> A causal, low-description-length long-range model may predict legal
> quantizer symbols better on held-out SwiGLU-MoE weights than on the complete
> matched-Gaussian control pipeline.

That hypothesis warrants an immediate source-model census.  It does **not**
yet justify building reverse-channel coding, a Born machine, a learned flow, a
Gray–Wyner container, a fibre enumerator, or an LDGM backend.

The corrected promotion order is:

```text
operational entropy census
    -> small tied nonnegative MPS / weighted automaton survivor
    -> optional dyadic multiscale lifting fitted to held-out codelength
    -> exact small-block entropy-constrained codeword search
    -> finite arithmetic/range-coded packet
    -> RCC only if a separately measured small-KL channel needs it
```

Orbit-aligned common/private coding remains a parallel, lower-priority oracle.
FIBER and spatially coupled LDGM are search/backend ideas, not plausible
sources of the required below-Gaussian advantage.

## Exact information requirement

At `R=2.5`, the repository's independently decoded finite baseline is

```text
D = 0.030902167403153148
F = D * 2^(2R) = 0.9888693569009007
s = -0.5*log2(F) = 0.008074080480766676 bpw
```

The target `F <= 0.8` is `s >= 0.16096404744368115 bpw`.  A standalone
lossless recoding of the existing reconstruction therefore needs the net
physical saving

```text
0.16096404744368115 - 0.008074080480766676
  = 0.15288996696291447 bpw.
```

This is a particularly clean route: if the same reconstruction can be stored
at approximately `2.347110033 bpw`, its distortion is unchanged and it meets
the rate-relative target.  The saving must be measured after model tables,
headers, termination, alignment, padding and cold pages.

The checked-in encoder transcripts also show that ordinary arithmetic-coder
inefficiency is not the opportunity.  Across all fifteen streams:

```text
logical arithmetic bits       = 70,677,314
ideal NLL under current prior  = 70,676,086.9641701
finite-coder excess            = 1,227.0358299 bits
finite-coder excess / weight   = 0.0000433404650 bpw
```

The new model must therefore change the predictive law by millions of bits;
replacing the range coder alone cannot matter.

The `0.11356063457 bpw` number has a narrower meaning.  It is the remaining
gap after the best *ideal*, non-finite composite.  It is relevant only if the
new source model is rerun on that exact nested residual and one literal
composite reconstruction is emitted.  It cannot be added to an independently
fitted entropy result.

## What is mathematically sound

For a whitened continuous random vector `X` and an orthogonal `Y=HX`,
differential entropy is invariant.  With standard-normal scalar reference
entropy `h_G`, scalar negentropy `J(Y_i)=h_G-h(Y_i)`, and total correlation
`TC(Y)=sum_i h(Y_i)-h(Y)`, the identity

```text
n*h_G - h(Y) = TC(Y) + sum_i J(Y_i)
```

is exact.  Consequently an orthogonal transform can make marginals look more
Gaussian while transferring, rather than erasing, joint non-Gaussianity.

At high resolution, the Shannon-lower-bound heuristic for a stationary source
gives

```text
D(R) approximately 2^(-2R) * 2^(-2*(h_G-h_rate)).
```

This makes an entropy-rate deficit a sensible diagnostic for the desired
rate-relative gain.

The limitation is important: the current experiment contains finite,
deterministic weight arrays, and quantization is not invertible.  Continuous
entropy invariance does not prove that a low-complexity model can compress the
discrete legal symbol stream.  For a bijection applied to an already discrete
vector, joint Shannon entropy is itself exactly invariant.  If the transform
precedes requantization, `H(Q(HX))` can change and no invariance identity
determines the result.  The usual `H(Q_delta(X))` to differential-entropy
bridge is an asymptotic `delta -> 0` statement and cannot certify behavior at
2.15–2.5 bpw.  An arbitrary dependency in one huge vector can require a
description as large as the vector itself.  Only held-out, fully-charged
operational codelength establishes usable structure.

MPS Born machines are established tractable joint-distribution models with
direct sampling, not a speculative mathematical object.  See
[Han et al., *Unsupervised Generative Modeling Using Matrix Product States*](https://arxiv.org/abs/1709.01662).
The formal links among uniform MPS, stochastic processes and weighted
automata are developed by
[Srinivasan et al.](https://arxiv.org/abs/2010.10653), and nonnegative-MPS,
HMM and Born-style representation separations are studied by
[Glasser et al.](https://arxiv.org/abs/1907.03741).  Bounded-state HMM output
processes can retain genuinely long observable memory; see the prediction and
compression analysis by
[Han, Jiang and Wu](https://proceedings.mlr.press/v247/han24a.html).
Recent work also reports a training "causality trap" in which gradient-trained
MPS models miss strong non-local interactions, so a negative Born-machine run
without robust initialization would not close the hypothesis; see
[Tang, Khoo and Ying](https://arxiv.org/abs/2505.06419).

## Why the first model should not be a Born MPS

Start with a nonnegative, tied weighted finite automaton, equivalently a
restricted nonnegative MPS/HMM-like model.  For a fixed horizon `n`, write

```text
S = sum_a A[a]
p(q_1,...,q_n)
  = alpha^T A[q_1] ... A[q_n] omega / (alpha^T S^n omega).
```

With right environments `r_t=S^(n-t) omega`, its exact causal conditional is

```text
p(q_t=a | q_1,...,q_(t-1))
  = f_(t-1)^T A[a] r_t / (f_(t-1)^T r_(t-1)).
```

It therefore has nonnegative probabilities, stable forward recursion, causal
conditionals, and an immediate arithmetic-coding interpretation.  The right
environment and global normalizer may not be omitted.  A conventional
row-stochastic HMM with `omega=1` is a useful simpler subclass, but it cannot
represent terminal constraints such as the fixed-block even-parity unit test
without an explicit decoder-visible terminal context.  Only if the
nonnegative model saturates while a controlled noncausal oracle remains
favorable should a Born-amplitude model be attempted.

For a four-symbol alphabet, bond/state size `chi`, fitted boundary vectors and
uint16/FP16 parameters:

```text
tied raw bytes = 2 * (4*chi^2 + 2*chi), before framing.
```

At `chi=64`, that is 33,024 bytes.  Conservatively amortized over the
six-expert Qwen panel it costs about `0.00933160 bpw`; an isolated 4 KiB-page
packet occupies 36,864 bytes and adds `0.025x` cold traffic relative to one
equal 2.5-bpw expert share.  This is feasible.  Procedurally frozen boundary
vectors can remove their bytes, but fitted boundaries are not free.
Training may use FP32/FP64, but an operational packet must name one exact
integer probability representation; "FP16/uint16" is not a portable coding
specification.

Those figures are for **one** tensor shared across roles.  Three independent
role tensors cost 99,072 raw bytes, `0.02799479 bpw` over the six-expert panel,
and occupy 102,400 contiguous page-rounded bytes, adding about `0.0694444x`
cold traffic.  At `chi=128`, three tensors cost 394,752 raw bytes and
`0.11154514 bpw` over that panel.  The existing 2.5-bpw expert has 1,224,704
cold bytes of headroom before the strict `2x` limit; three `chi=256` tensors
occupy 1,575,936 raw bytes before page rounding and already fail it.

Consequently the minimum *gross* held-out gain depends on the selected model.
One global `chi=64` FP16/uint16 tensor costs `0.00933160 bpw` over the literal
six-expert panel, so its gross predictive gain must exceed
`0.16222157 bpw` before framing.  Three role-specific `chi=64` tensors cost
`0.02799479 bpw`, raising that floor to `0.18088476 bpw`.  This is why a
roughly `0.18 bpw` promotion margin is sensible only when it is net, or when
the exact model ledger proves the gross-to-net conversion.

With isolated page charging, the corresponding gross floors are about
`0.16330663 bpw` for one model and `0.18182515 bpw` for three packed models,
before headers and arithmetic termination.

Dense causal contraction is not automatically compute-feasible: evaluating
all symbols of an unrestricted alphabet costs `O(K*chi^2)` per decoded symbol
(`O(chi^2+K*chi)` for a factorized HMM).  At `chi=64`, millions of weights
imply tens of billions of multiply-adds per expert.  A survivor needs sparse,
low-rank, blockwise or cached contractions, or an explicit one-time
materialization policy; read efficiency alone is not enough for MoE inference.

There is also a useful capacity warning.  A nonnegative bond-`chi` model
factors dependence across a sequence cut through a `chi`-valued latent, so
`I(left;right) <= log2(chi)`; a Born-amplitude construction has a roughly
`2*log2(chi)` ceiling.  At `chi=64` these are six and roughly twelve bits.
The standalone target is about `313.119` saved bits per 2,048 weights.  This
does not require `chi >= 2^313`, because a small recurrent state can be reused
many times, but it rules out explaining the gain as hundreds of independent
constraints crossing one cut.  A single parity saves only `1/2048 =
0.00048828125 bpw` and is an implementation test, not evidence for Qwen.

### Source-free nonlocal mechanism check

The isolated source-free prototype in
`research/nonlocal_wfa_global_state_synthetic_v0` now verifies that the
implementation can detect structure genuinely invisible to bounded suffixes.
Each 32-bit fixture block contains 26 iid bits followed by six disjoint parity
checks.  Every suffix through depth 25 has population rate exactly one
bit/symbol, while a sparse `chi=64` unifilar, symbol-conditioned WFA emitted
and independently decoded a real arithmetic stream at:

```text
structured logical rate             0.8168277956 bit/symbol
independently refit iid control      1.0000672979 bit/symbol
detected difference                  0.1832395022 bit/symbol
eight-stream model-charged saving    0.1675415039 bit/symbol
sparse model                         3,456 bytes / one 4 KiB page
```

The RunPod replay passed all twelve source tests and independently regenerated
both sources before decoding every arithmetic symbol.  This is a mechanism
and implementation pass only: it contains no model weights and is not Qwen
evidence.  It also exposes an expressivity distinction for the real ladder.
A factorized shared-transition HMM is the cheapest persistent-regime test,
but a sparse symbol-conditioned transition WFA is required for XOR-like
constraints and should be retained as a separate rung.

### First production-cell audit

The first dense factorized-HMM producer at
`research/tied_mps_entropy_census_stage0_v0` was **blocked before payload
access**.  Its independent audit is sealed under manifest SHA-256
`020c754f8eaf9bddb8a1ac10ee88ed44d477f21c3ea061befcd439b5712d061d`.
The important defects were not cosmetic:

- full-panel hyperparameter selection contaminated the claimed holdout;
- per-fold model selection made fold identity an implicit selector;
- decoded probabilities depended on unspecified CuPy float64 reductions;
- the kill bound scored continuous probabilities rather than the Q0.16
  physical coding law;
- Gaussian controls did not replay every original baseline;
- survival status could be set by the in-sample packet despite holdout failure;
- stated and implemented HMM timing differed, resets leaked prefix state, and
  leaf-symlink/bootstrap checks were not fail-closed; and
- the dense 45-cell, 12-EM-pass runtime was not a credible stage-0 gate.

That implementation has no launch authority and produced no Qwen number.  It
does not count as evidence against the source-model hypothesis.  Its successor
uses an `O(1)`-per-symbol exact-integer unifilar WFA bank and keeps physical
SC-stream recoding separate from the raw/transformed label-copula census.

Two corrected source-only packages then passed their producer tests but were
also independently blocked before payload access:

1. `research/unifilar_wfa_entropy_census_stage0_v1` passed exhaustive
   arithmetic tests, all 150 CPU/CuPy cells and linear-runtime scaling.  Its
   independent audit manifest is
   `df7b78f97c798a9ab0893ca17d4661874278cd21b89ba6418dcfee681fc64366`.
   Launch remains blocked because authenticated source could be replaced
   between hash and import, symlink ancestors were unpinned, declared current
   bytes were not bound to artifact size, final acceptance did not decode the
   serialized model, no literal complete container was emitted, and Gaussian
   control geometry was not enforced.
2. `research/label_copula_census_stage0_v0` passed orientation, Lloyd-label,
   all 240 transition-law, Q0.16, arithmetic and ledger checks.  Its audit
   manifest is
   `b8384c70534dbc062425648e1794ed20c2afaf01fdf5b07b43f6e7735640f803`.
   It remains blocked because completion was not write-final, symlinked
   packages were accepted, a one-layer degenerate bootstrap could be called a
   lower confidence bound, control provenance was insufficiently bound, and
   the reusable expert-slot universe was not validated.

These are evidence-integrity failures, not negative Qwen measurements.  No
model weights, current-codec payload or Gaussian controls were opened by any
of the blocked cells.  They are retained as audited design evidence rather
than silently repaired and launched.

A length-`L` site-specific MPS has approximately `L*K*chi^2` parameters.  A
length-2,048, four-symbol, FP16, `chi=64` unit cell is 64 MiB before headers;
even when reused across rows, it costs about `18.96 bpw` over the six-expert
panel and destroys routed locality.  Fully site-specific expert tensors are
worse: their asymptotic cost is `p*K*chi^2 bpw`, already 64 bpw for
`p=16,K=4,chi=1`.  Site-specific tensors are therefore forbidden in the first
census.  Eligible variants are tied, small-period, or small
coordinate-role-conditioned tables whose literal bytes are charged.

## Decisive entropy census

### Streams

Retain and score both views from the existing finite encoder:

1. the canonical legal reconstruction-label stream in source-coordinate
   order; and
2. the exact causal SC/branch-decision stream before its current arithmetic
   coder.

The first tests the proposed high-order copula mechanism.  The second tests a
direct physical recoding opportunity.  A result on packed or already
arithmetic-coded bytes alone is insufficient; the existing zstd-19 wrapper
saved only `0.0020726522 bpw`, but that does not contain a better symbol-law
test.

Prior categorical evidence is adverse but not dispositive.  The independent
bitplane/context screen found only `0.0000895960 bpw` after table charge, and
its deliberately leaky cross-role opportunity was `0.0037082566 bpw`.
Those tests close short causal neighborhoods, not a hidden-state model capable
of parity or other nonlocal constraints.  A deterministic suffix model alone
would merely repeat that family; the census must include genuinely latent
tied transitions or state explicitly that its kill is local-only.

Score raw/no-RHT, deterministic-RHT and current STRATA/POLARIS orderings.  A
dense RHT can turn simple source-coordinate dependence into a process with
large sequential bond complexity, so stream ordering is part of the model and
must be frozen without looking at the held-out fold.

### Models

Use the nested ladder:

```text
current causal probability model
factorized role/stratum marginals
finite-order Markov models
tied nonnegative WFA/MPS, chi in {4,8,16,32,64}
small-period WFA/MPS only after tied-model survival
```

All transition tensors must be quantized into an exactly decoded integer or
fixed-point representation.  Report ideal cross-entropy for diagnosis and a
real range/arithmetic codelength for promotion.

### Splits and controls

- Freeze the universal fitting algorithm, search space, seeds and thresholds
  before payload access.
- Fit only on auxiliary experts/layers; score untouched whole expert triplets
  and whole layers.  Leave both the held-out layer and expert out.
- Do not condition on model, checkpoint, layer or expert identity.  Role,
  shape, public coordinate phase, earlier decoded symbols and transmitted
  tables are eligible.
- Repeat the complete fit and selection on independently generated
  moment-matched Gaussian controls passed through the exact same transform and
  quantizer.
- Add within-role/stratum symbol permutations and multiscale block shuffles to
  identify the dependency length responsible for any gain.
- Charge every selected model byte over the literal scored artifact and in
  every routed-expert cold-page union.

Report at least:

```text
G_operational = (current physical symbol bits - candidate physical bits) / N
G_specific    = G_operational(Qwen) - G_operational(matched Gaussian)
```

The exact emitted Qwen codec ultimately decides the target:
`G_operational(Qwen)` itself must exceed the physical threshold.
`G_specific` is the scientific guard against mistaking ordinary finite-code
improvement or selection bias for below-Gaussian source structure; subtracting
a control may never turn a physically failing packet into a pass.

### Frozen decisions

For the standalone lossless-recoding claim, hard-kill goal attainment whenever
the net upper confidence bound is below `0.15288996696 bpw`.  The following
bands control whether the same statistical signal deserves work as a more
general lossy prior:

- `< 0.03 bpw` net held-out source-specific gain: hard kill.
- `0.03–0.10 bpw`: scientifically interesting, insufficient for the goal.
- `> 0.11356063457 bpw`: eligible for an exact nested composite only.
- `>= 0.15288996696 bpw`: standalone lossless route can meet the current
  finite target.
- `>= 0.18 bpw`: enough margin to consider expensive stochastic/RCC
  engineering.

Confidence intervals must be by whole expert/layer, not by treating millions
of correlated symbols as independent samples.

## MERA-inspired lifting

The useful part is the objective, not the quantum terminology: fit a small,
exactly invertible multiscale circuit to minimize held-out *operational symbol
codelength plus source-domain distortion*.  Do not optimize kurtosis alone.

Use 2x2–8x8 dyadic rotations or integer lifting steps, shared across shapes
and roles where possible.  Serialize every selected coefficient.  Evaluate
the inverse using exact source-domain MSE; a non-orthogonal lift induces a
pullback metric and can amplify errors.

This stage is promoted only after the fixed-transform census finds a real
long-range signal.  Otherwise it is a high-dimensional search over a signal
that has not been shown to exist.  MERA itself was developed as a
multiscale-entanglement representation, not as a weight codec; the analogy is
conceptually useful but does not supply a compression theorem.  See
[Evenbly and Vidal](https://arxiv.org/abs/1502.05385).

## Reverse-channel coding

The posterior

```text
Q_beta(q|x) proportional to p(q) * 2^(-beta*d(x,xhat(q)))
```

is a coherent rate-distortion test channel.  A prior-weighted MAP/beam/DMRG
search can use it immediately, without RCC.

For fixed `p`, the Gibbs tilt minimizes `D_KL(Q||p) + beta*E[d]`.  Averaged over
sources, its rate term obeys

```text
E_X D_KL(Q(.|X)||p) = I(X;Q) + D_KL(P_Q||p).
```

It equals a rate-distortion mutual-information objective only when the prior
matches the induced reproduction marginal; prior mismatch is not free.

RCC is only a transmission mechanism for sampling from that channel.  It does
not create a source-specific rate-distortion advantage.  Its theorems also
assume shared randomness and concern expected communication with one-shot
overheads, not this project's actual fixed physical reservoir.  General
relative-entropy coders have expected length related to `KL(Q||P)`, but general
runtime is exponential in that KL; the faster A* guarantees require
restrictive density-ratio structure.  See
[Flamich, Markou and Hernández-Lobato](https://arxiv.org/abs/2201.12857) and
[He, Flamich and Hernández-Lobato](https://arxiv.org/abs/2405.12203).

For a 2,048-weight block near 2.5 bpw, an unpartitioned channel may carry
thousands of bits of KL and is not a practical A* cell.  Any RCC promotion
must predeclare small channel partitions, worst-case rather than expected
capacity, tail/overflow behavior, deterministic shared seeds, decoder work,
no-retry fallback behavior, and exact byte framing.  It must also explain whether INT2 values are
materialized once or regenerated on every cold expert route.

The first finite backend should therefore be ordinary arithmetic/range coding
under the surviving causal model.  RCC is reserved for a measured small-KL
gap that deterministic entropy-constrained search cannot realize.

## Orbit alignment and Gray–Wyner coding

Cycle-consistent multi-graph matching is a legitimate way to synchronize
permutations into a common universe; see the primary multi-graph matching
description by [Dupé et al.](https://proceedings.mlr.press/v189/dupe23a.html).
Lossy Gray–Wyner common/private source coding and Gaussian constructions are
also established theory; see
[Shi, Liu and Ling](https://arxiv.org/abs/1603.05576) and
[Sula and Gastpar](https://arxiv.org/abs/2002.01348).

This does not imply that Qwen experts contain enough common information.
ResMoE already demonstrates the broad common-barycentre/residual pattern
([Ai et al.](https://arxiv.org/abs/2503.06881)), while CAMERA jointly analyzes
Gate/Up/Down micro-experts
([Xu et al.](https://arxiv.org/abs/2508.02322)).  The proposed exact-orbit,
multiway, lossy common stream is still a distinct oracle, but current local
evidence is strongly adverse: the favorable same-layer aligned Up/Down
superoracle captures only `0.01653490`, versus `0.14566208` required by the
ideal-composite screen.

The lossless analogue makes the ceiling transparent.  For any common latent
`U` and expert sources `X_1,...,X_E`,

```text
I(X_1,...,X_E;U) + sum_e H(X_e|U)
  = H(X_1,...,X_E) + TC(X_1,...,X_E|U).
```

Thus a common stream can remove only dependence that actually exists across
experts; any conditional total correlation left after alignment remains in
the private streams.  The lossy Gray–Wyner problem is richer, but orbit
synchronization still cannot manufacture common information.

Permutation synchronization can make noisy pairwise assignments consistent;
it cannot create common variance.  Promote only a multiway oracle with a
literal shared/private reconstruction, full permutation/gauge bytes, and a
cold-read ledger.  A dense common expert is unattractive because every routed
expert must read it.  Only a compact generator or low-dimensional common
latent can plausibly remain near `1x`.

For common bytes `B0`, private bytes `Bi`, and `E` experts, the symmetric cold
amplification is

```text
(B0 + Bi) / (B0/E + Bi).
```

The strict `<2x` condition requires `B0/Bi < E/(E-2)`: less than `1.5` for
the literal six-expert artifact and less than about `1.0159` for 128 experts,
before page rounding.  A hypothetical 128-expert amortization may not be used
to excuse bytes in a six-expert artifact.  The exact permutation descriptor is
also not zero: `log2(768!) = 6259.38` bits, about `0.00132654 bpw` per Qwen
expert before gauge values and framing.

## Quotient-aware nonlinear flows

An invertible flow can expose a copula to a simple latent law.  Its
change-of-variables likelihood is not enough: quantization error in latent
space must be scored through the inverse-flow pullback metric, or by exact
analysis-by-synthesis in raw source MSE.

Auxiliary-variable nonlinear ICA has genuine identifiability results under
specific conditional-independence assumptions; see
[Hyvärinen, Sasaki and Turner](https://arxiv.org/abs/1805.08651).  Those
theorems do not establish compressibility of model weights.  In this project,
model/layer/expert identity is forbidden as free auxiliary decoder state.
Role, shape, coordinate phase and serialized fitted state remain eligible.

Discrete invertible learned compressors exist, including integer-only flows
([Wang et al.](https://arxiv.org/abs/2206.08869)), so the engineering concept
is credible.  It is nevertheless lower priority than the census because flow
parameters, inverse compute and cold reads are expensive, and a generic
inverse flow does not leave weights in a native low-bit matmul layout.

## FIBER / Markov / Graver basis

Markov bases really do provide moves connecting all discrete tables in a
fibre with fixed sufficient statistics; see the algebraic-statistics overview
by [Aoki](https://arxiv.org/abs/1607.07600).  That connectivity can support a
constrained search.

The proposed entropy accounting, however, contains a decisive cancellation.
If `b=f(q)` is a deterministic statistic, then

```text
H(q) = H(b) + H(q|b).
```

Thus

```text
H(q) - H(q|b) - B(b) <= H(b) - H(b) = 0
```

for any honest code with `B(b) >= H(b)`.  Transmitting a type/statistic and an
enumerative rank does not reveal free information.  Fibres may improve lossy
codebook geometry, reduce mismatch of a restricted universal coder, or give
useful invariant-preserving search moves, but they are not by themselves an
entropy source.  Given the weak existing cross-role/local-context evidence,
this branch is a long shot and should not precede the MPS census.

## Spatially coupled LDGM

Spatially coupled LDGM with belief-propagation-guided decimation is a credible
low-complexity route toward an ensemble's lossy optimum; see
[Aref et al.](https://arxiv.org/abs/1202.4959).  It approaches a supplied
source rate-distortion law.  It cannot move that law 20% below the Gaussian
curve without a non-Gaussian prior or common-information source already found
elsewhere.  Retain it as a possible finite search backend after a source-model
survivor, not as an immediate experiment.

## Novelty boundary

Learned weight codecs are already public: Neural Weight Compression learns an
analysis transform, entropy-coded latent and synthesis transform for model
weights ([Ryu et al.](https://arxiv.org/abs/2510.11234)).  Learned context and
hierarchical priors are also standard in image compression.  The defensible
research novelty is narrower:

- a charged tensor-network probability law over legal low-bit PTQ outputs;
- a transform trained to reduce held-out discrete label complexity rather
  than only covariance, outliers or kurtosis;
- exact raw-MSE, matched-Gaussian and routed-read evaluation; and possibly
- RCC as the physical selector for a frozen hardware-constrained PTQ channel,
  if a finite small-KL implementation actually survives.

Even RCC applied to neural-network parameters is not new in the broad sense:
Minimal Random Code Learning transmits samples from variational weight
distributions using KL-governed random codes
([Havasi, Peharz and Hernández-Lobato](https://arxiv.org/abs/1810.00440)), and
COMBINER uses relative-entropy coding for Bayesian implicit-representation
weights ([Guo et al.](https://arxiv.org/abs/2305.19185)).  Any novelty claim
must therefore be the much narrower frozen-PTQ use over legal low-bit
reconstructions, not "RCC for weights."

This is a research-prior-art assessment, not a patentability opinion.

## Final recommendation

Adopt the entropy census, but do not freeze the full ORBIT–MERA–RCC stack.
The immediate architecture under test should be called a causal
tensor-source-model gate, not a codec result.  Its first question is exact:

> Can a tied, fully charged, exactly decoded long-range model save at least
> `0.15288996696 bpw` on held-out Qwen quantizer symbols, after subtracting
> the complete matched-Gaussian pipeline and respecting routed cold reads?

If no, the central premise is dead and the expensive stages stop.  If yes,
the same model should first be turned into a conventional finite arithmetic
codec.  Only a remaining operational gap earns MERA-style lifting, joint
rate-distortion search, or RCC.
