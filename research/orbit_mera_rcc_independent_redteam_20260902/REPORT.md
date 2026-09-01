# Independent red-team: ORBIT–MERA–RCC for universal SwiGLU-MoE PTQ

Date: 2026-09-02  
Scope: source-free mathematical and systems review; no Qwen/current/control
payload was opened, statted, hashed, enumerated, or otherwise accessed. The
active v3 producer directory was not inspected or edited.

## Verdict

The proposal identifies the right remaining scientific question but freezes
far too much architecture before that question is answered.

The one branch worth promoting now is:

> **A causally normalized, low-description-length long-memory probability
> model must beat the current SC probability law on held-out legal decisions,
> after literal model bytes, pages, framing, and cold reads.**

That is a source-model census, not yet an MPS/Born/RCC codec. A sparse unifilar
WFA or factorized HMM is the correct first instrument. Full Born-MPS, MERA,
RCC, Gray–Wyner, nonlinear flow, FIBER, and SC-LDGM are conditional later
branches.

| Branch | Mathematical status | Physical prognosis | Decision now |
|---|---|---|---|
| Causal WFA/HMM on SC decisions | Sound and directly operational | Small model, sequential expert-local stream | **Highest priority** |
| Tied/sparse MPS source law | Sound in principle | Normalization, model bytes and dense contraction are serious | Try only after WFA evidence |
| MERA-style reversible lifting | Useful transform idea; “entanglement” is only a proxy | Parameters, inverse compute, and non-native layout | Conditional small circuit only |
| Orbit/Gray–Wyner | Sound multiterminal theory | Requires large real common information; common pages hurt cold reads | Parallel strict oracle |
| Reverse-channel coding | Sound channel-simulation principle | Expected rate, exponential generic search, tails and shared randomness | Backend only after a small-KL oracle |
| Riemannian flow | Change-of-variables and pullback metric are correct locally | NLL is not RD; model/read/compute cost likely prohibitive | Low priority |
| FIBER/Graver | Connectivity idea is real | No entropy is created; basis/search can be exponential | Kill as standalone source |
| Spatially coupled LDGM | Credible asymptotic backend | Does not create below-Gaussian source structure | Retain only as backend |

## 1. Exact target and physical byte hurdle

The proposal's information arithmetic is correct before finite placement:

```text
N       = 28,311,552 weights
B0      = 8,847,360 bytes
R0      = 2.5 bpw
D0      = 0.030902167403153148
F0      = 0.9888693569009007
R needed at unchanged D = 0.5 log2(0.8/D0)
                        = 2.34711003303709 bpw
```

Thus an ideal unaligned recode needs `0.15288996696 bpw`. Under the required
4-KiB final-page placement, however, the largest passing object is:

```text
unaligned maximum       = 8,306,290.968... bytes
page-aligned maximum    = 8,302,592 bytes
page-aligned R          = 2.34606481481481 bpw
page-aligned F          = 0.798841655309746
required literal saving = 544,768 bytes
                        = 0.153935185185185 bpw
```

This is the relevant standalone gate. It includes the probability model,
transform, permutation/gauge, seeds, common stream, headers, checksums,
directories, arithmetic termination, alignment, and padding. A source-vs-null
entropy difference cannot substitute for this absolute byte result.

## 2. The entropy argument: correct identity, unproved opportunity

For a population of whitened continuous vectors and an orthogonal transform,

```text
n h_G - h(Y) = TC(Y) + sum_i J(Y_i)
```

is exact. Therefore Gaussian-looking RHT marginals do not establish joint
Gaussianity. The proposal is right that an RHT can relocate non-Gaussianity
into nonlocal dependence.

Three qualifications are decisive:

1. The weights are finite deterministic arrays. A block population and a
   transfer protocol must be declared before differential/entropy-rate language
   is meaningful.
2. Quantization is noninvertible. Continuous entropy invariance does not imply
   a low entropy for the legal discrete decision stream.
3. The Shannon-lower-bound expression is a high-resolution optimistic
   diagnostic. At 2.15–2.5 bpw, an entropy-rate deficit is necessary evidence,
   not a guarantee that a finite constrained code attains the corresponding
   distortion.

Most importantly, the identity says nothing about **description complexity**.
The hidden total correlation could require a model as large as the source or a
bond dimension exponential in the chosen 1-D ordering. The decisive quantity
is held-out operational codelength under a serialized model.

## 3. MPS/Born prior: promising hypothesis, incorrect immediate ABI

MPS Born machines are real tractable generative models with direct sampling;
the original work reports a partition function linear in sequence length and
training complexity scaling as `O(|T| N chi^3)` for bounded bond dimension
([Han et al., 2018](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031012)).
Uniform MPS, Born models, HMMs, and weighted automata have rigorous
representation links ([Srinivasan et al., 2020](https://arxiv.org/abs/2010.10653)),
but nonnegative MPS/HMM and Born representations can have large separations
([Glasser et al., 2019](https://arxiv.org/abs/1907.03741)). This supports a
ladder, not an assumption that one small MPS will work.

### 3.1 Hidden suffix marginalization

For

```text
p(q) proportional to |vL^T A_1[q1] ... A_n[qn] vR|^2,
```

an arithmetic encoder needs exact `p(q_t | q_<t)`. That conditional generally
requires contracting all suffix values. A site-canonical, fixed-context MPS
can precompute right environments. The proposed SC integration is harder:
future `original_freq1` contexts depend on future, not-yet-decoded decisions.
There is no fixed suffix-context sequence to contract.

Therefore one of the following must be proved before a Born model is a codec:

- a causally normalized autoregressive tensor/WFA law;
- a tractable dynamic program jointly over the MPS and complete future SC
  recursion; or
- a different fixed-context label stream whose suffix environments are known.

The second option is likely exponential. Calling a globally normalized Born
model “causal” without this proof is a mathematical gap. A row-stochastic HMM,
unifilar WFA, or explicitly normalized recurrent tensor avoids it and is the
right first production model.

### 3.2 Model-byte cliff

The suggested bond bank `{4,8,16,32,64}` is only plausible for aggressively
tied/factorized models. With 384 current public contexts, binary decisions,
one independent dense `chi x chi` uint16 matrix per bit/context, the model alone
is:

| `chi` | model bytes | model bpw | gross saving needed before other overhead |
|---:|---:|---:|---:|
| 4 | 24,576 | 0.006944 | 0.160880 |
| 8 | 98,304 | 0.027778 | 0.181713 |
| 16 | 393,216 | 0.111111 | 0.265046 |
| 32 | 1,572,864 | 0.444444 | 0.598380 |
| 64 | 6,291,456 | 1.777778 | 1.931713 |

This excludes boundaries, normalization state, selector, and framing. A
context-free tied four-symbol `chi=64` MPS is only roughly 33 KiB, but it is a
different, much more constrained law. Low-rank context modulation or a sparse
unifilar table is not optional engineering; it is the condition for viability.

### 3.3 Cold-read and compute cliff

If `G` bytes are global and total layer storage is `B` across `E` equally sized
experts, the symmetric cold amplification from global bytes alone is

```text
A = 1 + (E-1) G/B.
```

At the page-aligned target for the six-expert panel, `G < B/5 = 1,660,518.4`
bytes is necessary before other shared bytes. A dense context-conditioned
`chi=32` table consumes almost the entire allowance; `chi=64` fails outright.
For a large MoE, the global allowance remains approximately one compressed
expert frame, not `E` frames.

Dense causal inference costs `O(chi^2)` per binary decision, and the dependency
is sequential inside each stream. At `chi=64`, millions of expert decisions
mean tens of billions of multiply-adds with little stream-level parallelism.
The end product is MoE inference, so a source survivor must become `O(1)` or
low-rank/sparse per decision, not merely fit in storage.

### 3.4 What to test

Test the exact current causal SC decisions first, always conditioning on the
decoder-regenerated original SC frequency. Compare:

```text
current SC law
-> finite-order residual contexts
-> exact-integer unifilar WFA/HMM, states 4..64
-> low-rank/sparse causal tensor state
-> only then a globally normalized Born oracle
```

Also test raw reconstruction labels in a fixed source order. Gains from the two
streams may not be added unless one nested physical reconstruction is rebuilt.

## 4. MERA/lifting: keep the objective, drop the promise

MERA's genuine role is to remove entanglement before coarse graining
([Vidal, 2005](https://arxiv.org/abs/cond-mat/0512165)). Transferring that
intuition to a discrete label process is interesting, but the quantum
entanglement entropy of a square-root probability amplitude is not Shannon
entropy or arithmetic codelength. It also depends on ordering and on the
chosen amplitude/phase representation.

The useful version is much narrower:

- a tiny shared circuit of exactly invertible 2x2–8x8 dyadic rotations or
  integer lifting steps;
- coefficients fully serialized;
- objective = held-out **physical codelength + exact raw-source MSE**;
- exact inverse-transform analysis-by-synthesis; and
- measured decode compute/scratch and per-expert pages.

A nonorthogonal lift can reduce symbol entropy while amplifying reconstruction
error. A general coordinate-mixing inverse also means stored INT2 labels are
not native matmul weights; they must be materialized or the inverse must be
fused into inference. The proposal must not call the result “native packed
INT2” without specifying that execution path.

Do not fit a transform until a fixed-transform entropy model shows a real
held-out signal. Otherwise MERA is a large adaptive search over unproved
structure.

## 5. Reverse-channel coding: valid theorem, wrong first backend

The Gibbs posterior

```text
Q_beta(q|x) proportional to p(q) 2^(-beta d(x,xhat(q)))
```

is a coherent rate-distortion test channel. For fixed `p`, it minimizes
`KL(Q||p)+beta E[d]`. Averaging over sources gives the exact decomposition

```text
E_X KL(Q(.|X)||p) = I(X;Q) + KL(P_Q||p).
```

Thus the KL becomes the RD mutual-information term only when the proposal
matches the induced reproduction marginal. Prior mismatch is a real rate.

One-shot channel simulation can communicate near mutual information with an
additive logarithmic term under unlimited shared randomness
([Li and El Gamal, 2018](https://arxiv.org/abs/1701.02827)). But this is an
expected variable length, not a fixed physical cap. General relative-entropy
coding has runtime `Omega(exp(KL))`; A*-coding improvements require special
density-ratio structure and are stated for continuous/unimodal cases
([Flamich et al., 2022](https://proceedings.mlr.press/v162/flamich22a.html)).
A discrete MPS-plus-distortion posterior is typically multimodal, so that
escape hatch is unproved.

A useful sanity check exposes the chunking trap. A 2,048-weight block near
2.35 bpw carries roughly 4,800 communicated bits if RCC implements the entire
code. Generic search at that KL is impossible. Capping chunks at 20 bits makes
about 241 channels; the one-shot `log(I+1)+4` upper overhead is then nearly
one bpw. Larger chunks reduce framing overhead but restore exponential search.
This is not an impossibility theorem, but it shows why an actual runtime/tail
experiment is mandatory.

For losslessly reproducing an already chosen label vector, `Q` is a delta and
`KL(Q||p)=-log p(q)`: ordinary arithmetic/ANS coding under `p` is direct and
RCC adds no source gain. Therefore:

1. use MAP/beam/exact enumeration to test the small-block lossy opportunity;
2. use ordinary entropy coding for the first finite packet; and
3. consider RCC only if a measured small-KL stochastic channel beats the best
   deterministic entropy-constrained code after actual index bytes, PRNG/seed,
   overflow fallback, tail probability, and encoder/decoder work.

## 6. Orbit alignment and Gray–Wyner: sound but lower probability

Gray–Wyner genuinely trades common and private rates
([Gray and Wyner, 1974](https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1974.tb02812.x));
lossy common information for Gaussian sources is established theory
([Viswanatha, Akyol and Rose, 2014](https://arxiv.org/abs/1403.8093)). The
proposal's optimistic Gaussian toy formula is algebraically correct:

```text
F = rho 2^(2R-2r0) + (1-rho) 2^(2r0/E),
```

assuming an independent full-dimensional Gaussian common component, ideal
Gaussian coding of common/private parts, and no descriptors or finite loss.
Its conclusion is also severe: around one quarter of total variance must be a
shared component. Alignment cannot manufacture that energy.

The cold formula for a common rate `c` and private rate `p` per expert is

```text
R_physical = p + c/E
A_cold     = (p+c)/(p+c/E)
           = 1 + (1-1/E)c/R_physical.
```

At `R=2.35`, a 128-expert codec needs `c < 2.369 bpw` before common model,
index, orbit, and page overhead. A full-dimensional 4–8-bit common object
therefore fails; it must be lower-dimensional or procedurally generated.

Permutation and Up/Down gauge descriptors are affordable on the cited Qwen
shape (roughly `0.00263 bpw` together before framing), but this does not imply a
common source. For arbitrary SwiGLU shapes their bytes must be recomputed and
serialized.

Graph lifting is exactly invertible algebraically, yet random access is not
automatic. Reconstructing one odd leaf can require coarse nodes and sibling
details along a tree. Every leaf must be decoded against an actual instrumented
page reader; `O(log E)` ancestors can still exceed 2x.

Promotion oracle:

- exact cycle-consistent alignment;
- literal permutation/gauge bytes;
- one jointly fitted common/private reconstruction;
- common variance/dimension and actual common/private code lengths;
- no dense average expert hidden as a “latent”; and
- exact per-expert page unions.

Kill if the fully charged aligned oracle cannot expose about 25% compressible
common energy or if the common stream needed for the gain violates the cold
inequality.

## 7. Quotient-aware flow: locally correct, globally expensive

The change-of-variables likelihood and pullback metric

```text
G(z) = J_Tinv(z)^T J_Tinv(z)
```

are correct. Choosing a local cell generator near `G^-1/2` is the right
high-resolution geometry. Normalizing flows do support exact density
evaluation but have substantial expressive-power/compute tradeoffs
([Papamakarios et al., 2021](https://jmlr2020.csail.mit.edu/papers/v22/19-1028.html)).

At 2.15–2.5 bpw, however, local linearization may be poor. Exact candidate
scoring after the inverse is required. Likelihood gain alone is especially
misleading: a flow can lower latent NLL through volume change while worsening
the source-MSE metric by the corresponding Jacobian.

Every source-adapted flow parameter is a charged global byte and a cold page.
Model/layer/expert identity is forbidden as a free auxiliary key under the
universal contract; role, shape, public phase, and serialized adaptation are
eligible. A large flow also destroys native INT2 execution unless its inverse
is fused or weights are materialized.

Only a small triangular/integer lifting flow with a model below a predeclared
byte/read budget merits an oracle, and only after the causal census.

## 8. FIBER/Graver: no standalone entropy gain

If the transmitted statistic is deterministic, `b=f(q)`, then exactly

```text
H(q) = H(b) + H(q|b).
```

An honest statistic-plus-rank code pays at least `H(b)+H(q|b)`. The proposed
quantity

```text
H(q)-H(q|b)-B(b)
```

is nonpositive under ideal coding. It can look positive only relative to a
restricted/mismatched baseline model, in which case an ordinary joint entropy
model could exploit the same information.

Markov/Graver moves can still define a useful constrained lossy search, but
connectivity does not imply rapid mixing, low-MSE reachability, or compact
enumerative rank. Exponential Graver complexity is known even for structured
families ([Berstein and Onn, 2007](https://arxiv.org/abs/0709.1500)).

Keep FIBER only if a tiny exact enumeration proves better MSE at the same
literal rank bits than unconstrained entropy-constrained search. Do not run a
large Graver implementation based on an entropy claim.

## 9. Spatially coupled LDGM: backend, not breakthrough

Spatial coupling plus BP-guided decimation can approach the optimum of binary
LDGM ensembles; the primary results concern binary symmetric sources and
Hamming-like source coding, with the Shannon limit approached asymptotically
as degrees/coupling grow
([Aref et al., 2012](https://arxiv.org/abs/1202.4959)).

The proposed four-ary real-MSE/MPS factor graph is a substantial new problem,
not a direct corollary. Finite degree, termination, decimation failures, scale
metadata, and block boundaries all cost. A layer-global graph also destroys
expert random access; the generator must be expert-local with only a tiny
charged shared descriptor.

LDGM may realize a source law already shown to beat Gaussian. It cannot create
the missing 0.154 bpw of structure and should not run before that structure is
measured.

## 10. Recommended experimental sequence and kill rules

### Gate 0 — source-free causal ABI

Prove exact normalized conditionals, integer frequency serialization,
arithmetic re-encode, resets, independent expert frames, model bytes, and
`O(1)`/low-rank work per symbol. A globally normalized Born formula without a
tractable SC suffix marginal fails here.

### Gate 1 — direct SC recode

On authorized held-out data, compare the current SC law with the complete
causal WFA/HMM bank and identical full selection on matched Gaussian controls.
Emit the same reconstruction in a literal standalone container.

Hard goal gate under 4-KiB placement:

```text
container <= 8,302,592 bytes
net absolute saving >= 0.153935185185185 bpw
F <= 0.8
max actual routed read < 2x
```

If net held-out source-specific gain is below 0.03 bpw, kill this symbol-law
family. At 0.03–0.10 it is scientifically real but cannot justify RCC. Only a
fully charged result near 0.18 bpw gross justifies heavier tensor work.

### Gate 2 — sparse/tied tensor escalation

Escalate only if Gate 1 finds transferable signal but insufficient capacity.
Try factorized-context, low-rank, sparse, or small-period causal models. Reject
any site-specific tensor. Dense 384-context models above `chi=16` require an
extraordinary measured gross gain and cold proof before training.

### Gate 3 — tiny reversible lifting

Fit a predeclared small circuit to combined held-out physical codelength and
raw MSE. Rerun Gate 1 on its actual residual. Never add its separately fitted
gain to Gate 1.

### Parallel Gate O — orbit/common oracle

Run only the exact aligned common/private oracle and page formula. Kill before
a decoder if common energy is far below the roughly 25% optimistic requirement
or if required common bytes breach the strict cold bound.

### Gate 4 — exact small-block RD search

For blocks small enough to enumerate, compare nearest INT2, deterministic
entropy-constrained MAP, Gibbs posterior optimum, and actual source-domain MSE.
This establishes whether stochastic codeword selection adds anything beyond
lossless recoding.

### Gate 5 — finite backend

Use arithmetic/rANS first. RCC is admitted only for a demonstrated small-KL
channel with literal index/tail/fallback bytes and bounded runtime. LDGM is
admitted only after a source-law survivor and must remain expert-local.

## Bottom line

The proposal's central insight—RHT may hide usable high-order copula
information—is plausible and not closed by marginal, ICA, or short-context
tests. It is not evidence that 0.16 bits/weight exists, that a small MPS can
represent it, or that RCC can transmit it efficiently.

Freeze the immediate architecture as:

```text
decoder-regenerated SC context
-> sparse causally normalized latent state
-> exact integer probabilities
-> conventional expert-local arithmetic/rANS
```

This path uniquely tests the hypothesis while preserving identical MSE,
near-1x reads, a small model, and an honest physical ledger. MERA, Born
amplitudes, RCC, Gray–Wyner, flows, fibres, and LDGM earn compute only after
that gate supplies quantitative source evidence.
