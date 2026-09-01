# Spectral predictive-state and algebraic-renewal source models

Date: 2026-09-02
Status: source-free research checkpoint; no payload authority and no Qwen result

## Scope and evidence boundary

This report designs a successor if the frozen integrated sparse-unifilar UWFA
bank misses. While preparing it, no Qwen/current/control payload, payload
manifest, extracted decision array, or CUDA job was opened, statted, hashed, or
enumerated. Only source-free project documentation and primary literature were
consulted. The numerical quantities below are consequences of the already
documented panel constants, not new measurements.

The codec remains universal in the repository's precise sense: its fitting
rule and wire format must accept arbitrary positive SwiGLU-MoE dimensions.
Qwen can be an evaluation panel, never a decoder key. Layer, expert,
checkpoint, ancestry, router, activation, and token identity are forbidden
probability keys. Any source-fitted tap, state transition, emission frequency,
or duration parameter must be serialized and charged.

## Executive verdict

Two source laws remain sufficiently different from the frozen UWFA bank to
justify a decisive oracle:

1. **HANKEL-CSR**: use spectral Hankel/OOM/PSR learning to discover predictive
   state, then compile that teacher into a probability-safe causal-state and
   renewal machine with exact integer frequencies. The spectral model is a
   discovery instrument, not an unverified floating-point decoder. This is the
   most principled successor to a negative hand-designed UWFA bank because it
   learns the state partition instead of selecting one of five predetermined
   state updates.
2. **POLY-SYNDROME-R**: apply an invertible, unit-lower-triangular GF(2)
   lifting transform whose pivot symbols are long-range sparse parity
   discrepancies, then entropy-code those discrepancies with an observable
   renewal law. It can maintain hundreds of simultaneous open parity checks
   with a bitset, a regime that a generic `chi <= 64` automaton cannot express
   without exponential state. This is not Slepian-Wolf coding and does not
   assume another role as side information.

Neither candidate is evidence that the missing Qwen structure exists. Both
have hard, inexpensive failure modes. The correct strategy is to run a cheap
GF(2) recurrence/parity oracle in parallel with the more general predictive-
state census, kill either as soon as its exact remaining-byte upper bound
falls below target, and never add gains from separately fitted runs. If both
survive, the only admissible composite is one literal nested transform and one
independently decoded container.

## Exact opportunity and byte budget

For the documented evaluation panel:

```text
N                         = 28,311,552 weights
experts                   = 6
weights/expert            = 4,718,592
current R                 = 2.5 bpw
current F                 = 0.9888693569009007
required net rate saving  = 0.15288996696291447 bpw
conservative design gate  = 0.153 bpw
```

At unchanged reconstruction, the exact target is a saving of
`4,328,552.2499` bits or `541,069.0312` bytes. The conservative `0.153` gate is
`541,458.432` bytes. One equal 2.5-bpw expert share is exactly `1,474,560`
bytes. A global model is cheap in cold traffic but not free in physical rate:

| Page-rounded global packet | Physical charge | Extra cold read over one 2.5-bpw expert | Gross saving needed for net 0.153 |
|---:|---:|---:|---:|
| 16 KiB | 0.00462963 bpw | 0.011111x | 0.15762963 bpw |
| 32 KiB | 0.00925926 bpw | 0.022222x | 0.16225926 bpw |
| 64 KiB | 0.01851852 bpw | 0.044444x | 0.17151852 bpw |
| 128 KiB | 0.03703704 bpw | 0.088889x | 0.19003704 bpw |

The current documented cold read is `1.16944444x`. Thus a 32-KiB or 64-KiB
shared model would conservatively move it only to approximately `1.19167x` or
`1.21389x`, before any layout improvement. The binding difficulty is entropy,
not the `<2x` read cap. A source model larger than 128 KiB is unattractive: it
must first discover almost `0.19 bpw` gross, and its inference state is less
likely to fit cache.

### Common control contract

Both candidates require three null families, with the complete candidate
search rerun independently on every member:

1. moment-matched Gaussian source tensors with the same legal shapes, roles,
   transform, quantizer, SC recursion, stream partition, and byte framing;
2. exact within-public-context permutations, which retain all eligible
   one-symbol frequencies while destroying sequential dependence; and
3. dyadic chunk shuffles at several scales, which localise the dependency
   length without changing the contents of each chunk.

Controls must use their own train/validation selection, state/tap discovery,
integer refit, serialization, arithmetic coding, and page ledger. Refitting
only the source winner is anti-conservative. The source and controls get the
same candidate count, stopping rule, CuPy precision, and wall-clock budget.
Whole experts and whole layers—not symbols—are the independent units for
confidence statements.

## What the prior screens close—and what they do not

The frozen UWFA v2 bank covers five **predetermined** procedural state updates:
bounded suffix, one XOR/parity sketch, modular-one accumulator, rolling affine
sketch, and saturating signed count; `chi={2,4,8,16,32,64}`; and five reset
lengths. It fits only state/context Q0.16 emissions. A miss is strong evidence
against those 150 laws, but it does not test data-discovered causal states,
explicit unbounded age, a nonpositive low-rank observable representation, a
large structured parity register, or a learned sparse recurrence polynomial.

The earlier bit-context work closes short categorical neighborhoods and the
specified aligned Up-to-Down conditioning. It does not close dependencies
whose every proper low-order marginal is uniform, such as long parity checks.

The earlier syndrome no-go closes the specified Slepian-Wolf cell in which
another already decoded role is side information. POLY-SYNDROME-R instead
performs a bijective transform of one stream. It transmits all free symbols and
entropy-codes predictable parity pivots. There is no unknown coset decoding,
no belief propagation, and no external side information.

The earlier LZ/MDL cell looks for repeated substrings. A parity-constrained
sequence can have no useful repeated substrings and still have a low entropy
rate. Conversely, a causal-state machine can merge histories that have no
common suffix but induce the same future distribution.

The earlier procedural-generator screens model numeric weights or
initialization ancestry. POLY-SYNDROME-R models only the transmitted
quantizer-decision law, uses no provenance, and is charged as an ordinary
source-adaptive predictor.

| Family | State/structure it can expose | Boundary relative to this report |
|---|---|---|
| bounded bit context / CTW | suffix tree up to a fixed depth | cannot in general merge nonsuffix-equivalent histories or retain many remote parities |
| LZ grammar | repeated substrings in a finite window | a parity-constrained stream may have no long repeats |
| frozen UWFA v2 | one of five fixed update laws, at most 64 tabular states | HANKEL-CSR learns topology; POLY-SYNDROME-R has a factored register with potentially hundreds of live bits |
| positive HMM / nonnegative MPS | finite positive latent mixture; dense versions cost `O(chi^2)` per symbol | a signed low-rank spectral teacher can reveal a smaller predictive space, then must pass the positive causal compiler |
| value TT/MPO or matrix low rank | numerical low rank of the weight values | neither candidate approximates values by low rank; both model the discrete decision law |
| aligned-role syndrome | conditional entropy given a separately decoded role | POLY-SYNDROME-R is a bijection of one stream and uses no external role side information |

## Mathematical basis

### Hankel rank, observable operators, and predictive state

For a stationary symbol process, define the infinite Hankel matrix

```text
H[p,s] = P(prefix p followed by suffix s).
```

For a rational series, finite Hankel rank `r` is equivalent to an `r`-state
linear weighted-automaton realization. Spectral learning estimates a finite
Hankel block, takes a rank-`r` factorization, and constructs symbol operators.
This is attractive here because the rank concerns **future-prediction space**,
not covariance or numerical low rank of the weight matrix. Balle and Maillard
also provide a single-trajectory spectral analysis, which is the relevant data
regime for long weight-symbol streams.

An observable-operator state can be written

```text
z_t          = normalized prediction vector after history y_<t
score(y)     = c^T B[y,u_t] z_t
z_(t+1)      = B[y_t,u_t] z_t / score(y_t)
```

where `u_t` is eligible decoder-visible context such as semantic role, polar
level, regenerated baseline-frequency bin, coordinate phase, and reset phase.
OOMs/PSRs can be substantially smaller than a positive HMM realization because
their operators may be signed. This compactness is useful for **discovery**,
but creates the positive-realization problem: a finite-sample spectral model
can assign negative or unnormalised probabilities. Raw spectral floats are
therefore not a physical codec.

### Causal states

Two histories are causally equivalent when they induce the same distribution
over all futures. Their equivalence classes are predictive sufficient
statistics. CSSR-style state splitting starts from a coarse model, separates
histories whose future laws differ, and determinizes symbol transitions. The
result is unifilar: after the current state and decoded symbol are known, the
next state is known.

This offers a precise bridge from an expressive spectral teacher to a safe
decoder. Spectral coordinates propose which histories are predictively close;
state splitting enforces deterministic updates; held-out MDL merging controls
model size. Unlike the frozen UWFA bank, the transition topology itself is
learned and transmitted.

### Renewal and hidden semi-Markov structure

An ordinary finite HMM gives geometric state dwell times. An HSMM models
duration explicitly. For an observable event stream, the full hidden-state
machinery is unnecessary: the time `a_t` since the previous event is causally
known, and the exact event probability is a hazard

```text
h(a) = P(event at age a | no earlier event).
```

Discrete-time renewal theory shows that age can be the minimal predictive
state and that an unbounded interevent distribution can require countably many
causal states. This is a concrete blind spot of any fixed `chi <= 64` bank.
The decoder need only maintain an integer age and consult a compact dyadic or
piecewise-geometric hazard table. No hidden belief and no floating point are
required.

### Algebraic lifting

Let `x` be a binary decision vector in a fixed causal order. For a pivot set
`P`, choose sparse taps `J_p` strictly earlier than pivot `p` and define

```text
e_p = x_p XOR (XOR over j in J_p of x_j),   p in P
e_t = x_t,                                  t not in P.
```

This is multiplication by a unit-lower-triangular matrix over GF(2), hence a
bijection with determinant one. Decoding uses the same order:

```text
x_p = e_p XOR (XOR over j in J_p of x_j).
```

The transform alone cannot change joint entropy. Its purpose is to expose a
simple conditional law: if a long parity relation usually holds, the
discrepancy `e_p` is strongly biased or has a low renewal entropy rate. The
arithmetic coder then captures that entropy. This makes the accounting honest:
all nonpivot bits, all discrepancies, tap descriptions, seeds, tables,
framing, and padding are transmitted.

For a one-dimensional recurrence with tap polynomial `J`, the same operation
is

```text
e_t = x_t XOR XOR_{j in J} x_(t-j).
```

Massey's shift-register synthesis finds the shortest exact LFSR for a finite
sequence. Feng-Tzeng generalises synthesis to multiple sequences sharing one
connection polynomial. Exact synthesis is only an initializer here: real
streams can be noisy, so all recurrence candidates must be selected on
development data and evaluated on untouched experts/layers.

## Architecture A: HANKEL-CSR

### A1. Observable stream

Run the existing universal decoder logic at the encoder and expose each exact
binary arithmetic decision together with only decoder-regenerated context:

```text
u_t = (role, polar_level, bin(original_freq1), public_phase, reset_phase).
```

No extracted context array is transmitted. On decode, the same existing polar
recursion regenerates `original_freq1` and the other context fields before
asking HANKEL-CSR for the new exact frequency.

Use independent resets at literal stream/frame boundaries. Never carry state
between experts; that would harm random access and silently create a shared
read dependency.

### A2. Spectral discovery, not deployment

On auxiliary folds, construct sketched history/future moment blocks

```text
H_00 = E[phi(history_t) psi(future_t)^T]
H_uy = E[phi(history_t) 1{u_t=u,y_t=y} psi(future_(t+1))^T].
```

The frozen feature bank should include:

- exact suffix/future words only as a local control;
- public sparse parity probes over dyadic windows;
- event-age indicators and dyadic age bins;
- multiscale counts and modular phases;
- public CountSketch projections of longer prefix/future tests.

Use ranks `r={4,8,16,32}` and horizons
`{32,128,512,2048,4096}`. The SVD and operator recovery run in CuPy. A
matched control repeats feature selection, rank selection, and every search;
the source never gets a larger basis than the controls.

Future tests are training labels only. The emitted decoder receives neither a
future feature nor a spectral state trace; it reconstructs its compiled state
solely from the transmitted model, the public context, and earlier decoded
bits.

The singular spectrum is diagnostic only. The decisive float oracle is
held-out sequential log loss from the spectrally initialised model after
clipping every reachable conditional to `(0,1)`. If even this uncharged
teacher cannot find at least `0.19 bpw` gross on whole-expert/layer holdout,
there is too little margin for state compilation and bytes.

### A3. Compile the teacher into causal states

For each development history, form a prediction signature containing the
teacher probabilities of the frozen future tests. Then:

1. cluster signatures under future-distribution KL, never Euclidean state
   coordinate distance;
2. split any cluster for which `(state, public_action, decoded_bit)` has more
   than one successor cluster;
3. split on a held-out two-sample future-law test when histories remain
   predictively distinct;
4. greedily merge only when the exact integer arithmetic saving exceeds the
   extra serialized transition/emission bytes;
5. attach an explicit age register to event channels instead of allocating one
   table state per possible age;
6. refit on development and score the untouched outer cell.

Candidate state counts are `S={32,64,128,256,512}`. If determinization exceeds
512 states or requires model/layer/expert keys, the finite candidate fails even
if the spectral teacher is good.

### A4. Exact integer realization

Deployment never contracts a floating OOM. A compact candidate uses:

```text
state              uint16
delta[group,state,bit] -> uint16 next state
freq1[state,context]    -> uint16 in [1,65535]
age                     uint32, exact decoder-derived counter
hazard_freq1[class,age_bin] -> uint16
```

The context mapper is a serialized, bounded decision tree over eligible public
fields and yields at most 32 contexts. Transition action has at most four
groups. All smoothing and rounding are specified in integer arithmetic.
Every stream uses the existing canonical 32-bit arithmetic coder and must
decode and canonically re-encode byte-identically.

A representative `S=256` packet is:

| Field | Raw bytes |
|---|---:|
| four-group, two-symbol uint16 transition table | 4,096 |
| 32-context uint16 emission table | 16,384 |
| context mapper and reset descriptors | <=2,048 |
| renewal hazard tables | <=2,048 |
| header, hashes, checks and reserve | <=4,096 |
| page-rounded packet | 32,768 |

The 32-KiB charge is `0.00925926 bpw`; gross source saving must exceed
`0.16225926 bpw` to net `0.153`. An `S=512` form is capped at 64 KiB and must
gross more than `0.17151852 bpw`. Larger forms are killed before payload.

### A5. Decode work and cold traffic

The compiled machine performs one emission lookup, one integer range-coder
step, one transition lookup, and one optional age update per decision. At the
conservative `2.5N = 70,778,880` binary-decision envelope, roughly 8–15 small
integer operations per decision is `0.57–1.06` billion simple operations over
the entire six-expert panel, or `94–177` million per expert. The 32–64 KiB
model fits cache. This is much cheaper than a dense rank-32 OOM contraction,
which would require tens of billions of multiply-adds over the panel.

Conservatively adding the entire global model to each cold expert read gives
approximately `1.19167x` for 32 KiB and `1.21389x` for 64 KiB relative to the
current documented ledger. Both remain comfortably below `2x`.

### A6. Decisive gates

**Synthetic mechanism gate.** Before any model payload, require exact recovery
and compression on four source-free processes: separated parity checks,
an even process with merged nonsuffix histories, a long-tailed renewal process,
and a simple nonunifilar source whose observable causal-state set is much
larger than its hidden generator. Require independent integer decode/re-encode
and CPU/CuPy agreement for all spectral statistics.

**Rank gate.** On auxiliary data only, retain a rank/horizon cell only if its
future-prediction singular values exceed both within-context permutations and
all identically searched matched Gaussian controls. This is a diagnostic gate,
not a bitrate claim.

**Teacher kill.** Kill the family if the one-sided 95% whole-expert/layer
held-out upper confidence bound of the best float teacher is below `0.19 bpw`
gross, or if source advantage over the maximum identically searched control is
below `0.10 bpw`. The deliberately high 0.19 threshold protects against the
positive-realization and model-byte losses.

**Compiler kill.** For each candidate, serialize its real integer packet
before test scoring. At every range-coder checkpoint, use the exact physical
upper bound

```text
max_final_saving = baseline_total_bytes
                   - fixed_candidate_bytes
                   - bytes_already_irrevocably_emitted.
```

The unseen suffix is granted zero bytes. If `8*max_final_saving/N <= 0.153`,
stop that candidate immediately; no continuation can rescue it.

**Promotion.** Require all of the following:

- nested whole-layer/whole-expert operational holdout survives after each
  fold's model bytes;
- the final literal packet saves more than `541,458.432` bytes against a
  same-framing identity recode;
- actual rate remains in `[2.15,2.5]`, unchanged reconstruction gives
  `F<=0.8`, and exact page union is `<2x` for every expert;
- every matched Gaussian control repeats the complete basis/rank/state/model
  selection and stays below the predeclared specificity ceiling;
- a fresh independent process deserializes only the packet, reconstructs every
  original decision and weight, and recomputes rate, MSE, F, and cold pages.

### A7. Main risks

- Low Hankel rank need not have a comparably small positive or unifilar
  realization. This is the central scientific risk, not an implementation
  detail.
- Finite-sample spectral operators can be unstable or negative. Clipping can
  create an impressive teacher but poor compiled state model.
- A dense RHT may make the required causal state enormous even when source-
  coordinate structure is simple.
- Whole-expert folds provide far fewer independent units than raw symbol count;
  symbol-level confidence intervals are invalid.

## Architecture B: POLY-SYNDROME-R

### B1. Transform family

Use a frozen ladder of causal GF(2) transforms:

1. **One-dimensional sparse recurrences.** Taps come from public degrees
   `{2,3,5,8,12,16}` and spans `{32,128,512,2048,4096}`.
2. **Shape-derived two-dimensional checks.** Taps are earlier row/column,
   role-aligned, or dyadic-tile coordinates derived only from canonical shapes.
3. **Shared multisequence polynomials.** A single polynomial is fit across
   multiple development lanes or roles using multisequence shift-register
   synthesis; it is never keyed by expert identity.
4. **Block systematic checks.** Within a page-sized source block, each check
   has a unique causal pivot and sparse earlier taps. Many checks can be open
   simultaneously.

The candidate dictionary is finite and frozen before evaluation. A practical
stage-0 bank is at most 8,192 public candidates: public seeds generate sparse
tap sets, while exact Berlekamp-Massey/Feng-Tzeng outputs from development data
add a bounded serialized shortlist. Arbitrary noisy parity discovery is not
claimed; in general it approaches the learning-parity-with-noise problem. The
bounded bank and untouched holdout make the test computationally honest.

### B2. Why structured parity state matters

Suppose 128 checks overlap inside a page. A generic table automaton that tracks
all live check parities can require up to `2^128` states. POLY-SYNDROME-R stores
the 128 parity accumulators as a 128-bit register and updates them through a
sparse incidence schedule. Its probability law only needs to know whether the
current pivot discrepancy is zero, plus the causal renewal state of previous
discrepancies. This is a factored deterministic state representation, not a
larger dense WFA.

Every pivot can use the regenerated original arithmetic frequency. If parity
`q_p` predicts the original bit, map the baseline frequency to discrepancy
space exactly:

```text
base_freq(e_p=1) = original_freq1       if q_p=0
                 = 65536-original_freq1 if q_p=1.
```

A learned Q0.16 residual model may then replace or integer-blend this baseline,
provided the rule is serialized and exactly replayed. Nonpivots follow the
original model.

### B3. Renewal error law

For pivot discrepancy `e_p`, maintain:

```text
age = number of pivot opportunities since the previous e=1
type = public check-family / role / level class
```

Use exact hazard frequencies in dyadic age bins
`{0,1,2,3,4-7,8-15,...,>=2048}` with an optional periodic tail class. A
piecewise-geometric tail is especially cheap: after a threshold, the hazard is
constant or repeats modulo a small public period. This captures bursty
constraint failures that an iid syndrome model misses and avoids one table
state per age.

### B4. Required strength

For intuition only, assume a pivot's old conditional entropy is one bit and
its discrepancy is iid Bernoulli with error probability `epsilon`. If there
are `rho` pivots per source weight, gross saving is

```text
G ~= rho * (1 - h2(epsilon)) bpw.
```

With a conservative 16-KiB complete model charge (`0.00462963 bpw`):

| pivots/weight `rho` | error `epsilon` | iid gross | net after 16 KiB |
|---:|---:|---:|---:|
| 0.20 | 0.02 | 0.171712 | 0.167082 |
| 0.25 | 0.05 | 0.178401 | 0.173771 |
| 0.25 | 0.10 | 0.132751 | 0.128121 — fail |
| 0.3333 | 0.10 | 0.177001 | 0.172372 |
| 0.30 | 0.05 | 0.214081 | 0.209451 |

The real test uses exact old and new range-coder bytes, not this approximation.
Renewal clustering can make the error entropy rate lower than `h2(epsilon)`,
but only held-out arithmetic length may claim that gain. The table shows the
required phenomenon is strong: roughly one reliable nonlocal constraint per
3–5 weights. A handful of block parities cannot matter.

### B5. Packet and read ledger

A conservative packet budget is:

| Field | Bytes |
|---|---:|
| global transform/version/header page | 4,096 |
| <=32 tap/check descriptors, masks, pivots and selectors | <=4,096 |
| Q0.16 discrepancy/hazard tables | <=4,096 |
| directories, hashes, reserve and page rounding | <=4,096 |
| total | 16,384 |

Any source-adaptive per-frame selector is inside that expert's private frame
and charged. A 16-KiB global worst-case read adds `0.011111x`, giving about
`1.18056x` on the current ledger. If only the 4-KiB global page plus one small
private selector page is needed, the true union is lower.

With average tap degree 8 and pivots on 40% of the conservative 70.8-million
decision envelope, there are about 226.5 million XOR tap operations over the
whole panel, 37.7 million per expert, plus range coding and O(1) renewal
updates. A page incidence schedule can use 64/128-bit bitsets for training and
CuPy scoring. Decode remains sequential but integer-only and cache-resident.

### B6. Decisive gates

**Exact-recurrence diagnostic.** On development folds, run Berlekamp-Massey
and multisequence synthesis on each eligible lane. Score the resulting
connection polynomial on untouched folds. If its held-out discrepancy is
approximately fair or the polynomial order consumes its saving, discard it.
Training linear complexity alone is never evidence.

**Sparse-check oracle.** CuPy scores all frozen check candidates using packed
XOR kernels. Selection uses development only. The test statistic is exact
held-out range-coder saving of pivot discrepancies, after the literal tap and
selector packet. Repeat the entire 8,192-cell selection on every Gaussian
control and on within-public-context multiscale shuffles.

**Analytic early kill.** If a partially scored check set has pivot count `m`,
current exact baseline pivot bits `B0`, model/descriptor bytes `M`, and the
most favorable possible remaining discrepancy cost zero, its maximum net gain
is bounded by `(B0 - 8M)/N`. Kill immediately if this is `<=0.153`. During real
range coding, use the stronger irrevocably-emitted-byte bound from HANKEL-CSR.

**Family kill.** Kill POLY-SYNDROME-R without building a full container if no
whole-expert/layer held-out cell reaches `0.162 bpw` gross and no nested
source-minus-control lower bound reaches `0.10 bpw`. The `0.162` gate covers a
16-KiB packet, framing, and a small safety margin.

**Promotion.** The final criterion is identical to HANKEL-CSR: more than
`541,458.432` actual bytes saved against same framing, independent exact
inverse, unchanged reconstruction, `F<=0.8`, full matched-control search, and
`<2x` literal cold page union.

### B7. Main risks

- Exact low linear complexity is common in engineered sequences but not
  expected in Gaussianised neural weights.
- Approximate noisy parity discovery can be computationally hard; a huge
  adaptive check search will merely overfit and incur descriptor cost.
- A parity relation in raw labels may be destroyed by the current RHT or SC
  ordering. The ordering ladder must be frozen and charged, not selected on
  the test fold.
- Multiple check gains are not additive when pivots or information overlap.
  Only the rank of the final triangular transform and one emitted stream count.

## Combined architecture, if both survive

The only coherent composite is:

```text
original SC decisions
  -> one causal POLY-SYNDROME lifting transform
  -> one HANKEL-CSR model fitted to the transformed stream
  -> one exact arithmetic payload
  -> inverse lifting
  -> unchanged universal STRATA reconstruction
```

The HANKEL-CSR state must be retrained on the discrepancy stream. Separate
reported savings must not be added. The complete nested pipeline, model bytes,
selectors, arithmetic termination, padding, and page union are rescored from
scratch. A useful division of labour is that algebraic lifting exposes many
simultaneous parity constraints, while causal-state/renewal coding models the
remaining discrepancy dynamics.

## Experimental sequence after a UWFA miss

1. **Freeze and independently audit two source-only oracle specifications.**
   Include exact feature/check dictionaries, folds, control seeds, integer
   probability law, model caps, early-kill formulas, and byte layout before any
   payload access.
2. **Run source-free mechanisms.** Require synthetic parity, causal-state,
   nonunifilar, and heavy-tail renewal fixtures plus CPU/CuPy agreement.
3. **Run the cheap POLY-SYNDROME-R scanner.** It is linear/XOR-heavy and can be
   killed rapidly. Do not build a container unless held-out gross saving
   exceeds 0.162 bpw.
4. **Run the HANKEL spectral teacher.** Use CuPy randomized/sketched SVD, but
   promote only on sequential held-out codelength. Kill below 0.19 bpw float
   gross.
5. **Compile only a surviving teacher.** Enforce <=512 deterministic causal
   states, <=64-KiB model, exact Q0.16 frequencies, canonical re-encoding, and
   the physical streaming kill bound.
6. **Repeat full searches on controls.** A control does not merely refit the
   source winner; it repeats feature/tap/rank/state/hyperparameter selection.
7. **Build one sealed packet and audit in a fresh process.** Recompute actual
   `R`, source MSE, `F`, reconstruction digest, and exact owner-aware 4-KiB page
   unions.
8. **Universality follow-up.** A Qwen survivor remains panel evidence. Before a
   universal-performance claim, repeat on a disjoint SwiGLU-MoE family or at
   minimum a different-dimension portability panel.

## Why these could plausibly clear 0.153 bpw

The claim is conditional, not empirical:

- A low-rank predictive process can have nearly Gaussian one-symbol marginals
  while a small future-prediction state removes substantial entropy. Spectral
  Hankel learning tests this directly instead of guessing a state update.
- An array with many sparse parity constraints can have exactly fair marginals
  and zero low-order correlation. A systematic causal check transform extracts
  one nearly deterministic discrepancy per independent constraint. The rate
  table shows that a density of 0.20 constraints/weight at 2% error, or 0.25 at
  5% error, is enough after a 16-KiB packet.
- A renewal law can reduce discrepancy entropy even when its marginal error
  rate looks too high, because long quiet intervals and clustered failures are
  encoded through exact hazard rather than an iid Bernoulli table.

These are also decisive hypotheses. If the data-discovered spectral teacher
cannot gross 0.19 bpw and the best held-out causal algebraic transform cannot
gross 0.162 bpw, this entire predictive-state/parity/renewal frontier should be
deprioritised. Another larger HMM, suffix table, generic LZ parser, or denser
local context would not be an ambitious response to those failures.

## Primary literature and prior-art boundary

- Thon and Jaeger unify multiplicity automata, OOMs, and PSRs and their
  learning framework: [JMLR 16 (2015), 103–147](https://www.jmlr.org/papers/v16/thon15a.html).
- Balle and Maillard give spectral learning from a single dependent trajectory
  and state the Hankel-rank/WFA connection used here:
  [PMLR 70 (2017), 361–370](https://proceedings.mlr.press/v70/balle17a.html).
- Shalizi and Shalizi develop causal-state splitting reconstruction as a
  nonlinear recursive predictor for discrete sequences:
  [UAI 2004 preprint](https://arxiv.org/abs/cs/0406011).
- Marzen and Crutchfield identify minimal causal states and entropy structure
  for discrete-time renewal processes, including processes with unbounded
  causal-state sets:
  [Entropy 17 (2015), 4891–4917](https://arxiv.org/abs/1408.6876).
- Yu reviews explicit-duration/hidden semi-Markov models and explains the
  geometric-duration limitation of ordinary HMMs:
  [Artificial Intelligence 174 (2010), 215–243](https://doi.org/10.1016/j.artint.2009.11.011).
- Massey proves that the Berlekamp algorithm synthesizes the shortest LFSR for
  a finite sequence:
  [IEEE Transactions on Information Theory 15 (1969), 122–127](https://crypto.stanford.edu/~mironov/cs359/massey.pdf).
- Feng and Tzeng generalise shift-register synthesis to multiple sequences:
  [IEEE Transactions on Information Theory 37 (1991), 1274–1287](https://doi.org/10.1109/18.133246).
- Witten, Neal, and Cleary give the fixed-precision arithmetic-coding basis for
  separating the causal probability model from exact channel coding:
  [Communications of the ACM 30 (1987), 520–540](https://doi.org/10.1145/214762.214771).
- The negative-probability problem in finite-sample spectral automata is a
  documented issue, motivating the safe compiler rather than raw spectral
  deployment: [Glaude, Enderli, and Pietquin, 2015](https://inria.hal.science/hal-01225810/preview/ASRU_2015_HGCEOP.pdf).
- Classical syndrome source coding transmits a syndrome and relies on a source
  law to select a member of the coset. That is not the invertible triangular
  transform proposed here: [Ancheta, IEEE TIT 1976](https://ntrs.nasa.gov/citations/19760057252).

A targeted literature search did not find an LLM-weight PTQ codec that learns
an OOM/PSR from quantizer decisions and compiles it into an exact causal-state
wire model, or one that combines causal GF(2) parity lifting with a renewal
discrepancy coder. This is a research-novelty assessment, not a patentability
opinion. Spectral sequence models, arithmetic coding, LFSR synthesis, HSMMs,
and syndrome coding individually have extensive prior art.

## Final recommendation

If UWFA v2 misses, do **not** jump directly to a dense MPS or raw signed OOM.
Freeze HANKEL-CSR as the main successor and POLY-SYNDROME-R as a cheap,
orthogonal algebraic screen. The former answers whether a predictive state was
missed by the hand-designed automata; the latter answers whether the missing
information consists of many simultaneous long-range parity laws that generic
finite state represents exponentially badly. Both retain native random access,
have credible 16–64-KiB model ledgers, use exact integer decoding, and possess
hard early-kill bounds. Their required signal is severe enough that a negative
result would be scientifically meaningful.
