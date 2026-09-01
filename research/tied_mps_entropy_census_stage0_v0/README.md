# Tied persistent-regime HMM entropy census — sealed source-only Cell A

This package asks a narrow but consequential question: can a small, tied,
causal hidden-state probability model losslessly replace the current
POLARIS/STRATA arithmetic stream well enough to move the unchanged finite
reconstruction from `F=0.9888693569009007` to `F<=0.8`?

It is a producer source package, not a result.  At seal time **no model payload**,
decoded decision stream, Gaussian control, NumPy array, CuPy context, or CUDA job
was opened.  A real run is forbidden until a **separate independent source
review** binds this exact source-manifest hash.

## The rate question that is actually tested

The current finite artifact has `s=0.008074080480766676 bpw`.  Holding its
reconstruction fixed therefore requires a standalone physical saving of

```text
-0.5 log2(0.8) - 0.008074080480766676
  = 0.1528899669629145 bpw.
```

The `0.11356063457 bpw` number belongs to a speculative ideal composite whose
components have not been realized in one finite artifact.  It is reported but
is never used as this lossless cell's pass threshold.  Savings from separate
oracles are not added.

Every candidate packet must remain in `2.15<=R<=2.5 bpw`.  Storage charges the
model header and all uint16 tensors, selector, directory, integrity/framing
bytes, exact arithmetic termination and final-byte padding, page alignment,
and any padding needed to keep `R>=2.15`.  A routed expert read includes every
cold global page plus the worst-unaligned page union of its one sequential
expert frame, and must be strictly below `2x` the equal-share physical bytes.

## Cell A0: deliberately local-only

A0 fits deterministic suffix states of depths 0 through 8, crossed with polar
level, a 16-bin bucket of the decoder-regenerated current probability, and
small phase `P in {1,2,4}`.  Jeffreys-smoothed probabilities are rounded to
Q0.16 and their complete physical model is charged.  This is a cheap
finite-order Markov control.  A miss is **local-only**: it cannot close latent
regimes, edge-emitting automata, parity, or general tensor networks.

## Cell A1: the true latent model, and its exact boundary

A1 is a tied, nonnegative, state-emitting persistent-regime HMM with
`chi={4,8,16,32,64}`, `P in {1,2,4}`, three frozen starts, twelve scaled
Baum--Welch iterations, and a reset every 4096 selected decisions.  For public
context `c`, decoded symbol `y`, shared transition `T`, and state emission `E`,
its symbol-conditioned matrix is

```text
A_(c,y)[i,j] = T[i,j] E[c,j,y].
```

Because `E[c,j,0]+E[c,j,1]=1` and `T 1=1`, exact causal normalization is

```text
p(y_t | y_<t,c_t)
  = alpha_(t-1) A_(c_t,y_t) 1
    / (alpha_(t-1) (A_(c_t,0)+A_(c_t,1)) 1),
alpha_t = alpha_(t-1) A_(c_t,y_t) / sum(alpha_(t-1) A_(c_t,y_t)).
```

The serialized model contains a 256-byte model header, `chi` initial uint16
values, `chi^2` transition uint16 values, and `6*16*P*chi` emission uint16
values.  It is a genuine latent HMM and not a procedural suffix table.

This is nevertheless not an exhaustive generic MPS/WFA census.  Its shared
symbol-independent `T` followed by target-state emission is a persistent-regime
HMM subclass.  In particular, a canonical uniform fixed-horizon even-parity
language needs an edge-emitting symbol transition and either a terminal right
environment or an authenticated final-position context.  That family is
outside A1.  A negative result closes only the frozen `chi<=64`, `P<=4`,
reset-4096, 12-iteration, three-start cell.  It is **not a global HMM-MLE
proof**, and says still less about arbitrary MPS, TTN, MERA, or site-specific
models.

## Authentication and exact finite comparison

The stream lock has exact top-level and per-stream schemas.  It binds:

- the 2.5-bpw current artifact by absolute path, bytes, and SHA-256;
- immutable global and per-expert decoder bytes;
- selected bits, decoder-regenerated Q0.16 probabilities, and polar levels;
- every current arithmetic payload by absolute path, bytes, and SHA-256;
- whole-layer and whole-expert fold labels, which are partition metadata only.

The independent extraction receipt binds the artifact hash and a canonical
inventory hash over every descriptor.  The run holds regular-file descriptors,
rejects symlink leaves with `O_NOFOLLOW`, and verifies descriptor identities at
the end.  Before any candidate score is trusted, the current arithmetic coder
is rerun and must reproduce every logical length, byte sequence, and payload
hash exactly.

The only probability keys are role-equivalent polar level, a shape-independent
16-bin decoder probability bucket, phase at period at most four, and causal
decoded state.  Layer, expert, model, checkpoint, absolute site, source weight,
future decision, and encoder-only search identity are forbidden.  Qwen is an
evaluation panel, not decoder side information; the algorithm is intended to
be shape-derived and portable across SwiGLU MoE checkpoints.

Whole-layer/expert cross-fitting groups equal `(layer_group, expert_group)`
streams, trains after excluding every stream with the same layer **or** expert,
selects the frozen seed using training likelihood only, and charges one complete
quantized model packet per fold.  No identity enters a probability context.

## Gaussian control order

Eight matched tensors must each be independently sampled, passed through the
entire frozen current encoder, and independently extracted.  Only if the source
first reaches the absolute `0.1528899669629145 bpw` threshold may their paths
even be resolved.  The source-selected chi/P cell is independently refit from
all three seeds, quantized, selected, coded, and charged on every control.
Results report direct source gain and Qwen-specific excess over the independently
encoded Gaussian mean.  There is **no control subtraction** that can turn an
absolute failure into a pass.

## Critical assessment of the proposed broader architecture

- If `b=f(q)`, then `H(q)-H(q|b)=H(b)`.  Sending the sufficient statistic at
  its own entropy creates no net lossless saving.  **FIBER** can still change
  lossy geometry or search, but its label is not an entropy rebate.
- An **RCC** fixed cap cannot create entropy.  Tail/failure fallback,
  reservoir slack, search selectors, framing, and termination all require
  literal bytes.
- Orthogonal ORBIT/MERA operations preserve raw squared error; benefit must
  arise from a changed finite quantizer, and fitted transforms are side bytes.
  A volume-preserving Riemannian flow likewise preserves differential entropy;
  a non-volume-preserving one must transmit the parameters realizing its
  Jacobian and win in one decoded packet.
- The audited **Gray--Wyner**-like same-layer superoracle captured
  `0.015534903625203362` pooled, or `0.016534903625203354` under its favourable
  credit.  That is far below the `0.14566207552117194` speculative-gap capture
  and the `0.19099525693951513` standalone capture, and roughly fifteen times
  below a 25% common-variance aspiration.  Shared pages also enter every routed
  cold read.

## Primary mathematical sources

The experiment uses the classical finite-state latent-process likelihood and
its bounded EM search, while treating a negative local optimum as only a scoped
search result.  Useful primary sources are:

- Balle and Maillard, *Spectral Learning from a Single Trajectory under
  Finite-State Policies* (probabilistic/stochastic weighted automata):
  https://proceedings.mlr.press/v70/balle17a.html
- Srinivasan et al., *Quantum Tensor Networks, Stochastic Processes, and
  Weighted Automata* (explicit equivalence boundaries among uniform MPS,
  stochastic processes, and weighted automata):
  https://arxiv.org/abs/2010.10653
- Yang and Kieffer, *Fixed-slope universal lossy data compression* (individual
  sequences and explicit code-length-plus-distortion objective):
  https://ieeexplore.ieee.org/document/623145/

These sources motivate the hypothesis; none supplies evidence that this finite
Qwen panel passes.

## Fail-closed launch lifecycle

A wrong token rejects before output creation, input resolution, CuPy import, or
CUDA.  A correct token first reserves an absent output directory, authenticates
the separate review and sealed source, opens the stream lock through held
descriptors, and only then imports CuPy.  Gaussian controls remain unopened
unless the exact absolute source gate survives.  `COMPLETE.json` is written
exclusively last; an incomplete directory is never resumed or accepted.

After an independent auditor issues the required review receipt, the precise
launch form is:

```bash
/usr/bin/python3.12 -B -I research/tied_mps_entropy_census_stage0_v0/stage0_census.py \
  --authorization OPEN_AUTHENTICATED_TIED_MPS_ENTROPY_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V0 \
  --review-receipt /absolute/path/tied_mps_source_review.json \
  --stream-lock /absolute/path/tied_mps_current_stream_lock.json \
  --gaussian-control-lock /absolute/path/tied_mps_gaussian_control_lock.json \
  --output /absolute/path/new-absent-output-directory
```

Do not run that command against any payload until the review manifest hash
matches the frozen package exactly.
