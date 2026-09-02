# epsilon-TCQ/WFA joint-label early gate v0

Date: 2026-09-02

Status: source-only, no-payload early gate. No Qwen/model weight, current
POLARIS/STRATA object, legal-candidate trace, matched-Gaussian control, CUDA
device, or network is opened by the sealed runner or isolated standard-library
tests. A separately invoked source-free smoke opens only the RunPod CUDA device
to test the lazy CuPy kernels; it receives no payload path.

## Hypothesis

The completed UWFA experiment entropy-coded an unchanged nearest-label stream.
This branch tests a different source-coding question: can the encoder choose a
nearby *legal* quantizer reconstruction whose causal unifilar trajectory is
both lower-distortion and cheaper to transmit?

For target values `x_t`, legal choices `a_t`, a frozen model `M`, and a literal
finite arithmetic packet `P`, the encoder minimizes

```text
sum_t (x_t - reconstruction(a_t, pre_state_t))^2
    + lambda * 8*len(P)
```

by an exact search on tiny streams and full-state beam Viterbi on production
streams. Every beam path carries the legal-codec state, unifilar state, exact
binary arithmetic-coder interval, pending underflow bits, and emitted prefix.
Paths merge only when all decoder-replayable state and the literal emitted
prefix agree. Final ranking uses the exact padded packet bytes, not cross
entropy or `-log p`.

The fixed comparison is the nearest legal label encoded through the same WFA,
packet grammar, topology/model bytes, centroid bytes, and framing.

The production beam batches the FP64 source-domain branch-error calculation
through CuPy. Causal legal-state evolution, WFA evolution and exact arithmetic
interval updates remain on the host so the literal decoder path is bitwise
replayable. The bounded Hankel initializer also uses a CuPy singular spectrum,
but may only rank the already frozen model bank.

## Executable versus planned work

V0 executes source-free adapter mechanics, exact/beam search, finite packet
pack/parse/decode/re-encode, complete model and centroid serialization,
owner-component fold construction, control gates, one-pass page accounting and
the CuPy kernels. It does **not** execute a Qwen pilot. Still planned are the
independently authenticated POLARIS/STRATA legal-transition adapter, real
current-codec ingestion, nested outer-fold driver, eight full matched-Gaussian
pipelines, and a measured inference container/kernel. None is inferred from
the synthetic fixtures.

## No free four-level relabelling

There are two incompatible interfaces:

1. `strata_sc_6bit_legal_replay` is the primary integration. A candidate is an
   actual coordinate-aligned STRATA/POLARIS lattice index `0..63`, together
   with the exact six causal SC decisions, public contexts, and next legal
   decoder state returned by an authenticated codec adapter. `index+-epsilon`
   is only a request to that adapter; it is not automatically legal. The
   decoded event sequence must reproduce the index and literal reconstruction.
2. `direct_int2_4level_new_codec` is a separate replacement codec with labels
   `0..3`, an explicit four-value reproduction table, two literal label bits,
   and its own fully charged coarse packet. It may never stand in for current
   STRATA/POLARIS indices or inherit their rate/reconstruction.

V0 deliberately ships no Qwen/POLARIS legal-transition adapter and therefore
cannot run a Qwen source by silently falling back to the direct four-level
mode. The bound runner returns a typed HOLD until a separately frozen,
authenticated adapter proves candidate legality and decoder replay against the
literal current codec.

## Frozen model bank and initialization

The topology bank is fixed before source access:

```text
suffix(S=4,L=32)       suffix(S=8,L=128)
xor_sketch(S=8,L=64)  modular_ones(S=8,L=64)
rolling_affine(S=8,L=128)
signed_saturating(S=8,L=64)
```

Epsilon is in `{1,2}`, lambda exponent in `{-16,-12,-8,-4}`, and production
beam width in `{32,128,512}`. Exact search is capped at 12 labels with at most
three choices each. A bounded Hankel/CSSR-style initializer may inspect prefix
and suffix words of length at most three and numerical rank at most 16. It can
initialize emission counts or rank the frozen cells; it cannot synthesize a
new topology, state count, reset, alphabet, context law, or selector.

Emission probabilities are exact Q0.16 frequencies with Jeffreys half-counts.
The complete topology, reset, state count, interface, context count and every
frequency are serialized and charged.

## Posterior centroids

Three decoders are nested under identical whole-owner-component folds:

* nominal/local-only: one binary16 residual centroid per legal label;
* state-aware: one binary16 centroid per `(pre-state,label)`;
* state-permuted: the same state-aware capacity after a frozen per-stream
  state permutation preserving occupancy counts.

All heads are fit only on development owner components, serialized, parsed,
applied on untouched components, and included in physical bytes. The
state-aware head must beat both local-only and state-permuted heads after
bytes. A final all-owner head is separate from cross-fit discovery packets.

## Finite packet and one-pass read

`ETCQWF01` contains a 256-byte authenticated header, complete model packet,
complete centroid packet, a 128-byte stream frame, exact arithmetic payload,
and zero byte-padding. Parse, causal decode, legal adapter replay, centroid
application, canonical re-encode, and literal byte equality are mandatory.

The MoE layout projection page-aligns one shared header/model/centroid/
directory prefix and one owner-local frame region. A routed expert reads the
shared prefix and its selected owner pages once. The ledger separates:

* external storage ranges and unique pages;
* host parse/state/beam scratch;
* accelerator HBM and D2H traffic.

No compressed expert page may be fetched twice. Promotion requires the worst
owner-local unique-page ratio to be strictly below `2x`; source-only synthetic
tests establish only accounting identities, not inference traffic.

## Whole-owner cross-fitting and controls

Streams are split by connected components of the stream-owner bipartite graph.
Any shared stream joins its owners. Every outer component is untouched during
topology/epsilon/lambda/beam selection, frequency fit and centroid fit. A fold
must emit one literal nearest-label packet and one literal joint-label packet.

Only a source survivor opens all eight matched-Gaussian controls. Each control
must be generated by the full authenticated PTQ pipeline, regenerate its own
legal-candidate adapter trace, and repeat nested selection, joint Viterbi,
packet pack/parse/decode/re-encode and original-source scoring. Permuting a
residual or reusing Qwen legal candidates is not a matched-Gaussian control.

## Hard gates

A branch is killed if any legal replay or literal packet check fails; any
outer component loses; the state-aware centroid does not beat both controls;
actual rate leaves `[2.15,2.5]`; `F=D*2^(2R)>0.8`; the worst routed read is
`>=2x`; model/topology/centroid/header/padding bytes are omitted; or the pooled
Qwen-minus-strongest-control gain is below `0.03 bpw`.

At the pending N18 coarse result `D0=0.036975150060595235`, a final
`D<=0.025` requires 32.3870222053737% coarse-SSE removal. Ordinary Gaussian
use of the remaining rate preserves `F`, so epsilon-TCQ must demonstrate
source-specific excess, not generic extra-rate improvement.

## Source-only checks

```bash
python -I -B research/epsilon_tcq_wfa_early_gate_v0/test_source_only.py
python -I -B research/epsilon_tcq_wfa_early_gate_v0/verify_source.py \
  --package research/epsilon_tcq_wfa_early_gate_v0
```

Tests exercise both non-aliasable interfaces, exact and beam search, exact
arithmetic packet decode/re-encode, model/centroid byte charging, state-aware
and permuted centroids, owner components, hostile tampering, and one-pass page
accounting. A separate post-freeze source-free CuPy smoke is required before
any payload adapter may be opened.
