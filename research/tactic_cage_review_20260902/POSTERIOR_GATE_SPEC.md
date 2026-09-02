# CAGE posterior-MMSE gate

Date: 2026-09-02

## Claim boundary

This is the earliest finite discovery experiment for non-local posterior
reconstruction.  It is not TACTIC-DH384 and is not yet a universal codec
claim.  Six deterministic Qwen experts do not identify an unrestricted
`E[X|M]`; the experiment estimates a frozen restricted head under a declared
panel distribution.  Cross-fitting can establish a real held-out opportunity,
but deployment still requires one head trained on a disjoint checkpoint or
family, serialized in one literal packet and tested on another family.

## Literal decoder information

The finite branch requires an actual source-surviving `UWFCV8` object.  Bind
the following handoff fields before any posterior fit:

```text
literal_container_sha256
source_artifact_sha256
source_score_binding_sha256
source_full_geometry_sha256
source_structural_geometry_sha256
extraction_program_sha256
universal_decoder_sha256
universal_adapter_sha256
pipeline_sha256
source_snapshot_root_sha256
source_preflight_receipt_sha256
full_reconstruction_f64_sha256
semantic_packet_sha256
immutable_context_state_sha256
serialized_model_sha256
directory_sha256
decoded_sc_decision_triplet_commitment_sha256
```

Condition only on the causally decoded `UWFCV8` message `M`:

- 4-KiB header, semantic packet and immutable STRATA extension;
- serialized Q0.16 UWFA model;
- directory, owner regions, frame headers and arithmetic payloads;
- termination, padding and alignment;
- the proposed finite posterior-head packet;
- public shape, Gate/Up/Down role and fixed decoder constants.

The evaluator must re-decode the literal container and reproduce every
selected-bit, level and base-frequency-u16 commitment.  Extracted scratch
arrays are not an input.  A routed expert may not use model/checkpoint,
layer/expert identity, unread expert bytes, an external checkpoint, or original
weights as decoder side information.

Original BF16 values are separately authenticated score targets.  The panel is
the 18 matrices from:

```text
(L5/E18, L12/E7, L18/E20, L28/E83, L36/E76, L45/E41)
  x (Gate, Up, Down)
```

Each source is 3,145,728 bytes.  Bind the canonical source-record set to:

```text
768573dffbb7605a0993a0fd4485e4eb5fc5201529797a89d63d3c9fb18b51d6
```

The current predecessor input and its decoded FP64 reconstruction are:

```text
artifact  4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b
decoded   af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0
```

Score original matrix order, including Down as stored, after the exact inverse
RHT/XKLT.

## Leakage-proof split

Use the exact stream-owner connected components formed by the shared tails:

```text
C0 = {L5/E18,   L12/E7}   via block 12
C1 = {L18/E20,  L28/E83}  via block 13
C2 = {L36/E76,  L45/E41}  via block 14
```

For outer fold `k`, hold out all six role matrices in `Ck`.  Use the other two
components for development.  Select the head with both inner directions (fit
one component, validate on the other, then swap), including normalization,
binning, regularization and stopping.  Refit on both development components,
serialize and hash the head, and only then allow the evaluator to open held-out
BF16 targets.  Three connected components do not justify an iid confidence
interval.

Each outer fold emits a separate literal candidate.  Pooled fold distortion is
a discovery diagnostic, not one deployable packet.

## Smallest non-local heads

The current UWFA is unifilar, so ordinary forward-backward smoothing adds no
state information.  Reuse the winning deterministic pre-decision state trace.
For coefficient `i`, define the canonical decoded index
`k_i=previous_i in [0,63]` after the six SC levels and
`q(k_i)=0.25*(k_i-31)` before the transmitted block scale and inverse RHT.
For block `b`, SC level `l` and state `s`, define `p[b,l,s]` as the count of
selected decisions whose pre-decision UWFA state is `s`, divided by the exact
selected-symbol count at level `l`.  A zero-count level has the all-zero
occupancy vector.  Test, in that normalized pre-inverse-RHT domain:

```text
state-affine:
mu/scale = q_k + a + c*q_k
           + sum_(l,s) p[b,l,s] * (u[l,s] + v[l,s]*q_k)

state-centroid:
state-affine + correction[k]
```

At `S=64`, the maximum counts are 770 and 834 parameters.  FP16 payloads are
1,540 and 1,668 bytes before a fixed header/checksum and fit within one 4-KiB
page.  Reconstruct through the exact inverse transforms and score in FP64.

Before implementation this head must freeze a canonical ABI.  At minimum:

- `k` order is the literal `previous` order above and the 64-entry correction
  is indexed by ascending `k=0..63`;
- occupancy uses pre-decision states and the exact per-level denominators
  above;
- center nonempty occupancies by `p-1/S`, use a strictly positive frozen ridge
  penalty, and specify a deterministic solver, regularization grid and ordinal
  tie break so the intercept/state gauge has one result;
- serialize parameters as little-endian IEEE binary16 in one frozen order,
  with a version, dimensions, payload length, SHA-256 and CRC;
- promote binary16 exactly to FP64, evaluate in a specified left-to-right loop
  without contraction/FMA, then apply scale and the already frozen inverse
  transforms;
- independently decode the packet, reconstruct, and byte-reencode it.

Until these choices and hostile tests are frozen, the one-page count is a
plausible allocation rather than an independently decodable codec result.

The decisive ablations are:

1. identity lattice reconstruction;
2. local-only affine/centroid;
3. state-aware head;
4. independently permute the winner's `S` state-coordinate labels for every
   `(block, level)`, preserving that cell's occupancy multiset while destroying
   consistent state identity across blocks.

A global renaming of state labels is forbidden as an ablation: after refitting
it is isomorphic to the original model and destroys no dependence.  Likewise,
permuting decision positions while preserving the same cell counts is a no-op
because this head consumes only occupancies.  Derive each cell permutation by
sorting domain-separated SHA-256 values of
`(fixed_ablation_seed, block_ordinal, level, state_ordinal)`; freeze the seed,
byte encoding and digest-order tie rule before source access, with
`state_ordinal in [0,S)`.

Only (3) beating (2) and (4) on every outer component is evidence beyond the
already adverse RAVEL/local-correction family.

The finite branch above requires a literal Qwen UWFA survivor.  If the
standalone label-rate gate fails, a separately frozen
`NONPROMOTING_STATE_CENTROID_ORACLE` may run once because label predictability
is not a converse for a continuous within-cell mean.  It may reuse only the
hard-killed literal container and serialized model: it must independently
decode them and regenerate the state trace, never consume extracted state
scratch.  It emits no finite-codec or control claim and is hard-killed unless
`G_state,k >= 0.03 bpw` on every outer component under a same-rate
source-leaking upper-bound calculation.  Do not call that oracle the finite
survivor experiment.

## One physical accounting equation

Frozen `UWFCV8` rejects trailing bytes.  The first experiment must therefore
use a separately versioned outer `CAGEPST1` wrapper containing the unchanged
inner `UWFCV8` bytes plus the posterior head; alternatively, a future
independently specified integrated version may replace both.  The outer header
must bind its version/length, inner length/SHA-256, head length/SHA-256,
alignment/padding, the exact handoff root above and a header CRC.  No byte may
follow the declared logical end.  `B_k` is the complete outer object, including
wrapper/header/alignment delta--not merely the inner object plus an assumed
4-KiB model page.

For literal outer-fold packet `k`, score only the six held-out matrices in
`Ck`.  Never include the 12 matrices used to fit its head in the primary
distortion:

```text
B_k = len(canonical candidate container k)
R_k = 8*B_k / 28,311,552
D_k = heldout_SSE_k / heldout_energy_k
D0_k = heldout_baseline_SSE_k / heldout_energy_k
F_k = D_k * 2^(2R_k)
Delta_s_k = (2.5 - R_k) - 0.5*log2(D_k / D0_k)
```

Training-source and full-18 scores are diagnostics only.  A pooled cross-fit
distortion may sum the three held-out SSE/energy pairs, but the three heads and
containers differ; it is not one literal packet and has no final physical
`R` or `F`.

Charge header, UWFA model, posterior model, directory, frames, checksums,
arithmetic termination, padding and alignment through `len(container)`.  Never
add an entropy estimate to a separately reconstructed MSE.

One new 4-KiB page costs `0.00115740740740741` bpw on this panel and adds
`0.00277777777777778x` to the old equal-share routed-read denominator before
the exact owner ledger is recomputed.  At the weakest aligned UWFA pass of
8,302,592 bytes, an unreclaimed page raises `R` from `2.3460648148` to
`2.3472222222`; unchanged MSE gives `F=0.8001244315`, so the head must reduce
MSE by at least `0.0155515%` or occupy already charged padding.

Report unique pages, requested bytes with repetition, overlapping re-requests,
coalesced interval union, scratch/HBM traffic and the maximum owner-aware
ratio.  Full-sequence posterior work may revisit already resident symbols but
must not reread the compressed expert from storage.

## Controls and hard kills

Only after a source head survives, repeat the complete nested fit, pack,
decode and score on the eight matched-Gaussian seeds:

```text
10619863, 10619881, 10619909, 10619927,
10619953, 10619971, 10619999, 10620017
```

Each control needs its own authenticated 18 BF16 sources, exact moment replay,
source-derived metadata, literal UWFA container and posterior fit.  Also run
seeded within-2,048-group permutations, common to aligned Up/Down coordinates,
to preserve marginal values, group energies and Up/Down covariance while
destroying serial state structure.

Define the state-specific contribution on every outer fold as:

```text
G_state,k = Delta_s_k(state-aware)
            - max(Delta_s_k(local-only), Delta_s_k(state-permuted))
```

Hard-kill this frozen head family if any source/decision binding fails; any
outer component has `Delta_s_k <= 0` or `G_state,k <= 0`; FP16 serialization
reverses the FP64 gain; `R_k` leaves `[2.15,2.5]`; `F_k>0.8`; maximum cold read
is `>=2x`; or Qwen's `G_state,k` does not exceed the strongest matched or
structure-destroyed control.

## Coarse-programmed geometry and the 384-bit ceiling

For `G=Phi(C,shape,role,theta)`:

```text
H(G | C,shape,role,theta) = 0
I(X;G | C,shape,role,theta) = 0
```

The graph costs no descriptor only when it is deterministic from already
transmitted information; it contributes no new residual information.  Forcing
`h(C)` into one of `M` graph classes sacrifices roughly `log2(M)` bits of
coarse freedom or causes coarse distortion.  Only a joint search over legal
`(C,B)` can demonstrate a gain.

Per 4,096-weight block, 384 fine bits are 48 bytes or `0.09375` bpw, select at
most `2^384` reconstructions for a fixed coarse word, and satisfy
`I(X;B|C)<=384` bits.  There are 1,152 blocks per expert and 6,912 across the
six-expert panel, so the fine field is 55,296 bytes/expert and 331,776 bytes
total.  The planning identity is:

```text
8,487,936 coarse + 331,776 fine + 27,648 metadata = 8,847,360 bytes
```

A 4-KiB model consumes 32,768 fine bits, equal to 85 1/3 whole fine-block
budgets or 4.74074 bits/block when amortized uniformly.  The current 2.5-bpw
layout has no slack; model bytes must displace coarse/fine capacity.
