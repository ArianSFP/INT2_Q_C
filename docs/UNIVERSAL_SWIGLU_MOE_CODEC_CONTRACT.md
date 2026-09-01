# Universal SwiGLU-MoE codec contract

Date frozen: 2026-09-01
Status: normative research scope for all post-v9 candidates

## Purpose

The target is one compression architecture that can be applied to arbitrary
SwiGLU mixture-of-experts checkpoints.  Qwen matrices are an evaluation panel,
not privileged decoder side information.  A result that depends on a Qwen
model name, training lineage, public ancestor or model-specific external
network is outside scope even if its numerical rate and MSE are favorable.

Universality describes the codec rule, not a promise that every checkpoint
will attain identical distortion.  The encoder may adapt the universal rule
to the current tensors, as ordinary PTQ does, provided every adapted value is
physically serialized and independently decoded.

## Eligible decoder inputs

A decoder may use only:

1. the versioned universal codec specification;
2. integer tensor shapes and the semantic roles Gate, Up and Down;
3. the compressed bytes required for the selected routed expert;
4. explicitly transmitted layer-shared or model-shared bytes charged to the
   physical container and the cold page union; and
5. state causally reconstructed from earlier bytes in that same decode.

Examples of eligible adaptation include a transmitted codebook, entropy table,
predictor coefficient, neuron permutation or transform selector.  The encoder
may fit it to a checkpoint, but the decoder must obtain it from charged bytes.

## Forbidden side information

The decoder may not depend on:

- model, vendor, checkpoint, layer or expert identity as a hidden key;
- checkpoint ancestry, initialization seed or training provenance;
- an external base/reference checkpoint or separately stored expert;
- router weights, router activations, calibration activations or token data;
- a pretrained model-specific hypernetwork, dictionary or lookup service;
- source weights, untransmitted residual statistics or encoder-only random
  search state; or
- an amortization argument over hypothetical experts/models not present in the
  byte-derived artifact being scored.

These values may be used by a diagnostic oracle only when the result is
explicitly labeled impossible/source-leaking and cannot be promoted as a
codec.

## Shape and role portability

The universal format operates on a canonical expert triplet:

```text
Gate: [intermediate, hidden]
Up:   [intermediate, hidden]
Down: [hidden, intermediate]
```

Down may be viewed transposed while forming joint neuron vectors.  A codec may
tile or pad arbitrary positive dimensions, but shape-derived padding and tail
records are physical bytes unless the format proves they are implicit.  A
fixed `768 x 2048` Qwen evaluation geometry is not permission to bake those
dimensions into the universal algorithm.

Any exact SwiGLU symmetry used by the codec must be restored by the emitted
decoder for raw-source-MSE scoring.  Functional equivalence alone does not
change the frozen raw-MSE objective.

## Rate and routed-read requirements

For the emitted artifact:

```text
2.15 <= R <= 2.5 physical bits/weight
F = relative_MSE * 2^(2R) <= 0.8
maximum cold/page routed-expert read amplification < 2x
```

All payloads, tables, bases, labels, permutations, headers, checksums,
alignment, padding and reserved capacity are charged.  Shared bytes are
charged once to physical storage and again whenever their pages belong to a
cold expert read.  Warm-cache assumptions may be reported separately but are
not the gate.

The preferred layout is one sequential expert-local frame plus a small global
packet, keeping cold reads near `1x`.  A read amplification between `1x` and
`2x` requires a material MSE improvement.

## Scientific promotion ladder

Every new family proceeds in this order:

1. freeze the model-agnostic algorithm, all dimensions, seeds, fits and rate
   ledgers before opening the evaluation payload;
2. run a source-leaking or otherwise dominant oracle and stop when its upper
   opportunity is certainly below the required gain;
3. use matched Gaussian and structure-destroying controls when selection or
   dimensional null capture can imitate a source-specific signal;
4. test held-out layers/experts and, before a universality claim, at least one
   disjoint SwiGLU-MoE family or a source-free portability fixture with
   different legal dimensions;
5. build a finite packet only after the oracle survives;
6. independently decode and score the same residual under one joint rate
   ledger; and
7. seal source, result, packet, read ledger and hostile verifier tests.

Separate oracle gains are never added.  A composite must reconstruct a literal
common residual and perform one allocation, serialization and cold-read
calculation.

## Current eligible frontier

- RAVEL-6144 uses a transmitted shared table keyed only by decoded
  reconstruction and role/shape semantics.  Its corrected v1 cell is universal
  but numerically killed.
- TACTIC-DH384 derives a conditional dyadic frame from same-block coarse
  symbols and a physically charged universal selector packet.  Its source-only
  closure passes eleven hostile tests, its synthetic CPU/CuPy projection agrees,
  and it has an exact `73/72` cold-read ledger.  It remains unexecuted on model
  payload pending a real lower-rate coarse artifact and independent source
  audit.
- Cyclostationary/polyphase prediction, Hankel/annihilating-filter screens and
  decoder-synchronised lifting remain eligible when their coefficients are
  charged and their layout rules use shapes rather than model identity.
- Source-adaptive neuron seriation is eligible in principle, but an older
  stronger free-predecessor oracle must be treated as prior containment
  evidence before another payload run.

Checkpoint-reference/delta coding and initialization-provenance recovery are
outside this program because they are not model-agnostic decoder architectures.
