# UWFA-SC v2 integrated-container specification

Date: 2026-09-01
Status: source-independent normative design for the next audited producer.

## Purpose

The UWFA source model is useful only if it replaces the existing arithmetic
reservoir in one standalone STRATA container.  Extracted selected bits,
regenerated SC frequencies, decoded reconstruction arrays, the old container,
and an independent audit report are encoder inputs or evidence; none is free
decoder side information.

The emitted container must independently reconstruct the exact current
STRATA expert-affine reconstruction while deriving its physical rate and
routed-read cost from its own bytes.

## Decoder-visible source

The decoder receives only:

1. this versioned universal format and the frozen universal STRATA decoder;
2. the literal UWFA-SC v2 container;
3. integer tensor geometry and Gate/Up/Down semantics encoded in that
   container; and
4. state causally reconstructed from earlier container bytes.

It must not receive extracted `selected_bits`, `polar_level`, or
`regenerated_base_freq1` arrays.  Model, checkpoint, layer, expert and stream
identity are not probability keys.

## Integrated causal decode

The existing independent SC decoder already computes an original conditional
frequency and invokes an arithmetic-like `decode(freq1)` operation for every
selected decision.  UWFA-SC replaces that operation with a deterministic
adapter:

```text
set_level(polar_level)

decode(original_freq1):
    if position mod reset_length == 0:
        state = 0
        position_in_reset = 0
    context = public_context(
        polar_level,
        prior_bin(original_freq1),
        position_in_reset
    )
    uwfa_freq1 = transmitted_model[context,state]
    y = new_arithmetic_decoder.decode(uwfa_freq1)
    state = frozen_transition(state, y, context, position_in_reset)
    position_in_reset += 1
    return y
```

Thus the base frequency used as UWFA context is regenerated inside the
original polar recursion from already decoded state.  It is never serialized
as a per-symbol array.  After all levels, inverse signed RHT, group restoration
and inverse XKLT are unchanged.

The independent decoder must also re-encode every new arithmetic payload from
the decoded decisions and regenerated contexts and require literal byte and
logical-length equality.

## Physical layout

All integer fields are little-endian.  Every offset and length is checked for
overflow, canonical ordering, non-overlap and containment.

```text
4096-byte fixed global header
literal inherited STRATA metadata packet
page-rounded serialized UWFA model
page-rounded 15-record directory
15 block payload frames, each 64-byte aligned
zero final padding only when required by the 2.15-bpw floor or final page
```

The inherited metadata packet contains the literal information necessary to
reconstruct the current format, including its 128-byte semantic header,
144-byte route table, 5,184-byte three-bit labels, fifteen profile identifiers
and fifteen binary16 decoder scales.  It does not contain any byte of the old
arithmetic reservoir.

The global header binds at least:

- magic, major/minor version and header CRC;
- weight, expert, block and role counts;
- complete container byte count;
- actual inherited-metadata, model, directory and frame ranges;
- the selected topology/state/reset descriptor;
- source baseline container hash and byte count;
- baseline plan-lock and independent-audit hashes;
- independently audited baseline relative MSE and source-energy convention;
- universal decoder/source-manifest/audit-bootstrap hashes; and
- a root hash over all non-padding bytes.

Every block-directory record contains:

- canonical block ordinal and `log2(N)`;
- owner-expert bit mask;
- profile identifier and exact binary16 decoder scale bits;
- payload offset, physical byte length and logical bit length;
- payload SHA-256; and
- all reserved bytes zero.

Payload frames are in canonical block order.  A frame may be shared by the two
experts owning one tail block; no frame is duplicated merely to improve the
read ledger.

## Literal parser and independent reconstruction

Promotion requires a parser/decoder implementation independent of the
producer.  Starting from only the emitted container and universal source, it
must:

1. authenticate the complete layout and all bound hashes;
2. deserialize the transmitted model, rejecting noncanonical rows;
3. derive the original SC/RHT seeds from literal inherited metadata;
4. decode and canonically re-encode all fifteen new payloads;
5. restore every source group exactly once;
6. reproduce the exact decoded reconstruction hash when the arithmetic ABI is
   deterministic, and always recompute original-BF16 FP64 SSE/energy;
7. derive rate, `F` and cold reads from literal output bytes; and
8. reject any mismatch without consulting producer results or extracted
   arrays.

The current baseline constants are evidence inputs, not universal constants.
For the bound Qwen evaluation they must be authenticated against the held
baseline artifact and its independent audit.  The output score is

```text
R_actual = 8 * container_bytes / source_weights
F_actual = independently_scored_relative_MSE * 2^(2*R_actual).
```

The standalone pass requires `2.15 <= R_actual <= 2.5` and
`F_actual <= 0.8`.  If the unpadded result is below 2.15 bpw, zero padding is
charged up to the smallest legal page-aligned container.  No padding may hide
an overflow above 2.5 bpw.

## Routed-read attribution

Cold reads are computed from the exact union of 4-KiB pages containing:

- the global header, inherited metadata, model and addressed directory page;
- the two private frames for the routed expert; and
- its one shared tail frame.

Let `G` be all global and final-global-padding bytes, and let frame `b` have
physical bytes `B_b` and owner set `O_b`.  Expert `e` is attributed

```text
storage_share(e) = G/E + sum_{b: e in O_b} B_b / |O_b|.
```

Its amplification is exact touched-page bytes divided by this attributable
storage share.  The maximum must be strictly below `2x`.  A warm-cache result
may be reported separately but cannot replace this gate.

## Scientific selection and controls

The finite topology/state/reset bank, split rule, seeds and tie rules are
frozen before opening selected decisions.  Whole-layer/expert holdout is
scientific transfer evidence; the final two-part packet may fit its serialized
probability table to the whole artifact because every adapted value is
physically charged.

Matched-Gaussian controls are generated only after the absolute source packet
and heldout gates pass.  Every control repeats the complete transform,
quantizer, extraction, 150-cell selection, fitting, packing and literal decode
pipeline.  Geometry, block moments, seeds, intermediate hashes and output
containers are bound.  Refitting only the source-selected winner is not a
symmetric control.

Controls can reject source specificity; they can never turn a source packet
with `F>0.8`, illegal rate, or cold reads `>=2x` into a pass.

## Root of trust and lifecycle

The producer cannot authenticate itself.  Launch uses a separately pinned,
independently audited bootstrap whose hash is fixed outside the producer
package.  That bootstrap:

1. rejects symlink leaves and every symlink ancestor;
2. opens all source and input files without following links;
3. retains descriptors and authenticates exact bytes before execution;
4. executes/imports only an immutable snapshot of those authenticated bytes;
5. creates an absent output and never resumes it;
6. verifies held-file identities after the run; and
7. writes `COMPLETE.json` exclusively last, after which every write API is
   disabled.

No public token or self-sealed JSON alone grants payload authority.

## Claim boundary

An audited source package proves only that this format and experiment are safe
to run.  A Qwen result exists only after one literal container passes
independent source-domain reconstruction, actual-rate, `F` and routed-read
verification.  A miss closes the frozen unifilar selected-SC-bit cell, not
arbitrary MPS, MERA, TTN, nonlinear-flow or raw-label-copula architectures.
