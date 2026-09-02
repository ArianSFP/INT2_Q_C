# Information-accounting corrections

Date: 2026-09-02

An independent read-only red team found three accounting issues in the initial
TACTIC-CAGE review.  The corrected specifications in this directory supersede
the earlier wording.

## Charged source adaptation is universal

A universal codec fixes the decoder algorithm and packet language.  Its
encoder may fit parameters to the source being compressed when those
parameters are serialized in the packet and fully charged.  The forbidden
mechanisms are uncharged Qwen constants, model/checkpoint/layer identity
lookups, an external reference checkpoint, or any other decoder input outside
the authenticated packet and public shape/role contract.

## Cross-fit rate and distortion need the same ownership scope

The first posterior specification used the rate of a full 18-matrix packet
with distortion from only the six held-out matrices.  If private rates vary,
training-component bytes can then improve or worsen the held-out score even
though those components contribute no held-out distortion.

Every outer fold must instead use one of:

1. a literal heldout-only evaluation packet that contains the held-out source
   and charges the complete shared model; or
2. exact heldout-private bytes plus a predeclared weight-proportional share of
   global bytes, used only as a cross-fit selection ledger.

The baseline uses the same framing and ownership rule.  The allocated ledger
cannot be called a final physical packet.  After cross-fit selection, the
chosen family must emit one all-component packet whose literal full rate and
full distortion determine promotion.

## Deterministic decoder constants need not be repeated per model

The self-describing composite envelope reserves 16,384 bytes for the fixed
selector, 2,048 bytes for an all-zero QC region and 2,048 bytes for seed
fixtures.  In a deployment profile whose decoder algorithm and version are
already fixed, all three are reproducible constants.  They may be omitted and
bound by an algorithm ID in the remaining schema page; they do not need to be
charged once per compressed model.

Removing those 20,480 bytes gives, for the current six-expert panel:

```text
container bytes                 8,826,880
rate                            2.49421296296296 bpw
same-MSE F multiplier           0.99200955785219
favourable STRATA transfer F    0.980967853512842
one-schema-page read bytes      1,474,560
exact compressed owner share    1,471,146.66666667 bytes
byte read ratio                 1.00232018561485x
```

The 20,480-byte removal and `F` multiplier are exact layout arithmetic.  The
absolute `F` above transfers the audited STRATA baseline MSE into an unexecuted
lean envelope and is therefore planning-only, not a finite CAGE result.  Even
that favourable transfer is only about a 0.8% reduction--not the required 20%.
The same byte budget could instead carry a charged source/posterior model while
preserving the old 2.5-bpw length.  A self-contained archival profile may
retain the literal constant packets, but must not be confused with the lean
fixed-decoder deployment profile.
