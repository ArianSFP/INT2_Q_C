# Posterior-centroid v0 audit-interface review

Date: 2026-09-02

## Existing audit finding

The existing directory
`uwfa_sc_posterior_centroid_v0_final_independent_audit_20260902` is explicitly
a source-only audit. Its executable docstring, README, report, machine result
and access attestation all state that it does not open a completed v9
publication, completed posterior result, or BF16 score source. Its numerical
checks use synthetic observations and wrappers. Therefore it cannot answer
whether a real `RESULT.json`, its literal wrappers and its original-source
scores agree.

This is a scope gap, not a defect in that source audit. A separate result
auditor is required.

## Reused interface

The new auditor retains exactly one producer-owned executable interface:
`result_bridge.py` at SHA-256
`112efcad5fd3fe9bccfea11af03bd9124a3789b6107c4486dd961656398e4d79`.
It is used only to load the result-bound v8 decoder closure and regenerate the
coordinate panel from literal `UWFCV8.bin`. The bridge receives this auditor's
independent implementations of:

- posterior handoff-root construction;
- pre-decision state replay; and
- occupancy construction.

The producer's `diagnostic.py` and `posterior_core.py` are not imported. All
fitting, permutation, prediction, wrapper/head parsing, original-domain
scoring, rate-distortion arithmetic, read projection and terminal decisions
are implemented in `audit_core.py` and `result_auditor.py`.

## Required predecessor authority

Opening seven v9 files is not enough. The audit requires a successful
externally hashed receipt from
`uwfa_sc_v9_qwen_primary_result_audit_v0` whose publication-member map exactly
equals the independently pinned v9 directory. This prevents the posterior
audit from treating unaudited v9 producer JSON as predecessor authority.

The v9 receipt remains nonpromoting and its documented limitations continue
to apply. This auditor uses its result only as a predecessor integrity fact.

## Required posterior authority

The posterior publication has two allowed exact member sets:

- hard-kill form: nine fold wrappers, `RESULT.json`, `COMPLETE.json`;
- cross-fit-pass form: the same plus `FINAL_STATE_AWARE.cagepst1`.

All names, byte counts and hashes must be recorded by an external authority.
The final member's presence is then checked against the independently replayed
cross-fit decision; external pins cannot force either branch.

## Score aperture and deterministic replay

The auditor authenticates the source manifest before loading NumPy. Each
matrix is then opened through a no-follow path, checked against its exact BF16
byte/hash record and scored in a whole-owner-component replay. Across the
three folds this covers every Gate/Up/Down matrix. The final full-panel fit, if
present, rereads and reauthenticates the source.

The audit repeats all eight ridge exponents in both inner directions for each
of three laws and three outer components. The emitted head is accepted only
if independent refitting, binary16 conversion, packet hashing and CRC produce
the exact literal head bytes.

## Read API limit

The v8 codec exposes a descriptor-backed routed decoder, so the audit can
reexecute the real inner expert read trace. Posterior v0 exposes no API that
accumulates the decoded UWFA occupancies and applies the serialized posterior
head during that routed call. Its wrapper instrument reads/parses the suffix
but leaves posterior application offline.

Accordingly the auditor can exactly recompute storage bytes and the declared
projection:

```text
actual v8 inner routed ranges + one full 4096-byte suffix request
```

It cannot turn that projection into an inference result. A future interface
must expose one routed call that returns the posterior-adjusted selected
expert, proves no compressed second pass, and matches the offline FP64
reconstruction digest.

## Frozen external hashes

| Object | SHA-256 |
|---|---|
| posterior producer manifest | `0ef30253d4d31504fbd8f88b8203cf35bce6c14952e570aace44b7bc089cb713` |
| posterior producer source root | `ea3ad9cf9b723cdf7501eeff004bd7f2821af4d37ff186b72f2972482a05e11c` |
| result bridge | `112efcad5fd3fe9bccfea11af03bd9124a3789b6107c4486dd961656398e4d79` |
| v9 result-auditor manifest | `885f41e27c439c808e2118de52184feaec58efe9f14bbc0e02a377e3b189f5ee` |
| v8 source manifest | `a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6` |
| STRATA common | `3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1` |
| frozen auditor | `85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e` |

Run-specific publication, v9-audit-receipt and BF16-manifest hashes are
deliberately absent and must be supplied through a separately hashed external
pin file.
