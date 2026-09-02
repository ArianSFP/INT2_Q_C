# UWFA-SC v9 primary result-audit requirements

Date: 2026-09-02  
Status: implementation requirements only; no result or execution authority.

## Purpose

The v9 primary runner is deliberately nonpromoting. A completed output cannot
be used even as a Qwen primary-source result until a separately frozen audit
replays its evidence. A primary survivor additionally requires deferred
shuffles, matched-Gaussian controls and a new audit before any compression
claim.

## Exact external pins

The audit invocation must receive independently recorded hashes for:

- `COMPLETE.json`, `RESULT.json`, `BOUND_BASELINE_SCORE.json`,
  `SOURCE_PREFLIGHT.json`, `DECODER_BUNDLE.json`, `UWFCV8.bin`, and
  `IDENTITY_FRAMING.bin`;
- the complete v9 source manifest/root and `primary_gate.py`;
- the complete sealed-v8 source manifest/root and every executable member;
- the pinned exploratory support wrapper;
- STRATA common source and the frozen external auditor;
- the Qwen artifact bytes/hash and the original-source identity closure.

No digest copied only from the publication being audited is external
authority.

## Publication and claim-boundary checks

Require the exact seven data members plus completion seal, canonical JSON,
finite binary64 values, completion-last member order, exact byte counts and no
extra files. Verify every nonpromotion counter:

```text
positive_claim_authority = false
positive_claim_even_if_all_primary_gates_pass = false
controls_run = false
shuffles_run = false
coordinate_disjoint_diagnostic_run = false
```

The status must be recomputed from the sealed-v8 primary decision predicate;
it may not be trusted from text in `RESULT.json`.

## Independent scientific replay

From the authenticated decoded decision streams, independently reconstruct:

1. the exact stream-owner bipartite graph and three disjoint dependence
   components;
2. every train/validation/development/test index set;
3. the ordered 150-cell candidate bank;
4. exact Q0.16 count fitting, Jeffreys half-counts, state resets and causal
   lengths;
5. inner validation choices, held-out component scores, vote-selected final
   topology and full-panel refit;
6. serialized sparse model bytes and page-aligned literal rate deltas;
7. pooled held-out saving, every component saving and the final primary
   pass/kill decision.

The replay must prove that the model is fitted only on each declared
development side and that a duplicated owner label cannot create another
independent observation.

## Independent physical replay

Parse `UWFCV8.bin` using independent code, causally decode every selected SC
decision, reproduce the exact selected-decision triplet commitments, rebuild
the complete FP64 reconstruction, canonically re-encode the object, and
recompute:

- actual bytes and rational bpw;
- identical-reconstruction MSE and `F` from the externally pinned baseline
  score;
- model/directory/padding cost;
- standalone and routed page reads;
- maximum owner-aware cold-read amplification.

`IDENTITY_FRAMING.bin` is only a literal byte-cost counterfactual unless its
payload/model semantics are independently decodable. The audit must not infer
a semantic reconstruction for an intentionally mismatched identity frame.

## GPU and runtime evidence

Recompute the exact admitted update count `38,621,316,130`, authenticate the
source-free all-150 and representative receipts, bind the CUDA/NVML/device
identity, and check telemetry conservation. A CPU result audit need not rerun
the expensive GPU selection if it independently replays the selected model
and exhaustively validates all recorded candidate/fold commitments; any
unreplayed GPU fact must remain explicitly bounded rather than promoted.

## Filesystem race rule

Retain file descriptors and verify device/inode/type/link identities and every
name-to-inode binding. Do not include ancestor-directory mtime/ctime in the
identity of broad shared parents such as `/workspace`: creation of an unrelated
sibling otherwise causes a false integrity failure. Full metadata stability
remains appropriate for the actual source-package and publication directories.
Alternatively, run the unchanged broad-ancestor auditor only on a quiescent
parent.

## Terminal interpretation

- Primary hard kill: valid evidence against the declared sparse-unifilar WFA
  cell only.
- Primary survivor: nonpromoting; authorize only separately frozen shuffles,
  controls and posterior diagnostics.
- Any audit mismatch: no scientific result until resolved by a fresh,
  independently pinned replay.

No v9 outcome is a universal SwiGLU-MoE result by itself.
