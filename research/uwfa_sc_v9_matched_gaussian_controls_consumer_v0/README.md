# UWFA-SC v9 full-PTQ matched-Gaussian consumer/auditor v0

## Outcome

This source-only package closes the consumer ABI left open by the v8 control
phase and the v9 deferred-control design. It does not weaken v8 in place.

The runtime order is fail-closed:

1. validate an externally pinned independent primary-survivor authorization;
2. validate a separately audited run authorization and every runtime/source
   pin;
3. authenticate the root declaration and every byte in all eight full-PTQ
   control bundles, including every retained BF16 source chunk;
4. independently decode each literal control artifact, recompute its own full
   and structural geometry, validate its score, and replay its generator and
   moments;
5. only after all eight controls reach that state, construct the first CuPy
   backend and begin fitting;
6. for each executed control, fit and score the complete canonical 150-cell
   bank in every disjoint-component fold, retain all 150 cell receipts, select
   solely from that control's validation scores, pack and decode the literal
   final container, then apply the early-null rule;
7. stop immediately when a matched null's physical saving is greater than or
   equal to the source saving. A survivor requires all eight nulls to be
   strictly weaker.

The producer's source-dependent full and structural geometry are authenticated
per control. They are never required to equal the source model's geometry.
Only the pre-frozen universal contract is shared: six canonical slots, three
SwiGLU roles and shapes, 28,311,552 weights, and the fifteen logN/owner-slot
records. It contains no labels, profiles, symbol counts, contribution maps or
payload lengths.

## Why a new all-150 wrapper is needed

The sealed v8 `nested_holdout` really evaluates all 150 configurations, but its
result retains only the selected score. That is insufficient for an external
auditor to distinguish independent control selection from source-winner reuse.
`matched_controls_consumer.py` calls the unchanged v8 fit, exact-length and
literal-container scoring primitives and records, for every fold and cell:

- the canonical candidate identity and selector ordinal;
- exact charged validation bits;
- fitted-frequency and validation-length digests;
- a cell seal and a complete 150-row list seal.

Tie-breaking remains `(validation charged bits, selector ordinal)`. The source
winner is not accepted as an input.

## Input bindings

`consumer_contract.py` validates three independent layers:

- primary authorization: source gates, physical saving, reconstruction and
  source closures, plus externally supplied primary-auditor pins;
- run authorization: consumer and producer audit pins, exact control-root pin,
  v8 source snapshot/preflight/GPU identity, moment replayer, descriptor
  builder and source manifest closures;
- eight-control root: exact member set, member roots, all retained BF16 chunks,
  generator, moment contract, source panels, bindings, scores, geometries,
  plans, artifacts and per-control completion records.

The consumer does not enumerate or open a primary/Qwen payload. Its only
primary input is the independently audited authorization record.

## Output and claim boundary

Successful execution writes `INPUT_AUTHENTICATION.json` before any control
result, `CONTROL_00.json` onward for every executed null, `RESULT.json`, and
`COMPLETE.json` last. `INCOMPLETE` remains after any exception.

Terminal outcomes are:

- `HARD_KILL_MATCHED_GAUSSIAN_NOT_SPECIFIC`;
- `PASS_ALL_EIGHT_MATCHED_NULLS_NONPROMOTING_AWAITING_INDEPENDENT_RESULT_AUDIT`;
- a fail-closed resource, estimability or incomplete-sequence block.

Every outcome has `positive_claim_authority=false`. A fresh independent result
audit remains mandatory.

## Source-only verification

```text
python -I -B research/uwfa_sc_v9_matched_gaussian_controls_consumer_v0/test_source_only.py
python -I -B research/uwfa_sc_v9_matched_gaussian_controls_consumer_v0/verify_source.py \
  --package /absolute/path/to/research/uwfa_sc_v9_matched_gaussian_controls_consumer_v0
```

Direct invocation is inert and exits 3. This package was built without opening
RunPod, CUDA, Qwen/model payloads, a live primary result, or any generated
control payload.
