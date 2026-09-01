# Independent source audit: label-copula census stage 0 v1

Decision: **BLOCK_SOURCE_PREFLIGHT_READINESS**.

The exact producer manifest
`81fc371df5db5f654815d9fe0c673c34cb403d959b313425ae91c09c37fc5bd7`
and all seven listed member hashes were verified before producer import, both on
the local workspace and independently on the supplied RunPod.  No Qwen/model,
checkpoint, current-codec, or externally supplied Gaussian-control payload was
opened.  The producer bytes were not modified.

This is a source-level block.  It says nothing about whether the proposed
label-copula structure exists in Qwen weights.

## What passed

- The producer verifier passes, and all 23 producer source-free hostile tests
  pass on RunPod.
- An independent exhaustive probe covered all 240 nonlocal cells and all eight
  factorized resets.  Every Q0.16 row was valid, every model packet
  deserialized exactly, and every synthetic frame round-tripped from that
  deserialized model.
- The canonical `Gate[j,k], Up[j,k], Down[k,j]` traversal, Lloyd-4 threshold,
  ordered Gray map, role/plane schedule, arbitrary valid expert shapes, exact
  integer state transitions, and rectangular reusable expert-slot universe are
  implemented coherently.
- The outer test is held out by whole layer; inner validation is held out by
  reusable whole expert slot.  The paired bootstrap resamples whole test layers
  and its shared-byte allocation closes to the exact physical byte delta.
- A deliberately unequal five-frame test independently reproduced the frame
  offsets, 64-byte storage alignment, 4096-byte page unions, final container
  rounding, and denominator sum: `73728.0 == 73728` bytes.
- The source-free parity witness succeeds and remains explicitly non-scientific.
- The writer API creates completion last and rejects later writer-method calls;
  symlinked leaf and ancestor fixtures are rejected.  The actual local and
  RunPod source ancestor chains were also checked and were not reparse points or
  symlinks.

These passing primitive checks do not cure the integrated claim-path defects
below.

## Hard blocker 1: source search is not frozen to all 240 cells

`evaluate_raw_source_panel(raw_panel, bank=None)` exposes a caller-controlled
`bank` and forwards it to `evaluate_nested`.  There is no requirement that the
scientific source path use `candidate_bank()`.

The RunPod counterexample supplied a synthetic rectangular 10-layer by 3-slot
raw SwiGLU panel and a tuple containing only `Candidate("suffix", 2, 32)`.  The
function returned a normal `label-copula-census-result-v1` record with
`nonlocal_candidate_cells == 1` and one selection row.  The frozen requirement
is 240.

The control orchestrator does hard-code the full 240-cell bank and first
replays the source under that bank, so it would reject this result later.  That
does not make the source surface symmetric: it can already emit a scored source
result and absolute-survival field under a different search.

## Hard blocker 2: scored decode does not consume serialized model bytes

The low-level serializer and deserializer are correct.  The integration is not:

- `encode_panel` calls `len(model.serialize())` only for the byte ledger;
- `evaluate_nested` then calls `decode_stream` with the original in-memory
  `QuantizedModel`.

The RunPod counterexample temporarily replaced `QuantizedModel.serialize()`
with an all-zero packet of the correct charged length.  The packet was invalid
and `QuantizedModel.deserialize()` rejected it, yet `evaluate_nested` still
returned a scored result and all reported roundtrips succeeded.  Thus the score
does not prove that transmitted model bytes reproduce the labels.

There is also no literal container writer/parser.  The storage math is exact,
but headers, directory entries and frame records are accounted sizes rather
than bytes independently parsed by the decoder.

## Hard blocker 3: producer code executes before authentication

Both entrypoints dynamically execute same-directory producer code before
establishing their trust decisions:

- `stage0_census.py` executes `label_copula_common.py` at module import, before
  `main()` and the wrong-token branch;
- `verify_source.py` executes the same common module before `verify_manifest()`.

In a temporary clone, the audit added a marker write to the cloned common
module and invoked `stage0_census.py` with `WRONG` authorization.  The process
returned the expected code 2 and created no output directory—but the marker was
already written.  Manifest or review rejection after import cannot undo that
execution.

The `_review` check also accepts unsigned caller-authored JSON containing
publicly computable hashes.  A source-free RunPod counterexample confirmed it.
That is consistent with the package's explicit statement that it has no
payload or claim authority, but it means an independently pinned bootstrap
outside the producer package is a mandatory trust root.  This package cannot
self-authorize even a standalone preflight.

## Controls and statistical interpretation

The frozen high-level control orchestrator is ordered correctly: it replays the
source under the complete bank, checks the absolute source lower-bound gate,
then independently generates and runs every seed through all 240 nonlocal and
eight factorized cells.  Its provenance binds source tensor values, geometry,
binary64 block moments, generated tensor, generated labels, bank and complete
result.

The raw control-builder helper itself has no source-result or gate argument.
That is a library-misuse surface rather than a separate high-level runner
failure, because the scientific orchestrator gates it.  Since no payload runner
has been frozen, a later external runner would still need to exclude direct
helper use.

The controls report only a point estimate of source-specific excess; there is
no source-minus-controls confidence bound or veto.  This matches the explicit
design lock, which makes the absolute source lower bound primary and merely
forbids controls from creating a pass.  It is therefore not an additional
source-contract blocker.  It does mean the output alone cannot support a strong
claim that a survivor is Qwen-specific rather than equally present in matched
Gaussian controls.

The five-cluster minimum and layer-independence assumption likewise make the
bootstrap a prespecified screening interval, not a distribution-free 95%
guarantee over correlated layers from one checkpoint.

## Audit boundary

No payload launch, control launch, or Qwen claim is authorized.  A PASS on
primitive arithmetic and ledger tests cannot be promoted while any of the
three hard blockers remains in the sealed source.
