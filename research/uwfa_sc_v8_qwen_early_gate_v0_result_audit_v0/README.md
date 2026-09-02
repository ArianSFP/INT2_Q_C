# UWFA-SC v8 Qwen early-gate result audit v0

This is a separate, source-only, fail-closed audit package for publications
created by `uwfa_sc_v8_qwen_early_gate_v0/early_gate.py`.  It does not contain
Qwen data, controls, result data, or resolved run authority.  Building and
testing this package must use synthetic bytes only.

## Security boundary

`verify_result.py` performs no path access until all of the following have
passed pure lexical validation:

- the exact authorization string
  `AUDIT_EXACT_QWEN_EARLY_GATE_RESULT_NO_PROMOTION_V0`;
- an absolute output parent and exact safe final name;
- externally supplied SHA-256 pins for the early-gate runner, sealed v8
  manifest/source root, auxiliary STRATA sources, Qwen artifact, baseline
  score member, source-preflight member, COMPLETE file, and RESULT file; and
- the externally supplied exact artifact byte count.

The CLI additionally requires a fresh isolated CPython process (`-I -B`) and
POSIX `openat`/`pread` semantics.  Every directory component, source, artifact,
publication member, and final directory is retained by descriptor.  The audit
ends with descriptor stability, name/inode, byte, member-set, and final-name
rebinding checks.

No values in `UNRESOLVED_EXTERNAL_PINS.json` are authority.  All remain null
until a human/external authority independently supplies them after the run is
complete.

## What is independently checked

- The sealed v8 manifest, exact member set, every member hash/length, and the
  canonical source-snapshot root.
- The exact early-gate runner and its literal pinned constants.
- The current artifact hash/length, parser replay, full/structural geometry,
  and reconstruction digest.
- Completion-last schema, internal seal, exact member set/order/hash/length,
  strict duplicate-free finite JSON, and canonical pretty JSON bytes.
- The nonpromoting claim boundary and all controls/positive-authority
  counters.
- Baseline-score/preflight external file pins and their internal commitments.
- Decoder-bundle, source, pipeline, model, directory, reconstruction, header,
  and selected-decision commitments.
- Literal candidate-container parse, canonical rebuild, CPU causal
  decode/re-encode, full reconstruction, exact physical rate, F, and
  descriptor-backed bandwidth.
- Literal identity-framing parse, canonical rebuild, shared-section bindings,
  and physical byte accounting.
- The sealed decision order, distinguishing a physical hard kill (final
  regardless of controls) from a source survivor (controls and another fresh
  independent audit still required).

## Inherent evidence limitation

The sealed producer explicitly constructs `IDENTITY_FRAMING.bin` as a byte-cost
counterfactual: it pairs original arithmetic payloads with the selected
candidate model and deliberately calls only parse plus non-authoritative
physical metrics.  It is not a UWFA-coded object under that model.  Therefore
the audit can and does prove its canonical framing, hashes, directory/model
commitments, and literal byte costs, but semantic reconstruction from that
counterfactual is impossible under the sealed v8 ABI.  The receipt reports
this explicitly instead of manufacturing a decode claim.

The emitted artifact also lacks the original FP64 Qwen source tensors needed
to recompute the pinned SSE/energy.  The baseline-score file must therefore be
externally hash-pinned.  GPU preflight bytes are authenticated but not replayed
by this CPU-only result audit.  A survivor contains no controls by construction.

Two producer-provenance facts are also impossible to recover from the
publication alone.  The exploratory publisher emits no parent commit marker,
so exact out-of-band COMPLETE/RESULT file pins are mandatory.  Its unsealed
runner reads `Path(__file__)` late instead of executing a retained immutable
snapshot, so the pinned/self-reported runner file hash cannot prove historical
executed bytecode.  The receipt keeps this limitation explicit and never grants
positive-claim authority.

## Invocation

Run only after an external authority fills every pin independently:

```text
python3 -I -B verify_result.py \
  --authorization AUDIT_EXACT_QWEN_EARLY_GATE_RESULT_NO_PROMOTION_V0 \
  --output-parent /absolute/output/parent \
  --final-name exact-final-name \
  --runner /absolute/early_gate.py \
  --expected-runner-sha256 <64-lowercase-hex> \
  --v8-package /absolute/unifilar_wfa_entropy_census_stage0_v8 \
  --expected-v8-manifest-sha256 <64-lowercase-hex> \
  --expected-v8-source-root-sha256 <64-lowercase-hex> \
  --strata-common /absolute/common.py \
  --expected-strata-common-sha256 <64-lowercase-hex> \
  --frozen-auditor /absolute/frozen_auditor.py \
  --expected-frozen-auditor-sha256 <64-lowercase-hex> \
  --artifact /absolute/current-artifact.bin \
  --expected-artifact-sha256 <64-lowercase-hex> \
  --expected-artifact-bytes <exact-integer> \
  --expected-complete-file-sha256 <64-lowercase-hex> \
  --expected-result-file-sha256 <64-lowercase-hex> \
  --expected-baseline-score-sha256 <64-lowercase-hex> \
  --expected-source-preflight-sha256 <64-lowercase-hex>
```

The verifier prints one compact JSON receipt to stdout and never writes to the
producer publication.

## Synthetic tests

```text
python3 -I -B test_source_only.py
```

The tests use only synthetic JSON/bytes and exercise duplicate/nonfinite JSON,
completion-member tampering, claim/counter laundering, decision laundering,
noncanonical containers, identity decode overclaiming, unresolved pins, and
POSIX final-name substitution.
