# UWFA-SC v8 external production dispatcher v2

This package is a new sibling of sealed dispatcher v1 (manifest SHA-256
`e32512c2d5bbbb9cb60aa125bd8bda5986bca8bda0fa384fa6a3b8c5dca39050`)
and is outside the sealed UWFA-SC v8 producer.  V1 is never modified. It is the proposed
production dispatcher for the first Qwen source-phase census.  It is a
**source-only review candidate**: the embedded producer-review, public-commit,
runtime-manifest and decoder-bundle pins are intentionally unresolved, and the
dispatcher intentionally does not self-pin its own future audit.  Consequently
direct `bootstrap.py` execution fails before it accepts a request
path, imports CuPy, queries a GPU, or opens any Qwen/control payload.

The fail-closed placeholders are deliberate.  Replacing them is a separately
reviewed lifecycle transition; it is not permitted to pass the values on a
command line or environment variable.

## Production authority sequence

The only permitted launch is an independently authenticated, out-of-tree
launcher executing the held `bootstrap.py` snapshot under CPython `-I -B` and
calling `dispatch_production` with a typed `ExternalLaunchAuthority`.  The
launcher supplies a typed `ExternalLaunchAuthority` that pins the exact
request, baseline plan, independently reviewed legacy audit, and independently
reviewed original-source binding. The CLI request digest must equal the typed
request pin. Production pins are constructed by `dispatch_production` through
`ProductionPins.embedded()`; they are not a caller argument. The launcher then
supplies the following opaque arguments:

```text
/usr/bin/python3.12 -I -B bootstrap.py \
  --producer-package /absolute/no-symlink/producer \
  --producer-review /absolute/no-symlink/final-review.json \
  --dispatcher-audit /absolute/no-symlink/dispatcher-review.json \
  --request /absolute/no-symlink/source-request.json \
  --request-sha256 <64-lowercase-hex>
```

`bootstrap.py` enforces this order after receiving that typed authority:

1. prove CPython `-I -B`, reject direct launch, and reject unresolved embedded
   producer/runtime/decoder pins;
2. authenticate this dispatcher inventory and its independent audit;
3. authenticate and retain the producer manifest, final review, and all 17
   producer members through descriptor-relative, no-follow opens;
4. authenticate and retain the runtime tree manifest and decoder closure;
5. reject preloaded NumPy/CuPy/SciPy/safetensors and producer/decoder modules,
   reject ambient meta-path/path hooks, and replace the import surface with a
   held-byte/code manifest loader;
6. attach the process-start native RTLD-audit load/unload feed, close its
   baseline against held identities and `/proc/self/maps`, and start the
   append-only hash-chained import/native event ledger;
7. compile the authenticated Python snapshots from buffered held bytes, then
   import the numeric runtime exclusively through that manifest loader;
8. prove the enforcement checkpoint occurred and freshly execute/type-check
   all 150 CPU/CuPy cells, the representative
   outer fold, and an independent held-`nvidia-smi` GPU identity query;
9. only after the typed preflight passes, open the source request and its
   artifact, baseline score, and baseline plan descriptors;
10. run the source phase and the additional repeated/coalesced bandwidth gate;
11. finally rebind held descriptors/names/bytes, module objects, every recorded
    load/unload, and the current native-image set, then publish the complete
    hash-chained event ledger through the completion-last protocol.

The changed dispatcher manifest/design, runtime-tree, decoder-bundle,
independent-review, launch/publication, and structural-verification schemas are
all v2. A v1 review or runtime artifact therefore cannot authorize v2.

The request pathname is treated as an opaque string until step 9.  Neither it
nor an ancestor is statted, resolved, enumerated, or opened before the fresh
preflight succeeds.  Controls are not part of this source-phase dispatcher and
cannot be opened by it.

## Exact external closure

`design_lock.json` mirrors the five non-circular authority pins embedded in the
bootstrap.  The producer manifest, producer review, and reviewed producer
commit are now literal.  Two remain deliberately unresolved:

- exhaustive runtime-lock SHA-256; and
- authenticated decoder-bundle SHA-256.

The dispatcher manifest, independent dispatcher-audit receipt, dispatcher
public commit, and launcher source/review digests are supplied in typed
`ExternalLaunchAuthority`.  They cannot be embedded here: patching this file
with the digest of an audit that reviewed this file would invalidate that audit
and create an infinite self-pin cycle.  The out-of-tree launcher is itself
pinned by the human deployment command or a held descriptor from a separate
trust domain.  The launch receipt records all five external anchors.

The manifest and review descriptors stay open for the complete run.  Producer
members are enumerated only through the retained producer-directory fd after
the pinned manifest is authenticated.  Every member is no-follow opened,
bounded, hashed, retained, and re-fstatted before and after execution.  Python
members are compiled from those authenticated buffers; they are never imported
from the producer pathname.

`runtime_lock.json` points to a separately hashed exhaustive runtime-tree
manifest. Every file under each declared Python import root is held and
verified before ambient `sys.path`, `sys.path_hooks`, and `PathFinder` are
removed. Source and bytecode execute only from retained manifest bytes;
extension modules are opened through retained descriptors. Module authority
is a private object-to-member identity/hash binding, never mutable `__file__`.
All held identities and hashes are checked again after execution. The production lock must cover CPython,
NumPy, CuPy, SciPy, cuda-pathfinder, their native dependencies used by the
process, the process-start native loader auditor, and `nvidia-smi`.

V2 requires the native auditor to be active from process start and to emit
canonical, strictly sequenced `READY`, `LOAD`, `UNLOAD`, and `BASELINE_END`
events on a typed nonblocking descriptor. Thus an unmanifested transient image
still remains in the append-only ledger and fails even if it unloads before a
later `/proc/self/maps` snapshot. Imports, authenticated module removals, and
native events remain hash-chained in `IMPORT_NATIVE_EVENT_LEDGER.json`.

`decoder_bundle.json` binds the frozen STRATA format source, independent
decoder/auditor source, the producer's fixed STRATA adapter, and universal
adapter.  External Python sources are compiled from held buffers after the
runtime tree is authenticated.  No selected bits, levels, base frequencies,
owners, shapes, reconstructed values, or score arrays may appear in this
closure.

V2 binds the repaired `strata_ordinal_bridge.py` into the decoder and pipeline
roots. Its sole crossing converts NumPy integer scalars to built-in `int`
values with dtype round-trip proof, preserves source order and block
partitioning, and proves unique exact ordinal coverage. Both containers are
strict built-in lists (`list[list[int]]`); tuple rows fail because NumPy treats
them as multi-axis indexing rather than one-axis advanced indexing. The source
suite checks a real 2-D NumPy array for equivalence with `np.asarray(row)`.
The decoder bundle
must also reproduce the dispatcher's exact injective logical-to-producer
member map.

The output verifier retains every opened member descriptor until the final
decision, then rechecks descriptor identity, no-follow name identity, and all
bytes. Dispatcher launch rejects duplicate or hard-linked inodes within or
across authority, request, and output trust domains; the publication receipt
carries those inode domains for independent output-alias rejection.

## Source request ABI

The request must be canonical JSON (UTF-8, sorted keys, compact separators,
finite values, no duplicate keys) with exactly this non-circular shape:

```json
{
  "final_name": "source-phase-<safe-name>",
  "inputs": {
    "artifact": {"bytes": 0, "path": "/absolute/...", "sha256": "..."},
    "baseline_plan": {"bytes": 0, "path": "/absolute/...", "sha256": "..."},
    "legacy_independent_audit": {"bytes": 0, "path": "/absolute/...", "sha256": "..."},
    "original_source_binding": {"bytes": 0, "path": "/absolute/...", "sha256": "..."}
  },
  "output_parent": "/absolute/no-symlink/output-parent",
  "producer_public_commit": "<exact pinned commit>",
  "schema": "uwfa-sc-v8-external-source-phase-request-v1",
  "status": "AUTHORIZED_SOURCE_ONLY_NO_CONTROLS",
  "transaction_id": "<32 lowercase hex>"
}
```

The command-line request digest and external request pin must match the held
request bytes. Every input
is opened descriptor-relatively only after preflight, then checked against its
declared length/hash and retained until publication completes.

The baseline plan is canonical, internally sealed, parsed rather than merely
hashed, and cross-bound to the producer commit, artifact length/hash, exact 18
matrix ordinal/role order, score normalization, legacy audit, and original
source binding. The baseline-score wrapper is deliberately **not** a request input. Its
matrix-order reconstruction digest and source full-geometry digest do not
exist until the authenticated adapter has causally extracted and replayed the
artifact.  After that replay, the dispatcher validates the legacy independent
audit's artifact hash, canonical re-encode, SSE, energy and relative MSE
against an internally sealed 18-matrix original-source binding.  It then
constructs canonical `uwfa-bound-baseline-score-v8` bytes in memory and binds
their new SHA-256.  Source full/structural geometry, fresh-preflight digest,
baseline-score digest and every other `BoundEvidence` value are derived by the
dispatcher; none is accepted as a caller assertion.

## Additional bandwidth gate

The producer's normative gate is the strict maximum unique-touched-page ratio
below `2x`.  This dispatcher adds an independent conjunction:

- repeated requested bytes, including every overlapping read call, divided by
  both exact attributable-total and attributable-nonpadding bytes, is `<2` for
  every expert;
- the coalesced byte-range union divided by both denominators is `<2` for every
  expert; and
- the producer's descriptor-backed unique-page gate is true.

The dispatcher recomputes request counts, repeated bytes, coalesced bytes, and
overlap bytes from every literal `instrumented_routed_read_ranges` list.  It
does not trust the producer's summary fields.  An equality of exactly `2` is a
failure.  Installation authentication remains reported separately and is not
silently amortized into expert inference.

The v1 gate accepts the literal container bytes. The authenticated held
`container_codec.py` parses their framing and the dispatcher independently
derives exact owner-local total/nonpadding denominators from the byte ledger;
reported container lengths and denominator fractions must equal those derived
values. This gate is intentionally stricter than the sealed v8 gate. It prevents a
layout that passes only because duplicated/coalescible read requests were
discarded from the operational bandwidth ledger.

## Canonical publication

A successful or negative source phase publishes only the following canonical
members, with conditional binaries present only when the producer actually
emits them:

```text
LAUNCH_RECEIPT.json
RUNTIME_LOCK.authenticated.json
SOURCE_PREFLIGHT.json
BOUND_BASELINE_SCORE.json
SOURCE_PHASE.json
BANDWIDTH_GATE.json
UWFCV8.bin                         (conditional)
IDENTITY_FRAMING.bin               (conditional)
POSTERIOR_HANDOFF.json             (conditional)
COMPLETE.json                      (always last)
```

All generated JSON is the exact `canonical_json` encoding, and incoming score
bytes are copied verbatim with their authenticated digest.  Publication uses a
held output-parent descriptor, a hidden staging directory, fsync, no-replace
rename, exact post-move rehash, and a descriptor-linked sole-link parent marker.
`COMPLETE.json` alone is not publication authority.

No producer payload status is promoted to a final scientific `PASS`.  The most
positive status this dispatcher may publish is
`PASS_MATCHED_NULL_SPECIFICITY_AWAITING_EXTERNAL_RESULT_AUDIT`, and that status
is unreachable in the source-only phase because controls are not opened.

## Source-only verification

```text
python -I -B test_source_only.py
python -I -B verify_source.py --package /absolute/path/to/this/package
```

The hostile suite never opens Qwen or Gaussian-control payloads.  It exercises
unresolved-pin rejection, direct-launch rejection, duplicate JSON keys,
symlink and undeclared-member rejection, descriptor mutation, held-source path
substitution, hostile meta/path hooks, import-then-`sys.modules` deletion,
transient native load/unload closure, preloaded numeric-module rejection,
runtime/decoder closure mismatches, enforcement-before-preflight order,
real 2-D NumPy one-axis advanced-index equivalence and tuple-row rejection,
preflight-before-request order,
request and binding substitution, exact bandwidth recomputation, the strict
`2x` boundary, canonical output schemas, and completion-last member policy.
