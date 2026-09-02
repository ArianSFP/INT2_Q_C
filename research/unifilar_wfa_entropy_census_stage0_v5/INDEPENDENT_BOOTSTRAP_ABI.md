# UWFA-SC v5 independent bootstrap ABI

This document specifies the authority boundary. Nothing inside this producer
tree, including `dispatcher_contract.py`, may authenticate itself.

## Frozen external inputs

The independent bootstrap must hard-code and independently obtain:

- the SHA-256 of the exact `SOURCE_MANIFEST.json` bytes;
- the SHA-256 of an independent v5 source-review receipt whose schema is
  `unifilar-wfa-entropy-census-independent-source-review-v5`;
- the exact public Git commit used for any final CUDA replay; and
- its own separately reviewed source digest.

It must start under CPython `-I -B`. It rejects a symlink at every lexical
ancestor and at the leaf, opens components descriptor-relatively with no-follow
semantics, retains all descriptors, authenticates fstat size and SHA-256 from
the retained descriptor, and re-fstats before and after execution. Directory
enumeration occurs only after the package itself has authenticated and must
match the manifest exactly.

The entrypoint, common arithmetic, universal adapter, container parser, plugin,
CUDA backend, and every other executable module are compiled and executed from
the already-buffered authenticated bytes. A pathname import or public
self-signed token is not authority. The bootstrap binds the source manifest,
review, interpreter executable/version, public commit, and every payload
descriptor into its launch receipt.

## Snapshot module ABI

The bootstrap loads authenticated snapshots in dependency order and passes
module objects explicitly. The universal path requires:

- `uwfa_common.py`: exact candidate bank, Q0.16 fit, arithmetic coder, model
  serialization/deserialization;
- `protocol.py`: fixed 32-byte owner-set ABI and evidence validators;
- `universal_adapter.py`: charged 1..256-expert shape packet, exact role/scalar
  conservation, generic callback decode;
- `container_codec.py`: literal `UWFCV5` build/parse/rebuild/routed-read ledger;
- `stage0_census.py`: all-150 nested experiment and symmetric controls;
- `cupy_backend.py`: exact CUDA path and measured telemetry;
- `result_envelope.py`: completion-last verification.

`strata_sc_adapter.py` is a fixed STRATA15 evaluation plugin only. A production
128-expert launch must provide a separately authenticated adapter implementing:

1. `parse_streams(artifact_bytes)` without repository-relative imports;
2. one charged semantic shape per expert;
3. canonical owner sets and exact `(expert, role, source_offset, weight_count)`
   contributions for every stream;
4. causal public levels/base frequencies regenerated from transmitted immutable
   metadata and decoder state;
5. a standalone reconstruction callback yielding the exact matrix-order FP64
   digest; and
6. literal physical grouping: one region per distinct owner set, common bytes
   stored once, private regions expert-contiguous.

No extracted `levels`, `base`, owners, shape, score, or reconstruction array may
enter as an uncharged side channel. The same adapter and extraction pipeline are
repeated independently for each matched-Gaussian control.

## Required launch sequence

1. Authenticate and retain producer manifest, source-review receipt, external
   bootstrap, exact source members, baseline/current artifact, baseline score,
   source panel, adapter, and all binding receipts.
2. Verify no payload was opened before the external source and dispatcher gates.
3. Execute source-free all-150 CPU/CuPy equality and the 15-stream/6-owner
   representative outer-fold benchmark in a fresh RTX 5090 process from the
   exact public commit. Record real H2D/D2H categories, kernels, synchronized
   time, process-tree RSS/HWM, VRAM/pool peaks, runtime/driver/device. Obtain
   UUID, PCI bus id, and device name independently through NVML or `nvidia-smi`,
   seal that identity record, and require exact agreement with the CUDA receipt.
   The production validator, rather than merely the development emitter, must
   enforce the ordered unique selectors `0..149`, full canonical candidate and
   per-cell count/result/logical-length metadata and hashes, deterministic
   fixture seals, resource plan, and measured telemetry. It must likewise
   enforce the representative fixture/split, all-150 candidate scores,
   deterministic winner, fit/model/container/canonical-rebuild and decoded
   decision commitments, runtime projection, resource plan, and telemetry.
   Recomputed outer hashes do not make duplicate-selector or sparse records
   admissible.
4. Open the source artifact only after all authority and preflight gates pass.
5. Bind current-object fstat bytes/hash, exact source weights, audited SSE,
   source energy, relative MSE, reconstruction digest, geometry, adapter, and
   pipeline before the first fit.
6. Run every source outer fold over all 150 cells, winner refit, exact literal
   pack/parse/standalone decode/re-encode, physical R/F, and fresh routed cold
   reads.
7. Only if every source gate passes, derive the source artifact digest from the
   authenticated source state and require the caller digest to equal it.
   Authenticate all eight controls before any control fit. Each control must
   bind its source-dependent full geometry to itself while exactly reproducing
   the source baseline-plan, decoder, producer-manifest, external-bootstrap,
   extraction-program, universal-adapter, source-snapshot, source-free-preflight,
   and pipeline closure. Then repeat the entire 150-cell pipeline independently
   for each control.
8. Supply a retained, authenticated output-parent descriptor. Create a hidden
   staging directory descriptor-relatively, fsync each member, write
   `COMPLETE.json` exclusively and last, then publish with
   `renameat2(RENAME_NOREPLACE)` and fsync the parent. Never resolve/follow an
   output ancestor, resume an incomplete staging directory, overwrite a raced
   final name, or delete final members after a post-rename fault. Immediately
   before constructing `COMPLETE.json`, re-enumerate the retained staging
   descriptor and no-follow reopen/fstat/rehash every declared regular member;
   reject any mutation, missing name, or undeclared name. Repeat that exact
   check including `COMPLETE.json` immediately before the no-replace rename.
9. A fresh independent result auditor reopens the output and every bound input,
   recomputes all hashes, geometry, reconstruction/MSE, literal rate/F, exact
   owner-local cold pages, nested statistics, control symmetry, and telemetry.

## Literal container and routing ABI

All integers are little-endian and validated before dependent work. The header
is 4096 bytes; directory, region header, and frame header are each 256 bytes;
the owner set is always exactly 32 bytes. Expert count is exactly 1..256. Every
bit above the count is zero. Zero owner sets, duplicate owner regions, duplicate
stream ordinals, inconsistent repeated owner sets, missing experts/roles/scalars,
nonfinite scales, overflow, trailing bytes, and nonzero padding are fatal.

Fresh routed decode starts with `AuthenticatedDescriptorSource`, which fstats
and hashes the complete held regular-file descriptor once against the expected
container digest. Its installation scan ranges, bytes, and touched page union
are reported separately. Each expert
uses a fresh instrumented duplicate of that same descriptor identity, reads
global state plus only owned regions, causally decodes/re-encodes its payloads,
reconstructs all three matrices, and reports exact touched pages. There is no
full-body digest, unowned frame/region parse, or unowned-padding scan in the
routed phase; the separate installation hash is reported as such. Cold
amplification is divided by that expert's
exact private bytes plus exact fractional shares of every common region; both
total and nonpadding denominators are computed and the worse is gated. Total
container bytes divided by expert count is forbidden.

Receipts additionally expose total modeled symbols, shape-derived weights, and
their modeled-symbols-per-weight density in `modeled_symbol_density`, plus
per-expert metrics and aggregate read request/repetition accounting in
`routed_read_request_aggregates`. These are explicit audit diagnostics.
They do not change the frozen routed cold gate, whose numerator is the unique
4096-byte touched-page union.

The literal result also carries a `uwfa-sc-v5-posterior-diagnostic-handoff`
record. An external diagnostic must authenticate the named container and source
closure, then reproduce its decoded-decision commitment by causal re-decode.
The handoff binds source/reconstruction identity but contains no posterior or
MMSE result and cannot authorize a TACTIC-stage claim.

## Result and claim boundary

The producer must never emit `PASS` for payload performance. The last producer
status before independent numeric audit is
`PASS_MATCHED_NULL_SPECIFICITY_AWAITING_EXTERNAL_RESULT_AUDIT`. Promotion is the
conjunction of physical rate/F, strict cold read, nested heldout, matched-null
specificity, standalone reconstruction/re-encoding, all integrity gates, and
the external result audit.

Development direct-copy RunPod receipts have no evidentiary value. Only a fresh
receipt from the exact frozen public commit, authenticated by the external
bootstrap and independently audited, can support a claim.
