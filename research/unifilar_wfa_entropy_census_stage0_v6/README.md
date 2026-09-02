# Universal UWFA-SC v6 — pre-review source-only producer

This directory is a new, source-only successor to the stopped v5
postimplementation-review candidate. The v5 tree and its review/development
receipts remain unchanged. This v6 tree is
intentionally **not sealed**, has no payload authority, and must not contain
a `SOURCE_MANIFEST.json` until an independent postimplementation red-team has
accepted the exact tree. No Qwen, current-artifact, extracted-stream, or
matched-Gaussian payload was opened, statted, hashed, or enumerated while this
tree was built.

The experiment asks whether a universal sparse unifilar probability law can
recoup more than `0.15288996696291447` physical bits per source weight from a
complete existing codec object while preserving identical reconstruction. A
positive scientific result additionally requires actual rate in `[2.15,2.5]`,
`F = D*2^(2R) <= 0.8`, strict owner-local cold-read amplification `<2`, nested
held-out benefit, matched-Gaussian specificity, and a fresh independent result
audit. Producer self-tests cannot establish any of those payload claims.

## What is genuinely universal

`universal_adapter.py`, `protocol.py`, and `container_codec.py` define the
portable core. The core accepts 1 through 256 experts and an arbitrary legal
positive `(hidden, intermediate)` shape for every expert. A charged semantic
packet identifies the shape, and every directory contribution names an expert,
one of `gate/up/down`, a role-local scalar offset, and a positive scalar count.
The parser proves that every scalar of all three matrices of every expert is
covered exactly once. Empty experts, holes, overlaps, duplicate ordinals,
missing streams, noncanonical bitsets, and shape/weight mismatches are fatal.

Ownership has exactly one ABI everywhere: a fixed 32-byte little-endian bitset.
It is nonempty, bits at or above `expert_count` are zero, and it is repeated
byte-for-byte in the directory, region, frame, and ledger. There is no u16/u32
alias and no shift based on an unvalidated expert count.

`universal_adapter.decode_with_callbacks` is the generic adapter protocol. A
model-specific integration supplies only causal public-context regeneration and
reconstruction callbacks; all shape and scalar coverage metadata is physically
charged. `fixture_portability.py` exercises this protocol at 128 experts with
an awkward 17x19 shape, mixed owner sets including expert 127, and shared tails.
It also carries the independent-audit E250 unequal-shape fixture: 751 streams,
251 regions, and a shared Gate tail whose owners contribute one and two scalars.

`strata_sc_adapter.py` is only a fixed STRATA15 evaluation plugin. It retains
the inherited 15-block decoder path, derives exact role-local contributions,
and regenerates base-frequency contexts during decode. It is not the universal
core and cannot support a claim about arbitrary shapes by itself.

## Literal v6 object

The only measured object is one complete `UWFCV6` byte string:

1. 4096-byte authenticated header;
2. charged universal semantic packet and optional plugin extension;
3. immutable decoder state;
4. serialized exact Q0.16 unifilar model;
5. 256-byte-per-stream canonical directory;
6. one page-aligned region per distinct owner set, in canonical owner order;
7. 256-byte frame headers, contribution records, payloads, alignment, and
   explicit integer-described rate padding.

Every addition, multiplication, allocation, shift, loop bound, and GPU geometry
is checked against a hard maximum first. The parser rejects trailing bytes,
nonzero padding/reserved fields, duplicate owner regions, zero-length fields,
nonfinite scales, inconsistent repeated metadata, and any digest or coverage
mismatch. It deserializes the transmitted model; canonical rebuild must reproduce
the complete object byte-for-byte. Rate is exactly
`8*len(container)/shape_derived_weight_count`; the 2.15 floor is stored as the
integer rational 43/20, never inferred from a floating bpw value.

## Cold-read proof

A held regular-file descriptor is first fstat-bound and fully hashed once by
`AuthenticatedDescriptorSource`; that installation authentication scan is
reported separately with its exact chunk ranges, scan bytes, and full touched
page union. For each expert, `routed_read_expert` starts a fresh
instrumented duplicate of that same descriptor/inode. It
reads the global header/semantics/model/directory and only the owner regions
needed by that expert, re-fstats the descriptor identity, validates complete
role coverage, and reports the exact union of touched 4096-byte pages. The
routing receipt also reports the number and bytes of read requests *with
repetition*, both per expert and in the top-level
`routed_read_request_aggregates` record (including unique-requested, repeated-
overlap, and unique-page totals). These repeated request-byte fields are
diagnostic and auditable; they do not replace or alter
the frozen cold numerator, which remains the unique touched-page union.
The top-level `modeled_symbol_density` record reports unique directory modeled
symbols, shape-derived source weights, their modeled-symbols-per-weight
density, routed totals, and shared-stream symbol reuse alongside the
directory/accounting ledger.

The routed adapter then causally decodes and canonically re-encodes every owned
payload, reconstructs that expert's complete Gate/Up/Down matrices, and the
session finalizer reproduces the literal full-reconstruction digest across all
experts. It performs no body-hash, unselected-frame, unselected-region, or
rate-padding prepass. A hostile test corrupts an unselected private frame: full
parsing rejects it, while selected routed decode succeeds without touching
that frame's page.

The denominator is owner-local: each private byte plus that expert's exact fair
share of every shared/global byte. A second denominator excludes padding. The
gate uses the worse of the two ratios for every expert. `total_bytes/E` is
explicitly forbidden; the hostile suite includes an asymmetric shared-tail
case where that shortcut reports 1.882 while the correct owner-local page ratio
fails decisively.

## Scientific protocol

The bank is exactly 5 topologies x 6 state sizes x 5 reset lengths = 150 cells.
Frequency fitting uses exact counts, Jeffreys half-counts, nearest integer Q0.16
rounding, and clamp to 1..65535. The state resets before lookup at the first
symbol and at each public reset boundary.

The total primary ladder is exact semantic-identity exclusion. Every estimable
outer expert fold searches all 150 cells on development streams that do not
share the held-out identity, selects on exact physical validation length,
refits the winner on all development streams, and scores untouched outer data.
Legal panels with no possible development source return a typed nonpromotion
before fitting. The stricter layer-and-expert-coordinate-disjoint split runs
only as an explicitly nonpromoting diagnostic when it has nonempty folds.
Shared streams use exact semantic per-owner weight contributions. Student-t
degrees of freedom are derived from the actual nonempty fold count.

All eight matched-Gaussian controls must authenticate, recompute exact moment
matching through an independently injected replayer, and geometry/provenance
bind before fitting starts. The caller-supplied source artifact digest must
equal the digest derived from the authenticated source panel. Cross-source
equality uses only structural geometry; the source and each control bind their
source-dependent arithmetic lengths to their own artifact, score,
reconstruction, and full-geometry receipt. Every control must also reproduce
the source's exact baseline-plan, decoder, producer-manifest, external-bootstrap,
extraction-program, universal-adapter, source-snapshot, source-free-preflight,
and pipeline closure. All eight closures are authenticated before any numeric
fit begins. Each
control independently repeats extraction,
all-150 nested selection, refit, literal pack, parse, standalone decode, and
score. A control cannot rescue a source failure.

## CuPy accounting and feasibility

The exact CUDA backend accounts separately for payload, root descriptors,
subset descriptors, freshly copied launch offsets/lengths, every model-table
copy, and scalar kernel arguments. Opaque packed handles are backend-owned and
revalidated against an immutable private snapshot before any CuPy call; forged,
mutated, or cross-backend handles fail with zero device calls. D2H
is separately counted through an explicit transfer method. Synchronized phase
samples report process-tree RSS, process HWM, incremental VRAM from device free
memory, CuPy default-pool used/total bytes, pinned-pool free blocks, kernel
counts, symbol updates, JIT time, and kernel wall time.
Static host/device admission is derived from the declared maxima, and a second
dynamic gate uses current process-tree RSS and free VRAM before host blob
concatenation or any CuPy payload copy. Telemetry failure is fatal. The CUDA
receipt binds the device UUID and canonical PCI bus id to an independently
sealed `nvidia-smi`/NVML identity receipt.

The typed source-free preflight boundary does not accept summary booleans. The
all-150 receipt must contain the exact ordered, unique selector sequence
`0..149`, canonical candidate metadata, result/logical-length/count hashes and
counts for every cell, deterministic fixture identity, aggregate selector and
cell seals, a checked resource plan, measured telemetry, and the independently
matched GPU identity. The representative receipt must bind its complete
candidate coverage and deterministic winner, split, fixture, fit/model,
container/canonical-rebuild, decoded-decision, resource, runtime-projection,
and telemetry records. Duplicate-selector and sparse-representative receipts
are rejected even when their outer receipt hashes are recomputed honestly.

The source-free representative benchmark contains 15 streams and 6 owners. It
runs one complete outer fold (all-150 selection, winner refit, held-out score),
then serializes/parses/decodes/re-encodes the literal object. Its projection is
transparent: 13 complete pipelines (source plus four survivor searches plus
eight controls), six outer folds plus final fit/score, at 50% of measured
throughput. The source-free 128-expert fixture proves portability, not full
payload runtime.

Development smoke runs made from a direct source copy to an RTX 5090 are not
claim evidence. Final telemetry must be regenerated in a fresh process from the
exact public Git commit after postimplementation review, freeze, external
source audit, and external dispatcher audit.

## Frozen diagnostic handoff (no posterior computation)

`posterior_diagnostic_handoff` commits a later separately frozen diagnostic to
the literal container, original artifact, independent score, source geometry,
extraction/decoder/adapter sources, serialized model, immutable public-context
state, semantic routes, full FP64 reconstruction, and every decoded SC decision
via per-stream length-prefixed `(selected bits, public levels, public base-frequency
u16le)` triplet commitments. The decisions remain recoverable only by
causally re-decoding the authenticated literal payload. This producer performs
no posterior-centroid, MMSE, or TACTIC computation and the handoff grants no
payload or performance authority.

## Local source-only gate

Run from this directory with an isolated interpreter:

```text
python3 -I -B test_source_only.py
```

Before freeze, `verify_source.py` is expected to fail because the manifest is
intentionally absent. After independent review authorizes freeze, the exact
command is:

```text
python3 -I -B verify_source.py --package /absolute/path/to/this/directory
```

Direct execution of `stage0_census.py` is inert and returns status 2 before
argument or payload handling. The in-tree dispatcher is a contract specimen,
not a root of trust.

Publication is likewise fail-closed at the last output boundary.
`COMPLETE.json` is only a directory-local content seal; it is not publication
authority. Before and after the no-replace directory move, the producer
re-enumerates the retained staging-directory descriptor and no-follow reopens,
fstats, and rehashes every committed regular member. After the move it requires
the final name's `(st_dev, st_ino)` to equal that retained directory fd.

Only then does it write the parent commit record into a held regular-file fd.
It prefers `O_TMPFILE`; where the filesystem rejects that primitive, it uses a
unique no-follow creation anchor solely to obtain the fd. In both cases the
final marker is atomically linked from the exact fd with
`linkat(AT_EMPTY_PATH)`, or the kernel-owned `/proc/self/fd/<n>` equivalent when
the container lacks the capability needed for `AT_EMPTY_PATH`. The creation
anchor pathname is never the publication source. The producer immediately
rebinds both final and marker names to their retained fds and rehashes the held
directory again. Independent consumers require this parent marker, its expected
source root, parent/directory/marker inode commitments, exact sorted member
table, domain-separated directory root, and matching `COMPLETE.json`. A moved
directory with no marker is incomplete; a post-marker substitution is rejected.

This mechanism detects substitution but cannot make a directory immutable
against a continuously malicious writer holding the same parent-directory
privileges. That stronger threat model requires a verifier-owned parent,
privilege separation, or an externally authenticated signature/fs-verity layer.

## Claim boundary and remaining gates

This producer can establish arithmetic, grammar, conservation, routing, and
telemetry-contract properties on synthetic bytes. It does not establish a Qwen
gain, Gaussian specificity, production cold-read result, or universal
performance.

After source review and freeze, all of these remain mandatory:

- an externally pinned v6 source audit;
- an independently audited pinned dispatcher/bootstrap;
- a fresh RTX 5090 all-150 and representative replay from the exact public
  GitHub commit;
- no payload opening before external authority;
- a fresh-process independent audit of every numeric result and output byte.

The negative-result boundary is equally narrow: a failure closes this exact
sparse-unifilar recoder, not general WFA/MPS/MERA/flow/reverse-channel,
Gray-Wyner, or alternative quantizer architectures.
