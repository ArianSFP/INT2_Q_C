# UWFA-SC v7 independent source-only audit

Date: 2026-09-02

## Verdict

**BLOCK_SOURCE_FREEZE**

V7 substantively repairs four of the five v6 blockers and preserves the prior
regressions. One requested repair is incomplete: inner candidate selection uses
the serializer-shared literal layout, but the outer held-out saving removes the
model from that literal comparison and reintroduces its cost as unaligned raw
model bytes. That is not the promised exact 64-byte/4-KiB physical ledger.

No producer file was edited. No manifest/freeze or commit was created. No
Qwen/model payload was opened, and this report makes no production entropy or
compression claim.

## Authentication and execution summary

- Exact source inventory: 17 files, 625,600 bytes.
- Independent local/RunPod inventory root:
  `2cd5e4cc7f53ec0e9ab91f8a9f9a505b8e82e316ab7fe0c1e69e8dcac28bc97f`.
- Full authenticated POSIX suite: 66 tests, 65 pass plus one expected
  pre-manifest skip, zero failures/errors, 78.994 seconds.
- Independent source/receipt verifier:
  `766217ea5cf6aa7fbe52c36288a7cb4f797b715cb7a9d86ebff3cab934163ddc`.
- One source-free RTX 5090 receipt independently passed exact member, seal,
  marker, sole-link, directory-root, source-root and selector checks.
- Receipt directory root:
  `504d1b429346d20c6f22808e8fa5f2dd67590de5686fdc2123fc13286f35fdf9`.
- Parent commit:
  `81618051250a9bd1e6b9a3d3c5bda3af3f312beecb72375487009e490d47cd1b`.

Exact inventory and receipt evidence are in `SOURCE_INVENTORY.tsv` and
`TEST_EVIDENCE.md`.

## Six-gate review

### 1. Held and bounded authenticated result consumption — PASS

[`result_envelope.py`](../unifilar_wfa_entropy_census_stage0_v7/result_envelope.py)
now returns a context-managed `VerifiedOutputBundle` (lines 108-160). Marker,
final-directory and all member descriptors remain held through consumption
(lines 138-143 and 299-314). `read_member_bytes` reads only the held descriptor,
checks frozen per-member/aggregate caps, rechecks descriptor identity and
SHA-256, and returns immutable buffered bytes (lines 185-228). It does not
reopen a result member by pathname.

The POSIX suite verified consumption after both marker/final path replacement,
post-verification held-member mutation rejection, and cap freezing before
buffering (`test_source_only.py` lines 2138-2220). This satisfies the requested
held/bounded consumption contract.

### 2. Post-link marker bytes, exact link count and branch tests — PASS

[`uwfa_common.py`](../unifilar_wfa_entropy_census_stage0_v7/uwfa_common.py)
`_verify_held_commit_bytes` re-reads the retained marker, validates exact bytes,
SHA-256, stable identity and expected `st_nlink` (lines 1108-1139). Completion
does this before link and repeatedly after marker linking, anchor cleanup and
parent fsync, requiring the final marker to be the sole link (lines 1288-1345).

Tests cover content mutation, replacement, extra hardlink, forced `O_TMPFILE`,
`AT_EMPTY_PATH`, `/proc/self/fd`, and all three directory substitution windows
(`test_source_only.py` lines 2097-2136 and 2222-2353). The independently verified
RTX receipt marker had `st_nlink=1` and matched its sealed parent commit/root.

### 3. Exact literal inner and held-out physical scoring — BLOCK

The inner selector is repaired. `literal_validation_score` (`stage0_census.py`
lines 694-766) passes the candidate's serialized model and all stream ownership
geometry to `measure_literal_container_layout`. That shared function includes
frame metadata, 64-byte frame alignment, 4-KiB model/directory/regions and rate
padding (`container_codec.py` lines 181-300); the real builder calls the same
function and asserts its output agrees (lines 627-667). Serializer-equality and
alignment-order-reversal regressions pass.

The outer held-out score breaks that exactness:

1. Lines 1605-1611 compute a baseline hybrid layout and candidate hybrid layout
   using the same selected model and frequencies. The literal model section is
   therefore present in both totals and cancels in their difference.
2. Lines 1627-1629 then subtract
   `8 * model_ledger(selected)["physical_model_bytes"]`.
3. `physical_model_bytes` is only the raw serialization length
   (`uwfa_common.py` lines 297-310), while the literal grammar aligns the
   directory after the model to 4 KiB (`container_codec.py` lines 261-263).

For example, the two-state model is 1,602 raw bytes while its literal model
section advances the following directory by a 4,096-byte slot. Depending on
the comparator layout, the true incremental cost is governed by that aligned
boundary, not necessarily 1,602 bytes. Consequently
`literal_test_saving_after_model_bits`, the pooled target gate and component
positivity are not certified physical quantities.

No source regression asserts the outer formula against two real measured or
serialized comparator layouts; the existing measure/serializer test exercises
only a single layout.

Required repair:

- represent both the actual baseline/reference packet and selected WFA packet
  in the one shared measure/serializer grammar and compute held-out saving only
  as their literal byte difference; do not append raw model bytes outside that
  grammar;
- add zero-payload-gain and model-size boundary cases, including two-state
  1,602-byte and transitions across each 4-KiB boundary, and assert exact
  equality with serialized comparator containers.

### 4. Workload-bound telemetry and canonical UUID/PCI — PASS

Canonical UUID and PCI grammars are enforced by `protocol.py` and applied to
both CUDA and independently sealed identities (`stage0_census.py` lines
132-172). All-150 expected calls, kernels, updates and transfer bytes are
derived at lines 296-354 and compared field-by-field at lines 423-427.
Representative exact workload formulas are derived and enforced at lines
532-611. A resealed telemetry/workload forgery regression passes.

The authenticated RTX receipt contained canonical identity, exact selectors
`0..149`, and a representative phase sum of 142,804,641 count plus 42,047,565
length updates, exactly matching its 184,852,206 measured updates.

### 5. Disjoint dependence components and non-iid gate — PASS

`_dependence_components` builds connected components of the exact stream-owner
bipartite relation (`stage0_census.py` lines 1261-1289); a shared stream joins
all owners, and component folds prove no stream crosses a component boundary
(lines 1292-1343). Fewer than two components returns a typed hold before fit
(lines 1529-1542).

There is no operational Student-t promotion calculation. Promotion requires
the pooled absolute target and strictly positive saving in every component;
leave-one-component-out and means are diagnostic only (lines 1654-1700).
All-owner-shared and single-expert regressions pass. The numeric saving values
remain blocked only because gate 3's physical ledger is wrong.

### 6. Prior regressions — PASS subject to gate 3

The authenticated suite retains coverage of:

- nine-part source/control closure and all eight full matched controls;
- unique ordered preflight cells `0..149`;
- resource admission before packing/allocation;
- exact triplet commitments, not bit-only digests;
- final literal physical rate, explicit symbol density, unique touched pages,
  requested/repeated requested bytes, and routed read denominators;
- 128/250/256 expert portability, including E250 unequal shapes and shared
  tails;
- completion-last, exact member/root rehash, sole parent-marker authority and
  the three publication fault windows.

These regressions do not rescue the incorrect held-out model ledger.

## Disposition

| Gate | Verdict |
|---|---|
| Held/bounded verified consumption | PASS |
| Post-link marker bytes/link count/branches | PASS |
| Literal inner candidate score | PASS |
| Literal held-out/model score | **BLOCK** |
| Workload telemetry and canonical GPU identity | PASS |
| Dependency components and no-iid promotion | PASS, numeric input blocked by prior row |
| Prior source regressions | PASS |

The smallest next step is only the gate-3 repair and its exact serializer-
comparator boundary tests. After that, re-authenticate a new immutable source
inventory and rerun this same source-only audit. V7 must not be frozen or used
on Qwen payloads in its current form.
