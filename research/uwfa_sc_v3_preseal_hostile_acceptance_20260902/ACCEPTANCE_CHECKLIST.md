# UWFA-SC v3 pre-seal hostile acceptance checklist

Date: 2026-09-02  
Status: **SOURCE-FREE DESIGN REVIEW; BLOCK PAYLOAD UNTIL EVERY mandatory gate passes**

## 0. Evidence boundary and frozen inputs

This checklist is independent of the active v3 producer directory. It was
prepared without opening, stating, hashing, enumerating, or otherwise accessing
Qwen weights, the current finite artifact, extracted SC streams, or Gaussian
controls. It does not grant payload authority.

The review is bound to these source-free inputs:

- frozen UWFA-SC v2 producer manifest SHA-256:
  `223a96585444a0b3e4344c470e243dbd4b84662fddfda881185e879a4caee693`;
- frozen independent v2 audit manifest SHA-256:
  `be49ea3fa086c4faf740c1896ad5085c12f2f83b261f8d6949862c3e70708248`;
- universal SwiGLU-MoE codec contract SHA-256:
  `0c0941a2e38dbb4e043db91eda3eb500a5989f4c6c23aa4a4fa61e1ab11d860e`;
- UWFA-SC v2 integrated-container specification SHA-256:
  `99460ee4f77cef68408d043b162d8b0996853d363a0982712300223855dba367`.

The v2 audit status is `BLOCK_UNIVERSAL_LAUNCH_AND_PAYLOAD`. A new version,
new manifest, or producer self-seal cannot inherit a pass from v2. A new,
independently pinned source audit and dispatcher are mandatory.

## 1. Promotion predicate

The launch predicate is one explicit conjunction. Missing evidence is false:

```text
source_manifest_frozen
and independent_source_audit_pass
and external_bootstrap_pass
and source_free_hostile_tests_pass
and representative_runpod_preflight_pass
and exact_container_roundtrip_pass
and universal_portability_pass
and payload_authority_explicitly_granted
```

After an authorized source run, promotion is a second conjunction:

```text
independent_literal_decode
and independent_original_BF16_score
and exact_actual_rate_in_[2.15,2.5]
and F <= 0.8
and max_cold_read_amplification < 2.0
and nested_heldout_gate
and symmetric_control_gate
and fresh_process_independent_result_audit
```

Controls cannot rescue an absolute packet, reconstruction, rate, `F`, or cold
read failure. Separate experimental gains cannot be added.

## 2. Gate A — canonical expert universe and owner-set ABI

### A1. Bound the universe before using it

- The format supports **exactly `1 <= E <= 256`** experts.
- The fixed header is read first. `E` is validated before any shift, loop over
  experts, owner-set conversion, allocation, record-count product, or GPU call.
- `E=0`, `E=257`, the largest field value, and a correctly re-sealed malicious
  header are rejected in bounded time and bounded memory.
- No parser path evaluates `1 << E`, `(1 << E)-1`, or an equivalent giant
  integer from an untrusted count. Owner membership is decoded from bounded
  bytes.

### A2. One owner-bitset representation

Freeze one representation everywhere:

```text
owner_bytes = ceil(E / 8),  1 <= owner_bytes <= 32
expert i <=> owner_bits[i >> 3] & (1 << (i & 7))
```

- The header serializes `owner_bytes`; the parser requires it to equal
  `ceil(E/8)`, not merely to be large enough.
- Every inherited block record, directory record, frame header, region header,
  semantic record, result row, and cold-ledger row uses this same byte ABI.
- For `E=256`, the representation is exactly 32 bytes and bit 255 works.
- Bits `E..(8*owner_bytes-1)` are zero. Leading/trailing alternative encodings,
  shorter encodings, sign extension, integer masks, and mixed endian/bit order
  are rejected.
- Each owner set is nonzero. Parsing it produces a strictly increasing tuple
  of unique expert ordinals.
- Repeated copies of an owner set for one object are byte-identical, not merely
  set-equivalent.
- If a fixed 32-byte field is chosen instead, that must be the only ABI and all
  bytes/bits above `E` must be zero. Supporting both variable and fixed forms is
  noncanonical and is a failure.

### A3. Nonempty, complete ownership universe

- The union of stream owner sets is exactly `{0,...,E-1}`.
- Every declared expert owns at least one nonempty stream and positive source
  weight. Empty experts cannot amortize global bytes.
- No owner bit names an undeclared expert.
- The owner set is derived from decoded semantic block/group membership and is
  compared with the literal owner field; a producer-supplied mask is not an
  independent source of truth.
- A region has one owner set. All its frames and directory rows have exactly
  that set.
- Prefer and test one region per distinct owner set. If multiple regions with
  the same owner set are legal, their ordering and necessity must be uniquely
  specified; otherwise duplicate regions are rejected.

### A4. Boundary portability tests

Positive round trips are required for `E={1,2,7,8,9,31,32,33,127,128,255,256}`.
The fixture must address at least experts `0,7,8,31,32,127,128,254,255` where
legal and must include singleton and multi-owner sets crossing byte boundaries.

Hostile mutations, with all integrity fields re-sealed, must reject:

- zero owner set;
- a high unused bit;
- bit 127 represented as bit 31 or aliased through a group number;
- declared `E=128` with only expert 0 owning data;
- inconsistent owner copies among immutable metadata, directory, frame, and
  region;
- duplicate expert ordinals after decoding;
- altered `owner_bytes` with the same apparent low owners; and
- `E=257` before any owner conversion or allocation.

## 3. Gate B — all bounds before shifts, products, slicing, or allocation

### B1. File and header envelope

- An external reader obtains and bounds file size before reading or memory
  mapping the complete object. A function accepting an already allocated
  arbitrary `bytes` object does not prove resource safety.
- Freeze maxima for file bytes, weights, hidden width, intermediate width,
  matrices, blocks, streams, regions, symbols, logical bits, model bytes, and
  directory bytes. Parser and builder use the same values.
- Validate magic/version/header/page size, exact total bytes, `E`, counts, and
  shape ranks before computing any dependent quantity.
- All JSON evidence readers reject duplicate keys, nonfinite numbers, excessive
  size/depth, unknown fields, and noncanonical identifiers before constructing
  large objects.

### B2. Checked integer arithmetic

Every operation below has an explicit checked-add, checked-multiply,
checked-align, or checked-shift guard against both the format maximum and the
target machine type:

- `hidden * intermediate`, `3 * E * hidden * intermediate`, and padded shape
  products;
- `streams * directory_record_bytes` and `regions * region_record_bytes`;
- `offset + length`, `logical_bits + 7`, page rounding, frame rounding, and
  cumulative region offsets;
- `1 << logn` only after `logn` and resulting `n` are bounded;
- `states * contexts * frequency_bytes` and count tensor sizes;
- concatenated symbol bytes and `2 * symbol_count` base-frequency bytes; and
- rate-floor byte computation without binary floating-point round-trip.

Reject overflow or out-of-range values before slicing, list/range construction,
NumPy/CuPy allocation, or packing into a narrower integer.

### B3. GPU-specific limits

- Validate stream count before conversion to `uint32`.
- Validate offsets, lengths, total packed symbols, and every
  `offset+length <= packed_symbols` before conversion to `uint64` or kernel
  launch.
- Validate topology, states, reset length, context count, model length, and
  kernel grid dimensions against both the frozen bank and CUDA argument types.
- Require nonzero reset length, legal power-of-two state counts if bit masking
  assumes that property, and frequency values in `1..65535`.
- No `cp.asarray`, `cp.frombuffer`, advanced-index materialization, or RawKernel
  launch occurs before all relevant host-side bounds pass.

### B4. Hostile bounded-resource tests

Run correctly re-sealed mutations for huge stream/region/model counts, huge
logical lengths with tiny payloads, wraparound offsets, near-EOF additions,
`logn` extremes, zero/huge reset lengths, and GPU narrowing boundaries. Each
must fail before the patched allocation/shift/loop. Record wall time and peak
RSS/VRAM to prove bounded rejection.

## 4. Gate C — shape, role, weight, stream, and byte conservation

### C1. Universal SwiGLU semantics

- Geometry is literal and shape-driven, never fixed to `768x2048`, six experts,
  fifteen blocks, or a Qwen model/layer/expert identity.
- Each expert has exactly one Gate `[m,h]`, one Up `[m,h]`, and one Down
  `[h,m]`; `h>0`, `m>0`, and role/shape compatibility is checked.
- Shape-derived source weights use exact checked integers. Tensor padding is
  distinguished from real source weights and is physically represented or
  proved implicit.
- A source-free fixture with dimensions different from the evaluation panel
  must decode. Use awkward legal dimensions to exercise tails.

### C2. Scalar and group coverage

- Every original scalar belongs to exactly one semantic group and exactly one
  decoded reconstruction location.
- Maintain an integer coverage map or reject duplicates immediately. A narrow
  `uint8` counter that can wrap after repeated assignments is insufficient.
- `sum(block_weight_charge) == shape_derived_weights` exactly.
- If a block spans experts, serialize or deterministically derive exact
  per-owner weight contributions. Require every listed contribution positive,
  their sum equal to the block charge, and per-expert sums equal to that
  expert's three matrix sizes.
- Do **not** divide a shared block's weight equally by owner count unless the
  semantic mapping independently proves equal contributions. This is a likely
  v2 secondary error for arbitrary shapes/tails.

### C3. Stream bijection

- Stream ordinals are exactly `0..S-1`, each once.
- Each directory row identifies exactly one frame and each frame exactly one
  directory row. Payload ranges and complete frame ranges are pairwise
  nonoverlapping, even when payload hashes are equal.
- Region ordinals are canonical and complete; region order and frame order are
  uniquely defined. Parser acceptance plus later optional rebuild is not an
  excuse for accepting multiple byte encodings.
- The declared symbol count equals the exact number of SC calls to
  `decode(original_freq1)` and the exact number of causal decisions re-encoded.
- Every stream begins from the specified state and reset position. Cross-stream
  state is forbidden unless it is literal, charged, and compatible with routed
  random access.

### C4. Complete byte partition

Produce a machine-checkable interval ledger covering `[0, container_bytes)`
exactly once as header, immutable semantics, model, directory/index, region
header, frame header, payload, alignment padding, or rate-floor padding.

- No overlaps, holes, aliases, out-of-order regions, or bytes after canonical
  EOF.
- Every padding byte is zero and has a declared owner set.
- Global bytes are owned by all `E`; region/frame/payload/padding bytes use the
  exact region owner set. Zero-owner or unallocated padding is forbidden.
- Sum of exact rational owner allocations equals literal container bytes. Do
  not use floating tolerance to prove conservation.
- The same decomposition drives physical rate and the cold-read denominator;
  a separate spreadsheet/JSON ledger is not authoritative.

## 5. Gate D — literal parser, causal decode, and canonical re-encode

### D1. Standalone inputs

The independent decoder starts with only:

1. the literal v3 container;
2. independently authenticated universal decoder source;
3. role/shape semantics contained in the container; and
4. state causally reconstructed from bytes it has read.

It receives no old container, extracted bits, levels, base frequencies,
reconstruction array, input lock, model identity, stream identity as a
probability key, or producer result JSON.

### D2. Canonical model

- Deserialize the literal model before any payload.
- Enforce exact topology/state/reset selector bank, table dimensions, frequency
  range, reserved bytes, row normalization rules, and canonical serialization.
- Re-serialization is byte-identical. Duplicate encodings of one model reject.
- Every adapted value used by decoding is traceable to charged model bytes.

### D3. Causal SC/UWFA adapter

For every selected decision:

- SC regenerates `original_freq1` from previously decoded state;
- the level is set before decoding;
- reset happens before positions `0,K,2K,...`;
- context is derived only from decoder-visible level, prior bin, and position
  within reset;
- transmitted model plus current state yields `uwfa_freq1`;
- the arithmetic decoder emits one bit; and
- state transitions only after that bit.

Test reset edges, every level/prior bin, frequency extremes `1` and `65535`,
terminal underflow/flush behavior, and every frozen transition topology.

### D4. Re-encoding and reconstruction

- Re-encode every decoded stream from regenerated decisions and regenerated
  contexts. Require exact logical length and byte-for-byte payload equality.
- Reject truncated payloads, appended unused arithmetic bits, nonzero terminal
  bits, and alternate noncanonical encodings that decode to the same symbols.
- Canonically rebuild the complete container byte-for-byte using integer-stored
  padding/layout facts. Do not infer a padding floor by converting the current
  byte length to a float bpw and back.
- Restore each source group exactly once, invert transforms, and reproduce the
  complete reconstruction digest and per-matrix digests.
- A fresh result auditor recomputes original-BF16 FP64 SSE, source energy, and
  relative MSE. Header MSE and producer digest are evidence to compare, not
  authoritative score inputs.
- Compute `R=8*literal_bytes/source_weights` from integers and compute
  `F=D*2^(2R)` at sufficient precision. Gate the unrounded values.

### D5. Semantic hostile corpus

For mutations below, recompute CRC/SHA fields so rejection must come from
semantic checks:

- unknown flags/reserved bytes and NaN/Inf/nonpositive scale;
- zero symbols/logical bits/payload; inconsistent ceil-bit length;
- directory/frame/region mismatch;
- duplicate stream, missing stream, swapped frames, duplicate region owner set;
- overlapping or aliased frames, payload crossing region/EOF, a region hidden
  inside global bytes, and trailing bytes;
- nonzero shared/frame/region/rate padding;
- model selector/table mismatch and a noncanonical but numerically equal model;
- source digest mismatch after causal decode; and
- reconstruction/group coverage mismatch.

## 6. Gate E — real routed-read measurement and cold ledger

### E1. Do not use a post-hoc trace

A parser that first receives the complete container as a `bytes` object and
then constructs an `InstrumentedColdReader` does not prove routed I/O. The cold
test must launch a fresh decoder for one expert against a seekable,
instrumented reader. Every read/pread/mmap page fault exposed by the codec is
recorded before parsed state exists.

If installation-time authentication scans the whole object, report that cost
separately. Any persistent authenticated index/cache used to avoid that scan is
physical state: serialize it, charge its storage, and include its cold pages.

### E2. Exact page union

- Start each expert with an empty 4-KiB logical page cache.
- Trace the actual bytes needed for header, semantic metadata, model, addressed
  index/directory records, and owned frames.
- Count the exact union of touched pages; boundary pages count once.
- If the decoder reads an entire global directory or model, count it. Do not
  substitute a hypothetical addressed-page path.
- Decode only the routed expert and verify its Gate/Up/Down reconstruction.
- Repeat for every expert; report maximum unrounded amplification.

### E3. Exact denominator and anti-gaming report

For each physical interval with owner set `O`, allocate its byte count exactly
as the rational `bytes/|O|` to each owner. Global intervals use all `E`. Require
the exact allocation sum to equal total bytes.

Report both:

1. touched bytes / attributable **total physical** bytes; and
2. touched bytes / attributable **nonpadding decodable** bytes.

The strict gate uses the larger value, preventing unread rate-floor padding
from manufacturing a favorable read ratio. Owner-local padding placed inside
and actually read with a frame remains visible in both the trace and ledger.

The maximum must be strictly `<2.0`, not `<=2`, and not a rounded display value.
Warm-cache measurements are supplementary only.

### E4. Cold hostile tests

- Unequal private region lengths and shared tails.
- A directory spanning multiple pages.
- One frame beginning/ending one byte across a page boundary.
- Same payload hash in two differently owned frames.
- Rate-floor padding in global and owner-local positions.
- `E=128`, expert 127 ownership, and a multi-owner frame crossing bytes 7/8 and
  15/16 of the bitset.
- An attempted ledger that divides total bytes by `E` rather than exact owners.

## 7. Gate F — representative source-free RunPod/CuPy feasibility

The v2 four-stream/7,702-symbol microfixture is retained as a correctness unit
test but cannot satisfy this gate.

### F1. Representative fixture

Run on the supplied RunPod using the authenticated v3 snapshot and CuPy. The
fixture is deterministic and source-free:

- exactly 15 nonempty streams;
- exactly six semantic owners;
- at least twelve private streams and three shared-tail streams so every owner
  appears privately and in a cross-owner set;
- canonical owner bitsets and unequal stream lengths;
- production-representative total symbol count and length distribution, frozen
  from public shape/codec geometry rather than observed model symbols; and
- identities chosen so one complete outer fold has nonempty disjoint train,
  validation, and test subsets.

### F2. Work actually measured

At minimum execute one **complete** outer fold:

1. all 150 candidate count fits on inner train;
2. all 150 exact arithmetic scores on validation;
3. deterministic winner selection;
4. winner refit on complete development;
5. exact scoring on untouched outer test;
6. final full-panel fit, pack, literal parse, decode, and canonical re-encode.

Prefer all six folds. If only one is measured, project the exact six-fold,
four-shuffle, final-fit, and eight-control workload from counted symbol updates
using a conservative lower throughput bound. The receipt must show the exact
formula. Abort the family before payload if the frozen runtime budget is
exceeded.

Run a separate all-150 CPU/CuPy equality panel, including repeated GPU runs and
frequency extremes. A full production-size CPU replay is not required, but the
GPU performance fixture must produce repeatable integer hashes.

### F3. Exact transfer ledger

Count every host-to-device byte, not only initial streams:

- bits, levels, and base-frequency arrays;
- root offsets and lengths;
- subset index arrays and any host-created subset descriptors;
- every frequency/model table copied by `cp.asarray` or equivalent on every
  fit/score call, unless a proved cache avoids the transfer;
- scalar/descriptor packets copied through helper APIs; and
- final encoder/decoder GPU inputs.

Count device-to-host bytes separately, including count tensors, length arrays,
hash/result copies, and implicit `.get()`/`.tolist()` transfers. Record
device-to-device temporary traffic separately where practicable.

The receipt contains a symbolic expected-byte formula and an observed transfer
ledger. An independent profiler or audited transfer wrapper cross-checks them.
The v2 number that omitted repeated frequency-table transfers is an explicit
regression test.

### F4. Peak memory and runtime telemetry

- Synchronize CUDA at phase boundaries and after every timed kernel.
- Sample process-tree host RSS through the complete run; report absolute and
  incremental peak.
- Sample device process memory/free memory and CuPy default and pinned pools;
  report peak VRAM, not an instantaneous final value.
- Include JIT/compile time, warm-up time, measured work time, kernel count,
  count updates, length updates, candidate/fold count, and wall time.
- Record CuPy, CUDA runtime, driver, GPU name, compute capability, Python,
  platform, and authenticated source/fixture hashes.
- Repeat the representative measurement from a fresh process. Divergent integer
  outputs or material unexplained traffic/memory deltas fail.

## 8. Gate G — 128-expert and general portability

Before Qwen access, build and independently decode a source-free, non-Qwen
SwiGLU fixture with `E=128`, legal awkward `h,m`, all three roles, expert 127,
private and multi-owner tail frames, and a directory large enough to exercise
more than one page.

Required proofs:

- exact shape-derived weight count and scalar coverage;
- exact 16-byte owner bitsets for `E=128` and acceptance of bit 127;
- no expert identity used as a probability key;
- full expert ownership union with no empty expert;
- literal container parse/rebuild and independent per-expert routed decode;
- exact byte/owner conservation and cold trace; and
- no fixed six-expert, fifteen-stream, `768x2048`, twelve-coefficient, or
  Student-t `df=5` assumption on a universal decode path.

Also run a small `E=256` owner-ABI fixture to exercise the complete 32-byte
bitset. Scientific nested holdout may state a separate minimum fold count, but
container encode/decode portability must still support `E=1..256`.

Expert relabeling is a metamorphic test: permute expert ordinals together with
semantic routes and owner sets. Probability law and total coded result must be
equivariant up to the frozen canonical ordering; hidden expert/checkpoint keys
are a failure.

## 9. Gate H — source/control symmetry and scientific validity

### H1. Frozen selection procedure

- Candidate bank, topology/state/reset values, split digest, seeds, tie rule,
  model serialization, rate ledger, kill thresholds, and control seeds are
  frozen before source access.
- Outer identities are scientific partition labels only and never decoder
  probability keys.
- Shared streams are excluded from development whenever any owner violates the
  outer separation rule.
- Outer weight accounting uses exact per-owner contributions, not `weight/k`
  by assumption.
- Confidence calculations derive degrees of freedom from the actual valid fold
  count. A hard-coded six-expert `df=5` constant is permitted only in an
  explicitly panel-specific report layer, never the universal engine.

### H2. Matched controls repeat everything

Only after every absolute source gate passes, authenticate all controls before
the first control fit. Each control independently repeats:

- generation/transform/quantizer/extraction;
- complete 150-cell nested selection and deterministic tie handling;
- winner refit, model serialization, physical container pack;
- literal independent decode/re-encode, rate/`F`/cold ledger; and
- the same runtime/transfer accounting path.

Controls have the same shapes, semantic roles, ownership, source-weight
partition, public geometry, seeds policy, and model-byte charging. Each backend
and adapter starts without source-fitted state. Recompute and verify moment
matching rather than trusting a producer hash.

The source statistic and every control statistic are the same physical
quantity. Report source rank among controls and the finite-null resolution; do
not call `source > max(8 controls)` a conventional small-p significance result.
Controls can veto source specificity but cannot promote an absolute miss.

## 10. Gate I — external bootstrap and immutable source execution

### I1. New independent source audit

- Seal v3 into a new exact manifest with no payload authority.
- An independent auditor authenticates every declared member, rejects every
  undeclared member, reviews the actual v3 byte ABI, runs this hostile corpus,
  and emits a separately sealed PASS audit.
- The auditor explicitly verifies that all v2 blockers and every mandatory
  item above are repaired. A review written inside the producer closure or a
  public token is not authority.

### I2. Separately pinned dispatcher

The launcher is outside the producer package and hard-codes both the exact v3
manifest hash and independent PASS audit hash. Run isolated (`-I -B`) with a
controlled environment and import path.

It must:

- reject symlink/reparse-point leaves and every ancestor;
- hold and verify directory as well as file descriptor identities;
- enumerate the package through the held directory, not a replaceable path;
- open without following links, authenticate exact bytes, retain descriptors,
  and compile/import only those immutable bytes;
- prevent an unauthenticated `sys.modules`, `.pth`, `sitecustomize`, working
  directory, or repository-relative import from supplying producer modules;
- authenticate/inject the exact independent STRATA decoder and other non-system
  source dependencies;
- create an absent output directory and members exclusively, with no resume;
- verify every held input and source identity again after execution; and
- fsync files/directories and write `COMPLETE.json` exclusively last, after
  which producer write APIs are disabled.

Hardlinks, directory replacement, Windows junction/reparse points, package
renames, module-cache injection, and output-member races are hostile tests.

## 11. Gate J — fresh independent result audit

No producer result is self-authenticating. In a fresh isolated process, an
independent result auditor must:

- pin the v3 producer manifest, independent source audit, bootstrap receipt,
  universal decoder, input descriptor receipts, and completed result envelope;
- open the result directory and members through retained no-follow descriptors;
- use a byte-compatible decoder independently implemented from the producer,
  not import the producer parser or trust producer metrics;
- parse the literal container, deserialize its model, decode/re-encode every
  stream, rebuild it canonically, and recompute reconstruction;
- recompute original-source FP64 SSE/energy, actual physical rate, `F`, exact
  owner allocations, and actual instrumented per-expert cold page unions;
- verify source/control pipeline symmetry and all bound intermediate hashes;
- reject any files written after completion or any unlisted member; and
- emit a separate completion-last audit receipt whose manifest/hash is the only
  result authority.

The result audit reports identity-framing comparison as a diagnostic and the
literal final container as the sole promotion artifact.

## 12. Likely secondary v2 defects that v3 must not copy

These were not the headline four v2 blockers but deserve explicit regressions:

1. **Post-hoc cold tracing.** Parsing a complete in-memory container before
   tracing selected ranges does not demonstrate routed read bandwidth.
2. **Equal shared-weight attribution.** `weight_charge/owner_count` is wrong
   when a tail block contains unequal source contributions from its owners.
3. **Float canonical rebuild.** Inferring the rate floor from float bpw can add
   or remove a page at large sizes; layout/padding must be integer-canonical.
4. **Noncanonical duplicate regions/order.** A parser can accept multiple
   regions with one owner set or noncanonical frame ordering even if a later
   optional rebuild would differ.
5. **Incomplete scalar validation.** Zero symbols/bits, nonfinite scales,
   unbounded record products, and offset additions must reject in the parser,
   not be left to a downstream adapter or digest mismatch.
6. **Coverage-counter wraparound.** Narrow counters can turn repeated writes
   into apparent zero/one coverage on a generalized format.
7. **Hard-coded statistical geometry.** Six folds and `df=5` do not generalize
   to `E=1..256`.
8. **Unmeasured transfers.** Repeated model-table `cp.asarray` H2D, D2H result
   copies, advanced-index descriptors, and pools were absent from v2 telemetry.
9. **Path-based package/output enumeration.** Holding leaf descriptors is not
   enough if ancestor directories are replaced or enumeration reopens a path.
10. **Padding can game the denominator.** Unread globally attributed rate-floor
    padding can reduce an amplification ratio; report both total and nonpadding
    denominators and gate the stricter value.
11. **Header-bound score is not an independent score.** Identical reconstruction
    hashes are valuable, but final `D` still requires original-source BF16
    scoring in the result auditor.
12. **Eight controls have coarse null resolution.** The maximum-null comparison
    is a diagnostic/veto, not strong frequentist evidence by itself.

## 13. Minimum audit receipt contents

The independent preflight/audit receipt contains at least:

- exact v3 source manifest and independent audit hashes;
- all checklist gate IDs with PASS/FAIL and hostile-test counts;
- container ABI constants and owner-bitset definition;
- positive boundary fixtures and negative mutation hashes;
- 15-stream/6-owner RunPod fixture hash, exact workload, integer output hashes,
  repetitions, transfer ledger, peak RSS/VRAM, and environment;
- 128- and 256-expert portability fixture hashes/results;
- external bootstrap source/receipt hashes;
- explicit payload-access attestation; and
- one final `payload_authority` boolean that is true only if every mandatory
  pre-payload gate passed.

Until that receipt exists and is independently pinned, the only valid status is
`BLOCK_UNIVERSAL_LAUNCH_AND_PAYLOAD`.

