# Pre-seal hostile checklist for the SC unifilar census v2

Date: 2026-09-01

Status: source-only design red-team; **not** an audit receipt and **not** payload
authority.

This checklist was derived independently from the sealed v1 source, its
independent source audit, and the universal SwiGLU-MoE codec contract.  No v2
producer source was inspected.  No Qwen/model payload, current artifact,
extracted stream, or Gaussian-control payload was opened, statted, hashed, or
enumerated.

The v2 source should not seal until every mandatory item below has both an
implementation and a source-free hostile test.  Passing these source gates is
still only permission for a separately audited numeric run; it is not evidence
that Qwen contains the required entropy.

## 1. Exact scientific and codec scope

- [ ] Freeze the 150-cell bank, fitting rule, reset law, Q0.16 rounding,
  nested split, selection rule, control seeds, thresholds, container version,
  page size, and status lattice before payload access.
- [ ] Keep the standalone physical requirement
  `0.15288996696291447 bpw`; never substitute the nonfinite composite gap.
- [ ] State that a miss closes only the frozen selected-SC-decision recoder.
  It does not close arbitrary WFA/MPS, source-coordinate label copulas, MERA,
  TTN, RCC, or another lossy quantizer.
- [ ] State that a Qwen pass is panel evidence, not a universal performance
  guarantee.  Universality means the algorithm and byte format are
  model-agnostic.

## 2. Trusted launch and immutable source execution

- [ ] Do not execute the producer entrypoint directly.  Launch it through a
  separately reviewed dispatcher whose pinned closure includes the exact
  producer-manifest digest and exact independent-review digest (or verifies an
  auditor signature under a pinned public key).
- [ ] Do not accept a publicly reproducible self-seal plus a public token as
  independent authority.  A hostile test must fail for a fabricated
  `PASS_INDEPENDENT_SOURCE_REVIEW` receipt even when its internal SHA-256 is
  consistent.
- [ ] Authenticate the entrypoint and every imported producer module before
  any producer byte executes.  Read each source member once from a held
  descriptor, hash it, then `compile`/`exec` those exact authenticated bytes.
  `spec_from_file_location(...).exec_module(...)` on a pathname is forbidden.
- [ ] Keep the authenticated source bytes/descriptors alive through import.
  Replacing every source pathname after authentication must neither change
  executed code nor pass a later stability check.
- [ ] Load the CuPy backend from its authenticated buffered bytes, not by
  reopening its pathname after payload access.
- [ ] Reject undeclared package members and symlinks in any package path
  component.  The dispatcher itself must be externally pinned; otherwise
  self-verification has no root of trust.
- [ ] Use isolated Python (`-I -B`) and record the interpreter executable and
  version.  No repository-relative import, user-site import, or current-working
  directory import may enter the closure.

## 3. Descriptor-safe input and output paths

- [ ] Walk every absolute input path component with descriptor-relative
  `openat`/equivalent no-follow semantics.  Reject a symlink in the leaf or any
  ancestor.  Never call `resolve()` before the no-follow check.
- [ ] Hold the review, authorization, stream lock, artifact, extraction
  receipt, every stream member, control lock, and every control member through
  the run.  Bind results to the digest computed from the held descriptor; do
  not later call `sha256_file(path)`.
- [ ] Reserve an absent output directory and retain its directory descriptor.
  Create every member relative to that descriptor with exclusive creation.
  Renaming the directory and replacing its pathname with a symlink after
  reservation must not redirect any output.
- [ ] Fsync each completed member and the output directory.  Create
  `COMPLETE.json` exclusively and last, then fsync the directory again.
  Injected termination after every write boundary must leave an output that an
  independent verifier rejects as incomplete.
- [ ] A wrong token must touch no output or dynamic input and must import no
  project module or CuPy.  A valid token with any later failure may leave only
  an unmistakably incomplete, never-resumed directory.

## 4. Strict schemas, bounds, and panel geometry

- [ ] Reject unknown and duplicate fields, nonfinite numbers, booleans where
  integers are required, integer-like strings/floats, negative values, and
  values outside frozen resource bounds before allocation.
- [ ] Restrict partition identifiers to an unambiguous encoding, or hash a
  length-prefixed tuple.  NUL-separated unrestricted Unicode strings are
  ambiguous and may not define a split.
- [ ] Bind `weights` to parsed tensor shapes, not a lock assertion.  Bind every
  expert ordinal, role, stream, symbol count, weight charge, and immutable-byte
  range to the current artifact/extraction derivation.
- [ ] Require every declared expert to own the expected nonempty stream set;
  no empty or synthetic experts may amortize shared bytes.  Expert counts and
  shape-derived weights must agree exactly.
- [ ] Bound all counts and offset sums before conversion to uint32/uint64 or
  GPU allocation.  Prove no overlap, wrap, truncation, or out-of-range slice is
  possible.

## 5. Artifact, extraction, stream, and control binding

- [ ] One externally pinned authorization record must bind exact byte counts
  and SHA-256 values for the source lock, current physical artifact, extraction
  receipt/program, baseline score receipt, and Gaussian-control panel lock.
  An internally recomputable lock seal is insufficient.
- [ ] Parse the held current artifact.  Merely hashing and otherwise ignoring
  it is a hard failure.
- [ ] Independently derive, or independently verify a complete derivation of,
  `selected_bits`, `polar_level`, `regenerated_base_freq1`, stream boundaries,
  current logical lengths, and current payload bytes from that exact artifact.
  Arbitrary mutually consistent arrays plus an unrelated artifact must fail.
- [ ] Prove that every public context supplied to the WFA is decoder-visible
  from charged bytes.  `polar_level` and `base_freq1` may not be free input-lock
  side channels.
- [ ] Bind `current_object_bytes` to the held artifact descriptor size and bind
  the artifact scope to precisely the same experts and weights as the candidate
  container.
- [ ] Replay every current arithmetic stream byte-for-byte before importing
  CuPy.  A one-bit, one-length, one-context, one-stream-boundary, or one-byte
  mutation must fail.
- [ ] Bind each control to its declared generator/quantizer/extractor version,
  seed, source-lock digest, artifact digest, and full geometry.  Self-declared
  seed and geometry are insufficient.

## 6. Exact integer law and CPU/CuPy agreement

- [ ] The decoded probability law must be fully integer/fixed-point.  Reset
  state before the `t=0` emission; look up the causal frequency; encode/decode
  the symbol; transition only afterward.
- [ ] Serialize every fitted frequency and selector.  An independent decoder
  must deserialize them from bytes; no in-memory fit object may enter final
  acceptance.
- [ ] Independently test all 150 topology/state/reset cells, not four
  representatives, over reset boundaries, stream boundaries, all context
  bins, state extremes, frequency extremes, and multiple concurrent streams.
- [ ] Require exact CPU/CuPy equality for count tensors, fitted Q0.16 tables,
  per-stream logical lengths, and repeated GPU runs.  Synchronize after every
  kernel and surface asynchronous CUDA errors.
- [ ] Exercise the actual runtime CuPy/CUDA version and GPU architecture in the
  preflight, and record them.  The physical decoder must remain portable and
  independent of CuPy reduction order.
- [ ] Explicitly handle little-endian u16 input on the GPU or reject an
  unsupported host endianness before decoding.
- [ ] Retain exhaustive small arithmetic-code equivalence, causal decode,
  canonical re-encode, zero tail padding, and corrupt-model/payload rejection.

## 7. Complete literal container and standalone decoding

- [ ] Emit one versioned binary physical container containing every charged
  global header, checksum, shared model, directory, immutable global state,
  expert header, immutable local state, logical length, arithmetic payload,
  alignment byte, and rate-floor padding byte.
- [ ] Define a complete binary grammar with explicit endianness and checked
  integer widths.  The independent parser must reject truncation, trailing
  bytes, overlap, unsorted/duplicate entries, out-of-range offsets, nonzero
  reserved fields, noncanonical padding, checksum changes, and unknown
  versions.
- [ ] Expert payloads must be physically expert-contiguous (or their exact
  scatter pages must be counted).  Stream order in the source lock may not be
  silently reinterpreted as an expert-local layout.
- [ ] Reopen the emitted container in a fresh independent process, parse the
  serialized model, regenerate all contexts from container bytes, decode every
  arithmetic stream, and reproduce the selected-bit streams exactly without
  using the input lock or any producer in-memory object.
- [ ] Continue through the current universal decoder and reproduce the full
  current reconstruction, not merely the SC bit arrays.  The only allowed
  inputs are universal codec code, role/shape, and the emitted container.
- [ ] Generate `result.json` and all ledgers from a fresh parse of the emitted
  bytes.  In-memory predicted lengths are diagnostic only.
- [ ] Emit a same-framing identity/baseline recode or otherwise report
  separately (a) absolute saving versus the bound current artifact and (b)
  incremental WFA saving versus identical v2 framing.  Do not attribute header
  cleanup or dead-tail removal to the source model.

## 8. Reconstruction, rate, and F adjudication

- [ ] Bind the audited current relative MSE, normalization, weight count,
  reconstruction digest, artifact digest, and original-source panel in one
  score receipt.
- [ ] Independently prove the candidate reconstruction is bit-identical to the
  bound current reconstruction, or independently rescore both against the
  exact bound source weights.
- [ ] Calculate directly from authoritative objects:
  `R = 8*len(container)/N` and
  `F = relative_MSE * 2**(2*R)`.  Do not infer F only by multiplying a
  hard-coded `CURRENT_FINITE_F` by a claimed saving.
- [ ] Require `2.15 <= R <= 2.5` and `F <= 0.8` with a frozen high-precision
  numerical rule.  Record the exact integers and sufficient precision to
  reproduce each comparison.
- [ ] Treat physical Qwen failure as failure regardless of controls.  Controls
  may reject source specificity but may never turn a failing physical packet
  into a pass.

## 9. Exact cold-read accounting

- [ ] Classify every physical byte in the container as shared `S` or private
  to exactly one expert `L_i`; alignment and checksums are included.
- [ ] Use expert-specific amortized storage
  `S/E + L_i`, not `total_container_bytes/E`, as expert `i`'s denominator when
  private frame lengths differ.
- [ ] Decode one expert through an instrumented cold reader and record the
  exact union of 4096-byte page indices touched.  The numerator is the bytes of
  those physical pages, including all shared pages and any neighboring expert
  bytes brought in by an unaligned page.
- [ ] Derive offsets/page sets from the literal container, not a hypothetical
  ledger.  An independent implementation must reproduce every page set.
- [ ] Require
  `max_i(cold_page_bytes_i / (S/E + L_i)) < 2.0`.  Report warm-cache figures
  separately and never use them for the gate.

## 10. Nested holdout and matched-null adjudication

- [ ] For each outer `(layer, expert)` cell, exclude every stream sharing that
  layer **or** expert from development.  Select all hyperparameters on inner
  training/validation only, refit on development, and score the untouched
  outer test.
- [ ] Keep ephemeral fold models scientific only.  Never represent their
  pooled score as one deployable identity-keyed packet.
- [ ] Authenticate semantic layer/expert labels from artifact geometry; lock
  authors may not invent partitions.
- [ ] Charge the full selected model in every fold, and report every fold,
  pooled gain, minimum gain, and a predeclared whole-expert/layer confidence
  interval.  Millions of symbols are not independent samples.
- [ ] Each of the eight Gaussian controls must match source expert count,
  shapes, weights, stream order/counts, symbols, partitions, immutable bytes,
  current physical geometry, and quantizer/extractor law exactly.
- [ ] Run the **same complete 150-cell nested selection, refit, packing, and
  ledger pipeline independently on every control**.  Merely refitting the
  source-selected winner gives the source an unmatched winner-search advantage.
- [ ] Run predeclared within-role/stratum permutations and multiscale shuffles
  if a source survivor is used to claim nonlocal structure.
- [ ] Authenticate and byte-replay all control baselines before the first
  control candidate fit.  Source gates must pass before controls are opened.
- [ ] Freeze the specificity statistic and tie behavior.  Report both
  `G_operational(source)` and source minus the strongest matched-null gain.

## 11. Runtime and memory feasibility

- [ ] Report the honest complexity as
  `O(outer_folds * candidates * selected_symbols)` plus final coding; do not
  call the complete nested protocol simply `O(N)`.
- [ ] Run a source-free full-150-cell benchmark with representative stream
  count, lengths, reset boundaries, and at least one complete synthetic outer
  fold.  Freeze a maximum projected wall time and peak host/VRAM budget before
  payload access.
- [ ] Use authenticated metadata to compute the exact number of
  cell-symbol updates and a conservative runtime projection before fitting.
  If it exceeds the frozen budget, stop without issuing a scientific kill.
- [ ] Reuse packed GPU arrays and offsets across cells/folds.  Rejoining all
  bytes, copying the panel host-to-device, and expanding every byte into Python
  integer lists inside every candidate fit is not accepted without a measured
  full-scale feasibility proof.
- [ ] Record peak host RAM, peak VRAM, H2D bytes, kernel count, cell-symbol
  updates, elapsed fit time, final CPU/independent-decode time, and any timeout.

## 12. Status lattice and independent result audit

- [ ] Positive status must be a single boolean conjunction of literal physical
  target, strict cold-read target, nested heldout gate, all matched controls,
  standalone decode, and all integrity checks.  Add hostile unit tests that
  force each individual gate false and prove promotion becomes impossible.
- [ ] Distinguish physical/F failure, cold-read failure, heldout failure,
  specificity failure, runtime abort, and evidence-integrity failure.  Do not
  label all failures as the same scientific hard kill.
- [ ] Seal exact source, design, review, authorization bindings, container,
  result, page trace, environment, and completion manifest.  An independent
  result auditor must reparse the literal container and recompute rate, F,
  reconstruction equality, and cold pages without importing producer code.
- [ ] No universality claim is promoted from Qwen alone.  Before that claim,
  run a disjoint SwiGLU-MoE family or at minimum retain only the narrower
  model-agnostic-codec statement plus different-shape source-free portability
  fixtures.

## Additional v1 defects that motivate the new gates

These are independent of the six blockers already recorded by the v1 audit:

1. **No trusted authority root.**  The producer entrypoint executes before it
   authenticates itself, while a public token and self-sealed JSON can fabricate
   a passing review.
2. **Unproved decoder-visible contexts.**  The artifact and extraction receipt
   are held but ignored; mutually consistent bit/level/frequency arrays can be
   unrelated to the artifact, and final decode receives level/frequency arrays
   for free.
3. **Unbound score law.**  A hard-coded `CURRENT_FINITE_F` is not tied to the
   held artifact's size, MSE, reconstruction, or weight count.
4. **Incorrect unequal-expert cold denominator.**  `total/E` is not
   `S/E + L_i`, and the hypothetical per-expert layout is not the emitted
   payload order.
5. **Asymmetric null search.**  The source gets a 150-cell nested winner search;
   controls get only a refit of that winner, contrary to the frozen requirement
   to repeat the complete fit and selection.
6. **Output-path TOCTOU and durability gap.**  Output writes reopen a mutable
   pathname; neither a held directory descriptor nor directory fsync protects
   completion.
7. **Review-path TOCTOU.**  The authenticated review is closed and later
   rehashed by pathname for the result receipt.
8. **Weak production GPU preflight.**  Four representative cells cannot prove
   all 150 runtime laws on the actual driver/runtime.
9. **Unbounded/adversarial metadata.**  permissive `int(...)` conversion,
   ambiguous NUL-delimited split identifiers, unused experts, and unchecked
   size conversion permit geometry or resource manipulation.
10. **Unproved full-scale feasibility.**  v1 repeatedly repacks and transfers
    subsets for each cell and expands large arrays into Python integers; only a
    fixed-cell microbenchmark was measured.

