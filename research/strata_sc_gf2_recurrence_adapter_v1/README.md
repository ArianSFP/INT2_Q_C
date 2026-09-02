# STRATA selected-SC GF(2) recurrence adapter v1

## Disposition

This package is a **source-only, nonpromoting packet-mechanics pass** and a
**production launch hold**. It has not opened, statted, hashed, enumerated, or
decoded Qwen, STRATA, UWFCV8, identity, or control payloads.

Freeze note: the core/hostile suite and source verifier passed 9/9 on RunPod
against manifest SHA-256
`7588d1006a6c83e5be704b5fb4f1e53a338603bfbc72f774e948bbe8b67617e3`.
The payload-free bound module and its tenth test were added afterward. SSH then
became unavailable, so the current freeze is explicitly
`PENDING_FINAL_10_TEST_RERUN`; the new bound has not been represented as an
executed test result.

A passing UWFA-SC v9 result-audit receipt is necessary but **not sufficient**
to launch. Publication access additionally requires:

1. an independently supplied SHA-256 pin for the completed v9 audit receipt;
2. a separately frozen extraction/replay implementation that opens the pinned
   publication only through `publication_gate.read_publication_member`;
3. a separately pinned independent replay receipt accepted by `replay_gate`;
4. an exact pre-payload or first-open rate-floor check; and
5. a launch seal naming all source and receipt hashes.

The extraction/replay implementation in item 2 does not exist in this package.
Consequently the adapter **cannot lawfully launch merely because the running
v9 audit passes**. `replay_gate.py` is a fail-closed receipt verifier, not a
substitute for the missing independent decoder.

## What is encoded

This is not the earlier four-level literal-label BM packet. It targets the
current STRATA selected successive-cancellation decisions, preserving their
six level-major segments. Each level is deterministically split into at most
4,096 decisions. A chunk is represented by either:

- its literal bits (`n` payload bits); or
- the canonical GF(2) Berlekamp-Massey recurrence and initial state (`2L`
  payload bits).

The LFSR form is selected only when `2L < n`; ties are literal. Every chunk is
SHA-256 and CRC protected and rejected unless canonical re-encoding is
byte-identical. Each expert packet preserves exact FP16 scale bits, profile,
SC/RHT seeds, opaque decoder state, semantic state, all six boundaries, source
commitments, and the audited current/candidate arithmetic-payload identities.
Shared streams are duplicated into every owning expert packet and charged.

The packet itself is not enough to prove identical reconstruction. The missing
independent replay must feed packet-derived decisions through the frozen
STRATA SC traversal, regenerate every Q0.16 base-frequency context, recompute
each `(decisions, levels, base frequencies)` digest, canonically reproduce the
audited UWFCV8 arithmetic payload, and reproduce the pinned full-FP64
reconstruction digest. The production scorer refuses to run without a receipt
for exactly those operations.

## The decisive rate accounting

The number of selected SC decisions is **not a bit rate**. STRATA
arithmetic-codes those decisions; the sum of six level capacities can exceed
four decisions per source weight. Comparing that literal count with a two-bit
four-level label plane is invalid.

The audited physical comparators are instead:

- current STRATA object: `8,847,360` bytes / `28,311,552` weights = exactly
  `2.5 bpw`;
- UWFA-SC v9 candidate: `8,892,416` bytes / `28,311,552` weights =
  `2.51273148... bpw`.

The sealed v9 source also pins `126,627,266` selected decisions in 15
streams: `4.47263598` decisions/weight, again not a rate. Without opening a
payload, `prepayload_rate_gate.py` derives at least 30,915 unique chunks. Even
granting away every recurrence payload, all metadata, page padding, and
expert-local shared-stream duplication, the bare catalog/expert/chunk-header
floor is `2,478,832` bytes (`0.70044397 bpw`). It fits, so the grammar itself
cannot yet be killed. Conversely, adding one raw bit per unique decision gives
an optimistic raw floor of `18,307,241` bytes (`5.17308016 bpw`), already
`9,459,881` bytes over the 2.5-bpw object. The raw exact-decision branch is
therefore hard-killed before payload access.

Only an unusually low-complexity recurrence branch remains plausible. On the
same deliberately optimistic accounting, fitting 2.5 bpw requires aggregate
BM complexity no greater than `25,474,112`, mean complexity no greater than
`824.005` per minimum chunk, and at most `0.402348` recurrence payload bits per
unique selected decision. Metadata, alignment and expert duplication make the
true thresholds stricter.

For recurrence packets, the scorer charges the 4,096-byte catalog and all
page padding. With `C_e` chunks and canonical metadata `M_e`, the unconditional
zero-complexity physical floor is

```text
4096 + sum_e align4096(align64(256 + M_e) + 80*C_e).
```

The raw fallback adds `ceil(n_chunk/8)` bytes for every chunk. The actual form
adds `ceil(2L_chunk/8)` for an LFSR chunk or the raw bytes otherwise. These are
literal packet bytes, not entropy estimates. Before any positive recurrence
claim:

- if the zero-complexity floor exceeds `8,847,360` bytes, hard-kill;
- if the actual packet exceeds `8,847,360` bytes, hard-kill at the 2.5-bpw cap;
- compare actual bytes directly with both audited arithmetic objects;
- charge model and exception bytes. They are exactly zero here.

An exception/discrepancy-coded recurrence model could be useful, but it is a
separate **unimplemented** branch. This exact codec encodes no exceptions.

## Bounded-negative scope

The frozen 4,096-decision chunk cap bounds compute and random access, but it can
miss a recurrence spanning a boundary or having order above a chunk. Therefore
a negative result applies only to **independent per-level BM recurrences under
this exact chunking**. It does not close longer recurrence, streaming BM,
multisequence, or exception-coded hypotheses.

The 128-bit owner mask is also Qwen-shaped pilot plumbing. This packet grammar
is **not** a universal SwiGLU-MoE format. A universal successor must serialize
and charge variable expert cardinality and a variable-length ownership
descriptor, then re-audit page layout and routed reads.

## Cold-read claim boundary

Each expert packet is page-aligned and contiguous, and the derived ledger plans
one read with zero refetches. This is a layout statement only. Runtime I/O has
not been measured. The scorer reports amplification against that expert's
current 2.5-bpw weight-byte budget and never calls it an observed runtime
measurement.

## Source-only verification

On the RunPod:

```bash
cd /workspace/INT2__compression/INT2_Q_C/research/strata_sc_gf2_recurrence_adapter_v1
/workspace/int2-cupy-venv/bin/python -I -B prepayload_rate_gate.py
/workspace/int2-cupy-venv/bin/python -I -B test_source_only.py
/workspace/int2-cupy-venv/bin/python -I -B verify_source.py
```

The tests exhaustively compare BM with brute-force minimum recurrence through
length nine, exercise LFSR and random fallback, CRC/SHA/canonicality, all six
levels, exact scale/state retention, invalid ownership/role coverage, catalog
binding, external audit/replay capabilities, packet-derived scoring, and the
source-pinned pre-payload raw-rate hard-kill. They use source constants and
synthetic bytes only.
