# FUSEED-PMG1-v2 source-only design draft

Status: **conditionally defensible design; execution blocked; no implementation
authorization**.

This draft narrows FUSEED to the one already-frozen
`CURRENT_PMG_GATE_UP_DIRECT_BF16` compatibility ABI. It exhausts every u32 base
seed with one exact binary64 source objective. It does not create a new
coordinate namespace: stage 0 is exactly the ABI1 subset of the frozen v1 plan,
and the common stage1, full-selection and validation sets are reused byte for
byte. `verify_design.ps1` independently regenerates those plans with only
PowerShell/.NET standard libraries, emits the complete category-ordered bundle
array on request, and binds the parent v1 design, verifier, source provenance,
numeric semantics and independent audit blockers through `source_bindings.json`.

## Scientific verdict

Moving controls out of the exhaustive search is defensible only because the
claim changes. The 32 controls are no longer matched searches and cannot be
used as a p-value or as the mechanism that corrects the 2^32-way selection.
Instead, multiplicity is handled by a strict firewall: selection sees only the
selection panel, commits exactly one winning descriptor, and cannot retry;
only then is an untouched validation panel made visible. The controls run on
that same one frozen descriptor as a conservative one-shot falsification and
capture subtraction. Any source or control failure is terminal.

This logic becomes **BLOCK** if the ABI choice was informed by target scores,
if the validation panel was previously used to tune any part of this protocol,
if a control searches seeds, or if a failed validation can trigger a retry.
The draft therefore requires an independent chronology and validation-firewall
attestation before it can be frozen.

## Frozen hypothesis

- Candidates: exactly `base_seed = 0..2^32-1`, once each, for one ABI.
- Seed map: `base_seed + 1024 + 100*(expert//32)` with full u64 carry.
- Stage 0: 1,024 inherited coordinates, 512 fit and 512 score, arranged as
  `up_fit`, `down_fit`, `up_score`, `down_score` while preserving each inherited
  four-lane bundle.
- Objective: decoded-FP16-affine source capture computed wholly in binary64
  under one fixed operation and accumulation order. There is no FP32 shortlist
  or numerical interval screen.
- Total order: capture descending, then unsigned seed ascending.
- Cascade: exact Top-8192 from stage 0, exact Top-256 after 4,096 coordinates,
  then one winner after 48,624 coordinates.
- Validation: one descriptor, 16,912 fixed coordinates on four disjoint
  experts; source first, then 32 fixed-descriptor controls without changing the
  descriptor.

The exhaustive stage generates `4,398,046,511,104` normal values. The complete
maximum including later selection and validation is `4,398,092,530,192`.
One shard's binary64 score buffer is 128 MiB. All 256 packed shard Top-K files
together are 24 MiB; each record is exactly 12 bytes with no padding.

## Why this is a plausible breakthrough

The v1 cost came from three ABI families and 33 independently scored exhaustive
domains. PMG1-v2 searches one public-source-motivated ABI and one exact source
objective. Existing source-free timing evidence projects its kernel component
near 512.63 seconds, below the unchanged 900-second v1 wall limit. That
observation is feasibility evidence only: it omitted the final exact plan,
packed Top-K, journal, global merge and hardened identity receipts, so this
draft requires a new complete-shape calibration and independent audit.

## Audit repairs required before any run

The hardened calibration must close every v1 audit blocker. Its receipt binds
the driver/runtime/compiler binaries and hashes, ordered compiler options,
transitive headers, derived source, compiled intermediate and loaded executable
bytes, launch attributes and the allowlisted environment. Both reference
constructions—shifted offset with one vector draw and original offset with
`j+1` draws—must match the direct path bitwise in three independent
repetitions. Initial and terminal generator-state serialization is mandatory.
The timed path must use this exact emitted plan and include exact Top-K,
durable journaling and the global merge.

Journal closure is prospective and crash-safe. The output root must be absent,
all raw/canonical roots, component identities, file IDs, device/mount IDs and
ancestries are checked before the first journal write, direct entry without the
sealed launcher capability refuses, files are flushed and atomically renamed,
and two legal global merge trees must emit identical final bytes.

## Physical rate and read ledger

The conditional final wire remains conservative: an eight-byte global
descriptor plus 18 four-byte FP16 affine pairs, 80 bytes over 28,311,552
weights. That is `0.000022605613425925925 bpw`. The compatible base-codec caps
are `2.149977394386574 bpw` at total 2.15 and
`2.499977394386574 bpw` at total 2.5. Reading the global descriptor plus three
affines costs 20 metadata bytes per expert; the inherited worst cold-read
amplification becomes `1.1694597713582042x` at 2.15 bpw and
`1.1694575633680555x` at 2.5 bpw, safely below 2x.

This ledger is not yet bookable. An Up/Down survivor still needs one frozen,
one-shot Gate-half confirmation to supply all 18 affine cells.

## Verification

From the repository root:

```powershell
pwsh -NoLogo -NoProfile -File research/fuseed_pmg1_v2_design_draft/verify_design.ps1
```

Add `-PrintPlanFacts` for the compact digest ledger or `-PrintPlanJson` to emit
the complete ABI1 category-ordered bundle array and all reused coordinate sets
to standard output. The verifier never imports a third-party package and does
not open any target payload, model manifest, runtime device, or network
resource.

The verifier is producer self-checking, not an independent audit and not a
launch token.
