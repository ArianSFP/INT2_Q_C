# Integrated unifilar WFA entropy census v2 — sealed source only

This package is the repaired producer for the decisive selected-SC-decision
source-model census. It asks whether one universal, frozen bank of 150 sparse
unifilar laws can recode the existing STRATA arithmetic decisions by at least

```text
0.15288996696291447 physical bits/source-weight
```

while preserving the full decoded reconstruction, keeping the actual rate in
`[2.15,2.5]` bpw, and keeping every routed expert's literal 4-KiB cold read
strictly below `2x` its owner-attributed physical storage.

This is source-only infrastructure. It contains no Qwen/model payload,
current artifact, extracted decisions, Gaussian control, numeric result, or
payload authority.

## What v2 changes

The v1 independent audit found six binding defects. The v2 producer removes
each one:

1. Project modules are compiled from externally authenticated byte snapshots;
   direct producer execution is always rejected. A later independently pinned
   dispatcher, outside this package, must hard-code both the manifest and
   review digests.
2. Every input path component uses no-follow descriptor walking, and output
   members are written relative to a retained directory descriptor.
3. The baseline artifact byte count is taken from the held literal artifact,
   which is semantically parsed and replayed. There is no lock-provided
   `current_object_bytes` field to inflate.
4. Final acceptance deserializes the transmitted Q0.16 model. The model tensor
   has an internal checksum and the complete container binds it again.
5. One literal container emits its 4096-byte header, inherited STRATA metadata,
   page-rounded model and directory, owner regions, frame headers, payloads,
   alignment, and rate-floor padding. Rate, F, and pages are recomputed from a
   fresh parse of those bytes.
6. Every Gaussian control repeats artifact replay, the complete 150-cell
   nested selection, refit, physical packing, and standalone decode. Refitting
   only the source winner is forbidden.

The design red-team found additional issues in v1. V2 also binds audited SSE,
source energy, relative MSE, weight count, original-source panel,
reconstruction digest, baseline artifact, universal decoder, source manifest,
external bootstrap, and extraction program. Unknown schema fields, permissive
integer conversion, ambiguous partition strings, empty experts, and unbounded
counts are rejected.

## Decoder-visible context, without extracted arrays

`strata_sc_adapter.py` runs the frozen universal polar decoder. For each
selected decision, the polar recursion regenerates the original conditional
frequency and calls an arithmetic-like adapter:

```text
set_level(level)
decode(original_freq1):
    reset state before t=0 when required
    context = (level, prior_bin(original_freq1), t mod 4)
    uwfa_freq1 = transmitted_table[state, context]
    bit = new_arithmetic_decoder.decode(uwfa_freq1)
    state = frozen_transition(state, bit, context, t)
    return bit
```

Consequently, `selected_bits`, `polar_level`, and
`regenerated_base_freq1` are never decoder side files. They are regenerated
from the literal inherited 128-byte semantic header, 144-byte route table,
5,184-byte labels, fifteen profiles/scales, the new payload, and universal
code. The adapter then performs inverse RHT, group restoration, and inverse
KLT and binds the complete matrix-ordered FP64 reconstruction digest.

## Physical and read accounting

The binary grammar is implemented in `container_codec.py`. The global header
is exactly 4096 bytes. Inherited metadata, model, and directory occupy distinct
page-rounded shared ranges. The twelve private blocks form six expert-local
regions; the three tail blocks remain pair-owned and are not duplicated.
All frames are 64-byte aligned and all owner regions page aligned.

For expert `e`, storage attribution is

```text
G/E + sum(B_region / number_of_region_owners)
```

over regions owned by `e`. The numerator comes from an instrumented reader's
exact union of literal 4096-byte pages. `total_container_bytes/E` is never used
for unequal regions.

The authoritative calculations are

```text
R = 8 * len(literal_container) / shape_derived_weight_count
F = independently_audited_relative_MSE * 2**(2*R).
```

The producer separately reports absolute saving versus the bound current
artifact and incremental saving versus a same-framing counterfactual containing
the original arithmetic payloads and the identical model/framing charge.

## Scientific protocol

- Candidate bank: five fixed topologies × `chi={2,4,8,16,32,64}` × reset
  `{32,128,512,2048,4096}` = 150 cells.
- Fitting: exact uint64 counts, Jeffreys half-count smoothing, deterministic
  Q0.16 nearest rounding, clamp to `[1,65535]`.
- Outer folds: one semantic `(layer,expert)` from the parsed route; development
  excludes every stream sharing that layer or expert.
- Inner selection: frozen length-prefixed SHA-256 rank split, full model charge,
  exact byte-padded arithmetic score, ordinal tie break.
- Heldout evidence: all folds, owner-attributed pooled saving, minimum fold,
  and a predeclared whole-expert Student-t interval.
- Survivor diagnostics: one exact within-public-context permutation and
  multiscale chunk shuffles at 32, 128, and 512, each repeating the complete
  150-cell nested search.
- Controls: eight frozen seeds, all artifacts replayed before the first control
  fit, exact geometry/provenance/pipeline matching, and the full source search
  independently repeated per control.

The honest complexity is
`O(outer_folds * 150 * selected_symbols)` plus final coding, four survivor
shuffle searches, and eight matched-control searches. The CuPy backend packs a
panel once, forms descriptor-only subsets, synchronizes every kernel, and
records H2D bytes, kernel count, cell-symbol updates, runtime, and VRAM.

## Root of trust and launch state

Running `stage0_census.py` directly always exits with status 2 before parsing
arguments or touching inputs/output. `dispatcher_contract.py` is a testable
reference, not an authority root. The required independent bridge ABI is in
`INDEPENDENT_BOOTSTRAP_ABI.md`.

After this source seal, four gates remain mandatory:

1. an independent audit of an externally pinned dispatcher;
2. all-150 CPU/CuPy equality on the actual RunPod runtime;
3. no payload launch until both pass; and
4. a fresh-process independent result audit after any numeric run.

## Claim boundary

A negative result closes only this frozen selected-SC-decision recoder. It does
not close arbitrary WFA/MPS, a source-coordinate label copula, MERA/TTN,
nonlinear flow, RCC, Gray-Wyner, or another lossy quantizer.

A Qwen pass is panel evidence only. “Universal” here means the algorithm,
fitting rule, transmitted format, and decoder are model-agnostic across
SwiGLU-MoE shapes; it is not a performance guarantee until disjoint model
families are evaluated.
