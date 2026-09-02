# LOGIC-Q v3 authority successor

Date: 2026-09-02

Status: **sealed source-only authority design; no payload authority**.

This package is the narrow successor to the independent v2 disposition:

```text
MECHANISM_VALID__HOLD_PRODUCTION_PROVENANCE_BACKEND_AND_STRATA
```

It fixes the evidence boundary without enlarging the algebraic family.  The
frozen v2/v1/v0 four-level search remains byte-pinned and unmodified.  No Qwen,
model, current STRATA, coarse, or matched-control payload was opened while
building this source closure.

## What changed

V2 recomputed selection correctly over the rows supplied to it, but a public
self-seal did not prove those rows came from the independent scorer.  Its compact
packet receipt also omitted the scale and label/model payload bytes.  V3 removes
those inputs entirely.

The production selection entry point accepts only:

- an externally pinned precommit;
- literal train/validation raw source bytes whose hashes were in that
  precommit;
- the pinned worker path.

It has no parameters for scored rows, SSE, energy, counts, packet receipts,
packet geometry, an array backend, or a selected config.

For every frozen config and whole train/validation expert, the authority:

1. launches `gpu_worker.py` as `python -I -B` in a fresh temporary directory;
2. lets that child import CuPy before any repository module;
3. checks the precommitted CuPy module-file hash, version, device name, compute
   capability, runtime, driver, and synchronized 4,096-element GPU probe;
4. authenticates literal source files and runs the pinned v1 encoder;
5. reads the complete emitted inner packet itself;
6. canonical-decodes and re-encodes all inner bytes, including scales, family
   payload, headers, alignment, and final pages;
7. wraps it in a physical v3 authority envelope carrying config/source/context
   identity, the full inner hash and CRC, a whole-envelope CRC, and zero page
   padding;
8. reconstructs from that literal packet and recomputes raw unweighted FP64 SSE,
   source energy, count, physical rate, and `F` from the raw bytes;
9. selects only from those internally derived observations.

The content-bearing selection artifact embeds every authority packet.  Before
test launch, `authorize_selection` requires its independent external SHA-256,
reopens the original train/validation source bytes, re-scores every packet, and
reruns every frozen encoder invocation in a new child.  The packet and worker
receipt must match exactly.  This is intentionally expensive: it is the
authority path, not the eventual high-throughput encoder.

`run_selected_expert` accepts no config.  It first performs that authorization,
uses the recomputed winner, verifies the requested expert is in a whole held-out
test layer, and only then launches the winner.

## Two external commitments

The timing is explicit:

1. **Before rows or metrics exist**, externally pin the precommit.  It contains
   the complete panel/source hashes, public owner split, frozen grid, duplicate
   alias policy, worker hash, and CuPy/device policy.  Test is marked closed.
2. **Before test bytes are opened**, externally pin the complete content-bearing
   selection artifact produced by the authority.

Passing a newly fabricated hash at the same time as a fabricated object is not
an external commitment.  The surrounding experiment controller must record the
first pin before invoking selection and the second before granting test-file
access.

## Source alias policy

Duplicate source hashes are rejected by default, including aliases crossing
train, validation, or test owners.  If duplication is intentional, the
precommit must contain one complete group listing every component ordinal with
that digest and a nonempty reason.  Partial, overlapping, surplus, or
post-commit alias groups fail.

This closes content aliasing while preserving a narrow path for authenticated
weight tying.

## Physical authority packet

The v3 envelope is one contiguous 4 KiB-page-aligned object:

```text
binary prefix
canonical JSON authority header
complete canonical v1 expert packet
CRC32(prefix || header || inner packet)
zero page padding
```

The header binds:

- precommit and panel hashes;
- config, layer, expert slot, role source hashes, shape;
- inner byte length, SHA-256, CRC32, and decoded-geometry hash;
- fresh backend receipt hash;
- the pinned v1 source root;
- alphabet size four and `strata_compatible=false`.

All wrapper bytes and final page padding are charged in `physical_rate_bpw`.
An inner label mutation that remains a legal v1 packet is now detected by the
outer CRC/hash unless a different complete authority packet is deliberately
created; the external content-artifact pin then distinguishes that object.

The packet proves a one-object, one-fetch-capable storage layout.  It does not
measure runtime storage/HBM traffic.  Receipts therefore report:

```text
layout_addressable_read_amplification = 1.0
runtime_read_amplification_measured   = false
```

## Fresh CuPy boundary

Python cannot cryptographically attest itself against a hostile operating
system.  V3's narrower guarantee is a trusted-runner process boundary:

- `python -I -B` ignores `PYTHON*` and user-site injection;
- the child rejects CuPy already present in `sys.modules`;
- CuPy is imported before repository modules;
- only a small allowlist of CUDA/PATH environment variables is forwarded;
- the worker source hash is in the precommit;
- module-file hash and exact device/runtime fields are checked;
- a GPU ufunc probe is copied back and the stream synchronized;
- the authority consumes the packet file directly after child exit;
- authorization repeats the complete child invocation and requires byte
  identity.

This prevents the v2 name/module-shell and inherited-`sys.modules` attacks in
the intended trusted RunPod runner.  It is not remote hardware attestation.

## STRATA boundary

This source package still encodes one of four levels per weight.  It has no
semantic or rate bridge to current STRATA.

Current STRATA uses six complete level-major polar SC passes on `N=2^21` and
`N=2^20` blocks.  Completed polar output planes form a `0..63` reconstruction
index and causal Q0.16 branch decisions are arithmetic-coded.  A direct
`N=4096` six-plane RM/sub-RM packet is a separate proposed `STRATA-RM6` family.
Neither one is represented here.

Therefore a successful v3 four-level pilot would be evidence only for
label-flexible algebraic quantization on that abstract packet.  It cannot be
reported as a current STRATA gain or added to current STRATA results.

## Source-only replay

With Python and NumPy available beside the three frozen dependencies:

```bash
python -I -B research/logic_q_label_flexible_algebraic_gate_v3_authority/test_source_only.py
python -I -B research/logic_q_label_flexible_algebraic_gate_v3_authority/verify_source.py \
  --package research/logic_q_label_flexible_algebraic_gate_v3_authority \
  --expected-manifest-sha256 MANIFEST_SHA256
```

The hostile suite uses only a synthetic expert.  It attacks duplicate owners,
precommit resealing, raw-source bytes, outer CRC, inner payload hashes, page
padding, NaN scales, metric/packet/backend/config injection surfaces, fresh
worker replay requirements, and import isolation.  It never initializes CuPy.

A later independent RunPod audit must exercise the real worker with a synthetic
packet and freeze the literal backend receipt.  The source manifest deliberately
does not claim that replay has happened.

## Remaining holds

- No independent execution of this v3 closure has yet passed.
- No real v3 fresh-worker receipt has yet been produced.
- No matched-control and whole-test result container is implemented here.
- The physical authority wrapper is correctness-first and has not been optimized
  for large panel artifact storage.
- No current STRATA-RM6 semantic adapter exists.

Accordingly this package grants no payload authority.  Qwen must remain closed
until an independent source/backend audit promotes it, and even then any pilot
must be labelled abstract four-level.
