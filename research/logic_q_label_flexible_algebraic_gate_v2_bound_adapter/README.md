# LOGIC-Q v2 bound adapter

Date: 2026-09-02

Status: **sealed source-only successor awaiting an independent v2 audit**. It
has no Qwen/model payload path and grants no authority to open one. The frozen
v1 capped search package and its v0 codec dependency are authenticated member
by member before use and are not modified.

This package responds narrowly to the independent v1 disposition
`MECHANISM_VALID__HOLD_BOUND_SELECTOR_SCORER_AND_LIVE_BACKEND`. The audited v1
source root is
`5d145d89a20d2ae256ea60f569fab97cd6372cde66f7df75f3e86b08b3a88560`;
the v0 source root is
`2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a`.

## What v2 binds

1. **Selection is recomputed.** A selection receipt contains every canonical
   train/validation scored row and compact literal packet-header receipt. The
   authorizer revalidates partitions, packet/source bindings, derives pooled
   bits, counts, SSE and energy, recomputes validation `F`, and recomputes the
   selected frozen config. Changing only `selected_config_id` and publicly
   resealing now fails even if the attacker supplies the new self-hash.
2. **An external seal remains mandatory.** The panel and selection receipt
   hashes are explicit external inputs. A complete public rewrite and reseal
   cannot replace a previously pinned experiment. This is a commitment, not a
   claim that SHA-256 self-hashes sign their own contents.
3. **Counts and rate come from packet headers.** `packet_geometry` independently
   canonical-decodes the expert, emits literal expert/component headers, and
   derives each role count, total weights, packet bits and one-pass read ledger.
   Selection reparses those header bytes. Encoder-side label-count objects are
   never authoritative.
4. **Raw-source scoring is separate.** `independent_scorer.py` exposes no
   encoder entrypoint and accepts no weighted-SSE, source-energy, component or
   injected-array-backend metric object. It parses authenticated BF16/FP32/FP64
   source bytes, canonical-decodes the packet, and recomputes unweighted SSE
   and source energy in FP64.
5. **Live CuPy is launch-bound.** A name-only object and a bare module shell are
   rejected. The live receipt requires the canonical imported `cupy` module,
   its source-file hash, CuPy array type, active CUDA device properties,
   runtime/driver versions, a synchronized 4,096-element GPU arithmetic
   probe, and a nonce derived from the externally pinned panel, selection,
   config, layer, expert slot and shape. The receipt is recollected immediately
   before the frozen v1 live encoder runs.
6. **Physical expert closure remains exact.** Gate, Up and canonical
   Down-transposed shapes are equal; GF(2) gauge and ROMDD depth canonicalization
   remain inherited from byte-pinned v1; component alignment, expert header,
   final pages, `2.15 <= validation bpw <= 2.5`, and `1.0x` contiguous reads are
   enforced.

The compact packet receipt carries literal headers rather than duplicating a
multi-megabyte packet in every score row. Its expert/component SHA-256 values
bind the independently observed packet; the external selection seal commits
the complete row and packet-receipt set before test is opened.

## Frozen algorithm boundary

V2 changes orchestration only. It preserves v1's executable pre-family hard
kill and small scalable live bank:

- literal four-level labels;
- capped RM(1), two scales, eight words/plane, 64 exact word pairs, four retained
  pairs, and at most 16 exceptions per 256-weight block;
- nearest-scale-conditioned ROMDD depth 0/2/4/6 with at most 64 component
  exceptions.

It does not construct a full Qwen RM pair matrix or an unrestricted `O(N^2)`
exception search. GF(2), QTT, BDD, RM(2+), and uncapped algebraic families are
not scheduled and receive no global negative from this adapter.

## Critical STRATA boundary

This package is **not bound to the current STRATA six-pass 0..63 codec**. The
inherited v0/v1 mechanics reconstruct one of four profile levels per weight.
STRATA instead performs six complete level-major polar SC passes, polar
transforms each completed output plane, and assembles those planes into one
index in `0..63`. A coordinate-local four-level score, packet, or result cannot
be transferred.

[STRATA_RM6_ADAPTER_PLAN.md](STRATA_RM6_ADAPTER_PLAN.md) specifies the required
successor: one exact 64-way distortion table per coordinate, six completed
level-major planes, an explicitly budgeted RM/sub-RM bank, bounded CuPy
soft-decision/list search, canonical six-plane decode to 0..63 indices, and a
full literal packet/rate ledger.

## Source-only replay

From the repository root, with Python configured not to emit bytecode:

```bash
python -I -B research/logic_q_label_flexible_algebraic_gate_v2_bound_adapter/test_source_only.py
python -I -B research/logic_q_label_flexible_algebraic_gate_v2_bound_adapter/run_source_free_fixture.py
python -I -B research/logic_q_label_flexible_algebraic_gate_v2_bound_adapter/verify_source.py \
  --package research/logic_q_label_flexible_algebraic_gate_v2_bound_adapter \
  --expected-manifest-sha256 MANIFEST_SHA256
```

The hostile suite includes the three exact independent-audit attacks: changed
and publicly resealed config selection, fake encoder metrics/counts, and a
name-only CuPy backend. It also attacks complete receipt resealing, derived
aggregates, per-row count and partition bindings, panel external pins, packet
header counts, source bytes, SwiGLU shape closure, and launch context.

The fixture uses a synthetic 256x256 three-role expert. Its canonical literal
packet is 53,248 bytes for 196,608 weights: exactly 2.1666666666666665 bpw and
one contiguous read. These are mechanism numbers, not model evidence.

The independent GPU audit must additionally run the real CuPy receipt and v1
RM path on RunPod. Local source replay deliberately does not import or
initialize CuPy.

## Matched controls and payload authority

The inherited v1 package already freezes eight moment-matched controls and a
complete capped-pipeline rerun. A future payload experiment must regenerate
all eight from the authenticated source, run the same selected codec search,
and pass every resulting packet through this independent raw-source scorer.
Control receipts and the selection receipt require external pins. Controls are
diagnostic and cannot manufacture an absolute source pass.

No payload experiment is authorized by this source package. Qwen remains
closed until an independent auditor freezes this exact closure and separately
approves either (a) the abstract four-level research pilot, clearly labelled as
such, or (b) a real STRATA-RM6 semantic adapter. Only (b) can claim a current
STRATA codec result.

## Claim boundary

This package establishes source-only binding mechanics. It does not establish
a Qwen gain, `F <= 0.8`, below-Gaussian performance, a current STRATA result,
or a universal SwiGLU-MoE codec. Its 1.0x read ledger applies to its literal
synthetic page-contiguous expert object only; future runtime layout must replay
the ledger from actual bytes.
