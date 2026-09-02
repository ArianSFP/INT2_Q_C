# LOGIC-Q v1 capped adapter

Date: 2026-09-02

Status: **source-only successor, not authorized for Qwen until an independent
source audit pins this package**. It preserves the audited v0 package byte for
byte and authenticates every v0 member before importing its finite packet
mechanics.

`PREFLIGHT_HISTORY.json` preserves the provisional fail-closed verifier result
caused by a generated `__pycache__`; that failure is not silently replaced by
the clean retry. The final seal must be replayed from a fresh exact-member
staging root.

## What this repairs

The independent v0 audit found a valid synthetic mechanism but five production
holds. This successor turns those holds into executable invariants:

1. A mixed-family-safe pooled lower bound is executed before a family search
   callable. A killed family cannot accidentally run. The two other roles are
   granted impossible-best minimum packets, so this is not the unsafe
   uniform-family bound.
2. RM1 never constructs the complete affine-word pair matrix. Each Gray plane
   retains eight affine words under a separable surrogate, evaluates the 64
   resulting pairs under exact four-level distortion, and sends only the best
   four into a capped exception optimizer. Exceptions are capped at 16 per
   256-weight block. The optimizer is `O(n log n + cap^2)`, not an unrestricted
   quadratic prefix enumeration.
3. Each RM block retains two BF16 scales from a frozen weighted-nearest grid and
   jointly scores every retained scale with the structured labels. ROMDD is
   explicitly **nearest-scale-conditioned** and can produce only a bounded
   negative. No miss is promoted to a global algebraic conclusion.
4. Success is scored only from one page-padded three-role expert packet. Gate,
   Up and canonical Down-transposed shapes must be identical. All component
   headers, scale words, payload padding, 64-byte alignment, expert header,
   final 4 KiB page, and deterministic padding to at least 2.15 bpw are charged.
5. GF(2) packets use a unique pivot-column rank factorization (with
   deterministic zero padding to the declared rank), ROMDD's header depth is
   recomputed from its serialized graph, and the strict decoder demands an
   exact canonical re-encode.
6. The train/validation selector is executable. It accepts no test metrics,
   binds the frozen grid, full panel/source hashes and canonical component
   ordinals, and emits a sealed receipt before test authorization.

## Deliberately narrow live bank

Only three methods are scheduled for one full expert pilot:

- literal four-level labels;
- capped, label-flexible RM(1) plus capped exceptions;
- depth-0/2/4/6 ROMDD plus at most 64 whole-component exceptions.

GF(2) search is not scheduled. Its v0 raw-factor alternating optimizer is not a
credible Qwen-scale bounded search. v1 still canonicalizes and independently
decodes GF(2) packets so gauge aliases cannot pass a packet audit, but it makes
no negative claim about low-rank binary label planes. BDD, QTT and higher-order
RM families are outside this adapter for the same reason: they need separately
frozen scalable algorithms, not an uncapped family name.

The RM search is a bounded heuristic. Its miss does not kill RM codes globally.
ROMDD's nearest-first scale condition makes its miss narrower still.

## Production backend and read ledger

The module never imports CuPy during source verification. A live call passes a
backend module and must set `live=True`; the adapter then rejects any backend
whose name is not `cupy`. The scale census, affine shortlist and exact pair
scoring use that injected backend. Packet emission and canonical replay remain
deterministic finite operations inherited from the pinned v0 codec.

The routed representation is one contiguous expert object and is read once.
The recorded storage read is therefore exactly the physical object size, for
`1.0x` cold-read amplification. Scratch/HBM traffic is not relabeled as storage
traffic. Any implementation that fetches the compressed expert a second time
must produce a new ledger and would not inherit this result.

## Holdout and controls

The canonical panel requires at least ten layers, identical expert-slot
universes, and one fixed SwiGLU shape cohort. Five or more whole layers are the
outer test set. Among non-test layers, whole expert slots form validation and
are absent from training. Splits use only public layer/slot identifiers; source
hashes are bound before test but do not choose a split.

The eight frozen Gaussian controls are regenerated from the immutable source.
For every public 256-weight block they match binary64 mean and centered energy.
The complete pipeline is rerun, including pre-search decisions and every
family search they authorize, scale/label decisions, component selection,
packet framing and the pooled expert score. A
control comparison is diagnostic only; it cannot create an absolute source
pass.

## Source-only commands

From the repository root:

```bash
python -I -B research/logic_q_label_flexible_algebraic_gate_v1_capped_adapter/test_source_only.py
python -I -B research/logic_q_label_flexible_algebraic_gate_v1_capped_adapter/run_source_free_fixture.py
python -I -B research/logic_q_label_flexible_algebraic_gate_v1_capped_adapter/verify_source.py \
  --package research/logic_q_label_flexible_algebraic_gate_v1_capped_adapter
```

The 33 hostile tests include a bomb in the v0 full RM pair-matrix routine,
unbounded-cap rejection, scale/label co-search, GF(2) gauge aliases, ROMDD
depth tampering, role-shape mismatch, exact expert re-encode, selection/test
leakage, canonical ordinals, matched moments, and all eight capped-pipeline
control reruns.

## Claim boundary

This package contains no model path or payload adapter and has not opened Qwen
weights. It establishes finite source mechanics and audit closure only. It does
not claim `F <= 0.8`, a real model gain, or universality across SwiGLU-MoE
families. A Qwen result can be run only after an independent source audit; a
universal claim additionally requires a separately frozen shape cohort and a
disjoint non-Qwen SwiGLU-MoE transfer panel.
