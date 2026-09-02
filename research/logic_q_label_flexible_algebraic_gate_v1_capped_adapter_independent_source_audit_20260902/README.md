# Independent source audit: LOGIC-Q v1 capped adapter

Date: 2026-09-02

Audited v1 manifest:
`9bfd3d1225fb45a0518d2d4d6a4035262e87dc62563222e42e69665358b9aac5`
with source root
`5d145d89a20d2ae256ea60f569fab97cd6372cde66f7df75f3e86b08b3a88560`.

Pinned v0 dependency manifest:
`31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced`
with source root
`2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a`.

## Disposition

`MECHANISM_VALID__HOLD_BOUND_SELECTOR_SCORER_AND_LIVE_BACKEND`

The capped RM(1), literal, ROMDD, canonical component/expert packet, mixed-role
pre-search bound, matched-control generator, and exact role-shape mechanics are
sound source-only mechanisms. The exact frozen verifier, 33 hostile tests, and
source-free fixture pass on a fresh RunPod staging closure.

This audit adds three adversarial production probes that the package's tests do
not cover:

1. A selection receipt can be changed to another frozen configuration and
   publicly re-sealed; `authorize_test` checks the self-hash but does not
   recompute the selected configuration from its metrics or bind those metrics
   to literal per-row packet receipts.
2. `pooled_expert_score` authenticates packet syntax but consumes encoder-side
   `weighted_sse`, `source_energy`, and label-count objects. A final result
   requires an independent decoder/scorer that derives physical weights from
   the packet and recomputes SSE against authenticated original source bytes.
3. `require_live_cupy` accepts any object whose public `__name__` equals
   `cupy`. The audit separately exercises the real installed CuPy RM path, but
   a production launcher must bind the actual module, CUDA device, and launch
   receipt rather than inherit authority from the name check.

The audit therefore does not authorize Qwen access yet. A narrow successor can
close these orchestration bindings without changing the capped algebraic
search. No result from this adapter may be interpreted as a negative for
GF(2), QTT, BDD, higher-order RM, or LOGIC-Q generally.

The independent adversarial script and real CuPy path passed from an exact
RunPod staging closure. The receipt is 1,828 bytes with SHA-256
`4a255e570f0607d5c0872fff1967862146b269b6a397eb1866d40d9a52cdbcaa`.

The real accelerator path used CuPy `14.2.0` on an NVIDIA GeForce RTX 5090,
emitted a 76-byte RM component packet with SHA-256
`c0a0a22ed591de3aa6b4a4d903b4ec9a199234426915563824bc1e7702eae008`,
and independently decoded all 512 labels. The matched Gaussian probe preserved
per-block means to `5.551115123125783e-17` and centered energy to
`2.842170943040401e-14` absolute error.

`PREFLIGHT_HISTORY.json` preserves the audit's first nonpayload failure: an
initial receipt-copy helper passed the object through JSON, changing tuples to
lists and correctly triggering the target package's frozen-grid check before
the intended re-sealing probe. The corrected audit uses `copy.deepcopy` and
leaves the frozen v1 package untouched.

## Reproduction

```bash
python -I -B adversarial_checks.py \
  --package /exact/logic_q_label_flexible_algebraic_gate_v1_capped_adapter \
  --parent /exact/logic_q_label_flexible_algebraic_gate_v0 \
  --cupy

python -I -B verify_audit.py \
  --package /exact/logic_q_label_flexible_algebraic_gate_v1_capped_adapter_independent_source_audit_20260902 \
  --expected-manifest-sha256 MANIFEST_SHA256
```
