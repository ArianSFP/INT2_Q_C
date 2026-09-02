# Independent audit: epsilon-TCQ/WFA early gate v0

Date: 2026-09-02

Audited source pins:

- manifest SHA-256: `0d146f3510d0d42e90d5fd58fe283a8b2dcbf2bd278fa4201fd1601dd301383b`
- source-root SHA-256: `17581794ba7c6c35faf76f5c3926b59a71fa0b57b3df41c73082ceee202a13e0`

This audit is source-only. It opens no Qwen weights, current-codec packet,
legal-candidate trace, or matched-Gaussian control.

## Verdict

The frozen search/packet primitives pass their source-only boundary:

- the manifest and exact member closure authenticate;
- all 14 isolated tests pass;
- exact joint-label search matches an independently implemented exhaustive
  arithmetic-code oracle on the bounded direct-four fixture;
- a beam result independently rescores to its reported literal byte objective;
- packet parse, causal decode, label digest, and canonical re-encode agree;
- model and centroid tables are physically present when the canonical fixed
  byte helper is used;
- the direct-four and synthetic six-event interfaces are not aliases;
- the state-permuted map is a source-independent bijection for each stream;
- owner-component partition mechanics prevent a shared-owner stream from
  entering development; and
- the payload-free CuPy branch scorer/Hankel smoke passes on RunPod.

The package is **not safe to promote or open Qwen through as a measurement
gate**. The independent adversarial audit proves that:

1. `search_labels` accepts an arbitrary caller-supplied `fixed_packet_bytes`.
   Passing zero makes `SearchResult.physical_bytes` disagree with the literal
   packet subsequently built from the result.
2. `source_gate` validates a byte ledger internally but never requires
   `row["bytes"] == row["byte_ledger"]["total_bytes"]`.
3. `source_gate` trusts caller-supplied state/local/permuted gains and read
   amplification instead of deriving them from bound packets and page ranges.
   It can declare a survivor whose joint SSE is worse than its stated nearest
   SSE and whose physical ledger is 8192 bytes while its scored row claims
   1280 bytes.
4. `final_control_gate` accepts assertion booleans and any 64-character string
   as a control closure; it does not authenticate or recompute eight control
   receipts.
5. The outer-fold fit/selection driver, full matched-control pipelines,
   multi-owner container, and measured inference read path are explicitly
   planned rather than executable. The one-pass ledger is a projection over
   caller-declared byte counts, not a storage trace.
6. Production-length beam scalability is not established. Every expansion
   copies the complete emitted-bit, label, and pre-state tuples; the search
   performs causal evolution and sorting on the host, and its CuPy `topk_paths`
   helper is not wired into `search_labels`. The demonstrated CuPy path batches
   only branch-error arithmetic. This is correct for the tested fixture but is
   not evidence of a practical 4,096-label six-event encoder.

The state-permuted head is mechanically a bijective, source-independent,
equal-byte control. Its affine permutation family has only a short
stream-ordinal period, however, so one such control is not a strong inferential
null by itself. Likewise the owner-component helper partitions correctly, but
there is no executable outer-fold fitting driver proving that all future
selection and fitting calls obey the partition.

Accordingly the audit status is
`HOLD_BINDINGS_REQUIRED_BEFORE_PAYLOAD`.

## Safe next step

It is safe to begin a **separately frozen, read-only** STRATA/POLARIS
legal-transition adapter because the existing runner remains typed-HOLD and
accepts no payload. The adapter must be independently replayed against the
literal current codec and must not expose source values or a direct-four
fallback. It must not be joined to Qwen or used for a result until a successor
driver binds fixed bytes to the actual packet, derives every gate metric,
authenticates control receipts, and measures real one-pass ranges.

## Reproduction

```bash
python3 -I -B independent_audit.py \
  --package /absolute/path/to/epsilon_tcq_wfa_early_gate_v0 \
  --cupy
```
