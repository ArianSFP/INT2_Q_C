# Independent audit of LOGIC-Q label-flexible algebraic gate v0

Date: 2026-09-02

Audited parent manifest:
`31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced`
with source root
`2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a`.

## Disposition

`MECHANISM_VALID__HOLD_FOR_CAPPED_ADAPTER_SUCCESSOR`

The sealed package is internally coherent as a source-only finite mechanism:
its manifest verifies, all 25 declared tests pass, the fixture passes, and the
literal, RM1, tiny GF(2), ROMDD, component, and expert decoders reproduce their
synthetic packets. The core component functions are shape-parametric; the
Qwen `768 x 2048` constants occur in accounting helpers and the proposed pilot
protocol, not as a restriction of the packet decoder.

This does **not** authorize direct Qwen payload use. A separately frozen and
audited capped adapter successor is required first.

## Production holds

1. The documented hard-kill ordering is not an executable orchestrator.
   `encode_family_bank` invokes every family directly and never calls
   `optimistic_family_bound`.
2. The scalable-search boundary is much narrower than the family names imply.
   RM1 builds the complete affine pair-cost matrix before retaining a list;
   GF(2)'s non-exact search rejects ranks above 12; an unrestricted exception
   limit causes quadratic work in the current exact prefix enumeration.
3. Scale fitting precedes and ignores the algebraic label search. A miss cannot
   reject joint scale/label LOGIC-Q; a capped adapter should retain a frozen
   scale shortlist or state this bounded-negative limitation explicitly.
4. `absolute_source_gate` uses one component's bytes. It excludes the expert
   header, inter-component alignment, final 4 KiB page, and pooled three-role
   distortion. Controls and success decisions must use one packed expert
   ledger. The adversarial fixture exhibits components at 2.1875 bpw whose
   packed tiny expert is 2.6667 bpw.
5. The expert envelope checks role names but not canonical SwiGLU shape
   equality. It also accepts noncanonical GF(2) gauge-equivalent packets and a
   ROMDD depth header not bound to the serialized graph. These do not create a
   rate advantage for an honest encoder, but final canonical re-encode and
   tamper tests require closure.

## Universality and leakage assessment

- The numeric packet mechanics accept positive shape parameters and exact
  mixed-radix ROMDD domains; a non-Qwen `3 x 8` synthetic roundtrip is included
  in `adversarial_checks.py`.
- The panel protocol intentionally demands identical expert slots and role
  shapes across all layers. That is acceptable for one Qwen cohort, but a
  universal SwiGLU-MoE claim needs separately frozen shape/expert-count
  cohorts and transfer to a disjoint model family.
- The split hashes public layer and slot identifiers only. Test layers are
  wholly excluded from train/validation, and validation slots are excluded
  from training. No learned state or permutation exists in v0, so no present
  source-state leakage was found.
- The code states that global search choices are train/validation-selected,
  but no executable selection/receipt layer exists. The successor must bind
  the panel order, component ordinals, hyperparameter grid, selection receipt,
  and source hashes before opening test components.
- Matched Gaussian generation reruns continuous quantization and all families,
  but its component ordinal must be assigned canonically by the successor.

## Exact independent RunPod replay

The exact sealed parent was replayed with
`/workspace/int2-cupy-venv/bin/python -I -B` at the hash-named staging path.

| Receipt | Bytes | SHA-256 | Result |
|---|---:|---|---|
| source verifier | 1,865 | `272de43095a6a355844b4f119af968ca844d6334e9fcd6f171d6fe2a0fb1f54e` | pass |
| 25-test suite | 4,286 | `fb1d3a5c74ed8a56a60505474a85f96259f6f0f7fc1a1feb1f24a7a047d8d437` | pass |
| source-free fixture | 8,379 | `7da357ccb9bb8e5fdda7b16d8fd6c85bd0eddc63ce7b841e1f54ae1cef2ea050` | pass |
| independent adversarial checks | 1,189 | `e7f15ac3ea80da6fca6c0c5fabc7bb72ba9152ae528aa31046cbd7ac07ea54cb` | mechanism pass with production holds |

No model, Qwen weight, current-codec payload, coarse result, or prebuilt
matched-control artifact was opened by this audit branch.
