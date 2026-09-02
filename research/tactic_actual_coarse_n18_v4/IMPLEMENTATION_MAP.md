# V1/V2/V3 audit synthesis and shortest implementation path

## Reusable components

| Source | Retain | Do not inherit |
|---|---|---|
| v1 `stream_codec.py` | Proven bridge from POLARIS-SC `run_trial` to a fixed reservoir; causal independent decoder; integer inverse-RHT symbol identity | Qwen-only `block<6` seed rule, mutable live imports, absent universal tail policy |
| v2 `n18_common.py` | 128-byte shape/role/tile header; 1,024-bit arithmetic reserve; exact hard EOF and physical zero fill; owner-aware geometry | Full-reservoir tails presented without an explicit target-eligibility boundary; source/runtime authority defects |
| v3 | Exact warning that byte allocation is not a codec; page-union arithmetic; immutable-execution/publication ideas | Undefined 1,228-byte microblock language, zero-slack metadata relabeling, zero-byte tiny fallback as supposed universal target closure |

## Why the v3 microblock route is not the shortest path

At `N=4096`, the proposed 1,228-byte slot has only 32 bits beyond the nominal
`2.390625 bpw` test-channel budget.  That is insufficient to assume the
existing variable-length arithmetic payload, an FP32 scale, a logical length,
and finite overflow reserve.  A new fixed-length polar/coset construction
could eventually make such a packet real, but it is a new codec requiring a
new finite-length proof and numerical implementation.

At `N=2^18`, the inherited finite reservoir already has 1,024 spare logical
bits after its 128-byte header and has a working causal encode/decode bridge.
Preserving that language is therefore the shortest honest route to a real
coarse residual on the Qwen geometry.

## V4 closure matrix

| Required item | V4 implementation |
|---|---|
| Versioned packet | `TACN18C4`, 128-byte header, fixed 78,592-byte record |
| Arithmetic capacity | 627,712 physical payload bits; 626,688 nominal; 1,024-bit reserve; terminal failure on overflow |
| Scale | One positive FP32 decoder scale per nonzero record; canonical `1.0` for exact-zero tile |
| Tail | Shape-bound valid prefix followed by implicit BF16 `+0` before RHT; full record charged; compatibility-only outside exact-rate cell |
| Encoder | `numeric_encoder.py`, authenticated exact CuPy encoder bytes, procedural Q31-BEC flags, literal packet output |
| Independent decoder | `independent_decoder.py`, no encoder import, causal probability regeneration and exact arithmetic re-encode |
| Original-domain residual | Independent inverse signed RHT, shape-prefix reconstruction, BF16 source subtraction and FP64 pooled SSE/energy |
| Canonical frame | Gate -> Up -> DownT, ascending tile; every record repeats shape/role/tile; exact exhaustion |
| Model ledger | Zero source-fitted/shared coarse model bytes; downstream TACTIC common/fine bytes remain charged separately |
| Rate | Exact `307/128` only when each role has no tail; actual compatibility rate reported otherwise |
| Read schedule | One sequential compressed pass with buffered decoded state; unique pages and repeated bytes separated |
| Second pass | Frozen final topology explicitly reports `724/360 = 2.011111...x`, hence invalid |

## Remaining blockers are runtime/evidence blockers, not grammar blockers

V4 remains source-only until an independent auditor freezes the source and a
separate dispatcher binds the actual numerical runtime/import bytes.  The
first payload action should be three Qwen pilot records, because a single
arithmetic overflow kills this fixed profile without retry.  A successful
pilot still does not prove the other 105 records fit or that DH384 captures
enough residual energy.

The padded-tail language is universal and independently decodable, but its
rate is intentionally not target eligible.  Closing arbitrary non-divisible
shapes inside 2.5 bpw requires a future finite variable-length or small-block
tail codec; it cannot be obtained by changing only a ledger.

