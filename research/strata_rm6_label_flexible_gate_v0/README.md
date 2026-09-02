# STRATA-RM6 label-flexible gate v0

This source-only package tests an algebraic bridge without inventing a foreign
four-level label ABI. A reconstruction coordinate is always the completed
six-plane STRATA index

```
q[i] = sum(level_bitplane[level][i] << level, level=0..5),  0 <= q < 64.
```

The exact distortion objective is a float64 4,096-by-64 table formed from the
current `eta=0.25` alphabet and literal FP16 decoder scale.
Every API and packet fails closed unless it receives exactly six completed
level-major planes; there is no four-level-label adapter.

## Authenticated RM orientation

Current STRATA emits `polar_transform(internal[bit_reverse])`. Under that
convention the generator row controlled by internal phase `i` has Hamming
weight `2**popcount(i)`. Therefore exact RM(r,m) selects

```
popcount(i) >= m-r.
```

For RM(5,12), this selects 1,586 phases. Six complete planes contain 9,516
selected decisions, or 2.3232421875 raw information bits per coordinate.

## Two candidates—two different ledgers

The local direct candidate resets SC every 4,096 coordinates. Its 40-byte
header charges the bank, all six orders, FP16 scale, profile, coset selector,
SC seed and RHT seed. It then stores the canonical current integer-arithmetic
stream, a CRC32 and zero alignment to 128 bytes. Actual arithmetic length is
data-dependent and the encoder fails closed above 1,280 bytes (2.5 bpw). The
uniform RM(5,12)^6 dimension ledger, including metadata and alignment, is
exactly 1,280 bytes.

The target-rate contract is separate from packet validity. Promotion requires
the **actual emitted literal packet** to lie in `[2.15, 2.5]` bpw. Packets below
2.15 bpw, including the source-free controls, are mechanism fixtures only.
This version has no literal refinement or format-versioned extra-padding field,
so such fixtures are not target-eligible. The ledger reports selected RM
dimension and emitted Q0.16 arithmetic bits as different fields.

The cheap global candidate keeps the authenticated 2^20/2^21 blocks and every
current per-level selected count K, but ranks phases by descending popcount.
Unless K equals a complete RM dimension, this is an **RM-ordered truncated
polar set**, not exact RM. Its physical rate can only be established by a real
canonical arithmetic stream; the 4,096-local dimension calculation does not
apply. This candidate remains held.

## Coset control

Both zero-frozen and current procedural-random frozen modes are literal packet
options and the selector is charged. Zero frozen values expose low-degree
coordinate functions. Current random frozen values produce a public affine
coset: they preserve RM covering differences but destroy direct low-degree
appearance. The two controls must not be pooled.

## What passed

- authenticated six level-major passes and 0..63 reconstruction semantics;
- exact RM generator orientation and dimensions;
- exact per-coordinate 64-way costs;
- canonical packet encode/decode/re-encode with CRC and padding rejection;
- bounded exact joint six-plane enumeration at N=8;
- source-free CuPy RM(5,12)^6 exact-cost legal-flip search and packet replay.

## What remains held

- a production joint RM(5,12)^6 encoder;
- the current-global-block RM-ordered arithmetic-length experiment;
- outer-container/RHT/KLT integration and routed-read benchmarking;
- every Qwen, coarse and matched-control payload experiment.
- target promotion for any packet below 2.15 bpw until a literal charged
  refinement/padding representation is implemented.

Commands:

```bash
python test_source_only.py
python run_gate.py --auditor ../../strata_v2_klt_mixed_independent_auditor_v1.py
python cupy_soft_search_smoke.py --steps 6 --output cupy_receipt.json
python verify_source.py --package .
```
