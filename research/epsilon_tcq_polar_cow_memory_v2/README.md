# ε-TCQ compact whole-block polar state v2

This source-only successor closes the 7.147 GB state-layout blocker without
reviving the invalid coordinate-local ABI. It freezes separate verdicts:

- `GO_MEMORY_CAPACITY`
- `HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION`
- `HOLD_PAYLOAD`

The first verdict does **not** imply the second. A real CuPy allocation and
primitive-kernel smoke proves residency only; it is not throughput evidence.

## Scientific correction retained

The authenticated STRATA decoder (SHA-256
`85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e`)
decodes six complete level-major polar passes. It does not expose six independent
arithmetic events at each coordinate. This package therefore contains no
coordinate-local ε-TCQ adapter and no direct-INT2 fallback. Any future list
search must resume the actual whole-block SC state and reproduce the current
64-index reconstruction exactly.

## Why the dense allocation was avoidable

The reference decoder allocates rectangular `(N/2, log2(N))` likelihood and
partial-sum arrays. At column `c`, however, only `2^c` cells are semantically
active. The exact total is

```
1 + 2 + ... + N/2 = N - 1.
```

The compact layout uses `offset(c)=2^c-1`, `length(c)=2^c`. The source-only
tests replay the literal dense schedule and demand exact NumPy array equality
for output planes, causal uint16 frequencies, selected decisions and internal
SC decisions at five block lengths and three freeze families. A second harness
chains all six levels, updates the 0..63 indices, and performs canonical
arithmetic re-encoding at small source-free block lengths.

## Frozen state architecture

For each of `B` survivors the bound retains one explicit FP64 leaf-prior vector,
one ragged FP64 likelihood tree,
one ragged uint8 partial-sum tree and one lower-index vector. Internal decisions
are reconstructed from ancestry into one charged current-level plane scratch
per survivor and transformed in place. Layer-granular copy-on-write may share
unchanged layers, but the
peak ledger assumes no sharing gain. The `2B` children are scored and top-k
pruned before persistent state mutation, so only `B` physical state banks exist.

The explicit leaf-prior vectors close a conservative 512 MiB B=32 omission that
would otherwise require an unimplemented fused LUT-to-tree kernel.

The complete worst-case six-level ancestry is packed. Each survivor/event
symbol stores `log2(B)` parent bits plus one decision bit. A winning decision
sequence is recovered by backtrace, then must be replayed through the canonical
arithmetic encoder. No decoded-prefix copies are retained.

Packed partial sums and likelihood checkpoint/recompute are both represented in
the ledger. Neither is selected: the clearer uint8 ragged tree already fits,
while recomputation multiplies the already severe node-update count. Immutable
likelihood-layer sharing is allowed but contributes zero bytes of assumed gain.

## Exact work scope

At `N=2^21`, six levels and beam `B`, the conservative complete-pass ledger is:

```
likelihood updates       <= 6 B N log2(N)
partial-sum writes       <= 6 B (N log2(N)/2 + 1)
partial-sum XORs         <= 6 B (N(log2(N)-2)/2 + 1)
level-end polar XORs     = 3 B N log2(N)
selected rounds          <= 6 N
branch candidates        <= 12 B N
ancestry symbols         <= 6 B N
winner backtrace         <= 6 N
winner exact replay      adds one full six-pass one-path SC traversal
```

The ledger also charges lower-index gathers/adds, stable top-k comparators,
level-boundary ancestry reads and explicit COW copy bounds. These counts are why
compute remains held. Promotion requires a persistent
whole-six-level CuPy kernel, bounded launches, measured runtime and peak, exact
semantic replay, and an independent audit. Per-decision Python or kernel
launches are forbidden.

The CuPy f/g kernel is only an allocation/primitive smoke. CUDA contraction and
division can differ from the authenticated NumPy operation sequence at an
adversarial Q0.16 frequency-rounding boundary. It is excluded from semantic
evidence until a production kernel passes boundary-hostile frequency tests.

## Source-only commands

```bash
python test_source_only.py
python run_gate.py
python cupy_state_smoke.py --beams 4 8 16 32 --output cupy_receipt.json
python verify_source.py --package .
```

The CuPy receipt is an external execution artifact and is intentionally not a
manifest member. No command accepts a Qwen, current-codec or Gaussian payload.
