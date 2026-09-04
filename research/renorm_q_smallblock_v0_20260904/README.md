# RENORM-Q tiny-cell source-only probe

## Scientific question

This package tests whether a small, public collective variable can expose
long-range information that is invisible to pairwise or bounded raster
contexts, and whether the same variable supports **jointly selected** labels
under an exact hierarchical rate-distortion objective.

It deliberately separates two questions:

1. The RSMI-style census ranks a frozen map bank by the per-weight quantity
   `[I(z; environment) - beta H(z)] / cell_sites - map_descriptor_bpw`.
2. The exact tree contraction minimizes literal scalar distortion plus a
   fixed hierarchical model codelength while allowing every leaf label to
   move away from its nearest-neighbour choice.

The second is the relevant architectural discontinuity. An invertible
transform of fixed labels cannot create an entropy benefit; a structured
quantizer may choose a nearby, lower-description label field.

## Frozen tiny map bank

Cells contain at most six sites and each site has alphabet size at most four.
The source-independent bank contains:

- low- and high-Gray-bit parity;
- their two-bit product;
- modular label sum and modular support count;
- within-expert role syndromes;
- same-role cross-expert syndromes;
- a two-bit role/expert cube parity.

The map identity is explicitly charged in the census.  No neural map or
per-block source-derived lookup table is permitted.

## Exact label-flexible tree

For a power-of-two number of leaves, a public map assigns every cell tuple
`q_B` a small state `z_B`.  The package solves

```text
min(q,z) sum_B distortion_B(q_B)
       + lambda * [root_nll(z_root)
                    + sum_edges transition_nll(z_parent,z_child)
                    + sum_B leaf_nll(q_B | z_B)]
```

by bottom-up min-sum contraction and deterministic backtracking.  An
independent exhaustive enumerator checks the complete objective on tiny
instances.  This is a finite-state mathematical kernel, not yet a physical
entropy packet: probability tables are caller-supplied and may be treated as
free only for kill-only screening.

## Controls and gates

The source tests include:

- a complete XOR construction in which the environment is exactly one
  collective parity bit and the parity map obtains `I(z;E)=1`;
- a perfectly balanced block/environment Cartesian product where every map
  obtains exactly zero mutual information;
- exact min-sum versus global enumeration;
- a label-flexible example where the optimum changes a nearest label to enter
  a cheaper collective trajectory.

A future Qwen capability must train map/model selection only on auxiliary
whole layers and score untouched whole experts/layers. The entire selection
must be repeated on matched Gaussian controls. Frozen decisions are:

- control-corrected lower-confidence gain below `0.03 bpw`: hard kill;
- `0.03–0.045 bpw`: scientific signal only;
- at least `0.045 bpw`: eligible for a **separate** finite projection;
- direct architecture success still requires about `19.0995%` lower MSE at
  the current 2.5-bpw baseline, or an equivalent fully charged rate gain.

No separately fitted RSMI, entropy, and MSE gains may be added.

## Read-local projection

The preferred deployment projection is an entirely expert-local hierarchy:
all root, transition, and leaf streams for one expert are page-contiguous and
read once when that routed expert enters cache. Its ideal logical read
amplification is `1x`; page headers and alignment must later be measured.

If a collective variable crosses experts, it must be converted to a literal
common/private packet. For `E` experts, `C` common bits, and `P` private bits
per expert, the ideal routed amplification relative to equal ownership is

```text
(P + C) / (P + C/E).
```

The helper reports this quantity, but it excludes page rounding. Any finite
projection must remain strictly below `2x`; a monolithic interleaved
multi-expert stream is forbidden.

## Capability boundary

This package has no Qwen/model path, network, CuPy, GPU, payload, deployment,
or packet-writing entry point. `RUN_DISABLED.txt` is normative. A future local
RTX 3060 run requires a separately named, independently reviewed one-use
capability binding exact source hashes, tensor geometry, split, controls,
CuPy kernel receipt, physical ledger, and output schema.

Run only the source KATs:

```powershell
C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B `
  research\renorm_q_smallblock_v0_20260904\test_source.py
```
