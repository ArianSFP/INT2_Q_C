# STRATA-BMP/OBDD/QTT6 source-only mechanism gate v0

Date: 2026-09-02

Status: **source-only mechanism frozen but unexecuted; HOLD all model/Qwen
payload pending an independent source audit**. The provided RunPod refused
connections during this freeze, so no test or GPU-pass claim is made. See
[THREAT_MODEL.md](THREAT_MODEL.md).

This package tests a narrow LOGIC-Q successor: move all six completed STRATA
reconstruction planes toward a short coordinate function while optimizing the
literal rate-distortion objective. It is not a frozen-label entropy census.
For every coordinate the only distortion input is the exact table

```text
D[i,k],  k = 0,...,63.
```

The decoded index is always

```text
k[i] = x_0[i] + 2*x_1[i] + 4*x_2[i] + 8*x_3[i]
       + 16*x_4[i] + 32*x_5[i].
```

There is no four-level ABI, no projection from four labels, and no access to
internal SC decisions. Joint exceptions replace a complete 0..63 index and
are expanded back into six completed bitplanes.

## Coordinate contract

The v0 shape family is decoder-derived and contains no learned coordinate
table. An expert has three semantic roles and shape

```text
intermediate = 3*2^a,  hidden = 2^b,  role in {Gate, Up, Down}.
```

The required source-free fixture uses Qwen-compatible mixed radix
`768 = 3*2^8`, hidden width 2,048 and a 16x256 tile (`N=4096`). A coordinate is
expanded into the role trit, the row-sector trit, eight row bits and eleven
column bits. Constant bits are removed for a tile. A frozen four-member order
bank sorts the remaining twelve bits; its selector is physically charged.
The same rule, not model identity, derives the variables for every accepted
shape. The v0 restriction to `3*2^a` intermediate widths is explicit; it is a
shape-family mechanism, not proof of portability to every possible SwiGLU
width.

## Three representation families

1. **GF(2) physical matrix factor.** For each completed plane on an
   `n_r x n_c` tile, `B_l = U_l V_l^T mod 2`, with frozen ranks
   `{0,1,2,4}`. This is the finite-field low-rank screen.
2. **Reduced OBDD.** A canonical reduced ordered binary decision diagram maps
   the ordered coordinate bits to each plane. An exact bottom-up dynamic
   program optimizes the pruned ordered-tree subset using conditional 64-way
   costs, after which identical subfunctions are reduced and canonicalized.
3. **BMP/QTT over GF(2).** Binary three-index cores are selected by each
   coordinate bit and multiplied over GF(2) to yield one scalar label bit.
   Rank is frozen to `{1,2}` and a bounded core-bit coordinate descent changes
   labels according to the exact conditional 64-way distortion.

The third family is the sequential binary-matrix-product object related to
the Boolean-function BMP construction in [*A Matrix Product State
Representation of Boolean Functions*](https://arxiv.org/abs/2505.01930).
Binary-index folding and binary-tensor factorization are adjacent to
[*Factorizing binary tensors into quantics tensor
trains*](https://arxiv.org/abs/2606.04506). This package is a bounded codec
experiment, not an implementation or reproduction of either paper.

These are **discrete coordinate functions**. The BMP/QTT cores output a
completed label bit deterministically. They are not a value tensor train that
approximates floating-point weights, and they are not an MPS probability law,
Born machine, entropy model, or tensor-network prior over stochastic label
strings. The OBDD follows one deterministic branch per coordinate. No result
from an earlier value-TT or probability-MPS screen transfers to this gate.

## Exact physical descriptor formula

The binary header is 30 bytes and the trailing CRC32 is 4 bytes. The family
and order selectors consume two full header bytes (16 physical bits). Every
joint exception is a sorted `(uint16 position, uint8 index)` tuple, exactly
three bytes. Bit vectors are little-bit-packed with a mandatory zero tail.

For six per-plane ranks `r_l`, active coordinate depth `d`, node counts `n_l`,
and `E` exceptions:

```text
B_factor = 8 * [30 + 4 + 6
                + sum_l ceil(r_l*(n_r+n_c)/8) + 3E]

B_OBDD   = 8 * [30 + 4 + sum_l(4 + 5*n_l) + 3E]

C(d,r)   = 4r + 2(d-2)r^2                         (d >= 2)
B_BMPQTT = 8 * [30 + 4 + 6
                + sum_l ceil(C(d,r_l)/8) + 3E].
```

All byte padding is included. `descriptor_formula()` recomputes the selected
identity from the independently decoded packet and requires exact equality
with its byte length. The packet has CRC32, strict extents, canonical OBDD
reduction, topological references, zero bit tails, sorted nonredundant
exceptions, canonical decode and byte-identical re-encode.

At N=4096, the exact no-exception minima are 40 bytes / 0.078125 bpw for six
rank-0 matrix-factor planes and 58 bytes / 0.11328125 bpw for either six
terminal OBDD roots or six rank-1 depth-12 BMP/QTT cores. At the frozen caps,
factor rank 4 plus 64 exceptions is 1,048 bytes / 2.046875 bpw, aggregate
240-node OBDD plus 64 exceptions is 1,450 bytes / 2.83203125 bpw, and rank-2
BMP/QTT plus 64 exceptions is 298 bytes / 0.58203125 bpw. These omit production
STRATA fields and are mechanism-packet bounds only.

The packet encodes only the completed 0..63 reconstruction-index function for
one tile. It does **not** yet contain the authenticated STRATA scale, RHT/KLT,
profile, component/expert framing, or page padding. Consequently its fixture
bpw is a mechanism descriptor rate, not a complete production rate.

## Label-flexible search and caps

Every plane update derives its two binary costs from the same literal
`D[i,0..63]`, conditional on the other five current planes. Whole candidates
are reassembled and scored as

```text
sum_i D[i,k[i]] + lambda_bit * 8*len(canonical_packet).
```

The source-free implementation contains:

- greedy weighted GF(2) rank-one factor extraction;
- exact dynamic programming for a bounded ordered-tree subset followed by
  ROBDD reduction;
- bounded GF(2)-QTT core-bit coordinate descent;
- joint nearest-index exceptions only when their exact distortion saving
  exceeds their physical 24-bit cost;
- an independent exact `64^N` enumerator for hostile tests with `N<=3`.

Hard caps are 4,096 weights, twelve active variables, four BMP rank, two QTT
rank, 240 total OBDD nodes, 64 exceptions, 32 family candidates, 1,000,000
scored search operations, 64 MiB CPU workspace and 128 MiB CuPy workspace.
A cap hit fails closed and is only a bounded-negative for this bank.

## Matched-control contract

The executable fixture creates one structured synthetic source and one
Gaussian source with the exact same empirical mean and standard deviation.
Both receive the identical levels, 64-way table construction, lambda, family
bank, order bank, exception rule and caps. This is a mechanism check only.

A future payload protocol must freeze train/validation/test partitions before
access and regenerate at least eight moment-matched Gaussian PTQ controls from
authenticated raw BF16 source statistics. It must repeat the **entire** family
selection and label search on every control. Promotion uses whole-layer and
whole-expert uncertainty and source-minus-control advantage; it cannot compare
a selected Qwen winner against an unselected random baseline.

## Reproduction

From this directory's repository root:

```bash
python -I -B research/strata_bmp_qtt6_gate_v0/test_source_only.py
python -I -B research/strata_bmp_qtt6_gate_v0/run_source_free_fixture.py
python -I -B research/strata_bmp_qtt6_gate_v0/run_cupy_smoke.py
python -I -B research/strata_bmp_qtt6_gate_v0/verify_source.py \
  --expected-manifest-sha256 MANIFEST_SHA256
```

The ordinary tests and fixture never import or initialize CuPy. The explicit
GPU smoke uses only generated arrays, synchronizes the active device, and
compares exact GPU outputs with NumPy. It has no payload locator.

## Claim boundary and next gate

If independently executed successfully, this package would establish packet
mechanics, completed-plane semantics, bounded optimization plumbing and a
source-free CuPy path. This unexecuted freeze proves none of those runtime
properties. It does not prove
that Qwen labels have a short coordinate function, that MSE falls below a
Gaussian limit, that the packet lies in the complete 2.15--2.5 bpw ledger, or
that cold reads are below 2x.

No Qwen/model/coarse/control payload may be opened under this source package.
An independent auditor must freeze the exact source root, replay hostile tests
and the CuPy smoke, and separately approve a production adapter that binds the
real STRATA scale/transform/framing bytes and independent original-BF16 scorer.
