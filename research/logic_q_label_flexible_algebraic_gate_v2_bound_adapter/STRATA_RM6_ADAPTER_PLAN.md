# STRATA-RM6 adapter plan

Date: 2026-09-02

Status: design plan only. The frozen LOGIC-Q v0/v1 family uses an abstract
four-level alphabet. It is **not** byte-, index-, distortion-, or
rate-compatible with the current STRATA codec. No result from that four-level
mechanism transfers to this plan.

## Current semantic target

The authenticated STRATA decoder performs six complete level-major polar SC
passes. After pass `l` it has one completed output plane `x_l[0:N]`. The
reconstruction index and scalar transformed-domain level are

```text
k[i] = x_0[i] + 2*x_1[i] + 4*x_2[i] + 8*x_3[i]
       + 16*x_4[i] + 32*x_5[i]              in 0..63
y[i] = eta * (-31 + k[i]).
```

The output plane is not a vector of coordinate-local SC decisions. Internal
SC state is bit-reversed and polar transformed before it becomes a completed
plane. Any bridge must authenticate all six completed planes and their exact
assembly into `k`; a six-independent-events-per-coordinate ABI is forbidden.

## Exact 64-way distortion input

For every coordinate `i`, construct a literal table

```text
D[i,k] = source-domain distortion of reconstruction index k, k=0..63.
```

The table is evaluated with the frozen STRATA scale/stratum, KLT, RHT, BF16
staging, and inverse conventions. Where orthogonality proves equality, a
transformed-domain table may accelerate search; every promoted whole-block
candidate is nevertheless reconstructed by the independent current-semantic
decoder and scored in original BF16 source coordinates. The table is the only
label objective. A four-level nearest-label cost is never accepted.

At level `l`, lower planes give

```text
k_low[i] = sum_{j<l} 2^j*x_j[i].
```

An initialization soft cost can marginalize unknown higher planes,

```text
C_l[i,b] = min_h D[i, k_low[i] + 2^l*b + 2^(l+1)*h],
```

while alternating sweeps use the exact current 64-way index with the other
five planes fixed. Every list candidate is ranked by `sum_i D[i,k[i]]`, not by
Hamming distance or a sum of separately reported plane gains.

## Frozen RM/sub-RM dimension bank

The first physical block is `N=4096=2^12`. A plane in `RM(r,12)` has dimension

```text
K(r) = sum_{j=0}^r C(12,j).
K(3)=299, K(4)=794, K(5)=1586.
```

Six `RM(5,12)` planes contain 9,516 information bits, or
2.3232421875 bits/weight before metadata. A 2.5-bpw block has 10,240 bits, so
724 bits remain for every scale, selector, length, CRC, alignment, and other
packet field. One `RM(4,12)` plane plus five `RM(5,12)` planes contains 8,724
bits, or 2.1298828125 bpw before metadata.

The pre-payload bank must freeze:

- an order tuple `(r_0,...,r_5)`;
- per-level active dimensions `(K_0,...,K_5)`;
- one public degree-then-lexicographic monomial order;
- optional shortening only by taking a prefix of that monomial order;
- a total constraint `sum_l K_l + all_fixed_and_variable_packet_bits <= R*N`;
- `2.15 <= R <= 2.5` after literal byte alignment and page padding.

The initial bank should contain all six placements of the single `RM(4,12)`
plane among five `RM(5,12)` planes, the all-`RM(5,12)` point, and a small frozen
set of shortened boundary points. Family selection is charged explicitly. No
post-test order or dimension addition is allowed.

## Bounded joint search

1. Start from the current STRATA 0..63 index vector and its six completed
   output planes.
2. For level 0 through 5, compute exact binary soft costs from the 64-way table
   conditional on the other five current planes.
3. Run frozen soft-decision RPA/list-RPA or a bounded sub-RM list decoder on
   CuPy. Retain at most `L` completed output-plane candidates per level; the
   initial caps are `L in {4,8,16}`, at most three full six-level sweeps, and a
   fixed global whole-block beam cap of 64.
4. After each proposed plane, assemble all six planes into 0..63 indices and
   rescore the whole block exactly. Retain the current legal candidate so a
   sweep cannot silently worsen the objective.
5. Alternate levels in the frozen order `0,1,2,3,4,5`; a reverse-order second
   schedule may exist only as a separately frozen bank member.
6. Stop at the cap. A miss is a bounded-negative for that bank, not a negative
   for Reed-Muller, algebraic, or label-flexible quantization generally.

No implementation may construct a full Qwen-sized RM codeword-pair matrix or
an unrestricted `O(N^2)` exception table. GPU workspaces, list widths, sweeps,
and exception counts are hard caps checked before family search.

## Canonical decode and packet

For each level, serialize exactly `K_l` information coefficients in the frozen
monomial order. The decoder evaluates the unique RM/sub-RM Boolean polynomial
at all 4,096 public binary coordinates to recover completed output plane
`x_l`. It then assembles

```text
k[i] = sum_l 2^l*x_l[i]
```

and uses the authenticated STRATA reconstruction/index semantics. Decoder
output must include the six plane hashes, the 0..63 index-vector hash, and the
FP64 reconstruction hash. Canonical re-encoding must reproduce every packet
byte. If compatibility with an existing polar packet is claimed, the bridge
must additionally invert the polar transform, enforce all frozen internal SC
positions, replay causal Q0.16 arithmetic in level-major order, and reproduce
the current packet exactly; absent that proof the packet is a new STRATA-RM6
family, not a current-packet recoding.

The physical block header must minimally bind:

- magic, version, `STRATA_RM6` family, role, tensor shape, expert/block ordinal;
- `N`, `m=12`, all six orders and active dimensions;
- scale/eta and stratum/profile identifiers with their literal bytes;
- coefficient-stream bit lengths and hashes;
- any bounded exceptions with count, sorted positions, and replacement bits;
- logical payload bits, CRC/checksum, zero-tail rule, component alignment;
- current common fields (KLT/RHT seeds or procedural identifiers, strata and
  every other required reconstruction field);
- component bytes, expert header bytes, page padding, and final packet bytes.

The rate receipt must report each field separately, their exact sum, total
source weights derived from headers, `physical_rate_bpw`, contiguous routed
read bytes, read passes, and cold-read amplification. The target layout is one
page-contiguous expert object read once (`1.0x`); no second compressed-expert
fetch may be hidden as scratch traffic.

## Promotion gate

The entire frozen bank and selection must be rerun on eight moment-matched
Gaussian PTQ controls, with identical family search and independent raw-source
scoring. Train/validation chooses one bank member; whole test layers remain
closed until an externally pinned selection receipt exists. A promoted packet
must satisfy all of:

```text
2.15 <= physical bpw <= 2.5
F = raw_relative_MSE * 2^(2*physical_bpw) <= 0.8
cold routed reads < 2x
independent six-plane decode and canonical re-encode
```

No separately measured four-level saving, ideal RM dimension, control gap, or
plane-wise gain may be added to the final result.
