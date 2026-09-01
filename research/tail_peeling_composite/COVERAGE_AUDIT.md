# Sparse-tail subsumption audit

## Finding

Weight-level tail peeling was **not** subsumed by the prior Qwen experiments.
They establish useful negative evidence around it, but none constructs a
decoder-visible sparse support, pays for its values, removes those values from
the source, and then re-runs a single rate allocator on the resulting bulk.

| Prior branch | What it covers | Why it does not subsume this branch |
|---|---|---|
| Nonparametric BA | Cross-fitted scalar and adjacent 2-D/4-D stochastic test channels in raw and XKLT coordinates | Its reconstruction alphabet is stationary. It does not send an adaptive weight support or exact source values and does not score the post-peel residual. |
| STRATA | Three-bit energy class for complete natural 2,048-weight rows | Every weight remains in its row. A high-energy row label cannot identify or exactly restore its few largest weights. |
| Spectral scale field | Global, row, column, tile, low-rank, DCT and Haar variance predictors | A variance field changes allocation but carries no individual locations or values. |
| Composite super-oracle | Role KLT, row STRATA and polar/Stiefel decompositions of the complete source | It never zeros a charged sparse support and rebuilds the downstream geometry. Therefore its component spectra are not the spectra of a robust bulk residual. |
| RHT POLARIS/STRATA codec | Deterministically Gaussianizes heavy-tailed blocks | It removes the finite-code tail penalty but intentionally discards the possibility that exceptional values themselves are compressible structure. |
| Conditional hyperprior / bitplane context | Short-group scale prediction and bit-context entropy | Neither artifact exposes a separately random-access, losslessly decoded outlier plane followed by a newly optimized bulk channel. |

The distinction matters because the mask and exact values can be expensive,
while the remaining energy and dimension can fall by very different amounts.
Adding a previously measured “tail gain” to STRATA or XKLT would double count
rate and energy. The new oracle rebuilds those quantities from the source and
uses one joint waterfill.

## Most favourable honest bound

For matrix `j` with `N=1,572,864` values, the candidate selects the stable
top-`k_j` absolute BF16 values. Its support costs exactly

```text
ceil(log2 binomial(N, k_j)) bits.
```

The selected BF16 words are lossless. The charged value length is the minimum
of four self-describing codes: literal 16-bit words, canonical whole-word
Huffman, magnitude/sign Huffman, and sign/exponent/mantissa Huffman. Every
Huffman model stores its symbol and 16-bit canonical code length; its payload
length is the exact Huffman weighted path length. The two-bit mode is already
inside the fixed 128-bit matrix descriptor. Every live residual waterfill
component also receives a charged 64-bit scale/profile/length directory.

After peeling, aligned Gate/Up/Down coordinates are partitioned by their seven
possible nonempty support patterns. Each pattern receives its own orthogonal
role KLT; the required Givens-angle count is charged at 16 bits per angle. This
avoids the invalid assumption that a dense 3x3 transform preserves arbitrary
known-zero coordinates. A procedural orthogonal RHT then feeds the residual
components to one ideal Gaussian reverse-waterfiller. Integer allocations are
closed to the physical container one bit at a time by exact marginal
distortion reduction.

This bound is deliberately more favourable than an implementable
polar-lattice codec: it omits finite-block shaping loss, RHT padding,
quantized-angle error and all transform arithmetic error. Therefore a charged
coordinate-search `F=MSE*2^(2R_actual)>0.8` is evidence but not by itself a
global kill. The implementation additionally enumerates all `20^3` local tail
triples for each expert. For every positive Lagrange multiplier `mu`, it
analytically minimizes `D + mu*(side+payload)` over every local option and
sums the six minima. Weak duality makes that sum (after the physical-budget
constant) a lower bound for all `20^18` panel configurations. Only a dual
lower bound above `F=0.8` at every rate and for both raw and support-XKLT bulk
geometries hard-kills the grid. A pass would only justify the next finite-code
experiment; it would not itself be a codec result.

Two still more favourable diagnostic envelopes are kept separate:

- charge the exact mask but reveal the selected BF16 values and XKLT basis for
  free;
- reveal the mask, values and basis for free.

Neither envelope can support a compression claim. They show whether the
bottleneck is tail information rate or the remaining bulk geometry.
