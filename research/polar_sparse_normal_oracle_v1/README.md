# PSNO-v1: sparse polar-normal necessary oracle

Status: **source-only proposal; no tensor, CuPy, CUDA, or GPU execution is
authorized**.

This is the cheapest credible successor to the killed
`polar_normal_predictor` branch.  It asks whether the exact polar normal
correction is exceptionally sparse in the orthonormal symmetric-coordinate
basis, or spatially clustered enough that block support coding changes the
answer.

## Core representation

For each frozen `768 x 768` symmetric polar normal `N`, use the orthonormal
Frobenius coordinates

```text
a_ii = N_ii
a_ij = sqrt(2) N_ij,  i < j.
```

There are exactly

```text
m = 768 * 769 / 2 = 295,296
```

unique coordinates and `sum(a^2) = ||N||_F^2`.  Therefore selecting the `k`
largest `a^2` values is the exact best arbitrary `k`-coordinate symmetric
support.  No Jacobian or double-counting approximation enters this gate.

For a per-matrix arbitrary support, the exact enumerative support charge is

```text
h(k) = ceil(log2 binomial(295296, k)) bits.
```

Every score separately declares a value charge `b_value * k`.  The oracle
reconstructs the chosen continuous values exactly despite that finite charge,
so it is strictly more favourable than a `b_value`-bit quantizer under the
same support/value ledger.  A one-bit exact-value row is the principal gross
gate; it is not claimed to be an emitted representation.

## Exact source-only threshold

`threshold_math.py` reads only the sealed result metadata from the prior
normal predictor.  It does not read tensors.  Using the prior model/normal
energies and dimensions, it independently replays the Gaussian reverse
waterfill and asks what common normal-energy capture would be required for an
equal-`k` coordinate oracle.

With coefficient values granted free and count headers also granted free:

| `k` per matrix | exact support bpw | required normal capture | iid-Gaussian top-k capture | resulting iid `F` |
|---:|---:|---:|---:|---:|
| 64 | 0.000551224 | 83.529% | 0.337% | 0.95236 |
| 1,024 | 0.006254196 | 84.755% | 3.602% | 0.95545 |
| 4,096 | 0.019798915 | 87.326% | 10.899% | 0.96193 |
| 16,384 | 0.058054606 | 92.785% | 29.968% | 0.97566 |
| 32,768 | 0.094397227 | 96.942% | 46.803% | 0.98253 |
| 45,056 | 0.115693410 | 99.452% | 56.286% | 0.98305 |

The exact last equal-`k` row that could hit `F=0.8` even after capturing
**all** normal energy is `k=48,018`: its support is `189,133` bits per matrix,
its total support rate is `0.12024752298990885 bpw`, and its total side rate
including the existing polar side is `0.12141764605486835 bpw`, giving
`F=0.7999988173257864`.  At `k=48,019`, exact support rises to `189,136` bits
and even zero residual normal gives `F=0.8000011595439223`.

Once the deliberately impossible exact values are charged at one bit each,
the last full-capture equal-`k` survivor falls to `k=34,764`.  At `k=32,768`
it already requires `99.3968054%` normal-energy capture.  With exact values
charged at FP16 width, the last full-capture survivor is only `k=8,383`; the
nearby `k=8,192` row requires `99.6899308%` capture.  These are rate/dimension
thresholds, not assumptions about the unseen Qwen prefixes.

This is not yet a data-dependent kill because Qwen top-k normal-energy
prefixes were not stored.  It does show the required phenomenon: at every
coordinate-support operating point, the normal must be radically more
concentrated than an i.i.d. Gaussian field.  The best iid-Gaussian row is the
identity `k=0` baseline `F=0.9520339564260487`; arbitrary coordinate support
only worsens it.

## Strong containing gate

A future authorized runner supplies the already-reconstructed normal matrices
to `cupy_gate_proposal.py`.  That module has no paths and performs no action on
import.  It returns exact coordinate energy prefixes, fixed-block group
energies, offset-segment energies, and Gaussian-rank heavy-tail controls.

The hard decision is based on a Lagrangian dual, not on a heuristic joint
allocation.  For normalized component dimension `d`, energy `e`, payload rate
`r`, and multiplier `lambda`, define

```text
phi(d,e,lambda) = min_{r >= 0} e * 2^(-2r/d) + lambda*r.
```

For matrix `i` and support option `k`, the normal term is

```text
phi((normal_dof_i-k)/panel_values,
    residual_energy_i(k)/total_source_energy,
    lambda)
+ lambda * (support_bits_i(k) + value_bits_i(k))/panel_values.
```

The minimum over every `k` is separable across matrices.  Adding the fixed
manifold terms and subtracting
`lambda*(2.5-base_side_bpw)` gives a valid dual lower bound on the best oracle
distortion.  Any evaluated multiplier supplies a valid bound; maximizing over
a grid only strengthens it.  If

```text
dual_distortion * 2^(2*2.5) > 0.8
```

after a conservative numerical allowance, the entire declared support/value
family is killed.  Discrete support choices cannot invalidate this direction.

Decision order:

1. arbitrary coordinate top-k with exact enumerative support and exact
   one-bit-charged values;
2. analytic diagonal strata, fixed triangular tiles, and fixed offset
   segments for block sizes `8,16,32,64`;
3. heavy-tail-matched Gaussian-rank controls;
4. only if an impossible exact-value row survives, a finite BF16/FP16 value
   replay and complete frame/read ledger.

No finite quantizer work is justified before step 1 or 2 survives.

## Heavy-tail and structure controls

Ordinary Gaussian controls confound coordinate heavy tails with locality.  The
new matched control uses Gaussian ranks to permute the exact orthonormal
coefficient magnitudes and regenerates symmetric signs.  It preserves the
complete per-matrix absolute-value multiset and total normal energy while
destroying coordinate/block locality.

- Coordinate top-k capture is therefore exactly identical between Qwen and
  every heavy-tail-matched control.  Its gain is wholly a marginal-tail
  mechanism.
- Tile and offset-segment curves can exceed the control only through spatial
  clustering.  Report raw Qwen, control mean, control Monte-Carlo SE, and the
  optimistic `Qwen-control+3SE` structural excess.
- A diagonal-stratified permutation additionally preserves every offset's
  marginal distribution.  It isolates contiguous clustering from the fixed
  band effect already measured by `identity_band_*`.

The controls are diagnostic.  They are never subtracted from a containing
absolute hard bound.

## Prior-result overlap

- Arbitrary coordinate top-k is the same marginal-tail mechanism as lossy
  tail peeling, but in the nonlinear polar-normal domain.  Gains must be
  rebuilt on the polar residual and never added to the raw-tail result.
- Offset and band supports overlap `identity_band_*`.  A union must apply the
  second mask to the first mask's residual and charge the union support.
- The free-normal envelope (`F≈0.66397`) is the all-information ceiling, not a
  rate-feasible result.
- Orthonormal symmetric coordinates avoid the polar/Jacobian leakage exposed
  by the dual-polar audit.

## MoE read contract

All private support/value data is placed inside the routed expert's single
2.5-bpw frame.  Analytic partitions have no shared cold table.  Logical
compressed read is therefore `1.0x`; page amplification is the same one-frame
rounding class as the parent polar frame.  A surviving correction should be
consumed as `(x N_sparse) Q` inside the expert path so that no dense correction
needs to be read or materialized.

## Claim boundary

This package is a preregistered source-only gate.  It contains no source
binding, tensor path, normal matrix, result, payload, output writer,
authorization builder, or GPU entrypoint.  It does not reopen the prior kill.
It defines the minimum CuPy measurement needed to decide whether sparse normal
corrections deserve a finite experiment.
