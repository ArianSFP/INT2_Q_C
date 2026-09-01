# SPECTRAFLAG-v0: spectral-flag polar manifold oracle

## Hypothesis

The previous polar family used

```text
H_hat = c I + A_k,
```

which represents `k` singular values individually and collapses one contiguous
bulk window to one repeated value.  It does not contain a spectral flag with
multiple repeated-value bands.  SPECTRAFLAG instead models

```text
H_hat = sum_b c_b P_b,
```

where the `P_b` are mutually orthogonal spectral projectors whose contiguous
ordered multiplicities sum to 768.  This can spend the same number of
orientation degrees of freedom on coarse structure across the complete
singular spectrum instead of exact singleton outliers.

For band multiplicities `m_b`, the exact symmetric-factor manifold dimension
is

```text
flag_dof = (768^2 - sum_b m_b^2)/2 + number_of_bands.
```

Adding the row-Stiefel factor gives the complete modeled dimension.  The
unmodeled normal dimension is the complement in all `768*2048` matrix
coordinates, and its exact Frobenius energy is the within-band singular-value
SSE.  The oracle jointly reverse-waterfills modeled and normal components over
all 18 authenticated Qwen matrices.  There is no multiplication of old gains.

## Frozen exploratory curve

For each matrix the candidate curve is the exact union of:

- every prior rank/window row (`k=0..766`), to prove non-regression;
- all 767 contiguous two-band splits;
- equal-width contiguous flags with 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, and
  64 bands; and
- a deterministic greedy split path through 256 bands, selecting the largest
  exact within-band-SSE reduction per added flag degree of freedom.

Coordinate descent chooses one row per matrix using the exact global
reverse-waterfill at 2.5 bpw.  A conservative 32,768-bit global header plus a
full 767-bit boundary bitmap for every matrix is charged even when a row needs
less.  The corresponding external compressed-object read remains close to 1x;
this is an ideal manifold screen, not an emitted codec or fused-kernel traffic
measurement.

The first pass computes only the authenticated Qwen spectra in FP64 CuPy.  If
the best raw `F` is safely above 0.8, the branch stops before Gaussian controls.
If it approaches or crosses the target, the same search must be repeated on
multiple moment-matched Gaussian matrices before any finite implementation,
because polar/flag coordinate metrics can manufacture a Marchenko-Pastur null.

RunPod command:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B flag_oracle.py \
  --plan-dir /workspace/INT2__compression/strata_expert_affine_milestone_v1 \
  --output runpod_result.json
```

This v0 package is an exploratory, architecture-scoped ideal oracle.  A pass
would authorize controls and a corrected finite design, not a success claim.

## RunPod result

The frozen Qwen-only gross screen completed on the supplied RTX 5090 with
CuPy 14.2.0 in 9.842823 seconds.  It authenticated and decomposed all eighteen
source matrices, then evaluated 32,382 coordinate-descent alternatives.  All
eighteen selected rows came from the new greedy flag family, with 106--167
bands, so the result is a genuine test of the new architecture rather than a
fallback to the contained rank/window baseline.

| Quantity | Result |
|---|---:|
| Multi-band flag `F` | `0.9525151991324239` |
| Multi-band flag `s` | `0.03509299062205339 bpw` |
| Rank/window `F` under the same ledger | `0.9526609748482969` |
| Increment over rank/window | `0.00011038869037664073 bpw` |
| Flag/rank `F` ratio | `0.999846980489679` |
| Charged side rate | `0.0016450528745298 bpw` |

The exact decision is `EARLY_KILL_FLAG_FAMILY_RAW_F_FAR_ABOVE_TARGET`.
The new flag geometry removes only 0.0153% of the comparable rank-oracle
distortion and remains far above `F=0.8`.  Gaussian controls, Jacobian
corrections, finite coordinate quantization, and a codec implementation were
therefore stopped.  This kills the frozen contiguous multi-band family; it is
not a converse for every possible nonlinear spectral model.

Result SHA-256:
`1d6a9238f44963dccbe108244594946938d036ede8c9acc2c4031c24ffa79882`.
