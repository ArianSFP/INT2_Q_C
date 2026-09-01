# NanoQuant-style binary factor: negative result and SVD supersession

## Bottom line

The charged, discrete NanoQuant-style factorization is a clear negative on
the pinned Qwen panel.  It has excellent expert locality (exactly 1.0x cold
reads), but its MSE is several times the Gaussian rate-distortion reference
and its Qwen-specific advantage over the identically optimized Gaussian
control is only about 2%.

| physical R | tile rank | source D | matched Gaussian D | source / Gaussian | structural `s` | codec `F=D*2^(2R)` | cold read |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.1510417 | 59 | 0.14147834 | 0.14431730 | 0.98032832 | 0.01433155 | 2.79091159 | 1.0x |
| 2.5000000 | 71 | 0.09973008 | 0.10163337 | 0.98127297 | 0.01363678 | 3.19136257 | 1.0x |

The target is `F <= 0.8`, or equivalently `s >= 0.160964047443681`.
The 2.5-bpw result reaches only 8.47% of the required structural advantage
and misses the absolute codec target by almost 4x.

This is a sampled tile experiment rather than a complete encoded bitstream:
four deterministic `48x128` tiles from each of 18 matrices were evaluated at
each rate, giving 144 paired source/control observations.  That limitation
does not turn the result into a lower bound, but the gap is large enough to
hard-kill this branch.

## Architecture and exact storage ledger

The tested form follows [NanoQuant, arXiv:2602.06694v3](https://arxiv.org/abs/2602.06694)
and its [official implementation](https://github.com/SamsungLabs/NanoQuant):

\[
\widehat W=\operatorname{diag}(a)\,U_{\{\pm1\}}
V_{\{\pm1\}}^T\operatorname{diag}(b).
\]

For an `n x m` tile and rank `r`, the payload is `r(n+m)` sign bits plus
`16(n+m)` bits for FP16 row/column scales.  No rank scales or shared tables
are hidden in the result.

For the `48x128` tiles:

| requested R | rank | sign bits | FP16 scale bits | useful bits | byte-aligned physical bits | padding |
|---:|---:|---:|---:|---:|---:|---:|
| 2.15 | 59 | 10,384 | 2,816 | 13,200 | 13,216 | 16 |
| 2.5 | 71 | 12,496 | 2,816 | 15,312 | 15,360 | 48 |

The ranks exceed the smaller tile dimension.  This is intentional: a sum of
binary outer products can be overcomplete even though a continuous matrix
rank cannot exceed 48.  The official full-matrix implementation caps rank at
768, which would use only 1.40365 bpw for a `768x2048` matrix.  The spectral
ledger also records radical overcomplete full-matrix ranks 1,184 and 1,376
that fill the 2.15/2.5 budgets; the actual discrete test uses tiles so that
this overcomplete regime is computationally testable on CPU.

Each expert owns its three factor streams, scales, padding, and framing.
There are no cross-expert tables or coding dependencies, so reading one
expert reads each stored byte exactly once: cold amplification is 1.0x.

## Optimizer and matched control

[`nanoquant_binary_factor_discrete_pilot.py`](nanoquant_binary_factor_discrete_pilot.py)
uses a NumPy port of the paper's LB-ADMM/SVID construction:

- 400 outer updates and five SVID inner iterations;
- deterministic paired initialization for Qwen and control;
- exact binary signs in both factors;
- 20 alternating row/column scale refits;
- FP16 rounding of the scales before scoring;
- raw Qwen matrices, four deterministic tiles per matrix;
- a Gaussian tile with the exact source mean and centered energy for every
  source tile.

The same code, initialization, iteration count, scale fit, and FP16 rounding
are applied to both sides.  This makes `D_Qwen/D_Gaussian` a matched
structural comparison rather than a comparison to an idealized Gaussian
formula.

[`verify_nanoquant_binary_result.py`](verify_nanoquant_binary_result.py)
independently verifies the canonical result lock, algorithm hash, all 144
energy/SSE identities, both pooled aggregates, bit/padding ledgers, all
`F`/`s` identities, the plan/header seals, and all 18 live BF16 sources.
The frozen verification receipt is
[`nanoquant_discrete_verification_receipt.json`](nanoquant_discrete_verification_receipt.json).

## Why the rank-764 SVD `0.75797` is not a survivor

The preliminary full-matrix screen found

\[
0.75797074295=
\frac{0.00062639663}{0.00082641267},
\]

the ratio of source and matched-Gaussian energies discarded by their own
exact rank-764 SVDs.  The JSON called this an `F_ratio_identity`; that name is
misleading for codec selection.  It is a tail ratio, not either codec's
absolute `F=D*2^(2R)`.  The frozen screen is preserved for provenance, but
its promotion decision is superseded by
[`svd_survivor_supersession.json`](svd_survivor_supersession.json).

For one `768x2048` matrix and `k=764`, the exact rank-k manifold has

\[
p=k(n+m-k)=1,567,728
\]

continuous degrees of freedom.  Its normal complement has only

\[
(n-k)(m-k)=5,136
\]

dimensions.  The screen therefore grants an exact reconstruction for
99.6735% of ambient dimensions and evaluates only the 0.3265% discarded
normal space.

The omitted source-specific state is:

- `U` Stiefel DOF: 294,522;
- `V` Stiefel DOF: 1,272,442;
- orientation DOF: 1,566,964;
- retained singular values: 764.

Storing dense FP16 `U`, `V`, and singular values would cost 21.8932 bpw,
8.757x the entire 2.5-bpw budget.  A minimal chart still has 1,567,728
continuous coordinates, leaving only 2.508 bits per manifold DOF if the
whole 2.5-bpw budget were assigned to it.

The omitted error is not captured by a bare parameter count.  For
`W=U Sigma V^T`,

\[
dW=U\,d\Sigma\,V^T+dU\,\Sigma\,V^T+U\,\Sigma\,dV^T.
\]

Basis errors are weighted by the singular values; rotations of singular
directions couple through the quotient metric.  The SVD volume element also
contains powers of singular values and factors
`|sigma_i^2-sigma_j^2|`.  The tail-only score charges neither the coordinate
precision nor this metric/Jacobian distortion.  Leave-one-expert-out
selection does not repair the issue: it cross-fits the choice of rank and
representation, but still derives the held-out matrix's exact `U`, `V`, and
singular values for free.

The relevant optimistic comparator is the independently tested Stiefel/Gram
oracle in [`../../stiefel_gram_oracle/`](../../stiefel_gram_oracle/).  It
charges ideal rate to the model and normal components using Gaussian
reverse-waterfilling, while still granting continuous coordinates, free
shared tables, and no finite-block or curvature penalty.  Even that favorable
oracle obtains only:

- `F = 0.9504908806946374`;
- `s = 0.0366276548062078` bpw;
- shortfall `0.1243363926374734` bpw from the required structural gain.

That is the precise reason the valid rate-accounted answer is near 0.95 while
the uncharged SVD-tail ratio is near 0.758.  No Householder/Givens/sparse
tangent implementation was warranted after the survivor collapsed; the
actual charged binary representation above is also decisively negative.

## Reproduction

Run from the repository's RunPod workspace.  The pinned plan and sources are
external checkpoint inputs and are not duplicated here.

```bash
OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  /workspace/int2-cupy-venv/bin/python \
  research/procedural_subspace_oracle/nanoquant_binary_factor/nanoquant_binary_factor_discrete_pilot.py \
  --plan /workspace/INT2__compression/strata_expert_affine_milestone_v1/plan.lock.json \
  --output research/procedural_subspace_oracle/nanoquant_binary_factor/nanoquant_discrete_confirmation_raw_momentfixed.json \
  --representation raw --tile-rows 48 --tile-cols 128 \
  --tiles-per-matrix 4 --rates 2.15,2.5 \
  --outer-iters 400 --inner-iters 5 --scale-iters 20 \
  --reg 0.1 --restarts 1 --rank-scale-bits 0
```

Verify the discrete result and rehash every source:

```bash
/workspace/int2-cupy-venv/bin/python \
  research/procedural_subspace_oracle/nanoquant_binary_factor/verify_nanoquant_binary_result.py \
  --result research/procedural_subspace_oracle/nanoquant_binary_factor/nanoquant_discrete_confirmation_raw_momentfixed.json \
  --algorithm research/procedural_subspace_oracle/nanoquant_binary_factor/nanoquant_binary_factor_discrete_pilot.py \
  --plan /workspace/INT2__compression/strata_expert_affine_milestone_v1/plan.lock.json \
  --rehash-sources
```

Verify the SVD supersession against the sibling Stiefel/Gram result:

```bash
/workspace/int2-cupy-venv/bin/python \
  research/procedural_subspace_oracle/nanoquant_binary_factor/verify_svd_survivor_supersession.py \
  --spectral-result research/procedural_subspace_oracle/nanoquant_binary_factor/nanoquant_spectral_confirmation.json \
  --stiefel-result research/stiefel_gram_oracle/result.json \
  --binary-result research/procedural_subspace_oracle/nanoquant_binary_factor/nanoquant_discrete_confirmation_raw_momentfixed.json
```

Important seals:

- discrete algorithm SHA-256:
  `b6f85a6dbf6b76b18bdf6eee61baf20c5d9fbc198a4d1dac00d3985ebc640339`
- discrete result SHA-256:
  `1a01dae0ba38b76e13c2441eb8ee9bce93b07bd8909ac265bfe9d93066a69460`
- discrete result lock:
  `0d6784ea229b2def6b9ed254a585ba9cd3f25c3f09a52bd3ba3cfd27740d864c`
- discrete verification receipt lock:
  `b5dbc549609a8c87f07898a047f0de3ad511759f7868945fabf9df59800d8110`
- SVD supersession audit lock:
  `76bd5848d41caf28a6ced2507da8c35f9f2be3700cf66d44830c2e4b16a2babd`

## Claim boundary

The discrete result rejects the tested NanoQuant-style binary outer-product
factorization and optimizer at these tile sizes and rates.  The SVD audit
invalidates promotion of the free-tail screen.  Neither is a universal
impossibility proof for every learned binary factor, nonlinear decoder, or
procedural basis.
