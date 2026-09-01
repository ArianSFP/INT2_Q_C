# Qwen Stiefel/Gram manifold oracle

## Decision

**Hard kill.** A full-rank nonlinear Gram manifold does expose more structure
than the earlier low-rank and tensor probes, but it remains far from the
rate-relative target on the pinned 18-matrix Qwen3-30B-A3B panel.

The strongest construction is a source-specific, adaptively ranked polar
normal model

```text
W = H Q,       Q Q^T = I
H ~= c I + A_k,  rank(A_k) = k.
```

It was granted continuous manifold coordinates, ideal asymptotic Gaussian
coding for every orthogonal component, free charts/eigenpair indices, and free
rank labels for the primary decision. Even under those assumptions:

| Quantity | Best oracle | Required |
|---|---:|---:|
| Physical rate | 2.5 bpw | 2.5 bpw |
| Gaussian-normalized `F = D 2^(2R)` | **0.9504908807** | **<= 0.8** |
| Ideal relative MSE | **0.0297028400** | **<= 0.025** |
| Rate-equivalent gain `s = -0.5 log2(F)` | **0.0366276548 bpw** | **>= 0.1609640474 bpw** |
| Percentage below Gaussian | **4.9509%** | **20%** |

This supplies 22.76% of the required rate-equivalent gain and misses by
`0.1243363926 bpw`. Charging the ten-bit rank label for every matrix changes
the result to `F=0.9504992582`, `s=0.0366212970`, and MSE `0.0297031018`.
No GPU quantizer integration is warranted under the predeclared gate.

## Non-duplication boundary

This screen began by reviewing the existing low-rank, rank-one Kronecker,
Van-Loan tensor, TT/MPO, shared bilinear-basis, and direct cross-matrix
predictor evidence. It did not rerun those branches. The prior structural
screen, for example, found at most 0.501% energy in its shared bilinear basis,
0.335% in a rank-one Kronecker predictor, and 6.95% in a deliberately generous
rate-feasible per-matrix low-rank oracle.

The new hypothesis is different: a matrix can be full rank while lying near a
low-codimension nonlinear Stiefel/Gram manifold. For canonical `W` of shape
768 x 2048, the exact polar decomposition is

```text
H = (W W^T)^(1/2)
Q = H^(-1) W
Q Q^T = I.
```

A row-Stiefel `Q` has 1,277,568 degrees of freedom. The missing symmetric Gram
factor has 295,296 degrees, or 18.77% of the original matrix—almost exactly the
codimension that looked capable of closing the target gap.

## Strongest model: partial polar-normal rank

The first tight/diagonal frame tests showed that collapsing all of `H` was too
aggressive. The stronger model retains a rank-`k` source-specific symmetric
deviation:

```text
H_hat = c I + A_k.
```

For each matrix and every feasible `k=0,...,766`, the probe computes the exact
best spectrum. The `k` explicitly represented eigenpairs are arbitrary; all
other singular values share `c`. Because the singular values are sorted, the
minimum-variance unmodeled set is a contiguous window. Exhausting every window
with prefix sums gives the global optimum for each `k`, without a heuristic
SVD truncation.

The exact local dimension is

```text
d(k) = d_Stiefel + 1 + 768 k - k(k-1)/2.
```

The eigenvectors and eigenvalues of `A_k` are included in this dimension; they
are not free side information.

| Rank policy | Rank(s) | Mean model DOF | Residual energy | Structural-only `F` / `s` | Full panel `F` / `s` |
|---|---:|---:|---:|---:|---:|
| Best common rank | 219 | 1,421,890 | 0.0312356922 | 0.9557851782 / 0.0326208498 | 0.9505718317 / 0.0365662220 |
| **Per-matrix adaptive rank** | **179-241; mean 216.667** | **1,420,493.833** | **0.0318011329** | **0.9559268631 / 0.0325139257** | **0.9504908807 / 0.0366276548** |

The panel score lets every matrix's manifold and normal component receive its
own ideal rate allocation. This is more optimistic than the structural-only
two-component score and is the value used for the kill decision. Matrix-energy
allocation by itself has `F=0.9946184368`, `s=0.0038924605`; the polar-normal
manifold provides most, but not enough, of the final `s=0.0366276548`.

At the adaptive optimum, about 90.31% of dimensions describe the manifold and
9.69% describe its normal residual. The latter carries about 3.18% of the
energy. That variance separation is real, but even a perfect reverse-waterfill
converts it to only a 4.95% advantage over isotropic Gaussian coding.

The complete 767-point common-rank curve and all 18 adaptive selections are in
`result.json`.

## Exact tight and diagonal frame results

All Gram matrices were computed as FP64 `W W^T`, followed by a full symmetric
eigendecomposition. No randomized truncation was used.

| Model | Local model DOF | Codimension | Residual energy | Structural-only `F` / `s` | Full panel `F` / `s` |
|---|---:|---:|---:|---:|---:|
| Nearest scaled tight frame `cQ` | 1,277,569 | 295,295 | 0.1494292713 | 0.9945995579 / 0.0039061525 | 0.9867786270 / 0.0096008134 |
| Nearest left-diagonal frame `DQ` | 1,278,336 | 294,528 | 0.1454336110 | 0.9934730597 / 0.0047236244 | 0.9857757286 / 0.0103343175 |
| Fixed-polar right-diagonal frame `QC` | 1,279,616 | 293,248 | 0.1482291245 | 0.9945947174 / 0.0039096632 | 0.9868190022 / 0.0095712992 |

The per-matrix tight-frame residual ranges from 0.1123630476 to 0.1981017234;
the exact `DQ` residual ranges from 0.1088679806 to 0.1892121802.

The `DQ` fit is globally optimized. Eliminating `Q` by exact Procrustes and
writing `x_i=d_i^2` gives

```text
||W||_F^2 + sum(x)
  - 2 tr sqrt(sqrt(diag(x)) (W W^T) sqrt(diag(x))).
```

The trace term is matrix fidelity and is concave in `diag(x)`, so the objective
is convex. All 18 positive fixed points reached a relative KKT residual below
`1e-10`.

These results show why projection error alone is misleading. The best `DQ`
projection removes 14.54% of energy, but the model still occupies 81.27% of
dimensions and its normal occupies 18.73%. Their per-degree variances are too
similar, yielding only a 1.42% panel-level distortion advantage.

## Whole-expert held-out templates

Each fold excludes one complete layer/expert triplet: Gate, Up, and Down are
removed together. Role templates use only the five same-role training
matrices. Global templates use all fifteen remaining matrices. One target
scale is included in each local model dimension.

| Held-out model | Residual energy | Model DOF | Structural-only `F` / `s` | Full panel `F` / `s` |
|---|---:|---:|---:|---:|
| Role spectral template | 0.0025627916 | 1,572,097 | 0.9987326330 / 0.0009147919 | 0.9931516282 / 0.0049570494 |
| Global spectral template | 0.0031812627 | 1,572,097 | 0.9982188870 / 0.0012859470 | 0.9926387308 / 0.0053296740 |
| Role full-`H` template | 0.1763800993 | 1,277,569 | 0.9995629122 / 0.0003153611 | 0.9925690456 / 0.0053803158 |
| Global full-`H` template | 0.1596132008 | 1,277,569 | 0.9971829874 / 0.0020349126 | 0.9898525605 / 0.0073572222 |
| Role mean eigenbasis | 0.1468942504 | 1,278,336 | 0.9939509840 / 0.0043766934 | 0.9865702478 / 0.0097531572 |
| Global mean eigenbasis | 0.1470224427 | 1,278,336 | 0.9939919431 / 0.0043469685 | 0.9865874885 / 0.0097405515 |
| Best training role basis, target-aware | 0.1474712390 | 1,278,336 | 0.9941340951 / 0.0042438151 | 0.9866281690 / 0.0097108084 |
| Best training global basis, target-aware | 0.1474633650 | 1,278,336 | 0.9941316177 / 0.0042456127 | 0.9866257129 / 0.0097126041 |

The tiny spectral-template residual is not useful compression: fixing 767
singular-value ratios removes only 767 of 1,572,864 dimensions (0.0488%). The
source-specific singular frames carry nearly the entire matrix. Conversely,
a fixed eigenbasis removes the desired number of degrees, but held-out `H` is
not substantially more diagonal in that basis. Even decoder-illegal,
target-aware selection among training bases does not help.

## Physical shared-table ledger

The survival decision gives shared tables away for free. Actual FP16 storage
only worsens the result.

| Shared object | Panel-local bpw | Full 48 x 128 MoE amortized bpw |
|---|---:|---:|
| One spectrum | 0.000434028 | 0.000000424 |
| Three role spectra | 0.001302083 | 0.000001272 |
| One symmetric `H` template | 0.166883681 | 0.000162972 |
| Three role `H` templates | 0.500651042 | 0.000488917 |
| One eigenbasis, minimal-angle lower bound | 0.166449653 | 0.000162548 |
| Three role eigenbases, minimal-angle lower bound | 0.499348958 | 0.000487645 |
| One dense eigenbasis | 0.333333333 | 0.000325521 |
| Three dense role eigenbases | 1.000000000 | 0.000976563 |

The adaptive polar-normal model needs no shared dense table. Its ten-bit rank
label costs only `0.00000635783 bpw`, already included in the charged result.
Chart, eigenpair-order, finite-precision, and manifold-code costs remain free,
so the physical ledger is still deliberately favourable.

## Rate-distortion scoring

For each orthogonal component, `d_i` is its exact fraction of all panel degrees
and `e_i` its measured energy fraction. Its variance is `v_i=e_i/d_i`. After
deducting serialized shared bits, an ideal Gaussian reverse-waterfill chooses
`lambda` and reports

```text
D = sum_i d_i min(v_i, lambda)
F = D / 2^(-2R)
s = -0.5 log2(F).
```

The oracle is more permissive than a real PTQ codec: continuous coordinates,
infinite-block Gaussian rate-distortion, exact polar geometry, no curvature or
chart loss, no finite precision, and free model-selection indices except in
the separately charged rank-label score. Failure under these assumptions is
the basis for the hard kill.

## Reproduction and audit

The run was CPU-only on the provided RunPod while the production GPU encoder
continued independently:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/INT2_Q_C/research/stiefel_gram_oracle/stiefel_gram_oracle.py \
  --root /workspace/INT2__compression \
  --output /workspace/INT2__compression/INT2_Q_C/research/stiefel_gram_oracle/result.json

/workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/INT2_Q_C/research/stiefel_gram_oracle/verify_stiefel_gram_result.py \
  --root /workspace/INT2__compression \
  --result /workspace/INT2__compression/INT2_Q_C/research/stiefel_gram_oracle/result.json \
  --output /workspace/INT2__compression/INT2_Q_C/research/stiefel_gram_oracle/independent_audit.json
```

The independent verifier does not import the experiment. It rehashes all 18
BF16 sources, reconstructs every tight-frame error and every 767-point
structured-Gram curve from the serialized singular spectra, checks all `DQ`
traces, reselects common and adaptive ranks, recomputes every waterfill, and
independently selects the final decision. It passes.

`ARTIFACT_HASHES.json` freezes the scripts, result, audit, and this report. The
result records the pinned Qwen revision and all declared/observed source
hashes. This is a source-locked negative architecture result, not a claim that
the overall 20%-below-Gaussian goal is impossible. It closes the specific
route of whole-matrix polar/Gram manifold coding.
